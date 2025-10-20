#!/usr/bin/env python3
"""
Active prompting: detect over-prompting, flag redundant prompts, refine prompt set,
and (optionally) evaluate faithfulness vs GT.

- Decisions use GT-free proxies (PredIoU, Anchored weighted InfoNCE MI, Stability).
- If GT is given, only used for reporting IoU/Dice (faithfulness).

Outputs:
  - curves for PredIoU, Anchored MI, Stability, Uncertainty
  - metrics_per_step.csv (+ delta & EMA columns)
  - redundancy_report.csv (per-prompt)
  - overprompting_knee.json
  - refined_prompts.json (time-ordered, reindexed)
  - eval_refinement.csv (IoU/Dice original vs early-stop vs refined if GT provided)

Requires: segment-anything (facebookresearch)
"""

import argparse, json, math, csv
from pathlib import Path
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F

from segment_anything import sam_model_registry, SamPredictor


# ---------------- I/O helpers ----------------
def ensure_dir(p): Path(p).mkdir(parents=True, exist_ok=True)

def load_image_rgb(path): return np.asarray(Image.open(path).convert("RGB"))

def load_mask_any(path, gt_format: str):
    if path is None: return None
    if path.endswith(".npy"):
        arr = np.load(path)
        assert arr.ndim == 2
        if gt_format == "npy_zero_is_fg":
            fg = (arr == 0)
        else:
            fg = (arr == 1)
    else:
        g = np.asarray(Image.open(path).convert("L"))
        if gt_format == "png_nonzero_is_fg":
            fg = (g != 0)
        else:
            fg = (g == 0)
    return fg.astype(np.uint8)

def save_csv(rows, header, path):
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        if header: w.writerow(header)
        w.writerows(rows)

def binary_iou_dice(pred01, gt01):
    inter = np.logical_and(pred01, gt01).sum()
    union = np.logical_or (pred01, gt01).sum()
    iou  = inter / (union + 1e-8)
    dice = (2*inter) / (pred01.sum() + gt01.sum() + 1e-8)
    return float(iou), float(dice)

def ema(arr, alpha=0.4):
    out = []
    m = None
    for v in arr:
        m = v if m is None else alpha*v + (1-alpha)*m
        out.append(m)
    return np.array(out, dtype=np.float32)


# ------------- Geometry / resizing -------------
def area_pool_to_tokens(mask_hw_np, Ht, Wt):
    t = torch.from_numpy(mask_hw_np).float()[None, None]
    ds = F.interpolate(t, size=(Ht, Wt), mode="area")[0, 0]
    return ds.view(-1, 1).cpu().numpy()

def up_nearest(hw, H, W):
    return F.interpolate(torch.from_numpy(hw)[None,None].float(), size=(H,W), mode="nearest")[0,0].numpy()

def hflip_coords_xy(coords_xy, W):
    out = coords_xy.copy()
    out[:,0] = (W - 1) - out[:,0]
    return out

def hflip_mask(prob):
    return np.flip(prob, axis=1).copy()

def make_disk_mask(H, W, x, y, radius):
    """
    Returns (ys_slice, xs_slice, disk_bool) for a circular neighborhood centered at (x,y).
    If the ROI is empty, returns (None, None, None).
    """
    x = int(round(x)); y = int(round(y))
    if radius <= 0:
        return None, None, None
    x0 = max(0, x - radius); x1 = min(W, x + radius + 1)
    y0 = max(0, y - radius); y1 = min(H, y + radius + 1)
    if x0 >= x1 or y0 >= y1:
        return None, None, None
    xs = np.arange(x0, x1)
    ys = np.arange(y0, y1)
    xxg, yyg = np.meshgrid(xs, ys)          # shapes (h, w)
    disk = (xxg - x) ** 2 + (yyg - y) ** 2 <= (radius * radius)
    if not disk.any():
        return None, None, None
    return slice(y0, y1), slice(x0, x1), disk


# ------------- Anchors (fixed yardstick) -------------
def boundary_ring_weights_from_binary(mask_tok_hw, radius=2):
    t = torch.from_numpy(mask_tok_hw.astype(np.float32))[None, None]
    pad = radius
    dil = F.max_pool2d(t, kernel_size=2*radius+1, stride=1, padding=pad)
    ero = -F.max_pool2d(-t, kernel_size=2*radius+1, stride=1, padding=pad)
    ring = (dil - ero).clamp(min=0.0)[0,0]  # [Ht,Wt]
    w = ring.view(-1)
    w = w / (w.sum() + 1e-12)
    return w

def uncertainty_weights_from_prob_tok(prob_tok_hw, gamma=1.0):
    p = torch.from_numpy(prob_tok_hw.astype(np.float32))
    w = (p * (1.0 - p)).pow(gamma)
    w = w / (w.sum() + 1e-12)
    return w.view(-1)


# ------------- Two-Way hooks -------------
def attach_two_way_hooks(two_way):
    cache = {}
    def _get_qkv(args, kwargs):
        if kwargs and all(k in kwargs for k in ("q","k","v")):
            return kwargs["q"], kwargs["k"], kwargs["v"]
        if len(args) >= 3: return args[0], args[1], args[2]
        raise RuntimeError("Expected (q,k,v) in final attn hooks")

    def pre_hook_final(module, args, kwargs):
        q,k,v = _get_qkv(args, kwargs); cache["_final_inputs"]=(q.detach(),k.detach(),v.detach())

    def fwd_hook_final(module, args, kwargs, output):
        q,k,v = cache.get("_final_inputs", _get_qkv(args, kwargs))
        ql = module.q_proj(q); kl = module.k_proj(k)
        B, Nq, Cq = ql.shape; H = module.num_heads; Ch = Cq // H
        qh = ql.view(B, Nq, H, Ch).permute(0,2,1,3)  # [B,H,Nq,Ch]
        kh = kl.view(B, -1, H, Ch).permute(0,2,1,3)  # [B,H,Ni,Ch]
        attn = torch.matmul(qh, kh.transpose(-1,-2)) / (Ch**0.5)
        cache["final_t2i_attn"] = torch.softmax(attn, dim=-1).detach()

    def fwd_hook_block_out(module, args, out):
        q,k = out
        cache["last_queries"] = q.detach()
        cache["last_keys"]    = k.detach()

    hs=[]
    try:
        hs.append(two_way.final_attn_token_to_image.register_forward_pre_hook(pre_hook_final, with_kwargs=True))
        hs.append(two_way.final_attn_token_to_image.register_forward_hook    (fwd_hook_final, with_kwargs=True))
    except TypeError:
        hs.append(two_way.final_attn_token_to_image.register_forward_pre_hook(lambda m,a: pre_hook_final(m,a,{})))
        hs.append(two_way.final_attn_token_to_image.register_forward_hook    (lambda m,a,o: fwd_hook_final(m,a,{},o)))
    hs.append(two_way.register_forward_hook(fwd_hook_block_out))
    return hs, cache

def remove_hooks(handles):
    for h in handles: h.remove()


# ------------- Weighted InfoNCE -------------
class ProjectionHead(nn.Module):
    def __init__(self, in_dim, proj_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, proj_dim), nn.ReLU(inplace=True),
            nn.Linear(proj_dim, proj_dim),
        )
    def forward(self, x): return self.net(x)

def infoNCE_weighted_lower_bound(Z, Y, row_weights, steps=200, batch=2048, lr=1e-3, proj=128, temp=0.1, device="cuda"):
    N = Z.shape[0]
    fZ = ProjectionHead(Z.shape[1], proj_dim=proj).to(device)
    fY = ProjectionHead(Y.shape[1], proj_dim=proj).to(device)
    opt = torch.optim.Adam(list(fZ.parameters()) + list(fY.parameters()), lr=lr)

    Zn = (Z - Z.mean(0, keepdim=True)) / (Z.std(0, keepdim=True) + 1e-6)
    Yn = (Y - Y.mean(0, keepdim=True)) / (Y.std(0, keepdim=True) + 1e-6)

    rw = row_weights.clamp(min=0)
    rw = rw / (rw.sum() + 1e-12)

    for _ in range(steps):
        b = min(batch, N)
        idx = torch.multinomial(rw, num_samples=b, replacement=True)
        z_b = fZ(Zn[idx]); y_b = fY(Yn[idx])
        logits  = (z_b @ y_b.t()) / temp
        targets = torch.arange(b, device=device)
        loss_per = F.cross_entropy(logits, targets, reduction='none')
        rwb = rw[idx]; rwb = rwb / (rwb.sum() + 1e-12)
        loss = (rwb * loss_per).sum()
        opt.zero_grad(set_to_none=True); loss.backward(); opt.step()

    with torch.no_grad():
        b = min(4096, N)
        idx = torch.multinomial(rw, num_samples=b, replacement=True)
        z_b = fZ(Zn[idx]); y_b = fY(Yn[idx])
        logits  = (z_b @ y_b.t()) / temp
        targets = torch.arange(b, device=device)
        ce = F.cross_entropy(logits, targets, reduction='none')
        rwb = rw[idx]; rwb = rwb / (rwb.sum() + 1e-12)
        lb = float(-(rwb * ce).sum().item())
    return lb


# ------------- SAM utilities -------------
def down_up_to_original(lowres_logits, predictor, device):
    low = torch.as_tensor(lowres_logits, device=device)
    if low.ndim == 2: low = low[None, None]
    elif low.ndim == 3: low = low[:, None, :, :]
    elif low.ndim == 4 and low.shape[1] != 1: low = low[:, :1, ...]
    model = getattr(predictor, "model", None)
    if model is not None and hasattr(model, "postprocess_masks"):
        prob = torch.sigmoid(model.postprocess_masks(low, input_size=predictor.input_size, original_size=predictor.original_size)[0,0])
    else:
        H,W = predictor.original_size
        prob = torch.sigmoid(F.interpolate(low, size=(H,W), mode="bilinear", align_corners=False)[0,0])
    return prob.detach().cpu().numpy()


# ------------- Evaluation for a given prompt set -------------
def evaluate_prompt_set(predictor, sam, img, pts_xy, lbl, w_ref, args, hooks_cache_out=None):
    """
    Run SAM on (img, prompts) and compute:
      - prob (H,W), PredIoU, Stability (hflip), fused Z, pooled Y, final prompt ownership labels on token grid
      - K_last (for Z), final_t2i_attn
      - Anchored InfoNCE MI
    Optionally returns hooks cache for introspection.
    """
    device = next(sam.parameters()).device
    H, W = img.shape[:2]

    predictor.set_image(img)

    two_way = sam.mask_decoder.transformer
    handles, cache = attach_two_way_hooks(two_way)
    with torch.inference_mode():
        masks, scores, logits = predictor.predict(
            point_coords=pts_xy if len(pts_xy) else np.zeros((0,2), np.float32),
            point_labels=lbl      if len(pts_xy) else np.zeros((0,), np.int32),
            multimask_output=False,
        )
    remove_hooks(handles)

    prob = down_up_to_original(logits, predictor, device)
    pred_iou = float(scores[0]) if len(scores)>0 else float('nan')

    # Stability via hflip TTA
    img_flip = np.ascontiguousarray(np.fliplr(img))
    predictor.set_image(img_flip)
    pts_flip = hflip_coords_xy(pts_xy.copy(), W) if len(pts_xy) else np.zeros((0,2), np.float32)
    with torch.inference_mode():
        mf, sf, lf = predictor.predict(point_coords=pts_flip, point_labels=lbl if len(pts_xy) else np.zeros((0,), np.int32), multimask_output=False)
    prob_f = down_up_to_original(lf, predictor, device)
    stability = binary_iou_dice((prob>=0.5).astype(np.uint8), (hflip_mask(prob_f)>=0.5).astype(np.uint8))[0]
    predictor.set_image(img)  # restore

    # Tokens, Z, Y, ownership
    feats = predictor.features
    _, Cfeat, Hf, Wf = feats.shape
    Ni = Hf*Wf

    last_k = cache.get("last_keys"); Afin = cache.get("final_t2i_attn")
    if last_k is None or Afin is None:
        raise RuntimeError("Failed to capture internals (keys/attn).")
    A = Afin.mean(dim=1)  # [B,Nq,Ni]
    num_mask_tokens = getattr(sam.mask_decoder, "num_mask_tokens", 3)
    prompt_offset = 1 + num_mask_tokens
    Np = len(lbl)
    if Np>0:
        pr_idx = torch.arange(prompt_offset, prompt_offset+Np, device=A.device)
        A_pr = A[:, pr_idx, :][0]      # [Np, Ni]
        labels = torch.argmax(A_pr, dim=0).detach().cpu().numpy().astype(np.int32)  # [Ni]
    else:
        A_pr = torch.zeros((0, Ni), device=A.device)
        labels = -np.ones((Ni,), dtype=np.int32)

    K_last = last_k[0]  # [Ni, Ck]
    Z = torch.cat([K_last, A_pr.transpose(0,1)], dim=1).to(device)  # [Ni, Ck+Np]
    Ytok = torch.from_numpy(area_pool_to_tokens(prob, Hf, Wf)).float().to(device)  # [Ni,1]

    # Anchored MI
    mi_val = infoNCE_weighted_lower_bound(
        Z, Ytok, row_weights=w_ref.to(device),
        steps=args.infonce_steps, batch=args.infonce_batch, lr=args.infonce_lr,
        proj=args.infonce_proj, temp=args.infonce_temp, device=device
    )

    # Uncertainty metrics
    uncert_global = float((prob * (1-prob)).mean())
    out = {
        "prob": prob, "pred_iou": pred_iou, "stability": float(stability),
        "uncert_global": uncert_global, "uncert_boundary": uncert_global,  # boundary-only requires an anchor map; omitted here
        "Z": Z.detach(), "Ytok": Ytok.detach(), "labels_tok": labels, "token_hw": (Hf,Wf),
        "A_final": Afin.detach(), "last_keys": last_k.detach(), "last_queries": cache.get("last_queries", None)
    }
    if hooks_cache_out is not None:
        hooks_cache_out.update(out)
    return mi_val, out


# ---------------- Main pipeline ----------------
def run(args):
    device = "cuda" if torch.cuda.is_available() and not args.cpu else "cpu"
    out_dir = Path(args.out_dir); ensure_dir(out_dir)

    # ---- Load inputs ----
    img = load_image_rgb(args.image)
    H, W = img.shape[:2]
    with open(args.prompts, "r") as f:
        prom = json.load(f)
    prom_list = sorted(prom["prompts"], key=lambda d: d["t"])
    pts_all = np.array([[p["x"], p["y"]] for p in prom_list], dtype=np.float32)
    lbl_all = np.array([int(p["label"]) for p in prom_list], dtype=np.int32)

    # Clip OOB prompts to image bounds (safer than skipping)
    pts_all[:, 0] = np.clip(pts_all[:, 0], 0, W - 1)
    pts_all[:, 1] = np.clip(pts_all[:, 1], 0, H - 1)

    gt_mask = load_mask_any(args.gt_mask, args.gt_format) if args.gt_mask else None
    if gt_mask is not None and gt_mask.shape != (H, W):
        raise ValueError("GT mask must match image size")

    # ---- Build SAM ----
    sam = sam_model_registry[args.model_type](checkpoint=args.checkpoint)
    sam.to(device).eval()
    predictor = SamPredictor(sam)
    predictor.set_image(img)
    if predictor.features is None:
        raise RuntimeError("Predictor has no encoder features; set_image failed?")

    _, Cfeat, Hf, Wf = predictor.features.shape
    Ni = Hf * Wf

    # ---- Anchor weights (fixed) ----
    if gt_mask is not None:
        gt_tok = area_pool_to_tokens(gt_mask.astype(np.float32), Hf, Wf).reshape(Hf, Wf)
        w_ref = boundary_ring_weights_from_binary((gt_tok >= 0.5).astype(np.float32), radius=args.ring_radius)
        anchor_note = "GT boundary ring"
    else:
        # Step-0 prediction as anchor
        hooks_tmp = {}
        _mi0, out0 = evaluate_prompt_set(predictor, sam, img,
                                         np.zeros((0, 2), np.float32),
                                         np.zeros((0,), np.int32),
                                         w_ref=torch.ones((Ni,), dtype=torch.float32),
                                         args=args, hooks_cache_out=hooks_tmp)
        prob0 = out0["prob"]
        prob0_tok = area_pool_to_tokens(prob0, Hf, Wf).reshape(Hf, Wf)
        if args.anchor_mode == "boundary":
            w_ref = boundary_ring_weights_from_binary((prob0_tok >= 0.5).astype(np.float32), radius=args.ring_radius)
            anchor_note = "Step-0 boundary ring"
        else:
            w_ref = uncertainty_weights_from_prob_tok(prob0_tok, gamma=args.unc_gamma)
            anchor_note = "Step-0 uncertainty"
    w_ref = w_ref.to(device)

    # ---- Step-by-step metrics ----
    steps = []
    pred_iou, anch_mi, stab = [], [], []
    unc_g, unc_b = [], []
    iou_gt, dice_gt = [], []

    probs_per_step = []
    ZY_per_step = []
    owner_per_step = []
    prompt_records = []

    token_hw = (Hf, Wf)

    for k in range(1, len(pts_all) + 1):
        steps.append(k)
        pts_k = pts_all[:k]; lbl_k = lbl_all[:k]

        mi_k, out = evaluate_prompt_set(predictor, sam, img, pts_k, lbl_k, w_ref, args)
        anch_mi.append(mi_k)
        pred_iou.append(out["pred_iou"])
        stab.append(out["stability"])
        # Uncertainty (global); boundary variant omitted (set equal)
        uncert = out["uncert_global"]
        unc_g.append(uncert); unc_b.append(uncert)

        probs_per_step.append(out["prob"])
        ZY_per_step.append((out["Z"], out["Ytok"]))
        owner_per_step.append(out["labels_tok"])

        if gt_mask is not None:
            iou, dice = binary_iou_dice((out["prob"] >= 0.5).astype(np.uint8), gt_mask)
            iou_gt.append(iou); dice_gt.append(dice)

        # Per-prompt local stats at insertion
        x, y, lab = int(pts_k[-1, 0]), int(pts_k[-1, 1]), int(lbl_k[-1])
        rad_px = args.local_radius

        ys, xs, disk = make_disk_mask(H, W, x, y, rad_px)
        if ys is None:
            dU_loc = 0.0
            dMI_loc = 0.0
        else:
            # Local uncertainty delta
            if k > 1:
                prev_prob = probs_per_step[-2][ys, xs]
            else:
                prev_prob = probs_per_step[-1][ys, xs]
            now_prob = probs_per_step[-1][ys, xs]
            u_prev = (prev_prob[disk] * (1 - prev_prob[disk])).mean()
            u_now  = (now_prob[disk]  * (1 - now_prob[disk])).mean()
            dU_loc = float(u_prev - u_now)   # want positive

            # Local MI delta in token neighborhood
            ti = int(round(y * (token_hw[0] / H)))
            tj = int(round(x * (token_hw[1] / W)))
            Rtok = max(1, args.token_radius)
            i0, i1 = max(0, ti - Rtok), min(token_hw[0], ti + Rtok + 1)
            j0, j1 = max(0, tj - Rtok), min(token_hw[1], tj + Rtok + 1)
            token_mask = np.zeros(token_hw, dtype=np.uint8)
            token_mask[i0:i1, j0:j1] = 1
            w_loc = torch.from_numpy(token_mask.reshape(-1).astype(np.float32)).to(device)
            if w_loc.sum() == 0:
                w_loc = torch.ones((token_hw[0] * token_hw[1],), dtype=torch.float32, device=device)

            Z_prev, Y_prev = ZY_per_step[-2] if k > 1 else ZY_per_step[-1]
            Z_now,  Y_now  = ZY_per_step[-1]
            mi_prev = infoNCE_weighted_lower_bound(Z_prev, Y_prev, w_loc,
                                                   steps=args.infonce_local_steps,
                                                   batch=min(args.infonce_batch, 1024),
                                                   lr=args.infonce_lr,
                                                   proj=args.infonce_proj,
                                                   temp=args.infonce_temp,
                                                   device=device)
            mi_now  = infoNCE_weighted_lower_bound(Z_now,  Y_now,  w_loc,
                                                   steps=args.infonce_local_steps,
                                                   batch=min(args.infonce_batch, 1024),
                                                   lr=args.infonce_lr,
                                                   proj=args.infonce_proj,
                                                   temp=args.infonce_temp,
                                                   device=device)
            dMI_loc = float(mi_now - mi_prev)  # want positive

        # Ownership overlap with union of previous prompts (at token grid)
        if k > 1 and owner_per_step[-1].max() >= 0 and owner_per_step[-2].max() >= 0:
            curr_id = k - 1
            lab_tok  = owner_per_step[-1].copy()
            lab_prev = owner_per_step[-2].copy()
            curr_mask  = (lab_tok == curr_id).astype(np.uint8)
            prev_union = (lab_prev >= 0).astype(np.uint8)
            inter = (curr_mask & prev_union).sum()
            unio  = (curr_mask | prev_union).sum()
            overlap_iou = float(inter / (unio + 1e-8)) if unio > 0 else 0.0
        else:
            overlap_iou = 0.0

        prompt_records.append({
            "step": k, "x": x, "y": y, "label": int(lab),
            "dU_local": dU_loc, "dMI_local": dMI_loc, "overlap_iou": overlap_iou
        })

        # Step log
        msg = f"[step {k}] PredIoU={pred_iou[-1]:.3f} AnchMI={anch_mi[-1]:.3f} stab={stab[-1]:.3f} U={unc_g[-1]:.3f}"
        if gt_mask is not None:
            msg += f" | IoU={iou_gt[-1]:.3f} Dice={dice_gt[-1]:.3f}"
        print(msg)

    # ---- Deltas & knee detection ----
    def deltas(v):
        v = np.asarray(v, dtype=np.float32)
        d = np.diff(v, prepend=v[:1])
        d[0] = np.nan
        return v, d

    pred_iou_arr, d_pred = deltas(pred_iou)
    anch_mi_arr,  d_mi   = deltas(anch_mi)
    stab_arr,     d_stab = deltas(stab)

    d_pred_s = ema(np.nan_to_num(d_pred, nan=0.0), alpha=args.ema_alpha)
    d_mi_s   = ema(np.nan_to_num(d_mi,   nan=0.0), alpha=args.ema_alpha)
    d_stab_s = ema(np.nan_to_num(d_stab, nan=0.0), alpha=args.ema_alpha)

    K_star = None
    for k in range(2, len(steps)+1):
        ok = True
        for j in range(k - args.knee_m + 1, k + 1):
            if j <= 1: ok = False; break
            if not (d_pred_s[j-1] < args.eps_pred and d_mi_s[j-1] < args.eps_mi and d_stab_s[j-1] < args.eps_stab):
                ok = False; break
        if ok:
            K_star = k - 1
            break
    if K_star is None:
        K_star = len(steps)

    with open(out_dir / "overprompting_knee.json", "w") as f:
        json.dump({
            "K_star": int(K_star),
            "eps_pred": args.eps_pred, "eps_mi": args.eps_mi, "eps_stab": args.eps_stab,
            "knee_m": args.knee_m, "ema_alpha": args.ema_alpha,
            "anchor_note": "boundary" if args.anchor_mode=="boundary" else "uncertainty"
        }, f, indent=2)
    print(f"[KNEE] Over-prompting knee K* = {K_star}")

    # ---- Redundancy flags (per-prompt) ----
    for rec in prompt_records:
        k = rec["step"]
        rec["dPred_global"] = float(d_pred[k-1]) if k>1 else float('nan')
        rec["dMI_global"]   = float(d_mi[k-1])   if k>1 else float('nan')
        rec["dStab_global"] = float(d_stab[k-1]) if k>1 else float('nan')
        small_global = ( (np.nan_to_num(rec["dPred_global"]) < args.eps_pred) and
                         (np.nan_to_num(rec["dMI_global"])   < args.eps_mi)   and
                         (np.nan_to_num(rec["dStab_global"]) < args.eps_stab) )
        small_local  = (rec["dU_local"] < args.eps_dU_local) and (rec["dMI_local"] < args.eps_dMI_local)
        big_overlap  = (rec["overlap_iou"] >= args.th_overlap)
        rec["redundant"] = bool(small_global and small_local and big_overlap)

    # Write redundancy report
    header = ["step","x","y","label","dPred_global","dMI_global","dStab_global","dU_local","dMI_local","overlap_iou","redundant"]
    rows = []
    for r in prompt_records:
        rows.append([r["step"], r["x"], r["y"], r["label"],
                    f"{r['dPred_global']:.6f}", f"{r['dMI_global']:.6f}", f"{r['dStab_global']:.6f}",
                    f"{r['dU_local']:.6f}", f"{r['dMI_local']:.6f}", f"{r['overlap_iou']:.6f}", int(r["redundant"])])
    save_csv(rows, header, out_dir / "redundancy_report.csv")

    # ---- Save curves & per-step CSV ----
    def plot_curve(ys, title, ylabel, fname, extra=None):
        fig, ax = plt.subplots(1,1,figsize=(7,4))
        ax.plot(steps, ys, marker='o', label=ylabel)
        if extra:
            for (lab, arr, style) in extra:
                ax.plot(steps, arr, style, label=lab)
        ax.set_xlabel("# prompts"); ax.set_ylabel(ylabel); ax.set_title(title); ax.grid(True, alpha=0.3); ax.legend()
        fig.tight_layout(); fig.savefig(out_dir / fname, dpi=220); plt.close(fig)

    plot_curve(pred_iou, "PredIoU vs prompts", "PredIoU", "curve_pred_iou.png")
    plot_curve(anch_mi,  f"Anchored weighted InfoNCE ({'boundary' if args.anchor_mode=='boundary' else 'uncertainty'} anchor)", "I(Z;Y)", "curve_anchored_mi.png")
    plot_curve(stab,     "Stability (hflip TTA) vs prompts", "IoU(mask, hflip(mask))", "curve_stability.png")
    plot_curve(unc_g,    "Uncertainty vs prompts", "mean p(1-p)", "curve_uncertainty.png",
               extra=[("Boundary band (approx)", unc_b, "--")])
    if gt_mask is not None:
        plot_curve(iou_gt, "GT IoU vs prompts", "IoU", "curve_gt_iou.png",
                   extra=[("Dice", dice_gt, "--")])

    # per-step CSV with deltas & EMA
    rows = []
    header = ["step","pred_iou","anch_mi","stab","unc_global","unc_boundary"]
    if gt_mask is not None: header += ["iou_gt","dice_gt"]
    for i,k in enumerate(steps):
        rr = [k, pred_iou[i], anch_mi[i], stab[i], unc_g[i], unc_b[i]]
        if gt_mask is not None: rr += [iou_gt[i], dice_gt[i]]
        rows.append(rr)
    header += ["d_pred","d_mi","d_stab","d_pred_ema","d_mi_ema","d_stab_ema"]
    for i,r in enumerate(rows):
        r += [float(d_pred[i] if not math.isnan(d_pred[i]) else 0.0),
              float(d_mi[i]   if not math.isnan(d_mi[i])   else 0.0),
              float(d_stab[i] if not math.isnan(d_stab[i]) else 0.0),
              float(d_pred_s[i]), float(d_mi_s[i]), float(d_stab_s[i])]
    save_csv(rows, header, out_dir / "metrics_per_step.csv")

    # ---- Refinement: early stop + prune redundant prompts with guards ----
    keep_idx = [i for i in range(K_star)]  # keep 0..K*-1 initially

    # Evaluate baseline at early stop
    base_pts = pts_all[:K_star]; base_lbl = lbl_all[:K_star]
    base_mi, base_out = evaluate_prompt_set(predictor, sam, img, base_pts, base_lbl, w_ref, args)
    base_pred, base_stab = base_out["pred_iou"], base_out["stability"]
    if gt_mask is not None:
        base_iou, base_dice = binary_iou_dice((base_out["prob"]>=0.5).astype(np.uint8), gt_mask)
    else:
        base_iou = base_dice = float('nan')

    # Candidates to remove: marked redundant within 1..K*
    candidates = [r for r in prompt_records if (r["step"]<=K_star and r["redundant"])]
    # remove the worst first: small local MI, small U drop, big overlap
    candidates.sort(key=lambda r: (r["dMI_local"], r["dU_local"], -r["overlap_iou"]))

    keep_set = set(keep_idx)
    for r in candidates:
        pos = r["step"] - 1
        trial = sorted(list(keep_set - {pos}))
        pts_trial = pts_all[trial]; lbl_trial = lbl_all[trial]
        mi_t, out_t = evaluate_prompt_set(predictor, sam, img, pts_trial, lbl_trial, w_ref, args)
        # Guard tolerances
        if (out_t["pred_iou"] + 1e-6) >= (base_pred - args.tol_pred) and \
           (mi_t              + 1e-6) >= (base_mi   - args.tol_mi)   and \
           (out_t["stability"]+ 1e-6) >= (base_stab - args.tol_stab):
            keep_set = set(trial)
            base_pred, base_mi, base_stab = out_t["pred_iou"], mi_t, out_t["stability"]
            if gt_mask is not None:
                base_iou, base_dice = binary_iou_dice((out_t["prob"]>=0.5).astype(np.uint8), gt_mask)

    keep_sorted = sorted(list(keep_set))
    refined_prompts = [prom_list[i].copy() for i in keep_sorted]
    for i,p in enumerate(refined_prompts, start=1): p["t"] = i
    with open(out_dir / "refined_prompts.json", "w") as f:
        json.dump({"prompts": refined_prompts}, f, indent=2)
    print(f"[REFINE] Wrote refined_prompts.json with {len(refined_prompts)} prompts (from {K_star})")

    # ---- Faithfulness eval (optional) ----
    if gt_mask is not None:
        # Full original
        mi_full, out_full = evaluate_prompt_set(predictor, sam, img, pts_all, lbl_all, w_ref, args)
        iou_full, dice_full = binary_iou_dice((out_full["prob"]>=0.5).astype(np.uint8), gt_mask)

        # Early-stop baseline
        iou_base, dice_base = base_iou, base_dice

        # Refined
        pts_ref = np.array([[p["x"],p["y"]] for p in refined_prompts], dtype=np.float32)
        lbl_ref = np.array([int(p["label"]) for p in refined_prompts], dtype=np.int32)
        mi_ref, out_ref = evaluate_prompt_set(predictor, sam, img, pts_ref, lbl_ref, w_ref, args)
        iou_ref, dice_ref = binary_iou_dice((out_ref["prob"]>=0.5).astype(np.uint8), gt_mask)

        rows = [
            ["original_full", len(pts_all), out_full["pred_iou"], mi_full, out_full["stability"], iou_full, dice_full],
            ["early_stop",    len(base_pts), base_pred,           base_mi, base_stab,            iou_base, dice_base],
            ["refined",       len(pts_ref),  out_ref["pred_iou"], mi_ref,  out_ref["stability"], iou_ref,  dice_ref],
        ]
        header = ["setting","num_prompts","PredIoU","AnchMI","Stability","IoU","Dice"]
        save_csv(rows, header, out_dir / "eval_refinement.csv")
        print(f"[EVAL] Wrote eval_refinement.csv (GT-based faithfulness).")

    # ---- Meta ----
    meta = {
        "image": args.image,
        "prompts": args.prompts,
        "checkpoint": args.checkpoint,
        "model_type": args.model_type,
        "token_grid": [int(Hf), int(Wf)],
        "K_star": int(K_star),
        "anchor_mode": args.anchor_mode,
        "decision_tolerances": {"PredIoU": args.tol_pred, "AnchMI": args.tol_mi, "Stability": args.tol_stab},
        "redundancy_thresholds": {
            "eps_pred": args.eps_pred, "eps_mi": args.eps_mi, "eps_stab": args.eps_stab,
            "eps_dU_local": args.eps_dU_local, "eps_dMI_local": args.eps_dMI_local, "th_overlap": args.th_overlap
        }
    }
    with open(out_dir / "run_meta.json", "w") as f: json.dump(meta, f, indent=2)

    print(f"[OK] Outputs in {out_dir.resolve()}")
    print("  - metrics_per_step.csv (with deltas & EMA), curves, redundancy_report.csv")
    print("  - overprompting_knee.json, refined_prompts.json")
    if gt_mask is not None:
        print("  - eval_refinement.csv (IoU/Dice for original vs early-stop vs refined)")


def parse_args():
    ap = argparse.ArgumentParser("Active prompting: detect knee, prune redundancy, refine prompts")
    ap.add_argument("--image", required=True)
    ap.add_argument("--prompts", required=True, help="JSON with time-ordered prompts [{t,x,y,label}]")
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--model-type", default="vit_h", choices=["vit_h","vit_l","vit_b"])
    ap.add_argument("--gt-mask", default=None, help="Optional GT mask (.png or .npy)")
    ap.add_argument("--gt-format", default="png_zero_is_fg",
                    choices=["png_zero_is_fg","png_nonzero_is_fg","npy_one_is_fg","npy_zero_is_fg"])
    ap.add_argument("--out-dir", default="active_refine_out")

    # Anchored MI
    ap.add_argument("--anchor-mode", default="boundary", choices=["boundary","uncertainty"])
    ap.add_argument("--ring-radius", type=int, default=2)
    ap.add_argument("--unc-gamma",   type=float, default=1.0)
    ap.add_argument("--infonce-steps", type=int, default=200)
    ap.add_argument("--infonce-local-steps", type=int, default=80)
    ap.add_argument("--infonce-batch", type=int, default=2048)
    ap.add_argument("--infonce-lr",    type=float, default=1e-3)
    ap.add_argument("--infonce-proj",  type=int, default=128)
    ap.add_argument("--infonce-temp",  type=float, default=0.1)

    # Knee detection
    ap.add_argument("--ema-alpha", type=float, default=0.4)
    ap.add_argument("--knee-m", type=int, default=2)
    ap.add_argument("--eps-pred", type=float, default=0.005)
    ap.add_argument("--eps-mi",   type=float, default=0.01)
    ap.add_argument("--eps-stab", type=float, default=0.01)

    # Redundancy (local)
    ap.add_argument("--local-radius", type=int, default=12)  # px
    ap.add_argument("--token-radius", type=int, default=2)   # tokens around prompt
    ap.add_argument("--eps-dU-local", type=float, default=0.002)
    ap.add_argument("--eps-dMI-local", type=float, default=0.005)
    ap.add_argument("--th-overlap", type=float, default=0.7)

    # Refinement guards (do not accept removal if proxies drop by more than these)
    ap.add_argument("--tol-pred", type=float, default=0.002)
    ap.add_argument("--tol-mi",   type=float, default=0.004)
    ap.add_argument("--tol-stab", type=float, default=0.005)

    ap.add_argument("--cpu", action="store_true")
    return ap.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(args)
