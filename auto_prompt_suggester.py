#!/usr/bin/env python3
"""
Active prompting (two-stage):
  1) FIRST PART (from your spec): detect over-prompting knee K*, flag redundancy, export refined prompts
  2) THEN: offline auto-prompt suggestion starting from the K* working set (or refined/full, via --suggest-from)

Decisions use GT-free proxies (PredIoU, Anchored weighted InfoNCE MI, Stability).
If GT is given, it’s used only for reporting IoU/Dice (faithfulness), not for decision making.

Outputs (in --out-dir):
  FIRST PART:
    - curves_* for PredIoU, Anchored MI, Stability, Uncertainty
    - metrics_per_step.csv (+ delta & EMA columns)
    - redundancy_report.csv (per-prompt)
    - overprompting_knee.json
    - refined_prompts.json (time-ordered, reindexed)
    - eval_refinement.csv (IoU/Dice original vs early-stop vs refined) [if GT provided]

  AUTO-SUGGEST:
    - auto_prompts.json         (suggested prompts in time order)
    - final_prompts.json        (working set + auto suggestions)
    - explain/* (optional plots per round if --save-explain)

Requirements:
  pip install git+https://github.com/facebookresearch/segment-anything.git
  pip install torch pillow matplotlib numpy
"""

import argparse, json, math, csv
from pathlib import Path
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import Normalize
from matplotlib.patches import Patch

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

def up_bilinear(hw, H, W):
    return F.interpolate(torch.from_numpy(hw)[None,None].float(), size=(H,W), mode="bilinear", align_corners=False)[0,0].numpy()

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


# ------------- Weighted InfoNCE (train-per-call, FIRST PART) -------------
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


# ------------- Evaluation for a given prompt set (FIRST PART) -------------
def evaluate_prompt_set(predictor, sam, img, pts_xy, lbl, w_ref, args, hooks_cache_out=None):
    """
    Run SAM on (img, prompts) and compute:
      - prob (H,W), PredIoU, Stability (hflip)
      - fused Z, pooled Y, token ownership labels, final attention, keys
      - Anchored weighted InfoNCE MI (train-per-call)
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
        # from hard_prompt import HardPromptEnforcer, HardPromptConfig
        # # (initialize once near the top of your script)
        # enforcer = HardPromptEnforcer(HardPromptConfig(
        #     r_pos_px=12, r_neg_px=12,
        #     ownership_margin=0.05, ownership_min_conf=0.10,
        #     conflict_policy="pos_wins",  # or "neg_wins"
        # ))
        # # ...
        # masks, scores, logits, extras = enforcer.strict_predict(
        #     predictor, sam, img, pts_xy if len(pts_xy) else np.zeros((0,2), np.float32), lbl if len(pts_xy) else np.zeros((0,), np.int32), multimask_output=False
        # )

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

    # Anchored MI (train-per-call, weighted rows by w_ref)
    mi_val = infoNCE_weighted_lower_bound(
        Z, Ytok, row_weights=w_ref.to(device),
        steps=args.infonce_steps, batch=args.infonce_batch, lr=args.infonce_lr,
        proj=args.infonce_proj, temp=args.infonce_temp, device=device
    )

    uncert_global = float((prob * (1-prob)).mean())
    out = {
        "prob": prob, "pred_iou": pred_iou, "stability": float(stability),
        "uncert_global": uncert_global, "uncert_boundary": uncert_global,
        "Z": Z.detach(), "Ytok": Ytok.detach(), "labels_tok": labels, "token_hw": (Hf,Wf),
        "A_final": Afin.detach(), "last_keys": last_k.detach()
    }
    if hooks_cache_out is not None:
        hooks_cache_out.update(out)
    return mi_val, out


# -------------------- AUTO-SUGGEST (second stage) --------------------
# (reuse a cheaper head-fitting approach: train heads ONCE per accepted state and
# reuse them to score candidate Y; after accepting, retrain heads.)

class CheapProj(nn.Module):
    def __init__(self, in_dim, proj_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, proj_dim), nn.ReLU(inplace=True),
            nn.Linear(proj_dim, proj_dim),
        )
    def forward(self, x): return self.net(x)

def fit_heads_once(Z, Y, steps, batch, lr, proj_dim, temp, device):
    Zn = (Z - Z.mean(0, keepdim=True)) / (Z.std(0, keepdim=True) + 1e-6)
    Yn = (Y - Y.mean(0, keepdim=True)) / (Y.std(0, keepdim=True) + 1e-6)
    fZ = CheapProj(Z.shape[1], proj_dim).to(device)
    fY = CheapProj(Y.shape[1], proj_dim).to(device)
    opt = torch.optim.Adam(list(fZ.parameters()) + list(fY.parameters()), lr=lr)
    N = Z.shape[0]
    for _ in range(steps):
        b = min(batch, N)
        idx = torch.randint(0, N, (b,), device=device)
        z_b = fZ(Zn[idx]); y_b = fY(Yn[idx])
        logits  = (z_b @ y_b.t()) / temp
        targets = torch.arange(b, device=device)
        loss = F.cross_entropy(logits, targets)
        opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
    with torch.no_grad():
        PZ = []
        for i in range(0, N, 4096):
            j = min(N, i+4096)
            PZ.append(fZ(Zn[i:j]))
        PZ = torch.cat(PZ, 0)
    return fZ, fY, PZ

@torch.no_grad()
def anchored_mi_from_heads(PZ, fY, Y, temp):
    Yn = (Y - Y.mean(0, keepdim=True)) / (Y.std(0, keepdim=True) + 1e-6)
    PY = []
    N = Y.shape[0]
    for i in range(0, N, 4096):
        j = min(N, i+4096)
        PY.append(fY(Yn[i:j]))
    PY = torch.cat(PY, 0)
    logits = (PZ @ PY.t()) / temp
    targets = torch.arange(N, device=logits.device)
    ce = F.cross_entropy(logits, targets)
    return float(-ce.item())

def overlay_heatmap(bg_rgb_uint8, heat_01, alpha=0.55, cmap='jet'):
    Hbg, Wbg = bg_rgb_uint8.shape[:2]
    h = heat_01.astype(np.float32)
    h -= h.min(); h /= (h.max() - h.min() + 1e-8)
    if h.shape != (Hbg, Wbg): h = up_bilinear(h, Hbg, Wbg)
    base = bg_rgb_uint8.astype(np.float32)/255.0
    colored = cm.get_cmap(cmap)(np.clip(h,0,1))[..., :3]
    out = (1 - alpha) * base + alpha * colored
    return np.clip(out, 0, 1)

def build_candidate_score_maps(img, prob, prob_flip, A_final, prompt_offset, Np, token_hw, pts_used, args):
    H, W = img.shape[:2]
    Hf, Wf = token_hw
    # U: uncertainty
    U = prob * (1.0 - prob)

    # D: disagreement with hflip
    if prob_flip is None:
        D = np.zeros_like(prob)
    else:
        D = np.abs(prob - hflip_mask(prob_flip))

    # C: coverage gap (1 - ownership confidence)
    if Np > 0:
        A = A_final.mean(dim=1)[0]  # [Nq, Ni]
        pr_idx = torch.arange(prompt_offset, prompt_offset+Np, device=A.device)
        A_pr = A[pr_idx]  # [Np, Ni]
        conf = torch.max(A_pr, dim=0).values.view(Hf, Wf).detach().cpu().numpy()  # [Hf,Wf]
        C_tok = 1.0 - conf
        C = up_bilinear(C_tok, H, W)
    else:
        C = np.ones_like(prob, dtype=np.float32)

    # B: boundary ring
    binm = (prob >= 0.5).astype(np.float32)
    t = torch.from_numpy(binm)[None,None]
    rad = args.boundary_ring
    dil = F.max_pool2d(t, kernel_size=2*rad+1, stride=1, padding=rad)
    ero = -F.max_pool2d(-t, kernel_size=2*rad+1, stride=1, padding=rad)
    B = (dil - ero)[0,0].numpy()

    def nz_norm(x):
        x = x.astype(np.float32)
        mn, mx = x.min(), x.max()
        return (x - mn) / (mx - mn + 1e-8) if mx > mn else np.zeros_like(x, dtype=np.float32)

    U, D, C, B = map(nz_norm, (U, D, C, B))
    Hmap = args.wU*U + args.wD*D + args.wC*C + args.wB*B

    # Cool-down around existing prompts
    if len(pts_used):
        yy, xx = np.indices((H, W))
        cool = np.ones((H, W), dtype=np.float32)
        for (x,y) in pts_used:
            rr2 = (xx - x)**2 + (yy - y)**2
            cool *= (rr2 >= (args.min_dist**2)).astype(np.float32)
        Hmap *= cool

    return Hmap, {"U":U, "D":D, "C":C, "B":B}

def topk_local_maxima(Hmap, K, nms_radius):
    H, W = Hmap.shape
    m = Hmap.copy()
    pts = []
    for _ in range(K*4):  # oversample attempts
        yx = np.unravel_index(np.argmax(m), m.shape)
        y, x = int(yx[0]), int(yx[1])
        s = float(m[y, x])
        if s <= 0: break
        pts.append((y, x, s))
        y0, y1 = max(0, y - nms_radius), min(H, y + nms_radius + 1)
        x0, x1 = max(0, x - nms_radius), min(W, x + nms_radius + 1)
        m[y0:y1, x0:x1] = 0.0
        if len(pts) >= K: break
    return pts

def evaluate_state_for_suggest(predictor, predictor_flip, sam, img, pts_xy, lbl, w_anchor,
                               fY, PZ, args, want_flip=True):
    """
    Decode mask; compute stability; compute Y tokens; get anchored MI via (fixed) PZ,fY.
    (Approximates MI(Z_current; Y_candidate); Z is held at current state's embedding.)
    """
    device = next(sam.parameters()).device
    H, W = img.shape[:2]

    predictor.set_image(img)
    with torch.inference_mode():
        masks, scores, logits = predictor.predict(
            point_coords=pts_xy if len(pts_xy) else np.zeros((0,2), np.float32),
            point_labels=lbl      if len(pts_xy) else np.zeros((0,), np.int32),
            multimask_output=False,
        )
        # from hard_prompt import HardPromptEnforcer, HardPromptConfig
        # # (initialize once near the top of your script)
        # enforcer = HardPromptEnforcer(HardPromptConfig(
        #     r_pos_px=12, r_neg_px=12,
        #     ownership_margin=0.05, ownership_min_conf=0.10,
        #     conflict_policy="pos_wins",  # or "neg_wins"
        # ))
        # # ...
        # masks, scores, logits, extras = enforcer.strict_predict(
        #     predictor, sam, img, pts_xy if len(pts_xy) else np.zeros((0,2), np.float32), lbl if len(pts_xy) else np.zeros((0,), np.int32), multimask_output=False
        # )

    prob = down_up_to_original(logits, predictor, device)
    pred_iou = float(scores[0]) if len(scores)>0 else float('nan')

    prob_flip = None; stability = 1.0
    if want_flip and args.use_tta:
        pts_flip = hflip_coords_xy(pts_xy.copy(), W) if len(pts_xy) else np.zeros((0,2), np.float32)
        predictor_flip.set_image(np.ascontiguousarray(np.fliplr(img)))
        with torch.inference_mode():
            mf, sf, lf = predictor_flip.predict(point_coords=pts_flip, point_labels=lbl if len(pts_xy) else np.zeros((0,), np.int32), multimask_output=False)
        prob_f = down_up_to_original(lf, predictor_flip, device)
        prob_flip = prob_f
        stability = binary_iou_dice((prob>=0.5).astype(np.uint8),
                                    (hflip_mask(prob_f)>=0.5).astype(np.uint8))[0]

    feats = predictor.features
    _, Cfeat, Hf, Wf = feats.shape
    Ytok = torch.from_numpy(area_pool_to_tokens(prob, Hf, Wf)).float().to(device)
    # anchored MI (weighted by w_anchor) using fixed heads
    # NOTE: for scoring, we don't weight by anchor here (same weights for evaluation & heads training is fine)
    mi_val = anchored_mi_from_heads(PZ, fY, Ytok, args.infonce_temp)

    out = {"prob": prob, "pred_iou": pred_iou, "stability": float(stability), "prob_flip": prob_flip}
    return mi_val, out


# ---------------- Main pipeline ----------------
def run(args):
    device = "cuda" if torch.cuda.is_available() and not args.cpu else "cpu"
    out_dir = Path(args.out_dir); ensure_dir(out_dir)
    ex_dir  = out_dir / "explain"; ensure_dir(ex_dir)

    # ---- Load inputs ----
    img = load_image_rgb(args.image)
    H, W = img.shape[:2]
    with open(args.prompts, "r") as f:
        prom = json.load(f)
    prom_list = sorted(prom["prompts"], key=lambda d: d["t"])
    pts_all = np.array([[p["x"], p["y"]] for p in prom_list], dtype=np.float32)
    lbl_all = np.array([int(p["label"]) for p in prom_list], dtype=np.int32)

    # Clip OOB prompts to image bounds
    pts_all[:, 0] = np.clip(pts_all[:, 0], 0, W - 1)
    pts_all[:, 1] = np.clip(pts_all[:, 1], 0, H - 1)

    gt_mask = load_mask_any(args.gt_mask, args.gt_format) if args.gt_mask else None
    if gt_mask is not None and gt_mask.shape != (H, W):
        raise ValueError("GT mask must match image size")

    # ---- Build SAM + predictor(s) ----
    sam = sam_model_registry[args.model_type](checkpoint=args.checkpoint)
    sam.to(device).eval()
    predictor = SamPredictor(sam)
    predictor.set_image(img)
    if predictor.features is None:
        raise RuntimeError("Predictor has no encoder features; set_image failed?")
    predictor_flip = SamPredictor(sam)  # for TTA in suggest stage

    _, Cfeat, Hf, Wf = predictor.features.shape
    Ni = Hf * Wf

    # ---- Anchor weights (fixed for FIRST PART) ----
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

    # =========================
    # FIRST PART: step-by-step (as in your code) -> detect knee K*
    # =========================
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

            # Local MI delta in token neighborhood (train-per-call small steps)
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

    # ---- Refinement (optional): build a pruned set; we still start suggestion from knee by default
    keep_idx = [i for i in range(K_star)]  # keep 0..K*-1 initially
    base_pts = pts_all[:K_star]; base_lbl = lbl_all[:K_star]
    base_mi, base_out = evaluate_prompt_set(predictor, sam, img, base_pts, base_lbl, w_ref, args)
    base_pred, base_stab = base_out["pred_iou"], base_out["stability"]
    if gt_mask is not None:
        base_iou, base_dice = binary_iou_dice((base_out["prob"]>=0.5).astype(np.uint8), gt_mask)
    else:
        base_iou = base_dice = float('nan')

    candidates = [r for r in prompt_records if (r["step"]<=K_star and r["redundant"])]
    candidates.sort(key=lambda r: (r["dMI_local"], r["dU_local"], -r["overlap_iou"]))

    keep_set = set(keep_idx)
    for r in candidates:
        pos = r["step"] - 1
        trial = sorted(list(keep_set - {pos}))
        pts_trial = pts_all[trial]; lbl_trial = lbl_all[trial]
        mi_t, out_t = evaluate_prompt_set(predictor, sam, img, pts_trial, lbl_trial, w_ref, args)
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

    if gt_mask is not None:
        mi_full, out_full = evaluate_prompt_set(predictor, sam, img, pts_all, lbl_all, w_ref, args)
        iou_full, dice_full = binary_iou_dice((out_full["prob"]>=0.5).astype(np.uint8), gt_mask)
        pts_ref = np.array([[p["x"],p["y"]] for p in refined_prompts], dtype=np.float32)
        lbl_ref = np.array([int(p["label"]) for p in refined_prompts], dtype=np.int32)
        mi_ref, out_ref = evaluate_prompt_set(predictor, sam, img, pts_ref, lbl_ref, w_ref, args)
        iou_ref, dice_ref = binary_iou_dice((out_ref["prob"]>=0.5).astype(np.uint8), gt_mask)
        rows = [
            ["original_full", len(pts_all), out_full["pred_iou"], mi_full, out_full["stability"], iou_full, dice_full],
            ["early_stop",    len(base_pts), base_pred,           base_mi, base_stab,            base_iou, base_dice],
            ["refined",       len(pts_ref),  out_ref["pred_iou"], mi_ref,  out_ref["stability"], iou_ref,  dice_ref],
        ]
        header = ["setting","num_prompts","PredIoU","AnchMI","Stability","IoU","Dice"]
        save_csv(rows, header, out_dir / "eval_refinement.csv")
        print(f"[EVAL] Wrote eval_refinement.csv (GT-based faithfulness).")

    # =========================
    # SECOND PART: AUTO PROMPT SUGGESTION (offline)
    # =========================
    # choose base set to start from
    if args.suggest_from == "knee":
        pts_work = pts_all[:K_star].copy()
        lbl_work = lbl_all[:K_star].copy()
    elif args.suggest_from == "refined":
        pts_work = np.array([[p["x"], p["y"]] for p in refined_prompts], dtype=np.float32)
        lbl_work = np.array([int(p["label"]) for p in refined_prompts], dtype=np.int32)
    else:  # "full"
        pts_work = pts_all.copy()
        lbl_work = lbl_all.copy()

    # Evaluate working set to get Z/Y for heads
    mi_w, out_w = evaluate_prompt_set(predictor, sam, img, pts_work, lbl_work, w_ref, args)
    Z_w = out_w["Z"].to(device); Y_w = out_w["Ytok"].to(device)
    fZ, fY, PZ = fit_heads_once(Z_w, Y_w, args.infonce_steps, args.infonce_batch,
                                args.infonce_lr, args.infonce_proj, args.infonce_temp, device)
    mi_w = anchored_mi_from_heads(PZ, fY, Y_w, args.infonce_temp)

    auto_prompts = []
    rounds_hist = []
    print(f"[AUTO] Start suggestions from '{args.suggest_from}' set: {len(pts_work)} prompts")

    for round_id in range(1, args.budget_max + 1):
        # Knee gate on last deltas (based on FIRST PART’s emA thresholds)
        if len(pred_iou) >= 1 and len(anch_mi) >= 1 and len(stab) >= 1:
            if (d_pred_s[-1] < args.eps_pred) and (d_mi_s[-1] < args.eps_mi) and (d_stab_s[-1] < args.eps_stab):
                print(f"[AUTO] Knee already closed by proxies; stop suggestions.")
                break

        # Build candidate heatmap from current state
        # We need A_final and prompt_offset -> re-run one pass to cache internals
        mi_w_tmp, out_w = evaluate_prompt_set(predictor, sam, img, pts_work, lbl_work, w_ref, args)
        Afin = out_w["A_final"]; Hf, Wf = out_w["token_hw"]; Np = len(lbl_work)
        num_mask_tokens = getattr(sam.mask_decoder, "num_mask_tokens", 3)
        prompt_offset = 1 + num_mask_tokens

        Hmap, comps = build_candidate_score_maps(
            img=img, prob=out_w["prob"], prob_flip=None,
            A_final=Afin, prompt_offset=prompt_offset,
            Np=Np, token_hw=(Hf,Wf), pts_used=pts_work, args=args
        )
        cands = topk_local_maxima(Hmap, K=args.cand_topk, nms_radius=args.nms_radius)
        if len(cands) == 0:
            print("[AUTO] No candidate maxima found. Stopping.")
            break

        # Evaluate each candidate with (approx) anchored MI via fixed heads
        best = None
        for (y, x, score) in cands:
            for lab in (1, 0):
                pts_try = np.vstack([pts_work, np.array([[x, y]], dtype=np.float32)])
                lbl_try = np.concatenate([lbl_work, np.array([lab], dtype=np.int32)], axis=0)

                mi_try, out_try = evaluate_state_for_suggest(predictor, predictor_flip, sam, img,
                                                             pts_try, lbl_try, w_ref,
                                                             fY=fY, PZ=PZ, args=args, want_flip=args.use_tta)
                d_mi    = mi_try - mi_w
                prev_bin = (out_w["prob"]  >= 0.5).astype(np.uint8)
                now_bin  = (out_try["prob"]>= 0.5).astype(np.uint8)
                prev_empty = int(prev_bin.sum() == 0)
                now_empty  = int(now_bin.sum()  == 0)
                comp_pen   = float(max(0, now_empty - prev_empty))  # penalize if became empty

                d_stab = out_try["stability"] - out_w["stability"]
                utility = args.u_mi * d_mi + args.u_stab * d_stab - args.u_pen * comp_pen
                cand = {"x":int(x), "y":int(y), "label":int(lab), "score":float(score),
                        "d_mi":float(d_mi), "d_stab":float(d_stab), "pen":float(comp_pen),
                        "utility":float(utility), "prob": out_try["prob"], "mi": float(mi_try),
                        "pred_iou": out_try["pred_iou"], "stability": out_try["stability"]}
                if (best is None) or (utility > best["utility"]):
                    best = cand

        if (best is None) or (best["utility"] < args.util_min) or (best["d_mi"] < args.delta_mi_min):
            print("[AUTO] No candidate passes utility thresholds. Stopping.")
            break

        # Accept best candidate
        pts_work = np.vstack([pts_work, np.array([[best["x"], best["y"]]], dtype=np.float32)])
        lbl_work = np.concatenate([lbl_work, np.array([best["label"]], dtype=np.int32)], axis=0)

        # Re-fit heads on the NEW state (Z changes dimension by +1 attention channel)
        mi_tmp, out_tmp = evaluate_prompt_set(predictor, sam, img, pts_work, lbl_work, w_ref, args)
        Z_w = out_tmp["Z"].to(device); Y_w = out_tmp["Ytok"].to(device)
        fZ, fY, PZ = fit_heads_once(Z_w, Y_w, args.infonce_steps, args.infonce_batch,
                                    args.infonce_lr, args.infonce_proj, args.infonce_temp, device)
        mi_w = anchored_mi_from_heads(PZ, fY, Y_w, args.infonce_temp)

        auto_prompts.append({"t": len(auto_prompts)+1, "x": best["x"], "y": best["y"], "label": best["label"],
                             "utility": best["utility"], "d_mi": best["d_mi"], "d_stab": best["d_stab"]})
        print(f"[AUTO][ACCEPT #{round_id}] ({best['x']},{best['y']}) L={best['label']}  util={best['utility']:.4f}  "
              f"dMI={best['d_mi']:.4f}  dStab={best['d_stab']:.4f}")

        # Optional explain overlay
        if args.save_explain:
            fig, ax = plt.subplots(1, 1, figsize=(8, 8*(H/W)))
            ax.imshow(overlay_heatmap(img, Hmap, alpha=0.6, cmap=args.cmap))
            ax.scatter([best["x"]],[best["y"]], c='yellow' if best["label"]==1 else 'red',
                       s=70, marker='o', edgecolor='black', linewidths=0.8)
            ax.set_axis_off(); ax.set_title(f"Chosen ({best['x']},{best['y']}) label {best['label']}")
            fig.tight_layout(); fig.savefig(ex_dir / f"round_{round_id:02d}_candidate_heatmap.png", dpi=200); plt.close(fig)

    # Write suggestion outputs
    Path(out_dir / "auto_prompts.json").write_text(json.dumps({"prompts": auto_prompts}, indent=2))
    final_prompts = [p.copy() for p in prom_list[: (K_star if args.suggest_from=='knee' else len(refined_prompts) if args.suggest_from=='refined' else len(prom_list))]]
    offset = len(final_prompts)
    for j, ap in enumerate(auto_prompts, start=1):
        final_prompts.append({"t": offset + j, "x": ap["x"], "y": ap["y"], "label": ap["label"]})
    Path(out_dir / "final_prompts.json").write_text(json.dumps({"prompts": final_prompts}, indent=2))
    print(f"[AUTO][OK] Wrote auto_prompts.json and final_prompts.json to {out_dir.resolve()}")

    # ---- Meta ----
    meta = {
        "image": args.image,
        "prompts": args.prompts,
        "checkpoint": args.checkpoint,
        "model_type": args.model_type,
        "token_grid": [int(Hf), int(Wf)],
        "K_star": int(K_star),
        "anchor_mode": args.anchor_mode,
        "suggest_from": args.suggest_from,
        "suggest_budget": args.budget_max
    }
    with open(out_dir / "run_meta.json", "w") as f: json.dump(meta, f, indent=2)
    print(f"[DONE] Outputs in {out_dir.resolve()}")


def parse_args():
    ap = argparse.ArgumentParser("Active prompting: detect knee, prune redundancy, then offline auto-suggest")
    ap.add_argument("--image", required=True)
    ap.add_argument("--prompts", required=True, help="JSON with time-ordered prompts [{t,x,y,label}]")
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--model-type", default="vit_h", choices=["vit_h","vit_l","vit_b"])
    ap.add_argument("--gt-mask", default=None, help="Optional GT mask (.png or .npy)")
    ap.add_argument("--gt-format", default="png_zero_is_fg",
                    choices=["png_zero_is_fg","png_nonzero_is_fg","npy_one_is_fg","npy_zero_is_fg"])
    ap.add_argument("--out-dir", default="active_refine_out")

    # Anchored MI (FIRST PART)
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

    # Refinement guards
    ap.add_argument("--tol-pred", type=float, default=0.002)
    ap.add_argument("--tol-mi",   type=float, default=0.004)
    ap.add_argument("--tol-stab", type=float, default=0.005)

    # ---------------- AUTO SUGGEST OPTIONS ----------------
    ap.add_argument("--suggest-from", default="knee", choices=["knee","refined","full"],
                    help="Which set to start suggestions from.")
    # Candidate heatmap weights: H = wU*uncertainty + wD*flip-disagreement + wC*coverage-gap + wB*boundary-ring
    ap.add_argument("--wU", type=float, default=1.0)
    ap.add_argument("--wD", type=float, default=0.8)
    ap.add_argument("--wC", type=float, default=0.7)
    ap.add_argument("--wB", type=float, default=0.6)
    ap.add_argument("--boundary-ring", type=int, default=3)
    ap.add_argument("--cand-topk", type=int, default=20)
    ap.add_argument("--nms-radius", type=int, default=15)
    ap.add_argument("--min-dist",   type=int, default=16)

    # Utility thresholds
    ap.add_argument("--u-mi",   type=float, default=1.0)
    ap.add_argument("--u-stab", type=float, default=0.6)
    ap.add_argument("--u-pen",  type=float, default=0.0)
    ap.add_argument("--util-min",     type=float, default=0.005)
    ap.add_argument("--delta-mi-min", type=float, default=0.002)

    ap.add_argument("--budget-max", type=int, default=8)
    ap.add_argument("--use-tta", action="store_true", help="Use hflip stability in suggestion (extra decode)")
    ap.add_argument("--save-explain", action="store_true")
    ap.add_argument("--cmap", default="jet")

    ap.add_argument("--cpu", action="store_true")
    return ap.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(args)
