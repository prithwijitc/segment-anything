#!/usr/bin/env python3
"""
Interactive Active Prompting for SAM
------------------------------------

What you get (live, after each click):
- Current mask overlaid on the image (with "hard_prompt" constraints enforced).
- Auto-suggestion heatmap + best candidate marker (press 'a' to accept).
- Curves for Anchored InfoNCE MI and PredIoU (terminal prints also include Stability & Uncertainty).
- Redundancy flags and over-prompting knee K* detection (printed in terminal).
- Safe sampling for InfoNCE (fixes CUDA multinomial assertion with zero-mass weights).

Controls:
  Mouse: left-click on the left (main) panel to add a prompt at the current label.
  Keys:
    p / n   – switch current label to Positive or Negative
    a       – accept the shown auto-suggested prompt (best candidate)
    u       – undo last user prompt
    s       – save current prompts to out_dir/interactive_prompts.json
    q / ESC – quit

Requirements:
  pip install git+https://github.com/facebookresearch/segment-anything.git
  pip install torch pillow matplotlib numpy

Notes:
- GT mask (if provided) is used ONLY for terminal reporting (IoU/Dice), not for decisions.
- The "hard_prompt" enforcer forces positives to 1 and negatives to 0 inside small disks.
- Anchored MI uses Step-0 boundary ring (fallback to uncertainty) as fixed weights.
"""

import argparse, json, math, csv, time
from pathlib import Path

import numpy as np
from PIL import Image

import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.gridspec import GridSpec

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

def binary_iou_dice(pred01, gt01):
    inter = np.logical_and(pred01, gt01).sum()
    union = np.logical_or (pred01, gt01).sum()
    iou  = inter / (union + 1e-8)
    dice = (2*inter) / (pred01.sum() + gt01.sum() + 1e-8)
    return float(iou), float(dice)

def save_json(obj, path):
    Path(path).write_text(json.dumps(obj, indent=2))

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

def up_bilinear(hw, H, W):
    return F.interpolate(torch.from_numpy(hw)[None,None].float(), size=(H,W), mode="bilinear", align_corners=False)[0,0].numpy()

def up_nearest(hw, H, W):
    return F.interpolate(torch.from_numpy(hw)[None,None].float(), size=(H,W), mode="nearest")[0,0].numpy()

def hflip_coords_xy(coords_xy, W):
    out = coords_xy.copy()
    if out.size == 0: return out
    out[:,0] = (W - 1) - out[:,0]
    return out

def hflip_mask(prob):
    return np.flip(prob, axis=1).copy()


# ------------- Safe probability vector (for multinomial) -------------
def safe_prob_vector(w, N, device, eps=1e-12):
    """
    Returns a length-N, nonnegative, strictly positive-sum probability vector on `device`.
    Falls back to uniform if input is invalid / empty. Adds tiny smoothing to avoid exact zeros.
    """
    w = torch.as_tensor(w, device=device, dtype=torch.float32).view(-1)
    if w.numel() != N:
        return torch.full((N,), 1.0/float(N), device=device, dtype=torch.float32)
    w = torch.nan_to_num(w, nan=0.0, posinf=0.0, neginf=0.0)
    w.clamp_(min=0.0)
    s = w.sum()
    if (not torch.isfinite(s)) or (s.item() <= 0.0):
        return torch.full((N,), 1.0/float(N), device=device, dtype=torch.float32)
    w = w / s
    w = w + eps
    w = w / w.sum()
    return w


# ------------- Anchors (fixed yardstick) -------------
def boundary_ring_weights_from_binary(mask_tok_hw, radius=2):
    t = torch.from_numpy(mask_tok_hw.astype(np.float32))[None, None]
    pad = radius
    dil = F.max_pool2d(t, kernel_size=2*radius+1, stride=1, padding=pad)
    ero = -F.max_pool2d(-t, kernel_size=2*radius+1, stride=1, padding=pad)
    ring = (dil - ero).clamp(min=0.0)[0,0]  # [Ht,Wt]
    w = ring.view(-1)
    # normalize later with safe_prob_vector
    return w

def uncertainty_weights_from_prob_tok(prob_tok_hw, gamma=1.0):
    p = torch.from_numpy(prob_tok_hw.astype(np.float32))
    w = (p * (1.0 - p)).pow(gamma)
    return w.view(-1)


# ------------- Two-Way hooks (final t2i + last keys) -------------
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


# ------------- Hard Prompt Enforcer -------------
class HardPromptConfig:
    def __init__(self, r_pos_px=12, r_neg_px=12, conflict_policy="pos_wins"):
        self.r_pos_px = int(r_pos_px)
        self.r_neg_px = int(r_neg_px)
        assert conflict_policy in ("pos_wins","neg_wins")
        self.conflict_policy = conflict_policy

class HardPromptEnforcer:
    def __init__(self, cfg: HardPromptConfig):
        self.cfg = cfg

    @staticmethod
    def _disk_mask(H, W, x, y, r):
        x = int(round(x)); y = int(round(y))
        if r <= 0:
            return None
        x0 = max(0, x - r); x1 = min(W, x + r + 1)
        y0 = max(0, y - r); y1 = min(H, y + r + 1)
        if x0 >= x1 or y0 >= y1: return None
        xs = np.arange(x0, x1); ys = np.arange(y0, y1)
        xx, yy = np.meshgrid(xs, ys)
        disk = (xx - x)**2 + (yy - y)**2 <= r*r
        if not disk.any(): return None
        return (slice(y0,y1), slice(x0,x1), disk)

    @torch.no_grad()
    def strict_predict(self, predictor, sam, img, pts_xy, lbl, multimask_output=False):
        """
        Run SAM, decode to prob in original space, then enforce hard constraints:
        - Pos disks -> prob=1
        - Neg disks -> prob=0
        Returns masks/scores/logits compatible with downstream use, plus extras.
        """
        device = next(sam.parameters()).device
        H, W = img.shape[:2]

        predictor.set_image(img)
        masks, scores, logits = predictor.predict(
            point_coords=pts_xy if len(pts_xy) else np.zeros((0,2), np.float32),
            point_labels=lbl      if len(pts_xy) else np.zeros((0,), np.int32),
            multimask_output=multimask_output,
        )

        # Robust decode of lowres logits to original (H,W)
        low = torch.as_tensor(logits, device=device)
        if low.ndim == 2:           # (256,256)
            low = low[None, None]   # [1,1,256,256]
        elif low.ndim == 3:         # (1,256,256) or (N,256,256)
            if low.shape[0] == 1 and low.shape[1] == 256:
                low = low[:, None, :, :]     # [1,1,256,256]
            else:
                low = low[None, None, :, :]  # [1,1,H,W] (rare)
        elif low.ndim == 4 and low.shape[1] != 1:
            low = low[:, :1, ...]    # keep first channel
        model = getattr(predictor, "model", None)
        if model is not None and hasattr(model, "postprocess_masks"):
            prob = torch.sigmoid(model.postprocess_masks(low,
                                input_size=predictor.input_size,
                                original_size=predictor.original_size)[0,0]).cpu().numpy()
        else:
            prob = torch.sigmoid(F.interpolate(low, size=(H,W), mode="bilinear", align_corners=False)[0,0]).cpu().numpy()

        # Build positive/negative union disks
        pos_union = np.zeros((H, W), dtype=np.uint8)
        neg_union = np.zeros((H, W), dtype=np.uint8)
        for i in range(len(lbl)):
            x, y, lab = int(pts_xy[i,0]), int(pts_xy[i,1]), int(lbl[i])
            rad = self.cfg.r_pos_px if lab == 1 else self.cfg.r_neg_px
            d = self._disk_mask(H, W, x, y, rad)
            if d is None: continue
            ys, xs, disk = d
            if lab == 1:
                pos_union[ys, xs][disk] = 1
            else:
                neg_union[ys, xs][disk] = 1

        # Resolve conflicts
        if self.cfg.conflict_policy == "pos_wins":
            neg_union = neg_union * (1 - pos_union)
        else:
            pos_union = pos_union * (1 - neg_union)

        # Enforce
        prob[pos_union.astype(bool)] = 1.0
        prob[neg_union.astype(bool)] = 0.0

        # Return compatible tensors
        mask01 = (prob >= 0.5).astype(np.uint8)
        # Rebuild scores (keep SAM score if available)
        score = float(scores[0]) if len(scores)>0 else float(np.mean(mask01))
        out_masks = np.expand_dims(mask01, axis=0)  # [1,H,W]
        # Create a fake "lowres logits" consistent with prob
        eps = 1e-6
        logit = np.log(np.clip(prob, eps, 1-eps) / np.clip(1-prob, eps, 1-eps))
        # Down to 256x256 to mimic SAM's lowres (for others using it)
        lowres = up_nearest(up_bilinear(logit, 256, 256), 256, 256)

        extras = {"pos_union": pos_union, "neg_union": neg_union}
        return out_masks, np.array([score], np.float32), lowres, extras


# ------------- Weighted InfoNCE -------------
class ProjectionHead(nn.Module):
    def __init__(self, in_dim, proj_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, proj_dim), nn.ReLU(inplace=True),
            nn.Linear(proj_dim, proj_dim),
        )
    def forward(self, x): return self.net(x)

def infoNCE_weighted_lower_bound(Z, Y, row_weights, steps=150, batch=2048, lr=1e-3, proj=128, temp=0.1, device="cuda"):
    N = Z.shape[0]
    fZ = ProjectionHead(Z.shape[1], proj_dim=proj).to(device)
    fY = ProjectionHead(Y.shape[1], proj_dim=proj).to(device)
    opt = torch.optim.Adam(list(fZ.parameters()) + list(fY.parameters()), lr=lr)

    Zn = (Z - Z.mean(0, keepdim=True)) / (Z.std(0, keepdim=True) + 1e-6)
    Yn = (Y - Y.mean(0, keepdim=True)) / (Y.std(0, keepdim=True) + 1e-6)

    rw = safe_prob_vector(row_weights, N, device)

    for _ in range(steps):
        b = min(batch, N)
        idx = torch.multinomial(rw, num_samples=b, replacement=True)
        z_b = fZ(Zn[idx]); y_b = fY(Yn[idx])
        logits  = (z_b @ y_b.t()) / temp
        targets = torch.arange(b, device=device)
        loss_per = F.cross_entropy(logits, targets, reduction='none')
        rwb = safe_prob_vector(rw[idx], b, device)  # re-normalize batch weights
        loss = (rwb * loss_per).sum()
        opt.zero_grad(set_to_none=True); loss.backward(); opt.step()

    with torch.no_grad():
        b = min(4096, N)
        idx = torch.multinomial(rw, num_samples=b, replacement=True)
        z_b = fZ(Zn[idx]); y_b = fY(Yn[idx])
        logits  = (z_b @ y_b.t()) / temp
        targets = torch.arange(b, device=device)
        ce = F.cross_entropy(logits, targets, reduction='none')
        rwb = safe_prob_vector(rw[idx], b, device)
        lb = float(-(rwb * ce).sum().item())
    return lb


# ------------- SAM utilities -------------
def down_up_to_original(lowres_logits, predictor, device):
    low = torch.as_tensor(lowres_logits, device=device)
    if low.ndim == 2: low = low[None, None]
    elif low.ndim == 3: low = low[:, None, :, :]
    elif low.ndim == 4 and low.shape[1] != 1: low = low[:, :1, ...]
    model = getattr(predictor, "model", None)
    H,W = predictor.original_size
    if model is not None and hasattr(model, "postprocess_masks"):
        prob = torch.sigmoid(model.postprocess_masks(low, input_size=predictor.input_size, original_size=predictor.original_size)[0,0])
    else:
        prob = torch.sigmoid(F.interpolate(low, size=(H,W), mode="bilinear", align_corners=False)[0,0])
    return prob.detach().cpu().numpy()


# ------------- Anchored heads (fit once per state) -------------
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


# ------------- Candidate maps & NMS -------------
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
    U = prob * (1.0 - prob)
    if prob_flip is None:
        D = np.zeros_like(prob)
    else:
        D = np.abs(prob - hflip_mask(prob_flip))

    if Np > 0:
        A = A_final.mean(dim=1)[0]  # [Nq, Ni]
        pr_idx = torch.arange(prompt_offset, prompt_offset+Np, device=A.device)
        A_pr = A[pr_idx]  # [Np, Ni]
        conf = torch.max(A_pr, dim=0).values.view(Hf, Wf).detach().cpu().numpy()
        C_tok = 1.0 - conf
        C = up_bilinear(C_tok, H, W)
    else:
        C = np.ones_like(prob, dtype=np.float32)

    # boundary ring around current mask
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
    for _ in range(K*4):
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


# ------------- Evaluation (one state) -------------
def evaluate_prompt_set(predictor, sam, enforcer, img, pts_xy, lbl, w_anchor, args, want_flip=True):
    """
    Run SAM (+ hard_prompt), decode to prob, compute stability, tokens Z/Y, Anchored MI (train-per-call).
    Returns metrics + internals needed for suggestion.
    """
    device = next(sam.parameters()).device
    H, W = img.shape[:2]

    predictor.set_image(img)
    handles, cache = attach_two_way_hooks(sam.mask_decoder.transformer)
    with torch.inference_mode():
        masks, scores, logits, extras = enforcer.strict_predict(
            predictor, sam, img,
            pts_xy if len(pts_xy) else np.zeros((0,2), np.float32),
            lbl    if len(pts_xy) else np.zeros((0,), np.int32),
            multimask_output=False
        )
    remove_hooks(handles)

    prob = down_up_to_original(logits, predictor, device)
    pred_iou = float(scores[0]) if len(scores)>0 else float('nan')

    prob_flip = None; stability = 1.0
    if want_flip and args.use_tta:
        predictor_flip = SamPredictor(sam)
        predictor_flip.set_image(np.ascontiguousarray(np.fliplr(img)))
        pts_flip = hflip_coords_xy(pts_xy.copy(), W) if len(pts_xy) else np.zeros((0,2), np.float32)
        with torch.inference_mode():
            mf, sf, lf, _ = enforcer.strict_predict(
                predictor_flip, sam, np.ascontiguousarray(np.fliplr(img)),
                pts_flip if len(pts_xy) else np.zeros((0,2), np.float32),
                lbl      if len(pts_xy) else np.zeros((0,), np.int32),
                multimask_output=False
            )
        prob_f = down_up_to_original(lf, predictor_flip, device)
        prob_flip = prob_f
        stability = binary_iou_dice((prob>=0.5).astype(np.uint8),
                                    (hflip_mask(prob_f)>=0.5).astype(np.uint8))[0]

    feats = predictor.features
    _, Cfeat, Hf, Wf = feats.shape
    Ni = Hf*Wf

    # Z from last keys + per-prompt attention slice
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
    else:
        A_pr = torch.zeros((0, Ni), device=A.device)
    K_last = last_k[0]                 # [Ni, Ck]
    Z = torch.cat([K_last, A_pr.transpose(0,1)], dim=1).to(device)  # [Ni, Ck+Np]
    Ytok = torch.from_numpy(area_pool_to_tokens(prob, Hf, Wf)).float().to(device)  # [Ni,1]

    # Anchored MI (train-per-call)
    mi_val = infoNCE_weighted_lower_bound(
        Z, Ytok, row_weights=w_anchor.to(device),
        steps=args.infonce_steps, batch=args.infonce_batch, lr=args.infonce_lr,
        proj=args.infonce_proj, temp=args.infonce_temp, device=device
    )

    uncert_global = float((prob * (1-prob)).mean())
    out = {
        "prob": prob, "prob_flip": prob_flip, "pred_iou": pred_iou, "stability": float(stability),
        "uncert_global": uncert_global,
        "Z": Z.detach(), "Ytok": Ytok.detach(),
        "A_final": Afin.detach(), "token_hw": (Hf,Wf), "prompt_offset": prompt_offset
    }
    return mi_val, out


# ------------- Interactive App -------------
class InteractiveApp:
    def __init__(self, args):
        self.args = args
        self.device = "cuda:1" if torch.cuda.is_available() and not args.cpu else "cpu"
        ensure_dir(args.out_dir)

        # Load image & optional GT
        self.img = load_image_rgb(args.image)
        self.H, self.W = self.img.shape[:2]
        self.gt = load_mask_any(args.gt_mask, args.gt_format) if args.gt_mask else None
        if self.gt is not None and self.gt.shape != (self.H, self.W):
            raise ValueError("GT mask must match image size")

        # Load initial prompts (optional)
        self.pts = []
        self.lbl = []
        if args.prompts:
            prom = json.loads(Path(args.prompts).read_text())
            prom_list = sorted(prom["prompts"], key=lambda d: d["t"])
            for p in prom_list:
                x = int(np.clip(p["x"], 0, self.W-1))
                y = int(np.clip(p["y"], 0, self.H-1))
                self.pts.append([x,y]); self.lbl.append(int(p["label"]))

        # Build SAM
        self.sam = sam_model_registry[args.model_type](checkpoint=args.checkpoint)
        self.sam.to(self.device).eval()
        self.predictor = SamPredictor(self.sam)
        self.predictor.set_image(self.img)
        if self.predictor.features is None:
            raise RuntimeError("Predictor failed to cache features; set_image error?")
        _, Cfeat, self.Hf, self.Wf = self.predictor.features.shape
        self.Ni = self.Hf * self.Wf

        # Hard prompt enforcer
        self.enforcer = HardPromptEnforcer(HardPromptConfig(
            r_pos_px=args.r_pos, r_neg_px=args.r_neg, conflict_policy=args.conflict_policy
        ))

        # Anchor weights (fixed)
        self.w_anchor = self._init_anchor_weights()

        # Live metrics history
        self.steps = []
        self.mi_hist = []
        self.pred_hist = []
        self.stab_hist = []
        self.unc_hist = []
        self.knee_m = args.knee_m
        self.eps_pred, self.eps_mi, self.eps_stab = args.eps_pred, args.eps_mi, args.eps_stab
        self.ema_alpha = args.ema_alpha
        self.K_star = None

        # Auto-suggest (live heads cache)
        self.heads = None  # tuple(fZ, fY, PZ) fitted on current state

        # UI state
        self.current_label = 1  # 1=positive, 0=negative
        self.suggestion = None  # dict with x,y,label,utility, etc.

        # Figure & events
        self._build_figure()
        self._recompute_and_redraw(initial=True)
        plt.show()

    # --------- helpers ---------
    def _init_anchor_weights(self):
        # Anchor = Step-0 boundary ring (fallback to uncertainty)
        # Run a blank decode to get prob0
        enforcer = self.enforcer
        mi0, out0 = evaluate_prompt_set(self.predictor, self.sam, enforcer,
                                        self.img, np.zeros((0,2), np.float32), np.zeros((0,), np.int32),
                                        w_anchor=torch.ones((self.Ni,), dtype=torch.float32),
                                        args=self.args, want_flip=False)
        prob0 = out0["prob"]
        prob0_tok = area_pool_to_tokens(prob0, self.Hf, self.Wf).reshape(self.Hf, self.Wf)
        if self.args.anchor_mode == "boundary":
            w = boundary_ring_weights_from_binary((prob0_tok >= 0.5).astype(np.float32), radius=self.args.ring_radius)
        else:
            w = uncertainty_weights_from_prob_tok(prob0_tok, gamma=self.args.unc_gamma)
        w = safe_prob_vector(w, self.Ni, self.device)
        return w

    def _build_figure(self):
        self.fig = plt.figure(figsize=(12, 8))
        gs = GridSpec(2, 2, figure=self.fig, height_ratios=[1,1], width_ratios=[1,1])
        self.ax_main = self.fig.add_subplot(gs[0,0])
        self.ax_heat = self.fig.add_subplot(gs[0,1])
        self.ax_curve = self.fig.add_subplot(gs[1,0])
        self.ax_unc = self.fig.add_subplot(gs[1,1])

        self.ax_main.set_title("Image + current mask")
        self.ax_heat.set_title("Suggestion heatmap (topK + best)")
        self.ax_curve.set_title("Curves (Anchored MI & PredIoU)")
        self.ax_unc.set_title("Uncertainty p(1-p)")

        self.fig.canvas.mpl_connect('button_press_event', self._on_click)
        self.fig.canvas.mpl_connect('key_press_event', self._on_key)

    def _render_prompts(self, ax):
        if len(self.pts) == 0: return
        pts = np.array(self.pts, dtype=np.float32)
        lbl = np.array(self.lbl, dtype=np.int32)
        pos = pts[lbl==1] if (lbl==1).any() else np.zeros((0,2))
        neg = pts[lbl==0] if (lbl==0).any() else np.zeros((0,2))
        if len(pos): ax.scatter(pos[:,0], pos[:,1], c='lime', s=40, edgecolor='black', linewidths=0.6)
        if len(neg): ax.scatter(neg[:,0], neg[:,1], c='red',  s=40, edgecolor='black', linewidths=0.6)

    def _knee_detect(self):
        if len(self.steps) < 3:
            self.K_star = None
            return
        d_pred = np.diff(self.pred_hist, prepend=self.pred_hist[:1]); d_pred[0]=np.nan
        d_mi   = np.diff(self.mi_hist,   prepend=self.mi_hist[:1]);   d_mi[0]=np.nan
        d_stab = np.diff(self.stab_hist, prepend=self.stab_hist[:1]); d_stab[0]=np.nan

        d_pred_s = ema(np.nan_to_num(d_pred, nan=0.0), alpha=self.ema_alpha)
        d_mi_s   = ema(np.nan_to_num(d_mi,   nan=0.0), alpha=self.ema_alpha)
        d_stab_s = ema(np.nan_to_num(d_stab, nan=0.0), alpha=self.ema_alpha)

        K_star = None
        for k in range(2, len(self.steps)+1):
            ok = True
            for j in range(k - self.knee_m + 1, k + 1):
                if j <= 1: ok=False; break
                if not (d_pred_s[j-1] < self.eps_pred and d_mi_s[j-1] < self.eps_mi and d_stab_s[j-1] < self.eps_stab):
                    ok=False; break
            if ok:
                K_star = k - 1
                break
        self.K_star = K_star

    def _fit_heads_on_state(self, Z, Y):
        fZ, fY, PZ = fit_heads_once(Z.to(self.device), Y.to(self.device),
                                    self.args.infonce_steps, self.args.infonce_batch,
                                    self.args.infonce_lr, self.args.infonce_proj,
                                    self.args.infonce_temp, self.device)
        self.heads = (fZ, fY, PZ)

    def _suggest_next(self, out):
        # Build candidate heatmap
        Afin = out["A_final"]; Hf, Wf = out["token_hw"]; Np = len(self.lbl)
        prompt_offset = out["prompt_offset"]
        prob = out["prob"]; prob_flip = out.get("prob_flip", None)

        Hmap, comps = build_candidate_score_maps(
            img=self.img, prob=prob, prob_flip=prob_flip,
            A_final=Afin, prompt_offset=prompt_offset, Np=Np,
            token_hw=(Hf,Wf), pts_used=np.array(self.pts, dtype=np.float32),
            args=self.args
        )
        cands = topk_local_maxima(Hmap, K=self.args.cand_topk, nms_radius=self.args.nms_radius)
        if len(cands)==0:
            return None, Hmap, comps, []

        # Evaluate with heads (fit if missing)
        if self.heads is None:
            self._fit_heads_on_state(out["Z"], out["Ytok"])
        fZ, fY, PZ = self.heads
        base_mi = anchored_mi_from_heads(PZ, fY, out["Ytok"].to(self.device), self.args.infonce_temp)

        best = None
        scored = []
        for (y,x,score) in cands:
            for lab in (1,0):
                pts_try = np.vstack([self.pts, np.array([[x,y]], dtype=np.float32)]) if len(self.pts) else np.array([[x,y]], dtype=np.float32)
                lbl_try = np.concatenate([self.lbl, np.array([lab], np.int32)], axis=0) if len(self.lbl) else np.array([lab], np.int32)

                mi_try, out_try = evaluate_prompt_set(self.predictor, self.sam, self.enforcer, self.img,
                                                      pts_try, lbl_try, self.w_anchor, self.args, want_flip=self.args.use_tta)
                # approximate via heads on current state
                mi_head = anchored_mi_from_heads(PZ, fY, out_try["Ytok"].to(self.device), self.args.infonce_temp)
                d_mi = mi_head - base_mi
                d_stab = out_try["stability"] - out.get("stability", 1.0)
                prev_bin = (prob >= 0.5).astype(np.uint8)
                now_bin  = (out_try["prob"] >= 0.5).astype(np.uint8)
                comp_pen = float(max(0, int(now_bin.sum()==0) - int(prev_bin.sum()==0)))

                utility = self.args.u_mi * d_mi + self.args.u_stab * d_stab - self.args.u_pen * comp_pen
                cand = {"x":int(x), "y":int(y), "label":int(lab), "score":float(score),
                        "d_mi":float(d_mi), "d_stab":float(d_stab), "pen":float(comp_pen),
                        "utility":float(utility)}
                scored.append(cand)
                if (best is None) or (utility > best["utility"]):
                    best = cand

        if (best is None) or (best["utility"] < self.args.util_min) or (best["d_mi"] < self.args.delta_mi_min):
            return None, Hmap, comps, scored
        return best, Hmap, comps, scored

    # --------- recompute ---------
    def _recompute_and_redraw(self, initial=False):
        t0 = time.time()
        pts_xy = np.array(self.pts, dtype=np.float32) if len(self.pts) else np.zeros((0,2), np.float32)
        lbl = np.array(self.lbl, dtype=np.int32) if len(self.lbl) else np.zeros((0,), np.int32)

        mi_k, out = evaluate_prompt_set(self.predictor, self.sam, self.enforcer,
                                        self.img, pts_xy, lbl, self.w_anchor, self.args, want_flip=self.args.use_tta)

        # Update histories
        k = len(self.pts)
        if k>0:
            self.steps.append(k)
            self.mi_hist.append(mi_k)
            self.pred_hist.append(out["pred_iou"])
            self.stab_hist.append(out["stability"])
            self.unc_hist.append(out["uncert_global"])
            # Fit heads on new state (for faster suggestion scoring)
            self._fit_heads_on_state(out["Z"], out["Ytok"])

        # Knee detection
        self._knee_detect()

        # Redraw panels
        self.ax_main.clear(); self.ax_heat.clear(); self.ax_curve.clear(); self.ax_unc.clear()
        self.ax_main.set_title("Image + current mask")
        self.ax_heat.set_title("Suggestion heatmap (topK + best)")
        self.ax_curve.set_title("Curves (AnchMI & PredIoU)")
        self.ax_unc.set_title("Uncertainty p(1-p)")

        # Main overlay
        ov = overlay_heatmap(self.img, out["prob"], alpha=0.55, cmap=self.args.cmap)
        self.ax_main.imshow(ov); self._render_prompts(self.ax_main)
        self.ax_main.set_axis_off()

        # Suggestion heatmap + best
        best, Hmap, comps, scored = self._suggest_next(out)
        self.ax_heat.imshow(overlay_heatmap(self.img, Hmap, alpha=0.55, cmap=self.args.cmap))
        if best is not None:
            col = 'yellow' if best["label"]==1 else 'red'
            self.ax_heat.scatter([best["x"]],[best["y"]], c=col, s=70, marker='o', edgecolor='black', linewidths=0.8)
        self._render_prompts(self.ax_heat)
        self.ax_heat.set_axis_off()
        self.suggestion = best

        # Curves
        if len(self.steps):
            self.ax_curve.plot(self.steps, self.mi_hist,  '-o', label="Anchored MI")
            self.ax_curve.plot(self.steps, self.pred_hist,'-o', label="PredIoU")
            if self.K_star is not None:
                self.ax_curve.axvline(self.K_star, color='k', linestyle='--', alpha=0.6, label=f"K*={self.K_star}")
            self.ax_curve.grid(True, alpha=0.3); self.ax_curve.legend()

        # Uncertainty
        U = out["prob"] * (1 - out["prob"])
        self.ax_unc.imshow(overlay_heatmap(self.img, U, alpha=0.55, cmap='magma'))
        self.ax_unc.set_axis_off()

        self.fig.canvas.draw_idle()
        t1 = time.time()

        # Terminal readout
        print("-"*80)
        print(f"[STEP {len(self.pts)}] AnchMI={mi_k:.4f}  PredIoU={out['pred_iou']:.4f}  "
              f"Stab={out['stability']:.4f}  Unc={out['uncert_global']:.4f}  (compute {t1-t0:.2f}s)")
        if self.gt is not None:
            iou, dice = binary_iou_dice((out["prob"]>=0.5).astype(np.uint8), self.gt)
            print(f"   [GT] IoU={iou:.4f}  Dice={dice:.4f}")
        if self.K_star is not None:
            print(f"   [KNEE] Over-prompting knee detected at K*={self.K_star}")
        if best is None:
            print("   [SUGGEST] No candidate passes thresholds (or none found).")
        else:
            print(f"   [SUGGEST] ({best['x']},{best['y']}) label={best['label']}  util={best['utility']:.4f} "
                  f"dMI={best['d_mi']:.4f} dStab={best['d_stab']:.4f}")

    # --------- events ---------
    def _on_click(self, event):
        if event.inaxes != self.ax_main:  # only accept clicks in main panel
            return
        x, y = int(round(event.xdata)), int(round(event.ydata))
        x = int(np.clip(x, 0, self.W-1)); y = int(np.clip(y, 0, self.H-1))
        self.pts.append([x,y]); self.lbl.append(self.current_label)
        print(f"[CLICK] ({x},{y}) label={'+' if self.current_label==1 else '-'}")
        self._recompute_and_redraw()

    def _on_key(self, event):
        if event.key in ('p','P'):
            self.current_label = 1
            print("[MODE] Positive")
        elif event.key in ('n','N'):
            self.current_label = 0
            print("[MODE] Negative")
        elif event.key in ('a','A'):
            if self.suggestion is None:
                print("[AUTO] No suggestion to accept.")
            else:
                s = self.suggestion
                self.pts.append([s["x"], s["y"]]); self.lbl.append(s["label"])
                print(f"[AUTO][ACCEPT] ({s['x']},{s['y']}) label={s['label']} util={s['utility']:.4f}")
                self._recompute_and_redraw()
        elif event.key in ('u','U'):
            if len(self.pts):
                rem = self.pts[-1], self.lbl[-1]
                self.pts = self.pts[:-1]; self.lbl = self.lbl[:-1]
                self.steps = self.steps[:-1]
                self.mi_hist = self.mi_hist[:-1]
                self.pred_hist = self.pred_hist[:-1]
                self.stab_hist = self.stab_hist[:-1]
                self.unc_hist = self.unc_hist[:-1]
                print(f"[UNDO] Removed last prompt {rem}")
                self._recompute_and_redraw()
            else:
                print("[UNDO] Nothing to undo.")
        elif event.key in ('s','S'):
            out = {"prompts":[{"t":i+1,"x":int(x),"y":int(y),"label":int(l)}
                              for i,((x,y),l) in enumerate(zip(self.pts, self.lbl))]}
            save_json(out, Path(self.args.out_dir) / "interactive_prompts.json")
            print(f"[SAVE] Wrote {Path(self.args.out_dir) / 'interactive_prompts.json'}")
        elif event.key in ('q','Q','escape'):
            print("[QUIT]")
            plt.close(self.fig)


# ------------- CLI -------------
def parse_args():
    ap = argparse.ArgumentParser("Interactive Active Prompting for SAM")
    ap.add_argument("--image", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--model-type", default="vit_h", choices=["vit_h","vit_l","vit_b"])
    ap.add_argument("--prompts", default=None, help="Optional initial prompts JSON: {prompts:[{t,x,y,label},...]}")
    ap.add_argument("--gt-mask", default=None, help="Optional GT mask (PNG/NPY) for reporting only")
    ap.add_argument("--gt-format", default="png_zero_is_fg",
                    choices=["png_zero_is_fg","png_nonzero_is_fg","npy_one_is_fg","npy_zero_is_fg"])
    ap.add_argument("--out-dir", default="out_interactive")

    # Hard prompt
    ap.add_argument("--r-pos", type=int, default=12)
    ap.add_argument("--r-neg", type=int, default=12)
    ap.add_argument("--conflict-policy", default="pos_wins", choices=["pos_wins","neg_wins"])

    # Anchored MI (train-per-call)
    ap.add_argument("--anchor-mode", default="boundary", choices=["boundary","uncertainty"])
    ap.add_argument("--ring-radius", type=int, default=2)
    ap.add_argument("--unc-gamma",   type=float, default=1.0)
    ap.add_argument("--infonce-steps", type=int, default=120)
    ap.add_argument("--infonce-batch", type=int, default=2048)
    ap.add_argument("--infonce-lr",    type=float, default=1e-3)
    ap.add_argument("--infonce-proj",  type=int, default=128)
    ap.add_argument("--infonce-temp",  type=float, default=0.1)

    # Suggestion map + utility
    ap.add_argument("--wU", type=float, default=1.0)
    ap.add_argument("--wD", type=float, default=0.8)
    ap.add_argument("--wC", type=float, default=0.7)
    ap.add_argument("--wB", type=float, default=0.6)
    ap.add_argument("--boundary-ring", type=int, default=3)
    ap.add_argument("--cand-topk", type=int, default=20)
    ap.add_argument("--nms-radius", type=int, default=15)
    ap.add_argument("--min-dist",   type=int, default=16)
    ap.add_argument("--u-mi",   type=float, default=1.0)
    ap.add_argument("--u-stab", type=float, default=0.6)
    ap.add_argument("--u-pen",  type=float, default=0.0)
    ap.add_argument("--util-min",     type=float, default=0.005)
    ap.add_argument("--delta-mi-min", type=float, default=0.002)

    # Knee detection thresholds
    ap.add_argument("--ema-alpha", type=float, default=0.4)
    ap.add_argument("--knee-m", type=int, default=2)
    ap.add_argument("--eps-pred", type=float, default=0.005)
    ap.add_argument("--eps-mi",   type=float, default=0.01)
    ap.add_argument("--eps-stab", type=float, default=0.01)

    ap.add_argument("--use-tta", action="store_true", help="Use hflip for stability (slower)")
    ap.add_argument("--cmap", default="jet")
    ap.add_argument("--cpu", action="store_true")
    return ap.parse_args()


if __name__ == "__main__":
    args = parse_args()
    InteractiveApp(args)
