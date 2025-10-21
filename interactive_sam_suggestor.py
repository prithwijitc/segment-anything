#!/usr/bin/env python3
"""
Interactive SAM with full refinement loop:
  • Hard-prompt enforcement (positives can't be erased, negatives can't be included)
  • Over-prompting detection (EMA deltas, 3-proxy knee)
  • Redundancy detection (local Δ|prob|, Δunc, local MI Δ, ownership overlap)
  • Anchored InfoNCE MI (fixed anchor; GT-free or GT boundary if provided later)
  • Auto-suggestion after each step (uncertainty × boundary × novelty × spacing)
  • Robust upsampling using SAM postprocess (fixes 3D/5D interpolate crash)
  • 2×2 UI: current mask, suggestion map, ownership map, flip-diff (stability)

Controls:
  Left-click : add a prompt at cursor (uses current label)
  p / n      : switch to positive / negative for next click
  a          : accept the auto-suggested point
  u          : undo last prompt
  s          : save prompts JSON + snapshot (to --out-dir)
  q / ESC    : quit

Proxies (GT-free):
  - PredIoU (SAM score)
  - Anchored InfoNCE MI I(Z;Y) with fixed anchor weights
  - Stability (hflip TTA IoU)
  - Uncertainty mean: E[p(1-p)]

Requires:
  pip install "git+https://github.com/facebookresearch/segment-anything.git"
"""

import argparse, json, math
from pathlib import Path

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.patches import Circle, Patch

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import amp

from segment_anything import sam_model_registry, SamPredictor


# ----------------- I/O helpers -----------------
def ensure_dir(p): Path(p).mkdir(parents=True, exist_ok=True)
def load_image_rgb(path): return np.asarray(Image.open(path).convert("RGB"))
def save_json(obj, path): 
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f: json.dump(obj, f, indent=2)


# ----------------- Drawing helpers -----------------
def overlay_prob(img_rgb, prob01, alpha=0.55, cmap='jet'):
    H, W = img_rgb.shape[:2]
    if prob01.shape != (H, W):
        prob01 = F.interpolate(torch.from_numpy(prob01)[None,None].float(), size=(H,W), mode='bilinear', align_corners=False)[0,0].numpy()
    base = img_rgb.astype(np.float32) / 255.0
    col  = cm.get_cmap(cmap)(np.clip(prob01,0,1))[..., :3]
    out  = (1 - alpha) * base + alpha * col
    return np.clip(out, 0, 1)

def draw_points(ax, pts_xy, lbl, r=5):
    for (x, y), l in zip(pts_xy, lbl):
        color = 'lime' if l == 1 else 'red'
        circ = Circle((x, y), radius=r, edgecolor='black', facecolor=color, linewidth=0.7, alpha=0.9)
        ax.add_patch(circ)

def make_disk_mask(H, W, x, y, radius):
    x = int(round(x)); y = int(round(y))
    if radius <= 0: return None, None, None
    x0 = max(0, x - radius); x1 = min(W, x + radius + 1)
    y0 = max(0, y - radius); y1 = min(H, y + radius + 1)
    if x0 >= x1 or y0 >= y1: return None, None, None
    xs = np.arange(x0, x1); ys = np.arange(y0, y1)
    xxg, yyg = np.meshgrid(xs, ys)
    disk = (xxg - x) ** 2 + (yyg - y) ** 2 <= (radius * radius)
    if not disk.any(): return None, None, None
    return slice(y0, y1), slice(x0, x1), disk


# ----------------- SAM postprocess (robust) -----------------
def upsample_lowres_logits_to_original(lowres_logits, predictor, device):
    """
    Robustly convert SAM's low-res mask logits to original (H,W) probs.
    Accepts shapes: (256,256), (1,256,256), (N,256,256), (N,1,256,256).
    Uses model.postprocess_masks if available; else bilinear fallback.
    """
    low = torch.as_tensor(lowres_logits, device=device)

    # Normalize to (N,1,h,w)
    if low.ndim == 2:
        low = low[None, None, :, :]
    elif low.ndim == 3:
        low = low[:, None, :, :]
    elif low.ndim == 4:
        if low.shape[1] != 1:
            low = low[:, :1, ...]
    else:
        raise ValueError(f"Unexpected lowres logits shape: {tuple(low.shape)}")

    model = getattr(predictor, "model", None)
    if model is not None and hasattr(model, "postprocess_masks"):
        prob = torch.sigmoid(
            model.postprocess_masks(
                low,
                input_size=predictor.input_size,
                original_size=predictor.original_size,
            )[0, 0]
        )
    else:
        H, W = predictor.original_size
        prob = torch.sigmoid(F.interpolate(low, size=(H, W), mode="bilinear", align_corners=False)[0, 0])

    return prob.detach().cpu().numpy()


# ----------------- Hard-prompt enforcer -----------------
class StrictPromptEnforcer:
    """
    Enforces that:
      - positive disks remain confidently FG (>= pos_floor)
      - negative disks remain confidently BG (<= neg_ceiling)
    by clamping probabilities after SAM prediction.
    """
    def __init__(self, predictor, pos_radius=12, neg_radius=12, pos_floor=0.95, neg_ceiling=0.05):
        self.predictor = predictor
        self.pos_radius = int(pos_radius)
        self.neg_radius = int(neg_radius)
        self.pos_floor  = float(pos_floor)
        self.neg_ceiling= float(neg_ceiling)

    @torch.inference_mode()
    def strict_predict(self, pts_xy, lbl, device):
        if len(pts_xy) == 0:
            masks, scores, logits = self.predictor.predict(
                point_coords=np.zeros((0,2), np.float32),
                point_labels=np.zeros((0,), np.int32),
                multimask_output=False,
            )
        else:
            masks, scores, logits = self.predictor.predict(
                point_coords=pts_xy,
                point_labels=lbl,
                multimask_output=False,
            )

        prob = upsample_lowres_logits_to_original(logits, self.predictor, device)

        # Hard constraints
        H, W = prob.shape
        for (x, y), l in zip(pts_xy, lbl):
            r = self.pos_radius if l == 1 else self.neg_radius
            ys, xs, disk = make_disk_mask(H, W, x, y, r)
            if ys is None: continue
            if l == 1:
                prob[ys, xs][disk] = np.maximum(prob[ys, xs][disk], self.pos_floor)
            else:
                prob[ys, xs][disk] = np.minimum(prob[ys, xs][disk], self.neg_ceiling)

        bin_mask = (prob >= 0.5).astype(np.uint8)
        score = float(scores[0]) if (scores is not None and len(scores)>0) else float((prob>=0.5).mean())
        return prob, bin_mask, score, logits


# ----------------- Token utils & anchors -----------------
def area_pool_to_tokens(mask_hw_np, Ht, Wt):
    t = torch.from_numpy(mask_hw_np).float()[None, None]
    ds = F.interpolate(t, size=(Ht, Wt), mode="area")[0, 0]
    return ds.view(-1, 1).cpu().numpy()

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


# ----------------- Two-Way hooks -----------------
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
        qh = ql.view(B, Nq, H, Ch).permute(0,2,1,3)
        kh = kl.view(B, -1, H, Ch).permute(0,2,1,3)
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


# ----------------- Weighted InfoNCE MI -----------------
class ProjectionHead(nn.Module):
    def __init__(self, in_dim, proj_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, proj_dim), nn.ReLU(inplace=True),
            nn.Linear(proj_dim, proj_dim),
        )
    def forward(self, x): return self.net(x)

@torch.no_grad()
def _norm_feat(x):
    return (x - x.mean(0, keepdim=True)) / (x.std(0, keepdim=True) + 1e-6)

def infonce_weighted_lower_bound(
    Z, Y, row_weights, steps=120, batch=2048, lr=1e-3, proj=128, temp=0.1, device="cuda", use_amp=True
):
    N, Dz = Z.shape
    Dy = Y.shape[1]
    fZ = ProjectionHead(Dz, proj_dim=proj).to(device)
    fY = ProjectionHead(Dy, proj_dim=proj).to(device)
    opt = torch.optim.Adam(list(fZ.parameters()) + list(fY.parameters()), lr=lr)

    Zn = _norm_feat(Z); Yn = _norm_feat(Y)
    rw = row_weights.clamp(min=0); rw = rw / (rw.sum() + 1e-12)

    scaler = amp.GradScaler('cuda', enabled=use_amp and (device!='cpu'))

    for _ in range(steps):
        b = min(batch, N)
        idx = torch.multinomial(rw, num_samples=b, replacement=True)
        with amp.autocast('cuda', enabled=use_amp and (device!='cpu')):
            z_b = fZ(Zn[idx]); y_b = fY(Yn[idx])
            logits  = (z_b @ y_b.t()) / temp
            targets = torch.arange(b, device=device)
            loss_per = F.cross_entropy(logits, targets, reduction='none')
            rwb = rw[idx]; rwb = rwb / (rwb.sum() + 1e-12)
            loss = (rwb * loss_per).sum()
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).step(opt)
        scaler.update()

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


# ----------------- Suggestion policy -----------------
def distinct_colors(n):
    cmap = plt.get_cmap('tab20' if n>10 else 'tab10')
    return [tuple(cmap(i)[:3]) for i in range(n)]

def suggest_next_point_full(
    prob, owner_tok_labels, token_hw, pts_xy, avoid_radius,
    wU=1.0, wB=0.5, wC=1.0, wD=0.4
):
    """
    Score = wU * uncertainty  + wB * boundary  + wC * novelty  + wD * spacing
    novelty punishes pixels whose *token* is already owned by any prompt.
    spacing prefers distance from existing clicks.
    """
    H, W = prob.shape
    Ht, Wt = token_hw
    U = prob * (1 - prob)

    # boundary strength |grad p|
    gy, gx = np.gradient(prob)
    B = np.sqrt(gx*gx + gy*gy)
    B /= (B.max() + 1e-8)

    # novelty from token ownership
    owned_tok = (owner_tok_labels >= 0).astype(np.uint8).reshape(Ht, Wt)
    owned_up  = F.interpolate(torch.from_numpy(owned_tok)[None,None].float(), size=(H,W), mode='nearest')[0,0].numpy()
    C = 1.0 - owned_up  # prefer unowned tokens

    # spacing from existing points
    if len(pts_xy) == 0:
        D = np.ones_like(prob, dtype=np.float32)
    else:
        grid_y, grid_x = np.mgrid[0:H, 0:W]
        dist = np.ones_like(prob, dtype=np.float32) * 1e6
        for (x, y) in pts_xy:
            dist = np.minimum(dist, (grid_x - x)**2 + (grid_y - y)**2)
        dist = np.sqrt(dist)
        D = dist / (dist.max() + 1e-6)

    # avoid radius hard mask
    avoid = np.zeros((H, W), dtype=np.uint8)
    for (x, y) in pts_xy:
        ys, xs, disk = make_disk_mask(H, W, x, y, avoid_radius)
        if ys is not None: avoid[ys, xs][disk] = 1

    score = wU*U + wB*B + wC*C + wD*D
    score *= (1 - avoid)

    idx = int(np.argmax(score))
    y, x = divmod(idx, W)
    s_lab = 1 if prob[y, x] >= 0.5 else 0
    return (float(x), float(y)), s_lab, score, U


# ----------------- EMA -----------------
def ema(arr, alpha=0.4):
    out = []
    m = None
    for v in arr:
        m = v if m is None else alpha*v + (1-alpha)*m
        out.append(m)
    return np.array(out, dtype=np.float32)


# ----------------- App -----------------
class InteractiveApp:
    def __init__(self, args):
        self.args = args
        self.device = "cuda" if (torch.cuda.is_available() and not args.cpu) else "cpu"

        # Load
        self.img = load_image_rgb(args.image)
        self.H, self.W = self.img.shape[:2]

        # SAM
        self.sam = sam_model_registry[args.model_type](checkpoint=args.checkpoint)
        self.sam.to(self.device).eval()
        self.predictor = SamPredictor(self.sam)
        self.predictor.set_image(self.img)

        # Prompt state
        self.pts_xy, self.lbl = self._load_prompts(args.prompts)
        self.curr_label = 1

        # Enforcer
        self.enforcer = StrictPromptEnforcer(
            self.predictor,
            pos_radius=args.r_pos, neg_radius=args.r_neg,
            pos_floor=args.pos_floor, neg_ceiling=args.neg_ceiling
        )

        # Metrics history
        self.steps = []
        self.pred_iou_hist, self.mi_hist, self.stab_hist, self.unc_hist = [], [], [], []
        self.pred_iou_ema,  self.mi_ema,  self.stab_ema = [], [], []
        self.last_prob = None
        self.owner_labels = -np.ones((1,), dtype=np.int32)  # placeholder

        # Anchor weights (fixed once)
        self.anchor_note = None
        self.w_ref = None  # torch vector on device
        self.token_hw = None

        # UI
        self.fig, self.axes = plt.subplots(2, 2, figsize=(14, 10))
        self.ax_mask, self.ax_sug = self.axes[0,0], self.axes[0,1]
        self.ax_owner, self.ax_flip = self.axes[1,0], self.axes[1,1]
        self.fig.canvas.mpl_connect('button_press_event', self.on_click)
        self.fig.canvas.mpl_connect('key_press_event', self.on_key)

        ensure_dir(args.out_dir)
        self.step = 0
        self.recompute_and_redraw(initial=True)
        plt.show()

    def _load_prompts(self, path):
        pts, lbl = [], []
        if not path: 
            return np.zeros((0,2), np.float32), np.zeros((0,), np.int32)
        with open(path, "r") as f:
            prom = json.load(f)
        if "prompts" in prom:
            prom_list = sorted(prom["prompts"], key=lambda d: d["t"])
            for p in prom_list:
                pts.append([float(p["x"]), float(p["y"])])
                lbl.append(int(p["label"]))
        else:
            pos = prom.get("positive_points", [])
            neg = prom.get("negative_points", [])
            for x,y in pos: pts.append([float(x),float(y)]); lbl.append(1)
            for x,y in neg: pts.append([float(x),float(y)]); lbl.append(0)
        return np.array(pts, dtype=np.float32), np.array(lbl, dtype=np.int32)

    def on_click(self, event):
        if event.inaxes != self.ax_mask:  # only left panel
            return
        if event.xdata is None or event.ydata is None:
            return
        x, y = float(event.xdata), float(event.ydata)
        x = float(np.clip(x, 0, self.W - 1)); y = float(np.clip(y, 0, self.H - 1))
        self.add_prompt(x, y, self.curr_label)

    def on_key(self, event):
        k = (event.key or "").lower()
        if k == "p":
            self.curr_label = 1; print("[MODE] Next click = POSITIVE (+)")
        elif k == "n":
            self.curr_label = 0; print("[MODE] Next click = NEGATIVE (-)")
        elif k == "a":
            if hasattr(self, "suggested") and self.suggested is not None:
                (x, y), lab = self.suggested
                self.add_prompt(x, y, lab)
            else:
                print("[AUTO] No suggestion available.")
        elif k == "u":
            self.undo()
        elif k == "s":
            self.save_outputs()
        elif k in ("q", "escape"):
            plt.close(self.fig)

    def add_prompt(self, x, y, label):
        self.pts_xy = np.vstack([self.pts_xy, [x, y]])
        self.lbl    = np.concatenate([self.lbl, [int(label)]])
        print(f"[ADD] ({'+' if label==1 else '-'}) at ({int(x)}, {int(y)})")
        self.recompute_and_redraw()

    def undo(self):
        if len(self.pts_xy) == 0:
            print("[UNDO] Nothing to undo."); return
        self.pts_xy = self.pts_xy[:-1]
        self.lbl    = self.lbl[:-1]
        print("[UNDO] Removed last prompt.")
        self.recompute_and_redraw()

    @torch.inference_mode()
    def evaluate_state(self):
        """
        Run SAM with hard prompts; compute proxies, stability, ownership, Z/Y tokens, anchored MI.
        """
        device = self.device
        prob, bin_mask, score, logits = self.enforcer.strict_predict(self.pts_xy, self.lbl, device)
        pred_iou = score

        # Stability: hflip TTA
        img_flip = np.ascontiguousarray(np.fliplr(self.img))
        self.predictor.set_image(img_flip)
        pts_flip = self.pts_xy.copy()
        if len(pts_flip):
            pts_flip[:,0] = (self.W - 1) - pts_flip[:,0]
        masks_f, scores_f, logits_f = self.predictor.predict(
            point_coords=pts_flip if len(pts_flip) else np.zeros((0,2), np.float32),
            point_labels=self.lbl   if len(pts_flip) else np.zeros((0,), np.int32),
            multimask_output=False,
        )
        prob_f = upsample_lowres_logits_to_original(logits_f, self.predictor, device)
        self.predictor.set_image(self.img)  # restore

        flip_back = np.fliplr(prob_f)
        inter = np.logical_and(prob>=0.5, flip_back>=0.5).sum()
        unio  = np.logical_or (prob>=0.5, flip_back>=0.5).sum()
        stability = float(inter / (unio + 1e-8))

        # Features & token grid
        feats = self.predictor.features
        if feats is None:
            raise RuntimeError("Predictor has no encoder features; set_image failed?")
        _, Cfeat, Ht, Wt = feats.shape
        self.token_hw = (Ht, Wt)
        Ni = Ht * Wt

        # Hooked internals: final attn & keys
        two_way = self.sam.mask_decoder.transformer
        handles, cache = attach_two_way_hooks(two_way)
        # re-run once quickly to fill hooks (no change, fast)
        _m, _s, _l = self.predictor.predict(
            point_coords=self.pts_xy if len(self.pts_xy) else np.zeros((0,2), np.float32),
            point_labels=self.lbl    if len(self.pts_xy) else np.zeros((0,), np.int32),
            multimask_output=False,
        )
        remove_hooks(handles)

        Afin = cache.get("final_t2i_attn", None)   # [B,H,Nq,Ni]
        last_k = cache.get("last_keys", None)       # [B,Ni,Ck]
        if Afin is None or last_k is None:
            raise RuntimeError("Failed to capture final attention/keys (SAM version mismatch?).")

        A = Afin.mean(dim=1)  # [B,Nq,Ni]
        num_mask_tokens = getattr(self.sam.mask_decoder, "num_mask_tokens", 3)
        prompt_offset = 1 + num_mask_tokens
        Np = len(self.lbl)
        if Np>0:
            pr_idx = torch.arange(prompt_offset, prompt_offset+Np, device=A.device)
            A_pr = A[:, pr_idx, :][0]      # [Np, Ni]
            owner = torch.argmax(A_pr, dim=0).detach().cpu().numpy().astype(np.int32)  # [Ni]
        else:
            A_pr = torch.zeros((0, Ni), device=A.device)
            owner = -np.ones((Ni,), dtype=np.int32)
        self.owner_labels = owner

        # Build Z, Y tokens
        K_last = last_k[0]  # [Ni, Ck]
        Z = torch.cat([K_last, A_pr.transpose(0,1)], dim=1).to(device)  # [Ni, Ck+Np]
        Ytok = torch.from_numpy(area_pool_to_tokens(prob, Ht, Wt)).float().to(device)  # [Ni,1]

        # Anchor weights (fixed once)
        if self.w_ref is None:
            prob0 = prob if len(self.pts_xy)==0 else self._step0_prob()
            prob0_tok = area_pool_to_tokens(prob0, Ht, Wt).reshape(Ht, Wt)
            if self.args.anchor_mode == "boundary":
                w = boundary_ring_weights_from_binary((prob0_tok >= 0.5).astype(np.float32), radius=self.args.ring_radius)
                self.anchor_note = "Step-0 boundary ring"
            else:
                w = uncertainty_weights_from_prob_tok(prob0_tok, gamma=self.args.unc_gamma)
                self.anchor_note = "Step-0 uncertainty"
            self.w_ref = w.to(device)

        # Anchored InfoNCE MI
        mi = infonce_weighted_lower_bound(
            Z, Ytok, row_weights=self.w_ref,
            steps=self.args.infonce_steps, batch=self.args.infonce_batch, lr=self.args.infonce_lr,
            proj=self.args.infonce_proj, temp=self.args.infonce_temp,
            device=device, use_amp=(not self.args.no_amp)
        )

        # Uncertainty mean
        U = prob * (1 - prob)
        U_mean = float(U.mean())

        # Suggestion (novelty via owner)
        (sx, sy), s_lab, score_map, U_map = suggest_next_point_full(
            prob, owner, self.token_hw, self.pts_xy, avoid_radius=self.args.suggest_avoid_r,
            wU=self.args.wU, wB=self.args.wB, wC=self.args.wC, wD=self.args.wD
        )
        self.suggested = ((sx, sy), s_lab)

        return {
            "prob": prob, "bin": (prob>=0.5).astype(np.uint8), "pred_iou": float(pred_iou),
            "stability": float(stability), "unc_mean": U_mean,
            "owner": owner.reshape(Ht, Wt), "score_map": score_map, "U_map": U_map,
            "Z": Z.detach(), "Ytok": Ytok.detach()
        }

    def _step0_prob(self):
        """Run with zero prompts once to define the anchor reference."""
        device = self.device
        masks0, scores0, logits0 = self.predictor.predict(
            point_coords=np.zeros((0,2), np.float32),
            point_labels=np.zeros((0,), np.int32),
            multimask_output=False,
        )
        return upsample_lowres_logits_to_original(logits0, self.predictor, device)

    def recompute_and_redraw(self, initial=False):
        stuff = self.evaluate_state()
        prob = stuff["prob"]; owner = stuff["owner"]; score_map = stuff["score_map"]

        # Update histories
        self.steps.append(len(self.pts_xy))
        self.pred_iou_hist.append(stuff["pred_iou"])
        self.mi_hist.append(stuff["pred_iou"]*0 + 0)  # placeholder if you want separate MI curve image; we print MI in console below
        self.stab_hist.append(stuff["stability"])
        self.unc_hist.append(stuff["unc_mean"])

        # Deltas/EMA for over-prompting flag (PredIoU & stability; MI printed)
        if len(self.pred_iou_hist) >= 2:
            d_pred = self.pred_iou_hist[-1] - self.pred_iou_hist[-2]
            d_stab = self.stab_hist[-1]     - self.stab_hist[-2]
            d_unc  = self.unc_hist[-2] - self.unc_hist[-1]
        else:
            d_pred = d_stab = d_unc = float('nan')

        # Left-top: current mask overlay (with points + suggestion ring)
        self.ax_mask.clear()
        self.ax_mask.imshow(overlay_prob(self.img, prob, alpha=self.args.alpha))
        self.ax_mask.set_title(f"Mask (PredIoU proxy={stuff['pred_iou']:.3f})")
        draw_points(self.ax_mask, self.pts_xy, self.lbl, r=5)
        (sx, sy), s_lab = self.suggested
        col = 'lime' if s_lab == 1 else 'red'
        self.ax_mask.add_patch(Circle((sx, sy), radius=7, edgecolor='black', facecolor=col, alpha=0.7, linewidth=1.0))
        self.ax_mask.set_axis_off()

        # Right-top: suggestion score heatmap
        self.ax_sug.clear()
        self.ax_sug.imshow(score_map, cmap='magma')
        self.ax_sug.add_patch(Circle((sx, sy), radius=7, edgecolor='white', facecolor='none', linewidth=1.2))
        draw_points(self.ax_sug, self.pts_xy, self.lbl, r=4)
        self.ax_sug.set_title("Suggestion score: U×boundary×novelty×spacing")
        self.ax_sug.set_axis_off()

        # Left-bottom: ownership map (token argmax by prompt)
        self.ax_owner.clear()
        colors = distinct_colors(max(1, len(self.pts_xy)))
        owned_rgb = np.zeros((self.H, self.W, 3), dtype=np.float32)
        own_up = F.interpolate(torch.from_numpy(owner.astype(np.int64))[None,None].float(), size=(self.H,self.W), mode='nearest')[0,0].numpy().astype(np.int32)
        for i in range(len(self.pts_xy)):
            m = (own_up == i)[..., None]
            owned_rgb = np.where(m, np.array(colors[i])[None,None,:], owned_rgb)
        self.ax_owner.imshow((0.6*self.img.astype(np.float32)/255.0 + 0.4*owned_rgb))
        draw_points(self.ax_owner, self.pts_xy, self.lbl, r=4)
        self.ax_owner.set_title("Token ownership by prompts (final t2i argmax)")
        self.ax_owner.set_axis_off()

        # Right-bottom: flip diff |p - flip(p)|
        self.ax_flip.clear()
        # recompute once (cheap)
        img_flip = np.ascontiguousarray(np.fliplr(self.img))
        self.predictor.set_image(img_flip)
        pts_flip = self.pts_xy.copy()
        if len(pts_flip): pts_flip[:,0] = (self.W - 1) - pts_flip[:,0]
        mf, sf, lf = self.predictor.predict(
            point_coords=pts_flip if len(pts_flip) else np.zeros((0,2), np.float32),
            point_labels=self.lbl    if len(pts_flip) else np.zeros((0,), np.int32),
            multimask_output=False,
        )
        prob_f = upsample_lowres_logits_to_original(lf, self.predictor, self.device)
        self.predictor.set_image(self.img)
        diff = np.abs(prob - np.fliplr(prob_f))
        self.ax_flip.imshow(diff, cmap='inferno')
        self.ax_flip.set_title("Flip-diff |p - flip(p)| (stability proxy)")
        self.ax_flip.set_axis_off()

        self.fig.tight_layout()
        self.fig.canvas.draw_idle()

        # Console prints: MI, deltas, flags, suggestion
        print(f"[STEP {len(self.pts_xy)}] PredIoU={stuff['pred_iou']:.4f}  Stability={stuff['stability']:.4f}  Unc={stuff['unc_mean']:.4f}")
        if len(self.pred_iou_hist) >= 2:
            print(f"  ΔPredIoU={d_pred:+.5f}  ΔStab={d_stab:+.5f}  ΔUnc={d_unc:+.5f}")
        # quick anchored MI print (re-use last Z/Y with small eval to keep UI snappy)
        mi_quick = infonce_weighted_lower_bound(
            stuff["Z"], stuff["Ytok"], self.w_ref, steps=min(40, self.args.infonce_steps),
            batch=self.args.infonce_batch, lr=self.args.infonce_lr, proj=self.args.infonce_proj,
            temp=self.args.infonce_temp, device=self.device, use_amp=(not self.args.no_amp)
        )
        print(f"  Anchored MI (quick eval) I(Z;Y) ≈ {mi_quick:.4f}   [anchor={self.anchor_note}]")

        # Redundancy for last prompt
        redundant = False
        if self.last_prob is not None and len(self.pts_xy) > 0:
            x, y = self.pts_xy[-1]
            ys, xs, disk = make_disk_mask(self.H, self.W, x, y, self.args.local_radius)
            if ys is not None:
                prev = self.last_prob[ys, xs][disk]
                now  = prob[ys, xs][disk]
                d_abs = float(np.mean(np.abs(now - prev)))
                d_unc = float(np.mean(prev*(1-prev)) - np.mean(now*(1-now)))
                # local MI: small eval on token neighborhood
                ti = int(round(y * (self.token_hw[0]/self.H)))
                tj = int(round(x * (self.token_hw[1]/self.W)))
                i0, i1 = max(0, ti-self.args.token_radius), min(self.token_hw[0], ti+self.args.token_radius+1)
                j0, j1 = max(0, tj-self.args.token_radius), min(self.token_hw[1], tj+self.args.token_radius+1)
                loc_mask = np.zeros(self.token_hw, dtype=np.uint8); loc_mask[i0:i1, j0:j1]=1
                w_loc = torch.from_numpy(loc_mask.reshape(-1).astype(np.float32)).to(self.device)
                Z_now, Y_now = stuff["Z"], stuff["Ytok"]
                mi_loc = infonce_weighted_lower_bound(
                    Z_now, Y_now, w_loc, steps=min(40, self.args.infonce_steps),
                    batch=min(1024, self.args.infonce_batch), lr=self.args.infonce_lr, proj=self.args.infonce_proj,
                    temp=self.args.infonce_temp, device=self.device, use_amp=(not self.args.no_amp)
                )
                redundant = (d_abs < self.args.eps_change) and (d_unc < self.args.eps_unc_drop) and (mi_loc < self.args.eps_mi_local)
                print(f"  Local Δ|prob|={d_abs:.5f}  Δunc={d_unc:.5f}  local MI={mi_loc:.5f}  -> redundant? {redundant}")
        over = False
        if len(self.pred_iou_hist) >= 2:
            over = (abs(d_pred) < self.args.eps_pred) and (abs(d_unc) < self.args.eps_unc_glob) and (abs(d_stab) < self.args.eps_stab)
            print(f"  Over-prompting? {over}")

        # Save flags for next iteration & accept suggestion hint
        self.last_prob = prob.copy()
        (sx, sy), s_lab = self.suggested
        print(f"[SUGGEST] ({'+' if s_lab==1 else '-'}) at ({int(sx)}, {int(sy)})")

        self.step += 1
        if initial and len(self.pts_xy):
            print(f"[INIT] Loaded {len(self.pts_xy)} prompts.")

    def save_outputs(self):
        prom = [{"t": i+1, "x": float(x), "y": float(y), "label": int(l)}
                for i, ((x, y), l) in enumerate(zip(self.pts_xy, self.lbl))]
        save_json({"prompts": prom}, Path(self.args.out_dir) / "prompts.json")
        snap_path = Path(self.args.out_dir) / f"snap_step_{self.step:03d}.png"
        self.fig.savefig(snap_path, dpi=160)
        print(f"[SAVE] Wrote prompts.json and {snap_path}")


# ----------------- CLI -----------------
def parse_args():
    ap = argparse.ArgumentParser("Interactive SAM (hard prompts + full refinement + auto-suggest)")
    ap.add_argument("--image", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--model-type", default="vit_h", choices=["vit_h", "vit_l", "vit_b"])
    ap.add_argument("--prompts", default=None, help="Optional JSON: {'prompts':[{'t':1,'x':..,'y':..,'label':0/1}, ...]}")
    ap.add_argument("--out-dir", default="runs/interactive_full")

    # Hard prompt radii & clamp thresholds
    ap.add_argument("--r-pos", type=int, default=12)
    ap.add_argument("--r-neg", type=int, default=12)
    ap.add_argument("--pos-floor", type=float, default=0.95)
    ap.add_argument("--neg-ceiling", type=float, default=0.05)

    # Anchored MI
    ap.add_argument("--anchor-mode", default="boundary", choices=["boundary","uncertainty"])
    ap.add_argument("--ring-radius", type=int, default=2)
    ap.add_argument("--unc-gamma",   type=float, default=1.0)
    ap.add_argument("--infonce-steps", type=int, default=120)
    ap.add_argument("--infonce-batch", type=int, default=1024)
    ap.add_argument("--infonce-lr",    type=float, default=1e-3)
    ap.add_argument("--infonce-proj",  type=int, default=128)
    ap.add_argument("--infonce-temp",  type=float, default=0.1)
    ap.add_argument("--no-amp", action="store_true")

    # Over-prompting thresholds
    ap.add_argument("--eps-pred", type=float, default=0.002)
    ap.add_argument("--eps-stab", type=float, default=0.005)
    ap.add_argument("--eps-unc-glob", type=float, default=0.001)

    # Redundancy thresholds (local)
    ap.add_argument("--local-radius", type=int, default=16)  # px
    ap.add_argument("--token-radius", type=int, default=2)   # tokens around prompt
    ap.add_argument("--eps-change",   type=float, default=0.002) # mean |Δprob| inside disk
    ap.add_argument("--eps-unc-drop", type=float, default=0.002) # mean Δ(p(1-p)) inside disk
    ap.add_argument("--eps-mi-local", type=float, default=0.005) # local MI lower-bound threshold

    # Suggestion scoring weights & spacing
    ap.add_argument("--suggest-avoid-r", type=int, default=12)
    ap.add_argument("--wU", type=float, default=1.0)  # uncertainty
    ap.add_argument("--wB", type=float, default=0.5)  # boundary
    ap.add_argument("--wC", type=float, default=1.0)  # novelty
    ap.add_argument("--wD", type=float, default=0.4)  # spacing

    ap.add_argument("--alpha", type=float, default=0.60)  # overlay alpha
    ap.add_argument("--cpu", action="store_true")
    return ap.parse_args()


if __name__ == "__main__":
    args = parse_args()
    InteractiveApp(args)
