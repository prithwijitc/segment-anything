#!/usr/bin/env python3
"""
Monte-Carlo Shapley for SAM prompts (separate J1, J2, J3)

Metrics (kept separate, no mixing):
  J1: I(Z_last ; Y)        — MI of last Two-Way image latents vs SAM soft mask
  J2: I(Z_last ; Y_gt)     — MI vs GT soft map (requires --gt-mask)
  J3: IoU (and Dice)       — SAM final mask vs GT (requires --gt-mask)

We estimate MI with either MINE, InfoNCE, or their average (--mi-mode).
We compute per-prompt Shapley values for each J via random permutations,
and separate overall effectiveness scores:
  - E_J3  = IoU(full) ∈ [0,1]
  - E_J2N = min-max normalized J2(full) over ALL states seen in permutations
  - E_J1N = min-max normalized J1(full) over ALL states seen in permutations

Usage:
  python shapley_prompts_separate.py \
    --image IMG.jpg \
    --prompts prompts.json \
    --checkpoint sam_vit_h_4b8939.pth --model-type vit_h \
    --gt-mask GT.png \
    --out-dir /tmp/shapley_sep \
    --perms 30 --mi-mode avg --mine-steps 120 --infonce-steps 120
"""

import argparse, json, csv, math, sys
from pathlib import Path
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F

from segment_anything import sam_model_registry, SamPredictor


# ---------------- Utils ----------------
def ensure_dir(p): Path(p).mkdir(parents=True, exist_ok=True)
def load_image_rgb(path): return np.asarray(Image.open(path).convert("RGB"))

def load_binary_mask(path, target_hw):
    Ht, Wt = target_hw
    m = Image.open(path).convert("L")
    if m.size != (Wt, Ht):
        m = m.resize((Wt, Ht), resample=Image.NEAREST)
    return (np.asarray(m) != 0).astype(np.uint8)

def downsample_to_tokens(map_hw, Ht, Wt):
    t = torch.as_tensor(map_hw, dtype=torch.float32)[None, None]
    out = F.interpolate(t, size=(Ht, Wt), mode="area")[0, 0]
    return out.view(-1, 1)

def iou_and_dice(pred_bin, gt_bin):
    pred = pred_bin.astype(bool); gt = gt_bin.astype(bool)
    inter = np.logical_and(pred, gt).sum()
    union = np.logical_or(pred, gt).sum()
    iou = inter / (union + 1e-8)
    dice = (2 * inter) / (pred.sum() + gt.sum() + 1e-8)
    return float(iou), float(dice)

def maybe_add_noise(x, std):
    if std <= 0: return x
    return x + std * torch.randn_like(x)

def minmax_norm(val, pool):
    pool = np.array(pool, dtype=float)
    lo, hi = np.nanmin(pool), np.nanmax(pool)
    return float((val - lo) / (hi - lo + 1e-8)) if np.isfinite(lo) and np.isfinite(hi) else 0.0


# ---------------- Hook last Two-Way keys ----------------
def hook_last_keys(two_way):
    cache = {}
    def fwd_hook(module, args, out):
        q, k = out
        cache['last_keys'] = k.detach()
    handles = []
    for blk in two_way.layers:
        handles.append(blk.register_forward_hook(fwd_hook))
    return handles, cache

def remove_hooks(handles):
    for h in handles: h.remove()


# ---------------- MINE / InfoNCE ----------------
class MINECritic(nn.Module):
    def __init__(self, x_dim, z_dim, hidden=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(x_dim + z_dim, hidden), nn.ReLU(True),
            nn.Linear(hidden, hidden), nn.ReLU(True),
            nn.Linear(hidden, 1),
        )
    def forward(self, x, z): return self.net(torch.cat([x, z], dim=-1))

@torch.no_grad()
def _shuffle_rows(z):
    idx = torch.randperm(z.shape[0], device=z.device)
    return z[idx]

def mine_lower_bound(X, Z, steps=120, batch=1024, lr=1e-3, hidden=256, seeds=1, noise_std=0.0, device="cuda"):
    N, Dx = X.shape; Dz = Z.shape[1]
    vals = []
    for s in range(seeds):
        gen = torch.Generator(device=device).manual_seed(1234 + s)
        Tnet = MINECritic(Dx, Dz, hidden=hidden).to(device)
        opt = torch.optim.Adam(Tnet.parameters(), lr=lr)
        Xn = (X - X.mean(0, True)) / (X.std(0, True) + 1e-6)
        Zn = (Z - Z.mean(0, True)) / (Z.std(0, True) + 1e-6)
        Xn, Zn = maybe_add_noise(Xn, noise_std), maybe_add_noise(Zn, noise_std)
        for _ in range(steps):
            b = min(batch, N)
            idx = torch.randint(0, N, (b,), generator=gen, device=device)
            xb, zb = Xn[idx], Zn[idx]
            z_sh = _shuffle_rows(zb)
            Tj = Tnet(xb, zb).mean()
            Tm = torch.logsumexp(Tnet(xb, z_sh), dim=0) - math.log(b)
            loss = -(T_j := Tj - Tm)
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
        with torch.no_grad():
            def chunks(A, B, cs=4096):
                outs = []
                for i in range(0, N, cs): outs.append(Tnet(A[i:i+cs], B[i:i+cs]))
                return torch.cat(outs, 0)
            Tj = chunks(Xn, Zn).mean()
            Tm_vals = []
            for i in range(0, N, 4096):
                zsh = _shuffle_rows(Zn[i:i+4096])
                Tm_vals.append(Tnet(Xn[i:i+4096], zsh))
            Tm = torch.logsumexp(torch.cat(Tm_vals, 0), dim=0) - math.log(N)
            vals.append((Tj - Tm).item())
    return float(np.mean(vals))

class Proj(nn.Module):
    def __init__(self, d, p=128):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d, p), nn.ReLU(True), nn.Linear(p, p))
    def forward(self, x): return self.net(x)

def infonce_lower_bound(X, Z, steps=120, batch=1024, lr=1e-3, proj=128, temp=0.1, seeds=1, noise_std=0.0, device="cuda"):
    N, Dx = X.shape; Dz = Z.shape[1]
    vals = []
    for s in range(seeds):
        gen = torch.Generator(device=device).manual_seed(5678 + s)
        fX, fZ = Proj(Dx, proj).to(device), Proj(Dz, proj).to(device)
        opt = torch.optim.Adam(list(fX.parameters()) + list(fZ.parameters()), lr=lr)
        crit = nn.CrossEntropyLoss()
        Xn = (X - X.mean(0, True)) / (X.std(0, True) + 1e-6)
        Zn = (Z - Z.mean(0, True)) / (Z.std(0, True) + 1e-6)
        Xn, Zn = maybe_add_noise(Xn, noise_std), maybe_add_noise(Zn, noise_std)
        for _ in range(steps):
            b = min(batch, N)
            idx = torch.randint(0, N, (b,), generator=gen, device=device)
            xb, zb = fX(Xn[idx]), fZ(Zn[idx])
            logits = (xb @ zb.t()) / temp
            targets = torch.arange(b, device=device)
            loss = crit(logits, targets)
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
        with torch.no_grad():
            PX, PZ = [], []
            for i in range(0, N, 4096):
                PX.append(fX(Xn[i:i+4096])); PZ.append(fZ(Zn[i:i+4096]))
            PX, PZ = torch.cat(PX, 0), torch.cat(PZ, 0)
            b = min(2048, N)
            idx = torch.randint(0, N, (b,), generator=gen, device=device)
            xb, zb = PX[idx], PZ[idx]
            logits = (xb @ zb.t()) / temp
            ce = nn.CrossEntropyLoss()(logits, targets=torch.arange(b, device=device))
            vals.append(float(-ce.item()))
    return float(np.mean(vals))


# ---------------- One SAM run → Z_last & Y maps ----------------
@torch.no_grad()
def run_sam_once(predictor, sam, pts_xy, lbl01, gt_bin_or_none, bin_thresh=0.5, device="cuda"):
    """
    pts_xy: np.ndarray [N,2] (can be N=0); lbl01: np.ndarray [N] in {0,1}
    Returns dict with:
      Z_last: [N_tokens, Cz] torch
      Ytok: [N_tokens,1] torch (SAM prob pooled to token grid)
      Ygt_tok: [N_tokens,1] torch or None
      IoU, Dice: floats or None
    """
    two_way = sam.mask_decoder.transformer
    handles, cache = hook_last_keys(two_way)

    # Call predictor; allow zero prompts
    try:
        masks, scores, logits = predictor.predict(
            point_coords=pts_xy, point_labels=lbl01, multimask_output=False
        )
    except Exception as e:
        # Fallback: use a tiny "null" negative at (1,1) if empty caused issues
        if pts_xy.shape[0] == 0:
            dummy = np.array([[1.0, 1.0]], dtype=np.float32)
            dummy_lbl = np.array([0], dtype=np.int32)
            masks, scores, logits = predictor.predict(
                point_coords=dummy, point_labels=dummy_lbl, multimask_output=False
            )
        else:
            remove_hooks(handles); raise

    if 'last_keys' not in cache:
        remove_hooks(handles); raise RuntimeError("Failed to capture last keys (Two-Way).")
    Z_last = cache['last_keys'][0]  # [N_tokens, Cz]
    remove_hooks(handles)

    # Postprocess to original prob map
    lowres = torch.as_tensor(logits, device=device)
    if lowres.ndim == 2: lowres = lowres[None, None, :, :]
    elif lowres.ndim == 3:
        if lowres.shape[0] == 1 and lowres.shape[1] == 256: lowres = lowres[:, None, :, :]
        elif lowres.shape[0] >= 1 and lowres.shape[1] == 256: lowres = lowres[:, None, :, :]
        else: lowres = lowres[None, None, :, :]
    elif lowres.ndim == 4 and lowres.shape[1] != 1:
        lowres = lowres[:, :1, ...]

    if hasattr(sam, "postprocess_masks"):
        prob_t = torch.sigmoid(sam.postprocess_masks(lowres, predictor.input_size, predictor.original_size)[0, 0])
    elif hasattr(predictor, "model") and hasattr(predictor.model, "postprocess_masks"):
        prob_t = torch.sigmoid(predictor.model.postprocess_masks(lowres, predictor.input_size, predictor.original_size)[0, 0])
    else:
        prob_t = torch.as_tensor(masks[0].astype(np.float32), device=device)
    prob = prob_t.detach().cpu().numpy()

    # Pool to token grid
    _, C, Hf, Wf = predictor.features.shape
    Ytok = downsample_to_tokens(prob, Hf, Wf).to(device)

    # IoU / Dice & Y_gt tokens if provided
    IoU = Dice = None; Ygt_tok = None
    if gt_bin_or_none is not None:
        pred_bin = (prob >= bin_thresh).astype(np.uint8)
        IoU, Dice = iou_and_dice(pred_bin, gt_bin_or_none)
        Ygt_tok = downsample_to_tokens(gt_bin_or_none.astype(np.float32), Hf, Wf).to(device)

    return dict(Z_last=Z_last, Ytok=Ytok, Ygt_tok=Ygt_tok, IoU=IoU, Dice=Dice)


# ---------------- Monte-Carlo Shapley ----------------
def run(args):
    device = "cuda" if torch.cuda.is_available() and not args.cpu else "cpu"
    out_dir = Path(args.out_dir); ensure_dir(out_dir)

    # Load image & prompts
    img = load_image_rgb(args.image)
    H, W = img.shape[:2]
    with open(args.prompts, "r") as f:
        prom = json.load(f)
    pos = np.array(prom.get("positive_points", []), dtype=np.float32) if prom.get("positive_points") else np.zeros((0,2), np.float32)
    neg = np.array(prom.get("negative_points", []), dtype=np.float32) if prom.get("negative_points") else np.zeros((0,2), np.float32)
    pts_all = np.concatenate([pos, neg], axis=0)
    lbl_all = np.array([1]*len(pos) + [0]*len(neg), dtype=np.int32)
    K = len(pts_all)
    if K == 0: raise ValueError("No prompts provided.")

    gt_bin = load_binary_mask(args.gt_mask, (H, W)) if args.gt_mask else None
    have_gt = gt_bin is not None

    # Build SAM once
    sam = sam_model_registry[args.model_type](checkpoint=args.checkpoint)
    sam.to(device).eval()
    predictor = SamPredictor(sam)
    predictor.set_image(img)
    if predictor.features is None: raise RuntimeError("Predictor has no encoder features.")

    # Helper: MI(X=Z, Y) with chosen estimator mode
    def mi_pair(Z, Y):
        if args.mi_mode in ("mine", "avg"):
            m = mine_lower_bound(Z, Y,
                                 steps=args.mine_steps, batch=args.mine_batch, lr=args.mine_lr,
                                 hidden=args.mine_hidden, seeds=args.seeds,
                                 noise_std=args.token_noise_std, device=device)
        else:
            m = None
        if args.mi_mode in ("infonce", "avg"):
            n = infonce_lower_bound(Z, Y,
                                    steps=args.infonce_steps, batch=args.infonce_batch, lr=args.infonce_lr,
                                    proj=args.infonce_proj, temp=args.infonce_temp, seeds=args.seeds,
                                    noise_std=args.token_noise_std, device=device)
        else:
            n = None
        if args.mi_mode == "mine": return m
        if args.mi_mode == "infonce": return n
        return 0.5 * (m + n)

    # FULL metric values (for reporting overall scores)
    full = run_sam_once(predictor, sam, pts_all, lbl_all, gt_bin, args.bin_thresh, device)
    J1_full = mi_pair(full["Z_last"], full["Ytok"])
    J2_full = mi_pair(full["Z_last"], full["Ygt_tok"]) if have_gt else None
    J3_full, Dice_full = (full["IoU"], full["Dice"]) if have_gt else (None, None)

    print(f"[FULL] J1 MI(Z;Y)={J1_full:.4f}" +
          (f"  J2 MI(Z;Y_gt)={J2_full:.4f}" if have_gt else "") +
          (f"  J3 IoU={J3_full:.4f}  Dice={Dice_full:.4f}" if have_gt else ""))

    # Shapley accumulators
    shap_J1 = np.zeros(K, dtype=float)
    shap_J2 = np.zeros(K, dtype=float) if have_gt else None
    shap_J3 = np.zeros(K, dtype=float) if have_gt else None
    shap_Dice = np.zeros(K, dtype=float) if have_gt else None

    # Pools for overall normalization (include all intermediate states)
    pool_J1 = [J1_full]
    pool_J2 = [J2_full] if have_gt else []

    rng = np.random.default_rng(args.perm_seed)

    # Monte-Carlo permutations
    for p in range(args.perms):
        order = np.arange(K)
        rng.shuffle(order)

        # Start from empty set
        S_pts = np.zeros((0,2), dtype=np.float32)
        S_lbl = np.zeros((0,), dtype=np.int32)
        base = run_sam_once(predictor, sam, S_pts, S_lbl, gt_bin, args.bin_thresh, device)

        J1_prev = mi_pair(base["Z_last"], base["Ytok"])
        pool_J1.append(J1_prev)

        if have_gt:
            J2_prev = mi_pair(base["Z_last"], base["Ygt_tok"])
            J3_prev, D_prev = base["IoU"], base["Dice"]
            pool_J2.append(J2_prev)
        else:
            J2_prev = J3_prev = D_prev = None

        # Insert prompts one by one in this permutation
        for idx in order:
            S_pts = np.vstack([S_pts, pts_all[idx:idx+1]])
            S_lbl = np.hstack([S_lbl, lbl_all[idx:idx+1]])

            out = run_sam_once(predictor, sam, S_pts, S_lbl, gt_bin, args.bin_thresh, device)

            # J1 step
            J1_curr = mi_pair(out["Z_last"], out["Ytok"])
            shap_J1[idx] += (J1_curr - J1_prev)
            J1_prev = J1_curr
            pool_J1.append(J1_curr)

            if have_gt:
                # J2 step
                J2_curr = mi_pair(out["Z_last"], out["Ygt_tok"])
                shap_J2[idx] += (J2_curr - J2_prev)
                J2_prev = J2_curr
                pool_J2.append(J2_curr)
                # J3 step (IoU) and Dice
                J3_curr, D_curr = out["IoU"], out["Dice"]
                shap_J3[idx] += (J3_curr - J3_prev)
                shap_Dice[idx] += (D_curr - D_prev)
                J3_prev, D_prev = J3_curr, D_curr

        print(f"[perm {p+1}/{args.perms}] done.")

    # Average over permutations → Shapley values
    shap_J1 /= args.perms
    if have_gt:
        shap_J2 /= args.perms
        shap_J3 /= args.perms
        shap_Dice /= args.perms

    # Overall (separate)
    E_J1N = minmax_norm(J1_full, pool_J1)
    if have_gt:
        E_J2N = minmax_norm(J2_full, pool_J2)
        E_J3  = J3_full
    else:
        E_J2N = None; E_J3 = None

    # ----- Save CSVs -----
    ensure_dir(out_dir)
    with open(out_dir/"shapley_per_prompt_separate.csv", "w", newline="") as f:
        w = csv.writer(f)
        header = ["prompt_index","label(+1/-0)","phi_J1_MI_ZY","phi_J2_MI_ZYgt","phi_J3_IoU","phi_Dice"]
        w.writerow(header)
        for i in range(K):
            kind = "+" if i < len(pos) else "-"
            w.writerow([
                i, kind,
                f"{shap_J1[i]:.6f}",
                f"{(shap_J2[i] if have_gt else float('nan')):.6f}",
                f"{(shap_J3[i] if have_gt else float('nan')):.6f}",
                f"{(shap_Dice[i] if have_gt else float('nan')):.6f}",
            ])

    with open(out_dir/"overall_effectiveness_separate.txt","w") as f:
        f.write(f"J1 full MI(Z;Y)           = {J1_full:.6f}\n")
        f.write(f"E_J1 (min-max over pool)  = {E_J1N:.6f}\n")
        if have_gt:
            f.write(f"J2 full MI(Z;Y_gt)        = {J2_full:.6f}\n")
            f.write(f"E_J2 (min-max over pool)  = {E_J2N:.6f}\n")
            f.write(f"J3 full IoU               = {J3_full:.6f}\n")
            f.write(f"full Dice                 = {Dice_full:.6f}\n")

    # ----- Plots -----
    x = np.arange(K)

    # Bar plots of Shapley values (separate)
    def bar_plot(vals, title, ylabel, fname, color):
        fig, ax = plt.subplots(1,1, figsize=(max(8,0.5*K+4), 4.6))
        ax.bar(x, vals, color=color, alpha=0.9)
        ax.axhline(0, color='k', lw=0.8, alpha=0.6)
        ax.set_xlabel("Prompt index")
        ax.set_ylabel(ylabel)
        ax.set_title(title + " (Shapley; positive = helpful)")
        ax.grid(axis='y', alpha=0.3)
        fig.tight_layout(); fig.savefig(out_dir/fname, dpi=220); plt.close(fig)

    bar_plot(shap_J1, "J1 = MI(Z_last; Y)", "ϕ_J1", "shapley_J1.png", "#1f77b4")
    if have_gt:
        bar_plot(shap_J2, "J2 = MI(Z_last; Y_gt)", "ϕ_J2", "shapley_J2.png", "#9467bd")
        bar_plot(shap_J3, "J3 = IoU", "ϕ_J3 (ΔIoU)", "shapley_J3_IoU.png", "#2ca02c")

    # Histogram where FULL sits among all intermediate MI values (optional)
    fig, ax = plt.subplots(1, 2 if have_gt else 1, figsize=(10 if have_gt else 5, 4))
    ax = np.atleast_1d(ax)
    ax[0].hist(pool_J1, bins=24, color="#1f77b4", alpha=0.7)
    ax[0].axvline(J1_full, color="k", lw=2, label="FULL")
    ax[0].set_title("Pool of J1 (FULL + all states)"); ax[0].legend(); ax[0].grid(True, alpha=0.3)
    if have_gt:
        ax[1].hist(pool_J2, bins=24, color="#9467bd", alpha=0.7)
        ax[1].axvline(J2_full, color="k", lw=2, label="FULL")
        ax[1].set_title("Pool of J2 (FULL + all states)"); ax[1].legend(); ax[1].grid(True, alpha=0.3)
    fig.tight_layout(); fig.savefig(out_dir/"mi_pools.png", dpi=220); plt.close(fig)

    print("\n[SEPARATE OVERALL EFFECTIVENESS]")
    print(f"E_J1 (0..1, min-max in pool) = {E_J1N:.4f}")
    if have_gt:
        print(f"E_J2 (0..1, min-max in pool) = {E_J2N:.4f}")
        print(f"E_J3 (IoU)                   = {E_J3:.4f}")
        print(f"(Dice full)                  = {Dice_full:.4f}")
    print(f"\nOutputs: {out_dir.resolve()}")

# ---------------- CLI ----------------
def parse_args():
    ap = argparse.ArgumentParser("Monte-Carlo Shapley for SAM prompts (separate metrics)")
    ap.add_argument("--image", required=True)
    ap.add_argument("--prompts", required=True)  # JSON: positive_points / negative_points
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--model-type", default="vit_h", choices=["vit_h","vit_l","vit_b"])
    ap.add_argument("--gt-mask", default=None)
    ap.add_argument("--bin-thresh", type=float, default=0.5)
    ap.add_argument("--out-dir", default="shapley_out")

    # Monte-Carlo
    ap.add_argument("--perms", type=int, default=30, help="Number of random permutations")
    ap.add_argument("--perm-seed", type=int, default=2024)

    # MI estimator knobs
    ap.add_argument("--mi-mode", choices=["avg","mine","infonce"], default="avg")
    ap.add_argument("--mine-steps", type=int, default=120)
    ap.add_argument("--mine-batch", type=int, default=1024)
    ap.add_argument("--mine-lr", type=float, default=1e-3)
    ap.add_argument("--mine-hidden", type=int, default=256)
    ap.add_argument("--infonce-steps", type=int, default=120)
    ap.add_argument("--infonce-batch", type=int, default=1024)
    ap.add_argument("--infonce-lr", type=float, default=1e-3)
    ap.add_argument("--infonce-proj", type=int, default=128)
    ap.add_argument("--infonce-temp", type=float, default=0.1)
    ap.add_argument("--seeds", type=int, default=1)
    ap.add_argument("--token-noise-std", type=float, default=0.0)

    ap.add_argument("--cpu", action="store_true")
    return ap.parse_args()

if __name__ == "__main__":
    args = parse_args()
    run(args)
