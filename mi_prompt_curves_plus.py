#!/usr/bin/env python3
"""
Mutual Information vs Prompts in SAM (robust, visual, two estimators)

We estimate, after each incremental prompt k=1..N:
  1) I(X ; Z_l): input image tokens X vs Two-Way image-side latents (keys) Z_l, per block l
  2) I(Z_l ; Y): latents Z_l vs output mask Y (soft prob, pooled to token grid), per block l

Estimators:
  - MINE (Donsker–Varadhan) lower bound
  - InfoNCE (contrastive) lower bound with learned projections & temperature

We address caveats:
  - Tokens-as-samples (single image) with consistent tokenization across steps
  - Feature normalization for MINE; stable training loops; multi-seed averaging
  - Soft masks (logits->prob) + area pooling to tokens for Z↔Y MI
  - Optional tiny Gaussian token noise (off by default) to reduce degeneracy
  - Save curves (last block), heatmaps (steps×blocks), delta curves, CSVs for BOTH estimators

Usage:
  python mi_prompt_curves_plus.py \
    --image /path/to/image.jpg \
    --prompts /path/to/prompts.json \
    --checkpoint /path/to/sam_vit_h_4b8939.pth \
    --model-type vit_h \
    --out-dir /tmp/mi_prompts \
    --mine-steps 200 --infonce-steps 200 \
    --seeds 3
"""

import argparse
from pathlib import Path
import json
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F

# pip install 'git+https://github.com/facebookresearch/segment-anything.git'
from segment_anything import sam_model_registry, SamPredictor


# ----------------- IO & utils -----------------
def ensure_dir(p):
    Path(p).mkdir(parents=True, exist_ok=True)

def load_image_rgb(path):
    return np.asarray(Image.open(path).convert("RGB"))

def to_torch(x, device):
    return torch.as_tensor(x, dtype=torch.float32, device=device)

def flatten_tokens(feats_bchw):
    # feats: [1, C, Hf, Wf] -> [N_tokens, C]
    _, C, H, W = feats_bchw.shape
    return feats_bchw[0].permute(1, 2, 0).reshape(H * W, C).contiguous()

def downsample_to_tokens(mask_hw_np, Ht, Wt):
    """Area pool a (H,W) float map to token grid (Ht,Wt) and flatten to [N,1]."""
    t = torch.from_numpy(mask_hw_np).float()[None, None]
    ds = F.interpolate(t, size=(Ht, Wt), mode="area")[0, 0]
    return ds.view(-1, 1).cpu().numpy()

def maybe_add_noise(tensor, std):
    if std <= 0:
        return tensor
    return tensor + std * torch.randn_like(tensor)


# ----------------- Hook Two-Way keys per block -----------------
def hook_keys(two_way_transformer):
    """
    Forward-hook each TwoWayAttentionBlock to capture its output "keys"
    (image-side tokens after that block). Returns (handles, cache).
    """
    cache = {"keys_per_block": []}
    handles = []

    def fwd_hook(module, args, out):
        # block.forward returns (queries, keys)
        q, k = out
        cache["keys_per_block"].append(k.detach())

    for blk in two_way_transformer.layers:
        handles.append(blk.register_forward_hook(fwd_hook))
    return handles, cache

def remove_hooks(handles):
    for h in handles:
        h.remove()


# ----------------- MINE (Donsker–Varadhan) -----------------
class MINECritic(nn.Module):
    def __init__(self, x_dim, z_dim, hidden=256):
        super().__init__()
        in_dim = x_dim + z_dim
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(inplace=True),
            nn.Linear(hidden, hidden), nn.ReLU(inplace=True),
            nn.Linear(hidden, 1),
        )

    def forward(self, x, z):
        return self.net(torch.cat([x, z], dim=-1))


@torch.no_grad()
def _shuffle_rows(z):
    idx = torch.randperm(z.shape[0], device=z.device)
    return z[idx]


def estimate_mine_lower_bound(
    X, Z, steps=200, batch=1024, lr=1e-3, hidden=256, seeds=1, token_noise_std=0.0, verbose=False
):
    """
    Average MINE lower bound over multiple seeds for stability.
    X: [N, Dx], Z: [N, Dz]  (tokens as samples)
    returns: float (avg over seeds)
    """
    vals = []
    N, Dx = X.shape
    Dz = Z.shape[1]

    for s in range(seeds):
        g = torch.Generator(device=X.device)
        g.manual_seed(1234 + s)

        Tnet = MINECritic(Dx, Dz, hidden=hidden).to(X.device)
        opt = torch.optim.Adam(Tnet.parameters(), lr=lr)

        # Normalize each view; optional small noise
        Xn = (X - X.mean(0, keepdim=True)) / (X.std(0, keepdim=True) + 1e-6)
        Zn = (Z - Z.mean(0, keepdim=True)) / (Z.std(0, keepdim=True) + 1e-6)
        Xn = maybe_add_noise(Xn, token_noise_std)
        Zn = maybe_add_noise(Zn, token_noise_std)

        for t in range(steps):
            bsz = min(batch, N)
            idx = torch.randint(0, N, (bsz,), generator=g, device=X.device)
            x_b = Xn[idx]
            z_b = Zn[idx]
            z_shuf = _shuffle_rows(z_b)

            T_joint = Tnet(x_b, z_b).mean()
            # log E[exp(T)] via logsumexp
            T_marg = torch.logsumexp(Tnet(x_b, z_shuf), dim=0) - np.log(bsz)

            loss = -(T_joint - T_marg)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

        # Evaluate on full set approx
        with torch.no_grad():
            # chunk to avoid OOM
            def chunked_eval(A, B, chunk=4096):
                outs = []
                for i in range(0, N, chunk):
                    j = min(N, i + chunk)
                    outs.append(Tnet(A[i:j], B[i:j]))
                return torch.cat(outs, 0)

            Tj = chunked_eval(Xn, Zn).mean()
            # reuse negatives by shuffling per-chunk
            Tm_vals = []
            for i in range(0, N, 4096):
                j = min(N, i + 4096)
                zsh = _shuffle_rows(Zn[i:j])
                Tm_vals.append(Tnet(Xn[i:j], zsh))
            Tm = torch.logsumexp(torch.cat(Tm_vals, 0), dim=0) - np.log(N)
            lb = (Tj - Tm).item()
        vals.append(lb)

    return float(np.mean(vals))


# ----------------- InfoNCE (contrastive) -----------------
class ProjectionHead(nn.Module):
    def __init__(self, in_dim, proj_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, proj_dim),
            nn.ReLU(inplace=True),
            nn.Linear(proj_dim, proj_dim),
        )
    def forward(self, x):
        return self.net(x)

def estimate_infonce_lower_bound(
    X, Z, steps=200, batch=1024, lr=1e-3, proj_dim=128, temperature=0.1, seeds=1, token_noise_std=0.0, verbose=False
):
    """
    Train projections f,g to maximize InfoNCE on aligned pairs (tokens as samples).
    Returns the average (over seeds) of -CE loss (nats), i.e., the InfoNCE lower bound.
    """
    vals = []
    N, Dx = X.shape
    Dz = Z.shape[1]

    for s in range(seeds):
        g = torch.Generator(device=X.device)
        g.manual_seed(5678 + s)

        fX = ProjectionHead(Dx, proj_dim=proj_dim).to(X.device)
        fZ = ProjectionHead(Dz, proj_dim=proj_dim).to(X.device)
        opt = torch.optim.Adam(list(fX.parameters()) + list(fZ.parameters()), lr=lr)

        Xn = (X - X.mean(0, keepdim=True)) / (X.std(0, keepdim=True) + 1e-6)
        Zn = (Z - Z.mean(0, keepdim=True)) / (Z.std(0, keepdim=True) + 1e-6)
        Xn = maybe_add_noise(Xn, token_noise_std)
        Zn = maybe_add_noise(Zn, token_noise_std)

        criterion = nn.CrossEntropyLoss()

        for t in range(steps):
            bsz = min(batch, N)
            idx = torch.randint(0, N, (bsz,), generator=g, device=X.device)
            x_b = fX(Xn[idx])            # [B, Dp]
            z_b = fZ(Zn[idx])            # [B, Dp]

            # cosine or dot similarity; we use dot and temperature
            logits = (x_b @ z_b.t()) / temperature   # [B, B]
            targets = torch.arange(bsz, device=X.device)
            loss = criterion(logits, targets)        # nats (natural log)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

        # Evaluate bound (average log-softmax diag) on a subset/full set
        with torch.no_grad():
            projX = []
            projZ = []
            for i in range(0, N, 4096):
                j = min(N, i + 4096)
                projX.append(fX(Xn[i:j]))
                projZ.append(fZ(Zn[i:j]))
            PX = torch.cat(projX, 0)
            PZ = torch.cat(projZ, 0)

            bsz = min(2048, N)
            idx = torch.randint(0, N, (bsz,), generator=g, device=X.device)
            x_b = PX[idx]
            z_b = PZ[idx]
            logits = (x_b @ z_b.t()) / temperature
            targets = torch.arange(bsz, device=X.device)
            ce = nn.CrossEntropyLoss()(logits, targets)
            lb = (-ce).item()
        vals.append(lb)

    return float(np.mean(vals))


# ----------------- Main pipeline -----------------
def run(args):
    device = "cuda" if torch.cuda.is_available() and not args.cpu else "cpu"
    out_dir = Path(args.out_dir); ensure_dir(out_dir)

    # Load image + prompts
    img = load_image_rgb(args.image)
    H, W = img.shape[:2]
    with open(args.prompts, "r") as f:
        prom = json.load(f)
    pos = np.array(prom.get("positive_points", []), dtype=np.float32) if prom.get("positive_points") else np.zeros((0,2), np.float32)
    neg = np.array(prom.get("negative_points", []), dtype=np.float32) if prom.get("negative_points") else np.zeros((0,2), np.float32)
    pts = np.concatenate([pos, neg], axis=0)
    lbl = np.array([1]*len(pos) + [0]*len(neg), dtype=np.int32)
    if len(pts) == 0:
        raise ValueError("No prompts provided.")

    # Build SAM predictor
    sam = sam_model_registry[args.model_type](checkpoint=args.checkpoint)
    sam.to(device).eval()
    predictor = SamPredictor(sam)
    predictor.set_image(img)

    # Static input tokens X from image encoder
    if predictor.features is None:
        raise RuntimeError("Predictor has no encoder features.")
    feats = predictor.features  # [1, C, Hf, Wf]
    X = flatten_tokens(feats).to(device)     # [N_tokens, Cx]
    _, Cfeat, Hf, Wf = feats.shape
    N_tokens = Hf * Wf

    # Hook keys per block (Z_l)
    two_way = sam.mask_decoder.transformer
    handles, cache = hook_keys(two_way)

    # Containers (we’ll compute both estimators)
    steps = []
    I_XZ_mine, I_ZY_mine = [], []
    I_XZ_nce,  I_ZY_nce  = [], []

    # Incremental prompts: 1..K
    for k in range(1, len(pts) + 1):
        cache["keys_per_block"].clear()

        with torch.inference_mode():
            masks, scores, logits = predictor.predict(
                point_coords=pts[:k],
                point_labels=lbl[:k],
                multimask_output=False,  # use best single mask for simplicity
            )

        # Collect Z_l from hooks
        if len(cache["keys_per_block"]) == 0:
            remove_hooks(handles)
            raise RuntimeError("Failed to capture Two-Way keys. Check SAM version.")
        Z_list = [z[0] for z in cache["keys_per_block"]]  # each [N_tokens, Cz]
        L = len(Z_list)

        # --- Build Y (soft prob in original space), then pool to token grid ---
        # 'logits' from predictor.predict is low-res logits for the best mask (shape [1,256,256] or [256,256])
        lowres = torch.as_tensor(logits, device=device)
        if lowres.ndim == 2:
            # (256,256) -> [1,1,256,256]
            lowres = lowres[None, None, :, :]
        elif lowres.ndim == 3:
            # (1,256,256) or (N,256,256) -> [N,1,256,256]
            if lowres.shape[0] == 1 and lowres.shape[1] == 256:
                lowres = lowres[:, None, :, :]
            elif lowres.shape[0] >= 1 and lowres.shape[1] == 256:
                lowres = lowres[:, None, :, :]
            else:
                lowres = lowres[None, None, :, :]
        elif lowres.ndim == 4:
            # ensure channel dim == 1
            if lowres.shape[1] != 1:
                lowres = lowres[:, :1, ...]
        else:
            raise RuntimeError(f"Unexpected lowres shape: {tuple(lowres.shape)}")

        with torch.no_grad():
            if hasattr(sam, "postprocess_masks"):
                prob_orig_t = torch.sigmoid(
                    sam.postprocess_masks(
                        lowres,                      # [B,1,256,256] on same device as model
                        input_size=predictor.input_size,
                        original_size=predictor.original_size,
                    )[0, 0]
                )
            elif hasattr(predictor, "model") and hasattr(predictor.model, "postprocess_masks"):
                prob_orig_t = torch.sigmoid(
                    predictor.model.postprocess_masks(
                        lowres,
                        input_size=predictor.input_size,
                        original_size=predictor.original_size,
                    )[0, 0]
                )
            else:
                # Fallback for very old SAM versions: use the returned binary mask
                prob_orig_t = torch.as_tensor(masks[0].astype(np.float32), device=device)

        prob_orig = prob_orig_t.detach().cpu().numpy()            # (H,W)
        Y_tok = downsample_to_tokens(prob_orig, Hf, Wf)           # [N_tokens,1]
        Y = to_torch(Y_tok, device)                               # [N,1]

        # Per-block MI (both estimators)
        mi_xz_m, mi_zy_m = [], []
        mi_xz_n, mi_zy_n = [], []
        for l, Zl in enumerate(Z_list):
            Z = Zl.reshape(N_tokens, -1).to(device)

            # MINE
            mi1 = estimate_mine_lower_bound(
                X, Z,
                steps=args.mine_steps, batch=args.mine_batch, lr=args.mine_lr,
                hidden=args.mine_hidden, seeds=args.seeds, token_noise_std=args.token_noise_std,
                verbose=False
            )
            mi2 = estimate_mine_lower_bound(
                Z, Y,
                steps=args.mine_steps, batch=args.mine_batch, lr=args.mine_lr,
                hidden=args.mine_hidden, seeds=args.seeds, token_noise_std=args.token_noise_std,
                verbose=False
            )
            mi_xz_m.append(mi1); mi_zy_m.append(mi2)

            # InfoNCE
            in1 = estimate_infonce_lower_bound(
                X, Z,
                steps=args.infonce_steps, batch=args.infonce_batch, lr=args.infonce_lr,
                proj_dim=args.infonce_proj, temperature=args.infonce_temp,
                seeds=args.seeds, token_noise_std=args.token_noise_std,
                verbose=False
            )
            in2 = estimate_infonce_lower_bound(
                Z, Y,
                steps=args.infonce_steps, batch=args.infonce_batch, lr=args.infonce_lr,
                proj_dim=args.infonce_proj, temperature=args.infonce_temp,
                seeds=args.seeds, token_noise_std=args.token_noise_std,
                verbose=False
            )
            mi_xz_n.append(in1); mi_zy_n.append(in2)

        steps.append(k)
        I_XZ_mine.append(mi_xz_m); I_ZY_mine.append(mi_zy_m)
        I_XZ_nce.append(mi_xz_n);  I_ZY_nce.append(mi_zy_n)

        print(f"[step {k}] MINE  I(X;Z_last)={mi_xz_m[-1]:.4f}  I(Z_last;Y)={mi_zy_m[-1]:.4f}   | "
              f"NCE  I(X;Z_last)={mi_xz_n[-1]:.4f}  I(Z_last;Y)={mi_zy_n[-1]:.4f}")

    remove_hooks(handles)

    # Convert to arrays
    I_XZ_mine = np.array(I_XZ_mine)  # [S, L]
    I_ZY_mine = np.array(I_ZY_mine)
    I_XZ_nce  = np.array(I_XZ_nce)
    I_ZY_nce  = np.array(I_ZY_nce)
    S, L = I_XZ_mine.shape

    # ----- VISUALS -----
    def plot_curves_last_block():
        fig, ax = plt.subplots(1, 1, figsize=(7.5, 4.2))
        ax.plot(steps, I_XZ_mine[:, -1], label=r"MINE  $I(X;\,Z_{\mathrm{last}})$")
        ax.plot(steps, I_ZY_mine[:, -1], label=r"MINE  $I(Z_{\mathrm{last}};\,Y)$")
        ax.plot(steps, I_XZ_nce[:, -1],  '--', label=r"InfoNCE  $I(X;\,Z_{\mathrm{last}})$")
        ax.plot(steps, I_ZY_nce[:, -1],  '--', label=r"InfoNCE  $I(Z_{\mathrm{last}};\,Y)$")
        ax.set_xlabel("# prompts used")
        ax.set_ylabel("Lower bound (nats)")
        ax.set_title("MI lower bounds vs prompts (last Two-Way block)")
        ax.grid(True, alpha=0.3)
        ax.legend()
        fig.tight_layout(); fig.savefig(out_dir / "curves_last_block.png", dpi=220); plt.close(fig)

    def plot_delta_curves_last_block():
        def delta(v):  # step-to-step change
            d = np.diff(v, prepend=v[:1])
            d[0] = np.nan  # undefined for step 1; show as gap
            return d
        fig, ax = plt.subplots(1, 1, figsize=(7.5, 4.2))
        ax.plot(steps, delta(I_XZ_mine[:, -1]), label=r"Δ MINE  $I(X;\,Z_{\mathrm{last}})$")
        ax.plot(steps, delta(I_ZY_mine[:, -1]), label=r"Δ MINE  $I(Z_{\mathrm{last}};\,Y)$")
        ax.plot(steps, delta(I_XZ_nce[:, -1]),  '--', label=r"Δ InfoNCE  $I(X;\,Z_{\mathrm{last}})$")
        ax.plot(steps, delta(I_ZY_nce[:, -1]),  '--', label=r"Δ InfoNCE  $I(Z_{\mathrm{last}};\,Y)$")
        ax.set_xlabel("# prompts used")
        ax.set_ylabel("Step-to-step change (nats)")
        ax.set_title("Δ MI vs prompts (last block)")
        ax.grid(True, alpha=0.3)
        ax.legend()
        fig.tight_layout(); fig.savefig(out_dir / "delta_curves_last_block.png", dpi=220); plt.close(fig)

    def heatmap(M, title, fname):
        fig, ax = plt.subplots(1, 1, figsize=(min(12, 1.2*L), min(8, 0.8*S)))
        im = ax.imshow(M, cmap="viridis", aspect="auto")
        ax.set_xlabel("Two-Way block index (0..L-1)")
        ax.set_ylabel("# prompts used")
        ax.set_title(title)
        ax.set_xticks(np.arange(L))
        ax.set_yticks(np.arange(S)); ax.set_yticklabels(steps)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Lower bound (nats)")
        fig.tight_layout(); fig.savefig(out_dir / fname, dpi=220); plt.close(fig)

    plot_curves_last_block()
    plot_delta_curves_last_block()
    heatmap(I_XZ_mine, "MINE  I(X; Z_l) over prompts/blocks", "heatmap_MINE_XZ.png")
    heatmap(I_ZY_mine, "MINE  I(Z_l; Y) over prompts/blocks", "heatmap_MINE_ZY.png")
    heatmap(I_XZ_nce,  "InfoNCE  I(X; Z_l) over prompts/blocks", "heatmap_InfoNCE_XZ.png")
    heatmap(I_ZY_nce,  "InfoNCE  I(Z_l; Y) over prompts/blocks", "heatmap_InfoNCE_ZY.png")

    # ----- CSV dumps -----
    import csv
    def dump_csv(M, name):
        with open(out_dir / f"{name}.csv", "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["step"] + [f"Z_block_{i}" for i in range(L)])
            for s_idx, s in enumerate(steps):
                w.writerow([s] + [f"{v:.6f}" for v in M[s_idx]])
    dump_csv(I_XZ_mine, "mine_XZ")
    dump_csv(I_ZY_mine, "mine_ZY")
    dump_csv(I_XZ_nce,  "infonce_XZ")
    dump_csv(I_ZY_nce,  "infonce_ZY")

    print(f"[OK] Wrote outputs to {out_dir.resolve()}")
    print("  - curves_last_block.png, delta_curves_last_block.png")
    print("  - heatmap_MINE_*.png, heatmap_InfoNCE_*.png")
    print("  - mine_*.csv, infonce_*.csv")


def parse_args():
    ap = argparse.ArgumentParser("Mutual Information vs Prompts in SAM (MINE + InfoNCE, robust visuals)")
    ap.add_argument("--image", required=True)
    ap.add_argument("--prompts", required=True, help="JSON with positive_points/negative_points (original coords)")
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--model-type", default="vit_h", choices=["vit_h","vit_l","vit_b"])
    ap.add_argument("--out-dir", default="mi_out")

    # MINE
    ap.add_argument("--mine-steps", type=int, default=200)
    ap.add_argument("--mine-batch", type=int, default=1024)
    ap.add_argument("--mine-lr", type=float, default=1e-3)
    ap.add_argument("--mine-hidden", type=int, default=256)

    # InfoNCE
    ap.add_argument("--infonce-steps", type=int, default=200)
    ap.add_argument("--infonce-batch", type=int, default=1024)
    ap.add_argument("--infonce-lr", type=float, default=1e-3)
    ap.add_argument("--infonce-proj", type=int, default=128)
    ap.add_argument("--infonce-temp", type=float, default=0.1)

    # Stability knobs
    ap.add_argument("--seeds", type=int, default=1, help="Avg lower bounds over this many seeds")
    ap.add_argument("--token-noise-std", type=float, default=0.0, help="Optional tiny Gaussian noise on tokens")

    ap.add_argument("--cpu", action="store_true")
    return ap.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(args)
# ----------------- End of file -----------------