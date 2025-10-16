#!/usr/bin/env python3
"""
Gaussianity checks for SAM token variables:
- X: encoder tokens [N, Cx]
- Z_l: Two-Way image-side keys at block l [N, Cz]  (default last block)
- Y: mask probability -> logit -> pooled to token grid [N, 1]

Tests:
  1) Univariate normality (D’Agostino–Pearson K^2) + FDR(BH)
  2) Random-projection normality (Cramér–Wold surrogate) + FDR(BH)
  3) Henze–Zirkler multivariate normality (via pingouin) [optional]

Saves: JSON summary with reject rates and p-values (counts), and prints a table.

Usage:
  python gaussian_normality_sam.py \
    --image /path/to/image.jpg \
    --prompts /path/to/prompts.json \
    --checkpoint /path/to/sam_vit_h_4b8939.pth \
    --model-type vit_h \
    --out-dir /tmp/gauss_check \
    --block-idx -1 \
    --n-proj 256 \
    --alpha 0.05
"""

import argparse
import json
from pathlib import Path
import numpy as np
from PIL import Image

import torch
import torch.nn.functional as F

# stats
from scipy.stats import normaltest

# SAM
from segment_anything import sam_model_registry, SamPredictor


# ---------------- I/O utils ----------------
def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def load_image_rgb(path):
    return np.asarray(Image.open(path).convert("RGB"))

def to_torch(x, device):
    return torch.as_tensor(x, dtype=torch.float32, device=device)


# --------------- SAM feature helpers ---------------
def flatten_tokens(feats_bchw: torch.Tensor):
    # feats: [1, C, Hf, Wf] -> [N, C]
    _, C, H, W = feats_bchw.shape
    return feats_bchw[0].permute(1, 2, 0).reshape(H * W, C).contiguous()

def hook_two_way_keys(two_way_transformer):
    cache = {"keys_per_block": []}
    hs = []
    def fwd_hook(module, args, out):
        q, k = out  # block returns (queries, keys)
        cache["keys_per_block"].append(k.detach())
    for blk in two_way_transformer.layers:
        hs.append(blk.register_forward_hook(fwd_hook))
    return hs, cache

def remove_hooks(handles):
    for h in handles:
        h.remove()

def postprocess_probs_to_original(sam, predictor, lowres_logits, device):
    """
    lowres_logits: torch tensor, aim for [B,1,256,256]
    returns prob in original image space: torch [H, W]
    """
    lr = torch.as_tensor(lowres_logits, device=device)
    if lr.ndim == 2:              # (256,256)
        lr = lr[None, None]
    elif lr.ndim == 3:            # (1,256,256) or (N,256,256)
        if lr.shape[1] == 256:
            lr = lr[:, None]
        else:
            lr = lr[None, None]
    elif lr.ndim == 4:            # [B,C,H,W]
        if lr.shape[1] != 1:
            lr = lr[:, :1]
    else:
        raise RuntimeError(f"Unexpected lowres shape: {tuple(lr.shape)}")

    if hasattr(sam, "postprocess_masks"):
        prob = torch.sigmoid(
            sam.postprocess_masks(
                lr, input_size=predictor.input_size, original_size=predictor.original_size
            )[0, 0]
        )
    elif hasattr(predictor, "model") and hasattr(predictor.model, "postprocess_masks"):
        prob = torch.sigmoid(
            predictor.model.postprocess_masks(
                lr, input_size=predictor.input_size, original_size=predictor.original_size
            )[0, 0]
        )
    else:
        # fallback to binary mask if API not available
        raise RuntimeError("Could not find postprocess_masks on SAM; please update segment-anything.")
    return prob  # [H,W] prob


def downsample_to_tokens(mask_hw_np: np.ndarray, Ht: int, Wt: int) -> np.ndarray:
    t = torch.from_numpy(mask_hw_np).float()[None, None]
    ds = F.interpolate(t, size=(Ht, Wt), mode="area")[0, 0]
    return ds.view(-1, 1).cpu().numpy()


# ---------------- Normality tests ----------------
def fdr_bh(pvals, alpha=0.05):
    """
    Benjamini–Hochberg FDR control.
    Returns boolean reject mask and adjusted q-values (same order as input).
    """
    pvals = np.asarray(pvals, dtype=float)
    m = pvals.size
    if m == 0:
        return np.array([], dtype=bool), np.array([], dtype=float)

    order = np.argsort(pvals)
    ranks = np.arange(1, m + 1)
    p_sorted = pvals[order]

    # adjusted p = p_i * m / i ; then monotone non-decreasing from the right
    q_sorted = p_sorted * m / ranks
    q_sorted = np.minimum.accumulate(q_sorted[::-1])[::-1]
    q_sorted = np.clip(q_sorted, 0.0, 1.0)

    # reject if p_i <= (i/m)*alpha  <=> q_i <= alpha
    reject_sorted = q_sorted <= alpha

    # back to original order
    q = np.empty_like(q_sorted)
    q[order] = q_sorted
    reject = np.empty_like(reject_sorted, dtype=bool)
    reject[order] = reject_sorted
    return reject, q


def univariate_normality_screen(X: np.ndarray, alpha=0.05):
    """
    Run D’Agostino–Pearson K^2 on each feature column; control FDR.
    Returns summary + per-feature qvals count only (not saving all q-values).
    """
    X = np.asarray(X, dtype=np.float64)
    d = X.shape[1]
    pvals = []
    for j in range(d):
        try:
            p = normaltest(X[:, j], nan_policy='omit').pvalue
        except Exception:
            p = np.nan
        pvals.append(p if np.isfinite(p) else 1.0)
    pvals = np.array(pvals)
    reject, qvals = fdr_bh(pvals, alpha=alpha)
    return {
        "features": d,
        "alpha": alpha,
        "rejected": int(reject.sum()),
        "reject_rate": float(reject.mean()) if d else 0.0,
    }


def random_projection_test(X: np.ndarray, n_proj=256, alpha=0.05, seed=0):
    """
    Draw n_proj random unit directions a_k; test normality of (Xc @ a_k).
    FDR over projections.
    """
    X = np.asarray(X, dtype=np.float64)
    N, d = X.shape
    if N < 8:
        return {"n_proj": n_proj, "alpha": alpha, "rejected": 0, "reject_rate": 0.0, "note": "too few samples"}
    Xc = X - X.mean(axis=0, keepdims=True)
    rng = np.random.default_rng(seed)
    A = rng.normal(size=(d, n_proj))
    A /= (np.linalg.norm(A, axis=0, keepdims=True) + 1e-12)
    pvals = []
    for k in range(n_proj):
        y = Xc @ A[:, k]
        try:
            p = normaltest(y, nan_policy='omit').pvalue
        except Exception:
            p = np.nan
        pvals.append(p if np.isfinite(p) else 1.0)
    pvals = np.array(pvals)
    reject, _ = fdr_bh(pvals, alpha=alpha)
    return {
        "n_proj": int(n_proj),
        "alpha": alpha,
        "rejected": int(reject.sum()),
        "reject_rate": float(reject.mean()),
    }


def hz_multivariate_test(X: np.ndarray, alpha=0.05):
    """
    Henze–Zirkler test via pingouin (optional).
    """
    try:
        import pingouin as pg
    except Exception:
        return {"available": False, "normal": None, "stat": None, "p": None, "alpha": alpha,
                "note": "Install `pingouin` to run HZ test: pip install pingouin"}
    try:
        stat, p, normal = pg.multivariate_normality(X, alpha=alpha, method='hz')
        return {"available": True, "normal": bool(normal), "stat": float(stat), "p": float(p), "alpha": alpha}
    except Exception as e:
        return {"available": False, "normal": None, "stat": None, "p": None, "alpha": alpha,
                "note": f"HZ test failed: {e}"}


def logit_eps(y, eps=1e-3):
    y = np.clip(y, 0.0, 1.0)
    return np.log((y + eps) / (1 - y + eps))


# --------------- Main ---------------
def run(args):
    device = "cuda" if torch.cuda.is_available() and not args.cpu else "cpu"
    out_dir = Path(args.out_dir); ensure_dir(out_dir)

    # --- Inputs ---
    img = load_image_rgb(args.image)
    with open(args.prompts, "r") as f:
        prom = json.load(f)
    pos = np.array(prom.get("positive_points", []), dtype=np.float32) if prom.get("positive_points") else np.zeros((0,2), np.float32)
    neg = np.array(prom.get("negative_points", []), dtype=np.float32) if prom.get("negative_points") else np.zeros((0,2), np.float32)
    pts = np.concatenate([pos, neg], axis=0)
    lbl = np.array([1]*len(pos) + [0]*len(neg), dtype=np.int32)
    if pts.shape[0] == 0:
        raise ValueError("No prompts found in JSON.")

    # --- SAM ---
    sam = sam_model_registry[args.model_type](checkpoint=args.checkpoint)
    sam.to(device).eval()
    predictor = SamPredictor(sam)
    predictor.set_image(img)

    # Encoder tokens X
    if predictor.features is None:
        raise RuntimeError("Predictor has no encoder features.")
    feats = predictor.features  # [1, C, Hf, Wf]
    X = flatten_tokens(feats).cpu().numpy()  # [N, Cx]
    _, Cfeat, Hf, Wf = feats.shape

    # Hook Two-Way keys (Z_l)
    two_way = sam.mask_decoder.transformer
    handles, cache = hook_two_way_keys(two_way)

    # Forward with all prompts (order as provided)
    with torch.inference_mode():
        masks, scores, logits = predictor.predict(
            point_coords=pts,
            point_labels=lbl,
            multimask_output=False,
        )

    remove_hooks(handles)
    if len(cache["keys_per_block"]) == 0:
        raise RuntimeError("Did not capture Two-Way keys. Check SAM version.")
    Z_list = [z[0].cpu().numpy() for z in cache["keys_per_block"]]  # each [N, Cz]
    L = len(Z_list)

    # Select block
    blk_idx = args.block_idx if args.block_idx >= 0 else (L - 1)
    Z = Z_list[blk_idx]  # [N, Cz]

    # Build Y: postprocess logits -> prob in original -> area-pool to tokens -> logit transform
    prob_orig_t = postprocess_probs_to_original(sam, predictor, logits, device)  # torch [H,W]
    prob_orig = prob_orig_t.detach().cpu().numpy()
    Y_tok = downsample_to_tokens(prob_orig, Hf, Wf)  # [N,1]
    Y_log = logit_eps(Y_tok, eps=args.logit_eps).astype(np.float64)

    # Prepare joint arrays
    XZ = np.concatenate([X, Z], axis=1)
    ZY = np.concatenate([Z, Y_log], axis=1)

    # --------- Run tests ----------
    alpha = args.alpha
    n_proj = args.n_proj

    results = {
        "config": {
            "block_idx": blk_idx,
            "alpha": alpha,
            "n_proj": n_proj,
            "logit_eps": args.logit_eps,
            "Hf": int(Hf), "Wf": int(Wf),
            "N_tokens": int(Hf * Wf),
            "Cx": int(X.shape[1]), "Cz": int(Z.shape[1])
        },
        "univariate": {
            "X":  univariate_normality_screen(X, alpha),
            "Z":  univariate_normality_screen(Z, alpha),
            "Y_logit": univariate_normality_screen(Y_log, alpha),
        },
        "random_projection": {
            "X":  random_projection_test(X,  n_proj=n_proj, alpha=alpha, seed=0),
            "Z":  random_projection_test(Z,  n_proj=n_proj, alpha=alpha, seed=1),
            "XZ": random_projection_test(XZ, n_proj=n_proj, alpha=alpha, seed=2),
            "ZY": random_projection_test(ZY, n_proj=n_proj, alpha=alpha, seed=3),
        },
        "henze_zirkler": {
            "X":  hz_multivariate_test(X,  alpha),
            "Z":  hz_multivariate_test(Z,  alpha),
            "XZ": hz_multivariate_test(XZ, alpha),
            "ZY": hz_multivariate_test(ZY, alpha),
        }
    }

    # --------- Print a readable summary ----------
    def fmt_uni(u):
        return f"rejected {u['rejected']}/{u['features']} ({100.0*u['reject_rate']:.1f}%)"

    def fmt_rp(r):
        return f"rejected {r['rejected']}/{r['n_proj']} ({100.0*r['reject_rate']:.1f}%)"

    print("\n=== Gaussianity summary (alpha={:.3f}) on block {} ===".format(alpha, blk_idx))
    print("Tokens: N={}, Cx={}, Cz={}, grid={}x{}".format(int(Hf*Wf), X.shape[1], Z.shape[1], Hf, Wf))
    print("\n[Univariate K^2 + FDR]")
    print("  X:       " + fmt_uni(results["univariate"]["X"]))
    print("  Z:       " + fmt_uni(results["univariate"]["Z"]))
    print("  Y_logit: " + fmt_uni(results["univariate"]["Y_logit"]))
    print("\n[Random projections + FDR] (Cramér–Wold surrogate)")
    print("  X:   " + fmt_rp(results["random_projection"]["X"]))
    print("  Z:   " + fmt_rp(results["random_projection"]["Z"]))
    print("  XZ:  " + fmt_rp(results["random_projection"]["XZ"]))
    print("  ZY:  " + fmt_rp(results["random_projection"]["ZY"]))
    print("\n[Henze–Zirkler multivariate] (if pingouin available)")
    for k in ["X", "Z", "XZ", "ZY"]:
        hz = results["henze_zirkler"][k]
        if hz["available"]:
            verdict = "normal" if hz["normal"] else "non-normal"
            print(f"  {k:3s}: p={hz['p']:.4g}  -> {verdict}")
        else:
            print(f"  {k:3s}: (skipped) {hz.get('note','')}")
    print()

    # --------- Save report ----------
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "gaussianity_report.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"[OK] Saved report -> {out_dir / 'gaussianity_report.json'}")


def parse_args():
    ap = argparse.ArgumentParser("Gaussianity tests for SAM token variables")
    ap.add_argument("--image", required=True)
    ap.add_argument("--prompts", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--model-type", default="vit_h", choices=["vit_h","vit_l","vit_b"])
    ap.add_argument("--out-dir", default="gauss_out")
    ap.add_argument("--block-idx", type=int, default=-1, help="-1 for last block")
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--n-proj", type=int, default=256)
    ap.add_argument("--logit-eps", type=float, default=1e-3)
    ap.add_argument("--cpu", action="store_true")
    return ap.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(args)
