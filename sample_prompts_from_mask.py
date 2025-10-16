#!/usr/bin/env python3
"""
Sample positive/negative point prompts from an image + *binary* mask JPG.

Mask rule: anything != 0 (after converting to grayscale) is considered FOREGROUND.

Outputs prompts.json like:
{
  "positive_points": [[x, y], ...],
  "negative_points": [[x, y], ...]
}

Examples
--------
# medium density (default)
python sample_prompts_from_mask.py \
  --image dog.jpg \
  --mask dog_mask.jpg \
  --out prompts.json

# denser sampling
python sample_prompts_from_mask.py \
  --image dog.jpg \
  --mask dog_mask.jpg \
  --density dense \
  --out prompts_dense.json

# explicit stride + caps + reproducible jitter
python sample_prompts_from_mask.py \
  --image dog.jpg --mask dog_mask.jpg \
  --stride 20 --max-pos 200 --max-neg 200 \
  --seed 123 --out prompts_custom.json
"""

import argparse
import json
from pathlib import Path
import numpy as np
from PIL import Image

DENSITY_TO_STRIDE = {
    "ultra_sparse": 96,
    "sparse": 64,
    "medium": 32,
    "dense": 16,
    "ultra_dense": 8,
}

def load_image_size(path):
    with Image.open(path) as im:
        w, h = im.size
    return (h, w)  # (H, W)

def load_mask_bool_nonzero(path):
    """
    Load a JPG mask, convert to grayscale, and mark FOREGROUND where pixel != 0.
    Note: with JPEG, tiny compression noise can make edges nonzero; that's okay per spec.
    """
    with Image.open(path).convert("L") as m:
        arr = np.asarray(m)
    return (arr != 0)

def build_grid(H, W, stride, jitter_frac=0.3, rng=None):
    """
    Make a jittered grid of (x,y) points across the image.
    Returns int pixel coords in original space.
    """
    xs = np.arange(0, W, stride, dtype=np.float32)
    ys = np.arange(0, H, stride, dtype=np.float32)
    X, Y = np.meshgrid(xs, ys)  # [ny, nx]
    X = X.ravel()
    Y = Y.ravel()

    rng = rng or np.random.default_rng()
    jitter = jitter_frac * stride
    if jitter > 0:
        X += rng.uniform(-jitter, jitter, size=X.shape)
        Y += rng.uniform(-jitter, jitter, size=Y.shape)

    X = np.clip(np.rint(X), 0, W - 1).astype(np.int32)
    Y = np.clip(np.rint(Y), 0, H - 1).astype(np.int32)

    pts = np.stack([X, Y], axis=1)  # [N,2] [x,y]
    if len(pts) > 1:
        pts = np.unique(pts, axis=0)
    return pts

def split_pos_neg(pts_xy, mask_bool):
    xs = pts_xy[:, 0]
    ys = pts_xy[:, 1]
    inside = mask_bool[ys, xs]
    pos = pts_xy[inside]
    neg = pts_xy[~inside]
    return pos, neg

def subsample(pts_xy, max_count=None, rng=None):
    if max_count is None or len(pts_xy) <= max_count:
        return pts_xy
    rng = rng or np.random.default_rng()
    idx = rng.choice(len(pts_xy), size=max_count, replace=False)
    return pts_xy[idx]

def main():
    ap = argparse.ArgumentParser(description="Sample SAM point prompts from a nonzero-is-foreground JPG mask.")
    ap.add_argument("--image", required=True, help="Path to input image (used for size).")
    ap.add_argument("--mask", required=True, help="Path to binary mask JPG (anything != 0 is foreground).")
    ap.add_argument("--out", required=True, help="Output JSON path.")
    ap.add_argument("--density", default="medium",
                    choices=list(DENSITY_TO_STRIDE.keys()),
                    help="Sampling density preset (controls grid stride).")
    ap.add_argument("--stride", type=int, default=None,
                    help="Override stride in pixels (smaller = denser).")
    ap.add_argument("--jitter-frac", type=float, default=0.3,
                    help="Fraction of stride for random jitter (0..0.5 is typical).")
    ap.add_argument("--max-pos", type=int, default=None, help="Cap number of positive points (optional).")
    ap.add_argument("--max-neg", type=int, default=None, help="Cap number of negative points (optional).")
    ap.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility.")
    args = ap.parse_args()

    H, W = load_image_size(args.image)
    mask_bool = load_mask_bool_nonzero(args.mask)

    if mask_bool.shape != (H, W):
        raise ValueError(f"Mask size {mask_bool.shape[::-1]} does not match image size {(W, H)}")

    stride = args.stride if args.stride is not None else DENSITY_TO_STRIDE[args.density]
    rng = np.random.default_rng(args.seed)

    pts = build_grid(H, W, stride=stride, jitter_frac=args.jitter_frac, rng=rng)
    pos, neg = split_pos_neg(pts, mask_bool)

    pos = subsample(pos, args.max_pos, rng=rng)
    neg = subsample(neg, args.max_neg, rng=rng)

    out = {
        "positive_points": pos.astype(int).tolist(),  # [[x,y], ...]
        "negative_points": neg.astype(int).tolist(),
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)

    print(f"[OK] wrote {out_path}")
    print(f"  image size: {W}x{H}")
    print(f"  stride: {stride} px ({'override' if args.stride else args.density})")
    print(f"  positives: {len(pos)}  negatives: {len(neg)}")

if __name__ == "__main__":
    main()
