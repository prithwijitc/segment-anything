#!/usr/bin/env python3
"""
Sample prompts from a binary mask (.npy or .png) and preview SAM once.

- Mask can be:
    * .npy file: binary 0/1 or any nonzero=FG
    * image file (e.g., .png): background=0, any nonzero=foreground
- Foreground (1) -> inclusion (label=1)
- Background (0) -> exclusion (label=0)
- Time series alternates: POS (t=1), NEG (t=2), POS (t=3), ...

Usage example:
python sample_prompts_from_mask_any.py \
  --image /path/to/image.jpg \
  --mask /path/to/mask.png \
  --checkpoint /path/to/sam_vit_h_4b8939.pth \
  --model-type vit_h \
  --num-pairs 5 \
  --out-dir ./outputs
"""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import torch

# --- SAM ---
try:
    from segment_anything import sam_model_registry, SamPredictor
except Exception as e:
    raise SystemExit(
        "Could not import 'segment_anything'. "
        "Install with: pip install git+https://github.com/facebookresearch/segment-anything.git"
    ) from e


def load_rgb_bgr(image_path: str):
    bgr = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    return rgb, bgr


def ensure_binary_mask(mask: np.ndarray) -> np.ndarray:
    """Return boolean mask where True=foreground (nonzero)."""
    if mask.dtype == np.bool_:
        return mask
    return (mask.astype(np.float32) > 0)


def load_mask_any(path: str) -> np.ndarray:
    """
    Load mask from .npy or image.
    Image mask rule: background=0, any nonzero=foreground.
    Returns a boolean array (True=FG).
    """
    ext = Path(path).suffix.lower()
    if ext == ".npy":
        m = np.load(path)
        return ensure_binary_mask(m)
    # Image path
    # Use UNCHANGED to preserve 1/3/4 channels and depth (8/16/32-bit).
    raw = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if raw is None:
        raise FileNotFoundError(f"Could not read mask image: {path}")
    # If multi-channel, consider pixel FG if ANY channel is nonzero.
    if raw.ndim == 3:
        # Max across channels to detect any nonzero
        raw = raw.max(axis=2)
    # Now raw is single channel
    return (raw.astype(np.float32) > 0)


def maybe_resize_mask(mask_bool: np.ndarray, target_hw) -> np.ndarray:
    """Resize mask (NN) if shape doesn't match target (H, W)."""
    H, W = target_hw
    if mask_bool.shape[:2] == (H, W):
        return mask_bool
    print(f"[WARN] Mask shape {mask_bool.shape[:2]} != image shape {(H, W)}. Resizing with nearest-neighbor.")
    m = (mask_bool.astype(np.uint8) * 255)
    m = cv2.resize(m, (W, H), interpolation=cv2.INTER_NEAREST)
    return (m > 127)


def erode_bool(mask_bool: np.ndarray, iters: int) -> np.ndarray:
    if iters <= 0:
        return mask_bool
    kernel = np.ones((3, 3), np.uint8)
    m = cv2.erode(mask_bool.astype(np.uint8), kernel, iterations=iters)
    return m.astype(bool)


def sample_points_from_mask(mask_bool: np.ndarray, k: int, seed: int = 0):
    """Uniform random sample k (y, x) coordinates from True pixels."""
    ys, xs = np.where(mask_bool)
    n = ys.size
    if n == 0:
        return []
    k = min(k, n)
    rng = np.random.default_rng(seed)
    idx = rng.choice(n, size=k, replace=False)
    return [(int(xs[i]), int(ys[i])) for i in idx]  # (x, y)


def overlay_mask_on_bgr(bgr: np.ndarray, mask: np.ndarray, alpha: float = 0.5):
    """Overlay a binary mask in green over BGR image."""
    if mask.dtype != bool:
        mask = mask.astype(bool)
    out = bgr.copy()
    color = np.array([0, 255, 0], dtype=np.uint8)
    out[mask] = (out[mask] * (1 - alpha) + color * alpha).astype(np.uint8)
    return out


def draw_points_on_bgr(img: np.ndarray, points, labels, radius: int = 5):
    """Draw POS points green, NEG red."""
    disp = img.copy()
    for (x, y), lab in zip(points, labels):
        color = (0, 255, 0) if lab == 1 else (0, 0, 255)
        cv2.circle(disp, (int(x), int(y)), radius, color, thickness=-1, lineType=cv2.LINE_AA)
        cv2.circle(disp, (int(x), int(y)), radius + 3, color, thickness=1, lineType=cv2.LINE_AA)
    return disp


def build_time_series_prompts(pos_points, neg_points):
    """Alternate starting with POS; output list of dicts with t, x, y, label."""
    n = min(len(pos_points), len(neg_points))
    prompts = []
    t = 1
    for i in range(n):
        x_pos, y_pos = pos_points[i]
        x_neg, y_neg = neg_points[i]
        prompts.append({"t": t, "x": int(x_pos), "y": int(y_pos), "label": 1}); t += 1
        prompts.append({"t": t, "x": int(x_neg), "y": int(y_neg), "label": 0}); t += 1
    return prompts


def parse_args():
    p = argparse.ArgumentParser("Sample SAM prompts from a binary mask (.npy or .png) and preview once.")
    p.add_argument("--image", type=str, required=True, help="Path to input image (jpg/png/...)")
    p.add_argument("--mask", type=str, required=True, help="Path to binary mask (.npy or image). 0=BG, nonzero=FG")
    p.add_argument("--checkpoint", type=str, required=True, help="Path to SAM checkpoint .pth")
    p.add_argument("--model-type", type=str, default="vit_h", choices=["vit_h", "vit_l", "vit_b"],
                   help="SAM model type matching the checkpoint.")
    p.add_argument("--num-pairs", type=int, default=5, help="Number of (POS, NEG) pairs to sample.")
    p.add_argument("--seed", type=int, default=0, help="Random seed for sampling.")
    p.add_argument("--edge-margin", type=int, default=3,
                   help="Erode FG and BG by this many iterations to avoid boundary points. 0 to disable.")
    p.add_argument("--multimask", action="store_true",
                   help="Ask SAM for 3 candidates and pick best by score.")
    p.add_argument("--out-dir", type=str, default="./outputs", help="Directory to save prompts JSON.")
    p.add_argument("--max-window", type=int, default=1280, help="Max display size (preserves aspect).")
    return p.parse_args()


def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load image + mask
    img_rgb, img_bgr = load_rgb_bgr(args.image)
    H, W = img_bgr.shape[:2]

    mask_bool = load_mask_any(args.mask)
    mask_bool = maybe_resize_mask(mask_bool, (H, W))

    # Safe sampling regions via erosion on both FG and BG
    fg = mask_bool
    bg = ~mask_bool
    safe_fg = erode_bool(fg, args.edge_margin)
    safe_bg = erode_bool(bg, args.edge_margin)

    # Sample points
    pos_pts = sample_points_from_mask(safe_fg, args.num_pairs, seed=args.seed)
    neg_pts = sample_points_from_mask(safe_bg, args.num_pairs, seed=args.seed + 1)

    if len(pos_pts) == 0:
        raise SystemExit("[ERROR] No valid foreground pixels to sample from (after erosion).")
    if len(neg_pts) == 0:
        raise SystemExit("[ERROR] No valid background pixels to sample from (after erosion).")

    # Build alternating time series prompts starting with inclusion (POS)
    prompts = build_time_series_prompts(pos_pts, neg_pts)
    total_pairs = len(prompts) // 2
    if total_pairs < args.num_pairs:
        print(f"[WARN] Only {total_pairs} pairs available (limited by FG/BG pixels).")
    print(f"[INFO] Built {len(prompts)} prompts ({total_pairs} POS/NEG pairs).")

    # Save prompts JSON
    stem = Path(args.image).stem
    prompts_path = out_dir / f"{stem}_sampled_prompts.json"
    with open(prompts_path, "w") as f:
        json.dump({"prompts": prompts}, f, indent=2)
    print(f"[SAVED] Prompts JSON -> {prompts_path}")

    # --- Run SAM once with all prompts and display ---
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[INFO] Loading SAM '{args.model_type}' on {device}...")
    sam = sam_model_registry[args.model_type](checkpoint=args.checkpoint).to(device=device)
    predictor = SamPredictor(sam)
    predictor.set_image(img_rgb)

    # SAM expects Nx2 float32 (x, y) and labels int
    point_coords = np.array([(p["x"], p["y"]) for p in prompts], dtype=np.float32)
    point_labels = np.array([p["label"] for p in prompts], dtype=np.int32)

    with torch.inference_mode():
        masks, scores, _ = predictor.predict(
            point_coords=point_coords,
            point_labels=point_labels,
            multimask_output=args.multimask
        )
    if args.multimask:
        best = int(np.argmax(scores))
        sam_mask = masks[best]
    else:
        sam_mask = masks[0]

    # Display once (overlay + points), wait for a key, then close
    overlay = overlay_mask_on_bgr(img_bgr, sam_mask, alpha=0.5)
    overlay = draw_points_on_bgr(overlay, [(p["x"], p["y"]) for p in prompts],
                                 [p["label"] for p in prompts])
    scale = min(args.max_window / max(H, W), 1.0)
    disp_size = (int(round(W * scale)), int(round(H * scale)))
    disp = cv2.resize(overlay, disp_size, interpolation=cv2.INTER_LINEAR)

    win = "SAM preview (press any key to close)"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL | cv2.WINDOW_GUI_EXPANDED)
    cv2.resizeWindow(win, disp_size[0], disp_size[1])
    cv2.imshow(win, disp)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    print("[DONE] Preview closed.")

if __name__ == "__main__":
    main()
