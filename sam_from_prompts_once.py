#!/usr/bin/env python3
"""
Run SAM once with a prompt JSON and save:
  - best mask as .npy (0/1)
  - a JPG overlay on the image (mask + prompt dots)
Optionally:
  - load a GT mask (.npy/.png/.jpg) and print IoU_fg, IoU_bg, mIoU

JSON format:
{
  "prompts": [
    {"t": 1, "x": 659, "y": 239, "label": 1},
    {"t": 2, "x": 548, "y": 110, "label": 0},
    ...
  ]
}

GT mask convention (image masks): 0=background, nonzero=foreground.
"""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import torch

try:
    from segment_anything import sam_model_registry, SamPredictor
except Exception as e:
    raise SystemExit(
        "Install SAM first:\n  pip install git+https://github.com/facebookresearch/segment-anything.git"
    ) from e


# --------------------- IO helpers ---------------------

def load_rgb_bgr(image_path: str):
    bgr = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    return rgb, bgr


def load_mask_any(path: str) -> np.ndarray:
    """
    Load a binary mask from .npy or image (.png/.jpg/...).
    Image rule: background=0, any non-zero=foreground.
    Returns a boolean array (True=FG).
    """
    ext = Path(path).suffix.lower()
    if ext == ".npy":
        m = np.load(path)
        return (m.astype(np.float32) > 0)
    raw = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if raw is None:
        raise FileNotFoundError(f"Could not read mask image: {path}")
    if raw.ndim == 3:
        raw = raw.max(axis=2)
    return (raw.astype(np.float32) > 0)


def maybe_resize_mask(mask_bool: np.ndarray, target_hw) -> np.ndarray:
    H, W = target_hw
    if mask_bool.shape[:2] == (H, W):
        return mask_bool.astype(bool)
    m = (mask_bool.astype(np.uint8) * 255)
    m = cv2.resize(m, (W, H), interpolation=cv2.INTER_NEAREST)
    return (m > 127)


# --------------------- Viz helpers ---------------------

def overlay_mask_on_bgr(bgr: np.ndarray, mask: np.ndarray, alpha: float = 0.5):
    """Overlay boolean/0-1 mask in green on a BGR image."""
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


# --------------------- Metrics ---------------------

def iou_bool(a: np.ndarray, b: np.ndarray) -> float:
    """IoU for boolean masks of the same shape."""
    a = a.astype(bool); b = b.astype(bool)
    inter = np.logical_and(a, b).sum()
    uni = np.logical_or(a, b).sum()
    if uni == 0:
        # If both are empty for this class, define IoU=1.0 (perfect match on emptiness)
        return 1.0
    return float(inter) / float(uni)


def compute_binary_miou(pred_bool: np.ndarray, gt_bool: np.ndarray):
    """Returns (IoU_fg, IoU_bg, mIoU) for binary segmentation."""
    iou_fg = iou_bool(pred_bool, gt_bool)
    iou_bg = iou_bool(~pred_bool, ~gt_bool)
    miou = 0.5 * (iou_fg + iou_bg)
    return iou_fg, iou_bg, miou


# --------------------- Main ---------------------

def parse_args():
    p = argparse.ArgumentParser("SAM once: image + prompts.json -> best mask .npy + JPG overlay (+ optional mIoU)")
    p.add_argument("--image", required=True, help="Path to input image (jpg/png/...)")
    p.add_argument("--prompts", required=True, help="Path to prompts JSON")
    p.add_argument("--checkpoint", required=True, help="Path to SAM checkpoint .pth")
    p.add_argument("--model-type", default="vit_h", choices=["vit_h", "vit_l", "vit_b"],
                   help="SAM model variant matching the checkpoint")
    p.add_argument("--multimask", action="store_true",
                   help="Ask SAM for 3 masks and choose best by score")
    p.add_argument("--out-dir", default="./outputs", help="Where to save outputs")
    p.add_argument("--overlay-alpha", type=float, default=0.5, help="Mask opacity on overlay")
    p.add_argument("--no-points", dest="no_points", action="store_true",
                   help="Do NOT draw prompt points on overlay")
    p.add_argument("--gt-mask", type=str, default=None,
                   help="Optional GT mask (.npy/.png/.jpg). 0=BG, nonzero=FG; resized NN to image if needed.")
    return p.parse_args()


def main():
    args = parse_args()
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)

    # Load image
    img_rgb, img_bgr = load_rgb_bgr(args.image)
    H, W = img_bgr.shape[:2]
    stem = Path(args.image).stem

    # Load and order prompts
    with open(args.prompts, "r") as f:
        data = json.load(f)
    plist = data.get("prompts", [])
    if len(plist) == 0:
        print("[WARN] No prompts found; predicted mask will be empty.")
    if all(("t" in p) and isinstance(p["t"], (int, float)) for p in plist):
        plist = sorted(plist, key=lambda p: p["t"])

    point_coords = np.array([[p["x"], p["y"]] for p in plist], dtype=np.float32) if plist else None
    point_labels = np.array([int(p["label"]) for p in plist], dtype=np.int32) if plist else None

    # Prepare SAM
    device = "cuda:1" if torch.cuda.is_available() else "cpu"
    sam = sam_model_registry[args.model_type](checkpoint=args.checkpoint).to(device=device)
    predictor = SamPredictor(sam)
    predictor.set_image(img_rgb)

    # Predict
    if point_coords is None:
        best_mask = np.zeros((H, W), dtype=bool)
        scores = np.array([0.0])
    else:
        with torch.inference_mode():
            masks, scores, _ = predictor.predict(
                point_coords=point_coords,
                point_labels=point_labels,
                multimask_output=args.multimask
            )
        best_mask = masks[int(np.argmax(scores))] if args.multimask else masks[0]

    # Save predicted binary mask (0/1)
    mask_npy_path = out_dir / f"{stem}_sam_mask.npy"
    pred_bin = (best_mask.astype(np.uint8) > 0).astype(np.uint8)
    np.save(mask_npy_path, pred_bin)

    # Build and save overlay (mask + prompt dots by default)
    overlay = overlay_mask_on_bgr(img_bgr, pred_bin, alpha=args.overlay_alpha)
    if not args.no_points and len(plist) > 0:
        pts = [(int(p["x"]), int(p["y"])) for p in plist]
        labs = [int(p["label"]) for p in plist]
        overlay = draw_points_on_bgr(overlay, pts, labs)

    overlay_jpg_path = out_dir / f"{stem}_sam_overlay.jpg"
    cv2.imwrite(str(overlay_jpg_path), overlay, [int(cv2.IMWRITE_JPEG_QUALITY), 95])

    print(f"[SAVED] Predicted mask (0/1): {mask_npy_path}")
    print(f"[SAVED] Overlay JPG:         {overlay_jpg_path}")
    if point_coords is not None:
        print(f"[INFO] Used {len(point_coords)} prompts. "
              f"{'Best of 3' if args.multimask else 'Single-mask'} score(s): {np.array2string(scores, precision=4)}")

    # Optional: compute IoU metrics if GT mask provided
    if args.gt_mask:
        gt = load_mask_any(args.gt_mask)
        gt = maybe_resize_mask(gt, (H, W))
        pred_bool = pred_bin.astype(bool)
        gt_bool = gt.astype(bool)
        iou_fg, iou_bg, miou = compute_binary_miou(pred_bool, gt_bool)
        print(f"[METRICS] IoU_fg={iou_fg:.4f}  IoU_bg={iou_bg:.4f}  mIoU={miou:.4f}")


if __name__ == "__main__":
    main()
