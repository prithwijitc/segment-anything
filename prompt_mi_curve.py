#!/usr/bin/env python3
"""
Prompt MI vs mIoU with SAM (iterative prompt additions).

- Loads: SAM checkpoint, an RGB image (jpg/png), and a GT mask (jpg/png/npy; nonzero=foreground).
- Iterations: start with ONE (inclusion, exclusion) pair; then add ONE pair per iteration.
- At each iteration:
    * Run SAM with all accumulated points
    * Upscale logits to EXACT image size (H,W)
    * Compute:
        - mIoU (after thresholding)
        - MI-proxy = -CrossEntropy (higher = more MI; since H(Y) is constant across prompts)
    * Save: overlays, probs/logits as .npy, prompt JSON, and CSV log
- Plots: saves a figure with TWO SUBPLOTS (MI-proxy and mIoU).

Usage:
  python prompt_mi_curve.py \
    --image path/to/image.jpg \
    --gt path/to/gt.(png|jpg|npy) \
    --checkpoint /path/to/sam_vit_h_4b8939.pth \
    --model-type vit_h \
    --iterations 10 \
    --out-dir ./runs/demo1 \
    [--multimask] [--threshold 0.5] [--seed 0]

Requires:
  pip install opencv-python pillow numpy matplotlib torch
  # And SAM from: https://github.com/facebookresearch/segment-anything
"""

import argparse, json, random, csv
from pathlib import Path
from typing import Optional, List, Tuple, Set, Dict

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from PIL import Image

# --- SAM imports ---
try:
    from segment_anything import sam_model_registry, SamPredictor
except Exception as e:
    raise RuntimeError(
        "Could not import SAM. Install from https://github.com/facebookresearch/segment-anything "
        "and ensure it's on PYTHONPATH."
    ) from e


def set_seed(s: int) -> None:
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)


def load_image_rgb(path: str) -> np.ndarray:
    return np.array(Image.open(path).convert("RGB"))


def load_mask_binary(path: str, target_hw: Optional[Tuple[int, int]] = None) -> np.ndarray:
    p = Path(path)
    if p.suffix.lower() == ".npy":
        m = np.load(p)
        if m.ndim > 2:
            m = m[..., 0]
    else:
        m = np.array(Image.open(p))
        if m.ndim > 2:
            m = m[..., 0]
    m = (m != 0).astype(np.uint8)
    if target_hw is not None:
        H, W = target_hw
        if m.shape[:2] != (H, W):
            m = cv2.resize(m, (W, H), interpolation=cv2.INTER_NEAREST)
            m = (m > 0).astype(np.uint8)
    return m


def sample_prompt_pair(fg_xy: np.ndarray, bg_xy: np.ndarray, used: Set[Tuple[int, int]]) -> Tuple[Tuple[int, int], Tuple[int, int]]:
    """Sample one positive (inclusion) and one negative (exclusion) point in (x,y)."""
    # positive
    for _ in range(200):
        pos = tuple(map(int, fg_xy[np.random.randint(0, len(fg_xy))]))
        if pos not in used:
            used.add(pos)
            break
    else:
        pos = tuple(map(int, fg_xy[np.random.randint(0, len(fg_xy))]))
        used.add(pos)
    # negative
    for _ in range(200):
        neg = tuple(map(int, bg_xy[np.random.randint(0, len(bg_xy))]))
        if neg not in used:
            used.add(neg)
            break
    else:
        neg = tuple(map(int, bg_xy[np.random.randint(0, len(bg_xy))]))
        used.add(neg)
    return pos, neg


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def binary_cross_entropy(prob: np.ndarray, gt_bin: np.ndarray, eps: float = 1e-7) -> float:
    """Mean per-pixel CE (natural log). prob in [0,1], gt in {0,1}."""
    p = np.clip(prob, eps, 1.0 - eps).astype(np.float32)
    y = gt_bin.astype(np.float32)
    ce = -(y * np.log(p) + (1 - y) * np.log(1 - p))
    return float(ce.mean())


def miou_from_probs(prob: np.ndarray, gt_bin: np.ndarray, thresh: float = 0.5) -> float:
    pred = (prob >= thresh).astype(np.uint8)
    inter = (pred & gt_bin).sum()
    uni = (pred | gt_bin).sum()
    return 1.0 if uni == 0 else float(inter) / float(uni)


def overlay_mask(image_rgb: np.ndarray, prob_or_mask: np.ndarray, out_path: Path, title: Optional[str] = None) -> None:
    """Save an overlay (heatmap for probs in [0,1], red edges for binary mask)."""
    plt.figure(figsize=(6, 6))
    plt.imshow(image_rgb)
    if prob_or_mask.dtype != np.uint8 and np.issubdtype(prob_or_mask.dtype, np.floating) and prob_or_mask.max() <= 1.0:
        plt.imshow(prob_or_mask, alpha=0.4, cmap="viridis", interpolation="nearest")
        plt.colorbar(fraction=0.046, pad=0.04)
    else:
        edges = cv2.Canny((prob_or_mask * 255).astype(np.uint8), 50, 150) > 0
        edge_rgba = np.zeros((*edges.shape, 4), dtype=np.float32)
        edge_rgba[edges] = [1.0, 0.0, 0.0, 0.9]  # red edges
        plt.imshow(edge_rgba)
    if title:
        plt.title(title)
    plt.axis("off")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(str(out_path), dpi=200)
    plt.close()


def build_predictor(checkpoint: str, model_type: str, device: Optional[str] = None) -> SamPredictor:
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    sam = sam_model_registry[model_type](checkpoint=checkpoint).to(device)
    return SamPredictor(sam)


def upsample_logits_to_image(logits_np: np.ndarray, predictor: SamPredictor, out_hw: Tuple[int, int]) -> np.ndarray:
    """
    Map SAM low-res logits (e.g., 256x256) to original image size (H,W).
    Prefer SAM's postprocess_masks; else fallback to bilinear.
    """
    device = next(predictor.model.parameters()).device
    t = torch.from_numpy(logits_np).float().unsqueeze(0).unsqueeze(0).to(device)  # (1,1,h,w)
    H, W = out_hw
    if hasattr(predictor.model, "postprocess_masks"):
        up = predictor.model.postprocess_masks(
            t, predictor.input_size, predictor.original_size
        )  # (1,1,H,W)
        up = up[0, 0].detach().cpu().numpy()
    else:
        up = F.interpolate(t, size=(H, W), mode="bilinear", align_corners=False)[0, 0].detach().cpu().numpy()
    assert up.shape == (H, W), f"Upscaled logits shape mismatch: {up.shape} vs {(H, W)}"
    return up


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True, help="Path to RGB image (jpg/png)")
    ap.add_argument("--gt", required=True, help="Path to GT mask (jpg/png/npy). Non-zero=foreground.")
    ap.add_argument("--checkpoint", required=True, help="Path to SAM checkpoint .pth")
    ap.add_argument("--model-type", default="vit_h", choices=["vit_h", "vit_l", "vit_b"])
    ap.add_argument("--iterations", type=int, default=10, help="Start with 1 pair; add 1 pair per iteration")
    ap.add_argument("--out-dir", default="./runs/demo")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--threshold", type=float, default=0.5, help="Threshold for mIoU")
    ap.add_argument("--multimask", action="store_true", help="Use SAM's 3-mask output; pick best by score")
    args = ap.parse_args()

    set_seed(args.seed)
    out_dir = Path(args.out_dir)
    (out_dir / "overlays").mkdir(parents=True, exist_ok=True)
    (out_dir / "npy").mkdir(parents=True, exist_ok=True)

    # 1) Load image + GT and force size alignment
    image = load_image_rgb(args.image)
    H, W = image.shape[:2]
    gt = load_mask_binary(args.gt, target_hw=(H, W))
    assert gt.shape == (H, W), "GT must match image size after resize."

    # 2) Build predictor and set image
    predictor = build_predictor(args.checkpoint, args.model_type)
    predictor.set_image(image)
    assert predictor.original_size == (H, W), f"Predictor original_size {predictor.original_size} != {(H,W)}"

    # 3) Prompt pools (in (x,y) coords)
    fg_yx = np.argwhere(gt > 0)
    bg_yx = np.argwhere(gt == 0)
    if len(fg_yx) == 0 or len(bg_yx) == 0:
        raise ValueError("GT must have at least one foreground and one background pixel.")
    fg_xy = fg_yx[:, ::-1]
    bg_xy = bg_yx[:, ::-1]

    # 4) Iterative prompting loop
    points: List[Tuple[int, int]] = []
    labels: List[int] = []  # 1=pos, 0=neg
    used: Set[Tuple[int, int]] = set()
    rows: List[Dict[str, object]] = []

    for it in range(args.iterations):
        # Add exactly one (pos,neg) pair
        pos, neg = sample_prompt_pair(fg_xy, bg_xy, used)
        points.extend([pos, neg])
        labels.extend([1, 0])

        point_coords = np.array(points, dtype=np.float32)
        point_labels = np.array(labels, dtype=np.int32)

        with torch.no_grad():
            masks, scores, low_res_logits = predictor.predict(
                point_coords=point_coords,
                point_labels=point_labels,
                multimask_output=args.multimask,
                return_logits=True
            )

        # Choose a single mask/logits to evaluate
        if args.multimask:
            best = int(np.argmax(scores))
            lr = low_res_logits[best]  # (h,w) logits
        else:
            lr = low_res_logits[0]

        # Upscale logits to EXACT (H,W), get prob
        up_logits = upsample_logits_to_image(lr, predictor, (H, W))  # (H,W)
        prob = sigmoid(up_logits).astype(np.float32)
        assert prob.shape == (H, W)

        # Metrics at (H,W)
        ce = binary_cross_entropy(prob, gt)
        mi_proxy = -ce
        iou = miou_from_probs(prob, gt, thresh=args.threshold)

        # Save overlays and arrays
        overlay_mask(image, prob, out_dir / "overlays" / f"prob_iter_{it:02d}.jpg",
                     title=f"Prob overlay, iter {it} (pairs={it+1})")
        overlay_mask(image, (prob >= args.threshold).astype(np.uint8),
                     out_dir / "overlays" / f"mask_iter_{it:02d}.jpg",
                     title=f"Pred mask (τ={args.threshold}), iter {it} (pairs={it+1})")
        np.save(out_dir / "npy" / f"prob_iter_{it:02d}.npy", prob)
        np.save(out_dir / "npy" / f"logits_iter_{it:02d}.npy", up_logits)

        # Save prompts JSON (all so far)
        prompt_list = [{"x": int(x), "y": int(y), "label": int(l)}
                       for (x, y), l in zip(points, labels)]
        pj = out_dir / f"prompts_iter_{it:02d}.json"
        with open(pj, "w") as f:
            json.dump(prompt_list, f, indent=2)

        rows.append({
            "iteration": it,
            "num_pairs": it + 1,
            "num_points": len(points),
            "cross_entropy_nats": ce,
            "mi_proxy_neg_ce": mi_proxy,
            "miou_thr": args.threshold,
            "miou": iou,
            "prompts_json": pj.name
        })

        print(f"[iter {it:02d}] pairs={it+1}  mIoU={iou:.4f}  CE={ce:.6f}  -CE={mi_proxy:.6f}")

    # 5) Save CSV
    csv_path = out_dir / "log.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    # 6) Curves (two separate subplots)
    its = [r["iteration"] for r in rows]
    mi_proxy_vals = [r["mi_proxy_neg_ce"] for r in rows]
    miou_vals = [r["miou"] for r in rows]

    fig, axes = plt.subplots(2, 1, figsize=(7.2, 7.2), sharex=True)
    ax1, ax2 = axes

    ax1.plot(its, mi_proxy_vals, marker="o")
    ax1.set_ylabel("MI proxy (−CE) ↑ better")
    ax1.set_title("MI proxy vs iteration")

    ax2.plot(its, miou_vals, marker="s")
    ax2.set_xlabel("Iteration (pairs = it+1)")
    ax2.set_ylabel("mIoU (τ={:.2f})".format(args.threshold))
    ax2.set_title("mIoU vs iteration")

    fig.tight_layout()
    fig_path = out_dir / "curves_subplots.png"
    fig.savefig(fig_path, dpi=200)
    plt.close(fig)

    # 7) Summary JSON
    with open(out_dir / "summary.json", "w") as f:
        json.dump({
            "note": (
                "All tensors aligned to exact (H,W). Probabilities/logits saved per iteration. "
                "MI-proxy = -CE (higher means more MI). IoU may fluctuate due to thresholding."
            ),
            "csv": "log.csv",
            "plot_subplots": "curves_subplots.png",
            "overlays_dir": "overlays/",
            "npy_dir": "npy/",
            "iterations": len(rows)
        }, f, indent=2)

    print(f"\nSaved:\n- {csv_path}\n- {fig_path}\n- overlays/ & npy/ (per-iteration)\n- summary.json")


if __name__ == "__main__":
    main()
