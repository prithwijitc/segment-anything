#!/usr/bin/env python3
import argparse
from pathlib import Path

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.cm as cm

import torch

# pip install git+https://github.com/facebookresearch/segment-anything.git
from segment_anything import sam_model_registry, SamPredictor


def load_image_rgb(path):
    return np.asarray(Image.open(path).convert("RGB"))


def disk_mask(h, w, cx, cy, r):
    """Return a boolean disk mask of radius r centered at (cx,cy) in image coords."""
    yy, xx = np.ogrid[:h, :w]
    return (xx - cx) ** 2 + (yy - cy) ** 2 <= r ** 2


def run_sam_topmask(predictor: SamPredictor, pts_xy, pts_lbl):
    """
    Return top scoring binary mask (H,W) for given points/labels.
    """
    masks, scores, lowres = predictor.predict(
        point_coords=np.asarray(pts_xy, dtype=np.float32),
        point_labels=np.asarray(pts_lbl, dtype=np.int32),
        multimask_output=True,  # get 3 candidates; we'll take top-1 by predicted IoU
    )
    if masks.ndim == 3 and len(scores) > 0:
        k = int(np.argmax(scores))
        return masks[k].astype(np.float32), float(scores[k])
    # Fallback: no mask (shouldn't happen if prompts exist)
    H, W = predictor.original_size
    return np.zeros((H, W), dtype=np.float32), 0.0


def overlay_delta(bg_rgb, delta, alpha=0.55):
    """
    Overlay a signed delta map on the image.
    delta in {-1,0,1} or real; uses a diverging colormap (blue->white->red).
    """
    H, W = bg_rgb.shape[:2]
    base = bg_rgb.astype(np.float32) / 255.0

    # Normalize to [-1,1] for display (safe if {-1,0,1})
    d = delta.astype(np.float32)
    if d.max() == d.min():
        norm = np.zeros_like(d, dtype=np.float32)
    else:
        m = np.max(np.abs(d))
        norm = (d / (m + 1e-8) + 1.0) * 0.5  # [-1,1] -> [0,1]

    colored = cm.get_cmap("bwr")(np.clip(norm, 0.0, 1.0))[..., :3]  # drop alpha
    out = (1 - alpha) * base + alpha * colored
    return np.clip(out, 0, 1)


def main():
    ap = argparse.ArgumentParser(description="Analyze cross-prompt influence in SAM via leave-one-out deltas.")
    ap.add_argument("--image", required=True, help="Path to image (RGB)")
    ap.add_argument("--prompts", required=True, help="JSON with positive_points/negative_points")
    ap.add_argument("--checkpoint", required=True, help="SAM checkpoint (e.g., sam_vit_h_4b8939.pth)")
    ap.add_argument("--model-type", default="vit_h", choices=["vit_h", "vit_l", "vit_b"], help="SAM model type")
    ap.add_argument("--radius", type=int, default=40, help="Neighborhood radius (pixels) around each prompt to measure influence")
    ap.add_argument("--out-dir", default="prompt_influence_out", help="Folder to save results")
    ap.add_argument("--cpu", action="store_true", help="Force CPU")
    ap.add_argument("--alpha", type=float, default=0.55, help="Overlay alpha for delta maps")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- Load inputs
    import json
    img = load_image_rgb(args.image)
    H, W = img.shape[:2]
    with open(args.prompts, "r") as f:
        prom = json.load(f)

    pos = np.array(prom.get("positive_points", []), dtype=np.float32) if prom.get("positive_points") else np.zeros((0, 2), np.float32)
    neg = np.array(prom.get("negative_points", []), dtype=np.float32) if prom.get("negative_points") else np.zeros((0, 2), np.float32)
    pts = np.concatenate([pos, neg], axis=0)
    lbl = np.array([1] * len(pos) + [0] * len(neg), dtype=np.int32)

    if len(pts) == 0:
        raise ValueError("No prompts in JSON.")

    # --- Build predictor
    device = "cpu" if args.cpu or not torch.cuda.is_available() else "cuda"
    sam = sam_model_registry[args.model_type](checkpoint=args.checkpoint).to(device).eval()
    predictor = SamPredictor(sam)
    predictor.set_image(img)

    # --- Baseline with ALL prompts
    mask_all, score_all = run_sam_topmask(predictor, pts, lbl)  # (H,W) float {0,1}
    print(f"[BASELINE] top IoU score={score_all:.3f}")

    # Precompute disk masks around each prompt (in original pixel space)
    disks = []
    for i in range(len(pts)):
        cx, cy = float(pts[i, 0]), float(pts[i, 1])
        # clamp to image just in case
        cx = np.clip(cx, 0, W - 1)
        cy = np.clip(cy, 0, H - 1)
        disks.append(disk_mask(H, W, cx, cy, args.radius))

    N = len(pts)
    infl = np.zeros((N, N), dtype=np.float32)  # rows: influencer p; cols: neighborhood near q
    inc_frac = np.zeros((N, N), dtype=np.float32)  # fraction of pixels turned 0->1
    dec_frac = np.zeros((N, N), dtype=np.float32)  # fraction of pixels turned 1->0

    # --- Leave-one-out for each prompt p
    for p in range(N):
        keep_idx = [i for i in range(N) if i != p]
        pts_wo = pts[keep_idx]
        lbl_wo = lbl[keep_idx]
        mask_wo, score_wo = run_sam_topmask(predictor, pts_wo, lbl_wo)

        # delta: +1 where p causes inclusion, -1 where p prevents inclusion
        delta = mask_all - mask_wo  # {-1,0,1}

        # Save overlay for this p
        ov = overlay_delta(img, delta, alpha=args.alpha)
        fig, ax = plt.subplots(1, 1, figsize=(10, 10 * H / W))
        ax.imshow(ov); ax.set_axis_off()
        # plot all prompts; highlight the omitted one
        for i in range(N):
            x, y = pts[i, 0], pts[i, 1]
            col = "yellow" if i == p else ("lime" if lbl[i] == 1 else "red")
            ax.scatter([x], [y], c=col, s=40, marker="o", edgecolor="black", linewidths=0.6)
        ax.set_title(f"Δ mask when removing prompt p={p}  (red:+ include, blue:− include)")
        plt.tight_layout()
        out_p = out_dir / f"delta_remove_p{p:02d}.png"
        fig.savefig(out_p, dpi=200, bbox_inches="tight"); plt.close(fig)

        # Per-neighborhood metrics near each q
        for q in range(N):
            m = disks[q]
            area = float(m.sum())
            if area == 0:
                continue
            sub = delta[m]  # values in {-1,0,1}
            infl[p, q] = sub.mean()  # signed mean in that neighborhood
            inc_frac[p, q] = (sub > 0).mean()  # fraction of pixels flipped 0->1
            dec_frac[p, q] = (sub < 0).mean()  # fraction of pixels flipped 1->0

        print(f"[p={p}] IoU wo p={score_wo:.3f} | mean |Δ|={np.mean(np.abs(delta)):.4f}")

    # --- Save matrices as heatmaps + CSV
    def save_heatmap(M, title, fname, vmin=None, vmax=None, cmap="bwr"):
        fig, ax = plt.subplots(figsize=(max(6, N * 0.5), max(5, N * 0.5)))
        im = ax.imshow(M, cmap=cmap, vmin=vmin, vmax=vmax)
        ax.set_xlabel("Region near prompt q")
        ax.set_ylabel("Influencer prompt p (removed)")
        ax.set_title(title)
        ax.set_xticks(np.arange(N)); ax.set_yticks(np.arange(N))
        # colorbar
        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        plt.tight_layout()
        fig.savefig(out_dir / fname, dpi=200, bbox_inches="tight"); plt.close(fig)

    # Influence: signed mean in neighborhood
    vmax_abs = np.max(np.abs(infl)) + 1e-8
    save_heatmap(infl, "Signed influence (mean Δ in neighborhood)", "influence_signed.png",
                 vmin=-vmax_abs, vmax=vmax_abs, cmap="bwr")

    # Fractions 0->1 and 1->0
    save_heatmap(inc_frac, "Fraction of pixels turned 0→1 by prompt p (near q)", "influence_inc_frac.png",
                 vmin=0.0, vmax=1.0, cmap="Reds")
    save_heatmap(dec_frac, "Fraction of pixels turned 1→0 by prompt p (near q)", "influence_dec_frac.png",
                 vmin=0.0, vmax=1.0, cmap="Blues")

    # CSV dump
    import csv
    with open(out_dir / "influence_signed.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["p\\q"] + [f"q{j}" for j in range(N)])
        for i in range(N): w.writerow([f"p{i}"] + list(map(lambda x: f"{x:.6f}", infl[i].tolist())))
    with open(out_dir / "influence_inc_frac.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["p\\q"] + [f"q{j}" for j in range(N)])
        for i in range(N): w.writerow([f"p{i}"] + list(map(lambda x: f"{x:.6f}", inc_frac[i].tolist())))
    with open(out_dir / "influence_dec_frac.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["p\\q"] + [f"q{j}" for j in range(N)])
        for i in range(N): w.writerow([f"p{i}"] + list(map(lambda x: f"{x:.6f}", dec_frac[i].tolist())))

    print(f"[DONE] Saved results to: {out_dir.resolve()}")
    print("  - influence_signed.png / .csv (mean Δ near q; red=adds, blue=removes)")
    print("  - influence_inc_frac.png (fraction 0→1)")
    print("  - influence_dec_frac.png (fraction 1→0)")
    print("  - delta_remove_pXX.png overlays per prompt")
    

if __name__ == "__main__":
    main()
