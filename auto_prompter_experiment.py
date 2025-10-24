#!/usr/bin/env python3
"""
Auto-Prompter Experiment (SAM) with Single-Seed Filtering

Data layout (required):
  Images/<dataset_name>/sample_000000.jpg
  Masks/<dataset_name>/sample_000000.npy   # binary mask; any nonzero = FG

Outputs:
  <out_dir>/
    single_seed_index.json                     # {dataset: [relative_sample_ids,...]}
    results_per_sample.csv                     # one row per (dataset, sample, iteration)
    aggregated_by_dataset.csv                  # mean/std per dataset per iteration
    perf_normalized_mean.png                   # normalized mean curves (±std if --plot-std)
    perf_raw_mean.png                          # raw mean curves (±std if --plot-std)
    run_args.json                              # args snapshot

Key design choices:
  - "Single seed" = exactly one 8-connected foreground component (no islands).
  - Speed: compute SAM image embedding once per image (predictor.set_image),
           then iterate points-only prompting.
  - "Best mask" = pick the SAM proposal with highest SAM IoU score (not GT).
  - Points are strictly in (x, y) order for SAM.
"""

import argparse
import json
from pathlib import Path
import os
import sys
import math
from typing import Dict, List, Tuple
import numpy as np
import cv2
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt
import pandas as pd

# --- SAM imports ---
# pip install git+https://github.com/facebookresearch/segment-anything.git
try:
    from segment_anything import sam_model_registry, SamPredictor
except Exception as e:
    print("ERROR: Could not import Segment Anything. Install it:\n"
          "  pip install git+https://github.com/facebookresearch/segment-anything.git\n"
          "Also ensure torch+cuda are installed.", file=sys.stderr)
    raise

# ----------------- Utils -----------------

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def load_image_rgb(path: Path) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Failed to read image: {path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img

def load_binary_mask(path: Path) -> np.ndarray:
    """
    Load binary mask (.npy expected). Any nonzero treated as 1 (foreground).
    """
    if path.suffix.lower() == ".npy":
        arr = np.load(str(path))
    else:
        # Fallback: allow .png / .jpg masks (optional)
        arr = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if arr is None:
            raise FileNotFoundError(f"Failed to read mask: {path}")
    if arr.ndim > 2:
        arr = arr.squeeze()
    arr = (arr != 0).astype(np.uint8)
    return arr

def resize_mask_to_image(mask: np.ndarray, target_hw: Tuple[int, int]) -> np.ndarray:
    th, tw = target_hw
    if mask.shape[:2] == (th, tw):
        return mask
    return cv2.resize(mask, (tw, th), interpolation=cv2.INTER_NEAREST).astype(np.uint8)

def single_seed_check(mask_bin: np.ndarray) -> bool:
    """
    True if exactly one 8-connected foreground component exists.
    """
    if mask_bin.max() == 0:
        return False
    num_labels, _ = cv2.connectedComponents(mask_bin.astype(np.uint8), connectivity=8)
    # num_labels includes background label 0; FG components = num_labels - 1
    return (num_labels - 1) == 1

def iou_and_dice(pred_bin: np.ndarray, gt_bin: np.ndarray) -> Tuple[float, float]:
    pred = pred_bin.astype(bool)
    gt = gt_bin.astype(bool)
    inter = np.logical_and(pred, gt).sum()
    union = np.logical_or(pred, gt).sum()
    iou = (inter / union) if union > 0 else 0.0
    denom = pred.sum() + gt.sum()
    dice = (2.0 * inter / denom) if denom > 0 else 0.0
    return float(iou), float(dice)

def find_datasets(images_root: Path) -> List[str]:
    # datasets are subdirectories of Images/
    return sorted([p.name for p in images_root.iterdir() if p.is_dir()])

def list_samples_for_dataset(images_root: Path, dataset: str) -> List[str]:
    # returns list of sample IDs like "sample_000000" (without extension)
    ds_img_dir = images_root / dataset
    ids = []
    for img_path in sorted(ds_img_dir.glob("*.jpg")):
        ids.append(img_path.stem)  # "sample_000000"
    return ids

def build_paths(images_root: Path, masks_root: Path, dataset: str, sample_id: str) -> Tuple[Path, Path]:
    img_path = images_root / dataset / f"{sample_id}.jpg"
    # Primary expectation is .npy; if missing, check common img mask fallback
    mask_path = masks_root / dataset / f"{sample_id}.npy"
    if not mask_path.exists():
        alt_png = masks_root / dataset / f"{sample_id}.png"
        alt_jpg = masks_root / dataset / f"{sample_id}.jpg"
        if alt_png.exists(): mask_path = alt_png
        elif alt_jpg.exists(): mask_path = alt_jpg
    return img_path, mask_path

def sample_points(mask_bin: np.ndarray, num_fg: int, num_bg: int, rng: np.random.Generator) -> Tuple[np.ndarray, np.ndarray]:
    """
    Return arrays of *unique* FG and BG point coords (x,y) (if possible).
    If we run out of unique pixels, we allow replacement.
    """
    h, w = mask_bin.shape
    ys, xs = np.nonzero(mask_bin == 1)
    ysb, xsb = np.nonzero(mask_bin == 0)

    if len(xs) == 0 or len(xsb) == 0:
        # degenerate; caller should skip earlier but guard here
        return np.zeros((0,2), dtype=np.float32), np.zeros((0,2), dtype=np.float32)

    # Choose without replacement if enough pixels; else with replacement
    replace_fg = len(xs) < num_fg
    replace_bg = len(xsb) < num_bg

    idx_fg = rng.choice(len(xs), size=num_fg, replace=replace_fg)
    idx_bg = rng.choice(len(xsb), size=num_bg, replace=replace_bg)

    fg_pts_xy = np.stack([xs[idx_fg], ys[idx_fg]], axis=1).astype(np.float32)  # (x,y)
    bg_pts_xy = np.stack([xsb[idx_bg], ysb[idx_bg]], axis=1).astype(np.float32)  # (x,y)
    return fg_pts_xy, bg_pts_xy

def pick_best_sam_mask(masks: np.ndarray, iou_preds: np.ndarray) -> np.ndarray:
    """
    masks: (M, H, W), iou_preds: (M,)
    Return binary mask for the best predicted IoU.
    """
    best_idx = int(np.argmax(iou_preds))
    return (masks[best_idx] > 0).astype(np.uint8)

# --------------- Core Experiment ----------------

def run_experiment(
    images_root: Path,
    masks_root: Path,
    out_dir: Path,
    checkpoint: Path,
    model_type: str,
    iterations: int,
    device: str,
    seed: int,
    plot_std: bool,
    normalize_to: str,
    normalization_metric: str,
    max_datasets: int,
    max_samples_per_dataset: int,
    multimask: bool
):
    ensure_dir(out_dir)

    # 1) Scan and filter single-seed
    datasets = find_datasets(images_root)
    if max_datasets > 0:
        datasets = datasets[:max_datasets]

    single_seed_index: Dict[str, List[str]] = {ds: [] for ds in datasets}

    print("Scanning for single-seed masks ...")
    for ds in tqdm(datasets, desc="Datasets"):
        sample_ids = list_samples_for_dataset(images_root, ds)
        if max_samples_per_dataset > 0:
            sample_ids = sample_ids[:max_samples_per_dataset]
        for sid in sample_ids:
            img_path, mask_path = build_paths(images_root, masks_root, ds, sid)
            if not img_path.exists() or not mask_path.exists():
                continue
            try:
                # Quick check without loading image (connectedness doesn't depend on image size)
                mask_bin = load_binary_mask(mask_path)
                if single_seed_check(mask_bin):
                    single_seed_index[ds].append(sid)
            except Exception as e:
                print(f"[WARN] Skipping {ds}/{sid}: {e}")

    # Save index
    index_path = out_dir / "single_seed_index.json"
    with open(index_path, "w") as f:
        json.dump(single_seed_index, f, indent=2)
    print(f"Saved single-seed index: {index_path}")

    # 2) Load SAM model once
    print("Loading SAM model...")
    sam = sam_model_registry[model_type](checkpoint=str(checkpoint))
    sam.to(device=device)
    predictor = SamPredictor(sam)

    rng_master = np.random.default_rng(seed)

    # CSV accumulators
    rows = []

    print("Running auto-prompting experiment ...")
    with torch.inference_mode():
        for ds in tqdm(datasets, desc="Experiment by dataset"):
            sids = single_seed_index.get(ds, [])
            if len(sids) == 0:
                continue
            if max_samples_per_dataset > 0:
                sids = sids[:max_samples_per_dataset]

            for sid in tqdm(sids, leave=False, desc=f"{ds} samples"):
                img_path, mask_path = build_paths(images_root, masks_root, ds, sid)
                try:
                    img = load_image_rgb(img_path)
                    gt = load_binary_mask(mask_path)
                    gt = resize_mask_to_image(gt, img.shape[:2])  # (H,W)

                    # Skip degenerate masks (just in case)
                    if gt.max() == 0:
                        continue

                    # Set image ONCE (embedding cached)
                    predictor.set_image(img)

                    # Pre-sample K FG/BG points (we’ll grow prefixes 1..K)
                    # Use a per-sample RNG derived from master for reproducibility:
                    rng = np.random.default_rng(rng_master.integers(0, 2**31 - 1))

                    fg_all, bg_all = sample_points(gt, iterations, iterations, rng)
                    if len(fg_all) == 0 or len(bg_all) == 0:
                        continue

                    for k in range(1, iterations + 1):
                        # points for this iteration: first k from each pool
                        fg_k = fg_all[:k]
                        bg_k = bg_all[:k]
                        coords = np.vstack([fg_k, bg_k])  # shape (2k, 2) in (x,y)
                        labels = np.concatenate([
                            np.ones(len(fg_k), dtype=np.int32),
                            np.zeros(len(bg_k), dtype=np.int32)
                        ])

                        # Predict (no heavy outputs; choose best by SAM's IoU score)
                        masks, iou_preds, _ = predictor.predict(
                            point_coords=coords,
                            point_labels=labels,
                            multimask_output=multimask
                        )
                        if multimask:
                            pred = pick_best_sam_mask(masks, iou_preds)
                        else:
                            pred = (masks[0] > 0).astype(np.uint8)

                        miou, dice = iou_and_dice(pred, gt)

                        rows.append({
                            "dataset": ds,
                            "sample_id": sid,
                            "iteration": k,
                            "n_inclusion": k,
                            "n_exclusion": k,
                            "miou": miou,
                            "dice": dice
                        })

                except Exception as e:
                    print(f"[WARN] Error on {ds}/{sid}: {e}")

    # 3) Save per-sample CSV
    df = pd.DataFrame(rows)
    per_sample_csv = out_dir / "results_per_sample.csv"
    df.to_csv(per_sample_csv, index=False)
    print(f"Saved: {per_sample_csv}")

    if df.empty:
        print("No results collected. Exiting.")
        return

    # 4) Aggregate: mean/std per dataset per iteration
    agg = df.groupby(["dataset", "iteration"]).agg(
        miou_mean=("miou", "mean"),
        miou_std=("miou", "std"),
        dice_mean=("dice", "mean"),
        dice_std=("dice", "std"),
        n=("miou", "count")
    ).reset_index()

    # Normalization per dataset (divide by dataset-specific max across iterations)
    norm_target = f"{normalization_metric}_mean"  # "miou_mean" or "dice_mean"
    agg["norm_denom"] = agg.groupby("dataset")[norm_target].transform(lambda s: s.max() if s.max() > 0 else 1.0)

    if normalize_to == "max":
        agg["normalized"] = agg[norm_target] / agg["norm_denom"].replace(0, np.nan)
        # propagate std normalization consistently
        std_col = f"{normalization_metric}_std"
        agg["normalized_std"] = agg[std_col] / agg["norm_denom"].replace(0, np.nan)
    elif normalize_to == "range":
        # (x - min) / (max - min)
        g = agg.groupby("dataset")[norm_target]
        mins = g.transform("min")
        maxs = g.transform("max")
        denom = (maxs - mins).replace(0, np.nan)
        agg["normalized"] = (agg[norm_target] - mins) / denom
        std_col = f"{normalization_metric}_std"
        agg["normalized_std"] = agg[std_col] / denom
    else:
        raise ValueError("--normalize-to must be 'max' or 'range'")

    agg_csv = out_dir / "aggregated_by_dataset.csv"
    agg.to_csv(agg_csv, index=False)
    print(f"Saved: {agg_csv}")

    # 5) Plots
    def plot_curves(y_field: str, y_std_field: str, title: str, outfile: Path):
        plt.figure(figsize=(9, 6))
        for ds, sub in agg.groupby("dataset"):
            xs = sub["iteration"].values
            ys = sub[y_field].values
            plt.plot(xs, ys, label=ds, linewidth=2)
            if plot_std and y_std_field in sub.columns:
                ys_std = sub[y_std_field].values
                if np.all(np.isfinite(ys_std)):
                    plt.fill_between(xs, ys - ys_std, ys + ys_std, alpha=0.2)

        plt.xlabel("Iteration (pairs of +/− prompts)")
        plt.ylabel(y_field.replace("_", " ").title())
        plt.title(title)
        plt.grid(True, alpha=0.3)
        plt.legend(loc="best", frameon=False)
        plt.tight_layout()
        plt.savefig(outfile, dpi=160)
        plt.close()

    # Normalized curves (primary figure)
    plot_curves(
        y_field="normalized",
        y_std_field="normalized_std",
        title=f"Normalized {normalization_metric.upper()} vs Iteration (per dataset)",
        outfile=out_dir / "perf_normalized_mean.png",
    )

    # Raw mean curves (secondary)
    y_field_raw = f"{normalization_metric}_mean"
    y_std_raw  = f"{normalization_metric}_std"
    plot_curves(
        y_field=y_field_raw,
        y_std_field=y_std_raw,
        title=f"Raw {normalization_metric.upper()} vs Iteration (per dataset)",
        outfile=out_dir / "perf_raw_mean.png",
    )

    # 6) Save args snapshot
    args_snapshot = {
        "images_root": str(images_root),
        "masks_root": str(masks_root),
        "out_dir": str(out_dir),
        "checkpoint": str(checkpoint),
        "model_type": model_type,
        "iterations": iterations,
        "device": device,
        "seed": seed,
        "plot_std": plot_std,
        "normalize_to": normalize_to,
        "normalization_metric": normalization_metric,
        "max_datasets": max_datasets,
        "max_samples_per_dataset": max_samples_per_dataset,
        "multimask": multimask
    }
    with open(out_dir / "run_args.json", "w") as f:
        json.dump(args_snapshot, f, indent=2)

    print("\nDone.")
    print(f"- Normalized plot : {out_dir/'perf_normalized_mean.png'}")
    print(f"- Raw plot        : {out_dir/'perf_raw_mean.png'}")
    print(f"- Aggregates CSV  : {agg_csv}")
    print(f"- Per-sample CSV  : {per_sample_csv}")
    print(f"- Single-seed JSON: {index_path}")

# --------------- CLI ----------------

def parse_args():
    p = argparse.ArgumentParser(description="Auto-Prompter Experiment for SAM (with single-seed filtering)")
    p.add_argument("--images-root", type=Path, required=True, help="Root: Images/")
    p.add_argument("--masks-root", type=Path, required=True, help="Root: Masks/")
    p.add_argument("--out-dir", type=Path, required=True, help="Output directory")

    p.add_argument("--checkpoint", type=Path, required=True, help="Path to SAM checkpoint .pth")
    p.add_argument("--model-type", type=str, default="vit_h", choices=["vit_h", "vit_l", "vit_b"], help="SAM model type")
    p.add_argument("--iterations", type=int, default=100, help="Max iterations (pairs of +/− prompts)")
    p.add_argument("--seed", type=int, default=123, help="Master RNG seed")

    p.add_argument("--cpu", action="store_true", help="Force CPU")
    p.add_argument("--plot-std", action="store_true", help="Shade ±std on curves")
    p.add_argument("--normalize-to", type=str, default="max", choices=["max", "range"], help="Normalization scheme across iterations per dataset")
    p.add_argument("--normalization-metric", type=str, default="miou", choices=["miou", "dice"], help="Which metric to normalize/plot by default")

    p.add_argument("--max-datasets", type=int, default=0, help="Limit number of datasets (0=no limit)")
    p.add_argument("--max-samples-per-dataset", type=int, default=0, help="Limit number of samples per dataset (0=no limit)")
    p.add_argument("--singlemask-multimask", dest="multimask", action="store_true",
                   help="Use SAM multimask_output=True and pick best by SAM IoU score (default True)")
    p.add_argument("--no-multimask", dest="multimask", action="store_false",
                   help="Use multimask_output=False (faster, but no 'best-of-3')")
    p.set_defaults(multimask=True)

    args = p.parse_args()
    return args

def main():
    args = parse_args()

    device = "cpu" if args.cpu or not torch.cuda.is_available() else "cuda"
    print(f"Device: {device}")

    run_experiment(
        images_root=args.images_root,
        masks_root=args.masks_root,
        out_dir=args.out_dir,
        checkpoint=args.checkpoint,
        model_type=args.model_type,
        iterations=args.iterations,
        device=device,
        seed=args.seed,
        plot_std=args.plot_std,
        normalize_to=args.normalize_to,
        normalization_metric=args.normalization_metric,
        max_datasets=args.max_datasets,
        max_samples_per_dataset=args.max_samples_per_dataset,
        multimask=args.multimask
    )

if __name__ == "__main__":
    main()
