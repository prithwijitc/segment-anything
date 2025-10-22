#!/usr/bin/env python3
"""
Robust MI check for SAM prompts:
- Estimates conditional MI increments I(Y; C | X, P_t) by averaging CE over M random extra prompt pairs.
- Plots cumulative MI estimate (sum of increments) vs iteration, and base mIoU vs iteration.

Usage:
  python prompt_mi_cmi_curve.py \
    --image path/to/image.jpg \
    --gt path/to/gt.(png|jpg|npy) \
    --checkpoint /path/to/sam_vit_h_4b8939.pth \
    --model-type vit_h \
    --iterations 10 \
    --samples-per-iter 16 \
    --out-dir ./runs/robust_demo \
    [--multimask] [--threshold 0.5] [--seed 0] [--fit-temp-samples 0]

Notes:
- MI here is approximated via CE differences; in expectation it is monotone in the number of prompts.
- Temperature scaling (optional) improves faithfulness of CE to true conditional entropy.
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
        "Install SAM from https://github.com/facebookresearch/segment-anything "
        "and ensure it's on PYTHONPATH."
    ) from e


# -------------------- utils --------------------

def set_seed(s: int) -> None:
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)

def load_image_rgb(path: str) -> np.ndarray:
    return np.array(Image.open(path).convert("RGB"))

def load_mask_binary(path: str, target_hw: Optional[Tuple[int,int]]=None) -> np.ndarray:
    p = Path(path)
    if p.suffix.lower() == ".npy":
        m = np.load(p)
        if m.ndim > 2: m = m[...,0]
    else:
        m = np.array(Image.open(p))
        if m.ndim > 2: m = m[...,0]
    m = (m != 0).astype(np.uint8)
    if target_hw is not None:
        H, W = target_hw
        if m.shape[:2] != (H, W):
            m = cv2.resize(m, (W, H), interpolation=cv2.INTER_NEAREST)
            m = (m > 0).astype(np.uint8)
    return m

def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))

def binary_cross_entropy(prob: np.ndarray, gt_bin: np.ndarray, eps: float=1e-7) -> float:
    p = np.clip(prob, eps, 1.0 - eps).astype(np.float32)
    y = gt_bin.astype(np.float32)
    ce = -(y * np.log(p) + (1 - y) * np.log(1 - p))
    return float(ce.mean())

def miou_from_probs(prob: np.ndarray, gt_bin: np.ndarray, thresh: float=0.5) -> float:
    pred = (prob >= thresh).astype(np.uint8)
    inter = (pred & gt_bin).sum()
    uni = (pred | gt_bin).sum()
    return 1.0 if uni == 0 else float(inter) / float(uni)

def overlay_mask(image_rgb: np.ndarray, prob_or_mask: np.ndarray, out_path: Path, title: Optional[str]=None) -> None:
    plt.figure(figsize=(6,6))
    plt.imshow(image_rgb)
    if prob_or_mask.dtype != np.uint8 and np.issubdtype(prob_or_mask.dtype, np.floating) and prob_or_mask.max() <= 1.0:
        plt.imshow(prob_or_mask, alpha=0.4, cmap="viridis", interpolation="nearest")
        plt.colorbar(fraction=0.046, pad=0.04)
    else:
        edges = cv2.Canny((prob_or_mask * 255).astype(np.uint8), 50, 150) > 0
        edge_rgba = np.zeros((*edges.shape, 4), dtype=np.float32)
        edge_rgba[edges] = [1.0, 0.0, 0.0, 0.9]
        plt.imshow(edge_rgba)
    if title: plt.title(title)
    plt.axis("off"); out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout(); plt.savefig(str(out_path), dpi=200); plt.close()

# -------------------- SAM helpers --------------------

def build_predictor(checkpoint: str, model_type: str, device: Optional[str]=None) -> SamPredictor:
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    sam = sam_model_registry[model_type](checkpoint=checkpoint).to(device)
    return SamPredictor(sam)

def upsample_logits_to_image(logits_np: np.ndarray, predictor: SamPredictor, out_hw: Tuple[int,int]) -> np.ndarray:
    device = next(predictor.model.parameters()).device
    t = torch.from_numpy(logits_np).float().unsqueeze(0).unsqueeze(0).to(device)  # (1,1,h,w)
    H, W = out_hw
    if hasattr(predictor.model, "postprocess_masks"):
        up = predictor.model.postprocess_masks(t, predictor.input_size, predictor.original_size)[0,0]
    else:
        up = F.interpolate(t, size=(H,W), mode="bilinear", align_corners=False)[0,0]
    return up.detach().cpu().numpy()

# -------------------- prompt sampling --------------------

def sample_prompt_pair(fg_xy: np.ndarray, bg_xy: np.ndarray, used: Set[Tuple[int,int]]) -> Tuple[Tuple[int,int], Tuple[int,int]]:
    # positive
    for _ in range(200):
        pos = tuple(map(int, fg_xy[np.random.randint(0, len(fg_xy))]))
        if pos not in used:
            used.add(pos); break
    else:
        pos = tuple(map(int, fg_xy[np.random.randint(0, len(fg_xy))])); used.add(pos)
    # negative
    for _ in range(200):
        neg = tuple(map(int, bg_xy[np.random.randint(0, len(bg_xy))]))
        if neg not in used:
            used.add(neg); break
    else:
        neg = tuple(map(int, bg_xy[np.random.randint(0, len(bg_xy))])); used.add(neg)
    return pos, neg

def sample_extra_pair(fg_xy: np.ndarray, bg_xy: np.ndarray, forbidden: Set[Tuple[int,int]]) -> Tuple[Tuple[int,int], Tuple[int,int]]:
    # Like above but does not mutate 'forbidden'
    for _ in range(200):
        pos = tuple(map(int, fg_xy[np.random.randint(0, len(fg_xy))]))
        if pos not in forbidden: break
    for _ in range(200):
        neg = tuple(map(int, bg_xy[np.random.randint(0, len(bg_xy))]))
        if neg not in forbidden and neg != pos: break
    return pos, neg

# -------------------- temperature scaling (optional) --------------------

def fit_temperature_quick(predictor: SamPredictor,
                          image_rgb: np.ndarray,
                          H: int, W: int,
                          gt: np.ndarray,
                          fg_xy: np.ndarray, bg_xy: np.ndarray,
                          trials: int = 40) -> float:
    """
    Quick-and-dirty binary temperature scaling using random prompt sets on the same image.
    Returns a scalar T >= 0 applied as logits/T.
    """
    device = next(predictor.model.parameters()).device
    # Collect some (logits, gt) samples
    logits_list: List[np.ndarray] = []
    with torch.no_grad():
        for _ in range(trials):
            # random 1-3 pairs
            k = np.random.randint(1, 4)
            pts, labs, used = [], [], set()
            for _ in range(k):
                p, n = sample_prompt_pair(fg_xy, bg_xy, used)
                pts.extend([p, n]); labs.extend([1, 0])
            point_coords = np.array(pts, np.float32); point_labels = np.array(labs, np.int32)
            masks, scores, lrs = predictor.predict(point_coords=point_coords,
                                                   point_labels=point_labels,
                                                   multimask_output=False,
                                                   return_logits=True)
            lr = lrs[0]
            up = upsample_logits_to_image(lr, predictor, (H, W))
            logits_list.append(up.astype(np.float32))
    logits = np.stack(logits_list, axis=0)   # (T, H, W)
    y = np.broadcast_to(gt[None, ...].astype(np.float32), logits.shape)

    # Optimize T by line search on CE
    def ce_at_T(T: float) -> float:
        pr = 1.0 / (1.0 + np.exp(-(logits / max(T, 1e-6))))
        return binary_cross_entropy(pr, y)

    Ts = np.exp(np.linspace(np.log(0.25), np.log(4.0), 25))
    vals = [ce_at_T(float(T)) for T in Ts]
    T_best = float(Ts[int(np.argmin(vals))])
    return T_best

def apply_temperature(logits: np.ndarray, T: float) -> np.ndarray:
    return logits / max(T, 1e-6)

# -------------------- main --------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True)
    ap.add_argument("--gt", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--model-type", default="vit_h", choices=["vit_h","vit_l","vit_b"])
    ap.add_argument("--iterations", type=int, default=10)
    ap.add_argument("--samples-per-iter", type=int, default=16, help="M: extra prompt pairs sampled per iteration to estimate expected CE")
    ap.add_argument("--out-dir", default="./runs/robust_demo")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--multimask", action="store_true")
    ap.add_argument("--fit-temp-samples", type=int, default=0, help="If >0, fit a scalar temperature using this many random prompt trials on this image")
    args = ap.parse_args()

    set_seed(args.seed)
    out_dir = Path(args.out_dir)
    (out_dir/"overlays").mkdir(parents=True, exist_ok=True)
    (out_dir/"npy").mkdir(parents=True, exist_ok=True)

    # Load data
    image = load_image_rgb(args.image)
    H, W = image.shape[:2]
    gt = load_mask_binary(args.gt, target_hw=(H, W))
    assert gt.shape == (H, W)

    # Predictor
    predictor = build_predictor(args.checkpoint, args.model_type)
    predictor.set_image(image)
    assert predictor.original_size == (H, W)

    # Pools
    fg_yx = np.argwhere(gt > 0)
    bg_yx = np.argwhere(gt == 0)
    if len(fg_yx) == 0 or len(bg_yx) == 0:
        raise ValueError("GT must include ≥1 foreground and ≥1 background pixel.")
    fg_xy = fg_yx[:, ::-1]; bg_xy = bg_yx[:, ::-1]

    # Optional temperature fit
    T = 1.0
    if args.fit_temp_samples > 0:
        print(f"[calib] fitting temperature with {args.fit_temp_samples} trials ...")
        T = fit_temperature_quick(predictor, image, H, W, gt, fg_xy, bg_xy, trials=args.fit_temp_samples)
        print(f"[calib] best temperature T ≈ {T:.3f}")

    # Iterative base prompt construction
    base_points: List[Tuple[int,int]] = []
    base_labels: List[int] = []
    used: Set[Tuple[int,int]] = set()

    rows: List[Dict[str, object]] = []
    cum_mi_est = 0.0

    for it in range(args.iterations):
        # Add exactly one (pos,neg) to the *base* set
        pos, neg = sample_prompt_pair(fg_xy, bg_xy, used)
        base_points.extend([pos, neg]); base_labels.extend([1, 0])

        # ---- Base CE/mIoU with P_t ----
        point_coords = np.array(base_points, np.float32)
        point_labels = np.array(base_labels, np.int32)
        with torch.no_grad():
            masks, scores, lrs = predictor.predict(
                point_coords=point_coords,
                point_labels=point_labels,
                multimask_output=args.multimask,
                return_logits=True
            )
        lr = lrs[int(np.argmax(scores))] if args.multimask else lrs[0]
        up_logits = upsample_logits_to_image(lr, predictor, (H, W))
        if T != 1.0: up_logits = apply_temperature(up_logits, T)
        prob_base = sigmoid(up_logits).astype(np.float32)
        ce_base = binary_cross_entropy(prob_base, gt)
        miou_base = miou_from_probs(prob_base, gt, thresh=args.threshold)

        # ---- Expected CE after adding one extra random pair C ~ pi(C | X, P_t) ----
        ce_list = []
        miou_list = []
        for m in range(args.samples_per_iter):
            # sample an extra pair that doesn't duplicate base points
            p_extra, n_extra = sample_extra_pair(fg_xy, bg_xy, set(base_points))
            pts_aug = np.concatenate([point_coords, np.array([p_extra, n_extra], np.float32)], axis=0)
            labs_aug = np.concatenate([point_labels, np.array([1, 0], np.int32)], axis=0)

            with torch.no_grad():
                _, scores_m, lrs_m = predictor.predict(
                    point_coords=pts_aug,
                    point_labels=labs_aug,
                    multimask_output=args.multimask,
                    return_logits=True
                )
            lr_m = lrs_m[int(np.argmax(scores_m))] if args.multimask else lrs_m[0]
            up_m = upsample_logits_to_image(lr_m, predictor, (H, W))
            if T != 1.0: up_m = apply_temperature(up_m, T)
            prob_m = sigmoid(up_m).astype(np.float32)

            ce_m = binary_cross_entropy(prob_m, gt)
            iou_m = miou_from_probs(prob_m, gt, thresh=args.threshold)
            ce_list.append(ce_m); miou_list.append(iou_m)

        ce_after_mean = float(np.mean(ce_list))
        ce_after_std  = float(np.std(ce_list, ddof=1)) if len(ce_list) > 1 else 0.0
        iou_after_mean = float(np.mean(miou_list))

        # Conditional MI increment estimate (nats)
        mi_inc = ce_base - ce_after_mean      # >= 0 in expectation
        cum_mi_est += mi_inc

        # Save overlays for the base at this iteration
        overlay_mask(image, prob_base, out_dir/"overlays"/f"prob_base_iter_{it:02d}.jpg",
                     title=f"Base prob, iter {it} (pairs={it+1})")
        overlay_mask(image, (prob_base >= args.threshold).astype(np.uint8),
                     out_dir/"overlays"/f"mask_base_iter_{it:02d}.jpg",
                     title=f"Base mask (τ={args.threshold}), iter {it}")

        # Save npy
        np.save(out_dir/"npy"/f"logits_base_iter_{it:02d}.npy", up_logits)
        np.save(out_dir/"npy"/f"prob_base_iter_{it:02d}.npy", prob_base)

        rows.append({
            "iteration": it,
            "num_pairs": it + 1,
            "num_points": len(base_points),
            "base_ce_nats": ce_base,
            "expected_ce_after_extra": ce_after_mean,
            "expected_ce_after_std": ce_after_std,
            "mi_increment_nats": mi_inc,
            "mi_cumulative_nats": cum_mi_est,
            "base_miou": miou_base,
            "expected_miou_after_extra": iou_after_mean,
            "temperature": T
        })

        print(f"[iter {it:02d}] pairs={it+1}  base mIoU={miou_base:.4f}  "
              f"CE_base={ce_base:.5f}  E[CE_after]={ce_after_mean:.5f}  "
              f"I_hat={mi_inc:.5f}  CumMI={cum_mi_est:.5f}")

    # Save CSV
    csv_path = out_dir/"log_robust.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader(); writer.writerows(rows)

    # Plot: cumulative MI (with per-iter error bars for E[CE_after]) and base mIoU
    its = [r["iteration"] for r in rows]
    cum_mi = [r["mi_cumulative_nats"] for r in rows]
    base_miou = [r["base_miou"] for r in rows]
    ce_std = [r["expected_ce_after_std"] for r in rows]

    fig, axes = plt.subplots(2, 1, figsize=(7.5, 7.5), sharex=True)
    ax1, ax2 = axes

    ax1.plot(its, cum_mi, marker="o")
    ax1.set_ylabel("Cumulative MI estimate (nats)")
    ax1.set_title("Prompt additions: cumulative MI (expected)")

    # Visualize uncertainty of the increment via std of CE_after (optional cue)
    for t, cm, s in zip(its, cum_mi, ce_std):
        ax1.errorbar([t], [cm], yerr=[[s],[s]], fmt="none", ecolor="gray", alpha=0.5)

    ax2.plot(its, base_miou, marker="s")
    ax2.set_xlabel("Iteration (pairs = it+1)")
    ax2.set_ylabel("Base mIoU (τ=0.5)")
    ax2.set_title("Base mIoU vs iteration")

    fig.tight_layout()
    fig_path = out_dir/"curves_robust_subplots.png"
    fig.savefig(fig_path, dpi=200); plt.close(fig)

    # Summary
    with open(out_dir/"summary.json", "w") as f:
        json.dump({
            "note": (
                "Cumulative MI is sum_t [ CE(P_t) - E_C CE(P_t + C) ]. "
                "Expectation over C is approximated via Monte Carlo with M=samples-per-iter. "
                "Optionally applies scalar temperature scaling to logits to reduce model mismatch."
            ),
            "csv": "log_robust.csv",
            "plot_subplots": "curves_robust_subplots.png",
            "overlays_dir": "overlays/",
            "npy_dir": "npy/"
        }, f, indent=2)

    print(f"\nSaved:\n- {csv_path}\n- {fig_path}\n- overlays/ & npy/\n- summary.json")


if __name__ == "__main__":
    main()
