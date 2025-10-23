#!/usr/bin/env python3
"""
Robust temperature calibration for SAM on ONE image + GT, with verification.

What it does
------------
1) Loads the image and GT mask (same HxW; nonzero=FG).
2) Runs SAM over all JSON prompt files in:
     --calib-dir  (e.g., 200 files)   -> caches upsampled logits to disk
     --eval-dir   (e.g.,  20 files)   -> caches upsampled logits to disk
   For each JSON, uses SAM multimask candidates and stores their upsampled logits.
3) Fits a scalar temperature T by minimizing the **trimmed mean (10%)** cross-entropy
   across calibration files; applies the **1-SE rule** and picks the **largest T**
   within 1 standard error of the minimum (safer calibration).
4) Verifies T:
   - CE vs T curve (trimmed mean) + vertical line at T*
   - ECE & reliability curves at T=1 and T* (calibration + holdout)
   - Holdout CE/mIoU at T=1 vs T*, plus histogram of per-file CE deltas
5) Saves: calibration.json (T*, metrics), per-file CSVs, and plots.

Usage
-----
python robust_calibrate_T.py \
  --image path/to/image.png \
  --gt path/to/mask.png \
  --checkpoint /path/to/sam_vit_h_4b8939.pth \
  --model-type vit_h \
  --calib-dir ./out_prompts_strict/calib \
  --eval-dir  ./out_prompts_strict/eval \
  --multimask \
  --T-lo 0.3 --T-hi 3.0 --grid 61 --rounds 3 \
  --out-dir ./runs/calib_one_robust \
  --seed 0

Requirements
------------
Python 3.8+, PyTorch, numpy, Pillow, OpenCV, matplotlib, and the official SAM repo:
  https://github.com/facebookresearch/segment-anything
"""

import argparse, json, csv, os, math
from pathlib import Path
from typing import List, Tuple, Dict, Optional

import numpy as np
import cv2
from PIL import Image
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

# ---- SAM ----
try:
    from segment_anything import sam_model_registry, SamPredictor
except Exception as e:
    raise RuntimeError("Install SAM: https://github.com/facebookresearch/segment-anything") from e


# ---------------- I/O helpers ----------------

def load_image_rgb(path: str) -> np.ndarray:
    return np.array(Image.open(path).convert("RGB"))

def load_mask_binary(path: str, target_hw: Optional[Tuple[int,int]]=None) -> np.ndarray:
    m = np.array(Image.open(path))
    if m.ndim > 2: m = m[...,0]
    m = (m != 0).astype(np.uint8)
    if target_hw and m.shape[:2] != target_hw:
        H, W = target_hw
        m = cv2.resize(m, (W, H), interpolation=cv2.INTER_NEAREST)
        m = (m > 0).astype(np.uint8)
    return m

def load_prompts_json(path: str) -> Tuple[np.ndarray, np.ndarray]:
    with open(path, "r") as f:
        data = json.load(f)
    arr = data.get("prompts", [])
    arr = sorted(arr, key=lambda d: d.get("t", 0))
    coords = np.array([[int(d["x"]), int(d["y"])] for d in arr], dtype=np.float32)
    labels = np.array([int(d["label"]) for d in arr], dtype=np.int32)
    # sanity (equal + and -)
    npos = int((labels==1).sum()); nneg = int((labels==0).sum())
    assert npos == nneg, f"{path}: positives ({npos}) != negatives ({nneg})"
    return coords, labels


# ---------------- metrics ----------------

def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))

def ce_from_logits_T(logits: np.ndarray, gt_bin: np.ndarray, T: float) -> float:
    p = sigmoid(logits.astype(np.float32) / max(T, 1e-6))
    p = np.clip(p, 1e-7, 1-1e-7)
    y = gt_bin.astype(np.float32)
    ce = -(y*np.log(p) + (1-y)*np.log(1-p)).mean()  # nats
    return float(ce)

def hard_miou_from_logits_T(logits: np.ndarray, gt_bin: np.ndarray, T: float, thresh: float=0.5) -> float:
    p = sigmoid(logits.astype(np.float32) / max(T, 1e-6))
    pred = (p >= thresh).astype(np.uint8)
    inter = int((pred & gt_bin).sum()); uni = int((pred | gt_bin).sum())
    return 1.0 if uni == 0 else float(inter)/float(uni)

def ece_binary_from_logits(records_logits: List[np.ndarray], gt_bin: np.ndarray,
                           T: float, n_bins: int = 15) -> Tuple[float, np.ndarray, np.ndarray, np.ndarray]:
    """
    Expected Calibration Error (ECE) for binary segmentation.
    - records_logits: list of upsampled logits arrays (H,W) chosen per-file (CE-minimizing candidate).
    Returns: (ECE, bin_centers, bin_acc, bin_conf)
    """
    bin_tot = np.zeros(n_bins, dtype=np.float64)
    bin_correct = np.zeros(n_bins, dtype=np.float64)
    bin_conf_sum = np.zeros(n_bins, dtype=np.float64)

    for z in records_logits:
        p = sigmoid(z.astype(np.float32) / max(T, 1e-6))
        conf = np.maximum(p, 1.0 - p)          # confidence per pixel
        pred = (p >= 0.5).astype(np.uint8)
        corr = (pred == gt_bin).astype(np.float32)

        # bin by confidence
        # conf in [0,1]; map to bins [0, n_bins-1]
        idx = np.floor(conf * n_bins).astype(np.int32)
        idx = np.clip(idx, 0, n_bins-1)

        # accumulate
        for b in range(n_bins):
            mask = (idx == b)
            cnt = int(mask.sum())
            if cnt == 0: continue
            bin_tot[b] += cnt
            bin_correct[b] += float(corr[mask].sum())
            bin_conf_sum[b] += float(conf[mask].sum())

    N = float(bin_tot.sum()) + 1e-12
    bin_acc = np.divide(bin_correct, np.maximum(bin_tot, 1.0))
    bin_conf = np.divide(bin_conf_sum, np.maximum(bin_tot, 1.0))
    ece = float(np.sum((bin_tot / N) * np.abs(bin_acc - bin_conf)))
    centers = (np.arange(n_bins) + 0.5) / n_bins
    return ece, centers, bin_acc, bin_conf


# ---------------- SAM helpers ----------------

def build_predictor(checkpoint: str, model_type: str, device: Optional[str]=None) -> SamPredictor:
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    sam = sam_model_registry[model_type](checkpoint=checkpoint).to(device)
    return SamPredictor(sam)

def upsample_logits_to_image(lowres_logits: np.ndarray,
                             predictor: SamPredictor,
                             out_hw: Tuple[int,int]) -> np.ndarray:
    device = next(predictor.model.parameters()).device
    t = torch.from_numpy(lowres_logits).float().unsqueeze(0).unsqueeze(0).to(device)  # (1,1,h,w)
    H, W = out_hw
    if hasattr(predictor.model, "postprocess_masks"):
        up = predictor.model.postprocess_masks(t, predictor.input_size, predictor.original_size)[0,0]
    else:
        up = F.interpolate(t, size=(H,W), mode="bilinear", align_corners=False)[0,0]
    return up.detach().cpu().numpy()


# ---------------- caching logits ----------------

def collect_logits(prompts_dir: Path,
                   predictor: SamPredictor,
                   image_rgb: np.ndarray,
                   gt_bin: np.ndarray,
                   out_dir: Path,
                   multimask: bool=True) -> List[Dict]:
    """
    For each JSON in prompts_dir:
      - run SAM once, get candidate low-res logits
      - upsample to (H,W), save each candidate as float16 .npy
    Returns list of records: {"json": path, "cand_paths": [...]}.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    H, W = image_rgb.shape[:2]
    predictor.set_image(image_rgb)

    json_files = sorted([p for p in prompts_dir.glob("*.json") if p.is_file()])
    records = []
    for idx, jp in enumerate(json_files):
        coords, labels = load_prompts_json(str(jp))
        with torch.no_grad():
            _, _, lrs = predictor.predict(point_coords=coords,
                                          point_labels=labels,
                                          multimask_output=multimask,
                                          return_logits=True)
        cand_paths = []
        for ci in range(len(lrs)):
            up = upsample_logits_to_image(lrs[ci], predictor, (H, W)).astype(np.float16)
            fp = out_dir / f"{idx:04d}_cand{ci}.npy"
            np.save(fp, up)
            cand_paths.append(str(fp))
        records.append({"json": str(jp), "cand_paths": cand_paths})
        if (idx+1) % 20 == 0 or (idx+1) == len(json_files):
            print(f"[cache] {idx+1}/{len(json_files)} from {prompts_dir}")
    return records


# ---------------- robust aggregation helpers ----------------

def best_ce_per_file(records: List[Dict], gt_bin: np.ndarray, T: float) -> Tuple[List[float], List[int]]:
    """
    For each file, compute CE for each candidate at temperature T,
    return the best (min) CE and the index of the chosen candidate.
    """
    best_list = []
    best_idx = []
    for rec in records:
        best_ce = None; best_i = -1
        for i, cp in enumerate(rec["cand_paths"]):
            z = np.load(cp).astype(np.float32)
            ce = ce_from_logits_T(z, gt_bin, T)
            if (best_ce is None) or (ce < best_ce):
                best_ce = ce; best_i = i
        best_list.append(float(best_ce))
        best_idx.append(int(best_i))
    return best_list, best_idx

def trimmed_mean_and_se(values: List[float], trim: float = 0.10) -> Tuple[float, float, int]:
    """
    10% trimmed mean and standard error (on the trimmed sample).
    Returns (mean, se, n_eff).
    """
    arr = np.array(values, dtype=np.float64)
    n = len(arr)
    if n == 0:
        return float("nan"), float("nan"), 0
    k = int(math.floor(trim * n))
    if k*2 >= n:  # not enough to trim, fallback to plain mean
        m = float(arr.mean())
        se = float(arr.std(ddof=1) / math.sqrt(max(n,1))) if n > 1 else 0.0
        return m, se, n
    arr.sort()
    trimmed = arr[k:n-k]
    m = float(trimmed.mean())
    se = float(trimmed.std(ddof=1) / math.sqrt(len(trimmed))) if len(trimmed) > 1 else 0.0
    return m, se, len(trimmed)


# ---------------- plotting ----------------

def plot_ce_vs_T(Ts: List[float], means: List[float], best_T: float, out_path: Path, se_list: Optional[List[float]]=None):
    order = np.argsort(Ts)
    Tx = np.array(Ts)[order]; My = np.array(means)[order]
    plt.figure(figsize=(6.4,4.4))
    plt.plot(Tx, My, marker="o", label="Trimmed mean CE")
    if se_list is not None:
        Sy = np.array(se_list)[order]
        plt.fill_between(Tx, My-Sy, My+Sy, alpha=0.2, label="±1 SE")
    plt.axvline(best_T, color="k", linestyle="--", label=f"T*={best_T:.3f}")
    plt.xscale("log"); plt.xlabel("Temperature T (log scale)")
    plt.ylabel("CE (nats, trimmed mean)")
    plt.title("Calibration curve: CE vs T")
    plt.legend(); plt.tight_layout()
    plt.savefig(out_path, dpi=200); plt.close()

def plot_reliability(centers, acc, conf, out_path: Path, title: str):
    plt.figure(figsize=(5.0,5.0))
    plt.plot([0,1],[0,1], linestyle="--", color="gray", label="Perfect")
    plt.plot(conf, acc, marker="o", label="Observed")
    plt.xlabel("Confidence"); plt.ylabel("Accuracy")
    plt.title(title)
    plt.xlim(0,1); plt.ylim(0,1); plt.legend()
    plt.tight_layout(); plt.savefig(out_path, dpi=200); plt.close()

def plot_holdout_bars(ce1: float, ceT: float, ece1: float, eceT: float, out_path: Path):
    labels = ["CE", "ECE"]
    v1 = [ce1, ece1]; v2 = [ceT, eceT]
    x = np.arange(len(labels))
    w = 0.35
    plt.figure(figsize=(5.6,4.0))
    plt.bar(x - w/2, v1, width=w, label="T=1.0")
    plt.bar(x + w/2, v2, width=w, label="T*")
    plt.xticks(x, labels)
    plt.title("Hold-out summary (lower is better)")
    plt.legend(); plt.tight_layout()
    plt.savefig(out_path, dpi=200); plt.close()

def plot_hist_deltas(deltas: List[float], out_path: Path, title: str):
    plt.figure(figsize=(6.0,4.0))
    plt.hist(deltas, bins=20)
    plt.xlabel("CE(T=1) - CE(T*)  [nats]"); plt.ylabel("#files")
    plt.title(title)
    plt.tight_layout(); plt.savefig(out_path, dpi=200); plt.close()


# ---------------- main ----------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True)
    ap.add_argument("--gt", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--model-type", default="vit_h", choices=["vit_h","vit_l","vit_b"])
    ap.add_argument("--calib-dir", required=True)
    ap.add_argument("--eval-dir", required=True)
    ap.add_argument("--multimask", action="store_true")
    ap.add_argument("--T-lo", type=float, default=0.3)
    ap.add_argument("--T-hi", type=float, default=3.0)
    ap.add_argument("--grid", type=int, default=61)
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--trim", type=float, default=0.10)
    ap.add_argument("--bins", type=int, default=15)
    ap.add_argument("--out-dir", default="./runs/calib_one_robust")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    np.random.seed(args.seed)

    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir/"logits_calib").mkdir(exist_ok=True)
    (out_dir/"logits_eval").mkdir(exist_ok=True)

    # Load image/mask
    image = load_image_rgb(args.image)
    H, W = image.shape[:2]
    gt = load_mask_binary(args.gt, target_hw=(H, W))
    assert gt.shape == (H, W), "Image and GT must match size."
    if (gt>0).sum()==0 or (gt==0).sum()==0:
        raise ValueError("GT must contain both FG and BG.")

    # Build SAM predictor and cache logits
    predictor = build_predictor(args.checkpoint, args.model_type)
    print("[1/5] Caching logits for calibration files...")
    calib_records = collect_logits(Path(args.calib_dir), predictor, image, gt, out_dir/"logits_calib", multimask=args.multimask)
    if not calib_records:
        raise RuntimeError("No calibration JSONs found.")
    print("[2/5] Caching logits for hold-out (eval) files...")
    eval_records = collect_logits(Path(args.eval_dir), predictor, image, gt, out_dir/"logits_eval", multimask=args.multimask)
    if not eval_records:
        print("Warning: no eval JSONs found; verification will be limited.")

    # Line-search for T with trimmed mean + 1-SE rule
    print("[3/5] Searching T (trimmed mean CE + 1-SE rule)...")
    lo = math.log(args.T_lo); hi = math.log(args.T_hi)
    Ts_all, means_all, se_all = [], [], []
    best_mean, best_T = float("inf"), 1.0
    for r in range(max(1, args.rounds)):
        grid = np.linspace(lo, hi, max(3, args.grid))
        for t in grid:
            T = float(math.exp(t))
            ce_list, _ = best_ce_per_file(calib_records, gt, T)
            m, se, _ = trimmed_mean_and_se(ce_list, trim=args.trim)
            Ts_all.append(T); means_all.append(m); se_all.append(se)
            if m < best_mean:
                best_mean, best_T = m, T
        # zoom window
        span = (hi - lo) / (max(3, args.grid) - 1)
        lo = math.log(best_T) - 2.0*span
        hi = math.log(best_T) + 2.0*span
        print(f"  [round {r+1}] best so far: T≈{best_T:.4f}, trimmed CE≈{best_mean:.6f}")

    # 1-SE rule: choose largest T with mean <= min_mean + min_se
    # compute min_se at the T near the minimum
    idx_min = int(np.argmin(means_all))
    se_min = se_all[idx_min]
    thresh = best_mean + se_min
    candidates = [T for (T, m) in zip(Ts_all, means_all) if m <= thresh]
    T_star = max(candidates) if candidates else best_T  # safe fallback

    # Diagnostics on calibration at T=1 and T*
    ce1_list, idx1 = best_ce_per_file(calib_records, gt, 1.0)
    ceT_list, idxT = best_ce_per_file(calib_records, gt, T_star)
    mean_ce1, se_ce1, n1 = trimmed_mean_and_se(ce1_list, trim=args.trim)
    mean_ceT, se_ceT, nT = trimmed_mean_and_se(ceT_list, trim=args.trim)

    # ECE + reliability on calibration (use CE-best candidate per file at each T)
    cal_logits_T1 = []
    cal_logits_Ts = []
    for rec, i1, iT in zip(calib_records, idx1, idxT):
        cal_logits_T1.append(np.load(rec["cand_paths"][i1]).astype(np.float32))
        cal_logits_Ts.append(np.load(rec["cand_paths"][iT]).astype(np.float32))
    ece_cal_1, c1, acc1, conf1 = ece_binary_from_logits(cal_logits_T1, gt, T=1.0, n_bins=args.bins)
    ece_cal_s, cs, accs, confs = ece_binary_from_logits(cal_logits_Ts, gt, T=T_star, n_bins=args.bins)

    # Hold-out (eval) verification
    if eval_records:
        ev_ce1_list, e_idx1 = best_ce_per_file(eval_records, gt, 1.0)
        ev_ceT_list, e_idxT = best_ce_per_file(eval_records, gt, T_star)
        ev_mean_ce1 = float(np.mean(ev_ce1_list))
        ev_mean_ceT = float(np.mean(ev_ceT_list))
        ev_delta = float(ev_mean_ce1 - ev_mean_ceT)
        # ECE on hold-out
        ev_logits_T1 = [np.load(r["cand_paths"][i]).astype(np.float32) for r,i in zip(eval_records, e_idx1)]
        ev_logits_Ts = [np.load(r["cand_paths"][i]).astype(np.float32) for r,i in zip(eval_records, e_idxT)]
        ece_ev_1, _, ev_acc1, ev_conf1 = ece_binary_from_logits(ev_logits_T1, gt, T=1.0, n_bins=args.bins)
        ece_ev_s, _, ev_accs, ev_confs = ece_binary_from_logits(ev_logits_Ts, gt, T=T_star, n_bins=args.bins)
        # Hold-out mIoU
        ev_miou1 = np.mean([hard_miou_from_logits_T(z, gt, 1.0) for z in ev_logits_T1])
        ev_miouT = np.mean([hard_miou_from_logits_T(z, gt, T_star) for z in ev_logits_Ts])
    else:
        ev_mean_ce1 = ev_mean_ceT = ev_delta = float("nan")
        ece_ev_1 = ece_ev_s = ev_miou1 = ev_miouT = float("nan")

    # Save calibration summary
    calib_json = {
        "T_star": T_star,
        "grid_T": Ts_all,
        "grid_trimmed_mean_CE": means_all,
        "grid_SE": se_all,
        "best_trimmed_mean_CE": best_mean,
        "oneSE_threshold": thresh,
        "calib_trimmed_CE_T1": mean_ce1,
        "calib_trimmed_CE_Tstar": mean_ceT,
        "calib_ECE_T1": ece_cal_1,
        "calib_ECE_Tstar": ece_cal_s,
        "holdout_mean_CE_T1": ev_mean_ce1,
        "holdout_mean_CE_Tstar": ev_mean_ceT,
        "holdout_delta_CE": ev_delta,
        "holdout_ECE_T1": ece_ev_1,
        "holdout_ECE_Tstar": ece_ev_s,
        "holdout_mIoU_T1": ev_miou1,
        "holdout_mIoU_Tstar": ev_miouT,
        "notes": "Trim=10% trimmed mean. 1-SE rule picks largest T within one SE of the minimum."
    }
    with open(out_dir/"calibration.json", "w") as f:
        json.dump(calib_json, f, indent=2)

    # Per-file CSVs (calibration)
    with open(out_dir/"calibration_perfile.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["json","best_CE_T1","best_CE_Tstar","delta"])
        w.writeheader()
        for rec, c1, cs in zip(calib_records, ce1_list, ceT_list):
            w.writerow({"json": rec["json"], "best_CE_T1": c1, "best_CE_Tstar": cs, "delta": c1 - cs})

    # Per-file CSVs (holdout)
    if eval_records:
        with open(out_dir/"holdout_perfile.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["json","best_CE_T1","best_CE_Tstar","delta"])
            w.writeheader()
            for rec, c1, cs in zip(eval_records, ev_ce1_list, ev_ceT_list):
                w.writerow({"json": rec["json"], "best_CE_T1": c1, "best_CE_Tstar": cs, "delta": c1 - cs})

    # Plots
    print("[4/5] Writing plots...")
    plot_ce_vs_T(Ts_all, means_all, T_star, out_dir/"calib_curve_ce_vs_T.png", se_list=se_all)
    plot_reliability(c1, acc1, conf1, out_dir/"calib_reliability_T1.png", "Calibration reliability (T=1.0)")
    plot_reliability(cs, accs, confs, out_dir/"calib_reliability_Tstar.png", f"Calibration reliability (T*={T_star:.3f})")

    if eval_records:
        plot_holdout_bars(ev_mean_ce1, ev_mean_ceT, ece_ev_1, ece_ev_s, out_dir/"holdout_bars.png")
        deltas = [c1 - cs for c1, cs in zip(ev_ce1_list, ev_ceT_list)]
        plot_hist_deltas(deltas, out_dir/"holdout_delta_hist.png", "Hold-out CE(T=1) - CE(T*)")

    # Final console verdict
    print("[5/5] Verification summary")
    print(f"  T* ≈ {T_star:.6f}")
    print(f"  Calibration (trimmed mean CE):  T=1 → {mean_ce1:.6f}  |  T* → {mean_ceT:.6f}")
    print(f"  Calibration ECE:                T=1 → {ece_cal_1:.5f}  |  T* → {ece_cal_s:.5f}")
    if eval_records:
        print(f"  Hold-out mean CE:               T=1 → {ev_mean_ce1:.6f} |  T* → {ev_mean_ceT:.6f}  (Δ={ev_delta:+.6f})")
        print(f"  Hold-out ECE:                   T=1 → {ece_ev_1:.5f}  |  T* → {ece_ev_s:.5f}")
        print(f"  Hold-out mIoU:                  T=1 → {ev_miou1:.4f}  |  T* → {ev_miouT:.4f}")

    print("\nSaved files in:", out_dir)


if __name__ == "__main__":
    main()
