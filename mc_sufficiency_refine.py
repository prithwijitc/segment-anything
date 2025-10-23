#!/usr/bin/env python3
"""
Monte-Carlo prompt sufficiency, minimal sufficient set, and visualization (ONE image).

Inputs
------
--image          path/to/image.png
--gt             path/to/mask.png             (binary; same HxW; nonzero=FG)
--checkpoint     /path/to/sam_vit_*.pth
--model-type     vit_h | vit_l | vit_b
--eval-dir       directory with 20 hold-out JSONs (from your generator)
--calib-json     calibration.json (contains "T_star" from robust calibration)
[--multimask]    evaluate SAM's multiple candidates and pick CE-minimizer
[--mc-samples 64]     Monte-Carlo contexts per prompt for sufficiency
[--jitter-radius 2]   pixel jitter radius for coords in MC contexts (0 = off)
[--ce-eps 1e-4]       tolerance (nats/pixel) to treat sufficiency ~ 0 as redundant
[--ce-tol 1e-4]       how close refined CE must be to full CE (nats/pixel)
[--miou-tol 0.0]      how close refined mIoU must be to full mIoU (absolute)
[--out-dir ./runs/mc_sufficiency_one]
[--seed 0]

Outputs
-------
out-dir/
  perfile/
    <name>_sufficiency.csv      (per-prompt s_j and tag)
    <name>_refined.json         (smallest sufficient set, original order; reindexed t)
    <name>_sequence_full.csv    (full-policy CE/mIoU)
    <name>_sequence_refined.csv (refined-policy CE/mIoU)
  overlays/
    <name>_full_vs_refined.png  (left full, right refined; captions include mIoU & Dice)
  summary.csv                   (counts & sizes per file)
  readme.txt

Notes
-----
• CE uses natural logs (nats). Lower is better.
• Sufficiency score s_j > 0 helps; |s_j| <= ce_eps redundant; s_j < -ce_eps harmful.
• “Smallest sufficient set” is greedy on descending s_j until CE/refined within tolerances
  of the full policy (and skips harmful prompts).
"""

import argparse, json, csv, math, os
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
    raise RuntimeError("Install SAM from https://github.com/facebookresearch/segment-anything") from e


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
    arr = sorted(data.get("prompts", []), key=lambda d: d.get("t", 0))
    coords = np.array([[int(d["x"]), int(d["y"])] for d in arr], dtype=np.float32)
    labels = np.array([int(d["label"]) for d in arr], dtype=np.int32)
    # Require equal + and - in the input JSONs (your generator guarantees this)
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
    return float(-(y*np.log(p) + (1-y)*np.log(1-p)).mean())  # nats

def miou_from_logits_T(logits: np.ndarray, gt_bin: np.ndarray, T: float, thresh: float=0.5) -> float:
    p = sigmoid(logits.astype(np.float32) / max(T, 1e-6))
    pred = (p >= thresh).astype(np.uint8)
    inter = int((pred & gt_bin).sum()); uni = int((pred | gt_bin).sum())
    return 1.0 if uni == 0 else float(inter)/float(uni)

def dice_from_logits_T(logits: np.ndarray, gt_bin: np.ndarray, T: float, thresh: float=0.5) -> float:
    p = sigmoid(logits.astype(np.float32) / max(T, 1e-6))
    pred = (p >= thresh).astype(np.uint8)
    inter = int((pred & gt_bin).sum())
    a = int(pred.sum()); b = int(gt_bin.sum())
    return 1.0 if (a + b) == 0 else (2.0 * inter) / float(a + b)


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

def run_sam_best_logits(image_rgb: np.ndarray,
                        predictor: SamPredictor,
                        coords: np.ndarray,
                        labels: np.ndarray,
                        multimask: bool=True) -> np.ndarray:
    """
    Returns upsampled logits (H,W) of the CE-minimizing candidate vs GT is chosen later,
    so here we return ALL candidates? Simpler: return the best CE candidate once GT is known.
    To keep interface simple for CE eval, we return the best logits **after** CE eval.
    """
    # Note: we'll choose best by CE outside; so here we just run predict and return logits array list
    H, W = image_rgb.shape[:2]
    with torch.no_grad():
        masks, scores, lrs = predictor.predict(point_coords=coords,
                                               point_labels=labels,
                                               multimask_output=multimask,
                                               return_logits=True)
    ups = [upsample_logits_to_image(lrs[i], predictor, (H, W)) for i in range(len(lrs))]
    return ups  # List[np.ndarray (H,W)]

def ce_and_metrics_best(image_rgb: np.ndarray,
                        gt_bin: np.ndarray,
                        predictor: SamPredictor,
                        coords: np.ndarray,
                        labels: np.ndarray,
                        T: float,
                        multimask: bool=True) -> Tuple[float, float, float, np.ndarray]:
    """
    Run SAM, upsample logits for each candidate, pick the one with **lowest CE** vs GT.
    Return (CE, mIoU, Dice, logits_best).
    """
    ups = run_sam_best_logits(image_rgb, predictor, coords, labels, multimask=multimask)
    best_ce = None; best_logits = None
    for z in ups:
        ce = ce_from_logits_T(z, gt_bin, T)
        if (best_ce is None) or (ce < best_ce):
            best_ce = ce; best_logits = z
    miou = miou_from_logits_T(best_logits, gt_bin, T)
    dice = dice_from_logits_T(best_logits, gt_bin, T)
    return float(best_ce), float(miou), float(dice), best_logits


# ---------------- Monte-Carlo sufficiency ----------------

def jitter_point(x: int, y: int, r: int, H: int, W: int) -> Tuple[int,int]:
    if r <= 0: return x, y
    dx = int(np.round(np.random.uniform(-r, r)))
    dy = int(np.round(np.random.uniform(-r, r)))
    return int(np.clip(x+dx, 0, W-1)), int(np.clip(y+dy, 0, H-1))

def snap_to_class(x: int, y: int, label: int, fg: np.ndarray, bg: np.ndarray) -> Tuple[int,int]:
    want = (fg>0) if label==1 else (bg>0)
    if want[y, x]: return x, y
    ys, xs = np.where(want)
    if len(xs) == 0: return x, y
    d2 = (xs - x)**2 + (ys - y)**2
    j = int(np.argmin(d2))
    return int(xs[j]), int(ys[j])

def mc_sufficiency_scores(image: np.ndarray,
                          gt: np.ndarray,
                          predictor: SamPredictor,
                          coords_all: np.ndarray,
                          labels_all: np.ndarray,
                          T_star: float,
                          mc_samples: int,
                          jitter_radius: int,
                          multimask: bool,
                          rng: np.random.RandomState) -> np.ndarray:
    """
    Monte-Carlo conditional marginal contributions:
        s_j = E_S[ CE(S) - CE(S ∪ {j}) ]
    where S is a random non-empty subset of P\{j}. Uses jitter+snap for robustness.
    """
    H, W = image.shape[:2]
    n = len(coords_all)
    fg = (gt>0).astype(np.uint8); bg = (gt==0).astype(np.uint8)
    scores = np.zeros(n, dtype=np.float64)

    # Cache nothing; call SAM for each subset (keeps code simple & robust)
    for j in range(n):
        deltas = []
        others_idx = [k for k in range(n) if k != j]
        for _ in range(mc_samples):
            # choose random subset size in [1, len(others)]
            m = rng.randint(1, len(others_idx)+1)
            S_idx = list(rng.choice(others_idx, size=m, replace=False))
            # jitter everything lightly and snap to correct class
            C_S = []; L_S = []
            for k in S_idx:
                x, y = int(coords_all[k,0]), int(coords_all[k,1])
                x2, y2 = jitter_point(x, y, jitter_radius, H, W)
                x2, y2 = snap_to_class(x2, y2, int(labels_all[k]), fg, bg)
                C_S.append([x2, y2]); L_S.append(int(labels_all[k]))
            xj, yj = int(coords_all[j,0]), int(coords_all[j,1])
            xj2, yj2 = jitter_point(xj, yj, jitter_radius, H, W)
            xj2, yj2 = snap_to_class(xj2, yj2, int(labels_all[j]), fg, bg)
            C_Sj = C_S + [[xj2, yj2]]
            L_Sj = L_S + [int(labels_all[j])]

            C_S = np.array(C_S, dtype=np.float32); L_S = np.array(L_S, dtype=np.int32)
            C_Sj = np.array(C_Sj, dtype=np.float32); L_Sj = np.array(L_Sj, dtype=np.int32)

            ce_S, _, _, _  = ce_and_metrics_best(image, gt, predictor, C_S,  L_S,  T_star, multimask)
            ce_Sj, _, _, _ = ce_and_metrics_best(image, gt, predictor, C_Sj, L_Sj, T_star, multimask)
            deltas.append(ce_S - ce_Sj)
        scores[j] = float(np.mean(deltas))
    return scores  # nats/pixel (expected CE drop)


# ---------------- refinement & visualization ----------------

def build_smallest_sufficient(coords_all: np.ndarray,
                              labels_all: np.ndarray,
                              suff_scores: np.ndarray,
                              image: np.ndarray,
                              gt: np.ndarray,
                              predictor: SamPredictor,
                              T_star: float,
                              multimask: bool,
                              ce_full: float,
                              miou_full: float,
                              ce_tol: float,
                              miou_tol: float) -> List[int]:
    """
    Greedy: sort by descending sufficiency (>0 only), add until CE within ce_tol
    of full and mIoU within miou_tol (skip harmful). Return selected indices (original order retained later).
    """
    order = np.argsort(-suff_scores)  # descending
    selected = []
    ce_cur = float("inf"); miou_cur = 0.0

    for idx in order:
        if suff_scores[idx] <= 0:  # skip harmful & redundant (<=0)
            continue
        selected.append(idx)
        C = coords_all[selected]; L = labels_all[selected]
        ce_cur, miou_cur, _, _ = ce_and_metrics_best(image, gt, predictor, C, L, T_star, multimask)
        # stop when close enough to full-policy performance
        if (ce_cur <= ce_full + ce_tol) and (miou_cur >= miou_full - miou_tol):
            break

    # Fallback: if nothing positive, pick the single best prompt by actual CE drop from empty->add? (avoid empty set)
    if not selected:
        # Try each single prompt; pick best CE
        best_idx = None; best_ce = float("inf")
        for j in range(len(coords_all)):
            C = coords_all[[j]]; L = labels_all[[j]]
            ce, _, _, _ = ce_and_metrics_best(image, gt, predictor, C, L, T_star, multimask)
            if ce < best_ce:
                best_ce = ce; best_idx = j
        selected = [best_idx] if best_idx is not None else []

    return selected

def save_refined_json(json_in: Path,
                      json_out: Path,
                      selected_indices: List[int],
                      coords_all: np.ndarray,
                      labels_all: np.ndarray):
    """
    Save prompts with original order preserved, but only for selected indices.
    Reindex t = 1..k in that preserved order.
    """
    # preserve original order by filtering in increasing index order
    keep = sorted(selected_indices)
    prompts = []
    t = 1
    for i in keep:
        x, y = int(coords_all[i,0]), int(coords_all[i,1])
        lbl = int(labels_all[i])
        prompts.append({"t": t, "x": x, "y": y, "label": lbl})
        t += 1
    json_out.parent.mkdir(parents=True, exist_ok=True)
    with open(json_out, "w") as f:
        json.dump({"prompts": prompts}, f, indent=2)

def overlay_mask(im: np.ndarray, mask_bin: np.ndarray, alpha: float = 0.5) -> np.ndarray:
    """
    Overlay a binary mask (1=FG) in a colored tint on the image. Returns RGB uint8.
    """
    im = im.copy()
    color = np.array([255, 0, 0], dtype=np.uint8)  # red tint
    overlay = im.copy()
    overlay[mask_bin.astype(bool)] = (alpha*color + (1-alpha)*overlay[mask_bin.astype(bool)]).astype(np.uint8)
    return overlay

def draw_side_by_side(image: np.ndarray,
                      gt: np.ndarray,
                      logits_full: np.ndarray,
                      logits_ref: np.ndarray,
                      T_star: float,
                      title_left: str,
                      title_right: str,
                      out_path: Path):
    """
    Save a side-by-side subplot: left (full), right (refined) with captions that include mIoU and Dice.
    """
    pF = sigmoid(logits_full / max(T_star, 1e-6)); mF = (pF >= 0.5).astype(np.uint8)
    pR = sigmoid(logits_ref  / max(T_star, 1e-6)); mR = (pR >= 0.5).astype(np.uint8)

    miouF = miou_from_logits_T(logits_full, gt, T_star)
    diceF = dice_from_logits_T(logits_full, gt, T_star)
    miouR = miou_from_logits_T(logits_ref,  gt, T_star)
    diceR = dice_from_logits_T(logits_ref,  gt, T_star)

    ovF = overlay_mask(image, mF, alpha=0.45)
    ovR = overlay_mask(image, mR, alpha=0.45)

    fig, axs = plt.subplots(1, 2, figsize=(10.0, 5.2))
    axs[0].imshow(ovF); axs[0].axis('off')
    axs[0].set_title(f"{title_left}\n mIoU={miouF:.3f}, Dice={diceF:.3f}")
    axs[1].imshow(ovR); axs[1].axis('off')
    axs[1].set_title(f"{title_right}\n mIoU={miouR:.3f}, Dice={diceR:.3f}")
    plt.tight_layout(); out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=200); plt.close()


# ---------------- main ----------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True)
    ap.add_argument("--gt", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--model-type", default="vit_h", choices=["vit_h","vit_l","vit_b"])
    ap.add_argument("--eval-dir", required=True)
    ap.add_argument("--calib-json", required=True)
    ap.add_argument("--multimask", action="store_true")
    ap.add_argument("--mc-samples", type=int, default=64)
    ap.add_argument("--jitter-radius", type=int, default=2)
    ap.add_argument("--ce-eps", type=float, default=1e-4)
    ap.add_argument("--ce-tol", type=float, default=1e-4)
    ap.add_argument("--miou-tol", type=float, default=0.0)
    ap.add_argument("--out-dir", default="./runs/mc_sufficiency_one")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rng = np.random.RandomState(args.seed)

    out_dir = Path(args.out_dir)
    perfile_dir = out_dir / "perfile"; perfile_dir.mkdir(parents=True, exist_ok=True)
    overlay_dir = out_dir / "overlays"; overlay_dir.mkdir(parents=True, exist_ok=True)

    # Load image & GT
    image = load_image_rgb(args.image)
    H, W = image.shape[:2]
    gt = load_mask_binary(args.gt, target_hw=(H, W))
    assert gt.shape == (H, W), "Image & GT must match."
    if (gt>0).sum()==0 or (gt==0).sum()==0:
        raise ValueError("GT must contain both FG and BG.")

    # Load T*
    with open(args.calib_json, "r") as f:
        C = json.load(f)
    if "T_star" not in C:
        raise RuntimeError(f"{args.calib_json} must contain 'T_star'.")
    T_star = float(C["T_star"])
    print(f"[calibration] Using T* = {T_star:.6f}")

    # Build predictor
    predictor = build_predictor(args.checkpoint, args.model_type)
    predictor.set_image(image)

    # Collect eval JSONs
    files = sorted([p for p in Path(args.eval_dir).glob("*.json") if p.is_file()])
    if not files:
        raise RuntimeError("No JSONs found in eval-dir.")

    summary_rows = []

    for jp in files:
        print(f"\n[process] {jp.name}")
        coords_all, labels_all = load_prompts_json(str(jp))

        # Full policy metrics & logits
        ce_full, miou_full, dice_full, logits_full = ce_and_metrics_best(
            image, gt, predictor, coords_all, labels_all, T_star, args.multimask
        )

        # Monte-Carlo sufficiency scores
        suff_scores = mc_sufficiency_scores(
            image, gt, predictor, coords_all, labels_all, T_star,
            mc_samples=args.mc_samples, jitter_radius=args.jitter_radius,
            multimask=args.multimask, rng=rng
        )

        # Tag each prompt
        tags = []
        for s in suff_scores:
            if s < -args.ce_eps: tags.append("harmful")
            elif abs(s) <= args.ce_eps: tags.append("redundant")
            else: tags.append("useful")

        # Save per-prompt scores CSV
        with open(perfile_dir / f"{jp.stem}_sufficiency.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["index","x","y","label","sufficiency_nats","tag"])
            w.writeheader()
            for i,(s,lab) in enumerate(zip(suff_scores, labels_all)):
                w.writerow({
                    "index": i+1,
                    "x": int(coords_all[i,0]),
                    "y": int(coords_all[i,1]),
                    "label": int(lab),
                    "sufficiency_nats": float(s),
                    "tag": tags[i]
                })

        # Build smallest sufficient set (skip harmful, prefer largest s_j)
        selected_idx = build_smallest_sufficient(
            coords_all, labels_all, suff_scores, image, gt, predictor, T_star,
            args.multimask, ce_full, miou_full, args.ce_tol, args.miou_tol
        )

        # Save refined JSON (keep original order; reindex t)
        refined_json_path = perfile_dir / f"{jp.stem}_refined.json"
        save_refined_json(jp, refined_json_path, selected_idx, coords_all, labels_all)

        # Evaluate refined set & get logits for visualization
        if selected_idx:
            C_ref = coords_all[selected_idx]; L_ref = labels_all[selected_idx]
            ce_ref, miou_ref, dice_ref, logits_ref = ce_and_metrics_best(
                image, gt, predictor, C_ref, L_ref, T_star, args.multimask
            )
        else:
            # No selection: fall back to zeros (degenerate)
            logits_ref = np.zeros_like(logits_full, dtype=np.float32)
            ce_ref = ce_from_logits_T(logits_ref, gt, T_star)
            miou_ref = miou_from_logits_T(logits_ref, gt, T_star)
            dice_ref = dice_from_logits_T(logits_ref, gt, T_star)

        # Save “sequence” CSVs (just 1 row for full/refined summary here)
        with open(perfile_dir / f"{jp.stem}_sequence_full.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["num_prompts","ce","miou","dice"])
            w.writeheader(); w.writerow({"num_prompts": len(coords_all), "ce": ce_full, "miou": miou_full, "dice": dice_full})
        with open(perfile_dir / f"{jp.stem}_sequence_refined.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["num_prompts","ce","miou","dice"])
            w.writeheader(); w.writerow({"num_prompts": len(selected_idx), "ce": ce_ref, "miou": miou_ref, "dice": dice_ref})

        # Side-by-side overlay figure
        draw_side_by_side(
            image, gt, logits_full, logits_ref, T_star,
            title_left = f"Full prompts (N={len(coords_all)})",
            title_right= f"Refined (N={len(selected_idx)})",
            out_path = overlay_dir / f"{jp.stem}_full_vs_refined.png"
        )

        # Update summary
        n_h = tags.count("harmful"); n_r = tags.count("redundant"); n_u = tags.count("useful")
        summary_rows.append({
            "file": str(jp.name),
            "T_star": T_star,
            "N_full": len(coords_all),
            "N_refined": len(selected_idx),
            "harmful": n_h, "redundant": n_r, "useful": n_u,
            "CE_full": ce_full, "mIoU_full": miou_full, "Dice_full": dice_full,
            "CE_ref": ce_ref, "mIoU_ref": miou_ref, "Dice_ref": dice_ref
        })

        print(f"  CE_full={ce_full:.6f}, mIoU_full={miou_full:.3f}, Dice_full={dice_full:.3f}")
        print(f"  CE_ref ={ce_ref :.6f}, mIoU_ref ={miou_ref :.3f}, Dice_ref ={dice_ref :.3f}")
        print(f"  kept {len(selected_idx)} / {len(coords_all)} prompts (harmful={n_h}, redundant={n_r})")

    # Write summary & readme
    with open(out_dir/"summary.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=[
            "file","T_star","N_full","N_refined","harmful","redundant","useful",
            "CE_full","mIoU_full","Dice_full","CE_ref","mIoU_ref","Dice_ref"
        ])
        w.writeheader(); w.writerows(summary_rows)

    with open(out_dir/"readme.txt", "w") as f:
        f.write(
"""Sufficiency score (per prompt): s_j ≈ E_S[ CE(S) - CE(S ∪ {j}) ]
- Units: nats/pixel (expected cross-entropy drop).
- Tagging:
    s_j < -ce_eps     → harmful
    |s_j| ≤  ce_eps   → redundant
    s_j >   ce_eps    → useful

Smallest sufficient set:
- Sort prompts by descending s_j (skip ≤0).
- Greedily add until CE ≤ CE_full + ce_tol and mIoU ≥ mIoU_full − miou_tol.
- Save refined JSON (original order; reindexed t).

Overlays:
- Left: SAM result with ALL prompts; Right: result with refined set.
- Titles list mIoU and Dice vs GT.

All CEs use natural logs (nats); probabilities computed with calibrated T*.
"""
        )

    print("\nDone. Results in:", out_dir)


if __name__ == "__main__":
    main()
