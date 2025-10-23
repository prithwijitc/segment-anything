#!/usr/bin/env python3
"""
Monte-Carlo prompt sufficiency, smallest sufficient set, and visualization (ONE image)
— calibration ONLY for ranking/refinement; FINAL overlays/metrics on BASE SAM (no T scaling).

Inputs
------
--image          path/to/image.png
--gt             path/to/mask.png             (binary; same HxW; nonzero=FG)
--checkpoint     /path/to/sam_vit_*.pth
--model-type     vit_h | vit_l | vit_b
--eval-dir       directory with hold-out JSONs (e.g., out_prompts_strict/eval)
--calib-json     calibration.json (contains "T_star" from robust calibration)
[--multimask]            evaluate SAM's multiple candidates
[--mc-samples 64]        Monte-Carlo contexts per prompt for sufficiency (rank/refine ONLY)
[--jitter-radius 2]      pixel jitter radius when sampling contexts (0 = off)
[--ce-eps 1e-4]          sufficiency |s_j| <= ce_eps → redundant; s_j < -ce_eps → harmful
[--ce-tol 1e-4]          refined CE must be ≤ full CE + ce_tol (under T*)
[--miou-tol 0.0]         refined mIoU must be ≥ full mIoU − miou_tol (under T*)
[--out-dir ./runs/mc_sufficiency_base_eval]
[--seed 0]

Outputs
-------
out-dir/
  perfile/
    <name>_sufficiency.csv         (s_j under T*; tags: useful/redundant/harmful)
    <name>_refined.json            (smallest sufficient set; order preserved; reindexed t)
    <name>_summary_metrics.csv     (both calibrated T* metrics and base metrics)
  overlays/
    <name>_full_vs_refined_BASE.png  (left full, right refined; BASE masks; green/red dots)
  summary.csv                      (aggregate counts + base and calibrated metrics)
  readme.txt

Notes
-----
• Ranking/refinement uses T*: s_j ≈ E_S[ CE_T*(S) − CE_T*(S∪{j}) ].
• Final overlays + numbers are on BASE SAM: sigmoid(z), threshold=0.5, candidate picked by CE at T=1.
• Prompt dots: green (label=1) and red (label=0), diameter=6 px.
"""

import argparse, json, csv, math
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
    arr = sorted(data.get("prompts", []), key=lambda d: d.get("t", 0))
    coords = np.array([[int(d["x"]), int(d["y"])] for d in arr], dtype=np.float32)
    labels = np.array([int(d["label"]) for d in arr], dtype=np.int32)
    # Require equal + and − in your generated JSONs
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

def run_sam_candidates(image_rgb: np.ndarray,
                       predictor: SamPredictor,
                       coords: np.ndarray,
                       labels: np.ndarray,
                       multimask: bool=True) -> List[np.ndarray]:
    """Return list of upsampled logits (H,W), one per SAM candidate."""
    H, W = image_rgb.shape[:2]
    with torch.no_grad():
        _, _, lrs = predictor.predict(point_coords=coords,
                                      point_labels=labels,
                                      multimask_output=multimask,
                                      return_logits=True)
    return [upsample_logits_to_image(lrs[i], predictor, (H, W)) for i in range(len(lrs))]

def ce_and_metrics_best(image_rgb: np.ndarray,
                        gt_bin: np.ndarray,
                        predictor: SamPredictor,
                        coords: np.ndarray,
                        labels: np.ndarray,
                        T: float,
                        multimask: bool=True) -> Tuple[float, float, float, np.ndarray]:
    """
    Run SAM, upsample logits for each candidate, pick the one with **lowest CE** at temperature T.
    Return (CE, mIoU, Dice, logits_best) under that SAME T.
    """
    ups = run_sam_candidates(image_rgb, predictor, coords, labels, multimask=multimask)
    best_ce = None; best_logits = None
    for z in ups:
        ce = ce_from_logits_T(z, gt_bin, T)
        if (best_ce is None) or (ce < best_ce):
            best_ce = ce; best_logits = z
    miou = miou_from_logits_T(best_logits, gt_bin, T)
    dice = dice_from_logits_T(best_logits, gt_bin, T)
    return float(best_ce), float(miou), float(dice), best_logits


# ---------------- Monte-Carlo sufficiency (uses T* ONLY) ----------------

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
    Monte-Carlo conditional marginal contributions under T*:
        s_j = E_S[ CE_T*(S) - CE_T*(S ∪ {j}) ]
    where S is a random non-empty subset of P\{j}. Uses jitter+snap for robustness.
    """
    H, W = image.shape[:2]
    n = len(coords_all)
    fg = (gt>0).astype(np.uint8); bg = (gt==0).astype(np.uint8)
    scores = np.zeros(n, dtype=np.float64)

    for j in range(n):
        deltas = []
        others_idx = [k for k in range(n) if k != j]
        for _ in range(mc_samples):
            m = rng.randint(1, len(others_idx)+1)
            S_idx = list(rng.choice(others_idx, size=m, replace=False))
            # jitter and snap each point in S and j
            C_S = []; L_S = []
            for k in S_idx:
                x, y = int(coords_all[k,0]), int(coords_all[k,1])
                x2, y2 = jitter_point(x, y, jitter_radius, H, W)
                x2, y2 = snap_to_class(x2, y2, int(labels_all[k]), fg, bg)
                C_S.append([x2, y2]); L_S.append(int(labels_all[k]))
            xj, yj = int(coords_all[j,0]), int(coords_all[j,1])
            xj2, yj2 = jitter_point(xj, yj, jitter_radius, H, W)
            xj2, yj2 = snap_to_class(xj2, yj2, int(labels_all[j]), fg, bg)

            C_S = np.array(C_S, dtype=np.float32); L_S = np.array(L_S, dtype=np.int32)
            C_Sj = np.vstack([C_S, [xj2, yj2]]).astype(np.float32)
            L_Sj = np.concatenate([L_S, [int(labels_all[j])]]).astype(np.int32)

            ce_S, _, _, _  = ce_and_metrics_best(image, gt, predictor, C_S,  L_S,  T_star, multimask)
            ce_Sj, _, _, _ = ce_and_metrics_best(image, gt, predictor, C_Sj, L_Sj, T_star, multimask)
            deltas.append(ce_S - ce_Sj)
        scores[j] = float(np.mean(deltas))
    return scores  # nats/pixel under T*


# ---------------- refinement (uses T* ONLY) ----------------

def build_smallest_sufficient(coords_all: np.ndarray,
                              labels_all: np.ndarray,
                              suff_scores: np.ndarray,
                              image: np.ndarray,
                              gt: np.ndarray,
                              predictor: SamPredictor,
                              T_star: float,
                              multimask: bool,
                              ce_full_Tstar: float,
                              miou_full_Tstar: float,
                              ce_tol: float,
                              miou_tol: float) -> List[int]:
    """
    Greedy under T*: sort by descending s_j (>0 only), add until CE_T* within ce_tol
    of full and mIoU_T* within miou_tol. Skip s<=0 (harmful/redundant under T*).
    """
    order = np.argsort(-suff_scores)
    selected = []
    ce_cur = float("inf"); miou_cur = 0.0

    for idx in order:
        if suff_scores[idx] <= 0:
            continue
        selected.append(idx)
        C = coords_all[selected]; L = labels_all[selected]
        ce_cur, miou_cur, _, _ = ce_and_metrics_best(image, gt, predictor, C, L, T_star, multimask)
        if (ce_cur <= ce_full_Tstar + ce_tol) and (miou_cur >= miou_full_Tstar - miou_tol):
            break

    # Fallback: ensure at least one prompt if none selected (under T*)
    if not selected:
        best_idx = None; best_ce = float("inf")
        for j in range(len(coords_all)):
            C = coords_all[[j]]; L = labels_all[[j]]
            ce, _, _, _ = ce_and_metrics_best(image, gt, predictor, C, L, T_star, multimask)
            if ce < best_ce:
                best_ce = ce; best_idx = j
        selected = [best_idx] if best_idx is not None else []

    return selected


# ---------------- JSON, drawing, overlays ----------------

def save_refined_json(json_in: Path,
                      json_out: Path,
                      selected_indices: List[int],
                      coords_all: np.ndarray,
                      labels_all: np.ndarray):
    """
    Save prompts with original order preserved, but only for selected indices.
    Reindex t = 1..k in that preserved order.
    """
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

def overlay_mask(im: np.ndarray, mask_bin: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    """
    Overlay a binary mask (1=FG) in a red tint on the image. Returns RGB uint8.
    """
    im = im.copy()
    color = np.array([255, 0, 0], dtype=np.uint8)  # red in RGB
    idx = mask_bin.astype(bool)
    im[idx] = (alpha*color + (1-alpha)*im[idx]).astype(np.uint8)
    return im

def draw_prompt_dots(im_rgb: np.ndarray,
                     coords: np.ndarray,
                     labels: np.ndarray,
                     radius: int = 3) -> np.ndarray:
    """
    Draw filled circles at prompt locations on a copy of the image.
    - label==1 (positive/inclusion) -> green
    - label==0 (negative/exclusion) -> red
    Radius=3 px => 6 px diameter.
    """
    out = im_rgb.copy()
    H, W = out.shape[:2]
    for (x, y), lab in zip(coords.astype(int), labels.astype(int)):
        x = int(np.clip(x, 0, W-1)); y = int(np.clip(y, 0, H-1))
        color = (0, 255, 0) if lab == 1 else (255, 0, 0)  # (B,G,R) if using cv2? We're drawing on RGB array; cv2 uses BGR internally but writes directly; choose as below:
        # cv2 assumes BGR; convert: green=(0,255,0), red=(0,0,255) in BGR.
        bgr = (0,255,0) if lab==1 else (0,0,255)
        cv2.circle(out, (x, y), radius, bgr, thickness=-1, lineType=cv2.LINE_AA)
        cv2.circle(out, (x, y), radius, (0,0,0), thickness=1, lineType=cv2.LINE_AA)
    return out

def draw_side_by_side_BASE(image: np.ndarray,
                           gt: np.ndarray,
                           logits_full_base: np.ndarray,
                           logits_ref_base: np.ndarray,
                           coords_full: np.ndarray,
                           labels_full: np.ndarray,
                           coords_ref: np.ndarray,
                           labels_ref: np.ndarray,
                           out_path: Path):
    """
    Save a side-by-side subplot using BASE SAM (no T scaling):
      Left: full prompts result; Right: refined prompts result.
      Titles show BASE mIoU & Dice (computed with sigmoid(z), thresh=0.5).
    """
    # BASE masks (T=1)
    pF = sigmoid(logits_full_base); mF = (pF >= 0.5).astype(np.uint8)
    pR = sigmoid(logits_ref_base);  mR = (pR >= 0.5).astype(np.uint8)

    # BASE metrics (T=1)
    miouF = miou_from_logits_T(logits_full_base, gt, T=1.0)
    diceF = dice_from_logits_T(logits_full_base,  gt, T=1.0)
    miouR = miou_from_logits_T(logits_ref_base,  gt, T=1.0)
    diceR = dice_from_logits_T(logits_ref_base,   gt, T=1.0)

    ovF = draw_prompt_dots(overlay_mask(image, mF, alpha=0.45), coords_full, labels_full, radius=3)
    ovR = draw_prompt_dots(overlay_mask(image, mR, alpha=0.45), coords_ref,  labels_ref,  radius=3)

    fig, axs = plt.subplots(1, 2, figsize=(10.6, 5.3))
    axs[0].imshow(ovF); axs[0].axis('off')
    axs[0].set_title(f"FULL (BASE)  mIoU={miouF:.3f}, Dice={diceF:.3f}")
    axs[1].imshow(ovR); axs[1].axis('off')
    axs[1].set_title(f"REFINED (BASE)  mIoU={miouR:.3f}, Dice={diceR:.3f}")
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
    ap.add_argument("--calib-json", required=True)    # used only for ranking/refinement
    ap.add_argument("--multimask", action="store_true")
    ap.add_argument("--mc-samples", type=int, default=64)
    ap.add_argument("--jitter-radius", type=int, default=2)
    ap.add_argument("--ce-eps", type=float, default=1e-4)
    ap.add_argument("--ce-tol", type=float, default=1e-4)
    ap.add_argument("--miou-tol", type=float, default=0.0)
    ap.add_argument("--out-dir", default="./runs/mc_sufficiency_base_eval")
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

    # Load T* (for ranking/refinement only)
    with open(args.calib_json, "r") as f:
        C = json.load(f)
    if "T_star" not in C:
        raise RuntimeError(f"{args.calib_json} must contain 'T_star'.")
    T_star = float(C["T_star"])
    print(f"[calibration] T* (used ONLY for ranking/refinement) = {T_star:.6f}")

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

        # ---------- Ranking/Refinement under T* ----------
        # Full metrics under T* (for the stopping criterion in greedy refinement)
        ce_full_Ts, miou_full_Ts, dice_full_Ts, _ = ce_and_metrics_best(
            image, gt, predictor, coords_all, labels_all, T_star, args.multimask
        )
        # MC sufficiency under T*
        suff_scores = mc_sufficiency_scores(
            image, gt, predictor, coords_all, labels_all, T_star,
            mc_samples=args.mc_samples, jitter_radius=args.jitter_radius,
            multimask=args.multimask, rng=rng
        )
        # Tags from s_j (under T*)
        tags = []
        for s in suff_scores:
            if s < -args.ce_eps: tags.append("harmful")
            elif abs(s) <= args.ce_eps: tags.append("redundant")
            else: tags.append("useful")

        # Greedy smallest sufficient set under T*
        selected_idx = build_smallest_sufficient(
            coords_all, labels_all, suff_scores, image, gt, predictor, T_star,
            args.multimask, ce_full_Ts, miou_full_Ts, args.ce_tol, args.miou_tol
        )

        # Save per-prompt sufficiency CSV
        with open(perfile_dir / f"{jp.stem}_sufficiency.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["index","x","y","label","sufficiency_nats_Tstar","tag"])
            w.writeheader()
            for i,(s,lab) in enumerate(zip(suff_scores, labels_all)):
                w.writerow({
                    "index": i+1,
                    "x": int(coords_all[i,0]),
                    "y": int(coords_all[i,1]),
                    "label": int(lab),
                    "sufficiency_nats_Tstar": float(s),
                    "tag": tags[i]
                })

        # Save refined JSON (keep original order)
        refined_json_path = perfile_dir / f"{jp.stem}_refined.json"
        save_refined_json(jp, refined_json_path, selected_idx, coords_all, labels_all)

        # ---------- FINAL EVALUATION on BASE SAM (T=1) ----------
        # FULL on BASE
        ce_full_base, miou_full_base, dice_full_base, logits_full_base = ce_and_metrics_best(
            image, gt, predictor, coords_all, labels_all, T=1.0, multimask=args.multimask
        )
        # REFINED on BASE
        if selected_idx:
            C_ref = coords_all[selected_idx]; L_ref = labels_all[selected_idx]
            ce_ref_base, miou_ref_base, dice_ref_base, logits_ref_base = ce_and_metrics_best(
                image, gt, predictor, C_ref, L_ref, T=1.0, multimask=args.multimask
            )
        else:
            # Degenerate fallback (shouldn't happen due to fallback in builder)
            C_ref = np.zeros((0,2), dtype=np.float32); L_ref = np.zeros((0,), dtype=np.int32)
            logits_ref_base = np.zeros_like(logits_full_base, dtype=np.float32)
            ce_ref_base = ce_from_logits_T(logits_ref_base, gt, T=1.0)
            miou_ref_base = miou_from_logits_T(logits_ref_base, gt, T=1.0)
            dice_ref_base = dice_from_logits_T(logits_ref_base, gt, T=1.0)

        # (Optional) also record calibrated metrics for reference
        ce_full_Ts2, miou_full_Ts2, dice_full_Ts2, _ = ce_and_metrics_best(
            image, gt, predictor, coords_all, labels_all, T_star, args.multimask
        )
        if selected_idx:
            ce_ref_Ts, miou_ref_Ts, dice_ref_Ts, _ = ce_and_metrics_best(
                image, gt, predictor, C_ref, L_ref, T_star, args.multimask
            )
        else:
            ce_ref_Ts = ce_full_Ts2; miou_ref_Ts = miou_full_Ts2; dice_ref_Ts = dice_full_Ts2

        # Save per-file metrics (both BASE and T*)
        with open(perfile_dir / f"{jp.stem}_summary_metrics.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=[
                "set","num_prompts",
                "CE_BASE","mIoU_BASE","Dice_BASE",
                "CE_Tstar","mIoU_Tstar","Dice_Tstar"
            ])
            w.writeheader()
            w.writerow({
                "set": "FULL", "num_prompts": len(coords_all),
                "CE_BASE": ce_full_base, "mIoU_BASE": miou_full_base, "Dice_BASE": dice_full_base,
                "CE_Tstar": ce_full_Ts2, "mIoU_Tstar": miou_full_Ts2, "Dice_Tstar": dice_full_Ts2
            })
            w.writerow({
                "set": "REFINED", "num_prompts": len(C_ref),
                "CE_BASE": ce_ref_base, "mIoU_BASE": miou_ref_base, "Dice_BASE": dice_ref_base,
                "CE_Tstar": ce_ref_Ts, "mIoU_Tstar": miou_ref_Ts, "Dice_Tstar": dice_ref_Ts
            })

        # Side-by-side overlay (BASE)
        draw_side_by_side_BASE(
            image, gt,
            logits_full_base=logits_full_base,
            logits_ref_base=logits_ref_base,
            coords_full=coords_all, labels_full=labels_all,
            coords_ref=C_ref,     labels_ref=L_ref,
            out_path=overlay_dir / f"{jp.stem}_full_vs_refined_BASE.png"
        )

        # Aggregate summary row
        n_h = tags.count("harmful"); n_r = tags.count("redundant"); n_u = tags.count("useful")
        summary_rows.append({
            "file": str(jp.name),
            "T_star": T_star,
            "N_full": len(coords_all),
            "N_refined": len(C_ref),
            "harmful": n_h, "redundant": n_r, "useful": n_u,
            # BASE metrics (final)
            "CE_BASE_full": ce_full_base, "mIoU_BASE_full": miou_full_base, "Dice_BASE_full": dice_full_base,
            "CE_BASE_ref":  ce_ref_base,  "mIoU_BASE_ref":  miou_ref_base,  "Dice_BASE_ref":  dice_ref_base,
            # Calibrated (for reference)
            "CE_Tstar_full": ce_full_Ts2, "mIoU_Tstar_full": miou_full_Ts2, "Dice_Tstar_full": dice_full_Ts2,
            "CE_Tstar_ref":  ce_ref_Ts,   "mIoU_Tstar_ref":  miou_ref_Ts,   "Dice_Tstar_ref":  dice_ref_Ts,
        })

        print(f"  BASE  FULL:    CE={ce_full_base:.6f}, mIoU={miou_full_base:.3f}, Dice={dice_full_base:.3f}")
        print(f"  BASE  REFINED: CE={ce_ref_base :.6f}, mIoU={miou_ref_base :.3f}, Dice={dice_ref_base :.3f}")
        print(f"  kept {len(C_ref)} / {len(coords_all)} prompts  (harmful={n_h}, redundant={n_r})")

    # Write global summary
    with open(out_dir/"summary.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=[
            "file","T_star","N_full","N_refined","harmful","redundant","useful",
            "CE_BASE_full","mIoU_BASE_full","Dice_BASE_full",
            "CE_BASE_ref","mIoU_BASE_ref","Dice_BASE_ref",
            "CE_Tstar_full","mIoU_Tstar_full","Dice_Tstar_full",
            "CE_Tstar_ref","mIoU_Tstar_ref","Dice_Tstar_ref"
        ])
        w.writeheader(); w.writerows(summary_rows)

    with open(out_dir/"readme.txt", "w") as f:
        f.write(
"""This run uses calibration (T*) **only** for ranking/refinement; final overlays/metrics are BASE SAM.

- Sufficiency scores s_j and the greedy 'smallest sufficient set' are computed under T*.
- The side-by-side figures and the reported mIoU/Dice in titles are computed on BASE:
    p = sigmoid(z), threshold = 0.5, candidate chosen by lowest CE at T=1.
- perfile/*_summary_metrics.csv records both BASE and T* metrics for transparency.

Prompt dots: green = positive (inclusion), red = negative (exclusion), 6 px diameter.

All CEs use natural logs (nats).
"""
        )

    print("\nDone. Results in:", out_dir)


if __name__ == "__main__":
    main()
