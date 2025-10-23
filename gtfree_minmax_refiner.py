# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-
# """
# GT-FREE, THEORY-FIRST PROMPT REFINER (HARD CLICK CONSTRAINT + MIN–MAX ENERGY)
# =============================================================================

# Energy uses dynamic, per-image "weights" via Chebyshev (min–max) on normalized terms.
# Click consistency is enforced as a HARD constraint (no tradeoff).

# F(P) := max{  Ũ(P), Ř(P), Ė(P) }   subject to  C(P) ≤ ε
#  where:
#    C  = click consistency (BCE at click locations; lower is better)
#    U  = stability/uncertainty (predictive entropy + 1 - mean pairwise IoU across TTA/jitter)
#    R  = geometric regularity (TV(prob) + 0.2 * L^2 / A + 0.05 * topology)
#    E  = info separation proxy (H_in + H_out of intensity histograms)

# Normalization (per image):
#    T̃(P) = (T(P) - T_best) / (T_rand - T_best), clamped to [0,1],
#    where T_best = empirical lower envelope over feasible random subsets,
#          T_rand = median over feasible random subsets.

# Dynamic "weights":
# - No α,β,γ,δ are tuned.
# - The min–max solution induces KKT multipliers (implicit weights) per image.
# - We don't need to expose them; they explain *why* the chosen subset balances the bottleneck term(s).

# Outputs
# -------
# out_dir/
#   perfile/
#     sufficiency.csv               # per-prompt s_j, stderr, CI, tag
#     refined.json                  # only prompts with CI95_low > 0 (useful)
#     balanced_minmax.json          # (optional) greedily minimized Chebyshev subset
#     energy_breakdown.csv          # full vs refined (C,U,R,E, F=minmax) and envelopes
#     envelopes.json                # T_best / T_rand for terms, ε used, feasibility stats
#   overlays/
#     full_vs_refined_BASE.png      # left: FULL, right: REFINED (BASE SAM, T=1), prompts drawn
#     full_vs_minmax_BASE.png       # (optional) FULL vs balanced_minmax
#   summary.txt                     # human-readable summary

# Run
# ---
# python gtfree_minmax_refiner.py \
#   --image path/to/image.jpg \
#   --prompts path/to/full_prompts.json \
#   --checkpoint /path/to/sam_vit_h_4b8939.pth \
#   --model-type vit_h \
#   --out-dir ./runs/minmax \
#   --multimask \
#   --tta \
#   --mc-samples 48 \
#   --env-samples 96 \
#   --jitter-radius 2 \
#   --trim-frac 0.1 \
#   --eps-mode full \
#   --eps-scale 1.00 \
#   --plateau 1e-4 \
#   --seed 0
# """

# import argparse, json, csv, math, os
# from pathlib import Path
# from typing import List, Tuple, Optional, Dict

# import numpy as np
# import cv2
# from PIL import Image
# import torch
# import torch.nn.functional as F
# import matplotlib.pyplot as plt

# # ---- SAM ----
# try:
#     from segment_anything import sam_model_registry, SamPredictor
# except Exception as e:
#     raise RuntimeError("Install SAM: https://github.com/facebookresearch/segment-anything") from e


# # ===================== I/O =====================

# def load_image_rgb(path: str) -> np.ndarray:
#     return np.array(Image.open(path).convert("RGB"))

# def load_prompts_json(path: str) -> Tuple[np.ndarray, np.ndarray]:
#     with open(path, "r") as f:
#         data = json.load(f)
#     arr = sorted(data.get("prompts", []), key=lambda d: d.get("t", 0))
#     coords = np.array([[int(d["x"]), int(d["y"])] for d in arr], dtype=np.float32)
#     labels = np.array([int(d["label"]) for d in arr], dtype=np.int32)
#     return coords, labels


# # ===================== Utils =====================

# def sigmoid(x: np.ndarray) -> np.ndarray:
#     return 1.0 / (1.0 + np.exp(-x))

# def draw_prompt_dots(im_rgb: np.ndarray,
#                      coords: np.ndarray, labels: np.ndarray,
#                      radius: int = 3) -> np.ndarray:
#     out = im_rgb.copy()
#     H, W = out.shape[:2]
#     for (x, y), lab in zip(coords.astype(int), labels.astype(int)):
#         x = int(np.clip(x, 0, W-1)); y = int(np.clip(y, 0, H-1))
#         bgr = (0,255,0) if lab==1 else (0,0,255)
#         cv2.circle(out, (x, y), radius, bgr, thickness=-1, lineType=cv2.LINE_AA)
#         cv2.circle(out, (x, y), radius, (0,0,0), thickness=1, lineType=cv2.LINE_AA)
#     return out

# def overlay_mask(im: np.ndarray, mask_bin: np.ndarray, alpha: float=0.45) -> np.ndarray:
#     im = im.copy()
#     color = np.array([255, 0, 0], dtype=np.uint8)
#     idx = mask_bin.astype(bool)
#     im[idx] = (alpha*color + (1-alpha)*im[idx]).astype(np.uint8)
#     return im

# def jitter_point(x: int, y: int, r: int, H: int, W: int) -> Tuple[int,int]:
#     if r <= 0: return x, y
#     dx = int(np.round(np.random.uniform(-r, r)))
#     dy = int(np.round(np.random.uniform(-r, r)))
#     return int(np.clip(x+dx, 0, W-1)), int(np.clip(y+dy, 0, H-1))

# def trimmed_mean(vals: List[float], trim_frac: float=0.1) -> float:
#     if len(vals) == 0: return 0.0
#     arr = np.sort(np.asarray(vals, dtype=np.float64))
#     k = int(math.floor(trim_frac*len(arr)))
#     if 2*k >= len(arr): return float(arr.mean())
#     return float(arr[k:-k].mean())

# def stderr(vals: List[float]) -> float:
#     if len(vals) <= 1: return 0.0
#     a = np.asarray(vals, dtype=np.float64)
#     return float(a.std(ddof=1) / math.sqrt(len(a)))


# # ===================== SAM helpers =====================

# def build_predictor(checkpoint: str, model_type: str, device: Optional[str]=None) -> SamPredictor:
#     device = device or ("cuda" if torch.cuda.is_available() else "cpu")
#     sam = sam_model_registry[model_type](checkpoint=checkpoint).to(device)
#     return SamPredictor(sam)

# def upsample_logits_to_image(lowres_logits: np.ndarray,
#                              predictor: SamPredictor,
#                              out_hw: Tuple[int,int]) -> np.ndarray:
#     device = next(predictor.model.parameters()).device
#     t = torch.from_numpy(lowres_logits).float().unsqueeze(0).unsqueeze(0).to(device)
#     H, W = out_hw
#     if hasattr(predictor.model, "postprocess_masks"):
#         up = predictor.model.postprocess_masks(t, predictor.input_size, predictor.original_size)[0,0]
#     else:
#         up = F.interpolate(t, size=(H,W), mode="bilinear", align_corners=False)[0,0]
#     return up.detach().cpu().numpy()

# def run_sam_logits(image_rgb: np.ndarray,
#                    predictor: SamPredictor,
#                    coords: np.ndarray,
#                    labels: np.ndarray,
#                    multimask: bool=True) -> List[np.ndarray]:
#     """Return list of upsampled logits (H,W) for SAM candidates."""
#     H, W = image_rgb.shape[:2]
#     with torch.no_grad():
#         _, _, lrs = predictor.predict(point_coords=coords,
#                                       point_labels=labels,
#                                       multimask_output=multimask,
#                                       return_logits=True)
#     return [upsample_logits_to_image(lrs[i], predictor, (H,W)) for i in range(len(lrs))]


# # ===================== Energy terms =====================

# def click_consistency_BCE_at_points(logits: np.ndarray,
#                                     coords: np.ndarray,
#                                     labels: np.ndarray) -> float:
#     z = logits.astype(np.float32)
#     p = sigmoid(z)
#     H, W = p.shape
#     # bilinear sample for subpixel robustness
#     def bilinear(pmap, x, y):
#         x = np.clip(x, 0, W-1); y = np.clip(y, 0, H-1)
#         x0, y0 = int(np.floor(x)), int(np.floor(y))
#         x1, y1 = min(x0+1,W-1), min(y0+1,H-1)
#         dx, dy = x-x0, y-y0
#         v = (1-dx)*(1-dy)*pmap[y0,x0] + dx*(1-dy)*pmap[y0,x1] + (1-dx)*dy*pmap[y1,x0] + dx*dy*pmap[y1,x1]
#         return v
#     ce = 0.0
#     eps = 1e-7
#     for (x,y), lab in zip(coords, labels):
#         pv = float(bilinear(p, float(x), float(y)))
#         pv = min(max(pv, eps), 1-eps)
#         if lab == 1:
#             ce += -math.log(pv)
#         else:
#             ce += -math.log(1.0 - pv)
#     return ce / max(1, len(labels))

# def tv_on_prob(p: np.ndarray) -> float:
#     gx = np.diff(p, axis=1, append=p[:,-1:])
#     gy = np.diff(p, axis=0, append=p[-1:,:])
#     return float(np.mean(np.sqrt(gx*gx + gy*gy) + 1e-8))

# def mask_perimeter_and_area(mask: np.ndarray) -> Tuple[float,float]:
#     mask_u8 = (mask.astype(np.uint8))*255
#     cnts, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
#     L = 0.0
#     for c in cnts:
#         L += float(cv2.arcLength(c, True))
#     A = float(mask.sum())
#     return L, A

# def topo_penalty(mask: np.ndarray) -> float:
#     # penalize extra components and holes (lightly)
#     num_labels, _ = cv2.connectedComponents(mask.astype(np.uint8), connectivity=8)
#     K = int(num_labels) - 1  # exclude background
#     # quick hole check via closing
#     holes = int(cv2.countNonZero(cv2.morphologyEx(mask.astype(np.uint8)*255, cv2.MORPH_CLOSE, np.ones((3,3),np.uint8)) - mask.astype(np.uint8)*255) > 0)
#     return 0.1*max(0, K-1) + 0.2*holes

# def entropy_inside_outside(image_rgb: np.ndarray,
#                            mask: np.ndarray,
#                            nbins: int = 64) -> float:
#     # Use LAB-L channel for robustness
#     gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2LAB)[...,0]  # 0..255
#     m = mask.astype(bool)
#     if m.sum()==0 or (~m).sum()==0:
#         return 0.0
#     def H(vals):
#         hist, _ = np.histogram(vals, bins=nbins, range=(0,255), density=True)
#         hist = np.clip(hist, 1e-9, None)
#         hist = hist / hist.sum()
#         return float(-(hist*np.log(hist)).sum())
#     Hin = H(gray[m]); Hout = H(gray[~m])
#     return Hin + Hout

# def pairwise_iou(masks: List[np.ndarray]) -> float:
#     if len(masks) < 2: return 1.0
#     n = len(masks)
#     tot = 0.0; cnt = 0
#     for i in range(n):
#         for j in range(i+1, n):
#             a = masks[i].astype(bool); b = masks[j].astype(bool)
#             inter = float(np.logical_and(a,b).sum())
#             uni = float(np.logical_or(a,b).sum())
#             tot += (1.0 if uni==0 else inter/uni); cnt += 1
#     return tot / max(1, cnt)


# # ===================== TTA =====================

# def tta_configs(enable: bool):
#     if not enable:
#         return [("id", None)]
#     return [("id", None), ("hflip", None), ("rot", 3), ("rot", -3), ("scale", 0.97), ("scale", 1.03)]

# def apply_tta(image: np.ndarray, coords: np.ndarray, kind: str, param) -> Tuple[np.ndarray, np.ndarray, callable]:
#     H, W = image.shape[:2]
#     if kind == "id":
#         def inv_coords(xy): return xy
#         return image, coords, inv_coords

#     if kind == "hflip":
#         img2 = np.ascontiguousarray(image[:, ::-1, :])
#         coords2 = coords.copy()
#         coords2[:,0] = (W-1) - coords2[:,0]
#         def inv_coords(xy):
#             xy2 = xy.copy()
#             xy2[:,0] = (W-1) - xy2[:,0]
#             return xy2
#         return img2, coords2, inv_coords

#     if kind == "rot":
#         angle = float(param)
#         M = cv2.getRotationMatrix2D((W/2, H/2), angle, 1.0)
#         img2 = cv2.warpAffine(image, M, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
#         ones = np.ones((coords.shape[0],1), dtype=np.float32)
#         xy1 = np.hstack([coords, ones])
#         xy2 = (M @ xy1.T).T
#         Minv = cv2.invertAffineTransform(M)
#         def inv_coords(xy):
#             ones2 = np.ones((xy.shape[0],1), dtype=np.float32)
#             return (Minv @ np.hstack([xy, ones2]).T).T
#         return img2, xy2.astype(np.float32), inv_coords

#     if kind == "scale":
#         s = float(param)
#         M = np.array([[s,0,(1-s)*W/2],[0,s,(1-s)*H/2]], dtype=np.float32)
#         img2 = cv2.warpAffine(image, M, (W,H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
#         ones = np.ones((coords.shape[0],1), dtype=np.float32)
#         xy1 = np.hstack([coords, ones])
#         xy2 = (M @ xy1.T).T
#         Minv = cv2.invertAffineTransform(M)
#         def inv_coords(xy):
#             ones2 = np.ones((xy.shape[0],1), dtype=np.float32)
#             return (Minv @ np.hstack([xy, ones2]).T).T
#         return img2, xy2.astype(np.float32), inv_coords

#     raise ValueError("unknown TTA kind")


# # ===================== Core eval =====================

# def evaluate_terms(image: np.ndarray,
#                    predictor: SamPredictor,
#                    coords: np.ndarray,
#                    labels: np.ndarray,
#                    multimask: bool,
#                    tta_on: bool,
#                    jitter_radius: int,
#                    rng: np.random.RandomState) -> Tuple[float,float,float,float]:
#     """
#     Compute (C, U, R, E) for a subset, aggregating over TTA + jitter.
#     Candidate selection per TTA is by *lowest click BCE*.
#     """
#     H, W = image.shape[:2]
#     confs = tta_configs(tta_on)

#     prob_maps = []
#     masks = []
#     click_ce_list = []

#     for kind, param in confs:
#         # jitter prompts for this run
#         coords_j = coords.copy()
#         for i in range(coords_j.shape[0]):
#             coords_j[i,0], coords_j[i,1] = jitter_point(int(coords_j[i,0]), int(coords_j[i,1]),
#                                                          jitter_radius, H, W)
#         img_t, coords_t, inv = apply_tta(image, coords_j, kind, param)
#         predictor.set_image(img_t)
#         ups = run_sam_logits(img_t, predictor, coords_t, labels, multimask)

#         # choose candidate minimizing click BCE
#         best = None; best_click_ce = None
#         for z in ups:
#             ce = click_consistency_BCE_at_points(z, coords_t, labels)
#             if best is None or ce < best_click_ce:
#                 best = z; best_click_ce = ce

#         # invert prob to original frame
#         p_t = sigmoid(best.astype(np.float32))
#         if kind == "id":
#             p_back = p_t
#         elif kind == "hflip":
#             p_back = p_t[:, ::-1]
#         elif kind == "rot":
#             angle = float(param)
#             M = cv2.getRotationMatrix2D((W/2, H/2), -angle, 1.0)
#             p_back = cv2.warpAffine(p_t, M, (W,H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
#         elif kind == "scale":
#             s = float(param)
#             Minv = cv2.invertAffineTransform(np.array([[s,0,(1-s)*W/2],[0,s,(1-s)*H/2]], dtype=np.float32))
#             p_back = cv2.warpAffine(p_t, Minv, (W,H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
#         else:
#             p_back = p_t

#         prob_maps.append(p_back)
#         masks.append((p_back>=0.5).astype(np.uint8))
#         click_ce_list.append(float(best_click_ce))

#     # Aggregate
#     p_mean = np.mean(np.stack(prob_maps,0), axis=0)
#     m_mean = (p_mean>=0.5).astype(np.uint8)

#     # C: click BCE across TTAs
#     C_term = float(np.mean(click_ce_list))

#     # U: entropy + (1 - mean IoU across TTAs)
#     eps = 1e-7
#     H_pred = float(np.mean(-(np.clip(p_mean,eps,1-eps)*np.log(np.clip(p_mean,eps,1-eps)) +
#                             np.clip(1-p_mean,eps,1-eps)*np.log(np.clip(1-p_mean,eps,1-eps)))))
#     mean_iou = pairwise_iou(masks)
#     U_term = H_pred + (1.0 - mean_iou)

#     # R: TV(prob) + compactness + tiny topology penalty
#     tv = tv_on_prob(p_mean)
#     L, A = mask_perimeter_and_area(m_mean)
#     comp = (L*L) / max(A, 1.0)
#     topo = topo_penalty(m_mean)
#     R_term = tv + 0.2*comp + 0.05*topo

#     # E: entropy inside+outside
#     E_info = entropy_inside_outside(image, m_mean, nbins=64)

#     return C_term, U_term, R_term, E_info


# def estimate_envelopes(image: np.ndarray,
#                        predictor: SamPredictor,
#                        coords_all: np.ndarray,
#                        labels_all: np.ndarray,
#                        multimask: bool,
#                        tta_on: bool,
#                        jitter_radius: int,
#                        rng: np.random.RandomState,
#                        env_samples: int,
#                        eps: float) -> Dict[str, Dict[str,float]]:
#     """
#     Build per-image normalization envelopes for U,R,E using feasible random subsets (C ≤ ε).
#     Returns:
#       { 'U': {'best': ..., 'rand': ...}, 'R': {...}, 'E': {...}, 'feasible_frac': ... }
#     """
#     n = len(coords_all)
#     Uvals, Rvals, Evals = [], [], []
#     feas = 0; tried = 0

#     for _ in range(env_samples):
#         m = rng.randint(1, n+1)
#         idx = np.sort(rng.choice(np.arange(n), size=m, replace=False))
#         C, U, R, E = evaluate_terms(image, predictor, coords_all[idx], labels_all[idx],
#                                     multimask, tta_on, jitter_radius, rng)
#         tried += 1
#         if C <= eps:
#             feas += 1
#             Uvals.append(U); Rvals.append(R); Evals.append(E)

#     # Always include FULL set stats
#     C_full, U_full, R_full, E_full = evaluate_terms(image, predictor, coords_all, labels_all,
#                                                     multimask, tta_on, jitter_radius, rng)
#     if C_full <= eps:
#         Uvals.append(U_full); Rvals.append(R_full); Evals.append(E_full)

#     def stats(arr):
#         if len(arr)==0:
#             return {"best": 0.0, "rand": 1.0}  # fallback (will clamp)
#         arr = np.asarray(arr, dtype=np.float64)
#         return {"best": float(np.min(arr)), "rand": float(np.median(arr))}
#     env = {
#         "U": stats(Uvals),
#         "R": stats(Rvals),
#         "E": stats(Evals),
#         "feasible_frac": float(feas / max(1,tried))
#     }
#     return env


# def normalize_term(val: float, best: float, rand: float) -> float:
#     den = max(1e-9, rand - best)
#     return float(np.clip((val - best) / den, 0.0, 1.0))


# def chebyshev_energy(image: np.ndarray,
#                      predictor: SamPredictor,
#                      coords: np.ndarray,
#                      labels: np.ndarray,
#                      multimask: bool,
#                      tta_on: bool,
#                      jitter_radius: int,
#                      rng: np.random.RandomState,
#                      eps: float,
#                      env: Dict[str,Dict[str,float]]) -> Tuple[float, Dict[str,float]]:
#     """
#     Hard constraint: if C > eps, return (inf, ...).
#     Otherwise return Chebyshev (max of normalized U,R,E) and the breakdown dict.
#     """
#     C, U, R, E = evaluate_terms(image, predictor, coords, labels, multimask, tta_on, jitter_radius, rng)
#     if C > eps:
#         return float("inf"), {"C": C, "U": U, "R": R, "E": E, "U_t": 1.0, "R_t": 1.0, "E_t": 1.0, "F": float("inf")}
#     U_t = normalize_term(U, env["U"]["best"], env["U"]["rand"])
#     R_t = normalize_term(R, env["R"]["best"], env["R"]["rand"])
#     E_t = normalize_term(E, env["E"]["best"], env["E"]["rand"])
#     F_val = float(max(U_t, R_t, E_t))
#     return F_val, {"C": C, "U": U, "R": R, "E": E, "U_t": U_t, "R_t": R_t, "E_t": E_t, "F": F_val}


# # ===================== MC sufficiency & selection =====================

# def mc_sufficiency(image: np.ndarray,
#                    predictor: SamPredictor,
#                    coords_all: np.ndarray,
#                    labels_all: np.ndarray,
#                    multimask: bool,
#                    tta_on: bool,
#                    jitter_radius: int,
#                    rng: np.random.RandomState,
#                    mc_samples: int,
#                    trim_frac: float,
#                    eps: float,
#                    env: Dict[str,Dict[str,float]]) -> Tuple[np.ndarray, np.ndarray]:
#     """
#     s_j = E_S[ F(S) - F(S ∪ {j}) ], where F is Chebyshev energy with hard click constraint.
#     We resample S until both subsets are feasible (or give up and skip that draw).
#     """
#     n = len(coords_all)
#     scores = np.zeros(n, dtype=np.float64)
#     stderrs = np.zeros(n, dtype=np.float64)

#     for j in range(n):
#         others = [k for k in range(n) if k != j]
#         deltas = []
#         attempts = 0
#         while len(deltas) < mc_samples and attempts < mc_samples * 40:
#             attempts += 1
#             m = rng.randint(1, len(others)+1)
#             S_idx = list(np.sort(rng.choice(others, size=m, replace=False)))
#             F_S, bS = chebyshev_energy(image, predictor, coords_all[S_idx], labels_all[S_idx],
#                                        multimask, tta_on, jitter_radius, rng, eps, env)
#             if not np.isfinite(F_S):
#                 continue
#             Sj_idx = S_idx + [j]
#             F_Sj, bSj = chebyshev_energy(image, predictor, coords_all[Sj_idx], labels_all[Sj_idx],
#                                          multimask, tta_on, jitter_radius, rng, eps, env)
#             if not np.isfinite(F_Sj):
#                 continue
#             deltas.append(F_S - F_Sj)

#         if len(deltas) == 0:
#             scores[j] = 0.0; stderrs[j] = 0.0
#         else:
#             s = trimmed_mean(deltas, trim_frac)
#             se = stderr(deltas)
#             scores[j] = s; stderrs[j] = se

#     return scores, stderrs


# def greedy_minmax_subset(image: np.ndarray,
#                          predictor: SamPredictor,
#                          coords_all: np.ndarray,
#                          labels_all: np.ndarray,
#                          multimask: bool,
#                          tta_on: bool,
#                          jitter_radius: int,
#                          rng: np.random.RandomState,
#                          eps: float,
#                          env: Dict[str,Dict[str,float]],
#                          plateau: float) -> List[int]:
#     """
#     Greedy forward selection minimizing Chebyshev energy under hard constraint.
#     Stops when marginal drop < plateau.
#     """
#     n = len(coords_all)
#     remaining = list(range(n))
#     selected = []

#     # Start from empty set: if infeasible, add best single prompt
#     F_curr, _ = chebyshev_energy(image, predictor, np.zeros((0,2),np.float32),
#                                  np.zeros((0,),np.int32), multimask, tta_on, jitter_radius, rng, eps, env)
#     if not np.isfinite(F_curr):
#         # pick single j with finite F
#         best = None; bestF = None
#         for j in remaining:
#             F_j, _ = chebyshev_energy(image, predictor, coords_all[[j]], labels_all[[j]],
#                                       multimask, tta_on, jitter_radius, rng, eps, env)
#             if np.isfinite(F_j) and (best is None or F_j < bestF):
#                 best, bestF = j, F_j
#         if best is None:
#             return []  # nothing feasible
#         selected.append(best)
#         remaining.remove(best)
#         F_curr = bestF

#     improved = True
#     while improved and len(remaining) > 0:
#         improved = False
#         best_gain = 0.0; best_j = None; bestF = None
#         for j in list(remaining):
#             idx = selected + [j]
#             F_new, _ = chebyshev_energy(image, predictor, coords_all[idx], labels_all[idx],
#                                         multimask, tta_on, jitter_radius, rng, eps, env)
#             if not np.isfinite(F_new):
#                 continue
#             gain = F_curr - F_new
#             if gain > best_gain:
#                 best_gain = gain; best_j = j; bestF = F_new
#         if best_j is not None and best_gain >= plateau:
#             selected.append(best_j)
#             remaining.remove(best_j)
#             F_curr = bestF
#             improved = True
#         else:
#             break
#     return selected


# # ===================== Visualization (BASE SAM) =====================

# def best_logits_BASE(image: np.ndarray,
#                      predictor: SamPredictor,
#                      coords: np.ndarray,
#                      labels: np.ndarray,
#                      multimask: bool) -> np.ndarray:
#     ups = run_sam_logits(image, predictor, coords, labels, multimask)
#     # pick candidate that minimizes click BCE at T=1 (BASE)
#     best = None; best_ce = None
#     for z in ups:
#         ce = click_consistency_BCE_at_points(z, coords, labels)
#         if best is None or ce < best_ce:
#             best = z; best_ce = ce
#     return best

# def save_side_by_side_BASE(image: np.ndarray,
#                            coords_left: np.ndarray, labels_left: np.ndarray,
#                            coords_right: np.ndarray,  labels_right: np.ndarray,
#                            logits_left: np.ndarray, logits_right: np.ndarray,
#                            title_left: str, title_right: str,
#                            out_path: Path):
#     pL = sigmoid(logits_left); mL = (pL>=0.5).astype(np.uint8)
#     pR = sigmoid(logits_right); mR = (pR>=0.5).astype(np.uint8)
#     ovL = draw_prompt_dots(overlay_mask(image, mL, 0.45), coords_left, labels_left, radius=3)
#     ovR = draw_prompt_dots(overlay_mask(image, mR, 0.45), coords_right,  labels_right,  radius=3)
#     fig, axs = plt.subplots(1,2, figsize=(10.6,5.3))
#     axs[0].imshow(ovL); axs[0].axis('off'); axs[0].set_title(title_left)
#     axs[1].imshow(ovR); axs[1].axis('off'); axs[1].set_title(title_right)
#     plt.tight_layout(); out_path.parent.mkdir(parents=True, exist_ok=True)
#     plt.savefig(out_path, dpi=200); plt.close()


# # ===================== Main =====================

# def main():
#     ap = argparse.ArgumentParser()
#     ap.add_argument("--image", required=True)
#     ap.add_argument("--prompts", required=True)
#     ap.add_argument("--checkpoint", required=True)
#     ap.add_argument("--model-type", default="vit_h", choices=["vit_h","vit_l","vit_b"])
#     ap.add_argument("--out-dir", default="./runs/minmax_refiner")
#     ap.add_argument("--multimask", action="store_true")
#     ap.add_argument("--tta", action="store_true")
#     ap.add_argument("--mc-samples", type=int, default=48, help="MC contexts per prompt for sufficiency")
#     ap.add_argument("--env-samples", type=int, default=96, help="random subsets to estimate envelopes")
#     ap.add_argument("--jitter-radius", type=int, default=2)
#     ap.add_argument("--trim-frac", type=float, default=0.1)
#     ap.add_argument("--plateau", type=float, default=1e-4, help="greedy min–max stopping threshold")
#     ap.add_argument("--eps-mode", choices=["full","quantile"], default="full",
#                     help="how to set hard click ε: 'full' uses FULL set C; 'quantile' uses a quantile over random subsets")
#     ap.add_argument("--eps-quantile", type=float, default=0.3, help="if eps-mode=quantile, use this quantile in [0,1]")
#     ap.add_argument("--eps-scale", type=float, default=1.00, help="scale ε (e.g., 1.05 to loosen slightly)")
#     ap.add_argument("--seed", type=int, default=0)
#     args = ap.parse_args()

#     rng = np.random.RandomState(args.seed)
#     out_dir = Path(args.out_dir)
#     perfile = out_dir/"perfile"; perfile.mkdir(parents=True, exist_ok=True)
#     overlays_dir = out_dir/"overlays"; overlays_dir.mkdir(parents=True, exist_ok=True)

#     # Load
#     image = load_image_rgb(args.image)
#     coords_all, labels_all = load_prompts_json(args.prompts)
#     predictor = build_predictor(args.checkpoint, args.model_type)
#     predictor.set_image(image)

#     # Full set baseline terms
#     C_full, U_full, R_full, E_full = evaluate_terms(image, predictor, coords_all, labels_all,
#                                                     args.multimask, args.tta, args.jitter_radius, rng)

#     # Determine hard ε
#     if args.eps_mode == "full":
#         eps = float(C_full * args.eps_scale)
#     else:
#         # gather C over random subsets
#         Cvals = []
#         n = len(coords_all)
#         for _ in range(max(64, args.env_samples)):
#             m = rng.randint(1, n+1)
#             idx = np.sort(rng.choice(np.arange(n), size=m, replace=False))
#             C, _, _, _ = evaluate_terms(image, predictor, coords_all[idx], labels_all[idx],
#                                         args.multimask, args.tta, args.jitter_radius, rng)
#             Cvals.append(C)
#         q = max(0.0, min(1.0, args.eps_quantile))
#         eps = float(np.quantile(np.asarray(Cvals, dtype=np.float64), q) * args.eps_scale)

#     # Estimate envelopes for normalization using feasible subsets
#     env = estimate_envelopes(image, predictor, coords_all, labels_all,
#                              args.multimask, args.tta, args.jitter_radius, rng,
#                              args.env_samples, eps)

#     # Full-set Chebyshev energy
#     F_full, br_full = chebyshev_energy(image, predictor, coords_all, labels_all,
#                                        args.multimask, args.tta, args.jitter_radius, rng, eps, env)

#     # MC sufficiency with HARD constraint
#     s, se = mc_sufficiency(image, predictor, coords_all, labels_all,
#                            args.multimask, args.tta, args.jitter_radius, rng,
#                            args.mc_samples, args.trim_frac, eps, env)

#     # Tags via 95% CI
#     tags = []
#     for v, sei in zip(s,se):
#         ci = 1.96*sei
#         if (v - ci) > 0:  # CI entirely > 0
#             tags.append("useful")
#         elif (v + ci) < 0:  # CI entirely < 0
#             tags.append("harmful")
#         else:
#             tags.append("redundant")

#     # Refined.json: keep only statistically useful prompts (CI_low > 0)
#     keep_idx = [i for i,t in enumerate(tags) if t=="useful"]
#     prompts_refined = []
#     t_counter = 1
#     for i in sorted(keep_idx):
#         x, y = int(coords_all[i,0]), int(coords_all[i,1])
#         lbl = int(labels_all[i])
#         prompts_refined.append({"t": t_counter, "x": x, "y": y, "label": lbl})
#         t_counter += 1
#     with open(perfile/"refined.json", "w") as f:
#         json.dump({"prompts": prompts_refined}, f, indent=2)

#     # Also compute a balanced min–max subset (optional, useful to inspect)
#     minmax_idx = greedy_minmax_subset(image, predictor, coords_all, labels_all,
#                                       args.multimask, args.tta, args.jitter_radius, rng,
#                                       eps, env, args.plateau)
#     prompts_minmax = []
#     t_counter = 1
#     for i in sorted(minmax_idx):
#         x, y = int(coords_all[i,0]); y2 = int(coords_all[i,1]); lbl = int(labels_all[i])
#         prompts_minmax.append({"t": t_counter, "x": x, "y": y2, "label": lbl}); t_counter += 1
#     with open(perfile/"balanced_minmax.json", "w") as f:
#         json.dump({"prompts": prompts_minmax}, f, indent=2)

#     # Energy breakdowns
#     coords_ref = coords_all[keep_idx] if len(keep_idx)>0 else np.zeros((0,2),dtype=np.float32)
#     labels_ref = labels_all[keep_idx] if len(keep_idx)>0 else np.zeros((0,),dtype=np.int32)
#     F_ref, br_ref = chebyshev_energy(image, predictor, coords_ref, labels_ref,
#                                      args.multimask, args.tta, args.jitter_radius, rng, eps, env)

#     # Save envelopes / ε
#     with open(perfile/"envelopes.json", "w") as f:
#         json.dump({
#             "epsilon_click": eps,
#             "feasible_fraction_for_envelopes": env.get("feasible_frac", 0.0),
#             "U_env": env["U"], "R_env": env["R"], "E_env": env["E"],
#             "C_full": C_full, "U_full": U_full, "R_full": R_full, "E_full": E_full,
#             "F_full": F_full, "F_ref": F_ref
#         }, f, indent=2)

#     # Save energy breakdown CSV
#     with open(perfile/"energy_breakdown.csv", "w", newline="") as f:
#         w = csv.DictWriter(f, fieldnames=["set","N_prompts","C","U","R","E","U_t","R_t","E_t","F"])
#         w.writeheader()
#         w.writerow({"set":"FULL","N_prompts":len(coords_all), **br_full})
#         w.writerow({"set":"REFINED","N_prompts":len(coords_ref), **br_ref})

#     # Sufficiency table
#     with open(perfile/"sufficiency.csv", "w", newline="") as f:
#         w = csv.DictWriter(f, fieldnames=["index","x","y","label","sufficiency","stderr","CI95_low","CI95_high","tag"])
#         w.writeheader()
#         for i,(sc,sei,tag) in enumerate(zip(s,se,tags)):
#             ci = 1.96*sei
#             w.writerow({
#                 "index": i+1,
#                 "x": int(coords_all[i,0]), "y": int(coords_all[i,1]),
#                 "label": int(labels_all[i]),
#                 "sufficiency": float(sc),
#                 "stderr": float(sei),
#                 "CI95_low": float(sc-ci),
#                 "CI95_high": float(sc+ci),
#                 "tag": tag
#             })

#     # Overlays (BASE SAM, no temperature)
#     logits_full_base = best_logits_BASE(image, predictor, coords_all, labels_all, args.multimask)
#     logits_ref_base  = best_logits_BASE(image, predictor, coords_ref, labels_ref, args.multimask)
#     save_side_by_side_BASE(image,
#                            coords_all, labels_all, coords_ref, labels_ref,
#                            logits_full_base, logits_ref_base,
#                            f"FULL (N={len(coords_all)})",
#                            f"REFINED (N={len(coords_ref)})",
#                            overlays_dir/"full_vs_refined_BASE.png")

#     # Optional: also visualize balanced min–max subset
#     coords_min = coords_all[minmax_idx] if len(minmax_idx)>0 else np.zeros((0,2),dtype=np.float32)
#     labels_min = labels_all[minmax_idx] if len(minmax_idx)>0 else np.zeros((0,),dtype=np.int32)
#     logits_min_base  = best_logits_BASE(image, predictor, coords_min, labels_min, args.multimask)
#     save_side_by_side_BASE(image,
#                            coords_all, labels_all, coords_min, labels_min,
#                            logits_full_base, logits_min_base,
#                            f"FULL (N={len(coords_all)})",
#                            f"BALANCED MIN–MAX (N={len(coords_min)})",
#                            overlays_dir/"full_vs_minmax_BASE.png")

#     # Summary
#     with open(out_dir/"summary.txt", "w") as f:
#         f.write(
# f"""GT-free, theory-first prompt refiner (hard click constraint + min–max energy)

# Image: {args.image}
# Prompts: {args.prompts}
# SAM: {args.model_type}

# Click ε: {eps:.6f}  (mode={args.eps_mode}, scale={args.eps_scale})
# Feasible fraction used for envelopes: {env.get("feasible_frac",0.0):.3f}

# FULL:
#   C={br_full['C']:.6f}  U={br_full['U']:.6f}  R={br_full['R']:.6f}  E={br_full['E']:.6f}
#   U_t={br_full['U_t']:.3f} R_t={br_full['R_t']:.3f} E_t={br_full['E_t']:.3f}  F={br_full['F']:.6f}

# REFINED (useful only):
#   N={len(coords_ref)}  C={br_ref['C']:.6f} U={br_ref['U']:.6f} R={br_ref['R']:.6f} E={br_ref['E']:.6f}
#   U_t={br_ref['U_t']:.3f} R_t={br_ref['R_t']:.3f} E_t={br_ref['E_t']:.3f}  F={br_ref['F']:.6f}

# Files:
#   perfile/sufficiency.csv
#   perfile/refined.json
#   perfile/balanced_minmax.json
#   perfile/energy_breakdown.csv
#   perfile/envelopes.json
#   overlays/full_vs_refined_BASE.png
#   overlays/full_vs_minmax_BASE.png
# """
#         )

#     print("\nDone. Results in:", out_dir)


# if __name__ == "__main__":
#     main()


#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GT-FREE, THEORY-FIRST PROMPT REFINER (HARD CLICK CONSTRAINT + MIN–MAX ENERGY)
with optional tqdm progress bars (pass --progress)
"""

import argparse, json, csv, math
from pathlib import Path
from typing import List, Tuple, Optional, Dict

import numpy as np
import cv2
from PIL import Image
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

# Progress bars
from tqdm.auto import tqdm

# ---- SAM ----
try:
    from segment_anything import sam_model_registry, SamPredictor
except Exception as e:
    raise RuntimeError("Install SAM: https://github.com/facebookresearch/segment-anything") from e


# ===================== I/O =====================

def load_image_rgb(path: str) -> np.ndarray:
    return np.array(Image.open(path).convert("RGB"))

def load_prompts_json(path: str) -> Tuple[np.ndarray, np.ndarray]:
    with open(path, "r") as f:
        data = json.load(f)
    arr = sorted(data.get("prompts", []), key=lambda d: d.get("t", 0))
    coords = np.array([[int(d["x"]), int(d["y"])] for d in arr], dtype=np.float32)
    labels = np.array([int(d["label"]) for d in arr], dtype=np.int32)
    return coords, labels


# ===================== Utils =====================

def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))

def draw_prompt_dots(im_rgb: np.ndarray,
                     coords: np.ndarray, labels: np.ndarray,
                     radius: int = 3) -> np.ndarray:
    out = im_rgb.copy()
    H, W = out.shape[:2]
    for (x, y), lab in zip(coords.astype(int), labels.astype(int)):
        x = int(np.clip(x, 0, W-1)); y = int(np.clip(y, 0, H-1))
        bgr = (0,255,0) if lab==1 else (0,0,255)
        cv2.circle(out, (x, y), radius, bgr, thickness=-1, lineType=cv2.LINE_AA)
        cv2.circle(out, (x, y), radius, (0,0,0), thickness=1, lineType=cv2.LINE_AA)
    return out

def overlay_mask(im: np.ndarray, mask_bin: np.ndarray, alpha: float=0.45) -> np.ndarray:
    im = im.copy()
    color = np.array([255, 0, 0], dtype=np.uint8)
    idx = mask_bin.astype(bool)
    im[idx] = (alpha*color + (1-alpha)*im[idx]).astype(np.uint8)
    return im

def jitter_point(x: int, y: int, r: int, H: int, W: int) -> Tuple[int,int]:
    if r <= 0: return x, y
    dx = int(np.round(np.random.uniform(-r, r)))
    dy = int(np.round(np.random.uniform(-r, r)))
    return int(np.clip(x+dx, 0, W-1)), int(np.clip(y+dy, 0, H-1))

def trimmed_mean(vals: List[float], trim_frac: float=0.1) -> float:
    if len(vals) == 0: return 0.0
    arr = np.sort(np.asarray(vals, dtype=np.float64))
    k = int(math.floor(trim_frac*len(arr)))
    if 2*k >= len(arr): return float(arr.mean())
    return float(arr[k:-k].mean())

def stderr(vals: List[float]) -> float:
    if len(vals) <= 1: return 0.0
    a = np.asarray(vals, dtype=np.float64)
    return float(a.std(ddof=1) / math.sqrt(len(vals)))


# ===================== SAM helpers =====================

def build_predictor(checkpoint: str, model_type: str, device: Optional[str]=None) -> SamPredictor:
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    sam = sam_model_registry[model_type](checkpoint=checkpoint).to(device)
    return SamPredictor(sam)

def upsample_logits_to_image(lowres_logits: np.ndarray,
                             predictor: SamPredictor,
                             out_hw: Tuple[int,int]) -> np.ndarray:
    device = next(predictor.model.parameters()).device
    t = torch.from_numpy(lowres_logits).float().unsqueeze(0).unsqueeze(0).to(device)
    H, W = out_hw
    if hasattr(predictor.model, "postprocess_masks"):
        up = predictor.model.postprocess_masks(t, predictor.input_size, predictor.original_size)[0,0]
    else:
        up = F.interpolate(t, size=(H,W), mode="bilinear", align_corners=False)[0,0]
    return up.detach().cpu().numpy()

def run_sam_logits(image_rgb: np.ndarray,
                   predictor: SamPredictor,
                   coords: np.ndarray,
                   labels: np.ndarray,
                   multimask: bool=True) -> List[np.ndarray]:
    """Return list of upsampled logits (H,W) for SAM candidates."""
    H, W = image_rgb.shape[:2]
    with torch.no_grad():
        _, _, lrs = predictor.predict(point_coords=coords,
                                      point_labels=labels,
                                      multimask_output=multimask,
                                      return_logits=True)
    return [upsample_logits_to_image(lrs[i], predictor, (H,W)) for i in range(len(lrs))]


# ===================== Energy terms =====================

def click_consistency_BCE_at_points(logits: np.ndarray,
                                    coords: np.ndarray,
                                    labels: np.ndarray) -> float:
    z = logits.astype(np.float32)
    p = sigmoid(z)
    H, W = p.shape
    def bilinear(pmap, x, y):
        x = np.clip(x, 0, W-1); y = np.clip(y, 0, H-1)
        x0, y0 = int(np.floor(x)), int(np.floor(y))
        x1, y1 = min(x0+1,W-1), min(y0+1,H-1)
        dx, dy = x-x0, y-y0
        v = (1-dx)*(1-dy)*pmap[y0,x0] + dx*(1-dy)*pmap[y0,x1] + (1-dx)*dy*pmap[y1,x0] + dx*dy*pmap[y1,x1]
        return v
    ce = 0.0
    eps = 1e-7
    for (x,y), lab in zip(coords, labels):
        pv = float(bilinear(p, float(x), float(y)))
        pv = min(max(pv, eps), 1-eps)
        ce += -math.log(pv) if lab == 1 else -math.log(1.0 - pv)
    return ce / max(1, len(labels))

def tv_on_prob(p: np.ndarray) -> float:
    gx = np.diff(p, axis=1, append=p[:,-1:])
    gy = np.diff(p, axis=0, append=p[-1:,:])
    return float(np.mean(np.sqrt(gx*gx + gy*gy) + 1e-8))

def mask_perimeter_and_area(mask: np.ndarray) -> Tuple[float,float]:
    mask_u8 = (mask.astype(np.uint8))*255
    cnts, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    L = 0.0
    for c in cnts:
        L += float(cv2.arcLength(c, True))
    A = float(mask.sum())
    return L, A

def topo_penalty(mask: np.ndarray) -> float:
    num_labels, _ = cv2.connectedComponents(mask.astype(np.uint8), connectivity=8)
    K = int(num_labels) - 1
    filled = cv2.morphologyEx(mask.astype(np.uint8)*255, cv2.MORPH_CLOSE, np.ones((3,3),np.uint8))
    holes = int(cv2.countNonZero(filled - mask.astype(np.uint8)*255) > 0)
    return 0.1*max(0, K-1) + 0.2*holes

def entropy_inside_outside(image_rgb: np.ndarray,
                           mask: np.ndarray,
                           nbins: int = 64) -> float:
    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2LAB)[...,0]  # 0..255
    m = mask.astype(bool)
    if m.sum()==0 or (~m).sum()==0:
        return 0.0
    def H(vals):
        hist, _ = np.histogram(vals, bins=nbins, range=(0,255), density=True)
        hist = np.clip(hist, 1e-9, None)
        hist = hist / hist.sum()
        return float(-(hist*np.log(hist)).sum())
    Hin = H(gray[m]); Hout = H(gray[~m])
    return Hin + Hout

def pairwise_iou(masks: List[np.ndarray]) -> float:
    if len(masks) < 2: return 1.0
    n = len(masks)
    tot = 0.0; cnt = 0
    for i in range(n):
        for j in range(i+1, n):
            a = masks[i].astype(bool); b = masks[j].astype(bool)
            inter = float(np.logical_and(a,b).sum())
            uni = float(np.logical_or(a,b).sum())
            tot += (1.0 if uni==0 else inter/uni); cnt += 1
    return tot / max(1, cnt)


# ===================== TTA =====================

def tta_configs(enable: bool):
    if not enable:
        return [("id", None)]
    return [("id", None), ("hflip", None), ("rot", 3), ("rot", -3), ("scale", 0.97), ("scale", 1.03)]

def apply_tta(image: np.ndarray, coords: np.ndarray, kind: str, param) -> Tuple[np.ndarray, np.ndarray, callable]:
    H, W = image.shape[:2]
    if kind == "id":
        def inv_coords(xy): return xy
        return image, coords, inv_coords
    if kind == "hflip":
        img2 = np.ascontiguousarray(image[:, ::-1, :])
        coords2 = coords.copy(); coords2[:,0] = (W-1) - coords2[:,0]
        def inv_coords(xy):
            xy2 = xy.copy(); xy2[:,0] = (W-1) - xy2[:,0]; return xy2
        return img2, coords2, inv_coords
    if kind == "rot":
        angle = float(param)
        M = cv2.getRotationMatrix2D((W/2, H/2), angle, 1.0)
        img2 = cv2.warpAffine(image, M, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
        ones = np.ones((coords.shape[0],1), dtype=np.float32)
        xy1 = np.hstack([coords, ones]); xy2 = (M @ xy1.T).T
        Minv = cv2.invertAffineTransform(M)
        def inv_coords(xy): return (Minv @ np.hstack([xy, np.ones((xy.shape[0],1),dtype=np.float32)]).T).T
        return img2, xy2.astype(np.float32), inv_coords
    if kind == "scale":
        s = float(param)
        M = np.array([[s,0,(1-s)*W/2],[0,s,(1-s)*H/2]], dtype=np.float32)
        img2 = cv2.warpAffine(image, M, (W,H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
        ones = np.ones((coords.shape[0],1), dtype=np.float32)
        xy1 = np.hstack([coords, ones]); xy2 = (M @ xy1.T).T
        Minv = cv2.invertAffineTransform(M)
        def inv_coords(xy): return (Minv @ np.hstack([xy, np.ones((xy.shape[0],1),dtype=np.float32)]).T).T
        return img2, xy2.astype(np.float32), inv_coords
    raise ValueError("unknown TTA kind")


# ===================== Core eval =====================

def evaluate_terms(image: np.ndarray,
                   predictor: SamPredictor,
                   coords: np.ndarray,
                   labels: np.ndarray,
                   multimask: bool,
                   tta_on: bool,
                   jitter_radius: int,
                   rng: np.random.RandomState) -> Tuple[float,float,float,float]:
    """
    Compute (C, U, R, E) for a subset, aggregating over TTA + jitter.
    Candidate selection per TTA is by *lowest click BCE*.
    """
    H, W = image.shape[:2]
    confs = tta_configs(tta_on)

    prob_maps = []
    masks = []
    click_ce_list = []

    for kind, param in confs:
        coords_j = coords.copy()
        for i in range(coords_j.shape[0]):
            coords_j[i,0], coords_j[i,1] = jitter_point(int(coords_j[i,0]), int(coords_j[i,1]),
                                                         jitter_radius, H, W)
        img_t, coords_t, inv = apply_tta(image, coords_j, kind, param)
        predictor.set_image(img_t)
        ups = run_sam_logits(img_t, predictor, coords_t, labels, multimask)

        best = None; best_click_ce = None
        for z in ups:
            ce = click_consistency_BCE_at_points(z, coords_t, labels)
            if best is None or ce < best_click_ce:
                best = z; best_click_ce = ce

        p_t = sigmoid(best.astype(np.float32))
        if kind == "id":
            p_back = p_t
        elif kind == "hflip":
            p_back = p_t[:, ::-1]
        elif kind == "rot":
            angle = float(param)
            M = cv2.getRotationMatrix2D((W/2, H/2), -angle, 1.0)
            p_back = cv2.warpAffine(p_t, M, (W,H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
        elif kind == "scale":
            s = float(param)
            Minv = cv2.invertAffineTransform(np.array([[s,0,(1-s)*W/2],[0,s,(1-s)*H/2]], dtype=np.float32))
            p_back = cv2.warpAffine(p_t, Minv, (W,H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
        else:
            p_back = p_t

        prob_maps.append(p_back)
        masks.append((p_back>=0.5).astype(np.uint8))
        click_ce_list.append(float(best_click_ce))

    p_mean = np.mean(np.stack(prob_maps,0), axis=0)
    m_mean = (p_mean>=0.5).astype(np.uint8)

    C_term = float(np.mean(click_ce_list))

    eps = 1e-7
    H_pred = float(np.mean(-(np.clip(p_mean,eps,1-eps)*np.log(np.clip(p_mean,eps,1-eps)) +
                            np.clip(1-p_mean,eps,1-eps)*np.log(np.clip(1-p_mean,eps,1-eps)))))
    mean_iou = pairwise_iou(masks)
    U_term = H_pred + (1.0 - mean_iou)

    tv = tv_on_prob(p_mean)
    L, A = mask_perimeter_and_area(m_mean)
    comp = (L*L) / max(A, 1.0)
    topo = topo_penalty(m_mean)
    R_term = tv + 0.2*comp + 0.05*topo

    E_info = entropy_inside_outside(image, m_mean, nbins=64)

    return C_term, U_term, R_term, E_info


def estimate_envelopes(image: np.ndarray,
                       predictor: SamPredictor,
                       coords_all: np.ndarray,
                       labels_all: np.ndarray,
                       multimask: bool,
                       tta_on: bool,
                       jitter_radius: int,
                       rng: np.random.RandomState,
                       env_samples: int,
                       eps: float,
                       show_pb: bool=False) -> Dict[str, Dict[str,float]]:
    """
    Per-image normalization envelopes for U,R,E using feasible random subsets (C ≤ ε).
    """
    n = len(coords_all)
    Uvals, Rvals, Evals = [], [], []
    feas = 0; tried = 0

    iterator = range(env_samples)
    if show_pb:
        iterator = tqdm(iterator, total=env_samples, desc="Envelopes", leave=False)

    for _ in iterator:
        m = rng.randint(1, n+1)
        idx = np.sort(rng.choice(np.arange(n), size=m, replace=False))
        C, U, R, E = evaluate_terms(image, predictor, coords_all[idx], labels_all[idx],
                                    multimask, tta_on, jitter_radius, rng)
        tried += 1
        if C <= eps:
            feas += 1
            Uvals.append(U); Rvals.append(R); Evals.append(E)

    # include FULL if feasible
    C_full, U_full, R_full, E_full = evaluate_terms(image, predictor, coords_all, labels_all,
                                                    multimask, tta_on, jitter_radius, rng)
    if C_full <= eps:
        Uvals.append(U_full); Rvals.append(R_full); Evals.append(E_full)

    def stats(arr):
        if len(arr)==0:
            return {"best": 0.0, "rand": 1.0}
        arr = np.asarray(arr, dtype=np.float64)
        return {"best": float(np.min(arr)), "rand": float(np.median(arr))}
    env = {
        "U": stats(Uvals),
        "R": stats(Rvals),
        "E": stats(Evals),
        "feasible_frac": float(feas / max(1,tried))
    }
    return env


def normalize_term(val: float, best: float, rand: float) -> float:
    den = max(1e-9, rand - best)
    return float(np.clip((val - best) / den, 0.0, 1.0))


def chebyshev_energy(image: np.ndarray,
                     predictor: SamPredictor,
                     coords: np.ndarray,
                     labels: np.ndarray,
                     multimask: bool,
                     tta_on: bool,
                     jitter_radius: int,
                     rng: np.random.RandomState,
                     eps: float,
                     env: Dict[str,Dict[str,float]]) -> Tuple[float, Dict[str,float]]:
    """
    Hard constraint: if C > eps, return inf.
    Else: Chebyshev (max of normalized U,R,E) with breakdown.
    """
    C, U, R, E = evaluate_terms(image, predictor, coords, labels, multimask, tta_on, jitter_radius, rng)
    if C > eps:
        return float("inf"), {"C": C, "U": U, "R": R, "E": E, "U_t": 1.0, "R_t": 1.0, "E_t": 1.0, "F": float("inf")}
    U_t = normalize_term(U, env["U"]["best"], env["U"]["rand"])
    R_t = normalize_term(R, env["R"]["best"], env["R"]["rand"])
    E_t = normalize_term(E, env["E"]["best"], env["E"]["rand"])
    F_val = float(max(U_t, R_t, E_t))
    return F_val, {"C": C, "U": U, "R": R, "E": E, "U_t": U_t, "R_t": R_t, "E_t": E_t, "F": F_val}


# ===================== MC sufficiency & selection =====================

def mc_sufficiency(image: np.ndarray,
                   predictor: SamPredictor,
                   coords_all: np.ndarray,
                   labels_all: np.ndarray,
                   multimask: bool,
                   tta_on: bool,
                   jitter_radius: int,
                   rng: np.random.RandomState,
                   mc_samples: int,
                   trim_frac: float,
                   eps: float,
                   env: Dict[str,Dict[str,float]],
                   show_pb: bool=False) -> Tuple[np.ndarray, np.ndarray]:
    """
    s_j = E_S[ F(S) - F(S∪{j}) ] with hard click constraint. Monte Carlo with trimmed mean.
    """
    n = len(coords_all)
    scores = np.zeros(n, dtype=np.float64)
    stderrs = np.zeros(n, dtype=np.float64)

    prompt_iter = range(n)
    if show_pb:
        prompt_iter = tqdm(prompt_iter, total=n, desc="Sufficiency (prompts)")

    for j in prompt_iter:
        others = [k for k in range(n) if k != j]
        deltas = []
        attempts = 0

        inner_pb = None
        if show_pb:
            inner_pb = tqdm(total=mc_samples, desc=f"p{j+1}", leave=False)

        while len(deltas) < mc_samples and attempts < mc_samples * 40:
            attempts += 1
            m = rng.randint(1, len(others)+1)
            S_idx = list(np.sort(rng.choice(others, size=m, replace=False)))
            F_S, _ = chebyshev_energy(image, predictor, coords_all[S_idx], labels_all[S_idx],
                                      multimask, tta_on, jitter_radius, rng, eps, env)
            if not np.isfinite(F_S):
                continue
            Sj_idx = S_idx + [j]
            F_Sj, _ = chebyshev_energy(image, predictor, coords_all[Sj_idx], labels_all[Sj_idx],
                                       multimask, tta_on, jitter_radius, rng, eps, env)
            if not np.isfinite(F_Sj):
                continue
            deltas.append(F_S - F_Sj)
            if inner_pb: inner_pb.update(1)

        if inner_pb: inner_pb.close()

        if len(deltas) == 0:
            scores[j] = 0.0; stderrs[j] = 0.0
        else:
            s = trimmed_mean(deltas, trim_frac)
            se = stderr(deltas)
            scores[j] = s; stderrs[j] = se

    return scores, stderrs


def greedy_minmax_subset(image: np.ndarray,
                         predictor: SamPredictor,
                         coords_all: np.ndarray,
                         labels_all: np.ndarray,
                         multimask: bool,
                         tta_on: bool,
                         jitter_radius: int,
                         rng: np.random.RandomState,
                         eps: float,
                         env: Dict[str,Dict[str,float]],
                         plateau: float,
                         show_pb: bool=False) -> List[int]:
    """
    Greedy forward selection minimizing Chebyshev energy under hard constraint.
    """
    n = len(coords_all)
    remaining = list(range(n))
    selected = []

    F_curr, _ = chebyshev_energy(image, predictor, np.zeros((0,2),np.float32),
                                 np.zeros((0,),np.int32), multimask, tta_on, jitter_radius, rng, eps, env)
    if not np.isfinite(F_curr):
        best = None; bestF = None
        for j in remaining:
            F_j, _ = chebyshev_energy(image, predictor, coords_all[[j]], labels_all[[j]],
                                      multimask, tta_on, jitter_radius, rng, eps, env)
            if np.isfinite(F_j) and (best is None or F_j < bestF):
                best, bestF = j, F_j
        if best is None:
            return []
        selected.append(best); remaining.remove(best); F_curr = bestF

    pbar = tqdm(total=n, desc="Min–max select", leave=False, disable=not show_pb)

    improved = True
    while improved and len(remaining) > 0:
        improved = False
        best_gain = 0.0; best_j = None; bestF = None
        for j in list(remaining):
            idx = selected + [j]
            F_new, _ = chebyshev_energy(image, predictor, coords_all[idx], labels_all[idx],
                                        multimask, tta_on, jitter_radius, rng, eps, env)
            if not np.isfinite(F_new):
                continue
            gain = F_curr - F_new
            if gain > best_gain:
                best_gain = gain; best_j = j; bestF = F_new
        if best_j is not None and best_gain >= plateau:
            selected.append(best_j)
            remaining.remove(best_j)
            F_curr = bestF
            improved = True
            pbar.update(1)
        else:
            break
    pbar.close()
    return selected


# ===================== Visualization (BASE SAM) =====================

def best_logits_BASE(image: np.ndarray,
                     predictor: SamPredictor,
                     coords: np.ndarray,
                     labels: np.ndarray,
                     multimask: bool) -> np.ndarray:
    ups = run_sam_logits(image, predictor, coords, labels, multimask)
    best = None; best_ce = None
    for z in ups:
        ce = click_consistency_BCE_at_points(z, coords, labels)
        if best is None or ce < best_ce:
            best = z; best_ce = ce
    return best

def save_side_by_side_BASE(image: np.ndarray,
                           coords_left: np.ndarray, labels_left: np.ndarray,
                           coords_right: np.ndarray,  labels_right: np.ndarray,
                           logits_left: np.ndarray, logits_right: np.ndarray,
                           title_left: str, title_right: str,
                           out_path: Path):
    pL = sigmoid(logits_left); mL = (pL>=0.5).astype(np.uint8)
    pR = sigmoid(logits_right); mR = (pR>=0.5).astype(np.uint8)
    ovL = draw_prompt_dots(overlay_mask(image, mL, 0.45), coords_left, labels_left, radius=3)
    ovR = draw_prompt_dots(overlay_mask(image, mR, 0.45), coords_right,  labels_right,  radius=3)
    fig, axs = plt.subplots(1,2, figsize=(10.6,5.3))
    axs[0].imshow(ovL); axs[0].axis('off'); axs[0].set_title(title_left)
    axs[1].imshow(ovR); axs[1].axis('off'); axs[1].set_title(title_right)
    plt.tight_layout(); out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=200); plt.close()


# ===================== Main =====================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True)
    ap.add_argument("--prompts", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--model-type", default="vit_h", choices=["vit_h","vit_l","vit_b"])
    ap.add_argument("--out-dir", default="./runs/minmax_refiner")
    ap.add_argument("--multimask", action="store_true")
    ap.add_argument("--tta", action="store_true")
    ap.add_argument("--mc-samples", type=int, default=48, help="MC contexts per prompt for sufficiency")
    ap.add_argument("--env-samples", type=int, default=96, help="random subsets to estimate envelopes")
    ap.add_argument("--jitter-radius", type=int, default=2)
    ap.add_argument("--trim-frac", type=float, default=0.1)
    ap.add_argument("--plateau", type=float, default=1e-4, help="greedy min–max stopping threshold")
    ap.add_argument("--eps-mode", choices=["full","quantile"], default="full",
                    help="how to set hard click ε: 'full' uses FULL set C; 'quantile' uses a quantile over random subsets")
    ap.add_argument("--eps-quantile", type=float, default=0.3, help="if eps-mode=quantile, use this quantile in [0,1]")
    ap.add_argument("--eps-scale", type=float, default=1.00, help="scale ε (e.g., 1.05 to loosen slightly)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--progress", action="store_true", help="show tqdm progress bars")
    args = ap.parse_args()

    rng = np.random.RandomState(args.seed)
    out_dir = Path(args.out_dir)
    perfile = out_dir/"perfile"; perfile.mkdir(parents=True, exist_ok=True)
    overlays_dir = out_dir/"overlays"; overlays_dir.mkdir(parents=True, exist_ok=True)

    # Load
    image = load_image_rgb(args.image)
    coords_all, labels_all = load_prompts_json(args.prompts)
    predictor = build_predictor(args.checkpoint, args.model_type)
    predictor.set_image(image)

    # Full set baseline terms (quick)
    C_full, U_full, R_full, E_full = evaluate_terms(image, predictor, coords_all, labels_all,
                                                    args.multimask, args.tta, args.jitter_radius, rng)

    # Determine hard ε
    if args.eps_mode == "full":
        eps = float(C_full * args.eps_scale)
    else:
        Cvals = []
        n = len(coords_all)
        it = range(max(64, args.env_samples))
        if args.progress:
            it = tqdm(it, desc="ε quantile sampling", leave=False)
        for _ in it:
            m = rng.randint(1, n+1)
            idx = np.sort(rng.choice(np.arange(n), size=m, replace=False))
            C, _, _, _ = evaluate_terms(image, predictor, coords_all[idx], labels_all[idx],
                                        args.multimask, args.tta, args.jitter_radius, rng)
            Cvals.append(C)
        q = max(0.0, min(1.0, args.eps_quantile))
        eps = float(np.quantile(np.asarray(Cvals, dtype=np.float64), q) * args.eps_scale)

    # Envelopes for normalization (with progress)
    env = estimate_envelopes(image, predictor, coords_all, labels_all,
                             args.multimask, args.tta, args.jitter_radius, rng,
                             args.env_samples, eps, show_pb=args.progress)

    # Full-set Chebyshev energy
    F_full, br_full = chebyshev_energy(image, predictor, coords_all, labels_all,
                                       args.multimask, args.tta, args.jitter_radius, rng, eps, env)

    # MC sufficiency (with progress)
    s, se = mc_sufficiency(image, predictor, coords_all, labels_all,
                           args.multimask, args.tta, args.jitter_radius, rng,
                           args.mc_samples, args.trim_frac, eps, env, show_pb=args.progress)

    # Tags via 95% CI
    tags = []
    for v, sei in zip(s,se):
        ci = 1.96*sei
        if (v - ci) > 0:
            tags.append("useful")
        elif (v + ci) < 0:
            tags.append("harmful")
        else:
            tags.append("redundant")

    # Refined.json (keep useful)
    keep_idx = [i for i,t in enumerate(tags) if t=="useful"]
    prompts_refined = []
    t_counter = 1
    for i in sorted(keep_idx):
        x, y = int(coords_all[i,0]), int(coords_all[i,1])
        lbl = int(labels_all[i])
        prompts_refined.append({"t": t_counter, "x": x, "y": y, "label": lbl})
        t_counter += 1
    (perfile/"refined.json").write_text(json.dumps({"prompts": prompts_refined}, indent=2))

    # Also: balanced min–max subset (greedy) with progress
    minmax_idx = greedy_minmax_subset(image, predictor, coords_all, labels_all,
                                      args.multimask, args.tta, args.jitter_radius, rng,
                                      eps, env, args.plateau, show_pb=args.progress)
    prompts_minmax = []
    t_counter = 1
    for i in sorted(minmax_idx):
        x, y = int(coords_all[i,0]); y2 = int(coords_all[i,1]); lbl = int(labels_all[i])
        prompts_minmax.append({"t": t_counter, "x": x, "y": y2, "label": lbl}); t_counter += 1
    (perfile/"balanced_minmax.json").write_text(json.dumps({"prompts": prompts_minmax}, indent=2))

    # Energy breakdowns
    coords_ref = coords_all[keep_idx] if len(keep_idx)>0 else np.zeros((0,2),dtype=np.float32)
    labels_ref = labels_all[keep_idx] if len(keep_idx)>0 else np.zeros((0,),dtype=np.int32)
    F_ref, br_ref = chebyshev_energy(image, predictor, coords_ref, labels_ref,
                                     args.multimask, args.tta, args.jitter_radius, rng, eps, env)

    with open(perfile/"envelopes.json", "w") as f:
        json.dump({
            "epsilon_click": eps,
            "feasible_fraction_for_envelopes": env.get("feasible_frac", 0.0),
            "U_env": env["U"], "R_env": env["R"], "E_env": env["E"],
            "C_full": C_full, "U_full": U_full, "R_full": R_full, "E_full": E_full,
            "F_full": F_full, "F_ref": F_ref
        }, f, indent=2)

    with open(perfile/"energy_breakdown.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["set","N_prompts","C","U","R","E","U_t","R_t","E_t","F"])
        w.writeheader()
        w.writerow({"set":"FULL","N_prompts":len(coords_all), **br_full})
        w.writerow({"set":"REFINED","N_prompts":len(coords_ref), **br_ref})

    # Sufficiency table
    with open(perfile/"sufficiency.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["index","x","y","label","sufficiency","stderr","CI95_low","CI95_high","tag"])
        w.writeheader()
        for i,(sc,sei,tag) in enumerate(zip(s,se,tags)):
            ci = 1.96*sei
            w.writerow({
                "index": i+1,
                "x": int(coords_all[i,0]), "y": int(coords_all[i,1]),
                "label": int(labels_all[i]),
                "sufficiency": float(sc),
                "stderr": float(sei),
                "CI95_low": float(sc-ci),
                "CI95_high": float(sc+ci),
                "tag": tag
            })

    # Overlays (BASE SAM)
    logits_full_base = best_logits_BASE(image, predictor, coords_all, labels_all, args.multimask)
    logits_ref_base  = best_logits_BASE(image, predictor, coords_ref, labels_ref, args.multimask)
    save_side_by_side_BASE(image,
                           coords_all, labels_all, coords_ref, labels_ref,
                           logits_full_base, logits_ref_base,
                           f"FULL (N={len(coords_all)})",
                           f"REFINED (N={len(coords_ref)})",
                           overlays_dir/"full_vs_refined_BASE.png")

    coords_min = coords_all[minmax_idx] if len(minmax_idx)>0 else np.zeros((0,2),dtype=np.float32)
    labels_min = labels_all[minmax_idx] if len(minmax_idx)>0 else np.zeros((0,),dtype=np.int32)
    logits_min_base  = best_logits_BASE(image, predictor, coords_min, labels_min, args.multimask)
    save_side_by_side_BASE(image,
                           coords_all, labels_all, coords_min, labels_min,
                           logits_full_base, logits_min_base,
                           f"FULL (N={len(coords_all)})",
                           f"BALANCED MIN–MAX (N={len(coords_min)})",
                           overlays_dir/"full_vs_minmax_BASE.png")

    with open(out_dir/"summary.txt", "w") as f:
        f.write(
f"""GT-free, theory-first prompt refiner (hard click constraint + min–max energy)

Image: {args.image}
Prompts: {args.prompts}
SAM: {args.model_type}

Click ε: {eps:.6f}  (mode={args.eps_mode}, scale={args.eps_scale})
Feasible fraction used for envelopes: {env.get("feasible_frac",0.0):.3f}

FULL:
  C={br_full['C']:.6f}  U={br_full['U']:.6f}  R={br_full['R']:.6f}  E={br_full['E']:.6f}
  U_t={br_full['U_t']:.3f} R_t={br_full['R_t']:.3f} E_t={br_full['E_t']:.3f}  F={br_full['F']:.6f}

REFINED (useful only):
  N={len(coords_ref)}  C={br_ref['C']:.6f} U={br_ref['U']:.6f} R={br_ref['R']:.6f} E={br_ref['E']:.6f}
  U_t={br_ref['U_t']:.3f} R_t={br_ref['R_t']:.3f} E_t={br_ref['E_t']:.3f}  F={br_ref['F']:.6f}

Files:
  perfile/sufficiency.csv
  perfile/refined.json
  perfile/balanced_minmax.json
  perfile/energy_breakdown.csv
  perfile/envelopes.json
  overlays/full_vs_refined_BASE.png
  overlays/full_vs_minmax_BASE.png
"""
        )

    print("\nDone. Results in:", out_dir)


if __name__ == "__main__":
    main()
