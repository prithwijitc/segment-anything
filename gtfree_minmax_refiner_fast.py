#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GT-FREE, THEORY-FIRST PROMPT REFINER (HARD CLICK CONSTRAINT + MIN–MAX ENERGY)
FAST VERSION: reuses SAM embeddings per TTA and precomputes warps & LAB-L channel.
Functionality is identical to the previous version.

Run (example):
  python gtfree_minmax_refiner_fast.py \
    --image path/to/image.jpg \
    --prompts path/to/full_prompts.json \
    --checkpoint /path/to/sam_vit_h_4b8939.pth \
    --model-type vit_h \
    --multimask --tta --progress
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


# ===================== TTA =====================

def tta_configs(enable: bool):
    if not enable:
        return [("id", None)]
    return [("id", None), ("hflip", None), ("rot", 3), ("rot", -3), ("scale", 0.97), ("scale", 1.03)]


# ===================== SAM helpers =====================

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
        return (1-dx)*(1-dy)*pmap[y0,x0] + dx*(1-dy)*pmap[y0,x1] + (1-dx)*dy*pmap[y1,x0] + dx*dy*pmap[y1,x1]
    ce = 0.0
    eps = 1e-7
    for (x,y), lab in zip(coords, labels):
        pv = float(bilinear(p, float(x), float(y)))
        pv = min(max(pv, eps), 1-eps)
        ce += -math.log(pv) if lab == 1 else -math.log(1.0 - pv)
    return ce / max(1, len(labels))


# ===================== Precomputed TTA contexts =====================

class TTAContext:
    def __init__(self, name: str, predictor: SamPredictor,
                 inv_kind: str, Minv: Optional[np.ndarray], W: int, H: int):
        self.name = name
        self.pred = predictor         # SamPredictor with set_image already called
        self.inv_kind = inv_kind      # "id" | "hflip" | "affine"
        self.Minv = Minv              # 2x3 float32, if inv_kind == "affine"
        self.W = W
        self.H = H

def build_tta_contexts(sam_model, image_rgb: np.ndarray, tta_list: List[Tuple[str,Optional[float]]]) -> List[TTAContext]:
    H, W = image_rgb.shape[:2]
    ctxs: List[TTAContext] = []
    device = next(sam_model.parameters()).device
    for kind, param in tta_list:
        if kind == "id":
            img_t = image_rgb
            pred = SamPredictor(sam_model)
            pred.set_image(img_t)
            ctxs.append(TTAContext("id", pred, "id", None, W, H))
        elif kind == "hflip":
            img_t = np.ascontiguousarray(image_rgb[:, ::-1, :])
            pred = SamPredictor(sam_model)
            pred.set_image(img_t)
            ctxs.append(TTAContext("hflip", pred, "hflip", None, W, H))
        elif kind == "rot":
            angle = float(param)
            M = cv2.getRotationMatrix2D((W/2, H/2), angle, 1.0).astype(np.float32)
            img_t = cv2.warpAffine(image_rgb, M, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
            Minv = cv2.invertAffineTransform(M).astype(np.float32)
            pred = SamPredictor(sam_model)
            pred.set_image(img_t)
            ctxs.append(TTAContext(f"rot{angle:+.0f}", pred, "affine", Minv, W, H))
        elif kind == "scale":
            s = float(param)
            M = np.array([[s,0,(1-s)*W/2],[0,s,(1-s)*H/2]], dtype=np.float32)
            img_t = cv2.warpAffine(image_rgb, M, (W,H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
            Minv = cv2.invertAffineTransform(M).astype(np.float32)
            pred = SamPredictor(sam_model)
            pred.set_image(img_t)
            ctxs.append(TTAContext(f"scale{s:.2f}", pred, "affine", Minv, W, H))
        else:
            raise ValueError("unknown TTA kind")
    return ctxs

def warp_back_prob(p_t: np.ndarray, ctx: TTAContext) -> np.ndarray:
    if ctx.inv_kind == "id":
        return p_t
    if ctx.inv_kind == "hflip":
        return p_t[:, ::-1]
    # affine inverse
    return cv2.warpAffine(p_t, ctx.Minv, (ctx.W, ctx.H),
                          flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)

def run_sam_logits_ctx(ctx: TTAContext,
                       coords_t: np.ndarray,
                       labels: np.ndarray,
                       multimask: bool=True) -> List[np.ndarray]:
    """Return upsampled logits (H,W) using a context predictor already set with TTA image."""
    with torch.no_grad():
        _, _, lrs = ctx.pred.predict(point_coords=coords_t,
                                     point_labels=labels,
                                     multimask_output=multimask,
                                     return_logits=True)
    return [upsample_logits_to_image(lrs[i], ctx.pred, (ctx.H, ctx.W)) for i in range(len(lrs))]


# ===================== Energy terms =====================

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

def entropy_inside_outside_from_L(grayL: np.ndarray,
                                  mask: np.ndarray,
                                  nbins: int = 64) -> float:
    m = mask.astype(bool)
    if m.sum()==0 or (~m).sum()==0:
        return 0.0
    def H(vals):
        hist, _ = np.histogram(vals, bins=nbins, range=(0,255), density=True)
        hist = np.clip(hist, 1e-9, None)
        hist = hist / hist.sum()
        return float(-(hist*np.log(hist)).sum())
    Hin = H(grayL[m]); Hout = H(grayL[~m])
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


# ===================== Core eval (FAST: reuse TTA contexts) =====================

def evaluate_terms_fast(grayL: np.ndarray,
                        ctxs: List[TTAContext],
                        coords: np.ndarray,
                        labels: np.ndarray,
                        jitter_radius: int,
                        rng: np.random.RandomState) -> Tuple[float,float,float,float]:
    """
    Compute (C, U, R, E) with prebuilt TTA contexts (no re-embedding).
    Candidate selection per TTA is by lowest click BCE.
    """
    H, W = grayL.shape[:2]

    prob_maps = []
    masks = []
    click_ce_list = []

    for ctx in ctxs:
        # jitter coords in ORIGINAL frame, then map to ctx frame
        coords_j = coords.copy()
        for i in range(coords_j.shape[0]):
            coords_j[i,0], coords_j[i,1] = jitter_point(int(coords_j[i,0]), int(coords_j[i,1]),
                                                         jitter_radius, H, W)
        # forward map coords to ctx frame
        if ctx.name == "id":
            coords_t = coords_j
        elif ctx.name == "hflip":
            coords_t = coords_j.copy()
            coords_t[:,0] = (ctx.W-1) - coords_t[:,0]
        else:
            # affine forward = inverse of Minv
            M = cv2.invertAffineTransform(ctx.Minv).astype(np.float32)
            ones = np.ones((coords_j.shape[0],1), dtype=np.float32)
            xy1 = np.hstack([coords_j, ones])
            xy2 = (M @ xy1.T).T
            coords_t = xy2.astype(np.float32)

        ups = run_sam_logits_ctx(ctx, coords_t, labels, multimask=True)

        # choose candidate minimizing click BCE in ctx frame
        best = None; best_click_ce = None
        for z in ups:
            ce = click_consistency_BCE_at_points(z, coords_t, labels)
            if best is None or ce < best_click_ce:
                best = z; best_click_ce = ce

        p_t = sigmoid(best.astype(np.float32))
        p_back = warp_back_prob(p_t, ctx)

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

    E_info = entropy_inside_outside_from_L(grayL, m_mean, nbins=64)

    return C_term, U_term, R_term, E_info


# ===================== Normalization & Chebyshev =====================

def estimate_envelopes_fast(grayL: np.ndarray,
                            ctxs: List[TTAContext],
                            coords_all: np.ndarray,
                            labels_all: np.ndarray,
                            rng: np.random.RandomState,
                            env_samples: int,
                            eps: float,
                            show_pb: bool=False) -> Dict[str, Dict[str,float]]:
    n = len(coords_all)
    Uvals, Rvals, Evals = [], [], []
    feas = 0; tried = 0

    iterator = range(env_samples)
    if show_pb:
        iterator = tqdm(iterator, total=env_samples, desc="Envelopes", leave=False)

    for _ in iterator:
        m = rng.randint(1, n+1)
        idx = np.sort(rng.choice(np.arange(n), size=m, replace=False))
        C, U, R, E = evaluate_terms_fast(grayL, ctxs, coords_all[idx], labels_all[idx], jitter_radius=2, rng=rng)
        tried += 1
        if C <= eps:
            feas += 1
            Uvals.append(U); Rvals.append(R); Evals.append(E)

    # include FULL if feasible
    C_full, U_full, R_full, E_full = evaluate_terms_fast(grayL, ctxs, coords_all, labels_all, jitter_radius=2, rng=rng)
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

def chebyshev_energy_fast(grayL: np.ndarray,
                          ctxs: List[TTAContext],
                          coords: np.ndarray,
                          labels: np.ndarray,
                          rng: np.random.RandomState,
                          eps: float,
                          env: Dict[str,Dict[str,float]]) -> Tuple[float, Dict[str,float]]:
    C, U, R, E = evaluate_terms_fast(grayL, ctxs, coords, labels, jitter_radius=2, rng=rng)
    if C > eps:
        return float("inf"), {"C": C, "U": U, "R": R, "E": E, "U_t": 1.0, "R_t": 1.0, "E_t": 1.0, "F": float("inf")}
    U_t = normalize_term(U, env["U"]["best"], env["U"]["rand"])
    R_t = normalize_term(R, env["R"]["best"], env["R"]["rand"])
    E_t = normalize_term(E, env["E"]["best"], env["E"]["rand"])
    F_val = float(max(U_t, R_t, E_t))
    return F_val, {"C": C, "U": U, "R": R, "E": E, "U_t": U_t, "R_t": R_t, "E_t": E_t, "F": F_val}


# ===================== MC sufficiency & selection =====================

def mc_sufficiency_fast(grayL: np.ndarray,
                        ctxs: List[TTAContext],
                        coords_all: np.ndarray,
                        labels_all: np.ndarray,
                        rng: np.random.RandomState,
                        mc_samples: int,
                        trim_frac: float,
                        eps: float,
                        env: Dict[str,Dict[str,float]],
                        show_pb: bool=False) -> Tuple[np.ndarray, np.ndarray]:
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
            m = np.random.randint(1, len(others)+1)
            S_idx = list(np.sort(np.random.choice(others, size=m, replace=False)))
            F_S, _  = chebyshev_energy_fast(grayL, ctxs, coords_all[S_idx], labels_all[S_idx], rng, eps, env)
            if not np.isfinite(F_S): continue
            Sj_idx = S_idx + [j]
            F_Sj, _ = chebyshev_energy_fast(grayL, ctxs, coords_all[Sj_idx], labels_all[Sj_idx], rng, eps, env)
            if not np.isfinite(F_Sj): continue
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


def greedy_minmax_subset_fast(grayL: np.ndarray,
                              ctxs: List[TTAContext],
                              coords_all: np.ndarray,
                              labels_all: np.ndarray,
                              rng: np.random.RandomState,
                              eps: float,
                              env: Dict[str,Dict[str,float]],
                              plateau: float,
                              show_pb: bool=False) -> List[int]:
    n = len(coords_all)
    remaining = list(range(n))
    selected = []

    F_curr, _ = chebyshev_energy_fast(grayL, ctxs, np.zeros((0,2),np.float32),
                                      np.zeros((0,),np.int32), rng, eps, env)
    if not np.isfinite(F_curr):
        best = None; bestF = None
        for j in remaining:
            F_j, _ = chebyshev_energy_fast(grayL, ctxs, coords_all[[j]], labels_all[[j]], rng, eps, env)
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
            F_new, _ = chebyshev_energy_fast(grayL, ctxs, coords_all[idx], labels_all[idx], rng, eps, env)
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


# ===================== Visualization (BASE SAM; unchanged) =====================

def run_sam_logits(image_rgb: np.ndarray,
                   predictor: SamPredictor,
                   coords: np.ndarray,
                   labels: np.ndarray,
                   multimask: bool=True) -> List[np.ndarray]:
    H, W = image_rgb.shape[:2]
    with torch.no_grad():
        _, _, lrs = predictor.predict(point_coords=coords,
                                      point_labels=labels,
                                      multimask_output=multimask,
                                      return_logits=True)
    return [upsample_logits_to_image(lrs[i], predictor, (H,W)) for i in range(len(lrs))]

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
    ap.add_argument("--out-dir", default="./runs/minmax_refiner_fast")
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

    # Repro
    np.random.seed(args.seed)
    rng = np.random.RandomState(args.seed)

    out_dir = Path(args.out_dir)
    perfile = out_dir/"perfile"; perfile.mkdir(parents=True, exist_ok=True)
    overlays_dir = out_dir/"overlays"; overlays_dir.mkdir(parents=True, exist_ok=True)

    # Load
    image = load_image_rgb(args.image)
    coords_all, labels_all = load_prompts_json(args.prompts)

    # Precompute LAB-L once
    grayL = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)[...,0]

    # Build SAM model & TTA contexts (embedding computed once per TTA)
    device = "cuda:1" if torch.cuda.is_available() else "cpu"
    sam = sam_model_registry[args.model_type](checkpoint=args.checkpoint).to(device)

    tta_list = tta_configs(args.tta)
    if args.progress:
        tqdm.write("Building TTA contexts (embedding once per view)...")
    ctxs = build_tta_contexts(sam, image, tta_list)

    # Base predictor for overlays (unchanged behavior)
    base_pred = SamPredictor(sam)
    base_pred.set_image(image)

    # Full set baseline terms (FAST path)
    C_full, U_full, R_full, E_full = evaluate_terms_fast(grayL, ctxs, coords_all, labels_all,
                                                         jitter_radius=args.jitter_radius, rng=rng)

    # Determine hard ε
    if args.eps_mode == "full":
        eps = float(C_full * args.eps_scale)
    else:
        n = len(coords_all)
        Cvals = []
        it = range(max(64, args.env_samples))
        if args.progress:
            it = tqdm(it, desc="ε quantile sampling", leave=False)
        for _ in it:
            m = np.random.randint(1, n+1)
            idx = np.sort(np.random.choice(np.arange(n), size=m, replace=False))
            C, _, _, _ = evaluate_terms_fast(grayL, ctxs, coords_all[idx], labels_all[idx],
                                             jitter_radius=args.jitter_radius, rng=rng)
            Cvals.append(C)
        q = max(0.0, min(1.0, args.eps_quantile))
        eps = float(np.quantile(np.asarray(Cvals, dtype=np.float64), q) * args.eps_scale)

    # Envelopes
    env = estimate_envelopes_fast(grayL, ctxs, coords_all, labels_all, rng,
                                  args.env_samples, eps, show_pb=args.progress)

    # Full-set Chebyshev energy
    F_full, br_full = chebyshev_energy_fast(grayL, ctxs, coords_all, labels_all, rng, eps, env)

    # MC sufficiency
    s, se = mc_sufficiency_fast(grayL, ctxs, coords_all, labels_all,
                                rng, args.mc_samples, args.trim_frac, eps, env,
                                show_pb=args.progress)

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

    # Balanced min–max subset (greedy)
    minmax_idx = greedy_minmax_subset_fast(grayL, ctxs, coords_all, labels_all,
                                           rng, eps, env, args.plateau, show_pb=args.progress)
    prompts_minmax = []
    t_counter = 1
    for i in sorted(minmax_idx):
        x, y = int(coords_all[i,0]); y2 = int(coords_all[i,1]); lbl = int(labels_all[i])
        prompts_minmax.append({"t": t_counter, "x": x, "y": y2, "label": lbl}); t_counter += 1
    (perfile/"balanced_minmax.json").write_text(json.dumps({"prompts": prompts_minmax}, indent=2))

    # Energy breakdowns
    coords_ref = coords_all[keep_idx] if len(keep_idx)>0 else np.zeros((0,2),dtype=np.float32)
    labels_ref = labels_all[keep_idx] if len(keep_idx)>0 else np.zeros((0,),dtype=np.int32)
    F_ref, br_ref = chebyshev_energy_fast(grayL, ctxs, coords_ref, labels_ref, rng, eps, env)

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

    # Overlays (BASE SAM, unchanged behavior)
    logits_full_base = best_logits_BASE(image, base_pred, coords_all, labels_all, args.multimask)
    logits_ref_base  = best_logits_BASE(image, base_pred, coords_ref, labels_ref, args.multimask)
    save_side_by_side_BASE(image,
                           coords_all, labels_all, coords_ref, labels_ref,
                           logits_full_base, logits_ref_base,
                           f"FULL (N={len(coords_all)})",
                           f"REFINED (N={len(coords_ref)})",
                           overlays_dir/"full_vs_refined_BASE.png")

    coords_min = coords_all[minmax_idx] if len(minmax_idx)>0 else np.zeros((0,2),dtype=np.float32)
    labels_min = labels_all[minmax_idx] if len(minmax_idx)>0 else np.zeros((0,),dtype=np.int32)
    logits_min_base  = best_logits_BASE(image, base_pred, coords_min, labels_min, args.multimask)
    save_side_by_side_BASE(image,
                           coords_all, labels_all, coords_min, labels_min,
                           logits_full_base, logits_min_base,
                           f"FULL (N={len(coords_all)})",
                           f"BALANCED MIN–MAX (N={len(coords_min)})",
                           overlays_dir/"full_vs_minmax_BASE.png")

    with open(out_dir/"summary.txt", "w") as f:
        f.write(
f"""GT-free, theory-first prompt refiner (hard click constraint + min–max energy) — FAST

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
