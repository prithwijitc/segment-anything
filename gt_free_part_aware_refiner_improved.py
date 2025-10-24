#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GT-FREE, PART-AWARE PROMPT REFINER — IMPROVED (CALIBRATED + ROBUST ENVELOPES + SWAPS)

What this script does (single image):
  1) Feasibility gate in the ID view (no jitter/TTA): only click-consistent subsets are considered.
  2) Per-image temperature calibration (for scoring only) using feasible subsets.
  3) TTA (id/hflip/±3°/scale) with proper inverse warping for both probs and embeddings.
  4) Part-aware local regions via SAM-embedding clustering + Random Walker on superpixels
     (positives grouped; negatives merged into one background basin with min area/diameter).
  5) Unified energy terms (global & local): Uncertainty = entropy + TTA disagreement;
     Regularity = TV + perimeter^2/area + light topology; Separation = embedding-space Fisher ratio.
  6) Robust self-normalization using quantile envelopes (lo=10%, hi=70%) per image and per part.
  7) Monte-Carlo sufficiency per click under an explicit, stratified context distribution D
     (subset-size strata + light coverage), with percentile bootstrap CIs and optional FDR.
  8) Refined subset selection under constraints: feasible seed -> greedy min–max -> pairwise swaps.
  9) Overlays from BASE SAM (uncalibrated), plus extensive diagnostics and logs.

Outputs (under --out-dir):
  perfile/
    refined.json                     # selected prompt subset (ordered)
    sufficiency.csv                  # per-click s_j, CI, tags, p-values
    groups.json                      # part groups & indices
    local_regions.npz                # boolean masks Ω_g per part
    envelopes_global.json            # global quantile envelopes + feasible frac
    envelopes_local.json             # per-part quantile envelopes + feasible frac
    energy_breakdown.csv             # full vs refined: raw & normalized terms, F
    calibration.json                 # T*, calib diagnostics (ECE, TTA agreement monotonicity)
    diagnostics.json                 # feasible fraction, envelope spread, settings, seeds
  overlays/
    full_vs_refined_BASE.png         # BASE SAM (no calibration) — full vs refined overlay
    parts_colormap.png               # visualization of local regions (Ω_g) overlay on image
  summary.txt                        # human-readable run summary

Notes:
- Requires: segment_anything (SAM), numpy, torch, opencv-python, scikit-image, scikit-learn, scipy, matplotlib, tqdm.
- We use percentile bootstrap CIs (fast, robust) instead of studentized for simplicity.
- Per-part temperature is optional and omitted in this reference implementation to keep it compact; add if needed.
"""

import argparse, json, csv, math, warnings, time
from pathlib import Path
from typing import List, Tuple, Optional, Dict

import numpy as np
import cv2
from PIL import Image

import torch
import torch.nn.functional as F

import matplotlib.pyplot as plt
from tqdm.auto import tqdm

# SciPy / scikit-image / scikit-learn
from scipy import sparse
from scipy.sparse import linalg as spla
from skimage.segmentation import slic
from sklearn.decomposition import PCA
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import pairwise_distances

# ---- SAM ----
try:
    from segment_anything import sam_model_registry, SamPredictor
except Exception as e:
    raise RuntimeError("Install SAM: https://github.com/facebookresearch/segment-anything") from e

# ===================== I/O & Basics =====================

def load_image_rgb(path: str) -> np.ndarray:
    return np.array(Image.open(path).convert("RGB"))


def load_prompts_json(path: str) -> Tuple[np.ndarray, np.ndarray]:
    with open(path, "r") as f:
        data = json.load(f)
    arr = sorted(data.get("prompts", []), key=lambda d: d.get("t", 0))
    coords = np.array([[int(d["x"]), int(d["y"])] for d in arr], dtype=np.float32)
    labels = np.array([int(d["label"]) for d in arr], dtype=np.int32)
    return coords, labels


def save_json(obj, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


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
    return float(a.std(ddof=1) / math.sqrt(len(a)))

# ===================== TTA contexts & Warps =====================

class TTAContext:
    def __init__(self, name: str, predictor: SamPredictor,
                 inv_kind: str, Minv: Optional[np.ndarray], W: int, H: int):
        self.name = name
        self.pred = predictor
        self.inv_kind = inv_kind  # "id" | "hflip" | "affine"
        self.Minv = Minv          # 2x3 float32 if affine
        self.W = W; self.H = H


def tta_configs(enable: bool):
    if not enable:
        return [("id", None)]
    return [("id", None), ("hflip", None), ("rot", 3), ("rot", -3), ("scale", 0.97), ("scale", 1.03)]


def build_tta_contexts(sam_model, image_rgb: np.ndarray, tta_list) -> List[TTAContext]:
    H, W = image_rgb.shape[:2]
    ctxs: List[TTAContext] = []
    for kind, param in tta_list:
        if kind == "id":
            img_t = image_rgb
            pred = SamPredictor(sam_model)
            pred.set_image(img_t)
            ctxs.append(TTAContext("id", pred, "id", None, W, H))
        elif kind == "hflip":
            img_t = np.ascontiguousarray(image_rgb[:, ::-1, :])
            pred = SamPredictor(sam_model); pred.set_image(img_t)
            ctxs.append(TTAContext("hflip", pred, "hflip", None, W, H))
        elif kind == "rot":
            angle = float(param)
            M = cv2.getRotationMatrix2D((W/2, H/2), angle, 1.0).astype(np.float32)
            img_t = cv2.warpAffine(image_rgb, M, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
            Minv = cv2.invertAffineTransform(M).astype(np.float32)
            pred = SamPredictor(sam_model); pred.set_image(img_t)
            ctxs.append(TTAContext(f"rot{angle:+.0f}", pred, "affine", Minv, W, H))
        elif kind == "scale":
            s = float(param)
            M = np.array([[s,0,(1-s)*W/2],[0,s,(1-s)*H/2]], dtype=np.float32)
            img_t = cv2.warpAffine(image_rgb, M, (W,H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
            Minv = cv2.invertAffineTransform(M).astype(np.float32)
            pred = SamPredictor(sam_model); pred.set_image(img_t)
            ctxs.append(TTAContext(f"scale{s:.2f}", pred, "affine", Minv, W, H))
        else:
            raise ValueError("unknown TTA kind")
    return ctxs


def warp_back_prob(p_t: np.ndarray, ctx: TTAContext) -> np.ndarray:
    if ctx.inv_kind == "id":
        return p_t
    if ctx.inv_kind == "hflip":
        return p_t[:, ::-1]
    return cv2.warpAffine(p_t, ctx.Minv, (ctx.W, ctx.H),
                          flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)


def warp_back_feat(feat_t: np.ndarray, ctx: TTAContext) -> np.ndarray:
    """Warp CxHxW feature map back to image space. Input is (C,Ht,Wt)."""
    if ctx.inv_kind == "id":
        return feat_t
    if ctx.inv_kind == "hflip":
        return feat_t[:, :, ::-1]
    # affine
    C, Ht, Wt = feat_t.shape
    out = []
    for c in range(C):
        out.append(cv2.warpAffine(feat_t[c], ctx.Minv, (ctx.W, ctx.H),
                                  flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT))
    return np.stack(out, 0)


def upsample_to_image(tchw: np.ndarray, H: int, W: int) -> np.ndarray:
    # tchw: (C,h,w) -> (C,H,W)
    t = torch.from_numpy(tchw).unsqueeze(0)
    up = F.interpolate(t, size=(H,W), mode="bilinear", align_corners=False)[0].detach().cpu().numpy()
    return up

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


def run_sam_logits_ctx(ctx: TTAContext,
                       coords_t: np.ndarray,
                       labels: np.ndarray,
                       multimask: bool=True) -> List[np.ndarray]:
    with torch.no_grad():
        _, _, lrs = ctx.pred.predict(point_coords=coords_t,
                                     point_labels=labels,
                                     multimask_output=multimask,
                                     return_logits=True)
    return [upsample_logits_to_image(lrs[i], ctx.pred, (ctx.H, ctx.W)) for i in range(len(lrs))]


def click_bce_at_points(logits: np.ndarray, coords: np.ndarray, labels: np.ndarray) -> float:
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

# ===================== Uncertainty, Regularity, Separation =====================

def pairwise_iou(masks: List[np.ndarray], region: Optional[np.ndarray]=None) -> float:
    if len(masks) < 2: return 1.0
    tot = 0.0; cnt = 0
    if region is None:
        region = np.ones_like(masks[0], dtype=bool)
    r = region.astype(bool)
    for i in range(len(masks)):
        for j in range(i+1, len(masks)):
            a = np.logical_and(masks[i].astype(bool), r)
            b = np.logical_and(masks[j].astype(bool), r)
            inter = float(np.logical_and(a,b).sum())
            uni = float(np.logical_or(a,b).sum())
            tot += (1.0 if uni==0 else inter/uni); cnt += 1
    return tot / max(1, cnt)


def tv_on_prob(p: np.ndarray, region: Optional[np.ndarray]=None) -> float:
    if region is not None:
        p = p * region.astype(np.float32)
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


def fisher_separation(phi_hw_c: np.ndarray, mask: np.ndarray, region: Optional[np.ndarray]=None) -> float:
    """Fisher-style inside/outside separation on SAM embeddings (lower is better after sign flip).
       We return an energy = 1 / (Fisher ratio + eps) so that lower is better.
    """
    if region is None:
        region = np.ones(mask.shape, dtype=bool)
    r = region.astype(bool)
    m = np.logical_and(mask.astype(bool), r)
    n = np.logical_and(~mask.astype(bool), r)
    if m.sum() < 10 or n.sum() < 10:
        return 1.0  # neutral if too small
    X_in = phi_hw_c[m]
    X_out = phi_hw_c[n]
    mu_in = X_in.mean(axis=0, keepdims=True)
    mu_out = X_out.mean(axis=0, keepdims=True)
    Sb = float(np.sum((mu_in - mu_out)**2))
    Sw = float(np.trace(np.cov(X_in, rowvar=False) + np.cov(X_out, rowvar=False))) + 1e-6
    fisher = Sb / Sw
    return float(1.0 / (fisher + 1e-6))

# ===================== Probabilities via TTA (with calibration T) =====================

def best_logits_ctx(ctx: TTAContext, coords_t: np.ndarray, labels: np.ndarray, multimask: bool) -> np.ndarray:
    ups = run_sam_logits_ctx(ctx, coords_t, labels, multimask)
    best = None; best_ce = None
    for z in ups:
        ce = click_bce_at_points(z, coords_t, labels)
        if best is None or ce < best_ce:
            best = z; best_ce = ce
    return best


def evaluate_terms_calibrated(grayL: np.ndarray,
                              phi_hw_c: np.ndarray,
                              ctxs: List[TTAContext],
                              coords: np.ndarray,
                              labels: np.ndarray,
                              multimask: bool,
                              jitter_radius: int,
                              T: float,
                              rng: np.random.RandomState,
                              region: Optional[np.ndarray]=None) -> Tuple[float,float,float]:
    """Return (U, R, S) computed with calibrated probs and TTA disagreement; region=None => global.
       U = mean entropy + (1 - mean TTA IoU)
       R = TV + perimeter^2/area + topo
       S = embedding Fisher separation energy (lower is better)
    """
    H, W = grayL.shape[:2]
    prob_maps = []; masks = []

    # Prepare coords for each TTA and run SAM
    for ctx in ctxs:
        # jitter in ID frame
        coords_j = coords.copy()
        for i in range(coords_j.shape[0]):
            coords_j[i,0], coords_j[i,1] = jitter_point(int(coords_j[i,0]), int(coords_j[i,1]),
                                                         jitter_radius, H, W)
        # map to ctx frame
        if ctx.name == "id":
            coords_t = coords_j
        elif ctx.name == "hflip":
            coords_t = coords_j.copy(); coords_t[:,0] = (ctx.W-1) - coords_t[:,0]
        else:
            M = cv2.invertAffineTransform(ctx.Minv).astype(np.float32)
            ones = np.ones((coords_j.shape[0],1), dtype=np.float32)
            xy1 = np.hstack([coords_j, ones])
            xy2 = (M @ xy1.T).T
            coords_t = xy2.astype(np.float32)

        logits = best_logits_ctx(ctx, coords_t, labels, multimask)
        p_t = sigmoid(logits.astype(np.float32) / max(1e-6, T))
        p_back = warp_back_prob(p_t, ctx)
        prob_maps.append(p_back)
        masks.append((p_back>=0.5).astype(np.uint8))

    p_mean = np.mean(np.stack(prob_maps,0), axis=0)
    m_mean = (p_mean>=0.5).astype(np.uint8)

    r = region
    # U: entropy + (1 - TTA IoU)
    eps = 1e-7
    if r is None:
        H_pred = float(np.mean(-(np.clip(p_mean,eps,1-eps)*np.log(np.clip(p_mean,eps,1-eps)) +
                                 np.clip(1-p_mean,eps,1-eps)*np.log(np.clip(1-p_mean,eps,1-eps)))))
        mean_iou = pairwise_iou(masks, None)
    else:
        rr = r.astype(bool)
        vals = p_mean[rr]
        H_pred = float(np.mean(-(np.clip(vals,eps,1-eps)*np.log(np.clip(vals,eps,1-eps)) +
                                 np.clip(1-vals,eps,1-eps)*np.log(np.clip(1-vals,eps,1-eps)))))
        mean_iou = pairwise_iou(masks, rr)
    U_term = H_pred + (1.0 - mean_iou)

    # R: TV + perimeter^2/area + topo (restricted if region provided)
    tv = tv_on_prob(p_mean, r)
    if r is None:
        m_loc = m_mean
    else:
        m_loc = ( (p_mean*(r.astype(np.float32)))>=0.5 ).astype(np.uint8)
    L, A = mask_perimeter_and_area(m_loc)
    comp = (L*L) / max(A, 1.0)
    topo = topo_penalty(m_loc)
    R_term = tv + 0.2*comp + 0.05*topo

    # S: Fisher separation on embeddings
    S_term = fisher_separation(phi_hw_c, m_mean, r)

    return U_term, R_term, S_term

# ===================== Part grouping & Local Regions (Random Walker) =====================

def get_image_embedding(ctx: TTAContext) -> torch.Tensor:
    if hasattr(ctx.pred, "features") and ctx.pred.features is not None:
        return ctx.pred.features
    device = next(ctx.pred.model.parameters()).device
    return ctx.pred.model.image_encoder(ctx.pred.input_image.to(device))


def build_aligned_embedding(ctxs: List[TTAContext], H: int, W: int, progress: bool) -> np.ndarray:
    feats = []
    iterator = ctxs
    if progress:
        iterator = tqdm(iterator, desc="Embeddings (TTA align)", leave=False)
    for ctx in iterator:
        emb = get_image_embedding(ctx)  # (1,C,h,w)
        emb_np = emb[0].detach().cpu().numpy()  # (C,h,w)
        up = upsample_to_image(emb_np, ctx.H, ctx.W)    # (C,H,W) in TTA frame
        up_w = warp_back_feat(up, ctx)                  # (C,H,W) in ID frame
        # L2 normalize per-pixel
        up_chlast = np.transpose(up_w, (1,2,0)).astype(np.float32)  # (H,W,C)
        nrm = np.linalg.norm(up_chlast, axis=2, keepdims=True) + 1e-9
        up_chlast = up_chlast / nrm
        feats.append(up_chlast)
    phi_hw_c = np.mean(np.stack(feats,0), axis=0).astype(np.float32)
    return phi_hw_c


def sample_feature_at_click(phi_hw_c: np.ndarray, x: float, y: float) -> np.ndarray:
    H, W, C = phi_hw_c.shape
    x = float(np.clip(x, 0, W-1)); y = float(np.clip(y, 0, H-1))
    x0, y0 = int(np.floor(x)), int(np.floor(y))
    x1, y1 = min(x0+1,W-1), min(y0+1,H-1)
    dx, dy = x-x0, y-y0
    v = ((1-dx)*(1-dy)*phi_hw_c[y0,x0] + dx*(1-dy)*phi_hw_c[y0,x1] +
         (1-dx)*dy*phi_hw_c[y1,x0] + dx*dy*phi_hw_c[y1,x1])
    n = np.linalg.norm(v) + 1e-9
    return (v / n).astype(np.float32)


def cluster_clicks(features_2d: np.ndarray, pos: np.ndarray, min_link_dist: float) -> np.ndarray:
    X = np.hstack([features_2d, 0.5*pos])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        clus = AgglomerativeClustering(n_clusters=None,
                                       distance_threshold=min_link_dist,
                                       linkage='ward')
        labels = clus.fit_predict(X)
    return labels


def compute_superpixels(image_rgb: np.ndarray, n_segments: int=1200, compactness: float=10.0) -> np.ndarray:
    sp = slic(image_rgb, n_segments=n_segments, compactness=compactness, start_label=0, channel_axis=-1)
    return sp.astype(np.int32)


def superpixel_stats(sp: np.ndarray, phi_hw_c: np.ndarray, gray_edge: np.ndarray):
    H, W = sp.shape
    ids = np.unique(sp); N = ids.max()+1
    C = phi_hw_c.shape[2]
    sum_feat = np.zeros((N, C), dtype=np.float64)
    count = np.zeros((N,), dtype=np.int64)
    sum_x = np.zeros((N,), dtype=np.float64)
    sum_y = np.zeros((N,), dtype=np.float64)

    for y in range(H):
        for x in range(W):
            i = sp[y,x]
            sum_feat[i] += phi_hw_c[y,x]
            sum_x[i] += x; sum_y[i] += y; count[i] += 1

    phi_sp = sum_feat / np.maximum(1, count[:,None])
    nrm = np.linalg.norm(phi_sp, axis=1, keepdims=True) + 1e-9
    phi_sp = (phi_sp / nrm).astype(np.float32)
    ctr_sp = np.stack([sum_x/np.maximum(1,count), sum_y/np.maximum(1,count)], axis=1).astype(np.float32)

    # adjacency & boundary strength from gradients
    boundary_strengths: Dict[Tuple[int,int], float] = {}
    for y in range(H-1):
        for x in range(W-1):
            a = sp[y,x]; b = sp[y, x+1]
            c = sp[y,x]; d = sp[y+1, x]
            if a != b:
                k = (min(a,b), max(a,b))
                boundary_strengths.setdefault(k, []).append(float(max(gray_edge[y,x], gray_edge[y, x+1])))
            if c != d:
                k = (min(c,d), max(c,d))
                boundary_strengths.setdefault(k, []).append(float(max(gray_edge[y,x], gray_edge[y+1, x])))
    for k, vals in list(boundary_strengths.items()):
        boundary_strengths[k] = float(np.mean(vals))

    return phi_sp, ctr_sp, boundary_strengths


def build_rw_weights(phi_sp: np.ndarray, ctr_sp: np.ndarray, boundary_strengths, k_sigma: int = 10):
    N, C = phi_sp.shape
    nbrs = {i: [] for i in range(N)}
    edges = []; feat_dists = []; spat_d2 = []; bvals = []

    # scales
    all_d = []; all_b = []
    for (i,j), b in boundary_strengths.items():
        all_d.append(np.linalg.norm(ctr_sp[i]-ctr_sp[j]))
        all_b.append(b)
    rho = np.median(all_d) + 1e-9
    bq = np.quantile(all_b, 0.75) if len(all_b)>0 else 1.0
    kappa = (np.log(2.0) / (bq + 1e-9)) if bq > 1e-9 else 0.0

    for (i,j), b in boundary_strengths.items():
        nbrs[i].append(j); nbrs[j].append(i)
        edges.append((i,j))
        df = math.sqrt(max(1e-9, 2.0*(1.0 - float(np.clip(np.dot(phi_sp[i], phi_sp[j]), -1.0, 1.0)))))
        feat_dists.append(df)
        spat_d2.append(float(np.sum((ctr_sp[i]-ctr_sp[j])**2)))
        bvals.append(b)

    sigma = np.zeros((N,), dtype=np.float32) + 1e-3
    neigh_df = {i: [] for i in range(N)}
    for (i,j), df in zip(edges, feat_dists):
        neigh_df[i].append(df); neigh_df[j].append(df)
    for i in range(N):
        if len(neigh_df[i])==0:
            sigma[i] = 1.0
        else:
            arr = np.sort(np.asarray(neigh_df[i], dtype=np.float32))
            kk = min(k_sigma-1, len(arr)-1)
            sigma[i] = max(1e-3, float(arr[kk]))

    rows = []; cols = []; vals = []
    for (i,j), df, d2, b in zip(edges, feat_dists, spat_d2, bvals):
        w = math.exp(- (df*df) / (sigma[i]*sigma[j] + 1e-9)) * math.exp(- d2 / (rho*rho + 1e-9)) * math.exp(- kappa * b)
        rows += [i,j]; cols += [j,i]; vals += [w,w]

    W = sparse.csr_matrix((vals, (rows, cols)), shape=(N,N))
    return W, nbrs


def random_walker_superpixels(W: sparse.csr_matrix, seeds_labels: np.ndarray, n_labels: int) -> np.ndarray:
    N = W.shape[0]
    d = np.array(W.sum(axis=1)).ravel()
    L = sparse.diags(d) - W

    seeds_idx = np.where(seeds_labels >= 0)[0]
    unlab_idx = np.where(seeds_labels < 0)[0]
    if len(unlab_idx) == 0:
        P = np.zeros((N, n_labels), dtype=np.float64)
        for i in range(N):
            P[i, int(seeds_labels[i])] = 1.0
        return P

    Luu = L[unlab_idx[:,None], unlab_idx]
    Lus = L[unlab_idx[:,None], seeds_idx]

    Ys = np.zeros((len(seeds_idx), n_labels), dtype=np.float64)
    for k, i in enumerate(seeds_idx):
        Ys[k, int(seeds_labels[i])] = 1.0

    Luu = Luu.tocsc()
    solver = spla.factorized(Luu)
    rhs = -Lus @ Ys
    P_u = np.zeros((len(unlab_idx), n_labels), dtype=np.float64)
    for ell in range(n_labels):
        P_u[:, ell] = solver(rhs[:, ell])

    P = np.zeros((N, n_labels), dtype=np.float64)
    P[unlab_idx] = P_u
    P[seeds_idx] = Ys
    P = np.clip(P, 0.0, None)
    s = P.sum(axis=1, keepdims=True) + 1e-12
    P = P / s
    return P


def build_local_parts(image_rgb: np.ndarray,
                      ctxs: List[TTAContext],
                      coords: np.ndarray,
                      labels: np.ndarray,
                      sp_segments: int,
                      pca_dim: int,
                      min_link_dist: float,
                      margin: float,
                      neg_min_area_frac: float,
                      progress: bool):
    H, W = image_rgb.shape[:2]
    # TTA-aligned embedding
    phi_hw_c = build_aligned_embedding(ctxs, H, W, progress)

    # Cluster positives; negatives -> one basin
    pos_idx = np.where(labels==1)[0]
    neg_idx = np.where(labels==0)[0]

    # Feature at clicks + light spatial
    click_feats = []
    norm_xy = []
    for (x,y) in coords:
        click_feats.append(sample_feature_at_click(phi_hw_c, x, y))
        norm_xy.append([x/W, y/H])
    click_feats = np.stack(click_feats,0)
    norm_xy = np.asarray(norm_xy, dtype=np.float32)

    if pca_dim>0 and click_feats.shape[1] > 1:
        pca = PCA(n_components=min(pca_dim, click_feats.shape[1]))
        click_pca = pca.fit_transform(click_feats)
    else:
        click_pca = click_feats

    groups_pos: List[List[int]] = []
    if len(pos_idx) > 0:
        Xp = click_pca[pos_idx]
        Xp_pos = norm_xy[pos_idx]
        if len(pos_idx) > 1:
            dists = pairwise_distances(np.hstack([Xp, 0.5*Xp_pos]))
            thr = np.median(dists) * 0.6
        else:
            thr = 1.0
        cl = cluster_clicks(Xp, Xp_pos, min_link_dist or thr)
        for g in np.unique(cl):
            members = pos_idx[np.where(cl==g)[0]].tolist()
            groups_pos.append(members)

    # negatives: one basin, keep all neg indices together
    groups_neg = [neg_idx.tolist()] if len(neg_idx)>0 else []

    # Superpixels & RW
    sp = compute_superpixels(image_rgb, n_segments=sp_segments, compactness=10.0)
    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
    gx = cv2.Sobel(gray, cv2.CV_32F, 1,0,ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0,1,ksize=3)
    grad = np.sqrt(gx*gx + gy*gy)

    phi_sp, ctr_sp, bmap = superpixel_stats(sp, phi_hw_c, grad)
    W_rw, _ = build_rw_weights(phi_sp, ctr_sp, bmap, k_sigma=10)

    # Seeds: one label per group name
    group_names = []
    group_clicks: Dict[str, List[int]] = {}
    seeds = np.full((phi_sp.shape[0],), -1, dtype=np.int32)

    def add_group(name: str, members: List[int]):
        g_lbl = len(group_names)
        group_names.append(name)
        group_clicks[name] = members
        for idx in members:
            x, y = coords[idx]
            s = int(sp[int(round(y)), int(round(x))])
            seeds[s] = g_lbl

    for gi, members in enumerate(groups_pos):
        add_group(f"pos_{gi+1}", members)
    if len(groups_neg)>0:
        add_group("neg_bg", groups_neg[0])

    Psp = random_walker_superpixels(W_rw, seeds, n_labels=len(group_names))  # (Nsp, G)

    # expand to pixels; define Ω_g by margin
    regions = {}
    for g_lbl, gname in enumerate(group_names):
        Ppix = np.zeros_like(sp, dtype=np.float32)
        # assign P at superpixel level
        for s_id in range(Psp.shape[0]):
            Ppix[sp == s_id] = Psp[s_id, g_lbl]
        margin_map = Ppix.copy()
        # other max
        others = np.zeros_like(Ppix)
        for h in range(Psp.shape[1]):
            if h == g_lbl: continue
            Ph = np.zeros_like(sp, dtype=np.float32)
            for s_id in range(Psp.shape[0]):
                Ph[sp == s_id] = Psp[s_id, h]
            others = np.maximum(others, Ph)
        margin_map = margin_map - others
        Omask = (margin_map >= margin).astype(np.uint8)
        # clean small artifacts
        Omask = cv2.morphologyEx(Omask, cv2.MORPH_OPEN, np.ones((3,3),np.uint8))
        Omask = cv2.morphologyEx(Omask, cv2.MORPH_CLOSE, np.ones((3,3),np.uint8))

        # negative basin: enforce min area fraction
        if gname == "neg_bg":
            if Omask.sum() < neg_min_area_frac * Omask.size:
                # expand by lowering margin
                Omask = (margin_map >= 0.0).astype(np.uint8)
        regions[gname] = Omask.astype(bool)

    # map each click to group name
    group_of_click = np.empty((len(coords),), dtype=object)
    for gname, ids in group_clicks.items():
        for idx in ids:
            group_of_click[idx] = gname

    groups_info = {"pos": groups_pos, "neg": groups_neg, "names": group_names, "click_map": group_of_click}
    return phi_hw_c, groups_info, regions, sp

# ===================== Feasibility (ID view only) =====================

def bce_feasibility_ID(base_pred: SamPredictor, coords: np.ndarray, labels: np.ndarray, multimask: bool) -> float:
    H, W = base_pred.original_size
    with torch.no_grad():
        _, _, lrs = base_pred.predict(point_coords=coords,
                                      point_labels=labels,
                                      multimask_output=multimask,
                                      return_logits=True)
    best = None; best_ce = None
    for i in range(len(lrs)):
        z = upsample_logits_to_image(lrs[i], base_pred, (H,W))
        ce = click_bce_at_points(z, coords, labels)
        if best is None or ce < best_ce:
            best = z; best_ce = ce
    return float(best_ce)

# ===================== Calibration (per image T for scoring only) =====================

def fit_temperature_per_image(base_pred: SamPredictor,
                              ctxs: List[TTAContext],
                              phi_hw_c: np.ndarray,
                              coords_all: np.ndarray,
                              labels_all: np.ndarray,
                              rng: np.random.RandomState,
                              multimask: bool,
                              eps_feas: float,
                              n_subsets: int = 64,
                              iters: int = 50) -> Tuple[float, Dict[str,float]]:
    """Fit scalar T>0 minimizing click BCE over random feasible subsets (ID view feasibility)."""
    N = len(coords_all)
    # collect feasible subsets
    feas_sets = []
    for _ in range(n_subsets*3):
        m = rng.randint(1, N+1)
        idx = np.sort(rng.choice(np.arange(N), size=m, replace=False))
        C = bce_feasibility_ID(base_pred, coords_all[idx], labels_all[idx], multimask)
        if C <= eps_feas:
            feas_sets.append(idx)
        if len(feas_sets) >= n_subsets:
            break
    if len(feas_sets) == 0:
        return 1.0, {"feasible_sets": 0}

    # SGD on scalar T (really, golden-search on logT for stability)
    logT = 0.0  # T=1
    lr = 0.2
    for _ in range(iters):
        # finite-diff gradient on logT
        def loss_at(Tval: float) -> float:
            Tval = max(1e-4, Tval)
            losses = []
            for idx in feas_sets:
                # evaluate click BCE in ID ctx using calibrated probs
                # we reuse base_pred (ID), but compute logits best-of multimask
                H, W = base_pred.original_size
                with torch.no_grad():
                    _, _, lrs = base_pred.predict(point_coords=coords_all[idx],
                                                  point_labels=labels_all[idx],
                                                  multimask_output=multimask,
                                                  return_logits=True)
                best = None; best_ce = None
                for i in range(len(lrs)):
                    z = upsample_logits_to_image(lrs[i], base_pred, (H,W))
                    p = sigmoid(z / Tval)
                    ce = 0.0; eps=1e-7
                    for (x,y), lab in zip(coords_all[idx], labels_all[idx]):
                        x = float(np.clip(x, 0, W-1)); y = float(np.clip(y, 0, H-1))
                        x0, y0 = int(np.floor(x)), int(np.floor(y))
                        x1, y1 = min(x0+1,W-1), min(y0+1,H-1)
                        dx, dy = x-x0, y-y0
                        pv = (1-dx)*(1-dy)*p[y0,x0] + dx*(1-dy)*p[y0,x1] + (1-dx)*dy*p[y1,x0] + dx*dy*p[y1,x1]
                        pv = min(max(float(pv), eps), 1-eps)
                        ce += -math.log(pv) if lab==1 else -math.log(1-pv)
                    ce /= max(1,len(idx))
                    if best is None or ce < best_ce:
                        best_ce = ce
                losses.append(best_ce)
            return float(np.mean(losses))
        f0 = loss_at(math.exp(logT))
        f1 = loss_at(math.exp(logT + 1e-2))
        g = (f1 - f0) / 1e-2
        logT -= lr * g
        logT = float(np.clip(logT, -2.0, 2.5))  # T in [~0.14, ~12]
        lr *= 0.95
    Tstar = float(math.exp(logT))

    # simple reliability diagnostic: ECE-like at clicks (few bins) on held-out subsets
    bins = np.linspace(0.0,1.0,6)
    tot, err = 0, 0
    with torch.no_grad():
        for idx in feas_sets[:max(1,len(feas_sets)//3)]:
            H, W = base_pred.original_size
            _, _, lrs = base_pred.predict(point_coords=coords_all[idx], point_labels=labels_all[idx],
                                          multimask_output=multimask, return_logits=True)
            best = None; best_ce = None; best_p = None
            for i in range(len(lrs)):
                z = upsample_logits_to_image(lrs[i], base_pred, (H,W))
                p = sigmoid(z / Tstar)
                ce = click_bce_at_points(z, coords_all[idx], labels_all[idx])
                if best is None or ce < best_ce:
                    best = z; best_ce = ce; best_p = p
            # bucket
            for (x,y), lab in zip(coords_all[idx], labels_all[idx]):
                x = float(np.clip(x, 0, W-1)); y = float(np.clip(y, 0, H-1))
                x0, y0 = int(np.floor(x)), int(np.floor(y))
                x1, y1 = min(x0+1,W-1), min(y0+1,H-1)
                dx, dy = x-x0, y-y0
                pv = (1-dx)*(1-dy)*best_p[y0,x0] + dx*(1-dy)*best_p[y0,x1] + (1-dx)*dy*best_p[y1,x0] + dx*dy*best_p[y1,x1]
                pv = float(np.clip(pv, 1e-7, 1-1e-7))
                tot += 1
                err += int((pv>=0.5) != (lab==1))
    ece_like = err / max(1, tot)

    return Tstar, {"feasible_sets": len(feas_sets), "ece_like": float(ece_like)}

# ===================== Envelopes (quantile) =====================

def normalize_term(val: float, lo: float, hi: float) -> float:
    den = max(1e-9, hi - lo)
    return float((val - lo) / den)


def estimate_envelopes(grayL, phi_hw_c, ctxs, coords_all, labels_all, multimask, jitter_radius, T, rng,
                        regions: Optional[Dict[str,np.ndarray]],
                        env_samples: int, eps_feas: float, base_pred: SamPredictor,
                        progress: bool):
    """If regions is None -> global. Else compute per-part envelopes in those masks."""
    def collect_T_values(region_mask: Optional[np.ndarray]):
        Uvals, Rvals, Svals = [], [], []
        feas = 0; tried = 0
        it = range(env_samples)
        if progress: it = tqdm(it, total=env_samples, leave=False, desc=("Env local" if region_mask is not None else "Env global"))
        N = len(coords_all)
        for _ in it:
            m = rng.randint(1, N+1)
            idx = np.sort(rng.choice(np.arange(N), size=m, replace=False))
            C = bce_feasibility_ID(base_pred, coords_all[idx], labels_all[idx], multimask)
            tried += 1
            if C > eps_feas:
                continue
            feas += 1
            U, R, S = evaluate_terms_calibrated(grayL, phi_hw_c, ctxs, coords_all[idx], labels_all[idx],
                                                multimask, jitter_radius, T, rng, region_mask)
            Uvals.append(U); Rvals.append(R); Svals.append(S)
        return Uvals, Rvals, Svals, feas/max(1,tried)

    if regions is None:
        Uvals, Rvals, Svals, frac = collect_T_values(None)
        def qstats(arr):
            if len(arr)==0: return {"lo":0.0, "hi":1.0}
            arr = np.asarray(arr, dtype=np.float64)
            return {"lo": float(np.quantile(arr,0.10)), "hi": float(np.quantile(arr,0.70))}
        return {"U": qstats(Uvals), "R": qstats(Rvals), "S": qstats(Svals), "feasible_frac": float(frac)}
    else:
        env_local = {}
        for gname, Omask in (tqdm(list(regions.items()), desc="Env parts", leave=False) if progress else regions.items()):
            Uvals, Rvals, Svals, frac = collect_T_values(Omask)
            def qstats(arr):
                if len(arr)==0: return {"lo":0.0, "hi":1.0}
                arr = np.asarray(arr, dtype=np.float64)
                return {"lo": float(np.quantile(arr,0.10)), "hi": float(np.quantile(arr,0.70))}
            env_local[gname] = {"U": qstats(Uvals), "R": qstats(Rvals), "S": qstats(Svals), "feasible_frac": float(frac)}
        return env_local

# ===================== Robust Objective (Chebyshev min–max) =====================

def chebyshev_objective(grayL, phi_hw_c, ctxs, coords, labels, multimask, jitter_radius, T, rng,
                        env_global, env_local, regions) -> Tuple[float, Dict[str,float], Dict[str, Dict[str,float]]]:
    U, R, S = evaluate_terms_calibrated(grayL, phi_hw_c, ctxs, coords, labels, multimask, jitter_radius, T, rng, None)
    Cid = None  # feasibility is checked outside; here we just report terms

    Ug = normalize_term(U, env_global["U"]["lo"], env_global["U"]["hi"]) 
    Rg = normalize_term(R, env_global["R"]["lo"], env_global["R"]["hi"]) 
    Sg = normalize_term(S, env_global["S"]["lo"], env_global["S"]["hi"]) 
    Fg = float(max(Ug, Rg, Sg))
    br_global = {"U":U, "R":R, "S":S, "U_t":Ug, "R_t":Rg, "S_t":Sg, "F":Fg}

    Fmax_local = 0.0
    br_locals = {}
    for gname, Omask in regions.items():
        Ul, Rl, Sl = evaluate_terms_calibrated(grayL, phi_hw_c, ctxs, coords, labels, multimask, jitter_radius, T, rng, Omask)
        U_t = normalize_term(Ul, env_local[gname]["U"]["lo"], env_local[gname]["U"]["hi"]) 
        R_t = normalize_term(Rl, env_local[gname]["R"]["lo"], env_local[gname]["R"]["hi"]) 
        S_t = normalize_term(Sl, env_local[gname]["S"]["lo"], env_local[gname]["S"]["hi"]) 
        F_loc = float(max(U_t, R_t, S_t))
        br_locals[gname] = {"U":Ul, "R":Rl, "S":Sl, "U_t":U_t, "R_t":R_t, "S_t":S_t, "F":F_loc}
        Fmax_local = max(Fmax_local, F_loc)

    F_total = max(Fg, Fmax_local)
    return F_total, br_global, br_locals

# ===================== Context distribution D (stratified) =====================

def sample_context_indices_D(rng, N: int, strata: List[float], coverage_pos_groups: List[List[int]],
                              coverage_neg: List[int], ensure_feasible_fn, max_tries=1000):
    # strata are fractions of N (e.g., [0.25,0.5,0.75])
    frac = float(rng.choice(strata))
    m = max(1, int(round(frac * N)))
    for _ in range(max_tries):
        idx = np.sort(rng.choice(np.arange(N), size=m, replace=False))
        # coverage: at least 1 pos per positive group if available
        ok = True
        for g in coverage_pos_groups:
            if len(g)==0: continue
            if len(set(idx.tolist()).intersection(set(g))) == 0:
                ok = False; break
        if not ok:
            continue
        if not ensure_feasible_fn(idx):
            continue
        return idx
    return None

# ===================== Greedy + Swaps selection =====================

def feasible_seed(base_pred: SamPredictor, coords_all, labels_all, multimask,
                  groups_pos: List[List[int]], groups_neg: List[List[int]],
                  rpos: int, rneg: int, eps_feas: float, progress: bool):
    N = len(coords_all)
    seed = set()

    # per positive group, add up to rpos best BCE-reducing clicks
    for g in groups_pos:
        needed = max(0, rpos - sum(1 for idx in g if idx in seed))
        if needed <= 0: continue
        cand = []
        for idx in g:
            S = sorted(list(seed | {idx}))
            C = bce_feasibility_ID(base_pred, coords_all[S], labels_all[S], multimask)
            cand.append((C, idx))
        cand.sort();
        for k in range(min(needed, len(cand))):
            seed.add(cand[k][1])

    # negatives: one basin (groups_neg has either [] or [all_neg])
    for g in groups_neg:
        needed = max(0, rneg - sum(1 for idx in g if idx in seed))
        if needed <= 0: continue
        cand = []
        for idx in g:
            S = sorted(list(seed | {idx}))
            C = bce_feasibility_ID(base_pred, coords_all[S], labels_all[S], multimask)
            cand.append((C, idx))
        cand.sort();
        for k in range(min(needed, len(cand))):
            seed.add(cand[k][1])

    # then add best clicks until global feasibility achieved
    if progress: tqdm.write("Seeding global feasibility...")
    tried = 0
    while True:
        S = sorted(list(seed))
        if len(S)==0:
            # pick best singleton
            best = None; bestC = None
            for j in range(N):
                Cj = bce_feasibility_ID(base_pred, coords_all[[j]], labels_all[[j]], multimask)
                if best is None or Cj < bestC:
                    best, bestC = j, Cj
            seed.add(best); continue
        C = bce_feasibility_ID(base_pred, coords_all[S], labels_all[S], multimask)
        if not np.isfinite(C) or C <= eps_feas:
            break
        best_idx = None; bestC = None
        for j in range(N):
            if j in seed: continue
            Sj = sorted(S + [j])
            Cj = bce_feasibility_ID(base_pred, coords_all[Sj], labels_all[Sj], multimask)
            if np.isfinite(Cj) and (bestC is None or Cj < bestC):
                best_idx, bestC = j, Cj
        if best_idx is None:
            break
        seed.add(best_idx)
        tried += 1
        if tried > N: break

    return sorted(list(seed))


def greedy_minmax_with_swaps(grayL, phi_hw_c, ctxs, base_pred,
                             coords_all, labels_all, multimask, jitter_radius, T, rng,
                             env_global, env_local, regions,
                             seed_idx: List[int], eps_feas: float,
                             plateau: float, progress: bool) -> List[int]:
    selected = list(seed_idx)
    remaining = [i for i in range(len(coords_all)) if i not in selected]

    def F_of(idx_list):
        Fv, _, _ = chebyshev_objective(grayL, phi_hw_c, ctxs, coords_all[idx_list], labels_all[idx_list],
                                       multimask, jitter_radius, T, rng, env_global, env_local, regions)
        return Fv

    F_curr = F_of(selected) if len(selected)>0 else float("inf")
    pbar = tqdm(total=len(remaining), desc="Greedy select", leave=False, disable=not progress)
    improved = True
    while improved and len(remaining) > 0:
        improved = False
        best_gain = 0.0; best_j = None; bestF = None
        for j in list(remaining):
            Sj = selected + [j]
            if bce_feasibility_ID(base_pred, coords_all[Sj], labels_all[Sj], multimask) > eps_feas:
                continue
            F_new = F_of(Sj)
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

    # one pass of pairwise swaps
    if progress: tqdm.write("Swap pass...")
    swapped = True
    while swapped:
        swapped = False
        for i_idx in range(len(selected)):
            i = selected[i_idx]
            for j in list(remaining):
                S_new = selected.copy(); S_new[i_idx] = j
                if bce_feasibility_ID(base_pred, coords_all[S_new], labels_all[S_new], multimask) > eps_feas:
                    continue
                F_new = F_of(S_new)
                if (F_curr - F_new) >= plateau:
                    remaining.remove(j)
                    remaining.append(i)
                    selected[i_idx] = j
                    F_curr = F_new
                    swapped = True
                    break
            if swapped:
                break
    return selected

# ===================== Sufficiency (MC + bootstrap + FDR) =====================

def bootstrap_ci(trimmed_vals: List[float], B: int=500, alpha: float=0.05) -> Tuple[float,float]:
    arr = np.asarray(trimmed_vals, dtype=np.float64)
    if len(arr) == 0:
        return 0.0, 0.0
    rng = np.random.RandomState(123)
    means = []
    n = len(arr)
    k = max(1, int(0.1*n))
    for _ in range(B):
        idx = rng.randint(0, n, size=n)
        s = np.sort(arr[idx])
        if 2*k < len(s):
            means.append(float(s[k:-k].mean()))
        else:
            means.append(float(s.mean()))
    low = float(np.quantile(means, alpha/2))
    high = float(np.quantile(means, 1-alpha/2))
    return low, high


def benjamini_hochberg(pvals: List[float], q: float=0.1) -> List[bool]:
    m = len(pvals)
    order = np.argsort(pvals)
    passed = [False]*m
    thresh = 0.0
    for k, idx in enumerate(order, start=1):
        if pvals[idx] <= (q * k / m):
            thresh = pvals[idx]
    for i in range(m):
        passed[i] = pvals[i] <= thresh and thresh>0
    return passed


def mc_sufficiency(grayL, phi_hw_c, ctxs, base_pred,
                   coords_all, labels_all, multimask, jitter_radius, T, rng,
                   env_global, env_local, regions,
                   strata: List[float], eps_feas: float,
                   K: int, trim_frac: float, progress: bool,
                   coverage_pos: List[List[int]], coverage_neg: List[int]):
    N = len(coords_all)
    scores = np.zeros(N, dtype=np.float64)
    ci_lo = np.zeros(N, dtype=np.float64)
    ci_hi = np.zeros(N, dtype=np.float64)
    p_useful = np.ones(N, dtype=np.float64)
    p_harmful = np.ones(N, dtype=np.float64)

    def feasible_fn(idx):
        return bce_feasibility_ID(base_pred, coords_all[idx], labels_all[idx], multimask) <= eps_feas

    prompt_iter = range(N)
    if progress:
        prompt_iter = tqdm(prompt_iter, total=N, desc="Sufficiency")

    for j in prompt_iter:
        others = [k for k in range(N) if k != j]
        deltas = []
        attempts = 0
        target = K
        while len(deltas) < target and attempts < K*100:
            attempts += 1
            idx = sample_context_indices_D(rng, len(others), strata, coverage_pos, coverage_neg,
                                           ensure_feasible_fn=lambda I: feasible_fn(np.asarray(others)[I]),
                                           max_tries=200)
            if idx is None:
                continue
            S_idx = np.asarray(others)[idx]
            F_S, _, _ = chebyshev_objective(grayL, phi_hw_c, ctxs, coords_all[S_idx], labels_all[S_idx],
                                            multimask, jitter_radius, T, rng, env_global, env_local, regions)
            Sj_idx = np.concatenate([S_idx, [j]])
            if not feasible_fn(Sj_idx):
                continue
            F_Sj, _, _ = chebyshev_objective(grayL, phi_hw_c, ctxs, coords_all[Sj_idx], labels_all[Sj_idx],
                                             multimask, jitter_radius, T, rng, env_global, env_local, regions)
            deltas.append(F_S - F_Sj)
        if len(deltas) == 0:
            scores[j]=0.0; ci_lo[j]=0.0; ci_hi[j]=0.0; p_useful[j]=1.0; p_harmful[j]=1.0
            continue
        # trimmed mean
        arr = np.sort(np.asarray(deltas, dtype=np.float64))
        k = int(math.floor(trim_frac*len(arr)))
        if 2*k < len(arr):
            mean_trim = float(arr[k:-k].mean())
        else:
            mean_trim = float(arr.mean())
        scores[j] = mean_trim
        # bootstrap CI on trimmed mean
        lo, hi = bootstrap_ci(arr, B=400, alpha=0.05)
        ci_lo[j], ci_hi[j] = lo, hi
        # one-sided p-values from bootstrap distribution around 0
        p_useful[j] = float(np.mean(arr <= 0.0))  # prob improvement <= 0 (smaller => more useful)
        p_harmful[j] = float(np.mean(arr >= 0.0)) # prob improvement >= 0 (smaller => more harmful)

    return scores, ci_lo, ci_hi, p_useful, p_harmful

# ===================== BASE Overlays (no calibration) =====================

def best_logits_BASE(image: np.ndarray, predictor: SamPredictor, coords: np.ndarray, labels: np.ndarray, multimask: bool) -> np.ndarray:
    H, W = image.shape[:2]
    with torch.no_grad():
        _, _, lrs = predictor.predict(point_coords=coords,
                                      point_labels=labels,
                                      multimask_output=multimask,
                                      return_logits=True)
    best = None; best_ce = None
    for z in lrs:
        up = upsample_logits_to_image(z, predictor, (H,W))
        ce = click_bce_at_points(up, coords, labels)
        if best is None or ce < best_ce:
            best = up; best_ce = ce
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


def save_parts_colormap(image: np.ndarray, regions: Dict[str,np.ndarray], out_path: Path):
    H, W = image.shape[:2]
    overlay = image.copy().astype(np.float32)
    # assign colors
    rng = np.random.RandomState(42)
    colors = {}
    for i, g in enumerate(regions.keys()):
        colors[g] = rng.rand(3)
    masksum = np.zeros((H,W,3), dtype=np.float32)
    for g, m in regions.items():
        c = colors[g]
        m3 = np.stack([m,m,m], axis=2).astype(np.float32)
        masksum += m3 * c
    masksum = np.clip(masksum, 0, 1)
    blend = (0.5*overlay/255.0 + 0.5*masksum)
    blend = np.clip(blend*255.0, 0, 255).astype(np.uint8)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(blend).save(out_path)

# ===================== Main =====================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True)
    ap.add_argument("--prompts", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--model-type", default="vit_h", choices=["vit_h","vit_l","vit_b"])
    ap.add_argument("--out-dir", default="./runs/gtfree_refiner_improved")
    ap.add_argument("--multimask", action="store_true")
    ap.add_argument("--tta", action="store_true")
    ap.add_argument("--jitter-radius", type=int, default=2)
    ap.add_argument("--env-samples", type=int, default=96)
    ap.add_argument("--mc-samples", type=int, default=64)
    ap.add_argument("--trim-frac", type=float, default=0.1)
    ap.add_argument("--plateau", type=float, default=1e-4)
    # Feasibility epsilon
    ap.add_argument("--eps-mode", choices=["quantile","scale_full"], default="quantile")
    ap.add_argument("--eps-quantile", type=float, default=0.5)
    ap.add_argument("--eps-scale", type=float, default=1.10)
    ap.add_argument("--min-feas-frac", type=float, default=0.3)
    # Parts / RW
    ap.add_argument("--sp-segments", type=int, default=1200)
    ap.add_argument("--pca-dim", type=int, default=16)
    ap.add_argument("--group-margin", type=float, default=0.10)
    ap.add_argument("--group-minlink", type=float, default=0.8)
    ap.add_argument("--neg-min-area-frac", type=float, default=0.02)
    ap.add_argument("--rpos", type=int, default=1)
    ap.add_argument("--rneg", type=int, default=1)
    # Context D
    ap.add_argument("--strata", type=str, default="0.25,0.5,0.75")
    # Calibration
    ap.add_argument("--calibrate", action="store_true")
    ap.add_argument("--calib-subsets", type=int, default=64)
    ap.add_argument("--calib-iters", type=int, default=50)
    # Bootstrap / FDR
    ap.add_argument("--bootstrap", type=int, default=400)
    ap.add_argument("--fdr-q", type=float, default=0.15)
    # Misc
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--progress", action="store_true")
    args = ap.parse_args()

    np.random.seed(args.seed)
    rng = np.random.RandomState(args.seed)

    out_dir = Path(args.out_dir)
    perfile = out_dir/"perfile"; perfile.mkdir(parents=True, exist_ok=True)
    overlays_dir = out_dir/"overlays"; overlays_dir.mkdir(parents=True, exist_ok=True)

    image = load_image_rgb(args.image)
    coords_all, labels_all = load_prompts_json(args.prompts)

    # SAM
    device = "cuda" if torch.cuda.is_available() else "cpu"
    sam = sam_model_registry[args.model_type](checkpoint=args.checkpoint).to(device)
    # TTA contexts (for scoring)
    tta_list = tta_configs(args.tta)
    if args.progress: tqdm.write("Building TTA contexts (with proper warps)...")
    ctxs = build_tta_contexts(sam, image, tta_list)
    # Base predictor (ID view) for feasibility + overlays
    base_pred = SamPredictor(sam); base_pred.set_image(image)

    # Build aligned embedding for parts
    if args.progress: tqdm.write("Building parts (embedding + RW)...")
    phi_hw_c, groups_info, regions, sp_map = build_local_parts(
        image_rgb=image, ctxs=ctxs, coords=coords_all, labels=labels_all,
        sp_segments=args.sp_segments, pca_dim=args.pca_dim, min_link_dist=args.group_minlink,
        margin=args.group_margin, neg_min_area_frac=args.neg_min_area_frac, progress=args.progress
    )

    # Feasibility epsilon (ID view only)
    if args.progress: tqdm.write("Estimating feasibility threshold ε ...")
    C_full = bce_feasibility_ID(base_pred, coords_all, labels_all, args.multimask)
    if args.eps_mode == "scale_full":
        eps = float(C_full * args.eps_scale)
    else:
        Cvals = []
        for _ in (tqdm(range(max(64, args.env_samples)), desc="ε subsets", leave=False) if args.progress else range(max(64, args.env_samples))):
            m = rng.randint(1, len(coords_all)+1)
            idx = np.sort(rng.choice(np.arange(len(coords_all)), size=m, replace=False))
            C = bce_feasibility_ID(base_pred, coords_all[idx], labels_all[idx], args.multimask)
            Cvals.append(C)
        q = max(0.0, min(1.0, args.eps_quantile))
        eps = float(np.quantile(np.asarray(Cvals, dtype=np.float64), q) * args.eps_scale)

    # Calibration T (for scoring only)
    Tstar = 1.0; calib_diag = {"used": False}
    if args.calibrate:
        if args.progress: tqdm.write("Fitting temperature T* (per image)...")
        Tstar, diag = fit_temperature_per_image(base_pred, ctxs, phi_hw_c, coords_all, labels_all, rng,
                                                args.multimask, eps, n_subsets=args.calib_subsets, iters=args.calib_iters)
        calib_diag = {"used": True, "T": float(Tstar), **diag}
    save_json(calib_diag, perfile/"calibration.json")

    # Envelopes (global & per-part)
    grayL = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)[...,0]
    if args.progress: tqdm.write("Estimating quantile envelopes (global & local)...")
    env_global = estimate_envelopes(grayL, phi_hw_c, ctxs, coords_all, labels_all, args.multimask,
                                    args.jitter_radius, Tstar, rng, None, args.env_samples, eps, base_pred, args.progress)
    env_local  = estimate_envelopes(grayL, phi_hw_c, ctxs, coords_all, labels_all, args.multimask,
                                    args.jitter_radius, Tstar, rng, regions, args.env_samples, eps, base_pred, args.progress)
    save_json(env_global, perfile/"envelopes_global.json")
    save_json(env_local,  perfile/"envelopes_local.json")

    # Feasible seed
    seed_idx = feasible_seed(base_pred, coords_all, labels_all, args.multimask,
                             groups_info["pos"], groups_info["neg"],
                             args.rpos, args.rneg, eps, args.progress)

    # Greedy + swaps selection under min–max
    minmax_idx = greedy_minmax_with_swaps(grayL, phi_hw_c, ctxs, base_pred,
                                          coords_all, labels_all, args.multimask, args.jitter_radius, Tstar, rng,
                                          env_global, env_local, regions, seed_idx, eps, args.plateau, args.progress)

    # Save refined.json
    prompts_refined = []
    t_counter = 1
    for i in sorted(minmax_idx):
        x, y = int(coords_all[i,0]), int(coords_all[i,1])
        lbl = int(labels_all[i])
        prompts_refined.append({"t": t_counter, "x": x, "y": y, "label": lbl}); t_counter += 1
    save_json({"prompts": prompts_refined}, perfile/"refined.json")

    # Energy breakdown: full vs refined
    F_full, br_full_g, br_full_L = chebyshev_objective(grayL, phi_hw_c, ctxs, coords_all, labels_all,
                                                       args.multimask, args.jitter_radius, Tstar, rng,
                                                       env_global, env_local, regions)
    coords_ref = coords_all[minmax_idx] if len(minmax_idx)>0 else np.zeros((0,2),dtype=np.float32)
    labels_ref = labels_all[minmax_idx] if len(minmax_idx)>0 else np.zeros((0,),dtype=np.int32)
    F_ref, br_ref_g, br_ref_L = chebyshev_objective(grayL, phi_hw_c, ctxs, coords_ref, labels_ref,
                                                    args.multimask, args.jitter_radius, Tstar, rng,
                                                    env_global, env_local, regions)
    with open(perfile/"energy_breakdown.csv", "w", newline="") as f:
        fields = ["set","N_prompts","U","R","S","U_t","R_t","S_t","F"]
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerow({"set":"FULL","N_prompts":len(coords_all), **br_full_g})
        w.writerow({"set":"REFINED","N_prompts":len(coords_ref), **br_ref_g})

    # Sufficiency (MC + bootstrap + FDR)
    strata = [float(s) for s in args.strata.split(",")]
    s, ci_lo, ci_hi, p_useful, p_harmful = mc_sufficiency(grayL, phi_hw_c, ctxs, base_pred,
                                                           coords_all, labels_all, args.multimask, args.jitter_radius, Tstar, rng,
                                                           env_global, env_local, regions, strata, eps,
                                                           args.mc_samples, args.trim_frac, args.progress,
                                                           groups_info["pos"], (groups_info["neg"][0] if len(groups_info["neg"])>0 else []))
    # FDR control (optional): mark discoveries
    pass_useful = benjamini_hochberg(p_useful.tolist(), q=args.fdr_q)
    pass_harmful = benjamini_hochberg(p_harmful.tolist(), q=args.fdr_q)

    tags = []
    for j in range(len(coords_all)):
        if ci_lo[j] > 0 and pass_useful[j]:
            tags.append("useful")
        elif ci_hi[j] < 0 and pass_harmful[j]:
            tags.append("harmful")
        else:
            tags.append("redundant")

    with open(perfile/"sufficiency.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=[
            "index","x","y","label","sufficiency","CI95_low","CI95_high","p_useful","p_harmful","FDR_useful","FDR_harmful","tag"
        ])
        w.writeheader()
        for i in range(len(coords_all)):
            w.writerow({
                "index": i+1,
                "x": int(coords_all[i,0]), "y": int(coords_all[i,1]),
                "label": int(labels_all[i]),
                "sufficiency": float(s[i]),
                "CI95_low": float(ci_lo[i]),
                "CI95_high": float(ci_hi[i]),
                "p_useful": float(p_useful[i]),
                "p_harmful": float(p_harmful[i]),
                "FDR_useful": bool(pass_useful[i]),
                "FDR_harmful": bool(pass_harmful[i]),
                "tag": tags[i]
            })

    # Save group metadata and regions
    save_json({"pos_groups": groups_info["pos"], "neg_groups": groups_info["neg"], "group_names": groups_info["names"]}, perfile/"groups.json")
    np.savez_compressed(perfile/"local_regions.npz", **{k: v.astype(np.uint8) for k,v in regions.items()})

    # Overlays (BASE SAM)
    base_pred_vis = SamPredictor(sam); base_pred_vis.set_image(image)
    logits_full_base = best_logits_BASE(image, base_pred_vis, coords_all, labels_all, args.multimask)
    logits_ref_base  = best_logits_BASE(image, base_pred_vis, coords_ref, labels_ref, args.multimask)
    save_side_by_side_BASE(image,
                           coords_all, labels_all, coords_ref, labels_ref,
                           logits_full_base, logits_ref_base,
                           f"FULL (N={len(coords_all)})",
                           f"REFINED (N={len(coords_ref)})",
                           overlays_dir/"full_vs_refined_BASE.png")

    # Parts visualization
    save_parts_colormap(image, regions, overlays_dir/"parts_colormap.png")

    # Diagnostics
    diag = {
        "epsilon_click": float(eps),
        "feasible_fraction_env_global": float(env_global.get("feasible_frac",0.0)),
        "feasible_fraction_env_local": {k: float(v.get("feasible_frac",0.0)) for k,v in env_local.items()},
        "settings": {
            "tta": args.tta,
            "jitter_radius": args.jitter_radius,
            "env_samples": args.env_samples,
            "mc_samples": args.mc_samples,
            "trim_frac": args.trim_frac,
            "plateau": args.plateau,
            "strata": args.strata,
            "seed": args.seed,
        }
    }
    save_json(diag, perfile/"diagnostics.json")

    # Summary
    with open(out_dir / "summary.txt", "w", encoding="utf-8") as f:
        f.write(f"""GT-free, part-aware refiner — IMPROVED

    Image: {args.image}
    Prompts: {args.prompts}
    SAM: {args.model_type}

    Feasibility ε (ID view): {eps:.6f} (mode={args.eps_mode}, q={args.eps_quantile}, scale={args.eps_scale})
    Calibration used: {calib_diag.get('used')}  T*: {calib_diag.get('T','1.0')}  ECE-like: {calib_diag.get('ece_like','n/a')}
    Env feasible frac (global): {env_global.get('feasible_frac',0.0):.3f}

    FULL (global):
    U={br_full_g.get('U', float('nan')):.6f}  R={br_full_g.get('R', float('nan')):.6f}  S={br_full_g.get('S', float('nan')):.6f}
    U_t={br_full_g.get('U_t', float('nan')):.3f} R_t={br_full_g.get('R_t', float('nan')):.3f} S_t={br_full_g.get('S_t', float('nan')):.3f}  F={br_full_g.get('F', float('nan')):.6f}

    REFINED (global):
    N={len(coords_ref)}  U={br_ref_g.get('U', float('nan')):.6f}  R={br_ref_g.get('R', float('nan')):.6f}  S={br_ref_g.get('S', float('nan')):.6f}
    U_t={br_ref_g.get('U_t', float('nan')):.3f} R_t={br_ref_g.get('R_t', float('nan')):.3f} S_t={br_ref_g.get('S_t', float('nan')):.3f}  F={br_ref_g.get('F', float('nan')):.6f}

    Files: perfile/refined.json, perfile/sufficiency.csv, perfile/groups.json, perfile/local_regions.npz,
        perfile/energy_breakdown.csv, perfile/envelopes_global.json, perfile/envelopes_local.json,
        perfile/calibration.json, perfile/diagnostics.json, overlays/full_vs_refined_BASE.png, overlays/parts_colormap.png
    """)



if __name__ == "__main__":
    main()
