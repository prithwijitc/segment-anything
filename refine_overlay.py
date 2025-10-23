#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GT-FREE, PART-AWARE PROMPT REFINER (FAST) — with Local Region Color Overlay
- SAM embeddings cached once per TTA context
- Random-Walker local regions on superpixel graph (in SAM feature space)
- Hard global & local BCE constraints (+ partition-matroid coverage)
- Robust Chebyshev objective (global + worst part)
- MC prompt sufficiency & constrained greedy min–max selection
- tqdm progress bars
- NEW: colored overlay of local regions (clusters) saved as overlays/local_regions_overlay.png

Outputs (in --out-dir):
  perfile/refined.json
  perfile/sufficiency.csv
  perfile/groups.json                # part groups & basic stats
  perfile/local_regions.npz          # Ω_g per group (boolean masks)
  perfile/envelopes_global.json
  perfile/envelopes_local.json
  perfile/energy_breakdown.csv       # full / refined, global+local
  overlays/full_vs_refined_BASE.png  # SAM base model, no calibration
  overlays/local_regions_overlay.png # NEW: colorized Ω_g overlay
  summary.txt
"""

import argparse, json, csv, math, warnings
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

def save_json(obj, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


# ===================== Misc utils =====================

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


# ===================== TTA contexts (cache embeddings) =====================

class TTAContext:
    def __init__(self, name: str, predictor: SamPredictor,
                 inv_kind: str, Minv: Optional[np.ndarray], W: int, H: int):
        self.name = name
        self.pred = predictor         # SamPredictor with set_image already called
        self.inv_kind = inv_kind      # "id" | "hflip" | "affine"
        self.Minv = Minv              # 2x3 float32, if inv_kind == "affine"
        self.W = W
        self.H = H

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

def get_image_embedding(ctx: TTAContext) -> torch.Tensor:
    # returns (1,C,h,w) on device
    if hasattr(ctx.pred, "features") and ctx.pred.features is not None:
        return ctx.pred.features
    # fallback: re-encode (shouldn't happen since set_image was used)
    device = next(ctx.pred.model.parameters()).device
    return ctx.pred.model.image_encoder(ctx.pred.input_image.to(device))

def upsample_to_image(tensor_chw: np.ndarray, H: int, W: int) -> np.ndarray:
    # tensor_chw: (C,h,w) numpy float32
    t = torch.from_numpy(tensor_chw).unsqueeze(0)  # (1,C,h,w)
    up = F.interpolate(t, size=(H,W), mode="bilinear", align_corners=False)[0].detach().cpu().numpy()
    return up  # (C,H,W)


# ===================== SAM decode helpers =====================

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


# ===================== Global & Local energy terms =====================

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

def entropy_inside_outside_L(grayL: np.ndarray,
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


# ===================== Probability via TTA (fast; cached encoders) =====================

def evaluate_terms_fast(grayL: np.ndarray,
                        ctxs: List[TTAContext],
                        coords: np.ndarray,
                        labels: np.ndarray,
                        multimask: bool,
                        jitter_radius: int,
                        rng: np.random.RandomState) -> Tuple[float,float,float,float, np.ndarray]:
    """
    Returns (C, U, R, E, p_mean)
    """
    H, W = grayL.shape[:2]
    prob_maps = []; masks = []; click_ce_list = []

    for ctx in ctxs:
        # jitter coords in original, map to ctx frame
        coords_j = coords.copy()
        for i in range(coords_j.shape[0]):
            coords_j[i,0], coords_j[i,1] = jitter_point(int(coords_j[i,0]), int(coords_j[i,1]),
                                                         jitter_radius, H, W)
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

        ups = run_sam_logits_ctx(ctx, coords_t, labels, multimask=multimask)
        best = None; best_ce = None
        for z in ups:
            ce = click_consistency_BCE_at_points(z, coords_t, labels)
            if best is None or ce < best_ce:
                best = z; best_ce = ce
        p_t = sigmoid(best.astype(np.float32))
        p_back = warp_back_prob(p_t, ctx)
        prob_maps.append(p_back)
        masks.append((p_back>=0.5).astype(np.uint8))
        click_ce_list.append(float(best_ce))

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

    E_info = entropy_inside_outside_L(grayL, m_mean, nbins=64)
    return C_term, U_term, R_term, E_info, p_mean


# ===================== Normalization =====================

def normalize_term(val: float, best: float, rand: float) -> float:
    den = max(1e-9, rand - best)
    return float(np.clip((val - best) / den, 0.0, 1.0))


# ===================== Part grouping (click-level) =====================

def sample_feature_at_click(phi_hw_c: np.ndarray, x: float, y: float) -> np.ndarray:
    # phi_hw_c: (H,W,C) float32 (L2-normalized per-pixel)
    H, W, C = phi_hw_c.shape
    x = float(np.clip(x, 0, W-1)); y = float(np.clip(y, 0, H-1))
    x0, y0 = int(np.floor(x)), int(np.floor(y))
    x1, y1 = min(x0+1,W-1), min(y0+1,H-1)
    dx, dy = x-x0, y-y0
    v = ((1-dx)*(1-dy)*phi_hw_c[y0,x0] + dx*(1-dy)*phi_hw_c[y0,x1] +
         (1-dx)*dy*phi_hw_c[y1,x0] + dx*dy*phi_hw_c[y1,x1])
    n = np.linalg.norm(v) + 1e-9
    return (v / n).astype(np.float32)

def cluster_clicks(features_2d: np.ndarray, pos: np.ndarray,
                   min_link_dist: float) -> np.ndarray:
    """
    Agglomerative clustering with a distance threshold set to min_link_dist.
    features_2d: (N, D)
    pos: (N, 2) in [0,1]^2 (to incorporate spatial)
    """
    X = np.hstack([features_2d, 0.5*pos])  # small spatial weight
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        clus = AgglomerativeClustering(n_clusters=None,
                                       distance_threshold=min_link_dist,
                                       linkage='ward')
        labels = clus.fit_predict(X)
    return labels


# ===================== Superpixels & RW graph =====================

def compute_superpixels(image_rgb: np.ndarray, n_segments: int=1200, compactness: float=10.0) -> np.ndarray:
    sp = slic(image_rgb, n_segments=n_segments, compactness=compactness, start_label=0, channel_axis=-1)
    return sp.astype(np.int32)

def superpixel_stats(sp: np.ndarray,
                     phi_hw_c: np.ndarray,
                     gray_edge: np.ndarray) -> Tuple[np.ndarray, np.ndarray, Dict[Tuple[int,int], float]]:
    """
    Returns:
      phi_sp: (Nsp, C) L2-normalized mean SAM feature per superpixel
      ctr_sp: (Nsp, 2) centroid (x,y)
      boundary_strengths: dict {(i,j): b_ij} averaged gradient along border
    """
    H, W = sp.shape
    ids = np.unique(sp)
    N = ids.max()+1
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

def build_rw_weights(phi_sp: np.ndarray,
                     ctr_sp: np.ndarray,
                     boundary_strengths: Dict[Tuple[int,int], float],
                     k_sigma: int = 10) -> Tuple[sparse.csr_matrix, Dict[int, List[int]]]:
    """
    Build symmetric weight matrix W for Random Walker over superpixels.
    w_ij = exp(- d_feat^2 / (sigma_i sigma_j)) * exp(- ||x_i-x_j||^2 / rho^2) * exp(-kappa * b_ij)
    """
    N, C = phi_sp.shape
    nbrs = {i: [] for i in range(N)}
    edges = []
    feat_dists = []
    spat_d2 = []
    bvals = []

    all_d = []
    all_b = []
    for (i,j), b in boundary_strengths.items():
        all_d.append(np.linalg.norm(ctr_sp[i]-ctr_sp[j]))
        all_b.append(b)
    rho = np.median(all_d) + 1e-9
    bq = np.quantile(all_b, 0.75) if len(all_b)>0 else 1.0
    kappa = (np.log(2.0) / (bq + 1e-9)) if bq > 1e-9 else 0.0

    for (i,j), b in boundary_strengths.items():
        nbrs[i].append(j); nbrs[j].append(i)
        edges.append((i,j))
        df = math.sqrt(max(1e-9, 2.0*(1.0 - float(np.clip(np.dot(phi_sp[i], phi_sp[j]), -1.0, 1.0)))))  # cosine-angle distance
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


def random_walker_superpixels(W: sparse.csr_matrix,
                              seeds_labels: np.ndarray,
                              n_labels: int) -> np.ndarray:
    """
    Random Walker on superpixel graph.
    seeds_labels: (N,) with -1 for unlabeled, else in [0, n_labels-1]
    Returns P: (N, n_labels) probabilities.
    """
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

    P_u = np.zeros((len(unlab_idx), n_labels), dtype=np.float64)
    Luu = Luu.tocsc()
    solver = spla.factorized(Luu)
    rhs = -Lus @ Ys  # (|U| x n_labels)
    for ell in range(n_labels):
        P_u[:, ell] = solver(rhs[:, ell])

    P = np.zeros((N, n_labels), dtype=np.float64)
    P[unlab_idx] = P_u
    P[seeds_idx] = Ys
    P = np.clip(P, 0.0, None)
    s = P.sum(axis=1, keepdims=True) + 1e-12
    P = P / s
    return P


# ===================== Local regions Ω_g (part-aware) =====================

def build_local_regions(image_rgb: np.ndarray,
                        ctxs: List[TTAContext],
                        coords: np.ndarray,
                        labels: np.ndarray,
                        sp_segments: int,
                        pca_dim: int,
                        min_link_dist: float,
                        margin: float,
                        progress: bool) -> Tuple[Dict[str, List[int]], Dict[str, np.ndarray], np.ndarray, np.ndarray]:
    """
    Returns:
      groups: {"pos":[...], "neg":[...], "names":[...], "click_map":array}
      regions: dict[group_name] -> boolean mask Ω_g (H,W)
      sp: superpixel map (H,W) int
      label_map: (H,W) int, -1 where no group claims the pixel; 0..G-1 are group indices in 'names'
    """
    H, W = image_rgb.shape[:2]
    # 1) Build a single SAM-embedding feature map (H,W,C) averaged across TTA contexts
    feats = []
    it = ctxs
    if progress: it = tqdm(it, desc="Embeddings (TTA avg)", leave=False)
    for ctx in it:
        emb = get_image_embedding(ctx)  # (1,C,h,w), torch
        emb_np = emb[0].detach().cpu().numpy()  # (C,h,w)
        up = upsample_to_image(emb_np, H, W)    # (C,H,W)
        up_chlast = np.transpose(up, (1,2,0)).astype(np.float32)  # (H,W,C)
        nrm = np.linalg.norm(up_chlast, axis=2, keepdims=True) + 1e-9
        up_chlast = up_chlast / nrm
        feats.append(up_chlast)
    phi_hw_c = np.mean(np.stack(feats,0), axis=0).astype(np.float32)  # (H,W,C)

    # 2) Click features for grouping (PCA to pca_dim + normalized xy)
    click_feats = []
    norm_xy = []
    for (x,y) in coords:
        f = sample_feature_at_click(phi_hw_c, x, y)  # (C,)
        click_feats.append(f)
        norm_xy.append([x/W, y/H])
    click_feats = np.stack(click_feats,0)  # (N,C)
    norm_xy = np.asarray(norm_xy, dtype=np.float32)

    if pca_dim>0:
        pca = PCA(n_components=min(pca_dim, click_feats.shape[1]))
        click_pca = pca.fit_transform(click_feats)
    else:
        click_pca = click_feats

    # 3) Separate pos / neg clicks, cluster each (Agglomerative with distance threshold)
    pos_idx = np.where(labels==1)[0]
    neg_idx = np.where(labels==0)[0]
    groups: Dict[str, List[List[int]]] = {"pos": [], "neg": []}

    if len(pos_idx)>0:
        Xp = click_pca[pos_idx]
        Xp_pos = norm_xy[pos_idx]
        if len(pos_idx) > 1:
            from sklearn.metrics import pairwise_distances
            dists = pairwise_distances(np.hstack([Xp, 0.5*Xp_pos]))
            thr = np.median(dists) * 0.6
        else:
            thr = 1.0
        cl = cluster_clicks(Xp, Xp_pos, min_link_dist or thr)
        for g in np.unique(cl):
            members = pos_idx[np.where(cl==g)[0]].tolist()
            groups["pos"].append(members)

    if len(neg_idx)>0:
        Xn = click_pca[neg_idx]
        Xn_pos = norm_xy[neg_idx]
        if len(neg_idx) > 1:
            from sklearn.metrics import pairwise_distances
            dists = pairwise_distances(np.hstack([Xn, 0.5*Xn_pos]))
            thr = np.median(dists) * 0.6
        else:
            thr = 1.0
        cl = cluster_clicks(Xn, Xn_pos, min_link_dist or thr)
        for g in np.unique(cl):
            members = neg_idx[np.where(cl==g)[0]].tolist()
            groups["neg"].append(members)

    # 4) Superpixels, gradient edges, superpixel features
    sp = slic(image_rgb, n_segments=sp_segments, compactness=10.0, start_label=0, channel_axis=-1).astype(np.int32)
    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
    gx = cv2.Sobel(gray, cv2.CV_32F, 1,0,ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0,1,ksize=3)
    grad = np.sqrt(gx*gx + gy*gy)

    phi_sp, ctr_sp, bmap = superpixel_stats(sp, phi_hw_c, grad)
    W_rw, nbrs = build_rw_weights(phi_sp, ctr_sp, bmap, k_sigma=10)

    # 5) Build seeds per superpixel node (each group gets a unique label)
    seeds = np.full((phi_sp.shape[0],), -1, dtype=np.int32)
    group_names = []
    group_clicks: Dict[str, List[int]] = {}

    def assign_seed_for_click(idx_click: int, gname: str):
        x, y = coords[idx_click]
        s = int(sp[int(round(y)), int(round(x))])
        group_clicks.setdefault(gname, []).append(idx_click)
        return s

    # Create names & collect clicks
    for gi, members in enumerate(groups["pos"]):
        gname = f"pos_{gi+1}"
        group_names.append(gname)
        for idx in members:
            assign_seed_for_click(idx, gname)
    for gi, members in enumerate(groups["neg"]):
        gname = f"neg_{gi+1}"
        group_names.append(gname)
        for idx in members:
            assign_seed_for_click(idx, gname)

    # Rebuild seeds with final label indices per group_name
    seeds = np.full((phi_sp.shape[0],), -1, dtype=np.int32)
    for g_lbl, gname in enumerate(group_names):
        for idx_click in group_clicks.get(gname, []):
            x, y = coords[idx_click]
            s = int(sp[int(round(y)), int(round(x))])
            seeds[s] = g_lbl

    # 6) Random Walker over superpixels
    if len(group_names) == 0:
        # No groups; return empty structures
        regions = {}
        label_map = -np.ones((H,W), dtype=np.int32)
        group_of_click = np.empty((len(coords),), dtype=object)
        return {"pos": [], "neg": [], "names": [], "click_map": group_of_click}, regions, sp, label_map

    P = random_walker_superpixels(W_rw, seeds, n_labels=len(group_names))  # (Nsp, G)
    # expand to pixels
    Nsp = P.shape[0]
    Ppix = np.zeros((len(group_names), H, W), dtype=np.float32)
    for s_id in range(Nsp):
        mask = (sp == s_id)
        if not mask.any(): continue
        for g in range(len(group_names)):
            Ppix[g][mask] = P[s_id, g]

    # 7) Define Ω_g via margin
    regions = {}
    for g, gname in enumerate(group_names):
        margin_map = Ppix[g] - np.max(np.delete(Ppix, g, axis=0), axis=0)
        Omask = (margin_map >= margin).astype(np.uint8)
        Omask = cv2.morphologyEx(Omask, cv2.MORPH_OPEN, np.ones((3,3),np.uint8))
        Omask = cv2.morphologyEx(Omask, cv2.MORPH_CLOSE, np.ones((3,3),np.uint8))
        regions[gname] = Omask.astype(bool)

    # map each click to its group name
    group_of_click = np.empty((len(coords),), dtype=object)
    for gname, ids in group_clicks.items():
        for idx in ids:
            group_of_click[idx] = gname

    # --- Build a single label map from the Ω_g regions (for overlay) ---
    G = len(group_names)
    if G == 0:
        label_map = -np.ones((H, W), dtype=np.int32)
    else:
        valid = np.stack([regions[gname].astype(bool) for gname in group_names], axis=0)  # (G,H,W)
        Pmasked = np.where(valid, Ppix, -1.0)  # (G,H,W); -1 where pixel not in Ω_g
        label_map = np.argmax(Pmasked, axis=0).astype(np.int32)
        maxv = np.max(Pmasked, axis=0)
        label_map[maxv < 0] = -1  # pixels claimed by no group remain -1

    groups_summary = {
        "pos_groups": [groups["pos"][i] for i in range(len(groups["pos"]))],
        "neg_groups": [groups["neg"][i] for i in range(len(groups["neg"]))],
        "group_names": group_names
    }
    return {"pos": groups["pos"], "neg": groups["neg"], "names": group_names, "click_map": group_of_click}, regions, sp, label_map


# ===================== Extra overlay for local regions =====================

def save_local_regions_overlay(image_rgb: np.ndarray,
                               label_map: np.ndarray,   # (H,W) int, -1 for “no group”
                               group_names: List[str],
                               out_path: Path,
                               alpha: float = 0.45):
    """
    Color each local region (Ω_g) with a distinct color and overlay on the image.
    label_map: per-pixel group index in [0, G-1], or -1 if no group claims that pixel.
    """
    H, W = image_rgb.shape[:2]
    overlay = image_rgb.copy()

    cmap = plt.get_cmap('tab20')
    colors = [(np.array(cmap(i % 20)[:3]) * 255).astype(np.uint8) for i in range(len(group_names))]

    for g in range(len(group_names)):
        mask = (label_map == g)
        if not mask.any():
            continue
        color = colors[g]
        overlay[mask] = (alpha * color + (1.0 - alpha) * overlay[mask]).astype(np.uint8)

    # Optional: thin black boundaries for region edges
    boundaries = np.zeros((H, W), dtype=np.uint8)
    for g in range(len(group_names)):
        m = (label_map == g).astype(np.uint8)
        if m.any():
            cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            cv2.drawContours(boundaries, cnts, -1, 255, 1)
    overlay[boundaries > 0] = (0.3 * np.array([0,0,0]) + 0.7 * overlay[boundaries > 0]).astype(np.uint8)

    # Save
    plt.figure(figsize=(6.5, 6.5 * H / max(1, W)))
    plt.imshow(overlay)
    plt.axis('off')
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=200)
    plt.close()


# ===================== Global & Local envelopes, objective =====================

def estimate_envelopes_global(grayL, ctxs, coords_all, labels_all, multimask, jitter_radius, rng,
                              env_samples: int, eps: float, progress: bool):
    n = len(coords_all)
    Uvals, Rvals, Evals = [], [], []
    feas = 0; tried = 0
    iterator = range(env_samples)
    if progress: iterator = tqdm(iterator, total=env_samples, desc="Envelopes (global)", leave=False)
    for _ in iterator:
        m = rng.randint(1, n+1)
        idx = np.sort(rng.choice(np.arange(n), size=m, replace=False))
        C, U, R, E, _ = evaluate_terms_fast(grayL, ctxs, coords_all[idx], labels_all[idx],
                                            multimask, jitter_radius, rng)
        tried += 1
        if C <= eps:
            feas += 1
            Uvals.append(U); Rvals.append(R); Evals.append(E)
    def stats(arr):
        if len(arr)==0: return {"best": 0.0, "rand": 1.0}
        arr = np.asarray(arr, dtype=np.float64)
        return {"best": float(np.min(arr)), "rand": float(np.median(arr))}
    return {"U": stats(Uvals), "R": stats(Rvals), "E": stats(Evals), "feasible_frac": float(feas/max(1,tried))}

def estimate_envelopes_local(grayL, ctxs, coords_all, labels_all, multimask, jitter_radius, rng,
                             regions: Dict[str, np.ndarray],
                             env_samples: int, eps: float, progress: bool):
    H, W = grayL.shape[:2]
    n = len(coords_all)
    env_local: Dict[str, Dict[str, Dict[str,float]]] = {}
    iterator = regions.items()
    if progress: iterator = tqdm(iterator, total=len(regions), desc="Envelopes (local)", leave=False)
    for gname, Omask in iterator:
        Uvals, Rvals, Evals = [], [], []
        feas = 0; tried = 0
        yx = Omask
        for _ in range(env_samples):
            m = rng.randint(1, n+1)
            idx = np.sort(rng.choice(np.arange(n), size=m, replace=False))
            C, U, R, E, p = evaluate_terms_fast(grayL, ctxs, coords_all[idx], labels_all[idx],
                                                multimask, jitter_radius, rng)
            tried += 1
            if C > eps:  # local feasibility follows global click constraint
                continue
            p_local = p * yx
            m_local = (p_local>=0.5).astype(np.uint8)
            epss = 1e-7
            H_pred = float(np.sum(-(np.clip(p_local,epss,1-epss)*np.log(np.clip(p_local,epss,1-epss)) +
                                   np.clip(1-p_local,epss,1-epss)*np.log(np.clip(1-p_local,epss,1-epss)))) / (yx.sum()+1e-9))
            U_loc = H_pred
            tv = tv_on_prob(p_local)
            L, A = mask_perimeter_and_area(m_local)
            comp = (L*L) / max(A, 1.0)
            topo = topo_penalty(m_local)
            R_loc = tv + 0.2*comp + 0.05*topo
            E_loc = entropy_inside_outside_L(grayL, m_local, nbins=64)

            feas += 1
            Uvals.append(U_loc); Rvals.append(R_loc); Evals.append(E_loc)

        def stats(arr):
            if len(arr)==0: return {"best": 0.0, "rand": 1.0}
            arr = np.asarray(arr, dtype=np.float64)
            return {"best": float(np.min(arr)), "rand": float(np.median(arr))}
        env_local[gname] = {
            "U": stats(Uvals), "R": stats(Rvals), "E": stats(Evals),
            "feasible_frac": float(feas/max(1,tried))
        }
    return env_local


def chebyshev_objective(grayL, ctxs, coords, labels, multimask, jitter_radius, rng,
                        eps, env_global, env_local, regions) -> Tuple[float, Dict[str,float], Dict[str, Dict[str,float]]]:
    C, U, R, E, p = evaluate_terms_fast(grayL, ctxs, coords, labels, multimask, jitter_radius, rng)
    if C > eps:
        return float("inf"), {"C":C, "U":U, "R":R, "E":E, "U_t":1.0, "R_t":1.0, "E_t":1.0, "F":float("inf")}, {}

    Ug = normalize_term(U, env_global["U"]["best"], env_global["U"]["rand"])
    Rg = normalize_term(R, env_global["R"]["best"], env_global["R"]["rand"])
    Eg = normalize_term(E, env_global["E"]["best"], env_global["E"]["rand"])
    Fg = float(max(Ug, Rg, Eg))
    br_global = {"C":C, "U":U, "R":R, "E":E, "U_t":Ug, "R_t":Rg, "E_t":Eg, "F":Fg}

    Fmax_local = 0.0
    br_locals = {}
    for gname, Omask in regions.items():
        yx = Omask
        p_local = p * yx
        m_local = (p_local>=0.5).astype(np.uint8)
        epss = 1e-7
        H_pred = float(np.sum(-(np.clip(p_local,epss,1-epss)*np.log(np.clip(p_local,epss,1-epss)) +
                               np.clip(1-p_local,epss,1-epss)*np.log(np.clip(1-p_local,epss,1-epss)))) / (yx.sum()+1e-9))
        U_loc = H_pred
        tv = tv_on_prob(p_local)
        L, A = mask_perimeter_and_area(m_local)
        comp = (L*L) / max(A, 1.0)
        topo = topo_penalty(m_local)
        R_loc = tv + 0.2*comp + 0.05*topo
        E_loc = entropy_inside_outside_L(grayL, m_local, nbins=64)

        envg = env_local[gname]
        U_t = normalize_term(U_loc, envg["U"]["best"], envg["U"]["rand"])
        R_t = normalize_term(R_loc, envg["R"]["best"], envg["R"]["rand"])
        E_t = normalize_term(E_loc, envg["E"]["best"], envg["E"]["rand"])
        F_loc = float(max(U_t, R_t, E_t))

        br_locals[gname] = {"U":U_loc, "R":R_loc, "E":E_loc, "U_t":U_t, "R_t":R_t, "E_t":E_t, "F":F_loc}
        if F_loc > Fmax_local:
            Fmax_local = F_loc

    F_total = max(Fg, Fmax_local)
    return F_total, br_global, br_locals


# ===================== Hard constraints & seeding =====================

def bce_global(grayL, ctxs, coords, labels, multimask, jitter_radius, rng) -> float:
    C, _, _, _, _ = evaluate_terms_fast(grayL, ctxs, coords, labels, multimask, jitter_radius, rng)
    return C

def feasible_seed_by_bce(grayL, ctxs, coords_all, labels_all, multimask, jitter_radius, rng,
                         groups: Dict[str, List[List[int]]],
                         group_names: List[str],
                         rpos: int, rneg: int,
                         eps_global: float,
                         progress: bool) -> List[int]:
    """
    Build a feasible seed that satisfies:
      - global BCE <= eps_global
      - coverage: at least rpos in each pos group, rneg in each neg group
    Strategy: select per-group anchors greedily by best BCE drop, then add prompts that reduce BCE until <= eps.
    """
    N = len(coords_all)
    pos_groups = groups["pos"]; neg_groups = groups["neg"]
    seed = set()

    if progress: tqdm.write("Seeding coverage per-group...")
    for g in pos_groups:
        needed = max(0, rpos - sum(1 for idx in g if idx in seed))
        if needed <= 0: continue
        candidates = []
        for idx in g:
            idxs = sorted(list(seed | {idx}))
            C = bce_global(grayL, ctxs, coords_all[idxs], labels_all[idxs],
                           multimask, jitter_radius, rng)
            candidates.append((C, idx))
        candidates.sort()
        for k in range(min(needed, len(candidates))):
            seed.add(candidates[k][1])

    for g in neg_groups:
        needed = max(0, rneg - sum(1 for idx in g if idx in seed))
        if needed <= 0: continue
        candidates = []
        for idx in g:
            idxs = sorted(list(seed | {idx}))
            C = bce_global(grayL, ctxs, coords_all[idxs], labels_all[idxs],
                           multimask, jitter_radius, rng)
            candidates.append((C, idx))
        candidates.sort()
        for k in range(min(needed, len(candidates))):
            seed.add(candidates[k][1])

    if progress: tqdm.write("Seeding global feasibility...")
    tried = 0
    while True:
        idxs = sorted(list(seed))
        C = bce_global(grayL, ctxs, coords_all[idxs], labels_all[idxs],
                       multimask, jitter_radius, rng)
        if len(idxs)==0 or not np.isfinite(C) or C <= eps_global: break
        best_idx = None; bestC = None
        for j in range(N):
            if j in seed: continue
            cand = sorted(idxs + [j])
            Cj = bce_global(grayL, ctxs, coords_all[cand], labels_all[cand],
                            multimask, jitter_radius, rng)
            if np.isfinite(Cj) and (bestC is None or Cj < bestC):
                bestC, best_idx = Cj, j
        if best_idx is None:
            break
        seed.add(best_idx); tried += 1
        if tried > N: break

    return sorted(list(seed))


# ===================== Greedy min–max subset under constraints =====================

def greedy_minmax_subset(grayL, ctxs, coords_all, labels_all, multimask, jitter_radius, rng,
                         eps, env_global, env_local, regions,
                         seed_idx: List[int],
                         plateau: float,
                         progress: bool) -> List[int]:
    """
    Start from feasible seed; iteratively add best prompt that lowers F; stop at plateau.
    """
    selected = list(seed_idx)
    remaining = [i for i in range(len(coords_all)) if i not in selected]

    F_curr, _, _ = chebyshev_objective(grayL, ctxs, coords_all[selected], labels_all[selected],
                                       multimask, jitter_radius, rng, eps, env_global, env_local, regions)
    if not np.isfinite(F_curr):
        best = None; bestF = None
        for j in remaining:
            F_j, _, _ = chebyshev_objective(grayL, ctxs, coords_all[[j]], labels_all[[j]],
                                            multimask, jitter_radius, rng, eps, env_global, env_local, regions)
            if np.isfinite(F_j) and (best is None or F_j < bestF):
                best, bestF = j, F_j
        if best is None: return []
        selected = [best]; remaining.remove(best); F_curr = bestF

    pbar = tqdm(total=len(remaining), desc="Min–max select", leave=False, disable=not progress)
    improved = True
    while improved and len(remaining) > 0:
        improved = False
        best_gain = 0.0; best_j = None; bestF = None
        for j in list(remaining):
            idx = selected + [j]
            F_new, _, _ = chebyshev_objective(grayL, ctxs, coords_all[idx], labels_all[idx],
                                              multimask, jitter_radius, rng, eps, env_global, env_local, regions)
            if not np.isfinite(F_new): continue
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


# ===================== MC sufficiency =====================

def mc_sufficiency(grayL, ctxs, coords_all, labels_all, multimask, jitter_radius, rng,
                   eps, env_global, env_local, regions,
                   mc_samples: int, trim_frac: float, progress: bool) -> Tuple[np.ndarray, np.ndarray]:
    n = len(coords_all)
    scores = np.zeros(n, dtype=np.float64)
    stderrs = np.zeros(n, dtype=np.float64)
    prompt_iter = range(n)
    if progress:
        prompt_iter = tqdm(prompt_iter, total=n, desc="Sufficiency (prompts)")
    for j in prompt_iter:
        others = [k for k in range(n) if k != j]
        deltas = []
        attempts = 0
        inner_pb = tqdm(total=mc_samples, desc=f"p{j+1}", leave=False, disable=not progress)
        while len(deltas) < mc_samples and attempts < mc_samples * 50:
            attempts += 1
            m = rng.randint(1, len(others)+1)
            S_idx = list(np.sort(rng.choice(others, size=m, replace=False)))
            F_S, _, _ = chebyshev_objective(grayL, ctxs, coords_all[S_idx], labels_all[S_idx],
                                            multimask, jitter_radius, rng, eps, env_global, env_local, regions)
            if not np.isfinite(F_S): continue
            Sj_idx = S_idx + [j]
            F_Sj, _, _ = chebyshev_objective(grayL, ctxs, coords_all[Sj_idx], labels_all[Sj_idx],
                                             multimask, jitter_radius, rng, eps, env_global, env_local, regions)
            if not np.isfinite(F_Sj): continue
            deltas.append(F_S - F_Sj)
            inner_pb.update(1)
        inner_pb.close()
        if len(deltas) == 0:
            scores[j] = 0.0; stderrs[j] = 0.0
        else:
            s = trimmed_mean(deltas, trim_frac)
            se = stderr(deltas)
            scores[j] = s; stderrs[j] = se
    return scores, stderrs


# ===================== Visualization (BASE SAM) =====================

def run_sam_logits(image_rgb: np.ndarray,
                   predictor: SamPredictor,
                   coords: np.ndarray,
                   labels: np.ndarray,
                   multimask: bool) -> List[np.ndarray]:
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
    ap.add_argument("--out-dir", default="./runs/partaware_refiner_fast")
    ap.add_argument("--multimask", action="store_true")
    ap.add_argument("--tta", action="store_true")
    ap.add_argument("--jitter-radius", type=int, default=2)
    ap.add_argument("--env-samples", type=int, default=96)
    ap.add_argument("--mc-samples", type=int, default=48)
    ap.add_argument("--trim-frac", type=float, default=0.1)
    ap.add_argument("--plateau", type=float, default=1e-4)
    ap.add_argument("--eps-mode", choices=["full","quantile"], default="quantile")
    ap.add_argument("--eps-quantile", type=float, default=0.5)
    ap.add_argument("--eps-scale", type=float, default=1.10)
    # Part-aware / RW params
    ap.add_argument("--sp-segments", type=int, default=1200)
    ap.add_argument("--pca-dim", type=int, default=16)
    ap.add_argument("--group-margin", type=float, default=0.10)
    ap.add_argument("--group-minlink", type=float, default=0.8)
    ap.add_argument("--rpos", type=int, default=1, help="min positives per pos-group")
    ap.add_argument("--rneg", type=int, default=1, help="min negatives per neg-group")
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

    # Build SAM model and TTA contexts
    device = "cuda" if torch.cuda.is_available() else "cpu"
    sam = sam_model_registry[args.model_type](checkpoint=args.checkpoint).to(device)
    tta_list = tta_configs(args.tta)
    if args.progress: tqdm.write("Building TTA contexts (embedding once per view)...")
    ctxs = build_tta_contexts(sam, image, tta_list)

    # Base predictor (for overlays only; unchanged behavior)
    base_pred = SamPredictor(sam); base_pred.set_image(image)

    # Precompute Lab-L once
    grayL = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)[...,0]

    # ===== Part grouping & local regions Ω_g via Random Walker
    if args.progress: tqdm.write("Part grouping + Random-Walker local regions...")
    groups_info, regions, sp_map, label_map = build_local_regions(
        image_rgb=image,
        ctxs=ctxs,
        coords=coords_all,
        labels=labels_all,
        sp_segments=args.sp_segments,
        pca_dim=args.pca_dim,
        min_link_dist=args.group_minlink,
        margin=args.group_margin,
        progress=args.progress
    )
    save_json({"pos_groups": groups_info["pos"], "neg_groups": groups_info["neg"], "group_names": groups_info["names"]},
              perfile/"groups.json")
    # Save Ω_g masks
    np.savez_compressed(perfile/"local_regions.npz", **{k: v.astype(np.uint8) for k,v in regions.items()})

    # NEW: Save colored overlay of local regions
    save_local_regions_overlay(
        image_rgb=image,
        label_map=label_map,
        group_names=groups_info["names"],
        out_path=overlays_dir / "local_regions_overlay.png",
        alpha=0.45
    )

    # ===== Determine ε (global)
    if args.progress: tqdm.write("Calibrating ε (global click BCE threshold)...")
    C_full, U_full, R_full, E_full, p_full = evaluate_terms_fast(
        grayL, ctxs, coords_all, labels_all, args.multimask, args.jitter_radius, rng
    )
    if args.eps_mode == "full":
        eps = float(C_full * args.eps_scale)
    else:
        Cvals = []
        it = range(max(64, args.env_samples))
        if args.progress: it = tqdm(it, desc="ε quantile sampling", leave=False)
        for _ in it:
            m = rng.randint(1, len(coords_all)+1)
            idx = np.sort(rng.choice(np.arange(len(coords_all)), size=m, replace=False))
            C, _, _, _, _ = evaluate_terms_fast(grayL, ctxs, coords_all[idx], labels_all[idx],
                                                args.multimask, args.jitter_radius, rng)
            Cvals.append(C)
        q = max(0.0, min(1.0, args.eps_quantile))
        eps = float(np.quantile(np.asarray(Cvals, dtype=np.float64), q) * args.eps_scale)

    # ===== Envelopes global & local
    env_global = estimate_envelopes_global(grayL, ctxs, coords_all, labels_all,
                                           args.multimask, args.jitter_radius, rng,
                                           args.env_samples, eps, args.progress)
    env_local  = estimate_envelopes_local(grayL, ctxs, coords_all, labels_all,
                                          args.multimask, args.jitter_radius, rng,
                                          regions, args.env_samples, eps, args.progress)
    save_json({"epsilon_click": eps, **env_global}, perfile/"envelopes_global.json")
    save_json(env_local, perfile/"envelopes_local.json")

    # ===== Feasible seeding (hard constraints + coverage per group)
    seed_idx = feasible_seed_by_bce(grayL, ctxs, coords_all, labels_all,
                                    args.multimask, args.jitter_radius, rng,
                                    groups={"pos": groups_info["pos"], "neg": groups_info["neg"]},
                                    group_names=groups_info["names"],
                                    rpos=args.rpos, rneg=args.rneg,
                                    eps_global=eps, progress=args.progress)

    # ===== Greedy min–max selection (global + worst local)
    minmax_idx = greedy_minmax_subset(grayL, ctxs, coords_all, labels_all,
                                      args.multimask, args.jitter_radius, rng,
                                      eps, env_global, env_local, regions,
                                      seed_idx=seed_idx, plateau=args.plateau, progress=args.progress)

    # Build refined.json from selected (min–max set)
    prompts_refined = []
    t_counter = 1
    for i in sorted(minmax_idx):
        x, y = int(coords_all[i,0]), int(coords_all[i,1])
        lbl = int(labels_all[i])
        prompts_refined.append({"t": t_counter, "x": x, "y": y, "label": lbl}); t_counter += 1
    save_json({"prompts": prompts_refined}, perfile/"refined.json")

    # ===== Chebyshev breakdowns for full vs refined
    F_full, br_full_g, br_full_L = chebyshev_objective(grayL, ctxs, coords_all, labels_all,
                                                       args.multimask, args.jitter_radius, rng,
                                                       eps, env_global, env_local, regions)
    coords_ref = coords_all[minmax_idx] if len(minmax_idx)>0 else np.zeros((0,2),dtype=np.float32)
    labels_ref = labels_all[minmax_idx] if len(minmax_idx)>0 else np.zeros((0,),dtype=np.int32)
    F_ref, br_ref_g, br_ref_L = chebyshev_objective(grayL, ctxs, coords_ref, labels_ref,
                                                    args.multimask, args.jitter_radius, rng,
                                                    eps, env_global, env_local, regions)

    # Energy breakdown CSV
    with open(perfile/"energy_breakdown.csv", "w", newline="") as f:
        fields = ["set","N_prompts","C","U","R","E","U_t","R_t","E_t","F"]
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerow({"set":"FULL","N_prompts":len(coords_all), **br_full_g})
        w.writerow({"set":"REFINED","N_prompts":len(coords_ref), **br_ref_g})

    # ===== MC sufficiency per prompt (under same objective & constraints)
    s, se = mc_sufficiency(grayL, ctxs, coords_all, labels_all,
                           args.multimask, args.jitter_radius, rng,
                           eps, env_global, env_local, regions,
                           args.mc_samples, args.trim_frac, args.progress)
    tags = []
    for v, sei in zip(s,se):
        ci = 1.96*sei
        if (v - ci) > 0: tags.append("useful")
        elif (v + ci) < 0: tags.append("harmful")
        else: tags.append("redundant")
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

    # ===== Overlays (BASE SAM; no calibration)
    logits_full_base = best_logits_BASE(image, base_pred, coords_all, labels_all, args.multimask)
    logits_ref_base  = best_logits_BASE(image, base_pred, coords_ref, labels_ref, args.multimask)
    save_side_by_side_BASE(image,
                           coords_all, labels_all, coords_ref, labels_ref,
                           logits_full_base, logits_ref_base,
                           f"FULL (N={len(coords_all)})",
                           f"REFINED (N={len(coords_ref)})",
                           overlays_dir/"full_vs_refined_BASE.png")

    # ===== Summary
    with open(out_dir/"summary.txt", "w") as f:
        f.write(
f"""GT-free, part-aware prompt refiner (hard click constraints + robust min–max) — FAST

Image: {args.image}
Prompts: {args.prompts}
SAM: {args.model_type}

Click ε (global): {eps:.6f}  (mode={args.eps_mode}, q={args.eps_quantile}, scale={args.eps_scale})
Global envelope feasible frac: {env_global.get("feasible_frac",0.0):.3f}

FULL (global):
  C={br_full_g['C']:.6f}  U={br_full_g['U']:.6f}  R={br_full_g['R']:.6f}  E={br_full_g['E']:.6f}
  U_t={br_full_g['U_t']:.3f} R_t={br_full_g['R_t']:.3f} E_t={br_full_g['E_t']:.3f}  F={br_full_g['F']:.6f}

REFINED (global):
  N={len(coords_ref)}  C={br_ref_g['C']:.6f} U={br_ref_g['U']:.6f} R={br_ref_g['R']:.6f} E={br_ref_g['E']:.6f}
  U_t={br_ref_g['U_t']:.3f} R_t={br_ref_g['R_t']:.3f} E_t={br_ref_g['E_t']:.3f}  F={br_ref_g['F']:.6f}

Files:
  perfile/refined.json
  perfile/sufficiency.csv
  perfile/groups.json
  perfile/local_regions.npz
  perfile/energy_breakdown.csv
  perfile/envelopes_global.json
  perfile/envelopes_local.json
  overlays/local_regions_overlay.png
  overlays/full_vs_refined_BASE.png
"""
        )

    print("\nDone. Results in:", out_dir)


if __name__ == "__main__":
    main()
