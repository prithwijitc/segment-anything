#!/usr/bin/env python3
"""
Generate prompt JSONs with strict FG/BG sampling.

- Positives (label=1) are sampled ONLY from GT foreground pixels.
- Negatives (label=0) are sampled ONLY from GT background pixels.
- Alternating sequence: +, -, +, -, ... starting with +.
- Exactly `pairs` positives and `pairs` negatives per JSON.

Usage:
  python make_prompts_strict.py \
    --image path/to/image.png \
    --gt path/to/mask.png \
    --pairs 6 \
    --num-files 220 \
    --calib-count 200 \
    --out-dir ./out_prompts_strict \
    --seed 0
"""

import argparse, csv, json, random
from pathlib import Path
from typing import List, Tuple, Dict, Optional

import numpy as np
import cv2
from PIL import Image


# ----------------- IO helpers -----------------

def load_image_rgb(path: str) -> np.ndarray:
    return np.array(Image.open(path).convert("RGB"))

def load_mask_binary(path: str, target_hw: Optional[Tuple[int,int]]=None) -> np.ndarray:
    m = np.array(Image.open(path))
    if m.ndim > 2:
        m = m[...,0]
    m = (m != 0).astype(np.uint8)
    if target_hw and m.shape[:2] != target_hw:
        H, W = target_hw
        m = cv2.resize(m, (W, H), interpolation=cv2.INTER_NEAREST)
        m = (m > 0).astype(np.uint8)
    return m


# ----------------- geometry & sampling utils -----------------

def distance_bands(gt: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Distance (L2) from FG to BG and BG to FG, per pixel (float32)."""
    fg = (gt > 0).astype(np.uint8)
    bg = (gt == 0).astype(np.uint8)
    dist_fg = cv2.distanceTransform(fg, cv2.DIST_L2, 3).astype(np.float32)  # 0 in BG
    dist_bg = cv2.distanceTransform(bg, cv2.DIST_L2, 3).astype(np.float32)  # 0 in FG
    return dist_fg, dist_bg

def mask_to_coords(mask: np.ndarray) -> np.ndarray:
    """Return (N,2) coords as (x,y) for mask==True pixels."""
    ys, xs = np.where(mask)
    if len(xs) == 0:
        return np.zeros((0,2), dtype=np.int32)
    return np.stack([xs, ys], axis=1).astype(np.int32)

def sample_with_min_dist(coords: np.ndarray, k: int, min_dist: float = 0.0, max_tries: int = 8000) -> List[Tuple[int,int]]:
    """Greedy Poisson-disk-like sampling on discrete coords."""
    if len(coords) == 0 or k <= 0:
        return []
    chosen: List[Tuple[int,int]] = []
    tries = 0
    while len(chosen) < k and tries < max_tries:
        i = np.random.randint(0, len(coords))
        cand = (int(coords[i,0]), int(coords[i,1]))
        ok = True
        if min_dist > 0 and chosen:
            for (x,y) in chosen:
                dx, dy = cand[0]-x, cand[1]-y
                if (dx*dx + dy*dy) < (min_dist*min_dist):
                    ok = False; break
        if ok and (cand not in chosen):
            chosen.append(cand)
        tries += 1
    # fallback uniform fill if needed
    while len(chosen) < k and len(coords) > 0:
        i = np.random.randint(0, len(coords))
        cand = (int(coords[i,0]), int(coords[i,1]))
        if cand not in chosen:
            chosen.append(cand)
    return chosen


# ----------------- policies (ALL enforce FG for positives, BG for negatives) -----------------

def policy_uniform(fg_mask: np.ndarray, bg_mask: np.ndarray, pairs: int):
    pos_coords = mask_to_coords(fg_mask)
    neg_coords = mask_to_coords(bg_mask)
    pos = sample_with_min_dist(pos_coords, pairs, 0.0)
    neg = sample_with_min_dist(neg_coords, pairs, 0.0)
    return pos, neg, "uniform"

def policy_boundary_band(fg_mask: np.ndarray, bg_mask: np.ndarray,
                         dist_fg: np.ndarray, dist_bg: np.ndarray,
                         pairs: int, in_band=(2,12), out_band=(2,12)):
    pos_mask = (fg_mask > 0) & (dist_fg >= in_band[0]) & (dist_fg <= in_band[1])
    neg_mask = (bg_mask > 0) & (dist_bg >= out_band[0]) & (dist_bg <= out_band[1])
    if not pos_mask.any(): pos_mask = (fg_mask > 0)
    if not neg_mask.any(): neg_mask = (bg_mask > 0)
    pos = sample_with_min_dist(mask_to_coords(pos_mask), pairs, 4.0)
    neg = sample_with_min_dist(mask_to_coords(neg_mask), pairs, 4.0)
    return pos, neg, "boundary_band"

def policy_spread_poisson(fg_mask: np.ndarray, bg_mask: np.ndarray, pairs: int):
    pos = sample_with_min_dist(mask_to_coords(fg_mask > 0), pairs, 12.0)
    neg = sample_with_min_dist(mask_to_coords(bg_mask > 0), pairs, 12.0)
    return pos, neg, "spread_poisson"

def policy_near_both(fg_mask: np.ndarray, bg_mask: np.ndarray,
                     dist_fg: np.ndarray, dist_bg: np.ndarray, pairs: int):
    pos_mask = (fg_mask > 0) & (dist_fg >= 1.0) & (dist_fg <= 5.0)
    neg_mask = (bg_mask > 0) & (dist_bg >= 1.0) & (dist_bg <= 5.0)
    if not pos_mask.any(): pos_mask = (fg_mask > 0)
    if not neg_mask.any(): neg_mask = (bg_mask > 0)
    pos = sample_with_min_dist(mask_to_coords(pos_mask), pairs, 3.0)
    neg = sample_with_min_dist(mask_to_coords(neg_mask), pairs, 3.0)
    return pos, neg, "near_both"

def policy_deep_pos_near_neg(fg_mask: np.ndarray, bg_mask: np.ndarray,
                             dist_fg: np.ndarray, dist_bg: np.ndarray, pairs: int):
    pos_mask = (fg_mask > 0) & (dist_fg >= 15.0)
    neg_mask = (bg_mask > 0) & (dist_bg >= 2.0) & (dist_bg <= 6.0)
    if not pos_mask.any(): pos_mask = (fg_mask > 0)
    if not neg_mask.any(): neg_mask = (bg_mask > 0)
    pos = sample_with_min_dist(mask_to_coords(pos_mask), pairs, 6.0)
    neg = sample_with_min_dist(mask_to_coords(neg_mask), pairs, 3.0)
    return pos, neg, "deep_pos_near_neg"

def policy_grid_like(fg_mask: np.ndarray, bg_mask: np.ndarray, pairs: int, step: int = 64):
    H, W = fg_mask.shape
    xs = np.arange(step//2, W, max(1, step))
    ys = np.arange(step//2, H, max(1, step))
    grid = np.array([(x,y) for y in ys for x in xs], dtype=np.int32)
    fg_coords = mask_to_coords(fg_mask > 0)
    bg_coords = mask_to_coords(bg_mask > 0)
    def snap(grid_pts, target_coords, k, min_d):
        if len(target_coords) == 0: return []
        out = []
        for c in grid_pts:
            d2 = (target_coords[:,0]-c[0])**2 + (target_coords[:,1]-c[1])**2
            j = int(np.argmin(d2))
            out.append((int(target_coords[j,0]), int(target_coords[j,1])))
        # unique + spacing
        uniq = []
        for pt in out:
            if pt not in uniq:
                uniq.append(pt)
        if len(uniq) >= k:
            np.random.shuffle(uniq)
            sel = uniq[:k]
        else:
            filler = sample_with_min_dist(target_coords, k - len(uniq), min_d)
            sel = uniq + filler
        return sel
    pos = snap(grid, fg_coords, pairs, 6.0)
    neg = snap(grid[::-1], bg_coords, pairs, 6.0)
    return pos, neg, "grid_like"

def policy_cluster_pos(fg_mask: np.ndarray, bg_mask: np.ndarray, pairs: int):
    # Gaussian cluster inside FG; negatives uniformly in BG
    coords_fg = mask_to_coords(fg_mask > 0)
    coords_bg = mask_to_coords(bg_mask > 0)
    pos = gaussian_cluster(coords_fg, pairs, sigma=20.0)
    neg = sample_with_min_dist(coords_bg, pairs, 2.0)
    return pos, neg, "cluster_pos"

def policy_cluster_neg(fg_mask: np.ndarray, bg_mask: np.ndarray, pairs: int):
    coords_fg = mask_to_coords(fg_mask > 0)
    coords_bg = mask_to_coords(bg_mask > 0)
    pos = sample_with_min_dist(coords_fg, pairs, 2.0)
    neg = gaussian_cluster(coords_bg, pairs, sigma=20.0)
    return pos, neg, "cluster_neg"

def gaussian_cluster(coords: np.ndarray, k: int, sigma: float = 15.0) -> List[Tuple[int,int]]:
    if len(coords) == 0: return []
    seed = coords[np.random.randint(0, len(coords))]
    sx, sy = int(seed[0]), int(seed[1])
    out: List[Tuple[int,int]] = []
    tries = 0
    while len(out) < k and tries < 5000:
        dx = int(np.round(np.random.normal(0, sigma)))
        dy = int(np.round(np.random.normal(0, sigma)))
        cand = (sx+dx, sy+dy)
        # snap to nearest valid coord
        d2 = (coords[:,0]-cand[0])**2 + (coords[:,1]-cand[1])**2
        j = int(np.argmin(d2))
        nearest = (int(coords[j,0]), int(coords[j,1]))
        if nearest not in out:
            out.append(nearest)
        tries += 1
    while len(out) < k and len(coords) > 0:
        j = np.random.randint(0, len(coords))
        c = (int(coords[j,0]), int(coords[j,1]))
        if c not in out: out.append(c)
    return out

def policy_edge_heavy_neg(fg_mask: np.ndarray, bg_mask: np.ndarray, dist_bg: np.ndarray, pairs: int):
    pos = sample_with_min_dist(mask_to_coords(fg_mask > 0), pairs, 3.0)
    coords = mask_to_coords(bg_mask > 0)
    if len(coords) == 0:
        neg = []
    else:
        d = dist_bg[coords[:,1], coords[:,0]].astype(np.float32)
        w = 1.0 / (d + 1.0); w = w / (w.sum() + 1e-8)
        idxs = np.random.choice(np.arange(len(coords)), size=min(pairs, len(coords)), replace=False, p=w)
        neg = [(int(coords[i,0]), int(coords[i,1])) for i in idxs]
        if len(neg) < pairs:
            neg += sample_with_min_dist(coords, pairs-len(neg), 0.0)
    return pos, neg, "edge_heavy_neg"

POLICIES = [
    ("uniform",            policy_uniform),
    ("boundary_band",      policy_boundary_band),
    ("spread_poisson",     policy_spread_poisson),
    ("near_both",          policy_near_both),
    ("deep_pos_near_neg",  policy_deep_pos_near_neg),
    ("grid_like",          policy_grid_like),
    ("cluster_pos",        policy_cluster_pos),
    ("cluster_neg",        policy_cluster_neg),
    ("edge_heavy_neg",     policy_edge_heavy_neg),
]


# ----------------- strict validators -----------------

def enforce_strict_counts(pos: List[Tuple[int,int]], neg: List[Tuple[int,int]],
                          fg_mask: np.ndarray, bg_mask: np.ndarray,
                          pairs: int) -> Tuple[List[Tuple[int,int]], List[Tuple[int,int]]]:
    """
    Ensure:
      - all pos ∈ FG, all neg ∈ BG
      - exactly `pairs` each
      - no duplicates within pos or within neg
    """
    # filter strictly by mask
    pos = [(x,y) for (x,y) in pos if fg_mask[y, x] > 0]
    neg = [(x,y) for (x,y) in neg if bg_mask[y, x] > 0]
    # deduplicate
    pos = list(dict.fromkeys(pos))
    neg = list(dict.fromkeys(neg))

    # refill if short using uniform valid coords
    pos_coords = mask_to_coords(fg_mask > 0)
    neg_coords = mask_to_coords(bg_mask > 0)

    def refill(cur, coords):
        need = pairs - len(cur)
        if need <= 0:
            return cur[:pairs]
        extras = sample_with_min_dist(coords, need, 0.0)
        # avoid duplicates
        extras = [c for c in extras if c not in cur]
        cur = cur + extras
        if len(cur) < pairs:
            raise RuntimeError("Not enough valid pixels to sample required pairs.")
        return cur[:pairs]

    pos = refill(pos, pos_coords)
    neg = refill(neg, neg_coords)

    # final assertion
    assert len(pos) == pairs and len(neg) == pairs
    assert all(fg_mask[y,x] > 0 for x,y in pos)
    assert all(bg_mask[y,x] > 0 for x,y in neg)
    return pos, neg


# ----------------- JSON builder -----------------

def build_one_json(pos_pts: List[Tuple[int,int]], neg_pts: List[Tuple[int,int]]) -> Dict:
    """Interleave + and − starting with +, producing t=1..2k."""
    k = min(len(pos_pts), len(neg_pts))
    prompts = []
    t = 1
    for i in range(k):
        x,y = pos_pts[i]; prompts.append({"t": t, "x": int(x), "y": int(y), "label": 1}); t += 1
        x,y = neg_pts[i]; prompts.append({"t": t, "x": int(x), "y": int(y), "label": 0}); t += 1
    return {"prompts": prompts}


# ----------------- main -----------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True)
    ap.add_argument("--gt", required=True, help="PNG mask; nonzero=FG; SAME SIZE as image")
    ap.add_argument("--pairs", type=int, default=6, help="Number of (+,−) pairs per JSON (total prompts = 2*pairs)")
    ap.add_argument("--num-files", type=int, default=220, help="Total JSON files to generate")
    ap.add_argument("--calib-count", type=int, default=200, help="#files for calibration; remainder go to eval")
    ap.add_argument("--out-dir", default="./out_prompts_strict")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    random.seed(args.seed); np.random.seed(args.seed)

    out_dir = Path(args.out_dir)
    (out_dir/"calib").mkdir(parents=True, exist_ok=True)
    (out_dir/"eval").mkdir(parents=True, exist_ok=True)

    # Load and sanity check
    img = load_image_rgb(args.image)
    H, W = img.shape[:2]
    gt  = load_mask_binary(args.gt, target_hw=(H, W))
    if gt.shape != (H, W):
        raise ValueError("Image and GT must have the same size.")
    if (gt>0).sum()==0 or (gt==0).sum()==0:
        raise ValueError("GT must contain both FG and BG pixels.")

    fg = (gt > 0).astype(np.uint8)
    bg = (gt == 0).astype(np.uint8)
    dist_fg, dist_bg = distance_bands(gt)

    # Cycle policies to reach num-files
    N = args.num_files
    policy_cycle = []
    while len(policy_cycle) < N:
        policy_cycle.extend([p[0] for p in POLICIES])
    policy_cycle = policy_cycle[:N]

    manifest_rows = []

    for i in range(N):
        policy_name = policy_cycle[i]
        # pick policy function
        for (pid, fn) in POLICIES:
            if pid == policy_name:
                policy_fn = fn; break

        # build points using policy (ALL policies constrain pos to FG, neg to BG)
        if policy_name == "uniform":
            pos, neg, pid = policy_fn(fg, bg, args.pairs)
        elif policy_name == "boundary_band":
            pos, neg, pid = policy_fn(fg, bg, dist_fg, dist_bg, args.pairs, (2,12), (2,12))
        elif policy_name == "spread_poisson":
            pos, neg, pid = policy_fn(fg, bg, args.pairs)
        elif policy_name == "near_both":
            pos, neg, pid = policy_fn(fg, bg, dist_fg, dist_bg, args.pairs)
        elif policy_name == "deep_pos_near_neg":
            pos, neg, pid = policy_fn(fg, bg, dist_fg, dist_bg, args.pairs)
        elif policy_name == "grid_like":
            step = np.random.choice([48, 56, 64, 72, 80])
            pos, neg, pid = policy_fn(fg, bg, args.pairs, step=step)
        elif policy_name == "cluster_pos":
            pos, neg, pid = policy_fn(fg, bg, args.pairs)
        elif policy_name == "cluster_neg":
            pos, neg, pid = policy_fn(fg, bg, args.pairs)
        elif policy_name == "edge_heavy_neg":
            pos, neg, pid = policy_fn(fg, bg, dist_bg, args.pairs)
        else:
            pos, neg, pid = policy_uniform(fg, bg, args.pairs)

        # STRICT enforcement: positives in FG, negatives in BG, exact counts
        pos, neg = enforce_strict_counts(pos, neg, fg, bg, args.pairs)

        # Build JSON alternating +, − starting with +
        data = build_one_json(pos, neg)

        # Decide split & filename
        split = "calib" if i < args.calib_count else "eval"
        fname = f"{i:03d}_{pid}_pairs{args.pairs}.json"
        fpath = out_dir / split / fname
        with open(fpath, "w") as f:
            json.dump(data, f, indent=2)

        manifest_rows.append({
            "split": split,
            "filename": str(fpath.relative_to(out_dir)),
            "policy_id": pid,
            "pairs": args.pairs,
            "seed": args.seed,
            "notes": "strict_pos_in_FG_neg_in_BG"
        })

    # Save manifest
    with open(out_dir/"manifest.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["split","filename","policy_id","pairs","seed","notes"])
        writer.writeheader(); writer.writerows(manifest_rows)

    print(f"Done. Wrote {args.calib_count} JSONs to {out_dir/'calib'} and {N-args.calib_count} JSONs to {out_dir/'eval'}.")
    print(f"Manifest: {out_dir/'manifest.csv'}")


if __name__ == "__main__":
    main()
