#!/usr/bin/env python3
"""
Hard prompt faithfulness for SAM (no retraining).
- Enforces: positive regions are protected (can't be erased), negative regions are forbidden (can't be filled).
- Uses: small disks around clicks + attention ownership from final prompt→image attention.
- Implements: logit-space projection on SAM's low-res mask logits, then you can postprocess as usual.

Integration (minimal):
  from hard_prompt import HardPromptEnforcer
  enforcer = HardPromptEnforcer()  # tune knobs below if you want
  masks, scores, logits, extras = enforcer.strict_predict(predictor, sam, img, pts_xy, lbl, multimask_output=False)
  # 'masks' and 'logits' are modified to respect prompts; 'scores' is SAM's original predicted IoU.

This keeps the rest of your code exactly the same.
"""

from dataclasses import dataclass
from typing import Tuple, Optional, Dict
import numpy as np
import torch
import torch.nn.functional as F

# SAM
from segment_anything import SamPredictor

# ----------------- configurable knobs -----------------
@dataclass
class HardPromptConfig:
    # Click neighborhoods in ORIGINAL pixel space
    r_pos_px: int = 12          # radius around positive clicks (protected)
    r_neg_px: int = 12          # radius around negative clicks (forbidden)

    # Attention-ownership → token dominance
    use_attention_ownership: bool = True
    ownership_margin: float = 0.05   # require Aneg - Apos > margin (or vice versa)
    ownership_min_conf: float = 0.10 # and the dominant side must be at least this large

    # Projection strength (logit clamp)
    big_logit: float = 12.0  # ~prob≈(1 - 6e-6) for +12, and ~6e-6 for -12

    # Conflict resolution between P (protected) and F (forbidden)
    # "pos_wins" (default): positives override in conflicts (keeps user-included bits)
    # "neg_wins": negatives override in conflicts (strict exclusion)
    # "drop_conflict": leave conflicting pixels unconstrained
    conflict_policy: str = "pos_wins"  # {"pos_wins", "neg_wins", "drop_conflict"}

    # Computation details
    lowres_side: int = 256  # SAM decoder low-res logit size
    device: Optional[str] = None  # if None, infer from model
    # Safety: if no prompts of a kind exist, the corresponding constraint becomes a no-op
    enable_if_no_pos: bool = False
    enable_if_no_neg: bool = False


# ----------------- utility: attention capture -----------------
def _attach_final_t2i(two_way):
    """
    Hook TwoWay.final_attn_token_to_image to capture final Prompt→Image attention.
    Returns (handles, cache) where cache["A_final"] = [B,H,Nq,Ni] softmax attention.
    """
    cache = {}

    def _get_qkv(args, kwargs):
        if kwargs and all(k in kwargs for k in ("q", "k", "v")):
            return kwargs["q"], kwargs["k"], kwargs["v"]
        if len(args) >= 3:
            return args[0], args[1], args[2]
        raise RuntimeError("Expected (q,k,v) in final attn hook")

    def pre_hook(module, args, kwargs):
        q, k, v = _get_qkv(args, kwargs)
        cache["_final_qkv"] = (q.detach(), k.detach(), v.detach())

    def fwd_hook(module, args, kwargs, output):
        q, k, v = cache.get("_final_qkv", _get_qkv(args, kwargs))
        ql = module.q_proj(q); kl = module.k_proj(k)
        B, Nq, Cq = ql.shape; H = module.num_heads; Ch = Cq // H
        qh = ql.view(B, Nq, H, Ch).permute(0, 2, 1, 3)   # [B,H,Nq,Ch]
        kh = kl.view(B, -1, H, Ch).permute(0, 2, 1, 3)   # [B,H,Ni,Ch]
        logits = torch.matmul(qh, kh.transpose(-1, -2)) / (Ch ** 0.5)
        cache["A_final"] = torch.softmax(logits, dim=-1).detach()  # [B,H,Nq,Ni]

    hs = []
    try:
        hs.append(two_way.final_attn_token_to_image.register_forward_pre_hook(pre_hook, with_kwargs=True))
        hs.append(two_way.final_attn_token_to_image.register_forward_hook(fwd_hook, with_kwargs=True))
    except TypeError:
        # torch<2.0 fallback
        hs.append(two_way.final_attn_token_to_image.register_forward_pre_hook(lambda m, a: pre_hook(m, a, {})))
        hs.append(two_way.final_attn_token_to_image.register_forward_hook(lambda m, a, o: fwd_hook(m, a, {}, o)))
    return hs, cache


def _remove_hooks(handles):
    for h in handles:
        h.remove()


# ----------------- geometry helpers -----------------
def _draw_disks(H, W, pts_xy: np.ndarray, radius: int) -> np.ndarray:
    """
    Rasterize union of disks around points (x,y) in ORIGINAL space.
    """
    if radius <= 0 or pts_xy.size == 0:
        return np.zeros((H, W), dtype=np.uint8)
    out = np.zeros((H, W), dtype=np.uint8)
    xs = np.arange(W, dtype=np.int32)
    ys = np.arange(H, dtype=np.int32)
    xx, yy = np.meshgrid(xs, ys)
    for (x, y) in pts_xy:
        x = int(round(float(x))); y = int(round(float(y)))
        x0 = max(0, x - radius); x1 = min(W, x + radius + 1)
        y0 = max(0, y - radius); y1 = min(H, y + radius + 1)
        if x0 >= x1 or y0 >= y1: 
            continue
        sub = out[y0:y1, x0:x1]
        sub |= (((xx[y0:y1, x0:x1] - x) ** 2 + (yy[y0:y1, x0:x1] - y) ** 2) <= radius * radius).astype(np.uint8)
    return out


def _down_bin_to_lowres(mask_hw: np.ndarray, side=256) -> np.ndarray:
    """Binary mask → area pooled to low-res, then threshold."""
    t = torch.from_numpy(mask_hw.astype(np.float32))[None, None]
    ds = F.interpolate(t, size=(side, side), mode="area")[0, 0]
    return (ds > 0.01).cpu().numpy().astype(np.uint8)


def _tok_to_lowres(tok_hw_bin: np.ndarray, side=256) -> np.ndarray:
    """Token grid (Hf,Wf) binary → upsampled to low-res via nearest."""
    t = torch.from_numpy(tok_hw_bin.astype(np.float32))[None, None]
    up = F.interpolate(t, size=(side, side), mode="nearest")[0, 0]
    return (up > 0.5).cpu().numpy().astype(np.uint8)


# ----------------- main enforcer -----------------
class HardPromptEnforcer:
    def __init__(self, cfg: Optional[HardPromptConfig] = None):
        self.cfg = cfg or HardPromptConfig()

    @torch.no_grad()
    def strict_predict(
        self,
        predictor: SamPredictor,
        sam,
        img_hw3: np.ndarray,
        pts_xy: np.ndarray,
        labels: np.ndarray,
        multimask_output: bool = False,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict]:
        """
        Drop-in replacement for predictor.predict returning (masks, scores, logits, extras)
        but with logits projected to respect prompt faithfulness.
        """
        assert not multimask_output, "This enforcer is implemented for single-mask mode (multimask_output=False)."
        device = self.cfg.device or next(sam.parameters()).device
        H, W = img_hw3.shape[:2]

        # Ensure inputs are proper shape/dtype
        pts_xy = np.asarray(pts_xy, dtype=np.float32).reshape(-1, 2)
        labels = np.asarray(labels, dtype=np.int32).reshape(-1)

        # Fast path: no prompts → just run SAM
        if pts_xy.size == 0:
            masks, scores, logits = predictor.predict(
                point_coords=np.zeros((0, 2), np.float32),
                point_labels=np.zeros((0,), np.int32),
                multimask_output=False,
            )
            return masks, scores, logits, {"P_orig": np.zeros((H, W), np.uint8), "F_orig": np.zeros((H, W), np.uint8)}

        # 1) Run once to get logits and final attention
        predictor.set_image(img_hw3)
        two_way = sam.mask_decoder.transformer
        hs, cache = _attach_final_t2i(two_way)
        masks, scores, logits = predictor.predict(
            point_coords=pts_xy,
            point_labels=labels,
            multimask_output=False,
        )
        _remove_hooks(hs)

        # Features/attn shapes
        feats = predictor.features  # [1, C, Hf, Wf]
        if feats is None:
            raise RuntimeError("Predictor has no encoder features after set_image.")
        _, Cfeat, Hf, Wf = feats.shape
        Ni = Hf * Wf
        A = cache.get("A_final", None)
        if A is None:
            raise RuntimeError("Failed to capture final P→I attention.")
        A = A.mean(dim=1)  # [B,Nq,Ni]
        A = A[0]           # [Nq,Ni]

        # Map prompt tokens
        num_mask_tokens = getattr(sam.mask_decoder, "num_mask_tokens", 3)
        offset = 1 + num_mask_tokens
        Np = len(labels)
        qidx_all = torch.arange(offset, offset + Np, device=A.device)  # [Np]
        if qidx_all.numel() != Np:
            # Safety with older forks
            qidx_all = torch.arange(A.shape[0] - Np, A.shape[0], device=A.device)

        pos_sel = (labels == 1)
        neg_sel = (labels == 0)
        has_pos = bool(pos_sel.any())
        has_neg = bool(neg_sel.any())

        # 2) Build ownership masks on token grid (Hf,Wf)
        pos_tok = np.zeros((Hf, Wf), dtype=np.uint8)
        neg_tok = np.zeros((Hf, Wf), dtype=np.uint8)

        if self.cfg.use_attention_ownership and (has_pos or has_neg):
            # Max attention across pos/neg prompt tokens
            if has_pos:
                Apos = A[qidx_all[pos_sel], :]        # [Npos, Ni]
                Apos_max = Apos.max(dim=0).values     # [Ni]
            else:
                Apos_max = torch.zeros((Ni,), device=A.device)

            if has_neg:
                Aneg = A[qidx_all[neg_sel], :]        # [Nneg, Ni]
                Aneg_max = Aneg.max(dim=0).values     # [Ni]
            else:
                Aneg_max = torch.zeros((Ni,), device=A.device)

            margin = self.cfg.ownership_margin
            conf   = self.cfg.ownership_min_conf

            # pos dominance where Apos >= Aneg + margin and confident
            pos_dom = ((Apos_max - Aneg_max) >= margin) & (Apos_max >= conf)
            # neg dominance where Aneg >= Apos + margin and confident
            neg_dom = ((Aneg_max - Apos_max) >= margin) & (Aneg_max >= conf)

            pos_tok = pos_dom.view(Hf, Wf).detach().cpu().numpy().astype(np.uint8)
            neg_tok = neg_dom.view(Hf, Wf).detach().cpu().numpy().astype(np.uint8)

        # 3) Disks around clicks in ORIGINAL space
        pts_pos = pts_xy[labels == 1] if has_pos else np.zeros((0, 2), np.float32)
        pts_neg = pts_xy[labels == 0] if has_neg else np.zeros((0, 2), np.float32)

        P_orig = _draw_disks(H, W, pts_pos, self.cfg.r_pos_px) if (has_pos or self.cfg.enable_if_no_pos) else np.zeros((H, W), np.uint8)
        F_orig = _draw_disks(H, W, pts_neg, self.cfg.r_neg_px) if (has_neg or self.cfg.enable_if_no_neg) else np.zeros((H, W), np.uint8)

        # 4) Lift token-ownership to ORIGINAL via low-res (256) then postprocess
        pos_low = _tok_to_lowres(pos_tok, side=self.cfg.lowres_side) if pos_tok.any() else np.zeros((self.cfg.lowres_side, self.cfg.lowres_side), np.uint8)
        neg_low = _tok_to_lowres(neg_tok, side=self.cfg.lowres_side) if neg_tok.any() else np.zeros((self.cfg.lowres_side, self.cfg.lowres_side), np.uint8)

        # 5) Merge with disks (in low-res space where we'll project logits)
        P_low = np.clip(pos_low + _down_bin_to_lowres(P_orig, side=self.cfg.lowres_side), 0, 1).astype(np.uint8)
        F_low = np.clip(neg_low + _down_bin_to_lowres(F_orig, side=self.cfg.lowres_side), 0, 1).astype(np.uint8)

        # 6) Resolve conflicts per policy
        conflict = (P_low & F_low).astype(np.uint8)
        if conflict.any():
            if self.cfg.conflict_policy == "pos_wins":
                F_low = F_low & (~conflict)
            elif self.cfg.conflict_policy == "neg_wins":
                P_low = P_low & (~conflict)
            elif self.cfg.conflict_policy == "drop_conflict":
                P_low = P_low & (~conflict)
                F_low = F_low & (~conflict)

        # 7) Project logits (low-res)
        # logits: array (1, 256, 256) or (256,256)
        low = torch.as_tensor(logits).clone()
        if low.ndim == 2:
            low = low[None, :, :]
        assert low.shape[-1] == self.cfg.lowres_side and low.shape[-2] == self.cfg.lowres_side, \
            f"Expected low-res side {self.cfg.lowres_side}, got {tuple(low.shape)}"

        P_t = torch.from_numpy(P_low.astype(np.bool_))
        F_t = torch.from_numpy(F_low.astype(np.bool_))
        big = float(self.cfg.big_logit)

        # clamp: negatives → off, positives → on
        low[0][F_t] = -big
        low[0][P_t] = +big

        logits_proj = low.detach().cpu().numpy()  # keep shape (1,256,256)

        # 8) Recompute binary mask in ORIGINAL space for return
        # We keep SAM's scores untouched; your downstream postprocess will use logits.
        # But to match predictor.predict return signature, we'll return a binary mask (original HxW)
        # using the projected logits thresholded at 0.5 prob.
        # (If your pipeline re-postprocesses from logits, it'll overwrite this anyway.)
        # Here we do a quick postprocess like your helpers:
        prob = torch.sigmoid(
            F.interpolate(low[None, None].float(), size=(H, W), mode="bilinear", align_corners=False)[0, 0]
        ).cpu().numpy()
        mask_bin = (prob >= 0.5).astype(np.uint8)
        masks_proj = np.array([mask_bin], dtype=np.uint8)  # match predictor.predict: (N, H, W)

        extras = {
            "P_orig": P_orig, "F_orig": F_orig,
            "P_low": P_low, "F_low": F_low,
            "pos_tok": pos_tok, "neg_tok": neg_tok,
            "conflict_pixels": int(conflict.sum()),
        }
        return masks_proj, scores, logits_proj, extras
