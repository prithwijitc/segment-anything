#!/usr/bin/env python3
import argparse
import math
from pathlib import Path
from collections import Counter

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
import matplotlib.pyplot as plt

# ---- Import your SAM encoder implementation ----
try:
    from image_encoder import ImageEncoderViT
except ImportError:
    try:
        from segment_anything.modeling.image_encoder import ImageEncoderViT
    except ImportError:
        raise ImportError("Can't import ImageEncoderViT. Edit the import above to match your repo layout.")

# ---------------- Utilities ----------------
def load_prompts_json(path):
    import json
    with open(path, "r") as f:
        data = json.load(f)
    pos = data.get("positive_points", []) or []
    neg = data.get("negative_points", []) or []
    return [(int(x), int(y), "pos") for x, y in pos] + [(int(x), int(y), "neg") for x, y in neg]

def load_sam_state(checkpoint_path: str):
    obj = torch.load(checkpoint_path, map_location="cpu")
    if isinstance(obj, dict) and "model" in obj and isinstance(obj["model"], dict):
        state = obj["model"]
    elif isinstance(obj, dict) and "state_dict" in obj and isinstance(obj["state_dict"], dict):
        state = obj["state_dict"]
    elif isinstance(obj, dict):
        state = obj
    else:
        raise ValueError("Unrecognized checkpoint format.")

    if any(k.startswith("module.") for k in state.keys()):
        state = {k.replace("module.", "", 1): v for k, v in state.items()}

    if any(k.startswith("image_encoder.") for k in state.keys()):
        state = {k.replace("image_encoder.", "", 1): v
                 for k, v in state.items() if k.startswith("image_encoder.")}

    if "patch_embed.proj.weight" not in state:
        raise ValueError("Checkpoint does not contain image encoder weights (missing 'patch_embed.proj.weight').")
    return state

def infer_encoder_config_from_state(state):
    pe_w = state["patch_embed.proj.weight"]  # [embed_dim, in_chans, ks, ks]
    embed_dim = pe_w.shape[0]
    in_chans  = pe_w.shape[1]
    patch_sz  = pe_w.shape[2]

    block_idxs = sorted({int(k.split(".")[1]) for k in state.keys() if k.startswith("blocks.")})
    depth = max(block_idxs) + 1 if block_idxs else 0

    use_abs_pos = "pos_embed" in state
    if use_abs_pos:
        pos = state["pos_embed"]  # [1, Ht, Wt, embed_dim]
        Ht = int(pos.shape[1]); Wt = int(pos.shape[2])
    else:
        Ht = Wt = 1024 // patch_sz
    img_size = Ht * patch_sz

    use_rel_pos = any(f"blocks.{i}.attn.rel_pos_h" in state for i in block_idxs)

    if use_rel_pos:
        ex = next(i for i in block_idxs if f"blocks.{i}.attn.rel_pos_h" in state)
        head_dim = state[f"blocks.{ex}.attn.rel_pos_h"].shape[1]
        num_heads = embed_dim // head_dim
    else:
        num_heads = 12 if embed_dim == 768 else 16

    layer_I = []
    for i in block_idxs:
        k = f"blocks.{i}.attn.rel_pos_h"
        if k in state:
            L = int(state[k].shape[0])
            I = (L + 1) // 2
        else:
            I = Ht
        layer_I.append(I)

    globals_idx = tuple(i for i, I in enumerate(layer_I) if I == Ht)
    window_candidates = [I for I in layer_I if I != Ht]
    window_size = Counter(window_candidates).most_common(1)[0][0] if window_candidates else 0

    cfg = dict(
        img_size=img_size,
        patch_size=patch_sz,
        in_chans=in_chans,
        embed_dim=embed_dim,
        depth=depth,
        num_heads=num_heads,
        mlp_ratio=4.0,
        out_chans=256,
        qkv_bias=True,
        use_abs_pos=use_abs_pos,
        use_rel_pos=use_rel_pos,
        rel_pos_zero_init=True,
        window_size=window_size,
        global_attn_indexes=globals_idx,
    )
    return cfg, layer_I, Ht, Wt

def preprocess_pil_for_sam(pil: Image.Image, img_size: int) -> torch.Tensor:
    pil = pil.resize((img_size, img_size), Image.BILINEAR)
    arr = np.asarray(pil).astype(np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)[None, None, :]
    std  = np.array([0.229, 0.224, 0.225], dtype=np.float32)[None, None, :]
    arr = (arr - mean) / std
    return torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)

def upsample_to_img(heat_hw: np.ndarray, out_h: int, out_w: int):
    t = torch.from_numpy(heat_hw).float()[None, None]
    return F.interpolate(t, size=(out_h, out_w), mode="nearest").squeeze().cpu().numpy()

def overlay(ax, rgb_img: np.ndarray, heat: np.ndarray, title=None):
    h = heat.astype(np.float32)
    h -= h.min()
    h /= (h.max() - h.min() + 1e-8)
    ax.imshow(h, cmap="jet", alpha=1, interpolation="nearest")
    ax.set_axis_off()
    if title:
        ax.set_title(title, fontsize=8)

def window_grid_dims(H: int, W: int, ws: int):
    return math.ceil(H / ws), math.ceil(W / ws)

def token_index_from_xy(x, y, orig_w, orig_h, img_size, patch):
    sx = img_size / orig_w
    sy = img_size / orig_h
    x_p = np.clip(int(x * sx), 0, img_size - 1)
    y_p = np.clip(int(y * sy), 0, img_size - 1)
    i = y_p // patch
    j = x_p // patch
    return int(i), int(j)

# ---- Last layer save ----
def save_last_layer_for_token(encoder, proc_rgb, H_tokens, W_tokens, heads, t_idx, out_path, fig_scale=1.0):
    blk = encoder.blocks[-1]
    att = getattr(blk.attn, "last_attn", None)
    if att is None:
        raise RuntimeError(
            "[ERROR] last layer attn missing. Add `self.last_attn = attn.detach().cpu()` inside Attention.forward()."
        )
    ws = getattr(blk, "window_size", 0)
    att = att.float().numpy()

    fig, axes = plt.subplots(1, heads, figsize=(fig_scale * heads, fig_scale * 1.0), squeeze=False)

    if ws == 0:
        # Global: att has shape (B*heads, T, T) with B=1
        T = H_tokens * W_tokens
        att = att.reshape(heads, T, T)
        for h in range(heads):
            vec = att[h, t_idx, :]
            heat = vec.reshape(H_tokens, W_tokens)
            up = upsample_to_img(heat, encoder.img_size, encoder.img_size)
            overlay(axes[0, h], proc_rgb, up, title=f"Head {h+1}")
    else:
        # Windowed: (num_windows*heads, ws*ws, ws*ws)
        num_win_h, num_win_w = window_grid_dims(H_tokens, W_tokens, ws)
        num_windows = num_win_h * num_win_w
        att = att.reshape(num_windows, heads, ws * ws, ws * ws)

        i_tok = t_idx // W_tokens
        j_tok = t_idx %  W_tokens
        wr, wc = i_tok // ws, j_tok // ws
        win_idx = wr * num_win_w + wc
        ti_local = (i_tok % ws) * ws + (j_tok % ws)

        for h in range(heads):
            row_vec = att[win_idx, h, ti_local, :]  # (ws*ws,)
            heat = np.zeros((H_tokens, W_tokens), dtype=np.float32)
            i0, j0 = wr * ws, wc * ws
            h_span = min(ws, H_tokens - i0)
            w_span = min(ws, W_tokens - j0)
            block = row_vec.reshape(ws, ws)[:h_span, :w_span]
            heat[i0:i0 + h_span, j0:j0 + w_span] = block
            up = upsample_to_img(heat, encoder.img_size, encoder.img_size)
            overlay(axes[0, h], proc_rgb, up, title=f"Head {h+1}")

    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=500, bbox_inches="tight")
    plt.close(fig)

# ---------------- Main ----------------
def main():
    parser = argparse.ArgumentParser("SAM ImageEncoderViT last-layer attention viz per prompt.")
    parser.add_argument("--image", type=str, required=True, help="Path to an image")
    parser.add_argument("--checkpoint", type=str, required=True, help="Full SAM checkpoint (.pth) for vit_b / vit_h")
    parser.add_argument("--prompts", type=str, required=True, help="JSON with positive_points/negative_points (original pixel coords)")
    parser.add_argument("--out-dir", type=str, required=True, help="Directory to save one PNG per prompt (last layer only)")
    parser.add_argument("--fig-scale", type=float, default=1.0, help="Figure width scale (rows=1)")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load state & config
    state = load_sam_state(args.checkpoint)
    cfg, layer_I, Ht, Wt = infer_encoder_config_from_state(state)
    print("[CFG] inferred:", cfg)
    print("[CFG] per-layer input_size (I):", layer_I)

    # Build encoder
    encoder = ImageEncoderViT(
        img_size=cfg["img_size"],
        patch_size=cfg["patch_size"],
        in_chans=cfg["in_chans"],
        embed_dim=cfg["embed_dim"],
        depth=cfg["depth"],
        num_heads=cfg["num_heads"],
        mlp_ratio=cfg["mlp_ratio"],
        out_chans=cfg["out_chans"],
        qkv_bias=cfg["qkv_bias"],
        norm_layer=torch.nn.LayerNorm,
        act_layer=torch.nn.GELU,
        use_abs_pos=cfg["use_abs_pos"],
        use_rel_pos=cfg["use_rel_pos"],
        rel_pos_zero_init=cfg["rel_pos_zero_init"],
        window_size=cfg["window_size"],
        global_attn_indexes=cfg["global_attn_indexes"],
    ).to(device).eval()

    # Load weights
    try:
        encoder.load_state_dict(state, strict=True)
        print("[LOAD] strict=True OK.")
    except Exception as e:
        print(f"[LOAD] strict=True failed: {e}\n[LOAD] retrying strict=False …")
        missing, unexpected = encoder.load_state_dict(state, strict=False)
        print(f"[LOAD] strict=False. Missing={len(missing)} Unexpected={len(unexpected)}")

    # Sanity
    patch = encoder.patch_embed.proj.stride[0]
    H_tokens = encoder.img_size // patch
    W_tokens = encoder.img_size // patch
    heads = encoder.blocks[0].attn.num_heads
    print("[SANITY] img_size:", encoder.img_size, "patch:", patch,
          "grid:", f"{H_tokens}x{W_tokens}", "heads:", heads)

    # Image
    pil = Image.open(args.image).convert("RGB")
    orig_w, orig_h = pil.size
    inp = preprocess_pil_for_sam(pil, encoder.img_size).to(device)
    proc_rgb = np.asarray(pil.resize((encoder.img_size, encoder.img_size), Image.BILINEAR))

    # Forward (populate last layer attn)
    with torch.no_grad():
        _ = encoder(inp)

    # Save one figure per prompt (LAST LAYER ONLY)
    prompts = load_prompts_json(args.prompts)
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[INFO] {len(prompts)} prompt(s)")

    for idx, (x, y, lab) in enumerate(prompts):
        i_tok, j_tok = token_index_from_xy(x, y, orig_w, orig_h, encoder.img_size, patch)
        t_idx = i_tok * W_tokens + j_tok
        fname = f"lastlayer_prompt_{idx:03d}_{lab}_x{x}_y{y}_tok_{i_tok}-{j_tok}.png"
        out_path = out_dir / fname
        print(f"[SAVE] {fname}  (x={x}, y={y}) -> token (i={i_tok}, j={j_tok})")
        save_last_layer_for_token(
            encoder=encoder,
            proc_rgb=proc_rgb,
            H_tokens=H_tokens,
            W_tokens=W_tokens,
            heads=heads,
            t_idx=t_idx,
            out_path=out_path,
            fig_scale=args.fig_scale,
        )

    print(f"[DONE] wrote {len(prompts)} files to {out_dir.resolve()}")

if __name__ == "__main__":
    main()
