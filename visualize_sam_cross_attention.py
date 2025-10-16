
# #!/usr/bin/env python3
# """
# Visualize SAM Two-Way Transformer cross-attention with:
# - Per-prompt figure (t2i block, i2t block, final t2i, i2t argmax of this prompt)
# - Learned-token figure (IoU + mask tokens)
# - NEW: A single combined i2t-argmax ownership map across all prompts, with legend and prompt dots.

# Example:
#   python visualize_sam_cross_attention.py \
#     --image /path/to/image.jpg \
#     --prompts /path/to/prompts.json \
#     --checkpoint /path/to/sam_vit_h_4b8939.pth \
#     --model-type vit_h \
#     --space original \
#     --out /tmp/sam_cross_attn_prompts.png \
#     --out-learned /tmp/sam_cross_attn_learned.png \
#     --out-i2t-all /tmp/sam_cross_attn_i2t_all.png
# """

# import argparse
# import json
# import math
# from pathlib import Path

# import numpy as np
# import torch
# import torch.nn.functional as F
# import matplotlib.pyplot as plt
# import matplotlib.cm as cm
# from matplotlib.colors import Normalize
# from matplotlib.patches import Patch
# from PIL import Image

# # pip install git+https://github.com/facebookresearch/segment-anything.git
# from segment_anything import sam_model_registry, SamPredictor


# # ==========
# # Resizing
# # ==========
# def _resize_2d(arr_hw, out_hw):
#     """arr_hw: (H,W) numpy or torch -> numpy (H_out, W_out) via bilinear resize."""
#     if isinstance(arr_hw, torch.Tensor):
#         t = arr_hw.detach().float()
#     else:
#         t = torch.as_tensor(arr_hw, dtype=torch.float32)
#     t = t[None, None]  # [1,1,H,W]
#     out = F.interpolate(t, size=out_hw, mode="bilinear", align_corners=False)[0, 0]
#     return out.cpu().numpy()


# # =========================
# # Attention math helpers
# # =========================
# def _separate_heads(x, num_heads):
#     # x: [B, N, C] -> [B, H, N, C/H]
#     B, N, C = x.shape
#     x = x.view(B, N, num_heads, C // num_heads)
#     return x.permute(0, 2, 1, 3)


# @torch.no_grad()
# def _compute_attn_from_module(attn_module, q, k, v):
#     """
#     Reproduce the module's attention math to expose weights.
#     Returns (out, attn) where:
#       attn: [B, H, Nq, Nk]
#       out:  [B, Nq, C]
#     """
#     q = attn_module.q_proj(q)
#     k = attn_module.k_proj(k)
#     v = attn_module.v_proj(v)

#     q = _separate_heads(q, attn_module.num_heads)
#     k = _separate_heads(k, attn_module.num_heads)
#     v = _separate_heads(v, attn_module.num_heads)

#     Ch = q.shape[-1]
#     attn = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(Ch)  # [B,H,Nq,Nk]
#     attn = torch.softmax(attn, dim=-1)

#     out = torch.matmul(attn, v)                                  # [B,H,Nq,Ch]
#     out = out.permute(0, 2, 1, 3).reshape(q.shape[0], q.shape[2], -1)  # [B,Nq,H*Ch]
#     out = attn_module.out_proj(out)                              # [B,Nq,C]
#     return out, attn


# # =========================
# # Hooking utilities
# # =========================
# def attach_cross_attn_captures(two_way_transformer, block_idx=-1):
#     """
#     Hook one TwoWayAttentionBlock:
#       cache['t2i_attn']:  [B,H,Nq,Ni]  (Prompt → Image, in that block)
#       cache['i2t_attn']:  [B,H,Ni,Nq]  (Image  → Prompt, in that block)
#     """
#     block = two_way_transformer.layers[block_idx]
#     cache = {}

#     def _get_qkv(args, kwargs):
#         if kwargs and all(k in kwargs for k in ("q", "k", "v")):
#             return kwargs["q"], kwargs["k"], kwargs["v"]
#         if len(args) >= 3:
#             return args[0], args[1], args[2]
#         raise RuntimeError("Hook expected (q,k,v).")

#     # Prompt → Image (block)
#     def pre_hook_t2i(module, args, kwargs):
#         q, k, v = _get_qkv(args, kwargs)
#         cache['_t2i_inputs'] = (q.detach(), k.detach(), v.detach())

#     def fwd_hook_t2i(module, args, kwargs, output):
#         q, k, v = cache.get('_t2i_inputs', _get_qkv(args, kwargs))
#         _, attn = _compute_attn_from_module(module, q, k, v)
#         cache['t2i_attn'] = attn.detach()

#     # Image → Prompt (block)
#     def pre_hook_i2t(module, args, kwargs):
#         q, k, v = _get_qkv(args, kwargs)
#         cache['_i2t_inputs'] = (q.detach(), k.detach(), v.detach())

#     def fwd_hook_i2t(module, args, kwargs, output):
#         q, k, v = cache.get('_i2t_inputs', _get_qkv(args, kwargs))
#         _, attn_qk = _compute_attn_from_module(module, q, k, v)
#         cache['i2t_attn'] = attn_qk.detach()

#     handles = []
#     try:
#         handles.append(block.cross_attn_token_to_image.register_forward_pre_hook(pre_hook_t2i, with_kwargs=True))
#         handles.append(block.cross_attn_token_to_image.register_forward_hook(fwd_hook_t2i, with_kwargs=True))
#         handles.append(block.cross_attn_image_to_token.register_forward_pre_hook(pre_hook_i2t, with_kwargs=True))
#         handles.append(block.cross_attn_image_to_token.register_forward_hook(fwd_hook_i2t, with_kwargs=True))
#     except TypeError:
#         handles.append(block.cross_attn_token_to_image.register_forward_pre_hook(lambda m, a: pre_hook_t2i(m, a, {})))
#         handles.append(block.cross_attn_token_to_image.register_forward_hook(lambda m, a, o: fwd_hook_t2i(m, a, {}, o)))
#         handles.append(block.cross_attn_image_to_token.register_forward_pre_hook(lambda m, a: pre_hook_i2t(m, a, {})))
#         handles.append(block.cross_attn_image_to_token.register_forward_hook(lambda m, a, o: fwd_hook_i2t(m, a, {}, o)))
#     return handles, cache


# def attach_final_t2i_hook(two_way_transformer, cache):
#     """
#     Hook the final post-loop Prompt→Image attention:
#       cache['final_t2i_attn']: [B,H,Nq,Ni]
#     """
#     def _get_qkv(args, kwargs):
#         if kwargs and all(k in kwargs for k in ("q", "k", "v")):
#             return kwargs["q"], kwargs["k"], kwargs["v"]
#         if len(args) >= 3:
#             return args[0], args[1], args[2]
#         raise RuntimeError("Hook expected (q,k,v) for final_t2i.")

#     def pre_hook(module, args, kwargs):
#         q, k, v = _get_qkv(args, kwargs)
#         cache['_final_t2i_inputs'] = (q.detach(), k.detach(), v.detach())

#     def fwd_hook(module, args, kwargs, output):
#         q, k, v = cache.get('_final_t2i_inputs', _get_qkv(args, kwargs))
#         _, attn = _compute_attn_from_module(module, q, k, v)
#         cache['final_t2i_attn'] = attn.detach()

#     hs = []
#     try:
#         hs.append(two_way_transformer.final_attn_token_to_image.register_forward_pre_hook(pre_hook, with_kwargs=True))
#         hs.append(two_way_transformer.final_attn_token_to_image.register_forward_hook(fwd_hook, with_kwargs=True))
#     except TypeError:
#         hs.append(two_way_transformer.final_attn_token_to_image.register_forward_pre_hook(lambda m, a: pre_hook(m, a, {})))
#         hs.append(two_way_transformer.final_attn_token_to_image.register_forward_hook(lambda m, a, o: fwd_hook(m, a, {}, o)))
#     return hs


# def remove_hooks(handles):
#     for h in handles:
#         h.remove()


# # =========================
# # Heatmap & geometry
# # =========================
# @torch.no_grad()
# def prompt_to_image_heatmap(attn_t2i, prompt_idx, head_reduce="mean", hw_feat=None):
#     B, Hh, Nq, Ni = attn_t2i.shape
#     a = attn_t2i[:, :, prompt_idx, :]  # [B,Hh,Ni]
#     if head_reduce == "mean":
#         a = a.mean(dim=1)
#     elif head_reduce == "max":
#         a, _ = a.max(dim=1)
#     else:
#         raise ValueError("head_reduce must be 'mean' or 'max'.")
#     Hf, Wf = hw_feat
#     a = a.view(B, 1, Hf, Wf)
#     a = (a - a.amin(dim=(2, 3), keepdim=True)) / (a.amax(dim=(2, 3), keepdim=True) - a.amin(dim=(2, 3), keepdim=True) + 1e-8)
#     return a


# @torch.no_grad()
# def image_to_prompt_heatmap(attn_i2t, prompt_idx, head_reduce="mean", hw_feat=None):
#     B, Hh, Ni, Nq = attn_i2t.shape
#     a = attn_i2t[:, :, :, prompt_idx]  # [B,Hh,Ni]
#     if head_reduce == "mean":
#         a = a.mean(dim=1)
#     elif head_reduce == "max":
#         a, _ = a.max(dim=1)
#     else:
#         raise ValueError("head_reduce must be 'mean' or 'max'.")
#     Hf, Wf = hw_feat
#     a = a.view(B, 1, Hf, Wf)
#     a = (a - a.amin(dim=(2, 3), keepdim=True)) / (a.amax(dim=(2, 3), keepdim=True) - a.amin(dim=(2, 3), keepdim=True) + 1e-8)
#     return a


# def load_image_rgb(path):
#     im = Image.open(path).convert("RGB")
#     return np.asarray(im)


# def apply_model_transform_image(predictor, img_rgb):
#     return predictor.transform.apply_image(img_rgb)  # -> (target, target, 3)


# def apply_model_transform_points(predictor, pts_xy, orig_hw):
#     pts = predictor.transform.apply_coords(pts_xy.copy(), orig_hw)
#     return pts


# def feature_heatmap_to_model_space(h_feat, model_side):
#     return _resize_2d(h_feat, (model_side, model_side))


# def feature_heatmap_to_original_space(h_feat, orig_hw, predictor):
#     H_orig, W_orig = orig_hw
#     target = predictor.transform.target_length
#     h_sq = _resize_2d(h_feat, (target, target))
#     scale = float(target) / float(max(H_orig, W_orig))
#     new_h = int(round(H_orig * scale)); new_w = int(round(W_orig * scale))
#     h_cropped = h_sq[:new_h, :new_w]
#     h_orig = _resize_2d(h_cropped, (H_orig, W_orig))
#     return h_orig


# def overlay_heatmap(bg_rgb_uint8, heat_01, alpha=0.45, cmap='jet'):
#     Hbg, Wbg = bg_rgb_uint8.shape[:2]
#     if isinstance(heat_01, torch.Tensor):
#         heat_01 = heat_01.detach().cpu().float().numpy()
#     if heat_01.shape != (Hbg, Wbg):
#         heat_01 = _resize_2d(heat_01, (Hbg, Wbg))
#     base = bg_rgb_uint8.astype(np.float32) / 255.0
#     colored = cm.get_cmap(cmap)(np.clip(heat_01, 0.0, 1.0))[..., :3]
#     out = (1 - alpha) * base + alpha * colored
#     return np.clip(out, 0, 1)


# # =========================
# # i2t argmax (categorical, nearest) + combined overlay
# # =========================
# @torch.no_grad()
# def i2t_argmax_over_prompts(attn_i2t, prompt_indices, head_reduce="mean", hw_feat=None):
#     B, Hh, Ni, Nq = attn_i2t.shape
#     if head_reduce == "mean":
#         a = attn_i2t.mean(dim=1)  # [B,Ni,Nq]
#     elif head_reduce == "max":
#         a, _ = attn_i2t.max(dim=1)
#     else:
#         raise ValueError("head_reduce must be 'mean' or 'max'.")
#     a = a[:, :, prompt_indices]     # [B,Ni,Nprompts]
#     winner = a.argmax(dim=2)        # [B,Ni]
#     Hf, Wf = hw_feat
#     return winner.view(B, Hf, Wf)[0].detach().cpu().numpy().astype(np.int64)


# def categorical_to_model_space(argmax_feat_hw, model_side):
#     t = torch.as_tensor(argmax_feat_hw, dtype=torch.int64)[None, None].float()
#     up = F.interpolate(t, size=(model_side, model_side), mode='nearest')[0, 0]
#     return up.cpu().numpy().astype(np.int64)


# def categorical_to_original_space(argmax_feat_hw, orig_hw, predictor):
#     H_orig, W_orig = orig_hw
#     target = predictor.transform.target_length
#     t = torch.as_tensor(argmax_feat_hw, dtype=torch.int64)[None, None].float()
#     sq = F.interpolate(t, size=(target, target), mode='nearest')[0, 0]
#     scale = float(target) / float(max(H_orig, W_orig))
#     new_h = int(round(H_orig * scale)); new_w = int(round(W_orig * scale))
#     sq = sq[:new_h, :new_w]
#     out = F.interpolate(sq[None, None], size=(H_orig, W_orig), mode='nearest')[0, 0]
#     return out.cpu().numpy().astype(np.int64)


# def distinct_prompt_colors(n):
#     # Prefer tab20/tab10; fall back to HSV for >20
#     if n <= 10:
#         cmap = plt.get_cmap('tab10')
#         cols = [tuple(cmap(i)[:3]) for i in range(n)]
#     elif n <= 20:
#         cmap = plt.get_cmap('tab20')
#         cols = [tuple(cmap(i)[:3]) for i in range(n)]
#     else:
#         import colorsys
#         cols = [colorsys.hsv_to_rgb((i / n) % 1.0, 0.65, 0.95) for i in range(n)]
#     return cols


# def overlay_categorical_rgb(bg_rgb_uint8, labels_hw, colors_rgb, alpha=0.45):
#     """
#     Composite categorical labels (0..K-1) over an RGB background using class colors.
#     """
#     base = bg_rgb_uint8.astype(np.float32) / 255.0
#     out = base.copy()
#     H, W = labels_hw.shape
#     for i, col in enumerate(colors_rgb):
#         mask = (labels_hw == i).astype(np.float32)
#         if mask.sum() == 0:
#             continue
#         mask3 = mask[..., None]
#         col_arr = np.array(col, dtype=np.float32)[None, None, :]
#         out = (1 - alpha * mask3) * out + (alpha * mask3) * col_arr
#     return np.clip(out, 0, 1)


# # =========================
# # Main
# # =========================
# def run(args):
#     device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
#     img_rgb = load_image_rgb(args.image)
#     H_img, W_img = img_rgb.shape[:2]

#     with open(args.prompts, "r") as f:
#         prom = json.load(f)

#     pos_pts = np.array(prom.get("positive_points", []), dtype=np.float32) if prom.get("positive_points") else np.zeros((0, 2), np.float32)
#     neg_pts = np.array(prom.get("negative_points", []), dtype=np.float32) if prom.get("negative_points") else np.zeros((0, 2), np.float32)
#     all_pts = np.concatenate([pos_pts, neg_pts], axis=0) if (len(pos_pts) + len(neg_pts)) > 0 else np.zeros((0, 2), np.float32)
#     all_lbl = np.array([1] * len(pos_pts) + [0] * len(neg_pts), dtype=np.int32)

#     if all_pts.shape[0] == 0:
#         raise ValueError("No prompts found in JSON (need positive_points and/or negative_points).")

#     # Build SAM
#     sam = sam_model_registry[args.model_type](checkpoint=args.checkpoint)
#     sam.to(device).eval()
#     predictor = SamPredictor(sam)

#     # Set image (predictor caches transforms/features)
#     predictor.set_image(img_rgb)

#     # Hooks
#     two_way = sam.mask_decoder.transformer
#     handles_block, cache = attach_cross_attn_captures(two_way, block_idx=args.block_idx)
#     handles_final = attach_final_t2i_hook(two_way, cache)

#     # Run SAM once
#     with torch.inference_mode():
#         _masks, _scores, _logits = predictor.predict(
#             point_coords=all_pts,
#             point_labels=all_lbl,
#             multimask_output=False,
#         )

#     remove_hooks(handles_block + handles_final)

#     # Feature grid size
#     if predictor.features is None:
#         raise RuntimeError("No features cached in predictor.")
#     _, Cfeat, Hf, Wf = predictor.features.shape
#     hw_feat = (Hf, Wf)

#     for k in ['t2i_attn', 'i2t_attn', 'final_t2i_attn']:
#         if k not in cache:
#             raise RuntimeError(f"Failed to capture attention '{k}'. Check block index/SAM version.")

#     # Token layout: [IoU(1), mask_tokens(num_mask_tokens), ...user prompt tokens...]
#     num_mask_tokens = getattr(sam.mask_decoder, "num_mask_tokens", 3)
#     prompt_offset = 1 + num_mask_tokens
#     N_prompts = all_pts.shape[0]
#     prompt_indices = list(range(prompt_offset, prompt_offset + N_prompts))

#     # Visualization space
#     space = args.space.lower()
#     assert space in ("original", "model")
#     if space == "original":
#         bg_img = img_rgb
#         pts_vis = all_pts.copy()
#         H_vis, W_vis = H_img, W_img
#         to_space = lambda feat: feature_heatmap_to_original_space(feat, (H_img, W_img), predictor)
#         cat_to_space = lambda argmax_hw: categorical_to_original_space(argmax_hw, (H_img, W_img), predictor)
#     else:
#         target = predictor.transform.target_length
#         bg_img = apply_model_transform_image(predictor, img_rgb)
#         pts_vis = apply_model_transform_points(predictor, all_pts.copy(), (H_img, W_img))
#         H_vis, W_vis = target, target
#         to_space = lambda feat: feature_heatmap_to_model_space(feat, target)
#         cat_to_space = lambda argmax_hw: categorical_to_model_space(argmax_hw, target)

#     # ========= MAIN: per-prompt figure (4 columns, with colorbars) =========
#     N = N_prompts
#     ncols = 4
#     fig_h = max(2.8, 2.2 * N)
#     fig, axes = plt.subplots(N, ncols, figsize=(18, fig_h), squeeze=False)
#     blk_title = f"(block {args.block_idx})"
#     norm01 = Normalize(vmin=0.0, vmax=1.0)

#     # ARGMAX over prompts (feature resolution) and resize categorically (nearest)
#     argmax_idx_feat = i2t_argmax_over_prompts(cache['i2t_attn'], prompt_indices, head_reduce=args.head_reduce, hw_feat=hw_feat)
#     argmax_rs = cat_to_space(argmax_idx_feat)

#     for i in range(N):
#         tok_idx = prompt_indices[i]

#         hm_t2i_blk   = prompt_to_image_heatmap(cache['t2i_attn'],      tok_idx, args.head_reduce, hw_feat)[0, 0]
#         hm_i2t_blk   = image_to_prompt_heatmap(cache['i2t_attn'],      tok_idx, args.head_reduce, hw_feat)[0, 0]
#         hm_t2i_final = prompt_to_image_heatmap(cache['final_t2i_attn'],tok_idx, args.head_reduce, hw_feat)[0, 0]

#         hm_t2i_blk_rs   = to_space(hm_t2i_blk)
#         hm_i2t_blk_rs   = to_space(hm_i2t_blk)
#         hm_t2i_final_rs = to_space(hm_t2i_final)

#         bin_rs = (argmax_rs == i).astype(np.float32)

#         ov_t2i_blk   = overlay_heatmap(bg_img, hm_t2i_blk_rs,   alpha=args.alpha, cmap=args.cmap)
#         ov_i2t_blk   = overlay_heatmap(bg_img, hm_i2t_blk_rs,   alpha=args.alpha, cmap=args.cmap)
#         ov_t2i_final = overlay_heatmap(bg_img, hm_t2i_final_rs, alpha=args.alpha, cmap=args.cmap)

#         cmap_arg = 'Greens' if (i < len(all_lbl) and all_lbl[i] == 1) else 'Reds'
#         ov_arg = overlay_heatmap(bg_img, bin_rs, alpha=args.alpha, cmap=cmap_arg)

#         x, y = float(pts_vis[i, 0]), float(pts_vis[i, 1])
#         color = 'lime' if all_lbl[i] == 1 else 'red'
#         label_str = '(+)' if all_lbl[i] == 1 else '(-)'

#         ax = axes[i, 0]
#         ax.imshow(ov_t2i_blk); ax.scatter([x],[y], c=color, s=36, marker='o', edgecolor='black', linewidths=0.6)
#         ax.set_title(f'Prompt→Image {blk_title} {label_str} idx={i}', fontsize=10); ax.set_axis_off()
#         sm1 = cm.ScalarMappable(norm=norm01, cmap=args.cmap); sm1.set_array(hm_t2i_blk_rs)
#         fig.colorbar(sm1, ax=ax, fraction=0.035, pad=0.02).set_label('attention', fontsize=8)

#         ax = axes[i, 1]
#         ax.imshow(ov_i2t_blk); ax.scatter([x],[y], c=color, s=36, marker='o', edgecolor='black', linewidths=0.6)
#         ax.set_title(f'Image→Prompt {blk_title} {label_str} idx={i}', fontsize=10); ax.set_axis_off()
#         sm2 = cm.ScalarMappable(norm=norm01, cmap=args.cmap); sm2.set_array(hm_i2t_blk_rs)
#         fig.colorbar(sm2, ax=ax, fraction=0.035, pad=0.02).set_label('attention', fontsize=8)

#         ax = axes[i, 2]
#         ax.imshow(ov_t2i_final); ax.scatter([x],[y], c=color, s=36, marker='o', edgecolor='black', linewidths=0.6)
#         ax.set_title(f'Final Prompt→Image {label_str} idx={i}', fontsize=10); ax.set_axis_off()
#         sm3 = cm.ScalarMappable(norm=norm01, cmap=args.cmap); sm3.set_array(hm_t2i_final_rs)
#         fig.colorbar(sm3, ax=ax, fraction=0.035, pad=0.02).set_label('attention', fontsize=8)

#         ax = axes[i, 3]
#         ax.imshow(ov_arg); ax.scatter([x],[y], c=color, s=36, marker='o', edgecolor='black', linewidths=0.6)
#         ax.set_title(f'i2t argmax = this prompt {label_str} idx={i}', fontsize=10); ax.set_axis_off()
#         sm4 = cm.ScalarMappable(norm=norm01, cmap=cmap_arg); sm4.set_array(bin_rs)
#         fig.colorbar(sm4, ax=ax, fraction=0.035, pad=0.02).set_label('argmax mask (0/1)', fontsize=8)

#     plt.tight_layout()
#     out_path = Path(args.out); out_path.parent.mkdir(parents=True, exist_ok=True)
#     plt.savefig(out_path, dpi=200, bbox_inches='tight'); plt.close(fig)
#     print(f"[OK] Saved prompts figure: {out_path}")

#     # ========= LEARNED TOKENS figure =========
#     learned_names = ["iou"] + [f"mask{m}" for m in range(num_mask_tokens)]
#     learned_indices = list(range(0, 1 + num_mask_tokens))
#     M = len(learned_indices)
#     fig_l, axes_l = plt.subplots(M, 3, figsize=(14, max(2.8, 2.2 * M)), squeeze=False)
#     for j in range(M):
#         tok_idx = learned_indices[j]; name = learned_names[j]
#         hm_t2i_blk = prompt_to_image_heatmap(cache['t2i_attn'], tok_idx, args.head_reduce, hw_feat)[0, 0]
#         hm_i2t_blk = image_to_prompt_heatmap(cache['i2t_attn'], tok_idx, args.head_reduce, hw_feat)[0, 0]
#         hm_t2i_final = prompt_to_image_heatmap(cache['final_t2i_attn'], tok_idx, args.head_reduce, hw_feat)[0, 0]
#         hm_t2i_blk_rs, hm_i2t_blk_rs, hm_t2i_final_rs = to_space(hm_t2i_blk), to_space(hm_i2t_blk), to_space(hm_t2i_final)
#         ov1, ov2, ov3 = overlay_heatmap(bg_img, hm_t2i_blk_rs, args.alpha, args.cmap), overlay_heatmap(bg_img, hm_i2t_blk_rs, args.alpha, args.cmap), overlay_heatmap(bg_img, hm_t2i_final_rs, args.alpha, args.cmap)
#         for k, (ax, ov, title, hm) in enumerate([
#             (axes_l[j,0], ov1, f'[LEARNED:{name}] Prompt→Image {blk_title}', hm_t2i_blk_rs),
#             (axes_l[j,1], ov2, f'[LEARNED:{name}] Image→Prompt {blk_title}', hm_i2t_blk_rs),
#             (axes_l[j,2], ov3, f'[LEARNED:{name}] Final Prompt→Image', hm_t2i_final_rs),
#         ]):
#             ax.imshow(ov); ax.set_title(title, fontsize=10); ax.set_axis_off()
#             sm = cm.ScalarMappable(norm=Normalize(0,1), cmap=args.cmap); sm.set_array(hm)
#             fig_l.colorbar(sm, ax=ax, fraction=0.035, pad=0.02).set_label('attention', fontsize=8)
#     plt.tight_layout()
#     out_learned = Path(args.out_learned); out_learned.parent.mkdir(parents=True, exist_ok=True)
#     plt.savefig(out_learned, dpi=200, bbox_inches='tight'); plt.close(fig_l)
#     print(f"[OK] Saved learned-tokens figure: {out_learned}")

#     # ========= NEW: Combined i2t-argmax ownership map (single plot) =========
#     # Build a color palette for N prompts and overlay all at once.
#     colors = distinct_prompt_colors(N)
#     combined_overlay = overlay_categorical_rgb(bg_img, argmax_rs, colors, alpha=args.alpha)

#     # Plot + legend + prompt dots in same colors
#     fig_c, axc = plt.subplots(1, 1, figsize=(12, 12 * (H_vis / W_vis)))
#     axc.imshow(combined_overlay)
#     handles = []
#     for i in range(N):
#         x, y = float(pts_vis[i, 0]), float(pts_vis[i, 1])
#         axc.scatter([x], [y], c=[colors[i]], s=50, marker='o', edgecolor='black', linewidths=0.8)
#         lab = f"P{i} ({'+' if all_lbl[i]==1 else '-'})"
#         handles.append(Patch(facecolor=colors[i], edgecolor='black', label=lab))
#     axc.legend(handles=handles, loc='upper right', fontsize=9, frameon=True, title="Prompts")
#     axc.set_title("Image→Prompt ARGMAX (combined ownership map)", fontsize=12)
#     axc.set_axis_off()

#     out_i2t_all = Path(args.out_i2t_all); out_i2t_all.parent.mkdir(parents=True, exist_ok=True)
#     plt.tight_layout()
#     plt.savefig(out_i2t_all, dpi=200, bbox_inches='tight'); plt.close(fig_c)
#     print(f"[OK] Saved combined i2t-argmax figure: {out_i2t_all}")


# def parse_args():
#     ap = argparse.ArgumentParser(description="SAM cross-attn viz with per-prompt panels, learned tokens, and a combined i2t-argmax map.")
#     ap.add_argument("--image", required=True, help="Path to input image")
#     ap.add_argument("--prompts", required=True, help="JSON with positive_points/negative_points (coords in ORIGINAL image space)")
#     ap.add_argument("--checkpoint", required=True, help="Path to SAM checkpoint (e.g., sam_vit_h_4b8939.pth)")
#     ap.add_argument("--model-type", default="vit_h", choices=["vit_h", "vit_l", "vit_b"], help="SAM model type")
#     ap.add_argument("--block-idx", type=int, default=-1, help="Transformer block index to visualize (-1 = last)")
#     ap.add_argument("--head-reduce", default="mean", choices=["mean", "max"], help="Aggregate across heads")
#     ap.add_argument("--alpha", type=float, default=0.6, help="Overlay alpha")
#     ap.add_argument("--cmap", default="jet", help="Matplotlib colormap for attention overlays")
#     ap.add_argument("--space", default="original", choices=["original", "model"], help="Overlay space")
#     ap.add_argument("--out", default="cross_attn_prompts.png", help="Output PNG for user prompts figure")
#     ap.add_argument("--out-learned", default="cross_attn_learned.png", help="Output PNG for learned tokens figure")
#     ap.add_argument("--out-i2t-all", default="cross_attn_i2t_all.png", help="Output PNG for combined i2t-argmax map")
#     ap.add_argument("--cpu", action="store_true", help="Force CPU")
#     return ap.parse_args()


# if __name__ == "__main__":
#     args = parse_args()
#     run(args)


#!/usr/bin/env python3
"""
Visualize SAM Two-Way Transformer cross-attention with:
- Per-prompt figure (t2i block, i2t block, final t2i, i2t argmax of this prompt)
- Learned-token figure (IoU + mask tokens)
- Combined i2t-argmax ownership map across all prompts, with legend and prompt dots.
- NEW: Multimask figure (usually 3 masks) overlayed on the image with SAM's predicted IoU scores.

Example:
  python visualize_sam_cross_attention.py \
    --image /path/to/image.jpg \
    --prompts /path/to/prompts.json \
    --checkpoint /path/to/sam_vit_h_4b8939.pth \
    --model-type vit_h \
    --space original \
    --out /tmp/sam_cross_attn_prompts.png \
    --out-learned /tmp/sam_cross_attn_learned.png \
    --out-i2t-all /tmp/sam_cross_attn_i2t_all.png \
    --out-masks /tmp/sam_multimasks.png
"""

import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import Normalize
from matplotlib.patches import Patch
from PIL import Image

# pip install git+https://github.com/facebookresearch/segment-anything.git
from segment_anything import sam_model_registry, SamPredictor


# ==========
# Resizing
# ==========
def _resize_2d(arr_hw, out_hw):
    """arr_hw: (H,W) numpy or torch -> numpy (H_out, W_out) via bilinear resize."""
    if isinstance(arr_hw, torch.Tensor):
        t = arr_hw.detach().float()
    else:
        t = torch.as_tensor(arr_hw, dtype=torch.float32)
    t = t[None, None]  # [1,1,H,W]
    out = F.interpolate(t, size=out_hw, mode="bilinear", align_corners=False)[0, 0]
    return out.cpu().numpy()


# =========================
# Attention math helpers
# =========================
def _separate_heads(x, num_heads):
    # x: [B, N, C] -> [B, H, N, C/H]
    B, N, C = x.shape
    x = x.view(B, N, num_heads, C // num_heads)
    return x.permute(0, 2, 1, 3)


@torch.no_grad()
def _compute_attn_from_module(attn_module, q, k, v):
    """
    Reproduce the module's attention math to expose weights.
    Returns (out, attn) where:
      attn: [B, H, Nq, Nk]
      out:  [B, Nq, C]
    """
    q = attn_module.q_proj(q)
    k = attn_module.k_proj(k)
    v = attn_module.v_proj(v)

    q = _separate_heads(q, attn_module.num_heads)
    k = _separate_heads(k, attn_module.num_heads)
    v = _separate_heads(v, attn_module.num_heads)

    Ch = q.shape[-1]
    attn = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(Ch)  # [B,H,Nq,Nk]
    attn = torch.softmax(attn, dim=-1)

    out = torch.matmul(attn, v)                                  # [B,H,Nq,Ch]
    out = out.permute(0, 2, 1, 3).reshape(q.shape[0], q.shape[2], -1)  # [B,Nq,H*Ch]
    out = attn_module.out_proj(out)                              # [B,Nq,C]
    return out, attn


# =========================
# Hooking utilities
# =========================
def attach_cross_attn_captures(two_way_transformer, block_idx=-1):
    """
    Hook one TwoWayAttentionBlock:
      cache['t2i_attn']:  [B,H,Nq,Ni]  (Prompt → Image, in that block)
      cache['i2t_attn']:  [B,H,Ni,Nq]  (Image  → Prompt, in that block)
    """
    block = two_way_transformer.layers[block_idx]
    cache = {}

    def _get_qkv(args, kwargs):
        if kwargs and all(k in kwargs for k in ("q", "k", "v")):
            return kwargs["q"], kwargs["k"], kwargs["v"]
        if len(args) >= 3:
            return args[0], args[1], args[2]
        raise RuntimeError("Hook expected (q,k,v).")

    # Prompt → Image (block)
    def pre_hook_t2i(module, args, kwargs):
        q, k, v = _get_qkv(args, kwargs)
        cache['_t2i_inputs'] = (q.detach(), k.detach(), v.detach())

    def fwd_hook_t2i(module, args, kwargs, output):
        q, k, v = cache.get('_t2i_inputs', _get_qkv(args, kwargs))
        _, attn = _compute_attn_from_module(module, q, k, v)
        cache['t2i_attn'] = attn.detach()

    # Image → Prompt (block)
    def pre_hook_i2t(module, args, kwargs):
        q, k, v = _get_qkv(args, kwargs)
        cache['_i2t_inputs'] = (q.detach(), k.detach(), v.detach())

    def fwd_hook_i2t(module, args, kwargs, output):
        q, k, v = cache.get('_i2t_inputs', _get_qkv(args, kwargs))
        _, attn_qk = _compute_attn_from_module(module, q, k, v)
        cache['i2t_attn'] = attn_qk.detach()

    handles = []
    try:
        handles.append(block.cross_attn_token_to_image.register_forward_pre_hook(pre_hook_t2i, with_kwargs=True))
        handles.append(block.cross_attn_token_to_image.register_forward_hook(fwd_hook_t2i, with_kwargs=True))
        handles.append(block.cross_attn_image_to_token.register_forward_pre_hook(pre_hook_i2t, with_kwargs=True))
        handles.append(block.cross_attn_image_to_token.register_forward_hook(fwd_hook_i2t, with_kwargs=True))
    except TypeError:
        handles.append(block.cross_attn_token_to_image.register_forward_pre_hook(lambda m, a: pre_hook_t2i(m, a, {})))
        handles.append(block.cross_attn_token_to_image.register_forward_hook(lambda m, a, o: fwd_hook_t2i(m, a, {}, o)))
        handles.append(block.cross_attn_image_to_token.register_forward_pre_hook(lambda m, a: pre_hook_i2t(m, a, {})))
        handles.append(block.cross_attn_image_to_token.register_forward_hook(lambda m, a, o: fwd_hook_i2t(m, a, {}, o)))
    return handles, cache


def attach_final_t2i_hook(two_way_transformer, cache):
    """
    Hook the final post-loop Prompt→Image attention:
      cache['final_t2i_attn']: [B,H,Nq,Ni]
    """
    def _get_qkv(args, kwargs):
        if kwargs and all(k in kwargs for k in ("q", "k", "v")):
            return kwargs["q"], kwargs["k"], kwargs["v"]
        if len(args) >= 3:
            return args[0], args[1], args[2]
        raise RuntimeError("Hook expected (q,k,v) for final_t2i.")

    def pre_hook(module, args, kwargs):
        q, k, v = _get_qkv(args, kwargs)
        cache['_final_t2i_inputs'] = (q.detach(), k.detach(), v.detach())

    def fwd_hook(module, args, kwargs, output):
        q, k, v = cache.get('_final_t2i_inputs', _get_qkv(args, kwargs))
        _, attn = _compute_attn_from_module(module, q, k, v)
        cache['final_t2i_attn'] = attn.detach()

    hs = []
    try:
        hs.append(two_way_transformer.final_attn_token_to_image.register_forward_pre_hook(pre_hook, with_kwargs=True))
        hs.append(two_way_transformer.final_attn_token_to_image.register_forward_hook(fwd_hook, with_kwargs=True))
    except TypeError:
        hs.append(two_way_transformer.final_attn_token_to_image.register_forward_pre_hook(lambda m, a: pre_hook(m, a, {})))
        hs.append(two_way_transformer.final_attn_token_to_image.register_forward_hook(lambda m, a, o: fwd_hook(m, a, {}, o)))
    return hs


def remove_hooks(handles):
    for h in handles:
        h.remove()


# =========================
# Heatmap & geometry
# =========================
@torch.no_grad()
def prompt_to_image_heatmap(attn_t2i, prompt_idx, head_reduce="mean", hw_feat=None):
    B, Hh, Nq, Ni = attn_t2i.shape
    a = attn_t2i[:, :, prompt_idx, :]  # [B,Hh,Ni]
    if head_reduce == "mean":
        a = a.mean(dim=1)
    elif head_reduce == "max":
        a, _ = a.max(dim=1)
    else:
        raise ValueError("head_reduce must be 'mean' or 'max'.")
    Hf, Wf = hw_feat
    a = a.view(B, 1, Hf, Wf)
    a = (a - a.amin(dim=(2, 3), keepdim=True)) / (a.amax(dim=(2, 3), keepdim=True) - a.amin(dim=(2, 3), keepdim=True) + 1e-8)
    return a


@torch.no_grad()
def image_to_prompt_heatmap(attn_i2t, prompt_idx, head_reduce="mean", hw_feat=None):
    B, Hh, Ni, Nq = attn_i2t.shape
    a = attn_i2t[:, :, :, prompt_idx]  # [B,Hh,Ni]
    if head_reduce == "mean":
        a = a.mean(dim=1)
    elif head_reduce == "max":
        a, _ = a.max(dim=1)
    else:
        raise ValueError("head_reduce must be 'mean' or 'max'.")
    Hf, Wf = hw_feat
    a = a.view(B, 1, Hf, Wf)
    a = (a - a.amin(dim=(2, 3), keepdim=True)) / (a.amax(dim=(2, 3), keepdim=True) - a.amin(dim=(2, 3), keepdim=True) + 1e-8)
    return a


def load_image_rgb(path):
    im = Image.open(path).convert("RGB")
    return np.asarray(im)


def apply_model_transform_image(predictor, img_rgb):
    return predictor.transform.apply_image(img_rgb)  # -> (target, target, 3)


def apply_model_transform_points(predictor, pts_xy, orig_hw):
    pts = predictor.transform.apply_coords(pts_xy.copy(), orig_hw)
    return pts


def feature_heatmap_to_model_space(h_feat, model_side):
    return _resize_2d(h_feat, (model_side, model_side))


def feature_heatmap_to_original_space(h_feat, orig_hw, predictor):
    H_orig, W_orig = orig_hw
    target = predictor.transform.target_length
    h_sq = _resize_2d(h_feat, (target, target))
    scale = float(target) / float(max(H_orig, W_orig))
    new_h = int(round(H_orig * scale)); new_w = int(round(W_orig * scale))
    h_cropped = h_sq[:new_h, :new_w]
    h_orig = _resize_2d(h_cropped, (H_orig, W_orig))
    return h_orig


def overlay_heatmap(bg_rgb_uint8, heat_01, alpha=0.45, cmap='jet'):
    Hbg, Wbg = bg_rgb_uint8.shape[:2]
    if isinstance(heat_01, torch.Tensor):
        heat_01 = heat_01.detach().cpu().float().numpy()
    if heat_01.shape != (Hbg, Wbg):
        heat_01 = _resize_2d(heat_01, (Hbg, Wbg))
    base = bg_rgb_uint8.astype(np.float32) / 255.0
    colored = cm.get_cmap(cmap)(np.clip(heat_01, 0.0, 1.0))[..., :3]
    out = (1 - alpha) * base + alpha * colored
    return np.clip(out, 0, 1)


# ========= NEW: mask utilities (align mask to chosen space) =========
def mask_original_to_model_space(mask_bool_hw, orig_hw, predictor):
    """
    Map a binary mask from original HxW to the model's square (target x target)
    with the same resize + pad used by SAM's image transform.
    """
    H_orig, W_orig = orig_hw
    target = predictor.transform.target_length
    # resize to (new_h,new_w) with nearest
    scale = float(target) / float(max(H_orig, W_orig))
    new_h = int(round(H_orig * scale)); new_w = int(round(W_orig * scale))

    t = torch.as_tensor(mask_bool_hw.astype(np.float32))[None, None]
    resized = F.interpolate(t, size=(new_h, new_w), mode='nearest')[0, 0].cpu().numpy()
    # pad to square (top-left anchored like in predictor)
    sq = np.zeros((target, target), dtype=np.float32)
    sq[:new_h, :new_w] = resized
    return sq  # float 0/1


def overlay_binary_mask(bg_rgb_uint8, mask_01, color=(0, 1, 0), alpha=0.45):
    """
    Overlay a single binary mask (0..1) onto bg image using a solid color.
    Ensures shapes match via nearest resize on mask.
    """
    Hbg, Wbg = bg_rgb_uint8.shape[:2]
    if isinstance(mask_01, torch.Tensor):
        mask_01 = mask_01.detach().cpu().float().numpy()
    if mask_01.shape != (Hbg, Wbg):
        t = torch.as_tensor(mask_01)[None, None].float()
        mask_01 = F.interpolate(t, size=(Hbg, Wbg), mode='nearest')[0, 0].cpu().numpy()

    base = bg_rgb_uint8.astype(np.float32) / 255.0
    mask3 = mask_01[..., None]
    col = np.array(color, dtype=np.float32)[None, None, :]
    out = (1 - alpha * mask3) * base + (alpha * mask3) * col
    return np.clip(out, 0, 1)


# =========================
# i2t argmax (categorical, nearest) + combined overlay
# =========================
@torch.no_grad()
def i2t_argmax_over_prompts(attn_i2t, prompt_indices, head_reduce="mean", hw_feat=None):
    B, Hh, Ni, Nq = attn_i2t.shape
    if head_reduce == "mean":
        a = attn_i2t.mean(dim=1)  # [B,Ni,Nq]
    elif head_reduce == "max":
        a, _ = attn_i2t.max(dim=1)
    else:
        raise ValueError("head_reduce must be 'mean' or 'max'.")
    a = a[:, :, prompt_indices]     # [B,Ni,Nprompts]
    winner = a.argmax(dim=2)        # [B,Ni]
    Hf, Wf = hw_feat
    return winner.view(B, Hf, Wf)[0].detach().cpu().numpy().astype(np.int64)


def categorical_to_model_space(argmax_feat_hw, model_side):
    t = torch.as_tensor(argmax_feat_hw, dtype=torch.int64)[None, None].float()
    up = F.interpolate(t, size=(model_side, model_side), mode='nearest')[0, 0]
    return up.cpu().numpy().astype(np.int64)


def categorical_to_original_space(argmax_feat_hw, orig_hw, predictor):
    H_orig, W_orig = orig_hw
    target = predictor.transform.target_length
    t = torch.as_tensor(argmax_feat_hw, dtype=torch.int64)[None, None].float()
    sq = F.interpolate(t, size=(target, target), mode='nearest')[0, 0]
    scale = float(target) / float(max(H_orig, W_orig))
    new_h = int(round(H_orig * scale)); new_w = int(round(W_orig * scale))
    sq = sq[:new_h, :new_w]
    out = F.interpolate(sq[None, None], size=(H_orig, W_orig), mode='nearest')[0, 0]
    return out.cpu().numpy().astype(np.int64)


def distinct_prompt_colors(n):
    # Prefer tab20/tab10; fall back to HSV for >20
    if n <= 10:
        cmap = plt.get_cmap('tab10')
        cols = [tuple(cmap(i)[:3]) for i in range(n)]
    elif n <= 20:
        cmap = plt.get_cmap('tab20')
        cols = [tuple(cmap(i)[:3]) for i in range(n)]
    else:
        import colorsys
        cols = [colorsys.hsv_to_rgb((i / n) % 1.0, 0.65, 0.95) for i in range(n)]
    return cols


def overlay_categorical_rgb(bg_rgb_uint8, labels_hw, colors_rgb, alpha=0.45):
    """
    Composite categorical labels (0..K-1) over an RGB background using class colors.
    """
    base = bg_rgb_uint8.astype(np.float32) / 255.0
    out = base.copy()
    H, W = labels_hw.shape
    for i, col in enumerate(colors_rgb):
        mask = (labels_hw == i).astype(np.float32)
        if mask.sum() == 0:
            continue
        mask3 = mask[..., None]
        col_arr = np.array(col, dtype=np.float32)[None, None, :]
        out = (1 - alpha * mask3) * out + (alpha * mask3) * col_arr
    return np.clip(out, 0, 1)


# =========================
# Main
# =========================
def run(args):
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    img_rgb = load_image_rgb(args.image)
    H_img, W_img = img_rgb.shape[:2]

    with open(args.prompts, "r") as f:
        prom = json.load(f)

    pos_pts = np.array(prom.get("positive_points", []), dtype=np.float32) if prom.get("positive_points") else np.zeros((0, 2), np.float32)
    neg_pts = np.array(prom.get("negative_points", []), dtype=np.float32) if prom.get("negative_points") else np.zeros((0, 2), np.float32)
    all_pts = np.concatenate([pos_pts, neg_pts], axis=0) if (len(pos_pts) + len(neg_pts)) > 0 else np.zeros((0, 2), np.float32)
    all_lbl = np.array([1] * len(pos_pts) + [0] * len(neg_pts), dtype=np.int32)

    if all_pts.shape[0] == 0:
        raise ValueError("No prompts found in JSON (need positive_points and/or negative_points).")

    # Build SAM
    sam = sam_model_registry[args.model_type](checkpoint=args.checkpoint)
    sam.to(device).eval()
    predictor = SamPredictor(sam)

    # Set image (predictor caches transforms/features)
    predictor.set_image(img_rgb)

    # Hooks
    two_way = sam.mask_decoder.transformer
    handles_block, cache = attach_cross_attn_captures(two_way, block_idx=args.block_idx)
    handles_final = attach_final_t2i_hook(two_way, cache)

    # Run SAM once (now with multimask_output=True to get all 3 masks + IoU scores)
    with torch.inference_mode():
        masks, scores, lowres = predictor.predict(
            point_coords=all_pts,
            point_labels=all_lbl,
            multimask_output=True,
        )

    remove_hooks(handles_block + handles_final)

    # Feature grid size
    if predictor.features is None:
        raise RuntimeError("No features cached in predictor.")
    _, Cfeat, Hf, Wf = predictor.features.shape
    hw_feat = (Hf, Wf)

    for k in ['t2i_attn', 'i2t_attn', 'final_t2i_attn']:
        if k not in cache:
            raise RuntimeError(f"Failed to capture attention '{k}'. Check block index/SAM version.")

    # Token layout: [IoU(1), mask_tokens(num_mask_tokens), ...user prompt tokens...]
    num_mask_tokens = getattr(sam.mask_decoder, "num_mask_tokens", 3)
    prompt_offset = 1 + num_mask_tokens
    N_prompts = all_pts.shape[0]
    prompt_indices = list(range(prompt_offset, prompt_offset + N_prompts))

    # Visualization space
    space = args.space.lower()
    assert space in ("original", "model")
    if space == "original":
        bg_img = img_rgb
        pts_vis = all_pts.copy()
        H_vis, W_vis = H_img, W_img
        to_space = lambda feat: feature_heatmap_to_original_space(feat, (H_img, W_img), predictor)
        cat_to_space = lambda argmax_hw: categorical_to_original_space(argmax_hw, (H_img, W_img), predictor)
        mask_to_space = lambda m: m.astype(np.float32)  # masks are already original size
    else:
        target = predictor.transform.target_length
        bg_img = apply_model_transform_image(predictor, img_rgb)
        pts_vis = apply_model_transform_points(predictor, all_pts.copy(), (H_img, W_img))
        H_vis, W_vis = target, target
        to_space = lambda feat: feature_heatmap_to_model_space(feat, target)
        cat_to_space = lambda argmax_hw: categorical_to_model_space(argmax_hw, target)
        mask_to_space = lambda m: mask_original_to_model_space(m, (H_img, W_img), predictor)

    # ========= MAIN: per-prompt figure (4 columns, with colorbars) =========
    N = N_prompts
    ncols = 4
    fig_h = max(2.8, 2.2 * N)
    fig, axes = plt.subplots(N, ncols, figsize=(18, fig_h), squeeze=False)
    blk_title = f"(block {args.block_idx})"
    norm01 = Normalize(vmin=0.0, vmax=1.0)

    # ARGMAX over prompts (feature resolution) and resize categorically (nearest)
    argmax_idx_feat = i2t_argmax_over_prompts(cache['i2t_attn'], prompt_indices, head_reduce=args.head_reduce, hw_feat=hw_feat)
    argmax_rs = cat_to_space(argmax_idx_feat)

    for i in range(N):
        tok_idx = prompt_indices[i]

        hm_t2i_blk   = prompt_to_image_heatmap(cache['t2i_attn'],      tok_idx, args.head_reduce, hw_feat)[0, 0]
        hm_i2t_blk   = image_to_prompt_heatmap(cache['i2t_attn'],      tok_idx, args.head_reduce, hw_feat)[0, 0]
        hm_t2i_final = prompt_to_image_heatmap(cache['final_t2i_attn'],tok_idx, args.head_reduce, hw_feat)[0, 0]

        hm_t2i_blk_rs   = to_space(hm_t2i_blk)
        hm_i2t_blk_rs   = to_space(hm_i2t_blk)
        hm_t2i_final_rs = to_space(hm_t2i_final)

        bin_rs = (argmax_rs == i).astype(np.float32)

        ov_t2i_blk   = overlay_heatmap(bg_img, hm_t2i_blk_rs,   alpha=args.alpha, cmap=args.cmap)
        ov_i2t_blk   = overlay_heatmap(bg_img, hm_i2t_blk_rs,   alpha=args.alpha, cmap=args.cmap)
        ov_t2i_final = overlay_heatmap(bg_img, hm_t2i_final_rs, alpha=args.alpha, cmap=args.cmap)

        cmap_arg = 'Greens' if (i < len(all_lbl) and all_lbl[i] == 1) else 'Reds'
        ov_arg = overlay_heatmap(bg_img, bin_rs, alpha=args.alpha, cmap=cmap_arg)

        x, y = float(pts_vis[i, 0]), float(pts_vis[i, 1])
        color = 'lime' if all_lbl[i] == 1 else 'red'
        label_str = '(+)' if all_lbl[i] == 1 else '(-)'

        ax = axes[i, 0]
        ax.imshow(ov_t2i_blk); ax.scatter([x],[y], c=color, s=36, marker='o', edgecolor='black', linewidths=0.6)
        ax.set_title(f'Prompt→Image {blk_title} {label_str} idx={i}', fontsize=10); ax.set_axis_off()
        sm1 = cm.ScalarMappable(norm=norm01, cmap=args.cmap); sm1.set_array(hm_t2i_blk_rs)
        fig.colorbar(sm1, ax=ax, fraction=0.035, pad=0.02).set_label('attention', fontsize=8)

        ax = axes[i, 1]
        ax.imshow(ov_i2t_blk); ax.scatter([x],[y], c=color, s=36, marker='o', edgecolor='black', linewidths=0.6)
        ax.set_title(f'Image→Prompt {blk_title} {label_str} idx={i}', fontsize=10); ax.set_axis_off()
        sm2 = cm.ScalarMappable(norm=norm01, cmap=args.cmap); sm2.set_array(hm_i2t_blk_rs)
        fig.colorbar(sm2, ax=ax, fraction=0.035, pad=0.02).set_label('attention', fontsize=8)

        ax = axes[i, 2]
        ax.imshow(ov_t2i_final); ax.scatter([x],[y], c=color, s=36, marker='o', edgecolor='black', linewidths=0.6)
        ax.set_title(f'Final Prompt→Image {label_str} idx={i}', fontsize=10); ax.set_axis_off()
        sm3 = cm.ScalarMappable(norm=norm01, cmap=args.cmap); sm3.set_array(hm_t2i_final_rs)
        fig.colorbar(sm3, ax=ax, fraction=0.035, pad=0.02).set_label('attention', fontsize=8)

        ax = axes[i, 3]
        ax.imshow(ov_arg); ax.scatter([x],[y], c=color, s=36, marker='o', edgecolor='black', linewidths=0.6)
        ax.set_title(f'i2t argmax = this prompt {label_str} idx={i}', fontsize=10); ax.set_axis_off()
        sm4 = cm.ScalarMappable(norm=norm01, cmap=cmap_arg); sm4.set_array(bin_rs)
        fig.colorbar(sm4, ax=ax, fraction=0.035, pad=0.02).set_label('argmax mask (0/1)', fontsize=8)

    plt.tight_layout()
    out_path = Path(args.out); out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=200, bbox_inches='tight'); plt.close(fig)
    print(f"[OK] Saved prompts figure: {out_path}")

    # ========= LEARNED TOKENS figure =========
    learned_names = ["iou"] + [f"mask{m}" for m in range(num_mask_tokens)]
    learned_indices = list(range(0, 1 + num_mask_tokens))
    M = len(learned_indices)
    fig_l, axes_l = plt.subplots(M, 3, figsize=(14, max(2.8, 2.2 * M)), squeeze=False)
    for j in range(M):
        tok_idx = learned_indices[j]; name = learned_names[j]
        hm_t2i_blk = prompt_to_image_heatmap(cache['t2i_attn'], tok_idx, args.head_reduce, hw_feat)[0, 0]
        hm_i2t_blk = image_to_prompt_heatmap(cache['i2t_attn'], tok_idx, args.head_reduce, hw_feat)[0, 0]
        hm_t2i_final = prompt_to_image_heatmap(cache['final_t2i_attn'], tok_idx, args.head_reduce, hw_feat)[0, 0]
        hm_t2i_blk_rs, hm_i2t_blk_rs, hm_t2i_final_rs = to_space(hm_t2i_blk), to_space(hm_i2t_blk), to_space(hm_t2i_final)
        ov1, ov2, ov3 = overlay_heatmap(bg_img, hm_t2i_blk_rs, args.alpha, args.cmap), overlay_heatmap(bg_img, hm_i2t_blk_rs, args.alpha, args.cmap), overlay_heatmap(bg_img, hm_t2i_final_rs, args.alpha, args.cmap)
        for k, (ax, ov, title, hm) in enumerate([
            (axes_l[j,0], ov1, f'[LEARNED:{name}] Prompt→Image {blk_title}', hm_t2i_blk_rs),
            (axes_l[j,1], ov2, f'[LEARNED:{name}] Image→Prompt {blk_title}', hm_i2t_blk_rs),
            (axes_l[j,2], ov3, f'[LEARNED:{name}] Final Prompt→Image', hm_t2i_final_rs),
        ]):
            ax.imshow(ov); ax.set_title(title, fontsize=10); ax.set_axis_off()
            sm = cm.ScalarMappable(norm=Normalize(0,1), cmap=args.cmap); sm.set_array(hm)
            fig_l.colorbar(sm, ax=ax, fraction=0.035, pad=0.02).set_label('attention', fontsize=8)
    plt.tight_layout()
    out_learned = Path(args.out_learned); out_learned.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_learned, dpi=200, bbox_inches='tight'); plt.close(fig_l)
    print(f"[OK] Saved learned-tokens figure: {out_learned}")

    # ========= Combined i2t-argmax ownership map (single plot) =========
    colors = distinct_prompt_colors(N)
    combined_overlay = overlay_categorical_rgb(bg_img, argmax_rs, colors, alpha=args.alpha)
    fig_c, axc = plt.subplots(1, 1, figsize=(12, 12 * (H_vis / W_vis)))
    axc.imshow(combined_overlay)
    handles = []
    for i in range(N):
        x, y = float(pts_vis[i, 0]), float(pts_vis[i, 1])
        axc.scatter([x], [y], c=[colors[i]], s=50, marker='o', edgecolor='black', linewidths=0.8)
        lab = f"P{i} ({'+' if all_lbl[i]==1 else '-'})"
        handles.append(Patch(facecolor=colors[i], edgecolor='black', label=lab))
    axc.legend(handles=handles, loc='upper right', fontsize=9, frameon=True, title="Prompts")
    axc.set_title("Image→Prompt ARGMAX (combined ownership map)", fontsize=12)
    axc.set_axis_off()
    out_i2t_all = Path(args.out_i2t_all); out_i2t_all.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_i2t_all, dpi=200, bbox_inches='tight'); plt.close(fig_c)
    print(f"[OK] Saved combined i2t-argmax figure: {out_i2t_all}")

    # ========= NEW: Multimask overlays with IoU scores =========
    # masks: (M, H, W) in ORIGINAL space; scores: (M,)
    M_masks = masks.shape[0]
    mask_colors = plt.get_cmap('tab10')
    fig_m, axes_m = plt.subplots(1, M_masks, figsize=(5 * M_masks, 5 * (H_vis / W_vis)), squeeze=False)

    for m in range(M_masks):
        mask_orig = masks[m].astype(np.float32)  # original space
        mask_rs = mask_to_space(mask_orig)       # map to chosen space
        col = tuple(mask_colors(m % 10)[:3])

        ov_mask = overlay_binary_mask(bg_img, mask_rs, color=col, alpha=args.alpha)

        axm = axes_m[0, m]
        axm.imshow(ov_mask)

        # draw prompts for context
        for i in range(N):
            x, y = float(pts_vis[i, 0]), float(pts_vis[i, 1])
            dot_col = 'lime' if all_lbl[i] == 1 else 'red'
            axm.scatter([x], [y], c=dot_col, s=36, marker='o', edgecolor='black', linewidths=0.6)

        axm.set_title(f"Mask {m} — IoU {float(scores[m]):.3f}", fontsize=11)
        axm.set_axis_off()

    plt.tight_layout()
    out_masks = Path(args.out_masks); out_masks.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_masks, dpi=200, bbox_inches='tight'); plt.close(fig_m)
    print(f"[OK] Saved multimask figure: {out_masks}")


def parse_args():
    ap = argparse.ArgumentParser(description="SAM cross-attn viz with per-prompt panels, learned tokens, combined i2t-argmax, and multimask overlays.")
    ap.add_argument("--image", required=True, help="Path to input image")
    ap.add_argument("--prompts", required=True, help="JSON with positive_points/negative_points (coords in ORIGINAL image space)")
    ap.add_argument("--checkpoint", required=True, help="Path to SAM checkpoint (e.g., sam_vit_h_4b8939.pth)")
    ap.add_argument("--model-type", default="vit_h", choices=["vit_h", "vit_l", "vit_b"], help="SAM model type")
    ap.add_argument("--block-idx", type=int, default=-1, help="Transformer block index to visualize (-1 = last)")
    ap.add_argument("--head-reduce", default="mean", choices=["mean", "max"], help="Aggregate across heads")
    ap.add_argument("--alpha", type=float, default=0.6, help="Overlay alpha")
    ap.add_argument("--cmap", default="jet", help="Matplotlib colormap for attention overlays")
    ap.add_argument("--space", default="original", choices=["original", "model"], help="Overlay space")
    ap.add_argument("--out", default="cross_attn_prompts.png", help="Output PNG for user prompts figure")
    ap.add_argument("--out-learned", default="cross_attn_learned.png", help="Output PNG for learned tokens figure")
    ap.add_argument("--out-i2t-all", default="cross_attn_i2t_all.png", help="Output PNG for combined i2t-argmax map")
    ap.add_argument("--out-masks", default="sam_multimasks.png", help="Output PNG for multimask overlays")
    ap.add_argument("--cpu", action="store_true", help="Force CPU")
    return ap.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(args)
