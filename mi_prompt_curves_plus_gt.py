# #!/usr/bin/env python3
# """
# Mutual Information vs Prompts in SAM (MINE + InfoNCE) + GT faithfulness

# Adds:
#   - --gt-mask path/to/mask.(png|jpg|...)  (any nonzero = foreground; resized to image)
#   - IoU and Dice vs GT per prompt step
#   - MI wrt GT: I(Z_l ; Y_gt) using MINE + InfoNCE
#   - New plots/CSVs that include GT-based MI and task metrics

# Usage (example):
#   python mi_prompt_curves_plus_gt.py \
#     --image /path/to/image.jpg \
#     --prompts /path/to/prompts.json \
#     --checkpoint /path/to/sam_vit_h_4b8939.pth \
#     --model-type vit_h \
#     --out-dir /tmp/mi_prompts \
#     --gt-mask /path/to/gt.png \
#     --mine-steps 200 --infonce-steps 200 --seeds 3
# """

# import argparse
# from pathlib import Path
# import json
# import csv
# import numpy as np
# from PIL import Image
# import matplotlib.pyplot as plt

# import torch
# import torch.nn as nn
# import torch.nn.functional as F

# # pip install 'git+https://github.com/facebookresearch/segment-anything.git'
# from segment_anything import sam_model_registry, SamPredictor


# # ----------------- IO & utils -----------------
# def ensure_dir(p):
#     Path(p).mkdir(parents=True, exist_ok=True)

# def load_image_rgb(path):
#     return np.asarray(Image.open(path).convert("RGB"))

# def load_binary_mask(path, target_hw):
#     """Load mask image, make binary (nonzero=1), resize to target (H,W) w/ nearest."""
#     Ht, Wt = target_hw
#     m = Image.open(path).convert("L")
#     if m.size != (Wt, Ht):
#         m = m.resize((Wt, Ht), resample=Image.NEAREST)
#     m = np.asarray(m)
#     m = (m != 0).astype(np.uint8)
#     return m

# def to_torch(x, device):
#     return torch.as_tensor(x, dtype=torch.float32, device=device)

# def flatten_tokens(feats_bchw):
#     # feats: [1, C, Hf, Wf] -> [N_tokens, C]
#     _, C, H, W = feats_bchw.shape
#     return feats_bchw[0].permute(1, 2, 0).reshape(H * W, C).contiguous()

# def downsample_to_tokens(mask_hw_np, Ht, Wt):
#     """Area pool a (H,W) float/binary map to token grid (Ht,Wt), return [N,1]."""
#     t = torch.from_numpy(mask_hw_np.astype(np.float32)).float()[None, None]
#     ds = F.interpolate(t, size=(Ht, Wt), mode="area")[0, 0]
#     return ds.view(-1, 1).cpu().numpy()

# def maybe_add_noise(tensor, std):
#     if std <= 0:
#         return tensor
#     return tensor + std * torch.randn_like(tensor)

# def smooth01(y, eps):
#     """Label-smoothing: map {0,1} -> [eps, 1-eps]. Works for any 0..1 float array."""
#     return (1 - 2 * eps) * y + eps


# # ----------------- Hook Two-Way keys per block -----------------
# def hook_keys(two_way_transformer):
#     """
#     Forward-hook each TwoWayAttentionBlock to capture its output "keys"
#     (image-side tokens after that block). Returns (handles, cache).
#     """
#     cache = {"keys_per_block": []}
#     handles = []

#     def fwd_hook(module, args, out):
#         # block.forward returns (queries, keys)
#         q, k = out
#         cache["keys_per_block"].append(k.detach())

#     for blk in two_way_transformer.layers:
#         handles.append(blk.register_forward_hook(fwd_hook))
#     return handles, cache

# def remove_hooks(handles):
#     for h in handles:
#         h.remove()


# # ----------------- MINE (Donsker–Varadhan) -----------------
# class MINECritic(nn.Module):
#     def __init__(self, x_dim, z_dim, hidden=256):
#         super().__init__()
#         in_dim = x_dim + z_dim
#         self.net = nn.Sequential(
#             nn.Linear(in_dim, hidden), nn.ReLU(inplace=True),
#             nn.Linear(hidden, hidden), nn.ReLU(inplace=True),
#             nn.Linear(hidden, 1),
#         )
#     def forward(self, x, z):
#         return self.net(torch.cat([x, z], dim=-1))

# @torch.no_grad()
# def _shuffle_rows(z):
#     idx = torch.randperm(z.shape[0], device=z.device)
#     return z[idx]

# def estimate_mine_lower_bound(
#     X, Z, steps=200, batch=1024, lr=1e-3, hidden=256, seeds=1, token_noise_std=0.0
# ):
#     """
#     Average MINE lower bound over multiple seeds for stability.
#     X: [N, Dx], Z: [N, Dz]  (tokens as samples)
#     returns: float (avg over seeds)
#     """
#     vals = []
#     N, Dx = X.shape
#     Dz = Z.shape[1]

#     for s in range(seeds):
#         g = torch.Generator(device=X.device).manual_seed(1234 + s)

#         Tnet = MINECritic(Dx, Dz, hidden=hidden).to(X.device)
#         opt = torch.optim.Adam(Tnet.parameters(), lr=lr)

#         # Normalize each view; optional small noise
#         Xn = (X - X.mean(0, keepdim=True)) / (X.std(0, keepdim=True) + 1e-6)
#         Zn = (Z - Z.mean(0, keepdim=True)) / (Z.std(0, keepdim=True) + 1e-6)
#         Xn = maybe_add_noise(Xn, token_noise_std)
#         Zn = maybe_add_noise(Zn, token_noise_std)

#         for _ in range(steps):
#             bsz = min(batch, N)
#             idx = torch.randint(0, N, (bsz,), generator=g, device=X.device)
#             x_b = Xn[idx]
#             z_b = Zn[idx]
#             z_shuf = _shuffle_rows(z_b)

#             T_joint = Tnet(x_b, z_b).mean()
#             T_marg = torch.logsumexp(Tnet(x_b, z_shuf), dim=0) - np.log(bsz)
#             loss = -(T_joint - T_marg)

#             opt.zero_grad(set_to_none=True)
#             loss.backward()
#             opt.step()

#         with torch.no_grad():
#             # chunked eval for memory
#             def chunked_eval(A, B, chunk=4096):
#                 outs = []
#                 for i in range(0, N, chunk):
#                     outs.append(Tnet(A[i:i+chunk], B[i:i+chunk]))
#                 return torch.cat(outs, 0)

#             Tj = chunked_eval(Xn, Zn).mean()
#             Tm_vals = []
#             for i in range(0, N, 4096):
#                 zsh = _shuffle_rows(Zn[i:i+4096])
#                 Tm_vals.append(Tnet(Xn[i:i+4096], zsh))
#             Tm = torch.logsumexp(torch.cat(Tm_vals, 0), dim=0) - np.log(N)
#             lb = (Tj - Tm).item()
#         vals.append(lb)

#     return float(np.mean(vals))


# # ----------------- InfoNCE (contrastive) -----------------
# class ProjectionHead(nn.Module):
#     def __init__(self, in_dim, proj_dim=128):
#         super().__init__()
#         self.net = nn.Sequential(
#             nn.Linear(in_dim, proj_dim),
#             nn.ReLU(inplace=True),
#             nn.Linear(proj_dim, proj_dim),
#         )
#     def forward(self, x):
#         return self.net(x)

# def estimate_infonce_lower_bound(
#     X, Z, steps=200, batch=1024, lr=1e-3, proj_dim=128, temperature=0.1, seeds=1, token_noise_std=0.0
# ):
#     """
#     Train projections f,g to maximize InfoNCE on aligned pairs (tokens as samples).
#     Returns the average (over seeds) of -CE loss (nats), i.e., the InfoNCE lower bound.
#     """
#     vals = []
#     N, Dx = X.shape
#     Dz = Z.shape[1]

#     for s in range(seeds):
#         g = torch.Generator(device=X.device).manual_seed(5678 + s)

#         fX = ProjectionHead(Dx, proj_dim=proj_dim).to(X.device)
#         fZ = ProjectionHead(Dz, proj_dim=proj_dim).to(X.device)
#         opt = torch.optim.Adam(list(fX.parameters()) + list(fZ.parameters()), lr=lr)

#         Xn = (X - X.mean(0, keepdim=True)) / (X.std(0, keepdim=True) + 1e-6)
#         Zn = (Z - Z.mean(0, keepdim=True)) / (Z.std(0, keepdim=True) + 1e-6)
#         Xn = maybe_add_noise(Xn, token_noise_std)
#         Zn = maybe_add_noise(Zn, token_noise_std)

#         criterion = nn.CrossEntropyLoss()

#         for _ in range(steps):
#             bsz = min(batch, N)
#             idx = torch.randint(0, N, (bsz,), generator=g, device=X.device)
#             x_b = fX(Xn[idx])            # [B, Dp]
#             z_b = fZ(Zn[idx])            # [B, Dp]

#             logits = (x_b @ z_b.t()) / temperature   # [B, B]
#             targets = torch.arange(bsz, device=X.device)
#             loss = criterion(logits, targets)        # nats

#             opt.zero_grad(set_to_none=True)
#             loss.backward()
#             opt.step()

#         with torch.no_grad():
#             projX, projZ = [], []
#             for i in range(0, N, 4096):
#                 projX.append(fX(Xn[i:i+4096]))
#                 projZ.append(fZ(Zn[i:i+4096]))
#             PX = torch.cat(projX, 0)
#             PZ = torch.cat(projZ, 0)

#             bsz = min(2048, N)
#             idx = torch.randint(0, N, (bsz,), generator=g, device=X.device)
#             x_b = PX[idx]
#             z_b = PZ[idx]
#             logits = (x_b @ z_b.t()) / temperature
#             targets = torch.arange(bsz, device=X.device)
#             ce = nn.CrossEntropyLoss()(logits, targets)
#             lb = (-ce).item()
#         vals.append(lb)

#     return float(np.mean(vals))


# # ----------------- Metrics -----------------
# def iou_and_dice(pred_bin: np.ndarray, gt_bin: np.ndarray):
#     """Both arrays (H,W) in {0,1}."""
#     pred = pred_bin.astype(bool); gt = gt_bin.astype(bool)
#     inter = np.logical_and(pred, gt).sum()
#     union = np.logical_or(pred, gt).sum()
#     iou = inter / (union + 1e-8)
#     dice = (2 * inter) / (pred.sum() + gt.sum() + 1e-8)
#     return float(iou), float(dice)


# # ----------------- Main pipeline -----------------
# def run(args):
#     device = "cuda" if torch.cuda.is_available() and not args.cpu else "cpu"
#     out_dir = Path(args.out_dir); ensure_dir(out_dir)

#     # Load image + prompts
#     img = load_image_rgb(args.image)
#     H, W = img.shape[:2]
#     with open(args.prompts, "r") as f:
#         prom = json.load(f)
#     pos = np.array(prom.get("positive_points", []), dtype=np.float32) if prom.get("positive_points") else np.zeros((0,2), np.float32)
#     neg = np.array(prom.get("negative_points", []), dtype=np.float32) if prom.get("negative_points") else np.zeros((0,2), np.float32)
#     pts = np.concatenate([pos, neg], axis=0)
#     lbl = np.array([1]*len(pos) + [0]*len(neg), dtype=np.int32)
#     if len(pts) == 0:
#         raise ValueError("No prompts provided.")

#     # Optional GT
#     have_gt = args.gt_mask is not None
#     if have_gt:
#         gt_bin = load_binary_mask(args.gt_mask, (H, W)).astype(np.uint8)  # 0/1

#     # Build SAM predictor
#     sam = sam_model_registry[args.model_type](checkpoint=args.checkpoint)
#     sam.to(device).eval()
#     predictor = SamPredictor(sam)
#     predictor.set_image(img)

#     # Static input tokens X from image encoder
#     if predictor.features is None:
#         raise RuntimeError("Predictor has no encoder features.")
#     feats = predictor.features  # [1, C, Hf, Wf]
#     X = flatten_tokens(feats).to(device)     # [N_tokens, Cx]
#     _, Cfeat, Hf, Wf = feats.shape
#     N_tokens = Hf * Wf

#     # Hook keys per block (Z_l)
#     two_way = sam.mask_decoder.transformer
#     handles, cache = hook_keys(two_way)

#     # Containers (we’ll compute both estimators)
#     step_ids = []
#     I_XZ_mine, I_ZY_mine = [], []
#     I_XZ_nce,  I_ZY_nce  = [], []

#     # NEW: MI wrt GT and task metrics
#     I_ZYgt_mine, I_ZYgt_nce = [], []
#     IoU_per_step, Dice_per_step = [], []

#     # Incremental prompts: 1..K
#     for k in range(1, len(pts) + 1):
#         cache["keys_per_block"].clear()

#         with torch.inference_mode():
#             masks, scores, logits = predictor.predict(
#                 point_coords=pts[:k],
#                 point_labels=lbl[:k],
#                 multimask_output=False,  # use best single mask for simplicity
#             )

#         # Collect Z_l from hooks
#         if len(cache["keys_per_block"]) == 0:
#             remove_hooks(handles)
#             raise RuntimeError("Failed to capture Two-Way keys. Check SAM version.")
#         Z_list = [z[0] for z in cache["keys_per_block"]]  # each [N_tokens, Cz]
#         L = len(Z_list)

#         # --- Build Y (soft prob in original space), then pool to token grid ---
#         lowres = torch.as_tensor(logits, device=device)
#         if lowres.ndim == 2:
#             lowres = lowres[None, None, :, :]
#         elif lowres.ndim == 3:
#             if lowres.shape[0] == 1 and lowres.shape[1] == 256:
#                 lowres = lowres[:, None, :, :]
#             elif lowres.shape[0] >= 1 and lowres.shape[1] == 256:
#                 lowres = lowres[:, None, :, :]
#             else:
#                 lowres = lowres[None, None, :, :]
#         elif lowres.ndim == 4:
#             if lowres.shape[1] != 1:
#                 lowres = lowres[:, :1, ...]
#         else:
#             raise RuntimeError(f"Unexpected lowres shape: {tuple(lowres.shape)}")

#         with torch.no_grad():
#             if hasattr(sam, "postprocess_masks"):
#                 prob_orig_t = torch.sigmoid(
#                     sam.postprocess_masks(
#                         lowres, input_size=predictor.input_size, original_size=predictor.original_size
#                     )[0, 0]
#                 )
#             elif hasattr(predictor, "model") and hasattr(predictor.model, "postprocess_masks"):
#                 prob_orig_t = torch.sigmoid(
#                     predictor.model.postprocess_masks(
#                         lowres, input_size=predictor.input_size, original_size=predictor.original_size
#                     )[0, 0]
#                 )
#             else:
#                 # fallback to binary mask from predictor
#                 prob_orig_t = torch.as_tensor(masks[0].astype(np.float32), device=device)

#         prob_orig = prob_orig_t.detach().cpu().numpy()            # (H,W)
#         Y_tok = downsample_to_tokens(prob_orig, Hf, Wf)           # [N,1]
#         Y = to_torch(Y_tok, device)                               # [N,1]

#         # --- If GT provided: IoU/Dice + Y_gt tokens ---
#         if have_gt:
#             pred_bin = (prob_orig >= args.bin_thresh).astype(np.uint8)
#             iou, dice = iou_and_dice(pred_bin, gt_bin)
#             IoU_per_step.append(iou); Dice_per_step.append(dice)

#             # tokens for GT; smooth a bit to avoid degeneracy in MI
#             Ygt_hw = gt_bin.astype(np.float32)
#             Ygt_tok = downsample_to_tokens(Ygt_hw, Hf, Wf)        # [N,1] in [0,1]
#             if args.gt_smooth_eps > 0:
#                 Ygt_tok = smooth01(Ygt_tok, args.gt_smooth_eps)
#             Ygt = to_torch(Ygt_tok, device)
#         else:
#             IoU_per_step.append(np.nan); Dice_per_step.append(np.nan)

#         # Per-block MI (both estimators)
#         mi_xz_m, mi_zy_m = [], []
#         mi_xz_n, mi_zy_n = [], []
#         mi_zygt_m, mi_zygt_n = [], []

#         for l, Zl in enumerate(Z_list):
#             Z = Zl.reshape(N_tokens, -1).to(device)

#             # --- MINE ---
#             mi1 = estimate_mine_lower_bound(
#                 X, Z,
#                 steps=args.mine_steps, batch=args.mine_batch, lr=args.mine_lr,
#                 hidden=args.mine_hidden, seeds=args.seeds, token_noise_std=args.token_noise_std
#             )
#             mi2 = estimate_mine_lower_bound(
#                 Z, Y,
#                 steps=args.mine_steps, batch=args.mine_batch, lr=args.mine_lr,
#                 hidden=args.mine_hidden, seeds=args.seeds, token_noise_std=args.token_noise_std
#             )
#             mi_xz_m.append(mi1); mi_zy_m.append(mi2)

#             # --- InfoNCE ---
#             in1 = estimate_infonce_lower_bound(
#                 X, Z,
#                 steps=args.infonce_steps, batch=args.infonce_batch, lr=args.infonce_lr,
#                 proj_dim=args.infonce_proj, temperature=args.infonce_temp,
#                 seeds=args.seeds, token_noise_std=args.token_noise_std
#             )
#             in2 = estimate_infonce_lower_bound(
#                 Z, Y,
#                 steps=args.infonce_steps, batch=args.infonce_batch, lr=args.infonce_lr,
#                 proj_dim=args.infonce_proj, temperature=args.infonce_temp,
#                 seeds=args.seeds, token_noise_std=args.token_noise_std
#             )
#             mi_xz_n.append(in1); mi_zy_n.append(in2)

#             # --- MI wrt GT (if provided) ---
#             if have_gt:
#                 mi2gt = estimate_mine_lower_bound(
#                     Z, Ygt,
#                     steps=args.mine_steps, batch=args.mine_batch, lr=args.mine_lr,
#                     hidden=args.mine_hidden, seeds=args.seeds, token_noise_std=args.token_noise_std
#                 )
#                 in2gt = estimate_infonce_lower_bound(
#                     Z, Ygt,
#                     steps=args.infonce_steps, batch=args.infonce_batch, lr=args.infonce_lr,
#                     proj_dim=args.infonce_proj, temperature=args.infonce_temp,
#                     seeds=args.seeds, token_noise_std=args.token_noise_std
#                 )
#                 mi_zygt_m.append(mi2gt); mi_zygt_n.append(in2gt)

#         step_ids.append(k)
#         I_XZ_mine.append(mi_xz_m); I_ZY_mine.append(mi_zy_m)
#         I_XZ_nce.append(mi_xz_n);  I_ZY_nce.append(mi_zy_n)
#         if have_gt:
#             I_ZYgt_mine.append(mi_zygt_m); I_ZYgt_nce.append(mi_zygt_n)

#         msg = (f"[step {k}] "
#                f"MINE I(X;Z_last)={mi_xz_m[-1]:.4f}  I(Z_last;Y)={mi_zy_m[-1]:.4f} | "
#                f"NCE I(X;Z_last)={mi_xz_n[-1]:.4f}  I(Z_last;Y)={mi_zy_n[-1]:.4f}")
#         if have_gt:
#             msg += f" | GT: MINE I(Z_last;Ygt)={mi_zygt_m[-1]:.4f} NCE I(Z_last;Ygt)={mi_zygt_n[-1]:.4f}  IoU={iou:.3f} Dice={dice:.3f}"
#         print(msg)

#     remove_hooks(handles)

#     # Convert to arrays
#     I_XZ_mine = np.array(I_XZ_mine)  # [S, L]
#     I_ZY_mine = np.array(I_ZY_mine)
#     I_XZ_nce  = np.array(I_XZ_nce)
#     I_ZY_nce  = np.array(I_ZY_nce)
#     S, L = I_XZ_mine.shape

#     if have_gt:
#         I_ZYgt_mine = np.array(I_ZYgt_mine)  # [S, L]
#         I_ZYgt_nce  = np.array(I_ZYgt_nce)
#         IoU_per_step = np.array(IoU_per_step)
#         Dice_per_step = np.array(Dice_per_step)

#     # ----- VISUALS -----
#     def plot_curves_last_block():
#         fig, ax = plt.subplots(1, 1, figsize=(8.0, 4.6))
#         ax.plot(step_ids, I_XZ_mine[:, -1], label=r"MINE  $I(X;\,Z_{\mathrm{last}})$", color="#1f77b4")
#         ax.plot(step_ids, I_ZY_mine[:, -1], label=r"MINE  $I(Z_{\mathrm{last}};\,Y)$",  color="#ff7f0e")
#         ax.plot(step_ids, I_XZ_nce[:, -1],  '--', label=r"InfoNCE  $I(X;\,Z_{\mathrm{last}})$", color="#2ca02c")
#         ax.plot(step_ids, I_ZY_nce[:, -1],  '--', label=r"InfoNCE  $I(Z_{\mathrm{last}};\,Y)$", color="#d62728")
#         if have_gt:
#             ax.plot(step_ids, I_ZYgt_mine[:, -1], ':', label=r"MINE  $I(Z_{\mathrm{last}};\,Y_{\mathrm{gt}})$", color="#9467bd")
#             ax.plot(step_ids, I_ZYgt_nce[:, -1],  ':', label=r"InfoNCE  $I(Z_{\mathrm{last}};\,Y_{\mathrm{gt}})$", color="#8c564b")
#         ax.set_xlabel("# prompts used"); ax.set_ylabel("Lower bound (nats)")
#         ax.set_title("MI lower bounds vs prompts (last Two-Way block)")
#         ax.grid(True, alpha=0.3); ax.legend(ncol=2, fontsize=9)
#         fig.tight_layout(); fig.savefig(out_dir / "curves_last_block.png", dpi=220); plt.close(fig)

#     def plot_delta_curves_last_block():
#         def delta(v):
#             d = np.diff(v, prepend=v[:1]); d[0] = np.nan; return d
#         fig, ax = plt.subplots(1, 1, figsize=(8.0, 4.6))
#         ax.plot(step_ids, delta(I_XZ_mine[:, -1]), label=r"Δ MINE  $I(X;\,Z_{\mathrm{last}})$", color="#1f77b4")
#         ax.plot(step_ids, delta(I_ZY_mine[:, -1]), label=r"Δ MINE  $I(Z_{\mathrm{last}};\,Y)$",  color="#ff7f0e")
#         ax.plot(step_ids, delta(I_XZ_nce[:, -1]),  '--', label=r"Δ InfoNCE  $I(X;\,Z_{\mathrm{last}})$", color="#2ca02c")
#         ax.plot(step_ids, delta(I_ZY_nce[:, -1]),  '--', label=r"Δ InfoNCE  $I(Z_{\mathrm{last}};\,Y)$", color="#d62728")
#         if have_gt:
#             ax.plot(step_ids, delta(I_ZYgt_mine[:, -1]), ':', label=r"Δ MINE  $I(Z_{\mathrm{last}};\,Y_{\mathrm{gt}})$", color="#9467bd")
#             ax.plot(step_ids, delta(I_ZYgt_nce[:, -1]),  ':', label=r"Δ InfoNCE  $I(Z_{\mathrm{last}};\,Y_{\mathrm{gt}})$", color="#8c564b")
#         ax.set_xlabel("# prompts used"); ax.set_ylabel("Step-to-step change (nats)")
#         ax.set_title("Δ MI vs prompts (last block)")
#         ax.grid(True, alpha=0.3); ax.legend(ncol=2, fontsize=9)
#         fig.tight_layout(); fig.savefig(out_dir / "delta_curves_last_block.png", dpi=220); plt.close(fig)

#     def heatmap(M, title, fname):
#         fig, ax = plt.subplots(1, 1, figsize=(min(12, 1.2*L), min(8, 0.8*S)))
#         im = ax.imshow(M, cmap="viridis", aspect="auto")
#         ax.set_xlabel("Two-Way block index (0..L-1)")
#         ax.set_ylabel("# prompts used"); ax.set_yticks(np.arange(S)); ax.set_yticklabels(step_ids)
#         ax.set_title(title)
#         fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Lower bound (nats)")
#         fig.tight_layout(); fig.savefig(out_dir / fname, dpi=220); plt.close(fig)

#     # Base plots
#     plot_curves_last_block()
#     plot_delta_curves_last_block()
#     heatmap(I_XZ_mine, "MINE  I(X; Z_l) over prompts/blocks", "heatmap_MINE_XZ.png")
#     heatmap(I_ZY_mine, "MINE  I(Z_l; Y) over prompts/blocks",  "heatmap_MINE_ZY.png")
#     heatmap(I_XZ_nce,  "InfoNCE  I(X; Z_l) over prompts/blocks", "heatmap_InfoNCE_XZ.png")
#     heatmap(I_ZY_nce,  "InfoNCE  I(Z_l; Y) over prompts/blocks", "heatmap_InfoNCE_ZY.png")
#     if have_gt:
#         heatmap(I_ZYgt_mine, "MINE  I(Z_l; Y_gt) over prompts/blocks",  "heatmap_MINE_ZYgt.png")
#         heatmap(I_ZYgt_nce,  "InfoNCE  I(Z_l; Y_gt) over prompts/blocks", "heatmap_InfoNCE_ZYgt.png")

#     # NEW: combined last-block figure with IoU & Dice on secondary axis
#     if have_gt:
#         fig, ax1 = plt.subplots(1, 1, figsize=(8.5, 4.8))
#         ax1.plot(step_ids, I_ZY_mine[:, -1], label=r"MINE  $I(Z_{\mathrm{last}};\,Y)$", color="#ff7f0e")
#         ax1.plot(step_ids, I_ZY_nce[:, -1],  '--', label=r"InfoNCE  $I(Z_{\mathrm{last}};\,Y)$", color="#d62728")
#         ax1.plot(step_ids, I_ZYgt_mine[:, -1], ':', label=r"MINE  $I(Z_{\mathrm{last}};\,Y_{\mathrm{gt}})$", color="#9467bd")
#         ax1.plot(step_ids, I_ZYgt_nce[:, -1],  ':', label=r"InfoNCE  $I(Z_{\mathrm{last}};\,Y_{\mathrm{gt}})$", color="#8c564b")
#         ax1.set_xlabel("# prompts used"); ax1.set_ylabel("MI lower bound (nats)")
#         ax1.grid(True, alpha=0.3)

#         ax2 = ax1.twinx()
#         ax2.plot(step_ids, IoU_per_step,  color="black", linewidth=2.0, label="IoU (vs GT)")
#         ax2.plot(step_ids, Dice_per_step, color="gray",  linewidth=2.0, label="Dice (vs GT)")
#         ax2.set_ylabel("IoU / Dice")

#         # Build a joint legend
#         lines1, labels1 = ax1.get_legend_handles_labels()
#         lines2, labels2 = ax2.get_legend_handles_labels()
#         ax1.legend(lines1 + lines2, labels1 + labels2, ncol=2, fontsize=9, loc="best")

#         ax1.set_title("Faithfulness: MI vs GT and task metrics (last Two-Way block)")
#         fig.tight_layout(); fig.savefig(out_dir / "curves_last_block_with_gt_and_metrics.png", dpi=220); plt.close(fig)

#     # ----- CSV dumps -----
#     def dump_csv(M, name):
#         with open(out_dir / f"{name}.csv", "w", newline="") as f:
#             w = csv.writer(f)
#             w.writerow(["step"] + [f"Z_block_{i}" for i in range(L)])
#             for s_idx, s in enumerate(step_ids):
#                 w.writerow([s] + [f"{v:.6f}" for v in M[s_idx]])

#     dump_csv(I_XZ_mine, "mine_XZ")
#     dump_csv(I_ZY_mine, "mine_ZY")
#     dump_csv(I_XZ_nce,  "infonce_XZ")
#     dump_csv(I_ZY_nce,  "infonce_ZY")
#     if have_gt:
#         dump_csv(I_ZYgt_mine, "mine_ZYgt")
#         dump_csv(I_ZYgt_nce,  "infonce_ZYgt")
#         with open(out_dir / "metrics_iou_dice.csv", "w", newline="") as f:
#             w = csv.writer(f); w.writerow(["step", "IoU", "Dice"])
#             for s_idx, s in enumerate(step_ids):
#                 w.writerow([s, f"{IoU_per_step[s_idx]:.6f}", f"{Dice_per_step[s_idx]:.6f}"])

#     print(f"[OK] Wrote outputs to {out_dir.resolve()}")
#     print("  - curves_last_block.png, delta_curves_last_block.png")
#     if have_gt:
#         print("  - curves_last_block_with_gt_and_metrics.png")
#         print("  - heatmap_*_ZYgt.png, metrics_iou_dice.csv")
#     print("  - heatmap_MINE_*.png, heatmap_InfoNCE_*.png")
#     print("  - mine_*.csv, infonce_*.csv")


# def parse_args():
#     ap = argparse.ArgumentParser("Mutual Information vs Prompts in SAM (MINE + InfoNCE) with GT faithfulness")
#     ap.add_argument("--image", required=True)
#     ap.add_argument("--prompts", required=True, help="JSON with positive_points/negative_points (original coords)")
#     ap.add_argument("--checkpoint", required=True)
#     ap.add_argument("--model-type", default="vit_h", choices=["vit_h","vit_l","vit_b"])
#     ap.add_argument("--out-dir", default="mi_out")

#     # Ground-truth mask + settings
#     ap.add_argument("--gt-mask", type=str, default=None, help="Path to GT mask image (nonzero=FG). Optional.")
#     ap.add_argument("--bin-thresh", type=float, default=0.5, help="Threshold on prob map for IoU/Dice.")
#     ap.add_argument("--gt-smooth-eps", type=float, default=0.02, help="Label smoothing for Y_gt token map in MI.")

#     # MINE
#     ap.add_argument("--mine-steps", type=int, default=200)
#     ap.add_argument("--mine-batch", type=int, default=1024)
#     ap.add_argument("--mine-lr", type=float, default=1e-3)
#     ap.add_argument("--mine-hidden", type=int, default=256)

#     # InfoNCE
#     ap.add_argument("--infonce-steps", type=int, default=200)
#     ap.add_argument("--infonce-batch", type=int, default=1024)
#     ap.add_argument("--infonce-lr", type=float, default=1e-3)
#     ap.add_argument("--infonce-proj", type=int, default=128)
#     ap.add_argument("--infonce-temp", type=float, default=0.1)

#     # Stability knobs
#     ap.add_argument("--seeds", type=int, default=1, help="Avg lower bounds over this many seeds")
#     ap.add_argument("--token-noise-std", type=float, default=0.0, help="Optional tiny Gaussian noise on tokens")

#     ap.add_argument("--cpu", action="store_true")
#     return ap.parse_args()


# if __name__ == "__main__":
#     args = parse_args()
#     run(args)
# # ----------------- End of file -----------------

#!/usr/bin/env python3
"""
Mutual Information vs Prompts in SAM (MINE + InfoNCE) + GT faithfulness

Adds:
  - --gt-mask path/to/mask.(png|jpg|...)  (any nonzero = foreground; resized to image)
  - IoU and Dice vs GT per prompt step
  - MI wrt GT: I(Z_l ; Y_gt) using MINE + InfoNCE
  - Plots:
      * curves_last_block.png  — MI curves (last Two-Way block); if GT available, also I(Z_last;Y_gt)
      * delta_curves_last_block.png — Δ MI curves + Δ IoU/Δ Dice (moved here)
      * curves_last_block_with_gt_and_metrics.png — MI + IoU/Dice (no Δ curves)
      * heatmaps for per-block MI (and GT variants if provided)
  - CSVs for all MI matrices + IoU/Dice (with deltas in the CSV)

Usage (example):
  python mi_prompt_curves_plus_gt.py \
    --image /path/to/image.jpg \
    --prompts /path/to/prompts.json \
    --checkpoint /path/to/sam_vit_h_4b8939.pth \
    --model-type vit_h \
    --out-dir /tmp/mi_prompts \
    --gt-mask /path/to/gt.png \
    --mine-steps 200 --infonce-steps 200 --seeds 3
"""

import argparse
from pathlib import Path
import json
import csv
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F

# pip install 'git+https://github.com/facebookresearch/segment-anything.git'
from segment_anything import sam_model_registry, SamPredictor


# ----------------- IO & utils -----------------
def ensure_dir(p):
    Path(p).mkdir(parents=True, exist_ok=True)

def load_image_rgb(path):
    return np.asarray(Image.open(path).convert("RGB"))

def load_binary_mask(path, target_hw):
    """Load mask image, make binary (nonzero=1), resize to target (H,W) w/ nearest."""
    Ht, Wt = target_hw
    m = Image.open(path).convert("L")
    if m.size != (Wt, Ht):
        m = m.resize((Wt, Ht), resample=Image.NEAREST)
    m = np.asarray(m)
    m = (m != 0).astype(np.uint8)
    return m

def to_torch(x, device):
    return torch.as_tensor(x, dtype=torch.float32, device=device)

def flatten_tokens(feats_bchw):
    # feats: [1, C, Hf, Wf] -> [N_tokens, C]
    _, C, H, W = feats_bchw.shape
    return feats_bchw[0].permute(1, 2, 0).reshape(H * W, C).contiguous()

def downsample_to_tokens(mask_hw_np, Ht, Wt):
    """Area pool a (H,W) float/binary map to token grid (Ht,Wt), return [N,1]."""
    t = torch.from_numpy(mask_hw_np.astype(np.float32)).float()[None, None]
    ds = F.interpolate(t, size=(Ht, Wt), mode="area")[0, 0]
    return ds.view(-1, 1).cpu().numpy()

def maybe_add_noise(tensor, std):
    if std <= 0:
        return tensor
    return tensor + std * torch.randn_like(tensor)

def smooth01(y, eps):
    """Label-smoothing: map {0,1} -> [eps, 1-eps]. Works for any 0..1 float array."""
    return (1 - 2 * eps) * y + eps


# ----------------- Hook Two-Way keys per block -----------------
def hook_keys(two_way_transformer):
    """
    Forward-hook each TwoWayAttentionBlock to capture its output "keys"
    (image-side tokens after that block). Returns (handles, cache).
    """
    cache = {"keys_per_block": []}
    handles = []

    def fwd_hook(module, args, out):
        # block.forward returns (queries, keys)
        q, k = out
        cache["keys_per_block"].append(k.detach())

    for blk in two_way_transformer.layers:
        handles.append(blk.register_forward_hook(fwd_hook))
    return handles, cache

def remove_hooks(handles):
    for h in handles:
        h.remove()


# ----------------- MINE (Donsker–Varadhan) -----------------
class MINECritic(nn.Module):
    def __init__(self, x_dim, z_dim, hidden=256):
        super().__init__()
        in_dim = x_dim + z_dim
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(inplace=True),
            nn.Linear(hidden, hidden), nn.ReLU(inplace=True),
            nn.Linear(hidden, 1),
        )
    def forward(self, x, z):
        return self.net(torch.cat([x, z], dim=-1))

@torch.no_grad()
def _shuffle_rows(z):
    idx = torch.randperm(z.shape[0], device=z.device)
    return z[idx]

def estimate_mine_lower_bound(
    X, Z, steps=200, batch=1024, lr=1e-3, hidden=256, seeds=1, token_noise_std=0.0
):
    """
    Average MINE lower bound over multiple seeds for stability.
    X: [N, Dx], Z: [N, Dz]  (tokens as samples)
    returns: float (avg over seeds)
    """
    vals = []
    N, Dx = X.shape
    Dz = Z.shape[1]

    for s in range(seeds):
        g = torch.Generator(device=X.device).manual_seed(1234 + s)

        Tnet = MINECritic(Dx, Dz, hidden=hidden).to(X.device)
        opt = torch.optim.Adam(Tnet.parameters(), lr=lr)

        # Normalize each view; optional small noise
        Xn = (X - X.mean(0, keepdim=True)) / (X.std(0, keepdim=True) + 1e-6)
        Zn = (Z - Z.mean(0, keepdim=True)) / (Z.std(0, keepdim=True) + 1e-6)
        Xn = maybe_add_noise(Xn, token_noise_std)
        Zn = maybe_add_noise(Zn, token_noise_std)

        for _ in range(steps):
            bsz = min(batch, N)
            idx = torch.randint(0, N, (bsz,), generator=g, device=X.device)
            x_b = Xn[idx]
            z_b = Zn[idx]
            z_shuf = _shuffle_rows(z_b)

            T_joint = Tnet(x_b, z_b).mean()
            T_marg = torch.logsumexp(Tnet(x_b, z_shuf), dim=0) - np.log(bsz)
            loss = -(T_joint - T_marg)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

        with torch.no_grad():
            # chunked eval for memory
            def chunked_eval(A, B, chunk=4096):
                outs = []
                for i in range(0, N, chunk):
                    outs.append(Tnet(A[i:i+chunk], B[i:i+chunk]))
                return torch.cat(outs, 0)

            Tj = chunked_eval(Xn, Zn).mean()
            Tm_vals = []
            for i in range(0, N, 4096):
                zsh = _shuffle_rows(Zn[i:i+4096])
                Tm_vals.append(Tnet(Xn[i:i+4096], zsh))
            Tm = torch.logsumexp(torch.cat(Tm_vals, 0), dim=0) - np.log(N)
            lb = (Tj - Tm).item()
        vals.append(lb)

    return float(np.mean(vals))


# ----------------- InfoNCE (contrastive) -----------------
class ProjectionHead(nn.Module):
    def __init__(self, in_dim, proj_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, proj_dim),
            nn.ReLU(inplace=True),
            nn.Linear(proj_dim, proj_dim),
        )
    def forward(self, x):
        return self.net(x)

def estimate_infonce_lower_bound(
    X, Z, steps=200, batch=1024, lr=1e-3, proj_dim=128, temperature=0.1, seeds=1, token_noise_std=0.0
):
    """
    Train projections f,g to maximize InfoNCE on aligned pairs (tokens as samples).
    Returns the average (over seeds) of -CE loss (nats), i.e., the InfoNCE lower bound.
    """
    vals = []
    N, Dx = X.shape
    Dz = Z.shape[1]

    for s in range(seeds):
        g = torch.Generator(device=X.device).manual_seed(5678 + s)

        fX = ProjectionHead(Dx, proj_dim=proj_dim).to(X.device)
        fZ = ProjectionHead(Dz, proj_dim=proj_dim).to(X.device)
        opt = torch.optim.Adam(list(fX.parameters()) + list(fZ.parameters()), lr=lr)

        Xn = (X - X.mean(0, keepdim=True)) / (X.std(0, keepdim=True) + 1e-6)
        Zn = (Z - Z.mean(0, keepdim=True)) / (Z.std(0, keepdim=True) + 1e-6)
        Xn = maybe_add_noise(Xn, token_noise_std)
        Zn = maybe_add_noise(Zn, token_noise_std)

        criterion = nn.CrossEntropyLoss()

        for _ in range(steps):
            bsz = min(batch, N)
            idx = torch.randint(0, N, (bsz,), generator=g, device=X.device)
            x_b = fX(Xn[idx])            # [B, Dp]
            z_b = fZ(Zn[idx])            # [B, Dp]

            logits = (x_b @ z_b.t()) / temperature   # [B, B]
            targets = torch.arange(bsz, device=X.device)
            loss = criterion(logits, targets)        # nats

            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

        with torch.no_grad():
            projX, projZ = [], []
            for i in range(0, N, 4096):
                projX.append(fX(Xn[i:i+4096]))
                projZ.append(fZ(Zn[i:i+4096]))
            PX = torch.cat(projX, 0)
            PZ = torch.cat(projZ, 0)

            bsz = min(2048, N)
            idx = torch.randint(0, N, (bsz,), generator=g, device=X.device)
            x_b = PX[idx]
            z_b = PZ[idx]
            logits = (x_b @ z_b.t()) / temperature
            targets = torch.arange(bsz, device=X.device)
            ce = nn.CrossEntropyLoss()(logits, targets)
            lb = (-ce).item()
        vals.append(lb)

    return float(np.mean(vals))


# ----------------- Metrics -----------------
def iou_and_dice(pred_bin: np.ndarray, gt_bin: np.ndarray):
    """Both arrays (H,W) in {0,1}."""
    pred = pred_bin.astype(bool); gt = gt_bin.astype(bool)
    inter = np.logical_and(pred, gt).sum()
    union = np.logical_or(pred, gt).sum()
    iou = inter / (union + 1e-8)
    dice = (2 * inter) / (pred.sum() + gt.sum() + 1e-8)
    return float(iou), float(dice)


# ----------------- Main pipeline -----------------
def run(args):
    device = "cuda" if torch.cuda.is_available() and not args.cpu else "cpu"
    out_dir = Path(args.out_dir); ensure_dir(out_dir)

    # Load image + prompts
    img = load_image_rgb(args.image)
    H, W = img.shape[:2]
    with open(args.prompts, "r") as f:
        prom = json.load(f)
    pos = np.array(prom.get("positive_points", []), dtype=np.float32) if prom.get("positive_points") else np.zeros((0,2), np.float32)
    neg = np.array(prom.get("negative_points", []), dtype=np.float32) if prom.get("negative_points") else np.zeros((0,2), np.float32)
    pts = np.concatenate([pos, neg], axis=0)
    lbl = np.array([1]*len(pos) + [0]*len(neg), dtype=np.int32)
    if len(pts) == 0:
        raise ValueError("No prompts provided.")

    # Optional GT
    have_gt = args.gt_mask is not None
    if have_gt:
        gt_bin = load_binary_mask(args.gt_mask, (H, W)).astype(np.uint8)  # 0/1

    # Build SAM predictor
    sam = sam_model_registry[args.model_type](checkpoint=args.checkpoint)
    sam.to(device).eval()
    predictor = SamPredictor(sam)
    predictor.set_image(img)

    # Static input tokens X from image encoder
    if predictor.features is None:
        raise RuntimeError("Predictor has no encoder features.")
    feats = predictor.features  # [1, C, Hf, Wf]
    X = flatten_tokens(feats).to(device)     # [N_tokens, Cx]
    _, Cfeat, Hf, Wf = feats.shape
    N_tokens = Hf * Wf

    # Hook keys per block (Z_l)
    two_way = sam.mask_decoder.transformer
    handles, cache = hook_keys(two_way)

    # Containers (we’ll compute both estimators)
    step_ids = []
    I_XZ_mine, I_ZY_mine = [], []
    I_XZ_nce,  I_ZY_nce  = [], []

    # MI wrt GT and task metrics
    I_ZYgt_mine, I_ZYgt_nce = [], []
    IoU_per_step, Dice_per_step = [], []

    # Incremental prompts: 1..K
    for k in range(1, len(pts) + 1):
        cache["keys_per_block"].clear()

        with torch.inference_mode():
            masks, scores, logits = predictor.predict(
                point_coords=pts[:k],
                point_labels=lbl[:k],
                multimask_output=False,  # use best single mask for simplicity
            )

        # Collect Z_l from hooks
        if len(cache["keys_per_block"]) == 0:
            remove_hooks(handles)
            raise RuntimeError("Failed to capture Two-Way keys. Check SAM version.")
        Z_list = [z[0] for z in cache["keys_per_block"]]  # each [N_tokens, Cz]
        L = len(Z_list)

        # --- Build Y (soft prob in original space), then pool to token grid ---
        lowres = torch.as_tensor(logits, device=device)
        if lowres.ndim == 2:
            lowres = lowres[None, None, :, :]
        elif lowres.ndim == 3:
            if lowres.shape[0] == 1 and lowres.shape[1] == 256:
                lowres = lowres[:, None, :, :]
            elif lowres.shape[0] >= 1 and lowres.shape[1] == 256:
                lowres = lowres[:, None, :, :]
            else:
                lowres = lowres[None, None, :, :]
        elif lowres.ndim == 4:
            if lowres.shape[1] != 1:
                lowres = lowres[:, :1, ...]
        else:
            raise RuntimeError(f"Unexpected lowres shape: {tuple(lowres.shape)}")

        with torch.no_grad():
            if hasattr(sam, "postprocess_masks"):
                prob_orig_t = torch.sigmoid(
                    sam.postprocess_masks(
                        lowres, input_size=predictor.input_size, original_size=predictor.original_size
                    )[0, 0]
                )
            elif hasattr(predictor, "model") and hasattr(predictor.model, "postprocess_masks"):
                prob_orig_t = torch.sigmoid(
                    predictor.model.postprocess_masks(
                        lowres, input_size=predictor.input_size, original_size=predictor.original_size
                    )[0, 0]
                )
            else:
                # fallback to binary mask from predictor
                prob_orig_t = torch.as_tensor(masks[0].astype(np.float32), device=device)

        prob_orig = prob_orig_t.detach().cpu().numpy()            # (H,W)
        Y_tok = downsample_to_tokens(prob_orig, Hf, Wf)           # [N,1]
        Y = to_torch(Y_tok, device)                               # [N,1]

        # --- If GT provided: IoU/Dice + Y_gt tokens ---
        if have_gt:
            pred_bin = (prob_orig >= args.bin_thresh).astype(np.uint8)
            iou, dice = iou_and_dice(pred_bin, gt_bin)
            IoU_per_step.append(iou); Dice_per_step.append(dice)

            # tokens for GT; smooth a bit to avoid degeneracy in MI
            Ygt_hw = gt_bin.astype(np.float32)
            Ygt_tok = downsample_to_tokens(Ygt_hw, Hf, Wf)        # [N,1] in [0,1]
            if args.gt_smooth_eps > 0:
                Ygt_tok = smooth01(Ygt_tok, args.gt_smooth_eps)
            Ygt = to_torch(Ygt_tok, device)
        else:
            IoU_per_step.append(np.nan); Dice_per_step.append(np.nan)

        # Per-block MI (both estimators)
        mi_xz_m, mi_zy_m = [], []
        mi_xz_n, mi_zy_n = [], []
        mi_zygt_m, mi_zygt_n = [], []

        for l, Zl in enumerate(Z_list):
            Z = Zl.reshape(N_tokens, -1).to(device)

            # --- MINE ---
            mi1 = estimate_mine_lower_bound(
                X, Z,
                steps=args.mine_steps, batch=args.mine_batch, lr=args.mine_lr,
                hidden=args.mine_hidden, seeds=args.seeds, token_noise_std=args.token_noise_std
            )
            mi2 = estimate_mine_lower_bound(
                Z, Y,
                steps=args.mine_steps, batch=args.mine_batch, lr=args.mine_lr,
                hidden=args.mine_hidden, seeds=args.seeds, token_noise_std=args.token_noise_std
            )
            mi_xz_m.append(mi1); mi_zy_m.append(mi2)

            # --- InfoNCE ---
            in1 = estimate_infonce_lower_bound(
                X, Z,
                steps=args.infonce_steps, batch=args.infonce_batch, lr=args.infonce_lr,
                proj_dim=args.infonce_proj, temperature=args.infonce_temp,
                seeds=args.seeds, token_noise_std=args.token_noise_std
            )
            in2 = estimate_infonce_lower_bound(
                Z, Y,
                steps=args.infonce_steps, batch=args.infonce_batch, lr=args.infonce_lr,
                proj_dim=args.infonce_proj, temperature=args.infonce_temp,
                seeds=args.seeds, token_noise_std=args.token_noise_std
            )
            mi_xz_n.append(in1); mi_zy_n.append(in2)

            # --- MI wrt GT (if provided) ---
            if have_gt:
                mi2gt = estimate_mine_lower_bound(
                    Z, Ygt,
                    steps=args.mine_steps, batch=args.mine_batch, lr=args.mine_lr,
                    hidden=args.mine_hidden, seeds=args.seeds, token_noise_std=args.token_noise_std
                )
                in2gt = estimate_infonce_lower_bound(
                    Z, Ygt,
                    steps=args.infonce_steps, batch=args.infonce_batch, lr=args.infonce_lr,
                    proj_dim=args.infonce_proj, temperature=args.infonce_temp,
                    seeds=args.seeds, token_noise_std=args.token_noise_std
                )
                mi_zygt_m.append(mi2gt); mi_zygt_n.append(in2gt)

        step_ids.append(k)
        I_XZ_mine.append(mi_xz_m); I_ZY_mine.append(mi_zy_m)
        I_XZ_nce.append(mi_xz_n);  I_ZY_nce.append(mi_zy_n)
        if have_gt:
            I_ZYgt_mine.append(mi_zygt_m); I_ZYgt_nce.append(mi_zygt_n)

        msg = (f"[step {k}] "
               f"MINE I(X;Z_last)={mi_xz_m[-1]:.4f}  I(Z_last;Y)={mi_zy_m[-1]:.4f} | "
               f"NCE I(X;Z_last)={mi_xz_n[-1]:.4f}  I(Z_last;Y)={mi_zy_n[-1]:.4f}")
        if have_gt:
            msg += f" | GT: MINE I(Z_last;Ygt)={mi_zygt_m[-1]:.4f} NCE I(Z_last;Ygt)={mi_zygt_n[-1]:.4f}  IoU={iou:.3f} Dice={dice:.3f}"
        print(msg)

    remove_hooks(handles)

    # Convert to arrays
    I_XZ_mine = np.array(I_XZ_mine)  # [S, L]
    I_ZY_mine = np.array(I_ZY_mine)
    I_XZ_nce  = np.array(I_XZ_nce)
    I_ZY_nce  = np.array(I_ZY_nce)
    S, L = I_XZ_mine.shape

    if have_gt:
        I_ZYgt_mine = np.array(I_ZYgt_mine)  # [S, L]
        I_ZYgt_nce  = np.array(I_ZYgt_nce)
        IoU_per_step = np.array(IoU_per_step, dtype=float)
        Dice_per_step = np.array(Dice_per_step, dtype=float)

    # ----- VISUALS -----
    def plot_curves_last_block():
        fig, ax = plt.subplots(1, 1, figsize=(8.0, 4.6))
        ax.plot(step_ids, I_XZ_mine[:, -1], label=r"MINE  $I(X;\,Z_{\mathrm{last}})$", color="#1f77b4")
        ax.plot(step_ids, I_ZY_mine[:, -1], label=r"MINE  $I(Z_{\mathrm{last}};\,Y)$",  color="#ff7f0e")
        ax.plot(step_ids, I_XZ_nce[:, -1],  '--', label=r"InfoNCE  $I(X;\,Z_{\mathrm{last}})$", color="#2ca02c")
        ax.plot(step_ids, I_ZY_nce[:, -1],  '--', label=r"InfoNCE  $I(Z_{\mathrm{last}};\,Y)$", color="#d62728")
        if have_gt:
            ax.plot(step_ids, I_ZYgt_mine[:, -1], ':', label=r"MINE  $I(Z_{\mathrm{last}};\,Y_{\mathrm{gt}})$", color="#9467bd")
            ax.plot(step_ids, I_ZYgt_nce[:, -1],  ':', label=r"InfoNCE  $I(Z_{\mathrm{last}};\,Y_{\mathrm{gt}})$", color="#8c564b")
        ax.set_xlabel("# prompts used"); ax.set_ylabel("Lower bound (nats)")
        ax.set_title("MI lower bounds vs prompts (last Two-Way block)")
        ax.grid(True, alpha=0.3); ax.legend(ncol=2, fontsize=9)
        fig.tight_layout(); fig.savefig(out_dir / "curves_last_block.png", dpi=220); plt.close(fig)

    def plot_delta_curves_last_block():
        def delta(v):
            v = np.asarray(v, dtype=float)
            d = np.diff(v, prepend=v[:1])
            d[0] = np.nan
            return d

        fig, ax1 = plt.subplots(1, 1, figsize=(8.8, 5.0))

        # Left axis: Δ MI (nats)
        # ax1.plot(step_ids, delta(I_XZ_mine[:, -1]), label=r"Δ MINE  $I(X;\,Z_{\mathrm{last}})$", color="#1f77b4")
        ax1.plot(step_ids, delta(I_ZY_mine[:, -1]), label=r"Δ MINE  $I(Z_{\mathrm{last}};\,Y)$",  color="#ff7f0e")
        # ax1.plot(step_ids, delta(I_XZ_nce[:, -1]),  '--', label=r"Δ InfoNCE  $I(X;\,Z_{\mathrm{last}})$", color="#2ca02c")
        ax1.plot(step_ids, delta(I_ZY_nce[:, -1]),  '--', label=r"Δ InfoNCE  $I(Z_{\mathrm{last}};\,Y)$", color="#d62728")
        if have_gt:
            ax1.plot(step_ids, delta(I_ZYgt_mine[:, -1]), ':', label=r"Δ MINE  $I(Z_{\mathrm{last}};\,Y_{\mathrm{gt}})$", color="#9467bd")
            ax1.plot(step_ids, delta(I_ZYgt_nce[:, -1]),  ':', label=r"Δ InfoNCE  $I(Z_{\mathrm{last}};\,Y_{\mathrm{gt}})$", color="#8c564b")
        ax1.set_xlabel("# prompts used")
        ax1.set_ylabel("Δ MI (nats)")
        ax1.grid(True, alpha=0.3)

        # Right axis: Δ IoU / Δ Dice  (moved here)
        if have_gt:
            ax2 = ax1.twinx()
            dIoU  = delta(IoU_per_step)
            dDice = delta(Dice_per_step)
            ax2.plot(step_ids, dIoU,  color="green", linewidth=1.8, label="Δ IoU")
            ax2.plot(step_ids, dDice, color="blue",   linewidth=1.8, label="Δ Dice")
            ax2.set_ylabel("Δ IoU / Δ Dice")

            # Joint legend
            lines1, labels1 = ax1.get_legend_handles_labels()
            lines2, labels2 = ax2.get_legend_handles_labels()
            ax1.legend(lines1 + lines2, labels1 + labels2, ncol=2, fontsize=9, loc="best")
        else:
            ax1.legend(ncol=2, fontsize=9, loc="best")

        ax1.set_title("Δ MI and Δ IoU/Δ Dice vs prompts (last Two-Way block)")
        fig.tight_layout()
        fig.savefig(out_dir / "delta_curves_last_block.png", dpi=220)
        plt.close(fig)

    def heatmap(M, title, fname):
        fig, ax = plt.subplots(1, 1, figsize=(min(12, 1.2*L), min(8, 0.8*S)))
        im = ax.imshow(M, cmap="viridis", aspect="auto")
        ax.set_xlabel("Two-Way block index (0..L-1)")
        ax.set_ylabel("# prompts used"); ax.set_yticks(np.arange(S)); ax.set_yticklabels(step_ids)
        ax.set_title(title)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Lower bound (nats)")
        fig.tight_layout(); fig.savefig(out_dir / fname, dpi=220); plt.close(fig)

    # Base plots
    plot_curves_last_block()
    plot_delta_curves_last_block()
    heatmap(I_XZ_mine, "MINE  I(X; Z_l) over prompts/blocks", "heatmap_MINE_XZ.png")
    heatmap(I_ZY_mine, "MINE  I(Z_l; Y) over prompts/blocks",  "heatmap_MINE_ZY.png")
    heatmap(I_XZ_nce,  "InfoNCE  I(X; Z_l) over prompts/blocks", "heatmap_InfoNCE_XZ.png")
    heatmap(I_ZY_nce,  "InfoNCE  I(Z_l; Y) over prompts/blocks", "heatmap_InfoNCE_ZY.png")
    if have_gt:
        heatmap(I_ZYgt_mine, "MINE  I(Z_l; Y_gt) over prompts/blocks",  "heatmap_MINE_ZYgt.png")
        heatmap(I_ZYgt_nce,  "InfoNCE  I(Z_l; Y_gt) over prompts/blocks", "heatmap_InfoNCE_ZYgt.png")

    # Combined last-block figure: MI (left axis) + IoU/Dice (right axis)  [no Δ curves here]
    if have_gt:
        fig, ax1 = plt.subplots(1, 1, figsize=(8.8, 5.0))
        # Left axis: MI curves
        ax1.plot(step_ids, I_ZY_mine[:, -1], label=r"MINE  $I(Z_{\mathrm{last}};\,Y)$", color="#ff7f0e")
        ax1.plot(step_ids, I_ZY_nce[:, -1],  '--', label=r"InfoNCE  $I(Z_{\mathrm{last}};\,Y)$", color="#d62728")
        ax1.plot(step_ids, I_ZYgt_mine[:, -1], ':', label=r"MINE  $I(Z_{\mathrm{last}};\,Y_{\mathrm{gt}})$", color="#9467bd")
        ax1.plot(step_ids, I_ZYgt_nce[:, -1],  ':', label=r"InfoNCE  $I(Z_{\mathrm{last}};\,Y_{\mathrm{gt}})$", color="#8c564b")
        ax1.set_xlabel("# prompts used"); ax1.set_ylabel("MI lower bound (nats)")
        ax1.grid(True, alpha=0.3)

        # Right axis: IoU/Dice
        ax2 = ax1.twinx()
        ax2.plot(step_ids, IoU_per_step,  color="black", linewidth=2.0, label="IoU")
        ax2.plot(step_ids, Dice_per_step, color="gray",  linewidth=2.0, label="Dice")
        ax2.set_ylabel("IoU / Dice")

        # Joint legend
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, ncol=2, fontsize=9, loc="best")

        ax1.set_title("Faithfulness: MI vs GT and IoU/Dice (last Two-Way block)")
        fig.tight_layout()
        fig.savefig(out_dir / "curves_last_block_with_gt_and_metrics.png", dpi=220)
        plt.close(fig)

    # ----- CSV dumps -----
    def dump_csv(M, name):
        with open(out_dir / f"{name}.csv", "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["step"] + [f"Z_block_{i}" for i in range(L)])
            for s_idx, s in enumerate(step_ids):
                w.writerow([s] + [f"{v:.6f}" for v in M[s_idx]])

    dump_csv(I_XZ_mine, "mine_XZ")
    dump_csv(I_ZY_mine, "mine_ZY")
    dump_csv(I_XZ_nce,  "infonce_XZ")
    dump_csv(I_ZY_nce,  "infonce_ZY")
    if have_gt:
        dump_csv(I_ZYgt_mine, "mine_ZYgt")
        dump_csv(I_ZYgt_nce,  "infonce_ZYgt")
        with open(out_dir / "metrics_iou_dice.csv", "w", newline="") as f:
            w = csv.writer(f); w.writerow(["step", "IoU", "Dice", "Delta_IoU", "Delta_Dice"])
            dIoU  = np.diff(IoU_per_step, prepend=np.nan); dIoU[0] = np.nan
            dDice = np.diff(Dice_per_step, prepend=np.nan); dDice[0] = np.nan
            for s_idx, s in enumerate(step_ids):
                w.writerow([s,
                            f"{IoU_per_step[s_idx]:.6f}",
                            f"{Dice_per_step[s_idx]:.6f}",
                            f"{dIoU[s_idx]:.6f}",
                            f"{dDice[s_idx]:.6f}"])

    print(f"[OK] Wrote outputs to {out_dir.resolve()}")
    print("  - curves_last_block.png, delta_curves_last_block.png")
    if have_gt:
        print("  - curves_last_block_with_gt_and_metrics.png (MI + IoU/Dice)")
        print("  - heatmap_*_ZYgt.png, metrics_iou_dice.csv (with deltas)")
    print("  - heatmap_MINE_*.png, heatmap_InfoNCE_*.png")
    print("  - mine_*.csv, infonce_*.csv")


def parse_args():
    ap = argparse.ArgumentParser("Mutual Information vs Prompts in SAM (MINE + InfoNCE) with GT faithfulness")
    ap.add_argument("--image", required=True)
    ap.add_argument("--prompts", required=True, help="JSON with positive_points/negative_points (original coords)")
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--model-type", default="vit_h", choices=["vit_h","vit_l","vit_b"])
    ap.add_argument("--out-dir", default="mi_out")

    # Ground-truth mask + settings
    ap.add_argument("--gt-mask", type=str, default=None, help="Path to GT mask image (nonzero=FG). Optional.")
    ap.add_argument("--bin-thresh", type=float, default=0.5, help="Threshold on prob map for IoU/Dice.")
    ap.add_argument("--gt-smooth-eps", type=float, default=0.02, help="Label smoothing for Y_gt token map in MI.")

    # MINE
    ap.add_argument("--mine-steps", type=int, default=200)
    ap.add_argument("--mine-batch", type=int, default=1024)
    ap.add_argument("--mine-lr", type=float, default=1e-3)
    ap.add_argument("--mine-hidden", type=int, default=256)

    # InfoNCE
    ap.add_argument("--infonce-steps", type=int, default=200)
    ap.add_argument("--infonce-batch", type=int, default=1024)
    ap.add_argument("--infonce-lr", type=float, default=1e-3)
    ap.add_argument("--infonce-proj", type=int, default=128)
    ap.add_argument("--infonce-temp", type=float, default=0.1)

    # Stability knobs
    ap.add_argument("--seeds", type=int, default=1, help="Avg lower bounds over this many seeds")
    ap.add_argument("--token-noise-std", type=float, default=0.0, help="Optional tiny Gaussian noise on tokens")

    ap.add_argument("--cpu", action="store_true")
    return ap.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(args)
# ----------------- End of file -----------------
