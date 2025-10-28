#!/usr/bin/env python3
"""
Complete Bayesian SAM Head Training Pipeline (fixed)

- Frozen SAM encoder/decoder
- Spatial, prompt-aware correction head (1×1/3×3 convs)
- Correct SAM preprocessing & coordinate scaling
- Prompt rescaling in collate_fn
- Per-prompt noise epsilon in both augmentation & loss
- Optional Laplace on a tiny 2-parameter logit calibrator (stable)

Usage:
    python bayesian_sam_training.py --config config.yaml
    python bayesian_sam_training.py --config config.yaml --skip-laplace
"""

import argparse
import json
import yaml
import random
import numpy as np
from pathlib import Path
from functools import partial
from tqdm import tqdm
import matplotlib.pyplot as plt
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, TensorDataset
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR

from scipy.ndimage import distance_transform_edt
from sklearn.cluster import KMeans
from sklearn.calibration import calibration_curve

# --- SAM ---
from segment_anything import sam_model_registry

# --- Laplace (optional) ---
LAPLACE_AVAILABLE = False
try:
    from laplace import Laplace
    LAPLACE_AVAILABLE = True
    print("Laplace-torch available.")
except Exception:
    print("Laplace-torch not available. Use --skip-laplace or pip install laplace-torch.")
    LAPLACE_AVAILABLE = False

# --- W&B (optional via config flag) ---
try:
    import wandb
    WANDB_OK = True
except Exception:
    WANDB_OK = False


# ============================================================================
# Utilities
# ============================================================================

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def top_up_prompts(mask: np.ndarray, prompts, n_prompts: int):
    """Ensure we always return exactly n_prompts by topping up at random."""
    if len(prompts) >= n_prompts:
        return prompts[:n_prompts]
    need = n_prompts - len(prompts)
    fg = np.argwhere(mask == 1)
    bg = np.argwhere(mask == 0)
    # pick alternately from fg/bg to keep balance
    fb = [1, 0] * ((need + 1) // 2)
    fb = fb[:need]
    for lab in fb:
        pool = fg if lab == 1 else bg
        if len(pool) == 0:
            pool = fg if lab == 0 else bg
            if len(pool) == 0:
                break
        y, x = pool[np.random.randint(0, len(pool))]
        prompts.append({"location": [int(y), int(x)], "label": int(lab)})
    return prompts[:n_prompts]


# ============================================================================
# Prompt policies
# ============================================================================

class PromptPolicyGenerator:
    @staticmethod
    def random_uniform(mask, n_prompts):
        fg = np.argwhere(mask == 1)
        bg = np.argwhere(mask == 0)
        n_fg = np.random.randint(1, n_prompts) if n_prompts > 1 else 1
        n_bg = n_prompts - n_fg
        out = []
        if len(fg) and n_fg > 0:
            idx = np.random.choice(len(fg), min(n_fg, len(fg)), replace=False)
            out += [{"location": fg[i].tolist(), "label": 1} for i in idx]
        if len(bg) and n_bg > 0:
            idx = np.random.choice(len(bg), min(n_bg, len(bg)), replace=False)
            out += [{"location": bg[i].tolist(), "label": 0} for i in idx]
        return top_up_prompts(mask, out, n_prompts)

    @staticmethod
    def boundary_focused(mask, n_prompts, boundary_width=5):
        di = distance_transform_edt(mask)
        do = distance_transform_edt(1 - mask)
        bfg = (di > 0) & (di <= boundary_width)
        bbg = (do > 0) & (do <= boundary_width)
        Pfg = np.argwhere(bfg)
        Pbg = np.argwhere(bbg)
        n_fg = np.random.randint(1, n_prompts) if n_prompts > 1 else 1
        n_bg = n_prompts - n_fg
        out = []
        if len(Pfg) and n_fg > 0:
            idx = np.random.choice(len(Pfg), min(n_fg, len(Pfg)), replace=False)
            out += [{"location": Pfg[i].tolist(), "label": 1} for i in idx]
        if len(Pbg) and n_bg > 0:
            idx = np.random.choice(len(Pbg), min(n_bg, len(Pbg)), replace=False)
            out += [{"location": Pbg[i].tolist(), "label": 0} for i in idx]
        return top_up_prompts(mask, out, n_prompts)

    @staticmethod
    def grid_sampling(mask, n_prompts):
        H, W = mask.shape
        gs = max(1, int(np.sqrt(n_prompts * 2)))
        ys = np.linspace(0, H - 1, gs, dtype=int)
        xs = np.linspace(0, W - 1, gs, dtype=int)
        out = []
        for y in ys:
            for x in xs:
                if len(out) >= n_prompts: break
                out.append({"location": [int(y), int(x)], "label": int(mask[y, x])})
            if len(out) >= n_prompts: break
        return top_up_prompts(mask, out, n_prompts)

    @staticmethod
    def centroid_based(mask, n_prompts):
        fg = np.argwhere(mask == 1)
        bg = np.argwhere(mask == 0)
        n_fg = max(1, n_prompts // 2)
        n_bg = n_prompts - n_fg
        out = []
        if len(fg) > 0:
            k = min(n_fg, len(fg))
            if k > 0:
                km = KMeans(n_clusters=k, random_state=42, n_init=10)
                km.fit(fg)
                out += [{"location": c.astype(int).tolist(), "label": 1} for c in km.cluster_centers_]
        if len(bg) > 0:
            k = min(n_bg, len(bg))
            if k > 0:
                km = KMeans(n_clusters=k, random_state=42, n_init=10)
                km.fit(bg)
                out += [{"location": c.astype(int).tolist(), "label": 0} for c in km.cluster_centers_]
        return top_up_prompts(mask, out, n_prompts)

    @staticmethod
    def uncertainty_sampling(mask, n_prompts):
        dist = distance_transform_edt(mask) + distance_transform_edt(1 - mask)
        u = 1.0 / (dist + 1.0)
        p = (u / u.sum()).reshape(-1)
        idxs = np.random.choice(len(p), size=min(n_prompts, len(p)), replace=False, p=p)
        out = []
        H, W = mask.shape
        for idx in idxs:
            y, x = np.unravel_index(idx, (H, W))
            out.append({"location": [int(y), int(x)], "label": int(mask[y, x])})
        return top_up_prompts(mask, out, n_prompts)

    @staticmethod
    def extremal_points(mask, n_prompts):
        fg = np.argwhere(mask == 1)
        if len(fg) == 0:
            return PromptPolicyGenerator.random_uniform(mask, n_prompts)
        top = fg[fg[:, 0].argmin()]
        bot = fg[fg[:, 0].argmax()]
        lef = fg[fg[:, 1].argmin()]
        rig = fg[fg[:, 1].argmax()]
        out = [{"location": p.astype(int).tolist(), "label": 1} for p in [top, bot, lef, rig]]
        return top_up_prompts(mask, out, n_prompts)

    @classmethod
    def get_all_policies(cls):
        return [
            cls.random_uniform,
            cls.boundary_focused,
            cls.grid_sampling,
            cls.centroid_based,
            cls.uncertainty_sampling,
            cls.extremal_points
        ]


# ============================================================================
# Noise augmentation
# ============================================================================

def apply_noise_augmentation(prompts, mask, noise_rate=0.10, boundary_scale=2.0, bound=3):
    di = distance_transform_edt(mask)
    do = distance_transform_edt(1 - mask)
    dmin = np.minimum(di, do)
    for p in prompts:
        y, x = p['location']
        d = dmin[y, x]
        eps = noise_rate * (boundary_scale if d <= bound else 1.0)
        eps = float(min(eps, 0.5))
        p['eps'] = eps
        p['noisy_label'] = 1 - p['label'] if np.random.rand() < eps else p['label']
    return prompts


# ============================================================================
# Dataset & collate
# ============================================================================

class BayesianSAMDataset(Dataset):
    def __init__(self, config, split='train'):
        self.cfg = config
        self.split = split

        self.valid_samples = None
        if self.cfg['data'].get('valid_samples_json'):
            with open(self.cfg['data']['valid_samples_json']) as f:
                self.valid_samples = json.load(f)

        self.samples = self._collect_samples()

        # split
        np.random.seed(self.cfg['training'].get('seed', 42))
        idxs = np.random.permutation(len(self.samples))
        k = int(len(idxs) * self.cfg['training']['train_split'])
        self.indices = idxs[:k] if split == 'train' else idxs[k:]

        self.policies = PromptPolicyGenerator.get_all_policies()
        self.policy_names = [
            'random_uniform','boundary_focused','grid_sampling',
            'centroid_based','uncertainty_sampling','extremal_points'
        ]
        print(f"{split.upper()}: {len(self.indices)} images, "
              f"{len(self.indices)*self.cfg['training']['num_prompt_policies']} policy-instances")

    def _collect_samples(self):
        img_root = Path(self.cfg['data']['image_dir'])
        msk_root = Path(self.cfg['data']['mask_dir'])
        out = []
        for ds_dir in img_root.iterdir():
            if not ds_dir.is_dir(): continue
            ds = ds_dir.name
            mdir = msk_root / ds
            if not mdir.exists(): continue
            valid = set(self.valid_samples.get(ds, [])) if self.valid_samples and ds in self.valid_samples else None
            for imgp in ds_dir.glob("*.jpg"):
                name = imgp.stem
                if valid and name not in valid: continue
                mpath = mdir / f"{name}.npy"
                if mpath.exists():
                    out.append({'image_path': str(imgp), 'mask_path': str(mpath), 'dataset': ds, 'name': name})
        return out

    def __len__(self):
        return len(self.indices) * self.cfg['training']['num_prompt_policies']

    def __getitem__(self, i):
        n_pol = self.cfg['training']['num_prompt_policies']
        img_idx = self.indices[i // n_pol]
        pol_idx = i % n_pol
        s = self.samples[img_idx]

        image = Image.open(s['image_path']).convert('RGB')
        mask = (np.load(s['mask_path']) > 0.5).astype(np.uint8)

        n_prompts = np.random.randint(self.cfg['training']['prompts_per_policy'][0],
                                      self.cfg['training']['prompts_per_policy'][1] + 1)
        n_prompts = min(n_prompts, self.cfg['sam']['k_safe'])

        prompts = self.policies[pol_idx](mask, n_prompts)
        prompts = apply_noise_augmentation(
            prompts, mask,
            noise_rate=self.cfg['training']['noise_rate'],
            boundary_scale=self.cfg['training']['boundary_noise_scale']
        )
        return {
            'image': image,
            'mask': mask,
            'prompts': prompts,
            'policy': self.policy_names[pol_idx],
            'meta': s
        }


def collate_fn(batch, image_size=512):
    """Resize images & masks to image_size and rescale prompt coords to that space."""
    images, masks, prompts_out, metas = [], [], [], []
    for b in batch:
        img_np = np.array(b['image'])
        H0, W0 = b['mask'].shape
        sy, sx = image_size / H0, image_size / W0

        # image
        img = torch.from_numpy(img_np).permute(2, 0, 1).float() / 255.0
        img = F.interpolate(img[None], size=(image_size, image_size), mode='bilinear', align_corners=False)[0]
        images.append(img)

        # mask
        m = torch.from_numpy(b['mask']).float()
        m = F.interpolate(m[None, None], size=(image_size, image_size), mode='nearest')[0, 0]
        masks.append(m)

        # prompts -> resized space
        pr = []
        for p in b['prompts']:
            y, x = p['location']
            pr.append({**p, 'location': [int(round(y * sy)), int(round(x * sx))]})
        prompts_out.append(pr)

        metas.append(b['meta'])

    return {
        'images': torch.stack(images),
        'masks': torch.stack(masks),
        'prompts': prompts_out,
        'meta': metas
    }


# ============================================================================
# Model
# ============================================================================

class SpatialPromptHead(nn.Module):
    """Convolutional correction head over concatenated feature maps."""
    def __init__(self, in_ch: int, channels=(128, 64), dropout=0.1):
        super().__init__()
        layers = []
        prev = in_ch
        for ch in channels:
            layers += [nn.Conv2d(prev, ch, 3, padding=1), nn.ReLU(inplace=True), nn.Dropout2d(dropout)]
            prev = ch
        layers += [nn.Conv2d(prev, 1, 1)]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class BayesianSAMHead(nn.Module):
    """Frozen SAM + spatial prompt-aware correction (added to SAM logits)."""
    def __init__(self, config):
        super().__init__()
        self.cfg = config
        self.sam = sam_model_registry[self.cfg['sam']['model_type']](checkpoint=self.cfg['sam']['checkpoint'])
        for p in self.sam.parameters(): p.requires_grad_(False)
        self.sam.eval()

        self.img_size = self.sam.image_encoder.img_size  # typically 1024
        self.embed_dim = 256  # SAM ViT encoder channels

        # SAM pixel stats
        self.register_buffer('pixel_mean', torch.tensor(self.sam.pixel_mean).view(1, 3, 1, 1), persistent=False)
        self.register_buffer('pixel_std',  torch.tensor(self.sam.pixel_std).view(1, 3, 1, 1),  persistent=False)

        # Concatenate (image_embedding 256) + (dense 256) + (prompt-agg 256) -> 768
        in_ch = 256 + 256 + 256
        self.head = SpatialPromptHead(
            in_ch=in_ch,
            channels=tuple(self.cfg['training']['head_hidden_dims']),
            dropout=self.cfg['training']['head_dropout']
        )

    def _preprocess(self, images):
        """Images are [0,1] and already resized to training image_size (e.g., 512)."""
        images = images * 255.0
        images = (images - self.pixel_mean.to(images.device)) / self.pixel_std.to(images.device)
        # Resize to SAM encoder size (1024)
        images_1024 = F.interpolate(images, size=(self.img_size, self.img_size),
                                    mode='bilinear', align_corners=False)
        return images_1024

    def forward(self, images, prompts_batch):
        """
        images: (B,3,Ht,Wt) in [0,1] where Ht=Wt=image_size from collate
        prompts_batch: list of length B; each is list of dicts with 'location':[y,x] in Ht,Wt space
        """
        B, _, Ht, Wt = images.shape
        dev = images.device

        # Preprocess for SAM & embed
        img_1024 = self._preprocess(images)
        with torch.no_grad():
            img_emb = self.sam.image_encoder(img_1024)  # (B,256,64,64)

        # Prepare outputs
        sam_lowres_logits = []
        head_inputs = []

        # scale factors from training-size -> 1024
        sx = self.img_size / float(Wt)
        sy = self.img_size / float(Ht)

        for i in range(B):
            prompts = prompts_batch[i]
            if len(prompts) == 0:
                # ensure at least one background click at image center
                prompts = [{'location': [Ht//2, Wt//2], 'label': 0, 'noisy_label': 0, 'eps': 0.1}]

            # SAM expects (x,y) in 1024 coordinates
            pts_xy = np.array([[p['location'][1] * sx, p['location'][0] * sy] for p in prompts], dtype=np.float32)
            point_coords = torch.from_numpy(pts_xy)[None].to(dev)  # (1,N,2)
            point_labels = torch.tensor([p['noisy_label'] for p in prompts], dtype=torch.int64, device=dev)[None]  # (1,N)

            with torch.no_grad():
                sparse, dense = self.sam.prompt_encoder(points=(point_coords, point_labels),
                                                        boxes=None, masks=None)
                # dense: (1,256,64,64), sparse: (1,N,256)
                low_res, _ = self.sam.mask_decoder(
                    image_embeddings=img_emb[i:i+1],
                    image_pe=self.sam.prompt_encoder.get_dense_pe(),
                    sparse_prompt_embeddings=sparse,
                    dense_prompt_embeddings=dense,
                    multimask_output=False
                )  # (1,1,256,256)

            sam_lowres_logits.append(low_res[0])  # (1,256,256)

            # Build head features
            dense_64 = dense[0]                         # (256,64,64)
            img_e_64 = img_emb[i]                       # (256,64,64)
            sparseN = sparse[0].reshape(-1, self.embed_dim)  # (N,256) robust to API differences
            if sparseN.numel() > 0:
                prompt_agg = sparseN.mean(dim=0)        # (256,)
            else:
                prompt_agg = torch.zeros(self.embed_dim, device=dev)
            prompt_agg_64 = prompt_agg.view(self.embed_dim, 1, 1).expand(-1, img_e_64.shape[1], img_e_64.shape[2])

            head_in = torch.cat([img_e_64, dense_64, prompt_agg_64], dim=0)  # (768,64,64)
            head_inputs.append(head_in)

        head_inputs = torch.stack(head_inputs, dim=0)             # (B,768,64,64)
        head_corr_64 = self.head(head_inputs)                     # (B,1,64,64)
        head_corr_256 = F.interpolate(head_corr_64, size=(256, 256), mode='bilinear', align_corners=False)

        sam_lowres = torch.stack(sam_lowres_logits, dim=0)        # (B,1,256,256)
        combined_lowres = sam_lowres + head_corr_256              # (B,1,256,256)

        # Upsample to training image_size (Ht, Wt)
        logits = F.interpolate(combined_lowres, size=(Ht, Wt), mode='bilinear', align_corners=False)
        return logits  # (B,1,Ht,Wt)


# ============================================================================
# Loss
# ============================================================================

def noise_aware_bce_loss(logits, prompts_batch):
    """
    Per-prompt noise-aware BCE using stored per-prompt epsilon (eps).
    """
    total, n = 0.0, 0
    for i, prompts in enumerate(prompts_batch):
        for p in prompts:
            y, x = p['location']
            eps = p.get('eps', 0.1)
            noisy_label = p['noisy_label']
            logit = logits[i, 0, y, x]
            p_clean = torch.sigmoid(logit)
            p_noisy = eps + (1 - 2*eps) * p_clean  # P(~l=1 | q)
            loss = -torch.log(p_noisy.clamp_min(1e-10)) if noisy_label == 1 \
                   else -torch.log((1 - p_noisy).clamp_min(1e-10))
            total += loss
            n += 1
    return total / max(n, 1)


# ============================================================================
# Training
# ============================================================================

def maybe_log_wandb(enabled, metrics: dict):
    if enabled and WANDB_OK:
        wandb.log(metrics)


def train_bayesian_head(config):
    set_seed(config['training'].get('seed', 42))
    use_wandb = config['logging'].get('use_wandb', False) and WANDB_OK
    if use_wandb:
        wandb.init(project=config['logging']['wandb_project'],
                   entity=config['logging']['wandb_entity'],
                   config=config)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    train_ds = BayesianSAMDataset(config, split='train')
    val_ds   = BayesianSAMDataset(config, split='val')

    collate = partial(collate_fn, image_size=config['training'].get('image_size', 512))
    train_loader = DataLoader(train_ds, batch_size=config['training']['batch_size'],
                              shuffle=True, num_workers=config['training'].get('num_workers', 4),
                              collate_fn=collate)
    val_loader   = DataLoader(val_ds, batch_size=config['training']['batch_size'],
                              shuffle=False, num_workers=config['training'].get('num_workers', 4),
                              collate_fn=collate)

    model = BayesianSAMHead(config).to(device)
    optim = Adam(model.head.parameters(), lr=config['training']['learning_rate'],
                 weight_decay=config['training']['weight_decay'])
    sched = CosineAnnealingLR(optim, T_max=config['training']['num_epochs'])

    best_val = float('inf'); patience = 0

    for epoch in range(1, config['training']['num_epochs'] + 1):
        model.train(); model.sam.eval()
        running = 0.0
        for bi, batch in enumerate(tqdm(train_loader, desc=f"Epoch {epoch}")):
            images = batch['images'].to(device)
            prompts = batch['prompts']

            optim.zero_grad()
            logits = model(images, prompts)
            loss = noise_aware_bce_loss(logits, prompts)
            loss.backward()
            optim.step()

            running += loss.item()
            if bi % config['logging']['log_interval'] == 0:
                maybe_log_wandb(use_wandb, {'train/loss': loss.item(),
                                            'train/lr': optim.param_groups[0]['lr']})
        train_epoch_loss = running / max(1, len(train_loader))

        # validation
        model.eval()
        vloss = 0.0
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validation"):
                images = batch['images'].to(device)
                prompts = batch['prompts']
                logits = model(images, prompts)
                loss = noise_aware_bce_loss(logits, prompts)
                vloss += loss.item()
        val_epoch_loss = vloss / max(1, len(val_loader))

        sched.step()
        maybe_log_wandb(use_wandb, {'epoch': epoch,
                                    'train/epoch_loss': train_epoch_loss,
                                    'val/epoch_loss': val_epoch_loss})
        print(f"Epoch {epoch}: train {train_epoch_loss:.4f}  val {val_epoch_loss:.4f}")

        # save best
        if val_epoch_loss < best_val:
            best_val = val_epoch_loss
            patience = 0
            ckpt = Path(config['logging']['checkpoint_dir']) / 'best_model.pth'
            ckpt.parent.mkdir(parents=True, exist_ok=True)
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.head.state_dict(),
                'optimizer_state_dict': optim.state_dict(),
                'val_loss': val_epoch_loss,
                'config': config
            }, ckpt)
            print(f"Saved best model → {ckpt}")
        else:
            patience += 1

        if patience >= config['training']['early_stopping_patience']:
            print(f"Early stopping at epoch {epoch}")
            break

    if use_wandb:
        wandb.finish()
    return model


# ============================================================================
# Optional: Laplace on a 2-parameter logit calibrator
# ============================================================================

class LogitCalibrator(nn.Module):
    """y = sigmoid(a * x + b). We fit Laplace on [a,b] with (x,y) prompt-level pairs."""
    def __init__(self):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(1.0))
        self.bias  = nn.Parameter(torch.tensor(0.0))

    def forward(self, x):  # x: (...,) logits
        return self.scale * x + self.bias


def gather_prompt_logits(model, loader, device, max_batches=None):
    """Collect (logit_at_prompt, true_label) pairs to fit calibrator."""
    xs, ys = [], []
    model.eval()
    with torch.no_grad():
        for bi, batch in enumerate(tqdm(loader, desc="Gathering logits for calibration")):
            images = batch['images'].to(device)
            prompts = batch['prompts']
            logits = model(images, prompts)  # (B,1,H,W)
            for i, pr in enumerate(prompts):
                for p in pr:
                    y, x = p['location']
                    xs.append(logits[i, 0, y, x].detach().cpu().view(1))
                    ys.append(float(p['label']))
            if max_batches is not None and (bi + 1) >= max_batches:
                break
    if len(xs) == 0:
        raise RuntimeError("No prompt logits gathered for calibration.")
    X = torch.cat(xs, dim=0).float()
    Y = torch.tensor(ys).float()
    return X, Y


def fit_laplace_calibrator(config, model, train_loader, device):
    """Fit Laplace on 2-parameter calibrator using prompt-level logits."""
    if not LAPLACE_AVAILABLE:
        print("Laplace not available; skipping calibrator.")
        return None, None
    print("Fitting Laplace calibrator...")
    # Build a small calibration set (avoid making training slow)
    max_batches = config['laplace'].get('max_calib_batches', 50)
    X, Y = gather_prompt_logits(model, train_loader, device, max_batches=max_batches)

    calib = LogitCalibrator().to(device)
    # Quick MLE fit with Adam to initialize (optional, a few steps)
    opt = Adam(calib.parameters(), lr=5e-2)
    for _ in range(200):
        opt.zero_grad()
        logits = calib(X.to(device)).view(-1)
        prob = torch.sigmoid(logits)
        loss = F.binary_cross_entropy(prob, Y.to(device))
        loss.backward()
        opt.step()

    # Laplace on calibrator parameters
    dl = DataLoader(TensorDataset(X, Y.long()), batch_size=512, shuffle=False)
    la = Laplace(calib, likelihood='classification', subset_of_weights='all', hessian_structure='diagonal')
    la.fit(dl)
    try:
        la.optimize_prior_precision(method='marglik')
    except Exception:
        pass

    # Save
    ckpt_dir = Path(config['logging']['checkpoint_dir'])
    torch.save(calib.state_dict(), ckpt_dir / 'calibrator.pth')
    torch.save({'prior_precision': getattr(la, 'prior_precision', None)}, ckpt_dir / 'laplace_meta.pth')
    print("Calibrator + Laplace saved.")
    return calib, la


# ============================================================================
# Validation metrics
# ============================================================================

def expected_calibration_error(y_true, y_pred, n_bins=10):
    bins = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for lo, hi in zip(bins[:-1], bins[1:]):
        inb = (y_pred >= lo) & (y_pred < hi)
        if inb.any():
            acc = y_true[inb].mean()
            conf = y_pred[inb].mean()
            ece += abs(conf - acc) * inb.mean()
    return float(ece)


def negative_log_likelihood(y_true, y_pred):
    eps = 1e-10
    p = np.clip(y_pred, eps, 1 - eps)
    return float(-(y_true * np.log(p) + (1 - y_true) * np.log(1 - p)).mean())


def brier_score(y_true, y_pred):
    return float(np.mean((y_pred - y_true) ** 2))


def make_reliability_diagram(y_true, y_pred, n_bins, save_path):
    fop, mpv = calibration_curve(y_true, y_pred, n_bins=n_bins, strategy='uniform')
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot([0, 1], [0, 1], 'k--', label='Perfect')
    ax.plot(mpv, fop, 'o-', label='Model')
    ax.set_xlabel('Mean predicted value'); ax.set_ylabel('Fraction of positives'); ax.set_title('Reliability')
    ax.legend(); ax.grid(alpha=0.3)
    plt.savefig(save_path, dpi=150, bbox_inches='tight'); plt.close(fig)


@torch.no_grad()
def validate_model(config, model, val_loader, device, calibrator=None, laplace_obj=None):
    model.eval()
    y_true, y_pred = [], []

    for batch in tqdm(val_loader, desc="Validation (metrics)"):
        images = batch['images'].to(device)
        prompts = batch['prompts']
        logits = model(images, prompts)  # (B,1,H,W)
        probs = torch.sigmoid(logits)

        # If calibrator is provided, calibrate the *logits at prompt coords*
        for i, pr in enumerate(prompts):
            for p in pr:
                y, x = p['location']
                logit = logits[i, 0, y, x]
                if calibrator is not None:
                    z = calibrator(logit.view(1).to(device)).view(1)
                    pred = torch.sigmoid(z).item()
                else:
                    pred = probs[i, 0, y, x].item()
                y_true.append(float(p['label']))
                y_pred.append(float(pred))

    y_true = np.array(y_true, dtype=np.float32)
    y_pred = np.array(y_pred, dtype=np.float32)

    ece = expected_calibration_error(y_true, y_pred, n_bins=config['validation']['calibration_bins'])
    nll = negative_log_likelihood(y_true, y_pred)
    br  = brier_score(y_true, y_pred)

    outdir = Path(config['logging']['checkpoint_dir'])
    make_reliability_diagram(y_true, y_pred, config['validation']['calibration_bins'],
                             outdir / 'reliability_diagram.png')

    print("\n===== VALIDATION =====")
    print(f"ECE  : {ece:.4f}  (target < {config['validation']['ece_target']})")
    print(f"NLL  : {nll:.4f}  (target < {config['validation']['nll_target']})")
    print(f"Brier: {br:.4f}  (target < {config['validation']['brier_target']})")
    print("======================\n")

    metrics = {'ECE': ece, 'NLL': nll, 'Brier': br}
    with open(outdir / 'validation_metrics.json', 'w') as f:
        json.dump(metrics, f, indent=2)
    return metrics


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True, type=str)
    parser.add_argument('--skip-laplace', action='store_true')
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    print("="*60)
    print("BAYESIAN SAM HEAD TRAINING PIPELINE (fixed)")
    print("="*60)
    print(yaml.dump(config, sort_keys=False, default_flow_style=False))
    print("="*60)

    Path(config['logging']['checkpoint_dir']).mkdir(parents=True, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Train
    model = train_bayesian_head(config)

    # Rebuild loaders (for Laplace + validation)
    train_ds = BayesianSAMDataset(config, split='train')
    val_ds   = BayesianSAMDataset(config, split='val')
    collate  = partial(collate_fn, image_size=config['training'].get('image_size', 512))
    train_loader = DataLoader(train_ds, batch_size=config['training']['batch_size'],
                              shuffle=False, num_workers=config['training'].get('num_workers', 4),
                              collate_fn=collate)
    val_loader   = DataLoader(val_ds, batch_size=config['training']['batch_size'],
                              shuffle=False, num_workers=config['training'].get('num_workers', 4),
                              collate_fn=collate)

    # Load best head
    ckpt = Path(config['logging']['checkpoint_dir']) / 'best_model.pth'
    state = torch.load(ckpt, map_location=device)
    model = BayesianSAMHead(config).to(device)
    model.head.load_state_dict(state['model_state_dict'])
    model.eval()

    # Optional: Laplace on calibrator
    calibrator = None; la = None
    if (not args.skip_laplace) and LAPLACE_AVAILABLE and config['laplace'].get('enable_calibrator', True):
        calibrator, la = fit_laplace_calibrator(config, model, train_loader, device)
        print("Laplace calibrator ready.")
    else:
        print("Skipping Laplace calibrator.")

    # Validate
    metrics = validate_model(config, model, val_loader, device, calibrator=calibrator, laplace_obj=la)

    print("\nPIPELINE COMPLETE")
    print(f"Model checkpoint : {ckpt}")
    if calibrator is not None:
        print(f"Calibrator       : {Path(config['logging']['checkpoint_dir']) / 'calibrator.pth'}")
        print(f"Laplace meta     : {Path(config['logging']['checkpoint_dir']) / 'laplace_meta.pth'}")
    print(f"Metrics JSON     : {Path(config['logging']['checkpoint_dir']) / 'validation_metrics.json'}")
    print(f"Reliability plot : {Path(config['logging']['checkpoint_dir']) / 'reliability_diagram.png'}")


if __name__ == '__main__':
    main()
