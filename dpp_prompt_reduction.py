#!/usr/bin/env python3
"""
DPP-Based Prompt Reduction for SAM
===================================

Implements the theoretically-grounded Determinantal Point Process (DPP) sampling
approach for reducing OOD (out-of-distribution) dense prompts to in-distribution
size while maximizing coverage and diversity.

Based on: "OOD to In-Distribution Reduction: Principled Approach"
"""

import numpy as np
import json
import argparse
from pathlib import Path
from typing import List, Tuple, Dict
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
from scipy.linalg import eigh
import torch
from PIL import Image


# ============================================================================
# DPP Sampling Implementation
# ============================================================================

def sample_dpp(K: np.ndarray, k: int, seed: int = 42) -> List[int]:
    """
    Sample exactly k items from a Determinantal Point Process.
    
    Uses the k-DPP algorithm from Kulesza & Taskar (2012).
    
    Args:
        K: Kernel matrix (N x N), symmetric positive semi-definite
        k: Number of items to sample
        seed: Random seed for reproducibility
        
    Returns:
        List of k selected indices
    """
    np.random.seed(seed)
    N = K.shape[0]
    
    if k >= N:
        return list(range(N))
    
    # Step 1: Eigendecomposition
    eigenvalues, eigenvectors = eigh(K)
    
    # Sort in descending order
    idx = eigenvalues.argsort()[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]
    
    # Step 2: Sample k eigenvectors using elementary symmetric polynomials
    # For k-DPP, we need exactly k eigenvectors
    # Use a greedy approximation: select top k eigenvalues with highest values
    selected_eigenvectors = eigenvectors[:, :k]
    
    # Step 3: Sequential sampling from the selected eigenspace
    selected_indices = []
    V = selected_eigenvectors.copy()
    
    for _ in range(k):
        # Compute probabilities proportional to squared projection lengths
        remaining = [i for i in range(N) if i not in selected_indices]
        
        if len(remaining) == 0:
            break
            
        probs = np.zeros(len(remaining))
        for idx, i in enumerate(remaining):
            phi_i = V[i, :]
            probs[idx] = np.sum(phi_i ** 2)
        
        # Normalize probabilities
        probs = probs / np.sum(probs)
        
        # Sample one item
        selected_idx = np.random.choice(len(remaining), p=probs)
        selected_item = remaining[selected_idx]
        selected_indices.append(selected_item)
        
        # Orthogonalize V against the selected item's feature
        phi_selected = V[selected_item, :]
        norm = np.linalg.norm(phi_selected)
        
        if norm > 1e-10:
            phi_selected = phi_selected / norm
            # Remove component in direction of phi_selected from all rows of V
            V = V - np.outer(V @ phi_selected, phi_selected)
    
    return selected_indices


# ============================================================================
# SAM Integration
# ============================================================================

def load_sam_model(checkpoint_path: str, model_type: str = "vit_h"):
    """Load SAM model from checkpoint."""
    try:
        from segment_anything import sam_model_registry, SamPredictor
        
        sam = sam_model_registry[model_type](checkpoint=checkpoint_path)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        sam.to(device=device)
        predictor = SamPredictor(sam)
        
        return sam, predictor
    except ImportError:
        raise ImportError(
            "segment_anything package not found. Install with:\n"
            "pip install git+https://github.com/facebookresearch/segment-anything.git"
        )


def extract_sam_features(image: np.ndarray, prompts: List[Dict], 
                        sam_model, predictor) -> np.ndarray:
    """
    Extract SAM encoder features at prompt locations.
    
    Args:
        image: Input image (H x W x 3)
        prompts: List of prompt dicts with 'x', 'y' keys
        sam_model: SAM model
        predictor: SAM predictor
        
    Returns:
        Feature matrix (N x d) where N is number of prompts
    """
    # Set image in predictor
    predictor.set_image(image)
    
    # Get image embeddings from SAM encoder
    with torch.no_grad():
        image_embedding = predictor.get_image_embedding()
    
    # Image embedding is (1, 256, 64, 64) for default SAM
    # We need to extract features at specific (x, y) coordinates
    
    features = []
    H, W = image.shape[:2]
    embed_h, embed_w = image_embedding.shape[2], image_embedding.shape[3]
    
    for prompt in prompts:
        x, y = prompt['x'], prompt['y']
        
        # Map image coordinates to embedding coordinates
        embed_x = int(x * embed_w / W)
        embed_y = int(y * embed_h / H)
        
        # Clamp to valid range
        embed_x = max(0, min(embed_w - 1, embed_x))
        embed_y = max(0, min(embed_h - 1, embed_y))
        
        # Extract feature vector at this location
        feat = image_embedding[0, :, embed_y, embed_x].cpu().numpy()
        
        # L2 normalize
        feat = feat / (np.linalg.norm(feat) + 1e-10)
        
        features.append(feat)
    
    return np.array(features)


def get_sam_masks(image: np.ndarray, prompts: List[Dict], predictor) -> np.ndarray:
    """
    Get SAM segmentation mask for given prompts.
    
    Args:
        image: Input image
        prompts: List of prompt dicts with 'x', 'y', 'label' keys
        predictor: SAM predictor
        
    Returns:
        Binary mask (H x W)
    """
    predictor.set_image(image)
    
    # Prepare point prompts
    point_coords = np.array([[p['x'], p['y']] for p in prompts])
    point_labels = np.array([p['label'] for p in prompts])
    
    # Predict mask
    masks, scores, logits = predictor.predict(
        point_coords=point_coords,
        point_labels=point_labels,
        multimask_output=False
    )
    
    return masks[0]


# ============================================================================
# DPP-Based Reduction Algorithm
# ============================================================================

def construct_kernel(features: np.ndarray, coords: np.ndarray, 
                    image_shape: Tuple[int, int]) -> np.ndarray:
    """
    Construct DPP kernel combining semantic and spatial similarity.
    
    Args:
        features: Feature matrix (N x d)
        coords: Coordinate matrix (N x 2)
        image_shape: (height, width) of image
        
    Returns:
        Kernel matrix (N x N)
    """
    N = features.shape[0]
    
    # Semantic similarity: cosine similarity (already normalized features)
    K_sem = features @ features.T
    
    # Spatial similarity: Gaussian kernel
    diag = np.sqrt(image_shape[0]**2 + image_shape[1]**2)
    sigma = 0.1 * diag
    
    dists = cdist(coords, coords, metric='euclidean')
    K_spat = np.exp(-dists**2 / (2 * sigma**2))
    
    # Combined kernel (element-wise product)
    K = K_sem * K_spat
    
    # Ensure numerical stability
    K = (K + K.T) / 2  # Make symmetric
    K = K + 1e-6 * np.eye(N)  # Add small diagonal for stability
    
    return K


def reduce_prompts_dpp(image: np.ndarray, prompts: List[Dict], 
                       sam_model, predictor, k_safe: int = 15,
                       seed: int = 42) -> List[Dict]:
    """
    Reduce OOD dense prompts to in-distribution set using DPP sampling.
    
    Args:
        image: Input image
        prompts: List of prompt dictionaries
        sam_model: SAM model
        predictor: SAM predictor
        k_safe: Target number of prompts (default: 15)
        seed: Random seed
        
    Returns:
        Reduced list of prompts
    """
    N = len(prompts)
    
    if N <= k_safe:
        print(f"Number of prompts ({N}) <= k_safe ({k_safe}). No reduction needed.")
        return prompts
    
    print(f"Reducing {N} prompts to {k_safe} using DPP sampling...")
    
    # Step 1: Extract SAM features
    print("Extracting SAM features...")
    features = extract_sam_features(image, prompts, sam_model, predictor)
    
    # Step 2: Construct kernel
    print("Constructing kernel...")
    coords = np.array([[p['x'], p['y']] for p in prompts])
    K = construct_kernel(features, coords, image.shape[:2])
    
    # Step 3: Stratified sampling by label
    print("Performing stratified DPP sampling...")
    pos_indices = [i for i, p in enumerate(prompts) if p['label'] == 1]
    neg_indices = [i for i, p in enumerate(prompts) if p['label'] == 0]
    
    # Determine number of positive and negative samples
    k_pos = max(3, min(len(pos_indices), int(k_safe * len(pos_indices) / N)))
    k_neg = min(k_safe - k_pos, len(neg_indices))
    
    # Adjust if we don't have enough negatives
    if k_neg < k_safe - k_pos and len(pos_indices) > k_pos:
        k_pos = k_safe - k_neg
    
    print(f"  Sampling {k_pos} positive and {k_neg} negative prompts...")
    
    selected_indices = []
    
    # Sample from positive prompts
    if len(pos_indices) > 0 and k_pos > 0:
        K_pos = K[np.ix_(pos_indices, pos_indices)]
        selected_pos = sample_dpp(K_pos, min(k_pos, len(pos_indices)), seed=seed)
        selected_indices.extend([pos_indices[i] for i in selected_pos])
    
    # Sample from negative prompts
    if len(neg_indices) > 0 and k_neg > 0:
        K_neg = K[np.ix_(neg_indices, neg_indices)]
        selected_neg = sample_dpp(K_neg, min(k_neg, len(neg_indices)), seed=seed+1)
        selected_indices.extend([neg_indices[i] for i in selected_neg])
    
    # Create reduced prompt list
    reduced_prompts = [prompts[i] for i in sorted(selected_indices)]
    
    # Reassign time indices
    for t, prompt in enumerate(reduced_prompts, 1):
        prompt['t'] = t
    
    print(f"Reduction complete: {len(reduced_prompts)} prompts selected.")
    
    return reduced_prompts


# ============================================================================
# Visualization
# ============================================================================

def visualize_results(image: np.ndarray, 
                     original_prompts: List[Dict],
                     reduced_prompts: List[Dict],
                     original_mask: np.ndarray,
                     reduced_mask: np.ndarray,
                     output_path: str):
    """
    Create side-by-side visualization of original vs reduced prompts.
    
    Args:
        image: Input image
        original_prompts: Original dense prompts
        reduced_prompts: Reduced prompts
        original_mask: SAM mask from original prompts
        reduced_mask: SAM mask from reduced prompts
        output_path: Path to save visualization
    """
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    
    # Left: Original prompts
    ax = axes[0]
    ax.imshow(image)
    
    # Overlay mask
    mask_overlay = np.zeros_like(image)
    mask_overlay[original_mask > 0] = [0, 255, 0]  # Green
    ax.imshow(mask_overlay, alpha=0.3)
    
    # Plot prompts
    for p in original_prompts:
        color = 'lime' if p['label'] == 1 else 'red'
        ax.plot(p['x'], p['y'], 'o', color=color, markersize=8, 
                markeredgecolor='white', markeredgewidth=1.5)
    
    ax.set_title(f'Original Dense Prompts (N={len(original_prompts)})', 
                 fontsize=14, fontweight='bold')
    ax.axis('off')
    
    # Right: Reduced prompts
    ax = axes[1]
    ax.imshow(image)
    
    # Overlay mask
    mask_overlay = np.zeros_like(image)
    mask_overlay[reduced_mask > 0] = [0, 255, 0]  # Green
    ax.imshow(mask_overlay, alpha=0.3)
    
    # Plot prompts
    for p in reduced_prompts:
        color = 'lime' if p['label'] == 1 else 'red'
        ax.plot(p['x'], p['y'], 'o', color=color, markersize=10, 
                markeredgecolor='white', markeredgewidth=2)
    
    ax.set_title(f'DPP-Reduced Prompts (N={len(reduced_prompts)})', 
                 fontsize=14, fontweight='bold')
    ax.axis('off')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Visualization saved to: {output_path}")
    plt.close()


# ============================================================================
# Validation
# ============================================================================

def compute_iou(mask1: np.ndarray, mask2: np.ndarray) -> float:
    """Compute Intersection over Union between two binary masks."""
    intersection = np.logical_and(mask1, mask2).sum()
    union = np.logical_or(mask1, mask2).sum()
    
    if union == 0:
        return 1.0 if intersection == 0 else 0.0
    
    return intersection / union


def validate_reduction(original_mask: np.ndarray, 
                      reduced_mask: np.ndarray,
                      gt_mask: np.ndarray = None) -> Dict[str, float]:
    """
    Validate the quality of prompt reduction.
    
    Args:
        original_mask: Mask from original dense prompts
        reduced_mask: Mask from reduced prompts
        gt_mask: Ground truth mask (optional)
        
    Returns:
        Dictionary of validation metrics
    """
    metrics = {}
    
    # IoU between original and reduced
    iou_orig_reduced = compute_iou(original_mask, reduced_mask)
    metrics['iou_original_vs_reduced'] = iou_orig_reduced
    
    # If GT is available
    if gt_mask is not None:
        iou_orig_gt = compute_iou(original_mask, gt_mask)
        iou_reduced_gt = compute_iou(reduced_mask, gt_mask)
        
        metrics['iou_original_vs_gt'] = iou_orig_gt
        metrics['iou_reduced_vs_gt'] = iou_reduced_gt
        
        # Quality preservation ratio
        if iou_orig_gt > 0:
            preservation_ratio = iou_reduced_gt / iou_orig_gt
            metrics['quality_preservation_ratio'] = preservation_ratio
    
    return metrics


# ============================================================================
# Main Pipeline
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='DPP-based prompt reduction for SAM',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python dpp_prompt_reduction.py \\
      --image image.jpg \\
      --prompts dense_prompts.json \\
      --sam_checkpoint sam_vit_h_4b8939.pth \\
      --output reduced_prompts.json

  python dpp_prompt_reduction.py \\
      --image image.jpg \\
      --prompts dense_prompts.json \\
      --sam_checkpoint sam_vit_h_4b8939.pth \\
      --output reduced_prompts.json \\
      --gt_mask ground_truth.png \\
      --k_safe 15
        """
    )
    
    parser.add_argument('--image', type=str, required=True,
                       help='Path to input image (JPG/PNG)')
    parser.add_argument('--prompts', type=str, required=True,
                       help='Path to input prompts JSON file')
    parser.add_argument('--sam_checkpoint', type=str, required=True,
                       help='Path to SAM checkpoint file')
    parser.add_argument('--output', type=str, required=True,
                       help='Path to output reduced prompts JSON file')
    parser.add_argument('--gt_mask', type=str, default=None,
                       help='Path to ground truth mask (optional, for validation)')
    parser.add_argument('--k_safe', type=int, default=15,
                       help='Target number of prompts (default: 15)')
    parser.add_argument('--model_type', type=str, default='vit_h',
                       choices=['vit_h', 'vit_l', 'vit_b'],
                       help='SAM model type (default: vit_h)')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for reproducibility (default: 42)')
    parser.add_argument('--viz_output', type=str, default=None,
                       help='Path to save visualization PNG (default: auto-generated)')
    
    args = parser.parse_args()
    
    # Set default visualization output path
    if args.viz_output is None:
        output_stem = Path(args.output).stem
        args.viz_output = str(Path(args.output).parent / f"{output_stem}_visualization.png")
    
    print("="*70)
    print("DPP-BASED PROMPT REDUCTION FOR SAM")
    print("="*70)
    print(f"Input image:       {args.image}")
    print(f"Input prompts:     {args.prompts}")
    print(f"SAM checkpoint:    {args.sam_checkpoint}")
    print(f"Output prompts:    {args.output}")
    print(f"Visualization:     {args.viz_output}")
    print(f"k_safe:            {args.k_safe}")
    print(f"Random seed:       {args.seed}")
    print("="*70)
    
    # Load image
    print("\n[1/6] Loading image...")
    image = np.array(Image.open(args.image).convert('RGB'))
    print(f"  Image shape: {image.shape}")
    
    # Load prompts
    print("\n[2/6] Loading prompts...")
    with open(args.prompts, 'r') as f:
        data = json.load(f)
        prompts = data['prompts']
    print(f"  Loaded {len(prompts)} prompts")
    
    # Load ground truth mask if provided
    gt_mask = None
    if args.gt_mask:
        print("\n[2.5/6] Loading ground truth mask...")
        gt_mask = np.array(Image.open(args.gt_mask).convert('L')) > 0
        print(f"  GT mask shape: {gt_mask.shape}")
    
    # Load SAM model
    print("\n[3/6] Loading SAM model...")
    sam_model, predictor = load_sam_model(args.sam_checkpoint, args.model_type)
    print("  SAM model loaded successfully")
    
    # Get original mask
    print("\n[4/6] Generating mask from original prompts...")
    original_mask = get_sam_masks(image, prompts, predictor)
    print(f"  Original mask coverage: {original_mask.sum() / original_mask.size * 100:.2f}%")
    
    # Reduce prompts using DPP
    print("\n[5/6] Reducing prompts using DPP...")
    reduced_prompts = reduce_prompts_dpp(
        image, prompts, sam_model, predictor, 
        k_safe=args.k_safe, seed=args.seed
    )
    
    # Get reduced mask
    print("\n[6/6] Generating mask from reduced prompts...")
    reduced_mask = get_sam_masks(image, reduced_prompts, predictor)
    print(f"  Reduced mask coverage: {reduced_mask.sum() / reduced_mask.size * 100:.2f}%")
    
    # Validate
    print("\n" + "="*70)
    print("VALIDATION METRICS")
    print("="*70)
    metrics = validate_reduction(original_mask, reduced_mask, gt_mask)
    for key, value in metrics.items():
        print(f"{key:.<50} {value:.4f}")
    
    # Save reduced prompts
    print("\n" + "="*70)
    print("SAVING RESULTS")
    print("="*70)
    output_data = {'prompts': reduced_prompts}
    with open(args.output, 'w') as f:
        json.dump(output_data, f, indent=2)
    print(f"Reduced prompts saved to: {args.output}")
    
    # Create visualization
    visualize_results(
        image, prompts, reduced_prompts,
        original_mask, reduced_mask,
        args.viz_output
    )
    
    print("\n" + "="*70)
    print("PIPELINE COMPLETE")
    print("="*70)
    print(f"Original prompts: {len(prompts)}")
    print(f"Reduced prompts:  {len(reduced_prompts)}")
    print(f"Reduction ratio:  {len(reduced_prompts)/len(prompts)*100:.1f}%")
    print(f"\nMask similarity:  {metrics['iou_original_vs_reduced']:.4f}")
    if 'quality_preservation_ratio' in metrics:
        print(f"Quality preserved: {metrics['quality_preservation_ratio']*100:.1f}%")
    print("="*70)


if __name__ == '__main__':
    main()