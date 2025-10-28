# analyze_results.py

"""
Analyze training results and generate comprehensive report.
"""

import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

def analyze_training_run(checkpoint_dir):
    """Generate analysis report from training run."""
    
    checkpoint_dir = Path(checkpoint_dir)
    
    # Load metrics
    with open(checkpoint_dir / 'validation_metrics.json') as f:
        metrics = json.load(f)
    
    print("="*60)
    print("TRAINING RUN ANALYSIS")
    print("="*60)
    
    # Summary statistics
    print("\n1. VALIDATION METRICS")
    print("-"*60)
    for key, value in metrics.items():
        if isinstance(value, float):
            print(f"{key:20s}: {value:.4f}")
    
    # Target comparison
    print("\n2. TARGET COMPARISON")
    print("-"*60)
    
    targets = {
        'ECE': 0.05,
        'NLL': 0.3,
        'Brier': 0.15,
        'MI_correlation': 0.7
    }
    
    for metric, target in targets.items():
        if metric in metrics:
            value = metrics[metric]
            if metric == 'MI_correlation':
                status = "✓" if value > target else "✗"
                print(f"{metric:20s}: {value:.4f} (target: > {target}, {status})")
            else:
                status = "✓" if value < target else "✗"
                print(f"{metric:20s}: {value:.4f} (target: < {target}, {status})")
    
    # Model size
    print("\n3. MODEL INFORMATION")
    print("-"*60)
    
    checkpoint = torch.load(checkpoint_dir / 'best_model.pth')
    n_params = sum(p.numel() for p in checkpoint['model_state_dict'].values())
    print(f"Trainable parameters: {n_params:,}")
    print(f"Model size: {n_params * 4 / 1024**2:.2f} MB (float32)")
    
    # Laplace info
    laplace_state = torch.load(checkpoint_dir / 'laplace_state.pth')
    print(f"Prior precision: {laplace_state['prior_precision']:.4f}")
    print(f"Training samples: {laplace_state['n_data']:,}")
    
    # Recommendations
    print("\n4. RECOMMENDATIONS")
    print("-"*60)
    
    if metrics.get('ECE', 1.0) > 0.05:
        print("⚠ High ECE detected:")
        print("  → Consider temperature scaling")
        print("  → Check if prior precision is too low")
    
    if metrics.get('MI_correlation', 0.0) < 0.7:
        print("⚠ Low MI correlation:")
        print("  → Use fuller Hessian approximation (kfac or full)")
        print("  → Increase number of posterior samples")
    
    if metrics.get('NLL', 1.0) > 0.3:
        print("⚠ High NLL:")
        print("  → Train for more epochs")
        print("  → Reduce noise rate in training")
    
    print("\n" + "="*60)

if __name__ == '__main__':
    import sys
    checkpoint_dir = sys.argv[1] if len(sys.argv) > 1 else './checkpoints'
    analyze_training_run(checkpoint_dir)