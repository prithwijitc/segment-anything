#!/bin/bash
# quickstart.sh - One-command training pipeline

set -e

echo "==================================="
echo "Bayesian SAM Head - Quick Start"
echo "==================================="



# Run training
echo "Starting training..."
python train_bayesian_head.py --config config.yaml

# Analyze results
echo "Analyzing results..."
python analyze_results.py /home/prithwijit/Vit/bayesian_check/checkpoints/

echo "Done! Check /home/prithwijit/Vit/bayesian_check/checkpoints/ for trained model."