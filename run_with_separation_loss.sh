#!/bin/bash
# Quick start script for training with separation loss

set -e  # Exit on error

echo "=========================================="
echo "Separation Loss Training - Quick Start"
echo "=========================================="
echo ""

# Configuration
DATASET_ROOT="Stomata_Dataset"
IMAGE_SUBDIR="barley_all/images/train"
LABEL_SUBDIR="barley_all/labels/train"
OUTPUT_DIR="outputs/mtkd_separation_test"
EPOCHS=20
BATCH_SIZE=8

# Separation loss parameters
SEP_WEIGHT=0.5
SEP_LAYER=10
SEP_SAMPLES=5
SEP_MARGIN=0.2

echo "Configuration:"
echo "  Dataset: $DATASET_ROOT"
echo "  Output: $OUTPUT_DIR"
echo "  Epochs: $EPOCHS"
echo "  Batch size: $BATCH_SIZE"
echo "  Separation loss weight: $SEP_WEIGHT"
echo ""

# Check if dataset exists
if [ ! -d "$DATASET_ROOT" ]; then
    echo "❌ Error: Dataset not found at $DATASET_ROOT"
    exit 1
fi

echo "✓ Dataset found"
echo ""

# Create output directory
mkdir -p "$OUTPUT_DIR"

echo "Starting training with separation loss..."
echo ""

# Run training
python mtkd_framework/run_v2.py \
  --dataset-root "$DATASET_ROOT" \
  --image-subdir "$IMAGE_SUBDIR" \
  --label-subdir "$LABEL_SUBDIR" \
  --output-dir "$OUTPUT_DIR" \
  \
  --epochs "$EPOCHS" \
  --batch-size "$BATCH_SIZE" \
  --lr 1e-4 \
  --warmup-epochs 3 \
  \
  --separation-loss-weight "$SEP_WEIGHT" \
  --separation-target-layer "$SEP_LAYER" \
  --separation-sample-points "$SEP_SAMPLES" \
  --separation-valley-margin "$SEP_MARGIN" \
  \
  --burn-up-epochs 3 \
  --align-target-start 8 \
  --feature-align-weight 1.0 \
  \
  --device cuda

echo ""
echo "=========================================="
echo "Training completed!"
echo "=========================================="
echo ""
echo "Results saved to: $OUTPUT_DIR"
echo ""
echo "Next steps:"
echo "  1. Check training log: $OUTPUT_DIR/training.log"
echo "  2. Visualize results: tensorboard --logdir $OUTPUT_DIR"
echo "  3. Compare with baseline (no separation loss)"
echo ""
