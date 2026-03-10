# Run MTKD (DINO-Teacher Aligned)

## 1. Environment Setup

```bash
cd /home/oscar/Poaceae-Stomata-Detection
source Thesis/bin/activate

pip install -r requirements.txt
pip install -e dinov3-main
pip install -e ultralytics
```

## 2. Basic MTKD Run (DINO align, no pseudo)

```bash
cd /home/oscar/Poaceae-Stomata-Detection
source Thesis/bin/activate

python -m mtkd_framework.run_v2 \
  --dino-checkpoint "dinov3-main/weight folder/dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth" \
  --epochs 100 \
  --batch-size 8 \
  --lr 1e-4 \
  --dataset-root Stomata_Dataset \
  --image-subdir barley_category/barley_image_fresh-leaf \
  --label-subdir barley_category/barley_label_fresh-leaf \
  --output-dir outputs/mtkd_v2_base
```

## 3. MTKD With FFT (current insert style)

```bash
python -m mtkd_framework.run_v2 \
  --dino-checkpoint "dinov3-main/weight folder/dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth" \
  --fft-after-blocks 9 \
  --fft-init-gate -5 \
  --epochs 100 \
  --batch-size 8 \
  --lr 1e-4 \
  --dataset-root Stomata_Dataset \
  --image-subdir barley_category/barley_image_fresh-leaf \
  --label-subdir barley_category/barley_label_fresh-leaf \
  --output-dir outputs/mtkd_v2_fft
```

## 4. Full MTKD (with pseudo labels)

```bash
python -m mtkd_framework.run_v2 \
  --dino-checkpoint "dinov3-main/weight folder/dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth" \
  --pseudo-label-dir pseudo_labels_wheat100/fresh/labels \
  --pseudo-score-threshold 0.5 \
  --burn-up-epochs 5 \
  --align-target-start 10 \
  --unsup-loss-weight 4.0 \
  --epochs 100 \
  --batch-size 8 \
  --lr 1e-4 \
  --dataset-root Stomata_Dataset \
  --image-subdir barley_category/barley_image_fresh-leaf \
  --label-subdir barley_category/barley_label_fresh-leaf \
  --output-dir outputs/mtkd_v2_full
```

## 5. Quick Debug Run (fast sanity check)

```bash
python -m mtkd_framework.run_v2 \
  --dino-checkpoint "dinov3-main/weight folder/dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth" \
  --epochs 1 \
  --batch-size 2 \
  --num-workers 2 \
  --output-dir outputs/mtkd_v2_smoke
```

## 6. Common Useful Flags

```bash
# Disable FFT
--no-fft

# Resume training
--resume outputs/mtkd_v2_base/best_model.pth

# Disable mixed precision (if unstable)
--no-mixed-precision

# Force CPU
--device cpu
```

## Expected Outputs

- `outputs/.../training.log`
- `outputs/.../config.json`
- `outputs/.../best_model.pth`
- `outputs/.../checkpoint_epoch_*.pth`

## Notes

- Your current code path for MTKD is `python -m mtkd_framework.run_v2`.
- If CUDA memory is tight, reduce `--batch-size` and/or use `--student-align-layer p3`.
