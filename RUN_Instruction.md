### 1) Minimal smoke test (unlabeled-only, 2-stage)

This version removes GT supervision and keeps only:
- Stage 1: **feature alignment only**
- Stage 2: **feature alignment + pseudo prediction alignment**

You said Stage 0 (student pretraining) is handled separately, so this starts from your Stage 1.

Smoke stage allocation (10 epochs):
- Stage 1: `epoch < 4`
- Stage 2: `epoch >= 4`

```bash
cd /home/oscar/Poaceae-Stomata-Detection && \
source Thesis/bin/activate && \
python mtkd_framework/run_v2.py \
  --dino-checkpoint "/home/oscar/Poaceae-Stomata-Detection/dinov3-main/weight folder/dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth" \
  --student-weights /home/oscar/Poaceae-Stomata-Detection/ultralytics/yolo26x-obb.pt \
  --wheat-teacher-weights /home/oscar/Poaceae-Stomata-Detection/runs/obb/yolov12_train_with_wheat_20%_three_class/weights/best.pt \
  --wheat-teacher-score-threshold 0.4 \
  --dataset-root /home/oscar/Poaceae-Stomata-Detection/Stomata_Dataset/ALL_DATA_IMAGE \
  --image-subdir images \
  --pseudo-mode online \
  --supervision-mode pseudo-only \
  --prediction-align-mode ultralytics \
  --feature-align-weight 0.5 \
  --feature-align-weight-target 0.0 \
  --unsup-loss-weight 0.5 \
  --burn-up-epochs 0 \
  --align-target-start 4 \
  --no-target-alignment \
  --no-zero-pseudo-box-reg \
  --align-easy-only \
  --epochs 10 \
  --warmup-epochs 0 \
  --batch-size 16 \
  --num-workers 0 \
  --best-by fitness \
  --map-data /home/oscar/Poaceae-Stomata-Detection/ultralytics/ultralytics/cfg/datasets/stomata-barley.yaml \
  --map-conf 0.001 \
  --map-iou 0.7 \
  --output-dir /home/oscar/Poaceae-Stomata-Detection/outputs/mtkd_unlabeled_2stage_smoke
```

### 2) Long training (unlabeled-only, 2-stage)

Recommended long run (500 epochs):
- Stage 1 (feature alignment): `epoch < 120`
- Stage 2 (feature + pseudo prediction): `epoch >= 120`

```bash
cd /home/oscar/Poaceae-Stomata-Detection && \
source Thesis/bin/activate && \
python mtkd_framework/run_v2.py \
  --dino-checkpoint "/home/oscar/Poaceae-Stomata-Detection/dinov3-main/weight folder/dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth" \
  --student-weights /home/oscar/Poaceae-Stomata-Detection/ultralytics/yolo26x-obb.pt \
  --wheat-teacher-weights /home/oscar/Poaceae-Stomata-Detection/runs/obb/yolov12_train_with_wheat_20%_three_class/weights/best.pt \
  --wheat-teacher-score-threshold 0.4 \
  --dataset-root /home/oscar/Poaceae-Stomata-Detection/Stomata_Dataset/ALL_DATA_IMAGE \
  --image-subdir images \
  --pseudo-mode online \
  --supervision-mode pseudo-only \
  --prediction-align-mode ultralytics \
  --feature-align-weight 0.5 \
  --feature-align-weight-target 0.0 \
  --unsup-loss-weight 0.7 \
  --burn-up-epochs 0 \
  --align-target-start 120 \
  --no-target-alignment \
  --no-zero-pseudo-box-reg \
  --align-easy-only \
  --epochs 500 \
  --early-stopping-patience 150 \
  --warmup-epochs 5 \
  --num-workers 1 \
  --best-by fitness \
  --map-data /home/oscar/Poaceae-Stomata-Detection/ultralytics/ultralytics/cfg/datasets/stomata-barley.yaml \
  --map-conf 0.001 \
  --map-iou 0.7 \
  --output-dir /home/oscar/Poaceae-Stomata-Detection/outputs/mtkd_unlabeled_2stage_500
```

### 3) Quick health checks during training

```bash
cd /home/oscar/Poaceae-Stomata-Detection && \
rg -n "Supervision mode|Pseudo source|Epoch [0-9]+|Train breakdown|Val breakdown|mAP eval" \
  outputs/mtkd_unlabeled_2stage_smoke/training.log -S | tail -n 100
```

Expected signs (unlabeled-only):
- Stage 1 (`epoch < align_target_start`): `align>0`, `pseudo_boxes=0`
- Stage 2 (`epoch >= align_target_start`): `align>0`, `pseudo_boxes>0`
- `loss_det` may stay near 0 in pseudo-only mode (by design)

### 4) Notes for pseudo-only mode

- `supervision-mode pseudo-only` disables GT supervision.
- `best-by fitness` now matches Ultralytics style:
  - `fitness = 0.1 * map50 + 0.9 * map50-95`
- If `pseudo_boxes` remains 0 in Stage 2, lower:
  - `--wheat-teacher-score-threshold` from `0.4` to `0.25~0.35`.

Ultralytics native `best.pt` rule (for standalone YOLO/OBB training) is:
- `fitness = 0.1 * mAP50 + 0.9 * mAP50-95`
- best updates when current fitness > best_fitness

MTKD `run_v2.py` is different:
- `--best-by loss`   -> select best by minimizing loss
- `--best-by map50`  -> select best by maximizing mAP50
- `--best-by map5095` -> select best by maximizing mAP50-95
- `--best-by fitness` -> select best by maximizing `0.1*map50 + 0.9*map50-95`

If you want MTKD best checkpoint behavior to be closest to Ultralytics fitness,
prefer `--best-by fitness`.

### 5) DINO-style dual-stream (all unlabeled)

If you want explicit dual-stream sampling while still unlabeled-only,
point both streams to unlabeled roots and keep supervision pseudo-only:

```bash
cd /home/oscar/Poaceae-Stomata-Detection && \
source Thesis/bin/activate && \
python mtkd_framework/run_v2.py \
  --dino-checkpoint "/home/oscar/Poaceae-Stomata-Detection/dinov3-main/weight folder/dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth" \
  --student-weights /home/oscar/Poaceae-Stomata-Detection/ultralytics/yolo26x-obb.pt \
  --wheat-teacher-weights /home/oscar/Poaceae-Stomata-Detection/runs/obb/yolov12_train_with_wheat_20%_three_class/weights/best.pt \
  --wheat-teacher-score-threshold 0.4 \
  --dataset-root /home/oscar/Poaceae-Stomata-Detection/Stomata_Dataset/ALL_DATA_IMAGE \
  --image-subdir images \
  --separate-source-target-data \
  --unlabeled-dataset-root /home/oscar/Poaceae-Stomata-Detection/Stomata_Dataset/ALL_DATA_IMAGE \
  --unlabeled-image-subdir images \
  --batch-size-label 8 \
  --batch-size-unlabel 8 \
  --pseudo-mode online \
  --supervision-mode pseudo-only \
  --prediction-align-mode ultralytics \
  --feature-align-weight 0.5 \
  --feature-align-weight-target 0.0 \
  --unsup-loss-weight 0.7 \
  --burn-up-epochs 0 \
  --align-target-start 120 \
  --no-target-alignment \
  --no-zero-pseudo-box-reg \
  --align-easy-only \
  --epochs 500 \
  --early-stopping-patience 150 \
  --warmup-epochs 5 \
  --num-workers 1 \
  --best-by fitness \
  --map-data /home/oscar/Poaceae-Stomata-Detection/ultralytics/ultralytics/cfg/datasets/stomata-barley.yaml \
  --map-conf 0.001 \
  --map-iou 0.7 \
  --output-dir /home/oscar/Poaceae-Stomata-Detection/outputs/mtkd_unlabeled_dualstream_500
```
