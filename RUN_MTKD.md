# Run MTKDv2 (Offline Pseudo Workflow)

This project currently uses two different ultralytics codebases:

- `Myultralytics_All` contains custom modules needed by wheat teacher checkpoints (e.g. `CA`).
- `Poaceae-Stomata-Detection/ultralytics` contains YOLOv12 support for student training.

To avoid import conflicts, use this two-step pipeline:

1. Generate pseudo labels offline with `Myultralytics_All`.
2. Train MTKDv2 with offline pseudo labels using Poaceae ultralytics.

## 1. Environment Setup

```bash
cd /home/oscar/Poaceae-Stomata-Detection
source Thesis/bin/activate

pip install -r requirements.txt
pip install -e dinov3-main
pip install -e ultralytics
```

## 2. Generate Offline Pseudo Labels (with Myultralytics)

Use wheat teacher to label barley training images and save YOLO txt files.

```bash
cd /home/oscar/Poaceae-Stomata-Detection
source Thesis/bin/activate

PYTHONPATH=/home/oscar/Myultralytics_All:$PYTHONPATH \
yolo obb predict \
  model=/home/oscar/Poaceae-Stomata-Detection/runs/obb/wheat_teacher_pretrained_20%_three_class/weights/best.pt \
  source=/home/oscar/Poaceae-Stomata-Detection/Stomata_Dataset/BARLEY/1%/images/train \
  conf=0.30 \
  save_txt=True \
  save_conf=True \
  project=outputs/pseudo \
  name=barley1_from_wheat20
```

Pseudo label directory produced by the command above:

`outputs/pseudo/barley1_from_wheat20/labels`

## 3. Train MTKDv2 with Offline Pseudo Labels

Train with DINO frozen teacher + YOLOv12 student + offline pseudo labels.

```bash
cd /home/oscar/Poaceae-Stomata-Detection
source Thesis/bin/activate

python -m mtkd_framework.run_v2 \
  --dino-checkpoint "dinov3-main/weight folder/dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth" \
  --student-weights yolo12l.pt \
  --pseudo-mode offline \
  --pseudo-label-dir outputs/pseudo/barley1_from_wheat20/labels \
  --pseudo-score-threshold 0.3 \
  --dataset-root Stomata_Dataset \
  --image-subdir BARLEY/1%/images/train \
  --label-subdir BARLEY/1%/labels/train \
  --val-ratio 0.1 \
  --epochs 1300 \
  --burn-up-epochs 0 \
  --align-target-start 0 \
  --warmup-epochs 30 \
  --save-freq 50 \
  --output-dir outputs/mtkd_v2_barley1_offline_1300ep
```

For 5% barley, only replace the three paths below:

- `source=.../BARLEY/5%/images/train` in pseudo-generation command.
- `--image-subdir BARLEY/5%/images/train`
- `--label-subdir BARLEY/5%/labels/train`

## 4. Optional: Quick Smoke Test

```bash
python -m mtkd_framework.run_v2 \
  --dino-checkpoint "dinov3-main/weight folder/dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth" \
  --student-weights yolo12l.pt \
  --pseudo-mode offline \
  --pseudo-label-dir outputs/pseudo/barley1_from_wheat20/labels \
  --dataset-root Stomata_Dataset \
  --image-subdir BARLEY/1%/images/train \
  --label-subdir BARLEY/1%/labels/train \
  --epochs 1 \
  --batch-size 2 \
  --num-workers 2 \
  --output-dir outputs/mtkd_v2_smoke
```

## 5. Expected Outputs

- `outputs/.../training.log`
- `outputs/.../config.json`
- `outputs/.../best_model.pth`
- `outputs/.../checkpoint_epoch_*.pth`

## 6. Notes

- MTKDv2 entry point is `python -m mtkd_framework.run_v2`.
- In offline mode, `--wheat-teacher-weights` is not required.
- To disable burn-in completely, use `--burn-up-epochs 0 --align-target-start 0`.
- If GPU memory is tight, reduce `--batch-size` or switch to `--student-align-layer p3`.



3/30 testing instruction can run
cd /home/oscar/Poaceae-Stomata-Detection && \
python mtkd_framework/run_v2.py \
  --dino-checkpoint "/home/oscar/Poaceae-Stomata-Detection/dinov3-main/weight folder/dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth" \
  --dataset-root /home/oscar/Poaceae-Stomata-Detection/outputs/images \
  --image-subdir "" \
  --label-subdir "" \
  --pseudo-label-dir /home/oscar/Poaceae-Stomata-Detection/outputs/pseudo_labels \
  --pseudo-mode offline \
  --burn-up-epochs 0 \
  --align-target-start 0 \
  --no-zero-pseudo-box-reg \
  --epochs 1300 \
  --warmup-epochs 20 \
  --student-weights yolo12l.pt \
  --batch-size 2 \
  --num-workers 0 \
  --output-dir /home/oscar/Poaceae-Stomata-Detection/outputs/mtkd_final_yolo12l_1300ep


## 7. OBB Student Route (No yolo12-obb.pt required)

If you do not have a pretrained `yolo12-obb.pt`, initialize from OBB YAML directly:

```bash
cd /home/oscar/Poaceae-Stomata-Detection && \
python mtkd_framework/run_v2.py \
  --dino-checkpoint "/home/oscar/Poaceae-Stomata-Detection/dinov3-main/weight folder/dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth" \
  --student-weights ultralytics/ultralytics/cfg/models/12/yolo12-obb.yaml \
  --dataset-root /home/oscar/Poaceae-Stomata-Detection/outputs/images \
  --image-subdir "" \
  --label-subdir "" \
  --pseudo-label-dir /home/oscar/Poaceae-Stomata-Detection/outputs/pseudo_labels \
  --pseudo-mode offline \
  --burn-up-epochs 0 \
  --align-target-start 0 \
  --no-zero-pseudo-box-reg \
  --epochs 1300 \
  --warmup-epochs 20 \
  --batch-size 1 \
  --num-workers 0 \
  --output-dir /home/oscar/Poaceae-Stomata-Detection/outputs/mtkd_final_yolo12_obb_1300ep
```

This route is now supported by MTKDv2 wrappers, pseudo-label parser, and trainer.