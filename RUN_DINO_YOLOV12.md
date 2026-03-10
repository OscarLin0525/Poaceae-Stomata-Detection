# Run DINO + YOLOv12

## 1. Environment Setup

```bash
cd /home/oscar/Poaceae-Stomata-Detection
source Thesis/bin/activate

# Core deps
pip install -r requirements.txt

# Install local DINOv3 package
pip install -e dinov3-main

# Install local Ultralytics package (workspace version)
pip install -e ultralytics
```

## 2. Prepare YOLO Data YAML (barley_all)

Create `barley_all.yaml` in project root:

```yaml
path: /home/oscar/Poaceae-Stomata-Detection/Stomata_Dataset/barley_all
train: images/train
val: images/val

names:
  0: stomata
```

## 3. Quick Check DINO Checkpoint

```bash
cd /home/oscar/Poaceae-Stomata-Detection
python - << 'PY'
import torch
p = 'dinov3-main/weight folder/dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth'
ckpt = torch.load(p, map_location='cpu', weights_only=False)
print('DINO checkpoint loaded:', isinstance(ckpt, dict))
print('Path:', p)
PY
```

## 4. Train YOLOv12

If your environment already has YOLOv12 weights, use `yolo12s.pt`.
If not available, switch to `yolo11s.pt`.

```bash
cd /home/oscar/Poaceae-Stomata-Detection
source Thesis/bin/activate

# Preferred (YOLOv12)
yolo detect train \
  model=yolo12s.pt \
  data=barley_all.yaml \
  epochs=100 \
  imgsz=640 \
  batch=8 \
  device=0 \
  project=runs/yolo \
  name=barley_yolo12
```

Fallback:

```bash
yolo detect train \
  model=yolo11s.pt \
  data=barley_all.yaml \
  epochs=100 \
  imgsz=640 \
  batch=8 \
  device=0 \
  project=runs/yolo \
  name=barley_yolo11
```

## 5. Inference

```bash
yolo detect predict \
  model=runs/yolo/barley_yolo12/weights/best.pt \
  source=Stomata_Dataset/barley_all/images/val \
  conf=0.25 \
  save=True
```

## 6. Validate

```bash
yolo detect val \
  model=runs/yolo/barley_yolo12/weights/best.pt \
  data=barley_all.yaml \
  imgsz=640
```

## Notes

- If `yolo12s.pt` cannot be downloaded or is unavailable in your installed Ultralytics version, use `yolo11s.pt`.
- DINO in this repo is mainly used as teacher/alignment backbone (for MTKD), not as standalone detector.
