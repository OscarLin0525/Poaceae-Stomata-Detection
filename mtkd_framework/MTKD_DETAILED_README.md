# Multi-Teacher Knowledge Distillation (MTKD) Framework

用於禾本科氣孔檢測的多教師知識蒸餾框架

## 目錄

1. [框架概述](#框架概述)
2. [架構設計](#架構設計)
3. [安裝與依賴](#安裝與依賴)
4. [快速開始](#快速開始)
5. [模組詳解](#模組詳解)
6. [函數參考](#函數參考)
7. [配置說明](#配置說明)
8. [訓練指南](#訓練指南)
9. [自定義擴展](#自定義擴展)

---

## 框架概述

### 什麼是 MTKD？

Multi-Teacher Knowledge Distillation (MTKD) 是一種結合多個教師模型來訓練學生模型的知識蒸餾方法。本框架專為物體檢測任務設計，結合：

1. **DINO Feature Teacher**: 提供強大的視覺特徵表示，用於特徵對齊
2. **Ensemble Detection Teachers**: 多個預訓練檢測模型的集成，用於預測對齊
3. **Student Detector**: 要訓練的輕量級學生模型

### 核心優勢

- **多源知識**: 從特徵和預測兩個維度進行知識蒸餾
- **Ensemble 增強**: 通過 Weighted Box Fusion 融合多個教師的預測
- **靈活架構**: 支持自定義 backbone、head 和損失函數
- **易於擴展**: 模組化設計，方便添加新組件

---

## 架構設計

### 整體架構圖

```
┌─────────────────────────────────────────────────────────────────────┐
│                          Input Image                                │
│                              │                                      │
│          ┌───────────────────┼───────────────────┐                 │
│          │                   │                   │                 │
│          ▼                   ▼                   ▼                 │
│  ┌───────────────┐  ┌───────────────┐  ┌────────────────────┐     │
│  │ DINO Teacher  │  │   Student     │  │ Ensemble Teachers  │     │
│  │   (Frozen)    │  │  (Trainable)  │  │     (Frozen)       │     │
│  │               │  │               │  │                    │     │
│  │  ViT-Base/    │  │  Backbone     │  │  Teacher1          │     │
│  │  Large        │  │     ↓         │  │  Teacher2          │     │
│  │               │  │   Neck        │  │     ↓              │     │
│  │               │  │     ↓         │  │  Weighted Box      │     │
│  │               │  │   Head        │  │  Fusion            │     │
│  └───────┬───────┘  └───────┬───────┘  └──────────┬─────────┘     │
│          │                  │                     │                │
│          │ Features         │ Features &          │ Predictions    │
│          │                  │ Predictions         │                │
│          │                  │                     │                │
│          └────────────┬─────┴─────────────────────┘                │
│                       │                                             │
│                       ▼                                             │
│          ┌─────────────────────────────────────────┐               │
│          │            MTKD Loss                    │               │
│          │                                         │               │
│          │  ┌─────────────┐  ┌─────────────────┐  │               │
│          │  │  Feature    │  │   Prediction    │  │               │
│          │  │  Alignment  │  │   Alignment     │  │               │
│          │  │  Loss       │  │   Loss          │  │               │
│          │  └─────────────┘  └─────────────────┘  │               │
│          │                                         │               │
│          │  ┌─────────────────────────────────┐   │               │
│          │  │  Detection Loss (Optional)      │   │               │
│          │  └─────────────────────────────────┘   │               │
│          └─────────────────────────────────────────┘               │
└─────────────────────────────────────────────────────────────────────┘
```

### 資料夾結構

```
mtkd_framework/
├── __init__.py                 # 框架入口
├── README.md                   # 本文檔
├── train.py                    # 訓練腳本
│
├── models/                     # 模型模組
│   ├── __init__.py
│   ├── mtkd_model.py          # 主要 MTKD 模型
│   ├── student_model.py       # Student 模型
│   └── teacher_ensemble.py    # Teacher Ensemble
│
├── losses/                     # 損失函數模組
│   ├── __init__.py
│   ├── feature_alignment.py   # 特徵對齊損失
│   ├── prediction_alignment.py # 預測對齊損失
│   └── combined_loss.py       # 組合損失
│
└── utils/                      # 工具函數
    ├── __init__.py
    └── helpers.py             # 輔助函數
```

---

## 安裝與依賴

### 依賴項

```bash
pip install torch>=2.0.0
pip install torchvision>=0.15.0
pip install numpy
pip install scipy  # 用於 Hungarian matching
```

### 可選依賴

```bash
pip install fvcore  # 用於計算 FLOPs
pip install tensorboard  # 用於訓練可視化
```

---

## 快速開始

### 基本使用

```python
from mtkd_framework import MTKDModel, MTKDLoss

# 1. 創建模型
model = MTKDModel(
    student_config={
        "backbone_config": {"backbone_type": "resnet50"},
        "head_config": {"num_classes": 1, "num_queries": 100},
    },
    dino_teacher_config={"model_name": "vit_base"},
    num_classes=1,
)

# 2. 載入預訓練的 teachers
model.dino_teacher.load_pretrained("path/to/dino_checkpoint.pth")
model.ensemble_teachers.add_teacher(teacher1_model, weight=0.6)
model.ensemble_teachers.add_teacher(teacher2_model, weight=0.4)

# 3. 訓練
optimizer = torch.optim.AdamW(model.get_trainable_parameters(), lr=1e-4)

for epoch in range(100):
    for images, targets in train_loader:
        loss, loss_dict = model.training_step(images, targets, epoch=epoch)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

# 4. 推理
model.eval()
detections = model.inference(test_images, score_threshold=0.5)
```

### 使用訓練腳本

```bash
# 測試運行（使用 dummy 數據）
python -m mtkd_framework.train --test_run --epochs 2

# 正式訓練
python -m mtkd_framework.train \
    --config config.json \
    --dino_checkpoint path/to/dino.pth \
    --epochs 100 \
    --batch_size 8 \
    --lr 1e-4
```

---

## 模組詳解

### 1. Feature Alignment Loss (`losses/feature_alignment.py`)

用於對齊 DINO teacher 和 student 的特徵表示。

#### FeatureAlignmentLoss

```python
class FeatureAlignmentLoss(nn.Module):
    """
    特徵對齊損失

    支持多種損失類型：
    - "l2": MSE 損失
    - "cosine": Cosine Similarity 損失
    - "kl": KL Divergence 損失
    - "smooth_l1": Smooth L1 損失
    """
```

**初始化參數：**

| 參數 | 類型 | 默認值 | 說明 |
|------|------|--------|------|
| `loss_type` | str | "l2" | 損失類型 |
| `temperature` | float | 1.0 | KL divergence 溫度 |
| `normalize` | bool | True | 是否 L2 正規化 |
| `reduction` | str | "mean" | Reduction 方式 |

**使用示例：**

```python
from mtkd_framework.losses import FeatureAlignmentLoss

# 創建損失函數
loss_fn = FeatureAlignmentLoss(loss_type="cosine", normalize=True)

# 計算損失
student_feat = torch.randn(4, 768)  # [B, D]
teacher_feat = torch.randn(4, 768)
loss = loss_fn(student_feat, teacher_feat)
```

**損失公式：**

- **L2 Loss**: $L = \frac{1}{N} \sum_{i=1}^{N} ||s_i - t_i||_2^2$

- **Cosine Loss**: $L = 1 - \frac{s \cdot t}{||s|| \cdot ||t||}$

- **KL Loss**: $L = T^2 \cdot \sum_i p_t(i) \log \frac{p_t(i)}{p_s(i)}$

  其中 $p_t = \text{softmax}(t/T)$, $p_s = \text{softmax}(s/T)$

#### MultiScaleFeatureAlignmentLoss

```python
class MultiScaleFeatureAlignmentLoss(nn.Module):
    """
    多尺度特徵對齊損失

    用於對齊 FPN 等多尺度特徵。
    """
```

**使用示例：**

```python
from mtkd_framework.losses import MultiScaleFeatureAlignmentLoss

loss_fn = MultiScaleFeatureAlignmentLoss(
    num_scales=4,
    student_channels=[256, 256, 256, 256],
    teacher_channels=[768, 768, 768, 768],
    scale_weights=[1.0, 1.0, 1.0, 1.0],
)

student_feats = [torch.randn(4, 256, 56, 56) for _ in range(4)]
teacher_feats = [torch.randn(4, 768, 56, 56) for _ in range(4)]
loss = loss_fn(student_feats, teacher_feats)
```

---

### 2. Prediction Alignment Loss (`losses/prediction_alignment.py`)

用於對齊 ensemble teachers 和 student 的預測。

#### BoxAlignmentLoss

```python
class BoxAlignmentLoss(nn.Module):
    """
    Bounding Box 對齊損失

    支持：
    - "l1": L1 損失
    - "smooth_l1": Smooth L1 損失
    - "giou": Generalized IoU 損失
    - "ciou": Complete IoU 損失
    """
```

**GIoU 計算公式：**

$$\text{GIoU} = \text{IoU} - \frac{|C - (A \cup B)|}{|C|}$$

其中 $C$ 是包含 $A$ 和 $B$ 的最小閉包區域。

**CIoU 計算公式：**

$$\text{CIoU} = \text{IoU} - \frac{\rho^2(b, b^{gt})}{c^2} - \alpha v$$

其中：
- $\rho$: 中心點距離
- $c$: 閉包對角線長度
- $v$: 寬高比一致性
- $\alpha$: 權重參數

**使用示例：**

```python
from mtkd_framework.losses import BoxAlignmentLoss

loss_fn = BoxAlignmentLoss(loss_type="giou", box_format="cxcywh")

student_boxes = torch.randn(100, 4)  # [N, 4] (cx, cy, w, h)
teacher_boxes = torch.randn(100, 4)
loss = loss_fn(student_boxes, teacher_boxes)
```

#### ClassAlignmentLoss

```python
class ClassAlignmentLoss(nn.Module):
    """
    類別預測對齊損失

    使用 KL Divergence 對齊分類預測的概率分佈。
    """
```

**使用示例：**

```python
from mtkd_framework.losses import ClassAlignmentLoss

loss_fn = ClassAlignmentLoss(
    loss_type="kl",
    temperature=4.0,  # 軟化概率分佈
)

student_logits = torch.randn(100, 10)  # [N, num_classes]
teacher_logits = torch.randn(100, 10)
loss = loss_fn(student_logits, teacher_logits)
```

#### PredictionAlignmentLoss

```python
class PredictionAlignmentLoss(nn.Module):
    """
    完整的預測對齊損失

    組合 Box 對齊和 Class 對齊損失。
    """
```

**使用示例：**

```python
from mtkd_framework.losses import PredictionAlignmentLoss

loss_fn = PredictionAlignmentLoss(
    box_loss_type="giou",
    class_loss_type="kl",
    box_weight=2.0,
    class_weight=1.0,
    temperature=4.0,
)

student_pred = {
    "boxes": torch.randn(4, 100, 4),   # [B, N, 4]
    "logits": torch.randn(4, 100, 10),  # [B, N, num_classes]
}
teacher_pred = {
    "boxes": torch.randn(4, 100, 4),
    "logits": torch.randn(4, 100, 10),
}

loss, loss_dict = loss_fn(student_pred, teacher_pred)
```

---

### 3. Teacher Ensemble (`models/teacher_ensemble.py`)

管理多個教師模型並融合它們的預測。

#### WeightedBoxFusion

```python
class WeightedBoxFusion(nn.Module):
    """
    Weighted Box Fusion (WBF)

    將多個模型的 bounding box 預測進行加權融合。
    與 NMS 不同，WBF 會融合重疊的 boxes 而非抑制它們。
    """
```

**WBF 算法流程：**

1. 按信心度排序所有預測
2. 對每個類別分別處理
3. 將 IoU 超過閾值的 boxes 聚類
4. 對每個 cluster 計算加權平均 box
5. 根據參與的模型數量調整信心度

**使用示例：**

```python
from mtkd_framework.models import WeightedBoxFusion

wbf = WeightedBoxFusion(
    iou_threshold=0.55,
    weights=[0.6, 0.4],  # 兩個模型的權重
    conf_type="avg",
)

# 來自兩個模型的預測
boxes_list = [
    torch.tensor([[0.1, 0.1, 0.3, 0.3]]),  # model 1
    torch.tensor([[0.12, 0.11, 0.31, 0.32]]),  # model 2
]
scores_list = [torch.tensor([0.9]), torch.tensor([0.85])]
labels_list = [torch.tensor([0]), torch.tensor([0])]

# 融合
fused_boxes, fused_scores, fused_labels = wbf(boxes_list, scores_list, labels_list)
```

#### TeacherEnsemble

```python
class TeacherEnsemble(nn.Module):
    """
    Teacher Ensemble 模組

    組合多個預訓練的 detection teacher 模型。
    """
```

**使用示例：**

```python
from mtkd_framework.models import TeacherEnsemble

# 創建 ensemble
ensemble = TeacherEnsemble(
    teacher_weights=[0.6, 0.4],
    fusion_method="wbf",
    num_classes=1,
)

# 添加教師模型
ensemble.add_teacher(teacher1_model, weight=0.6)
ensemble.add_teacher(teacher2_model, weight=0.4)

# 獲取融合預測
predictions = ensemble(images)
# predictions = {"boxes": [...], "scores": [...], "labels": [...], "valid_mask": [...]}
```

---

### 4. Student Model (`models/student_model.py`)

學生檢測模型，包含特徵適配器。

#### FeatureAdapter

```python
class FeatureAdapter(nn.Module):
    """
    特徵適配器

    將 student 的特徵維度轉換到與 teacher 相同。

    支持類型：
    - "linear": 線性投影
    - "mlp": 兩層 MLP
    - "attention": 使用 attention 的適配
    """
```

**使用示例：**

```python
from mtkd_framework.models.student_model import FeatureAdapter

adapter = FeatureAdapter(
    student_dim=256,
    teacher_dim=768,
    adapter_type="mlp",
)

student_feat = torch.randn(4, 256)
aligned_feat = adapter(student_feat)  # [4, 768]
```

#### StudentDetector

```python
class StudentDetector(nn.Module):
    """
    Student Detector

    完整的學生檢測模型：
    - Backbone: 特徵提取
    - Neck (FPN): 多尺度融合
    - Head: 檢測預測
    - Feature Adapter: 與 teacher 對齊
    """
```

**使用示例：**

```python
from mtkd_framework.models import StudentDetector

student = StudentDetector(
    backbone_config={"backbone_type": "resnet50", "pretrained": True},
    head_config={"num_classes": 1, "num_queries": 100},
    dino_teacher_dim=768,
)

outputs = student(images, return_adapted_features=True)
# outputs = {
#     "boxes": [...],
#     "logits": [...],
#     "adapted_features": [...],  # 用於特徵對齊
# }
```

---

### 5. MTKD Model (`models/mtkd_model.py`)

整合所有組件的完整模型。

#### MTKDModel

```python
class MTKDModel(nn.Module):
    """
    Multi-Teacher Knowledge Distillation Model

    整合：
    1. DINO Feature Teacher
    2. Ensemble Detection Teachers
    3. Student Detector
    4. MTKD Loss
    """
```

**主要方法：**

| 方法 | 說明 |
|------|------|
| `forward()` | 完整前向傳播，返回輸出和損失 |
| `training_step()` | 訓練步驟，返回損失和損失字典 |
| `inference()` | 推理，返回檢測結果 |
| `get_trainable_parameters()` | 獲取可訓練參數 |

**使用示例：**

```python
from mtkd_framework import MTKDModel

model = MTKDModel(
    student_config={...},
    dino_teacher_config={"model_name": "vit_base"},
    ensemble_config={"teacher_weights": [0.6, 0.4]},
    loss_config={
        "feature_weight": 1.0,
        "prediction_weight": 2.0,
    },
)

# 訓練
loss, loss_dict = model.training_step(images, targets, epoch=epoch)

# 推理
detections = model.inference(images, score_threshold=0.5)
```

---

### 6. Combined Loss (`losses/combined_loss.py`)

組合所有損失的模組。

#### MTKDLoss

```python
class MTKDLoss(nn.Module):
    """
    組合 MTKD 損失

    包含：
    - Feature Alignment Loss
    - Prediction Alignment Loss
    - Detection Loss (可選)

    支持動態權重調整和 warmup。
    """
```

**損失公式：**

$$L_{total} = \alpha \cdot L_{feat} + \beta \cdot L_{pred} + \gamma \cdot L_{det}$$

其中：
- $L_{feat}$: 特徵對齊損失
- $L_{pred}$: 預測對齊損失
- $L_{det}$: 檢測損失（與 ground truth）
- $\alpha, \beta, \gamma$: 權重係數

**使用示例：**

```python
from mtkd_framework.losses import MTKDLoss

loss_fn = MTKDLoss(
    feature_loss_config={"loss_type": "cosine"},
    prediction_loss_config={"box_loss_type": "giou"},
    feature_weight=1.0,
    prediction_weight=2.0,
    warmup_epochs=5,
    weight_schedule="cosine",
)

total_loss, loss_dict = loss_fn(
    student_features=student_feat,
    dino_teacher_features=dino_feat,
    student_predictions=student_pred,
    ensemble_teacher_predictions=ensemble_pred,
    epoch=current_epoch,
)
```

#### UncertaintyWeightedMTKDLoss

```python
class UncertaintyWeightedMTKDLoss(MTKDLoss):
    """
    基於不確定性的自動加權損失

    使用可學習的權重參數，根據同方差不確定性自動學習損失權重。
    參考: "Multi-Task Learning Using Uncertainty to Weigh Losses"
    """
```

**損失公式：**

$$L_{total} = \sum_i \frac{1}{2\sigma_i^2} L_i + \log \sigma_i$$

其中 $\sigma_i$ 是可學習的參數。

---

## 函數參考

### 工具函數 (`utils/helpers.py`)

#### save_checkpoint

```python
def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    loss: float,
    save_path: str,
    scheduler: Optional = None,
    extra_info: Optional[Dict] = None,
):
    """保存訓練 checkpoint"""
```

#### load_checkpoint

```python
def load_checkpoint(
    model: nn.Module,
    checkpoint_path: str,
    optimizer: Optional = None,
    scheduler: Optional = None,
    strict: bool = True,
) -> Dict:
    """載入訓練 checkpoint"""
```

#### AverageMeterDict

```python
class AverageMeterDict:
    """追蹤多個指標的平均值"""

    def update(self, values: Dict[str, float], n: int = 1): ...
    def get_averages(self) -> Dict[str, float]: ...
```

#### EarlyStopping

```python
class EarlyStopping:
    """Early Stopping 機制"""

    def __call__(self, score: float) -> bool:
        """返回是否應該停止訓練"""
```

---

## 配置說明

### 默認配置

```python
config = {
    "model": {
        "num_classes": 1,
        "student_config": {
            "backbone_config": {
                "backbone_type": "resnet50",
                "pretrained": True,
            },
            "head_config": {
                "num_classes": 1,
                "num_queries": 100,
            },
        },
        "dino_teacher_config": {
            "model_name": "vit_base",
            "patch_size": 16,
        },
        "ensemble_config": {
            "fusion_method": "wbf",
            "fusion_config": {"iou_threshold": 0.55},
        },
        "loss_config": {
            "feature_weight": 1.0,
            "prediction_weight": 2.0,
            "warmup_epochs": 5,
        },
    },
    "training": {
        "epochs": 100,
        "batch_size": 8,
        "learning_rate": 1e-4,
        "weight_decay": 1e-4,
    },
}
```

---

## 訓練指南

### 完整訓練流程

```python
import torch
from mtkd_framework import MTKDModel
from mtkd_framework.train import MTKDTrainer, get_default_config

# 1. 準備配置
config = get_default_config()
config["training"]["epochs"] = 100
config["training"]["batch_size"] = 8

# 2. 創建數據載入器
train_loader = create_your_dataloader(...)
val_loader = create_your_dataloader(...)

# 3. 創建模型
model = MTKDModel(
    student_config=config["model"]["student_config"],
    dino_teacher_config=config["model"]["dino_teacher_config"],
    loss_config=config["model"]["loss_config"],
)

# 載入預訓練的 DINO
model.dino_teacher.load_pretrained("dino_vitb16.pth")

# 添加預訓練的檢測 teachers
model.ensemble_teachers.add_teacher(load_detection_model("teacher1.pth"))
model.ensemble_teachers.add_teacher(load_detection_model("teacher2.pth"))

# 4. 創建訓練器
trainer = MTKDTrainer(
    config=config,
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
)

# 5. 開始訓練
trainer.train()
```

### 訓練技巧

1. **學習率調度**: 使用 cosine annealing 配合 warmup
2. **梯度裁剪**: 設置 `gradient_clip_max_norm=1.0`
3. **混合精度**: 啟用 `mixed_precision=True` 加速訓練
4. **損失權重 warmup**: 設置 `warmup_epochs=5` 逐漸增加 KD 損失權重

---

## 自定義擴展

### 自定義 Student Backbone

```python
import torch.nn as nn
from mtkd_framework.models import StudentDetector

class CustomBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        # 你的自定義架構
        self.layers = nn.Sequential(...)
        self._out_channels = [64, 128, 256, 512]

    @property
    def out_channels_list(self):
        return self._out_channels

    def forward(self, x):
        features = {}
        # 返回多尺度特徵
        return features

# 使用自定義 backbone
student = StudentDetector(
    custom_backbone=CustomBackbone(),
    head_config={...},
)
```

### 自定義損失函數

```python
from mtkd_framework.losses import FeatureAlignmentLoss

class CustomFeatureLoss(FeatureAlignmentLoss):
    def __init__(self, *args, custom_param=1.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.custom_param = custom_param

    def forward(self, student_feat, teacher_feat, mask=None):
        # 自定義損失計算
        base_loss = super().forward(student_feat, teacher_feat, mask)
        custom_term = ...
        return base_loss + self.custom_param * custom_term
```

### 添加新的 Teacher

```python
from mtkd_framework.models import TeacherEnsemble

# 創建 ensemble
ensemble = TeacherEnsemble(num_classes=1)

# 添加自定義 teacher
class MyDetector(nn.Module):
    def forward(self, images):
        # 返回 {"boxes": ..., "scores": ..., "labels": ...}
        return predictions

teacher = MyDetector()
teacher.load_state_dict(torch.load("my_detector.pth"))
ensemble.add_teacher(teacher, weight=0.5)
```

---

## 常見問題

### Q1: 如何處理 teacher 和 student 特徵維度不匹配？

使用 `FeatureAdapter` 自動對齊維度：

```python
from mtkd_framework.models.student_model import FeatureAdapter

adapter = FeatureAdapter(
    student_dim=256,
    teacher_dim=768,
    adapter_type="mlp",
)
```

### Q2: 如何只使用特徵對齊或預測對齊？

調整損失權重：

```python
loss_config = {
    "feature_weight": 1.0,  # 只用特徵對齊
    "prediction_weight": 0.0,  # 禁用預測對齊
}
```

### Q3: 如何處理不同數量的預測？

使用 `HungarianMatchingLoss` 進行匹配：

```python
from mtkd_framework.losses import HungarianMatchingLoss

loss_fn = HungarianMatchingLoss(
    box_cost_weight=5.0,
    class_cost_weight=2.0,
)
```

---

## 參考文獻

1. DINO: Emerging Properties in Self-Supervised Vision Transformers
2. Knowledge Distillation: A Survey
3. Weighted Boxes Fusion: Ensembling boxes from different object detection models
4. Multi-Task Learning Using Uncertainty to Weigh Losses

---

## License

MIT License
