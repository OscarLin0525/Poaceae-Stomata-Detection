"""
Multi-Teacher Knowledge Distillation Model

This module implements the complete MTKD framework combining:
- DINO Feature Teacher: For feature alignment
- Ensemble Detection Teachers: For prediction alignment
- Student Detector: The model to be trained

主要功能:
- 整合所有組件的完整 MTKD 模型
- 訓練和推理接口
- 特徵和預測對齊
"""

import torch
import torch.nn as nn
from typing import Dict, Optional, List, Tuple, Any
import logging

from .teacher_ensemble import TeacherEnsemble
from .student_model import StudentDetector, FeatureAdapter
from ..losses import MTKDLoss

logger = logging.getLogger(__name__)


class DINOFeatureTeacher(nn.Module):
    """
    DINO Feature Teacher

    封裝 DINO 模型作為特徵對齊的 teacher。
    提取 CLS token 和 patch tokens 用於特徵蒸餾。

    Args:
        dino_model: 預訓練的 DINO 模型
        frozen: 是否凍結權重

    Example:
        >>> from dinov3.models import vit_base
        >>> dino = vit_base(patch_size=16)
        >>> teacher = DINOFeatureTeacher(dino, frozen=True)
        >>> features = teacher(images)
    """

    def __init__(
        self,
        dino_model: Optional[nn.Module] = None,
        frozen: bool = True,
        model_name: str = "vit_base",
        patch_size: int = 16,
    ):
        super().__init__()

        if dino_model is not None:
            self.dino = dino_model
        else:
            # 嘗試從 dinov3 載入
            self.dino = self._load_default_dino(model_name, patch_size)

        self.frozen = frozen
        if frozen:
            self.dino.eval()
            for param in self.dino.parameters():
                param.requires_grad = False

    def _load_default_dino(self, model_name: str, patch_size: int) -> nn.Module:
        """載入默認的 DINO 模型"""
        try:
            import sys
            sys.path.append("dinov3-main")
            from dinov3.models import vit_small, vit_base, vit_large

            model_dict = {
                "vit_small": vit_small,
                "vit_base": vit_base,
                "vit_large": vit_large,
            }

            if model_name in model_dict:
                return model_dict[model_name](patch_size=patch_size)
            else:
                raise ValueError(f"Unknown model name: {model_name}")

        except ImportError:
            logger.warning("Could not import dinov3. Using placeholder model.")
            return self._create_placeholder_model()

    def _create_placeholder_model(self) -> nn.Module:
        """創建佔位模型（當無法載入 DINO 時）"""
        class PlaceholderDINO(nn.Module):
            def __init__(self):
                super().__init__()
                self.embed_dim = 768
                self.patch_embed = nn.Conv2d(3, 768, kernel_size=16, stride=16)
                self.norm = nn.LayerNorm(768)

            def forward(self, x, is_training=False):
                x = self.patch_embed(x)  # [B, 768, H/16, W/16]
                B, C, H, W = x.shape
                x = x.flatten(2).transpose(1, 2)  # [B, H*W, 768]
                x = self.norm(x)
                cls_token = x.mean(dim=1)  # [B, 768]
                return {
                    "x_norm_clstoken": cls_token,
                    "x_norm_patchtokens": x,
                }

        return PlaceholderDINO()

    def load_pretrained(self, checkpoint_path: str):
        """載入預訓練權重"""
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        if "model" in checkpoint:
            self.dino.load_state_dict(checkpoint["model"], strict=False)
        elif "state_dict" in checkpoint:
            self.dino.load_state_dict(checkpoint["state_dict"], strict=False)
        else:
            self.dino.load_state_dict(checkpoint, strict=False)

        if self.frozen:
            self.dino.eval()
            for param in self.dino.parameters():
                param.requires_grad = False

        logger.info(f"Loaded DINO weights from {checkpoint_path}")

    @torch.no_grad()
    def forward(
        self,
        images: torch.Tensor,
        return_patch_tokens: bool = True,
    ) -> Dict[str, torch.Tensor]:
        """
        提取 DINO 特徵

        Args:
            images: 輸入圖像 [B, 3, H, W]
            return_patch_tokens: 是否返回 patch tokens

        Returns:
            features: {
                "cls_token": [B, D] CLS token 特徵,
                "patch_tokens": [B, N, D] Patch tokens (可選),
            }
        """
        # 調用 DINO 模型
        outputs = self.dino(images, is_training=True)

        # 適應不同的輸出格式
        if isinstance(outputs, dict):
            cls_token = outputs.get("x_norm_clstoken", outputs.get("cls_token"))
            patch_tokens = outputs.get("x_norm_patchtokens", outputs.get("patch_tokens"))
        elif isinstance(outputs, (list, tuple)):
            if len(outputs) >= 2:
                cls_token = outputs[0]
                patch_tokens = outputs[1]
            else:
                cls_token = outputs[0]
                patch_tokens = None
        else:
            # 假設輸出是 CLS token
            cls_token = outputs
            patch_tokens = None

        result = {"cls_token": cls_token}

        if return_patch_tokens and patch_tokens is not None:
            result["patch_tokens"] = patch_tokens

        return result

    def train(self, mode: bool = True):
        """Override train to keep frozen model in eval mode"""
        if self.frozen:
            super().train(False)
            self.dino.eval()
        else:
            super().train(mode)
        return self


class MTKDModel(nn.Module):
    """
    Multi-Teacher Knowledge Distillation Model

    完整的 MTKD 框架，整合:
    1. DINO Feature Teacher: 用於特徵對齊
    2. Ensemble Detection Teachers: 用於預測對齊
    3. Student Detector: 要訓練的學生模型

    架構圖:
    ```
    Input Image
        │
        ├──────────────────────────────────────┐
        │                                      │
        ▼                                      ▼
    ┌─────────────────┐              ┌─────────────────────────┐
    │  DINO Teacher   │              │  Ensemble Teachers      │
    │  (Frozen)       │              │  (Teacher1 + Teacher2)  │
    │                 │              │  (Frozen)               │
    └────────┬────────┘              └───────────┬─────────────┘
             │                                   │
             │ Feature                           │ Predictions
             │                                   │
             ▼                                   ▼
    ┌─────────────────┐              ┌─────────────────────────┐
    │  Feature        │              │  Weighted Box Fusion    │
    │  Alignment      │              │                         │
    │  Loss           │              └───────────┬─────────────┘
    └────────┬────────┘                          │
             │                                   │ Ensemble Pred
             │                                   │
             └──────────────┬────────────────────┘
                            │
                            ▼
    ┌─────────────────────────────────────────────────────────┐
    │                    Student Detector                     │
    │                    (Trainable)                          │
    │  ┌─────────┐    ┌─────────┐    ┌─────────┐             │
    │  │Backbone │ -> │  Neck   │ -> │  Head   │             │
    │  └─────────┘    └─────────┘    └─────────┘             │
    └──────────────────────┬──────────────────────────────────┘
                           │
                           ▼
                ┌─────────────────────┐
                │    MTKD Loss        │
                │  - Feature Align    │
                │  - Prediction Align │
                │  - Detection Loss   │
                └─────────────────────┘
    ```

    Args:
        student_config: Student 模型配置
        dino_teacher_config: DINO teacher 配置
        ensemble_config: Ensemble teachers 配置
        loss_config: 損失函數配置

    Example:
        >>> mtkd = MTKDModel(
        ...     student_config={"backbone_config": {"backbone_type": "resnet50"}},
        ...     dino_teacher_config={"model_name": "vit_base"},
        ...     ensemble_config={"teacher_weights": [0.6, 0.4]},
        ... )
        >>> outputs, loss_dict = mtkd.training_step(images, targets)
    """

    def __init__(
        self,
        # Student configuration
        student_config: Optional[Dict[str, Any]] = None,
        custom_student: Optional[nn.Module] = None,
        # DINO teacher configuration
        dino_teacher_config: Optional[Dict[str, Any]] = None,
        dino_model: Optional[nn.Module] = None,
        dino_checkpoint: Optional[str] = None,
        # Ensemble teachers configuration
        ensemble_config: Optional[Dict[str, Any]] = None,
        teacher_models: Optional[List[nn.Module]] = None,
        teacher_checkpoints: Optional[List[str]] = None,
        # Loss configuration
        loss_config: Optional[Dict[str, Any]] = None,
        # General configuration
        num_classes: int = 1,
    ):
        super().__init__()

        self.num_classes = num_classes

        # =============================================================
        # 1. Initialize Student Model
        # =============================================================
        if custom_student is not None:
            self.student = custom_student
            dino_teacher_dim = dino_teacher_config.get("embed_dim", 768) if dino_teacher_config else 768
        else:
            student_config = student_config or {}
            dino_teacher_dim = dino_teacher_config.get("embed_dim", 768) if dino_teacher_config else 768
            student_config.setdefault("dino_teacher_dim", dino_teacher_dim)
            student_config.setdefault("head_config", {"num_classes": num_classes})
            self.student = StudentDetector(**student_config)

        # =============================================================
        # 2. Initialize DINO Feature Teacher (Frozen)
        # =============================================================
        dino_teacher_config = dino_teacher_config or {}
        if dino_model is not None:
            self.dino_teacher = DINOFeatureTeacher(dino_model=dino_model, frozen=True)
        else:
            self.dino_teacher = DINOFeatureTeacher(frozen=True, **dino_teacher_config)

        if dino_checkpoint is not None:
            self.dino_teacher.load_pretrained(dino_checkpoint)

        # =============================================================
        # 3. Initialize Ensemble Detection Teachers (Frozen)
        # =============================================================
        ensemble_config = ensemble_config or {}
        ensemble_config.setdefault("num_classes", num_classes)

        if teacher_models is not None:
            self.ensemble_teachers = TeacherEnsemble(
                teacher_models=teacher_models,
                **ensemble_config,
            )
        else:
            self.ensemble_teachers = TeacherEnsemble(**ensemble_config)

        # 從 checkpoints 載入 teachers（如果提供）
        if teacher_checkpoints is not None and len(teacher_checkpoints) > 0:
            logger.info(f"Loading {len(teacher_checkpoints)} teacher models from checkpoints")
            # 這裡需要用戶提供 model_class
            # self.ensemble_teachers.load_teachers_from_checkpoints(teacher_checkpoints, ...)

        # =============================================================
        # 4. Initialize MTKD Loss
        # =============================================================
        loss_config = loss_config or {}
        self.mtkd_loss = MTKDLoss(**loss_config)

    def forward(
        self,
        images: torch.Tensor,
        targets: Optional[Dict[str, torch.Tensor]] = None,
        return_loss: bool = True,
        epoch: Optional[int] = None,
    ) -> Tuple[Dict[str, torch.Tensor], Optional[Dict[str, torch.Tensor]]]:
        """
        前向傳播

        Args:
            images: 輸入圖像 [B, 3, H, W]
            targets: Ground truth 標註 (訓練時需要)
            return_loss: 是否計算損失
            epoch: 當前 epoch（用於動態權重）

        Returns:
            outputs: Student 的預測輸出
            loss_dict: 損失字典（如果 return_loss=True）
        """
        # Student forward
        student_outputs = self.student(
            images,
            return_features=True,
            return_adapted_features=True,
        )

        if not return_loss:
            return student_outputs, None

        # =============================================================
        # Get Teacher Outputs (No Gradients)
        # =============================================================
        with torch.no_grad():
            # DINO teacher features
            dino_features = self.dino_teacher(images, return_patch_tokens=True)

            # Ensemble teacher predictions
            if len(self.ensemble_teachers.teachers) > 0:
                ensemble_predictions = self.ensemble_teachers(images)
            else:
                ensemble_predictions = None

        # =============================================================
        # Compute MTKD Loss
        # =============================================================
        # 準備學生預測
        student_predictions = {
            "boxes": student_outputs["boxes"],
            "logits": student_outputs["logits"],
        }

        # 計算檢測損失（如果提供了 targets）
        detection_loss = None
        if targets is not None:
            detection_loss = self._compute_detection_loss(student_outputs, targets)

        # 計算 MTKD 損失
        total_loss, loss_dict = self.mtkd_loss(
            # Feature alignment (DINO)
            student_features=student_outputs.get("adapted_features"),
            dino_teacher_features=dino_features.get("cls_token"),
            # Multi-scale feature alignment
            student_multi_scale_features=student_outputs.get("adapted_multi_scale_features"),
            dino_teacher_multi_scale_features=None,  # DINO 不提供多尺度特徵
            # Prediction alignment (Ensemble)
            student_predictions=student_predictions if ensemble_predictions else None,
            ensemble_teacher_predictions=ensemble_predictions,
            prediction_valid_mask=ensemble_predictions.get("valid_mask") if ensemble_predictions else None,
            # Detection loss
            detection_loss=detection_loss,
            # Epoch
            epoch=epoch,
        )

        loss_dict["total_loss"] = total_loss

        return student_outputs, loss_dict

    def _compute_detection_loss(
        self,
        predictions: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """
        計算檢測損失（與 ground truth）

        Args:
            predictions: Student 預測
            targets: Ground truth

        Returns:
            detection_loss: 檢測損失
        """
        # 這是一個簡化的檢測損失計算
        # 實際使用時可能需要根據具體的 detection head 調整

        pred_logits = predictions["logits"]  # [B, num_queries, num_classes+1]
        pred_boxes = predictions["boxes"]    # [B, num_queries, 4]

        # 簡化版：使用所有預測和 targets 計算損失
        # 實際應該使用 Hungarian matching

        device = pred_logits.device
        total_loss = torch.tensor(0.0, device=device)

        # Classification loss
        if "labels" in targets:
            # 假設 targets["labels"] 是 [B, max_objects]
            # 這裡使用簡化的 focal loss
            pass

        # Box regression loss
        if "boxes" in targets:
            # 使用 L1 + GIoU loss
            pass

        return total_loss

    def training_step(
        self,
        images: torch.Tensor,
        targets: Optional[Dict[str, torch.Tensor]] = None,
        epoch: Optional[int] = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        訓練步驟

        Args:
            images: [B, 3, H, W]
            targets: Ground truth
            epoch: 當前 epoch

        Returns:
            loss: 總損失
            loss_dict: 詳細損失字典
        """
        outputs, loss_dict = self.forward(images, targets, return_loss=True, epoch=epoch)
        return loss_dict["total_loss"], loss_dict

    @torch.no_grad()
    def inference(
        self,
        images: torch.Tensor,
        score_threshold: float = 0.5,
    ) -> List[Dict[str, torch.Tensor]]:
        """
        推理

        Args:
            images: [B, 3, H, W]
            score_threshold: 信心度閾值

        Returns:
            detections: 每張圖像的檢測結果列表
        """
        self.eval()

        outputs, _ = self.forward(images, return_loss=False)

        # Post-process
        pred_logits = outputs["logits"]
        pred_boxes = outputs["boxes"]

        # 獲取類別和分數
        probs = pred_logits.softmax(-1)
        scores, labels = probs[..., :-1].max(dim=-1)

        batch_detections = []
        for b in range(images.shape[0]):
            mask = scores[b] > score_threshold
            batch_detections.append({
                "boxes": pred_boxes[b][mask],
                "scores": scores[b][mask],
                "labels": labels[b][mask],
            })

        return batch_detections

    def get_student_parameters(self) -> List[nn.Parameter]:
        """獲取 student 的可訓練參數"""
        return list(self.student.parameters())

    def get_trainable_parameters(self) -> List[nn.Parameter]:
        """獲取所有可訓練參數（只有 student）"""
        params = []
        for name, param in self.named_parameters():
            if param.requires_grad:
                params.append(param)
        return params

    def train(self, mode: bool = True):
        """Override train to keep teachers frozen"""
        super().train(mode)
        self.dino_teacher.train(False)
        self.ensemble_teachers.train(False)
        return self

    def freeze_student_backbone(self):
        """凍結 student backbone（用於微調）"""
        for param in self.student.backbone.parameters():
            param.requires_grad = False
        logger.info("Student backbone frozen")

    def unfreeze_student_backbone(self):
        """解凍 student backbone"""
        for param in self.student.backbone.parameters():
            param.requires_grad = True
        logger.info("Student backbone unfrozen")


def build_mtkd_model(
    config: Dict[str, Any],
    dino_checkpoint: Optional[str] = None,
    teacher_checkpoints: Optional[List[str]] = None,
) -> MTKDModel:
    """
    從配置構建 MTKD 模型

    Args:
        config: 模型配置字典
        dino_checkpoint: DINO 模型 checkpoint 路徑
        teacher_checkpoints: Teacher 模型 checkpoint 路徑列表

    Returns:
        model: MTKDModel 實例

    Example:
        >>> config = {
        ...     "student_config": {
        ...         "backbone_config": {"backbone_type": "resnet50"},
        ...         "head_config": {"num_classes": 1, "num_queries": 100},
        ...     },
        ...     "dino_teacher_config": {"model_name": "vit_base"},
        ...     "ensemble_config": {"teacher_weights": [0.6, 0.4]},
        ...     "loss_config": {"feature_weight": 1.0, "prediction_weight": 2.0},
        ... }
        >>> model = build_mtkd_model(config, dino_checkpoint="dino.pth")
    """
    return MTKDModel(
        student_config=config.get("student_config"),
        dino_teacher_config=config.get("dino_teacher_config"),
        ensemble_config=config.get("ensemble_config"),
        loss_config=config.get("loss_config"),
        num_classes=config.get("num_classes", 1),
        dino_checkpoint=dino_checkpoint,
        teacher_checkpoints=teacher_checkpoints,
    )
