"""
YOLO Wrappers for MTKD

This module provides:
1. YOLOStudentDetector: YOLOv11 student wrapper with P3/P4/P5 feature access
2. YOLODetectionTeacher: YOLOv8/YOLOv11 teacher wrapper for ensemble distillation
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple, Literal

import torch
import torch.nn as nn

from .student_model import FeatureAdapter

logger = logging.getLogger(__name__)


def _decode_ultralytics_output(
    output: object,
) -> Tuple[Optional[torch.Tensor], Optional[List[torch.Tensor]]]:
    """Normalize Ultralytics forward output to (decoded_pred, raw_levels)."""
    if isinstance(output, tuple):
        if len(output) >= 2 and isinstance(output[1], list):
            return output[0], output[1]
        if len(output) == 1 and isinstance(output[0], list):
            return None, output[0]
    if isinstance(output, list):
        return None, output
    return None, None


def _pred_to_boxes_and_logits(
    pred: torch.Tensor,
    image_h: int,
    image_w: int,
    num_classes: Optional[int] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Convert decoded YOLO prediction [B, 4+nc, N] to MTKD format.

    Returns:
        boxes: [B, N, 4] normalized cxcywh
        logits: [B, N, nc+1] class logits (+ background)
        scores: [B, N]
        labels: [B, N]
    """
    if pred.dim() != 3 or pred.shape[1] < 5:
        raise ValueError(f"Unexpected decoded prediction shape: {tuple(pred.shape)}")

    # Ultralytics decoded boxes are in pixel cxcywh.
    boxes = pred[:, :4, :].permute(0, 2, 1).contiguous()
    scale = boxes.new_tensor([image_w, image_h, image_w, image_h]).view(1, 1, 4)
    boxes = (boxes / scale).clamp(0.0, 1.0)

    # Decoded cls scores are probabilities in [0, 1].
    cls_prob = pred[:, 4:, :].permute(0, 2, 1).contiguous().clamp(1e-6, 1 - 1e-6)
    if num_classes is not None:
        if cls_prob.shape[-1] >= num_classes:
            cls_prob = cls_prob[..., :num_classes]
        else:
            pad = cls_prob.new_full(
                (cls_prob.shape[0], cls_prob.shape[1], num_classes - cls_prob.shape[-1]),
                1e-6,
            )
            cls_prob = torch.cat([cls_prob, pad], dim=-1)
    cls_logits = torch.log(cls_prob / (1.0 - cls_prob))

    scores, labels = cls_prob.max(dim=-1)
    bg_prob = (1.0 - scores.unsqueeze(-1)).clamp(1e-6, 1 - 1e-6)
    bg_logit = torch.log(bg_prob / (1.0 - bg_prob))
    logits = torch.cat([cls_logits, bg_logit], dim=-1)

    return boxes, logits, scores, labels


class YOLOStudentDetector(nn.Module):
    """
    YOLO student wrapper for MTKD.

    - Loads YOLOv11 weights
    - Captures P3/P4/P5 features from Detect pre-hook
    - Exposes MTKD-compatible outputs: boxes/logits/adapted_features
    """

    def __init__(
        self,
        weights: str = "yolo11s.pt",
        dino_teacher_dim: int = 768,
        feature_level: Literal["p3", "p4", "p5"] = "p4",
        adapter_type: Literal["linear", "mlp", "attention"] = "mlp",
        num_classes: Optional[int] = None,
    ):
        super().__init__()
        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise ImportError("ultralytics is required for YOLO wrappers") from exc

        yolo_obj = YOLO(weights)
        self.det_model = yolo_obj.model
        # Ultralytics checkpoints may load with frozen params; ensure student is trainable.
        for p in self.det_model.parameters():
            p.requires_grad = True
        self.dino_teacher_dim = dino_teacher_dim
        self.feature_level = feature_level.lower()
        self.num_classes = num_classes
        self.feature_level_to_idx = {"p3": 0, "p4": 1, "p5": 2}
        if self.feature_level not in self.feature_level_to_idx:
            raise ValueError(f"Unsupported feature level: {feature_level}")

        self.adapter_type = adapter_type
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.feature_adapter: Optional[FeatureAdapter] = None

        self._last_neck_features: Optional[List[torch.Tensor]] = None
        self._register_detect_prehook()

    def _register_detect_prehook(self) -> None:
        detect_module = None
        if hasattr(self.det_model, "model") and len(self.det_model.model) > 0:
            # Usually the last module is Detect.
            for module in reversed(self.det_model.model):
                if module.__class__.__name__.lower() == "detect":
                    detect_module = module
                    break
        if detect_module is None:
            raise RuntimeError("Could not locate Detect module in YOLO model")

        def _cache_neck_input(_module, inputs):
            if not inputs:
                self._last_neck_features = None
                return
            x = inputs[0]
            if isinstance(x, (list, tuple)):
                self._last_neck_features = [feat for feat in x]
            else:
                self._last_neck_features = None

        detect_module.register_forward_pre_hook(_cache_neck_input)

    def _ensure_feature_adapter(self, in_channels: int, device: torch.device) -> None:
        if self.feature_adapter is None:
            self.feature_adapter = FeatureAdapter(
                student_dim=in_channels,
                teacher_dim=self.dino_teacher_dim,
                adapter_type=self.adapter_type,
            ).to(device)

    def _forward_decoded(self, images: torch.Tensor) -> Tuple[torch.Tensor, Optional[List[torch.Tensor]], Optional[List[torch.Tensor]]]:
        """
        Run YOLO in eval forward format to get decoded predictions with gradients enabled.
        """
        self._last_neck_features = None

        was_training = self.det_model.training
        self.det_model.eval()
        output = self.det_model(images)
        if was_training:
            self.det_model.train()

        pred, raw_levels = _decode_ultralytics_output(output)
        if pred is None:
            raise RuntimeError("YOLO output does not contain decoded predictions in eval forward path")

        return pred, raw_levels, self._last_neck_features

    def forward(
        self,
        images: torch.Tensor,
        return_features: bool = False,
        return_adapted_features: bool = True,
    ) -> Dict[str, torch.Tensor]:
        pred, raw_levels, neck_features = self._forward_decoded(images)

        image_h, image_w = images.shape[-2:]
        boxes, logits, scores, labels = _pred_to_boxes_and_logits(
            pred, image_h, image_w, num_classes=self.num_classes
        )

        outputs: Dict[str, torch.Tensor] = {
            "boxes": boxes,
            "logits": logits,
        }

        pyramid = None
        if neck_features is not None and len(neck_features) >= 3:
            pyramid = neck_features[:3]
        elif raw_levels is not None and len(raw_levels) >= 3:
            # Fallback: use detect outputs when neck input is unavailable.
            pyramid = raw_levels[:3]

        if return_features and pyramid is not None:
            outputs["p3_features"] = pyramid[0]
            outputs["p4_features"] = pyramid[1]
            outputs["p5_features"] = pyramid[2]

        if return_adapted_features and pyramid is not None:
            feat = pyramid[self.feature_level_to_idx[self.feature_level]]
            self._ensure_feature_adapter(feat.shape[1], feat.device)
            pooled = self.global_pool(feat).flatten(1)
            outputs["adapted_features"] = self.feature_adapter(pooled)

        # Keep score/label for optional debugging.
        outputs["scores"] = scores
        outputs["labels"] = labels
        return outputs

    @torch.no_grad()
    def get_detection_output(self, images: torch.Tensor) -> Dict[str, torch.Tensor]:
        pred, _, _ = self._forward_decoded(images)
        image_h, image_w = images.shape[-2:]
        boxes, _, scores, labels = _pred_to_boxes_and_logits(
            pred, image_h, image_w, num_classes=self.num_classes
        )
        return {"boxes": boxes, "scores": scores, "labels": labels}


class YOLODetectionTeacher(nn.Module):
    """
    YOLO detection teacher wrapper for MTKD teacher ensemble.
    """

    def __init__(
        self,
        weights: str = "yolov8s.pt",
        score_threshold: float = 0.001,
        max_detections: int = 300,
        num_classes: Optional[int] = None,
    ):
        super().__init__()
        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise ImportError("ultralytics is required for YOLO wrappers") from exc

        yolo_obj = YOLO(weights)
        self.det_model = yolo_obj.model
        self.score_threshold = score_threshold
        self.max_detections = max_detections
        self.num_classes = num_classes

        for p in self.det_model.parameters():
            p.requires_grad = False
        self.det_model.eval()

    @torch.no_grad()
    def forward(self, images: torch.Tensor) -> Dict[str, torch.Tensor]:
        was_training = self.det_model.training
        self.det_model.eval()
        output = self.det_model(images)
        if was_training:
            self.det_model.train()

        pred, _ = _decode_ultralytics_output(output)
        if pred is None:
            raise RuntimeError("YOLO teacher output does not contain decoded predictions")

        image_h, image_w = images.shape[-2:]
        boxes, _, scores, labels = _pred_to_boxes_and_logits(
            pred, image_h, image_w, num_classes=self.num_classes
        )

        # Apply per-image threshold and top-k, then pad to batch tensor.
        batch_boxes: List[torch.Tensor] = []
        batch_scores: List[torch.Tensor] = []
        batch_labels: List[torch.Tensor] = []
        max_keep = 1

        for b in range(boxes.shape[0]):
            mask = scores[b] >= self.score_threshold
            if self.num_classes is not None:
                mask = mask & (labels[b] < self.num_classes)
            b_boxes = boxes[b][mask]
            b_scores = scores[b][mask]
            b_labels = labels[b][mask]

            if b_scores.numel() > self.max_detections:
                topk = torch.topk(b_scores, k=self.max_detections)
                keep = topk.indices
                b_boxes = b_boxes[keep]
                b_scores = b_scores[keep]
                b_labels = b_labels[keep]

            max_keep = max(max_keep, b_scores.numel())
            batch_boxes.append(b_boxes)
            batch_scores.append(b_scores)
            batch_labels.append(b_labels)

        padded_boxes = boxes.new_zeros((boxes.shape[0], max_keep, 4))
        padded_scores = scores.new_full((scores.shape[0], max_keep), -1.0)
        padded_labels = labels.new_zeros((labels.shape[0], max_keep), dtype=torch.long)

        for b, (b_boxes, b_scores, b_labels) in enumerate(zip(batch_boxes, batch_scores, batch_labels)):
            n = b_scores.numel()
            if n == 0:
                continue
            padded_boxes[b, :n] = b_boxes
            padded_scores[b, :n] = b_scores
            padded_labels[b, :n] = b_labels

        return {
            "boxes": padded_boxes,
            "scores": padded_scores,
            "labels": padded_labels,
        }
