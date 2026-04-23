"""
YOLO Wrappers for MTKD

This module provides:
1. YOLOStudentDetector: YOLOv11 student wrapper with P3/P4/P5 feature access
2. YOLODetectionTeacher: YOLOv8/YOLOv11 teacher wrapper for ensemble distillation
"""

from __future__ import annotations

import copy
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple, Literal

import torch
import torch.nn as nn

from .student_model import FeatureAdapter

logger = logging.getLogger(__name__)


def _decode_ultralytics_output(
    output: object,
) -> Tuple[Optional[torch.Tensor], Optional[List[torch.Tensor]], Dict[str, Any]]:
    """Normalize Ultralytics forward output to (decoded_pred, raw_levels, extra)."""
    extra: Dict[str, Any] = {}
    if isinstance(output, tuple):
        if len(output) >= 2 and isinstance(output[1], list):
            return output[0], output[1], extra
        if len(output) >= 2 and isinstance(output[1], dict):
            extra = output[1]
            feats = extra.get("feats")
            raw_levels = feats if isinstance(feats, list) else None
            decoded = output[0] if isinstance(output[0], torch.Tensor) else None
            return decoded, raw_levels, extra
        # ultralytics >=8.4: (decoded_pred, extra_dict)
        if (len(output) >= 2
                and isinstance(output[0], torch.Tensor)
                and output[0].ndim == 3):
            return output[0], None, extra
        if len(output) == 1 and isinstance(output[0], list):
            return None, output[0], extra
    if isinstance(output, list):
        return None, output, extra
    if isinstance(output, dict):
        extra = output
        feats = extra.get("feats")
        raw_levels = feats if isinstance(feats, list) else None
        return None, raw_levels, extra
    if isinstance(output, torch.Tensor) and output.ndim == 3:
        return output, None, extra
    return None, None, extra


def _pred_to_boxes_and_logits(
    pred: torch.Tensor,
    image_h: int,
    image_w: int,
    num_classes: Optional[int] = None,
    task: str = "detect",
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
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

    task = task.lower()
    is_obb = task == "obb"

    def _prob_like_ratio(x: Optional[torch.Tensor]) -> float:
        if x is None or x.numel() == 0:
            return -1.0
        return float(((x >= 0.0) & (x <= 1.0)).float().mean().item())

    if is_obb:
        # Ultralytics OBB decoded tensors can appear in two layouts:
        # A) [x, y, w, h, angle, cls...]
        # B) [x, y, w, h, cls..., angle]
        c = int(pred.shape[1])
        if num_classes is None:
            num_classes = max(1, c - 5)

        if c < num_classes + 5:
            raise ValueError(
                f"Unexpected OBB decoded channels: C={c}, num_classes={num_classes}"
            )

        cls_a = pred[:, 5:5 + num_classes, :] if c >= 5 + num_classes else None
        angle_a = pred[:, 4:5, :]
        score_a = _prob_like_ratio(cls_a) + (1.0 - _prob_like_ratio(angle_a))

        cls_b = pred[:, 4:4 + num_classes, :] if c >= 4 + num_classes else None
        angle_b = (
            pred[:, 4 + num_classes:4 + num_classes + 1, :]
            if c >= 4 + num_classes + 1
            else None
        )
        score_b = _prob_like_ratio(cls_b) + (1.0 - _prob_like_ratio(angle_b))

        use_layout_b = cls_b is not None and angle_b is not None and score_b >= score_a
        if use_layout_b:
            boxes = pred[:, :4, :].permute(0, 2, 1).contiguous()
            angles = angle_b.permute(0, 2, 1).contiguous()
            cls_raw = cls_b
        else:
            boxes_all = pred[:, :5, :].permute(0, 2, 1).contiguous()
            boxes = boxes_all[..., :4]
            angles = boxes_all[..., 4:5]
            if cls_a is None:
                raise ValueError(
                    f"Cannot parse OBB class channels from decoded shape {tuple(pred.shape)}"
                )
            cls_raw = cls_a
    else:
        boxes = pred[:, :4, :].permute(0, 2, 1).contiguous()
        angles = None
        if num_classes is None:
            num_classes = int(pred.shape[1] - 4)
        cls_raw = pred[:, 4:4 + num_classes, :]

    scale = boxes.new_tensor([image_w, image_h, image_w, image_h]).view(1, 1, 4)
    boxes = (boxes / scale).clamp(0.0, 1.0)

    cls_raw = cls_raw.permute(0, 2, 1).contiguous()

    # Ultralytics heads can expose either probabilities ([0, 1]) or logits.
    # If values fall outside [0, 1], interpret them as logits.
    if float(cls_raw.min()) < 0.0 or float(cls_raw.max()) > 1.0:
        cls_prob = cls_raw.sigmoid()
    else:
        cls_prob = cls_raw
    cls_prob = cls_prob.clamp(1e-6, 1 - 1e-6)
    cls_logits = torch.log(cls_prob / (1.0 - cls_prob))

    scores, labels = cls_prob.max(dim=-1)
    bg_prob = (1.0 - scores.unsqueeze(-1)).clamp(1e-6, 1 - 1e-6)
    bg_logit = torch.log(bg_prob / (1.0 - bg_prob))
    logits = torch.cat([cls_logits, bg_logit], dim=-1)

    return boxes, logits, scores, labels, angles


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
        self.task = str(getattr(yolo_obj, "task", "detect")).lower()
        self.box_dim = 5 if self.task == "obb" else 4
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
            # Usually the last module is Detect (or OBB, which subclasses Detect).
            try:
                from ultralytics.nn.modules.head import Detect
            except Exception:  # pragma: no cover - compatibility fallback
                Detect = None
            for module in reversed(self.det_model.model):
                if Detect is not None and isinstance(module, Detect):
                    detect_module = module
                    break
                if module.__class__.__name__.lower() in {"detect", "obb"}:
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

    def _forward_decoded(
        self,
        images: torch.Tensor,
    ) -> Tuple[torch.Tensor, Optional[List[torch.Tensor]], Optional[List[torch.Tensor]], Dict[str, Any]]:
        """
        Run YOLO in eval forward format to get decoded predictions with gradients enabled.
        """
        self._last_neck_features = None

        was_training = self.det_model.training
        self.det_model.eval()
        output = self.det_model(images)
        if was_training:
            self.det_model.train()

        pred, raw_levels, extra = _decode_ultralytics_output(output)
        if pred is None:
            raise RuntimeError("YOLO output does not contain decoded predictions in eval forward path")

        return pred, raw_levels, self._last_neck_features, extra

    def forward(
        self,
        images: torch.Tensor,
        return_features: bool = False,
        return_adapted_features: bool = True,
    ) -> Dict[str, torch.Tensor]:
        pred, raw_levels, neck_features, extra = self._forward_decoded(images)

        image_h, image_w = images.shape[-2:]
        boxes, logits, scores, labels, angles = _pred_to_boxes_and_logits(
            pred, image_h, image_w, num_classes=self.num_classes, task=self.task
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
        elif isinstance(extra.get("feats"), list) and len(extra["feats"]) >= 3:
            pyramid = extra["feats"][:3]

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
        if angles is not None:
            outputs["angles"] = angles
        return outputs

    # ------------------------------------------------------------------
    # Training-mode forward (for v8DetectionLoss)
    # ------------------------------------------------------------------
    def _ensure_criterion(self):
        """Lazy-init YOLO detection loss using the model's own ``init_criterion``."""
        if hasattr(self, "_criterion") and self._criterion is not None:
            return
        # v8DetectionLoss reads model.args for hyp (box/cls/dfl gains).
        # In ultralytics >=8.4, model.args is a plain dict missing box/cls/dfl.
        # Replace it with a full IterableSimpleNamespace from get_cfg().
        try:
            from ultralytics.cfg import get_cfg
            cfg = get_cfg()  # IterableSimpleNamespace with all defaults
            # Merge any existing model-specific args
            existing = getattr(self.det_model, "args", None) or {}
            if isinstance(existing, dict):
                for k, v in existing.items():
                    setattr(cfg, k, v)
            self.det_model.args = cfg
        except ImportError:
            from types import SimpleNamespace
            self.det_model.args = SimpleNamespace(box=7.5, cls=0.5, dfl=1.5)
        self._criterion = self.det_model.init_criterion()

    def forward_train_raw(
        self,
        images: torch.Tensor,
    ) -> Tuple[object, Dict[str, torch.Tensor]]:
        """
        Run the student YOLO in **training mode** to obtain raw predictions
        (before decoding) and capture neck spatial features via the Detect
        pre-hook.

        This **does not** compute any loss — call :meth:`compute_loss`
        separately with the desired target batch(es).

        Args:
            images: ``[B, 3, H, W]``

        Returns:
            raw_preds:  Whatever the Detect head returns in train mode
                        (list of ``[B, no, H_i, W_i]`` tensors).
            features:   Dict with ``p3_features``, ``p4_features``,
                        ``p5_features`` spatial feature maps.
        """
        self._last_neck_features = None
        self.det_model.train()
        # In train mode the Detect head returns raw predictions (not decoded).
        # Use direct __call__ (nn.Module forward), NOT .predict() which is
        # the high-level Ultralytics inference API with NMS/postprocessing.
        raw_preds = self.det_model(images)

        features: Dict[str, torch.Tensor] = {}
        if self._last_neck_features is not None and len(self._last_neck_features) >= 3:
            features["p3_features"] = self._last_neck_features[0]
            features["p4_features"] = self._last_neck_features[1]
            features["p5_features"] = self._last_neck_features[2]
        elif isinstance(raw_preds, dict) and isinstance(raw_preds.get("feats"), list):
            feats = raw_preds["feats"]
            if len(feats) >= 3:
                features["p3_features"] = feats[0]
                features["p4_features"] = feats[1]
                features["p5_features"] = feats[2]

        return raw_preds, features

    def compute_loss(
        self,
        raw_preds: object,
        yolo_batch: Dict[str, torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute YOLO detection loss from raw predictions and a target batch.

        Can be called **multiple times** with different target batches on the
        same ``raw_preds`` (e.g. once for GT, once for pseudo-labels).

        Args:
            raw_preds:  Output of :meth:`forward_train_raw`.
            yolo_batch: Dict with ``batch_idx`` ``[N]``, ``cls`` ``[N, 1]``,
                        ``bboxes`` ``[N, 4 or 5]``  (normalised cxcywh[+angle]).

        Returns:
            loss:       Scalar (= ``sum(box + cls + dfl) * batch_size``).
            loss_items: Detached ``[box, cls, dfl]`` for logging.
        """
        self._ensure_criterion()
        loss, loss_items = self._criterion(raw_preds, yolo_batch)
        # ultralytics >=8.4 returns loss as [box, cls, dfl] (3 elements);
        # sum to get the scalar total.
        if loss.ndim > 0 and loss.numel() > 1:
            loss = loss.sum()
        return loss, loss_items

    def get_loss_hyp(self, key: str) -> float:
        """Read a loss hyperparameter (e.g. ``'box'``, ``'cls'``, ``'dfl'``)."""
        self._ensure_criterion()
        return getattr(self._criterion.hyp, key)

    @torch.no_grad()
    def get_detection_output(self, images: torch.Tensor) -> Dict[str, torch.Tensor]:
        pred, _, _, _ = self._forward_decoded(images)
        image_h, image_w = images.shape[-2:]
        boxes, _, scores, labels, angles = _pred_to_boxes_and_logits(
            pred, image_h, image_w, num_classes=self.num_classes, task=self.task
        )
        if angles is not None:
            boxes = torch.cat([boxes, angles], dim=-1)
        return {"boxes": boxes, "scores": scores, "labels": labels}

    def export_ultralytics_pt(
        self,
        save_path: str,
        num_classes: int = 1,
        class_names: Optional[Dict[int, str]] = None,
        epoch: int = -1,
        best_fitness: Optional[float] = None,
    ) -> str:
        """
        Export the trained YOLO student as a standard ultralytics ``.pt`` file.

        The exported file can be loaded directly with:
            ``from ultralytics import YOLO; model = YOLO('best.pt')``

        Then used for inference:
            ``results = model.predict(source='images/', conf=0.25)``

        Args:
            save_path: Output ``.pt`` path.
            num_classes: Number of classes in the detection head.
            class_names: Optional mapping ``{0: 'stomata', ...}``.
            epoch: Training epoch to record.
            best_fitness: Best fitness to record.

        Returns:
            The save_path.
        """
        if class_names is None:
            class_names = {i: f"class_{i}" for i in range(num_classes)}

        # Build a clean copy by creating a fresh model and loading state_dict.
        # This avoids pickling issues with forward hooks.
        from ultralytics.nn.tasks import DetectionModel as _DM, OBBModel as _OM

        yaml_cfg = getattr(self.det_model, "yaml", None)
        if yaml_cfg is None:
            raise RuntimeError("det_model has no .yaml attribute — cannot export")

        if self.task == "obb":
            clean_model = _OM(cfg=yaml_cfg, nc=num_classes, verbose=False)
        else:
            clean_model = _DM(cfg=yaml_cfg, nc=num_classes, verbose=False)
        # Load compatible tensors and partially map class heads when nc differs.
        # PyTorch strict=False still errors on same-key shape mismatch, so we
        # build a filtered/mapped dict explicitly.
        src_state = self.det_model.state_dict()
        dst_state = clean_model.state_dict()
        compatible_state = {}
        for k, src_v in src_state.items():
            if k not in dst_state:
                continue
            dst_v = dst_state[k]
            if dst_v.shape == src_v.shape:
                compatible_state[k] = src_v
                continue

            # Handle reduced class heads, e.g. src [80, C, 1, 1] -> dst [3, C, 1, 1]
            # or src [80] -> dst [3]. Copy leading channels/classes.
            if (
                src_v.ndim == dst_v.ndim
                and src_v.shape[0] >= dst_v.shape[0]
                and list(src_v.shape[1:]) == list(dst_v.shape[1:])
            ):
                mapped = dst_v.clone()
                mapped[: dst_v.shape[0]] = src_v[: dst_v.shape[0]]
                compatible_state[k] = mapped

        clean_model.load_state_dict(compatible_state, strict=False)
        clean_model = clean_model.cpu().half()
        clean_model.names = class_names
        clean_model.nc = num_classes

        if hasattr(clean_model, "criterion"):
            clean_model.criterion = None

        ckpt = {
            "epoch": epoch,
            "best_fitness": best_fitness,
            "model": clean_model,
            "ema": None,
            "updates": None,
            "optimizer": None,
            "train_args": {
                "task": self.task,
                "model": "yolo11s.yaml",
                "data": "stomata.yaml",
                "imgsz": 640,
            },
            "train_metrics": {},
            "train_results": None,
            "date": datetime.now().isoformat(),
            "version": "8.4.19",
            "license": "AGPL-3.0 License (https://ultralytics.com/license)",
            "docs": "https://docs.ultralytics.com",
        }

        torch.save(ckpt, save_path)
        return save_path


class YOLODetectionTeacher(nn.Module):
    """
    YOLO detection teacher wrapper for MTKD teacher ensemble.
    """

    def __init__(
        self,
        weights: str = "yolov8s.pt",
        score_threshold: Optional[float] = None,
        nms_iou: Optional[float] = None,
        max_detections: int = 300,
        num_classes: Optional[int] = None,
    ):
        super().__init__()
        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise ImportError("ultralytics is required for YOLO wrappers") from exc

        yolo_obj = YOLO(weights)
        # Keep Ultralytics high-level runner out of nn.Module children.
        # Otherwise parent ``model.train()`` may recurse into ``YOLO.train``
        # with a boolean ``mode`` and trigger a TypeError.
        object.__setattr__(self, "_yolo_infer", yolo_obj)
        self.det_model = yolo_obj.model
        self.task = str(getattr(yolo_obj, "task", "detect")).lower()
        self.box_dim = 5 if self.task == "obb" else 4
        self.score_threshold = score_threshold
        self.nms_iou = nms_iou
        self.max_detections = max_detections
        self.num_classes = num_classes

        for p in self.det_model.parameters():
            p.requires_grad = False
        self.det_model.eval()

    def train(self, mode: bool = True):
        # Teacher is frozen; always keep it in eval mode.
        super().train(False)
        self.det_model.eval()
        return self

    @torch.no_grad()
    def forward(self, images: torch.Tensor) -> Dict[str, torch.Tensor]:
        # Use Ultralytics post-NMS results for pseudo labels.
        device_arg: object = images.device.index if images.is_cuda else "cpu"
        predict_kwargs = {
            "verbose": False,
            "device": device_arg,
        }
        if self.score_threshold is not None:
            predict_kwargs["conf"] = float(self.score_threshold)
        if self.nms_iou is not None:
            predict_kwargs["iou"] = float(self.nms_iou)
        if self.max_detections is not None:
            predict_kwargs["max_det"] = int(self.max_detections)

        results = self._yolo_infer(images, **predict_kwargs)

        # Apply class filtering and pad to batch tensor.
        batch_boxes: List[torch.Tensor] = []
        batch_scores: List[torch.Tensor] = []
        batch_labels: List[torch.Tensor] = []
        max_keep = 1

        for r in results:
            b_boxes: torch.Tensor
            b_scores: torch.Tensor
            b_labels: torch.Tensor

            if self.task == "obb" and getattr(r, "obb", None) is not None and len(r.obb) > 0:
                obb = r.obb
                h, w = r.orig_shape
                xywhr = obb.xywhr.to(device=images.device, dtype=torch.float32)
                norm = xywhr.new_tensor([w, h, w, h, 1.0]).view(1, 5)
                b_boxes = (xywhr / norm).to(dtype=torch.float32)
                b_boxes[:, :4] = b_boxes[:, :4].clamp(0.0, 1.0)
                b_scores = obb.conf.to(device=images.device, dtype=torch.float32)
                b_labels = obb.cls.to(device=images.device, dtype=torch.long)
            elif getattr(r, "boxes", None) is not None and len(r.boxes) > 0:
                bb = r.boxes
                b_boxes = bb.xywhn.to(device=images.device, dtype=torch.float32)
                b_boxes = b_boxes.clamp(0.0, 1.0)
                b_scores = bb.conf.to(device=images.device, dtype=torch.float32)
                b_labels = bb.cls.to(device=images.device, dtype=torch.long)
            else:
                b_boxes = images.new_zeros((0, self.box_dim))
                b_scores = images.new_zeros((0,), dtype=torch.float32)
                b_labels = images.new_zeros((0,), dtype=torch.long)

            if self.num_classes is not None and b_labels.numel() > 0:
                keep = b_labels < int(self.num_classes)
                b_boxes = b_boxes[keep]
                b_scores = b_scores[keep]
                b_labels = b_labels[keep]

            max_keep = max(max_keep, int(b_scores.numel()))
            batch_boxes.append(b_boxes)
            batch_scores.append(b_scores)
            batch_labels.append(b_labels)

        batch_size = len(batch_boxes)
        padded_boxes = images.new_zeros((batch_size, max_keep, self.box_dim), dtype=torch.float32)
        padded_scores = images.new_full((batch_size, max_keep), -1.0, dtype=torch.float32)
        padded_labels = images.new_zeros((batch_size, max_keep), dtype=torch.long)
        valid_mask = torch.zeros((batch_size, max_keep), dtype=torch.bool, device=images.device)

        for b, (b_boxes, b_scores, b_labels) in enumerate(zip(batch_boxes, batch_scores, batch_labels)):
            n = b_scores.numel()
            if n == 0:
                continue
            padded_boxes[b, :n] = b_boxes
            padded_scores[b, :n] = b_scores
            padded_labels[b, :n] = b_labels
            valid_mask[b, :n] = True

        return {
            "boxes": padded_boxes,
            "scores": padded_scores,
            "labels": padded_labels,
            "valid_mask": valid_mask,
        }
