"""
Prediction Alignment Loss Module

This module implements prediction alignment losses for knowledge distillation
between ensemble teachers and student for object detection.

主要功能:
- Bounding Box 對齊損失 (L1, GIoU, DIoU, CIoU)
- 類別預測對齊損失 (KL Divergence, Focal)
- 組合預測對齊損失
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Tuple, Literal, List
import math


def box_cxcywh_to_xyxy(boxes: torch.Tensor) -> torch.Tensor:
    """
    將 box 從 (cx, cy, w, h) 格式轉換為 (x1, y1, x2, y2) 格式

    Args:
        boxes: [N, 4] 或 [B, N, 4]，格式為 (cx, cy, w, h)

    Returns:
        boxes: 相同形狀，格式為 (x1, y1, x2, y2)
    """
    cx, cy, w, h = boxes.unbind(-1)
    x1 = cx - 0.5 * w
    y1 = cy - 0.5 * h
    x2 = cx + 0.5 * w
    y2 = cy + 0.5 * h
    return torch.stack([x1, y1, x2, y2], dim=-1)


def box_xyxy_to_cxcywh(boxes: torch.Tensor) -> torch.Tensor:
    """
    將 box 從 (x1, y1, x2, y2) 格式轉換為 (cx, cy, w, h) 格式
    """
    x1, y1, x2, y2 = boxes.unbind(-1)
    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2
    w = x2 - x1
    h = y2 - y1
    return torch.stack([cx, cy, w, h], dim=-1)


def box_iou(boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
    """
    計算兩組 boxes 的 IoU

    Args:
        boxes1: [N, 4] 格式為 (x1, y1, x2, y2)
        boxes2: [M, 4] 格式為 (x1, y1, x2, y2)

    Returns:
        iou: [N, M] IoU 矩陣
    """
    area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
    area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])

    # 計算交集
    lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])  # [N, M, 2]
    rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])  # [N, M, 2]

    wh = (rb - lt).clamp(min=0)  # [N, M, 2]
    inter = wh[:, :, 0] * wh[:, :, 1]  # [N, M]

    union = area1[:, None] + area2 - inter

    iou = inter / (union + 1e-8)
    return iou


def generalized_box_iou(boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
    """
    計算 Generalized IoU (GIoU)

    GIoU = IoU - (C - Union) / C
    其中 C 是包含兩個 box 的最小閉包區域

    Args:
        boxes1: [N, 4] 格式為 (x1, y1, x2, y2)
        boxes2: [N, 4] 格式為 (x1, y1, x2, y2)

    Returns:
        giou: [N] GIoU 值，範圍 [-1, 1]
    """
    # 計算基本 IoU
    area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
    area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])

    lt = torch.max(boxes1[:, :2], boxes2[:, :2])
    rb = torch.min(boxes1[:, 2:], boxes2[:, 2:])

    wh = (rb - lt).clamp(min=0)
    inter = wh[:, 0] * wh[:, 1]

    union = area1 + area2 - inter
    iou = inter / (union + 1e-8)

    # 計算閉包區域
    lt_c = torch.min(boxes1[:, :2], boxes2[:, :2])
    rb_c = torch.max(boxes1[:, 2:], boxes2[:, 2:])

    wh_c = (rb_c - lt_c).clamp(min=0)
    area_c = wh_c[:, 0] * wh_c[:, 1]

    giou = iou - (area_c - union) / (area_c + 1e-8)
    return giou


def complete_box_iou(boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
    """
    計算 Complete IoU (CIoU)

    CIoU = IoU - d²/c² - αv
    其中:
    - d: 兩個 box 中心點的距離
    - c: 閉包區域的對角線長度
    - α: 權重參數
    - v: 寬高比一致性

    Args:
        boxes1: [N, 4] 格式為 (x1, y1, x2, y2)
        boxes2: [N, 4] 格式為 (x1, y1, x2, y2)

    Returns:
        ciou: [N] CIoU 值
    """
    # 計算基本 IoU
    area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
    area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])

    lt = torch.max(boxes1[:, :2], boxes2[:, :2])
    rb = torch.min(boxes1[:, 2:], boxes2[:, 2:])

    wh = (rb - lt).clamp(min=0)
    inter = wh[:, 0] * wh[:, 1]

    union = area1 + area2 - inter
    iou = inter / (union + 1e-8)

    # 計算中心點距離
    center1 = (boxes1[:, :2] + boxes1[:, 2:]) / 2
    center2 = (boxes2[:, :2] + boxes2[:, 2:]) / 2
    center_dist = ((center1 - center2) ** 2).sum(dim=-1)

    # 計算閉包對角線
    lt_c = torch.min(boxes1[:, :2], boxes2[:, :2])
    rb_c = torch.max(boxes1[:, 2:], boxes2[:, 2:])
    diag_c = ((rb_c - lt_c) ** 2).sum(dim=-1)

    # 計算寬高比一致性
    w1 = boxes1[:, 2] - boxes1[:, 0]
    h1 = boxes1[:, 3] - boxes1[:, 1]
    w2 = boxes2[:, 2] - boxes2[:, 0]
    h2 = boxes2[:, 3] - boxes2[:, 1]

    v = (4 / (math.pi ** 2)) * ((torch.atan(w2 / (h2 + 1e-8)) - torch.atan(w1 / (h1 + 1e-8))) ** 2)

    with torch.no_grad():
        alpha = v / (1 - iou + v + 1e-8)

    ciou = iou - center_dist / (diag_c + 1e-8) - alpha * v
    return ciou


class BoxAlignmentLoss(nn.Module):
    """
    Bounding Box 對齊損失

    用於對齊 student 和 teacher ensemble 的 bounding box 預測。

    Args:
        loss_type: 損失類型 ("l1", "smooth_l1", "giou", "diou", "ciou")
        box_format: Box 格式 ("cxcywh", "xyxy")
        reduction: Reduction 方式

    Example:
        >>> loss_fn = BoxAlignmentLoss(loss_type="giou")
        >>> student_boxes = torch.randn(100, 4)  # [N, 4]
        >>> teacher_boxes = torch.randn(100, 4)  # [N, 4]
        >>> loss = loss_fn(student_boxes, teacher_boxes)
    """

    def __init__(
        self,
        loss_type: Literal["l1", "smooth_l1", "giou", "diou", "ciou"] = "giou",
        box_format: Literal["cxcywh", "xyxy"] = "cxcywh",
        reduction: Literal["mean", "sum", "none"] = "mean",
    ):
        super().__init__()
        self.loss_type = loss_type
        self.box_format = box_format
        self.reduction = reduction

    def forward(
        self,
        student_boxes: torch.Tensor,
        teacher_boxes: torch.Tensor,
        weights: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        計算 box 對齊損失

        Args:
            student_boxes: Student 預測的 boxes [N, 4] 或 [B, N, 4]
            teacher_boxes: Teacher 預測的 boxes [N, 4] 或 [B, N, 4]
            weights: 可選的權重 [N] 或 [B, N]

        Returns:
            loss: Box 對齊損失
        """
        # 轉換格式（如果需要）
        if self.box_format == "cxcywh" and self.loss_type in ["giou", "diou", "ciou"]:
            student_boxes_xyxy = box_cxcywh_to_xyxy(student_boxes)
            teacher_boxes_xyxy = box_cxcywh_to_xyxy(teacher_boxes)
        elif self.box_format == "xyxy":
            student_boxes_xyxy = student_boxes
            teacher_boxes_xyxy = teacher_boxes
        else:
            student_boxes_xyxy = student_boxes
            teacher_boxes_xyxy = teacher_boxes

        # 計算損失
        if self.loss_type == "l1":
            loss = F.l1_loss(student_boxes, teacher_boxes, reduction="none")
            loss = loss.sum(dim=-1)  # Sum over box dimensions

        elif self.loss_type == "smooth_l1":
            loss = F.smooth_l1_loss(student_boxes, teacher_boxes, reduction="none")
            loss = loss.sum(dim=-1)

        elif self.loss_type == "giou":
            # 展平處理批次維度
            orig_shape = student_boxes_xyxy.shape[:-1]
            s_flat = student_boxes_xyxy.reshape(-1, 4)
            t_flat = teacher_boxes_xyxy.reshape(-1, 4)
            giou = generalized_box_iou(s_flat, t_flat)
            loss = 1 - giou
            loss = loss.reshape(orig_shape)

        elif self.loss_type == "ciou":
            orig_shape = student_boxes_xyxy.shape[:-1]
            s_flat = student_boxes_xyxy.reshape(-1, 4)
            t_flat = teacher_boxes_xyxy.reshape(-1, 4)
            ciou = complete_box_iou(s_flat, t_flat)
            loss = 1 - ciou
            loss = loss.reshape(orig_shape)

        else:
            raise ValueError(f"Unknown loss type: {self.loss_type}")

        # 應用權重
        if weights is not None:
            loss = loss * weights

        # Reduction
        if self.reduction == "mean":
            if weights is not None:
                return loss.sum() / (weights.sum() + 1e-8)
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss


class ClassAlignmentLoss(nn.Module):
    """
    類別預測對齊損失

    用於對齊 student 和 teacher 的類別預測分佈。

    Args:
        loss_type: 損失類型 ("kl", "mse", "focal_kl")
        temperature: Softmax 溫度參數
        focal_alpha: Focal 權重的 alpha 參數
        focal_gamma: Focal 權重的 gamma 參數

    Example:
        >>> loss_fn = ClassAlignmentLoss(loss_type="kl", temperature=4.0)
        >>> student_logits = torch.randn(100, 10)  # [N, num_classes]
        >>> teacher_logits = torch.randn(100, 10)
        >>> loss = loss_fn(student_logits, teacher_logits)
    """

    def __init__(
        self,
        loss_type: Literal["kl", "mse", "focal_kl"] = "kl",
        temperature: float = 4.0,
        focal_alpha: float = 0.25,
        focal_gamma: float = 2.0,
        reduction: Literal["mean", "sum", "none"] = "mean",
    ):
        super().__init__()
        self.loss_type = loss_type
        self.temperature = temperature
        self.focal_alpha = focal_alpha
        self.focal_gamma = focal_gamma
        self.reduction = reduction

    def forward(
        self,
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
        weights: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        計算類別對齊損失

        Args:
            student_logits: Student 的類別 logits [N, C] 或 [B, N, C]
            teacher_logits: Teacher 的類別 logits [N, C] 或 [B, N, C]
            weights: 可選的權重

        Returns:
            loss: 類別對齊損失
        """
        if self.loss_type == "kl":
            # Standard KL divergence with temperature
            student_log_prob = F.log_softmax(student_logits / self.temperature, dim=-1)
            teacher_prob = F.softmax(teacher_logits / self.temperature, dim=-1)

            loss = F.kl_div(student_log_prob, teacher_prob, reduction="none")
            loss = loss.sum(dim=-1) * (self.temperature ** 2)

        elif self.loss_type == "mse":
            # MSE between softmax probabilities
            student_prob = F.softmax(student_logits / self.temperature, dim=-1)
            teacher_prob = F.softmax(teacher_logits / self.temperature, dim=-1)
            loss = F.mse_loss(student_prob, teacher_prob, reduction="none")
            loss = loss.sum(dim=-1)

        elif self.loss_type == "focal_kl":
            # Focal-weighted KL divergence
            student_log_prob = F.log_softmax(student_logits / self.temperature, dim=-1)
            student_prob = F.softmax(student_logits / self.temperature, dim=-1)
            teacher_prob = F.softmax(teacher_logits / self.temperature, dim=-1)

            # 計算 focal 權重
            focal_weight = self.focal_alpha * (1 - student_prob) ** self.focal_gamma

            kl = teacher_prob * (teacher_prob.log() - student_log_prob)
            loss = (focal_weight * kl).sum(dim=-1) * (self.temperature ** 2)

        else:
            raise ValueError(f"Unknown loss type: {self.loss_type}")

        # 應用權重
        if weights is not None:
            loss = loss * weights

        # Reduction
        if self.reduction == "mean":
            if weights is not None:
                return loss.sum() / (weights.sum() + 1e-8)
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss


class PredictionAlignmentLoss(nn.Module):
    """
    完整的預測對齊損失

    組合 Box 對齊和 Class 對齊損失。

    Args:
        box_loss_type: Box 損失類型
        class_loss_type: Class 損失類型
        box_weight: Box 損失權重
        class_weight: Class 損失權重
        objectness_weight: Objectness 損失權重（可選）
        temperature: 類別損失的溫度參數

    Example:
        >>> loss_fn = PredictionAlignmentLoss(
        ...     box_loss_type="giou",
        ...     class_loss_type="kl",
        ...     box_weight=2.0,
        ...     class_weight=1.0,
        ... )
        >>> student_pred = {
        ...     "boxes": torch.randn(4, 100, 4),
        ...     "logits": torch.randn(4, 100, 10),
        ... }
        >>> teacher_pred = {
        ...     "boxes": torch.randn(4, 100, 4),
        ...     "logits": torch.randn(4, 100, 10),
        ... }
        >>> loss, loss_dict = loss_fn(student_pred, teacher_pred)
    """

    def __init__(
        self,
        box_loss_type: str = "giou",
        class_loss_type: str = "kl",
        box_weight: float = 2.0,
        class_weight: float = 1.0,
        objectness_weight: float = 1.0,
        temperature: float = 4.0,
        box_format: str = "cxcywh",
    ):
        super().__init__()
        self.box_weight = box_weight
        self.class_weight = class_weight
        self.objectness_weight = objectness_weight

        self.box_loss = BoxAlignmentLoss(
            loss_type=box_loss_type,
            box_format=box_format,
            reduction="none",
        )

        self.class_loss = ClassAlignmentLoss(
            loss_type=class_loss_type,
            temperature=temperature,
            reduction="none",
        )

    def forward(
        self,
        student_predictions: Dict[str, torch.Tensor],
        teacher_predictions: Dict[str, torch.Tensor],
        valid_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        計算預測對齊損失

        Args:
            student_predictions: Student 預測字典
                - "boxes": [B, N, 4] bounding boxes
                - "logits": [B, N, C] 類別 logits
                - "objectness": [B, N] objectness scores (可選)
            teacher_predictions: Teacher 預測字典（同上格式）
            valid_mask: 有效預測的遮罩 [B, N]

        Returns:
            total_loss: 總損失
            loss_dict: 各項損失字典
        """
        loss_dict = {}

        # Box alignment loss
        box_loss = self.box_loss(
            student_predictions["boxes"],
            teacher_predictions["boxes"],
        )

        # Class alignment loss
        class_loss = self.class_loss(
            student_predictions["logits"],
            teacher_predictions["logits"],
        )

        # Objectness alignment loss (如果存在)
        if "objectness" in student_predictions and "objectness" in teacher_predictions:
            obj_loss = F.binary_cross_entropy_with_logits(
                student_predictions["objectness"],
                torch.sigmoid(teacher_predictions["objectness"]),
                reduction="none",
            )
        else:
            obj_loss = torch.zeros_like(box_loss)

        # 應用 valid mask
        if valid_mask is not None:
            box_loss = box_loss * valid_mask
            class_loss = class_loss * valid_mask
            obj_loss = obj_loss * valid_mask
            num_valid = valid_mask.sum() + 1e-8
        else:
            num_valid = box_loss.numel()

        # 計算平均損失
        box_loss_mean = box_loss.sum() / num_valid
        class_loss_mean = class_loss.sum() / num_valid
        obj_loss_mean = obj_loss.sum() / num_valid

        # 加權總損失
        total_loss = (
            self.box_weight * box_loss_mean +
            self.class_weight * class_loss_mean +
            self.objectness_weight * obj_loss_mean
        )

        loss_dict["pred_align_box_loss"] = box_loss_mean.detach()
        loss_dict["pred_align_class_loss"] = class_loss_mean.detach()
        loss_dict["pred_align_obj_loss"] = obj_loss_mean.detach()
        loss_dict["pred_align_total_loss"] = total_loss.detach()

        return total_loss, loss_dict


class HungarianMatchingLoss(nn.Module):
    """
    基於 Hungarian Matching 的預測對齊損失

    當 student 和 teacher 的預測數量不同時使用。
    先用 Hungarian 算法找到最佳匹配，再計算損失。

    Args:
        box_cost_weight: 匹配時 box 成本的權重
        class_cost_weight: 匹配時 class 成本的權重
        box_loss_type: Box 損失類型
        class_loss_type: Class 損失類型
    """

    def __init__(
        self,
        box_cost_weight: float = 5.0,
        class_cost_weight: float = 2.0,
        box_loss_type: str = "giou",
        class_loss_type: str = "kl",
        box_weight: float = 2.0,
        class_weight: float = 1.0,
        temperature: float = 4.0,
    ):
        super().__init__()
        self.box_cost_weight = box_cost_weight
        self.class_cost_weight = class_cost_weight

        self.prediction_loss = PredictionAlignmentLoss(
            box_loss_type=box_loss_type,
            class_loss_type=class_loss_type,
            box_weight=box_weight,
            class_weight=class_weight,
            temperature=temperature,
        )

    @torch.no_grad()
    def compute_matching_cost(
        self,
        student_boxes: torch.Tensor,
        teacher_boxes: torch.Tensor,
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
    ) -> torch.Tensor:
        """
        計算匹配成本矩陣

        Args:
            student_boxes: [N, 4]
            teacher_boxes: [M, 4]
            student_logits: [N, C]
            teacher_logits: [M, C]

        Returns:
            cost_matrix: [N, M] 成本矩陣
        """
        N = student_boxes.shape[0]
        M = teacher_boxes.shape[0]

        # Box cost (1 - GIoU)
        student_boxes_xyxy = box_cxcywh_to_xyxy(student_boxes)
        teacher_boxes_xyxy = box_cxcywh_to_xyxy(teacher_boxes)

        # 計算成對 GIoU
        box_cost = torch.zeros(N, M, device=student_boxes.device)
        for i in range(N):
            giou = generalized_box_iou(
                student_boxes_xyxy[i:i+1].expand(M, -1),
                teacher_boxes_xyxy,
            )
            box_cost[i] = 1 - giou

        # Class cost (KL divergence)
        student_prob = F.softmax(student_logits, dim=-1)  # [N, C]
        teacher_prob = F.softmax(teacher_logits, dim=-1)  # [M, C]

        # [N, M] 成對 KL divergence
        class_cost = torch.zeros(N, M, device=student_logits.device)
        for i in range(N):
            kl = F.kl_div(
                student_prob[i:i+1].log().expand(M, -1),
                teacher_prob,
                reduction="none",
            ).sum(dim=-1)
            class_cost[i] = kl

        # 總成本
        cost_matrix = (
            self.box_cost_weight * box_cost +
            self.class_cost_weight * class_cost
        )

        return cost_matrix

    def forward(
        self,
        student_predictions: Dict[str, torch.Tensor],
        teacher_predictions: Dict[str, torch.Tensor],
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        使用 Hungarian matching 計算損失

        Args:
            student_predictions: {"boxes": [B, N, 4], "logits": [B, N, C]}
            teacher_predictions: {"boxes": [B, M, 4], "logits": [B, M, C]}

        Returns:
            loss: 總損失
            loss_dict: 損失字典
        """
        try:
            from scipy.optimize import linear_sum_assignment
        except ImportError:
            raise ImportError("scipy is required for Hungarian matching")

        B = student_predictions["boxes"].shape[0]
        device = student_predictions["boxes"].device

        total_loss = 0.0
        loss_dict = {"matched_box_loss": 0.0, "matched_class_loss": 0.0}

        for b in range(B):
            s_boxes = student_predictions["boxes"][b]
            t_boxes = teacher_predictions["boxes"][b]
            s_logits = student_predictions["logits"][b]
            t_logits = teacher_predictions["logits"][b]

            # 計算成本矩陣
            cost_matrix = self.compute_matching_cost(s_boxes, t_boxes, s_logits, t_logits)

            # Hungarian matching
            row_ind, col_ind = linear_sum_assignment(cost_matrix.cpu().numpy())
            row_ind = torch.tensor(row_ind, device=device)
            col_ind = torch.tensor(col_ind, device=device)

            # 提取匹配的預測
            matched_student = {
                "boxes": s_boxes[row_ind].unsqueeze(0),
                "logits": s_logits[row_ind].unsqueeze(0),
            }
            matched_teacher = {
                "boxes": t_boxes[col_ind].unsqueeze(0),
                "logits": t_logits[col_ind].unsqueeze(0),
            }

            # 計算損失
            loss, batch_loss_dict = self.prediction_loss(matched_student, matched_teacher)
            total_loss += loss

            for k, v in batch_loss_dict.items():
                if k in loss_dict:
                    loss_dict[k] = loss_dict.get(k, 0) + v

        total_loss = total_loss / B
        for k in loss_dict:
            loss_dict[k] = loss_dict[k] / B

        return total_loss, loss_dict
