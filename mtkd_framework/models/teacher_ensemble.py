"""
Teacher Ensemble Module

This module implements the ensemble of multiple detection teachers
for Multi-Teacher Knowledge Distillation.

主要功能:
- 多個 Teacher 模型的 Ensemble
- Weighted Box Fusion (WBF) 用於合併預測
- 支持不同的 ensemble 策略
"""

import torch
import torch.nn as nn
from typing import List, Dict, Optional, Tuple, Literal, Any
import numpy as np


class WeightedBoxFusion(nn.Module):
    """
    Weighted Box Fusion (WBF)

    將多個模型的 bounding box 預測進行加權融合。

    與 NMS 不同，WBF 會融合重疊的 boxes 而非簡單地抑制它們，
    這樣可以產生更準確的定位。

    Args:
        iou_threshold: IoU 閾值，用於判斷 boxes 是否屬於同一物體
        skip_box_threshold: 信心度閾值，低於此值的 box 會被忽略
        weights: 各個模型的權重
        conf_type: 信心度融合方式 ("avg", "max", "box_and_model_avg")

    Example:
        >>> wbf = WeightedBoxFusion(iou_threshold=0.5, weights=[0.6, 0.4])
        >>> boxes_list = [
        ...     torch.tensor([[0.1, 0.1, 0.3, 0.3], [0.5, 0.5, 0.7, 0.7]]),  # model 1
        ...     torch.tensor([[0.12, 0.11, 0.31, 0.32], [0.6, 0.6, 0.8, 0.8]]),  # model 2
        ... ]
        >>> scores_list = [
        ...     torch.tensor([0.9, 0.8]),
        ...     torch.tensor([0.85, 0.7]),
        ... ]
        >>> labels_list = [torch.tensor([0, 1]), torch.tensor([0, 1])]
        >>> fused_boxes, fused_scores, fused_labels = wbf(boxes_list, scores_list, labels_list)
    """

    def __init__(
        self,
        iou_threshold: float = 0.55,
        skip_box_threshold: float = 0.0,
        weights: Optional[List[float]] = None,
        conf_type: Literal["avg", "max", "box_and_model_avg"] = "avg",
    ):
        super().__init__()
        self.iou_threshold = iou_threshold
        self.skip_box_threshold = skip_box_threshold
        self.conf_type = conf_type

        if weights is not None:
            self.register_buffer("weights", torch.tensor(weights, dtype=torch.float32))
        else:
            self.weights = None

    def _box_iou(self, box1: torch.Tensor, box2: torch.Tensor) -> torch.Tensor:
        """計算兩個 box 的 IoU"""
        x1 = torch.max(box1[0], box2[0])
        y1 = torch.max(box1[1], box2[1])
        x2 = torch.min(box1[2], box2[2])
        y2 = torch.min(box1[3], box2[3])

        inter = torch.clamp(x2 - x1, min=0) * torch.clamp(y2 - y1, min=0)

        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])

        return inter / (area1 + area2 - inter + 1e-8)

    def _find_matching_box(
        self,
        boxes_list: List[torch.Tensor],
        new_box: torch.Tensor,
        match_iou: float,
    ) -> int:
        """找到與 new_box 匹配的 cluster"""
        best_iou = match_iou
        best_index = -1

        for i, box in enumerate(boxes_list):
            iou = self._box_iou(box, new_box)
            if iou > best_iou:
                best_iou = iou
                best_index = i

        return best_index

    @torch.no_grad()
    def forward(
        self,
        boxes_list: List[torch.Tensor],
        scores_list: List[torch.Tensor],
        labels_list: List[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        執行 Weighted Box Fusion

        Args:
            boxes_list: 各模型的 boxes 列表，每個元素形狀 [N_i, 4]，格式 (x1, y1, x2, y2)
            scores_list: 各模型的信心度列表，每個元素形狀 [N_i]
            labels_list: 各模型的類別標籤列表，每個元素形狀 [N_i]

        Returns:
            fused_boxes: 融合後的 boxes [M, 4]
            fused_scores: 融合後的信心度 [M]
            fused_labels: 融合後的類別標籤 [M]
        """
        num_models = len(boxes_list)

        # 設定權重
        if self.weights is not None:
            weights = self.weights.to(boxes_list[0].device)
        else:
            weights = torch.ones(num_models, device=boxes_list[0].device) / num_models

        # 收集所有 boxes 和相關信息
        all_boxes = []
        all_scores = []
        all_labels = []
        all_model_indices = []

        for model_idx, (boxes, scores, labels) in enumerate(zip(boxes_list, scores_list, labels_list)):
            # 過濾低信心度的 boxes
            mask = scores >= self.skip_box_threshold
            boxes = boxes[mask]
            scores = scores[mask]
            labels = labels[mask]

            # 應用模型權重到信心度
            weighted_scores = scores * weights[model_idx]

            all_boxes.append(boxes)
            all_scores.append(weighted_scores)
            all_labels.append(labels)
            all_model_indices.extend([model_idx] * len(boxes))

        # 合併所有數據
        if len(all_boxes) == 0 or all(len(b) == 0 for b in all_boxes):
            device = boxes_list[0].device
            return (
                torch.zeros((0, 4), device=device),
                torch.zeros((0,), device=device),
                torch.zeros((0,), dtype=torch.long, device=device),
            )

        all_boxes = torch.cat(all_boxes, dim=0)
        all_scores = torch.cat(all_scores, dim=0)
        all_labels = torch.cat(all_labels, dim=0)
        all_model_indices = torch.tensor(all_model_indices, device=all_boxes.device)

        # 按信心度排序
        sorted_indices = torch.argsort(all_scores, descending=True)
        all_boxes = all_boxes[sorted_indices]
        all_scores = all_scores[sorted_indices]
        all_labels = all_labels[sorted_indices]
        all_model_indices = all_model_indices[sorted_indices]

        # 獲取所有唯一的類別
        unique_labels = torch.unique(all_labels)

        fused_boxes_list = []
        fused_scores_list = []
        fused_labels_list = []

        # 對每個類別分別進行 WBF
        for label in unique_labels:
            label_mask = all_labels == label
            label_boxes = all_boxes[label_mask]
            label_scores = all_scores[label_mask]
            label_model_indices = all_model_indices[label_mask]

            # Clustering boxes
            clusters = []  # List of (weighted_box, [scores], [model_indices])

            for i in range(len(label_boxes)):
                box = label_boxes[i]
                score = label_scores[i]
                model_idx = label_model_indices[i]

                # 找到匹配的 cluster
                match_idx = -1
                best_iou = self.iou_threshold
                for j, (cluster_box, _, _) in enumerate(clusters):
                    iou = self._box_iou(cluster_box, box)
                    if iou > best_iou:
                        best_iou = iou
                        match_idx = j

                if match_idx >= 0:
                    # 添加到現有 cluster
                    clusters[match_idx][1].append((box, score, model_idx))
                else:
                    # 創建新 cluster
                    clusters.append([box.clone(), [(box, score, model_idx)], label])

            # 融合每個 cluster 中的 boxes
            for cluster_box, box_info_list, cluster_label in clusters:
                if len(box_info_list) == 0:
                    continue

                # 加權平均 boxes
                weighted_box = torch.zeros(4, device=all_boxes.device)
                total_weight = 0.0

                for box, score, model_idx in box_info_list:
                    weight = score.item()
                    weighted_box += box * weight
                    total_weight += weight

                weighted_box /= (total_weight + 1e-8)

                # 計算融合信心度
                if self.conf_type == "avg":
                    fused_score = sum(s.item() for _, s, _ in box_info_list) / len(box_info_list)
                elif self.conf_type == "max":
                    fused_score = max(s.item() for _, s, _ in box_info_list)
                else:  # "box_and_model_avg"
                    model_scores = {}
                    for _, s, m_idx in box_info_list:
                        m_idx = m_idx.item()
                        if m_idx not in model_scores:
                            model_scores[m_idx] = []
                        model_scores[m_idx].append(s.item())
                    avg_per_model = [sum(v) / len(v) for v in model_scores.values()]
                    fused_score = sum(avg_per_model) / len(avg_per_model)

                # 根據參與 cluster 的模型數量調整信心度
                n_models_in_cluster = len(set(m.item() for _, _, m in box_info_list))
                fused_score *= min(n_models_in_cluster, num_models) / num_models

                fused_boxes_list.append(weighted_box)
                fused_scores_list.append(fused_score)
                fused_labels_list.append(cluster_label)

        if len(fused_boxes_list) == 0:
            device = boxes_list[0].device
            return (
                torch.zeros((0, 4), device=device),
                torch.zeros((0,), device=device),
                torch.zeros((0,), dtype=torch.long, device=device),
            )

        fused_boxes = torch.stack(fused_boxes_list, dim=0)
        fused_scores = torch.tensor(fused_scores_list, device=fused_boxes.device)
        fused_labels = torch.stack(fused_labels_list, dim=0)

        # 最終按信心度排序
        sorted_idx = torch.argsort(fused_scores, descending=True)
        return fused_boxes[sorted_idx], fused_scores[sorted_idx], fused_labels[sorted_idx]


class SoftNMS(nn.Module):
    """
    Soft-NMS

    相比傳統 NMS，Soft-NMS 不直接移除重疊的 boxes，
    而是降低它們的信心度。

    Args:
        iou_threshold: IoU 閾值
        sigma: Gaussian 衰減的 sigma 參數
        score_threshold: 最低信心度閾值
        method: "linear" 或 "gaussian"
    """

    def __init__(
        self,
        iou_threshold: float = 0.5,
        sigma: float = 0.5,
        score_threshold: float = 0.001,
        method: Literal["linear", "gaussian"] = "gaussian",
    ):
        super().__init__()
        self.iou_threshold = iou_threshold
        self.sigma = sigma
        self.score_threshold = score_threshold
        self.method = method

    @torch.no_grad()
    def forward(
        self,
        boxes: torch.Tensor,
        scores: torch.Tensor,
        labels: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        執行 Soft-NMS

        Args:
            boxes: [N, 4] 格式 (x1, y1, x2, y2)
            scores: [N] 信心度
            labels: [N] 類別標籤

        Returns:
            filtered_boxes, filtered_scores, filtered_labels
        """
        unique_labels = torch.unique(labels)
        all_boxes, all_scores, all_labels = [], [], []

        for label in unique_labels:
            mask = labels == label
            label_boxes = boxes[mask].clone()
            label_scores = scores[mask].clone()

            # Soft-NMS
            keep = []
            while len(label_scores) > 0:
                max_idx = torch.argmax(label_scores)
                keep.append(max_idx.item())

                if len(label_scores) == 1:
                    break

                current_box = label_boxes[max_idx]
                other_boxes = torch.cat([label_boxes[:max_idx], label_boxes[max_idx+1:]])
                other_scores = torch.cat([label_scores[:max_idx], label_scores[max_idx+1:]])

                # 計算 IoU
                ious = self._box_iou_batch(current_box.unsqueeze(0), other_boxes).squeeze(0)

                # 更新信心度
                if self.method == "linear":
                    decay = torch.where(
                        ious > self.iou_threshold,
                        1 - ious,
                        torch.ones_like(ious),
                    )
                else:  # gaussian
                    decay = torch.exp(-(ious ** 2) / self.sigma)

                other_scores = other_scores * decay

                # 過濾低信心度的 boxes
                valid_mask = other_scores >= self.score_threshold
                label_boxes = other_boxes[valid_mask]
                label_scores = other_scores[valid_mask]

            if len(keep) > 0:
                all_boxes.append(boxes[mask][keep])
                all_scores.append(scores[mask][keep])
                all_labels.append(labels[mask][keep])

        if len(all_boxes) == 0:
            device = boxes.device
            return (
                torch.zeros((0, 4), device=device),
                torch.zeros((0,), device=device),
                torch.zeros((0,), dtype=torch.long, device=device),
            )

        return torch.cat(all_boxes), torch.cat(all_scores), torch.cat(all_labels)

    def _box_iou_batch(self, boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
        """批次計算 IoU"""
        area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
        area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])

        lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])
        rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])

        wh = (rb - lt).clamp(min=0)
        inter = wh[:, :, 0] * wh[:, :, 1]

        return inter / (area1[:, None] + area2 - inter + 1e-8)


class TeacherEnsemble(nn.Module):
    """
    Teacher Ensemble 模組

    組合多個預訓練的 detection teacher 模型，
    使用 Weighted Box Fusion 融合它們的預測。

    Args:
        teacher_models: Teacher 模型列表（已預訓練並 frozen）
        teacher_weights: 各 teacher 的權重
        fusion_method: 融合方法 ("wbf", "soft_nms", "nms")
        fusion_config: 融合方法的配置

    Example:
        >>> teacher1 = load_pretrained_model("teacher1.pth")
        >>> teacher2 = load_pretrained_model("teacher2.pth")
        >>> ensemble = TeacherEnsemble(
        ...     teacher_models=[teacher1, teacher2],
        ...     teacher_weights=[0.6, 0.4],
        ...     fusion_method="wbf",
        ... )
        >>> ensemble_predictions = ensemble(images)
    """

    def __init__(
        self,
        teacher_models: Optional[List[nn.Module]] = None,
        teacher_weights: Optional[List[float]] = None,
        fusion_method: Literal["wbf", "soft_nms", "nms"] = "wbf",
        fusion_config: Optional[Dict[str, Any]] = None,
        num_classes: int = 1,  # 氣孔檢測通常是單類
    ):
        super().__init__()

        self.num_classes = num_classes
        self.fusion_method = fusion_method

        # 初始化 teacher 模型
        if teacher_models is not None:
            self.teachers = nn.ModuleList(teacher_models)
            # Freeze all teachers
            for teacher in self.teachers:
                teacher.eval()
                for param in teacher.parameters():
                    param.requires_grad = False
        else:
            self.teachers = nn.ModuleList()

        # Teacher 權重
        if teacher_weights is not None:
            assert len(teacher_weights) == len(self.teachers)
            self.register_buffer("teacher_weights", torch.tensor(teacher_weights))
        else:
            n = len(self.teachers) if self.teachers else 1
            self.register_buffer("teacher_weights", torch.ones(n) / n)

        # 初始化融合模組
        fusion_config = fusion_config or {}
        if fusion_method == "wbf":
            self.fusion = WeightedBoxFusion(
                weights=teacher_weights,
                **fusion_config,
            )
        elif fusion_method == "soft_nms":
            self.fusion = SoftNMS(**fusion_config)
        else:  # nms
            self.fusion = None  # 使用 torchvision.ops.nms

    def add_teacher(
        self,
        teacher_model: nn.Module,
        weight: float = 1.0,
    ):
        """
        添加一個 teacher 模型

        Args:
            teacher_model: 預訓練的 teacher 模型
            weight: 該 teacher 的權重
        """
        teacher_model.eval()
        for param in teacher_model.parameters():
            param.requires_grad = False

        self.teachers.append(teacher_model)

        # 更新權重
        new_weights = torch.cat([
            self.teacher_weights,
            torch.tensor([weight], device=self.teacher_weights.device)
        ])
        # 正規化
        new_weights = new_weights / new_weights.sum()
        self.teacher_weights = new_weights

    def load_teachers_from_checkpoints(
        self,
        checkpoint_paths: List[str],
        model_class: type,
        model_kwargs: Optional[Dict] = None,
        weights: Optional[List[float]] = None,
    ):
        """
        從 checkpoint 載入 teacher 模型

        Args:
            checkpoint_paths: Checkpoint 文件路徑列表
            model_class: 模型類
            model_kwargs: 模型初始化參數
            weights: 各 teacher 的權重
        """
        model_kwargs = model_kwargs or {}

        for path in checkpoint_paths:
            model = model_class(**model_kwargs)
            checkpoint = torch.load(path, map_location="cpu")

            # 嘗試不同的 state_dict key
            if "model" in checkpoint:
                model.load_state_dict(checkpoint["model"])
            elif "state_dict" in checkpoint:
                model.load_state_dict(checkpoint["state_dict"])
            else:
                model.load_state_dict(checkpoint)

            model.eval()
            for param in model.parameters():
                param.requires_grad = False

            self.teachers.append(model)

        # 設置權重
        if weights is not None:
            self.teacher_weights = torch.tensor(weights)
        else:
            n = len(self.teachers)
            self.teacher_weights = torch.ones(n) / n

        # 更新 fusion 的權重
        if isinstance(self.fusion, WeightedBoxFusion):
            self.fusion.weights = self.teacher_weights

    @torch.no_grad()
    def forward(
        self,
        images: torch.Tensor,
        return_individual: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """
        獲取 ensemble teacher 的預測

        Args:
            images: 輸入圖像 [B, C, H, W]
            return_individual: 是否同時返回各個 teacher 的預測

        Returns:
            predictions: 融合後的預測字典
                - "boxes": [B, N, 4] 融合後的 boxes
                - "scores": [B, N] 融合後的信心度
                - "labels": [B, N] 融合後的類別標籤
                - "logits": [B, N, num_classes] 類別 logits
        """
        if len(self.teachers) == 0:
            raise ValueError("No teacher models loaded")

        batch_size = images.shape[0]
        device = images.device

        # 收集所有 teacher 的預測
        all_predictions = []
        for teacher in self.teachers:
            pred = teacher(images)
            all_predictions.append(pred)

        # 對每張圖像進行融合
        fused_boxes_list = []
        fused_scores_list = []
        fused_labels_list = []

        for b in range(batch_size):
            # 收集該圖像的所有 teacher 預測
            boxes_list = []
            scores_list = []
            labels_list = []

            for pred in all_predictions:
                # 適應不同的預測格式
                if isinstance(pred, dict):
                    boxes = pred["boxes"][b] if "boxes" in pred else pred["pred_boxes"][b]
                    scores = pred["scores"][b] if "scores" in pred else pred["pred_logits"][b].softmax(-1).max(-1)[0]
                    labels = pred["labels"][b] if "labels" in pred else pred["pred_logits"][b].argmax(-1)
                elif isinstance(pred, (list, tuple)):
                    boxes = pred[0][b]
                    scores = pred[1][b]
                    labels = pred[2][b] if len(pred) > 2 else torch.zeros_like(scores, dtype=torch.long)
                else:
                    raise ValueError(f"Unsupported prediction format: {type(pred)}")

                boxes_list.append(boxes)
                scores_list.append(scores)
                labels_list.append(labels)

            # 融合預測
            if self.fusion_method == "wbf":
                fused_boxes, fused_scores, fused_labels = self.fusion(
                    boxes_list, scores_list, labels_list
                )
            elif self.fusion_method == "soft_nms":
                # 先合併再 soft_nms
                all_boxes = torch.cat(boxes_list, dim=0)
                all_scores = torch.cat(scores_list, dim=0)
                all_labels = torch.cat(labels_list, dim=0)
                fused_boxes, fused_scores, fused_labels = self.fusion(
                    all_boxes, all_scores, all_labels
                )
            else:  # nms
                from torchvision.ops import nms
                all_boxes = torch.cat(boxes_list, dim=0)
                all_scores = torch.cat(scores_list, dim=0)
                all_labels = torch.cat(labels_list, dim=0)
                keep = nms(all_boxes, all_scores, iou_threshold=0.5)
                fused_boxes = all_boxes[keep]
                fused_scores = all_scores[keep]
                fused_labels = all_labels[keep]

            fused_boxes_list.append(fused_boxes)
            fused_scores_list.append(fused_scores)
            fused_labels_list.append(fused_labels)

        # Pad to same length and stack
        max_detections = max(len(b) for b in fused_boxes_list)
        max_detections = max(max_detections, 1)  # 至少 1 個

        padded_boxes = torch.zeros(batch_size, max_detections, 4, device=device)
        padded_scores = torch.zeros(batch_size, max_detections, device=device)
        padded_labels = torch.zeros(batch_size, max_detections, dtype=torch.long, device=device)
        valid_mask = torch.zeros(batch_size, max_detections, dtype=torch.bool, device=device)

        for b, (boxes, scores, labels) in enumerate(zip(fused_boxes_list, fused_scores_list, fused_labels_list)):
            n = len(boxes)
            if n > 0:
                padded_boxes[b, :n] = boxes
                padded_scores[b, :n] = scores
                padded_labels[b, :n] = labels
                valid_mask[b, :n] = True

        # 構建 logits（用於知識蒸餾）
        # 將 scores 和 labels 轉換為 logits 格式
        padded_logits = torch.zeros(batch_size, max_detections, self.num_classes + 1, device=device)
        for b in range(batch_size):
            for i in range(max_detections):
                if valid_mask[b, i]:
                    # 將信心度分配到對應類別，其餘分配到背景類
                    label = padded_labels[b, i]
                    score = padded_scores[b, i]
                    # 使用 logit 形式：log(p/(1-p))
                    padded_logits[b, i, label] = torch.log(score / (1 - score + 1e-8) + 1e-8)
                    # 背景類（最後一個）
                    padded_logits[b, i, -1] = torch.log((1 - score) / (score + 1e-8) + 1e-8)

        result = {
            "boxes": padded_boxes,
            "scores": padded_scores,
            "labels": padded_labels,
            "logits": padded_logits,
            "valid_mask": valid_mask,
        }

        if return_individual:
            result["individual_predictions"] = all_predictions

        return result

    def train(self, mode: bool = True):
        """Override train to keep teachers in eval mode"""
        super().train(mode)
        for teacher in self.teachers:
            teacher.eval()
        return self
