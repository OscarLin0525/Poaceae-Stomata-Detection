"""
Combined MTKD Loss Module

This module combines feature alignment and prediction alignment losses
for the Multi-Teacher Knowledge Distillation framework.

主要功能:
- 組合特徵對齊和預測對齊損失
- 支持動態權重調整
- 損失監控和記錄
"""

import torch
import torch.nn as nn
from typing import Dict, Tuple, Optional, List, Any
from .feature_alignment import (
    FeatureAlignmentLoss,
    MultiScaleFeatureAlignmentLoss,
    AttentionAlignmentLoss,
    TokenMatchingLoss,
)
from .prediction_alignment import (
    PredictionAlignmentLoss,
    HungarianMatchingLoss,
)


class MTKDLoss(nn.Module):
    """
    Multi-Teacher Knowledge Distillation 損失

    組合以下損失:
    1. Feature Alignment Loss: DINO teacher 與 student 的特徵對齊
    2. Prediction Alignment Loss: Ensemble teachers 與 student 的預測對齊
    3. (可選) Detection Loss: Student 的檢測損失

    Args:
        feature_loss_config: 特徵對齊損失配置
        prediction_loss_config: 預測對齊損失配置
        feature_weight: 特徵對齊損失權重
        prediction_weight: 預測對齊損失權重
        detection_weight: 檢測損失權重
        warmup_epochs: 損失權重 warmup 的 epoch 數
        use_multi_scale_feature: 是否使用多尺度特徵對齊
        use_attention_alignment: 是否使用注意力對齊

    Example:
        >>> loss_fn = MTKDLoss(
        ...     feature_loss_config={"loss_type": "cosine"},
        ...     prediction_loss_config={"box_loss_type": "giou"},
        ...     feature_weight=1.0,
        ...     prediction_weight=2.0,
        ... )
        >>> loss, loss_dict = loss_fn(
        ...     student_features=...,
        ...     teacher_features=...,
        ...     student_predictions=...,
        ...     teacher_predictions=...,
        ... )
    """

    def __init__(
        self,
        # Feature alignment config
        feature_loss_config: Optional[Dict[str, Any]] = None,
        # Prediction alignment config
        prediction_loss_config: Optional[Dict[str, Any]] = None,
        # Loss weights
        feature_weight: float = 1.0,
        prediction_weight: float = 2.0,
        detection_weight: float = 1.0,
        # Multi-scale feature config
        use_multi_scale_feature: bool = False,
        multi_scale_config: Optional[Dict[str, Any]] = None,
        # Attention alignment config
        use_attention_alignment: bool = False,
        attention_loss_config: Optional[Dict[str, Any]] = None,
        # Token matching config
        use_token_matching: bool = True,
        token_matching_config: Optional[Dict[str, Any]] = None,
        # Dynamic weight config
        warmup_epochs: int = 0,
        weight_schedule: Optional[str] = None,  # "linear", "cosine", None
    ):
        super().__init__()

        # 保存配置
        self.feature_weight = feature_weight
        self.prediction_weight = prediction_weight
        self.detection_weight = detection_weight
        self.warmup_epochs = warmup_epochs
        self.weight_schedule = weight_schedule

        # 初始化特徵對齊損失
        feature_loss_config = feature_loss_config or {}
        self.feature_loss = FeatureAlignmentLoss(**feature_loss_config)

        # 初始化預測對齊損失
        prediction_loss_config = prediction_loss_config or {}
        self.prediction_loss = PredictionAlignmentLoss(**prediction_loss_config)

        # 多尺度特徵對齊（可選）
        self.use_multi_scale_feature = use_multi_scale_feature
        if use_multi_scale_feature:
            multi_scale_config = multi_scale_config or {}
            self.multi_scale_feature_loss = MultiScaleFeatureAlignmentLoss(**multi_scale_config)

        # 注意力對齊（可選）
        self.use_attention_alignment = use_attention_alignment
        if use_attention_alignment:
            attention_loss_config = attention_loss_config or {}
            self.attention_loss = AttentionAlignmentLoss(**attention_loss_config)

        # Token 匹配損失（可選）
        self.use_token_matching = use_token_matching
        if use_token_matching:
            token_matching_config = token_matching_config or {"token_type": "cls"}
            self.token_matching_loss = TokenMatchingLoss(**token_matching_config)

        # 訓練狀態
        self.register_buffer("current_epoch", torch.tensor(0))

    def get_dynamic_weights(self, epoch: int) -> Tuple[float, float, float]:
        """
        獲取動態調整的損失權重

        Args:
            epoch: 當前 epoch

        Returns:
            (feature_weight, prediction_weight, detection_weight)
        """
        if epoch < self.warmup_epochs:
            # Warmup 期間逐漸增加 KD 損失權重
            warmup_factor = epoch / self.warmup_epochs

            if self.weight_schedule == "linear":
                factor = warmup_factor
            elif self.weight_schedule == "cosine":
                import math
                factor = 0.5 * (1 - math.cos(math.pi * warmup_factor))
            else:
                factor = 1.0

            return (
                self.feature_weight * factor,
                self.prediction_weight * factor,
                self.detection_weight,
            )

        return self.feature_weight, self.prediction_weight, self.detection_weight

    def forward(
        self,
        # DINO feature alignment
        student_features: Optional[torch.Tensor] = None,
        dino_teacher_features: Optional[torch.Tensor] = None,
        # Multi-scale features
        student_multi_scale_features: Optional[List[torch.Tensor]] = None,
        dino_teacher_multi_scale_features: Optional[List[torch.Tensor]] = None,
        # Attention alignment
        student_attention: Optional[torch.Tensor] = None,
        dino_teacher_attention: Optional[torch.Tensor] = None,
        # Prediction alignment (from ensemble teachers)
        student_predictions: Optional[Dict[str, torch.Tensor]] = None,
        ensemble_teacher_predictions: Optional[Dict[str, torch.Tensor]] = None,
        prediction_valid_mask: Optional[torch.Tensor] = None,
        # Detection loss (optional, from ground truth)
        detection_loss: Optional[torch.Tensor] = None,
        # Current epoch for dynamic weighting
        epoch: Optional[int] = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        計算 MTKD 總損失

        Args:
            student_features: Student 的全局特徵 [B, D]
            dino_teacher_features: DINO teacher 的全局特徵 [B, D]
            student_multi_scale_features: Student 多尺度特徵列表
            dino_teacher_multi_scale_features: DINO teacher 多尺度特徵列表
            student_attention: Student 的注意力權重
            dino_teacher_attention: DINO teacher 的注意力權重
            student_predictions: Student 的預測 {"boxes", "logits"}
            ensemble_teacher_predictions: Ensemble teacher 的預測
            prediction_valid_mask: 有效預測的遮罩
            detection_loss: 來自 ground truth 的檢測損失
            epoch: 當前 epoch

        Returns:
            total_loss: 總損失
            loss_dict: 各項損失字典
        """
        loss_dict = {}
        total_loss = torch.tensor(0.0, device=self._get_device())

        # 獲取動態權重
        if epoch is not None:
            feat_w, pred_w, det_w = self.get_dynamic_weights(epoch)
        else:
            feat_w, pred_w, det_w = self.feature_weight, self.prediction_weight, self.detection_weight

        # =============================================================
        # 1. Feature Alignment Loss (DINO Teacher <-> Student)
        # =============================================================
        if student_features is not None and dino_teacher_features is not None:
            # 全局特徵對齊
            feat_align_loss = self.feature_loss(student_features, dino_teacher_features)
            loss_dict["feature_alignment_loss"] = feat_align_loss.detach()
            total_loss = total_loss + feat_w * feat_align_loss

            # Token 匹配損失（如果提供了完整的 token 序列）
            if self.use_token_matching and student_features.dim() == 3:
                token_loss = self.token_matching_loss(student_features, dino_teacher_features)
                loss_dict["token_matching_loss"] = token_loss.detach()
                total_loss = total_loss + feat_w * token_loss

        # 多尺度特徵對齊
        if self.use_multi_scale_feature:
            if student_multi_scale_features is not None and dino_teacher_multi_scale_features is not None:
                ms_feat_loss = self.multi_scale_feature_loss(
                    student_multi_scale_features,
                    dino_teacher_multi_scale_features,
                )
                loss_dict["multi_scale_feature_loss"] = ms_feat_loss.detach()
                total_loss = total_loss + feat_w * ms_feat_loss

        # 注意力對齊
        if self.use_attention_alignment:
            if student_attention is not None and dino_teacher_attention is not None:
                attn_loss = self.attention_loss(student_attention, dino_teacher_attention)
                loss_dict["attention_alignment_loss"] = attn_loss.detach()
                total_loss = total_loss + feat_w * 0.5 * attn_loss  # 注意力損失權重較小

        # =============================================================
        # 2. Prediction Alignment Loss (Ensemble Teachers <-> Student)
        # =============================================================
        if student_predictions is not None and ensemble_teacher_predictions is not None:
            pred_loss, pred_loss_dict = self.prediction_loss(
                student_predictions,
                ensemble_teacher_predictions,
                valid_mask=prediction_valid_mask,
            )
            loss_dict.update(pred_loss_dict)
            total_loss = total_loss + pred_w * pred_loss

        # =============================================================
        # 3. Detection Loss (Optional, from Ground Truth)
        # =============================================================
        if detection_loss is not None:
            loss_dict["detection_loss"] = detection_loss.detach()
            total_loss = total_loss + det_w * detection_loss

        # 記錄總損失和權重
        loss_dict["total_loss"] = total_loss.detach()
        loss_dict["feature_weight"] = torch.tensor(feat_w)
        loss_dict["prediction_weight"] = torch.tensor(pred_w)
        loss_dict["detection_weight"] = torch.tensor(det_w)

        return total_loss, loss_dict

    def _get_device(self) -> torch.device:
        """獲取模組所在的設備"""
        return next(self.parameters()).device if len(list(self.parameters())) > 0 else torch.device("cpu")


class AdaptiveMTKDLoss(MTKDLoss):
    """
    自適應 MTKD 損失

    根據 teacher 和 student 的表現差異動態調整損失權重。

    Args:
        base_feature_weight: 基礎特徵損失權重
        base_prediction_weight: 基礎預測損失權重
        adaptation_rate: 自適應調整速率
    """

    def __init__(
        self,
        base_feature_weight: float = 1.0,
        base_prediction_weight: float = 2.0,
        adaptation_rate: float = 0.1,
        **kwargs,
    ):
        super().__init__(
            feature_weight=base_feature_weight,
            prediction_weight=base_prediction_weight,
            **kwargs,
        )
        self.adaptation_rate = adaptation_rate

        # 追蹤歷史損失
        self.register_buffer("feature_loss_history", torch.zeros(100))
        self.register_buffer("prediction_loss_history", torch.zeros(100))
        self.register_buffer("history_idx", torch.tensor(0))

    def update_history(self, feature_loss: float, prediction_loss: float):
        """更新損失歷史"""
        idx = self.history_idx.item() % 100
        self.feature_loss_history[idx] = feature_loss
        self.prediction_loss_history[idx] = prediction_loss
        self.history_idx += 1

    def get_adaptive_weights(self) -> Tuple[float, float]:
        """
        根據歷史損失計算自適應權重

        Returns:
            (feature_weight, prediction_weight)
        """
        if self.history_idx < 10:
            return self.feature_weight, self.prediction_weight

        # 計算最近的平均損失
        n = min(self.history_idx.item(), 100)
        feat_mean = self.feature_loss_history[:n].mean().item()
        pred_mean = self.prediction_loss_history[:n].mean().item()

        # 根據損失比例調整權重
        total = feat_mean + pred_mean + 1e-8
        feat_ratio = feat_mean / total
        pred_ratio = pred_mean / total

        # 損失較高的部分獲得更多權重
        adapted_feat_w = self.feature_weight * (1 + self.adaptation_rate * (feat_ratio - 0.5))
        adapted_pred_w = self.prediction_weight * (1 + self.adaptation_rate * (pred_ratio - 0.5))

        return adapted_feat_w, adapted_pred_w


class UncertaintyWeightedMTKDLoss(MTKDLoss):
    """
    基於不確定性的加權 MTKD 損失

    使用可學習的權重參數，根據同方差不確定性自動學習損失權重。
    參考: Multi-Task Learning Using Uncertainty to Weigh Losses

    Args:
        num_tasks: 任務數量（損失項數量）
    """

    def __init__(self, num_tasks: int = 3, **kwargs):
        # 暫時設置權重為 1，實際權重由 log_vars 決定
        super().__init__(
            feature_weight=1.0,
            prediction_weight=1.0,
            detection_weight=1.0,
            **kwargs,
        )

        # 可學習的 log variance 參數
        # log_var = log(σ²)，實際權重 = 1/(2σ²) = exp(-log_var)/2
        self.log_vars = nn.Parameter(torch.zeros(num_tasks))

    def forward(
        self,
        student_features: Optional[torch.Tensor] = None,
        dino_teacher_features: Optional[torch.Tensor] = None,
        student_predictions: Optional[Dict[str, torch.Tensor]] = None,
        ensemble_teacher_predictions: Optional[Dict[str, torch.Tensor]] = None,
        detection_loss: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        使用不確定性加權計算損失
        """
        loss_dict = {}
        losses = []

        # Feature alignment loss
        if student_features is not None and dino_teacher_features is not None:
            feat_loss = self.feature_loss(student_features, dino_teacher_features)
            losses.append(feat_loss)
            loss_dict["feature_alignment_loss"] = feat_loss.detach()
        else:
            losses.append(torch.tensor(0.0, device=self.log_vars.device))

        # Prediction alignment loss
        if student_predictions is not None and ensemble_teacher_predictions is not None:
            pred_loss, pred_loss_dict = self.prediction_loss(
                student_predictions,
                ensemble_teacher_predictions,
            )
            losses.append(pred_loss)
            loss_dict.update(pred_loss_dict)
        else:
            losses.append(torch.tensor(0.0, device=self.log_vars.device))

        # Detection loss
        if detection_loss is not None:
            losses.append(detection_loss)
            loss_dict["detection_loss"] = detection_loss.detach()
        else:
            losses.append(torch.tensor(0.0, device=self.log_vars.device))

        # 計算不確定性加權的總損失
        # L_total = Σ (1/(2σ²) * L_i + log(σ))
        #         = Σ (exp(-log_var)/2 * L_i + log_var/2)
        total_loss = torch.tensor(0.0, device=self.log_vars.device)
        for i, loss in enumerate(losses):
            precision = torch.exp(-self.log_vars[i])
            total_loss = total_loss + precision * loss + self.log_vars[i]

        total_loss = total_loss / 2  # 常數因子

        # 記錄學習到的權重
        loss_dict["total_loss"] = total_loss.detach()
        loss_dict["learned_feature_weight"] = torch.exp(-self.log_vars[0]).detach()
        loss_dict["learned_prediction_weight"] = torch.exp(-self.log_vars[1]).detach()
        loss_dict["learned_detection_weight"] = torch.exp(-self.log_vars[2]).detach()

        return total_loss, loss_dict
