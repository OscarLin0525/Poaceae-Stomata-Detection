"""
Feature Alignment Loss Module

This module implements various feature alignment losses for knowledge distillation
between DINO teacher and student model.

主要功能:
- 特徵對齊損失 (L2, Cosine Similarity, KL Divergence)
- 多尺度特徵對齊
- 注意力圖對齊
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List, Dict, Literal


class FeatureAlignmentLoss(nn.Module):
    """
    特徵對齊損失

    用於對齊 DINO teacher 和 student 的特徵表示。
    支持多種對齊方式：L2、Cosine Similarity、KL Divergence。

    Args:
        loss_type: 損失類型 ("l2", "cosine", "kl", "smooth_l1")
        temperature: KL divergence 的溫度參數
        normalize: 是否對特徵進行 L2 正規化
        reduction: 損失的 reduction 方式 ("mean", "sum", "none")

    Example:
        >>> loss_fn = FeatureAlignmentLoss(loss_type="cosine", normalize=True)
        >>> student_feat = torch.randn(4, 768)  # [B, D]
        >>> teacher_feat = torch.randn(4, 768)  # [B, D]
        >>> loss = loss_fn(student_feat, teacher_feat)
    """

    def __init__(
        self,
        loss_type: Literal["l2", "cosine", "kl", "smooth_l1"] = "l2",
        temperature: float = 1.0,
        normalize: bool = True,
        reduction: Literal["mean", "sum", "none"] = "mean",
    ):
        super().__init__()
        self.loss_type = loss_type
        self.temperature = temperature
        self.normalize = normalize
        self.reduction = reduction

        # 預定義的損失函數
        self._loss_fns = {
            "l2": self._l2_loss,
            "cosine": self._cosine_loss,
            "kl": self._kl_loss,
            "smooth_l1": self._smooth_l1_loss,
        }

    def forward(
        self,
        student_features: torch.Tensor,
        teacher_features: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        計算特徵對齊損失

        Args:
            student_features: Student 模型的特徵 [B, D] 或 [B, N, D]
            teacher_features: Teacher 模型的特徵 [B, D] 或 [B, N, D]
            mask: 可選的遮罩，用於忽略某些位置 [B] 或 [B, N]

        Returns:
            loss: 標量損失值
        """
        # 確保維度匹配
        assert student_features.shape == teacher_features.shape, \
            f"Shape mismatch: student {student_features.shape} vs teacher {teacher_features.shape}"

        # 可選的正規化
        if self.normalize:
            student_features = F.normalize(student_features, p=2, dim=-1)
            teacher_features = F.normalize(teacher_features, p=2, dim=-1)

        # 計算損失
        loss = self._loss_fns[self.loss_type](student_features, teacher_features)

        # 應用遮罩
        if mask is not None:
            # 擴展 mask 到正確的維度
            while mask.dim() < loss.dim():
                mask = mask.unsqueeze(-1)
            loss = loss * mask.float()

            if self.reduction == "mean":
                return loss.sum() / (mask.sum() + 1e-8)
            elif self.reduction == "sum":
                return loss.sum()
            return loss

        # 應用 reduction
        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss

    def _l2_loss(
        self,
        student: torch.Tensor,
        teacher: torch.Tensor,
    ) -> torch.Tensor:
        """
        L2 損失 (MSE)

        公式: L = ||student - teacher||^2
        """
        return F.mse_loss(student, teacher, reduction="none")

    def _cosine_loss(
        self,
        student: torch.Tensor,
        teacher: torch.Tensor,
    ) -> torch.Tensor:
        """
        Cosine Similarity 損失

        公式: L = 1 - cos(student, teacher)

        範圍: [0, 2]，0 表示完全相同，2 表示完全相反
        """
        # 計算 cosine similarity
        cos_sim = F.cosine_similarity(student, teacher, dim=-1)
        # 轉換為損失 (1 - similarity)
        return 1 - cos_sim

    def _kl_loss(
        self,
        student: torch.Tensor,
        teacher: torch.Tensor,
    ) -> torch.Tensor:
        """
        KL Divergence 損失

        公式: L = KL(softmax(teacher/T) || softmax(student/T))

        適用於將特徵視為概率分佈的情況
        """
        student_log_prob = F.log_softmax(student / self.temperature, dim=-1)
        teacher_prob = F.softmax(teacher / self.temperature, dim=-1)

        # KL divergence: sum(p * log(p/q)) = sum(p * (log_p - log_q))
        kl_div = F.kl_div(student_log_prob, teacher_prob, reduction="none")

        # 乘以 T^2 以保持梯度規模
        return kl_div.sum(dim=-1) * (self.temperature ** 2)

    def _smooth_l1_loss(
        self,
        student: torch.Tensor,
        teacher: torch.Tensor,
    ) -> torch.Tensor:
        """
        Smooth L1 損失 (Huber Loss)

        對異常值更魯棒
        """
        return F.smooth_l1_loss(student, teacher, reduction="none")


class MultiScaleFeatureAlignmentLoss(nn.Module):
    """
    多尺度特徵對齊損失

    用於對齊多個尺度/層級的特徵，適用於 FPN 等多尺度架構。

    Args:
        num_scales: 尺度數量
        loss_type: 損失類型
        scale_weights: 各尺度的權重，None 表示等權重
        align_channels: 是否需要通道對齊（當 teacher 和 student 通道數不同時）
        student_channels: Student 各尺度的通道數列表
        teacher_channels: Teacher 各尺度的通道數列表

    Example:
        >>> loss_fn = MultiScaleFeatureAlignmentLoss(
        ...     num_scales=4,
        ...     student_channels=[256, 256, 256, 256],
        ...     teacher_channels=[768, 768, 768, 768],
        ... )
        >>> student_feats = [torch.randn(4, 256, 56, 56), ...]  # 4 scales
        >>> teacher_feats = [torch.randn(4, 768, 56, 56), ...]
        >>> loss = loss_fn(student_feats, teacher_feats)
    """

    def __init__(
        self,
        num_scales: int = 4,
        loss_type: str = "l2",
        scale_weights: Optional[List[float]] = None,
        align_channels: bool = True,
        student_channels: Optional[List[int]] = None,
        teacher_channels: Optional[List[int]] = None,
        normalize: bool = True,
    ):
        super().__init__()
        self.num_scales = num_scales
        self.align_channels = align_channels
        self.normalize = normalize

        # 設定尺度權重
        if scale_weights is None:
            scale_weights = [1.0] * num_scales
        assert len(scale_weights) == num_scales
        self.register_buffer("scale_weights", torch.tensor(scale_weights))

        # 基礎損失函數
        self.base_loss = FeatureAlignmentLoss(
            loss_type=loss_type,
            normalize=normalize,
            reduction="mean",
        )

        # 通道對齊層（如果需要）
        self.channel_aligners = None
        if align_channels and student_channels and teacher_channels:
            self.channel_aligners = nn.ModuleList([
                nn.Conv2d(s_ch, t_ch, kernel_size=1, bias=False)
                if s_ch != t_ch else nn.Identity()
                for s_ch, t_ch in zip(student_channels, teacher_channels)
            ])

    def forward(
        self,
        student_features: List[torch.Tensor],
        teacher_features: List[torch.Tensor],
    ) -> torch.Tensor:
        """
        計算多尺度特徵對齊損失

        Args:
            student_features: Student 特徵列表 [scale1, scale2, ...]
                             每個元素形狀: [B, C, H, W]
            teacher_features: Teacher 特徵列表

        Returns:
            total_loss: 加權總損失
        """
        assert len(student_features) == len(teacher_features) == self.num_scales, \
            f"Expected {self.num_scales} scales, got student: {len(student_features)}, teacher: {len(teacher_features)}"

        total_loss = 0.0
        loss_dict = {}

        for i, (s_feat, t_feat) in enumerate(zip(student_features, teacher_features)):
            # 通道對齊
            if self.channel_aligners is not None:
                s_feat = self.channel_aligners[i](s_feat)

            # 空間尺寸對齊（如果需要）
            if s_feat.shape[-2:] != t_feat.shape[-2:]:
                s_feat = F.interpolate(
                    s_feat,
                    size=t_feat.shape[-2:],
                    mode="bilinear",
                    align_corners=False,
                )

            # 重塑為 [B, N, C] 格式
            B, C, H, W = s_feat.shape
            s_feat = s_feat.flatten(2).transpose(1, 2)  # [B, H*W, C]
            t_feat = t_feat.flatten(2).transpose(1, 2)  # [B, H*W, C]

            # 計算損失
            scale_loss = self.base_loss(s_feat, t_feat)
            weighted_loss = scale_loss * self.scale_weights[i]

            total_loss += weighted_loss
            loss_dict[f"scale_{i}_loss"] = scale_loss.detach()

        return total_loss


class AttentionAlignmentLoss(nn.Module):
    """
    注意力圖對齊損失

    對齊 teacher 和 student 的注意力權重分佈。

    Args:
        loss_type: 損失類型 ("kl", "mse", "cosine")
        temperature: softmax 溫度

    Example:
        >>> loss_fn = AttentionAlignmentLoss(loss_type="kl", temperature=1.0)
        >>> student_attn = torch.randn(4, 12, 197, 197)  # [B, heads, N, N]
        >>> teacher_attn = torch.randn(4, 12, 197, 197)
        >>> loss = loss_fn(student_attn, teacher_attn)
    """

    def __init__(
        self,
        loss_type: Literal["kl", "mse", "cosine"] = "kl",
        temperature: float = 1.0,
    ):
        super().__init__()
        self.loss_type = loss_type
        self.temperature = temperature

    def forward(
        self,
        student_attention: torch.Tensor,
        teacher_attention: torch.Tensor,
    ) -> torch.Tensor:
        """
        計算注意力對齊損失

        Args:
            student_attention: Student 注意力權重 [B, heads, N, N]
            teacher_attention: Teacher 注意力權重 [B, heads, N, N]

        Returns:
            loss: 標量損失
        """
        if self.loss_type == "kl":
            # 對最後一個維度計算 softmax
            student_prob = F.log_softmax(student_attention / self.temperature, dim=-1)
            teacher_prob = F.softmax(teacher_attention / self.temperature, dim=-1)

            loss = F.kl_div(student_prob, teacher_prob, reduction="batchmean")
            return loss * (self.temperature ** 2)

        elif self.loss_type == "mse":
            return F.mse_loss(student_attention, teacher_attention)

        elif self.loss_type == "cosine":
            # 展平後計算 cosine similarity
            B, H, N1, N2 = student_attention.shape
            s_flat = student_attention.reshape(B * H, -1)
            t_flat = teacher_attention.reshape(B * H, -1)
            cos_sim = F.cosine_similarity(s_flat, t_flat, dim=-1)
            return (1 - cos_sim).mean()

        else:
            raise ValueError(f"Unknown loss type: {self.loss_type}")


class TokenMatchingLoss(nn.Module):
    """
    Token 匹配損失

    用於對齊 student 和 teacher 的特定 token（如 CLS token、patch tokens）。
    支持選擇性對齊特定位置的 tokens。

    Args:
        token_type: Token 類型 ("cls", "patch", "all")
        loss_type: 損失類型

    Example:
        >>> loss_fn = TokenMatchingLoss(token_type="cls")
        >>> student_tokens = torch.randn(4, 197, 768)  # [B, 1+196, D]
        >>> teacher_tokens = torch.randn(4, 197, 768)
        >>> loss = loss_fn(student_tokens, teacher_tokens)
    """

    def __init__(
        self,
        token_type: Literal["cls", "patch", "all"] = "all",
        loss_type: str = "cosine",
        normalize: bool = True,
    ):
        super().__init__()
        self.token_type = token_type
        self.base_loss = FeatureAlignmentLoss(
            loss_type=loss_type,
            normalize=normalize,
            reduction="mean",
        )

    def forward(
        self,
        student_tokens: torch.Tensor,
        teacher_tokens: torch.Tensor,
        num_prefix_tokens: int = 1,  # CLS token 數量
    ) -> torch.Tensor:
        """
        計算 token 匹配損失

        Args:
            student_tokens: Student tokens [B, N, D]
            teacher_tokens: Teacher tokens [B, N, D]
            num_prefix_tokens: 前綴 token 數量（通常是 CLS token）

        Returns:
            loss: 標量損失
        """
        if self.token_type == "cls":
            # 只對齊 CLS token
            s_tokens = student_tokens[:, :num_prefix_tokens, :]
            t_tokens = teacher_tokens[:, :num_prefix_tokens, :]

        elif self.token_type == "patch":
            # 只對齊 patch tokens
            s_tokens = student_tokens[:, num_prefix_tokens:, :]
            t_tokens = teacher_tokens[:, num_prefix_tokens:, :]

        else:  # "all"
            s_tokens = student_tokens
            t_tokens = teacher_tokens

        return self.base_loss(s_tokens, t_tokens)
