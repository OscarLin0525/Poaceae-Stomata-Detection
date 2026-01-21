"""
Student Model Module

This module implements the student detector model that will be trained
using Multi-Teacher Knowledge Distillation.

主要功能:
- Student Backbone (可替換)
- Student Detection Head
- 特徵適配器（用於與 DINO teacher 對齊）
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, List, Tuple, Any, Literal
import math


class FeatureAdapter(nn.Module):
    """
    特徵適配器

    將 student 的特徵維度轉換到與 teacher 相同的維度，
    用於特徵對齊損失計算。

    Args:
        student_dim: Student 特徵維度
        teacher_dim: Teacher 特徵維度
        adapter_type: 適配器類型
            - "linear": 簡單線性投影
            - "mlp": 兩層 MLP
            - "attention": 使用 attention 進行適配

    Example:
        >>> adapter = FeatureAdapter(256, 768, adapter_type="mlp")
        >>> student_feat = torch.randn(4, 256)
        >>> aligned_feat = adapter(student_feat)  # [4, 768]
    """

    def __init__(
        self,
        student_dim: int,
        teacher_dim: int,
        adapter_type: Literal["linear", "mlp", "attention"] = "mlp",
        hidden_dim: Optional[int] = None,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.student_dim = student_dim
        self.teacher_dim = teacher_dim
        self.adapter_type = adapter_type

        hidden_dim = hidden_dim or max(student_dim, teacher_dim)

        if adapter_type == "linear":
            self.adapter = nn.Linear(student_dim, teacher_dim)

        elif adapter_type == "mlp":
            self.adapter = nn.Sequential(
                nn.Linear(student_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, teacher_dim),
            )

        elif adapter_type == "attention":
            self.adapter = nn.MultiheadAttention(
                embed_dim=teacher_dim,
                num_heads=8,
                dropout=dropout,
                batch_first=True,
            )
            self.query_proj = nn.Linear(student_dim, teacher_dim)
            self.key_proj = nn.Linear(student_dim, teacher_dim)
            self.value_proj = nn.Linear(student_dim, teacher_dim)

        else:
            raise ValueError(f"Unknown adapter type: {adapter_type}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        適配 student 特徵

        Args:
            x: Student 特徵 [B, D_s] 或 [B, N, D_s]

        Returns:
            adapted: 適配後的特徵 [B, D_t] 或 [B, N, D_t]
        """
        if self.adapter_type in ["linear", "mlp"]:
            return self.adapter(x)

        elif self.adapter_type == "attention":
            # Self-attention based adaptation
            q = self.query_proj(x)
            k = self.key_proj(x)
            v = self.value_proj(x)

            if x.dim() == 2:
                q = q.unsqueeze(1)
                k = k.unsqueeze(1)
                v = v.unsqueeze(1)

            out, _ = self.adapter(q, k, v)

            if x.dim() == 2:
                out = out.squeeze(1)

            return out


class MultiScaleFeatureAdapter(nn.Module):
    """
    多尺度特徵適配器

    用於適配 FPN 等多尺度特徵。

    Args:
        student_channels: Student 各尺度的通道數列表
        teacher_channels: Teacher 各尺度的通道數列表
        adapter_type: 適配器類型
    """

    def __init__(
        self,
        student_channels: List[int],
        teacher_channels: List[int],
        adapter_type: str = "mlp",
    ):
        super().__init__()
        assert len(student_channels) == len(teacher_channels)

        self.adapters = nn.ModuleList([
            FeatureAdapter(s_ch, t_ch, adapter_type=adapter_type)
            for s_ch, t_ch in zip(student_channels, teacher_channels)
        ])

    def forward(self, features: List[torch.Tensor]) -> List[torch.Tensor]:
        """
        適配多尺度特徵

        Args:
            features: Student 多尺度特徵列表

        Returns:
            adapted_features: 適配後的特徵列表
        """
        return [adapter(feat) for adapter, feat in zip(self.adapters, features)]


class StudentBackbone(nn.Module):
    """
    Student Backbone 模組

    封裝 student 的 backbone 網絡，提供統一的接口。
    支持多種 backbone 類型。

    Args:
        backbone_type: Backbone 類型 ("resnet50", "efficientnet", "swin", "custom")
        pretrained: 是否使用預訓練權重
        out_channels: 輸出通道數
        return_layers: 要返回的層

    Example:
        >>> backbone = StudentBackbone(backbone_type="resnet50", pretrained=True)
        >>> features = backbone(images)  # Dict of multi-scale features
    """

    def __init__(
        self,
        backbone_type: str = "resnet50",
        pretrained: bool = True,
        out_channels: int = 256,
        return_layers: Optional[Dict[str, str]] = None,
        custom_backbone: Optional[nn.Module] = None,
    ):
        super().__init__()
        self.backbone_type = backbone_type
        self.out_channels = out_channels

        if backbone_type == "custom" and custom_backbone is not None:
            self.backbone = custom_backbone
            self._out_channels = out_channels

        elif backbone_type == "resnet50":
            self._init_resnet50(pretrained, return_layers)

        elif backbone_type == "efficientnet":
            self._init_efficientnet(pretrained)

        elif backbone_type == "swin":
            self._init_swin(pretrained)

        else:
            raise ValueError(f"Unknown backbone type: {backbone_type}")

    def _init_resnet50(self, pretrained: bool, return_layers: Optional[Dict]):
        """初始化 ResNet-50 backbone"""
        try:
            from torchvision.models import resnet50, ResNet50_Weights
            from torchvision.models._utils import IntermediateLayerGetter

            if pretrained:
                weights = ResNet50_Weights.IMAGENET1K_V2
            else:
                weights = None

            backbone = resnet50(weights=weights)

            # 移除分類頭
            return_layers = return_layers or {
                "layer1": "0",
                "layer2": "1",
                "layer3": "2",
                "layer4": "3",
            }

            self.backbone = IntermediateLayerGetter(backbone, return_layers=return_layers)
            self._out_channels = [256, 512, 1024, 2048]

        except ImportError:
            raise ImportError("torchvision is required for ResNet backbone")

    def _init_efficientnet(self, pretrained: bool):
        """初始化 EfficientNet backbone"""
        try:
            from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights

            if pretrained:
                weights = EfficientNet_B0_Weights.IMAGENET1K_V1
            else:
                weights = None

            self.backbone = efficientnet_b0(weights=weights).features
            self._out_channels = [24, 40, 112, 320]  # EfficientNet-B0 通道數

        except ImportError:
            raise ImportError("torchvision is required for EfficientNet backbone")

    def _init_swin(self, pretrained: bool):
        """初始化 Swin Transformer backbone"""
        try:
            from torchvision.models import swin_t, Swin_T_Weights

            if pretrained:
                weights = Swin_T_Weights.IMAGENET1K_V1
            else:
                weights = None

            self.backbone = swin_t(weights=weights).features
            self._out_channels = [96, 192, 384, 768]  # Swin-T 通道數

        except ImportError:
            raise ImportError("torchvision is required for Swin backbone")

    @property
    def out_channels_list(self) -> List[int]:
        """返回各層的輸出通道數"""
        return self._out_channels

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        提取多尺度特徵

        Args:
            x: 輸入圖像 [B, 3, H, W]

        Returns:
            features: 特徵字典 {"0": feat0, "1": feat1, ...}
        """
        if self.backbone_type == "resnet50":
            return self.backbone(x)

        elif self.backbone_type in ["efficientnet", "swin"]:
            # 手動提取多尺度特徵
            features = {}
            for i, layer in enumerate(self.backbone):
                x = layer(x)
                if i in [1, 2, 4, 6]:  # 特定層的輸出
                    features[str(len(features))] = x
            return features

        else:
            return self.backbone(x)


class DetectionHead(nn.Module):
    """
    Detection Head

    用於物體檢測的預測頭，輸出 bounding boxes 和類別。

    Args:
        in_channels: 輸入通道數
        num_classes: 類別數
        num_queries: 查詢數量（類 DETR 架構）
        hidden_dim: 隱藏層維度
        num_heads: 注意力頭數
        num_layers: Transformer 層數
    """

    def __init__(
        self,
        in_channels: int = 256,
        num_classes: int = 1,
        num_queries: int = 100,
        hidden_dim: int = 256,
        num_heads: int = 8,
        num_layers: int = 6,
        use_focal_loss: bool = True,
    ):
        super().__init__()
        self.num_queries = num_queries
        self.num_classes = num_classes
        self.hidden_dim = hidden_dim
        self.use_focal_loss = use_focal_loss

        # 輸入投影
        self.input_proj = nn.Conv2d(in_channels, hidden_dim, kernel_size=1)

        # 可學習的查詢
        self.query_embed = nn.Embedding(num_queries, hidden_dim)

        # Transformer decoder
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=0.1,
            activation="relu",
            batch_first=True,
        )
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)

        # 預測頭
        self.class_embed = nn.Linear(hidden_dim, num_classes + 1)  # +1 for background
        self.bbox_embed = MLP(hidden_dim, hidden_dim, 4, 3)

    def forward(
        self,
        features: torch.Tensor,
        pos_embed: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        前向傳播

        Args:
            features: 輸入特徵 [B, C, H, W]
            pos_embed: 位置編碼 [B, H*W, C]

        Returns:
            predictions: {"boxes": [B, num_queries, 4], "logits": [B, num_queries, num_classes+1]}
        """
        B = features.shape[0]
        device = features.device

        # 投影特徵
        src = self.input_proj(features)  # [B, hidden_dim, H, W]
        H, W = src.shape[-2:]
        src = src.flatten(2).transpose(1, 2)  # [B, H*W, hidden_dim]

        # 獲取查詢
        queries = self.query_embed.weight.unsqueeze(0).expand(B, -1, -1)  # [B, num_queries, hidden_dim]

        # Transformer decoder
        hs = self.transformer_decoder(queries, src)  # [B, num_queries, hidden_dim]

        # 預測
        outputs_class = self.class_embed(hs)  # [B, num_queries, num_classes+1]
        outputs_coord = self.bbox_embed(hs).sigmoid()  # [B, num_queries, 4]

        return {
            "logits": outputs_class,
            "boxes": outputs_coord,
        }


class MLP(nn.Module):
    """簡單的 MLP"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(
            nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim])
        )

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


class StudentDetector(nn.Module):
    """
    Student Detector

    完整的 student 檢測模型，包含:
    - Backbone: 特徵提取
    - Neck (FPN): 多尺度特徵融合
    - Head: 檢測預測
    - Feature Adapter: 與 teacher 特徵對齊

    Args:
        backbone_config: Backbone 配置
        neck_config: Neck 配置
        head_config: Head 配置
        adapter_config: 特徵適配器配置
        dino_teacher_dim: DINO teacher 的特徵維度

    Example:
        >>> student = StudentDetector(
        ...     backbone_config={"backbone_type": "resnet50"},
        ...     head_config={"num_classes": 1, "num_queries": 100},
        ...     dino_teacher_dim=768,
        ... )
        >>> outputs = student(images)
    """

    def __init__(
        self,
        backbone_config: Optional[Dict[str, Any]] = None,
        neck_config: Optional[Dict[str, Any]] = None,
        head_config: Optional[Dict[str, Any]] = None,
        adapter_config: Optional[Dict[str, Any]] = None,
        dino_teacher_dim: int = 768,
        custom_backbone: Optional[nn.Module] = None,
        custom_head: Optional[nn.Module] = None,
    ):
        super().__init__()

        # 默認配置
        backbone_config = backbone_config or {"backbone_type": "resnet50", "pretrained": True}
        head_config = head_config or {"num_classes": 1, "num_queries": 100}
        adapter_config = adapter_config or {"adapter_type": "mlp"}

        # 初始化 backbone
        if custom_backbone is not None:
            self.backbone = custom_backbone
            backbone_out_channels = [256, 512, 1024, 2048]  # 假設的默認值
        else:
            self.backbone = StudentBackbone(**backbone_config)
            backbone_out_channels = self.backbone.out_channels_list

        # 初始化 FPN Neck
        self.neck = self._build_fpn(backbone_out_channels, 256)

        # 初始化檢測頭
        if custom_head is not None:
            self.head = custom_head
        else:
            self.head = DetectionHead(in_channels=256, **head_config)

        # 初始化特徵適配器（用於與 DINO teacher 對齊）
        # 使用全局平均池化後的特徵維度
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.feature_adapter = FeatureAdapter(
            student_dim=256,  # FPN 輸出通道數
            teacher_dim=dino_teacher_dim,
            **adapter_config,
        )

        # 多尺度特徵適配器
        self.multi_scale_adapter = MultiScaleFeatureAdapter(
            student_channels=[256, 256, 256, 256],  # FPN 統一的通道數
            teacher_channels=[dino_teacher_dim] * 4,
            adapter_type=adapter_config.get("adapter_type", "mlp"),
        )

    def _build_fpn(self, in_channels_list: List[int], out_channels: int) -> nn.Module:
        """構建 FPN neck"""
        return SimpleFPN(in_channels_list, out_channels)

    def forward(
        self,
        images: torch.Tensor,
        return_features: bool = False,
        return_adapted_features: bool = True,
    ) -> Dict[str, torch.Tensor]:
        """
        前向傳播

        Args:
            images: 輸入圖像 [B, 3, H, W]
            return_features: 是否返回原始特徵
            return_adapted_features: 是否返回適配後的特徵

        Returns:
            outputs: {
                "boxes": [B, num_queries, 4],
                "logits": [B, num_queries, num_classes+1],
                "features": 原始特徵 (可選),
                "adapted_features": 適配後的特徵 (可選),
                "adapted_multi_scale_features": 多尺度適配特徵 (可選),
            }
        """
        # 提取 backbone 特徵
        backbone_features = self.backbone(images)

        # FPN 特徵融合
        fpn_features = self.neck(list(backbone_features.values()))

        # 使用最高解析度的特徵進行檢測
        detection_features = fpn_features[0]  # [B, 256, H/4, W/4]

        # 檢測預測
        predictions = self.head(detection_features)

        outputs = {
            "boxes": predictions["boxes"],
            "logits": predictions["logits"],
        }

        # 可選：返回原始特徵
        if return_features:
            outputs["features"] = fpn_features
            outputs["backbone_features"] = backbone_features

        # 可選：返回適配後的特徵（用於與 DINO teacher 對齊）
        if return_adapted_features:
            # 全局特徵（使用全局平均池化）
            global_feat = self.global_pool(detection_features).flatten(1)  # [B, 256]
            adapted_global_feat = self.feature_adapter(global_feat)  # [B, teacher_dim]
            outputs["adapted_features"] = adapted_global_feat

            # 多尺度適配特徵
            adapted_ms_feats = self.multi_scale_adapter(fpn_features)
            outputs["adapted_multi_scale_features"] = adapted_ms_feats

        return outputs

    def get_detection_output(self, images: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        只獲取檢測輸出（用於推理）

        Args:
            images: [B, 3, H, W]

        Returns:
            {"boxes": [...], "scores": [...], "labels": [...]}
        """
        outputs = self.forward(images, return_features=False, return_adapted_features=False)

        # 將 logits 轉換為 scores 和 labels
        probs = F.softmax(outputs["logits"], dim=-1)
        scores, labels = probs[..., :-1].max(dim=-1)  # 排除背景類

        return {
            "boxes": outputs["boxes"],
            "scores": scores,
            "labels": labels,
        }


class SimpleFPN(nn.Module):
    """
    簡化版 FPN (Feature Pyramid Network)

    將多尺度 backbone 特徵融合成統一通道數的金字塔特徵。
    """

    def __init__(
        self,
        in_channels_list: List[int],
        out_channels: int,
    ):
        super().__init__()
        self.in_channels_list = in_channels_list
        self.out_channels = out_channels

        # 橫向連接（1x1 conv）
        self.lateral_convs = nn.ModuleList([
            nn.Conv2d(in_ch, out_channels, kernel_size=1)
            for in_ch in in_channels_list
        ])

        # 輸出卷積（3x3 conv）
        self.output_convs = nn.ModuleList([
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
            for _ in in_channels_list
        ])

    def forward(self, features: List[torch.Tensor]) -> List[torch.Tensor]:
        """
        FPN 前向傳播

        Args:
            features: 多尺度特徵列表 [low_res -> high_res]

        Returns:
            fpn_features: 融合後的特徵列表
        """
        assert len(features) == len(self.lateral_convs)

        # 橫向連接
        laterals = [conv(feat) for conv, feat in zip(self.lateral_convs, features)]

        # 自頂向下融合
        for i in range(len(laterals) - 2, -1, -1):
            # 上採樣
            upsampled = F.interpolate(
                laterals[i + 1],
                size=laterals[i].shape[-2:],
                mode="nearest",
            )
            laterals[i] = laterals[i] + upsampled

        # 輸出卷積
        outputs = [conv(lat) for conv, lat in zip(self.output_convs, laterals)]

        return outputs
