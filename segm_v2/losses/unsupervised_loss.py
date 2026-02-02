"""
SEGM v2 自監督損失函數
=====================

核心思想：
---------
由於沒有標註資料，我們需要設計損失函數來引導 SEGM 學習：
1. Grid 的峰值位置應該有「一致」的特徵（同類物體）
2. Grid 的峰值和谷值區域應該有「對比」（不同類物體）
3. Grid 應該是週期性的
4. Grid 應該是稀疏的（大部分區域是背景）
5. 相鄰行的頻率應該平滑變化

損失函數組成：
------------
```
L_total = w1 * L_intra      # 峰值區域特徵一致性
        + w2 * L_inter      # 峰值 vs 谷值對比
        + w3 * L_period     # 週期性約束
        + w4 * L_sparse     # 稀疏性約束
        + w5 * L_freq_smooth # 頻率平滑約束
```

關鍵原理（為什麼這些 Loss 能 work）：
----------------------------------

1. L_intra（群體一致性）:
   - 假設：stomata 在 DINO 特徵空間中彼此相似
   - 作用：讓 Grid 峰值聚集在「特徵一致」的位置
   - 效果：自然排除特徵不一致的 noise

2. L_inter（群體對比）:
   - 假設：stomata 和背景的特徵不同
   - 作用：確保 Grid 真的在區分前景和背景
   - 效果：避免 Grid 退化成全亮或全暗

3. L_period（週期性）:
   - 假設：stomata 呈週期性排列
   - 作用：強制 Grid 保持週期結構
   - 效果：這是核心機制 - 不在週期位置的 noise 被壓低

4. L_sparse（稀疏性）:
   - 假設：stomata 只佔圖像的小部分
   - 作用：避免 Grid 全亮
   - 效果：確保只有「真正的」峰值被保留

5. L_freq_smooth（頻率平滑）:
   - 假設：相鄰行的 stomata 頻率應該相似
   - 作用：避免頻率估計劇烈跳動
   - 效果：產生更穩定的 Grid
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Dict, Optional
import math


class IntraGroupConsistencyLoss(nn.Module):
    """
    L_intra: 群體內一致性損失

    目的：讓 Grid 峰值位置的特徵彼此相似

    原理：
    -----
    如果 Grid 正確地標記了 stomata 位置，
    那這些位置的 DINO 特徵應該彼此相似（因為都是 stomata）。

    計算方式：
    ---------
    1. 用 Grid 作為權重，提取「峰值加權平均特徵」
    2. 計算每個位置特徵與平均特徵的相似度
    3. 用 Grid 加權這些相似度
    4. 最大化相似度（最小化負相似度）

    公式：
    -----
    feat_mean = Σ(grid * features) / Σ(grid)
    similarity = cosine_sim(features, feat_mean)
    L_intra = -Σ(grid * similarity) / Σ(grid)
    """

    def __init__(self, temperature: float = 0.1):
        """
        Args:
            temperature: 相似度計算的溫度參數
        """
        super().__init__()
        self.temperature = temperature

    def forward(
        self,
        grid: Tensor,
        features: Tensor,
    ) -> Tensor:
        """
        Args:
            grid: (B, 1, H, W) - Grid 值，範圍 [0, 1]
            features: (B, H, W, C) - 空間特徵

        Returns:
            loss: scalar - 一致性損失（越小越好）
        """
        B, _, H, W = grid.shape
        C = features.shape[-1]

        # (B, 1, H, W) → (B, H, W, 1)
        grid_weights = grid.permute(0, 2, 3, 1)

        # 避免除以零
        grid_sum = grid_weights.sum(dim=(1, 2), keepdim=True) + 1e-8

        # =========================================
        # Step 1: 計算 Grid 加權平均特徵
        # =========================================
        # (B, H, W, C) * (B, H, W, 1) → (B, H, W, C)
        weighted_features = features * grid_weights
        # (B, 1, 1, C)
        mean_feature = weighted_features.sum(dim=(1, 2), keepdim=True) / grid_sum

        # =========================================
        # Step 2: 計算每個位置與平均的相似度
        # =========================================
        # Normalize
        features_norm = F.normalize(features, dim=-1)
        mean_feature_norm = F.normalize(mean_feature, dim=-1)

        # Cosine similarity: (B, H, W)
        similarity = (features_norm * mean_feature_norm).sum(dim=-1)

        # =========================================
        # Step 3: Grid 加權的相似度
        # =========================================
        # 我們希望 Grid 高的位置相似度也高
        weighted_similarity = (similarity * grid_weights.squeeze(-1)).sum(dim=(1, 2))
        weighted_similarity = weighted_similarity / grid_sum.squeeze(-1).squeeze(-1)

        # =========================================
        # Step 4: 計算損失（最大化相似度 = 最小化負相似度）
        # =========================================
        loss = -weighted_similarity.mean()

        return loss


class InterGroupContrastLoss(nn.Module):
    """
    L_inter: 群體間對比損失

    目的：讓 Grid 峰值區域和谷值區域的特徵不同

    原理：
    -----
    如果 Grid 正確區分了 stomata 和背景，
    那峰值區域（stomata）和谷值區域（背景）的特徵應該不同。

    計算方式：
    ---------
    1. 用 Grid 提取峰值區域的平均特徵
    2. 用 (1-Grid) 提取谷值區域的平均特徵
    3. 最小化兩者的相似度

    公式：
    -----
    fg_feat = Σ(grid * features) / Σ(grid)
    bg_feat = Σ((1-grid) * features) / Σ(1-grid)
    L_inter = cosine_sim(fg_feat, bg_feat)
    """

    def __init__(self, margin: float = 0.5):
        """
        Args:
            margin: 對比邊界，希望相似度低於這個值
        """
        super().__init__()
        self.margin = margin

    def forward(
        self,
        grid: Tensor,
        features: Tensor,
    ) -> Tensor:
        """
        Args:
            grid: (B, 1, H, W) - Grid 值
            features: (B, H, W, C) - 空間特徵

        Returns:
            loss: scalar - 對比損失
        """
        B, _, H, W = grid.shape

        # (B, 1, H, W) → (B, H, W, 1)
        grid_weights = grid.permute(0, 2, 3, 1)
        bg_weights = 1.0 - grid_weights

        # 避免除以零
        grid_sum = grid_weights.sum(dim=(1, 2), keepdim=True) + 1e-8
        bg_sum = bg_weights.sum(dim=(1, 2), keepdim=True) + 1e-8

        # =========================================
        # 計算前景和背景的平均特徵
        # =========================================
        fg_feat = (features * grid_weights).sum(dim=(1, 2)) / grid_sum.squeeze(2)  # (B, C)
        bg_feat = (features * bg_weights).sum(dim=(1, 2)) / bg_sum.squeeze(2)      # (B, C)

        # Normalize
        fg_feat_norm = F.normalize(fg_feat, dim=-1)
        bg_feat_norm = F.normalize(bg_feat, dim=-1)

        # =========================================
        # 計算相似度（希望越低越好）
        # =========================================
        similarity = (fg_feat_norm * bg_feat_norm).sum(dim=-1)  # (B,)

        # Margin loss: 只有相似度 > margin 時才有損失
        loss = F.relu(similarity - self.margin).mean()

        return loss


class PeriodicityLoss(nn.Module):
    """
    L_period: 週期性約束損失

    目的：確保 Grid 保持週期性結構

    原理：
    -----
    這是 FilterBank 的核心機制。
    我們希望 Grid 的實際波形與估計的頻率一致。

    計算方式：
    ---------
    1. 對 Grid 每行做 FFT
    2. 找出 Grid 的主頻率
    3. 與 RowFrequencyEstimator 估計的頻率比較
    4. 最小化差異

    這個 Loss 確保：
    - 頻率估計和 Grid 生成是一致的
    - Grid 真的是週期性的（不是隨機的）
    """

    def __init__(self):
        super().__init__()

    def forward(
        self,
        grid: Tensor,
        estimated_freq: Tensor,
    ) -> Tensor:
        """
        Args:
            grid: (B, 1, H, W) - 生成的 Grid
            estimated_freq: (B, H) - RowFrequencyEstimator 估計的頻率

        Returns:
            loss: scalar - 週期性損失
        """
        B, _, H, W = grid.shape

        # 取出 Grid (去掉 channel 維度)
        grid_2d = grid.squeeze(1)  # (B, H, W)

        # =========================================
        # 對每行做 FFT，找出 Grid 的實際主頻率
        # =========================================
        # Row-wise FFT
        grid_fft = torch.fft.rfft(grid_2d, dim=-1)  # (B, H, W//2+1)
        grid_spectrum = torch.abs(grid_fft)

        # 排除 DC (index 0)
        grid_spectrum_no_dc = grid_spectrum[:, :, 1:]  # (B, H, W//2)

        # 找主頻率
        peak_indices = grid_spectrum_no_dc.argmax(dim=-1) + 1  # (B, H)

        # 正規化到 [0, 1]
        grid_freq = peak_indices.float() / W

        # =========================================
        # 計算與估計頻率的差異
        # =========================================
        freq_diff = (grid_freq - estimated_freq).abs()
        loss = freq_diff.mean()

        return loss


class SparsityLoss(nn.Module):
    """
    L_sparse: 稀疏性約束損失

    目的：確保 Grid 是稀疏的（大部分區域是低值）

    原理：
    -----
    Stomata 只佔圖像的小部分，所以 Grid 應該大部分是低值。
    如果沒有這個約束，Grid 可能退化成全亮。

    計算方式：
    ---------
    L_sparse = mean(grid)

    目標稀疏度：
    - 典型的 stomata 可能佔 5-20% 的面積
    - 所以 target_sparsity ≈ 0.1-0.2
    """

    def __init__(self, target_sparsity: float = 0.15):
        """
        Args:
            target_sparsity: 目標稀疏度（Grid 平均值的目標）
        """
        super().__init__()
        self.target_sparsity = target_sparsity

    def forward(self, grid: Tensor) -> Tensor:
        """
        Args:
            grid: (B, 1, H, W) - Grid 值

        Returns:
            loss: scalar - 稀疏性損失
        """
        # 計算 Grid 的平均值
        grid_mean = grid.mean()

        # 與目標稀疏度的差異
        loss = (grid_mean - self.target_sparsity).abs()

        return loss


class FrequencySmoothLoss(nn.Module):
    """
    L_freq_smooth: 頻率平滑約束損失

    目的：確保相鄰行的頻率估計平滑變化

    原理：
    -----
    Stomata 的排列通常是連續的，相鄰行的頻率不應該劇烈變化。
    這個約束可以：
    1. 避免頻率估計的雜訊
    2. 產生更穩定的 Grid

    計算方式：
    ---------
    L_freq_smooth = mean(|freq[h] - freq[h-1]|)
    """

    def __init__(self):
        super().__init__()

    def forward(self, dominant_freq: Tensor) -> Tensor:
        """
        Args:
            dominant_freq: (B, H) - 每行的主頻率

        Returns:
            loss: scalar - 頻率平滑損失
        """
        # 計算相鄰行的頻率差異
        freq_diff = (dominant_freq[:, 1:] - dominant_freq[:, :-1]).abs()

        loss = freq_diff.mean()

        return loss


class UnsupervisedSEGMLoss(nn.Module):
    """
    SEGM 自監督損失函數（組合所有子損失）

    使用方式：
    ---------
    ```python
    loss_fn = UnsupervisedSEGMLoss()

    # 在訓練迴圈中
    output = model(images)
    intermediates = model.get_intermediates()

    loss, loss_dict = loss_fn(
        grid=intermediates['grid'],
        features=intermediates['spatial_features'],
        dominant_freq=intermediates['freq_info']['dominant_freq'],
    )
    ```
    """

    def __init__(
        self,
        intra_weight: float = 1.0,
        inter_weight: float = 0.5,
        period_weight: float = 0.5,
        sparse_weight: float = 0.3,
        freq_smooth_weight: float = 0.2,
        target_sparsity: float = 0.15,
        inter_margin: float = 0.3,
    ):
        """
        Args:
            intra_weight: L_intra 權重
            inter_weight: L_inter 權重
            period_weight: L_period 權重
            sparse_weight: L_sparse 權重
            freq_smooth_weight: L_freq_smooth 權重
            target_sparsity: 目標稀疏度
            inter_margin: 對比損失的邊界
        """
        super().__init__()

        self.weights = {
            "intra": intra_weight,
            "inter": inter_weight,
            "period": period_weight,
            "sparse": sparse_weight,
            "freq_smooth": freq_smooth_weight,
        }

        # 子損失函數
        self.intra_loss = IntraGroupConsistencyLoss()
        self.inter_loss = InterGroupContrastLoss(margin=inter_margin)
        self.period_loss = PeriodicityLoss()
        self.sparse_loss = SparsityLoss(target_sparsity=target_sparsity)
        self.freq_smooth_loss = FrequencySmoothLoss()

    def forward(
        self,
        grid: Tensor,
        features: Tensor,
        dominant_freq: Tensor,
        freq_confidence: Optional[Tensor] = None,
    ) -> tuple:
        """
        計算總損失

        Args:
            grid: (B, 1, H, W) - 生成的 Grid
            features: (B, H, W, C) - 空間特徵
            dominant_freq: (B, H) - 估計的主頻率
            freq_confidence: (B, H) - 頻率信心度（可選）

        Returns:
            total_loss: scalar - 總損失
            loss_dict: dict - 各項損失的詳細資訊
        """
        loss_dict = {}

        # =========================================
        # 計算各項損失
        # =========================================

        # L_intra: 群體內一致性
        l_intra = self.intra_loss(grid, features)
        loss_dict["l_intra"] = l_intra.item()

        # L_inter: 群體間對比
        l_inter = self.inter_loss(grid, features)
        loss_dict["l_inter"] = l_inter.item()

        # L_period: 週期性約束
        l_period = self.period_loss(grid, dominant_freq)
        loss_dict["l_period"] = l_period.item()

        # L_sparse: 稀疏性約束
        l_sparse = self.sparse_loss(grid)
        loss_dict["l_sparse"] = l_sparse.item()

        # L_freq_smooth: 頻率平滑約束
        l_freq_smooth = self.freq_smooth_loss(dominant_freq)
        loss_dict["l_freq_smooth"] = l_freq_smooth.item()

        # =========================================
        # 加權總和
        # =========================================
        total_loss = (
            self.weights["intra"] * l_intra +
            self.weights["inter"] * l_inter +
            self.weights["period"] * l_period +
            self.weights["sparse"] * l_sparse +
            self.weights["freq_smooth"] * l_freq_smooth
        )

        loss_dict["total"] = total_loss.item()

        # 額外統計
        loss_dict["grid_mean"] = grid.mean().item()
        loss_dict["grid_std"] = grid.std().item()
        loss_dict["freq_mean"] = dominant_freq.mean().item()

        return total_loss, loss_dict


# =============================================================================
# 測試程式碼
# =============================================================================

if __name__ == "__main__":
    print("測試 UnsupervisedSEGMLoss...")

    B, H, W, C = 2, 14, 14, 768

    # 模擬輸入
    grid = torch.rand(B, 1, H, W) * 0.3  # 稀疏的 Grid
    features = torch.randn(B, H, W, C)
    dominant_freq = torch.rand(B, H) * 0.3 + 0.1  # 頻率在 0.1-0.4

    # 建立損失函數
    loss_fn = UnsupervisedSEGMLoss()

    # 計算損失
    total_loss, loss_dict = loss_fn(grid, features, dominant_freq)

    print(f"總損失: {total_loss.item():.4f}")
    print(f"各項損失:")
    for k, v in loss_dict.items():
        print(f"  - {k}: {v:.4f}")

    # 驗證梯度
    grid.requires_grad = True
    total_loss, _ = loss_fn(grid, features, dominant_freq)
    total_loss.backward()

    print(f"\nGrid 梯度範圍: [{grid.grad.min():.4f}, {grid.grad.max():.4f}]")

    print("\n✅ UnsupervisedSEGMLoss 測試通過！")
