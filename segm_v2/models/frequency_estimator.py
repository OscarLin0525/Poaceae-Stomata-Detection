"""
Row-wise 頻率估計器
==================

核心思想：
---------
Stomata 在同一行內呈週期性排列，我們對每一行的特徵做 FFT，
找出該行的「主頻率」，這個頻率代表 stomata 的出現間隔。

為什麼要 row-wise？
-----------------
1. 不同行的 stomata 頻率可能不同
2. 有些行沒有 stomata（頻率 ≈ 0）
3. Row-wise 可以更精準地適應局部變化

FFT 的作用：
----------
```
輸入：一行的特徵序列
      [feat_0, feat_1, feat_2, ..., feat_13]  (14 個 patch)

FFT：
      找出這個序列中的週期性成分

輸出：
      dominant_freq = 0.25  (每 4 個 patch 一個 stomata)
```

注意：14×14 的解析度是粗略的初始估計，
Loss 函數會在訓練過程中做更精細的調整。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Dict, Optional


class RowFrequencyEstimator(nn.Module):
    """
    對每一行進行頻率分析，估計 stomata 的週期性

    輸入：空間特徵 (B, H, W, C)
    輸出：
        - dominant_freq: (B, H) 每行的主頻率
        - freq_confidence: (B, H) 頻率估計的信心度
        - row_features: (B, H, C) 每行的聚合特徵

    架構：
    -----
    1. Feature Projection: 降維以減少計算量
    2. Row-wise FFT: 對每行做 1D FFT
    3. Power Spectrum: 計算頻率能量分佈
    4. Peak Detection: 找出主頻率
    5. Feature Encoding: 編碼頻率資訊供後續使用
    """

    def __init__(
        self,
        embed_dim: int = 768,
        hidden_dim: int = 256,
        num_freq_bins: int = 32,
        use_learnable_filter: bool = True,
        min_freq: float = 0.1,
        max_freq: float = 0.5,
    ):
        """
        Args:
            embed_dim: DINO 特徵維度 (ViT-B = 768)
            hidden_dim: 內部投影維度
            num_freq_bins: 頻率 bin 數量（用於頻率特徵編碼）
            use_learnable_filter: 是否使用可學習的頻率濾波器
            min_freq: 最小有效頻率（排除 DC 和極低頻）
            max_freq: 最大有效頻率
        """
        super().__init__()

        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.num_freq_bins = num_freq_bins
        self.min_freq = min_freq
        self.max_freq = max_freq

        # =========================================
        # 1. Feature Projection
        # =========================================
        # 將高維特徵 (768) 投影到低維 (256)
        # 目的：減少 FFT 計算量，同時保留重要資訊
        self.feature_proj = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )

        # =========================================
        # 2. 可學習的頻率濾波器（可選）
        # =========================================
        # 讓模型學習哪些頻率更重要
        # 例如：stomata 的典型頻率範圍
        if use_learnable_filter:
            # 初始化為 1（不過濾），訓練時會調整
            self.freq_filter = nn.Parameter(torch.ones(num_freq_bins))
        else:
            self.register_buffer("freq_filter", torch.ones(num_freq_bins))

        # =========================================
        # 3. 頻率特徵編碼器
        # =========================================
        # 將頻率資訊編碼回原始維度，供後續 Grid 生成使用
        self.freq_encoder = nn.Sequential(
            nn.Linear(num_freq_bins, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, embed_dim),
        )

        # =========================================
        # 4. 信心度估計器
        # =========================================
        # 估計頻率估計的可信度
        # 如果一行沒有明顯週期性，信心度低，Grid 響應應該也低
        self.confidence_estimator = nn.Sequential(
            nn.Linear(num_freq_bins, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid(),  # 輸出 0~1
        )

    def forward(self, spatial_features: Tensor) -> Dict[str, Tensor]:
        """
        對每一行進行頻率分析

        Args:
            spatial_features: (B, H, W, C) - 空間格式的 DINO 特徵

        Returns:
            dict with:
                - dominant_freq: (B, H) 每行的主頻率 (正規化到 0~1)
                - freq_confidence: (B, H) 頻率估計的信心度
                - freq_spectrum: (B, H, num_freq_bins) 頻率譜
                - row_features: (B, H, C) 頻率編碼後的行特徵
        """
        B, H, W, C = spatial_features.shape

        # =========================================
        # Step 1: Feature Projection
        # =========================================
        # (B, H, W, C) → (B, H, W, hidden_dim)
        projected = self.feature_proj(spatial_features)

        # =========================================
        # Step 2: 計算每行的「信號」
        # =========================================
        # 方法：使用特徵的 L2 norm 作為信號強度
        # 這樣 stomata 位置（特徵較強）會有較高的信號
        # (B, H, W, hidden_dim) → (B, H, W)
        row_signal = projected.norm(dim=-1)

        # 去除 DC 成分（每行減去平均值）
        row_signal = row_signal - row_signal.mean(dim=-1, keepdim=True)

        # =========================================
        # Step 3: Row-wise 1D FFT
        # =========================================
        # 對每行沿著 W 方向做 FFT
        # (B, H, W) → (B, H, W//2+1) [complex]
        fft_result = torch.fft.rfft(row_signal, dim=-1)

        # 計算 power spectrum（能量）
        # (B, H, W//2+1)
        power_spectrum = torch.abs(fft_result) ** 2

        # =========================================
        # Step 4: 頻率 Binning
        # =========================================
        # 將 FFT 結果 interpolate 到固定的 bin 數量
        # 這樣不同輸入尺寸也能用同一個網路處理
        # (B, H, W//2+1) → (B, H, num_freq_bins)
        spectrum_for_interp = power_spectrum.unsqueeze(1)  # (B, 1, H, freq_dim)
        spectrum_for_interp = spectrum_for_interp.permute(0, 1, 3, 2)  # (B, 1, freq_dim, H)

        freq_spectrum = F.interpolate(
            spectrum_for_interp,
            size=(self.num_freq_bins, H),
            mode="bilinear",
            align_corners=False,
        )  # (B, 1, num_freq_bins, H)

        freq_spectrum = freq_spectrum.squeeze(1).permute(0, 2, 1)  # (B, H, num_freq_bins)

        # =========================================
        # Step 5: 應用可學習濾波器
        # =========================================
        # 讓模型學習強調某些頻率範圍
        freq_spectrum = freq_spectrum * F.softplus(self.freq_filter)

        # =========================================
        # Step 6: 找出主頻率
        # =========================================
        # 排除 DC (index 0) 和極低頻
        min_bin = max(1, int(self.min_freq * self.num_freq_bins))
        max_bin = min(self.num_freq_bins, int(self.max_freq * self.num_freq_bins))

        # 在有效範圍內找最大值
        valid_spectrum = freq_spectrum[:, :, min_bin:max_bin]  # (B, H, valid_bins)
        peak_indices = valid_spectrum.argmax(dim=-1)  # (B, H)

        # 轉換為正規化頻率 (0~1)
        dominant_freq = (peak_indices + min_bin).float() / self.num_freq_bins

        # =========================================
        # Step 7: 估計信心度
        # =========================================
        # 基於頻率譜估計這個頻率有多可信
        freq_confidence = self.confidence_estimator(freq_spectrum).squeeze(-1)  # (B, H)

        # =========================================
        # Step 8: 頻率特徵編碼
        # =========================================
        # 將頻率資訊編碼為特徵，供 Grid 生成器使用
        row_features = self.freq_encoder(freq_spectrum)  # (B, H, embed_dim)

        return {
            "dominant_freq": dominant_freq,        # (B, H) 主頻率
            "freq_confidence": freq_confidence,    # (B, H) 信心度
            "freq_spectrum": freq_spectrum,        # (B, H, num_freq_bins) 頻率譜
            "row_features": row_features,          # (B, H, embed_dim) 行特徵
        }


# =============================================================================
# 測試程式碼
# =============================================================================

if __name__ == "__main__":
    print("測試 RowFrequencyEstimator...")

    # 模擬輸入
    B, H, W, C = 2, 14, 14, 768
    spatial_features = torch.randn(B, H, W, C)

    # 建立模型
    estimator = RowFrequencyEstimator(
        embed_dim=C,
        hidden_dim=256,
        num_freq_bins=32,
    )

    # Forward
    output = estimator(spatial_features)

    print(f"輸入: {spatial_features.shape}")
    print(f"dominant_freq: {output['dominant_freq'].shape}")
    print(f"freq_confidence: {output['freq_confidence'].shape}")
    print(f"freq_spectrum: {output['freq_spectrum'].shape}")
    print(f"row_features: {output['row_features'].shape}")

    # 檢查值範圍
    assert output["dominant_freq"].min() >= 0
    assert output["dominant_freq"].max() <= 1
    assert output["freq_confidence"].min() >= 0
    assert output["freq_confidence"].max() <= 1

    print("✅ RowFrequencyEstimator 測試通過！")
