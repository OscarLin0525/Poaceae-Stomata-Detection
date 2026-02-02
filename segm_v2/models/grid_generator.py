"""
週期性 Grid 生成器
=================

核心功能：
---------
根據 RowFrequencyEstimator 估計的頻率，生成週期性的 Grid。
這個 Grid 作為「attention mask」，在週期位置給予高響應，
非週期位置給予低響應。

週期性波形選擇：
--------------
使用 Von Mises 分佈（圓形高斯）作為波形：
- 比 sin 波更「尖銳」，更適合點狀目標（stomata）
- kappa 參數控制尖銳度：kappa 大 → 更尖，kappa 小 → 更平緩
- 可微分，適合梯度訓練

Von Mises 公式：
    f(x) = exp(kappa * cos(2π * freq * x + phase)) / I_0(kappa)

簡化版（我們用這個）：
    f(x) = exp(kappa * (cos(2π * freq * x + phase) - 1))

Grid 生成邏輯：
-------------
```
對於每一行 h：
    freq_h = dominant_freq[h]        # 從 FFT 估計
    phase_h = learned_phase[h]       # 可學習相位
    confidence_h = freq_confidence[h] # 頻率估計信心度

    row_wave[h, :] = VonMises(x_coords; freq_h, phase_h, kappa)
    row_wave[h, :] *= confidence_h    # 信心度低的行，響應也低

grid = row_wave  # (B, H, W)
```
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
import math
from typing import Optional, Tuple


class PeriodicGridGenerator(nn.Module):
    """
    根據估計的頻率生成週期性 Grid

    輸入：
        - dominant_freq: (B, H) 每行的主頻率
        - freq_confidence: (B, H) 頻率估計信心度
        - H, W: 空間維度

    輸出：
        - grid: (B, 1, H, W) 週期性遮罩

    特點：
    -----
    1. Row-wise 不同頻率：每行可以有不同的週期
    2. 可學習相位：讓模型調整波形對齊
    3. 可學習尖銳度：控制峰值的寬度
    4. 信心度加權：無週期的行響應低
    """

    def __init__(
        self,
        max_height: int = 14,
        init_kappa: float = 3.0,
        learnable_phase: bool = True,
        learnable_kappa: bool = True,
        use_column_modulation: bool = False,
    ):
        """
        Args:
            max_height: 最大行數（用於初始化可學習參數）
            init_kappa: Von Mises 尖銳度初始值
                        - kappa=1: 很平緩（接近均勻）
                        - kappa=3: 中等尖銳
                        - kappa=10: 很尖銳（接近脈衝）
            learnable_phase: 是否學習每行的相位
            learnable_kappa: 是否學習尖銳度
            use_column_modulation: 是否加入垂直方向的調變（實驗性）
        """
        super().__init__()

        self.max_height = max_height
        self.use_column_modulation = use_column_modulation

        # =========================================
        # 可學習的相位 (per-row)
        # =========================================
        # 每行可以有不同的相位偏移
        # 因為 stomata 不一定從 x=0 開始
        if learnable_phase:
            # 初始化為 0（無相位偏移）
            self.row_phase = nn.Parameter(torch.zeros(max_height))
        else:
            self.register_buffer("row_phase", torch.zeros(max_height))

        # =========================================
        # 可學習的尖銳度 (global 或 per-row)
        # =========================================
        if learnable_kappa:
            # 使用 log-scale 參數，確保 kappa > 0
            # kappa = softplus(kappa_raw) + 1
            self.kappa_raw = nn.Parameter(torch.tensor(math.log(math.exp(init_kappa - 1) - 1)))
        else:
            self.register_buffer("kappa_raw", torch.tensor(math.log(math.exp(init_kappa - 1) - 1)))

        # =========================================
        # 可選：垂直方向調變
        # =========================================
        # 如果 stomata 在垂直方向也有規律，可以加入
        if use_column_modulation:
            self.col_freq = nn.Parameter(torch.tensor(0.2))
            self.col_phase = nn.Parameter(torch.tensor(0.0))

    @property
    def kappa(self) -> Tensor:
        """確保 kappa > 1（否則波形太平）"""
        return F.softplus(self.kappa_raw) + 1.0

    def _generate_von_mises_wave(
        self,
        freq: Tensor,
        phase: Tensor,
        length: int,
        kappa: Tensor,
    ) -> Tensor:
        """
        生成 Von Mises 週期波形

        Von Mises 是「圓形高斯」，在週期位置產生尖銳的峰值，
        比 sin 波更適合點狀目標。

        Args:
            freq: (B, H) 頻率（正規化，0~1 代表 0~Nyquist）
            phase: (H,) 或 (B, H) 相位偏移
            length: 波形長度 W
            kappa: 尖銳度參數

        Returns:
            wave: (B, H, W) Von Mises 波形
        """
        B, H = freq.shape

        # 建立 x 座標: [0, 1, 2, ..., W-1]
        x = torch.arange(length, dtype=freq.dtype, device=freq.device)
        x = x.view(1, 1, length)  # (1, 1, W)

        # 頻率和相位的維度調整
        freq = freq.unsqueeze(-1)  # (B, H, 1)

        if phase.dim() == 1:
            phase = phase.view(1, H, 1)  # (1, H, 1)
        else:
            phase = phase.unsqueeze(-1)  # (B, H, 1)

        # Von Mises 核心公式：
        # f(x) = exp(kappa * (cos(2π * freq * x + phase) - 1))
        #
        # 減 1 是為了讓峰值 = 1（當 cos = 1 時）
        # 谷值 = exp(-2*kappa)（當 cos = -1 時）
        angle = 2 * math.pi * freq * x + phase * 2 * math.pi
        wave = torch.exp(kappa * (torch.cos(angle) - 1))

        return wave  # (B, H, W)

    def forward(
        self,
        dominant_freq: Tensor,
        freq_confidence: Tensor,
        H: int,
        W: int,
    ) -> Tensor:
        """
        生成週期性 Grid

        Args:
            dominant_freq: (B, H) 每行的主頻率
            freq_confidence: (B, H) 頻率估計信心度
            H: 空間高度
            W: 空間寬度

        Returns:
            grid: (B, 1, H, W) 週期性遮罩，值域 [0, 1]
        """
        B = dominant_freq.shape[0]
        device = dominant_freq.device

        # =========================================
        # Step 1: 取得相位參數
        # =========================================
        # 確保相位在有效範圍 [0, 1]
        phase = torch.sigmoid(self.row_phase[:H])  # (H,)

        # =========================================
        # Step 2: 生成水平方向的週期波形
        # =========================================
        row_wave = self._generate_von_mises_wave(
            freq=dominant_freq,       # (B, H)
            phase=phase,              # (H,)
            length=W,
            kappa=self.kappa,
        )  # (B, H, W)

        # =========================================
        # Step 3: 信心度加權
        # =========================================
        # 頻率估計信心度低的行，Grid 響應也應該低
        # 這避免在「沒有週期性」的行產生錯誤的 Grid
        confidence = freq_confidence.unsqueeze(-1)  # (B, H, 1)
        row_wave = row_wave * confidence

        # =========================================
        # Step 4: 可選的垂直調變
        # =========================================
        if self.use_column_modulation:
            # 生成垂直方向的調變
            y = torch.arange(H, dtype=row_wave.dtype, device=device)
            y = y.view(1, H, 1)  # (1, H, 1)
            col_wave = torch.exp(
                self.kappa * (torch.cos(2 * math.pi * self.col_freq * y + self.col_phase * 2 * math.pi) - 1)
            )  # (1, H, 1)
            row_wave = row_wave * col_wave

        # =========================================
        # Step 5: 正規化到 [0, 1]
        # =========================================
        # 確保輸出在合理範圍
        grid = row_wave.clamp(0, 1)

        # 加入 channel 維度: (B, H, W) → (B, 1, H, W)
        grid = grid.unsqueeze(1)

        return grid

    def visualize_grid(
        self,
        grid: Tensor,
        save_path: Optional[str] = None,
    ):
        """
        視覺化生成的 Grid（debug 用）
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            print("matplotlib not installed")
            return

        # 取第一個 batch
        grid_np = grid[0, 0].detach().cpu().numpy()

        plt.figure(figsize=(10, 8))
        plt.imshow(grid_np, cmap='hot', aspect='auto')
        plt.colorbar(label='Grid Value')
        plt.xlabel('Width (patches)')
        plt.ylabel('Height (patches)')
        plt.title(f'Periodic Grid (kappa={self.kappa.item():.2f})')

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        else:
            plt.show()
        plt.close()


# =============================================================================
# 測試程式碼
# =============================================================================

if __name__ == "__main__":
    print("測試 PeriodicGridGenerator...")

    B, H, W = 2, 14, 14

    # 模擬頻率估計結果
    # 假設：
    # - 有些行有明顯頻率 (stomata 行)
    # - 有些行沒有 (非 stomata 行)
    dominant_freq = torch.zeros(B, H)
    freq_confidence = torch.zeros(B, H)

    # 模擬 stomata 行（每隔一行）
    for h in [0, 2, 4, 6, 8, 10, 12]:
        dominant_freq[:, h] = 0.25  # 每 4 個 patch 一個 stomata
        freq_confidence[:, h] = 0.9

    # 非 stomata 行
    for h in [1, 3, 5, 7, 9, 11, 13]:
        dominant_freq[:, h] = 0.0   # 無週期
        freq_confidence[:, h] = 0.1

    # 建立 Grid 生成器
    generator = PeriodicGridGenerator(
        max_height=H,
        init_kappa=3.0,
    )

    # 生成 Grid
    grid = generator(dominant_freq, freq_confidence, H, W)

    print(f"dominant_freq: {dominant_freq.shape}")
    print(f"freq_confidence: {freq_confidence.shape}")
    print(f"grid: {grid.shape}")
    print(f"grid value range: [{grid.min():.4f}, {grid.max():.4f}]")
    print(f"kappa: {generator.kappa.item():.2f}")

    # 驗證
    assert grid.shape == (B, 1, H, W)
    assert grid.min() >= 0
    assert grid.max() <= 1

    print("✅ PeriodicGridGenerator 測試通過！")

    # 可選：視覺化
    # generator.visualize_grid(grid, save_path="test_grid.png")
