"""
SEGM Adapter
============

這是插入 DINO Block 之間的核心模組，整合了：
1. RowFrequencyEstimator - 頻率估計
2. PeriodicGridGenerator - Grid 生成
3. Feature Modulation - 特徵調變

設計原則：
---------
1. Zero-Init Gate: 初始時 gate=0，不影響 DINO 輸出
   - 這確保訓練開始時模型行為與原始 DINO 相同
   - 隨著訓練，gate 逐漸打開，SEGM 開始起作用

2. 殘差連接: output = input + gate * delta
   - 這樣即使 SEGM 預測錯誤，也不會完全破壞原始特徵

3. 輕量設計: 只增加約 2-3M 參數
   - DINO ViT-B 有約 86M 參數
   - SEGM 額外增加很少

架構圖：
-------
```
輸入: patch_tokens (B, H*W, C)
        │
        ▼
┌───────────────────────┐
│  Reshape to Spatial   │  (B, H*W, C) → (B, H, W, C)
└───────────────────────┘
        │
        ▼
┌───────────────────────┐
│ RowFrequencyEstimator │  估計每行的週期頻率
│                       │  輸出: dominant_freq (B, H)
│                       │        freq_confidence (B, H)
└───────────────────────┘
        │
        ▼
┌───────────────────────┐
│ PeriodicGridGenerator │  生成週期性遮罩
│                       │  輸出: grid (B, 1, H, W)
└───────────────────────┘
        │
        ▼
┌───────────────────────┐
│   Channel Projection  │  (B, 1, H, W) → (B, C, H, W)
│      (1×1 Conv)       │
└───────────────────────┘
        │
        ▼
┌───────────────────────┐
│    Zero-Init Gate     │  delta = sigmoid(gate) * projected
│                       │  gate 初始化為 -5 (sigmoid ≈ 0)
└───────────────────────┘
        │
        ▼
┌───────────────────────┐
│  Residual Connection  │  output = input + delta
└───────────────────────┘
        │
        ▼
輸出: enhanced_tokens (B, H*W, C)
```
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Dict, Optional, Tuple

from .tensor_utils import reshape_patch_tokens_to_spatial, reshape_spatial_to_patch_tokens
from .frequency_estimator import RowFrequencyEstimator
from .grid_generator import PeriodicGridGenerator


class SEGMAdapter(nn.Module):
    """
    SEGM 適配器：插入 DINO Block 之間的週期性增強模組

    這個模組：
    1. 接收 DINO 的 patch tokens
    2. 分析週期性
    3. 生成週期性 Grid
    4. 用 Grid 調變特徵
    5. 返回增強後的 tokens

    關鍵設計：
    ---------
    - 輸入輸出格式與 DINO Block 相同: (B, N, C)
    - Zero-init gate 確保初始時不影響 DINO
    - 可以獨立訓練，DINO 參數凍結
    """

    def __init__(
        self,
        embed_dim: int = 768,
        num_freq_bins: int = 32,
        hidden_dim: int = 256,
        init_kappa: float = 3.0,
        init_gate: float = -5.0,
        modulation_mode: str = "multiplicative",
        use_fft_filter: bool = False,
        fft_filter_type: str = "highpass",
        fft_cutoff: float = 0.2,
        fft_bandwidth: float = 0.15,
        fft_strength: float = 0.5,
    ):
        """
        Args:
            embed_dim: DINO 特徵維度 (ViT-B = 768, ViT-L = 1024)
            num_freq_bins: 頻率分析的 bin 數量
            hidden_dim: 內部隱藏層維度
            init_kappa: Von Mises 尖銳度初始值
            init_gate: Gate 初始值 (sigmoid(init_gate) ≈ 0 時不影響輸出)
            modulation_mode: 調變模式
                - "multiplicative": output = input * (1 + gate * grid)
                - "additive": output = input + gate * projected_grid
            use_fft_filter: 是否啟用 2D FFT FilterBank
            fft_filter_type: 濾波類型 ("highpass" 或 "bandpass")
            fft_cutoff: 截止頻率（0~1，正規化半徑）
            fft_bandwidth: bandpass 頻寬（0~1）
            fft_strength: 濾波混合強度（0~1）
        """
        super().__init__()

        self.embed_dim = embed_dim
        self.modulation_mode = modulation_mode
        self.use_fft_filter = use_fft_filter
        self.fft_filter_type = fft_filter_type
        self.fft_cutoff = fft_cutoff
        self.fft_bandwidth = fft_bandwidth
        self.fft_strength = fft_strength

        if self.fft_filter_type not in {"highpass", "bandpass"}:
            raise ValueError(f"Unknown fft_filter_type: {self.fft_filter_type}")
        if not 0.0 <= self.fft_cutoff <= 1.0:
            raise ValueError(f"fft_cutoff must be in [0, 1], got {self.fft_cutoff}")
        if not 0.0 <= self.fft_bandwidth <= 1.0:
            raise ValueError(f"fft_bandwidth must be in [0, 1], got {self.fft_bandwidth}")
        if not 0.0 <= self.fft_strength <= 1.0:
            raise ValueError(f"fft_strength must be in [0, 1], got {self.fft_strength}")

        # =========================================
        # 1. 頻率估計器
        # =========================================
        self.freq_estimator = RowFrequencyEstimator(
            embed_dim=embed_dim,
            hidden_dim=hidden_dim,
            num_freq_bins=num_freq_bins,
        )

        # =========================================
        # 2. Grid 生成器
        # =========================================
        self.grid_generator = PeriodicGridGenerator(
            max_height=32,  # 支援到 32×32 patch grid
            init_kappa=init_kappa,
        )

        # =========================================
        # 3. Channel Projection（用於 additive 模式）
        # =========================================
        # 將單通道 Grid 投影到 embed_dim 通道
        if modulation_mode == "additive":
            self.channel_proj = nn.Conv2d(
                in_channels=1,
                out_channels=embed_dim,
                kernel_size=1,
                bias=False,
            )
            # 初始化為小值
            nn.init.normal_(self.channel_proj.weight, std=0.02)

        # =========================================
        # 4. Zero-Init Gate（關鍵！）
        # =========================================
        # 初始化為很小的值，sigmoid(init_gate) ≈ 0
        # 這樣訓練開始時 SEGM 不會影響 DINO 的輸出
        self.gate = nn.Parameter(torch.tensor(init_gate))

        # =========================================
        # 5. 儲存中間結果（用於 Loss 計算和視覺化）
        # =========================================
        self._cache = {}

    def _build_fft_mask(
        self,
        H: int,
        W: int,
        device: torch.device,
    ) -> Tensor:
        """建立 2D 頻域遮罩，半徑範圍正規化到 [0, 1]。"""
        fy = torch.fft.fftfreq(H, device=device, dtype=torch.float32)
        fx = torch.fft.fftfreq(W, device=device, dtype=torch.float32)
        yy, xx = torch.meshgrid(fy, fx, indexing="ij")
        radius = torch.sqrt(yy ** 2 + xx ** 2) / (2.0 ** 0.5 * 0.5 + 1e-8)

        if self.fft_filter_type == "highpass":
            mask = (radius >= self.fft_cutoff).to(torch.float32)
        else:
            half_bw = 0.5 * self.fft_bandwidth
            low = max(0.0, self.fft_cutoff - half_bw)
            high = min(1.0, self.fft_cutoff + half_bw)
            mask = ((radius >= low) & (radius <= high)).to(torch.float32)

        return mask

    def _apply_fft_filter(self, spatial_features: Tensor) -> Tuple[Tensor, Dict]:
        """
        對空間特徵做 2D FFT 濾波，並以 residual 方式混合回原始特徵。
        """
        _, H, W, _ = spatial_features.shape
        x_bchw = spatial_features.permute(0, 3, 1, 2)  # (B, C, H, W)
        x_for_fft = x_bchw.float()

        fft_x = torch.fft.fft2(x_for_fft, dim=(-2, -1))
        mask = self._build_fft_mask(H, W, x_bchw.device).unsqueeze(0).unsqueeze(0)
        filtered_fft = fft_x * mask

        filtered = torch.fft.ifft2(filtered_fft, dim=(-2, -1)).real
        blended = (1.0 - self.fft_strength) * x_for_fft + self.fft_strength * filtered
        blended = blended.to(x_bchw.dtype)

        fft_info = {
            "mask_mean": mask.mean(),
            "strength": torch.tensor(self.fft_strength, device=x_bchw.device, dtype=torch.float32),
        }
        return blended.permute(0, 2, 3, 1), fft_info

    def forward(
        self,
        patch_tokens: Tensor,
        H: int,
        W: int,
        return_intermediates: bool = False,
    ) -> Tensor | Tuple[Tensor, Dict]:
        """
        對 patch tokens 進行週期性增強

        Args:
            patch_tokens: (B, N, C) - DINO 的 patch tokens（不含 CLS）
            H: 空間高度（patch 數量）
            W: 空間寬度（patch 數量）
            return_intermediates: 是否返回中間結果（用於 Loss 計算）

        Returns:
            enhanced_tokens: (B, N, C) - 增強後的 tokens
            intermediates: dict (可選) - 中間結果
        """
        B, N, C = patch_tokens.shape

        # 驗證維度
        assert N == H * W, f"N ({N}) != H*W ({H}*{W}={H*W})"
        assert C == self.embed_dim, f"C ({C}) != embed_dim ({self.embed_dim})"

        # =========================================
        # Step 1: 轉換為空間格式
        # =========================================
        # (B, N, C) → (B, H, W, C)
        spatial_features = reshape_patch_tokens_to_spatial(patch_tokens, H, W)
        filtered_features = spatial_features
        fft_info = {}

        # 可選：先做一次 2D FFT FilterBank，再進入頻率估計與 Grid 調變
        if self.use_fft_filter:
            filtered_features, fft_info = self._apply_fft_filter(spatial_features)

        # =========================================
        # Step 2: 頻率估計
        # =========================================
        freq_info = self.freq_estimator(filtered_features)
        # freq_info 包含:
        #   - dominant_freq: (B, H) 主頻率
        #   - freq_confidence: (B, H) 信心度
        #   - freq_spectrum: (B, H, num_freq_bins) 頻率譜
        #   - row_features: (B, H, embed_dim) 行特徵

        # =========================================
        # Step 3: 生成週期性 Grid
        # =========================================
        grid = self.grid_generator(
            dominant_freq=freq_info["dominant_freq"],
            freq_confidence=freq_info["freq_confidence"],
            H=H,
            W=W,
        )  # (B, 1, H, W)

        # =========================================
        # Step 4: 計算 Gate 值
        # =========================================
        gate_value = torch.sigmoid(self.gate)

        # =========================================
        # Step 5: 特徵調變
        # =========================================
        if self.modulation_mode == "multiplicative":
            # 乘法調變: output = input * (1 + gate * (grid - 0.5) * 2)
            # grid 在 [0,1]，轉換到 [-1, 1]
            # 這樣 grid=0.5 時不改變，grid=1 時增強，grid=0 時抑制
            grid_centered = (grid - 0.5) * 2  # [-1, 1]

            # (B, 1, H, W) → (B, H, W, 1)
            grid_for_mult = grid_centered.permute(0, 2, 3, 1)

            # 調變
            modulation = 1.0 + gate_value * grid_for_mult
            enhanced_spatial = filtered_features * modulation  # (B, H, W, C)

        else:  # additive
            # 加法調變: output = input + gate * projected_grid

            # Channel projection: (B, 1, H, W) → (B, C, H, W)
            projected_grid = self.channel_proj(grid)

            # (B, C, H, W) → (B, H, W, C)
            projected_grid = projected_grid.permute(0, 2, 3, 1)

            # 調變
            delta = gate_value * projected_grid
            enhanced_spatial = filtered_features + delta  # (B, H, W, C)

        # =========================================
        # Step 6: 轉回序列格式
        # =========================================
        # (B, H, W, C) → (B, N, C)
        enhanced_tokens = reshape_spatial_to_patch_tokens(enhanced_spatial)

        # =========================================
        # Step 7: 儲存中間結果
        # =========================================
        self._cache = {
            "spatial_features_raw": spatial_features,
            "spatial_features": filtered_features,
            "freq_info": freq_info,
            "grid": grid,
            "gate_value": gate_value,
            "enhanced_spatial": enhanced_spatial,
            "fft_info": fft_info,
        }

        if return_intermediates:
            return enhanced_tokens, self._cache
        else:
            return enhanced_tokens

    def get_grid(self) -> Optional[Tensor]:
        """取得最後一次 forward 生成的 Grid（用於視覺化）"""
        return self._cache.get("grid", None)

    def get_gate_value(self) -> float:
        """取得當前的 gate 值"""
        return torch.sigmoid(self.gate).item()

    def get_freq_info(self) -> Optional[Dict]:
        """取得最後一次 forward 的頻率資訊"""
        return self._cache.get("freq_info", None)


# =============================================================================
# 測試程式碼
# =============================================================================

if __name__ == "__main__":
    print("測試 SEGMAdapter...")

    # 模擬 DINO patch tokens
    B, H, W, C = 2, 14, 14, 768
    N = H * W
    patch_tokens = torch.randn(B, N, C)

    # 建立 SEGM Adapter
    adapter = SEGMAdapter(
        embed_dim=C,
        num_freq_bins=32,
        hidden_dim=256,
        init_gate=-5.0,  # sigmoid(-5) ≈ 0.007
    )

    # Forward (不返回中間結果)
    enhanced_tokens = adapter(patch_tokens, H, W)
    print(f"輸入: {patch_tokens.shape}")
    print(f"輸出: {enhanced_tokens.shape}")
    print(f"Gate 值: {adapter.get_gate_value():.4f}")

    # 驗證初始時幾乎不改變輸出
    diff = (enhanced_tokens - patch_tokens).abs().mean()
    print(f"初始差異 (應該很小): {diff:.6f}")

    # Forward (返回中間結果)
    enhanced_tokens, intermediates = adapter(patch_tokens, H, W, return_intermediates=True)
    print(f"\n中間結果:")
    print(f"  - grid: {intermediates['grid'].shape}")
    print(f"  - dominant_freq: {intermediates['freq_info']['dominant_freq'].shape}")
    print(f"  - freq_confidence: {intermediates['freq_info']['freq_confidence'].shape}")

    # 驗證輸出形狀
    assert enhanced_tokens.shape == patch_tokens.shape

    print("\n✅ SEGMAdapter 測試通過！")

    # 測試 multiplicative 模式
    print("\n測試 multiplicative 模式...")
    adapter_mult = SEGMAdapter(
        embed_dim=C,
        modulation_mode="multiplicative",
    )
    enhanced_mult = adapter_mult(patch_tokens, H, W)
    print(f"Multiplicative 輸出: {enhanced_mult.shape}")
    print("✅ Multiplicative 模式測試通過！")
