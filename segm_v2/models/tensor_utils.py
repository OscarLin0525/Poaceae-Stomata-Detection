"""
Tensor 格式轉換工具
==================

DINOv3 與 SEGM 的 Tensor 格式不同，需要轉換：

DINOv3 格式（序列）:
    patch_tokens: (B, N, C)
    - B: batch size
    - N: patch 數量 (例如 196 = 14×14)
    - C: 特徵維度 (例如 768)

SEGM 格式（空間）:
    spatial_features: (B, H, W, C)
    - B: batch size
    - H, W: 空間維度 (例如 14×14)
    - C: 特徵維度

轉換關係：
    N = H × W
    序列順序是 row-major (先橫後縱)
"""

import torch
from torch import Tensor
from typing import Tuple


def reshape_patch_tokens_to_spatial(
    patch_tokens: Tensor,
    H: int = None,
    W: int = None,
) -> Tensor:
    """
    將 DINOv3 的 patch tokens 轉換為空間格式

    這是 SEGM 處理的第一步：把 1D 序列恢復成 2D 空間結構，
    這樣才能做 row-wise 的頻率分析。

    Args:
        patch_tokens: (B, N, C) - DINOv3 輸出的 patch tokens
        H: 空間高度，如果不指定則假設正方形
        W: 空間寬度，如果不指定則假設正方形

    Returns:
        spatial_features: (B, H, W, C) - 空間格式的特徵

    Example:
        >>> tokens = torch.randn(2, 196, 768)  # batch=2, 14×14 patches, dim=768
        >>> spatial = reshape_patch_tokens_to_spatial(tokens)
        >>> spatial.shape
        torch.Size([2, 14, 14, 768])
    """
    B, N, C = patch_tokens.shape

    # 如果沒指定 H, W，假設是正方形
    if H is None or W is None:
        H = W = int(N ** 0.5)
        # 驗證是否為完美正方形
        if H * W != N:
            raise ValueError(
                f"patch_tokens 數量 {N} 不是完美正方形。"
                f"請明確指定 H 和 W。"
            )

    # 驗證 H × W = N
    if H * W != N:
        raise ValueError(
            f"H×W ({H}×{W}={H*W}) 與 patch 數量 ({N}) 不匹配"
        )

    # Reshape: (B, N, C) → (B, H, W, C)
    # DINOv3 的 patch 是 row-major 順序
    spatial_features = patch_tokens.view(B, H, W, C)

    return spatial_features


def reshape_spatial_to_patch_tokens(
    spatial_features: Tensor,
) -> Tensor:
    """
    將空間格式轉回 DINOv3 的 patch tokens 格式

    這是 SEGM 處理的最後一步：把處理完的 2D 特徵
    轉回 1D 序列，以便繼續 DINO 的後續 Block。

    Args:
        spatial_features: (B, H, W, C) - 空間格式的特徵

    Returns:
        patch_tokens: (B, N, C) - 序列格式的 patch tokens

    Example:
        >>> spatial = torch.randn(2, 14, 14, 768)
        >>> tokens = reshape_spatial_to_patch_tokens(spatial)
        >>> tokens.shape
        torch.Size([2, 196, 768])
    """
    B, H, W, C = spatial_features.shape

    # Reshape: (B, H, W, C) → (B, H*W, C)
    patch_tokens = spatial_features.view(B, H * W, C)

    return patch_tokens


def get_spatial_dims_from_tokens(
    patch_tokens: Tensor,
    assume_square: bool = True,
) -> Tuple[int, int]:
    """
    從 patch tokens 推斷空間維度

    Args:
        patch_tokens: (B, N, C)
        assume_square: 是否假設正方形

    Returns:
        (H, W): 空間維度
    """
    N = patch_tokens.shape[1]

    if assume_square:
        H = W = int(N ** 0.5)
        if H * W != N:
            raise ValueError(f"N={N} 不是完美正方形")
    else:
        # 嘗試因數分解，優先選擇接近正方形的
        for h in range(int(N ** 0.5), 0, -1):
            if N % h == 0:
                H = h
                W = N // h
                break

    return H, W


# =============================================================================
# 測試程式碼
# =============================================================================

if __name__ == "__main__":
    print("測試 Tensor 格式轉換...")

    # 模擬 DINOv3 輸出
    B, N, C = 2, 196, 768  # batch=2, 14×14 patches, dim=768
    patch_tokens = torch.randn(B, N, C)

    # 轉換為空間格式
    spatial = reshape_patch_tokens_to_spatial(patch_tokens)
    print(f"Patch tokens: {patch_tokens.shape} → Spatial: {spatial.shape}")
    assert spatial.shape == (B, 14, 14, C)

    # 轉換回序列格式
    tokens_back = reshape_spatial_to_patch_tokens(spatial)
    print(f"Spatial: {spatial.shape} → Patch tokens: {tokens_back.shape}")
    assert tokens_back.shape == patch_tokens.shape

    # 驗證數值一致性
    assert torch.allclose(patch_tokens, tokens_back)
    print("✅ 轉換測試通過！數值一致。")
