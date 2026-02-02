"""
SEGM-Enhanced DINOv3 Vision Transformer
======================================

這是主要的模型類別，繼承 DINOv3 的 Vision Transformer，
並在指定的 Block 後插入 SEGM Adapter。

設計原則：
---------
1. 不修改 DINOv3 原始碼（使用繼承）
2. 凍結 DINO 預訓練參數
3. 只訓練 SEGM 參數
4. 保持與原始 DINO 相同的輸入輸出介面

使用方式：
---------
```python
# 方法 1: 從頭建立（需要手動載入權重）
model = SEGMDinoVisionTransformer(
    img_size=224,
    patch_size=16,
    embed_dim=768,
    depth=12,
    num_heads=12,
    segm_after_blocks=[10],
)

# 方法 2: 使用 Hub 載入預訓練權重（推薦）
model = create_segm_dino_vitb16(pretrained=True, segm_after_blocks=[10])
```

架構圖：
-------
```
┌─────────────────────────────────────────────────────────────────────────┐
│                        SEGMDinoVisionTransformer                        │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  Input Image (B, 3, 224, 224)                                           │
│       │                                                                  │
│       ▼                                                                  │
│  ┌─────────────┐                                                        │
│  │ Patch Embed │ (B, 196, 768)                                          │
│  └─────────────┘                                                        │
│       │                                                                  │
│       ▼                                                                  │
│  ┌─────────────┐                                                        │
│  │  + CLS Token│ (B, 197, 768)                                          │
│  └─────────────┘                                                        │
│       │                                                                  │
│       ▼                                                                  │
│  ┌─────────────┐                                                        │
│  │  Block 0-9  │  ← 凍結                                                │
│  └─────────────┘                                                        │
│       │                                                                  │
│       ▼                                                                  │
│  ┌─────────────┐                                                        │
│  │  Block 10   │  ← 凍結                                                │
│  └─────────────┘                                                        │
│       │                                                                  │
│       ▼                                                                  │
│  ┌─────────────────────────────────────────────┐                        │
│  │            ★ SEGM Adapter ★                 │  ← 可訓練               │
│  │  ┌─────────────────────────────────────┐   │                        │
│  │  │ 1. RowFrequencyEstimator            │   │                        │
│  │  │ 2. PeriodicGridGenerator            │   │                        │
│  │  │ 3. Feature Modulation (zero-init)   │   │                        │
│  │  └─────────────────────────────────────┘   │                        │
│  └─────────────────────────────────────────────┘                        │
│       │                                                                  │
│       ▼                                                                  │
│  ┌─────────────┐                                                        │
│  │  Block 11   │  ← 凍結                                                │
│  └─────────────┘                                                        │
│       │                                                                  │
│       ▼                                                                  │
│  ┌─────────────┐                                                        │
│  │ Layer Norm  │                                                        │
│  └─────────────┘                                                        │
│       │                                                                  │
│       ▼                                                                  │
│  輸出: x_norm_clstoken, x_norm_patchtokens                              │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```
"""

import sys
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
from torch import Tensor

# 將 dinov3 加入 path
DINOV3_PATH = Path(__file__).parent.parent.parent / "dinov3-main"
if str(DINOV3_PATH) not in sys.path:
    sys.path.insert(0, str(DINOV3_PATH))

from .segm_adapter import SEGMAdapter


class SEGMDinoVisionTransformer(nn.Module):
    """
    帶有 SEGM 增強的 DINOv3 Vision Transformer

    這個類別：
    1. 包裝一個 DINOv3 模型
    2. 在指定的 Block 後插入 SEGM Adapter
    3. 凍結 DINO 參數，只訓練 SEGM 參數

    注意：
    -----
    這不是直接繼承 DinoVisionTransformer，而是使用組合（composition）。
    這樣做的好處是：
    1. 不需要深入了解 DINOv3 的內部實作
    2. 更容易升級到新版本的 DINOv3
    3. 代碼更清晰

    缺點是需要 hook 來插入 SEGM，但這是值得的 trade-off。
    """

    def __init__(
        self,
        dino_model: nn.Module,
        segm_after_blocks: List[int] = [10],
        segm_config: Optional[Dict] = None,
        freeze_dino: bool = True,
    ):
        """
        Args:
            dino_model: 預訓練的 DINOv3 模型
            segm_after_blocks: 在哪些 Block 後插入 SEGM（例如 [10] 表示 Block 10 後）
            segm_config: SEGM Adapter 的配置
            freeze_dino: 是否凍結 DINO 參數
        """
        super().__init__()

        self.dino = dino_model
        self.segm_after_blocks = segm_after_blocks
        self.freeze_dino = freeze_dino

        # 取得 DINO 的 embed_dim
        self.embed_dim = dino_model.embed_dim

        # 預設 SEGM 配置
        default_segm_config = {
            "embed_dim": self.embed_dim,
            "num_freq_bins": 32,
            "hidden_dim": 256,
            "init_kappa": 3.0,
            "init_gate": -5.0,
            "modulation_mode": "multiplicative",
        }
        if segm_config:
            default_segm_config.update(segm_config)

        # =========================================
        # 建立 SEGM Adapters
        # =========================================
        self.segm_adapters = nn.ModuleDict()
        for block_idx in segm_after_blocks:
            self.segm_adapters[str(block_idx)] = SEGMAdapter(**default_segm_config)

        # =========================================
        # 凍結 DINO 參數
        # =========================================
        if freeze_dino:
            self._freeze_dino_parameters()

        # =========================================
        # 註冊 Hook 來插入 SEGM
        # =========================================
        self._hooks = []
        self._register_hooks()

        # 儲存中間結果
        self._intermediates = {}

    def _freeze_dino_parameters(self):
        """凍結 DINO 的所有參數"""
        for param in self.dino.parameters():
            param.requires_grad = False

        print(f"已凍結 DINO 參數 (共 {sum(p.numel() for p in self.dino.parameters()):,} 個)")

    def _register_hooks(self):
        """
        註冊 forward hook 來在指定 Block 後插入 SEGM

        Hook 的工作方式：
        1. 在 Block forward 完成後被調用
        2. 接收 Block 的輸出
        3. 對輸出應用 SEGM
        4. 返回修改後的輸出
        """
        # 清除舊的 hooks
        for hook in self._hooks:
            hook.remove()
        self._hooks = []

        # 取得 DINO 的 blocks
        if hasattr(self.dino, 'blocks'):
            blocks = self.dino.blocks
        else:
            print("警告: 無法找到 DINO blocks，SEGM 將不會被插入")
            return

        # 為指定的 Block 註冊 hook
        for block_idx in self.segm_after_blocks:
            if block_idx >= len(blocks):
                print(f"警告: Block {block_idx} 不存在（共 {len(blocks)} 個 blocks）")
                continue

            # 建立 hook 函數
            adapter = self.segm_adapters[str(block_idx)]
            hook = self._create_hook(adapter, block_idx)

            # 註冊 hook
            handle = blocks[block_idx].register_forward_hook(hook)
            self._hooks.append(handle)

        print(f"已在 Block {self.segm_after_blocks} 後註冊 SEGM hooks")

    def _create_hook(self, adapter: SEGMAdapter, block_idx: int):
        """
        建立一個 forward hook

        DINOv3 Block 的輸出格式比較複雜，需要特別處理。
        根據 vision_transformer.py，Block forward 的輸入是 list of tensors。
        """
        def hook(module, input, output):
            """
            Hook 函數

            Args:
                module: 被 hook 的 Block
                input: Block 的輸入
                output: Block 的輸出

            Returns:
                modified_output: 經過 SEGM 處理的輸出
            """
            # DINOv3 Block 輸出是 list of tensors
            if isinstance(output, (list, tuple)):
                modified_outputs = []
                for i, x in enumerate(output):
                    modified_x = self._apply_segm_to_tensor(adapter, x, block_idx, i)
                    modified_outputs.append(modified_x)
                return type(output)(modified_outputs)
            else:
                # 單一 tensor
                return self._apply_segm_to_tensor(adapter, output, block_idx, 0)

        return hook

    def _apply_segm_to_tensor(
        self,
        adapter: SEGMAdapter,
        x: Tensor,
        block_idx: int,
        tensor_idx: int,
    ) -> Tensor:
        """
        對單一 tensor 應用 SEGM

        Args:
            adapter: SEGM Adapter
            x: (B, N, C) - Block 輸出的 tokens
                N = 1 (CLS) + n_storage + H*W (patches)
            block_idx: Block 索引
            tensor_idx: tensor 在 list 中的索引

        Returns:
            enhanced_x: (B, N, C) - 增強後的 tokens
        """
        B, N, C = x.shape

        # 取得 storage tokens 數量
        n_storage = getattr(self.dino, 'n_storage_tokens', 0)

        # 分離 prefix (CLS + storage) 和 patch tokens
        prefix_len = 1 + n_storage
        prefix_tokens = x[:, :prefix_len, :]      # (B, prefix_len, C)
        patch_tokens = x[:, prefix_len:, :]       # (B, H*W, C)

        # 計算空間維度
        N_patches = patch_tokens.shape[1]
        H = W = int(N_patches ** 0.5)

        if H * W != N_patches:
            # 非正方形，嘗試推斷
            print(f"警告: patch 數量 {N_patches} 不是完美正方形")
            # 假設最接近的正方形
            H = W = int(N_patches ** 0.5)

        # =========================================
        # 應用 SEGM Adapter
        # =========================================
        enhanced_patches, intermediates = adapter(
            patch_tokens, H, W, return_intermediates=True
        )

        # 儲存中間結果（用於 Loss 計算）
        key = f"block_{block_idx}_tensor_{tensor_idx}"
        self._intermediates[key] = intermediates

        # 重新組合
        enhanced_x = torch.cat([prefix_tokens, enhanced_patches], dim=1)

        return enhanced_x

    def forward(self, x: Tensor) -> Dict[str, Tensor]:
        """
        Forward pass

        Args:
            x: (B, 3, H, W) - 輸入圖像

        Returns:
            dict with:
                - x_norm_clstoken: (B, C) - CLS token
                - x_norm_patchtokens: (B, N, C) - Patch tokens
                (與原始 DINO 相同的輸出格式)
        """
        # 清除舊的中間結果
        self._intermediates = {}

        # 調用 DINO 的 forward
        # SEGM 會通過 hook 自動插入
        output = self.dino.forward_features(x)

        return output

    def forward_features(self, x: Tensor) -> Dict[str, Tensor]:
        """與 forward 相同，保持 API 一致性"""
        return self.forward(x)

    def get_intermediates(self) -> Dict:
        """取得最後一次 forward 的 SEGM 中間結果"""
        return self._intermediates

    def get_segm_parameters(self) -> List[nn.Parameter]:
        """取得所有 SEGM 參數（用於優化器）"""
        params = []
        for adapter in self.segm_adapters.values():
            params.extend(adapter.parameters())
        return params

    def get_trainable_parameters(self) -> List[nn.Parameter]:
        """取得所有可訓練參數"""
        return [p for p in self.parameters() if p.requires_grad]

    def print_trainable_parameters(self):
        """印出可訓練參數統計"""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        frozen_params = total_params - trainable_params

        print(f"參數統計:")
        print(f"  - 總參數: {total_params:,}")
        print(f"  - 可訓練 (SEGM): {trainable_params:,} ({100*trainable_params/total_params:.2f}%)")
        print(f"  - 凍結 (DINO): {frozen_params:,} ({100*frozen_params/total_params:.2f}%)")


# =============================================================================
# 工廠函數：快速建立模型
# =============================================================================

def create_segm_dino(
    model_name: str = "vitb16",
    pretrained: bool = True,
    segm_after_blocks: List[int] = [10],
    segm_config: Optional[Dict] = None,
    device: str = "cuda",
) -> SEGMDinoVisionTransformer:
    """
    建立帶有 SEGM 的 DINOv3 模型

    Args:
        model_name: DINO 模型名稱 ("vits16", "vitb16", "vitl16")
        pretrained: 是否載入預訓練權重
        segm_after_blocks: 在哪些 Block 後插入 SEGM
        segm_config: SEGM 配置
        device: 設備

    Returns:
        SEGMDinoVisionTransformer 模型
    """
    # 載入 DINOv3 模型
    try:
        from dinov3.hub.backbones import dinov3_vits16, dinov3_vitb16, dinov3_vitl16

        model_dict = {
            "vits16": dinov3_vits16,
            "vitb16": dinov3_vitb16,
            "vitl16": dinov3_vitl16,
        }

        if model_name not in model_dict:
            raise ValueError(f"Unknown model: {model_name}. Choose from {list(model_dict.keys())}")

        print(f"載入 DINOv3 {model_name}...")
        dino_model = model_dict[model_name](pretrained=pretrained)

    except ImportError as e:
        raise ImportError(
            f"無法載入 DINOv3。請確保 dinov3-main 在 Python path 中。\n"
            f"錯誤: {e}"
        )

    # 建立 SEGM 包裝模型
    model = SEGMDinoVisionTransformer(
        dino_model=dino_model,
        segm_after_blocks=segm_after_blocks,
        segm_config=segm_config,
        freeze_dino=True,
    )

    # 移到指定設備
    model = model.to(device)
    model.eval()

    # 印出參數統計
    model.print_trainable_parameters()

    return model


# 便捷函數
def create_segm_dino_vits16(**kwargs) -> SEGMDinoVisionTransformer:
    """建立 SEGM-DINOv3 ViT-S/16"""
    return create_segm_dino(model_name="vits16", **kwargs)


def create_segm_dino_vitb16(**kwargs) -> SEGMDinoVisionTransformer:
    """建立 SEGM-DINOv3 ViT-B/16"""
    return create_segm_dino(model_name="vitb16", **kwargs)


def create_segm_dino_vitl16(**kwargs) -> SEGMDinoVisionTransformer:
    """建立 SEGM-DINOv3 ViT-L/16"""
    return create_segm_dino(model_name="vitl16", **kwargs)


# =============================================================================
# 測試程式碼
# =============================================================================

if __name__ == "__main__":
    print("測試 SEGMDinoVisionTransformer...")
    print("=" * 60)

    # 檢查是否可以載入 DINOv3
    try:
        from dinov3.hub.backbones import dinov3_vitb16
        print("✓ DINOv3 可以載入")
    except ImportError as e:
        print(f"✗ 無法載入 DINOv3: {e}")
        print("請確保 dinov3-main 目錄存在且在 Python path 中")
        sys.exit(1)

    # 建立模型
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n使用設備: {device}")

    model = create_segm_dino_vitb16(
        pretrained=True,
        segm_after_blocks=[10],
        device=device,
    )

    # 測試 forward
    print("\n測試 Forward...")
    x = torch.randn(2, 3, 224, 224).to(device)

    with torch.no_grad():
        output = model(x)

    print(f"輸入: {x.shape}")
    print(f"輸出 cls_token: {output['x_norm_clstoken'].shape}")
    print(f"輸出 patch_tokens: {output['x_norm_patchtokens'].shape}")

    # 取得 SEGM 中間結果
    intermediates = model.get_intermediates()
    print(f"\nSEGM 中間結果 keys: {list(intermediates.keys())}")

    if intermediates:
        key = list(intermediates.keys())[0]
        print(f"  Grid shape: {intermediates[key]['grid'].shape}")
        print(f"  Gate value: {intermediates[key]['gate_value'].item():.4f}")

    print("\n✅ SEGMDinoVisionTransformer 測試通過！")
