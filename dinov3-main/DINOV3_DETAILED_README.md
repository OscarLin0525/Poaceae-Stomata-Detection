# DINOv3 模型詳細文檔

## 目錄
1. [模型概述](#模型概述)
2. [核心架構](#核心架構)
3. [主要組件詳解](#主要組件詳解)
4. [函數參考](#函數參考)
5. [使用指南](#使用指南)
6. [組件拆解與自定義](#組件拆解與自定義)

---

## 模型概述

DINOv3 (Distillation with NO labels v3) 是 Meta AI 開發的自監督視覺 Transformer 模型。它使用 teacher-student 架構進行知識蒸餾訓練，無需標註數據即可學習高質量的視覺特徵表示。

### 主要特點
- **自監督學習**: 無需標註數據
- **Vision Transformer (ViT)**: 基於 Transformer 架構
- **RoPE 位置編碼**: 旋轉位置嵌入 (Rotary Position Embedding)
- **多頭自注意力機制**: Self-Attention 機制
- **Teacher-Student 蒸餾**: 知識蒸餾框架

---

## 核心架構

### 1. DinoVisionTransformer (主模型類)

**檔案位置**: `dinov3-main/dinov3/models/vision_transformer.py`

這是 DINOv3 的核心模型類，繼承自 `nn.Module`。

#### 初始化參數

```python
DinoVisionTransformer(
    img_size: int = 224,                    # 輸入圖像大小
    patch_size: int = 16,                   # patch 大小
    in_chans: int = 3,                      # 輸入通道數 (RGB=3)
    embed_dim: int = 768,                   # 嵌入維度
    depth: int = 12,                        # Transformer 層數
    num_heads: int = 12,                    # 注意力頭數
    ffn_ratio: float = 4.0,                 # FFN 隱藏層維度倍數
    qkv_bias: bool = True,                  # QKV 線性層是否使用 bias
    drop_path_rate: float = 0.0,            # DropPath 比率
    layerscale_init: float | None = None,   # LayerScale 初始值
    norm_layer: str = "layernorm",          # 正規化層類型
    ffn_layer: str = "mlp",                 # FFN 層類型
    n_storage_tokens: int = 0,              # 額外的可學習 token 數量
    # RoPE 相關參數
    pos_embed_rope_base: float = 100.0,
    pos_embed_rope_min_period: float | None = None,
    pos_embed_rope_max_period: float | None = None,
    ...
)
```

#### 核心屬性

- `patch_embed`: PatchEmbed 層，將圖像轉換為 patch embeddings
- `cls_token`: 可學習的分類 token (形狀: [1, 1, embed_dim])
- `storage_tokens`: 可選的額外 tokens (形狀: [1, n_storage_tokens, embed_dim])
- `rope_embed`: RoPE 位置編碼模組
- `blocks`: Transformer block 列表 (ModuleList)
- `norm`: 正規化層
- `mask_token`: 用於遮罩的可學習 token

---

## 主要組件詳解

### 2. PatchEmbed (圖像分塊嵌入)

**檔案位置**: `dinov3-main/dinov3/layers/patch_embed.py`

**功能**: 將 2D 圖像轉換為 patch embeddings

**輸入/輸出**:
- 輸入: `(B, C, H, W)` - Batch, Channels, Height, Width
- 輸出: `(B, N, D)` 或 `(B, H', W', D)` - N 為 patch 數量, D 為嵌入維度

#### 核心方法

```python
def forward(self, x: Tensor) -> Tensor:
    """
    參數:
        x: 輸入圖像 [B, C, H, W]

    返回:
        embeddings: [B, H', W', D] 如果 flatten_embedding=False
                    [B, N, D] 如果 flatten_embedding=True
    """
    x = self.proj(x)                      # 使用卷積層進行投影
    H, W = x.size(2), x.size(3)
    x = x.flatten(2).transpose(1, 2)      # 展平並轉置
    x = self.norm(x)                      # 正規化
    if not self.flatten_embedding:
        x = x.reshape(-1, H, W, self.embed_dim)
    return x
```

**關鍵實作**:
- 使用 `nn.Conv2d` 作為投影層，kernel_size 和 stride 都等於 patch_size
- 這相當於將圖像分割成不重疊的 patches 並進行線性投影

---

### 3. RopePositionEmbedding (旋轉位置編碼)

**檔案位置**: `dinov3-main/dinov3/layers/rope_position_encoding.py`

**功能**: 實現 RoPE (Rotary Position Embedding)，為 tokens 添加位置信息

#### 核心方法

```python
def forward(self, *, H: int, W: int) -> tuple[Tensor, Tensor]:
    """
    生成 sin 和 cos 位置編碼

    參數:
        H: 圖像高度方向的 patch 數量
        W: 圖像寬度方向的 patch 數量

    返回:
        (sin, cos): 兩個 Tensor，形狀都是 [HW, D]
    """
```

**RoPE 工作原理**:
1. 為每個位置 (h, w) 生成坐標
2. 將坐標標準化到 [-1, 1] 範圍
3. 根據頻率參數計算角度
4. 返回 sin 和 cos 值用於旋轉嵌入

**RoPE 應用函數** (在 `attention.py` 中):

```python
def rope_apply(x: Tensor, sin: Tensor, cos: Tensor) -> Tensor:
    """
    將 RoPE 應用到輸入張量

    公式: RoPE(x) = x * cos + rotate_half(x) * sin
    """
    return (x * cos) + (rope_rotate_half(x) * sin)

def rope_rotate_half(x: Tensor) -> Tensor:
    """
    將向量的前半部分和後半部分交換並取負

    示例: [x0, x1, x2, x3, x4, x5] -> [-x3, -x4, -x5, x0, x1, x2]
    """
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat([-x2, x1], dim=-1)
```

---

### 4. SelfAttention (自注意力機制)

**檔案位置**: `dinov3-main/dinov3/layers/attention.py`

**功能**: 實現多頭自注意力機制，支持 RoPE

#### 核心方法

```python
def forward(self, x: Tensor, attn_bias=None, rope: Tensor = None) -> Tensor:
    """
    前向傳播

    參數:
        x: 輸入 [B, N, D]
        attn_bias: 注意力偏置 (可選)
        rope: RoPE 編碼 (sin, cos) tuple

    返回:
        輸出 [B, N, D]
    """
    qkv = self.qkv(x)                              # [B, N, 3*D]
    attn_v = self.compute_attention(qkv=qkv, attn_bias=attn_bias, rope=rope)
    x = self.proj(attn_v)                          # 投影回原維度
    x = self.proj_drop(x)
    return x
```

```python
def compute_attention(self, qkv: Tensor, attn_bias=None, rope=None) -> Tensor:
    """
    計算注意力

    步驟:
    1. 將 qkv 重塑為 [B, N, 3, num_heads, head_dim]
    2. 分離出 Q, K, V
    3. 如果提供 rope，應用 RoPE 到 Q 和 K
    4. 使用 scaled_dot_product_attention 計算注意力
    """
    B, N, _ = qkv.shape
    C = self.qkv.in_features

    # 重塑並分離 Q, K, V
    qkv = qkv.reshape(B, N, 3, self.num_heads, C // self.num_heads)
    q, k, v = torch.unbind(qkv, 2)
    q, k, v = [t.transpose(1, 2) for t in [q, k, v]]  # [B, heads, N, head_dim]

    # 應用 RoPE
    if rope is not None:
        q, k = self.apply_rope(q, k, rope)

    # 計算注意力
    x = torch.nn.functional.scaled_dot_product_attention(q, k, v)
    x = x.transpose(1, 2)
    return x.reshape([B, N, C])
```

**注意力公式**:
```
Attention(Q, K, V) = softmax(QK^T / √d_k) V
```

其中 `scaled_dot_product_attention` 實現了高效的 Flash Attention。

---

### 5. SelfAttentionBlock (注意力塊)

**檔案位置**: `dinov3-main/dinov3/layers/block.py`

**功能**: 完整的 Transformer block，包含注意力層和 FFN 層

#### 結構

```
輸入 x
  ↓
LayerNorm (norm1)
  ↓
SelfAttention
  ↓
LayerScale (ls1) [可選]
  ↓
殘差連接 (+x)
  ↓
LayerNorm (norm2)
  ↓
MLP/FFN
  ↓
LayerScale (ls2) [可選]
  ↓
殘差連接 (+)
  ↓
輸出
```

#### 核心方法

```python
def forward(self, x_or_x_list, rope_or_rope_list=None) -> List[Tensor]:
    """
    前向傳播，支持單個 tensor 或 tensor 列表

    參數:
        x_or_x_list: 單個 Tensor [B, N, D] 或列表
        rope_or_rope_list: RoPE 編碼或列表

    返回:
        處理後的 tensor 或列表
    """
```

**實作細節**:
- 支持 DropPath (Stochastic Depth) 用於正則化
- 使用 LayerScale 穩定訓練
- 殘差連接確保梯度流動

---

### 6. DINOHead (DINO 投影頭)

**檔案位置**: `dinov3-main/dinov3/layers/dino_head.py`

**功能**: 將 backbone 特徵投影到原型空間用於對比學習

#### 結構

```python
class DINOHead(nn.Module):
    def __init__(
        self,
        in_dim,                    # 輸入維度
        out_dim,                   # 輸出原型數量 (K)
        use_bn=False,              # 是否使用 BatchNorm
        nlayers=3,                 # MLP 層數
        hidden_dim=2048,           # 隱藏層維度
        bottleneck_dim=256,        # 瓶頸維度
        mlp_bias=True,             # 是否使用 bias
    ):
```

#### 核心方法

```python
def forward(self, x, no_last_layer=False, only_last_layer=False):
    """
    前向傳播

    參數:
        x: 輸入特徵 [B, D]
        no_last_layer: 如果為 True，不應用最後的投影層
        only_last_layer: 如果為 True，只應用最後的投影層

    返回:
        投影後的特徵
    """
    if not only_last_layer:
        x = self.mlp(x)                        # MLP 投影
        eps = 1e-6 if x.dtype == torch.float16 else 1e-12
        x = nn.functional.normalize(x, dim=-1, p=2, eps=eps)  # L2 正規化
    if not no_last_layer:
        x = self.last_layer(x)                 # 投影到原型空間
    return x
```

**MLP 結構**:
```
Linear(in_dim → hidden_dim)
  ↓
[BatchNorm1d] (可選)
  ↓
GELU
  ↓
... (重複 nlayers-2 次)
  ↓
Linear(hidden_dim → bottleneck_dim)
  ↓
L2 Normalize
  ↓
Linear(bottleneck_dim → out_dim) [last_layer]
```

---

## 函數參考

### DinoVisionTransformer 核心方法

#### 1. `prepare_tokens_with_masks`

```python
def prepare_tokens_with_masks(self, x: Tensor, masks=None) -> Tuple[Tensor, Tuple[int]]:
    """
    準備輸入 tokens，包括 patch embedding 和特殊 tokens

    參數:
        x: 輸入圖像 [B, C, H, W]
        masks: 遮罩 [B, num_patches] (可選)

    返回:
        tokens: [B, 1+n_storage+num_patches, D]
        hw_tuple: (H', W') patch 網格大小

    流程:
    1. 通過 patch_embed 獲得 patch embeddings [B, H', W', D]
    2. 如果有 masks，將被遮罩的 patches 替換為 mask_token
    3. 展平 patches: [B, H'*W', D]
    4. 添加 cls_token 和 storage_tokens
    5. 拼接: [cls, storage_tokens, patch_tokens]
    """
    x = self.patch_embed(x)              # [B, H', W', D]
    B, H, W, _ = x.shape
    x = x.flatten(1, 2)                  # [B, H'*W', D]

    # 應用遮罩
    if masks is not None:
        x = torch.where(masks.unsqueeze(-1), self.mask_token.to(x.dtype).unsqueeze(0), x)
        cls_token = self.cls_token
    else:
        cls_token = self.cls_token + 0 * self.mask_token  # 用於初始化

    # 準備 storage tokens
    if self.n_storage_tokens > 0:
        storage_tokens = self.storage_tokens
    else:
        storage_tokens = torch.empty(1, 0, cls_token.shape[-1],
                                     dtype=cls_token.dtype,
                                     device=cls_token.device)

    # 拼接所有 tokens
    x = torch.cat([
        cls_token.expand(B, -1, -1),         # [B, 1, D]
        storage_tokens.expand(B, -1, -1),    # [B, n_storage, D]
        x,                                    # [B, num_patches, D]
    ], dim=1)

    return x, (H, W)
```

#### 2. `forward_features`

```python
def forward_features(self, x: Tensor | List[Tensor], masks: Optional[Tensor] = None) -> List[Dict[str, Tensor]]:
    """
    通過 Transformer blocks 提取特徵

    參數:
        x: 輸入圖像或圖像列表
        masks: 遮罩或遮罩列表

    返回:
        特徵字典列表，每個字典包含:
        - x_norm_clstoken: 正規化的 cls token [B, D]
        - x_storage_tokens: 正規化的 storage tokens [B, n_storage, D]
        - x_norm_patchtokens: 正規化的 patch tokens [B, num_patches, D]
        - x_prenorm: 正規化前的所有 tokens [B, 1+n_storage+num_patches, D]
        - masks: 遮罩
    """
```

**實作步驟**:
1. 調用 `forward_features_list` 處理輸入
2. 對於每個輸入:
   - 準備 tokens (包括遮罩處理)
   - 通過所有 Transformer blocks
   - 生成 RoPE 編碼
   - 應用正規化
   - 分離 cls token, storage tokens, patch tokens
3. 返回結構化的特徵字典

#### 3. `get_intermediate_layers`

```python
def get_intermediate_layers(
    self,
    x: torch.Tensor,
    *,
    n: Union[int, Sequence] = 1,           # 要提取的層數或層索引
    reshape: bool = False,                 # 是否重塑為 2D 網格
    return_class_token: bool = False,      # 是否返回 cls token
    return_extra_tokens: bool = False,     # 是否返回 storage tokens
    norm: bool = True,                     # 是否應用正規化
) -> Tuple[Union[torch.Tensor, Tuple[torch.Tensor, ...]]]:
    """
    提取中間層的特徵

    用途:
    - 特徵金字塔
    - 多尺度特徵
    - 可視化中間表示

    返回:
        根據參數組合返回不同格式的 tuple
    """
```

#### 4. `forward`

```python
def forward(self, *args, is_training: bool = False, **kwargs) -> List[Dict[str, Tensor]] | Tensor:
    """
    模型前向傳播

    參數:
        is_training: 訓練模式返回完整特徵，推理模式返回 cls token

    返回:
        訓練模式: 特徵字典列表
        推理模式: cls token 特徵 [B, D]
    """
    ret = self.forward_features(*args, **kwargs)
    if is_training:
        return ret
    else:
        return self.head(ret["x_norm_clstoken"])
```

---

### 預定義模型配置

**檔案位置**: `dinov3-main/dinov3/models/vision_transformer.py` (底部)

#### 可用模型尺寸

```python
# ViT-Small
def vit_small(patch_size=16, **kwargs):
    return DinoVisionTransformer(
        patch_size=patch_size,
        embed_dim=384,      # 嵌入維度
        depth=12,           # 12 層
        num_heads=6,        # 6 個注意力頭
        ffn_ratio=4,
        **kwargs,
    )

# ViT-Base
def vit_base(patch_size=16, **kwargs):
    return DinoVisionTransformer(
        patch_size=patch_size,
        embed_dim=768,
        depth=12,
        num_heads=12,
        ffn_ratio=4,
        **kwargs,
    )

# ViT-Large
def vit_large(patch_size=16, **kwargs):
    return DinoVisionTransformer(
        patch_size=patch_size,
        embed_dim=1024,
        depth=24,           # 24 層
        num_heads=16,
        ffn_ratio=4,
        **kwargs,
    )

# ViT-Giant2
def vit_giant2(patch_size=16, **kwargs):
    return DinoVisionTransformer(
        patch_size=patch_size,
        embed_dim=1536,
        depth=40,
        num_heads=24,
        ffn_ratio=4,
        **kwargs,
    )
```

**模型尺寸比較**:

| 模型 | 嵌入維度 | 層數 | 注意力頭 | 參數量 (約) |
|------|---------|------|---------|------------|
| Small | 384 | 12 | 6 | ~22M |
| Base | 768 | 12 | 12 | ~86M |
| Large | 1024 | 24 | 16 | ~304M |
| Giant2 | 1536 | 40 | 24 | ~1.1B |

---

## 使用指南

### 基本使用

#### 1. 載入預訓練模型

```python
from dinov3.models import build_model_for_eval
from dinov3.configs import get_default_config

# 載入配置
config = get_default_config()
config.student.arch = "vit_base"  # 選擇模型尺寸
config.student.patch_size = 16

# 載入模型
model = build_model_for_eval(
    config=config,
    pretrained_weights="/path/to/checkpoint",  # 預訓練權重路徑
    shard_unsharded_model=False
)

model.eval()
```

#### 2. 特徵提取

```python
import torch
from PIL import Image
from torchvision import transforms

# 圖像預處理
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225]),
])

# 載入圖像
img = Image.open("image.jpg").convert("RGB")
img_tensor = transform(img).unsqueeze(0)  # [1, 3, 224, 224]

# 提取特徵
with torch.no_grad():
    # 獲取 CLS token 特徵 (全局特徵)
    features = model(img_tensor)  # [1, embed_dim]

    # 或獲取完整特徵
    features_dict = model(img_tensor, is_training=True)
    cls_token = features_dict["x_norm_clstoken"]          # [1, D]
    patch_tokens = features_dict["x_norm_patchtokens"]    # [1, num_patches, D]
```

#### 3. 提取多尺度特徵

```python
# 提取最後 4 層的特徵
with torch.no_grad():
    intermediate_features = model.get_intermediate_layers(
        img_tensor,
        n=4,                          # 最後 4 層
        reshape=True,                 # 重塑為 2D 網格
        return_class_token=True,      # 返回 cls token
        norm=True                     # 應用正規化
    )

# intermediate_features 是一個 tuple
# 每個元素是 (patch_features, cls_token)
# patch_features: [B, D, H', W']
# cls_token: [B, D]
```

---

### 下游任務使用

#### 圖像分類

```python
import torch.nn as nn

class DINOClassifier(nn.Module):
    def __init__(self, dinov3_backbone, num_classes=1000):
        super().__init__()
        self.backbone = dinov3_backbone
        self.backbone.eval()  # 凍結 backbone
        for param in self.backbone.parameters():
            param.requires_grad = False

        # 分類頭
        self.classifier = nn.Linear(self.backbone.embed_dim, num_classes)

    def forward(self, x):
        with torch.no_grad():
            features = self.backbone(x)  # [B, D]
        logits = self.classifier(features)
        return logits

# 使用
classifier = DINOClassifier(model, num_classes=100)
```

#### 目標檢測/語義分割

```python
class DINOFeatureExtractor(nn.Module):
    def __init__(self, dinov3_backbone):
        super().__init__()
        self.backbone = dinov3_backbone
        self.backbone.eval()

    def forward(self, x):
        # 提取多層特徵用於 FPN
        features = self.backbone.get_intermediate_layers(
            x,
            n=[3, 5, 7, 11],  # 選擇特定層
            reshape=True,      # [B, D, H, W] 格式
        )
        return features

# 可以接到 DETR, Mask2Former 等檢測/分割頭
```

---

## 組件拆解與自定義

### 自定義 Attention 機制

如果你想修改注意力機制，可以繼承 `SelfAttention` 類:

```python
from dinov3.layers.attention import SelfAttention

class CustomAttention(SelfAttention):
    def __init__(self, *args, custom_param=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.custom_param = custom_param

    def compute_attention(self, qkv: Tensor, attn_bias=None, rope=None) -> Tensor:
        # 你的自定義邏輯
        B, N, _ = qkv.shape
        C = self.qkv.in_features

        qkv = qkv.reshape(B, N, 3, self.num_heads, C // self.num_heads)
        q, k, v = torch.unbind(qkv, 2)

        # 添加自定義處理
        # ...

        return custom_output
```

### 替換 FFN 層

DINOv3 支持不同的 FFN 實作:

```python
from dinov3.layers.ffn_layers import Mlp, SwiGLUFFN

# 在創建模型時指定
model = DinoVisionTransformer(
    ...,
    ffn_layer="swiglu",  # 選項: "mlp", "swiglu", "swiglu32", "swiglu64", "swiglu128"
    ...
)
```

**自定義 FFN**:

```python
class CustomFFN(nn.Module):
    def __init__(self, in_features, hidden_features, act_layer=nn.GELU, drop=0., bias=True, device=None):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_features, bias=bias, device=device)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, in_features, bias=bias, device=device)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

# 註冊到 ffn_layer_dict
from dinov3.models.vision_transformer import ffn_layer_dict
ffn_layer_dict["custom"] = CustomFFN

# 使用
model = DinoVisionTransformer(..., ffn_layer="custom", ...)
```

### 修改位置編碼

#### 使用絕對位置編碼替代 RoPE

```python
class AbsolutePositionEmbedding(nn.Module):
    def __init__(self, num_patches, embed_dim):
        super().__init__()
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def forward(self, x):
        return x + self.pos_embed

# 修改 DinoVisionTransformer 的 __init__
# 替換 self.rope_embed 為 AbsolutePositionEmbedding
```

### 添加額外的 Token 類型

```python
# 在 DinoVisionTransformer.__init__ 中添加
self.custom_tokens = nn.Parameter(torch.zeros(1, num_custom_tokens, embed_dim))
nn.init.normal_(self.custom_tokens, std=0.02)

# 在 prepare_tokens_with_masks 中拼接
x = torch.cat([
    cls_token.expand(B, -1, -1),
    storage_tokens.expand(B, -1, -1),
    self.custom_tokens.expand(B, -1, -1),  # 添加自定義 tokens
    x,  # patch tokens
], dim=1)
```

### 自定義損失函數

DINOv3 使用多個損失函數組合:

```python
from dinov3.loss import DINOLoss, KoLeoLoss, iBOTPatchLoss

# DINO 損失: 對比學習損失
dino_loss = DINOLoss(out_dim=65536)  # out_dim 是原型數量

# KoLeo 正則化: 鼓勵特徵分散
koleo_loss = KoLeoLoss()

# iBOT 損失: 遮罩圖像建模
ibot_loss = iBOTPatchLoss(out_dim=65536)

# 組合使用
total_loss = (
    1.0 * dino_loss(student_logits, teacher_probs) +
    0.1 * koleo_loss(student_features) +
    1.0 * ibot_loss(student_patches, teacher_patches)
)
```

---

## 訓練流程

### SSLMetaArch (自監督訓練架構)

**檔案位置**: `dinov3-main/dinov3/train/ssl_meta_arch.py`

這是 DINOv3 的完整訓練框架。

#### 核心組件

```python
class SSLMetaArch(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        # 構建 student 和 teacher 模型
        student_backbone, teacher_backbone, embed_dim = build_model_from_cfg(cfg)

        # DINO 頭
        self.student.dino_head = DINOHead(...)
        self.teacher.dino_head = DINOHead(...)

        # iBOT 頭
        self.student.ibot_head = DINOHead(...)
        self.teacher.ibot_head = DINOHead(...)

        # 損失函數
        self.dino_loss = DINOLoss(...)
        self.koleo_loss = KoLeoLoss(...)
        self.ibot_patch_loss = iBOTPatchLoss(...)
```

#### 訓練步驟

```python
def forward_backward(self, data, *, teacher_temp, iteration=0):
    """
    完整的前向和反向傳播

    步驟:
    1. 準備數據 (global crops, local crops, masks)
    2. Teacher 前向傳播 (無梯度)
    3. Student 前向傳播
    4. 計算損失
    5. 反向傳播
    6. 返回損失和指標
    """

    # 1. 獲取數據
    global_crops = data["collated_global_crops"].cuda()
    local_crops = data["collated_local_crops"].cuda()
    masks = data["collated_masks"].cuda()

    # 2. Teacher 輸出
    teacher_output = self.get_teacher_output(global_crops, ...)

    # 3. Student 輸出
    student_global, student_local = self.get_student_output(
        global_crops=global_crops,
        local_crops=local_crops,
        masks=masks
    )

    # 4. 計算損失
    total_loss, loss_dict = self.compute_losses(
        teacher_global=teacher_output,
        student_global=student_global,
        student_local=student_local,
        ...
    )

    # 5. 反向傳播
    total_loss.backward()

    return total_loss, loss_dict
```

#### EMA 更新

```python
def update_ema(self, m):
    """
    指數移動平均更新 teacher 參數

    teacher = m * teacher + (1-m) * student

    典型的 m 值: 0.996 - 0.999
    """
    with torch.no_grad():
        for param_s, param_t in zip(self.student.parameters(),
                                     self.teacher.parameters()):
            param_t.data.mul_(m).add_(param_s.data, alpha=1-m)
```

---

## 數據增強

DINOv3 使用特殊的數據增強策略:

```python
from dinov3.data import DataAugmentationDINO

data_aug = DataAugmentationDINO(
    global_crops_scale=(0.4, 1.0),      # global crops 的縮放範圍
    local_crops_scale=(0.05, 0.4),      # local crops 的縮放範圍
    local_crops_number=8,                # local crops 數量
    global_crops_size=224,               # global crops 大小
    local_crops_size=96,                 # local crops 大小
)

# 對一張圖像生成多個視圖
augmented_data = data_aug(image)
# 返回:
# - 2 個 global crops (大視圖)
# - 8 個 local crops (小視圖)
# - 對應的遮罩
```

---

## 進階技巧

### 1. 使用混合精度訓練

```python
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()
model = model.cuda()

for data in dataloader:
    optimizer.zero_grad()

    with autocast():
        loss, metrics = model.forward_backward(data, teacher_temp=0.04)

    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
```

### 2. 梯度累積

```python
accumulation_steps = 4

for i, data in enumerate(dataloader):
    loss, metrics = model.forward_backward(data, teacher_temp=0.04)
    loss = loss / accumulation_steps
    loss.backward()

    if (i + 1) % accumulation_steps == 0:
        optimizer.step()
        optimizer.zero_grad()
```

### 3. 層級學習率衰減

```python
from dinov3.train.param_groups import get_params_groups_with_decay_fsdp

param_groups = get_params_groups_with_decay_fsdp(
    model=model,
    lr_decay_rate=0.65,              # 每層衰減 65%
    patch_embed_lr_mult=0.2,         # patch embedding 學習率倍數
)

optimizer = torch.optim.AdamW(param_groups, lr=1e-4)
```

### 4. 可視化注意力圖

```python
def visualize_attention(model, image, layer_idx=-1):
    """
    可視化指定層的注意力權重
    """
    # 提取注意力權重需要修改 forward 以返回 attention weights
    # 這裡提供一個簡化版本

    with torch.no_grad():
        # Hook 到特定層
        attention_weights = []

        def hook_fn(module, input, output):
            # 保存注意力權重
            attention_weights.append(output)

        handle = model.blocks[layer_idx].attn.register_forward_hook(hook_fn)

        _ = model(image)

        handle.remove()

    # 處理並可視化 attention_weights
    return attention_weights
```

---

## 常見問題與解決方案

### Q1: 如何調整輸入圖像大小?

DINOv3 可以處理不同尺寸的圖像，但需要確保尺寸是 patch_size 的倍數:

```python
# 如果 patch_size=16，圖像尺寸應該是 16 的倍數
# 例如: 224, 256, 384, 512 等

model = DinoVisionTransformer(
    img_size=384,        # 訓練時的圖像尺寸
    patch_size=16,
    ...
)

# 推理時可以使用不同尺寸
img = torch.randn(1, 3, 512, 512)  # 512 也是 16 的倍數
features = model(img)
```

### Q2: 如何凍結部分層?

```python
# 凍結 backbone，只訓練分類頭
for param in model.backbone.parameters():
    param.requires_grad = False

# 或凍結前 N 層
for i in range(8):  # 凍結前 8 層
    for param in model.blocks[i].parameters():
        param.requires_grad = False
```

### Q3: 記憶體不足怎麼辦?

```python
# 1. 使用梯度檢查點
import torch.utils.checkpoint as checkpoint

class DinoViTWithCheckpoint(DinoVisionTransformer):
    def forward(self, x):
        # 使用 checkpoint 減少記憶體
        for block in self.blocks:
            x = checkpoint.checkpoint(block, x)
        return x

# 2. 減少批次大小
batch_size = 16  # 改為 8 或 4

# 3. 使用較小的模型
model = vit_small()  # 代替 vit_large
```

### Q4: 如何保存和載入模型?

```python
# 保存
torch.save({
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'epoch': epoch,
}, 'checkpoint.pth')

# 載入
checkpoint = torch.load('checkpoint.pth')
model.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
epoch = checkpoint['epoch']
```

---

## 參考資料

### 論文
- DINOv2: Learning Robust Visual Features without Supervision
- Vision Transformer (ViT): An Image is Worth 16x16 Words
- RoFormer: Enhanced Transformer with Rotary Position Embedding

### 相關連結
- [DINOv3 GitHub](https://github.com/facebookresearch/dinov3)
- [Meta AI Blog](https://ai.facebook.com/)

---

## 附錄: 完整代碼範例

### 端到端分類訓練範例

```python
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from dinov3.models import vit_base

# 1. 載入預訓練 DINOv3
backbone = vit_base(patch_size=16)
checkpoint = torch.load('dinov3_vitb16_pretrain.pth')
backbone.load_state_dict(checkpoint['model'], strict=False)

# 2. 構建分類器
class ImageClassifier(nn.Module):
    def __init__(self, backbone, num_classes):
        super().__init__()
        self.backbone = backbone
        # 凍結 backbone
        for param in self.backbone.parameters():
            param.requires_grad = False

        self.classifier = nn.Linear(backbone.embed_dim, num_classes)

    def forward(self, x):
        with torch.no_grad():
            features = self.backbone(x)  # CLS token features
        logits = self.classifier(features)
        return logits

model = ImageClassifier(backbone, num_classes=10).cuda()

# 3. 準備數據
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

train_dataset = datasets.CIFAR10(root='./data', train=True,
                                  download=True, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

# 4. 訓練
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.classifier.parameters(), lr=1e-3)

for epoch in range(10):
    model.train()
    for images, labels in train_loader:
        images, labels = images.cuda(), labels.cuda()

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

    print(f'Epoch {epoch+1}, Loss: {loss.item():.4f}')
```

### 特徵提取與相似度檢索

```python
import torch
import torch.nn.functional as F
from PIL import Image

# 載入模型
model = vit_base(patch_size=16)
model.load_state_dict(torch.load('dinov3_vitb16_pretrain.pth')['model'])
model.eval().cuda()

def extract_features(image_path):
    """提取圖像特徵"""
    img = Image.open(image_path).convert('RGB')
    img_tensor = transform(img).unsqueeze(0).cuda()

    with torch.no_grad():
        features = model(img_tensor)  # [1, 768]

    # L2 正規化
    features = F.normalize(features, p=2, dim=-1)
    return features

# 提取多張圖像的特徵
image_paths = ['img1.jpg', 'img2.jpg', 'img3.jpg']
features_list = [extract_features(path) for path in image_paths]
features_db = torch.cat(features_list, dim=0)  # [N, 768]

# 檢索相似圖像
query_features = extract_features('query.jpg')
similarities = torch.matmul(query_features, features_db.T)  # [1, N]
top_k_indices = similarities.topk(k=5).indices

print(f"Most similar images: {[image_paths[i] for i in top_k_indices[0]]}")
```

---

這份文檔涵蓋了 DINOv3 模型的核心架構、詳細的函數說明、使用方法以及如何拆解和自定義各個組件。你可以根據需要選擇相應的部分進行修改和擴展。
