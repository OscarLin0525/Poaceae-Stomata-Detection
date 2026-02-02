"""
SEGM v2: Spatially-Elastic Grid Modulation
==========================================

利用氣孔（Stomata）的水平週期性排列規律，在 DINO 中間層引導特徵提取，
通過週期性 Grid 過濾不在週期位置上的 noise。

核心思想：
---------
1. Stomata 在禾本科植物葉片上呈週期性排列
2. Noise（反光、葉脈、細胞壁）雖然可能長得像 stomata，但不遵循週期規律
3. FilterBank 生成週期性 Grid，只在「符合週期的位置」給予高響應
4. 不在週期位置的 noise 自然被壓低

架構：
-----
```
DINOv3 ViT
├── Block 0-9 (凍結)
├── Block 10
│   └── ★ SEGM Adapter 插入點 ★
│       ├── RowFrequencyEstimator (每行 FFT 估計頻率)
│       ├── PeriodicGridGenerator (生成週期性 Grid)
│       └── Modulator (調變特徵)
├── Block 11 (凍結)
└── Output
```

使用方式：
---------
```python
from segm_v2 import SEGMDinoVisionTransformer

model = SEGMDinoVisionTransformer(
    segm_after_blocks=[10],
    segm_config={'num_freq_bins': 32}
)
```
"""

from .models import (
    SEGMDinoVisionTransformer,
    SEGMAdapter,
    RowFrequencyEstimator,
    PeriodicGridGenerator,
)

from .losses import UnsupervisedSEGMLoss

__version__ = "2.0.0"
__all__ = [
    "SEGMDinoVisionTransformer",
    "SEGMAdapter",
    "RowFrequencyEstimator",
    "PeriodicGridGenerator",
    "UnsupervisedSEGMLoss",
]
