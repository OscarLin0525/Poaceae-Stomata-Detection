"""
SEGM v2 Loss Functions
======================

自監督損失函數，不需要標註資料即可訓練 SEGM。
"""

from .unsupervised_loss import (
    UnsupervisedSEGMLoss,
    IntraGroupConsistencyLoss,
    InterGroupContrastLoss,
    PeriodicityLoss,
    SparsityLoss,
    FrequencySmoothLoss,
)

__all__ = [
    "UnsupervisedSEGMLoss",
    "IntraGroupConsistencyLoss",
    "InterGroupContrastLoss",
    "PeriodicityLoss",
    "SparsityLoss",
    "FrequencySmoothLoss",
]
