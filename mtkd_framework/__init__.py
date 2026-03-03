# Multi-Teacher Knowledge Distillation Framework
# For Poaceae Stomata Detection

from .models import (
    MTKDModel,
    MTKDModelV2,
    StudentDetector,
    TeacherEnsemble,
    YOLOStudentDetector,
    YOLODetectionTeacher,
    build_mtkd_model,
    build_mtkd_model_v2,
)
from .losses import FeatureAlignmentLoss, PredictionAlignmentLoss, MTKDLoss
from .engine import (
    TeacherStudentAlignHead,
    DinoFeatureExtractor,
    PluggableFFTBlock,
    inject_fft_blocks,
    load_pseudo_labels_dir,
    load_pseudo_labels_csv,
    build_yolo_batch_from_pseudo,
    targets_to_yolo_batch,
)

__version__ = "2.1.0"
__all__ = [
    # v1
    "MTKDModel",
    "build_mtkd_model",
    # v2 (DINO-Teacher aligned)
    "MTKDModelV2",
    "build_mtkd_model_v2",
    # engine
    "TeacherStudentAlignHead",
    "DinoFeatureExtractor",
    "PluggableFFTBlock",
    "inject_fft_blocks",
    # pseudo-label utilities
    "load_pseudo_labels_dir",
    "load_pseudo_labels_csv",
    "build_yolo_batch_from_pseudo",
    "targets_to_yolo_batch",
    # shared
    "StudentDetector",
    "YOLOStudentDetector",
    "YOLODetectionTeacher",
    "TeacherEnsemble",
    "FeatureAlignmentLoss",
    "PredictionAlignmentLoss",
    "MTKDLoss",
]
