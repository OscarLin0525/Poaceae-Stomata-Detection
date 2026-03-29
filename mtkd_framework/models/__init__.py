# MTKD Models

from .teacher_ensemble import TeacherEnsemble, WeightedBoxFusion
from .student_model import StudentDetector, StudentBackbone
from .yolo_wrappers import YOLOStudentDetector, YOLODetectionTeacher
from .mtkd_model_v2 import MTKDModelV2, build_mtkd_model_v2

__all__ = [
    "TeacherEnsemble",
    "WeightedBoxFusion",
    "StudentDetector",
    "StudentBackbone",
    "YOLOStudentDetector",
    "YOLODetectionTeacher",
    "MTKDModelV2",
    "build_mtkd_model_v2",
]
