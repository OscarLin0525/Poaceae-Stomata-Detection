"""
Spatial Feature Alignment Head  (Student → Teacher)
=====================================================

Ported from  DINO_Teacher/dinoteacher/engine/align_head.py
with adjustments for the MTKD stomata-detection setting.

Core operation
--------------
1. Project the student backbone feature map to the teacher dimension
   using 1×1 Conv2d (various head types).
2. Bilinearly interpolate to the DINO teacher spatial resolution.
3. (Optional) L2 normalise along the channel axis.
4. Compute a per-pixel cosine or L2 alignment loss.

Key difference from the original
---------------------------------
* We keep *exactly* the same four head types (attention / MLP / MLP3 / linear)
  so that config parity with DINO Teacher is maintained.
* The class is framework-agnostic — it only needs (student_dim, teacher_dim)
  and does not depend on Detectron2.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
import warnings


class TeacherStudentAlignHead(nn.Module):
    """
    Per-pixel projection head that bridges the student backbone feature map
    to the frozen DINO teacher feature space.

    Args:
        student_dim:  Channel count of the student feature map.
        teacher_dim:  Channel count of the DINO teacher feature map.
        head_type:    ``"linear"`` | ``"MLP"`` | ``"MLP3"`` | ``"attention"``
        proj_dim:     Hidden dimension for MLP / MLP3 heads.
        normalize:    Whether to L2-normalise projected features.
        use_gelu:     If True use GELU, otherwise ReLU (mirrors ALIGN_PROJ_GELU).
    """

    def __init__(
        self,
        student_dim: int,
        teacher_dim: int,
        head_type: str = "MLP",
        proj_dim: int = 1024,
        normalize: bool = True,
        use_gelu: bool = False,
    ):
        super().__init__()
        self.proj_dim = proj_dim
        self.normalize_feature = normalize

        nl_layer = nn.GELU() if use_gelu else nn.ReLU()

        if head_type == "attention":
            self.projection_layer = _MHALayer(student_dim, teacher_dim)
        elif head_type == "MLP":
            self.projection_layer = nn.Sequential(
                nn.Conv2d(student_dim, proj_dim, 1, 1),
                nl_layer,
                nn.Conv2d(proj_dim, teacher_dim, 1, 1),
            )
        elif head_type == "MLP3":
            self.projection_layer = nn.Sequential(
                nn.Conv2d(student_dim, proj_dim, 1, 1),
                nl_layer,
                nn.Conv2d(proj_dim, proj_dim, 1, 1),
                nl_layer,
                nn.Conv2d(proj_dim, teacher_dim, 1, 1),
            )
        elif head_type == "linear":
            self.projection_layer = nn.Conv2d(student_dim, teacher_dim, 1, 1)
        else:
            raise NotImplementedError(f"Alignment head type '{head_type}' not supported.")

    # ------------------------------------------------------------------
    # forward / project
    # ------------------------------------------------------------------
    def forward(
        self,
        feat_student: torch.Tensor,
        teacher_feat_shape: Tuple[int, int],
    ) -> torch.Tensor:
        """
        Project student features and interpolate to teacher spatial resolution.

        Args:
            feat_student:      ``[B, C_s, H_s, W_s]``  student backbone feature map.
            teacher_feat_shape: ``(H_t, W_t)`` target spatial size from DINO teacher.

        Returns:
            projected:  ``[B, C_t, H_t, W_t]``
        """
        return self.project_student_feat(feat_student, teacher_feat_shape)

    def project_student_feat(
        self,
        feat_cnn: torch.Tensor,
        teacher_feat_shape: Tuple[int, int],
    ) -> torch.Tensor:
        h, w = teacher_feat_shape
        feat_cnn = self.projection_layer(feat_cnn)
        feat_cnn = F.interpolate(feat_cnn, (h, w), mode="bilinear", align_corners=False)
        if self.normalize_feature:
            feat_cnn = F.normalize(feat_cnn, p=2, dim=1)
        return feat_cnn

    # ------------------------------------------------------------------
    # alignment loss
    # ------------------------------------------------------------------
    def align_loss(
        self,
        feat_student: torch.Tensor,
        feat_teacher: torch.Tensor,
        return_sim: bool = False,
    ) -> torch.Tensor:
        """
        Per-pixel alignment loss.

        * If ``normalize_feature`` is True  → cosine distance ``(1 − s·t).mean()``
        * Otherwise → mean L2 distance / 100

        Args:
            feat_student: ``[B, C, H, W]``  (already projected + normalised)
            feat_teacher: ``[B, C, H, W]``
            return_sim:   Also return similarity map.

        Returns:
            loss – scalar.
            (loss, sim) if ``return_sim`` is True.
        """
        if feat_student.ndim != 4 or feat_teacher.ndim != 4:
            raise ValueError(
                "align_loss expects 4D tensors [B, C, H, W], got "
                f"student={tuple(feat_student.shape)} teacher={tuple(feat_teacher.shape)}"
            )

        # Defensive fallback: if a stale cache or a short last-batch causes
        # B mismatch, keep the overlapping samples instead of crashing.
        if feat_student.shape[0] != feat_teacher.shape[0]:
            min_b = min(int(feat_student.shape[0]), int(feat_teacher.shape[0]))
            warnings.warn(
                "align_loss batch mismatch "
                f"student={int(feat_student.shape[0])}, teacher={int(feat_teacher.shape[0])}; "
                f"using first {min_b} samples.",
                RuntimeWarning,
            )
            if min_b <= 0:
                zero = feat_student.new_tensor(0.0)
                empty_sim = feat_student.new_empty((0, 1, 1, 1, 1))
                return (zero, empty_sim) if return_sim else zero
            feat_student = feat_student[:min_b]
            feat_teacher = feat_teacher[:min_b]

        # Keep spatial resolution aligned if caller forgot interpolation.
        if feat_student.shape[2:] != feat_teacher.shape[2:]:
            feat_student = F.interpolate(
                feat_student,
                size=feat_teacher.shape[2:],
                mode="bilinear",
                align_corners=False,
            )

        if feat_student.shape[1] != feat_teacher.shape[1]:
            raise ValueError(
                "align_loss channel mismatch after projection: "
                f"student_C={int(feat_student.shape[1])}, teacher_C={int(feat_teacher.shape[1])}"
            )

        if self.normalize_feature:
            # per-pixel dot product
            feat_student = feat_student.permute(0, 2, 3, 1)  # [B, H, W, C]
            feat_teacher = feat_teacher.permute(0, 2, 3, 1)
            sim = torch.matmul(
                feat_student.unsqueeze(-2), feat_teacher.unsqueeze(-1)
            )  # [B, H, W, 1, 1]
            loss = (1 - sim).mean()
        else:
            sim = torch.linalg.norm(feat_student - feat_teacher, dim=1, ord=2)
            loss = sim.mean() / 100.0

        return (loss, sim) if return_sim else loss


# ------------------------------------------------------------------
# Internal helpers
# ------------------------------------------------------------------
class _MHALayer(nn.Module):
    """Multi-head self-attention + 1×1 projection (same as DINO Teacher)."""

    def __init__(self, cnn_dim: int, dino_dim: int):
        super().__init__()
        self.attn_layer = nn.MultiheadAttention(
            cnn_dim, num_heads=4, batch_first=True
        )
        self.projection = nn.Conv2d(cnn_dim, dino_dim, 1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        x = x.reshape(b, c, h * w).transpose(1, 2)       # [B, HW, C]
        x, _ = self.attn_layer(x, x, x, need_weights=False)
        x = self.projection(x.transpose(1, 2).reshape(b, c, h, w))
        return x
