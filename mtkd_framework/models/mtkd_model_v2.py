"""
MTKDModel v2 — refactored following the DINO Teacher engine pattern
=====================================================================

Changes from ``mtkd_model.py`` (v1):
-------------------------------------
1. **DINOFeatureTeacher** → ``engine.DinoFeatureExtractor`` (spatial output,
   proper preprocessing, frozen).
2. **FeatureAdapter** (``nn.Linear`` + global pool) → ``engine.TeacherStudentAlignHead``
   (``nn.Conv2d(1×1)`` per-pixel + ``F.interpolate``).  This fixes **C1/M1**.
3. ``PluggableFFTBlock`` optionally injected into the frozen DINO encoder.
4. Pseudo-label pathway mirrors DINO Teacher: filtered pseudo-labels are treated
   as GT supervision (standard detection loss), **not** element-wise KL.
   The user's pretrained wheat model supplies these PLs externally (see
   ``MTKDTrainerV2.load_pseudo_labels``).
5. ``forward()`` returns spatial features so the trainer can compute the
   alignment loss *outside* the model — exactly as in DINO Teacher.

Backwards compat
-----------------
* ``MTKDModel`` in ``mtkd_model.py`` is untouched.
* This module is a **standalone** alternative that can be swapped in via config.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..engine.build_dino import DinoFeatureExtractor
from ..engine.align_head import TeacherStudentAlignHead
from ..engine.pluggable_fft_block import inject_fft_blocks, PluggableFFTBlock

logger = logging.getLogger(__name__)


# ======================================================================
# Build helpers
# ======================================================================

def _build_student(student_cfg: Dict[str, Any], num_classes: int) -> nn.Module:
    """Instantiate student from config (YOLO or generic)."""
    stype = student_cfg.pop("student_type", "yolo").lower()
    if stype == "yolo":
        from .yolo_wrappers import YOLOStudentDetector

        student_cfg.setdefault("num_classes", num_classes)
        # Remove dino_teacher_dim — not needed for v2 (align head is external)
        student_cfg.pop("dino_teacher_dim", None)
        return YOLOStudentDetector(**student_cfg)
    else:
        from .student_model import StudentDetector

        student_cfg.setdefault("head_config", {"num_classes": num_classes})
        return StudentDetector(**student_cfg)


# ======================================================================
# MTKDModelV2
# ======================================================================

class MTKDModelV2(nn.Module):
    """
    Multi-Teacher Knowledge Distillation Model — v2 (DINO Teacher aligned).

    Components
    ----------
    1. **Student detector** — YOLO or DETR-like (trainable).
    2. **Frozen DINO encoder** — ``DinoFeatureExtractor`` returning a spatial
       feature map ``[B, D, H_p, W_p]``.
    3. **Alignment head** — ``TeacherStudentAlignHead`` projecting the
       student backbone feature map to the DINO teacher space (per-pixel).
    4. *(Optional)* **FFT blocks** injected into the DINO encoder.
    5. *(Optional)* **Ensemble detection teachers** for WBF pseudo-labels.

    The trainer is responsible for orchestrating the loss computation
    (feature alignment, pseudo-label supervision, etc.) — exactly as in
    DINO Teacher.  ``forward()`` here only runs the student forward pass
    and (optionally) the DINO teacher feature extraction.
    """

    def __init__(
        self,
        # Student
        student_config: Optional[Dict[str, Any]] = None,
        custom_student: Optional[nn.Module] = None,
        num_classes: int = 1,
        # DINO teacher
        dino_config: Optional[Dict[str, Any]] = None,
        dino_checkpoint: Optional[str] = None,
        # Alignment head
        align_head_config: Optional[Dict[str, Any]] = None,
        student_align_layer: str = "p4",  # which pyramid level to align
        # FFT block injection
        fft_block_config: Optional[Dict[str, Any]] = None,
        # Ensemble teachers (kept for backwards compat)
        ensemble_config: Optional[Dict[str, Any]] = None,
        teacher_specs: Optional[List[Dict[str, Any]]] = None,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.student_align_layer = student_align_layer

        # =============================================================
        # 1. Student
        # =============================================================
        if custom_student is not None:
            self.student = custom_student
        else:
            student_config = dict(student_config or {})
            self.student = _build_student(student_config, num_classes)

        # =============================================================
        # 2. Frozen DINO Feature Extractor
        # =============================================================
        dino_config = dict(dino_config or {})
        dino_config.setdefault("model_name", "vit_base")
        dino_config.setdefault("patch_size", 16)
        dino_config.setdefault("embed_dim", 768)
        dino_config.setdefault("normalize_feature", True)
        if dino_checkpoint:
            dino_config["pretrained_path"] = dino_checkpoint
        self.dino_teacher = DinoFeatureExtractor(**dino_config)
        self._dino_embed_dim = dino_config["embed_dim"]

        # =============================================================
        # 3. FFT block injection (before alignment head so encoder is ready)
        # =============================================================
        self.fft_blocks: List[PluggableFFTBlock] = []
        if fft_block_config is not None:
            fft_cfg = dict(fft_block_config)
            after_blocks = fft_cfg.pop("after_blocks", [10])
            fft_cfg.setdefault("embed_dim", self._dino_embed_dim)
            fft_cfg.setdefault("freeze_original", True)
            self.fft_blocks = inject_fft_blocks(
                self.dino_teacher.encoder,
                after_blocks=after_blocks,
                **fft_cfg,
            )
            # Make them part of the model so they are saved / loaded
            self.fft_block_list = nn.ModuleList(self.fft_blocks)

        # =============================================================
        # 4. Alignment Head
        # =============================================================
        align_head_config = dict(align_head_config or {})
        # student_dim will be inferred lazily on first forward
        self._align_head_config = align_head_config
        self.align_head: Optional[TeacherStudentAlignHead] = None
        self._align_head_teacher_dim = self._dino_embed_dim

        # =============================================================
        # 5. Ensemble teachers (optional — kept from v1)
        # =============================================================
        self.ensemble_teachers: Optional[nn.Module] = None
        if teacher_specs:
            from .teacher_ensemble import TeacherEnsemble
            from .yolo_wrappers import YOLODetectionTeacher

            built: List[nn.Module] = []
            weights: List[float] = []
            for spec in teacher_specs:
                spec = dict(spec)
                t_type = spec.pop("type", "yolo").lower()
                w = float(spec.pop("weight", 1.0))
                if t_type == "yolo":
                    spec.setdefault("num_classes", num_classes)
                    built.append(YOLODetectionTeacher(**spec))
                else:
                    raise ValueError(f"Unsupported teacher type: {t_type}")
                weights.append(w)

            ens_cfg = dict(ensemble_config or {})
            ens_cfg.setdefault("num_classes", num_classes)
            ens_cfg.setdefault("teacher_weights", weights)
            self.ensemble_teachers = TeacherEnsemble(
                teacher_models=built, **ens_cfg,
            )

        # Hook storage for student backbone features
        self._student_backbone_feat: Dict[str, torch.Tensor] = {}

    # ------------------------------------------------------------------
    # Lazy alignment head init (needs student feature channel count)
    # ------------------------------------------------------------------
    def _ensure_align_head(self, student_dim: int, device: torch.device):
        if self.align_head is not None:
            return
        cfg = dict(self._align_head_config)
        cfg.setdefault("head_type", "MLP")
        cfg.setdefault("proj_dim", 1024)
        cfg.setdefault("normalize", True)
        self.align_head = TeacherStudentAlignHead(
            student_dim=student_dim,
            teacher_dim=self._align_head_teacher_dim,
            **cfg,
        ).to(device)
        logger.info(
            f"Alignment head built: student_dim={student_dim} → "
            f"teacher_dim={self._align_head_teacher_dim}, "
            f"head_type={cfg['head_type']}"
        )

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------
    def forward(
        self,
        images: torch.Tensor,
        return_teacher_features: bool = False,
        return_student_spatial_feat: bool = True,
    ) -> Dict[str, Any]:
        """
        Run student forward.  Optionally run the frozen DINO teacher.

        Returns a dict with:
        * ``boxes``, ``logits``, ``scores``, ``labels`` — student preds.
        * ``student_spatial_feat`` — ``[B, C_s, H_s, W_s]`` backbone feat
          (for alignment head; only if ``return_student_spatial_feat``).
        * ``dino_features`` — ``[B, D, H_p, W_p]`` teacher spatial feat
          (only if ``return_teacher_features``).
        """
        # ----- Student forward -----
        student_out = self.student(
            images,
            return_features=True,
            return_adapted_features=False,  # we do our own alignment now
        )

        result: Dict[str, Any] = {
            "boxes":  student_out["boxes"],
            "logits": student_out["logits"],
            "scores": student_out.get("scores"),
            "labels": student_out.get("labels"),
        }

        # Extract the chosen pyramid level as spatial feat for alignment
        if return_student_spatial_feat:
            feat_key = f"{self.student_align_layer}_features"
            spatial = student_out.get(feat_key)
            if spatial is not None:
                result["student_spatial_feat"] = spatial
                # lazy-init align head
                self._ensure_align_head(spatial.shape[1], spatial.device)

        # ----- Frozen DINO teacher forward -----
        if return_teacher_features:
            with torch.no_grad():
                dino_feat = self.dino_teacher(images)  # [B, D, H_p, W_p]
            result["dino_features"] = dino_feat

        return result

    # ------------------------------------------------------------------
    # Training-mode forward  (single pass, multiple losses)
    # ------------------------------------------------------------------
    def forward_train(
        self,
        images: torch.Tensor,
        gt_yolo_batch: Dict[str, torch.Tensor],
        compute_dino: bool = True,
    ) -> Dict[str, Any]:
        """
        Efficient training forward:

        1. Run student in **train mode** → raw predictions + P3/P4/P5.
        2. Compute GT detection loss from ``gt_yolo_batch``.
        3. Run frozen DINO → teacher spatial features (optional).

        Pseudo-label loss should be computed **separately** by the trainer
        using ``student.compute_loss(out["raw_preds"], pseudo_batch)``
        so that loss hyper-parameters can be temporarily modified
        (e.g. ``zero_pseudo_box_reg``).

        Args:
            images:         ``[B, 3, H, W]``.
            gt_yolo_batch:  YOLO batch dict (``batch_idx/cls/bboxes``).
            compute_dino:   Whether to forward the frozen DINO teacher.

        Returns:
            dict with ``raw_preds``, ``det_loss``, ``det_loss_items``,
            ``student_spatial_feat``, ``dino_features``.
        """
        # ---- Student raw forward (train mode) ----
        raw_preds, student_feats = self.student.forward_train_raw(images)

        # ---- GT detection loss ----
        det_loss, det_items = self.student.compute_loss(raw_preds, gt_yolo_batch)

        # ---- Extract spatial feature for alignment ----
        feat_key = f"{self.student_align_layer}_features"
        student_spatial = student_feats.get(feat_key)
        if student_spatial is not None:
            self._ensure_align_head(student_spatial.shape[1], student_spatial.device)

        # ---- Frozen DINO teacher ----
        dino_feat: Optional[torch.Tensor] = None
        if compute_dino:
            with torch.no_grad():
                dino_feat = self.dino_teacher(images)  # [B, D, H_p, W_p]

        return {
            "raw_preds": raw_preds,
            "det_loss": det_loss,
            "det_loss_items": det_items,
            "student_spatial_feat": student_spatial,
            "dino_features": dino_feat,
        }

    # ------------------------------------------------------------------
    # Alignment helpers (called by the Trainer, not forward)
    # ------------------------------------------------------------------
    def compute_align_loss(
        self,
        student_spatial_feat: torch.Tensor,
        dino_features: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute per-pixel spatial alignment loss.

        Args:
            student_spatial_feat: ``[B, C_s, H_s, W_s]``
            dino_features:  ``[B, D, H_p, W_p]``

        Returns:
            scalar loss.
        """
        assert self.align_head is not None, "Alignment head not initialised yet"
        projected = self.align_head(student_spatial_feat, dino_features.shape[2:])
        return self.align_head.align_loss(projected, dino_features)

    # ------------------------------------------------------------------
    # Trainable parameter helpers
    # ------------------------------------------------------------------
    def get_trainable_parameters(self) -> List[nn.Parameter]:
        params: List[nn.Parameter] = []
        for p in self.student.parameters():
            if p.requires_grad:
                params.append(p)
        if self.align_head is not None:
            params.extend(self.align_head.parameters())
        for blk in self.fft_blocks:
            params.extend(blk.parameters())
        return params

    def get_fft_parameters(self) -> List[nn.Parameter]:
        params: List[nn.Parameter] = []
        for blk in self.fft_blocks:
            params.extend(blk.parameters())
        return params

    def train(self, mode: bool = True):
        super().train(mode)
        # Keep frozen components in eval
        self.dino_teacher.train(False)
        if self.ensemble_teachers is not None:
            self.ensemble_teachers.train(False)
        return self


# ======================================================================
# Factory
# ======================================================================

def build_mtkd_model_v2(config: Dict[str, Any]) -> MTKDModelV2:
    """
    Build ``MTKDModelV2`` from a flat config dict.

    Expected keys (all optional with defaults):

    .. code-block:: python

        {
            "num_classes": 1,
            "student_config": {"student_type": "yolo", "weights": "yolo11s.pt", ...},
            "dino_config": {"model_name": "vit_base", "patch_size": 16, "embed_dim": 768},
            "dino_checkpoint": None,
            "align_head_config": {"head_type": "MLP", "proj_dim": 1024, "normalize": True},
            "student_align_layer": "p4",
            "fft_block_config": {"after_blocks": [10], "num_freq_bins": 32},
            "ensemble_config": {...},
            "teacher_specs": [...],
        }
    """
    return MTKDModelV2(
        student_config=config.get("student_config"),
        num_classes=config.get("num_classes", 1),
        dino_config=config.get("dino_config"),
        dino_checkpoint=config.get("dino_checkpoint"),
        align_head_config=config.get("align_head_config"),
        student_align_layer=config.get("student_align_layer", "p4"),
        fft_block_config=config.get("fft_block_config"),
        ensemble_config=config.get("ensemble_config"),
        teacher_specs=config.get("teacher_specs"),
    )
