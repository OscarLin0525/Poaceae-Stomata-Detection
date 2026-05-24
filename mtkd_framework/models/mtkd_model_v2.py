"""
MTKDModel v2 — refactored following the DINO Teacher engine pattern
=====================================================================

Changes from ``mtkd_model.py`` (v1):
-------------------------------------
1. **DINOFeatureTeacher** → ``engine.DinoFeatureExtractor`` (spatial output,
   proper preprocessing, frozen).
2. **FeatureAdapter** (``nn.Linear`` + global pool) → ``engine.TeacherStudentAlignHead``
   (``nn.Conv2d(1×1)`` per-pixel + ``F.interpolate``).  This fixes **C1/M1**.
3. MTKD-side FFT injection is disabled; DINO stays frozen and unmodified.
4. Pseudo-label pathway mirrors DINO Teacher: filtered pseudo-labels are treated
   as GT supervision (standard detection loss), **not** element-wise KL.
   The user's pretrained wheat model supplies these PLs externally (see
   ``MTKDTrainerV2.load_pseudo_labels``).
5. ``forward()`` returns spatial features so the trainer can compute the
   alignment loss *outside* the model — exactly as in DINO Teacher.

Backwards compat
-----------------
* This module is the active MTKD implementation used by ``run_v2.py``.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.batchnorm import _BatchNorm

from ..engine.build_dino import DinoFeatureExtractor
from ..engine.align_head import TeacherStudentAlignHead

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


class StomataPriorHead(nn.Module):
    """Small spatial head that predicts a stomata prior map ``M(h, w)``."""

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int = 256,
        use_gelu: bool = True,
    ) -> None:
        super().__init__()
        activation: nn.Module = nn.GELU() if use_gelu else nn.ReLU()
        self.layers = nn.Sequential(
            nn.Conv2d(in_dim, hidden_dim, kernel_size=1, stride=1),
            activation,
            nn.Conv2d(hidden_dim, 1, kernel_size=1, stride=1),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.layers(features)


class AnchorConditionedPeriodicPriorHead(nn.Module):
    """Student-side prior head with horizontal propagation conditioned on anchor seeds."""

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int = 256,
        use_gelu: bool = True,
        prop_kernel_size: int = 9,
        prop_dilations: Sequence[int] = (1, 2, 4),
        anchor_pool_kernel: int = 11,
        propagation_weight: float = 0.85,
    ) -> None:
        super().__init__()
        activation: nn.Module = nn.GELU() if use_gelu else nn.ReLU()
        self.hidden = nn.Sequential(
            nn.Conv2d(in_dim, hidden_dim, kernel_size=1, stride=1),
            activation,
        )
        self.base_head = nn.Conv2d(hidden_dim, 1, kernel_size=1, stride=1)
        k = max(3, int(prop_kernel_size))
        if k % 2 == 0:
            k += 1
        self.anchor_pool_kernel = max(3, int(anchor_pool_kernel))
        if self.anchor_pool_kernel % 2 == 0:
            self.anchor_pool_kernel += 1
        dilations = [max(1, int(d)) for d in prop_dilations] or [1]
        self.prop_branches = nn.ModuleList()
        for dilation in dilations:
            pad = dilation * (k // 2)
            self.prop_branches.append(
                nn.Conv2d(
                    hidden_dim,
                    hidden_dim,
                    kernel_size=(1, k),
                    stride=1,
                    padding=(0, pad),
                    dilation=(1, dilation),
                    groups=hidden_dim,
                )
            )
        self.propagation_fuse = nn.Sequential(
            nn.Conv2d(hidden_dim * len(self.prop_branches), hidden_dim, kernel_size=1, stride=1),
            activation,
        )
        self.propagation_head = nn.Conv2d(hidden_dim, 1, kernel_size=1, stride=1)
        self.propagation_weight = float(propagation_weight)

    def forward_components(self, features: torch.Tensor) -> Dict[str, torch.Tensor]:
        hidden = self.hidden(features)
        base_logits = self.base_head(hidden)
        base_prob = torch.sigmoid(base_logits)
        anchor_band = F.max_pool2d(
            base_prob,
            kernel_size=(1, self.anchor_pool_kernel),
            stride=1,
            padding=(0, self.anchor_pool_kernel // 2),
        )
        context = hidden * (0.35 + 0.65 * anchor_band)
        prop_feats = [branch(context) for branch in self.prop_branches]
        propagation_hidden = self.propagation_fuse(torch.cat(prop_feats, dim=1))
        propagation_logits = self.propagation_head(propagation_hidden)
        logits = base_logits + self.propagation_weight * propagation_logits
        return {
            "logits": logits,
            "base_logits": base_logits,
            "propagation_logits": propagation_logits,
            "base_prob": base_prob,
            "anchor_band": anchor_band,
            "propagation_prob": torch.sigmoid(propagation_logits),
        }

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.forward_components(features)["logits"]


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
        student_freeze_config: Optional[Dict[str, Any]] = None,
        # DINO teacher
        dino_config: Optional[Dict[str, Any]] = None,
        dino_checkpoint: Optional[str] = None,
        # Wheat detection teacher (for online pseudo labels)
        wheat_teacher_config: Optional[Dict[str, Any]] = None,
        # Alignment head
        align_head_config: Optional[Dict[str, Any]] = None,
        prior_head_config: Optional[Dict[str, Any]] = None,
        student_align_layer: str = "p4",  # which pyramid level to align
        # Legacy arg kept for config compatibility (ignored)
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

        self.student_freeze_config = dict(student_freeze_config or {})
        self._student_freeze_enabled = bool(self.student_freeze_config.get("enabled", False))
        self._apply_student_freeze_config()

        # MTKD v2 trainer currently relies on Ultralytics-style student APIs.
        required_student_api = (
            "forward_train_raw",
            "compute_loss",
            "_ensure_criterion",
        )
        missing_api = [
            name for name in required_student_api
            if not callable(getattr(self.student, name, None))
        ]
        if missing_api:
            raise TypeError(
                "MTKDModelV2 requires a YOLO-style student interface. "
                f"Got {self.student.__class__.__name__} missing methods: {missing_api}. "
                "Use YOLOStudentDetector or adapt the student implementation to expose "
                "forward_train_raw/compute_loss/_ensure_criterion."
            )

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
        # 2.5 Optional frozen Wheat detection teacher
        # =============================================================
        self.wheat_teacher: Optional[nn.Module] = None
        if wheat_teacher_config is not None:
            from .yolo_wrappers import YOLODetectionTeacher

            wheat_cfg = dict(wheat_teacher_config)
            wheat_cfg.setdefault("num_classes", num_classes)
            self.wheat_teacher = YOLODetectionTeacher(**wheat_cfg)

        # =============================================================
        # 3. MTKD-side FFT injection disabled
        # =============================================================
        self.fft_blocks: List[nn.Module] = []
        if fft_block_config is not None:
            logger.warning(
                "fft_block_config is ignored: MTKD FFT/PluggableFFTBlock has been disabled."
            )

        # =============================================================
        # 4. Alignment Head
        # =============================================================
        align_head_config = dict(align_head_config or {})
        # student_dim will be inferred lazily on first forward
        self._align_head_config = align_head_config
        self.align_head: Optional[TeacherStudentAlignHead] = None
        self._align_head_teacher_dim = self._dino_embed_dim

        # =============================================================
        # 4.5 Stomata prior head (DINO -> spatial prior map)
        # =============================================================
        prior_cfg = dict(prior_head_config or {})
        self.use_stomata_prior = bool(prior_cfg.get("enabled", False))
        self.prior_head_mode = str(prior_cfg.get("mode", "origin")).strip().lower()
        if self.prior_head_mode not in {"origin", "freq_adaption"}:
            raise ValueError(
                "prior_head_config.mode must be one of: origin / freq_adaption, got "
                f"{self.prior_head_mode!r}"
            )
        self.prior_detach_for_align = bool(prior_cfg.get("detach_for_align", True))
        self.prior_apply_to_detection = bool(prior_cfg.get("apply_to_detection", False))
        self.prior_gate_strength = float(prior_cfg.get("gate_strength", 1.0))
        self.prior_gate_hard = bool(prior_cfg.get("gate_hard", False))
        self.prior_gate_threshold = float(prior_cfg.get("gate_threshold", 0.5))
        self.prior_gate_bg_scale = (
            None if prior_cfg.get("gate_bg_scale", None) is None else float(prior_cfg.get("gate_bg_scale"))
        )
        self.prior_gate_fg_scale = (
            None if prior_cfg.get("gate_fg_scale", None) is None else float(prior_cfg.get("gate_fg_scale"))
        )
        self.stomata_prior_head: Optional[StomataPriorHead] = None
        self.student_prior_head: Optional[StomataPriorHead] = None
        self._student_prior_head_config = {
            "hidden_dim": int(prior_cfg.get("hidden_dim", 256)),
            "use_gelu": bool(prior_cfg.get("use_gelu", True)),
            "prop_kernel_size": int(prior_cfg.get("prop_kernel_size", 9)),
            "prop_dilations": tuple(prior_cfg.get("prop_dilations", (1, 2, 4))),
            "anchor_pool_kernel": int(prior_cfg.get("anchor_pool_kernel", 11)),
            "propagation_weight": float(prior_cfg.get("propagation_weight", 0.85)),
        }
        if self.use_stomata_prior:
            if self.prior_head_mode == "origin":
                self.stomata_prior_head = StomataPriorHead(
                    in_dim=self._dino_embed_dim,
                    hidden_dim=self._student_prior_head_config["hidden_dim"],
                    use_gelu=self._student_prior_head_config["use_gelu"],
                )
                logger.info(
                    "Stomata prior head enabled in origin mode: dino_dim=%d hidden_dim=%d "
                    "detach_for_align=%s",
                    self._dino_embed_dim,
                    self._student_prior_head_config["hidden_dim"],
                    self.prior_detach_for_align,
                )
            else:
                logger.info(
                    "Stomata prior head enabled in freq_adaption mode: student_dim=<lazy> "
                    "hidden_dim=%d detach_for_align=%s apply_to_detection=%s gate_strength=%.3f "
                    "gate_hard=%s gate_threshold=%.3f bg_scale=%s fg_scale=%s",
                    self._student_prior_head_config["hidden_dim"],
                    self.prior_detach_for_align,
                    self.prior_apply_to_detection,
                    self.prior_gate_strength,
                    self.prior_gate_hard,
                    self.prior_gate_threshold,
                    self.prior_gate_bg_scale,
                    self.prior_gate_fg_scale,
                )

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

    def _apply_student_freeze_config(self) -> None:
        cfg = dict(self.student_freeze_config or {})
        if not bool(cfg.get("enabled", False)):
            return

        trainable_indices = {
            str(int(idx))
            for idx in cfg.get("trainable_layer_indices", [])
        }
        trainable_keywords = [
            str(x) for x in cfg.get("trainable_name_keywords", [])
            if str(x).strip()
        ]

        total = 0
        kept = 0
        for name, param in self.student.named_parameters():
            total += param.numel()
            param.requires_grad_(False)
            keep = False
            parts = name.split(".")
            if len(parts) >= 4 and parts[0] == "det_model" and parts[1] == "model":
                if parts[2] in trainable_indices:
                    keep = True
                if not keep:
                    try:
                        layer_idx = int(parts[2])
                        layer = self.student.det_model.model[layer_idx]
                        if bool(getattr(layer, "is_support_gate", False)):
                            keep = True
                    except Exception:
                        pass
            if not keep and trainable_keywords:
                keep = any(keyword in name for keyword in trainable_keywords)
            if keep:
                param.requires_grad_(True)
                kept += param.numel()

        logger.info(
            "Student freeze-majority enabled: trainable student params kept=%d / total=%d "
            "(layer_indices=%s, keywords=%s)",
            kept,
            total,
            sorted(trainable_indices, key=lambda x: int(x)) if trainable_indices else [],
            trainable_keywords,
        )
        self._set_frozen_student_modules_eval()

    def _set_frozen_student_modules_eval(self) -> None:
        if not getattr(self, "_student_freeze_enabled", False):
            return

        frozen_modules = 0
        frozen_bn = 0
        for module in self.student.modules():
            params = list(module.parameters(recurse=False))
            if not params:
                continue
            if any(param.requires_grad for param in params):
                continue
            module.eval()
            frozen_modules += 1
            if isinstance(module, _BatchNorm):
                frozen_bn += 1

        logger.info(
            "Frozen student modules forced to eval(): modules=%d batchnorm=%d",
            frozen_modules,
            frozen_bn,
        )

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
        # Flag so the trainer can add these params to the optimizer
        self._align_head_just_created = True

    def _ensure_student_prior_head(self, student_dim: int, device: torch.device):
        if self.student_prior_head is not None:
            return
        cfg = dict(self._student_prior_head_config)
        self.student_prior_head = AnchorConditionedPeriodicPriorHead(
            in_dim=student_dim,
            hidden_dim=int(cfg.get("hidden_dim", 256)),
            use_gelu=bool(cfg.get("use_gelu", True)),
            prop_kernel_size=int(cfg.get("prop_kernel_size", 9)),
            prop_dilations=tuple(cfg.get("prop_dilations", (1, 2, 4))),
            anchor_pool_kernel=int(cfg.get("anchor_pool_kernel", 11)),
            propagation_weight=float(cfg.get("propagation_weight", 0.85)),
        ).to(device)
        logger.info(
            "Student prior head built: student_dim=%d hidden_dim=%d mode=%s prop_kernel=%d prop_dilations=%s anchor_pool=%d prop_weight=%.3f",
            student_dim,
            int(cfg.get("hidden_dim", 256)),
            self.prior_head_mode,
            int(cfg.get("prop_kernel_size", 9)),
            tuple(cfg.get("prop_dilations", (1, 2, 4))),
            int(cfg.get("anchor_pool_kernel", 11)),
            float(cfg.get("propagation_weight", 0.85)),
        )
        self._student_prior_head_just_created = True

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
                if self.use_stomata_prior and self.prior_head_mode == "freq_adaption":
                    self._ensure_student_prior_head(spatial.shape[1], spatial.device)

        # ----- DINO teacher forward -----
        if return_teacher_features:
            with torch.no_grad():
                dino_feat = self.dino_teacher(images)
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
        teacher_images: Optional[torch.Tensor] = None,
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
            teacher_images: Optional separate images for the DINO teacher
                            (e.g. un-augmented copies when
                            ``align_easy_only=True``).  Falls back to
                            ``images`` when ``None``.

        Returns:
            dict with ``raw_preds``, ``det_loss``, ``det_loss_items``,
            ``student_spatial_feat``, ``dino_features``.
        """
        # ---- Student raw forward (train mode) ----
        # ---- Extract spatial feature for alignment ----
        raw_preds, student_feats = self.student.forward_train_raw(images)
        feat_key = f"{self.student_align_layer}_features"
        student_spatial = student_feats.get(feat_key)
        if student_spatial is not None:
            self._ensure_align_head(student_spatial.shape[1], student_spatial.device)
            if self.use_stomata_prior and self.prior_head_mode == "freq_adaption":
                self._ensure_student_prior_head(student_spatial.shape[1], student_spatial.device)

        if (
            self.use_stomata_prior
            and self.prior_head_mode == "freq_adaption"
            and self.prior_apply_to_detection
            and student_spatial is not None
        ):
            prior_out = self.predict_stomata_prior(
                student_spatial_feat=student_spatial,
                output_hw=student_spatial.shape[2:],
            )
            if prior_out is not None:
                raw_preds, student_feats = self.student.forward_train_raw(
                    images,
                    feature_gate={
                        "level": self.student_align_layer,
                        "support": prior_out["prob"],
                        "strength": self.prior_gate_strength,
                        "hard": self.prior_gate_hard,
                        "threshold": self.prior_gate_threshold,
                        "bg_scale": self.prior_gate_bg_scale,
                        "fg_scale": self.prior_gate_fg_scale,
                    },
                )
                student_spatial = student_feats.get(feat_key)
                if student_spatial is not None:
                    self._ensure_align_head(student_spatial.shape[1], student_spatial.device)

        # ---- GT detection loss ----
        det_loss, det_items = self.student.compute_loss(raw_preds, gt_yolo_batch)

        # ---- DINO teacher ----
        dino_feat: Optional[torch.Tensor] = None
        if compute_dino:
            dino_input = teacher_images if teacher_images is not None else images
            with torch.no_grad():
                dino_feat = self.dino_teacher(dino_input)  # [B, D, H_p, W_p]

        return {
            "raw_preds": raw_preds,
            "det_loss": det_loss,
            "det_loss_items": det_items,
            "student_spatial_feat": student_spatial,
            "dino_features": dino_feat,
        }

    # ------------------------------------------------------------------
    # Optional frozen detection-teacher helper
    # ------------------------------------------------------------------
    @torch.no_grad()
    def get_wheat_teacher_predictions(
        self,
        images: torch.Tensor,
    ) -> Optional[Dict[str, torch.Tensor]]:
        """
        Run the optional frozen wheat teacher for online pseudo-label generation.

        Returns ``None`` if wheat teacher is not configured.
        """
        if self.wheat_teacher is None:
            return None
        return self.wheat_teacher(images)

    # ------------------------------------------------------------------
    # Alignment helpers (called by the Trainer, not forward)
    # ------------------------------------------------------------------
    def compute_align_loss(
        self,
        student_spatial_feat: torch.Tensor,
        dino_features: torch.Tensor,
        prior_mask: Optional[torch.Tensor] = None,
        return_details: bool = False,
    ) -> torch.Tensor | Dict[str, torch.Tensor]:
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
        loss_map, sim_map = self._compute_align_loss_map(projected, dino_features)

        weight_map: Optional[torch.Tensor] = None
        if prior_mask is not None:
            if prior_mask.ndim != 4 or prior_mask.shape[1] != 1:
                raise ValueError(
                    "prior_mask must have shape [B, 1, H, W], got "
                    f"{tuple(prior_mask.shape)}"
                )
            if prior_mask.shape[2:] != loss_map.shape[1:]:
                prior_mask = F.interpolate(
                    prior_mask,
                    size=loss_map.shape[1:],
                    mode="bilinear",
                    align_corners=False,
                )
            weight_map = prior_mask.squeeze(1)
            if self.prior_detach_for_align:
                weight_map = weight_map.detach()
            denom = weight_map.sum().clamp_min(1e-6)
            loss = (loss_map * weight_map).sum() / denom
        else:
            loss = loss_map.mean()

        if not return_details:
            return loss

        return {
            "loss": loss,
            "projected_student_feat": projected,
            "loss_map": loss_map,
            "similarity_map": sim_map,
            "weight_map": weight_map,
        }

    @staticmethod
    def _normalize_map_per_sample(values: torch.Tensor) -> torch.Tensor:
        if values.ndim != 3:
            raise ValueError(
                "_normalize_map_per_sample expects [B, H, W], got "
                f"{tuple(values.shape)}"
            )
        flat = values.reshape(values.shape[0], -1)
        vmin = flat.min(dim=1, keepdim=True).values
        vmax = flat.max(dim=1, keepdim=True).values
        norm = (flat - vmin) / (vmax - vmin).clamp_min(1e-6)
        return norm.reshape_as(values)

    def _build_pca_prior_support(
        self,
        dino_features: torch.Tensor,
        *,
        pca_components: int = 3,
        target_coverage: float = 0.12,
    ) -> tuple[torch.Tensor, Dict[str, float]]:
        """
        Build a PCA-derived spatial prior directly from DINO patch features.

        This approximates the user's standalone PCA inspection workflow:
        project DINO tokens onto the top principal directions, try both signs,
        and keep the candidate with the strongest sparse / peaky stomata-like
        response.
        """
        if dino_features.ndim != 4:
            raise ValueError(
                "_build_pca_prior_support expects [B, C, H, W], got "
                f"{tuple(dino_features.shape)}"
            )

        batch, _channels, height, width = dino_features.shape
        device = dino_features.device
        dtype = dino_features.dtype
        coverage = max(0.01, min(0.50, float(target_coverage)))
        q = max(1, min(int(pca_components), min(dino_features.shape[1], height * width) - 1))

        supports: List[torch.Tensor] = []
        best_component_ids: List[float] = []
        best_signs: List[float] = []
        best_scores: List[float] = []
        peak_densities: List[float] = []

        for b in range(batch):
            tokens = dino_features[b].permute(1, 2, 0).reshape(height * width, -1).float()
            tokens = tokens - tokens.mean(dim=0, keepdim=True)
            if q <= 0 or tokens.shape[0] < 2:
                zero_map = torch.zeros((height, width), device=device, dtype=dtype)
                supports.append(zero_map)
                best_component_ids.append(-1.0)
                best_signs.append(0.0)
                best_scores.append(0.0)
                peak_densities.append(0.0)
                continue

            try:
                _u, _s, v = torch.pca_lowrank(tokens, q=q, center=False)
                comp_scores = tokens @ v[:, :q]
            except RuntimeError:
                # Fall back to a simple energy map if PCA is numerically unstable.
                fallback = torch.linalg.norm(
                    dino_features[b].float(), dim=0
                )
                fallback = self._normalize_map_per_sample(fallback.unsqueeze(0))[0]
                supports.append(fallback.to(device=device, dtype=dtype))
                best_component_ids.append(-1.0)
                best_signs.append(0.0)
                best_scores.append(0.0)
                peak_densities.append(0.0)
                continue

            best_map: Optional[torch.Tensor] = None
            best_meta: tuple[float, float, float, float] = (-1.0, 0.0, -1e9, 0.0)

            for comp_idx in range(q):
                raw_comp = comp_scores[:, comp_idx].reshape(height, width)
                for sign in (1.0, -1.0):
                    candidate = raw_comp * sign
                    flat = candidate.reshape(-1)
                    cmin = flat.min()
                    cmax = flat.max()
                    norm = (candidate - cmin) / (cmax - cmin).clamp_min(1e-6)

                    thresh = torch.quantile(norm.reshape(-1), q=max(0.0, 1.0 - coverage))
                    hard = (norm >= thresh).float()

                    pooled = F.max_pool2d(
                        norm.unsqueeze(0).unsqueeze(0),
                        kernel_size=3,
                        stride=1,
                        padding=1,
                    )[0, 0]
                    peak_mask = ((norm >= pooled - 1e-6) & (norm >= max(float(thresh), 0.65))).float()
                    peak_density = float(peak_mask.mean().item())
                    contrast = float(norm.std().item())
                    row_std = float(norm.mean(dim=1).std().item())
                    col_std = float(norm.mean(dim=0).std().item())
                    hard_cov = float(hard.mean().item())
                    cov_penalty = abs(hard_cov - coverage)
                    score = (
                        2.2 * peak_density
                        + 0.9 * contrast
                        + 0.6 * row_std
                        + 0.6 * col_std
                        - 1.8 * cov_penalty
                    )

                    if score > best_meta[2]:
                        best_map = norm
                        best_meta = (
                            float(comp_idx + 1),
                            float(sign),
                            float(score),
                            peak_density,
                        )

            if best_map is None:
                best_map = torch.zeros((height, width), device=device, dtype=torch.float32)

            supports.append(best_map.to(device=device, dtype=dtype))
            best_component_ids.append(best_meta[0])
            best_signs.append(best_meta[1])
            best_scores.append(best_meta[2])
            peak_densities.append(best_meta[3])

        support = torch.stack(supports, dim=0)
        stats = {
            "align_pattern_pca_component_mean": float(sum(best_component_ids) / max(len(best_component_ids), 1)),
            "align_pattern_pca_sign_mean": float(sum(best_signs) / max(len(best_signs), 1)),
            "align_pattern_pca_score_mean": float(sum(best_scores) / max(len(best_scores), 1)),
            "align_pattern_pca_peak_density_mean": float(sum(peak_densities) / max(len(peak_densities), 1)),
        }
        return support, stats

    def _build_structural_row_prior_support(
        self,
        evidence: torch.Tensor,
        *,
        period_ratio_x: float = 0.0,
        row_period_ratio_y: float = 0.0,
        row_tolerance_ratio_y: float = 0.0,
        seed_threshold: float = 0.55,
        min_row_seeds: int = 2,
        cross_row_strength: float = 0.0,
    ) -> tuple[torch.Tensor, Dict[str, float]]:
        """
        Build a lightweight row / frequency prior in DINO patch space.

        This keeps train-time support aligned with the spacing prior bank
        without reusing the entire standalone detect_test pipeline.
        """
        if evidence.ndim != 3:
            raise ValueError(
                "_build_structural_row_prior_support expects [B, H, W], got "
                f"{tuple(evidence.shape)}"
            )

        batch, height, width = evidence.shape
        device = evidence.device
        dtype = evidence.dtype
        zero = torch.zeros_like(evidence)

        period_cells = max(float(period_ratio_x) * float(width), 0.0)
        row_period_cells = max(float(row_period_ratio_y) * float(height), 0.0)
        row_tol_cells = max(float(row_tolerance_ratio_y) * float(height), 1.0)
        if period_cells < 2.0:
            return zero, {
                "align_pattern_struct_period_cells_mean": period_cells,
                "align_pattern_struct_row_period_cells_mean": row_period_cells,
                "align_pattern_struct_row_tol_cells_mean": row_tol_cells,
                "align_pattern_struct_seed_count_mean": 0.0,
                "align_pattern_struct_used_row_count_mean": 0.0,
            }

        pooled = F.max_pool2d(
            evidence.unsqueeze(1),
            kernel_size=3,
            stride=1,
            padding=1,
        ).squeeze(1)
        seed_mask = (evidence >= pooled - 1e-6) & (evidence >= float(seed_threshold))

        grid_y = torch.arange(height, device=device, dtype=torch.float32).view(height, 1)
        grid_x = torch.arange(width, device=device, dtype=torch.float32).view(1, width)
        row_sigma = max(row_tol_cells * 0.80, 1.0)
        period_sigma = max(period_cells * 0.22, 1.0)
        row_period_sigma = max(row_period_cells * 0.28, 1.0)
        min_row_seeds = max(1, int(min_row_seeds))

        priors: List[torch.Tensor] = []
        seed_counts: List[float] = []
        row_counts: List[float] = []

        for batch_idx in range(batch):
            peaks = torch.nonzero(seed_mask[batch_idx], as_tuple=False)
            seed_counts.append(float(peaks.shape[0]))
            if peaks.numel() == 0:
                priors.append(torch.zeros((height, width), device=device, dtype=dtype))
                row_counts.append(0.0)
                continue

            if peaks.shape[0] > 256:
                scores = evidence[batch_idx, peaks[:, 0], peaks[:, 1]]
                keep = torch.topk(scores, k=min(256, int(scores.numel())), largest=True).indices
                peaks = peaks[keep]

            peaks = peaks[torch.argsort(peaks[:, 0])]
            rows: List[Dict[str, Any]] = []

            for peak in peaks:
                y = float(peak[0].item())
                x = float(peak[1].item())
                val = float(evidence[batch_idx, int(peak[0].item()), int(peak[1].item())].item())

                best_row_idx = -1
                best_delta = 1e9
                for row_idx, row in enumerate(rows):
                    delta = abs(y - row["mean_y"])
                    if delta <= row_tol_cells and delta < best_delta:
                        best_delta = delta
                        best_row_idx = row_idx

                if best_row_idx < 0:
                    rows.append({"xs": [x], "ys": [y], "vals": [val], "mean_y": y})
                else:
                    row = rows[best_row_idx]
                    row["xs"].append(x)
                    row["ys"].append(y)
                    row["vals"].append(val)
                    row["mean_y"] = sum(row["ys"]) / max(len(row["ys"]), 1)

            valid_rows = [row for row in rows if len(row["xs"]) >= min_row_seeds]
            row_counts.append(float(len(valid_rows)))
            if not valid_rows:
                priors.append(torch.zeros((height, width), device=device, dtype=dtype))
                continue

            prior = torch.zeros((height, width), device=device, dtype=torch.float32)
            row_centers: List[float] = []
            row_weights: List[float] = []

            for row in valid_rows:
                xs = sorted(float(v) for v in row["xs"])
                ys = [float(v) for v in row["ys"]]
                vals = [float(v) for v in row["vals"]]
                mean_y = float(sum(ys) / max(len(ys), 1))
                row_weight = float(sum(vals) / max(len(vals), 1))
                row_centers.append(mean_y)
                row_weights.append(row_weight)

                remainders = torch.tensor(
                    [x % period_cells for x in xs],
                    device=device,
                    dtype=torch.float32,
                )
                phase = float(remainders.mean().item())
                phase_dist = torch.remainder(grid_x - phase + 0.5 * period_cells, period_cells) - 0.5 * period_cells
                period_score = torch.exp(-0.5 * (phase_dist / period_sigma) ** 2)
                row_score = torch.exp(-0.5 * ((grid_y - mean_y) / row_sigma) ** 2)
                x_pad = max(period_cells * 1.25, 2.0)
                x_min = min(xs) - x_pad
                x_max = max(xs) + x_pad
                x_support = ((grid_x >= x_min) & (grid_x <= x_max)).to(torch.float32)
                row_prior = row_score * period_score * x_support * row_weight
                prior = torch.maximum(prior, row_prior)

            if float(cross_row_strength) > 0.0 and row_period_cells >= 2.0 and row_centers:
                cross_prior = torch.zeros_like(prior)
                for center_y, row_weight in zip(row_centers, row_weights):
                    min_k = int((-center_y) / row_period_cells) - 1
                    max_k = int((float(height) - center_y) / row_period_cells) + 1
                    for k in range(min_k, max_k + 1):
                        band_y = center_y + float(k) * row_period_cells
                        if band_y < -3.0 * row_period_sigma or band_y > float(height) + 3.0 * row_period_sigma:
                            continue
                        band = torch.exp(-0.5 * ((grid_y - band_y) / row_period_sigma) ** 2) * row_weight
                        cross_prior = torch.maximum(cross_prior, band)
                prior = torch.maximum(
                    prior,
                    float(cross_row_strength) * cross_prior * (0.35 + 0.65 * evidence[batch_idx].float()),
                )

            prior = prior / prior.max().clamp_min(1e-6)
            priors.append(prior.to(dtype=dtype))

        support = torch.stack(priors, dim=0)
        stats = {
            "align_pattern_struct_period_cells_mean": period_cells,
            "align_pattern_struct_row_period_cells_mean": row_period_cells,
            "align_pattern_struct_row_tol_cells_mean": row_tol_cells,
            "align_pattern_struct_seed_count_mean": float(sum(seed_counts) / max(len(seed_counts), 1)),
            "align_pattern_struct_used_row_count_mean": float(sum(row_counts) / max(len(row_counts), 1)),
        }
        return support, stats

    def build_pattern_align_mask(
        self,
        student_spatial_feat: torch.Tensor,
        dino_features: torch.Tensor,
        *,
        target_coverage: float = 0.12,
        min_coverage: float = 0.02,
        max_coverage: float = 0.30,
        sim_weight: float = 0.55,
        dino_weight: float = 0.30,
        student_weight: float = 0.15,
        temperature: float = 0.08,
        mask_floor: float = 0.10,
        mode: str = "legacy",
        pca_components: int = 3,
        hybrid_pca_weight: float = 0.5,
        filtered_prior_strength: float = 0.75,
        filtered_completion_strength: float = 0.65,
        filtered_completion_gamma: float = 1.0,
        filtered_noise_suppress: float = 0.25,
        period_ratio_x: float = 0.0,
        row_period_ratio_y: float = 0.0,
        row_tolerance_ratio_y: float = 0.0,
        structural_prior_strength: float = 0.0,
        structural_cross_row_strength: float = 0.0,
        structural_seed_threshold: float = 0.55,
        structural_min_row_seeds: int = 2,
        hard_mask: bool = False,
        hard_mask_threshold: float = 0.5,
        detach: bool = True,
    ) -> tuple[torch.Tensor, Dict[str, float]]:
        """
        Build a soft pattern-guided alignment mask in DINO patch space.

        The mask combines:
        1) student-teacher cosine agreement,
        2) DINO feature energy,
        3) projected-student feature energy.

        Returns:
            mask: [B, 1, H, W] in [0, 1]
            stats: scalar diagnostics for logging
        """
        align_mode = str(mode).strip().lower()
        if align_mode not in {"legacy", "pca_prior", "hybrid", "filtered_support"}:
            raise ValueError(
                "build_pattern_align_mask mode must be one of: legacy / pca_prior / hybrid / filtered_support, got "
                f"{align_mode!r}"
            )

        if self.align_head is None and align_mode in {"legacy", "hybrid", "filtered_support"}:
            raise RuntimeError("Alignment head not initialised")

        projected: Optional[torch.Tensor] = None
        sim_support: Optional[torch.Tensor] = None
        dino_energy: Optional[torch.Tensor] = None
        student_energy: Optional[torch.Tensor] = None
        loss_map: Optional[torch.Tensor] = None
        sim_map: Optional[torch.Tensor] = None
        legacy_support: Optional[torch.Tensor] = None

        if align_mode in {"legacy", "hybrid", "filtered_support"}:
            assert self.align_head is not None
            projected = self.align_head(student_spatial_feat, dino_features.shape[2:])
            loss_map, sim_map = self._compute_align_loss_map(projected, dino_features)

            if self.align_head.normalize_feature:
                sim_support = ((sim_map + 1.0) * 0.5).clamp(0.0, 1.0)
            else:
                sim_support = self._normalize_map_per_sample((-loss_map).float())

            dino_energy = self._normalize_map_per_sample(torch.linalg.norm(dino_features.float(), dim=1))
            student_energy = self._normalize_map_per_sample(torch.linalg.norm(projected.float(), dim=1))

            weights = torch.tensor(
                [float(sim_weight), float(dino_weight), float(student_weight)],
                device=dino_features.device,
                dtype=dino_features.dtype,
            ).clamp_min(0.0)
            if float(weights.sum().item()) <= 0.0:
                weights = torch.tensor([1.0, 0.0, 0.0], device=weights.device, dtype=weights.dtype)
            weights = weights / weights.sum().clamp_min(1e-6)

            legacy_support = (
                weights[0] * sim_support
                + weights[1] * dino_energy
                + weights[2] * student_energy
            ).clamp(0.0, 1.0)

        pca_support: Optional[torch.Tensor] = None
        pca_stats: Dict[str, float] = {}
        use_pca_prior = align_mode in {"pca_prior", "hybrid"} or (
            align_mode == "filtered_support" and int(pca_components) > 0
        )
        if use_pca_prior:
            pca_support, pca_stats = self._build_pca_prior_support(
                dino_features,
                pca_components=pca_components,
                target_coverage=target_coverage,
            )

        if align_mode == "legacy":
            assert legacy_support is not None
            support = legacy_support
        elif align_mode == "pca_prior":
            assert pca_support is not None
            support = pca_support
        elif align_mode == "filtered_support":
            if legacy_support is not None:
                local_evidence = legacy_support
            elif dino_energy is not None:
                local_evidence = dino_energy
            else:
                local_evidence = self._normalize_map_per_sample(torch.linalg.norm(dino_features.float(), dim=1))
            prior_map = pca_support if pca_support is not None else local_evidence
            structural_stats: Dict[str, float] = {}
            if float(structural_prior_strength) > 0.0:
                structural_prior, structural_stats = self._build_structural_row_prior_support(
                    torch.maximum(local_evidence, prior_map).clamp(0.0, 1.0),
                    period_ratio_x=period_ratio_x,
                    row_period_ratio_y=row_period_ratio_y,
                    row_tolerance_ratio_y=row_tolerance_ratio_y,
                    seed_threshold=structural_seed_threshold,
                    min_row_seeds=structural_min_row_seeds,
                    cross_row_strength=structural_cross_row_strength,
                )
                prior_map = torch.maximum(
                    prior_map,
                    (float(structural_prior_strength) * structural_prior).clamp(0.0, 1.0),
                )

            pattern_base = (prior_map * (0.35 + 0.65 * local_evidence)).clamp(0.0, 1.0)
            support = (
                local_evidence
                + float(filtered_prior_strength) * pattern_base * (1.0 - local_evidence)
            ).clamp(0.0, 1.0)
            completion = torch.clamp(pattern_base - local_evidence, min=0.0)
            completion = completion.pow(max(float(filtered_completion_gamma), 1e-3))
            support = torch.maximum(
                support,
                (
                    local_evidence
                    + float(filtered_completion_strength) * completion
                ).clamp(0.0, 1.0),
            )
            noise_term = (
                torch.clamp(local_evidence - prior_map, min=0.0)
                * torch.clamp(1.0 - prior_map, min=0.0)
            )
            support = (
                support - float(filtered_noise_suppress) * noise_term
            ).clamp(0.0, 1.0)
        else:
            assert legacy_support is not None and pca_support is not None
            pca_w = max(0.0, min(1.0, float(hybrid_pca_weight)))
            support = ((1.0 - pca_w) * legacy_support + pca_w * pca_support).clamp(0.0, 1.0)

        coverage = float(target_coverage)
        coverage = max(float(min_coverage), min(float(max_coverage), coverage))
        q = max(0.0, min(1.0, 1.0 - coverage))
        flat = support.reshape(support.shape[0], -1)
        thresh = torch.quantile(flat, q=q, dim=1, keepdim=True)

        temp = max(float(temperature), 1e-4)
        soft = torch.sigmoid((flat - thresh) / temp)
        floor = max(0.0, min(1.0, float(mask_floor)))
        soft = floor + (1.0 - floor) * soft
        mask = soft.reshape_as(support).clamp(0.0, 1.0)
        hard_thresh = max(0.0, min(1.0, float(hard_mask_threshold)))
        if hard_mask:
            mask = (mask >= hard_thresh).to(dtype=support.dtype)
        if detach:
            mask = mask.detach()

        hard = (mask >= hard_thresh).float()
        stats = {
            "align_pattern_mode_is_pca": 1.0 if align_mode == "pca_prior" else 0.0,
            "align_pattern_mode_is_hybrid": 1.0 if align_mode == "hybrid" else 0.0,
            "align_pattern_mode_is_filtered": 1.0 if align_mode == "filtered_support" else 0.0,
            "align_pattern_mask_mean": float(mask.mean().item()),
            "align_pattern_mask_hard_coverage": float(hard.mean().item()),
            "align_pattern_support_mean": float(support.mean().item()),
            "align_pattern_support_max": float(support.max().item()),
            "align_pattern_hard_mask_enabled": 1.0 if hard_mask else 0.0,
            "align_pattern_hard_mask_threshold": hard_thresh,
        }
        if sim_support is not None:
            stats["align_pattern_similarity_mean"] = float(sim_support.mean().item())
        if dino_energy is not None:
            stats["align_pattern_dino_energy_mean"] = float(dino_energy.mean().item())
        if student_energy is not None:
            stats["align_pattern_student_energy_mean"] = float(student_energy.mean().item())
        if legacy_support is not None:
            stats["align_pattern_legacy_support_mean"] = float(legacy_support.mean().item())
        if pca_support is not None:
            stats["align_pattern_pca_support_mean"] = float(pca_support.mean().item())
            stats.update(pca_stats)
        elif align_mode == "filtered_support":
            stats["align_pattern_pca_disabled"] = 1.0
        if align_mode == "filtered_support":
            stats["align_pattern_filtered_prior_strength"] = float(filtered_prior_strength)
            stats["align_pattern_filtered_completion_strength"] = float(filtered_completion_strength)
            stats["align_pattern_filtered_noise_suppress"] = float(filtered_noise_suppress)
            stats["align_pattern_structural_prior_strength"] = float(structural_prior_strength)
            stats["align_pattern_structural_cross_row_strength"] = float(structural_cross_row_strength)
            stats.update(structural_stats)
        return mask.unsqueeze(1), stats

    def _compute_align_loss_map(
        self,
        projected_student_feat: torch.Tensor,
        dino_features: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if projected_student_feat.shape[2:] != dino_features.shape[2:]:
            projected_student_feat = F.interpolate(
                projected_student_feat,
                size=dino_features.shape[2:],
                mode="bilinear",
                align_corners=False,
            )

        if self.align_head is None:
            raise RuntimeError("Alignment head not initialised")

        if self.align_head.normalize_feature:
            student_n = F.normalize(projected_student_feat, p=2, dim=1)
            teacher_n = F.normalize(dino_features, p=2, dim=1)
            sim_map = (student_n * teacher_n).sum(dim=1)
            loss_map = 1.0 - sim_map
            return loss_map, sim_map

        diff = projected_student_feat - dino_features
        loss_map = torch.linalg.norm(diff, dim=1, ord=2) / 100.0
        sim_map = -loss_map
        return loss_map, sim_map

    def predict_stomata_prior(
        self,
        dino_features: Optional[torch.Tensor] = None,
        student_spatial_feat: Optional[torch.Tensor] = None,
        output_hw: Optional[Tuple[int, int]] = None,
    ) -> Optional[Dict[str, torch.Tensor]]:
        if not self.use_stomata_prior:
            return None

        if self.prior_head_mode == "origin":
            if dino_features is None or self.stomata_prior_head is None:
                return None
            logits = self.stomata_prior_head(dino_features)
        else:
            if student_spatial_feat is None:
                return None
            self._ensure_student_prior_head(
                student_spatial_feat.shape[1],
                student_spatial_feat.device,
            )
            assert self.student_prior_head is not None
            if hasattr(self.student_prior_head, "forward_components"):
                comp = self.student_prior_head.forward_components(student_spatial_feat)
                logits = comp["logits"]
            else:
                comp = {}
                logits = self.student_prior_head(student_spatial_feat)
            if output_hw is not None and logits.shape[2:] != output_hw:
                logits = F.interpolate(
                    logits,
                    size=output_hw,
                    mode="bilinear",
                    align_corners=False,
                )
                resized: Dict[str, torch.Tensor] = {}
                for key, value in comp.items():
                    if isinstance(value, torch.Tensor) and value.shape[2:] != output_hw:
                        resized[key] = F.interpolate(
                            value,
                            size=output_hw,
                            mode="bilinear",
                            align_corners=False,
                        )
                    else:
                        resized[key] = value
                comp = resized

        prob = torch.sigmoid(logits)
        out = {
            "logits": logits,
            "prob": prob,
        }
        for key, value in comp.items():
            if key == "logits":
                continue
            out[key] = value
        return out

    @torch.no_grad()
    def get_alignment_debug(
        self,
        images: torch.Tensor,
        teacher_images: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Return intermediate tensors for visualizing feature alignment.

        Outputs:
            ``student_spatial_feat``: raw student feature map at the configured
                pyramid level, shape ``[B, C_s, H_s, W_s]``.
            ``projected_student_feat``: student feature after alignment-head
                projection into DINO space, shape ``[B, C_t, H_t, W_t]``.
            ``dino_features``: frozen DINO teacher feature map,
                shape ``[B, C_t, H_t, W_t]``.
            ``similarity_map``: per-pixel cosine similarity map from
                ``align_loss(return_sim=True)``, shape ``[B, H_t, W_t]``.
            ``align_loss``: scalar alignment loss for the batch.
        """
        was_training = self.training
        self.eval()

        student_out = self.student(
            images,
            return_features=True,
            return_adapted_features=False,
        )

        feat_key = f"{self.student_align_layer}_features"
        student_spatial = student_out.get(feat_key)
        if student_spatial is None:
            raise RuntimeError(
                f"Student output does not contain {feat_key}; cannot inspect alignment."
            )

        self._ensure_align_head(student_spatial.shape[1], student_spatial.device)
        assert self.align_head is not None, "Alignment head not initialised yet"

        dino_input = teacher_images if teacher_images is not None else images
        dino_feat = self.dino_teacher(dino_input)
        projected = self.align_head(student_spatial, dino_feat.shape[2:])
        debug = self.compute_align_loss(
            student_spatial,
            dino_feat,
            return_details=True,
        )
        align_loss = debug["loss"]
        sim_map = debug["similarity_map"]
        prior = self.predict_stomata_prior(
            dino_features=dino_feat,
            student_spatial_feat=student_spatial,
            output_hw=dino_feat.shape[2:],
        )

        if was_training:
            self.train(True)
        else:
            self.eval()

        return {
            "student_spatial_feat": student_spatial.detach(),
            "projected_student_feat": projected.detach(),
            "dino_features": dino_feat.detach(),
            "similarity_map": sim_map.detach(),
            "align_loss": align_loss.detach(),
            "stomata_prior_prob": None if prior is None else prior["prob"].detach(),
            "stomata_prior_logits": None if prior is None else prior["logits"].detach(),
            "stomata_prior_base_prob": None if prior is None or not isinstance(prior.get("base_prob"), torch.Tensor) else prior["base_prob"].detach(),
            "stomata_prior_propagation_prob": None if prior is None or not isinstance(prior.get("propagation_prob"), torch.Tensor) else prior["propagation_prob"].detach(),
        }

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
        if self.stomata_prior_head is not None:
            params.extend(self.stomata_prior_head.parameters())
        if self.student_prior_head is not None:
            params.extend(self.student_prior_head.parameters())
        return params

    def get_fft_parameters(self) -> List[nn.Parameter]:
        # MTKD-side FFT injection disabled.
        return []

    def train(self, mode: bool = True):
        super().train(mode)
        # Keep frozen components in eval
        self.dino_teacher.train(False)
        if self.wheat_teacher is not None:
            self.wheat_teacher.train(False)
        if self.ensemble_teachers is not None:
            self.ensemble_teachers.train(False)
        if mode:
            self._set_frozen_student_modules_eval()
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
            "student_freeze_config": {"enabled": False, "trainable_layer_indices": [17, 18, 20, 21]},
            "dino_config": {"model_name": "vit_base", "patch_size": 16, "embed_dim": 768},
            "dino_checkpoint": None,
            "wheat_teacher_config": {"weights": "best.pt", "score_threshold": 0.3},
            "align_head_config": {"head_type": "MLP", "proj_dim": 1024, "normalize": True},
            "prior_head_config": {"enabled": True, "mode": "origin", "hidden_dim": 256},
            "student_align_layer": "p4",
            "fft_block_config": {"after_blocks": [10], "num_freq_bins": 32},
            "ensemble_config": {...},
            "teacher_specs": [...],
        }
    """
    return MTKDModelV2(
        student_config=config.get("student_config"),
        num_classes=config.get("num_classes", 1),
        student_freeze_config=config.get("student_freeze_config"),
        dino_config=config.get("dino_config"),
        dino_checkpoint=config.get("dino_checkpoint"),
        wheat_teacher_config=config.get("wheat_teacher_config"),
        align_head_config=config.get("align_head_config"),
        prior_head_config=config.get("prior_head_config"),
        student_align_layer=config.get("student_align_layer", "p4"),
        fft_block_config=config.get("fft_block_config"),
        ensemble_config=config.get("ensemble_config"),
        teacher_specs=config.get("teacher_specs"),
    )
