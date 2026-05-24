"""
MTKDTrainerV2 — training loop modelled after DINO Teacher
==========================================================

Reference:  ``DINO_Teacher/dinoteacher/engine/trainer.py``
            ───  class ``DINOTeacherTrainer(DefaultTrainer)``

Key parallels
-------------
* **2-stage training**:
     1. Source alignment — supervised detection + feature alignment
         (student ↔ DINO teacher) from epoch 0.
     2. Full — add target alignment and pseudo-label supervision after
         ``align_target_start_epoch``.

* **Pseudo-labels from pretrained wheat model** — supplied externally as
  YOLO-format ``.txt`` files (one file per image, each line
  ``class cx cy w h [conf]`` or OBB format). They can also be supplied as
  a ``.csv``.  No EMA teacher is needed because the user's pretrained
  YOLO wheat detector *is* the labeller.

* **Feature alignment** follows ``TeacherStudentAlignHead`` in
  ``engine/align_head.py`` — per-pixel spatial, **not** global-pool CLS.

* **Loss weighting** mirrors DINO Teacher: pseudo bbox-regression loss is
  zeroed; pseudo classification loss weighted by ``unsup_loss_weight``.

* **Detection loss** — uses ``v8DetectionLoss`` from Ultralytics (CIoU +
  BCE + DFL), invoked via ``YOLOStudentDetector.compute_loss``.
"""

from __future__ import annotations

import copy
from contextlib import contextmanager
import json
import math
import logging
import os
import shutil
import sys
import time
from collections import OrderedDict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont
from torch.utils.data import DataLoader

import matplotlib

from .models.mtkd_model_v2 import MTKDModelV2, build_mtkd_model_v2
from .engine.pseudo_labels import (
    load_pseudo_labels_dir,
    load_pseudo_labels_csv,
    build_yolo_batch_from_pseudo,
    targets_to_yolo_batch,
)
from .losses.prediction_alignment import UltralyticsCriterionAlignmentLoss
from .utils import (
    AverageMeterDict,
    EarlyStopping,
    GradientClipper,
    ModelEMA,
    format_time,
    save_checkpoint,
    load_checkpoint,
    save_config,
    seed_everything,
    count_parameters,
    setup_logger,
)

logger = logging.getLogger(__name__)


# ======================================================================
# Default config
# ======================================================================

def get_default_config_v2() -> Dict[str, Any]:
    """
    Default configuration for ``MTKDTrainerV2``.

    Adds DINO-Teacher-style keys that the v1 config was missing:
    ``align_head_config``, ``align_target_start``,
    ``unsup_loss_weight``, etc.
    """
    return {
        # ---- model ----
        "model": {
            "num_classes": 3,  # barley dataset merged to 3 classes
            "student_config": {
                "student_type": "yolo",
                "weights": "yolo12s.pt",
                "feature_level": "p4",
            },
            "student_freeze_config": {
                "enabled": False,
                "trainable_layer_indices": [17, 18, 20, 21],
                "trainable_name_keywords": [],
            },
            "dino_config": {
                "model_name": "vit_base",
                "patch_size": 16,
                "embed_dim": 768,
                "normalize_feature": True,
            },
            "dino_checkpoint": None,
            "wheat_teacher_config": None,
            "align_head_config": {
                "head_type": "MLP",           # "attention" / "MLP" / "MLP3" / "linear"
                "proj_dim": 1024,
                "normalize": True,
                "use_gelu": False,
            },
            "prior_head_config": {
                "enabled": True,
                "mode": "origin",
                "hidden_dim": 256,
                "use_gelu": True,
                "detach_for_align": True,
                "apply_to_detection": False,
                "gate_strength": 1.0,
                "gate_hard": False,
                "gate_threshold": 0.5,
            },
            "student_align_layer": "p4",      # which YOLO pyramid level to align
            # MTKD no longer injects FFT blocks into DINO.
            "fft_block_config": None,
            # Ensemble teachers (optional)
            "ensemble_config": None,
            "teacher_specs": None,
        },
        # ---- training ----
        "training": {
            "epochs": 100,
            "batch_size": 8,
            "batch_size_label": None,
            "batch_size_unlabel": None,
            "num_workers": 4,
            "learning_rate": 1e-4,
            "weight_decay": 1e-4,
            "lr_scheduler": "cosine",
            "warmup_epochs": 5,
            "min_lr": 1e-6,
            "gradient_clip_max_norm": 1.0,
            "accumulation_steps": 1,
            "mixed_precision": True,
            "early_stopping_patience": 20,
            "ema_enabled": True,
            "ema_decay": 0.9999,
            "ema_tau": 0.0,
            # ---- 2-stage schedule ----
            # Kept for backward compatibility only; ignored by trainer.
            "burn_up_epochs": 0,
            "align_target_start_epoch": 10,    # add target alignment + pseudo-labels after this
            # ---- loss weights (mirrors DINO Teacher) ----
            "feature_align_loss_weight": 1.0,
            "feature_align_loss_weight_target": 0.2,
            "use_source_alignment": True,
            "prior_mask_loss_weight": 1.0,
            "prior_mask_loss_weight_target": 1.0,
            "prior_mask_positive_weight": 4.0,
            "prior_propagation_loss_weight": 0.35,
            "prior_periodic_loss_weight": 0.40,
            "prior_periodic_row_weight": 0.35,
            "prior_periodic_col_weight": 1.00,
            # In single-dataset MTKD (no explicit source/target split), a
            # second target alignment term duplicates pressure on the same
            # mini-batch. Keep it disabled unless separate source/target data
            # streams are explicitly provided.
            "use_target_alignment": False,
            "separate_source_target_data": False,
            "unsup_loss_weight": 4.0,
            # prediction alignment mode:
            # - ultralytics: use Ultralytics criterion/assigner path (recommended)
            # - legacy: direct pseudo criterion call in trainer (backwards-compat)
            "prediction_align_mode": "ultralytics",
            "zero_pseudo_box_reg": False,      # default: keep pseudo bbox/dfl regression enabled
            "supervision_mode": "gt+pseudo",  # gt+pseudo | gt-only | pseudo-only
            # ALIGN_EASY_ONLY — When True, the DINO teacher receives *only*
            # the original (non-augmented) images for alignment, preventing
            # augmentation artefacts from polluting the teacher signal.
            # Requires the dataset to supply ``batch["images_weak"]`` (a
            # second, unaugmented copy).  If False or the key is absent,
            # the same augmented images are sent to both student and teacher.
            "align_easy_only": False,
            # Pattern-guided alignment mask (in DINO patch space).
            "pattern_align_enabled": True,
            "pattern_align_on_target": True,
            "pattern_align_target_coverage": 0.12,
            "pattern_align_min_coverage": 0.02,
            "pattern_align_max_coverage": 0.30,
            "pattern_align_mode": "legacy",
            "pattern_align_pca_components": 3,
            "pattern_align_hybrid_pca_weight": 0.5,
            "pattern_align_filtered_prior_strength": 0.75,
            "pattern_align_filtered_completion_strength": 0.65,
            "pattern_align_filtered_completion_gamma": 1.0,
            "pattern_align_filtered_noise_suppress": 0.25,
            "pattern_align_sim_weight": 0.55,
            "pattern_align_dino_weight": 0.30,
            "pattern_align_student_weight": 0.15,
            "pattern_align_species_prior_bank": None,
            "pattern_align_source_prior_name": None,
            "pattern_align_target_prior_name": None,
            "pattern_align_structural_prior_strength": 0.0,
            "pattern_align_structural_cross_row_strength": 0.0,
            "pattern_align_structural_seed_threshold": 0.55,
            "pattern_align_structural_min_row_seeds": 2,
            "pattern_align_temperature": 0.08,
            "pattern_align_mask_floor": 0.10,
            "pattern_align_hard_mask": False,
            "pattern_align_hard_mask_threshold": 0.5,
            "pattern_align_detach_mask": True,
            "pattern_align_use_dino_bypass": False,
            "student_gate_distill_enabled": False,
            "student_gate_distill_weight": 0.10,
            "student_gate_distill_bce_weight": 0.70,
            "student_gate_distill_mse_weight": 0.30,
            "student_gate_distill_positive_weight": 3.0,
            "pseudo_hard_mask_gate_enabled": False,
            "weight_anchor_enabled": False,
            "weight_anchor_lambda": 0.0,
        },
        # ---- pseudo labels ----
        "pseudo_labels": {
            # Pseudo source mode: auto / offline / online / none
            # - auto: prefer offline files, fallback to online wheat teacher
            # - offline: require pseudo files only
            # - online: require wheat teacher only
            # - none: disable pseudo labels entirely
            "mode": "auto",
            # Directory of .txt files (YOLO format, one per image)
            "label_dir": None,
            # OR path to a .csv
            "csv_path": None,
            # Extra MTKD-side filtering. ``None`` preserves the native
            # Ultralytics thresholds from the source model / saved labels.
            "score_threshold": None,
            # Optional per-class confidence overrides for pseudo filtering.
            # Keep ``None`` to avoid custom MTKD-side thresholds.
            "class_score_thresholds": None,
            # Optional area-guided class prior for online teacher pseudo labels.
            # Boxes are in normalized xywh/xywhr space, so area means ``w * h``.
            # Example:
            #   {
            #       "source_class": 0,
            #       "target_class": 1,
            #       "max_area": 0.001,
            #   }
            "area_class_prior": None,
            # Auto-convert OBB 8-coord to axis-aligned bbox
            "convert_obb": True,
        },
        # ---- data ----
        "data": {
            "dataset_root": "Stomata_Dataset",
            "image_subdir": "barley_category/barley_image_fresh-leaf",
            "label_subdir": "barley_category/barley_label_fresh-leaf",
            "unlabeled_dataset_root": None,
            "unlabeled_image_subdir": None,
            "unlabeled_label_subdir": None,
            "val_ratio": 0.1,
            "image_size": 640,
            "augmentation": True,
        },
        # ---- output ----
        "output": {
            "save_dir": "outputs/mtkd_v2",
            "save_freq": 5,
            "save_pth_checkpoints": True,
            "log_freq": 10,
            # Best model selection metric:
            # - loss: lower is better
            # - map50: higher is better
            # - map5095: higher is better
            # - fitness: higher is better (default), Ultralytics-style
            #            fitness = 0.1*map50 + 0.9*map50-95
            "best_by": "fitness",
            # mAP selection settings (used only when best_by != loss)
            "map_data": None,
            "map_split": "val",
            "map_imgsz": 640,
            "map_batch": 16,
            # ``None`` lets Ultralytics val() use its native defaults
            # (conf=0.001, iou=0.7).
            "map_conf": None,
            "map_iou": None,
            "map_eval_interval": 1,
            "debug_export_interval": 0,
            "debug_tile_size": 320,
            "debug_pred_conf": 0.10,
            "debug_pred_max_boxes": 300,
        },
        # ---- misc ----
        "seed": 42,
        "device": "cuda" if torch.cuda.is_available() else "cpu",
    }


def _candidate_component_score_stems(image_path: str) -> List[str]:
    stem = Path(image_path).stem
    candidates: List[str] = [stem]
    if stem.startswith("result_"):
        candidates.append(stem[len("result_") :])
    candidates.append(stem.removeprefix("result_"))
    unique: List[str] = []
    seen = set()
    for item in candidates:
        if item and item not in seen:
            seen.add(item)
            unique.append(item)
    return unique


def _estimate_spacing_px(xs: np.ndarray) -> float:
    xs = np.sort(xs.astype(np.float32))
    if xs.size < 2:
        return 0.0
    diffs = np.diff(xs)
    diffs = diffs[diffs > 1.0]
    if diffs.size == 0:
        return 0.0
    rough = float(np.median(diffs))
    keep = diffs[(diffs >= 0.45 * rough) & (diffs <= 1.65 * rough)]
    if keep.size:
        return float(np.median(keep))
    return rough


def _load_image_prior_index(path: Optional[Path]) -> Dict[str, Dict[str, object]]:
    if path is None or not path.is_file():
        return {}
    try:
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        return {}
    images = data.get("images", [])
    if not isinstance(images, list):
        return {}
    index: Dict[str, Dict[str, object]] = {}
    for item in images:
        if not isinstance(item, dict):
            continue
        image_name = str(item.get("image") or Path(str(item.get("annotated_image", ""))).name)
        if not image_name:
            continue
        for stem in _candidate_component_score_stems(image_name):
            index[stem] = item
    return index


def _load_species_prior_bank(path: Optional[Path]) -> Dict[str, Dict[str, float]]:
    if path is None or not path.is_file():
        return {}
    try:
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        return {}
    priors = data.get("priors", [])
    if not isinstance(priors, list):
        return {}
    bank: Dict[str, Dict[str, float]] = {}
    for item in priors:
        if not isinstance(item, dict):
            continue
        name = str(item.get("name", "")).strip()
        if not name:
            continue
        bank[name] = {
            "image_width": float(item.get("image_width", 0.0) or 0.0),
            "image_height": float(item.get("image_height", 0.0) or 0.0),
            "x_period_px": float(item.get("x_period_px", 0.0) or 0.0),
            "row_period_px": float(item.get("row_period_px", 0.0) or 0.0),
        }
    return bank


def _infer_species_prior_name(
    image_path: str,
    *,
    image_size: tuple[int, int],
    bank: Dict[str, Dict[str, float]],
) -> Optional[str]:
    if not bank:
        return None
    text = image_path.lower()
    stem = Path(image_path).stem.lower()
    if "rice" in text or stem.startswith("202112"):
        return "rice_annotate" if "rice_annotate" in bank else None
    if "wheat" in text or "tl" in stem:
        return "wheat10_label" if "wheat10_label" in bank else None
    if "barley" in text or stem.startswith("202408"):
        return "barley20_label" if "barley20_label" in bank else None

    img_w, img_h = float(image_size[0]), float(image_size[1])
    best_name: Optional[str] = None
    best_score = float("inf")
    for name, prior in bank.items():
        dw = abs(float(prior.get("image_width", 0.0)) - img_w)
        dh = abs(float(prior.get("image_height", 0.0)) - img_h)
        score = dw + dh
        if score < best_score:
            best_score = score
            best_name = name
    return best_name


def _resolve_external_pattern_prior(
    image_path: str,
    *,
    image_size: tuple[int, int],
    image_prior_index: Dict[str, Dict[str, object]],
    species_prior_bank: Dict[str, Dict[str, float]],
) -> Dict[str, object]:
    resolved: Dict[str, object] = {
        "source": "none",
        "family": "",
        "x_period_px": 0.0,
        "row_period_px": 0.0,
        "row_tolerance_px": 0.0,
        "row_centers_px": [],
    }
    for stem in _candidate_component_score_stems(image_path):
        item = image_prior_index.get(stem)
        if not isinstance(item, dict):
            continue
        rows = item.get("rows", [])
        if not isinstance(rows, list):
            rows = []
        good_rows = [
            row for row in rows
            if isinstance(row, dict) and float(row.get("count", 0.0) or 0.0) >= 3.0
        ]
        periods = [
            float(row.get("x_period_median_gap", 0.0) or 0.0)
            for row in good_rows
            if float(row.get("x_period_median_gap", 0.0) or 0.0) > 3.0
        ]
        row_centers = [
            float(row.get("row_y", 0.0) or 0.0)
            for row in rows
            if isinstance(row, dict) and float(row.get("count", 0.0) or 0.0) >= 2.0
        ]
        row_centers = sorted(y for y in row_centers if y >= 0.0)
        row_gap = float(item.get("row_gap_median", 0.0) or 0.0)
        resolved.update(
            {
                "source": "image_priors",
                "family": "rice_annotate",
                "x_period_px": float(np.median(periods)) if periods else 0.0,
                "row_period_px": row_gap if row_gap > 0.0 else _estimate_spacing_px(np.asarray(row_centers, dtype=np.float32)),
                "row_tolerance_px": float(item.get("row_tolerance_px", 0.0) or 0.0),
                "row_centers_px": row_centers,
            }
        )
        return resolved

    prior_name = _infer_species_prior_name(image_path, image_size=image_size, bank=species_prior_bank)
    if prior_name and prior_name in species_prior_bank:
        prior = species_prior_bank[prior_name]
        resolved.update(
            {
                "source": "species_prior_bank",
                "family": prior_name,
                "x_period_px": float(prior.get("x_period_px", 0.0) or 0.0),
                "row_period_px": float(prior.get("row_period_px", 0.0) or 0.0),
                "row_tolerance_px": max(float(prior.get("row_period_px", 0.0) or 0.0) * 0.22, 18.0),
                "row_centers_px": [],
            }
        )
    return resolved


# ======================================================================
# MTKDTrainerV2
# ======================================================================

class MTKDTrainerV2:
    """
    Trainer modelled after ``DINOTeacherTrainer``.

     Stages (epoch-based rather than iter-based for simplicity)
     -----------------------------------------------------------
     1. ``epoch < align_target_start_epoch``
         → Supervised loss  **+**  source feature alignment (student ↔ DINO).

     2. ``epoch >= align_target_start_epoch``
         → Supervised  +  source align  +  target align  +  pseudo-label loss.
    """

    def __init__(
        self,
        config: Dict[str, Any],
        model: Optional[MTKDModelV2] = None,
        train_loader: Optional[DataLoader] = None,
        val_loader: Optional[DataLoader] = None,
    ):
        self.config = config
        self.device = torch.device(config.get("device", "cuda"))

        os.makedirs(config["output"]["save_dir"], exist_ok=True)
        log_file = os.path.join(config["output"]["save_dir"], "training.log")
        self.logger = setup_logger("mtkd_trainer_v2", log_file=log_file)

        seed_everything(config.get("seed", 42))
        save_config(config, os.path.join(config["output"]["save_dir"], "config.json"))

        # ---- model ----
        if model is not None:
            self.model = model
        else:
            self.model = build_mtkd_model_v2(config["model"])
        self.model = self.model.to(self.device)

        total_p = count_parameters(self.model)
        train_p = count_parameters(self.model, trainable_only=True)
        self.logger.info(f"Total params: {total_p:,}  Trainable: {train_p:,}")

        # Student task metadata (detect vs obb)
        student = getattr(self.model, "student", None)
        self.student_task = str(getattr(student, "task", "detect")).lower()
        self.student_box_dim = int(getattr(student, "box_dim", 4))
        self.logger.info(
            f"Student task: {self.student_task} | target box dim: {self.student_box_dim}"
        )
        self.num_classes = int(config.get("model", {}).get("num_classes", 1))
        self.weight_anchor_enabled = False
        self.weight_anchor_lambda = 0.0
        self._student_anchor_params: Dict[str, torch.Tensor] = {}

        # ---- data ----
        self.train_loader = train_loader
        self.val_loader = val_loader

        # ---- optimiser ----
        self._setup_optimizer()
        self._setup_scheduler()

        # ---- AMP ----
        self.scaler = None
        tc = config["training"]
        self.weight_anchor_enabled = bool(tc.get("weight_anchor_enabled", False))
        self.weight_anchor_lambda = float(tc.get("weight_anchor_lambda", 0.0))
        self.dual_stream = bool(tc.get("separate_source_target_data", False))
        if self.dual_stream:
            self.logger.info(
                "Dual-stream mode enabled (labeled + unlabeled per training step)"
            )
        if tc.get("mixed_precision") and self.device.type == "cuda":
            self.scaler = torch.amp.GradScaler("cuda")

        self.gradient_clipper = GradientClipper(
            max_norm=tc.get("gradient_clip_max_norm", 1.0)
        )
        self.early_stopping = EarlyStopping(
            patience=tc.get("early_stopping_patience", 20), mode="min",
        )

        # ---- stage threshold ----
        deprecated_burn_up_epochs = int(tc.get("burn_up_epochs", 0) or 0)
        self.align_target_start_epoch = max(0, int(tc.get("align_target_start_epoch", 10)))
        if deprecated_burn_up_epochs > 0:
            self.logger.warning(
                "training.burn_up_epochs=%d is deprecated and ignored in two-stage mode; "
                "source alignment starts at epoch 0.",
                deprecated_burn_up_epochs,
            )

        # ---- loss weights ----
        self.feature_align_w = tc.get("feature_align_loss_weight", 0.0)
        self.feature_align_w_target = tc.get("feature_align_loss_weight_target", 0.0)
        self.use_source_alignment = bool(tc.get("use_source_alignment", True))
        self.prior_mask_loss_w = float(tc.get("prior_mask_loss_weight", 1.0))
        self.prior_mask_loss_w_target = float(tc.get("prior_mask_loss_weight_target", 1.0))
        self.prior_mask_pos_w = float(tc.get("prior_mask_positive_weight", 4.0))
        self.prior_propagation_loss_w = float(tc.get("prior_propagation_loss_weight", 0.35))
        self.prior_periodic_loss_w = float(tc.get("prior_periodic_loss_weight", 0.40))
        self.prior_periodic_row_w = float(tc.get("prior_periodic_row_weight", 0.35))
        self.prior_periodic_col_w = float(tc.get("prior_periodic_col_weight", 1.00))
        self.use_target_alignment = bool(tc.get("use_target_alignment", False))
        self.separate_source_target_data = bool(tc.get("separate_source_target_data", False))
        self.unsup_loss_w = tc.get("unsup_loss_weight", 4.0)
        self.prediction_align_mode = str(tc.get("prediction_align_mode", "ultralytics")).lower()
        if self.prediction_align_mode not in {"ultralytics", "legacy"}:
            raise ValueError(
                "training.prediction_align_mode must be one of: ultralytics / legacy"
            )
        self.pred_align_loss = UltralyticsCriterionAlignmentLoss(
            unsup_weight=self.unsup_loss_w,
            zero_box_dfl=tc.get("zero_pseudo_box_reg", False),
        )
        self.zero_pseudo_box_reg = tc.get("zero_pseudo_box_reg", False)
        self.align_easy_only = tc.get("align_easy_only", False)
        self.pattern_align_enabled = bool(tc.get("pattern_align_enabled", True))
        self.pattern_align_on_target = bool(tc.get("pattern_align_on_target", True))
        self.pattern_align_target_coverage = float(tc.get("pattern_align_target_coverage", 0.12))
        self.pattern_align_min_coverage = float(tc.get("pattern_align_min_coverage", 0.02))
        self.pattern_align_max_coverage = float(tc.get("pattern_align_max_coverage", 0.30))
        self.pattern_align_mode = str(tc.get("pattern_align_mode", "legacy")).strip().lower()
        if self.pattern_align_mode not in {"legacy", "pca_prior", "hybrid", "filtered_support"}:
            raise ValueError(
                "training.pattern_align_mode must be one of: legacy / pca_prior / hybrid / filtered_support"
            )
        self.pattern_align_pca_components = int(tc.get("pattern_align_pca_components", 3))
        self.pattern_align_hybrid_pca_weight = float(tc.get("pattern_align_hybrid_pca_weight", 0.5))
        self.pattern_align_filtered_prior_strength = float(tc.get("pattern_align_filtered_prior_strength", 0.75))
        self.pattern_align_filtered_completion_strength = float(tc.get("pattern_align_filtered_completion_strength", 0.65))
        self.pattern_align_filtered_completion_gamma = float(tc.get("pattern_align_filtered_completion_gamma", 1.0))
        self.pattern_align_filtered_noise_suppress = float(tc.get("pattern_align_filtered_noise_suppress", 0.25))
        self.pattern_align_sim_weight = float(tc.get("pattern_align_sim_weight", 0.55))
        self.pattern_align_dino_weight = float(tc.get("pattern_align_dino_weight", 0.30))
        self.pattern_align_student_weight = float(tc.get("pattern_align_student_weight", 0.15))
        self.pattern_align_species_prior_bank = tc.get("pattern_align_species_prior_bank", None)
        self.pattern_align_source_prior_name = tc.get("pattern_align_source_prior_name", None)
        self.pattern_align_target_prior_name = tc.get("pattern_align_target_prior_name", None)
        self.pattern_align_structural_prior_strength = float(tc.get("pattern_align_structural_prior_strength", 0.0))
        self.pattern_align_structural_cross_row_strength = float(tc.get("pattern_align_structural_cross_row_strength", 0.0))
        self.pattern_align_structural_seed_threshold = float(tc.get("pattern_align_structural_seed_threshold", 0.55))
        self.pattern_align_structural_min_row_seeds = int(tc.get("pattern_align_structural_min_row_seeds", 2))
        self.pattern_align_temperature = float(tc.get("pattern_align_temperature", 0.08))
        self.pattern_align_mask_floor = float(tc.get("pattern_align_mask_floor", 0.10))
        self.pattern_align_hard_mask = bool(tc.get("pattern_align_hard_mask", False))
        self.pattern_align_hard_mask_threshold = float(tc.get("pattern_align_hard_mask_threshold", 0.5))
        self.pattern_align_detach_mask = bool(tc.get("pattern_align_detach_mask", True))
        self.pattern_align_use_dino_bypass = bool(tc.get("pattern_align_use_dino_bypass", False))
        self.student_gate_distill_enabled = bool(tc.get("student_gate_distill_enabled", False))
        self.student_gate_distill_w = float(tc.get("student_gate_distill_weight", 0.10))
        self.student_gate_distill_bce_w = float(tc.get("student_gate_distill_bce_weight", 0.70))
        self.student_gate_distill_mse_w = float(tc.get("student_gate_distill_mse_weight", 0.30))
        self.student_gate_distill_pos_w = float(tc.get("student_gate_distill_positive_weight", 3.0))
        self.pseudo_hard_mask_gate_enabled = bool(tc.get("pseudo_hard_mask_gate_enabled", False))
        self.pattern_align_source_prior_cfg = self._resolve_pattern_spacing_prior(self.pattern_align_source_prior_name)
        self.pattern_align_target_prior_cfg = self._resolve_pattern_spacing_prior(self.pattern_align_target_prior_name)
        self._dino_bypass_cfg = dict(self.config.get("dino_bypass", {}) or {})
        self._dino_bypass_image_prior_index: Dict[str, Dict[str, object]] = {}
        self._dino_bypass_species_prior_bank: Dict[str, Dict[str, float]] = {}
        self._dino_bypass_period_prior_px = float(self._dino_bypass_cfg.get("pattern_period_prior_px", 0.0) or 0.0)
        self._dino_bypass_row_period_prior_px = float(self._dino_bypass_cfg.get("pattern_row_period_prior_px", 0.0) or 0.0)
        self._dino_bypass_row_tolerance_override_px = float(
            self._dino_bypass_cfg.get("pattern_row_tolerance_override_px", 0.0) or 0.0
        )
        self._dino_bypass_row_tolerance_scale = float(
            self._dino_bypass_cfg.get("pattern_row_tolerance_scale", 0.0) or 0.0
        )
        if self.pattern_align_use_dino_bypass:
            prior_path = self._dino_bypass_cfg.get("pattern_image_priors")
            if prior_path:
                self._dino_bypass_image_prior_index = _load_image_prior_index(
                    Path(str(prior_path)).expanduser().resolve()
                )
            species_path = self._dino_bypass_cfg.get("pattern_species_prior_bank") or self.pattern_align_species_prior_bank
            if species_path:
                self._dino_bypass_species_prior_bank = _load_species_prior_bank(
                    Path(str(species_path)).expanduser().resolve()
                )
            if not self._dino_bypass_image_prior_index and not self._dino_bypass_species_prior_bank:
                self.logger.warning(
                    "pattern_align_use_dino_bypass enabled but no image/species prior bank found; using static ratios"
                )
        if self.use_target_alignment and not self.separate_source_target_data:
            self.logger.warning(
                "use_target_alignment=True but separate_source_target_data=False; "
                "disable target alignment to avoid duplicate pressure on the same batch."
            )
            self.use_target_alignment = False
        self.logger.info(
            "Source alignment term: %s",
            "enabled" if self.use_source_alignment else "disabled",
        )
        self.logger.info(
            "Target alignment term: %s",
            "enabled" if self.use_target_alignment else "disabled",
        )
        prior_mode = str(getattr(self.model, "prior_head_mode", "origin"))
        if (
            not self.use_source_alignment
            and getattr(self.model, "use_stomata_prior", False)
            and prior_mode == "origin"
        ):
            self.logger.info(
                "Stomata prior head is source-align-only in the current implementation; "
                "it will stay inactive while source alignment is disabled."
            )
        elif getattr(self.model, "use_stomata_prior", False):
            self.logger.info("Stomata prior head mode: %s", prior_mode)
        if not self.use_source_alignment and not self.use_target_alignment:
            self.logger.warning(
                "Both source and target alignment are disabled; training will run without feature alignment."
            )
        if self.zero_pseudo_box_reg:
            self.logger.info("Pseudo box regression: disabled (cls-only pseudo supervision)")
        else:
            self.logger.info("Pseudo box regression: enabled (Ultralytics IoU/probIoU + DFL)")
        if self.prediction_align_mode == "ultralytics":
            self.logger.info("Prediction alignment: Ultralytics criterion/assigner path")
        else:
            self.logger.info("Prediction alignment: legacy direct pseudo criterion path")
        self.logger.info(
            "Pattern-guided alignment mask: %s (target=%s, mode=%s, cov=%.3f, floor=%.3f)",
            "enabled" if self.pattern_align_enabled else "disabled",
            "enabled" if self.pattern_align_on_target else "disabled",
            self.pattern_align_mode,
            self.pattern_align_target_coverage,
            self.pattern_align_mask_floor,
        )
        if self.pattern_align_use_dino_bypass:
            self.logger.info(
                "Pattern align uses dino_bypass priors (image_priors=%s, species_prior=%s)",
                "yes" if self._dino_bypass_image_prior_index else "no",
                "yes" if self._dino_bypass_species_prior_bank else "no",
            )
        if self.pseudo_hard_mask_gate_enabled:
            self.logger.info(
                "Pseudo hard-mask gate: enabled (mode=%s, threshold=%.3f)",
                self.pattern_align_mode,
                self.pattern_align_hard_mask_threshold,
            )
        if self.student_gate_distill_enabled:
            self.logger.info(
                "Student support-gate distillation: enabled weight=%.4f bce=%.2f mse=%.2f pos_weight=%.2f",
                self.student_gate_distill_w,
                self.student_gate_distill_bce_w,
                self.student_gate_distill_mse_w,
                self.student_gate_distill_pos_w,
            )
        self.supervision_mode = str(tc.get("supervision_mode", "gt+pseudo")).lower()
        if self.supervision_mode not in {"gt+pseudo", "gt-only", "pseudo-only"}:
            raise ValueError(
                f"Invalid training.supervision_mode={self.supervision_mode}. "
                "Choose one of: gt+pseudo / gt-only / pseudo-only"
            )
        self.use_gt_supervision = self.supervision_mode in {"gt+pseudo", "gt-only"}
        self.use_pseudo_supervision = self.supervision_mode in {"gt+pseudo", "pseudo-only"}
        self.enable_early_stopping = self.supervision_mode != "pseudo-only"

        self.logger.info(
            f"Supervision mode: {self.supervision_mode} "
            f"(use_gt={self.use_gt_supervision}, use_pseudo={self.use_pseudo_supervision})"
        )
        if self.weight_anchor_enabled and self.weight_anchor_lambda > 0.0:
            self._student_anchor_params = {
                name: param.detach().clone()
                for name, param in self.model.student.named_parameters()
                if param.requires_grad
            }
            self.logger.info(
                "Weight anchoring (L2-SP): enabled lambda=%.6f on %d student tensors",
                self.weight_anchor_lambda,
                len(self._student_anchor_params),
            )
        else:
            self.weight_anchor_enabled = False
        if self.supervision_mode == "pseudo-only":
            self.logger.info(
                "Pseudo supervision starts at epoch %d",
                int(self.align_target_start_epoch),
            )
        if not self.enable_early_stopping:
            self.logger.info("Early stopping disabled in pseudo-only mode")
        
        if tc.get("separation_loss_weight", 0.0) > 0:
            self.logger.warning(
                "training.separation_loss_weight is ignored: Valley separation loss has been removed from MTKD."
            )

        # ---- pseudo labels ----
        self.pseudo_labels: Optional[Dict] = None
        pl_cfg = config.get("pseudo_labels", {})
        self.pseudo_mode = str(pl_cfg.get("mode", "auto")).lower()
        if self.pseudo_mode not in {"auto", "offline", "online", "none"}:
            raise ValueError(
                f"Invalid pseudo_labels.mode={self.pseudo_mode}. "
                "Choose one of: auto/offline/online/none"
            )
        self.pl_score_threshold = self._parse_optional_threshold(
            pl_cfg.get("score_threshold", None),
            name="pseudo_labels.score_threshold",
        )
        self.pl_class_score_thresholds = self._parse_class_score_thresholds(
            pl_cfg.get("class_score_thresholds", None),
            num_classes=self.num_classes,
        )
        self.pl_area_class_prior = self._parse_area_class_prior(
            pl_cfg.get("area_class_prior", None),
            num_classes=self.num_classes,
        )
        self._last_area_prior_relabel_count = 0
        if self.pl_class_score_thresholds:
            cls_parts = ", ".join(
                f"{cid}:{thr:.3f}"
                for cid, thr in sorted(self.pl_class_score_thresholds.items())
            )
            self.logger.info(
                "Pseudo class score thresholds: %s (global default=%s)",
                cls_parts,
                (
                    f"{float(self.pl_score_threshold):.3f}"
                    if self.pl_score_threshold is not None
                    else "Ultralytics-native"
                ),
            )
        elif self.pl_score_threshold is None:
            self.logger.info("Pseudo score filtering: Ultralytics-native only (no MTKD extra threshold)")
        else:
            self.logger.info(
                "Pseudo score threshold: %.3f",
                float(self.pl_score_threshold),
            )
        if self.pl_area_class_prior:
            source_class = self.pl_area_class_prior.get("source_class", None)
            source_str = "any" if source_class is None else str(int(source_class))
            area_bits: List[str] = []
            min_area = self.pl_area_class_prior.get("min_area", None)
            max_area = self.pl_area_class_prior.get("max_area", None)
            if min_area is not None:
                area_bits.append(f"min_area={float(min_area):.6f}")
            if max_area is not None:
                area_bits.append(f"max_area={float(max_area):.6f}")
            self.logger.info(
                "Pseudo area class prior: source=%s -> target=%d (%s)",
                source_str,
                int(self.pl_area_class_prior["target_class"]),
                ", ".join(area_bits) if area_bits else "no area bounds",
            )
        convert_obb = pl_cfg.get("convert_obb", True)

        pl_dir = pl_cfg.get("label_dir")
        pl_csv = pl_cfg.get("csv_path")
        if pl_dir and os.path.isdir(pl_dir):
            self.pseudo_labels = load_pseudo_labels_dir(
                pl_dir,
                score_threshold=self.pl_score_threshold,
                convert_obb=convert_obb,
                target_box_format=("xywhr" if self.student_box_dim == 5 else "xywh"),
            )
        elif pl_csv and os.path.isfile(pl_csv):
            self.pseudo_labels = load_pseudo_labels_csv(
                pl_csv, score_threshold=self.pl_score_threshold,
            )

        has_wheat_teacher = getattr(self.model, "wheat_teacher", None) is not None
        has_offline_pseudo = self.pseudo_labels is not None

        self.use_online_teacher_pseudo = False
        self.use_offline_pseudo = False

        if self.pseudo_mode == "none":
            pass
        elif self.pseudo_mode == "offline":
            if not has_offline_pseudo:
                raise ValueError(
                    "pseudo_labels.mode=offline but no offline pseudo labels were loaded. "
                    "Set pseudo_label_dir/pseudo_csv correctly."
                )
            self.use_offline_pseudo = True
        elif self.pseudo_mode == "online":
            if not has_wheat_teacher:
                raise ValueError(
                    "pseudo_labels.mode=online but wheat teacher is not configured. "
                    "Provide model.wheat_teacher_config or --wheat-teacher-weights."
                )
            self.use_online_teacher_pseudo = True
        else:
            # auto
            if has_offline_pseudo:
                self.use_offline_pseudo = True
            elif has_wheat_teacher:
                self.use_online_teacher_pseudo = True

        if self.use_offline_pseudo:
            self.logger.info("Pseudo source: offline files")
            if self.config.get("data", {}).get("augmentation", True):
                self.logger.warning(
                    "Offline pseudo labels + enabled augmentation can cause geometric misalignment "
                    "between pseudo boxes and student images. Consider --no-augmentation or online pseudo mode."
                )
        elif self.use_online_teacher_pseudo:
            self.logger.info("Pseudo source: online frozen wheat teacher")
        else:
            self.logger.info("Pseudo source: disabled")

        # ---- training state ----
        self.start_epoch = 0
        self.best_loss = float("inf")
        self.student_ema: Optional[ModelEMA] = None
        self._ema_target = self._resolve_ema_target()
        self.ema_enabled = bool(tc.get("ema_enabled", True))
        self.ema_decay = float(tc.get("ema_decay", 0.9999))
        self.ema_tau = float(tc.get("ema_tau", 2000.0))
        if self.ema_enabled and self._ema_target is not None:
            self.student_ema = ModelEMA(
                self._ema_target,
                decay=self.ema_decay,
                tau=self.ema_tau,
            )
            self.logger.info(
                "Student EMA: enabled (decay=%.5f, tau=%.1f)",
                self.ema_decay,
                self.ema_tau,
            )
        elif self.ema_enabled:
            self.logger.warning("Student EMA requested, but no EMA-compatible student module was found")
        else:
            self.logger.info("Student EMA: disabled")

        # ---- best model selection policy ----
        oc = self.config.get("output", {})
        self.best_by = str(oc.get("best_by", "loss")).lower()
        if self.best_by not in {"loss", "map50", "map5095", "fitness"}:
            raise ValueError("output.best_by must be one of: loss / map50 / map5095 / fitness")
        self._best_higher_is_better = self.best_by in {"map50", "map5095", "fitness"}
        self.best_score = float("-inf") if self._best_higher_is_better else float("inf")

        self.map_data = oc.get("map_data")
        self.map_split = str(oc.get("map_split", "val"))
        self.map_imgsz = int(oc.get("map_imgsz", 640))
        self.map_batch = int(oc.get("map_batch", 16))
        self.map_conf = self._parse_optional_threshold(
            oc.get("map_conf", None),
            name="output.map_conf",
        )
        self.map_iou = self._parse_optional_threshold(
            oc.get("map_iou", None),
            name="output.map_iou",
        )
        self.save_freq = max(0, int(oc.get("save_freq", 5)))
        self.save_pth_checkpoints = bool(oc.get("save_pth_checkpoints", True))
        self.map_eval_interval = max(1, int(oc.get("map_eval_interval", 1)))
        self.debug_export_interval = max(0, int(oc.get("debug_export_interval", 0)))
        self.debug_tile_size = max(128, int(oc.get("debug_tile_size", 320)))
        self.debug_pred_conf = float(oc.get("debug_pred_conf", 0.10))
        self.debug_pred_max_boxes = max(1, int(oc.get("debug_pred_max_boxes", 300)))
        self.debug_dir = os.path.join(config["output"]["save_dir"], "debug_exports")
        os.makedirs(self.debug_dir, exist_ok=True)

        if self.best_by != "loss" and not self.map_data:
            raise ValueError(
                "output.best_by is set to mAP mode, but output.map_data is not set. "
                "Provide a dataset yaml path via --map-data."
            )

        self.logger.info(f"Best selection metric: {self.best_by}")
        if self.save_pth_checkpoints:
            if self.save_freq > 0:
                self.logger.info(
                    "PyTorch checkpoints: enabled (every %d epochs + best_model.pth)",
                    self.save_freq,
                )
            else:
                self.logger.info("PyTorch checkpoints: best-only (.pth periodic disabled)")
        else:
            self.logger.info("PyTorch checkpoints: disabled (.pth suppressed, student_best.pt only)")
        if self.debug_export_interval > 0:
            self.logger.info(
                "Debug exports: every %d epochs -> %s",
                self.debug_export_interval,
                self.debug_dir,
            )
        resume_path = config.get("checkpoints", {}).get("resume")
        if resume_path:
            self._load_checkpoint(resume_path)

        # For mAP-based selection, do not inherit historical best from a loss-only checkpoint.
        if self.best_by != "loss":
            self.best_score = float("-inf")

    # ------------------------------------------------------------------
    # Pseudo labels
    # ------------------------------------------------------------------
    def load_pseudo_labels_from_dir(self, label_dir: str):
        """
        Load pseudo-labels from a directory of YOLO ``.txt`` files.

        Each file should be named ``{image_stem}.txt`` and contain one
        detection per line in standard YOLO format
        (``class cx cy w h [conf]``) or OBB format.
        """
        self.pseudo_labels = load_pseudo_labels_dir(
            label_dir,
            score_threshold=self.pl_score_threshold,
        )

    def load_pseudo_labels_from_csv(self, csv_path: str, **kwargs):
        """Load pseudo-labels from a CSV file."""
        self.pseudo_labels = load_pseudo_labels_csv(
            csv_path,
            score_threshold=self.pl_score_threshold,
            **kwargs,
        )

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------
    def _setup_optimizer(self):
        tc = self.config["training"]
        params = self.model.get_trainable_parameters()
        self.optimizer = torch.optim.AdamW(
            params, lr=tc["learning_rate"], weight_decay=tc["weight_decay"],
        )

    def _setup_scheduler(self):
        tc = self.config["training"]
        stype = tc.get("lr_scheduler", "cosine")
        warmup_epochs = max(0, int(tc.get("warmup_epochs", 0)))
        total_epochs = max(1, int(tc.get("epochs", 1)))
        warmup_start_factor = float(tc.get("warmup_start_factor", 0.1))
        warmup_start_factor = min(max(warmup_start_factor, 1e-8), 1.0)

        base_scheduler = None
        if stype == "cosine":
            cosine_tmax = max(1, total_epochs - min(warmup_epochs, total_epochs))
            base_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, T_max=tc["epochs"], eta_min=tc.get("min_lr", 1e-6),
            )
            if warmup_epochs > 0 and warmup_epochs < total_epochs:
                base_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                    self.optimizer,
                    T_max=cosine_tmax,
                    eta_min=tc.get("min_lr", 1e-6),
                )
        elif stype == "step":
            base_scheduler = torch.optim.lr_scheduler.StepLR(
                self.optimizer, step_size=30, gamma=0.1,
            )
        else:
            self.scheduler = None
            return

        if warmup_epochs > 0:
            warmup_iters = min(warmup_epochs, total_epochs)
            warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
                self.optimizer,
                start_factor=warmup_start_factor,
                end_factor=1.0,
                total_iters=warmup_iters,
            )
            if base_scheduler is None or warmup_iters >= total_epochs:
                self.scheduler = warmup_scheduler
            else:
                self.scheduler = torch.optim.lr_scheduler.SequentialLR(
                    self.optimizer,
                    schedulers=[warmup_scheduler, base_scheduler],
                    milestones=[warmup_iters],
                )
        else:
            self.scheduler = base_scheduler

    def _load_checkpoint(self, path: str):
        # Resume with non-strict matching because the alignment head is
        # lazily built after the first forward and may be absent at load time.
        ckpt = load_checkpoint(
            self.model,
            path,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            strict=False,
        )
        self.start_epoch = ckpt.get("epoch", 0) + 1
        self.best_loss = ckpt.get("loss", float("inf"))
        if self.student_ema is not None and self._ema_target is not None:
            ema_state = ckpt.get("ema_student_state_dict", None)
            if ema_state is None:
                extra_info = ckpt.get("extra_info", {}) or {}
                ema_state = extra_info.get("ema_student_state_dict", None)

            if ema_state is not None:
                self.student_ema.load_state_dict(
                    {
                        "ema_state_dict": ema_state,
                        "updates": ckpt.get("ema_updates", 0),
                        "decay": ckpt.get("ema_decay", self.ema_decay),
                        "tau": ckpt.get("ema_tau", self.ema_tau),
                    }
                )
                self.logger.info(
                    "Restored student EMA (updates=%d)",
                    int(self.student_ema.updates),
                )
            else:
                self.student_ema.set(self._ema_target)
                self.logger.info("Checkpoint has no EMA state; seeded student EMA from loaded model")
        self.logger.info(f"Resumed from epoch {self.start_epoch}")

    def _resolve_ema_target(self) -> Optional[nn.Module]:
        student = getattr(self.model, "student", None)
        if student is None:
            return None
        det_model = getattr(student, "det_model", None)
        if isinstance(det_model, nn.Module):
            return det_model
        if isinstance(student, nn.Module):
            return student
        return None

    def _checkpoint_extra_state(self) -> Optional[Dict[str, Any]]:
        if self.student_ema is None:
            return None
        ema_state = self.student_ema.state_dict()
        return {
            "ema_student_state_dict": ema_state["ema_state_dict"],
            "ema_updates": ema_state["updates"],
            "ema_decay": ema_state["decay"],
            "ema_tau": ema_state["tau"],
        }

    @contextmanager
    def _use_ema_student(self):
        if self.student_ema is None or self._ema_target is None:
            yield
            return
        self.student_ema.store(self._ema_target)
        self.student_ema.copy_to(self._ema_target)
        try:
            yield
        finally:
            self.student_ema.restore(self._ema_target)

    def _export_best_ultralytics_pt(self, epoch: int):
        """Export student branch to Ultralytics-compatible .pt if available."""
        student = getattr(self.model, "student", None)
        export_fn = getattr(student, "export_ultralytics_pt", None)
        if student is None or export_fn is None:
            return

        out_path = os.path.join(self.config["output"]["save_dir"], "student_best.pt")
        num_classes = int(self.config.get("model", {}).get("num_classes", 1))
        class_names = {i: f"class_{i}" for i in range(num_classes)}

        export_fitness = (
            float(self.best_score)
            if self._best_higher_is_better
            else float(-self.best_score)
        )

        try:
            with self._use_ema_student():
                export_fn(
                    save_path=out_path,
                    num_classes=num_classes,
                    class_names=class_names,
                    epoch=epoch,
                    best_fitness=export_fitness,
                )
            self.logger.info(f"Exported Ultralytics student weights: {out_path}")
        except Exception as exc:
            self.logger.warning(f"Failed to export Ultralytics .pt: {exc}")

    @torch.no_grad()
    def _evaluate_student_map(self, epoch: int) -> Optional[Dict[str, float]]:
        """Evaluate exported student with Ultralytics val() and return mAP metrics."""
        if not self.map_data:
            return None

        student = getattr(self.model, "student", None)
        export_fn = getattr(student, "export_ultralytics_pt", None)
        if student is None or export_fn is None:
            return None

        tmp_pt = os.path.join(self.config["output"]["save_dir"], f"student_eval_epoch_{epoch}.pt")
        tmp_eval_root = os.path.join(self.config["output"]["save_dir"], ".map_eval_tmp")
        tmp_eval_name = f"epoch_{epoch}"
        tmp_eval_dir = os.path.join(tmp_eval_root, tmp_eval_name)
        num_classes = int(self.config.get("model", {}).get("num_classes", 1))
        class_names = {i: f"class_{i}" for i in range(num_classes)}

        try:
            with self._use_ema_student():
                export_fn(
                    save_path=tmp_pt,
                    num_classes=num_classes,
                    class_names=class_names,
                    epoch=epoch,
                )

            from ultralytics import YOLO

            model = YOLO(tmp_pt)
            val_kwargs = {
                "data": self.map_data,
                "split": self.map_split,
                "imgsz": self.map_imgsz,
                "batch": self.map_batch,
                "device": self.config.get("device", "cuda"),
                "project": tmp_eval_root,
                "name": tmp_eval_name,
                "exist_ok": True,
                "plots": False,
                "verbose": False,
            }
            if self.map_conf is not None:
                val_kwargs["conf"] = float(self.map_conf)
            if self.map_iou is not None:
                val_kwargs["iou"] = float(self.map_iou)

            metrics = model.val(**val_kwargs)

            result = {
                "map50": float(metrics.box.map50),
                "map5095": float(metrics.box.map),
            }
            result["fitness"] = 0.1 * result["map50"] + 0.9 * result["map5095"]
            self.logger.info(
                "mAP eval | map50=%.4f map50-95=%.4f fitness=%.4f",
                result["map50"], result["map5095"], result["fitness"],
            )
            return result
        except Exception as exc:
            self.logger.warning(f"mAP evaluation failed at epoch {epoch}: {exc}")
            return None
        finally:
            try:
                if os.path.isfile(tmp_pt):
                    os.remove(tmp_pt)
            except Exception:
                pass
            try:
                if os.path.isdir(tmp_eval_dir):
                    shutil.rmtree(tmp_eval_dir, ignore_errors=True)
                if os.path.isdir(tmp_eval_root) and not os.listdir(tmp_eval_root):
                    os.rmdir(tmp_eval_root)
            except Exception:
                pass

    # ------------------------------------------------------------------
    # Train one epoch
    # ------------------------------------------------------------------
    def _parse_optional_threshold(self, raw: Any, name: str) -> Optional[float]:
        """Parse a threshold-like config value that may be omitted."""
        if raw is None:
            return None

        try:
            value = float(raw)
        except (TypeError, ValueError):
            self.logger.warning("%s should be a float or null, got %r; using Ultralytics default", name, raw)
            return None

        return min(max(value, 0.0), 1.0)

    def _parse_class_score_thresholds(
        self,
        raw: Any,
        num_classes: int,
    ) -> Dict[int, float]:
        """Parse optional per-class pseudo score thresholds from config."""
        parsed: Dict[int, float] = {}
        if raw is None:
            return parsed

        if isinstance(raw, dict):
            items = raw.items()
        elif isinstance(raw, (list, tuple)):
            items = enumerate(raw)
        else:
            self.logger.warning(
                "pseudo_labels.class_score_thresholds should be dict/list, got %s; ignored",
                type(raw).__name__,
            )
            return parsed

        for key, value in items:
            try:
                cls_id = int(key)
            except (TypeError, ValueError):
                self.logger.warning(
                    "Invalid class id in pseudo_labels.class_score_thresholds: %r; ignored",
                    key,
                )
                continue

            if cls_id < 0 or cls_id >= max(1, num_classes):
                self.logger.warning(
                    "Class id %d out of range [0, %d) in pseudo_labels.class_score_thresholds; ignored",
                    cls_id,
                    max(1, num_classes),
                )
                continue

            try:
                threshold = float(value)
            except (TypeError, ValueError):
                self.logger.warning(
                    "Invalid threshold value for class %d in pseudo_labels.class_score_thresholds: %r; ignored",
                    cls_id,
                    value,
                )
                continue

            threshold = min(max(threshold, 0.0), 1.0)
            parsed[cls_id] = threshold

        return parsed

    def _parse_area_class_prior(
        self,
        raw: Any,
        num_classes: int,
    ) -> Optional[Dict[str, Any]]:
        """Parse optional area-guided class relabeling for online pseudo labels."""
        if raw is None or raw is False:
            return None
        if not isinstance(raw, dict):
            self.logger.warning(
                "pseudo_labels.area_class_prior should be a dict or null, got %s; ignored",
                type(raw).__name__,
            )
            return None

        if not bool(raw.get("enabled", True)):
            return None

        source_class_raw = raw.get("source_class", 0)
        source_class: Optional[int]
        if source_class_raw is None:
            source_class = None
        else:
            try:
                source_class = int(source_class_raw)
            except (TypeError, ValueError):
                self.logger.warning(
                    "Invalid source_class in pseudo_labels.area_class_prior: %r; ignored",
                    source_class_raw,
                )
                return None
            if source_class < 0 or source_class >= max(1, num_classes):
                self.logger.warning(
                    "source_class %d out of range [0, %d) in pseudo_labels.area_class_prior; ignored",
                    source_class,
                    max(1, num_classes),
                )
                return None

        try:
            target_class = int(raw.get("target_class", 1))
        except (TypeError, ValueError):
            self.logger.warning(
                "Invalid target_class in pseudo_labels.area_class_prior: %r; ignored",
                raw.get("target_class", 1),
            )
            return None
        if target_class < 0 or target_class >= max(1, num_classes):
            self.logger.warning(
                "target_class %d out of range [0, %d) in pseudo_labels.area_class_prior; ignored",
                target_class,
                max(1, num_classes),
            )
            return None

        def _parse_area_bound(value: Any, key: str) -> Optional[float]:
            if value is None:
                return None
            try:
                parsed = float(value)
            except (TypeError, ValueError):
                self.logger.warning(
                    "Invalid %s in pseudo_labels.area_class_prior: %r; ignored",
                    key,
                    value,
                )
                return None
            if parsed < 0.0:
                self.logger.warning(
                    "%s in pseudo_labels.area_class_prior must be >= 0, got %.6f; ignored",
                    key,
                    parsed,
                )
                return None
            return parsed

        min_area = _parse_area_bound(raw.get("min_area", None), "min_area")
        max_area = _parse_area_bound(raw.get("max_area", None), "max_area")
        if min_area is None and max_area is None:
            self.logger.warning(
                "pseudo_labels.area_class_prior requires at least one of min_area/max_area; ignored"
            )
            return None
        if min_area is not None and max_area is not None and min_area > max_area:
            self.logger.warning(
                "pseudo_labels.area_class_prior has min_area > max_area (%.6f > %.6f); ignored",
                min_area,
                max_area,
            )
            return None

        return {
            "source_class": source_class,
            "target_class": target_class,
            "min_area": min_area,
            "max_area": max_area,
        }

    def _format_pseudo_class_counts(self, metrics: Dict[str, float]) -> str:
        """Return compact pseudo class-count summary for train/val logs."""
        parts: List[str] = []
        for cls_id in range(max(1, self.num_classes)):
            key = f"pseudo_box_count_c{cls_id}"
            if key in metrics:
                parts.append(f"pseudo_c{cls_id}={metrics.get(key, 0.0):.1f}")
        if "pseudo_area_relabel_count" in metrics:
            parts.append(f"area_relabel={metrics.get('pseudo_area_relabel_count', 0.0):.1f}")
        return ", ".join(parts)

    def _resolve_pattern_spacing_prior(self, prior_name: Optional[str]) -> Dict[str, float]:
        empty = {
            "period_ratio_x": 0.0,
            "row_period_ratio_y": 0.0,
            "row_tolerance_ratio_y": 0.0,
        }
        name = str(prior_name or "").strip()
        bank_path_raw = self.pattern_align_species_prior_bank
        if not name or not bank_path_raw:
            return empty

        bank_path = Path(str(bank_path_raw)).expanduser().resolve()
        if not bank_path.is_file():
            self.logger.warning("pattern_align_species_prior_bank not found: %s", bank_path)
            return empty

        try:
            with bank_path.open("r", encoding="utf-8") as fh:
                payload = json.load(fh)
        except Exception as exc:
            self.logger.warning("Failed to load pattern spacing prior bank %s: %s", bank_path, exc)
            return empty

        priors = payload.get("priors", []) if isinstance(payload, dict) else []
        for item in priors:
            if str(item.get("name", "")).strip() != name:
                continue
            try:
                img_w = float(item.get("image_width", 0.0) or 0.0)
                img_h = float(item.get("image_height", 0.0) or 0.0)
                period_px = float(
                    item.get("x_period_px", item.get("horizontal_period_px", item.get("period_px", 0.0))) or 0.0
                )
                row_period_px = float(
                    item.get("row_period_px", item.get("y_period_px", item.get("vertical_period_px", 0.0))) or 0.0
                )
                row_tol_px = float(item.get("row_tolerance_px", 0.0) or 0.0)
            except (TypeError, ValueError):
                self.logger.warning("Invalid spacing prior entry for %s in %s", name, bank_path)
                return empty

            if row_tol_px <= 0.0 and row_period_px > 0.0 and img_h > 0.0:
                row_tol_px = max(row_period_px * 0.22, 18.0)

            resolved = {
                "period_ratio_x": period_px / img_w if img_w > 0.0 else 0.0,
                "row_period_ratio_y": row_period_px / img_h if img_h > 0.0 else 0.0,
                "row_tolerance_ratio_y": row_tol_px / img_h if img_h > 0.0 else 0.0,
            }
            self.logger.info(
                "Loaded pattern spacing prior '%s' from %s | x_period_ratio=%.4f row_period_ratio=%.4f row_tol_ratio=%.4f",
                name,
                bank_path,
                resolved["period_ratio_x"],
                resolved["row_period_ratio_y"],
                resolved["row_tolerance_ratio_y"],
            )
            return resolved

        self.logger.warning("pattern spacing prior named %r not found in %s", name, bank_path)
        return empty

    @staticmethod
    def _aggregate_pattern_stats(stats_list: Sequence[Dict[str, float]]) -> Dict[str, float]:
        if not stats_list:
            return {}
        keys = {key for stats in stats_list for key in stats.keys()}
        aggregated: Dict[str, float] = {}
        for key in keys:
            values = [float(stats.get(key, 0.0) or 0.0) for stats in stats_list]
            aggregated[key] = float(np.mean(values)) if values else 0.0
        return aggregated

    def _resolve_bypass_ratios_for_batch(
        self,
        image_paths: Sequence[str],
        images: torch.Tensor,
        default_cfg: Dict[str, float],
    ) -> Optional[List[Dict[str, float]]]:
        if not self.pattern_align_use_dino_bypass:
            return None
        if not image_paths or images is None:
            return None
        if images.ndim < 4:
            return None
        if (
            not self._dino_bypass_image_prior_index
            and not self._dino_bypass_species_prior_bank
            and self._dino_bypass_period_prior_px <= 0.0
            and self._dino_bypass_row_period_prior_px <= 0.0
            and self._dino_bypass_row_tolerance_override_px <= 0.0
        ):
            return None

        img_w = float(images.shape[-1])
        img_h = float(images.shape[-2])
        ratios: List[Dict[str, float]] = []
        for path in image_paths:
            prior = _resolve_external_pattern_prior(
                path,
                image_size=(img_w, img_h),
                image_prior_index=self._dino_bypass_image_prior_index,
                species_prior_bank=self._dino_bypass_species_prior_bank,
            )
            period_px = float(prior.get("x_period_px", 0.0) or 0.0)
            row_period_px = float(prior.get("row_period_px", 0.0) or 0.0)
            row_tol_px = float(prior.get("row_tolerance_px", 0.0) or 0.0)
            if self._dino_bypass_period_prior_px > 0.0:
                period_px = self._dino_bypass_period_prior_px
            if self._dino_bypass_row_period_prior_px > 0.0:
                row_period_px = self._dino_bypass_row_period_prior_px
            if self._dino_bypass_row_tolerance_override_px > 0.0:
                row_tol_px = self._dino_bypass_row_tolerance_override_px
            if row_tol_px <= 0.0 and row_period_px > 0.0:
                if self._dino_bypass_row_tolerance_scale > 0.0:
                    row_tol_px = row_period_px * self._dino_bypass_row_tolerance_scale
                else:
                    row_tol_px = max(row_period_px * 0.22, 18.0)

            ratios.append(
                {
                    "period_ratio_x": period_px / img_w if period_px > 0.0 else default_cfg.get("period_ratio_x", 0.0),
                    "row_period_ratio_y": row_period_px / img_h if row_period_px > 0.0 else default_cfg.get("row_period_ratio_y", 0.0),
                    "row_tolerance_ratio_y": row_tol_px / img_h if row_tol_px > 0.0 else default_cfg.get("row_tolerance_ratio_y", 0.0),
                }
            )
        return ratios

    def _build_pattern_align_mask_for_batch(
        self,
        student_feat: torch.Tensor,
        dino_feat: torch.Tensor,
        image_paths: Sequence[str],
        images: torch.Tensor,
        default_prior_cfg: Dict[str, float],
        *,
        use_bypass: bool,
        hard_mask: bool,
        hard_mask_threshold: float,
        detach: bool,
    ) -> Tuple[Optional[torch.Tensor], Dict[str, float]]:
        if not self.pattern_align_enabled or not hasattr(self.model, "build_pattern_align_mask"):
            return None, {}

        ratios = self._resolve_bypass_ratios_for_batch(image_paths, images, default_prior_cfg) if use_bypass else None
        if ratios and len(ratios) == int(student_feat.shape[0]):
            masks: List[torch.Tensor] = []
            stats_list: List[Dict[str, float]] = []
            for idx, ratio in enumerate(ratios):
                mask, stats = self.model.build_pattern_align_mask(
                    student_feat[idx : idx + 1],
                    dino_feat[idx : idx + 1],
                    target_coverage=self.pattern_align_target_coverage,
                    min_coverage=self.pattern_align_min_coverage,
                    max_coverage=self.pattern_align_max_coverage,
                    mode=self.pattern_align_mode,
                    pca_components=self.pattern_align_pca_components,
                    hybrid_pca_weight=self.pattern_align_hybrid_pca_weight,
                    filtered_prior_strength=self.pattern_align_filtered_prior_strength,
                    filtered_completion_strength=self.pattern_align_filtered_completion_strength,
                    filtered_completion_gamma=self.pattern_align_filtered_completion_gamma,
                    filtered_noise_suppress=self.pattern_align_filtered_noise_suppress,
                    sim_weight=self.pattern_align_sim_weight,
                    dino_weight=self.pattern_align_dino_weight,
                    student_weight=self.pattern_align_student_weight,
                    period_ratio_x=ratio["period_ratio_x"],
                    row_period_ratio_y=ratio["row_period_ratio_y"],
                    row_tolerance_ratio_y=ratio["row_tolerance_ratio_y"],
                    structural_prior_strength=self.pattern_align_structural_prior_strength,
                    structural_cross_row_strength=self.pattern_align_structural_cross_row_strength,
                    structural_seed_threshold=self.pattern_align_structural_seed_threshold,
                    structural_min_row_seeds=self.pattern_align_structural_min_row_seeds,
                    temperature=self.pattern_align_temperature,
                    mask_floor=self.pattern_align_mask_floor,
                    hard_mask=hard_mask,
                    hard_mask_threshold=hard_mask_threshold,
                    detach=detach,
                )
                masks.append(mask)
                stats_list.append(stats)
            return torch.cat(masks, dim=0), self._aggregate_pattern_stats(stats_list)

        mask, stats = self.model.build_pattern_align_mask(
            student_feat,
            dino_feat,
            target_coverage=self.pattern_align_target_coverage,
            min_coverage=self.pattern_align_min_coverage,
            max_coverage=self.pattern_align_max_coverage,
            mode=self.pattern_align_mode,
            pca_components=self.pattern_align_pca_components,
            hybrid_pca_weight=self.pattern_align_hybrid_pca_weight,
            filtered_prior_strength=self.pattern_align_filtered_prior_strength,
            filtered_completion_strength=self.pattern_align_filtered_completion_strength,
            filtered_completion_gamma=self.pattern_align_filtered_completion_gamma,
            filtered_noise_suppress=self.pattern_align_filtered_noise_suppress,
            sim_weight=self.pattern_align_sim_weight,
            dino_weight=self.pattern_align_dino_weight,
            student_weight=self.pattern_align_student_weight,
            period_ratio_x=default_prior_cfg.get("period_ratio_x", 0.0),
            row_period_ratio_y=default_prior_cfg.get("row_period_ratio_y", 0.0),
            row_tolerance_ratio_y=default_prior_cfg.get("row_tolerance_ratio_y", 0.0),
            structural_prior_strength=self.pattern_align_structural_prior_strength,
            structural_cross_row_strength=self.pattern_align_structural_cross_row_strength,
            structural_seed_threshold=self.pattern_align_structural_seed_threshold,
            structural_min_row_seeds=self.pattern_align_structural_min_row_seeds,
            temperature=self.pattern_align_temperature,
            mask_floor=self.pattern_align_mask_floor,
            hard_mask=hard_mask,
            hard_mask_threshold=hard_mask_threshold,
            detach=detach,
        )
        return mask, stats

    def _move_batch_to_device(
        self,
        batch: Dict[str, Any],
    ) -> Tuple[torch.Tensor, Dict[str, Any], List[str], Optional[torch.Tensor], Optional[torch.Tensor]]:
        images = batch["images"].to(self.device)
        targets = {
            k: (v.to(self.device) if isinstance(v, torch.Tensor) else v)
            for k, v in batch.get("targets", {}).items()
        }
        image_paths: List[str] = batch.get("image_paths", [])

        images_weak: Optional[torch.Tensor] = None
        if "images_weak" in batch and isinstance(batch["images_weak"], torch.Tensor):
            images_weak = batch["images_weak"].to(self.device)

        strong_hflip = batch.get("strong_hflip")
        if isinstance(strong_hflip, torch.Tensor):
            strong_hflip = strong_hflip.to(self.device)

        return images, targets, image_paths, images_weak, strong_hflip

    def _gate_pseudo_batch_by_hard_mask(
        self,
        pseudo_batch: Optional[Dict[str, torch.Tensor]],
        hard_mask: Optional[torch.Tensor],
    ) -> Tuple[Optional[Dict[str, torch.Tensor]], Dict[str, float]]:
        if pseudo_batch is None or hard_mask is None:
            return pseudo_batch, {}
        if not isinstance(hard_mask, torch.Tensor) or hard_mask.ndim != 4 or hard_mask.shape[1] != 1:
            return pseudo_batch, {}
        if pseudo_batch.get("bboxes") is None or pseudo_batch["bboxes"].shape[0] == 0:
            return pseudo_batch, {}

        mask = hard_mask.float()
        mask = F.max_pool2d(mask, kernel_size=3, stride=1, padding=1)
        batch_idx = pseudo_batch["batch_idx"].long()
        bboxes = pseudo_batch["bboxes"]
        if bboxes.ndim != 2 or bboxes.shape[0] != batch_idx.shape[0] or bboxes.shape[1] < 2:
            return pseudo_batch, {}

        height = int(mask.shape[2])
        width = int(mask.shape[3])
        cx = bboxes[:, 0].clamp(0.0, 1.0 - 1e-6)
        cy = bboxes[:, 1].clamp(0.0, 1.0 - 1e-6)
        xs = torch.clamp((cx * width).long(), min=0, max=max(width - 1, 0))
        ys = torch.clamp((cy * height).long(), min=0, max=max(height - 1, 0))
        keep = mask[batch_idx, 0, ys, xs] > 0.5

        total = int(keep.numel())
        kept = int(keep.sum().item())
        stats = {
            "pseudo_gate_total_boxes": float(total),
            "pseudo_gate_kept_boxes": float(kept),
            "pseudo_gate_keep_ratio": float(kept / max(total, 1)),
            "pseudo_gate_mask_coverage": float(mask.mean().item()),
        }

        if kept <= 0:
            empty_batch = {
                "batch_idx": pseudo_batch["batch_idx"][:0],
                "cls": pseudo_batch["cls"][:0],
                "bboxes": pseudo_batch["bboxes"][:0],
            }
            return empty_batch, stats
        if kept == total:
            return pseudo_batch, stats

        gated = {}
        for key, value in pseudo_batch.items():
            if isinstance(value, torch.Tensor) and value.ndim >= 1 and value.shape[0] == total:
                gated[key] = value[keep]
            else:
                gated[key] = value
        return gated, stats

    @staticmethod
    def _merge_loss_dict(
        base: Dict[str, float],
        update: Dict[str, float],
    ) -> Dict[str, float]:
        for key, val in update.items():
            if key.endswith("_per_img"):
                continue
            if isinstance(val, torch.Tensor):
                val = val.item()
            try:
                fval = float(val)
            except (TypeError, ValueError):
                continue
            base[key] = base.get(key, 0.0) + fval
        return base

    @staticmethod
    def _finalize_loss_dict(
        loss_dict: Dict[str, float],
        batch_size: int,
    ) -> Dict[str, float]:
        bs = float(max(1, batch_size))
        if "total_loss" in loss_dict:
            loss_dict["total_loss_per_img"] = loss_dict["total_loss"] / bs
        if "loss_det" in loss_dict:
            loss_dict["loss_det_per_img"] = loss_dict["loss_det"] / bs
        if "loss_pseudo" in loss_dict:
            loss_dict["loss_pseudo_per_img"] = loss_dict["loss_pseudo"] / bs
        if "loss_pseudo_weighted" in loss_dict:
            loss_dict["loss_pseudo_weighted_per_img"] = (
                loss_dict["loss_pseudo_weighted"] / bs
            )
        return loss_dict

    @staticmethod
    def _build_patch_prior_mask(
        targets: Dict[str, Any],
        spatial_hw: Tuple[int, int],
    ) -> Optional[torch.Tensor]:
        boxes = targets.get("boxes")
        valid_mask = targets.get("valid_mask")
        if not isinstance(boxes, torch.Tensor) or not isinstance(valid_mask, torch.Tensor):
            return None
        if boxes.ndim != 3 or valid_mask.ndim != 2 or boxes.shape[:2] != valid_mask.shape:
            return None

        h, w = int(spatial_hw[0]), int(spatial_hw[1])
        if h <= 0 or w <= 0:
            return None

        boxes_xywh = boxes[..., :4].float()
        valid = valid_mask.bool()
        if not bool(valid.any().item()):
            return torch.zeros(
                (boxes.shape[0], 1, h, w),
                dtype=torch.float32,
                device=boxes.device,
            )

        cx = boxes_xywh[..., 0].clamp(0.0, 1.0)
        cy = boxes_xywh[..., 1].clamp(0.0, 1.0)
        bw = boxes_xywh[..., 2].clamp(0.0, 1.0)
        bh = boxes_xywh[..., 3].clamp(0.0, 1.0)
        x0 = (cx - 0.5 * bw).clamp(0.0, 1.0)
        x1 = (cx + 0.5 * bw).clamp(0.0, 1.0)
        y0 = (cy - 0.5 * bh).clamp(0.0, 1.0)
        y1 = (cy + 0.5 * bh).clamp(0.0, 1.0)

        ys = (torch.arange(h, device=boxes.device, dtype=torch.float32) + 0.5) / float(h)
        xs = (torch.arange(w, device=boxes.device, dtype=torch.float32) + 0.5) / float(w)
        yy, xx = torch.meshgrid(ys, xs, indexing="ij")
        xx = xx.unsqueeze(0).unsqueeze(0)
        yy = yy.unsqueeze(0).unsqueeze(0)

        inside = (
            (xx >= x0.unsqueeze(-1).unsqueeze(-1))
            & (xx <= x1.unsqueeze(-1).unsqueeze(-1))
            & (yy >= y0.unsqueeze(-1).unsqueeze(-1))
            & (yy <= y1.unsqueeze(-1).unsqueeze(-1))
            & valid.unsqueeze(-1).unsqueeze(-1)
        )
        mask = inside.any(dim=1, keepdim=True).float()
        return mask

    @staticmethod
    def _merge_prior_targets(
        *masks: Optional[torch.Tensor],
    ) -> Optional[torch.Tensor]:
        merged: Optional[torch.Tensor] = None
        for mask in masks:
            if not isinstance(mask, torch.Tensor):
                continue
            if merged is None:
                merged = mask.float()
                continue
            if merged.shape[2:] != mask.shape[2:]:
                mask = F.interpolate(
                    mask.float(),
                    size=merged.shape[2:],
                    mode="bilinear",
                    align_corners=False,
                )
            merged = torch.maximum(merged, mask.float())
        return merged

    def _compute_student_gate_distill_loss(
        self,
        teacher_target: Optional[torch.Tensor],
    ) -> Tuple[Optional[torch.Tensor], Dict[str, float]]:
        if (
            not self.student_gate_distill_enabled
            or teacher_target is None
            or self.student_gate_distill_w <= 0.0
        ):
            return None, {}
        student = getattr(self.model, "student", None)
        get_gate_maps = getattr(student, "get_support_gate_maps", None)
        if not callable(get_gate_maps):
            return None, {}

        records = get_gate_maps()
        if not records:
            return None, {"student_gate_count": 0.0}

        target = teacher_target.detach().float()
        losses: List[torch.Tensor] = []
        means: List[float] = []
        target_means: List[float] = []
        for rec in records:
            logits = rec.get("logits") if isinstance(rec, dict) else None
            mask = rec.get("mask") if isinstance(rec, dict) else None
            if not isinstance(logits, torch.Tensor) or not isinstance(mask, torch.Tensor):
                continue
            gate_target = target.to(device=logits.device, dtype=logits.dtype)
            if gate_target.shape[2:] != logits.shape[2:]:
                gate_target = F.interpolate(
                    gate_target,
                    size=logits.shape[2:],
                    mode="bilinear",
                    align_corners=False,
                )
            pos_weight = torch.tensor(
                [max(float(self.student_gate_distill_pos_w), 1e-6)],
                dtype=logits.dtype,
                device=logits.device,
            )
            bce = F.binary_cross_entropy_with_logits(logits, gate_target, pos_weight=pos_weight)
            mse = F.mse_loss(mask, gate_target)
            losses.append(float(self.student_gate_distill_bce_w) * bce + float(self.student_gate_distill_mse_w) * mse)
            means.append(float(mask.detach().mean().item()))
            target_means.append(float(gate_target.detach().mean().item()))

        if not losses:
            return None, {"student_gate_count": float(len(records))}
        loss = torch.stack(losses).mean()
        return loss, {
            "student_gate_count": float(len(losses)),
            "student_gate_mask_mean": float(np.mean(means)) if means else 0.0,
            "student_gate_target_mean": float(np.mean(target_means)) if target_means else 0.0,
        }

    @staticmethod
    def _profile_autocorr(profile: torch.Tensor) -> torch.Tensor:
        if profile.ndim != 2:
            raise ValueError(f"_profile_autocorr expects [B, L], got {tuple(profile.shape)}")
        centered = profile - profile.mean(dim=1, keepdim=True)
        denom = centered.square().sum(dim=1, keepdim=True).clamp_min(1e-6)
        n = int(centered.shape[1])
        fft = torch.fft.rfft(centered, n=2 * n, dim=1)
        acf = torch.fft.irfft(fft * torch.conj(fft), n=2 * n, dim=1)[:, :n]
        return acf / denom

    def _compute_prior_periodic_loss(
        self,
        student_prob: torch.Tensor,
        teacher_target: torch.Tensor,
    ) -> torch.Tensor:
        if student_prob.ndim != 4 or teacher_target.ndim != 4:
            raise ValueError("prior periodic loss expects [B,1,H,W] tensors")
        if student_prob.shape[2:] != teacher_target.shape[2:]:
            teacher_target = F.interpolate(
                teacher_target.float(),
                size=student_prob.shape[2:],
                mode="bilinear",
                align_corners=False,
            )
        student_map = student_prob.squeeze(1).float()
        teacher_map = teacher_target.squeeze(1).float()

        # Horizontal periodicity is the primary signal; row profile is auxiliary.
        student_col = student_map.mean(dim=1)
        teacher_col = teacher_map.mean(dim=1)
        student_row = student_map.mean(dim=2)
        teacher_row = teacher_map.mean(dim=2)

        col_profile_loss = F.mse_loss(student_col, teacher_col)
        row_profile_loss = F.mse_loss(student_row, teacher_row)
        col_acf_loss = F.mse_loss(
            self._profile_autocorr(student_col),
            self._profile_autocorr(teacher_col),
        )
        row_acf_loss = F.mse_loss(
            self._profile_autocorr(student_row),
            self._profile_autocorr(teacher_row),
        )

        col_term = 0.35 * col_profile_loss + 0.65 * col_acf_loss
        row_term = 0.35 * row_profile_loss + 0.65 * row_acf_loss
        return (
            float(self.prior_periodic_col_w) * col_term
            + float(self.prior_periodic_row_w) * row_term
        )

    def train_epoch(self, epoch: int) -> Dict[str, float]:
        """
        One epoch following the 2-stage MTKD schedule.
        """
        self.model.train()
        meters = AverageMeterDict()
        accum = self.config["training"].get("accumulation_steps", 1)
        start_time = time.time()
        num_batches = len(self.train_loader)
        self.optimizer.zero_grad()

        for batch_idx, batch in enumerate(self.train_loader):
            # === Determine stage ===
            do_target_align = epoch >= self.align_target_start_epoch
            do_source_align = self.use_source_alignment

            is_dual_stream_batch = (
                isinstance(batch, (tuple, list))
                and len(batch) == 2
                and isinstance(batch[0], dict)
                and isinstance(batch[1], dict)
            )

            if is_dual_stream_batch:
                label_batch, unlabel_batch = batch
                (
                    label_images,
                    label_targets,
                    label_image_paths,
                    label_images_weak,
                    label_hflip,
                ) = self._move_batch_to_device(label_batch)
                (
                    unlabel_images,
                    unlabel_targets,
                    unlabel_image_paths,
                    unlabel_images_weak,
                    unlabel_hflip,
                ) = self._move_batch_to_device(unlabel_batch)

                total_batch_size = int(label_images.size(0) + unlabel_images.size(0))

                def _forward_dual_stream() -> Tuple[torch.Tensor, Dict[str, float]]:
                    merged_dict: Dict[str, float] = {}

                    label_loss, label_loss_dict = self._forward_and_loss(
                        label_images,
                        label_targets,
                        label_image_paths,
                        epoch,
                        need_teacher_feat=do_source_align,
                        do_source_align=do_source_align,
                        do_target_align=False,
                        images_weak=label_images_weak,
                        strong_hflip=label_hflip,
                        force_gt_supervision=self.use_gt_supervision,
                        force_pseudo_supervision=False,
                    )
                    merged_dict = self._merge_loss_dict(merged_dict, label_loss_dict)
                    total_loss = label_loss

                    run_unlabel_branch = do_target_align and (
                        self.use_pseudo_supervision or self.use_target_alignment
                    )
                    if run_unlabel_branch:
                        unlabel_need_teacher = bool(
                            self.use_target_alignment or self.pseudo_hard_mask_gate_enabled
                        )
                        unlabel_loss, unlabel_loss_dict = self._forward_and_loss(
                            unlabel_images,
                            unlabel_targets,
                            unlabel_image_paths,
                            epoch,
                            need_teacher_feat=unlabel_need_teacher,
                            do_source_align=False,
                            do_target_align=True,
                            images_weak=unlabel_images_weak,
                            strong_hflip=unlabel_hflip,
                            force_gt_supervision=False,
                            force_pseudo_supervision=self.use_pseudo_supervision,
                        )
                        merged_dict = self._merge_loss_dict(merged_dict, unlabel_loss_dict)
                        total_loss = total_loss + unlabel_loss

                    merged_dict["total_loss"] = float(total_loss.detach().item())
                    merged_dict = self._finalize_loss_dict(merged_dict, total_batch_size)
                    return total_loss, merged_dict

                if self.scaler is not None:
                    with torch.amp.autocast("cuda"):
                        loss, loss_dict = _forward_dual_stream()
                        loss = loss / accum
                else:
                    loss, loss_dict = _forward_dual_stream()
                    loss = loss / accum

                metrics_n = total_batch_size
            else:
                images, targets, image_paths, images_weak, strong_hflip = self._move_batch_to_device(batch)

                need_teacher = do_source_align or (
                    do_target_align and (self.use_target_alignment or self.pseudo_hard_mask_gate_enabled)
                )
                if self.scaler is not None:
                    with torch.amp.autocast("cuda"):
                        loss, loss_dict = self._forward_and_loss(
                            images,
                            targets,
                            image_paths,
                            epoch,
                            need_teacher_feat=need_teacher,
                            do_source_align=do_source_align,
                            do_target_align=do_target_align,
                            images_weak=images_weak,
                            strong_hflip=strong_hflip,
                        )
                        loss = loss / accum
                else:
                    loss, loss_dict = self._forward_and_loss(
                        images,
                        targets,
                        image_paths,
                        epoch,
                        need_teacher_feat=need_teacher,
                        do_source_align=do_source_align,
                        do_target_align=do_target_align,
                        images_weak=images_weak,
                        strong_hflip=strong_hflip,
                    )
                    loss = loss / accum

                metrics_n = int(images.size(0))

            # === Backward ===
            if self.scaler is not None:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()

            if (batch_idx + 1) % accum == 0:
                if self.scaler is not None:
                    self.scaler.unscale_(self.optimizer)
                self.gradient_clipper(self.model.get_trainable_parameters())
                if self.scaler is not None:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    self.optimizer.step()
                if self.student_ema is not None and self._ema_target is not None:
                    self.student_ema.update(self._ema_target)
                self.optimizer.zero_grad()

            meters.update(loss_dict, n=metrics_n)

            if (batch_idx + 1) % self.config["output"]["log_freq"] == 0:
                elapsed = time.time() - start_time
                eta = elapsed / (batch_idx + 1) * (num_batches - batch_idx - 1)
                self.logger.info(
                    f"Epoch [{epoch}][{batch_idx + 1}/{num_batches}] "
                    f"Loss: {loss_dict.get('total_loss', 0):.4f} "
                    f"({loss_dict.get('total_loss_per_img', 0):.4f}/img) "
                    f"LR: {self.optimizer.param_groups[0]['lr']:.6f} "
                    f"ETA: {format_time(eta)}"
                )

        return meters.get_averages()

    # ------------------------------------------------------------------
    # Core loss computation (mirrors run_step_full_semisup)
    # ------------------------------------------------------------------
    def _forward_and_loss(
        self,
        images: torch.Tensor,
        targets: Dict[str, Any],
        image_paths: List[str],
        epoch: int,
        need_teacher_feat: bool = False,
        do_source_align: bool = False,
        do_target_align: bool = False,
        images_weak: Optional[torch.Tensor] = None,
        strong_hflip: Optional[torch.Tensor] = None,
        force_gt_supervision: Optional[bool] = None,
        force_pseudo_supervision: Optional[bool] = None,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute losses following the DINO Teacher pattern.

        1. Convert MTKD targets → YOLO flat-tensor batch.
        2. Run the student **once** in train mode (``forward_train``).
        3. Stack detection, alignment, and pseudo-label losses.

        Returns:
            total_loss:  Scalar tensor for ``.backward()``.
            loss_dict:   Dict of detached loss values for logging.
        """
        loss_dict: Dict[str, float] = {}
        losses: List[torch.Tensor] = []

        # ---- Convert targets to YOLO batch format ----
        gt_batch = targets_to_yolo_batch(
            targets,
            self.device,
            box_dim=self.student_box_dim,
        )

        # ---- Build pseudo-label batch (full stage only) ----
        pseudo_batch = None
        use_gt_supervision = (
            self.use_gt_supervision if force_gt_supervision is None else bool(force_gt_supervision)
        )
        use_pseudo_supervision = (
            self.use_pseudo_supervision
            if force_pseudo_supervision is None
            else bool(force_pseudo_supervision)
        )

        # ---- Single student forward (train mode) + DINO teacher ----
        # ALIGN_EASY_ONLY: feed un-augmented images to the DINO teacher so
        # that alignment targets are free of augmentation artefacts.  The
        # dataset must supply ``batch["images_weak"]`` for this to work.
        teacher_images_align = images
        if self.align_easy_only:
            # images_weak is injected by train_epoch when available
            weak_images = images_weak
            if weak_images is None and hasattr(self, "_current_images_weak"):
                weak_images = self._current_images_weak
            if weak_images is not None:
                if weak_images.shape[0] == images.shape[0]:
                    teacher_images_align = weak_images
                else:
                    # Guard against stale cached weak-view batches (e.g. when
                    # switching from train to val with different last-batch size).
                    self.logger.warning(
                        "align_easy_only weak-view batch mismatch (weak=%d, strong=%d); "
                        "fallback to strong images for teacher alignment.",
                        int(weak_images.shape[0]),
                        int(images.shape[0]),
                    )

        # ---- Build pseudo-label batch (full stage only) ----
        # Priority:
        # 1) offline pseudo-label files
        # 2) online frozen wheat-teacher predictions
        if do_target_align and use_pseudo_supervision:
            hflip_mask = None
            hflip_src = strong_hflip
            if hflip_src is None and hasattr(self, "_current_strong_hflip"):
                hflip_src = self._current_strong_hflip
            if hflip_src is not None:
                if isinstance(hflip_src, torch.Tensor):
                    hflip_mask = hflip_src.detach().cpu().tolist()
                else:
                    hflip_mask = list(hflip_src)

            if self.use_offline_pseudo and self.pseudo_labels is not None and image_paths:
                stems = [Path(p).stem for p in image_paths]
                pseudo_batch = build_yolo_batch_from_pseudo(
                    self.pseudo_labels, stems, self.device,
                    horizontal_flip_mask=hflip_mask,
                )
            elif self.use_online_teacher_pseudo:
                # Keep online pseudo in the same geometric space as student
                # predictions (student sees ``images`` with strong aug).
                teacher_pred = self.model.get_wheat_teacher_predictions(images)
                pseudo_batch = self._build_yolo_batch_from_teacher_output(teacher_pred)

        out = self.model.forward_train(
            images,
            gt_yolo_batch=gt_batch,
            compute_dino=need_teacher_feat,
            teacher_images=teacher_images_align if need_teacher_feat else None,
        )

        pseudo_gate_mask = None
        if (
            self.pseudo_hard_mask_gate_enabled
            and do_target_align
            and use_pseudo_supervision
            and pseudo_batch is not None
            and pseudo_batch["bboxes"].shape[0] > 0
        ):
            student_feat_for_gate = out.get("student_spatial_feat")
            dino_feat_for_gate = out.get("dino_features")
            if (
                student_feat_for_gate is not None
                and dino_feat_for_gate is not None
                and hasattr(self.model, "build_pattern_align_mask")
            ):
                pseudo_gate_mask, pseudo_gate_stats = self._build_pattern_align_mask_for_batch(
                    student_feat_for_gate,
                    dino_feat_for_gate,
                    image_paths,
                    images,
                    self.pattern_align_target_prior_cfg,
                    use_bypass=self.pattern_align_use_dino_bypass,
                    hard_mask=True,
                    hard_mask_threshold=self.pattern_align_hard_mask_threshold,
                    detach=True,
                )
                for key, val in pseudo_gate_stats.items():
                    loss_dict[f"{key}_pseudo_gate"] = float(val)
                pseudo_batch, gate_stats = self._gate_pseudo_batch_by_hard_mask(
                    pseudo_batch,
                    pseudo_gate_mask,
                )
                for key, val in gate_stats.items():
                    loss_dict[key] = float(val)

        # ----- Add align head params to optimizer if just lazily created -----
        if getattr(self.model, "_align_head_just_created", False):
            self.optimizer.add_param_group({
                "params": list(self.model.align_head.parameters()),
                "lr": self.config["training"]["learning_rate"],
            })
            self.model._align_head_just_created = False
            self.logger.info("Added alignment head parameters to optimizer")
        if getattr(self.model, "_student_prior_head_just_created", False):
            self.optimizer.add_param_group({
                "params": list(self.model.student_prior_head.parameters()),
                "lr": self.config["training"]["learning_rate"],
            })
            self.model._student_prior_head_just_created = False
            self.logger.info("Added student prior head parameters to optimizer")

        # ----- Detection loss (GT supervision) -----
        if use_gt_supervision:
            det_loss = out["det_loss"]       # already = (box+cls+dfl)*B
            det_items = out["det_loss_items"]  # detached [box, cls, dfl]
            losses.append(det_loss)
            loss_dict["loss_det"] = det_loss.item()
            if isinstance(det_items, torch.Tensor) and det_items.numel() >= 1:
                loss_dict["loss_det_box"] = det_items[0].item()
            if isinstance(det_items, torch.Tensor) and det_items.numel() >= 2:
                loss_dict["loss_det_cls"] = det_items[1].item()
            if isinstance(det_items, torch.Tensor) and det_items.numel() >= 3:
                loss_dict["loss_det_dfl"] = det_items[2].item()
            if isinstance(det_items, torch.Tensor) and det_items.numel() >= 4:
                loss_dict["loss_det_angle"] = det_items[3].item()

        # ----- Feature alignment (source) -----
        if do_source_align:
            student_feat = out.get("student_spatial_feat")
            dino_feat = out.get("dino_features")
            if student_feat is not None and dino_feat is not None:
                pattern_mask_for_align = None
                if self.pattern_align_enabled and hasattr(self.model, "build_pattern_align_mask"):
                    pattern_mask_for_align, pattern_stats = self._build_pattern_align_mask_for_batch(
                        student_feat,
                        dino_feat,
                        image_paths,
                        images,
                        self.pattern_align_source_prior_cfg,
                        use_bypass=False,
                        hard_mask=self.pattern_align_hard_mask,
                        hard_mask_threshold=self.pattern_align_hard_mask_threshold,
                        detach=self.pattern_align_detach_mask,
                    )
                    for key, val in pattern_stats.items():
                        loss_dict[key] = float(val)

                prior_mask_for_align = None
                if getattr(self.model, "use_stomata_prior", False):
                    prior_mode = str(getattr(self.model, "prior_head_mode", "origin"))
                    prior_out = self.model.predict_stomata_prior(
                        dino_features=dino_feat,
                        student_spatial_feat=student_feat,
                        output_hw=dino_feat.shape[2:],
                    )
                    gt_prior_mask = self._build_patch_prior_mask(
                        targets,
                        dino_feat.shape[2:],
                    )
                    teacher_prior_target = None
                    if prior_mode == "origin":
                        teacher_prior_target = gt_prior_mask
                    else:
                        teacher_prior_target = self._merge_prior_targets(
                            gt_prior_mask,
                            pattern_mask_for_align,
                        )

                    if prior_out is not None and teacher_prior_target is not None:
                        pos_weight = torch.tensor(
                            [self.prior_mask_pos_w],
                            dtype=prior_out["logits"].dtype,
                            device=prior_out["logits"].device,
                        )
                        prior_loss = F.binary_cross_entropy_with_logits(
                            prior_out["logits"],
                            teacher_prior_target,
                            pos_weight=pos_weight,
                        )
                        weighted_prior = prior_loss * self.prior_mask_loss_w
                        losses.append(weighted_prior)
                        loss_dict["loss_prior_mask"] = prior_loss.item()
                        loss_dict["prior_mask_mean"] = prior_out["prob"].mean().item()
                        loss_dict["prior_teacher_coverage"] = teacher_prior_target.mean().item()
                        if (
                            self.prior_propagation_loss_w > 0.0
                            and isinstance(prior_out.get("propagation_logits"), torch.Tensor)
                        ):
                            prop_loss = F.binary_cross_entropy_with_logits(
                                prior_out["propagation_logits"],
                                teacher_prior_target,
                                pos_weight=pos_weight,
                            )
                            losses.append(prop_loss * self.prior_propagation_loss_w)
                            loss_dict["loss_prior_propagation"] = prop_loss.item()
                            if isinstance(prior_out.get("propagation_prob"), torch.Tensor):
                                loss_dict["prior_propagation_mean"] = prior_out["propagation_prob"].mean().item()
                        if self.prior_periodic_loss_w > 0.0:
                            periodic_loss = self._compute_prior_periodic_loss(
                                prior_out["prob"],
                                teacher_prior_target,
                            )
                            losses.append(periodic_loss * self.prior_periodic_loss_w)
                            loss_dict["loss_prior_periodic"] = periodic_loss.item()
                        if gt_prior_mask is not None:
                            loss_dict["prior_gt_coverage"] = gt_prior_mask.mean().item()
                        if pattern_mask_for_align is not None:
                            loss_dict["prior_pattern_coverage"] = pattern_mask_for_align.mean().item()
                        prior_mask_for_align = prior_out["prob"]

                if prior_mask_for_align is not None and pattern_mask_for_align is not None:
                    prior_mask_for_align = torch.maximum(prior_mask_for_align, pattern_mask_for_align)
                elif pattern_mask_for_align is not None:
                    prior_mask_for_align = pattern_mask_for_align

                align_loss = self.model.compute_align_loss(
                    student_feat,
                    dino_feat,
                    prior_mask=prior_mask_for_align,
                )
                weighted = align_loss * self.feature_align_w
                losses.append(weighted)
                loss_dict["loss_align"] = align_loss.item()

        # ----- Target feature alignment (full stage) -----
        # Mirrors DINO Teacher's ``loss_align_target``.  In DINO_Teacher this
        # is computed on *unlabeled target-domain* images with a separate
        # weight.  In MTKD (single-dataset), the same images serve both
        # roles, so we add a second weighted alignment signal in full stage to
        # guide the student backbone toward the DINO teacher space on the
        # target distribution.
        if self.use_target_alignment and do_target_align:
            student_feat = out.get("student_spatial_feat")
            dino_feat = out.get("dino_features")
            if student_feat is not None and dino_feat is not None:
                target_mask_for_align = None
                if (
                    self.pattern_align_enabled
                    and self.pattern_align_on_target
                    and hasattr(self.model, "build_pattern_align_mask")
                ):
                    target_mask_for_align, target_pattern_stats = self._build_pattern_align_mask_for_batch(
                        student_feat,
                        dino_feat,
                        image_paths,
                        images,
                        self.pattern_align_target_prior_cfg,
                        use_bypass=self.pattern_align_use_dino_bypass,
                        hard_mask=self.pattern_align_hard_mask,
                        hard_mask_threshold=self.pattern_align_hard_mask_threshold,
                        detach=self.pattern_align_detach_mask,
                    )
                    for key, val in target_pattern_stats.items():
                        loss_dict[f"{key}_target"] = float(val)

                if (
                    getattr(self.model, "use_stomata_prior", False)
                    and str(getattr(self.model, "prior_head_mode", "origin")) == "freq_adaption"
                ):
                    target_prior_out = self.model.predict_stomata_prior(
                        dino_features=dino_feat,
                        student_spatial_feat=student_feat,
                        output_hw=dino_feat.shape[2:],
                    )
                    teacher_prior_target = self._merge_prior_targets(target_mask_for_align)
                    if target_prior_out is not None and teacher_prior_target is not None:
                        pos_weight = torch.tensor(
                            [self.prior_mask_pos_w],
                            dtype=target_prior_out["logits"].dtype,
                            device=target_prior_out["logits"].device,
                        )
                        prior_loss_target = F.binary_cross_entropy_with_logits(
                            target_prior_out["logits"],
                            teacher_prior_target,
                            pos_weight=pos_weight,
                        )
                        weighted_prior_target = prior_loss_target * self.prior_mask_loss_w_target
                        losses.append(weighted_prior_target)
                        loss_dict["loss_prior_mask_target"] = prior_loss_target.item()
                        loss_dict["prior_mask_mean_target"] = target_prior_out["prob"].mean().item()
                        loss_dict["prior_teacher_coverage_target"] = teacher_prior_target.mean().item()
                        if (
                            self.prior_propagation_loss_w > 0.0
                            and isinstance(target_prior_out.get("propagation_logits"), torch.Tensor)
                        ):
                            prop_loss_target = F.binary_cross_entropy_with_logits(
                                target_prior_out["propagation_logits"],
                                teacher_prior_target,
                                pos_weight=pos_weight,
                            )
                            losses.append(prop_loss_target * self.prior_propagation_loss_w)
                            loss_dict["loss_prior_propagation_target"] = prop_loss_target.item()
                            if isinstance(target_prior_out.get("propagation_prob"), torch.Tensor):
                                loss_dict["prior_propagation_mean_target"] = target_prior_out["propagation_prob"].mean().item()
                        if self.prior_periodic_loss_w > 0.0:
                            periodic_loss_target = self._compute_prior_periodic_loss(
                                target_prior_out["prob"],
                                teacher_prior_target,
                            )
                            losses.append(periodic_loss_target * self.prior_periodic_loss_w)
                            loss_dict["loss_prior_periodic_target"] = periodic_loss_target.item()
                        if target_mask_for_align is not None:
                            target_mask_for_align = torch.maximum(
                                target_mask_for_align,
                                target_prior_out["prob"],
                            )
                        else:
                            target_mask_for_align = target_prior_out["prob"]

                gate_distill_loss, gate_distill_stats = self._compute_student_gate_distill_loss(target_mask_for_align)
                if gate_distill_loss is not None:
                    losses.append(gate_distill_loss * self.student_gate_distill_w)
                    loss_dict["loss_student_gate_distill"] = float(gate_distill_loss.item())
                    for key, val in gate_distill_stats.items():
                        loss_dict[key] = float(val)

                # Reuse align_loss if already computed, else recompute
                if "loss_align" not in loss_dict or target_mask_for_align is not None:
                    align_loss_target = self.model.compute_align_loss(
                        student_feat,
                        dino_feat,
                        prior_mask=target_mask_for_align,
                    )
                else:
                    align_loss_target = align_loss  # same data → same value
                weighted_target = align_loss_target * self.feature_align_w_target
                losses.append(weighted_target)
                loss_dict["loss_align_target"] = align_loss_target.item()

        # ----- Pseudo-label loss (target) -----
        # Computed separately so we can optionally zero out box regression.
        # Reuses the same raw_preds from the single student forward pass.
        if (
            use_pseudo_supervision
            and do_target_align
            and pseudo_batch is not None
            and pseudo_batch["bboxes"].shape[0] > 0
        ):
            if self._last_area_prior_relabel_count > 0:
                loss_dict["pseudo_area_relabel_count"] = float(self._last_area_prior_relabel_count)
            pseudo_box_count = int(pseudo_batch["bboxes"].shape[0])
            loss_dict["pseudo_box_count"] = float(pseudo_box_count)
            cls_tensor = pseudo_batch.get("cls")
            if isinstance(cls_tensor, torch.Tensor) and cls_tensor.numel() > 0:
                cls_ids = cls_tensor.view(-1).long()
                for cls_id in range(max(1, self.num_classes)):
                    loss_dict[f"pseudo_box_count_c{cls_id}"] = float(
                        (cls_ids == cls_id).sum().item()
                    )
            self.model.student._ensure_criterion()
            criterion = self.model.student._criterion

            if self.prediction_align_mode == "ultralytics":
                # Use the same criterion/assigner route as Ultralytics GT training.
                weighted_pseudo, pl_metrics = self.pred_align_loss(
                    raw_preds=out["raw_preds"],
                    pseudo_batch=pseudo_batch,
                    criterion=criterion,
                    unsup_weight=self.unsup_loss_w,
                    zero_box_dfl=self.zero_pseudo_box_reg,
                )
                losses.append(weighted_pseudo)
                for k, v in pl_metrics.items():
                    if isinstance(v, torch.Tensor):
                        loss_dict[k] = v.item()
            else:
                if self.zero_pseudo_box_reg:
                    # Temporarily zero box & DFL gains; keep cls only
                    hyp_refs = self.pred_align_loss._hyp_refs_for_box_dfl(criterion)
                    if hyp_refs:
                        originals = [(h, h.box, h.dfl) for h in hyp_refs]
                        for h in hyp_refs:
                            h.box = 0.0
                            h.dfl = 0.0
                        try:
                            pl_loss, pl_items = criterion(out["raw_preds"], pseudo_batch)
                        finally:
                            for h, orig_box, orig_dfl in originals:
                                h.box = orig_box
                                h.dfl = orig_dfl
                    else:
                        pl_loss, pl_items = criterion(out["raw_preds"], pseudo_batch)
                else:
                    pl_loss, pl_items = criterion(out["raw_preds"], pseudo_batch)

                # ultralytics >=8.4 returns loss as [box, cls, dfl]; sum to scalar
                if pl_loss.ndim > 0 and pl_loss.numel() > 1:
                    pl_loss = pl_loss.sum()

                weighted_pseudo = pl_loss * self.unsup_loss_w
                losses.append(weighted_pseudo)
                loss_dict["loss_pseudo"] = pl_loss.item()
                loss_dict["loss_pseudo_weighted"] = weighted_pseudo.item()
                if isinstance(pl_items, torch.Tensor) and pl_items.numel() >= 1:
                    loss_dict["loss_pseudo_box"] = pl_items[0].item()
                if isinstance(pl_items, torch.Tensor) and pl_items.numel() >= 2:
                    loss_dict["loss_pseudo_cls"] = pl_items[1].item()
                if isinstance(pl_items, torch.Tensor) and pl_items.numel() >= 3:
                    loss_dict["loss_pseudo_dfl"] = pl_items[2].item()
                if isinstance(pl_items, torch.Tensor) and pl_items.numel() >= 4:
                    loss_dict["loss_pseudo_angle"] = pl_items[3].item()

        # ----- Unified detection-view metrics -----
        # Both GT and pseudo branches use the same Ultralytics detection
        # criterion; keep explicit aliases so pseudo-only runs are readable.
        gt_det = float(loss_dict.get("loss_det", 0.0))
        pseudo_det = float(loss_dict.get("loss_pseudo", 0.0))
        pseudo_det_weighted = float(loss_dict.get("loss_pseudo_weighted", 0.0))
        loss_dict["loss_det_gt"] = gt_det
        loss_dict["loss_det_pseudo"] = pseudo_det
        loss_dict["loss_det_pseudo_weighted"] = pseudo_det_weighted
        loss_dict["loss_det_total"] = gt_det + pseudo_det_weighted

        # For pseudo-only training, keep legacy `loss_det` field populated so
        # breakdown logs no longer misleadingly show det=0.0.
        if "loss_det" not in loss_dict and "loss_pseudo_weighted" in loss_dict:
            loss_dict["loss_det"] = pseudo_det_weighted

        if "loss_det_box" not in loss_dict and "loss_pseudo_box" in loss_dict:
            loss_dict["loss_det_box"] = loss_dict["loss_pseudo_box"]
        if "loss_det_cls" not in loss_dict and "loss_pseudo_cls" in loss_dict:
            loss_dict["loss_det_cls"] = loss_dict["loss_pseudo_cls"]
        if "loss_det_dfl" not in loss_dict and "loss_pseudo_dfl" in loss_dict:
            loss_dict["loss_det_dfl"] = loss_dict["loss_pseudo_dfl"]
        if "loss_det_angle" not in loss_dict and "loss_pseudo_angle" in loss_dict:
            loss_dict["loss_det_angle"] = loss_dict["loss_pseudo_angle"]

        if self.weight_anchor_enabled and self.weight_anchor_lambda > 0.0 and self._student_anchor_params:
            anchor_loss = images.new_tensor(0.0)
            counted = 0
            for name, param in self.model.student.named_parameters():
                if not param.requires_grad:
                    continue
                anchor = self._student_anchor_params.get(name)
                if anchor is None:
                    continue
                anchor_loss = anchor_loss + torch.sum((param - anchor) ** 2)
                counted += 1
            if counted > 0:
                weighted_anchor = anchor_loss * self.weight_anchor_lambda
                losses.append(weighted_anchor)
                loss_dict["loss_weight_anchor"] = float(anchor_loss.detach().item())
                loss_dict["loss_weight_anchor_weighted"] = float(weighted_anchor.detach().item())

        if len(losses) == 0:
            total_loss = images.new_tensor(0.0)
        else:
            total_loss = sum(losses)  # type: ignore[arg-type]
        loss_dict["total_loss"] = total_loss.item()

        # Ultralytics criterion returns batch-scaled scalar losses; keep those
        # for optimization but also log per-image normalized values.
        batch_size = float(max(1, images.size(0)))
        loss_dict["total_loss_per_img"] = loss_dict["total_loss"] / batch_size
        if "loss_det" in loss_dict:
            loss_dict["loss_det_per_img"] = loss_dict["loss_det"] / batch_size
        if "loss_det_total" in loss_dict:
            loss_dict["loss_det_total_per_img"] = loss_dict["loss_det_total"] / batch_size
        if "loss_det_pseudo_weighted" in loss_dict:
            loss_dict["loss_det_pseudo_weighted_per_img"] = (
                loss_dict["loss_det_pseudo_weighted"] / batch_size
            )
        if "loss_pseudo" in loss_dict:
            loss_dict["loss_pseudo_per_img"] = loss_dict["loss_pseudo"] / batch_size
            # Explicit alias: raw Ultralytics pseudo criterion (per-image scale).
            loss_dict["loss_pseudo_ultra_per_img"] = loss_dict["loss_pseudo_per_img"]
        if "loss_pseudo_weighted" in loss_dict:
            loss_dict["loss_pseudo_weighted_per_img"] = (
                loss_dict["loss_pseudo_weighted"] / batch_size
            )

        return total_loss, loss_dict

    def _build_yolo_batch_from_teacher_output(
        self,
        teacher_pred: Optional[Dict[str, torch.Tensor]],
    ) -> Optional[Dict[str, torch.Tensor]]:
        """
        Convert frozen teacher predictions to YOLO target-batch format.

        Expected teacher dict keys: ``boxes`` [B,N,4] (normalised cxcywh),
        ``scores`` [B,N], ``labels`` [B,N], optional ``valid_mask`` [B,N].
        """
        self._last_area_prior_relabel_count = 0
        if teacher_pred is None:
            return None

        boxes = teacher_pred.get("boxes")
        scores = teacher_pred.get("scores")
        labels = teacher_pred.get("labels")
        if boxes is None or scores is None or labels is None:
            return None

        if boxes.shape[-1] < self.student_box_dim:
            pad = torch.zeros(
                (*boxes.shape[:-1], self.student_box_dim - boxes.shape[-1]),
                dtype=boxes.dtype,
                device=boxes.device,
            )
            boxes = torch.cat([boxes, pad], dim=-1)
        elif boxes.shape[-1] > self.student_box_dim:
            boxes = boxes[..., : self.student_box_dim]

        if self.pl_class_score_thresholds:
            labels_long = labels.long()
            default_threshold = (
                float(self.pl_score_threshold)
                if self.pl_score_threshold is not None
                else 0.0
            )
            cls_threshold = torch.full_like(scores, default_threshold)
            for cls_id, threshold in self.pl_class_score_thresholds.items():
                cls_threshold[labels_long == int(cls_id)] = float(threshold)
            valid_mask = scores >= cls_threshold
        elif self.pl_score_threshold is not None:
            valid_mask = scores >= float(self.pl_score_threshold)
        else:
            valid_mask = torch.ones_like(scores, dtype=torch.bool)
        if "valid_mask" in teacher_pred and teacher_pred["valid_mask"] is not None:
            valid_mask = valid_mask & teacher_pred["valid_mask"].bool()

        labels_to_use = labels.clone()
        if self.pl_area_class_prior:
            area = boxes[..., 2] * boxes[..., 3]
            relabel_mask = valid_mask.clone()
            source_class = self.pl_area_class_prior.get("source_class", None)
            if source_class is not None:
                relabel_mask = relabel_mask & (labels_to_use.long() == int(source_class))
            min_area = self.pl_area_class_prior.get("min_area", None)
            max_area = self.pl_area_class_prior.get("max_area", None)
            if min_area is not None:
                relabel_mask = relabel_mask & (area >= float(min_area))
            if max_area is not None:
                relabel_mask = relabel_mask & (area <= float(max_area))
            if relabel_mask.any():
                labels_to_use[relabel_mask] = int(self.pl_area_class_prior["target_class"])
                self._last_area_prior_relabel_count = int(relabel_mask.sum().item())

        all_idx: List[int] = []
        all_cls: List[float] = []
        all_bboxes: List[List[float]] = []

        for b in range(boxes.shape[0]):
            mask = valid_mask[b]
            if mask.ndim == 0:
                continue
            b_boxes = boxes[b][mask]
            b_labels = labels_to_use[b][mask]
            if b_boxes.numel() == 0:
                continue

            all_idx.extend([b] * b_boxes.shape[0])
            all_cls.extend(b_labels.detach().float().cpu().tolist())
            all_bboxes.extend(b_boxes.detach().float().cpu().tolist())

        if not all_idx:
            return None

        return {
            "batch_idx": torch.tensor(all_idx, dtype=torch.float32, device=self.device),
            "cls": torch.tensor(all_cls, dtype=torch.float32, device=self.device).unsqueeze(-1),
            "bboxes": torch.tensor(all_bboxes, dtype=torch.float32, device=self.device),
        }

    @staticmethod
    def _map_to_uint8(values: np.ndarray) -> np.ndarray:
        values = values.astype(np.float32)
        lo = float(np.percentile(values, 5))
        hi = float(np.percentile(values, 95))
        if hi <= lo:
            hi = lo + 1e-6
        norm = np.clip((values - lo) / (hi - lo), 0.0, 1.0)
        return (norm * 255.0).astype(np.uint8)

    @staticmethod
    def _map_to_color_pil(values: np.ndarray, size: Tuple[int, int]) -> Image.Image:
        values = values.astype(np.float32)
        lo = float(np.percentile(values, 5))
        hi = float(np.percentile(values, 95))
        if hi <= lo:
            hi = lo + 1e-6
        norm = np.clip((values - lo) / (hi - lo), 0.0, 1.0)
        rgba = matplotlib.colormaps.get_cmap("turbo")(norm)
        rgb = (rgba[..., :3] * 255.0).astype(np.uint8)
        image = Image.fromarray(rgb, mode="RGB")
        return image.resize(size, resample=Image.Resampling.BILINEAR)

    @staticmethod
    def _tensor_image_to_pil(image_tensor: torch.Tensor) -> Image.Image:
        image = (
            image_tensor.detach()
            .cpu()
            .float()
            .clamp(0.0, 1.0)
            .permute(1, 2, 0)
            .numpy()
        )
        return Image.fromarray((image * 255.0).astype(np.uint8), mode="RGB")

    @classmethod
    def _feature_response_to_pil(
        cls,
        feat: torch.Tensor,
        size: Tuple[int, int],
    ) -> Image.Image:
        response = torch.linalg.norm(feat.detach().float(), dim=0).cpu().numpy()
        return cls._map_to_color_pil(response, size)

    @classmethod
    def _mask_to_pil(
        cls,
        mask: torch.Tensor,
        size: Tuple[int, int],
    ) -> Image.Image:
        if mask.ndim == 3:
            mask = mask.squeeze(0)
        return cls._map_to_color_pil(mask.detach().float().cpu().numpy(), size)

    @classmethod
    def _overlay_mask_on_image(
        cls,
        image: Image.Image,
        mask: torch.Tensor,
        *,
        alpha: float = 0.55,
    ) -> Image.Image:
        if mask.ndim == 3:
            mask = mask.squeeze(0)
        mask_np = mask.detach().float().cpu().numpy()
        mask_img = cls._mask_to_pil(mask, image.size).convert("RGBA")
        alpha_img = Image.fromarray(
            cls._map_to_uint8(mask_np),
            mode="L",
        ).resize(image.size, resample=Image.Resampling.BILINEAR)
        base = image.convert("RGBA")
        mask_img.putalpha(alpha_img.point(lambda x: int(alpha * x)))
        return Image.alpha_composite(base, mask_img).convert("RGB")

    @staticmethod
    def _find_detect_module(det_model: nn.Module) -> nn.Module:
        detect_module = None
        modules = getattr(det_model, "model", None)
        if modules is not None and len(modules) > 0:
            for module in reversed(modules):
                if module.__class__.__name__.lower() in {"detect", "obb"}:
                    detect_module = module
                    break
        if detect_module is None:
            raise RuntimeError("Could not locate Detect/OBB module for debug export")
        return detect_module

    @staticmethod
    def _normalized_box_to_poly(box: np.ndarray, image_size: Tuple[int, int]) -> List[Tuple[float, float]]:
        w_img, h_img = image_size
        if box.shape[0] >= 5:
            cx, cy, bw, bh, angle = [float(v) for v in box[:5]]
            cx *= w_img
            cy *= h_img
            bw *= w_img
            bh *= h_img
            cos_a = math.cos(angle)
            sin_a = math.sin(angle)
            corners = [
                (-bw / 2.0, -bh / 2.0),
                (bw / 2.0, -bh / 2.0),
                (bw / 2.0, bh / 2.0),
                (-bw / 2.0, bh / 2.0),
            ]
            points: List[Tuple[float, float]] = []
            for px, py in corners:
                rx = cx + px * cos_a - py * sin_a
                ry = cy + px * sin_a + py * cos_a
                points.append((rx, ry))
            return points

        cx, cy, bw, bh = [float(v) for v in box[:4]]
        cx *= w_img
        cy *= h_img
        bw *= w_img
        bh *= h_img
        x1 = cx - bw / 2.0
        y1 = cy - bh / 2.0
        x2 = cx + bw / 2.0
        y2 = cy + bh / 2.0
        return [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]

    @staticmethod
    def _get_debug_font(size: int) -> ImageFont.ImageFont:
        try:
            font_path = matplotlib.font_manager.findfont("DejaVu Sans")
            return ImageFont.truetype(font_path, size=max(1, int(size)))
        except Exception:
            return ImageFont.load_default()

    @classmethod
    def _draw_debug_predictions(
        cls,
        image: Image.Image,
        pred: Dict[str, torch.Tensor],
        *,
        color: Tuple[int, int, int],
        title: Optional[str] = None,
        score_threshold: float = 0.1,
        max_boxes: int = 300,
    ) -> Image.Image:
        canvas = image.copy()
        draw = ImageDraw.Draw(canvas)
        boxes = pred.get("boxes")
        scores = pred.get("scores")
        labels = pred.get("labels")
        if not isinstance(boxes, torch.Tensor) or not isinstance(scores, torch.Tensor):
            return canvas
        if boxes.ndim == 3:
            boxes = boxes[0]
        if scores.ndim == 2:
            scores = scores[0]
        if isinstance(labels, torch.Tensor) and labels.ndim == 2:
            labels = labels[0]
        keep = scores >= float(score_threshold)
        if keep.ndim > 1:
            keep = keep.view(-1)
        boxes = boxes[keep][:max_boxes].detach().cpu().float().numpy()
        scores = scores[keep][:max_boxes].detach().cpu().float().numpy()
        labels_np = (
            labels[keep][:max_boxes].detach().cpu().long().numpy()
            if isinstance(labels, torch.Tensor)
            else np.zeros((boxes.shape[0],), dtype=np.int64)
        )
        for box, score, cls_id in zip(boxes, scores, labels_np):
            poly = cls._normalized_box_to_poly(np.asarray(box, dtype=np.float32), canvas.size)
            draw.line(poly + [poly[0]], fill=color, width=2)
            if title is not None:
                text = f"{int(cls_id)}:{float(score):.2f}"
                tx, ty = poly[0]
                font = cls._get_debug_font(18)
                draw.text(
                    (tx + 2, ty + 2),
                    text,
                    fill=color,
                    font=font,
                    stroke_width=2,
                    stroke_fill=(0, 0, 0),
                )
        return canvas

    def _predict_student_with_support(
        self,
        images: torch.Tensor,
        support_map: Optional[torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        student = self.model.student
        if support_map is None:
            return student.get_detection_output(images)
        detect_module = self._find_detect_module(student.det_model)
        level_name = str(getattr(self.model, "student_align_layer", getattr(student, "feature_level", "p4"))).lower()
        level_idx = getattr(student, "feature_level_to_idx", {"p3": 0, "p4": 1, "p5": 2}).get(level_name, 1)
        support = support_map.detach()
        gate_hard = bool(getattr(self.model, "prior_gate_hard", False))
        gate_threshold = float(getattr(self.model, "prior_gate_threshold", 0.5))
        bg_scale = getattr(self.model, "prior_gate_bg_scale", None)
        fg_scale = getattr(self.model, "prior_gate_fg_scale", None)
        gate_strength = float(getattr(self.model, "prior_gate_strength", 1.0))

        def _enhance_detect_input(_module, inputs):
            if not inputs:
                return None
            x = inputs[0]
            if not isinstance(x, (list, tuple)):
                return None
            feats = list(x)
            if level_idx >= len(feats):
                return None
            target = feats[level_idx]
            gate = F.interpolate(
                support.to(device=target.device, dtype=target.dtype),
                size=target.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
            if gate_hard:
                gate = (gate >= gate_threshold).to(dtype=target.dtype)
            if bg_scale is not None or fg_scale is not None:
                bg = float(1.0 if bg_scale is None else bg_scale)
                fg = float((1.0 + gate_strength) if fg_scale is None else fg_scale)
                scale = bg + (fg - bg) * gate
            else:
                scale = 1.0 + gate_strength * gate
            feats[level_idx] = target * scale
            return (tuple(feats),) if isinstance(x, tuple) else (feats,)

        handle = detect_module.register_forward_pre_hook(_enhance_detect_input)
        try:
            return student.get_detection_output(images)
        finally:
            handle.remove()

    @staticmethod
    def _make_tile(image: Image.Image, title: str, tile_size: int) -> Image.Image:
        canvas_h = tile_size + 42
        canvas = Image.new("RGB", (tile_size, canvas_h), color=(255, 255, 255))
        resized = image.resize((tile_size, tile_size), resample=Image.Resampling.BICUBIC)
        canvas.paste(resized, (0, 0))
        draw = ImageDraw.Draw(canvas)
        font = MTKDTrainerV2._get_debug_font(20)
        draw.text(
            (8, tile_size + 8),
            title,
            fill=(0, 0, 0),
            font=font,
            stroke_width=2,
            stroke_fill=(255, 255, 255),
        )
        return canvas

    @classmethod
    def _make_grid(cls, tiles: List[Image.Image], cols: int) -> Image.Image:
        if not tiles:
            raise ValueError("tiles must not be empty")
        cols = max(1, cols)
        tile_w, tile_h = tiles[0].size
        rows = int(math.ceil(len(tiles) / cols))
        panel = Image.new("RGB", (tile_w * cols, tile_h * rows), color=(245, 245, 245))
        for idx, tile in enumerate(tiles):
            x = (idx % cols) * tile_w
            y = (idx // cols) * tile_h
            panel.paste(tile, (x, y))
        return panel

    def _select_debug_batch(self) -> Optional[Tuple[Dict[str, Any], bool]]:
        if self.train_loader is None and self.val_loader is None:
            return None
        prefer_train = self.dual_stream or self.use_pseudo_supervision or self.use_target_alignment
        loader = self.train_loader if prefer_train and self.train_loader is not None else self.val_loader
        if loader is None:
            loader = self.train_loader
        if loader is None:
            return None

        batch = next(iter(loader))
        if (
            isinstance(batch, (tuple, list))
            and len(batch) == 2
            and isinstance(batch[0], dict)
            and isinstance(batch[1], dict)
        ):
            use_unlabeled = self.use_pseudo_supervision or self.use_target_alignment
            return (batch[1] if use_unlabeled else batch[0], use_unlabeled)
        if isinstance(batch, dict):
            return batch, False
        return None

    def _build_debug_pseudo_batch(
        self,
        images: torch.Tensor,
        image_paths: List[str],
        *,
        strong_hflip: Optional[torch.Tensor],
    ) -> Optional[Dict[str, torch.Tensor]]:
        if not self.use_pseudo_supervision:
            return None

        hflip_mask = None
        if strong_hflip is not None:
            hflip_mask = strong_hflip.detach().cpu().tolist()

        if self.use_offline_pseudo and self.pseudo_labels is not None and image_paths:
            stems = [Path(p).stem for p in image_paths]
            return build_yolo_batch_from_pseudo(
                self.pseudo_labels,
                stems,
                self.device,
                horizontal_flip_mask=hflip_mask,
            )
        if self.use_online_teacher_pseudo:
            teacher_pred = self.model.get_wheat_teacher_predictions(images)
            return self._build_yolo_batch_from_teacher_output(teacher_pred)
        return None

    def _save_debug_pseudo_label(
        self,
        pseudo_batch: Optional[Dict[str, torch.Tensor]],
        sample_index: int,
        save_path: str,
    ) -> None:
        path = Path(save_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        if pseudo_batch is None:
            path.write_text("", encoding="utf-8")
            return

        batch_idx = pseudo_batch.get("batch_idx")
        cls_tensor = pseudo_batch.get("cls")
        bboxes = pseudo_batch.get("bboxes")
        if not isinstance(batch_idx, torch.Tensor) or not isinstance(cls_tensor, torch.Tensor) or not isinstance(bboxes, torch.Tensor):
            path.write_text("", encoding="utf-8")
            return

        mask = batch_idx.long() == int(sample_index)
        if not bool(mask.any().item()):
            path.write_text("", encoding="utf-8")
            return

        cls_np = cls_tensor[mask].detach().cpu().view(-1).numpy()
        box_np = bboxes[mask].detach().cpu().numpy()
        lines: List[str] = []
        for cls_id, box in zip(cls_np.tolist(), box_np.tolist()):
            tokens = [str(int(round(float(cls_id))))] + [f"{float(v):.6f}" for v in box]
            lines.append(" ".join(tokens))
        path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")

    def _export_student_dino_bypass_debug_panels(
        self,
        image_paths: Sequence[str],
        epoch_dir: Path,
        epoch: int,
    ) -> bool:
        if not self.pattern_align_use_dino_bypass:
            return False
        if not image_paths:
            return False

        image_path = Path(str(image_paths[0])).expanduser()
        if not image_path.is_file():
            return False

        student = getattr(self.model, "student", None)
        export_fn = getattr(student, "export_ultralytics_pt", None)
        if student is None or export_fn is None:
            return False

        try:
            import train_dino_bypass_offline as dino_bypass
        except Exception as exc:
            self.logger.warning("Failed to import train_dino_bypass_offline for debug export: %s", exc)
            return False

        debug_input_dir = epoch_dir / "_student_dino_bypass_input"
        debug_input_dir.mkdir(parents=True, exist_ok=True)
        linked_image = debug_input_dir / image_path.name
        if not linked_image.exists():
            try:
                linked_image.symlink_to(image_path)
            except Exception:
                shutil.copy2(image_path, linked_image)

        weights_path = epoch_dir / "_student_debug_current.pt"
        num_classes = int(self.config.get("model", {}).get("num_classes", 1))
        class_names = {i: f"class_{i}" for i in range(num_classes)}
        with self._use_ema_student():
            export_fn(
                save_path=str(weights_path),
                num_classes=num_classes,
                class_names=class_names,
                epoch=epoch,
                best_fitness=float(self.best_score) if np.isfinite(self.best_score) else None,
            )

        cfg = dict(self._dino_bypass_cfg or {})
        cfg.update(
            {
                "input_dir": str(debug_input_dir),
                "output_dir": str(epoch_dir),
                "yolo_weights": str(weights_path),
                "num_samples": 1,
                "device": str(self.device),
                "export_pseudo_dir": None,
            }
        )
        cfg.setdefault("feature_level", str(getattr(self.model, "student_align_layer", "p3")))
        cfg_path = epoch_dir / "_student_dino_bypass_debug_config.json"
        cfg_path.write_text(json.dumps({"dino_bypass": cfg}, indent=2), encoding="utf-8")

        old_argv = list(sys.argv)
        try:
            sys.argv = ["train_dino_bypass_offline.py", "--config", str(cfg_path)]
            args = dino_bypass.parse_args()
        finally:
            sys.argv = old_argv

        dataset = type("_SingleImageDataset", (), {"files": [linked_image]})()
        try:
            dino_bypass.run_detect_test(args, str(args.device), dataset)
        except Exception as exc:
            self.logger.warning("Student dino_bypass debug export failed: %s", exc)
            return False

        self.logger.info(
            "Saved student dino_bypass debug panels for %s using %s",
            image_path.name,
            weights_path.name,
        )
        return True

    @torch.no_grad()
    def _export_epoch_debug(self, epoch: int) -> None:
        debug_pick = self._select_debug_batch()
        if debug_pick is None:
            return
        debug_batch, _ = debug_pick

        images, targets, image_paths, images_weak, strong_hflip = self._move_batch_to_device(debug_batch)
        if images.size(0) == 0:
            return

        teacher_images = images
        if self.align_easy_only and images_weak is not None and images_weak.shape[0] == images.shape[0]:
            teacher_images = images_weak

        epoch_dir = Path(self.debug_dir) / f"epoch_{epoch:04d}"
        epoch_dir.mkdir(parents=True, exist_ok=True)

        exported_student_bypass_panels = self._export_student_dino_bypass_debug_panels(image_paths, epoch_dir, epoch)
        if exported_student_bypass_panels:
            return

        with self._use_ema_student():
            self.model.eval()
            debug = self.model.get_alignment_debug(
                images[:1],
                teacher_images=teacher_images[:1],
            )

            student_feat = debug["student_spatial_feat"]
            dino_feat = debug["dino_features"]
            mask_for_display: Optional[torch.Tensor] = None
            student_prior_prob: Optional[torch.Tensor] = None
            native_gate_prob: Optional[torch.Tensor] = None
            get_gate_maps = getattr(getattr(self.model, "student", None), "get_support_gate_maps", None)
            if callable(get_gate_maps):
                gate_records = get_gate_maps()
                if gate_records and isinstance(gate_records[0].get("mask"), torch.Tensor):
                    native_gate_prob = gate_records[0]["mask"][:1]
            if getattr(self.model, "use_stomata_prior", False):
                prior_out = self.model.predict_stomata_prior(
                    dino_features=dino_feat,
                    student_spatial_feat=student_feat,
                    output_hw=dino_feat.shape[2:],
                )
                if prior_out is not None:
                    student_prior_prob = prior_out["prob"][:1]
            if self.pattern_align_enabled and hasattr(self.model, "build_pattern_align_mask"):
                mask_for_display, _ = self._build_pattern_align_mask_for_batch(
                    student_feat,
                    dino_feat,
                    image_paths,
                    images,
                    self.pattern_align_target_prior_cfg,
                    use_bypass=self.pattern_align_use_dino_bypass,
                    hard_mask=self.pattern_align_hard_mask,
                    hard_mask_threshold=self.pattern_align_hard_mask_threshold,
                    detach=True,
                )
            if mask_for_display is None:
                prior = self.model.predict_stomata_prior(
                    dino_features=dino_feat,
                    student_spatial_feat=student_feat,
                    output_hw=dino_feat.shape[2:],
                )
                if prior is not None:
                    mask_for_display = prior["prob"]

            if mask_for_display is None:
                return

            applied_support = mask_for_display[:1]
            if student_prior_prob is not None:
                applied_support = torch.maximum(applied_support, student_prior_prob)

            mask_small = F.interpolate(
                applied_support,
                size=student_feat.shape[2:],
                mode="bilinear",
                align_corners=False,
            )
            gate_hard = bool(getattr(self.model, "prior_gate_hard", False))
            gate_threshold = float(getattr(self.model, "prior_gate_threshold", 0.5))
            if gate_hard:
                mask_small = (mask_small >= gate_threshold).to(dtype=student_feat.dtype)
            bg_scale = getattr(self.model, "prior_gate_bg_scale", None)
            fg_scale = getattr(self.model, "prior_gate_fg_scale", None)
            gate_strength = float(getattr(self.model, "prior_gate_strength", 1.0))
            if bg_scale is not None or fg_scale is not None:
                bg = float(1.0 if bg_scale is None else bg_scale)
                fg = float((1.0 + gate_strength) if fg_scale is None else fg_scale)
                feat_scale = bg + (fg - bg) * mask_small
            else:
                feat_scale = 1.0 + gate_strength * mask_small
            after_feat = student_feat[:1] * feat_scale
            before_pred = self.model.student.get_detection_output(images[:1])
            after_pred = self._predict_student_with_support(images[:1], applied_support)

        original = self._tensor_image_to_pil(images[0])
        support_img = self._mask_to_pil(applied_support[0], original.size)
        support_overlay = self._overlay_mask_on_image(original, applied_support[0], alpha=0.55)
        native_gate_img = self._mask_to_pil(native_gate_prob[0], original.size) if native_gate_prob is not None else None
        before_img = self._feature_response_to_pil(student_feat[0], original.size)
        after_img = self._feature_response_to_pil(after_feat[0], original.size)
        before_pred_img = self._draw_debug_predictions(
            original,
            before_pred,
            color=(0, 245, 255),
            score_threshold=self.debug_pred_conf,
            max_boxes=self.debug_pred_max_boxes,
        )
        after_pred_img = self._draw_debug_predictions(
            original,
            after_pred,
            color=(255, 35, 210),
            score_threshold=self.debug_pred_conf,
            max_boxes=self.debug_pred_max_boxes,
        )

        tiles = [
            self._make_tile(original, "Origin Image", self.debug_tile_size),
            self._make_tile(support_overlay, "Pattern Support Overlay", self.debug_tile_size),
            self._make_tile(support_img, "Pattern Support Heatmap", self.debug_tile_size),
            self._make_tile(native_gate_img if native_gate_img is not None else original, "Student Native Gate", self.debug_tile_size),
            self._make_tile(before_img, "Student Feature Before", self.debug_tile_size),
            self._make_tile(after_img, "Student Feature After", self.debug_tile_size),
        ]
        feature_panel = self._make_grid(tiles, cols=3)
        prediction_tiles = [
            self._make_tile(before_pred_img, "Before Alignment Prediction", self.debug_tile_size),
            self._make_tile(after_pred_img, "After Alignment Prediction", self.debug_tile_size),
        ]
        prediction_panel = self._make_grid(prediction_tiles, cols=2)

        stem = Path(image_paths[0]).stem if image_paths else f"sample0_epoch{epoch:04d}"
        feature_panel_path = epoch_dir / f"{stem}_feature_panel.png"
        prediction_panel_path = epoch_dir / f"{stem}_prediction_panel.png"

        feature_panel.save(feature_panel_path)
        prediction_panel.save(prediction_panel_path)
        self.logger.info("Saved debug export: %s", feature_panel_path)

    # ------------------------------------------------------------------
    # Validate
    # ------------------------------------------------------------------
    @torch.no_grad()
    def validate(self, epoch: int) -> Dict[str, float]:
        if self.val_loader is None:
            return {}
        meters = AverageMeterDict()
        # In pseudo-only mode, include pseudo supervision in validation once
        # the full stage starts so model selection tracks detection behaviour instead
        # of alignment-only loss.
        val_do_target_align = (
            self.use_pseudo_supervision
            and epoch >= self.align_target_start_epoch
        )
        val_do_source_align = self.use_source_alignment
        with self._use_ema_student():
            self.model.eval()
            for batch in self.val_loader:
                images = batch["images"].to(self.device)
                # ALIGN_EASY_ONLY: ensure weak-view cache is from the current
                # validation batch, not stale training state.
                images_weak = None
                if self.align_easy_only and "images_weak" in batch:
                    images_weak = batch["images_weak"].to(self.device)
                strong_hflip = batch.get("strong_hflip")

                targets = {
                    k: (v.to(self.device) if isinstance(v, torch.Tensor) else v)
                    for k, v in batch.get("targets", {}).items()
                }
                image_paths: List[str] = batch.get("image_paths", [])
                # Validation uses the eval-mode forward for alignment metrics.
                # Detection loss is still computed (student in train mode internally).
                need_teacher_feat = val_do_source_align or (
                    val_do_target_align and self.use_target_alignment
                )
                _, loss_dict = self._forward_and_loss(
                    images, targets, image_paths, epoch,
                    need_teacher_feat=need_teacher_feat,
                    do_source_align=val_do_source_align,
                    do_target_align=val_do_target_align,
                    images_weak=images_weak,
                    strong_hflip=strong_hflip,
                )
                meters.update(loss_dict, n=images.size(0))
        return meters.get_averages()

    def _stage_name(self, epoch: int) -> str:
        """Human-readable training stage for the given epoch index."""
        if epoch < self.align_target_start_epoch:
            if self.use_source_alignment:
                return "source alignment (det + align)"
            return "detection warmup (source align disabled)"
        if self.use_source_alignment and self.use_target_alignment:
            return "full (source-align + target-align + pseudo-label)"
        if self.use_target_alignment:
            return "full (target-align + pseudo-label)"
        if self.use_source_alignment:
            return "full (source-align + pseudo-label)"
        return "full (pseudo-label only)"

    def _log_metrics_breakdown(self, label: str, metrics: Dict[str, float]) -> None:
        """Log a compact loss breakdown for train/val summaries."""
        pseudo_cls = self._format_pseudo_class_counts(metrics)
        prior_mask = metrics.get("loss_prior_mask", 0.0) + metrics.get("loss_prior_mask_target", 0.0)
        prior_prop = metrics.get("loss_prior_propagation", 0.0) + metrics.get("loss_prior_propagation_target", 0.0)
        prior_periodic = metrics.get("loss_prior_periodic", 0.0) + metrics.get("loss_prior_periodic_target", 0.0)
        self.logger.info(
            f"{label:<15} | "
            f"det={metrics.get('loss_det', 0.0):.4f}, "
            f"det_gt={metrics.get('loss_det_gt', 0.0):.4f}, "
            f"det_pseudo={metrics.get('loss_det_pseudo_weighted', 0.0):.4f}, "
            f"det_total={metrics.get('loss_det_total', 0.0):.4f}, "
            f"align={metrics.get('loss_align', 0.0):.4f}, "
            f"align_target={metrics.get('loss_align_target', 0.0):.4f}, "
            f"gate_distill={metrics.get('loss_student_gate_distill', 0.0):.4f}, "
            f"gate_mean={metrics.get('student_gate_mask_mean', 0.0):.4f}, "
            f"gate_target={metrics.get('student_gate_target_mean', 0.0):.4f}, "
            f"prior_mask={prior_mask:.4f}, "
            f"prior_prop={prior_prop:.4f}, "
            f"prior_periodic={prior_periodic:.4f}, "
            f"pseudo_raw={metrics.get('loss_pseudo', 0.0):.4f}, "
            f"pseudo_ultra_img={metrics.get('loss_pseudo_ultra_per_img', metrics.get('loss_pseudo_per_img', 0.0)):.4f}, "
            f"pseudo_weighted={metrics.get('loss_pseudo_weighted', 0.0):.4f}, "
            f"pseudo_weighted_img={metrics.get('loss_pseudo_weighted_per_img', 0.0):.4f}, "
            f"pseudo_boxes={metrics.get('pseudo_box_count', 0.0):.1f}, "
            f"{pseudo_cls + ', ' if pseudo_cls else ''}"
            f"total={metrics.get('total_loss', 0.0):.4f}, "
            f"total_img={metrics.get('total_loss_per_img', 0.0):.4f}"
        )

    def _select_current_loss(
        self,
        train_metrics: Dict[str, float],
        val_metrics: Dict[str, float],
    ) -> float:
        """Choose the scalar loss proxy used for best-model selection."""
        current_loss = val_metrics.get("total_loss", train_metrics.get("total_loss", 0))

        # In pseudo-only setup, choose "best" by raw Ultralytics pseudo
        # criterion per-image (not weighted/total), so the scale matches
        # standalone Ultralytics loss comparisons.
        if self.supervision_mode == "pseudo-only":
            val_pseudo_boxes = float(val_metrics.get("pseudo_box_count", 0.0) or 0.0)
            val_pseudo_loss = val_metrics.get(
                "loss_pseudo_per_img",
                val_metrics.get("loss_pseudo", None),
            )
            train_pseudo_loss = train_metrics.get(
                "loss_pseudo_per_img",
                train_metrics.get("loss_pseudo", None),
            )
            if val_pseudo_loss is not None and val_pseudo_boxes > 0:
                current_loss = val_pseudo_loss
            elif train_pseudo_loss is not None:
                current_loss = train_pseudo_loss

        return float(current_loss)

    def _select_current_score(
        self,
        epoch: int,
        current_loss: float,
        *,
        force_map_eval: bool = False,
    ) -> float:
        """Return the score tracked by the best-model policy."""
        if self.best_by == "loss":
            return float(current_loss)

        current_score = float("-inf")
        should_eval_map = force_map_eval or (((epoch + 1) % self.map_eval_interval) == 0)
        if should_eval_map:
            map_metrics = self._evaluate_student_map(epoch)
            if map_metrics is not None:
                current_score = float(map_metrics.get(self.best_by, float("-inf")))
        else:
            self.logger.info(
                "Skip mAP eval at epoch %d (interval=%d)",
                epoch,
                self.map_eval_interval,
            )

        return current_score

    # ------------------------------------------------------------------
    # Full training loop
    # ------------------------------------------------------------------
    def train(self):
        tc = self.config["training"]
        oc = self.config["output"]

        self.logger.info("=" * 60)
        self.logger.info("Starting MTKDTrainerV2  (DINO-Teacher aligned)")
        self.logger.info("=" * 60)

        total_start = time.time()

        initial_eval_epoch = self.start_epoch - 1
        if self.val_loader is not None:
            current_epoch = self.start_epoch
            self.logger.info(
                "Initial validation before epoch %d  —  stage: %s",
                current_epoch,
                self._stage_name(current_epoch),
            )
            initial_val_metrics = self.validate(current_epoch)
            initial_loss = self._select_current_loss({}, initial_val_metrics)
            initial_score = self._select_current_score(
                initial_eval_epoch,
                initial_loss,
                force_map_eval=True,
            )

            self.logger.info(
                "Initial validation done | Val: %.4f",
                float(initial_val_metrics.get("total_loss", 0.0)),
            )
            self._log_metrics_breakdown("Initial Val", initial_val_metrics)

            improved = (
                initial_score > self.best_score
                if self._best_higher_is_better
                else initial_score < self.best_score
            )
            if improved:
                self.best_score = float(initial_score)
                self.best_loss = float(initial_loss)
                if self.save_pth_checkpoints:
                    bp = os.path.join(oc["save_dir"], "best_model.pth")
                    save_checkpoint(
                        self.model,
                        self.optimizer,
                        initial_eval_epoch,
                        initial_loss,
                        bp,
                        scheduler=self.scheduler,
                        extra_state=self._checkpoint_extra_state(),
                    )
                self._export_best_ultralytics_pt(initial_eval_epoch)
                if self.best_by == "loss":
                    self.logger.info(
                        "Seeded best model from initial validation — loss = %.4f",
                        self.best_score,
                    )
                elif self.best_by == "map50":
                    self.logger.info(
                        "Seeded best model from initial validation — mAP50 = %.4f",
                        self.best_score,
                    )
                elif self.best_by == "fitness":
                    self.logger.info(
                        "Seeded best model from initial validation — fitness = %.4f",
                        self.best_score,
                    )
                else:
                    self.logger.info(
                        "Seeded best model from initial validation — mAP50-95 = %.4f",
                        self.best_score,
                    )

        for epoch in range(self.start_epoch, tc["epochs"]):
            t0 = time.time()

            # --- stage banner ---
            self.logger.info(f"Epoch {epoch}  —  stage: {self._stage_name(epoch)}")

            train_metrics = self.train_epoch(epoch)
            val_metrics = self.validate(epoch)

            if self.scheduler is not None:
                self.scheduler.step()

            epoch_time = time.time() - t0
            self.logger.info(
                f"Epoch {epoch} done in {format_time(epoch_time)} | "
                f"Train: {train_metrics.get('total_loss', 'n/a'):.4f} | "
                f"Val: {val_metrics.get('total_loss', 'n/a')}"
            )
            self._log_metrics_breakdown("Train breakdown", train_metrics)
            self._log_metrics_breakdown("Val breakdown", val_metrics)

            if self.debug_export_interval > 0 and ((epoch + 1) % self.debug_export_interval == 0):
                try:
                    self._export_epoch_debug(epoch + 1)
                except Exception as exc:
                    self.logger.warning("Debug export failed at epoch %d: %s", epoch + 1, exc)

            current_loss = self._select_current_loss(train_metrics, val_metrics)
            current_score = self._select_current_score(epoch, current_loss)

            if self.save_pth_checkpoints and self.save_freq > 0 and (epoch + 1) % self.save_freq == 0:
                sp = os.path.join(oc["save_dir"], f"checkpoint_epoch_{epoch}.pth")
                save_checkpoint(self.model, self.optimizer, epoch, current_loss, sp,
                                scheduler=self.scheduler,
                                extra_state=self._checkpoint_extra_state())

            improved = (
                current_score > self.best_score
                if self._best_higher_is_better
                else current_score < self.best_score
            )

            if improved:
                self.best_score = float(current_score)
                self.best_loss = float(current_loss)
                if self.save_pth_checkpoints:
                    bp = os.path.join(oc["save_dir"], "best_model.pth")
                    save_checkpoint(self.model, self.optimizer, epoch, current_loss, bp,
                                    scheduler=self.scheduler,
                                    extra_state=self._checkpoint_extra_state())
                self._export_best_ultralytics_pt(epoch)
                if self.best_by == "loss":
                    self.logger.info(f"New best model — loss = {self.best_score:.4f}")
                elif self.best_by == "map50":
                    self.logger.info(f"New best model — mAP50 = {self.best_score:.4f}")
                elif self.best_by == "fitness":
                    self.logger.info(f"New best model — fitness = {self.best_score:.4f}")
                else:
                    self.logger.info(f"New best model — mAP50-95 = {self.best_score:.4f}")

            # Early stopping target should match best-model metric:
            # - best_by=loss   -> minimize val loss
            # - best_by=map*   -> maximize mAP (equivalent to minimizing -mAP)
            early_stop_value: Optional[float] = None
            if self.best_by == "loss":
                early_stop_value = float(current_loss)
            elif np.isfinite(current_score):
                early_stop_value = -float(current_score)

            if (
                self.enable_early_stopping
                and early_stop_value is not None
                and self.early_stopping(early_stop_value)
            ):
                self.logger.info(f"Early stopping at epoch {epoch}")
                break

        total_time = time.time() - total_start
        self.logger.info("=" * 60)
        self.logger.info(f"Training finished in {format_time(total_time)}")
        if self.best_by == "loss":
            self.logger.info(f"Best loss: {self.best_score:.4f}")
        elif self.best_by == "map50":
            self.logger.info(f"Best mAP50: {self.best_score:.4f}")
        elif self.best_by == "fitness":
            self.logger.info(f"Best fitness: {self.best_score:.4f}")
        else:
            self.logger.info(f"Best mAP50-95: {self.best_score:.4f}")
        self.logger.info("=" * 60)
