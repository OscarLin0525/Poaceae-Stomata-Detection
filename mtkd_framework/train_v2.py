"""
MTKDTrainerV2 — training loop modelled after DINO Teacher
==========================================================

Reference:  ``DINO_Teacher/dinoteacher/engine/trainer.py``
            ───  class ``DINOTeacherTrainer(DefaultTrainer)``

Key parallels
-------------
* **3-stage training**:
    1. Burn-in — supervised only (labelled source GT).
    2. Source alignment — add feature alignment loss (student ↔ DINO teacher).
    3. Full — add pseudo-label supervision on target domain.

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
import logging
import os
import time
from collections import OrderedDict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .models.mtkd_model_v2 import MTKDModelV2, build_mtkd_model_v2
from .engine.pseudo_labels import (
    load_pseudo_labels_dir,
    load_pseudo_labels_csv,
    build_yolo_batch_from_pseudo,
    targets_to_yolo_batch,
)
from .losses.prediction_alignment import UltralyticsCriterionAlignmentLoss
from .losses.separation import ValleySeparationLoss
from .utils import (
    AverageMeterDict,
    EarlyStopping,
    GradientClipper,
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
    ``align_head_config``, ``burn_up_steps``, ``align_target_start``,
    ``unsup_loss_weight``, ``fft_block_config``, etc.
    """
    return {
        # ---- model ----
        "model": {
            "num_classes": 3,  # barley dataset merged to 3 classes
            "student_config": {
                "student_type": "yolo",
                "weights": "yolo12l.pt",
                "feature_level": "p4",
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
            "student_align_layer": "p4",      # which YOLO pyramid level to align
            # FFT block injection into DINO (set None to disable)
            "fft_block_config": {
                "after_blocks": [9],
                "num_freq_bins": 32,
                "hidden_dim": 256,
                "init_gate": -5.0,
                "modulation_mode": "multiplicative",
            },
            # Ensemble teachers (optional)
            "ensemble_config": None,
            "teacher_specs": None,
        },
        # ---- training ----
        "training": {
            "epochs": 100,
            "batch_size": 8,
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
            # ---- DINO Teacher style stages ----
            "burn_up_epochs": 5,               # pure supervised
            "align_target_start_epoch": 10,    # add target alignment after this
            # ---- loss weights (mirrors DINO Teacher) ----
            "feature_align_loss_weight": 1.0,
            "feature_align_loss_weight_target": 1.0,
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
            "separation_loss_weight": 0.0,     # valley separation loss (disabled by default)
            "separation_sample_points": 5,     # points to sample along connecting lines
            "separation_valley_margin": 0.2,   # minimum valley depth factor
            "separation_target_layer": 10,     # which DINO layer to extract features from
            "zero_pseudo_box_reg": False,      # default: keep pseudo bbox/dfl regression enabled
            "supervision_mode": "gt+pseudo",  # gt+pseudo | gt-only | pseudo-only
            # ALIGN_EASY_ONLY — When True, the DINO teacher receives *only*
            # the original (non-augmented) images for alignment, preventing
            # augmentation artefacts from polluting the teacher signal.
            # Requires the dataset to supply ``batch["images_weak"]`` (a
            # second, unaugmented copy).  If False or the key is absent,
            # the same augmented images are sent to both student and teacher.
            "align_easy_only": False,
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
            # Confidence threshold for filtering
            "score_threshold": 0.5,
            # Auto-convert OBB 8-coord to axis-aligned bbox
            "convert_obb": True,
        },
        # ---- data ----
        "data": {
            "dataset_root": "Stomata_Dataset",
            "image_subdir": "barley_category/barley_image_fresh-leaf",
            "label_subdir": "barley_category/barley_label_fresh-leaf",
            "val_ratio": 0.1,
            "image_size": 640,
            "augmentation": True,
        },
        # ---- output ----
        "output": {
            "save_dir": "outputs/mtkd_v2",
            "save_freq": 5,
            "log_freq": 10,
            # Best model selection metric:
            # - loss: lower is better (default)
            # - map50: higher is better
            # - map5095: higher is better
            "best_by": "loss",
            # mAP selection settings (used only when best_by != loss)
            "map_data": None,
            "map_split": "val",
            "map_imgsz": 640,
            "map_batch": 16,
            "map_conf": 0.25,
            "map_iou": 0.6,
            "map_eval_interval": 1,
        },
        # ---- misc ----
        "seed": 42,
        "device": "cuda" if torch.cuda.is_available() else "cpu",
    }


# ======================================================================
# MTKDTrainerV2
# ======================================================================

class MTKDTrainerV2:
    """
    Trainer modelled after ``DINOTeacherTrainer``.

    Stages (epoch-based rather than iter-based for simplicity)
    -----------------------------------------------------------
    1. ``epoch < burn_up_epochs``
       → Only supervised loss on labelled source data.

    2. ``burn_up_epochs <= epoch < align_target_start_epoch``
       → Supervised loss  **+**  source feature alignment (student ↔ DINO).

    3. ``epoch >= align_target_start_epoch``
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

        # ---- data ----
        self.train_loader = train_loader
        self.val_loader = val_loader

        # ---- optimiser ----
        self._setup_optimizer()
        self._setup_scheduler()

        # ---- AMP ----
        self.scaler = None
        tc = config["training"]
        if tc.get("mixed_precision") and self.device.type == "cuda":
            self.scaler = torch.amp.GradScaler("cuda")

        self.gradient_clipper = GradientClipper(
            max_norm=tc.get("gradient_clip_max_norm", 1.0)
        )
        self.early_stopping = EarlyStopping(
            patience=tc.get("early_stopping_patience", 20), mode="min",
        )

        # ---- stage thresholds ----
        self.burn_up_epochs = tc.get("burn_up_epochs", 5)
        self.align_target_start_epoch = tc.get("align_target_start_epoch", 10)

        # ---- loss weights ----
        self.feature_align_w = tc.get("feature_align_loss_weight", 1.0)
        self.feature_align_w_target = tc.get("feature_align_loss_weight_target", 1.0)
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
        self.separation_loss_w = tc.get("separation_loss_weight", 0.0)  # disabled by default
        self.zero_pseudo_box_reg = tc.get("zero_pseudo_box_reg", False)
        self.align_easy_only = tc.get("align_easy_only", False)
        if self.use_target_alignment and not self.separate_source_target_data:
            self.logger.warning(
                "use_target_alignment=True but separate_source_target_data=False; "
                "disable target alignment to avoid duplicate pressure on the same batch."
            )
            self.use_target_alignment = False
        self.logger.info(
            "Target alignment term: %s",
            "enabled" if self.use_target_alignment else "disabled",
        )
        if self.zero_pseudo_box_reg:
            self.logger.info("Pseudo box regression: disabled (cls-only pseudo supervision)")
        else:
            self.logger.info("Pseudo box regression: enabled (Ultralytics IoU/probIoU + DFL)")
        if self.prediction_align_mode == "ultralytics":
            self.logger.info("Prediction alignment: Ultralytics criterion/assigner path")
        else:
            self.logger.info("Prediction alignment: legacy direct pseudo criterion path")
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
        if not self.enable_early_stopping:
            self.logger.info("Early stopping disabled in pseudo-only mode")
        
        # ---- separation loss ----
        self.separation_loss_fn = None
        if self.separation_loss_w > 0:
            self.separation_loss_fn = ValleySeparationLoss(
                sample_points=tc.get("separation_sample_points", 5),
                valley_margin=tc.get("separation_valley_margin", 0.2),
            )
            self.logger.info(f"Separation loss enabled (weight={self.separation_loss_w})")
            # Target DINO layer for separation loss
            self.separation_target_layer = tc.get("separation_target_layer", 10)

        # ---- pseudo labels ----
        self.pseudo_labels: Optional[Dict] = None
        pl_cfg = config.get("pseudo_labels", {})
        self.pseudo_mode = str(pl_cfg.get("mode", "auto")).lower()
        if self.pseudo_mode not in {"auto", "offline", "online", "none"}:
            raise ValueError(
                f"Invalid pseudo_labels.mode={self.pseudo_mode}. "
                "Choose one of: auto/offline/online/none"
            )
        self.pl_score_threshold = pl_cfg.get("score_threshold", 0.5)
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

        # ---- best model selection policy ----
        oc = self.config.get("output", {})
        self.best_by = str(oc.get("best_by", "loss")).lower()
        if self.best_by not in {"loss", "map50", "map5095"}:
            raise ValueError("output.best_by must be one of: loss / map50 / map5095")
        self._best_higher_is_better = self.best_by in {"map50", "map5095"}
        self.best_score = float("-inf") if self._best_higher_is_better else float("inf")

        self.map_data = oc.get("map_data")
        self.map_split = str(oc.get("map_split", "val"))
        self.map_imgsz = int(oc.get("map_imgsz", 640))
        self.map_batch = int(oc.get("map_batch", 16))
        self.map_conf = float(oc.get("map_conf", 0.25))
        self.map_iou = float(oc.get("map_iou", 0.6))
        self.map_eval_interval = max(1, int(oc.get("map_eval_interval", 1)))

        if self.best_by != "loss" and not self.map_data:
            raise ValueError(
                "output.best_by is set to mAP mode, but output.map_data is not set. "
                "Provide a dataset yaml path via --map-data."
            )

        self.logger.info(f"Best selection metric: {self.best_by}")
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
        if stype == "cosine":
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, T_max=tc["epochs"], eta_min=tc.get("min_lr", 1e-6),
            )
        elif stype == "step":
            self.scheduler = torch.optim.lr_scheduler.StepLR(
                self.optimizer, step_size=30, gamma=0.1,
            )
        else:
            self.scheduler = None

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
        self.logger.info(f"Resumed from epoch {self.start_epoch}")

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
        num_classes = int(self.config.get("model", {}).get("num_classes", 1))
        class_names = {i: f"class_{i}" for i in range(num_classes)}

        try:
            export_fn(
                save_path=tmp_pt,
                num_classes=num_classes,
                class_names=class_names,
                epoch=epoch,
            )

            from ultralytics import YOLO

            model = YOLO(tmp_pt)
            metrics = model.val(
                data=self.map_data,
                split=self.map_split,
                imgsz=self.map_imgsz,
                batch=self.map_batch,
                conf=self.map_conf,
                iou=self.map_iou,
                device=self.config.get("device", "cuda"),
                plots=False,
                verbose=False,
            )

            result = {
                "map50": float(metrics.box.map50),
                "map5095": float(metrics.box.map),
            }
            self.logger.info(
                "mAP eval | map50=%.4f map50-95=%.4f",
                result["map50"], result["map5095"],
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

    # ------------------------------------------------------------------
    # Train one epoch
    # ------------------------------------------------------------------
    def train_epoch(self, epoch: int) -> Dict[str, float]:
        """
        One epoch following the 3-stage pattern of DINO Teacher.
        """
        self.model.train()
        meters = AverageMeterDict()
        accum = self.config["training"].get("accumulation_steps", 1)
        start_time = time.time()
        num_batches = len(self.train_loader)

        for batch_idx, batch in enumerate(self.train_loader):
            images = batch["images"].to(self.device)
            targets = {
                k: (v.to(self.device) if isinstance(v, torch.Tensor) else v)
                for k, v in batch.get("targets", {}).items()
            }
            image_paths: List[str] = batch.get("image_paths", [])

            # ALIGN_EASY_ONLY: stash un-augmented images for the teacher
            self._current_images_weak = None
            if self.align_easy_only and "images_weak" in batch:
                self._current_images_weak = batch["images_weak"].to(self.device)

            # Track strong-view flips so offline pseudo boxes can be transformed
            # into the same geometric space as student images.
            self._current_strong_hflip = batch.get("strong_hflip")

            # === Determine stage ===
            do_target_align = epoch >= self.align_target_start_epoch
            do_source_align = epoch >= self.burn_up_epochs

            # === Forward ===
            need_teacher = do_source_align
            if self.scaler is not None:
                with torch.amp.autocast("cuda"):
                    loss, loss_dict = self._forward_and_loss(
                        images, targets, image_paths, epoch,
                        need_teacher_feat=need_teacher,
                        do_source_align=do_source_align,
                        do_target_align=do_target_align,
                    )
                    loss = loss / accum
            else:
                loss, loss_dict = self._forward_and_loss(
                    images, targets, image_paths, epoch,
                    need_teacher_feat=need_teacher,
                    do_source_align=do_source_align,
                    do_target_align=do_target_align,
                )
                loss = loss / accum

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
                self.optimizer.zero_grad()

            meters.update(loss_dict, n=images.size(0))

            if (batch_idx + 1) % self.config["output"]["log_freq"] == 0:
                elapsed = time.time() - start_time
                eta = elapsed / (batch_idx + 1) * (num_batches - batch_idx - 1)
                self.logger.info(
                    f"Epoch [{epoch}][{batch_idx + 1}/{num_batches}] "
                    f"Loss: {loss_dict.get('total_loss', 0):.4f} "
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

        # ---- Build pseudo-label batch (if stage 3) ----
        pseudo_batch = None

        # ---- Single student forward (train mode) + DINO teacher ----
        # ALIGN_EASY_ONLY: feed un-augmented images to the DINO teacher so
        # that alignment targets are free of augmentation artefacts.  The
        # dataset must supply ``batch["images_weak"]`` for this to work.
        teacher_images_align = images
        if self.align_easy_only:
            # images_weak is injected by train_epoch when available
            if hasattr(self, "_current_images_weak") and self._current_images_weak is not None:
                teacher_images_align = self._current_images_weak

        # ---- Build pseudo-label batch (if stage 3) ----
        # Priority:
        # 1) offline pseudo-label files
        # 2) online frozen wheat-teacher predictions
        if do_target_align:
            hflip_mask = None
            if hasattr(self, "_current_strong_hflip") and self._current_strong_hflip is not None:
                if isinstance(self._current_strong_hflip, torch.Tensor):
                    hflip_mask = self._current_strong_hflip.detach().cpu().tolist()
                else:
                    hflip_mask = list(self._current_strong_hflip)

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

        # ----- Add align head params to optimizer if just lazily created -----
        if getattr(self.model, "_align_head_just_created", False):
            self.optimizer.add_param_group({
                "params": list(self.model.align_head.parameters()),
                "lr": self.config["training"]["learning_rate"],
            })
            self.model._align_head_just_created = False
            self.logger.info("Added alignment head parameters to optimizer")

        # ----- Detection loss (GT supervision) -----
        if self.use_gt_supervision:
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
                align_loss = self.model.compute_align_loss(student_feat, dino_feat)
                weighted = align_loss * self.feature_align_w
                losses.append(weighted)
                loss_dict["loss_align"] = align_loss.item()

        # ----- Target feature alignment (stage 3) -----
        # Mirrors DINO Teacher's ``loss_align_target``.  In DINO_Teacher this
        # is computed on *unlabeled target-domain* images with a separate
        # weight.  In MTKD (single-dataset), the same images serve both
        # roles, so we add a second weighted alignment signal in stage 3 to
        # guide the student backbone toward the DINO teacher space on the
        # target distribution.
        if self.use_target_alignment and do_target_align and do_source_align:
            student_feat = out.get("student_spatial_feat")
            dino_feat = out.get("dino_features")
            if student_feat is not None and dino_feat is not None:
                # Reuse align_loss if already computed, else recompute
                if "loss_align" not in loss_dict:
                    align_loss_target = self.model.compute_align_loss(
                        student_feat, dino_feat)
                else:
                    align_loss_target = align_loss  # same data → same value
                weighted_target = align_loss_target * self.feature_align_w_target
                losses.append(weighted_target)
                loss_dict["loss_align_target"] = align_loss_target.item()

        # ----- Pseudo-label loss (target) -----
        # Computed separately so we can optionally zero out box regression.
        # Reuses the same raw_preds from the single student forward pass.
        if (
            self.use_pseudo_supervision
            and do_target_align
            and pseudo_batch is not None
            and pseudo_batch["bboxes"].shape[0] > 0
        ):
            pseudo_box_count = int(pseudo_batch["bboxes"].shape[0])
            loss_dict["pseudo_box_count"] = float(pseudo_box_count)
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
                    orig_box = criterion.hyp.box
                    orig_dfl = criterion.hyp.dfl
                    criterion.hyp.box = 0.0
                    criterion.hyp.dfl = 0.0
                    try:
                        pl_loss, pl_items = criterion(out["raw_preds"], pseudo_batch)
                    finally:
                        criterion.hyp.box = orig_box
                        criterion.hyp.dfl = orig_dfl
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

        # ----- Separation loss -----
        # Encourages valleys between adjacent GT stomata in feature space
        if self.separation_loss_fn is not None and gt_batch is not None:
            # Extract layer features from DINO teacher
            layer_features = self._extract_dino_layer_features(
                teacher_images_align, self.separation_target_layer
            )
            # Parse GT centers from YOLO batch
            gt_centers_list = self._extract_gt_centers_from_batch(
                gt_batch, layer_features.shape[1:3]
            )
            # Compute separation loss
            if layer_features is not None and len(gt_centers_list) > 0:
                sep_loss = self.separation_loss_fn(layer_features, gt_centers_list)
                weighted_sep = sep_loss * self.separation_loss_w
                losses.append(weighted_sep)
                loss_dict["loss_separation"] = sep_loss.item()
        
        if len(losses) == 0:
            total_loss = images.new_tensor(0.0)
        else:
            total_loss = sum(losses)  # type: ignore[arg-type]
        loss_dict["total_loss"] = total_loss.item()

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

        valid_mask = scores >= float(self.pl_score_threshold)
        if "valid_mask" in teacher_pred and teacher_pred["valid_mask"] is not None:
            valid_mask = valid_mask & teacher_pred["valid_mask"].bool()

        all_idx: List[int] = []
        all_cls: List[float] = []
        all_bboxes: List[List[float]] = []

        for b in range(boxes.shape[0]):
            mask = valid_mask[b]
            if mask.ndim == 0:
                continue
            b_boxes = boxes[b][mask]
            b_labels = labels[b][mask]
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

    # ------------------------------------------------------------------
    # Separation Loss Helpers
    # ------------------------------------------------------------------
    def _extract_dino_layer_features(
        self,
        images: torch.Tensor,
        layer_idx: int
    ) -> Optional[torch.Tensor]:
        """
        Extract features from a specific DINO layer using a forward hook.
        
        Args:
            images: [B, 3, H, W] input images
            layer_idx: Which layer block to extract (e.g., 10 for layer 10)
        
        Returns:
            features: [B, h, w, C] spatial features from the specified layer
                     where h, w = H//patch_size, W//patch_size
        """
        if not hasattr(self.model, 'dino_teacher') or self.model.dino_teacher is None:
            return None
        
        dino_model = self.model.dino_teacher.model
        if not hasattr(dino_model, 'blocks') or layer_idx >= len(dino_model.blocks):
            return None
        
        extracted_features = {}
        
        def hook_fn(module, input, output):
            # output is [B, N_tokens+1, C] where N_tokens = (H/patch_size) * (W/patch_size)
            extracted_features['output'] = output
        
        # Register hook
        handle = dino_model.blocks[layer_idx].register_forward_hook(hook_fn)
        
        try:
            # Run forward pass
            with torch.no_grad():
                _ = dino_model(images)
            
            # Extract features
            if 'output' not in extracted_features:
                return None
            
            feat = extracted_features['output']  # [B, N+1, C]
            B, N_plus_1, C = feat.shape
            
            # Remove CLS token (first token)
            spatial_tokens = feat[:, 1:, :]  # [B, N, C]
            
            # Reshape to spatial grid
            N = N_plus_1 - 1
            h = w = int(N ** 0.5)
            assert h * w == N, f"Non-square token count: {N}"
            
            spatial_features = spatial_tokens.reshape(B, h, w, C)  # [B, h, w, C]
            
            return spatial_features
        
        finally:
            handle.remove()
    
    def _extract_gt_centers_from_batch(
        self,
        yolo_batch: Dict[str, torch.Tensor],
        spatial_shape: Tuple[int, int]
    ) -> List[torch.Tensor]:
        """
        Extract GT center coordinates from YOLO batch format.
        
        Args:
            yolo_batch: Dict with keys 'bboxes' [N, 5 or 6], 'cls' [N], 'batch_idx' [N]
                       bboxes format: [batch_idx, class, cx, cy, w, h] (normalized 0-1)
            spatial_shape: (h, w) - spatial dimensions of the feature map
        
        Returns:
            List of [N_i, 2] tensors containing (row, col) for each image in batch
        """
        if yolo_batch is None or 'bboxes' not in yolo_batch:
            return []
        
        bboxes = yolo_batch['bboxes']  # [N, 5 or 6]
        if bboxes.shape[0] == 0:
            return []
        
        # Parse batch indices and centers
        # Format: [batch_idx, cls, cx, cy, w, h] or [cls, cx, cy, w, h]
        if bboxes.shape[1] == 6:
            batch_indices = bboxes[:, 0].long()  # [N]
            centers_norm = bboxes[:, 2:4]  # [N, 2] (cx, cy) in [0, 1]
        else:
            # Need to use batch_idx from yolo_batch
            batch_indices = yolo_batch.get('batch_idx', torch.zeros(bboxes.shape[0], dtype=torch.long))
            centers_norm = bboxes[:, 1:3]  # [N, 2]
        
        h, w = spatial_shape
        
        # Convert normalized centers to spatial coordinates
        centers_spatial = centers_norm.clone()
        centers_spatial[:, 0] = centers_norm[:, 1] * h  # cy -> row
        centers_spatial[:, 1] = centers_norm[:, 0] * w  # cx -> col
        centers_spatial = centers_spatial.round().long()
        
        # Clamp to valid range
        centers_spatial[:, 0] = centers_spatial[:, 0].clamp(0, h - 1)
        centers_spatial[:, 1] = centers_spatial[:, 1].clamp(0, w - 1)
        
        # Group by batch index
        B = batch_indices.max().item() + 1
        centers_list = []
        for b in range(B):
            mask = batch_indices == b
            if mask.sum() > 0:
                centers_list.append(centers_spatial[mask])  # [N_b, 2]
            else:
                centers_list.append(torch.empty((0, 2), dtype=torch.long, device=centers_spatial.device))
        
        return centers_list

    # ------------------------------------------------------------------
    # Validate
    # ------------------------------------------------------------------
    @torch.no_grad()
    def validate(self, epoch: int) -> Dict[str, float]:
        if self.val_loader is None:
            return {}
        self.model.eval()
        meters = AverageMeterDict()
        # In pseudo-only mode, include pseudo supervision in validation once
        # stage-3 starts so model selection tracks detection behaviour instead
        # of alignment-only loss.
        val_do_target_align = (
            self.use_pseudo_supervision
            and epoch >= self.align_target_start_epoch
        )
        for batch in self.val_loader:
            images = batch["images"].to(self.device)
            targets = {
                k: (v.to(self.device) if isinstance(v, torch.Tensor) else v)
                for k, v in batch.get("targets", {}).items()
            }
            image_paths: List[str] = batch.get("image_paths", [])
            # Validation uses the eval-mode forward for alignment metrics.
            # Detection loss is still computed (student in train mode internally).
            _, loss_dict = self._forward_and_loss(
                images, targets, image_paths, epoch,
                need_teacher_feat=True, do_source_align=True,
                do_target_align=val_do_target_align,
            )
            meters.update(loss_dict, n=images.size(0))
        return meters.get_averages()

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

        for epoch in range(self.start_epoch, tc["epochs"]):
            t0 = time.time()

            # --- stage banner ---
            if epoch < self.burn_up_epochs:
                stage = "burn-in (supervised only)"
            elif epoch < self.align_target_start_epoch:
                stage = "source alignment"
            else:
                stage = "full (align + pseudo-label)"
            self.logger.info(f"Epoch {epoch}  —  stage: {stage}")

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
            self.logger.info(
                "Train breakdown | "
                f"det={train_metrics.get('loss_det', 0.0):.4f}, "
                f"align={train_metrics.get('loss_align', 0.0):.4f}, "
                f"align_target={train_metrics.get('loss_align_target', 0.0):.4f}, "
                f"pseudo_raw={train_metrics.get('loss_pseudo', 0.0):.4f}, "
                f"pseudo_weighted={train_metrics.get('loss_pseudo_weighted', 0.0):.4f}, "
                f"pseudo_boxes={train_metrics.get('pseudo_box_count', 0.0):.1f}, "
                f"total={train_metrics.get('total_loss', 0.0):.4f}"
            )
            self.logger.info(
                "Val breakdown   | "
                f"det={val_metrics.get('loss_det', 0.0):.4f}, "
                f"align={val_metrics.get('loss_align', 0.0):.4f}, "
                f"align_target={val_metrics.get('loss_align_target', 0.0):.4f}, "
                f"pseudo_raw={val_metrics.get('loss_pseudo', 0.0):.4f}, "
                f"pseudo_weighted={val_metrics.get('loss_pseudo_weighted', 0.0):.4f}, "
                f"pseudo_boxes={val_metrics.get('pseudo_box_count', 0.0):.1f}, "
                f"total={val_metrics.get('total_loss', 0.0):.4f}"
            )

            current_loss = val_metrics.get("total_loss", train_metrics.get("total_loss", 0))

            # In pseudo-only setup, choose "best" by pseudo detection loss
            # when available; alignment-only val loss is not a good proxy for
            # prediction quality.
            if self.supervision_mode == "pseudo-only":
                val_pseudo_boxes = float(val_metrics.get("pseudo_box_count", 0.0) or 0.0)
                val_pseudo_loss = val_metrics.get("loss_pseudo_weighted", None)
                train_pseudo_loss = train_metrics.get("loss_pseudo_weighted", None)
                if val_pseudo_loss is not None and val_pseudo_boxes > 0:
                    current_loss = val_pseudo_loss
                elif train_pseudo_loss is not None:
                    current_loss = train_pseudo_loss

            if self.best_by == "loss":
                current_score = float(current_loss)
            else:
                current_score = float("-inf")
                if ((epoch + 1) % self.map_eval_interval) == 0:
                    map_metrics = self._evaluate_student_map(epoch)
                    if map_metrics is not None:
                        current_score = float(map_metrics.get(self.best_by, float("-inf")))
                else:
                    self.logger.info(
                        "Skip mAP eval at epoch %d (interval=%d)",
                        epoch,
                        self.map_eval_interval,
                    )

            if (epoch + 1) % oc["save_freq"] == 0:
                sp = os.path.join(oc["save_dir"], f"checkpoint_epoch_{epoch}.pth")
                save_checkpoint(self.model, self.optimizer, epoch, current_loss, sp,
                                scheduler=self.scheduler)

            improved = (
                current_score > self.best_score
                if self._best_higher_is_better
                else current_score < self.best_score
            )

            if improved:
                self.best_score = float(current_score)
                self.best_loss = float(current_loss)
                bp = os.path.join(oc["save_dir"], "best_model.pth")
                save_checkpoint(self.model, self.optimizer, epoch, current_loss, bp,
                                scheduler=self.scheduler)
                self._export_best_ultralytics_pt(epoch)
                if self.best_by == "loss":
                    self.logger.info(f"New best model — loss = {self.best_score:.4f}")
                elif self.best_by == "map50":
                    self.logger.info(f"New best model — mAP50 = {self.best_score:.4f}")
                else:
                    self.logger.info(f"New best model — mAP50-95 = {self.best_score:.4f}")

            if self.enable_early_stopping and self.early_stopping(current_loss):
                self.logger.info(f"Early stopping at epoch {epoch}")
                break

        total_time = time.time() - total_start
        self.logger.info("=" * 60)
        self.logger.info(f"Training finished in {format_time(total_time)}")
        if self.best_by == "loss":
            self.logger.info(f"Best loss: {self.best_score:.4f}")
        elif self.best_by == "map50":
            self.logger.info(f"Best mAP50: {self.best_score:.4f}")
        else:
            self.logger.info(f"Best mAP50-95: {self.best_score:.4f}")
        self.logger.info("=" * 60)
