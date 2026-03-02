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

* **Pseudo-labels from pretrained wheat model** — supplied externally via
  a pickle/dict (same format as DINO Teacher's ``LABELER_TARGET_PSEUDOGT``).
  No EMA teacher is needed because the user's pretrained wheat detector
  *is* the labeller.

* **Feature alignment** follows ``TeacherStudentAlignHead`` in
  ``engine/align_head.py`` — per-pixel spatial, **not** global-pool CLS.

* **Loss weighting** mirrors DINO Teacher: pseudo bbox-regression loss is
  zeroed; pseudo classification loss weighted by ``unsup_loss_weight``.
"""

from __future__ import annotations

import copy
import logging
import os
import pickle
import time
from collections import OrderedDict
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .models.mtkd_model_v2 import MTKDModelV2, build_mtkd_model_v2
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
            "num_classes": 1,
            "student_config": {
                "student_type": "yolo",
                "weights": "yolo11s.pt",
                "feature_level": "p4",
            },
            "dino_config": {
                "model_name": "vit_base",
                "patch_size": 16,
                "embed_dim": 768,
                "normalize_feature": True,
            },
            "dino_checkpoint": None,
            "align_head_config": {
                "head_type": "MLP",           # "attention" / "MLP" / "MLP3" / "linear"
                "proj_dim": 1024,
                "normalize": True,
                "use_gelu": False,
            },
            "student_align_layer": "p4",      # which YOLO pyramid level to align
            # FFT block injection into DINO (set None to disable)
            "fft_block_config": {
                "after_blocks": [10],
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
            "unsup_loss_weight": 4.0,
            "zero_pseudo_box_reg": True,       # zero out pseudo bbox regression loss
        },
        # ---- pseudo labels ----
        "pseudo_labels": {
            # Path to .pkl produced by pretrained wheat model (or None)
            "pickle_path": None,
            # Confidence threshold for filtering
            "score_threshold": 0.5,
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
        self.unsup_loss_w = tc.get("unsup_loss_weight", 4.0)
        self.zero_pseudo_box_reg = tc.get("zero_pseudo_box_reg", True)

        # ---- pseudo labels ----
        self.pseudo_labels: Optional[Dict] = None
        pl_cfg = config.get("pseudo_labels", {})
        pl_path = pl_cfg.get("pickle_path")
        self.pl_score_threshold = pl_cfg.get("score_threshold", 0.5)
        if pl_path and os.path.isfile(pl_path):
            self.load_pseudo_labels(pl_path)

        # ---- training state ----
        self.start_epoch = 0
        self.best_loss = float("inf")
        resume_path = config.get("checkpoints", {}).get("resume")
        if resume_path:
            self._load_checkpoint(resume_path)

    # ------------------------------------------------------------------
    # Pseudo labels
    # ------------------------------------------------------------------
    def load_pseudo_labels(self, pkl_path: str):
        """
        Load pseudo-labels produced by the pretrained wheat model.

        Expected format (same as DINO Teacher):
          list of dicts, each with at least ``image_id`` and ``instances_dino``
          (an object with ``.pred_boxes``, ``.scores``, ``.pred_classes``).

        Or a simpler format:
          dict  { image_id: {"boxes": Tensor, "scores": Tensor, "labels": Tensor} }
        """
        with open(pkl_path, "rb") as f:
            data = pickle.load(f)

        if isinstance(data, list):
            self.pseudo_labels = {}
            for entry in data:
                self.pseudo_labels[entry["image_id"]] = entry
        elif isinstance(data, dict):
            self.pseudo_labels = data
        else:
            raise ValueError(f"Unsupported pseudo-label format: {type(data)}")
        self.logger.info(f"Loaded {len(self.pseudo_labels)} pseudo-labels from {pkl_path}")

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
        ckpt = load_checkpoint(self.model, path, optimizer=self.optimizer,
                               scheduler=self.scheduler)
        self.start_epoch = ckpt.get("epoch", 0) + 1
        self.best_loss = ckpt.get("loss", float("inf"))
        self.logger.info(f"Resumed from epoch {self.start_epoch}")

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

            # === Determine stage ===
            in_burn_in = epoch < self.burn_up_epochs
            do_target_align = epoch >= self.align_target_start_epoch
            do_source_align = epoch >= self.burn_up_epochs

            # === Forward ===
            need_teacher = do_source_align
            if self.scaler is not None:
                with torch.amp.autocast("cuda"):
                    loss, loss_dict = self._forward_and_loss(
                        images, targets, epoch,
                        need_teacher_feat=need_teacher,
                        do_source_align=do_source_align,
                        do_target_align=do_target_align,
                    )
                    loss = loss / accum
            else:
                loss, loss_dict = self._forward_and_loss(
                    images, targets, epoch,
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
        epoch: int,
        need_teacher_feat: bool = False,
        do_source_align: bool = False,
        do_target_align: bool = False,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute losses following the DINO Teacher pattern.

        Returns:
            total_loss:  Scalar tensor for ``.backward()``.
            loss_dict:   Dict of detached loss values for logging.
        """
        loss_dict: Dict[str, float] = {}
        losses: List[torch.Tensor] = []

        # ----- Student forward -----
        out = self.model(
            images,
            return_teacher_features=need_teacher_feat,
            return_student_spatial_feat=need_teacher_feat,
        )

        # ----- Supervised detection loss (placeholder — student's own loss) -----
        # This is always active; it is the user's responsibility to wire in
        # the actual detection loss from YOLO or a custom head.
        det_loss = torch.tensor(0.0, device=self.device)
        loss_dict["loss_det"] = det_loss.item()
        losses.append(det_loss)

        # ----- Feature alignment (source) -----
        if do_source_align:
            student_feat = out.get("student_spatial_feat")
            dino_feat = out.get("dino_features")
            if student_feat is not None and dino_feat is not None:
                align_loss = self.model.compute_align_loss(student_feat, dino_feat)
                weighted = align_loss * self.feature_align_w
                losses.append(weighted)
                loss_dict["loss_align"] = align_loss.item()

        # ----- Pseudo-label loss (target) -----
        # (Activated after align_target_start_epoch and if PLs available)
        if do_target_align and self.pseudo_labels is not None:
            # In a full implementation the trainer would fetch PLs per image_id,
            # threshold them, and feed them as GT to the student's detection loss.
            # Below is a structural placeholder — the actual detection loss wiring
            # depends on the student architecture (YOLO / Faster-RCNN / etc.).
            pseudo_loss = torch.tensor(0.0, device=self.device)
            loss_dict["loss_pseudo"] = pseudo_loss.item()
            weighted_pseudo = pseudo_loss * self.unsup_loss_weight
            losses.append(weighted_pseudo)

        total_loss = sum(losses)  # type: ignore[arg-type]
        loss_dict["total_loss"] = total_loss.item()

        return total_loss, loss_dict

    # ------------------------------------------------------------------
    # Validate
    # ------------------------------------------------------------------
    @torch.no_grad()
    def validate(self, epoch: int) -> Dict[str, float]:
        if self.val_loader is None:
            return {}
        self.model.eval()
        meters = AverageMeterDict()
        for batch in self.val_loader:
            images = batch["images"].to(self.device)
            targets = {
                k: (v.to(self.device) if isinstance(v, torch.Tensor) else v)
                for k, v in batch.get("targets", {}).items()
            }
            _, loss_dict = self._forward_and_loss(
                images, targets, epoch,
                need_teacher_feat=True, do_source_align=True,
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

            current_loss = val_metrics.get("total_loss", train_metrics.get("total_loss", 0))

            if (epoch + 1) % oc["save_freq"] == 0:
                sp = os.path.join(oc["save_dir"], f"checkpoint_epoch_{epoch}.pth")
                save_checkpoint(self.model, self.optimizer, epoch, current_loss, sp,
                                scheduler=self.scheduler)

            if current_loss < self.best_loss:
                self.best_loss = current_loss
                bp = os.path.join(oc["save_dir"], "best_model.pth")
                save_checkpoint(self.model, self.optimizer, epoch, current_loss, bp,
                                scheduler=self.scheduler)
                self.logger.info(f"New best model — loss = {self.best_loss:.4f}")

            if self.early_stopping(current_loss):
                self.logger.info(f"Early stopping at epoch {epoch}")
                break

        total_time = time.time() - total_start
        self.logger.info("=" * 60)
        self.logger.info(f"Training finished in {format_time(total_time)}")
        self.logger.info(f"Best loss: {self.best_loss:.4f}")
        self.logger.info("=" * 60)
