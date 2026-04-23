#!/usr/bin/env python3
"""
MTKDv2 Training Entry Point
============================

CLI script that mirrors ``DINO_Teacher/train_net.py`` for easy
experiment launching.

Usage
-----
.. code-block:: bash

    # Default barley training with DINO alignment
    python -m mtkd_framework.run_v2 \\
        --dino-checkpoint "dinov3-main/weight folder/dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth"

    # With pseudo-labels from wheat model
    python -m mtkd_framework.run_v2 \\
        --dino-checkpoint "dinov3-main/weight folder/dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth" \\
        --pseudo-label-dir runs/detect/predict/labels

    # Custom config via JSON or YAML
    python -m mtkd_framework.run_v2 --config my_config.yaml

    # Override individual params
    python -m mtkd_framework.run_v2 \\
        --epochs 200 --batch-size 4 --lr 5e-5 \\
        --align-target-start 20 \
        --output-dir outputs/exp_01
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch


# Allow running as either:
# 1) python -m mtkd_framework.run_v2
# 2) python mtkd_framework/run_v2.py
if __package__ in {None, ""}:
    repo_root = Path(__file__).resolve().parents[1]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="MTKDv2 Trainer — DINO-Teacher-aligned training",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # ---- config file (optional — overrides defaults) ----
    p.add_argument("--config", type=str, default=None,
                   help="Path to a JSON/YAML config file. CLI args override values in the file.")

    # ---- model ----
    g = p.add_argument_group("Model")
    g.add_argument("--num-classes", type=int, default=3)
    g.add_argument("--student-weights", type=str, default="yolo12s.pt",
                   help="YOLO student pretrained weights")
    g.add_argument("--student-align-layer", type=str, default="p4",
                   choices=["p3", "p4", "p5"],
                   help="Which YOLO pyramid level to align with DINO")
    g.add_argument("--dino-model", type=str, default="vit_base",
                   choices=["vit_small", "vit_base", "vit_large"])
    g.add_argument("--dino-checkpoint", type=str, default=None,
                   help="Path to DINO pretrained checkpoint (.pth)")
    g.add_argument("--align-head-type", type=str, default="MLP",
                   choices=["attention", "MLP", "MLP3", "linear"])
    g.add_argument("--wheat-teacher-weights", type=str, default=None,
                   help="Optional frozen wheat teacher weights for online pseudo-label generation")
    g.add_argument("--wheat-teacher-score-threshold", type=float, default=None,
                   help="Optional score threshold for online wheat teacher predictions; None uses Ultralytics default")
    g.add_argument("--wheat-teacher-max-detections", type=int, default=300,
                   help="Max detections per image for online wheat teacher predictions")

    # ---- training ----
    g3 = p.add_argument_group("Training")
    g3.add_argument("--epochs", type=int, default=100)
    g3.add_argument("--batch-size", type=int, default=8)
    g3.add_argument("--batch-size-label", type=int, default=None,
                    help="Per-step labeled stream batch size (dual-stream mode)")
    g3.add_argument("--batch-size-unlabel", type=int, default=None,
                    help="Per-step unlabeled stream batch size (dual-stream mode)")
    g3.add_argument("--num-workers", type=int, default=4)
    g3.add_argument("--early-stopping-patience", type=int, default=None,
                    help="Early stopping patience in epochs (None = keep config default)")
    g3.add_argument("--lr", type=float, default=1e-4)
    g3.add_argument("--weight-decay", type=float, default=1e-4)
    g3.add_argument("--lr-scheduler", type=str, default="cosine",
                    choices=["cosine", "step"])
    g3.add_argument("--warmup-epochs", type=int, default=5)
    g3.add_argument("--gradient-clip", type=float, default=1.0)
    g3.add_argument("--mixed-precision", action="store_true", default=True)
    g3.add_argument("--no-mixed-precision", dest="mixed_precision",
                    action="store_false")
    g3.add_argument("--ema", dest="ema_enabled", action="store_true", default=True,
                    help="Enable student EMA for validation/export")
    g3.add_argument("--no-ema", dest="ema_enabled", action="store_false",
                    help="Disable student EMA")
    g3.add_argument("--ema-decay", type=float, default=0.9999,
                    help="EMA decay factor for the student model")
    g3.add_argument("--ema-tau", type=float, default=2000.0,
                    help="EMA warmup horizon in updates (0 disables EMA ramp)")

    # ---- stages ----
    g4 = p.add_argument_group("Two-Stage Schedule")
    g4.add_argument("--burn-up-epochs", type=int, default=0,
                    help="Deprecated/ignored (kept for backward compatibility)")
    g4.add_argument("--align-target-start", type=int, default=10,
                    help="Epoch to start target alignment + pseudo-label loss")
    g4.add_argument("--feature-align-weight", type=float, default=1.0,
                    help="Source feature alignment loss weight")
    g4.add_argument("--feature-align-weight-target", type=float, default=1.0,
                    help="Target feature alignment loss weight (full stage)")
    g4.add_argument("--use-target-alignment", dest="use_target_alignment",
                    action="store_true", default=False,
                    help="Enable target feature alignment (only when source/target data are separate)")
    g4.add_argument("--no-target-alignment", dest="use_target_alignment",
                    action="store_false",
                    help="Disable target feature alignment (recommended for single-domain runs)")
    g4.add_argument("--separate-source-target-data", action="store_true", default=False,
                    help="Set true only if training uses distinct source and target data streams")
    g4.add_argument("--unsup-loss-weight", type=float, default=4.0,
                    help="Pseudo-label (unsupervised) loss multiplier")
    g4.add_argument(
        "--prediction-align-mode",
        type=str,
        default="ultralytics",
        choices=["ultralytics", "legacy"],
        help="Prediction alignment path: Ultralytics criterion/assigner or legacy trainer branch",
    )
    g4.add_argument("--zero-pseudo-box-reg", action="store_true", default=False,
                    help="Disable pseudo bbox/dfl regression (cls only)")
    g4.add_argument("--no-zero-pseudo-box-reg", dest="zero_pseudo_box_reg",
                    action="store_false",
                    help="Enable pseudo bbox/dfl regression (IoU/probIoU + DFL)")
    g4.add_argument("--align-easy-only", action="store_true", default=False,
                    help="DINO teacher sees only un-augmented images "
                         "(requires dataset images_weak support)")
    g4.add_argument(
        "--supervision-mode",
        type=str,
        default="gt+pseudo",
        choices=["gt+pseudo", "gt-only", "pseudo-only"],
        help="Supervision recipe for detection losses",
    )

    # ---- pseudo labels ----
    g5 = p.add_argument_group("Pseudo Labels")
    g5.add_argument("--pseudo-label-dir", type=str, default=None,
                    help="Directory of YOLO .txt pseudo-label files")
    g5.add_argument("--pseudo-csv", type=str, default=None,
                    help="CSV file with pseudo-labels")
    g5.add_argument("--pseudo-mode", type=str, default="auto",
                    choices=["auto", "offline", "online", "none"],
                    help="Pseudo label source mode")
    g5.add_argument("--pseudo-score-threshold", type=float, default=None,
                    help="Optional extra MTKD-side pseudo score filter; None keeps Ultralytics-native thresholds")

    # ---- data ----
    g6 = p.add_argument_group("Data")
    g6.add_argument("--dataset-root", type=str,
                    default="Stomata_Dataset")
    g6.add_argument("--image-subdir", type=str,
                    default="barley_category/barley_image_fresh-leaf")
    g6.add_argument("--label-subdir", type=str,
                    default="barley_category/barley_label_fresh-leaf")
    g6.add_argument("--unlabeled-dataset-root", type=str, default=None,
                    help="Optional root for unlabeled stream (dual-stream mode)")
    g6.add_argument("--unlabeled-image-subdir", type=str, default=None,
                    help="Image subdir for unlabeled stream (dual-stream mode)")
    g6.add_argument("--unlabeled-label-subdir", type=str, default=None,
                    help="Optional label subdir for unlabeled stream")
    g6.add_argument("--image-size", type=int, default=640)
    g6.add_argument("--val-ratio", type=float, default=0.1)
    g6.add_argument("--augmentation", dest="augmentation", action="store_true", default=True,
                    help="Enable geometric augmentation for student images")
    g6.add_argument("--no-augmentation", dest="augmentation", action="store_false",
                    help="Disable geometric augmentation (recommended for offline pseudo labels)")

    # ---- output ----
    g7 = p.add_argument_group("Output")
    g7.add_argument("--output-dir", type=str, default="outputs/mtkd_v2")
    g7.add_argument("--save-freq", type=int, default=5,
                    help="Save checkpoint every N epochs")
    g7.add_argument("--save-pth-checkpoints", dest="save_pth_checkpoints",
                    action="store_true", default=True,
                    help="Write .pth checkpoints (best_model.pth and periodic checkpoint_epoch_*.pth)")
    g7.add_argument("--no-save-pth-checkpoints", dest="save_pth_checkpoints",
                    action="store_false",
                    help="Disable .pth checkpoints and keep only student_best.pt export")
    g7.add_argument("--log-freq", type=int, default=10,
                    help="Log every N batches")
    g7.add_argument("--best-by", type=str, default="fitness",
                    choices=["loss", "map50", "map5095", "fitness"],
                    help="Metric used to select best model")
    g7.add_argument("--map-data", type=str, default=None,
                    help="Dataset yaml for mAP-based best selection")
    g7.add_argument("--map-split", type=str, default="val",
                    help="Split for mAP evaluation (val/test)")
    g7.add_argument("--map-imgsz", type=int, default=640,
                    help="Image size for mAP evaluation")
    g7.add_argument("--map-batch", type=int, default=16,
                    help="Batch size for mAP evaluation")
    g7.add_argument("--map-conf", type=float, default=None,
                    help="Confidence threshold for mAP evaluation; None uses Ultralytics val default")
    g7.add_argument("--map-iou", type=float, default=None,
                    help="NMS IoU for mAP evaluation; None uses Ultralytics val default")
    g7.add_argument("--map-eval-interval", type=int, default=1,
                    help="Run mAP evaluation every N epochs when best-by is mAP")

    # ---- misc ----
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", type=str,
                   default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--resume", type=str, default=None,
                   help="Path to checkpoint to resume from")

    return p


def _looks_like_known_option(
    token: str,
    option_to_action: dict[str, argparse.Action],
) -> bool:
    if token in option_to_action:
        return True
    if token.startswith("--") and "=" in token:
        return token.split("=", 1)[0] in option_to_action
    return False


def _advance_cli_index(
    argv: list[str],
    index: int,
    action: argparse.Action,
    option_to_action: dict[str, argparse.Action],
) -> int:
    nargs = action.nargs

    if nargs == 0:
        return index + 1

    if nargs is None:
        return min(index + 2, len(argv))

    if nargs == "?":
        if index + 1 >= len(argv):
            return index + 1
        next_token = argv[index + 1]
        if next_token == "--" or _looks_like_known_option(next_token, option_to_action):
            return index + 1
        return min(index + 2, len(argv))

    if isinstance(nargs, int):
        return min(index + 1 + nargs, len(argv))

    if nargs in {"*", "+"}:
        j = index + 1
        while j < len(argv):
            next_token = argv[j]
            if next_token == "--" or _looks_like_known_option(next_token, option_to_action):
                break
            j += 1
        return j

    return index + 1


def _collect_explicit_cli_dests(
    parser: argparse.ArgumentParser,
    argv: list[str],
) -> set[str]:
    option_to_action: dict[str, argparse.Action] = {}
    for action in parser._actions:
        for option in action.option_strings:
            option_to_action[option] = action

    explicit: set[str] = set()
    i = 0
    while i < len(argv):
        token = argv[i]
        if token == "--":
            break

        action: argparse.Action | None = None
        if token.startswith("-") and token != "-":
            option_key = token
            if token.startswith("--") and "=" in token:
                option_key = token.split("=", 1)[0]
            action = option_to_action.get(option_key)

        if action is None:
            i += 1
            continue

        explicit.add(action.dest)

        if token.startswith("--") and "=" in token:
            i += 1
            continue

        i = _advance_cli_index(argv, i, action, option_to_action)

    return explicit


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = _build_parser()
    args = parser.parse_args(argv)
    raw_argv = list(sys.argv[1:] if argv is None else argv)
    setattr(args, "_explicit_cli_dests", _collect_explicit_cli_dests(parser, raw_argv))
    return args


def _load_user_config(config_path: Path) -> dict:
    """Load a user config from JSON or YAML file."""
    suffix = config_path.suffix.lower()
    if suffix in {".yaml", ".yml"}:
        try:
            import yaml
        except ImportError as exc:
            raise ImportError(
                "YAML config requires PyYAML. Install with: pip install pyyaml"
            ) from exc

        with open(config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
            config = {} if config is None else config
    else:
        with open(config_path, "r", encoding="utf-8") as f:
            config = json.load(f)

    if not isinstance(config, dict):
        raise ValueError(
            f"Config file must contain a mapping/object at top level, got {type(config).__name__}"
        )

    return config


def _deep_update_dict(base: dict, update: dict) -> dict:
    for key, value in update.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            _deep_update_dict(base[key], value)
        else:
            base[key] = value
    return base


def args_to_config(
    args: argparse.Namespace,
    explicit_cli_dests: set[str] | None = None,
) -> dict:
    """Convert parsed CLI args into the nested config dict expected by
    ``MTKDTrainerV2``."""
    try:
        from .train_v2 import get_default_config_v2
    except ImportError:
        from mtkd_framework.train_v2 import get_default_config_v2

    if explicit_cli_dests is None:
        explicit_cli_dests = getattr(args, "_explicit_cli_dests", None)

    def _should_override(dest: str) -> bool:
        return explicit_cli_dests is None or dest in explicit_cli_dests

    def _set_if_override(section: dict, key: str, value, dest: str) -> None:
        if _should_override(dest):
            section[key] = value

    config = get_default_config_v2()
    if args.config:
        config_path = Path(args.config)
        if not config_path.is_file():
            raise FileNotFoundError(f"Config file not found: {config_path}")
        user_config = _load_user_config(config_path)
        _deep_update_dict(config, user_config)

    # ---- model ----
    m = config.setdefault("model", {})
    _set_if_override(m, "num_classes", args.num_classes, "num_classes")
    if _should_override("student_weights"):
        m.setdefault("student_config", {})["weights"] = args.student_weights
    _set_if_override(m, "student_align_layer", args.student_align_layer, "student_align_layer")
    if _should_override("dino_model"):
        m.setdefault("dino_config", {})["model_name"] = args.dino_model
    if _should_override("dino_checkpoint"):
        m["dino_checkpoint"] = args.dino_checkpoint
    if _should_override("align_head_type"):
        m.setdefault("align_head_config", {})["head_type"] = args.align_head_type
    if _should_override("wheat_teacher_weights") and args.wheat_teacher_weights:
        m["wheat_teacher_config"] = {
            "weights": args.wheat_teacher_weights,
            "score_threshold": args.wheat_teacher_score_threshold,
            "max_detections": args.wheat_teacher_max_detections,
            "num_classes": args.num_classes,
        }
    elif _should_override("wheat_teacher_score_threshold") or _should_override("wheat_teacher_max_detections"):
        wheat_cfg = m.get("wheat_teacher_config")
        if isinstance(wheat_cfg, dict):
            if _should_override("wheat_teacher_score_threshold"):
                wheat_cfg["score_threshold"] = args.wheat_teacher_score_threshold
            if _should_override("wheat_teacher_max_detections"):
                wheat_cfg["max_detections"] = args.wheat_teacher_max_detections
            if _should_override("num_classes"):
                wheat_cfg["num_classes"] = args.num_classes

    # Keep explicit None to avoid legacy config enabling MTKD-side FFT blocks.
    m["fft_block_config"] = None

    # ---- training ----
    t = config.setdefault("training", {})
    _set_if_override(t, "epochs", args.epochs, "epochs")
    _set_if_override(t, "batch_size", args.batch_size, "batch_size")
    _set_if_override(t, "batch_size_label", args.batch_size_label, "batch_size_label")
    _set_if_override(t, "batch_size_unlabel", args.batch_size_unlabel, "batch_size_unlabel")
    _set_if_override(t, "num_workers", args.num_workers, "num_workers")
    if _should_override("early_stopping_patience") and args.early_stopping_patience is not None:
        t["early_stopping_patience"] = int(args.early_stopping_patience)
    _set_if_override(t, "learning_rate", args.lr, "lr")
    _set_if_override(t, "weight_decay", args.weight_decay, "weight_decay")
    _set_if_override(t, "lr_scheduler", args.lr_scheduler, "lr_scheduler")
    _set_if_override(t, "warmup_epochs", args.warmup_epochs, "warmup_epochs")
    _set_if_override(t, "gradient_clip_max_norm", args.gradient_clip, "gradient_clip")
    _set_if_override(t, "mixed_precision", args.mixed_precision, "mixed_precision")
    _set_if_override(t, "ema_enabled", args.ema_enabled, "ema_enabled")
    _set_if_override(t, "ema_decay", args.ema_decay, "ema_decay")
    _set_if_override(t, "ema_tau", args.ema_tau, "ema_tau")
    _set_if_override(t, "burn_up_epochs", args.burn_up_epochs, "burn_up_epochs")
    _set_if_override(t, "align_target_start_epoch", args.align_target_start, "align_target_start")
    _set_if_override(t, "feature_align_loss_weight", args.feature_align_weight, "feature_align_weight")
    _set_if_override(t, "feature_align_loss_weight_target", args.feature_align_weight_target,
                     "feature_align_weight_target")
    _set_if_override(t, "use_target_alignment", args.use_target_alignment, "use_target_alignment")
    _set_if_override(t, "separate_source_target_data", args.separate_source_target_data,
                     "separate_source_target_data")
    _set_if_override(t, "unsup_loss_weight", args.unsup_loss_weight, "unsup_loss_weight")
    _set_if_override(t, "prediction_align_mode", args.prediction_align_mode, "prediction_align_mode")
    _set_if_override(t, "zero_pseudo_box_reg", args.zero_pseudo_box_reg, "zero_pseudo_box_reg")
    _set_if_override(t, "align_easy_only", args.align_easy_only, "align_easy_only")
    _set_if_override(t, "supervision_mode", args.supervision_mode, "supervision_mode")

    # ---- pseudo labels ----
    pl = config.setdefault("pseudo_labels", {})
    if _should_override("pseudo_label_dir") and args.pseudo_label_dir is not None:
        pl["label_dir"] = args.pseudo_label_dir
    if _should_override("pseudo_csv") and args.pseudo_csv is not None:
        pl["csv_path"] = args.pseudo_csv
    _set_if_override(pl, "mode", args.pseudo_mode, "pseudo_mode")
    _set_if_override(pl, "score_threshold", args.pseudo_score_threshold, "pseudo_score_threshold")

    # ---- data ----
    d = config.setdefault("data", {})
    _set_if_override(d, "dataset_root", args.dataset_root, "dataset_root")
    _set_if_override(d, "image_subdir", args.image_subdir, "image_subdir")
    _set_if_override(d, "label_subdir", args.label_subdir, "label_subdir")
    _set_if_override(d, "unlabeled_dataset_root", args.unlabeled_dataset_root, "unlabeled_dataset_root")
    _set_if_override(d, "unlabeled_image_subdir", args.unlabeled_image_subdir, "unlabeled_image_subdir")
    _set_if_override(d, "unlabeled_label_subdir", args.unlabeled_label_subdir, "unlabeled_label_subdir")
    _set_if_override(d, "image_size", args.image_size, "image_size")
    _set_if_override(d, "val_ratio", args.val_ratio, "val_ratio")
    _set_if_override(d, "augmentation", args.augmentation, "augmentation")

    # ---- output ----
    o = config.setdefault("output", {})
    _set_if_override(o, "save_dir", args.output_dir, "output_dir")
    _set_if_override(o, "save_freq", args.save_freq, "save_freq")
    _set_if_override(o, "save_pth_checkpoints", args.save_pth_checkpoints, "save_pth_checkpoints")
    _set_if_override(o, "log_freq", args.log_freq, "log_freq")
    _set_if_override(o, "best_by", args.best_by, "best_by")
    _set_if_override(o, "map_data", args.map_data, "map_data")
    _set_if_override(o, "map_split", args.map_split, "map_split")
    _set_if_override(o, "map_imgsz", args.map_imgsz, "map_imgsz")
    _set_if_override(o, "map_batch", args.map_batch, "map_batch")
    _set_if_override(o, "map_conf", args.map_conf, "map_conf")
    _set_if_override(o, "map_iou", args.map_iou, "map_iou")
    _set_if_override(o, "map_eval_interval", args.map_eval_interval, "map_eval_interval")

    # ---- misc ----
    if _should_override("seed"):
        config["seed"] = args.seed
    if _should_override("device"):
        config["device"] = args.device
    if _should_override("resume") and args.resume:
        config.setdefault("checkpoints", {})["resume"] = args.resume

    return config


def main():
    args = parse_args()
    config = args_to_config(args)

    sw = (args.student_weights or "").lower()
    if sw.endswith(".yaml") or sw.endswith(".yml"):
        print(
            "[WARN] --student-weights points to a YAML architecture. "
            "This starts YOLO student from random initialization, not from a pretrained .pt checkpoint."
        )

    try:
        from .train_v2 import MTKDTrainerV2
        from .data.stomata_dataset import (
            build_stomata_dataloaders,
            build_stomata_semisup_dataloaders,
        )
    except ImportError:
        from mtkd_framework.train_v2 import MTKDTrainerV2
        from mtkd_framework.data.stomata_dataset import (
            build_stomata_dataloaders,
            build_stomata_semisup_dataloaders,
        )

    data_cfg = config.get("data", {})
    training_cfg = config.get("training", {})
    if bool(training_cfg.get("separate_source_target_data", False)):
        unlabeled_image_subdir = data_cfg.get("unlabeled_image_subdir")
        if not unlabeled_image_subdir:
            raise ValueError(
                "separate_source_target_data=True requires data.unlabeled_image_subdir "
                "(or --unlabeled-image-subdir)."
            )

        bs_label = training_cfg.get("batch_size_label")
        bs_unlabel = training_cfg.get("batch_size_unlabel")
        if bs_label is None:
            bs_label = training_cfg.get("batch_size", 8)
        if bs_unlabel is None:
            bs_unlabel = training_cfg.get("batch_size", 8)

        train_loader, val_loader = build_stomata_semisup_dataloaders(
            dataset_root=data_cfg.get("dataset_root", "Stomata_Dataset"),
            image_subdir=data_cfg.get("image_subdir", "barley_category/barley_image_fresh-leaf"),
            label_subdir=data_cfg.get("label_subdir", "barley_category/barley_label_fresh-leaf"),
            unlabeled_dataset_root=data_cfg.get("unlabeled_dataset_root"),
            unlabeled_image_subdir=unlabeled_image_subdir,
            unlabeled_label_subdir=data_cfg.get("unlabeled_label_subdir"),
            image_size=data_cfg.get("image_size", 640),
            val_ratio=data_cfg.get("val_ratio", 0.1),
            batch_size_label=int(bs_label),
            batch_size_unlabel=int(bs_unlabel),
            num_workers=training_cfg.get("num_workers", 4),
            seed=config.get("seed", 42),
            augmentation=data_cfg.get("augmentation", True),
        )
    else:
        train_loader, val_loader = build_stomata_dataloaders(
            dataset_root=data_cfg.get("dataset_root", "Stomata_Dataset"),
            image_subdir=data_cfg.get("image_subdir", "barley_category/barley_image_fresh-leaf"),
            label_subdir=data_cfg.get("label_subdir", "barley_category/barley_label_fresh-leaf"),
            image_size=data_cfg.get("image_size", 640),
            val_ratio=data_cfg.get("val_ratio", 0.1),
            batch_size=training_cfg.get("batch_size", 8),
            num_workers=training_cfg.get("num_workers", 4),
            seed=config.get("seed", 42),
            augmentation=data_cfg.get("augmentation", True),
        )

    trainer = MTKDTrainerV2(config, train_loader=train_loader, val_loader=val_loader)

    print("=" * 60)
    print("  MTKDv2 — DINO-Teacher-aligned training")
    print("=" * 60)
    print(f"  Device        : {config['device']}")
    print(f"  Epochs        : {config['training']['epochs']}")
    print(f"  Batch size    : {config['training']['batch_size']}")
    print(
        f"  Student EMA   : {config['training'].get('ema_enabled', True)} "
        f"(decay={config['training'].get('ema_decay', 0.9999)}, "
        f"tau={config['training'].get('ema_tau', 2000.0)})"
    )
    print("  Source align  : epoch 0+")
    print(f"  Target align  : epoch {config['training']['align_target_start_epoch']}+")
    if int(config['training'].get('burn_up_epochs', 0) or 0) > 0:
        print("  Note          : burn_up_epochs is deprecated and ignored")
    print(f"  Dual-stream   : {config['training'].get('separate_source_target_data', False)}")
    if config['training'].get('separate_source_target_data', False):
        print(
            f"  Batch(L/U)    : "
            f"{config['training'].get('batch_size_label', config['training']['batch_size'])}/"
            f"{config['training'].get('batch_size_unlabel', config['training']['batch_size'])}"
        )
        print(
            f"  Unlabeled dir : "
            f"{config['data'].get('unlabeled_dataset_root', config['data'].get('dataset_root'))}"
            f"/{config['data'].get('unlabeled_image_subdir', '')}"
        )
    print(f"  Use target-align term: {config['training'].get('use_target_alignment', True)}")
    print(f"  DINO ckpt     : {config['model'].get('dino_checkpoint', 'None')}")
    wheat_cfg = config["model"].get("wheat_teacher_config") or {}
    print(
        f"  Wheat teacher : "
        f"{wheat_cfg.get('weights', 'disabled')}"
    )
    print(f"  Pseudo mode   : {config['pseudo_labels'].get('mode', 'auto')}")
    print(f"  Pseudo labels : {config['pseudo_labels'].get('label_dir', 'None')}")
    print(f"  Augmentation  : {config['data'].get('augmentation', True)}")
    print(f"  Output        : {config['output']['save_dir']}")
    print("=" * 60)

    trainer.train()


if __name__ == "__main__":
    main()
