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
        --pseudo-label-dir runs/detect/predict/labels \\
        --pseudo-score-threshold 0.3

    # Custom config via JSON
    python -m mtkd_framework.run_v2 --config my_config.json

    # Override individual params
    python -m mtkd_framework.run_v2 \\
        --epochs 200 --batch-size 4 --lr 5e-5 \\
        --burn-up-epochs 10 --align-target-start 20 \\
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


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="MTKDv2 Trainer — DINO-Teacher-aligned training",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # ---- config file (optional — overrides defaults) ----
    p.add_argument("--config", type=str, default=None,
                   help="Path to a JSON config file. CLI args override values in the file.")

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
    g.add_argument("--wheat-teacher-score-threshold", type=float, default=0.3,
                   help="Score threshold for online wheat teacher predictions")
    g.add_argument("--wheat-teacher-max-detections", type=int, default=300,
                   help="Max detections per image for online wheat teacher predictions")

    # ---- FFT ----
    g2 = p.add_argument_group("FFT Block")
    g2.add_argument("--no-fft", action="store_true",
                    help="Disable FFT block injection into DINO")
    g2.add_argument("--fft-after-blocks", type=int, nargs="+", default=[9],
                    help="Insert FFT block after these DINO block indices")
    g2.add_argument("--fft-init-gate", type=float, default=-5.0)

    # ---- training ----
    g3 = p.add_argument_group("Training")
    g3.add_argument("--epochs", type=int, default=100)
    g3.add_argument("--batch-size", type=int, default=8)
    g3.add_argument("--num-workers", type=int, default=4)
    g3.add_argument("--lr", type=float, default=1e-4)
    g3.add_argument("--weight-decay", type=float, default=1e-4)
    g3.add_argument("--lr-scheduler", type=str, default="cosine",
                    choices=["cosine", "step"])
    g3.add_argument("--warmup-epochs", type=int, default=5)
    g3.add_argument("--gradient-clip", type=float, default=1.0)
    g3.add_argument("--mixed-precision", action="store_true", default=True)
    g3.add_argument("--no-mixed-precision", dest="mixed_precision",
                    action="store_false")

    # ---- stages ----
    g4 = p.add_argument_group("DINO-Teacher Stages")
    g4.add_argument("--burn-up-epochs", type=int, default=5,
                    help="Epochs of pure supervised burn-in")
    g4.add_argument("--align-target-start", type=int, default=10,
                    help="Epoch to start target alignment + pseudo-label loss")
    g4.add_argument("--feature-align-weight", type=float, default=1.0,
                    help="Source feature alignment loss weight")
    g4.add_argument("--feature-align-weight-target", type=float, default=1.0,
                    help="Target feature alignment loss weight (stage 3)")
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
    g4.add_argument("--separation-loss-weight", type=float, default=0.0,
                    help="Valley separation loss weight (encourages spatial gaps between stomata)")
    g4.add_argument("--separation-target-layer", type=int, default=10,
                    help="DINO layer to extract features for separation loss")
    g4.add_argument("--separation-sample-points", type=int, default=5,
                    help="Number of points to sample along connecting lines")
    g4.add_argument("--separation-valley-margin", type=float, default=0.2,
                    help="Minimum valley depth factor for separation loss")
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
    g5.add_argument("--pseudo-score-threshold", type=float, default=0.5)

    # ---- data ----
    g6 = p.add_argument_group("Data")
    g6.add_argument("--dataset-root", type=str,
                    default="Stomata_Dataset")
    g6.add_argument("--image-subdir", type=str,
                    default="barley_category/barley_image_fresh-leaf")
    g6.add_argument("--label-subdir", type=str,
                    default="barley_category/barley_label_fresh-leaf")
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
    g7.add_argument("--log-freq", type=int, default=10,
                    help="Log every N batches")
    g7.add_argument("--best-by", type=str, default="loss",
                    choices=["loss", "map50", "map5095"],
                    help="Metric used to select best model")
    g7.add_argument("--map-data", type=str, default=None,
                    help="Dataset yaml for mAP-based best selection")
    g7.add_argument("--map-split", type=str, default="val",
                    help="Split for mAP evaluation (val/test)")
    g7.add_argument("--map-imgsz", type=int, default=640,
                    help="Image size for mAP evaluation")
    g7.add_argument("--map-batch", type=int, default=16,
                    help="Batch size for mAP evaluation")
    g7.add_argument("--map-conf", type=float, default=0.25,
                    help="Confidence threshold for mAP evaluation")
    g7.add_argument("--map-iou", type=float, default=0.6,
                    help="NMS IoU for mAP evaluation")
    g7.add_argument("--map-eval-interval", type=int, default=1,
                    help="Run mAP evaluation every N epochs when best-by is mAP")

    # ---- misc ----
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", type=str,
                   default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--resume", type=str, default=None,
                   help="Path to checkpoint to resume from")

    return p.parse_args()


def args_to_config(args: argparse.Namespace) -> dict:
    """Convert parsed CLI args into the nested config dict expected by
    ``MTKDTrainerV2``."""
    try:
        from .train_v2 import get_default_config_v2
    except ImportError:
        from mtkd_framework.train_v2 import get_default_config_v2

    # Start from a JSON file if provided, else use defaults
    if args.config and Path(args.config).is_file():
        with open(args.config) as f:
            config = json.load(f)
    else:
        config = get_default_config_v2()

    # ---- model ----
    m = config.setdefault("model", {})
    m["num_classes"] = args.num_classes
    m.setdefault("student_config", {})["weights"] = args.student_weights
    m["student_align_layer"] = args.student_align_layer
    m.setdefault("dino_config", {})["model_name"] = args.dino_model
    if args.dino_checkpoint:
        m["dino_checkpoint"] = args.dino_checkpoint
    m.setdefault("align_head_config", {})["head_type"] = args.align_head_type
    if args.wheat_teacher_weights:
        m["wheat_teacher_config"] = {
            "weights": args.wheat_teacher_weights,
            "score_threshold": args.wheat_teacher_score_threshold,
            "max_detections": args.wheat_teacher_max_detections,
            "num_classes": args.num_classes,
        }

    if args.no_fft:
        m["fft_block_config"] = None
    else:
        m.setdefault("fft_block_config", {})["after_blocks"] = args.fft_after_blocks
        m["fft_block_config"]["init_gate"] = args.fft_init_gate

    # ---- training ----
    t = config.setdefault("training", {})
    t["epochs"] = args.epochs
    t["batch_size"] = args.batch_size
    t["num_workers"] = args.num_workers
    t["learning_rate"] = args.lr
    t["weight_decay"] = args.weight_decay
    t["lr_scheduler"] = args.lr_scheduler
    t["warmup_epochs"] = args.warmup_epochs
    t["gradient_clip_max_norm"] = args.gradient_clip
    t["mixed_precision"] = args.mixed_precision
    t["burn_up_epochs"] = args.burn_up_epochs
    t["align_target_start_epoch"] = args.align_target_start
    t["feature_align_loss_weight"] = args.feature_align_weight
    t["feature_align_loss_weight_target"] = args.feature_align_weight_target
    t["use_target_alignment"] = args.use_target_alignment
    t["separate_source_target_data"] = args.separate_source_target_data
    t["unsup_loss_weight"] = args.unsup_loss_weight
    t["prediction_align_mode"] = args.prediction_align_mode
    t["separation_loss_weight"] = args.separation_loss_weight
    t["separation_target_layer"] = args.separation_target_layer
    t["separation_sample_points"] = args.separation_sample_points
    t["separation_valley_margin"] = args.separation_valley_margin
    t["zero_pseudo_box_reg"] = args.zero_pseudo_box_reg
    t["align_easy_only"] = args.align_easy_only
    t["supervision_mode"] = args.supervision_mode

    # ---- pseudo labels ----
    pl = config.setdefault("pseudo_labels", {})
    if args.pseudo_label_dir:
        pl["label_dir"] = args.pseudo_label_dir
    if args.pseudo_csv:
        pl["csv_path"] = args.pseudo_csv
    pl["mode"] = args.pseudo_mode
    pl["score_threshold"] = args.pseudo_score_threshold

    # ---- data ----
    d = config.setdefault("data", {})
    d["dataset_root"] = args.dataset_root
    d["image_subdir"] = args.image_subdir
    d["label_subdir"] = args.label_subdir
    d["image_size"] = args.image_size
    d["val_ratio"] = args.val_ratio
    d["augmentation"] = args.augmentation

    # ---- output ----
    o = config.setdefault("output", {})
    o["save_dir"] = args.output_dir
    o["save_freq"] = args.save_freq
    o["log_freq"] = args.log_freq
    o["best_by"] = args.best_by
    o["map_data"] = args.map_data
    o["map_split"] = args.map_split
    o["map_imgsz"] = args.map_imgsz
    o["map_batch"] = args.map_batch
    o["map_conf"] = args.map_conf
    o["map_iou"] = args.map_iou
    o["map_eval_interval"] = args.map_eval_interval

    # ---- misc ----
    config["seed"] = args.seed
    config["device"] = args.device
    if args.resume:
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
        from .data.stomata_dataset import build_stomata_dataloaders
    except ImportError:
        from mtkd_framework.train_v2 import MTKDTrainerV2
        from mtkd_framework.data.stomata_dataset import build_stomata_dataloaders

    data_cfg = config.get("data", {})
    train_loader, val_loader = build_stomata_dataloaders(
        dataset_root=data_cfg.get("dataset_root", "Stomata_Dataset"),
        image_subdir=data_cfg.get("image_subdir", "barley_category/barley_image_fresh-leaf"),
        label_subdir=data_cfg.get("label_subdir", "barley_category/barley_label_fresh-leaf"),
        image_size=data_cfg.get("image_size", 640),
        val_ratio=data_cfg.get("val_ratio", 0.1),
        batch_size=config.get("training", {}).get("batch_size", 8),
        num_workers=config.get("training", {}).get("num_workers", 4),
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
    print(f"  Burn-up       : {config['training']['burn_up_epochs']} epochs")
    print(f"  Target align  : epoch {config['training']['align_target_start_epoch']}+")
    print(f"  Use target-align term: {config['training'].get('use_target_alignment', True)}")
    print(f"  DINO ckpt     : {config['model'].get('dino_checkpoint', 'None')}")
    wheat_cfg = config["model"].get("wheat_teacher_config") or {}
    print(
        f"  Wheat teacher : "
        f"{wheat_cfg.get('weights', 'disabled')}"
    )
    print(f"  FFT blocks    : {config['model'].get('fft_block_config', 'disabled')}")
    print(f"  Pseudo mode   : {config['pseudo_labels'].get('mode', 'auto')}")
    print(f"  Pseudo labels : {config['pseudo_labels'].get('label_dir', 'None')}")
    print(f"  Augmentation  : {config['data'].get('augmentation', True)}")
    print(f"  Output        : {config['output']['save_dir']}")
    print("=" * 60)

    trainer.train()


if __name__ == "__main__":
    main()
