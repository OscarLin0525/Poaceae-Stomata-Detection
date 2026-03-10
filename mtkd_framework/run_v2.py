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
    g.add_argument("--num-classes", type=int, default=1)
    g.add_argument("--student-weights", type=str, default="yolo11s.pt",
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
    g4.add_argument("--unsup-loss-weight", type=float, default=4.0,
                    help="Pseudo-label (unsupervised) loss multiplier")
    g4.add_argument("--zero-pseudo-box-reg", action="store_true", default=True,
                    help="Zero out pseudo bbox/dfl loss (cls only)")
    g4.add_argument("--no-zero-pseudo-box-reg", dest="zero_pseudo_box_reg",
                    action="store_false")
    g4.add_argument("--align-easy-only", action="store_true", default=False,
                    help="DINO teacher sees only un-augmented images "
                         "(requires dataset images_weak support)")

    # ---- pseudo labels ----
    g5 = p.add_argument_group("Pseudo Labels")
    g5.add_argument("--pseudo-label-dir", type=str, default=None,
                    help="Directory of YOLO .txt pseudo-label files")
    g5.add_argument("--pseudo-csv", type=str, default=None,
                    help="CSV file with pseudo-labels")
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

    # ---- output ----
    g7 = p.add_argument_group("Output")
    g7.add_argument("--output-dir", type=str, default="outputs/mtkd_v2")
    g7.add_argument("--save-freq", type=int, default=5,
                    help="Save checkpoint every N epochs")
    g7.add_argument("--log-freq", type=int, default=10,
                    help="Log every N batches")

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
    from .train_v2 import get_default_config_v2

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
    t["unsup_loss_weight"] = args.unsup_loss_weight
    t["zero_pseudo_box_reg"] = args.zero_pseudo_box_reg
    t["align_easy_only"] = args.align_easy_only

    # ---- pseudo labels ----
    pl = config.setdefault("pseudo_labels", {})
    if args.pseudo_label_dir:
        pl["label_dir"] = args.pseudo_label_dir
    if args.pseudo_csv:
        pl["csv_path"] = args.pseudo_csv
    pl["score_threshold"] = args.pseudo_score_threshold

    # ---- data ----
    d = config.setdefault("data", {})
    d["dataset_root"] = args.dataset_root
    d["image_subdir"] = args.image_subdir
    d["label_subdir"] = args.label_subdir
    d["image_size"] = args.image_size
    d["val_ratio"] = args.val_ratio

    # ---- output ----
    o = config.setdefault("output", {})
    o["save_dir"] = args.output_dir
    o["save_freq"] = args.save_freq
    o["log_freq"] = args.log_freq

    # ---- misc ----
    config["seed"] = args.seed
    config["device"] = args.device
    if args.resume:
        config.setdefault("checkpoints", {})["resume"] = args.resume

    return config


def main():
    args = parse_args()
    config = args_to_config(args)

    from .train_v2 import MTKDTrainerV2
    trainer = MTKDTrainerV2(config)

    print("=" * 60)
    print("  MTKDv2 — DINO-Teacher-aligned training")
    print("=" * 60)
    print(f"  Device        : {config['device']}")
    print(f"  Epochs        : {config['training']['epochs']}")
    print(f"  Batch size    : {config['training']['batch_size']}")
    print(f"  Burn-up       : {config['training']['burn_up_epochs']} epochs")
    print(f"  Target align  : epoch {config['training']['align_target_start_epoch']}+")
    print(f"  DINO ckpt     : {config['model'].get('dino_checkpoint', 'None')}")
    print(f"  FFT blocks    : {config['model'].get('fft_block_config', 'disabled')}")
    print(f"  Pseudo labels : {config['pseudo_labels'].get('label_dir', 'None')}")
    print(f"  Output        : {config['output']['save_dir']}")
    print("=" * 60)

    trainer.train()


if __name__ == "__main__":
    main()
