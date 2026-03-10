#!/usr/bin/env python3
"""
MTKD v2 Ablation Experiments
=============================

Four settings compared on barley fresh-leaf (84 images):

  A — Baseline:      YOLO11s supervised only (no DINO, no FFT)
  B — +DINO Align:   A + frozen DINO feature alignment
  C — +FFT:          B + FFT block injected into DINO
  D — +Pseudo:       C + wheat→barley pseudo-labels

Usage:
  python run_ablation.py                     # run all 4 settings
  python run_ablation.py --settings A B      # run specific settings
  python run_ablation.py --epochs 50         # override epoch count
  python run_ablation.py --settings D --epochs 100 --batch-size 4
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import sys
import time
from pathlib import Path

import torch

# Ensure the project root is importable
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from mtkd_framework.data.stomata_dataset import build_stomata_dataloaders
from mtkd_framework.train_v2 import MTKDTrainerV2, get_default_config_v2
from mtkd_framework.models.mtkd_model_v2 import build_mtkd_model_v2
from mtkd_framework.utils.detection_eval import evaluate_detection, save_detection_visualizations


# ======================================================================
# Experiment definitions
# ======================================================================

DINO_CHECKPOINT = str(
    PROJECT_ROOT / "dinov3-main" / "weight folder"
    / "dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth"
)

PSEUDO_LABEL_DIR = str(PROJECT_ROOT / "pseudo_labels_wheat100" / "fresh" / "labels")


def make_setting_A(base: dict) -> dict:
    """A — Baseline: YOLO11s supervised only."""
    cfg = copy.deepcopy(base)
    # Still build DINO (unavoidable in v2 model), but don't load weights
    # and disable all DINO-dependent stages via high epoch thresholds.
    cfg["model"]["dino_checkpoint"] = None
    cfg["model"]["fft_block_config"] = None  # no FFT
    # Disable alignment and pseudo stages
    cfg["training"]["burn_up_epochs"] = 9999  # always burn-in
    cfg["training"]["align_target_start_epoch"] = 9999
    cfg["pseudo_labels"]["label_dir"] = None
    cfg["output"]["save_dir"] = str(PROJECT_ROOT / "outputs" / "ablation_A_baseline")
    return cfg


def make_setting_B(base: dict) -> dict:
    """B — +DINO Align: add frozen DINO feature alignment."""
    cfg = copy.deepcopy(base)
    cfg["model"]["dino_checkpoint"] = DINO_CHECKPOINT
    cfg["model"]["fft_block_config"] = None  # no FFT
    cfg["training"]["burn_up_epochs"] = 5
    cfg["training"]["align_target_start_epoch"] = 9999  # no pseudo
    cfg["pseudo_labels"]["label_dir"] = None
    cfg["output"]["save_dir"] = str(PROJECT_ROOT / "outputs" / "ablation_B_dino_align")
    return cfg


def make_setting_C(base: dict) -> dict:
    """C — +FFT: add FFT block injected into DINO."""
    cfg = copy.deepcopy(base)
    cfg["model"]["dino_checkpoint"] = DINO_CHECKPOINT
    cfg["model"]["fft_block_config"] = {
        "after_blocks": [9],
        "num_freq_bins": 32,
        "hidden_dim": 256,
        "init_gate": -5.0,
        "modulation_mode": "multiplicative",
    }
    cfg["training"]["burn_up_epochs"] = 5
    cfg["training"]["align_target_start_epoch"] = 9999  # no pseudo
    cfg["pseudo_labels"]["label_dir"] = None
    cfg["output"]["save_dir"] = str(PROJECT_ROOT / "outputs" / "ablation_C_fft")
    return cfg


def make_setting_D(base: dict) -> dict:
    """D — +Pseudo: full MTKD (DINO + FFT + pseudo-labels)."""
    cfg = copy.deepcopy(base)
    cfg["model"]["dino_checkpoint"] = DINO_CHECKPOINT
    cfg["model"]["fft_block_config"] = {
        "after_blocks": [9],
        "num_freq_bins": 32,
        "hidden_dim": 256,
        "init_gate": -5.0,
        "modulation_mode": "multiplicative",
    }
    cfg["training"]["burn_up_epochs"] = 5
    cfg["training"]["align_target_start_epoch"] = 10
    cfg["training"]["unsup_loss_weight"] = 4.0
    cfg["training"]["zero_pseudo_box_reg"] = True
    cfg["pseudo_labels"]["label_dir"] = PSEUDO_LABEL_DIR
    cfg["pseudo_labels"]["score_threshold"] = 0.5
    cfg["pseudo_labels"]["convert_obb"] = True
    cfg["output"]["save_dir"] = str(PROJECT_ROOT / "outputs" / "ablation_D_full")
    return cfg


SETTINGS = {
    "A": ("Baseline (supervised only)", make_setting_A),
    "B": ("+DINO Align", make_setting_B),
    "C": ("+DINO +FFT", make_setting_C),
    "D": ("+DINO +FFT +Pseudo (full MTKD)", make_setting_D),
}


# ======================================================================
# Runner
# ======================================================================

def run_experiment(name: str, config: dict, train_loader, val_loader):
    """Run one experiment setting and return metrics."""
    print(f"\n{'='*60}")
    print(f"  Setting {name}: {SETTINGS[name][0]}")
    print(f"  Output: {config['output']['save_dir']}")
    print(f"{'='*60}\n")

    os.makedirs(config["output"]["save_dir"], exist_ok=True)

    # Build model
    model = build_mtkd_model_v2(config["model"])

    # Build trainer with dataloaders
    trainer = MTKDTrainerV2(
        config,
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
    )

    # Train
    t0 = time.time()
    trainer.train()
    elapsed = time.time() - t0

    # --- Object detection evaluation (mAP, Precision, Recall) ---
    device = torch.device(config.get("device", "cpu"))
    print(f"  Running detection evaluation ...")
    det_metrics = evaluate_detection(
        trainer.model, val_loader, device,
        conf_threshold=0.25, iou_threshold_nms=0.45,
        num_classes=config["model"]["num_classes"],
    )
    print(f"  mAP@0.5={det_metrics['mAP@0.5']:.4f}  "
          f"P={det_metrics['precision@0.5']:.4f}  "
          f"R={det_metrics['recall@0.5']:.4f}  "
          f"({det_metrics['n_predictions']} preds / {det_metrics['n_ground_truth']} GT)")

    # --- Save bbox visualization images ---
    print(f"  Saving detection visualizations ...")
    n_saved = save_detection_visualizations(
        trainer.model, val_loader, device,
        save_dir=config["output"]["save_dir"],
        conf_threshold=0.25, iou_threshold_nms=0.45,
        num_classes=config["model"]["num_classes"],
    )
    print(f"  Saved {n_saved} images to {config['output']['save_dir']}/detections/")

    # --- Export YOLO student as ultralytics .pt ---
    yolo_pt_path = os.path.join(config["output"]["save_dir"], "best.pt")
    trainer.model.student.export_ultralytics_pt(
        save_path=yolo_pt_path,
        num_classes=config["model"]["num_classes"],
        class_names={0: "stomata"},
        epoch=config["training"]["epochs"] - 1,
        best_fitness=trainer.best_loss,
    )
    print(f"  Exported YOLO weights: {yolo_pt_path}")
    print(f"  → Use: yolo predict model={yolo_pt_path} source=YOUR_IMAGES/")

    result = {
        "setting": name,
        "description": SETTINGS[name][0],
        "epochs": config["training"]["epochs"],
        "train_time_sec": round(elapsed, 1),
        "best_loss": round(trainer.best_loss, 4),
        **det_metrics,
    }

    # Save per-setting result
    result_path = os.path.join(config["output"]["save_dir"], "result.json")
    with open(result_path, "w") as f:
        json.dump(result, f, indent=2)

    return result


def print_summary_table(results: list):
    """Print a nice comparison table."""
    print(f"\n{'='*96}")
    print("  ABLATION RESULTS SUMMARY  (Object Detection Metrics)")
    print(f"{'='*96}")

    header = (f"{'Setting':<8} {'Description':<24} {'mAP@.5':<9} {'mAP@.75':<9} "
              f"{'mAP@.5:.95':<11} {'Prec':<8} {'Rec':<8} {'F1':<8} {'Loss':<9} {'Time':<8}")
    print(header)
    print("-" * 96)

    for r in results:
        line = (
            f"{r['setting']:<8} "
            f"{r['description']:<24} "
            f"{r.get('mAP@0.5', 0):<9.4f} "
            f"{r.get('mAP@0.75', 0):<9.4f} "
            f"{r.get('mAP@0.5:0.95', 0):<11.4f} "
            f"{r.get('precision@0.5', 0):<8.4f} "
            f"{r.get('recall@0.5', 0):<8.4f} "
            f"{r.get('f1@0.5', 0):<8.4f} "
            f"{r['best_loss']:<9.4f} "
            f"{r['train_time_sec']}s"
        )
        print(line)

    print(f"{'='*96}\n")


def main():
    parser = argparse.ArgumentParser(description="MTKD v2 Ablation Study")
    parser.add_argument("--settings", nargs="+", default=["A", "B", "C", "D"],
                        choices=["A", "B", "C", "D"],
                        help="Which settings to run (default: all)")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--image-size", type=int, default=640)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str,
                        default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    print(f"Device: {args.device}")
    print(f"Settings to run: {args.settings}")
    print(f"Epochs: {args.epochs}, Batch: {args.batch_size}, LR: {args.lr}")

    # Build dataloaders once (shared across settings for fair comparison)
    print("\nBuilding dataloaders...")
    train_loader, val_loader = build_stomata_dataloaders(
        dataset_root="Stomata_Dataset",
        image_subdir="barley_category/barley_image_fresh-leaf",
        label_subdir="barley_category/barley_label_fresh-leaf",
        image_size=args.image_size,
        val_ratio=0.1,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        seed=args.seed,
    )
    print(f"  Train: {len(train_loader)} batches ({len(train_loader.dataset)} images)")
    print(f"  Val:   {len(val_loader)} batches ({len(val_loader.dataset)} images)")

    # Build base config
    base_config = get_default_config_v2()
    base_config["training"]["epochs"] = args.epochs
    base_config["training"]["batch_size"] = args.batch_size
    base_config["training"]["learning_rate"] = args.lr
    base_config["training"]["num_workers"] = args.num_workers
    base_config["data"]["image_size"] = args.image_size
    base_config["output"]["log_freq"] = 5
    base_config["seed"] = args.seed
    base_config["device"] = args.device

    # Run each setting
    results = []
    for name in args.settings:
        _, make_fn = SETTINGS[name]
        config = make_fn(base_config)
        result = run_experiment(name, config, train_loader, val_loader)
        results.append(result)
        torch.cuda.empty_cache()

    # Save combined results
    output_dir = str(PROJECT_ROOT / "outputs")
    os.makedirs(output_dir, exist_ok=True)
    combined_path = os.path.join(output_dir, "ablation_results.json")
    with open(combined_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nCombined results saved to: {combined_path}")

    # Print summary
    print_summary_table(results)


if __name__ == "__main__":
    main()
