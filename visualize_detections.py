#!/usr/bin/env python3
"""
Visualize detection results from a trained MTKD checkpoint.

Usage:
  # Visualize a specific setting's best model
  python visualize_detections.py --checkpoint outputs/ablation_B_dino_align/best_model.pth

  # Visualize all 4 ablation settings (auto-find best_model.pth)
  python visualize_detections.py --all-settings

  # Use train set instead of val
  python visualize_detections.py --checkpoint outputs/ablation_A_baseline/best_model.pth --split train
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from mtkd_framework.data.stomata_dataset import build_stomata_dataloaders
from mtkd_framework.models.mtkd_model_v2 import build_mtkd_model_v2
from mtkd_framework.utils.detection_eval import (
    evaluate_detection,
    save_detection_visualizations,
)
from mtkd_framework.utils import load_checkpoint


SETTING_DIRS = {
    "A": "outputs/ablation_A_baseline",
    "B": "outputs/ablation_B_dino_align",
    "C": "outputs/ablation_C_fft",
    "D": "outputs/ablation_D_full",
}


def load_model_from_setting(setting_dir: str, device: torch.device):
    """Load model from a setting directory's config + best_model.pth."""
    config_path = os.path.join(setting_dir, "config.json")
    ckpt_path = os.path.join(setting_dir, "best_model.pth")

    if not os.path.isfile(config_path):
        print(f"  [SKIP] No config.json in {setting_dir}")
        return None
    if not os.path.isfile(ckpt_path):
        print(f"  [SKIP] No best_model.pth in {setting_dir}")
        return None

    with open(config_path) as f:
        config = json.load(f)

    model = build_mtkd_model_v2(config["model"])
    load_checkpoint(model, ckpt_path)
    model = model.to(device)
    model.eval()
    return model


def main():
    parser = argparse.ArgumentParser(description="Visualize MTKD detection results")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--checkpoint", type=str, help="Path to a specific checkpoint")
    group.add_argument("--all-settings", action="store_true",
                       help="Visualize all 4 ablation settings")
    parser.add_argument("--split", choices=["val", "train"], default="val")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    parser.add_argument("--max-images", type=int, default=50)
    parser.add_argument("--image-size", type=int, default=640)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--device", type=str,
                        default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    device = torch.device(args.device)

    # Build dataloader
    train_loader, val_loader = build_stomata_dataloaders(
        dataset_root="Stomata_Dataset",
        image_subdir="barley_category/barley_image_fresh-leaf",
        label_subdir="barley_category/barley_label_fresh-leaf",
        image_size=args.image_size,
        val_ratio=0.1,
        batch_size=args.batch_size,
        num_workers=0,
        seed=42,
    )
    loader = val_loader if args.split == "val" else train_loader
    print(f"Using {args.split} set: {len(loader.dataset)} images")

    if args.all_settings:
        for name, sdir in SETTING_DIRS.items():
            print(f"\n--- Setting {name}: {sdir} ---")
            model = load_model_from_setting(sdir, device)
            if model is None:
                continue
            n = save_detection_visualizations(
                model, loader, device,
                save_dir=sdir,
                conf_threshold=args.conf,
                num_classes=1,
                max_images=args.max_images,
            )
            print(f"  Saved {n} images → {sdir}/detections/")
            torch.cuda.empty_cache()
    else:
        ckpt_dir = os.path.dirname(args.checkpoint)
        config_path = os.path.join(ckpt_dir, "config.json")
        if os.path.isfile(config_path):
            with open(config_path) as f:
                config = json.load(f)
            model = build_mtkd_model_v2(config["model"])
        else:
            # Fallback: default config
            from mtkd_framework.train_v2 import get_default_config_v2
            config = get_default_config_v2()
            model = build_mtkd_model_v2(config["model"])

        load_checkpoint(model, args.checkpoint)
        model = model.to(device)
        model.eval()

        save_dir = ckpt_dir or "outputs/vis"
        n = save_detection_visualizations(
            model, loader, device,
            save_dir=save_dir,
            conf_threshold=args.conf,
            num_classes=1,
            max_images=args.max_images,
        )
        print(f"Saved {n} images → {save_dir}/detections/")


if __name__ == "__main__":
    main()
