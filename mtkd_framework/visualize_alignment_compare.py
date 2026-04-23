#!/usr/bin/env python3
"""
Compare pre-training and post-training student PCA against DINO PCA.

Example:
    python -m mtkd_framework.visualize_alignment_compare \
        --config outputs/mtkd_train_epoch100_version0.1/config.json \
        --trained-checkpoint outputs/mtkd_train_epoch100_version0.1/best_model.pth \
        --num-samples 8
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
from PIL import Image

# Prefer the repository's bundled Ultralytics fork over site-packages.
REPO_ROOT = Path(__file__).resolve().parents[1]
ULTRALYTICS_ROOT = REPO_ROOT / "ultralytics"
for candidate in (REPO_ROOT, ULTRALYTICS_ROOT):
    if candidate.is_dir():
        candidate_str = str(candidate)
        if candidate_str not in sys.path:
            sys.path.insert(0, candidate_str)

from .models.mtkd_model_v2 import build_mtkd_model_v2
from .utils import load_checkpoint, seed_everything
from .visualize_alignment import (
    _build_loader,
    _build_merged_config,
    _feature_to_tokens,
    _fit_joint_pca,
    _fit_pca,
    _make_tile,
    _resolve_checkpoint,
    _save_metrics_csv,
    _teacher_view,
    _tensor_image_to_pil,
    _tokens_to_pca_image,
)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Compare pre/post student PCA against frozen DINO PCA.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--config", type=str, required=True, help="Training config JSON/YAML path")
    p.add_argument(
        "--trained-checkpoint",
        type=str,
        default=None,
        help="Trained MTKD checkpoint (.pth). Defaults to best_model.pth resolved from config.",
    )
    p.add_argument(
        "--baseline-student-weights",
        type=str,
        default=None,
        help="Initial student .pt weights. Defaults to model.student_config.weights from config.",
    )
    p.add_argument("--split", type=str, default="val", choices=["train", "val"])
    p.add_argument("--num-samples", type=int, default=8)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--output-dir", type=str, default=None)
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--tile-size", type=int, default=320)
    return p.parse_args()


def _build_debug_model_config(config: Dict, baseline_weights: Optional[str] = None) -> Dict:
    model_cfg = copy.deepcopy(config["model"])
    model_cfg["wheat_teacher_config"] = None
    if baseline_weights:
        model_cfg.setdefault("student_config", {})["weights"] = baseline_weights
    return model_cfg


def _build_panel(
    image: Image.Image,
    before_pca: Image.Image,
    after_pca: Image.Image,
    dino_pca: Image.Image,
    tile_size: int,
) -> Image.Image:
    tiles = [
        _make_tile(image, "Image", tile_size),
        _make_tile(before_pca, "Before Student PCA", tile_size),
        _make_tile(after_pca, "After Student PCA", tile_size),
        _make_tile(dino_pca, "DINO PCA", tile_size),
    ]
    gap = 8
    width = sum(tile.width for tile in tiles) + gap * (len(tiles) - 1)
    height = max(tile.height for tile in tiles)
    panel = Image.new("RGB", (width, height), color=(245, 245, 245))
    x = 0
    for tile in tiles:
        panel.paste(tile, (x, 0))
        x += tile.width + gap
    return panel


def main() -> None:
    args = _parse_args()
    config_path = Path(args.config).resolve()
    if not config_path.is_file():
        raise FileNotFoundError(f"Config not found: {config_path}")

    config = _build_merged_config(config_path)
    seed_everything(int(config.get("seed", 42)))

    device_str = args.device or config.get("device") or ("cuda" if torch.cuda.is_available() else "cpu")
    if str(device_str).startswith("cuda") and not torch.cuda.is_available():
        device_str = "cpu"
    device = torch.device(device_str)
    config["device"] = str(device)

    trained_ckpt = _resolve_checkpoint(config_path, config, args.trained_checkpoint)
    baseline_weights = (
        args.baseline_student_weights
        or config.get("model", {}).get("student_config", {}).get("weights")
    )
    if not baseline_weights:
        raise ValueError("Could not resolve baseline student weights from config")

    output_dir = (
        Path(args.output_dir).resolve()
        if args.output_dir
        else trained_ckpt.parent / "feature_alignment_compare"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    loader = _build_loader(config, split=args.split, batch_size=args.batch_size, num_workers=args.num_workers)
    try:
        warmup_batch = next(iter(loader))
    except StopIteration as exc:
        raise RuntimeError(f"No samples available in {args.split} split") from exc

    align_easy_only = bool(config.get("training", {}).get("align_easy_only", False))
    warmup_images = warmup_batch["images"][:1].to(device)
    warmup_teacher = _teacher_view(warmup_batch, align_easy_only)
    if warmup_teacher is not None:
        warmup_teacher = warmup_teacher[:1].to(device)

    baseline_model = build_mtkd_model_v2(_build_debug_model_config(config, baseline_weights)).to(device)
    _ = baseline_model.get_alignment_debug(warmup_images, teacher_images=warmup_teacher)
    baseline_model.eval()

    trained_model = build_mtkd_model_v2(_build_debug_model_config(config, baseline_weights)).to(device)
    _ = trained_model.get_alignment_debug(warmup_images, teacher_images=warmup_teacher)
    load_checkpoint(trained_model, str(trained_ckpt), strict=False, map_location=str(device))
    trained_model.eval()

    rows: List[Dict[str, object]] = []
    sample_count = 0

    for batch in loader:
        images = batch["images"].to(device)
        teacher_images = _teacher_view(batch, align_easy_only)
        if teacher_images is not None:
            teacher_images = teacher_images.to(device)

        before_debug = baseline_model.get_alignment_debug(images, teacher_images=teacher_images)
        after_debug = trained_model.get_alignment_debug(images, teacher_images=teacher_images)

        before_student = before_debug["student_spatial_feat"]
        after_student = after_debug["student_spatial_feat"]
        dino_feat = after_debug["dino_features"]

        batch_paths = batch.get("image_paths", [])
        for i in range(images.shape[0]):
            if sample_count >= args.num_samples:
                break

            image_path = str(batch_paths[i]) if i < len(batch_paths) else f"sample_{sample_count:04d}"
            image_stem = Path(image_path).stem

            before_tokens, before_h, before_w = _feature_to_tokens(before_student[i])
            after_tokens, after_h, after_w = _feature_to_tokens(after_student[i])
            if (before_h, before_w) != (after_h, after_w):
                raise RuntimeError(
                    f"Student feature resolution mismatch: {(before_h, before_w)} vs {(after_h, after_w)}"
                )
            before_pca, after_pca = _fit_joint_pca(before_tokens, after_tokens)
            before_pca_img = _tokens_to_pca_image(before_pca, before_h, before_w)
            after_pca_img = _tokens_to_pca_image(after_pca, after_h, after_w)

            dino_tokens, dino_h, dino_w = _feature_to_tokens(dino_feat[i])
            dino_pca = _fit_pca(dino_tokens)
            dino_pca_img = _tokens_to_pca_image(dino_pca, dino_h, dino_w)

            input_pil = _tensor_image_to_pil(images[i].detach().cpu())
            panel = _build_panel(
                image=input_pil,
                before_pca=before_pca_img,
                after_pca=after_pca_img,
                dino_pca=dino_pca_img,
                tile_size=args.tile_size,
            )

            panel_path = output_dir / f"{sample_count:04d}_{image_stem}.png"
            panel.save(panel_path)

            before_np = before_student[i].detach().cpu().float().numpy()
            after_np = after_student[i].detach().cpu().float().numpy()
            before_flat = before_student[i].detach().cpu().float().reshape(before_student.shape[1], -1).T
            after_flat = after_student[i].detach().cpu().float().reshape(after_student.shape[1], -1).T
            before_flat = torch.nn.functional.normalize(before_flat, dim=1)
            after_flat = torch.nn.functional.normalize(after_flat, dim=1)
            raw_cos = (before_flat * after_flat).sum(dim=1).cpu().numpy()

            row = {
                "index": sample_count,
                "image_path": image_path,
                "panel_path": str(panel_path),
                "before_align_loss": float(before_debug["align_loss"].item()),
                "after_align_loss": float(after_debug["align_loss"].item()),
                "before_student_mean": float(before_np.mean()),
                "after_student_mean": float(after_np.mean()),
                "raw_before_after_cosine_mean": float(raw_cos.mean()),
                "raw_before_after_cosine_median": float(np.median(raw_cos)),
            }
            rows.append(row)
            sample_count += 1

        if sample_count >= args.num_samples:
            break

    if not rows:
        raise RuntimeError("No comparison samples were generated")

    _save_metrics_csv(output_dir / "metrics.csv", rows)
    summary = {
        "config": str(config_path),
        "trained_checkpoint": str(trained_ckpt),
        "baseline_student_weights": str(Path(baseline_weights).resolve()),
        "split": args.split,
        "num_samples": len(rows),
        "mean_before_align_loss": float(np.mean([float(r["before_align_loss"]) for r in rows])),
        "mean_after_align_loss": float(np.mean([float(r["after_align_loss"]) for r in rows])),
        "mean_raw_before_after_cosine": float(np.mean([float(r["raw_before_after_cosine_mean"]) for r in rows])),
        "output_dir": str(output_dir),
    }
    with (output_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
