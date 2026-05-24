#!/usr/bin/env python3
"""
Minimal prediction-first visualization.

Outputs per image:

1. Feature panel
   - original image
   - YOLO feature map before pattern
   - YOLO feature map after pattern

2. Prediction panel
   - before-alignment YOLO prediction
   - after-alignment YOLO prediction
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Iterator, List, Optional

import numpy as np
import torch
from PIL import Image


REPO_ROOT = Path(__file__).resolve().parents[1]
ULTRALYTICS_ROOT = REPO_ROOT / "ultralytics"
for candidate in (REPO_ROOT, ULTRALYTICS_ROOT):
    if candidate.is_dir():
        candidate_str = str(candidate)
        if candidate_str not in sys.path:
            sys.path.insert(0, candidate_str)

from ultralytics import YOLO

from .visualize_alignment import (
    _build_loader,
    _build_merged_config,
    _make_tile,
    _save_metrics_csv,
    _tensor_image_to_pil,
)
from .visualize_dino_effect import (
    _draw_predictions,
    _feature_level_label,
    _feature_response_image,
    _make_grid,
    _predict_one,
)
from .visualize_yolo_prediction_panels_core import (
    _build_dino,
    _build_pca_guide_map_from_image,
    _build_yolo_student,
    _extract_yolo_pyramid,
    _iter_image_dir_batches,
    _iter_image_path_batches,
    _load_image_prior_index,
    _load_species_prior_bank,
    _parse_feature_levels,
    _pattern_enhance_feature,
    _prediction_center_points,
    _resolve_external_pattern_prior,
    _resolve_weight_path,
    _save_json,
    _transform_anchor_points,
)


PATTERN_DEFAULTS: Dict[str, float] = {
    "mask_target_coverage": 0.12,
    "mask_min_coverage": 0.02,
    "mask_max_coverage": 0.30,
    "mask_fuse_bg": 0.5,
    "mask_fuse_fg": 1.8,
    "seed_threshold": 0.01,
    "seed_topk": 96,
    "gate_threshold": 0.05,
    "min_row_seeds": 2.0,
    "line_max_slope": 0.18,
    "period_prior_ratio": 1.8,
    "period_scale": 1.0,
    "row_sigma_scale": 0.45,
    "period_sigma_scale": 0.28,
    "prior_strength": 0.72,
    "completion_strength": 0.45,
    "completion_gamma": 1.0,
    "noise_suppress": 0.18,
    "cross_row_strength": 0.24,
    "cross_row_noise_suppress": 0.10,
    "response_period_blend": 0.60,
    "pca_guide_strength": 0.55,
    "pca_guide_image_size": 518.0,
    "pca_components": 6.0,
}


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Minimal visualization for YOLO feature maps and before/after alignment predictions.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--config", type=str, required=True, help="Training config JSON/YAML path")
    p.add_argument("--after-weights", type=str, required=True, help="Aligned/exported YOLO .pt")
    p.add_argument("--before-weights", type=str, default=None, help="Baseline YOLO .pt")
    p.add_argument("--image", type=str, default=None, help="Optional single image path")
    p.add_argument("--image-dir", type=str, default=None, help="Optional image directory")
    p.add_argument("--split", type=str, default="val", choices=["train", "val"])
    p.add_argument("--num-samples", type=int, default=8)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--imgsz", type=int, default=None)
    p.add_argument("--feature-level", type=str, default=None, choices=["p3", "p4", "p5"])
    p.add_argument(
        "--feature-source",
        type=str,
        default="after",
        choices=["before", "after"],
        help="Which YOLO checkpoint provides the feature-map panel's before/after pattern feature.",
    )
    p.add_argument("--conf", type=float, default=0.25)
    p.add_argument("--iou", type=float, default=0.7)
    p.add_argument("--max-det", type=int, default=300)
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--output-dir", type=str, default=None)
    p.add_argument("--tile-size", type=int, default=420)
    p.add_argument("--annotation-scale", type=float, default=1.0)
    p.add_argument("--box-scale", type=float, default=0.45)
    p.add_argument("--pattern-enhance", action="store_true", help="Apply pattern enhancement to the feature panel")
    p.add_argument("--pca-guide", action="store_true", help="Use best PCA component as soft guide")
    p.add_argument(
        "--pca-guide-score-dir",
        type=str,
        default=None,
        help="Directory containing *_component_scores.json from dino_component_explorer",
    )
    p.add_argument(
        "--pattern-prior-levels",
        type=str,
        default="p3,p4",
        help="Comma-separated YOLO pyramid levels used to build the pattern prior",
    )
    p.add_argument(
        "--pattern-image-priors",
        type=str,
        default=str(REPO_ROOT / "outputs" / "rice_info" / "rice_annotate_spacing_prior_v2" / "image_priors.json"),
        help="Optional exact-image row/spacing prior JSON",
    )
    p.add_argument(
        "--pattern-species-prior-bank",
        type=str,
        default=str(REPO_ROOT / "outputs" / "spacing_info" / "species_spacing_prior_bank.json"),
        help="Optional species-level spacing prior bank JSON",
    )
    return p.parse_args()


def _iter_batches(args: argparse.Namespace, config: Dict, image_size: int) -> Iterator[Dict[str, object]]:
    if args.image and args.image_dir:
        raise ValueError("Use either --image or --image-dir, not both")
    if args.image:
        image_path = Path(args.image).resolve()
        if not image_path.is_file():
            raise FileNotFoundError(f"Image not found: {image_path}")
        return _iter_image_path_batches([image_path], image_size, args.batch_size)
    if args.image_dir:
        return _iter_image_dir_batches(Path(args.image_dir), image_size, args.batch_size)
    return _build_loader(config, split=args.split, batch_size=args.batch_size, num_workers=args.num_workers)


def _effective_pattern_settings(
    *,
    args: argparse.Namespace,
    original_size: tuple[int, int],
    image_path: str,
    image_prior_index: Dict[str, Dict[str, object]],
    species_prior_bank: Dict[str, Dict[str, float]],
    image_size: int,
) -> Dict[str, object]:
    external = _resolve_external_pattern_prior(
        image_path,
        image_size=original_size,
        image_prior_index=image_prior_index,
        species_prior_bank=species_prior_bank,
    )

    external_row_centers: List[float] = []
    raw_row_centers = external.get("row_centers_px", [])
    if isinstance(raw_row_centers, (list, tuple)) and raw_row_centers:
        row_pts = np.asarray([[0.0, float(y), 1.0] for y in raw_row_centers], dtype=np.float32)
        transformed = _transform_anchor_points(
            row_pts,
            src_size=original_size,
            dst_size=(image_size, image_size),
            keep_aspect=True,
        )
        external_row_centers = transformed[:, 1].astype(np.float32).tolist()

    row_tolerance_px = float(external.get("row_tolerance_px", 0.0) or 0.0)
    if row_tolerance_px <= 0.0:
        row_tolerance_px = 42.0

    return {
        "external": external,
        "row_centers": external_row_centers,
        "period_prior_px": float(external.get("x_period_px", 0.0) or 0.0),
        "row_period_prior_px": float(external.get("row_period_px", 0.0) or 0.0),
        "row_tolerance_px": row_tolerance_px,
    }


def main() -> None:
    args = _parse_args()
    config_path = Path(args.config).resolve()
    if not config_path.is_file():
        raise FileNotFoundError(f"Config not found: {config_path}")

    config = _build_merged_config(config_path)
    device_str = args.device or config.get("device") or ("cuda" if torch.cuda.is_available() else "cpu")
    if str(device_str).startswith("cuda") and not torch.cuda.is_available():
        device_str = "cpu"
    device = torch.device(device_str)

    image_size = int(args.imgsz or config.get("data", {}).get("image_size", 640))
    feature_level = (
        args.feature_level
        or config.get("model", {}).get("student_align_layer")
        or config.get("model", {}).get("student_config", {}).get("feature_level")
        or "p4"
    )
    feature_level = str(feature_level).lower()
    pattern_prior_levels = _parse_feature_levels(args.pattern_prior_levels, fallback=feature_level)

    config_before = config.get("model", {}).get("student_config", {}).get("weights")
    before_weights = _resolve_weight_path(args.before_weights, config_before, root=REPO_ROOT)
    after_weights = _resolve_weight_path(args.after_weights, None, root=REPO_ROOT)

    output_dir = Path(args.output_dir).resolve() if args.output_dir else REPO_ROOT / "outputs" / "yolo_prediction_panels_minimal"
    output_dir.mkdir(parents=True, exist_ok=True)

    pca_score_dir = Path(args.pca_guide_score_dir).resolve() if args.pca_guide_score_dir else None
    image_prior_index = _load_image_prior_index(Path(args.pattern_image_priors).resolve()) if args.pattern_image_priors else {}
    species_prior_bank = _load_species_prior_bank(Path(args.pattern_species_prior_bank).resolve()) if args.pattern_species_prior_bank else {}

    print(f"[yolo-pred-panels] output_dir={output_dir}", flush=True)
    print(f"[yolo-pred-panels] device={device} imgsz={image_size} feature_level={feature_level}", flush=True)
    print(f"[yolo-pred-panels] before={before_weights}", flush=True)
    print(f"[yolo-pred-panels] after={after_weights}", flush=True)

    before_student = _build_yolo_student(before_weights, config, feature_level, device)
    after_student = _build_yolo_student(after_weights, config, feature_level, device)
    before_yolo = YOLO(str(before_weights))
    after_yolo = YOLO(str(after_weights))
    dino = _build_dino(config, device)
    before_detect = before_student.detect_module
    after_detect = after_student.detect_module

    rows: List[Dict[str, object]] = []
    sample_count = 0

    for batch in _iter_batches(args, config, image_size):
        images = batch["images"].to(device)
        image_paths = [str(p) for p in batch.get("image_paths", [])]
        if not image_paths:
            image_paths = [f"sample_{sample_count + i:04d}" for i in range(images.shape[0])]

        with torch.no_grad():
            before_pyramid = _extract_yolo_pyramid(before_student, images)
            after_pyramid = _extract_yolo_pyramid(after_student, images)
            dino_batch = dino(images).detach()

        for batch_idx in range(images.shape[0]):
            if sample_count >= args.num_samples:
                break

            image_path = image_paths[batch_idx]
            stem = Path(image_path).stem
            original_pil = (
                Image.open(image_path).convert("RGB")
                if Path(image_path).is_file()
                else _tensor_image_to_pil(images[batch_idx].detach().cpu())
            )

            before_preds = (
                _predict_one(before_yolo, image_path, imgsz=image_size, conf=args.conf, iou=args.iou, max_det=args.max_det, device=device)
                if Path(image_path).is_file()
                else []
            )
            after_preds = (
                _predict_one(after_yolo, image_path, imgsz=image_size, conf=args.conf, iou=args.iou, max_det=args.max_det, device=device)
                if Path(image_path).is_file()
                else []
            )

            feature_source_pyramid = after_pyramid if args.feature_source == "after" else before_pyramid
            feature_source_feat = feature_source_pyramid[feature_level][batch_idx]
            feature_source_detect = after_detect if args.feature_source == "after" else before_detect

            pattern_cfg = _effective_pattern_settings(
                args=args,
                original_size=original_pil.size,
                image_path=image_path,
                image_prior_index=image_prior_index,
                species_prior_bank=species_prior_bank,
                image_size=image_size,
            )

            guide_override: Optional[torch.Tensor] = None
            guide_stats: Dict[str, float] = {}
            if args.pca_guide and Path(image_path).is_file():
                guide_override, guide_stats, _overlay = _build_pca_guide_map_from_image(
                    image_path,
                    dino=dino,
                    device=device,
                    image_size=int(PATTERN_DEFAULTS["pca_guide_image_size"]),
                    n_components=int(PATTERN_DEFAULTS["pca_components"]),
                    target_size=tuple(int(v) for v in feature_source_feat.shape[-2:]),
                    score_dir=pca_score_dir,
                )

            anchor_points = _transform_anchor_points(
                _prediction_center_points(after_preds if args.feature_source == "after" else before_preds),
                src_size=original_pil.size,
                dst_size=(image_size, image_size),
                keep_aspect=False,
            )

            if args.pattern_enhance or args.pca_guide:
                enhanced_feat, _prior_map, _support_map, _guide_map, pattern_stats = _pattern_enhance_feature(
                    {level: feat[batch_idx] for level, feat in feature_source_pyramid.items()},
                    dino_batch[batch_idx],
                    detect_module=feature_source_detect,
                    target_level=feature_level,
                    prior_levels=pattern_prior_levels,
                    image_size=(image_size, image_size),
                    anchor_points=anchor_points,
                    mask_target_coverage=PATTERN_DEFAULTS["mask_target_coverage"],
                    mask_min_coverage=PATTERN_DEFAULTS["mask_min_coverage"],
                    mask_max_coverage=PATTERN_DEFAULTS["mask_max_coverage"],
                    mask_fuse_bg=PATTERN_DEFAULTS["mask_fuse_bg"],
                    mask_fuse_fg=PATTERN_DEFAULTS["mask_fuse_fg"],
                    pattern_seed_threshold=PATTERN_DEFAULTS["seed_threshold"],
                    pattern_seed_topk=int(PATTERN_DEFAULTS["seed_topk"]),
                    pattern_gate_threshold=PATTERN_DEFAULTS["gate_threshold"],
                    pattern_row_tolerance_px=float(pattern_cfg["row_tolerance_px"]),
                    pattern_min_row_seeds=int(PATTERN_DEFAULTS["min_row_seeds"]),
                    pattern_line_max_slope=PATTERN_DEFAULTS["line_max_slope"],
                    pattern_period_prior_px=float(pattern_cfg["period_prior_px"]),
                    pattern_period_prior_ratio=PATTERN_DEFAULTS["period_prior_ratio"],
                    pattern_period_scale=PATTERN_DEFAULTS["period_scale"],
                    pattern_row_sigma_scale=PATTERN_DEFAULTS["row_sigma_scale"],
                    pattern_period_sigma_scale=PATTERN_DEFAULTS["period_sigma_scale"],
                    pattern_row_period_prior_px=float(pattern_cfg["row_period_prior_px"]),
                    pattern_row_center_priors_px=pattern_cfg["row_centers"],
                    pattern_prior_strength=PATTERN_DEFAULTS["prior_strength"] if args.pattern_enhance else 0.0,
                    pattern_completion_strength=PATTERN_DEFAULTS["completion_strength"] if args.pattern_enhance else 0.0,
                    pattern_completion_gamma=PATTERN_DEFAULTS["completion_gamma"],
                    pattern_noise_suppress=PATTERN_DEFAULTS["noise_suppress"] if args.pattern_enhance else 0.0,
                    pattern_cross_row_strength=PATTERN_DEFAULTS["cross_row_strength"] if args.pattern_enhance else 0.0,
                    pattern_cross_row_noise_suppress=PATTERN_DEFAULTS["cross_row_noise_suppress"] if args.pattern_enhance else 0.0,
                    response_period_blend=PATTERN_DEFAULTS["response_period_blend"],
                    nmf_enabled=bool(args.pca_guide),
                    nmf_components=int(PATTERN_DEFAULTS["pca_components"]),
                    nmf_topk=1,
                    nmf_strength=PATTERN_DEFAULTS["pca_guide_strength"] if args.pca_guide else 0.0,
                    nmf_init="nndsvda",
                    nmf_max_iter=1200,
                    nmf_override_map=guide_override,
                )
            else:
                enhanced_feat = feature_source_feat
                pattern_stats = {}

            pattern_stats.update(guide_stats)

            before_feature_img = _feature_response_image(feature_source_feat, original_pil.size)
            after_feature_img = _feature_response_image(enhanced_feat, original_pil.size)
            before_overlay = _draw_predictions(
                original_pil,
                before_preds,
                (0, 245, 255),
                label_prefix="B:",
                max_labels=0,
                annotation_scale=args.annotation_scale,
                line_width_scale=args.box_scale,
                shadow=False,
            )
            after_overlay = _draw_predictions(
                original_pil,
                after_preds,
                (255, 35, 210),
                label_prefix="A:",
                max_labels=0,
                annotation_scale=args.annotation_scale,
                line_width_scale=args.box_scale,
                shadow=False,
            )

            feature_panel = _make_grid(
                [
                    _make_tile(original_pil, "Origin Image", args.tile_size),
                    _make_tile(
                        before_feature_img,
                        f"{args.feature_source.title()} YOLO {_feature_level_label(feature_level)} Before Pattern",
                        args.tile_size,
                    ),
                    _make_tile(
                        after_feature_img,
                        f"{args.feature_source.title()} YOLO {_feature_level_label(feature_level)} After Pattern",
                        args.tile_size,
                    ),
                ],
                cols=3,
            )
            feature_panel_path = output_dir / f"{sample_count:04d}_{stem}_feature_panel.png"
            feature_panel.save(feature_panel_path)

            prediction_panel = _make_grid(
                [
                    _make_tile(before_overlay, "Before Alignment Prediction", args.tile_size),
                    _make_tile(after_overlay, "After Alignment Prediction", args.tile_size),
                ],
                cols=2,
            )
            prediction_panel_path = output_dir / f"{sample_count:04d}_{stem}_prediction_panel.png"
            prediction_panel.save(prediction_panel_path)

            row: Dict[str, object] = {
                "index": sample_count,
                "image_path": image_path,
                "feature_panel_path": str(feature_panel_path),
                "prediction_panel_path": str(prediction_panel_path),
                "before_weights": str(before_weights),
                "after_weights": str(after_weights),
                "feature_source": args.feature_source,
                "feature_level": feature_level,
                "before_box_count": float(len(before_preds)),
                "after_box_count": float(len(after_preds)),
                "pattern_enabled": float(1.0 if args.pattern_enhance else 0.0),
                "pca_enabled": float(1.0 if args.pca_guide else 0.0),
                "pattern_prior_source": str(pattern_cfg["external"].get("source", "none")),
                "pattern_prior_family": str(pattern_cfg["external"].get("family", "")),
            }
            row.update({k: float(v) for k, v in pattern_stats.items()})
            rows.append(row)

            print(
                f"[yolo-pred-panels] saved feature={feature_panel_path.name} pred={prediction_panel_path.name} "
                f"before={len(before_preds)} after={len(after_preds)}",
                flush=True,
            )
            sample_count += 1

        if sample_count >= args.num_samples:
            break

    if not rows:
        raise RuntimeError("No samples were generated")

    _save_metrics_csv(output_dir / "metrics.csv", rows)
    summary = {
        "config": str(config_path),
        "before_weights": str(before_weights),
        "after_weights": str(after_weights),
        "num_samples": len(rows),
        "feature_source": args.feature_source,
        "feature_level": feature_level,
        "mean_before_box_count": float(np.mean([float(r["before_box_count"]) for r in rows])),
        "mean_after_box_count": float(np.mean([float(r["after_box_count"]) for r in rows])),
        "output_dir": str(output_dir),
    }
    _save_json(output_dir / "summary.json", summary)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
