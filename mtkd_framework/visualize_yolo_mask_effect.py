#!/usr/bin/env python3
"""
Visualize how DINO-mask-gated YOLO features change final detection outputs.

This is a standalone experiment helper. It does not touch MTKD training code.

Compared with ``visualize_dino_effect.py``, this script focuses on a single
YOLO checkpoint:

1. Run raw YOLO prediction.
2. Build a DINO-derived foreground mask.
3. Inject the mask back into one YOLO feature level.
4. Run prediction again.
5. Save a compact panel and CSV metrics for raw vs mask-fused predictions.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Sequence, Tuple

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

from .engine.build_dino import DinoFeatureExtractor
from .models.yolo_wrappers import YOLOStudentDetector
from .visualize_alignment import (
    _build_loader,
    _build_merged_config,
    _make_tile,
    _save_metrics_csv,
    _tensor_image_to_pil,
)
from .visualize_dino_effect import (
    IMAGE_EXTS,
    _apply_dino_mask_fuse_to_feature,
    _candidates_from_pseudo_boxes,
    _dino_mask_images,
    _draw_prediction_difference,
    _draw_predictions,
    _draw_spacing_points,
    _estimate_spacing_phase,
    _estimate_spacing_px,
    _feature_level_label,
    _feature_response_image,
    _filter_pseudo_boxes_by_reference_overlap,
    _find_detect_module,
    _head_cls_score_map,
    _make_grid,
    _poly_center,
    _predictions_to_center_points,
    _predict_one,
    _predict_one_dino_gated,
    _prediction_delta_metrics,
    _refine_spacing_candidate,
    _sample_grid_value,
)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Run raw YOLO and DINO-mask-fused YOLO on the same images.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--config", type=str, required=True, help="Training config JSON/YAML path")
    p.add_argument(
        "--weights",
        type=str,
        default=None,
        help="YOLO checkpoint to test. Defaults to config model.student_config.weights",
    )
    p.add_argument("--image", type=str, default=None, help="Optional single image path")
    p.add_argument("--image-dir", type=str, default=None, help="Optional image directory")
    p.add_argument("--split", type=str, default="val", choices=["train", "val"])
    p.add_argument("--num-samples", type=int, default=8)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--imgsz", type=int, default=None)
    p.add_argument("--feature-level", type=str, default=None, choices=["p3", "p4", "p5"])
    p.add_argument("--mask-level", type=str, default=None, choices=["p3", "p4", "p5"])
    p.add_argument("--conf", type=float, default=0.25)
    p.add_argument("--iou", type=float, default=0.7)
    p.add_argument("--max-det", type=int, default=300)
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--output-dir", type=str, default=None)
    p.add_argument("--tile-size", type=int, default=520)
    p.add_argument("--single-panel-cols", type=int, default=4)
    p.add_argument("--annotation-scale", type=float, default=1.0)
    p.add_argument("--box-scale", type=float, default=0.45)
    p.add_argument("--mask-target-coverage", type=float, default=0.12)
    p.add_argument("--mask-min-coverage", type=float, default=0.02)
    p.add_argument("--mask-max-coverage", type=float, default=0.30)
    p.add_argument("--mask-blur-radius", type=float, default=2.0)
    p.add_argument("--mask-fuse-bg", type=float, default=0.5)
    p.add_argument("--mask-fuse-fg", type=float, default=1.8)
    p.add_argument("--pattern-filter", action="store_true",
                   help="Apply row-aware prediction denoising without adding completion boxes")
    p.add_argument("--spacing-complete", action="store_true",
                   help="Apply row/spacing-limited completion on top of mask-fused predictions")
    p.add_argument("--spacing-anchor-source", type=str, default="both",
                   choices=["raw", "mask-fused", "both"],
                   help="Which prediction set provides row anchors")
    p.add_argument("--spacing-score-threshold", type=float, default=0.01,
                   help="Minimum cls score for row-seed candidates")
    p.add_argument("--spacing-proposal-score-threshold", type=float, default=0.005,
                   help="Minimum cls score for proposed completion candidates")
    p.add_argument("--spacing-dino-threshold", type=float, default=0.18,
                   help="Minimum DINO gate value for spacing candidates")
    p.add_argument("--spacing-row-tolerance-px", type=float, default=42.0,
                   help="Y tolerance when grouping anchors into rows")
    p.add_argument("--spacing-nms-distance-px", type=float, default=24.0,
                   help="Minimum center distance between proposals/anchors")
    p.add_argument("--spacing-min-row-seeds", type=int, default=2,
                   help="Minimum anchor count in a row before completion is allowed")
    p.add_argument("--spacing-max-candidates", type=int, default=80)
    p.add_argument("--spacing-period-prior-px", type=float, default=0.0,
                   help="Optional known horizontal stomata period in pixels")
    p.add_argument("--spacing-period-prior-ratio", type=float, default=1.8)
    p.add_argument("--spacing-period-scale", type=float, default=1.0)
    p.add_argument("--spacing-refine-window-px", type=float, default=20.0,
                   help="Local search radius around each spacing proposal")
    p.add_argument("--spacing-refine-step-px", type=float, default=4.0)
    p.add_argument("--spacing-refine-prior-sigma-px", type=float, default=12.0)
    p.add_argument("--spacing-pseudo-min-anchor-per-row", type=int, default=2,
                   help="Minimum anchors on a row for mask-fused boxes to be kept")
    p.add_argument("--spacing-pseudo-overlap-iou", type=float, default=0.10,
                   help="Drop completion boxes if they overlap existing kept boxes")
    p.add_argument("--spacing-pseudo-row-frequency-ratio", type=float, default=1.75,
                   help="Drop rows whose horizontal spacing differs too much from other rows")
    p.add_argument("--spacing-pseudo-row-frequency-min-points", type=int, default=4)
    p.add_argument("--spacing-line-max-slope", type=float, default=0.18,
                   help="Max absolute dy/dx slope allowed for a stomata row fit")
    p.add_argument("--spacing-line-residual-px", type=float, default=18.0,
                   help="Max residual to fitted slanted row before a box/candidate is rejected")
    p.add_argument("--shape-filter", action="store_true",
                   help="Filter unusually long/large/small boxes using the image-level stomata size prior")
    p.add_argument("--shape-width-min-scale", type=float, default=0.45)
    p.add_argument("--shape-width-max-scale", type=float, default=2.20)
    p.add_argument("--shape-height-min-scale", type=float, default=0.45)
    p.add_argument("--shape-height-max-scale", type=float, default=2.20)
    p.add_argument("--shape-area-min-scale", type=float, default=0.25)
    p.add_argument("--shape-area-max-scale", type=float, default=3.20)
    p.add_argument("--shape-aspect-max-scale", type=float, default=2.10,
                   help="Allow aspect ratio within median/aspect_max_scale to median*aspect_max_scale")
    p.add_argument("--edge-keep-margin-px", type=float, default=24.0,
                   help="Always keep boxes that touch the image border within this many pixels")
    p.add_argument("--edge-keep-margin-ratio", type=float, default=0.05,
                   help="Always keep boxes that touch the image border within this fraction of min(image size)")
    return p.parse_args()


def _resolve_weights(config: Dict, explicit: Optional[str]) -> str:
    weights = explicit or config.get("model", {}).get("student_config", {}).get("weights")
    if not weights:
        raise ValueError("Could not resolve YOLO weights from --weights or config")
    weight_path = Path(weights)
    if not weight_path.is_file():
        raise FileNotFoundError(f"YOLO weights not found: {weight_path}")
    return str(weight_path.resolve())


def _list_images(image_dir: Path) -> List[Path]:
    return sorted(p for p in image_dir.rglob("*") if p.suffix.lower() in IMAGE_EXTS)


def _image_to_tensor(path: Path, image_size: int) -> torch.Tensor:
    image = Image.open(path).convert("RGB")
    image = image.resize((image_size, image_size), resample=Image.Resampling.BILINEAR)
    arr = np.asarray(image, dtype=np.float32) / 255.0
    return torch.from_numpy(arr).permute(2, 0, 1).contiguous()


def _iter_image_path_batches(
    paths: Sequence[Path],
    image_size: int,
    batch_size: int,
) -> Iterator[Dict[str, object]]:
    if not paths:
        raise FileNotFoundError("No images were provided")
    for start in range(0, len(paths), batch_size):
        chunk = paths[start : start + batch_size]
        images = torch.stack([_image_to_tensor(path, image_size) for path in chunk], dim=0)
        yield {
            "images": images,
            "image_paths": [str(path) for path in chunk],
        }


def _iter_image_dir_batches(
    image_dir: Path,
    image_size: int,
    batch_size: int,
) -> Iterator[Dict[str, object]]:
    paths = _list_images(image_dir)
    if not paths:
        raise FileNotFoundError(f"No images found under: {image_dir}")
    yield from _iter_image_path_batches(paths, image_size, batch_size)


def _build_yolo_student(
    weights: str,
    config: Dict,
    feature_level: str,
    device: torch.device,
) -> YOLOStudentDetector:
    num_classes = int(config.get("model", {}).get("num_classes", 1))
    student = YOLOStudentDetector(
        weights=weights,
        feature_level=feature_level,
        num_classes=num_classes,
    ).to(device)
    student.eval()
    return student


def _build_dino(config: Dict, device: torch.device) -> DinoFeatureExtractor:
    dino_cfg = dict(config.get("model", {}).get("dino_config", {}) or {})
    dino_ckpt = config.get("model", {}).get("dino_checkpoint")
    if dino_ckpt:
        dino_cfg["pretrained_path"] = dino_ckpt
    dino = DinoFeatureExtractor(**dino_cfg).to(device)
    dino.eval()
    return dino


@torch.no_grad()
def _extract_yolo_feature(
    student: YOLOStudentDetector,
    images: torch.Tensor,
    feature_level: str,
) -> torch.Tensor:
    out = student(images, return_features=True, return_adapted_features=False)
    key = f"{feature_level}_features"
    feat = out.get(key)
    if feat is None:
        raise RuntimeError(f"YOLO output did not contain {key}")
    return feat.detach()


def _save_json(path: Path, data: Dict[str, object]) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def _prediction_shape(pred: Dict[str, object]) -> Dict[str, float]:
    poly = np.asarray(pred["poly"], dtype=np.float32).reshape(4, 2)
    edges = []
    for idx in range(4):
        nxt = (idx + 1) % 4
        edges.append(float(np.linalg.norm(poly[nxt] - poly[idx])))
    side_a = 0.5 * (edges[0] + edges[2])
    side_b = 0.5 * (edges[1] + edges[3])
    width = max(side_a, side_b, 1e-6)
    height = max(min(side_a, side_b), 1e-6)
    area = width * height
    aspect = width / max(height, 1e-6)
    return {
        "width": float(width),
        "height": float(height),
        "area": float(area),
        "aspect": float(aspect),
    }


def _prediction_aabb(pred: Dict[str, object]) -> np.ndarray:
    poly = np.asarray(pred["poly"], dtype=np.float32).reshape(4, 2)
    return np.asarray(
        [poly[:, 0].min(), poly[:, 1].min(), poly[:, 0].max(), poly[:, 1].max()],
        dtype=np.float32,
    )


def _is_edge_prediction(
    pred: Dict[str, object],
    image_size: Tuple[int, int],
    *,
    margin_px: float,
    margin_ratio: float,
) -> bool:
    img_w, img_h = image_size
    border = max(float(margin_px), float(min(img_w, img_h)) * float(margin_ratio))
    x1, y1, x2, y2 = _prediction_aabb(pred).tolist()
    return bool(x1 <= border or y1 <= border or x2 >= float(img_w) - border or y2 >= float(img_h) - border)


def _shape_reference_stats(predictions: Sequence[Dict[str, object]], image_size: Tuple[int, int]) -> Dict[str, float]:
    widths: List[float] = []
    heights: List[float] = []
    areas: List[float] = []
    aspects: List[float] = []
    for pred in predictions:
        shape = _prediction_shape(pred)
        widths.append(shape["width"])
        heights.append(shape["height"])
        areas.append(shape["area"])
        aspects.append(shape["aspect"])
    if widths:
        return {
            "width": float(np.median(widths)),
            "height": float(np.median(heights)),
            "area": float(np.median(areas)),
            "aspect": float(np.median(aspects)),
        }
    img_w, img_h = image_size
    width = max(8.0, float(img_w) * 0.07)
    height = max(6.0, float(img_h) * 0.047)
    return {
        "width": float(width),
        "height": float(height),
        "area": float(width * height),
        "aspect": float(width / max(height, 1e-6)),
    }


def _box_size_priors(predictions: Sequence[Dict[str, object]], image_size: Tuple[int, int]) -> Tuple[float, float]:
    stats = _shape_reference_stats(predictions, image_size)
    return float(stats["width"]), float(stats["height"])


def _filter_predictions_by_shape(
    predictions: Sequence[Dict[str, object]],
    reference_predictions: Sequence[Dict[str, object]],
    *,
    image_size: Tuple[int, int],
    width_min_scale: float,
    width_max_scale: float,
    height_min_scale: float,
    height_max_scale: float,
    area_min_scale: float,
    area_max_scale: float,
    aspect_max_scale: float,
    edge_keep_margin_px: float,
    edge_keep_margin_ratio: float,
) -> Tuple[List[Dict[str, object]], List[Dict[str, object]], Dict[str, float]]:
    preds = list(predictions)
    if not preds:
        return [], [], {
            "shape_removed_count": 0.0,
            "shape_ref_width_px": 0.0,
            "shape_ref_height_px": 0.0,
            "shape_ref_area_px2": 0.0,
            "shape_ref_aspect": 0.0,
        }

    ref_stats = _shape_reference_stats(reference_predictions, image_size)
    ref_width = max(float(ref_stats["width"]), 1e-6)
    ref_height = max(float(ref_stats["height"]), 1e-6)
    ref_area = max(float(ref_stats["area"]), 1e-6)
    ref_aspect = max(float(ref_stats["aspect"]), 1e-6)

    kept: List[Dict[str, object]] = []
    removed: List[Dict[str, object]] = []
    edge_kept = 0
    aspect_scale = max(float(aspect_max_scale), 1.01)
    for pred in preds:
        if _is_edge_prediction(
            pred,
            image_size,
            margin_px=edge_keep_margin_px,
            margin_ratio=edge_keep_margin_ratio,
        ):
            kept.append(pred)
            edge_kept += 1
            continue
        shape = _prediction_shape(pred)
        width_ok = ref_width * float(width_min_scale) <= shape["width"] <= ref_width * float(width_max_scale)
        height_ok = ref_height * float(height_min_scale) <= shape["height"] <= ref_height * float(height_max_scale)
        area_ok = ref_area * float(area_min_scale) <= shape["area"] <= ref_area * float(area_max_scale)
        aspect_ok = ref_aspect / aspect_scale <= shape["aspect"] <= ref_aspect * aspect_scale
        if width_ok and height_ok and area_ok and aspect_ok:
            kept.append(pred)
        else:
            removed.append(pred)

    stats = {
        "shape_removed_count": float(len(removed)),
        "shape_ref_width_px": float(ref_width),
        "shape_ref_height_px": float(ref_height),
        "shape_ref_area_px2": float(ref_area),
        "shape_ref_aspect": float(ref_aspect),
        "shape_edge_kept_count": float(edge_kept),
    }
    return kept, removed, stats


def _fallback_candidate_poly(
    center: Tuple[float, float],
    image_size: Tuple[int, int],
    *,
    width_px: float,
    height_px: float,
) -> np.ndarray:
    img_w, img_h = image_size
    width = float(width_px) if width_px > 0.0 else max(8.0, float(img_w) * 0.07)
    height = float(height_px) if height_px > 0.0 else max(6.0, float(img_h) * 0.047)
    cx, cy = float(center[0]), float(center[1])
    half_w = width * 0.5
    half_h = height * 0.5
    return np.asarray(
        [
            [cx - half_w, cy - half_h],
            [cx + half_w, cy - half_h],
            [cx + half_w, cy + half_h],
            [cx - half_w, cy + half_h],
        ],
        dtype=np.float32,
    )


def _spacing_candidate_boxes(
    proposals: np.ndarray,
    reference_predictions: Sequence[Dict[str, object]],
    image_size: Tuple[int, int],
    *,
    fallback_width_px: float = 0.0,
    fallback_height_px: float = 0.0,
) -> List[Dict[str, object]]:
    if proposals.size == 0:
        return []

    reference_polys = [np.asarray(pred["poly"], dtype=np.float32).reshape(4, 2) for pred in reference_predictions]
    reference_centers = np.asarray([_poly_center(poly) for poly in reference_polys], dtype=np.float32)
    pseudo: List[Dict[str, object]] = []
    for idx, (x, y, score, mask_value) in enumerate(proposals.tolist(), start=1):
        center = np.asarray([float(x), float(y)], dtype=np.float32)
        cls_id = 0
        name = f"freq{idx}"
        if len(reference_polys) > 0:
            nearest = int(np.argmin(np.linalg.norm(reference_centers - center[None, :], axis=1)))
            template_poly = reference_polys[nearest]
            poly = template_poly + (center - _poly_center(template_poly))[None, :]
            cls_id = int(reference_predictions[nearest].get("cls", 0))
            name = str(reference_predictions[nearest].get("name", f"freq{idx}"))
        else:
            poly = _fallback_candidate_poly(
                (float(x), float(y)),
                image_size,
                width_px=fallback_width_px,
                height_px=fallback_height_px,
            )
        pseudo.append(
            {
                "poly": poly.astype(np.float32),
                "conf": float(score) * float(mask_value),
                "cls": cls_id,
                "name": name,
                "candidate": np.asarray([float(x), float(y), float(score), float(mask_value)], dtype=np.float32),
            }
        )
    return pseudo


def _merge_anchor_points(points: Sequence[np.ndarray], min_distance_px: float) -> np.ndarray:
    arrays = [np.asarray(arr, dtype=np.float32).reshape(-1, 3) for arr in points if np.asarray(arr).size]
    if not arrays:
        return np.zeros((0, 3), dtype=np.float32)
    merged = np.concatenate(arrays, axis=0)
    order = np.argsort(-merged[:, 2])
    keep: List[np.ndarray] = []
    min_distance = max(float(min_distance_px), 1.0)
    for idx in order.tolist():
        candidate = merged[idx]
        if keep and any(np.linalg.norm(candidate[:2] - prev[:2]) <= min_distance for prev in keep):
            continue
        keep.append(candidate)
    return np.stack(keep, axis=0).astype(np.float32) if keep else np.zeros((0, 3), dtype=np.float32)


def _fit_slanted_row(points: np.ndarray, max_abs_slope: float) -> Tuple[float, float, float]:
    pts = np.asarray(points, dtype=np.float32)
    if pts.shape[0] == 0:
        return 0.0, 0.0, 0.0
    xs = pts[:, 0].astype(np.float64)
    ys = pts[:, 1].astype(np.float64)
    if pts.shape[0] < 2 or float(np.std(xs)) < 1e-6:
        slope = 0.0
        intercept = float(np.median(ys))
        residual = float(np.median(np.abs(ys - intercept)))
        return slope, intercept, residual
    slope, intercept = np.polyfit(xs, ys, deg=1)
    slope = float(np.clip(slope, -abs(float(max_abs_slope)), abs(float(max_abs_slope))))
    intercept = float(np.median(ys - slope * xs))
    residual = float(np.median(np.abs(ys - (slope * xs + intercept))))
    return float(slope), float(intercept), residual


def _group_points_by_y_px(points: np.ndarray, row_tolerance_px: float) -> List[np.ndarray]:
    if points.size == 0:
        return []
    order = np.argsort(points[:, 1])
    pts = points[order]
    rows: List[List[np.ndarray]] = []
    current: List[np.ndarray] = [pts[0]]
    current_y = float(pts[0, 1])
    tol = max(float(row_tolerance_px), 1.0)
    for pt in pts[1:]:
        if abs(float(pt[1]) - current_y) <= tol:
            current.append(pt)
            current_y = float(np.mean([p[1] for p in current]))
        else:
            rows.append(current)
            current = [pt]
            current_y = float(pt[1])
    rows.append(current)
    return [np.asarray(row, dtype=np.float32) for row in rows]


def _build_slanted_rows(
    points: np.ndarray,
    *,
    row_tolerance_px: float,
    max_abs_slope: float,
) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for row_pts in _group_points_by_y_px(points, row_tolerance_px=row_tolerance_px):
        slope, intercept, residual = _fit_slanted_row(row_pts, max_abs_slope=max_abs_slope)
        rows.append(
            {
                "points": row_pts,
                "slope": float(slope),
                "intercept": float(intercept),
                "residual": float(residual),
                "x_min": float(row_pts[:, 0].min()),
                "x_max": float(row_pts[:, 0].max()),
                "y_center": float(np.median(row_pts[:, 1])),
            }
        )
    return rows


def _row_y_at_x(row: Dict[str, object], x: float) -> float:
    return float(row["slope"]) * float(x) + float(row["intercept"])


def _distance_to_row(row: Dict[str, object], point: np.ndarray) -> float:
    x = float(point[0])
    y = float(point[1])
    y_hat = _row_y_at_x(row, x)
    return abs(y - y_hat)


def _prediction_center(pred: Dict[str, object]) -> np.ndarray:
    return _poly_center(np.asarray(pred["poly"], dtype=np.float32))


def _spacing_complete_candidates_slanted(
    score_map: torch.Tensor,
    gate: torch.Tensor,
    *,
    image_size: Tuple[int, int],
    anchor_points: np.ndarray,
    row_tolerance_px: float,
    nms_distance_px: float,
    min_row_seeds: int,
    max_candidates: int,
    period_prior_px: float,
    period_prior_ratio: float,
    period_scale: float,
    proposal_score_threshold: float,
    gate_threshold: float,
    refine_window_px: float,
    refine_step_px: float,
    refine_prior_sigma_px: float,
    max_abs_slope: float,
    line_residual_px: float,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, float], List[Tuple[float, float, float, float]]]:
    anchors = np.asarray(anchor_points, dtype=np.float32)
    if anchors.size == 0:
        stats = {
            "spacing_anchor_count": 0.0,
            "spacing_row_count": 0.0,
            "spacing_used_rows": 0.0,
            "spacing_mean_period_px": 0.0,
            "spacing_candidate_count": 0.0,
            "spacing_mean_row_residual_px": 0.0,
        }
        return np.zeros((0, 3), dtype=np.float32), np.zeros((0, 4), dtype=np.float32), stats, []

    rows = _build_slanted_rows(
        anchors[:, :2],
        row_tolerance_px=row_tolerance_px,
        max_abs_slope=max_abs_slope,
    )
    img_w, img_h = image_size
    nms = max(float(nms_distance_px), 1.0)
    proposals: List[Tuple[float, float, float, float]] = []
    row_segments: List[Tuple[float, float, float, float]] = []
    periods: List[float] = []
    used_rows = 0
    residuals: List[float] = []

    for row in rows:
        row_pts = np.asarray(row["points"], dtype=np.float32)
        if row_pts.shape[0] < int(min_row_seeds):
            continue

        seed_x = np.sort(row_pts[:, 0].astype(np.float32))
        period_est = _estimate_spacing_px(seed_x)
        period = float(period_est)
        prior = max(float(period_prior_px), 0.0)
        ratio = max(float(period_prior_ratio), 1.01)
        if prior > 0.0 and (period < 3.0 or period > prior * ratio or period < prior / ratio):
            period = prior
        if period < 3.0:
            continue
        period *= max(float(period_scale), 0.05)

        phase = _estimate_spacing_phase(seed_x, period)
        x_min = max(0.0, float(seed_x.min()) - period)
        x_max = min(float(img_w), float(seed_x.max()) + period)
        row_segments.append((x_min, _row_y_at_x(row, x_min), x_max, _row_y_at_x(row, x_max)))
        used_rows += 1
        periods.append(period)
        residuals.append(float(row["residual"]))

        k0 = int(np.floor((x_min - phase) / period))
        k1 = int(np.ceil((x_max - phase) / period))
        for k in range(k0, k1 + 1):
            x = phase + k * period
            if x < 0.0 or x >= float(img_w):
                continue
            y = _row_y_at_x(row, x)
            if y < 0.0 or y >= float(img_h):
                continue
            if np.min(np.linalg.norm(anchors[:, :2] - np.asarray([x, y], dtype=np.float32), axis=1)) <= nms:
                continue
            refined = _refine_spacing_candidate(
                score_map,
                gate,
                x,
                y,
                image_size=image_size,
                proposal_score_threshold=proposal_score_threshold,
                gate_threshold=gate_threshold,
                refine_window_px=refine_window_px,
                refine_step_px=refine_step_px,
                refine_prior_sigma_px=refine_prior_sigma_px,
            )
            if refined is None:
                continue
            refined_x, refined_y, score, mask_value = refined
            residual_tol = max(float(line_residual_px), float(row["residual"]) * 2.0 + 1.0)
            if abs(float(refined_y) - _row_y_at_x(row, refined_x)) > residual_tol:
                continue
            proposals.append((float(refined_x), float(refined_y), float(score), float(mask_value)))

    if not proposals:
        stats = {
            "spacing_anchor_count": float(anchors.shape[0]),
            "spacing_row_count": float(len(rows)),
            "spacing_used_rows": float(used_rows),
            "spacing_mean_period_px": float(np.mean(periods)) if periods else 0.0,
            "spacing_candidate_count": 0.0,
            "spacing_mean_row_residual_px": float(np.mean(residuals)) if residuals else 0.0,
        }
        return anchors[:, :3], np.zeros((0, 4), dtype=np.float32), stats, row_segments

    proposals_arr = np.asarray(proposals, dtype=np.float32)
    order = np.argsort(-(proposals_arr[:, 2] * proposals_arr[:, 3]))
    selected: List[np.ndarray] = []
    for idx in order:
        cand = proposals_arr[idx]
        if selected and any(np.linalg.norm(cand[:2] - prev[:2]) <= nms for prev in selected):
            continue
        selected.append(cand)
        if len(selected) >= int(max_candidates):
            break
    selected_arr = np.asarray(selected, dtype=np.float32) if selected else np.zeros((0, 4), dtype=np.float32)
    stats = {
        "spacing_anchor_count": float(anchors.shape[0]),
        "spacing_row_count": float(len(rows)),
        "spacing_used_rows": float(used_rows),
        "spacing_mean_period_px": float(np.mean(periods)) if periods else 0.0,
        "spacing_candidate_count": float(selected_arr.shape[0]),
        "spacing_mean_row_residual_px": float(np.mean(residuals)) if residuals else 0.0,
    }
    return anchors[:, :3], selected_arr, stats, row_segments


def _filter_predictions_by_slanted_rows(
    predictions: Sequence[Dict[str, object]],
    *,
    anchor_points: Optional[np.ndarray] = None,
    image_size: Tuple[int, int],
    row_tolerance_px: float,
    min_anchor_points: int,
    frequency_ratio: float,
    frequency_min_points: int,
    max_abs_slope: float,
    line_residual_px: float,
    edge_keep_margin_px: float,
    edge_keep_margin_ratio: float,
) -> Tuple[List[Dict[str, object]], List[Dict[str, object]], Dict[str, float]]:
    preds = list(predictions)
    if not preds:
        return [], [], {
            "pattern_row_count": 0.0,
            "pattern_supported_row_count": 0.0,
            "pattern_removed_count": 0.0,
            "pattern_mean_period_px": 0.0,
            "pattern_edge_kept_count": 0.0,
        }

    if anchor_points is not None:
        anchors = np.asarray(anchor_points, dtype=np.float32)
        if anchors.size == 0:
            anchors = np.stack([_prediction_center(pred) for pred in preds], axis=0).astype(np.float32)
        else:
            anchors = anchors.reshape(-1, anchors.shape[-1])[:, :2].astype(np.float32)
    else:
        anchors = np.stack([_prediction_center(pred) for pred in preds], axis=0).astype(np.float32)
    rows = _build_slanted_rows(
        anchors,
        row_tolerance_px=row_tolerance_px,
        max_abs_slope=max_abs_slope,
    )
    min_points = max(int(min_anchor_points), 1)
    freq_min_points = max(int(frequency_min_points), 2)
    row_periods: List[Tuple[int, float]] = []
    for idx, row in enumerate(rows):
        row_pts = np.asarray(row["points"], dtype=np.float32)
        if row_pts.shape[0] < freq_min_points:
            continue
        period = _estimate_spacing_px(np.sort(row_pts[:, 0]))
        if period >= 3.0:
            row_periods.append((idx, period))
    global_period = float(np.median([period for _, period in row_periods])) if row_periods else 0.0
    bad_rows: set[int] = set()
    if global_period >= 3.0 and frequency_ratio > 1.0:
        for row_idx, period in row_periods:
            mismatch = max(period / max(global_period, 1e-6), global_period / max(period, 1e-6))
            if mismatch > float(frequency_ratio):
                bad_rows.add(int(row_idx))

    supported_rows = 0
    supported_row_records: List[Dict[str, object]] = []
    for row_idx, row in enumerate(rows):
        row_pts = np.asarray(row["points"], dtype=np.float32)
        row_supported = row_pts.shape[0] >= min_points and row_idx not in bad_rows
        if row_supported:
            supported_rows += 1
            supported_row_records.append(row)

    if not supported_row_records:
        return preds, [], {
            "pattern_row_count": float(len(rows)),
            "pattern_supported_row_count": 0.0,
            "pattern_removed_count": 0.0,
            "pattern_mean_period_px": global_period,
            "pattern_edge_kept_count": 0.0,
        }

    kept: List[Dict[str, object]] = []
    removed: List[Dict[str, object]] = []
    edge_kept = 0
    for pred in preds:
        if _is_edge_prediction(
            pred,
            image_size,
            margin_px=edge_keep_margin_px,
            margin_ratio=edge_keep_margin_ratio,
        ):
            kept.append(pred)
            edge_kept += 1
            continue
        center = _prediction_center(pred)
        keep = False
        for row in supported_row_records:
            residual_tol = max(float(line_residual_px), float(row["residual"]) * 2.0 + 1.0)
            if _distance_to_row(row, center) <= residual_tol:
                keep = True
                break
        if keep:
            kept.append(pred)
        else:
            removed.append(pred)

    stats = {
        "pattern_row_count": float(len(rows)),
        "pattern_supported_row_count": float(supported_rows),
        "pattern_removed_count": float(len(removed)),
        "pattern_mean_period_px": global_period,
        "pattern_edge_kept_count": float(edge_kept),
    }
    return kept, removed, stats


def _draw_slanted_row_segments(
    image: Image.Image,
    row_segments: Sequence[Tuple[float, float, float, float]],
    *,
    color: Tuple[int, int, int] = (255, 220, 30),
    annotation_scale: float = 1.0,
) -> Image.Image:
    canvas = image.convert("RGB").copy()
    from PIL import ImageDraw
    draw = ImageDraw.Draw(canvas)
    width = max(2, int(round(max(canvas.size) / 360.0 * max(float(annotation_scale), 0.1))))
    for x0, y0, x1, y1 in row_segments:
        draw.line((x0, y0, x1, y1), fill=(0, 0, 0), width=max(1, width + 2))
        draw.line((x0, y0, x1, y1), fill=color, width=width)
    return canvas


def main() -> None:
    args = _parse_args()
    if args.image and args.image_dir:
        raise ValueError("Use either --image or --image-dir, not both")

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
    mask_level = str(args.mask_level or feature_level).lower()
    weights = _resolve_weights(config, args.weights)

    output_dir = (
        Path(args.output_dir).resolve()
        if args.output_dir
        else Path("outputs") / "yolo_mask_effect"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.image:
        image_path = Path(args.image).resolve()
        if not image_path.is_file():
            raise FileNotFoundError(f"Image not found: {image_path}")
        batches = _iter_image_path_batches([image_path], image_size, args.batch_size)
    elif args.image_dir:
        batches = _iter_image_dir_batches(Path(args.image_dir), image_size, args.batch_size)
    else:
        batches = _build_loader(config, split=args.split, batch_size=args.batch_size, num_workers=args.num_workers)

    print(f"[yolo-mask] output_dir={output_dir}", flush=True)
    print(f"[yolo-mask] device={device} imgsz={image_size} raw_level={feature_level} mask_level={mask_level}", flush=True)

    student = _build_yolo_student(weights, config, feature_level, device)
    mask_student = student if mask_level == feature_level else _build_yolo_student(weights, config, mask_level, device)
    yolo = YOLO(weights)
    dino = _build_dino(config, device)
    detect_module = _find_detect_module(mask_student.det_model)
    detect_level_idx = {"p3": 0, "p4": 1, "p5": 2}[mask_level]

    rows: List[Dict[str, object]] = []
    sample_count = 0

    for batch in batches:
        images = batch["images"].to(device)
        image_paths = [str(p) for p in batch.get("image_paths", [])]
        if not image_paths:
            image_paths = [f"sample_{sample_count + i:04d}" for i in range(images.shape[0])]

        with torch.no_grad():
            raw_feat = _extract_yolo_feature(student, images, feature_level)
            mask_src_feat = raw_feat if mask_level == feature_level else _extract_yolo_feature(mask_student, images, mask_level)
            dino_feat = dino(images).detach()

        for i in range(images.shape[0]):
            if sample_count >= args.num_samples:
                break

            image_path = image_paths[i]
            stem = Path(image_path).stem
            original_pil = Image.open(image_path).convert("RGB") if Path(image_path).is_file() else _tensor_image_to_pil(images[i].detach().cpu())

            raw_preds = (
                _predict_one(
                    yolo,
                    image_path,
                    imgsz=image_size,
                    conf=args.conf,
                    iou=args.iou,
                    max_det=args.max_det,
                    device=device,
                )
                if Path(image_path).is_file()
                else []
            )
            mask_fused_preds = (
                _predict_one_dino_gated(
                    yolo,
                    dino,
                    image_path,
                    imgsz=image_size,
                    conf=args.conf,
                    iou=args.iou,
                    max_det=args.max_det,
                    device=device,
                    feature_level=mask_level,
                    gate_min=args.mask_fuse_bg,
                    gate_max=args.mask_fuse_fg,
                    target_coverage=args.mask_target_coverage,
                    min_coverage=args.mask_min_coverage,
                    max_coverage=args.mask_max_coverage,
                )
                if Path(image_path).is_file()
                else []
            )

            dino_mask, dino_mask_overlay, dino_mask_stats = _dino_mask_images(
                dino_feat[i],
                original_pil,
                target_coverage=args.mask_target_coverage,
                min_coverage=args.mask_min_coverage,
                max_coverage=args.mask_max_coverage,
                blur_radius=args.mask_blur_radius,
            )
            fused_feat, _gate = _apply_dino_mask_fuse_to_feature(
                mask_src_feat[i],
                dino_feat[i],
                bg_multiplier=args.mask_fuse_bg,
                fg_multiplier=args.mask_fuse_fg,
                target_coverage=args.mask_target_coverage,
                min_coverage=args.mask_min_coverage,
                max_coverage=args.mask_max_coverage,
            )
            cls_fused_map, _ = _head_cls_score_map(detect_module, fused_feat, detect_level_idx)

            raw_response = _feature_response_image(raw_feat[i], original_pil.size)
            fused_response = _feature_response_image(fused_feat, original_pil.size)
            raw_points = _predictions_to_center_points(raw_preds)
            fused_points = _predictions_to_center_points(mask_fused_preds)
            if args.spacing_anchor_source == "raw":
                anchor_points = raw_points
            elif args.spacing_anchor_source == "both":
                anchor_points = _merge_anchor_points([raw_points, fused_points], args.spacing_nms_distance_px)
            else:
                anchor_points = fused_points
            if anchor_points.size:
                anchor_points = anchor_points[anchor_points[:, 2] >= float(args.spacing_score_threshold)]

            kept_pattern_preds: List[Dict[str, object]] = list(mask_fused_preds)
            removed_pattern_preds: List[Dict[str, object]] = []
            added_pattern_preds: List[Dict[str, object]] = []
            pattern_final_preds: List[Dict[str, object]] = list(mask_fused_preds)
            spacing_seed_points = anchor_points
            spacing_proposals = np.zeros((0, 4), dtype=np.float32)
            pattern_candidate_points = np.zeros((0, 4), dtype=np.float32)
            row_segments: List[Tuple[float, float, float, float]] = []
            spacing_overlap_removed = 0
            shape_removed_existing = 0
            shape_removed_added = 0
            pattern_stats: Dict[str, float] = {
                "pattern_row_count": 0.0,
                "pattern_supported_row_count": 0.0,
                "pattern_removed_count": 0.0,
                "pattern_mean_period_px": 0.0,
                "pattern_edge_kept_count": 0.0,
            }
            spacing_stats: Dict[str, float] = {
                "spacing_anchor_count": float(anchor_points.shape[0]) if anchor_points.size else 0.0,
                "spacing_row_count": 0.0,
                "spacing_used_rows": 0.0,
                "spacing_mean_period_px": 0.0,
                "spacing_candidate_count": 0.0,
                "spacing_mean_row_residual_px": 0.0,
            }
            shape_stats: Dict[str, float] = {
                "shape_removed_count": 0.0,
                "shape_ref_width_px": 0.0,
                "shape_ref_height_px": 0.0,
                "shape_ref_area_px2": 0.0,
                "shape_ref_aspect": 0.0,
                "shape_edge_kept_count": 0.0,
            }
            use_pattern_filter = bool(args.pattern_filter or args.spacing_complete)
            if use_pattern_filter:
                kept_pattern_preds, removed_pattern_preds, pattern_stats = _filter_predictions_by_slanted_rows(
                    mask_fused_preds,
                    anchor_points=anchor_points,
                    image_size=original_pil.size,
                    row_tolerance_px=args.spacing_row_tolerance_px,
                    min_anchor_points=args.spacing_pseudo_min_anchor_per_row,
                    frequency_ratio=args.spacing_pseudo_row_frequency_ratio,
                    frequency_min_points=args.spacing_pseudo_row_frequency_min_points,
                    max_abs_slope=args.spacing_line_max_slope,
                    line_residual_px=args.spacing_line_residual_px,
                    edge_keep_margin_px=args.edge_keep_margin_px,
                    edge_keep_margin_ratio=args.edge_keep_margin_ratio,
                )
            shape_reference_preds = raw_preds or kept_pattern_preds or mask_fused_preds
            if args.shape_filter:
                kept_pattern_preds, shape_removed_preds, shape_stats = _filter_predictions_by_shape(
                    kept_pattern_preds,
                    shape_reference_preds,
                    image_size=original_pil.size,
                    width_min_scale=args.shape_width_min_scale,
                    width_max_scale=args.shape_width_max_scale,
                    height_min_scale=args.shape_height_min_scale,
                    height_max_scale=args.shape_height_max_scale,
                    area_min_scale=args.shape_area_min_scale,
                    area_max_scale=args.shape_area_max_scale,
                    aspect_max_scale=args.shape_aspect_max_scale,
                    edge_keep_margin_px=args.edge_keep_margin_px,
                    edge_keep_margin_ratio=args.edge_keep_margin_ratio,
                )
                removed_pattern_preds.extend(shape_removed_preds)
                shape_removed_existing = len(shape_removed_preds)
            if args.spacing_complete:
                kept_points = _predictions_to_center_points(kept_pattern_preds)
                if args.spacing_anchor_source == "raw":
                    completion_anchors = raw_points
                elif args.spacing_anchor_source == "both":
                    completion_anchors = _merge_anchor_points([raw_points, kept_points], args.spacing_nms_distance_px)
                else:
                    completion_anchors = kept_points
                if completion_anchors.size:
                    completion_anchors = completion_anchors[completion_anchors[:, 2] >= float(args.spacing_score_threshold)]
                if completion_anchors.size == 0:
                    completion_anchors = anchor_points
                spacing_seed_points, spacing_proposals, spacing_stats, row_segments = _spacing_complete_candidates_slanted(
                    cls_fused_map,
                    _gate,
                    image_size=original_pil.size,
                    anchor_points=completion_anchors,
                    row_tolerance_px=args.spacing_row_tolerance_px,
                    nms_distance_px=args.spacing_nms_distance_px,
                    min_row_seeds=args.spacing_min_row_seeds,
                    max_candidates=args.spacing_max_candidates,
                    period_prior_px=args.spacing_period_prior_px,
                    period_prior_ratio=args.spacing_period_prior_ratio,
                    period_scale=args.spacing_period_scale,
                    proposal_score_threshold=args.spacing_proposal_score_threshold,
                    gate_threshold=args.spacing_dino_threshold,
                    refine_window_px=args.spacing_refine_window_px,
                    refine_step_px=args.spacing_refine_step_px,
                    refine_prior_sigma_px=args.spacing_refine_prior_sigma_px,
                    max_abs_slope=args.spacing_line_max_slope,
                    line_residual_px=args.spacing_line_residual_px,
                )
                reference_preds = kept_pattern_preds or mask_fused_preds or raw_preds
                fallback_width_px, fallback_height_px = _box_size_priors(reference_preds, original_pil.size)
                added_pattern_preds = _spacing_candidate_boxes(
                    spacing_proposals,
                    reference_preds,
                    original_pil.size,
                    fallback_width_px=fallback_width_px,
                    fallback_height_px=fallback_height_px,
                )
                added_pattern_preds, spacing_overlap_removed = _filter_pseudo_boxes_by_reference_overlap(
                    added_pattern_preds,
                    kept_pattern_preds or mask_fused_preds,
                    iou_threshold=args.spacing_pseudo_overlap_iou,
                )
                if args.shape_filter and added_pattern_preds:
                    added_pattern_preds, removed_added_shape_preds, _ = _filter_predictions_by_shape(
                        added_pattern_preds,
                        shape_reference_preds,
                        image_size=original_pil.size,
                        width_min_scale=args.shape_width_min_scale,
                        width_max_scale=args.shape_width_max_scale,
                        height_min_scale=args.shape_height_min_scale,
                        height_max_scale=args.shape_height_max_scale,
                        area_min_scale=args.shape_area_min_scale,
                        area_max_scale=args.shape_area_max_scale,
                        aspect_max_scale=args.shape_aspect_max_scale,
                        edge_keep_margin_px=args.edge_keep_margin_px,
                        edge_keep_margin_ratio=args.edge_keep_margin_ratio,
                    )
                    shape_removed_added = len(removed_added_shape_preds)
                pattern_candidate_points = _candidates_from_pseudo_boxes(added_pattern_preds)
                pattern_final_preds = kept_pattern_preds + added_pattern_preds
            else:
                pattern_final_preds = kept_pattern_preds

            raw_overlay = _draw_predictions(
                original_pil,
                raw_preds,
                (0, 245, 255),
                label_prefix="Y:",
                max_labels=0,
                annotation_scale=args.annotation_scale,
                line_width_scale=args.box_scale,
                shadow=False,
            )
            fused_overlay = _draw_predictions(
                original_pil,
                mask_fused_preds,
                (255, 35, 210),
                label_prefix="D:",
                max_labels=0,
                annotation_scale=args.annotation_scale,
                line_width_scale=args.box_scale,
                shadow=False,
            )
            pattern_overlay = _draw_slanted_row_segments(
                original_pil,
                row_segments,
                color=(80, 220, 255),
                annotation_scale=args.annotation_scale,
            )
            pattern_overlay = _draw_predictions(
                pattern_overlay,
                removed_pattern_preds,
                (150, 150, 150),
                label_prefix="R:",
                max_labels=0,
                annotation_scale=args.annotation_scale,
                line_width_scale=args.box_scale,
                shadow=False,
            )
            pattern_overlay = _draw_predictions(
                pattern_overlay,
                kept_pattern_preds,
                (255, 35, 210),
                label_prefix="K:",
                max_labels=0,
                annotation_scale=args.annotation_scale,
                line_width_scale=args.box_scale,
                shadow=False,
            )
            pattern_overlay = _draw_predictions(
                pattern_overlay,
                added_pattern_preds,
                (255, 220, 30),
                label_prefix="A:",
                max_labels=0,
                annotation_scale=args.annotation_scale,
                line_width_scale=args.box_scale,
                shadow=False,
            )
            spacing_debug = _draw_spacing_points(
                original_pil,
                spacing_seed_points,
                pattern_candidate_points,
                pseudo_boxes=added_pattern_preds,
                annotation_scale=args.annotation_scale,
            )
            spacing_debug = _draw_slanted_row_segments(
                spacing_debug,
                row_segments,
                color=(80, 220, 255),
                annotation_scale=args.annotation_scale,
            )
            delta_overlay = _draw_prediction_difference(
                original_pil,
                raw_preds,
                mask_fused_preds,
                annotation_scale=args.annotation_scale,
            )
            pattern_delta_overlay = _draw_prediction_difference(
                original_pil,
                raw_preds,
                pattern_final_preds,
                annotation_scale=args.annotation_scale,
            )

            level_label = _feature_level_label(mask_level)
            tiles = [
                _make_tile(original_pil, "Origin Image", args.tile_size),
                _make_tile(raw_response, f"YOLO {feature_level.upper()} Feature", args.tile_size),
                _make_tile(dino_mask, "DINO Mask", args.tile_size),
                _make_tile(fused_response, f"DINO x YOLO {level_label}", args.tile_size),
                _make_tile(dino_mask_overlay, "Mask on Original", args.tile_size),
                _make_tile(raw_overlay, "Raw YOLO Boxes cyan", args.tile_size),
                _make_tile(fused_overlay, "Mask-Fused Boxes magenta", args.tile_size),
                _make_tile(pattern_overlay, "Pattern-Limited magenta kept yellow add gray rm", args.tile_size),
                _make_tile(spacing_debug, "Row Prior cyan line green seed yellow add", args.tile_size),
                _make_tile(delta_overlay, "Prediction Difference", args.tile_size),
                _make_tile(pattern_delta_overlay, "YOLO vs Pattern-Limited", args.tile_size),
            ]
            panel = _make_grid(tiles, cols=max(1, int(args.single_panel_cols)))
            panel_path = output_dir / f"{sample_count:04d}_{stem}_panel.png"
            panel.save(panel_path)

            pred_metrics = _prediction_delta_metrics(raw_preds, mask_fused_preds)
            pattern_metrics = _prediction_delta_metrics(raw_preds, pattern_final_preds)
            row: Dict[str, object] = {
                "index": sample_count,
                "image_path": image_path,
                "panel_path": str(panel_path),
                "weights": weights,
                "feature_level": feature_level,
                "mask_level": mask_level,
                "mask_component": dino_mask_stats["component"],
                "mask_inverted": dino_mask_stats["inverted"],
                "mask_threshold": dino_mask_stats["threshold"],
                "mask_grid_coverage": dino_mask_stats["coverage"],
                "mask_display_coverage": dino_mask_stats["display_coverage"],
                "raw_box_count": float(len(raw_preds)),
                "mask_fused_box_count": float(len(mask_fused_preds)),
                "pattern_kept_box_count": float(len(kept_pattern_preds)),
                "pattern_removed_noise_count": float(len(removed_pattern_preds)),
                "pattern_removed_shape_existing_count": float(shape_removed_existing),
                "pattern_removed_shape_added_count": float(shape_removed_added),
                "spacing_added_box_count": float(len(added_pattern_preds)),
                "pattern_final_box_count": float(len(pattern_final_preds)),
                "spacing_overlap_removed_count": float(spacing_overlap_removed),
                "matched_box_count": pred_metrics["matched_box_count"],
                "mean_matched_aabb_iou": pred_metrics["mean_matched_aabb_iou"],
                "mean_center_shift_px": pred_metrics["mean_center_shift_px"],
                "pattern_matched_box_count": pattern_metrics["matched_box_count"],
                "pattern_mean_matched_aabb_iou": pattern_metrics["mean_matched_aabb_iou"],
                "pattern_mean_center_shift_px": pattern_metrics["mean_center_shift_px"],
                "mask_fuse_bg": float(args.mask_fuse_bg),
                "mask_fuse_fg": float(args.mask_fuse_fg),
                "spacing_anchor_source": args.spacing_anchor_source,
            }
            row.update({k: float(v) for k, v in pattern_stats.items()})
            row.update({k: float(v) for k, v in spacing_stats.items()})
            row.update({k: float(v) for k, v in shape_stats.items()})
            rows.append(row)
            print(
                f"[yolo-mask] saved panel={panel_path} raw={len(raw_preds)} mask={len(mask_fused_preds)} "
                f"kept={len(kept_pattern_preds)} add={len(added_pattern_preds)} rm={len(removed_pattern_preds)} "
                f"shape_rm={shape_removed_existing}+{shape_removed_added} "
                f"final={len(pattern_final_preds)}",
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
        "weights": weights,
        "num_samples": len(rows),
        "feature_level": feature_level,
        "mask_level": mask_level,
        "mask_fuse_bg": float(args.mask_fuse_bg),
        "mask_fuse_fg": float(args.mask_fuse_fg),
        "mean_raw_box_count": float(np.mean([float(r["raw_box_count"]) for r in rows])),
        "mean_mask_fused_box_count": float(np.mean([float(r["mask_fused_box_count"]) for r in rows])),
        "mean_pattern_final_box_count": float(np.mean([float(r["pattern_final_box_count"]) for r in rows])),
        "mean_pattern_removed_noise_count": float(np.mean([float(r["pattern_removed_noise_count"]) for r in rows])),
        "mean_pattern_removed_shape_existing_count": float(np.mean([float(r["pattern_removed_shape_existing_count"]) for r in rows])),
        "mean_pattern_removed_shape_added_count": float(np.mean([float(r["pattern_removed_shape_added_count"]) for r in rows])),
        "mean_spacing_added_box_count": float(np.mean([float(r["spacing_added_box_count"]) for r in rows])),
        "mean_matched_aabb_iou": float(np.mean([float(r["mean_matched_aabb_iou"]) for r in rows])),
        "mean_center_shift_px": float(np.mean([float(r["mean_center_shift_px"]) for r in rows])),
        "mean_pattern_matched_aabb_iou": float(np.mean([float(r["pattern_mean_matched_aabb_iou"]) for r in rows])),
        "mean_pattern_center_shift_px": float(np.mean([float(r["pattern_mean_center_shift_px"]) for r in rows])),
        "mean_pattern_row_count": float(np.mean([float(r["pattern_row_count"]) for r in rows])),
        "mean_pattern_supported_row_count": float(np.mean([float(r["pattern_supported_row_count"]) for r in rows])),
        "mean_pattern_edge_kept_count": float(np.mean([float(r["pattern_edge_kept_count"]) for r in rows])),
        "mean_spacing_candidate_count": float(np.mean([float(r["spacing_candidate_count"]) for r in rows])),
        "mean_spacing_period_px": float(np.mean([float(r["spacing_mean_period_px"]) for r in rows])),
        "mean_shape_edge_kept_count": float(np.mean([float(r["shape_edge_kept_count"]) for r in rows])),
        "output_dir": str(output_dir),
    }
    _save_json(output_dir / "summary.json", summary)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
