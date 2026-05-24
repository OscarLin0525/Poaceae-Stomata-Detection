#!/usr/bin/env python3
"""
Visualize whether DINO-aligned training changed YOLO features and predictions.

This script compares two exported Ultralytics YOLO checkpoints:

1. ``--before-weights``: the baseline student before MTKD/DINO alignment.
2. ``--after-weights``: the student exported after MTKD/DINO alignment.

For each sampled image it saves a panel with:
- input image
- DINO PCA feature map
- baseline YOLO feature PCA
- after-training YOLO feature PCA
- baseline predictions
- after-training predictions
- before/after prediction overlay

It also writes ``metrics.csv`` with linear CKA(feature, DINO) before/after and
simple box-count / matched-box movement statistics.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFilter, ImageFont

# Prefer the repository's bundled Ultralytics fork over site-packages.
REPO_ROOT = Path(__file__).resolve().parents[1]
ULTRALYTICS_ROOT = REPO_ROOT / "ultralytics"
for candidate in (REPO_ROOT, ULTRALYTICS_ROOT):
    if candidate.is_dir():
        candidate_str = str(candidate)
        if candidate_str not in sys.path:
            sys.path.insert(0, candidate_str)

from .engine.build_dino import DinoFeatureExtractor
from .models.yolo_wrappers import YOLOStudentDetector
from .visualize_alignment import (
    _build_loader,
    _build_merged_config,
    _feature_to_tokens,
    _fit_joint_pca,
    _fit_pca,
    _make_tile,
    _save_metrics_csv,
    _tensor_image_to_pil,
    _tokens_to_pca_image,
)


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Compare YOLO feature maps and predictions before/after DINO-aligned training.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--config", type=str, required=True, help="MTKD training config JSON/YAML")
    p.add_argument(
        "--before-weights",
        type=str,
        default=None,
        help="Baseline YOLO .pt. Defaults to model.student_config.weights from config.",
    )
    p.add_argument(
        "--after-weights",
        type=str,
        required=True,
        help="DINO-aligned exported YOLO .pt, e.g. outputs/.../student_best.pt.",
    )
    p.add_argument(
        "--image",
        type=str,
        default=None,
        help="Optional single image path for a focused smoke test.",
    )
    p.add_argument(
        "--image-dir",
        type=str,
        default=None,
        help="Optional image directory. If omitted, samples from the config dataloader split.",
    )
    p.add_argument("--split", type=str, default="val", choices=["train", "val"])
    p.add_argument("--num-samples", type=int, default=12)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument(
        "--preset",
        type=str,
        default="none",
        choices=["none", "single-panel", "single-panel-spacing"],
        help="Shortcut for common visualization settings.",
    )
    p.add_argument("--imgsz", type=int, default=None, help="Image size for feature extraction/prediction")
    p.add_argument("--feature-level", type=str, default=None, choices=["p3", "p4", "p5"])
    p.add_argument("--conf", type=float, default=0.25, help="Prediction confidence threshold for overlays")
    p.add_argument("--iou", type=float, default=0.7, help="Prediction NMS IoU threshold")
    p.add_argument("--max-det", type=int, default=300)
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--output-dir", type=str, default=None)
    p.add_argument("--tile-size", type=int, default=300)
    p.add_argument(
        "--single-panel",
        action="store_true",
        help=(
            "Save one compact per-image panel focused on raw image, YOLO feature response, "
            "DINO mask, DINO-mask-multiplied YOLO feature, and prediction difference. "
            "This auto-enables --dino-mask-fuse."
        ),
    )
    p.add_argument("--single-panel-cols", type=int, default=4, help="Columns in --single-panel output.")
    p.add_argument(
        "--single-panel-box-scale",
        type=float,
        default=1.0,
        help=(
            "Line-width scale for bbox overlays in --single-panel. This is separate from "
            "--annotation-scale so small stomata boxes do not become filled blocks."
        ),
    )
    p.add_argument(
        "--single-panel-marker-scale",
        type=float,
        default=1.3,
        help="Marker/text scale for candidate and spacing overlays in --single-panel.",
    )
    p.add_argument(
        "--annotation-scale",
        type=float,
        default=1.0,
        help="Scale marker radius, line width, and diagnostic text in saved panels.",
    )
    p.add_argument("--mask-target-coverage", type=float, default=0.12)
    p.add_argument("--mask-min-coverage", type=float, default=0.02)
    p.add_argument("--mask-max-coverage", type=float, default=0.30)
    p.add_argument("--mask-blur-radius", type=float, default=2.0)
    p.add_argument(
        "--dino-gate",
        action="store_true",
        help="Run an extra prediction with DINO-derived multiplicative gating on one YOLO feature level.",
    )
    p.add_argument(
        "--dino-gate-level",
        type=str,
        default=None,
        choices=["p3", "p4", "p5"],
        help="YOLO feature level to gate. Defaults to --feature-level.",
    )
    p.add_argument(
        "--dino-gate-min",
        type=float,
        default=0.5,
        help="Feature multiplier for DINO background regions.",
    )
    p.add_argument(
        "--dino-gate-max",
        type=float,
        default=1.5,
        help="Feature multiplier for DINO foreground regions.",
    )
    p.add_argument(
        "--dino-fuse",
        action="store_true",
        help="Run an extra prediction after fusing projected DINO feature maps into YOLO features.",
    )
    p.add_argument(
        "--dino-fuse-level",
        type=str,
        default=None,
        choices=["p3", "p4", "p5"],
        help="YOLO feature level to fuse with DINO. Defaults to --feature-level.",
    )
    p.add_argument(
        "--dino-fuse-mode",
        type=str,
        default="blend",
        choices=["blend", "add", "mul", "replace"],
        help="How to combine YOLO and projected DINO features.",
    )
    p.add_argument(
        "--dino-fuse-strength",
        type=float,
        default=0.5,
        help="Fusion strength. For blend, 0.5 means 50%% YOLO + 50%% projected DINO.",
    )
    p.add_argument(
        "--dino-fuse-components",
        type=int,
        default=0,
        help="Number of DINO PCA channels before repeat/pad to YOLO channels. 0 = YOLO channel count.",
    )
    p.add_argument(
        "--dino-mask-fuse",
        action="store_true",
        help="Fuse YOLO features with the DINO binary mask and save a feature-response comparison panel.",
    )
    p.add_argument(
        "--dino-mask-fuse-level",
        type=str,
        default=None,
        choices=["p3", "p4", "p5"],
        help="YOLO feature level for DINO-mask fusion. Defaults to --feature-level.",
    )
    p.add_argument(
        "--dino-mask-fuse-bg",
        type=float,
        default=0.5,
        help="Multiplier for YOLO feature regions outside the DINO mask.",
    )
    p.add_argument(
        "--dino-mask-fuse-fg",
        type=float,
        default=1.8,
        help="Multiplier for YOLO feature regions inside the DINO mask.",
    )
    p.add_argument(
        "--head-diagnostic",
        action="store_true",
        help="Save class-score and top-candidate diagnostic panels for the selected YOLO head level.",
    )
    p.add_argument("--head-topk", type=int, default=80, help="Top class-score cells to draw in diagnostics.")
    p.add_argument(
        "--head-score-threshold",
        type=float,
        default=0.05,
        help="Minimum class probability for diagnostic candidate markers.",
    )
    p.add_argument(
        "--spacing-complete",
        action="store_true",
        help="Estimate row spacing from P3 cls candidates and propose missing stomata centers.",
    )
    p.add_argument(
        "--spacing-anchor-source",
        type=str,
        default="none",
        choices=["none", "raw", "mask-fused", "both"],
        help=(
            "Use existing predicted bbox centers as spacing anchors so completion mainly fills gaps "
            "between already-detected stomata."
        ),
    )
    p.add_argument("--spacing-score-threshold", type=float, default=0.05)
    p.add_argument("--spacing-proposal-score-threshold", type=float, default=0.02)
    p.add_argument("--spacing-dino-threshold", type=float, default=0.35)
    p.add_argument("--spacing-row-tolerance-px", type=float, default=28.0)
    p.add_argument("--spacing-nms-distance-px", type=float, default=24.0)
    p.add_argument("--spacing-min-row-seeds", type=int, default=3)
    p.add_argument("--spacing-max-candidates", type=int, default=80)
    p.add_argument(
        "--spacing-period-prior-px",
        type=float,
        default=0.0,
        help="Known horizontal stomata period in this image's pixel coordinates; used when seed-derived period is unreliable.",
    )
    p.add_argument(
        "--spacing-period-prior-ratio",
        type=float,
        default=1.8,
        help="Replace seed-derived period with the prior if it differs by more than this ratio.",
    )
    p.add_argument(
        "--spacing-period-scale",
        type=float,
        default=1.0,
        help="Scale the final horizontal period before proposing candidates. Use <1.0 for denser rice-like spacing.",
    )
    p.add_argument(
        "--spacing-prior-bank-json",
        type=str,
        default=None,
        help="Optional multi-species prior bank JSON. Each prior can define name, image_width, image_height, x_period_px, row_period_px.",
    )
    p.add_argument(
        "--spacing-prior-bank-sigma",
        type=float,
        default=0.45,
        help="Log-period sigma for soft suppression when a row does not match any prior-bank frequency.",
    )
    p.add_argument(
        "--spacing-prior-min-weight",
        type=float,
        default=0.25,
        help="Minimum multiplier for candidates from rows that poorly match the prior bank.",
    )
    p.add_argument(
        "--spacing-row-period-prior-px",
        type=float,
        default=0.0,
        help="Known vertical row gap in this image's pixel coordinates; used to regularize row centers.",
    )
    p.add_argument(
        "--spacing-row-period-prior-ratio",
        type=float,
        default=1.8,
        help="Replace seed-derived row gap with the vertical prior if it differs by more than this ratio.",
    )
    p.add_argument(
        "--spacing-row-period-scale",
        type=float,
        default=1.0,
        help="Scale the final vertical row period. Use <1.0 when rows are denser than the prior dataset.",
    )
    p.add_argument(
        "--spacing-row-prior-tolerance-px",
        type=float,
        default=0.0,
        help="Max distance from a vertical prior row center. 0 = auto from row tolerance / prior.",
    )
    p.add_argument(
        "--spacing-complete-missing-rows",
        action="store_true",
        help="Use the vertical row prior to propose candidates on missing rows with DINO/YOLO evidence.",
    )
    p.add_argument(
        "--spacing-refine-window-px",
        type=float,
        default=0.0,
        help="Search radius around each frequency proposal and move it to the best local YOLO*DINO response. 0 disables refinement.",
    )
    p.add_argument(
        "--spacing-refine-step-px",
        type=float,
        default=4.0,
        help="Pixel step for local proposal refinement.",
    )
    p.add_argument(
        "--spacing-refine-prior-sigma-px",
        type=float,
        default=0.0,
        help="Gaussian sigma that keeps refined points near the frequency proposal. 0 = auto from refine window.",
    )
    p.add_argument(
        "--spacing-draw-pseudo-boxes",
        action="store_true",
        help="Draw estimated bbox/OBB shapes around spacing-completed candidates using nearest YOLO boxes as templates.",
    )
    p.add_argument(
        "--spacing-pseudo-box-width-px",
        type=float,
        default=0.0,
        help="Fallback pseudo bbox width when no YOLO template box is available. 0 = image-size heuristic.",
    )
    p.add_argument(
        "--spacing-pseudo-box-height-px",
        type=float,
        default=0.0,
        help="Fallback pseudo bbox height when no YOLO template box is available. 0 = image-size heuristic.",
    )
    p.add_argument(
        "--spacing-pseudo-overlap-iou",
        type=float,
        default=0.10,
        help="Drop a completed pseudo bbox when it overlaps an existing prediction by this AABB IoU. 0 disables.",
    )
    p.add_argument(
        "--spacing-pseudo-row-frequency-ratio",
        type=float,
        default=1.75,
        help=(
            "Drop completed pseudo boxes from rows whose horizontal period differs from the image's "
            "other rows by more than this ratio. 0 disables."
        ),
    )
    p.add_argument(
        "--spacing-pseudo-row-frequency-min-points",
        type=int,
        default=4,
        help="Minimum existing+completed boxes in a row before row-frequency filtering is applied.",
    )
    p.add_argument(
        "--spacing-pseudo-min-anchor-per-row",
        type=int,
        default=3,
        help=(
            "Drop completed pseudo boxes unless their horizontal row has at least this many "
            "existing YOLO/DINOxYOLO anchor boxes. 0 disables."
        ),
    )
    p.add_argument(
        "--spacing-pseudo-anchor-row-tolerance-px",
        type=float,
        default=0.0,
        help="Y tolerance for matching completed boxes to anchor rows. 0 = use --spacing-row-tolerance-px.",
    )
    return p.parse_args()


def _list_images(image_dir: Path) -> List[Path]:
    return sorted(p for p in image_dir.rglob("*") if p.suffix.lower() in IMAGE_EXTS)


def _image_to_tensor(path: Path, image_size: int) -> torch.Tensor:
    image = Image.open(path).convert("RGB")
    image = image.resize((image_size, image_size), resample=Image.BILINEAR)
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


def _resolve_before_weights(config: Dict, explicit: Optional[str]) -> str:
    weights = explicit or config.get("model", {}).get("student_config", {}).get("weights")
    if not weights:
        raise ValueError("Could not resolve --before-weights and config has no model.student_config.weights")
    path = Path(weights)
    if not path.is_file():
        raise FileNotFoundError(f"Baseline weights not found: {path}")
    return str(path)


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


def _tokens_at_dino_grid(yolo_feat: torch.Tensor, dino_hw: Tuple[int, int]) -> torch.Tensor:
    feat = F.interpolate(
        yolo_feat.unsqueeze(0),
        size=dino_hw,
        mode="bilinear",
        align_corners=False,
    ).squeeze(0)
    tokens, _, _ = _feature_to_tokens(feat.detach().cpu())
    return tokens


def _linear_cka(tokens_a: torch.Tensor, tokens_b: torch.Tensor) -> float:
    x = tokens_a.float().cpu()
    y = tokens_b.float().cpu()
    if x.shape[0] != y.shape[0]:
        raise ValueError(f"Token count mismatch for CKA: {x.shape[0]} vs {y.shape[0]}")
    x = x - x.mean(dim=0, keepdim=True)
    y = y - y.mean(dim=0, keepdim=True)
    xty = x.T @ y
    xtx = x.T @ x
    yty = y.T @ y
    numerator = torch.linalg.matrix_norm(xty, ord="fro").pow(2)
    denominator = torch.linalg.matrix_norm(xtx, ord="fro") * torch.linalg.matrix_norm(yty, ord="fro")
    return float((numerator / denominator.clamp_min(1e-12)).item())


def _feature_pca_image(feat: torch.Tensor) -> Image.Image:
    tokens, h, w = _feature_to_tokens(feat.detach().cpu())
    pca = _fit_pca(tokens)
    return _tokens_to_pca_image(pca, h, w)


def _fit_pca_deterministic(tokens: torch.Tensor, n_components: int = 3) -> torch.Tensor:
    x = tokens.float().cpu()
    if x.numel() == 0:
        return torch.zeros((0, max(int(n_components), 1)), dtype=torch.float32)
    x = x - x.mean(dim=0, keepdim=True)
    requested = max(int(n_components), 1)
    q = int(max(1, min(requested, x.shape[0], x.shape[1])))
    try:
        _, _, vh = torch.linalg.svd(x, full_matrices=False)
        basis = vh[:q].T.contiguous()
    except RuntimeError:
        _, _, v = torch.pca_lowrank(x, q=q, center=False)
        basis = v[:, :q]
    proj = x @ basis
    if q < requested:
        proj = torch.cat([proj, torch.zeros((proj.shape[0], requested - q), dtype=proj.dtype)], dim=1)
    return proj[:, :requested]


def _student_pca_images(before_feat: torch.Tensor, after_feat: torch.Tensor) -> Tuple[Image.Image, Image.Image]:
    before_tokens, before_h, before_w = _feature_to_tokens(before_feat.detach().cpu())
    after_tokens, after_h, after_w = _feature_to_tokens(after_feat.detach().cpu())
    same_shape = (before_h, before_w) == (after_h, after_w)
    same_dim = before_tokens.shape[1] == after_tokens.shape[1]
    if same_shape and same_dim:
        before_pca, after_pca = _fit_joint_pca(before_tokens, after_tokens)
        return (
            _tokens_to_pca_image(before_pca, before_h, before_w),
            _tokens_to_pca_image(after_pca, after_h, after_w),
        )
    return _feature_pca_image(before_feat), _feature_pca_image(after_feat)


def _normalize_01_np(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float32)
    lo = float(x.min())
    hi = float(x.max())
    if hi <= lo:
        return np.zeros_like(x, dtype=np.float32)
    return (x - lo) / (hi - lo)


def _otsu_threshold(values: np.ndarray) -> float:
    values = values.astype(np.float32)
    try:
        from skimage import filters

        return float(filters.threshold_otsu(values))
    except Exception:
        return float(np.percentile(values, 75.0))


def _cleanup_mask(mask: np.ndarray) -> np.ndarray:
    try:
        from skimage import morphology

        cleaned = morphology.remove_small_objects(mask.astype(bool), min_size=2)
        return cleaned.astype(np.uint8)
    except Exception:
        return mask.astype(np.uint8)


def _mask_component_stats(
    mask: np.ndarray,
    *,
    target_coverage: float,
    min_coverage: float,
    max_coverage: float,
) -> Dict[str, float]:
    cleaned = _cleanup_mask(mask)
    area = int(cleaned.sum())
    coverage = float(area / max(cleaned.size, 1))

    num_components = 0
    largest_component_ratio = 1.0
    if area > 0:
        try:
            from skimage import measure

            labeled = measure.label(cleaned.astype(bool), connectivity=1)
            counts = np.bincount(labeled.ravel())[1:]
            num_components = int(counts.size)
            largest_component_ratio = float(counts.max() / max(area, 1)) if counts.size else 1.0
        except Exception:
            num_components = 1
            largest_component_ratio = 1.0

    target = max(float(target_coverage), 1e-6)
    coverage_score = max(0.0, 1.0 - abs(coverage - target) / target)
    if coverage < min_coverage:
        coverage_score -= float(min_coverage - coverage) / max(float(min_coverage), 1e-6)
    if coverage > max_coverage:
        coverage_score -= float(coverage - max_coverage) / max(1.0 - float(max_coverage), 1e-6)
    component_score = min(float(num_components) / 12.0, 1.0)
    score = (2.0 * coverage_score) + (1.5 * component_score) - (1.25 * largest_component_ratio)

    return {
        "coverage": coverage,
        "num_components": float(num_components),
        "largest_component_ratio": largest_component_ratio,
        "score": float(score),
    }


def _overlay_mask(image: Image.Image, mask_alpha: np.ndarray, alpha: float = 0.35) -> Image.Image:
    base = np.asarray(image.convert("RGB"), dtype=np.float32)
    mask = np.clip(mask_alpha.astype(np.float32), 0.0, 1.0)[..., None]
    color = np.zeros_like(base)
    color[:, :, 0] = 255.0
    out = base * (1.0 - alpha * mask) + color * (alpha * mask)
    return Image.fromarray(np.clip(out, 0.0, 255.0).astype(np.uint8), mode="RGB")


def _dino_mask_images(
    dino_feat: torch.Tensor,
    original_image: Image.Image,
    *,
    target_coverage: float,
    min_coverage: float,
    max_coverage: float,
    blur_radius: float,
) -> Tuple[Image.Image, Image.Image, Dict[str, float]]:
    tokens, h, w = _feature_to_tokens(dino_feat.detach().cpu())
    pca = _fit_pca_deterministic(tokens).reshape(h, w, 3).numpy()

    best_mask = np.zeros((h, w), dtype=np.uint8)
    best_stats: Dict[str, float] = {
        "coverage": 0.0,
        "num_components": 0.0,
        "largest_component_ratio": 1.0,
        "score": -1e9,
        "component": 0.0,
        "inverted": 0.0,
        "threshold": 0.0,
    }

    for component_idx in range(3):
        component = _normalize_01_np(pca[:, :, component_idx])
        threshold = _otsu_threshold(component)
        for inverted, candidate in (
            (False, component >= threshold),
            (True, component < threshold),
        ):
            stats = _mask_component_stats(
                candidate.astype(np.uint8),
                target_coverage=target_coverage,
                min_coverage=min_coverage,
                max_coverage=max_coverage,
            )
            if stats["score"] > best_stats["score"]:
                best_mask = _cleanup_mask(candidate.astype(np.uint8))
                best_stats = dict(stats)
                best_stats.update(
                    {
                        "component": float(component_idx + 1),
                        "inverted": float(inverted),
                        "threshold": float(threshold),
                    }
                )

    mask_small = Image.fromarray((best_mask.astype(np.uint8) * 255), mode="L")
    mask_big = mask_small.resize(original_image.size, resample=Image.Resampling.BILINEAR)
    if blur_radius > 0:
        mask_big = mask_big.filter(ImageFilter.GaussianBlur(radius=float(blur_radius)))
    mask_alpha = np.asarray(mask_big, dtype=np.float32) / 255.0
    smooth_binary = (mask_alpha >= 0.35).astype(np.uint8)
    mask_rgb = Image.fromarray(np.stack([smooth_binary * 255] * 3, axis=-1).astype(np.uint8), mode="RGB")
    overlay = _overlay_mask(original_image, mask_alpha)

    best_stats["display_coverage"] = float(smooth_binary.mean())
    return mask_rgb, overlay, best_stats


def _robust_normalize(values: np.ndarray, low_pct: float = 2.0, high_pct: float = 98.0) -> np.ndarray:
    values = values.astype(np.float32)
    lo = float(np.percentile(values, low_pct))
    hi = float(np.percentile(values, high_pct))
    if hi <= lo:
        hi = lo + 1e-6
    return np.clip((values - lo) / (hi - lo), 0.0, 1.0)


def _heatmap_image(values: np.ndarray, size: Tuple[int, int], cmap_name: str = "turbo") -> Image.Image:
    norm = _robust_normalize(values)
    try:
        from matplotlib import colormaps

        rgb = colormaps[cmap_name](norm)[..., :3]
    except Exception:
        rgb = np.stack([norm, norm, norm], axis=-1)
    image = Image.fromarray((rgb * 255.0).astype(np.uint8), mode="RGB")
    return image.resize(size, resample=Image.Resampling.BICUBIC)


def _feature_response_image(feat: torch.Tensor, size: Tuple[int, int]) -> Image.Image:
    response = feat.detach().float().cpu().pow(2).mean(dim=0).sqrt().numpy()
    return _heatmap_image(response, size=size, cmap_name="turbo")


def _feature_delta_image(before_feat: torch.Tensor, after_feat: torch.Tensor, size: Tuple[int, int]) -> Image.Image:
    before = before_feat.detach().float().cpu().pow(2).mean(dim=0).sqrt().numpy()
    after = after_feat.detach().float().cpu().pow(2).mean(dim=0).sqrt().numpy()
    delta = after - before
    scale = float(np.percentile(np.abs(delta), 98.0))
    if scale <= 0:
        scale = 1e-6
    norm = np.clip(delta / scale, -1.0, 1.0)
    rgb = np.zeros((*norm.shape, 3), dtype=np.float32)
    rgb[..., 0] = np.clip(norm, 0.0, 1.0)
    rgb[..., 2] = np.clip(-norm, 0.0, 1.0)
    rgb[..., 1] = 1.0 - np.abs(norm)
    image = Image.fromarray((rgb * 255.0).astype(np.uint8), mode="RGB")
    return image.resize(size, resample=Image.Resampling.BICUBIC)


def _delta_heatmap_image(before: torch.Tensor, after: torch.Tensor, size: Tuple[int, int]) -> Image.Image:
    delta = after.detach().float().cpu().numpy() - before.detach().float().cpu().numpy()
    scale = float(np.percentile(np.abs(delta), 98.0))
    if scale <= 0:
        scale = 1e-6
    norm = np.clip(delta / scale, -1.0, 1.0)
    rgb = np.zeros((*norm.shape, 3), dtype=np.float32)
    rgb[..., 0] = np.clip(norm, 0.0, 1.0)
    rgb[..., 2] = np.clip(-norm, 0.0, 1.0)
    rgb[..., 1] = 1.0 - np.abs(norm)
    image = Image.fromarray((rgb * 255.0).astype(np.uint8), mode="RGB")
    return image.resize(size, resample=Image.Resampling.BICUBIC)


def _annotation_font(size: int) -> ImageFont.ImageFont:
    try:
        return ImageFont.truetype("DejaVuSans-Bold.ttf", size=max(8, int(size)))
    except OSError:
        return ImageFont.load_default()


def _draw_text_with_bg(
    draw: ImageDraw.ImageDraw,
    xy: Tuple[float, float],
    text: str,
    *,
    fill: Tuple[int, int, int],
    font: ImageFont.ImageFont,
) -> None:
    padding = max(2, int(round(getattr(font, "size", 10) / 7.0)))
    bbox = draw.textbbox(xy, text, font=font)
    draw.rectangle(
        (
            bbox[0] - padding,
            bbox[1] - padding,
            bbox[2] + padding,
            bbox[3] + padding,
        ),
        fill=(0, 0, 0),
    )
    draw.text(xy, text, fill=fill, font=font)


def _head_cls_score_map(
    detect_module: torch.nn.Module,
    feat: torch.Tensor,
    level_idx: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    cls_head = getattr(detect_module, "cv3", None)
    if bool(getattr(detect_module, "end2end", False)) and hasattr(detect_module, "one2one_cv3"):
        cls_head = getattr(detect_module, "one2one_cv3")
    if cls_head is None:
        raise RuntimeError("Detect module does not expose a classification head")

    with torch.no_grad():
        logits = cls_head[level_idx](feat.unsqueeze(0))
        probs = logits.sigmoid().squeeze(0)
        scores, cls_ids = probs.max(dim=0)
    return scores.detach().cpu(), cls_ids.detach().cpu()


def _draw_score_candidates(
    image: Image.Image,
    score_map: torch.Tensor,
    color: Tuple[int, int, int],
    *,
    topk: int,
    threshold: float,
    label_prefix: str,
    annotation_scale: float = 1.0,
    max_labels: int = 12,
) -> Image.Image:
    canvas = image.convert("RGB").copy()
    draw = ImageDraw.Draw(canvas)
    scores = score_map.detach().float().cpu()
    if scores.numel() == 0:
        return canvas

    local_max = scores == F.max_pool2d(
        scores.unsqueeze(0).unsqueeze(0),
        kernel_size=3,
        stride=1,
        padding=1,
    ).squeeze(0).squeeze(0)
    valid = local_max & (scores >= float(threshold))
    flat_scores = scores.flatten()
    flat_valid = valid.flatten()
    valid_idx = torch.nonzero(flat_valid, as_tuple=False).flatten()
    if valid_idx.numel() == 0:
        return canvas

    k = min(int(topk), int(valid_idx.numel()))
    top_order = torch.topk(flat_scores[valid_idx], k=k).indices
    selected = valid_idx[top_order]

    h, w = scores.shape
    img_w, img_h = canvas.size
    scale = max(float(annotation_scale), 0.1)
    radius = max(4, int(round(max(canvas.size) / 220.0 * scale)))
    width = max(2, int(round(max(canvas.size) / 360.0 * scale)))
    font = _annotation_font(int(round(max(canvas.size) / 90.0 * scale)))
    for rank, flat_idx in enumerate(selected.tolist()):
        y = flat_idx // w
        x = flat_idx % w
        cx = (float(x) + 0.5) / float(w) * float(img_w)
        cy = (float(y) + 0.5) / float(h) * float(img_h)
        draw.ellipse(
            (cx - radius - width, cy - radius - width, cx + radius + width, cy + radius + width),
            outline=(0, 0, 0),
            width=max(1, width),
        )
        draw.ellipse(
            (cx - radius, cy - radius, cx + radius, cy + radius),
            outline=color,
            width=width,
        )
        if rank < int(max_labels):
            _draw_text_with_bg(
                draw,
                (cx + radius + width + 1, cy - radius),
                f"{label_prefix}{float(scores[y, x]):.2f}",
                fill=color,
                font=font,
            )
    return canvas


def _score_stats(score_map: torch.Tensor, threshold: float) -> Dict[str, float]:
    scores = score_map.detach().float().cpu()
    return {
        "max": float(scores.max().item()) if scores.numel() else 0.0,
        "mean": float(scores.mean().item()) if scores.numel() else 0.0,
        "cells_ge_threshold": float((scores >= float(threshold)).sum().item()) if scores.numel() else 0.0,
    }


def _local_max_points(
    score_map: torch.Tensor,
    *,
    image_size: Tuple[int, int],
    threshold: float,
    topk: int,
    gate: Optional[torch.Tensor] = None,
    gate_threshold: float = 0.0,
) -> np.ndarray:
    scores = score_map.detach().float().cpu()
    if scores.numel() == 0:
        return np.zeros((0, 3), dtype=np.float32)
    valid = scores >= float(threshold)
    if gate is not None:
        gate_cpu = gate.detach().float().cpu()
        if gate_cpu.ndim == 3:
            gate_cpu = gate_cpu.squeeze(0)
        if tuple(gate_cpu.shape) != tuple(scores.shape):
            gate_cpu = F.interpolate(
                gate_cpu.unsqueeze(0).unsqueeze(0),
                size=scores.shape,
                mode="bilinear",
                align_corners=False,
            ).squeeze(0).squeeze(0)
        valid = valid & (gate_cpu >= float(gate_threshold))

    local_max = scores == F.max_pool2d(
        scores.unsqueeze(0).unsqueeze(0),
        kernel_size=3,
        stride=1,
        padding=1,
    ).squeeze(0).squeeze(0)
    valid = valid & local_max
    idx = torch.nonzero(valid.flatten(), as_tuple=False).flatten()
    if idx.numel() == 0:
        return np.zeros((0, 3), dtype=np.float32)

    k = min(int(topk), int(idx.numel()))
    selected = idx[torch.topk(scores.flatten()[idx], k=k).indices]
    h, w = scores.shape
    img_w, img_h = image_size
    points: List[Tuple[float, float, float]] = []
    for flat_idx in selected.tolist():
        gy = flat_idx // w
        gx = flat_idx % w
        x = (float(gx) + 0.5) / float(w) * float(img_w)
        y = (float(gy) + 0.5) / float(h) * float(img_h)
        points.append((x, y, float(scores[gy, gx].item())))
    return np.asarray(points, dtype=np.float32)


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


def _estimate_spacing_px(xs: np.ndarray) -> float:
    xs = np.sort(xs.astype(np.float32))
    if xs.size < 2:
        return 0.0
    diffs = np.diff(xs)
    diffs = diffs[diffs > 1.0]
    if diffs.size == 0:
        return 0.0
    rough = float(np.median(diffs))
    keep = diffs[(diffs >= 0.45 * rough) & (diffs <= 1.65 * rough)]
    if keep.size:
        return float(np.median(keep))
    return rough


def _estimate_spacing_phase(xs: np.ndarray, period: float) -> float:
    if xs.size == 0 or period <= 1e-6:
        return 0.0
    phases = np.mod(xs.astype(np.float32), period)
    angles = 2.0 * np.pi * phases / period
    mean_vec = np.mean(np.exp(1j * angles))
    if abs(mean_vec) < 1e-6:
        return float(np.median(phases))
    angle = float(np.angle(mean_vec))
    if angle < 0.0:
        angle += 2.0 * np.pi
    return float(period * angle / (2.0 * np.pi))


def _maybe_use_period_prior(
    estimated: float,
    prior: float,
    ratio: float,
) -> Tuple[float, bool]:
    prior = max(float(prior), 0.0)
    if prior <= 0.0:
        return float(estimated), False
    ratio = max(float(ratio), 1.01)
    if estimated < 3.0 or estimated > prior * ratio or estimated < prior / ratio:
        return prior, True
    return float(estimated), False


def _load_spacing_prior_bank(path: Optional[str]) -> List[Dict[str, float | str]]:
    if not path:
        return []
    prior_path = Path(path)
    if not prior_path.is_file():
        raise FileNotFoundError(f"Spacing prior bank not found: {prior_path}")
    with prior_path.open("r", encoding="utf-8") as f:
        raw = json.load(f)
    entries = raw.get("priors", raw) if isinstance(raw, dict) else raw
    if not isinstance(entries, list):
        raise ValueError("Spacing prior bank JSON must be a list or contain a 'priors' list")

    priors: List[Dict[str, float | str]] = []
    for idx, entry in enumerate(entries):
        if not isinstance(entry, dict):
            continue
        name = str(entry.get("name", f"prior_{idx}"))
        ref_w = float(entry.get("image_width", entry.get("reference_width", 0.0)) or 0.0)
        ref_h = float(entry.get("image_height", entry.get("reference_height", 0.0)) or 0.0)
        x_period = float(
            entry.get("x_period_px", entry.get("horizontal_period_px", entry.get("period_px", 0.0))) or 0.0
        )
        y_period = float(
            entry.get("row_period_px", entry.get("y_period_px", entry.get("vertical_period_px", 0.0))) or 0.0
        )
        x_ratio = float(entry.get("x_period_ratio", 0.0) or 0.0)
        y_ratio = float(entry.get("row_period_ratio", entry.get("y_period_ratio", 0.0)) or 0.0)
        if x_ratio <= 0.0 and ref_w > 0.0 and x_period > 0.0:
            x_ratio = x_period / ref_w
        if y_ratio <= 0.0 and ref_h > 0.0 and y_period > 0.0:
            y_ratio = y_period / ref_h
        if x_ratio <= 0.0 and y_ratio <= 0.0:
            continue
        priors.append({"name": name, "x_ratio": x_ratio, "y_ratio": y_ratio})
    return priors


def _scaled_prior_period(
    prior: Dict[str, float | str],
    axis: str,
    image_size: Tuple[int, int],
) -> float:
    img_w, img_h = image_size
    if axis == "x":
        return float(prior.get("x_ratio", 0.0)) * float(img_w)
    return float(prior.get("y_ratio", 0.0)) * float(img_h)


def _nearest_prior_match(
    period_px: float,
    prior_bank: Sequence[Dict[str, float | str]],
    *,
    axis: str,
    image_size: Tuple[int, int],
    sigma: float,
    min_weight: float,
) -> Tuple[float, str, float, float]:
    period = float(period_px)
    best_period = 0.0
    best_name = ""
    best_mismatch = float("inf")
    for prior in prior_bank:
        prior_period = _scaled_prior_period(prior, axis, image_size)
        if prior_period <= 0.0:
            continue
        if period > 0.0:
            mismatch = abs(float(np.log(max(period, 1e-6) / max(prior_period, 1e-6))))
        else:
            mismatch = 0.0
        if mismatch < best_mismatch:
            best_mismatch = mismatch
            best_period = float(prior_period)
            best_name = str(prior.get("name", ""))
    if best_period <= 0.0:
        return 0.0, "", 1.0, 0.0
    if not np.isfinite(best_mismatch):
        best_mismatch = 0.0
    sigma = max(float(sigma), 1e-6)
    weight = float(np.exp(-0.5 * (best_mismatch / sigma) ** 2))
    weight = float(np.clip(weight, float(min_weight), 1.0))
    return best_period, best_name, weight, float(best_mismatch)


def _sample_grid_value(map_2d: torch.Tensor, x_px: float, y_px: float, image_size: Tuple[int, int]) -> float:
    values = map_2d.detach().float().cpu()
    if values.ndim == 3:
        values = values.squeeze(0)
    h, w = values.shape
    img_w, img_h = image_size
    gx = int(np.clip(round((float(x_px) / max(float(img_w), 1.0)) * w - 0.5), 0, w - 1))
    gy = int(np.clip(round((float(y_px) / max(float(img_h), 1.0)) * h - 0.5), 0, h - 1))
    return float(values[gy, gx].item())


def _merge_spacing_anchor_points(
    seed_points: np.ndarray,
    anchor_points: Optional[np.ndarray],
    *,
    nms_distance_px: float,
) -> Tuple[np.ndarray, int]:
    if anchor_points is None or anchor_points.size == 0:
        return seed_points, 0
    anchors = np.asarray(anchor_points, dtype=np.float32)
    if anchors.ndim != 2 or anchors.shape[1] < 2:
        return seed_points, 0
    if anchors.shape[1] == 2:
        anchors = np.concatenate(
            [anchors[:, :2], np.ones((anchors.shape[0], 1), dtype=np.float32)],
            axis=1,
        )
    anchors = anchors[:, :3].astype(np.float32)

    merged: List[np.ndarray] = [anchor for anchor in anchors]
    nms = max(float(nms_distance_px), 1.0)
    for seed in seed_points.tolist() if seed_points.size else []:
        seed_arr = np.asarray(seed[:3], dtype=np.float32)
        if merged and min(float(np.linalg.norm(seed_arr[:2] - point[:2])) for point in merged) <= nms:
            continue
        merged.append(seed_arr)
    if not merged:
        return np.zeros((0, 3), dtype=np.float32), 0
    return np.stack(merged, axis=0).astype(np.float32), int(anchors.shape[0])


def _refine_spacing_candidate(
    score_map: torch.Tensor,
    gate: torch.Tensor,
    x_px: float,
    y_px: float,
    *,
    image_size: Tuple[int, int],
    proposal_score_threshold: float,
    gate_threshold: float,
    refine_window_px: float,
    refine_step_px: float,
    refine_prior_sigma_px: float,
) -> Optional[Tuple[float, float, float, float]]:
    base_score = _sample_grid_value(score_map, x_px, y_px, image_size)
    base_mask = _sample_grid_value(gate, x_px, y_px, image_size)
    if refine_window_px <= 0.0:
        if base_score < float(proposal_score_threshold) or base_mask < float(gate_threshold):
            return None
        return float(x_px), float(y_px), float(base_score), float(base_mask)

    img_w, img_h = image_size
    window = max(float(refine_window_px), 0.0)
    step = max(float(refine_step_px), 1.0)
    sigma = float(refine_prior_sigma_px) if refine_prior_sigma_px > 0.0 else max(window * 0.55, step)
    offsets = np.arange(-window, window + 0.5 * step, step, dtype=np.float32)
    best: Optional[Tuple[float, float, float, float, float]] = None

    for dy in offsets:
        yy = float(y_px + dy)
        if yy < 0.0 or yy >= float(img_h):
            continue
        for dx in offsets:
            xx = float(x_px + dx)
            if xx < 0.0 or xx >= float(img_w):
                continue
            score = _sample_grid_value(score_map, xx, yy, image_size)
            mask_value = _sample_grid_value(gate, xx, yy, image_size)
            if score < float(proposal_score_threshold) or mask_value < float(gate_threshold):
                continue
            dist2 = float(dx * dx + dy * dy)
            prior_weight = float(np.exp(-0.5 * dist2 / max(sigma * sigma, 1e-6)))
            combined = float(score) * float(mask_value) * prior_weight
            if best is None or combined > best[0]:
                best = (combined, xx, yy, float(score), float(mask_value))

    if best is None:
        return None
    _combined, best_x, best_y, best_score, best_mask = best
    return best_x, best_y, best_score, best_mask


def _spacing_complete_candidates(
    score_map: torch.Tensor,
    gate: torch.Tensor,
    *,
    image_size: Tuple[int, int],
    score_threshold: float,
    proposal_score_threshold: float,
    gate_threshold: float,
    row_tolerance_px: float,
    nms_distance_px: float,
    min_row_seeds: int,
    max_candidates: int,
    period_prior_px: float = 0.0,
    period_prior_ratio: float = 1.8,
    period_scale: float = 1.0,
    prior_bank: Optional[Sequence[Dict[str, float | str]]] = None,
    prior_bank_sigma: float = 0.45,
    prior_min_weight: float = 0.25,
    row_period_prior_px: float = 0.0,
    row_period_prior_ratio: float = 1.8,
    row_period_scale: float = 1.0,
    row_prior_tolerance_px: float = 0.0,
    complete_missing_rows: bool = False,
    refine_window_px: float = 0.0,
    refine_step_px: float = 4.0,
    refine_prior_sigma_px: float = 0.0,
    anchor_points: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, float], np.ndarray]:
    seed_points = _local_max_points(
        score_map,
        image_size=image_size,
        threshold=score_threshold,
        topk=max(int(max_candidates) * 4, 64),
        gate=gate,
        gate_threshold=gate_threshold,
    )
    seed_points, anchor_count = _merge_spacing_anchor_points(
        seed_points,
        anchor_points,
        nms_distance_px=nms_distance_px,
    )
    rows = _group_points_by_y_px(seed_points, row_tolerance_px)
    proposals: List[Tuple[float, float, float, float]] = []
    periods: List[float] = []
    row_lines: List[Tuple[float, float]] = []
    row_templates: List[Dict[str, object]] = []
    used_rows = 0
    prior_used_rows = 0
    row_aligned_count = 0
    missing_row_count = 0
    missing_row_candidate_count = 0
    img_w, img_h = image_size
    nms = max(float(nms_distance_px), 1.0)
    period_scale = max(float(period_scale), 0.05)
    row_period_scale = max(float(row_period_scale), 0.05)
    prior_bank = list(prior_bank or [])
    prior_ratio = max(float(period_prior_ratio), 1.01)
    row_prior_ratio = max(float(row_period_prior_ratio), 1.01)
    bank_used_rows = 0
    bank_weights: List[float] = []
    row_centers = np.asarray([float(np.median(row[:, 1])) for row in rows], dtype=np.float32)
    row_period_est = _estimate_spacing_px(row_centers) if row_centers.size >= 2 else 0.0
    if prior_bank:
        bank_row_period, _bank_row_name, bank_row_weight, bank_row_mismatch = _nearest_prior_match(
            row_period_est,
            prior_bank,
            axis="y",
            image_size=image_size,
            sigma=prior_bank_sigma,
            min_weight=prior_min_weight,
        )
        if row_period_est >= 3.0 and bank_row_mismatch <= float(np.log(row_prior_ratio)):
            row_period = row_period_est
            row_prior_used = False
        else:
            row_period = bank_row_period if bank_row_period > 0.0 else row_period_est
            row_prior_used = bank_row_period > 0.0
    else:
        row_period, row_prior_used = _maybe_use_period_prior(
            row_period_est,
            row_period_prior_px,
            row_period_prior_ratio,
        )
    if row_period >= 3.0:
        row_period *= row_period_scale
    row_prior_enabled = bool(prior_bank) or float(row_period_prior_px) > 0.0 or bool(complete_missing_rows)
    row_phase = _estimate_spacing_phase(row_centers, row_period) if row_period >= 3.0 else 0.0
    auto_row_tol = max(float(row_tolerance_px), float(row_period) * 0.35 if row_period >= 3.0 else 0.0, 1.0)
    row_prior_tol = max(float(row_prior_tolerance_px), auto_row_tol) if float(row_prior_tolerance_px) > 0 else auto_row_tol

    def align_y_to_row_prior(row_y: float) -> Tuple[float, bool]:
        if not row_prior_enabled or row_period < 3.0 or row_centers.size == 0:
            return row_y, False
        k = int(round((float(row_y) - row_phase) / row_period))
        aligned_y = row_phase + k * row_period
        if 0.0 <= aligned_y < float(img_h) and abs(aligned_y - float(row_y)) <= row_prior_tol:
            return float(aligned_y), True
        return row_y, False

    def add_row_candidates(
        *,
        row_y: float,
        seed_x: np.ndarray,
        period: float,
        phase: float,
        x_min: float,
        x_max: float,
        require_seed_distance: bool,
        prior_weight: float,
    ) -> int:
        if period < 3.0:
            return 0
        k0 = int(np.floor((float(x_min) - phase) / period))
        k1 = int(np.ceil((float(x_max) - phase) / period))
        added = 0
        for k in range(k0, k1 + 1):
            x = phase + k * period
            if x < 0.0 or x >= float(img_w):
                continue
            if require_seed_distance and seed_x.size and np.min(np.abs(seed_x - x)) <= nms:
                continue
            if seed_points.size and np.min(np.linalg.norm(seed_points[:, :2] - np.asarray([x, row_y]), axis=1)) <= nms:
                continue
            refined = _refine_spacing_candidate(
                score_map,
                gate,
                x,
                row_y,
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
            score = float(score) * float(prior_weight)
            if seed_points.size and np.min(
                np.linalg.norm(seed_points[:, :2] - np.asarray([refined_x, refined_y]), axis=1)
            ) <= nms:
                continue
            proposals.append((float(refined_x), float(refined_y), float(score), float(mask_value)))
            added += 1
        return added

    for row in rows:
        if row.shape[0] < int(min_row_seeds):
            continue
        raw_row_y = float(np.median(row[:, 1]))
        row_y, row_aligned = align_y_to_row_prior(raw_row_y)
        if row_aligned:
            row_aligned_count += 1
            row_lines.append((row_y, 1.0))
        else:
            row_lines.append((row_y, 0.0))
        seed_x = np.sort(row[:, 0])
        period_est = _estimate_spacing_px(seed_x)
        row_prior_weight = 1.0
        if prior_bank:
            bank_period, _bank_name, bank_weight, bank_mismatch = _nearest_prior_match(
                period_est,
                prior_bank,
                axis="x",
                image_size=image_size,
                sigma=prior_bank_sigma,
                min_weight=prior_min_weight,
            )
            bank_weights.append(bank_weight)
            row_prior_weight = bank_weight
            if period_est >= 3.0 and bank_mismatch <= float(np.log(prior_ratio)):
                period = period_est
                used_period_prior = False
            else:
                period = bank_period if bank_period > 0.0 else period_est
                used_period_prior = bank_period > 0.0
            if used_period_prior:
                bank_used_rows += 1
        else:
            period, used_period_prior = _maybe_use_period_prior(
                period_est,
                period_prior_px,
                period_prior_ratio,
            )
        if used_period_prior:
            prior_used_rows += 1
        if period >= 3.0:
            period *= period_scale
        if period < 3.0:
            continue
        phase = _estimate_spacing_phase(seed_x, period)
        x_min = max(0.0, float(seed_x.min()) - period)
        x_max = min(float(img_w), float(seed_x.max()) + period)
        used_rows += 1
        periods.append(period)
        row_templates.append(
            {
                "row_y": row_y,
                "seed_x": seed_x,
                "period": period,
                "phase": phase,
                "x_min": x_min,
                "x_max": x_max,
                "prior_weight": row_prior_weight,
            }
        )
        add_row_candidates(
            row_y=row_y,
            seed_x=seed_x,
            period=period,
            phase=phase,
            x_min=x_min,
            x_max=x_max,
            require_seed_distance=True,
            prior_weight=row_prior_weight,
        )

    if complete_missing_rows and row_period >= 3.0 and row_templates:
        template_ys = np.asarray([float(t["row_y"]) for t in row_templates], dtype=np.float32)
        y_min = max(0.0, float(template_ys.min()) - row_period)
        y_max = min(float(img_h), float(template_ys.max()) + row_period)
        k0 = int(np.floor((y_min - row_phase) / row_period))
        k1 = int(np.ceil((y_max - row_phase) / row_period))
        for k in range(k0, k1 + 1):
            row_y = row_phase + k * row_period
            if row_y < 0.0 or row_y >= float(img_h):
                continue
            if np.min(np.abs(template_ys - row_y)) <= row_prior_tol:
                continue
            nearest_idx = int(np.argmin(np.abs(template_ys - row_y)))
            template = row_templates[nearest_idx]
            row_lines.append((float(row_y), 2.0))
            missing_row_count += 1
            missing_row_candidate_count += add_row_candidates(
                row_y=float(row_y),
                seed_x=np.asarray([], dtype=np.float32),
                period=float(template["period"]),
                phase=float(template["phase"]),
                x_min=float(template["x_min"]),
                x_max=float(template["x_max"]),
                require_seed_distance=False,
                prior_weight=float(template.get("prior_weight", 1.0)),
            )

    if not proposals:
        stats = {
            "spacing_seed_count": float(seed_points.shape[0]),
            "spacing_anchor_count": float(anchor_count),
            "spacing_row_count": float(len(rows)),
            "spacing_used_rows": float(used_rows),
            "spacing_prior_used_rows": float(prior_used_rows),
            "spacing_row_period_px": float(row_period),
            "spacing_row_prior_used": float(row_prior_used),
            "spacing_row_aligned_count": float(row_aligned_count),
            "spacing_missing_row_count": float(missing_row_count),
            "spacing_missing_row_candidate_count": float(missing_row_candidate_count),
            "spacing_prior_bank_used_rows": float(bank_used_rows),
            "spacing_prior_bank_mean_weight": float(np.mean(bank_weights)) if bank_weights else 1.0,
            "spacing_mean_period_px": float(np.mean(periods)) if periods else 0.0,
            "spacing_candidate_count": 0.0,
        }
        return (
            seed_points,
            np.zeros((0, 4), dtype=np.float32),
            stats,
            np.asarray(row_lines, dtype=np.float32) if row_lines else np.zeros((0, 2), dtype=np.float32),
        )

    proposals_arr = np.asarray(proposals, dtype=np.float32)
    order = np.argsort(-(proposals_arr[:, 2] * proposals_arr[:, 3]))
    selected: List[np.ndarray] = []
    for idx in order:
        cand = proposals_arr[idx]
        if any(np.linalg.norm(cand[:2] - prev[:2]) <= nms for prev in selected):
            continue
        selected.append(cand)
        if len(selected) >= int(max_candidates):
            break
    selected_arr = np.asarray(selected, dtype=np.float32) if selected else np.zeros((0, 4), dtype=np.float32)
    stats = {
        "spacing_seed_count": float(seed_points.shape[0]),
        "spacing_anchor_count": float(anchor_count),
        "spacing_row_count": float(len(rows)),
        "spacing_used_rows": float(used_rows),
        "spacing_prior_used_rows": float(prior_used_rows),
        "spacing_row_period_px": float(row_period),
        "spacing_row_prior_used": float(row_prior_used),
        "spacing_row_aligned_count": float(row_aligned_count),
        "spacing_missing_row_count": float(missing_row_count),
        "spacing_missing_row_candidate_count": float(missing_row_candidate_count),
        "spacing_prior_bank_used_rows": float(bank_used_rows),
        "spacing_prior_bank_mean_weight": float(np.mean(bank_weights)) if bank_weights else 1.0,
        "spacing_mean_period_px": float(np.mean(periods)) if periods else 0.0,
        "spacing_candidate_count": float(selected_arr.shape[0]),
    }
    return (
        seed_points,
        selected_arr,
        stats,
        np.asarray(row_lines, dtype=np.float32) if row_lines else np.zeros((0, 2), dtype=np.float32),
    )


def _draw_spacing_points(
    image: Image.Image,
    seed_points: np.ndarray,
    proposals: np.ndarray,
    *,
    row_lines: Optional[np.ndarray] = None,
    pseudo_boxes: Optional[Sequence[Dict[str, object]]] = None,
    annotation_scale: float = 1.0,
    max_labels: int = 999999,
) -> Image.Image:
    canvas = image.convert("RGB").copy()
    draw = ImageDraw.Draw(canvas)
    scale = max(float(annotation_scale), 0.1)
    radius_seed = max(3, int(round(max(canvas.size) / 320.0 * scale)))
    radius_prop = max(6, int(round(max(canvas.size) / 220.0 * scale)))
    width = max(2, int(round(max(canvas.size) / 360.0 * scale)))
    font = _annotation_font(int(round(max(canvas.size) / 95.0 * scale)))
    if row_lines is not None and row_lines.size:
        for y, kind in row_lines.tolist():
            if kind >= 2.0:
                color = (80, 190, 255)
                line_width = max(1, width)
            elif kind >= 1.0:
                color = (80, 120, 255)
                line_width = max(1, width // 2)
            else:
                continue
            draw.line((0, y, canvas.width, y), fill=(0, 0, 0), width=max(1, line_width + width))
            draw.line((0, y, canvas.width, y), fill=color, width=line_width)
    if pseudo_boxes:
        box_color = (80, 220, 255)
        for pred in pseudo_boxes:
            poly = np.asarray(pred["poly"], dtype=np.float32).reshape(4, 2)
            points = [tuple(p) for p in poly.tolist()]
            draw.line(points + [points[0]], fill=(0, 0, 0), width=max(1, width * 2))
            draw.line(points + [points[0]], fill=box_color, width=max(1, width))
    for x, y, score in seed_points[:, :3].tolist() if seed_points.size else []:
        draw.ellipse(
            (x - radius_seed - width, y - radius_seed - width, x + radius_seed + width, y + radius_seed + width),
            outline=(0, 0, 0),
            width=max(1, width),
        )
        draw.ellipse((x - radius_seed, y - radius_seed, x + radius_seed, y + radius_seed), outline=(30, 170, 60), width=width)
    for idx, (x, y, score, mask_value) in enumerate(proposals.tolist() if proposals.size else [], start=1):
        for offset_width, color in ((width * 3, (0, 0, 0)), (width, (255, 220, 30))):
            draw.line((x - radius_prop, y, x + radius_prop, y), fill=color, width=max(1, offset_width))
            draw.line((x, y - radius_prop, x, y + radius_prop), fill=color, width=max(1, offset_width))
        if idx <= int(max_labels):
            _draw_text_with_bg(
                draw,
                (x + radius_prop + width + 1, y - radius_prop),
                f"M{idx}:{float(score):.3g}",
                fill=(255, 220, 30),
                font=font,
            )
    return canvas


def _spacing_prior_image(
    proposals: np.ndarray,
    image_size: Tuple[int, int],
    blur_radius: float = 10.0,
    annotation_scale: float = 1.0,
) -> Image.Image:
    canvas = Image.new("L", image_size, color=0)
    draw = ImageDraw.Draw(canvas)
    radius = max(4, int(round(max(image_size) / 180.0 * max(float(annotation_scale), 0.1))))
    for x, y, score, mask_value in proposals.tolist() if proposals.size else []:
        intensity = int(np.clip(120 + 135 * float(score) * float(mask_value), 0, 255))
        draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=intensity)
    if blur_radius > 0:
        canvas = canvas.filter(ImageFilter.GaussianBlur(radius=float(blur_radius)))
    return _heatmap_image(np.asarray(canvas, dtype=np.float32), image_size, cmap_name="viridis")


def _dino_gate_tensor(
    dino_feat: torch.Tensor,
    *,
    target_coverage: float,
    min_coverage: float,
    max_coverage: float,
) -> torch.Tensor:
    gates: List[torch.Tensor] = []
    for sample in dino_feat.detach().cpu():
        tokens, h, w = _feature_to_tokens(sample)
        pca = _fit_pca_deterministic(tokens).reshape(h, w, 3).numpy()

        best_mask = np.zeros((h, w), dtype=np.uint8)
        best_score = -1e9
        for component_idx in range(3):
            component = _normalize_01_np(pca[:, :, component_idx])
            threshold = _otsu_threshold(component)
            for candidate in (component >= threshold, component < threshold):
                stats = _mask_component_stats(
                    candidate.astype(np.uint8),
                    target_coverage=target_coverage,
                    min_coverage=min_coverage,
                    max_coverage=max_coverage,
                )
                if float(stats["score"]) > best_score:
                    best_score = float(stats["score"])
                    best_mask = _cleanup_mask(candidate.astype(np.uint8))

        gates.append(torch.from_numpy(best_mask.astype(np.float32)).unsqueeze(0))

    return torch.stack(gates, dim=0)


def _apply_dino_mask_fuse_to_feature(
    yolo_feat: torch.Tensor,
    dino_feat: torch.Tensor,
    *,
    bg_multiplier: float,
    fg_multiplier: float,
    target_coverage: float,
    min_coverage: float,
    max_coverage: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    gate = _dino_gate_tensor(
        dino_feat.unsqueeze(0),
        target_coverage=target_coverage,
        min_coverage=min_coverage,
        max_coverage=max_coverage,
    ).to(device=yolo_feat.device, dtype=yolo_feat.dtype)
    gate = F.interpolate(
        gate,
        size=yolo_feat.shape[-2:],
        mode="bilinear",
        align_corners=False,
    ).squeeze(0)
    scale = float(bg_multiplier) + (float(fg_multiplier) - float(bg_multiplier)) * gate
    return yolo_feat * scale, gate


def _find_detect_module(det_model: torch.nn.Module) -> torch.nn.Module:
    try:
        from ultralytics.nn.modules.head import Detect
    except Exception:
        Detect = None

    model_seq = getattr(det_model, "model", None)
    if model_seq is None:
        raise RuntimeError("Could not inspect Ultralytics model modules")

    for module in reversed(model_seq):
        if Detect is not None and isinstance(module, Detect):
            return module
        if module.__class__.__name__.lower() in {"detect", "obb"}:
            return module
    raise RuntimeError("Could not locate Detect/OBB module in YOLO model")


def _predict_one_dino_gated(
    yolo_model: object,
    dino: DinoFeatureExtractor,
    image_path: str,
    *,
    imgsz: int,
    conf: float,
    iou: float,
    max_det: int,
    device: torch.device,
    feature_level: str,
    gate_min: float,
    gate_max: float,
    target_coverage: float,
    min_coverage: float,
    max_coverage: float,
) -> List[Dict[str, object]]:
    det_model = yolo_model.model
    detect_module = _find_detect_module(det_model)
    level_to_idx = {"p3": 0, "p4": 1, "p5": 2}
    level_idx = level_to_idx[feature_level]
    state: Dict[str, Optional[torch.Tensor]] = {"gate": None}

    first_module = det_model.model[0]

    def _cache_gate(_module, inputs):
        if not inputs:
            state["gate"] = None
            return None
        images = inputs[0].detach().float()
        with torch.no_grad():
            feat = dino(images.to(device)).detach()
            gate = _dino_gate_tensor(
                feat,
                target_coverage=target_coverage,
                min_coverage=min_coverage,
                max_coverage=max_coverage,
            )
        state["gate"] = gate.to(device)
        return None

    def _gate_detect_input(_module, inputs):
        if not inputs or state.get("gate") is None:
            return None
        x = inputs[0]
        if not isinstance(x, (list, tuple)) or len(x) <= level_idx:
            return None

        feats = list(x)
        target_feat = feats[level_idx]
        gate = state["gate"].to(device=target_feat.device, dtype=target_feat.dtype)
        if gate.shape[0] != target_feat.shape[0]:
            return None

        gate_resized = F.interpolate(
            gate,
            size=target_feat.shape[-2:],
            mode="bilinear",
            align_corners=False,
        ).clamp(0.0, 1.0)
        scale = float(gate_min) + (float(gate_max) - float(gate_min)) * gate_resized
        feats[level_idx] = target_feat * scale
        return (feats,) + tuple(inputs[1:])

    handle_input = first_module.register_forward_pre_hook(_cache_gate)
    handle_detect = detect_module.register_forward_pre_hook(_gate_detect_input)
    try:
        return _predict_one(
            yolo_model,
            image_path,
            imgsz=imgsz,
            conf=conf,
            iou=iou,
            max_det=max_det,
            device=device,
        )
    finally:
        handle_detect.remove()
        handle_input.remove()


def _project_dino_to_yolo_channels(
    dino_feat: torch.Tensor,
    target_feat: torch.Tensor,
    *,
    components: int,
) -> torch.Tensor:
    out_channels = int(target_feat.shape[1])
    h, w = target_feat.shape[-2:]
    requested_components = out_channels if int(components) <= 0 else min(int(components), out_channels)
    dino_resized = F.interpolate(
        dino_feat.detach().float(),
        size=(h, w),
        mode="bilinear",
        align_corners=False,
    )

    projected: List[torch.Tensor] = []
    for sample in dino_resized.cpu():
        tokens, feat_h, feat_w = _feature_to_tokens(sample)
        pca = _fit_pca_deterministic(tokens, n_components=requested_components)
        channels = pca.T.contiguous().view(requested_components, feat_h, feat_w)
        if requested_components < out_channels:
            repeat = int(np.ceil(float(out_channels) / float(requested_components)))
            channels = channels.repeat(repeat, 1, 1)[:out_channels]
        projected.append(channels)

    dino_projected = torch.stack(projected, dim=0).to(device=target_feat.device, dtype=target_feat.dtype)
    dino_mean = dino_projected.mean(dim=(2, 3), keepdim=True)
    dino_std = dino_projected.std(dim=(2, 3), keepdim=True).clamp_min(1e-4)
    dino_norm = (dino_projected - dino_mean) / dino_std

    target_mean = target_feat.detach().mean(dim=(2, 3), keepdim=True)
    target_std = target_feat.detach().std(dim=(2, 3), keepdim=True).clamp_min(1e-4)
    return dino_norm * target_std + target_mean


def _fuse_yolo_dino_features(
    target_feat: torch.Tensor,
    dino_projected: torch.Tensor,
    *,
    mode: str,
    strength: float,
) -> torch.Tensor:
    strength = float(strength)
    if mode == "replace":
        return dino_projected
    if mode == "blend":
        alpha = float(np.clip(strength, 0.0, 1.0))
        return target_feat * (1.0 - alpha) + dino_projected * alpha
    if mode == "add":
        residual = dino_projected - dino_projected.mean(dim=(2, 3), keepdim=True)
        return target_feat + strength * residual
    if mode == "mul":
        dino_norm = (dino_projected - dino_projected.mean(dim=(2, 3), keepdim=True)) / (
            dino_projected.std(dim=(2, 3), keepdim=True).clamp_min(1e-4)
        )
        scale = 1.0 + strength * torch.tanh(dino_norm)
        return target_feat * scale
    raise ValueError(f"Unsupported DINO fuse mode: {mode}")


def _predict_one_dino_fused(
    yolo_model: object,
    dino: DinoFeatureExtractor,
    image_path: str,
    *,
    imgsz: int,
    conf: float,
    iou: float,
    max_det: int,
    device: torch.device,
    feature_level: str,
    mode: str,
    strength: float,
    components: int,
) -> List[Dict[str, object]]:
    det_model = yolo_model.model
    detect_module = _find_detect_module(det_model)
    level_to_idx = {"p3": 0, "p4": 1, "p5": 2}
    level_idx = level_to_idx[feature_level]
    state: Dict[str, Optional[torch.Tensor]] = {"dino_feat": None}

    first_module = det_model.model[0]

    def _cache_dino_feature(_module, inputs):
        if not inputs:
            state["dino_feat"] = None
            return None
        images = inputs[0].detach().float()
        with torch.no_grad():
            state["dino_feat"] = dino(images.to(device)).detach()
        return None

    def _fuse_detect_input(_module, inputs):
        dino_feat = state.get("dino_feat")
        if not inputs or dino_feat is None:
            return None
        x = inputs[0]
        if not isinstance(x, (list, tuple)) or len(x) <= level_idx:
            return None

        feats = list(x)
        target_feat = feats[level_idx]
        dino_projected = _project_dino_to_yolo_channels(
            dino_feat,
            target_feat,
            components=components,
        )
        feats[level_idx] = _fuse_yolo_dino_features(
            target_feat,
            dino_projected,
            mode=mode,
            strength=strength,
        )
        return (feats,) + tuple(inputs[1:])

    handle_input = first_module.register_forward_pre_hook(_cache_dino_feature)
    handle_detect = detect_module.register_forward_pre_hook(_fuse_detect_input)
    try:
        return _predict_one(
            yolo_model,
            image_path,
            imgsz=imgsz,
            conf=conf,
            iou=iou,
            max_det=max_det,
            device=device,
        )
    finally:
        handle_detect.remove()
        handle_input.remove()


def _result_to_predictions(result: object) -> List[Dict[str, object]]:
    predictions: List[Dict[str, object]] = []
    names = getattr(result, "names", {}) or {}

    obb = getattr(result, "obb", None)
    if obb is not None and len(obb) > 0:
        try:
            polys = obb.xyxyxyxy.detach().cpu().float().numpy()
        except Exception:
            from ultralytics.utils.ops import xywhr2xyxyxyxy

            polys = xywhr2xyxyxyxy(obb.xywhr).detach().cpu().float().numpy()
        polys = polys.reshape(-1, 4, 2)
        confs = obb.conf.detach().cpu().float().numpy()
        classes = obb.cls.detach().cpu().long().numpy()
        for poly, conf, cls_id in zip(polys, confs, classes):
            predictions.append(
                {
                    "poly": poly.astype(np.float32),
                    "conf": float(conf),
                    "cls": int(cls_id),
                    "name": str(names.get(int(cls_id), int(cls_id))),
                }
            )
        return predictions

    boxes = getattr(result, "boxes", None)
    if boxes is not None and len(boxes) > 0:
        xyxy = boxes.xyxy.detach().cpu().float().numpy()
        confs = boxes.conf.detach().cpu().float().numpy()
        classes = boxes.cls.detach().cpu().long().numpy()
        for box, conf, cls_id in zip(xyxy, confs, classes):
            x1, y1, x2, y2 = box.tolist()
            poly = np.asarray([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], dtype=np.float32)
            predictions.append(
                {
                    "poly": poly,
                    "conf": float(conf),
                    "cls": int(cls_id),
                    "name": str(names.get(int(cls_id), int(cls_id))),
                }
            )
    return predictions


def _draw_predictions(
    image: Image.Image,
    predictions: Sequence[Dict[str, object]],
    color: Tuple[int, int, int],
    *,
    label_prefix: str = "",
    max_labels: int = 60,
    annotation_scale: float = 1.0,
    line_width_scale: float = 1.0,
    shadow: bool = False,
) -> Image.Image:
    canvas = image.convert("RGB").copy()
    draw = ImageDraw.Draw(canvas)
    scale = max(float(annotation_scale), 0.1)
    line_width = max(3, int(round(max(canvas.size) / 160.0 * scale * float(line_width_scale))))
    font = _annotation_font(int(round(max(canvas.size) / 90.0 * scale)))
    for i, pred in enumerate(predictions):
        poly = np.asarray(pred["poly"], dtype=np.float32).reshape(4, 2)
        points = [tuple(p) for p in poly.tolist()]
        if shadow:
            draw.line(points + [points[0]], fill=(0, 0, 0), width=max(line_width + 4, line_width * 2))
        draw.line(points + [points[0]], fill=color, width=line_width)
        if i < max_labels:
            text = f"{label_prefix}{pred['name']} {float(pred['conf']):.2f}"
            x = float(poly[:, 0].min())
            y = float(poly[:, 1].min())
            _draw_text_with_bg(draw, (x + 2, y + 2), text, fill=color, font=font)
    return canvas


def _poly_center(poly: np.ndarray) -> np.ndarray:
    return np.asarray(poly, dtype=np.float32).reshape(4, 2).mean(axis=0)


def _predictions_to_center_points(predictions: Sequence[Dict[str, object]]) -> np.ndarray:
    points: List[Tuple[float, float, float]] = []
    for pred in predictions:
        center = _poly_center(np.asarray(pred["poly"], dtype=np.float32))
        points.append((float(center[0]), float(center[1]), float(pred.get("conf", 1.0))))
    if not points:
        return np.zeros((0, 3), dtype=np.float32)
    return np.asarray(points, dtype=np.float32)


def _feature_level_label(feature_level: str) -> str:
    level = str(feature_level).lower()
    mapping = {"p3": "L3/P3", "p4": "L4/P4", "p5": "L5/P5"}
    return mapping.get(level, level.upper())


def _draw_prediction_difference(
    image: Image.Image,
    raw_predictions: Sequence[Dict[str, object]],
    fused_predictions: Sequence[Dict[str, object]],
    *,
    annotation_scale: float = 1.0,
    match_iou_threshold: float = 0.25,
    max_unmatched_labels: int = 0,
) -> Image.Image:
    canvas = image.convert("RGB").copy()
    draw = ImageDraw.Draw(canvas)
    scale = max(float(annotation_scale), 0.1)
    width = max(2, int(round(max(canvas.size) / 300.0 * scale)))
    raw_color = (0, 245, 255)
    fused_color = (255, 35, 210)
    shift_color = (255, 235, 30)
    font = _annotation_font(int(round(max(canvas.size) / 125.0 * scale)))

    used_fused: set[int] = set()
    matched_pairs: List[Tuple[int, int, float]] = []
    for raw_idx, raw_pred in enumerate(raw_predictions):
        if raw_pred.get("cls") is None:
            continue
        raw_box = _aabb_from_poly(np.asarray(raw_pred["poly"], dtype=np.float32))
        best_idx = -1
        best_iou = 0.0
        for fused_idx, fused_pred in enumerate(fused_predictions):
            if fused_idx in used_fused or int(fused_pred["cls"]) != int(raw_pred["cls"]):
                continue
            fused_box = _aabb_from_poly(np.asarray(fused_pred["poly"], dtype=np.float32))
            iou = _aabb_iou(raw_box, fused_box)
            if iou > best_iou:
                best_iou = iou
                best_idx = fused_idx
        if best_idx >= 0 and best_iou >= float(match_iou_threshold):
            used_fused.add(best_idx)
            matched_pairs.append((raw_idx, best_idx, best_iou))

    def draw_poly(pred: Dict[str, object], color: Tuple[int, int, int], extra_width: int = 0) -> None:
        poly = np.asarray(pred["poly"], dtype=np.float32).reshape(4, 2)
        points = [tuple(p) for p in poly.tolist()]
        draw.line(points + [points[0]], fill=(0, 0, 0), width=width + 2 + extra_width)
        draw.line(points + [points[0]], fill=color, width=width + extra_width)

    for pred in raw_predictions:
        draw_poly(pred, raw_color)
    for pred in fused_predictions:
        draw_poly(pred, fused_color, extra_width=1)

    raw_matched = {raw_idx for raw_idx, _fused_idx, _iou in matched_pairs}
    fused_matched = {fused_idx for _raw_idx, fused_idx, _iou in matched_pairs}
    for raw_idx, fused_idx, _iou in matched_pairs:
        raw_center = _poly_center(np.asarray(raw_predictions[raw_idx]["poly"], dtype=np.float32))
        fused_center = _poly_center(np.asarray(fused_predictions[fused_idx]["poly"], dtype=np.float32))
        shift = float(np.linalg.norm(fused_center - raw_center))
        if shift >= 2.0:
            draw.line(
                (float(raw_center[0]), float(raw_center[1]), float(fused_center[0]), float(fused_center[1])),
                fill=(0, 0, 0),
                width=max(1, width // 2 + 3),
            )
            draw.line(
                (float(raw_center[0]), float(raw_center[1]), float(fused_center[0]), float(fused_center[1])),
                fill=shift_color,
                width=max(1, width // 2),
            )

    raw_label_count = 0
    for raw_idx, pred in enumerate(raw_predictions):
        if raw_idx in raw_matched:
            continue
        if raw_label_count >= int(max_unmatched_labels):
            continue
        center = _poly_center(np.asarray(pred["poly"], dtype=np.float32))
        _draw_text_with_bg(
            draw,
            (float(center[0]) + width, float(center[1]) - width),
            "YOLO only",
            fill=raw_color,
            font=font,
        )
        raw_label_count += 1
    fused_label_count = 0
    for fused_idx, pred in enumerate(fused_predictions):
        if fused_idx in fused_matched:
            continue
        if fused_label_count >= int(max_unmatched_labels):
            continue
        center = _poly_center(np.asarray(pred["poly"], dtype=np.float32))
        _draw_text_with_bg(
            draw,
            (float(center[0]) + width, float(center[1]) + width),
            "DINO only",
            fill=fused_color,
            font=font,
        )
        fused_label_count += 1

    legend = f"cyan YOLO={len(raw_predictions)}   magenta DINOxYOLO={len(fused_predictions)}"
    _draw_text_with_bg(draw, (12, 12), legend, fill=(255, 255, 255), font=font)
    return canvas


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
        if len(reference_polys) > 0:
            nearest = int(np.argmin(np.linalg.norm(reference_centers - center[None, :], axis=1)))
            template_poly = reference_polys[nearest]
            poly = template_poly + (center - _poly_center(template_poly))[None, :]
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
                "cls": -1,
                "name": f"freq{idx}",
                "candidate": np.asarray([float(x), float(y), float(score), float(mask_value)], dtype=np.float32),
            }
        )
    return pseudo


def _aabb_from_poly(poly: np.ndarray) -> np.ndarray:
    poly = np.asarray(poly, dtype=np.float32).reshape(4, 2)
    return np.asarray(
        [poly[:, 0].min(), poly[:, 1].min(), poly[:, 0].max(), poly[:, 1].max()],
        dtype=np.float32,
    )


def _aabb_iou(box_a: np.ndarray, box_b: np.ndarray) -> float:
    xa1, ya1, xa2, ya2 = box_a.tolist()
    xb1, yb1, xb2, yb2 = box_b.tolist()
    ix1 = max(xa1, xb1)
    iy1 = max(ya1, yb1)
    ix2 = min(xa2, xb2)
    iy2 = min(ya2, yb2)
    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    inter = iw * ih
    area_a = max(0.0, xa2 - xa1) * max(0.0, ya2 - ya1)
    area_b = max(0.0, xb2 - xb1) * max(0.0, yb2 - yb1)
    return float(inter / max(area_a + area_b - inter, 1e-12))


def _filter_pseudo_boxes_by_reference_overlap(
    pseudo_boxes: Sequence[Dict[str, object]],
    reference_predictions: Sequence[Dict[str, object]],
    *,
    iou_threshold: float,
) -> Tuple[List[Dict[str, object]], int]:
    threshold = float(iou_threshold)
    if threshold <= 0.0 or not pseudo_boxes or not reference_predictions:
        return list(pseudo_boxes), 0

    reference_aabbs = [
        _aabb_from_poly(np.asarray(pred["poly"], dtype=np.float32))
        for pred in reference_predictions
    ]
    kept: List[Dict[str, object]] = []
    removed = 0
    for pseudo in pseudo_boxes:
        pseudo_aabb = _aabb_from_poly(np.asarray(pseudo["poly"], dtype=np.float32))
        if any(_aabb_iou(pseudo_aabb, ref_aabb) >= threshold for ref_aabb in reference_aabbs):
            removed += 1
            continue
        kept.append(pseudo)
    return kept, removed


def _filter_pseudo_boxes_by_anchor_row_support(
    pseudo_boxes: Sequence[Dict[str, object]],
    anchor_points: Optional[np.ndarray],
    *,
    row_tolerance_px: float,
    min_anchor_points: int,
) -> Tuple[List[Dict[str, object]], int]:
    min_anchors = int(min_anchor_points)
    if min_anchors <= 0 or not pseudo_boxes:
        return list(pseudo_boxes), 0
    anchors = np.asarray(anchor_points, dtype=np.float32) if anchor_points is not None else np.zeros((0, 3), dtype=np.float32)
    if anchors.size == 0:
        return [], len(pseudo_boxes)
    anchors = anchors.reshape(-1, anchors.shape[-1])
    if anchors.shape[1] < 2:
        return [], len(pseudo_boxes)

    anchor_records = [{"x": float(point[0]), "y": float(point[1])} for point in anchors]
    anchor_records.sort(key=lambda item: float(item["y"]))
    tol = max(float(row_tolerance_px), 1.0)
    rows: List[List[Dict[str, float]]] = []
    current: List[Dict[str, float]] = [anchor_records[0]]
    current_y = float(anchor_records[0]["y"])
    for record in anchor_records[1:]:
        record_y = float(record["y"])
        if abs(record_y - current_y) <= tol:
            current.append(record)
            current_y = float(np.mean([float(item["y"]) for item in current]))
        else:
            rows.append(current)
            current = [record]
            current_y = record_y
    rows.append(current)

    supported_rows: List[Tuple[float, int]] = []
    for row in rows:
        if len(row) >= min_anchors:
            row_y = float(np.median([float(item["y"]) for item in row]))
            supported_rows.append((row_y, len(row)))
    if not supported_rows:
        return [], len(pseudo_boxes)

    kept: List[Dict[str, object]] = []
    removed = 0
    for pseudo in pseudo_boxes:
        center = _poly_center(np.asarray(pseudo["poly"], dtype=np.float32))
        nearest_distance = min(abs(float(center[1]) - row_y) for row_y, _count in supported_rows)
        if nearest_distance <= tol:
            kept.append(pseudo)
        else:
            removed += 1
    return kept, removed


def _filter_pseudo_boxes_by_row_frequency(
    pseudo_boxes: Sequence[Dict[str, object]],
    reference_predictions: Sequence[Dict[str, object]],
    *,
    row_tolerance_px: float,
    period_ratio: float,
    min_row_points: int,
) -> Tuple[List[Dict[str, object]], int]:
    ratio = float(period_ratio)
    if ratio <= 1.0 or not pseudo_boxes:
        return list(pseudo_boxes), 0

    records: List[Dict[str, object]] = []
    for pred in reference_predictions:
        center = _poly_center(np.asarray(pred["poly"], dtype=np.float32))
        records.append({"x": float(center[0]), "y": float(center[1]), "pseudo_idx": -1})
    for idx, pred in enumerate(pseudo_boxes):
        center = _poly_center(np.asarray(pred["poly"], dtype=np.float32))
        records.append({"x": float(center[0]), "y": float(center[1]), "pseudo_idx": idx})
    if not records:
        return list(pseudo_boxes), 0

    tol = max(float(row_tolerance_px), 1.0)
    records.sort(key=lambda item: float(item["y"]))
    rows: List[List[Dict[str, object]]] = []
    current: List[Dict[str, object]] = [records[0]]
    current_y = float(records[0]["y"])
    for record in records[1:]:
        record_y = float(record["y"])
        if abs(record_y - current_y) <= tol:
            current.append(record)
            current_y = float(np.mean([float(item["y"]) for item in current]))
        else:
            rows.append(current)
            current = [record]
            current_y = record_y
    rows.append(current)

    min_points = max(int(min_row_points), 2)
    row_periods: List[Tuple[int, float]] = []
    for row_idx, row in enumerate(rows):
        if len(row) < min_points:
            continue
        xs = np.asarray([float(item["x"]) for item in row], dtype=np.float32)
        period = _estimate_spacing_px(xs)
        if period >= 3.0:
            row_periods.append((row_idx, period))
    if len(row_periods) < 2:
        return list(pseudo_boxes), 0

    global_period = float(np.median([period for _row_idx, period in row_periods]))
    if global_period < 3.0:
        return list(pseudo_boxes), 0

    bad_rows: set[int] = set()
    for row_idx, period in row_periods:
        mismatch = max(period / max(global_period, 1e-6), global_period / max(period, 1e-6))
        if mismatch > ratio:
            bad_rows.add(row_idx)
    if not bad_rows:
        return list(pseudo_boxes), 0

    bad_pseudo_indices: set[int] = set()
    for row_idx in bad_rows:
        for record in rows[row_idx]:
            pseudo_idx = int(record["pseudo_idx"])
            if pseudo_idx >= 0:
                bad_pseudo_indices.add(pseudo_idx)

    kept = [pred for idx, pred in enumerate(pseudo_boxes) if idx not in bad_pseudo_indices]
    return kept, len(bad_pseudo_indices)


def _candidates_from_pseudo_boxes(pseudo_boxes: Sequence[Dict[str, object]]) -> np.ndarray:
    candidates: List[np.ndarray] = []
    for pred in pseudo_boxes:
        candidate = pred.get("candidate")
        if candidate is None:
            center = _poly_center(np.asarray(pred["poly"], dtype=np.float32))
            candidates.append(np.asarray([center[0], center[1], float(pred.get("conf", 0.0)), 1.0], dtype=np.float32))
        else:
            candidates.append(np.asarray(candidate, dtype=np.float32).reshape(4))
    if not candidates:
        return np.zeros((0, 4), dtype=np.float32)
    return np.stack(candidates, axis=0).astype(np.float32)


def _prediction_delta_metrics(
    before: Sequence[Dict[str, object]],
    after: Sequence[Dict[str, object]],
) -> Dict[str, float]:
    used_after: set[int] = set()
    ious: List[float] = []
    shifts: List[float] = []

    for before_pred in before:
        before_cls = int(before_pred["cls"])
        before_box = _aabb_from_poly(np.asarray(before_pred["poly"], dtype=np.float32))
        best_j = -1
        best_iou = 0.0
        for j, after_pred in enumerate(after):
            if j in used_after or int(after_pred["cls"]) != before_cls:
                continue
            after_box = _aabb_from_poly(np.asarray(after_pred["poly"], dtype=np.float32))
            iou = _aabb_iou(before_box, after_box)
            if iou > best_iou:
                best_iou = iou
                best_j = j

        if best_j < 0:
            continue

        used_after.add(best_j)
        after_box = _aabb_from_poly(np.asarray(after[best_j]["poly"], dtype=np.float32))
        before_center = np.asarray([(before_box[0] + before_box[2]) / 2, (before_box[1] + before_box[3]) / 2])
        after_center = np.asarray([(after_box[0] + after_box[2]) / 2, (after_box[1] + after_box[3]) / 2])
        ious.append(best_iou)
        shifts.append(float(np.linalg.norm(after_center - before_center)))

    return {
        "before_box_count": float(len(before)),
        "after_box_count": float(len(after)),
        "matched_box_count": float(len(ious)),
        "mean_matched_aabb_iou": float(np.mean(ious)) if ious else 0.0,
        "mean_center_shift_px": float(np.mean(shifts)) if shifts else 0.0,
    }


def _predict_one(
    yolo_model: object,
    image_path: str,
    *,
    imgsz: int,
    conf: float,
    iou: float,
    max_det: int,
    device: torch.device,
) -> List[Dict[str, object]]:
    device_arg: object = device.index if device.type == "cuda" else "cpu"
    results = yolo_model.predict(
        source=image_path,
        imgsz=imgsz,
        conf=conf,
        iou=iou,
        max_det=max_det,
        device=device_arg,
        verbose=False,
    )
    if not results:
        return []
    return _result_to_predictions(results[0])


def _make_grid(tiles: Sequence[Image.Image], cols: int = 4, gap: int = 8) -> Image.Image:
    if not tiles:
        raise ValueError("No tiles to place in grid")
    tile_w = max(tile.width for tile in tiles)
    tile_h = max(tile.height for tile in tiles)
    rows = int(np.ceil(len(tiles) / cols))
    panel = Image.new(
        "RGB",
        (cols * tile_w + (cols - 1) * gap, rows * tile_h + (rows - 1) * gap),
        color=(245, 245, 245),
    )
    for idx, tile in enumerate(tiles):
        row = idx // cols
        col = idx % cols
        panel.paste(tile, (col * (tile_w + gap), row * (tile_h + gap)))
    return panel


def _save_json(path: Path, data: Dict[str, object]) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def _apply_preset_args(args: argparse.Namespace) -> None:
    preset = str(getattr(args, "preset", "none") or "none")
    if preset == "none":
        return

    args.single_panel = True
    args.dino_mask_fuse = True
    if args.feature_level is None:
        args.feature_level = "p3"
    if args.dino_mask_fuse_level is None:
        args.dino_mask_fuse_level = "p3"
    if int(args.tile_size) == 300:
        args.tile_size = 520
    if float(args.dino_mask_fuse_bg) == 0.5:
        args.dino_mask_fuse_bg = 0.6
    if float(args.dino_mask_fuse_fg) == 1.8:
        args.dino_mask_fuse_fg = 1.5

    if preset != "single-panel-spacing":
        return

    args.spacing_complete = True
    args.spacing_draw_pseudo_boxes = True
    args.spacing_complete_missing_rows = True
    if args.spacing_anchor_source == "none":
        args.spacing_anchor_source = "mask-fused"
    if int(args.head_topk) == 80:
        args.head_topk = 200
    if float(args.head_score_threshold) == 0.05:
        args.head_score_threshold = 0.005
    if float(args.spacing_score_threshold) == 0.05:
        args.spacing_score_threshold = 0.005
    if float(args.spacing_proposal_score_threshold) == 0.02:
        args.spacing_proposal_score_threshold = 0.00005
    if float(args.spacing_dino_threshold) == 0.35:
        args.spacing_dino_threshold = 0.12
    if float(args.spacing_row_tolerance_px) == 28.0:
        args.spacing_row_tolerance_px = 70.0
    if float(args.spacing_nms_distance_px) == 24.0:
        args.spacing_nms_distance_px = 26.0
    if int(args.spacing_min_row_seeds) == 3:
        args.spacing_min_row_seeds = 1
    if int(args.spacing_max_candidates) == 80:
        args.spacing_max_candidates = 120
    if int(args.spacing_pseudo_min_anchor_per_row) == 0:
        args.spacing_pseudo_min_anchor_per_row = 2
    if float(args.spacing_row_prior_tolerance_px) == 0.0:
        args.spacing_row_prior_tolerance_px = 90.0
    if float(args.spacing_refine_window_px) == 0.0:
        args.spacing_refine_window_px = 36.0
    if float(args.spacing_refine_prior_sigma_px) == 0.0:
        args.spacing_refine_prior_sigma_px = 22.0
    if args.spacing_prior_bank_json is None:
        default_bank = Path("outputs/spacing_info/species_spacing_prior_bank.json")
        if default_bank.is_file():
            args.spacing_prior_bank_json = str(default_bank)


def main() -> None:
    args = _parse_args()
    _apply_preset_args(args)
    if args.image and args.image_dir:
        raise ValueError("Use either --image or --image-dir, not both")
    if args.single_panel:
        args.dino_mask_fuse = True

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
    dino_gate_level = str(args.dino_gate_level or feature_level).lower()
    dino_fuse_level = str(args.dino_fuse_level or feature_level).lower()
    dino_mask_fuse_level = str(args.dino_mask_fuse_level or feature_level).lower()
    spacing_prior_bank = _load_spacing_prior_bank(args.spacing_prior_bank_json)

    before_weights = _resolve_before_weights(config, args.before_weights)
    after_weights_path = Path(args.after_weights)
    if not after_weights_path.is_file():
        raise FileNotFoundError(f"After weights not found: {after_weights_path}")
    after_weights = str(after_weights_path)

    output_dir = (
        Path(args.output_dir).resolve()
        if args.output_dir
        else Path("outputs") / "dino_effect_compare"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"[dino-effect] output_dir={output_dir}", flush=True)
    print(
        f"[dino-effect] device={device} imgsz={image_size} feature_level={feature_level}",
        flush=True,
    )
    if spacing_prior_bank:
        names = ", ".join(str(item.get("name", "")) for item in spacing_prior_bank)
        print(f"[dino-effect] spacing_prior_bank={names}", flush=True)

    if args.image:
        image_path = Path(args.image).resolve()
        if not image_path.is_file():
            raise FileNotFoundError(f"Image not found: {image_path}")
        batches = _iter_image_path_batches([image_path], image_size, args.batch_size)
    elif args.image_dir:
        batches = _iter_image_dir_batches(Path(args.image_dir), image_size, args.batch_size)
    else:
        batches = _build_loader(config, split=args.split, batch_size=args.batch_size, num_workers=args.num_workers)

    print("[dino-effect] loading YOLO students and DINO encoder...", flush=True)
    before_student = _build_yolo_student(before_weights, config, feature_level, device)
    after_student = _build_yolo_student(after_weights, config, feature_level, device)
    dino = _build_dino(config, device)

    from ultralytics import YOLO

    before_yolo = YOLO(before_weights)
    after_yolo = YOLO(after_weights)
    print("[dino-effect] models loaded", flush=True)

    rows: List[Dict[str, object]] = []
    sample_count = 0

    for batch in batches:
        images = batch["images"].to(device)
        image_paths = [str(p) for p in batch.get("image_paths", [])]
        if not image_paths:
            image_paths = [f"sample_{sample_count + i:04d}" for i in range(images.shape[0])]

        with torch.no_grad():
            before_feat = _extract_yolo_feature(before_student, images, feature_level)
            after_feat = _extract_yolo_feature(after_student, images, feature_level)
            if args.dino_mask_fuse and dino_mask_fuse_level != feature_level:
                mask_fuse_after_feat = _extract_yolo_feature(after_student, images, dino_mask_fuse_level)
            else:
                mask_fuse_after_feat = after_feat
            dino_feat = dino(images).detach()

        for i in range(images.shape[0]):
            if sample_count >= args.num_samples:
                break

            image_path = image_paths[i]
            stem = Path(image_path).stem
            print(
                f"[dino-effect] processing {sample_count + 1}/{args.num_samples}: {Path(image_path).name}",
                flush=True,
            )

            dino_tokens, dino_h, dino_w = _feature_to_tokens(dino_feat[i].detach().cpu())
            before_tokens_dino_grid = _tokens_at_dino_grid(before_feat[i], (dino_h, dino_w))
            after_tokens_dino_grid = _tokens_at_dino_grid(after_feat[i], (dino_h, dino_w))
            cka_before = _linear_cka(before_tokens_dino_grid, dino_tokens)
            cka_after = _linear_cka(after_tokens_dino_grid, dino_tokens)

            before_pca_img, after_pca_img = _student_pca_images(before_feat[i], after_feat[i])
            dino_pca_img = _tokens_to_pca_image(_fit_pca(dino_tokens), dino_h, dino_w)
            input_pil = _tensor_image_to_pil(images[i].detach().cpu())
            original_pil = Image.open(image_path).convert("RGB") if Path(image_path).is_file() else input_pil
            before_response_align = _feature_response_image(before_feat[i], original_pil.size)
            after_response_align = _feature_response_image(after_feat[i], original_pil.size)
            response_delta_align = _feature_delta_image(before_feat[i], after_feat[i], original_pil.size)

            before_preds = _predict_one(
                before_yolo,
                image_path,
                imgsz=image_size,
                conf=args.conf,
                iou=args.iou,
                max_det=args.max_det,
                device=device,
            ) if Path(image_path).is_file() else []
            after_preds = _predict_one(
                after_yolo,
                image_path,
                imgsz=image_size,
                conf=args.conf,
                iou=args.iou,
                max_det=args.max_det,
                device=device,
            ) if Path(image_path).is_file() else []
            gated_preds = (
                _predict_one_dino_gated(
                    after_yolo,
                    dino,
                    image_path,
                    imgsz=image_size,
                    conf=args.conf,
                    iou=args.iou,
                    max_det=args.max_det,
                    device=device,
                    feature_level=dino_gate_level,
                    gate_min=args.dino_gate_min,
                    gate_max=args.dino_gate_max,
                    target_coverage=args.mask_target_coverage,
                    min_coverage=args.mask_min_coverage,
                    max_coverage=args.mask_max_coverage,
                )
                if args.dino_gate and Path(image_path).is_file()
                else []
            )
            fused_preds = (
                _predict_one_dino_fused(
                    after_yolo,
                    dino,
                    image_path,
                    imgsz=image_size,
                    conf=args.conf,
                    iou=args.iou,
                    max_det=args.max_det,
                    device=device,
                    feature_level=dino_fuse_level,
                    mode=args.dino_fuse_mode,
                    strength=args.dino_fuse_strength,
                    components=args.dino_fuse_components,
                )
                if args.dino_fuse and Path(image_path).is_file()
                else []
            )
            mask_fused_preds = (
                _predict_one_dino_gated(
                    after_yolo,
                    dino,
                    image_path,
                    imgsz=image_size,
                    conf=args.conf,
                    iou=args.iou,
                    max_det=args.max_det,
                    device=device,
                    feature_level=dino_mask_fuse_level,
                    gate_min=args.dino_mask_fuse_bg,
                    gate_max=args.dino_mask_fuse_fg,
                    target_coverage=args.mask_target_coverage,
                    min_coverage=args.mask_min_coverage,
                    max_coverage=args.mask_max_coverage,
                )
                if args.dino_mask_fuse and Path(image_path).is_file()
                else []
            )

            before_overlay = _draw_predictions(original_pil, before_preds, (230, 40, 40), label_prefix="B:")
            after_overlay = _draw_predictions(original_pil, after_preds, (30, 170, 60), label_prefix="A:")
            delta_overlay = _draw_predictions(original_pil, before_preds, (230, 40, 40), label_prefix="B:", max_labels=0)
            delta_overlay = _draw_predictions(delta_overlay, after_preds, (30, 170, 60), label_prefix="A:", max_labels=0)
            dino_mask, dino_mask_overlay, dino_mask_stats = _dino_mask_images(
                dino_feat[i],
                original_pil,
                target_coverage=args.mask_target_coverage,
                min_coverage=args.mask_min_coverage,
                max_coverage=args.mask_max_coverage,
                blur_radius=args.mask_blur_radius,
            )

            mask_path: Optional[Path] = None
            mask_overlay_path: Optional[Path] = None
            if not args.single_panel:
                mask_path = output_dir / f"{sample_count:04d}_{stem}_dino_mask.png"
                mask_overlay_path = output_dir / f"{sample_count:04d}_{stem}_dino_mask_overlay.png"
                dino_mask.save(mask_path)
                dino_mask_overlay.save(mask_overlay_path)

            tiles = [
                _make_tile(input_pil, "Input", args.tile_size),
                _make_tile(dino_pca_img, "DINO PCA", args.tile_size),
                _make_tile(dino_mask, "DINO Mask", args.tile_size),
                _make_tile(dino_mask_overlay, "DINO Mask Overlay", args.tile_size),
                _make_tile(before_pca_img, f"Before YOLO {feature_level.upper()}", args.tile_size),
                _make_tile(after_pca_img, f"After YOLO {feature_level.upper()}", args.tile_size),
                _make_tile(before_response_align, f"Before {feature_level.upper()} Heatmap", args.tile_size),
                _make_tile(after_response_align, f"After {feature_level.upper()} Heatmap", args.tile_size),
                _make_tile(response_delta_align, "Heatmap Delta red=up blue=down", args.tile_size),
                _make_tile(before_overlay, "Before Ckpt Boxes", args.tile_size),
                _make_tile(after_overlay, "After Ckpt Boxes", args.tile_size),
                _make_tile(delta_overlay, "Box Delta B=red A=green", args.tile_size),
            ]
            gate_compare_path = None
            fuse_compare_path = None
            mask_fuse_feature_path = None
            mask_fuse_head_path = None
            spacing_complete_path = None
            compact_panel_path = None
            cls_before_stats: Optional[Dict[str, float]] = None
            cls_fused_stats: Optional[Dict[str, float]] = None
            cls_delta_img: Optional[Image.Image] = None
            cls_delta_centers: Optional[Image.Image] = None
            spacing_overlay: Optional[Image.Image] = None
            spacing_prior: Optional[Image.Image] = None
            pseudo_box_overlay: Optional[Image.Image] = None
            spacing_stats: Optional[Dict[str, float]] = None
            spacing_pseudo_boxes: List[Dict[str, object]] = []
            if args.dino_gate:
                gated_overlay = _draw_predictions(original_pil, gated_preds, (45, 90, 240), label_prefix="G:")
                gate_delta_overlay = _draw_predictions(
                    original_pil,
                    after_preds,
                    (30, 170, 60),
                    label_prefix="A:",
                    max_labels=0,
                )
                gate_delta_overlay = _draw_predictions(
                    gate_delta_overlay,
                    gated_preds,
                    (45, 90, 240),
                    label_prefix="G:",
                    max_labels=0,
                )
                if not args.single_panel:
                    tiles.extend(
                        [
                            _make_tile(gated_overlay, f"After + DINO Gate {dino_gate_level.upper()}", args.tile_size),
                            _make_tile(gate_delta_overlay, "Gate Delta A=green G=blue", args.tile_size),
                        ]
                    )
                    gate_compare_tiles = [
                        _make_tile(original_pil, "Input", args.tile_size),
                        _make_tile(after_overlay, "After Ckpt", args.tile_size),
                        _make_tile(gated_overlay, f"After + Gate {dino_gate_level.upper()}", args.tile_size),
                        _make_tile(gate_delta_overlay, "A=green G=blue", args.tile_size),
                    ]
                    gate_compare = _make_grid(gate_compare_tiles, cols=4)
                    gate_compare_path = output_dir / f"{sample_count:04d}_{stem}_gate_compare.png"
                    gate_compare.save(gate_compare_path)
            if args.dino_fuse:
                fused_overlay = _draw_predictions(original_pil, fused_preds, (155, 45, 230), label_prefix="F:")
                fuse_delta_overlay = _draw_predictions(
                    original_pil,
                    after_preds,
                    (30, 170, 60),
                    label_prefix="A:",
                    max_labels=0,
                )
                fuse_delta_overlay = _draw_predictions(
                    fuse_delta_overlay,
                    fused_preds,
                    (155, 45, 230),
                    label_prefix="F:",
                    max_labels=0,
                )
                if not args.single_panel:
                    tiles.extend(
                        [
                            _make_tile(
                                fused_overlay,
                                f"After + DINO Fuse {dino_fuse_level.upper()}",
                                args.tile_size,
                            ),
                            _make_tile(fuse_delta_overlay, "Fuse Delta A=green F=purple", args.tile_size),
                        ]
                    )
                    fuse_compare_tiles = [
                        _make_tile(original_pil, "Input", args.tile_size),
                        _make_tile(after_overlay, "After Ckpt", args.tile_size),
                        _make_tile(
                            fused_overlay,
                            f"After + Fuse {dino_fuse_level.upper()} {args.dino_fuse_mode}",
                            args.tile_size,
                        ),
                        _make_tile(fuse_delta_overlay, "A=green F=purple", args.tile_size),
                    ]
                    fuse_compare = _make_grid(fuse_compare_tiles, cols=4)
                    fuse_compare_path = output_dir / f"{sample_count:04d}_{stem}_fuse_compare.png"
                    fuse_compare.save(fuse_compare_path)
            if args.dino_mask_fuse:
                yolo_feat_for_mask = mask_fuse_after_feat[i]
                mask_fused_feat, _mask_gate = _apply_dino_mask_fuse_to_feature(
                    yolo_feat_for_mask,
                    dino_feat[i],
                    bg_multiplier=args.dino_mask_fuse_bg,
                    fg_multiplier=args.dino_mask_fuse_fg,
                    target_coverage=args.mask_target_coverage,
                    min_coverage=args.mask_min_coverage,
                    max_coverage=args.mask_max_coverage,
                )
                mask_fused_overlay = _draw_predictions(
                    original_pil,
                    mask_fused_preds,
                    (255, 150, 20),
                    label_prefix="M:",
                )
                mask_fuse_delta_overlay = _draw_predictions(
                    original_pil,
                    after_preds,
                    (30, 170, 60),
                    label_prefix="A:",
                    max_labels=0,
                )
                mask_fuse_delta_overlay = _draw_predictions(
                    mask_fuse_delta_overlay,
                    mask_fused_preds,
                    (255, 150, 20),
                    label_prefix="M:",
                    max_labels=0,
                )
                before_response = _feature_response_image(yolo_feat_for_mask, original_pil.size)
                fused_response = _feature_response_image(mask_fused_feat, original_pil.size)
                response_delta = _feature_delta_image(yolo_feat_for_mask, mask_fused_feat, original_pil.size)
                if not args.single_panel:
                    mask_fuse_tiles = [
                        _make_tile(original_pil, "Input", args.tile_size),
                        _make_tile(dino_mask, "DINO Mask", args.tile_size),
                        _make_tile(before_response, f"YOLO {dino_mask_fuse_level.upper()} Response", args.tile_size),
                        _make_tile(
                            fused_response,
                            f"Mask-Fused YOLO {dino_mask_fuse_level.upper()}",
                            args.tile_size,
                        ),
                        _make_tile(response_delta, "Response Delta red=up blue=down", args.tile_size),
                        _make_tile(after_overlay, "After Ckpt Boxes", args.tile_size),
                        _make_tile(mask_fused_overlay, "After + DINO Mask Fuse", args.tile_size),
                        _make_tile(mask_fuse_delta_overlay, "A=green M=orange", args.tile_size),
                    ]
                    mask_fuse_feature = _make_grid(mask_fuse_tiles, cols=4)
                    mask_fuse_feature_path = output_dir / f"{sample_count:04d}_{stem}_mask_fuse_feature_compare.png"
                    mask_fuse_feature.save(mask_fuse_feature_path)
                if args.head_diagnostic or args.dino_mask_fuse:
                    level_idx = {"p3": 0, "p4": 1, "p5": 2}[dino_mask_fuse_level]
                    detect_module = _find_detect_module(after_student.det_model)
                    cls_before_map, _cls_before_ids = _head_cls_score_map(
                        detect_module,
                        yolo_feat_for_mask,
                        level_idx,
                    )
                    cls_fused_map, _cls_fused_ids = _head_cls_score_map(
                        detect_module,
                        mask_fused_feat,
                        level_idx,
                    )
                    cls_before_stats = _score_stats(cls_before_map, args.head_score_threshold)
                    cls_fused_stats = _score_stats(cls_fused_map, args.head_score_threshold)
                    cls_before_img = _heatmap_image(cls_before_map.numpy(), original_pil.size, cmap_name="magma")
                    cls_fused_img = _heatmap_image(cls_fused_map.numpy(), original_pil.size, cmap_name="magma")
                    cls_delta_img = _delta_heatmap_image(cls_before_map, cls_fused_map, original_pil.size)
                    marker_scale = (
                        args.single_panel_marker_scale
                        if args.single_panel
                        else args.annotation_scale
                    )
                    cls_before_centers = _draw_score_candidates(
                        original_pil,
                        cls_before_map,
                        (30, 170, 60),
                        topk=args.head_topk,
                        threshold=args.head_score_threshold,
                        label_prefix="A:",
                        annotation_scale=marker_scale,
                        max_labels=0 if args.single_panel else 12,
                    )
                    cls_fused_centers = _draw_score_candidates(
                        original_pil,
                        cls_fused_map,
                        (255, 150, 20),
                        topk=args.head_topk,
                        threshold=args.head_score_threshold,
                        label_prefix="M:",
                        annotation_scale=marker_scale,
                        max_labels=0 if args.single_panel else 12,
                    )
                    cls_delta_centers = _draw_score_candidates(
                        cls_before_centers,
                        cls_fused_map,
                        (255, 150, 20),
                        topk=args.head_topk,
                        threshold=args.head_score_threshold,
                        label_prefix="M:",
                        annotation_scale=marker_scale,
                        max_labels=0 if args.single_panel else 12,
                    )
                    if not args.single_panel:
                        head_diag_tiles = [
                            _make_tile(original_pil, "Input", args.tile_size),
                            _make_tile(dino_mask, "DINO Mask", args.tile_size),
                            _make_tile(before_response, f"YOLO {dino_mask_fuse_level.upper()} Response", args.tile_size),
                            _make_tile(
                                fused_response,
                                f"Mask-Fused {dino_mask_fuse_level.upper()} Response",
                                args.tile_size,
                            ),
                            _make_tile(
                                cls_before_img,
                                f"{dino_mask_fuse_level.upper()} Cls Score Before",
                                args.tile_size,
                            ),
                            _make_tile(
                                cls_fused_img,
                                f"{dino_mask_fuse_level.upper()} Cls Score Mask-Fused",
                                args.tile_size,
                            ),
                            _make_tile(cls_delta_img, "Cls Delta red=up blue=down", args.tile_size),
                            _make_tile(cls_delta_centers, "Cls Candidates A=green M=orange", args.tile_size),
                            _make_tile(after_overlay, "Final Boxes Before", args.tile_size),
                            _make_tile(mask_fused_overlay, "Final Boxes Mask-Fused", args.tile_size),
                            _make_tile(mask_fuse_delta_overlay, "Box Delta A=green M=orange", args.tile_size),
                        ]
                        head_diag = _make_grid(head_diag_tiles, cols=4)
                        mask_fuse_head_path = output_dir / f"{sample_count:04d}_{stem}_mask_fuse_head_diagnostic.png"
                        head_diag.save(mask_fuse_head_path)
                    if args.spacing_complete:
                        anchor_parts: List[np.ndarray] = []
                        if args.spacing_anchor_source in {"raw", "both"}:
                            anchor_parts.append(_predictions_to_center_points(after_preds))
                        if args.spacing_anchor_source in {"mask-fused", "both"}:
                            anchor_parts.append(_predictions_to_center_points(mask_fused_preds))
                        spacing_anchor_points = (
                            np.concatenate([part for part in anchor_parts if part.size], axis=0)
                            if any(part.size for part in anchor_parts)
                            else np.zeros((0, 3), dtype=np.float32)
                        )
                        seed_points, spacing_candidates, spacing_stats, spacing_row_lines = _spacing_complete_candidates(
                            cls_fused_map,
                            _mask_gate,
                            image_size=original_pil.size,
                            score_threshold=args.spacing_score_threshold,
                            proposal_score_threshold=args.spacing_proposal_score_threshold,
                            gate_threshold=args.spacing_dino_threshold,
                            row_tolerance_px=args.spacing_row_tolerance_px,
                            nms_distance_px=args.spacing_nms_distance_px,
                            min_row_seeds=args.spacing_min_row_seeds,
                            max_candidates=args.spacing_max_candidates,
                            period_prior_px=args.spacing_period_prior_px,
                            period_prior_ratio=args.spacing_period_prior_ratio,
                            period_scale=args.spacing_period_scale,
                            prior_bank=spacing_prior_bank,
                            prior_bank_sigma=args.spacing_prior_bank_sigma,
                            prior_min_weight=args.spacing_prior_min_weight,
                            row_period_prior_px=args.spacing_row_period_prior_px,
                            row_period_prior_ratio=args.spacing_row_period_prior_ratio,
                            row_period_scale=args.spacing_row_period_scale,
                            row_prior_tolerance_px=args.spacing_row_prior_tolerance_px,
                            complete_missing_rows=args.spacing_complete_missing_rows,
                            refine_window_px=args.spacing_refine_window_px,
                            refine_step_px=args.spacing_refine_step_px,
                            refine_prior_sigma_px=args.spacing_refine_prior_sigma_px,
                            anchor_points=spacing_anchor_points,
                        )
                        if spacing_stats is not None:
                            spacing_stats["spacing_anchor_source"] = args.spacing_anchor_source
                        if args.spacing_draw_pseudo_boxes:
                            raw_spacing_candidate_count = int(spacing_candidates.shape[0])
                            spacing_pseudo_boxes = _spacing_candidate_boxes(
                                spacing_candidates,
                                mask_fused_preds,
                                original_pil.size,
                                fallback_width_px=args.spacing_pseudo_box_width_px,
                                fallback_height_px=args.spacing_pseudo_box_height_px,
                            )
                            raw_pseudo_box_count = len(spacing_pseudo_boxes)
                            anchor_row_tolerance = (
                                args.spacing_pseudo_anchor_row_tolerance_px
                                if float(args.spacing_pseudo_anchor_row_tolerance_px) > 0.0
                                else args.spacing_row_tolerance_px
                            )
                            spacing_pseudo_boxes, anchor_row_removed = _filter_pseudo_boxes_by_anchor_row_support(
                                spacing_pseudo_boxes,
                                spacing_anchor_points,
                                row_tolerance_px=anchor_row_tolerance,
                                min_anchor_points=args.spacing_pseudo_min_anchor_per_row,
                            )
                            spacing_pseudo_boxes, overlap_removed = _filter_pseudo_boxes_by_reference_overlap(
                                spacing_pseudo_boxes,
                                mask_fused_preds,
                                iou_threshold=args.spacing_pseudo_overlap_iou,
                            )
                            spacing_pseudo_boxes, frequency_removed = _filter_pseudo_boxes_by_row_frequency(
                                spacing_pseudo_boxes,
                                mask_fused_preds,
                                row_tolerance_px=args.spacing_row_tolerance_px,
                                period_ratio=args.spacing_pseudo_row_frequency_ratio,
                                min_row_points=args.spacing_pseudo_row_frequency_min_points,
                            )
                            spacing_candidates = _candidates_from_pseudo_boxes(spacing_pseudo_boxes)
                            if spacing_stats is not None:
                                spacing_stats["spacing_candidate_count_raw"] = float(raw_spacing_candidate_count)
                                spacing_stats["spacing_pseudo_box_count_raw"] = float(raw_pseudo_box_count)
                                spacing_stats["spacing_pseudo_anchor_row_removed"] = float(anchor_row_removed)
                                spacing_stats["spacing_pseudo_overlap_removed"] = float(overlap_removed)
                                spacing_stats["spacing_pseudo_frequency_removed"] = float(frequency_removed)
                                spacing_stats["spacing_candidate_count"] = float(spacing_candidates.shape[0])
                                spacing_stats["spacing_pseudo_box_count"] = float(len(spacing_pseudo_boxes))
                        spacing_overlay = _draw_spacing_points(
                            original_pil,
                            seed_points,
                            spacing_candidates,
                            row_lines=spacing_row_lines,
                            pseudo_boxes=spacing_pseudo_boxes,
                            annotation_scale=marker_scale,
                            max_labels=0 if args.single_panel else 999999,
                        )
                        spacing_prior = _spacing_prior_image(
                            spacing_candidates,
                            original_pil.size,
                            annotation_scale=marker_scale,
                        )
                        pseudo_base = (
                            _draw_predictions(
                                original_pil,
                                mask_fused_preds,
                                (255, 150, 20),
                                label_prefix="M:",
                                max_labels=0,
                                annotation_scale=args.single_panel_box_scale,
                                line_width_scale=0.45,
                            )
                            if args.single_panel
                            else mask_fused_overlay
                        )
                        pseudo_box_overlay = _draw_predictions(
                            pseudo_base,
                            spacing_pseudo_boxes,
                            (80, 220, 255),
                            label_prefix="C:",
                            max_labels=0 if args.single_panel else 20,
                            annotation_scale=args.single_panel_box_scale if args.single_panel else 1.0,
                        )
                        if not args.single_panel:
                            spacing_tiles = [
                                _make_tile(original_pil, "Input", args.tile_size),
                                _make_tile(dino_mask, "DINO Mask", args.tile_size),
                                _make_tile(cls_fused_img, "Mask-Fused P3 Cls Score", args.tile_size),
                                _make_tile(cls_delta_centers, "Cls Seeds A=green M=orange", args.tile_size),
                                _make_tile(spacing_overlay, "Missing center=yellow box=cyan", args.tile_size),
                                _make_tile(spacing_prior, "Spacing Prior", args.tile_size),
                                _make_tile(mask_fused_overlay, "Mask-Fused Boxes", args.tile_size),
                                _make_tile(pseudo_box_overlay, "Mask-Fused + Completed Boxes", args.tile_size),
                            ]
                            spacing_panel = _make_grid(spacing_tiles, cols=4)
                            spacing_complete_path = output_dir / f"{sample_count:04d}_{stem}_spacing_complete.png"
                            spacing_panel.save(spacing_complete_path)
                if args.single_panel:
                    level_label = _feature_level_label(dino_mask_fuse_level)
                    raw_pred_overlay = _draw_predictions(
                        original_pil,
                        after_preds,
                        (0, 245, 255),
                        label_prefix="Y:",
                        max_labels=0,
                        annotation_scale=args.single_panel_box_scale,
                        line_width_scale=0.45,
                        shadow=False,
                    )
                    mask_fused_pred_overlay = _draw_predictions(
                        original_pil,
                        mask_fused_preds,
                        (255, 35, 210),
                        label_prefix="D:",
                        max_labels=0,
                        annotation_scale=args.single_panel_box_scale,
                        line_width_scale=0.45,
                        shadow=False,
                    )
                    compact_delta_overlay = _draw_prediction_difference(
                        original_pil,
                        after_preds,
                        mask_fused_preds,
                        annotation_scale=args.single_panel_box_scale,
                    )
                    compact_delta_img = cls_delta_img if cls_delta_img is not None else response_delta
                    compact_delta_title = (
                        f"{level_label} Head Delta" if cls_delta_img is not None else f"{level_label} Feature Delta"
                    )
                    compact_tiles = [
                        _make_tile(original_pil, "Origin Image", args.tile_size),
                        _make_tile(before_response, f"YOLO {level_label} Feature", args.tile_size),
                        _make_tile(dino_mask, "DINO Mask", args.tile_size),
                        _make_tile(fused_response, f"DINO x YOLO {level_label}", args.tile_size),
                        _make_tile(raw_pred_overlay, "Raw YOLO Boxes cyan", args.tile_size),
                        _make_tile(mask_fused_pred_overlay, "DINO x YOLO Boxes magenta", args.tile_size),
                        _make_tile(compact_delta_overlay, "Prediction Difference", args.tile_size),
                        _make_tile(compact_delta_img, compact_delta_title, args.tile_size),
                    ]
                    if args.spacing_complete:
                        if cls_delta_centers is not None:
                            compact_tiles.append(
                                _make_tile(cls_delta_centers, "Candidates green=YOLO orange=DINOx", args.tile_size)
                            )
                        if spacing_overlay is not None:
                            compact_tiles.append(
                                _make_tile(spacing_overlay, "Missing Center yellow + pseudo cyan", args.tile_size)
                            )
                        if spacing_prior is not None:
                            compact_tiles.append(_make_tile(spacing_prior, "Spacing Prior", args.tile_size))
                        if pseudo_box_overlay is not None and spacing_pseudo_boxes:
                            compact_tiles.append(
                                _make_tile(pseudo_box_overlay, "DINOxYOLO + Completed Boxes", args.tile_size)
                            )
                    compact_panel = _make_grid(compact_tiles, cols=max(1, int(args.single_panel_cols)))
                    compact_panel_path = output_dir / f"{sample_count:04d}_{stem}_panel.png"
                    compact_panel.save(compact_panel_path)
                else:
                    tiles.extend(
                        [
                            _make_tile(
                                fused_response,
                                f"Mask-Fused {dino_mask_fuse_level.upper()} Resp",
                                args.tile_size,
                            ),
                            _make_tile(mask_fused_overlay, "After + DINO Mask Fuse", args.tile_size),
                        ]
                    )
            if args.single_panel:
                if compact_panel_path is None:
                    raise RuntimeError("--single-panel requires DINO mask fuse panel generation")
                panel_path = compact_panel_path
            else:
                panel = _make_grid(tiles, cols=4)
                panel_path = output_dir / f"{sample_count:04d}_{stem}.png"
                panel.save(panel_path)

            box_metrics = _prediction_delta_metrics(before_preds, after_preds)
            row: Dict[str, object] = {
                "index": sample_count,
                "image_path": image_path,
                "panel_path": str(panel_path),
                "dino_mask_path": str(mask_path) if mask_path is not None else "",
                "dino_mask_overlay_path": str(mask_overlay_path) if mask_overlay_path is not None else "",
                "gate_compare_path": str(gate_compare_path) if gate_compare_path is not None else "",
                "fuse_compare_path": str(fuse_compare_path) if fuse_compare_path is not None else "",
                "mask_fuse_feature_path": (
                    str(mask_fuse_feature_path) if mask_fuse_feature_path is not None else ""
                ),
                "mask_fuse_head_path": str(mask_fuse_head_path) if mask_fuse_head_path is not None else "",
                "spacing_complete_path": str(spacing_complete_path) if spacing_complete_path is not None else "",
                "feature_level": feature_level,
                "cka_before_yolo_dino": cka_before,
                "cka_after_yolo_dino": cka_after,
                "cka_delta_after_minus_before": cka_after - cka_before,
                "dino_mask_component": dino_mask_stats["component"],
                "dino_mask_inverted": dino_mask_stats["inverted"],
                "dino_mask_threshold": dino_mask_stats["threshold"],
                "dino_mask_grid_coverage": dino_mask_stats["coverage"],
                "dino_mask_display_coverage": dino_mask_stats["display_coverage"],
                "dino_mask_num_components": dino_mask_stats["num_components"],
                **box_metrics,
            }
            if args.dino_gate:
                gate_metrics = _prediction_delta_metrics(after_preds, gated_preds)
                row.update(
                    {
                        "dino_gate_level": dino_gate_level,
                        "dino_gate_min": args.dino_gate_min,
                        "dino_gate_max": args.dino_gate_max,
                        "gated_box_count": float(len(gated_preds)),
                        "after_gated_matched_box_count": gate_metrics["matched_box_count"],
                        "mean_after_gated_aabb_iou": gate_metrics["mean_matched_aabb_iou"],
                        "mean_after_gated_center_shift_px": gate_metrics["mean_center_shift_px"],
                    }
                )
            if args.dino_fuse:
                fuse_metrics = _prediction_delta_metrics(after_preds, fused_preds)
                row.update(
                    {
                        "dino_fuse_level": dino_fuse_level,
                        "dino_fuse_mode": args.dino_fuse_mode,
                        "dino_fuse_strength": args.dino_fuse_strength,
                        "dino_fuse_components": args.dino_fuse_components,
                        "fused_box_count": float(len(fused_preds)),
                        "after_fused_matched_box_count": fuse_metrics["matched_box_count"],
                        "mean_after_fused_aabb_iou": fuse_metrics["mean_matched_aabb_iou"],
                        "mean_after_fused_center_shift_px": fuse_metrics["mean_center_shift_px"],
                    }
                )
            if args.dino_mask_fuse:
                mask_fuse_metrics = _prediction_delta_metrics(after_preds, mask_fused_preds)
                mask_fuse_row = {
                    "dino_mask_fuse_level": dino_mask_fuse_level,
                    "dino_mask_fuse_bg": args.dino_mask_fuse_bg,
                    "dino_mask_fuse_fg": args.dino_mask_fuse_fg,
                    "mask_fused_box_count": float(len(mask_fused_preds)),
                    "after_mask_fused_matched_box_count": mask_fuse_metrics["matched_box_count"],
                    "mean_after_mask_fused_aabb_iou": mask_fuse_metrics["mean_matched_aabb_iou"],
                    "mean_after_mask_fused_center_shift_px": mask_fuse_metrics["mean_center_shift_px"],
                }
                if cls_before_stats is not None and cls_fused_stats is not None:
                    mask_fuse_row.update(
                        {
                            "head_cls_before_max": cls_before_stats["max"],
                            "head_cls_fused_max": cls_fused_stats["max"],
                            "head_cls_before_mean": cls_before_stats["mean"],
                            "head_cls_fused_mean": cls_fused_stats["mean"],
                            "head_cls_before_cells_ge_threshold": cls_before_stats["cells_ge_threshold"],
                            "head_cls_fused_cells_ge_threshold": cls_fused_stats["cells_ge_threshold"],
                        }
                    )
                if spacing_stats is not None:
                    mask_fuse_row.update(spacing_stats)
                row.update(mask_fuse_row)
            rows.append(row)
            print(f"[dino-effect] saved panel={panel_path}", flush=True)
            if spacing_complete_path is not None:
                print(f"[dino-effect] saved spacing={spacing_complete_path}", flush=True)
            sample_count += 1

        if sample_count >= args.num_samples:
            break

    if not rows:
        raise RuntimeError("No samples were generated")

    _save_metrics_csv(output_dir / "metrics.csv", rows)
    summary = {
        "config": str(config_path),
        "before_weights": str(Path(before_weights).resolve()),
        "after_weights": str(Path(after_weights).resolve()),
        "split": args.split if not args.image_dir and not args.image else None,
        "image": args.image,
        "image_dir": args.image_dir,
        "num_samples": len(rows),
        "preset": args.preset,
        "feature_level": feature_level,
        "annotation_scale": args.annotation_scale,
        "dino_gate": bool(args.dino_gate),
        "dino_gate_level": dino_gate_level if args.dino_gate else None,
        "dino_gate_min": args.dino_gate_min if args.dino_gate else None,
        "dino_gate_max": args.dino_gate_max if args.dino_gate else None,
        "dino_fuse": bool(args.dino_fuse),
        "dino_fuse_level": dino_fuse_level if args.dino_fuse else None,
        "dino_fuse_mode": args.dino_fuse_mode if args.dino_fuse else None,
        "dino_fuse_strength": args.dino_fuse_strength if args.dino_fuse else None,
        "dino_fuse_components": args.dino_fuse_components if args.dino_fuse else None,
        "dino_mask_fuse": bool(args.dino_mask_fuse),
        "dino_mask_fuse_level": dino_mask_fuse_level if args.dino_mask_fuse else None,
        "dino_mask_fuse_bg": args.dino_mask_fuse_bg if args.dino_mask_fuse else None,
        "dino_mask_fuse_fg": args.dino_mask_fuse_fg if args.dino_mask_fuse else None,
        "spacing_complete": bool(args.spacing_complete),
        "spacing_anchor_source": args.spacing_anchor_source if args.spacing_complete else None,
        "spacing_score_threshold": args.spacing_score_threshold if args.spacing_complete else None,
        "spacing_proposal_score_threshold": args.spacing_proposal_score_threshold if args.spacing_complete else None,
        "spacing_dino_threshold": args.spacing_dino_threshold if args.spacing_complete else None,
        "spacing_row_tolerance_px": args.spacing_row_tolerance_px if args.spacing_complete else None,
        "spacing_period_prior_px": args.spacing_period_prior_px if args.spacing_complete else None,
        "spacing_period_prior_ratio": args.spacing_period_prior_ratio if args.spacing_complete else None,
        "spacing_period_scale": args.spacing_period_scale if args.spacing_complete else None,
        "spacing_prior_bank_json": args.spacing_prior_bank_json if args.spacing_complete else None,
        "spacing_prior_bank_names": (
            [str(item.get("name", "")) for item in spacing_prior_bank]
            if args.spacing_complete and spacing_prior_bank
            else None
        ),
        "spacing_prior_bank_sigma": args.spacing_prior_bank_sigma if args.spacing_complete else None,
        "spacing_prior_min_weight": args.spacing_prior_min_weight if args.spacing_complete else None,
        "spacing_row_period_prior_px": args.spacing_row_period_prior_px if args.spacing_complete else None,
        "spacing_row_period_prior_ratio": args.spacing_row_period_prior_ratio if args.spacing_complete else None,
        "spacing_row_period_scale": args.spacing_row_period_scale if args.spacing_complete else None,
        "spacing_row_prior_tolerance_px": args.spacing_row_prior_tolerance_px if args.spacing_complete else None,
        "spacing_complete_missing_rows": bool(args.spacing_complete_missing_rows) if args.spacing_complete else None,
        "spacing_refine_window_px": args.spacing_refine_window_px if args.spacing_complete else None,
        "spacing_refine_step_px": args.spacing_refine_step_px if args.spacing_complete else None,
        "spacing_refine_prior_sigma_px": args.spacing_refine_prior_sigma_px if args.spacing_complete else None,
        "spacing_draw_pseudo_boxes": bool(args.spacing_draw_pseudo_boxes) if args.spacing_complete else None,
        "spacing_pseudo_overlap_iou": args.spacing_pseudo_overlap_iou if args.spacing_complete else None,
        "spacing_pseudo_row_frequency_ratio": (
            args.spacing_pseudo_row_frequency_ratio if args.spacing_complete else None
        ),
        "spacing_pseudo_row_frequency_min_points": (
            args.spacing_pseudo_row_frequency_min_points if args.spacing_complete else None
        ),
        "spacing_pseudo_min_anchor_per_row": (
            args.spacing_pseudo_min_anchor_per_row if args.spacing_complete else None
        ),
        "spacing_pseudo_anchor_row_tolerance_px": (
            args.spacing_pseudo_anchor_row_tolerance_px if args.spacing_complete else None
        ),
        "mean_cka_before_yolo_dino": float(np.mean([float(r["cka_before_yolo_dino"]) for r in rows])),
        "mean_cka_after_yolo_dino": float(np.mean([float(r["cka_after_yolo_dino"]) for r in rows])),
        "mean_cka_delta_after_minus_before": float(
            np.mean([float(r["cka_delta_after_minus_before"]) for r in rows])
        ),
        "mean_before_box_count": float(np.mean([float(r["before_box_count"]) for r in rows])),
        "mean_after_box_count": float(np.mean([float(r["after_box_count"]) for r in rows])),
        "mean_matched_aabb_iou": float(np.mean([float(r["mean_matched_aabb_iou"]) for r in rows])),
        "mean_center_shift_px": float(np.mean([float(r["mean_center_shift_px"]) for r in rows])),
        "output_dir": str(output_dir),
    }
    if args.dino_gate:
        summary.update(
            {
                "mean_gated_box_count": float(np.mean([float(r["gated_box_count"]) for r in rows])),
                "mean_after_gated_aabb_iou": float(
                    np.mean([float(r["mean_after_gated_aabb_iou"]) for r in rows])
                ),
                "mean_after_gated_center_shift_px": float(
                    np.mean([float(r["mean_after_gated_center_shift_px"]) for r in rows])
                ),
            }
        )
    if args.dino_fuse:
        summary.update(
            {
                "mean_fused_box_count": float(np.mean([float(r["fused_box_count"]) for r in rows])),
                "mean_after_fused_aabb_iou": float(
                    np.mean([float(r["mean_after_fused_aabb_iou"]) for r in rows])
                ),
                "mean_after_fused_center_shift_px": float(
                    np.mean([float(r["mean_after_fused_center_shift_px"]) for r in rows])
                ),
            }
        )
    if args.dino_mask_fuse:
        summary.update(
            {
                "mean_mask_fused_box_count": float(
                    np.mean([float(r["mask_fused_box_count"]) for r in rows])
                ),
                "mean_after_mask_fused_aabb_iou": float(
                    np.mean([float(r["mean_after_mask_fused_aabb_iou"]) for r in rows])
                ),
                "mean_after_mask_fused_center_shift_px": float(
                    np.mean([float(r["mean_after_mask_fused_center_shift_px"]) for r in rows])
                ),
            }
        )
    if args.spacing_complete:
        spacing_counts = [float(r.get("spacing_candidate_count", 0.0)) for r in rows]
        spacing_periods = [float(r.get("spacing_mean_period_px", 0.0)) for r in rows]
        spacing_row_periods = [float(r.get("spacing_row_period_px", 0.0)) for r in rows]
        spacing_missing_rows = [float(r.get("spacing_missing_row_count", 0.0)) for r in rows]
        spacing_anchor_row_removed = [float(r.get("spacing_pseudo_anchor_row_removed", 0.0)) for r in rows]
        spacing_overlap_removed = [float(r.get("spacing_pseudo_overlap_removed", 0.0)) for r in rows]
        spacing_frequency_removed = [float(r.get("spacing_pseudo_frequency_removed", 0.0)) for r in rows]
        summary.update(
            {
                "mean_spacing_candidate_count": float(np.mean(spacing_counts)) if spacing_counts else 0.0,
                "mean_spacing_period_px": float(np.mean(spacing_periods)) if spacing_periods else 0.0,
                "mean_spacing_row_period_px": float(np.mean(spacing_row_periods)) if spacing_row_periods else 0.0,
                "mean_spacing_missing_row_count": float(np.mean(spacing_missing_rows)) if spacing_missing_rows else 0.0,
                "mean_spacing_pseudo_anchor_row_removed": (
                    float(np.mean(spacing_anchor_row_removed)) if spacing_anchor_row_removed else 0.0
                ),
                "mean_spacing_pseudo_overlap_removed": (
                    float(np.mean(spacing_overlap_removed)) if spacing_overlap_removed else 0.0
                ),
                "mean_spacing_pseudo_frequency_removed": (
                    float(np.mean(spacing_frequency_removed)) if spacing_frequency_removed else 0.0
                ),
            }
        )
    _save_json(output_dir / "summary.json", summary)
    print(f"[dino-effect] saved metrics={output_dir / 'metrics.csv'}", flush=True)
    print(f"[dino-effect] saved summary={output_dir / 'summary.json'}", flush=True)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
