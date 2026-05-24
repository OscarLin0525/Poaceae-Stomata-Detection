#!/usr/bin/env python3
"""
Prediction-first visualization for before/after feature alignment.

This standalone script does not modify MTKD training. It generates two panels
per image:

1. Feature panel:
   - original image
   - YOLO feature map
   - DINO RMS feature map
   - DINO PCA
   - pattern prior/support
   - pattern-enhanced YOLO feature map

2. Prediction panel:
   - before-alignment YOLO prediction
   - after-alignment YOLO prediction
   - before/after prediction difference
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from sklearn.decomposition import NMF, PCA


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
    _draw_predictions,
    _feature_level_label,
    _feature_response_image,
    _find_detect_module,
    _head_cls_score_map,
    _local_max_points,
    _make_grid,
    _predict_one,
    _prediction_delta_metrics,
)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Visualize YOLO features and predictions before/after feature alignment.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--config", type=str, required=True, help="Training config JSON/YAML path")
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
        help="Aligned/exported YOLO .pt, e.g. outputs/.../student_best.pt",
    )
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
        help="Which YOLO checkpoint provides the feature-map panel's YOLO feature.",
    )
    p.add_argument("--conf", type=float, default=0.25)
    p.add_argument("--iou", type=float, default=0.7)
    p.add_argument("--max-det", type=int, default=300)
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--output-dir", type=str, default=None)
    p.add_argument("--tile-size", type=int, default=420)
    p.add_argument("--annotation-scale", type=float, default=1.0)
    p.add_argument("--box-scale", type=float, default=0.45)
    p.add_argument("--mask-target-coverage", type=float, default=0.12)
    p.add_argument("--mask-min-coverage", type=float, default=0.02)
    p.add_argument("--mask-max-coverage", type=float, default=0.30)
    p.add_argument("--mask-fuse-bg", type=float, default=0.5)
    p.add_argument("--mask-fuse-fg", type=float, default=1.8)
    p.add_argument("--pattern-enhance", action="store_true", help="Apply row/period-aware feature completion before prediction")
    p.add_argument(
        "--pattern-prior-levels",
        type=str,
        default="p3,p4",
        help="Comma-separated YOLO pyramid levels used to build the pattern prior",
    )
    p.add_argument("--pattern-seed-threshold", type=float, default=0.01)
    p.add_argument("--pattern-seed-topk", type=int, default=96)
    p.add_argument("--pattern-gate-threshold", type=float, default=0.05)
    p.add_argument("--pattern-row-tolerance-px", type=float, default=42.0)
    p.add_argument("--pattern-min-row-seeds", type=int, default=2)
    p.add_argument("--pattern-line-max-slope", type=float, default=0.18)
    p.add_argument("--pattern-period-prior-px", type=float, default=0.0)
    p.add_argument("--pattern-period-prior-ratio", type=float, default=1.8)
    p.add_argument("--pattern-period-scale", type=float, default=1.0)
    p.add_argument("--pattern-row-sigma-scale", type=float, default=0.45)
    p.add_argument("--pattern-period-sigma-scale", type=float, default=0.28)
    p.add_argument("--pattern-prior-strength", type=float, default=0.72)
    p.add_argument("--pattern-completion-strength", type=float, default=0.45)
    p.add_argument("--pattern-completion-gamma", type=float, default=1.0)
    p.add_argument("--pattern-noise-suppress", type=float, default=0.18)
    p.add_argument("--pattern-cross-row-strength", type=float, default=0.24)
    p.add_argument("--pattern-cross-row-noise-suppress", type=float, default=0.10)
    p.add_argument("--pattern-response-period-blend", type=float, default=0.60)
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
    p.add_argument("--nmf-guide", action="store_true", help="Use top-scoring DINO NMF component(s) as a soft guide for YOLO feature gating")
    p.add_argument("--nmf-guide-components", type=int, default=6, help="How many NMF components to fit on DINO tokens")
    p.add_argument("--nmf-guide-topk", type=int, default=1, help="How many top-scoring NMF components to merge into the guide")
    p.add_argument("--nmf-guide-strength", type=float, default=0.55, help="How strongly the NMF guide boosts the feature support")
    p.add_argument("--nmf-guide-init", type=str, default="nndsvda", choices=["random", "nndsvd", "nndsvda", "nndsvdar"])
    p.add_argument("--nmf-guide-max-iter", type=int, default=1200)
    p.add_argument("--nmf-guide-image-size", type=int, default=518, help="Explorer-matching square resize used to build the NMF guide")
    p.add_argument(
        "--nmf-guide-score-dir",
        type=str,
        default=None,
        help="Optional directory containing *_component_scores.json from dino_component_explorer; if present, force the selected best NMF component",
    )
    p.add_argument("--pca-guide", action="store_true", help="Use top-scoring DINO PCA component as a soft guide for YOLO feature gating")
    p.add_argument("--pca-guide-strength", type=float, default=0.55, help="How strongly the PCA guide boosts the feature support")
    p.add_argument("--pca-guide-image-size", type=int, default=518, help="Explorer-matching square resize used to build the PCA guide")
    p.add_argument(
        "--pca-guide-score-dir",
        type=str,
        default=None,
        help="Optional directory containing *_component_scores.json from dino_component_explorer; if present, force the selected best PCA component",
    )
    return p.parse_args()


def _resolve_weight_path(
    explicit: Optional[str],
    fallback: Optional[str],
    *,
    root: Path,
) -> Path:
    raw = explicit or fallback
    if not raw:
        raise ValueError("Missing weights path")
    candidate = Path(raw)
    if candidate.is_file():
        return candidate.resolve()

    parts = candidate.parts
    tail_lengths = [4, 3, 2, 1]
    for tail_len in tail_lengths:
        if len(parts) < tail_len:
            continue
        suffix = Path(*parts[-tail_len:])
        matches = list(root.rglob(str(suffix)))
        files = [match for match in matches if match.is_file()]
        if len(files) == 1:
            return files[0].resolve()

    basename_matches = [match for match in root.rglob(candidate.name) if match.is_file()]
    if len(basename_matches) == 1:
        return basename_matches[0].resolve()

    raise FileNotFoundError(f"Could not resolve weights path: {raw}")


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
        yield {"images": images, "image_paths": [str(path) for path in chunk]}


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
    weights: Path,
    config: Dict,
    feature_level: str,
    device: torch.device,
) -> YOLOStudentDetector:
    num_classes = int(config.get("model", {}).get("num_classes", 1))
    student = YOLOStudentDetector(
        weights=str(weights),
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


@torch.no_grad()
def _extract_yolo_pyramid(
    student: YOLOStudentDetector,
    images: torch.Tensor,
) -> Dict[str, torch.Tensor]:
    out = student(images, return_features=True, return_adapted_features=False)
    pyramid: Dict[str, torch.Tensor] = {}
    for level in ("p3", "p4", "p5"):
        key = f"{level}_features"
        feat = out.get(key)
        if feat is not None:
            pyramid[level] = feat.detach()
    if not pyramid:
        raise RuntimeError("YOLO output did not contain any pyramid feature maps")
    return pyramid


def _save_json(path: Path, data: Dict[str, object]) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def _candidate_component_score_stems(image_path: str) -> List[str]:
    stem = Path(image_path).stem
    candidates: List[str] = [stem]
    if stem.startswith("result_"):
        candidates.append(stem[len("result_") :])
    candidates.append(stem.removeprefix("result_"))
    unique: List[str] = []
    seen = set()
    for item in candidates:
        if item and item not in seen:
            seen.add(item)
            unique.append(item)
    return unique


def _load_forced_nmf_indices(image_path: str, score_dir: Path, topk: int) -> Tuple[List[int], Dict[str, float]]:
    stats: Dict[str, float] = {}
    if not score_dir.is_dir():
        return [], stats
    score_path: Optional[Path] = None
    for stem in _candidate_component_score_stems(image_path):
        exact = score_dir / f"{stem}_component_scores.json"
        if exact.is_file():
            score_path = exact
            break
    if score_path is None:
        for stem in _candidate_component_score_stems(image_path):
            matches = sorted(score_dir.glob(f"*{stem}*_component_scores.json"))
            if matches:
                score_path = matches[0]
                break
    if score_path is None:
        return [], stats
    try:
        with score_path.open("r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        return [], stats
    ranked = data.get("ranked_candidates", [])
    nmf_ranked = [row for row in ranked if str(row.get("family", "")).upper() == "NMF"]
    selected = nmf_ranked[: max(1, int(topk))]
    indices = [int(row.get("component_index", 0)) for row in selected if int(row.get("component_index", 0)) > 0]
    if not indices:
        best = data.get("best_nmf", {})
        idx = int(best.get("component_index", 0)) if isinstance(best, dict) else 0
        if idx > 0:
            indices = [idx]
            selected = [best] if isinstance(best, dict) else []
    if selected:
        first = selected[0]
        stats["nmf_forced_best_index"] = float(int(first.get("component_index", 0)))
        stats["nmf_forced_best_score"] = float(first.get("score", first.get("metrics", {}).get("score", 0.0)))
    stats["nmf_forced_selected_count"] = float(len(indices))
    return indices, stats


def _load_forced_pca_component(image_path: str, score_dir: Path) -> Tuple[Optional[int], str, Dict[str, float]]:
    stats: Dict[str, float] = {}
    if not score_dir.is_dir():
        return None, "pos", stats
    score_path: Optional[Path] = None
    for stem in _candidate_component_score_stems(image_path):
        exact = score_dir / f"{stem}_component_scores.json"
        if exact.is_file():
            score_path = exact
            break
    if score_path is None:
        for stem in _candidate_component_score_stems(image_path):
            matches = sorted(score_dir.glob(f"*{stem}*_component_scores.json"))
            if matches:
                score_path = matches[0]
                break
    if score_path is None:
        return None, "pos", stats
    try:
        with score_path.open("r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        return None, "pos", stats
    best = data.get("best_pca", {})
    if not isinstance(best, dict):
        return None, "pos", stats
    idx = int(best.get("component_index", 0))
    sign = str(best.get("sign", "pos")).lower()
    if idx <= 0:
        return None, sign, stats
    stats["pca_forced_best_index"] = float(idx)
    stats["pca_forced_best_score"] = float(best.get("score", best.get("metrics", {}).get("score", 0.0)))
    return idx, sign, stats


def _load_image_prior_index(path: Optional[Path]) -> Dict[str, Dict[str, object]]:
    if path is None or not path.is_file():
        return {}
    try:
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        return {}
    images = data.get("images", [])
    if not isinstance(images, list):
        return {}
    index: Dict[str, Dict[str, object]] = {}
    for item in images:
        if not isinstance(item, dict):
            continue
        image_name = str(item.get("image") or Path(str(item.get("annotated_image", ""))).name)
        if not image_name:
            continue
        for stem in _candidate_component_score_stems(image_name):
            index[stem] = item
    return index


def _load_species_prior_bank(path: Optional[Path]) -> Dict[str, Dict[str, float]]:
    if path is None or not path.is_file():
        return {}
    try:
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        return {}
    priors = data.get("priors", [])
    if not isinstance(priors, list):
        return {}
    bank: Dict[str, Dict[str, float]] = {}
    for item in priors:
        if not isinstance(item, dict):
            continue
        name = str(item.get("name", "")).strip()
        if not name:
            continue
        bank[name] = {
            "image_width": float(item.get("image_width", 0.0) or 0.0),
            "image_height": float(item.get("image_height", 0.0) or 0.0),
            "x_period_px": float(item.get("x_period_px", 0.0) or 0.0),
            "row_period_px": float(item.get("row_period_px", 0.0) or 0.0),
        }
    return bank


def _infer_species_prior_name(
    image_path: str,
    *,
    image_size: tuple[int, int],
    bank: Dict[str, Dict[str, float]],
) -> Optional[str]:
    if not bank:
        return None
    text = image_path.lower()
    stem = Path(image_path).stem.lower()
    if "rice" in text or stem.startswith("202112"):
        return "rice_annotate" if "rice_annotate" in bank else None
    if "wheat" in text or "tl" in stem:
        return "wheat10_label" if "wheat10_label" in bank else None
    if "barley" in text or stem.startswith("202408"):
        return "barley20_label" if "barley20_label" in bank else None

    img_w, img_h = float(image_size[0]), float(image_size[1])
    best_name: Optional[str] = None
    best_score = float("inf")
    for name, prior in bank.items():
        dw = abs(float(prior.get("image_width", 0.0)) - img_w)
        dh = abs(float(prior.get("image_height", 0.0)) - img_h)
        score = dw + dh
        if score < best_score:
            best_score = score
            best_name = name
    return best_name


def _resolve_external_pattern_prior(
    image_path: str,
    *,
    image_size: tuple[int, int],
    image_prior_index: Dict[str, Dict[str, object]],
    species_prior_bank: Dict[str, Dict[str, float]],
) -> Dict[str, object]:
    resolved: Dict[str, object] = {
        "source": "none",
        "family": "",
        "x_period_px": 0.0,
        "row_period_px": 0.0,
        "row_tolerance_px": 0.0,
        "row_centers_px": [],
    }
    for stem in _candidate_component_score_stems(image_path):
        item = image_prior_index.get(stem)
        if not isinstance(item, dict):
            continue
        rows = item.get("rows", [])
        if not isinstance(rows, list):
            rows = []
        good_rows = [row for row in rows if isinstance(row, dict) and float(row.get("count", 0.0) or 0.0) >= 3.0]
        periods = [
            float(row.get("x_period_median_gap", 0.0) or 0.0)
            for row in good_rows
            if float(row.get("x_period_median_gap", 0.0) or 0.0) > 3.0
        ]
        row_centers = [
            float(row.get("row_y", 0.0) or 0.0)
            for row in rows
            if isinstance(row, dict) and float(row.get("count", 0.0) or 0.0) >= 2.0
        ]
        row_centers = sorted(y for y in row_centers if y >= 0.0)
        row_gap = float(item.get("row_gap_median", 0.0) or 0.0)
        resolved.update(
            {
                "source": "image_priors",
                "family": "rice_annotate",
                "x_period_px": float(np.median(periods)) if periods else 0.0,
                "row_period_px": row_gap if row_gap > 0.0 else _estimate_spacing_px(np.asarray(row_centers, dtype=np.float32)),
                "row_tolerance_px": float(item.get("row_tolerance_px", 0.0) or 0.0),
                "row_centers_px": row_centers,
            }
        )
        return resolved

    prior_name = _infer_species_prior_name(image_path, image_size=image_size, bank=species_prior_bank)
    if prior_name and prior_name in species_prior_bank:
        prior = species_prior_bank[prior_name]
        resolved.update(
            {
                "source": "species_prior_bank",
                "family": prior_name,
                "x_period_px": float(prior.get("x_period_px", 0.0) or 0.0),
                "row_period_px": float(prior.get("row_period_px", 0.0) or 0.0),
                "row_tolerance_px": max(float(prior.get("row_period_px", 0.0) or 0.0) * 0.22, 18.0),
                "row_centers_px": [],
            }
        )
    return resolved


def _overlay_component_on_image(image: Image.Image, values: np.ndarray, threshold: float) -> Image.Image:
    norm = _normalize_pos_percentile(values)
    mask = norm >= float(threshold)
    mask_img = Image.fromarray((mask.astype(np.uint8) * 255), mode="L").resize(image.size, resample=Image.Resampling.NEAREST)
    base = np.asarray(image, dtype=np.float32)
    overlay = base.copy()
    alpha = (np.asarray(mask_img, dtype=np.float32) / 255.0)[..., None] * 0.58
    color = np.asarray([255.0, 64.0, 32.0], dtype=np.float32)[None, None, :]
    overlay = overlay * (1.0 - alpha) + color * alpha
    return Image.fromarray(np.clip(overlay, 0.0, 255.0).astype(np.uint8), mode="RGB")


def _normalize_pos(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float32)
    arr = np.where(np.isfinite(arr), arr, 0.0)
    arr = arr - float(arr.min(initial=0.0))
    vmax = float(arr.max(initial=0.0))
    if vmax <= 1e-8:
        return np.zeros_like(arr, dtype=np.float32)
    return (arr / vmax).astype(np.float32)


def _normalize_pos_percentile(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float32)
    lo = float(np.percentile(arr, 2))
    hi = float(np.percentile(arr, 98))
    if not np.isfinite(lo):
        lo = float(arr.min(initial=0.0))
    if not np.isfinite(hi):
        hi = float(arr.max(initial=0.0))
    if hi <= lo:
        return np.zeros_like(arr, dtype=np.float32)
    return np.clip((arr - lo) / (hi - lo), 0.0, 1.0).astype(np.float32)


def _fit_nmf_tokens(tokens: np.ndarray, n_components: int, init: str, max_iter: int) -> Tuple[np.ndarray, np.ndarray, float]:
    shifted = np.asarray(tokens, dtype=np.float32)
    shifted = shifted - shifted.min(axis=0, keepdims=True)
    shifted = np.clip(shifted, 0.0, None)
    shifted += 1e-6
    n_components = int(max(1, min(int(n_components), shifted.shape[0], shifted.shape[1])))
    nmf = NMF(
        n_components=n_components,
        init=init,
        max_iter=int(max_iter),
        random_state=0,
    )
    activations = nmf.fit_transform(shifted).astype(np.float32)
    basis = np.asarray(nmf.components_, dtype=np.float32)
    return activations, basis, float(nmf.reconstruction_err_)


def _fit_pca_tokens(tokens: np.ndarray, n_components: int) -> Tuple[np.ndarray, np.ndarray]:
    n_components = int(max(1, min(int(n_components), tokens.shape[0], tokens.shape[1])))
    pca = PCA(n_components=n_components)
    coords = pca.fit_transform(np.asarray(tokens, dtype=np.float32))
    explained = np.asarray(pca.explained_variance_ratio_, dtype=np.float32)
    return coords.astype(np.float32), explained


def _component_share(values: np.ndarray) -> np.ndarray:
    total = float(np.sum(values))
    if total <= 1e-8:
        return np.zeros((values.shape[1],), dtype=np.float32)
    return (values.sum(axis=0) / total).astype(np.float32)


def _connected_component_areas(binary_map: np.ndarray) -> List[int]:
    mask = np.asarray(binary_map, dtype=np.uint8)
    h, w = mask.shape
    visited = np.zeros((h, w), dtype=np.uint8)
    areas: List[int] = []
    for y in range(h):
        for x in range(w):
            if mask[y, x] == 0 or visited[y, x]:
                continue
            stack = [(y, x)]
            visited[y, x] = 1
            area = 0
            while stack:
                cy, cx = stack.pop()
                area += 1
                for ny in range(max(0, cy - 1), min(h, cy + 2)):
                    for nx in range(max(0, cx - 1), min(w, cx + 2)):
                        if mask[ny, nx] == 0 or visited[ny, nx]:
                            continue
                        visited[ny, nx] = 1
                        stack.append((ny, nx))
            areas.append(area)
    return areas


def _count_local_peaks(values: np.ndarray, threshold: float) -> int:
    tensor = torch.from_numpy(np.asarray(values, dtype=np.float32)).unsqueeze(0).unsqueeze(0)
    pooled = F.max_pool2d(tensor, kernel_size=3, stride=1, padding=1)
    peaks = (tensor >= float(threshold)) & (tensor == pooled)
    return int(peaks.sum().item())


def _hoyer_sparsity(values: np.ndarray) -> float:
    flat = np.asarray(values, dtype=np.float32).reshape(-1)
    if flat.size == 0:
        return 0.0
    l1 = float(np.sum(np.abs(flat)))
    l2 = float(np.sqrt(np.sum(flat ** 2)))
    if l2 <= 1e-8:
        return 0.0
    n = float(flat.size)
    return float(np.clip((np.sqrt(n) - l1 / l2) / max(np.sqrt(n) - 1.0, 1e-6), 0.0, 1.0))


def _coverage_score(coverage: float, target: float = 0.06) -> float:
    cov = max(float(coverage), 1e-6)
    tgt = max(float(target), 1e-6)
    score = 1.0 - abs(np.log(cov) - np.log(tgt)) / np.log(8.0)
    return float(np.clip(score, 0.0, 1.0))


def _score_discrete_component(values: np.ndarray) -> Dict[str, float]:
    norm = _normalize_pos_percentile(values)
    h, w = norm.shape
    n = max(h * w, 1)

    active_thresh = max(0.55, float(np.percentile(norm, 88)))
    peak_thresh = max(0.70, float(np.percentile(norm, 94)))
    binary = norm >= active_thresh
    active_pixels = int(binary.sum())
    coverage = float(active_pixels) / float(n)
    areas = _connected_component_areas(binary)
    comp_count = len(areas)
    mean_area = float(np.mean(areas)) if areas else 0.0
    largest_frac = float(max(areas) / max(active_pixels, 1)) if areas else 1.0
    peak_count = _count_local_peaks(norm, threshold=peak_thresh)
    peak_density = float(peak_count) / max(float(active_pixels), 1.0)
    sparsity = _hoyer_sparsity(norm)
    row_sums = binary.sum(axis=1).astype(np.float32)
    col_sums = binary.sum(axis=0).astype(np.float32)
    row_dom = float(row_sums.max() / max(active_pixels, 1)) if active_pixels else 1.0
    col_dom = float(col_sums.max() / max(active_pixels, 1)) if active_pixels else 1.0
    stripe_penalty = max(row_dom, col_dom)

    coverage_term = _coverage_score(coverage, target=0.06)
    component_term = float(np.clip(comp_count / 18.0, 0.0, 1.0))
    peak_term = float(np.clip(peak_density * 10.0, 0.0, 1.0))
    area_term = float(np.clip(1.0 - mean_area / 12.0, 0.0, 1.0))
    compact_term = float(np.clip(1.0 - largest_frac, 0.0, 1.0))

    score = (
        1.35 * sparsity
        + 1.15 * coverage_term
        + 1.00 * peak_term
        + 0.85 * compact_term
        + 0.60 * component_term
        + 0.40 * area_term
        - 0.55 * stripe_penalty
    )
    return {
        "score": float(score),
        "coverage": float(coverage),
        "active_threshold": float(active_thresh),
        "component_count": float(comp_count),
        "peak_count": float(peak_count),
        "largest_component_fraction": float(largest_frac),
        "row_dominance": float(row_dom),
        "col_dominance": float(col_dom),
        "sparsity": float(sparsity),
    }


def _build_nmf_guide_map(
    dino_feat: torch.Tensor,
    *,
    n_components: int,
    topk: int,
    init: str,
    max_iter: int,
    target_size: tuple[int, int],
    forced_component_indices: Optional[Sequence[int]] = None,
) -> tuple[torch.Tensor, Dict[str, float], np.ndarray, float]:
    feat = dino_feat.detach().float().cpu()
    if feat.ndim != 3:
        raise ValueError(f"Expected DINO feature [C,H,W], got {tuple(feat.shape)}")
    c, h, w = feat.shape
    tokens = feat.permute(1, 2, 0).reshape(h * w, c).numpy().astype(np.float32)
    activations, _basis, recon_err = _fit_nmf_tokens(tokens, n_components=n_components, init=init, max_iter=max_iter)
    share = _component_share(activations)
    component_maps = [activations[:, idx].reshape(h, w).astype(np.float32) for idx in range(activations.shape[1])]
    candidates: List[Dict[str, object]] = []
    for idx, component in enumerate(component_maps):
        metrics = _score_discrete_component(component)
        candidates.append(
            {
                "name": f"NMF-{idx + 1}",
                "component_index": int(idx + 1),
                "map": _normalize_pos_percentile(component),
                "share": float(share[idx]) if idx < share.shape[0] else 0.0,
                "metrics": metrics,
            }
        )
    candidates.sort(key=lambda item: float(item["metrics"]["score"]), reverse=True)
    if forced_component_indices:
        forced_set = [int(idx) for idx in forced_component_indices if int(idx) > 0]
        selected = [
            next((cand for cand in candidates if int(cand["component_index"]) == idx), None)
            for idx in forced_set
        ]
        selected = [cand for cand in selected if cand is not None]
        if not selected:
            selected = candidates[: max(1, int(topk))]
    else:
        selected = candidates[: max(1, int(topk))]
    raw_weights = np.asarray(
        [
            max(float(item["metrics"]["score"]), 0.05) * (0.35 + 0.65 * float(item["share"]))
            for item in selected
        ],
        dtype=np.float32,
    )
    weight_sum = float(raw_weights.sum())
    if weight_sum <= 1e-8:
        weights = np.full((len(selected),), 1.0 / max(len(selected), 1), dtype=np.float32)
    else:
        weights = raw_weights / weight_sum
    merged = np.zeros((h, w), dtype=np.float32)
    for weight, item in zip(weights.tolist(), selected):
        merged += float(weight) * np.asarray(item["map"], dtype=np.float32)
    merged = _normalize_pos_percentile(merged)
    merged_t = torch.from_numpy(merged)
    if tuple(merged_t.shape[-2:]) != tuple(target_size):
        merged_t = _resize_spatial_map(merged_t, target_size)
    stats: Dict[str, float] = {
        "nmf_guide_component_count": float(len(component_maps)),
        "nmf_guide_selected_count": float(len(selected)),
        "nmf_guide_reconstruction_err": float(recon_err),
        "nmf_guide_mean": float(merged_t.mean().item()) if merged_t.numel() else 0.0,
        "nmf_guide_max": float(merged_t.max().item()) if merged_t.numel() else 0.0,
    }
    if selected:
        best = selected[0]
        stats["nmf_best_index"] = float(best["component_index"])
        stats["nmf_best_score"] = float(best["metrics"]["score"])
        stats["nmf_best_share"] = float(best["share"])
        stats["nmf_best_coverage"] = float(best["metrics"]["coverage"])
    if len(selected) > 1:
        second = selected[1]
        stats["nmf_second_index"] = float(second["component_index"])
        stats["nmf_second_score"] = float(second["metrics"]["score"])
    primary_map = np.asarray(selected[0]["map"], dtype=np.float32) if selected else np.zeros((h, w), dtype=np.float32)
    primary_thresh = float(selected[0]["metrics"]["active_threshold"]) if selected else 0.5
    return merged_t.float().cpu().clamp(0.0, 1.0), stats, primary_map, primary_thresh


def _build_pca_guide_map(
    dino_feat: torch.Tensor,
    *,
    n_components: int,
    target_size: tuple[int, int],
    forced_component_index: Optional[int] = None,
    forced_sign: str = "pos",
) -> tuple[torch.Tensor, Dict[str, float], np.ndarray, float]:
    feat = dino_feat.detach().float().cpu()
    if feat.ndim != 3:
        raise ValueError(f"Expected DINO feature [C,H,W], got {tuple(feat.shape)}")
    c, h, w = feat.shape
    tokens = feat.permute(1, 2, 0).reshape(h * w, c).numpy().astype(np.float32)
    coords, explained = _fit_pca_tokens(tokens, n_components=n_components)
    component_maps = [coords[:, idx].reshape(h, w).astype(np.float32) for idx in range(coords.shape[1])]
    candidates: List[Dict[str, object]] = []
    for idx, component in enumerate(component_maps):
        for sign_name, signed_values in (("pos", component), ("neg", -component)):
            metrics = _score_discrete_component(signed_values)
            candidates.append(
                {
                    "name": f"PCA-{idx + 1}-{sign_name}",
                    "component_index": int(idx + 1),
                    "sign": sign_name,
                    "map": _normalize_pos_percentile(signed_values),
                    "weight": float(explained[idx]) if idx < explained.shape[0] else 0.0,
                    "metrics": metrics,
                }
            )
    candidates.sort(key=lambda item: float(item["metrics"]["score"]), reverse=True)
    if forced_component_index is not None and int(forced_component_index) > 0:
        forced_sign = "neg" if str(forced_sign).lower() == "neg" else "pos"
        selected = next(
            (
                cand
                for cand in candidates
                if int(cand["component_index"]) == int(forced_component_index) and str(cand["sign"]) == forced_sign
            ),
            None,
        )
    else:
        selected = None
    if selected is None:
        selected = candidates[0]
    primary_map = np.asarray(selected["map"], dtype=np.float32)
    primary_thresh = float(selected["metrics"]["active_threshold"])
    guide_map = torch.from_numpy(primary_map)
    if tuple(guide_map.shape[-2:]) != tuple(target_size):
        guide_map = _resize_spatial_map(guide_map, target_size)
    stats: Dict[str, float] = {
        "pca_guide_component_count": float(len(component_maps)),
        "pca_guide_mean": float(guide_map.mean().item()) if guide_map.numel() else 0.0,
        "pca_guide_max": float(guide_map.max().item()) if guide_map.numel() else 0.0,
        "pca_best_index": float(selected["component_index"]),
        "pca_best_sign_neg": float(1.0 if str(selected["sign"]) == "neg" else 0.0),
        "pca_best_score": float(selected["metrics"]["score"]),
        "pca_best_weight": float(selected["weight"]),
        "pca_best_coverage": float(selected["metrics"]["coverage"]),
    }
    return guide_map.float().cpu().clamp(0.0, 1.0), stats, primary_map, primary_thresh


def _load_image_for_dino(path: Path, image_size: int) -> torch.Tensor:
    image = Image.open(path).convert("RGB")
    image = image.resize((image_size, image_size), resample=Image.Resampling.BILINEAR)
    arr = np.asarray(image, dtype=np.float32) / 255.0
    return torch.from_numpy(arr).permute(2, 0, 1).contiguous()


@torch.no_grad()
def _build_nmf_guide_map_from_image(
    image_path: str,
    *,
    dino: DinoFeatureExtractor,
    device: torch.device,
    image_size: int,
    n_components: int,
    topk: int,
    init: str,
    max_iter: int,
    target_size: tuple[int, int],
    score_dir: Optional[Path] = None,
) -> tuple[torch.Tensor, Dict[str, float], Image.Image]:
    image_tensor = _load_image_for_dino(Path(image_path), image_size).unsqueeze(0).to(device)
    dino_feat = dino(image_tensor).squeeze(0).detach().cpu()
    original_image = Image.open(image_path).convert("RGB")
    forced_indices: List[int] = []
    forced_stats: Dict[str, float] = {}
    if score_dir is not None:
        forced_indices, forced_stats = _load_forced_nmf_indices(image_path, score_dir, topk=topk)
    guide_map, stats, primary_map, primary_thresh = _build_nmf_guide_map(
        dino_feat,
        n_components=n_components,
        topk=topk,
        init=init,
        max_iter=max_iter,
        target_size=target_size,
        forced_component_indices=forced_indices,
    )
    stats["nmf_guide_image_size"] = float(image_size)
    stats.update(forced_stats)
    raw_overlay = _overlay_component_on_image(original_image, primary_map, primary_thresh)
    return guide_map, stats, raw_overlay


@torch.no_grad()
def _build_pca_guide_map_from_image(
    image_path: str,
    *,
    dino: DinoFeatureExtractor,
    device: torch.device,
    image_size: int,
    n_components: int,
    target_size: tuple[int, int],
    score_dir: Optional[Path] = None,
) -> tuple[torch.Tensor, Dict[str, float], Image.Image]:
    image_tensor = _load_image_for_dino(Path(image_path), image_size).unsqueeze(0).to(device)
    dino_feat = dino(image_tensor).squeeze(0).detach().cpu()
    original_image = Image.open(image_path).convert("RGB")
    forced_idx: Optional[int] = None
    forced_sign = "pos"
    forced_stats: Dict[str, float] = {}
    if score_dir is not None:
        forced_idx, forced_sign, forced_stats = _load_forced_pca_component(image_path, score_dir)
    guide_map, stats, primary_map, primary_thresh = _build_pca_guide_map(
        dino_feat,
        n_components=n_components,
        target_size=target_size,
        forced_component_index=forced_idx,
        forced_sign=forced_sign,
    )
    stats["pca_guide_image_size"] = float(image_size)
    stats.update(forced_stats)
    raw_overlay = _overlay_component_on_image(original_image, primary_map, primary_thresh)
    return guide_map, stats, raw_overlay


def _prediction_center_points(predictions: Sequence[Dict[str, object]]) -> np.ndarray:
    points: List[List[float]] = []
    for pred in predictions:
        poly = np.asarray(pred["poly"], dtype=np.float32).reshape(4, 2)
        center = poly.mean(axis=0)
        points.append([float(center[0]), float(center[1]), float(pred.get("conf", 1.0))])
    return np.asarray(points, dtype=np.float32) if points else np.zeros((0, 3), dtype=np.float32)


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


def _parse_feature_levels(raw: str, *, fallback: str) -> List[str]:
    levels: List[str] = []
    for token in str(raw).split(","):
        level = token.strip().lower()
        if level in {"p3", "p4", "p5"} and level not in levels:
            levels.append(level)
    if not levels:
        levels = [str(fallback).lower()]
    return levels


def _resize_spatial_map(values: torch.Tensor, size: tuple[int, int]) -> torch.Tensor:
    tensor = values.detach().float()
    if tensor.ndim == 2:
        tensor = tensor.unsqueeze(0).unsqueeze(0)
    elif tensor.ndim == 3:
        tensor = tensor.unsqueeze(1)
    else:
        raise ValueError(f"Expected 2D/3D map, got shape={tuple(values.shape)}")
    resized = F.interpolate(tensor, size=size, mode="bilinear", align_corners=False)
    return resized.squeeze(0).squeeze(0).cpu()


def _transform_anchor_points(
    points: Optional[np.ndarray],
    *,
    src_size: tuple[int, int],
    dst_size: tuple[int, int],
    keep_aspect: bool,
) -> np.ndarray:
    pts = (
        np.asarray(points, dtype=np.float32).reshape(-1, 3)
        if points is not None and np.asarray(points).size
        else np.zeros((0, 3), dtype=np.float32)
    )
    if pts.size == 0:
        return pts
    src_w, src_h = float(src_size[0]), float(src_size[1])
    dst_w, dst_h = float(dst_size[0]), float(dst_size[1])
    out = pts.copy()
    if keep_aspect:
        scale = min(dst_w / max(src_w, 1e-6), dst_h / max(src_h, 1e-6))
        new_w = src_w * scale
        new_h = src_h * scale
        pad_x = 0.5 * max(dst_w - new_w, 0.0)
        pad_y = 0.5 * max(dst_h - new_h, 0.0)
        out[:, 0] = pts[:, 0] * scale + pad_x
        out[:, 1] = pts[:, 1] * scale + pad_y
    else:
        out[:, 0] = pts[:, 0] * (dst_w / max(src_w, 1e-6))
        out[:, 1] = pts[:, 1] * (dst_h / max(src_h, 1e-6))
    return out.astype(np.float32)


def _estimate_orientation_angle(points: np.ndarray) -> float:
    pts = np.asarray(points, dtype=np.float32)
    if pts.ndim != 2 or pts.shape[0] < 2:
        return 0.0
    coords = pts[:, :2].astype(np.float64)
    weights = pts[:, 2].astype(np.float64) if pts.shape[1] >= 3 else np.ones((pts.shape[0],), dtype=np.float64)
    weights = np.clip(weights, 1e-6, None)
    mean = np.average(coords, axis=0, weights=weights)
    centered = coords - mean[None, :]
    cov = np.zeros((2, 2), dtype=np.float64)
    for idx in range(centered.shape[0]):
        vec = centered[idx : idx + 1].T
        cov += weights[idx] * (vec @ vec.T)
    cov /= max(float(weights.sum()), 1e-6)
    eigvals, eigvecs = np.linalg.eigh(cov)
    direction = eigvecs[:, int(np.argmax(eigvals))]
    angle = float(np.arctan2(direction[1], direction[0]))
    while angle <= -0.5 * np.pi:
        angle += np.pi
    while angle > 0.5 * np.pi:
        angle -= np.pi
    return angle


def _rotate_points(points: np.ndarray, *, center: tuple[float, float], angle_rad: float) -> np.ndarray:
    pts = np.asarray(points, dtype=np.float32)
    if pts.size == 0:
        return pts.copy()
    cx, cy = float(center[0]), float(center[1])
    cos_t = float(np.cos(angle_rad))
    sin_t = float(np.sin(angle_rad))
    shifted = pts[:, :2].astype(np.float32) - np.asarray([cx, cy], dtype=np.float32)[None, :]
    x_new = cos_t * shifted[:, 0] - sin_t * shifted[:, 1] + cx
    y_new = sin_t * shifted[:, 0] + cos_t * shifted[:, 1] + cy
    out = pts.copy()
    out[:, 0] = x_new.astype(np.float32)
    out[:, 1] = y_new.astype(np.float32)
    return out


def _rotate_grid(
    x_grid: torch.Tensor,
    y_grid: torch.Tensor,
    *,
    center: tuple[float, float],
    angle_rad: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    cx, cy = float(center[0]), float(center[1])
    cos_t = float(np.cos(angle_rad))
    sin_t = float(np.sin(angle_rad))
    x_shift = x_grid - cx
    y_shift = y_grid - cy
    x_rot = cos_t * x_shift - sin_t * y_shift + cx
    y_rot = sin_t * x_shift + cos_t * y_shift + cy
    return x_rot, y_rot


def _estimate_period_from_profile(
    profile: np.ndarray,
    *,
    step_px: float,
    min_period_px: float,
    max_period_px: float,
) -> tuple[float, float]:
    values = np.asarray(profile, dtype=np.float32)
    if values.size < 8 or step_px <= 0.0:
        return 0.0, 0.0
    values = values - float(values.mean())
    std = float(values.std())
    if std <= 1e-6:
        return 0.0, 0.0
    values = values / std
    corr = np.correlate(values, values, mode="full")[values.size - 1 :]
    corr[0] = 0.0
    min_lag = max(1, int(round(float(min_period_px) / step_px)))
    max_lag = min(int(values.size - 1), int(round(float(max_period_px) / step_px)))
    if max_lag <= min_lag:
        return 0.0, 0.0
    window = corr[min_lag : max_lag + 1]
    if window.size == 0:
        return 0.0, 0.0
    peak_idx = int(np.argmax(window))
    peak_val = float(window[peak_idx])
    if not np.isfinite(peak_val) or peak_val <= 0.0:
        return 0.0, 0.0
    lag = min_lag + peak_idx
    period_px = float(lag) * float(step_px)
    strength = float(np.clip(peak_val / max(float(corr[0]) + 1e-6, 1.0), 0.0, 1.0))
    return period_px, strength


def _fit_slanted_row(points: np.ndarray, max_abs_slope: float) -> tuple[float, float, float]:
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
            }
        )
    return rows


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


def _maybe_use_period_prior(estimated: float, prior: float, ratio: float) -> float:
    prior = max(float(prior), 0.0)
    if prior <= 0.0:
        return float(estimated)
    ratio = max(float(ratio), 1.01)
    if estimated < 3.0 or estimated > prior * ratio or estimated < prior / ratio:
        return prior
    return float(estimated)


def _normalize_score_map(score_map: torch.Tensor) -> torch.Tensor:
    score_norm = score_map.detach().float().cpu()
    if score_norm.numel():
        score_norm = score_norm - float(score_norm.min().item())
        score_norm = score_norm / max(float(score_norm.max().item()), 1e-6)
    return score_norm


def _aggregate_pattern_stats(stats_list: Sequence[Dict[str, float]]) -> Dict[str, float]:
    if not stats_list:
        return {
            "pattern_seed_count": 0.0,
            "pattern_row_count": 0.0,
            "pattern_used_row_count": 0.0,
            "pattern_row_only_count": 0.0,
            "pattern_cross_row_used": 0.0,
            "pattern_mean_period_px": 0.0,
            "pattern_mean_row_period_px": 0.0,
            "pattern_orientation_deg": 0.0,
            "pattern_response_period_count": 0.0,
            "pattern_prior_level_count": 0.0,
        }
    keys = (
        "pattern_row_count",
        "pattern_used_row_count",
        "pattern_row_only_count",
        "pattern_cross_row_used",
        "pattern_mean_period_px",
        "pattern_mean_row_period_px",
        "pattern_orientation_deg",
        "pattern_response_period_count",
    )
    aggregated = {key: float(np.mean([float(stat.get(key, 0.0)) for stat in stats_list])) for key in keys}
    aggregated["pattern_seed_count"] = float(max(float(stat.get("pattern_seed_count", 0.0)) for stat in stats_list))
    aggregated["pattern_prior_level_count"] = float(len(stats_list))
    return aggregated


def _pattern_prior_from_maps(
    score_map: torch.Tensor,
    gate_map: torch.Tensor,
    *,
    image_size: tuple[int, int],
    anchor_points: Optional[np.ndarray],
    seed_threshold: float,
    seed_topk: int,
    gate_threshold: float,
    row_tolerance_px: float,
    min_row_seeds: int,
    max_abs_slope: float,
    period_prior_px: float,
    period_prior_ratio: float,
    period_scale: float,
    row_sigma_scale: float,
    period_sigma_scale: float,
    row_period_prior_px: float,
    row_center_priors_px: Optional[Sequence[float]],
    cross_row_strength: float,
    response_period_blend: float,
    force_horizontal: bool = False,
    full_row_support: bool = False,
) -> tuple[torch.Tensor, np.ndarray, Dict[str, float]]:
    gate_2d = gate_map.detach().float().cpu()
    if gate_2d.ndim == 3:
        gate_2d = gate_2d.squeeze(0)
    score_2d = score_map.detach().float().cpu()
    h, w = score_2d.shape
    prior = torch.zeros((h, w), dtype=torch.float32)
    local_seed_points = _local_max_points(
        score_2d,
        image_size=image_size,
        threshold=float(seed_threshold),
        topk=int(seed_topk),
        gate=gate_2d,
        gate_threshold=float(gate_threshold),
    )
    anchor_arr = (
        np.asarray(anchor_points, dtype=np.float32).reshape(-1, 3)
        if anchor_points is not None and np.asarray(anchor_points).size
        else np.zeros((0, 3), dtype=np.float32)
    )
    seed_points = _merge_anchor_points(
        [local_seed_points, anchor_arr],
        min_distance_px=max(float(row_tolerance_px) * 0.45, 10.0),
    )
    if seed_points.size == 0:
        return prior, seed_points, {
            "pattern_seed_count": 0.0,
            "pattern_row_count": 0.0,
            "pattern_used_row_count": 0.0,
            "pattern_row_only_count": 0.0,
            "pattern_cross_row_used": 0.0,
            "pattern_mean_period_px": 0.0,
            "pattern_mean_row_period_px": 0.0,
            "pattern_orientation_deg": 0.0,
            "pattern_response_period_count": 0.0,
        }

    img_w, img_h = image_size
    center = (0.5 * float(img_w), 0.5 * float(img_h))
    orientation_rad = 0.0 if bool(force_horizontal) else _estimate_orientation_angle(seed_points)
    seed_points_rot = _rotate_points(seed_points, center=center, angle_rad=-orientation_rad)
    rows = _build_slanted_rows(
        seed_points_rot[:, :2],
        row_tolerance_px=row_tolerance_px,
        max_abs_slope=max_abs_slope,
    )
    xs = (torch.arange(w, dtype=torch.float32) + 0.5) / float(w) * float(img_w)
    ys = (torch.arange(h, dtype=torch.float32) + 0.5) / float(h) * float(img_h)
    x_grid = xs[None, :].repeat(h, 1)
    y_grid = ys[:, None].repeat(1, w)
    x_rot_grid, y_rot_grid = _rotate_grid(x_grid, y_grid, center=center, angle_rad=-orientation_rad)
    used_rows = 0
    periods: List[float] = []
    row_only_count = 0
    response_period_count = 0

    for row in rows:
        row_pts = np.asarray(row["points"], dtype=np.float32)
        if row_pts.shape[0] < int(min_row_seeds):
            continue
        seed_x = np.sort(row_pts[:, 0].astype(np.float32))
        period_seed = _estimate_spacing_px(seed_x)
        period_seed = _maybe_use_period_prior(period_seed, period_prior_px, period_prior_ratio)

        slope = float(row["slope"])
        intercept = float(row["intercept"])
        residual = float(row["residual"])
        row_line = slope * x_rot_grid + intercept
        row_dist = torch.abs(y_rot_grid - row_line)
        row_sigma = max(float(row_tolerance_px) * float(row_sigma_scale), residual * 2.0 + 1.0)
        row_score = torch.exp(-0.5 * (row_dist / max(row_sigma, 1e-6)) ** 2)
        row_response = (score_2d * gate_2d * row_score).cpu().numpy().astype(np.float32)
        bins = max(w, 16)
        x_bins = np.clip((x_rot_grid.cpu().numpy() / max(float(img_w), 1e-6) * bins).astype(np.int32), 0, bins - 1)
        row_weights = row_score.cpu().numpy().astype(np.float32)
        response_sum = np.bincount(x_bins.reshape(-1), weights=row_response.reshape(-1), minlength=bins).astype(np.float32)
        weight_sum = np.bincount(x_bins.reshape(-1), weights=row_weights.reshape(-1), minlength=bins).astype(np.float32)
        profile = response_sum / np.maximum(weight_sum, 1e-6)
        step_px = float(img_w) / float(bins)
        min_period_px = max(6.0, float(period_seed) * 0.55 if period_seed > 0.0 else float(img_w) * 0.04)
        max_period_px = min(float(img_w) * 0.60, float(period_seed) * 1.80 if period_seed > 0.0 else float(img_w) * 0.30)
        period_resp, resp_strength = _estimate_period_from_profile(
            profile,
            step_px=step_px,
            min_period_px=min_period_px,
            max_period_px=max_period_px,
        )
        if period_seed > 0.0 and period_resp > 0.0:
            blend = float(np.clip(float(response_period_blend) * max(float(resp_strength), 0.20), 0.0, 1.0))
            period = (1.0 - blend) * float(period_seed) + blend * float(period_resp)
            response_period_count += 1
        elif period_resp > 0.0:
            period = float(period_resp)
            response_period_count += 1
        else:
            period = float(period_seed)
        period = _maybe_use_period_prior(period, period_prior_px, period_prior_ratio)
        use_period = period >= 3.0
        if use_period:
            period *= max(float(period_scale), 0.05)
            phase = _estimate_spacing_phase(seed_x, period)
            if bool(full_row_support):
                x_min = 0.0
                x_max = float(img_w)
            else:
                x_min = float(row["x_min"]) - period
                x_max = float(row["x_max"]) + period
            period_dist = torch.remainder(x_rot_grid - phase + 0.5 * period, period) - 0.5 * period
            period_sigma = max(float(period) * float(period_sigma_scale), 2.0)
            period_score = torch.exp(-0.5 * (period_dist / max(period_sigma, 1e-6)) ** 2)
        else:
            extend = 0.35 * float(img_w)
            if bool(full_row_support):
                x_min = 0.0
                x_max = float(img_w)
            else:
                x_min = float(row["x_min"]) - extend
                x_max = float(row["x_max"]) + extend
            period_score = torch.ones((h, w), dtype=torch.float32)
        x_support = ((x_rot_grid >= x_min) & (x_rot_grid <= x_max)).float()
        row_weight = float(np.clip(np.mean(row_pts[:, 2]) if row_pts.shape[1] >= 3 else 1.0, 0.0, 1.0))
        row_prior = row_score * period_score * x_support * row_weight
        prior = torch.maximum(prior, row_prior)
        used_rows += 1
        if use_period:
            periods.append(float(period))
        else:
            row_only_count += 1

    row_center_seed = np.sort(
        np.asarray(
            [float(np.mean(np.asarray(row["points"], dtype=np.float32)[:, 1])) for row in rows if np.asarray(row["points"]).shape[0] >= int(min_row_seeds)],
            dtype=np.float32,
        )
    )
    if row_center_priors_px:
        prior_y = np.asarray([float(v) for v in row_center_priors_px if float(v) >= 0.0], dtype=np.float32)
        if prior_y.size:
            prior_pts = np.stack(
                [
                    np.full((prior_y.size,), center[0], dtype=np.float32),
                    prior_y,
                    np.ones((prior_y.size,), dtype=np.float32),
                ],
                axis=1,
            )
            row_center_basis = np.sort(_rotate_points(prior_pts, center=center, angle_rad=-orientation_rad)[:, 1].astype(np.float32))
        else:
            row_center_basis = row_center_seed
    else:
        row_center_basis = row_center_seed

    row_period_seed = _estimate_spacing_px(row_center_seed)
    row_period = _maybe_use_period_prior(row_period_seed, row_period_prior_px, period_prior_ratio)
    cross_row_used = 0.0
    if row_center_basis.size and float(cross_row_strength) > 0.0:
        if row_period < 3.0 and row_center_basis.size >= 2:
            row_period = _estimate_spacing_px(row_center_basis)
        if row_period >= 3.0:
            phase_y = _estimate_spacing_phase(row_center_basis, row_period)
            band_dist = torch.remainder(y_rot_grid - phase_y + 0.5 * row_period, row_period) - 0.5 * row_period
            band_sigma = max(float(row_tolerance_px) * 1.10, float(row_period) * 0.22)
            cross_row_prior = torch.exp(-0.5 * (band_dist / max(band_sigma, 1e-6)) ** 2)
        else:
            band_sigma = max(float(row_tolerance_px) * 1.10, 8.0)
            cross_row_prior = torch.zeros((h, w), dtype=torch.float32)
            for row_y in row_center_basis.tolist():
                row_band = torch.exp(-0.5 * ((y_rot_grid - float(row_y)) / max(band_sigma, 1e-6)) ** 2)
                cross_row_prior = torch.maximum(cross_row_prior, row_band)
        prior = torch.maximum(prior, (1.0 - float(cross_row_strength)) * prior + float(cross_row_strength) * cross_row_prior)
        cross_row_used = 1.0

    if prior.numel() and float(prior.max().item()) > 0.0:
        prior = prior / float(prior.max().item())
    prior = prior * gate_2d.clamp(0.0, 1.0)
    stats = {
        "pattern_seed_count": float(seed_points.shape[0]),
        "pattern_row_count": float(len(rows)),
        "pattern_used_row_count": float(used_rows),
        "pattern_row_only_count": float(row_only_count),
        "pattern_cross_row_used": float(cross_row_used),
        "pattern_mean_period_px": float(np.mean(periods)) if periods else 0.0,
        "pattern_mean_row_period_px": float(row_period) if row_period >= 3.0 else 0.0,
        "pattern_orientation_deg": float(np.degrees(orientation_rad)),
        "pattern_response_period_count": float(response_period_count),
    }
    return prior, seed_points, stats


def _pattern_enhance_feature(
    pyramid_feats: Dict[str, torch.Tensor],
    dino_feat: torch.Tensor,
    *,
    detect_module: torch.nn.Module,
    target_level: str,
    prior_levels: Sequence[str],
    image_size: tuple[int, int],
    anchor_points: Optional[np.ndarray],
    mask_target_coverage: float,
    mask_min_coverage: float,
    mask_max_coverage: float,
    mask_fuse_bg: float,
    mask_fuse_fg: float,
    pattern_seed_threshold: float,
    pattern_seed_topk: int,
    pattern_gate_threshold: float,
    pattern_row_tolerance_px: float,
    pattern_min_row_seeds: int,
    pattern_line_max_slope: float,
    pattern_period_prior_px: float,
    pattern_period_prior_ratio: float,
    pattern_period_scale: float,
    pattern_row_sigma_scale: float,
    pattern_period_sigma_scale: float,
    pattern_row_period_prior_px: float,
    pattern_row_center_priors_px: Optional[Sequence[float]],
    pattern_prior_strength: float,
    pattern_completion_strength: float,
    pattern_completion_gamma: float,
    pattern_noise_suppress: float,
    pattern_cross_row_strength: float,
    pattern_cross_row_noise_suppress: float,
    response_period_blend: float,
    nmf_enabled: bool,
    nmf_components: int,
    nmf_topk: int,
    nmf_strength: float,
    nmf_init: str,
    nmf_max_iter: int,
    nmf_override_map: Optional[torch.Tensor] = None,
    apply_feature: bool = True,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, float]]:
    if target_level not in pyramid_feats:
        raise KeyError(f"Missing target level in pyramid: {target_level}")
    level_order = {"p3": 0, "p4": 1, "p5": 2}
    target_feat = pyramid_feats[target_level]
    target_size = tuple(int(v) for v in target_feat.shape[-2:])
    _base_fused_target, target_gate = _apply_dino_mask_fuse_to_feature(
        target_feat,
        dino_feat,
        bg_multiplier=mask_fuse_bg,
        fg_multiplier=mask_fuse_fg,
        target_coverage=mask_target_coverage,
        min_coverage=mask_min_coverage,
        max_coverage=mask_max_coverage,
    )
    target_gate_2d = target_gate.squeeze(0).detach().float().cpu().clamp(0.0, 1.0)
    target_score_map, _ = _head_cls_score_map(detect_module, target_feat, level_order[target_level])
    target_score_norm = _normalize_score_map(target_score_map)
    level_maps: List[Dict[str, torch.Tensor]] = []
    level_stats: List[Dict[str, float]] = []
    for level in prior_levels:
        source_feat = pyramid_feats.get(level)
        if source_feat is None:
            continue
        level_idx = level_order[level]
        _base_fused_level, level_gate = _apply_dino_mask_fuse_to_feature(
            source_feat,
            dino_feat,
            bg_multiplier=mask_fuse_bg,
            fg_multiplier=mask_fuse_fg,
            target_coverage=mask_target_coverage,
            min_coverage=mask_min_coverage,
            max_coverage=mask_max_coverage,
        )
        score_map, _ = _head_cls_score_map(detect_module, source_feat, level_idx)
        prior_map, _seed_points, prior_stats = _pattern_prior_from_maps(
            score_map,
            level_gate,
            image_size=image_size,
            anchor_points=anchor_points,
            seed_threshold=pattern_seed_threshold,
            seed_topk=pattern_seed_topk,
            gate_threshold=pattern_gate_threshold,
            row_tolerance_px=pattern_row_tolerance_px,
            min_row_seeds=pattern_min_row_seeds,
            max_abs_slope=pattern_line_max_slope,
            period_prior_px=pattern_period_prior_px,
            period_prior_ratio=pattern_period_prior_ratio,
            period_scale=pattern_period_scale,
            row_sigma_scale=pattern_row_sigma_scale,
            period_sigma_scale=pattern_period_sigma_scale,
            row_period_prior_px=pattern_row_period_prior_px,
            row_center_priors_px=pattern_row_center_priors_px,
            cross_row_strength=pattern_cross_row_strength,
            response_period_blend=response_period_blend,
        )
        gate_2d = level_gate.squeeze(0).detach().float().cpu().clamp(0.0, 1.0)
        score_norm = _normalize_score_map(score_map)
        local_evidence = (0.55 * gate_2d + 0.45 * score_norm).clamp(0.0, 1.0)
        pattern_base = (prior_map * (0.35 + 0.65 * gate_2d)).clamp(0.0, 1.0)
        if tuple(prior_map.shape[-2:]) != target_size:
            prior_map = _resize_spatial_map(prior_map, target_size)
            gate_2d = _resize_spatial_map(gate_2d, target_size)
            score_norm = _resize_spatial_map(score_norm, target_size)
            local_evidence = _resize_spatial_map(local_evidence, target_size)
            pattern_base = _resize_spatial_map(pattern_base, target_size)
        level_maps.append(
            {
                "prior": prior_map.clamp(0.0, 1.0),
                "gate": gate_2d.clamp(0.0, 1.0),
                "score": score_norm.clamp(0.0, 1.0),
                "local": local_evidence.clamp(0.0, 1.0),
                "pattern_base": pattern_base.clamp(0.0, 1.0),
            }
        )
        level_stats.append(prior_stats)

    if level_maps:
        weights: List[float] = []
        for level in prior_levels:
            if level not in pyramid_feats:
                continue
            diff = abs(level_order[level] - level_order[target_level])
            weights.append(1.0 if diff == 0 else 0.86 if diff == 1 else 0.70)
        weight_tensor = torch.tensor(weights, dtype=torch.float32).view(-1, 1, 1)
        prior_stack = torch.stack([entry["prior"] for entry in level_maps], dim=0)
        gate_stack = torch.stack([entry["gate"] for entry in level_maps], dim=0)
        score_stack = torch.stack([entry["score"] for entry in level_maps], dim=0)
        local_stack = torch.stack([entry["local"] for entry in level_maps], dim=0)
        pattern_stack = torch.stack([entry["pattern_base"] for entry in level_maps], dim=0)
        weight_sum = weight_tensor.sum().clamp_min(1e-6)
        prior_map = (0.55 * ((prior_stack * weight_tensor).sum(dim=0) / weight_sum) + 0.45 * prior_stack.max(dim=0).values).clamp(0.0, 1.0)
        gate_2d = ((gate_stack * weight_tensor).sum(dim=0) / weight_sum).clamp(0.0, 1.0)
        score_norm = ((score_stack * weight_tensor).sum(dim=0) / weight_sum).clamp(0.0, 1.0)
        local_evidence = ((local_stack * weight_tensor).sum(dim=0) / weight_sum).clamp(0.0, 1.0)
        pattern_base = (0.55 * ((pattern_stack * weight_tensor).sum(dim=0) / weight_sum) + 0.45 * pattern_stack.max(dim=0).values).clamp(0.0, 1.0)
    else:
        prior_map = torch.zeros(target_size, dtype=torch.float32)
        gate_2d = target_gate_2d
        score_norm = target_score_norm
        local_evidence = (0.55 * gate_2d + 0.45 * score_norm).clamp(0.0, 1.0)
        pattern_base = torch.zeros(target_size, dtype=torch.float32)

    nmf_stats: Dict[str, float] = {}
    if nmf_enabled and float(nmf_strength) > 0.0:
        if nmf_override_map is not None:
            nmf_guide = _resize_spatial_map(nmf_override_map, target_size).clamp(0.0, 1.0)
        else:
            nmf_guide, nmf_stats = _build_nmf_guide_map(
                dino_feat,
                n_components=nmf_components,
                topk=nmf_topk,
                init=nmf_init,
                max_iter=nmf_max_iter,
                target_size=target_size,
            )
        nmf_guide = (nmf_guide * (0.35 + 0.65 * gate_2d)).clamp(0.0, 1.0)
    else:
        nmf_guide = torch.zeros(target_size, dtype=torch.float32)

    support = (local_evidence + float(pattern_prior_strength) * pattern_base * (1.0 - local_evidence)).clamp(0.0, 1.0)
    completion = torch.clamp(pattern_base - local_evidence, min=0.0)
    completion_gamma = max(float(pattern_completion_gamma), 1e-3)
    completion = completion.pow(completion_gamma)
    support = torch.maximum(
        support,
        (local_evidence + float(pattern_completion_strength) * completion).clamp(0.0, 1.0),
    )
    if nmf_guide.numel() and float(nmf_strength) > 0.0:
        support = torch.maximum(
            support,
            (support + float(nmf_strength) * nmf_guide * (1.0 - support)).clamp(0.0, 1.0),
        )
    noise_term = torch.clamp(score_norm - prior_map, min=0.0) * torch.clamp(1.0 - gate_2d, min=0.0)
    if float(pattern_cross_row_noise_suppress) > 0.0:
        cross_row_noise = torch.clamp(score_norm - prior_map, min=0.0) * torch.clamp(1.0 - prior_map, min=0.0) * gate_2d
        noise_term = noise_term + float(pattern_cross_row_noise_suppress) * cross_row_noise
    support = (support - float(pattern_noise_suppress) * noise_term).clamp(0.0, 1.0)
    scale = float(mask_fuse_bg) + (float(mask_fuse_fg) - float(mask_fuse_bg)) * support.to(device=target_feat.device, dtype=target_feat.dtype)
    if apply_feature:
        enhanced_feat = target_feat * scale.unsqueeze(0)
    else:
        enhanced_feat = target_feat.clone()
    stats = _aggregate_pattern_stats(level_stats)
    stats.update(
        {
            "pattern_completion_mean": float(completion.mean().item()) if completion.numel() else 0.0,
            "pattern_completion_max": float(completion.max().item()) if completion.numel() else 0.0,
            "pattern_support_mean": float(support.mean().item()) if support.numel() else 0.0,
            "pattern_support_max": float(support.max().item()) if support.numel() else 0.0,
            "pattern_feature_modulation_enabled": 1.0 if apply_feature else 0.0,
            "pattern_feature_scale_mean": float(scale.mean().item()) if scale.numel() else 1.0,
        }
    )
    stats.update(nmf_stats)
    return enhanced_feat, prior_map, support, nmf_guide, stats


def _predict_one_pattern_enhanced(
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
    prior_levels: Sequence[str],
    anchor_points: Optional[np.ndarray],
    image_size_xy: tuple[int, int],
    mask_target_coverage: float,
    mask_min_coverage: float,
    mask_max_coverage: float,
    mask_fuse_bg: float,
    mask_fuse_fg: float,
    pattern_seed_threshold: float,
    pattern_seed_topk: int,
    pattern_gate_threshold: float,
    pattern_row_tolerance_px: float,
    pattern_min_row_seeds: int,
    pattern_line_max_slope: float,
    pattern_period_prior_px: float,
    pattern_period_prior_ratio: float,
    pattern_period_scale: float,
    pattern_row_sigma_scale: float,
    pattern_period_sigma_scale: float,
    pattern_row_period_prior_px: float,
    pattern_row_center_priors_px: Optional[Sequence[float]],
    pattern_prior_strength: float,
    pattern_completion_strength: float,
    pattern_completion_gamma: float,
    pattern_noise_suppress: float,
    pattern_cross_row_strength: float,
    pattern_cross_row_noise_suppress: float,
    response_period_blend: float,
    nmf_enabled: bool,
    nmf_components: int,
    nmf_topk: int,
    nmf_strength: float,
    nmf_init: str,
    nmf_max_iter: int,
    nmf_override_map: Optional[torch.Tensor],
) -> List[Dict[str, object]]:
    det_model = yolo_model.model
    detect_module = _find_detect_module(det_model)
    level_idx = {"p3": 0, "p4": 1, "p5": 2}[feature_level]
    state: Dict[str, Optional[torch.Tensor]] = {"dino": None}
    first_module = det_model.model[0]

    def _cache_dino(_module, inputs):
        if not inputs:
            state["dino"] = None
            return None
        images = inputs[0].detach().float()
        with torch.no_grad():
            state["dino"] = dino(images.to(device)).detach()
        return None

    def _enhance_detect_input(_module, inputs):
        if not inputs or state.get("dino") is None:
            return None
        x = inputs[0]
        if not isinstance(x, (list, tuple)) or len(x) <= level_idx:
            return None
        feats = list(x)
        target_feat = feats[level_idx]
        dino_batch = state["dino"]
        if dino_batch is None or dino_batch.shape[0] != target_feat.shape[0]:
            return None
        enhanced: List[torch.Tensor] = []
        for batch_idx in range(target_feat.shape[0]):
            pyramid_feats = {
                level: feats[idx][batch_idx]
                for idx, level in enumerate(("p3", "p4", "p5"))
                if idx < len(feats)
            }
            enhanced_feat, _prior_map, _support, _nmf_guide, _stats = _pattern_enhance_feature(
                pyramid_feats,
                dino_batch[batch_idx],
                detect_module=detect_module,
                target_level=feature_level,
                prior_levels=prior_levels,
                image_size=image_size_xy,
                anchor_points=anchor_points,
                mask_target_coverage=mask_target_coverage,
                mask_min_coverage=mask_min_coverage,
                mask_max_coverage=mask_max_coverage,
                mask_fuse_bg=mask_fuse_bg,
                mask_fuse_fg=mask_fuse_fg,
                pattern_seed_threshold=pattern_seed_threshold,
                pattern_seed_topk=pattern_seed_topk,
                pattern_gate_threshold=pattern_gate_threshold,
                pattern_row_tolerance_px=pattern_row_tolerance_px,
                pattern_min_row_seeds=pattern_min_row_seeds,
                pattern_line_max_slope=pattern_line_max_slope,
        pattern_period_prior_px=pattern_period_prior_px,
        pattern_period_prior_ratio=pattern_period_prior_ratio,
        pattern_period_scale=pattern_period_scale,
        pattern_row_sigma_scale=pattern_row_sigma_scale,
        pattern_period_sigma_scale=pattern_period_sigma_scale,
        pattern_row_period_prior_px=pattern_row_period_prior_px,
        pattern_row_center_priors_px=pattern_row_center_priors_px,
        pattern_prior_strength=pattern_prior_strength,
        pattern_completion_strength=pattern_completion_strength,
        pattern_completion_gamma=pattern_completion_gamma,
        pattern_noise_suppress=pattern_noise_suppress,
        pattern_cross_row_strength=pattern_cross_row_strength,
        pattern_cross_row_noise_suppress=pattern_cross_row_noise_suppress,
        response_period_blend=response_period_blend,
        nmf_enabled=nmf_enabled,
                nmf_components=nmf_components,
                nmf_topk=nmf_topk,
                nmf_strength=nmf_strength,
                nmf_init=nmf_init,
                nmf_max_iter=nmf_max_iter,
                nmf_override_map=nmf_override_map,
            )
            enhanced.append(enhanced_feat)
        feats[level_idx] = torch.stack(enhanced, dim=0)
        return (feats,) + tuple(inputs[1:])

    handle_input = first_module.register_forward_pre_hook(_cache_dino)
    handle_detect = detect_module.register_forward_pre_hook(_enhance_detect_input)
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
    pattern_prior_levels = _parse_feature_levels(args.pattern_prior_levels, fallback=feature_level)
    config_before = config.get("model", {}).get("student_config", {}).get("weights")
    before_weights = _resolve_weight_path(args.before_weights, config_before, root=REPO_ROOT)
    after_weights = _resolve_weight_path(args.after_weights, None, root=REPO_ROOT)

    output_dir = (
        Path(args.output_dir).resolve()
        if args.output_dir
        else Path("outputs") / "yolo_prediction_panels"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    nmf_score_dir = Path(args.nmf_guide_score_dir).resolve() if args.nmf_guide_score_dir else None
    pca_score_dir = Path(args.pca_guide_score_dir).resolve() if args.pca_guide_score_dir else None
    image_prior_index = _load_image_prior_index(Path(args.pattern_image_priors).resolve()) if args.pattern_image_priors else {}
    species_prior_bank = _load_species_prior_bank(Path(args.pattern_species_prior_bank).resolve()) if args.pattern_species_prior_bank else {}
    guide_family = "pca" if args.pca_guide else "nmf" if args.nmf_guide else ""
    guide_enabled = bool(guide_family)
    guide_label = "PCA" if guide_family == "pca" else "NMF"

    if args.image:
        image_path = Path(args.image).resolve()
        if not image_path.is_file():
            raise FileNotFoundError(f"Image not found: {image_path}")
        batches = _iter_image_path_batches([image_path], image_size, args.batch_size)
    elif args.image_dir:
        batches = _iter_image_dir_batches(Path(args.image_dir), image_size, args.batch_size)
    else:
        batches = _build_loader(config, split=args.split, batch_size=args.batch_size, num_workers=args.num_workers)

    print(f"[yolo-pred-panels] output_dir={output_dir}", flush=True)
    print(f"[yolo-pred-panels] device={device} imgsz={image_size} feature_level={feature_level}", flush=True)
    print(f"[yolo-pred-panels] pattern_prior_levels={pattern_prior_levels}", flush=True)
    print(f"[yolo-pred-panels] before={before_weights}", flush=True)
    print(f"[yolo-pred-panels] after={after_weights}", flush=True)
    print(
        f"[yolo-pred-panels] external_priors image={bool(image_prior_index)} species={bool(species_prior_bank)}",
        flush=True,
    )

    before_student = _build_yolo_student(before_weights, config, feature_level, device)
    after_student = _build_yolo_student(after_weights, config, feature_level, device)
    before_yolo = YOLO(str(before_weights))
    after_yolo = YOLO(str(after_weights))
    dino = _build_dino(config, device)
    before_detect = _find_detect_module(before_student.det_model)
    after_detect = _find_detect_module(after_student.det_model)
    detect_level_idx = {"p3": 0, "p4": 1, "p5": 2}[feature_level]

    rows: List[Dict[str, object]] = []
    sample_count = 0

    for batch in batches:
        images = batch["images"].to(device)
        image_paths = [str(p) for p in batch.get("image_paths", [])]
        if not image_paths:
            image_paths = [f"sample_{sample_count + i:04d}" for i in range(images.shape[0])]

        with torch.no_grad():
            before_pyramid = _extract_yolo_pyramid(before_student, images)
            after_pyramid = _extract_yolo_pyramid(after_student, images)
            dino_feat = dino(images).detach()

        for i in range(images.shape[0]):
            if sample_count >= args.num_samples:
                break

            image_path = image_paths[i]
            stem = Path(image_path).stem
            original_pil = (
                Image.open(image_path).convert("RGB")
                if Path(image_path).is_file()
                else _tensor_image_to_pil(images[i].detach().cpu())
            )
            external_pattern_prior = _resolve_external_pattern_prior(
                image_path,
                image_size=original_pil.size,
                image_prior_index=image_prior_index,
                species_prior_bank=species_prior_bank,
            )
            external_row_centers = []
            raw_row_centers = external_pattern_prior.get("row_centers_px", [])
            if isinstance(raw_row_centers, (list, tuple)) and raw_row_centers:
                row_pts = np.asarray([[0.0, float(y), 1.0] for y in raw_row_centers], dtype=np.float32)
                external_row_centers = _transform_anchor_points(
                    row_pts,
                    src_size=original_pil.size,
                    dst_size=(image_size, image_size),
                    keep_aspect=True,
                )[:, 1].astype(np.float32).tolist()
            effective_period_prior_px = float(args.pattern_period_prior_px)
            if effective_period_prior_px <= 0.0:
                effective_period_prior_px = float(external_pattern_prior.get("x_period_px", 0.0) or 0.0)
            effective_row_tolerance_px = float(args.pattern_row_tolerance_px)
            external_row_tolerance_px = float(external_pattern_prior.get("row_tolerance_px", 0.0) or 0.0)
            if external_row_tolerance_px > 0.0:
                effective_row_tolerance_px = external_row_tolerance_px
            effective_row_period_px = float(external_pattern_prior.get("row_period_px", 0.0) or 0.0)

            before_preds = (
                _predict_one(
                    before_yolo,
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
            after_preds = (
                _predict_one(
                    after_yolo,
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
            feature_source_pyramid = after_pyramid if args.feature_source == "after" else before_pyramid
            feature_source_feat = feature_source_pyramid[feature_level][i]
            feature_source_detect = after_detect if args.feature_source == "after" else before_detect
            panel_anchor_points = _transform_anchor_points(
                _prediction_center_points(after_preds if args.feature_source == "after" else before_preds),
                src_size=original_pil.size,
                dst_size=(image_size, image_size),
                keep_aspect=False,
            )
            panel_guide_override: Optional[torch.Tensor] = None
            panel_guide_stats: Dict[str, float] = {}
            if guide_enabled and Path(image_path).is_file():
                if guide_family == "pca":
                    panel_guide_override, panel_guide_stats, _panel_guide_raw_overlay = _build_pca_guide_map_from_image(
                        image_path,
                        dino=dino,
                        device=device,
                        image_size=args.pca_guide_image_size,
                        n_components=6,
                        target_size=tuple(int(v) for v in feature_source_feat.shape[-2:]),
                        score_dir=pca_score_dir,
                    )
                else:
                    panel_guide_override, panel_guide_stats, _panel_guide_raw_overlay = _build_nmf_guide_map_from_image(
                        image_path,
                        dino=dino,
                        device=device,
                        image_size=args.nmf_guide_image_size,
                        n_components=args.nmf_guide_components,
                        topk=args.nmf_guide_topk,
                        init=args.nmf_guide_init,
                        max_iter=args.nmf_guide_max_iter,
                        target_size=tuple(int(v) for v in feature_source_feat.shape[-2:]),
                        score_dir=nmf_score_dir,
                    )
            enhanced_feat, _prior_map, _support_map, _guide_map, pattern_stats = _pattern_enhance_feature(
                {level: feat[i] for level, feat in feature_source_pyramid.items()},
                dino_feat[i],
                detect_module=feature_source_detect,
                target_level=feature_level,
                prior_levels=pattern_prior_levels,
                image_size=(image_size, image_size),
                anchor_points=panel_anchor_points,
                mask_target_coverage=args.mask_target_coverage,
                mask_min_coverage=args.mask_min_coverage,
                mask_max_coverage=args.mask_max_coverage,
                mask_fuse_bg=args.mask_fuse_bg,
                mask_fuse_fg=args.mask_fuse_fg,
                pattern_seed_threshold=args.pattern_seed_threshold,
                pattern_seed_topk=args.pattern_seed_topk,
                pattern_gate_threshold=args.pattern_gate_threshold,
                pattern_row_tolerance_px=effective_row_tolerance_px,
                pattern_min_row_seeds=args.pattern_min_row_seeds,
                pattern_line_max_slope=args.pattern_line_max_slope,
                pattern_period_prior_px=effective_period_prior_px,
                pattern_period_prior_ratio=args.pattern_period_prior_ratio,
                pattern_period_scale=args.pattern_period_scale,
                pattern_row_sigma_scale=args.pattern_row_sigma_scale,
                pattern_period_sigma_scale=args.pattern_period_sigma_scale,
                pattern_row_period_prior_px=effective_row_period_px,
                pattern_row_center_priors_px=external_row_centers,
                pattern_prior_strength=args.pattern_prior_strength if args.pattern_enhance else 0.0,
                pattern_completion_strength=args.pattern_completion_strength if args.pattern_enhance else 0.0,
                pattern_completion_gamma=args.pattern_completion_gamma,
                pattern_noise_suppress=args.pattern_noise_suppress if args.pattern_enhance else 0.0,
                pattern_cross_row_strength=args.pattern_cross_row_strength if args.pattern_enhance else 0.0,
                pattern_cross_row_noise_suppress=args.pattern_cross_row_noise_suppress if args.pattern_enhance else 0.0,
                response_period_blend=args.pattern_response_period_blend,
                nmf_enabled=guide_enabled,
                nmf_components=args.nmf_guide_components,
                nmf_topk=args.nmf_guide_topk,
                nmf_strength=args.pca_guide_strength if guide_family == "pca" else args.nmf_guide_strength if guide_enabled else 0.0,
                nmf_init=args.nmf_guide_init,
                nmf_max_iter=args.nmf_guide_max_iter,
                nmf_override_map=panel_guide_override,
            )
            pattern_stats.update(panel_guide_stats)

            yolo_feature_img = _feature_response_image(feature_source_feat, original_pil.size)
            enhanced_feature_img = _feature_response_image(enhanced_feat, original_pil.size)

            nmf_preds = (
                _predict_one_pattern_enhanced(
                    after_yolo,
                    dino,
                    image_path,
                    imgsz=image_size,
                    conf=args.conf,
                    iou=args.iou,
                    max_det=args.max_det,
                    device=device,
                    feature_level=feature_level,
                    prior_levels=pattern_prior_levels,
                    anchor_points=_transform_anchor_points(
                        _prediction_center_points(after_preds),
                        src_size=original_pil.size,
                        dst_size=(image_size, image_size),
                        keep_aspect=True,
                    ),
                    image_size_xy=(image_size, image_size),
                    mask_target_coverage=args.mask_target_coverage,
                    mask_min_coverage=args.mask_min_coverage,
                    mask_max_coverage=args.mask_max_coverage,
                    mask_fuse_bg=args.mask_fuse_bg,
                    mask_fuse_fg=args.mask_fuse_fg,
                    pattern_seed_threshold=args.pattern_seed_threshold,
                    pattern_seed_topk=args.pattern_seed_topk,
                    pattern_gate_threshold=args.pattern_gate_threshold,
                    pattern_row_tolerance_px=effective_row_tolerance_px,
                    pattern_min_row_seeds=args.pattern_min_row_seeds,
                    pattern_line_max_slope=args.pattern_line_max_slope,
                    pattern_period_prior_px=effective_period_prior_px,
                    pattern_period_prior_ratio=args.pattern_period_prior_ratio,
                    pattern_period_scale=args.pattern_period_scale,
                    pattern_row_sigma_scale=args.pattern_row_sigma_scale,
                    pattern_period_sigma_scale=args.pattern_period_sigma_scale,
                    pattern_row_period_prior_px=effective_row_period_px,
                    pattern_row_center_priors_px=external_row_centers,
                    pattern_prior_strength=0.0,
                    pattern_completion_strength=0.0,
                    pattern_completion_gamma=args.pattern_completion_gamma,
                    pattern_noise_suppress=0.0,
                    pattern_cross_row_strength=0.0,
                    pattern_cross_row_noise_suppress=0.0,
                    response_period_blend=args.pattern_response_period_blend,
                    nmf_enabled=guide_enabled,
                    nmf_components=args.nmf_guide_components,
                    nmf_topk=args.nmf_guide_topk,
                    nmf_strength=args.pca_guide_strength if guide_family == "pca" else args.nmf_guide_strength if guide_enabled else 0.0,
                    nmf_init=args.nmf_guide_init,
                    nmf_max_iter=args.nmf_guide_max_iter,
                    nmf_override_map=panel_guide_override,
                )
                if Path(image_path).is_file() and guide_enabled
                else []
            )
            pattern_preds = (
                _predict_one_pattern_enhanced(
                    after_yolo,
                    dino,
                    image_path,
                    imgsz=image_size,
                    conf=args.conf,
                    iou=args.iou,
                    max_det=args.max_det,
                    device=device,
                    feature_level=feature_level,
                    prior_levels=pattern_prior_levels,
                    anchor_points=_transform_anchor_points(
                        _prediction_center_points(after_preds),
                        src_size=original_pil.size,
                        dst_size=(image_size, image_size),
                        keep_aspect=True,
                    ),
                    image_size_xy=(image_size, image_size),
                    mask_target_coverage=args.mask_target_coverage,
                    mask_min_coverage=args.mask_min_coverage,
                    mask_max_coverage=args.mask_max_coverage,
                    mask_fuse_bg=args.mask_fuse_bg,
                    mask_fuse_fg=args.mask_fuse_fg,
                    pattern_seed_threshold=args.pattern_seed_threshold,
                    pattern_seed_topk=args.pattern_seed_topk,
                    pattern_gate_threshold=args.pattern_gate_threshold,
                    pattern_row_tolerance_px=effective_row_tolerance_px,
                    pattern_min_row_seeds=args.pattern_min_row_seeds,
                    pattern_line_max_slope=args.pattern_line_max_slope,
                    pattern_period_prior_px=effective_period_prior_px,
                    pattern_period_prior_ratio=args.pattern_period_prior_ratio,
                    pattern_period_scale=args.pattern_period_scale,
                    pattern_row_sigma_scale=args.pattern_row_sigma_scale,
                    pattern_period_sigma_scale=args.pattern_period_sigma_scale,
                    pattern_row_period_prior_px=effective_row_period_px,
                    pattern_row_center_priors_px=external_row_centers,
                    pattern_prior_strength=args.pattern_prior_strength if args.pattern_enhance else 0.0,
                    pattern_completion_strength=args.pattern_completion_strength if args.pattern_enhance else 0.0,
                    pattern_completion_gamma=args.pattern_completion_gamma,
                    pattern_noise_suppress=args.pattern_noise_suppress if args.pattern_enhance else 0.0,
                    pattern_cross_row_strength=args.pattern_cross_row_strength if args.pattern_enhance else 0.0,
                    pattern_cross_row_noise_suppress=args.pattern_cross_row_noise_suppress if args.pattern_enhance else 0.0,
                    response_period_blend=args.pattern_response_period_blend,
                    nmf_enabled=guide_enabled,
                    nmf_components=args.nmf_guide_components,
                    nmf_topk=args.nmf_guide_topk,
                    nmf_strength=args.pca_guide_strength if guide_family == "pca" else args.nmf_guide_strength if guide_enabled else 0.0,
                    nmf_init=args.nmf_guide_init,
                    nmf_max_iter=args.nmf_guide_max_iter,
                    nmf_override_map=panel_guide_override,
                )
                if Path(image_path).is_file() and (args.pattern_enhance or guide_enabled)
                else []
            )
            if not guide_enabled:
                nmf_preds = list(after_preds)
            if not (args.pattern_enhance or guide_enabled):
                pattern_preds = list(after_preds)

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
            feature_label = _feature_level_label(feature_level)
            feature_tiles = [
                _make_tile(original_pil, "Origin Image", args.tile_size),
                _make_tile(
                    yolo_feature_img,
                    f"{args.feature_source.title()} YOLO {feature_label} Before Pattern",
                    args.tile_size,
                ),
                _make_tile(
                    enhanced_feature_img,
                    f"{args.feature_source.title()} YOLO {feature_label} After Pattern",
                    args.tile_size,
                ),
            ]
            feature_panel = _make_grid(feature_tiles, cols=3)
            feature_panel_path = output_dir / f"{sample_count:04d}_{stem}_feature_panel.png"
            feature_panel.save(feature_panel_path)

            prediction_tiles = [
                _make_tile(before_overlay, "Before Alignment Prediction", args.tile_size),
                _make_tile(after_overlay, "After Alignment Prediction", args.tile_size),
            ]
            prediction_panel = _make_grid(prediction_tiles, cols=2)
            prediction_panel_path = output_dir / f"{sample_count:04d}_{stem}_prediction_panel.png"
            prediction_panel.save(prediction_panel_path)

            pred_metrics = _prediction_delta_metrics(before_preds, after_preds)
            nmf_metrics = _prediction_delta_metrics(after_preds, nmf_preds)
            pattern_metrics = _prediction_delta_metrics(after_preds, pattern_preds)
            before_pattern_metrics = _prediction_delta_metrics(before_preds, pattern_preds)
            row: Dict[str, object] = {
                "index": sample_count,
                "image_path": image_path,
                "feature_panel_path": str(feature_panel_path),
                "prediction_panel_path": str(prediction_panel_path),
                "before_weights": str(before_weights),
                "after_weights": str(after_weights),
                "feature_source": args.feature_source,
                "feature_level": feature_level,
                "pattern_prior_levels": ",".join(pattern_prior_levels),
                "before_box_count": float(len(before_preds)),
                "after_box_count": float(len(after_preds)),
                "nmf_box_count": float(len(nmf_preds)),
                "guide_box_count": float(len(nmf_preds)),
                "pattern_box_count": float(len(pattern_preds)),
                "matched_box_count": pred_metrics["matched_box_count"],
                "mean_matched_aabb_iou": pred_metrics["mean_matched_aabb_iou"],
                "mean_center_shift_px": pred_metrics["mean_center_shift_px"],
                "nmf_matched_box_count": nmf_metrics["matched_box_count"],
                "nmf_mean_matched_aabb_iou": nmf_metrics["mean_matched_aabb_iou"],
                "nmf_mean_center_shift_px": nmf_metrics["mean_center_shift_px"],
                "guide_matched_box_count": nmf_metrics["matched_box_count"],
                "guide_mean_matched_aabb_iou": nmf_metrics["mean_matched_aabb_iou"],
                "guide_mean_center_shift_px": nmf_metrics["mean_center_shift_px"],
                "pattern_matched_box_count": pattern_metrics["matched_box_count"],
                "pattern_mean_matched_aabb_iou": pattern_metrics["mean_matched_aabb_iou"],
                "pattern_mean_center_shift_px": pattern_metrics["mean_center_shift_px"],
                "before_pattern_matched_box_count": before_pattern_metrics["matched_box_count"],
                "before_pattern_mean_matched_aabb_iou": before_pattern_metrics["mean_matched_aabb_iou"],
                "before_pattern_mean_center_shift_px": before_pattern_metrics["mean_center_shift_px"],
                "mask_fuse_bg": float(args.mask_fuse_bg),
                "mask_fuse_fg": float(args.mask_fuse_fg),
                "pattern_enabled": float(1.0 if args.pattern_enhance else 0.0),
                "nmf_enabled": float(1.0 if guide_family == "nmf" else 0.0),
                "pca_enabled": float(1.0 if guide_family == "pca" else 0.0),
            }
            row.update({k: float(v) for k, v in pattern_stats.items()})
            row["pattern_prior_source"] = str(external_pattern_prior.get("source", "none"))
            row["pattern_prior_family"] = str(external_pattern_prior.get("family", ""))
            row["pattern_external_x_period_px"] = float(effective_period_prior_px)
            row["pattern_external_row_period_px"] = float(effective_row_period_px)
            row["pattern_external_row_tolerance_px"] = float(effective_row_tolerance_px)
            rows.append(row)
            print(
                f"[yolo-pred-panels] saved feature={feature_panel_path.name} pred={prediction_panel_path.name} "
                f"before={len(before_preds)} after={len(after_preds)} nmf={len(nmf_preds)} pattern={len(pattern_preds)} "
                f"matched={pred_metrics['matched_box_count']}",
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
        "pattern_prior_levels": pattern_prior_levels,
        "mean_before_box_count": float(np.mean([float(r["before_box_count"]) for r in rows])),
        "mean_after_box_count": float(np.mean([float(r["after_box_count"]) for r in rows])),
        "mean_guide_box_count": float(np.mean([float(r["guide_box_count"]) for r in rows])),
        "mean_nmf_box_count": float(np.mean([float(r["nmf_box_count"]) for r in rows])),
        "mean_pattern_box_count": float(np.mean([float(r["pattern_box_count"]) for r in rows])),
        "mean_matched_aabb_iou": float(np.mean([float(r["mean_matched_aabb_iou"]) for r in rows])),
        "mean_center_shift_px": float(np.mean([float(r["mean_center_shift_px"]) for r in rows])),
        "mean_guide_matched_aabb_iou": float(np.mean([float(r["guide_mean_matched_aabb_iou"]) for r in rows])),
        "mean_guide_center_shift_px": float(np.mean([float(r["guide_mean_center_shift_px"]) for r in rows])),
        "mean_nmf_matched_aabb_iou": float(np.mean([float(r["nmf_mean_matched_aabb_iou"]) for r in rows])),
        "mean_nmf_center_shift_px": float(np.mean([float(r["nmf_mean_center_shift_px"]) for r in rows])),
        "mean_pattern_matched_aabb_iou": float(np.mean([float(r["pattern_mean_matched_aabb_iou"]) for r in rows])),
        "mean_pattern_center_shift_px": float(np.mean([float(r["pattern_mean_center_shift_px"]) for r in rows])),
        "mean_pattern_seed_count": float(np.mean([float(r["pattern_seed_count"]) for r in rows])),
        "mean_pattern_row_count": float(np.mean([float(r["pattern_row_count"]) for r in rows])),
        "mean_pattern_used_row_count": float(np.mean([float(r["pattern_used_row_count"]) for r in rows])),
        "mean_pattern_row_only_count": float(np.mean([float(r["pattern_row_only_count"]) for r in rows])),
        "mean_pattern_cross_row_used": float(np.mean([float(r["pattern_cross_row_used"]) for r in rows])),
        "mean_pattern_period_px": float(np.mean([float(r["pattern_mean_period_px"]) for r in rows])),
        "mean_pattern_row_period_px": float(np.mean([float(r["pattern_mean_row_period_px"]) for r in rows])),
        "mean_pattern_orientation_deg": float(np.mean([float(r["pattern_orientation_deg"]) for r in rows])),
        "mean_pattern_response_period_count": float(np.mean([float(r["pattern_response_period_count"]) for r in rows])),
        "mean_pattern_prior_level_count": float(np.mean([float(r["pattern_prior_level_count"]) for r in rows])),
        "mean_pattern_completion_mean": float(np.mean([float(r["pattern_completion_mean"]) for r in rows])),
        "mean_pattern_support_mean": float(np.mean([float(r["pattern_support_mean"]) for r in rows])),
        "mean_pattern_external_x_period_px": float(np.mean([float(r["pattern_external_x_period_px"]) for r in rows])),
        "mean_pattern_external_row_period_px": float(np.mean([float(r["pattern_external_row_period_px"]) for r in rows])),
        "mean_pattern_external_row_tolerance_px": float(np.mean([float(r["pattern_external_row_tolerance_px"]) for r in rows])),
        "mean_pca_best_score": float(np.mean([float(r.get("pca_best_score", 0.0)) for r in rows])),
        "mean_pca_guide_mean": float(np.mean([float(r.get("pca_guide_mean", 0.0)) for r in rows])),
        "mean_nmf_best_score": float(np.mean([float(r.get("nmf_best_score", 0.0)) for r in rows])),
        "mean_nmf_guide_mean": float(np.mean([float(r.get("nmf_guide_mean", 0.0)) for r in rows])),
        "output_dir": str(output_dir),
    }
    _save_json(output_dir / "summary.json", summary)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
