#!/usr/bin/env python3
"""Export individual tiles and after-prediction heatmaps from debug_exports.

Run:
    python export_individual_debug_panels.py

Outputs:
    individual_exports/
      epoch_xxxx/
        <image_stem>/
          origin_image.png
          dino_response.png
          frequency_prior.png
          score_boost_heatmap.png
          before_pattern_prediction.png
          after_pattern_prediction.png
          score_boosted_proposals.png
          heatmap_after_density.png
          heatmap_after_confidence.png
          heatmap_after_pattern_support.png
          heatmap_after_synthetic.png
          overlay_after_density.png
          overlay_after_confidence.png
          overlay_after_pattern_support.png
          overlay_after_synthetic.png
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image


DEBUG_ROOT = Path(__file__).resolve().parent

FEATURE_TILES = [
    "origin_image",
    "dino_response",
    "frequency_prior",
    "score_boost_heatmap",
]

PREDICTION_TILES = [
    "before_pattern_prediction",
    "after_pattern_prediction",
    "score_boosted_proposals",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Split debug panels and export after-prediction heatmaps.")
    parser.add_argument("--debug-root", type=str, default=str(DEBUG_ROOT), help="Root containing epoch_* folders.")
    parser.add_argument("--output-dir", type=str, default=None, help="Output directory. Default: debug-root/individual_exports.")
    parser.add_argument("--epochs", type=str, default="", help="Comma-separated epoch names, e.g. epoch_0005,epoch_0010. Empty means all.")
    parser.add_argument("--heatmap-sigma", type=float, default=18.0, help="Gaussian radius in pixels for center heatmaps.")
    parser.add_argument("--heatmap-alpha", type=float, default=0.45, help="Overlay heatmap opacity.")
    parser.add_argument("--skip-existing", action="store_true", help="Skip writing existing files.")
    return parser.parse_args()


def round_bounds(total: int, parts: int) -> List[Tuple[int, int]]:
    bounds: List[Tuple[int, int]] = []
    for idx in range(parts):
        left = int(round(total * idx / parts))
        right = int(round(total * (idx + 1) / parts))
        bounds.append((left, right))
    return bounds


def stem_from_panel(panel_path: Path, suffix: str) -> str:
    name = panel_path.name
    if name.endswith(suffix):
        return name[: -len(suffix)]
    return panel_path.stem


def save_if_needed(image: Image.Image, path: Path, skip_existing: bool) -> None:
    if skip_existing and path.exists():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path)


def split_panel(
    panel_path: Path,
    tile_names: Sequence[str],
    out_dir: Path,
    skip_existing: bool,
) -> int:
    image = Image.open(panel_path).convert("RGB")
    w, h = image.size
    count = 0
    for tile_name, (left, right) in zip(tile_names, round_bounds(w, len(tile_names))):
        tile = image.crop((left, 0, right, h))
        save_if_needed(tile, out_dir / f"{tile_name}.png", skip_existing)
        count += 1
    return count


def iter_epoch_dirs(root: Path, epochs: str) -> Iterable[Path]:
    if epochs.strip():
        wanted = [item.strip() for item in epochs.split(",") if item.strip()]
        for name in wanted:
            path = root / name
            if path.is_dir():
                yield path
            else:
                print(f"[warn] missing epoch dir: {path}")
        return
    yield from sorted(path for path in root.glob("epoch_*") if path.is_dir())


def center_from_poly(poly: Sequence[Sequence[float]]) -> Optional[Tuple[float, float]]:
    try:
        arr = np.asarray(poly, dtype=np.float32).reshape(-1, 2)
    except Exception:
        return None
    if arr.size == 0:
        return None
    return float(arr[:, 0].mean()), float(arr[:, 1].mean())


def add_gaussian(heatmap: np.ndarray, cx: float, cy: float, weight: float, sigma: float) -> None:
    h, w = heatmap.shape
    if sigma <= 0:
        ix, iy = int(round(cx)), int(round(cy))
        if 0 <= ix < w and 0 <= iy < h:
            heatmap[iy, ix] += weight
        return
    radius = max(2, int(round(3.0 * sigma)))
    x0 = max(0, int(round(cx)) - radius)
    x1 = min(w, int(round(cx)) + radius + 1)
    y0 = max(0, int(round(cy)) - radius)
    y1 = min(h, int(round(cy)) + radius + 1)
    if x1 <= x0 or y1 <= y0:
        return
    yy, xx = np.mgrid[y0:y1, x0:x1]
    patch = np.exp(-0.5 * (((xx - cx) / sigma) ** 2 + ((yy - cy) / sigma) ** 2))
    heatmap[y0:y1, x0:x1] += float(weight) * patch


def normalize_map(values: np.ndarray) -> np.ndarray:
    values = np.nan_to_num(values.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    if float(values.max()) <= 1e-8:
        return np.zeros_like(values, dtype=np.float32)
    # Robust normalization keeps a few very high peaks from washing out the map.
    hi = float(np.quantile(values, 0.995))
    if hi <= 1e-8:
        hi = float(values.max())
    return np.clip(values / max(hi, 1e-8), 0.0, 1.0)


def heatmap_to_rgb(norm: np.ndarray) -> Image.Image:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        rgba = plt.get_cmap("jet")(norm)
        rgb = (rgba[:, :, :3] * 255).astype(np.uint8)
        return Image.fromarray(rgb, mode="RGB")
    except Exception:
        arr = (norm * 255).astype(np.uint8)
        return Image.fromarray(arr, mode="L").convert("RGB")


def load_original_image(pred_item: Dict[str, object], fallback_size: Tuple[int, int]) -> Image.Image:
    image_path = Path(str(pred_item.get("image_path", ""))).expanduser()
    if image_path.is_file():
        return Image.open(image_path).convert("RGB")
    return Image.new("RGB", fallback_size, (0, 0, 0))


def export_prediction_heatmaps(
    predictions_json: Path,
    output_epoch_dir: Path,
    sigma: float,
    alpha: float,
    skip_existing: bool,
) -> int:
    if not predictions_json.exists():
        return 0
    data = json.loads(predictions_json.read_text(encoding="utf-8"))
    images = data.get("images", []) if isinstance(data, dict) else data
    count = 0
    for item in images:
        if not isinstance(item, dict):
            continue
        width = int(item.get("width", 0) or 0)
        height = int(item.get("height", 0) or 0)
        if width <= 0 or height <= 0:
            continue
        image_path = Path(str(item.get("image_path", "")))
        image_stem = image_path.stem if image_path.name else "image"
        out_dir = output_epoch_dir / image_stem
        original = load_original_image(item, (width, height)).resize((width, height))
        preds = item.get("predictions", [])
        if not isinstance(preds, list):
            continue

        heatmaps = {
            "after_density": np.zeros((height, width), dtype=np.float32),
            "after_confidence": np.zeros((height, width), dtype=np.float32),
            "after_pattern_support": np.zeros((height, width), dtype=np.float32),
            "after_synthetic": np.zeros((height, width), dtype=np.float32),
        }

        for pred in preds:
            if not isinstance(pred, dict):
                continue
            center = center_from_poly(pred.get("poly", []))
            if center is None:
                continue
            cx, cy = center
            conf = float(pred.get("conf", pred.get("conf_after_rescore", 1.0)) or 0.0)
            support = float(pred.get("pattern_support", 0.0) or 0.0)
            synthetic = 1.0 if bool(pred.get("synthetic", False)) else 0.0
            add_gaussian(heatmaps["after_density"], cx, cy, 1.0, sigma)
            add_gaussian(heatmaps["after_confidence"], cx, cy, conf, sigma)
            add_gaussian(heatmaps["after_pattern_support"], cx, cy, support, sigma)
            if synthetic > 0:
                add_gaussian(heatmaps["after_synthetic"], cx, cy, synthetic, sigma)

        for name, heatmap in heatmaps.items():
            norm = normalize_map(heatmap)
            rgb = heatmap_to_rgb(norm)
            save_if_needed(rgb, out_dir / f"heatmap_{name}.png", skip_existing)

            overlay = Image.blend(original, rgb, alpha=max(0.0, min(1.0, float(alpha))))
            save_if_needed(overlay, out_dir / f"overlay_{name}.png", skip_existing)
            count += 2
    return count


def main() -> None:
    args = parse_args()
    debug_root = Path(args.debug_root).expanduser().resolve()
    output_root = Path(args.output_dir).expanduser().resolve() if args.output_dir else debug_root / "individual_exports"

    split_count = 0
    heatmap_count = 0
    for epoch_dir in iter_epoch_dirs(debug_root, args.epochs):
        output_epoch_dir = output_root / epoch_dir.name

        for feature_panel in sorted(epoch_dir.glob("*_feature_panel.png")):
            stem = stem_from_panel(feature_panel, "_feature_panel.png")
            out_dir = output_epoch_dir / stem
            split_count += split_panel(feature_panel, FEATURE_TILES, out_dir, bool(args.skip_existing))

        for prediction_panel in sorted(epoch_dir.glob("*_prediction_panel.png")):
            stem = stem_from_panel(prediction_panel, "_prediction_panel.png")
            out_dir = output_epoch_dir / stem
            split_count += split_panel(prediction_panel, PREDICTION_TILES, out_dir, bool(args.skip_existing))

        heatmap_count += export_prediction_heatmaps(
            epoch_dir / "predictions.json",
            output_epoch_dir,
            sigma=float(args.heatmap_sigma),
            alpha=float(args.heatmap_alpha),
            skip_existing=bool(args.skip_existing),
        )

    print(f"debug_root={debug_root}")
    print(f"output_root={output_root}")
    print(f"split_tiles={split_count}")
    print(f"heatmap_images={heatmap_count}")


if __name__ == "__main__":
    main()
