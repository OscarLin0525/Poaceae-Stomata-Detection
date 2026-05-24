#!/usr/bin/env python3
"""Analyze stomata row spacing/frequency from YOLO label files."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image


IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp")


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Estimate horizontal/vertical stomata frequency from YOLO bbox labels.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--label-dir", type=str, required=True)
    p.add_argument("--image-dir", type=str, default=None)
    p.add_argument("--output-dir", type=str, default="outputs/stomata_spacing_frequency")
    p.add_argument("--class-filter", type=str, default=None, help="Comma-separated class ids, empty = all.")
    p.add_argument("--row-tolerance-scale", type=float, default=1.35)
    p.add_argument("--min-row-points", type=int, default=3)
    p.add_argument("--gap-factor", type=float, default=1.55, help="Treat gaps above factor*period as missing candidates.")
    p.add_argument("--strides", type=str, default="8,16,32", help="Feature-map strides to report.")
    p.add_argument("--plot", action="store_true", help="Save histogram plots when matplotlib is available.")
    return p.parse_args()


def _parse_class_filter(spec: Optional[str]) -> Optional[set[int]]:
    if spec is None or not str(spec).strip():
        return None
    return {int(x.strip()) for x in str(spec).split(",") if x.strip()}


def _find_image(image_dir: Optional[Path], stem: str) -> Optional[Path]:
    if image_dir is None:
        return None
    for ext in IMAGE_EXTS:
        path = image_dir / f"{stem}{ext}"
        if path.is_file():
            return path
    return None


def _edge_lengths_px(points_norm: np.ndarray, image_size: Tuple[int, int]) -> Tuple[float, float]:
    w, h = image_size
    pts = points_norm.astype(np.float32).copy()
    pts[:, 0] *= float(w)
    pts[:, 1] *= float(h)
    top = float(np.linalg.norm(pts[1] - pts[0]))
    right = float(np.linalg.norm(pts[2] - pts[1]))
    bottom = float(np.linalg.norm(pts[2] - pts[3]))
    left = float(np.linalg.norm(pts[3] - pts[0]))
    return 0.5 * (top + bottom), 0.5 * (left + right)


def _read_label_file(
    label_path: Path,
    image_size: Tuple[int, int],
    class_filter: Optional[set[int]],
) -> List[Dict[str, float]]:
    width, height = image_size
    boxes: List[Dict[str, float]] = []
    for raw in label_path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) not in {5, 9}:
            continue
        cls_id = int(float(parts[0]))
        if class_filter is not None and cls_id not in class_filter:
            continue
        vals = np.asarray([float(v) for v in parts[1:]], dtype=np.float32)
        if len(vals) == 4:
            cx, cy, bw, bh = vals.tolist()
            bw_px = float(bw) * width
            bh_px = float(bh) * height
        else:
            pts = vals.reshape(4, 2)
            cx = float(pts[:, 0].mean())
            cy = float(pts[:, 1].mean())
            bw_px, bh_px = _edge_lengths_px(pts, image_size)
        boxes.append(
            {
                "cls": float(cls_id),
                "cx_norm": float(cx),
                "cy_norm": float(cy),
                "cx_px": float(cx) * width,
                "cy_px": float(cy) * height,
                "width_px": bw_px,
                "height_px": bh_px,
            }
        )
    return boxes


def _group_rows(boxes: Sequence[Dict[str, float]], tolerance_px: float) -> List[List[Dict[str, float]]]:
    if not boxes:
        return []
    ordered = sorted(boxes, key=lambda b: b["cy_px"])
    rows: List[List[Dict[str, float]]] = []
    current: List[Dict[str, float]] = [ordered[0]]
    current_y = float(ordered[0]["cy_px"])
    tol = max(float(tolerance_px), 1.0)
    for box in ordered[1:]:
        y = float(box["cy_px"])
        if abs(y - current_y) <= tol:
            current.append(box)
            current_y = float(np.mean([b["cy_px"] for b in current]))
        else:
            rows.append(sorted(current, key=lambda b: b["cx_px"]))
            current = [box]
            current_y = y
    rows.append(sorted(current, key=lambda b: b["cx_px"]))
    return rows


def _robust_period(values: Sequence[float]) -> Dict[str, float]:
    arr = np.asarray([v for v in values if v > 0], dtype=np.float32)
    if arr.size == 0:
        return {"median": 0.0, "mean": 0.0, "q25": 0.0, "q75": 0.0, "std": 0.0, "count": 0.0}
    return {
        "median": float(np.median(arr)),
        "mean": float(np.mean(arr)),
        "q25": float(np.percentile(arr, 25)),
        "q75": float(np.percentile(arr, 75)),
        "std": float(np.std(arr)),
        "count": float(arr.size),
    }


def _row_period(row: Sequence[Dict[str, float]]) -> Tuple[float, List[float]]:
    xs = np.asarray([b["cx_px"] for b in row], dtype=np.float32)
    if xs.size < 2:
        return 0.0, []
    diffs = np.diff(np.sort(xs))
    diffs = diffs[diffs > 1.0]
    if diffs.size == 0:
        return 0.0, []
    # Use the lower half mode-ish spacing to avoid missing-label gaps inflating the period.
    q75 = float(np.percentile(diffs, 75))
    base = diffs[diffs <= q75]
    period = float(np.median(base if base.size else diffs))
    return period, [float(x) for x in diffs.tolist()]


def _linear_y_at_x(row: Sequence[Dict[str, float]], x: float) -> float:
    xs = np.asarray([b["cx_px"] for b in row], dtype=np.float32)
    ys = np.asarray([b["cy_px"] for b in row], dtype=np.float32)
    if xs.size < 2 or float(np.std(xs)) < 1e-6:
        return float(np.median(ys)) if ys.size else 0.0
    slope, intercept = np.polyfit(xs, ys, deg=1)
    return float(slope * x + intercept)


def _infer_missing_candidates(
    row: Sequence[Dict[str, float]],
    period_px: float,
    gap_factor: float,
    image_name: str,
    row_index: int,
) -> List[Dict[str, float]]:
    if period_px <= 1.0 or len(row) < 2:
        return []
    candidates: List[Dict[str, float]] = []
    xs = [float(b["cx_px"]) for b in row]
    for left, right in zip(xs[:-1], xs[1:]):
        gap = right - left
        if gap <= float(gap_factor) * period_px:
            continue
        missing = max(0, int(round(gap / period_px)) - 1)
        if missing <= 0:
            continue
        adjusted = gap / float(missing + 1)
        for k in range(1, missing + 1):
            x = left + adjusted * k
            y = _linear_y_at_x(row, x)
            candidates.append(
                {
                    "image": image_name,
                    "row_index": float(row_index),
                    "x_px": float(x),
                    "y_px": float(y),
                    "gap_px": float(gap),
                    "row_period_px": float(period_px),
                    "adjusted_period_px": float(adjusted),
                    "missing_in_gap": float(missing),
                }
            )
    return candidates


def _write_csv(path: Path, rows: Sequence[Dict[str, object]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _plot_hist(values: Sequence[float], path: Path, title: str, xlabel: str) -> None:
    if not values:
        return
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return
    plt.figure(figsize=(8, 5))
    plt.hist(values, bins=30, color="#3572A5", alpha=0.85)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel("count")
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()


def main() -> None:
    args = _parse_args()
    label_dir = Path(args.label_dir).resolve()
    image_dir = Path(args.image_dir).resolve() if args.image_dir else None
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    if not label_dir.is_dir():
        raise FileNotFoundError(f"Label directory not found: {label_dir}")
    class_filter = _parse_class_filter(args.class_filter)
    strides = [int(s.strip()) for s in str(args.strides).split(",") if s.strip()]

    per_image_rows: List[Dict[str, object]] = []
    per_row_rows: List[Dict[str, object]] = []
    h_spacing_rows: List[Dict[str, object]] = []
    v_spacing_rows: List[Dict[str, object]] = []
    missing_rows: List[Dict[str, object]] = []
    all_h_spacings: List[float] = []
    all_v_gaps: List[float] = []
    all_widths: List[float] = []
    all_heights: List[float] = []

    for label_path in sorted(label_dir.glob("*.txt")):
        image_path = _find_image(image_dir, label_path.stem)
        image_size = (1, 1)
        if image_path is not None:
            with Image.open(image_path) as img:
                image_size = img.size
        boxes = _read_label_file(label_path, image_size, class_filter)
        if not boxes:
            continue
        all_widths.extend([b["width_px"] for b in boxes])
        all_heights.extend([b["height_px"] for b in boxes])

        median_h = float(np.median([b["height_px"] for b in boxes]))
        tolerance_px = max(8.0, median_h * float(args.row_tolerance_scale))
        rows = _group_rows(boxes, tolerance_px=tolerance_px)
        valid_rows = [row for row in rows if len(row) >= int(args.min_row_points)]
        row_y = [float(np.median([b["cy_px"] for b in row])) for row in valid_rows]
        v_gaps = np.diff(np.sort(np.asarray(row_y, dtype=np.float32))).tolist() if len(row_y) >= 2 else []
        all_v_gaps.extend([float(v) for v in v_gaps if v > 0])

        image_h_spacings: List[float] = []
        row_periods: List[float] = []
        for row_index, row in enumerate(valid_rows):
            period, diffs = _row_period(row)
            if period > 0:
                row_periods.append(period)
            for d in diffs:
                image_h_spacings.append(d)
                all_h_spacings.append(d)
                h_spacing_rows.append(
                    {
                        "image": label_path.stem,
                        "row_index": row_index,
                        "spacing_px": d,
                        "spacing_norm_x": d / float(image_size[0]),
                    }
                )
            per_row_rows.append(
                {
                    "image": label_path.stem,
                    "row_index": row_index,
                    "num_boxes": len(row),
                    "row_y_px": float(np.median([b["cy_px"] for b in row])),
                    "row_period_px": period,
                    "row_period_norm_x": period / float(image_size[0]) if period > 0 else 0.0,
                    "row_frequency_cyc_per_px": 1.0 / period if period > 0 else 0.0,
                }
            )
            missing_rows.extend(
                _infer_missing_candidates(
                    row=row,
                    period_px=period,
                    gap_factor=float(args.gap_factor),
                    image_name=label_path.stem,
                    row_index=row_index,
                )
            )
        for gap in v_gaps:
            v_spacing_rows.append(
                {
                    "image": label_path.stem,
                    "row_gap_px": float(gap),
                    "row_gap_norm_y": float(gap) / float(image_size[1]),
                }
            )

        h_stats = _robust_period(image_h_spacings)
        v_stats = _robust_period([float(v) for v in v_gaps])
        per_image_rows.append(
            {
                "image": label_path.stem,
                "image_width": image_size[0],
                "image_height": image_size[1],
                "num_boxes": len(boxes),
                "num_rows": len(valid_rows),
                "row_tolerance_px": tolerance_px,
                "median_box_width_px": float(np.median([b["width_px"] for b in boxes])),
                "median_box_height_px": median_h,
                "horizontal_period_median_px": h_stats["median"],
                "horizontal_period_q25_px": h_stats["q25"],
                "horizontal_period_q75_px": h_stats["q75"],
                "vertical_row_gap_median_px": v_stats["median"],
                "vertical_row_gap_q25_px": v_stats["q25"],
                "vertical_row_gap_q75_px": v_stats["q75"],
                "missing_candidate_count": sum(1 for r in missing_rows if r["image"] == label_path.stem),
            }
        )

    h_stats = _robust_period(all_h_spacings)
    v_stats = _robust_period(all_v_gaps)
    width_stats = _robust_period(all_widths)
    height_stats = _robust_period(all_heights)
    stride_summary = {
        f"stride_{stride}": {
            "horizontal_period_cells": h_stats["median"] / stride if h_stats["median"] else 0.0,
            "vertical_row_gap_cells": v_stats["median"] / stride if v_stats["median"] else 0.0,
            "box_width_cells": width_stats["median"] / stride if width_stats["median"] else 0.0,
            "box_height_cells": height_stats["median"] / stride if height_stats["median"] else 0.0,
        }
        for stride in strides
    }

    summary = {
        "label_dir": str(label_dir),
        "image_dir": str(image_dir) if image_dir else None,
        "num_images": len(per_image_rows),
        "num_boxes": int(sum(int(r["num_boxes"]) for r in per_image_rows)),
        "num_rows": int(sum(int(r["num_rows"]) for r in per_image_rows)),
        "horizontal_period_px": h_stats,
        "vertical_row_gap_px": v_stats,
        "box_width_px": width_stats,
        "box_height_px": height_stats,
        "horizontal_frequency_cyc_per_px": 1.0 / h_stats["median"] if h_stats["median"] else 0.0,
        "vertical_frequency_cyc_per_px": 1.0 / v_stats["median"] if v_stats["median"] else 0.0,
        "missing_candidate_count": len(missing_rows),
        "feature_map_recommendation": {
            "preferred_level": "p3",
            "row_tolerance_px": float(np.median([r["row_tolerance_px"] for r in per_image_rows])) if per_image_rows else 0.0,
            "gap_factor": float(args.gap_factor),
            "strides": stride_summary,
        },
    }

    _write_csv(output_dir / "per_image_spacing.csv", per_image_rows)
    _write_csv(output_dir / "per_row_frequency.csv", per_row_rows)
    _write_csv(output_dir / "horizontal_spacings.csv", h_spacing_rows)
    _write_csv(output_dir / "vertical_row_gaps.csv", v_spacing_rows)
    _write_csv(output_dir / "missing_candidates.csv", missing_rows)
    with (output_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    if args.plot:
        _plot_hist(all_h_spacings, output_dir / "horizontal_spacing_hist.png", "Horizontal stomata spacing", "px")
        _plot_hist(all_v_gaps, output_dir / "vertical_row_gap_hist.png", "Vertical row gap", "px")

    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
