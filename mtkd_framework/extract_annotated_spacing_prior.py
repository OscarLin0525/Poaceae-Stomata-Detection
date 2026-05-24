#!/usr/bin/env python3
"""
Extract per-image row/count spacing priors from annotated result images.

The expected input is a folder of images where stomata are marked with bright
green rectangle outlines, e.g. result_*.jpg files exported by a detector UI.
The script recovers the green rectangles, groups them into horizontal rows, and
exports row-wise count/uniform-spacing priors that can be used as candidate
generators before local YOLO/DINO refinement.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp", ".JPG"}


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Extract rice/stomata row spacing priors from green-box annotated images.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--annotate-dir", type=str, required=True)
    p.add_argument("--image-dir", type=str, default=None, help="Optional raw image folder for preview backgrounds.")
    p.add_argument("--output-dir", type=str, required=True)
    p.add_argument("--max-images", type=int, default=0, help="0 = process all images.")
    p.add_argument("--min-green", type=int, default=120)
    p.add_argument("--green-delta", type=int, default=80)
    p.add_argument("--min-box-width", type=float, default=18.0)
    p.add_argument("--min-box-height", type=float, default=18.0)
    p.add_argument("--min-box-aspect", type=float, default=0.35)
    p.add_argument("--max-box-aspect", type=float, default=2.5)
    p.add_argument("--max-box-width-ratio", type=float, default=0.22)
    p.add_argument("--max-box-height-ratio", type=float, default=0.18)
    p.add_argument("--side-y-tolerance", type=float, default=5.0)
    p.add_argument("--line-density-threshold", type=float, default=0.55)
    p.add_argument("--row-tolerance-scale", type=float, default=0.75)
    p.add_argument("--min-row-boxes", type=int, default=2)
    p.add_argument("--preview-count", type=int, default=12)
    return p.parse_args()


def _list_images(path: Path) -> List[Path]:
    return sorted(p for p in path.iterdir() if p.is_file() and p.suffix in IMAGE_EXTS)


def _raw_name_from_annotated(path: Path) -> str:
    name = path.name
    if name.startswith("result_"):
        name = name[len("result_") :]
    return name


def _resolve_raw_image(image_dir: Optional[Path], annotated_path: Path) -> Optional[Path]:
    if image_dir is None:
        return None
    raw_name = _raw_name_from_annotated(annotated_path)
    direct = image_dir / raw_name
    if direct.is_file():
        return direct
    stem = Path(raw_name).stem
    for ext in IMAGE_EXTS:
        candidate = image_dir / f"{stem}{ext}"
        if candidate.is_file():
            return candidate
    return None


def _annotation_font(size: int) -> ImageFont.ImageFont:
    try:
        return ImageFont.truetype("DejaVuSans-Bold.ttf", size=max(8, int(size)))
    except OSError:
        return ImageFont.load_default()


def _extract_green_boxes(
    image_path: Path,
    *,
    min_green: int,
    green_delta: int,
    min_box_width: float,
    min_box_height: float,
    min_box_aspect: float,
    max_box_aspect: float,
    max_box_width_ratio: float,
    max_box_height_ratio: float,
    side_y_tolerance: float,
    line_density_threshold: float,
) -> Tuple[List[Dict[str, float]], np.ndarray]:
    image = Image.open(image_path).convert("RGB")
    arr = np.asarray(image, dtype=np.uint8)
    h, w = arr.shape[:2]
    r = arr[:, :, 0].astype(np.int16)
    g = arr[:, :, 1].astype(np.int16)
    b = arr[:, :, 2].astype(np.int16)
    mask = (
        (g >= int(min_green))
        & ((g - r) >= int(green_delta))
        & ((g - b) >= int(green_delta))
    ).astype(np.uint8)

    max_w = max(float(w) * float(max_box_width_ratio), float(min_box_width))
    max_h = max(float(h) * float(max_box_height_ratio), float(min_box_height))
    boxes = _extract_boxes_from_rectangle_sides(
        mask,
        min_box_width=float(min_box_width),
        min_box_height=float(min_box_height),
        min_box_aspect=float(min_box_aspect),
        max_box_aspect=float(max_box_aspect),
        max_box_width=float(max_w),
        max_box_height=float(max_h),
        side_y_tolerance=float(side_y_tolerance),
        line_density_threshold=float(line_density_threshold),
    )
    if boxes:
        return boxes, mask

    contours, _hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes = []
    for contour in contours:
        x, y, bw, bh = cv2.boundingRect(contour)
        if bw < min_box_width or bh < min_box_height:
            continue
        aspect = float(bw) / float(max(bh, 1))
        if aspect < min_box_aspect or aspect > max_box_aspect:
            continue
        if bw > max_w or bh > max_h:
            continue
        component_area = float(cv2.contourArea(contour))
        rect_area = float(max(bw * bh, 1))
        fill_ratio = component_area / rect_area
        if fill_ratio < 0.01 or fill_ratio > 0.75:
            continue
        boxes.append(
            {
                "x1": float(x),
                "y1": float(y),
                "x2": float(x + bw),
                "y2": float(y + bh),
                "cx": float(x + bw / 2.0),
                "cy": float(y + bh / 2.0),
                "w": float(bw),
                "h": float(bh),
                "fill_ratio": fill_ratio,
            }
        )
    boxes.sort(key=lambda item: (item["cy"], item["cx"]))
    return boxes, mask


def _rect_iou(a: Sequence[float], b: Sequence[float]) -> float:
    ax1, ay1, ax2, ay2 = [float(v) for v in a[:4]]
    bx1, by1, bx2, by2 = [float(v) for v in b[:4]]
    ix = max(0.0, min(ax2, bx2) - max(ax1, bx1))
    iy = max(0.0, min(ay2, by2) - max(ay1, by1))
    inter = ix * iy
    area_a = max((ax2 - ax1) * (ay2 - ay1), 1.0)
    area_b = max((bx2 - bx1) * (by2 - by1), 1.0)
    return inter / max(area_a + area_b - inter, 1.0)


def _extract_boxes_from_rectangle_sides(
    mask: np.ndarray,
    *,
    min_box_width: float,
    min_box_height: float,
    min_box_aspect: float,
    max_box_aspect: float,
    max_box_width: float,
    max_box_height: float,
    side_y_tolerance: float,
    line_density_threshold: float,
) -> List[Dict[str, float]]:
    h, w = mask.shape[:2]
    vertical_kernel_h = max(8, int(round(float(min_box_height) * 0.65)))
    vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, vertical_kernel_h))
    vertical_mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, vertical_kernel, iterations=1)
    contours, _hierarchy = cv2.findContours(vertical_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    sides: List[Tuple[float, float, float, float, float]] = []
    for contour in contours:
        x, y, bw, bh = cv2.boundingRect(contour)
        if bh < min_box_height * 0.8:
            continue
        if bw > max(24.0, min_box_width * 0.9):
            continue
        sides.append((float(x + bw / 2.0), float(y), float(y + bh), float(bw), float(bh)))

    def horizontal_density(x1: float, x2: float, y: float) -> float:
        y_int = int(round(float(y)))
        y0 = max(0, y_int - 3)
        y1 = min(h, y_int + 4)
        x0 = int(max(0, round(float(x1))))
        x3 = int(min(w, round(float(x2))))
        if x3 <= x0:
            return 0.0
        band = mask[y0:y1, x0:x3]
        if band.size == 0:
            return 0.0
        return float(band.max(axis=0).sum()) / float(max(x3 - x0, 1))

    candidates: List[Tuple[float, float, float, float, float]] = []
    tol = max(float(side_y_tolerance), 1.0)
    min_density = float(line_density_threshold)
    for i, side_a in enumerate(sides):
        ax, ay1, ay2, _aw, ah = side_a
        for side_b in sides[i + 1 :]:
            bx, by1, by2, _bw, bh = side_b
            x1, x2 = sorted((ax, bx))
            box_w = x2 - x1
            if box_w < min_box_width or box_w > max_box_width:
                continue
            if abs(ay1 - by1) > tol or abs(ay2 - by2) > tol:
                continue
            top = (ay1 + by1) / 2.0
            bottom = (ay2 + by2) / 2.0
            box_h = bottom - top
            if box_h < min_box_height or box_h > max_box_height:
                continue
            aspect = box_w / max(box_h, 1.0)
            if aspect < min_box_aspect or aspect > max_box_aspect:
                continue
            side_balance = min(ah, bh) / max(max(ah, bh), 1.0)
            if side_balance < 0.6:
                continue
            density = (horizontal_density(x1, x2, top) + horizontal_density(x1, x2, bottom)) / 2.0
            if density < min_density:
                continue
            score = density + side_balance
            candidates.append((x1, top, x2, bottom, score))

    selected: List[Tuple[float, float, float, float, float]] = []
    for candidate in sorted(candidates, key=lambda item: item[4], reverse=True):
        if all(_rect_iou(candidate, existing) < 0.2 for existing in selected):
            selected.append(candidate)

    boxes: List[Dict[str, float]] = []
    for x1, y1, x2, y2, score in selected:
        bw = x2 - x1
        bh = y2 - y1
        boxes.append(
            {
                "x1": float(x1),
                "y1": float(y1),
                "x2": float(x2),
                "y2": float(y2),
                "cx": float((x1 + x2) / 2.0),
                "cy": float((y1 + y2) / 2.0),
                "w": float(bw),
                "h": float(bh),
                "fill_ratio": float(score),
            }
        )
    boxes.sort(key=lambda item: (item["cy"], item["cx"]))
    return boxes


def _group_rows(boxes: Sequence[Dict[str, float]], row_tolerance_px: float) -> List[List[Dict[str, float]]]:
    if not boxes:
        return []
    ordered = sorted(boxes, key=lambda item: item["cy"])
    rows: List[List[Dict[str, float]]] = []
    current = [ordered[0]]
    current_y = float(ordered[0]["cy"])
    tol = max(float(row_tolerance_px), 1.0)
    for box in ordered[1:]:
        if abs(float(box["cy"]) - current_y) <= tol:
            current.append(box)
            current_y = float(np.mean([b["cy"] for b in current]))
        else:
            rows.append(sorted(current, key=lambda item: item["cx"]))
            current = [box]
            current_y = float(box["cy"])
    rows.append(sorted(current, key=lambda item: item["cx"]))
    return rows


def _row_record(image_name: str, image_size: Tuple[int, int], row_index: int, row: Sequence[Dict[str, float]]) -> Dict[str, float | int | str]:
    xs = np.asarray([float(b["cx"]) for b in row], dtype=np.float32)
    ys = np.asarray([float(b["cy"]) for b in row], dtype=np.float32)
    ws = np.asarray([float(b["w"]) for b in row], dtype=np.float32)
    hs = np.asarray([float(b["h"]) for b in row], dtype=np.float32)
    count = int(len(row))
    x_min = float(xs.min()) if count else 0.0
    x_max = float(xs.max()) if count else 0.0
    span = max(x_max - x_min, 0.0)
    uniform_period = float(span / max(count - 1, 1)) if count >= 2 else 0.0
    gaps = np.diff(np.sort(xs)) if count >= 2 else np.asarray([], dtype=np.float32)
    return {
        "image": image_name,
        "image_width": int(image_size[0]),
        "image_height": int(image_size[1]),
        "row_index": int(row_index),
        "row_y": float(np.median(ys)) if count else 0.0,
        "count": count,
        "x_min": x_min,
        "x_max": x_max,
        "x_span": span,
        "x_period_uniform": uniform_period,
        "x_period_median_gap": float(np.median(gaps)) if gaps.size else 0.0,
        "x_period_q25": float(np.percentile(gaps, 25)) if gaps.size else 0.0,
        "x_period_q75": float(np.percentile(gaps, 75)) if gaps.size else 0.0,
        "box_width_median": float(np.median(ws)) if count else 0.0,
        "box_height_median": float(np.median(hs)) if count else 0.0,
    }


def _uniform_points_for_row(row: Dict[str, float | int | str]) -> List[Tuple[float, float]]:
    count = int(row["count"])
    if count <= 0:
        return []
    if count == 1:
        return [(float(row["x_min"]), float(row["row_y"]))]
    xs = np.linspace(float(row["x_min"]), float(row["x_max"]), count)
    y = float(row["row_y"])
    return [(float(x), y) for x in xs.tolist()]


def _draw_preview(
    background_path: Path,
    out_path: Path,
    boxes: Sequence[Dict[str, float]],
    rows: Sequence[Dict[str, float | int | str]],
) -> None:
    image = Image.open(background_path).convert("RGB")
    draw = ImageDraw.Draw(image)
    scale = max(image.size) / 1920.0
    width = max(2, int(round(4 * scale)))
    radius = max(3, int(round(6 * scale)))
    font = _annotation_font(int(round(26 * scale)))

    for box in boxes:
        draw.rectangle(
            (box["x1"], box["y1"], box["x2"], box["y2"]),
            outline=(30, 230, 30),
            width=width,
        )

    for row in rows:
        y = float(row["row_y"])
        draw.line((0, y, image.width, y), fill=(80, 140, 255), width=max(1, width))
        for x, yy in _uniform_points_for_row(row):
            draw.line((x - radius, yy, x + radius, yy), fill=(255, 220, 30), width=width)
            draw.line((x, yy - radius, x, yy + radius), fill=(255, 220, 30), width=width)
        text = f"row {int(row['row_index'])} n={int(row['count'])} p={float(row['x_period_uniform']):.1f}"
        draw.text((8, y + 4), text, fill=(255, 220, 30), font=font)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(out_path)


def _summary(values: Sequence[float]) -> Dict[str, float | int]:
    arr = np.asarray([v for v in values if float(v) > 0.0], dtype=np.float32)
    if arr.size == 0:
        return {"count": 0, "median": 0.0, "mean": 0.0, "q25": 0.0, "q75": 0.0}
    return {
        "count": int(arr.size),
        "median": float(np.median(arr)),
        "mean": float(np.mean(arr)),
        "q25": float(np.percentile(arr, 25)),
        "q75": float(np.percentile(arr, 75)),
    }


def _save_csv(path: Path, rows: Sequence[Dict[str, object]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = _parse_args()
    annotate_dir = Path(args.annotate_dir)
    image_dir = Path(args.image_dir) if args.image_dir else None
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    preview_dir = output_dir / "previews"

    paths = _list_images(annotate_dir)
    if args.max_images > 0:
        paths = paths[: int(args.max_images)]
    if not paths:
        raise FileNotFoundError(f"No annotated images found under: {annotate_dir}")

    all_box_rows: List[Dict[str, object]] = []
    all_row_rows: List[Dict[str, object]] = []
    image_priors: List[Dict[str, object]] = []

    for image_index, annotated_path in enumerate(paths):
        image = Image.open(annotated_path)
        image_size = image.size
        boxes, _mask = _extract_green_boxes(
            annotated_path,
            min_green=args.min_green,
            green_delta=args.green_delta,
            min_box_width=args.min_box_width,
            min_box_height=args.min_box_height,
            min_box_aspect=args.min_box_aspect,
            max_box_aspect=args.max_box_aspect,
            max_box_width_ratio=args.max_box_width_ratio,
            max_box_height_ratio=args.max_box_height_ratio,
            side_y_tolerance=args.side_y_tolerance,
            line_density_threshold=args.line_density_threshold,
        )
        median_h = float(np.median([b["h"] for b in boxes])) if boxes else 32.0
        row_tolerance_px = max(8.0, median_h * float(args.row_tolerance_scale))
        grouped_rows = [
            row
            for row in _group_rows(boxes, row_tolerance_px)
            if len(row) >= int(args.min_row_boxes)
        ]
        row_records = [
            _row_record(_raw_name_from_annotated(annotated_path), image_size, idx, row)
            for idx, row in enumerate(grouped_rows)
        ]
        row_ys = np.asarray([float(r["row_y"]) for r in row_records], dtype=np.float32)
        row_gaps = np.diff(np.sort(row_ys)) if row_ys.size >= 2 else np.asarray([], dtype=np.float32)

        for box_idx, box in enumerate(boxes):
            all_box_rows.append(
                {
                    "image": _raw_name_from_annotated(annotated_path),
                    "box_index": box_idx,
                    **box,
                }
            )
        all_row_rows.extend(row_records)
        image_priors.append(
            {
                "annotated_image": str(annotated_path),
                "image": _raw_name_from_annotated(annotated_path),
                "image_width": int(image_size[0]),
                "image_height": int(image_size[1]),
                "box_count": int(len(boxes)),
                "row_count": int(len(row_records)),
                "row_tolerance_px": float(row_tolerance_px),
                "row_gap_median": float(np.median(row_gaps)) if row_gaps.size else 0.0,
                "rows": row_records,
            }
        )

        if image_index < int(args.preview_count):
            raw_path = _resolve_raw_image(image_dir, annotated_path)
            background = raw_path if raw_path is not None else annotated_path
            _draw_preview(
                background,
                preview_dir / f"{annotated_path.stem}_prior_preview.jpg",
                boxes,
                row_records,
            )

    summary = {
        "annotate_dir": str(annotate_dir),
        "image_dir": str(image_dir) if image_dir is not None else None,
        "num_images": int(len(image_priors)),
        "num_boxes": int(sum(int(item["box_count"]) for item in image_priors)),
        "num_rows": int(sum(int(item["row_count"]) for item in image_priors)),
        "x_period_uniform_px": _summary([float(row["x_period_uniform"]) for row in all_row_rows]),
        "x_period_median_gap_px": _summary([float(row["x_period_median_gap"]) for row in all_row_rows]),
        "row_gap_px": _summary([float(item["row_gap_median"]) for item in image_priors]),
        "box_width_px": _summary([float(row["box_width_median"]) for row in all_row_rows]),
        "box_height_px": _summary([float(row["box_height_median"]) for row in all_row_rows]),
        "output_dir": str(output_dir),
    }

    _save_csv(output_dir / "boxes.csv", all_box_rows)
    _save_csv(output_dir / "rows.csv", all_row_rows)
    with (output_dir / "image_priors.json").open("w", encoding="utf-8") as f:
        json.dump({"images": image_priors}, f, indent=2, ensure_ascii=False)
    with (output_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
