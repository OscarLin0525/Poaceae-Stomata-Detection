#!/usr/bin/env python3
"""Convert stomata YOLO-OBB labels to horizontal-box baseline formats.

Outputs:
- YOLO horizontal bbox labels for ConfMix / YOLOv5:
  ``class cx cy w h`` normalized to [0, 1].
- COCO detection json for DINO_Teacher / Detectron2:
  horizontal ``bbox=[x, y, w, h]`` in pixels.

The conversion is intentionally conservative: no boxes are rotated, removed, or
reclassified.  A YOLO-OBB quadrilateral is converted to the smallest
axis-aligned box that encloses its four corners.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from PIL import Image


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
DEFAULT_NAMES = ["complete", "incomplete"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-root",
        type=Path,
        required=True,
        help="Dataset root containing images/<split> and labels/<split>.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        required=True,
        help="Output root for converted files.",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["train", "val", "test"],
        help="Splits to convert.",
    )
    parser.add_argument(
        "--formats",
        nargs="+",
        choices=["yolo", "coco"],
        default=["yolo", "coco"],
        help="Output formats to generate.",
    )
    parser.add_argument(
        "--names",
        nargs="+",
        default=DEFAULT_NAMES,
        help="Class names in class-id order.",
    )
    parser.add_argument(
        "--copy-images",
        action="store_true",
        help="Copy images into output-root/images/<split>. Default is symlink.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing output labels/json files.",
    )
    parser.add_argument(
        "--category-id-offset",
        type=int,
        default=0,
        help="COCO category id offset. Use 0 for Detectron2, 1 for strict COCO style.",
    )
    return parser.parse_args()


def image_files(image_dir: Path) -> List[Path]:
    if not image_dir.is_dir():
        return []
    return sorted(p for p in image_dir.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES)


def label_path_for_image(image_path: Path, label_dir: Path) -> Path:
    return label_dir / f"{image_path.stem}.txt"


def ensure_empty_or_overwrite(path: Path, overwrite: bool) -> None:
    if path.exists() and any(path.iterdir()) and not overwrite:
        raise FileExistsError(f"Output directory is non-empty; pass --overwrite: {path}")
    path.mkdir(parents=True, exist_ok=True)


def link_or_copy_image(src: Path, dst: Path, copy_images: bool) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        return
    if copy_images:
        shutil.copy2(src, dst)
        return
    try:
        dst.symlink_to(src.resolve())
    except OSError:
        shutil.copy2(src, dst)


def parse_yolo_line(line: str) -> Optional[Tuple[int, List[float]]]:
    fields = line.strip().split()
    if not fields:
        return None
    try:
        cls_id = int(float(fields[0]))
        values = [float(x) for x in fields[1:]]
    except ValueError:
        return None
    if len(values) not in {4, 8}:
        return None
    return cls_id, values


def yolo_values_to_xyxy_norm(values: Sequence[float]) -> Optional[Tuple[float, float, float, float]]:
    if len(values) == 8:
        xs = [float(values[i]) for i in range(0, 8, 2)]
        ys = [float(values[i]) for i in range(1, 8, 2)]
        x1, y1, x2, y2 = min(xs), min(ys), max(xs), max(ys)
    elif len(values) == 4:
        cx, cy, w, h = [float(v) for v in values]
        x1, y1 = cx - 0.5 * w, cy - 0.5 * h
        x2, y2 = cx + 0.5 * w, cy + 0.5 * h
    else:
        return None
    x1 = min(max(x1, 0.0), 1.0)
    y1 = min(max(y1, 0.0), 1.0)
    x2 = min(max(x2, 0.0), 1.0)
    y2 = min(max(y2, 0.0), 1.0)
    if x2 <= x1 or y2 <= y1:
        return None
    return x1, y1, x2, y2


def xyxy_norm_to_yolo_hbox(xyxy: Tuple[float, float, float, float]) -> Tuple[float, float, float, float]:
    x1, y1, x2, y2 = xyxy
    w = x2 - x1
    h = y2 - y1
    return x1 + 0.5 * w, y1 + 0.5 * h, w, h


def xyxy_norm_to_coco_bbox(
    xyxy: Tuple[float, float, float, float],
    width: int,
    height: int,
) -> Tuple[float, float, float, float]:
    x1, y1, x2, y2 = xyxy
    px1 = x1 * float(width)
    py1 = y1 * float(height)
    px2 = x2 * float(width)
    py2 = y2 * float(height)
    return px1, py1, max(0.0, px2 - px1), max(0.0, py2 - py1)


def read_labels(label_path: Path) -> List[Tuple[int, Tuple[float, float, float, float]]]:
    boxes: List[Tuple[int, Tuple[float, float, float, float]]] = []
    if not label_path.is_file():
        return boxes
    for line in label_path.read_text(encoding="utf-8").splitlines():
        parsed = parse_yolo_line(line)
        if parsed is None:
            continue
        cls_id, values = parsed
        xyxy = yolo_values_to_xyxy_norm(values)
        if xyxy is None:
            continue
        boxes.append((cls_id, xyxy))
    return boxes


def write_yolo_split(
    *,
    input_root: Path,
    output_root: Path,
    split: str,
    copy_images: bool,
) -> Dict[str, int]:
    in_images = input_root / "images" / split
    in_labels = input_root / "labels" / split
    out_images = output_root / "images" / split
    out_labels = output_root / "labels" / split
    out_labels.mkdir(parents=True, exist_ok=True)
    out_images.mkdir(parents=True, exist_ok=True)

    image_count = 0
    box_count = 0
    for image_path in image_files(in_images):
        image_count += 1
        link_or_copy_image(image_path, out_images / image_path.name, copy_images)
        labels = read_labels(label_path_for_image(image_path, in_labels))
        lines: List[str] = []
        for cls_id, xyxy in labels:
            cx, cy, w, h = xyxy_norm_to_yolo_hbox(xyxy)
            lines.append(f"{cls_id} {cx:.10f} {cy:.10f} {w:.10f} {h:.10f}")
        box_count += len(lines)
        (out_labels / f"{image_path.stem}.txt").write_text(
            "\n".join(lines) + ("\n" if lines else ""),
            encoding="utf-8",
        )
    return {"images": image_count, "boxes": box_count}


def write_coco_split(
    *,
    input_root: Path,
    output_root: Path,
    split: str,
    names: Sequence[str],
    category_id_offset: int,
    copy_images: bool,
) -> Dict[str, int]:
    in_images = input_root / "images" / split
    in_labels = input_root / "labels" / split
    out_images = output_root / "images" / split
    ann_dir = output_root / "annotations"
    out_images.mkdir(parents=True, exist_ok=True)
    ann_dir.mkdir(parents=True, exist_ok=True)

    coco_images: List[Dict[str, object]] = []
    coco_annotations: List[Dict[str, object]] = []
    ann_id = 1
    for image_id, image_path in enumerate(image_files(in_images), start=1):
        link_or_copy_image(image_path, out_images / image_path.name, copy_images)
        with Image.open(image_path) as im:
            width, height = im.size
        coco_images.append(
            {
                "id": image_id,
                "file_name": image_path.name,
                "width": width,
                "height": height,
            }
        )
        for cls_id, xyxy in read_labels(label_path_for_image(image_path, in_labels)):
            x, y, w, h = xyxy_norm_to_coco_bbox(xyxy, width, height)
            if w <= 0.0 or h <= 0.0:
                continue
            coco_annotations.append(
                {
                    "id": ann_id,
                    "image_id": image_id,
                    "category_id": int(cls_id) + int(category_id_offset),
                    "bbox": [round(x, 4), round(y, 4), round(w, 4), round(h, 4)],
                    "area": round(w * h, 4),
                    "iscrowd": 0,
                }
            )
            ann_id += 1

    categories = [
        {"id": idx + int(category_id_offset), "name": name}
        for idx, name in enumerate(names)
    ]
    payload = {
        "images": coco_images,
        "annotations": coco_annotations,
        "categories": categories,
    }
    (ann_dir / f"{split}.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return {"images": len(coco_images), "boxes": len(coco_annotations)}


def main() -> None:
    args = parse_args()
    ensure_empty_or_overwrite(args.output_root, args.overwrite)

    summary: Dict[str, Dict[str, Dict[str, int]]] = {}
    for split in args.splits:
        summary[split] = {}
        if "yolo" in args.formats:
            summary[split]["yolo"] = write_yolo_split(
                input_root=args.input_root,
                output_root=args.output_root / "yolo_hbox",
                split=split,
                copy_images=args.copy_images,
            )
        if "coco" in args.formats:
            summary[split]["coco"] = write_coco_split(
                input_root=args.input_root,
                output_root=args.output_root / "coco_hbox",
                split=split,
                names=args.names,
                category_id_offset=args.category_id_offset,
                copy_images=args.copy_images,
            )

    report = {
        "input_root": str(args.input_root),
        "output_root": str(args.output_root),
        "splits": args.splits,
        "formats": args.formats,
        "class_names": list(args.names),
        "category_id_offset": int(args.category_id_offset),
        "summary": summary,
    }
    report_path = args.output_root / "conversion_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"Saved report: {report_path}")


if __name__ == "__main__":
    main()
