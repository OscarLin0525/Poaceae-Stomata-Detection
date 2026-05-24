#!/usr/bin/env python3
"""Draw YOLO labels on images.

Supports:
  - YOLO OBB polygon labels: cls x1 y1 x2 y2 x3 y3 x4 y4
  - YOLO OBB polygon + confidence: cls x1 ... y4 conf
  - YOLO box labels: cls cx cy w h
  - YOLO box + confidence: cls cx cy w h conf

Examples:
  # Draw one image/label pair
  python draw_yolo_labels.py \
    --image BARLEY/20%/images/test/20240829103457.jpg \
    --label BARLEY/20%/labels/test/20240829103457.txt \
    --output outputs/label_vis

  # Draw a whole split
  python draw_yolo_labels.py \
    --image-dir BARLEY/20%/images/test \
    --label-dir BARLEY/20%/labels/test \
    --output outputs/label_vis/barley20_test

  # Customize classes, colors, and line width
  python draw_yolo_labels.py \
    --image-dir RICE/images/train \
    --label-dir RICE/labels/train \
    --class-names "0:complete,1:incomplete" \
    --colors "0:#00FFFF,1:#FF00FF" \
    --line-width 5 \
    --font-size 24 \
    --output outputs/label_vis/rice_train
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from PIL import Image, ImageColor, ImageDraw, ImageFont


IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")
DEFAULT_COLORS = {
    0: "#00D5FF",
    1: "#FF2BC2",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Draw YOLO txt labels on image(s).")
    src = parser.add_argument_group("input")
    src.add_argument("--image", type=str, default=None, help="Single image path.")
    src.add_argument("--label", type=str, default=None, help="Single label .txt path.")
    src.add_argument("--image-dir", type=str, default=None, help="Directory containing images.")
    src.add_argument("--label-dir", type=str, default=None, help="Directory containing label .txt files.")
    src.add_argument("--output", type=str, required=True, help="Output image file or directory.")

    style = parser.add_argument_group("style")
    style.add_argument(
        "--class-names",
        type=str,
        default="0:class 0,1:class 1",
        help='Class names, e.g. "0:complete,1:incomplete" or "complete,incomplete".',
    )
    style.add_argument(
        "--colors",
        type=str,
        default="0:#00D5FF,1:#FF2BC2",
        help='Class colors, e.g. "0:red,1:#00FF00".',
    )
    style.add_argument("--line-width", type=int, default=4, help="Bounding box line width.")
    style.add_argument("--font-size", type=int, default=22, help="Class label font size.")
    style.add_argument("--hide-labels", action="store_true", help="Do not draw class names.")
    style.add_argument("--hide-conf", action="store_true", help="Do not draw confidence if label includes it.")
    style.add_argument("--label-bg-alpha", type=int, default=190, help="Label background alpha, 0-255.")

    misc = parser.add_argument_group("misc")
    misc.add_argument("--suffix", type=str, default="_labels", help="Output suffix for directory mode.")
    misc.add_argument("--max-images", type=int, default=0, help="Limit images in directory mode; 0 means all.")
    misc.add_argument("--overwrite", action="store_true", help="Overwrite existing output files.")
    return parser.parse_args()


def parse_mapping(text: str, default_prefix: str = "class") -> Dict[int, str]:
    if not text:
        return {}
    out: Dict[int, str] = {}
    parts = [p.strip() for p in text.split(",") if p.strip()]
    if all(":" in p for p in parts):
        for part in parts:
            key, value = part.split(":", 1)
            out[int(key.strip())] = value.strip()
    else:
        for idx, value in enumerate(parts):
            out[idx] = value
    return out or {0: f"{default_prefix} 0", 1: f"{default_prefix} 1"}


def parse_colors(text: str) -> Dict[int, Tuple[int, int, int]]:
    raw = parse_mapping(text)
    colors: Dict[int, Tuple[int, int, int]] = {}
    for cls_id, value in raw.items():
        colors[cls_id] = ImageColor.getrgb(str(value))
    for cls_id, color in DEFAULT_COLORS.items():
        colors.setdefault(cls_id, ImageColor.getrgb(color))
    return colors


def load_font(font_size: int) -> ImageFont.ImageFont:
    candidates = [
        "/usr/share/fonts/truetype/msttcorefonts/Times_New_Roman.ttf",
        "/usr/share/fonts/truetype/liberation2/LiberationSerif-Regular.ttf",
        "/usr/share/fonts/opentype/urw-base35/NimbusRoman-Regular.otf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSerif.ttf",
    ]
    for path in candidates:
        if Path(path).is_file():
            try:
                return ImageFont.truetype(path, font_size)
            except OSError:
                continue
    return ImageFont.load_default()


def denorm_points(values: Sequence[float], width: int, height: int) -> List[Tuple[float, float]]:
    coords = list(values)
    normalized = bool(coords) and max(abs(v) for v in coords) <= 1.5
    points: List[Tuple[float, float]] = []
    for x, y in zip(coords[0::2], coords[1::2]):
        px = x * width if normalized else x
        py = y * height if normalized else y
        points.append((float(px), float(py)))
    return points


def xywh_to_polygon(values: Sequence[float], width: int, height: int) -> List[Tuple[float, float]]:
    cx, cy, bw, bh = [float(v) for v in values[:4]]
    normalized = max(abs(cx), abs(cy), abs(bw), abs(bh)) <= 1.5
    if normalized:
        cx, bw = cx * width, bw * width
        cy, bh = cy * height, bh * height
    x0, x1 = cx - bw / 2.0, cx + bw / 2.0
    y0, y1 = cy - bh / 2.0, cy + bh / 2.0
    return [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]


def parse_label_file(label_path: Path, width: int, height: int) -> List[Dict[str, object]]:
    boxes: List[Dict[str, object]] = []
    if not label_path.exists():
        return boxes

    for line_no, line in enumerate(label_path.read_text(encoding="utf-8").splitlines(), start=1):
        line = line.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) < 5:
            print(f"[warn] skip short label {label_path}:{line_no}: {line}")
            continue
        try:
            cls_id = int(float(parts[0]))
            vals = [float(v) for v in parts[1:]]
        except ValueError:
            print(f"[warn] skip invalid label {label_path}:{line_no}: {line}")
            continue

        conf: Optional[float] = None
        if len(vals) == 4:
            poly = xywh_to_polygon(vals, width, height)
        elif len(vals) == 5:
            poly = xywh_to_polygon(vals[:4], width, height)
            conf = vals[4]
        elif len(vals) == 8:
            poly = denorm_points(vals, width, height)
        elif len(vals) == 9:
            poly = denorm_points(vals[:8], width, height)
            conf = vals[8]
        else:
            print(f"[warn] skip unsupported label {label_path}:{line_no}: {line}")
            continue
        boxes.append({"cls": cls_id, "poly": poly, "conf": conf})
    return boxes


def text_bbox(draw: ImageDraw.ImageDraw, xy: Tuple[float, float], text: str, font: ImageFont.ImageFont):
    try:
        return draw.textbbox(xy, text, font=font)
    except AttributeError:
        w, h = draw.textsize(text, font=font)
        x, y = xy
        return (x, y, x + w, y + h)


def draw_boxes(
    image_path: Path,
    label_path: Path,
    output_path: Path,
    class_names: Dict[int, str],
    colors: Dict[int, Tuple[int, int, int]],
    line_width: int,
    font_size: int,
    hide_labels: bool,
    hide_conf: bool,
    label_bg_alpha: int,
) -> int:
    image = Image.open(image_path).convert("RGB")
    width, height = image.size
    boxes = parse_label_file(label_path, width, height)
    draw = ImageDraw.Draw(image, "RGBA")
    font = load_font(font_size)

    for box in boxes:
        cls_id = int(box["cls"])
        poly = [(float(x), float(y)) for x, y in box["poly"]]
        color = colors.get(cls_id, (255, 255, 0))
        rgba = (*color, 255)
        draw.line(poly + [poly[0]], fill=rgba, width=max(1, int(line_width)), joint="curve")

        if hide_labels:
            continue
        name = class_names.get(cls_id, f"class {cls_id}")
        conf = box.get("conf")
        if conf is not None and not hide_conf:
            text = f"{name} {float(conf):.2f}"
        else:
            text = str(name)

        x_min = min(x for x, _ in poly)
        y_min = min(y for _, y in poly)
        x_text = max(0.0, x_min)
        y_text = max(0.0, y_min - font_size - 6)
        bbox = text_bbox(draw, (x_text, y_text), text, font)
        pad = 3
        bg = (*color, max(0, min(255, int(label_bg_alpha))))
        draw.rectangle(
            (bbox[0] - pad, bbox[1] - pad, bbox[2] + pad, bbox[3] + pad),
            fill=bg,
        )
        draw.text((x_text, y_text), text, font=font, fill=(0, 0, 0, 255))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path)
    return len(boxes)


def find_image_for_label(image_dir: Path, stem: str) -> Optional[Path]:
    for ext in IMAGE_EXTS:
        candidate = image_dir / f"{stem}{ext}"
        if candidate.exists():
            return candidate
    return None


def iter_pairs(args: argparse.Namespace) -> Iterable[Tuple[Path, Path, Path]]:
    output = Path(args.output).expanduser().resolve()

    if args.image and args.label:
        image_path = Path(args.image).expanduser().resolve()
        label_path = Path(args.label).expanduser().resolve()
        if output.suffix:
            out_path = output
        else:
            out_path = output / f"{image_path.stem}{args.suffix}{image_path.suffix}"
        yield image_path, label_path, out_path
        return

    if not args.image_dir or not args.label_dir:
        raise ValueError("Use either --image/--label or --image-dir/--label-dir.")

    image_dir = Path(args.image_dir).expanduser().resolve()
    label_dir = Path(args.label_dir).expanduser().resolve()
    limit = max(0, int(args.max_images))
    count = 0
    for label_path in sorted(label_dir.glob("*.txt")):
        image_path = find_image_for_label(image_dir, label_path.stem)
        if image_path is None:
            print(f"[warn] no image for label: {label_path.name}")
            continue
        out_path = output / f"{image_path.stem}{args.suffix}{image_path.suffix}"
        yield image_path, label_path, out_path
        count += 1
        if limit and count >= limit:
            break


def main() -> None:
    args = parse_args()
    class_names = parse_mapping(args.class_names)
    colors = parse_colors(args.colors)

    rendered = 0
    for image_path, label_path, output_path in iter_pairs(args):
        if output_path.exists() and not args.overwrite:
            print(f"[skip] exists: {output_path}")
            continue
        n = draw_boxes(
            image_path=image_path,
            label_path=label_path,
            output_path=output_path,
            class_names=class_names,
            colors=colors,
            line_width=args.line_width,
            font_size=args.font_size,
            hide_labels=bool(args.hide_labels),
            hide_conf=bool(args.hide_conf),
            label_bg_alpha=int(args.label_bg_alpha),
        )
        print(f"[ok] {image_path.name}: boxes={n} -> {output_path}")
        rendered += 1
    print(f"[done] rendered={rendered}")


if __name__ == "__main__":
    main()
