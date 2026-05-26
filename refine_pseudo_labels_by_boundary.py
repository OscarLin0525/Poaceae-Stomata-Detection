#!/usr/bin/env python3
"""Refine OBB pseudo-label classes using distance to the image boundary.

This tool is deliberately separate from spatial completion. It does not add,
remove, or reshape boxes by default; it promotes class-0 pseudo boxes within a
configured hard boundary zone to class 1 and records the resulting statistics.
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Promote boundary-adjacent class-0 YOLO OBB pseudo labels to class 1 "
            "and export boundary-distance statistics and figures."
        )
    )
    parser.add_argument("--input-root", type=Path, required=True, help="Dataset root containing images/<split> and labels/<split>.")
    parser.add_argument("--output-root", type=Path, required=True, help="New refined dataset and report output directory.")
    parser.add_argument("--splits", nargs="+", default=["train"], help="Dataset splits to process, for example: train val.")
    parser.add_argument("--source-class", type=int, default=0, help="Pseudo-label class eligible for boundary promotion.")
    parser.add_argument("--target-class", type=int, default=1, help="Class assigned to promoted boundary instances.")
    parser.add_argument(
        "--hard-threshold-percent",
        type=float,
        required=True,
        help="Hard boundary threshold as percent of shorter image side, e.g. 0.21 means 0.21%%.",
    )
    parser.add_argument(
        "--soft-threshold-percent",
        type=float,
        default=None,
        help="Optional ambiguous-zone outer threshold as percent of shorter image side.",
    )
    parser.add_argument(
        "--soft-policy",
        choices=["keep", "ignore", "promote"],
        default="keep",
        help="Treatment for class-0 boxes in the soft-only boundary zone.",
    )
    parser.add_argument(
        "--copy-images",
        action="store_true",
        help="Copy source images to output images/<split>, creating a training-ready output dataset.",
    )
    parser.add_argument("--overwrite", action="store_true", help="Allow writing to a non-empty output directory.")
    parser.add_argument(
        "--plot-max-percent",
        type=float,
        default=6.0,
        help="Minimum upper boundary-distance percentage displayed in the histogram.",
    )
    parser.add_argument("--bins", type=int, default=48, help="Histogram bin count.")
    return parser.parse_args()


def configure_plot_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "Nimbus Roman", "Liberation Serif", "DejaVu Serif"],
            "font.size": 12,
            "axes.labelsize": 12,
            "xtick.labelsize": 12,
            "ytick.labelsize": 12,
            "legend.fontsize": 12,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def discover_images(image_dir: Path) -> Dict[str, Path]:
    return {
        path.stem: path
        for path in image_dir.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    }


def boundary_distance_percent(coords_norm: np.ndarray, image_size: Tuple[int, int]) -> Tuple[float, float]:
    width, height = image_size
    points = coords_norm * np.asarray([float(width), float(height)], dtype=np.float64)
    distance_px = min(
        float(points[:, 0].min()),
        float(points[:, 1].min()),
        float(width) - float(points[:, 0].max()),
        float(height) - float(points[:, 1].max()),
    )
    distance_px = max(distance_px, 0.0)
    distance_percent = distance_px / max(float(min(width, height)), 1.0) * 100.0
    return distance_px, distance_percent


def parse_obb_line(line: str) -> Optional[Tuple[int, List[str], np.ndarray]]:
    fields = line.strip().split()
    if len(fields) < 9:
        return None
    try:
        cls_id = int(float(fields[0]))
        coords = np.asarray([float(value) for value in fields[1:9]], dtype=np.float64).reshape(4, 2)
    except ValueError:
        return None
    return cls_id, fields, coords


def prepare_output_root(output_root: Path, overwrite: bool) -> None:
    if output_root.exists() and any(output_root.iterdir()) and not overwrite:
        raise FileExistsError(f"Output directory is non-empty; pass --overwrite to reuse it: {output_root}")
    output_root.mkdir(parents=True, exist_ok=True)


def process_split(
    *,
    split: str,
    input_root: Path,
    output_root: Path,
    source_class: int,
    target_class: int,
    hard_threshold_percent: float,
    soft_threshold_percent: Optional[float],
    soft_policy: str,
    copy_images: bool,
) -> Tuple[List[Dict[str, object]], Dict[str, object]]:
    input_images = input_root / "images" / split
    input_labels = input_root / "labels" / split
    output_images = output_root / "images" / split
    output_labels = output_root / "labels" / split
    if not input_images.is_dir() or not input_labels.is_dir():
        raise FileNotFoundError(f"Missing images/labels split under {input_root}: {split}")
    output_labels.mkdir(parents=True, exist_ok=True)
    if copy_images:
        output_images.mkdir(parents=True, exist_ok=True)

    images = discover_images(input_images)
    label_paths = sorted(input_labels.glob("*.txt"))
    rows: List[Dict[str, object]] = []
    before_classes: Counter[int] = Counter()
    after_classes: Counter[int] = Counter()
    missing_images: List[str] = []
    ignored_count = 0
    promoted_hard_count = 0
    promoted_soft_count = 0
    soft_only_count = 0

    for label_path in label_paths:
        image_path = images.get(label_path.stem)
        if image_path is None:
            missing_images.append(label_path.name)
            continue
        with Image.open(image_path) as image:
            image_size = image.size
        output_lines: List[str] = []
        for line_number, line in enumerate(label_path.read_text(encoding="utf-8").splitlines(), start=1):
            if not line.strip():
                continue
            parsed = parse_obb_line(line)
            if parsed is None:
                output_lines.append(line)
                continue
            cls_before, fields, coords = parsed
            distance_px, distance_percent = boundary_distance_percent(coords, image_size)
            hard_zone = distance_percent <= hard_threshold_percent
            soft_zone = (
                soft_threshold_percent is not None
                and distance_percent <= soft_threshold_percent
            )
            soft_only = bool(soft_zone and not hard_zone)
            if soft_only:
                soft_only_count += 1

            cls_after = cls_before
            action = "retain"
            ignored = False
            if cls_before == source_class and hard_zone:
                cls_after = target_class
                action = "promote_hard"
                promoted_hard_count += 1
            elif cls_before == source_class and soft_only:
                if soft_policy == "promote":
                    cls_after = target_class
                    action = "promote_soft"
                    promoted_soft_count += 1
                elif soft_policy == "ignore":
                    action = "ignore_soft"
                    ignored = True
                    ignored_count += 1

            before_classes[cls_before] += 1
            if not ignored:
                fields[0] = str(cls_after)
                output_lines.append(" ".join(fields))
                after_classes[cls_after] += 1

            rows.append(
                {
                    "split": split,
                    "image": image_path.name,
                    "label": label_path.name,
                    "line_number": line_number,
                    "class_before": cls_before,
                    "class_after": "" if ignored else cls_after,
                    "boundary_distance_px": distance_px,
                    "boundary_distance_percent": distance_percent,
                    "hard_zone": hard_zone,
                    "soft_only_zone": soft_only,
                    "action": action,
                }
            )

        (output_labels / label_path.name).write_text(
            "\n".join(output_lines) + ("\n" if output_lines else ""),
            encoding="utf-8",
        )
        if copy_images:
            shutil.copy2(image_path, output_images / image_path.name)

    summary = {
        "split": split,
        "image_count": len(images),
        "label_file_count": len(label_paths),
        "processed_box_count": len(rows),
        "missing_image_labels": missing_images,
        "class_counts_before": {str(key): int(value) for key, value in sorted(before_classes.items())},
        "class_counts_after": {str(key): int(value) for key, value in sorted(after_classes.items())},
        "promoted_hard_count": promoted_hard_count,
        "soft_only_count": soft_only_count,
        "promoted_soft_count": promoted_soft_count,
        "ignored_soft_count": ignored_count,
    }
    return rows, summary


def save_csv(path: Path, rows: Sequence[Dict[str, object]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def save_figure(fig: plt.Figure, path_without_suffix: Path) -> None:
    fig.savefig(path_without_suffix.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(path_without_suffix.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_boundary_histogram(
    rows: Sequence[Dict[str, object]],
    *,
    out_dir: Path,
    source_class: int,
    hard_threshold_percent: float,
    bins: int,
    max_percent: float,
) -> None:
    candidates = [row for row in rows if int(row["class_before"]) == source_class]
    all_values = [float(row["boundary_distance_percent"]) for row in candidates]
    if not all_values:
        return

    full_max = max(max_percent, float(np.ceil(max(all_values) / 5.0) * 5.0))

    def draw_histogram(output_name: str, display_max: float, display_bins: int) -> None:
        fig, ax = plt.subplots(figsize=(7.0, 4.4))
        edges = np.linspace(0.0, display_max, display_bins + 1)
        edges = np.unique(
            np.concatenate([edges, [hard_threshold_percent] if 0.0 < hard_threshold_percent < display_max else []])
        )
        shown_values = [value for value in all_values if 0.0 <= value <= display_max]
        candidate_weight = 100.0 / len(candidates)
        ax.hist(
            shown_values,
            bins=edges,
            weights=np.full(len(shown_values), candidate_weight),
            color="#3572A5",
            edgecolor="black",
            linewidth=0.5,
            hatch="oo",
            label=f"Original Class {source_class} Candidates",
        )
        ax.axvline(hard_threshold_percent, color="black", linestyle="--", linewidth=1.4, label="Class-Conversion Threshold")
        ax.set_xlabel("Normalized Boundary Distance (% of Shorter Image Side)")
        ax.set_ylabel(f"Class-{source_class} Pseudo Instances (%)")
        ax.set_xlim(0.0, display_max)
        ax.legend(frameon=False)
        ax.grid(axis="y", color="#D9D9D9", linewidth=0.6)
        save_figure(fig, out_dir / output_name)

    draw_histogram("boundary_distance_histogram", full_max, max(bins, 50))


def plot_class_counts(
    summaries: Sequence[Dict[str, object]],
    *,
    out_dir: Path,
    source_class: int,
    target_class: int,
) -> None:
    before = Counter()
    after = Counter()
    for summary in summaries:
        before.update({int(key): int(value) for key, value in summary["class_counts_before"].items()})
        after.update({int(key): int(value) for key, value in summary["class_counts_after"].items()})
    x = np.arange(2)
    width = 0.34
    fig, ax = plt.subplots(figsize=(5.8, 4.4))
    ax.bar(
        x - width / 2,
        [before[source_class], before[target_class]],
        width,
        color="#3572A5",
        edgecolor="black",
        linewidth=0.8,
        hatch="oo",
        label="Before",
    )
    ax.bar(
        x + width / 2,
        [after[source_class], after[target_class]],
        width,
        color="#C13B35",
        edgecolor="black",
        linewidth=0.8,
        hatch="//",
        label="After",
    )
    ax.set_xticks(x, [f"Class {source_class} (Complete)", f"Class {target_class} (Incomplete)"])
    ax.set_ylabel("Number of Pseudo Instances")
    ax.legend(frameon=False)
    ax.grid(axis="y", color="#D9D9D9", linewidth=0.6)
    save_figure(fig, out_dir / "class_counts_before_after")


def remove_legacy_report_artifacts(report_dir: Path) -> None:
    for name in (
        "boundary_distance_ecdf.pdf",
        "boundary_distance_ecdf.png",
        "boundary_distance_threshold_zoom.pdf",
        "boundary_distance_threshold_zoom.png",
        "promotion_count_by_threshold.csv",
        "promotion_count_by_threshold.pdf",
        "promotion_count_by_threshold.png",
    ):
        (report_dir / name).unlink(missing_ok=True)


def main() -> None:
    args = parse_args()
    if args.hard_threshold_percent < 0.0:
        raise ValueError("--hard-threshold-percent must be non-negative.")
    if args.soft_threshold_percent is not None and args.soft_threshold_percent < args.hard_threshold_percent:
        raise ValueError("--soft-threshold-percent must be >= --hard-threshold-percent.")

    input_root = args.input_root.expanduser().resolve()
    output_root = args.output_root.expanduser().resolve()
    prepare_output_root(output_root, args.overwrite)
    configure_plot_style()

    all_rows: List[Dict[str, object]] = []
    summaries: List[Dict[str, object]] = []
    for split in args.splits:
        rows, summary = process_split(
            split=split,
            input_root=input_root,
            output_root=output_root,
            source_class=args.source_class,
            target_class=args.target_class,
            hard_threshold_percent=float(args.hard_threshold_percent),
            soft_threshold_percent=args.soft_threshold_percent,
            soft_policy=args.soft_policy,
            copy_images=args.copy_images,
        )
        all_rows.extend(rows)
        summaries.append(summary)

    report_dir = output_root / "boundary_refinement_report"
    report_dir.mkdir(parents=True, exist_ok=True)
    remove_legacy_report_artifacts(report_dir)
    save_csv(report_dir / "per_instance_boundary_distances.csv", all_rows)
    report = {
        "method": "Boundary-Distance Semantic Refinement Teacher",
        "input_root": str(input_root),
        "output_root": str(output_root),
        "source_class": args.source_class,
        "target_class": args.target_class,
        "hard_threshold_percent": float(args.hard_threshold_percent),
        "soft_threshold_percent": args.soft_threshold_percent,
        "soft_policy": args.soft_policy,
        "rule": (
            f"class {args.source_class} -> class {args.target_class} when normalized boundary "
            f"distance <= {float(args.hard_threshold_percent):.6f}%"
        ),
        "splits": summaries,
    }
    (report_dir / "boundary_refinement_report.json").write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )
    plot_boundary_histogram(
        all_rows,
        out_dir=report_dir,
        source_class=args.source_class,
        hard_threshold_percent=float(args.hard_threshold_percent),
        bins=args.bins,
        max_percent=float(args.plot_max_percent),
    )
    plot_class_counts(summaries, out_dir=report_dir, source_class=args.source_class, target_class=args.target_class)

    total_promoted = sum(int(summary["promoted_hard_count"]) + int(summary["promoted_soft_count"]) for summary in summaries)
    total_boxes = sum(int(summary["processed_box_count"]) for summary in summaries)
    print(f"[done] refined labels written to: {output_root}")
    print(f"[done] report and figures written to: {report_dir}")
    print(f"[summary] boxes={total_boxes} class_{args.source_class}_to_{args.target_class}={total_promoted}")
    for summary in summaries:
        print(
            f"[{summary['split']}] boxes={summary['processed_box_count']} "
            f"promoted_hard={summary['promoted_hard_count']} "
            f"soft_only={summary['soft_only_count']} ignored_soft={summary['ignored_soft_count']} "
            f"before={summary['class_counts_before']} after={summary['class_counts_after']}"
        )


if __name__ == "__main__":
    main()
