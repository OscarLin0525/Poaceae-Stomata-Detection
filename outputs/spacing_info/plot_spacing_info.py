#!/usr/bin/env python3
"""Plot normalized-spacing histograms and export Excel-friendly series.

Run from the repo root:

    python outputs/spacing_info/plot_spacing_info.py

This script regenerates histogram PNG/PDF files for every spacing-info
subdirectory. The x-axis is normalized spatial period:

    T_norm = period_norm / median(period_norm)

Thus T_norm = 1 means the species median spacing. This is a linear
normalization, so the histogram distribution keeps the same shape as the
original pixel-spacing histogram. PDF export uses bbox_inches="tight" for
clean vector figures.
"""

from __future__ import annotations

import csv
import statistics
from pathlib import Path
from typing import Iterable


ROOT = Path(__file__).resolve().parent

FONT_CANDIDATES = [
    "Times New Roman",
    "Times",
    "Nimbus Roman",
    "Liberation Serif",
    "DejaVu Serif",
]
FONT_SIZE_LABEL = 20
FONT_SIZE_TICK = 18
FIGSIZE = (8.8, 5.6)
DPI = 160
BINS = 30
BAR_COLOR = "#3572A5"
EXPORT_EXCEL_SERIES = True

PLOTS = [
    {
        "input_csv": "horizontal_spacings.csv",
        "period_column": "spacing_norm_x",
        "fallback_period_column": "spacing_px",
        "output_png": "horizontal_spacing_hist.png",
        "output_pdf": "horizontal_spacing_hist.pdf",
        "extra_output_png": "horizontal_normalized_spacing_hist.png",
        "extra_output_pdf": "horizontal_normalized_spacing_hist.pdf",
        "output_series_csv": "horizontal_normalized_spacing_series_excel.csv",
        "xlabel": "Normalized Spatial Period",
    },
    {
        "input_csv": "vertical_row_gaps.csv",
        "period_column": "row_gap_norm_y",
        "fallback_period_column": "row_gap_px",
        "output_png": "vertical_row_gap_hist.png",
        "output_pdf": "vertical_row_gap_hist.pdf",
        "extra_output_png": "vertical_normalized_spacing_hist.png",
        "extra_output_pdf": "vertical_normalized_spacing_hist.pdf",
        "output_series_csv": "vertical_normalized_spacing_series_excel.csv",
        "xlabel": "Normalized Spatial Period",
    },
]


def iter_spacing_dirs(root: Path) -> Iterable[Path]:
    for path in sorted(root.iterdir()):
        if path.is_dir() and not path.name.startswith("original_font_backup"):
            yield path


def load_values(csv_path: Path, value_column: str, fallback_column: str | None = None) -> list[float]:
    values: list[float] = []
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            try:
                values.append(float(row[value_column]))
            except (KeyError, TypeError, ValueError):
                if fallback_column is None:
                    continue
                try:
                    values.append(float(row[fallback_column]))
                except (KeyError, TypeError, ValueError):
                    continue
    return values


def periods_to_normalized_spacing(periods: list[float]) -> list[float]:
    positive = [v for v in periods if v > 0]
    if not positive:
        return []
    median_period = statistics.median(positive)
    if median_period <= 0:
        return []
    return [v / median_period for v in positive]


def export_series_csv(path: Path, values: list[float]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["數列 1"])
        for value in sorted(values):
            writer.writerow([f"{value:.6f}".rstrip("0").rstrip(".")])


def choose_font(font_manager) -> str:
    available = {font.name for font in font_manager.fontManager.ttflist}
    for candidate in FONT_CANDIDATES:
        if candidate in available:
            return candidate
    return "serif"


def main() -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager

    font_name = choose_font(font_manager)
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": FONT_CANDIDATES,
            "mathtext.fontset": "stix",
        }
    )

    updated_png: list[Path] = []
    updated_pdf: list[Path] = []
    exported_series: list[Path] = []

    for spacing_dir in iter_spacing_dirs(ROOT):
        for spec in PLOTS:
            csv_path = spacing_dir / spec["input_csv"]
            if not csv_path.exists():
                continue

            periods = load_values(csv_path, spec["period_column"], spec.get("fallback_period_column"))
            values = periods_to_normalized_spacing(periods)
            if not values:
                continue

            fig, ax = plt.subplots(figsize=FIGSIZE)
            ax.hist(values, bins=BINS, color=BAR_COLOR, alpha=0.85)
            ax.set_xlabel(spec["xlabel"], fontsize=FONT_SIZE_LABEL, labelpad=10, fontfamily="serif")
            ax.set_ylabel("count", fontsize=FONT_SIZE_LABEL, labelpad=10, fontfamily="serif")
            ax.tick_params(axis="both", labelsize=FONT_SIZE_TICK)
            ax.grid(axis="y", alpha=0.22, linewidth=0.8)

            for label in ax.get_xticklabels() + ax.get_yticklabels():
                label.set_fontfamily("serif")
            for spine in ax.spines.values():
                spine.set_linewidth(1.1)

            fig.tight_layout(pad=1.3)

            png_path = spacing_dir / spec["output_png"]
            pdf_path = spacing_dir / spec["output_pdf"]
            fig.savefig(png_path, dpi=DPI)
            fig.savefig(pdf_path, bbox_inches="tight")
            extra_png = spacing_dir / spec.get("extra_output_png", "")
            extra_pdf = spacing_dir / spec.get("extra_output_pdf", "")
            if extra_png.name:
                fig.savefig(extra_png, dpi=DPI)
                updated_png.append(extra_png)
            if extra_pdf.name:
                fig.savefig(extra_pdf, bbox_inches="tight")
                updated_pdf.append(extra_pdf)
            plt.close(fig)
            updated_png.append(png_path)
            updated_pdf.append(pdf_path)

            if EXPORT_EXCEL_SERIES:
                series_path = spacing_dir / spec["output_series_csv"]
                export_series_csv(series_path, values)
                exported_series.append(series_path)

    print(f"font_used={font_name}")
    print(f"updated_png={len(updated_png)}")
    for path in updated_png:
        print(path)
    print(f"updated_pdf={len(updated_pdf)}")
    for path in updated_pdf:
        print(path)
    if EXPORT_EXCEL_SERIES:
        print(f"exported_series_csv={len(exported_series)}")
        for path in exported_series:
            print(path)


if __name__ == "__main__":
    main()
