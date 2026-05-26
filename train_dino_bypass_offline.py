from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont
from torch.utils.data import Dataset

REPO_ROOT = Path(__file__).resolve().parent
ULTRALYTICS_ROOT = REPO_ROOT / "ultralytics"
for candidate in (REPO_ROOT, ULTRALYTICS_ROOT):
    if candidate.is_dir():
        candidate_str = str(candidate)
        if candidate_str not in sys.path:
            sys.path.insert(0, candidate_str)

try:
    import yaml
except ImportError:  # pragma: no cover - optional dependency
    yaml = None

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


def _configure_report_plot_style() -> None:
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


def _save_completion_box_count_figure(rows: Sequence[Dict[str, object]], output_dir: Path) -> Dict[str, float]:
    if not rows:
        return {}
    before_total = float(sum(float(row.get("before_box_count", 0.0)) for row in rows))
    after_total = float(sum(float(row.get("after_box_count", 0.0)) for row in rows))
    added_total = float(
        sum(
            max(0.0, float(row.get("after_box_count", 0.0)) - float(row.get("matched_box_count", 0.0)))
            for row in rows
        )
    )
    _configure_report_plot_style()
    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    labels = ["Before Pattern", "Added BBoxes", "After Pattern"]
    values = [before_total, added_total, after_total]
    bars = ax.bar(
        labels,
        values,
        color=["#3572A5", "#767676", "#C13B35"],
        edgecolor="black",
        linewidth=0.8,
    )
    for bar, hatch in zip(bars, ("oo", "xx", "//")):
        bar.set_hatch(hatch)
    ax.set_ylabel("Number of Predicted BBoxes")
    ax.grid(axis="y", color="#D9D9D9", linewidth=0.6)
    ax.set_axisbelow(True)
    fig.savefig(output_dir / "bbox_counts_before_after.pdf", bbox_inches="tight")
    fig.savefig(output_dir / "bbox_counts_before_after.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    return {
        "total_before_box_count": before_total,
        "total_added_box_count": added_total,
        "total_after_box_count": after_total,
        "mean_added_box_count": added_total / max(len(rows), 1),
    }


def _load_config_dict(config_path: str) -> Dict[str, object]:
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    suffix = path.suffix.lower()
    if suffix in {".yaml", ".yml"}:
        if yaml is None:
            raise ImportError("PyYAML is required for --config .yaml files. Install with: pip install pyyaml")
        with path.open("r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
    elif suffix == ".json":
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
    else:
        raise ValueError("Config file must be .yaml/.yml or .json")

    if data is None:
        data = {}
    if not isinstance(data, dict):
        raise ValueError("Top-level config must be a mapping/dict")
    if isinstance(data.get("dino_bypass"), dict):
        data = data["dino_bypass"]
    return data


def _serialize_predictions(predictions: Sequence[Dict[str, object]], img_size: Tuple[int, int]) -> List[Dict[str, object]]:
    img_w, img_h = float(img_size[0]), float(img_size[1])
    exported: List[Dict[str, object]] = []
    for pred in predictions:
        if pred.get("poly") is None:
            continue
        poly_abs = np.asarray(pred.get("poly"), dtype=np.float32).reshape(4, 2)
        poly_norm = poly_abs.copy()
        poly_norm[:, 0] = np.clip(poly_norm[:, 0] / max(img_w, 1.0), 0.0, 1.0)
        poly_norm[:, 1] = np.clip(poly_norm[:, 1] / max(img_h, 1.0), 0.0, 1.0)
        exported.append(
            {
                "cls": int(pred.get("cls", 0)),
                "conf": float(pred.get("conf", pred.get("conf_after_rescore", 0.0)) or 0.0),
                "poly": poly_abs.astype(float).tolist(),
                "poly_norm": poly_norm.astype(float).tolist(),
                "synthetic": bool(pred.get("synthetic", False)),
                "conf_before_rescore": float(pred.get("conf_before_rescore", pred.get("conf", 0.0)) or 0.0),
                "conf_after_rescore": float(pred.get("conf_after_rescore", pred.get("conf", 0.0)) or 0.0),
                "pattern_support": float(pred.get("pattern_support", 0.0) or 0.0),
            }
        )
    return exported


def parse_args() -> argparse.Namespace:
    bootstrap = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    bootstrap.add_argument("--config", type=str, default=None, help="Path to YAML/JSON config file")
    boot_args, _ = bootstrap.parse_known_args(sys.argv[1:])

    p = argparse.ArgumentParser(
        description=(
            "DINO-guided detect_test pipeline: build row/frequency prior, "
            "enhance YOLO detect input, refine predictions, and optionally export pseudo labels."
        ),
        allow_abbrev=False,
    )
    p.add_argument("--config", type=str, default=None, help="Path to YAML/JSON config file")
    p.add_argument("--input-dir", type=str, default=None, help="Image directory")
    p.add_argument(
        "--repo-dir",
        type=str,
        default="/home/oscar/Poaceae-Stomata-Detection/dinov3-main",
        help="DINOv3 repo directory",
    )
    p.add_argument("--weights", type=str, default=None, help="Path to DINO checkpoint")
    p.add_argument("--model-name", type=str, default="dinov3_vitb16", help="DINO model name")
    p.add_argument("--output-dir", type=str, default="outputs/detect_test", help="Output directory")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--imgsz", type=int, default=640, help="YOLO inference image size")

    p.add_argument("--yolo-weights", type=str, default=None, help="YOLO weights used in detect_test mode")
    p.add_argument(
        "--feature-level",
        type=str,
        default="p4",
        choices=["p3", "p4", "p5"],
        help="YOLO pyramid level to modify before detect head in detect_test mode",
    )
    p.add_argument("--num-samples", type=int, default=8, help="Number of images to visualize in detect_test mode")
    p.add_argument("--conf", type=float, default=0.1, help="YOLO confidence threshold for detect_test mode")
    p.add_argument(
        "--proposal-conf",
        type=float,
        default=0.01,
        help="Low confidence threshold used to collect rice proposals before pattern filtering",
    )
    p.add_argument("--iou", type=float, default=0.7, help="YOLO NMS IoU for detect_test mode")
    p.add_argument("--max-det", type=int, default=300, help="YOLO max detections for detect_test mode")
    p.add_argument(
        "--completion-preset",
        type=str,
        default="none",
        choices=["none", "rice_aggressive", "rice_conservative_boxes"],
        help="Optional detect_test preset that biases the pipeline toward stronger completion or cleaner pseudo boxes",
    )
    p.add_argument(
        "--pattern-apply-mode",
        type=str,
        default="rescore_only",
        choices=["rescore_only", "feature"],
        help=(
            "How row/frequency prior is applied. "
            "'rescore_only' keeps YOLO features unchanged and uses support only for proposal rescoring/filtering; "
            "'feature' also modulates the detect-input feature map before prediction."
        ),
    )
    p.add_argument(
        "--export-pseudo-dir",
        type=str,
        default=None,
        help="If set, export final filtered predictions as YOLO OBB pseudo labels under images/<split> and labels/<split>",
    )
    p.add_argument(
        "--export-split-preds",
        action="store_true",
        help="Export separate predictions_before.json and predictions_after.json for eval comparisons",
    )
    p.add_argument(
        "--export-split",
        type=str,
        default="train",
        help="Dataset split name used under export-pseudo-dir, e.g. train/val",
    )
    p.add_argument(
        "--pseudo-edge-as-incomplete",
        action="store_true",
        help="When exporting pseudo labels, mark edge-touching predictions as incomplete class",
    )
    p.add_argument(
        "--pseudo-complete-class-id",
        type=int,
        default=0,
        help="Class id used for non-edge pseudo predictions when forcing edge predictions to incomplete",
    )
    p.add_argument(
        "--pseudo-incomplete-class-id",
        type=int,
        default=1,
        help="Class id used for edge-touching pseudo predictions when --pseudo-edge-as-incomplete is enabled",
    )
    p.add_argument(
        "--pseudo-hard-edge-margin-px",
        type=float,
        default=6.0,
        help=(
            "Hard crop margin in pixels for pseudo relabeling. Only boxes touching this "
            "near-boundary band become incomplete by default."
        ),
    )
    p.add_argument(
        "--pseudo-soft-edge-policy",
        type=str,
        default="keep",
        choices=["keep", "ignore", "incomplete"],
        help=(
            "Policy for boxes inside the wider soft edge zone but not touching the hard boundary. "
            "'keep' leaves them as complete, 'ignore' drops them from exported pseudo labels, "
            "'incomplete' restores the old conservative behavior."
        ),
    )
    p.add_argument(
        "--pattern-filter",
        action="store_true",
        help="Filter low-conf proposals using pattern support instead of relying only on conf threshold",
    )
    p.add_argument(
        "--pattern-support-threshold",
        type=float,
        default=0.22,
        help="Minimum pattern-support value at prediction center to keep a proposal",
    )
    p.add_argument(
        "--pattern-anchor-conf",
        type=float,
        default=-1.0,
        help="Confidence threshold used to define row anchors for score propagation; negative uses --conf",
    )
    p.add_argument(
        "--pattern-propagation-strength",
        type=float,
        default=0.95,
        help="How strongly row anchor scores are propagated onto on-frequency proposals",
    )
    p.add_argument(
        "--pattern-offfreq-suppress",
        type=float,
        default=0.55,
        help="How strongly off-frequency proposals are suppressed during rescoring",
    )
    p.add_argument(
        "--pattern-support-power",
        type=float,
        default=1.0,
        help="Exponent applied to support score before propagation; >1 makes completion more conservative",
    )
    p.add_argument(
        "--pattern-row-tolerance-scale",
        type=float,
        default=0.85,
        help="Row grouping tolerance scale relative to reference stomata height in original-image pixels",
    )
    p.add_argument(
        "--pattern-dense-suppress",
        type=float,
        default=0.55,
        help="How strongly to suppress over-dense same-row responses that violate expected spacing",
    )
    p.add_argument(
        "--pattern-min-gap-ratio",
        type=float,
        default=0.55,
        help="Minimum allowed within-row center gap as a ratio of prior period before dense suppression triggers",
    )
    p.add_argument(
        "--dino-pooling-prior-strength",
        type=float,
        default=0.85,
        help="Strength of DINO pooled row/frequency prior when fusing with support map",
    )
    p.add_argument(
        "--dino-pooling-row-threshold",
        type=float,
        default=0.55,
        help="Relative threshold for choosing strong row seeds from pooled DINO energy",
    )
    p.add_argument(
        "--dino-pooling-peak-threshold",
        type=float,
        default=0.50,
        help="Relative threshold for choosing x-direction peaks inside a DINO row band",
    )
    p.add_argument(
        "--dino-pooling-period-ratio",
        type=float,
        default=1.8,
        help="Allowed mismatch ratio when DINO pooling estimates x-period around prior period",
    )
    p.add_argument(
        "--shape-filter",
        action="store_true",
        help="Further filter proposals by image-level stomata size/aspect prior",
    )
    p.add_argument("--shape-width-min-scale", type=float, default=0.45)
    p.add_argument("--shape-width-max-scale", type=float, default=2.20)
    p.add_argument("--shape-height-min-scale", type=float, default=0.45)
    p.add_argument("--shape-height-max-scale", type=float, default=2.20)
    p.add_argument("--shape-area-min-scale", type=float, default=0.25)
    p.add_argument("--shape-area-max-scale", type=float, default=3.20)
    p.add_argument("--shape-aspect-max-scale", type=float, default=2.10)
    p.add_argument("--edge-keep-margin-px", type=float, default=24.0)
    p.add_argument("--edge-keep-margin-ratio", type=float, default=0.05)
    p.add_argument(
        "--pattern-image-priors",
        type=str,
        default=str(
            Path("/home/oscar/Poaceae-Stomata-Detection")
            / "outputs"
            / "rice_info"
            / "rice_annotate_spacing_prior_v2"
            / "image_priors.json"
        ),
        help="Optional exact-image spacing prior JSON for detect_test mode",
    )
    p.add_argument(
        "--pattern-species-prior-bank",
        type=str,
        default=str(
            Path("/home/oscar/Poaceae-Stomata-Detection")
            / "outputs"
            / "spacing_info"
            / "species_spacing_prior_bank.json"
        ),
        help="Optional species-level spacing prior JSON for detect_test mode",
    )
    p.add_argument("--pattern-seed-threshold", type=float, default=0.01)
    p.add_argument("--pattern-seed-topk", type=int, default=96)
    p.add_argument("--pattern-gate-threshold", type=float, default=0.05)
    p.add_argument("--pattern-row-tolerance-px", type=float, default=42.0)
    p.add_argument(
        "--pattern-period-prior-px",
        type=float,
        default=0.0,
        help="Override horizontal rice period in original-image pixels; <=0 uses image/species prior",
    )
    p.add_argument(
        "--pattern-row-period-prior-px",
        type=float,
        default=0.0,
        help="Override vertical row gap in original-image pixels; <=0 uses image/species prior",
    )
    p.add_argument(
        "--pattern-row-tolerance-override-px",
        type=float,
        default=0.0,
        help="Override row grouping tolerance in original-image pixels; <=0 uses image/species/default prior",
    )
    p.add_argument("--pattern-min-row-seeds", type=int, default=2)
    p.add_argument("--pattern-line-max-slope", type=float, default=0.18)
    p.add_argument("--pattern-period-prior-ratio", type=float, default=1.8)
    p.add_argument("--pattern-period-scale", type=float, default=1.0)
    p.add_argument("--pattern-row-sigma-scale", type=float, default=0.45)
    p.add_argument("--pattern-period-sigma-scale", type=float, default=0.28)
    p.add_argument("--pattern-cross-row-strength", type=float, default=0.24)
    p.add_argument("--pattern-response-period-blend", type=float, default=0.60)
    p.add_argument(
        "--pattern-force-horizontal-prior",
        action="store_true",
        help="Disable global orientation estimation and build frequency rows in image-horizontal coordinates",
    )
    p.add_argument(
        "--pattern-full-row-support",
        action="store_true",
        help="Extend each frequency row across the full image width instead of only around anchor span",
    )
    p.add_argument(
        "--template-match-topk",
        type=int,
        default=3,
        help="Number of high-confidence YOLO boxes used as DINO 2D templates",
    )
    p.add_argument(
        "--template-match-anchor-conf",
        type=float,
        default=0.70,
        help="Confidence threshold for template anchors; negative uses --pattern-anchor-conf or --conf",
    )
    p.add_argument(
        "--template-match-min-cells",
        type=int,
        default=2,
        help="Minimum DINO grid crop size for each template side",
    )
    p.add_argument(
        "--template-match-max-cells",
        type=int,
        default=11,
        help="Maximum DINO grid crop size for each template side; larger high-conf boxes are skipped",
    )
    p.add_argument(
        "--template-match-padding-cells",
        type=int,
        default=1,
        help="DINO grid padding around the anchor crop before matching",
    )
    p.add_argument(
        "--template-frequency-strength",
        type=float,
        default=0.35,
        help="How strongly row/frequency prior boosts the DINO template response",
    )
    p.add_argument(
        "--synthetic-completion",
        dest="synthetic_completion",
        action="store_true",
        default=False,
        help="Add synthetic boxes at high-support frequency-grid locations that have no YOLO proposal",
    )
    p.add_argument(
        "--no-synthetic-completion",
        dest="synthetic_completion",
        action="store_false",
        help="Disable synthetic box completion even if enabled by config",
    )
    p.add_argument(
        "--synthetic-support-threshold",
        type=float,
        default=0.72,
        help="Minimum fused support score required for a synthetic no-proposal box",
    )
    p.add_argument(
        "--synthetic-template-threshold",
        type=float,
        default=0.42,
        help="Minimum DINO template-response score required for a synthetic no-proposal box",
    )
    p.add_argument(
        "--synthetic-frequency-threshold",
        type=float,
        default=0.58,
        help="Minimum frequency-prior score required for a synthetic no-proposal box",
    )
    p.add_argument(
        "--synthetic-min-distance-scale",
        type=float,
        default=0.55,
        help="Minimum distance from existing proposal centers as a ratio of the horizontal period prior",
    )
    p.add_argument(
        "--synthetic-min-row-proposals",
        type=int,
        default=2,
        help="Minimum existing YOLO proposals on the same horizontal row before synthetic boxes are allowed",
    )
    p.add_argument(
        "--synthetic-row-tolerance-scale",
        type=float,
        default=1.25,
        help="Same-row tolerance for synthetic completion as a ratio of reference stomata height",
    )
    p.add_argument(
        "--synthetic-row-span-margin-scale",
        type=float,
        default=1.75,
        help="How far beyond the existing row proposal span synthetic boxes may extrapolate, in period units",
    )
    p.add_argument(
        "--synthetic-conf",
        type=float,
        default=0.12,
        help="Base confidence assigned to synthetic completion boxes before final filtering",
    )
    p.add_argument(
        "--synthetic-max-boxes",
        type=int,
        default=80,
        help="Maximum synthetic completion boxes to add per image",
    )
    p.add_argument(
        "--no-template-energy-fallback",
        dest="template_energy_fallback",
        action="store_false",
        default=True,
        help="Disable DINO energy fallback when no usable high-confidence template exists",
    )
    if boot_args.config:
        cfg = _load_config_dict(boot_args.config)
        valid_dests = {a.dest for a in p._actions}
        cleaned = {}
        for k, v in cfg.items():
            if k in valid_dests:
                cleaned[k] = v
            else:
                print(f"[warn] config key ignored (unknown arg): {k}")
        p.set_defaults(**cleaned)

    args = p.parse_args()
    _apply_detect_test_preset(args)

    missing = []
    if not args.input_dir:
        missing.append("--input-dir")
    if not args.weights:
        missing.append("--weights")
    if not args.yolo_weights:
        missing.append("--yolo-weights")
    if missing:
        p.error("the following arguments are required (via CLI or --config): " + ", ".join(missing))

    return args


def _apply_detect_test_preset(args: argparse.Namespace) -> None:
    preset = str(getattr(args, "completion_preset", "none")).strip().lower()
    if preset == "none":
        return
    if preset not in {"rice_aggressive", "rice_conservative_boxes"}:
        raise ValueError(f"Unsupported completion preset: {preset}")

    args.pattern_filter = True
    args.shape_filter = True
    args.pattern_apply_mode = "rescore_only"
    args.dino_pooling_prior_strength = 1.00
    args.dino_pooling_row_threshold = 0.50
    args.dino_pooling_peak_threshold = 0.42

    if preset == "rice_aggressive":
        # Completion-heavy preset tuned to trust the row/frequency prior more.
        args.conf = 0.05
        args.proposal_conf = 0.005
        args.pattern_support_threshold = 0.16
        args.pattern_propagation_strength = 1.15
        args.pattern_offfreq_suppress = 0.28
        args.pattern_support_power = 1.0
        args.pattern_dense_suppress = 0.30
        args.pattern_min_gap_ratio = 0.45
        args.pattern_cross_row_strength = 0.28
        args.template_frequency_strength = 0.45
        return

    # Support remains aggressive, but final pseudo boxes are kept conservative and cleaner.
    args.conf = 0.12
    args.proposal_conf = 0.008
    args.pattern_support_threshold = 0.24
    args.pattern_propagation_strength = 0.92
    args.pattern_offfreq_suppress = 0.60
    args.pattern_support_power = 1.25
    args.pattern_dense_suppress = 0.78
    args.pattern_min_gap_ratio = 0.68
    args.pattern_cross_row_strength = 0.16
    args.template_frequency_strength = 0.30


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)



def normalize_01(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float32)
    mn, mx = float(x.min()), float(x.max())
    if mx <= mn:
        return np.zeros_like(x, dtype=np.float32)
    return (x - mn) / (mx - mn)


class ImageFolderDataset(Dataset):
    def __init__(self, input_dir: Path):
        self.files: List[Path] = sorted([p for p in input_dir.rglob("*") if p.suffix.lower() in IMAGE_EXTS])
        if not self.files:
            raise FileNotFoundError(f"No images found in {input_dir}")

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int):
        path = self.files[idx]
        img = Image.open(path).convert("RGB")
        return img, str(path)


def get_patch_tokens(model, tensor: torch.Tensor) -> torch.Tensor:
    feat = model.forward_features(tensor)
    return feat["x_norm_patchtokens"]

def _load_detect_test_deps() -> Dict[str, Any]:
    from ultralytics import YOLO

    from mtkd_framework.visualize_alignment import _make_tile, _save_metrics_csv
    from mtkd_framework.visualize_dino_effect import (
        _draw_predictions,
        _feature_level_label,
        _feature_response_image,
        _find_detect_module,
        _make_grid,
        _poly_center,
        _predict_one,
        _prediction_delta_metrics,
        _sample_grid_value,
    )
    from mtkd_framework.visualize_yolo_mask_effect import _filter_predictions_by_shape
    from mtkd_framework.visualize_yolo_prediction_panels_core import (
        _load_image_prior_index,
        _load_species_prior_bank,
        _pattern_prior_from_maps,
        _resolve_external_pattern_prior,
        _transform_anchor_points,
    )

    return {
        "YOLO": YOLO,
        "make_tile": _make_tile,
        "save_metrics_csv": _save_metrics_csv,
        "draw_predictions": _draw_predictions,
        "feature_level_label": _feature_level_label,
        "feature_response_image": _feature_response_image,
        "find_detect_module": _find_detect_module,
        "make_grid": _make_grid,
        "poly_center": _poly_center,
        "predict_one": _predict_one,
        "prediction_delta_metrics": _prediction_delta_metrics,
        "sample_grid_value": _sample_grid_value,
        "filter_predictions_by_shape": _filter_predictions_by_shape,
        "load_image_prior_index": _load_image_prior_index,
        "load_species_prior_bank": _load_species_prior_bank,
        "pattern_prior_from_maps": _pattern_prior_from_maps,
        "resolve_external_pattern_prior": _resolve_external_pattern_prior,
        "transform_anchor_points": _transform_anchor_points,
    }


def _dino_imagenet_normalize(images: torch.Tensor) -> torch.Tensor:
    mean = torch.tensor((0.485, 0.456, 0.406), device=images.device, dtype=images.dtype).view(1, 3, 1, 1)
    std = torch.tensor((0.229, 0.224, 0.225), device=images.device, dtype=images.dtype).view(1, 3, 1, 1)
    return (images - mean) / std


def _map_to_feature_response_image(
    map_tensor: Optional[torch.Tensor],
    *,
    size: Tuple[int, int],
    feature_response_image,
) -> Optional[Image.Image]:
    if map_tensor is None:
        return None
    tensor = map_tensor.detach().float().cpu()
    if tensor.ndim == 2:
        tensor = tensor.unsqueeze(0)
    elif tensor.ndim == 3 and tensor.shape[0] != 1:
        tensor = tensor.mean(dim=0, keepdim=True)
    elif tensor.ndim != 3:
        return None
    return feature_response_image(tensor, size)


def _overlay_map_on_image(
    image: Image.Image,
    map_tensor: Optional[torch.Tensor],
    *,
    feature_response_image,
    alpha: float = 0.55,
) -> Optional[Image.Image]:
    heatmap = _map_to_feature_response_image(map_tensor, size=image.size, feature_response_image=feature_response_image)
    if heatmap is None:
        return None
    base = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
    heat = np.asarray(heatmap.convert("RGB"), dtype=np.float32) / 255.0
    tensor = map_tensor.detach().float().cpu()
    if tensor.ndim == 3:
        tensor = tensor.mean(dim=0)
    norm = normalize_01(tensor.numpy().astype(np.float32))
    alpha_map = Image.fromarray((norm * 255.0).astype(np.uint8), mode="L").resize(image.size, resample=Image.Resampling.BICUBIC)
    alpha_arr = (np.asarray(alpha_map, dtype=np.float32) / 255.0)[..., None] * float(np.clip(alpha, 0.0, 1.0))
    out = base * (1.0 - alpha_arr) + heat * alpha_arr
    return Image.fromarray(np.clip(out * 255.0, 0.0, 255.0).astype(np.uint8), mode="RGB")


def _draw_rescore_effect_overlay(
    image: Image.Image,
    predictions: Sequence[Dict[str, object]],
    *,
    final_conf_threshold: float,
    proposal_conf_threshold: float,
    max_boxes: int = 500,
) -> Image.Image:
    canvas = image.copy()
    draw = ImageDraw.Draw(canvas)
    line_width = max(3, int(round(min(canvas.size) / 260.0)))
    kept = [
        pred
        for pred in predictions
        if pred.get("poly") is not None
        and float(pred.get("conf_after_rescore", pred.get("conf", 0.0))) >= float(proposal_conf_threshold)
    ]
    kept = sorted(
        kept,
        key=lambda item: float(item.get("conf_after_rescore", item.get("conf", 0.0))),
        reverse=True,
    )[: max(1, int(max_boxes))]
    for pred in kept:
        poly = np.asarray(pred["poly"], dtype=np.float32).reshape(-1, 2)
        points = [(float(x), float(y)) for x, y in poly.tolist()]
        before = float(pred.get("conf_before_rescore", pred.get("conf", 0.0)))
        after = float(pred.get("conf_after_rescore", pred.get("conf", 0.0)))
        support = float(pred.get("pattern_support", 0.0))
        is_new = bool(pred.get("synthetic", False)) or before < final_conf_threshold <= after
        if is_new:
            color = (255, 35, 35)
            width = line_width + 2
        elif after >= final_conf_threshold:
            color = (0, 220, 80)
            width = line_width + 1
        else:
            color = (150, 150, 150)
            width = line_width
        draw.line(points + [points[0]], fill=color, width=width)
        if support >= 0.55:
            cx = float(poly[:, 0].mean())
            cy = float(poly[:, 1].mean())
            r = max(2.0, float(line_width + 1))
            draw.ellipse((cx - r, cy - r, cx + r, cy + r), fill=color)
    return canvas


def _score_boost_heatmap(
    predictions: Sequence[Dict[str, object]],
    *,
    image_size: Tuple[int, int],
    max_boxes: int = 800,
) -> Optional[torch.Tensor]:
    width, height = int(image_size[0]), int(image_size[1])
    if width <= 0 or height <= 0:
        return None
    heat = np.zeros((height, width), dtype=np.float32)
    ranked = sorted(
        [
            pred
            for pred in predictions
            if pred.get("poly") is not None
            and float(pred.get("conf_after_rescore", pred.get("conf", 0.0)))
            > float(pred.get("conf_before_rescore", pred.get("conf", 0.0))) + 1e-6
        ],
        key=lambda item: float(item.get("conf_after_rescore", item.get("conf", 0.0)))
        - float(item.get("conf_before_rescore", item.get("conf", 0.0))),
        reverse=True,
    )[: max(1, int(max_boxes))]
    for pred in ranked:
        poly = np.asarray(pred["poly"], dtype=np.float32).reshape(-1, 2)
        before = float(pred.get("conf_before_rescore", pred.get("conf", 0.0)))
        after = float(pred.get("conf_after_rescore", pred.get("conf", 0.0)))
        delta = max(0.0, after - before)
        if delta <= 0.0:
            continue
        cx = float(np.clip(poly[:, 0].mean(), 0.0, width - 1.0))
        cy = float(np.clip(poly[:, 1].mean(), 0.0, height - 1.0))
        box_w = max(float(poly[:, 0].max() - poly[:, 0].min()), 6.0)
        box_h = max(float(poly[:, 1].max() - poly[:, 1].min()), 6.0)
        sigma = max(4.0, 0.35 * max(box_w, box_h))
        radius = int(max(6.0, 2.5 * sigma))
        x0 = max(0, int(round(cx)) - radius)
        x1 = min(width, int(round(cx)) + radius + 1)
        y0 = max(0, int(round(cy)) - radius)
        y1 = min(height, int(round(cy)) + radius + 1)
        if x0 >= x1 or y0 >= y1:
            continue
        yy, xx = np.mgrid[y0:y1, x0:x1].astype(np.float32)
        blob = np.exp(-0.5 * (((xx - cx) ** 2 + (yy - cy) ** 2) / max(sigma * sigma, 1e-6)))
        heat[y0:y1, x0:x1] = np.maximum(heat[y0:y1, x0:x1], delta * blob.astype(np.float32))
    if float(heat.max()) <= 1e-8:
        return None
    heat = heat / float(heat.max())
    return torch.from_numpy(heat)


def _prediction_confidence_heatmap(
    predictions: Sequence[Dict[str, object]],
    *,
    image_size: Tuple[int, int],
    max_boxes: int = 800,
) -> torch.Tensor:
    width, height = int(image_size[0]), int(image_size[1])
    heat = np.zeros((max(height, 1), max(width, 1)), dtype=np.float32)
    if width <= 0 or height <= 0:
        return torch.from_numpy(heat)
    ranked = sorted(
        [pred for pred in predictions if pred.get("poly") is not None],
        key=lambda item: float(item.get("conf", item.get("conf_after_rescore", 0.0))),
        reverse=True,
    )[: max(1, int(max_boxes))]
    for pred in ranked:
        poly = np.asarray(pred["poly"], dtype=np.float32).reshape(-1, 2)
        conf = float(pred.get("conf", pred.get("conf_after_rescore", 0.0)) or 0.0)
        if conf <= 0.0:
            continue
        cx = float(np.clip(poly[:, 0].mean(), 0.0, width - 1.0))
        cy = float(np.clip(poly[:, 1].mean(), 0.0, height - 1.0))
        box_w = max(float(poly[:, 0].max() - poly[:, 0].min()), 6.0)
        box_h = max(float(poly[:, 1].max() - poly[:, 1].min()), 6.0)
        sigma = max(4.0, 0.40 * max(box_w, box_h))
        radius = int(max(6.0, 2.5 * sigma))
        x0 = max(0, int(round(cx)) - radius)
        x1 = min(width, int(round(cx)) + radius + 1)
        y0 = max(0, int(round(cy)) - radius)
        y1 = min(height, int(round(cy)) + radius + 1)
        if x0 >= x1 or y0 >= y1:
            continue
        yy, xx = np.mgrid[y0:y1, x0:x1].astype(np.float32)
        blob = np.exp(-0.5 * (((xx - cx) ** 2 + (yy - cy) ** 2) / max(sigma * sigma, 1e-6)))
        heat[y0:y1, x0:x1] = np.maximum(heat[y0:y1, x0:x1], conf * blob.astype(np.float32))
    if float(heat.max()) > 1e-8:
        heat = heat / float(heat.max())
    return torch.from_numpy(heat)


def _save_individual_debug_images(
    *,
    output_dir: Path,
    prefix: str,
    images: Dict[str, Optional[Image.Image]],
    make_tile,
    tile_size: int,
) -> Dict[str, str]:
    indiv_dir = output_dir / f"{prefix}_individual"
    indiv_dir.mkdir(parents=True, exist_ok=True)
    paths: Dict[str, str] = {}
    for name, image in images.items():
        if image is None:
            continue
        raw_path = indiv_dir / f"{name}.png"
        image.save(raw_path)
        paths[f"{name}_path"] = str(raw_path)

        tile_path = indiv_dir / f"{name}_tile.png"
        title = name.replace("_", " ").title()
        _make_debug_tile(image, title, tile_size, caption=False, font_size=10).save(tile_path)
        paths[f"{name}_tile_path"] = str(tile_path)
    return paths


def _make_debug_tile(
    image: Image.Image,
    title: str,
    tile_size: int,
    *,
    caption: bool = False,
    font_size: int = 10,
) -> Image.Image:
    img = image.resize((tile_size, tile_size), resample=Image.Resampling.BICUBIC)
    if not caption:
        return img
    caption_h = max(18, int(round(font_size * 2.2)))
    canvas = Image.new("RGB", (tile_size, tile_size + caption_h), color=(255, 255, 255))
    canvas.paste(img, (0, 0))
    draw = ImageDraw.Draw(canvas)
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", int(font_size))
    except Exception:
        font = ImageFont.load_default()
    draw.text((8, tile_size + max(2, (caption_h - font_size) // 2)), title, fill=(0, 0, 0), font=font)
    return canvas


def _extract_dino_spatial_from_yolo_batch(model, images: torch.Tensor) -> torch.Tensor:
    with torch.no_grad():
        x = images.detach().float()
        if x.max() > 1.5:
            x = x / 255.0
        x = _dino_imagenet_normalize(x)
        tokens = get_patch_tokens(model, x)
        patch_size = int(getattr(model, "patch_size", 16))
        grid_h = x.shape[-2] // patch_size
        grid_w = x.shape[-1] // patch_size
        return tokens.permute(0, 2, 1).contiguous().reshape(x.shape[0], tokens.shape[-1], grid_h, grid_w)


def _effective_detect_test_pattern_settings(
    *,
    args: argparse.Namespace,
    original_size: Tuple[int, int],
    model_input_size: Tuple[int, int],
    image_path: str,
    image_prior_index: Dict[str, Dict[str, object]],
    species_prior_bank: Dict[str, Dict[str, float]],
    transform_anchor_points,
    resolve_external_pattern_prior,
) -> Dict[str, object]:
    external = resolve_external_pattern_prior(
        image_path,
        image_size=original_size,
        image_prior_index=image_prior_index,
        species_prior_bank=species_prior_bank,
    )
    src_w, src_h = float(original_size[0]), float(original_size[1])
    dst_w, dst_h = float(model_input_size[0]), float(model_input_size[1])
    scale = min(dst_w / max(src_w, 1e-6), dst_h / max(src_h, 1e-6))

    external_row_centers: List[float] = []
    raw_row_centers = external.get("row_centers_px", [])
    if isinstance(raw_row_centers, (list, tuple)) and raw_row_centers:
        row_pts = np.asarray([[0.0, float(y), 1.0] for y in raw_row_centers], dtype=np.float32)
        transformed = transform_anchor_points(
            row_pts,
            src_size=original_size,
            dst_size=model_input_size,
            keep_aspect=True,
        )
        external_row_centers = transformed[:, 1].astype(np.float32).tolist()

    raw_period_px = float(external.get("x_period_px", 0.0) or 0.0)
    raw_row_period_px = float(external.get("row_period_px", 0.0) or 0.0)
    raw_row_tolerance_px = float(external.get("row_tolerance_px", 0.0) or 0.0)

    if float(getattr(args, "pattern_period_prior_px", 0.0) or 0.0) > 0.0:
        raw_period_px = float(args.pattern_period_prior_px)
    if float(getattr(args, "pattern_row_period_prior_px", 0.0) or 0.0) > 0.0:
        raw_row_period_px = float(args.pattern_row_period_prior_px)
    if float(getattr(args, "pattern_row_tolerance_override_px", 0.0) or 0.0) > 0.0:
        raw_row_tolerance_px = float(args.pattern_row_tolerance_override_px)

    row_tolerance_px = raw_row_tolerance_px
    if row_tolerance_px > 0.0:
        row_tolerance_px = row_tolerance_px * scale
    else:
        row_tolerance_px = float(args.pattern_row_tolerance_px)

    return {
        "source": str(external.get("source", "none")),
        "family": str(external.get("family", "")),
        "period_prior_px": raw_period_px * scale,
        "row_period_prior_px": raw_row_period_px * scale,
        "row_tolerance_px": row_tolerance_px,
        "row_centers": external_row_centers,
    }


def _predict_one_capture_feature(
    yolo_model: object,
    image_path: str,
    *,
    imgsz: int,
    conf: float,
    iou: float,
    max_det: int,
    device: torch.device,
    feature_level: str,
    predict_one,
    find_detect_module,
) -> Tuple[List[Dict[str, object]], Dict[str, object]]:
    det_model = yolo_model.model
    detect_module = find_detect_module(det_model)
    level_idx = {"p3": 0, "p4": 1, "p5": 2}[feature_level]
    first_module = det_model.model[0]
    state: Dict[str, object] = {"feature": None, "image_size": None}

    def _capture_image_shape(_module, inputs):
        if not inputs:
            return None
        images = inputs[0]
        state["image_size"] = (int(images.shape[-1]), int(images.shape[-2]))
        return None

    def _capture_detect_input(_module, inputs):
        if not inputs:
            return None
        x = inputs[0]
        if not isinstance(x, (list, tuple)) or len(x) <= level_idx:
            return None
        state["feature"] = x[level_idx][0].detach().cpu()
        return None

    handle_input = first_module.register_forward_pre_hook(_capture_image_shape)
    handle_detect = detect_module.register_forward_pre_hook(_capture_detect_input)
    try:
        preds = predict_one(
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
    return preds, state


def _transform_xy_to_model_space(
    points_xy: np.ndarray,
    *,
    src_size: Tuple[int, int],
    dst_size: Tuple[int, int],
    keep_aspect: bool = True,
) -> np.ndarray:
    pts = np.asarray(points_xy, dtype=np.float32).reshape(-1, 2)
    if pts.size == 0:
        return pts.copy()
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
    out[:, 0] = np.clip(out[:, 0], 0.0, max(dst_w - 1.0, 0.0))
    out[:, 1] = np.clip(out[:, 1], 0.0, max(dst_h - 1.0, 0.0))
    return out.astype(np.float32)


def _transform_xy_from_model_space(
    points_xy: np.ndarray,
    *,
    model_input_size: Tuple[int, int],
    original_size: Tuple[int, int],
    keep_aspect: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    pts = np.asarray(points_xy, dtype=np.float32).reshape(-1, 2)
    if pts.size == 0:
        return pts.copy(), np.zeros((0,), dtype=bool)
    model_w, model_h = float(model_input_size[0]), float(model_input_size[1])
    orig_w, orig_h = float(original_size[0]), float(original_size[1])
    out = pts.copy()
    if keep_aspect:
        scale = min(model_w / max(orig_w, 1e-6), model_h / max(orig_h, 1e-6))
        new_w = orig_w * scale
        new_h = orig_h * scale
        pad_x = 0.5 * max(model_w - new_w, 0.0)
        pad_y = 0.5 * max(model_h - new_h, 0.0)
        out[:, 0] = (pts[:, 0] - pad_x) / max(scale, 1e-6)
        out[:, 1] = (pts[:, 1] - pad_y) / max(scale, 1e-6)
    else:
        out[:, 0] = pts[:, 0] * (orig_w / max(model_w, 1e-6))
        out[:, 1] = pts[:, 1] * (orig_h / max(model_h, 1e-6))
    valid = (
        (out[:, 0] >= 0.0)
        & (out[:, 0] <= max(orig_w - 1.0, 0.0))
        & (out[:, 1] >= 0.0)
        & (out[:, 1] <= max(orig_h - 1.0, 0.0))
    )
    out[:, 0] = np.clip(out[:, 0], 0.0, max(orig_w - 1.0, 0.0))
    out[:, 1] = np.clip(out[:, 1], 0.0, max(orig_h - 1.0, 0.0))
    return out.astype(np.float32), valid.astype(bool)


def _select_template_anchor_predictions(
    predictions: Sequence[Dict[str, object]],
    *,
    anchor_conf: float,
) -> List[Dict[str, object]]:
    usable = [
        pred
        for pred in predictions
        if pred.get("poly") is not None and float(pred.get("conf", 0.0)) >= float(anchor_conf)
    ]
    if not usable and predictions:
        usable = [
            pred
            for pred in predictions
            if pred.get("poly") is not None
        ]
        usable = sorted(usable, key=lambda item: float(item.get("conf", 0.0)), reverse=True)[:1]
    return sorted(usable, key=lambda item: float(item.get("conf", 0.0)), reverse=True)


def _normalize_torch_map(values: torch.Tensor, *, low_q: float = 0.02, high_q: float = 0.98) -> torch.Tensor:
    out = values.detach().float()
    out = torch.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)
    if out.numel() <= 1:
        return torch.zeros_like(out)
    flat = out.reshape(-1)
    try:
        lo = torch.quantile(flat, max(0.0, min(1.0, float(low_q))))
        hi = torch.quantile(flat, max(0.0, min(1.0, float(high_q))))
    except RuntimeError:
        lo = flat.min()
        hi = flat.max()
    if float((hi - lo).abs().item()) <= 1e-8:
        lo = flat.min()
        hi = flat.max()
    return ((out - lo) / (hi - lo).clamp_min(1e-6)).clamp(0.0, 1.0)


def _dino_energy_support(dino_feat: torch.Tensor) -> torch.Tensor:
    feat = dino_feat.detach().float()
    if feat.ndim == 4:
        feat = feat.squeeze(0)
    if feat.ndim != 3:
        return torch.zeros((1, 1), dtype=torch.float32)
    return _normalize_torch_map(feat.pow(2).mean(dim=0).sqrt())


def _crop_dino_template(
    feat_norm: torch.Tensor,
    poly_model: np.ndarray,
    *,
    model_input_size: Tuple[int, int],
    min_cells: int,
    max_cells: int,
    padding_cells: int,
) -> Optional[torch.Tensor]:
    if feat_norm.ndim != 3:
        return None
    _channels, h, w = feat_norm.shape
    model_w, model_h = float(model_input_size[0]), float(model_input_size[1])
    poly = np.asarray(poly_model, dtype=np.float32).reshape(-1, 2)
    if poly.size == 0:
        return None

    x0 = float(np.clip(poly[:, 0].min(), 0.0, max(model_w - 1.0, 0.0)))
    y0 = float(np.clip(poly[:, 1].min(), 0.0, max(model_h - 1.0, 0.0)))
    x1 = float(np.clip(poly[:, 0].max(), 0.0, max(model_w - 1.0, 0.0)))
    y1 = float(np.clip(poly[:, 1].max(), 0.0, max(model_h - 1.0, 0.0)))
    if x1 <= x0 or y1 <= y0:
        return None

    fx0 = int(np.floor(x0 / max(model_w, 1e-6) * float(w)))
    fx1 = int(np.ceil(x1 / max(model_w, 1e-6) * float(w)))
    fy0 = int(np.floor(y0 / max(model_h, 1e-6) * float(h)))
    fy1 = int(np.ceil(y1 / max(model_h, 1e-6) * float(h)))
    fx0 = max(0, min(fx0, w - 1))
    fy0 = max(0, min(fy0, h - 1))
    fx1 = max(fx0 + 1, min(fx1, w))
    fy1 = max(fy0 + 1, min(fy1, h))

    raw_w = fx1 - fx0
    raw_h = fy1 - fy0
    max_cells = max(int(max_cells), 1)
    if raw_w > max_cells or raw_h > max_cells:
        return None

    min_cells = max(int(min_cells), 1)
    pad = max(int(padding_cells), 0)
    target_w = min(max(raw_w + 2 * pad, min_cells), max_cells)
    target_h = min(max(raw_h + 2 * pad, min_cells), max_cells)
    cx = 0.5 * (fx0 + fx1)
    cy = 0.5 * (fy0 + fy1)
    tx0 = int(round(cx - 0.5 * target_w))
    ty0 = int(round(cy - 0.5 * target_h))
    tx0 = max(0, min(tx0, max(w - target_w, 0)))
    ty0 = max(0, min(ty0, max(h - target_h, 0)))
    tx1 = min(tx0 + target_w, w)
    ty1 = min(ty0 + target_h, h)
    if tx1 <= tx0 or ty1 <= ty0:
        return None
    template = feat_norm[:, ty0:ty1, tx0:tx1].contiguous()
    if template.shape[-1] < min_cells or template.shape[-2] < min_cells:
        return None
    return template


def _template_response_map(
    dino_feat: torch.Tensor,
    anchor_predictions: Sequence[Dict[str, object]],
    *,
    original_size: Tuple[int, int],
    model_input_size: Tuple[int, int],
    topk: int,
    anchor_conf: float,
    min_cells: int,
    max_cells: int,
    padding_cells: int,
    energy_fallback: bool,
) -> Tuple[torch.Tensor, Dict[str, float], List[Dict[str, object]]]:
    feat = dino_feat.detach().float()
    if feat.ndim == 4:
        feat = feat.squeeze(0)
    if feat.ndim != 3:
        empty = torch.zeros((1, 1), dtype=torch.float32)
        return empty, {"template_anchor_count": 0.0, "template_used_count": 0.0, "template_energy_fallback": 0.0}, []

    selected = _select_template_anchor_predictions(anchor_predictions, anchor_conf=anchor_conf)
    template_candidates = selected[: max(1, int(topk))]
    centered = feat - feat.mean(dim=(1, 2), keepdim=True)
    feat_norm = F.normalize(centered, p=2, dim=0, eps=1e-6)
    responses: List[torch.Tensor] = []
    used = 0
    for pred in template_candidates:
        poly = np.asarray(pred.get("poly"), dtype=np.float32).reshape(4, 2)
        poly_model = _transform_xy_to_model_space(
            poly,
            src_size=original_size,
            dst_size=model_input_size,
            keep_aspect=True,
        )
        template = _crop_dino_template(
            feat_norm,
            poly_model,
            model_input_size=model_input_size,
            min_cells=min_cells,
            max_cells=max_cells,
            padding_cells=padding_cells,
        )
        if template is None:
            continue
        kh, kw = int(template.shape[-2]), int(template.shape[-1])
        pad_y0 = kh // 2
        pad_y1 = kh - 1 - pad_y0
        pad_x0 = kw // 2
        pad_x1 = kw - 1 - pad_x0
        padded = F.pad(
            feat_norm.unsqueeze(0),
            (pad_x0, pad_x1, pad_y0, pad_y1),
            mode="replicate",
        )
        response = F.conv2d(padded, template.unsqueeze(0)).squeeze(0).squeeze(0)
        response = response / max(float(kh * kw), 1.0)
        responses.append(response)
        used += 1

    fallback_used = 0.0
    if responses:
        merged = torch.stack([_normalize_torch_map(resp) for resp in responses], dim=0).max(dim=0).values
    elif energy_fallback:
        merged = _dino_energy_support(feat)
        fallback_used = 1.0
    else:
        merged = torch.zeros(tuple(int(v) for v in feat.shape[-2:]), dtype=torch.float32)

    stats = {
        "template_anchor_count": float(len(selected)),
        "template_used_count": float(used),
        "template_response_mean": float(merged.mean().item()) if merged.numel() else 0.0,
        "template_response_max": float(merged.max().item()) if merged.numel() else 0.0,
        "template_energy_fallback": float(fallback_used),
    }
    return merged.detach().float().cpu().clamp(0.0, 1.0), stats, selected


def _anchor_points_model_space(
    predictions: Sequence[Dict[str, object]],
    *,
    original_size: Tuple[int, int],
    model_input_size: Tuple[int, int],
    limit: int = 128,
) -> np.ndarray:
    points: List[List[float]] = []
    for pred in sorted(predictions, key=lambda item: float(item.get("conf", 0.0)), reverse=True)[: max(1, int(limit))]:
        if pred.get("poly") is None:
            continue
        poly = np.asarray(pred.get("poly"), dtype=np.float32).reshape(4, 2)
        poly_model = _transform_xy_to_model_space(
            poly,
            src_size=original_size,
            dst_size=model_input_size,
            keep_aspect=True,
        )
        center = poly_model.mean(axis=0)
        points.append([float(center[0]), float(center[1]), float(pred.get("conf", 1.0))])
    return np.asarray(points, dtype=np.float32) if points else np.zeros((0, 3), dtype=np.float32)


def _build_template_support_map(
    dino_feat: torch.Tensor,
    anchor_predictions: Sequence[Dict[str, object]],
    *,
    original_size: Tuple[int, int],
    model_input_size: Tuple[int, int],
    topk: int,
    anchor_conf: float,
    min_cells: int,
    max_cells: int,
    padding_cells: int,
    frequency_strength: float,
    energy_fallback: bool,
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
    pattern_cross_row_strength: float,
    pattern_response_period_blend: float,
    pattern_force_horizontal_prior: bool,
    pattern_full_row_support: bool,
    pattern_prior_from_maps,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, float]]:
    response, stats, selected = _template_response_map(
        dino_feat,
        anchor_predictions,
        original_size=original_size,
        model_input_size=model_input_size,
        topk=topk,
        anchor_conf=anchor_conf,
        min_cells=min_cells,
        max_cells=max_cells,
        padding_cells=padding_cells,
        energy_fallback=energy_fallback,
    )
    anchor_points = _anchor_points_model_space(
        selected,
        original_size=original_size,
        model_input_size=model_input_size,
    )

    freq_prior = torch.zeros_like(response)
    freq_stats: Dict[str, float] = {}
    if anchor_points.size > 0 and callable(pattern_prior_from_maps):
        try:
            freq_prior, _seed_points, freq_stats = pattern_prior_from_maps(
                response,
                torch.ones_like(response),
                image_size=model_input_size,
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
                response_period_blend=pattern_response_period_blend,
                force_horizontal=pattern_force_horizontal_prior,
                full_row_support=pattern_full_row_support,
            )
            freq_prior = freq_prior.detach().float().cpu().clamp(0.0, 1.0)
        except Exception as exc:
            stats["template_frequency_failed"] = 1.0
            stats["template_frequency_error_len"] = float(len(str(exc)))

    freq_strength = max(0.0, min(1.0, float(frequency_strength)))
    support = (response + freq_strength * freq_prior * (1.0 - response)).clamp(0.0, 1.0)
    stats.update(
        {
            "template_frequency_strength": float(freq_strength),
            "template_frequency_prior_mean": float(freq_prior.mean().item()) if freq_prior.numel() else 0.0,
            "template_frequency_prior_max": float(freq_prior.max().item()) if freq_prior.numel() else 0.0,
            "template_support_mean": float(support.mean().item()) if support.numel() else 0.0,
            "template_support_max": float(support.max().item()) if support.numel() else 0.0,
        }
    )
    for key, val in freq_stats.items():
        try:
            stats[f"template_freq_{key}"] = float(val)
        except (TypeError, ValueError):
            continue
    return support, freq_prior, response, stats


def _predict_one_pattern_enhanced(
    yolo_model: object,
    dino_model,
    image_path: str,
    *,
    imgsz: int,
    conf: float,
    iou: float,
    max_det: int,
    device: torch.device,
    feature_level: str,
    period_prior_px: float,
    row_period_prior_px: float,
    row_tolerance_px: float,
    row_center_priors_px: Sequence[float],
    pattern_seed_threshold: float,
    pattern_seed_topk: int,
    pattern_gate_threshold: float,
    pattern_min_row_seeds: int,
    pattern_line_max_slope: float,
    pattern_period_prior_ratio: float,
    pattern_period_scale: float,
    pattern_row_sigma_scale: float,
    pattern_period_sigma_scale: float,
    pattern_cross_row_strength: float,
    pattern_response_period_blend: float,
    pattern_force_horizontal_prior: bool,
    pattern_full_row_support: bool,
    pattern_apply_mode: str,
    original_size: Tuple[int, int],
    template_predictions: Sequence[Dict[str, object]],
    template_match_topk: int,
    template_match_anchor_conf: float,
    template_match_min_cells: int,
    template_match_max_cells: int,
    template_match_padding_cells: int,
    template_frequency_strength: float,
    template_energy_fallback: bool,
    predict_one,
    find_detect_module,
    pattern_prior_from_maps,
) -> Tuple[List[Dict[str, object]], Dict[str, object]]:
    det_model = yolo_model.model
    detect_module = find_detect_module(det_model)
    level_idx = {"p3": 0, "p4": 1, "p5": 2}[feature_level]
    first_module = det_model.model[0]
    state: Dict[str, object] = {
        "dino": None,
        "dino_feature": None,
        "feature": None,
        "pattern_stats": {},
        "image_size": None,
        "prior_map": None,
        "support_map": None,
        "guide_map": None,
    }
    apply_feature = str(pattern_apply_mode).strip().lower() == "feature"

    def _cache_dino(_module, inputs):
        if not inputs:
            state["dino"] = None
            return None
        images = inputs[0].detach().float().to(device)
        state["image_size"] = (int(images.shape[-1]), int(images.shape[-2]))
        state["dino"] = _extract_dino_spatial_from_yolo_batch(dino_model, images)
        return None

    def _enhance_detect_input(_module, inputs):
        dino_batch = state.get("dino")
        if not inputs or dino_batch is None:
            return None
        x = inputs[0]
        if not isinstance(x, (list, tuple)) or len(x) <= level_idx:
            return None
        feats = list(x)
        target_feat = feats[level_idx]
        enhanced_batch: List[torch.Tensor] = []
        for batch_idx in range(target_feat.shape[0]):
            model_input_size = (int(state["image_size"][0]), int(state["image_size"][1]))
            _support_map, _prior_map, _guide_map, pattern_stats = _build_template_support_map(
                dino_batch[batch_idx],
                template_predictions,
                original_size=original_size,
                model_input_size=model_input_size,
                topk=template_match_topk,
                anchor_conf=template_match_anchor_conf,
                min_cells=template_match_min_cells,
                max_cells=template_match_max_cells,
                padding_cells=template_match_padding_cells,
                frequency_strength=template_frequency_strength,
                energy_fallback=template_energy_fallback,
                pattern_seed_threshold=pattern_seed_threshold,
                pattern_seed_topk=pattern_seed_topk,
                pattern_gate_threshold=pattern_gate_threshold,
                pattern_row_tolerance_px=row_tolerance_px,
                pattern_min_row_seeds=pattern_min_row_seeds,
                pattern_line_max_slope=pattern_line_max_slope,
                pattern_period_prior_px=period_prior_px,
                pattern_period_prior_ratio=pattern_period_prior_ratio,
                pattern_period_scale=pattern_period_scale,
                pattern_row_sigma_scale=pattern_row_sigma_scale,
                pattern_period_sigma_scale=pattern_period_sigma_scale,
                pattern_row_period_prior_px=row_period_prior_px,
                pattern_row_center_priors_px=row_center_priors_px,
                pattern_cross_row_strength=pattern_cross_row_strength,
                pattern_response_period_blend=pattern_response_period_blend,
                pattern_force_horizontal_prior=pattern_force_horizontal_prior,
                pattern_full_row_support=pattern_full_row_support,
                pattern_prior_from_maps=pattern_prior_from_maps,
            )
            support_for_feature = _support_map.to(device=target_feat.device, dtype=target_feat.dtype)
            if tuple(support_for_feature.shape[-2:]) != tuple(target_feat.shape[-2:]):
                support_for_feature = F.interpolate(
                    support_for_feature.unsqueeze(0).unsqueeze(0),
                    size=target_feat.shape[-2:],
                    mode="bilinear",
                    align_corners=False,
                ).squeeze(0).squeeze(0)
            scale = 0.5 + (1.8 - 0.5) * support_for_feature.clamp(0.0, 1.0)
            if apply_feature:
                enhanced_feat = target_feat[batch_idx] * scale.unsqueeze(0)
            else:
                enhanced_feat = target_feat[batch_idx].clone()
            enhanced_batch.append(enhanced_feat)
            if batch_idx == 0:
                state["dino_feature"] = dino_batch[batch_idx].detach().cpu()
                state["feature"] = enhanced_feat.detach().cpu()
                state["pattern_stats"] = dict(pattern_stats)
                state["pattern_stats"]["pattern_apply_mode_feature"] = 1.0 if apply_feature else 0.0
                state["pattern_stats"]["template_match_enabled"] = 1.0
                state["prior_map"] = _prior_map.detach().cpu()
                state["support_map"] = _support_map.detach().cpu()
                state["guide_map"] = _guide_map.detach().cpu() if isinstance(_guide_map, torch.Tensor) else None
        feats[level_idx] = torch.stack(enhanced_batch, dim=0)
        return (feats,) + tuple(inputs[1:])

    handle_input = first_module.register_forward_pre_hook(_cache_dino)
    handle_detect = detect_module.register_forward_pre_hook(_enhance_detect_input)
    try:
        preds = predict_one(
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
    return preds, state


def _prediction_edge_flags(
    pred: Dict[str, object],
    *,
    image_size: Tuple[int, int],
    edge_margin_px: float,
    edge_margin_ratio: float,
    hard_edge_margin_px: float,
) -> Dict[str, bool]:
    poly = np.asarray(pred.get("poly"), dtype=np.float32).reshape(-1, 2)
    if poly.shape != (4, 2):
        return {
            "touch_left": False,
            "touch_top": False,
            "touch_right": False,
            "touch_bottom": False,
            "hard_touch_left": False,
            "hard_touch_top": False,
            "hard_touch_right": False,
            "hard_touch_bottom": False,
            "is_soft_edge": False,
            "is_hard_edge": False,
            "is_edge": False,
        }
    x1 = float(poly[:, 0].min())
    y1 = float(poly[:, 1].min())
    x2 = float(poly[:, 0].max())
    y2 = float(poly[:, 1].max())
    soft_border = max(float(edge_margin_px), float(min(image_size)) * float(edge_margin_ratio))
    hard_border = max(0.0, float(hard_edge_margin_px))
    touch_left = bool(x1 <= soft_border)
    touch_top = bool(y1 <= soft_border)
    touch_right = bool(x2 >= float(image_size[0]) - soft_border)
    touch_bottom = bool(y2 >= float(image_size[1]) - soft_border)
    hard_touch_left = bool(x1 <= hard_border)
    hard_touch_top = bool(y1 <= hard_border)
    hard_touch_right = bool(x2 >= float(image_size[0]) - hard_border)
    hard_touch_bottom = bool(y2 >= float(image_size[1]) - hard_border)
    is_soft_edge = bool(touch_left or touch_top or touch_right or touch_bottom)
    is_hard_edge = bool(hard_touch_left or hard_touch_top or hard_touch_right or hard_touch_bottom)
    return {
        "touch_left": touch_left,
        "touch_top": touch_top,
        "touch_right": touch_right,
        "touch_bottom": touch_bottom,
        "hard_touch_left": hard_touch_left,
        "hard_touch_top": hard_touch_top,
        "hard_touch_right": hard_touch_right,
        "hard_touch_bottom": hard_touch_bottom,
        "is_soft_edge": is_soft_edge,
        "is_hard_edge": is_hard_edge,
        # Backward-compatible name: the refined-label relabel decision now uses
        # the stricter hard edge, while soft edge is tracked separately.
        "is_edge": is_hard_edge,
    }


def _prediction_to_yolo_obb_line(
    pred: Dict[str, object],
    *,
    image_size: Tuple[int, int],
    edge_as_incomplete: bool,
    complete_class_id: int,
    incomplete_class_id: int,
    edge_margin_px: float,
    edge_margin_ratio: float,
    hard_edge_margin_px: float,
    soft_edge_policy: str,
) -> Optional[str]:
    poly = np.asarray(pred.get("poly"), dtype=np.float32).reshape(-1, 2)
    if poly.shape != (4, 2):
        return None
    width = max(float(image_size[0]), 1.0)
    height = max(float(image_size[1]), 1.0)
    poly_norm = poly.copy()
    poly_norm[:, 0] = np.clip(poly_norm[:, 0] / width, 0.0, 1.0)
    poly_norm[:, 1] = np.clip(poly_norm[:, 1] / height, 0.0, 1.0)
    if not np.isfinite(poly_norm).all():
        return None
    cls_id = int(pred.get("cls", complete_class_id))
    edge_flags = _prediction_edge_flags(
        pred,
        image_size=image_size,
        edge_margin_px=edge_margin_px,
        edge_margin_ratio=edge_margin_ratio,
        hard_edge_margin_px=hard_edge_margin_px,
    )
    if edge_as_incomplete:
        soft_policy = str(soft_edge_policy).strip().lower()
        if edge_flags["is_hard_edge"]:
            cls_id = int(incomplete_class_id)
        elif edge_flags["is_soft_edge"] and soft_policy == "incomplete":
            cls_id = int(incomplete_class_id)
        elif edge_flags["is_soft_edge"] and soft_policy == "ignore":
            pred["pseudo_export_cls"] = int(cls_id)
            pred["pseudo_export_edge"] = bool(edge_flags["is_hard_edge"])
            pred["pseudo_export_soft_edge"] = bool(edge_flags["is_soft_edge"])
            pred["pseudo_export_hard_edge"] = bool(edge_flags["is_hard_edge"])
            pred["pseudo_export_ignored"] = True
            return None
        else:
            cls_id = int(complete_class_id)
    pred["pseudo_export_cls"] = int(cls_id)
    pred["pseudo_export_edge"] = bool(edge_flags["is_hard_edge"])
    pred["pseudo_export_soft_edge"] = bool(edge_flags["is_soft_edge"])
    pred["pseudo_export_hard_edge"] = bool(edge_flags["is_hard_edge"])
    pred["pseudo_export_ignored"] = False
    coords = " ".join(f"{float(v):.6f}" for v in poly_norm.reshape(-1))
    return f"{cls_id} {coords}"


def _export_predictions_as_pseudo_labels(
    image_path: Path,
    predictions: Sequence[Dict[str, object]],
    *,
    image_size: Tuple[int, int],
    export_root: Path,
    split: str,
    edge_as_incomplete: bool,
    complete_class_id: int,
    incomplete_class_id: int,
    edge_margin_px: float,
    edge_margin_ratio: float,
    hard_edge_margin_px: float,
    soft_edge_policy: str,
) -> Dict[str, object]:
    labels_dir = export_root / "labels" / split
    images_dir = export_root / "images" / split
    labels_dir.mkdir(parents=True, exist_ok=True)
    images_dir.mkdir(parents=True, exist_ok=True)

    lines: List[str] = []
    complete_count = 0
    incomplete_count = 0
    soft_edge_count = 0
    hard_edge_count = 0
    ignored_count = 0
    for pred in predictions:
        line = _prediction_to_yolo_obb_line(
            pred,
            image_size=image_size,
            edge_as_incomplete=edge_as_incomplete,
            complete_class_id=complete_class_id,
            incomplete_class_id=incomplete_class_id,
            edge_margin_px=edge_margin_px,
            edge_margin_ratio=edge_margin_ratio,
            hard_edge_margin_px=hard_edge_margin_px,
            soft_edge_policy=soft_edge_policy,
        )
        if bool(pred.get("pseudo_export_soft_edge", False)):
            soft_edge_count += 1
        if bool(pred.get("pseudo_export_hard_edge", False)):
            hard_edge_count += 1
        if bool(pred.get("pseudo_export_ignored", False)):
            ignored_count += 1
        if line is not None:
            lines.append(line)
            export_cls = int(pred.get("pseudo_export_cls", complete_class_id))
            if export_cls == int(incomplete_class_id):
                incomplete_count += 1
            elif export_cls == int(complete_class_id):
                complete_count += 1

    label_path = labels_dir / f"{image_path.stem}.txt"
    label_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")

    target_image_path = images_dir / image_path.name
    if target_image_path.resolve() != image_path.resolve():
        shutil.copy2(image_path, target_image_path)

    return {
        "export_label_path": str(label_path),
        "export_image_path": str(target_image_path),
        "export_box_count": float(len(lines)),
        "export_complete_count": float(complete_count),
        "export_incomplete_count": float(incomplete_count),
        "export_soft_edge_count": float(soft_edge_count),
        "export_hard_edge_count": float(hard_edge_count),
        "export_soft_edge_ignored_count": float(ignored_count),
    }


def _prediction_box_height_px(pred: Dict[str, object]) -> float:
    poly = np.asarray(pred["poly"], dtype=np.float32).reshape(4, 2)
    edges = []
    for idx in range(4):
        nxt = (idx + 1) % 4
        edges.append(float(np.linalg.norm(poly[nxt] - poly[idx])))
    side_a = 0.5 * (edges[0] + edges[2])
    side_b = 0.5 * (edges[1] + edges[3])
    return float(max(min(side_a, side_b), 1e-6))


def _prediction_box_size_px(pred: Dict[str, object]) -> Tuple[float, float]:
    poly = np.asarray(pred["poly"], dtype=np.float32).reshape(4, 2)
    edges = []
    for idx in range(4):
        nxt = (idx + 1) % 4
        edges.append(float(np.linalg.norm(poly[nxt] - poly[idx])))
    side_a = 0.5 * (edges[0] + edges[2])
    side_b = 0.5 * (edges[1] + edges[3])
    return float(max(side_a, side_b, 1e-6)), float(max(min(side_a, side_b), 1e-6))


def _fallback_candidate_poly(
    center_xy: np.ndarray,
    image_size: Tuple[int, int],
    *,
    width_px: float,
    height_px: float,
) -> np.ndarray:
    img_w, img_h = float(image_size[0]), float(image_size[1])
    cx, cy = float(center_xy[0]), float(center_xy[1])
    half_w = 0.5 * max(float(width_px), max(8.0, img_w * 0.04))
    half_h = 0.5 * max(float(height_px), max(6.0, img_h * 0.03))
    poly = np.asarray(
        [
            [cx - half_w, cy - half_h],
            [cx + half_w, cy - half_h],
            [cx + half_w, cy + half_h],
            [cx - half_w, cy + half_h],
        ],
        dtype=np.float32,
    )
    poly[:, 0] = np.clip(poly[:, 0], 0.0, max(img_w - 1.0, 0.0))
    poly[:, 1] = np.clip(poly[:, 1], 0.0, max(img_h - 1.0, 0.0))
    return poly


def _complete_synthetic_predictions(
    predictions: Sequence[Dict[str, object]],
    reference_predictions: Sequence[Dict[str, object]],
    support_map: Optional[torch.Tensor],
    frequency_prior_map: Optional[torch.Tensor],
    template_response_map: Optional[torch.Tensor],
    *,
    image_size: Tuple[int, int],
    model_input_size: Tuple[int, int],
    final_conf_threshold: float,
    period_prior_px: float,
    support_threshold: float,
    template_threshold: float,
    frequency_threshold: float,
    min_distance_scale: float,
    min_row_proposals: int,
    row_tolerance_scale: float,
    row_span_margin_scale: float,
    synthetic_conf: float,
    max_boxes: int,
    poly_center,
    sample_grid_value,
) -> Tuple[List[Dict[str, object]], Dict[str, float]]:
    stats = {
        "synthetic_completion_enabled": 1.0,
        "synthetic_candidate_count": 0.0,
        "synthetic_added_count": 0.0,
        "synthetic_mean_support": 0.0,
        "synthetic_mean_template": 0.0,
        "synthetic_mean_frequency": 0.0,
        "synthetic_min_distance_px": 0.0,
        "synthetic_final_count": 0.0,
        "synthetic_row_rejected_count": 0.0,
        "synthetic_span_rejected_count": 0.0,
        "synthetic_row_tolerance_px": 0.0,
    }
    if support_map is None or frequency_prior_map is None or template_response_map is None:
        return [], stats
    if max_boxes <= 0:
        return [], stats

    support = support_map.detach().float().cpu()
    freq = frequency_prior_map.detach().float().cpu()
    template = template_response_map.detach().float().cpu()
    if support.ndim == 3:
        support = support.mean(dim=0)
    if freq.ndim == 3:
        freq = freq.mean(dim=0)
    if template.ndim == 3:
        template = template.mean(dim=0)
    if support.ndim != 2 or freq.ndim != 2 or template.ndim != 2:
        return [], stats
    if tuple(freq.shape) != tuple(support.shape):
        freq = F.interpolate(
            freq.unsqueeze(0).unsqueeze(0),
            size=support.shape,
            mode="bilinear",
            align_corners=False,
        ).squeeze(0).squeeze(0)
    if tuple(template.shape) != tuple(support.shape):
        template = F.interpolate(
            template.unsqueeze(0).unsqueeze(0),
            size=support.shape,
            mode="bilinear",
            align_corners=False,
        ).squeeze(0).squeeze(0)

    support = support.clamp(0.0, 1.0)
    freq = freq.clamp(0.0, 1.0)
    template = template.clamp(0.0, 1.0)
    evidence = (0.50 * support + 0.30 * template + 0.20 * freq).clamp(0.0, 1.0)
    local_max = F.max_pool2d(evidence.unsqueeze(0).unsqueeze(0), kernel_size=3, stride=1, padding=1).squeeze(0).squeeze(0)
    mask = (
        (evidence >= local_max - 1e-6)
        & (support >= float(support_threshold))
        & (template >= float(template_threshold))
        & (freq >= float(frequency_threshold))
    )
    peak_indices = torch.nonzero(mask, as_tuple=False)
    if peak_indices.numel() == 0:
        return [], stats

    scores = evidence[peak_indices[:, 0], peak_indices[:, 1]]
    order = torch.argsort(scores, descending=True)
    # Keep enough candidates to survive de-duplication without letting noisy maps dominate runtime.
    candidate_limit = max(int(max_boxes) * 8, int(max_boxes))
    peak_indices = peak_indices[order[:candidate_limit]]
    scores = scores[order[:candidate_limit]]
    stats["synthetic_candidate_count"] = float(int(peak_indices.shape[0]))

    h, w = int(evidence.shape[0]), int(evidence.shape[1])
    model_w, model_h = float(model_input_size[0]), float(model_input_size[1])
    model_points = np.stack(
        [
            (peak_indices[:, 1].numpy().astype(np.float32) + 0.5) / max(float(w), 1.0) * model_w,
            (peak_indices[:, 0].numpy().astype(np.float32) + 0.5) / max(float(h), 1.0) * model_h,
        ],
        axis=1,
    )
    original_points, valid_points = _transform_xy_from_model_space(
        model_points,
        model_input_size=model_input_size,
        original_size=image_size,
        keep_aspect=True,
    )

    existing_centers = [
        poly_center(np.asarray(pred["poly"], dtype=np.float32)).astype(np.float32)
        for pred in predictions
        if pred.get("poly") is not None
    ]
    accepted_centers: List[np.ndarray] = []

    ref_preds = [pred for pred in reference_predictions if pred.get("poly") is not None]
    if not ref_preds:
        ref_preds = [pred for pred in predictions if pred.get("poly") is not None]
    ref_polys = [np.asarray(pred["poly"], dtype=np.float32).reshape(4, 2) for pred in ref_preds]
    ref_centers = (
        np.asarray([poly_center(poly).astype(np.float32) for poly in ref_polys], dtype=np.float32)
        if ref_polys
        else np.zeros((0, 2), dtype=np.float32)
    )
    ref_widths: List[float] = []
    ref_heights: List[float] = []
    for pred in ref_preds:
        try:
            width_px, height_px = _prediction_box_size_px(pred)
        except Exception:
            continue
        ref_widths.append(width_px)
        ref_heights.append(height_px)
    fallback_width = float(np.median(ref_widths)) if ref_widths else max(8.0, float(image_size[0]) * 0.04)
    fallback_height = float(np.median(ref_heights)) if ref_heights else max(6.0, float(image_size[1]) * 0.03)

    src_w, src_h = float(image_size[0]), float(image_size[1])
    dst_w, dst_h = float(model_input_size[0]), float(model_input_size[1])
    model_to_orig_scale = 1.0 / max(min(dst_w / max(src_w, 1e-6), dst_h / max(src_h, 1e-6)), 1e-6)
    prior_gap_px = float(period_prior_px) * model_to_orig_scale if period_prior_px > 0.0 else fallback_width
    min_distance = max(4.0, float(min_distance_scale) * max(prior_gap_px, fallback_width))
    stats["synthetic_min_distance_px"] = float(min_distance)
    row_tolerance = max(6.0, float(row_tolerance_scale) * max(fallback_height, 1.0))
    row_span_margin = max(0.0, float(row_span_margin_scale) * max(prior_gap_px, fallback_width))
    stats["synthetic_row_tolerance_px"] = float(row_tolerance)

    synthetic: List[Dict[str, object]] = []
    support_scores: List[float] = []
    template_scores: List[float] = []
    freq_scores: List[float] = []
    base_conf = max(float(final_conf_threshold) + 0.01, float(synthetic_conf))
    row_rejected = 0
    span_rejected = 0
    existing_center_arr = np.asarray(existing_centers, dtype=np.float32) if existing_centers else np.zeros((0, 2), dtype=np.float32)
    for rank, (point, valid, peak_idx, score_tensor) in enumerate(
        zip(original_points, valid_points, peak_indices, scores),
        start=1,
    ):
        if not bool(valid):
            continue
        center = np.asarray(point, dtype=np.float32).reshape(2)
        if existing_centers:
            d_existing = np.linalg.norm(np.asarray(existing_centers, dtype=np.float32) - center[None, :], axis=1)
            if float(d_existing.min()) < min_distance:
                continue
        if accepted_centers:
            d_new = np.linalg.norm(np.asarray(accepted_centers, dtype=np.float32) - center[None, :], axis=1)
            if float(d_new.min()) < min_distance:
                continue

        if int(min_row_proposals) > 0:
            if not existing_center_arr.size:
                row_rejected += 1
                continue
            same_row_mask = np.abs(existing_center_arr[:, 1] - float(center[1])) <= row_tolerance
            same_row = existing_center_arr[same_row_mask]
            if same_row.shape[0] < int(min_row_proposals):
                row_rejected += 1
                continue
            x_min = float(same_row[:, 0].min())
            x_max = float(same_row[:, 0].max())
            if float(center[0]) < x_min - row_span_margin or float(center[0]) > x_max + row_span_margin:
                span_rejected += 1
                continue

        gy = int(peak_idx[0].item())
        gx = int(peak_idx[1].item())
        support_score = float(support[gy, gx].item())
        freq_score = float(freq[gy, gx].item())
        template_score = float(template[gy, gx].item())
        if support_score < float(support_threshold) or template_score < float(template_threshold) or freq_score < float(frequency_threshold):
            continue

        cls_id = 0
        name = "synthetic"
        if ref_polys and ref_centers.size:
            nearest = int(np.argmin(np.linalg.norm(ref_centers - center[None, :], axis=1)))
            template_poly = ref_polys[nearest]
            poly = template_poly + (center - poly_center(template_poly))[None, :]
            cls_id = int(ref_preds[nearest].get("cls", 0))
            name = str(ref_preds[nearest].get("name", name))
        else:
            poly = _fallback_candidate_poly(
                center,
                image_size,
                width_px=fallback_width,
                height_px=fallback_height,
            )
        poly = np.asarray(poly, dtype=np.float32).reshape(4, 2)
        poly[:, 0] = np.clip(poly[:, 0], 0.0, max(float(image_size[0]) - 1.0, 0.0))
        poly[:, 1] = np.clip(poly[:, 1], 0.0, max(float(image_size[1]) - 1.0, 0.0))
        evidence_score = float(score_tensor.item())
        conf = float(np.clip(base_conf * (0.75 + 0.25 * evidence_score), 0.0, 1.0))
        synthetic.append(
            {
                "poly": poly.astype(np.float32),
                "conf": conf,
                "cls": cls_id,
                "name": name,
                "synthetic": True,
                "synthetic_rank": float(rank),
                "conf_before_rescore": 0.0,
                "conf_after_rescore": conf,
                "pattern_support": support_score,
                "pattern_support_local": support_score,
                "pattern_support_frequency": freq_score,
                "pattern_support_template": template_score,
            }
        )
        accepted_centers.append(center)
        support_scores.append(support_score)
        template_scores.append(template_score)
        freq_scores.append(freq_score)
        if len(synthetic) >= int(max_boxes):
            break

    stats.update(
        {
            "synthetic_added_count": float(len(synthetic)),
            "synthetic_mean_support": float(np.mean(support_scores)) if support_scores else 0.0,
            "synthetic_mean_template": float(np.mean(template_scores)) if template_scores else 0.0,
            "synthetic_mean_frequency": float(np.mean(freq_scores)) if freq_scores else 0.0,
            "synthetic_row_rejected_count": float(row_rejected),
            "synthetic_span_rejected_count": float(span_rejected),
        }
    )
    return synthetic, stats


def _draw_synthetic_completion_overlay(
    image: Image.Image,
    predictions: Sequence[Dict[str, object]],
    *,
    max_boxes: int = 500,
) -> Image.Image:
    canvas = image.copy()
    draw = ImageDraw.Draw(canvas)
    line_width = max(2, int(round(min(canvas.size) / 520.0)))
    non_synthetic = [
        pred for pred in predictions if pred.get("poly") is not None and not bool(pred.get("synthetic", False))
    ][: max(1, int(max_boxes))]
    synthetic = [
        pred for pred in predictions if pred.get("poly") is not None and bool(pred.get("synthetic", False))
    ][: max(1, int(max_boxes))]
    for pred in non_synthetic:
        poly = np.asarray(pred["poly"], dtype=np.float32).reshape(-1, 2)
        points = [(float(x), float(y)) for x, y in poly.tolist()]
        draw.line(points + [points[0]], fill=(170, 170, 170), width=max(1, line_width - 1))
    for pred in synthetic:
        poly = np.asarray(pred["poly"], dtype=np.float32).reshape(-1, 2)
        points = [(float(x), float(y)) for x, y in poly.tolist()]
        draw.line(points + [points[0]], fill=(0, 255, 80), width=line_width + 1)
        cx = float(poly[:, 0].mean())
        cy = float(poly[:, 1].mean())
        r = max(3.0, float(line_width + 2))
        draw.ellipse((cx - r, cy - r, cx + r, cy + r), fill=(0, 255, 80))
    return canvas


def _group_prediction_rows(
    predictions: Sequence[Dict[str, object]],
    *,
    row_tolerance_px: float,
    poly_center,
) -> List[List[int]]:
    if not predictions:
        return []
    records = []
    for idx, pred in enumerate(predictions):
        center = poly_center(np.asarray(pred["poly"], dtype=np.float32))
        records.append((idx, float(center[0]), float(center[1])))
    records.sort(key=lambda item: item[2])
    rows: List[List[int]] = []
    current: List[int] = [records[0][0]]
    current_y = float(records[0][2])
    tol = max(float(row_tolerance_px), 1.0)
    for idx, _x, y in records[1:]:
        if abs(float(y) - current_y) <= tol:
            current.append(idx)
            current_y = float(np.mean([float(poly_center(np.asarray(predictions[j]["poly"], dtype=np.float32))[1]) for j in current]))
        else:
            rows.append(current)
            current = [idx]
            current_y = float(y)
    rows.append(current)
    return rows


def _fit_row_line(
    row_indices: Sequence[int],
    predictions: Sequence[Dict[str, object]],
    *,
    poly_center,
    max_slope: float,
) -> Dict[str, object]:
    centers = [
        poly_center(np.asarray(predictions[idx]["poly"], dtype=np.float32)).astype(np.float32)
        for idx in row_indices
    ]
    pts = np.asarray(centers, dtype=np.float32).reshape(-1, 2)
    if pts.size == 0:
        origin = np.zeros((2,), dtype=np.float32)
        direction = np.asarray([1.0, 0.0], dtype=np.float32)
        normal = np.asarray([0.0, 1.0], dtype=np.float32)
        return {
            "origin": origin,
            "direction": direction,
            "normal": normal,
            "slope": 0.0,
            "intercept": 0.0,
            "points": pts,
        }

    xs = pts[:, 0].astype(np.float64)
    ys = pts[:, 1].astype(np.float64)
    slope = 0.0
    intercept = float(np.mean(ys))
    if pts.shape[0] >= 2 and float(np.std(xs)) > 1e-6:
        slope_fit, intercept_fit = np.polyfit(xs, ys, deg=1)
        slope = float(np.clip(slope_fit, -abs(float(max_slope)), abs(float(max_slope))))
        intercept = float(intercept_fit)
    direction = np.asarray([1.0, slope], dtype=np.float32)
    direction = direction / max(float(np.linalg.norm(direction)), 1e-6)
    normal = np.asarray([-direction[1], direction[0]], dtype=np.float32)
    origin_x = float(np.mean(xs))
    origin_y = float(slope * origin_x + intercept)
    origin = np.asarray([origin_x, origin_y], dtype=np.float32)
    return {
        "origin": origin,
        "direction": direction,
        "normal": normal,
        "slope": float(slope),
        "intercept": float(intercept),
        "points": pts,
    }


def _project_onto_row_line(center_xy: np.ndarray, row_line: Dict[str, object]) -> Tuple[float, float]:
    pt = np.asarray(center_xy, dtype=np.float32).reshape(2)
    origin = np.asarray(row_line["origin"], dtype=np.float32).reshape(2)
    direction = np.asarray(row_line["direction"], dtype=np.float32).reshape(2)
    normal = np.asarray(row_line["normal"], dtype=np.float32).reshape(2)
    rel = pt - origin
    along = float(np.dot(rel, direction))
    dist = float(abs(np.dot(rel, normal)))
    return along, dist


def _periodic_distance_1d(xs: np.ndarray, phase: float, period: float) -> np.ndarray:
    xs = np.asarray(xs, dtype=np.float32)
    period = max(float(period), 1e-6)
    return np.abs(np.remainder(xs - float(phase) + 0.5 * period, period) - 0.5 * period)


def _estimate_row_phase(seed_x: np.ndarray, period: float) -> float:
    seed_x = np.asarray(seed_x, dtype=np.float32).reshape(-1)
    period = max(float(period), 1e-6)
    if seed_x.size == 0:
        return 0.0
    remainders = np.remainder(seed_x, period)
    angles = remainders / period * (2.0 * np.pi)
    vector = np.exp(1j * angles).mean()
    if abs(vector) <= 1e-8:
        return float(np.median(remainders))
    return float((np.angle(vector) % (2.0 * np.pi)) / (2.0 * np.pi) * period)


def _smooth_profile_np(profile: np.ndarray, kernel_size: int = 5) -> np.ndarray:
    arr = np.asarray(profile, dtype=np.float32).reshape(-1)
    k = max(1, int(kernel_size))
    if k <= 1 or arr.size <= 2:
        return arr
    if k % 2 == 0:
        k += 1
    kernel = np.ones((k,), dtype=np.float32) / float(k)
    return np.convolve(arr, kernel, mode="same").astype(np.float32)


def _greedy_peak_indices(profile: np.ndarray, threshold: float, min_distance: int) -> np.ndarray:
    arr = np.asarray(profile, dtype=np.float32).reshape(-1)
    if arr.size == 0:
        return np.zeros((0,), dtype=np.int32)
    candidates: List[Tuple[int, float]] = []
    for idx in range(arr.size):
        value = float(arr[idx])
        if value < threshold:
            continue
        left = float(arr[idx - 1]) if idx > 0 else value
        right = float(arr[idx + 1]) if idx + 1 < arr.size else value
        if value >= left and value >= right:
            candidates.append((idx, value))
    if not candidates:
        return np.zeros((0,), dtype=np.int32)
    candidates.sort(key=lambda item: item[1], reverse=True)
    keep: List[int] = []
    dist = max(int(min_distance), 1)
    for idx, _value in candidates:
        if all(abs(idx - chosen) >= dist for chosen in keep):
            keep.append(int(idx))
    keep.sort()
    return np.asarray(keep, dtype=np.int32)


def _estimate_profile_period_cells(profile: np.ndarray, prior_period_cells: float, period_ratio: float) -> float:
    signal = np.asarray(profile, dtype=np.float32).reshape(-1)
    n = int(signal.size)
    if n < 8:
        return 0.0
    centered = signal - float(signal.mean())
    energy = float(np.dot(centered, centered))
    if energy <= 1e-8:
        return 0.0
    acf = np.correlate(centered, centered, mode="full")[n - 1 :]
    lag_min = 2
    lag_max = max(lag_min + 1, n // 2)
    if lag_max <= lag_min:
        return 0.0
    candidate_lags = np.arange(lag_min, lag_max + 1, dtype=np.int32)
    if prior_period_cells > 2.0:
        ratio = max(float(period_ratio), 1.05)
        lower = max(lag_min, int(np.floor(prior_period_cells / ratio)))
        upper = min(lag_max, int(np.ceil(prior_period_cells * ratio)))
        candidate_lags = candidate_lags[(candidate_lags >= lower) & (candidate_lags <= upper)]
    if candidate_lags.size == 0:
        return 0.0
    scores = acf[candidate_lags]
    best_idx = int(np.argmax(scores))
    return float(candidate_lags[best_idx])


def _build_dino_pooling_prior(
    dino_feat: Optional[torch.Tensor],
    *,
    model_input_size: Tuple[int, int],
    period_prior_px: float,
    row_period_prior_px: float,
    row_tolerance_px: float,
    row_threshold: float,
    peak_threshold: float,
    period_ratio: float,
) -> Tuple[Optional[torch.Tensor], Dict[str, float]]:
    stats = {
        "dino_pool_rows": 0.0,
        "dino_pool_x_period_cells": 0.0,
        "dino_pool_row_period_cells": 0.0,
        "dino_pool_prior_mean": 0.0,
        "dino_pool_prior_max": 0.0,
    }
    if dino_feat is None:
        return None, stats

    feat = dino_feat.detach().float().cpu()
    if feat.ndim == 4:
        feat = feat.squeeze(0)
    if feat.ndim != 3:
        return None, stats
    energy = feat.pow(2).mean(dim=0).sqrt()
    energy = energy / max(float(energy.max().item()), 1e-6)
    h, w = int(energy.shape[0]), int(energy.shape[1])
    model_w, model_h = int(model_input_size[0]), int(model_input_size[1])

    row_profile = _smooth_profile_np(energy.mean(dim=1).numpy(), kernel_size=5)
    row_period_cells = float(row_period_prior_px) / max(float(model_h), 1e-6) * float(h) if row_period_prior_px > 0.0 else 0.0
    row_tolerance_cells = max(1.0, float(row_tolerance_px) / max(float(model_h), 1e-6) * float(h))
    row_period_est = _estimate_profile_period_cells(row_profile, row_period_cells, period_ratio)
    row_peak_threshold = float(np.mean(row_profile) + max(float(row_profile.max()) - float(np.mean(row_profile)), 0.0) * float(row_threshold))
    row_peaks = _greedy_peak_indices(
        row_profile,
        threshold=row_peak_threshold,
        min_distance=max(1, int(round(max(row_tolerance_cells, row_period_est * 0.35 if row_period_est > 0 else row_tolerance_cells)))),
    )
    if row_peaks.size == 0:
        return None, stats

    xs = np.arange(w, dtype=np.float32)[None, :]
    ys = np.arange(h, dtype=np.float32)[:, None]
    prior = np.zeros((h, w), dtype=np.float32)
    x_periods: List[float] = []
    for row_idx in row_peaks.tolist():
        y0 = max(0, int(np.floor(float(row_idx) - row_tolerance_cells)))
        y1 = min(h, int(np.ceil(float(row_idx) + row_tolerance_cells)) + 1)
        if y0 >= y1:
            continue
        band = energy[y0:y1, :].mean(dim=0).numpy()
        band = _smooth_profile_np(band, kernel_size=5)
        x_period_prior_cells = float(period_prior_px) / max(float(model_w), 1e-6) * float(w) if period_prior_px > 0.0 else 0.0
        x_period = _estimate_profile_period_cells(band, x_period_prior_cells, period_ratio)
        if x_period < 2.0:
            if x_period_prior_cells > 2.0:
                x_period = float(x_period_prior_cells)
            else:
                continue
        x_periods.append(float(x_period))
        band_threshold = float(np.mean(band) + max(float(band.max()) - float(np.mean(band)), 0.0) * float(peak_threshold))
        x_peaks = _greedy_peak_indices(
            band,
            threshold=band_threshold,
            min_distance=max(1, int(round(0.5 * x_period))),
        )
        if x_peaks.size == 0:
            continue
        phase = _estimate_row_phase(x_peaks.astype(np.float32), float(x_period))
        sigma_x = max(1.0, 0.22 * float(x_period))
        sigma_y = max(1.0, 0.55 * row_tolerance_cells)
        dist_x = _periodic_distance_1d(xs, phase, float(x_period))
        comb_x = np.exp(-0.5 * (dist_x / sigma_x) ** 2)
        band_y = np.exp(-0.5 * (((ys - float(row_idx)) / sigma_y) ** 2))
        row_strength = float(row_profile[int(row_idx)])
        prior = np.maximum(prior, row_strength * (band_y * comb_x))

    if float(np.max(prior)) <= 1e-8:
        return None, stats
    prior = prior / float(prior.max())
    prior_t = torch.from_numpy(prior.astype(np.float32))
    stats.update(
        {
            "dino_pool_rows": float(len(row_peaks)),
            "dino_pool_x_period_cells": float(np.mean(x_periods)) if x_periods else 0.0,
            "dino_pool_row_period_cells": float(row_period_est) if row_period_est > 0.0 else 0.0,
            "dino_pool_prior_mean": float(prior.mean()),
            "dino_pool_prior_max": float(prior.max()),
        }
    )
    return prior_t, stats


def _rescore_predictions_by_pattern(
    predictions: Sequence[Dict[str, object]],
    support_map: Optional[torch.Tensor],
    dino_pool_prior: Optional[torch.Tensor],
    *,
    image_size: Tuple[int, int],
    anchor_conf: float,
    support_threshold: float,
    propagation_strength: float,
    off_freq_suppress: float,
    support_power: float,
    dino_pooling_strength: float,
    period_prior_px: float,
    dense_suppress: float,
    min_gap_ratio: float,
    line_max_slope: float,
    row_tolerance_px: float,
    poly_center,
    sample_grid_value,
) -> Tuple[List[Dict[str, object]], Dict[str, float]]:
    preds = [dict(pred) for pred in predictions]
    if (support_map is None and dino_pool_prior is None) or not preds:
        return preds, {
            "pattern_anchor_count": 0.0,
            "pattern_row_count_rescore": 0.0,
            "pattern_rescore_mean_before": 0.0,
            "pattern_rescore_mean_after": 0.0,
            "pattern_rescore_promoted_count": 0.0,
            "pattern_rescore_suppressed_count": 0.0,
            "dino_pool_mean_support": 0.0,
            "pattern_dense_pair_count": 0.0,
            "pattern_dense_suppressed_count": 0.0,
            "pattern_dense_mean_gap_px": 0.0,
            "pattern_line_prior_mean": 0.0,
            "pattern_line_count": 0.0,
        }

    dino_pool_scores: List[float] = []
    for pred in preds:
        center = poly_center(np.asarray(pred["poly"], dtype=np.float32))
        support_local = (
            float(sample_grid_value(support_map, float(center[0]), float(center[1]), image_size))
            if support_map is not None
            else 0.0
        )
        support_dino = (
            float(sample_grid_value(dino_pool_prior, float(center[0]), float(center[1]), image_size))
            if dino_pool_prior is not None
            else 0.0
        )
        combined_support = max(support_local, float(dino_pooling_strength) * support_dino)
        pred["pattern_support_local"] = support_local
        pred["pattern_support_dino_pool"] = support_dino
        pred["pattern_support"] = combined_support
        dino_pool_scores.append(support_dino)

    rows = _group_prediction_rows(preds, row_tolerance_px=row_tolerance_px, poly_center=poly_center)
    row_lines = {
        row_id: _fit_row_line(row, preds, poly_center=poly_center, max_slope=line_max_slope)
        for row_id, row in enumerate(rows)
    }
    row_anchor_scores: Dict[int, float] = {}
    row_id_map: Dict[int, int] = {}
    anchor_threshold = max(float(anchor_conf), 0.0)
    support_thr = float(support_threshold)
    anchor_scores_global: List[float] = []
    for row_id, row in enumerate(rows):
        anchor_scores = [
            float(preds[idx].get("conf", 0.0))
            for idx in row
            if float(preds[idx].get("conf", 0.0)) >= anchor_threshold
            and float(preds[idx].get("pattern_support", 0.0)) >= support_thr
        ]
        for idx in row:
            row_id_map[idx] = row_id
        if anchor_scores:
            row_anchor_scores[row_id] = float(np.median(anchor_scores))
            anchor_scores_global.extend(anchor_scores)

    global_anchor = float(np.median(anchor_scores_global)) if anchor_scores_global else anchor_threshold
    line_prior_scores: List[float] = []
    promoted = 0
    suppressed = 0
    before_scores: List[float] = []
    after_scores: List[float] = []

    for idx, pred in enumerate(preds):
        base_conf = float(pred.get("conf", 0.0))
        row_id = row_id_map.get(idx, -1)
        row_anchor = row_anchor_scores.get(row_id, global_anchor)
        row_line = row_lines.get(row_id)
        center = poly_center(np.asarray(pred["poly"], dtype=np.float32))
        line_prior = 0.0
        if row_line is not None and period_prior_px > 4.0:
            alongs = np.asarray(
                [
                    _project_onto_row_line(poly_center(np.asarray(preds[j]["poly"], dtype=np.float32)), row_line)[0]
                    for j in rows[row_id]
                ],
                dtype=np.float32,
            )
            phase = _estimate_row_phase(alongs, float(period_prior_px))
            along, dist = _project_onto_row_line(center, row_line)
            sigma_line = max(1.0, 0.55 * float(row_tolerance_px))
            sigma_period = max(1.0, 0.22 * float(period_prior_px))
            line_gate = float(np.exp(-0.5 * (dist / sigma_line) ** 2))
            period_dist = float(_periodic_distance_1d(np.asarray([along], dtype=np.float32), phase, float(period_prior_px))[0])
            freq_gate = float(np.exp(-0.5 * (period_dist / sigma_period) ** 2))
            line_prior = float(np.clip(line_gate * freq_gate, 0.0, 1.0))
            line_prior_scores.append(line_prior)
        support = max(float(pred.get("pattern_support", 0.0)), line_prior)
        pred["pattern_support_line"] = line_prior
        pred["pattern_support"] = support
        propagated = float(propagation_strength) * row_anchor * (max(support, 0.0) ** max(float(support_power), 1e-6))
        boosted = max(base_conf, propagated)
        suppressed_score = boosted * (1.0 - float(off_freq_suppress) * max(0.0, 1.0 - support))
        if base_conf >= anchor_threshold and support >= support_thr:
            final_conf = max(base_conf, suppressed_score)
        else:
            final_conf = suppressed_score
        final_conf = float(np.clip(final_conf, 0.0, 1.0))
        pred["conf_before_rescore"] = base_conf
        pred["conf_after_rescore"] = final_conf
        pred["conf"] = final_conf
        before_scores.append(base_conf)
        after_scores.append(final_conf)
        if final_conf > base_conf + 1e-6:
            promoted += 1
        elif final_conf < base_conf - 1e-6:
            suppressed += 1

    dense_pair_count = 0
    dense_suppressed = 0
    dense_gaps: List[float] = []
    min_gap_px = max(4.0, float(period_prior_px) * max(float(min_gap_ratio), 0.05))
    dense_strength = float(np.clip(dense_suppress, 0.0, 1.0))
    if dense_strength > 0.0 and min_gap_px > 0.0:
        for row in rows:
            if len(row) < 2:
                continue
            row_id = row_id_map.get(row[0], -1)
            row_line = row_lines.get(row_id)
            row_sorted = sorted(
                row,
                key=lambda ridx: _project_onto_row_line(
                    poly_center(np.asarray(preds[ridx]["poly"], dtype=np.float32)),
                    row_line if row_line is not None else _fit_row_line(row, preds, poly_center=poly_center, max_slope=line_max_slope),
                )[0],
            )
            centers_t = [
                _project_onto_row_line(
                    poly_center(np.asarray(preds[ridx]["poly"], dtype=np.float32)),
                    row_line if row_line is not None else _fit_row_line(row, preds, poly_center=poly_center, max_slope=line_max_slope),
                )[0]
                for ridx in row_sorted
            ]
            for left_pos in range(len(row_sorted) - 1):
                left_idx = row_sorted[left_pos]
                right_idx = row_sorted[left_pos + 1]
                dx = float(centers_t[left_pos + 1] - centers_t[left_pos])
                if dx >= min_gap_px:
                    continue
                dense_pair_count += 1
                dense_gaps.append(dx)
                overlap_ratio = max(0.0, 1.0 - dx / max(min_gap_px, 1e-6))
                left_conf = float(preds[left_idx].get("conf", 0.0))
                right_conf = float(preds[right_idx].get("conf", 0.0))
                left_support = float(preds[left_idx].get("pattern_support", 0.0))
                right_support = float(preds[right_idx].get("pattern_support", 0.0))
                left_score = 0.65 * left_conf + 0.35 * left_support
                right_score = 0.65 * right_conf + 0.35 * right_support
                target_idx = left_idx if left_score <= right_score else right_idx
                old_conf = float(preds[target_idx].get("conf", 0.0))
                penalty = 1.0 - dense_strength * overlap_ratio
                new_conf = float(np.clip(old_conf * penalty, 0.0, 1.0))
                if new_conf < old_conf - 1e-6:
                    preds[target_idx]["conf"] = new_conf
                    preds[target_idx]["conf_after_dense"] = new_conf
                    dense_suppressed += 1

    stats = {
        "pattern_anchor_count": float(len(anchor_scores_global)),
        "pattern_row_count_rescore": float(len(rows)),
        "pattern_rescore_mean_before": float(np.mean(before_scores)) if before_scores else 0.0,
        "pattern_rescore_mean_after": float(np.mean(after_scores)) if after_scores else 0.0,
        "pattern_rescore_promoted_count": float(promoted),
        "pattern_rescore_suppressed_count": float(suppressed),
        "dino_pool_mean_support": float(np.mean(dino_pool_scores)) if dino_pool_scores else 0.0,
        "pattern_dense_pair_count": float(dense_pair_count),
        "pattern_dense_suppressed_count": float(dense_suppressed),
        "pattern_dense_mean_gap_px": float(np.mean(dense_gaps)) if dense_gaps else 0.0,
        "pattern_line_prior_mean": float(np.mean(line_prior_scores)) if line_prior_scores else 0.0,
        "pattern_line_count": float(len(row_lines)),
    }
    return preds, stats


def run_detect_test(args: argparse.Namespace, device: str, dataset: ImageFolderDataset) -> None:
    deps = _load_detect_test_deps()
    YOLO = deps["YOLO"]
    make_tile = deps["make_tile"]
    save_metrics_csv = deps["save_metrics_csv"]
    draw_predictions = deps["draw_predictions"]
    feature_level_label = deps["feature_level_label"]
    feature_response_image = deps["feature_response_image"]
    find_detect_module = deps["find_detect_module"]
    make_grid = deps["make_grid"]
    poly_center = deps["poly_center"]
    predict_one = deps["predict_one"]
    prediction_delta_metrics = deps["prediction_delta_metrics"]
    sample_grid_value = deps["sample_grid_value"]
    filter_predictions_by_shape = deps["filter_predictions_by_shape"]
    load_image_prior_index = deps["load_image_prior_index"]
    load_species_prior_bank = deps["load_species_prior_bank"]
    pattern_prior_from_maps = deps["pattern_prior_from_maps"]
    resolve_external_pattern_prior = deps["resolve_external_pattern_prior"]
    transform_anchor_points = deps["transform_anchor_points"]

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    export_pseudo_root = (
        Path(args.export_pseudo_dir).expanduser().resolve()
        if args.export_pseudo_dir
        else None
    )
    if export_pseudo_root is not None:
        (export_pseudo_root / "labels" / args.export_split).mkdir(parents=True, exist_ok=True)
        (export_pseudo_root / "images" / args.export_split).mkdir(parents=True, exist_ok=True)

    yolo_weights = Path(args.yolo_weights).expanduser().resolve()
    if not yolo_weights.is_file():
        raise FileNotFoundError(f"YOLO weights not found: {yolo_weights}")

    yolo = YOLO(str(yolo_weights))
    _ = find_detect_module(yolo.model)  # validate detect head exists
    dino_model = torch.hub.load(
        args.repo_dir,
        args.model_name,
        source="local",
        weights=args.weights,
    ).to(device).eval()

    image_prior_index = load_image_prior_index(Path(args.pattern_image_priors).resolve()) if args.pattern_image_priors else {}
    species_prior_bank = load_species_prior_bank(Path(args.pattern_species_prior_bank).resolve()) if args.pattern_species_prior_bank else {}

    rows: List[Dict[str, object]] = []
    prediction_exports: List[Dict[str, object]] = []
    prediction_exports_before: List[Dict[str, object]] = []
    prediction_exports_after: List[Dict[str, object]] = []
    image_paths = dataset.files[: max(0, int(args.num_samples))] if args.num_samples > 0 else dataset.files
    tile_size = min(420, int(args.imgsz))

    print(f"[detect_test] output_dir={output_dir}")
    print(f"[detect_test] yolo={yolo_weights}")
    print(f"[detect_test] dino={args.weights}")
    print(f"[detect_test] feature_level={args.feature_level} imgsz={args.imgsz}")
    print(f"[detect_test] pattern_apply_mode={args.pattern_apply_mode}")

    for idx, image_path in enumerate(image_paths):
        original_pil = Image.open(image_path).convert("RGB")

        before_preds, before_state = _predict_one_capture_feature(
            yolo,
            str(image_path),
            imgsz=args.imgsz,
            conf=args.conf,
            iou=args.iou,
            max_det=args.max_det,
            device=torch.device(device),
            feature_level=args.feature_level,
            predict_one=predict_one,
            find_detect_module=find_detect_module,
        )

        before_feat = before_state.get("feature")
        model_input_size = before_state.get("image_size")
        if before_feat is None or model_input_size is None:
            raise RuntimeError(f"Failed to capture detect-input feature for {image_path}")

        pattern_cfg = _effective_detect_test_pattern_settings(
            args=args,
            original_size=original_pil.size,
            model_input_size=model_input_size,
            image_path=str(image_path),
            image_prior_index=image_prior_index,
            species_prior_bank=species_prior_bank,
            transform_anchor_points=transform_anchor_points,
            resolve_external_pattern_prior=resolve_external_pattern_prior,
        )
        template_anchor_conf = float(args.template_match_anchor_conf)
        if template_anchor_conf < 0.0:
            template_anchor_conf = float(args.pattern_anchor_conf if args.pattern_anchor_conf >= 0.0 else args.conf)

        enhanced_conf = args.proposal_conf if args.pattern_filter else args.conf
        after_preds_raw, after_state = _predict_one_pattern_enhanced(
            yolo,
            dino_model,
            str(image_path),
            imgsz=args.imgsz,
            conf=enhanced_conf,
            iou=args.iou,
            max_det=args.max_det,
            device=torch.device(device),
            feature_level=args.feature_level,
            period_prior_px=float(pattern_cfg["period_prior_px"]),
            row_period_prior_px=float(pattern_cfg["row_period_prior_px"]),
            row_tolerance_px=float(pattern_cfg["row_tolerance_px"]),
            row_center_priors_px=pattern_cfg["row_centers"],
            pattern_seed_threshold=args.pattern_seed_threshold,
            pattern_seed_topk=args.pattern_seed_topk,
            pattern_gate_threshold=args.pattern_gate_threshold,
            pattern_min_row_seeds=args.pattern_min_row_seeds,
            pattern_line_max_slope=args.pattern_line_max_slope,
            pattern_period_prior_ratio=args.pattern_period_prior_ratio,
            pattern_period_scale=args.pattern_period_scale,
            pattern_row_sigma_scale=args.pattern_row_sigma_scale,
            pattern_period_sigma_scale=args.pattern_period_sigma_scale,
            pattern_cross_row_strength=args.pattern_cross_row_strength,
            pattern_response_period_blend=args.pattern_response_period_blend,
            pattern_force_horizontal_prior=bool(args.pattern_force_horizontal_prior),
            pattern_full_row_support=bool(args.pattern_full_row_support),
            pattern_apply_mode=args.pattern_apply_mode,
            original_size=original_pil.size,
            template_predictions=before_preds,
            template_match_topk=args.template_match_topk,
            template_match_anchor_conf=template_anchor_conf,
            template_match_min_cells=args.template_match_min_cells,
            template_match_max_cells=args.template_match_max_cells,
            template_match_padding_cells=args.template_match_padding_cells,
            template_frequency_strength=args.template_frequency_strength,
            template_energy_fallback=bool(args.template_energy_fallback),
            predict_one=predict_one,
            find_detect_module=find_detect_module,
            pattern_prior_from_maps=pattern_prior_from_maps,
        )

        after_feat = after_state.get("feature")
        if after_feat is None:
            raise RuntimeError(f"Failed to capture enhanced detect-input feature for {image_path}")

        support_map = after_state.get("support_map")
        dino_feature = after_state.get("dino_feature")
        after_preds = list(after_preds_raw)
        dino_mask_map: Optional[torch.Tensor] = None
        support_stats: Dict[str, float] = {
            "pattern_support_removed_count": 0.0,
            "pattern_support_kept_count": float(len(after_preds)),
            "pattern_support_mean_kept": 0.0,
            "pattern_support_mean_removed": 0.0,
        }
        rescore_stats: Dict[str, float] = {
            "pattern_anchor_count": 0.0,
            "pattern_row_count_rescore": 0.0,
            "pattern_rescore_mean_before": 0.0,
            "pattern_rescore_mean_after": 0.0,
            "pattern_rescore_promoted_count": 0.0,
            "pattern_rescore_suppressed_count": 0.0,
            "pattern_dense_pair_count": 0.0,
            "pattern_dense_suppressed_count": 0.0,
            "pattern_dense_mean_gap_px": 0.0,
            "pattern_line_prior_mean": 0.0,
            "pattern_line_count": 0.0,
        }
        synthetic_stats: Dict[str, float] = {
            "synthetic_completion_enabled": 1.0 if bool(args.synthetic_completion) else 0.0,
            "synthetic_candidate_count": 0.0,
            "synthetic_added_count": 0.0,
            "synthetic_mean_support": 0.0,
            "synthetic_mean_template": 0.0,
            "synthetic_mean_frequency": 0.0,
            "synthetic_min_distance_px": 0.0,
            "synthetic_final_count": 0.0,
            "synthetic_row_rejected_count": 0.0,
            "synthetic_span_rejected_count": 0.0,
            "synthetic_row_tolerance_px": 0.0,
        }
        shape_stats: Dict[str, float] = {
            "shape_removed_count": 0.0,
            "shape_ref_width_px": 0.0,
            "shape_ref_height_px": 0.0,
            "shape_ref_area_px2": 0.0,
            "shape_ref_aspect": 0.0,
            "shape_edge_kept_count": 0.0,
        }
        if args.pattern_filter:
            rescored_preds_all = list(after_preds)
            ref_predictions = before_preds if before_preds else after_preds
            ref_heights = [_prediction_box_height_px(pred) for pred in ref_predictions] if ref_predictions else []
            row_tol_rescore = (
                float(np.median(ref_heights)) * float(args.pattern_row_tolerance_scale)
                if ref_heights
                else max(float(args.pattern_row_tolerance_px), 18.0)
            )
            dino_pool_prior, dino_pool_stats = _build_dino_pooling_prior(
                dino_feature,
                model_input_size=model_input_size,
                period_prior_px=float(pattern_cfg["period_prior_px"]),
                row_period_prior_px=float(pattern_cfg["row_period_prior_px"]),
                row_tolerance_px=float(pattern_cfg["row_tolerance_px"]),
                row_threshold=float(args.dino_pooling_row_threshold),
                peak_threshold=float(args.dino_pooling_peak_threshold),
                period_ratio=float(args.dino_pooling_period_ratio),
            )
            dino_mask_map = dino_pool_prior if dino_pool_prior is not None else support_map
            anchor_conf = float(args.conf if args.pattern_anchor_conf < 0.0 else args.pattern_anchor_conf)
            after_preds, rescore_stats = _rescore_predictions_by_pattern(
                after_preds,
                support_map,
                dino_pool_prior,
                image_size=original_pil.size,
                anchor_conf=anchor_conf,
                support_threshold=args.pattern_support_threshold,
                propagation_strength=args.pattern_propagation_strength,
                off_freq_suppress=args.pattern_offfreq_suppress,
                support_power=args.pattern_support_power,
                dino_pooling_strength=args.dino_pooling_prior_strength,
                period_prior_px=float(pattern_cfg["period_prior_px"]),
                dense_suppress=args.pattern_dense_suppress,
                min_gap_ratio=args.pattern_min_gap_ratio,
                line_max_slope=args.pattern_line_max_slope,
                row_tolerance_px=row_tol_rescore,
                poly_center=poly_center,
                sample_grid_value=sample_grid_value,
            )
            rescore_stats.update(dino_pool_stats)
            if bool(args.synthetic_completion):
                reference_for_synthetic = before_preds if before_preds else after_preds
                synthetic_preds, synthetic_stats = _complete_synthetic_predictions(
                    after_preds,
                    reference_for_synthetic,
                    support_map,
                    after_state.get("prior_map"),
                    after_state.get("guide_map"),
                    image_size=original_pil.size,
                    model_input_size=model_input_size,
                    final_conf_threshold=float(args.conf),
                    period_prior_px=float(pattern_cfg["period_prior_px"]),
                    support_threshold=float(args.synthetic_support_threshold),
                    template_threshold=float(args.synthetic_template_threshold),
                    frequency_threshold=float(args.synthetic_frequency_threshold),
                    min_distance_scale=float(args.synthetic_min_distance_scale),
                    min_row_proposals=int(args.synthetic_min_row_proposals),
                    row_tolerance_scale=float(args.synthetic_row_tolerance_scale),
                    row_span_margin_scale=float(args.synthetic_row_span_margin_scale),
                    synthetic_conf=float(args.synthetic_conf),
                    max_boxes=int(args.synthetic_max_boxes),
                    poly_center=poly_center,
                    sample_grid_value=sample_grid_value,
                )
                if synthetic_preds:
                    after_preds.extend(synthetic_preds)
            rescored_preds_all = list(after_preds)
            before_filter_count = len(after_preds)
            after_preds = [pred for pred in after_preds if float(pred.get("conf", 0.0)) >= float(args.conf)]
            removed_count = max(0, before_filter_count - len(after_preds))
            kept_scores = [float(pred.get("pattern_support", 0.0)) for pred in after_preds]
            removed_scores = [
                float(pred.get("pattern_support", 0.0))
                for pred in rescored_preds_all
                if float(pred.get("conf", 0.0)) < float(args.conf)
            ]
            support_stats = {
                "pattern_support_removed_count": float(removed_count),
                "pattern_support_kept_count": float(len(after_preds)),
                "pattern_support_mean_kept": float(np.mean(kept_scores)) if kept_scores else 0.0,
                "pattern_support_mean_removed": float(np.mean(removed_scores)) if removed_scores else 0.0,
            }
        else:
            rescored_preds_all = list(after_preds)
            if bool(args.synthetic_completion):
                reference_for_synthetic = before_preds if before_preds else after_preds
                synthetic_preds, synthetic_stats = _complete_synthetic_predictions(
                    after_preds,
                    reference_for_synthetic,
                    support_map,
                    after_state.get("prior_map"),
                    after_state.get("guide_map"),
                    image_size=original_pil.size,
                    model_input_size=model_input_size,
                    final_conf_threshold=float(args.conf),
                    period_prior_px=float(pattern_cfg["period_prior_px"]),
                    support_threshold=float(args.synthetic_support_threshold),
                    template_threshold=float(args.synthetic_template_threshold),
                    frequency_threshold=float(args.synthetic_frequency_threshold),
                    min_distance_scale=float(args.synthetic_min_distance_scale),
                    min_row_proposals=int(args.synthetic_min_row_proposals),
                    row_tolerance_scale=float(args.synthetic_row_tolerance_scale),
                    row_span_margin_scale=float(args.synthetic_row_span_margin_scale),
                    synthetic_conf=float(args.synthetic_conf),
                    max_boxes=int(args.synthetic_max_boxes),
                    poly_center=poly_center,
                    sample_grid_value=sample_grid_value,
                )
                if synthetic_preds:
                    after_preds.extend(synthetic_preds)
                    rescored_preds_all = list(after_preds)
        if args.shape_filter:
            ref_predictions = before_preds if before_preds else after_preds
            after_preds, _removed_shape, shape_stats = filter_predictions_by_shape(
                after_preds,
                ref_predictions,
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
        synthetic_stats["synthetic_final_count"] = float(
            sum(1 for pred in after_preds if bool(pred.get("synthetic", False)))
        )

        template_response_map = after_state.get("guide_map")
        frequency_prior_map = after_state.get("prior_map")
        template_response_img = _map_to_feature_response_image(
            template_response_map,
            size=original_pil.size,
            feature_response_image=feature_response_image,
        )
        frequency_prior_img = _map_to_feature_response_image(
            frequency_prior_map,
            size=original_pil.size,
            feature_response_image=feature_response_image,
        )
        score_boost_map = _score_boost_heatmap(
            rescored_preds_all,
            image_size=original_pil.size,
            max_boxes=int(args.max_det),
        )
        score_boost_img = _map_to_feature_response_image(
            score_boost_map,
            size=original_pil.size,
            feature_response_image=feature_response_image,
        )
        before_prediction_heatmap = _prediction_confidence_heatmap(
            before_preds,
            image_size=original_pil.size,
            max_boxes=int(args.max_det),
        )
        after_prediction_heatmap = _prediction_confidence_heatmap(
            after_preds,
            image_size=original_pil.size,
            max_boxes=int(args.max_det),
        )
        before_prediction_heatmap_img = _map_to_feature_response_image(
            before_prediction_heatmap,
            size=original_pil.size,
            feature_response_image=feature_response_image,
        )
        after_prediction_heatmap_img = _map_to_feature_response_image(
            after_prediction_heatmap,
            size=original_pil.size,
            feature_response_image=feature_response_image,
        )
        before_prediction_heatmap_overlay = _overlay_map_on_image(
            original_pil,
            before_prediction_heatmap,
            feature_response_image=feature_response_image,
            alpha=0.45,
        )
        after_prediction_heatmap_overlay = _overlay_map_on_image(
            original_pil,
            after_prediction_heatmap,
            feature_response_image=feature_response_image,
            alpha=0.45,
        )
        before_overlay = draw_predictions(
            original_pil,
            before_preds,
            (0, 245, 255),
            label_prefix="B:",
            max_labels=0,
            annotation_scale=1.0,
            line_width_scale=0.45,
            shadow=False,
        )
        after_overlay = draw_predictions(
            original_pil,
            after_preds,
            (255, 35, 210),
            label_prefix="A:",
            max_labels=0,
            annotation_scale=1.0,
            line_width_scale=0.45,
            shadow=False,
        )
        score_boost_overlay = _draw_rescore_effect_overlay(
            original_pil,
            rescored_preds_all,
            final_conf_threshold=float(args.conf),
            proposal_conf_threshold=float(args.proposal_conf if args.pattern_filter else args.conf),
            max_boxes=int(args.max_det),
        )
        feature_panel = make_grid(
            [
                _make_debug_tile(original_pil, "Origin Image", tile_size, caption=False, font_size=10),
                _make_debug_tile(
                    template_response_img if template_response_img is not None else original_pil,
                    "DINO Response",
                    tile_size,
                    caption=False,
                    font_size=10,
                ),
                _make_debug_tile(
                    frequency_prior_img if frequency_prior_img is not None else original_pil,
                    "Frequency Prior",
                    tile_size,
                    caption=False,
                    font_size=10,
                ),
                _make_debug_tile(
                    score_boost_img if score_boost_img is not None else original_pil,
                    "Score Boost Heatmap",
                    tile_size,
                    caption=False,
                    font_size=10,
                ),
            ],
            cols=4,
        )
        prediction_panel = make_grid(
            [
                _make_debug_tile(before_overlay, "Before Pattern Prediction", tile_size, caption=False, font_size=10),
                _make_debug_tile(after_overlay, "After Pattern Prediction", tile_size, caption=False, font_size=10),
                _make_debug_tile(score_boost_overlay, "Score Boosted Proposals", tile_size, caption=False, font_size=10),
            ],
            cols=3,
        )

        stem = image_path.stem
        individual_prefix = f"{idx:04d}_{stem}"
        feature_panel_path = output_dir / f"{idx:04d}_{stem}_feature_panel.png"
        prediction_panel_path = output_dir / f"{idx:04d}_{stem}_prediction_panel.png"
        feature_panel.save(feature_panel_path)
        prediction_panel.save(prediction_panel_path)
        individual_paths = _save_individual_debug_images(
            output_dir=output_dir,
            prefix=individual_prefix,
            images={
                "origin_image": original_pil,
                "dino_response": template_response_img if template_response_img is not None else Image.new("RGB", original_pil.size, (0, 0, 0)),
                "frequency_prior": frequency_prior_img if frequency_prior_img is not None else Image.new("RGB", original_pil.size, (0, 0, 0)),
                "score_boost_heatmap": score_boost_img if score_boost_img is not None else Image.new("RGB", original_pil.size, (0, 0, 0)),
                "before_prediction_heatmap": before_prediction_heatmap_img,
                "after_prediction_heatmap": after_prediction_heatmap_img,
                "before_prediction_heatmap_overlay": before_prediction_heatmap_overlay,
                "after_prediction_heatmap_overlay": after_prediction_heatmap_overlay,
                "before_pattern_prediction": before_overlay,
                "after_pattern_prediction": after_overlay,
                "score_boosted_proposals": score_boost_overlay,
            },
            make_tile=make_tile,
            tile_size=tile_size,
        )

        row = {
            "image_path": str(image_path),
            "feature_panel_path": str(feature_panel_path),
            "prediction_panel_path": str(prediction_panel_path),
            "individual_dir": str(output_dir / f"{individual_prefix}_individual"),
            "pattern_apply_mode": str(args.pattern_apply_mode),
            "prior_source": str(pattern_cfg["source"]),
            "prior_family": str(pattern_cfg["family"]),
            "period_prior_px": float(pattern_cfg["period_prior_px"]),
            "row_period_prior_px": float(pattern_cfg["row_period_prior_px"]),
            "row_tolerance_px": float(pattern_cfg["row_tolerance_px"]),
            "after_raw_box_count": float(len(after_preds_raw)),
        }
        row.update(individual_paths)
        exported_preds = _serialize_predictions(after_preds, original_pil.size)
        prediction_exports.append(
            {
                "image_path": str(image_path),
                "width": int(original_pil.size[0]),
                "height": int(original_pil.size[1]),
                "predictions": exported_preds,
            }
        )
        if args.export_split_preds:
            prediction_exports_before.append(
                {
                    "image_path": str(image_path),
                    "width": int(original_pil.size[0]),
                    "height": int(original_pil.size[1]),
                    "predictions": _serialize_predictions(before_preds, original_pil.size),
                }
            )
            prediction_exports_after.append(
                {
                    "image_path": str(image_path),
                    "width": int(original_pil.size[0]),
                    "height": int(original_pil.size[1]),
                    "predictions": exported_preds,
                }
            )
        if export_pseudo_root is not None:
            row.update(
                _export_predictions_as_pseudo_labels(
                    Path(image_path),
                    after_preds,
                    image_size=original_pil.size,
                    export_root=export_pseudo_root,
                    split=str(args.export_split),
                    edge_as_incomplete=bool(args.pseudo_edge_as_incomplete),
                    complete_class_id=int(args.pseudo_complete_class_id),
                    incomplete_class_id=int(args.pseudo_incomplete_class_id),
                    edge_margin_px=float(args.edge_keep_margin_px),
                    edge_margin_ratio=float(args.edge_keep_margin_ratio),
                    hard_edge_margin_px=float(args.pseudo_hard_edge_margin_px),
                    soft_edge_policy=str(args.pseudo_soft_edge_policy),
                )
            )
        row.update(prediction_delta_metrics(before_preds, after_preds))
        row.update({f"pattern_{k}": v for k, v in dict(after_state.get("pattern_stats", {})).items()})
        row.update(rescore_stats)
        row.update(synthetic_stats)
        row.update(support_stats)
        row.update(shape_stats)
        rows.append(row)

        print(
            "[detect_test %d/%d] %s before=%d raw_after=%d synthetic=%d after=%d matched=%.0f iou=%.4f shift=%.2f prior=%s"
            % (
                idx + 1,
                len(image_paths),
                image_path.name,
                int(row["before_box_count"]),
                int(row["after_raw_box_count"]),
                int(row["synthetic_added_count"]),
                int(row["after_box_count"]),
                float(row["matched_box_count"]),
                float(row["mean_matched_aabb_iou"]),
                float(row["mean_center_shift_px"]),
                str(row["prior_source"]),
            )
        )

    if not rows:
        print("[warn] detect_test produced no samples.")
        return

    metrics_path = output_dir / "metrics.csv"
    save_metrics_csv(metrics_path, rows)
    predictions_path = output_dir / "predictions.json"
    predictions_path.write_text(
        json.dumps(
            {
                "format": "dino_bypass_obb_predictions_v1",
                "num_images": int(len(prediction_exports)),
                "images": prediction_exports,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    predictions_before_path = None
    predictions_after_path = None
    if args.export_split_preds:
        predictions_before_path = output_dir / "predictions_before.json"
        predictions_after_path = output_dir / "predictions_after.json"
        predictions_before_path.write_text(
            json.dumps(
                {
                    "format": "dino_bypass_obb_predictions_v1",
                    "num_images": int(len(prediction_exports_before)),
                    "images": prediction_exports_before,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        predictions_after_path.write_text(
            json.dumps(
                {
                    "format": "dino_bypass_obb_predictions_v1",
                    "num_images": int(len(prediction_exports_after)),
                    "images": prediction_exports_after,
                },
                indent=2,
            ),
            encoding="utf-8",
        )

    completion_box_counts = _save_completion_box_count_figure(rows, output_dir)
    summary = {
        "num_samples": int(len(rows)),
        "pattern_apply_mode": str(args.pattern_apply_mode),
        "predictions_json": str(predictions_path),
        "mean_before_box_count": float(np.mean([float(r["before_box_count"]) for r in rows])),
        "mean_after_raw_box_count": float(np.mean([float(r["after_raw_box_count"]) for r in rows])),
        "mean_after_box_count": float(np.mean([float(r["after_box_count"]) for r in rows])),
        "mean_matched_aabb_iou": float(np.mean([float(r["mean_matched_aabb_iou"]) for r in rows])),
        "mean_center_shift_px": float(np.mean([float(r["mean_center_shift_px"]) for r in rows])),
        "mean_pattern_support_removed_count": float(np.mean([float(r["pattern_support_removed_count"]) for r in rows])),
        "mean_synthetic_added_count": float(np.mean([float(r["synthetic_added_count"]) for r in rows])),
        "mean_synthetic_final_count": float(np.mean([float(r["synthetic_final_count"]) for r in rows])),
        "mean_shape_removed_count": float(np.mean([float(r["shape_removed_count"]) for r in rows])),
        "bbox_counts_before_after_pdf": str(output_dir / "bbox_counts_before_after.pdf"),
        "bbox_counts_before_after_png": str(output_dir / "bbox_counts_before_after.png"),
    }
    summary.update(completion_box_counts)
    if predictions_before_path is not None and predictions_after_path is not None:
        summary["predictions_before_json"] = str(predictions_before_path)
        summary["predictions_after_json"] = str(predictions_after_path)
    if export_pseudo_root is not None:
        summary["export_pseudo_dir"] = str(export_pseudo_root)
        summary["mean_export_box_count"] = float(np.mean([float(r.get("export_box_count", 0.0)) for r in rows]))
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"[done] detect_test results saved to {output_dir}")
    if export_pseudo_root is not None:
        print(f"[done] exported pseudo labels to {export_pseudo_root}")


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        print("[warn] CUDA not available, using CPU")
        device = "cpu"

    input_dir = Path(args.input_dir)
    dataset = ImageFolderDataset(input_dir)
    run_detect_test(args, device, dataset)


if __name__ == "__main__":
    main()
