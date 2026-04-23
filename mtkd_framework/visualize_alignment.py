#!/usr/bin/env python3
"""
Visualize MTKD feature alignment with PCA panels and cosine-similarity maps.

Example:
    python -m mtkd_framework.visualize_alignment \
        --config outputs/mtkd_train_epoch100_version0.1/config.json \
        --checkpoint outputs/mtkd_train_epoch100_version0.1/best_model.pth \
        --split val \
        --num-samples 8
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
from PIL import Image, ImageDraw

# Prefer the repository's bundled Ultralytics fork over site-packages.
REPO_ROOT = Path(__file__).resolve().parents[1]
ULTRALYTICS_ROOT = REPO_ROOT / "ultralytics"
for candidate in (REPO_ROOT, ULTRALYTICS_ROOT):
    if candidate.is_dir():
        candidate_str = str(candidate)
        if candidate_str not in sys.path:
            sys.path.insert(0, candidate_str)

from .data.stomata_dataset import build_stomata_dataloaders
from .models.mtkd_model_v2 import build_mtkd_model_v2
from .train_v2 import get_default_config_v2
from .utils import load_checkpoint, load_config, seed_everything


def _deep_update_dict(base: dict, update: dict) -> dict:
    for key, value in update.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            _deep_update_dict(base[key], value)
        else:
            base[key] = value
    return base


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Visualize MTKD feature alignment using PCA panels and cosine maps.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--config", type=str, required=True, help="Training config JSON/YAML path")
    p.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="MTKD checkpoint (.pth). Defaults to best_model.pth next to config/save_dir.",
    )
    p.add_argument("--split", type=str, default="val", choices=["train", "val"])
    p.add_argument("--num-samples", type=int, default=8, help="How many images to visualize")
    p.add_argument("--batch-size", type=int, default=1, help="Debug dataloader batch size")
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--output-dir", type=str, default=None, help="Directory to save panels and metrics")
    p.add_argument("--device", type=str, default=None, help="cuda / cpu; defaults to config or auto")
    p.add_argument("--tile-size", type=int, default=320, help="Panel tile size")
    return p.parse_args()


def _resolve_checkpoint(config_path: Path, config: Dict, explicit_path: Optional[str]) -> Path:
    candidates: List[Path] = []
    if explicit_path:
        candidates.append(Path(explicit_path))

    save_dir = config.get("output", {}).get("save_dir")
    if save_dir:
        candidates.append(Path(save_dir) / "best_model.pth")

    candidates.append(config_path.parent / "best_model.pth")

    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()

    searched = "\n".join(f"- {str(p)}" for p in candidates)
    raise FileNotFoundError(
        "Could not locate an MTKD checkpoint.\n"
        "Expected a .pth checkpoint, not the exported Ultralytics student .pt.\n"
        f"Searched:\n{searched}"
    )


def _build_merged_config(config_path: Path) -> Dict:
    config = get_default_config_v2()
    user_config = load_config(str(config_path))
    return _deep_update_dict(config, user_config)


def _build_loader(config: Dict, split: str, batch_size: int, num_workers: int):
    data_cfg = config.get("data", {})
    train_loader, val_loader = build_stomata_dataloaders(
        dataset_root=data_cfg.get("dataset_root", "Stomata_Dataset"),
        image_subdir=data_cfg.get("image_subdir", ""),
        label_subdir=data_cfg.get("label_subdir", ""),
        image_size=int(data_cfg.get("image_size", 640)),
        val_ratio=float(data_cfg.get("val_ratio", 0.1)),
        batch_size=batch_size,
        num_workers=num_workers,
        seed=int(config.get("seed", 42)),
        augmentation=bool(data_cfg.get("augmentation", True)),
    )
    return train_loader if split == "train" else val_loader


def _teacher_view(batch: Dict, align_easy_only: bool) -> Optional[torch.Tensor]:
    if align_easy_only and "images_weak" in batch:
        weak = batch["images_weak"]
        strong = batch["images"]
        if isinstance(weak, torch.Tensor) and weak.shape[0] == strong.shape[0]:
            return weak
    return None


def _fit_pca(tokens: torch.Tensor) -> torch.Tensor:
    x = tokens.float().cpu()
    if x.numel() == 0:
        return torch.zeros((0, 3), dtype=torch.float32)
    x = x - x.mean(dim=0, keepdim=True)
    q = int(max(1, min(3, x.shape[0], x.shape[1])))
    _, _, v = torch.pca_lowrank(x, q=q, center=False)
    proj = x @ v[:, :q]
    if q < 3:
        proj = torch.cat([proj, torch.zeros((proj.shape[0], 3 - q), dtype=proj.dtype)], dim=1)
    return proj[:, :3]


def _fit_joint_pca(tokens_a: torch.Tensor, tokens_b: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    a = tokens_a.float().cpu()
    b = tokens_b.float().cpu()
    both = torch.cat([a, b], dim=0)
    mean = both.mean(dim=0, keepdim=True)
    both = both - mean
    q = int(max(1, min(3, both.shape[0], both.shape[1])))
    _, _, v = torch.pca_lowrank(both, q=q, center=False)
    basis = v[:, :q]
    a_proj = (a - mean) @ basis
    b_proj = (b - mean) @ basis
    if q < 3:
        pad_a = torch.zeros((a_proj.shape[0], 3 - q), dtype=a_proj.dtype)
        pad_b = torch.zeros((b_proj.shape[0], 3 - q), dtype=b_proj.dtype)
        a_proj = torch.cat([a_proj, pad_a], dim=1)
        b_proj = torch.cat([b_proj, pad_b], dim=1)
    return a_proj[:, :3], b_proj[:, :3]


def _normalize_rgb(rgb: np.ndarray) -> np.ndarray:
    out = np.zeros_like(rgb, dtype=np.float32)
    for c in range(rgb.shape[-1]):
        channel = rgb[..., c]
        lo = float(channel.min())
        hi = float(channel.max())
        if hi <= lo:
            out[..., c] = 0.5
        else:
            out[..., c] = (channel - lo) / (hi - lo)
    return np.clip(out, 0.0, 1.0)


def _tokens_to_pca_image(pca_tokens: torch.Tensor, height: int, width: int) -> Image.Image:
    rgb = pca_tokens.reshape(height, width, 3).numpy()
    rgb = _normalize_rgb(rgb)
    return Image.fromarray((rgb * 255.0).astype(np.uint8), mode="RGB")


def _feature_to_tokens(feat: torch.Tensor) -> Tuple[torch.Tensor, int, int]:
    if feat.ndim != 3:
        raise ValueError(f"Expected [C, H, W] feature map, got {tuple(feat.shape)}")
    c, h, w = feat.shape
    tokens = feat.permute(1, 2, 0).reshape(h * w, c).contiguous()
    return tokens, h, w


def _heatmap_to_image(sim_map: np.ndarray) -> Image.Image:
    sim_map = sim_map.astype(np.float32)
    lo = float(np.percentile(sim_map, 5))
    hi = float(np.percentile(sim_map, 95))
    if hi <= lo:
        hi = lo + 1e-6
    norm = np.clip((sim_map - lo) / (hi - lo), 0.0, 1.0)

    try:
        from matplotlib import colormaps

        rgb = colormaps["viridis"](norm)[..., :3]
    except Exception:
        rgb = np.stack([norm, norm, norm], axis=-1)

    return Image.fromarray((rgb * 255.0).astype(np.uint8), mode="RGB")


def _tensor_image_to_pil(image_tensor: torch.Tensor) -> Image.Image:
    image = image_tensor.detach().cpu().float().clamp(0.0, 1.0).permute(1, 2, 0).numpy()
    return Image.fromarray((image * 255.0).astype(np.uint8), mode="RGB")


def _make_tile(image: Image.Image, title: str, tile_size: int) -> Image.Image:
    canvas_h = tile_size + 28
    canvas = Image.new("RGB", (tile_size, canvas_h), color=(255, 255, 255))
    img = image.resize((tile_size, tile_size), resample=Image.Resampling.BICUBIC)
    canvas.paste(img, (0, 0))
    draw = ImageDraw.Draw(canvas)
    draw.text((8, tile_size + 6), title, fill=(0, 0, 0))
    return canvas


def _build_panel(
    image: Image.Image,
    raw_student_pca: Image.Image,
    projected_pca: Image.Image,
    dino_pca: Image.Image,
    sim_image: Image.Image,
    tile_size: int,
) -> Image.Image:
    tiles = [
        _make_tile(image, "Image", tile_size),
        _make_tile(raw_student_pca, "Student PCA", tile_size),
        _make_tile(projected_pca, "Projected PCA", tile_size),
        _make_tile(dino_pca, "DINO PCA", tile_size),
        _make_tile(sim_image, "Cosine Map", tile_size),
    ]
    gap = 8
    width = sum(tile.width for tile in tiles) + gap * (len(tiles) - 1)
    height = max(tile.height for tile in tiles)
    panel = Image.new("RGB", (width, height), color=(245, 245, 245))
    x = 0
    for tile in tiles:
        panel.paste(tile, (x, 0))
        x += tile.width + gap
    return panel


def _save_metrics_csv(path: Path, rows: Sequence[Dict[str, object]]) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = _parse_args()
    config_path = Path(args.config).resolve()
    if not config_path.is_file():
        raise FileNotFoundError(f"Config not found: {config_path}")

    config = _build_merged_config(config_path)
    device_str = args.device or config.get("device") or ("cuda" if torch.cuda.is_available() else "cpu")
    if str(device_str).startswith("cuda") and not torch.cuda.is_available():
        device_str = "cpu"
    device = torch.device(device_str)
    config["device"] = str(device)

    seed_everything(int(config.get("seed", 42)))
    checkpoint_path = _resolve_checkpoint(config_path, config, args.checkpoint)

    output_dir = (
        Path(args.output_dir).resolve()
        if args.output_dir
        else checkpoint_path.parent / "feature_alignment_debug"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    loader = _build_loader(config, split=args.split, batch_size=args.batch_size, num_workers=args.num_workers)
    try:
        warmup_batch = next(iter(loader))
    except StopIteration as exc:
        raise RuntimeError(f"No samples available in {args.split} split") from exc

    model = build_mtkd_model_v2(config["model"]).to(device)
    align_easy_only = bool(config.get("training", {}).get("align_easy_only", False))

    warmup_images = warmup_batch["images"][:1].to(device)
    warmup_teacher = _teacher_view(warmup_batch, align_easy_only)
    if warmup_teacher is not None:
        warmup_teacher = warmup_teacher[:1].to(device)

    # Initialize the lazy alignment head before loading checkpoint weights.
    _ = model.get_alignment_debug(warmup_images, teacher_images=warmup_teacher)
    load_checkpoint(model, str(checkpoint_path), strict=False, map_location=str(device))
    model.to(device).eval()

    rows: List[Dict[str, object]] = []
    sample_count = 0

    for batch in loader:
        images = batch["images"].to(device)
        teacher_images = _teacher_view(batch, align_easy_only)
        if teacher_images is not None:
            teacher_images = teacher_images.to(device)

        debug = model.get_alignment_debug(images, teacher_images=teacher_images)
        raw_student = debug["student_spatial_feat"]
        projected = debug["projected_student_feat"]
        dino = debug["dino_features"]
        sim_map = debug["similarity_map"]
        align_loss_value = float(debug["align_loss"].item())

        batch_paths = batch.get("image_paths", [])
        for i in range(images.shape[0]):
            if sample_count >= args.num_samples:
                break

            image_path = str(batch_paths[i]) if i < len(batch_paths) else f"sample_{sample_count:04d}"
            image_stem = Path(image_path).stem

            raw_tokens, raw_h, raw_w = _feature_to_tokens(raw_student[i])
            raw_pca = _fit_pca(raw_tokens)
            raw_pca_img = _tokens_to_pca_image(raw_pca, raw_h, raw_w)

            proj_tokens, proj_h, proj_w = _feature_to_tokens(projected[i])
            dino_tokens, dino_h, dino_w = _feature_to_tokens(dino[i])
            if (proj_h, proj_w) != (dino_h, dino_w):
                raise RuntimeError(
                    f"Projected/DINO spatial mismatch: {(proj_h, proj_w)} vs {(dino_h, dino_w)}"
                )
            proj_pca, dino_pca = _fit_joint_pca(proj_tokens, dino_tokens)
            proj_pca_img = _tokens_to_pca_image(proj_pca, proj_h, proj_w)
            dino_pca_img = _tokens_to_pca_image(dino_pca, dino_h, dino_w)

            sim_np = sim_map[i].detach().cpu().float().numpy()
            sim_img = _heatmap_to_image(sim_np)
            input_pil = _tensor_image_to_pil(images[i].detach().cpu())

            panel = _build_panel(
                image=input_pil,
                raw_student_pca=raw_pca_img,
                projected_pca=proj_pca_img,
                dino_pca=dino_pca_img,
                sim_image=sim_img,
                tile_size=args.tile_size,
            )
            panel_path = output_dir / f"{sample_count:04d}_{image_stem}.png"
            panel.save(panel_path)

            row = {
                "index": sample_count,
                "image_path": image_path,
                "panel_path": str(panel_path),
                "align_loss": align_loss_value,
                "mean_cosine": float(sim_np.mean()),
                "median_cosine": float(np.median(sim_np)),
                "p10_cosine": float(np.percentile(sim_np, 10)),
                "p90_cosine": float(np.percentile(sim_np, 90)),
                "min_cosine": float(sim_np.min()),
                "max_cosine": float(sim_np.max()),
            }
            rows.append(row)
            sample_count += 1

        if sample_count >= args.num_samples:
            break

    if not rows:
        raise RuntimeError("No alignment samples were generated")

    _save_metrics_csv(output_dir / "metrics.csv", rows)

    summary = {
        "config": str(config_path),
        "checkpoint": str(checkpoint_path),
        "split": args.split,
        "num_samples": len(rows),
        "mean_align_loss": float(np.mean([float(r["align_loss"]) for r in rows])),
        "mean_cosine": float(np.mean([float(r["mean_cosine"]) for r in rows])),
        "median_cosine": float(np.median([float(r["median_cosine"]) for r in rows])),
        "mean_p10_cosine": float(np.mean([float(r["p10_cosine"]) for r in rows])),
        "mean_p90_cosine": float(np.mean([float(r["p90_cosine"]) for r in rows])),
        "output_dir": str(output_dir),
    }
    with (output_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
