#!/usr/bin/env python3
"""
FFT Periodicity Hypothesis Verification
========================================

在不訓練任何東西的情況下，驗證 DINO patch features 是否在氣孔圖片中
呈現行方向(row-wise)的週期性 — 這是 PluggableFFTBlock 的核心假設。

產出多張診斷圖 + 1 份量化報告：

1. raw_fft_power_spectrum.png  — 每張圖的 row-wise FFT power spectrum heatmap
2. pmr_per_row.png             — Peak-to-Mean Ratio bar chart（PMR > 3 = 有週期性）
3. grid_overlay.png            — FFT Block 生成的 periodic grid 疊加在原圖上
4. feature_pca.png             — Patch feature PCA 前 3 component 映射為 RGB
5. feature_pca_pc46.png        — Patch feature PCA 第 4~6 component 映射為 RGB
6. dominant_row_fft_examples.png
                               — 每張圖 PMR 最高 row 的空間訊號 + FFT 頻譜

用法：
------
python verify_fft_hypothesis.py \\
    --image_dir Stomata_Dataset/barley_all/images/train \\
    --weights "dinov3-main/weight folder/dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth" \\
    --output_dir outputs/fft_verification \\
    --num_images 8
"""

import argparse
import os
import sys
import random
from pathlib import Path
from typing import List, Tuple

import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
from torchvision import transforms

# ── Paths ──────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "dinov3-main"))


# ======================================================================
# Data loading
# ======================================================================

SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"}

IMG_TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

DENORM_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
DENORM_STD  = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


def load_images(
    image_dir: str,
    num_images: int = 8,
    seed: int = 42,
) -> Tuple[torch.Tensor, List[str], List[np.ndarray]]:
    """Load and preprocess images. Returns (batch, filenames, raw_rgb_list)."""
    data_dir = Path(image_dir)
    paths = []
    for ext in SUPPORTED_EXTENSIONS:
        paths.extend(data_dir.glob(f"*{ext}"))
        paths.extend(data_dir.glob(f"*{ext.upper()}"))

    if not paths:
        raise FileNotFoundError(f"No images found in {image_dir}")

    random.seed(seed)
    paths = sorted(paths)
    selected = random.sample(paths, min(num_images, len(paths)))

    tensors, names, raw_imgs = [], [], []
    for p in selected:
        img = Image.open(p).convert("RGB")
        raw_imgs.append(np.array(img.resize((224, 224))))  # for overlay
        tensors.append(IMG_TRANSFORM(img))
        names.append(p.name)

    batch = torch.stack(tensors)
    return batch, names, raw_imgs


# ======================================================================
# DINO feature extraction
# ======================================================================

def load_dino_model(weights_path: str, device: str) -> torch.nn.Module:
    """Load frozen DINOv3 ViT-B/16 with local weights."""
    from dinov3.models.vision_transformer import DinoVisionTransformer

    # Use the factory function through hub/backbones
    from dinov3.hub.backbones import dinov3_vitb16

    abs_weights = str(Path(weights_path).resolve())
    # Pass as file URI so _make_dinov3_vit can load local file
    file_uri = Path(abs_weights).as_uri()
    model = dinov3_vitb16(pretrained=True, weights=file_uri)

    model.eval()
    for p in model.parameters():
        p.requires_grad = False

    model.to(device)
    return model


def extract_patch_features(
    model: torch.nn.Module,
    images: torch.Tensor,
    device: str,
) -> torch.Tensor:
    """
    Extract spatial patch features from frozen DINO.

    Returns: [B, H, W, C]  where H=W=14 for 224px input with patch_size=16.
    """
    images = images.to(device)
    with torch.no_grad():
        # forward_features returns a dict with x_norm_patchtokens [B, H*W, C]
        out = model.forward_features(images)
        patches = out["x_norm_patchtokens"]  # [B, H*W, C]

    B, N_patches, C = patches.shape
    H = W = int(N_patches ** 0.5)
    assert H * W == N_patches, f"Non-square patch grid: {N_patches} patches"

    spatial = patches.reshape(B, H, W, C)
    return spatial


# ======================================================================
# Layer 1: Raw FFT analysis (no learnable parameters)
# ======================================================================

def compute_raw_fft(spatial: torch.Tensor) -> dict:
    """
    Compute row-wise FFT on DINO patch features.

    Args:
        spatial: [B, H, W, C]

    Returns:
        dict with power_spectrum [B, H, W//2+1], pmr [B, H], mean_pmr [B]
    """
    # Compute L2 norm per patch → scalar signal per row
    row_signal = spatial.norm(dim=-1)  # [B, H, W]

    # Remove DC (subtract row mean)
    row_signal = row_signal - row_signal.mean(dim=-1, keepdim=True)

    # Row-wise FFT along W axis
    fft_result = torch.fft.rfft(row_signal, dim=-1)  # [B, H, W//2+1]
    power = torch.abs(fft_result) ** 2  # [B, H, W//2+1]

    # Skip DC bin (index 0) for PMR calculation
    power_no_dc = power[:, :, 1:]  # [B, H, W//2]

    # Peak-to-Mean Ratio per row
    peak_power = power_no_dc.max(dim=-1).values     # [B, H]
    mean_power = power_no_dc.mean(dim=-1)            # [B, H]
    pmr = peak_power / (mean_power + 1e-8)           # [B, H]

    mean_pmr = pmr.mean(dim=-1)  # [B]

    return {
        "power_spectrum": power.cpu().numpy(),      # [B, H, W//2+1]
        "pmr": pmr.cpu().numpy(),                    # [B, H]
        "mean_pmr": mean_pmr.cpu().numpy(),          # [B]
        "row_signal": row_signal.cpu().numpy(),      # [B, H, W]
        "peak_freq_bin": power_no_dc.argmax(dim=-1).cpu().numpy() + 1,  # [B, H]
    }


# ======================================================================
# Layer 2: FFT Block grid generation (random init, zero-init gate)
# ======================================================================

def generate_fft_grid(
    model: torch.nn.Module,
    images: torch.Tensor,
    device: str,
) -> dict:
    """
    Inject a PluggableFFTBlock into DINO and run forward to get the
    generated periodic grid.
    """
    from mtkd_framework.engine.pluggable_fft_block import (
        inject_fft_blocks,
        PluggableFFTBlock,
    )

    import copy
    model_copy = copy.deepcopy(model)

    # Inject FFT block after block 10
    fft_blocks = inject_fft_blocks(
        model_copy,
        after_blocks=[10],
        embed_dim=768,
        # auto-detect n_storage_tokens from model
    )
    fft_block = fft_blocks[0]

    model_copy.eval()
    images = images.to(device)

    with torch.no_grad():
        _ = model_copy(images)

    cache = fft_block.get_cache()

    return {
        "grid": cache["grid"].cpu().numpy(),              # [B, 1, H, W]
        "gate_value": cache["gate_value"].item(),         # scalar
        "dominant_freq": cache["freq_info"]["dominant_freq"].cpu().numpy(),     # [B, H]
        "freq_confidence": cache["freq_info"]["freq_confidence"].cpu().numpy(), # [B, H]
        "freq_spectrum": cache["freq_info"]["freq_spectrum"].cpu().numpy(),     # [B, H, bins]
    }


# ======================================================================
# Feature PCA
# ======================================================================

def compute_feature_pca(
    spatial: torch.Tensor,
    num_components: int = 6,
) -> np.ndarray:
    """
    PCA of patch features.

    Args:
        spatial: [B, H, W, C]

    Returns:
        pca_components: [B, H, W, num_components] in [0, 1]
    """
    if num_components < 3:
        raise ValueError("num_components must be >= 3 for RGB visualization.")

    B, H, W, C = spatial.shape
    flat = spatial.reshape(B * H * W, C).cpu().float()

    # Center
    mean = flat.mean(dim=0)
    centered = flat - mean

    # SVD for PCA
    U, S, Vt = torch.linalg.svd(centered, full_matrices=False)
    # Project onto first N components
    n_comp = min(num_components, Vt.shape[0])
    proj = centered @ Vt[:n_comp].T  # [B*H*W, n_comp]

    # Normalize each component to [0, 1]
    proj = proj.reshape(B, H, W, n_comp)
    for c in range(n_comp):
        cmin = proj[..., c].min()
        cmax = proj[..., c].max()
        proj[..., c] = (proj[..., c] - cmin) / (cmax - cmin + 1e-8)

    return proj.numpy()


# ======================================================================
# Visualization
# ======================================================================

def plot_raw_fft_spectrum(
    fft_data: dict,
    names: List[str],
    output_path: str,
):
    """Fig 1: Row-wise FFT power spectrum heatmaps."""
    import matplotlib.pyplot as plt

    power = fft_data["power_spectrum"]  # [B, H, freq_bins]
    B = power.shape[0]
    cols = min(4, B)
    rows = (B + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 3.5 * rows))
    if rows == 1 and cols == 1:
        axes = np.array([[axes]])
    elif rows == 1:
        axes = axes[np.newaxis, :]
    elif cols == 1:
        axes = axes[:, np.newaxis]

    for i in range(B):
        r, c = divmod(i, cols)
        # Skip DC bin for better contrast
        spec = power[i, :, 1:]  # [H, freq_bins-1]
        im = axes[r, c].imshow(
            spec, aspect="auto", cmap="hot", interpolation="nearest",
        )
        axes[r, c].set_title(names[i][:20], fontsize=9)
        axes[r, c].set_xlabel("Freq bin (excl. DC)")
        axes[r, c].set_ylabel("Row index")
        plt.colorbar(im, ax=axes[r, c], fraction=0.046, pad=0.04)

    # Hide unused axes
    for i in range(B, rows * cols):
        r, c = divmod(i, cols)
        axes[r, c].axis("off")

    fig.suptitle("Raw FFT Power Spectrum (DINO patch features, row-wise)", fontsize=13)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {output_path}")


def plot_pmr(
    fft_data: dict,
    names: List[str],
    output_path: str,
):
    """Fig 2: Peak-to-Mean Ratio per row."""
    import matplotlib.pyplot as plt

    pmr = fft_data["pmr"]  # [B, H]
    B, H = pmr.shape
    cols = min(4, B)
    rows = (B + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 3.5 * rows))
    if rows == 1 and cols == 1:
        axes = np.array([[axes]])
    elif rows == 1:
        axes = axes[np.newaxis, :]
    elif cols == 1:
        axes = axes[:, np.newaxis]

    for i in range(B):
        r, c = divmod(i, cols)
        row_pmr = pmr[i]
        colors = ["#e74c3c" if v > 3 else "#3498db" for v in row_pmr]
        axes[r, c].barh(range(H), row_pmr, color=colors)
        axes[r, c].axvline(x=3, color="black", linestyle="--", linewidth=1, label="PMR=3")
        axes[r, c].set_title(f"{names[i][:18]} (mean={row_pmr.mean():.1f})", fontsize=9)
        axes[r, c].set_xlabel("PMR")
        axes[r, c].set_ylabel("Row")
        axes[r, c].invert_yaxis()
        axes[r, c].legend(fontsize=7)

    for i in range(B, rows * cols):
        r, c = divmod(i, cols)
        axes[r, c].axis("off")

    fig.suptitle(
        "Peak-to-Mean Ratio per Row (red = PMR > 3 → periodic)",
        fontsize=13,
    )
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {output_path}")


def plot_grid_overlay(
    grid_data: dict,
    raw_imgs: List[np.ndarray],
    names: List[str],
    output_path: str,
):
    """Fig 3: Periodic grid overlaid on original images."""
    import matplotlib.pyplot as plt

    grids = grid_data["grid"]  # [B, 1, H, W]
    B = grids.shape[0]
    cols = min(4, B)
    rows = (B + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols * 2, figsize=(3.5 * cols * 2, 3.5 * rows))
    if rows == 1:
        axes = axes[np.newaxis, :] if cols * 2 > 1 else np.array([[axes]])

    for i in range(B):
        r = i // cols
        c = (i % cols) * 2

        # Original image
        axes[r, c].imshow(raw_imgs[i])
        axes[r, c].set_title(names[i][:18], fontsize=8)
        axes[r, c].axis("off")

        # Grid upsampled to image size + overlay
        grid_i = grids[i, 0]  # [H, W] e.g. [14, 14]
        # Upsample grid to 224×224
        grid_up = torch.from_numpy(grid_i).unsqueeze(0).unsqueeze(0).float()
        grid_up = F.interpolate(grid_up, size=(224, 224), mode="bilinear", align_corners=False)
        grid_up = grid_up.squeeze().numpy()

        axes[r, c + 1].imshow(raw_imgs[i])
        axes[r, c + 1].imshow(grid_up, cmap="hot", alpha=0.5, vmin=0, vmax=1)
        axes[r, c + 1].set_title(f"Grid overlay (gate={grid_data['gate_value']:.4f})", fontsize=8)
        axes[r, c + 1].axis("off")

    # Hide unused
    for i in range(B, rows * cols):
        r = i // cols
        c = (i % cols) * 2
        axes[r, c].axis("off")
        axes[r, c + 1].axis("off")

    fig.suptitle("Periodic Grid Overlay (PluggableFFTBlock, untrained)", fontsize=13)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {output_path}")


def plot_feature_pca(
    pca_components: np.ndarray,
    raw_imgs: List[np.ndarray],
    names: List[str],
    output_path: str,
    component_indices: Tuple[int, int, int] = (0, 1, 2),
):
    """Visualize selected 3 PCA components as RGB."""
    import matplotlib.pyplot as plt

    B = pca_components.shape[0]
    n_comp = pca_components.shape[-1]
    if max(component_indices) >= n_comp:
        raise ValueError(
            f"Requested components {component_indices} but only {n_comp} are available."
        )

    c1, c2, c3 = component_indices
    cols = min(4, B)
    rows = (B + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols * 2, figsize=(3.5 * cols * 2, 3.5 * rows))
    if rows == 1:
        axes = axes[np.newaxis, :] if cols * 2 > 1 else np.array([[axes]])

    for i in range(B):
        r = i // cols
        c = (i % cols) * 2

        axes[r, c].imshow(raw_imgs[i])
        axes[r, c].set_title(names[i][:18], fontsize=8)
        axes[r, c].axis("off")

        # Pick requested components and upsample from [H, W, 3] to [224, 224, 3]
        pca_i_np = np.stack(
            [
                pca_components[i, :, :, c1],
                pca_components[i, :, :, c2],
                pca_components[i, :, :, c3],
            ],
            axis=-1,
        )
        pca_i = torch.from_numpy(pca_i_np).permute(2, 0, 1).unsqueeze(0).float()
        pca_up = F.interpolate(pca_i, size=(224, 224), mode="bilinear", align_corners=False)
        pca_up = pca_up.squeeze().permute(1, 2, 0).numpy()

        axes[r, c + 1].imshow(pca_up)
        axes[r, c + 1].set_title(
            f"PCA (PC{c1 + 1}→R, PC{c2 + 1}→G, PC{c3 + 1}→B)", fontsize=8
        )
        axes[r, c + 1].axis("off")

    for i in range(B, rows * cols):
        r = i // cols
        c = (i % cols) * 2
        axes[r, c].axis("off")
        axes[r, c + 1].axis("off")

    fig.suptitle("DINO Patch Feature PCA (spatial structure visualization)", fontsize=13)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {output_path}")


def plot_dominant_row_fft_examples(
    fft_data: dict,
    names: List[str],
    output_path: str,
):
    """
    For each image, plot:
    - strongest periodic row signal (after DC removal)
    - its FFT power spectrum
    """
    import matplotlib.pyplot as plt

    pmr = fft_data["pmr"]                # [B, H]
    row_signal = fft_data["row_signal"]  # [B, H, W]
    power = fft_data["power_spectrum"]   # [B, H, W//2+1]

    B = pmr.shape[0]
    fig, axes = plt.subplots(B, 2, figsize=(12, max(3, 2.6 * B)))
    if B == 1:
        axes = np.array([axes])

    for i in range(B):
        best_row = int(np.argmax(pmr[i]))
        signal = row_signal[i, best_row]      # [W]
        spec = power[i, best_row, 1:]         # skip DC
        peak_bin = int(np.argmax(spec) + 1)   # convert back to full bin index

        ax_l, ax_r = axes[i]
        ax_l.plot(signal, color="#2c7fb8", linewidth=1.6)
        ax_l.set_title(
            f"{names[i][:22]} | row={best_row}, PMR={pmr[i, best_row]:.2f}",
            fontsize=9,
        )
        ax_l.set_xlabel("Patch column")
        ax_l.set_ylabel("Signal")
        ax_l.grid(alpha=0.25)

        bins = np.arange(1, len(spec) + 1)
        ax_r.plot(bins, spec, color="#d95f0e", linewidth=1.8)
        ax_r.axvline(peak_bin, linestyle="--", color="black", linewidth=1.0)
        ax_r.set_title(f"FFT power (peak bin={peak_bin})", fontsize=9)
        ax_r.set_xlabel("Freq bin (excl. DC)")
        ax_r.set_ylabel("Power")
        ax_r.grid(alpha=0.25)

    fig.suptitle("Dominant Row Periodicity Examples (per-image strongest row)", fontsize=13)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {output_path}")


# ======================================================================
# Quantitative report
# ======================================================================

def write_report(
    fft_data: dict,
    grid_data: dict,
    names: List[str],
    output_path: str,
):
    """Write a text report with quantitative analysis."""
    pmr = fft_data["pmr"]  # [B, H]
    B, H = pmr.shape

    lines = []
    lines.append("=" * 60)
    lines.append("FFT Periodicity Hypothesis — Verification Report")
    lines.append("=" * 60)
    lines.append("")

    # Per-image stats
    lines.append("Per-image statistics:")
    lines.append("-" * 60)
    lines.append(f"{'Image':<25} {'Mean PMR':>10} {'Med PMR':>10} {'Rows>3':>10} {'Max PMR':>10}")
    lines.append("-" * 60)

    all_pmr_flat = pmr.flatten()

    for i in range(B):
        row_pmr = pmr[i]
        n_periodic = (row_pmr > 3).sum()
        lines.append(
            f"{names[i][:25]:<25} "
            f"{row_pmr.mean():>10.2f} "
            f"{np.median(row_pmr):>10.2f} "
            f"{n_periodic:>8d}/{H:<1d} "
            f"{row_pmr.max():>10.2f}"
        )

    lines.append("")
    lines.append("=" * 60)
    lines.append("Global statistics:")
    lines.append(f"  Total rows analyzed: {B * H}")
    lines.append(f"  Rows with PMR > 3:   {(all_pmr_flat > 3).sum()} ({(all_pmr_flat > 3).mean() * 100:.1f}%)")
    lines.append(f"  Rows with PMR > 5:   {(all_pmr_flat > 5).sum()} ({(all_pmr_flat > 5).mean() * 100:.1f}%)")
    lines.append(f"  Global mean PMR:     {all_pmr_flat.mean():.2f}")
    lines.append(f"  Global median PMR:   {np.median(all_pmr_flat):.2f}")
    lines.append("")

    # FFT Block info
    lines.append("FFT Block (untrained, zero-init gate):")
    lines.append(f"  Gate value (sigmoid): {grid_data['gate_value']:.6f}")
    lines.append(f"  Mean dominant freq:   {grid_data['dominant_freq'].mean():.4f}")
    lines.append(f"  Mean freq confidence: {grid_data['freq_confidence'].mean():.4f}")
    lines.append("")

    # Conclusion
    pct_periodic = (all_pmr_flat > 3).mean() * 100
    lines.append("=" * 60)
    lines.append("CONCLUSION:")
    if pct_periodic > 30:
        lines.append(f"  ✅ {pct_periodic:.1f}% of rows show PMR > 3")
        lines.append("  → Row-wise periodicity EXISTS in DINO features.")
        lines.append("  → PluggableFFTBlock hypothesis is SUPPORTED.")
        lines.append("  → Proceed with FFT block in MTKD training.")
    elif pct_periodic > 15:
        lines.append(f"  ⚠️  {pct_periodic:.1f}% of rows show PMR > 3")
        lines.append("  → Weak periodicity signal detected.")
        lines.append("  → FFT block may help but effect could be marginal.")
        lines.append("  → Consider testing with and without FFT block.")
    else:
        lines.append(f"  ❌ Only {pct_periodic:.1f}% of rows show PMR > 3")
        lines.append("  → No clear row-wise periodicity in DINO features.")
        lines.append("  → PluggableFFTBlock hypothesis is NOT supported.")
        lines.append("  → FFT block is unlikely to help; consider removing it.")
    lines.append("=" * 60)

    report = "\n".join(lines)
    print("\n" + report)

    with open(output_path, "w") as f:
        f.write(report)
    print(f"\n  Report saved: {output_path}")


# ======================================================================
# Main
# ======================================================================

def parse_args():
    p = argparse.ArgumentParser(description="FFT Periodicity Hypothesis Verification")
    p.add_argument("--image_dir", type=str, required=True,
                   help="Path to stomata images directory")
    p.add_argument("--weights", type=str, required=True,
                   help="Path to DINOv3 ViT-B/16 pretrained .pth file")
    p.add_argument("--output_dir", type=str, default="outputs/fft_verification",
                   help="Output directory for plots and report")
    p.add_argument("--num_images", type=int, default=8,
                   help="Number of images to analyze")
    p.add_argument("--seed", type=int, default=42,
                   help="Random seed for image sampling")
    p.add_argument("--device", type=str, default=None,
                   help="Device (auto-detected if omitted)")
    return p.parse_args()


def main():
    args = parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Step 1: Load images ──────────────────────────────────────────
    print("\n[1/5] Loading images...")
    images, names, raw_imgs = load_images(args.image_dir, args.num_images, args.seed)
    print(f"  Loaded {len(names)} images from {args.image_dir}")

    # ── Step 2: Load DINO & extract features ─────────────────────────
    print("\n[2/5] Loading DINOv3 ViT-B/16...")
    model = load_dino_model(args.weights, device)
    n_storage = getattr(model, "n_storage_tokens", "?")
    print(f"  Model loaded (n_storage_tokens={n_storage})")

    print("  Extracting patch features...")
    spatial = extract_patch_features(model, images, device)
    B, H, W, C = spatial.shape
    print(f"  Feature shape: [{B}, {H}, {W}, {C}]")

    # ── Step 3: Raw FFT analysis ─────────────────────────────────────
    print("\n[3/5] Computing raw FFT...")
    fft_data = compute_raw_fft(spatial)
    mean_pmr_all = fft_data["mean_pmr"].mean()
    print(f"  Global mean PMR: {mean_pmr_all:.2f}")
    print(f"  Rows with PMR > 3: {(fft_data['pmr'].flatten() > 3).sum()}/{B * H}")

    # ── Step 4: FFT Block grid generation ────────────────────────────
    print("\n[4/5] Generating FFT Block grid (untrained)...")
    grid_data = generate_fft_grid(model, images, device)
    print(f"  Gate value: {grid_data['gate_value']:.6f}")
    print(f"  Grid shape: {grid_data['grid'].shape}")

    # ── Step 5: Feature PCA ──────────────────────────────────────────
    print("\n[5/5] Computing feature PCA...")
    pca_components = compute_feature_pca(spatial, num_components=6)
    print(f"  PCA shape: {pca_components.shape}")

    # ── Generate all visualizations ──────────────────────────────────
    print("\nGenerating visualizations...")

    plot_raw_fft_spectrum(
        fft_data, names,
        str(out_dir / "raw_fft_power_spectrum.png"),
    )
    plot_pmr(
        fft_data, names,
        str(out_dir / "pmr_per_row.png"),
    )
    plot_grid_overlay(
        grid_data, raw_imgs, names,
        str(out_dir / "grid_overlay.png"),
    )
    plot_feature_pca(
        pca_components, raw_imgs, names,
        str(out_dir / "feature_pca.png"),
        component_indices=(0, 1, 2),
    )
    plot_feature_pca(
        pca_components, raw_imgs, names,
        str(out_dir / "feature_pca_pc46.png"),
        component_indices=(3, 4, 5),
    )
    plot_dominant_row_fft_examples(
        fft_data, names,
        str(out_dir / "dominant_row_fft_examples.png"),
    )

    # ── Report ───────────────────────────────────────────────────────
    write_report(
        fft_data, grid_data, names,
        str(out_dir / "report.txt"),
    )

    print(f"\n✅ Done! All outputs in: {out_dir}")


if __name__ == "__main__":
    main()
