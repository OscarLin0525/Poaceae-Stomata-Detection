import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import pywt
import torch
from PIL import Image, ImageDraw
from sklearn.decomposition import PCA
from skimage import exposure
from torchvision.transforms import v2
import matplotlib.pyplot as plt

from dino_zero_init_bypass import inject_zero_init_bypass

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


def parse_args():
    p = argparse.ArgumentParser(
        description="Wavelet Amplifier + DINOv3 patch-token extraction"
    )
    # ===== Core Mode =====
    p.add_argument("--mode", type=str, default="build", choices=["build", "visualize"], help="build: export enhanced images; visualize: PCA debug panels")
    p.add_argument("--input-dir", type=str, required=True, help="Input image folder")
    p.add_argument("--output-dir", type=str, required=True, help="Output folder for enhanced images (build) or visualizations (visualize)")

    # ===== Wavelet Enhancement Parameters =====
    p.add_argument("--wavelet", type=str, default="sym4", help="Wavelet basis (sym4/db4/bior2.2)")
    p.add_argument("--amp-lh", type=float, default=4.0, help="Amplification for LH (horizontal) band")
    p.add_argument("--amp-hl", type=float, default=4.0, help="Amplification for HL (vertical) band")
    p.add_argument("--amp-hh", type=float, default=0.0, help="Amplification for HH (diagonal) band. Set 0 to disable.")
    p.add_argument("--ll-scale", type=float, default=1.0, help="LL band scale. <1.0 attenuates low-freq background.")
    p.add_argument("--gamma", type=float, default=0.9, help="Non-linearity exponent in signed-power amplification")
    p.add_argument("--clip-percentile", type=float, default=99.5, help="Soft clip percentile. ≤0 to disable.")
    p.add_argument("--illumination", type=str, default="none", choices=["none", "clahe", "grayworld"], help="Pre-wavelet illumination correction")
    p.add_argument(
        "--color-mode",
        type=str,
        default="luma",
        choices=["luma", "rgb"],
        help="Wavelet output mode: luma uses Y channel only, rgb applies wavelet to each RGB channel.",
    )
    p.add_argument(
        "--random-gray-prob",
        type=float,
        default=0.0,
        help="Probability to convert input into grayscale (replicated to RGB) before wavelet. Use 0.3~0.6 for mixed RGB/gray training.",
    )
    p.add_argument(
        "--random-seed",
        type=int,
        default=42,
        help="Random seed for grayscale sampling.",
    )
    p.add_argument(
        "--dual-view-consistency",
        action="store_true",
        help="Also export paired RGB and Luma wavelet views for consistency training.",
    )
    p.add_argument(
        "--dual-view-dir",
        type=str,
        default=None,
        help="Output directory for dual-view pairs. Defaults to <output-dir>_dual.",
    )

    # ===== DINO Model Configuration =====
    p.add_argument("--repo-dir", type=str, default="/home/oscar/Poaceae-Stomata-Detection/dinov3-main", help="DINOv3 repo directory")
    p.add_argument("--weights", type=str, required=True, help="Path to DINOv3 .pth checkpoint")
    p.add_argument("--model-name", type=str, default="dinov3_vitb16", help="DINO model name from hub")
    p.add_argument("--device", type=str, default="cuda", help="cuda or cpu")
    p.add_argument("--image-size", type=int, default=518, help="Input image size for DINO forward pass")
    p.add_argument("--enable-bypass", action="store_true", help="Inject stomata-aware adaptive bypass into a DINO block.")
    p.add_argument("--bypass-block-index", type=int, default=6, help="Target DINO block index for bypass injection.")
    p.add_argument("--bypass-bottleneck", type=int, default=64, help="Bypass bottleneck dimension.")
    p.add_argument("--bypass-alpha-init", type=float, default=0.0, help="Initial bypass residual gate alpha.")
    p.add_argument(
        "--bypass-row-min-instances",
        type=float,
        default=4.0,
        help="Suppress rows with expected stomata count below this value (set 0 to disable).",
    )
    p.add_argument(
        "--bypass-row-gate-temperature",
        type=float,
        default=8.0,
        help="Sharpness of row-consistency suppression gate.",
    )
    p.add_argument(
        "--bypass-row-axis",
        type=str,
        default="horizontal",
        choices=["horizontal", "vertical"],
        help="Direction for row gating: horizontal counts across rows, vertical across columns.",
    )
    p.add_argument(
        "--bypass-checkpoint",
        type=str,
        default=None,
        help="Path to trained bypass .pt checkpoint (loads bypass_state_dict).",
    )

    # ===== Optional Diagnostic/Advanced Parameters =====
    p.add_argument("--stomata-size-px", type=float, default=None, help="Approx stomata diameter (pixels) for patch-size diagnostic")
    p.add_argument("--tile-size", type=int, default=None, help="Optional tile crop size. If set, split each image into tiles.")
    p.add_argument("--tile-stride", type=int, default=None, help="Tile stride. Defaults to tile-size if omitted.")
    p.add_argument("--row-prior-strength", type=float, default=0.0, help="Row prior modulation strength. 0=disabled.")
    p.add_argument(
        "--compare-bypass",
        action="store_true",
        help="Visualize base DINO vs bypass DINO side-by-side (visualize mode, requires --enable-bypass).",
    )
    p.add_argument(
        "--save-feature-npy",
        action="store_true",
        help="Save extracted patch-token feature maps as .npy files in visualize mode.",
    )
    p.add_argument(
        "--compare-input",
        type=str,
        default="original",
        choices=["original", "amplified"],
        help="Input used for base-vs-bypass token comparison in visualize mode.",
    )
    p.add_argument(
        "--bypass-alpha-override",
        type=float,
        default=None,
        help="Override bypass alpha at inference time for diagnostics (e.g., 0.3).",
    )
    p.add_argument("--show", action="store_true", help="Show visualizations interactively (visualize mode only)")
    p.add_argument("--invert-pca", action="store_true", help="Invert PCA color map (visualize mode only)")
    
    return p.parse_args()


def build_transform(image_size=518):
    return v2.Compose(
        [
            v2.ToImage(),
            v2.Resize((image_size, image_size), antialias=True),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ]
    )


def normalize_01(x):
    x = x.astype(np.float32)
    mn, mx = float(x.min()), float(x.max())
    if mx <= mn:
        return np.zeros_like(x, dtype=np.float32)
    return (x - mn) / (mx - mn)


def _apply_illumination(image, mode):
    if mode == "none":
        return image

    ycbcr = image.convert("YCbCr")
    arr = np.asarray(ycbcr).astype(np.float32)
    y = arr[:, :, 0] / 255.0

    if mode == "clahe":
        y2 = exposure.equalize_adapthist(y, clip_limit=0.01)
    elif mode == "grayworld":
        rgb = np.asarray(image).astype(np.float32)
        means = rgb.reshape(-1, 3).mean(axis=0) + 1e-6
        target = float(means.mean())
        rgb = rgb * (target / means)[None, None, :]
        rgb = np.clip(rgb, 0, 255).astype(np.uint8)
        return Image.fromarray(rgb, mode="RGB")
    else:
        y2 = y

    arr[:, :, 0] = np.clip(y2 * 255.0, 0.0, 255.0)
    return Image.fromarray(arr.astype(np.uint8), mode="YCbCr").convert("RGB")


def signed_power(x, gamma):
    return np.sign(x) * (np.abs(x) ** gamma)


def amplify_channel(ch, wavelet, amp_lh, amp_hl, amp_hh, gamma, ll_scale=1.0):
    """2D DWT + amplification + IDWT for a single channel."""
    ll, (lh, hl, hh) = pywt.dwt2(ch, wavelet)
    ll = ll * float(ll_scale)
    lh_amp = signed_power(lh, gamma) * amp_lh
    hl_amp = signed_power(hl, gamma) * amp_hl
    hh_amp = signed_power(hh, gamma) * amp_hh
    rec = pywt.idwt2((ll, (lh_amp, hl_amp, hh_amp)), wavelet)
    return rec


def soft_clip_and_scale(rgb, clip_percentile):
    out = np.zeros_like(rgb, dtype=np.float32)
    for c in range(3):
        ch = rgb[:, :, c]
        if clip_percentile > 0:
            hi = np.percentile(ch, clip_percentile)
            lo = np.percentile(ch, 100.0 - clip_percentile)
            if hi <= lo:
                out[:, :, c] = normalize_01(ch)
                continue
            ch = np.clip(ch, lo, hi)
        out[:, :, c] = normalize_01(ch)
    return np.clip(out, 0.0, 1.0)


def wavelet_amplify_luma(
    image,
    wavelet="sym4",
    amp_lh=4.0,
    amp_hl=4.0,
    amp_hh=0.0,
    gamma=0.9,
    clip_percentile=99.5,
    ll_scale=1.0,
    luma_rgb3=True,
):
    """Extract Y channel, apply wavelet enhancement, return luma-based RGB3 image."""
    ycbcr = image.convert("YCbCr")
    ycbcr_arr = np.asarray(ycbcr).astype(np.float32)

    y = ycbcr_arr[:, :, 0] / 255.0
    y_amp = amplify_channel(y, wavelet, amp_lh, amp_hl, amp_hh, gamma, ll_scale=ll_scale)
    y_amp = y_amp[: y.shape[0], : y.shape[1]]

    if clip_percentile > 0:
        hi = np.percentile(y_amp, clip_percentile)
        lo = np.percentile(y_amp, 100.0 - clip_percentile)
        if hi > lo:
            y_amp = np.clip(y_amp, lo, hi)

    y_amp = normalize_01(y_amp)
    
    # Always output 3-channel RGB so DINO sees a valid natural-image-shaped tensor.
    y8 = np.clip(y_amp * 255.0, 0.0, 255.0).astype(np.uint8)
    rgb = np.stack([y8, y8, y8], axis=-1)
    return Image.fromarray(rgb, mode="RGB")


def wavelet_amplify_rgb(
    image,
    wavelet="sym4",
    amp_lh=4.0,
    amp_hl=4.0,
    amp_hh=0.0,
    gamma=0.9,
    clip_percentile=99.5,
    ll_scale=1.0,
):
    """Apply the same wavelet enhancement independently on R, G, and B channels."""
    rgb = np.asarray(image).astype(np.float32) / 255.0
    out = np.zeros_like(rgb, dtype=np.float32)

    for c in range(3):
        ch = rgb[:, :, c]
        ch_amp = amplify_channel(ch, wavelet, amp_lh, amp_hl, amp_hh, gamma, ll_scale=ll_scale)
        ch_amp = ch_amp[: ch.shape[0], : ch.shape[1]]

        if clip_percentile > 0:
            hi = np.percentile(ch_amp, clip_percentile)
            lo = np.percentile(ch_amp, 100.0 - clip_percentile)
            if hi > lo:
                ch_amp = np.clip(ch_amp, lo, hi)

        out[:, :, c] = normalize_01(ch_amp)

    out = np.clip(out * 255.0, 0.0, 255.0).astype(np.uint8)
    return Image.fromarray(out, mode="RGB")


def wavelet_amplify(image, color_mode="luma", **kwargs):
    if color_mode == "rgb":
        return wavelet_amplify_rgb(image, **kwargs)
    return wavelet_amplify_luma(image, **kwargs)


def maybe_random_grayscale(image, prob, rng):
    """Stochastically convert image to grayscale and replicate into RGB."""
    if prob <= 0.0:
        return image, False
    if rng.random() >= prob:
        return image, False
    gray = image.convert("L")
    gray_rgb = Image.merge("RGB", (gray, gray, gray))
    return gray_rgb, True


def print_patch_diagnostics(model, image_size, stomata_size_px=None):
    patch = int(getattr(model, "patch_size", 16))
    grid = image_size // patch
    print(f"[diag] DINO patch_size={patch}, resized={image_size}, token_grid={grid}x{grid}")

    if stomata_size_px is not None and stomata_size_px > 0:
        ratio = stomata_size_px / float(patch)
        print(f"[diag] approx stomata_size_px={stomata_size_px:.2f}, size_vs_patch={ratio:.2f}")
        if ratio < 0.5:
            print("[warn] stomata likely too small per patch. Consider tiling or a model with smaller patch size.")
        elif ratio < 1.0:
            print("[warn] stomata near/below one patch. Features may be diluted.")


def generate_tiles(image, tile_size, tile_stride=None):
    w, h = image.size
    if tile_size is None or tile_size <= 0:
        return [("full", image)]

    stride = tile_stride if (tile_stride is not None and tile_stride > 0) else tile_size
    tiles = []
    y_positions = list(range(0, max(1, h - tile_size + 1), stride))
    x_positions = list(range(0, max(1, w - tile_size + 1), stride))

    if y_positions[-1] != max(0, h - tile_size):
        y_positions.append(max(0, h - tile_size))
    if x_positions[-1] != max(0, w - tile_size):
        x_positions.append(max(0, w - tile_size))

    for y in y_positions:
        for x in x_positions:
            x2 = min(x + tile_size, w)
            y2 = min(y + tile_size, h)
            crop = image.crop((x, y, x2, y2))
            tiles.append((f"tile_x{x}_y{y}", crop))
    return tiles


def apply_row_prior(amplified_image, spacing_px, strength=0.0):
    """Apply horizontal row prior modulation if spacing is known and strength > 0."""
    if spacing_px is None or spacing_px <= 2 or strength <= 0:
        return amplified_image

    arr = np.asarray(amplified_image.convert("YCbCr")).astype(np.float32)
    y = arr[:, :, 0] / 255.0
    h, w = y.shape

    # Estimate phase from vertical gradient energy
    e = np.abs(np.gradient(y, axis=0)).mean(axis=1)
    yy = np.arange(h, dtype=np.float32)
    phases = np.linspace(0.0, 2.0 * np.pi, 36, endpoint=False)
    best_phase = 0.0
    best_score = -1e18

    for ph in phases:
        tpl = (0.5 + 0.5 * np.cos((2.0 * np.pi * yy / spacing_px) + ph))
        score = float((tpl * e).sum())
        if score > best_score:
            best_score = score
            best_phase = ph

    prior = 0.5 + 0.5 * np.cos((2.0 * np.pi * yy / spacing_px) + best_phase)
    prior = np.tile(prior[:, None], (1, w))

    mod = 1.0 + strength * (2.0 * prior - 1.0)
    y2 = np.clip(y * mod, 0.0, 1.0)

    arr[:, :, 0] = y2 * 255.0
    out = Image.fromarray(arr.astype(np.uint8), mode="YCbCr").convert("RGB")
    return out


def normalize_pca_image(pca_image):
    pca_image_normalized = np.zeros_like(pca_image, dtype=np.float32)
    for i in range(3):
        pca_image_normalized[:, :, i] = normalize_01(pca_image[:, :, i])
    return pca_image_normalized


def _fit_pair_pca(tokens_a, tokens_b):
    both = np.concatenate([tokens_a, tokens_b], axis=0)
    pca = PCA(n_components=3)
    pca.fit(both)
    return pca.transform(tokens_a), pca.transform(tokens_b)


def _pca_to_image(feat_pca, h, w, invert=False):
    pca_img = feat_pca.reshape((h, w, 3)).astype(np.float32)
    pca_img = normalize_pca_image(pca_img)
    if invert:
        pca_img = 1.0 - pca_img
    return pca_img


def build_bypass_comparison_visualization(
    original_image,
    amplified_image,
    base_tokens,
    bypass_tokens,
    h,
    w,
    invert_pca=False,
):
    base_pca, bypass_pca = _fit_pair_pca(base_tokens, bypass_tokens)

    base_pca_img = _pca_to_image(base_pca, h, w, invert=invert_pca)
    bypass_pca_img = _pca_to_image(bypass_pca, h, w, invert=invert_pca)

    base_pc1n = normalize_01(base_pca[:, 0].reshape((h, w)).astype(np.float32))
    bypass_pc1n = normalize_01(bypass_pca[:, 0].reshape((h, w)).astype(np.float32))
    delta = bypass_pc1n - base_pc1n
    delta_abs = np.abs(delta)

    base_pca_pil = Image.fromarray((base_pca_img * 255).astype(np.uint8), mode="RGB")
    bypass_pca_pil = Image.fromarray((bypass_pca_img * 255).astype(np.uint8), mode="RGB")

    base_pca_big = base_pca_pil.resize(original_image.size, Image.Resampling.BICUBIC)
    bypass_pca_big = bypass_pca_pil.resize(original_image.size, Image.Resampling.BICUBIC)
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))

    axes[0, 0].imshow(original_image)
    axes[0, 0].set_title("Original")
    axes[0, 0].axis("off")

    axes[0, 1].imshow(amplified_image)
    axes[0, 1].set_title("Wavelet Amplified")
    axes[0, 1].axis("off")

    axes[0, 2].imshow(base_pca_big)
    axes[0, 2].set_title("Base DINO PCA")
    axes[0, 2].axis("off")

    axes[1, 0].imshow(bypass_pca_big)
    axes[1, 0].set_title("Bypass DINO PCA")
    axes[1, 0].axis("off")

    im2 = axes[1, 1].imshow(delta, cmap="seismic")
    axes[1, 1].set_title("PC1 Delta (Bypass - Base)")
    axes[1, 1].axis("off")
    fig.colorbar(im2, ax=axes[1, 1], fraction=0.046, pad=0.04)

    im3 = axes[1, 2].imshow(delta_abs, cmap="magma")
    axes[1, 2].set_title("|PC1 Delta|")
    axes[1, 2].axis("off")
    fig.colorbar(im3, ax=axes[1, 2], fraction=0.046, pad=0.04)

    plt.tight_layout()
    return fig


def build_visualization(original_image, amplified_image, model, transform, device, invert_pca=False):
    img_tensor = transform(amplified_image).unsqueeze(0).to(device)

    with torch.inference_mode():
        feat_dict = model.forward_features(img_tensor)

    patch_tokens = feat_dict["x_norm_patchtokens"].squeeze(0).detach().cpu().numpy()
    h = img_tensor.shape[-2] // model.patch_size
    w = img_tensor.shape[-1] // model.patch_size

    pca = PCA(n_components=3)
    feat_pca = pca.fit_transform(patch_tokens)

    pca_img = feat_pca.reshape((h, w, 3))
    pca_img = normalize_pca_image(pca_img)
    if invert_pca:
        pca_img = 1.0 - pca_img

    pca_pil = Image.fromarray((pca_img * 255).astype(np.uint8), mode="RGB")
    pca_nearest = pca_pil.resize(original_image.size, Image.Resampling.NEAREST)
    pca_bicubic = pca_pil.resize(original_image.size, Image.Resampling.BICUBIC)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    axes[0, 0].imshow(original_image)
    axes[0, 0].set_title("Original")
    axes[0, 0].axis("off")

    axes[0, 1].imshow(amplified_image)
    axes[0, 1].set_title("Wavelet Amplified")
    axes[0, 1].axis("off")

    axes[1, 0].imshow(pca_nearest)
    axes[1, 0].set_title("PCA - NEAREST")
    axes[1, 0].axis("off")

    axes[1, 1].imshow(pca_bicubic)
    axes[1, 1].set_title("PCA - BICUBIC")
    axes[1, 1].axis("off")

    plt.tight_layout()
    return fig


def extract_patch_tokens(image, model, transform, device):
    t = transform(image).unsqueeze(0).to(device)
    with torch.inference_mode():
        f = model.forward_features(t)
    x = f["x_norm_patchtokens"].squeeze(0).detach().cpu().numpy()
    return x


def main():
    args = parse_args()
    rng = np.random.default_rng(args.random_seed)

    if args.wavelet.lower() in {"haar", "db1"}:
        print("[warn] haar/db1 wavelet may introduce block artifacts. Prefer sym4/db4/bior2.2")

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    dual_view_dir = None
    if args.dual_view_consistency:
        dual_view_dir = Path(args.dual_view_dir) if args.dual_view_dir else Path(f"{args.output_dir}_dual")
        (dual_view_dir / "rgb").mkdir(parents=True, exist_ok=True)
        (dual_view_dir / "luma").mkdir(parents=True, exist_ok=True)

    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")

    image_files = sorted([p for p in input_dir.rglob("*") if p.suffix.lower() in IMAGE_EXTS])
    if not image_files:
        raise FileNotFoundError(f"No images found in: {input_dir}")

    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        print("[warn] CUDA not available, using CPU")
        device = "cpu"

    if args.compare_bypass and args.mode != "visualize":
        print("[warn] --compare-bypass only applies to visualize mode; ignored in build mode")
    if args.compare_bypass and not args.enable_bypass:
        print("[warn] --compare-bypass requires --enable-bypass; running normal visualize")
        args.compare_bypass = False
    if args.bypass_checkpoint and not args.enable_bypass:
        print("[warn] --bypass-checkpoint was provided without --enable-bypass; enabling bypass automatically")
        args.enable_bypass = True

    transform = build_transform(args.image_size)

    model = torch.hub.load(
        args.repo_dir,
        args.model_name,
        source="local",
        weights=args.weights,
    )
    base_model = None

    # Compare mode: keep a plain baseline model and a separate bypass model.
    if args.enable_bypass and args.compare_bypass:
        base_model = model.to(device).eval()
        model = torch.hub.load(
            args.repo_dir,
            args.model_name,
            source="local",
            weights=args.weights,
        )

    if args.enable_bypass:
        bypass = inject_zero_init_bypass(
            model,
            block_index=args.bypass_block_index,
            bottleneck_dim=args.bypass_bottleneck,
            alpha_init=args.bypass_alpha_init,
            row_min_instances=args.bypass_row_min_instances,
            row_gate_temperature=args.bypass_row_gate_temperature,
            row_axis=args.bypass_row_axis,
        )
        if args.bypass_checkpoint:
            ckpt = torch.load(args.bypass_checkpoint, map_location="cpu")
            state = ckpt.get("bypass_state_dict", ckpt)
            load_result = bypass.load_state_dict(state, strict=False)
            print(
                "[bypass] loaded checkpoint: %s (missing=%d unexpected=%d)"
                % (
                    args.bypass_checkpoint,
                    len(load_result.missing_keys),
                    len(load_result.unexpected_keys),
                )
            )
            print("[bypass] checkpoint alpha=%.6f" % float(bypass.alpha.detach().cpu().item()))
        if args.bypass_alpha_override is not None:
            with torch.no_grad():
                bypass.alpha.fill_(float(args.bypass_alpha_override))
            print("[bypass] alpha overridden to %.6f for visualization" % float(bypass.alpha.detach().cpu().item()))
        alpha_now = float(bypass.alpha.detach().cpu().item())
        if abs(alpha_now) < 1e-4:
            print("[warn] bypass alpha is very close to zero; visual differences may look negligible")
        print(
            "[bypass] enabled: block=%d bottleneck=%d row_axis=%s row_min_instances=%.2f row_gate_temp=%.2f (stomata-aware adaptive) alpha_init=%.4f"
            % (
                args.bypass_block_index,
                args.bypass_bottleneck,
                args.bypass_row_axis,
                args.bypass_row_min_instances,
                args.bypass_row_gate_temperature,
                alpha_now,
            )
        )

    model.to(device).eval()

    # Optional patch-size diagnostics
    print_patch_diagnostics(model, args.image_size, args.stomata_size_px)

    for idx, image_path in enumerate(image_files, start=1):
        original = Image.open(image_path).convert("RGB")
        original = _apply_illumination(original, args.illumination)
        original, used_gray = maybe_random_grayscale(original, args.random_gray_prob, rng)

        tiles = generate_tiles(original, args.tile_size, args.tile_stride)

        for tile_name, tile_image in tiles:
            # Apply wavelet enhancement
            amplified = wavelet_amplify(
                tile_image,
                color_mode=args.color_mode,
                wavelet=args.wavelet,
                amp_lh=args.amp_lh,
                amp_hl=args.amp_hl,
                amp_hh=args.amp_hh,
                gamma=args.gamma,
                clip_percentile=args.clip_percentile,
                ll_scale=args.ll_scale,
            )

            # Optionally apply row prior if enabled (requires external row spacing info)
            # In standard usage, row_prior_strength=0 (disabled by default)
            if args.row_prior_strength > 0:
                # For now, only process if we have an external row_spacing argument
                # (Could be extended to estimate from image, but that requires labels)
                pass  # Placeholder for future per-image spacing estimation

            if args.mode == "build":
                rel = image_path.relative_to(input_dir)
                stem_suffix = f"_{tile_name}" if tile_name != "full" else ""
                out_img = output_dir / rel.parent / f"{rel.stem}{stem_suffix}{rel.suffix}"
                out_img.parent.mkdir(parents=True, exist_ok=True)
                amplified.save(out_img)
                if args.dual_view_consistency and dual_view_dir is not None:
                    pair_rgb = wavelet_amplify(
                        tile_image,
                        color_mode="rgb",
                        wavelet=args.wavelet,
                        amp_lh=args.amp_lh,
                        amp_hl=args.amp_hl,
                        amp_hh=args.amp_hh,
                        gamma=args.gamma,
                        clip_percentile=args.clip_percentile,
                        ll_scale=args.ll_scale,
                    )
                    pair_luma = wavelet_amplify(
                        tile_image,
                        color_mode="luma",
                        wavelet=args.wavelet,
                        amp_lh=args.amp_lh,
                        amp_hl=args.amp_hl,
                        amp_hh=args.amp_hh,
                        gamma=args.gamma,
                        clip_percentile=args.clip_percentile,
                        ll_scale=args.ll_scale,
                    )
                    rgb_path = dual_view_dir / "rgb" / rel.parent / f"{rel.stem}{stem_suffix}{rel.suffix}"
                    luma_path = dual_view_dir / "luma" / rel.parent / f"{rel.stem}{stem_suffix}{rel.suffix}"
                    rgb_path.parent.mkdir(parents=True, exist_ok=True)
                    luma_path.parent.mkdir(parents=True, exist_ok=True)
                    pair_rgb.save(rgb_path)
                    pair_luma.save(luma_path)
                gray_tag = " gray" if used_gray else " rgb"
                print(f"[{idx}/{len(image_files)}] saved:{gray_tag} {out_img}")

            elif args.mode == "visualize":
                suffix = "wavelet_dino"
                if args.compare_bypass and base_model is not None:
                    compare_image = tile_image if args.compare_input == "original" else amplified
                    base_tokens = extract_patch_tokens(compare_image, base_model, transform, device)
                    bypass_tokens = extract_patch_tokens(compare_image, model, transform, device)
                    h = args.image_size // int(getattr(model, "patch_size", 16))
                    w = h
                    token_delta = bypass_tokens - base_tokens

                    print(
                        "[compare] input=%s mean_abs_delta=%.8f max_abs_delta=%.8f"
                        % (
                            args.compare_input,
                            float(np.mean(np.abs(token_delta))),
                            float(np.max(np.abs(token_delta))),
                        )
                    )
                    fig = build_bypass_comparison_visualization(
                        tile_image,
                        amplified,
                        base_tokens,
                        bypass_tokens,
                        h,
                        w,
                        invert_pca=args.invert_pca,
                    )
                    suffix = "wavelet_dino_bypass_compare"

                    if args.save_feature_npy:
                        feat_dir = output_dir / "feature_maps"
                        feat_dir.mkdir(parents=True, exist_ok=True)
                        np.save(feat_dir / f"{image_path.stem}_{tile_name}_base_tokens.npy", base_tokens)
                        np.save(feat_dir / f"{image_path.stem}_{tile_name}_bypass_tokens.npy", bypass_tokens)
                else:
                    fig = build_visualization(
                        tile_image,
                        amplified,
                        model,
                        transform,
                        device,
                        invert_pca=args.invert_pca,
                    )
                    if args.save_feature_npy:
                        feat = extract_patch_tokens(amplified, model, transform, device)
                        feat_dir = output_dir / "feature_maps"
                        feat_dir.mkdir(parents=True, exist_ok=True)
                        np.save(feat_dir / f"{image_path.stem}_{tile_name}_tokens.npy", feat)

                out_file = output_dir / f"{image_path.stem}_{tile_name}_{suffix}.png"
                out_file.parent.mkdir(parents=True, exist_ok=True)
                fig.savefig(out_file, dpi=200, bbox_inches="tight")
                if args.show:
                    plt.show()
                plt.close(fig)
                print(f"[{idx}/{len(image_files)}] saved: {out_file}")

if __name__ == "__main__":
    main()
