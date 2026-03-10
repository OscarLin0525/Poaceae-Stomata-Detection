#!/usr/bin/env python3
"""
Generate DINO layer-10 contrast comparison: FFT OFF vs FFT ON.

Outputs one figure with 4 columns per image:
1) RGB + GT OBB
2) Layer10 contrast (FFT OFF)
3) Layer10 contrast (FFT ON)
4) Difference (ON - OFF)
"""

from __future__ import annotations

import argparse
import math
import random
import sys
from pathlib import Path
from typing import List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from torchvision import transforms

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "dinov3-main"))

from mtkd_framework.engine.build_dino import DinoFeatureExtractor
from mtkd_framework.engine.pluggable_fft_block import inject_fft_blocks

IMG_TF = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]
)

# Box color rule requested by user:
# - Most boxes use one unified color.
# - Only edge-cut stomata use a different color.
UNIFIED_BOX_COLOR = "#00ff88"
EDGE_CUT_BOX_COLOR = "#ff4d4d"
EDGE_CUT_CLASS_IDS = {3}


def load_samples(image_dir: Path, label_dir: Path, n: int, seed: int):
    paths = sorted(image_dir.glob("*.jpg"))
    pairs = [(p, label_dir / f"{p.stem}.txt") for p in paths if (label_dir / f"{p.stem}.txt").exists()]
    if not pairs:
        raise RuntimeError(f"No image-label pairs found under {image_dir} and {label_dir}")
    random.seed(seed)
    choose = random.sample(pairs, min(n, len(pairs)))

    imgs, raws, names, labels = [], [], [], []
    for ip, lp in choose:
        im = Image.open(ip).convert("RGB")
        raws.append(np.array(im.resize((224, 224))) / 255.0)
        imgs.append(IMG_TF(im))
        names.append(ip.stem)

        cur = []
        with open(lp, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 9:
                    cur.append([int(parts[0])] + [float(x) for x in parts[1:9]])
        labels.append(np.array(cur) if cur else np.zeros((0, 9), dtype=np.float32))

    return torch.stack(imgs), raws, names, labels


def obb_centers(lbls: np.ndarray) -> np.ndarray:
    if len(lbls) == 0:
        return np.zeros((0, 2), dtype=np.float32)
    pts = lbls[:, 1:9].reshape(-1, 4, 2)
    return pts.mean(axis=1)


def draw_obb(ax, lbls: np.ndarray, img_size: int = 224, lw: float = 1.0):
    for row in lbls:
        cls_id = int(row[0])
        pts = row[1:9].reshape(4, 2) * img_size
        color = EDGE_CUT_BOX_COLOR if cls_id in EDGE_CUT_CLASS_IDS else UNIFIED_BOX_COLOR
        poly = plt.Polygon(
            pts,
            closed=True,
            fill=False,
            edgecolor=color,
            linewidth=lw,
        )
        ax.add_patch(poly)


def upscale_224(arr: np.ndarray) -> np.ndarray:
    pil = Image.fromarray((arr * 255).clip(0, 255).astype(np.uint8))
    return np.array(pil.resize((224, 224), Image.BILINEAR)) / 255.0


def compute_contrast_map(feat_hwc: torch.Tensor, lbls: np.ndarray) -> np.ndarray:
    # feat_hwc: [H,W,C]
    h, w, c = feat_hwc.shape
    centers = obb_centers(lbls)
    if len(centers) == 0:
        return np.zeros((h, w), dtype=np.float32)

    gt_feats = []
    for cx, cy in centers:
        px = min(max(int(cx * w), 0), w - 1)
        py = min(max(int(cy * h), 0), h - 1)
        gt_feats.append(feat_hwc[py, px, :])
    gt_feats = torch.stack(gt_feats)  # [N,C]

    all_feats = feat_hwc.reshape(h * w, c)
    dist = torch.cdist(all_feats, gt_feats, p=2)  # [H*W,N]
    dmin = dist.min(dim=1)[0].reshape(h, w).cpu().numpy()

    sim = 1.0 / (dmin + 1.0)
    sim = (sim - sim.min()) / (sim.max() - sim.min() + 1e-8)
    return sim.astype(np.float32)


def build_dino_encoder(weight_path: str, device: str):
    model = DinoFeatureExtractor(
        model_name="vit_base",
        patch_size=16,
        embed_dim=768,
        normalize_feature=False,
        pretrained_path=weight_path,
    )
    model.to(device)
    model.eval()
    return model


def extract_layer10_tokens(model: DinoFeatureExtractor, imgs: torch.Tensor, fft_on: bool, fft_init_gate: float):
    target = {}

    if fft_on:
        inject_fft_blocks(
            model.encoder,
            after_blocks=[9],
            embed_dim=768,
            num_freq_bins=32,
            hidden_dim=256,
            init_gate=fft_init_gate,
            modulation_mode="multiplicative",
            freeze_original=True,
        )

    def hook_fn(_m, _inp, out):
        target["tok"] = out[0] if isinstance(out, list) else out

    h = model.encoder.blocks[10].register_forward_hook(hook_fn)
    with torch.no_grad():
        _ = model(imgs)
    h.remove()

    tok = target["tok"]  # [B,N,C]
    b, n, c = tok.shape
    hw = int(math.sqrt(n - 1))
    feat = tok[:, 1:, :].reshape(b, hw, hw, c).cpu()  # drop CLS

    gate_val = None
    if fft_on:
        for blk in model.encoder.blocks:
            if hasattr(blk, "get_gate_value"):
                gate_val = blk.get_gate_value()
                break

    return feat, gate_val


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image_dir", default="Stomata_Dataset/barley_all/images/val")
    ap.add_argument("--label_dir", default="Stomata_Dataset/barley_all/labels/val")
    ap.add_argument(
        "--dino_weights",
        default="dinov3-main/weight folder/dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth",
    )
    ap.add_argument("--num_images", type=int, default=8)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--fft_init_gate", type=float, default=-5.0)
    ap.add_argument("--output", default="outputs/layer10_fft_compare.png")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    imgs, raws, names, labels = load_samples(Path(args.image_dir), Path(args.label_dir), args.num_images, args.seed)
    imgs = imgs.to(device)

    # FFT OFF model
    m_off = build_dino_encoder(args.dino_weights, device)
    feat_off, _ = extract_layer10_tokens(m_off, imgs, fft_on=False, fft_init_gate=args.fft_init_gate)

    # FFT ON model
    m_on = build_dino_encoder(args.dino_weights, device)
    feat_on, gate_val = extract_layer10_tokens(m_on, imgs, fft_on=True, fft_init_gate=args.fft_init_gate)

    b = len(raws)
    fig, axes = plt.subplots(b, 4, figsize=(14, b * 3.0))
    if b == 1:
        axes = axes[np.newaxis, :]

    for i in range(b):
        c_off = compute_contrast_map(feat_off[i], labels[i])
        c_on = compute_contrast_map(feat_on[i], labels[i])
        d = c_on - c_off

        c_off_u = upscale_224(c_off)
        c_on_u = upscale_224(c_on)
        d_u = upscale_224((d - d.min()) / (d.max() - d.min() + 1e-8))

        axes[i, 0].imshow(raws[i])
        draw_obb(axes[i, 0], labels[i], lw=1.2)
        axes[i, 0].set_title(f"{names[i]} + GT", fontsize=8)
        axes[i, 0].axis("off")

        axes[i, 1].imshow(raws[i])
        axes[i, 1].imshow(c_off_u, cmap="jet", alpha=0.65, vmin=0, vmax=1)
        draw_obb(axes[i, 1], labels[i], lw=0.8)
        axes[i, 1].set_title("Layer10 FFT OFF", fontsize=8)
        axes[i, 1].axis("off")

        axes[i, 2].imshow(raws[i])
        axes[i, 2].imshow(c_on_u, cmap="jet", alpha=0.65, vmin=0, vmax=1)
        draw_obb(axes[i, 2], labels[i], lw=0.8)
        gtxt = f"alpha={gate_val:.4f}" if gate_val is not None else "alpha=n/a"
        axes[i, 2].set_title(f"Layer10 FFT ON ({gtxt})", fontsize=8)
        axes[i, 2].axis("off")

        axes[i, 3].imshow(raws[i])
        axes[i, 3].imshow(d_u, cmap="RdBu_r", alpha=0.68)
        draw_obb(axes[i, 3], labels[i], lw=0.8)
        axes[i, 3].set_title("Diff (ON - OFF)", fontsize=8)
        axes[i, 3].axis("off")

    plt.suptitle("DINO Contrast Map @ Layer10: FFT OFF vs FFT ON", fontsize=12, y=0.995)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()

    print("=" * 60)
    print("Saved:", out_path)
    if gate_val is not None:
        print(f"FFT gate alpha: {gate_val:.6f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
