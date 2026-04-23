from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List

import numpy as np
import pywt
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import v2

from dino_zero_init_bypass import inject_zero_init_bypass, freeze_backbone_only_train_bypass

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Offline trainer for DINO zero-init bypass")
    p.add_argument("--input-dir", type=str, required=True, help="Image directory for offline bypass training")
    p.add_argument("--repo-dir", type=str, default="/home/oscar/Poaceae-Stomata-Detection/dinov3-main", help="DINOv3 repo directory")
    p.add_argument("--weights", type=str, required=True, help="Path to DINO checkpoint")
    p.add_argument("--model-name", type=str, default="dinov3_vitb16", help="DINO model name")
    p.add_argument("--output-dir", type=str, default="outputs/bypass_offline", help="Output directory")

    p.add_argument("--image-size", type=int, default=518, help="Input size for DINO")
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--log-interval", type=int, default=20)
    p.add_argument("--save-every", type=int, default=1)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", type=str, default="cuda")

    p.add_argument("--wavelet", type=str, default="sym4")
    p.add_argument("--amp-lh", type=float, default=4.0)
    p.add_argument("--amp-hl", type=float, default=4.0)
    p.add_argument("--amp-hh", type=float, default=0.0)
    p.add_argument("--gamma", type=float, default=0.9)
    p.add_argument("--ll-scale", type=float, default=1.0)

    p.add_argument("--bypass-block-index", type=int, default=6)
    p.add_argument("--bypass-bottleneck", type=int, default=64)
    p.add_argument("--bypass-alpha-init", type=float, default=0.0)
    p.add_argument(
        "--bypass-row-min-instances",
        type=float,
        default=4.0,
        help="Suppress rows with expected stomata count below this value (0 disables).",
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
        "--locator-sparsity-weight",
        type=float,
        default=0.02,
        help="Weight for sparsity regularization on locator probability map.",
    )
    p.add_argument(
        "--locator-row-penalty-weight",
        type=float,
        default=0.10,
        help="Weight for penalizing weak rows that look active but have too few instances.",
    )
    p.add_argument(
        "--locator-row-min-instances",
        type=float,
        default=4.0,
        help="Target minimum expected instances per active row for row-penalty loss.",
    )
    p.add_argument(
        "--locator-row-presence-threshold",
        type=float,
        default=0.5,
        help="Row expected-count threshold to consider a row as active.",
    )
    p.add_argument(
        "--locator-row-presence-temperature",
        type=float,
        default=8.0,
        help="Sharpness for soft row-presence estimation in row-penalty loss.",
    )
    return p.parse_args()


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def signed_power(x: np.ndarray, gamma: float) -> np.ndarray:
    return np.sign(x) * (np.abs(x) ** gamma)


def amplify_channel(
    ch: np.ndarray,
    wavelet: str,
    amp_lh: float,
    amp_hl: float,
    amp_hh: float,
    gamma: float,
    ll_scale: float,
) -> np.ndarray:
    ll, (lh, hl, hh) = pywt.dwt2(ch, wavelet)
    ll = ll * float(ll_scale)
    lh_amp = signed_power(lh, gamma) * float(amp_lh)
    hl_amp = signed_power(hl, gamma) * float(amp_hl)
    hh_amp = signed_power(hh, gamma) * float(amp_hh)
    return pywt.idwt2((ll, (lh_amp, hl_amp, hh_amp)), wavelet)


def normalize_01(x: np.ndarray) -> np.ndarray:
    mn, mx = float(x.min()), float(x.max())
    if mx <= mn:
        return np.zeros_like(x, dtype=np.float32)
    return (x - mn) / (mx - mn)


def wavelet_luma_rgb(
    image: Image.Image,
    wavelet: str,
    amp_lh: float,
    amp_hl: float,
    amp_hh: float,
    gamma: float,
    ll_scale: float,
) -> Image.Image:
    ycbcr = image.convert("YCbCr")
    arr = np.asarray(ycbcr).astype(np.float32)
    y = arr[:, :, 0] / 255.0
    y_amp = amplify_channel(y, wavelet, amp_lh, amp_hl, amp_hh, gamma, ll_scale)
    y_amp = y_amp[: y.shape[0], : y.shape[1]]
    y_amp = normalize_01(y_amp)
    y8 = np.clip(y_amp * 255.0, 0.0, 255.0).astype(np.uint8)
    rgb = np.stack([y8, y8, y8], axis=-1)
    return Image.fromarray(rgb, mode="RGB")


def build_transform(image_size: int):
    return v2.Compose(
        [
            v2.ToImage(),
            v2.Resize((image_size, image_size), antialias=True),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ]
    )


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


def collate_images(batch):
    images = [x[0] for x in batch]
    paths = [x[1] for x in batch]
    return images, paths


def get_patch_tokens(model, tensor: torch.Tensor) -> torch.Tensor:
    feat = model.forward_features(tensor)
    return feat["x_norm_patchtokens"]


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        print("[warn] CUDA not available, using CPU")
        device = "cpu"

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    transform = build_transform(args.image_size)
    dataset = ImageFolderDataset(input_dir)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_images,
    )

    # Base teacher model (frozen, no bypass) for wavelet-target features.
    base_model = torch.hub.load(
        args.repo_dir,
        args.model_name,
        source="local",
        weights=args.weights,
    ).to(device).eval()

    # Train model with bypass injected.
    train_model = torch.hub.load(
        args.repo_dir,
        args.model_name,
        source="local",
        weights=args.weights,
    )
    bypass = inject_zero_init_bypass(
        train_model,
        block_index=args.bypass_block_index,
        bottleneck_dim=args.bypass_bottleneck,
        alpha_init=args.bypass_alpha_init,
        row_min_instances=args.bypass_row_min_instances,
        row_gate_temperature=args.bypass_row_gate_temperature,
        row_axis=args.bypass_row_axis,
    )
    train_model = train_model.to(device).train()

    trainable = freeze_backbone_only_train_bypass(train_model)
    print(
        "[setup] trainable_bypass_params=%d (row_min_instances=%.2f row_gate_temp=%.2f sparse_w=%.4f row_penalty_w=%.4f)"
        % (
            trainable,
            args.bypass_row_min_instances,
            args.bypass_row_gate_temperature,
            args.locator_sparsity_weight,
            args.locator_row_penalty_weight,
        )
    )
    print(f"[setup] row_axis={args.bypass_row_axis}")

    optim = torch.optim.AdamW(
        [p for p in train_model.parameters() if p.requires_grad],
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    history = []
    global_step = 0

    for epoch in range(1, args.epochs + 1):
        losses_total = []
        losses_feat = []
        losses_sparse = []
        losses_row = []
        row_active_ratios = []
        for images, _paths in loader:
            global_step += 1
            imgs_org = [transform(im) for im in images]
            imgs_wav = [
                transform(
                    wavelet_luma_rgb(
                        im,
                        wavelet=args.wavelet,
                        amp_lh=args.amp_lh,
                        amp_hl=args.amp_hl,
                        amp_hh=args.amp_hh,
                        gamma=args.gamma,
                        ll_scale=args.ll_scale,
                    )
                )
                for im in images
            ]

            org_t = torch.stack(imgs_org, dim=0).to(device)
            wav_t = torch.stack(imgs_wav, dim=0).to(device)

            with torch.no_grad():
                target_tokens = get_patch_tokens(base_model, wav_t).detach()

            pred_tokens = get_patch_tokens(train_model, org_t)

            # Match bypass-enhanced features to wavelet-target teacher features.
            loss_l2 = F.mse_loss(pred_tokens, target_tokens)
            loss_cos = (1.0 - F.cosine_similarity(pred_tokens, target_tokens, dim=-1)).mean()
            loss_feat = loss_l2 + 0.5 * loss_cos

            locator_prob = bypass.last_stomata_prob
            loss_sparse = pred_tokens.new_zeros(())
            if locator_prob is not None:
                prefix_tokens = int(getattr(bypass, "last_prefix_tokens", 1) or 0)
                if locator_prob.shape[1] > prefix_tokens:
                    patch_prob = locator_prob[:, prefix_tokens:, :]
                else:
                    patch_prob = locator_prob
                loss_sparse = patch_prob.mean()

            row_counts = bypass.last_row_counts
            row_active_ratio = 0.0
            loss_row = pred_tokens.new_zeros(())
            if row_counts is not None:
                row_counts = row_counts.squeeze(-1)
                row_presence = torch.sigmoid(
                    (row_counts - args.locator_row_presence_threshold) * args.locator_row_presence_temperature
                )
                row_shortfall = F.relu(args.locator_row_min_instances - row_counts)
                loss_row = (row_presence * row_shortfall).mean()
                row_active_ratio = float((row_presence > 0.5).float().mean().detach().cpu().item())

            loss = (
                loss_feat
                + args.locator_sparsity_weight * loss_sparse
                + args.locator_row_penalty_weight * loss_row
            )

            optim.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_([p for p in train_model.parameters() if p.requires_grad], max_norm=1.0)
            optim.step()

            losses_total.append(float(loss.detach().cpu().item()))
            losses_feat.append(float(loss_feat.detach().cpu().item()))
            losses_sparse.append(float(loss_sparse.detach().cpu().item()))
            losses_row.append(float(loss_row.detach().cpu().item()))
            row_active_ratios.append(row_active_ratio)

            if global_step % args.log_interval == 0:
                alpha = float(bypass.alpha.detach().cpu().item())
                locator_norm = float(sum(p.norm().item() for p in bypass.locator.parameters()))
                strength_norm = float(sum(p.norm().item() for p in bypass.strength_head.parameters()))
                fill_norm = float(sum(p.norm().item() for p in bypass.fill_head.parameters()))
                print(
                    "[step %d] loss=%.6f feat=%.6f sparse=%.6f row=%.6f row_active=%.4f alpha=%.6f locator_norm=%.6f strength_norm=%.6f fill_norm=%.6f"
                    % (
                        global_step,
                        float(np.mean(losses_total[-args.log_interval:])),
                        float(np.mean(losses_feat[-args.log_interval:])),
                        float(np.mean(losses_sparse[-args.log_interval:])),
                        float(np.mean(losses_row[-args.log_interval:])),
                        float(np.mean(row_active_ratios[-args.log_interval:])),
                        alpha,
                        locator_norm,
                        strength_norm,
                        fill_norm,
                    )
                )

        epoch_loss = float(np.mean(losses_total)) if losses_total else 0.0
        epoch_feat = float(np.mean(losses_feat)) if losses_feat else 0.0
        epoch_sparse = float(np.mean(losses_sparse)) if losses_sparse else 0.0
        epoch_row = float(np.mean(losses_row)) if losses_row else 0.0
        epoch_row_active = float(np.mean(row_active_ratios)) if row_active_ratios else 0.0
        alpha = float(bypass.alpha.detach().cpu().item())
        locator_norm = float(sum(p.norm().item() for p in bypass.locator.parameters()))
        strength_norm = float(sum(p.norm().item() for p in bypass.strength_head.parameters()))
        fill_norm = float(sum(p.norm().item() for p in bypass.fill_head.parameters()))
        rec = {
            "epoch": epoch,
            "loss": epoch_loss,
            "loss_feat": epoch_feat,
            "loss_sparse": epoch_sparse,
            "loss_row": epoch_row,
            "row_active_ratio": epoch_row_active,
            "alpha": alpha,
            "locator_norm": locator_norm,
            "strength_norm": strength_norm,
            "fill_norm": fill_norm,
        }
        history.append(rec)
        print(
            "[epoch %d/%d] loss=%.6f feat=%.6f sparse=%.6f row=%.6f row_active=%.4f alpha=%.6f locator_norm=%.6f strength_norm=%.6f fill_norm=%.6f"
            % (
                epoch,
                args.epochs,
                epoch_loss,
                epoch_feat,
                epoch_sparse,
                epoch_row,
                epoch_row_active,
                alpha,
                locator_norm,
                strength_norm,
                fill_norm,
            )
        )

        if epoch % args.save_every == 0:
            ckpt = {
                "epoch": epoch,
                "bypass_state_dict": bypass.state_dict(),
                "config": vars(args),
                "stats": rec,
            }
            torch.save(ckpt, output_dir / f"bypass_epoch_{epoch}.pt")

    final_ckpt = {
        "epoch": args.epochs,
        "bypass_state_dict": bypass.state_dict(),
        "config": vars(args),
        "history": history,
    }
    torch.save(final_ckpt, output_dir / "bypass_last.pt")
    (output_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    print(f"[done] saved checkpoint to {output_dir / 'bypass_last.pt'}")


if __name__ == "__main__":
    main()
