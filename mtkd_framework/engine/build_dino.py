"""
Frozen DINO Feature Extractor (for teacher feature alignment)
=============================================================

Ported from  DINO_Teacher/dinoteacher/engine/build_dino.py
Adapted to work with the DINOv3 models available in the workspace
(``dinov3-main/dinov3/``).

Pipeline
--------
1. BGR → RGB conversion (data loaders typically give BGR).
2. ImageNet normalisation  (μ = [123.675, 116.280, 103.530],
                             σ = [58.395,  57.120,  57.375]).
3. Pad to ``patch_size``-divisible dims.
4. ``encoder.get_intermediate_layers(x)[0]`` → ``[B, N_patches, D]``.
5. Strip CLS token (DINOv1 only; v2/v3 already exclude it).
6. Optional L2 normalisation.
7. Reshape to spatial feature map ``[B, D, H/p, W/p]``.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T

# ---------------------------------------------------------------------------
# Make dinov3 importable
# ---------------------------------------------------------------------------
# engine/build_dino.py sits at  mtkd_framework/engine/build_dino.py
# We need the workspace root:   .parent (engine) → .parent (mtkd_framework) → .parent (workspace)
_DINOV3_ROOT = Path(__file__).resolve().parent.parent.parent / "dinov3-main"
if str(_DINOV3_ROOT) not in sys.path:
    sys.path.insert(0, str(_DINOV3_ROOT))


class _DinoPreprocessing:
    """ImageNet normalisation (pixel values in [0, 255])."""

    def __init__(
        self,
        pixel_mean: list[float] = (123.675, 116.280, 103.530),
        pixel_std:  list[float] = (58.395, 57.120, 57.375),
    ):
        self.normalize = T.Normalize(mean=list(pixel_mean), std=list(pixel_std))

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return self.normalize(x)


class DinoFeatureExtractor(nn.Module):
    """
    Frozen DINO Feature Extractor — returns a **spatial** feature map.

    This is the MTKD counterpart of ``DinoVitFeatureExtractor`` in DINO Teacher.

    Supports:
    * DINOv3 models shipped in ``dinov3-main/`` (vit_small / vit_base / vit_large).
    * DINOv2 models when the corresponding hub functions are importable.

    Args:
        model_name:          One of ``"vit_small"`` / ``"vit_base"`` / ``"vit_large"``
                             or a DINOv2 name like ``"dinov2_vitb14"``.
        patch_size:          Must match the model's patch_size (16 for v3, 14 for v2).
        embed_dim:           Expected embedding dimension (used for sanity check).
        normalize_feature:   L2-normalise output features along the embedding dim.
        is_bgr:              If True, convert input from BGR → RGB.
        freeze:              Freeze all encoder parameters (default: True).
        pretrained_path:     Optional path to a ``*.pth`` checkpoint.
    """

    def __init__(
        self,
        model_name: str = "vit_base",
        patch_size: int = 16,
        embed_dim: int = 768,
        normalize_feature: bool = True,
        is_bgr: bool = False,
        freeze: bool = True,
        pretrained_path: Optional[str] = None,
    ):
        super().__init__()

        self.model_name = model_name
        self.patch_size = patch_size
        self.normalize_feature = normalize_feature
        self.is_bgr = is_bgr
        self.preprocessing = _DinoPreprocessing()

        # ----- build encoder -----
        self.encoder: nn.Module = self._build_encoder(model_name, patch_size, pretrained_path)
        self.embed_dim: int = getattr(self.encoder, "embed_dim", embed_dim)
        assert self.embed_dim == embed_dim, (
            f"Expected embed_dim={embed_dim} but encoder has {self.embed_dim}"
        )

        if freeze:
            for p in self.encoder.parameters():
                p.requires_grad = False
            self.encoder.eval()

    # ------------------------------------------------------------------
    # Encoder construction helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _build_encoder(
        model_name: str,
        patch_size: int,
        pretrained_path: Optional[str],
    ) -> nn.Module:
        """Try DINOv3 first, fall back to DINOv2, else raise."""

        # --- DINOv3 hub  (dinov3-main/dinov3/models/vision_transformer.py) ---
        try:
            from dinov3.models.vision_transformer import vit_small, vit_base, vit_large

            _v3_map = {
                "vit_small": vit_small,
                "vit_base":  vit_base,
                "vit_large": vit_large,
            }
            if model_name in _v3_map:
                model = _v3_map[model_name](patch_size=patch_size)
                if pretrained_path and os.path.isfile(pretrained_path):
                    state = torch.load(pretrained_path, map_location="cpu")
                    if "model" in state:
                        state = state["model"]
                    model.load_state_dict(state, strict=False)
                return model
        except ImportError:
            pass

        # --- DINOv2 hub (if available) ---
        try:
            from dinov2.hub.backbones import (
                dinov2_vits14,
                dinov2_vitb14,
                dinov2_vitl14,
            )

            _v2_map = {
                "dinov2_vits14": dinov2_vits14,
                "dinov2_vitb14": dinov2_vitb14,
                "dinov2_vitl14": dinov2_vitl14,
            }
            if model_name in _v2_map:
                model = _v2_map[model_name](pretrained=False)
                if pretrained_path and os.path.isfile(pretrained_path):
                    model.load_state_dict(
                        torch.load(pretrained_path, map_location="cpu"),
                        strict=False,
                    )
                return model
        except ImportError:
            pass

        raise RuntimeError(
            f"Cannot build DINO encoder for model_name='{model_name}'. "
            "Make sure dinov3-main/ or dinov2/ is in the workspace."
        )

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """
        Extract a **spatial** feature map from the DINO encoder.

        Note: ``@torch.no_grad()`` is intentionally **not** used here so
        that gradients can flow through any injected trainable blocks
        (e.g. ``PluggableFFTBlock``).  The original encoder parameters
        already have ``requires_grad=False``, so no extra memory is
        allocated for their gradients.

        Args:
            images: ``[B, 3, H, W]``  (float, value range 0-255 or 0-1).

        Returns:
            features: ``[B, D, H_p, W_p]``  where ``H_p = H // patch_size``,
                      ``W_p = W // patch_size``.
        """
        x = images

        # --- BGR → RGB ---
        if self.is_bgr:
            x = x[:, [2, 1, 0], :, :]

        # --- Ensure [0, 255] range (dataset may provide [0, 1]) ---
        if x.max() <= 1.0:
            x = x * 255.0

        # --- ImageNet normalisation ---
        x = self.preprocessing(x)

        B, _, H, W = x.shape

        # --- Pad to patch_size-divisible ---
        pad_h = (self.patch_size - H % self.patch_size) % self.patch_size
        pad_w = (self.patch_size - W % self.patch_size) % self.patch_size
        if pad_h or pad_w:
            x = F.pad(x, (0, pad_w, 0, pad_h))

        _, _, H_padded, W_padded = x.shape
        f_h = H_padded // self.patch_size
        f_w = W_padded // self.patch_size

        # --- Extract intermediate features ---
        if hasattr(self.encoder, "get_intermediate_layers"):
            feats = self.encoder.get_intermediate_layers(x, n=1)
            if isinstance(feats, (list, tuple)):
                feats = feats[0]  # [B, N_patches(+cls), D]
        elif hasattr(self.encoder, "forward_features"):
            out = self.encoder.forward_features(x)
            if isinstance(out, dict):
                feats = out.get("x_norm_patchtokens", out.get("x_prenorm"))
                if feats is None:
                    raise RuntimeError("Cannot find patch tokens in forward_features output")
            else:
                feats = out
        else:
            feats = self.encoder(x)

        # --- Strip CLS / storage tokens ---
        expected_patches = f_h * f_w
        if feats.shape[1] > expected_patches:
            feats = feats[:, -expected_patches:, :]

        # --- L2 normalise ---
        if self.normalize_feature:
            feats = F.normalize(feats, p=2, dim=2)

        # --- Reshape to spatial ---
        feats = (
            feats
            .contiguous()
            .transpose(1, 2)
            .contiguous()
            .view(B, self.embed_dim, f_h, f_w)
        )

        return feats

    # ------------------------------------------------------------------
    # Keep eval mode when caller does .train()
    # ------------------------------------------------------------------
    def train(self, mode: bool = True):
        super().train(False)
        self.encoder.eval()
        return self
