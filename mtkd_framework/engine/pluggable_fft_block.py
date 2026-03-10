"""
Pluggable FFT Block — A drop-in ``nn.Module`` for DINO's ``nn.ModuleList``
==========================================================================

Design Goal
-----------
DINO (v2 / v3) stores all Transformer blocks in::

    self.blocks = nn.ModuleList([Block_0, Block_1, …, Block_11])

Each block's ``forward`` signature is::

    # DINOv3 — SelfAttentionBlock
    def forward(self, x_or_x_list, rope_or_rope_list=None):
        # x_or_x_list : Tensor [B, N, C]  OR  List[Tensor]
        # rope_or_rope_list : rope embedding tuple(sin, cos), or list of them

We create a lightweight ``PluggableFFTBlock`` (based on ``segm_v2``'s
``SEGMAdapter``) that has the **exact same call signature** so it can be
seamlessly inserted into the ``blocks`` list.

The block:
1. Accepts ``[B, N, C]`` or ``List[Tensor]`` ← same as DINOv3 blocks.
2. Strips CLS + storage prefix tokens.
3. Applies row-wise FFT → periodic grid → feature modulation (zero-init gate).
4. Re-concatenates prefix tokens.
5. Returns the same format as the input (single ``Tensor`` or ``List``).

Because of the **zero-init gate**, at t = 0 this block is the identity,
meaning it does not disturb the pretrained DINO weights at all.

Insertion
---------
``inject_fft_blocks()`` (bottom of this file) performs the "micro-surgery":
it splices one or more ``PluggableFFTBlock`` instances *into* the existing
``nn.ModuleList`` at the requested positions.  No DINO source code is
modified.

Usage
-----
.. code-block:: python

    from mtkd_framework.engine import PluggableFFTBlock, inject_fft_blocks

    # --- Method 1: automatic injection ---
    inject_fft_blocks(
        dino_model,                # any model that has .blocks : nn.ModuleList
        after_blocks=[9],          # insert after block 9  (0-indexed)
        embed_dim=768,
        n_storage_tokens=0,        # depends on the model
    )

    # --- Method 2: manual construction ---
    fft_blk = PluggableFFTBlock(embed_dim=768, n_storage_tokens=0)
    dino_model.blocks.insert(11, fft_blk)   # after block 10

"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


# ======================================================================
# Core FFT primitives (self-contained — no dependency on segm_v2)
# ======================================================================


class _RowFrequencyEstimator(nn.Module):
    """Light-weight row-wise FFT + peak detection (same idea as segm_v2)."""

    def __init__(
        self,
        embed_dim: int = 768,
        hidden_dim: int = 256,
        num_freq_bins: int = 32,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_freq_bins = num_freq_bins

        self.feature_proj = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )
        # learnable frequency filter
        self.freq_filter = nn.Parameter(torch.ones(num_freq_bins))
        # encode freq info back to embed_dim
        self.freq_encoder = nn.Sequential(
            nn.Linear(num_freq_bins, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, embed_dim),
        )
        # confidence estimator
        self.confidence_estimator = nn.Sequential(
            nn.Linear(num_freq_bins, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid(),
        )

    def forward(self, spatial: Tensor) -> Dict[str, Tensor]:
        """
        Args:
            spatial: ``[B, H, W, C]``

        Returns:
            dict with  ``dominant_freq`` ``[B, H]``, ``freq_confidence`` ``[B, H]``,
            ``freq_spectrum`` ``[B, H, bins]``, ``row_features`` ``[B, H, C]``.
        """
        B, H, W, C = spatial.shape

        projected = self.feature_proj(spatial)  # [B, H, W, hid]
        row_signal = projected.norm(dim=-1)     # [B, H, W]
        row_signal = row_signal - row_signal.mean(dim=-1, keepdim=True)

        fft_result = torch.fft.rfft(row_signal, dim=-1)  # [B, H, W//2+1] complex
        power = torch.abs(fft_result) ** 2

        # bin to fixed size
        ps = power.unsqueeze(1).permute(0, 1, 3, 2)  # [B, 1, freq_dim, H]
        freq_spectrum = F.interpolate(
            ps, size=(self.num_freq_bins, H), mode="bilinear", align_corners=False,
        ).squeeze(1).permute(0, 2, 1)  # [B, H, bins]

        freq_spectrum = freq_spectrum * F.softplus(self.freq_filter)

        # dominant frequency per row
        dominant_idx = freq_spectrum.argmax(dim=-1)  # [B, H]
        dominant_freq = dominant_idx.float() / self.num_freq_bins

        # confidence
        freq_confidence = self.confidence_estimator(freq_spectrum).squeeze(-1)  # [B, H]

        # row features for downstream
        row_features = self.freq_encoder(freq_spectrum)  # [B, H, C]

        return {
            "dominant_freq": dominant_freq,
            "freq_confidence": freq_confidence,
            "freq_spectrum": freq_spectrum,
            "row_features": row_features,
        }


class _PeriodicGridGenerator(nn.Module):
    """Generate a Von-Mises periodic grid from per-row frequencies."""

    def __init__(self, max_height: int = 32, init_kappa: float = 3.0):
        super().__init__()
        self.row_phase = nn.Parameter(torch.zeros(max_height))
        self.kappa = nn.Parameter(torch.tensor(float(init_kappa)))

    def forward(
        self,
        dominant_freq: Tensor,
        freq_confidence: Tensor,
        H: int,
        W: int,
    ) -> Tensor:
        """
        Returns:
            grid: ``[B, 1, H, W]``  values in ``[0, 1]``.
        """
        B = dominant_freq.shape[0]
        device = dominant_freq.device

        x_coords = torch.linspace(0, 1, W, device=device).unsqueeze(0).unsqueeze(0)  # [1, 1, W]

        phase = self.row_phase[:H].unsqueeze(0)  # [1, H]
        freq = dominant_freq  # [B, H]

        # cos arg: 2π * freq * x + phase
        angle = 2 * math.pi * freq.unsqueeze(-1) * x_coords + phase.unsqueeze(-1)  # [B, H, W]

        kappa = F.softplus(self.kappa)
        wave = torch.exp(kappa * (torch.cos(angle) - 1))  # [B, H, W]

        # scale to [0, 1]
        wave_min = wave.amin(dim=-1, keepdim=True)
        wave_max = wave.amax(dim=-1, keepdim=True)
        wave = (wave - wave_min) / (wave_max - wave_min + 1e-8)

        # weight by confidence
        wave = wave * freq_confidence.unsqueeze(-1)

        return wave.unsqueeze(1)  # [B, 1, H, W]


# ======================================================================
# PluggableFFTBlock
# ======================================================================


class PluggableFFTBlock(nn.Module):
    """
    A drop-in replacement / insertion block for DINOv3's ``SelfAttentionBlock``.

    **Contract**: identical ``forward`` signature to ``SelfAttentionBlock``:

    * ``forward(x_or_x_list, rope_or_rope_list=None)``
    * Input / output shape: ``[B, N, C]``  or  ``List[Tensor]``

    Internally the block:

    1. Strips the ``1 + n_storage_tokens`` prefix.
    2. Reshapes patches to ``[B, H, W, C]``.
    3. Runs row-FFT → grid → modulation (with a zero-init gate).
    4. Returns the residual ``x + \alpha \cdot \delta`` repacked with prefix.
   ``\alpha = \text{sigmoid}(\text{gate})`` starts near 0 (zero-init).

    Parameters
    ----------
    embed_dim : int
        Token dimension (must match the DINO model's ``embed_dim``).
    n_storage_tokens : int
        Number of storage / register tokens prepended *after* CLS.
        For standard DINOv3 with no registers this is 0.
    num_freq_bins : int
        Number of frequency bins for the row-FFT analysis.
    hidden_dim : int
        Internal hidden dimension.
    init_gate : float
        Initial gate logit.  ``sigmoid(-5) ≈ 0.007`` → near-identity.
        This is the *tuning coefficient* (α) in the residual
        ``output = x + α · δ``.
    modulation_mode : str
        ``"multiplicative"`` or ``"additive"``.
    """

    def __init__(
        self,
        embed_dim: int = 768,
        n_storage_tokens: int = 0,
        num_freq_bins: int = 32,
        hidden_dim: int = 256,
        init_gate: float = -5.0,
        modulation_mode: str = "multiplicative",
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.n_storage_tokens = n_storage_tokens
        self.prefix_len = 1 + n_storage_tokens  # CLS + storage
        self.modulation_mode = modulation_mode

        # --- FFT components ---
        self.freq_estimator = _RowFrequencyEstimator(
            embed_dim=embed_dim,
            hidden_dim=hidden_dim,
            num_freq_bins=num_freq_bins,
        )
        self.grid_generator = _PeriodicGridGenerator(max_height=64)

        # additive needs a channel projection
        if modulation_mode == "additive":
            self.channel_proj = nn.Conv2d(1, embed_dim, 1, bias=False)
            nn.init.normal_(self.channel_proj.weight, std=0.02)

        # --- Zero-init gate ---
        self.gate = nn.Parameter(torch.tensor(float(init_gate)))

        # cache for diagnostics
        self._cache: Dict[str, Tensor] = {}

    # ------------------------------------------------------------------
    # Single tensor
    # ------------------------------------------------------------------
    def _forward_single(self, x: Tensor, rope=None) -> Tensor:
        """Process a single ``[B, N, C]`` tensor."""
        B, N, C = x.shape

        prefix = x[:, : self.prefix_len, :]    # CLS + storage
        patches = x[:, self.prefix_len :, :]    # [B, H*W, C]

        N_patches = patches.shape[1]
        H = W = int(math.sqrt(N_patches))
        if H * W != N_patches:
            # non-square fallback: find closest factors
            for h_try in range(int(math.sqrt(N_patches)), 0, -1):
                if N_patches % h_try == 0:
                    H = h_try
                    W = N_patches // h_try
                    break

        spatial = patches.reshape(B, H, W, C)

        # --- FFT analysis ---
        freq_info = self.freq_estimator(spatial)
        grid = self.grid_generator(
            freq_info["dominant_freq"],
            freq_info["freq_confidence"],
            H,
            W,
        )  # [B, 1, H, W]

        # ---- Residual connection: output = x + α · δ(x) ----
        alpha = torch.sigmoid(self.gate)  # tuning coefficient, starts near 0

        if self.modulation_mode == "multiplicative":
            # δ = x ⊙ grid_centered  (feature-dependent modulation)
            grid_centered = (grid - 0.5) * 2  # -> [-1, 1]
            grid_for_mult = grid_centered.permute(0, 2, 3, 1)  # [B, H, W, 1]
            delta = spatial * grid_for_mult
        else:
            # δ = proj(grid)  (input-independent additive offset)
            proj_grid = self.channel_proj(grid)               # [B, C, H, W]
            delta = proj_grid.permute(0, 2, 3, 1)              # [B, H, W, C]

        enhanced = spatial + alpha * delta

        enhanced_patches = enhanced.reshape(B, N_patches, C)

        self._cache = {
            "grid": grid,
            "gate_value": alpha,
            "freq_info": freq_info,
        }

        return torch.cat([prefix, enhanced_patches], dim=1)

    # ------------------------------------------------------------------
    # DINOv3-compatible forward (single tensor **or** list of tensors)
    # ------------------------------------------------------------------
    def forward(
        self,
        x_or_x_list: Union[Tensor, List[Tensor]],
        rope_or_rope_list=None,
    ) -> Union[Tensor, List[Tensor]]:
        """
        Drop-in replacement for ``SelfAttentionBlock.forward``.

        Args:
            x_or_x_list:      ``[B, N, C]`` or ``List[Tensor]``.
            rope_or_rope_list: Ignored (FFT block does not use RoPE).

        Returns:
            Same type / shape as the input.
        """
        if isinstance(x_or_x_list, Tensor):
            return self._forward_single(x_or_x_list)
        elif isinstance(x_or_x_list, list):
            return [self._forward_single(x) for x in x_or_x_list]
        else:
            raise TypeError(f"Unsupported input type: {type(x_or_x_list)}")

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------
    def get_gate_value(self) -> float:
        return torch.sigmoid(self.gate).item()

    def get_cache(self) -> Dict[str, Tensor]:
        return self._cache


# ======================================================================
# Injection utility
# ======================================================================


def inject_fft_blocks(
    model: nn.Module,
    after_blocks: Sequence[int] = (10,),
    embed_dim: int = 768,
    n_storage_tokens: Optional[int] = None,
    num_freq_bins: int = 32,
    hidden_dim: int = 256,
    init_gate: float = -5.0,
    modulation_mode: str = "multiplicative",
    freeze_original: bool = True,
) -> List[PluggableFFTBlock]:
    """
    Insert ``PluggableFFTBlock`` instances into a DINO model's
    ``blocks`` (``nn.ModuleList``).

    This is the "micro-surgery" helper.  It:
    1. Locates ``model.blocks`` (an ``nn.ModuleList``).
    2. Inserts a new ``PluggableFFTBlock`` *after* each requested index.
    3. Optionally freezes all *original* blocks so only FFT params train.

    Args:
        model:              A DINO-family model possessing ``model.blocks``.
        after_blocks:       0-based indices **of original blocks** after which to
                            insert FFT blocks.  ``[10]`` ⇒ insert between
                            block 10 and block 11.
        embed_dim:          Token embedding dimension.
        n_storage_tokens:   Number of storage/register tokens.
                            Auto-detected from ``model.n_storage_tokens`` if None.
        num_freq_bins:      Frequency bins for the FFT analysis.
        hidden_dim:         Hidden dim inside FFT block.
        init_gate:          Initial gate logit for zero-init.
        modulation_mode:    ``"multiplicative"`` or ``"additive"``.
        freeze_original:    Freeze all *original* DINO blocks (keep FFT trainable).

    Returns:
        A list of the newly created ``PluggableFFTBlock`` instances
        (useful for e.g. parameter groups).
    """
    if not hasattr(model, "blocks"):
        raise AttributeError(
            "model does not have a `.blocks` attribute — "
            "expected an nn.ModuleList of Transformer blocks."
        )

    blocks: nn.ModuleList = model.blocks

    if n_storage_tokens is None:
        n_storage_tokens = getattr(model, "n_storage_tokens", 0)

    # Sort descending so insertions don't shift earlier indices
    sorted_positions = sorted(after_blocks, reverse=True)
    created: List[PluggableFFTBlock] = []

    for pos in sorted_positions:
        if pos < 0 or pos >= len(blocks):
            raise IndexError(
                f"after_blocks={pos} is out of range for "
                f"model.blocks (length {len(blocks)})"
            )
        fft_blk = PluggableFFTBlock(
            embed_dim=embed_dim,
            n_storage_tokens=n_storage_tokens,
            num_freq_bins=num_freq_bins,
            hidden_dim=hidden_dim,
            init_gate=init_gate,
            modulation_mode=modulation_mode,
        )
        # Move to the same device as existing blocks
        try:
            existing_device = next(blocks.parameters()).device
            fft_blk = fft_blk.to(existing_device)
        except StopIteration:
            pass  # no existing parameters — stay on CPU
        # insert right after the requested position
        insert_idx = pos + 1
        blocks.insert(insert_idx, fft_blk)
        created.append(fft_blk)

    # Optionally freeze original blocks
    if freeze_original:
        for i, blk in enumerate(blocks):
            if not isinstance(blk, PluggableFFTBlock):
                for p in blk.parameters():
                    p.requires_grad = False

    print(
        f"[inject_fft_blocks] Inserted {len(created)} PluggableFFTBlock(s) "
        f"after original block(s) {list(after_blocks)}.  "
        f"model.blocks now has {len(blocks)} entries."
    )
    return list(reversed(created))  # return in ascending order
