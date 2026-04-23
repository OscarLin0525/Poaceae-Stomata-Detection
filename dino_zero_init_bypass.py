from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn


def _infer_hw(num_tokens: int) -> Optional[Tuple[int, int]]:
    if num_tokens <= 0:
        return None
    side = int(math.sqrt(num_tokens))
    if side * side == num_tokens:
        return side, side
    return None


def _split_prefix_and_patch_tokens(
    tokens: torch.Tensor,
    max_prefix_search: int = 32,
) -> Optional[Tuple[torch.Tensor, torch.Tensor, int, int, int]]:
    """Split token sequence into prefix tokens (CLS/REG/extra) and square patch tokens.

    DINO variants may prepend multiple non-patch tokens. We search for a prefix length
    such that the remaining token count forms a square patch grid.
    """
    if tokens.ndim != 3:
        return None

    n = int(tokens.shape[1])
    if n <= 0:
        return None

    max_prefix = min(max_prefix_search, max(0, n - 1))
    for prefix_len in range(0, max_prefix + 1):
        patch_count = n - prefix_len
        hw = _infer_hw(patch_count)
        if hw is None:
            continue
        h, w = hw
        prefix_tokens = tokens[:, :prefix_len, :]
        patch_tokens = tokens[:, prefix_len:, :]
        return prefix_tokens, patch_tokens, h, w, prefix_len

    return None


class StomataLocator(nn.Module):
    """Detects stomata positions from DINO features.
    
    Output: [B, N, 1] - probability that each token contains a stomata.
    """
    def __init__(self, embed_dim: int, hidden_dim: int = 64) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B, N, C] -> [B, N, 1]"""
        return torch.sigmoid(self.layers(x))


class StomataPatchAdaptiveBypass(nn.Module):
    """Adaptive bypass that detects and enhances existing stomata, then fills missing ones.

    Three-stage process:
    1. Detect stomata positions from DINO features
    2. Enhance/strengthen existing stomata signals
    3. Predict and fill missing stomata based on context
    
    Input/Output shape: [B, N, C]
    """

    def __init__(
        self,
        embed_dim: int,
        bottleneck_dim: int = 64,
        alpha_init: float = 0.0,
        row_min_instances: float = 0.0,
        row_gate_temperature: float = 8.0,
        row_axis: str = "horizontal",
    ) -> None:
        super().__init__()

        # Stage 1: Stomata locator - detects where stomata are
        self.locator = StomataLocator(embed_dim, bottleneck_dim)

        # Stage 2: Strength head - enhances existing stomata
        self.strength_head = nn.Sequential(
            nn.Linear(embed_dim, bottleneck_dim),
            nn.GELU(),
            nn.Linear(bottleneck_dim, embed_dim),
        )

        # Stage 3: Fill head - predicts and fills missing stomata
        self.fill_head = nn.Sequential(
            nn.Linear(embed_dim, bottleneck_dim),
            nn.GELU(),
            nn.Linear(bottleneck_dim, embed_dim),
        )

        # Zero-init safety: both enhancement and filling start as no-op
        nn.init.zeros_(self.strength_head[-1].weight)
        nn.init.zeros_(self.strength_head[-1].bias)
        nn.init.zeros_(self.fill_head[-1].weight)
        nn.init.zeros_(self.fill_head[-1].bias)

        # Learnable residual gate; start from 0 to protect pretrained distribution
        self.alpha = nn.Parameter(torch.tensor(float(alpha_init), dtype=torch.float32))
        self.row_min_instances = float(row_min_instances)
        self.row_gate_temperature = float(row_gate_temperature)
        if row_axis not in {"horizontal", "vertical"}:
            raise ValueError(f"row_axis must be 'horizontal' or 'vertical', got {row_axis!r}")
        self.row_axis = row_axis
        self.last_stomata_prob = None
        self.last_row_counts = None
        self.last_row_counts_horizontal = None
        self.last_row_counts_vertical = None
        self.last_row_gate = None
        self.last_row_axis = row_axis
        self.last_prefix_tokens = None
        self.last_patch_hw = None
        self.last_row_diag_reason = None

    def _apply_row_consistency_gate(self, stomata_prob: torch.Tensor) -> torch.Tensor:
        self.last_row_counts = None
        self.last_row_counts_horizontal = None
        self.last_row_counts_vertical = None
        self.last_row_gate = None
        self.last_row_axis = self.row_axis
        self.last_prefix_tokens = None
        self.last_patch_hw = None
        self.last_row_diag_reason = None

        if stomata_prob.shape[1] <= 1:
            self.last_row_diag_reason = "not-enough-tokens"
            return stomata_prob

        split = _split_prefix_and_patch_tokens(stomata_prob)
        if split is None:
            self.last_row_diag_reason = f"no-square-patch-tail(total_tokens={int(stomata_prob.shape[1])})"
            return stomata_prob

        prefix_prob, patch_prob, h, w, prefix_len = split
        self.last_prefix_tokens = int(prefix_len)
        self.last_patch_hw = (int(h), int(w))

        b = patch_prob.shape[0]
        patch_2d = patch_prob.reshape(b, h, w, 1)

        # Directional counts for diagnostics and gating control.
        row_counts_h = patch_2d.sum(dim=2)
        row_counts_v = patch_2d.sum(dim=1)
        self.last_row_counts_horizontal = row_counts_h
        self.last_row_counts_vertical = row_counts_v

        # Soft row-level support: suppress rows whose expected instance count is too small.
        row_counts = row_counts_h if self.row_axis == "horizontal" else row_counts_v
        self.last_row_counts = row_counts

        if self.row_min_instances <= 0.0:
            self.last_row_gate = torch.ones_like(row_counts)
            self.last_row_diag_reason = "row-gate-disabled"
            return stomata_prob

        row_gate = torch.sigmoid((row_counts - self.row_min_instances) * self.row_gate_temperature)
        self.last_row_gate = row_gate
        if self.row_axis == "horizontal":
            patch_2d = patch_2d * row_gate.unsqueeze(2)
        else:
            patch_2d = patch_2d * row_gate.unsqueeze(1)

        patch_prob = patch_2d.reshape(b, h * w, 1)
        return torch.cat([prefix_prob, patch_prob], dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, N, C] DINO features
        
        Returns:
            [B, N, C] enhanced features with detected stomata strengthened and missing ones filled
        """
        # Stage 1: Detect stomata positions
        stomata_prob = self.locator(x)  # [B, N, 1]
        stomata_prob = self._apply_row_consistency_gate(stomata_prob)
        self.last_stomata_prob = stomata_prob

        # Stage 2: Enhance existing stomata
        strength = self.strength_head(x)  # [B, N, C]
        enhanced = x + strength * stomata_prob  # Only strengthen where stomata detected

        # Stage 3: Predict and fill missing stomata
        missing_pred = self.fill_head(enhanced)  # [B, N, C]
        missing_mask = 1.0 - stomata_prob  # [B, N, 1] - regions without detected stomata
        filled = enhanced + missing_pred * missing_mask  # Only fill where stomata missing

        # Apply residual gate with zero-init protection
        return self.alpha * filled


class BlockWithBypass(nn.Module):
    def __init__(self, base_block: nn.Module, bypass: StomataPatchAdaptiveBypass) -> None:
        super().__init__()
        self.base_block = base_block
        self.bypass = bypass

    def forward(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        y = self.base_block(x, *args, **kwargs)
        return self._apply_bypass(y)

    def _apply_bypass(self, y):
        if isinstance(y, torch.Tensor):
            return y + self.bypass(y)
        if isinstance(y, list):
            return [self._apply_bypass(item) for item in y]
        if isinstance(y, tuple):
            return tuple(self._apply_bypass(item) for item in y)
        raise TypeError(f"BlockWithBypass expected Tensor/list/tuple output, got {type(y)!r}")


def _get_vit_blocks(model: nn.Module) -> nn.ModuleList:
    blocks = getattr(model, "blocks", None)
    if blocks is None:
        raise AttributeError("Model has no 'blocks' attribute; expected a ViT-like DINO model.")
    if not isinstance(blocks, nn.ModuleList):
        raise TypeError("Model.blocks is not nn.ModuleList; unsupported layout for bypass injection.")
    return blocks


def inject_zero_init_bypass(
    model: nn.Module,
    block_index: int = 6,
    bottleneck_dim: int = 64,
    alpha_init: float = 0.0,
    row_min_instances: float = 0.0,
    row_gate_temperature: float = 8.0,
    row_axis: str = "horizontal",
) -> StomataPatchAdaptiveBypass:
    """Inject stomata-aware adaptive bypass into DINO model.
    
    Args:
        model: DINO ViT model with 'blocks' and 'embed_dim' attributes
        block_index: Which block to inject bypass into
        bottleneck_dim: Hidden dimension for locator/strength/fill heads
        alpha_init: Initial value for residual gate (0.0 = identity)
        row_min_instances: Suppress rows with expected stomata count below this value (0 disables)
        row_gate_temperature: Sharpness of row suppression gate
        row_axis: Gating direction; 'horizontal' counts across image rows, 'vertical' across columns
    
    Returns:
        StomataPatchAdaptiveBypass instance
    """
    blocks = _get_vit_blocks(model)
    if block_index < 0 or block_index >= len(blocks):
        raise IndexError(f"block_index={block_index} out of range [0, {len(blocks)-1}]")

    target = blocks[block_index]
    dim = getattr(model, "embed_dim", None)
    if dim is None:
        raise AttributeError("Model has no 'embed_dim' attribute; cannot infer bypass dimensions.")

    bypass = StomataPatchAdaptiveBypass(
        embed_dim=int(dim),
        bottleneck_dim=int(bottleneck_dim),
        alpha_init=float(alpha_init),
        row_min_instances=float(row_min_instances),
        row_gate_temperature=float(row_gate_temperature),
        row_axis=row_axis,
    )

    blocks[block_index] = BlockWithBypass(target, bypass)
    return bypass


def freeze_backbone_only_train_bypass(model: nn.Module) -> int:
    for p in model.parameters():
        p.requires_grad = False

    trainable = 0
    for name, p in model.named_parameters():
        if "bypass" in name:
            p.requires_grad = True
            trainable += p.numel()
    return trainable
