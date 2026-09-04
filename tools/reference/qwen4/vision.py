"""Independent Qwen4 Vision patch-merger oracle.

The checkpoint profile is 1152 -> 4608 -> 4608 -> 2560.  The
implementation is shape-parameterized so tests can exercise the complete
formula without allocating the real matrices.  ``patch_groups`` names the
logical input layout explicitly: each group is a row-major 2x2 patch tile.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch

from .common import as_f64, linear


@dataclass(frozen=True, slots=True)
class VisionPatchMergerResult:
    """Observable oracle stages for qualifying a fused merger route."""

    output: torch.Tensor
    normalized_tokens: torch.Tensor
    merged_tokens: torch.Tensor
    fc1_pre_activation: torch.Tensor
    fc1_activation: torch.Tensor


def learned_bias_layer_norm(
    value: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    *,
    eps: float = 1e-6,
) -> torch.Tensor:
    """LayerNorm with population variance and learned scale and bias."""

    x = as_f64(value)
    w = as_f64(weight)
    b = as_f64(bias)
    if w.shape != (x.shape[-1],) or b.shape != w.shape:
        raise ValueError("LayerNorm weight and bias must equal the patch width")
    centered = x - x.mean(dim=-1, keepdim=True)
    inv_std = torch.rsqrt(torch.mean(centered * centered, dim=-1, keepdim=True) + eps)
    return centered * inv_std * w + b


def arrange_2x2(patch_groups: torch.Tensor) -> torch.Tensor:
    """Concatenate each logical 2x2 group in TL, TR, BL, BR order.

    The pinned checkpoint consumer receives this merge-major order as four
    consecutive patch tokens and implements the same transform with a view.
    """

    value = as_f64(patch_groups)
    if value.ndim != 4 or value.shape[1:3] != (2, 2):
        raise ValueError("patch_groups must have shape [groups,2,2,patch_width]")
    return value.reshape(value.shape[0], 4 * value.shape[-1])


def exact_gelu(value: torch.Tensor) -> torch.Tensor:
    """Exact GELU used by ``nn.GELU(approximate='none')``."""

    x = as_f64(value)
    return 0.5 * x * (1.0 + torch.erf(x / math.sqrt(2.0)))


def patch_merger(
    patch_groups: torch.Tensor,
    norm_weight: torch.Tensor,
    norm_bias: torch.Tensor,
    fc1_weight: torch.Tensor,
    fc1_bias: torch.Tensor,
    fc2_weight: torch.Tensor,
    fc2_bias: torch.Tensor,
    *,
    eps: float = 1e-6,
) -> VisionPatchMergerResult:
    """Evaluate per-patch LayerNorm, 2x2 merge, and the two biased linears."""

    patches = as_f64(patch_groups)
    if patches.ndim != 4 or patches.shape[1:3] != (2, 2):
        raise ValueError("patch_groups must have shape [groups,2,2,patch_width]")
    patch_width = patches.shape[-1]
    merge_width = 4 * patch_width
    if fc1_weight.shape != (merge_width, merge_width):
        raise ValueError("fc1_weight must have shape [4*patch_width,4*patch_width]")
    if fc1_bias.shape != (merge_width,):
        raise ValueError("fc1_bias must have shape [4*patch_width]")
    if fc2_weight.ndim != 2 or fc2_weight.shape[1] != merge_width:
        raise ValueError("fc2_weight must have shape [output_width,4*patch_width]")
    if fc2_bias.shape != (fc2_weight.shape[0],):
        raise ValueError("fc2_bias must have shape [output_width]")

    normalized = learned_bias_layer_norm(
        patches,
        norm_weight,
        norm_bias,
        eps=eps,
    )
    merged = arrange_2x2(normalized)
    fc1_pre_activation = linear(merged, fc1_weight) + as_f64(fc1_bias)
    fc1_activation = exact_gelu(fc1_pre_activation)
    output = linear(fc1_activation, fc2_weight) + as_f64(fc2_bias)
    return VisionPatchMergerResult(
        output=output,
        normalized_tokens=normalized,
        merged_tokens=merged,
        fc1_pre_activation=fc1_pre_activation,
        fc1_activation=fc1_activation,
    )


__all__ = [
    "VisionPatchMergerResult",
    "arrange_2x2",
    "exact_gelu",
    "learned_bias_layer_norm",
    "patch_merger",
]
