"""Independent Gated Residual (hyper-connection) formulas."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import torch

from .common import (
    actual_gguf_grouped_rmsnorm,
    linear,
    sigmoid,
    silu,
    source_grouped_rmsnorm,
)


@dataclass(frozen=True, slots=True)
class GatedResidualRead:
    mixed: torch.Tensor
    normalized_branches: torch.Tensor
    original_branches: torch.Tensor
    injection_scales: torch.Tensor | None


GroupedRmsNorm = Callable[..., torch.Tensor]


def _read(
    branches: torch.Tensor,
    norm_weight: torch.Tensor,
    down_weight: torch.Tensor,
    up_weight: torch.Tensor,
    inject_weight: torch.Tensor | None,
    *,
    grouped_rmsnorm: GroupedRmsNorm,
    eps: float = 1e-6,
) -> GatedResidualRead:
    if branches.ndim < 2:
        raise ValueError("branches must have shape [..., branch_count, hidden]")
    branch_count, hidden = branches.shape[-2:]
    flat = branches.detach().to(dtype=torch.float64, device="cpu").flatten(-2)
    normalized = grouped_rmsnorm(
        flat,
        norm_weight,
        group_size=hidden,
        eps=eps,
    )
    low_rank = silu(linear(normalized, down_weight) / branch_count)
    input_gate = sigmoid(linear(low_rank, up_weight)).reshape(
        *branches.shape[:-2], branch_count, hidden
    )
    normalized_branches = normalized.reshape(
        *branches.shape[:-2], branch_count, hidden
    )
    mixed = torch.mean(input_gate * normalized_branches, dim=-2)
    injection_scales = None
    if inject_weight is not None:
        injection_scales = 2.0 * sigmoid(linear(normalized, inject_weight) / branch_count)
    return GatedResidualRead(
        mixed=mixed,
        normalized_branches=normalized_branches,
        original_branches=branches.detach().to(dtype=torch.float64, device="cpu"),
        injection_scales=injection_scales,
    )


def source_read(
    branches: torch.Tensor,
    norm_weight: torch.Tensor,
    down_weight: torch.Tensor,
    up_weight: torch.Tensor,
    inject_weight: torch.Tensor | None,
    *,
    eps: float = 1e-6,
) -> GatedResidualRead:
    """Evaluate GR read from source-checkpoint zero-centered norm weights."""

    return _read(
        branches,
        norm_weight,
        down_weight,
        up_weight,
        inject_weight,
        grouped_rmsnorm=source_grouped_rmsnorm,
        eps=eps,
    )


def actual_gguf_read(
    branches: torch.Tensor,
    norm_gamma: torch.Tensor,
    down_weight: torch.Tensor,
    up_weight: torch.Tensor,
    inject_weight: torch.Tensor | None,
    *,
    eps: float = 1e-6,
) -> GatedResidualRead:
    """Evaluate GR read from actual-GGUF already-folded norm gamma."""

    return _read(
        branches,
        norm_gamma,
        down_weight,
        up_weight,
        inject_weight,
        grouped_rmsnorm=actual_gguf_grouped_rmsnorm,
        eps=eps,
    )


def inject(
    branches: torch.Tensor,
    block_output: torch.Tensor,
    write_scale: torch.Tensor,
) -> torch.Tensor:
    """Apply GR inject from its represented public inputs."""

    if branches.ndim < 2:
        raise ValueError("branches must have shape [..., branch_count, hidden]")
    if write_scale.shape != branches.shape[:-1]:
        raise ValueError("write_scale must have shape [..., branch_count]")
    if block_output.shape != branches.shape[:-2] + branches.shape[-1:]:
        raise ValueError("block_output must have shape [..., hidden]")
    update = block_output.detach().to(dtype=torch.float64, device="cpu").unsqueeze(-2)
    represented_scale = write_scale.detach().to(dtype=torch.float64, device="cpu")
    represented_branches = branches.detach().to(dtype=torch.float64, device="cpu")
    return represented_branches + represented_scale.unsqueeze(-1) * update


def source_final_read(
    branches: torch.Tensor,
    norm_weight: torch.Tensor,
    down_weight: torch.Tensor,
    up_weight: torch.Tensor,
    *,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Read-only final mixer used before the language-model head."""

    return source_read(
        branches, norm_weight, down_weight, up_weight, None, eps=eps
    ).mixed


def actual_gguf_final_read(
    branches: torch.Tensor,
    norm_gamma: torch.Tensor,
    down_weight: torch.Tensor,
    up_weight: torch.Tensor,
    *,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Read-only final mixer from actual-GGUF already-folded norm gamma."""

    return actual_gguf_read(
        branches, norm_gamma, down_weight, up_weight, None, eps=eps
    ).mixed


__all__ = [
    "GatedResidualRead",
    "actual_gguf_final_read",
    "actual_gguf_read",
    "inject",
    "source_final_read",
    "source_read",
]
