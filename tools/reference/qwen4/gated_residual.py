"""Independent Gated Residual (hyper-connection) formulas."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from .common import grouped_zero_centered_rmsnorm, linear, sigmoid, silu


@dataclass(frozen=True, slots=True)
class GatedResidualRead:
    mixed: torch.Tensor
    normalized_branches: torch.Tensor
    original_branches: torch.Tensor
    injection_scales: torch.Tensor | None


def read(
    branches: torch.Tensor,
    norm_weight: torch.Tensor,
    down_weight: torch.Tensor,
    up_weight: torch.Tensor,
    inject_weight: torch.Tensor | None,
    *,
    eps: float = 1e-6,
) -> GatedResidualRead:
    """Evaluate the GR read and optional scalar-per-branch write gate."""

    if branches.ndim < 2:
        raise ValueError("branches must have shape [..., branch_count, hidden]")
    branch_count, hidden = branches.shape[-2:]
    flat = branches.detach().to(dtype=torch.float64, device="cpu").flatten(-2)
    normalized = grouped_zero_centered_rmsnorm(
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


def final_read(
    branches: torch.Tensor,
    norm_weight: torch.Tensor,
    down_weight: torch.Tensor,
    up_weight: torch.Tensor,
    *,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Read-only final mixer used before the language-model head."""

    return read(branches, norm_weight, down_weight, up_weight, None, eps=eps).mixed


__all__ = ["GatedResidualRead", "final_read", "inject", "read"]
