"""Shared scalar/tensor math for the Qwen4 oracles.

These routines intentionally use float64 and direct formulas.  They are not an
executable model path and must not be replaced with production kernels.
"""

from __future__ import annotations

import math

import torch


def as_f64(value: torch.Tensor) -> torch.Tensor:
    """Return represented tensor values in the oracle arithmetic type."""

    return value.detach().to(dtype=torch.float64, device="cpu")


def silu(value: torch.Tensor) -> torch.Tensor:
    value = as_f64(value)
    return value / (1.0 + torch.exp(-value))


def sigmoid(value: torch.Tensor) -> torch.Tensor:
    value = as_f64(value)
    return 1.0 / (1.0 + torch.exp(-value))


def linear(value: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """Linear projection for weights stored as ``[out_features, in_features]``."""

    return as_f64(value) @ as_f64(weight).transpose(-1, -2)


def _grouped_rmsnorm(
    value: torch.Tensor,
    gamma: torch.Tensor,
    *,
    group_size: int,
    eps: float = 1e-6,
) -> torch.Tensor:
    x = as_f64(value)
    represented_gamma = as_f64(gamma)
    if x.shape[-1] != represented_gamma.numel():
        raise ValueError("RMSNorm weight width must equal the input width")
    if x.shape[-1] % group_size:
        raise ValueError("RMSNorm width must be divisible by group_size")
    grouped = x.reshape(*x.shape[:-1], -1, group_size)
    inv_rms = torch.rsqrt(torch.mean(grouped * grouped, dim=-1, keepdim=True) + eps)
    normalized = (grouped * inv_rms).flatten(-2)
    return normalized * represented_gamma


def source_grouped_rmsnorm(
    value: torch.Tensor,
    zero_centered_weight: torch.Tensor,
    *,
    group_size: int,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Source-checkpoint grouped RMSNorm with zero-centered ``(1 + weight)``."""

    return _grouped_rmsnorm(
        value,
        1.0 + as_f64(zero_centered_weight),
        group_size=group_size,
        eps=eps,
    )


def actual_gguf_grouped_rmsnorm(
    value: torch.Tensor,
    gamma: torch.Tensor,
    *,
    group_size: int,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Actual-GGUF grouped RMSNorm with its already-folded direct gamma."""

    return _grouped_rmsnorm(value, gamma, group_size=group_size, eps=eps)


def ordinary_rmsnorm(
    value: torch.Tensor,
    weight: torch.Tensor,
    *,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Learned-scale RMSNorm (no unit offset), used by GDN's output norm."""

    x = as_f64(value)
    w = as_f64(weight)
    if x.shape[-1] != w.numel():
        raise ValueError("RMSNorm weight width must equal the input width")
    return x * torch.rsqrt(torch.mean(x * x, dim=-1, keepdim=True) + eps) * w


def _mrope_angles(
    positions: torch.Tensor,
    *,
    rotary_dim: int,
    theta: float,
    mrope_section: tuple[int, int, int] | None,
) -> torch.Tensor:
    if rotary_dim % 2:
        raise ValueError("rotary_dim must be even")
    position_values = as_f64(positions)
    frequencies = theta ** (-torch.arange(0, rotary_dim, 2, dtype=torch.float64) / rotary_dim)
    if position_values.ndim == 1:
        return position_values[:, None] * frequencies[None, :]
    if position_values.ndim != 2 or position_values.shape[0] != 3:
        raise ValueError("positions must have shape [T] or [3,T]")
    axes = position_values[:, :, None] * frequencies[None, None, :]
    angles = axes[0].clone()
    if mrope_section is None:
        raise ValueError("three-axis positions require mrope_section")
    if sum(mrope_section) != rotary_dim // 2:
        raise ValueError("mrope_section must cover rotary_dim / 2 frequencies")
    for axis, offset in ((1, 1), (2, 2)):
        angles[:, offset : mrope_section[axis] * 3 : 3] = axes[
            axis, :, offset : mrope_section[axis] * 3 : 3
        ]
    return angles


def partial_rope(
    value: torch.Tensor,
    positions: torch.Tensor,
    *,
    rotary_dim: int,
    theta: float = 10_000_000.0,
    mrope_section: tuple[int, int, int] | None = None,
) -> torch.Tensor:
    """Apply Qwen's rotate-half RoPE to the leading dimensions of ``[T,...,D]``."""

    x = as_f64(value)
    if rotary_dim > x.shape[-1]:
        raise ValueError("rotary_dim cannot exceed the head width")
    angles = _mrope_angles(
        positions,
        rotary_dim=rotary_dim,
        theta=theta,
        mrope_section=mrope_section,
    )
    broadcast = (angles.shape[0],) + (1,) * (x.ndim - 2) + (angles.shape[-1],)
    cos = torch.cos(angles).reshape(broadcast)
    sin = torch.sin(angles).reshape(broadcast)
    half = rotary_dim // 2
    first = x[..., :half]
    second = x[..., half:rotary_dim]
    rotated = torch.cat((first * cos - second * sin, second * cos + first * sin), dim=-1)
    return torch.cat((rotated, x[..., rotary_dim:]), dim=-1)


def ideal_softmax(value: torch.Tensor) -> torch.Tensor:
    """Stable float64 softmax over the final dimension."""

    x = as_f64(value)
    shifted = x - x.max(dim=-1, keepdim=True).values
    exponent = torch.exp(shifted)
    return exponent / exponent.sum(dim=-1, keepdim=True)


SQRT_128 = math.sqrt(128.0)
