"""BF16-boundary tensor operators for the Vision reference schedule.

Each operator accumulates in the library's natural precision and rounds its
public output to BF16, matching the Op boundary that the parity campaign
compares.
"""

from __future__ import annotations

import torch


def linear(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    return (x.to(torch.bfloat16) @ weight.to(torch.bfloat16).t()).to(torch.bfloat16)


def residual_add(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return (x.float() + y.float()).to(torch.bfloat16)


__all__ = ["linear", "residual_add"]
