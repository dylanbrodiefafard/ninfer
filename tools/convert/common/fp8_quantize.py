"""Offline row-scaled E4M3FN encoding from represented BF16 matrices."""

import torch

from tools.artifact.layouts import encode_fp8_row_scaled


def encode_source_fp8(weights: torch.Tensor) -> bytes:
    if weights.dtype != torch.bfloat16 or weights.ndim != 2:
        raise ValueError('source must be a BF16 matrix')
    values = weights.float()
    if not torch.isfinite(values).all():
        raise ValueError('nonfinite source weight')
    maximum = values.abs().amax(dim=1)
    scale = (maximum/448).to(torch.bfloat16)
    if ((maximum != 0) & (scale == 0)).any():
        raise ValueError('BF16 row-scale underflow')
    denominator = torch.where(scale == 0, torch.ones_like(scale), scale).float()
    codes = (values/denominator[:, None]).clamp(-448, 448).to(torch.float8_e4m3fn).view(torch.uint8)
    return encode_fp8_row_scaled(codes, scale, tuple(weights.shape))
