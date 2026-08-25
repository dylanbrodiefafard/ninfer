"""DFlash2-owned BF16→NVFP4 encoder.

NInfer has no canonical source-to-NVFP4 encoder. This profile is the two-level
max-abs used to append DFlash2 companions. Stored ``d_w`` matches the A16 Linear
contract ``W = e2m1(c) * e4m3(s) / d_w``: one positive FP32 weight divisor for
the matrix, one nonnegative finite E4M3FN scale per K=16 group, and E2M1 codes.
"""

from __future__ import annotations

from typing import Sequence

import torch

from tools.artifact.layouts import encode_nvfp4
from tools.artifact.numeric import decode_e2m1_word, decode_e4m3fn_word

ENCODER_PROFILE = "DFLASH2_NVFP4_MAXABS_TWOLEVEL_V2"
_E2M1_MAX = 6.0
_E4M3_MAX = 448.0
_GROUP = 16
_E2M1_MAGNITUDES = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)


def _quantize_e2m1_codes(values: torch.Tensor) -> torch.Tensor:
    magnitudes = torch.tensor(_E2M1_MAGNITUDES, device=values.device, dtype=torch.float32)
    abs_values = values.abs()
    index = (abs_values.unsqueeze(-1) - magnitudes).abs().argmin(dim=-1).to(torch.uint8)
    codes = index.clone()
    codes = torch.where(values < 0, codes | torch.tensor(0x08, dtype=torch.uint8, device=values.device), codes)
    return codes


def quantize_nvfp4_matrix(weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Quantize a contiguous logical ``[N,K]`` matrix. Returns packed codes, E4M3 scales, d_w.

    ``d_w`` is the kernel denominator: ``e2m1 * e4m3 / d_w`` reconstructs the source.
    It is ``(6 * 448) / amax`` so the global-amax group uses scale 448.
    """

    if weight.dim() != 2:
        raise ValueError("NVFP4 quantization requires a rank-2 matrix")
    rows, columns = int(weight.shape[0]), int(weight.shape[1])
    if rows <= 0 or columns <= 0 or rows % 128 != 0 or columns % 64 != 0:
        raise ValueError("NVFP4 quantization requires N%128=0 and K%64=0")
    source = weight.detach().to(dtype=torch.float32, memory_format=torch.contiguous_format)
    if not torch.isfinite(source).all():
        raise ValueError("NVFP4 source contains NaN or infinity")

    amax = source.abs().amax()
    divisor = (
        torch.tensor(1.0, dtype=torch.float32, device=source.device)
        if float(amax) == 0.0
        else ((_E2M1_MAX * _E4M3_MAX) / amax).clamp(min=torch.finfo(torch.float32).tiny)
    )
    groups = source.view(rows, columns // _GROUP, _GROUP)
    group_amax = groups.abs().amax(dim=-1)
    scale_fp32 = (group_amax * divisor / _E2M1_MAX).clamp(min=0.0, max=_E4M3_MAX)
    scale_e4m3 = scale_fp32.to(torch.float8_e4m3fn)
    scale_words = scale_e4m3.view(torch.uint8).contiguous()
    illegal = ((scale_words & 0x80) != 0) | (scale_words == 0x7F)
    if bool(illegal.any()):
        raise ValueError("NVFP4 encoder emitted a negative or NaN E4M3FN scale")
    decoded = scale_e4m3.to(torch.float32)
    safe = torch.where(decoded > 0, decoded, torch.ones_like(decoded))
    quantized = groups * divisor / safe.unsqueeze(-1)
    codes = _quantize_e2m1_codes(quantized)
    packed = (codes[..., 0::2] | (codes[..., 1::2] << 4)).to(torch.uint8)
    packed = packed.contiguous().cpu().reshape(rows, columns // 2)
    return packed, scale_words.cpu().reshape(rows, columns // _GROUP), divisor.detach().cpu()


def decode_nvfp4_logical(packed: torch.Tensor, scales: torch.Tensor, divisor: torch.Tensor) -> torch.Tensor:
    """Host decode of ``e2m1 * e4m3 / d_w`` matching A16 Linear / codebook gather."""

    rows, packed_k = int(packed.shape[0]), int(packed.shape[1])
    columns = packed_k * 2
    packed_u8 = packed.detach().to(dtype=torch.uint8, device="cpu").contiguous()
    scale_u8 = scales.detach().to(dtype=torch.uint8, device="cpu").contiguous()
    d_w = float(divisor.detach().reshape(()).cpu())
    if d_w <= 0.0:
        raise ValueError("NVFP4 divisor must be positive")
    inv_dw = 1.0 / d_w
    e2m1 = torch.empty((16,), dtype=torch.float32)
    for word in range(16):
        e2m1[word] = decode_e2m1_word(word)
    e4m3 = torch.empty((128,), dtype=torch.float32)
    for word in range(128):
        e4m3[word] = decode_e4m3fn_word(word)
    lo = e2m1[(packed_u8 & 0x0F).long()]
    hi = e2m1[(packed_u8 >> 4).long()]
    codes = torch.empty((rows, columns), dtype=torch.float32)
    codes[:, 0::2] = lo
    codes[:, 1::2] = hi
    group_scale = e4m3[scale_u8.long()].repeat_interleave(_GROUP, dim=1)
    return codes * group_scale * inv_dw


def encode_nvfp4_from_bf16(weight: torch.Tensor, shape: Sequence[int]) -> bytes:
    if tuple(int(dim) for dim in weight.shape) != tuple(int(dim) for dim in shape):
        raise ValueError(f"NVFP4 source shape {tuple(weight.shape)} does not match {tuple(shape)}")
    packed, scales, divisor = quantize_nvfp4_matrix(weight)
    return encode_nvfp4(packed, scales, divisor, shape)
