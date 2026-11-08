from __future__ import annotations

import numpy as np
import pytest
import torch

from tools.artifact.layouts import decode_row_split_codes, dequantize_row_split
from tools.artifact.numeric import QuantFormat, get_format
from tools.convert.common.quantize import quantize_and_encode, quantize_matrix


def _oracle_codes(source: torch.Tensor, format_name: str) -> tuple[np.ndarray, np.ndarray]:
    """Quantize logical ``[N,K]`` binary32 rows with the registered grouped formula.

    Scales are ``RNE16(RNE32(max_abs / qmax))`` with a nonzero group raised to the
    smallest binary16 subnormal. Codes are ``RNE(RNE32(x * RNE32(1 / scale)))``
    clamped to ``[qmin, qmax]``; the K padding is zero.
    """

    format_spec = get_format(format_name)
    assert isinstance(format_spec, QuantFormat)
    rows, k = source.shape
    group = format_spec.group_size
    k_pad = -(-k // group) * group
    values = np.zeros((rows, k_pad), dtype=np.float32)
    values[:, :k] = source.numpy().astype(np.float32)
    grouped = values.reshape(rows, k_pad // group, group)
    max_abs = np.max(np.abs(grouped), axis=-1)
    scales = (max_abs.astype(np.float64) / float(format_spec.qmax)).astype(np.float32)
    scales = scales.astype(np.float16)
    scales[(scales == 0) & (max_abs > 0)] = np.float16(2.0**-24)
    reciprocal = np.zeros(scales.shape, dtype=np.float32)
    positive = scales > 0
    reciprocal[positive] = (1.0 / scales[positive].astype(np.float64)).astype(np.float32)
    # Products of two binary32 values are exact in binary64, so the explicit
    # cast supplies the specified binary32 rounding before ties-to-even.
    normalized = (grouped.astype(np.float64) * reciprocal.astype(np.float64)[..., None]).astype(
        np.float32
    )
    codes = np.clip(np.rint(normalized), format_spec.qmin, format_spec.qmax).astype(np.int8)
    return scales, codes


@pytest.mark.parametrize(
    ("format_name", "k"),
    (
        ("Q4G64_F16S", 65),
        ("Q5G64_F16S", 130),
        ("Q6G64_F16S", 129),
        ("W8G32_F16S", 33),
    ),
)
def test_stored_row_split_codes_match_independent_oracle(format_name: str, k: int) -> None:
    source = torch.linspace(-3.0, 4.0, 5 * k, dtype=torch.float32).reshape(5, k)
    payload = quantize_and_encode(source, format_name, device="cpu")
    stored_scales, stored_codes = decode_row_split_codes(payload, format_name, (5, k))
    expected_scales, expected_codes = _oracle_codes(source, format_name)

    assert np.array_equal(stored_scales.numpy().view(np.uint16), expected_scales.view(np.uint16))
    assert np.array_equal(stored_codes.numpy(), expected_codes)


def test_quantization_uses_stored_fp16_scale_and_zero_padding() -> None:
    weight = torch.tensor(
        [[-7.0, -1.0, 0.0, 1.0, 7.0, 3.5, -3.5, 0.25] + [0.0] * 57],
        dtype=torch.bfloat16,
    )
    quantized = quantize_matrix(weight, "Q4G64_F16S", device="cpu")
    assert quantized.codes.shape == (1, 2, 64)
    assert quantized.scales.shape == (1, 2)
    assert torch.count_nonzero(quantized.codes[0, 1]) == 0
    assert quantized.scales[0, 1] == 0

    payload = quantize_and_encode(weight, "Q4G64_F16S", device="cpu")
    scales, codes = decode_row_split_codes(payload, "Q4G64_F16S", tuple(weight.shape))
    assert torch.equal(scales, quantized.scales)
    assert torch.equal(codes, quantized.codes)
    decoded = dequantize_row_split(payload, "Q4G64_F16S", tuple(weight.shape), dtype=torch.float32)
    expected = (quantized.codes.float() * quantized.scales.float().unsqueeze(-1)).reshape(1, 128)[
        :, :65
    ]
    assert torch.equal(decoded, expected)


def test_quantization_uses_canonical_scale_rounding_on_cuda() -> None:
    # BF16 word 0x3636 divided by Q4 qmax is a known CUDA division boundary:
    # approximate device division rounds the FP16 scale to word 7, while the
    # ordered RNE oracle requires word 6.
    value = torch.tensor([0x3636], dtype=torch.int16).view(torch.bfloat16)
    weight = value.repeat(64).reshape(1, 64)
    cpu = quantize_matrix(weight, "Q4G64_F16S", device="cpu")
    assert int(cpu.scales.view(torch.int16)[0, 0]) == 6
    if torch.cuda.is_available():
        cuda = quantize_matrix(weight, "Q4G64_F16S", device="cuda")
        assert torch.equal(cuda.scales.cpu().view(torch.int16), cpu.scales.view(torch.int16))
        assert torch.equal(cuda.codes.cpu(), cpu.codes)


def test_quantization_uses_reciprocal_multiply_and_ties_to_even() -> None:
    words = torch.tensor([0x41A0B334, 0x417C7BFF], dtype=torch.int32).view(torch.float32)
    weight = torch.zeros((1, 64), dtype=torch.float32)
    weight[0, :2] = words
    quantized = quantize_matrix(weight, "Q4G64_F16S", device="cpu")
    assert int(quantized.scales.view(torch.int16)[0, 0]) == 0x41BD
    assert quantized.codes[0, 0, :2].tolist() == [7, 6]
    if torch.cuda.is_available():
        cuda = quantize_matrix(weight, "Q4G64_F16S", device="cuda")
        assert torch.equal(cuda.scales.cpu(), quantized.scales)
        assert torch.equal(cuda.codes.cpu(), quantized.codes)

    tie_weight = torch.zeros((1, 64), dtype=torch.float32)
    tie_weight[0, :7] = torch.tensor([7.0, 0.5, 1.5, 2.5, -0.5, -1.5, -2.5])
    ties = quantize_matrix(tie_weight, "Q4G64_F16S", device="cpu")
    assert ties.codes[0, 0, :7].tolist() == [7, 0, 2, 2, 0, -2, -2]


def test_nonzero_scale_underflow_uses_smallest_fp16_subnormal() -> None:
    weight = torch.zeros((1, 64), dtype=torch.float32)
    weight[0, 0] = torch.finfo(torch.float32).tiny
    quantized = quantize_matrix(weight, "Q4G64_F16S", device="cpu")
    assert int(quantized.scales.view(torch.int16)[0, 0]) == 1


@pytest.mark.parametrize(
    "value",
    (float("nan"), float("inf"), -float("inf"), torch.finfo(torch.float32).max),
)
def test_quantization_rejects_nonfinite_source_or_fp16_scale(value: float) -> None:
    weight = torch.full((1, 64), value, dtype=torch.float32)
    with pytest.raises(ValueError):
        quantize_matrix(weight, "Q4G64_F16S", device="cpu")
