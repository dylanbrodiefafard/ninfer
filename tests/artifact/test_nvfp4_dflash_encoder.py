from __future__ import annotations

import torch

from tools.convert.common.nvfp4_quantize import (
    decode_nvfp4_logical,
    quantize_nvfp4_matrix,
)


def test_nvfp4_encoder_matches_linear_decode_contract() -> None:
    torch.manual_seed(0)
    source = torch.randn(128, 64, dtype=torch.float32) * 0.05
    source[0, 0] = 2.5
    source[17, 33] = -1.75
    packed, scales, divisor = quantize_nvfp4_matrix(source)
    reconstructed = decode_nvfp4_logical(packed, scales, divisor)
    assert torch.isfinite(reconstructed).all()
    amax = float(source.abs().amax())
    assert float(reconstructed.abs().amax()) < 2.0 * amax
    flat_s = source.reshape(-1)
    flat_r = reconstructed.reshape(-1)
    cosine = float(torch.nn.functional.cosine_similarity(flat_s, flat_r, dim=0))
    assert cosine > 0.99

    inverted = decode_nvfp4_logical(packed, scales, 1.0 / divisor)
    assert float(inverted.abs().amax()) > 10.0 * amax
