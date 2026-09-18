"""Exact source-word preservation and independent expert-plane addressing."""
import struct

import pytest
import torch

from tools.artifact.layouts import (
    decode_nvfp4_expert_words, encode_nvfp4_experts, encoded_size,
    expert_block_scale_geometry,
)


@pytest.mark.parametrize("shape", [(3, 128, 64), (2, 640, 2560), (2, 2560, 640)])
def test_expert_source_words_and_independent_offsets(shape):
    e, n, k = shape
    codes = (torch.arange(e * n * k // 2) % 256).to(torch.uint8).reshape(e, n, k // 2)
    scales = (torch.arange(e * n * k // 16) % 127).to(torch.uint8).reshape(e, n, k // 16)
    weights = torch.tensor([0.00006612142169615254 * (i + 1) for i in range(e)], dtype=torch.float32)
    inputs = torch.tensor([0.0013892764691263437 * (i + 1) for i in range(e)], dtype=torch.float32)
    payload = encode_nvfp4_experts(codes, scales, weights, inputs, shape)
    g = expert_block_scale_geometry("NVFP4_EXPERT_F32M", shape)
    assert len(payload) == e * n * k * 9 // 16 + 8 * e
    assert encoded_size("expert-blockscale-k16-m128x4-v1", "NVFP4_EXPERT_F32M", shape) == len(payload)
    for expert in range(e):
        for row in (0, 31, 32, 127, n - 1):
            for group in (0, k // 16 - 1):
                # Formula independent of the encoder's reshape/permute operations.
                offset = expert * n * k // 16 + (row // 128) * (k // 64) * 512
                offset += (group // 4) * 512 + (row % 32) * 16 + ((row % 128) // 32) * 4 + group % 4
                assert payload[g.scale_plane_offset + offset] == scales[expert, row, group]
        assert payload[g.weight_multiplier_offset + 4 * expert:g.weight_multiplier_offset + 4 * expert + 4] == struct.pack("<f", weights[expert])
        assert payload[g.input_multiplier_offset + 4 * expert:g.input_multiplier_offset + 4 * expert + 4] == struct.pack("<f", inputs[expert])
    actual = decode_nvfp4_expert_words(payload, shape)
    for expected, decoded in zip((codes, scales, weights, inputs), actual):
        assert torch.equal(expected.view(torch.uint8), decoded.view(torch.uint8))


def test_expert_codec_rejects_invalid_source_scales_and_rank():
    shape = (2, 128, 64)
    codes = torch.zeros((2, 128, 32), dtype=torch.uint8)
    scales = torch.zeros((2, 128, 4), dtype=torch.uint8)
    ones = torch.ones(2)
    for invalid in (0.0, -1.0, float("inf"), float("nan")):
        bad = torch.tensor([1.0, invalid])
        with pytest.raises(ValueError):
            encode_nvfp4_experts(codes, scales, bad, ones, shape)
        with pytest.raises(ValueError):
            encode_nvfp4_experts(codes, scales, ones, bad, shape)
    for invalid in (0x7F, 0x80, 0xFF):
        scales[1, 0, 0] = invalid
        with pytest.raises(ValueError):
            encode_nvfp4_experts(codes, scales, ones, ones, shape)
    with pytest.raises(ValueError):
        expert_block_scale_geometry("NVFP4_EXPERT_F32M", (128, 64))
