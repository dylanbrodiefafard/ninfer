"""DFlash exact storage and independent formula/state witnesses.

Predeclared criteria: exact words/layout/positions; FP64 identities atol=rtol=1e-12.
These are oracle/inventory tests, not admission of quantized model quality.
"""

from __future__ import annotations

import math
import struct

import pytest
import torch

from tools.artifact.layouts import encode_nvfp4
from tools.parity.qwen4.native_dflash_fixture import SHAPES, validate_header
from tools.reference.qwen4.dflash import (
    Config, attention, block_inputs, decode_nvfp4_weight, decode_nvfp4_words_exact, forward,
)


def weights(config: Config) -> dict[str, torch.Tensor]:
    c = config
    generator = torch.Generator().manual_seed(8173)
    def matrix(n, k):
        return (torch.randn(n, k, generator=generator) / math.sqrt(k) * .2).bfloat16()
    def norm(n):
        return (.5 + torch.rand(n, generator=generator)).bfloat16()
    result = {"fc.weight": matrix(c.hidden, c.taps * c.hidden),
              "hidden_norm.weight": norm(c.hidden), "norm.weight": norm(c.hidden)}
    for i in range(c.layers):
        p = f"layers.{i}."
        result.update({p + name: value for name, value in {
            "input_layernorm.weight": norm(c.hidden),
            "post_attention_layernorm.weight": norm(c.hidden),
            "self_attn.q_proj.weight": matrix(c.heads * c.head_dim, c.hidden),
            "self_attn.k_proj.weight": matrix(c.kv_heads * c.head_dim, c.hidden),
            "self_attn.v_proj.weight": matrix(c.kv_heads * c.head_dim, c.hidden),
            "self_attn.o_proj.weight": matrix(c.hidden, c.heads * c.head_dim),
            "self_attn.q_norm.weight": norm(c.head_dim),
            "self_attn.k_norm.weight": norm(c.head_dim),
            "mlp.gate_proj.weight": matrix(c.intermediate, c.hidden),
            "mlp.up_proj.weight": matrix(c.intermediate, c.hidden),
            "mlp.down_proj.weight": matrix(c.hidden, c.intermediate),
        }.items()})
    return result


def test_nvfp4_exact_all_codes_scales_and_address_tiles():
    n, k = 256, 128
    packed = torch.arange(n * k // 2).reshape(n, k // 2).to(torch.uint8)
    scales = (torch.arange(n * k // 16).reshape(n, k // 16) % 127).to(torch.uint8)
    divisor = torch.tensor(13.125, dtype=torch.float32)
    payload = encode_nvfp4(packed, scales, divisor, (n, k))
    actual_codes, actual_scales, actual_divisor = decode_nvfp4_words_exact(payload, (n, k))
    assert torch.equal(actual_codes, packed)
    assert torch.equal(actual_scales, scales)
    assert torch.equal(actual_divisor.view(torch.int32), divisor.view(torch.int32))
    actual = decode_nvfp4_weight(payload, (n, k))
    magnitude = (0., .5, 1., 1.5, 2., 3., 4., 6.)
    for row in (0, 31, 32, 63, 64, 95, 96, 127, 128, 255):
        for col in range(k):
            word = int(packed[row, col // 2]) >> (4 * (col % 2)) & 15
            scale = int(scales[row, col // 16])
            exponent, fraction = scale >> 3, scale & 7
            scale_value = math.ldexp(fraction, -9) if exponent == 0 else math.ldexp(1 + fraction / 8, exponent - 7)
            expected = (-1 if word & 8 else 1) * magnitude[word & 7] * scale_value / float(divisor)
            assert float(actual[row, col]) == expected
            if expected == 0:
                assert bool(torch.signbit(actual[row, col])) == bool(word & 8)


def test_nvfp4_rejects_illegal_scale_or_divisor():
    shape = (128, 64)
    payload = bytearray(encode_nvfp4(torch.zeros(128, 32, dtype=torch.uint8),
                                  torch.ones(128, 4, dtype=torch.uint8), torch.tensor(1.), shape))
    payload[128 * 32] = 127
    with pytest.raises(ValueError, match="scale"):
        decode_nvfp4_weight(payload, shape)
    payload[128 * 32] = 1
    struct.pack_into("<f", payload, len(payload) - 4, 0.)
    with pytest.raises(ValueError, match="divisor"):
        decode_nvfp4_weight(payload, shape)


def test_source_inventory_cannot_substitute_dense_dflash2_or_extra_codebook():
    cursor = 0
    header = {}
    for name, shape in SHAPES.items():
        size = math.prod(shape) * 2
        header[name] = dict(dtype="BF16", shape=list(shape), data_offsets=[cursor, cursor + size])
        cursor += size
    validate_header(header, 6136)
    header["codebook.weight"] = dict(dtype="BF16", shape=[248320, 256], data_offsets=[cursor, cursor + 127139840])
    with pytest.raises(ValueError, match="58 pinned"):
        validate_header(header, 6136)


def test_anchor_query_zero_predicts_next_without_extra_query():
    noise, positions = block_inputs(torch.tensor([7., -2.]), torch.tensor([.5, -9.]), 33, 7)
    assert torch.equal(positions, torch.arange(33, 40))
    assert torch.equal(noise[0], torch.tensor([7., -2.], dtype=torch.float64))
    assert torch.equal(noise[1:], torch.tensor([.5, -9.], dtype=torch.float64).expand(6, -1))


def test_noncausal_noise_context_boundary_and_gqa_hand_formula():
    # Q=0 gives exactly uniform probabilities. Context at anchor is forbidden;
    # the second noise key must still contribute to the first query.
    q = torch.zeros(2, 4, 2, dtype=torch.float64)
    k = torch.zeros(4, 2, 2, dtype=torch.float64)
    v = torch.tensor([[[1., 2.], [10., 20.]], [[1000., 2000.], [9000., 8000.]],
                      [[3., 6.], [30., 60.]], [[8., 13.], [80., 130.]]], dtype=torch.float64)
    result = attention(q, k, v, torch.tensor([4, 5]), anchor=5, noise_count=2)
    expected = torch.tensor([[4., 7.], [4., 7.], [40., 70.], [40., 70.]], dtype=torch.float64)
    torch.testing.assert_close(result, expected[None].expand(2, -1, -1), rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("count", (1, 4, 5, 7))
@pytest.mark.parametrize("materialize", (False, True))
def test_five_layer_accepted_context_cache_equals_recompute_and_discards_noise(count, materialize):
    c = Config(hidden=8, intermediate=12, heads=4, kv_heads=2, head_dim=4)
    w = weights(c)
    gen = torch.Generator().manual_seed(319)
    features = torch.randn(11, 40, generator=gen).bfloat16()
    anchor_embed = torch.randn(8, generator=gen).bfloat16()
    mask_embed = torch.randn(8, generator=gen).bfloat16()
    noise, query_pos = block_inputs(anchor_embed, mask_embed, 9, count)
    first = forward(features[:3], noise, torch.arange(3), query_pos, w, config=c,
                    materialize_public_bf16=materialize)
    incremental = forward(features[3:9], noise, torch.arange(3, 9), query_pos, w, config=c,
                          cache=first.cache, materialize_public_bf16=materialize)
    whole = forward(features[:9], noise, torch.arange(9), query_pos, w, config=c,
                    materialize_public_bf16=materialize)
    torch.testing.assert_close(incremental.hidden, whole.hidden, rtol=1e-12, atol=1e-12)
    # Two lookahead target rows are not yet accepted; poison must be invisible.
    poisoned = features.clone()
    poisoned[9:] = 1000
    suffix = forward(poisoned, noise, torch.arange(11), query_pos, w, config=c,
                     materialize_public_bf16=materialize)
    torch.testing.assert_close(suffix.hidden, whole.hidden, rtol=1e-12, atol=1e-12)
    for i in range(5):
        assert torch.equal(whole.cache[i].positions, torch.arange(9))
        assert torch.equal(suffix.cache[i].positions, torch.arange(9))
        torch.testing.assert_close(incremental.cache[i].keys, whole.cache[i].keys, rtol=1e-12, atol=1e-12)
        torch.testing.assert_close(incremental.cache[i].values, whole.cache[i].values, rtol=1e-12, atol=1e-12)
        assert len(whole.cache[i].keys) == 9  # not 9+count noise rows


def test_context_norm_is_plain_and_not_query_layer_norm():
    c = Config(hidden=8, intermediate=12, heads=4, kv_heads=2, head_dim=4)
    w = weights(c)
    features = torch.arange(80).reshape(2, 40).bfloat16() / 64
    noise, query_pos = block_inputs(torch.arange(8).bfloat16(), torch.ones(8).bfloat16(), 4, 2)
    result = forward(features, noise, torch.tensor([1, 3]), query_pos, w, config=c)
    fused = features.double() @ w["fc.weight"].double().t()
    expected = fused / torch.sqrt((fused * fused).mean(-1, keepdim=True) + 1e-6) * w["hidden_norm.weight"].double()
    torch.testing.assert_close(result.context, expected, rtol=1e-12, atol=1e-12)
    values = expected @ w["layers.0.self_attn.v_proj.weight"].double().t()
    torch.testing.assert_close(result.layers[0]["context_values"], values.reshape(2, 2, 4), rtol=1e-12, atol=1e-12)
    assert not torch.allclose(expected, fused / torch.sqrt((fused * fused).mean(-1, keepdim=True) + 1e-6)
                              * (1 + w["hidden_norm.weight"].double()))


def test_empty_context_and_five_layer_residual_formula():
    c = Config(hidden=8, intermediate=12, heads=4, kv_heads=2, head_dim=4)
    w = weights(c)
    for i in range(5):
        w[f"layers.{i}.self_attn.o_proj.weight"].zero_()
        w[f"layers.{i}.mlp.down_proj.weight"].zero_()
    noise, positions = block_inputs(torch.arange(8).bfloat16(), -torch.arange(8).bfloat16(), 0, 7)
    result = forward(torch.empty(0, 40), noise, torch.empty(0, dtype=torch.int64), positions, w, config=c)
    expected = noise / torch.sqrt((noise * noise).mean(-1, keepdim=True) + 1e-6) * w["norm.weight"].double()
    torch.testing.assert_close(result.hidden, expected, rtol=1e-12, atol=1e-12)
    assert all(len(cache.positions) == 0 for cache in result.cache)
