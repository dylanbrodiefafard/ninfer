"""Independent FP64 formula for the Qwen4 PixelML DeepSpec DFlash companion.

No upstream/production forward calls, private BF16 staging casts, or target
execution. Inputs are represented source features/embedding rows and weights.
The exact fixture has five layers; small shapes below are solely oracle tests.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import struct
from typing import Mapping

import torch

from .common import as_f64, ideal_softmax, linear, ordinary_rmsnorm, partial_rope, silu


def decode_nvfp4_words_exact(payload: bytes, shape: tuple[int, int]) -> tuple[torch.Tensor, ...]:
    """Exact independent address formula for blockscale-k16-m128x4-v1."""
    n, k = shape
    if n % 128 or k % 64:
        raise ValueError("NVFP4 matrix must have N%128=0,K%64=0")
    code_bytes = n * k // 2
    scale_base = (code_bytes + 255) // 256 * 256
    groups = k // 16
    divisor_base = scale_base + n * groups
    if len(payload) != divisor_base + 4:
        raise ValueError("NVFP4 payload length mismatch")
    raw = torch.frombuffer(bytearray(payload), dtype=torch.uint8)
    codes = raw[:code_bytes].clone().reshape(n, k // 2)
    row = torch.arange(n, dtype=torch.int64)[:, None]
    group = torch.arange(groups, dtype=torch.int64)[None, :]
    address = (((((row // 128) * (groups // 4) + group // 4) * 32
                  + row % 32) * 4 + (row % 128) // 32) * 4 + group % 4)
    scales = raw[scale_base + address]
    if bool(((scales >= 128) | (scales == 127)).any()):
        raise ValueError("NVFP4 scale must be nonnegative finite E4M3")
    divisor = torch.tensor(struct.unpack_from("<f", payload, divisor_base)[0], dtype=torch.float32)
    if not math.isfinite(float(divisor)) or not float(divisor) > 0:
        raise ValueError("NVFP4 divisor must be positive finite FP32")
    return codes, scales, divisor


def decode_nvfp4_weight(payload: bytes, shape: tuple[int, int]) -> torch.Tensor:
    """Decode each signed nibble times exact E4M3 scale / stored FP32 divisor in FP64."""
    packed, scales, divisor = decode_nvfp4_words_exact(payload, shape)
    n, k = shape
    nibbles = torch.stack((packed & 15, packed >> 4), dim=-1).reshape(n, k).long()
    magnitudes = torch.tensor((0., .5, 1., 1.5, 2., 3., 4., 6.), dtype=torch.float64)
    coefficients = magnitudes[nibbles & 7] * torch.where(nibbles < 8, 1., -1.).double()
    exponent = (scales.long() >> 3)
    fraction = (scales.long() & 7).double()
    decoded_scales = torch.where(exponent == 0, fraction * 2. ** -9,
                                 (1. + fraction / 8.) * torch.pow(2., exponent.double() - 7.))
    return coefficients * decoded_scales.repeat_interleave(16, dim=-1) / float(divisor)


@dataclass(frozen=True)
class Config:
    hidden: int = 2560
    intermediate: int = 7680
    heads: int = 24
    kv_heads: int = 2
    head_dim: int = 256
    layers: int = 5
    taps: int = 5
    epsilon: float = 1e-6
    theta: float = 1e7


@dataclass(frozen=True)
class LayerCache:
    keys: torch.Tensor
    values: torch.Tensor
    positions: torch.Tensor


@dataclass(frozen=True)
class Result:
    hidden: torch.Tensor
    context: torch.Tensor
    layers: tuple[dict[str, torch.Tensor], ...]
    cache: tuple[LayerCache, ...]


def block_inputs(anchor_embedding: torch.Tensor, mask_embedding: torch.Tensor,
                 anchor: int, count: int) -> tuple[torch.Tensor, torch.Tensor]:
    if count < 1 or anchor < 0 or anchor_embedding.ndim != 1 or mask_embedding.shape != anchor_embedding.shape:
        raise ValueError("invalid DFlash anchor/mask embedding or block length")
    return (torch.cat((as_f64(anchor_embedding)[None],
                       as_f64(mask_embedding)[None].expand(count - 1, -1)), dim=0),
            torch.arange(anchor, anchor + count, dtype=torch.int64))


def attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
              context_positions: torch.Tensor, anchor: int, noise_count: int) -> torch.Tensor:
    """Grouped noncausal block attention; context at/after anchor is masked out."""
    queries, heads, width = q.shape
    if (queries != noise_count or k.shape != v.shape or k.shape[-1] != width
            or heads % k.shape[1] or k.shape[0] != len(context_positions) + noise_count):
        raise ValueError("inconsistent DFlash attention shapes")
    indices = torch.arange(heads) // (heads // k.shape[1])
    scores = torch.einsum("qhd,khd->hqk", as_f64(q), as_f64(k)[:, indices]) / math.sqrt(width)
    valid = torch.cat((context_positions < anchor, torch.ones(noise_count, dtype=torch.bool)))
    probabilities = ideal_softmax(scores.masked_fill(~valid[None, None, :], -math.inf))
    return torch.einsum("hqk,khd->qhd", probabilities, as_f64(v)[:, indices])


def forward(features: torch.Tensor, noise_embeddings: torch.Tensor,
            context_positions: torch.Tensor, query_positions: torch.Tensor,
            weights: Mapping[str, torch.Tensor], *, config: Config = Config(),
            cache: tuple[LayerCache, ...] | None = None,
            materialize_public_bf16: bool = False) -> Result:
    """Whole logical five-layer formula, with accepted-context-only cache output.

    Optional cache is preexisting accepted context only. New features may include
    rejected/lookahead rows at or beyond the anchor; attention masks them and they
    are excluded from returned cache. No noise K/V ever enters persistent cache.
    Optional public materialization rounds only the explicitly composed Linear,
    RMSNorm, RoPE, attention, residual-add and SiLU-multiply outputs. It is a
    supplementary composition trace, never a replacement for each Op's oracle.
    """
    c = config
    if (features.ndim != 2 or features.shape[1] != c.taps * c.hidden
            or noise_embeddings.ndim != 2 or noise_embeddings.shape[1] != c.hidden
            or context_positions.shape != (len(features),) or query_positions.shape != (len(noise_embeddings),)
            or len(query_positions) < 1 or c.heads % c.kv_heads):
        raise ValueError("DFlash public input shape mismatch")
    anchor = int(query_positions[0])
    if anchor < 0 or not torch.equal(query_positions, torch.arange(anchor, anchor + len(query_positions))):
        raise ValueError("DFlash queries must occupy anchor+[0,K)")
    if len(context_positions) and (bool((context_positions < 0).any())
                                  or bool((context_positions[1:] <= context_positions[:-1]).any())):
        raise ValueError("DFlash context positions must be strictly increasing nonnegative values")
    if cache is not None and len(cache) != c.layers:
        raise ValueError("DFlash cache layer count mismatch")
    def materialize(value: torch.Tensor) -> torch.Tensor:
        return value.bfloat16().double() if materialize_public_bf16 else value

    context = materialize(ordinary_rmsnorm(materialize(linear(features, weights["fc.weight"])),
                                           weights["hidden_norm.weight"], eps=c.epsilon))
    hidden = as_f64(noise_embeddings)
    traces, caches = [], []
    for layer in range(c.layers):
        p = f"layers.{layer}."
        x = materialize(ordinary_rmsnorm(hidden, weights[p + "input_layernorm.weight"], eps=c.epsilon))
        q = materialize(linear(x, weights[p + "self_attn.q_proj.weight"])).reshape(-1, c.heads, c.head_dim)
        q = materialize(ordinary_rmsnorm(q, weights[p + "self_attn.q_norm.weight"], eps=c.epsilon))
        q = materialize(partial_rope(q, query_positions, rotary_dim=c.head_dim, theta=c.theta))
        # Context is deliberately NOT passed through this layer's input norm.
        k_context = materialize(linear(context, weights[p + "self_attn.k_proj.weight"])).reshape(-1, c.kv_heads, c.head_dim)
        v_context = materialize(linear(context, weights[p + "self_attn.v_proj.weight"])).reshape(-1, c.kv_heads, c.head_dim)
        k_context = materialize(ordinary_rmsnorm(k_context, weights[p + "self_attn.k_norm.weight"], eps=c.epsilon))
        k_context = materialize(partial_rope(k_context, context_positions, rotary_dim=c.head_dim, theta=c.theta))
        positions = context_positions.clone()
        if cache is not None:
            prior = cache[layer]
            if (prior.keys.shape != prior.values.shape or prior.keys.shape != (len(prior.positions), c.kv_heads, c.head_dim)
                    or bool((prior.positions >= anchor).any())
                    or (len(positions) and len(prior.positions) and int(positions[0]) <= int(prior.positions[-1]))):
                raise ValueError("DFlash cache must precede new context and anchor")
            k_context = torch.cat((prior.keys, k_context), dim=0)
            v_context = torch.cat((prior.values, v_context), dim=0)
            positions = torch.cat((prior.positions, positions), dim=0)
        k_noise = materialize(linear(x, weights[p + "self_attn.k_proj.weight"])).reshape(-1, c.kv_heads, c.head_dim)
        v_noise = materialize(linear(x, weights[p + "self_attn.v_proj.weight"])).reshape(-1, c.kv_heads, c.head_dim)
        k_noise = materialize(ordinary_rmsnorm(k_noise, weights[p + "self_attn.k_norm.weight"], eps=c.epsilon))
        k_noise = materialize(partial_rope(k_noise, query_positions, rotary_dim=c.head_dim, theta=c.theta))
        a = materialize(attention(q, torch.cat((k_context, k_noise)), torch.cat((v_context, v_noise)),
                                  positions, anchor, len(query_positions)))
        projected = materialize(linear(a.reshape(len(query_positions), c.heads * c.head_dim),
                                       weights[p + "self_attn.o_proj.weight"]))
        after_attention = materialize(hidden + projected)
        mlp_input = materialize(ordinary_rmsnorm(after_attention, weights[p + "post_attention_layernorm.weight"], eps=c.epsilon))
        gate = materialize(linear(mlp_input, weights[p + "mlp.gate_proj.weight"]))
        up = materialize(linear(mlp_input, weights[p + "mlp.up_proj.weight"]))
        activated = materialize(silu(gate) * up)
        hidden = materialize(after_attention + materialize(linear(activated, weights[p + "mlp.down_proj.weight"])))
        keep = positions < anchor
        caches.append(LayerCache(k_context[keep].clone(), v_context[keep].clone(), positions[keep].clone()))
        traces.append(dict(input_norm=x, query=q, context_keys=k_context, context_values=v_context,
                           noise_keys=k_noise, noise_values=v_noise, attention=a,
                           after_attention=after_attention, mlp_input=mlp_input, gate=gate,
                           up=up, activated=activated, hidden=hidden))
    return Result(materialize(ordinary_rmsnorm(hidden, weights["norm.weight"], eps=c.epsilon)), context,
                  tuple(traces), tuple(caches))
