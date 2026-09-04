"""Naive Qwen Sparse Attention selector and gathered-attention oracles."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Sequence

import torch

from .common import (
    grouped_zero_centered_rmsnorm,
    ideal_softmax,
    linear,
    partial_rope,
    sigmoid,
)


@dataclass(frozen=True, slots=True)
class QSASelection:
    token_ids: torch.Tensor
    selected_block_ids: torch.Tensor
    scores: torch.Tensor
    complete_block_count: int


@dataclass(frozen=True, slots=True)
class QSAAttentionResult:
    output: torch.Tensor
    core: torch.Tensor
    probabilities: tuple[torch.Tensor, ...]


def _represented_pool(raw_keys: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    """Semantic FP32 mean followed by a cast to the represented raw-key dtype."""

    selected = raw_keys.detach().to(device="cpu").index_select(0, indices)
    pooled = selected.to(torch.float32).mean(dim=0)
    return pooled.to(raw_keys.dtype).to(torch.float64)


def select(
    index_queries: torch.Tensor,
    raw_keys: torch.Tensor,
    visible_token_ids: Sequence[torch.Tensor],
    query_positions: torch.Tensor,
    key_positions: torch.Tensor,
    query_norm_weight: torch.Tensor,
    key_norm_weight: torch.Tensor,
    *,
    token_budget: int = 2048,
    compress_ratio: int = 4,
    rotary_dim: int = 64,
    theta: float = 10_000_000.0,
    mrope_section: tuple[int, int, int] | None = None,
    eps: float = 1e-6,
) -> tuple[QSASelection, ...]:
    """Select request-local visible blocks, with deterministic lower-id ties."""

    if index_queries.ndim != 3 or raw_keys.ndim != 2:
        raise ValueError("index queries and raw keys must be [Q,H,D] and [K,D]")
    if len(visible_token_ids) != index_queries.shape[0]:
        raise ValueError("one visible-token list is required for each query")
    if token_budget % compress_ratio:
        raise ValueError("token_budget must be divisible by compress_ratio")
    head_dim = index_queries.shape[-1]
    queries = grouped_zero_centered_rmsnorm(
        index_queries,
        query_norm_weight,
        group_size=head_dim,
        eps=eps,
    )
    queries = partial_rope(
        queries,
        query_positions,
        rotary_dim=rotary_dim,
        theta=theta,
        mrope_section=mrope_section,
    )
    block_budget = token_budget // compress_ratio
    results: list[QSASelection] = []
    for query_index, visible_value in enumerate(visible_token_ids):
        visible = visible_value.detach().to(dtype=torch.int64, device="cpu").reshape(-1)
        complete_count = visible.numel() // compress_ratio
        score_tensor = torch.empty((0,), dtype=torch.float64)
        selected_blocks = torch.empty((0,), dtype=torch.int64)
        selected_tokens = torch.empty((0,), dtype=torch.int64)
        if complete_count:
            blocks = visible[: complete_count * compress_ratio].reshape(
                complete_count, compress_ratio
            )
            pooled = torch.stack(
                [_represented_pool(raw_keys, block) for block in blocks], dim=0
            )
            pooled = grouped_zero_centered_rmsnorm(
                pooled,
                key_norm_weight,
                group_size=head_dim,
                eps=eps,
            )
            position_axis = 0 if key_positions.ndim == 1 else 1
            block_positions = key_positions.index_select(position_axis, blocks[:, 0])
            pooled = partial_rope(
                pooled,
                block_positions,
                rotary_dim=rotary_dim,
                theta=theta,
                mrope_section=mrope_section,
            )
            score_tensor = torch.clamp(
                queries[query_index] @ pooled.transpose(0, 1), min=0.0
            ).sum(dim=0) / math.sqrt(head_dim)
            order = sorted(
                range(complete_count),
                key=lambda block_id: (-float(score_tensor[block_id]), block_id),
            )
            selected_blocks = torch.tensor(
                order[: min(block_budget, complete_count)], dtype=torch.int64
            )
            selected_tokens = blocks.index_select(0, selected_blocks).flatten()
        tail = visible[complete_count * compress_ratio :]
        results.append(
            QSASelection(
                token_ids=torch.cat((selected_tokens, tail)),
                selected_block_ids=selected_blocks,
                scores=score_tensor,
                complete_block_count=complete_count,
            )
        )
    return tuple(results)


def sparse_attention(
    queries: torch.Tensor,
    keys: torch.Tensor,
    values: torch.Tensor,
    selections: Sequence[torch.Tensor | QSASelection],
    query_positions: torch.Tensor,
    key_positions: torch.Tensor,
    query_norm_weight: torch.Tensor,
    key_norm_weight: torch.Tensor,
    output_gate: torch.Tensor,
    output_weight: torch.Tensor,
    *,
    core_cache_dtype: torch.dtype | None = None,
    rotary_dim: int = 64,
    theta: float = 10_000_000.0,
    mrope_section: tuple[int, int, int] | None = None,
    eps: float = 1e-6,
) -> QSAAttentionResult:
    """Naive selected-id GQA through an explicit represented core-cache boundary.

    ``core_cache_dtype`` models a plain floating cache representation such as
    preview BF16.  Packed INT8/NVFP4 profiles require their own exact codec
    round-trip and are intentionally not approximated here.
    """

    if queries.ndim != 3 or keys.ndim != 3 or values.ndim != 3:
        raise ValueError("Q/K/V must have shape [T,H,D]")
    query_count, query_heads, head_dim = queries.shape
    if keys.shape != values.shape or keys.shape[-1] != head_dim:
        raise ValueError("K/V geometry must match the query head width")
    kv_heads = keys.shape[1]
    if query_heads % kv_heads:
        raise ValueError("query head count must be divisible by KV head count")
    if len(selections) != query_count:
        raise ValueError("one selection is required per query")
    q = grouped_zero_centered_rmsnorm(
        queries, query_norm_weight, group_size=head_dim, eps=eps
    )
    k = grouped_zero_centered_rmsnorm(
        keys, key_norm_weight, group_size=head_dim, eps=eps
    )
    q = partial_rope(
        q, query_positions, rotary_dim=rotary_dim, theta=theta, mrope_section=mrope_section
    )
    k = partial_rope(
        k, key_positions, rotary_dim=rotary_dim, theta=theta, mrope_section=mrope_section
    )
    cache_dtype = keys.dtype if core_cache_dtype is None else core_cache_dtype
    if not cache_dtype.is_floating_point:
        raise ValueError("packed core-cache profiles require an exact codec oracle")
    k = k.to(cache_dtype).to(torch.float64)
    cached_values = values.detach().to(device="cpu").to(cache_dtype).to(torch.float64)
    core = torch.empty_like(q)
    probabilities: list[torch.Tensor] = []
    groups = query_heads // kv_heads
    for token in range(query_count):
        selection = selections[token]
        ids = selection.token_ids if isinstance(selection, QSASelection) else selection
        ids = ids.detach().to(dtype=torch.int64, device="cpu").reshape(-1)
        if not ids.numel():
            raise ValueError("sparse attention selection cannot be empty")
        token_probabilities = torch.empty(
            (query_heads, ids.numel()), dtype=torch.float64
        )
        for head in range(query_heads):
            kv_head = head // groups
            selected_keys = k.index_select(0, ids)[:, kv_head]
            scores = selected_keys @ q[token, head] / math.sqrt(head_dim)
            probs = ideal_softmax(scores)
            token_probabilities[head] = probs
            selected_values = cached_values.index_select(0, ids)[:, kv_head]
            core[token, head] = torch.sum(probs[:, None] * selected_values, dim=0)
        probabilities.append(token_probabilities)
    flat_core = core.flatten(-2)
    if output_gate.shape != flat_core.shape:
        raise ValueError("output gate must have shape [Q, query_heads * head_dim]")
    output = linear(flat_core * sigmoid(output_gate), output_weight)
    return QSAAttentionResult(output=output, core=core, probabilities=tuple(probabilities))


__all__ = ["QSAAttentionResult", "QSASelection", "select", "sparse_attention"]
