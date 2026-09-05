from __future__ import annotations

import math

import pytest
import torch

from tools.reference.qwen4.common import partial_rope
from tools.reference.qwen4.qsa import (
    actual_gguf_select,
    actual_gguf_sparse_attention,
    source_select,
    source_sparse_attention,
)


def _selector(
    raw_keys: torch.Tensor,
    visible: list[int],
    *,
    token_budget: int = 4,
):
    queries = torch.zeros((1, 4, 2), dtype=raw_keys.dtype)
    return source_select(
        queries,
        raw_keys,
        [torch.tensor(visible)],
        query_positions=torch.zeros(1),
        key_positions=torch.zeros(raw_keys.shape[0]),
        query_norm_weight=torch.zeros(2),
        key_norm_weight=torch.zeros(2),
        token_budget=token_budget,
        compress_ratio=4,
        rotary_dim=2,
        eps=1e-6,
    )[0]


def test_qsa_uses_request_local_visible_rank_blocks_and_appends_tail() -> None:
    raw_keys = torch.arange(20, dtype=torch.float32).reshape(10, 2)
    result = _selector(raw_keys, [1, 3, 4, 7, 8])
    assert result.complete_block_count == 1
    assert result.selected_block_ids.tolist() == [0]
    assert result.token_ids.tolist() == [1, 3, 4, 7, 8]


def test_qsa_equal_scores_select_lower_logical_block_id() -> None:
    raw_keys = torch.zeros((8, 2), dtype=torch.bfloat16)
    result = _selector(raw_keys, list(range(8)))
    assert result.selected_block_ids.tolist() == [0]
    assert result.token_ids.tolist() == [0, 1, 2, 3]


@pytest.mark.parametrize("visible_count", [0, 1, 2, 3, 4, 5, 2047, 2048, 2049, 2050, 2051, 2052, 2053])
def test_qsa_block_and_2048_token_budget_frontiers(visible_count: int) -> None:
    raw_keys = torch.zeros((visible_count, 2), dtype=torch.float32)
    result = _selector(raw_keys, list(range(visible_count)), token_budget=2048)
    complete = visible_count // 4
    tail = visible_count % 4
    assert result.complete_block_count == complete
    assert result.selected_block_ids.tolist() == list(range(min(complete, 512)))
    assert result.token_ids.numel() == min(complete, 512) * 4 + tail
    if tail:
        assert result.token_ids[-tail:].tolist() == list(range(visible_count - tail, visible_count))


def test_qsa_fp32_pool_is_cast_back_before_key_normalization() -> None:
    raw_keys = torch.tensor(
        [[1.0, 0.125], [1.5, 0.25], [2.0, 0.5], [3.0, 1.0]],
        dtype=torch.bfloat16,
    )
    queries = torch.tensor([[[1.0, 0.0]] * 4], dtype=torch.bfloat16)
    result = source_select(
        queries,
        raw_keys,
        [torch.arange(4)],
        query_positions=torch.zeros(1),
        key_positions=torch.zeros(4),
        query_norm_weight=torch.zeros(2),
        key_norm_weight=torch.tensor([0.2, -0.1]),
        token_budget=4,
        compress_ratio=4,
        rotary_dim=2,
        eps=1e-6,
    )[0]
    pooled = raw_keys.float().mean(0).to(torch.bfloat16).double()
    pooled = pooled / torch.sqrt((pooled**2).mean() + 1e-6)
    # The oracle begins from the represented FP32 norm weights, not decimal reals.
    pooled *= 1.0 + torch.tensor([0.2, -0.1], dtype=torch.float32).double()
    query = torch.tensor([1.0, 0.0], dtype=torch.float64)
    query /= torch.sqrt((query**2).mean() + 1e-6)
    expected = 4.0 * torch.relu(query @ pooled) / math.sqrt(2.0)
    torch.testing.assert_close(result.scores, expected.reshape(1), atol=1e-12, rtol=1e-12)

    actual_gguf = actual_gguf_select(
        queries,
        raw_keys,
        [torch.arange(4)],
        query_positions=torch.zeros(1),
        key_positions=torch.zeros(4),
        query_norm_gamma=torch.ones(2),
        key_norm_gamma=1.0 + torch.tensor([0.2, -0.1], dtype=torch.float32).double(),
        token_budget=4,
        compress_ratio=4,
        rotary_dim=2,
        eps=1e-6,
    )[0]
    torch.testing.assert_close(actual_gguf.scores, expected.reshape(1), atol=1e-12, rtol=1e-12)


def test_partial_interleaved_mrope_takes_each_frequency_from_its_axis() -> None:
    value = torch.tensor([[1.0, 1.0, 1.0, 0.0, 0.0, 0.0]])
    positions = torch.tensor([[0.0], [1.0], [2.0]])
    actual = partial_rope(
        value,
        positions,
        rotary_dim=6,
        theta=1.0,
        mrope_section=(1, 1, 1),
    )
    expected = torch.tensor(
        [[1.0, math.cos(1.0), math.cos(2.0), 0.0, math.sin(1.0), math.sin(2.0)]],
        dtype=torch.float64,
    )
    torch.testing.assert_close(actual, expected, atol=1e-14, rtol=1e-14)


def test_sparse_gqa_uses_exact_selected_ids_gate_and_projection() -> None:
    queries = torch.tensor([[[1.0, 0.0], [0.0, 1.0]]])
    keys = torch.tensor([[[1.0, 0.0]], [[0.0, 1.0]], [[-1.0, 0.0]]])
    values = torch.tensor([[[2.0, 0.0]], [[0.0, 4.0]], [[100.0, 100.0]]])
    result = source_sparse_attention(
        queries,
        keys,
        values,
        [torch.tensor([0, 1])],
        query_positions=torch.zeros(1),
        key_positions=torch.zeros(3),
        query_norm_weight=torch.zeros(2),
        key_norm_weight=torch.zeros(2),
        output_gate=torch.zeros(1, 4),
        output_weight=torch.eye(4),
        core_cache_dtype=torch.float64,
        rotary_dim=2,
        eps=0.0,
    )
    scale = math.sqrt(2.0)
    first_probs = torch.softmax(torch.tensor([scale, 0.0], dtype=torch.float64), dim=0)
    second_probs = torch.softmax(torch.tensor([0.0, scale], dtype=torch.float64), dim=0)
    expected_core = torch.stack(
        (
            first_probs[0] * values[0, 0].double() + first_probs[1] * values[1, 0].double(),
            second_probs[0] * values[0, 0].double() + second_probs[1] * values[1, 0].double(),
        )
    ).unsqueeze(0)
    torch.testing.assert_close(result.core, expected_core, atol=1e-14, rtol=1e-14)
    torch.testing.assert_close(
        result.output,
        0.5 * expected_core.flatten(-2),
        atol=1e-14,
        rtol=1e-14,
    )

    actual_gguf = actual_gguf_sparse_attention(
        queries,
        keys,
        values,
        [torch.tensor([0, 1])],
        query_positions=torch.zeros(1),
        key_positions=torch.zeros(3),
        query_norm_gamma=torch.ones(2),
        key_norm_gamma=torch.ones(2),
        output_gate=torch.zeros(1, 4),
        output_weight=torch.eye(4),
        core_cache_dtype=torch.float64,
        rotary_dim=2,
        eps=0.0,
    )
    torch.testing.assert_close(actual_gguf.output, result.output, atol=1e-14, rtol=1e-14)


def test_sparse_attention_reads_bf16_encoded_core_cache() -> None:
    queries = torch.tensor([[[1.0, 0.25]]])
    keys = torch.tensor([[[1.0, 0.2]], [[0.1, 1.0]]])
    values = torch.tensor([[[2.01, -0.3]], [[0.2, 4.03]]])
    result = source_sparse_attention(
        queries,
        keys,
        values,
        [torch.tensor([0, 1])],
        query_positions=torch.zeros(1),
        key_positions=torch.zeros(2),
        query_norm_weight=torch.zeros(2),
        key_norm_weight=torch.zeros(2),
        output_gate=torch.zeros(1, 2),
        output_weight=torch.eye(2),
        core_cache_dtype=torch.bfloat16,
        rotary_dim=2,
    )
    q = queries.double()
    q /= torch.sqrt((q * q).mean(-1, keepdim=True) + 1e-6)
    ideal_k = keys.double()
    ideal_k /= torch.sqrt((ideal_k * ideal_k).mean(-1, keepdim=True) + 1e-6)
    represented_k = ideal_k.to(torch.bfloat16).double()
    represented_v = values.to(torch.bfloat16).double()
    scores = represented_k[:, 0] @ q[0, 0] / math.sqrt(2.0)
    probabilities = torch.softmax(scores, dim=0)
    expected_core = torch.sum(probabilities[:, None] * represented_v[:, 0], dim=0)
    torch.testing.assert_close(result.core[0, 0], expected_core, atol=1e-14, rtol=1e-14)

    ideal_scores = ideal_k[:, 0] @ q[0, 0] / math.sqrt(2.0)
    ideal_probabilities = torch.softmax(ideal_scores, dim=0)
    bypassed_core = torch.sum(ideal_probabilities[:, None] * values[:, 0].double(), dim=0)
    assert not torch.allclose(result.core[0, 0], bypassed_core, atol=1e-5, rtol=1e-5)
