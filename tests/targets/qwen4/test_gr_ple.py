from __future__ import annotations

import math

import torch

from tools.reference.qwen4.common import grouped_zero_centered_rmsnorm
from tools.reference.qwen4.gated_residual import final_read, inject, read
from tools.reference.qwen4.ple import dilated_depthwise_conv
from tools.reference.qwen4.ple import inject as ple_inject


def test_grouped_zero_centered_rmsnorm_matches_hand_computation() -> None:
    value = torch.tensor([[3.0, 4.0, 0.0, 2.0]])
    weight = torch.tensor([0.0, 1.0, -0.5, 0.25])
    actual = grouped_zero_centered_rmsnorm(value, weight, group_size=2, eps=0.0)
    expected = torch.tensor(
        [[3.0 / math.sqrt(12.5), 8.0 / math.sqrt(12.5), 0.0, 2.5 / math.sqrt(2.0)]],
        dtype=torch.float64,
    )
    torch.testing.assert_close(actual, expected, atol=1e-14, rtol=1e-14)


def test_gated_residual_read_inject_and_final_read_match_direct_formula() -> None:
    branches = torch.tensor([[[1.0, 2.0], [3.0, 4.0]]])
    norm_weight = torch.tensor([0.1, -0.2, 0.3, -0.4])
    down = torch.tensor([[0.2, -0.1, 0.4, 0.3]])
    up = torch.tensor([[0.5], [-0.25], [0.75], [0.1]])
    write = torch.tensor([[0.2, 0.1, -0.1, 0.4], [-0.3, 0.5, 0.2, -0.2]])
    result = read(branches, norm_weight, down, up, write, eps=1e-6)

    flat = branches.double().flatten(-2)
    grouped = flat.reshape(1, 2, 2)
    normalized = grouped / torch.sqrt((grouped**2).mean(-1, keepdim=True) + 1e-6)
    normalized = normalized.flatten(-2) * (1.0 + norm_weight.double())
    low = torch.nn.functional.silu(normalized @ down.double().t() / 2.0)
    gates = torch.sigmoid(low @ up.double().t()).reshape(1, 2, 2)
    expected_mixed = (gates * normalized.reshape(1, 2, 2)).mean(1)
    expected_write = 2.0 * torch.sigmoid(normalized @ write.double().t() / 2.0)
    torch.testing.assert_close(result.mixed, expected_mixed)
    torch.testing.assert_close(result.injection_scales, expected_write)

    block = torch.tensor([[0.25, -0.5]])
    represented_write = result.injection_scales.to(torch.bfloat16)
    expected_branches = branches.double() + represented_write.double().unsqueeze(-1) * block.double().unsqueeze(1)
    torch.testing.assert_close(inject(branches, block, represented_write), expected_branches)
    assert not torch.equal(represented_write.double(), result.injection_scales)
    torch.testing.assert_close(
        final_read(branches, norm_weight, down, up),
        result.mixed,
    )


def _ple_fixture(tokens: int = 5):
    generator = torch.Generator().manual_seed(71)
    branches, hidden, heads, row_width = 2, 3, 2, 2
    return {
        "residual_branches": torch.randn(tokens, branches, hidden, generator=generator),
        "table_rows": torch.randn(tokens, heads, row_width, generator=generator),
        "key_weight": torch.randn(branches * hidden, heads * row_width, generator=generator),
        "value_weight": torch.randn(hidden, heads * row_width, generator=generator),
        "key_norm_weight": torch.randn(branches * hidden, generator=generator) * 0.1,
        "query_norm_weight": torch.randn(branches * hidden, generator=generator) * 0.1,
        "conv_norm_weight": torch.randn(branches * hidden, generator=generator) * 0.1,
        "conv_weight": torch.randn(branches * hidden, 4, generator=generator),
        "conv_state": torch.randn(branches * hidden, 9, generator=generator),
    }


def test_ple_context_gate_matches_hand_formula_when_conv_is_zero() -> None:
    residual = torch.tensor([[[1.0, 0.0], [0.0, -1.0]]])
    rows = torch.tensor([[[2.0, -1.0]]])
    identity = torch.eye(2)
    zeros = torch.zeros(4)
    result = ple_inject(
        residual,
        rows,
        key_weight=torch.cat((identity, identity)),
        value_weight=identity,
        key_norm_weight=zeros,
        query_norm_weight=zeros,
        conv_norm_weight=zeros,
        conv_weight=torch.zeros(4, 4),
        conv_state=torch.zeros(4, 9),
        eps=0.0,
    )
    key = rows.double().reshape(1, 2)
    key = key / torch.sqrt((key**2).mean(-1, keepdim=True))
    keys = key.unsqueeze(1).expand(-1, 2, -1)
    queries = residual.double() / torch.sqrt((residual.double() ** 2).mean(-1, keepdim=True))
    raw_gate = (keys * queries).sum(-1, keepdim=True) / math.sqrt(2.0)
    expected_gate = raw_gate.sign() * raw_gate.abs().clamp_min(1e-6).sqrt()
    expected_gated = torch.sigmoid(expected_gate) * rows.double()
    torch.testing.assert_close(result.gate, expected_gate)
    torch.testing.assert_close(result.gated_value, expected_gated)
    torch.testing.assert_close(result.injection, expected_gated)
    torch.testing.assert_close(result.output, residual.double() + expected_gated)


def test_ple_dilated_convolution_matches_lag_and_weight_order_fixture() -> None:
    value = torch.tensor([[10.0], [11.0], [12.0], [13.0]])
    state = torch.arange(1.0, 10.0).reshape(1, 9)
    weight = torch.tensor([[1.0, 10.0, 100.0, 1000.0]])
    actual, next_state = dilated_depthwise_conv(value, weight, state, dilation=3)
    expected = torch.tensor([[10_741.0], [11_852.0], [12_963.0], [14_074.0]])
    torch.testing.assert_close(actual, expected.double(), atol=0.0, rtol=0.0)
    torch.testing.assert_close(
        next_state,
        torch.tensor(
            [[5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0]],
            dtype=torch.float64,
        ),
        atol=0.0,
        rtol=0.0,
    )


def test_ple_one_shot_chunked_and_t1_continuation_are_identical() -> None:
    fixture = _ple_fixture()
    fixture["conv_state"] = fixture["conv_state"].to(torch.bfloat16)
    one_shot = ple_inject(**fixture)
    first_args = {key: value[:2] if key in {"residual_branches", "table_rows"} else value for key, value in fixture.items()}
    first = ple_inject(**first_args)
    second_args = {
        key: (value[2:] if key in {"residual_branches", "table_rows"} else value)
        for key, value in fixture.items()
    }
    second_args["conv_state"] = first.next_conv_state
    second = ple_inject(**second_args)
    torch.testing.assert_close(one_shot.output, torch.cat((first.output, second.output)), atol=1e-12, rtol=1e-12)
    torch.testing.assert_close(one_shot.next_conv_state, second.next_conv_state, atol=0.0, rtol=0.0)

    state = fixture["conv_state"]
    token_outputs = []
    for token in range(5):
        token_args = {
            key: (value[token : token + 1] if key in {"residual_branches", "table_rows"} else value)
            for key, value in fixture.items()
        }
        token_args["conv_state"] = state
        token_result = ple_inject(**token_args)
        token_outputs.append(token_result.output)
        state = token_result.next_conv_state
    torch.testing.assert_close(one_shot.output, torch.cat(token_outputs), atol=1e-12, rtol=1e-12)
    torch.testing.assert_close(one_shot.next_conv_state, state, atol=0.0, rtol=0.0)
    assert one_shot.next_conv_state.dtype == torch.bfloat16


def test_ple_convolution_reads_current_rows_after_represented_state_rounding() -> None:
    residual = torch.tensor([[[0.0]]], dtype=torch.bfloat16)
    rows = torch.tensor([[[1.00390625]]], dtype=torch.float32)
    result = ple_inject(
        residual,
        rows,
        key_weight=torch.zeros(1, 1),
        value_weight=torch.eye(1),
        key_norm_weight=torch.zeros(1),
        query_norm_weight=torch.zeros(1),
        conv_norm_weight=torch.zeros(1),
        conv_weight=torch.tensor([[0.0, 0.0, 0.0, 1.0]]),
        conv_state=torch.zeros(1, 9, dtype=torch.bfloat16),
        eps=1.0,
    )
    gated = 0.5 * rows.double()
    ideal_normalized = gated / torch.sqrt(gated * gated + 1.0)
    represented = ideal_normalized.to(torch.bfloat16).double()
    expected = gated.flatten(-2) + torch.nn.functional.silu(represented.flatten(-2))
    torch.testing.assert_close(result.injection.flatten(-2), expected, atol=1e-14, rtol=1e-14)
    torch.testing.assert_close(result.output.flatten(-2), expected, atol=1e-14, rtol=1e-14)
    assert not torch.equal(represented, ideal_normalized)
    assert result.next_conv_state.dtype == torch.bfloat16
    assert result.next_conv_state[0, -1].item() == represented.item()
