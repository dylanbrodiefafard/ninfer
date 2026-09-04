from __future__ import annotations

import math

import torch

from tools.reference.qwen4.gdn import (
    GDNState,
    GDNWeights,
    causal_depthwise_convolution,
    control_gates,
    output_projection,
    recurrence,
    repeat_key_value_heads,
    sublayer,
)
from tools.reference.qwen4.moe import route, sparse_moe


def test_router_uses_full_softmax_lower_id_ties_and_selected_normalization() -> None:
    logits = torch.tensor([[1.0, 1.0, -2.0, 0.5]], dtype=torch.float32)
    weights, ids = route(logits, 3)
    assert ids.tolist() == [[0, 1, 3]]
    expected = torch.softmax(torch.tensor([1.0, 1.0, 0.5], dtype=torch.float64), dim=0)
    torch.testing.assert_close(weights[0], expected, atol=1e-14, rtol=1e-14)
    assert len(set(ids[0].tolist())) == 3


def test_sparse_moe_matches_direct_routed_and_shared_swiglu_formula() -> None:
    hidden = torch.tensor([[1.0, 2.0], [-1.0, 0.5]])
    router = torch.zeros((3, 2))
    expert_gate = torch.tensor([[[1.0, 0.0]], [[0.0, 1.0]], [[1.0, 1.0]]])
    expert_up = torch.tensor([[[0.0, 1.0]], [[1.0, 0.0]], [[1.0, -1.0]]])
    expert_down = torch.tensor([[[1.0], [2.0]], [[-1.0], [1.0]], [[3.0], [1.0]]])
    shared_gate = torch.tensor([[1.0, 0.0]])
    shared_up = torch.tensor([[0.0, 1.0]])
    shared_down = torch.tensor([[2.0], [-1.0]])
    shared_scale = torch.tensor([[0.5, -0.25]])
    result = sparse_moe(
        hidden,
        router,
        expert_gate,
        expert_up,
        expert_down,
        shared_gate,
        shared_up,
        shared_down,
        shared_scale,
        experts_per_token=2,
    )
    assert result.expert_ids.tolist() == [[0, 1], [0, 1]]
    expected = []
    for token in hidden.double():
        routed = torch.zeros(2, dtype=torch.float64)
        for expert in (0, 1):
            gate = expert_gate[expert].double() @ token
            up = expert_up[expert].double() @ token
            activation = torch.nn.functional.silu(gate) * up
            routed += 0.5 * (expert_down[expert].double() @ activation)
        gate = shared_gate.double() @ token
        up = shared_up.double() @ token
        shared = shared_down.double() @ (torch.nn.functional.silu(gate) * up)
        scale = torch.sigmoid(shared_scale.double() @ token)
        expected.append(routed + scale * shared)
    torch.testing.assert_close(result.output, torch.stack(expected), atol=1e-14, rtol=1e-14)


def test_gdn_recurrence_matches_hand_state_updates_and_fp32_boundary() -> None:
    query = torch.ones((2, 1, 1))
    key = torch.ones((2, 1, 1))
    value = torch.tensor([[[2.0], [4.0]], [[6.0], [10.0]]])
    decay = torch.tensor([[0.0, 0.0], [math.log(0.5), math.log(0.25)]])
    beta = torch.tensor([[1.0, 1.0], [0.5, 0.25]])
    initial = torch.zeros((2, 1, 1), dtype=torch.float32)
    result = recurrence(query, key, value, decay, beta, initial, eps=0.0)
    first_state = torch.tensor([[[2.0]], [[4.0]]], dtype=torch.float32)
    second_state = torch.tensor(
        [[[0.5 * 2.0 + 0.5 * (6.0 - 0.5 * 2.0)]], [[0.25 * 4.0 + 0.25 * (10.0 - 0.25 * 4.0)]]],
        dtype=torch.float32,
    )
    assert result.final_state.dtype == torch.float32
    assert torch.equal(result.state_records[0], first_state)
    assert torch.equal(result.final_state, second_state)
    torch.testing.assert_close(result.output[0], first_state.double().squeeze(1), atol=0.0, rtol=0.0)
    torch.testing.assert_close(result.output[1], second_state.double().squeeze(1), atol=0.0, rtol=0.0)


def test_gdn_output_uses_plain_per_head_norm_sigmoid_gate_and_projection() -> None:
    core = torch.tensor([[[3.0, 4.0], [0.0, 2.0]]])
    gate = torch.tensor(
        [[[0.0, math.log(3.0)], [-math.log(3.0), 0.0]]], dtype=torch.float64
    )
    norm_weight = torch.tensor([2.0, 0.5])
    projection = torch.eye(4)
    actual = output_projection(core, gate, norm_weight, projection, eps=0.0)
    normalized = torch.tensor(
        [[[6.0 / math.sqrt(12.5), 2.0 / math.sqrt(12.5)], [0.0, 1.0 / math.sqrt(2.0)]]],
        dtype=torch.float64,
    )
    expected_gate = torch.tensor([[[0.5, 0.75], [0.25, 0.5]]], dtype=torch.float64)
    expected = (normalized * expected_gate).to(torch.bfloat16).flatten(-2).double()
    torch.testing.assert_close(actual, expected, atol=0.0, rtol=0.0)


def test_gdn_one_shot_partitioned_prefill_and_t1_state_are_identical() -> None:
    generator = torch.Generator().manual_seed(97)
    tokens, key_heads, value_heads, width = 6, 2, 4, 3
    query = torch.randn(tokens, key_heads, width, generator=generator)
    key = torch.randn(tokens, key_heads, width, generator=generator)
    value = torch.randn(tokens, value_heads, width, generator=generator)
    decay = -torch.rand(tokens, value_heads, generator=generator)
    beta = torch.rand(tokens, value_heads, generator=generator)
    state = torch.randn(value_heads, width, width, generator=generator).to(torch.float32)
    one_shot = recurrence(query, key, value, decay, beta, state)
    first = recurrence(query[:2], key[:2], value[:2], decay[:2], beta[:2], state)
    second = recurrence(query[2:], key[2:], value[2:], decay[2:], beta[2:], first.final_state)
    torch.testing.assert_close(one_shot.output, torch.cat((first.output, second.output)), atol=0.0, rtol=0.0)
    assert torch.equal(one_shot.final_state, second.final_state)

    current = state
    outputs = []
    for token in range(tokens):
        step = recurrence(
            query[token : token + 1],
            key[token : token + 1],
            value[token : token + 1],
            decay[token : token + 1],
            beta[token : token + 1],
            current,
        )
        outputs.append(step.output)
        current = step.final_state
    torch.testing.assert_close(one_shot.output, torch.cat(outputs), atol=0.0, rtol=0.0)
    assert torch.equal(one_shot.final_state, current)


def test_gdn_causal_convolution_reads_and_records_represented_raw_qkv() -> None:
    history = torch.tensor([[1.0, 2.0, 3.0], [10.0, 20.0, 30.0]], dtype=torch.bfloat16)
    current = torch.tensor([[4.01, 40.1], [5.02, 50.2]], dtype=torch.float64)
    weight = torch.tensor([[1.0, 2.0, 3.0, 4.0], [-1.0, 0.5, 0.25, 2.0]])
    actual, final_history = causal_depthwise_convolution(current, weight, history)

    represented = current.to(torch.bfloat16)
    first_windows = (
        [history[0, 0], history[0, 1], history[0, 2], current[0, 0]],
        [history[1, 0], history[1, 1], history[1, 2], current[0, 1]],
    )
    expected_pre_activation = torch.tensor(
        [
            sum(float(x) * float(w) for x, w in zip(window, row))
            for window, row in zip(first_windows, weight)
        ],
        dtype=torch.float64,
    )
    expected_first = expected_pre_activation / (1.0 + torch.exp(-expected_pre_activation))
    torch.testing.assert_close(actual[0], expected_first, atol=1e-14, rtol=1e-14)
    assert final_history.dtype == torch.bfloat16
    assert torch.equal(
        final_history,
        torch.stack((history[:, 2], represented[0], represented[1]), dim=1),
    )


def test_gdn_controls_and_contiguous_three_way_head_repeat() -> None:
    a = torch.tensor([[0.0, math.log(3.0)]])
    b = torch.tensor([[0.0, math.log(3.0)]])
    a_log = torch.tensor([0.0, math.log(2.0)])
    dt_bias = torch.tensor([0.0, -math.log(3.0)])
    log_decay, beta = control_gates(a, b, a_log, dt_bias)
    expected_decay = -torch.exp(a_log.double()) * torch.nn.functional.softplus(
        a.double() + dt_bias.double()
    ).squeeze(0)
    torch.testing.assert_close(log_decay[0], expected_decay, atol=1e-14, rtol=1e-14)
    expected_beta = 1.0 / (1.0 + torch.exp(-b.double()))
    torch.testing.assert_close(beta, expected_beta, atol=1e-14, rtol=1e-14)

    heads = torch.tensor([[[1.0], [2.0]]])
    repeated = repeat_key_value_heads(heads, 6)
    assert repeated[:, :, 0].tolist() == [[1.0, 1.0, 1.0, 2.0, 2.0, 2.0]]


def test_complete_gdn_sublayer_matches_hand_scalar_head_formula() -> None:
    hidden = torch.ones((1, 1))
    qkv = torch.tensor([[1.0], [2.0], [3.0], [4.0], [5.0]])
    weights = GDNWeights(
        qkv=qkv,
        z=torch.zeros((3, 1)),
        a=torch.zeros((3, 1)),
        b=torch.zeros((3, 1)),
        conv=torch.tensor([[0.0, 0.0, 0.0, 1.0]]).repeat(5, 1),
        a_log=torch.zeros(3),
        dt_bias=torch.zeros(3),
        norm=torch.ones(1),
        output=torch.ones((1, 3)),
    )
    state = GDNState(
        conv_history=torch.zeros((5, 3), dtype=torch.bfloat16),
        recurrent=torch.zeros((3, 1, 1), dtype=torch.float32),
    )
    result = sublayer(
        hidden,
        weights,
        state,
        key_heads=1,
        value_heads=3,
        key_width=1,
        value_width=1,
        eps=0.0,
    )

    values = torch.tensor([3.0, 4.0, 5.0], dtype=torch.float64)
    values = (values / (1.0 + torch.exp(-values))).to(torch.bfloat16).double()
    expected_state = (0.5 * values).to(torch.float32).reshape(3, 1, 1)
    assert torch.equal(result.final_state.recurrent, expected_state)
    torch.testing.assert_close(
        result.log_decay,
        torch.full((1, 3), -math.log(2.0), dtype=torch.float32),
        atol=0.0,
        rtol=0.0,
    )
    torch.testing.assert_close(
        result.beta,
        torch.full((1, 3), 0.5, dtype=torch.float32),
        atol=0.0,
        rtol=0.0,
    )
    torch.testing.assert_close(
        result.output,
        torch.tensor([[1.5]], dtype=torch.float64),
        atol=1e-14,
        rtol=1e-14,
    )


def _complete_gdn_case() -> tuple[torch.Tensor, GDNWeights, GDNState]:
    generator = torch.Generator().manual_seed(211)
    tokens, hidden_width = 7, 4
    key_heads, value_heads, width = 2, 6, 3
    qkv_width = 2 * key_heads * width + value_heads * width
    weights = GDNWeights(
        qkv=0.2 * torch.randn(qkv_width, hidden_width, generator=generator),
        z=0.2 * torch.randn(value_heads * width, hidden_width, generator=generator),
        a=0.2 * torch.randn(value_heads, hidden_width, generator=generator),
        b=0.2 * torch.randn(value_heads, hidden_width, generator=generator),
        conv=0.2 * torch.randn(qkv_width, 4, generator=generator),
        a_log=0.2 * torch.randn(value_heads, generator=generator),
        dt_bias=0.2 * torch.randn(value_heads, generator=generator),
        norm=1.0 + 0.2 * torch.randn(width, generator=generator),
        output=0.2 * torch.randn(5, value_heads * width, generator=generator),
    )
    state = GDNState(
        conv_history=(0.2 * torch.randn(qkv_width, 3, generator=generator)).to(torch.bfloat16),
        recurrent=(0.2 * torch.randn(value_heads, width, width, generator=generator)).to(
            torch.float32
        ),
    )
    return torch.randn(tokens, hidden_width, generator=generator), weights, state


def test_complete_gdn_publishes_each_declared_consumer_representation() -> None:
    hidden, weights, initial = _complete_gdn_case()
    geometry = dict(key_heads=2, value_heads=6, key_width=3, value_width=3)
    result = sublayer(hidden, weights, initial, **geometry)

    assert result.convolved_qkv.dtype == torch.bfloat16
    assert result.repeated_query.dtype == torch.float64
    assert result.repeated_key.dtype == torch.float64
    assert result.log_decay.dtype == torch.float32
    assert result.beta.dtype == torch.float32
    assert result.core.dtype == torch.bfloat16
    assert result.final_state.conv_history.dtype == torch.bfloat16
    assert result.final_state.recurrent.dtype == torch.float32
    assert result.output.dtype == torch.float64

    projected = torch.nn.functional.linear(hidden.double(), weights.qkv.double())
    ideal_conv, _ = causal_depthwise_convolution(projected, weights.conv, initial.conv_history)
    assert torch.equal(result.convolved_qkv, ideal_conv.to(torch.bfloat16))


def test_complete_gdn_real_16_48_128_geometry_and_nonzero_fp32_state() -> None:
    generator = torch.Generator().manual_seed(313)
    key_heads, value_heads, width, hidden_width = 16, 48, 128, 1
    qkv_width = 2 * key_heads * width + value_heads * width
    hidden = torch.tensor([[0.75]], dtype=torch.bfloat16)
    weights = GDNWeights(
        qkv=0.01 * torch.randn(qkv_width, hidden_width, generator=generator),
        z=0.01 * torch.randn(value_heads * width, hidden_width, generator=generator),
        a=0.01 * torch.randn(value_heads, hidden_width, generator=generator),
        b=0.01 * torch.randn(value_heads, hidden_width, generator=generator),
        conv=0.01 * torch.randn(qkv_width, 4, generator=generator),
        a_log=0.01 * torch.randn(value_heads, generator=generator),
        dt_bias=0.01 * torch.randn(value_heads, generator=generator),
        norm=torch.ones(width),
        output=0.01 * torch.randn(2, value_heads * width, generator=generator),
    )
    initial = GDNState(
        conv_history=(0.01 * torch.randn(qkv_width, 3, generator=generator)).to(torch.bfloat16),
        recurrent=(0.01 * torch.randn(value_heads, width, width, generator=generator)).to(
            torch.float32
        ),
    )
    result = sublayer(
        hidden,
        weights,
        initial,
        key_heads=key_heads,
        value_heads=value_heads,
        key_width=width,
        value_width=width,
    )

    assert result.convolved_qkv.shape == (1, qkv_width)
    assert result.repeated_query.shape == (1, value_heads, width)
    assert result.repeated_key.shape == (1, value_heads, width)
    assert result.core.shape == (1, value_heads, width)
    assert result.output.shape == (1, 2)
    assert result.final_state.recurrent.shape == (value_heads, width, width)
    assert torch.isfinite(result.output).all()
    assert torch.isfinite(result.final_state.recurrent).all()

    # Independent witnesses at the public boundaries, using the real head and
    # state geometry rather than another call into the oracle helpers.
    projected_channel_zero = hidden.double()[0, 0] * weights.qkv.double()[0, 0]
    conv_window = torch.cat((initial.conv_history[0].double(), projected_channel_zero[None]))
    preactivation = torch.sum(conv_window * weights.conv.double()[0])
    expected_channel_zero = (preactivation / (1.0 + torch.exp(-preactivation))).to(
        torch.bfloat16
    )
    assert torch.equal(result.convolved_qkv[0, 0], expected_channel_zero)

    a = hidden.double()[0, 0] * weights.a.double()[:, 0]
    b = hidden.double()[0, 0] * weights.b.double()[:, 0]
    expected_g = (
        -torch.exp(weights.a_log.double())
        * torch.nn.functional.softplus(a + weights.dt_bias.double())
    ).to(torch.float32)
    expected_beta = torch.sigmoid(b).to(torch.float32)
    assert torch.equal(result.log_decay[0], expected_g)
    assert torch.equal(result.beta[0], expected_beta)

    q = result.convolved_qkv[0, :width].double()
    k = result.convolved_qkv[0, key_heads * width : (key_heads + 1) * width].double()
    v_offset = 2 * key_heads * width
    value = result.convolved_qkv[0, v_offset : v_offset + width].double()
    q = q / torch.sqrt(torch.sum(q * q) + 1e-6) / math.sqrt(width)
    k = k / torch.sqrt(torch.sum(k * k) + 1e-6)
    decay = initial.recurrent[0].double() * torch.exp(expected_g[0].double())
    prediction = torch.sum(decay * k[:, None], dim=0)
    delta = expected_beta[0].double() * (value - prediction)
    expected_state = (decay + k[:, None] * delta[None, :]).to(torch.float32)
    assert torch.equal(result.final_state.recurrent[0], expected_state)
    expected_core = torch.sum(expected_state.double() * q[:, None], dim=0).to(torch.bfloat16)
    assert torch.equal(result.core[0, 0], expected_core)
