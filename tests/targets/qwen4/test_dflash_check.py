"""Regressions for same-input oracles and exact representability diagnostics."""

import torch

from tools.parity.qwen4.native_dflash_check import EPSILON, LocalOracle, compare, nearest_bf16


def test_same_input_oracle_uses_actual_prior_gpu_output_not_propagated_reference():
    prior = torch.tensor([[2., -3., .5, 4.]], dtype=torch.float64)
    norm_weight = torch.tensor([.5, 1., 1.5, 2.], dtype=torch.bfloat16)
    oracle = LocalOracle({("LayerOutput", 0): prior},
                         {"layers.1.input_layernorm.weight": norm_weight},
                         {"input/noise_embeddings": torch.ones(1, 4)}, 0, 1)
    expected, kind, _ = oracle.expected("InputNorm", 1)
    direct = prior / torch.sqrt(prior.square().mean(-1, keepdim=True) + EPSILON) * norm_weight.double()
    torch.testing.assert_close(expected, direct, atol=1e-15, rtol=1e-15)
    assert kind == "rmsnorm"
    assert compare(nearest_bf16(expected), expected, kind)["passed"]
    assert not compare(torch.zeros_like(expected), expected, kind)["passed"]


def test_bf16_floor_does_not_silently_turn_failed_gate_into_pass():
    expected = torch.tensor([4.07759365476083], dtype=torch.float64)
    actual = torch.tensor([4.0625], dtype=torch.float64)
    metrics = compare(actual, expected, "rmsnorm")
    assert not metrics["passed"]
    assert metrics["bf16_representability"]["unavoidable_gross_violations"] == 1
    assert metrics["bf16_representability"]["actual_is_nearest_at_all_gross_violations"]


def test_nearest_bf16_resolves_double_rounding_and_even_ties():
    # FP32 cannot distinguish these FP64 neighbors around a BF16 tie.
    values = torch.tensor([4.078125 - 2. ** -35, 4.078125, 4.078125 + 2. ** -35,
                           -4.078125 + 2. ** -35, -4.078125, -4.078125 - 2. ** -35], dtype=torch.float64)
    expected = torch.tensor([4.0625, 4.0625, 4.09375, -4.0625, -4.0625, -4.09375], dtype=torch.float64)
    assert torch.equal(nearest_bf16(values), expected)


def test_attention_error_above_representability_floor_is_not_excused():
    expected = torch.tensor([4.593678281149969], dtype=torch.float64)
    metrics = compare(torch.tensor([4.625], dtype=torch.float64), expected, "attention")
    assert not metrics["passed"]
    assert metrics["bf16_representability"]["unavoidable_gross_violations"] == 0
    assert not metrics["bf16_representability"]["actual_is_nearest_at_all_gross_violations"]


def test_checker_rejects_nonfinite_and_rope_uses_input_pair_scale():
    assert not compare(torch.tensor([float("nan")]), torch.tensor([0.]), "linear")["passed"]
    pair = torch.tensor([[[1., 1.]]], dtype=torch.float64)
    expected = torch.tensor([[[0., 2. ** .5]]], dtype=torch.float64)
    assert compare(expected, expected, "rope", pair)["passed"]
    wrong = expected.clone()
    wrong[..., 0] = .02
    assert not compare(wrong, expected, "rope", pair)["passed"]
