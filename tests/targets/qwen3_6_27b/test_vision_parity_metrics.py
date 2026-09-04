"""Fault-sensitivity checks for the exhaustive Vision parity metric."""

from __future__ import annotations

import torch

from tools.parity.qwen3_6_27b.vision import (
    LOCAL_GROUP_COSINE_MINIMUM,
    LOCAL_GROUP_SCALED_RMSE_LIMIT,
    WEIGHT_CRITERIA,
    metrics,
    stored_weight_group_metrics,
)


def _fails_group(value: dict[str, object], group: str) -> bool:
    result = value[group]
    return bool(
        result["worst_scaled_rmse"] > LOCAL_GROUP_SCALED_RMSE_LIMIT
        or result["worst_cosine"] < LOCAL_GROUP_COSINE_MINIMUM
    )


def test_group_metrics_reject_one_bad_token_hidden_by_aggregate() -> None:
    expected = torch.ones((1100, 5120), dtype=torch.bfloat16)
    actual = expected.clone()
    actual[731] = 0
    result = metrics(actual, expected)
    assert result["relative_rmse"] < 0.05
    assert result["cosine"] > 0.999
    assert _fails_group(result, "tokens")


def test_group_metrics_reject_one_bad_feature_hidden_by_aggregate() -> None:
    expected = torch.ones((1100, 5120), dtype=torch.bfloat16)
    actual = expected.clone()
    actual[:, 319] *= 2
    result = metrics(actual, expected)
    assert result["relative_rmse"] < 0.02
    assert result["cosine"] > 0.999
    assert _fails_group(result, "features")


def test_inactive_reference_group_uses_error_instead_of_unstable_cosine() -> None:
    expected = torch.ones((32, 64), dtype=torch.bfloat16)
    expected[:, 17] = 0
    actual = expected.clone()
    actual[:, 17] = 1.0e-4
    result = metrics(actual, expected)
    assert result["features"]["worst_cosine"] > 0.999
    assert not _fails_group(result, "features")


def test_stored_q4_group_gate_rejects_corruption_hidden_by_matrix_axes() -> None:
    expected = torch.ones((3456, 1152), dtype=torch.bfloat16)
    actual = expected.clone()
    actual[1729, 7 * 64 : 8 * 64] = 0
    aggregate = metrics(actual, expected)
    criterion = WEIGHT_CRITERIA["Q4G64_F16S"]
    assert aggregate["relative_rmse"] < criterion["relative_rmse"]
    assert aggregate["tokens"]["worst_scaled_rmse"] < criterion["row"]
    assert aggregate["features"]["worst_scaled_rmse"] < criterion["column"]
    stored = stored_weight_group_metrics(actual, expected, 64)
    assert stored["worst_scaled_rmse"] > criterion["stored_group"]
    assert stored["worst_cosine"] < criterion["stored_group_cosine"]


def test_stored_w8_group_gate_rejects_doubled_group_hidden_by_matrix_axes() -> None:
    expected = torch.ones((1152, 5120), dtype=torch.bfloat16)
    actual = expected.clone()
    actual[619, 31 * 32 : 32 * 32] *= 1.04
    aggregate = metrics(actual, expected)
    criterion = WEIGHT_CRITERIA["W8G32_F16S"]
    assert aggregate["relative_rmse"] < criterion["relative_rmse"]
    assert aggregate["tokens"]["worst_scaled_rmse"] < criterion["row"]
    assert aggregate["features"]["worst_scaled_rmse"] < criterion["column"]
    stored = stored_weight_group_metrics(actual, expected, 32)
    assert stored["worst_scaled_rmse"] > criterion["stored_group"]


def test_stored_group_gate_checks_partial_final_k_group() -> None:
    expected = torch.ones((4, 80), dtype=torch.bfloat16)
    actual = expected.clone()
    actual[2, 64:] = 0
    stored = stored_weight_group_metrics(actual, expected, 64)
    assert stored["logical_k"] == 80
    assert stored["padded_k"] == 128
    assert stored["count"] == 8
    assert stored["worst_scaled_rmse_location"] == {"row": 2, "k_group": 1}
    assert stored["worst_scaled_rmse"] == 1.0
    assert stored["worst_cosine"] == 0.0
