import struct
from pathlib import Path

import pytest

from tools.parity.qwen4.paired_nll import evaluate, metrics, parse_external, parse_native


def test_trace_parsers_require_exact_token_alignment_and_f32_length(tmp_path: Path) -> None:
    external = tmp_path / "external.log"
    external.write_text(
        "TOKEN_NLL position=0 input_id=4 target_id=5 nll=0.25\n"
        "TOKEN_NLL position=1 input_id=5 target_id=6 nll=1.5\n",
        encoding="utf-8",
    )
    native = tmp_path / "native.nllf32"
    native.write_bytes(struct.pack("<2f", 0.5, 1.0))
    assert parse_external(external, [4, 5, 6]) == [0.25, 1.5]
    assert parse_native(native, 2) == [0.5, 1.0]
    with pytest.raises(ValueError, match="token pair mismatch"):
        parse_external(external, [4, 7, 6])
    with pytest.raises(ValueError, match="expected 12"):
        parse_native(native, 3)


def test_declared_per_token_gate_fails_independently_of_external_controls() -> None:
    reference = [0.0, 0.0, 0.0]
    native = [0.1, 0.2, 1.2]
    controls = [("schedule", reference, [0.2, 0.4, 1.5])]
    result = evaluate(reference, native, True, controls, 0.5, 1.0)
    assert result["passed"] is False
    assert result["mean_gate_passed"] is True
    assert result["per_token_gate_passed"] is False
    assert result["external_self_comparisons_affect_acceptance"] is False
    assert result["native_vs_reference"] == {
        "mean_absolute": 0.5,
        "maximum_absolute": 1.2,
        "count_absolute_strictly_greater_than_1_0": 1,
    }
    assert evaluate(reference, native, False, [], 0.5, 2.0)["passed"] is False
    assert evaluate(reference, [0.1, 0.2, 0.3], True, [], 0.5, 1.0)["passed"] is True


def test_metrics_reject_misaligned_inputs() -> None:
    with pytest.raises(ValueError, match="equal nonzero length"):
        metrics([1.0], [])
