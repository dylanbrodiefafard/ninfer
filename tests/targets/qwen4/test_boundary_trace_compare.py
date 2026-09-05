from pathlib import Path

import pytest

from tools.parity.qwen4.compare_boundary_traces import Sample, compare, parse_external, parse_native


def _complete_direct_samples(positions: list[int], sumsq: float) -> dict[tuple[int, str, int], Sample]:
    return {
        (position, seam, layer): Sample(
            position,
            seam,
            layer,
            1.0,
            sumsq,
            sumsq**0.5,
        )
        for position in positions
        for layer in range(47)
        for seam in ("attn_residual", "ffn_residual")
    }


def _external_lines(token_count: int, microbatch_size: int) -> list[str]:
    local_tokens = [
        local
        for begin in range(0, token_count, microbatch_size)
        for local in range(min(microbatch_size, token_count - begin))
    ]
    names = ["hc_init"] + [
        f"{seam}-{layer}"
        for layer in range(47)
        for seam in ("attn_residual", "ffn_residual")
    ]
    return [
        f"NINFER_QWEN4_TAP name={name} token={token} sum=1 sumsq=4 max_abs=2"
        for name in names
        for token in local_tokens
    ]


def test_external_parser_ignores_derived_and_incomplete_series(tmp_path: Path) -> None:
    trace = tmp_path / "external.log"
    lines = _external_lines(2, 2)
    lines.extend(
        (
            "NINFER_QWEN4_TAP name=attn_residual-2 (reshaped) token=0 sum=99 sumsq=99 max_abs=99",
            "NINFER_QWEN4_TAP name=attn_residual-47 token=0 sum=1 sumsq=1 max_abs=1",
            "NINFER_QWEN4_TAP name=result_norm token=0 sum=1 sumsq=1 max_abs=1",
        )
    )
    trace.write_text("\n".join(lines), encoding="utf-8")
    parsed = parse_external(trace, token_count=2, microbatch_size=2)
    assert parsed[(0, "attn_residual", 2)].l2 == 2.0
    assert parsed[(1, "attn_residual", 2)].l2 == 2.0
    assert all(sample.seam != "final_gr" for sample in parsed.values())


@pytest.mark.parametrize(
    ("mutation", "message"),
    (
        ("reorder", "invalid local-token segmentation"),
        ("duplicate", "invalid local-token segmentation"),
        ("missing", "missing required external series"),
        ("wrong_segment", "invalid local-token segmentation"),
    ),
)
def test_external_parser_rejects_incomplete_or_misaligned_series(
    tmp_path: Path, mutation: str, message: str
) -> None:
    lines = _external_lines(3, 2)
    if mutation == "reorder":
        lines[1], lines[2] = lines[2], lines[1]
    elif mutation == "duplicate":
        lines.insert(1, lines[0])
    elif mutation == "missing":
        lines = [line for line in lines if "name=ffn_residual-46 " not in line]
    else:
        # The records describe one width-three microbatch, not the declared 2+1 segmentation.
        lines[2] = lines[2].replace("token=0", "token=2")
    trace = tmp_path / f"{mutation}.log"
    trace.write_text("\n".join(lines), encoding="utf-8")
    with pytest.raises(ValueError, match=message):
        parse_external(trace, token_count=3, microbatch_size=2)


def test_native_parser_accepts_a_single_record_for_low_level_inspection(tmp_path: Path) -> None:
    trace = tmp_path / "native.log"
    trace.write_text(
        "TENSOR position=7 seam=ffn_residual layer=3 elements=2 dtype=bf16 "
        "sum=1 sumsq=16 max_abs=4 raw_fnv1a64=0000000000000000\n",
        encoding="utf-8",
    )
    native = parse_native(trace)
    assert native[(7, "ffn_residual", 3)].l2 == 4.0


def test_comparator_reports_complete_direct_relative_norm_deltas() -> None:
    native = _complete_direct_samples([7], 16.0)
    external = _complete_direct_samples([7], 25.0)
    rows = compare(native, external, [7])
    assert len(rows) == 94
    target = next(
        row for row in rows if row["seam"] == "ffn_residual" and row["layer"] == 3
    )
    assert target["relative_l2_norm_delta"] == pytest.approx(0.2)


def test_comparator_rejects_a_missing_native_seam() -> None:
    native = _complete_direct_samples([7], 16.0)
    external = _complete_direct_samples([7], 25.0)
    del native[(7, "ffn_residual", 46)]

    with pytest.raises(
        ValueError,
        match=r"left trace missing required record position=7 seam=ffn_residual layer=46",
    ):
        compare(native, external, [7])


def test_comparator_rejects_a_missing_requested_native_probe() -> None:
    native = _complete_direct_samples([7], 16.0)
    external = _complete_direct_samples([7, 8], 25.0)

    with pytest.raises(
        ValueError,
        match=r"left trace missing required record position=8 seam=attn_residual layer=0",
    ):
        compare(native, external, [7, 8])


def test_parser_rejects_nonfinite_summaries(tmp_path: Path) -> None:
    trace = tmp_path / "bad.log"
    trace.write_text(
        "TENSOR position=0 seam=final_gr layer=48 elements=1 dtype=bf16 "
        "sum=nan sumsq=1 max_abs=1\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="invalid summary"):
        parse_native(trace)
