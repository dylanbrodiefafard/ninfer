#!/usr/bin/env python3
"""Compare Qwen4 represented-boundary summary traces without claiming tensor parity."""

from __future__ import annotations

import argparse
import csv
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


_EXTERNAL = re.compile(
    r"^NINFER_QWEN4_TAP name=(hc_init|attn_residual-(\d+)|ffn_residual-(\d+)|result_norm) "
    r"token=(\d+) sum=([^ ]+) sumsq=([^ ]+) max_abs=([^ ]+)$"
)
_NATIVE = re.compile(
    r"^TENSOR position=(\d+) seam=(attn_residual|ffn_residual|final_gr) layer=(\d+) "
    r"elements=\d+ dtype=bf16 sum=([^ ]+) sumsq=([^ ]+) max_abs=([^ ]+)"
)
_COMMON_DIRECT_SEAMS = tuple(
    (seam, layer)
    for layer in range(47)
    for seam in ("attn_residual", "ffn_residual")
)


@dataclass(frozen=True)
class Sample:
    position: int
    seam: str
    layer: int
    total: float
    sumsq: float
    max_abs: float

    @property
    def l2(self) -> float:
        return math.sqrt(self.sumsq)


def _finite(sample: Sample, path: Path) -> Sample:
    if sample.sumsq < 0.0 or sample.max_abs < 0.0 or not all(
        math.isfinite(value) for value in (sample.total, sample.sumsq, sample.max_abs)
    ):
        raise ValueError(f"{path}: invalid summary at {sample.seam}/{sample.layer}")
    return sample


def parse_native(path: Path) -> dict[tuple[int, str, int], Sample]:
    result: dict[tuple[int, str, int], Sample] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        match = _NATIVE.match(line)
        if match is None:
            continue
        position, seam, layer, total, sumsq, maximum = match.groups()
        sample = _finite(
            Sample(int(position), seam, int(layer), float(total), float(sumsq), float(maximum)),
            path,
        )
        key = (sample.position, sample.seam, sample.layer)
        if key in result:
            raise ValueError(f"{path}: duplicate native record {key}")
        result[key] = sample
    if not result:
        raise ValueError(f"{path}: no native tensor summaries")
    return result


def _expected_local_tokens(token_count: int, microbatch_size: int) -> list[int]:
    if token_count <= 0 or microbatch_size <= 0:
        raise ValueError("token count and external microbatch size must be positive")
    return [
        local
        for begin in range(0, token_count, microbatch_size)
        for local in range(min(microbatch_size, token_count - begin))
    ]


def parse_external(
    path: Path,
    token_count: int,
    microbatch_size: int,
) -> dict[tuple[int, str, int], Sample]:
    expected_tokens = _expected_local_tokens(token_count, microbatch_size)
    series: dict[tuple[str, int], list[tuple[int, float, float, float]]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        match = _EXTERNAL.match(line)
        if match is None:
            continue
        name, attention_layer, ffn_layer, local_token, total, sumsq, maximum = match.groups()
        if name == "hc_init":
            seam, layer = "hc_init", -1
        elif name == "result_norm":
            # llama-debug requests logits only for selected output rows. In tokenwise prompt
            # processing it still invokes the callback for skipped rows backed by zero storage,
            # so occurrence count cannot establish an aligned final-GR series.
            continue
        elif attention_layer is not None:
            seam, layer = "attn_residual", int(attention_layer)
        else:
            seam, layer = "ffn_residual", int(ffn_layer)
        # The last decoder layer and final mixer see only llama.cpp's requested output rows.
        # They are the only explicitly excluded direct series.
        if layer == 47:
            continue
        series.setdefault((seam, layer), []).append(
            (int(local_token), float(total), float(sumsq), float(maximum))
        )

    expected_series = {("hc_init", -1)} | {
        (seam, layer)
        for layer in range(47)
        for seam in ("attn_residual", "ffn_residual")
    }
    missing = sorted(expected_series - series.keys())
    unexpected = sorted(series.keys() - expected_series)
    if missing:
        raise ValueError(f"{path}: missing required external series {missing[0]}")
    if unexpected:
        raise ValueError(f"{path}: unexpected direct external series {unexpected[0]}")

    result: dict[tuple[int, str, int], Sample] = {}
    for seam_layer in sorted(expected_series):
        values = series[seam_layer]
        seam, layer = seam_layer
        local_tokens = [value[0] for value in values]
        if local_tokens != expected_tokens:
            mismatch = next(
                (
                    index
                    for index, (actual, expected) in enumerate(
                        zip(local_tokens, expected_tokens)
                    )
                    if actual != expected
                ),
                min(len(local_tokens), len(expected_tokens)),
            )
            raise ValueError(
                f"{path}: invalid local-token segmentation for {seam_layer} "
                f"at occurrence {mismatch}; expected {len(expected_tokens)} records"
            )
        for position, (_, total, sumsq, maximum) in enumerate(values):
            sample = _finite(Sample(position, seam, layer, total, sumsq, maximum), path)
            result[(position, seam, layer)] = sample
    return result


def compare(
    left: dict[tuple[int, str, int], Sample],
    right: dict[tuple[int, str, int], Sample],
    positions: Iterable[int],
) -> list[dict[str, int | float | str]]:
    wanted = set(positions)
    if not wanted:
        raise ValueError("at least one comparison position is required")

    for side, samples in (("left", left), ("right", right)):
        for position in sorted(wanted):
            for seam, layer in _COMMON_DIRECT_SEAMS:
                key = (position, seam, layer)
                if key not in samples:
                    raise ValueError(
                        f"{side} trace missing required record "
                        f"position={position} seam={seam} layer={layer}"
                    )

    rows: list[dict[str, int | float | str]] = []
    for position in sorted(wanted):
        for seam, layer in _COMMON_DIRECT_SEAMS:
            key = (position, seam, layer)
            lhs, rhs = left[key], right[key]
            denominator = max(lhs.l2, rhs.l2, math.ulp(0.0))
            rows.append(
                {
                    "position": position,
                    "seam": seam,
                    "layer": layer,
                    "left_l2": lhs.l2,
                    "right_l2": rhs.l2,
                    "signed_left_over_right_l2_minus_one": (
                        lhs.l2 / rhs.l2 - 1.0
                        if rhs.l2 != 0.0
                        else (0.0 if lhs.l2 == 0.0 else math.inf)
                    ),
                    "relative_l2_norm_delta": abs(lhs.l2 - rhs.l2) / denominator,
                    "left_sum": lhs.total,
                    "right_sum": rhs.total,
                    "left_max_abs": lhs.max_abs,
                    "right_max_abs": rhs.max_abs,
                }
            )
    return rows


def _parse_positions(value: str) -> list[int]:
    result = [int(item) for item in value.split(",")]
    if not result or any(position < 0 for position in result):
        raise argparse.ArgumentTypeError("positions must be nonnegative comma-separated integers")
    return result


def _parse(
    path: Path,
    kind: str,
    token_count: int,
    external_microbatch_size: int | None,
) -> dict[tuple[int, str, int], Sample]:
    if kind == "native":
        return parse_native(path)
    if external_microbatch_size is None:
        raise ValueError("external traces require their exact microbatch size")
    return parse_external(path, token_count, external_microbatch_size)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Compare Qwen4 boundary norms for localization; this is not a parity oracle."
    )
    parser.add_argument("--left", type=Path, required=True)
    parser.add_argument("--left-kind", choices=("native", "external"), required=True)
    parser.add_argument("--right", type=Path, required=True)
    parser.add_argument("--right-kind", choices=("native", "external"), required=True)
    parser.add_argument("--left-external-microbatch-size", type=int)
    parser.add_argument("--right-external-microbatch-size", type=int)
    parser.add_argument("--token-count", type=int, default=601)
    parser.add_argument("--positions", type=_parse_positions, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.token_count <= 0:
        parser.error("--token-count must be positive")

    rows = compare(
        _parse(
            args.left,
            args.left_kind,
            args.token_count,
            args.left_external_microbatch_size,
        ),
        _parse(
            args.right,
            args.right_kind,
            args.token_count,
            args.right_external_microbatch_size,
        ),
        args.positions,
    )
    if not rows:
        raise SystemExit("no aligned records for the requested positions")
    fieldnames = list(rows[0])
    if args.output is None:
        writer = csv.DictWriter(__import__("sys").stdout, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    else:
        with args.output.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
