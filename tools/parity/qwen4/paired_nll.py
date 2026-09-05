#!/usr/bin/env python3
"""Evaluate the declared Qwen4 paired-NLL integration gates from preserved traces."""

from __future__ import annotations

import argparse
import json
import math
import re
import struct
from dataclasses import asdict, dataclass
from pathlib import Path


_TOKEN_NLL = re.compile(
    r"^TOKEN_NLL position=(\d+) input_id=(\d+) target_id=(\d+) nll=([^ ]+)$"
)


@dataclass(frozen=True)
class Metrics:
    mean_absolute: float
    maximum_absolute: float
    count_absolute_strictly_greater_than_1_0: int


def frozen_prompt_ids(manifest: dict) -> list[int]:
    paragraph = manifest["tokenization"]["paragraph_with_lf_token_ids"]
    return paragraph * 6 + paragraph[:-1]


def parse_external(path: Path, prompt_ids: list[int]) -> list[float]:
    records: dict[int, tuple[int, int, float]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        match = _TOKEN_NLL.match(line)
        if match is None:
            continue
        position, input_id, target_id, nll = match.groups()
        key = int(position)
        if key in records:
            raise ValueError(f"{path}: duplicate TOKEN_NLL position {key}")
        records[key] = (int(input_id), int(target_id), float(nll))
    expected = len(prompt_ids) - 1
    if set(records) != set(range(expected)):
        raise ValueError(f"{path}: expected exactly positions 0..{expected - 1}")
    result: list[float] = []
    for position in range(expected):
        input_id, target_id, nll = records[position]
        if (input_id, target_id) != (prompt_ids[position], prompt_ids[position + 1]):
            raise ValueError(f"{path}: token pair mismatch at position {position}")
        if not math.isfinite(nll) or nll < 0.0:
            raise ValueError(f"{path}: invalid NLL at position {position}")
        result.append(nll)
    return result


def parse_native(path: Path, expected_count: int) -> list[float]:
    data = path.read_bytes()
    if len(data) != expected_count * 4:
        raise ValueError(f"{path}: expected {expected_count * 4} native NLL bytes")
    values = list(struct.unpack(f"<{expected_count}f", data))
    if any(not math.isfinite(value) or value < 0.0 for value in values):
        raise ValueError(f"{path}: native sidecar contains an invalid NLL")
    return values


def metrics(left: list[float], right: list[float]) -> Metrics:
    if len(left) != len(right) or not left:
        raise ValueError("paired NLL traces must have equal nonzero length")
    deltas = [abs(a - b) for a, b in zip(left, right)]
    return Metrics(
        mean_absolute=sum(deltas) / len(deltas),
        maximum_absolute=max(deltas),
        count_absolute_strictly_greater_than_1_0=sum(delta > 1.0 for delta in deltas),
    )


def evaluate(
    reference: list[float],
    native: list[float],
    native_replay_exact: bool,
    controls: list[tuple[str, list[float], list[float]]],
    mean_limit: float,
    per_token_limit: float,
) -> dict:
    if mean_limit < 0.0 or per_token_limit < 0.0:
        raise ValueError("paired NLL limits must be nonnegative")
    native_metrics = metrics(reference, native)
    control_metrics = {name: metrics(left, right) for name, left, right in controls}
    mean_gate_passed = native_metrics.mean_absolute <= mean_limit
    per_token_gate_passed = native_metrics.maximum_absolute <= per_token_limit
    passed = native_replay_exact and mean_gate_passed and per_token_gate_passed
    return {
        "schema": "ninfer.qwen4.paired_nll.v1",
        "transition_count": len(reference),
        "native_vs_reference": asdict(native_metrics),
        "external_self_comparisons": {
            name: asdict(value) for name, value in control_metrics.items()
        },
        "external_self_comparisons_affect_acceptance": False,
        "mean_absolute_limit": mean_limit,
        "per_token_absolute_limit": per_token_limit,
        "mean_gate_passed": mean_gate_passed,
        "per_token_gate_passed": per_token_gate_passed,
        "native_replay_exact": native_replay_exact,
        "passed": passed,
        "interpretation": (
            "external integration criterion only; controlled external self-sensitivity is "
            "diagnostic context and does not alter the declared gates; independent "
            "represented-input Op/state oracles remain the numerical correctness gates"
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--native", type=Path, required=True)
    parser.add_argument("--native-replay", type=Path, required=True)
    parser.add_argument(
        "--self-pair",
        action="append",
        nargs=3,
        metavar=("NAME", "LEFT_LOG", "RIGHT_LOG"),
        default=[],
        help="optional external self-comparison reported as diagnostic context only",
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    prompt_ids = frozen_prompt_ids(manifest)
    reference = parse_external(args.reference, prompt_ids)
    native = parse_native(args.native, len(reference))
    replay = parse_native(args.native_replay, len(reference))
    controls = [
        (name, parse_external(Path(left), prompt_ids), parse_external(Path(right), prompt_ids))
        for name, left, right in args.self_pair
    ]
    result = evaluate(
        reference,
        native,
        native == replay and args.native.read_bytes() == args.native_replay.read_bytes(),
        controls,
        float(
            manifest["paired_context_601"]["criterion"][
                "maximum_mean_absolute_delta_nll"
            ]
        ),
        float(
            manifest["paired_context_601"]["criterion"][
                "maximum_per_token_absolute_delta_nll"
            ]
        ),
    )
    encoded = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output is None:
        print(encoded, end="")
    else:
        args.output.write_text(encoded, encoding="utf-8")
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
