#!/usr/bin/env python3
"""Run the Qwen3.8-27B NVFP4 accuracy campaign.

Default: 8k then 32k, both prefill and decode, BF16 then INT8 then NVFP4,
CUDA graphs on. Writes results.json / results.md and per-cell .nllf32 dumps
for token-level outliers and BF16 prefill-vs-decode |Δnll|.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import struct
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from schemes import BASELINE, ORDER, SCHEMES

DEFAULT_WEIGHTS = (
    "/ssdpool2nvme/local_llm/models/qwen3.8-nvfp4-Osfralla-mtp-ninfer/"
    "qwen3_8_27b_nvfp4.ninfer"
)
DEFAULT_TOKENS = 8192
LONG_TOKENS = 32768
PAGE = 64
MID_PAGE_SKIP = DEFAULT_TOKENS // 2 + PAGE // 2  # 4128: decode starts mid-page
SHORT_TOKENS = 257
SHORT_SKIP = 1
REPO = Path(__file__).resolve().parents[2]
SCHEDULES = ("prefill", "decode")
TERRIBLE_NLL = 10.0


def default_ppl_bin() -> Path:
    env = os.environ.get("NINFER_PPL")
    if env:
        return Path(env)
    return REPO / "build" / "apps" / "ninfer-ppl"


def parse_gates(values: list[str]) -> dict[str, float]:
    gates: dict[str, float] = {}
    for item in values:
        if "=" not in item:
            raise SystemExit(f"invalid --gate {item!r}; expected scheme=delta_mean_nll")
        name, raw = item.split("=", 1)
        if name not in SCHEMES:
            raise SystemExit(f"unknown scheme in --gate: {name}")
        gates[name] = float(raw)
    return gates


def select_schemes(names: list[str] | None) -> list[str]:
    if not names:
        return list(ORDER)
    selected: list[str] = []
    for name in names:
        if name not in SCHEMES:
            raise SystemExit(f"unknown scheme {name!r}; known: {', '.join(ORDER)}")
        if name not in selected:
            selected.append(name)
    if BASELINE not in selected:
        selected.insert(0, BASELINE)
    else:
        selected = [BASELINE] + [name for name in selected if name != BASELINE]
    return selected


def select_schedules(raw: str | None) -> list[str]:
    if not raw:
        return list(SCHEDULES)
    selected: list[str] = []
    for name in raw.split(","):
        name = name.strip()
        if name not in SCHEDULES:
            raise SystemExit(f"unknown schedule {name!r}; known: {', '.join(SCHEDULES)}")
        if name not in selected:
            selected.append(name)
    return selected


def corpus_token_count(path: Path) -> int:
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return 0
    return len(text.split())


def ensure_corpus(corpus: Path, tokens: int, weights: Path, ppl_bin: Path) -> None:
    if corpus.is_file() and corpus_token_count(corpus) >= tokens:
        return
    baker = Path(__file__).resolve().parent / "bake_corpus.py"
    cmd = [
        sys.executable,
        str(baker),
        "--weights",
        str(weights),
        "--tokens",
        str(tokens),
        "--out-dir",
        str(corpus.parent),
        "--ppl-bin",
        str(ppl_bin),
    ]
    print(f"baking corpus: {' '.join(cmd)}", file=sys.stderr)
    subprocess.run(cmd, check=True)
    if corpus_token_count(corpus) < tokens:
        raise SystemExit(f"baked corpus {corpus} has fewer than {tokens} tokens")


def load_nlls(cell_path: Path) -> list[float]:
    nll_path = cell_path.with_suffix(".nllf32")
    if not nll_path.is_file():
        return []
    data = nll_path.read_bytes()
    count = len(data) // 4
    return list(struct.unpack("<" + "f" * count, data[: count * 4]))


def paired_nll_stats(prefill: list[float], decode: list[float]) -> dict | None:
    n = min(len(prefill), len(decode))
    if n == 0:
        return None
    abs_delta = [abs(prefill[i] - decode[i]) for i in range(n)]
    return {
        "tokens": n,
        "mean_abs_delta_nll": sum(abs_delta) / n,
        "max_abs_delta_nll": max(abs_delta),
    }


def run_cell(
    ppl_bin: Path,
    weights: Path,
    ids: Path,
    scheme_name: str,
    schedule: str,
    skip: str,
    tokens: int,
    prefill_chunk: int,
    device: int,
    cell_path: Path,
    extra: list[str],
) -> dict:
    scheme = SCHEMES[scheme_name]
    cmd = [
        str(ppl_bin),
        "--weights",
        str(weights),
        "--ids",
        str(ids),
        "--scheme",
        scheme.name,
        "--kv-dtype",
        scheme.kv_dtype,
        "--schedule",
        schedule,
        "--skip",
        skip,
        "--tokens",
        str(tokens),
        "--prefill-chunk",
        str(prefill_chunk),
        "--device",
        str(device),
        "--out-json",
        str(cell_path),
        *scheme.extra_args,
        *extra,
    ]
    print(f"running {cell_path.name}: {' '.join(cmd)}", file=sys.stderr)
    subprocess.run(cmd, check=True)
    return json.loads(cell_path.read_text(encoding="utf-8"))


def expected_tokens_scored(cell: dict) -> int:
    """Prefill: n - skip - 1. Decode rewrites skip_tokens to prefix, so the same formula."""
    n = int(cell.get("prompt_tokens", 0))
    skip = int(cell.get("skip_tokens", 0))
    return n - skip - 1


def cell_ok(cell: dict) -> bool:
    expected = expected_tokens_scored(cell)
    return (
        cell.get("non_finite", 1) == 0
        and math.isfinite(float(cell.get("mean_nll", float("nan"))))
        and expected > 0
        and int(cell.get("tokens_scored", 0)) == expected
    )


def apply_baseline(cell: dict, baseline_nll: float | None, gates: dict[str, float], name: str) -> tuple[float | None, bool]:
    failed = False
    if name == BASELINE and baseline_nll is None:
        baseline_nll = cell["mean_nll"]
        cell["delta_mean_nll"] = 0.0
        cell["gate"] = None
        cell["pass"] = cell_ok(cell)
        failed = not cell["pass"]
        return baseline_nll, failed
    if baseline_nll is None:
        raise SystemExit("baseline scheme did not produce mean_nll")
    cell["delta_mean_nll"] = cell["mean_nll"] - baseline_nll
    if name in gates:
        cell["gate"] = gates[name]
        cell["pass"] = cell_ok(cell) and cell["delta_mean_nll"] <= gates[name]
    else:
        cell["gate"] = None
        cell["pass"] = cell_ok(cell)
    failed = not cell["pass"]
    return baseline_nll, failed


def write_markdown(path: Path, payload: dict) -> None:
    lines = [
        f"# Perplexity: {payload['model_id']} {payload['weights_id']}",
        "",
        f"- artifact: `{payload['weights']}`",
        f"- lengths: {payload['lengths']}",
        f"- skip default: {payload['skip']}",
        f"- cuda graphs: on unless a cell sets cuda_graph=false",
        f"- terrible token: nll >= {TERRIBLE_NLL}",
        f"- baseline: `{payload['baseline']}` per (length, schedule, spec)",
        "",
        "| length | schedule | scheme | spec | graph | skip | scored | mean_nll | max_nll | terrible | ppl | Δ mean_nll | gate |",
        "|---:|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for cell in payload["cells"]:
        delta = cell.get("delta_mean_nll")
        delta_text = "-" if delta is None else f"{delta:.6f}"
        gate = cell.get("gate")
        if gate is None:
            gate_text = "report"
        else:
            gate_text = "PASS" if cell.get("pass") else "FAIL"
        graph = "on" if cell.get("cuda_graph", True) else "off"
        lines.append(
            f"| {cell.get('prompt_tokens', '')} | {cell.get('schedule', '')} | `{cell['scheme']}` | "
            f"{cell.get('spec', 'none')} | {graph} | {cell.get('skip_tokens', '')} | "
            f"{cell['tokens_scored']} | {cell['mean_nll']:.6f} | {cell.get('max_nll', 0):.4f} | "
            f"{cell.get('terrible_tokens', 0)} | {cell['ppl']:.4f} | {delta_text} | {gate_text} |"
        )
    if payload.get("parity"):
        lines.extend(["", "## BF16 prefill vs decode |Δnll|", ""])
        for row in payload["parity"]:
            lines.append(
                f"- {row['tokens']} tokens: mean |Δ|={row['mean_abs_delta_nll']:.6f}, "
                f"max |Δ|={row['max_abs_delta_nll']:.6f}"
            )
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", type=Path, default=Path(DEFAULT_WEIGHTS))
    parser.add_argument("--ids", type=Path, default=Path(__file__).resolve().parent / "corpus.ids")
    parser.add_argument("--tokens", type=int, default=None, help="single length (default: 8k then 32k)")
    parser.add_argument("--long", action="store_true", help=f"only {LONG_TOKENS} tokens")
    parser.add_argument("--schemes", default=None, help="comma-separated scheme names")
    parser.add_argument(
        "--schedule",
        default="prefill,decode",
        help="prefill, decode, or comma-separated list",
    )
    parser.add_argument("--skip", default="half", help="warmup tokens not scored: half (default) or an integer")
    parser.add_argument("--gate", action="append", default=[], help="scheme=max_delta_mean_nll")
    parser.add_argument("--prefill-chunk", type=int, default=4096)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--ppl-bin", type=Path, default=default_ppl_bin())
    parser.add_argument("--no-extras", action="store_true", help="skip mid-page, short-context, graphs-off, mtp")
    parser.add_argument("--no-mtp", action="store_true", help="skip MTP decode cells")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()
    if args.long:
        lengths = [LONG_TOKENS]
    elif args.tokens is not None:
        lengths = [args.tokens]
    else:
        lengths = [DEFAULT_TOKENS, LONG_TOKENS]
    schemes = select_schemes(None if args.schemes is None else args.schemes.split(","))
    schedules = select_schedules(args.schedule)
    gates = parse_gates(args.gate)
    if not args.ppl_bin.is_file():
        raise SystemExit(f"ninfer-ppl not found: {args.ppl_bin} (build apps/ninfer-ppl)")
    if not args.weights.is_file():
        raise SystemExit(f"artifact not found: {args.weights}")

    ensure_corpus(args.ids, max(lengths), args.weights, args.ppl_bin)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%SZ")
    out_dir = args.out or (REPO / "profiles" / "ppl" / stamp)
    out_dir.mkdir(parents=True, exist_ok=True)

    cells: list[dict] = []
    failed = False

    def score_matrix(tokens: int) -> None:
        nonlocal failed
        for schedule in schedules:
            baseline_nll = None
            for name in schemes:
                cell_path = out_dir / f"{tokens}.{schedule}.{name}.json"
                cell = run_cell(
                    args.ppl_bin, args.weights, args.ids, name, schedule, args.skip,
                    tokens, args.prefill_chunk, args.device, cell_path, [],
                )
                baseline_nll, cell_failed = apply_baseline(cell, baseline_nll, gates, name)
                failed = failed or cell_failed
                cells.append(cell)

    def score_8k_extras() -> None:
        nonlocal failed
        extras = [
            ("midpage", str(MID_PAGE_SKIP), DEFAULT_TOKENS, []),
            ("short", str(SHORT_SKIP), SHORT_TOKENS, []),
            ("eager", "half", DEFAULT_TOKENS, ["--no-cuda-graph"]),
        ]
        for label, skip, tokens, extra in extras:
            cell_path = out_dir / f"{tokens}.decode.kv-bf16.{label}.json"
            cell = run_cell(
                args.ppl_bin, args.weights, args.ids, BASELINE, "decode", skip,
                tokens, args.prefill_chunk, args.device, cell_path, extra,
            )
            cell["delta_mean_nll"] = None
            cell["gate"] = None
            cell["pass"] = cell_ok(cell)
            failed = failed or not cell["pass"]
            cells.append(cell)

        if args.no_mtp:
            return
        mtp_baseline = None
        for name in schemes:
            cell_path = out_dir / f"{DEFAULT_TOKENS}.decode.{name}.mtp.json"
            cell = run_cell(
                args.ppl_bin, args.weights, args.ids, name, "decode", args.skip,
                DEFAULT_TOKENS, args.prefill_chunk, args.device, cell_path,
                ["--spec", "mtp", "--draft-tokens", "4"],
            )
            mtp_baseline, cell_failed = apply_baseline(cell, mtp_baseline, gates, name)
            failed = failed or cell_failed
            cells.append(cell)

    for tokens in lengths:
        score_matrix(tokens)
        if (
            tokens == DEFAULT_TOKENS
            and not args.no_extras
            and "decode" in schedules
        ):
            score_8k_extras()

    parity = []
    for tokens in lengths:
        prefill_path = out_dir / f"{tokens}.prefill.{BASELINE}.json"
        decode_path = out_dir / f"{tokens}.decode.{BASELINE}.json"
        stats = paired_nll_stats(load_nlls(prefill_path), load_nlls(decode_path))
        if stats:
            stats["tokens"] = tokens
            parity.append(stats)

    payload = {
        "weights": str(args.weights),
        "model_id": cells[0]["model_id"] if cells else "",
        "weights_id": cells[0]["weights_id"] if cells else "",
        "corpus": {"path": str(args.ids), "tokens": max(lengths)},
        "lengths": lengths,
        "skip": args.skip,
        "prefill_chunk": args.prefill_chunk,
        "schedules": schedules,
        "baseline": BASELINE,
        "gates": gates,
        "terrible_nll": TERRIBLE_NLL,
        "parity": parity,
        "cells": cells,
    }
    json_path = out_dir / "results.json"
    md_path = out_dir / "results.md"
    json_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    write_markdown(md_path, payload)
    print(f"ppl results: {out_dir}")
    print(f"  {json_path}")
    print(f"  {md_path}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
