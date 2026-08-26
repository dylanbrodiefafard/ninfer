#!/usr/bin/env python3
"""INT8 vs NVFP4 KV context sweep for qwen3.8-27b/nvfp4 with MTP.

C=1 measures prefill and decode at 5k/20k/50k/100k/150k with chunk 1024.
C=2 measures concurrent Engine-lane decode at 20k/50k.
A separate C=1 chunk sweep at 100k/150k compares 1024/2048/4096.
Decode tok/s at C=2 is aggregate over both lanes.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Any, Sequence

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_BENCH = REPO_ROOT / "build/bench/ninfer_bench"
DEFAULT_WEIGHTS = REPO_ROOT / "out/qwen3_8_27b.ninfer"
DEFAULT_CORPUS = REPO_ROOT / "bench/fixtures/bench_corpus.ids"

C1_PROMPTS = (5000, 20000, 50000, 100000, 150000)
C2_PROMPTS = (20000, 50000)
CHUNK_SWEEP_PROMPTS = (100000, 150000)
CHUNK_SIZES = (1024, 2048, 4096)
DECODE_GEN = 128
KV_DTYPES = ("int8", "nvfp4")
PRIMARY_CHUNK = 1024


def shell_join(command: Sequence[str]) -> str:
    return " ".join(shlex.quote(str(part)) for part in command)


def utc_stamp() -> str:
    return dt.datetime.now(dt.UTC).strftime("%Y%m%d-%H%M%S")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bench", type=Path, default=DEFAULT_BENCH)
    parser.add_argument("--weights", type=Path, default=DEFAULT_WEIGHTS)
    parser.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS)
    parser.add_argument("--out-dir", type=Path, default=REPO_ROOT / "profiles/bench")
    parser.add_argument("--resume", type=Path, help="reuse an existing nvfp4-kv-* directory")
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def case_command(
    args: argparse.Namespace,
    kv_dtype: str,
    prompts: Sequence[int],
    concurrency: int,
    prefill_chunk: int,
) -> list[str]:
    pairs = ";".join(f"{p},{DECODE_GEN}" for p in prompts)
    return [
        str(args.bench),
        "--weights",
        str(args.weights),
        "--corpus",
        str(args.corpus),
        "--prompt-gen",
        pairs,
        "--kv-dtype",
        kv_dtype,
        "--concurrency",
        str(concurrency),
        "--spec",
        "mtp",
        "--draft-tokens",
        "3",
        "--lm-head-draft",
        "--prefill-chunk",
        str(prefill_chunk),
        "--repetitions",
        str(args.repetitions),
        "--warmup",
        str(args.warmup),
        "--output",
        "json",
    ]


def run_case(command: list[str], output_file: Path, dry_run: bool) -> dict[str, Any] | None:
    command = [*command, "--output-file", str(output_file)]
    print(shell_join(command), flush=True)
    if dry_run:
        return None
    output_file.parent.mkdir(parents=True, exist_ok=True)
    if output_file.is_file():
        print(f"skip existing {output_file}", flush=True)
        return json.loads(output_file.read_text())
    completed = subprocess.run(command, check=False)
    if completed.returncode != 0:
        raise SystemExit(f"ninfer_bench failed ({completed.returncode}): {shell_join(command)}")
    return json.loads(output_file.read_text())


def summarize(
    report: dict[str, Any], kv_dtype: str, concurrency: int, prefill_chunk: int
) -> list[dict[str, Any]]:
    rows = []
    for test in report.get("tests", []):
        rows.append(
            {
                "kv_dtype": kv_dtype,
                "concurrency": concurrency,
                "prefill_chunk": prefill_chunk,
                "label": test["label"],
                "n_prompt": test["n_prompt"],
                "n_gen": test["n_gen"],
                "prefill_tok_s_mean": test.get("prefill_tok_s_mean"),
                "prefill_tok_s_stddev": test.get("prefill_tok_s_stddev"),
                "decode_output_tok_s_mean": test.get("decode_output_tok_s_mean"),
                "decode_output_tok_s_stddev": test.get("decode_output_tok_s_stddev"),
                "decode_engine_tok_s_mean": test.get("decode_engine_tok_s_mean"),
                "decode_engine_tok_s_stddev": test.get("decode_engine_tok_s_stddev"),
            }
        )
    return rows


def main() -> int:
    args = parse_args()
    if not args.dry_run and not args.bench.is_file():
        raise SystemExit(f"missing ninfer_bench: {args.bench}")
    if not args.dry_run and not args.weights.is_file():
        raise SystemExit(f"missing artifact: {args.weights}")

    stamp = utc_stamp()
    out_dir = args.resume if args.resume is not None else args.out_dir / f"nvfp4-kv-{stamp}"
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_rows: list[dict[str, Any]] = []
    for kv_dtype in KV_DTYPES:
        c1_file = out_dir / f"{kv_dtype}-c1-chunk{PRIMARY_CHUNK}.json"
        c1 = run_case(
            case_command(args, kv_dtype, C1_PROMPTS, 1, PRIMARY_CHUNK), c1_file, args.dry_run
        )
        if c1 is not None:
            summary_rows.extend(summarize(c1, kv_dtype, 1, PRIMARY_CHUNK))
        c2_file = out_dir / f"{kv_dtype}-c2-chunk{PRIMARY_CHUNK}.json"
        c2 = run_case(
            case_command(args, kv_dtype, C2_PROMPTS, 2, PRIMARY_CHUNK), c2_file, args.dry_run
        )
        if c2 is not None:
            summary_rows.extend(summarize(c2, kv_dtype, 2, PRIMARY_CHUNK))
        for chunk in CHUNK_SIZES:
            if chunk == PRIMARY_CHUNK:
                continue
            chunk_file = out_dir / f"{kv_dtype}-c1-chunk{chunk}.json"
            chunk_report = run_case(
                case_command(args, kv_dtype, CHUNK_SWEEP_PROMPTS, 1, chunk),
                chunk_file,
                args.dry_run,
            )
            if chunk_report is not None:
                summary_rows.extend(summarize(chunk_report, kv_dtype, 1, chunk))

    if args.dry_run:
        return 0
    summary_path = out_dir / "summary.json"
    summary_path.write_text(json.dumps({"cases": summary_rows}, indent=2) + "\n")
    print(f"wrote {summary_path}", flush=True)
    print(json.dumps(summary_rows, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
