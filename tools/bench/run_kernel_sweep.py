#!/usr/bin/env python3
"""Sweep attention-kernel variants across the PPL + in-process speed bench.

Each kernel variant is selected by env vars the engine reads at launch
(NINFER_S3_TMA, NINFER_TMA_STAGES, ...). PPL runs offline (ninfer-ppl), and
the speed bench runs in-process (ninfer_bench) -- no live server, no engine
restart, so the production serve is never touched.

Add a new kernel variant by appending one entry to KERNELS below; the driver
runs the same cells against every entry and prints a side-by-side table.

Usage (inside the GPU container with the build tree):
  python3 tools/bench/run_kernel_sweep.py --bin-dir /build-dev \
      --weights /models/qwen3_8_27b_nvfp4.ninfer
  python3 tools/bench/run_kernel_sweep.py --bench ppl --kernels cpasync,tma-s2
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Sequence

REPO_ROOT = Path(__file__).resolve().parents[2]

# --- kernel variants: name + env the engine sees at launch ------------------
# Extend this list as new attention implementations land (e.g. the
# warp-level-sync decode rework). env is merged over the process env, so a
# variant is fully self-describing; extra PPL flags are only for the cell JSON.
KERNELS: list[dict[str, Any]] = [
    {"name": "cpasync", "env": {}, "ppl_args": []},
    {"name": "tma-s2", "env": {"NINFER_S3_TMA": "1"}, "ppl_args": ["--s3-tma"]},
    {"name": "tma-s3", "env": {"NINFER_S3_TMA": "1", "NINFER_TMA_STAGES": "3"},
     "ppl_args": ["--s3-tma"]},
    # {"name": "wsync", "env": {"NINFER_S3_WSYNC": "1"}, "ppl_args": []},
]

# PPL cells: (scheme, kv_dtype, extra ninfer-ppl flags). kv-bf16 is kernel-
# agnostic and is run once for the whole sweep; attn-sage is the S3 prefill
# lane every variant re-implements, so it runs per kernel.
PPL_CELLS: list[tuple[str, str, list[str]]] = [
    ("kv-bf16", "bf16", []),
    ("attn-sage", "nvfp4", ["--sage"]),
]
SHARED_CELLS = {"kv-bf16"}

# Speed-bench cases for ninfer_bench (same arg style as run_ninfer_bench_matrix.py).
SPEED_CASES_CORE: list[tuple[str, str, list[str]]] = [
    ("prefill", "prefill_p8192_k3", ["-p", "8192", "--mtp-draft-tokens", "3", "--lm-head-draft"]),
    ("decode", "tg64_k3", ["-n", "64", "--mtp-draft-tokens", "3", "--lm-head-draft"]),
    ("ctx", "ctx_p8192_g64_k3",
     ["-pg", "8192,64", "--max-ctx", "8256", "--mtp-draft-tokens", "3", "--lm-head-draft"]),
]
SPEED_CASES_SMOKE: list[tuple[str, str, list[str]]] = [
    ("prefill", "prefill_p1024_k3", ["-p", "1024", "--mtp-draft-tokens", "3", "--lm-head-draft"]),
    ("decode", "tg16_k3", ["-n", "16", "--mtp-draft-tokens", "3", "--lm-head-draft"]),
]

WEIGHTS_CANDIDATES = [
    "/models/qwen3_8_27b_nvfp4.ninfer",
    "/ssdpool2nvme/local_llm/qwen3_8_27b_nvfp4.ninfer",
    str(REPO_ROOT / "out/qwen3_6_27b.ninfer"),
]
PPL_CORPUS_CANDIDATES = ["/tmp/corpus.ids", str(REPO_ROOT / "tools/ppl/corpus.ids")]
BENCH_CORPUS_CANDIDATES = [
    "/tmp/bench_corpus.ids",
    str(REPO_ROOT / "bench/fixtures/bench_corpus.ids"),
]


def first_existing(candidates: Sequence[str]) -> Path:
    for candidate in candidates:
        if Path(candidate).is_file():
            return Path(candidate)
    return Path(candidates[0])


def utc_stamp() -> str:
    return dt.datetime.now(dt.UTC).strftime("%Y%m%d-%H%M%S")


def run(cmd: list[str], env: dict[str, str], log: Path) -> int:
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("w", encoding="utf-8") as handle:
        handle.write("$ " + " ".join(cmd) + "\nenv: " + json.dumps(env, sort_keys=True) + "\n")
        handle.flush()
        process = subprocess.run(cmd, env=env, cwd=REPO_ROOT, stdout=handle, stderr=subprocess.STDOUT,
                                 check=False)
    return process.returncode


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bin-dir", type=Path, default=REPO_ROOT / "build",
                        help="dir containing ninfer-ppl and bench/ninfer_bench")
    parser.add_argument("--kernels", default=None, help="comma-separated KERNELS names")
    parser.add_argument("--bench", choices=("ppl", "speed", "both"), default="both")
    parser.add_argument("--weights", type=Path, default=None)
    parser.add_argument("--ppl-corpus", type=Path, default=None)
    parser.add_argument("--bench-corpus", type=Path, default=None)
    parser.add_argument("--ppl-tokens", type=int, default=8192)
    parser.add_argument("--ppl-schedule", choices=("prefill", "decode"), default="prefill")
    parser.add_argument("--speed-cases", choices=("core", "smoke", "none"), default="core")
    parser.add_argument("--reps", type=int, default=2)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    known = {entry["name"] for entry in KERNELS}
    selected = args.kernels.split(",") if args.kernels else sorted(known)
    unknown = [name for name in selected if name not in known]
    if unknown:
        raise SystemExit(f"unknown kernels {unknown}; known: {sorted(known)}")
    kernels = [entry for entry in KERNELS if entry["name"] in selected]

    weights = args.weights or first_existing(WEIGHTS_CANDIDATES)
    ppl_corpus = args.ppl_corpus or first_existing(PPL_CORPUS_CANDIDATES)
    bench_corpus = args.bench_corpus or first_existing(BENCH_CORPUS_CANDIDATES)
    for label, path in (("weights", weights), ("PPL corpus", ppl_corpus)):
        if not path.is_file():
            raise SystemExit(f"{label} not found: {path} (pass it explicitly)")

    bin_dir = args.bin_dir.expanduser().resolve()

    def find_binary(rel: Sequence[str], layouts: Sequence[Sequence[str]]) -> Path:
        for layout in layouts:
            candidate = bin_dir.joinpath(*layout, rel[0])
            if candidate.is_file():
                return candidate
        return bin_dir / rel[0]

    ppl_bin = find_binary(("ninfer-ppl",), (("apps",), ()))
    speed_bin = find_binary(("ninfer_bench",), (("bench",), ()))
    for binary in (ppl_bin, speed_bin):
        if args.bench in ("ppl", "both") and binary == ppl_bin and not binary.is_file():
            raise SystemExit(f"{binary} missing (build it first)")
        if args.bench in ("speed", "both") and binary == speed_bin and not binary.is_file():
            raise SystemExit(f"{binary} missing (build it first)")

    out_dir = args.out or Path("/tmp") / f"kernel-sweep-{utc_stamp()}"
    out_dir.mkdir(parents=True, exist_ok=True)
    speed_cases = (
        SPEED_CASES_CORE if args.speed_cases == "core"
        else SPEED_CASES_SMOKE if args.speed_cases == "smoke" else []
    )

    summary: dict[str, Any] = {
        "started": dt.datetime.now(dt.UTC).isoformat(),
        "weights": str(weights),
        "kernels": [entry["name"] for entry in kernels],
        "results": {},
    }
    cell_cache: dict[str, Path] = {}

    for kernel in kernels:
        name = kernel["name"]
        env = {**os.environ, **kernel["env"]}
        print(f"=== kernel: {name}  env={kernel['env']}")
        record: dict[str, Any] = {"env": kernel["env"], "ppl": {}, "speed": {}}

        if args.bench in ("ppl", "both"):
            for scheme, kv_dtype, extra in PPL_CELLS:
                if scheme in SHARED_CELLS and cell_cache.get(scheme):
                    path = cell_cache[scheme]
                else:
                    path = out_dir / name / f"ppl-{scheme}.json"
                    cmd = [str(ppl_bin), "--weights", str(weights), "--ids", str(ppl_corpus),
                           "--scheme", scheme, "--kv-dtype", kv_dtype,
                           "--schedule", args.ppl_schedule, "--tokens", str(args.ppl_tokens),
                           "--skip", "half", "--out-json", str(path)] + list(extra) + list(
                               kernel.get("ppl_args", []))
                    rc = run(cmd, env, out_dir / name / f"ppl-{scheme}.log")
                    if rc != 0:
                        print(f"  PPL {scheme}: FAILED (rc={rc}); see {out_dir / name / (scheme + '.log')}")
                        continue
                    if scheme in SHARED_CELLS:
                        cell_cache[scheme] = path
                data = read_json(path)
                record["ppl"][scheme] = {
                    "mean_nll": data.get("mean_nll"),
                    "tokens_scored": data.get("tokens_scored"),
                    "non_finite": data.get("non_finite"),
                }
                print(f"  PPL {scheme}: mean_nll={record['ppl'][scheme]['mean_nll']}")

        if args.bench in ("speed", "both") and speed_cases:
            for suite, case_name, case_args in speed_cases:
                path = out_dir / name / f"speed-{suite}-{case_name}.json"
                cmd = [str(speed_bin), "--weights", str(weights), "--corpus", str(bench_corpus),
                       "--device", "0", *case_args,
                       "-r", str(args.reps), "--warmup", str(args.warmup),
                       "--output", str(path)]
                rc = run(cmd, env, out_dir / name / f"speed-{suite}-{case_name}.log")
                if rc != 0:
                    print(f"  SPEED {case_name}: FAILED (rc={rc})")
                    continue
                report = read_json(path)
                for test in report.get("tests", []):
                    spec = test.get("speculative", {})
                    record["speed"][case_name] = {
                        "suite": suite,
                        "prefill_tok_s": test.get("prefill_tok_s_mean"),
                        "decode_tok_s": test.get("decode_output_tok_s_mean"),
                        "accept_rate": spec.get("acceptance_rate"),
                    }
                    print(f"  SPEED {case_name}: prefill={record['speed'][case_name]['prefill_tok_s']}"
                          f" decode={record['speed'][case_name]['decode_tok_s']}"
                          f" accept={record['speed'][case_name]['accept_rate']}")

        summary["results"][name] = record

    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    # --- side-by-side table ---------------------------------------------------
    def fmt(value: Any) -> str:
        if value is None:
            return "-"
        if isinstance(value, float):
            return f"{value:.4g}"
        return str(value)

    lines: list[str] = []
    lines.append("| metric | " + " | ".join(k["name"] for k in kernels) + " |")
    lines.append("|---" * (len(kernels) + 1) + "|")

    def row(label: str, getter) -> None:
        lines.append(f"| {label} | " + " | ".join(
            fmt(getter(summary["results"].get(name, {}))) for name in (k["name"] for k in kernels))
         + " |")

    row("ppl:bf16 mean_nll", lambda r: r.get("ppl", {}).get("kv-bf16", {}).get("mean_nll"))
    row("ppl:sage mean_nll", lambda r: r.get("ppl", {}).get("attn-sage", {}).get("mean_nll"))
    if speed_cases:
        row("prefill tok/s (8k)", lambda r: r.get("speed", {}).get("prefill_p8192_k3", {}).get("prefill_tok_s")
            if args.speed_cases == "core" else
            r.get("speed", {}).get("prefill_p1024_k3", {}).get("prefill_tok_s"))
        row("decode tok/s", lambda r: r.get("speed", {}).get(
            "tg64_k3" if args.speed_cases == "core" else "tg16_k3", {}).get("decode_tok_s"))
        row("accept rate", lambda r: r.get("speed", {}).get(
            "ctx_p8192_g64_k3" if args.speed_cases == "core" else "tg16_k3", {}).get("accept_rate"))

    table = "\n".join(lines)
    print("\n=== sweep table ===\n" + table)
    (out_dir / "summary.md").write_text(table + "\n", encoding="utf-8")
    print(f"raw results + logs: {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())