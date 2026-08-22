"""sage: SageAttention3 (nvfp4s3) oracle + FP4-P floor decomposition.

The sage quality question is "bug, floor, or tile-skip?" — this mode runs the
repo's own conformance test (tests/ops/test_gqa_attention.cpp) in sage-only mode
with the per-case floor diagnostic (GQA_SAGE_FLOOR), and parses each case's
SAGE_FLOOR line into a decomposition:

  floor           = rel_L2(sage_ideal, ideal)   — the FP4 P-quant cost that NO
                   correct kernel can avoid (documented 0.053-0.059 on this op)
  device_vs_exact = rel_L2(kernel, ideal)       — floor + bug + tile-skip
  dev_vs_step_*   = rel_L2(kernel, step-exact FP64 emulation of the kernel's
                   online-softmax + e2m1 P-quant loop). This is the BUG signal:
                   a correct kernel sits within ~1e-4 (FP32 exp2/divide noise);
                   O(0.01+) localizes a real kernel bug (see the test's comment
                   at sage_step_emulation).

Env knobs (all set by this module; the test has no CLI for them):
  GQA_SAGE_ONLY=1    run only the sage (U8 nvfp4s3) cases
  GQA_SAGE_FLOOR=1   print the per-case SAGE_FLOOR decomposition
  GQA_SAGE_FAST=1    drop the two T=128 multi-tile A1 cases (fast tier)
  GQA_KEEP_FRAC=x    A/B the tile-skip lever: the test allocates the sage k_mean
                     proxy plane and the A1 Prompt route engages tile-skip; the
                     bench runs with NINFER_KEEP_FRAC=x. keep_frac<1 measures
                     tile-skip loss as device_vs_exact - floor.

Usage:
    python3 -m tools.kdev gqa_attention --sage [--fast] [--keep-frac 0.2] [--bench]
"""

from __future__ import annotations

import json
import os
import re
import time

from . import harness, registry, verdict

# A correct kernel sits ~1e-4 from the step-exact emulation (its residual is FP32
# exp2/divide/accumulate noise). O(0.01+) is bug-scale (test_gqa_attention.cpp,
# sage_step_emulation comment). 1e-3 = 10x the documented noise band.
_NOISE_BAND = 1e-3

_FLOOR_LINE_RE = re.compile(r"^(?P<label>.+?) SAGE_FLOOR: (?P<kv>.*)$")
_KV_RE = re.compile(r"([a-z_0-9]+)=(\S+)")
# nvfp4s3 bench table row: context window fill(us) attn_median(us) attn_p95(us) gbps
_BENCH_ROW_RE = re.compile(
    r"^(?P<ctx>\d+)\s+(?P<window>\d+)\s+(?P<fill>[\d.]+)\s+(?P<median>[\d.]+)\s+"
    r"(?P<p95>[\d.]+)\s+(?P<gbps>[\d.]+)\s*$", re.M,
)


def _bug_class(value: float, floor: float | None = None) -> str:
    # A correct kernel sits within ~1e-4 of its kernel-faithful step emulation
    # (test comment). Bands are stated against the case's P-quant floor so the
    # verdict reads as "how much extra error does the kernel add beyond the floor".
    if value < _NOISE_BAND:
        return "noise"
    if value < 0.02:
        return "sub-floor-dev"
    if floor is not None and value < floor:
        return "sub-floor-dev"
    return ">=floor-dev"


def _parse_floor_lines(output: str) -> list:
    cases = []
    for line in output.splitlines():
        match = _FLOOR_LINE_RE.match(line.strip())
        if not match:
            continue
        label = match.group("label")
        values = {k: v for k, v in _KV_RE.findall(match.group("kv"))}
        record = {"case": label}
        for key in ("floor", "device_vs_exact", "device_vs_sage", "dev_vs_step_pref64",
                    "dev_vs_step_dec32", "step64_vs_sage", "step32_vs_sage",
                    "step64_vs_exact", "n"):
            if key in values:
                record[key] = float(values[key])
        # A1 (gqa_attention, prefill s3, Bc=64 tiles) uses the prefill step
        # emulation; A3 (gqa_attention_cached, small-t decode, Bc=32) uses the
        # decode one. The bug metric is whichever matches the route.
        if label.startswith("gqa_attention_cached"):
            record["route"] = "A3"
            # Best mirror depends on the decode KeyBlock tier: window <= 2054 runs the
            # Bc=32 tier (step32 mirrors it); window 2055..8198 runs the Bc=64
            # dynamic-arena tier, whose 64-key-tile P-quant structure the prefill
            # step64 emulation mirrors better. Use the smaller residual of the two.
            a3_window = int(label.rsplit("envelope_max=", 1)[-1]) if "envelope_max=" in label else 0
            record["bc64_tier"] = 2054 < a3_window <= 8198
            if record["bc64_tier"]:
                record["bug_residual"] = min(
                    record.get("dev_vs_step_pref64", float("inf")),
                    record.get("dev_vs_step_dec32", float("inf")))
            else:
                record["bug_residual"] = record.get("dev_vs_step_dec32")
        else:
            record["route"] = "A1"
            record["bug_residual"] = record.get("dev_vs_step_pref64")
        # Single-tile A1 (keys <= 64): the step64 emulation is bit-identical to the
        # closed-form sage_ideal (one tile => running max == block max), so the
        # step residual is NOT an independent signal there — only multi-tile A1
        # cases can classify a kernel bug.
        keys = int(label.rsplit("keys=", 1)[-1].split()[0]) if "keys=" in label else 0
        record["single_tile"] = keys <= 64
        if record["bug_residual"] is not None and not record["single_tile"]:
            record["bug_class"] = _bug_class(record["bug_residual"], record.get("floor"))
        else:
            record["bug_class"] = "no-signal" if record["route"] == "A1" else _bug_class(record["bug_residual"], record.get("floor"))
        cases.append(record)
    return cases


def run_sage(op_name: str, fast: bool = False, keep_frac: float | None = None,
             run_bench: bool = False) -> dict:
    op = registry.get(op_name)
    if not op.sage:
        raise SystemExit(f"op '{op_name}' has no sage mode; only gqa_attention does (registry)")

    harness.build_target(op.test_target)
    if run_bench:
        harness.build_target(op.bench_target)

    env = {"GQA_SAGE_ONLY": "1", "GQA_SAGE_FLOOR": "1"}
    if fast:
        env["GQA_SAGE_FAST"] = "1"
    if keep_frac is not None:
        if not (0.0 < keep_frac <= 1.0):
            raise SystemExit(f"keep_frac must be in (0, 1]; got {keep_frac}")
        env["GQA_KEEP_FRAC"] = f"{keep_frac:g}"

    result = harness.run(f"cd {harness.BUILD} && {harness.test_binary(op)}",
                         env=env, check=False, timeout=1800.0)
    output = result.output
    lines = output.strip().splitlines()
    last = lines[-1] if lines else ""
    # Gate on the exit code + the absence of a FAIL verdict line: the SAGE_FLOOR
    # diagnostics go to cerr and can land AFTER the stdout "PASS" line in the
    # merged capture, so the last line is not the verdict.
    fail_line = next((l for l in lines if l.startswith("FAIL")), None)
    gate_passed = result.ok and fail_line is None

    cases = _parse_floor_lines(output)
    classifiable = [c for c in cases
                    if c.get("bug_residual") is not None and c.get("bug_class") != "no-signal"]
    worst = max((c["bug_residual"] for c in classifiable), default=None)
    floors = [c["floor"] for c in cases if c.get("floor") is not None]
    avg_floor = sum(floors) / len(floors) if floors else None

    verdict_dict = {
        "op": op.name,
        "tier": "sage-" + ("fast" if fast else "full"),
        "git": harness.repo_head(),
        "build": harness.build_stamp(op),
        "ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "keep_frac": keep_frac if keep_frac is not None else 1.0,
        "oracle": {
            "passed": gate_passed,
            "kpi_error_to_limit": worst,  # worst bug-residual across cases (lower = better)
            "cases": [c["case"] for c in cases],
            "sage_cases": cases,
            "avg_floor": avg_floor,
            "tail": last,
        },
        "bench": None,
        "profile": None,
        "sanitizer": None,
        "perf_valid": gate_passed,
    }

    if run_bench:
        bench_env = {"NINFER_KEEP_FRAC": f"{keep_frac:g}"} if keep_frac is not None else None
        bench_result = harness.run(f"cd {harness.BUILD} && {harness.bench_binary(op)}",
                                   env=bench_env, check=False, timeout=1800.0)
        points = [
            {"context": int(m.group("ctx")), "window": int(m.group("window")),
             "fill_us": float(m.group("fill")), "median_us": float(m.group("median")),
             "p95_us": float(m.group("p95")), "gbps": float(m.group("gbps"))}
            for m in _BENCH_ROW_RE.finditer(bench_result.output)
        ]
        verdict_dict["bench"] = {
            "keep_frac": verdict_dict["keep_frac"],
            "points": points,
            "median_us": points[0]["median_us"] if points else None,
            "ok": bench_result.ok,
            "tail": bench_result.output.strip().splitlines()[-1]
            if bench_result.output.strip() else "",
        }

    return verdict_dict


def render(v: dict) -> str:
    o = v["oracle"]
    keep = v["keep_frac"]
    flag = "PASS" if o["passed"] else "FAIL"
    lines = [
        f"[kdev-sage] {v['op']} (sage-{v['tier'][5:]}) gate={flag} keep_frac={keep:g}"
        f"  git@{v['git']}",
    ]
    if not o["sage_cases"]:
        lines.append("  no SAGE_FLOOR lines parsed — check the test output tail:\n"
                     + (o.get("tail") or "").strip())
        return "\n".join(lines)

    header = (f"  {'case':<58s} {'rt':<3s} {'floor':>7s} {'vs_exact':>9s} "
              f"{'vs_sage':>8s} {'bug_resid':>10s}  class")
    lines.append(header)
    for c in o["sage_cases"]:
        def f(key):
            value = c.get(key)
            return f"{value:.4g}" if isinstance(value, (int, float)) else "-"
        lines.append(
            f"  {c['case'][:58]:<58s} {c['route']:<3s} {f('floor'):>7s} "
            f"{f('device_vs_exact'):>9s} {f('device_vs_sage'):>8s} "
            f"{f('bug_residual'):>10s}  {c.get('bug_class', '-')}")

    floors = [c["floor"] for c in o["sage_cases"] if c.get("floor") is not None]
    # Only classifiable cases count toward the worst residual: single-tile A1 cases
    # have no independent step signal (step64 == closed-form), so their residual is
    # the expected independent-rounding distance, not a bug candidate.
    classifiable = [c for c in o["sage_cases"]
                    if c.get("bug_residual") is not None and c.get("bug_class") != "no-signal"]
    worst = max((c["bug_residual"] for c in classifiable), default=None)
    if worst is not None:
        lines.append(f"  floor: avg {o.get('avg_floor'):.4f} (documented 0.053-0.059) | "
                     f"worst bug-residual (classifiable cases) {worst:.4g}")
    else:
        lines.append(f"  floor: avg {o.get('avg_floor'):.4f}")
    if worst is None:
        return "\n".join(lines)
    if keep < 1.0:
        skip = [c.get("device_vs_exact", 0) - c["floor"] for c in o["sage_cases"]
                if c.get("floor") is not None]
        avg_skip = sum(skip) / len(skip) if skip else None
        lines.append(f"  tile-skip loss (keep_frac={keep:g}): device_vs_exact - floor = "
                     f"{avg_skip:.4g}" if avg_skip is not None else
                     "  tile-skip loss: no floor values to subtract")
    if worst < _NOISE_BAND:
        lines.append("  => within the FP32 noise band (1e-03) of the kernel-faithful "
                     "emulation: no kernel-bug signal. The keep_frac=1.0 quality loss is the "
                     "FP4-P quant floor — intrinsic, not a bug.")
    else:
        worst_class = max((c["bug_class"] for c in classifiable),
                          key=lambda x: {"noise": 0, "sub-floor-dev": 1, ">=floor-dev": 2}[x])
        lines.append(
            f"  => the kernel adds {worst:.4g} rel_L2 on top of the FP4-P floor "
            f"(avg floor {o.get('avg_floor'):.4f}; class {worst_class}): a systematic "
            "deviation beyond the intrinsic quant floor. Localize it: the S3_ORC_DUMP / "
            "SAGE_DUMP env dumps of this test already per-block-dump the s3 P-quant "
            "stages, and the P2 side-band port (op_dump + diff.py) localizes the first "
            "divergent stage + index.")
    b = v.get("bench")
    if b and b.get("points"):
        p = b["points"][0]
        lines.append(f"  bench (keep_frac={b['keep_frac']:g}): ctx={p['context']} "
                     f"attn median={p['median_us']:.1f}us p95={p['p95_us']:.1f}us "
                     f"{p['gbps']:.0f} GB/s")
    return "\n".join(lines)


def persist(v: dict) -> str:
    return verdict.persist(v)