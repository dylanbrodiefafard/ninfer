"""Verdict: one JSON (single source of truth) + JSONL trend + a human table.

Every kdev run emits exactly one verdict dict. The LLM reads the JSON schema; the
engineer reads the one-line table. Perf numbers are only reported as valid when
the oracle passed — a "fast-but-wrong" result is flagged, never silently emitted.

Artifacts (all under profiles/kdev/<op>/, gitignore-able):
  * <ts>.json      — the full verdict for this run
  * trend.jsonl     — one compact row per run (op, tier, git, build, kpi, us, ts)
"""

import json
import os
import time

from . import oracle as oracle_mod

_PROFILE_DIR = os.path.join(os.getcwd(), "profiles", "kdev")


def assemble(op, tier, oracle, bench=None, profile=None, san=None,
             git=None, build=None) -> dict:
    kpi = oracle_mod.oracle_kpi(oracle.get("stats", [])) if oracle else None
    perf_valid = bool(oracle and oracle.get("passed"))
    verdict = {
        "op": op.name,
        "tier": tier,
        "git": git or "unknown",
        "build": build or "unknown",
        "ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "oracle": {
            "passed": bool(oracle and oracle.get("passed")),
            "kpi_error_to_limit": kpi,
            "cases": (oracle or {}).get("cases", []),
            "stats": (oracle or {}).get("stats", []),
            "tail": (oracle or {}).get("tail", ""),
        },
        "bench": bench,
        "profile": profile,
        "sanitizer": san,
        "perf_valid": perf_valid,  # bench numbers are only meaningful if the oracle passed
    }
    return verdict


def persist(verdict: dict) -> str:
    op = verdict["op"]
    folder = os.path.join(_PROFILE_DIR, op)
    os.makedirs(folder, exist_ok=True)
    ts = time.strftime("%Y%m%d-%H%M%S")
    json_path = os.path.join(folder, f"{ts}.json")
    with open(json_path, "w") as handle:
        json.dump(verdict, handle, indent=2)
        handle.write("\n")

    row = {
        "op": op,
        "tier": verdict["tier"],
        "git": verdict["git"],
        "build": verdict["build"],
        "ts": verdict["ts"],
        "oracle_passed": verdict["oracle"]["passed"],
        "kpi": verdict["oracle"]["kpi_error_to_limit"],
        "median_us": (verdict.get("bench") or {}).get("median_us"),
        "perf_valid": verdict["perf_valid"],
    }
    trend_path = os.path.join(folder, "trend.jsonl")
    with open(trend_path, "a") as handle:
        handle.write(json.dumps(row) + "\n")
    return json_path


def render(verdict: dict) -> str:
    """A single low-noise table line + a few detail lines. No floods."""
    o = verdict["oracle"]
    b = verdict.get("bench") or {}
    flag = "PASS" if o["passed"] else "FAIL"
    perf = ""
    if b.get("median_us") is not None:
        p0 = (b.get("points") or [{}])[0]
        gbps = p0.get("gbps")
        roof = p0.get("roofline_pct")
        perf = f"  med={b['median_us']:.2f}us"
        if gbps is not None:
            perf += f"  gbps={gbps:.0f} ({roof:.0f}% roof)"
        if not verdict["perf_valid"]:
            perf += "  [INVALID: oracle failed]"
    kpi = o.get("kpi_error_to_limit")
    kpi_s = f"{kpi:.4f}" if isinstance(kpi, (int, float)) else "-"
    cases = ", ".join(o.get("cases") or []) or "-"
    lines = [
        f"[kdev] {verdict['op']} ({verdict['tier']}) oracle={flag} kpi={kpi_s}"
        f"{perf}  git@{verdict['git']}",
        f"       cases: {cases}",
    ]
    san = verdict.get("sanitizer")
    if san:
        lines.append(f"       san({san['tool']}): {'clean' if san['clean'] else str(san['findings']) + ' finding(s)'}")
    prof = verdict.get("profile")
    if prof:
        lines.append(f"       ncu: {prof.get('report','')}")
    return "\n".join(lines)