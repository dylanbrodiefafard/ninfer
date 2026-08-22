"""Bench: run the Op's CUDA-event median bench and pull out the headline numbers.

The bench binary reports, per point, the median/min/p95 in us plus an effective
bandwidth and its % of the 1792 GB/s read roofline. We capture the median (the
tracked perf KPI) and the roofline % for the human table. The roofline % can exceed
100% by construction of the bench's logical-bytes model — it is a convenience
readout, not a gate (see bench/ninfer_bench_common.h).
"""

import re

from . import harness

_LINE_RE = re.compile(
    r"(?P<label>T=\d+[^\s]*\s*\[[^\]]*\])\s*median=\s*(?P<median>[\d.]+)\s*us"
    r".*?(?P<gbps>[\d.]+)\s*GB/s\s*\((?P<roofline>[\d.]+)%",
)


def _parse(text: str) -> list:
    rows = []
    for match in _LINE_RE.finditer(text):
        rows.append({
            "point": match.group("label").strip(),
            "median_us": float(match.group("median")),
            "gbps": float(match.group("gbps")),
            "roofline_pct": float(match.group("roofline")),
        })
    return rows


def run_op_bench(op, tier: str) -> dict:
    """tier in {'fast','full'}. Returns {points, median_us, output}."""
    args = list(op.fast_bench_args if tier == "fast" else op.full_bench_args)
    cmd = f"cd {harness.BUILD} && {harness.bench_binary(op)} " + " ".join(args)
    result = harness.run(cmd, check=False)
    rows = _parse(result.output)
    median = rows[0]["median_us"] if rows else None
    return {
        "op": op.name,
        "tier": tier,
        "points": rows,
        "median_us": median,
        "ok": result.ok,
        "tail": result.output.strip().splitlines()[-1] if result.output.strip() else "",
    }