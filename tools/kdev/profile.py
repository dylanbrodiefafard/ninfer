"""Profile: optional ncu pass, bracketed to a single L2-cold kernel launch.

Off by default so the fast tier stays fast. ``--profile`` runs ncu over one launch
of the Op's bench point (ncu flushes L2 per launch, matching the bench's cold-L2
method) and captures a small SoL metric set. Raw CSV is saved under
profiles/kdev/<op>/ and the headline metrics are lifted into the verdict.
"""

import csv
import io
import os
import time

from . import harness

# A small set of speed-of-light metrics: wall time, SM busy %, DRAM busy %,
# issued-instruction efficiency. Enough to say "DRAM-bound vs mma-bound".
_NCU_METRICS = [
    "gpu__time_duration.sum",
    "sm__throughput.avg.pct_of_peak_sustained_elapsed",
    "dram__throughput.avg.pct_of_peak_sustained_elapsed",
    "sm__inst_executed.avg.pct_of_peak_sustained_active",
]


def _report_dir(op) -> str:
    path = os.path.join(os.getcwd(), "profiles", "kdev", op.name)
    os.makedirs(path, exist_ok=True)
    return path


def run_profile(op, tier: str = "fast") -> dict:
    ts = time.strftime("%Y%m%d-%H%M%S")
    out = os.path.join(_report_dir(op), f"ncu_{ts}.txt")
    args = list(op.fast_bench_args if tier == "fast" else op.full_bench_args)
    cmd = (
        f"cd {harness.BUILD} && ncu --target-processes all --launch-count 1 "
        f"--metrics {','.join(_NCU_METRICS)} --csv "
        f"{harness.bench_binary(op)} " + " ".join(args)
    )
    result = harness.run(cmd, check=False)
    with open(out, "w") as handle:
        handle.write(result.output)
    highlights = _extract(result.output)
    return {
        "op": op.name,
        "tier": tier,
        "ok": result.ok and bool(highlights),
        "report": out,
        "highlights": highlights,
        "tail": result.output.strip().splitlines()[-1] if result.output.strip() else "",
    }


def _extract(text: str) -> dict:
    """Lift the SoL metric values out of the ncu CSV (best-effort, low noise)."""
    lines = text.splitlines()
    start = next((i for i, line in enumerate(lines) if line.startswith('"ID"')), None)
    if start is None:
        return {}
    reader = csv.reader(io.StringIO("\n".join(lines[start:])))
    header = next(reader, None)
    if not header:
        return {}
    name_i = header.index("Metric Name")
    unit_i = header.index("Metric Unit")
    value_i = header.index("Metric Value")
    out = {}
    for row in reader:
        if len(row) > max(name_i, unit_i, value_i):
            out[row[name_i]] = {"value": row[value_i], "unit": row[unit_i]}
    return out