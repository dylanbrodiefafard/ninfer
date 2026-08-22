"""Oracle: run the Op's independent reference test and extract the fast-signal.

The test binary is the repo's per-Op oracle (independent FP32/FP64 reference per
AGENTS.md). With NINFER_OP_REPORT_STATS=1 it emits one OP_ERROR_STATS record per
case (error-to-limit ratio, max-error, first-violation, non-finite count) and a
final OK/FAIL line. The oracle tier only picks which cases run (fast = cheapest,
full = representative matrix) — it never edits source.
"""

import os

from . import harness


def run_op_test(op, tier: str) -> dict:
    """tier in {'fast','full'}. Returns {passed, cases, stats, exit, output}."""
    args = list(op.fast_test_args if tier == "fast" else op.full_test_args)
    argv = " ".join(args)
    cmd = (
        f"cd {harness.BUILD} && NINFER_OP_REPORT_STATS=1 "
        f"{harness.test_binary(op)} {argv}".rstrip()
    )
    result = harness.run(cmd, check=False)
    output = result.output
    passed = result.ok and "FAIL" not in output.splitlines()[-1:]
    # The binary's final line is the authoritative verdict.
    last = output.strip().splitlines()[-1] if output.strip() else ""
    passed = result.ok and last.startswith("OK")
    stats = harness.parse_op_stats(output)
    # Reduce to the KPIs that matter (drop the raw rmse/rms noise, keep ratios).
    for record in stats:
        for key in ("rel_l2_ratio", "gross_ratio", "non_finite", "max_abs", "max_reference"):
            if key in record:
                try:
                    record[key] = float(record[key])
                except ValueError:
                    pass
    return {
        "op": op.name,
        "tier": tier,
        "passed": bool(passed),
        "exit": result.code,
        "cases": [r.get("case") for r in stats],
        "stats": stats,
        "tail": last,
    }


def oracle_kpi(stats: list) -> float | None:
    """Worst (largest) error-to-limit ratio across cases — the tracked KPI.
    Lower is better; >1.0 means at least one case exceeds its tolerance."""
    ratios = []
    for record in stats:
        for key in ("rel_l2_ratio", "gross_ratio"):
            value = record.get(key)
            if isinstance(value, (int, float)):
                ratios.append(float(value))
    return max(ratios) if ratios else None