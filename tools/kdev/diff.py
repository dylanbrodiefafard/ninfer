"""diff: localize the first divergent intermediate stage for a dumped Op.

Runs the Op's test with ``--dump`` (the op_dump side-band), reads the kernel and
FP64 reference intermediates, and reports the FIRST stage + row index where they
disagree beyond tolerance — in one run, no source edits. This is the fast-signal
for "which computation stage went wrong" (e.g. the reduction vs the rsqrt),
mirroring the NoVf16 wrong-V-scale class of bug (project memory #117).

For Ops with the s3 prefill side-band (registry ``s3_dump``), the test's
``--s3-dump`` mode pre-computes the per-stage first divergence (score / psf /
p_code / v_scale / m / l / acc) against a tile-exact FP64 reference and writes a
compact JSON; diff.py renders it. A SHARP first divergence in one stage = bug;
only 1-step e2m1/e4m3 code flips = the intrinsic FP4-P quant floor.

Usage:
    python3 -m tools.kdev.diff <op> [--rel-tol 1e-4] [--abs-tol 1e-5]
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

from . import harness, registry


def run_diff(op_name: str, rel_tol: float = 1e-4, abs_tol: float = 1e-5) -> dict:
    op = registry.get(op_name)
    harness.build_target(op.test_target)

    rel_dir = os.path.join("profiles", "kdev", op.name)
    os.makedirs(os.path.join(os.getcwd(), rel_dir), exist_ok=True)
    host_path = os.path.join(os.getcwd(), rel_dir, "last_dump.json")
    container_path = os.path.join("/src", rel_dir, "last_dump.json")

    result = harness.run(
        f"cd {harness.BUILD} && {harness.test_binary(op)} --dump {container_path}",
        check=False,
    )
    if not result.ok or not os.path.exists(host_path):
        return {"op": op.name, "ok": False, "error": result.output[-500:], "case": None,
                "stages": [], "first": None}

    with open(host_path) as handle:
        dump = json.load(handle)

    stages = []
    for stage in dump.get("stages", []):
        kernel = stage.get("kernel", [])
        ref = stage.get("ref", [])
        first = None
        max_rel = 0.0
        for i, (kv, rv) in enumerate(zip(kernel, ref)):
            denom = max(abs(rv), 1e-30)
            rel = abs(kv - rv) / denom
            max_rel = max(max_rel, rel)
            if abs(kv - rv) > abs_tol and rel > rel_tol and first is None:
                first = {"stage": stage["name"], "index": i, "kernel": kv, "ref": rv, "rel": rel}
        stages.append({
            "name": stage["name"],
            "rows": len(kernel),
            "clean": first is None,
            "max_rel": max_rel,
            "first": first,
        })

    # The pipeline's earliest divergent stage is the root-cause candidate.
    first_stage = next((s for s in stages if s["first"] is not None), None)
    return {
        "op": op.name,
        "ok": True,
        "case": dump.get("case"),
        "rows": dump.get("rows"),
        "rel_tol": rel_tol,
        "abs_tol": abs_tol,
        "stages": stages,
        "first": first_stage["first"] if first_stage else None,
    }


def render(report: dict) -> str:
    if not report.get("ok"):
        return f"[kdev-diff] {report['op']}: could not produce dump:\n{report.get('error','')}"
    lines = [f"[kdev-diff] {report['op']} case='{report['case']}' rows={report['rows']}"
             f" (rel_tol={report['rel_tol']:.0e} abs_tol={report['abs_tol']:.0e})"]
    for stage in report["stages"]:
        if stage["clean"]:
            lines.append(f"  {stage['name']:<6} clean (max rel {stage['max_rel']:.2e})")
        else:
            f = stage["first"]
            lines.append(f"  {stage['name']:<6} DIVERGES at index {f['index']}: "
                         f"kernel={f['kernel']:.6g} ref={f['ref']:.6g} (rel {f['rel']:.3f})")
    if report["first"]:
        f = report["first"]
        lines.append(f"  => first divergence: stage={f['stage']} index={f['index']} "
                     f"(kernel {f['kernel']:.6g} vs ref {f['ref']:.6g})")
    else:
        lines.append("  => no divergence (all stages within tolerance)")
    return "\n".join(lines)


def run_s3_diff(op_name: str) -> dict:
    """s3 prefill side-band: run the test's --s3-dump and render its
    pre-computed per-stage first-divergence JSON (no full-array re-scan)."""
    op = registry.get(op_name)
    if not op.s3_dump:
        raise ValueError(f"op '{op_name}' has no s3_dump side-band")
    harness.build_target(op.test_target)

    rel_dir = os.path.join("profiles", "kdev", op.name)
    os.makedirs(os.path.join(os.getcwd(), rel_dir), exist_ok=True)
    host_path = os.path.join(os.getcwd(), rel_dir, "last_s3_dump.json")
    container_path = os.path.join("/src", rel_dir, "last_s3_dump.json")

    result = harness.run(
        f"cd {harness.BUILD} && {harness.test_binary(op)} --s3-dump {container_path}",
        check=False,
    )
    if not result.ok or not os.path.exists(host_path):
        return {"op": op.name, "ok": False, "error": result.output[-500:], "case": None,
                "stages": [], "first": None, "verdict": None}

    with open(host_path) as handle:
        dump = json.load(handle)
    stages = dump.get("stages", [])
    first_stage = next((s for s in stages if not s.get("clean", True) and s.get("first")), None)
    return {
        "op": op.name,
        "ok": True,
        "kind": dump.get("kind"),
        "case": dump.get("case"),
        "heads": dump.get("heads"),
        "rows": dump.get("rows"),
        "stages": stages,
        "first": first_stage["first"] if first_stage else None,
        "verdict": dump.get("verdict"),
    }


def render_s3(report: dict) -> str:
    if not report.get("ok"):
        return f"[kdev-s3diff] {report['op']}: could not produce s3 dump:\n{report.get('error','')}"
    lines = [f"[kdev-s3diff] {report['op']} case='{report['case']}' "
             f"(heads={report.get('heads')} rows={report.get('rows')})"]
    for s in report["stages"]:
        name = s["name"]
        if name in ("p_code", "psf"):
            stat = (f"flips={s.get('one_step_flips', 0)} "
                    f"two_plus={s.get('two_plus', 0)} max_diff={s.get('max_diff', 0)}")
        elif name == "v_scale":
            stat = f"diffs={s.get('diffs', 0)}"
        elif name == "acc":
            stat = (f"kv-codes={s.get('max_rel_kernel_codes', 0):.2e} "
                    f"ref-codes={s.get('max_rel_ref_codes', 0):.2e}")
        else:
            stat = f"max_rel={s.get('max_rel', 0):.2e}"
        state = "clean" if s["clean"] else "DIVERGES"
        lines.append(f"  {name:<7} {state:<9} {stat}")
        if not s["clean"] and s.get("first"):
            lines.append(f"          first: {s['first']}")
    if report["first"]:
        lines.append(f"  => first divergence: {report['first']}")
    lines.append(f"  => verdict: {report.get('verdict')}")
    return "\n".join(lines)


def persist_s3(report: dict) -> str:
    """Write the s3 dump JSON + one trend row under profiles/kdev/<op>/."""
    op = report["op"]
    folder = os.path.join(os.getcwd(), "profiles", "kdev", op)
    os.makedirs(os.path.join(folder, "s3_dumps"), exist_ok=True)
    ts = time.strftime("%Y%m%d-%H%M%S")
    json_path = os.path.join(folder, "s3_dumps", f"{ts}_s3dump.json")
    with open(json_path, "w") as handle:
        json.dump(report, handle, indent=2)
        handle.write("\n")
    row = {
        "op": op,
        "kind": "s3_dump",
        "ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "case": report.get("case"),
        "verdict": report.get("verdict"),
        "clean": report.get("first") is None,
    }
    trend_path = os.path.join(folder, "trend.jsonl")
    with open(trend_path, "a") as handle:
        handle.write(json.dumps(row) + "\n")
    return json_path


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="kdev.diff", description=__doc__)
    parser.add_argument("op")
    parser.add_argument("--rel-tol", type=float, default=1e-4)
    parser.add_argument("--abs-tol", type=float, default=1e-5)
    args = parser.parse_args(argv)

    op = registry.get(args.op)
    if op.s3_dump:
        report = run_s3_diff(args.op)
        print(render_s3(report))
    else:
        report = run_diff(args.op, args.rel_tol, args.abs_tol)
        print(render(report))
    return 0 if report.get("ok") else 2


if __name__ == "__main__":
    raise SystemExit(main())