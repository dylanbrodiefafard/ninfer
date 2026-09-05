"""CLI entry point.

Layer 0–3:
    python3 -m tools.kdev recipe [--preset ... --t T --idea ...]
    python3 -m tools.kdev bound ...
    python3 -m tools.kdev mma

Projection public Ops (not kdev <op> commands):
    ./build/bench/ninfer_linear_bench --qtype ... --n N --k K --t T
    ./build/bench/ninfer_ggml_block_linear_bench --qtype ggml_... --n N --k K --t T

Op loop:
    python3 -m tools.kdev <op> [--fast|--full] [--bench] [--profile] [--san] [--json]
"""

import argparse
import json
import sys

from . import bench, bound, harness, mma, oracle, profile, recipe, registry, san, sage, verdict

_USAGE = """\
usage: python3 -m tools.kdev <command>

Kernel iteration (docs/maintainer/kernel-iteration.md):
  recipe    gate card: procedure, or fill a supported projection point
  bound     Layer-0 registered-Linear roof / GGML profile-required gate (host, no GPU)
  mma       Layer-1 MMA issue-rate probe (writes profiles/kdev/mma_issue.json)
  diff      first-divergence localization: python3 -m tools.kdev.diff <op>

Projection benches (not kdev <op> commands): ninfer_linear_bench, ninfer_ggml_block_linear_bench
Op loop (registered: {ops}):
  python3 -m tools.kdev <op> [--fast|--full] [--bench] [--profile] [--san] [--json]
"""


def _build(op, need_bench: bool) -> bool:
    """Incrementally build the test (and, if benching, the bench) targets."""
    test = harness.build_target(op.test_target)
    if not test.ok:
        print("[kdev] test build failed:\n" + test.output[-2000:])
        return False
    if need_bench:
        bench_build = harness.build_target(op.bench_target)
        if not bench_build.ok:
            print("[kdev] bench build failed:\n" + bench_build.output[-2000:])
            return False
    return True


def _run_op(argv) -> int:
    parser = argparse.ArgumentParser(prog="kdev", description=__doc__)
    parser.add_argument("op", help=f"op name (registered: {', '.join(registry.names())})")
    parser.add_argument("--fast", action="store_true", help="run only the cheapest case (default)")
    parser.add_argument("--full", action="store_true", help="run the representative matrix (overrides --fast)")
    parser.add_argument("--bench", action="store_true", help="also run the op bench (median us)")
    parser.add_argument("--profile", action="store_true", help="also run ncu over the bench point")
    parser.add_argument("--san", action="store_true", help="also run compute-sanitizer (memory axis)")
    parser.add_argument("--san-tool", default="memcheck",
                        choices=["memcheck", "racecheck", "initcheck", "synccheck", "leakcheck"])
    parser.add_argument("--json", action="store_true", help="emit the full JSON verdict (default: human table)")
    parser.add_argument("--sage", action="store_true",
                        help="sage (nvfp4s3) quality mode: FP4-P floor + bug-residual decomposition")
    parser.add_argument("--s3-dump", action="store_true",
                        help="s3 prefill op-dump: per-stage intermediate localization (score/psf/p_code/v_scale/m/l/acc)")
    parser.add_argument("--keep-frac", type=float, default=None,
                        help="rejected with --sage; exact-NVFP4 Sparge skip lives in ninfer-ppl --keep-frac")
    args = parser.parse_args(argv)

    try:
        op = registry.get(args.op)
    except KeyError as exc:
        print(str(exc))
        return 2

    if args.sage:
        if not op.sage:
            print(f"op '{op.name}' has no sage mode (only gqa_attention)")
            return 2
        if args.keep_frac is not None:
            print("--keep-frac is exact-NVFP4 Sparge; it cannot be combined with --sage")
            return 2
        if not _build(op, need_bench=args.bench):
            return 2
        v = sage.run_sage(op.name, fast=args.fast, keep_frac=args.keep_frac,
                          run_bench=args.bench)
        if args.json:
            print(json.dumps(v, indent=2))
        else:
            path = sage.persist(v)
            print(sage.render(v))
            print(f"       verdict: {path}")
        return 0 if v["oracle"]["passed"] else 1

    if args.s3_dump:
        if not op.s3_dump:
            print(f"op '{op.name}' has no s3-dump side-band (only gqa_attention)")
            return 2
        if not _build(op, need_bench=False):
            return 2
        from . import diff as diff_mod
        v = diff_mod.run_s3_diff(op.name)
        if args.json:
            print(json.dumps(v, indent=2))
        else:
            path = diff_mod.persist_s3(v)
            print(diff_mod.render_s3(v))
            print(f"       dump: {path}")
        return 0 if v.get("ok") else 1

    tier = "full" if args.full else "fast"
    if not _build(op, need_bench=args.bench or args.profile):
        return 2

    oracle_result = oracle.run_op_test(op, tier)
    v = verdict.assemble(
        op, tier, oracle_result,
        git=harness.repo_head(), build=harness.build_stamp(op),
    )
    if args.bench:
        v["bench"] = bench.run_op_bench(op, tier)
    if args.profile:
        v["profile"] = profile.run_profile(op, tier)
    if args.san:
        v["sanitizer"] = san.run_sanitize(op, tier, args.san_tool)

    if args.json:
        print(json.dumps(v, indent=2))
    else:
        path = verdict.persist(v)
        print(verdict.render(v))
        print(f"       verdict: {path}")

    return 0 if v["oracle"]["passed"] else 1


def main(argv=None) -> int:
    argv = sys.argv[1:] if argv is None else list(argv)
    if not argv or argv[0] in ("-h", "--help"):
        print(_USAGE.format(ops=", ".join(registry.names())).rstrip())
        return 0
    if argv[0] == "bound":
        return bound.main(argv[1:])
    if argv[0] == "recipe":
        return recipe.main(argv[1:])
    if argv[0] == "mma":
        return mma.main(argv[1:])
    return _run_op(argv)


if __name__ == "__main__":
    raise SystemExit(main())
