"""Print the Layer 0–3 kernel-iteration recipe, filled for a supported projection point.

The contract is docs/maintainer/kernel-iteration.md. Without geometry this command
prints the procedure, idea catalog, and exact public-Op commands. Registered Linear
points use a roofline bound; GGML block-row points use exact bytes and a profile-required gate.

    python3 -m tools.kdev recipe
    python3 -m tools.kdev recipe --preset attn_in --t 1 --idea occupancy
    python3 -m tools.kdev recipe --n 14336 --k 5120 --t 1024 --qtype nvfp4 \\
        --policy a4 --measured-us 152.6 --idea tile_shape
"""

from __future__ import annotations

import argparse
import json
import os

from . import bound, registry

_LINEAR_BENCH = "./build/bench/ninfer_linear_bench"
_GGML_LINEAR_BENCH = "./build/bench/ninfer_ggml_block_linear_bench"
_NCU_QUESTIONS = (
    "DRAM vs 1674.5 GB/s",
    "tensor-pipe busy",
    "spills",
    "DRAM bytes vs model_bytes",
)


def recipe_cmd(card: dict, *, measured_placeholder: bool = False) -> str:
    p = card["problem"]
    if card.get("preset"):
        parts = [f"python3 -m tools.kdev recipe --preset {card['preset']} --t {p['t']}"]
        if args_qtype_overridden(card):
            parts.append(f"--qtype {p['qtype']}")
        if p["policy"] != "a16":
            parts.append(f"--policy {p['policy']}")
    else:
        parts = [
            "python3 -m tools.kdev recipe",
            f"--n {p['n']} --k {p['k']} --t {p['t']}",
            f"--qtype {p['qtype']} --policy {p['policy']}",
        ]
    idea = card.get("idea")
    if idea:
        parts.append(f"--idea {idea['name']}")
    if measured_placeholder:
        parts.append("--measured-us <median>")
    elif card.get("measured_us") is not None:
        parts.append(f"--measured-us {card['measured_us']}")
    return " ".join(parts)


def args_qtype_overridden(card: dict) -> bool:
    preset = card.get("preset")
    if not preset:
        return False
    return card["problem"]["qtype"] != bound.PRESETS[preset]["qtype"]


def linear_bench_cmd(n: int, k: int, t: int, qtype: str, policy: str, *, profile: bool = False) -> str:
    if qtype.startswith("ggml_"):
        cmd = (
            f"{_GGML_LINEAR_BENCH} --qtype {qtype} "
            f"--n {n} --k {k} --t {t}"
        )
        if profile:
            return f"ncu --profile-from-start off --set full {cmd} --profile --repetitions 1"
        return cmd
    cmd = (
        f"{_LINEAR_BENCH} --qtype {qtype} --policy {policy} "
        f"--n {n} --k {k} --t {t}"
    )
    if profile:
        return f"ncu --profile-from-start off --set full {cmd} --profile"
    return cmd


def status_of(card: dict) -> str:
    """Agent-facing gate: stop / measure / go / fill."""
    idea = card.get("idea")
    if not card["sm120"]["legal"]:
        return "stop"
    if idea and idea["verdict"] == "refuse":
        return "stop"
    if card.get("measured_us") is None:
        return "measure"
    if idea is None:
        return "fill"
    if idea["verdict"] == "measure":
        return "measure"
    return "go"


def _fmt_bytes(n: int) -> str:
    if n >= 1024 * 1024:
        return f"{n} ({n / (1024 * 1024):.2f} MiB)"
    if n >= 1024:
        return f"{n} ({n / 1024:.1f} KiB)"
    return str(n)


def _fmt_flops(n: int) -> str:
    if n >= 1e12:
        return f"{n} ({n / 1e12:.2f} TFLOP)"
    if n >= 1e9:
        return f"{n} ({n / 1e9:.2f} GFLOP)"
    return f"{n} ({n / 1e6:.2f} MFLOP)"


def _matching_roof(card: dict) -> str:
    if card["bound"] == "profile-required":
        return "the one-read represented-byte throughput roof (1674.5 GB/s)"
    if card["bound"] == "DRAM":
        return "1674.5 GB/s sustained read"
    t_issue = card.get("t_issue_us")
    t_comp = card["t_comp_us"]
    if t_issue is not None and t_issue >= t_comp:
        return "MMA issue roof (t_issue from profiles/kdev/mma_issue.json)"
    return "1676 TFLOP/s dense FP4"


def _policy_hint(problem: dict) -> str | None:
    if problem["qtype"] == "nvfp4" and problem["t"] >= 256 and problem["policy"] == "a16":
        return (
            "NVFP4 prefill speed point is typically --policy a4 "
            "(AllowA4; the production resolver picks the qualified route)."
        )
    return None


def _layer2(card: dict) -> dict:
    p = card["problem"]
    bench = linear_bench_cmd(p["n"], p["k"], p["t"], p["qtype"], p["policy"])
    ncu = linear_bench_cmd(p["n"], p["k"], p["t"], p["qtype"], p["policy"], profile=True)
    return {
        "op": "ggml_block_linear" if p["qtype"].startswith("ggml_") else "linear",
        "bench": bench,
        "ncu": ncu,
        "ncu_questions": list(_NCU_QUESTIONS),
        "note": (
            "Linear is not a kdev <op>. Measure the named public Op through the selected "
            "registered-Linear or GGML-block-row benchmark. "
            "Registered kdev ops: " + ", ".join(registry.names()) + "."
        ),
    }


def _classified(card: dict) -> list[dict]:
    p = card["problem"]
    rows = []
    for name in bound.IDEAS:
        rows.append(bound.classify_idea(name, card["bound"], p["t"], p["phase"]))
    return rows


def _group_lines(rows: list[dict]) -> list[str]:
    illegal, allow, measure, refuse = [], [], [], []
    for row in rows:
        name, verdict, reason = row["name"], row["verdict"], row["reason"]
        if name in bound.ILLEGAL_IDEAS:
            illegal.append(name)
        elif verdict == "allow":
            allow.append((name, reason))
        elif verdict == "refuse":
            refuse.append(name)
        else:
            measure.append((name, reason))
    lines = []
    if allow:
        lines.append("allow")
        for name, reason in allow:
            lines.append(f"  {name:<22} {reason}")
    if measure:
        lines.append("measure")
        for name, reason in measure:
            lines.append(f"  {name:<22} {reason}")
    if refuse:
        lines.append("refuse (this bound)    " + ", ".join(refuse))
    if illegal:
        lines.append("refuse (illegal sm120) " + ", ".join(illegal))
    return lines


def _status_banner(status: str, card: dict) -> str:
    idea = card.get("idea")
    if status == "stop":
        if idea:
            return f"STOP  {card['next']}"
        return "STOP  sm120=ILLEGAL. " + "; ".join(card["sm120"]["reasons"])
    if status == "measure" and card.get("measured_us") is None:
        return (
            "MEASURE  No public-Op median yet. Do not write CUDA. "
            f"After the Layer 2 bench: {recipe_cmd(card, measured_placeholder=True)}"
        )
    if status == "measure":
        return f"MEASURE  {card['next']}"
    if status == "fill":
        return "FILL  Name --idea before writing CUDA. unknown means measure, do not implement."
    return f"GO  {card['next']}"


def render_filled(card: dict) -> str:
    p = card["problem"]
    idea = card.get("idea")
    layer2 = _layer2(card)
    status = status_of(card)
    label = card.get("label") or ""
    is_ggml = p["qtype"].startswith("ggml_")
    op_name = "ggml_block_linear" if is_ggml else "linear"
    t_issue = (
        f"{card['t_issue_us']:.2f}us" if card["t_issue_us"] is not None else "n/a"
    )
    hint = _policy_hint(p)
    lines = [
        f"NInfer kernel gate card — {op_name}  RTX 5090 / sm_120a",
        "Authority: docs/maintainer/kernel-iteration.md",
        "",
        f"1. Op     {op_name}  n={p['n']} k={p['k']} T={p['t']} qtype={p['qtype']} "
        f"policy={p['policy']}  phase={p['phase']}"
        + (f"  ({label})" if label else ""),
        f"2. Floor  bound={card['bound']}  model_bytes={_fmt_bytes(card['model_bytes'])}",
        f"          weight={_fmt_bytes(card['weight_bytes'])}  "
        f"act={_fmt_bytes(card['activation_bytes'])}",
    ]
    if is_ggml:
        lines.extend([
            f"          useful_flops={_fmt_flops(card['useful_flops'])}  "
            f"one-read AI={card['ai_flop_per_byte']:.1f} FLOP/B",
            f"          one_read_mem={card['t_mem_us']:.2f}us  codec_compute=unmodeled  "
            f"lower_bound={card['floor_us']:.2f}us",
        ])
    else:
        lines.extend([
            f"          useful_flops={_fmt_flops(card['useful_flops'])}  "
            f"AI={card['ai_flop_per_byte']:.1f} FLOP/B  "
            f"ridge={card['ridge_flop_per_byte']:.0f} FLOP/B"
            + (f"  ridge_T≈{card['ridge_t']:.0f}" if card.get("ridge_t") else ""),
            f"          t_mem={card['t_mem_us']:.2f}us  t_comp={card['t_comp_us']:.2f}us  "
            f"t_issue={t_issue}  floor={card['floor_us']:.2f}us",
        ])
    if hint:
        lines.append(f"          note: {hint}")
    if card["measured_us"] is None:
        if status == "stop":
            lines.append("3. Now    measured=unknown")
            lines.append(f"          {layer2['bench']}")
        else:
            lines.append("3. Now    measured=unknown  — do not implement on an unknown baseline.")
            lines.append(f"          {layer2['bench']}")
            lines.append(f"          then: {recipe_cmd(card, measured_placeholder=True)}")
    else:
        lines.append(
            f"3. Now    measured={card['measured_us']:.2f}us  "
            f"floor/measured={card['floor_pct_of_measured']:.1f}% of {_matching_roof(card)}"
        )
    if idea:
        lines.append(f"4. Idea   {idea['name']}  verdict={idea['verdict'].upper()}")
        lines.append(f"          {idea['reason']}")
    else:
        lines.append("4. Idea   unknown  — pass --idea <class>. If it does not attack this bound, stop.")
    if card["sm120"]["legal"] and is_ggml:
        lines.append(
            f"5. SM120  legal  current scalar GGML codec family  "
            f"smem<={bound.SMEM_LIMIT_BYTES} B  cluster=1  no TMEM/tcgen05"
        )
    elif card["sm120"]["legal"]:
        atom = card["mma_atom"]
        lines.append(
            f"5. SM120  legal  warp mma.sync m{atom['m']}n{atom['n']}k{atom['k']}  "
            f"smem<={bound.SMEM_LIMIT_BYTES} B  cluster=1  no TMEM/tcgen05"
        )
    else:
        lines.append("5. SM120  ILLEGAL  " + "; ".join(card["sm120"]["reasons"]))
    if card["bound"] == "DRAM":
        lines.append(
            "6. Family DRAM-bound. Do not fork a compute family. "
            "Attack extra bytes or raise T per weight pass."
        )
    elif is_ggml:
        lines.append(
            "6. Family GGML codec compute is unmodeled. Profile physical bytes/throughput, "
            "instruction pipes, occupancy, and spills before choosing a compute-side idea."
        )
    else:
        lines.append(
            "6. Family Stay in the current SM120 NVFP4 family. "
            "Search tile/TMA/pipeline; do not invent a new MMA ISA."
        )
    lines.append(f"7. Layer2 {layer2['bench']}")
    lines.append("          Oracle first, then this public Op at the exact point. Fast-but-wrong is invalid.")
    lines.append("          NCU is one named question, not an open report: " + "; ".join(_NCU_QUESTIONS) + ".")
    lines.append(f"          {layer2['ncu']}")
    lines.append("          No Engine / ninfer_bench / serve A/B until this Op wins.")
    lines.append("8. Lose   Delete the candidate. Do not leave a second path.")
    lines.append("")
    lines.extend(_group_lines(_classified(card)))
    if card["t_issue_us"] is None and card["bound"] != "DRAM" and not is_ggml:
        lines.append("")
        lines.append(
            "t_issue=n/a  run `python3 -m tools.kdev mma` once; bound/recipe pick up "
            "profiles/kdev/mma_issue.json. 1676 TFLOP/s is the datasheet fallback."
        )
    lines.append("")
    lines.append(_status_banner(status, card))
    return "\n".join(lines)


def _guide_payload() -> dict:
    presets = {
        name: {"n": spec["n"], "k": spec["k"], "qtype": spec["qtype"], "label": spec["label"]}
        for name, spec in bound.PRESETS.items()
    }
    ideas = []
    for name in bound.IDEAS:
        meta = dict(bound.IDEA_CATALOG[name])
        meta["name"] = name
        ideas.append(meta)
    return {
        "mode": "guide",
        "hardware": {
            "gpu": "RTX 5090",
            "sm": "sm_120a",
            "dram_spec_gb_s": bound.DRAM_SPEC_GB_S,
            "sustained_read_gb_s": bound.SUSTAINED_READ_GB_S,
            "dense_fp4_tflop_s": bound.DENSE_FP4_TFLOP_S,
            "smem_limit_bytes": bound.SMEM_LIMIT_BYTES,
        },
        "scope": {
            "bound": (
                "Registered Linear uses its GEMM roofline; GGML block-row uses exact "
                "representation bytes with codec compute profile-required. Not GQA/l2norm."
            ),
            "legal": "warp mma.sync (mxf4nvf4), single-CTA TMA, acc in registers, smem <= 99 KiB, cluster=1",
            "illegal": sorted(bound.ILLEGAL_IDEAS),
        },
        "presets": presets,
        "ideas": ideas,
        "linear_bench": (
            f"{_LINEAR_BENCH} --qtype QTYPE --policy a16|a4 --n N --k K --t T"
        ),
        "ggml_block_linear_bench": (
            f"{_GGML_LINEAR_BENCH} --qtype ggml_QTYPE --n N --k K --t T"
        ),
        "kdev_ops": registry.names(),
        "fill": [
            "python3 -m tools.kdev recipe --preset attn_in --t 1 --idea occupancy",
            "python3 -m tools.kdev recipe --n 14336 --k 5120 --t 1024 --qtype nvfp4 "
            "--policy a4 --measured-us 152.6 --idea tile_shape",
        ],
    }


def render_guide() -> str:
    ops = ", ".join(registry.names())
    mma_present = os.path.isfile(os.path.join(os.getcwd(), "profiles", "kdev", "mma_issue.json"))
    mma_line = (
        "profiles/kdev/mma_issue.json present — bound/recipe will set t_issue."
        if mma_present
        else "profiles/kdev/mma_issue.json absent — t_issue=n/a; 1676 TFLOP/s datasheet fallback."
    )
    return f"""\
NInfer kernel iteration — RTX 5090 / sm_120a
Authority: docs/maintainer/kernel-iteration.md
Registered Linear cards use its GEMM roofline (ninfer::ops::linear). GGML block-row
cards compute exact represented bytes but require a profile for the codec compute bound.
Neither applies to GQA or l2norm; those use SM120 legality + the public-Op loop.

Fill this card for a supported projection point (host, no GPU):
  python3 -m tools.kdev recipe --preset attn_in --t 1 --idea occupancy
  python3 -m tools.kdev recipe --n 14336 --k 5120 --t 1024 --qtype nvfp4 --policy a4 \\
      --measured-us 152.6 --idea tile_shape
  python3 -m tools.kdev recipe --n 10240 --k 2560 --t 512 --qtype ggml_q5_k \\
      --measured-us 5279.8 --idea aggregate_T
  python3 -m tools.kdev recipe --list-ideas
  python3 -m tools.kdev recipe --list-presets

Layer 0  bound (host). Predicted time = max(t_mem, t_comp[, t_issue]).
  DRAM (typical decode T≲16; ridge_T≈380 on 27B NVFP4 attn-in): only ideas that cut
  extra bytes or raise useful work per weight pass (weight_replay, aggregate_T).
  tensor-core (typical prefill T=1024): tile / TMA / pipeline / epilogue inside the
  existing SM120 NVFP4 family.
  profile-required (GGML block rows): exact one-read bytes only; profile physical
  traffic and instruction pipes. At T>16, aggregate_T/weight_replay may proceed to an
  A/B because they provably reduce packed-weight passes; T<=16 already uses one pass.
  STOP if verdict=refuse or sm120=ILLEGAL. Do not write CUDA.

Layer 1  legality + issue roof (once per machine, or after ops/common/mma.cuh changes)
  python3 -m tools.kdev mma
  {mma_line}
  Legal: warp mma.sync (kind::mxf4nvf4), single-CTA TMA, acc in registers, smem <= 99 KiB.
  Illegal: tcgen05, TMEM, 2-SM MMA, cluster-multicast TMA, SM100 128×128 TMEM tiles.

Layer 2  public Op at the exact point. Oracle first. Fast-but-wrong is invalid.
  Linear (the bound's subject; not a kdev <op>):
    {_LINEAR_BENCH} --qtype nvfp4 --policy a16 --n 14336 --k 5120 --t 1
    {_LINEAR_BENCH} --qtype nvfp4 --policy a4  --n 14336 --k 5120 --t 1024
  GGML block-row projection (codec bound requires profiling):
    {_GGML_LINEAR_BENCH} --qtype ggml_q5_k --n 10240 --k 2560 --t 512
  Registered kdev ops ({ops}):
    python3 -m tools.kdev <op> --fast --bench
    python3 -m tools.kdev <op> --fast --bench --profile
  NCU is one named question, not an open report: {'; '.join(_NCU_QUESTIONS)}.
  Temporary candidate sweeps may call private launchers; keep one winner and delete the rest.

Layer 3  qualify the reachable production route against the independent oracle, then
  the public Op bench at the exact point. Engine / ninfer_bench / serve A/B only when
  the deliverable is end-to-end tok/s. If the candidate loses, delete it.

Idea classes (name one; the classifier gates it for this bound):
{bound.format_idea_catalog()}

27B NVFP4 Linear presets (--preset):
{chr(10).join(f'  {name:10} n={spec["n"]:<6} k={spec["k"]:<6} {spec["qtype"]}  {spec["label"]}' for name, spec in bound.PRESETS.items())}

Gate: any unknown means measure, do not implement.
  1 Op + exact (N,K,T,qtype,policy) + phase
  2 Linear: model bytes + compute/issue floors; GGML: exact bytes + one-read floor,
    codec compute unmodeled
  3 measured µs and registered-Linear roof %, or GGML one-read throughput efficiency
  4 idea class — must attack this bound
  5 SM120 legality
  6 parameters inside the current family, or why a new family is required
  7 public Op microbench at that exact point
  8 delete losers
"""


def filled_payload(card: dict) -> dict:
    p = card["problem"]
    return {
        "mode": "card",
        "status": status_of(card),
        "gate": render_filled(card),
        "bound": card,
        "layer2": _layer2(card),
        "policy_hint": _policy_hint(p),
        "ideas": _classified(card),
    }


def _self_test() -> int:
    failures = []

    def check(name, cond, detail=""):
        if not cond:
            failures.append(f"{name}: {detail}")

    guide = render_guide()
    check("guide-linear-bench", _LINEAR_BENCH in guide)
    check("guide-public-op", "ninfer::ops::linear" in guide)
    check("guide-weight-replay", "weight_replay" in guide)
    check("guide-illegal", "tcgen05" in guide and "ILLEGAL" in guide)
    check("guide-not-kdev-linear", "not a kdev <op>" in guide)

    refuse = bound.analyze(14336, 5120, 1, "nvfp4", idea="occupancy", label="attn in")
    text = render_filled(refuse)
    check("t1-stop", text.splitlines()[-1].startswith("STOP"), text.splitlines()[-1])
    check("t1-dram", "bound=DRAM" in text)
    check("t1-refuse", "verdict=REFUSE" in text)
    check("t1-bench", "--n 14336 --k 5120 --t 1" in text)
    check("t1-status", status_of(refuse) == "stop", status_of(refuse))

    allow = bound.analyze(
        14336, 5120, 1024, "nvfp4", policy="a4", idea="tile_shape", measured_us=152.6,
    )
    text_go = render_filled(allow)
    check("t1024-tc", "bound=tensor-core" in text_go)
    check("t1024-allow", "verdict=ALLOW" in text_go)
    check("t1024-go", status_of(allow) == "go", status_of(allow))
    check("t1024-policy", "--policy a4" in text_go)
    check("t1024-pct", "floor/measured=" in text_go)

    baseline = bound.analyze(14336, 5120, 1024, "nvfp4", idea="tile_shape")
    check("no-us-measure", status_of(baseline) == "measure", status_of(baseline))
    check("reinvoke", "--measured-us <median>" in render_filled(baseline))
    refuse["preset"] = "attn_in"
    check(
        "preset-cmd",
        recipe_cmd(refuse, measured_placeholder=True)
        == "python3 -m tools.kdev recipe --preset attn_in --t 1 --idea occupancy --measured-us <median>",
        recipe_cmd(refuse, measured_placeholder=True),
    )

    no_idea = bound.analyze(14336, 5120, 1, "nvfp4", measured_us=30.0)
    check("no-idea-fill", status_of(no_idea) == "fill", status_of(no_idea))

    payload = filled_payload(allow)
    check("json-status", payload["status"] == "go")
    check("json-layer2", "--t 1024" in payload["layer2"]["bench"])
    ggml = bound.analyze(10240, 2560, 512, "ggml_q5_k", idea="aggregate_T")
    check("ggml-layer2", _GGML_LINEAR_BENCH in _layer2(ggml)["bench"])
    check("ggml-no-policy", "--policy" not in _layer2(ggml)["bench"])
    check("ggml-op", _layer2(ggml)["op"] == "ggml_block_linear", _layer2(ggml)["op"])
    check("ggml-profile-required", "bound=profile-required" in render_filled(ggml))
    check("ggml-profile-one-rep", "--repetitions 1" in _layer2(ggml)["ncu"])

    if failures:
        print("self-test FAIL")
        for item in failures:
            print("  " + item)
        return 1
    print("self-test OK  (guide, DRAM refuse, TC allow, status, layer2)")
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        prog="kdev recipe",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    bound.add_problem_arguments(parser)
    parser.add_argument("--json", action="store_true", help="JSON guide or filled card")
    parser.add_argument("--list-ideas", action="store_true")
    parser.add_argument("--list-presets", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args(argv)

    if args.self_test:
        rc = bound.main(["--self-test"])
        return rc or _self_test()
    if args.list_ideas:
        print(bound.format_idea_catalog())
        return 0
    if args.list_presets:
        for name, spec in bound.PRESETS.items():
            print(f"{name:10} n={spec['n']:<6} k={spec['k']:<6} {spec['qtype']}  {spec['label']}")
        return 0

    filling = bound.problem_is_complete(args)
    wants_fill = any((
        args.preset, args.n is not None, args.k is not None, args.t is not None,
        args.idea, args.measured_us is not None, args.qtype,
        args.mma_json, args.mma_per_s is not None,
        args.needs_tmem, args.needs_tcgen05, args.smem_bytes is not None,
        args.cluster != 1, args.phase is not None,
    ))
    if wants_fill and not filling:
        parser.error("filling the card requires --preset or --n/--k, plus --t")

    if not filling:
        if args.json:
            print(json.dumps(_guide_payload(), indent=2))
        else:
            print(render_guide().rstrip())
        return 0

    try:
        card = bound.analyze_from_args(args)
    except ValueError as exc:
        parser.error(str(exc))

    if args.json:
        print(json.dumps(filled_payload(card), indent=2))
    else:
        print(render_filled(card))
    if card.get("idea") and card["idea"]["verdict"] == "refuse":
        return 2
    if not card["sm120"]["legal"]:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
