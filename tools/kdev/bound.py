"""Layer-0 Linear bound classifier.

Host-only. Uses the public Linear model-byte floor from
``docs/maintainer/linear-benchmark.md`` and RTX 5090 roofs. It does not emulate
caches. It decides which idea class is legal to implement.

    python3 -m tools.kdev bound --preset attn_in --t 1 --idea occupancy
    python3 -m tools.kdev bound --n 14336 --k 5120 --t 1024 --qtype nvfp4 --json
    python3 -m tools.kdev bound --self-test
"""

from __future__ import annotations

import argparse
import json
import os
import sys

# RTX 5090 facts used by the Linear bench contract.
DRAM_SPEC_GB_S = 1792.0
SUSTAINED_READ_GB_S = 1674.5
# NVIDIA dense FP4 TOPS; the Linear bench Dense-FP4-peak column uses this roof.
DENSE_FP4_TFLOP_S = 1676.0
SMEM_LIMIT_BYTES = 101376  # sm_120 shared memory per SM
RIDGE_FLOP_PER_BYTE = (DENSE_FP4_TFLOP_S * 1e12) / (SUSTAINED_READ_GB_S * 1e9)

# Warp MMA atoms in ops/common/mma.cuh.
NVFP4_MMA = (16, 8, 64)  # m16n8k64, 16384 FLOP
BF16_MMA = (16, 8, 16)  # m16n8k16, 4096 FLOP
S8_MMA = (16, 8, 32)  # m16n8k32, 8192 FLOP

QTYPES = {
    "q4": {"group": 64, "bytes_per_group": 34, "pad_k": 128},
    "q5": {"group": 64, "bytes_per_group": 42, "pad_k": 128},
    "q6": {"group": 64, "bytes_per_group": 50, "pad_k": 128},
    "w8": {"group": 32, "bytes_per_group": 34, "pad_k": 128},
    "nvfp4": {"group": 16, "bytes_per_group": 9, "pad_k": 0},
    "bf16": {"group": 1, "bytes_per_group": 2, "pad_k": 0},
}

PRESETS = {
    "attn_in": {"n": 14336, "k": 5120, "qtype": "nvfp4", "label": "attn in"},
    "gdn_in": {"n": 16384, "k": 5120, "qtype": "nvfp4", "label": "GDN in"},
    "mlp_up": {"n": 34816, "k": 5120, "qtype": "nvfp4", "label": "MLP up"},
    "out_proj": {"n": 5120, "k": 6144, "qtype": "nvfp4", "label": "out proj"},
    "mlp_down": {"n": 5120, "k": 17408, "qtype": "nvfp4", "label": "MLP down"},
}

# Idea classes an agent must name. Verdict depends on the bound.
IDEAS = (
    "occupancy",
    "more_smem",
    "software_pipeline",
    "tma",
    "tile_shape",
    "epilogue_fusion",
    "split_k",
    "weight_replay",
    "aggregate_T",
    "force_mma_small_t",
    "flashattention_tiling",
    "new_family",
    "tcgen05",
    "tmem",
    "sm100",
    "2sm_mma",
    "cluster_multicast",
)


def _align_up(value: int, alignment: int) -> int:
    if alignment <= 1:
        return value
    return ((value + alignment - 1) // alignment) * alignment


def weight_bytes(n: int, k: int, qtype: str) -> int:
    spec = QTYPES[qtype]
    if qtype == "bf16":
        return 2 * n * k
    if qtype == "nvfp4":
        # Production NVFP4 packing: N%128==0, K%64==0, no K-pad. Code + UE4M3 scales.
        return n * k * spec["bytes_per_group"] // spec["group"]
    padded_k = _align_up(k, spec["pad_k"])
    groups = n * padded_k // spec["group"]
    return groups * spec["bytes_per_group"]


def activation_bytes(n: int, k: int, t: int) -> int:
    # Public Linear floor is always BF16 x + BF16 out, including AllowA4.
    return 2 * t * (n + k)


def model_bytes(n: int, k: int, t: int, qtype: str) -> int:
    return weight_bytes(n, k, qtype) + activation_bytes(n, k, t)


def useful_flops(n: int, k: int, t: int) -> int:
    return 2 * n * k * t


def mma_count(n: int, k: int, t: int, atom: tuple[int, int, int]) -> int:
    m, nn, kk = atom
    return ((n + m - 1) // m) * ((t + nn - 1) // nn) * ((k + kk - 1) // kk)


def infer_phase(t: int) -> str:
    if t <= 16:
        return "decode"
    if t >= 256:
        return "prefill"
    return "mixed"


def _load_mma_rate(path: str | None) -> float | None:
    if path is None:
        default = os.path.join(os.getcwd(), "profiles", "kdev", "mma_issue.json")
        path = default if os.path.isfile(default) else None
    if not path:
        return None
    with open(path) as handle:
        payload = json.load(handle)
    nvfp4 = payload.get("nvfp4") or {}
    rate = nvfp4.get("mma_per_s")
    return float(rate) if rate else None


def classify_sm120(*, needs_tmem: bool, needs_tcgen05: bool, cluster: int,
                   smem_bytes: int | None) -> dict:
    reasons = []
    if needs_tmem or needs_tcgen05:
        reasons.append("sm_120a has no TMEM / tcgen05; warp mma.sync only")
    if cluster > 1:
        reasons.append("sm_120a has no cluster-multicast TMA; cluster must be 1")
    if smem_bytes is not None and smem_bytes > SMEM_LIMIT_BYTES:
        reasons.append(f"smem {smem_bytes} B exceeds sm_120 limit {SMEM_LIMIT_BYTES} B")
    return {"legal": not reasons, "reasons": reasons}


def classify_idea(idea: str, bound: str, t: int, phase: str) -> dict:
    idea = idea.strip()
    if idea in {"tcgen05", "tmem", "sm100", "2sm_mma", "cluster_multicast"}:
        return {
            "name": idea,
            "verdict": "refuse",
            "reason": "illegal on sm_120a (RTX 5090). Stay on warp mma.sync + single-CTA TMA.",
        }
    if idea == "weight_replay":
        return {
            "name": idea,
            "verdict": "allow",
            "reason": "Cutting extra DRAM bytes always attacks the model-byte floor.",
        }
    if idea == "aggregate_T":
        return {
            "name": idea,
            "verdict": "allow",
            "reason": "More useful work per weight pass (B, MTP/DFlash width). Schedule, not a new MMA family.",
        }
    if idea == "force_mma_small_t":
        if t <= 16:
            return {
                "name": idea,
                "verdict": "refuse",
                "reason": "T<=16 is DRAM + issue overhead. Do not force W4A4/MMA off the SIMT/A16 crossover.",
            }
        return {
            "name": idea,
            "verdict": "measure",
            "reason": "T is above the decode band; compare the production crossover, do not assume MMA wins.",
        }
    if idea == "flashattention_tiling":
        if phase == "decode" or bound == "DRAM":
            return {
                "name": idea,
                "verdict": "refuse",
                "reason": "FlashAttention-style compute pipelining does not help a DRAM-bound Linear/GEMV.",
            }
        return {
            "name": idea,
            "verdict": "allow",
            "reason": "Legal for compute-bound GQA prefill. Not a Linear idea.",
        }
    if bound == "DRAM":
        if idea in {"occupancy", "more_smem", "software_pipeline", "tile_shape", "new_family"}:
            return {
                "name": idea,
                "verdict": "refuse",
                "reason": "DRAM-bound. These are compute-side tools. They add instructions and often extra bytes.",
            }
        if idea == "tma":
            return {
                "name": idea,
                "verdict": "measure",
                "reason": "Only if NCU shows DRAM well below the 1674.5 GB/s probe because of load inefficiency.",
            }
        if idea in {"epilogue_fusion", "split_k"}:
            return {
                "name": idea,
                "verdict": "measure",
                "reason": "Only if it reduces model-byte traffic. Split-K that re-reads weights is refuse.",
            }
        return {"name": idea, "verdict": "measure", "reason": "Name how this cuts bytes or raises T per weight pass."}
    # tensor-core bound
    if idea in {"tma", "tile_shape", "software_pipeline", "epilogue_fusion"}:
        return {
            "name": idea,
            "verdict": "allow",
            "reason": "Compute-bound. Search these as parameters inside the existing SM120 NVFP4 family.",
        }
    if idea in {"occupancy", "more_smem", "split_k", "new_family"}:
        return {
            "name": idea,
            "verdict": "measure",
            "reason": "Legal to try only inside the SM120 family, after CUTLASS/current-kernel gap is known.",
        }
    return {"name": idea, "verdict": "measure", "reason": "State the metric this idea moves (tensor-pipe busy, not DRAM %)."}


def analyze(
    n: int,
    k: int,
    t: int,
    qtype: str,
    *,
    policy: str = "a16",
    phase: str | None = None,
    idea: str | None = None,
    measured_us: float | None = None,
    mma_per_s: float | None = None,
    needs_tmem: bool = False,
    needs_tcgen05: bool = False,
    cluster: int = 1,
    smem_bytes: int | None = None,
    label: str = "",
) -> dict:
    qtype = qtype.lower()
    if qtype not in QTYPES:
        raise ValueError(f"unknown qtype '{qtype}'; expected {', '.join(QTYPES)}")
    if n <= 0 or k <= 0 or t <= 0:
        raise ValueError("n, k, t must be positive")
    w_bytes = weight_bytes(n, k, qtype)
    a_bytes = activation_bytes(n, k, t)
    bytes_ = w_bytes + a_bytes
    flops = useful_flops(n, k, t)
    ai = flops / bytes_
    t_mem_us = (bytes_ / (SUSTAINED_READ_GB_S * 1e9)) * 1e6
    t_comp_us = (flops / (DENSE_FP4_TFLOP_S * 1e12)) * 1e6
    atom = NVFP4_MMA if qtype == "nvfp4" else (BF16_MMA if qtype == "bf16" else S8_MMA)
    count = mma_count(n, k, t, atom)
    t_issue_us = None
    if mma_per_s and mma_per_s > 0:
        t_issue_us = (count / mma_per_s) * 1e6
    floors = [t_mem_us, t_comp_us]
    if t_issue_us is not None:
        floors.append(t_issue_us)
    floor_us = max(floors)
    compute_us = t_comp_us if t_issue_us is None else max(t_comp_us, t_issue_us)
    bound = "DRAM" if t_mem_us >= compute_us else "tensor-core"
    phase = phase or infer_phase(t)
    sm120 = classify_sm120(
        needs_tmem=needs_tmem, needs_tcgen05=needs_tcgen05,
        cluster=cluster, smem_bytes=smem_bytes,
    )
    idea_card = classify_idea(idea, bound, t, phase) if idea else None
    if idea_card and not sm120["legal"]:
        idea_card = {
            "name": idea,
            "verdict": "refuse",
            "reason": "; ".join(sm120["reasons"]),
        }

    allowed, refused, measured = [], [], []
    for name in IDEAS:
        card = classify_idea(name, bound, t, phase)
        if card["verdict"] == "allow":
            allowed.append(name)
        elif card["verdict"] == "refuse":
            refused.append(name)
        else:
            measured.append(name)

    next_step = (
        "Do not write CUDA. Attack bytes or raise T per weight pass (aggregate_T / weight_replay)."
        if bound == "DRAM"
        else "Stay in the SM120 NVFP4 family. Autotune tile/TMA/pipeline; do not invent a new MMA ISA."
    )
    if idea_card and idea_card["verdict"] == "refuse":
        next_step = f"REFUSE this idea. {idea_card['reason']}"
    elif idea_card and idea_card["verdict"] == "allow":
        next_step = f"Allowed. Implement only as a parameter inside the current family, then Layer 2 microbench."

    measured_pct = None
    if measured_us is not None and floor_us > 0:
        measured_pct = 100.0 * floor_us / measured_us

    denom = (2 * n * k / (DENSE_FP4_TFLOP_S * 1e12)) - (2 * (n + k) / (SUSTAINED_READ_GB_S * 1e9))
    ridge_t = (w_bytes / (SUSTAINED_READ_GB_S * 1e9) / denom) if denom > 0 else None

    return {
        "layer": 0,
        "label": label,
        "problem": {"n": n, "k": k, "t": t, "qtype": qtype, "policy": policy, "phase": phase},
        "weight_bytes": w_bytes,
        "activation_bytes": a_bytes,
        "model_bytes": bytes_,
        "useful_flops": flops,
        "ai_flop_per_byte": ai,
        "ridge_flop_per_byte": RIDGE_FLOP_PER_BYTE,
        "ridge_t": ridge_t,
        "t_mem_us": t_mem_us,
        "t_comp_us": t_comp_us,
        "t_issue_us": t_issue_us,
        "mma_count": count,
        "mma_atom": {"m": atom[0], "n": atom[1], "k": atom[2]},
        "floor_us": floor_us,
        "bound": bound,
        "sm120": sm120,
        "idea": idea_card,
        "allowed": allowed,
        "measure": measured,
        "refused": refused,
        "measured_us": measured_us,
        "floor_pct_of_measured": measured_pct,
        "next": next_step,
        "roofs": {
            "dram_spec_gb_s": DRAM_SPEC_GB_S,
            "sustained_read_gb_s": SUSTAINED_READ_GB_S,
            "dense_fp4_tflop_s": DENSE_FP4_TFLOP_S,
        },
    }


def render(card: dict) -> str:
    p = card["problem"]
    idea = card.get("idea")
    verdict = (idea or {}).get("verdict", "n/a")
    lines = [
        f"[kdev-bound] {card.get('label') or 'linear'} "
        f"[{p['n']},{p['k']}] T={p['t']} {p['qtype']} {p['policy']} phase={p['phase']}",
        f"       bound={card['bound']}  AI={card['ai_flop_per_byte']:.1f} FLOP/B  "
        f"ridge={card['ridge_flop_per_byte']:.0f} FLOP/B"
        + (f"  ridge_T≈{card['ridge_t']:.0f}" if card.get("ridge_t") else ""),
        f"       t_mem={card['t_mem_us']:.2f}us  t_comp={card['t_comp_us']:.2f}us"
        + (f"  t_issue={card['t_issue_us']:.2f}us" if card["t_issue_us"] is not None else "  t_issue=n/a")
        + f"  floor={card['floor_us']:.2f}us",
        f"       model_bytes={card['model_bytes']}  weight={card['weight_bytes']}  act={card['activation_bytes']}",
    ]
    if card["measured_us"] is not None:
        lines.append(
            f"       measured={card['measured_us']:.2f}us  "
            f"floor/measured={card['floor_pct_of_measured']:.1f}%"
        )
    if not card["sm120"]["legal"]:
        lines.append("       sm120=ILLEGAL  " + "; ".join(card["sm120"]["reasons"]))
    if idea:
        lines.append(f"       idea={idea['name']}  verdict={verdict.upper()}  {idea['reason']}")
    lines.append(f"       allow: {', '.join(card['allowed'])}")
    lines.append(f"       refuse: {', '.join(card['refused'])}")
    lines.append(f"       next: {card['next']}")
    return "\n".join(lines)


def _self_test() -> int:
    failures = []

    def check(name, cond, detail=""):
        if not cond:
            failures.append(f"{name}: {detail}")

    # linear-benchmark.md §9: Q4 [4096,5120] T=1 weight bytes = 11141120.
    check("q4-weight", weight_bytes(4096, 5120, "q4") == 11141120,
          str(weight_bytes(4096, 5120, "q4")))
    # NVFP4 9/16 bytes, attn-in.
    check("nvfp4-weight", weight_bytes(14336, 5120, "nvfp4") == 14336 * 5120 * 9 // 16,
          str(weight_bytes(14336, 5120, "nvfp4")))
    d1 = analyze(14336, 5120, 1, "nvfp4")
    check("t1-dram", d1["bound"] == "DRAM", d1["bound"])
    d1024 = analyze(14336, 5120, 1024, "nvfp4")
    check("t1024-tc", d1024["bound"] == "tensor-core", d1024["bound"])
    occ = classify_idea("occupancy", "DRAM", 1, "decode")
    check("occ-refuse", occ["verdict"] == "refuse", occ)
    tc = classify_idea("tcgen05", "tensor-core", 1024, "prefill")
    check("tcgen05-refuse", tc["verdict"] == "refuse", tc)
    tile = classify_idea("tile_shape", "tensor-core", 1024, "prefill")
    check("tile-allow", tile["verdict"] == "allow", tile)
    agg = classify_idea("aggregate_T", "DRAM", 1, "decode")
    check("agg-allow", agg["verdict"] == "allow", agg)
    illegal = analyze(14336, 5120, 1024, "nvfp4", idea="tile_shape", needs_tcgen05=True)
    check("tmem-gate", illegal["idea"]["verdict"] == "refuse", illegal["idea"])
    if failures:
        print("self-test FAIL")
        for item in failures:
            print("  " + item)
        return 1
    print("self-test OK  (q4 bytes, nvfp4 bytes, T=1 DRAM, T=1024 TC, idea gates)")
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        prog="kdev bound",
        description="Layer-0 Linear bound classifier. Host-only. Refuses illegal idea classes.",
    )
    parser.add_argument("--preset", choices=sorted(PRESETS), help="27B NVFP4 Linear problem")
    parser.add_argument("--n", type=int)
    parser.add_argument("--k", type=int)
    parser.add_argument("--t", type=int)
    parser.add_argument("--qtype", default=None, help="q4|q5|q6|w8|nvfp4|bf16")
    parser.add_argument("--policy", default="a16", choices=["a16", "a4"])
    parser.add_argument("--phase", choices=["prefill", "decode", "mixed"])
    parser.add_argument("--idea", help="idea class to gate; see --list-ideas")
    parser.add_argument("--measured-us", type=float, help="current public-Op median µs")
    parser.add_argument("--mma-json", help="profiles/kdev/mma_issue.json from `kdev mma`")
    parser.add_argument("--mma-per-s", type=float, help="override NVFP4 MMA/s from the issue probe")
    parser.add_argument("--needs-tmem", action="store_true")
    parser.add_argument("--needs-tcgen05", action="store_true")
    parser.add_argument("--cluster", type=int, default=1)
    parser.add_argument("--smem-bytes", type=int)
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--list-ideas", action="store_true")
    parser.add_argument("--list-presets", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args(argv)

    if args.self_test:
        return _self_test()
    if args.list_ideas:
        print("\n".join(IDEAS))
        return 0
    if args.list_presets:
        for name, spec in PRESETS.items():
            print(f"{name:10} n={spec['n']:<6} k={spec['k']:<6} {spec['qtype']}  {spec['label']}")
        return 0

    if args.preset:
        spec = PRESETS[args.preset]
        n, k = spec["n"], spec["k"]
        qtype = args.qtype or spec["qtype"]
        label = spec["label"]
    else:
        if args.n is None or args.k is None:
            parser.error("provide --preset or both --n and --k")
        n, k = args.n, args.k
        qtype = args.qtype or "nvfp4"
        label = ""
    if args.t is None:
        parser.error("--t is required")

    mma_rate = args.mma_per_s if args.mma_per_s is not None else _load_mma_rate(args.mma_json)
    try:
        card = analyze(
            n, k, args.t, qtype,
            policy=args.policy, phase=args.phase, idea=args.idea,
            measured_us=args.measured_us, mma_per_s=mma_rate,
            needs_tmem=args.needs_tmem, needs_tcgen05=args.needs_tcgen05,
            cluster=args.cluster, smem_bytes=args.smem_bytes, label=label,
        )
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 2

    if args.json:
        print(json.dumps(card, indent=2))
    else:
        print(render(card))
    if card.get("idea") and card["idea"]["verdict"] == "refuse":
        return 2
    if not card["sm120"]["legal"]:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
