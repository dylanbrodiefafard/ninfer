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
import math
import os

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
# Candidate atoms for `profiled_new_family`, keyed like profiles/kdev/mma_issue.json.
CANDIDATE_ATOMS = {
    "nvfp4": NVFP4_MMA,
    "bf16": BF16_MMA,
    "s8": S8_MMA,
    "fp8": (16, 8, 32),
    "fp8_k16": (16, 8, 16),
}
# `profiled_new_family` admits a point only while every achieved-bandwidth
# source (model bytes over the measured time, same-kernel DRAM) stays below
# this fraction of the sustained read.
NEW_FAMILY_MAX_READ_FRACTION = 0.6

QTYPES = {
    "q4": {"group": 64, "bytes_per_group": 34, "pad_k": 128},
    "q5": {"group": 64, "bytes_per_group": 42, "pad_k": 128},
    "q6": {"group": 64, "bytes_per_group": 50, "pad_k": 128},
    "w8": {"group": 32, "bytes_per_group": 34, "pad_k": 128},
    "nvfp4": {"group": 16, "bytes_per_group": 9, "pad_k": 0},
    "bf16": {"group": 1, "bytes_per_group": 2, "pad_k": 0},
    "fp8": {"group": 1, "bytes_per_group": 1, "pad_k": 0},
}

PRESETS = {
    "attn_in": {"n": 14336, "k": 5120, "qtype": "nvfp4", "label": "attn in"},
    "gdn_in": {"n": 16384, "k": 5120, "qtype": "nvfp4", "label": "GDN in"},
    "mlp_up": {"n": 34816, "k": 5120, "qtype": "nvfp4", "label": "MLP up"},
    "out_proj": {"n": 5120, "k": 6144, "qtype": "nvfp4", "label": "out proj"},
    "mlp_down": {"n": 5120, "k": 17408, "qtype": "nvfp4", "label": "MLP down"},
}

# Idea classes an agent must name. Verdict depends on the bound.
# Catalog `dram`/`tc` are the typical gate; classify_idea() is the authority
# (force_mma_small_t depends on T; flashattention_tiling depends on phase).
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
    "quality_tradeoff",
    "grid_underfill",
    "profiled_overhead",
    "profiled_new_family",
    "tcgen05",
    "tmem",
    "sm100",
    "2sm_mma",
    "cluster_multicast",
)
ILLEGAL_IDEAS = frozenset({"tcgen05", "tmem", "sm100", "2sm_mma", "cluster_multicast"})
IDEA_CATALOG = {
    "occupancy": {
        "lever": "more CTAs/warps to hide latency",
        "dram": "refuse",
        "tc": "measure",
        "note": "Does not cut Linear weight bytes.",
    },
    "more_smem": {
        "lever": "larger shared-memory tiles or staging",
        "dram": "refuse",
        "tc": "measure",
        "note": "Occupancy cost. Stay under sm_120 99 KiB.",
    },
    "software_pipeline": {
        "lever": "overlap TMA/loads with MMA",
        "dram": "refuse",
        "tc": "allow",
        "note": "Compute-side stages inside the current family.",
    },
    "tma": {
        "lever": "async single-CTA TMA copies",
        "dram": "measure",
        "tc": "allow",
        "note": "DRAM: only if NCU shows load inefficiency well below 1674.5 GB/s. cluster>1 illegal.",
    },
    "tile_shape": {
        "lever": "tile M/N/K, warps, K-stages",
        "dram": "refuse",
        "tc": "allow",
        "note": "Parameters inside the existing family, not a new algorithm.",
    },
    "epilogue_fusion": {
        "lever": "fuse epilogue to cut extra DRAM traffic",
        "dram": "measure",
        "tc": "allow",
        "note": "DRAM: only if it reduces model-byte traffic.",
    },
    "split_k": {
        "lever": "split K with a partial reduction",
        "dram": "measure",
        "tc": "measure",
        "note": "Refuse if it re-reads weights.",
    },
    "weight_replay": {
        "lever": "one weight pass; cut extra bytes / more useful work",
        "dram": "allow",
        "tc": "allow",
        "note": "Always attacks the model-byte floor. Typical decode win.",
    },
    "aggregate_T": {
        "lever": "raise T per weight pass (B, MTP/DFlash width)",
        "dram": "allow",
        "tc": "allow",
        "note": "Schedule, not a new MMA family. Typical decode/concurrent win.",
    },
    "force_mma_small_t": {
        "lever": "force tensor-core MMA at small T",
        "dram": "refuse",
        "tc": "measure",
        "note": "T<=16 is DRAM+issue. Do not force W4A4 off the SIMT/A16 crossover.",
    },
    "flashattention_tiling": {
        "lever": "FlashAttention-style online-softmax pipeline",
        "dram": "refuse",
        "tc": "allow",
        "note": "Not a Linear idea. Legal for compute-bound GQA prefill only.",
    },
    "new_family": {
        "lever": "new kernel family / algorithm",
        "dram": "refuse",
        "tc": "measure",
        "note": "Prefer parameters in the current family. TC: only after current-kernel gap is known.",
    },
    "quality_tradeoff": {
        "lever": "explicitly requested activation-precision qualification",
        "dram": "allow",
        "tc": "allow",
        "note": "Requires a user-requested quality comparison, represented-weight preservation, and a measured baseline. The existing roof is not a prediction of candidate arithmetic.",
    },
    "grid_underfill": {
        "lever": "partition output rows to populate idle SMs",
        "dram": "measure",
        "tc": "measure",
        "note": "Requires a measured Op baseline, profiled grid below 170 CTAs and DRAM below 80% of sustained read. No split-K, new arithmetic, or duplicated weight pass.",
    },
    "profiled_overhead": {
        "lever": "reduce profiler-attributed instruction, address, or synchronization overhead",
        "dram": "measure",
        "tc": "measure",
        "note": "Requires a measured public-Op baseline, same-kernel profiler evidence, and a bounded hypothesis inside the current legal family. Preserve semantics and weight passes; instruction or stall shares are not latency savings.",
    },
    "profiled_new_family": {
        "lever": "replace a kernel well below the byte roof with a different legal SM120 family",
        "dram": "measure",
        "tc": "measure",
        "note": "DRAM points only. Requires a measured public-Op baseline achieving <60% of sustained read (model bytes, and same-kernel DRAM if profiled) and a hypothesis whose max(t_mem, candidate t_issue) beats it. Oracle criteria unchanged.",
    },
    "tcgen05": {
        "lever": "SM100 tcgen05 MMA",
        "dram": "refuse",
        "tc": "refuse",
        "note": "Illegal on sm_120a. Warp mma.sync only.",
        "illegal": True,
    },
    "tmem": {
        "lever": "tensor memory (TMEM)",
        "dram": "refuse",
        "tc": "refuse",
        "note": "Illegal on sm_120a. Accumulators stay in registers.",
        "illegal": True,
    },
    "sm100": {
        "lever": "B200/SM100 pipeline (128x128 TMEM tiles)",
        "dram": "refuse",
        "tc": "refuse",
        "note": "RTX 5090 is sm_120a, not B200.",
        "illegal": True,
    },
    "2sm_mma": {
        "lever": "2-SM MMA",
        "dram": "refuse",
        "tc": "refuse",
        "note": "Illegal on sm_120a.",
        "illegal": True,
    },
    "cluster_multicast": {
        "lever": "cluster-multicast TMA",
        "dram": "refuse",
        "tc": "refuse",
        "note": "Illegal on sm_120a. Single-CTA TMA only (cluster=1).",
        "illegal": True,
    },
}


def format_idea_catalog() -> str:
    """One line per idea. First token is the name (machine-parseable)."""
    lines = []
    for name in IDEAS:
        meta = IDEA_CATALOG[name]
        if meta.get("illegal"):
            gate = f"{'ILLEGAL':<23}"
        else:
            gate = f"DRAM={meta['dram']:<7} TC={meta['tc']:<7}"
        lines.append(f"{name:<22} {gate}  {meta['lever']}. {meta['note']}")
    return "\n".join(lines)


def add_problem_arguments(parser: argparse.ArgumentParser) -> None:
    """Linear point flags shared by `bound` and `recipe`."""
    parser.add_argument("--preset", choices=sorted(PRESETS), help="27B NVFP4 Linear problem")
    parser.add_argument("--n", type=int)
    parser.add_argument("--k", type=int)
    parser.add_argument("--t", type=int)
    parser.add_argument("--qtype", default=None, help="q4|q5|q6|w8|nvfp4|bf16|fp8")
    parser.add_argument("--policy", default="a16", choices=["a16", "a8", "a4"])
    parser.add_argument("--phase", choices=["prefill", "decode", "mixed"])
    parser.add_argument("--idea", help="idea class to gate; see --list-ideas")
    parser.add_argument("--measured-us", type=float, help="current public-Op median µs")
    parser.add_argument("--profiled-ctas", type=int, help="NCU grid size of the measured kernel")
    parser.add_argument("--profiled-dram-gbs", type=float, help="NCU DRAM GB/s of the same kernel")
    parser.add_argument("--bottleneck", choices=["instruction", "address", "synchronization"])
    parser.add_argument(
        "--profile-evidence", help="same-kernel profiler report, metric, and attribution summary"
    )
    parser.add_argument(
        "--hypothesis", help="bounded change and predicted effect on the identified bottleneck"
    )
    parser.add_argument(
        "--candidate-atom",
        choices=sorted(CANDIDATE_ATOMS),
        help="profiled_new_family candidate MMA atom (calibrated by `kdev mma`)",
    )
    parser.add_argument(
        "--operand-terms",
        type=int,
        default=1,
        help="MMAs per candidate tile product (2 for a BF16 hi+lo operand split)",
    )
    parser.add_argument(
        "--candidate-mma-per-s", type=float, help="override the candidate atom MMA/s"
    )
    parser.add_argument("--mma-json", help="profiles/kdev/mma_issue.json from `kdev mma`")
    parser.add_argument(
        "--mma-per-s",
        type=float,
        help="override matching arithmetic atom MMA/s from the issue probe",
    )
    parser.add_argument("--needs-tmem", action="store_true")
    parser.add_argument("--needs-tcgen05", action="store_true")
    parser.add_argument("--cluster", type=int, default=1)
    parser.add_argument("--smem-bytes", type=int)


def problem_is_complete(args) -> bool:
    if getattr(args, "preset", None):
        return getattr(args, "t", None) is not None
    return (
        getattr(args, "n", None) is not None
        and getattr(args, "k", None) is not None
        and getattr(args, "t", None) is not None
    )


def nvfp4_a8_active(n: int, k: int, t: int, policy: str) -> bool:
    return (
        policy == "a8"
        and t >= 2
        and any((n, k) == (shape["n"], shape["k"]) for shape in PRESETS.values())
    )


def analyze_from_args(args) -> dict:
    """Build the Layer-0 card from shared argparse flags. Raises ValueError."""
    if args.preset:
        spec = PRESETS[args.preset]
        n, k = spec["n"], spec["k"]
        qtype = args.qtype or spec["qtype"]
        label = spec["label"]
    else:
        if args.n is None or args.k is None:
            raise ValueError("provide --preset or both --n and --k")
        n, k = args.n, args.k
        qtype = args.qtype or "nvfp4"
        label = ""
    if args.t is None:
        raise ValueError("--t is required")
    atom = (
        fp8_compute_atom(n, k, args.t, args.policy)
        if qtype == "fp8"
        else "fp8_k16"
        if qtype == "nvfp4" and nvfp4_a8_active(n, k, args.t, args.policy)
        else "nvfp4"
    )
    mma_rate = args.mma_per_s if args.mma_per_s is not None else _load_mma_rate(args.mma_json, atom)
    candidate_rate = None
    if args.candidate_atom is not None:
        candidate_rate = (
            args.candidate_mma_per_s
            if args.candidate_mma_per_s is not None
            else _load_mma_rate(args.mma_json, args.candidate_atom)
        )
    card = analyze(
        n,
        k,
        args.t,
        qtype,
        policy=args.policy,
        phase=args.phase,
        idea=args.idea,
        measured_us=args.measured_us,
        mma_per_s=mma_rate,
        needs_tmem=args.needs_tmem,
        needs_tcgen05=args.needs_tcgen05,
        cluster=args.cluster,
        smem_bytes=args.smem_bytes,
        label=label,
        profiled_ctas=args.profiled_ctas,
        profiled_dram_gbs=args.profiled_dram_gbs,
        bottleneck=args.bottleneck,
        profile_evidence=args.profile_evidence,
        hypothesis=args.hypothesis,
        candidate_atom=args.candidate_atom,
        operand_terms=args.operand_terms,
        candidate_mma_per_s=candidate_rate,
    )
    if args.preset:
        card["preset"] = args.preset
    return card


def _align_up(value: int, alignment: int) -> int:
    if alignment <= 1:
        return value
    return ((value + alignment - 1) // alignment) * alignment


def weight_bytes(n: int, k: int, qtype: str) -> int:
    spec = QTYPES[qtype]
    if qtype == "bf16":
        return 2 * n * k
    if qtype == "fp8":
        return n * k + 2 * n
    if qtype == "nvfp4":
        # Production NVFP4 packing: N%128==0, K%64==0, no K-pad. Code + UE4M3 scales.
        return n * k * spec["bytes_per_group"] // spec["group"]
    padded_k = _align_up(k, spec["pad_k"])
    groups = n * padded_k // spec["group"]
    return groups * spec["bytes_per_group"]


def fp8_compute_atom(n: int, k: int, t: int, policy: str) -> str:
    """Public pure-Linear route; fused Ops have their own measured crossover."""
    if policy not in ("a16", "a8"):
        raise ValueError("FP8 bound supports a16 or a8")
    if policy == "a16" or n == 248320:
        return "bf16"
    if (n, k) == (34816, 5120):
        return "fp8" if t == 1 or t >= 5 else "bf16"
    threshold = {(14336, 5120): 12, (16384, 5120): 11, (5120, 6144): 25, (5120, 17408): 25}.get(
        (n, k)
    )
    if threshold is None:
        raise ValueError("unregistered FP8 Linear geometry")
    return "fp8" if t >= threshold else "bf16"


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


def _load_mma_rate(path: str | None, atom: str = "nvfp4") -> float | None:
    if path is None:
        default = os.path.join(os.getcwd(), "profiles", "kdev", "mma_issue.json")
        path = default if os.path.isfile(default) else None
    if not path:
        return None
    with open(path) as handle:
        payload = json.load(handle)
    nvfp4 = payload.get(atom) or {}
    rate = nvfp4.get("mma_per_s")
    return float(rate) if rate else None


def classify_sm120(
    *, needs_tmem: bool, needs_tcgen05: bool, cluster: int, smem_bytes: int | None
) -> dict:
    reasons = []
    if needs_tmem or needs_tcgen05:
        reasons.append("sm_120a has no TMEM / tcgen05; warp mma.sync only")
    if cluster > 1:
        reasons.append("sm_120a has no cluster-multicast TMA; cluster must be 1")
    if smem_bytes is not None and smem_bytes > SMEM_LIMIT_BYTES:
        reasons.append(f"smem {smem_bytes} B exceeds sm_120 limit {SMEM_LIMIT_BYTES} B")
    return {"legal": not reasons, "reasons": reasons}


def _positive_finite(value: float | None) -> bool:
    return value is not None and 0 < value < float("inf")


def _classify_profiled_new_family(
    bound: str,
    *,
    measured_us: float | None,
    measured_read_gbs: float | None,
    profiled_dram_gbs: float | None,
    hypothesis: str | None,
    candidate_predicted_us: float | None,
) -> dict:
    """Admit a different legal family at a DRAM point on measured evidence.

    The byte roof says the point is DRAM-bound, so a compute-side family is
    only worth building while the current kernel stays well below that roof
    (model bytes over the measured time and, when profiled, same-kernel DRAM
    throughput) and the candidate's own floor beats the measured time.
    """
    if bound != "DRAM":
        return {
            "name": "profiled_new_family",
            "verdict": "measure",
            "reason": "Not a DRAM point; use new_family against the measured current-kernel gap.",
        }
    read_limit_gbs = NEW_FAMILY_MAX_READ_FRACTION * SUSTAINED_READ_GB_S
    missing = []
    if not _positive_finite(measured_us):
        missing.append("positive finite --measured-us")
    elif not measured_read_gbs < read_limit_gbs:
        missing.append(
            f"model-byte rate {measured_read_gbs:.0f} GB/s below {read_limit_gbs:.0f} GB/s"
        )
    if profiled_dram_gbs is not None and not (0 < profiled_dram_gbs < read_limit_gbs):
        missing.append(f"--profiled-dram-gbs below {read_limit_gbs:.0f}")
    if not (hypothesis and hypothesis.strip()):
        missing.append("concrete --hypothesis")
    if not _positive_finite(candidate_predicted_us):
        missing.append("--candidate-atom with calibrated MMA/s")
    elif _positive_finite(measured_us) and candidate_predicted_us >= measured_us:
        missing.append(f"candidate floor {candidate_predicted_us:.2f}us below the measured time")
    if missing:
        return {
            "name": "profiled_new_family",
            "verdict": "measure",
            "reason": "Do not write CUDA: require " + "; ".join(missing) + ".",
        }
    return {
        "name": "profiled_new_family",
        "verdict": "allow",
        "reason": (
            f"DRAM point at {measured_read_gbs:.0f} GB/s achieved; candidate floor "
            f"{candidate_predicted_us:.2f}us < measured {measured_us:.2f}us. Build the stated "
            "legal SM120 family only; meet the unchanged oracle criteria and keep the current "
            "route wherever it measures faster."
        ),
    }


def classify_idea(
    idea: str,
    bound: str,
    t: int,
    phase: str,
    *,
    profiled_ctas: int | None = None,
    profiled_dram_gbs: float | None = None,
    measured_us: float | None = None,
    bottleneck: str | None = None,
    profile_evidence: str | None = None,
    hypothesis: str | None = None,
    measured_read_gbs: float | None = None,
    candidate_predicted_us: float | None = None,
) -> dict:
    idea = idea.strip()
    if idea == "profiled_new_family":
        return _classify_profiled_new_family(
            bound,
            measured_us=measured_us,
            measured_read_gbs=measured_read_gbs,
            profiled_dram_gbs=profiled_dram_gbs,
            hypothesis=hypothesis,
            candidate_predicted_us=candidate_predicted_us,
        )
    if idea == "profiled_overhead":
        admitted = (
            measured_us is not None
            and 0 < measured_us < float("inf")
            and bottleneck in {"instruction", "address", "synchronization"}
            and bool(profile_evidence and profile_evidence.strip())
            and bool(hypothesis and hypothesis.strip())
        )
        return {
            "name": idea,
            "verdict": "allow" if admitted else "measure",
            "reason": (
                "Profiler-supported bounded experiment inside the current legal family; preserve semantics and weight passes. Verify the predicted metric change, oracle correctness, and public-Op latency. Profile shares do not predict latency savings."
                if admitted
                else "Do not write CUDA: require a positive finite public-Op baseline, --bottleneck, same-kernel --profile-evidence with attribution, and a specific --hypothesis."
            ),
        }
    if idea == "grid_underfill":
        admitted = (
            measured_us is not None
            and 0 < measured_us < float("inf")
            and profiled_ctas is not None
            and 0 < profiled_ctas < 170
            and profiled_dram_gbs is not None
            and 0 < profiled_dram_gbs < 0.8 * SUSTAINED_READ_GB_S
        )
        return {
            "name": idea,
            "verdict": "allow" if admitted else "measure",
            "reason": (
                "Profiled underfilled grid and low DRAM throughput: partition output rows within the existing family; preserve the K reduction and one weight pass."
                if admitted
                else "Do not write CUDA: require a positive public-Op baseline and same-kernel NCU grid <170 CTAs and DRAM <80% sustained read; no speculative occupancy tuning."
            ),
        }
    if idea == "quality_tradeoff":
        return {
            "name": idea,
            "verdict": "allow",
            "reason": "Explicit quality-comparison experiment only: qualify against the mathematical oracle and measure both quality and latency; no speed inference from the existing roof.",
        }
    if idea in ILLEGAL_IDEAS:
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
        if idea == "epilogue_fusion":
            return {
                "name": idea,
                "verdict": "measure",
                "reason": "Only if it reduces model-byte traffic (fused store, not extra workspace bytes).",
            }
        if idea == "split_k":
            return {
                "name": idea,
                "verdict": "measure",
                "reason": "Only if it reduces model-byte traffic. Split-K that re-reads weights is refuse.",
            }
        return {
            "name": idea,
            "verdict": "measure",
            "reason": "Name how this cuts bytes or raises T per weight pass.",
        }
    # tensor-core bound
    if idea in {"tma", "tile_shape", "software_pipeline", "epilogue_fusion"}:
        return {
            "name": idea,
            "verdict": "allow",
            "reason": "Compute-bound. Search these as parameters inside the existing SM120 kernel family.",
        }
    if idea in {"occupancy", "more_smem", "split_k", "new_family"}:
        return {
            "name": idea,
            "verdict": "measure",
            "reason": "Legal to try only inside the SM120 family, after CUTLASS/current-kernel gap is known.",
        }
    return {
        "name": idea,
        "verdict": "measure",
        "reason": "State the metric this idea moves (tensor-pipe busy, not DRAM %).",
    }


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
    profiled_ctas: int | None = None,
    profiled_dram_gbs: float | None = None,
    bottleneck: str | None = None,
    profile_evidence: str | None = None,
    hypothesis: str | None = None,
    candidate_atom: str | None = None,
    operand_terms: int = 1,
    candidate_mma_per_s: float | None = None,
) -> dict:
    qtype = qtype.lower()
    if qtype not in QTYPES:
        raise ValueError(f"unknown qtype '{qtype}'; expected {', '.join(QTYPES)}")
    if n <= 0 or k <= 0 or t <= 0:
        raise ValueError("n, k, t must be positive")
    if idea is not None and idea.strip() not in IDEAS:
        raise ValueError(f"unknown idea '{idea}'; expected one of: {', '.join(IDEAS)}")
    if measured_us is not None and (not math.isfinite(measured_us) or measured_us <= 0):
        raise ValueError("measured_us must be positive and finite")
    w_bytes = weight_bytes(n, k, qtype)
    a_bytes = activation_bytes(n, k, t)
    bytes_ = w_bytes + a_bytes
    flops = useful_flops(n, k, t)
    ai = flops / bytes_
    t_mem_us = (bytes_ / (SUSTAINED_READ_GB_S * 1e9)) * 1e6
    # FP8 qualification requires its measured instruction roof; never borrow
    # the calibrated NVFP4 rate for an instruction with a different K extent.
    if qtype == "fp8" and (policy not in ("a8", "a16") or not mma_per_s or mma_per_s <= 0):
        raise ValueError("FP8 requires a16/a8 and matching kdev mma calibration (or --mma-per-s)")
    atom = NVFP4_MMA if qtype == "nvfp4" else (BF16_MMA if qtype == "bf16" else S8_MMA)
    nvfp4_a8 = qtype == "nvfp4" and nvfp4_a8_active(n, k, t, policy)
    if nvfp4_a8:
        if not mma_per_s or mma_per_s <= 0:
            raise ValueError(
                "NVFP4 A8 requires matching fp8_k16 kdev mma calibration (or --mma-per-s)"
            )
        atom = (16, 8, 16)
    if qtype == "fp8":
        atom = (16, 8, 32) if fp8_compute_atom(n, k, t, policy) == "fp8" else BF16_MMA
    compute_tflops = (
        mma_per_s * 2 * atom[0] * atom[1] * atom[2] / 1e12
        if qtype == "fp8" or nvfp4_a8
        else DENSE_FP4_TFLOP_S
    )
    t_comp_us = flops / (compute_tflops * 1e6)
    count = mma_count(n, k, t, atom)
    if nvfp4_a8:
        # The admitted route puts tokens on MMA-M and output rows on MMA-N.
        # Count its actual M16/M32 padding, not the legacy output-major floor.
        block_t = 16 if t <= 16 else 32
        count = ((t + block_t - 1) // block_t) * (block_t // 16) * ((n + 7) // 8) * ((k + 15) // 16)
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
    candidate = None
    if candidate_atom is not None:
        if candidate_atom not in CANDIDATE_ATOMS:
            raise ValueError(
                f"unknown candidate atom '{candidate_atom}'; expected {', '.join(CANDIDATE_ATOMS)}"
            )
        if operand_terms < 1:
            raise ValueError("operand_terms must be at least 1")
        if not _positive_finite(candidate_mma_per_s):
            raise ValueError("candidate atom requires matching kdev mma calibration")
        # Output rows on MMA-M and tokens on MMA-N, padded to the atom extents.
        candidate_count = mma_count(n, k, t, CANDIDATE_ATOMS[candidate_atom]) * operand_terms
        candidate_issue_us = candidate_count / candidate_mma_per_s * 1e6
        candidate = {
            "atom": candidate_atom,
            "operand_terms": operand_terms,
            "mma_per_s": candidate_mma_per_s,
            "mma_count": candidate_count,
            "t_issue_us": candidate_issue_us,
            "predicted_us": max(t_mem_us, candidate_issue_us),
        }
    evidence = dict(
        profiled_ctas=profiled_ctas,
        profiled_dram_gbs=profiled_dram_gbs,
        measured_us=measured_us,
        bottleneck=bottleneck,
        profile_evidence=profile_evidence,
        hypothesis=hypothesis,
        measured_read_gbs=bytes_ / (measured_us * 1e3) if measured_us is not None else None,
        candidate_predicted_us=candidate["predicted_us"] if candidate else None,
    )
    sm120 = classify_sm120(
        needs_tmem=needs_tmem,
        needs_tcgen05=needs_tcgen05,
        cluster=cluster,
        smem_bytes=smem_bytes,
    )
    idea_card = classify_idea(idea, bound, t, phase, **evidence) if idea else None
    if idea_card and not sm120["legal"]:
        idea_card = {
            "name": idea,
            "verdict": "refuse",
            "reason": "; ".join(sm120["reasons"]),
        }

    allowed, refused, measured = [], [], []
    for name in IDEAS:
        card = classify_idea(name, bound, t, phase, **evidence)
        if card["verdict"] == "allow":
            allowed.append(name)
        elif card["verdict"] == "refuse":
            refused.append(name)
        else:
            measured.append(name)

    next_step = (
        "Attack bytes, raise T per weight pass, or establish a profiled_overhead hypothesis before writing CUDA."
        if bound == "DRAM"
        else "Stay in the current SM120 kernel family. Autotune tile/TMA/pipeline; do not invent a new MMA ISA."
    )
    if idea_card and idea_card["verdict"] == "refuse":
        next_step = f"REFUSE this idea. {idea_card['reason']}"
    elif idea_card and idea_card["verdict"] == "allow":
        next_step = "Allowed. Implement only as a parameter inside the current family, then Layer 2 microbench."
        if idea_card["name"] == "quality_tradeoff":
            next_step = "Requested quality candidate admitted. Verify candidate ISA and oracle, then measure public-Op latency before paired model quality. Existing roof is baseline-only."
        elif idea_card["name"] in {"profiled_overhead", "profiled_new_family"}:
            next_step = idea_card["reason"]
    elif idea_card and idea_card["name"] in {
        "grid_underfill",
        "profiled_overhead",
        "profiled_new_family",
    }:
        next_step = idea_card["reason"]

    measured_pct = None
    if measured_us is not None and floor_us > 0:
        measured_pct = 100.0 * floor_us / measured_us

    denom = (2 * n * k / (compute_tflops * 1e12)) - (2 * (n + k) / (SUSTAINED_READ_GB_S * 1e9))
    ridge_t = (w_bytes / (SUSTAINED_READ_GB_S * 1e9) / denom) if denom > 0 else None

    return {
        "layer": 0,
        "profiled_ctas": profiled_ctas,
        "profiled_dram_gbs": profiled_dram_gbs,
        "bottleneck": bottleneck,
        "profile_evidence": profile_evidence,
        "hypothesis": hypothesis,
        "candidate": candidate,
        "label": label,
        "problem": {"n": n, "k": k, "t": t, "qtype": qtype, "policy": policy, "phase": phase},
        "weight_bytes": w_bytes,
        "activation_bytes": a_bytes,
        "model_bytes": bytes_,
        "useful_flops": flops,
        "ai_flop_per_byte": ai,
        "ridge_flop_per_byte": compute_tflops * 1000 / SUSTAINED_READ_GB_S,
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


# Card evidence fields shown by `render` and replayed by `recipe`.
EVIDENCE_FIELDS = (
    "profiled_ctas",
    "profiled_dram_gbs",
    "bottleneck",
    "profile_evidence",
    "hypothesis",
)


def render(card: dict) -> str:
    p = card["problem"]
    idea = card.get("idea")
    verdict = (idea or {}).get("verdict", "n/a")
    lines = [
        (
            f"[kdev-bound] {card.get('label') or 'linear'} "
            f"[{p['n']},{p['k']}] T={p['t']} {p['qtype']} {p['policy']} phase={p['phase']}"
        ),
        f"       bound={card['bound']}  AI={card['ai_flop_per_byte']:.1f} FLOP/B  "
        f"ridge={card['ridge_flop_per_byte']:.0f} FLOP/B"
        + (f"  ridge_T≈{card['ridge_t']:.0f}" if card.get("ridge_t") else ""),
        f"       t_mem={card['t_mem_us']:.2f}us  t_comp={card['t_comp_us']:.2f}us"
        + (
            f"  t_issue={card['t_issue_us']:.2f}us"
            if card["t_issue_us"] is not None
            else "  t_issue=n/a"
        )
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
    for field in EVIDENCE_FIELDS:
        if card.get(field) is not None:
            lines.append(f"       {field}={card[field]}")
    candidate = card.get("candidate")
    if candidate:
        lines.append(
            f"       candidate={candidate['atom']} x{candidate['operand_terms']}  "
            f"mma={candidate['mma_count']}  t_issue={candidate['t_issue_us']:.2f}us  "
            f"predicted={candidate['predicted_us']:.2f}us"
        )
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
    check(
        "q4-weight", weight_bytes(4096, 5120, "q4") == 11141120, str(weight_bytes(4096, 5120, "q4"))
    )
    # NVFP4 9/16 bytes, attn-in.
    check(
        "nvfp4-weight",
        weight_bytes(14336, 5120, "nvfp4") == 14336 * 5120 * 9 // 16,
        str(weight_bytes(14336, 5120, "nvfp4")),
    )
    check("fp8-weight", weight_bytes(14336, 5120, "fp8") == 14336 * 5120 + 2 * 14336)
    fp8 = analyze(14336, 5120, 1024, "fp8", policy="a8", mma_per_s=1e10)
    check("fp8-atom", fp8["mma_atom"] == dict(m=16, n=8, k=32))
    check("fp8-roof", abs(fp8["t_comp_us"] - useful_flops(14336, 5120, 1024) / 81.92e6) < 1e-8)
    w4a8 = analyze(16384, 5120, 24, "nvfp4", policy="a8", mma_per_s=1e11)
    check("nvfp4-a8-atom", w4a8["mma_atom"] == dict(m=16, n=8, k=16))
    check("nvfp4-a8-padded-count", w4a8["mma_count"] == 1310720)
    check("nvfp4-a8-roof", abs(w4a8["t_issue_us"] - 13.1072) < 1e-8)
    check("nvfp4-a8-compressed-bytes", w4a8["weight_bytes"] == 47185920)
    check("nvfp4-a8-t1-fallback", not nvfp4_a8_active(5120, 17408, 1, "a8"))
    check("nvfp4-a8-t2", nvfp4_a8_active(5120, 17408, 2, "a8"))
    check("nvfp4-a8-t3", nvfp4_a8_active(5120, 17408, 3, "a8"))
    check(
        "nvfp4-a8-t4",
        analyze(5120, 17408, 4, "nvfp4", policy="a8", mma_per_s=1e11)["mma_atom"]
        == dict(m=16, n=8, k=16),
    )
    try:
        analyze(5120, 17408, 4, "nvfp4", policy="a8")
        check("nvfp4-a8-t4-calibration-required", False)
    except ValueError:
        check("nvfp4-a8-t4-calibration-required", True)
    for ctas, rate, baseline, expected in (
        (80, 691.1, 65.536, "allow"),
        (80, 691.1, None, "measure"),
        (None, 691.1, 65.536, "measure"),
        (170, 691.1, 65.536, "measure"),
        (80, SUSTAINED_READ_GB_S, 65.536, "measure"),
    ):
        gate = classify_idea(
            "grid_underfill",
            "DRAM",
            5,
            "decode",
            profiled_ctas=ctas,
            profiled_dram_gbs=rate,
            measured_us=baseline,
        )
        check(f"grid-underfill-{ctas}-{rate}-{baseline}", gate["verdict"] == expected)
    evidence = dict(
        measured_us=104.448,
        bottleneck="instruction",
        profile_evidence="same-kernel NCU source attribution: FP4 expansion instruction share",
        hypothesis="replace exact sign assembly with byte permutation; reduce emitted instructions",
    )
    for missing in (None, *evidence):
        inputs = dict(evidence)
        if missing is not None:
            inputs[missing] = None
        gate = classify_idea("profiled_overhead", "DRAM", 8, "decode", **inputs)
        check(
            f"profiled-overhead-{missing}",
            gate["verdict"] == ("allow" if missing is None else "measure"),
        )
    for baseline in (0.0, -1.0, float("nan"), float("inf")):
        gate = classify_idea(
            "profiled_overhead", "DRAM", 8, "decode", **(evidence | {"measured_us": baseline})
        )
        check(f"profiled-overhead-baseline-{baseline}", gate["verdict"] == "measure")
    for field in ("profile_evidence", "hypothesis"):
        gate = classify_idea("profiled_overhead", "DRAM", 8, "decode", **(evidence | {field: "  "}))
        check(f"profiled-overhead-blank-{field}", gate["verdict"] == "measure")
    for name in ("occupancy", "tile_shape", "new_family", "tcgen05"):
        gate = classify_idea(name, "DRAM", 8, "decode", **evidence)
        check(f"profiled-overhead-no-bypass-{name}", gate["verdict"] == "refuse")
    for baseline in (0.0, -1.0, float("nan"), float("inf")):
        try:
            analyze(14336, 5120, 8, "nvfp4", measured_us=baseline)
            check(f"invalid-baseline-{baseline}", False)
        except ValueError:
            check(f"invalid-baseline-{baseline}", True)
    illegal_profile = analyze(
        14336, 5120, 8, "nvfp4", idea="profiled_overhead", needs_tcgen05=True, **evidence
    )
    check("profiled-overhead-illegal", illegal_profile["idea"]["verdict"] == "refuse")
    # Hyper-connection down [324,10240] FP8 at T=8: SIMT kernel, BF16 hi+lo candidate.
    family = dict(
        measured_us=9.5,
        hypothesis="m16n8k16 BF16 hi+lo, tokens padded to n8; replaces per-token FMA chains",
    )
    candidate = dict(candidate_atom="bf16", operand_terms=2, candidate_mma_per_s=61.95e9)

    def gate_family(t=8, **overrides):
        inputs = family | candidate | overrides
        return analyze(
            324, 10240, t, "fp8", mma_per_s=61.95e9, idea="profiled_new_family", **inputs
        )

    admitted = gate_family()
    check("new-family-allow", admitted["idea"]["verdict"] == "allow", admitted["idea"])
    check("new-family-count", admitted["candidate"]["mma_count"] == 21 * 640 * 2)
    check(
        "new-family-floor",
        admitted["candidate"]["predicted_us"]
        == max(admitted["t_mem_us"], admitted["candidate"]["t_issue_us"]),
    )
    check(
        "new-family-profiled-allow",
        gate_family(profiled_dram_gbs=350.0)["idea"]["verdict"] == "allow",
    )
    # Slightly faster than the 60% rate, so the achieved read sits just above the limit.
    near_roof_us = 0.99 * admitted["model_bytes"] / (0.6 * SUSTAINED_READ_GB_S * 1e3)
    for name, overrides in (
        ("no-baseline", {"measured_us": None}),
        ("no-hypothesis", {"hypothesis": None}),
        ("blank-hypothesis", {"hypothesis": "  "}),
        ("near-roof-measured", {"measured_us": near_roof_us}),
        ("near-roof-profiled", {"profiled_dram_gbs": 0.6 * SUSTAINED_READ_GB_S}),
        ("zero-profiled", {"profiled_dram_gbs": 0.0}),
        ("slow-candidate", {"operand_terms": 200}),
    ):
        gate = gate_family(**overrides)
        check(f"new-family-{name}", gate["idea"]["verdict"] == "measure", gate["idea"])
    no_candidate = analyze(
        324, 10240, 8, "fp8", mma_per_s=61.95e9, idea="profiled_new_family", **family
    )
    check("new-family-no-candidate", no_candidate["idea"]["verdict"] == "measure")
    tc_point = gate_family(t=4096, measured_us=500.0)
    check("new-family-tc-point", tc_point["idea"]["verdict"] == "measure", tc_point["bound"])
    illegal_family = gate_family(needs_tcgen05=True)
    check("new-family-illegal", illegal_family["idea"]["verdict"] == "refuse")
    for name in ("new_family", "force_mma_small_t", "tile_shape", "occupancy"):
        gate = analyze(324, 10240, 8, "fp8", mma_per_s=61.95e9, idea=name, **family, **candidate)
        check(f"new-family-no-bypass-{name}", gate["idea"]["verdict"] == "refuse")
    try:
        analyze(324, 10240, 8, "fp8", mma_per_s=61.95e9, candidate_atom="bf16")
        check("new-family-calibration-required", False)
    except ValueError:
        check("new-family-calibration-required", True)
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
    check("catalog-keys", tuple(IDEA_CATALOG) == IDEAS, str(tuple(IDEA_CATALOG)))
    catalog_names = tuple(line.split()[0] for line in format_idea_catalog().splitlines())
    check("catalog-lines", catalog_names == IDEAS, str(catalog_names))
    try:
        analyze(14336, 5120, 1, "nvfp4", idea="not_an_idea")
        check("unknown-idea", False, "expected ValueError")
    except ValueError:
        check("unknown-idea", True)
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
    add_problem_arguments(parser)
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--list-ideas", action="store_true")
    parser.add_argument("--list-presets", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args(argv)

    if args.self_test:
        return _self_test()
    if args.list_ideas:
        print(format_idea_catalog())
        return 0
    if args.list_presets:
        for name, spec in PRESETS.items():
            print(f"{name:10} n={spec['n']:<6} k={spec['k']:<6} {spec['qtype']}  {spec['label']}")
        return 0

    try:
        card = analyze_from_args(args)
    except ValueError as exc:
        parser.error(str(exc))

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
