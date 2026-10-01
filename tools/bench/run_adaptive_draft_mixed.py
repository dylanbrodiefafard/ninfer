#!/usr/bin/env python3
"""Frozen k=3/4/5 on mixed-fixture C=2/C=3 batches (slots want different K).

Also homogeneous C=3 for T_B(K). Resume-friendly. stop_token is WARN; jsonl kept.

  python3 tools/bench/run_adaptive_draft_mixed.py
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
RUNNER = REPO / "tools/bench/run_serve_concurrency.py"
SERVE = REPO / "build/apps/ninfer-serve"
OUT = REPO / "profiles/bench/adaptive-tk-20260830-mixed"
DFLASH = Path(
    "/ssdpool2nvme/local_llm/models/qwen3.8-nvfp4-flash2-nvfp4-bf16codebook-from-bf16/"
    "qwen3_8_27b_nvfp4_dflash_nvfp4.ninfer"
)

COMMON = [
    sys.executable,
    str(RUNNER),
    "--serve",
    str(SERVE),
    "--sampling",
    "stochastic",
    "--temperature",
    "0.6",
    "--top-p",
    "0.95",
    "--top-k",
    "20",
    "--min-p",
    "0",
    "--presence-penalty",
    "0",
    "--suite",
    "decode-saturation",
    "--max-context",
    "16384",
    "--kv-dtype",
    "nvfp4",
]


def job(name: str, fixtures: tuple[str, ...], decode: int, concurrency: int) -> dict:
    return {
        "name": name,
        "fixtures": fixtures,
        "decode": decode,
        "concurrency": concurrency,
        "kv_capacity": "auto" if concurrency > 1 else "16384",
    }


# Homogeneous C=3: T_B(K) for the picker. Same fixture, three seeds.
HOMO_C3 = [
    job("c3-dialogue", ("scenario_story_zh_dialogue",), 1024, 3),
    job("c3-python", ("scenario_code_python",), 2048, 3),
    job("c3-jsonl", ("scenario_structured_jsonl",), 2048, 3),
    job("c3-story", ("scenario_story_en_mystery",), 1024, 3),
    job("c3-cuda", ("scenario_code_cuda",), 2048, 3),
]

# Mixed C=2: (want-k from C=1 frozen) pairs that disagree.
MIX_C2 = [
    job("mix2-dlg-py", ("scenario_story_zh_dialogue", "scenario_code_python"), 1024, 2),  # 3 vs 5
    job("mix2-sci-jsonl", ("scenario_story_zh_scifi", "scenario_structured_jsonl"), 1024, 2),  # 3 vs 5
    job("mix2-dlg-story", ("scenario_story_zh_dialogue", "scenario_story_en_mystery"), 1024, 2),  # 3 vs 4
    job("mix2-cuda-py", ("scenario_code_cuda", "scenario_code_python"), 2048, 2),  # 4 vs 5
    job("mix2-sql-jsonl", ("scenario_structured_sql", "scenario_structured_jsonl"), 2048, 2),  # 4 vs 5
    job("mix2-story-jsonl", ("scenario_story_en_mystery", "scenario_structured_jsonl"), 1024, 2),  # 4 vs 5
    job("mix2-dlg-cuda", ("scenario_story_zh_dialogue", "scenario_code_cuda"), 1024, 2),  # 3 vs 4
]

# Mixed C=3: one slot each of want-3 / want-4 / want-5.
MIX_C3 = [
    job(
        "mix3-dlg-story-py",
        ("scenario_story_zh_dialogue", "scenario_story_en_mystery", "scenario_code_python"),
        1024,
        3,
    ),
    job(
        "mix3-sci-cuda-jsonl",
        ("scenario_story_zh_scifi", "scenario_code_cuda", "scenario_structured_jsonl"),
        1024,
        3,
    ),
]

# Extra 3-vs-5 pairs + T(K) vs B (dialogue T is workload-stable; python checks best-K).
MIX_C2_MORE = [
    job("mix2-logic-dlg", ("thinking_logic_grid", "scenario_story_zh_dialogue"), 1024, 2),
    job("mix2-md-py", ("scenario_translation_markdown", "scenario_code_python"), 1024, 2),
]
HOMO_B = [
    job("c4-dialogue", ("scenario_story_zh_dialogue",), 1024, 4),
    job("c8-dialogue", ("scenario_story_zh_dialogue",), 1024, 8),
    job("c4-python", ("scenario_code_python",), 2048, 4),
]

# Second mixed suite: 2× confirmation that ΣE/T_B matches frozen winners.
# C=1 wants: 3=dialogue/scifi, 4=story/CUDA/SQL, 5=python/jsonl/csv/logic.
MIX_C2_CONFIRM = [
    job("mix2-dlg-jsonl", ("scenario_story_zh_dialogue", "scenario_structured_jsonl"), 1024, 2),  # 3 vs 5
    job("mix2-sci-py", ("scenario_story_zh_scifi", "scenario_code_python"), 1024, 2),  # 3 vs 5
    job("mix2-csv-scifi", ("scenario_structured_csv", "scenario_story_zh_scifi"), 1024, 2),  # 5 vs 3
    job("mix2-sci-story", ("scenario_story_zh_scifi", "scenario_story_en_mystery"), 1024, 2),  # 3 vs 4
    job("mix2-sci-cuda", ("scenario_story_zh_scifi", "scenario_code_cuda"), 1024, 2),  # 3 vs 4
    job("mix2-dlg-sql", ("scenario_story_zh_dialogue", "scenario_structured_sql"), 1024, 2),  # 3 vs 4
    job("mix2-story-py", ("scenario_story_en_mystery", "scenario_code_python"), 1024, 2),  # 4 vs 5
    job("mix2-cuda-jsonl", ("scenario_code_cuda", "scenario_structured_jsonl"), 2048, 2),  # 4 vs 5
    job("mix2-sql-py", ("scenario_structured_sql", "scenario_code_python"), 2048, 2),  # 4 vs 5
]
MIX_C3_CONFIRM = [
    job(
        "mix3-dlg-sql-jsonl",
        ("scenario_story_zh_dialogue", "scenario_structured_sql", "scenario_structured_jsonl"),
        1024,
        3,
    ),
    job(
        "mix3-sci-story-py",
        ("scenario_story_zh_scifi", "scenario_story_en_mystery", "scenario_code_python"),
        1024,
        3,
    ),
]

JOBS = HOMO_C3 + MIX_C2 + MIX_C3 + MIX_C2_MORE + HOMO_B + MIX_C2_CONFIRM + MIX_C3_CONFIRM
MODES = ("dflash3", "dflash4", "dflash5")
ADAPTIVE_OUT = REPO / "profiles/bench/adaptive-tk-20260830-mixed-adapt"
ADAPTIVE_JOBS = [
    {**spec, "name": spec["name"] + "-adaptive"}
    for spec in MIX_C2 + MIX_C3 + MIX_C2_MORE + MIX_C2_CONFIRM + MIX_C3_CONFIRM
]


def point_name(mode: str, concurrency: int, adaptive: bool = False) -> str:
    tag = f"{mode}_adaptive" if adaptive else mode
    return f"qwen3_8_27b_{tag}_stochastic_decode_saturation_c{concurrency}.json"


def jsonl_has_request_done(path: Path) -> bool:
    if not path.is_file():
        return False
    try:
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line:
                continue
            if json.loads(line).get("event") == "request_done":
                return True
    except (OSError, json.JSONDecodeError):
        return False
    return False


def remaining(spec: dict, dest_root: Path, modes: tuple[str, ...], adaptive: bool) -> list[str]:
    dest = dest_root / spec["name"]
    missing = []
    for mode in modes:
        stem = point_name(mode, spec["concurrency"], adaptive)
        if (dest / "points" / stem).is_file():
            continue
        jsonl = dest / "server" / stem.replace(".json", ".jsonl")
        if jsonl_has_request_done(jsonl):
            continue
        missing.append(mode)
    return missing


def run(spec: dict, mode: str, dest_root: Path, adaptive: bool) -> None:
    dest = dest_root / spec["name"]
    cmd = [
        *COMMON,
        "--artifact",
        f"qwen3_8_27b={DFLASH}",
        "--decode-tokens",
        str(spec["decode"]),
        "--concurrency",
        str(spec["concurrency"]),
        "--kv-capacity",
        spec["kv_capacity"],
        "--output",
        str(dest),
        "--mode",
        mode,
    ]
    if adaptive:
        cmd.append("--adaptive-draft")
    for fixture in spec["fixtures"]:
        cmd.extend(["--saturation-fixture", fixture])
    print(
        "RUN",
        spec["name"],
        spec["fixtures"],
        mode + ("+adaptive" if adaptive else ""),
        "C=" + str(spec["concurrency"]),
        "->",
        dest,
        flush=True,
    )
    try:
        subprocess.check_call(cmd, cwd=REPO)
    except subprocess.CalledProcessError as exc:
        print("WARN", spec["name"], mode, "exit", exc.returncode, flush=True)


def main() -> int:
    if not DFLASH.is_file() or not SERVE.is_file():
        print("missing artifact or ninfer-serve", file=sys.stderr)
        return 2
    phase = "adaptive" if "adaptive" in sys.argv[1:] else "frozen"
    if phase == "adaptive":
        jobs, dest_root, modes, adaptive = ADAPTIVE_JOBS, ADAPTIVE_OUT, ("dflash7",), True
    else:
        jobs, dest_root, modes, adaptive = JOBS, OUT, MODES, False
    dest_root.mkdir(parents=True, exist_ok=True)
    print("PHASE", phase, "dest", dest_root, flush=True)
    for spec in jobs:
        miss = remaining(spec, dest_root, modes, adaptive)
        if not miss:
            print("SKIP", spec["name"], "complete", flush=True)
            continue
        for mode in miss:
            run(spec, mode, dest_root, adaptive)
    print("SWEEP_DONE", phase, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
