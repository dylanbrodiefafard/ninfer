#!/usr/bin/env python3
"""Resume-friendly C=1/C=2 static + adaptive sweep for adaptive draft T(K)/E.

Frozen k=3/4/5: profiles/bench/adaptive-tk-20260830/
Adaptive default: profiles/bench/adaptive-tk-20260830-r9/
Override adaptive dest with ADAPTIVE_OUT=... or argv: frozen | adaptive.

usage:
  python3 tools/bench/run_adaptive_draft_sweep.py frozen     # k=3/4/5, skip complete
  python3 tools/bench/run_adaptive_draft_sweep.py adaptive   # live K, ADAPTIVE_OUT
  python3 tools/bench/run_adaptive_draft_sweep.py frozen --new
  python3 tools/bench/run_adaptive_draft_sweep.py adaptive --new
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
RUNNER = REPO / "tools/bench/run_serve_concurrency.py"
SERVE = REPO / "build/apps/ninfer-serve"
FROZEN_OUT = REPO / "profiles/bench/adaptive-tk-20260830"
ADAPTIVE_OUT = Path(
    os.environ.get(
        "ADAPTIVE_OUT",
        str(REPO / "profiles/bench/adaptive-tk-20260830-r9"),
    )
)

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


def job(
    name: str,
    modes: tuple[str, ...],
    fixture: str,
    decode: int,
    concurrency: int,
    adaptive: bool,
    kv_capacity: str,
    *,
    new: bool = False,
) -> dict:
    return {
        "name": name,
        "modes": modes,
        "fixture": fixture,
        "decode": decode,
        "concurrency": concurrency,
        "adaptive": adaptive,
        "kv_capacity": kv_capacity,
        "new": new,
    }


# Original C=1 + C=2 set (13 adaptive rows).
EXISTING = [
    job("aime-01-dflash", ("dflash3", "dflash4", "dflash5"), "long_decode_aime26_01", 1280, 1, False, "16384"),
    job("sql-dflash", ("dflash3", "dflash4", "dflash5"), "scenario_structured_sql", 2048, 1, False, "16384"),
    job("logic-dflash", ("dflash3", "dflash4", "dflash5"), "thinking_logic_grid", 2048, 1, False, "16384"),
    job("scifi-dflash", ("dflash3", "dflash4", "dflash5"), "scenario_story_zh_scifi", 1024, 1, False, "16384"),
    job("md-dflash", ("dflash3", "dflash4", "dflash5"), "scenario_translation_markdown", 1024, 1, False, "16384"),
    job("dialogue-dflash", ("dflash3", "dflash4", "dflash5"), "scenario_story_zh_dialogue", 1024, 1, False, "16384"),
    job("story-dflash", ("dflash3", "dflash4", "dflash5"), "scenario_story_en_mystery", 1024, 1, False, "16384"),
    job("aime-dflash", ("dflash3", "dflash4", "dflash5"), "long_decode_aime26_15", 4096, 1, False, "16384"),
    job("aime-30-dflash", ("dflash3", "dflash4", "dflash5"), "long_decode_aime26_30", 4096, 1, False, "16384"),
    job("code-cuda-dflash", ("dflash3", "dflash4", "dflash5"), "scenario_code_cuda", 2048, 1, False, "16384"),
    job("code-python-dflash", ("dflash3", "dflash4", "dflash5"), "scenario_code_python", 2048, 1, False, "16384"),
    job("c2-aime-dflash", ("dflash3", "dflash4", "dflash5"), "long_decode_aime26_15", 4096, 2, False, "auto"),
    job("c2-story-dflash", ("dflash3", "dflash4", "dflash5"), "scenario_story_en_mystery", 1024, 2, False, "auto"),
    job("dialogue-dflash-adaptive", ("dflash7",), "scenario_story_zh_dialogue", 1024, 1, True, "16384"),
    job("story-dflash-adaptive", ("dflash7",), "scenario_story_en_mystery", 1024, 1, True, "16384"),
    job("aime-dflash-adaptive", ("dflash7",), "long_decode_aime26_15", 4096, 1, True, "16384"),
    job("aime-30-dflash-adaptive", ("dflash7",), "long_decode_aime26_30", 4096, 1, True, "16384"),
    job("aime-01-dflash-adaptive", ("dflash7",), "long_decode_aime26_01", 1024, 1, True, "16384"),
    job("code-cuda-dflash-adaptive", ("dflash7",), "scenario_code_cuda", 2048, 1, True, "16384"),
    job("code-python-dflash-adaptive", ("dflash7",), "scenario_code_python", 2048, 1, True, "16384"),
    job("sql-dflash-adaptive", ("dflash7",), "scenario_structured_sql", 2048, 1, True, "16384"),
    job("c2-aime-dflash-adaptive", ("dflash7",), "long_decode_aime26_15", 4096, 2, True, "auto"),
    job("c2-story-dflash-adaptive", ("dflash7",), "scenario_story_en_mystery", 1024, 2, True, "auto"),
    job("logic-dflash-adaptive", ("dflash7",), "thinking_logic_grid", 2048, 1, True, "16384"),
    job("scifi-dflash-adaptive", ("dflash7",), "scenario_story_zh_scifi", 1024, 1, True, "16384"),
    job("md-dflash-adaptive", ("dflash7",), "scenario_translation_markdown", 1024, 1, True, "16384"),
]

# Second half: remaining text scenarios + C=2 of the original C=1 set that lacked C=2.
NEW = [
    job("ts-dflash", ("dflash3", "dflash4", "dflash5"), "scenario_code_typescript", 2048, 1, False, "16384", new=True),
    job("jsonl-dflash", ("dflash3", "dflash4", "dflash5"), "scenario_structured_jsonl", 2048, 1, False, "16384", new=True),
    job("csv-dflash", ("dflash3", "dflash4", "dflash5"), "scenario_structured_csv", 2048, 1, False, "16384", new=True),
    job("zhen-dflash", ("dflash3", "dflash4", "dflash5"), "scenario_translation_zh_en", 1024, 1, False, "16384", new=True),
    job("enzh-dflash", ("dflash3", "dflash4", "dflash5"), "scenario_translation_en_zh", 1024, 1, False, "16384", new=True),
    job("c2-cuda-dflash", ("dflash3", "dflash4", "dflash5"), "scenario_code_cuda", 2048, 2, False, "auto", new=True),
    job("c2-python-dflash", ("dflash3", "dflash4", "dflash5"), "scenario_code_python", 2048, 2, False, "auto", new=True),
    job("c2-dialogue-dflash", ("dflash3", "dflash4", "dflash5"), "scenario_story_zh_dialogue", 1024, 2, False, "auto", new=True),
    job("c2-sql-dflash", ("dflash3", "dflash4", "dflash5"), "scenario_structured_sql", 2048, 2, False, "auto", new=True),
    job("c2-scifi-dflash", ("dflash3", "dflash4", "dflash5"), "scenario_story_zh_scifi", 1024, 2, False, "auto", new=True),
    job("c2-logic-dflash", ("dflash3", "dflash4", "dflash5"), "thinking_logic_grid", 2048, 2, False, "auto", new=True),
    job("c2-md-dflash", ("dflash3", "dflash4", "dflash5"), "scenario_translation_markdown", 1024, 2, False, "auto", new=True),
    job("c2-ts-dflash", ("dflash3", "dflash4", "dflash5"), "scenario_code_typescript", 2048, 2, False, "auto", new=True),
    job("ts-dflash-adaptive", ("dflash7",), "scenario_code_typescript", 2048, 1, True, "16384", new=True),
    job("jsonl-dflash-adaptive", ("dflash7",), "scenario_structured_jsonl", 2048, 1, True, "16384", new=True),
    job("csv-dflash-adaptive", ("dflash7",), "scenario_structured_csv", 2048, 1, True, "16384", new=True),
    job("zhen-dflash-adaptive", ("dflash7",), "scenario_translation_zh_en", 1024, 1, True, "16384", new=True),
    job("enzh-dflash-adaptive", ("dflash7",), "scenario_translation_en_zh", 1024, 1, True, "16384", new=True),
    job("c2-cuda-dflash-adaptive", ("dflash7",), "scenario_code_cuda", 2048, 2, True, "auto", new=True),
    job("c2-python-dflash-adaptive", ("dflash7",), "scenario_code_python", 2048, 2, True, "auto", new=True),
    job("c2-dialogue-dflash-adaptive", ("dflash7",), "scenario_story_zh_dialogue", 1024, 2, True, "auto", new=True),
    job("c2-sql-dflash-adaptive", ("dflash7",), "scenario_structured_sql", 2048, 2, True, "auto", new=True),
    job("c2-scifi-dflash-adaptive", ("dflash7",), "scenario_story_zh_scifi", 1024, 2, True, "auto", new=True),
    job("c2-logic-dflash-adaptive", ("dflash7",), "thinking_logic_grid", 2048, 2, True, "auto", new=True),
    job("c2-md-dflash-adaptive", ("dflash7",), "scenario_translation_markdown", 1024, 2, True, "auto", new=True),
    job("c2-ts-dflash-adaptive", ("dflash7",), "scenario_code_typescript", 2048, 2, True, "auto", new=True),
]

JOBS = EXISTING + NEW


def point_name(mode: str, adaptive: bool, concurrency: int) -> str:
    tag = f"{mode}_adaptive" if adaptive else mode
    return f"qwen3_8_27b_{tag}_stochastic_decode_saturation_c{concurrency}.json"


def out_dir(spec: dict) -> Path:
    return (ADAPTIVE_OUT if spec["adaptive"] else FROZEN_OUT) / spec["name"]


def jsonl_has_request_done(path: Path) -> bool:
    if not path.is_file():
        return False
    try:
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line:
                continue
            rec = json.loads(line)
            if rec.get("event") == "request_done":
                return True
    except (OSError, json.JSONDecodeError):
        return False
    return False


def remaining(spec: dict) -> list[str]:
    dest = out_dir(spec)
    missing = []
    for mode in spec["modes"]:
        stem = point_name(mode, spec["adaptive"], spec["concurrency"])
        if (dest / "points" / stem).is_file():
            continue
        jsonl = dest / "server" / stem.replace(".json", ".jsonl")
        if jsonl_has_request_done(jsonl):
            continue
        missing.append(mode)
    return missing


def run(spec: dict, mode: str) -> None:
    dest = out_dir(spec)
    cmd = [
        *COMMON,
        "--artifact",
        f"qwen3_8_27b={DFLASH}",
        "--saturation-fixture",
        spec["fixture"],
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
    if spec["adaptive"]:
        cmd.append("--adaptive-draft")
    print("RUN", spec["name"], mode, "->", dest, flush=True)
    try:
        subprocess.check_call(cmd, cwd=REPO)
    except subprocess.CalledProcessError as exc:
        print("WARN", spec["name"], mode, "exit", exc.returncode, flush=True)


def main() -> int:
    global ADAPTIVE_OUT
    args = [a for a in sys.argv[1:] if a]
    phase = "adaptive"
    new_only = False
    for a in args:
        if a in ("frozen", "adaptive"):
            phase = a
        elif a == "--new":
            new_only = True
        elif a.startswith("ADAPTIVE_OUT="):
            ADAPTIVE_OUT = Path(a.split("=", 1)[1])
        else:
            print("usage: run_adaptive_draft_sweep.py [frozen|adaptive] [--new]", file=sys.stderr)
            return 2
    want_adaptive = phase == "adaptive"
    if not DFLASH.is_file():
        print("missing artifact", file=sys.stderr)
        return 2
    if not SERVE.is_file():
        print("missing ninfer-serve", file=sys.stderr)
        return 2
    print("PHASE", phase, "ADAPTIVE_OUT", ADAPTIVE_OUT, "new_only", new_only, flush=True)
    for spec in JOBS:
        if spec["adaptive"] != want_adaptive:
            continue
        if new_only and not spec["new"]:
            continue
        miss = remaining(spec)
        if not miss:
            print("SKIP", spec["name"], "complete", flush=True)
            continue
        for mode in miss:
            run(spec, mode)
    print("SWEEP_DONE", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
