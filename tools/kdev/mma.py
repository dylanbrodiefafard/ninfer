"""Layer-1 MMA issue probe runner.

Builds and runs ``ninfer_mma_issue_probe`` in the dev container, then writes
``profiles/kdev/mma_issue.json`` for ``kdev bound`` to consume as t_issue.

    python3 -m tools.kdev mma
    python3 -m tools.kdev mma --atom nvfp4
"""

from __future__ import annotations

import argparse
import json
import os

from . import harness

_TARGET = "ninfer_mma_issue_probe"


def persist(payload: dict) -> str:
    folder = os.path.join(os.getcwd(), "profiles", "kdev")
    os.makedirs(folder, exist_ok=True)
    path = os.path.join(folder, "mma_issue.json")
    with open(path, "w") as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")
    return path


def _parse_probe_json(text: str) -> dict:
    decoder = json.JSONDecoder()
    idx = 0
    while True:
        start = text.find("{", idx)
        if start < 0:
            break
        try:
            payload, _ = decoder.raw_decode(text, start)
        except json.JSONDecodeError:
            idx = start + 1
            continue
        if isinstance(payload, dict) and "sm_count" in payload:
            return payload
        idx = start + 1
    raise json.JSONDecodeError("no mma probe object", text, 0)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="kdev mma", description=__doc__)
    parser.add_argument("--atom", default="all", choices=["nvfp4", "bf16", "s8", "all"])
    parser.add_argument("--iters", type=int, default=8192)
    parser.add_argument("--warps", type=int, default=8)
    parser.add_argument("--blocks-per-sm", type=int, default=2)
    args = parser.parse_args(argv)

    built = harness.build_target(_TARGET)
    if not built.ok:
        print("[kdev-mma] build failed:\n" + built.output[-2000:])
        return 2
    binary = f"{harness.BUILD}/bench/{_TARGET}"
    cmd = (
        f"{binary} --atom {args.atom} --iters {args.iters} "
        f"--warps {args.warps} --blocks-per-sm {args.blocks_per_sm} --json"
    )
    result = harness.run(cmd, check=False)
    if not result.ok:
        print("[kdev-mma] probe failed:\n" + result.output[-2000:])
        return 2
    text = result.stdout + (("\n" + result.stderr) if result.stderr.strip() else "")
    try:
        payload = _parse_probe_json(text)
    except json.JSONDecodeError as exc:
        print(f"[kdev-mma] could not parse JSON ({exc}):\n" + text[-2000:])
        return 2
    path = persist(payload)
    nvfp4 = payload.get("nvfp4") or {}
    tflop = nvfp4.get("tflop_s")
    rate = nvfp4.get("mma_per_s")
    if isinstance(tflop, (int, float)) and isinstance(rate, (int, float)):
        print(
            f"[kdev-mma] device={payload.get('device')} SMs={payload.get('sm_count')}  "
            f"nvfp4={tflop:.2f} TFLOP/s  mma/s={rate:.4e}"
        )
    else:
        print("[kdev-mma] probe completed")
    print(f"       json: {path}")
    return 0
