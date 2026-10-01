#!/usr/bin/env python3
"""Measure tool-heavy synchronized client waves against an already configured server.

Start the server separately with NVFP4 KV, fixed C and speculative configuration.
Use Nsight Systems CUDA+NVTX tracing on that server for callback exposure; this
client records complete responses and request wall times, not inferred GPU times.
Run the same body/seed before and after a candidate, with prefix reuse disabled.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import os
import statistics
import threading
import time
import urllib.request
from pathlib import Path


def qualification(response: dict, expected: dict) -> list[str]:
    """Check the observable complete tool transaction, including its task payload."""
    failures = []
    choices = response.get("choices", [])
    if len(choices) != 1:
        return ["expected one completion choice"]
    choice = choices[0]
    if choice.get("finish_reason") != "tool_calls":
        failures.append(f"finish_reason={choice.get('finish_reason')!r}; expected tool_calls")
    calls = choice.get("message", {}).get("tool_calls", [])
    if len(calls) != 1:
        return [*failures, f"expected one tool call, got {len(calls)}"]
    function = calls[0].get("function", {})
    if function.get("name") != expected["name"]:
        failures.append("wrong completed tool name")

    def unique_object(pairs: list[tuple[str, object]]) -> dict:
        result = dict(pairs)
        if len(result) != len(pairs):
            raise ValueError("duplicate tool argument key")
        return result

    try:
        arguments = json.loads(function.get("arguments", ""), object_pairs_hook=unique_object)
    except (TypeError, ValueError):
        failures.append("tool arguments are not valid JSON")
    else:
        # Canonical JSON preserves numeric/bool types that Python equality folds
        # together (1 == 1.0 == True), while allowing object key reordering.
        if json.dumps(arguments, sort_keys=True) != json.dumps(
            expected["arguments"], sort_keys=True
        ):
            failures.append("tool arguments do not match requested records")
    return failures


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8081")
    parser.add_argument("--model", default="qwen3.8-27b")
    parser.add_argument(
        "--fixture", type=Path, default=Path(__file__).parent / "fixtures/tool_mask_records.json"
    )
    parser.add_argument("--concurrency", type=int, choices=(1, 2, 3, 4, 5, 6), required=True)
    parser.add_argument("--max-tokens", type=int, default=2048)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument(
        "--transitions",
        action="store_true",
        help="qualify serial/mixed/all-tool waves on one server",
    )
    parser.add_argument(
        "--wave",
        choices=("tools", "mixed"),
        default="tools",
        help="measured wave body: every slot calls the tool, or even slots call it "
        "while odd slots decode plain 512-token text",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.runs < 1 or args.warmup < 0 or args.max_tokens < 1:
        parser.error("runs/max-tokens must be positive and warmup nonnegative")
    body = json.loads(args.fixture.read_text())
    if not isinstance(body, dict) or not body.get("messages") or not body.get("tools"):
        parser.error("fixture must contain messages and current tool declarations")
    expected = body.pop("_expected_tool_call", None)
    if not isinstance(expected, dict) or not {"name", "arguments"} <= expected.keys():
        parser.error("fixture needs _expected_tool_call with name and exact arguments")
    body.update(
        model=args.model,
        stream=False,
        temperature=0,
        seed=42,
        max_completion_tokens=args.max_tokens,
        tool_choice="auto",
    )
    body.pop("max_tokens", None)
    body.pop("stream_options", None)
    payload = json.dumps(body).encode()
    headers = {"Content-Type": "application/json"}
    if key := os.environ.get("NINFER_API_KEY"):
        headers["Authorization"] = "Bearer " + key
    plain_body = {
        "model": args.model,
        "stream": False,
        "temperature": 0,
        "seed": 42,
        "max_completion_tokens": 512,
        "chat_template_kwargs": {"enable_thinking": False},
        "messages": [
            {
                "role": "user",
                "content": "List every integer from 1 through 1000 in order, one per line. Do not abbreviate or skip any integers.",
            }
        ],
    }
    records = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        modes = (
            ["plain", "tools", "mixed", "tools", "plain"]
            if args.transitions
            else [args.wave] * (args.warmup + args.runs)
        )
        for wave, mode in enumerate(modes):
            barrier = threading.Barrier(args.concurrency)

            def request(
                slot: int, *, mode: str = mode, barrier: threading.Barrier = barrier
            ) -> dict:
                tool_request = mode == "tools" or (mode == "mixed" and slot % 2 == 0)
                request_body = body if tool_request else plain_body
                req = urllib.request.Request(
                    args.base_url.rstrip("/") + "/v1/chat/completions",
                    data=json.dumps(request_body).encode(),
                    headers=headers,
                )
                barrier.wait(timeout=30)
                started = time.perf_counter()
                with urllib.request.urlopen(req, timeout=1800) as response:
                    result = json.load(response)
                if result.get("error") or not result.get("choices"):
                    raise RuntimeError(f"invalid completion response: {result}")
                if tool_request:
                    failures = qualification(result, expected)
                else:
                    choice = result["choices"][0]
                    message = choice.get("message", {})
                    failures = []
                    if (
                        choice.get("finish_reason") != "length"
                        or result.get("usage", {}).get("completion_tokens") != 512
                    ):
                        failures.append("plain request did not complete its 512-token budget")
                    if message.get("tool_calls") or not message.get("content"):
                        failures.append("plain request had missing text or unexpected tool calls")
                return {
                    "slot": slot,
                    "tool_request": tool_request,
                    "wall_seconds": time.perf_counter() - started,
                    "response": result,
                    "qualification_failures": failures,
                }

            started = time.perf_counter()
            requests = list(pool.map(request, range(args.concurrency)))
            row = {
                "wave": wave,
                "mode": mode,
                "warmup": not args.transitions and wave < args.warmup,
                "wall_seconds": time.perf_counter() - started,
                "requests": requests,
            }
            records.append(row)
            print(
                json.dumps(
                    {"wave": wave, "warmup": row["warmup"], "wall_seconds": row["wall_seconds"]}
                ),
                flush=True,
            )
    measured = [r["wall_seconds"] for r in records if not r["warmup"]]
    report = {
        "artifact_type": "ninfer_tool_mask_overlap_benchmark",
        "schema_version": 1,
        "transitions": args.transitions,
        "plain_request": plain_body if args.transitions else None,
        "fixture": str(args.fixture.resolve()),
        "request_sha256": hashlib.sha256(payload).hexdigest(),
        "request": body,
        "concurrency": args.concurrency,
        "expected_tool_call": expected,
        "qualified": all(not q["qualification_failures"] for r in records for q in r["requests"]),
        "median_wave_wall_seconds": statistics.median(measured),
        "waves": records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    if not report["qualified"]:
        raise SystemExit("tool transaction qualification failed; diagnostics saved in output")


if __name__ == "__main__":
    main()
