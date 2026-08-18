#!/usr/bin/env python3
"""Fast-iteration speed suite for the live NInfer serve (default :8081).

Runs a scenario matrix (small/large prompts, thinking on/off, multi-turn,
tools, context prefill, decode-heavy) against the running server and reports
per-case prefill tok/s, decode tok/s, TTFT, and end-to-end wall time.

Metrics come from the server's llama.cpp-style `timings` block:
  prefill_tok_s  = timings.prompt_per_second
  ttft_ms        = client wall time to first content/reasoning delta (stream)
                   or timings.prompt_ms (non-stream; server-side prefill time)
  decode_tok_s   = timings.predicted_per_second
  e2e_s          = client wall time, request sent -> response complete

Outputs (under --out-dir, default profiles/bench/speed-suite/):
  speed-suite-<label>.json   full per-run records + per-case aggregates
  speed-suite-<label>.csv    one aggregate row per case (publishable)

Stdlib only; never restarts or reconfigures the engine, hits the live endpoint.
Run from the repo root or anywhere; fixture paths are resolved relative to the
repo root (auto-detected) unless given as an absolute path.

    python3 tools/bench/run_speed_suite.py --label LABEL --runs 2 --warmup 1
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import statistics
import sys
import time
import urllib.request
from pathlib import Path


def repo_root() -> Path:
    """Repo root = the directory containing CMakeLists.txt (walk up from here)."""
    here = Path(__file__).resolve().parent
    for candidate in (here, *here.parents[:6]):
        if (candidate / "CMakeLists.txt").is_file():
            return candidate
    return here.parents[2]


ROOT = repo_root()
FIXTURES_DIR = ROOT / "tools" / "bench" / "fixtures"


def load_key(api_key: str | None) -> str:
    if api_key:
        return api_key
    for name in ("NINFER_API_KEY", "LLAMA_CPP_LOCAL_API_KEY"):
        if os.environ.get(name):
            return os.environ[name]
    for env in (
        Path("/home/dylan/docker/.env"),
        ROOT.parent.parent / ".env",  # <workspace>/.env when repo is <workspace>/ninfer/repo
        Path.cwd() / ".env",
    ):
        if env.exists():
            for line in env.read_text().splitlines():
                for prefix in ("LLAMA_CPP_REMOTE_API_KEY=", "LLAMA_CPP_LOCAL_API_KEY="):
                    if line.startswith(prefix):
                        return line.split("=", 1)[1].strip().strip('"')
    return ""


def resolve_fixture(ref: str) -> Path:
    """Absolute path as-is; otherwise repo-root-relative (or under tools/bench/fixtures/)."""
    p = Path(ref)
    if p.is_absolute():
        return p
    if (ROOT / ref).exists():
        return ROOT / ref
    if (FIXTURES_DIR / ref).exists():
        return FIXTURES_DIR / ref
    return ROOT / ref


def case_body(case: dict, model: str, stream: bool) -> dict:
    body: dict = {
        "model": model,
        "stream": stream,
    }
    if case.get("max_tokens") is not None:
        body["max_tokens"] = case["max_tokens"]
    if case.get("prompt") is not None:
        body["messages"] = [{"role": "user", "content": case["prompt"]}]
    elif case.get("prompt_file") is not None:
        path = resolve_fixture(case["prompt_file"])
        body["messages"] = [{"role": "user", "content": path.read_text(encoding="utf-8").strip()}]
    elif case.get("messages") is not None:
        body["messages"] = case["messages"]
    elif case.get("fixture") is not None:
        raw = json.loads(resolve_fixture(case["fixture"]).read_text(encoding="utf-8"))
        if isinstance(raw, list):
            body["messages"] = raw
        else:
            body["messages"] = raw.get("messages", [])
            for key in ("tools", "tool_choice"):
                if key in raw:
                    body[key] = raw[key]
    else:
        raise ValueError(f"case {case.get('id')!r} has no prompt/messages/fixture")
    thinking = bool(case.get("thinking", False))
    body["chat_template_kwargs"] = {"enable_thinking": thinking}
    body["enable_thinking"] = thinking
    if case.get("seed") is not None:
        body["seed"] = case["seed"]
    if not body["messages"]:
        raise ValueError(f"case {case.get('id')!r} resolved to zero messages")
    return body


def cache_bust(body: dict, nonce: str) -> None:
    """Force a fresh prefill: a unique per-request nonce defeats prefix reuse."""
    messages = body.get("messages") or []
    for message in reversed(messages):
        content = message.get("content")
        if isinstance(content, str):
            message["content"] = f"[bench {nonce}] {content}"
            return
    raise ValueError("no string-content message to cache-bust")


def post(base: str, key: str, body: dict, timeout: float) -> tuple[dict, dict]:
    """Non-streaming request. Returns (response, {wall_s})."""
    t0 = time.perf_counter()
    req = urllib.request.Request(
        f"{base.rstrip('/')}/v1/chat/completions",
        data=json.dumps(body).encode(),
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        data = json.loads(resp.read().decode())
    wall = time.perf_counter() - t0
    return data, {"wall_s": wall}


def post_stream(base: str, key: str, body: dict, timeout: float) -> tuple[dict, dict]:
    """Streaming request. Returns (final usage/timings dict, {ttft_ms, wall_s, text, reasoning})."""
    payload = dict(body, stream=True)
    payload.pop("stream_options", None)
    req = urllib.request.Request(
        f"{base.rstrip('/')}/v1/chat/completions",
        data=json.dumps(payload).encode(),
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
        method="POST",
    )
    t0 = time.perf_counter()
    ttft_ms: float | None = None
    final: dict = {}
    text_parts: list[str] = []
    reasoning_parts: list[str] = []
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        for raw in resp:
            line = raw.decode("utf-8", errors="replace").strip()
            if not line.startswith("data:"):
                continue
            chunk = line[len("data:"):].strip()
            if chunk == "[DONE]":
                break
            try:
                obj = json.loads(chunk)
            except json.JSONDecodeError:
                continue
            if obj.get("usage") is not None:
                final = obj
            for choice in obj.get("choices") or []:
                delta = choice.get("delta") or {}
                content = delta.get("content") or ""
                reasoning = delta.get("reasoning_content") or delta.get("reasoning") or ""
                if content:
                    text_parts.append(content)
                if reasoning:
                    reasoning_parts.append(reasoning)
                if (content or reasoning) and ttft_ms is None:
                    ttft_ms = (time.perf_counter() - t0) * 1000.0
    wall = time.perf_counter() - t0
    return final, {"ttft_ms": ttft_ms, "wall_s": wall, "text": "".join(text_parts),
                   "reasoning": "".join(reasoning_parts)}


def num(*values) -> float | None:
    for v in values:
        if isinstance(v, (int, float)):
            return float(v)
    return None


def extract_metrics(resp: dict, extra: dict, streamed: bool) -> dict:
    usage = resp.get("usage") or {}
    timings = resp.get("timings") or {}
    draft_n = int(num(usage.get("draft_n"), timings.get("draft_n")) or 0)
    accepted = int(num(usage.get("draft_n_accepted"), timings.get("draft_n_accepted")) or 0)
    completion = int(num(usage.get("completion_tokens")) or 0)
    m = {
        "prompt_tokens": int(num(usage.get("prompt_tokens"), usage.get("prompt_n")) or 0),
        "completion_tokens": completion,
        "prefill_tok_s": num(usage.get("prompt_per_second"), timings.get("prompt_per_second")),
        "prefill_tail_tok_s": num(usage.get("prefill_tail_tok_s"), timings.get("prefill_tail_tok_s")),
        "prefill_tail_window_s": num(usage.get("prefill_tail_window_s"),
                                     timings.get("prefill_tail_window_s")),
        "decode_tok_s": num(usage.get("predicted_per_second"), timings.get("predicted_per_second")),
        "prefill_ms": num(usage.get("prompt_ms"), timings.get("prompt_ms")),
        "decode_ms": num(usage.get("predicted_ms"), timings.get("predicted_ms")),
        "draft_n": draft_n,
        "draft_n_accepted": accepted,
        "accept_rate": (accepted / draft_n) if draft_n > 0 else None,
        "stream": streamed,
    }
    decode_tok_s = m["decode_tok_s"]
    # Accept-rate-invariant kernel speed: each MTP verify round emits exactly one
    # non-drafted token plus the accepted drafts, so rounds = completion - accepted.
    # (cross-check: rounds * drafts_per_round ~= draft_n). Use this to compare
    # kernel changes; raw decode_tok_s conflates kernel speed with accept rate.
    m["verify_rounds_s"] = (
        (completion - accepted) * decode_tok_s / completion
        if decode_tok_s and completion > accepted
        else None
    )
    m.update(extra)
    if m.get("ttft_ms") is None and not streamed:
        # Server-side proxy for TTFT: first token leaves after prefill.
        m["ttft_ms"] = m.get("prefill_ms")
    text = extra.get("text", "")
    reasoning = extra.get("reasoning", "")
    if not streamed:
        choice = (resp.get("choices") or [{}])[0]
        msg = choice.get("message") or {}
        text = msg.get("content") or ""
        reasoning = msg.get("reasoning_content") or msg.get("reasoning") or ""
        m["finish_reason"] = choice.get("finish_reason")
    m["output_sha"] = hashlib.sha256(
        (reasoning + "\x00" + text).encode("utf-8")
    ).hexdigest()[:16]
    return m


AGG_METRICS = ("prefill_tok_s", "prefill_tail_tok_s", "decode_tok_s", "verify_rounds_s", "ttft_ms", "wall_s")


def aggregate(records: list[dict]) -> dict:
    agg = {
        "runs": len(records),
        "prompt_tokens": records[0].get("prompt_tokens"),
        "completion_tokens": records[0].get("completion_tokens"),
        "output_sha": records[-1].get("output_sha"),
        "accept_rate_mean": statistics.fmean(
            r["accept_rate"] for r in records if r.get("accept_rate") is not None
        ) if any(r.get("accept_rate") is not None for r in records) else None,
    }
    for name in AGG_METRICS:
        values = [r[name] for r in records if r.get(name) is not None]
        if values:
            agg[f"{name}_mean"] = statistics.fmean(values)
            agg[f"{name}_median"] = statistics.median(values)
            agg[f"{name}_min"] = min(values)
            agg[f"{name}_max"] = max(values)
    return agg


def fmt(x: float | None, digits: int = 1) -> str:
    return "—" if x is None else f"{x:.{digits}f}"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--base", default="http://127.0.0.1:8081")
    ap.add_argument("--model", default=None, help="served model id (default: first from /v1/models)")
    ap.add_argument("--api-key", default=None, help="override bearer key (else NINFER_API_KEY/.env)")
    ap.add_argument("--cases", type=Path, default=ROOT / "tools" / "bench" / "speed_suite_cases.json")
    ap.add_argument("--suite", default=None,
                    help="comma list: smoke,small,multiturn,tools,context,decode")
    ap.add_argument("--include-slow", action="store_true")
    ap.add_argument("--runs", type=int, default=2, help="measured trials per case (in-session)")
    ap.add_argument("--warmup", type=int, default=1)
    ap.add_argument("--stream", action=argparse.BooleanOptionalAction, default=True,
                    help="stream for client-side TTFT (default on)")
    ap.add_argument("--cache-bust", action=argparse.BooleanOptionalAction, default=True,
                    help="inject a per-request nonce so prefix reuse cannot shortcut prefill "
                         "(default on; --no-cache-bust measures production reuse-on behavior)")
    ap.add_argument("--label", default="")
    ap.add_argument("--out-dir", type=Path, default=ROOT / "profiles" / "bench" / "speed-suite")
    ap.add_argument("--timeout", type=float, default=900.0)
    args = ap.parse_args()

    key = load_key(args.api_key)
    if not key:
        print("missing API key (set NINFER_API_KEY or LLAMA_CPP_LOCAL_API_KEY, "
              "or pass --api-key)", file=sys.stderr)
        return 2
    model = args.model
    if not model:
        req = urllib.request.Request(
            f"{args.base.rstrip('/')}/v1/models",
            headers={"Authorization": f"Bearer {key}"},
        )
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = json.loads(resp.read().decode())
        ids = [d.get("id") for d in data.get("data", [])]
        if not ids:
            print("server reports no models", file=sys.stderr)
            return 2
        model = ids[0]

    cases_doc = json.loads(args.cases.read_text(encoding="utf-8"))
    selected = []
    suites = args.suite.split(",") if args.suite else None
    for case in cases_doc["cases"]:
        if case.get("slow") and not args.include_slow:
            continue
        if suites and case.get("suite") not in suites:
            continue
        selected.append(case)
    if not selected:
        print("no cases selected", file=sys.stderr)
        return 2

    def run_case(case: dict, index: int) -> dict:
        body = case_body(case, model, args.stream)
        if args.cache_bust:
            cache_bust(body, f"{case['id']}-r{index}-{os.getpid()}")
        if args.stream:
            final, extra = post_stream(args.base, key, body, args.timeout)
            return extract_metrics(final, extra, streamed=True)
        resp, extra = post(args.base, key, body, args.timeout)
        return extract_metrics(resp, extra, streamed=False)

    label = args.label or time.strftime("%Y%m%d-%H%M%S")
    print(f"model={model} base={args.base} cases={len(selected)} stream={args.stream}")
    report = {
        "artifact_type": "ninfer_speed_suite",
        "schema_version": 1,
        "label": label,
        "base": args.base,
        "model": model,
        "stream": args.stream,
        "cache_bust": args.cache_bust,
        "cases_file": str(args.cases),
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "cases": [],
    }
    for case in selected:
        record = {
            "id": case["id"],
            "suite": case.get("suite"),
            "desc": case.get("desc"),
            "thinking": bool(case.get("thinking", False)),
            "max_tokens": case.get("max_tokens"),
            "runs": [],
        }
        for i in range(args.warmup + args.runs):
            m = run_case(case, i)
            m["case"] = case["id"]
            m["warmup"] = i < args.warmup
            if not m["warmup"]:
                record["runs"].append(m)
            print(
                f"  [{case['id']} {('warmup' if m['warmup'] else 'run' + str(i - args.warmup + 1))}] "
                f"prompt={m['prompt_tokens']} gen={m['completion_tokens']} "
                f"prefill={fmt(m['prefill_tok_s'])} tok/s tail={fmt(m.get('prefill_tail_tok_s'))} tok/s "
                f"ttft={fmt(m.get('ttft_ms'))} ms "
                f"decode={fmt(m['decode_tok_s'])} tok/s wall={fmt(m['wall_s'], 2)} s"
            )
        if not record["runs"]:
            print(f"  {case['id']}: no measured runs", file=sys.stderr)
            report["cases"].append(record)
            continue
        record["aggregate"] = aggregate(record["runs"])
        report["cases"].append(record)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.out_dir / f"speed-suite-{label}.json"
    json_path.write_text(json.dumps(report, indent=1) + "\n")

    csv_path = args.out_dir / f"speed-suite-{label}.csv"
    fields = [
        "case", "suite", "desc", "thinking", "max_tokens", "runs",
        "prompt_tokens", "completion_tokens",
        "prefill_tok_s_mean", "prefill_tok_s_median", "prefill_tok_s_min",
        "prefill_tail_tok_s_mean", "prefill_tail_tok_s_median", "prefill_tail_tok_s_min",
        "prefill_tail_tok_s_max", "prefill_tail_window_s_mean",
        "ttft_ms_mean", "ttft_ms_median", "ttft_ms_min", "ttft_ms_max",
        "decode_tok_s_mean", "decode_tok_s_median", "decode_tok_s_min", "decode_tok_s_max",
        "verify_rounds_s_mean", "verify_rounds_s_median", "verify_rounds_s_min", "verify_rounds_s_max",
        "wall_s_mean", "wall_s_median", "wall_s_min", "wall_s_max",
        "accept_rate_mean", "output_sha",
    ]
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for record in report["cases"]:
            agg = record.get("aggregate") or {}
            row = {k: agg.get(k) for k in AGG_METRICS}
            for name in AGG_METRICS:
                for suffix in ("_mean", "_median", "_min", "_max"):
                    row[f"{name}{suffix}"] = agg.get(f"{name}{suffix}")
            row.update({
                "case": record["id"], "suite": record.get("suite"), "desc": record.get("desc"),
                "thinking": record.get("thinking"), "max_tokens": record.get("max_tokens"),
                "runs": len(record.get("runs") or []),
                "prompt_tokens": agg.get("prompt_tokens"),
                "completion_tokens": agg.get("completion_tokens"),
                "accept_rate_mean": agg.get("accept_rate_mean"),
                "output_sha": agg.get("output_sha"),
            })
            writer.writerow(row)

    print(f"\n{'case':<16} {'suite':<10} {'think':<5} {'prompt':>7} {'gen':>6} "
          f"{'pre-t/s':>8} {'tail-t/s':>9} {'ttft-ms':>8} {'dec-t/s':>8} {'rounds/s':>8} {'wall-s':>7} {'acc%':>6}")
    for record in report["cases"]:
        agg = record.get("aggregate")
        if not agg:
            continue
        acc = agg.get("accept_rate_mean")
        print(
            f"{record['id']:<16} {str(record.get('suite')):<10} "
            f"{'on' if record.get('thinking') else 'off':<5} "
            f"{str(agg.get('prompt_tokens')):>7} {str(agg.get('completion_tokens')):>6} "
            f"{fmt(agg.get('prefill_tok_s_mean')):>8} {fmt(agg.get('prefill_tail_tok_s_mean')):>9} "
            f"{fmt(agg.get('ttft_ms_mean')):>8} "
            f"{fmt(agg.get('decode_tok_s_mean')):>8} {fmt(agg.get('verify_rounds_s_mean')):>8} {fmt(agg.get('wall_s_mean'), 2):>7} "
            f"{('' if acc is None else f'{100*acc:.0f}'):>6}"
        )
    print(f"\njson: {json_path}\ncsv:  {csv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())