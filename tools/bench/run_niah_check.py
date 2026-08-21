#!/usr/bin/env python3
"""NIAH (needle-in-a-haystack) recall check against the live NInfer serve.

This is the practical quality gate for approximate (sage / nvfp4) attention:
a broken long-context attention path loses the needle. The script hits the
*already-running* server (no engine restart / reconfig) and checks whether the
exact needle string is retrieved in the model's answer to a long-context
question. It reports per-run recall (e.g. 3/3) rather than a speed metric.

Stdlib only; never restarts or reconfigures the engine. Fixtures resolve
relative to the repo root (auto-detected) unless given as an absolute path.

    python3 tools/bench/run_niah_check.py --label SAGE --runs 3
    python3 tools/bench/run_niah_check.py --fixture examples/cli/messages/long_niah_64k.json \
        --needle "ORCHID=493817; COLOR=COBALT" --max-tokens 64 --label SAGE

Outputs a per-run JSON record under profiles/bench/niah-check/ (gitignored).
"""
from __future__ import annotations

import argparse
import json
import os
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
DEFAULT_BASE = os.environ.get("NINFER_BASE", "http://127.0.0.1:8081")
DEFAULT_MODEL = "coding"
DEFAULT_NEEDLE = "ORCHID=493817; COLOR=COBALT"
DEFAULT_FIXTURES = [
    ("context_8k", "examples/cli/messages/long_niah_8k.json"),
    ("context_64k", "examples/cli/messages/long_niah_64k.json"),
]


def load_key(api_key: str | None) -> str:
    if api_key:
        return api_key
    for name in ("NINFER_API_KEY", "LLAMA_CPP_LOCAL_API_KEY"):
        if os.environ.get(name):
            return os.environ[name]
    for env in (
        Path("/home/dylan/docker/.env"),
        ROOT.parent.parent / ".env",
        Path.cwd() / ".env",
    ):
        if env.exists():
            for line in env.read_text().splitlines():
                for prefix in ("LLAMA_CPP_REMOTE_API_KEY=", "LLAMA_CPP_LOCAL_API_KEY="):
                    if line.startswith(prefix):
                        return line.split("=", 1)[1].strip().strip('"')
    return ""


def resolve_fixture(ref: str) -> Path:
    p = Path(ref)
    if p.is_absolute():
        return p
    for base in (ROOT, ROOT / "tools" / "bench" / "fixtures"):
        candidate = base / p
        if candidate.is_file():
            return candidate
    return p


def load_messages(fixture: str) -> list:
    raw = json.loads(resolve_fixture(fixture).read_text(encoding="utf-8"))
    if isinstance(raw, list):
        return raw
    return raw.get("messages", [])


def post(base: str, key: str, body: dict, timeout: float) -> dict:
    req = urllib.request.Request(
        f"{base.rstrip('/')}/v1/chat/completions",
        data=json.dumps(body).encode(),
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode())


def content_of(data: dict) -> str:
    parts = []
    for choice in data.get("choices") or []:
        msg = choice.get("message") or {}
        if msg.get("content"):
            parts.append(msg["content"])
        if msg.get("reasoning_content"):
            parts.append(msg["reasoning_content"])
    return "\n".join(parts)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--base", default=DEFAULT_BASE)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--key", default=None, help="API key (else read from env/.env)")
    ap.add_argument("--needle", default=DEFAULT_NEEDLE,
                    help="substring that must appear in the answer for a PASS")
    ap.add_argument("--fixture", action="append", default=None,
                    help="fixture ref (repeatable). Default: 8k + 64k NIAH.")
    ap.add_argument("--max-tokens", type=int, default=64)
    ap.add_argument("--thinking", action="store_true", default=False,
                    help="enable thinking (off by default so the answer is the needle)")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--runs", type=int, default=1, help="repeat each case N times")
    ap.add_argument("--timeout", type=float, default=600.0)
    ap.add_argument("--label", default="niah")
    args = ap.parse_args()

    key = load_key(args.key)
    if not key:
        print("ERROR: no API key (pass --key or set NINFER_API_KEY / LLAMA_CPP_LOCAL_API_KEY)",
              file=__import__("sys").stderr)
        return 2

    cases = []
    if args.fixture:
        for ref in args.fixture:
            label = Path(ref).stem
            cases.append((label, ref))
    else:
        cases = DEFAULT_FIXTURES

    out_dir = ROOT / "profiles" / "bench" / "niah-check"
    out_dir.mkdir(parents=True, exist_ok=True)
    record = {
        "label": args.label,
        "base": args.base,
        "model": args.model,
        "needle": args.needle,
        "thinking": args.thinking,
        "runs": args.runs,
        "ts": int(time.time()),
        "cases": [],
    }

    all_pass = True
    for label, ref in cases:
        messages = load_messages(ref)
        n_prompt_chars = sum(len(m.get("content", "")) for m in messages)
        retrieved = 0
        total = 0
        snippets = []
        for run in range(args.runs):
            body = {
                "model": args.model,
                "messages": [dict(m) for m in messages],
                "max_tokens": args.max_tokens,
                "temperature": 0.0,
                "chat_template_kwargs": {"enable_thinking": args.thinking},
                "enable_thinking": args.thinking,
            }
            if args.seed is not None:
                body["seed"] = args.seed
            t0 = time.perf_counter()
            try:
                data = post(args.base, key, body, args.timeout)
            except Exception as exc:  # noqa: BLE001 - report and continue
                total += 1
                snippets.append(f"RUN ERROR: {exc}")
                print(f"  [{label}] run {run + 1}/{args.runs}: ERROR {exc}")
                continue
            wall = time.perf_counter() - t0
            text = content_of(data)
            total += 1
            ok = args.needle in text
            retrieved += 1 if ok else 0
            status = "PASS" if ok else "FAIL"
            all_pass = all_pass and ok
            preview = text.strip().replace("\n", " ")
            if len(preview) > 240:
                preview = preview[:240] + "..."
            snippets.append(f"{status}: {preview}")
            print(f"  [{label}] run {run + 1}/{args.runs}: {status} "
                  f"(wall {wall:.1f}s, needle={ok})\n      {preview}")
        passed = retrieved == total and total > 0
        record["cases"].append({
            "label": label,
            "fixture": str(ref),
            "prompt_chars": n_prompt_chars,
            "retrieved": retrieved,
            "total": total,
            "recall": (retrieved / total) if total else 0.0,
            "passed": passed,
            "snippets": snippets,
        })

    rec_path = out_dir / f"niah-check-{args.label}-{record['ts']}.json"
    rec_path.write_text(json.dumps(record, indent=2), encoding="utf-8")
    print(f"\nrecord -> {rec_path.relative_to(ROOT)}")
    print("NIAH gate:", "PASS" if all_pass else "FAIL",
          "(all cases must retrieve the needle in every run)")
    return 0 if all_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())