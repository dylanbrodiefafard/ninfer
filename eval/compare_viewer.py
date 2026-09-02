#!/usr/bin/env python3
"""Live production vs p-less AIME transcript viewer.

Reads eval JSONL in place (no regenerate step). Incomplete last lines and
jobs that have not started yet are skipped. Open http://127.0.0.1:8765
"""

from __future__ import annotations

import argparse
import json
import re
import signal
import sys
import threading
import traceback
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

DISCONNECT = (BrokenPipeError, ConnectionResetError, ConnectionAbortedError, TimeoutError)


JOB_TEMP = {
    "aime25_t06": ("aime25", 0.6),
    "aime26_t06": ("aime26", 0.6),
    "aime25_t10": ("aime25", 1.0),
    "aime26_t10": ("aime26", 1.0),
    "aime25_t15": ("aime25", 1.5),
    "aime26_t15": ("aime26", 1.5),
    "aime25_t20": ("aime25", 2.0),
    "aime26_t20": ("aime26", 2.0),
}

SAFE_ID = re.compile(r"^[A-Za-z0-9._-]+$")
HERE = Path(__file__).resolve().parent
HTML_PATH = HERE / "compare_viewer.html"


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def remap_src_path(raw: str, repo: Path) -> Path:
    text = raw.strip()
    if text.startswith("/src/"):
        return repo / text[len("/src/") :]
    return Path(text)


def jsonl_paths(job_dir: Path, kind: str) -> list[Path]:
    root = job_dir / kind
    if not root.is_dir():
        return []
    return sorted(path for path in root.rglob("*.jsonl") if path.is_file())


def content_parts(content: Any) -> tuple[str, str]:
    if isinstance(content, str):
        return "", content
    if not isinstance(content, list):
        return "", ""
    reasoning = ""
    visible = ""
    for part in content:
        if not isinstance(part, dict):
            continue
        kind = part.get("type")
        if kind == "reasoning":
            reasoning = str(part.get("reasoning") or part.get("text") or "")
        elif kind in ("text", "output_text"):
            visible = str(part.get("text") or "")
        elif kind in ("thinking", "thought"):
            reasoning = str(part.get("thinking") or part.get("text") or reasoning)
    return reasoning, visible


def score_fields(record: dict[str, Any]) -> tuple[float | None, str, str]:
    sample = record.get("sample_score")
    if not isinstance(sample, dict):
        return None, "", str(record.get("target") or record.get("gold") or "")
    score = sample.get("score")
    gold = str(record.get("target") or record.get("gold") or "")
    if not isinstance(score, dict):
        return None, "", gold
    extracted = str(score.get("extracted_prediction") or "")
    value = score.get("value")
    acc: float | None = None
    if isinstance(value, dict) and isinstance(value.get("acc"), (int, float)):
        acc = float(value["acc"])
    elif isinstance(score.get("acc"), (int, float)):
        acc = float(score["acc"])
    return acc, extracted, gold


def usage_fields(record: dict[str, Any]) -> tuple[int | None, int | None, str]:
    output_tokens = None
    reasoning_tokens = None
    stop = ""
    for message in record.get("messages") or []:
        if not isinstance(message, dict):
            continue
        perf = message.get("perf_metrics") or {}
        if message.get("role") == "assistant" and isinstance(perf, dict):
            if isinstance(perf.get("output_tokens"), int):
                output_tokens = perf["output_tokens"]
    model_output = record.get("model_output")
    if isinstance(model_output, dict):
        usage = model_output.get("usage") or {}
        if isinstance(usage, dict):
            if isinstance(usage.get("output_tokens"), int):
                output_tokens = usage["output_tokens"]
            if isinstance(usage.get("reasoning_tokens"), int):
                reasoning_tokens = usage["reasoning_tokens"]
        choices = model_output.get("choices") or []
        if choices and isinstance(choices[0], dict):
            stop = str(choices[0].get("stop_reason") or choices[0].get("finish_reason") or "")
    return output_tokens, reasoning_tokens, stop


def parse_record(record: dict[str, Any], *, include_text: bool = False) -> dict[str, Any]:
    question = ""
    reasoning = ""
    visible = ""
    for message in record.get("messages") or []:
        if not isinstance(message, dict):
            continue
        role = message.get("role")
        content = message.get("content")
        if role == "user":
            question = content if isinstance(content, str) else question
        elif role == "assistant":
            reasoning, visible = content_parts(content)
    if not reasoning and not visible:
        model_output = record.get("model_output")
        if isinstance(model_output, dict):
            choices = model_output.get("choices") or []
            if choices and isinstance(choices[0], dict):
                reasoning, visible = content_parts((choices[0].get("message") or {}).get("content"))
    acc, extracted, gold = score_fields(record)
    output_tokens, reasoning_tokens, stop = usage_fields(record)
    index = record.get("index")
    parsed = {
        "index": int(index) if isinstance(index, int) else index,
        "gold": gold,
        "acc": acc,
        "extracted": extracted,
        "output_tokens": output_tokens,
        "reasoning_tokens": reasoning_tokens,
        "stop_reason": stop,
        "reasoning_chars": len(reasoning),
        "visible_chars": len(visible),
    }
    if include_text:
        parsed["question"] = question
        parsed["reasoning"] = reasoning
        parsed["visible"] = visible
    return parsed


class JsonlCache:
    def __init__(self) -> None:
        self._state: dict[str, dict[str, Any]] = {}
        self._lock = threading.Lock()

    def records(self, path: Path) -> list[dict[str, Any]]:
        key = str(path)
        stat = path.stat()
        with self._lock:
            entry = self._state.get(key)
            if (
                entry is not None
                and entry["mtime"] == stat.st_mtime
                and entry["size"] == stat.st_size
            ):
                return entry["records"]
            records: list[dict[str, Any]] = []
            start = 0
            if (
                entry is not None
                and stat.st_size > entry["size"]
                and entry["mtime"] <= stat.st_mtime
            ):
                records = list(entry["records"])
                start = entry["size"]
            with path.open("rb") as handle:
                handle.seek(start)
                consumed = start
                while True:
                    offset = handle.tell()
                    line = handle.readline()
                    if not line:
                        consumed = offset
                        break
                    if not line.endswith(b"\n"):
                        consumed = offset
                        break
                    consumed = handle.tell()
                    stripped = line.strip()
                    if not stripped:
                        continue
                    try:
                        payload = json.loads(stripped)
                    except (json.JSONDecodeError, UnicodeDecodeError):
                        continue
                    if isinstance(payload, dict):
                        parsed = parse_record(payload, include_text=False)
                        parsed["offset"] = offset
                        parsed["path"] = key
                        records.append(parsed)
            self._state[key] = {
                "mtime": stat.st_mtime,
                "size": consumed,
                "records": records,
            }
            return records


def group_samples(records: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        key = str(record.get("index"))
        grouped.setdefault(key, []).append(record)
    for samples in grouped.values():
        for sample_i, record in enumerate(samples):
            record["sample"] = sample_i
    return grouped


def merge_job_records(job_dir: Path, cache: JsonlCache) -> dict[str, list[dict[str, Any]]]:
    reviews: list[dict[str, Any]] = []
    predictions: list[dict[str, Any]] = []
    for path in jsonl_paths(job_dir, "reviews"):
        reviews.extend(cache.records(path))
    for path in jsonl_paths(job_dir, "predictions"):
        predictions.extend(cache.records(path))
    if not reviews and not predictions:
        return {}
    if not reviews:
        return group_samples(predictions)
    primary = group_samples(reviews)
    if not predictions:
        return primary
    pred_groups = group_samples(predictions)
    for index, extras in pred_groups.items():
        if index not in primary:
            primary[index] = extras
            continue
        samples = primary[index]
        for i, sample in enumerate(samples):
            if i >= len(extras):
                break
            extra = extras[i]
            if sample.get("output_tokens") is None:
                sample["output_tokens"] = extra.get("output_tokens")
            if sample.get("reasoning_tokens") is None:
                sample["reasoning_tokens"] = extra.get("reasoning_tokens")
            if not sample.get("stop_reason"):
                sample["stop_reason"] = extra.get("stop_reason")
            if not sample.get("question") and extra.get("question"):
                sample["question"] = extra["question"]
    return primary


def summarize_sample(record: dict[str, Any]) -> dict[str, Any]:
    return {
        "sample": record.get("sample", 0),
        "acc": record.get("acc"),
        "extracted": record.get("extracted") or "",
        "gold": record.get("gold") or "",
        "output_tokens": record.get("output_tokens"),
        "reasoning_tokens": record.get("reasoning_tokens"),
        "stop_reason": record.get("stop_reason") or "",
        "reasoning_chars": record.get("reasoning_chars") or 0,
        "visible_chars": record.get("visible_chars") or 0,
    }


def job_status(run_dir: Path, job_id: str) -> dict[str, Any]:
    state_path = run_dir / "state.json"
    status = "unknown"
    if state_path.exists():
        state = load_json(state_path)
        status = str(((state.get("jobs") or {}).get(job_id) or {}).get("status") or state.get("status") or "unknown")
    progress_path = run_dir / "backends" / job_id / "progress.json"
    processed = None
    planned = None
    if progress_path.exists():
        progress = load_json(progress_path)
        processed = progress.get("processed_count")
        planned = progress.get("total_count")
        if progress.get("status"):
            status = str(progress["status"])
    result_path = run_dir / "backends" / job_id / "job-result.json"
    metrics: dict[str, Any] = {}
    if result_path.exists():
        result = load_json(result_path)
        status = str(result.get("status") or status)
        metrics = result.get("metrics") or {}
        counts = result.get("counts") or {}
        processed = counts.get("completed", processed)
        planned = counts.get("planned", planned)
    dataset, temp = JOB_TEMP.get(job_id, (job_id, None))
    return {
        "status": status,
        "processed": processed,
        "planned": planned,
        "metrics": metrics,
        "dataset": dataset,
        "temperature": temp,
    }


def discover_runs(runs_dir: Path, logs_dir: Path, repo: Path) -> list[dict[str, Any]]:
    found: list[dict[str, Any]] = []
    if not runs_dir.is_dir():
        return found
    for path in sorted(runs_dir.iterdir()):
        if not path.is_dir() or not (path / "manifest.json").exists():
            continue
        if not (path / "backends").is_dir():
            continue
        manifest = load_json(path / "manifest.json")
        method = None
        method_path = path / "method.json"
        if method_path.exists():
            method = (load_json(method_path) or {}).get("method")
        state = load_json(path / "state.json") if (path / "state.json").exists() else {}
        found.append(
            {
                "id": path.name,
                "path": str(path),
                "suite": manifest.get("suite"),
                "created_at": manifest.get("created_at"),
                "method": method,
                "status": state.get("status"),
            }
        )
    pointers = {}
    for name in ("production", "p-less"):
        pointer = logs_dir / f"{name}.run_dir"
        if not pointer.exists():
            continue
        mapped = remap_src_path(pointer.read_text(encoding="utf-8"), repo)
        if mapped.exists():
            pointers[name] = mapped.name
    for run in found:
        if run["method"]:
            continue
        for name, run_id in pointers.items():
            if run["id"] == run_id:
                run["method"] = name
    aime = [run for run in found if run.get("suite") == "aime_temp"]
    aime.sort(key=lambda item: item.get("created_at") or item["id"])
    tagged = {run["method"]: run["id"] for run in aime if run.get("method")}
    if "production" not in tagged and aime:
        aime[0]["method"] = aime[0]["method"] or "production"
        tagged["production"] = aime[0]["id"]
    if "p-less" not in tagged:
        for run in reversed(aime):
            if run["id"] != tagged.get("production"):
                run["method"] = run["method"] or "p-less"
                break
    return found


def default_pair(runs: list[dict[str, Any]]) -> tuple[str | None, str | None]:
    production = next((run["id"] for run in runs if run.get("method") == "production"), None)
    pless = next((run["id"] for run in runs if run.get("method") == "p-less"), None)
    aime = [run for run in runs if run.get("suite") == "aime_temp"]
    aime.sort(key=lambda item: item.get("created_at") or item["id"])
    if production is None and aime:
        production = aime[0]["id"]
    if pless is None:
        for run in reversed(aime):
            if run["id"] != production:
                pless = run["id"]
                break
    return production, pless


def side_payload(run_dir: Path, job_id: str, cache: JsonlCache) -> dict[str, Any]:
    info = job_status(run_dir, job_id)
    grouped = merge_job_records(run_dir / "backends" / job_id, cache)
    items = []
    for index in sorted(grouped, key=lambda value: int(value) if str(value).isdigit() else value):
        samples = grouped[index]
        gold = next((sample.get("gold") for sample in samples if sample.get("gold")), "")
        items.append(
            {
                "index": int(index) if str(index).isdigit() else index,
                "gold": gold,
                "samples": [summarize_sample(sample) for sample in samples],
            }
        )
    info["items"] = items
    info["n"] = sum(len(item["samples"]) for item in items)
    scored = [
        sample
        for item in items
        for sample in item["samples"]
        if sample.get("acc") is not None
    ]
    info["correct"] = sum(1 for sample in scored if sample["acc"])
    info["scored"] = len(scored)
    return info


def read_full_record(record: dict[str, Any]) -> dict[str, Any] | None:
    path = Path(str(record.get("path") or ""))
    offset = record.get("offset")
    if not path.is_file() or not isinstance(offset, int):
        return None
    with path.open("rb") as handle:
        handle.seek(offset)
        line = handle.readline()
    if not line.strip():
        return None
    try:
        payload = json.loads(line)
    except json.JSONDecodeError:
        return None
    if not isinstance(payload, dict):
        return None
    parsed = parse_record(payload, include_text=True)
    for key in (
        "output_tokens",
        "reasoning_tokens",
        "stop_reason",
        "acc",
        "extracted",
        "gold",
    ):
        if parsed.get(key) in (None, "") and record.get(key) not in (None, ""):
            parsed[key] = record.get(key)
    return parsed


def full_sample(run_dir: Path, job_id: str, index: str, sample: int, cache: JsonlCache) -> dict[str, Any] | None:
    grouped = merge_job_records(run_dir / "backends" / job_id, cache)
    records = grouped.get(str(index)) or grouped.get(index)
    if not records or sample < 0 or sample >= len(records):
        return None
    parsed = read_full_record(records[sample])
    if parsed is None:
        return None
    parsed["sample"] = sample
    return parsed


def resolve_run(runs_dir: Path, run_id: str) -> Path:
    if not SAFE_ID.match(run_id):
        raise ValueError("invalid run id")
    path = (runs_dir / run_id).resolve()
    if path.parent != runs_dir.resolve() or not path.is_dir():
        raise ValueError("unknown run")
    return path


class ViewerApp:
    def __init__(self, repo: Path, runs_dir: Path, logs_dir: Path) -> None:
        self.repo = repo
        self.runs_dir = runs_dir
        self.logs_dir = logs_dir
        self.cache = JsonlCache()

    def state(self, production_id: str | None, pless_id: str | None) -> dict[str, Any]:
        runs = discover_runs(self.runs_dir, self.logs_dir, self.repo)
        default_prod, default_pless = default_pair(runs)
        production_id = production_id or default_prod
        pless_id = pless_id or default_pless
        jobs = []
        prod_dir = resolve_run(self.runs_dir, production_id) if production_id else None
        pless_dir = resolve_run(self.runs_dir, pless_id) if pless_id else None
        job_ids = list(JOB_TEMP)
        if prod_dir and (prod_dir / "state.json").exists():
            for job_id in (load_json(prod_dir / "state.json").get("jobs") or {}):
                if job_id not in job_ids:
                    job_ids.append(job_id)
        if pless_dir and (pless_dir / "state.json").exists():
            for job_id in (load_json(pless_dir / "state.json").get("jobs") or {}):
                if job_id not in job_ids:
                    job_ids.append(job_id)
        for job_id in job_ids:
            dataset, temp = JOB_TEMP.get(job_id, (job_id, None))
            jobs.append(
                {
                    "id": job_id,
                    "dataset": dataset,
                    "temperature": temp,
                    "production": side_payload(prod_dir, job_id, self.cache) if prod_dir else {"items": []},
                    "p_less": side_payload(pless_dir, job_id, self.cache) if pless_dir else {"items": []},
                }
            )
        return {
            "runs": runs,
            "production_id": production_id,
            "p_less_id": pless_id,
            "jobs": jobs,
        }


def send_json(handler: BaseHTTPRequestHandler, payload: Any, status: int = 200) -> None:
    try:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8", errors="replace")
    except (TypeError, ValueError, UnicodeEncodeError):
        body = json.dumps({"error": "failed to encode transcript"}, ensure_ascii=True).encode("utf-8")
        status = 500
    handler.send_response(status)
    handler.send_header("Content-Type", "application/json; charset=utf-8")
    handler.send_header("Content-Length", str(len(body)))
    handler.send_header("Cache-Control", "no-store")
    handler.send_header("Connection", "close")
    handler.end_headers()
    handler.wfile.write(body)


def send_bytes(handler: BaseHTTPRequestHandler, body: bytes, content_type: str, status: int = 200) -> None:
    handler.send_response(status)
    handler.send_header("Content-Type", content_type)
    handler.send_header("Content-Length", str(len(body)))
    handler.send_header("Cache-Control", "no-store")
    handler.send_header("Connection", "close")
    handler.end_headers()
    handler.wfile.write(body)


class ViewerServer(ThreadingHTTPServer):
    allow_reuse_address = True
    daemon_threads = True


def make_handler(app: ViewerApp) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        timeout = 120
        protocol_version = "HTTP/1.1"

        def handle_one_request(self) -> None:
            try:
                super().handle_one_request()
            except DISCONNECT:
                pass

        def do_GET(self) -> None:
            parsed = urlparse(self.path)
            query = parse_qs(parsed.query)
            try:
                if parsed.path in ("/", "/index.html"):
                    send_bytes(self, HTML_PATH.read_bytes(), "text/html; charset=utf-8")
                    return
                if parsed.path == "/api/state":
                    production = (query.get("production") or [None])[0]
                    pless = (query.get("p_less") or [None])[0]
                    send_json(self, app.state(production, pless))
                    return
                if parsed.path == "/api/sample":
                    run_id = (query.get("run") or [""])[0]
                    job_id = (query.get("job") or [""])[0]
                    index = (query.get("index") or [""])[0]
                    sample = int((query.get("sample") or ["0"])[0])
                    if not SAFE_ID.match(job_id):
                        raise ValueError("invalid job")
                    record = full_sample(
                        resolve_run(app.runs_dir, run_id),
                        job_id,
                        index,
                        sample,
                        app.cache,
                    )
                    if record is None:
                        send_json(self, {"error": "sample not available yet"}, 404)
                        return
                    send_json(self, record)
                    return
                send_json(self, {"error": "not found"}, 404)
            except ValueError as exc:
                send_json(self, {"error": str(exc)}, 400)
            except DISCONNECT:
                return
            except Exception:
                traceback.print_exc()
                try:
                    send_json(self, {"error": "internal error"}, 500)
                except DISCONNECT:
                    return

    return Handler


def main() -> int:
    try:
        signal.signal(signal.SIGPIPE, signal.SIG_IGN)
    except (AttributeError, ValueError):
        pass
    repo = HERE.parent
    parser = argparse.ArgumentParser(description="Compare production vs p-less AIME transcripts")
    parser.add_argument("--runs-dir", type=Path, default=HERE / "runs")
    parser.add_argument("--logs-dir", type=Path, default=HERE / "server-logs")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--open", action="store_true", help="open the viewer in a browser")
    args = parser.parse_args()
    if not HTML_PATH.exists():
        print(f"missing {HTML_PATH}", file=sys.stderr)
        return 1
    app = ViewerApp(repo, args.runs_dir.resolve(), args.logs_dir.resolve())
    handler = make_handler(app)
    server = ViewerServer((args.host, args.port), handler)
    url = f"http://{args.host}:{args.port}/"
    print(f"AIME compare viewer: {url}", flush=True)
    print("Reads eval/runs JSONL live. Refresh the page or click Refresh after new items finish.", flush=True)
    if args.open:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print()
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
