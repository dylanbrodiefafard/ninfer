#!/usr/bin/env python3
"""Compare two sequential AIME25/AIME26 temperature-sweep eval runs."""

from __future__ import annotations

import argparse
import json
import math
import re
from collections import defaultdict
from pathlib import Path
from typing import Any


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

BOX_RE = re.compile(r"\\boxed\{([^{}]+)\}")
ANSWER_LINE_RE = re.compile(
    r"(?:final answer|the answer is|answer)\s*[:\s]*\$?\\?boxed\{?(-?\d{1,3})\}?\$?",
    re.IGNORECASE,
)
INT_RE = re.compile(r"-?\d{1,3}")


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def score_of(record: dict[str, Any]) -> float | None:
    for key in ("score", "is_correct", "correct"):
        if key not in record:
            continue
        value = record[key]
        if isinstance(value, bool):
            return 1.0 if value else 0.0
        if isinstance(value, (int, float)):
            return float(value)
    return None


def prediction_text(record: dict[str, Any]) -> str:
    for key in ("prediction", "output", "model_prediction"):
        value = record.get(key)
        if isinstance(value, str) and value.strip():
            return value
    output = record.get("model_output")
    if isinstance(output, dict):
        choices = output.get("choices") or []
        if choices:
            message = choices[0].get("message") or {}
            content = message.get("content")
            if isinstance(content, str):
                return content
            reasoning = message.get("reasoning_content")
            if isinstance(reasoning, str):
                return reasoning
        content = output.get("content")
        if isinstance(content, str):
            return content
    return ""


def gold_text(record: dict[str, Any]) -> str:
    for key in ("gold", "answer", "target", "label"):
        value = record.get(key)
        if value is None:
            continue
        if isinstance(value, (int, float)):
            return str(int(value)) if float(value).is_integer() else str(value)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def item_key(record: dict[str, Any], index: int) -> str:
    for key in ("index", "id", "idx", "question_id"):
        value = record.get(key)
        if value is not None:
            return str(value)
    question = record.get("question") or record.get("input") or record.get("prompt")
    if isinstance(question, str) and question.strip():
        return question.strip()[:240]
    return str(index)


def extract_answer(text: str) -> str:
    matches = BOX_RE.findall(text)
    if matches:
        token = matches[-1].strip()
        numbers = INT_RE.findall(token)
        return numbers[-1] if numbers else token
    match = ANSWER_LINE_RE.search(text[-4000:])
    if match:
        return match.group(1)
    numbers = INT_RE.findall(text[-400:])
    return numbers[-1] if numbers else ""


def load_predictions(run_dir: Path) -> dict[str, list[dict[str, Any]]]:
    out: dict[str, list[dict[str, Any]]] = {}
    backends = run_dir / "backends"
    if not backends.exists():
        return out
    for job_dir in sorted(path for path in backends.iterdir() if path.is_dir()):
        items: list[dict[str, Any]] = []
        for path in sorted(job_dir.glob("predictions/**/*.jsonl")):
            for line_i, line in enumerate(path.read_text(encoding="utf-8").splitlines()):
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if not isinstance(record, dict):
                    continue
                text = prediction_text(record)
                gold = gold_text(record)
                score = score_of(record)
                items.append(
                    {
                        "key": item_key(record, line_i),
                        "gold": gold,
                        "score": score,
                        "prediction": text,
                        "extracted": extract_answer(text),
                        "chars": len(text),
                    }
                )
        out[job_dir.name] = items
    return out


def request_stats(path: Path, expected_p_less: bool) -> dict[str, Any]:
    empty = {
        "n": 0,
        "completion_tokens": 0,
        "prompt_tokens": 0,
        "decode_seconds": 0.0,
        "total_seconds": 0.0,
        "mean_completion_tokens": None,
        "decode_tok_s": None,
        "e2e_tok_s": None,
        "finish_reasons": {},
        "request_starts": 0,
        "p_less_observed": 0,
    }
    if not path.exists():
        raise FileNotFoundError(f"request log does not exist: {path}")
    n = 0
    completion = 0
    prompt = 0
    decode_s = 0.0
    total_s = 0.0
    request_starts = 0
    p_less_observed = 0
    reasons: dict[str, int] = defaultdict(int)
    for line_i, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path}:{line_i}: invalid JSON: {exc}") from exc
        event = record.get("event") or record.get("type")
        if event == "request_start":
            request = record.get("request") or {}
            sampling = request.get("sampling") or {}
            observed = sampling.get("p_less")
            if expected_p_less:
                if observed is not True:
                    raise ValueError(
                        f"{path}:{line_i}: expected request_start sampling.p_less=true, "
                        f"got {observed!r}"
                    )
            elif observed not in (None, False):
                raise ValueError(
                    f"{path}:{line_i}: expected request_start sampling.p_less=false or absent, "
                    f"got {observed!r}"
                )
            request_starts += 1
            p_less_observed += int(observed is True)
            continue
        if event != "request_done":
            continue
        result = record.get("result") or {}
        timings = record.get("timings_seconds") or {}
        completion += int(result.get("completion_tokens") or 0)
        prompt += int(result.get("prompt_tokens") or 0)
        decode_s += float(timings.get("decode") or 0.0)
        total_s += float(timings.get("total") or 0.0)
        reason = str(result.get("finish_reason") or "unknown")
        reasons[reason] += 1
        n += 1
    if request_starts == 0:
        raise ValueError(f"{path}: no request_start records found")
    if n == 0:
        empty["request_starts"] = request_starts
        empty["p_less_observed"] = p_less_observed
        return empty
    return {
        "n": n,
        "completion_tokens": completion,
        "prompt_tokens": prompt,
        "decode_seconds": decode_s,
        "total_seconds": total_s,
        "mean_completion_tokens": completion / n,
        "decode_tok_s": (completion / decode_s) if decode_s > 0 else None,
        "e2e_tok_s": (completion / total_s) if total_s > 0 else None,
        "finish_reasons": dict(reasons),
        "request_starts": request_starts,
        "p_less_observed": p_less_observed,
    }


def summary_jobs(run_dir: Path) -> dict[str, dict[str, Any]]:
    payload = load_json(run_dir / "summary.json")
    return {item["job_id"]: item for item in payload.get("results", [])}


def fmt_pct(score: float | None, completed: int | None) -> str:
    if score is None:
        return "-"
    if completed:
        correct = int(round(score * completed))
        return f"{100.0 * score:.2f}% ({correct}/{completed})"
    return f"{100.0 * score:.2f}%"


def fmt_num(value: float | None, digits: int = 1) -> str:
    if value is None or (isinstance(value, float) and not math.isfinite(value)):
        return "-"
    return f"{value:.{digits}f}"


def snippet(text: str, limit: int = 900) -> str:
    stripped = text.strip()
    if len(stripped) <= limit:
        return stripped
    head = stripped[: limit // 3].rstrip()
    tail = stripped[-(limit - len(head) - 20) :].lstrip()
    return f"{head}\n...\n{tail}"


def render(
    production_dir: Path,
    pless_dir: Path,
    production_requests: Path,
    pless_requests: Path,
) -> str:
    prod_jobs = summary_jobs(production_dir)
    pless_jobs = summary_jobs(pless_dir)
    prod_pred = load_predictions(production_dir)
    pless_pred = load_predictions(pless_dir)
    prod_req = request_stats(production_requests, expected_p_less=False)
    pless_req = request_stats(pless_requests, expected_p_less=True)

    lines = [
        "# P-less vs production AIME temperature sweep",
        "",
        f"- Production run: `{production_dir}`",
        f"- P-less run: `{pless_dir}`",
        "",
        "Claim under test: p-less degrades less than the production sampler as "
        "temperature rises (not a guaranteed win at T=0.6).",
        "",
        "## Mechanism verification",
        "",
        f"- Production p_less observed: {prod_req['p_less_observed']}/"
        f"{prod_req['request_starts']} request starts (expected 0).",
        f"- P-less p_less observed: {pless_req['p_less_observed']}/"
        f"{pless_req['request_starts']} request starts.",
        "",
        "## Accuracy",
        "",
        "| Job | Dataset | T | Production | P-less | Δ (p-less − prod) |",
        "|---|---|---:|---|---|---:|",
    ]
    prod_correct = 0
    pless_correct = 0
    prod_n = 0
    pless_n = 0
    by_temp: dict[float, dict[str, list[float]]] = defaultdict(
        lambda: {"production": [], "p_less": []}
    )
    disagreements: list[dict[str, Any]] = []

    for job_id, (dataset, temp) in JOB_TEMP.items():
        left = prod_jobs.get(job_id, {})
        right = pless_jobs.get(job_id, {})
        left_score = (left.get("metrics") or {}).get("accuracy")
        right_score = (right.get("metrics") or {}).get("accuracy")
        left_n = (left.get("counts") or {}).get("completed")
        right_n = (right.get("counts") or {}).get("completed")
        if isinstance(left_score, (int, float)):
            by_temp[temp]["production"].append(float(left_score))
            if left_n:
                prod_correct += int(round(float(left_score) * int(left_n)))
                prod_n += int(left_n)
        if isinstance(right_score, (int, float)):
            by_temp[temp]["p_less"].append(float(right_score))
            if right_n:
                pless_correct += int(round(float(right_score) * int(right_n)))
                pless_n += int(right_n)
        delta = ""
        if isinstance(left_score, (int, float)) and isinstance(right_score, (int, float)):
            delta = f"{100.0 * (float(right_score) - float(left_score)):+.2f} pp"
        lines.append(
            f"| `{job_id}` | {dataset} | {temp:g} | "
            f"{fmt_pct(left_score, left_n)} | {fmt_pct(right_score, right_n)} | {delta or '-'} |"
        )

        left_items = {item["key"]: item for item in prod_pred.get(job_id, [])}
        right_items = {item["key"]: item for item in pless_pred.get(job_id, [])}
        for key in sorted(set(left_items) & set(right_items)):
            a = left_items[key]
            b = right_items[key]
            if a.get("score") is None or b.get("score") is None:
                continue
            if bool(a["score"]) == bool(b["score"]) and a.get("extracted") == b.get(
                "extracted"
            ):
                continue
            disagreements.append(
                {
                    "job_id": job_id,
                    "dataset": dataset,
                    "temp": temp,
                    "key": key,
                    "gold": a.get("gold") or b.get("gold"),
                    "prod_score": a["score"],
                    "pless_score": b["score"],
                    "prod_extract": a.get("extracted"),
                    "pless_extract": b.get("extracted"),
                    "prod_chars": a.get("chars"),
                    "pless_chars": b.get("chars"),
                    "prod_text": a.get("prediction") or "",
                    "pless_text": b.get("prediction") or "",
                }
            )

    lines.extend(
        [
            "",
            f"Overall production: {prod_correct}/{prod_n}" if prod_n else "",
            f"Overall p-less: {pless_correct}/{pless_n}" if pless_n else "",
            "",
            "## Temperature trend",
            "",
            "| T | Production mean acc | P-less mean acc | Δ |",
            "|---:|---:|---:|---:|",
        ]
    )
    for temp in sorted(by_temp):
        p = by_temp[temp]["production"]
        q = by_temp[temp]["p_less"]
        p_mean = sum(p) / len(p) if p else None
        q_mean = sum(q) / len(q) if q else None
        delta = (
            f"{100.0 * (q_mean - p_mean):+.2f} pp"
            if p_mean is not None and q_mean is not None
            else "-"
        )
        lines.append(
            f"| {temp:g} | {fmt_pct(p_mean, None)} | {fmt_pct(q_mean, None)} | {delta} |"
        )

    lines.extend(
        [
            "",
            "## Length and throughput (request log `request_done`)",
            "",
            "| Method | Requests | Mean completion tokens | Decode tok/s | E2E tok/s | Finish reasons |",
            "|---|---:|---:|---:|---:|---|",
            "| Production | {n} | {mean} | {dec} | {e2e} | {fr} |".format(
                n=prod_req["n"],
                mean=fmt_num(prod_req["mean_completion_tokens"]),
                dec=fmt_num(prod_req["decode_tok_s"]),
                e2e=fmt_num(prod_req["e2e_tok_s"]),
                fr=prod_req["finish_reasons"] or "-",
            ),
            "| P-less | {n} | {mean} | {dec} | {e2e} | {fr} |".format(
                n=pless_req["n"],
                mean=fmt_num(pless_req["mean_completion_tokens"]),
                dec=fmt_num(pless_req["decode_tok_s"]),
                e2e=fmt_num(pless_req["e2e_tok_s"]),
                fr=pless_req["finish_reasons"] or "-",
            ),
            "",
            "## Job duration",
            "",
            "| Job | Production seconds | P-less seconds |",
            "|---|---:|---:|",
        ]
    )
    for job_id in JOB_TEMP:
        left = prod_jobs.get(job_id, {})
        right = pless_jobs.get(job_id, {})
        lines.append(
            f"| `{job_id}` | {fmt_num(left.get('duration_seconds'))} | "
            f"{fmt_num(right.get('duration_seconds'))} |"
        )

    split = {"p_less_only": [], "prod_only": [], "both_wrong_diff": []}
    for item in disagreements:
        prod_ok = bool(item["prod_score"])
        pless_ok = bool(item["pless_score"])
        if pless_ok and not prod_ok:
            split["p_less_only"].append(item)
        elif prod_ok and not pless_ok:
            split["prod_only"].append(item)
        else:
            split["both_wrong_diff"].append(item)

    lines.extend(
        [
            "",
            "## Disagreements",
            "",
            f"- P-less correct / production wrong: {len(split['p_less_only'])}",
            f"- Production correct / p-less wrong: {len(split['prod_only'])}",
            f"- Both wrong or score-tied with different extracted answers: "
            f"{len(split['both_wrong_diff'])}",
            "",
        ]
    )

    def dump_cases(title: str, cases: list[dict[str, Any]], limit: int = 8) -> None:
        lines.append(f"### {title}")
        lines.append("")
        if not cases:
            lines.append("None.")
            lines.append("")
            return
        for item in cases[:limit]:
            lines.extend(
                [
                    f"**{item['job_id']}** item `{item['key']}` (T={item['temp']:g}, "
                    f"gold `{item['gold']}`)",
                    "",
                    f"- Production: score={item['prod_score']} extracted=`{item['prod_extract']}` "
                    f"chars={item['prod_chars']}",
                    f"- P-less: score={item['pless_score']} extracted=`{item['pless_extract']}` "
                    f"chars={item['pless_chars']}",
                    "",
                    "<details><summary>Production tail</summary>",
                    "",
                    "```",
                    snippet(item["prod_text"]),
                    "```",
                    "",
                    "</details>",
                    "",
                    "<details><summary>P-less tail</summary>",
                    "",
                    "```",
                    snippet(item["pless_text"]),
                    "```",
                    "",
                    "</details>",
                    "",
                ]
            )
        if len(cases) > limit:
            lines.append(f"... {len(cases) - limit} more")
            lines.append("")

    dump_cases("P-less recovered items", split["p_less_only"])
    dump_cases("Production-only correct items", split["prod_only"])
    dump_cases("Different wrong answers", split["both_wrong_diff"], limit=4)
    return "\n".join(line for line in lines if line is not None) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--production", required=True, type=Path)
    parser.add_argument("--p-less", required=True, type=Path)
    parser.add_argument("--production-requests", required=True, type=Path)
    parser.add_argument("--p-less-requests", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    text = render(
        args.production.resolve(),
        args.p_less.resolve(),
        args.production_requests.resolve(),
        args.p_less_requests.resolve(),
    )
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "summary.md").write_text(text, encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
