#!/usr/bin/env python3
"""Paired analysis of actual-verifier TSVs; no inference or corpus acquisition."""

from __future__ import annotations

import argparse
import array
import csv
import heapq
import json
import math
import random
import struct
import sys
from pathlib import Path


def read_scores(path: Path, replicated_lanes: bool = False) -> dict:
    rows = {}
    with path.open() as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
            key = (int(row["document"]), int(row["position"]))
            value = (int(row["token"]), float(row["nll"]))
            if key in rows or not math.isfinite(value[1]) or value[1] < 0:
                raise ValueError(f"invalid or duplicate score: {path}: {key}")
            rows[key] = value
    if not rows:
        raise ValueError(f"empty scores: {path}")
    if replicated_lanes:
        reference = {pos: value for (doc, pos), value in rows.items() if doc == 0}
        if not reference:
            raise ValueError("replicated lanes require lane zero")
        for doc in {doc for doc, _ in rows}:
            lane = {pos: value for (d, pos), value in rows.items() if d == doc}
            if lane != reference:
                raise ValueError(f"replicated lane {doc} differs from lane zero: {path}")
        rows = {(0, pos): value for pos, value in reference.items()}
    return rows


def quantile(values: list[float], q: float) -> float:
    ordered = sorted(values)
    index = (len(ordered) - 1) * q
    low = int(index)
    return ordered[low] + (ordered[min(low + 1, len(ordered) - 1)] - ordered[low]) * (index - low)


def compare(a: dict, b: dict, block_size: int = 1024) -> dict:
    if a.keys() != b.keys() or any(a[k][0] != b[k][0] for k in a):
        raise ValueError("paired positions or gold tokens differ")
    keys = sorted(a)
    deltas = [b[k][1] - a[k][1] for k in keys]
    blocks = []
    for doc in sorted({d for d, _ in keys}):
        doc_keys = [k for k in keys if k[0] == doc]
        for start in range(0, len(doc_keys), block_size):
            part = doc_keys[start:start + block_size]
            blocks.append((math.fsum(b[k][1] - a[k][1] for k in part), len(part)))
    rng = random.Random(20260923)
    bootstrap = []
    for _ in range(10000 if len(blocks) > 1 else 0):
        sample = rng.choices(blocks, k=len(blocks))
        bootstrap.append(math.fsum(v for v, _ in sample) / sum(n for _, n in sample))
    mean_a = math.fsum(a[k][1] for k in keys) / len(keys)
    mean_b = math.fsum(b[k][1] for k in keys) / len(keys)
    worst = sorted(keys, key=lambda k: b[k][1] - a[k][1], reverse=True)[:10]
    trends = []
    for doc in sorted({d for d, _ in keys}):
        doc_keys = [k for k in keys if k[0] == doc]
        for start in range(0, len(doc_keys), 1024):
            part = doc_keys[start:start + 1024]
            trends.append({"first": part[0], "last": part[-1], "tokens": len(part),
                           "delta_nll": math.fsum(b[k][1] - a[k][1] for k in part) / len(part)})
    return {
        "tokens": len(keys), "baseline_nll": mean_a, "candidate_nll": mean_b,
        "baseline_ppl": math.exp(mean_a), "candidate_ppl": math.exp(mean_b),
        "delta_nll": mean_b - mean_a, "ppl_ratio": math.exp(mean_b - mean_a),
        "block_size": block_size, "blocks": len(blocks),
        "block_bootstrap_95_ci": [quantile(bootstrap, .025), quantile(bootstrap, .975)]
            if len(blocks) > 1 else None,
        "uncertainty_scope": "conditional on this corpus; contiguous blocks, not independent tokens",
        "delta_p95": quantile(deltas, .95), "delta_p99": quantile(deltas, .99),
        "max_abs_delta": max(map(abs, deltas)),
        "changed_tokens": sum(x != 0 for x in deltas),
        "worst": [{"document": k[0], "position": k[1], "token": a[k][0],
                   "baseline_nll": a[k][1], "candidate_nll": b[k][1],
                   "delta_nll": b[k][1] - a[k][1]} for k in worst],
        "position_trends": trends,
    }


def read_logits(path: Path) -> dict:
    records = {}
    with path.open("rb") as handle:
        while header := handle.read(16):
            domain, lane, position, gold = struct.unpack("<4I", header)
            bits = array.array("H")
            bits.frombytes(handle.read(domain * 2))
            if len(bits) != domain or gold >= domain:
                raise ValueError(f"invalid logit record: {path}")
            if sys.byteorder != "little":
                bits.byteswap()
            words = array.array("I", (value << 16 for value in bits))
            values = array.array("f")
            values.frombytes(words.tobytes())
            key = (lane, position)
            if key in records or not all(map(math.isfinite, values)):
                raise ValueError(f"duplicate or nonfinite logits: {path}")
            records[key] = (gold, values)
    return records


def logit_comparison(a: Path, b: Path) -> list[dict]:
    left, right = read_logits(a), read_logits(b)
    if left.keys() != right.keys():
        raise ValueError("logit capture positions differ")
    result = []
    for key, (gold, la) in left.items():
        gb, lb = right[key]
        if gold != gb or len(la) != len(lb):
            raise ValueError("logit domains/gold tokens differ")
        ma, mb = max(la), max(lb)
        za = ma + math.log(math.fsum(math.exp(v - ma) for v in la))
        zb = mb + math.log(math.fsum(math.exp(v - mb) for v in lb))
        pa = [math.exp(v - za) for v in la]
        pb = [math.exp(v - zb) for v in lb]
        ta, tb = heapq.nlargest(2, range(len(la)), key=la.__getitem__), heapq.nlargest(2, range(len(lb)), key=lb.__getitem__)
        result.append({"lane": key[0], "position": key[1], "gold": gold,
                       "delta_nll": (zb - lb[gold]) - (za - la[gold]),
                       "kl_baseline_candidate": math.fsum(p * ((x - za) - (y - zb)) for p, x, y in zip(pa, la, lb)),
                       "total_variation": .5 * math.fsum(abs(p - q) for p, q in zip(pa, pb)),
                       "baseline_top": ta[0], "candidate_top": tb[0],
                       "baseline_margin": la[ta[0]] - la[ta[1]], "candidate_margin": lb[tb[0]] - lb[tb[1]],
                       "baseline_gold_probability": pa[gold], "candidate_gold_probability": pb[gold]})
    return result


def campaign_report(directory: Path) -> dict:
    def path(profile, cell, suffix="tsv"):
        if profile == "C" and cell == "rounding-c1" and suffix == "tsv":
            cell = "c1-w5-8k"
        return directory / f"phase-c-{profile.lower()}-{cell}.{suffix}"

    def paired(left, right, cell):
        a, b = path(left, cell), path(right, cell)
        if not a.exists() or not b.exists():
            return None
        result = compare(read_scores(a, True), read_scores(b, True))
        la, lb = path(left, cell, "logits"), path(right, cell, "logits")
        if la.exists() and lb.exists():
            result["logits"] = logit_comparison(la, lb)
        return result

    report = {"comparisons": {}, "geometry": {}, "matrix": {}}
    cells = ["c1-w5-8k", "c4-w5-8k", "cumulative-c1", "cumulative-c4", "rounding-c1"]
    cells += [f"common-p{p}" for p in (8, 8192, 32500, 4393, 7118, 4224, 21038, 13004, 31348)]
    cells += [f"fixture-{domain}" for domain in ("chinese", "prose", "code", "math")]
    for cell in cells:
        for left, right in (("A", "B"), ("B", "C"), ("A", "C"), ("C", "D")):
            if result := paired(left, right, cell):
                report["comparisons"][f"{cell}/{right}-{left}"] = result
    for profile in "ABC":
        for width in range(2, 7):
            for c in range(1, 5):
                cell = f"matrix-c{c}-w{width}"
                if path(profile, cell).exists():
                    data = read_scores(path(profile, cell), True)
                    reference = read_scores(path(profile, f"matrix-c1-w{width}"), True)
                    result = compare(reference, data)
                    report["geometry"][f"{profile}/{cell}"] = {
                        k: result[k] for k in ("tokens", "max_abs_delta", "changed_tokens")}
                if profile != "A" and (result := paired("A", profile, cell)):
                    report["matrix"][f"{profile}-A/{cell}"] = {
                        k: result[k] for k in ("tokens", "delta_nll", "ppl_ratio", "max_abs_delta", "changed_tokens")}
        for first, fourth in (("c1-w5-8k", "c4-w5-8k"), ("cumulative-c1", "cumulative-c4")):
            if path(profile, first).exists() and path(profile, fourth).exists():
                result = compare(read_scores(path(profile, first), True), read_scores(path(profile, fourth), True))
                report["geometry"][f"{profile}/{fourth}"] = {
                    k: result[k] for k in ("tokens", "max_abs_delta", "changed_tokens")}
        if path(profile, "c1-w5-8k").exists() and path(profile, "cumulative-c1").exists():
            smoke = read_scores(path(profile, "c1-w5-8k"), True)
            extended = read_scores(path(profile, "cumulative-c1"), True)
            result = compare(smoke, {k: extended[k] for k in smoke})
            report["geometry"][f"{profile}/smoke-versus-extended-prefix"] = {
                k: result[k] for k in ("tokens", "max_abs_delta", "changed_tokens")}
    expected = [(p, f"matrix-c{c}-w{w}") for p in "ABC" for c in range(1, 5) for w in range(2, 7)]
    expected += [(p, f"c{c}-w5-8k") for p in "ABC" for c in (1, 4)]
    expected += [(p, f"cumulative-c{c}") for p in "ABC" for c in (1, 4)]
    expected += [(p, f"common-p{n}") for p in "ABCD" for n in (8, 8192, 32500, 4393, 7118, 4224, 21038, 13004, 31348)]
    expected += [(p, f"fixture-{d}") for p in "ABC" for d in ("chinese", "prose", "code", "math")]
    expected += [("D", "rounding-c1")]
    for width in (5, 6):
        singles = [f"mixed-w{width}-{offset}" for offset in (0, 512, 1024, 1536)]
        packs = [f"mixed-w{width}-{order}" for order in ("0-512-1024-1536", "1536-1024-512-0")]
        expected += [("C", cell) for cell in singles + packs]
        if all(path("C", cell).exists() for cell in singles + packs):
            reference = {}
            for cell in singles:
                reference.update(read_scores(path("C", cell)))
            for cell in packs:
                result = compare(reference, read_scores(path("C", cell)))
                report["geometry"][f"C/{cell}"] = {
                    k: result[k] for k in ("tokens", "max_abs_delta", "changed_tokens")}
    report["missing_cells"] = [f"{p}/{c}" for p, c in expected if not path(p, c).exists()]
    captures = [(p, c) for p, c in expected if c.startswith(("common-", "cumulative-"))]
    captures.append(("D", "rounding-c1"))
    report["missing_captures"] = [f"{p}/{c}" for p, c in captures
                                  if not path(p, c, "logits").exists() or path(p, c, "logits").stat().st_size == 0]
    report["geometry_mismatches"] = {k: v for k, v in report["geometry"].items() if v["changed_tokens"]}
    return report


def document_report(directory: Path, block_size: int = 1024) -> dict:
    documents = json.loads((directory / "documents.json").read_text())
    groups = {}
    comparisons = {}
    for index, doc in enumerate(documents):
        scores = {}
        for profile in "ABC":
            path = directory / f"phase-c-{profile.lower()}-document-{doc['name']}.tsv"
            rows = read_scores(path)
            if {d for d, _ in rows} != {0}:
                raise ValueError("document campaign requires one lane per independent document")
            scores[profile] = {(index, position): value for (_, position), value in rows.items()}
        for left, right in (("A", "B"), ("B", "C"), ("A", "C")):
            comparisons[f"{doc['name']}/{right}-{left}"] = compare(scores[left], scores[right], block_size)
        group = groups.setdefault(doc["domain"], {profile: {} for profile in "ABC"})
        for profile in "ABC":
            group[profile].update(scores[profile])
    domains = {}
    for domain, scores in groups.items():
        for left, right in (("A", "B"), ("B", "C"), ("A", "C")):
            domains[f"{domain}/{right}-{left}"] = compare(scores[left], scores[right], block_size)
    return {"documents": documents, "comparisons": comparisons, "domains": domains}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("baseline", type=Path, nargs="?")
    parser.add_argument("candidate", type=Path, nargs="?")
    parser.add_argument("--campaign-dir", type=Path)
    parser.add_argument("--documents-dir", type=Path)
    parser.add_argument("--brief", action="store_true", help="compact campaign table instead of JSON")
    parser.add_argument("--logits", action="store_true", help="compare two binary logit captures")
    parser.add_argument("--replicated-lanes", action="store_true",
                        help="require exact lane replication, then count lane zero only")
    parser.add_argument("--block-size", type=int, default=1024)
    args = parser.parse_args()
    if args.block_size < 1:
        parser.error("--block-size must be positive")
    if args.documents_dir:
        result = document_report(args.documents_dir, args.block_size)
    elif args.campaign_dir:
        result = campaign_report(args.campaign_dir)
    else:
        if args.baseline is None or args.candidate is None:
            parser.error("provide baseline/candidate or --campaign-dir")
        result = ({"logits": logit_comparison(args.baseline, args.candidate)} if args.logits else
                  compare(read_scores(args.baseline, args.replicated_lanes),
                          read_scores(args.candidate, args.replicated_lanes), args.block_size))
    if args.brief and args.campaign_dir:
        print("cell/pair | tokens | baseline PPL | candidate PPL | delta NLL | ratio | block 95% CI")
        for name, value in result["comparisons"].items():
            print(f"{name} | {value['tokens']} | {value['baseline_ppl']:.7f} | {value['candidate_ppl']:.7f} | "
                  f"{value['delta_nll']:.7f} | {value['ppl_ratio']:.7f} | {value['block_bootstrap_95_ci']}")
        print("Geometry comparisons:", len(result["geometry"]), "mismatches:", result["geometry_mismatches"])
        print("Missing cells:", result["missing_cells"])
        print("Missing captures:", result["missing_captures"])
        extended = result["comparisons"].get("cumulative-c1/C-A")
        if extended:
            trends = extended["position_trends"]
            for start in range(0, len(trends), 8):
                band = trends[start:start + 8]
                delta = math.fsum(x["delta_nll"] * x["tokens"] for x in band) / sum(x["tokens"] for x in band)
                print("C-A context band:", band[0]["first"], "through", band[-1]["last"], "delta NLL:", delta)
        for name, value in result["comparisons"].items():
            if name.startswith("common-") and name.endswith("/C-A"):
                print(name, "first-token logits:", value["logits"][0] if value.get("logits") else "MISSING")
    else:
        print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
