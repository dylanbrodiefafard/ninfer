#!/usr/bin/env python3
"""Sequential, local-only Phase C verifier qualification cells (not Engine::score)."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
ARTIFACT = "/models/qwen3.8-nvfp4-flash2-nvfp4-bf16codebook-from-bf16/qwen3_8_27b_nvfp4_dflash_nvfp4.ninfer"
BINARIES = {
    "A": "ninfer-verify-score-a16-phase-c",
    "B": "ninfer-verify-score-a4-projections-phase-c",
    "C": "ninfer-verify-score-a4-fp32-gdn",
    "D": "ninfer-verify-score-a4-bf16-gdn",
}


def record_success(path: Path, entry: dict) -> None:
    previous = json.loads(path.read_text()) if path.exists() else []
    entries = {(item["profile"], item["cell"]): item for item in previous}
    entries[(entry["profile"], entry["cell"])] = entry
    path.write_text(json.dumps(list(entries.values()), indent=2) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("matrix", "cumulative", "checkpoints", "rounding", "fixtures", "mixed", "documents"))
    parser.add_argument("--documents", type=Path, help="JSON list of fixed {name, domain, path} reference documents")
    parser.add_argument("--profiles", nargs="+", choices=BINARIES, default=["A", "B", "C"])
    parser.add_argument("--container", default="ninfer-builder-dylan")
    parser.add_argument("--cells", nargs="+", help="run only these explicit cell names")
    parser.add_argument("--out", type=Path, default=ROOT / "profiles/bench/dflash-a4-qualification")
    args = parser.parse_args()
    args.out = args.out.resolve()
    args.out.mkdir(parents=True, exist_ok=True)
    corpus = "/src/tools/ppl/corpus.ids"
    cells = []
    document_offsets = {}
    if args.stage == "documents":
        if args.documents is None:
            parser.error("documents requires --documents")
        documents = json.loads(args.documents.read_text())
        if len({doc["name"] for doc in documents}) != len(documents):
            parser.error("document names must be unique")
        for doc in documents:
            name = doc["name"]
            if not name or any(c not in "abcdefghijklmnopqrstuvwxyz0123456789-_" for c in name):
                parser.error("document names must contain only lowercase letters, digits, - or _")
            source_path = Path(doc["path"]).resolve()
            local = args.out / f"document-{name}.txt"
            local.write_text(source_path.read_text(encoding="utf-8"), encoding="utf-8")
            source = "/src/" + str(local.relative_to(ROOT))
            cells.append((f"document-{name}", 1, 5, 8, 8192, source, False))
        (args.out / "documents.json").write_text(json.dumps(documents, indent=2) + "\n")
    elif args.stage == "matrix":
        cells = [(f"matrix-c{c}-w{w}", c, w, 8, 256, corpus, True)
                 for c in range(1, 5) for w in range(2, 7)]
    elif args.stage == "cumulative":
        cells = [(f"cumulative-c{c}", c, 5, 8, 32739, corpus, True) for c in (1, 4)]
    elif args.stage == "checkpoints":
        # One block from common prefill state: short, 8K, near 32K, and the three
        # largest positive C-A outliers observed in the initial 8192-token smoke.
        cells = [(f"common-p{p}", 1, 5, p, 5, corpus, True)
                 for p in (8, 8192, 32500, 4393, 7118, 4224, 21038, 13004, 31348)]
    elif args.stage == "rounding":
        cells = [("rounding-c1", 1, 5, 8, 8192, corpus, True)]
    elif args.stage == "mixed":
        if args.profiles != ["C"]:
            parser.error("mixed-history checks use --profiles C")
        for width in (5, 6):
            for offsets in ([0], [512], [1024], [1536], [0, 512, 1024, 1536], [1536, 1024, 512, 0]):
                name = f"mixed-w{width}-" + "-".join(map(str, offsets))
                document_offsets[name] = offsets
                cells.append((name, len(offsets), width, 8, 128, corpus, True))
    else:
        # Use each existing curated paragraph ONCE. Never use the tiled 65536-token
        # benchmark corpus as evidence for held-out NLL or an effective sample size.
        from tools.bench.make_bench_corpus import PARAGRAPHS
        groups = {"chinese": [0, 1, 2, 3, 4, 15, 16], "prose": [5, 6, 7, 8, 9, 17],
                  "code": [10, 11, 12], "math": [13, 14]}
        for domain, indices in groups.items():
            path = args.out / f"fixture-{domain}.txt"
            path.write_text("\n\n".join(PARAGRAPHS[i] for i in indices), encoding="utf-8")
            source = "/src/" + str(path.relative_to(ROOT))
            cells.append((f"fixture-{domain}", 1, 5, 8, 2048, source, False))
    if args.cells:
        cells = [cell for cell in cells if cell[0] in args.cells]
        if {cell[0] for cell in cells} != set(args.cells):
            parser.error("unknown cell name for stage")
    manifest_path = args.out / f"phase-c-{args.stage}-{'-'.join(args.profiles)}.json"
    for profile in args.profiles:
        for name, batch, width, prefix, limit, source, ids in cells:
            stem = args.out / f"phase-c-{profile.lower()}-{name}"
            binary = "/build/tests/" + BINARIES[profile]
            if args.stage == "mixed":
                binary += "-mixed"
            command = ["docker", "exec", "-e", f"NINFER_QWEN3_8_27B_NVFP4_DFLASH_WEIGHTS={ARTIFACT}"]
            if name in document_offsets:
                command += ["-e", "NINFER_VERIFY_DOCUMENT_OFFSETS=" + ",".join(map(str, document_offsets[name]))]
            if args.stage in ("checkpoints", "cumulative", "rounding"):
                dump = "/src/" + str(stem.with_suffix(".logits").relative_to(ROOT))
                command += ["-e", f"NINFER_VERIFY_LOGITS_DUMP={dump}"]
                if args.stage != "checkpoints":
                    command += ["-e", "NINFER_VERIFY_LOGIT_POSITIONS=9,1564,2267,4225,4394,7119,8199,16389,24579,32744"]
            command += [args.container, binary, str(batch), str(width), str(prefix), str(limit), source]
            if ids:
                command.append("--ids")
            print("RUN", " ".join(command), flush=True)
            with stem.with_suffix(".tsv").open("w") as scores, stem.with_suffix(".log").open("w") as log:
                subprocess.run(command, stdout=scores, stderr=log, check=True)
            print(stem.with_suffix(".log").read_text().strip(), flush=True)
            record_success(manifest_path, {"profile": profile, "cell": name, "command": command,
                                           "replicated_lanes": batch > 1 and name not in document_offsets})


if __name__ == "__main__":
    main()
