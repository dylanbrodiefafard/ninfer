"""Acquire explicit disjoint real-text panels for resident Qwen4 precision studies."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from tools.parity.qwen4.native_source import REPOSITORY, REVISION
from tools.parity.qwen4.native_text_fixture import build


def run(specification: Path, source: Path, ple: Path, output: Path) -> None:
    specification = specification.resolve()
    entries = json.loads(specification.read_text())["panels"]
    names, documents = set(), set()
    for entry in entries:
        name = entry["id"]
        if (name in names or not name.replace("_", "").isalnum()
                or entry["split"] not in ("calibration", "heldout")
                or not 136 <= entry["tokens"] <= 4096):
            raise ValueError("invalid corpus identity, split or width")
        names.add(name)
        for document in entry["texts"]:
            if document in documents:
                raise ValueError("overlapping documents across corpus panels")
            documents.add(document)
    if {entry["split"] for entry in entries} != {"calibration", "heldout"}:
        raise ValueError("both independent splits are required")
    output.mkdir(parents=True, exist_ok=True)
    manifest = output / "manifest.json"
    if manifest.exists():
        raise FileExistsError(manifest)
    panels = []
    for entry in entries:
        path = (output / (entry["id"] + ".ninfer")).resolve()
        if path.exists():
            # Resume complete explicit panels only after their source and input identity match.
            previous = json.loads(path.with_suffix(".json").read_text())
            if any(previous[key] != value for key, value in {
                "repository": REPOSITORY, "revision": REVISION, "texts": entry["texts"],
                "truncated_to": entry["tokens"], "split": entry["split"],
            }.items()):
                raise ValueError("existing panel belongs to different inputs")
        else:
            build(path, source / "qwen4-ple-component.json", texts=entry["texts"],
                  width=entry["tokens"], split=entry["split"], ple_table=ple)
        panels.append({key: entry[key] for key in ("id", "split", "tokens")} | {"panel": str(path)})
    # Tokenization can map different source strings to the same represented document.
    sequences = [tuple(json.loads(Path(panel["panel"]).with_suffix(".json").read_text())["token_ids"])
                 for panel in panels]
    if len(set(sequences)) != len(sequences):
        raise ValueError("duplicate represented token documents")
    manifest.write_text(json.dumps({"repository": REPOSITORY, "revision": REVISION,
        "specification": str(specification), "panels": panels,
        "scope": "Disjoint authored documents, source embeddings and exact n-gram rows; bounded component calibration/holdout, not model PPL."}, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--specification", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--ple", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    run(args.specification, args.source, args.ple, args.out)
