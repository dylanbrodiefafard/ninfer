#!/usr/bin/env python3
"""Bake a unique (non-tiled) token-id corpus for tools/ppl/run.py.

Encodes with ninfer-ppl --encode so ids match the tokenizer inside the artifact.
Prefers WikiText-2; does not tile.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import tempfile
import urllib.request
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
DEFAULT_WEIGHTS = (
    "/ssdpool2nvme/local_llm/models/qwen3.8-nvfp4-Osfralla-mtp-ninfer/"
    "qwen3_8_27b_nvfp4.ninfer"
)
DEFAULT_TOKENS = 8192
LONG_TOKENS = 32768
WIKITEXT_URLS = (
    "https://raw.githubusercontent.com/pytorch/examples/main/word_language_model/data/wikitext-2/test.txt",
    "https://raw.githubusercontent.com/pytorch/examples/main/word_language_model/data/wikitext-2/valid.txt",
)


def default_ppl_bin() -> Path:
    env = os.environ.get("NINFER_PPL")
    if env:
        return Path(env)
    return REPO / "build" / "apps" / "ninfer-ppl"


def load_wikitext() -> str | None:
    try:
        from datasets import load_dataset
    except ImportError:
        load_dataset = None
    if load_dataset is not None:
        try:
            splits = [
                load_dataset("wikitext", "wikitext-2-raw-v1", split="test"),
                load_dataset("wikitext", "wikitext-2-raw-v1", split="validation"),
            ]
            parts = [
                str(row.get("text") or "").strip()
                for split in splits
                for row in split
            ]
            text = "\n\n".join(part for part in parts if part)
            if text:
                return text
        except Exception:
            pass
    chunks: list[str] = []
    for url in WIKITEXT_URLS:
        try:
            with urllib.request.urlopen(url, timeout=60) as response:
                chunks.append(response.read().decode("utf-8"))
        except Exception:
            return None
    text = "\n\n".join(chunk.strip() for chunk in chunks if chunk.strip())
    return text or None


def load_source_text(paths: list[str]) -> str:
    chunks: list[str] = []
    for raw in paths:
        path = Path(raw).expanduser()
        text = path.read_text(encoding="utf-8")
        if not text.strip():
            raise SystemExit(f"source text is empty: {path}")
        chunks.append(text)
    return "\n\n".join(chunks)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", type=Path, default=Path(DEFAULT_WEIGHTS))
    parser.add_argument("--tokens", type=int, default=DEFAULT_TOKENS)
    parser.add_argument("--long", action="store_true", help=f"bake {LONG_TOKENS} tokens")
    parser.add_argument("--source-text", action="append", default=[])
    parser.add_argument("--ppl-bin", type=Path, default=default_ppl_bin())
    parser.add_argument("--out-dir", type=Path, default=Path(__file__).resolve().parent)
    args = parser.parse_args()
    tokens = LONG_TOKENS if args.long else args.tokens
    if tokens < 2:
        raise SystemExit("--tokens must be at least 2")
    if not args.weights.is_file():
        raise SystemExit(f"artifact not found: {args.weights}")
    if not args.ppl_bin.is_file():
        raise SystemExit(f"ninfer-ppl not found: {args.ppl_bin}")

    source_kind = "wikitext-2"
    if args.source_text:
        text = load_source_text(args.source_text)
        source_kind = "source-text"
    else:
        text = load_wikitext()
        if text is None:
            raise SystemExit(
                "could not load WikiText-2; pass --source-text or retry with network access"
            )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    ids_path = args.out_dir / "corpus.ids"
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", suffix=".txt", delete=False) as handle:
        handle.write(text)
        text_path = Path(handle.name)
    try:
        subprocess.run(
            [
                str(args.ppl_bin),
                "--encode",
                "--weights",
                str(args.weights),
                "--text",
                str(text_path),
                "--ids",
                str(ids_path),
                "--tokens",
                str(tokens),
            ],
            check=True,
        )
    finally:
        text_path.unlink(missing_ok=True)

    payload = ids_path.read_text(encoding="utf-8")
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    token_count = len(payload.split())
    if token_count < tokens:
        raise SystemExit(f"source produced {token_count} tokens, below target {tokens}")
    (args.out_dir / "corpus.manifest.json").write_text(
        json.dumps(
            {
                "artifact_type": "ninfer_ppl_corpus",
                "schema_version": 1,
                "weights": str(args.weights),
                "add_special_tokens": False,
                "chat_template": False,
                "tokens": token_count,
                "ids_sha256": digest,
                "source": source_kind,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"wrote {ids_path} tokens={token_count} sha256={digest}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
