"""Acquire a bounded real-token first-block panel from the pinned native checkpoint.

This is an input-distribution witness, not calibration data or a full-model quality test.
The tokenizer, token embeddings and hash-selected FP8 PLE rows all share one source pin.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import struct
import urllib.request

from tokenizers import Tokenizer

from tools.artifact.container import ArtifactIdentity, ArtifactWriter, TensorSpec
from tools.parity.qwen4.native_source import BASE, REPOSITORY, REVISION, read_header, read_range
from tools.parity.qwen4.native_ple_fixture import SHARD_ROWS
from tools.reference.qwen4.ngram import NGramConfig, ids, layer_multipliers, layout


def bounded_file(name: str, limit: int) -> bytes:
    with urllib.request.urlopen(BASE + name, timeout=120) as response:
        data = response.read(limit + 1)
    if len(data) > limit:
        raise ValueError(f"source file exceeds bounded acquisition: {name}")
    return data


def build(output: Path, component_report: Path) -> None:
    if output.exists() or output.with_suffix(".json").exists():
        raise FileExistsError(output)
    config = NGramConfig()
    constants = json.loads(component_report.read_text())["exact_source_i64_buffers"]
    prefix = "model.language_model.layers.1.ple.ple_embedding."
    table = layout(config)
    for name, expected in (("layer_multipliers", layer_multipliers(config)),
                           ("ngram_heads_vocab_sizes", table.head_vocab_sizes),
                           ("ngram_heads_offsets", table.head_offsets)):
        if constants[prefix + name] != list(expected):
            raise ValueError(f"independent hash constants disagree with source: {name}")
    tokenizer = Tokenizer.from_str(bounded_file("tokenizer.json", 32 * 1024 * 1024).decode())
    texts = ["Explain why the sky appears blue during the day and red at sunset.",
             "def square(x):\n    return x * x\nWhat is square(7)?"]
    tokens = (tokenizer.encode(texts[0], add_special_tokens=False).ids + [config.eos_token_id]
              + tokenizer.encode(texts[1], add_special_tokens=False).ids)[:33]
    if len(tokens) != 33 or any(not 0 <= token < config.unigram_vocab_size for token in tokens):
        raise ValueError("pinned token panel changed")
    rows, _ = ids(tokens, config=config)
    global_rows = rows.reshape(-1).tolist()
    unique_rows = sorted(set(global_rows))
    lookup = {row: index for index, row in enumerate(unique_rows)}
    local_rows = [lookup[row] for row in global_rows]
    source_index = json.loads(bounded_file("model.safetensors.index.json", 128 * 1024 * 1024))
    embedding_name = "model.language_model.embed_tokens.weight"
    embedding_shard = source_index["weight_map"][embedding_name]
    embedding_base, embedding_header = read_header(embedding_shard)
    embedding = embedding_header[embedding_name]
    if embedding["dtype"] != "BF16" or embedding["shape"] != [248320, 2560]:
        raise ValueError("source token embedding representation changed")
    ple_shard = "model-fp8-mtp-ple.safetensors"
    ple_base, ple_header = read_header(ple_shard)
    ple_prefix = prefix + "ngram_embedding."

    def embedding_row(token: int) -> bytes:
        begin = embedding_base + embedding["data_offsets"][0] + token * 5120
        return read_range(BASE + embedding_shard, begin, begin + 5119)

    def ple_row(row: int) -> bytes:
        part, local = divmod(row, SHARD_ROWS)
        item = ple_header[ple_prefix + f"shard_{part}.weight"]
        if item["dtype"] != "F8_E4M3" or item["shape"] != [SHARD_ROWS, 160]:
            raise ValueError("source PLE representation changed")
        begin = ple_base + item["data_offsets"][0] + local * 160
        return read_range(BASE + ple_shard, begin, begin + 159)

    with ThreadPoolExecutor(max_workers=8) as pool:
        token_bytes = b"".join(pool.map(embedding_row, tokens))
        print(f"acquired {len(tokens)} source token embedding rows", flush=True)
        ple_bytes = b"".join(pool.map(ple_row, unique_rows))
    scalar = ple_header[ple_prefix + "weight_scale"]
    if scalar["dtype"] != "BF16" or scalar["shape"] != [1]:
        raise ValueError("source PLE scale representation changed")
    begin, end = scalar["data_offsets"]
    scale = read_range(BASE + ple_shard, ple_base + begin, ple_base + end - 1)
    if scale != b"\x51\x39" or any(code & 0x7f == 0x7f for code in ple_bytes):
        raise ValueError("source PLE scale or finite-code audit failed")
    specs = [TensorSpec("token.embeddings", (len(tokens), 2560), "BF16", "contiguous-le-v1"),
             TensorSpec("ple.rows", (len(unique_rows), 160), "FP8_E4M3FN_TENSOR_BF16S", "tensor-scale-v1"),
             TensorSpec("token.ids", (len(tokens),), "I32", "contiguous-le-v1"),
             TensorSpec("ple.global_rows", (len(tokens), 16), "I32", "contiguous-le-v1"),
             TensorSpec("ple.local_rows", (len(tokens), 16), "I32", "contiguous-le-v1")]
    output.parent.mkdir(parents=True, exist_ok=True)
    with ArtifactWriter(output, ArtifactIdentity("qwen4/native-text-qualification", "nvidia-source-33"), specs) as writer:
        writer.write("token.embeddings", token_bytes)
        writer.write("ple.rows", ple_bytes + scale)
        for name, values in (("token.ids", tokens), ("ple.global_rows", global_rows), ("ple.local_rows", local_rows)):
            writer.write(name, struct.pack(f"<{len(values)}i", *values))
    with output.with_suffix(".json").open("x") as stream:
        json.dump(dict(repository=REPOSITORY, revision=REVISION, texts=texts, token_ids=tokens,
                       truncated_to=33, unique_ple_rows=unique_rows,
                       source_payload_bytes=len(token_bytes) + len(ple_bytes) + 2,
                       boundary="real token embeddings and exact hashed FP8 rows; no calibration or PPL"), stream, indent=2)
        stream.write("\n")
    print(f"wrote {output}: {len(tokens)} tokens, {len(unique_rows)} unique PLE rows", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--component-report", type=Path, required=True)
    args = parser.parse_args()
    build(args.out, args.component_report)
