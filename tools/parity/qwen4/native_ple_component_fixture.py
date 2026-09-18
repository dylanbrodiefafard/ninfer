"""Acquire only native layer-1 PLE projections/controls; never the embedding table."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import struct
import urllib.request

from tools.artifact.container import ArtifactIdentity, ArtifactWriter, TensorSpec
from tools.parity.qwen4.native_source import BASE, REPOSITORY, REVISION, read_header, read_range

PREFIX = "model.language_model.layers.1.ple."
SHAPES = {
    "key_proj.weight": (10240, 2560),
    "value_proj.weight": (2560, 2560),
    "norm_key.weight": (10240,),
    "norm_query.weight": (10240,),
    "norm_conv.weight": (10240,),
    "conv1d.weight": (10240, 1, 4),
}
INTEGER_SHAPES = {"ple_embedding.layer_multipliers": (3,),
                  "ple_embedding.ngram_heads_vocab_sizes": (16,),
                  "ple_embedding.ngram_heads_offsets": (16,)}


def build(output: Path) -> None:
    if output.exists() or output.with_suffix(".json").exists():
        raise FileExistsError(output)
    with urllib.request.urlopen(BASE + "model.safetensors.index.json", timeout=120) as response:
        index = json.load(response)["weight_map"]
    headers, tensors, integers, inventory = {}, {}, {}, {}
    total = 0
    for name, shape in (SHAPES | INTEGER_SHAPES).items():
        full_name = PREFIX + name
        shard = index[full_name]
        if shard not in headers:
            headers[shard] = read_header(shard)
        data, header = headers[shard]
        item = header[full_name]
        dtype = "BF16" if name in SHAPES else "I64"
        count = 1
        for extent in shape:
            count *= extent
        begin, end = item["data_offsets"]
        if item["dtype"] != dtype or item["shape"] != list(shape) or end - begin != count * (2 if dtype == "BF16" else 8):
            raise ValueError("pinned PLE component inventory mismatch")
        payload = bytearray()
        for start in range(begin, end, 8 * 1024 * 1024):
            stop = min(end, start + 8 * 1024 * 1024)
            payload.extend(read_range(BASE + shard, data + start, data + stop - 1))
        total += len(payload)
        inventory[full_name] = dict(shard=shard, dtype=dtype, shape=shape, payload_bytes=len(payload))
        if dtype == "BF16":
            tensors[full_name] = payload
        else:
            integers[full_name] = struct.unpack(f"<{count}q", payload)
    if total != 65_679_640:
        raise ValueError("pinned PLE component byte count mismatch")
    output.parent.mkdir(parents=True, exist_ok=True)
    with ArtifactWriter(output, ArtifactIdentity("qwen4/native-ple-component-qualification",
                                                 "nvidia-bf16-source"), [
            TensorSpec(PREFIX + name, shape, "BF16", "contiguous-le-v1")
            for name, shape in SHAPES.items()]) as writer:
        for name, payload in tensors.items():
            writer.write(name, payload)
    report = dict(repository=REPOSITORY, revision=REVISION, payload_bytes=total,
                  tensors=inventory, exact_source_i64_buffers=integers,
                  limitation="PLE component only; no full-table residency, full layer, or model-quality claim")
    with output.with_suffix(".json").open("x") as stream:
        json.dump(report, stream, indent=2); stream.write("\n")
    print(f"Acquired {total} source bytes; six BF16 tensors plus three exact I64 metadata buffers")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    build(parser.parse_args().out)
