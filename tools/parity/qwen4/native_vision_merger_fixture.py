"""Acquire six exact native BF16 Vision-merger tensors, not a Vision checkpoint."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import urllib.request

from tools.artifact.container import ArtifactIdentity, ArtifactWriter, TensorSpec
from tools.parity.qwen4.native_source import BASE, REPOSITORY, REVISION, read_header, read_range

PREFIX = "model.visual.merger."
SHAPES = {
    "norm.weight": (1152,), "norm.bias": (1152,),
    "linear_fc1.weight": (4608, 4608), "linear_fc1.bias": (4608,),
    "linear_fc2.weight": (2560, 4608), "linear_fc2.bias": (2560,),
}


def build(output: Path) -> None:
    if output.exists() or output.with_suffix(".json").exists():
        raise FileExistsError(output)
    with urllib.request.urlopen(BASE + "model.safetensors.index.json", timeout=120) as response:
        index = json.load(response)["weight_map"]
    headers, tensors, inventory = {}, {}, {}
    for role, shape in SHAPES.items():
        name = PREFIX + role
        shard = index[name]
        if shard not in headers:
            headers[shard] = read_header(shard)
        data, header = headers[shard]
        item = header[name]
        begin, end = item["data_offsets"]
        if item["dtype"] != "BF16" or item["shape"] != list(shape) or end-begin != math.prod(shape)*2:
            raise ValueError("pinned Vision merger inventory mismatch")
        payload = bytearray()
        for start in range(begin, end, 8*1024*1024):
            stop = min(end, start+8*1024*1024)
            payload.extend(read_range(BASE+shard, data+start, data+stop-1))
        tensors[name] = payload
        inventory[name] = dict(shard=shard, dtype="BF16", shape=shape, payload_bytes=len(payload))
    output.parent.mkdir(parents=True, exist_ok=True)
    with ArtifactWriter(output, ArtifactIdentity("qwen4/native-vision-merger-qualification",
                                                 "nvidia-bf16-source"), [
            TensorSpec(PREFIX+role, shape, "BF16", "contiguous-le-v1")
            for role, shape in SHAPES.items()]) as writer:
        for name, payload in tensors.items():
            writer.write(name, payload)
    report = dict(repository=REPOSITORY, revision=REVISION, tensors=inventory,
                  payload_bytes=sum(map(len, tensors.values())),
                  limitation="Exact merger only; no Vision tower, image-input, or full-model qualification")
    with output.with_suffix(".json").open("x") as stream:
        json.dump(report, stream, indent=2); stream.write("\n")
    print(f"Acquired {report['payload_bytes']} source bytes for six BF16 Vision merger tensors")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    build(parser.parse_args().out)
