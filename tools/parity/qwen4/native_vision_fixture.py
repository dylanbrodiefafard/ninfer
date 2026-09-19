"""Acquire the pinned complete BF16 Qwen4 Vision tower, never a text checkpoint."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import urllib.request

from tools.artifact.container import ArtifactIdentity, ArtifactWriter, TensorSpec
from tools.parity.qwen4.native_source import BASE, REPOSITORY, REVISION, read_header, read_range
from tools.parity.qwen4.native_vision_merger_fixture import SHAPES as MERGER_SHAPES


def inventory() -> dict[str, tuple[int, ...]]:
    shapes = {"patch_embed.proj.weight": (1152, 3, 2, 16, 16),
              "patch_embed.proj.bias": (1152,), "pos_embed.weight": (2304, 1152)}
    for layer in range(27):
        prefix = f"blocks.{layer}."
        for name in ("norm1", "norm2"):
            shapes[prefix + name + ".weight"] = (1152,)
            shapes[prefix + name + ".bias"] = (1152,)
        for name, n, k in (("attn.qkv", 3456, 1152), ("attn.proj", 1152, 1152),
                           ("mlp.linear_fc1", 4304, 1152), ("mlp.linear_fc2", 1152, 4304)):
            shapes[prefix + name + ".weight"] = (n, k)
            shapes[prefix + name + ".bias"] = (n,)
    shapes.update({"merger." + key: value for key, value in MERGER_SHAPES.items()})
    return {"model.visual." + key: value for key, value in shapes.items()}


def build(output: Path) -> None:
    if output.exists() or output.with_suffix(".json").exists():
        raise FileExistsError(output)
    with urllib.request.urlopen(BASE + "model.safetensors.index.json", timeout=120) as response:
        index = json.load(response)["weight_map"]
    shapes = inventory()
    if {name for name in index if name.startswith("model.visual.")} != shapes.keys():
        raise ValueError("pinned complete Vision inventory differs")
    headers = {shard: read_header(shard) for shard in {index[name] for name in shapes}}
    sources = {}
    for name, shape in shapes.items():
        shard = index[name]
        base, header = headers[shard]
        item = header[name]
        begin, end = item["data_offsets"]
        if item["dtype"] != "BF16" or item["shape"] != list(shape) or end-begin != math.prod(shape)*2:
            raise ValueError(f"invalid pinned Vision tensor: {name}")
        sources[name] = dict(shard=shard, begin=base+begin, end=base+end,
                             shape=shape, dtype="BF16", payload_bytes=end-begin)
    output.parent.mkdir(parents=True, exist_ok=True)
    with ArtifactWriter(output, ArtifactIdentity("qwen4/native-vision-qualification", "nvidia-bf16-source"),
                        [TensorSpec(name, shape, "BF16", "contiguous-le-v1")
                         for name, shape in shapes.items()]) as writer:
        completed = 0
        for name, source in sources.items():
            def chunks():
                for begin in range(source["begin"], source["end"], 8*1024*1024):
                    yield read_range(BASE + source["shard"], begin,
                                     min(source["end"], begin+8*1024*1024)-1)
            writer.write(name, chunks())
            completed += source["payload_bytes"]
            print(f"Vision: {completed} bytes ({name})", flush=True)
    report = dict(repository=REPOSITORY, revision=REVISION, tensors=sources,
                  payload_bytes=sum(s["payload_bytes"] for s in sources.values()),
                  scope="Exact complete 27-block BF16 Vision tower, patch, position and merger; no text weights")
    with output.with_suffix(".json").open("x") as stream:
        json.dump(report, stream, indent=2)
        stream.write("\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    build(parser.parse_args().out)
