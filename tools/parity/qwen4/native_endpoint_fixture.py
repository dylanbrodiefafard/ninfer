"""Acquire exact pinned BF16 endpoints, MTP stem, or one Vision block, never full shards."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import urllib.request

from tools.artifact.container import ArtifactIdentity, ArtifactWriter, TensorSpec
from tools.parity.qwen4.native_source import BASE, REPOSITORY, REVISION, read_header, read_range

ENDPOINT = {
    "model.language_model.hyper_connection_mixer.hc_norm.weight": (10240,),
    "model.language_model.hyper_connection_mixer.input_mix_weight_down.weight": (320, 10240),
    "model.language_model.hyper_connection_mixer.input_mix_weight_up.weight": (10240, 320),
    "lm_head.weight": (248320, 2560),
}
MTP_STEM = {
    "mtp.fc_embedding.weight": (2560, 2560),
    "mtp.fc_hidden.weight": (2560, 2560),
    "mtp.pre_fc_norm_embedding.weight": (2560,),
    "mtp.pre_fc_norm_hidden.weight": (10240,),
}
VISION_BLOCK = {
    "model.visual.patch_embed.proj.weight": (1152, 3, 2, 16, 16),
    "model.visual.patch_embed.proj.bias": (1152,),
    "model.visual.pos_embed.weight": (2304, 1152),
    **{f"model.visual.blocks.0.{role}": shape for role, shape in {
        "norm1.weight": (1152,), "norm1.bias": (1152,),
        "attn.qkv.weight": (3456, 1152), "attn.qkv.bias": (3456,),
        "attn.proj.weight": (1152, 1152), "attn.proj.bias": (1152,),
        "norm2.weight": (1152,), "norm2.bias": (1152,),
        "mlp.linear_fc1.weight": (4304, 1152), "mlp.linear_fc1.bias": (4304,),
        "mlp.linear_fc2.weight": (1152, 4304), "mlp.linear_fc2.bias": (1152,),
    }.items()},
}


def build(output: Path, component: str, audit_only: bool = False) -> None:
    shapes = {"endpoint": ENDPOINT, "mtp-stem": MTP_STEM, "vision-block": VISION_BLOCK}[component]
    with urllib.request.urlopen(BASE + "model.safetensors.index.json", timeout=120) as response:
        index = json.load(response)["weight_map"]
    headers, inventory = {}, {}
    for name, shape in shapes.items():
        shard = index[name]
        if shard not in headers:
            headers[shard] = read_header(shard)
        base, header = headers[shard]
        item = header[name]
        begin, end = item["data_offsets"]
        if item["dtype"] != "BF16" or item["shape"] != list(shape) or end - begin != 2 * math.prod(shape):
            raise ValueError(f"pinned native {component} inventory mismatch: {name}")
        inventory[name] = dict(shard=shard, dtype="BF16", shape=shape,
                               source_file_interval=[base + begin, base + end], payload_bytes=end - begin)
    report = dict(repository=REPOSITORY, revision=REVISION, component=component,
                  tensors=inventory, payload_bytes=sum(item["payload_bytes"] for item in inventory.values()),
                  boundary="exact source BF16 words; independent bounded component, no model-quality claim")
    print(json.dumps(report, indent=2), flush=True)
    if audit_only:
        return
    if output.exists() or output.with_suffix(".json").exists():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)

    def chunks(item: dict):
        begin, end = item["source_file_interval"]
        for start in range(begin, end, 16 * 1024 * 1024):
            stop = min(end, start + 16 * 1024 * 1024)
            yield read_range(BASE + item["shard"], start, stop - 1)

    partial = output.with_suffix(".partial")
    if partial.exists():
        raise FileExistsError(partial)
    with ArtifactWriter(partial, ArtifactIdentity(f"qwen4/native-{component}-qualification", "nvidia-bf16-source"),
                        [TensorSpec(name, shape, "BF16", "contiguous-le-v1") for name, shape in shapes.items()]) as writer:
        for name, item in inventory.items():
            writer.write(name, chunks(item))
            print(f"Acquired {name}: {item['payload_bytes']} bytes", flush=True)
    partial.rename(output)
    with output.with_suffix(".json").open("x") as stream:
        json.dump(report, stream, indent=2)
        stream.write("\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--component", choices=("endpoint", "mtp-stem", "vision-block"), required=True)
    parser.add_argument("--audit-only", action="store_true")
    args = parser.parse_args()
    build(args.out, args.component, args.audit_only)
