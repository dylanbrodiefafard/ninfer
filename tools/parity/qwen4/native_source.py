"""Acquire exact main compute layers 0..3 from the pinned NVIDIA source, without full shards.

Outputs source safetensors for offline conversion, never a C++ runtime format.
Every HTTP payload request must honor its exact byte range. Only selected tensor
intervals are downloaded; output codes, scales and BF16 controls are unchanged.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import shutil
import struct
import urllib.request

REPOSITORY = "nvidia/Qwen3.8-Flash-Next-NVFP4"
REVISION = "fc694b54fb0174e0913e6adf86691ef85a4ead47"
BASE = f"https://huggingface.co/{REPOSITORY}/resolve/{REVISION}/"
ITEM_BYTES = {"BF16": 2, "F32": 4, "U8": 1, "F8_E4M3": 1}
CHUNK_BYTES = 16 * 1024 * 1024


def read_range(url: str, begin: int, end: int) -> bytes:
    """Inclusive range; inspect status/headers before reading any response body."""
    request = urllib.request.Request(
        f"{url}?qwen4_range={begin}-{end}",
        headers={"Range": f"bytes={begin}-{end}", "Accept-Encoding": "identity"},
    )
    with urllib.request.urlopen(request, timeout=120) as response:
        content_range = response.headers.get("Content-Range", "")
        if response.status != 206 or not content_range.startswith(f"bytes {begin}-{end}/"):
            raise ValueError(f"server did not honor exact Range: {response.status} {content_range}")
        data = response.read(end - begin + 2)
    if len(data) != end - begin + 1:
        raise ValueError("range response length mismatch")
    return data


def read_header(shard: str) -> tuple[int, dict]:
    url = BASE + shard
    size = struct.unpack("<Q", read_range(url, 0, 7))[0]
    if not 2 <= size <= 32 * 1024 * 1024:
        raise ValueError("unexpected source header size")
    return 8 + size, json.loads(read_range(url, 8, size + 7))


def selected_tensors(index: dict, layer: int) -> dict[str, str]:
    prefix = f"model.language_model.layers.{layer}."
    selected = {name: shard for name, shard in index["weight_map"].items()
                if name.startswith(prefix) and not name.startswith(prefix + "ple.")}
    # Layer 1 crosses a physical shard boundary. Its separately acquired PLE component is
    # not duplicated here, and the host embedding table never belongs to this compute fixture.
    expected_shards = {0: {"model-00001-of-00010.safetensors"},
                       1: {"model-00001-of-00010.safetensors", "model-00002-of-00010.safetensors"},
                       2: {"model-00002-of-00010.safetensors"},
                       3: {"model-00002-of-00010.safetensors"}}[layer]
    if len(selected) != 6166 or set(selected.values()) != expected_shards:
        raise ValueError("pinned native layer inventory mismatch")
    return selected


def extract_layer(output: Path, layer: int, names: dict[str, str],
                  headers: dict[str, tuple[int, dict]]) -> dict:
    ordered = sorted(names, key=lambda name: (names[name], headers[names[name]][1][name]["data_offsets"][0]))
    destination_header = {}
    total = 0
    spans = []
    for name in ordered:
        shard = names[name]
        data_base, header = headers[shard]
        item = header[name]
        begin, end = item["data_offsets"]
        count = math.prod(item["shape"]) * ITEM_BYTES[item["dtype"]]
        if end - begin != count or begin < 0:
            raise ValueError(f"invalid source tensor span: {name}")
        destination_header[name] = dict(item, data_offsets=[total, total + count])
        total += count
        begin += data_base
        end += data_base
        if spans and spans[-1][0] == shard and spans[-1][2] == begin:
            spans[-1] = (shard, spans[-1][1], end)
        else:
            if spans and spans[-1][0] == shard and spans[-1][2] > begin:
                raise ValueError("overlapping source tensors")
            spans.append((shard, begin, end))
    expected = {0: 1_570_383_296, 1: 1_570_383_296, 2: 1_570_383_296,
                3: 1_557_359_104}[layer]
    if total != expected:
        raise ValueError("pinned native layer byte count mismatch")
    if shutil.disk_usage(output.parent).free < total + 128 * 1024 * 1024:
        raise ValueError("insufficient disk space for selected layer")
    encoded = json.dumps(destination_header, separators=(",", ":")).encode()
    encoded += b" " * (-len(encoded) % 8)
    partial = output.with_suffix(".partial")
    if output.exists() or partial.exists():
        raise FileExistsError(f"refusing to replace existing source or partial: {output}")
    with partial.open("xb") as stream:
        stream.write(struct.pack("<Q", len(encoded)))
        stream.write(encoded)
        completed = 0
        for shard, begin, end in spans:
            for start in range(begin, end, CHUNK_BYTES):
                stop = min(start + CHUNK_BYTES, end)
                stream.write(read_range(BASE + shard, start, stop - 1))
                completed += stop - start
                print(f"layer {layer}: {completed}/{total} bytes", flush=True)
    partial.rename(output)
    return dict(layer=layer, shards=sorted(headers), tensors=len(names), payload_bytes=total,
                output=output.name, source_file_intervals=spans,
                excluded="PLE is acquired independently")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--layer", type=int, choices=(0, 1, 2, 3), required=True)
    args = parser.parse_args()
    with urllib.request.urlopen(BASE + "model.safetensors.index.json", timeout=120) as response:
        index = json.load(response)
    names = selected_tensors(index, args.layer)
    headers = {shard: read_header(shard) for shard in sorted(set(names.values()))}
    args.out_dir.mkdir(parents=True, exist_ok=True)
    report = extract_layer(args.out_dir / f"qwen4-layer-{args.layer}.safetensors",
                           args.layer, names, headers)
    report.update(repository=REPOSITORY, revision=REVISION,
                  boundary="exact source payload bytes; no quantization or runtime admission")
    report_path = args.out_dir / f"qwen4-layer-{args.layer}.json"
    with report_path.open("x") as stream:
        json.dump(report, stream, indent=2)
        stream.write("\n")


if __name__ == "__main__":
    main()
