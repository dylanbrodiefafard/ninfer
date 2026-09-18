"""Acquire sixteen authentic FP8 PLE rows, not a full-table residency fixture."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from tools.artifact.container import ArtifactIdentity, ArtifactWriter, TensorSpec
from tools.parity.qwen4.native_source import BASE, REPOSITORY, REVISION, read_header, read_range

SHARD_ROWS = 2_500_012
ROW_IDS = tuple(row for part in (0, 1, 2, 63, 64, 126, 127)
                for row in (part * SHARD_ROWS, (part + 1) * SHARD_ROWS - 1)) + (0, 320_001_535)


def build(output: Path) -> None:
    if output.exists():
        raise FileExistsError(output)
    shard = "model-fp8-mtp-ple.safetensors"
    prefix = "model.language_model.layers.1.ple.ple_embedding.ngram_embedding."
    data_base, header = read_header(shard)
    encoded = bytearray()
    for row in ROW_IDS:
        part, local = divmod(row, SHARD_ROWS)
        tensor = header[prefix + f"shard_{part}.weight"]
        if tensor["dtype"] != "F8_E4M3" or tensor["shape"] != [SHARD_ROWS, 160]:
            raise ValueError("unexpected pinned PLE shard encoding")
        begin, end = tensor["data_offsets"]
        if end - begin != SHARD_ROWS * 160:
            raise ValueError("unexpected PLE shard size")
        start = data_base + begin + local * 160
        encoded.extend(read_range(BASE + shard, start, start + 159))
    scalar = header[prefix + "weight_scale"]
    if scalar["dtype"] != "BF16" or scalar["shape"] != [1]:
        raise ValueError("unexpected PLE scale encoding")
    begin, end = scalar["data_offsets"]
    if end - begin != 2:
        raise ValueError("unexpected PLE scale size")
    scale = read_range(BASE + shard, data_base + begin, data_base + end - 1)
    if scale != b"\x51\x39":
        raise ValueError("pinned PLE scale changed")
    if any(code & 0x7f == 0x7f for code in encoded):
        raise ValueError("nonfinite PLE source code")
    output.parent.mkdir(parents=True, exist_ok=True)
    with ArtifactWriter(output, ArtifactIdentity("qwen4/native-ple-qualification",
                                                 "nvidia-fp8-source-rows"), [
            TensorSpec("ple.rows", (16, 160), "FP8_E4M3FN_TENSOR_BF16S",
                       "tensor-scale-v1")]) as writer:
        writer.write("ple.rows", encoded + scale)
    with output.with_suffix(".json").open("x") as stream:
        json.dump(dict(repository=REPOSITORY, revision=REVISION, global_row_ids=ROW_IDS,
                       limitation="selected-row decode only; not full-table residency"), stream, indent=2)
        stream.write("\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True, type=Path)
    build(parser.parse_args().out)
