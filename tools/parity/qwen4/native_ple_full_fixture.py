"""Acquire/stream-convert only the complete pinned NVFP4 PLE table, never text weights.

Source downloads and the converted artifact are retained. Conversion uses bounded
row chunks, preserving every packed code, E4M3 scale and partition F32 word.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import math
from pathlib import Path
import struct
import urllib.request

import numpy as np

from tools.artifact.container import ArtifactIdentity, ArtifactWriter, TensorSpec
from tools.parity.qwen4.native_nvfp4_ple_fixture import (
    REPOSITORY, REVISION, SHARD_ROWS, bf16_word, decode_row,
)

PARTITIONS = 128
PAYLOAD_BYTES = PARTITIONS * SHARD_ROWS * 90 + PARTITIONS * 4


def header(path: Path) -> tuple[int, dict]:
    with path.open("rb") as stream:
        size = struct.unpack("<Q", stream.read(8))[0]
        if not 2 <= size <= 32 * 1024 * 1024:
            raise ValueError("invalid source header size")
        result = json.loads(stream.read(size))
    spans = []
    for name, dtype, shape, nbytes in (
        ("weight_e2m1", "U8", [SHARD_ROWS, 80], SHARD_ROWS * 80),
        ("weight_scale", "F8_E4M3", [SHARD_ROWS, 10], SHARD_ROWS * 10),
        ("weight_scale_2", "F32", [], 4),
    ):
        item = result[name]
        begin, end = item["data_offsets"]
        if item["dtype"] != dtype or item["shape"] != shape or end - begin != nbytes:
            raise ValueError(f"source inventory differs: {path} {name}")
        spans.append((begin, end))
    spans.sort()
    if spans[0][0] != 0 or any(a[1] != b[0] for a, b in zip(spans, spans[1:])):
        raise ValueError("source spans overlap or have gaps")
    if path.stat().st_size != 8 + size + spans[-1][1]:
        raise ValueError("source file is incomplete or has trailing bytes")
    return size + 8, result


def acquire(root: Path, workers: int) -> None:
    root.mkdir(parents=True, exist_ok=True)

    def shard(partition: int) -> None:
        output = root / f"shard_{partition}.safetensors"
        if output.exists():
            header(output)
            return
        temporary = output.with_suffix(".safetensors.partial")
        # Exclusive creation: interrupted downloads are deliberately not trusted.
        url = f"https://huggingface.co/{REPOSITORY}/resolve/{REVISION}/ples_nvfp4/{output.name}"
        request = urllib.request.Request(url, headers={"Accept-Encoding": "identity"})
        with urllib.request.urlopen(request, timeout=180) as response, temporary.open("xb") as stream:
            while chunk := response.read(8 * 1024 * 1024):
                stream.write(chunk)
        header(temporary)
        temporary.rename(output)
        print(f"PLE source {partition + 1}/{PARTITIONS}: {output.stat().st_size} bytes", flush=True)

    with ThreadPoolExecutor(max_workers=workers) as pool:
        list(pool.map(shard, range(PARTITIONS)))


def chunks(root: Path):
    multipliers = bytearray()
    for partition in range(PARTITIONS):
        path = root / f"shard_{partition}.safetensors"
        base, entries = header(path)
        with path.open("rb") as codes, path.open("rb") as scales, path.open("rb") as scalar:
            codes.seek(base + entries["weight_e2m1"]["data_offsets"][0])
            scales.seek(base + entries["weight_scale"]["data_offsets"][0])
            scalar.seek(base + entries["weight_scale_2"]["data_offsets"][0])
            word = scalar.read(4)
            multiplier = struct.unpack("<f", word)[0]
            if not math.isfinite(multiplier) or multiplier <= 0:
                raise ValueError("invalid source multiplier")
            multipliers.extend(word)
            for begin in range(0, SHARD_ROWS, 16384):
                count = min(16384, SHARD_ROWS - begin)
                c = np.frombuffer(codes.read(count * 80), dtype=np.uint8).reshape(count, 80)
                s = np.frombuffer(scales.read(count * 10), dtype=np.uint8).reshape(count, 10)
                if np.any(s >= 127):
                    raise ValueError("nonfinite/negative source E4M3 scale")
                yield np.concatenate((c, s), axis=1).tobytes()
        print(f"PLE convert partition {partition + 1}/{PARTITIONS}", flush=True)
    yield bytes(multipliers)


def convert(root: Path, output: Path) -> None:
    reference = output.with_name(output.stem + "-boundary-reference.ninfer")
    report = output.with_suffix(".json")
    temporary = output.with_suffix(".ninfer.partial")
    for path in (output, reference, report, temporary):
        if path.exists():
            raise FileExistsError(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    spec = TensorSpec("ple.table", (PARTITIONS, SHARD_ROWS, 160), "NVFP4_PARTITION_F32M",
                      "partitioned-row-blockscale-k16-v1")
    with ArtifactWriter(temporary, ArtifactIdentity("qwen4/native-ple-qualification",
                        "primitive-nvfp4-complete-table"), [spec]) as writer:
        writer.write("ple.table", chunks(root))
    temporary.rename(output)
    # Independent scalar formula from original source words at both ends of every
    # partition. No production codec or converted payload enters this oracle.
    expected = bytearray()
    ids = []
    for partition in range(PARTITIONS):
        path = root / f"shard_{partition}.safetensors"
        base, entries = header(path)
        with path.open("rb") as stream:
            stream.seek(base + entries["weight_scale_2"]["data_offsets"][0])
            multiplier = stream.read(4)
            for row in (0, SHARD_ROWS - 1):
                stream.seek(base + entries["weight_e2m1"]["data_offsets"][0] + row * 80)
                codes = stream.read(80)
                stream.seek(base + entries["weight_scale"]["data_offsets"][0] + row * 10)
                scales = stream.read(10)
                words = [bf16_word(value) for value in decode_row(codes, scales, multiplier)]
                expected.extend(struct.pack("<160H", *words))
                ids.append(partition * SHARD_ROWS + row)
    with ArtifactWriter(reference, ArtifactIdentity("qwen4/native-ple-reference", "source-scalar-bf16"),
                        [TensorSpec("row_ids", (256,), "I32", "contiguous-le-v1"),
                         TensorSpec("embedding", (256, 160), "BF16", "contiguous-le-v1")]) as writer:
        writer.write("row_ids", struct.pack("<256i", *ids))
        writer.write("embedding", expected)
    with report.open("x") as stream:
        json.dump(dict(repository=REPOSITORY, revision=REVISION, partitions=PARTITIONS,
                       rows_per_partition=SHARD_ROWS, payload_bytes=PAYLOAD_BYTES,
                       conversion="exact U8 codes/E4M3 words interleaved per row; exact F32 partition words",
                       working_chunk_rows=16384, reference="independent scalar source boundary rows"),
                  stream, indent=2)
        stream.write("\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sources", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--workers", type=int, choices=range(1, 5), default=4)
    args = parser.parse_args()
    if args.download:
        acquire(args.sources, args.workers)
    convert(args.sources, args.out)
