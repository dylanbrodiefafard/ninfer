"""Acquire bounded authentic NVFP4 PLE rows and matching upstream BF16 rows.

This source fixture is not a product artifact or a complete-table residency test.
It preserves one multiplier per selected row, including differing source shards.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import struct
import urllib.request

from tools.parity.qwen4.native_source import read_range

REPOSITORY = "primitive-ai/Qwen3.8-Flash-Next-PLE-quant"
REVISION = "a0fa93f2b9ed5fcab0cf01b60e88ff1ce13a4763"
BF16_REPOSITORY = "Qwen/Qwen3.8-Flash-Next"
BF16_REVISION = "de4b8e4d43b917e7706784d8bb445c9af86a3540"
SHARD_ROWS = 2_500_012
PARTITIONS = (0, 1, 127)
LOCAL_ROWS = (0, SHARD_ROWS - 1, 17, 123456)
ROW_IDS = tuple(part * SHARD_ROWS + row for part in PARTITIONS for row in LOCAL_ROWS)


def read_header_url(url: str) -> tuple[int, dict]:
    size = struct.unpack("<Q", read_range(url, 0, 7))[0]
    if not 2 <= size <= 32 * 1024 * 1024:
        raise ValueError("unexpected source header size")
    return size + 8, json.loads(read_range(url, 8, size + 7))


def row_span(item: dict, dtype: str, shape: list[int], row: int, row_bytes: int) -> tuple[int, int]:
    if item["dtype"] != dtype or item["shape"] != shape:
        raise ValueError("unexpected source tensor encoding")
    begin, end = item["data_offsets"]
    if end - begin != shape[0] * row_bytes or not 0 <= row < shape[0]:
        raise ValueError("invalid source row span")
    return begin + row * row_bytes, begin + (row + 1) * row_bytes - 1


def decode_row(codes: bytes, scales: bytes, multiplier: bytes) -> list[float]:
    """Independent scalar formula; no producer/kernel decoder dependency."""
    if len(codes) != 80 or len(scales) != 10 or len(multiplier) != 4:
        raise ValueError("invalid encoded PLE row")
    m = struct.unpack("<f", multiplier)[0]
    if not math.isfinite(m) or m <= 0:
        raise ValueError("invalid shard multiplier")
    values = []
    magnitudes = (0., .5, 1., 1.5, 2., 3., 4., 6.)
    for k in range(160):
        code = codes[k // 2] >> (4 * (k % 2)) & 15
        s = scales[k // 16]
        if s >= 127:
            raise ValueError("invalid nonnegative finite E4M3 scale")
        exponent, fraction = s >> 3, s & 7
        scale = math.ldexp(fraction, -9) if exponent == 0 else math.ldexp(8 + fraction, exponent - 10)
        value = math.copysign(magnitudes[code & 7], -1. if code & 8 else 1.)
        # E2M1 times E4M3 is exact in FP32; final product has its specified FP32 rounding.
        values.append(struct.unpack("<f", struct.pack("<f", value * scale * m))[0])
    return values


def bf16_word(value: float) -> int:
    word = struct.unpack("<I", struct.pack("<f", value))[0]
    return ((word + 0x7fff + ((word >> 16) & 1)) >> 16) & 0xffff


def build(output: Path) -> None:
    if output.exists() or output.with_suffix(".json").exists():
        raise FileExistsError(output)
    base = f"https://huggingface.co/{REPOSITORY}/resolve/{REVISION}/"
    original = f"https://huggingface.co/{BF16_REPOSITORY}/resolve/{BF16_REVISION}/"
    with urllib.request.urlopen(original + "model.safetensors.index.json", timeout=120) as response:
        index = json.load(response)["weight_map"]
    headers = {}
    def tensor_row(url, name, dtype, shape, row, size):
        if url not in headers:
            headers[url] = read_header_url(url)
        data, header = headers[url]
        begin, end = row_span(header[name], dtype, shape, row, size)
        return read_range(url, data + begin, data + end)
    codes, scales, bf16, decoded = (bytearray() for _ in range(4))
    source_scales = {}
    error, public_error, energy = 0., 0., 0.
    prefix = "model.language_model.layers.1.ple.ple_embedding.ngram_embedding."
    for row in ROW_IDS:
        shard, local = divmod(row, SHARD_ROWS)
        url = base + f"ples_nvfp4/shard_{shard}.safetensors"
        q = tensor_row(url, "weight_e2m1", "U8", [SHARD_ROWS, 80], local, 80)
        s = tensor_row(url, "weight_scale", "F8_E4M3", [SHARD_ROWS, 10], local, 10)
        data, header = headers[url]
        scalar = header["weight_scale_2"]
        if scalar["dtype"] != "F32" or scalar["shape"] != []:
            raise ValueError("unexpected source multiplier")
        begin, end = scalar["data_offsets"]
        if end - begin != 4:
            raise ValueError("invalid source multiplier span")
        m = read_range(url, data + begin, data + end - 1)
        name = prefix + f"shard_{shard}.weight"
        b = tensor_row(original + index[name], name, "BF16", [SHARD_ROWS, 160], local, 320)
        values = decode_row(q, s, m)
        words = struct.unpack("<160H", b)
        for value, word in zip(values, words):
            ref = struct.unpack("<f", struct.pack("<I", word << 16))[0]
            error += (value - ref) ** 2
            public_value = struct.unpack("<f", struct.pack("<I", bf16_word(value) << 16))[0]
            public_error += (public_value - ref) ** 2
            energy += ref ** 2
        codes.extend(q); scales.extend(s); bf16.extend(b)
        decoded.extend(struct.pack("<160H", *(bf16_word(x) for x in values)))
        source_scales[str(shard)] = m.hex()
    # Source-only safetensors: no generic numeric format is silently reinterpreted.
    count = len(ROW_IDS)
    tensors = (("codes", "U8", [3, 4, 80], codes), ("scales", "F8_E4M3", [3, 4, 10], scales),
               ("multipliers", "F32", [3], b"".join(bytes.fromhex(source_scales[str(p)]) for p in PARTITIONS)),
               ("source_bf16", "BF16", [count, 160], bf16),
               ("decoded_bf16", "BF16", [count, 160], decoded))
    header, payload = {}, bytearray()
    for name, dtype, shape, data in tensors:
        header[name] = dict(dtype=dtype, shape=shape, data_offsets=[len(payload), len(payload) + len(data)])
        payload.extend(data)
    encoded = json.dumps(header, separators=(",", ":")).encode()
    encoded += b" " * (-len(encoded) % 8)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("xb") as stream:
        stream.write(struct.pack("<Q", len(encoded)) + encoded + payload)
    report = dict(repository=REPOSITORY, revision=REVISION, bf16_repository=BF16_REPOSITORY,
                  bf16_revision=BF16_REVISION, global_row_ids=ROW_IDS, shard_multiplier_le_hex=source_scales,
                  fp32_reconstruction_vs_source_bf16_relative_l2=math.sqrt(error / energy),
                  bf16_gather_vs_source_bf16_relative_l2=math.sqrt(public_error / energy),
                  limitation="selected rows only; source BF16 revision is independently pinned, not producer-provenance certified; no model quality or complete residency claim")
    with output.with_suffix(".json").open("x") as stream:
        json.dump(report, stream, indent=2); stream.write("\n")
    print(json.dumps(report, indent=2))


def convert(source: Path, output: Path) -> None:
    """Offline source words to the canonical partitioned PLE artifact layout."""
    import torch
    from tools.artifact.container import ArtifactIdentity, ArtifactWriter, TensorSpec
    from tools.artifact.layouts import pack_nvfp4_partitions, unpack_nvfp4_partitions

    if output.exists():
        raise FileExistsError(output)
    raw = source.read_bytes()
    size = struct.unpack_from("<Q", raw)[0]
    header = json.loads(raw[8:8 + size])
    def tensor(name, dtype, shape, source_dtype):
        item = header[name]
        if item["shape"] != shape or item["dtype"] != source_dtype:
            raise ValueError("unexpected source fixture representation")
        begin, end = item["data_offsets"]
        return torch.frombuffer(bytearray(raw[8 + size + begin:8 + size + end]), dtype=dtype).reshape(shape)
    codes = tensor("codes", torch.uint8, [3, 4, 80], "U8")
    scales = tensor("scales", torch.uint8, [3, 4, 10], "F8_E4M3")
    multipliers = tensor("multipliers", torch.float32, [3], "F32")
    payload = pack_nvfp4_partitions(codes, scales, multipliers)
    restored = unpack_nvfp4_partitions(payload, (3, 4, 160))
    if any(not torch.equal(a.view(torch.uint8), b.view(torch.uint8))
           for a, b in zip((codes, scales, multipliers), restored)):
        raise ValueError("source code/scale/multiplier words changed")
    with ArtifactWriter(output, ArtifactIdentity("qwen4/native-ple-qualification",
                                                 "primitive-nvfp4-source-rows"), [
            TensorSpec("ple.rows", (3, 4, 160), "NVFP4_PARTITION_F32M",
                       "partitioned-row-blockscale-k16-v1")]) as writer:
        writer.write("ple.rows", payload)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--convert-source", type=Path,
                        help="convert an existing bounded source fixture offline")
    args = parser.parse_args()
    if args.convert_source:
        convert(args.convert_source, args.out)
    else:
        build(args.out)
