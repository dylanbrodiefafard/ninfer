"""Acquire exact calibrated per-tensor FP8 projections from the pinned producer.

No expert bank, embedding table, requantization or runtime weight repacking.
The source-only safetensors fixture can be converted offline once its exact
canonical numeric format is available.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import struct
import urllib.request

from tools.parity.qwen4.native_source import read_range
from tools.parity.qwen4.native_nvfp4_ple_fixture import read_header_url

REPOSITORY = "senfu/Qwen3.8-Flash-Next-NVFP4"
REVISION = "5d37b3b3711d8406174b96ff950c0aa16324b266"
BASE = f"https://huggingface.co/{REPOSITORY}/resolve/{REVISION}/"
PREFIX = "model.language_model.layers."
SHAPES = {
    "0.linear_attn.in_proj_qkv": (10240, 2560),
    "0.linear_attn.in_proj_z": (6144, 2560),
    "0.linear_attn.out_proj": (2560, 6144),
    "3.self_attn.q_proj": (12288, 2560),
    "3.self_attn.k_proj": (512, 2560),
    "3.self_attn.v_proj": (512, 2560),
    "3.self_attn.o_proj": (2560, 6144),
    **{f"{layer}.mlp.shared_expert.{role}": shape
       for layer in (0, 3)
       for role, shape in (("gate_proj", (640, 2560)), ("up_proj", (640, 2560)),
                           ("down_proj", (2560, 640)))},
}


def validate_scalar(raw: bytes) -> float:
    if len(raw) != 4:
        raise ValueError("expected exact FP32 scalar word")
    value = struct.unpack("<f", raw)[0]
    if not math.isfinite(value) or value <= 0:
        raise ValueError("expected finite positive FP32 multiplier")
    return value


def build(output: Path) -> None:
    if output.exists() or output.with_suffix(".json").exists():
        raise FileExistsError(output)
    with urllib.request.urlopen(BASE + "model.safetensors.index.json", timeout=120) as response:
        index = json.load(response)["weight_map"]
    headers, tensors, source_report = {}, {}, {}
    total = 0
    for role, shape in SHAPES.items():
        report = {}
        for suffix in ("weight", "weight_scale", "input_scale"):
            name = PREFIX + role + "." + suffix
            shard = index[name]
            if shard not in headers:
                headers[shard] = read_header_url(BASE + shard)
            data, header = headers[shard]
            item = header[name]
            dtype = "F8_E4M3" if suffix == "weight" else "F32"
            expected_shape = list(shape) if suffix == "weight" else []
            size = math.prod(shape) if suffix == "weight" else 4
            begin, end = item["data_offsets"]
            if item["dtype"] != dtype or item["shape"] != expected_shape or end - begin != size:
                raise ValueError(f"pinned FP8 role inventory mismatch: {name}")
            raw = bytearray()
            for start in range(begin, end, 8 * 1024 * 1024):
                stop = min(end, start + 8 * 1024 * 1024)
                raw.extend(read_range(BASE + shard, data + start, data + stop - 1))
            if suffix != "weight":
                report[suffix] = dict(value=validate_scalar(raw), little_endian_hex=raw.hex())
            else:
                if 0x7f in raw or 0xff in raw:
                    raise ValueError("nonfinite source E4M3 code")
                report.update(shape=shape, shard=shard)
            tensors[name] = (dtype, expected_shape, raw)
            total += size
        source_report[PREFIX + role] = report
    if total != sum(math.prod(shape) + 8 for shape in SHAPES.values()):
        raise ValueError("selected FP8 byte count mismatch")
    header, offset = {}, 0
    for name, (dtype, shape, raw) in tensors.items():
        header[name] = dict(dtype=dtype, shape=shape, data_offsets=[offset, offset + len(raw)])
        offset += len(raw)
    encoded = json.dumps(header, separators=(",", ":")).encode()
    encoded += b" " * (-len(encoded) % 8)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("xb") as stream:
        stream.write(struct.pack("<Q", len(encoded)) + encoded)
        for _, _, raw in tensors.values():
            stream.write(raw)
    with output.with_suffix(".json").open("x") as stream:
        json.dump(dict(repository=REPOSITORY, revision=REVISION, payload_bytes=total,
                       projections=source_report,
                       scalar_semantics="weight=E4M3*weight_scale; activation_pack=E4M3(x/input_scale); both stored scales are dequant multipliers",
                       limitation="bounded projection fixture; protected-role compatibility with other checkpoints is not assumed"),
                  stream, indent=2)
        stream.write("\n")
    print(json.dumps(source_report, indent=2))
    print(f"Acquired {total} bytes across {len(SHAPES)} projections")


def audit_controls(source_root: Path, report_path: Path) -> None:
    """Compare every unquantized selected-layer control byte to local NVIDIA source."""
    if report_path.exists():
        raise FileExistsError(report_path)
    with urllib.request.urlopen(BASE + "model.safetensors.index.json", timeout=120) as response:
        index = json.load(response)["weight_map"]
    headers, comparisons = {}, {}
    total = 0
    projection_names = {PREFIX + role + ".weight" for role in SHAPES}
    for layer in (0, 3):
        with (source_root / f"qwen4-layer-{layer}.safetensors").open("rb") as stream:
            size = struct.unpack("<Q", stream.read(8))[0]
            local_header = json.loads(stream.read(size))
            for name, item in local_header.items():
                if item.get("dtype") != "BF16" or ".mlp.experts." in name or name in projection_names:
                    continue
                shard = index[name]
                if shard not in headers:
                    headers[shard] = read_header_url(BASE + shard)
                data, header = headers[shard]
                source = header[name]
                if source["dtype"] != "BF16" or source["shape"] != item["shape"]:
                    raise ValueError("protected-role source representation differs")
                begin, end = source["data_offsets"]
                local_begin, local_end = item["data_offsets"]
                if end - begin != local_end - local_begin:
                    raise ValueError("protected-role source size differs")
                stream.seek(8 + size + local_begin)
                equal = True
                for start in range(begin, end, 8 * 1024 * 1024):
                    stop = min(end, start + 8 * 1024 * 1024)
                    raw = read_range(BASE + shard, data + start, data + stop - 1)
                    equal &= raw == stream.read(len(raw))
                comparisons[name] = equal
                total += end - begin
    report = dict(repository=REPOSITORY, revision=REVISION,
                  compared_with_repository="nvidia/Qwen3.8-Flash-Next-NVFP4",
                  compared_with_revision="fc694b54fb0174e0913e6adf86691ef85a4ead47",
                  source_root=str(source_root), compared_bytes=total, exact_equal=comparisons,
                  limitation="unquantized controls only; no expert or full-model identity assertion")
    with report_path.open("x") as stream:
        json.dump(report, stream, indent=2); stream.write("\n")
    print(f"Protected controls: {sum(comparisons.values())}/{len(comparisons)} exact, {total} bytes")
    if not all(comparisons.values()):
        raise ValueError("protected source roles differ; reuse is not admitted")


def convert(source: Path, output: Path) -> None:
    """Exact code/scalar rearrangement into the native calibrated tensor format."""
    import torch
    from safetensors import safe_open
    from tools.artifact.container import ArtifactIdentity, ArtifactWriter, TensorSpec
    from tools.artifact.layouts import encode_fp8_calibrated, decode_fp8_calibrated_words

    if output.exists():
        raise FileExistsError(output)
    specs = [TensorSpec(PREFIX + name + ".weight", shape, "FP8_E4M3FN_TENSOR_F32M",
                        "tensor-calibrated-v1") for name, shape in SHAPES.items()]
    with safe_open(str(source), framework="pt", device="cpu") as src:
        expected = {PREFIX + role + "." + suffix for role in SHAPES
                    for suffix in ("weight", "weight_scale", "input_scale")}
        if set(src.keys()) != expected:
            raise ValueError("requires exact selected source projection inventory")
        with ArtifactWriter(output, ArtifactIdentity("qwen4/native-fp8-projection-qualification",
                                                     "senfu-fp8-source"), specs) as writer:
            for role, shape in SHAPES.items():
                name = PREFIX + role
                weight = src.get_tensor(name + ".weight")
                if weight.dtype != torch.float8_e4m3fn or tuple(weight.shape) != shape:
                    raise ValueError("unexpected source FP8 matrix")
                codes = weight.view(torch.uint8)
                scales = [src.get_tensor(name + "." + suffix)
                          for suffix in ("weight_scale", "input_scale")]
                payload = encode_fp8_calibrated(codes, *scales)
                restored = decode_fp8_calibrated_words(payload, shape)
                for actual, original in zip(restored, (codes, *scales)):
                    if not torch.equal(actual.reshape(-1).view(torch.uint8),
                                       original.reshape(-1).view(torch.uint8)):
                        raise ValueError("source code/scalar words changed")
                writer.write(name + ".weight", payload)
    print(f"Converted {len(specs)} exact source projections; code and both scalar words roundtrip passed")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    action = parser.add_mutually_exclusive_group()
    action.add_argument("--audit-controls", type=Path, help="local pinned NVIDIA layer source directory")
    action.add_argument("--convert-source", type=Path, help="existing source-only projection safetensors")
    args = parser.parse_args()
    if args.audit_controls:
        audit_controls(args.audit_controls, args.out)
    elif args.convert_source:
        convert(args.convert_source, args.out)
    else:
        build(args.out)
