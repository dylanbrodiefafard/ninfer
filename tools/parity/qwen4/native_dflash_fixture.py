"""Acquire and convert the pinned, exact-target Qwen4 DeepSpec DFlash companion.

The publisher's matrices are BF16, despite NVFP4 in the target's name. Our NVFP4
fixture explicitly adopts the existing max-abs encoder with A16 activations; it
is not publisher calibration. Norms remain BF16. No selector/codebook exists.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import shutil
import struct
import urllib.request

import torch
from safetensors import safe_open

from tools.artifact.container import Artifact, ArtifactIdentity, ArtifactWriter, TensorSpec
from tools.artifact.layouts import encode_nvfp4
from tools.convert.common.nvfp4_quantize import ENCODER_PROFILE, quantize_nvfp4_matrix
from tools.parity.qwen4.native_source import read_range
from tools.reference.qwen4.dflash import decode_nvfp4_words_exact, decode_nvfp4_weight

REPOSITORY = "PixelML/Qwen3.8-Flash-Next-NVFP4-DFlash"
REVISION = "9cd660f9050c92fedc88cbe547bd53af0392abe1"
BASE = f"https://huggingface.co/{REPOSITORY}/resolve/{REVISION}/"
SOURCE_FILE_BYTES = 996_219_904
SOURCE_PAYLOAD_BYTES = 996_213_760
SOURCE_NAME = "qwen4-dflash-source.safetensors"
SOURCE_IDENTITY = "qwen4/native-dflash-qualification"
LAYER_SHAPES = {
    "input_layernorm.weight": (2560,),
    "post_attention_layernorm.weight": (2560,),
    "self_attn.q_proj.weight": (6144, 2560),
    "self_attn.k_proj.weight": (512, 2560),
    "self_attn.v_proj.weight": (512, 2560),
    "self_attn.o_proj.weight": (2560, 6144),
    "self_attn.q_norm.weight": (256,),
    "self_attn.k_norm.weight": (256,),
    "mlp.gate_proj.weight": (7680, 2560),
    "mlp.up_proj.weight": (7680, 2560),
    "mlp.down_proj.weight": (2560, 7680),
}
SHAPES = {"fc.weight": (2560, 12800), "hidden_norm.weight": (2560,),
          "norm.weight": (2560,), **{
              f"layers.{layer}.{name}": shape
              for layer in range(5) for name, shape in LAYER_SHAPES.items()}}


def validate_header(header: dict, header_bytes: int) -> None:
    entries = {name: item for name, item in header.items() if name != "__metadata__"}
    if entries.keys() != SHAPES.keys():
        raise ValueError("DFlash source must contain exactly the 58 pinned tensors")
    cursor = 0
    for name in sorted(entries, key=lambda name: entries[name]["data_offsets"][0]):
        item = entries[name]
        begin, end = item["data_offsets"]
        if (item["dtype"] != "BF16" or item["shape"] != list(SHAPES[name])
                or begin != cursor or end - begin != 2 * math.prod(SHAPES[name])):
            raise ValueError(f"pinned DFlash source inventory mismatch: {name}")
        cursor = end
    if cursor != SOURCE_PAYLOAD_BYTES or header_bytes + 8 + cursor != SOURCE_FILE_BYTES:
        raise ValueError("pinned DFlash source byte count mismatch")


def validate_config(config: dict) -> None:
    expected = dict(architectures=["Qwen3DSparkModel"], hidden_size=2560,
                    intermediate_size=7680, num_hidden_layers=5, num_attention_heads=24,
                    num_key_value_heads=2, head_dim=256, vocab_size=248320,
                    block_size=7, mask_token_id=248077, markov_rank=0,
                    enable_confidence_head=False, attention_bias=False,
                    tie_word_embeddings=False, hidden_act="silu", rms_norm_eps=1e-6,
                    target_layer_ids=[3, 15, 23, 35, 43],
                    target_model_type="qwen4_exp", target_text_model_type="qwen4_exp_text",
                    layer_types=["full_attention"] * 5,
                    rope_parameters={"rope_theta": 10000000.0, "rope_type": "default"})
    for name, value in expected.items():
        if config.get(name) != value:
            raise ValueError(f"pinned DFlash config mismatch: {name}")


def acquire(directory: Path) -> None:
    destination = directory / SOURCE_NAME
    partial = destination.with_suffix(".partial")
    names = (destination, partial, directory / "qwen4-dflash-source.json",
             directory / "config.json", directory / "README.md", directory / "train_config.py")
    for path in names:
        if path.exists():
            raise FileExistsError(path)
    metadata = {}
    for name in ("config.json", "README.md", "train_config.py"):
        with urllib.request.urlopen(BASE + name, timeout=120) as response:
            data = response.read(1024 * 1024 + 1)
        if len(data) > 1024 * 1024:
            raise ValueError("unexpected DFlash metadata size")
        metadata[name] = data
    validate_config(json.loads(metadata["config.json"]))
    url = BASE + "model.safetensors"
    first = read_range(url, 0, 7)
    length = struct.unpack("<Q", first)[0]
    if length != 6136:
        raise ValueError("pinned DFlash safetensors header size mismatch")
    header_raw = read_range(url, 8, length + 7)
    validate_header(json.loads(header_raw), length)
    directory.mkdir(parents=True, exist_ok=True)
    if shutil.disk_usage(directory).free < SOURCE_FILE_BYTES + 128 * 1024 * 1024:
        raise ValueError("insufficient disk space for exact DFlash source")
    with partial.open("xb") as stream:
        stream.write(first)
        stream.write(header_raw)
        for begin in range(length + 8, SOURCE_FILE_BYTES, 16 * 1024 * 1024):
            end = min(SOURCE_FILE_BYTES, begin + 16 * 1024 * 1024)
            stream.write(read_range(url, begin, end - 1))
            print(f"DFlash source {end}/{SOURCE_FILE_BYTES} bytes", flush=True)
    partial.rename(destination)
    for name, data in metadata.items():
        with (directory / name).open("xb") as stream:
            stream.write(data)
    with (directory / "qwen4-dflash-source.json").open("x") as stream:
        json.dump(dict(repository=REPOSITORY, revision=REVISION, file=SOURCE_NAME,
                       file_bytes=SOURCE_FILE_BYTES, payload_bytes=SOURCE_PAYLOAD_BYTES,
                       source_tensors=len(SHAPES), target_revision=
                       "fc694b54fb0174e0913e6adf86691ef85a4ead47"), stream, indent=2)
        stream.write("\n")


def convert(directory: Path, output: Path, precision: str) -> None:
    if precision not in ("bf16", "nvfp4"):
        raise ValueError("DFlash qualification precision must be bf16 or nvfp4")
    partial = output.with_suffix(".partial")
    report_path = output.with_suffix(".json")
    for path in (output, partial, report_path):
        if path.exists():
            raise FileExistsError(path)
    with (directory / "qwen4-dflash-source.json").open() as stream:
        manifest = json.load(stream)
    if manifest.get("repository") != REPOSITORY or manifest.get("revision") != REVISION:
        raise ValueError("DFlash source pin mismatch")
    with (directory / "config.json").open() as stream:
        validate_config(json.load(stream))
    source_path = directory / SOURCE_NAME
    if source_path.stat().st_size != SOURCE_FILE_BYTES:
        raise ValueError("pinned DFlash source file size mismatch")
    with source_path.open("rb") as stream:
        length = struct.unpack("<Q", stream.read(8))[0]
        if length != 6136:
            raise ValueError("pinned DFlash source header size mismatch")
        validate_header(json.loads(stream.read(length)), length)
    specs = [TensorSpec(name, shape,
                        "NVFP4" if precision == "nvfp4" and len(shape) == 2 else "BF16",
                        "blockscale-k16-m128x4-v1" if precision == "nvfp4" and len(shape) == 2
                        else "contiguous-le-v1") for name, shape in SHAPES.items()]
    output.parent.mkdir(parents=True, exist_ok=True)
    loss = {}
    with safe_open(source_path, framework="pt", device="cpu") as source, ArtifactWriter(
            partial, ArtifactIdentity(SOURCE_IDENTITY, f"pixelml-{precision}-a16"), specs) as writer:
        for spec in specs:
            weight = source.get_tensor(spec.name)
            if weight.dtype != torch.bfloat16 or tuple(weight.shape) != spec.shape:
                raise ValueError(f"source weight changed: {spec.name}")
            if not torch.isfinite(weight).all():
                raise ValueError(f"non-finite DFlash source: {spec.name}")
            if spec.format == "BF16":
                payload = weight.view(torch.uint16).numpy().tobytes()
            else:
                codes, scales, divisor = quantize_nvfp4_matrix(weight)
                payload = encode_nvfp4(codes, scales, divisor, spec.shape)
                # Independent explicit-index layout oracle, not production unswizzle.
                actual_codes, actual_scales, actual_divisor = decode_nvfp4_words_exact(payload, spec.shape)
                if (not torch.equal(actual_codes, codes) or not torch.equal(actual_scales, scales)
                        or not torch.equal(actual_divisor.view(torch.int32), divisor.view(torch.int32))):
                    raise AssertionError(f"DFlash exact NVFP4 word round-trip failed: {spec.name}")
                decoded = decode_nvfp4_weight(payload, spec.shape)
                represented = weight.double()
                delta = decoded - represented
                loss[spec.name] = dict(relative_l2=float(torch.linalg.vector_norm(delta)
                                                       / torch.linalg.vector_norm(represented).clamp_min(1e-30)),
                                       max_abs=float(delta.abs().max()))
            writer.write(spec.name, payload)
            print(f"DFlash {precision}: {spec.name} {len(payload)} bytes", flush=True)
    partial.rename(output)
    # Every unquantized stored source word is checked through the artifact reader.
    with safe_open(source_path, framework="pt", device="cpu") as source, Artifact(output) as artifact:
        for spec in specs:
            if spec.format == "BF16":
                expected = source.get_tensor(spec.name).view(torch.uint16).numpy().tobytes()
                if bytes(artifact.payload(spec.name)) != expected:
                    raise AssertionError(f"BF16 source word mismatch: {spec.name}")
    with report_path.open("x") as stream:
        json.dump(dict(source=manifest, precision=precision, activations="A16",
                       encoder=ENCODER_PROFILE if precision == "nvfp4" else None,
                       exact_storage_checks="all BF16 words; all NVFP4 code/scale/divisor words",
                       source_weight_approximation=loss,
                       limitation="Drafter component only, not target acceptance, PPL, or speed qualification",
                       no_codebook=True), stream, indent=2)
        stream.write("\n")


def acquire_inputs(text_panel: Path, output: Path) -> None:
    """One exact mask row plus deduplicated authentic target anchor rows."""
    from tools.parity.qwen4.native_source import BASE as TARGET_BASE, read_header

    for path in (output, output.with_suffix(".json")):
        if path.exists():
            raise FileExistsError(path)
    with Artifact(text_panel) as panel:
        if panel.identity != ArtifactIdentity("qwen4/native-text-qualification", "nvidia-source-33"):
            raise ValueError("DFlash anchors require the pinned real-token qualification panel")
        ids = struct.unpack("<33i", bytes(panel.payload("token.ids")))
        embeddings = bytes(panel.payload("token.embeddings"))
        if len(embeddings) != 33 * 2560 * 2:
            raise ValueError("unexpected target anchor geometry")
        unique = list(dict.fromkeys(ids))
        anchor_rows = b"".join(embeddings[ids.index(token) * 5120:(ids.index(token) + 1) * 5120]
                               for token in unique)
    with urllib.request.urlopen(TARGET_BASE + "model.safetensors.index.json", timeout=120) as response:
        index = json.load(response)["weight_map"]
    name = "model.language_model.embed_tokens.weight"
    shard = index[name]
    base, header = read_header(shard)
    item = header[name]
    if item["shape"] != [248320, 2560] or item["dtype"] != "BF16":
        raise ValueError("target embedding representation mismatch")
    begin = base + item["data_offsets"][0] + 248077 * 5120
    mask_row = read_range(TARGET_BASE + shard, begin, begin + 5119)
    output.parent.mkdir(parents=True, exist_ok=True)
    with ArtifactWriter(output, ArtifactIdentity("qwen4/native-dflash-inputs-qualification",
                                                 "nvidia-bf16-source"), [
            TensorSpec("anchor.ids", (len(unique),), "I32", "contiguous-le-v1"),
            TensorSpec("anchor.embeddings", (len(unique), 2560), "BF16", "contiguous-le-v1"),
            TensorSpec("mask.embedding", (2560,), "BF16", "contiguous-le-v1")]) as writer:
        writer.write("anchor.ids", struct.pack(f"<{len(unique)}i", *unique))
        writer.write("anchor.embeddings", anchor_rows)
        writer.write("mask.embedding", mask_row)
    with output.with_suffix(".json").open("x") as stream:
        json.dump(dict(repository="nvidia/Qwen3.8-Flash-Next-NVFP4",
                       revision="fc694b54fb0174e0913e6adf86691ef85a4ead47",
                       anchor_source=str(text_panel), anchor_ids=unique, mask_token_id=248077,
                       mask_source_tensor=name, mask_source_shard=shard,
                       mask_file_interval=[begin, begin + 5120],
                       boundary="Exact BF16 embedding words; no synthetic weight or sixth feature tap"), stream, indent=2)
        stream.write("\n")
    print(f"DFlash inputs: {len(unique)} unique exact anchor rows and mask row 248077", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--acquire", action="store_true")
    parser.add_argument("--precision", choices=("bf16", "nvfp4"))
    parser.add_argument("--out", type=Path)
    parser.add_argument("--inputs-text-panel", type=Path)
    parser.add_argument("--inputs-out", type=Path)
    args = parser.parse_args()
    if (bool(args.precision) != bool(args.out) or bool(args.inputs_text_panel) != bool(args.inputs_out)
            or not (args.acquire or args.out or args.inputs_out)):
        parser.error("use --acquire, --precision/--out, or --inputs-text-panel/--inputs-out")
    torch.set_num_threads(8)
    if args.acquire:
        acquire(args.source_dir)
    if args.out:
        convert(args.source_dir, args.out, args.precision)
    if args.inputs_out:
        acquire_inputs(args.inputs_text_panel, args.inputs_out)


if __name__ == "__main__":
    main()
