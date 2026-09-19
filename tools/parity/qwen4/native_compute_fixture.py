"""Prepare exact native first-block execution artifacts offline; no runtime repacking.

The source qualification artifacts remain unchanged. NVFP4 banks and protected BF16
matrices are copied byte-for-byte except the documented bijective GDN V-head layout.
Effective FP32 controls are derived here, never by the native runtime binder.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import torch

from tools.artifact.container import Artifact, ArtifactIdentity, ArtifactWriter, TensorSpec
from tools.artifact.layouts import encode_direct
from tools.convert.qwen4.native_prepare import transformed, output_role

IDENTITY = ArtifactIdentity("qwen4/native-first-block-qualification", "nvidia-nvfp4-a16-tiled")


def read_bf16(artifact, obj):
    payload = artifact.payload(obj)
    value = torch.frombuffer(bytearray(payload), dtype=torch.bfloat16).reshape(obj.shape)
    payload.release()
    return value


def prepare(source_path, output, layer):
    if output.exists():
        raise FileExistsError(output)
    prefix = f"model.language_model.layers.{layer}."
    with Artifact(source_path) as source:
        if source.identity != ArtifactIdentity("qwen4/native-layer-qualification", "nvidia-nvfp4-source"):
            raise ValueError("requires exact source native-layer artifact")
        prepared, specs = {}, []
        for obj in source.objects:
            if not obj.name.startswith(prefix):
                raise ValueError("source artifact layer mismatch")
            role = obj.name[len(prefix):]
            if obj.format == "BF16":
                value = transformed(role, read_bf16(source, obj))
                prepared[obj.name] = value
                fmt = "FP32" if value.dtype == torch.float32 else "BF16"
                specs.append(TensorSpec(prefix + output_role(role), tuple(value.shape), fmt, "contiguous-le-v1"))
            elif obj.format == "NVFP4_EXPERT_F32M":
                specs.append(TensorSpec(obj.name, obj.shape, obj.format, obj.layout))
            else:
                raise ValueError("unexpected native first-block numeric format")
        output.parent.mkdir(parents=True, exist_ok=True)
        with ArtifactWriter(output, IDENTITY, specs) as writer:
            for obj, spec in zip(source.objects, specs):
                if obj.name in prepared:
                    writer.write(spec.name, encode_direct(prepared[obj.name], spec.format))
                else:
                    payload = source.payload(obj)
                    writer.write(spec.name, payload)
                    payload.release()
    verify(source_path, output, layer)


def verify(source_path, output, layer):
    """Independent scalar head-address checks, not a call to transformed()."""
    prefix = f"model.language_model.layers.{layer}."
    with Artifact(source_path) as source, Artifact(output) as actual:
        if actual.identity != IDENTITY or len(actual.objects) != len(source.objects):
            raise ValueError("prepared native inventory mismatch")
        for obj in source.objects:
            role = obj.name[len(prefix):]
            target = actual.find(prefix + output_role(role))
            if obj.format != "BF16":
                a, b = source.payload(obj), actual.payload(target)
                equal = a == b
                a.release(); b.release()
                if not equal:
                    raise ValueError("native expert bytes changed during preparation")
                continue
            original = read_bf16(source, obj)
            payload = actual.payload(target)
            result = torch.frombuffer(bytearray(payload), dtype=torch.float32 if target.format == "FP32" else torch.bfloat16).reshape(target.shape)
            payload.release()
            expected = original.clone()
            if role.startswith("linear_attn."):
                name = role[len("linear_attn."):]
                if name == "A_log":
                    expected = torch.empty((48,), dtype=torch.float32)
                for h in range(48):
                    source_head = 3 * (h % 16) + h // 16
                    dst, src = slice(h * 128, (h + 1) * 128), slice(source_head * 128, (source_head + 1) * 128)
                    if name in ("in_proj_qkv.weight", "conv1d.weight"):
                        expected[4096 + h*128:4096 + (h+1)*128] = original[4096 + source_head*128:4096 + (source_head+1)*128]
                    elif name == "in_proj_z.weight": expected[dst] = original[src]
                    elif name == "out_proj.weight": expected[:, dst] = original[:, src]
                    elif name in ("in_proj_a.weight", "in_proj_b.weight", "dt_bias"): expected[h] = original[source_head]
                    elif name == "A_log": expected[h] = -math.exp(float(original[source_head]))
                expected = expected.to(result.dtype)
            elif ".hc_norm.weight" in role or (role.startswith("self_attn.") and
                    (role.endswith("_norm.weight") or role.endswith("_layernorm.weight"))):
                expected = original.float() + 1
            else:
                expected = expected.to(result.dtype)
            if not torch.equal(result.view(torch.uint8), expected.contiguous().view(torch.uint8)):
                raise ValueError(f"native preparation exact oracle mismatch: {role}")
    print(f"Prepared native layer {layer}: every code/control/permuted word verified", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(2)
    for layer in range(4):
        source = args.source_root / f"qwen4-layer-{layer}.ninfer"
        output = args.out_dir / f"qwen4-native-layer-{layer}.ninfer"
        (verify if args.verify_only else prepare)(source, output, layer)
