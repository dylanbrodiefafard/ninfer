"""Append transport, fused matrix ordering and direct DFlash2 layouts (CPU only)."""

from __future__ import annotations

import json
import struct

import pytest
import torch

from tools.artifact.container import (
    Artifact,
    ArtifactError,
    ArtifactIdentity,
    ArtifactWriter,
    ResourceSpec,
    TensorSpec,
)
from tools.convert.common.nvfp4_quantize import encode_nvfp4_from_bf16
from tools.convert.qwen3_8_flash_next import dflash


def _config():
    return {
        "architectures": ["DFlash2DraftModel"],
        "hidden_size": 2560,
        "intermediate_size": 7680,
        "num_hidden_layers": 5,
        "num_target_layers": 48,
        "num_attention_heads": 32,
        "num_key_value_heads": 8,
        "head_dim": 128,
        "vocab_size": 248320,
        "sliding_window": 2048,
        "rms_norm_eps": 1e-6,
        "attention_bias": False,
        "layer_types": ["sliding_attention"] * 5,
        "rope_parameters": {"rope_theta": 10000, "rope_type": "default"},
        "dflash_config": {
            "block_size": 8,
            "conv_group_size": 16,
            "conv_kernel_size": 2,
            "mask_token_id": 248077,
            "selector_rank": 256,
            "selector_top_k": 16,
            "target_layer_ids": [5, 15, 25, 35, 45],
        },
    }


def _checkpoint(path, tensors):
    """Write independently framed BF16 safetensors, without checkpoint Python code."""
    path.mkdir()
    (path / "config.json").write_text(json.dumps(_config()))
    header, payload = {}, bytearray()
    for name, tensor in tensors.items():
        raw = tensor.contiguous().view(torch.uint8).numpy().tobytes()
        header[name] = {
            "dtype": "BF16",
            "shape": list(tensor.shape),
            "data_offsets": [len(payload), len(payload) + len(raw)],
        }
        payload.extend(raw)
    encoded = json.dumps(header).encode()
    encoded += b" " * (-len(encoded) % 8)
    (path / "model.safetensors").write_bytes(struct.pack("<Q", len(encoded)) + encoded + payload)


def _base(path):
    specs = [
        ResourceSpec("frontend/tokenizer.json", "raw-bytes-v1", 2),
        TensorSpec("text/sentinel", (2,), "I32", "contiguous-le-v1"),
    ]
    payloads = {"frontend/tokenizer.json": b"{}", "text/sentinel": struct.pack("<ii", -19, 73)}
    with ArtifactWriter(
        path, ArtifactIdentity(dflash.MODEL_ID, dflash.WEIGHTS_ID), specs
    ) as writer:
        for name, payload in payloads.items():
            writer.write(name, payload)
    return payloads


def test_append_preserves_base_and_orders_fused_and_direct_payloads(tmp_path, monkeypatch):
    # Small, codec-valid matrix geometry keeps the transport test independent of a real model.
    # The same append path handles these plans and the target's full-size inventory.
    plans, tensors, expected = [], {}, {}
    for name, parts in (("attention/query_key_value", 3), ("mlp/gate_up", 2)):
        sources = tuple(f"{name}/{index}" for index in range(parts))
        for index, source in enumerate(sources):
            tensors[source] = torch.full((128, 128), index + 1, dtype=torch.bfloat16)
        spec = TensorSpec("dflash/" + name, (128 * parts, 128), "NVFP4", "blockscale-k16-m128x4-v1")
        plans.append(dflash.DrafterTensor(spec, sources, ((128, 128),) * parts))
        reference = torch.arange(1, parts + 1, dtype=torch.bfloat16).repeat_interleave(128)
        reference = reference[:, None].expand(-1, 128).contiguous()
        expected[spec.name] = encode_nvfp4_from_bf16(reference, spec.shape)
    for name, source_shape, stored_shape in (
        ("attention_conv/base_kernel", (2, 2, 16), (16, 2, 2)),
        ("selector/predecessor_codebook", (8, 4), (4, 8)),
    ):
        tensor = torch.arange(1, 1 + int(torch.tensor(source_shape).prod())).reshape(source_shape)
        tensors[name] = tensor.to(torch.bfloat16)
        spec = TensorSpec("dflash/" + name, stored_shape, "BF16", "contiguous-le-v1")
        plans.append(dflash.DrafterTensor(spec, (name,), (source_shape,)))
        expected[spec.name] = tensors[name].view(torch.uint8).numpy().tobytes()
    monkeypatch.setattr(dflash, "inventory", lambda: tuple(plans))
    checkpoint, base, output = (
        tmp_path / "checkpoint",
        tmp_path / "base.ninfer",
        tmp_path / "out.ninfer",
    )
    _checkpoint(checkpoint, tensors)
    expected.update(_base(base))
    dflash.append(base, checkpoint, output)
    with Artifact.open(output) as artifact:
        assert artifact.identity == ArtifactIdentity(dflash.MODEL_ID, dflash.WEIGHTS_ID)
        assert {obj.name for obj in artifact.objects} == expected.keys()
        for obj in artifact.objects:
            assert bytes(artifact.payload(obj)) == expected[obj.name]
    original = output.read_bytes()
    with pytest.raises(FileExistsError):
        dflash.append(base, checkpoint, output)
    assert output.read_bytes() == original


@pytest.mark.parametrize("bad_kind", ["geometry", "source_shape", "nonfinite"])
def test_append_rejects_incompatible_checkpoint(tmp_path, monkeypatch, bad_kind):
    plan = dflash.DrafterTensor(
        TensorSpec("dflash/context_norm", (16,), "BF16", "contiguous-le-v1"),
        ("hidden_norm.weight",),
        ((16,),),
    )
    monkeypatch.setattr(dflash, "inventory", lambda: (plan,))
    values = torch.ones(16 if bad_kind != "source_shape" else 15, dtype=torch.bfloat16)
    if bad_kind == "nonfinite":
        values[0] = float("nan")
    checkpoint, base, output = (
        tmp_path / "checkpoint",
        tmp_path / "base.ninfer",
        tmp_path / "out.ninfer",
    )
    _checkpoint(checkpoint, {"hidden_norm.weight": values})
    _base(base)
    if bad_kind == "geometry":
        config = _config()
        config["dflash_config"]["target_layer_ids"] = [4, 14, 24, 34, 44]
        (checkpoint / "config.json").write_text(json.dumps(config))
    with pytest.raises(ValueError):
        dflash.append(base, checkpoint, output)
    if output.exists():
        with pytest.raises(ArtifactError):
            Artifact.open(output)
    assert not output.with_suffix(".ninfer.conversion.json").exists()
