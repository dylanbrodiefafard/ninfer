"""Append the fixed Flash-Next DFlash2 companion: NVFP4 matrices and BF16 codebooks.

Only config JSON and safetensors data are read from the checkpoint; its Python code is never
imported. Run through Python 3.11 isolated mode with the trusted repository on sys.path.
The base artifact's object payloads are copied unchanged to a distinct output file.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import torch

from tools.artifact.container import (
    Artifact,
    ArtifactIdentity,
    ArtifactWriter,
    ResourceObject,
    ResourceSpec,
    TensorSpec,
)
from tools.artifact.layouts import encode_direct
from tools.convert.common.nvfp4_quantize import ENCODER_PROFILE, encode_nvfp4_from_bf16
from tools.convert.common.safetensors import ShardReader

from .inventory import MODEL_ID, WEIGHTS_ID

HIDDEN = 2560
INTERMEDIATE = 7680
LAYERS = 5
VOCAB = 248320
FEATURE_LAYERS = (5, 15, 25, 35, 45)


@dataclass(frozen=True, slots=True)
class DrafterTensor:
    spec: TensorSpec
    sources: tuple[str, ...]
    source_shapes: tuple[tuple[int, ...], ...]


def inventory() -> tuple[DrafterTensor, ...]:
    tensors: list[DrafterTensor] = []

    def add(
        name: str,
        shape: tuple[int, ...],
        sources: tuple[str, ...],
        source_shapes: tuple[tuple[int, ...], ...],
        *,
        matrix: bool = False,
    ) -> None:
        tensors.append(
            DrafterTensor(
                TensorSpec(
                    "dflash/" + name,
                    shape,
                    "NVFP4" if matrix else "BF16",
                    "blockscale-k16-m128x4-v1" if matrix else "contiguous-le-v1",
                ),
                sources,
                source_shapes,
            )
        )

    add(
        "feature_projection",
        (HIDDEN, 5 * HIDDEN),
        ("fc.weight",),
        ((HIDDEN, 5 * HIDDEN),),
        matrix=True,
    )
    add("context_norm", (HIDDEN,), ("hidden_norm.weight",), ((HIDDEN,),))
    for layer in range(LAYERS):
        dst = f"layers/{layer}/"
        src = f"layers.{layer}."
        add(dst + "input_norm", (HIDDEN,), (src + "input_layernorm.weight",), ((HIDDEN,),))
        add(
            dst + "attention/query_key_value",
            (6144, HIDDEN),
            tuple(src + f"self_attn.{kind}_proj.weight" for kind in ("q", "k", "v")),
            ((4096, HIDDEN), (1024, HIDDEN), (1024, HIDDEN)),
            matrix=True,
        )
        for kind in ("query", "key"):
            add(
                dst + f"attention/{kind}_norm",
                (128,),
                (src + f"self_attn.{kind[0]}_norm.weight",),
                ((128,),),
            )
        add(
            dst + "attention/output",
            (HIDDEN, 4096),
            (src + "self_attn.o_proj.weight",),
            ((HIDDEN, 4096),),
            matrix=True,
        )
        add(
            dst + "attention_conv/base_kernel",
            (HIDDEN, 2, 2),
            (src + "attention_conv.base_kernel",),
            ((2, 2, HIDDEN),),
        )
        add(
            dst + "attention_conv/kernel_projection",
            (640, HIDDEN),
            (src + "attention_conv.kernel_projection.weight",),
            ((640, HIDDEN),),
            matrix=True,
        )
        add(
            dst + "post_attention_norm",
            (HIDDEN,),
            (src + "post_attention_layernorm.weight",),
            ((HIDDEN,),),
        )
        add(
            dst + "mlp/gate_up",
            (2 * INTERMEDIATE, HIDDEN),
            (src + "mlp.gate_proj.weight", src + "mlp.up_proj.weight"),
            ((INTERMEDIATE, HIDDEN), (INTERMEDIATE, HIDDEN)),
            matrix=True,
        )
        add(
            dst + "mlp/down",
            (HIDDEN, INTERMEDIATE),
            (src + "mlp.down_proj.weight",),
            ((HIDDEN, INTERMEDIATE),),
            matrix=True,
        )
        add(
            dst + "mlp_conv/base_kernel",
            (HIDDEN, 2, 2),
            (src + "mlp_conv.base_kernel",),
            ((2, 2, HIDDEN),),
        )
        add(
            dst + "mlp_conv/kernel_projection",
            (640, HIDDEN),
            (src + "mlp_conv.kernel_projection.weight",),
            ((640, HIDDEN),),
            matrix=True,
        )
    add("final_norm", (HIDDEN,), ("norm.weight",), ((HIDDEN,),))
    add(
        "selector/hidden_projection",
        (256, HIDDEN),
        ("candidate_selector.hidden_projection.weight",),
        ((256, HIDDEN),),
        matrix=True,
    )
    for kind in ("predecessor", "successor"):
        add(
            f"selector/{kind}_codebook",
            (256, VOCAB),
            (f"candidate_selector.{kind}_codebook",),
            ((VOCAB, 256),),
        )
    return tuple(tensors)


def validate_config(path: Path) -> None:
    config = json.loads(path.read_text())
    expected = {
        "architectures": ["DFlash2DraftModel"],
        "hidden_size": HIDDEN,
        "intermediate_size": INTERMEDIATE,
        "num_hidden_layers": LAYERS,
        "num_target_layers": 48,
        "num_attention_heads": 32,
        "num_key_value_heads": 8,
        "head_dim": 128,
        "vocab_size": VOCAB,
        "sliding_window": 2048,
        "rms_norm_eps": 1e-6,
        "attention_bias": False,
        "layer_types": ["sliding_attention"] * LAYERS,
        "rope_parameters": {"rope_theta": 10000, "rope_type": "default"},
        "dflash_config": {
            "block_size": 8,
            "conv_group_size": 16,
            "conv_kernel_size": 2,
            "mask_token_id": 248077,
            "selector_rank": 256,
            "selector_top_k": 16,
            "target_layer_ids": list(FEATURE_LAYERS),
        },
    }
    for name, value in expected.items():
        if config.get(name) != value:
            raise ValueError(f"DFlash2 config {name}: expected {value!r}, got {config.get(name)!r}")


def _chunks(data: memoryview) -> Iterator[memoryview]:
    raw = data.cast("B")
    for start in range(0, len(raw), 16 << 20):
        yield raw[start : start + (16 << 20)]


def append(base: Path, checkpoint: Path, output: Path) -> None:
    if output.exists():
        raise FileExistsError(f"output already exists: {output}")
    torch.set_num_threads(8)
    validate_config(checkpoint / "config.json")
    plans = inventory()
    with (
        Artifact.open(base) as source,
        ShardReader.from_file(checkpoint / "model.safetensors") as reader,
    ):
        if source.identity != ArtifactIdentity(MODEL_ID, WEIGHTS_ID):
            raise ValueError("the base must be qwen3.8-flash-next/f3-fp8-nvfp4")
        if any(obj.name.startswith("dflash/") for obj in source.objects):
            raise ValueError("the base already contains a DFlash2 companion")
        names = [name for plan in plans for name in plan.sources]
        metadata = reader.metadata(names)
        for plan in plans:
            for name, shape in zip(plan.sources, plan.source_shapes, strict=True):
                item = metadata[name]
                if item.dtype != "BF16" or tuple(item.shape) != shape:
                    raise ValueError(f"{name}: expected BF16 {shape}, got {item}")
        specs = [
            ResourceSpec(obj.name, obj.encoding, obj.bytes)
            if isinstance(obj, ResourceObject)
            else TensorSpec(obj.name, obj.shape, obj.format, obj.layout)
            for obj in source.objects
        ]
        specs.extend(plan.spec for plan in plans)
        output.parent.mkdir(parents=True, exist_ok=True)
        with ArtifactWriter(output, source.identity, specs) as writer:
            for obj in source.objects:
                writer.write(obj.name, _chunks(source.payload(obj)))
            for plan in plans:
                parts = [reader.get(name) for name in plan.sources]
                tensor = torch.cat(parts, dim=0) if len(parts) > 1 else parts[0]
                if plan.spec.format == "NVFP4":
                    payload = encode_nvfp4_from_bf16(tensor, plan.spec.shape)
                else:
                    if not bool(torch.isfinite(tensor).all()):
                        raise ValueError(f"{plan.spec.name}: nonfinite checkpoint values")
                    payload = encode_direct(tensor, "BF16")
                writer.write(plan.spec.name, payload)
                print(plan.spec.name, flush=True)
    report = {
        "base": str(base.resolve()),
        "checkpoint": str(checkpoint.resolve()),
        "encoder": ENCODER_PROFILE,
        "companion_objects": len(plans),
        "matrices": "NVFP4",
        "selector_codebooks": "BF16",
    }
    output.with_suffix(output.suffix + ".conversion.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-artifact", required=True, type=Path)
    parser.add_argument("--dflash-model", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    append(args.base_artifact, args.dflash_model, args.out)


if __name__ == "__main__":
    main()
