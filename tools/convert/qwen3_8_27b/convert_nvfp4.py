"""Append DFlash2 onto a Qwen3.8-27B NVFP4 `.ninfer` by copying the live artifact.

Canonical invocation on this machine (no BF16/NVFP4 safetensor sources)::

    python3 -m tools.convert.qwen3_8_27b.convert_nvfp4 \
      --base-artifact /path/to/qwen3_8_27b_nvfp4.ninfer \
      --dflash-model /path/to/Qwen3.8-27B-DFlash2 \
      --dflash-format w8 \
      --out out/qwen3_8_27b_nvfp4_dflash_w8.ninfer

The Text/MTP/Vision object bytes stay identical and in the same order. DFlash2
objects are appended after Vision. `--dflash-format q4` emits a sibling that
differs only in DFlash2 matrix QType.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
import time
from typing import Iterator, Mapping

import torch

from tools.artifact.container import (
    Artifact,
    ArtifactIdentity,
    ArtifactWriter,
    ResourceObject,
    ResourceSpec as ArtifactResourceSpec,
    TensorObject,
    TensorSpec as ArtifactTensorSpec,
)
from tools.artifact.layouts import encode_direct
from tools.convert.common.quantize import pick_device
from tools.convert.common.safetensors import ShardReader
from tools.convert.qwen3_6.common import conversion as family_conversion
from tools.convert.qwen3_6.common.inventory import TensorSpec

from . import inventory_dflash2 as dflash2


RECIPE_ID = "qwen3_8_27b_nvfp4_dflash2-v1"
_COPY_CHUNK = 64 << 20

_DFLASH_ROOT = {
    "architectures": ["DFlash2DraftModel"],
    "model_type": "qwen3",
    "hidden_size": 5120,
    "intermediate_size": 17408,
    "num_hidden_layers": 5,
    "num_attention_heads": 32,
    "num_key_value_heads": 8,
    "head_dim": 128,
    "vocab_size": 248320,
    "rms_norm_eps": 1e-6,
    "hidden_act": "silu",
    "attention_bias": False,
    "is_causal": False,
    "sliding_window": 2048,
    "use_sliding_window": True,
    "num_target_layers": 64,
}
_DFLASH_INNER = {
    "block_size": 8,
    "conv_group_size": 16,
    "conv_kernel_size": 2,
    "mask_token_id": 248070,
    "selector_rank": 256,
    "selector_top_k": 16,
    "target_layer_ids": [5, 19, 33, 47, 61],
}
_ROPE = {"rope_theta": 10000000, "rope_type": "default"}


@dataclass(frozen=True, slots=True)
class ConversionPreflight:
    base_artifact: Path
    dflash_model: Path
    dflash_format: str
    identity: ArtifactIdentity
    dflash_specs: tuple[TensorSpec, ...]
    dflash_config: dict[str, object]


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def validate_dflash2_config(config: Mapping[str, object]) -> dict[str, object]:
    family_conversion.check_members("dflash2 config", config, _DFLASH_ROOT)
    inner = config.get("dflash_config")
    rope = config.get("rope_parameters")
    if not isinstance(inner, Mapping) or not isinstance(rope, Mapping):
        raise ValueError("DFlash2 config.json must contain dflash_config and rope_parameters")
    family_conversion.check_members("dflash2 config.dflash_config", inner, _DFLASH_INNER)
    family_conversion.check_members("dflash2 config.rope_parameters", rope, _ROPE)
    layer_types = config.get("layer_types")
    if layer_types != ["sliding_attention"] * 5:
        raise ValueError("DFlash2 layer_types must be five sliding_attention entries")
    return {
        "architecture": "DFlash2DraftModel",
        "hidden_size": 5120,
        "layers": 5,
        "sliding_window": 2048,
        "mask_token_id": 248070,
        "target_layer_ids": list(_DFLASH_INNER["target_layer_ids"]),
        "dflash_format": None,
    }


def _require_qwen38_nvfp4(identity: ArtifactIdentity) -> None:
    if identity.model_id != "qwen3.8-27b" or identity.weights_id != "nvfp4":
        raise ValueError(
            f"base artifact identity must be qwen3.8-27b/nvfp4, got "
            f"{identity.model_id}/{identity.weights_id}"
        )


def _artifact_specs(src: Artifact) -> list[ArtifactTensorSpec | ArtifactResourceSpec]:
    specs: list[ArtifactTensorSpec | ArtifactResourceSpec] = []
    for obj in src.objects:
        if isinstance(obj, TensorObject):
            specs.append(
                ArtifactTensorSpec(obj.name, obj.shape, obj.format, obj.layout)
            )
        elif isinstance(obj, ResourceObject):
            specs.append(ArtifactResourceSpec(obj.name, obj.encoding, obj.bytes))
        else:
            raise TypeError(f"unsupported artifact object: {type(obj)!r}")
    return specs


def _inventory_to_artifact_spec(spec: TensorSpec) -> ArtifactTensorSpec:
    return ArtifactTensorSpec(spec.name, spec.shape, spec.format, spec.layout)


def _payload_chunks(view: memoryview) -> Iterator[memoryview]:
    blob = view.cast("B")
    for begin in range(0, len(blob), _COPY_CHUNK):
        yield blob[begin : begin + _COPY_CHUNK]


def _cat_rows(reader: ShardReader, names: tuple[str, ...]) -> torch.Tensor:
    return torch.cat([reader.get(name) for name in names], dim=0)


def _materialize_dflash2(
    spec: TensorSpec, reader: ShardReader
) -> torch.Tensor:
    name = spec.name
    if name == "dflash/feature_projection":
        return reader.get("fc.weight")
    if name == "dflash/context_norm":
        return reader.get("hidden_norm.weight")
    if name == "dflash/final_norm":
        return reader.get("norm.weight")
    if name == "dflash/selector/hidden_projection":
        return reader.get("candidate_selector.hidden_projection.weight")
    if name == "dflash/selector/predecessor_codebook":
        return reader.get("candidate_selector.predecessor_codebook")
    if name == "dflash/selector/successor_codebook":
        return reader.get("candidate_selector.successor_codebook")

    prefix = "dflash/layers/"
    if not name.startswith(prefix):
        raise ValueError(f"unhandled DFlash2 object: {name}")
    rest = name[len(prefix) :]
    layer_s, _, role = rest.partition("/")
    layer = int(layer_s)
    source = f"layers.{layer}."
    mapping = {
        "input_norm": "input_layernorm.weight",
        "attention/query_norm": "self_attn.q_norm.weight",
        "attention/key_norm": "self_attn.k_norm.weight",
        "attention/output": "self_attn.o_proj.weight",
        "attention_conv/base_kernel": "attention_conv.base_kernel",
        "attention_conv/kernel_projection": "attention_conv.kernel_projection.weight",
        "post_attention_norm": "post_attention_layernorm.weight",
        "mlp/down": "mlp.down_proj.weight",
        "mlp_conv/base_kernel": "mlp_conv.base_kernel",
        "mlp_conv/kernel_projection": "mlp_conv.kernel_projection.weight",
    }
    if role == "attention/query_key_value":
        return _cat_rows(
            reader,
            (
                source + "self_attn.q_proj.weight",
                source + "self_attn.k_proj.weight",
                source + "self_attn.v_proj.weight",
            ),
        )
    if role == "mlp/gate_up":
        return _cat_rows(
            reader,
            (source + "mlp.gate_proj.weight", source + "mlp.up_proj.weight"),
        )
    try:
        return reader.get(source + mapping[role])
    except KeyError as exc:
        raise ValueError(f"unhandled DFlash2 object: {name}") from exc


def preflight_conversion(
    base_artifact: str | Path,
    dflash_model_dir: str | Path,
    dflash_format: str,
) -> ConversionPreflight:
    base = Path(base_artifact)
    dflash_model = Path(dflash_model_dir)
    dflash2.dflash2_matrix_format(dflash_format)
    with Artifact.open(base) as src:
        _require_qwen38_nvfp4(src.identity)
        if any(obj.name.startswith("dflash/") for obj in src.objects):
            raise ValueError(f"{base}: already contains dflash/ objects")
        identity = src.identity
    config = family_conversion.load_json(dflash_model / "config.json")
    summary = validate_dflash2_config(config)
    summary["dflash_format"] = dflash_format
    safetensors = dflash_model / "model.safetensors"
    if not safetensors.is_file():
        raise ValueError(f"{safetensors}: DFlash2 companion is missing")
    return ConversionPreflight(
        base_artifact=base,
        dflash_model=dflash_model,
        dflash_format=dflash_format,
        identity=identity,
        dflash_specs=dflash2.build_dflash2_specs(dflash_format),
        dflash_config=summary,
    )


def convert(
    base_artifact: str | Path,
    dflash_model_dir: str | Path,
    out_path: str | Path,
    *,
    dflash_format: str = "w8",
    device: str | torch.device = "cuda",
) -> Path:
    started = time.perf_counter()
    output = Path(out_path)
    resolved_device = pick_device(device)
    preflight = preflight_conversion(base_artifact, dflash_model_dir, dflash_format)
    output.parent.mkdir(parents=True, exist_ok=True)

    print(
        f"preflight complete: copy {preflight.base_artifact} + "
        f"{len(preflight.dflash_specs)} DFlash2 {dflash_format} objects, "
        f"device={resolved_device}",
        flush=True,
    )

    with Artifact.open(preflight.base_artifact) as src, ShardReader.from_file(
        preflight.dflash_model / "model.safetensors"
    ) as reader:
        specs = _artifact_specs(src)
        specs.extend(_inventory_to_artifact_spec(spec) for spec in preflight.dflash_specs)
        with ArtifactWriter(output, src.identity, specs) as writer:
            for index, obj in enumerate(src.objects, start=1):
                writer.write(obj.name, _payload_chunks(src.payload(obj)))
                print(f"[{index}/{len(specs)}] {obj.name} (copy)", flush=True)
            copied = len(src.objects)
            for offset, spec in enumerate(preflight.dflash_specs, start=1):
                tensor = _materialize_dflash2(spec, reader)
                payload = family_conversion.encode_tensor_payload(
                    tensor, spec, resolved_device
                )
                del tensor
                writer.write(spec.name, payload)
                del payload
                print(
                    f"[{copied + offset}/{len(specs)}] {spec.name}",
                    flush=True,
                )

    report = {
        "identity": {
            "model_id": preflight.identity.model_id,
            "weights_id": preflight.identity.weights_id,
        },
        "target_key": "qwen3_8_27b",
        "recipe_id": RECIPE_ID,
        "source": {
            "base_artifact": str(preflight.base_artifact.resolve()),
            "dflash": {
                "repository": dflash2.DFLASH2_REPOSITORY,
                "revision": dflash2.DFLASH2_REVISION,
                "model_py_commit": dflash2.DFLASH2_MODEL_PY_COMMIT,
                "model_path": str(preflight.dflash_model.resolve()),
                "format": dflash_format,
            },
        },
        "dflash_config": preflight.dflash_config,
        "converter": {
            "revision": family_conversion.converter_revision(_repo_root()),
            "environment": family_conversion.environment(resolved_device),
        },
        "elapsed_seconds": time.perf_counter() - started,
        "artifact": {
            "path": str(output.resolve()),
            "bytes": output.stat().st_size,
        },
    }
    report_path = Path(str(output) + ".conversion.json")
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    print(f"wrote {output} ({output.stat().st_size} bytes) in {report['elapsed_seconds']:.1f}s")
    return report_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-artifact", required=True, type=Path)
    parser.add_argument("--dflash-model", required=True, type=Path)
    parser.add_argument("--dflash-format", choices=("w8", "q4"), default="w8")
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    convert(
        args.base_artifact,
        args.dflash_model,
        args.out,
        dflash_format=args.dflash_format,
        device=args.device,
    )


if __name__ == "__main__":
    main()
