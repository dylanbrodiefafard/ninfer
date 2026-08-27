"""Rewrite the five MTP parents to NVFP4 from Qwen3.8-27B BF16.

Copies a `qwen3.8-27b/nvfp4` artifact and replaces only `mtp/` Linear parents.
Text / Vision / heads / norms / object order stay identical. Identity stays
`qwen3.8-27b/nvfp4`. Encoder is `DFLASH2_NVFP4_MAXABS_TWOLEVEL_V2`.

    python3 -m tools.convert.qwen3_8_27b.convert_mtp_nvfp4 \
      --base-artifact /ssdpool2nvme/local_llm/models/qwen3.8-nvfp4-Osfralla-mtp-ninfer/qwen3_8_27b_nvfp4.ninfer \
      --model /ssdpool2nvme/local_llm/models/qwen3.8-27b-bf16 \
      --out /ssdpool2nvme/local_llm/models/qwen3.8-nvfp4-mtp-nvfp4-from-bf16/qwen3_8_27b_nvfp4.ninfer \
      --device cuda
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
import time
from typing import Iterator

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
from tools.convert.common.nvfp4_quantize import ENCODER_PROFILE, encode_nvfp4_from_bf16
from tools.convert.common.quantize import pick_device
from tools.convert.common.safetensors import ShardReader
from tools.convert.qwen3_6.common import conversion as family_conversion
from tools.convert.qwen3_6.common.recipe import materialize_recipe, preflight_source_reader
from tools.convert.qwen3_6_27b import convert as qwen3_6_convert
from tools.convert.qwen3_6_27b import recipe
from tools.convert.qwen3_6_27b.inventory import W8
from tools.convert.qwen3_8_27b.inventory_dflash2 import BLOCK_SCALE_LAYOUT, NVFP4


RECIPE_ID = "qwen3_8_27b_nvfp4_mtp_nvfp4-v2"
PINNED_REPO = "Qwen/Qwen3.8-27B"
PINNED_REVISION = "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0"
_COPY_CHUNK = 64 << 20

MTP_NVFP4_PARENTS: tuple[tuple[str, tuple[int, int]], ...] = (
    ("mtp/input_projection", (5120, 10240)),
    ("mtp/layer/attention/query_key_gate_value", (14336, 5120)),
    ("mtp/layer/attention/output", (5120, 6144)),
    ("mtp/layer/mlp/gate_up", (34816, 5120)),
    ("mtp/layer/mlp/down", (5120, 17408)),
)
_MTP_NAMES = {name for name, _ in MTP_NVFP4_PARENTS}
_MTP_RECIPES = tuple(recipe.RECIPES_BY_NAME[name] for name, _ in MTP_NVFP4_PARENTS)


@dataclass(frozen=True, slots=True)
class ConversionPreflight:
    base_artifact: Path
    model_dir: Path
    identity: ArtifactIdentity
    config_summary: dict[str, object]


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


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
            if obj.name in _MTP_NAMES:
                specs.append(
                    ArtifactTensorSpec(obj.name, obj.shape, NVFP4, BLOCK_SCALE_LAYOUT)
                )
            else:
                specs.append(
                    ArtifactTensorSpec(obj.name, obj.shape, obj.format, obj.layout)
                )
        elif isinstance(obj, ResourceObject):
            specs.append(ArtifactResourceSpec(obj.name, obj.encoding, obj.bytes))
        else:
            raise TypeError(f"unsupported artifact object: {type(obj)!r}")
    return specs


def _payload_chunks(view: memoryview) -> Iterator[memoryview]:
    blob = view.cast("B")
    for begin in range(0, len(blob), _COPY_CHUNK):
        yield blob[begin : begin + _COPY_CHUNK]


def preflight_conversion(
    base_artifact: str | Path, model_dir: str | Path
) -> ConversionPreflight:
    base = Path(base_artifact)
    model = Path(model_dir)
    config_summary = qwen3_6_convert.validate_config(
        family_conversion.load_json(model / "config.json")
    )
    with Artifact.open(base) as src:
        _require_qwen38_nvfp4(src.identity)
        by_name = {obj.name: obj for obj in src.objects}
        for name, shape in MTP_NVFP4_PARENTS:
            obj = by_name.get(name)
            if not isinstance(obj, TensorObject):
                raise ValueError(f"{base}: missing MTP parent {name}")
            if tuple(int(dim) for dim in obj.shape) != shape:
                raise ValueError(f"{name}: shape {tuple(obj.shape)} != {shape}")
            if obj.format == NVFP4:
                if obj.layout != BLOCK_SCALE_LAYOUT:
                    raise ValueError(
                        f"{name}: NVFP4 parent must use {BLOCK_SCALE_LAYOUT}"
                    )
            elif obj.format != W8:
                raise ValueError(f"{name}: expected {W8} or {NVFP4}, got {obj.format}")
            identity = src.identity
    with ShardReader(model) as reader:
        preflight_source_reader(reader, _MTP_RECIPES)
    return ConversionPreflight(
        base_artifact=base,
        model_dir=model,
        identity=identity,
        config_summary=config_summary,
    )


def convert(
    base_artifact: str | Path,
    model_dir: str | Path,
    out_path: str | Path,
    *,
    device: str | torch.device = "cuda",
) -> Path:
    started = time.perf_counter()
    output = Path(out_path)
    resolved_device = pick_device(device)
    preflight = preflight_conversion(base_artifact, model_dir)
    output.parent.mkdir(parents=True, exist_ok=True)
    tmp = output.with_name(output.name + ".tmp")
    if tmp.exists():
        tmp.unlink()

    print(
        f"preflight complete: copy {preflight.base_artifact}, encode "
        f"{len(MTP_NVFP4_PARENTS)} MTP matrices from {preflight.model_dir} "
        f"BF16, device={resolved_device}",
        flush=True,
    )

    with Artifact.open(preflight.base_artifact) as src, ShardReader(
        preflight.model_dir
    ) as reader:
        specs = _artifact_specs(src)
        with ArtifactWriter(tmp, src.identity, specs) as writer:
            for index, obj in enumerate(src.objects, start=1):
                if isinstance(obj, TensorObject) and obj.name in _MTP_NAMES:
                    bf16 = materialize_recipe(recipe.RECIPES_BY_NAME[obj.name], reader)
                    payload = encode_nvfp4_from_bf16(
                        bf16.to(device=resolved_device), obj.shape
                    )
                    del bf16
                    writer.write(obj.name, payload)
                    del payload
                    print(f"[{index}/{len(specs)}] {obj.name} (BF16→NVFP4)", flush=True)
                    continue
                writer.write(obj.name, _payload_chunks(src.payload(obj)))
                print(f"[{index}/{len(specs)}] {obj.name} (copy)", flush=True)

    tmp.replace(output)
    report = {
        "identity": {
            "model_id": preflight.identity.model_id,
            "weights_id": preflight.identity.weights_id,
        },
        "target_key": "qwen3_8_27b",
        "recipe_id": RECIPE_ID,
        "source": {
            "base_artifact": str(preflight.base_artifact.resolve()),
            "mtp_nvfp4": {
                "encoder": ENCODER_PROFILE,
                "repository": PINNED_REPO,
                "revision": PINNED_REVISION,
                "model_path": str(preflight.model_dir.resolve()),
                "rewritten": [name for name, _ in MTP_NVFP4_PARENTS],
            },
        },
        "config_summary": preflight.config_summary,
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
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    convert(
        args.base_artifact, args.model, args.out, device=args.device
    )


if __name__ == "__main__":
    main()
