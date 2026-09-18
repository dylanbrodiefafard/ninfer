"""Byte-preserving offline native Qwen4 layer fixture, not an Engine model."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from safetensors import safe_open

from tools.artifact.container import ArtifactIdentity, ArtifactWriter, TensorSpec
from tools.artifact.layouts import encode_direct, encode_nvfp4_experts


def build(source: Path, output: Path, layer: int) -> None:
    if output.exists():
        raise FileExistsError(output)
    prefix = f"model.language_model.layers.{layer}."
    with safe_open(str(source), framework="pt", device="cpu") as tensors:
        names = list(tensors.keys())
        if len(names) != 6166 or any(not name.startswith(prefix) for name in names):
            raise ValueError("requires complete exact selected main layer")
        ordinary = [name for name in names if ".mlp.experts." not in name]
        specs = []
        for name in ordinary:
            view = tensors.get_slice(name)
            if view.get_dtype() != "BF16":
                raise ValueError(f"unexpected non-expert dtype: {name}")
            specs.append(TensorSpec(name, tuple(view.get_shape()), "BF16", "contiguous-le-v1"))
        for projection, shape in (("gate_proj", (512, 640, 2560)),
                                  ("up_proj", (512, 640, 2560)),
                                  ("down_proj", (512, 2560, 640))):
            specs.append(TensorSpec(prefix + f"mlp.experts.{projection}.weight", shape,
                                    "NVFP4_EXPERT_F32M", "expert-blockscale-k16-m128x4-v1"))
        output.parent.mkdir(parents=True, exist_ok=True)
        with ArtifactWriter(output, ArtifactIdentity("qwen4/native-layer-qualification",
                                                     "nvidia-nvfp4-source"), specs) as writer:
            for name in ordinary:
                writer.write(name, encode_direct(tensors.get_tensor(name), "BF16"))
            for spec in specs[len(ordinary):]:
                projection = spec.name.split(".")[-2]
                codes, scales, weight_multipliers, input_multipliers = [], [], [], []
                for expert in range(512):
                    stem = prefix + f"mlp.experts.{expert}.{projection}."
                    codes.append(tensors.get_tensor(stem + "weight"))
                    scales.append(tensors.get_tensor(stem + "weight_scale").view(torch.uint8))
                    weight_multipliers.append(tensors.get_tensor(stem + "weight_scale_2").reshape(()))
                    input_multipliers.append(tensors.get_tensor(stem + "input_scale").reshape(()))
                payload = encode_nvfp4_experts(torch.stack(codes), torch.stack(scales),
                                               torch.stack(weight_multipliers),
                                               torch.stack(input_multipliers), spec.shape)
                writer.write(spec.name, payload)
                print(spec.name, len(payload), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--layer", choices=(0, 1, 2, 3), type=int, required=True)
    args = parser.parse_args()
    torch.set_num_threads(2)
    build(args.source, args.out, args.layer)
