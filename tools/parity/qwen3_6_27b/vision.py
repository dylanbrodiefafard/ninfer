"""Exhaustive Qwen3.8-27B/NVFP4 Vision intermediate validation."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import subprocess
import tempfile
from typing import Callable

import torch
from safetensors import safe_open

from tools.artifact import dequantize_row_split
from tools.artifact.numeric import QUANT_FORMATS
from tools.reference.qwen3_6.common.frontend import Frontend
from tools.reference.qwen3_6.common.multimodal import MultimodalBatch, load_messages
from tools.reference.qwen3_6.common.vision_ops import (
    add_bias,
    apply_vision_rope,
    bilinear_indices_and_weights,
    gelu,
    interpolate_position_embedding,
    layer_norm,
    vision_attention,
    vision_cu_seqlens,
    vision_position_ids,
)
from tools.reference.qwen3_6_27b.bindings import VisionArtifactBinding
from tools.reference.qwen3_6_27b.config import VISION_CFG
from tools.reference.qwen3_6_27b.ops import linear, residual_add
from tools.reference.qwen3_6_27b.vision import VisionEncoder
from tools.reference.qwen3_6_27b.weights import WeightStore


# The cross-runtime profile permits independent BF16 reduction association at every public Op.
# sqrt(54) times the linear A16 gross allowance (two residual branches in each of 27 blocks), plus
# the merger, is about 6.2%; cosine additionally catches coherent directional drift that a
# norm-only bound can miss. This is a schedule diagnostic criterion, not a replacement for the
# tighter independent FP32/FP64 criteria in each C++ Op test.
PRODUCTION_RELATIVE_RMSE_LIMIT = 7.0e-2
PRODUCTION_COSINE_MINIMUM = 0.998
PRODUCTION_FINAL_RELATIVE_RMSE_LIMIT = 5.0e-2
# Per-token and per-feature gates prevent real-shape averaging from hiding a dropped row or a
# bad strided column. Each group uses its own reference RMS with a 5% whole-tensor activity floor,
# so near-dead groups do not create unbounded ratios. The limits remain above normal BF16
# association drift while a replaced or zeroed material group has scaled RMSE near one.
LOCAL_RELATIVE_RMSE_LIMIT = 6.0e-3
LOCAL_COSINE_MINIMUM = 0.99998
LOCAL_GROUP_SCALED_RMSE_LIMIT = 1.6e-2
LOCAL_GROUP_COSINE_MINIMUM = 0.99996
# Independent FFmpeg/OpenCV decoding may differ by one source-pixel code. The relative full-image
# bound prevents that local allowance from hiding a systematic resize or normalization mismatch.
PREPROCESSING_ABSOLUTE_LIMIT = 7.844e-3
PREPROCESSING_RELATIVE_RMSE_LIMIT = 1.0e-4
POSITION_WEIGHT_ABSOLUTE_LIMIT = 6.0e-6
# Q4/Q5 matrices appear twice per residual block. A root-sum-square accumulation of a 5% Q4
# weight profile across 54 branches is about 37%. The final merger is materially tighter because
# LayerNorm bounds its input and both merger matrices are W8.
SOURCE_RELATIVE_RMSE_LIMIT = 3.8e-1
SOURCE_COSINE_MINIMUM = 0.93
SOURCE_FINAL_RELATIVE_RMSE_LIMIT = 2.5e-1
SOURCE_FINAL_COSINE_MINIMUM = 0.97
GROUP_ACTIVITY_FLOOR = 5.0e-2
WEIGHT_CRITERIA = {
    "Q4G64_F16S": {
        "relative_rmse": 2.2e-1,
        "cosine": 0.975,
        "row": 4.0e-1,
        "column": 8.0e-1,
        "stored_group": 6.0e-1,
        "stored_group_cosine": 0.8,
    },
    "Q5G64_F16S": {
        "relative_rmse": 8.0e-2,
        "cosine": 0.996,
        "row": 2.0e-1,
        "column": 9.0e-1,
        "stored_group": 3.0e-1,
        "stored_group_cosine": 0.9,
    },
    "Q6G64_F16S": {
        "relative_rmse": 4.0e-2,
        "cosine": 0.999,
        "row": 8.0e-2,
        "column": 8.0e-2,
        "stored_group": 1.5e-1,
        "stored_group_cosine": 0.97,
    },
    "W8G32_F16S": {
        "relative_rmse": 1.0e-2,
        "cosine": 0.9999,
        "row": 1.5e-2,
        "column": 1.5e-2,
        "stored_group": 3.0e-2,
        "stored_group_cosine": 0.995,
    },
}


def _expected_trace_names() -> tuple[str, ...]:
    names = [
        "input/patch_f32",
        "input/patch_bf16",
        "patch/linear",
        "patch/bias",
        "patch/position",
    ]
    block_names = (
        "norm1",
        "qkv_linear",
        "qkv_bias",
        "q_rope",
        "k_rope",
        "value",
        "attention",
        "projection_linear",
        "projection_bias",
        "attention_residual",
        "norm2",
        "fc1_linear",
        "fc1_bias",
        "gelu",
        "fc2_linear",
        "fc2_bias",
        "mlp_residual",
    )
    for index in range(VISION_CFG.depth):
        names.extend(f"block_{index:02d}/{name}" for name in block_names)
    names.extend(
        (
            "merger/norm",
            "merger/grouped",
            "merger/fc1_linear",
            "merger/fc1_bias",
            "merger/gelu",
            "merger/fc2_linear",
            "merger/fc2_bias",
        )
    )
    return tuple(names)


EXPECTED_TRACE_NAMES = _expected_trace_names()
assert len(EXPECTED_TRACE_NAMES) == 471


def load_hf_vision(model_dir: Path):
    from transformers import AutoConfig, AutoModel

    config = AutoConfig.from_pretrained(model_dir, local_files_only=True)
    model = AutoModel.from_config(config.vision_config).to(dtype=torch.bfloat16)
    weight_map = json.loads((model_dir / "model.safetensors.index.json").read_text())["weight_map"]
    shards: dict[str, list[str]] = {}
    for name, shard in weight_map.items():
        if name.startswith("model.visual."):
            shards.setdefault(shard, []).append(name)
    state = {}
    for shard, names in sorted(shards.items()):
        with safe_open(model_dir / shard, framework="pt", device="cpu") as source:
            for name in names:
                state[name.removeprefix("model.visual.")] = source.get_tensor(name)
    missing, unexpected = model.load_state_dict(state)
    if missing or unexpected:
        raise RuntimeError(f"HF vision state mismatch: missing={missing}, unexpected={unexpected}")
    return model


def _vision_weight_pairs(binding: VisionArtifactBinding, model):
    vision = binding.vision
    pairs = [
        (vision.patch_embedding, model.patch_embed.proj.weight),
        (vision.patch_embedding_bias, model.patch_embed.proj.bias),
        (vision.position_embedding, model.pos_embed.weight),
    ]
    for bound, source in zip(vision.layers, model.blocks, strict=True):
        pairs.extend(
            (
                (bound.attention_qkv, source.attn.qkv.weight),
                (bound.attention_qkv_bias, source.attn.qkv.bias),
                (bound.attention_output, source.attn.proj.weight),
                (bound.attention_output_bias, source.attn.proj.bias),
                (bound.mlp_fc1, source.mlp.linear_fc1.weight),
                (bound.mlp_fc1_bias, source.mlp.linear_fc1.bias),
                (bound.mlp_fc2, source.mlp.linear_fc2.weight),
                (bound.mlp_fc2_bias, source.mlp.linear_fc2.bias),
                (bound.norm1_weight, source.norm1.weight),
                (bound.norm1_bias, source.norm1.bias),
                (bound.norm2_weight, source.norm2.weight),
                (bound.norm2_bias, source.norm2.bias),
            )
        )
    merger = vision.merger
    pairs.extend(
        (
            (merger.fc1, model.merger.linear_fc1.weight),
            (merger.fc1_bias, model.merger.linear_fc1.bias),
            (merger.fc2, model.merger.linear_fc2.weight),
            (merger.fc2_bias, model.merger.linear_fc2.bias),
            (merger.norm_weight, model.merger.norm.weight),
            (merger.norm_bias, model.merger.norm.bias),
        )
    )
    if len(pairs) != 333 or {block.tensor_id for block, _ in pairs} != {
        block.tensor_id for block in binding.tensors
    }:
        raise RuntimeError("source Vision weight mapping does not cover all 333 artifact tensors")
    return pairs


def _vision_weight_report(binding: VisionArtifactBinding, model) -> dict[str, object]:
    direct = []
    quantized = []
    store = WeightStore(
        binding,
        "cpu",
        capacity=1,
        text=False,
        vision=True,
        compile_codec=False,
    )
    try:
        for block, source in _vision_weight_pairs(binding, model):
            actual = (
                store.tensor(block).cpu()
                if block.layout == "contiguous-le-v1"
                else dequantize_row_split(
                    binding.payload(block),
                    block.format,
                    block.shape,
                    device="cpu",
                    dtype=torch.float32,
                    compiled=False,
                )
            )
            expected = source.detach().cpu().reshape(actual.shape)
            name = block.descriptor.name
            if block.layout == "contiguous-le-v1":
                if actual.dtype != torch.bfloat16 or expected.dtype != torch.bfloat16:
                    raise RuntimeError(
                        f"direct Vision weight {name} is not represented as BF16"
                    )
                direct.append(
                    {
                        "name": name,
                        "format": block.format,
                        "shape": list(actual.shape),
                        "exact": bool(
                            torch.equal(
                                actual.contiguous().view(torch.uint16),
                                expected.contiguous().view(torch.uint16),
                            )
                        ),
                    }
                )
                continue
            value = metrics(actual, expected)
            value["rows"] = value.pop("tokens")
            value["columns"] = value.pop("features")
            numeric_format = QUANT_FORMATS[block.format]
            value["stored_groups"] = stored_weight_group_metrics(
                actual, expected, numeric_format.group_size
            )
            criterion = WEIGHT_CRITERIA.get(block.format)
            if criterion is None:
                raise RuntimeError(f"no source-weight criterion for {block.format}")
            value["passed"] = bool(
                value["actual_finite"]
                and value["reference_finite"]
                and value["relative_rmse"] <= criterion["relative_rmse"]
                and value["cosine"] >= criterion["cosine"]
                and value["rows"]["worst_scaled_rmse"] <= criterion["row"]
                and value["columns"]["worst_scaled_rmse"] <= criterion["column"]
                and value["stored_groups"]["worst_scaled_rmse"]
                <= criterion["stored_group"]
                and value["stored_groups"]["worst_cosine"]
                >= criterion["stored_group_cosine"]
            )
            quantized.append(
                {"name": name, "format": block.format, **value}
            )
    finally:
        store.close()

    direct_failures = [value["name"] for value in direct if not value["exact"]]
    quantized_failures = [value["name"] for value in quantized if not value["passed"]]
    by_format = {}
    for format_name in WEIGHT_CRITERIA:
        values = [value for value in quantized if value["format"] == format_name]
        if not values:
            continue
        by_format[format_name] = {
            "count": len(values),
            "worst_relative_rmse": max(values, key=lambda value: value["relative_rmse"])[
                "relative_rmse"
            ],
            "worst_cosine": min(values, key=lambda value: value["cosine"])["cosine"],
            "worst_row_scaled_rmse": max(
                value["rows"]["worst_scaled_rmse"] for value in values
            ),
            "worst_column_scaled_rmse": max(
                value["columns"]["worst_scaled_rmse"] for value in values
            ),
            "worst_stored_group_scaled_rmse": max(
                value["stored_groups"]["worst_scaled_rmse"] for value in values
            ),
            "worst_stored_group_cosine": min(
                value["stored_groups"]["worst_cosine"] for value in values
            ),
        }
    return {
        "passed": not direct_failures and not quantized_failures,
        "criteria": WEIGHT_CRITERIA,
        "direct": {
            "count": len(direct),
            "all_exact": not direct_failures,
            "failures": direct_failures,
            "comparisons": direct,
        },
        "quantized": {
            "count": len(quantized),
            "failures": quantized_failures,
            "by_format": by_format,
            "comparisons": quantized,
        },
    }


def stored_weight_group_metrics(
    actual: torch.Tensor, expected: torch.Tensor, group_size: int
) -> dict[str, object]:
    """Measure every independently scaled stored K-axis quantization group."""
    if tuple(actual.shape) != tuple(expected.shape) or actual.ndim != 2:
        raise ValueError("stored weight group comparison requires equal rank-2 tensors")
    rows, k = actual.shape
    if group_size <= 0:
        raise ValueError("stored group size must be positive")
    padded_k = math.ceil(k / group_size) * group_size
    if padded_k != k:
        padding = padded_k - k
        actual = torch.nn.functional.pad(actual, (0, padding))
        expected = torch.nn.functional.pad(expected, (0, padding))
    af = actual.detach().to(device="cpu", dtype=torch.float64).reshape(
        rows, padded_k // group_size, group_size
    )
    ef = expected.detach().to(device="cpu", dtype=torch.float64).reshape(
        rows, padded_k // group_size, group_size
    )
    diff = af - ef
    group_rmse = diff.square().mean(dim=2).sqrt()
    reference_rms = ef.square().mean().sqrt()
    group_reference_rms = ef.square().mean(dim=2).sqrt()
    group_actual_rms = af.square().mean(dim=2).sqrt()
    activity_floor = max(
        float(reference_rms) * GROUP_ACTIVITY_FLOOR, torch.finfo(torch.float64).tiny
    )
    scaled_rmse = group_rmse / torch.clamp(group_reference_rms, min=activity_floor)
    dot = (af * ef).sum(dim=2)
    group_denominator = group_actual_rms * group_reference_rms * group_size
    group_cosine = torch.ones_like(group_denominator)
    reference_active = group_reference_rms > activity_floor
    actual_active = group_actual_rms > torch.finfo(torch.float64).tiny
    active = reference_active & actual_active
    group_cosine[active] = torch.clamp(
        dot[active] / group_denominator[active], -1.0, 1.0
    )
    group_cosine[reference_active & ~actual_active] = 0.0
    flat_rmse = int(torch.argmax(scaled_rmse))
    flat_cosine = int(torch.argmin(group_cosine))
    groups_per_row = padded_k // group_size

    def location(flat: int) -> dict[str, int]:
        return {"row": flat // groups_per_row, "k_group": flat % groups_per_row}

    return {
        "group_size": group_size,
        "logical_k": k,
        "padded_k": padded_k,
        "count": rows * groups_per_row,
        "worst_scaled_rmse": float(scaled_rmse.flatten()[flat_rmse]),
        "worst_scaled_rmse_location": location(flat_rmse),
        "worst_scaled_rmse_reference_rms": float(
            group_reference_rms.flatten()[flat_rmse]
        ),
        "worst_scaled_rmse_actual_rms": float(group_actual_rms.flatten()[flat_rmse]),
        "worst_cosine": float(group_cosine.flatten()[flat_cosine]),
        "worst_cosine_location": location(flat_cosine),
        "worst_cosine_reference_rms": float(
            group_reference_rms.flatten()[flat_cosine]
        ),
        "worst_cosine_actual_rms": float(group_actual_rms.flatten()[flat_cosine]),
    }


def metrics(actual: torch.Tensor, expected: torch.Tensor) -> dict[str, object]:
    if tuple(actual.shape) != tuple(expected.shape):
        raise ValueError(
            f"comparison shape mismatch: actual={tuple(actual.shape)} "
            f"expected={tuple(expected.shape)}"
        )
    af = actual.detach().to(device="cpu", dtype=torch.float64)
    ef = expected.detach().to(device="cpu", dtype=torch.float64)
    diff = af - ef
    rmse = diff.square().mean().sqrt()
    reference_rms = ef.square().mean().sqrt()
    actual_norm = af.norm()
    reference_norm = ef.norm()
    denominator = actual_norm * reference_norm
    cosine = torch.tensor(1.0, dtype=torch.float64)
    if float(denominator) != 0.0:
        cosine = torch.clamp(torch.dot(af.flatten(), ef.flatten()) / denominator, -1.0, 1.0)
    relative_rmse = rmse / reference_rms if float(reference_rms) != 0.0 else rmse

    matrix_actual = af.reshape(af.shape[0], -1)
    matrix_expected = ef.reshape(ef.shape[0], -1)
    matrix_diff = diff.reshape(diff.shape[0], -1)

    def grouped(axis: int) -> dict[str, object]:
        reduce = 1 - axis
        count = matrix_expected.shape[reduce]
        group_rmse = matrix_diff.square().mean(dim=reduce).sqrt()
        group_reference_rms = matrix_expected.square().mean(dim=reduce).sqrt()
        group_actual_rms = matrix_actual.square().mean(dim=reduce).sqrt()
        scale = max(float(reference_rms), torch.finfo(torch.float64).tiny)
        activity_floor = scale * GROUP_ACTIVITY_FLOOR
        scaled_rmse = group_rmse / torch.clamp(group_reference_rms, min=activity_floor)
        actual_group_norm = group_actual_rms * math.sqrt(count)
        reference_group_norm = group_reference_rms * math.sqrt(count)
        dot = (matrix_actual * matrix_expected).sum(dim=reduce)
        group_denominator = actual_group_norm * reference_group_norm
        # Direction is meaningful only when the reference group has material energy. A zeroed
        # material actual group gets cosine zero; inactive reference groups remain neutral and
        # their absolute excursion is handled by the floored scaled-RMSE criterion.
        norm_floor = activity_floor * math.sqrt(count)
        group_cosine = torch.ones_like(group_denominator)
        reference_active = reference_group_norm > norm_floor
        active = reference_active & (actual_group_norm > torch.finfo(torch.float64).tiny)
        missing = reference_active & ~active
        group_cosine[active] = torch.clamp(
            dot[active] / group_denominator[active], -1.0, 1.0
        )
        group_cosine[missing] = 0.0
        worst_rmse = int(torch.argmax(scaled_rmse))
        worst_cosine = int(torch.argmin(group_cosine))
        return {
            "worst_scaled_rmse": float(scaled_rmse[worst_rmse]),
            "worst_scaled_rmse_index": worst_rmse,
            "worst_scaled_rmse_reference_rms": float(group_reference_rms[worst_rmse]),
            "worst_scaled_rmse_actual_rms": float(group_actual_rms[worst_rmse]),
            "worst_cosine": float(group_cosine[worst_cosine]),
            "worst_cosine_index": worst_cosine,
            "worst_cosine_reference_rms": float(group_reference_rms[worst_cosine]),
            "worst_cosine_actual_rms": float(group_actual_rms[worst_cosine]),
        }

    return {
        "shape": list(actual.shape),
        "rmse": float(rmse),
        "relative_rmse": float(relative_rmse),
        "max_absolute": float(diff.abs().max()),
        "cosine": float(cosine),
        "actual_rms": float(af.square().mean().sqrt()),
        "reference_rms": float(reference_rms),
        "actual_finite": bool(torch.isfinite(af).all()),
        "reference_finite": bool(torch.isfinite(ef).all()),
        "tokens": grouped(0),
        "features": grouped(1),
    }


def _read_tensor(root: Path, record: dict[str, object]) -> torch.Tensor:
    dtype = {
        "bf16": torch.bfloat16,
        "fp16": torch.float16,
        "fp32": torch.float32,
        "i32": torch.int32,
    }[str(record["dtype"])]
    payload = bytearray((root / str(record["file"])).read_bytes())
    value = torch.frombuffer(payload, dtype=dtype)
    expected = math.prod(record["shape"])
    if value.numel() != expected:
        raise ValueError(
            f"trace tensor {record['file']} has {value.numel()} values; expected {expected}"
        )
    return value.reshape(record["shape"]).clone()


def _write_bf16(path: Path, value: torch.Tensor) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = value.detach().to(device="cpu", dtype=torch.bfloat16).contiguous()
    path.write_bytes(data.view(torch.uint16).numpy().tobytes())


def _read_bf16(path: Path, shape: torch.Size) -> torch.Tensor:
    value = torch.frombuffer(bytearray(path.read_bytes()), dtype=torch.bfloat16)
    return value.reshape(shape).clone()


def _source_forward(
    model,
    pixels: torch.Tensor,
    grid: torch.Tensor,
    tap: Callable[[str, torch.Tensor], None],
) -> torch.Tensor:
    device = next(model.parameters()).device
    pixels = pixels.to(device=device, dtype=torch.float32)
    tap("input/patch_f32", pixels)
    x = pixels.to(torch.bfloat16)
    tap("input/patch_bf16", x)

    patch = model.patch_embed.proj
    x = linear(x, patch.weight.reshape(VISION_CFG.hidden, VISION_CFG.patch_dim))
    tap("patch/linear", x)
    x = add_bias(x, patch.bias)
    tap("patch/bias", x)
    x = residual_add(x, interpolate_position_embedding(model.pos_embed.weight, grid.to(device)))
    tap("patch/position", x)
    pos_ids = vision_position_ids(grid.to(device))
    cu_seqlens = vision_cu_seqlens(grid.to(device))

    for index, block in enumerate(model.blocks):
        prefix = f"block_{index:02d}/"
        h = layer_norm(x, block.norm1.weight, block.norm1.bias)
        tap(prefix + "norm1", h)
        qkv = linear(h, block.attn.qkv.weight)
        tap(prefix + "qkv_linear", qkv)
        qkv = add_bias(qkv, block.attn.qkv.bias)
        tap(prefix + "qkv_bias", qkv)
        qkv = qkv.reshape(-1, 3, VISION_CFG.heads, VISION_CFG.head_dim)
        q, k, v = qkv.unbind(1)
        q, k = apply_vision_rope(q, k, pos_ids)
        tap(prefix + "q_rope", q)
        tap(prefix + "k_rope", k)
        tap(prefix + "value", v)
        attended = vision_attention(q, k, v, cu_seqlens).reshape(-1, VISION_CFG.hidden)
        tap(prefix + "attention", attended)
        projected = linear(attended, block.attn.proj.weight)
        tap(prefix + "projection_linear", projected)
        projected = add_bias(projected, block.attn.proj.bias)
        tap(prefix + "projection_bias", projected)
        x = residual_add(x, projected)
        tap(prefix + "attention_residual", x)
        h = layer_norm(x, block.norm2.weight, block.norm2.bias)
        tap(prefix + "norm2", h)
        h = linear(h, block.mlp.linear_fc1.weight)
        tap(prefix + "fc1_linear", h)
        h = add_bias(h, block.mlp.linear_fc1.bias)
        tap(prefix + "fc1_bias", h)
        h = gelu(h, approximate=True)
        tap(prefix + "gelu", h)
        h = linear(h, block.mlp.linear_fc2.weight)
        tap(prefix + "fc2_linear", h)
        h = add_bias(h, block.mlp.linear_fc2.bias)
        tap(prefix + "fc2_bias", h)
        x = residual_add(x, h)
        tap(prefix + "mlp_residual", x)

    merger = model.merger
    x = layer_norm(x, merger.norm.weight, merger.norm.bias)
    tap("merger/norm", x)
    x = x.reshape(-1, VISION_CFG.merger_hidden)
    tap("merger/grouped", x)
    x = linear(x, merger.linear_fc1.weight)
    tap("merger/fc1_linear", x)
    x = add_bias(x, merger.linear_fc1.bias)
    tap("merger/fc1_bias", x)
    x = gelu(x, approximate=False)
    tap("merger/gelu", x)
    x = linear(x, merger.linear_fc2.weight)
    tap("merger/fc2_linear", x)
    x = add_bias(x, merger.linear_fc2.bias)
    tap("merger/fc2_bias", x)
    return x


def _artifact_local_oracles(
    encoder: VisionEncoder,
    trace_root: Path,
    records: dict[tuple[int, str], dict[str, object]],
    item: int,
    pixels: torch.Tensor,
    control: dict[str, object],
) -> list[dict[str, object]]:
    """Check each production boundary from its captured production inputs.

    This deliberately does not carry the Python result into the next operation. Each logical Op
    receives the C++ tensor from the immediately preceding boundary, so a discrepancy is charged
    to the Op that creates it instead of being amplified through the remaining tower.
    """

    comparisons: list[dict[str, object]] = []
    device = encoder.device

    def captured(name: str) -> torch.Tensor:
        return _read_tensor(trace_root, records[(item, name)]).to(device=device)

    def check(name: str, expected: torch.Tensor) -> torch.Tensor:
        actual = captured(name)
        comparisons.append({"item": item, "name": name, **metrics(actual, expected)})
        return actual

    vision = encoder.binding.vision
    actual_pixels = check(
        "input/patch_f32", pixels.to(device=device, dtype=torch.float32)
    )
    x = check("input/patch_bf16", actual_pixels.to(torch.bfloat16))
    x = check("patch/linear", linear(x, encoder._weight(vision.patch_embedding)))
    x = check(
        "patch/bias", add_bias(x, encoder._weight(vision.patch_embedding_bias))
    )
    patch_count = pixels.shape[0]
    position_indices = torch.tensor(
        control["position_table_indices"], device=device, dtype=torch.long
    ).reshape(patch_count, 4)
    position_weights = torch.tensor(
        control["position_table_weights"], device=device, dtype=torch.float32
    ).reshape(patch_count, 4)
    position_table = encoder._weight(vision.position_embedding)
    gathered = position_table.index_select(0, position_indices.flatten()).reshape(
        patch_count, 4, position_table.shape[-1]
    )
    position = (gathered.float() * position_weights[:, :, None]).sum(1).to(torch.bfloat16)
    x = check("patch/position", residual_add(x, position))
    pos_ids = torch.tensor(control["position_ids"], device=device, dtype=torch.long).reshape(
        2, patch_count
    ).t()
    cu_seqlens = torch.tensor(control["cu_seqlens"], device=device, dtype=torch.int32)

    for layer in vision.layers:
        prefix = f"block_{layer.index:02d}/"
        h = check(
            prefix + "norm1",
            layer_norm(
                x,
                encoder._weight(layer.norm1_weight),
                encoder._weight(layer.norm1_bias),
            ),
        )
        qkv = check(
            prefix + "qkv_linear", linear(h, encoder._weight(layer.attention_qkv))
        )
        qkv = check(
            prefix + "qkv_bias",
            add_bias(qkv, encoder._weight(layer.attention_qkv_bias)),
        )
        qkv_view = qkv.reshape(-1, 3, VISION_CFG.heads, VISION_CFG.head_dim)
        q_input, k_input, v_input = qkv_view.unbind(1)
        q_expected, k_expected = apply_vision_rope(q_input, k_input, pos_ids)
        q = check(prefix + "q_rope", q_expected)
        k = check(prefix + "k_rope", k_expected)
        v = check(prefix + "value", v_input)
        attended = check(
            prefix + "attention",
            vision_attention(q, k, v, cu_seqlens).reshape(-1, VISION_CFG.hidden),
        )
        projected = check(
            prefix + "projection_linear",
            linear(attended, encoder._weight(layer.attention_output)),
        )
        projected = check(
            prefix + "projection_bias",
            add_bias(projected, encoder._weight(layer.attention_output_bias)),
        )
        x = check(prefix + "attention_residual", residual_add(x, projected))
        h = check(
            prefix + "norm2",
            layer_norm(
                x,
                encoder._weight(layer.norm2_weight),
                encoder._weight(layer.norm2_bias),
            ),
        )
        h = check(prefix + "fc1_linear", linear(h, encoder._weight(layer.mlp_fc1)))
        h = check(
            prefix + "fc1_bias", add_bias(h, encoder._weight(layer.mlp_fc1_bias))
        )
        h = check(prefix + "gelu", gelu(h, approximate=True))
        h = check(prefix + "fc2_linear", linear(h, encoder._weight(layer.mlp_fc2)))
        h = check(
            prefix + "fc2_bias", add_bias(h, encoder._weight(layer.mlp_fc2_bias))
        )
        x = check(prefix + "mlp_residual", residual_add(x, h))

    merger = vision.merger
    x = check(
        "merger/norm",
        layer_norm(
            x,
            encoder._weight(merger.norm_weight),
            encoder._weight(merger.norm_bias),
        ),
    )
    x = check("merger/grouped", x.reshape(-1, VISION_CFG.merger_hidden))
    x = check("merger/fc1_linear", linear(x, encoder._weight(merger.fc1)))
    x = check("merger/fc1_bias", add_bias(x, encoder._weight(merger.fc1_bias)))
    x = check("merger/gelu", gelu(x, approximate=False))
    x = check("merger/fc2_linear", linear(x, encoder._weight(merger.fc2)))
    check("merger/fc2_bias", add_bias(x, encoder._weight(merger.fc2_bias)))
    actual_names = tuple(value["name"] for value in comparisons)
    if actual_names != EXPECTED_TRACE_NAMES:
        raise RuntimeError(
            f"local Vision oracle item {item} boundary order/domain mismatch: "
            f"count={len(actual_names)}, expected={len(EXPECTED_TRACE_NAMES)}"
        )
    return comparisons


def _record_ordered_tap(names: list[str], name: str, *, item: int, label: str) -> None:
    index = len(names)
    if index >= len(EXPECTED_TRACE_NAMES) or name != EXPECTED_TRACE_NAMES[index]:
        expected = None if index >= len(EXPECTED_TRACE_NAMES) else EXPECTED_TRACE_NAMES[index]
        raise RuntimeError(
            f"{label} item {item} boundary {index} is {name!r}; expected {expected!r}"
        )
    names.append(name)


def _finish_ordered_taps(names: list[str], *, item: int, label: str) -> None:
    if tuple(names) != EXPECTED_TRACE_NAMES:
        raise RuntimeError(
            f"{label} item {item} produced {len(names)} boundaries; "
            f"expected {len(EXPECTED_TRACE_NAMES)}"
        )


def _split_frontend_items(
    batch: MultimodalBatch, items: list[dict[str, object]]
) -> tuple[list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]], dict[str, int]]:
    image_patch = video_patch = image_grid = video_grid = 0
    image_indices = torch.nonzero(batch.mm_token_type_ids == 1).flatten()
    video_indices = torch.nonzero(batch.mm_token_type_ids == 2).flatten()
    image_scatter = video_scatter = 0
    result = []
    for item in items:
        patches = int(item["patch_count"])
        merged = int(item["merged_count"])
        modality = item["modality"]
        if modality == "image":
            if batch.pixel_values is None or batch.image_grid_thw is None:
                raise ValueError("trace contains an image item but the Python frontend does not")
            if image_grid >= batch.image_grid_thw.shape[0]:
                raise ValueError("trace contains more image items than the Python frontend")
            values = batch.pixel_values[image_patch : image_patch + patches]
            grid = batch.image_grid_thw[image_grid]
            scatter = image_indices[image_scatter : image_scatter + merged]
            image_patch += patches
            image_grid += 1
            image_scatter += merged
        elif modality == "video":
            if batch.pixel_values_videos is None or batch.video_grid_thw is None:
                raise ValueError("trace contains a video item but the Python frontend does not")
            if video_grid >= batch.video_grid_thw.shape[0]:
                raise ValueError("trace contains more video items than the Python frontend")
            values = batch.pixel_values_videos[video_patch : video_patch + patches]
            grid = batch.video_grid_thw[video_grid]
            scatter = video_indices[video_scatter : video_scatter + merged]
            video_patch += patches
            video_grid += 1
            video_scatter += merged
        else:
            raise ValueError(f"trace has unsupported media modality {modality!r}")
        expected_patches = int(torch.prod(grid).item())
        expected_merged = expected_patches // (VISION_CFG.spatial_merge**2)
        if patches != expected_patches or values.shape[0] != patches:
            raise ValueError(
                f"{modality} trace item patch count {patches} does not match grid/frontend "
                f"count {expected_patches}/{values.shape[0]}"
            )
        if merged != expected_merged or scatter.numel() != merged:
            raise ValueError(
                f"{modality} trace item merged count {merged} does not match grid/frontend "
                f"count {expected_merged}/{scatter.numel()}"
            )
        result.append((values, grid, scatter))

    available_image_patches = 0 if batch.pixel_values is None else batch.pixel_values.shape[0]
    available_video_patches = (
        0 if batch.pixel_values_videos is None else batch.pixel_values_videos.shape[0]
    )
    available_image_grids = 0 if batch.image_grid_thw is None else batch.image_grid_thw.shape[0]
    available_video_grids = 0 if batch.video_grid_thw is None else batch.video_grid_thw.shape[0]
    consumed = {
        "items": len(result),
        "image_grids": image_grid,
        "video_grids": video_grid,
        "image_patches": image_patch,
        "video_patches": video_patch,
        "image_scatter": image_scatter,
        "video_scatter": video_scatter,
    }
    available = {
        "items": available_image_grids + available_video_grids,
        "image_grids": available_image_grids,
        "video_grids": available_video_grids,
        "image_patches": available_image_patches,
        "video_patches": available_video_patches,
        "image_scatter": image_indices.numel(),
        "video_scatter": video_indices.numel(),
    }
    if consumed != available:
        raise ValueError(
            f"trace media inventory does not exhaust the Python frontend: "
            f"consumed={consumed}, available={available}"
        )
    return result, consumed


def _preprocessing_report(
    binding: VisionArtifactBinding,
    manifest: dict[str, object],
    trace_root: Path,
    records: dict[tuple[int, str], dict[str, object]],
) -> tuple[dict[str, object], list[tuple[torch.Tensor, torch.Tensor]]]:
    frontend = Frontend(binding)
    batch = frontend.process(load_messages(manifest["messages"]), thinking=False)
    exact = {
        "token_ids": manifest["token_ids"] == batch.input_ids.tolist(),
        "token_types": manifest["token_types"] == batch.mm_token_type_ids.tolist(),
        "positions": manifest["positions"] == batch.position_ids.flatten().tolist(),
        "rope_delta": int(manifest["rope_delta"]) == batch.rope_delta,
        "aggregate_matches_individual": bool(manifest["aggregate_matches_individual"]),
    }
    item_inputs, inventory = _split_frontend_items(batch, manifest["items"])
    item_reports = []
    encoded_inputs: list[tuple[torch.Tensor, torch.Tensor]] = []
    for index, (pixels, grid, scatter) in enumerate(item_inputs):
        item = manifest["items"][index]
        cpp_pixels = _read_tensor(trace_root, records[(index, "input/patch_f32")])
        patch_metrics = metrics(cpp_pixels, pixels)
        expected_pos = vision_position_ids(grid.unsqueeze(0)).t().flatten().tolist()
        expected_cu = vision_cu_seqlens(grid.unsqueeze(0)).tolist()
        interp_indices, interp_weights = bilinear_indices_and_weights(grid.unsqueeze(0))
        controls_exact = {
            "grid": item["grid"] == grid.tolist(),
            "position_ids": item["position_ids"] == expected_pos,
            "cu_seqlens": item["cu_seqlens"] == expected_cu,
            "scatter_indices": item["scatter_indices"] == scatter.tolist(),
            "position_table_indices": item["position_table_indices"]
            == interp_indices.t().flatten().tolist(),
        }
        actual_weights = torch.tensor(item["position_table_weights"], dtype=torch.float32)
        expected_weights = interp_weights.t().flatten().float()
        weight_max = float((actual_weights - expected_weights).abs().max())
        item_reports.append(
            {
                "item": index,
                "modality": item["modality"],
                "grid": item["grid"],
                "patches": patch_metrics,
                "controls_exact": controls_exact,
                "position_weight_max_absolute": weight_max,
            }
        )
        encoded_inputs.append((cpp_pixels, torch.tensor([item["grid"]], dtype=torch.long)))
    passed = (
        all(exact.values())
        and all(all(entry["controls_exact"].values()) for entry in item_reports)
        and all(
            entry["patches"]["max_absolute"] <= PREPROCESSING_ABSOLUTE_LIMIT
            for entry in item_reports
        )
        and all(
            entry["patches"]["relative_rmse"] <= PREPROCESSING_RELATIVE_RMSE_LIMIT
            for entry in item_reports
        )
        and all(
            entry["position_weight_max_absolute"] <= POSITION_WEIGHT_ABSOLUTE_LIMIT
            for entry in item_reports
        )
    )
    return {
        "passed": passed,
        "exact": exact,
        "inventory": inventory,
        "items": item_reports,
    }, encoded_inputs


def _summarize(
    comparisons: list[dict[str, object]], *, profile: str
) -> dict[str, object]:
    worst_relative = max(comparisons, key=lambda value: value["relative_rmse"])
    worst_cosine = min(comparisons, key=lambda value: value["cosine"])
    worst_token_rmse = max(
        comparisons, key=lambda value: value["tokens"]["worst_scaled_rmse"]
    )
    worst_token_cosine = min(
        comparisons, key=lambda value: value["tokens"]["worst_cosine"]
    )
    worst_feature_rmse = max(
        comparisons, key=lambda value: value["features"]["worst_scaled_rmse"]
    )
    worst_feature_cosine = min(
        comparisons, key=lambda value: value["features"]["worst_cosine"]
    )
    nonfinite = [
        value["name"]
        for value in comparisons
        if not value["actual_finite"] or not value["reference_finite"]
    ]
    result = {
        "count": len(comparisons),
        "all_finite": not nonfinite,
        "nonfinite": nonfinite,
        "worst_relative_rmse": {
            "item": worst_relative["item"],
            "name": worst_relative["name"],
            "value": worst_relative["relative_rmse"],
        },
        "worst_cosine": {
            "item": worst_cosine["item"],
            "name": worst_cosine["name"],
            "value": worst_cosine["cosine"],
        },
        "worst_token_scaled_rmse": {
            "item": worst_token_rmse["item"],
            "name": worst_token_rmse["name"],
            "index": worst_token_rmse["tokens"]["worst_scaled_rmse_index"],
            "value": worst_token_rmse["tokens"]["worst_scaled_rmse"],
        },
        "worst_token_cosine": {
            "item": worst_token_cosine["item"],
            "name": worst_token_cosine["name"],
            "index": worst_token_cosine["tokens"]["worst_cosine_index"],
            "value": worst_token_cosine["tokens"]["worst_cosine"],
        },
        "worst_feature_scaled_rmse": {
            "item": worst_feature_rmse["item"],
            "name": worst_feature_rmse["name"],
            "index": worst_feature_rmse["features"]["worst_scaled_rmse_index"],
            "value": worst_feature_rmse["features"]["worst_scaled_rmse"],
        },
        "worst_feature_cosine": {
            "item": worst_feature_cosine["item"],
            "name": worst_feature_cosine["name"],
            "index": worst_feature_cosine["features"]["worst_cosine_index"],
            "value": worst_feature_cosine["features"]["worst_cosine"],
        },
        "comparisons": comparisons,
    }
    if profile == "source":
        failures = [
            value
            for value in comparisons
            if value["relative_rmse"] > SOURCE_RELATIVE_RMSE_LIMIT
            or value["cosine"] < SOURCE_COSINE_MINIMUM
        ]
        finals = [value for value in comparisons if value["name"] == "merger/fc2_bias"]
        if len(finals) != len(comparisons) // len(EXPECTED_TRACE_NAMES):
            raise RuntimeError("source schedule final-boundary inventory is incomplete")
        final_failures = [
            value
            for value in finals
            if value["relative_rmse"] > SOURCE_FINAL_RELATIVE_RMSE_LIMIT
            or value["cosine"] < SOURCE_FINAL_COSINE_MINIMUM
        ]
        result["passed"] = (
            not nonfinite
            and not failures
            and not final_failures
        )
        result["criterion_failures"] = [
            {"item": value["item"], "name": value["name"], "criterion": "source_general"}
            for value in failures
        ] + [
            {"item": value["item"], "name": value["name"], "criterion": "source_final"}
            for value in final_failures
        ]
    elif profile == "local":
        failures = [
            value
            for value in comparisons
            if value["relative_rmse"] > LOCAL_RELATIVE_RMSE_LIMIT
            or value["cosine"] < LOCAL_COSINE_MINIMUM
            or value["tokens"]["worst_scaled_rmse"] > LOCAL_GROUP_SCALED_RMSE_LIMIT
            or value["tokens"]["worst_cosine"] < LOCAL_GROUP_COSINE_MINIMUM
            or value["features"]["worst_scaled_rmse"] > LOCAL_GROUP_SCALED_RMSE_LIMIT
            or value["features"]["worst_cosine"] < LOCAL_GROUP_COSINE_MINIMUM
        ]
        result["passed"] = not nonfinite and not failures
        result["criterion_failures"] = [
            {"item": value["item"], "name": value["name"], "criterion": "local_op"}
            for value in failures
        ]
    elif profile == "production":
        failures = [
            value
            for value in comparisons
            if value["relative_rmse"] > PRODUCTION_RELATIVE_RMSE_LIMIT
            or value["cosine"] < PRODUCTION_COSINE_MINIMUM
        ]
        finals = [value for value in comparisons if value["name"] == "merger/fc2_bias"]
        if len(finals) != len(comparisons) // len(EXPECTED_TRACE_NAMES):
            raise RuntimeError("production schedule final-boundary inventory is incomplete")
        final_failures = [
            value
            for value in finals
            if value["relative_rmse"] > PRODUCTION_FINAL_RELATIVE_RMSE_LIMIT
        ]
        result["passed"] = (
            not nonfinite
            and not failures
            and not final_failures
        )
        result["criterion_failures"] = [
            {
                "item": value["item"],
                "name": value["name"],
                "criterion": "production_general",
            }
            for value in failures
        ] + [
            {"item": value["item"], "name": value["name"], "criterion": "production_final"}
            for value in final_failures
        ]
    else:
        raise ValueError(f"unknown comparison profile {profile!r}")
    return result


def run_campaign(args: argparse.Namespace, trace_root: Path) -> dict[str, object]:
    manifest = json.loads((trace_root / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("format") != "ninfer_vision_intermediate_trace_v1":
        raise ValueError("C++ Vision trace has an unsupported format")
    manifest_artifact = Path(str(manifest["artifact"])).resolve()
    requested_artifact = Path(args.weights).resolve()
    manifest_messages = Path(str(manifest["messages"])).resolve()
    requested_messages = Path(args.messages).resolve()
    if manifest_artifact != requested_artifact:
        raise ValueError(
            f"C++ Vision trace artifact {manifest_artifact} does not match "
            f"--weights {requested_artifact}"
        )
    if manifest_messages != requested_messages:
        raise ValueError(
            f"C++ Vision trace messages {manifest_messages} do not match "
            f"--messages {requested_messages}"
        )
    item_count = len(manifest["items"])
    captures = manifest["captures"]
    expected_keys = {
        (item, name) for item in range(item_count) for name in EXPECTED_TRACE_NAMES
    }
    records: dict[tuple[int, str], dict[str, object]] = {}
    for record in captures:
        key = (int(record["item"]), str(record["name"]))
        if key in records:
            raise ValueError(f"C++ Vision trace has duplicate capture {key}")
        records[key] = record
    actual_keys = set(records)
    if len(captures) != 471 * item_count or actual_keys != expected_keys:
        missing = sorted(expected_keys - actual_keys)
        extra = sorted(actual_keys - expected_keys)
        raise ValueError(
            f"C++ Vision trace inventory mismatch: captures={len(captures)}, "
            f"expected={471 * item_count}, missing={missing[:8]}, extra={extra[:8]}"
        )

    with VisionArtifactBinding.open(args.weights) as binding:
        preprocessing, item_inputs = _preprocessing_report(binding, manifest, trace_root, records)
        production_comparisons: list[dict[str, object]] = []
        local_oracle_comparisons: list[dict[str, object]] = []
        with tempfile.TemporaryDirectory(prefix="ninfer-vision-artifact-") as artifact_tmp:
            artifact_root = Path(artifact_tmp)
            with VisionEncoder(
                binding, args.device, compile_codec=True
            ) as encoder, torch.inference_mode():
                for item, (pixels, grid) in enumerate(item_inputs):
                    modality = manifest["items"][item]["modality"]
                    artifact_names: list[str] = []

                    def artifact_tap(name: str, value: torch.Tensor, *, item=item) -> None:
                        _record_ordered_tap(
                            artifact_names, name, item=item, label="artifact schedule"
                        )
                        cpp = _read_tensor(trace_root, records[(item, name)])
                        production_comparisons.append(
                            {"item": item, "name": name, **metrics(cpp, value)}
                        )
                        _write_bf16(
                            artifact_root / f"item_{item:02d}" / (name + ".bf16"), value
                        )

                    encoder.encode(
                        pixels if modality == "image" else None,
                        grid if modality == "image" else None,
                        pixels if modality == "video" else None,
                        grid if modality == "video" else None,
                        tap=artifact_tap,
                    )
                    _finish_ordered_taps(
                        artifact_names, item=item, label="artifact schedule"
                    )
                    local_oracle_comparisons.extend(
                        _artifact_local_oracles(
                            encoder,
                            trace_root,
                            records,
                            item,
                            pixels,
                            manifest["items"][item],
                        )
                    )

            if torch.device(args.device).type == "cuda":
                torch.cuda.empty_cache()
            source = load_hf_vision(Path(args.model_dir))
            weight_report = _vision_weight_report(binding, source)
            source = source.to(device=args.device, dtype=torch.bfloat16).eval()
            source_comparisons: list[dict[str, object]] = []
            source_hf_final = []
            with torch.inference_mode():
                for item, (pixels, grid) in enumerate(item_inputs):
                    source_names: list[str] = []

                    def source_tap(name: str, value: torch.Tensor, *, item=item) -> None:
                        _record_ordered_tap(
                            source_names, name, item=item, label="source schedule"
                        )
                        artifact = _read_bf16(
                            artifact_root / f"item_{item:02d}" / (name + ".bf16"), value.shape
                        )
                        source_comparisons.append(
                            {"item": item, "name": name, **metrics(artifact, value)}
                        )

                    manual = _source_forward(source, pixels, grid.to(args.device), source_tap)
                    _finish_ordered_taps(source_names, item=item, label="source schedule")
                    hf = source(
                        pixels.to(device=args.device, dtype=torch.bfloat16),
                        grid_thw=grid.to(args.device),
                        return_dict=True,
                    ).pooler_output
                    source_hf_final.append({"item": item, **metrics(manual, hf)})

    production = _summarize(production_comparisons, profile="production")
    local_oracles = _summarize(local_oracle_comparisons, profile="local")
    source_drift = _summarize(source_comparisons, profile="source")
    source_hf_passed = all(
        value["relative_rmse"] <= PRODUCTION_RELATIVE_RMSE_LIMIT
        and value["cosine"] >= PRODUCTION_COSINE_MINIMUM
        for value in source_hf_final
    )
    passed = bool(
        preprocessing["passed"]
        and weight_report["passed"]
        and production["passed"]
        and local_oracles["passed"]
        and source_drift["passed"]
        and source_hf_passed
    )
    return {
        "format": "ninfer_vision_intermediate_validation_v2",
        "passed": passed,
        "weights": str(Path(args.weights).resolve()),
        "model_dir": str(Path(args.model_dir).resolve()),
        "messages": str(Path(args.messages).resolve()),
        "criteria": {
            "production_relative_rmse": PRODUCTION_RELATIVE_RMSE_LIMIT,
            "production_cosine": PRODUCTION_COSINE_MINIMUM,
            "production_final_relative_rmse": PRODUCTION_FINAL_RELATIVE_RMSE_LIMIT,
            "local_relative_rmse": LOCAL_RELATIVE_RMSE_LIMIT,
            "local_cosine": LOCAL_COSINE_MINIMUM,
            "local_group_scaled_rmse": LOCAL_GROUP_SCALED_RMSE_LIMIT,
            "local_group_cosine": LOCAL_GROUP_COSINE_MINIMUM,
            "group_activity_floor": GROUP_ACTIVITY_FLOOR,
            "preprocessing_absolute": PREPROCESSING_ABSOLUTE_LIMIT,
            "preprocessing_relative_rmse": PREPROCESSING_RELATIVE_RMSE_LIMIT,
            "position_weight_absolute": POSITION_WEIGHT_ABSOLUTE_LIMIT,
            "source_relative_rmse": SOURCE_RELATIVE_RMSE_LIMIT,
            "source_cosine": SOURCE_COSINE_MINIMUM,
            "source_final_relative_rmse": SOURCE_FINAL_RELATIVE_RMSE_LIMIT,
            "source_final_cosine": SOURCE_FINAL_COSINE_MINIMUM,
        },
        "preprocessing": preprocessing,
        "artifact_weights_vs_source_bf16": weight_report,
        "production_vs_artifact_reference": production,
        "production_local_op_oracles": local_oracles,
        "artifact_vs_source_bf16": source_drift,
        "source_manual_vs_hf_final": {
            "passed": source_hf_passed,
            "comparisons": source_hf_final,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", required=True)
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--messages", required=True)
    parser.add_argument("--trace-exe", required=True)
    parser.add_argument("--trace-dir")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output")
    args = parser.parse_args()

    if args.trace_dir:
        report = run_campaign(args, Path(args.trace_dir))
    else:
        with tempfile.TemporaryDirectory(prefix="ninfer-vision-trace-") as temporary:
            trace_root = Path(temporary)
            subprocess.run(
                [args.trace_exe, args.weights, args.messages, str(trace_root)], check=True
            )
            report = run_campaign(args, trace_root)
    rendered = json.dumps(report, ensure_ascii=False, indent=2)
    if args.output:
        Path(args.output).write_text(rendered + "\n", encoding="utf-8")
    print(rendered)
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
