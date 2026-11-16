"""Source tensors and dense-study records to stored payloads of `qwen3.8-flash-next` objects.

Every transform is an exact move of source words; dense-study payloads are copied word for word.
The validator derives its expected values independently (`validate.py`).
"""

from __future__ import annotations

import struct
import sys

import torch

from tools.artifact.container import TensorSpec
from tools.artifact.layouts import (
    ExpertRecordWords,
    encode_direct,
    encode_expert_tiles_record,
    encode_fp8_row_scaled,
    encode_nvfp4,
)
from tools.convert.common.safetensors import ShardReader

from .inventory import (
    BF16,
    E2M1,
    FP32,
    I32,
    INDEX_QUERY,
    MOE_CONTROL_ROWS,
    QSA_HEAD_DIM,
    QSA_HEADS,
    ObjectPlan,
    Transform,
)
from .study_dense import DenseStudy, fp8_words, load_attention_parent, nvfp4_words
from .study_store import HIDDEN, INTERMEDIATE, MATRIX_SHAPES


def _source(reader: ShardReader, name: str, dtype: torch.dtype) -> torch.Tensor:
    tensor = reader.get(name)
    if tensor.dtype != dtype:
        raise ValueError(f"{name}: expected {dtype}, got {tensor.dtype}")
    return tensor


def split_query_gate(q_proj: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Separate the per-head `[query 256, gate 256]` halves of a `[24 * 512, K]` q_proj."""

    heads = q_proj.reshape(QSA_HEADS, 2, QSA_HEAD_DIM, q_proj.shape[-1])
    return (
        heads[:, 0].reshape(QSA_HEADS * QSA_HEAD_DIM, -1),
        heads[:, 1].reshape(QSA_HEADS * QSA_HEAD_DIM, -1),
    )


def pack_moe_control(router: torch.Tensor, shared_gate: torch.Tensor) -> torch.Tensor:
    """Router rows 0..511, the shared-expert gate row 512, positive-zero rows 513..543."""

    padding = torch.zeros(
        (MOE_CONTROL_ROWS - router.shape[0] - shared_gate.shape[0], router.shape[1]),
        dtype=router.dtype,
    )
    return torch.cat((router, shared_gate, padding), dim=0)


def int64_words(values: torch.Tensor) -> torch.Tensor:
    """Exact little-endian `(low, high)` I32 word pairs of int64 values."""

    if sys.byteorder != "little":
        raise RuntimeError("int64 word split assumes a little-endian host")
    return values.contiguous().view(torch.int32).reshape(-1, 2)


def materialize(plan: ObjectPlan, reader: ShardReader) -> torch.Tensor:
    """Build the logical stored tensor of one dense plan from its source tensors."""

    spec = plan.spec
    assert isinstance(spec, TensorSpec)
    transform = plan.transform
    if transform is Transform.INT64_WORDS:
        tensor = int64_words(_source(reader, plan.sources[0], torch.int64))
    elif transform is Transform.WIDEN_FP32:
        tensor = _source(reader, plan.sources[0], torch.bfloat16).to(torch.float32)
    else:
        sources = [_source(reader, name, torch.bfloat16) for name in plan.sources]
        if transform is Transform.ROWS:
            tensor = sources[0] if len(sources) == 1 else torch.cat(sources, dim=0)
        elif transform is Transform.INDEX_QUERY:
            tensor = sources[0][:INDEX_QUERY]
        elif transform is Transform.INDEX_KEY:
            tensor = sources[0][INDEX_QUERY:]
        elif transform is Transform.MOE_CONTROL:
            tensor = pack_moe_control(sources[0], sources[1])
        elif transform is Transform.CONV_TAP_MAJOR:
            tensor = sources[0].reshape(sources[0].shape[0], -1).t()
        elif transform is Transform.CONV_TAPS_FASTEST:
            tensor = sources[0].reshape(sources[0].shape[0], -1)
        else:
            raise ValueError(f"{spec.name}: transform {transform} is not a dense source transform")
    if tuple(tensor.shape) != spec.shape:
        raise ValueError(f"{spec.name}: source yields {tuple(tensor.shape)}, expected {spec.shape}")
    return tensor.contiguous()


def encode_payload(spec: TensorSpec, tensor: torch.Tensor) -> bytes:
    """The direct-word payload of a checkpoint-derived tensor (BF16, FP32 or I32)."""

    if spec.format not in (BF16, FP32, I32):
        raise ValueError(f"{spec.name}: {spec.format} is not a direct checkpoint format")
    return encode_direct(tensor, spec.format)


def attention_parent_rows() -> torch.Tensor:
    """Checkpoint parent row of each `[query, key, gate, value]` row (`q_proj` de-interleaved).

    A study parent stores `q_proj` (24 heads of `[query 256, gate 256]`), `k_proj`, `v_proj`;
    NVFP4 groups run along K, so permuting whole code and scale rows is exact.
    """

    query, gate = split_query_gate(torch.arange(QSA_HEADS * 2 * QSA_HEAD_DIM).unsqueeze(1))
    key = torch.arange(12288, 12800).unsqueeze(1)
    value = torch.arange(12800, 13312).unsqueeze(1)
    return torch.cat((query, key, gate, value)).reshape(-1)


def encode_study_payload(plan: ObjectPlan, study: DenseStudy) -> bytes:
    """Stored payload of a dense-study object; codes, scales and multipliers are copied exactly."""

    spec = plan.spec
    assert isinstance(spec, TensorSpec)
    transform = plan.transform
    if transform is Transform.STUDY_FP8_ROWS:
        columns = spec.shape[1]
        parts = [study.load(path) for path in plan.sources]
        words = [fp8_words(part, part.codes.shape[0], columns) for part in parts]
        codes = torch.cat([code for code, _ in words])
        scales = torch.cat([scale for _, scale in words])
        del parts, words
        return encode_fp8_row_scaled(codes, scales, spec.shape)
    if transform in (Transform.STUDY_SHARED_EXPERT, Transform.STUDY_SHARED_SCALES):
        gate, up, down = (
            nvfp4_words(study.load(path), rows, columns)
            for path, (rows, columns) in zip(plan.sources, MATRIX_SHAPES, strict=True)
        )
        if transform is Transform.STUDY_SHARED_SCALES:
            words = torch.tensor([gate[1], up[1], down[1]], dtype=torch.int64).to(torch.int32)
            return encode_direct(words.view(torch.float32), FP32)
        record = encode_expert_tiles_record(
            ExpertRecordWords(gate[0], up[0], down[0]), E2M1, HIDDEN, INTERMEDIATE
        )
        return record.numpy().tobytes()
    if transform is Transform.STUDY_ATTENTION_PARENT:
        (path,) = plan.sources
        parent = load_attention_parent(study, path)
        rows = attention_parent_rows()
        return encode_nvfp4(
            torch.from_numpy(parent.codes)[rows],
            torch.from_numpy(parent.scales)[rows],
            struct.pack("<I", parent.divisor_word),
            spec.shape,
        )
    raise ValueError(f"{spec.name}: transform {transform} is not a dense-study transform")
