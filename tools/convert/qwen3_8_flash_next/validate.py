"""Re-read a `qwen3.8-flash-next/f3-fp8-nvfp4` artifact and check every object against its sources.

The expected inventory is rebuilt from the sources and must match the directory exactly (names,
order, kinds, shapes, formats, layouts, lengths). Payload checks derive the expected values
independently of the converter:

- resources equal their source files byte for byte;
- checkpoint-derived direct tensors equal their source words bit for bit, with row partitions
  and the convolution transposes addressed by explicit index formulas;
- every dense-study object decodes to exactly the words of its re-read, fingerprint-checked
  records (the attention-input parent through an explicit artifact-row to source-row formula);
  for sampled rows the decoded values also equal an independent FP64 bit-field decode of the
  study payload, and their relative error against the BF16 checkpoint rows is reported;
- every routed expert record decodes (through the layout's index-formula decoder) to exactly the
  codes and scales of its study record, and the codec map, F3 books and matrix-scale table equal
  the study's words.
"""

from __future__ import annotations

import math
import os
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch

from tools.artifact.container import Artifact, ResourceObject, ResourceSpec, TensorObject
from tools.artifact.layouts import (
    decode_direct,
    decode_expert_tiles_record,
    decode_fp8_row_scaled_words,
    decode_nvfp4_words,
    expert_tiles_geometry,
)
from tools.artifact.numeric import decode_e2m1_word, decode_e4m3fn_word
from tools.convert.common.safetensors import ShardReader

from .inventory import (
    E2M1,
    E2M1_RECORDS,
    EXPERT_CODECS,
    F3,
    F3_CODEBOOKS,
    F3_RECORDS,
    MATRIX_SCALES,
    MODEL_ID,
    QSA_HEAD_DIM,
    QSA_HEADS,
    QSA_KV,
    STUDY_TRANSFORMS,
    WEIGHTS_ID,
    ObjectPlan,
    Transform,
    build_inventory,
)
from .study_dense import DenseStudy, fp8_words, load_attention_parent, nvfp4_words
from .study_store import (
    EXPERTS,
    HIDDEN,
    INTERMEDIATE,
    LAYERS,
    MATRICES,
    MATRIX_SHAPES,
    ExpertCodec,
)

# Rows compared per step; bounds the temporaries of the vocabulary-sized direct tensors.
_ROW_CHUNK = 4096
_E2M1 = torch.tensor([decode_e2m1_word(word) for word in range(16)], dtype=torch.float64)
_E4M3 = torch.tensor(
    [decode_e4m3fn_word(word) if word & 0x7F != 0x7F else math.nan for word in range(256)],
    dtype=torch.float64,
)


class ValidationError(ValueError):
    """The artifact disagrees with its sources."""


@dataclass
class ValidationReport:
    objects: int = 0
    exact_objects: int = 0
    study_cross_checks: list[dict[str, object]] = field(default_factory=list)
    routed_experts: int = 0
    seconds: float = 0.0

    def to_json(self) -> dict[str, object]:
        return {
            "objects_checked": self.objects,
            "exact_objects": self.exact_objects,
            "study_cross_checks": self.study_cross_checks,
            "routed_experts_checked": self.routed_experts,
            "seconds": round(self.seconds, 1),
            "passed": True,
        }


class _PayloadFile:
    """pread access to object payloads; avoids mapping 59 GB into the validator's RSS."""

    def __init__(self, artifact: Artifact) -> None:
        self._fd = os.open(artifact.path, os.O_RDONLY)
        self._base = artifact.payload_offset

    def read(self, obj: TensorObject | ResourceObject, begin: int = 0, length: int | None = None):
        length = obj.bytes - begin if length is None else length
        offset = self._base + obj.offset + begin
        data = bytearray(length)
        view = memoryview(data)
        done = 0
        while done < length:
            count = os.preadv(self._fd, [view[done:]], offset + done)
            if count <= 0:
                raise ValidationError(f"{obj.name}: short read")
            done += count
        os.posix_fadvise(self._fd, offset, length, os.POSIX_FADV_DONTNEED)
        return data

    def close(self) -> None:
        os.close(self._fd)


def _bits(tensor: torch.Tensor) -> torch.Tensor:
    width = {2: torch.int16, 4: torch.int32}[tensor.element_size()]
    return tensor.contiguous().view(width)


def _expected_rows(
    plan: ObjectPlan, loaded: list[torch.Tensor], begin: int, end: int
) -> torch.Tensor:
    """Expected logical rows [begin, end) of a dense plan, addressed by explicit row indices.

    ``loaded`` holds the plan's source tensors in ``plan.sources`` order.
    """

    transform = plan.transform
    if transform is Transform.ROWS:
        parts = loaded
        if parts[0].dim() == 1:
            return torch.cat(parts)[begin:end]
        offsets = np.cumsum([0] + [part.shape[0] for part in parts])
        rows = torch.arange(begin, end)
        part = torch.as_tensor(np.searchsorted(offsets, rows.numpy(), side="right") - 1)
        out = torch.empty((end - begin, parts[0].shape[1]), dtype=parts[0].dtype)
        for index, source in enumerate(parts):
            mask = part == index
            out[mask] = source[rows[mask] - int(offsets[index])]
        return out
    if transform is Transform.QSA_QUERY_KEY_GATE_VALUE:
        q_proj, k_proj, v_proj = loaded
        width = QSA_HEADS * QSA_HEAD_DIM
        rows = torch.arange(begin, end)
        out = torch.empty((end - begin, q_proj.shape[1]), dtype=q_proj.dtype)
        for lo, hi, source, index in (
            (0, width, q_proj, lambda r: (r // QSA_HEAD_DIM) * 512 + r % QSA_HEAD_DIM),
            (width, width + QSA_KV, k_proj, lambda r: r),
            (
                width + QSA_KV,
                2 * width + QSA_KV,
                q_proj,
                lambda r: (r // QSA_HEAD_DIM) * 512 + QSA_HEAD_DIM + r % QSA_HEAD_DIM,
            ),
            (2 * width + QSA_KV, 2 * width + 2 * QSA_KV, v_proj, lambda r: r),
        ):
            mask = (rows >= lo) & (rows < hi)
            out[mask] = source[index(rows[mask] - lo)]
        return out
    source = loaded[0]
    if transform is Transform.INDEX_QUERY:
        return source[begin:end]
    if transform is Transform.INDEX_KEY:
        return source[512 + begin : 512 + end]
    if transform is Transform.MOE_CONTROL:
        shared = loaded[1]
        out = torch.zeros((end - begin, source.shape[1]), dtype=source.dtype)
        for row in range(begin, end):
            if row < EXPERTS:
                out[row - begin] = source[row]
            elif row == EXPERTS:
                out[row - begin] = shared[0]
        return out
    if transform is Transform.CONV_TAP_MAJOR:
        return torch.stack([source[:, 0, tap] for tap in range(4)])[begin:end]
    if transform is Transform.CONV_TAPS_FASTEST:
        return source[:, 0, :][begin:end]
    if transform is Transform.WIDEN_FP32:
        return torch.tensor([float(value) for value in source.tolist()], dtype=torch.float32)
    if transform is Transform.INT64_WORDS:
        words = [
            ((value & 0xFFFFFFFF) - ((value & 0x80000000) << 1), value >> 32)
            for value in source.tolist()
        ]
        return torch.tensor(words, dtype=torch.int32)
    raise ValidationError(f"{plan.spec.name}: unexpected transform {transform}")


def _check_direct(plan: ObjectPlan, obj: TensorObject, payload, loaded) -> None:
    stored = decode_direct(payload, obj.format, obj.shape)
    rows = obj.shape[0]
    if len(obj.shape) == 1:
        expected = _expected_rows(plan, loaded, 0, rows)
        if not torch.equal(_bits(stored), _bits(expected.to(stored.dtype))):
            raise ValidationError(f"{obj.name}: stored words differ from the source")
        return
    for begin in range(0, rows, _ROW_CHUNK):
        end = min(rows, begin + _ROW_CHUNK)
        expected = _expected_rows(plan, loaded, begin, end)
        if expected.dtype != stored.dtype:
            raise ValidationError(f"{obj.name}: source dtype {expected.dtype} differs")
        if not torch.equal(_bits(stored[begin:end]), _bits(expected)):
            raise ValidationError(
                f"{obj.name}: stored words differ from the source in rows {begin}..{end}"
            )


def _natural_nibbles(packed: torch.Tensor) -> torch.Tensor:
    return torch.stack((packed & 0x0F, packed >> 4), dim=-1).reshape(packed.shape[0], -1)


def _check_routed_experts(artifact: Artifact, payloads: _PayloadFile, store, log) -> int:
    codecs = decode_direct(payloads.read(artifact.find(EXPERT_CODECS)), "I32", (LAYERS, EXPERTS))
    if not np.array_equal(codecs.numpy(), store.codecs.astype(np.int32)):
        raise ValidationError(f"{EXPERT_CODECS}: codec map differs from the study bitmap")
    books = decode_direct(payloads.read(artifact.find(F3_CODEBOOKS)), "I32", (LAYERS, 3))
    if not np.array_equal(books.numpy().view(np.uint32), store.book_words):
        raise ValidationError(f"{F3_CODEBOOKS}: books differ from the study books")
    table = decode_direct(payloads.read(artifact.find(MATRIX_SCALES)), "FP32", (LAYERS, EXPERTS, 3))
    table_words = table.view(torch.int32).numpy().view(np.uint32)
    records = {
        ExpertCodec.NVFP4: (artifact.find(E2M1_RECORDS), E2M1),
        ExpertCodec.F3: (artifact.find(F3_RECORDS), F3),
    }
    next_record = {ExpertCodec.NVFP4: 0, ExpertCodec.F3: 0}
    checked = 0
    started = time.perf_counter()
    for layer in range(LAYERS):
        for expert in range(EXPERTS):
            source = store.read(layer, expert)
            obj, numeric_format = records[source.codec]
            geometry = expert_tiles_geometry(numeric_format, obj.shape)
            index = next_record[source.codec]
            next_record[source.codec] = index + 1
            raw = payloads.read(obj, index * geometry.record_bytes, geometry.record_bytes)
            stored = decode_expert_tiles_record(raw, numeric_format, HIDDEN, INTERMEDIATE)
            for name in MATRICES:
                want, have = getattr(source.words, name), getattr(stored, name)
                if not (
                    torch.equal(want.codes, have.codes) and torch.equal(want.scales, have.scales)
                ):
                    raise ValidationError(f"expert {layer}/{expert} {name}: words differ")
            if tuple(int(word) for word in table_words[layer, expert]) != source.matrix_scale_words:
                raise ValidationError(f"expert {layer}/{expert}: matrix scales differ")
            checked += 1
        log(
            f"routed experts: layer {layer} exact "
            f"({checked} experts, {time.perf_counter() - started:.0f}s)"
        )
    for codec, (obj, _) in records.items():
        if next_record[codec] != obj.shape[0]:
            raise ValidationError(f"{obj.name}: {obj.shape[0]} records, {next_record[codec]} used")
    return checked


def _study_e4m3(words: np.ndarray) -> np.ndarray:
    """E4M3FN values of the study's code words, decoded from their bit fields in FP64."""

    words = words.astype(np.int64)
    sign = np.where(words & 0x80, -1.0, 1.0)
    exponent = (words >> 3) & 0xF
    fraction = (words & 0x7).astype(np.float64)
    normal = (1.0 + fraction / 8.0) * np.exp2(exponent - 7.0)
    value = np.where(exponent == 0, fraction * 2.0**-9, normal)
    return sign * np.where((words & 0x7F) == 0x7F, np.nan, value)


def _study_e2m1(nibbles: np.ndarray) -> np.ndarray:
    magnitude = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])[nibbles & 7]
    return np.where(nibbles & 8, -magnitude, magnitude)


def _sample_rows(rows: int) -> tuple[int, ...]:
    return tuple(sorted({0, rows // 2, rows - 1}))


def _reference_rows(plan: ObjectPlan, reader: ShardReader, rows: list[int]) -> torch.Tensor:
    """BF16 checkpoint rows that a study object quantizes, for its error report."""

    reference = ObjectPlan(plan.spec, plan.family, plan.reference_transform, plan.reference)
    loaded = [reader.get(name) for name in plan.reference]
    return torch.stack([_expected_rows(reference, loaded, row, row + 1)[0] for row in rows])


def _check_study_fp8(
    plan: ObjectPlan, obj: TensorObject, payload, study: DenseStudy, reader: ShardReader
) -> dict[str, object]:
    codes, scales = decode_fp8_row_scaled_words(payload, obj.shape)
    begin = 0
    sampled: list[int] = []
    study_values: list[np.ndarray] = []
    for path in plan.sources:
        record = study.load(path)
        rows = record.codes.shape[0]
        want_codes, want_scales = fp8_words(record, rows, obj.shape[1])
        if not (
            torch.equal(codes[begin : begin + rows], want_codes)
            and torch.equal(_bits(scales[begin : begin + rows]), _bits(want_scales))
        ):
            raise ValidationError(f"{obj.name}: rows {begin}.. differ from {path}")
        for row in _sample_rows(rows):
            # Independent decode of the study payload: E4M3FN bit fields times the BF16 word.
            scale = np.array([record.scales[row, 0]], dtype=np.uint32) << 16
            sampled.append(begin + row)
            study_values.append(_study_e4m3(record.codes[row]) * float(scale.view(np.float32)[0]))
        begin += rows
    if begin != obj.shape[0]:
        raise ValidationError(f"{obj.name}: records cover {begin} rows")
    stored = _E4M3[codes[sampled].long()] * scales[sampled].to(torch.float64).unsqueeze(1)
    return _cross_check(obj, plan, reader, sampled, stored, np.stack(study_values))


def _cross_check(obj, plan, reader, sampled, stored, study_values) -> dict[str, object]:
    if not np.array_equal(stored.numpy(), study_values):
        raise ValidationError(f"{obj.name}: decoded rows differ from the study decode")
    reference = _reference_rows(plan, reader, sampled).to(torch.float64)
    error = (stored - reference).norm() / reference.norm().clamp_min(1e-300)
    return {
        "name": obj.name,
        "rows_cross_checked": len(sampled),
        "relative_error_vs_bf16": float(error),
    }


def _check_study_shared(
    plan: ObjectPlan, obj: TensorObject, payload, study: DenseStudy, reader: ShardReader
) -> dict[str, object] | None:
    records = [study.load(path) for path in plan.sources]
    expected = [
        nvfp4_words(record, rows, columns)
        for record, (rows, columns) in zip(records, MATRIX_SHAPES, strict=True)
    ]
    if plan.transform is Transform.STUDY_SHARED_SCALES:
        stored_words = decode_direct(payload, "FP32", obj.shape).view(torch.int32)
        words = [int(word) & 0xFFFFFFFF for word in stored_words.tolist()]
        if words != [multiplier for _, multiplier in expected]:
            raise ValidationError(f"{obj.name}: multipliers differ from the study records")
        return None
    stored = decode_expert_tiles_record(payload, obj.format, HIDDEN, INTERMEDIATE)
    results = []
    for name, record, (want, multiplier) in zip(MATRICES, records, expected, strict=True):
        have = getattr(stored, name)
        if not (torch.equal(have.codes, want.codes) and torch.equal(have.scales, want.scales)):
            raise ValidationError(f"{obj.name} {name}: words differ from {record.path}")
        rows = _sample_rows(have.codes.shape[0])
        factor = float(np.array([multiplier], dtype=np.uint32).view(np.float32)[0])
        # Independent decode of the study payload: packed nibbles, E4M3FN bit fields, multiplier.
        packed = record.codes[list(rows)]
        nibbles = np.stack((packed & 0x0F, packed >> 4), axis=-1).reshape(len(rows), -1)
        block = np.repeat(_study_e4m3(record.scales[list(rows)]), 16, axis=1)
        study_values = _study_e2m1(nibbles) * block * factor
        stored_values = (
            _E2M1[have.codes[list(rows)].long()]
            * _E4M3[have.scales[list(rows)].long()].repeat_interleave(16, dim=1)
            * factor
        )
        matrix_plan = ObjectPlan(
            plan.spec,
            plan.family,
            plan.transform,
            plan.sources,
            (plan.reference[MATRICES.index(name)],),
            Transform.ROWS,
        )
        results.append(
            _cross_check(obj, matrix_plan, reader, list(rows), stored_values, study_values)
        )
    return {
        "name": obj.name,
        "rows_cross_checked": sum(item["rows_cross_checked"] for item in results),
        "relative_error_vs_bf16": max(item["relative_error_vs_bf16"] for item in results),
    }


def _parent_row(row: int) -> int:
    """Checkpoint row (study parent order q_proj, k_proj, v_proj) of artifact row `row`."""

    width = QSA_HEADS * QSA_HEAD_DIM
    if row < width:
        return (row // QSA_HEAD_DIM) * 2 * QSA_HEAD_DIM + row % QSA_HEAD_DIM
    if row < width + QSA_KV:
        return 2 * width + (row - width)
    if row < 2 * width + QSA_KV:
        gate = row - width - QSA_KV
        return (gate // QSA_HEAD_DIM) * 2 * QSA_HEAD_DIM + QSA_HEAD_DIM + gate % QSA_HEAD_DIM
    return 2 * width + QSA_KV + (row - 2 * width - QSA_KV)


def _check_study_attention(
    plan: ObjectPlan, obj: TensorObject, payload, study: DenseStudy, reader: ShardReader
) -> dict[str, object]:
    """The NVFP4 attention-input parent against its study record, row by row."""

    (path,) = plan.sources
    parent = load_attention_parent(study, path)
    codes, scales, divisor = decode_nvfp4_words(payload, obj.shape)
    if int(divisor.view(torch.int32)) & 0xFFFFFFFF != parent.divisor_word:
        raise ValidationError(f"{obj.name}: weight divisor differs from {path}")
    source_rows = torch.tensor([_parent_row(row) for row in range(obj.shape[0])])
    if sorted(source_rows.tolist()) != list(range(obj.shape[0])):
        raise ValidationError(f"{obj.name}: row map is not a permutation")
    if not (
        torch.equal(codes, torch.from_numpy(parent.codes)[source_rows])
        and torch.equal(scales, torch.from_numpy(parent.scales)[source_rows])
    ):
        raise ValidationError(f"{obj.name}: codes or block scales differ from {path}")
    # Independent FP64 decode of the study rows (value = E2M1 * E4M3FN / d_w, ATTN_PARENT.md)
    # at the first, middle and last row of every part, against the stored artifact rows.
    d_w = float(np.array([parent.divisor_word], dtype=np.uint32).view(np.float32)[0])
    width = QSA_HEADS * QSA_HEAD_DIM
    sampled = [
        row
        for begin, end in (
            (0, width),
            (width, width + QSA_KV),
            (width + QSA_KV, 2 * width + QSA_KV),
            (2 * width + QSA_KV, obj.shape[0]),
        )
        for row in (begin, (begin + end) // 2, end - 1)
    ]
    packed = parent.codes[[_parent_row(row) for row in sampled]]
    nibbles = np.stack((packed & 0x0F, packed >> 4), axis=-1).reshape(len(sampled), -1)
    block = np.repeat(_study_e4m3(parent.scales[[_parent_row(row) for row in sampled]]), 16, axis=1)
    study_values = _study_e2m1(nibbles) * block / d_w
    stored_values = (
        _E2M1[_natural_nibbles(codes[sampled]).long()]
        * _E4M3[scales[sampled].long()].repeat_interleave(16, dim=1)
        / d_w
    )
    return _cross_check(obj, plan, reader, sampled, stored_values, study_values)


def validate_artifact(
    path: Path,
    model_dir: Path,
    store,
    study: DenseStudy,
    frontend_bytes: dict[str, int],
    log=print,
) -> ValidationReport:
    """Validate one artifact against its sources; raises ValidationError on any difference."""

    started = time.perf_counter()
    report = ValidationReport()
    with Artifact(path) as artifact:
        if (artifact.identity.model_id, artifact.identity.weights_id) != (MODEL_ID, WEIGHTS_ID):
            raise ValidationError(f"unexpected identity {artifact.identity}")
        plans = build_inventory(store.nvfp4_experts, store.f3_experts, frontend_bytes)
        if len(plans) != len(artifact.objects):
            raise ValidationError(f"{len(artifact.objects)} objects, expected {len(plans)}")
        for plan, obj in zip(plans, artifact.objects, strict=True):
            spec = plan.spec
            if isinstance(spec, ResourceSpec):
                same = isinstance(obj, ResourceObject) and (
                    obj.name,
                    obj.encoding,
                    obj.bytes,
                ) == (spec.name, spec.encoding, spec.bytes)
            else:
                same = isinstance(obj, TensorObject) and (
                    obj.name,
                    obj.shape,
                    obj.format,
                    obj.layout,
                ) == (spec.name, spec.shape, spec.format, spec.layout)
            if not same:
                raise ValidationError(f"directory entry {obj.name} differs from the inventory")
        payloads = _PayloadFile(artifact)
        try:
            with ShardReader(model_dir) as reader:
                for plan, obj in zip(plans, artifact.objects, strict=True):
                    if plan.transform is Transform.ROUTED_EXPERTS:
                        continue
                    payload = payloads.read(obj)
                    if plan.transform in STUDY_TRANSFORMS:
                        assert isinstance(obj, TensorObject)
                        if plan.transform is Transform.STUDY_FP8_ROWS:
                            check = _check_study_fp8(plan, obj, payload, study, reader)
                        elif plan.transform is Transform.STUDY_ATTENTION_PARENT:
                            check = _check_study_attention(plan, obj, payload, study, reader)
                        else:
                            check = _check_study_shared(plan, obj, payload, study, reader)
                        if check is not None:
                            report.study_cross_checks.append(check)
                    elif isinstance(obj, ResourceObject):
                        if bytes(payload) != (model_dir / plan.sources[0]).read_bytes():
                            raise ValidationError(f"{obj.name}: resource bytes differ")
                    else:
                        loaded = [reader.get(name) for name in plan.sources]
                        _check_direct(plan, obj, payload, loaded)
                        del loaded
                    report.exact_objects += 1
                    report.objects += 1
                    del payload
            log(f"dense and resource objects: {report.objects} checked")
            report.routed_experts = _check_routed_experts(artifact, payloads, store, log)
            routed = sum(plan.transform is Transform.ROUTED_EXPERTS for plan in plans)
            report.objects += routed
            report.exact_objects += routed
        finally:
            payloads.close()
    report.seconds = time.perf_counter() - started
    return report
