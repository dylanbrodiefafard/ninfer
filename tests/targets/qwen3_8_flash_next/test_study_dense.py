"""Exact ingestion of dense-study records (synthetic records in the study's NPZ format)."""

from __future__ import annotations

import hashlib
import json
import struct

import numpy as np
import pytest
import torch

from tools.artifact.container import TensorSpec
from tools.artifact.layouts import (
    decode_direct,
    decode_expert_tiles_record,
    decode_fp8_row_scaled_words,
    decode_nvfp4_words,
)
from tools.convert.qwen3_8_flash_next.inventory import Family, ObjectPlan, Transform
from tools.convert.qwen3_8_flash_next.materialize import encode_study_payload
from tools.convert.qwen3_8_flash_next.study_dense import DenseStudy

_FAMILIES = {
    "gdn_in": "fp8_gptq",
    "gdn_out": "fp8_gptq",
    "attn_in": "nv_gptq_mse",
    "attn_out": "fp8_gptq",
    "shared": "nv_gptq_mse",
    "head": "fp8_gptq",
    "embed": "bf16",
    "ple": "bf16",
}


class _Study:
    """A study directory pair whose records follow `fnstudy/dense.py` `save_record`."""

    def __init__(self, root) -> None:
        self.records = root / "quant" / "dense"
        self.results = root / "results" / "dense"
        self.results.mkdir(parents=True)
        self.hashes: dict[str, dict[str, object]] = {}
        self.parents: dict[str, dict[str, object]] = {}
        self.hc: dict[str, dict[str, object]] = {}
        self.damping: dict[str, float] = {}
        self.rng = np.random.default_rng(11)

    def save(self, relative: str, q, scale, global_scale: float, damping: float = 0.1):
        hyper_connection = relative.startswith("hc/")
        layer, key, file_name = relative.removeprefix("hc/").split("/")
        path = self.records / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            path,
            q=q,
            scale=scale,
            global_scale=np.array(global_scale, np.float32),
            method=file_name.removesuffix(".npz"),
            damping=damping,
        )
        data = path.read_bytes()
        fingerprint = {"sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data)}
        if hyper_connection:
            self.hc[f"{layer}:{key}"] = {
                "path": "quant/dense/" + relative,
                "damping": damping,
                **fingerprint,
            }
            return
        self.hashes["quant/dense/" + relative] = fingerprint
        self.damping[f"{layer}:{key}"] = damping

    def fp8(self, relative: str, rows: int, columns: int):
        codes = self.rng.integers(0, 256, size=(rows, columns), dtype=np.uint8)
        codes[(codes & 0x7F) == 0x7F] = 0
        scales = (
            torch.from_numpy(self.rng.random(rows).astype(np.float32) + 0.5)
            .to(torch.bfloat16)
            .view(torch.int16)
            .numpy()
            .view(np.uint16)
            .reshape(rows, 1)
        )
        self.save(relative, codes, scales, 1.0)
        return codes, scales

    def nvfp4(self, relative: str, rows: int, columns: int, multiplier: float):
        codes = self.rng.integers(0, 16, size=(rows, columns), dtype=np.uint8)
        packed = codes[:, ::2] | (codes[:, 1::2] << 4)
        scales = self.rng.integers(0, 0x7F, size=(rows, columns // 16), dtype=np.uint8)
        self.save(relative, packed, scales, multiplier)
        return codes, scales

    def parent(self, layer: int, divisor: float, value_damping: float):
        """An attention-input parent record in the `ATTN_PARENT.md` format."""

        rows, columns = 13312, 2560
        codes = self.rng.integers(0, 256, size=(rows, columns // 2), dtype=np.uint8)
        scales = self.rng.integers(0, 0x7F, size=(rows, columns // 16), dtype=np.uint8)
        path = self.records / f"attn_parent/{layer}.npz"
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            path,
            q=codes,
            scale=scales,
            weight_divisor=np.array(divisor, np.float32),
            format="NVFP4",
            version=np.array(1, np.int32),
            shape=np.array([rows, columns], np.int64),
            parts=np.array(
                ["self_attn.q_proj.weight", "self_attn.k_proj.weight", "self_attn.v_proj.weight"]
            ),
            row_offsets=np.array([0, 12288, 12800, 13312], np.int64),
            rounding=np.array(["RTN", "RTN", "GPTQ"]),
            damping=np.array([0.0, 0.0, value_damping], np.float32),
        )
        data = path.read_bytes()
        self.parents[str(layer)] = {
            "path": f"quant/dense/attn_parent/{layer}.npz",
            "sha256": hashlib.sha256(data).hexdigest(),
            "bytes": len(data),
        }
        self.damping[f"{layer}:self_attn.v_proj.weight"] = value_damping
        return codes, scales

    def open(self) -> DenseStudy:
        (self.results / "hc_validation.json").write_text(
            json.dumps({"status": "complete", "records": self.hc})
        )
        calibration = {
            key: {"path": item["path"], "selected_damping": item["damping"]}
            for key, item in self.hc.items()
        }
        (self.results / "hc_calibration.json").write_text(
            json.dumps({"status": "complete", "matrices": calibration})
        )
        (self.results / "attn_parent_validation.json").write_text(
            json.dumps({"status": "complete", "records": self.parents})
        )
        (self.results / "artifact_hashes.json").write_text(
            json.dumps({"status": "complete", "records": self.hashes})
        )
        recipe = {
            family: {
                "method": method,
                "damping_by_tensor": self.damping if family == "gdn_in" else {},
            }
            for family, method in _FAMILIES.items()
        }
        (self.results / "summary.json").write_text(json.dumps({"recommended_recipe": recipe}))
        return DenseStudy(self.records, self.results)


def _plan(name, shape, numeric_format, layout, transform, *sources):
    return ObjectPlan(
        TensorSpec(name, shape, numeric_format, layout), Family.GDN_INPUT, transform, sources
    )


def test_fp8_records_concatenate_exactly(tmp_path):
    study = _Study(tmp_path)
    first = study.fp8("0/a.weight/fp8_gptq.npz", 3, 64)
    second = study.fp8("0/b.weight/fp8_gptq.npz", 5, 64)
    plan = _plan(
        "x",
        (8, 64),
        "FP8_E4M3FN_ROW_BF16S",
        "row-scale-v1",
        Transform.STUDY_FP8_ROWS,
        "0/a.weight/fp8_gptq.npz",
        "0/b.weight/fp8_gptq.npz",
    )
    codes, scales = decode_fp8_row_scaled_words(encode_study_payload(plan, study.open()), (8, 64))
    assert np.array_equal(codes.numpy(), np.concatenate((first[0], second[0])))
    expected = np.concatenate((first[1], second[1])).reshape(-1)
    assert np.array_equal(scales.view(torch.int16).numpy().view(np.uint16), expected)


def test_shared_expert_record_and_multipliers_are_exact(tmp_path):
    study = _Study(tmp_path)
    shapes = ((640, 2560), (640, 2560), (2560, 640))
    names = ("gate", "up", "down")
    multipliers = (3.5e-5, 4.25e-5, 1.0e-4)
    words = [
        study.nvfp4(f"7/mlp.shared_expert.{name}_proj.weight/nv_gptq_mse.npz", *shape, multiplier)
        for name, shape, multiplier in zip(names, shapes, multipliers, strict=True)
    ]
    sources = tuple(f"7/mlp.shared_expert.{name}_proj.weight/nv_gptq_mse.npz" for name in names)
    opened = study.open()
    record_plan = _plan(
        "s",
        (1, 2560, 640),
        "E2M1G16_E4M3FN",
        "moe-expert-tiles-v1",
        Transform.STUDY_SHARED_EXPERT,
        *sources,
    )
    record = encode_study_payload(record_plan, opened)
    assert len(record) == 2_764_800
    decoded = decode_expert_tiles_record(record, "E2M1G16_E4M3FN", 2560, 640)
    for name, (codes, scales) in zip(names, words, strict=True):
        assert np.array_equal(getattr(decoded, name).codes.numpy(), codes)
        assert np.array_equal(getattr(decoded, name).scales.numpy(), scales)
    scale_plan = _plan(
        "m", (3,), "FP32", "contiguous-le-v1", Transform.STUDY_SHARED_SCALES, *sources
    )
    stored = decode_direct(encode_study_payload(scale_plan, opened), "FP32", (3,))
    expected = [struct.unpack("<f", struct.pack("<f", value))[0] for value in multipliers]
    assert stored.tolist() == expected


def test_records_are_checked_before_use(tmp_path):
    study = _Study(tmp_path)
    study.fp8("0/a.weight/fp8_gptq.npz", 2, 32)
    opened = study.open()
    path = study.records / "0/a.weight/fp8_gptq.npz"
    data = bytearray(path.read_bytes())
    data[-1] ^= 1
    path.write_bytes(bytes(data))
    with pytest.raises(ValueError, match="artifact_hashes"):
        opened.load("0/a.weight/fp8_gptq.npz")
    with pytest.raises(ValueError, match="fingerprinted"):
        opened.load("0/missing.weight/fp8_gptq.npz")


def test_records_need_the_recommended_damping(tmp_path):
    study = _Study(tmp_path)
    study.fp8("0/a.weight/fp8_gptq.npz", 2, 32)
    study.damping["0:a.weight"] = 0.3
    with pytest.raises(ValueError, match="damping"):
        study.open().load("0/a.weight/fp8_gptq.npz")


def test_attention_parent_rows_are_permuted_exactly(tmp_path):
    study = _Study(tmp_path)
    codes, scales = study.parent(3, 3044.8142, 0.1)
    plan = _plan(
        "a",
        (13312, 2560),
        "NVFP4",
        "blockscale-k16-m128x4-v1",
        Transform.STUDY_ATTENTION_PARENT,
        "attn_parent/3.npz",
    )
    stored_codes, stored_scales, divisor = decode_nvfp4_words(
        encode_study_payload(plan, study.open()), (13312, 2560)
    )
    assert float(divisor) == float(np.float32(3044.8142))
    # Artifact rows [query 6144, key 512, gate 6144, value 512]; head h's query row d is q_proj
    # row 512 h + d and its gate row d is q_proj row 512 h + 256 + d.
    for row, source in ((0, 0), (255, 255), (256, 512), (6143, 23 * 512 + 255)):
        assert np.array_equal(stored_codes[row].numpy(), codes[source])
        assert np.array_equal(stored_scales[row].numpy(), scales[source])
    for row, source in ((6144, 12288), (6655, 12799), (6656, 256), (12799, 23 * 512 + 511)):
        assert np.array_equal(stored_codes[row].numpy(), codes[source])
    assert np.array_equal(stored_codes[12800:].numpy(), codes[12800:])


def test_hyper_connection_records_use_their_own_fingerprints(tmp_path):
    study = _Study(tmp_path)
    codes, scales = study.fp8(
        "hc/final/hyper_connection_mixer.input_mix_weight_up.weight/fp8_gptq.npz", 64, 32
    )
    plan = _plan(
        "h",
        (64, 32),
        "FP8_E4M3FN_ROW_BF16S",
        "row-scale-v1",
        Transform.STUDY_FP8_ROWS,
        "hc/final/hyper_connection_mixer.input_mix_weight_up.weight/fp8_gptq.npz",
    )
    opened = study.open()
    stored_codes, stored_scales = decode_fp8_row_scaled_words(
        encode_study_payload(plan, opened), (64, 32)
    )
    assert np.array_equal(stored_codes.numpy(), codes)
    assert np.array_equal(stored_scales.view(torch.int16).numpy().view(np.uint16), scales[:, 0])
    with pytest.raises(ValueError, match="hc_validation"):
        opened.load("hc/0/attn_hyper_connection.input_mix_weight_up.weight/fp8_gptq.npz")
