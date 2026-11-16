"""Qwen3.8-Flash-Next converter transforms, study-store decoding and inventory (no GPU)."""

from __future__ import annotations

import struct

import numpy as np
import pytest
import torch

from tools.artifact.container import TensorSpec
from tools.convert.qwen3_8_flash_next.inventory import (
    FRONTEND_RESOURCES,
    Family,
    ObjectPlan,
    Transform,
    build_inventory,
)
from tools.convert.qwen3_8_flash_next.materialize import (
    attention_parent_rows,
    int64_words,
    materialize,
    pack_moe_control,
    split_query_gate,
)
from tools.convert.qwen3_8_flash_next.study_store import (
    MATRIX_SHAPES,
    ExpertCodec,
    StudyExpertStore,
    decode_study_record,
)

FRONTEND = dict.fromkeys(FRONTEND_RESOURCES, 10)


def _study_record(codec: ExpertCodec, seed: int):
    """A study record built bit by bit from its documented stream rule, and its words."""

    rng = np.random.default_rng(seed)
    record = bytearray()
    expected = []
    for rows, columns in MATRIX_SHAPES:
        codes = rng.integers(0, 1 << codec.code_bits, size=rows * columns, dtype=np.uint8)
        scales = rng.integers(0, 0x7F, size=rows * columns // codec.group_size, dtype=np.uint8)
        multiplier = struct.pack("<f", float(rng.random()))
        stream = bytearray(rows * columns * codec.code_bits // 8)
        for index, code in enumerate(codes.tolist()):
            for bit in range(codec.code_bits):
                position = index * codec.code_bits + bit
                stream[position // 8] |= ((code >> bit) & 1) << (position % 8)
        record += multiplier + scales.tobytes() + stream
        expected.append((codes.reshape(rows, columns), scales, multiplier))
    return bytes(record), expected


@pytest.mark.parametrize("codec", list(ExpertCodec))
def test_study_record_decode_matches_bitstream(codec):
    record, expected = _study_record(codec, seed=int(codec))
    decoded = decode_study_record(record, codec)
    assert decoded.codec is codec
    for matrix, (codes, scales, _) in zip(
        (decoded.words.gate, decoded.words.up, decoded.words.down), expected, strict=True
    ):
        assert np.array_equal(matrix.codes.numpy(), codes)
        assert np.array_equal(matrix.scales.numpy().reshape(-1), scales)
    assert decoded.matrix_scale_words == tuple(struct.unpack("<I", item[2])[0] for item in expected)


def test_study_store_addresses_records_by_bitmap(tmp_path):
    """A sparse full-size store with three planted records and a sparse NVFP4 bitmap."""

    f3, nvfp4 = ExpertCodec.F3.record_bytes, ExpertCodec.NVFP4.record_bytes
    high = [(0, 3), (2, 511), (47, 0)]
    nvfp4_count = len(high)
    header_bytes = 64 + 576 + 3072
    size = header_bytes + (48 * 512 - nvfp4_count) * f3 + nvfp4_count * nvfp4
    path = tmp_path / "experts.bin"
    bitmap = bytearray(3072)
    for layer, expert in high:
        index = layer * 512 + expert
        bitmap[index // 8] |= 1 << (index % 8)
    books = bytes(index % 251 for index in range(48 * 3 * 4))
    header = struct.pack(
        "<8s14I", b"FSTUDY01", 1, 0, 0, 48, 512, 2560, 640, header_bytes, f3, nvfp4, 3, 32, 4, 16
    )
    planted = {
        (2, 511): _study_record(ExpertCodec.NVFP4, 5)[0],
        (2, 510): _study_record(ExpertCodec.F3, 6)[0],
        (47, 1): _study_record(ExpertCodec.F3, 7)[0],
    }
    with path.open("wb") as handle:
        handle.truncate(size)
        handle.write(header + books + bitmap)
        for (layer, expert), record in planted.items():
            index = layer * 512 + expert
            before = sum(1 for item in high if item[0] * 512 + item[1] < index)
            handle.seek(header_bytes + index * f3 + before * (nvfp4 - f3))
            handle.write(record)
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        '{"format": "mixed_mse_gptq", "completed_layers": '
        + str(list(range(48)))
        + ', "storage": {"high_precision_experts": 3, "total_bytes": '
        + str(size)
        + "}}"
    )
    with StudyExpertStore(path, manifest) as store:
        assert (store.nvfp4_experts, store.f3_experts) == (3, 48 * 512 - 3)
        assert store.book_words[0, 0] == int.from_bytes(bytes(range(4)), "little")
        assert store.codec(2, 511) is ExpertCodec.NVFP4 and store.codec(2, 510) is ExpertCodec.F3
        for key, record in planted.items():
            read = store.read(*key)
            reference = decode_study_record(record, store.codec(*key))
            assert torch.equal(read.words.down.codes, reference.words.down.codes)
            assert read.matrix_scale_words == reference.matrix_scale_words


def test_query_gate_deinterleave_follows_head_index_formula():
    q_proj = torch.arange(24 * 512, dtype=torch.float32).unsqueeze(1).repeat(1, 3)
    query, gate = split_query_gate(q_proj)
    rows = torch.arange(24 * 256)
    head, lane = rows // 256, rows % 256
    assert torch.equal(query[:, 0], (head * 512 + lane).float())
    assert torch.equal(gate[:, 0], (head * 512 + 256 + lane).float())


class _Reader:
    def __init__(self, tensors: dict[str, torch.Tensor]) -> None:
        self.tensors = tensors

    def get(self, name: str) -> torch.Tensor:
        return self.tensors[name]


def _plan(shape, transform, *sources, numeric_format="BF16"):
    layout = "contiguous-le-v1"
    return ObjectPlan(
        TensorSpec("x", shape, numeric_format, layout), Family.PLE, transform, sources
    )


def _bf16(*shape: int) -> torch.Tensor:
    return torch.arange(int(np.prod(shape)), dtype=torch.float32).reshape(shape).to(torch.bfloat16)


def test_moe_control_packing():
    router, shared = _bf16(512, 4), _bf16(1, 4) + 7
    control = pack_moe_control(router, shared)
    assert control.shape == (544, 4)
    assert torch.equal(control[:512], router) and torch.equal(control[512], shared[0])
    assert control[513:].view(torch.int16).eq(0).all()


def test_convolution_and_index_transforms():
    conv = _bf16(10240, 1, 4)
    reader = _Reader({"c": conv, "i": _bf16(640, 3)})
    tap_major = materialize(_plan((4, 10240), Transform.CONV_TAP_MAJOR, "c"), reader)
    taps_fastest = materialize(_plan((10240, 4), Transform.CONV_TAPS_FASTEST, "c"), reader)
    for tap in range(4):
        assert torch.equal(tap_major[tap], conv[:, 0, tap])
        assert torch.equal(taps_fastest[:, tap], conv[:, 0, tap])
    query = materialize(_plan((512, 3), Transform.INDEX_QUERY, "i"), reader)
    key = materialize(_plan((128, 3), Transform.INDEX_KEY, "i"), reader)
    assert torch.equal(torch.cat((query, key)), reader.get("i"))


def test_int64_words_are_exact_little_endian_pairs():
    values = torch.tensor([0, 23703573157769, -1, (1 << 63) - 1, -(1 << 63)], dtype=torch.int64)
    words = int64_words(values)
    for value, (low, high) in zip(values.tolist(), words.tolist(), strict=True):
        assert (high << 32) | (low & 0xFFFFFFFF) == value


def test_inventory():
    plans = build_inventory(1583, 22993, FRONTEND)
    by_name = {plan.spec.name: plan for plan in plans}
    # The shared expert is one study NVFP4 record plus its three multipliers.
    assert len(plans) == 901
    assert by_name["text/token_embedding"].spec.format == "BF16"
    head = by_name["text/output_head"]
    assert head.spec.format == "FP8_E4M3FN_ROW_BF16S"
    assert head.sources == ("head/lm_head/fp8_gptq.npz",)
    gdn = by_name["text/layers/0/gdn/query_key_value_z"]
    assert gdn.sources == (
        "0/linear_attn.in_proj_qkv.weight/fp8_gptq.npz",
        "0/linear_attn.in_proj_z.weight/fp8_gptq.npz",
    )
    assert by_name["text/layers/3/attention/query_key_gate_value"].spec.format == "NVFP4"
    assert by_name["text/layers/3/attention/output"].spec.layout == "row-scale-v1"
    shared = by_name["text/layers/5/moe/shared_expert"]
    assert shared.spec.shape == (1, 2560, 640) and shared.spec.format == "E2M1G16_E4M3FN"
    assert shared.sources[2] == "5/mlp.shared_expert.down_proj.weight/nv_gptq_mse.npz"
    assert by_name["text/layers/5/moe/shared_matrix_scales"].spec.shape == (3,)
    assert "text/layers/5/moe/shared_gate_up" not in by_name
    down = by_name["text/layers/5/moe_hyper_connection/down"]
    assert down.spec.format == "FP8_E4M3FN_ROW_BF16S"
    assert down.sources == ("hc/5/mlp_hyper_connection.input_mix_weight_down.weight/fp8_gptq.npz",)
    final_up = by_name["text/final_hyper_connection/up"]
    assert final_up.sources == (
        "hc/final/hyper_connection_mixer.input_mix_weight_up.weight/fp8_gptq.npz",
    )
    assert by_name["text/layers/5/mixer_hyper_connection/norm"].spec.format == "BF16"
    assert by_name["text/layers/5/mixer_hyper_connection/write"].spec.format == "BF16"
    names = [plan.spec.name for plan in plans]
    assert len(names) == len(set(names))
    assert sum(name.startswith("text/layers/1/ple/") for name in names) == 9
    assert sum(name.endswith("/attention/output") for name in names) == 12
    assert by_name["text/routed_experts/f3_records"].spec.shape == (22993, 2560, 640)
    assert by_name["text/routed_experts/matrix_scales"].spec.shape == (48, 512, 3)
    parent = by_name["text/layers/3/attention/query_key_gate_value"]
    assert parent.sources == ("attn_parent/3.npz",)


def test_attention_parent_row_permutation():
    rows = attention_parent_rows()
    assert sorted(rows.tolist()) == list(range(13312))
    query, gate = split_query_gate(torch.arange(12288).unsqueeze(1))
    assert torch.equal(rows[:6144], query[:, 0])
    assert torch.equal(rows[6144:6656], torch.arange(12288, 12800))
    assert torch.equal(rows[6656:12800], gate[:, 0])
    assert torch.equal(rows[12800:], torch.arange(12800, 13312))
