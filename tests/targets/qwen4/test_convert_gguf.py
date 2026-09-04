from __future__ import annotations

from collections import Counter
import struct

import pytest

from tools.convert.qwen4 import convert, inventory
from tools.convert.qwen4.gguf import GGUFError, read_gguf


def _string(value: str) -> bytes:
    data = value.encode("utf-8")
    return struct.pack("<Q", len(data)) + data


def _fixture(*, tensor_type: int = 8) -> bytes:
    # One u32 metadata member, then a Q8_0 matrix [2,32] and BF16 vector [2].
    header = b"GGUF" + struct.pack("<IQQ", 3, 2, 1)
    metadata = _string("general.alignment") + struct.pack("<II", 4, 32)
    q8 = _string("q8") + struct.pack("<IQQIQ", 2, 32, 2, tensor_type, 0)
    bf16 = _string("bf16") + struct.pack("<IQIQ", 1, 2, 30, 96)
    directory = header + metadata + q8 + bf16
    data_offset = (len(directory) + 31) // 32 * 32
    return directory + bytes(data_offset - len(directory)) + bytes(68) + bytes(28) + b"\x01\x02\x03\x04"


def test_dependency_free_gguf_reader_exposes_exact_tensor_spans(tmp_path) -> None:
    path = tmp_path / "fixture.gguf"
    path.write_bytes(_fixture())
    source = read_gguf(path)

    assert source.metadata == {"general.alignment": 32}
    assert [(t.name, t.shape, t.format, t.offset, t.bytes) for t in source.tensors] == [
        ("q8", (2, 32), "Q8_0", 0, 68),
        ("bf16", (2,), "BF16", 96, 4),
    ]
    assert source.absolute_span(source.tensors[1]) == (
        source.data_offset + 96,
        source.data_offset + 100,
    )


def test_gguf_matrix_shape_is_mathematical_but_payload_remains_k_fastest(tmp_path) -> None:
    # GGUF directory dimensions are physical K,N. The reader exposes mathematical N,K while the
    # byte span stays untouched: each channel/output row therefore owns one contiguous K run.
    values = (11.0, 12.0, 13.0, 21.0, 22.0, 23.0)
    header = b"GGUF" + struct.pack("<IQQ", 3, 1, 1)
    metadata = _string("general.alignment") + struct.pack("<II", 4, 32)
    tensor = _string("matrix") + struct.pack("<IQQIQ", 2, 3, 2, 0, 0)
    directory = header + metadata + tensor
    data_offset = (len(directory) + 31) // 32 * 32
    path = tmp_path / "matrix.gguf"
    path.write_bytes(
        directory + bytes(data_offset - len(directory)) + struct.pack("<6f", *values)
    )

    source = read_gguf(path)
    matrix = source.tensors[0]
    assert matrix.shape == (2, 3)
    begin, end = source.absolute_span(matrix)
    assert struct.unpack("<6f", path.read_bytes()[begin:end]) == values


def test_gguf_reader_rejects_unknown_tensor_codec_and_truncation(tmp_path) -> None:
    path = tmp_path / "fixture.gguf"
    path.write_bytes(_fixture(tensor_type=999))
    with pytest.raises(GGUFError, match="unsupported ggml type 999"):
        read_gguf(path)

    path.write_bytes(_fixture()[:-1])
    with pytest.raises(GGUFError, match="outside the file"):
        read_gguf(path)


def test_verifier_inventory_is_closed_with_exact_placement_totals() -> None:
    geometry = {
        "BF16": (1, 2),
        "FP32": (1, 4),
        "Q8_0": (32, 34),
        "Q4_K": (256, 144),
        "Q5_K": (256, 176),
        "Q6_K": (256, 210),
        "IQ2_XXS": (256, 66),
        "IQ1_S": (256, 50),
        "IQ4_NL": (32, 18),
    }
    counts: Counter[str] = Counter()
    format_bytes: Counter[str] = Counter()
    mapped_bytes = 0
    for spec in inventory.TENSOR_SPECS:
        values = 1
        for dim in spec.shape:
            values *= dim
        block_values, block_bytes = geometry[spec.format]
        assert values % block_values == 0
        encoded = values // block_values * block_bytes
        counts[spec.format] += 1
        format_bytes[spec.format] += encoded
        if spec.mapped_host:
            mapped_bytes += encoded

    assert len(inventory.TENSOR_SPECS) == 1224
    assert dict(counts) == convert.EXPECTED_FORMAT_COUNTS
    assert dict(format_bytes) == convert.EXPECTED_FORMAT_BYTES
    assert sum(format_bytes.values()) == convert.EXPECTED_TENSOR_BYTES
    assert mapped_bytes == convert.EXPECTED_MAPPED_BYTES
    assert sum(format_bytes.values()) - mapped_bytes == convert.EXPECTED_DEVICE_BYTES
