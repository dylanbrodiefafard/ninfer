from __future__ import annotations

import ctypes
import hashlib
import os
from pathlib import Path
import random
import struct

import pytest
import torch

from tools.reference.qwen4.ggml_k_codecs import (
    Q4_K_BYTES,
    Q5_K_BYTES,
    Q6_K_BYTES,
    Q8_0_BYTES,
    decode_q4_k,
    decode_q5_k,
    decode_q6_k,
    decode_q8_0,
)


_PINNED_LLAMA_CPP_SOURCE_HASHES = {
    "ggml/src/ggml-quants.c": (
        "07143d7068936ae46b3c528b2f3d4bbb666e74d88992165716174d243573965d"
    ),
    "ggml/src/ggml-common.h": (
        "0061131b615c5721fc88a78feeb22c1f8c450f1c2646a317d80796a653bf595c"
    ),
}


def _pack_scale_min(scales: list[int], minimums: list[int]) -> bytes:
    assert len(scales) == len(minimums) == 8
    assert all(0 <= value < 64 for value in scales + minimums)
    packed = bytearray(12)
    for group in range(4):
        packed[group] = scales[group]
        packed[group + 4] = minimums[group]
    for group in range(4, 8):
        packed[group - 4] |= (scales[group] >> 4) << 6
        packed[group] |= (minimums[group] >> 4) << 6
        packed[group + 4] = (scales[group] & 0x0F) | ((minimums[group] & 0x0F) << 4)
    return bytes(packed)


def _pack_q4_k(
    scale: float,
    minimum_scale: float,
    scales: list[int],
    minimums: list[int],
    codes: list[int],
) -> bytes:
    assert len(codes) == 256 and all(0 <= code < 16 for code in codes)
    quants = bytearray(128)
    for pair in range(4):
        for lane in range(32):
            quants[32 * pair + lane] = (
                codes[64 * pair + lane] | (codes[64 * pair + 32 + lane] << 4)
            )
    return (
        struct.pack("<ee", scale, minimum_scale)
        + _pack_scale_min(scales, minimums)
        + quants
    )


def _pack_q5_k(
    scale: float,
    minimum_scale: float,
    scales: list[int],
    minimums: list[int],
    codes: list[int],
) -> bytes:
    assert len(codes) == 256 and all(0 <= code < 32 for code in codes)
    high_bits = bytearray(32)
    low_bits = bytearray(128)
    for group in range(8):
        for lane in range(32):
            code = codes[32 * group + lane]
            high_bits[lane] |= (code >> 4) << group
            low_offset = 32 * (group // 2) + lane
            low_bits[low_offset] |= (code & 0x0F) << (4 * (group & 1))
    return (
        struct.pack("<ee", scale, minimum_scale)
        + _pack_scale_min(scales, minimums)
        + high_bits
        + low_bits
    )


def _pack_q6_k(scale: float, scales: list[int], codes: list[int]) -> bytes:
    assert len(scales) == 16 and all(-128 <= value <= 127 for value in scales)
    assert len(codes) == 256 and all(-32 <= code <= 31 for code in codes)
    low_bits = bytearray(128)
    high_bits = bytearray(64)
    for half in range(2):
        for group32 in range(4):
            for lane in range(32):
                unsigned_code = codes[128 * half + 32 * group32 + lane] + 32
                low_offset = 64 * half + lane + 32 * (group32 & 1)
                low_bits[low_offset] |= (unsigned_code & 0x0F) << (4 * (group32 // 2))
                high_bits[32 * half + lane] |= (unsigned_code >> 4) << (2 * group32)
    signed_scales = bytes(value & 0xFF for value in scales)
    return bytes(low_bits) + bytes(high_bits) + signed_scales + struct.pack("<e", scale)


def _q4_or_q5_expected(
    scale: float,
    minimum_scale: float,
    scales: list[int],
    minimums: list[int],
    codes: list[int],
) -> torch.Tensor:
    return torch.tensor(
        [
            scale * scales[group] * codes[32 * group + lane]
            - minimum_scale * minimums[group]
            for group in range(8)
            for lane in range(32)
        ],
        dtype=torch.float64,
    )


def _q6_expected(scale: float, scales: list[int], codes: list[int]) -> torch.Tensor:
    return torch.tensor(
        [scale * scales[index // 16] * code for index, code in enumerate(codes)],
        dtype=torch.float64,
    )


def test_q8_0_little_endian_scale_and_signed_codes() -> None:
    codes = [-128, -127, -1, 0, 1, 126, 127] + list(range(-12, 13))
    assert len(codes) == 32
    block = struct.pack("<e32b", -0.625, *codes)

    actual = decode_q8_0(block)
    expected = torch.tensor([-0.625 * code for code in codes], dtype=torch.float64)

    assert block[:2] == bytes((0x00, 0xB9))
    torch.testing.assert_close(actual, expected, atol=0.0, rtol=0.0)


def test_q4_k_fixed_scale_min_and_nibble_layout() -> None:
    scale = 0.375
    minimum_scale = 0.15625
    scales = [0, 1, 15, 16, 17, 31, 48, 63]
    minimums = [63, 48, 31, 17, 16, 15, 1, 0]
    codes = [
        (11 * group + 3 * lane) & 0x0F for group in range(8) for lane in range(32)
    ]
    block = _pack_q4_k(scale, minimum_scale, scales, minimums, codes)

    assert len(block) == Q4_K_BYTES
    assert block[4:16] == bytes(
        (0x40, 0x41, 0xCF, 0xD0, 0x7F, 0x30, 0x1F, 0x11, 0x01, 0xFF, 0x10, 0x0F)
    )
    torch.testing.assert_close(
        decode_q4_k(block),
        _q4_or_q5_expected(scale, minimum_scale, scales, minimums, codes),
        atol=0.0,
        rtol=0.0,
    )


def test_q5_k_fixed_high_bit_scale_min_and_nibble_layout() -> None:
    scale = -0.21875
    minimum_scale = 0.5
    scales = [63, 0, 33, 18, 49, 2, 31, 16]
    minimums = [1, 62, 17, 32, 47, 16, 3, 63]
    codes = [
        (17 * group + 5 * lane) & 0x1F for group in range(8) for lane in range(32)
    ]
    block = _pack_q5_k(scale, minimum_scale, scales, minimums, codes)

    assert len(block) == Q5_K_BYTES
    assert any(code >= 16 for code in codes)
    assert block[16] == sum(
        ((codes[32 * group] >> 4) & 1) << group for group in range(8)
    )
    torch.testing.assert_close(
        decode_q5_k(block),
        _q4_or_q5_expected(scale, minimum_scale, scales, minimums, codes),
        atol=0.0,
        rtol=0.0,
    )


def test_q6_k_fixed_signed_code_scale_and_plane_layout() -> None:
    scale = 0.3125
    scales = [
        -128,
        -63,
        -17,
        -1,
        0,
        1,
        16,
        31,
        47,
        63,
        95,
        127,
        -96,
        -32,
        2,
        7,
    ]
    codes = [((19 * index + 7) & 0x3F) - 32 for index in range(256)]
    codes[0], codes[31], codes[32], codes[64], codes[96] = -32, 31, -1, 0, 30
    block = _pack_q6_k(scale, scales, codes)

    assert len(block) == Q6_K_BYTES
    assert block[192:208] == bytes(value & 0xFF for value in scales)
    assert block[208:] == struct.pack("<e", scale)
    torch.testing.assert_close(
        decode_q6_k(block), _q6_expected(scale, scales, codes), atol=0.0, rtol=0.0
    )


def test_randomized_packed_blocks_match_logical_format_formulas() -> None:
    source = random.Random(0x38456)
    half_values = (-2.0, -0.625, -0.0, 0.03125, 0.75, 3.5)
    for _ in range(24):
        q8_scale = source.choice(half_values)
        q8_codes = [source.randrange(-128, 128) for _ in range(32)]
        q8_block = struct.pack("<e32b", q8_scale, *q8_codes)
        torch.testing.assert_close(
            decode_q8_0(q8_block),
            torch.tensor([q8_scale * code for code in q8_codes], dtype=torch.float64),
            atol=0.0,
            rtol=0.0,
        )

        affine_scale = source.choice(half_values)
        minimum_scale = source.choice(half_values)
        scales = [source.randrange(64) for _ in range(8)]
        minimums = [source.randrange(64) for _ in range(8)]
        q4_codes = [source.randrange(16) for _ in range(256)]
        q5_codes = [source.randrange(32) for _ in range(256)]
        torch.testing.assert_close(
            decode_q4_k(
                _pack_q4_k(affine_scale, minimum_scale, scales, minimums, q4_codes)
            ),
            _q4_or_q5_expected(affine_scale, minimum_scale, scales, minimums, q4_codes),
            atol=0.0,
            rtol=0.0,
        )
        torch.testing.assert_close(
            decode_q5_k(
                _pack_q5_k(affine_scale, minimum_scale, scales, minimums, q5_codes)
            ),
            _q4_or_q5_expected(affine_scale, minimum_scale, scales, minimums, q5_codes),
            atol=0.0,
            rtol=0.0,
        )

        q6_scale = source.choice(half_values)
        q6_scales = [source.randrange(-128, 128) for _ in range(16)]
        q6_codes = [source.randrange(-32, 32) for _ in range(256)]
        torch.testing.assert_close(
            decode_q6_k(_pack_q6_k(q6_scale, q6_scales, q6_codes)),
            _q6_expected(q6_scale, q6_scales, q6_codes),
            atol=0.0,
            rtol=0.0,
        )


@pytest.mark.parametrize(
    ("decoder", "size"),
    (
        (decode_q8_0, Q8_0_BYTES),
        (decode_q4_k, Q4_K_BYTES),
        (decode_q5_k, Q5_K_BYTES),
        (decode_q6_k, Q6_K_BYTES),
    ),
)
def test_k_decoders_reject_non_block_sized_input(decoder: object, size: int) -> None:
    with pytest.raises(ValueError, match=f"exactly {size} bytes"):
        decoder(bytes(size - 1))  # type: ignore[operator]
    with pytest.raises(ValueError, match=f"exactly {size} bytes"):
        decoder(bytes(size + 1))  # type: ignore[operator]


def test_random_blocks_match_pinned_ggml_dequantization_when_available() -> None:
    """Optional, provenance-checked cross-check against a pinned shared ggml build.

    Set ``NINFER_LLAMA_CPP_DIR`` to the pinned source checkout and
    ``NINFER_GGML_BASE_LIBRARY`` to that checkout's shared ``ggml-base``
    library.  The independent logical-formula tests above are always run.
    """

    source_text = os.environ.get("NINFER_LLAMA_CPP_DIR")
    library_text = os.environ.get("NINFER_GGML_BASE_LIBRARY")
    if source_text is None or library_text is None:
        pytest.skip("pinned llama.cpp shared-library cross-check was not requested")

    source_dir = Path(source_text).resolve()
    library_path = Path(library_text).resolve()
    for relative_path, expected_hash in _PINNED_LLAMA_CPP_SOURCE_HASHES.items():
        source_hash = hashlib.sha256((source_dir / relative_path).read_bytes()).hexdigest()
        assert source_hash == expected_hash
    assert library_path.is_file()

    library = ctypes.CDLL(str(library_path))
    functions = {
        decode_q8_0: (library.dequantize_row_q8_0, 32, Q8_0_BYTES),
        decode_q4_k: (library.dequantize_row_q4_K, 256, Q4_K_BYTES),
        decode_q5_k: (library.dequantize_row_q5_K, 256, Q5_K_BYTES),
        decode_q6_k: (library.dequantize_row_q6_K, 256, Q6_K_BYTES),
    }
    for function, _, _ in functions.values():
        function.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_float), ctypes.c_int64]
        function.restype = None

    source = random.Random(0x9A4843CF)
    for _ in range(12):
        finite_half = source.choice((-1.75, -0.125, 0.0, 0.046875, 0.5, 2.25))
        blocks = {
            decode_q8_0: struct.pack("<e", finite_half) + source.randbytes(32),
            decode_q4_k: struct.pack("<ee", finite_half, source.choice((0.0, 0.25, 1.5)))
            + source.randbytes(140),
            decode_q5_k: struct.pack("<ee", finite_half, source.choice((0.0, 0.25, 1.5)))
            + source.randbytes(172),
            decode_q6_k: source.randbytes(208) + struct.pack("<e", finite_half),
        }
        for decoder, block in blocks.items():
            function, values, expected_bytes = functions[decoder]
            assert len(block) == expected_bytes
            output = (ctypes.c_float * values)()
            input_buffer = ctypes.create_string_buffer(block)
            function(input_buffer, output, values)
            ggml_result = torch.tensor(list(output), dtype=torch.float32)
            torch.testing.assert_close(
                decoder(block).to(torch.float32), ggml_result, atol=0.0, rtol=0.0
            )
