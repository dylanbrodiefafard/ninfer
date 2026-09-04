from __future__ import annotations

import ctypes
import hashlib
import os
from pathlib import Path
import random
import struct

import pytest
import torch

from tools.reference.qwen4.ggml_codecs import (
    IQ1_S_CODEBOOK,
    IQ2_XXS_CODEBOOK,
    IQ4_NL_CODEBOOK,
    decode_iq1_s,
    decode_iq2_xxs,
    decode_iq4_nl,
)


_PINNED_LLAMA_CPP_SOURCE_HASHES = {
    "ggml/src/ggml-quants.c": (
        "07143d7068936ae46b3c528b2f3d4bbb666e74d88992165716174d243573965d"
    ),
    "ggml/src/ggml-common.h": (
        "0061131b615c5721fc88a78feeb22c1f8c450f1c2646a317d80796a653bf595c"
    ),
}


def _manual_iq1_s(block: bytes) -> torch.Tensor:
    d = struct.unpack_from("<e", block)[0]
    result = []
    for group in range(8):
        qh = int.from_bytes(block[34 + 2 * group : 36 + 2 * group], "little")
        dl = d * (2 * ((qh >> 12) & 7) + 1)
        delta = -1 / 8 if qh & 0x8000 else 1 / 8
        for lane in range(4):
            grid_id = block[2 + 4 * group + lane] + 256 * ((qh >> (3 * lane)) & 7)
            for grid_value in IQ1_S_CODEBOOK[grid_id]:
                result.append(dl * (grid_value + delta))
    return torch.tensor(result, dtype=torch.float64)


def _manual_iq2_xxs(block: bytes) -> torch.Tensor:
    d = struct.unpack_from("<e", block)[0]
    result = []
    for group in range(8):
        q0, q1 = struct.unpack_from("<II", block, 2 + 8 * group)
        db = d * (q1 // (1 << 28) + 1 / 2) / 4
        grid_ids = q0.to_bytes(4, "little")
        for lane, grid_id in enumerate(grid_ids):
            sign_id = (q1 >> (7 * lane)) & 127
            sign_bits = sign_id | ((sign_id.bit_count() & 1) << 7)
            for element, magnitude in enumerate(IQ2_XXS_CODEBOOK[grid_id]):
                result.append(db * magnitude * (-1 if sign_bits & (1 << element) else 1))
    return torch.tensor(result, dtype=torch.float64)


def _manual_iq4_nl(block: bytes) -> torch.Tensor:
    d = struct.unpack_from("<e", block)[0]
    result = [d * IQ4_NL_CODEBOOK[value & 15] for value in block[2:]]
    result.extend(d * IQ4_NL_CODEBOOK[value // 16] for value in block[2:])
    return torch.tensor(result, dtype=torch.float64)


def test_format_codebook_anchors_are_not_linear_quantizers() -> None:
    assert IQ1_S_CODEBOOK[0] == (-1,) * 8
    assert IQ1_S_CODEBOOK[1] == (1, -1, -1, -1, -1, -1, -1, -1)
    assert IQ1_S_CODEBOOK[2] == (0, 0, -1, -1, -1, -1, -1, -1)
    assert IQ1_S_CODEBOOK[2047] == (1,) * 8
    assert IQ2_XXS_CODEBOOK[0] == (8,) * 8
    assert IQ2_XXS_CODEBOOK[1] == (43, 8, 8, 8, 8, 8, 8, 8)
    assert IQ2_XXS_CODEBOOK[2] == (25, 25, 8, 8, 8, 8, 8, 8)
    assert IQ2_XXS_CODEBOOK[255] == (8, 25, 8, 8, 25, 43, 43, 43)


def test_iq4_nl_low_nibbles_precede_high_nibbles() -> None:
    block = struct.pack("<e", 0.5) + bytes(((15 - i) << 4) | i for i in range(16))
    actual = decode_iq4_nl(block)
    expected = torch.tensor(
        [0.5 * value for value in IQ4_NL_CODEBOOK]
        + [0.5 * IQ4_NL_CODEBOOK[15 - i] for i in range(16)],
        dtype=torch.float64,
    )
    torch.testing.assert_close(actual, expected, atol=0.0, rtol=0.0)


def test_iq2_xxs_hand_packed_endian_scale_sign_and_grid_witness() -> None:
    sign_ids = (1, 2, 3, 127)
    q1 = (10 << 28) | sum(value << (7 * lane) for lane, value in enumerate(sign_ids))
    first_group = bytes((0, 1, 2, 255)) + q1.to_bytes(4, "little")
    block = struct.pack("<e", -0.75) + first_group + bytes(7 * 8)
    actual = decode_iq2_xxs(block)

    db = -0.75 * 10.5 / 4
    expected_first_group = []
    for grid_id, sign_id in zip((0, 1, 2, 255), sign_ids, strict=True):
        sign_bits = sign_id | ((sign_id.bit_count() & 1) << 7)
        expected_first_group.extend(
            db * value * (-1 if sign_bits & (1 << element) else 1)
            for element, value in enumerate(IQ2_XXS_CODEBOOK[grid_id])
        )
    torch.testing.assert_close(
        actual[:32],
        torch.tensor(expected_first_group, dtype=torch.float64),
        atol=0.0,
        rtol=0.0,
    )
    assert actual[0].item() == -db * 8
    assert actual[7].item() == -db * 8  # parity-completed eighth sign bit


def test_iq1_s_hand_packed_high_indices_odd_scale_and_delta_witness() -> None:
    low = bytearray(32)
    low[:4] = bytes((0, 255, 0, 255))
    high_ids = (0, 7, 1, 6)
    qh = sum(value << (3 * lane) for lane, value in enumerate(high_ids))
    qh |= 5 << 12
    qh |= 1 << 15
    controls = qh.to_bytes(2, "little") + bytes(14)
    block = struct.pack("<e", 0.625) + low + controls
    actual = decode_iq1_s(block)

    dl = 0.625 * 11
    expected = []
    for low_id, high_id in zip(low[:4], high_ids, strict=True):
        expected.extend(dl * (value - 1 / 8) for value in IQ1_S_CODEBOOK[low_id + 256 * high_id])
    torch.testing.assert_close(
        actual[:32], torch.tensor(expected, dtype=torch.float64), atol=0.0, rtol=0.0
    )


def test_random_packed_blocks_match_direct_formulas() -> None:
    random_source = random.Random(0x5138)
    for _ in range(20):
        iq1 = struct.pack("<e", random_source.choice((-1.5, -0.25, 0.125, 2.0))) + random_source.randbytes(48)
        iq2 = struct.pack("<e", random_source.choice((-2.0, -0.5, 0.25, 1.25))) + random_source.randbytes(64)
        iq4 = struct.pack("<e", random_source.choice((-1.0, -0.375, 0.5, 1.75))) + random_source.randbytes(16)
        torch.testing.assert_close(decode_iq1_s(iq1), _manual_iq1_s(iq1), atol=0.0, rtol=0.0)
        torch.testing.assert_close(decode_iq2_xxs(iq2), _manual_iq2_xxs(iq2), atol=0.0, rtol=0.0)
        torch.testing.assert_close(decode_iq4_nl(iq4), _manual_iq4_nl(iq4), atol=0.0, rtol=0.0)


@pytest.mark.parametrize(
    ("decoder", "size"),
    ((decode_iq1_s, 50), (decode_iq2_xxs, 66), (decode_iq4_nl, 18)),
)
def test_decoders_reject_non_block_sized_input(decoder: object, size: int) -> None:
    with pytest.raises(ValueError, match=f"exactly {size} bytes"):
        decoder(bytes(size - 1))  # type: ignore[operator]
    with pytest.raises(ValueError, match=f"exactly {size} bytes"):
        decoder(bytes(size + 1))  # type: ignore[operator]


def test_random_iq_blocks_match_pinned_ggml_dequantization_when_available() -> None:
    """Optional cross-check; the direct mathematical tests above remain the oracle."""

    source_text = os.environ.get("NINFER_LLAMA_CPP_DIR")
    library_text = os.environ.get("NINFER_GGML_BASE_LIBRARY")
    if source_text is None or library_text is None:
        pytest.skip("pinned llama.cpp shared-library cross-check was not requested")

    source_dir = Path(source_text).resolve()
    library_path = Path(library_text).resolve()
    for relative_path, expected_hash in _PINNED_LLAMA_CPP_SOURCE_HASHES.items():
        assert hashlib.sha256((source_dir / relative_path).read_bytes()).hexdigest() == expected_hash
    assert library_path.is_file()

    library = ctypes.CDLL(str(library_path))
    functions = {
        decode_iq1_s: (library.dequantize_row_iq1_s, 256, 50),
        decode_iq2_xxs: (library.dequantize_row_iq2_xxs, 256, 66),
        decode_iq4_nl: (library.dequantize_row_iq4_nl, 32, 18),
    }
    for function, _, _ in functions.values():
        function.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_float), ctypes.c_int64]
        function.restype = None

    source = random.Random(0x15384EED)
    for _ in range(12):
        scale = source.choice((-1.75, -0.125, 0.0, 0.046875, 0.5, 2.25))
        blocks = {
            decode_iq1_s: struct.pack("<e", scale) + source.randbytes(48),
            decode_iq2_xxs: struct.pack("<e", scale) + source.randbytes(64),
            decode_iq4_nl: struct.pack("<e", scale) + source.randbytes(16),
        }
        for decoder, block in blocks.items():
            function, values, expected_bytes = functions[decoder]
            assert len(block) == expected_bytes
            output = (ctypes.c_float * values)()
            input_buffer = ctypes.create_string_buffer(block)
            function(input_buffer, output, values)
            torch.testing.assert_close(
                decoder(block).to(torch.float32),
                torch.tensor(list(output), dtype=torch.float32),
                atol=0.0,
                rtol=0.0,
            )
