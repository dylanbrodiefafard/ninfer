"""`moe-expert-tiles-v1` against a scalar encoder written from the layout definition."""

from __future__ import annotations

import pytest
import torch

from tools.artifact.layouts import (
    ExpertMatrixWords,
    ExpertRecordWords,
    decode_expert_tiles_record,
    encode_expert_tiles_record,
    encoded_size,
    expert_tiles_geometry,
)

FORMATS = (("F3G32_E4M3FN", 3, 32), ("E2M1G16_E4M3FN", 4, 16))


def _random_words(bits: int, group: int, hidden: int, intermediate: int, seed: int):
    generator = torch.Generator().manual_seed(seed)

    def matrix(rows: int, columns: int) -> ExpertMatrixWords:
        codes = torch.randint(0, 1 << bits, (rows, columns), generator=generator, dtype=torch.uint8)
        scales = torch.randint(
            0, 0x7F, (rows, columns // group), generator=generator, dtype=torch.uint8
        )
        return ExpertMatrixWords(codes, scales)

    return ExpertRecordWords(
        matrix(intermediate, hidden), matrix(intermediate, hidden), matrix(hidden, intermediate)
    )


def _scalar_unit_words(codes: list[int]) -> list[int]:
    """The three code words of one F3 unit, placing every bit by the documented rule."""

    words = [0, 0, 0]
    for i, code in enumerate(codes):
        if i < 24:
            words[i // 8] |= code << (4 * (i % 8))
        else:
            for bit in range(3):
                words[bit] |= ((code >> bit) & 1) << (4 * (i - 24) + 3)
    return words


def _scalar_chunk(codes, scales, bits: int, chunk_bytes: int) -> bytearray:
    """One chunk from ``codes[rho][kappa]`` and ``scales[rho][group]`` by the header's formulas."""

    out = bytearray(chunk_bytes)
    for lane in range(32):
        g, c = lane // 4, lane % 4
        for k in range(4):
            rho = g + 8 * k
            unit = codes[rho][32 * c : 32 * c + 32]
            if bits == 3:
                out[4 * lane + k] = scales[rho][c]
                for j, word in enumerate(_scalar_unit_words(unit)):
                    at = 128 + 16 * (32 * j + lane) + 4 * k
                    out[at : at + 4] = word.to_bytes(4, "little")
            else:
                for h in range(2):
                    out[8 * lane + 2 * k + h] = scales[rho][2 * c + h]
                for j in range(4):
                    for b in range(4):
                        low = unit[8 * j + 2 * b]
                        out[256 + 16 * (32 * j + lane) + 4 * k + b] = low | (
                            unit[8 * j + 2 * b + 1] << 4
                        )
    return out


def _scalar_record(words: ExpertRecordWords, bits: int, group: int, hidden: int, intermediate: int):
    gate, up, down = (
        (matrix.codes.tolist(), matrix.scales.tolist())
        for matrix in (words.gate, words.up, words.down)
    )
    chunk_bytes = 1664 if bits == 3 else 2304
    per_chunk = 128 // group
    record = bytearray()
    for r in range(intermediate // 16):
        for q in range(hidden // 128):
            rows = [(gate, 16 * r + rho) for rho in range(16)] + [
                (up, 16 * r + rho) for rho in range(16)
            ]
            codes = [m[0][row][128 * q : 128 * q + 128] for m, row in rows]
            scales = [m[1][row][per_chunk * q : per_chunk * q + per_chunk] for m, row in rows]
            record += _scalar_chunk(codes, scales, bits, chunk_bytes)
    for r in range(hidden // 32):
        for q in range(intermediate // 128):
            rows = range(32 * r, 32 * r + 32)
            codes = [down[0][row][128 * q : 128 * q + 128] for row in rows]
            scales = [down[1][row][per_chunk * q : per_chunk * q + per_chunk] for row in rows]
            record += _scalar_chunk(codes, scales, bits, chunk_bytes)
    record += bytes(-len(record) % 256)
    return bytes(record)


@pytest.mark.parametrize(
    ("format_name", "record_bytes", "chunk_bytes"),
    [("F3G32_E4M3FN", 1_996_800, 1_664), ("E2M1G16_E4M3FN", 2_764_800, 2_304)],
)
def test_flash_next_expert_geometry(format_name, record_bytes, chunk_bytes):
    geometry = expert_tiles_geometry(format_name, (7, 2560, 640))
    assert geometry.record_bytes == record_bytes
    assert geometry.chunk_bytes == chunk_bytes
    assert (geometry.gate_up_tiles, geometry.gate_up_k_chunks) == (40, 20)
    assert (geometry.down_tiles, geometry.down_k_chunks) == (80, 5)
    assert encoded_size("moe-expert-tiles-v1", format_name, (7, 2560, 640)) == 7 * record_bytes


def test_f3_unit_known_answer():
    codes = list(range(8)) * 3 + [0b101, 0b010, 0b111, 0, 1, 2, 4, 6]
    words = ExpertRecordWords(
        *(
            ExpertMatrixWords(
                torch.zeros(shape, dtype=torch.uint8), torch.zeros(scales, dtype=torch.uint8)
            )
            for shape, scales in (
                ((128, 128), (128, 4)),
                ((128, 128), (128, 4)),
                ((128, 128), (128, 4)),
            )
        )
    )
    # Gate row 9 is local row rho = 9 (g = 1, k = 1) of chunk 0; K offsets 32..63 are c = 1, so
    # lane L = 5 and the unit's word j lies at 128 + 16 (32 j + 5) + 4.
    words.gate.codes[9, 32:64] = torch.tensor(codes, dtype=torch.uint8)
    words.gate.scales[9, 1] = 0x42
    record = encode_expert_tiles_record(words, "F3G32_E4M3FN", 128, 128)
    assert int(record[4 * 5 + 1]) == 0x42
    unit = [
        int.from_bytes(bytes(record[at : at + 4].tolist()), "little")
        for at in (128 + 16 * (32 * j + 5) + 4 for j in range(3))
    ]
    assert unit == _scalar_unit_words(codes)
    # Word 0 holds codes 0..7 in its nibbles' low bits and bit 0 of codes 24..31 in their MSBs:
    # codes 24 (0b101), 26 (0b111) and 28 (0b001) set the MSBs of nibbles 0, 2 and 4.
    assert unit[0] == 0x76543210 | 0x8 | 0x800 | 0x80000
    # Word 2 holds bit 2: codes 24 (0b101), 26, 30 (0b100) and 31 (0b110).
    assert unit[2] == 0x76543210 | 0x8 | 0x800 | 0x8000000 | 0x80000000


@pytest.mark.parametrize(("format_name", "bits", "group"), FORMATS)
def test_record_matches_scalar_encoder(format_name, bits, group):
    words = _random_words(bits, group, 256, 128, seed=bits)
    record = encode_expert_tiles_record(words, format_name, 256, 128)
    assert bytes(record.tolist()) == _scalar_record(words, bits, group, 256, 128)


@pytest.mark.parametrize(("format_name", "bits", "group"), FORMATS)
def test_real_shape_round_trip(format_name, bits, group):
    words = _random_words(bits, group, 2560, 640, seed=7 + bits)
    record = encode_expert_tiles_record(words, format_name, 2560, 640)
    decoded = decode_expert_tiles_record(memoryview(record.numpy()), format_name, 2560, 640)
    for name in ("gate", "up", "down"):
        assert torch.equal(getattr(decoded, name).codes, getattr(words, name).codes)
        assert torch.equal(getattr(decoded, name).scales, getattr(words, name).scales)


@pytest.mark.parametrize("word", [0x7F, 0x80, 0xFF])
def test_invalid_scale_words_are_rejected(word):
    words = _random_words(3, 32, 256, 128, seed=3)
    words.down.scales[5, 0] = word
    with pytest.raises(ValueError, match="E4M3FN"):
        encode_expert_tiles_record(words, "F3G32_E4M3FN", 256, 128)


def test_geometry_rejects_untileable_shapes():
    with pytest.raises(ValueError):
        expert_tiles_geometry("F3G32_E4M3FN", (1, 2560 + 64, 640))
    with pytest.raises(ValueError):
        expert_tiles_geometry("E2M1G16_E4M3FN", (1, 2560, 640 + 32))
    with pytest.raises(ValueError):
        expert_tiles_geometry("NVFP4", (1, 2560, 640))
