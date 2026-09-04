"""Exact scalar decoders for the non-IQ GGML blocks in UD-IQ1_S Text.

These functions are mathematical format oracles, not production arithmetic
routes.  Every multi-byte field is little-endian and every ``d``/``dmin`` is
an IEEE binary16 value.  Binary16 values are represented exactly in Python
``float``; outputs are returned as ``torch.float64`` so the integer scale,
minimum, and code operations below do not add a float32 rounding boundary.

The semantic outputs of the four formats are:

* Q8_0: 32 signed int8 codes, ``y[i] = d * q[i]``.
* Q4_K: eight 32-value groups with unsigned 4-bit codes,
  ``y = d * scale[group] * q - dmin * minimum[group]``.
* Q5_K: the same affine formula with unsigned 5-bit codes.  The fifth bit is
  stored in a separate 32-byte plane.
* Q6_K: sixteen 16-value groups with signed int8 group scales and signed
  6-bit codes represented as unsigned bits minus 32,
  ``y = d * scale[group] * (unsigned_code - 32)``.

Q4_K and Q5_K pack eight unsigned 6-bit scales and eight unsigned 6-bit
minimum multipliers into 12 bytes.  Bytes 0..3/4..7 hold the low six bits for
groups 0..3; for groups 4..7, bytes 8..11 hold the low nibbles of scale/min,
while the high two bits live in the high bits of bytes 0..3/4..7.  The
subtracted ``dmin * minimum`` term is the format's represented affine offset;
neither the stored minimum multiplier nor ``dmin`` is sign-extended.
"""

from __future__ import annotations

import struct

import torch


Q8_0_VALUES = 32
Q8_0_BYTES = 34
Q4_K_VALUES = 256
Q4_K_BYTES = 144
Q5_K_VALUES = 256
Q5_K_BYTES = 176
Q6_K_VALUES = 256
Q6_K_BYTES = 210


def _block_bytes(block: bytes | bytearray | memoryview, expected: int, name: str) -> bytes:
    value = bytes(block)
    if len(value) != expected:
        raise ValueError(f"{name} block must contain exactly {expected} bytes")
    return value


def _scale_min_6bit(packed: bytes, group: int) -> tuple[int, int]:
    """Return the unsigned 6-bit scale/minimum pair for one 32-value group."""

    if group < 4:
        return packed[group] & 0x3F, packed[group + 4] & 0x3F
    scale = (packed[group + 4] & 0x0F) | ((packed[group - 4] >> 6) << 4)
    minimum = (packed[group + 4] >> 4) | ((packed[group] >> 6) << 4)
    return scale, minimum


def decode_q8_0(block: bytes | bytearray | memoryview) -> torch.Tensor:
    """Decode one 34-byte Q8_0 block into 32 exact represented values.

    Bytes 0..1 are little-endian binary16 ``d``.  Bytes 2..33 are two's
    complement signed int8 codes in output order; ``y[i] = d * q[i]``.
    """

    packed = _block_bytes(block, Q8_0_BYTES, "Q8_0")
    scale = struct.unpack_from("<e", packed, 0)[0]
    codes = struct.unpack_from("<32b", packed, 2)
    return torch.tensor([scale * code for code in codes], dtype=torch.float64)


def decode_q4_k(block: bytes | bytearray | memoryview) -> torch.Tensor:
    """Decode one 144-byte Q4_K block into 256 exact represented values.

    The little-endian binary16 ``d`` and ``dmin`` occupy bytes 0..3, the
    packed 6-bit scale/minimum table bytes 4..15, and quants bytes 16..143.
    Each run of 32 quant bytes carries one low-nibble output group followed
    by one high-nibble group.  Group ``g`` therefore uses quant-byte run
    ``g//2`` and nibble ``g%2``.
    """

    packed = _block_bytes(block, Q4_K_BYTES, "Q4_K")
    scale, minimum_scale = struct.unpack_from("<ee", packed, 0)
    scale_min = packed[4:16]
    quants = packed[16:]
    values: list[float] = []
    for group in range(8):
        group_scale, group_minimum = _scale_min_6bit(scale_min, group)
        delta = scale * group_scale
        offset = minimum_scale * group_minimum
        quant_offset = 32 * (group // 2)
        shift = 4 * (group & 1)
        values.extend(
            delta * ((quants[quant_offset + lane] >> shift) & 0x0F) - offset
            for lane in range(32)
        )
    return torch.tensor(values, dtype=torch.float64)


def decode_q5_k(block: bytes | bytearray | memoryview) -> torch.Tensor:
    """Decode one 176-byte Q5_K block into 256 exact represented values.

    Bytes 0..15 have the Q4_K scale/minimum layout.  Bytes 16..47 contain
    the fifth-bit plane and bytes 48..175 contain paired low/high nibbles.
    Bit ``g`` of high-plane byte ``lane`` is the fifth bit for lane ``lane``
    of group ``g``.  Codes are unsigned 0..31 and use the Q4_K affine
    ``d * scale * q - dmin * minimum`` formula.
    """

    packed = _block_bytes(block, Q5_K_BYTES, "Q5_K")
    scale, minimum_scale = struct.unpack_from("<ee", packed, 0)
    scale_min = packed[4:16]
    high_bits = packed[16:48]
    low_bits = packed[48:]
    values: list[float] = []
    for group in range(8):
        group_scale, group_minimum = _scale_min_6bit(scale_min, group)
        delta = scale * group_scale
        offset = minimum_scale * group_minimum
        quant_offset = 32 * (group // 2)
        shift = 4 * (group & 1)
        values.extend(
            delta
            * (
                ((low_bits[quant_offset + lane] >> shift) & 0x0F)
                | (((high_bits[lane] >> group) & 1) << 4)
            )
            - offset
            for lane in range(32)
        )
    return torch.tensor(values, dtype=torch.float64)


def decode_q6_k(block: bytes | bytearray | memoryview) -> torch.Tensor:
    """Decode one 210-byte Q6_K block into 256 exact represented values.

    Bytes 0..127 hold low nibbles, bytes 128..191 hold two-bit high planes,
    bytes 192..207 are sixteen two's-complement signed int8 group scales,
    and bytes 208..209 are little-endian binary16 ``d``.  In each 128-value
    half, a 32-byte high-plane run interleaves four 2-bit planes; the two
    corresponding 32-byte low-nibble runs interleave groups 0/2 and 1/3.
    Reassembled codes are unsigned 0..63, then explicitly centered by 32.
    """

    packed = _block_bytes(block, Q6_K_BYTES, "Q6_K")
    low_bits = packed[:128]
    high_bits = packed[128:192]
    group_scales = struct.unpack_from("<16b", packed, 192)
    scale = struct.unpack_from("<e", packed, 208)[0]

    values: list[float] = []
    for index in range(256):
        half = index // 128
        within_half = index & 127
        group32 = within_half // 32
        lane = within_half & 31
        low_byte = low_bits[64 * half + lane + 32 * (group32 & 1)]
        low_nibble = (low_byte >> (4 * (group32 // 2))) & 0x0F
        high_pair = (high_bits[32 * half + lane] >> (2 * group32)) & 0x03
        signed_code = (low_nibble | (high_pair << 4)) - 32
        group_scale = group_scales[index // 16]
        values.append(scale * group_scale * signed_code)
    return torch.tensor(values, dtype=torch.float64)


__all__ = [
    "Q8_0_VALUES",
    "Q8_0_BYTES",
    "Q4_K_VALUES",
    "Q4_K_BYTES",
    "Q5_K_VALUES",
    "Q5_K_BYTES",
    "Q6_K_VALUES",
    "Q6_K_BYTES",
    "decode_q8_0",
    "decode_q4_k",
    "decode_q5_k",
    "decode_q6_k",
]
