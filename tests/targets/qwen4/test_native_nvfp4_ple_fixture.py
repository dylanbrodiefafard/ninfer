"""Source PLE nibble/sign/scale domains and bounded row addressing."""

import math
import struct

import pytest

from tools.parity.qwen4.native_nvfp4_ple_fixture import bf16_word, decode_row, row_span


def test_decode_codes_scales_and_distinct_partition_multipliers():
    # Every signed E2M1 word, low nibble first, with exact unit E4M3 scale.
    codes = bytes((2 * i) | ((2 * i + 1) << 4) for i in range(8)) * 10
    scales = bytes([0x38]) * 10
    expected = (0., .5, 1., 1.5, 2., 3., 4., 6., -0., -.5, -1., -1.5, -2., -3., -4., -6.) * 10
    for multiplier in (.5, 2.):
        values = decode_row(codes, scales, struct.pack("<f", multiplier))
        assert values == [v * multiplier for v in expected]
        assert math.copysign(1., values[8]) == -1.
    # Minimum positive scale and maximum finite E4M3 scale.
    assert decode_row(bytes([0x22]) * 80, bytes([1]) * 10, struct.pack("<f", 1.))[0] == 2 ** -9
    assert decode_row(bytes([0x22]) * 80, bytes([126]) * 10, struct.pack("<f", 1.))[0] == 448.


@pytest.mark.parametrize("scale", [127, 128, 255])
def test_reject_invalid_scale(scale):
    with pytest.raises(ValueError, match="scale"):
        decode_row(bytes(80), bytes([scale]) * 10, struct.pack("<f", 1.))


def test_bf16_rounding_and_row_boundaries():
    assert bf16_word(1. + 2 ** -8) == 0x3f80
    assert bf16_word(1. + 3 * 2 ** -8) == 0x3f82
    assert bf16_word(-0.) == 0x8000
    item = dict(dtype="U8", shape=[3, 80], data_offsets=[19, 259])
    assert row_span(item, "U8", [3, 80], 0, 80) == (19, 98)
    assert row_span(item, "U8", [3, 80], 2, 80) == (179, 258)
    with pytest.raises(ValueError, match="span"):
        row_span(item, "U8", [3, 80], 3, 80)
