"""Exact scalar oracles for the GGML blocks used by the preview GGUF.

These are format decoders, not implementations of a production arithmetic
route.  Every multi-byte field is little-endian.  The first two bytes of each
block are an IEEE binary16 scale; converting that represented value to
``float64`` is exact.  The returned tensors use ``float64`` so the remaining
rational scale/codebook operations do not introduce a production-specific
float32 rounding boundary.

The two importance-quant codebooks below are immutable format data.  They are
stored compactly as zlib-compressed, base64-encoded little-endian uint16
values.  Each uint16 contains eight 2-bit digits, least-significant digit
first.  IQ1_S maps digits 0/1/2 to -1/0/+1; IQ2_XXS maps them to 8/25/43.
"""

from __future__ import annotations

import base64
import struct
import zlib

import torch


IQ1_S_VALUES = 256
IQ1_S_BYTES = 50
IQ2_XXS_VALUES = 256
IQ2_XXS_BYTES = 66
IQ4_NL_VALUES = 32
IQ4_NL_BYTES = 18

IQ4_NL_CODEBOOK = (
    -127,
    -104,
    -83,
    -65,
    -49,
    -35,
    -22,
    -10,
    1,
    13,
    25,
    38,
    53,
    69,
    89,
    113,
)


def _packed_codebook(encoded: bytes, entries: int, values: tuple[int, int, int]) -> tuple[tuple[int, ...], ...]:
    packed = zlib.decompress(base64.b64decode(encoded))
    if len(packed) != 2 * entries:
        raise AssertionError("invalid embedded GGML codebook length")
    result = []
    for offset in range(0, len(packed), 2):
        word = int.from_bytes(packed[offset : offset + 2], "little")
        digits = tuple(values[(word >> (2 * lane)) & 0x3] for lane in range(8))
        result.append(digits)
    return tuple(result)


_IQ2_XXS_PACKED = (
    b"eNoFwcFq4kAYAOB/hn9kzGlGosQ9jRIXt0/xJyQSexqLlWxPiSi4bzEJo3R7isVCe7PFQ/ct9/sAOAiQEICGEAxMYQ53kEAG"
    b"FkqoYQcOWniBL2AMmWJDRsyzdwac8ylvOUNEhROMkTDFAits0GOHNwTBhRRGLMQf4cRZ9HuqRz3fu/VACillKOcykZm08kE+"
    b"S+xTnwdhwBQqpSaKVKW86tSbAi210VY/aafjQT6AUIYmnIXL0IXRcDuE0WjEIoxURFETwViMr+PVDzDcBCYztXGmNb8mxQSm"
    b"fMpijFVM8SaGWTLbzbqfcm7ml3l2xwhJUUQTiomooA1V1JCnjiAxSZ245JiwVKVFukkPKWQyM5nLPrJBHuewYAUWqqAiL2BZ"
    b"Lutlu8zvu3uw0mo7tsY6+2xxla5uq/NDt35dB49ss95AqctxuSvpt3zCSlVUHSqoH+toW21hJ3ZqP9j7fXoAx13gEtc6bKKG"
    b"Gt0yj175oSdfeTgujpdj/0Sn6sSf8a9+wU51cUfdZzc6u3P+CpfpJbus3pp388Gv9nq89j/hK7rZb/z3HwV8b10="
)

_IQ1_S_PACKED = (
    b"eNoNl0HKslAAAEsyylWGSrayyKhOkR8q2convoe6ylDJTlGRYa4yfJKtLDKqU/7/CWYzMEytRtTIWqtG1egaWxNqw9qkNqsp"
    b"NVAza6jm13a1Qy2uJTVcK2vP2qf2qzXqZJ2uM3Wuztf7dbEu1dW6VjfqsO7U3bpXD+qbelRP61k9r1f1GkEQLYIiWEIghsSE"
    b"mBEKAQib8IiQ2BEHIiYSIiUwURAl8SQ+xI+gG0yDa4gNqaE1YMNpeA2/kTaKRvUf3CBJskmyZI/sk1NyTirkktRJQJokJBFp"
    b"kytyTfrkhtySezIlMXkj7+SDfJEV+Sa/JNPkm1JTbupNowmbVnPVdJtBM2xGzbSZNYtmrUW0Wi2qxbaE1rA1ac1aSgu0UMtv"
    b"7VqHVtxKWrhVtp6tT+vXItt0m2nz7VFbbEttvQ3asO22vXbYTttZO28X7apdowiqRVEUSwnUkJpQM0qhAGVTLuVTO+pARVRM"
    b"JRSmSupJfagf1enQHabDd0YdsSN15I7egR2r43a8jt8JO2kn6+SdR6fq1OkG3aTbdIfu0izdowf0iJ7SCr2kdRrQBm3SkEa0"
    b"Ta9onz7SVxrTD/pF012my3XF7ryrdrWu0YVdq+t0vW7QjbppN+vm3apbZ5pMm2EYlukxPDNgxozEKIzKLJglAxiTgQxibGbN"
    b"+MyGOTJn5sJkDGZuTMHcmQfzYirmyxAsyVIszTIsy3IszwrskBXZCTtjJVZmFVZlAWuwJgtZxNqsw7qsx/pswIbsjj2wRzZi"
    b"YzZhL2zKZixmc7Zg72zJPtmKrXMNjuSaHMtxXI/rcwNuzM25P07mFG7BLTnAQQ5xFmdzLudxPrfhQm7LnbiEu3KYe3Ffju5x"
    b"PbEn9eSe2tN6eg/2rJ7TW/XcntcLemEv6qW9rFf1OnyXZ/k+P+BF/o+XeYVf8IA3ecgj3uYdfsX7/JY/8yl/5TF/4x/8m//y"
    b"7T7XH/XHfbmv9rW+3jf6sG/13X7QD/tRP+3n/Xu/JhBCS6AEVhCEoSAKE2EmKAIQbMEVfGEnHIRYSAQslMJTqISP8BPIAT1g"
    b"BvxAHPwN5IE2gANr4AzcgTfwB8EgGlwG+aAYVIP6sDWkhvSQHQrD4XAynA2VIRiioT30h/thPEyG6RAPy+Fz+Bn+huSIGXEj"
    b"fiSO5JEyUkfayBjBkTVyRsEoGqWjbFSMHqPqv96sOBCn4lxUxIUIRFOEoi2uRV/ciGfxImYiFm/iXXyIL/EtfkVy3Bl3x/xY"
    b"HEtjbQzHq7E7DsfH8Wl8Gd/HtQkxaU2oCTsRJsPJZDKbKBMwMSf+ZDc5TOJJMiknz8ln8pu0p/SUmfJTcapOtakxhVN36k2D"
    b"aTiNppdpPi2mr2k1rc2IWWtGzYTZcDaZzWbKDMzQzJ75s93sMItnyQzPytlz9pn9ZuScnnNzca7NjTmcW3Nn7s69eTDP5sX8"
    b"MX/Pa1JdakhNqS11JVbipJ7UlwbSWJpKirSQlhKQTAlKSLIlR/KljbSV9tJROkkX6SphqZQe0lMi/+g/5o/7E/+kP+MP/jl/"
    b"3l/4d/7L/qq/uszKvDyVFXkhL2UgmzKUkezKa9mXN/JW3ssn+Sxf5KuM5VJ+yF+5rhAKqVAKrTAKq3AKrwiKqMwUSZEVRVEV"
    b"TdEVoJgKVJBiKbbiKp7iK4ESKgflqERKrKRKpmAlVwrlrlTKR/kpdZVU2yqjsmpP7asDdaRO1bn6pyrqQtVVoBoqVJFqq2vV"
    b"VzfqXo3URM1UrD7Ul/pWyQW9YBf9hbj4W2gLfQEX1sJdeItgES6iRbrIFvmiWFSLukZqTY3SOhqjsVpPG2gjbazNNUVbakAz"
    b"NFODGtJsbaWtNV8LtK120s7aVcNart20h/bS3tpX45byUl1qS7i0ls7SW4bLbFkt63pDJ/Wm3ta7Oqv39YE+0qf6XFf0hQ50"
    b"U4c60m3d1zf6ST/rWL/ppf7Q3/r3fyhaoA0oQAMGsIADPcADAYhgDCZgBiQgAwWoQAM6AMAAJoAAAQvYwAEu8IAPAhCCA4hA"
    b"CjKAQQ4KUIIK/EDdaBpdgzX6xsAYGX+GYiwNYJgGNJBhG2vDNyLjYlwNbBTG3XgZDZM0aZMxWZMzeyZvDkzRnJgzUzJlUzFV"
    b"UzOXpm4C0zShiUzLtE3HdM216Zm+GZihuTNjMzFTMzOxmZuF+TBfZmX+zDokYAOSsAnbsANp2IUMZCEHediHAziCIhzDKZxD"
    b"Cf5BGSpQhQuoQR0CaEATQoigBW3owBV0oQd9GMANDOEW7uERRvAME3iBKcwghjm8wQI+4AtW8A2/sIbqiEAN1EQt1EY0YhCL"
    b"eohHAhqgIRohEY3RBE2RhBSkogXS0BLpCCADmQgihCxkIwe5yEM+CtEBReiEYnRGCUoRRnf0RBV6ow/6ooZFWk2rbXUs1upZ"
    b"A2tmKdbCWlrAMi1oIcuybGtlrS3P8q2DdbYu1tXC1s16WF+rbhM2aVM2bTM2a3M2b4u2ZMu2Yqu2Zus2sA3btKGNbMu2bcd2"
    b"bc/27cAO7b0d2Wc7tTMb27l9swu7shtOy2GdviM4ojN2po7iLBzNAQ50kGM5trN2fGfjbJ29kzhXBzu5c3PuzsMhV8yKW/Er"
    b"cSWv9BVcoZW1clbuylsFq3C1X+WralV3G27Tbbtdl3UH7tAdu1NXcTUXuNBFru36buBu3aObuFcXuw/37X5des2t+fV8La3VNVyj"
    b"tbVeraN1us7X1Zr2uh7rcV7fG3hjb+rNvT9P8Rbe0gOe6UEPebbneCtv7fne0Yu8s3fxrl7mYS/3bt7de3gv7+uRfsun"
    b"fNpnfc7nfdlXfNXXfN0HvulDH/m27/qe7/uBH/on/+wnfupjP/cL/+4//cp/+x+fCNoBGwjBOJgE02AeKMEiWAYgMAMYoMAK"
    b"nGAV+MEm2AWHIAqSIAvy4BYUQRm8gnfwC7gNvxE30sbYwI2zcTfhJtqkm9vmvamHjbATsuEgHIXjcBrOQylUQjVchCA0Qxii"
    b"0A5XoR9uw0N4DJMQh4/wFVbhN6S33La3lbbyVtvqW7i1ts7W2/rbcHvaZtvb9r59b2s7YtfaUTthN9xNdrOdstN3YGfu0M7e"
    b"+bvd7rCLd8kO78rdc/fZ/Xbknt4ze27P78W9tJf32l7fG3u4R3trb++9fbAP99H+vM/2+b7YV/vagTi0DtSBPQiH4WFymB3A"
    b"wTzYB/+wOxwO8SE54EN5eB4+h9+BOfJH6SgfwREenaN79I7hMTsWx3rUjroRG/WjcTSN5pEUKdEiApEZwQhFduREfhREm2gb"
    b"7aNjdIrO0SW6Rji6Re+IPnEn/iSepJN80k7Lk36CJ/vknNxTcNqeolN6up9ep1pMxK2YitlYiIfxJJ7FUqzEIDZjO/bjMN7F"
    b"hziOkxjHZfyMP/EvJs/NM31mztxZPEtn+ayetbN+Ns7w7Jzds3eOzvm5OFfnWkIkrYRK2ERIhskkmSVKAhIzQckuOSRxkiQ4"
    b"KZNn8kl+CXmhL9yld+Ev4kW6qBftAi/WxbmEl+0luqSX7JJfikt1qaeNtJm2007Kpr20nw7SUTpOp+k8VVI9BamZwhSldrpO"
    b"/fSYntJritNb+khf6Tv9puSVvjJX/ipe5at61a761bjCq3UNruE1umbX/Pq91rNG1sw6GZv1snE2z5YZyMwMZiizMjtbZW62"
    b"zvzsmJ2ya5ZlOLtlj+yb1TCJW5jCHUxjBrOYwzweYBFP8QxLWMYKVrGGdQywgU0MMcIWtrGDXexhHwc4xHsc4Rin+IozjHGO"
    b"C3zHJX7iCn/wD9fzRt7J2ZzPhXycT3MlX+RaDnIjhznK7dzPN/khP+bnPMmveZbj/JW/829O3rgbfxNv0k296TfjBm/o5ty8"
    b"m3+LbuktvxW36tYomkW76BTdgi16Rb8QikExKsbFvPgrlGJRLAtQmAUsUGEX68IvgmJb7ItjcSmuBS7uxaN4F+SdvYt3+a7e"
    b"tbt+h3fr7t6je3rP7vie32slUbZKqmRLoRyWk3JWKiUozRKVdrkrD2VcJiUuy/JZfspfST7aD/rBPLgH/+g/1If2AA/4sB7O"
    b"w314j+hxeVwf+aN4EM/Wk3p2nvxz+Jw8Z0/lCZ7oaT/95+55eMbP5Imf5fP5/Dx/T/4lvqSX/NJf5gu+rJfzcl/+K3htXuEr"
    b"ejWrdtWpuhVb9apxNa3+KqUClVnBClV25Vfbal8dq6g6VefqWuHq9n8+6Tf37r8Hb/Etv9X38m284Ru9rffqvX6f3pc3fufv"
    b"4v14v97vd+1DfFof6iN8hp/JZ/YBH/ODPvZn9zl84k/ywZ/y8/x8Pr8P+WW+/HfwFb/SV//Cr/N1v8E3/F6++bf2I36tH/UT"
    b"fsPf5Df7gZ/5Q7/d7/CLf8kP/8rf8/f5/X7/AK17TfE="
)

IQ1_S_CODEBOOK = _packed_codebook(_IQ1_S_PACKED, 2048, (-1, 0, 1))
IQ2_XXS_CODEBOOK = _packed_codebook(_IQ2_XXS_PACKED, 256, (8, 25, 43))


def _block_bytes(block: bytes | bytearray | memoryview, expected: int, name: str) -> bytes:
    value = bytes(block)
    if len(value) != expected:
        raise ValueError(f"{name} block must contain exactly {expected} bytes")
    return value


def _f16_le(block: bytes) -> float:
    return struct.unpack_from("<e", block, 0)[0]


def decode_iq4_nl(block: bytes | bytearray | memoryview) -> torch.Tensor:
    """Decode one 18-byte IQ4_NL block into 32 represented values.

    Byte ``2+j`` carries output ``j`` in its low nibble and output ``16+j``
    in its high nibble.  For code ``q``, ``y = binary16(d) * codebook[q]``.
    """

    packed = _block_bytes(block, IQ4_NL_BYTES, "IQ4_NL")
    scale = _f16_le(packed)
    codes = packed[2:]
    values = [scale * IQ4_NL_CODEBOOK[byte & 0x0F] for byte in codes]
    values.extend(scale * IQ4_NL_CODEBOOK[byte >> 4] for byte in codes)
    return torch.tensor(values, dtype=torch.float64)


def decode_iq2_xxs(block: bytes | bytearray | memoryview) -> torch.Tensor:
    """Decode one 66-byte IQ2_XXS block into 256 represented values.

    After ``d``, each 8-byte/32-value group is two little-endian uint32s.
    The four bytes of the first word select 8-value magnitude grids.  Bits
    0..27 of the second word are four 7-bit sign-pattern indices; bits 28..31
    hold the group scale ``s``.  The sign table completes the seven stored
    bits with an eighth parity bit.  Thus ``db = d*(s + 1/2)/4`` and
    ``y[32*g+8*l+j] = db * grid[index][j] * sign``.
    """

    packed = _block_bytes(block, IQ2_XXS_BYTES, "IQ2_XXS")
    scale = _f16_le(packed)
    values: list[float] = []
    for group in range(8):
        low, high = struct.unpack_from("<II", packed, 2 + 8 * group)
        group_scale = scale * (0.5 + (high >> 28)) * 0.25
        for lane in range(4):
            grid_index = (low >> (8 * lane)) & 0xFF
            sign_index = (high >> (7 * lane)) & 0x7F
            signs = sign_index | ((sign_index.bit_count() & 1) << 7)
            values.extend(
                group_scale * magnitude * (-1 if signs & (1 << j) else 1)
                for j, magnitude in enumerate(IQ2_XXS_CODEBOOK[grid_index])
            )
    return torch.tensor(values, dtype=torch.float64)


def decode_iq1_s(block: bytes | bytearray | memoryview) -> torch.Tensor:
    """Decode one 50-byte IQ1_S block into 256 represented values.

    Layout is ``f16 d``, 32 low grid-index bytes, then eight little-endian
    uint16 high/control words.  In a 32-value group, bits 0..11 are four
    3-bit grid-index extensions, bits 12..14 select odd multiplier
    ``m = 2*s+1``, and bit 15 selects delta ``+1/8`` or ``-1/8``.  Therefore
    ``y[32*g+8*l+j] = d*m*(grid[index][j] + delta)``.
    """

    packed = _block_bytes(block, IQ1_S_BYTES, "IQ1_S")
    scale = _f16_le(packed)
    low_indices = packed[2:34]
    values: list[float] = []
    for group in range(8):
        control = int.from_bytes(packed[34 + 2 * group : 36 + 2 * group], "little")
        group_scale = scale * (2 * ((control >> 12) & 0x7) + 1)
        delta = -0.125 if control & 0x8000 else 0.125
        for lane in range(4):
            index = low_indices[4 * group + lane] | (((control >> (3 * lane)) & 0x7) << 8)
            values.extend(group_scale * (value + delta) for value in IQ1_S_CODEBOOK[index])
    return torch.tensor(values, dtype=torch.float64)


__all__ = [
    "decode_iq1_s",
    "decode_iq2_xxs",
    "decode_iq4_nl",
    "IQ1_S_BYTES",
    "IQ1_S_CODEBOOK",
    "IQ1_S_VALUES",
    "IQ2_XXS_BYTES",
    "IQ2_XXS_CODEBOOK",
    "IQ2_XXS_VALUES",
    "IQ4_NL_BYTES",
    "IQ4_NL_CODEBOOK",
    "IQ4_NL_VALUES",
]
