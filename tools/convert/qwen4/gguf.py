"""Small, dependency-free GGUF v3 reader used only by the verifier converter.

This module deliberately stops at metadata and tensor byte spans.  It does not
decode tensor values and is never linked into the C++ runtime.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import struct
from typing import BinaryIO


MAGIC = b"GGUF"
VERSION = 3

_VALUE_WIDTHS = {
    0: 1,   # uint8
    1: 1,   # int8
    2: 2,   # uint16
    3: 2,   # int16
    4: 4,   # uint32
    5: 4,   # int32
    6: 4,   # float32
    7: 1,   # bool
    10: 8,  # uint64
    11: 8,  # int64
    12: 8,  # float64
}
_STRING = 8
_ARRAY = 9

# ggml type id -> (name, logical values per block, encoded bytes per block)
GGML_TYPES = {
    0: ("FP32", 1, 4),
    8: ("Q8_0", 32, 34),
    12: ("Q4_K", 256, 144),
    13: ("Q5_K", 256, 176),
    14: ("Q6_K", 256, 210),
    16: ("IQ2_XXS", 256, 66),
    19: ("IQ1_S", 256, 50),
    20: ("IQ4_NL", 32, 18),
    30: ("BF16", 1, 2),
}


class GGUFError(ValueError):
    """The source does not satisfy the converter's closed GGUF contract."""


@dataclass(frozen=True, slots=True)
class TensorInfo:
    name: str
    # GGUF writes the contiguous K dimension first.  Shape is exposed in
    # mathematical order, so a matrix is [N,K] and an expert bank [E,N,K].
    shape: tuple[int, ...]
    format: str
    offset: int
    bytes: int


@dataclass(frozen=True, slots=True)
class GGUFFile:
    path: Path
    metadata: dict[str, object]
    tensors: tuple[TensorInfo, ...]
    data_offset: int
    file_bytes: int

    def absolute_span(self, tensor: TensorInfo) -> tuple[int, int]:
        begin = self.data_offset + tensor.offset
        return begin, begin + tensor.bytes


def _read_exact(file: BinaryIO, size: int, field: str) -> bytes:
    data = file.read(size)
    if len(data) != size:
        raise GGUFError(f"truncated GGUF {field}")
    return data


def _u32(file: BinaryIO, field: str) -> int:
    return struct.unpack("<I", _read_exact(file, 4, field))[0]


def _u64(file: BinaryIO, field: str) -> int:
    return struct.unpack("<Q", _read_exact(file, 8, field))[0]


def _string(file: BinaryIO, field: str) -> str:
    size = _u64(file, field + " length")
    if size > 1 << 30:
        raise GGUFError(f"unreasonable GGUF {field} length: {size}")
    try:
        return _read_exact(file, size, field).decode("utf-8")
    except UnicodeDecodeError as exc:
        raise GGUFError(f"GGUF {field} is not UTF-8") from exc


def _scalar(file: BinaryIO, value_type: int, field: str) -> object:
    if value_type == _STRING:
        return _string(file, field)
    width = _VALUE_WIDTHS.get(value_type)
    if width is None:
        raise GGUFError(f"unsupported GGUF metadata type {value_type} in {field}")
    raw = _read_exact(file, width, field)
    formats = {
        0: "<B",
        1: "<b",
        2: "<H",
        3: "<h",
        4: "<I",
        5: "<i",
        6: "<f",
        7: "<?",
        10: "<Q",
        11: "<q",
        12: "<d",
    }
    return struct.unpack(formats[value_type], raw)[0]


def _value(file: BinaryIO, value_type: int, field: str) -> object:
    if value_type != _ARRAY:
        return _scalar(file, value_type, field)
    element_type = _u32(file, field + " element type")
    if element_type == _ARRAY:
        raise GGUFError(f"nested GGUF arrays are not admitted in {field}")
    length = _u64(file, field + " length")
    if length > 1 << 31:
        raise GGUFError(f"unreasonable GGUF array length in {field}: {length}")
    return tuple(_scalar(file, element_type, f"{field}[{index}]") for index in range(length))


def _align_up(value: int, alignment: int) -> int:
    return (value + alignment - 1) // alignment * alignment


def read_gguf(path: str | Path) -> GGUFFile:
    source = Path(path)
    file_bytes = source.stat().st_size
    with source.open("rb") as file:
        if _read_exact(file, 4, "magic") != MAGIC:
            raise GGUFError("GGUF magic mismatch")
        version = _u32(file, "version")
        if version != VERSION:
            raise GGUFError(f"GGUF version must be {VERSION}, got {version}")
        tensor_count = _u64(file, "tensor count")
        metadata_count = _u64(file, "metadata count")
        if tensor_count > 1 << 20 or metadata_count > 1 << 20:
            raise GGUFError("unreasonable GGUF directory counts")

        metadata: dict[str, object] = {}
        for _ in range(metadata_count):
            key = _string(file, "metadata key")
            if key in metadata:
                raise GGUFError(f"duplicate GGUF metadata key: {key}")
            metadata[key] = _value(file, _u32(file, f"metadata type for {key}"), key)

        pending: list[tuple[str, tuple[int, ...], int, int]] = []
        names: set[str] = set()
        for _ in range(tensor_count):
            name = _string(file, "tensor name")
            if not name or name in names:
                raise GGUFError(f"empty or duplicate GGUF tensor name: {name!r}")
            names.add(name)
            rank = _u32(file, f"{name} rank")
            if not 1 <= rank <= 4:
                raise GGUFError(f"{name}: unsupported rank {rank}")
            physical_shape = tuple(_u64(file, f"{name} dimension") for _ in range(rank))
            if any(dim == 0 for dim in physical_shape):
                raise GGUFError(f"{name}: zero shape dimension")
            ggml_type = _u32(file, f"{name} type")
            relative_offset = _u64(file, f"{name} offset")
            pending.append((name, tuple(reversed(physical_shape)), ggml_type, relative_offset))

        alignment = metadata.get("general.alignment", 32)
        if type(alignment) is not int or alignment <= 0 or alignment & (alignment - 1):
            raise GGUFError("general.alignment must be a positive power of two")
        data_offset = _align_up(file.tell(), alignment)

    tensors: list[TensorInfo] = []
    for name, shape, ggml_type, relative_offset in pending:
        try:
            format_name, block_values, block_bytes = GGML_TYPES[ggml_type]
        except KeyError:
            raise GGUFError(f"{name}: unsupported ggml type {ggml_type}") from None
        values = 1
        for dim in shape:
            values *= dim
        if values % block_values:
            raise GGUFError(
                f"{name}: {values} values are not divisible by {format_name} block {block_values}"
            )
        payload_bytes = values // block_values * block_bytes
        begin = data_offset + relative_offset
        end = begin + payload_bytes
        if relative_offset % alignment or begin < data_offset or end > file_bytes:
            raise GGUFError(f"{name}: tensor span is misaligned or outside the file")
        tensors.append(TensorInfo(name, shape, format_name, relative_offset, payload_bytes))

    by_offset = sorted(tensors, key=lambda item: item.offset)
    for previous, current in zip(by_offset, by_offset[1:]):
        if previous.offset + previous.bytes > current.offset:
            raise GGUFError(f"{current.name}: tensor payload overlaps {previous.name}")
    return GGUFFile(source, metadata, tuple(tensors), data_offset, file_bytes)


__all__ = ["GGML_TYPES", "GGUFError", "GGUFFile", "TensorInfo", "read_gguf"]
