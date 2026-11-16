"""Reader for the expert-format study's packed routed-expert store (`mixed_mse_gptq`).

The store is the authority for the routed-expert codes, block scales, matrix scales, and F3
books; the converter repacks its words without changing any value. Its byte format, defined by
`fnstudy/formats/store.py` and `codecs.py` of the study:

- a 64-byte little-endian header `<8s14I`: magic `FSTUDY01`, version 1, two zero fields, the
  geometry `(48, 512, 2560, 640)`, the header+books+bitmap byte count, the F3 and NVFP4 record
  byte counts, then `(3, 32, 4, 16)` (low bits, low group, high bits, high group);
- 48 x 3 F3 books (layer-major, then gate/up/down), each four bytes of eight E2M1 nibbles with
  entry `2j` in the low nibble of byte `j`;
- a 3,072-byte bitmap over `layer * 512 + expert`, bit `i % 8` of byte `i / 8`, set for the
  NVFP4 experts;
- expert-major records at `header + index * f3_bytes + nvfp4_before(index) * (nvfp4_bytes -
  f3_bytes)`. A record holds gate `[640,2560]`, up `[640,2560]`, down `[2560,640]`, each as an
  FP32 matrix multiplier, row-major E4M3FN block scales (one per 32 F3 / 16 NVFP4 columns), then
  row-major codes: F3 packs eight 3-bit book indices in three bytes (index `i` at bit `3i` of
  the little-endian 24-bit word), NVFP4 two E2M1 nibbles per byte (low nibble first).

A weight decodes to `E2M1(book[code] or code) * E4M3FN(scale) * multiplier`.
"""

from __future__ import annotations

import enum
import json
import os
import struct
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from tools.artifact.layouts import ExpertMatrixWords, ExpertRecordWords

LAYERS = 48
EXPERTS = 512
HIDDEN = 2560
INTERMEDIATE = 640
MATRICES = ("gate", "up", "down")
MATRIX_SHAPES = ((INTERMEDIATE, HIDDEN), (INTERMEDIATE, HIDDEN), (HIDDEN, INTERMEDIATE))

_HEADER = struct.Struct("<8s14I")
_MAGIC = b"FSTUDY01"
_BOOK_BYTES = 4
_BOOKS_OFFSET = _HEADER.size
_BITMAP_OFFSET = _BOOKS_OFFSET + LAYERS * len(MATRICES) * _BOOK_BYTES
_BITMAP_BYTES = LAYERS * EXPERTS // 8
_RECORDS_OFFSET = _BITMAP_OFFSET + _BITMAP_BYTES
STORE_FORMAT = "mixed_mse_gptq"


class ExpertCodec(enum.IntEnum):
    """Codec of one routed expert; the value is `ops::MoeExpertCodec`, stored in the codec map."""

    NVFP4 = 0
    F3 = 1

    @property
    def code_bits(self) -> int:
        return 3 if self is ExpertCodec.F3 else 4

    @property
    def group_size(self) -> int:
        return 32 if self is ExpertCodec.F3 else 16

    @property
    def record_bytes(self) -> int:
        return sum(
            4 + rows * columns // self.group_size + rows * columns * self.code_bits // 8
            for rows, columns in MATRIX_SHAPES
        )


@dataclass(frozen=True, slots=True)
class StudyExpert:
    codec: ExpertCodec
    words: ExpertRecordWords
    # Exact FP32 matrix multiplier words of gate, up, down.
    matrix_scale_words: tuple[int, int, int]


def _decode_f3_codes(packed: bytes, rows: int, columns: int) -> torch.Tensor:
    triples = np.frombuffer(packed, dtype=np.uint8).reshape(-1, 3).astype(np.uint32)
    word = triples[:, 0] | (triples[:, 1] << 8) | (triples[:, 2] << 16)
    shifts = 3 * np.arange(8, dtype=np.uint32)
    codes = ((word[:, None] >> shifts) & 7).astype(np.uint8)
    return torch.from_numpy(codes.reshape(rows, columns))


def _decode_nibble_codes(packed: bytes, rows: int, columns: int) -> torch.Tensor:
    pairs = np.frombuffer(packed, dtype=np.uint8)
    codes = np.empty(pairs.size * 2, dtype=np.uint8)
    codes[0::2] = pairs & 0x0F
    codes[1::2] = pairs >> 4
    return torch.from_numpy(codes.reshape(rows, columns))


def decode_study_record(record: bytes, codec: ExpertCodec) -> StudyExpert:
    """Decode one study record into natural-order stored words; exact, no arithmetic on values."""

    if len(record) != codec.record_bytes:
        raise ValueError(f"{codec.name} study record has {len(record)} bytes")
    cursor = 0
    matrices: list[ExpertMatrixWords] = []
    scale_words: list[int] = []
    for rows, columns in MATRIX_SHAPES:
        scale_words.append(struct.unpack_from("<I", record, cursor)[0])
        cursor += 4
        scale_bytes = rows * columns // codec.group_size
        scales = np.frombuffer(record, dtype=np.uint8, count=scale_bytes, offset=cursor)
        cursor += scale_bytes
        code_bytes = rows * columns * codec.code_bits // 8
        packed = record[cursor : cursor + code_bytes]
        cursor += code_bytes
        decode = _decode_f3_codes if codec is ExpertCodec.F3 else _decode_nibble_codes
        matrices.append(
            ExpertMatrixWords(
                codes=decode(packed, rows, columns),
                scales=torch.from_numpy(scales.copy()).reshape(rows, columns // codec.group_size),
            )
        )
    return StudyExpert(codec, ExpertRecordWords(*matrices), tuple(scale_words))


class StudyExpertStore:
    """Validated, read-only view of the study's mixed F3/NVFP4 expert file."""

    def __init__(self, experts_path: str | Path, manifest_path: str | Path) -> None:
        self.path = Path(experts_path)
        self.manifest_path = Path(manifest_path)
        manifest = json.loads(self.manifest_path.read_text())
        if manifest.get("format") != STORE_FORMAT:
            raise ValueError(f"{self.manifest_path}: format is not {STORE_FORMAT}")
        if manifest.get("completed_layers") != list(range(LAYERS)):
            raise ValueError(f"{self.manifest_path}: the store is incomplete")
        self.manifest = manifest
        self._fd = os.open(self.path, os.O_RDONLY)
        try:
            self._validate_header(manifest)
        except BaseException:
            os.close(self._fd)
            raise

    def _read(self, offset: int, length: int) -> bytes:
        data = os.pread(self._fd, length, offset)
        if len(data) != length:
            raise ValueError(f"{self.path}: short read at {offset}")
        return data

    def _validate_header(self, manifest: dict) -> None:
        fields = _HEADER.unpack(self._read(0, _HEADER.size))
        expected = (
            _MAGIC,
            1,
            0,
            0,
            LAYERS,
            EXPERTS,
            HIDDEN,
            INTERMEDIATE,
            _RECORDS_OFFSET,
            ExpertCodec.F3.record_bytes,
            ExpertCodec.NVFP4.record_bytes,
            3,
            32,
            4,
            16,
        )
        if fields != expected:
            raise ValueError(f"{self.path}: unexpected header {fields}")
        books = self._read(_BOOKS_OFFSET, _BITMAP_OFFSET - _BOOKS_OFFSET)
        self.book_words = np.frombuffer(books, dtype="<u4").reshape(LAYERS, len(MATRICES)).copy()
        bitmap = np.frombuffer(self._read(_BITMAP_OFFSET, _BITMAP_BYTES), dtype=np.uint8)
        nvfp4 = np.unpackbits(bitmap, bitorder="little").astype(bool)
        self.codecs = np.where(nvfp4, ExpertCodec.NVFP4, ExpertCodec.F3).reshape(LAYERS, EXPERTS)
        self._nvfp4_before = np.concatenate(([0], np.cumsum(nvfp4, dtype=np.int64)))
        self.nvfp4_experts = int(nvfp4.sum())
        self.f3_experts = LAYERS * EXPERTS - self.nvfp4_experts
        storage = manifest["storage"]
        if storage.get("high_precision_experts") != self.nvfp4_experts:
            raise ValueError(f"{self.manifest_path}: NVFP4 expert count disagrees with the bitmap")
        size = (
            _RECORDS_OFFSET
            + self.f3_experts * ExpertCodec.F3.record_bytes
            + self.nvfp4_experts * ExpertCodec.NVFP4.record_bytes
        )
        actual = os.fstat(self._fd).st_size
        if actual != size or storage.get("total_bytes") != size:
            raise ValueError(f"{self.path}: {actual} bytes, the header implies {size}")

    def codec(self, layer: int, expert: int) -> ExpertCodec:
        return ExpertCodec(int(self.codecs[layer, expert]))

    def record_span(self, layer: int, expert: int) -> tuple[int, int]:
        index = layer * EXPERTS + expert
        offset = (
            _RECORDS_OFFSET
            + index * ExpertCodec.F3.record_bytes
            + int(self._nvfp4_before[index])
            * (ExpertCodec.NVFP4.record_bytes - ExpertCodec.F3.record_bytes)
        )
        return offset, self.codec(layer, expert).record_bytes

    def read(self, layer: int, expert: int) -> StudyExpert:
        offset, length = self.record_span(layer, expert)
        record = self._read(offset, length)
        # The store is read once per pass; keep it from displacing the shared host's page cache.
        os.posix_fadvise(self._fd, offset, length, os.POSIX_FADV_DONTNEED)
        return decode_study_record(record, self.codec(layer, expert))

    def close(self) -> None:
        if self._fd >= 0:
            os.close(self._fd)
            self._fd = -1

    def __enter__(self) -> StudyExpertStore:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()
