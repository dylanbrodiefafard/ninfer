"""Protect the full-table converter's partition/row/chunk byte ordering."""
import json
import struct

import numpy as np

from tools.parity.qwen4 import native_ple_full_fixture as fixture


def test_streamed_source_words_across_chunk_and_partition_boundaries(tmp_path, monkeypatch):
    rows = 16385  # includes a partial final conversion chunk
    monkeypatch.setattr(fixture, "SHARD_ROWS", rows)
    monkeypatch.setattr(fixture, "PARTITIONS", 2)
    expected = bytearray()
    multipliers = (0x37B30C31, 0x379F3CF3)
    for partition in range(2):
        codes = ((np.arange(rows * 80, dtype=np.uint32) + partition * 11) % 256).astype(np.uint8)
        scales = ((np.arange(rows * 10, dtype=np.uint32) + partition * 17) % 127).astype(np.uint8)
        # Deliberately put the scalar first and scales before codes in safetensors.
        header = {
            "weight_scale_2": dict(dtype="F32", shape=[], data_offsets=[0, 4]),
            "weight_scale": dict(dtype="F8_E4M3", shape=[rows, 10], data_offsets=[4, 4 + rows * 10]),
            "weight_e2m1": dict(dtype="U8", shape=[rows, 80],
                                data_offsets=[4 + rows * 10, 4 + rows * 90]),
        }
        metadata = json.dumps(header).encode()
        (tmp_path / f"shard_{partition}.safetensors").write_bytes(
            struct.pack("<Q", len(metadata)) + metadata + struct.pack("<I", multipliers[partition])
            + scales.tobytes() + codes.tobytes())
        for row in range(rows):
            expected.extend(codes[row * 80:(row + 1) * 80])
            expected.extend(scales[row * 10:(row + 1) * 10])
    expected.extend(struct.pack("<2I", *multipliers))
    assert b"".join(fixture.chunks(tmp_path)) == expected
