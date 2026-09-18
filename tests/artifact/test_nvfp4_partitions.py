"""Independent byte-offset evidence for partition-scaled NVFP4 embeddings."""
import struct

import pytest
import torch

from tools.artifact.layouts import encoded_size, pack_nvfp4_partitions, unpack_nvfp4_partitions


@pytest.mark.parametrize("shape", [(3, 4, 160), (3, 1, 16)])
def test_exact_partition_words(shape):
    p, r, k = shape
    codes = (torch.arange(p * r * k // 2) % 256).to(torch.uint8).reshape(p, r, k // 2)
    scales = (torch.arange(p * r * k // 16) % 127).to(torch.uint8).reshape(p, r, k // 16)
    multipliers = torch.tensor([0x37b30c31, 0x379f3cf3, 0x3f808000], dtype=torch.int32).view(torch.float32)
    payload = pack_nvfp4_partitions(codes, scales, multipliers)
    expected = b"".join(codes[i, j].numpy().tobytes() + scales[i, j].numpy().tobytes()
                        for i in range(p) for j in range(r))
    expected += b"".join(struct.pack("<f", v) for v in multipliers)
    assert payload == expected
    assert len(payload) == encoded_size("partitioned-row-blockscale-k16-v1", "NVFP4_PARTITION_F32M", shape)
    for actual, original in zip(unpack_nvfp4_partitions(payload, shape), (codes, scales, multipliers)):
        assert torch.equal(actual.contiguous().view(torch.uint8), original.contiguous().view(torch.uint8))


def test_invalid_partition_payload():
    codes = torch.zeros((1, 1, 8), dtype=torch.uint8)
    scales = torch.zeros((1, 1, 1), dtype=torch.uint8)
    for invalid in (0x7f, 0x80, 0xff):
        with pytest.raises(ValueError):
            pack_nvfp4_partitions(codes, torch.full_like(scales, invalid), torch.ones(1))
    for invalid in (0., -1., float("inf"), float("nan")):
        with pytest.raises(ValueError):
            pack_nvfp4_partitions(codes, scales, torch.tensor([invalid]))
        with pytest.raises(ValueError):
            unpack_nvfp4_partitions(bytes(9) + struct.pack("<f", invalid), (1, 1, 16))
    with pytest.raises(ValueError):
        unpack_nvfp4_partitions(bytes(12), (1, 1, 16))
