import struct

import pytest
import torch

from tools.artifact.layouts import (
    decode_fp8_tensor_scaled_words, encode_fp8_tensor_scaled, encoded_size,
    decode_fp8_calibrated_words, encode_fp8_calibrated,
)


@pytest.mark.parametrize("shape", [(16, 160), (3, 3)])
def test_tensor_fp8_exact_source_words(shape):
    count = shape[0] * shape[1]
    # Both signs and signed zeros; exclude the two NaN encodings.
    codes = ((torch.arange(count) % 254) // 127 * 128 + torch.arange(count) % 127).to(torch.uint8).reshape(shape)
    multiplier = torch.tensor([0x3951], dtype=torch.int16).view(torch.bfloat16).reshape(())
    payload = encode_fp8_tensor_scaled(codes, multiplier)
    assert payload == codes.numpy().tobytes() + struct.pack("<H", 0x3951)
    assert len(payload) == encoded_size("tensor-scale-v1", "FP8_E4M3FN_TENSOR_BF16S", shape)
    decoded_codes, decoded_scale = decode_fp8_tensor_scaled_words(payload, shape)
    assert torch.equal(decoded_codes, codes)
    assert decoded_scale.view(torch.int16) == 0x3951


def test_tensor_fp8_invalid_numeric_words():
    valid = torch.tensor([[0x00, 0x80, 0x7E, 0xFE]], dtype=torch.uint8)
    multiplier = torch.ones((), dtype=torch.bfloat16)
    for word in (0x7F, 0xFF):
        codes = valid.clone()
        codes[0, 0] = word
        with pytest.raises(ValueError):
            encode_fp8_tensor_scaled(codes, multiplier)
        with pytest.raises(ValueError):
            decode_fp8_tensor_scaled_words(bytes([word, 0, 0, 0, 0x80, 0x3F]), (1, 4))
    for value in (0.0, -0.0, -1.0, float("inf"), float("nan")):
        with pytest.raises(ValueError):
            encode_fp8_tensor_scaled(valid, torch.tensor(value, dtype=torch.bfloat16))


@pytest.mark.parametrize("shape", [(3, 3), (512, 2560)])
def test_calibrated_fp8_exact_source_words(shape):
    count = shape[0] * shape[1]
    codes = ((torch.arange(count) % 254) // 127 * 128 + torch.arange(count) % 127).to(torch.uint8).reshape(shape)
    words = torch.tensor([0x3AE2B719, 0x3D1A4925], dtype=torch.int32)
    weight, activation = words.view(torch.float32)
    payload = encode_fp8_calibrated(codes, weight, activation)
    assert payload == codes.numpy().tobytes() + bytes((-count) % 4) + struct.pack("<II", *words.tolist())
    assert len(payload) == encoded_size("tensor-calibrated-v1", "FP8_E4M3FN_TENSOR_F32M", shape)
    decoded_codes, decoded_weight, decoded_activation = decode_fp8_calibrated_words(payload, shape)
    assert torch.equal(decoded_codes, codes)
    assert torch.stack((decoded_weight, decoded_activation)).view(torch.int32).tolist() == words.tolist()


def test_calibrated_fp8_rejects_invalid_source_words_and_padding():
    codes = torch.tensor([[0, 128, 126]], dtype=torch.uint8)
    one = torch.ones((), dtype=torch.float32)
    for bad in (0.0, -1.0, float("inf"), float("nan")):
        with pytest.raises(ValueError):
            encode_fp8_calibrated(codes, one, torch.tensor(bad))
    payload = bytearray(encode_fp8_calibrated(codes, one, one))
    payload[3] = 1
    with pytest.raises(ValueError):
        decode_fp8_calibrated_words(payload, (1, 3))
    payload[3] = 0
    payload[0] = 127
    with pytest.raises(ValueError):
        decode_fp8_calibrated_words(payload, (1, 3))
