"""Protect byte-exact bounded acquisition from accidental whole-shard reads."""

from unittest.mock import patch

import pytest

from tools.parity.qwen4.native_source import read_range, selected_tensors


class Response:
    def __init__(self, status, content_range, body):
        self.status = status
        self.headers = {"Content-Range": content_range}
        self.body = body
        self.read_called = False

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass

    def read(self, size):
        self.read_called = True
        return self.body[:size]


@pytest.mark.parametrize("status,header", [(200, ""), (206, "bytes 0-3/100")])
def test_rejects_ignored_or_wrong_range_before_read(status, header):
    response = Response(status, header, b"full shard must not be read")
    with patch("urllib.request.urlopen", return_value=response):
        with pytest.raises(ValueError, match="exact Range"):
            read_range("https://example.invalid/shard", 8, 11)
    assert not response.read_called


def test_exact_range_and_length():
    response = Response(206, "bytes 8-11/100", bytes([0, 128, 255, 7]))
    with patch("urllib.request.urlopen", return_value=response):
        assert read_range("https://example.invalid/shard", 8, 11) == response.body
    response.body = b"12345"
    with patch("urllib.request.urlopen", return_value=response):
        with pytest.raises(ValueError, match="length mismatch"):
            read_range("https://example.invalid/shard", 8, 11)


def test_selection_excludes_mtp_and_other_main_layers():
    shard = "model-00001-of-00010.safetensors"
    names = {f"model.language_model.layers.0.tensor{i}": shard for i in range(6166)}
    names.update({"mtp.layers.0.weight": "other", "model.language_model.layers.1.weight": shard})
    selected = selected_tensors({"weight_map": names}, 0)
    assert set(selected.values()) == {shard}
    assert len(selected) == 6166
    assert all(name.startswith("model.language_model.layers.0.") for name in selected)


def test_layer_one_crosses_shards_without_duplicating_ple():
    shards = ("model-00001-of-00010.safetensors", "model-00002-of-00010.safetensors")
    names = {f"model.language_model.layers.1.tensor{i}": shards[i % 2] for i in range(6166)}
    names["model.language_model.layers.1.ple.key_proj.weight"] = shards[0]
    names["model.language_model.layers.1.ple.hash_offsets"] = shards[1]
    selected = selected_tensors({"weight_map": names}, 1)
    assert len(selected) == 6166
    assert set(selected.values()) == set(shards)
    assert not any(".ple." in name for name in selected)
