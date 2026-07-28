from __future__ import annotations

from collections import Counter
import struct

import pytest
import torch

from tools.convert.qwen3_6_27b import convert_nvfp4
from tools.convert.qwen3_6_27b import inventory_nvfp4 as inventory
from tools.convert.qwen3_6_27b import recipe_nvfp4 as recipe


FULL_ATTENTION = tuple(range(3, 64, 4))
EARLY_INPUT = (3, 7, 11, 15, 19, 23)
NVFP4_INPUT = (27, 31, 35, 39, 43, 47, 51, 55, 59, 63)
BF16_OUTPUT = (3, 7)
NVFP4_OUTPUT = (11, 15, 19, 23, 27, 31, 35, 39, 43, 47, 51, 55, 59, 63)
GDN = tuple(layer for layer in range(64) if layer not in FULL_ATTENTION)


def test_closed_allocation_inventory_and_site_coverage():
    assert inventory.FULL_ATTENTION_LAYERS == FULL_ATTENTION
    assert inventory.EARLY_ATTENTION_INPUT_LAYERS == EARLY_INPUT
    assert inventory.NVFP4_ATTENTION_INPUT_LAYERS == NVFP4_INPUT
    assert inventory.BF16_ATTENTION_OUTPUT_LAYERS == BF16_OUTPUT
    assert inventory.NVFP4_ATTENTION_OUTPUT_LAYERS == NVFP4_OUTPUT
    assert inventory.GDN_LAYERS == GDN
    assert inventory.BF16_GDN_OUTPUT_LAYERS == (4,)

    assert len(inventory.OBJECT_SPECS) == 1371
    assert len(inventory.NVFP4_TENSOR_SPECS) == 305
    assert len(inventory.INPUT_SCALE_DIVISOR_SPECS) == 247
    assert inventory.FORMAT_COUNTS == {
        "BF16": 597,
        "FP32": 343,
        "I32": 1,
        "Q4G64_F16S": 55,
        "Q5G64_F16S": 54,
        "Q6G64_F16S": 3,
        "W8G32_F16S": 7,
        "NVFP4": 305,
    }

    bound = [
        weight
        for site in recipe.INPUT_DIVISOR_RECIPES
        for weight in site.weight_names
    ]
    assert len(bound) == 305
    assert Counter(bound) == Counter(recipe.NVFP4_WEIGHTS_BY_NAME.keys())


def test_input_divisor_materialization_preserves_positive_fp32_word():
    class Reader:
        def __init__(self, word: int):
            self.tensor = torch.frombuffer(
                bytearray(struct.pack("<I", word)), dtype=torch.float32
            )

        def get(self, _name: str) -> torch.Tensor:
            return self.tensor

    selected = recipe.INPUT_DIVISOR_RECIPES[0]
    positive_word = struct.unpack("<I", struct.pack("<f", 1.25))[0]
    actual = recipe.materialize_input_divisor(selected, Reader(positive_word))
    assert actual.shape == torch.Size([])
    assert int(actual.view(torch.int32)) & 0xFFFFFFFF == positive_word

    with pytest.raises(ValueError, match="finite and positive"):
        recipe.materialize_input_divisor(selected, Reader(0))


def test_converter_rejects_wrong_basename_before_source_or_file_creation(tmp_path):
    output = tmp_path / "qwen3_6_27b.ninfer"
    with pytest.raises(ValueError, match="output basename"):
        convert_nvfp4.convert(
            tmp_path / "missing-base",
            tmp_path / "missing-nvfp4",
            output,
            device="cpu",
        )
    assert not output.exists()
