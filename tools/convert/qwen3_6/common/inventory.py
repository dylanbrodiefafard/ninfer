"""Leaf storage vocabulary shared by Qwen3.6-family artifact converters.

This module owns only the format/layout vocabulary, the passive tensor record,
and the checkpoint-invariant Vision layer range. Each target owns its complete
ordered object inventory.
"""

from __future__ import annotations

from dataclasses import dataclass

CONTIGUOUS_LAYOUT = "contiguous-le-v1"
ROW_SPLIT_LAYOUT = "row-split-k128-v1"

BF16 = "BF16"
FP32 = "FP32"
I32 = "I32"
Q4 = "Q4G64_F16S"
W8 = "W8G32_F16S"

DIRECT_FORMATS = frozenset((BF16, FP32, I32))

VISION_LAYERS = tuple(range(27))


@dataclass(frozen=True, slots=True)
class TensorSpec:
    name: str
    shape: tuple[int, ...]
    format: str
    layout: str


def tensor_spec(
    name: str,
    shape: tuple[int, ...],
    numeric_format: str,
) -> TensorSpec:
    """Build a tensor spec with the canonical layout for its numeric format."""

    layout = CONTIGUOUS_LAYOUT if numeric_format in DIRECT_FORMATS else ROW_SPLIT_LAYOUT
    return TensorSpec(name=name, shape=shape, format=numeric_format, layout=layout)


__all__ = [
    "BF16",
    "CONTIGUOUS_LAYOUT",
    "DIRECT_FORMATS",
    "FP32",
    "I32",
    "Q4",
    "ROW_SPLIT_LAYOUT",
    "VISION_LAYERS",
    "W8",
    "TensorSpec",
    "tensor_spec",
]
