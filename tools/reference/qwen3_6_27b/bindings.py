"""Typed Vision binding of a generic NInfer artifact for the Qwen3.8-27B target.

The binding validates the frontend resources and the complete 333-tensor Vision
inventory of a `qwen3.8-27b/nvfp4` artifact, resolves persistent names once, and
exposes only typed block objects to the Vision schedule. Text, MTP, draft-head,
and DFlash2 objects are outside this reference and are not inspected.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TypeAlias

from tools.artifact import (
    Artifact,
    ArtifactIdentity,
    ResourceObject,
    TensorObject,
)

MODEL_ID = "qwen3.8-27b"
WEIGHTS_ID = "nvfp4"

CONTIGUOUS = "contiguous-le-v1"
ROW_SPLIT = "row-split-k128-v1"
RESOURCE_ENCODING = "raw-bytes-v1"

BF16 = "BF16"
Q4 = "Q4G64_F16S"
Q5 = "Q5G64_F16S"
Q6 = "Q6G64_F16S"
W8 = "W8G32_F16S"

VISION_LAYERS = tuple(range(27))
VISION_TENSOR_COUNT = 333


class BindingError(ValueError):
    """The generic artifact does not implement the registered Vision contract."""


@dataclass(frozen=True, slots=True)
class PhysicalBlock:
    """One stored Vision tensor after its target role has been resolved.

    ``tensor_id`` is the block's index in Vision inventory order, not its index
    in the complete artifact directory.
    """

    tensor_id: int
    descriptor: TensorObject

    @property
    def shape(self) -> tuple[int, ...]:
        return self.descriptor.shape

    @property
    def format(self) -> str:
        return self.descriptor.format

    @property
    def layout(self) -> str:
        return self.descriptor.layout


@dataclass(frozen=True, slots=True)
class BoundResource:
    descriptor: ResourceObject


@dataclass(frozen=True, slots=True)
class FrontendResources:
    tokenizer_json: BoundResource
    tokenizer_config_json: BoundResource
    chat_template_jinja: BoundResource
    generation_config_json: BoundResource
    preprocessor_config_json: BoundResource
    video_preprocessor_config_json: BoundResource


@dataclass(frozen=True, slots=True)
class VisionLayerBinding:
    index: int
    attention_qkv: PhysicalBlock
    attention_qkv_bias: PhysicalBlock
    attention_output: PhysicalBlock
    attention_output_bias: PhysicalBlock
    mlp_fc1: PhysicalBlock
    mlp_fc1_bias: PhysicalBlock
    mlp_fc2: PhysicalBlock
    mlp_fc2_bias: PhysicalBlock
    norm1_weight: PhysicalBlock
    norm1_bias: PhysicalBlock
    norm2_weight: PhysicalBlock
    norm2_bias: PhysicalBlock


@dataclass(frozen=True, slots=True)
class VisionMergerBinding:
    fc1: PhysicalBlock
    fc1_bias: PhysicalBlock
    fc2: PhysicalBlock
    fc2_bias: PhysicalBlock
    norm_weight: PhysicalBlock
    norm_bias: PhysicalBlock


@dataclass(frozen=True, slots=True)
class VisionBinding:
    patch_embedding: PhysicalBlock
    patch_embedding_bias: PhysicalBlock
    position_embedding: PhysicalBlock
    layers: tuple[VisionLayerBinding, ...]
    merger: VisionMergerBinding


@dataclass(frozen=True, slots=True)
class _ExpectedTensor:
    name: str
    shape: tuple[int, ...]
    format: str
    layout: str


@dataclass(frozen=True, slots=True)
class _ExpectedResource:
    name: str
    encoding: str = RESOURCE_ENCODING


_ExpectedObject: TypeAlias = _ExpectedTensor | _ExpectedResource


def _tensor(name: str, shape: tuple[int, ...], format_name: str) -> _ExpectedTensor:
    layout = CONTIGUOUS if format_name == BF16 else ROW_SPLIT
    return _ExpectedTensor(name, shape, format_name, layout)


def _vision_contract() -> tuple[_ExpectedTensor, ...]:
    tensors: list[_ExpectedTensor] = [
        _tensor("vision/patch_embedding", (1152, 1536), Q6),
        _tensor("vision/patch_embedding_bias", (1152,), BF16),
        _tensor("vision/position_embedding", (2304, 1152), BF16),
    ]
    for layer in VISION_LAYERS:
        prefix = f"vision/layers/{layer}/"
        tensors.extend(
            (
                _tensor(prefix + "attention/qkv", (3456, 1152), Q4),
                _tensor(prefix + "attention/qkv_bias", (3456,), BF16),
                _tensor(prefix + "attention/output", (1152, 1152), Q5),
                _tensor(prefix + "attention/output_bias", (1152,), BF16),
                _tensor(prefix + "mlp/fc1", (4304, 1152), Q4),
                _tensor(prefix + "mlp/fc1_bias", (4304,), BF16),
                _tensor(prefix + "mlp/fc2", (1152, 4304), Q5),
                _tensor(prefix + "mlp/fc2_bias", (1152,), BF16),
                _tensor(prefix + "norm1/weight", (1152,), BF16),
                _tensor(prefix + "norm1/bias", (1152,), BF16),
                _tensor(prefix + "norm2/weight", (1152,), BF16),
                _tensor(prefix + "norm2/bias", (1152,), BF16),
            )
        )
    tensors.extend(
        (
            _tensor("vision/merger/fc1", (4608, 4608), W8),
            _tensor("vision/merger/fc1_bias", (4608,), BF16),
            _tensor("vision/merger/fc2", (5120, 4608), W8),
            _tensor("vision/merger/fc2_bias", (5120,), BF16),
            _tensor("vision/merger/norm/weight", (1152,), BF16),
            _tensor("vision/merger/norm/bias", (1152,), BF16),
        )
    )
    return tuple(tensors)


_RESOURCE_CONTRACT = tuple(
    _ExpectedResource(name)
    for name in (
        "frontend/tokenizer.json",
        "frontend/tokenizer_config.json",
        "frontend/chat_template.jinja",
        "frontend/generation_config.json",
        "frontend/preprocessor_config.json",
        "frontend/video_preprocessor_config.json",
    )
)
_VISION_CONTRACT = _vision_contract()
if len(_VISION_CONTRACT) != VISION_TENSOR_COUNT:
    raise RuntimeError(f"Vision reference contract must contain {VISION_TENSOR_COUNT} tensors")


def _validate_contract(artifact: Artifact) -> None:
    expected_identity = ArtifactIdentity(MODEL_ID, WEIGHTS_ID)
    if artifact.identity != expected_identity:
        raise BindingError(
            f"artifact identity is {artifact.identity!r}; expected {expected_identity!r}"
        )
    actual_by_name = {obj.name: obj for obj in artifact.objects}
    expected_objects: tuple[_ExpectedObject, ...] = _RESOURCE_CONTRACT + _VISION_CONTRACT
    for expected in expected_objects:
        actual = actual_by_name.get(expected.name)
        if isinstance(expected, _ExpectedTensor):
            signature = (expected.name, expected.shape, expected.format, expected.layout)
            if (
                not isinstance(actual, TensorObject)
                or (actual.name, actual.shape, actual.format, actual.layout) != signature
            ):
                raise BindingError(
                    f"object {expected.name!r} does not match Vision tensor signature {signature!r}"
                )
        else:
            signature = (expected.name, expected.encoding)
            if (
                not isinstance(actual, ResourceObject)
                or (actual.name, actual.encoding) != signature
            ):
                raise BindingError(
                    f"object {expected.name!r} does not match frontend resource "
                    f"signature {signature!r}"
                )


class VisionArtifactBinding:
    """Vision and frontend binding over one open `qwen3.8-27b/nvfp4` artifact."""

    def __init__(self, artifact: Artifact, *, owns_artifact: bool = False):
        _validate_contract(artifact)
        self._artifact = artifact
        self._owns_artifact = owns_artifact

        actual_by_name = {obj.name: obj for obj in artifact.objects}
        resources = {
            expected.name: BoundResource(actual_by_name[expected.name])
            for expected in _RESOURCE_CONTRACT
        }
        blocks = {
            expected.name: PhysicalBlock(index, actual_by_name[expected.name])
            for index, expected in enumerate(_VISION_CONTRACT)
        }
        self.tensors = tuple(blocks.values())
        self.frontend = FrontendResources(
            resources["frontend/tokenizer.json"],
            resources["frontend/tokenizer_config.json"],
            resources["frontend/chat_template.jinja"],
            resources["frontend/generation_config.json"],
            resources["frontend/preprocessor_config.json"],
            resources["frontend/video_preprocessor_config.json"],
        )
        vision_layers: list[VisionLayerBinding] = []
        for layer in VISION_LAYERS:
            prefix = f"vision/layers/{layer}/"
            vision_layers.append(
                VisionLayerBinding(
                    index=layer,
                    attention_qkv=blocks[prefix + "attention/qkv"],
                    attention_qkv_bias=blocks[prefix + "attention/qkv_bias"],
                    attention_output=blocks[prefix + "attention/output"],
                    attention_output_bias=blocks[prefix + "attention/output_bias"],
                    mlp_fc1=blocks[prefix + "mlp/fc1"],
                    mlp_fc1_bias=blocks[prefix + "mlp/fc1_bias"],
                    mlp_fc2=blocks[prefix + "mlp/fc2"],
                    mlp_fc2_bias=blocks[prefix + "mlp/fc2_bias"],
                    norm1_weight=blocks[prefix + "norm1/weight"],
                    norm1_bias=blocks[prefix + "norm1/bias"],
                    norm2_weight=blocks[prefix + "norm2/weight"],
                    norm2_bias=blocks[prefix + "norm2/bias"],
                )
            )
        self.vision = VisionBinding(
            patch_embedding=blocks["vision/patch_embedding"],
            patch_embedding_bias=blocks["vision/patch_embedding_bias"],
            position_embedding=blocks["vision/position_embedding"],
            layers=tuple(vision_layers),
            merger=VisionMergerBinding(
                fc1=blocks["vision/merger/fc1"],
                fc1_bias=blocks["vision/merger/fc1_bias"],
                fc2=blocks["vision/merger/fc2"],
                fc2_bias=blocks["vision/merger/fc2_bias"],
                norm_weight=blocks["vision/merger/norm/weight"],
                norm_bias=blocks["vision/merger/norm/bias"],
            ),
        )

    @classmethod
    def open(cls, path: str | Path) -> VisionArtifactBinding:
        artifact = Artifact.open(path)
        try:
            return cls(artifact, owns_artifact=True)
        except BaseException:
            artifact.close()
            raise

    @property
    def identity(self) -> ArtifactIdentity:
        return self._artifact.identity

    def payload(self, block: PhysicalBlock) -> memoryview:
        return self._artifact.payload(block.descriptor)

    def resource_bytes(self, resource: BoundResource) -> bytes:
        return bytes(self._artifact.payload(resource.descriptor))

    def close(self) -> None:
        if self._owns_artifact:
            self._artifact.close()

    def __enter__(self) -> VisionArtifactBinding:
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.close()


__all__ = [
    "BindingError",
    "BoundResource",
    "FrontendResources",
    "PhysicalBlock",
    "VisionArtifactBinding",
    "VisionBinding",
    "VisionLayerBinding",
    "VisionMergerBinding",
]
