"""Fixed Vision-tower configuration for the Qwen3.8-27B reference path."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class VisionConfig:
    depth: int = 27
    hidden: int = 1152
    intermediate: int = 4304
    out_hidden: int = 5120
    heads: int = 16
    in_channels: int = 3
    patch: int = 16
    temporal_patch: int = 2
    spatial_merge: int = 2
    position_embeddings: int = 2304
    rope_theta: float = 10000.0
    norm_eps: float = 1.0e-6

    @property
    def head_dim(self) -> int:
        return self.hidden // self.heads

    @property
    def patch_dim(self) -> int:
        return self.in_channels * self.temporal_patch * self.patch * self.patch

    @property
    def merge_unit(self) -> int:
        return self.spatial_merge * self.spatial_merge

    @property
    def merger_hidden(self) -> int:
        return self.hidden * self.merge_unit

    @property
    def position_side(self) -> int:
        return int(self.position_embeddings**0.5)


VISION_CFG = VisionConfig()
