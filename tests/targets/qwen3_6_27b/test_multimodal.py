from __future__ import annotations

import torch

from tools.reference.qwen3_6.common.multimodal import build_mrope_positions


def test_mrope_positions_for_images_and_video_frames():
    # Two image items and a two-frame video.  Timestamp text separates the
    # video frames exactly as the library processor emits them.
    types = torch.tensor([0, 1, 1, 1, 1, 0, 2, 2, 0, 2, 2, 0, 1, 0], dtype=torch.long)
    image_grid = torch.tensor([[1, 4, 4], [1, 2, 2]], dtype=torch.long)
    video_grid = torch.tensor([[2, 2, 4]], dtype=torch.long)
    positions, delta = build_mrope_positions(types, image_grid, video_grid)

    assert positions.tolist() == [
        [0, 1, 1, 1, 1, 3, 4, 4, 6, 7, 7, 9, 10, 11],
        [0, 1, 1, 2, 2, 3, 4, 4, 6, 7, 7, 9, 10, 11],
        [0, 1, 2, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
    ]
    assert delta == -2
