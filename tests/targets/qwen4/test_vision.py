from __future__ import annotations

import math

import torch

from tools.reference.qwen4.vision import arrange_2x2, patch_merger


def test_vision_merge_order_is_top_left_top_right_bottom_left_bottom_right() -> None:
    groups = torch.tensor(
        [
            [[[10.0, 11.0], [20.0, 21.0]], [[30.0, 31.0], [40.0, 41.0]]],
            [[[50.0, 51.0], [60.0, 61.0]], [[70.0, 71.0], [80.0, 81.0]]],
        ]
    )
    expected = torch.tensor(
        [
            [10.0, 11.0, 20.0, 21.0, 30.0, 31.0, 40.0, 41.0],
            [50.0, 51.0, 60.0, 61.0, 70.0, 71.0, 80.0, 81.0],
        ],
        dtype=torch.float64,
    )
    torch.testing.assert_close(arrange_2x2(groups), expected, atol=0.0, rtol=0.0)

    # A column-major or BL/TR-swapped implementation must not pass this witness.
    wrong = groups.permute(0, 2, 1, 3).reshape(2, 8).double()
    assert not torch.equal(expected, wrong)


def test_vision_patch_merger_matches_complete_hand_formula() -> None:
    patches = torch.tensor(
        [[[[1.0, 3.0], [2.0, -2.0]], [[4.0, 0.0], [-1.0, 5.0]]]]
    )
    norm_weight = torch.tensor([2.0, 0.5])
    norm_bias = torch.tensor([0.25, -0.75])
    fc1_weight = torch.tensor(
        [
            [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -0.5],
            [0.0, 1.0, 0.0, 0.0, 0.0, 0.25, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0, -0.5, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.5, 0.0],
            [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.25],
            [0.5, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
            [0.0, -0.25, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.5, 0.0, 0.0, 0.0, 0.0, 1.0],
        ]
    )
    fc1_bias = torch.tensor([0.1, -0.2, 0.3, -0.4, 0.5, -0.6, 0.7, -0.8])
    fc2_weight = torch.tensor(
        [
            [1.0, -0.5, 0.25, 0.0, 0.0, 0.5, -0.25, 1.0],
            [0.0, 0.5, 1.0, -1.0, 0.25, 0.0, 0.5, -0.5],
        ]
    )
    fc2_bias = torch.tensor([1.25, -2.5])

    actual = patch_merger(
        patches,
        norm_weight,
        norm_bias,
        fc1_weight,
        fc1_bias,
        fc2_weight,
        fc2_bias,
    )

    x = patches.double()
    centered = x - x.mean(dim=-1, keepdim=True)
    normalized = centered / torch.sqrt(
        (centered * centered).mean(dim=-1, keepdim=True) + 1e-6
    )
    normalized = normalized * norm_weight.double() + norm_bias.double()
    merged = normalized.reshape(1, 8)
    pre_activation = merged @ fc1_weight.double().t() + fc1_bias.double()
    activated = 0.5 * pre_activation * (
        1.0 + torch.erf(pre_activation / math.sqrt(2.0))
    )
    expected = activated @ fc2_weight.double().t() + fc2_bias.double()

    torch.testing.assert_close(
        actual.normalized_tokens, normalized, atol=1e-14, rtol=1e-14
    )
    torch.testing.assert_close(actual.merged_tokens, merged, atol=1e-14, rtol=1e-14)
    torch.testing.assert_close(
        actual.fc1_pre_activation, pre_activation, atol=1e-14, rtol=1e-14
    )
    torch.testing.assert_close(actual.fc1_activation, activated, atol=1e-14, rtol=1e-14)
    torch.testing.assert_close(actual.output, expected, atol=1e-14, rtol=1e-14)


def test_vision_layer_norm_is_per_patch_before_the_2x2_merge() -> None:
    patches = torch.tensor(
        [[[[1.0, 2.0], [10.0, 14.0]], [[-2.0, 4.0], [100.0, 102.0]]]]
    )
    identity = torch.eye(8)
    result = patch_merger(
        patches,
        norm_weight=torch.ones(2),
        norm_bias=torch.tensor([0.5, -0.25]),
        fc1_weight=identity,
        fc1_bias=torch.zeros(8),
        fc2_weight=identity,
        fc2_bias=torch.zeros(8),
        eps=0.0,
    )

    x = patches.double()
    per_patch = (x - x.mean(-1, keepdim=True)) / torch.sqrt(
        ((x - x.mean(-1, keepdim=True)) ** 2).mean(-1, keepdim=True)
    )
    per_patch = per_patch + torch.tensor([0.5, -0.25], dtype=torch.float64)
    torch.testing.assert_close(
        result.merged_tokens, per_patch.reshape(1, 8), atol=0.0, rtol=0.0
    )

    merged = patches.double().reshape(1, 8)
    postshuffle = (merged - merged.mean(-1, keepdim=True)) / torch.sqrt(
        ((merged - merged.mean(-1, keepdim=True)) ** 2).mean(-1, keepdim=True)
    )
    assert not torch.allclose(result.merged_tokens, postshuffle)
