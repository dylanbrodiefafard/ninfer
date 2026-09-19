"""Regression for actual represented historical inputs in the local MTP QSA oracle."""

import copy
import math

import torch

from tools.reference.qwen4.mtp import RecordedQsaState, check_qsa_state, local_qsa


def fixture():
    p = "layers.0.self_attn."
    weights = {
        p + name + ".weight": torch.zeros(shape, dtype=torch.float64)
        for name, shape in {
            "q_proj": (12288, 2), "k_proj": (512, 2), "v_proj": (512, 2),
            "indexer.index_qk_proj": (640, 2), "o_proj": (2, 6144),
            "q_norm": (256,), "k_norm": (256,),
        }.items()
    }
    for head in range(24):
        weights[p + "q_proj.weight"][512 * head, 0] = 1
    weights[p + "v_proj.weight"][:, 0] = 100  # New row must not enter the frozen domain.
    weights[p + "o_proj.weight"][0, 0] = 1
    weights[p + "o_proj.weight"][1, 1] = 1
    keys = torch.zeros((2, 2, 256), dtype=torch.float64)
    keys[0, :, 0], keys[1, :, 0] = 1, 2  # Already transformed cache; not unit-RMS vectors.
    values = torch.full_like(keys, 2)
    values[1] = 10
    ids = torch.full((2051,), -1, dtype=torch.long)
    ids[:2] = torch.tensor([0, 1])
    previous = RecordedQsaState(keys, values, torch.full((2, 128), 13., dtype=torch.float64),
        torch.tensor([[91, 92], [73, 74], [55, 56]]), ids, 2)
    return weights, previous


def test_local_qsa_consumes_actual_historical_cache_without_retransform_or_tail():
    weights, previous = fixture()
    hidden = torch.tensor([[1., 0.]], dtype=torch.float64)
    positions = torch.zeros((3, 1), dtype=torch.long)
    output, expected = local_qsa(weights, hidden, positions, True, previous)
    # q[0]/sqrt(256) = 1/sqrt(1+256e-6); old stored K magnitudes remain 1 and 2.
    probability_second = 1 / (1 + math.exp(-1 / math.sqrt(1 + 256e-6)))
    wanted = (2 + 8 * probability_second) * .5
    torch.testing.assert_close(output, torch.full_like(output, wanted), atol=1e-12, rtol=1e-12)
    assert torch.equal(expected.keys[:2], previous.keys)
    assert torch.equal(expected.raw_keys[:2], previous.raw_keys)
    assert torch.equal(expected.positions[:, :2], previous.positions)
    assert torch.equal(expected.selected_ids, previous.selected_ids)
    assert expected.selected_count == 2
    assert torch.equal(expected.values[-1], torch.full_like(expected.values[-1], 100))
    assert check_qsa_state("unchanged history", expected, expected, previous) == 0

    # A newly supplied represented historical V row, rather than a reconstructed projection,
    # must immediately alter the local oracle. No independently evolved history is consulted.
    changed = copy.deepcopy(previous)
    changed.values[1] += 8
    changed_output, _ = local_qsa(weights, hidden, positions, True, changed)
    torch.testing.assert_close(changed_output - output,
        torch.full_like(output, 4 * probability_second), atol=1e-12, rtol=1e-12)


def test_local_qsa_state_gate_rejects_historical_mutation_and_changed_frozen_domain():
    weights, previous = fixture()
    _, expected = local_qsa(weights, torch.tensor([[1., 0.]], dtype=torch.float64),
                            torch.zeros((3, 1), dtype=torch.long), True, previous)
    corrupted = copy.deepcopy(expected)
    corrupted.values[0, 0, 0] += 1
    corrupted.positions[0, 0] += 1
    corrupted.selected_ids[1] = 2
    corrupted.selected_count = 1
    assert check_qsa_state("corrupted history", corrupted, expected, previous) == 4
