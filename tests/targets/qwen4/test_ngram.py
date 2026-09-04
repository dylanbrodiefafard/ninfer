from __future__ import annotations

import torch

from tools.reference.qwen4.ngram import (
    NGramConfig,
    NGramState,
    ids,
    layer_multipliers,
    layout,
    splitmix64,
)


def test_splitmix_and_preview_layout_have_exact_frozen_values() -> None:
    assert splitmix64(0) == 16_294_208_416_658_607_535
    assert splitmix64(1) == 10_451_216_379_200_822_465
    assert splitmix64((1 << 64) - 1) == 16_490_336_266_968_443_936
    assert layer_multipliers() == (
        23_703_573_157_769,
        20_109_073_645_365,
        8_052_911_324_071,
    )
    table = layout()
    assert table.head_vocab_sizes == (
        20_000_003,
        20_000_023,
        20_000_033,
        20_000_047,
        20_000_059,
        20_000_063,
        20_000_069,
        20_000_077,
        20_000_081,
        20_000_093,
        20_000_107,
        20_000_147,
        20_000_153,
        20_000_159,
        20_000_161,
        20_000_171,
    )
    assert table.head_offsets == (
        0,
        20_000_003,
        40_000_026,
        60_000_059,
        80_000_106,
        100_000_165,
        120_000_228,
        140_000_297,
        160_000_374,
        180_000_455,
        200_000_548,
        220_000_655,
        240_000_802,
        260_000_955,
        280_001_114,
        300_001_275,
    )
    assert table.padded_vocab_size == 320_001_536


def test_preview_ids_are_exact_and_eos_resets_both_histories() -> None:
    actual, state = ids([1, 248_044, 2, 3])
    assert actual.tolist() == [
        [
            16_121_432,
            28_938_500,
            59_087_997,
            73_487_090,
            81_148_277,
            104_500_129,
            120_276_032,
            149_373_875,
            176_283_436,
            184_305_849,
            216_528_839,
            231_080_079,
            257_961_536,
            266_068_568,
            289_043_455,
            305_959_965,
        ],
        [
            2_760_535,
            21_194_529,
            54_821_219,
            66_837_341,
            96_008_662,
            113_339_850,
            120_218_497,
            157_703_057,
            168_406_994,
            185_107_469,
            219_451_979,
            237_849_981,
            246_295_160,
            275_701_920,
            292_384_545,
            317_399_567,
        ],
        [
            557_681,
            34_951_092,
            45_888_726,
            61_391_243,
            89_998_281,
            113_665_401,
            129_914_206,
            159_642_585,
            174_754_477,
            190_524_023,
            208_451_907,
            222_160_281,
            242_915_276,
            264_896_005,
            285_828_740,
            312_534_661,
        ],
        [
            14_605_717,
            24_410_875,
            49_313_567,
            72_177_428,
            86_060_820,
            104_022_010,
            130_963_819,
            146_886_202,
            165_072_300,
            195_686_070,
            213_219_648,
            231_243_893,
            250_325_810,
            269_767_253,
            296_327_789,
            309_728_713,
        ],
    ]
    assert state == NGramState((2, 3))


def test_prefill_and_token_steps_produce_identical_ids_and_state() -> None:
    tokens = torch.tensor([7, 11, 248_044, 19, 23, 29], dtype=torch.int64)
    one_shot, one_shot_state = ids(tokens)
    pieces = []
    state = None
    for token in tokens:
        piece, state = ids([token], state)
        pieces.append(piece)
    assert torch.equal(one_shot, torch.cat(pieces))
    assert state == one_shot_state


def test_continuation_state_keeps_raw_ids_while_hashing_ignores_pre_eos_ids() -> None:
    _, state = ids([17, 248_044])
    assert state == NGramState((17, 248_044))
    after_eos, next_state = ids([19, 23], state)
    from_reset, _ = ids([19, 23], NGramState.initial())
    assert torch.equal(after_eos, from_reset)
    assert next_state == NGramState((19, 23))


def test_small_config_has_exact_head_order_offsets_and_rows() -> None:
    config = NGramConfig(
        unigram_vocab_size=3,
        ngram_size=3,
        heads_per_ngram=1,
        vocab_size_base=5,
        table_multiple=1,
        eos_token_id=2,
        seed=(1 << 64) - 2,
    )
    result, _ = ids([2, 1], config=config)
    table = layout(config)
    assert table.head_vocab_sizes == (5, 7)
    assert table.head_offsets == (0, 5)
    assert table.padded_vocab_size == 12
    assert result.tolist() == [[2, 6], [1, 5]]
