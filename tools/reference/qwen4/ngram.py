"""Exact integer oracle for Qwen4 PLE n-gram addressing."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterable

import torch


MASK64 = (1 << 64) - 1
SIGN64 = 1 << 63
SPLITMIX_GAMMA = 0x9E3779B97F4A7C15
SPLITMIX_M1 = 0xBF58476D1CE4E5B9
SPLITMIX_M2 = 0x94D049BB133111EB
LAYER_SEED_PRIME = 10007


@dataclass(frozen=True, slots=True)
class NGramConfig:
    unigram_vocab_size: int = 248_320
    ngram_size: int = 3
    heads_per_ngram: int = 8
    vocab_size_base: int = 20_000_000
    table_multiple: int = 128
    eos_token_id: int = 248_044
    seed: int = 1234
    ple_layer_index: int = 0


@dataclass(frozen=True, slots=True)
class NGramLayout:
    head_vocab_sizes: tuple[int, ...]
    head_offsets: tuple[int, ...]
    padded_vocab_size: int


@dataclass(frozen=True, slots=True)
class NGramState:
    """The raw represented predecessor ids carried by the preview PLE cache."""

    history: tuple[int, ...]

    @classmethod
    def initial(cls, config: NGramConfig = NGramConfig()) -> "NGramState":
        return cls((config.eos_token_id,) * (config.ngram_size - 1))


def _u64(value: int) -> int:
    return value & MASK64


def _i64(value: int) -> int:
    value &= MASK64
    return value - (1 << 64) if value & SIGN64 else value


def splitmix64(value: int) -> int:
    value = _u64(value + SPLITMIX_GAMMA)
    value = _u64((value ^ (value >> 30)) * SPLITMIX_M1)
    value = _u64((value ^ (value >> 27)) * SPLITMIX_M2)
    return _u64(value ^ (value >> 31))


def layer_multipliers(config: NGramConfig = NGramConfig()) -> tuple[int, ...]:
    max_long = SIGN64 - 1
    multiplier_max = max_long // max(config.unigram_vocab_size, 1)
    half_bound = max(1, multiplier_max // 2)
    base_seed = config.seed + LAYER_SEED_PRIME * config.ple_layer_index
    return tuple(
        2
        * (
            splitmix64(_u64(base_seed + SPLITMIX_GAMMA * (index + 1)))
            % half_bound
        )
        + 1
        for index in range(config.ngram_size)
    )


def is_prime(value: int) -> bool:
    if value < 2:
        return False
    if value % 2 == 0:
        return value == 2
    return all(value % divisor for divisor in range(3, math.isqrt(value) + 1, 2))


def nth_prime_after(start: int, count: int) -> int:
    value = start
    for _ in range(count):
        value += 1
        while not is_prime(value):
            value += 1
    return value


def layout(config: NGramConfig = NGramConfig()) -> NGramLayout:
    head_count = (config.ngram_size - 1) * config.heads_per_ngram
    sizes = tuple(
        nth_prime_after(
            config.vocab_size_base - 1,
            config.ple_layer_index * head_count + head + 1,
        )
        for head in range(head_count)
    )
    offsets: list[int] = []
    total = 0
    for size in sizes:
        offsets.append(total)
        total += size
    padded = math.ceil(total / config.table_multiple) * config.table_multiple
    return NGramLayout(sizes, tuple(offsets), padded)


def _mixed_id(tokens: tuple[int, ...], multipliers: tuple[int, ...]) -> int:
    mixed = _u64(tokens[0] * multipliers[0])
    for token, multiplier in zip(tokens[1:], multipliers[1:]):
        mixed ^= _u64(token * multiplier)
    return _i64(mixed)


def ids(
    token_ids: Iterable[int] | torch.Tensor,
    state: NGramState | None = None,
    config: NGramConfig = NGramConfig(),
) -> tuple[torch.Tensor, NGramState]:
    """Return exact table rows and the continuation history for one request."""

    tokens = [int(value) for value in torch.as_tensor(token_ids).reshape(-1).tolist()]
    if state is None:
        state = NGramState.initial(config)
    if len(state.history) != config.ngram_size - 1:
        raise ValueError("n-gram state has the wrong history length")
    multipliers = layer_multipliers(config)
    table = layout(config)
    history = list(state.history)
    rows: list[list[int]] = []
    for token in tokens:
        shifted = [token]
        for shift in range(1, config.ngram_size):
            predecessor_window = history[-shift:]
            if (
                len(predecessor_window) != shift
                or config.eos_token_id in predecessor_window
            ):
                shifted.append(config.eos_token_id)
            else:
                shifted.append(predecessor_window[0])
        token_rows: list[int] = []
        for ngram in range(2, config.ngram_size + 1):
            mixed = _mixed_id(tuple(shifted[:ngram]), multipliers[:ngram])
            begin = (ngram - 2) * config.heads_per_ngram
            for head in range(begin, begin + config.heads_per_ngram):
                token_rows.append(
                    mixed % table.head_vocab_sizes[head] + table.head_offsets[head]
                )
        rows.append(token_rows)
        if history:
            history = (history + [token])[-(config.ngram_size - 1) :]
    width = (config.ngram_size - 1) * config.heads_per_ngram
    result = torch.tensor(rows, dtype=torch.int64).reshape(len(tokens), width)
    return result, NGramState(tuple(history))


__all__ = [
    "NGramConfig",
    "NGramLayout",
    "NGramState",
    "ids",
    "is_prime",
    "layer_multipliers",
    "layout",
    "nth_prime_after",
    "splitmix64",
]
