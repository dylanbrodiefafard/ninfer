"""Compare ninfer_qwen3_8_flash_next_parity log-probabilities with the study's parity reference.

Both inputs hold float16 log-softmax values [S, T-1, 248320] over every head row at positions
0..T-2: the engine's `logprobs.npy` (parity driver, `--logits all`) and the reference `.npz`
(`logprobs` key, produced by the study harness from the same tokens and artifact recipe). Prints
per-sequence and overall KL(reference || engine) mean, p50, p90, p99 and max and top-1 agreement. Differences
come from implementation precision (BF16 residuals, kernel rounding, routing flips), so the
result is a distribution, not an equality check.
"""

from __future__ import annotations

import argparse
import pathlib

import numpy as np

POSITION_CHUNK = 32


def summary(kl: np.ndarray, top1: np.ndarray) -> str:
    p50, p90, p99 = np.quantile(kl, (0.5, 0.9, 0.99))
    return (
        f"KL mean {kl.mean():.5f} p50 {p50:.5f} p90 {p90:.5f} p99 {p99:.5f} max {kl.max():.5f}"
        f"  top-1 {top1.mean():.4f}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("engine", type=pathlib.Path, help="logprobs.npy from the parity driver")
    parser.add_argument("reference", type=pathlib.Path, help="reference .npz with key logprobs")
    args = parser.parse_args()

    engine = np.load(args.engine, mmap_mode="r")
    with np.load(args.reference) as archive:
        reference = archive["logprobs"]
    if engine.shape != reference.shape:
        raise SystemExit(f"shape mismatch: engine {engine.shape}, reference {reference.shape}")

    all_kl: list[np.ndarray] = []
    all_top1: list[np.ndarray] = []
    for s in range(engine.shape[0]):
        kl_rows: list[np.ndarray] = []
        top1_rows: list[np.ndarray] = []
        for p0 in range(0, engine.shape[1], POSITION_CHUNK):
            log_p = reference[s, p0 : p0 + POSITION_CHUNK].astype(np.float64)
            log_q = engine[s, p0 : p0 + POSITION_CHUNK].astype(np.float64)
            kl_rows.append((np.exp(log_p) * (log_p - log_q)).sum(axis=-1))
            top1_rows.append(log_p.argmax(axis=-1) == log_q.argmax(axis=-1))
        kl = np.concatenate(kl_rows)
        top1 = np.concatenate(top1_rows)
        print(f"sequence {s}: {summary(kl, top1)}")
        all_kl.append(kl)
        all_top1.append(top1)
    kl = np.concatenate(all_kl)
    top1 = np.concatenate(all_top1)
    print(f"all {kl.size} positions: {summary(kl, top1)}")


if __name__ == "__main__":
    main()
