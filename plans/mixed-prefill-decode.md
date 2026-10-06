# Mixed prefill/decode: finish small-chunk prefill, then design scheduling

This is a temporary handoff plan. Delete it when phase 3 is complete, and move any lasting result
into `docs/performance.md` and `docs/maintainer/concurrent-inference-architecture.md`.

## Goal

The work has three phases, done in order:

1. **Prefill (complete).** The small-chunk PDL pass improves the resumed tree by 4.65% at
   T=64 and 3.27% at T=128. Paired sweeps cover tiny, irregular, and boundary widths through
   4095. Focused numerical, graph, sanitizer, and real-model regression checks pass.
2. **Score table.** Reproduce the AMD repository's mixed prefill/decode score table on the RTX 5090.
   The AMD repository is `../ninfer-amd-r9700`: see `docs/performance.md`, section "Mixed
   prefill/decode frontier (2026-09-29)", and commits `4a9382b5` and `98f30bc3`.
3. **Scheduler (deferred).** Design scheduling with the user after prefill qualification. The
   decision must account for small versus large prefill jobs, the number of decoding lanes, and
   measured score above pure time sharing. No scheduling policy is selected by this handoff.

**Score definition.** score = (mixed prefill tok/s ÷ same-build prefill-first tok/s) + decode share.
- Decode share = decode rounds/s while the prefill runs ÷ decode rounds/s with no prefill.
- Pure time-sharing scores 1.0.
- The AMD optimum scores about 1.06, with a 1024-column forward, D = 1 and S = 1024 − W·(C−1).

**Standing user instructions:**
- Get a read-only subagent code review at every checkpoint, before any GPU work.
- Do every clearly beneficial optimization you find.
- A4 verify (NVFP4 activations for the speculative verify forward) is not allowed.
- Commit only when the user asks.

## Phase 1: finish small-chunk prefill

The branch is `experimental`, with uncommitted work on top of `945515ea`. The diff covers about 112
files, roughly +4000/−1200 lines. These five new files are marked intent-to-add (`git add -N`):
- `include/ninfer/ops/a4_activation.h`
- `src/ops/common/a4_activation_check.h`
- `src/ops/kernel/a4_activation.cuh`
- `tests/ops/test_a4_activation.cpp`
- `tests/targets/qwen3_6_27b/test_engine_mixed_forward_real.cpp`

### What the tree contains

**Prefill performance work.** Every item was measured with an interleaved A/B on the 5090.
- The NVFP4 W4A4 MMA kernels order their CTAs token-tile first. One route table,
  `nvfp4_w4a4_route()` in `nvfp4_config.h`, serves all five NVFP4 dispatchers.
- SwiGLU with A4 output is always fused; the Linear+silu route was removed.
- Partial token tiles:
  - The kernels skip fragments that hold only padding (`nvfp4_with_active_token_fragments`).
  - TMA loads only the active fragments, using at most four power-of-two boxes.
  - Only reachable schedules are instantiated.
  - The `NINFER_NVFP4_TMA_SCHEDULE` A/B hook and `nvfp4_tma_ab_bench` were removed.
- The DFlash feature and KV projections use A16 prefill tile tables (`nvfp4_dflash_feature_a16_tile`
  and `nvfp4_dflash_kv_a16_tile` in `nvfp4_config.h`).
- GDN uses its chunked kernel from T ≥ 60 (`kChunkedMinTokens`) and handles a partial last chunk.
- Fused NVFP4 activation producers, using the `A4Activation` tiled scale plane:
  - rmsnorm feeds the attention input and gate_up projections;
  - the SwiGLU epilogue feeds down;
  - sigmoid_mul and gated_rmsnorm feed o_proj and out_proj;
  - the GDN norm feeds the GDN input projection.
- `zero_slot` resets a slot with two `cudaMemset2DAsync` calls instead of 96 memsets.
- **Result** (whole-model prefill against the base commit):

  | T | Change |
  |---|---|
  | 96 / 160 / 192 | −18 to −23% |
  | 256 | −11.5% |
  | 320 / 384 | −8% |
  | 640 | −4% |
  | 1000 | −2.7% |
  | ≥ 1024 | about 0 to −1.5% |

  Evidence: `profiles/bench/small-chunk-prefill-20261004/`.

**Mixed prefill/decode infrastructure.**
- New `EngineOptions::mixed_forward` (0 = off; otherwise a multiple of 256, at most
  min(prefill_chunk, 4096)) and `mixed_forward_rounds` (≥ 1), each with a serve flag
  (`--mixed-forward`, `--mixed-forward-rounds`).
- `PrefillPace {Exclusive, Shared}`: under `Shared`, the prefill owner steps `mixed_forward` tokens
  at a time and `mixed_forward_rounds` decode rounds run between its steps.
- A mixed DFlash round runs the owner's prefill chunk inside the verify forward:
  - Components: `MixedPrefillOwner` and `LayerPartner` / `MixedOwnerForward` in `text_context.h`.
  - The owner's DFlash ingress row is the last row and is restaged before every chunk.
  - Tool-grammar prefill sampling is rebound before the final chunk
    (`bind_prefill_sampling`).
- The executor's owner branch lives in `src/runtime/engine/concurrent_executor.h`.
- The bench gained `--contention P,R`, `--contention-context`, `--contention-lanes` and
  `--contention-baseline`, which produce the score table.
- `docs/serving.md` and §2.5/§7.3 of `docs/maintainer/concurrent-inference-architecture.md` are
  updated.
- New real test `ninfer_qwen3_8_27b_mixed_forward_real_test`. It passes: the owner's and lanes'
  greedy streams equal their solo runs.

**Clang-tidy fixes from 2026-10-05 (13 findings):**
- Optionals are read through `.value()` in the bench contention code,
  `stage_dflash_prefill_owner`, the DFlash ingress copy and `MixedPrefillOwner::result`.
- A no-op `std::move` was removed in `concurrent_executor.h`.
- `run_decode_round` now receives `prefill_lane_` directly.
- A misplaced widening cast was fixed in `dflash_impl.h`.
- Two findings were in files this work never touched, and were fixed so the tree is clean: a duplicate
  include in `test_grouped_dynamic_conv_composite.cpp` and a misplaced const in
  `normalized_rope_kv_append_bench.cpp`.

### Gate status

The gates are those of `docs/maintainer/merging-to-master.md`.

| Gate | Status |
|---|---|
| 1. `pre-commit run --all-files` | Pass, with no rewrites on the final run |
| 2. Full `-Werror` build (tests and benchmarks on) | Pass |
| 3. clang-tidy over the whole tree | Pass: 0 diagnostics in 490 translation units |
| 4. All unit tests, slow ones included | Pass: 123 passed, 2 artifact-dependent skips |
| 5. `--real` Engine tests | Pass for available artifacts: 12 passed, 8 skips |
| 6. compute-sanitizer memcheck | Pass: all 44 sanitizer cases |
| 7. compute-sanitizer racecheck | 28 cases passed; broad run stopped to focus on prefill |
| 8. compute-sanitizer initcheck | Deferred; no merge qualification claimed |
| 9. AddressSanitizer and UBSan | Deferred; no merge qualification claimed |

Earlier checkpoint results, from before the final mixed-round and clang-tidy changes:
- The fast suite passed, and the full suite passed 144/144.
- `ninfer_a4_activation_test` passed memcheck and racecheck.
- The mixed-forward real test passed.

**Resume scope (2026-10-05).** The RTX 5090 is back on the NVIDIA driver. The user narrowed
this run to small-chunk prefill; the score table and scheduling are subsequent joint work.
`master` is already an ancestor of `experimental`. No commit or merge is requested.

The resumed checks above cover the inherited implementation and the strengthened mixed-forward
regression: `RuntimeStats::mixed_decode_rounds` proves that the owner actually entered two mixed
rounds, rather than merely observing decode rounds later in its generation. Numerical or
performance changes made during this resume require their own focused qualification.

Fresh eager Engine baselines span T=1..4095, including 59/60, 64/65, 128/129, and 256/257.
Nsight Systems at 64 and 160 tokens shows about 1.5 ms of already-queued inter-kernel gaps per
prefill. At T=64, Nsight Compute measures the 5120x17408 down projection at 160 CTAs and
1197.6 GB/s DRAM reads, admitting an output-row partition candidate through `tools.kdev`.
Evidence: `profiles/bench/small-chunk-prefill-20261005/`,
`profiles/nsys/small-chunk-20261005/`, and `profiles/ncu/small-chunk-20261005/`.

### Retained prefill result

The family W4A4 launcher selects PDL except for the M64N64 residual schedule at 113–128 tokens,
which retains ordinary stream ordering. The quantizer and A4 RMSNorm consumers also use PDL.
Every dependent read or write waits for predecessor completion; MMA producers release setup at
the epilogue. Arithmetic and codec boundaries are unchanged, and A4 verify remains prohibited.

Three alternating Engine pairs (seven repetitions, three warmups) show 4.38–4.65% lower latency
at 16–64 tokens, 3.27% at 128, 1.64% at 256, and 2.26% at 257. Larger chunks are effectively
unchanged. Two alternating pairs on an 8192-token prompt show 3.15% lower owner latency with
128-token chunks and 1.57% with 256-token chunks. Full results are in `docs/performance.md`,
section "Qwen3.8-27B NVFP4 small-chunk prefill (2026-10-05)".

The admitted N32 output-row partition candidate passed correctness but lost about 14% at T=64
and was deleted. The retained M64N64 ordering removes the repeatable T=128 PDL regression.
No losing kernel, environment toggle, or alternative mathematical path remains.

Independent Linear/LinearAdd oracles, exact A4 codec checks, refreshed-input graph replay at
64/128, and focused A4 memcheck/racecheck pass. All 10 final affected-Op tests pass. The real
DFlash mixed regression passes with two actual mixed rounds and greedy-stream parity. Final
Nsight Systems timelines show reduced phase spans and launch gaps at both 64 and 128 tokens.
The final build passes with `-Werror`; clang-tidy reports zero diagnostics across 209 affected
translation units. The final formatting check passes without rewrites.

### Current steps

The authorized 128–4096-token follow-up tested five kernel-family candidates against the
completed PDL baseline. None provided a reliable whole-prefill gain, so all candidate code was
removed. Public-Op wins from a localized residual tile and TMA PDL did not survive the paired
Engine criterion. Results and the baseline numerical diagnostic are recorded in
`docs/performance.md`, section "Qwen3.8-27B NVFP4 128–4096-token prefill follow-up (2026-10-05)".
The restored `-Werror` build passes. Focused NVFP4 Linear/LinearAdd, BF16 Linear, and attention
projection oracles pass. Both original real DFlash mixed cases pass with two actual mixed rounds
and owner/decode-lane greedy parity. The follow-up is complete.

Score-table measurement and scheduling await the next joint decision with the user; no
scheduling policy is selected.

### Gate 5 caveat

Gate 5 also needs `qwen3_8_27b_nvfp4.ninfer` and the MTP variant. Only the DFlash artifact is on
this machine, so the resumed full `--real` run skipped eight Engine tests. Either place the missing artifacts
or state the skips as a limitation.

## Environment and gotchas

- **Builder container.** Use `ninfer-builder-sandbox`. It mounts this checkout at `/src`, with the
  build tree in `build/` mounted at `/build`. The default `ninfer-builder` mounts the other
  checkout (`/ssdpool2nvme/local_llm/ninfer`). Always set:
  - `NINFER_DEV_CONTAINER=ninfer-builder-sandbox` for `scripts/run-unit-tests.sh` and
    `scripts/run-clang-tidy.py`;
  - `NINFER_DEV_JOBS=8`.
- **Real tests.** Set
  `NINFER_QWEN3_8_27B_NVFP4_DFLASH_WEIGHTS=/ssdpool2nvme/local_llm/models/qwen3.8-nvfp4-flash2-nvfp4-bf16codebook-from-bf16/qwen3_8_27b_nvfp4_dflash_nvfp4.ninfer`.
  Inside the container the same file is under `/models/...`. The sandbox has no
  `models/weights.env`.
- **Host load.** Use at most 8 build jobs. Run one heavy job at a time. Before a GPU run, check
  `free -g` and host `nvidia-smi`. Take the lock: `flock /tmp/ninfer-gpu-queue.lock <command>`.
- **Parallel builds.** Never edit files or build while a GPU chain that uses `run-unit-tests.sh`
  is running, because that script rebuilds.
- **clang-tidy.** It must run in the container against `/build`. CMake must have been configured
  with `-S /src`, or clang-tidy selects nothing.
- **Shell.** zsh does not word-split `$VAR`; use a file plus `xargs`.
- **Chain scripts.** Grep ctest output for `tests passed|FAILED`, not `tail`.
- **A/B toggles.** Verify that the off binary actually differs, and abort if `cmp` says the two
  binaries are identical.
- **Profiling.** nsys with CUDA graphs fails with "graph exec update". Profile with
  `--no-cuda-graph`.
- **`setmaxnreg` deadlock.** `setmaxnreg` only redistributes the CTA's launch register allocation
  (threads × compiled registers; 384 × 168 = 64512 for the TMA kernels). Requesting more than that
  hangs the `.inc` forever, and it hung the shared GPU once.

## Phase 2: score table findings so far

**Measured on 2026-10-04.** Setup:
- C4, i.e. three decode lanes at 512 context, plus an owner that prefills 8K prompts;
  `--contention 8192,3`.
- DFlash k5, adaptive draft, LM-head draft.
- Evidence: `profiles/bench/mixed-forward-20261004/`; the `sep_*` files are separate steps and the
  `mix_*` files are mixed rounds.

| Config (`mixed_forward` × rounds) | Separate steps | Mixed rounds |
|---|---|---|
| prefill-first | 1.02–1.04 (noise ±0.02) | — |
| 2048 × 1 | **1.01** | 1.004 |
| 1024 × 1 | 0.963 | 0.956 |
| 1024 × 2 | 0.96 | 0.952 |
| 512 × 1 | 0.88 | 0.857 |

Bench command, run in the container (`--mixed-forward 0` is the prefill-first baseline):

```bash
flock /tmp/ninfer-gpu-queue.lock docker exec ninfer-builder-sandbox /build/bench/ninfer_bench \
  --weights /models/qwen3.8-nvfp4-flash2-nvfp4-bf16codebook-from-bf16/qwen3_8_27b_nvfp4_dflash_nvfp4.ninfer \
  --spec dflash --draft-tokens 5 --adaptive-draft --lm-head-draft --concurrency 4 \
  --contention 8192,3 --mixed-forward 1024 --mixed-forward-rounds 1 -o json --output-file <file>
```

**Why mixed rounds give nothing on the 5090:**
- The verify W4A8 GEMMs take 10.7 of the 13.6 ms per round.
- They are latency/issue-bound at about 1.2 TB/s, and are no faster with the weights already in L2.
- Interleaving layers therefore shares nothing.
- Running owner and verify on two concurrent streams recovers only 25–45% of the verify GEMM time.

**Fused dual-precision GEMM: tried 2026-10-04, reverted, do not retry.**
- Design: the verify A8 columns were computed inside the owner's TMA SwiGLU CTAs.
- It was bit-exact with the W4A8 route but slower:
  - +154 to +290 µs added at owner widths 1024 and 2048, against 64 µs for the separate verify;
  - +12 to +32 µs of that is structural overhead with the verify math disabled.
- Cause: at these widths the owner W4A4 GEMM is tensor-pipe bound. The verify needs one FP8 k16
  MMA per 16-element scale group, with an issue floor of about 45 µs against a 61 µs DRAM floor,
  so sharing the weight read cannot pay.

**Remaining verify headroom.** Per-projection bounds at T = 18 (A8):

| Projection | DRAM floor | Issue floor | Measured in round |
|---|---|---|---|
| gate_up | 60.7 µs | 44.9 µs | 79 µs (64 µs in a hot loop) |
| down | 30.4 µs | 22.5 µs | 45 µs |
| o_proj | 10.8 µs | 7.9 µs | 22 µs |
| gdn_in | 28.6 µs | 21.1 µs | — |
| attn_in | 25.1 µs | 18.5 µs | — |

Per round, the verify GEMMs total 10.7 ms against an 8.3 ms DRAM floor, so at most about 2 ms per
round (about 2% of a cycle) can be recovered. The next step was to profile the 64 → 79 µs
in-round gap with nsys.

**Completed eager-prefill optimization.** PDL was measured and retained during phase 1; see
the retained prefill result above. Its measured gains replace the earlier estimate of this
launch-gap opportunity.

## Phase 3: scheduler (deferred joint design)

The user will work with the agent on scheduling after small-chunk prefill optimization. The
policy must be informed by measured score above pure time sharing across small and large
prefill jobs and the number of decode lanes. The earlier scheduling suggestions are not an
approved design. Any selected policy must respect the startup-fixed one-to-six-request workload,
bounded FIFO ingress, and no-preemption product contract, unless the user explicitly changes it.

The subsequent work will establish the score frontier with the optimized prefill build, select
and implement the policy with the user, validate real request behavior, and update the active
performance and architecture references. No commit or merge is authorized by this plan.
