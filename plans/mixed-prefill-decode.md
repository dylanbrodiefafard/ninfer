# Mixed prefill/decode: finish small-chunk prefill, then design scheduling

This is a temporary handoff plan. Delete it when phase 3 is complete, and move any lasting result
into `docs/performance.md` and `docs/maintainer/concurrent-inference-architecture.md`.

## Goal

The work has three phases, done in order:

1. **Prefill (complete).** The small-chunk PDL pass improves the resumed tree by 4.65% at
   T=64 and 3.27% at T=128. Paired sweeps cover tiny, irregular, and boundary widths through
   4095. Focused numerical, graph, sanitizer, and real-model regression checks pass.
2. **Score table (complete).** Reproduce the AMD repository's mixed prefill/decode score table on the RTX 5090.
   The AMD repository is `../ninfer-amd-r9700`: see `docs/performance.md`, section "Mixed
   prefill/decode frontier (2026-09-29)", and commits `4a9382b5` and `98f30bc3`.
3. **Scheduler (deferred).** Design scheduling with the user after prefill qualification. The
   decision must account for small versus large prefill jobs, the number of decoding lanes, and
   measured score above pure time sharing. No scheduling policy is selected by this handoff.

**Score definition.** score = (mixed prefill tok/s ÷ same-build exclusive-prefill tok/s) + decode share.
- Decode share = decode rounds/s while the prefill runs ÷ the mean decode-only rate before and
  after that run.
- An ideal division of time between constant-rate phases scores 1.0. The final campaign measures
  independent exclusive-prefill rates at every matching startup capacity before and after the
  contention runs, with interpolation by measurement time.
- The contention harness services decode between successive jobs even in its prefill-first
  control. Report the score difference from adjacent controls with the same exclusive
  denominator. Preserve the earlier contended-control normalization separately; there the
  observed control scores `1 + control decode share`.
- The AMD optimum reports about 1.06 under its own benchmark normalization, with a 1024-column
  forward, D = 1 and S = 1024 − W·(C−1). Comparisons require matching the baseline definition.

**Standing user instructions:**
- Get a read-only subagent code review at every checkpoint, before any GPU work.
- Do every clearly beneficial optimization you find.
- A4 verify (NVFP4 activations for the speculative verify forward) is not allowed.
- Commit only when the user asks.

## Phase 1: finish small-chunk prefill

The original phase-1 snapshot used `experimental`, with uncommitted work on top of `945515ea`. Its diff covered about 112
files, roughly +4000/−1200 lines. These five files were marked intent-to-add (`git add -N`):
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

The prefill work is committed at `74541d23` and pushed on
`perf/mixed-prefill-small-chunks`; the remote `experimental` branch contains a separate
concurrent commit. The A4 T=1500 diagnostic reproduces on base `945515ea`. Its complete
7,680,000-output FP64 oracle passes the unchanged criterion on both base and retained trees;
the 49-output sample was unrepresentative. LinearAdd now retains every original sampled
coordinate, adds a 32-point grid on each axis, and includes T=1500 for both residual geometries.
The expanded A16/A8/A4 suite passes. Numerical evidence is recorded in `docs/performance.md`,
section "LinearAdd A4 numerical follow-up (2026-10-06)". The fix is complete: read-only review,
`-Werror` build, the fast suite (119 passes and two artifact-dependent skips), affected
clang-tidy (zero diagnostics), and formatting/file checks pass.

### Profiler limit confirmation and score table (requested 2026-10-06)

The user requested profiler evidence establishing the practical prefill performance limit before
building the mixed-execution score table. The completed prefill passes remain the baseline.
Scheduling design remains joint work after the table is available.

- [x] Attribute the dominant prefill costs and calibrated roofs across 128–4096-token chunks and
      a growing-context prompt.
- [x] Test credible profiler-driven candidates through independent numerical qualification,
      public-Op measurements, and paired Engine comparisons; retain only reproducible wins.
- [x] Build the same-build mixed prefill/decode score table across small and large prefill jobs,
      decode-lane counts, chunk widths, and decode rounds; present the measured limits and frontier.

The selective short-query attention route is retained: fresh ABBA Engine gains are 0.29–0.64%
at 127–256 tokens, with independent FP64 qualification, complete focused memcheck/racecheck,
fast tests, clang-tidy and both DFlash/mixed-forward real-model tests passing. The profiler
limit report and final score table are in `docs/performance.md`, sections "Prefill limit
profiling (2026-10-06)" and "Mixed prefill/decode frontier (2026-10-06)". The final score campaign
has 40 exclusive baselines, 100 candidate measurements and 60 controls over 82 settings.
Evidence is `profiles/bench/mixed-score-20261006-gqa-selective/`.

No scheduling policy is selected. Absolute optimality is not a measurable claim; the confirmation
must state the hardware-bound headroom and any remaining material uncertainty.

### Mixed-round profiling and graph opportunity (complete, 2026-10-07)

This investigation targets actual mixed execution at 1024- and 2048-token owner steps, preserving
A8 speculative verify and leaving scheduler selection for joint work. The preceding campaign
qualified the best tested standalone prefill route; it did not prove hardware optimality.

- [x] Attribute actual owner-plus-verify round costs, host launch gaps and recoverable overhead
      at representative decode-lane counts, with a direct 2048-token kernel profile if needed.
- [x] Assess mixed CUDA Graph feasibility under startup capture, exact-batch topology,
      numerical/state and address-lifetime contracts. Implement and qualify a coherent graph
      route if the evidence supports a net Engine benefit; otherwise pursue the next supported
      mixed-round bottleneck.
- [x] Retain only reproducible qualified improvements and update the mixed-score table using
      matched exclusive baselines and adjacent controls, including throughput, decode progress,
      TTFT and uncertainty. Report a supported limit finding if no credible candidate wins.

Read-only subagent review precedes every GPU checkpoint. A4 verify remains excluded, and no
commit is requested by this goal.

Six successful eager Nsight Systems captures cover P=8192, R=3, D=1/3 and owner widths
0/1024/2048. Complete mixed rounds are attributed through runtime launch correlation and the
worker's CUDA event synchronization boundaries, including proposal, verify tail and owner context
append. At W=1024, mean full GPU spans are 86.67/88.51 ms for D=1/3, with 2.65/2.98 ms of
gaps; at W=2048 they are 153.59/155.87 ms, with 2.65/2.95 ms of gaps. Host-late submissions
account for less than 0.005 ms per round. These are profiled eager observations, not graph-enabled
score measurements or a promised graph gain. Evidence is in
`profiles/bench/mixed-graph-20261007/mixed-profile-summary.json` and the corresponding Nsight
SQLite analyses.

Full mixed graph capture would require stable owner token/control staging, device-selected owner
state and KV views, dynamic cursor controls and startup inventory for supported owner geometries
and exact decode batches. The current callback uses request-owned host token addresses and
host-selected cursor/state values. Across the complete owner window, measured GPU gaps occupy
3.29–3.51% at width 1024 and 1.81–2.01% at width 2048; host-late submissions occupy less than
0.17%. These are optimistic bounds on observed idle-time recovery, not bounds on every possible
graph benefit. No graph route or graph Engine A/B was implemented.

The next supported bottleneck was A8 output projection underfill. Same-kernel Nsight Compute
profiles at N=5120, K=6144/17408 and T=18/36 admitted narrower row partitioning through
`tools.kdev`. Two N=16 tile candidates passed independent Linear/LinearAdd numerical oracles
and focused memcheck/racecheck with zero errors or race warnings. Each lost at every changed
public-Op point across two cold ABBA pairs: the first by 6.34–84.54%, the second by
14.23–36.36%. Both were removed, without advancing losing Ops to Engine measurements.

This is a supported limit finding for the investigated paths, not proof of absolute optimality.
No production kernel or mixed graph is retained from this follow-up. The preceding qualified
score/F/Q/TTFT table remains the measured frontier, with no reproducible active-mixing advantage
over adjacent controls. The full report is in `docs/performance.md`, section "Actual mixed-round
launch profile (2026-10-07)". Evidence is under `profiles/bench/mixed-graph-20261007/`,
`profiles/nsys/mixed-graph-20261007/` and `profiles/ncu/mixed-graph-20261007/`.
The final restored fast suite has 119 passes, two artifact-dependent load-plan skips and zero
failures; the benchmark targets build successfully. Clang-tidy reports zero diagnostics across
four affected translation units, and affected-file formatting/lint hooks and `git diff --check`
pass. The read-only review found no material claim blocker. Scheduling selection remains joint work.

### First-chunk mixed admission (complete, 2026-10-07)

This goal tests the remaining admission difference with AMD while preserving A8 verify and
leaving scheduling selection joint. AMD batches owner and verify columns through aggregate
projections; the 5090 uses separate A4 owner and A8 verify projections. Their score definitions,
width meanings, artifact formats and draft settings also differ, so the AMD score is not a
hardware-independent prediction.

- [x] Attribute first-chunk admission, steady mixed rounds and finalization; establish matched
      score/control definitions at owner widths 1024/2048 and decode lanes 1/3/5.
- [x] Test eligible first-chunk mixing, including short/exact-width/partial-tail prompts,
      independent of the cadence between subsequent chunks. Preserve state/publication and
      cancellation semantics; retain the change only with a qualified Engine benefit.
- [x] Compare paired Engine score/control differences, prefill throughput, accepted decode
      progress, TTFT and maximum decode gap; report uncertainty and a supported limit if it loses.
- [x] Complete relevant real-model, numerical, sanitizer, fast and lint checks; update the active
      performance reference. Do not repeat rejected fusion unchanged or commit without a request.

Every GPU checkpoint has read-only review and holds `/tmp/ninfer-gpu-queue.lock`; no source or
build edits occur during a GPU measurement chain. Evidence is under
`profiles/bench/mixed-admission-20261007/`.

Unconditional first-chunk mixing was rejected and its four runtime/test changes removed.
Long-owner mean scores remain below adjacent controls, with TTFT 1.7–2.8% higher; short-owner
TTFT rises 32.8–48.7% and score falls. D3 eager profiled completion gaps improve from
148.372 → 94.061 ms at width 1024 and 274.398 → 162.540 ms at width 2048. This establishes
a decode-service/owner-latency tradeoff, not a general improvement or proof of absolute
optimality. No threshold or scheduling policy was added. All campaign GPU ledgers are clean.
The restored benchmark and real mixed-forward tests pass; the fast suite has 119 passes,
two artifact-dependent skips and zero failures. No candidate kernel math changed, so the
existing retained-Op numerical and sanitizer qualification applies. The active report is
in `docs/performance.md`, section "First-chunk admission experiment (2026-10-07)".
Clang-tidy reports zero diagnostics across four affected translation units. Affected-file
formatting/lint hooks and `git diff --check` pass; final read-only review found no material
claim or ledger blocker. Scheduling selection remains joint work; no commit was made.

### Remaining score-advantage investigations (scoped Engine qualification complete)

Investigate all concrete remaining ideas for a reproducible score above one and above matched
prefill-first controls. Preserve A8 verify, NVFP4 KV and the explicit DFlash2 artifact. This is
an evidence-backed finite candidate investigation; scheduling policy selection remains joint.

- [x] Inventory prior rejected candidates and remaining profiler-supported bottlenecks; establish
      concrete candidate gate cards before selecting experiments.
- [x] Investigate materially different owner/verify projection amortization. Do not repeat
      rejected dual-precision fusion unchanged; require classifier admission before CUDA work.
- [x] Investigate remaining operator or mixed-execution opportunities at 128–4096 tokens,
      explicitly including 1024 and 2048. Qualify viable candidates and remove production losers.
- [x] Assess workload-dependent execution choices against existing width/cadence evidence;
      measure material gaps without selecting a scheduling policy.
- [x] Measure qualified improvements with paired Engine runs, matched exclusive baselines and
      adjacent controls. Report score/control difference, prefill and accepted decode throughput,
      TTFT, decode gaps and uncertainty; distinguish tested limits from absolute optimality.
- [x] Complete relevant qualification and read-only review, update the active performance
      reference and close every concrete candidate with evidence. Do not commit without a request.

Concrete candidate inventory (prior results are in `docs/performance.md`):

| Idea | Material difference from prior experiments | Next evidence / disposition |
| --- | --- | --- |
| Separate-stream A4 owner / A8 verify | Independent qualified kernels and scratch; no dual-precision fused kernel | Retained qualified public pairs and per-layer overlap; final merged-route W4K beats one and adjacent controls in both observations at all D1–5 |
| Projection proximity / weight-cache reuse | Match corresponding projections instead of interleaving whole layers | Full post-mixer proximity tested: D3/D5 gains at W1K are 0.67%/0.18%, W2K 0.05%/−0.05%; separate streams are stronger |
| SwiGLU four-stage M128 pipeline | Prior M128 candidate used two stages | Rejected after oracle/codec passes: public Op slower by 9.28–24.30% at T512–4096; M256/S4 exceeds the shared-memory limit |
| SwiGLU M256 with four N warp partitions | Prior eight-M-warp candidate changed token partitioning | Rejected after oracle/codec passes: public Op slower by 34.02–62.29% at T512–4096; legal geometry spills |
| A8 weight expansion / scale reuse | Prior experiments partitioned output rows without changing expansion reuse | Existing code already reuses expansion/scales outside token loops; new arithmetic family refused by the DRAM-bound classifier at real verify widths |
| GDN state-passing issue / barriers | Prior strip32 candidate reduced input loads and lost | Retained after fresh recurrence, memcheck/racecheck and Engine qualification: public Op latency is 2.61%/2.73% lower at 2K/4K; W2K Engine throughput improves 0.268% and TTFT falls 0.270%; W1K/W4K Engine differences are inconclusive |
| Workload-dependent execution | Existing score coverage omits intermediate owner lengths and some mixed context cases | Completed 20 supported points: D1–5 at W1K/2K/4K, D3 W256/512 and three selected prompt/context variations; W128 is unsupported for mixed-forward; no policy selected |
| Disjoint SM resources | Ordinary streams can contend for registers/CTA residency | Rejected: both 146/24 and 162/8-SM post-mixer pairs pass parity/memcheck/racecheck but lose to ordinary streams at 1K/4K; no production partition policy introduced |
| Owner register residency | Owner SwiGLU CTAs use nearly the SM register budget; zero-shared-memory verify recurrence may co-reside with a lower legal allocation | Admitted after the required matching CUTLASS/current-kernel gap: comparator passes exact oracle, memcheck and racecheck; CUTLASS is 8.97%/8.21%/5.83% faster at 1K/2K/4K. 128-register candidate passes body/codec/memcheck/racecheck but slows concurrent post-mixer pairs by 27.95–35.70%; actual masked-tree history pair also loses by 33.61–37.43%; candidate rejected with clean oracle/memcheck/racecheck evidence |
| Verify stream priority | Queued eligible verify kernels are delayed while owner CTAs occupy resources | Rejected: highest-priority verify passes parity/memcheck/racecheck but increases pair latency by 1.30%/2.39%/2.11% at 1K/2K/4K; no Engine priority change |
| Mixed graph / launch opportunity | Prior eager attribution assessed W1024/2048 | Fresh merged-route eager traces at W1K/2K/4K show host-late gaps of 0.264%/0.104%/0.126%; no material identified host-launch opportunity for a mixed-graph rewrite |

Public-Op checkpoint: all 30 Linear pairs, 30 LinearAdd pairs and 15 full post-mixer pairs
completed with clean exclusive-GPU ledgers and exact schedule parity. Separate independent
mathematical Op oracle tests pass; focused
memcheck and racecheck report zero findings. Each full post-mixer point has four median observations
per schedule, with both serial orders as controls and exact serial-output byte checks. Separate
streams improve the best full post-mixer pair time by 6.19–7.98% at owner width 1024,
4.15–4.82% at 2048 and 3.10–3.30% at 4096 across verify widths 6/18/30. These are synthetic
public-Op fixtures, not Engine score claims. Evidence is under
`profiles/bench/mixed-remaining-20261009/`, including `post-mixer-summary.json`.
The production-path candidate uses startup-owned owner compute resources, disjoint bounded planned
workspace regions, matching-layer barriers and a full tail join before verify publication.

Provisional graph-enabled D3 Engine ABBA pilot (owner 8192, three jobs, initial decode
context 512, route-specific decode calibration and adjacent controls, time-interpolated
matched exclusive prefill baseline):

| Mixed owner width | Serial score mean | Stream score mean / observed range | Stream advantage over its control |
| --- | --- | --- | --- |
| 1024 | 0.92834 | 0.98397 / 0.97973–0.98822 | −0.01812 to −0.01137 |
| 2048 | 0.97107 | 1.00021 / 0.99651–1.00392 | +0.00009 to +0.00596 |
| 4096 | 1.00233 | 1.01383 / 1.01217–1.01549 | +0.00969 to +0.01162 |

Two observations per route are observed ranges, not confidence intervals or lane-generality
claims. Both 4K observations exceed one and their controls; 2K straddles one, and 1K remains
below one. Stream owner TTFT means improve by 5.67%, 2.80% and 1.13% respectively. Accepted
decode tokens/s are reported separately in `layer-overlap-engine-abba/analysis.json`.
The full real fixture and full memcheck pass; broad Engine racechecks exceeded the container
host-memory limit. Focused lifetime normal/memcheck pass; its unthrottled racecheck also OOM-killed. The bounded all-kernel recovery was stopped after about an hour without a terminal result;
it is incomplete and supplies no pass. The October 15 reviewed qualification uses focused mathematical-kernel racechecks plus full
Engine parity/memcheck and cancellation/drain checks for the changed risks; broad Engine
racecheck remains explicitly incomplete.
Persistent bounded-arena peak accounting and checked partial event-creation cleanup are
implemented; the warning-as-error build and changed-line clang-tidy pass. The October 15 final-build
qualification below supersedes that checkpoint status. These pre-pull timings are historical
pilot evidence; the merged-route frontier is in the October 15 performance reference.

Whole-Engine eager Nsight Systems attribution at these widths shows 14.67%/8.16%/3.13%
owner/verify kernel overlap. Eligible queued verify launch delays covered by owner kernels
are 12.79%/16.94%/14.42% of the captured window, an inference of resource contention rather
than guaranteed recoverable headroom. Host-late portions of GPU gaps are only
0.185%/0.135%/0.128%; this does not justify a mixed graph rewrite for the outstanding 1K
score deficit. Profiling diagnostics are separate from graph-enabled timing claims.

Prior unchanged losers remain excluded: dual-precision projection fusion, first-chunk admission,
A8 N16 row partitioning, ordinary TMA PDL, residual M128/S2 and S3, BF16 attention-input warp
tiles, SwiGLU M128/S2 and eight-M-warp M256, GDN strip32, and all-width short-query attention.
Selective query-width-at-most-256 attention remains qualified and retained.

Every GPU checkpoint requires read-only review, `/tmp/ninfer-gpu-queue.lock` and the exclusive
GPU guard. Preserve unrelated changes; keep source/build inputs fixed during measurement chains.
The sandbox startup blocker prevented experiments on October 8. Repository and container access
were restored on October 9; no experiments from the blocked period are claimed as evidence.

All four queued checkpoints are terminal, and their processes have exited. No foreign GPU
workload was observed in any checkpoint ledger. The GDN candidate passes all four real-model
parity cases, focused lifetime normal/memcheck and kernel-trace coverage. Its filtered Engine
racecheck is incomplete: exit 9 after 618 seconds, with the kernel journal confirming a container
memory-cgroup OOM kill of target PID 2896316 at approximately 95 GiB anonymous RSS. Its zero
hazards displayed summary is not a pass. No automatic rerun was submitted.
The accounted-workspace production build passes all four real-model parity cases and focused
lifetime normal/memcheck. Priority public post-mixer qualification, memcheck and racecheck pass;
priority timing loses and closes that candidate. Matching CUTLASS qualification, exact FP64
oracle, memcheck and racecheck pass, with four median observations per route at each width.
The comparator includes quantization on both routes and measures BF16 Linear with constant
represented weights/scales; it does not measure fused SwiGLU or general scale conversion.
Summaries are `priority-post-mixer-summary.json` and `cutlass-gap-summary.json` under the current
evidence root. CPU warning-as-error builds, changed-line clang-tidy and affected formatting/lint
checks pass. The register-cap candidate, separate GDN Engine ABBA and final supported frontier are resolved
by the October 15 results below. The attention follow-up passes its oracle, memcheck and racecheck. Final read-only review
finds no material claim, ownership, qualification or scope blocker. All investigation items
are measured; the scoped Engine racecheck result is recorded below. Phase 3 remains
deferred joint work. No commit is authorized.

### October 15 resume

The requested latest integration uses `origin/experimental` at `9edf680b`; the tracked perf
branch had no incoming commits. A no-commit merge preserves the branch's broader A4 contracts
and mixed-stream execution while integrating GDN history, logprobs, token grammar, disposable
cache writes and the masked NVFP4 attention-tail fix. The original uncommitted work is preserved
in the named stash and the pre-pull binary patch under `profiles/bench/mixed-resume-20261015/`.
The merge is not committed or pushed. The warning-clean full build and 241-TU clang-tidy pass. The fast selection has zero failures out of 126 selected tests (two missing-artifact load-plan skips); all five selected real-artifact mixed/cache/constraint/DFlash/logprob tests pass.

The bounded register-cap candidate preserves M256/N128/S3 and the BF16 production route.
The pulled history-record kernels use 80–123 registers across their dense/masked chain/tree
specializations, all with 128 threads and zero shared memory. Their largest rounded CTA
allocation is 16,384 registers, so the candidate uses a 128-register owner launch cap, producer
24 and consumer 176: the requested 48,128 registers fit the 49,152 launch allocation and leave
16,384 registers on the SM. The earlier 144-register compile could not satisfy this pulled
runtime constraint and was not run. Native compilation confirms 128 registers, a 104-byte
stack, 108-byte spill stores and 116-byte spill loads; the unchanged BF16 specialization remains
168 registers without spills. The extra spill traffic is a measured implementation cost to
assess in the bounded public-Op checkpoint, not evidence of a speedup. Independent FP64 body, exact A4 codec, schedule parity, focused memcheck and racecheck pass. External ABBA timings at 1K/2K/4K show owner post-mixer latency increases of 36.73%/38.01%/37.50%, and concurrent pair increases of 27.95%/33.63%/35.70%. The cap loses this measured scope and is not selected for production. A separate reviewed public pair uses the actual masked-tree history-record kernel with B3/T6
and zero retained history. Raw/provisional outputs and immutable inputs agree across serial and
concurrent orders; memcheck and racecheck pass. Nsight confirms the intended specialization and
10.624/11.200 us of history/SwiGLU interval overlap in the two concurrent orders. Unprofiled
ABBA pair latency nevertheless increases 33.61%/36.67%/37.43% at 1K/2K/4K. This closes the
bounded register-cap candidate as a loser; no production cap is installed.

Fresh GDN qualification uses the pulled runtime. The independent full recurrence oracle,
focused raw64/raw129/normalized257 memcheck/racecheck, all four mixed-Engine parity cases,
full Engine memcheck and focused cancellation/drain normal/memcheck pass. Fresh public-Op
ABBA latency is 2.61%/2.73% lower at 2K/4K; overlapping 1K observations are inconclusive.
All 28 incremental Engine calls have clean GPU ledgers. The candidate is retained for its
qualified Op improvement and small W2048 Engine benefit (+0.268% throughput, -0.270% TTFT).
W1024/W4096 Engine ranges overlap; no general Engine or incremental score gain is claimed.

The tested header is copied byte-identically into production. The final 190-target warning-clean
build, affected CUDA tidy (zero diagnostics), formatting/lint, fast selection (124 passes,
two missing-artifact load-plan skips) and all four rebuilt mixed-Engine parity cases pass.

The supported frontier is complete: 20 settings and 120 clean valid-call ledgers. Ten settings
exceed one and their adjacent controls in both observations. All five main W4K points qualify;
W2K qualifies at D3–5 in this sweep, while no W1K point qualifies in both observations. The
separate incremental W2K comparison straddled zero control advantage, so no universal
scheduling threshold is inferred. The W128 mixed-forward attempt was rejected before GPU
launch: the public contract requires multiples of 256. Its partial controls and rejection are
preserved separately and excluded from the table; 128-token prefill-Op evidence is separate.
The original 15 main points were preserved while the remaining supported points completed
with the same frozen binaries.

Three fresh eager traces report maximum completion gaps of 142.430/266.131/534.159 ms at
W1K/2K/4K. Host-late gaps account for only 0.104–0.264% of their windows. These are diagnostic
round completions, not graph-enabled maximum or token-publication latency. The full table,
uncertainty and candidate dispositions are in `docs/performance.md`, section "Qualified
mixed-execution frontier on the merged runtime (2026-10-15)".

The merged masked-tail attention follow-up is required because the fast selection excludes
long GQA fixtures. Its independent oracle, memcheck and racecheck without launch skips pass
with zero errors or hazards. The permanent regression covers masked W129/W257 and poisoned graph replays in both
schedules. W257 uses base256 (visible bound333), satisfying the resource-envelope bound while
remaining non-split. The initial invalid base63 fixture rejection is preserved separately and
is not kernel failure evidence.

Read-only review confirms that focused kernel oracles and sanitizers address the changed
async-copy/shared-memory risks; Engine parity, full memcheck and cancellation/drain checks
address integration. Earlier all-kernel Engine racechecks did not complete and are not passes; the
user-narrowed follow-up below passes. The combined `git diff HEAD --check` flags only blank context lines in the imported
unified patch `cmake/xgrammar-json-constraints.patch`; its syntax is preserved under the
existing patch exclusions in the formatting hooks. The source/document working diff check
passes. No merge-to-master qualification is claimed.

### Scoped Engine racecheck completion

The user requested qualification in a fraction of the all-kernel runtime. This supersedes
all-kernel instrumentation for this follow-up. Earlier broad attempts remain incomplete;
no all-kernel Engine racecheck or global inter-kernel race-detection claim is made.

- [x] Complete the real Engine fixture with changed-risk GDN/history and NVFP4 attention
      racecheck instrumentation, owner parity, two mixed rounds, five active decode lanes,
      cancellation/drain and destruction. Retain every workload assertion.
- [x] Complete terminal evidence review and reconcile the active performance reference.

Final read-only review accepts the scoped evidence and closure. The scoped run passes in
1,252.85 seconds (20 minutes 53 seconds), with zero hazards, errors or warnings, exit 0, clean exclusivity/monitoring ledgers and no resource abort.
The fixture uses the explicit DFlash2 artifact, NVFP4 KV, A8 verification, optimized k5
proposal head, eager execution, D5/W512 and a 1,100-token owner against five 96-token
background prompts. The tool uses eight workers, lazy loading and
`--force-synchronization-limit 32`; periodic synchronization changes launch scheduling.
Peak process RSS is 31.87 GiB. The GPU lock is released and no work is queued.

A current normal Nsight trace confirms state-passing, history-record/commit/count and
NVFP4 prefill attention/fill/page-fill/merge dispatch. The filter covers every launch of
those named families. GEMMs and other kernels outside those families are not instrumented;
the existing independent Op oracles, concurrent Op sanitizers, full Engine numerical and
memcheck evidence remain applicable. Normal fixture execution passes in 9.70 seconds.

The long readiness stall was a fixture defect: only its polling deadline was extended,
leaving the actual Engine queue budget at 30 seconds. A 37-second background prefill let
four pending requests expire. The corrected fixture sets a 12-hour sanitizer queue budget
and resolves request outcomes when published counters suggest quiescence, instead of
polling blindly. Production queue behavior is unchanged. An 8.61-second public-Engine
control reproduces four typed admission timeouts. In the corrected sanitizer run, all five
lanes become ready, both mixed rounds commit, owner output matches its solo reference,
and cancellation/drain plus Engine destruction complete.

Evidence: `profiles/bench/engine-race-resume-20261016/scoped-engine-queue-budget/`,
including `summary.json`, recorded commands, terminal logs and resource/exclusivity ledgers.
Interrupted broad and expired-admission attempts are retained as incomplete evidence.
No commit, push or merge-to-master qualification is claimed.

### Gate 5 caveat

Gate 5 also needs `qwen3_8_27b_nvfp4.ninfer` and the MTP variant. Only the DFlash artifact is on
this machine, so the resumed full `--real` run skipped eight Engine tests. Either place the missing artifacts
or state the skips as a limitation.

## Environment and gotchas

- **Builder container.** Use `ninfer-builder-sandbox`. It mounts this checkout at `/src`, with the
  build tree in `build/` mounted at `/build`. The default `ninfer-builder` mounts the other
  checkout (`/path/to/ninfer`). Always set:
  - `NINFER_DEV_CONTAINER=ninfer-builder-sandbox` for `scripts/run-unit-tests.sh` and
    `scripts/run-clang-tidy.py`;
  - `NINFER_DEV_JOBS=8`.
- **Real tests.** Set
  `NINFER_QWEN3_8_27B_NVFP4_DFLASH_WEIGHTS=/path/to/models/qwen3.8-nvfp4-flash2-nvfp4-bf16codebook-from-bf16/qwen3_8_27b_nvfp4_dflash_nvfp4.ninfer`.
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

**Historical control-normalized measurements from 2026-10-04.** The final exclusive-prefill
normalization and table are in the 2026-10-06 performance reference above. Setup:
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

**Historical serial mixed-round bottleneck (2026-10-04):**
- The verify W4A8 GEMMs take 10.7 of the 13.6 ms per round.
- They are latency/issue-bound at about 1.2 TB/s, and are no faster with the weights already in L2.
- Serial layer interleaving did not amortize the dominant verify projection cost.
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

After the remaining-investigation ledger closes, subsequent joint work will select and implement the
policy with the user, validate real request behavior, and update the active
performance and architecture references. The user authorized pulling experimental and preparing the W4096/rounds1 default on October 18.
Commit and push remain pending explicit approval; size-dependent scheduling remains deferred.


## Selected default integration (2026-10-18)

- [x] Fetch and integrate experimental through `edaad03a`, preserving retained mixed work.
- [x] Select Engine/server W4096, rounds1; preserve explicit prefill-first diagnostics.
- [x] Document ratio controls and point to the measured score/decode tables.
- [x] Complete focused integration/default checks and read-only review.
- [x] Prepare the concise commit-message draft without committing or pushing.

The latest-runtime D3/W4096 confirmation scores 1.01659 (two observations, both above 1
and their matched prefill-first controls). The implementation, qualification, and score
checkpoint passed read-only review. Evidence is in `profiles/bench/default-mixed-20261018/`;
the concise draft is `commit-message.txt` there. The GPU lock is released. Commit and push
still await user approval.
