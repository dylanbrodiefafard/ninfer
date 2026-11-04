# R9700 performance status

This document covers the native HIP/gfx1201 implementation. Measurements from the retired backend are not comparable.

Historical `profiles/` paths below refer to local evidence, removed from the current tracked tree for publication. The original files remain in a private recovery bundle and a verified maintainer archive; `bench/results/historical-evidence.json` records the source commit, archive checksum and restore command. Paths in prose are examples. Published summaries, benchmark tools and fixtures remain in the repository; measured values and model artifacts are unchanged.

## Evidence levels

NInfer distinguishes four scopes:

1. raw kernel candidates;
2. complete semantic Ops;
3. runtime phases/rounds;
4. complete Engine inference at fixed concurrency.

A result is stated only at the scope directly measured. Kernel or Op timings do not establish tokens per second. Final production selection requires a complete model artifact, numerical quality guardrails, same-candidate graph/eager parity, resolved capacity, and end-to-end C=1..8 measurements on an otherwise idle R9700.

Benchmark schema 21 sums each request's active prepare, Vision and prefill service time across lanes. Aggregate prefill throughput divides all prompt tokens by that summed prefill service time. Batched decode retains the maximum shared lane duration; complete inference separately measures elapsed wall time. Service costs are not an additive wall-time decomposition. Schema 20 instead used the maximum lane prefill duration and overstated aggregate C>1 prefill throughput. Those concurrent prefill fields are ineligible for selection; retained C1, capacity, decode and independently measured whole-wall evidence remain valid in their respective scopes. The stopped attempt and valid raw results are retained in `profiles/bench/r9700-terminal-base-fp8-context-recovery-20260921/whole/closure.json` as historical evidence. The FP8 shared-context resource bridge and the benchmark-reporting bridge that re-admitted the hipBLASLt-era four-role hybrid evaluation artifact were retired with hipBLASLt (438f6a89): their bound runtime sources and qualifier no longer exist, and the Pareto, cutover, and DFlash-selection tools now reject any authority that binds either bridge.

## Platform

The latest measurements use:

| Component | Value |
|---|---|
| GPU | AMD Radeon AI PRO R9700 |
| architecture | `gfx1201`, wave32 |
| driver | `7.1.3.31500000` |
| kernel | `7.0.0-34-generic` |
| HIP | `7.15.26333` |
| compiler | AMD Clang 23 under `/opt/rocm/core-10.0` |
| build | Release, exact `gfx1201` code object |

The GPU must otherwise be idle. Kernel and service timings use HIP events around the claimed physical work after warmup; serving throughput below uses explicitly identified wall intervals. Operators are checked against independent exact/FP64 reference calculations before their timings are used.

### Device-count MTP packed attention (2026-11-04)

PR #3 routes non-tree MTP attention with device-selected active row counts through the
existing packed decode implementation. Its setup and merge already implement inactive-prefix
semantics; tree attention retains its existing routes. Startup MTP planning already covers
packed scratch, and each concrete call now requests that scratch consistently with dispatch.

A same-host public-Engine comparison used the selected `r9700-lq4` artifact on one gfx1201
R9700, HIP 7.15.26333 / AMD Clang 23, kernel 7.0.0-38-generic, and unchanged `auto` power mode.
Both builds ran the bench corpus with C1, MTP K3, Device Graphs, prefill chunk 2048,
`-pg '8192,128;64000,128' --warmup 1 -r 3 --retain-token-ids` and the fixed production cache.
The rates below are mean decode output tokens/s over three repetitions; prefill is excluded.

| Prompt tokens | Before | Packed device-count route | Speedup |
|---:|---:|---:|---:|
| 8192 | 87.26 | 116.57 | 1.34x |
| 64000 | 31.56 | 108.84 | 3.45x |

All generated token IDs and speculative statistics matched exactly between builds in every
measured repetition. The target attention qualifier also checks the represented BF16-query,
FP8-key, signed-INT4-value/FP16-scale formula against its independent FP64 oracle. Captured
T1/T4/T8 panels at 64K pass with zero, partial, full and malformed device counts, poisoned
inactive positions, exact positive-zero tails, recovery and guarded output/scratch; the largest
observed full-panel absolute error was 1.47e-5. Device-count panels use the same existing
BF16-Q/FP16-probability numerical criterion as host-fixed packed panels.

The full attention qualifier passes device memcheck and initcheck. At C1/8K, eager and
Device Graph execution produce identical token IDs and speculative statistics. A separate
C8/8K graph run also matches the no-speculation control token-for-token across all eight lanes.

These measurements establish the C1 MTP change, not a new concurrency-wide throughput matrix.
Local commands and reports are retained under `profiles/pr3-review/`.

### README refresh and product defaults (2026-10-17)

CLI and server default to DFlash with the optimized proposal head and adaptive length up to seven drafts. `--spec none` disables speculation; `--no-lm-head-draft` uses the full head. `--draft-tokens N` overrides the maximum; `--no-adaptive-draft` fixes the length. Server shared KV capacity defaults to `auto`. P-less sampling defaults to temperature 1.5 unless overridden; the standard sampler retains the registered model/mode temperature 2.0. Focused CLI/server option and sampling-resolution tests cover these defaults and overrides.

Fresh measurements use the production `r9700-lq4` artifact on one R9700, HIP 7.15.26333, Release, power `auto`, fixed G16 FP8-K/INT4-V/FP16-scale cache, chunk 2048, Vision off, optimized proposal head and Compose's bulk host-argument graph environment. No clock, voltage or power policy was changed. The 65,536-token curated benchmark corpus is repeated to 262,144 tokens for the context ladder. Each point discards one warmup and reports the mean of three repetitions. RAM/SSD prefix tiers are off.

| C1 prompt tokens | Mean prefill service tok/s | Stddev tok/s | Mean trailing prefill estimate tok/s |
|---:|---:|---:|---:|
| 8,192 | 3,375.1 | 2.16 | 3,273.0 |
| 32,768 | 2,903.5 | 2.15 | 2,489.3 |
| 65,536 | 2,477.4 | 2.91 | 1,915.3 |
| 131,072 | 1,906.6 | 0.24 | 1,308.0 |
| 245,760 | 1,354.5 | 0.25 | 836.7 |

Prefill service excludes model load, input preparation and HTTP transport. The trailing estimate is the Engine's prorated last-second service window, not a separate incremental prompt experiment. The benchmark uses `-p 8192,32768,65536,131072,245760`, `--max-ctx 262144 --kv-capacity auto --prefill-chunk 2048`, and `--spec dflash --draft-tokens 7 --adaptive-draft --lm-head-draft --warmup 1 -r 3`.

The following historical generation measurements use the same adaptive setup with a greedy target and 256 post-prefill tokens per lane, after an untimed prefix-reuse seed (`--isolate-prompt-decode`). They exclude prefill and model load; prompt content and sampling change DFlash acceptance and throughput. The C1 8K/32K/64K cells measured 192.7/207.1/223.9 output tok/s. Adaptive selection retained K7 in all measured C1 rounds. Acceptance also changes with context on this corpus; these rates do not establish a general context-scaling benefit. C6 at 8K measured 672.6 aggregate output tok/s (stddev 9.88), with 18/38/26/714 measured rounds at K3/K5/K6/K7. This is a phase-throughput baseline, not an optimization A/B. The workload matrix below replaces these single-corpus figures in the README.

The refresh also reproduced C1 full-context graph preparation consuming 72 MiB against a stale 56-MiB allowance. The calibrated C1 budget is now 72 MiB; the startup guard remains. Other native-ceiling graph budgets are unchanged. Reports and commands are under `profiles/bench/readme-refresh-20261017/`.

| C | Graph budget / observed, MiB | Resolved shared KV tokens |
|---:|---:|---:|
| 1 | 72 / 72 | 262,144 |
| 2 | 56 / 56 | 524,288 |
| 3 | 60 / 60 | 633,024 |
| 4 | 60 / 60 | 624,896 |
| 5 | 62 / 62 | 616,832 |
| 6 | 62 / 62 | 608,704 |
| 7 | 62 / 62 | 600,576 |
| 8 | 62 / 62 | 592,512 |

These native-ceiling (`--max-ctx 262144`) checks cover startup and short decode at C1..C8; the table describes allocated capacity, not a new all-lane near-capacity qualification. C1 additionally completes the prefill ladder through 245,760 tokens. The real CLI smoke omits temperature, draft count and adaptive flags and confirms p-less temperature 1.5, a seven-draft maximum and 32 generated tokens. Three focused host tests and affected clang-tidy checks pass. This refresh does not change weights or kernel arithmetic.


### Workload-dependent answer generation (2026-10-17)

The README now uses real chat workloads at the product's p-less temperature 1.5, replacing the single greedy-corpus decode number. Measurements use the production LQ4 artifact, one R9700/gfx1201, Release, HIP 7.15.26333 under ROCm 10, Device Graphs, power `auto`, adaptive DFlash up to K7, optimized proposal head, chunk 2048 and Compose's graph environment. Each process has a 65,536-token per-request ceiling and automatic shared KV sizing; RAM/SSD prefix tiers and Vision are off.

The committed runner is `tools/bench/run_readme_workloads.py`, with workload inputs in `tools/bench/fixtures/readme_workloads.json`. Story, code and math use existing CLI examples; math has thinking enabled. Code explicitly requests code in the response with no tools. Document summaries use the named local architecture, cache and model documentation, truncated near 8K and 32K with the local tokenizer. Prompt counts include chat-template tokens.

Each lane has a distinct prompt marker and seed. An untimed one-token request first seeds every lane's prefix; a concurrent wave then generates up to 512 output tokens per lane, with model stop tokens enabled. All measured requests reached 512 tokens. There is one warmup wave and three measured waves per cell, using different fixed seeds. Reasoning tokens count toward output and throughput.

The rate is committed decode tokens divided by wall time from complete 100-ms server throughput intervals with zero prefill, exactly C active decode lanes, and exactly C rows in every decode round. It includes host scheduling between rounds and excludes startup, prefill, and intervals after any lane finishes. There is no fallback to prefill-inclusive makespan or summed per-lane rates. This differs from the earlier Engine service-time baseline. C6 is aggregate; dividing by six gives the average rate per active request in the selected intervals.

| Workload | Prompt tokens | C1 mean ± sample stddev tok/s | C6 aggregate mean ± sample stddev tok/s | C1 / C6 draft acceptance |
|---|---:|---:|---:|---:|
| English story | 146 | 88.3 ± 1.1 | 355.9 ± 3.6 | 23.1% / 23.1% |
| Python package | 171 | 135.1 ± 22.1 | 534.6 ± 24.4 | 43.1% / 43.4% |
| Math reasoning | 346 | 126.3 ± 8.8 | 491.1 ± 7.8 | 40.2% / 37.1% |
| Document summary 8K | 8,192 | 92.6 ± 2.2 | 340.4 ± 12.4 | 26.0% / 29.1% |
| Document summary 32K | 32,779 | 97.4 ± 5.6 | 299.7 ± 7.1 | 31.0% / 41.7% |

These samples do not isolate context-length effects: content, sampling, live K and draft acceptance all influence speed. The 32K C1 summary sample has higher acceptance than 8K; C6 slows at the longer context despite higher acceptance. Task quality is not scored by this throughput benchmark.

The public machine-readable summary is `bench/results/readme-workloads-20261017.json`; raw reports below are local evidence, not files required to reproduce the benchmark.

Run the command documented in `tools/bench/README.md` with the explicit local LQ4 artifact and BF16 tokenizer. The reports are under `profiles/bench/readme-workloads-20261017-run3/`; the admitted code cells use the corrected prompt from `profiles/bench/readme-workloads-code-20261017/`, produced by adding `--workload "Python package"`. The original code cells requested tools and are excluded. Inputs, output text, per-request counters, startup metadata and raw periodic timings are retained with the reports.


### Windowed KV reservations at C8 (2026-10-22)

A matched full-workload comparison on one R9700 measured automatic shared KV capacity of
509,504 tokens with 12 GiB of RAM backing, adaptive DFlash up to seven drafts and p-less
sampling at temperature 1.5. Both runs submitted the same eight distinct 60,000-token raw
prompts and generated exactly 32,768 tokens per request. Model stop tokens, prefix writes and
generation recovery were disabled. The windowed run used output reservations of at most 8,192
tokens; the control retained full-lifetime GPU reservations. Neither run used a reasoning cap.

| Measurement | Full reservation | Windowed reservation |
|---|---:|---:|
| Eight-request makespan, s | 1,141.620 | 891.936 |
| Committed decode tokens / makespan, tok/s | 229.618 | 293.895 |
| Actual mean decode batch, rows per round | 2.647 | 4.051 |
| Mean time to first token, s | 207.360 | 108.960 |
| Live RAM suspensions / resumes | 0 / 0 | 3 / 3 |

Windowing improved whole-workload decode throughput by **28.0%** and reduced makespan by
**21.9%** for this pair. Both runs evaluated all 480,000 prompt tokens, committed 262,136 decode
tokens and returned 262,144 output tokens. All eight requests reached their original output
limit without cancellation or censoring, and final active/suspended ownership was zero. The
windowed run spent 1.064 s saving live state and 1.135 s restoring it in total. Makespan ends at
the final client completion timestamp, excluding observer polling delay.

An earlier fixed 300-second comparison showed no throughput gain while the windowed scheduler
performed more eager prefill. That interval ended with different amounts of queued work
completed; it remains a negative result for that bounded horizon. The complete-workload pair
measures the cost and benefit of finishing the same eight requests. This is one raw-token stress
workload with model stopping disabled, not a universal speedup, a useful-action benchmark or a
model-quality result. Evidence: `profiles/chunked-kv/throughput-completion-comparison.json`.


### Reasoning budgets and KV-window interaction (2026-10-22)

A private 25-case reasoning-stall corpus replayed at C1 with two paired seeds produced
50/50 schema-valid, nonduplicate tool-call candidates with a 4,096-token native reasoning
close, versus 49/50 without a cap. The remaining unlimited response reached its output
limit without an answer or tool call. Mean candidate latency by seed was 32.525/35.787 s
with the cap and 69.979/82.437 s without; 27/50 capped requests actually intervened.
The one discordant case completed in both arms at a third seed. A separate 16k replay
completed naturally, so it does not demonstrate rescue by that cap.

A single-seed C8 interaction check used eight original requests, 243,572 total prompt
tokens, an explicitly constrained 327,680-token pool, RAM backing, no prefix reuse,
recovery off and a common 300-second observation horizon. Both 4k-capped configurations
completed 8/8 candidates: windowed allocation in 152.654 s, full reservation in 184.586 s.
Unlimited full reservation completed 8/8 in 231.743 s; unlimited windowing completed
5/8 with three right-censored requests at 300 s. The latter generated 92,911 decode
tokens versus 38,803 for unlimited full reservation. Different sampled trajectories
prevent interpreting these candidate times as a clean scheduler-throughput comparison.
Every cell drained cleanly; none required live suspension in this corpus check.

These are schema-valid proposals; task correctness, actual tool execution and historical
failure reproduction are unverified. Controlled replays preserved thinking explicitly,
so they do not measure the preserve-thinking-off deployment profile. Healthy HTTP checks
passed 26/26 limiter/transport/tool/schema assertions, while six of eight strict answer
format checks failed identically with unlimited and nonbinding limits. A lower 4k cap
has stronger latency evidence here than 16k, without establishing universal answer quality.
NInfer's default remains unlimited; the local deployment selects 16k. Generic evidence:
`profiles/reasoning-limit/full-close4k-aggregate.json`,
`profiles/reasoning-limit/c8-interaction-aggregate.json` and
`profiles/reasoning-limit/http-close-aggregate.json`. Private corpus contents stay outside
this repository.


### Decode optimization profile at 512 through 64K context (2026-10-16)

Fresh characterization of `6f069b6c`, the production `r9700-lq4` artifact, fixed G16 FP8-K/INT4-V/FP16-scale cache, chunk 2048, and the Compose bulk host-argument graph environment. Each unprofiled point discards one repetition and measures three greedy corpus P+G128 requests under `auto`. These are diagnostic baselines, not an optimization A/B; some points have a slow repetition, with the full ranges retained in the report. No clock, voltage or power-policy setting was changed.

| C1 prompt tokens | Plain decode median ms/token | DFlash K7 median ms/round, including fallback | Plain attention kernel ms/graph | DFlash attention kernel ms/graph |
|---:|---:|---:|---:|---:|
| 512 | 27.997 | 29.293 | 0.368 | 0.392 |
| 8,192 | 28.480 | 30.211 | 0.766 | 1.034 |
| 32,768 | 30.045 | 32.067 | 2.316 | 2.835 |
| 65,536 | 31.618 | 33.998 | 3.932 | 4.742 |

The first two timing columns are unprofiled. Attention columns are consumer-plus-merge duration sums from matching P+G128 traces and support attribution only. Attention rises from about 1.4% to 13.3% of plain kernel time and 14.3% of DFlash kernel time. C1 target projection cost stays nearly constant: plain LQ4 T1 projections take about 21.1 ms/graph; DFlash persistent target phases take about 25.1 ms/graph and its Q4 drafter projections about 2.4 ms/graph. The persistent bucket includes several Ops; this trace does not resolve its internal phases or establish current physical HBM saturation. Output throughput across contexts also changes with acceptance and is not an isolated measure of context overhead.

The C6/P8192/G128 compact decode point measures 572.6 aggregate output tok/s median on this greedy corpus. Its matching trace contains 25 B6, two B5 and one B3 graphs. Across those graphs, GDN replay folding takes 2.96 ms and recurrence recording 2.51 ms per graph; all GDN state/front work totals 6.02 ms, about 13.6% of kernel duration. The B6 class alone totals 44.83 ms of kernel duration, including 6.08 ms GDN state/front and 4.94 ms attention. These are kernel sums, not unprofiled physical round timings or application sampling throughput.

Candidates identified by this initial profile (follow-up decisions below):

- **Batched GDN state traffic:** reconstruct committed recurrent state at the next record's initial-state load to avoid another state read. Preserve early convolution-state publication, FP32 state evolution and old/new replay-record lifetimes. Merely moving the fold into the front removes no state traffic. This needs a new complete-boundary oracle and timing decision.
- **Long-context attention access:** improve the physical K/V access pattern or remove repeated traffic while preserving the three planes and QK/PV arithmetic. The logical Text-cache minimum is 26,624 bytes per context token, or 1.745 GB at 64K. At the previously measured 636 GB/s read probe, an ideal single pass alone costs 2.74 ms, before partials, compute, barriers and read amplification. This is an optimistic traffic floor, not a new bandwidth measurement or an achievable latency. Prior rejected rings, vector-load substitutions and chunk-count sweeps need a new mechanism before repetition.
- **Plain T1 persistent hosting:** `lq4_t1_kernel` is absent from the persistent registry; about 245 invocations/token remain separate launches and interrupt hosted producer/projection runs. Hosting the exact T1 body could reduce boundaries without reducing weight traffic. Register/occupancy and streaming effects still require qualification and whole-decode timing; the earlier failed B>1 lowering must not be treated as a promoted solution.

All 27 baseline repetitions repeat their tokens exactly within each workload, and all nine matching-geometry traces reproduce baseline tokens exactly. Initial G32 traces changed capacity and short-context trajectories; they are preliminary diagnostics, not the final attribution. Known dependent nodes have overlapping timestamps in the ROCm graph traces. The classic control also retains such overlap for plain decode, while its summed kernel time differs from bulk by less than 0.05% at both short C1 points, with exact tokens. The cause is not established. Replay dispatch counts are reconciled by graph/node occurrence and dispatch order; no causal gap, idle-fraction or stall claim is made from those timestamps. This initial study changed no kernel or production route. It did not cover mixed rounds, tools, HTTP, adaptive p-less acceptance or concurrent 64K contexts.

Evidence, exact commands, ranges, per-symbol/grid attribution and control checks: `profiles/bench/r9700-decode-opportunities-20261016/report.txt`, `analysis.json`, `manifest.json`, `matched-manifest.json`, and `trace-controls.json`.

### Decode profile follow-up (2026-10-16)

The follow-up retains the exact T1 LQ4 arithmetic in a shared CTA body and hosts its Store, residual and SiLU-pair entries in the existing persistent decode dispatcher. The C1/P8192/G128 ordinary-decode A/B uses six balanced processes, each with one discarded and three measured repetitions. Paired candidate/control ratios are 0.91654, 0.91587 and 0.91631; the median reduction is **8.37%** (3.6454 to 3.3398 seconds per 128 tokens). Generated tokens are exact. The independent FP64 Linear qualifier and all 23 exact persistent-lowering scenarios pass. The persistent kernel uses 238 VGPRs, 22,504 LDS bytes and no scratch/spills.

Batched DFlash chain verification above 32 aggregate tokens preserves old committed raw records in caller workspace, publishes convolution history before projection, then folds FP32 recurrent state at the next record consumer's initial load. This removes one state read without publishing current unaccepted recurrence state. The independent FP64 fold/recurrence oracle, exact ordinary state/history/record comparison, permuted slots, partial prefixes and graph replays pass at B2/W6 and B6,B8/W8. The fused record kernel uses 56 VGPRs, 18,560 LDS bytes and no spills. The isolated C6/P8192/K7 A/B (six balanced processes, discarded warmup plus three repetitions) reduces decode time by 1.15%, with exact tokens. Both arms of that mechanism comparison used the same subsequently rejected attention candidate; final production comparisons are recorded in the evidence directory. The finite crossover retains the existing front Fold through 32 aggregate tokens: forcing the new route at C2/C3/C4 K7 or C6 K3 slows decode by 1.1–1.3%. Above that boundary, the C8/K7 comparison reduces decode by 1.72%, with exact tokens.

The final persistent kernel also binds the next phase entry by reference and reads it only after the phase-slot publication barrier. Its previous by-value argument read LDS before the writer synchronized; this could select a stale instruction-prefetch entry. The final native 23-scenario exact-buffer qualification passes. The final same-session three-repetition A/Bs are:

| Workload, G128 | Control median decode seconds | Final median decode seconds | Time reduction |
|---|---:|---:|---:|
| C1 plain, P8192 | 3.63709 | 3.34141 | 8.13% |
| C1 plain, P65536 | 4.03783 | 3.74278 | 7.31% |
| C1 DFlash K7, P512 | 1.40741 | 1.40521 | 0.16% (neutral) |
| C6 DFlash K7, P8192 | 1.34016 | 1.32283 | 1.29% |

The short points discard one warmup; the 64K point measures three complete P+G requests without a discarded repetition. All retained tokens match the control exactly. Fixed-K7 C6 eager and C5 mixed-forward-256 whole requests also match original tokens exactly. Adaptive C7 graph control/final tokens match; adaptive eager selects different timing-dependent draft-width histories and differs even in the original build, so cross-mode adaptive exact-token parity is not claimed.

Standalone T1 race checking passes 2,085 workgroups with zero hazards; the staged GDN fold/record qualifier passes 168,384 workgroups with zero hazards. The persistent checker no longer reports the phase-publication read. It still reports shared-memory hazards across the scheduler's atomic wave-group barriers, which that checker explicitly does not model. Those barriers were reviewed against their release/acquire protocol, and exact lowering/graph replay passes; the whole persistent race-check run is **limited**, not a clean sanitizer pass. No suppression or weakened checker was added. Final persistent resources remain 238 VGPRs, 100 SGPRs, 22,504 LDS bytes, and zero private scratch/spills.

Long-context attention retained its existing physical layout and consumer. A full BF16 K-page staging candidate slows C1/P65536/K7 decode by 5.55%; a double-buffered raw-FP8 K-page candidate slows it by 4.60% and C6/P8192/K7 by 2.86%. Both passed the independent FP64 oracle and exact model tokens, but are rejected on whole-decode timing. A 16-token physical K tile keeps the original arithmetic and LDS footprint; nine balanced event pairs at 64K show ratios 0.97784 for T1 and 0.99583 for T8. The latter is only 0.42% of the attention operator, below a material whole-round gain for the DFlash bottleneck, so the format change is rejected. No alternate cache format or candidate dispatch remains in production. The attention qualifier now accepts `--long-context 65536` for direct T1/T8 FP64, fragmented-page, publication and graph checks.

**Compaction latency mitigation:** initial 512+1536-token runs reproduce roughly 0.6–0.9-second rounds. KFD traces align these with `EVICT_USERPTR` intervals and repeated restore retries, while individual traced kernels remain below 2.4 ms. Spin waiting does not remove the pause. The profiler inserts registered system-memory queue rings, so memory-policy decisions were also checked without profiler interception: corrected production defaults reach 745/708 ms, `HSA_USE_SVM=0` reaches 758 ms, and combined legacy/GTT settings with ordinary-RAM I/O staging still reach 168 and 428 ms. The I/O-staging control reaches 896 ms. Tokens remain exact. Those partial allocator, staging, synchronization and runtime-setting candidates were rejected; the coherent process-local change selected below supersedes that initial investigation.

A maintainer-executed privileged capture on 2026-10-16 identifies the caller: `kcompactd0` reaches `amdgpu_hmm_invalidate_hsa` through `compact_node`, `compact_zone`, `migrate_pages`, and `try_to_migrate_one`. Three native P512/G1536 runs reproduce 550/565/611-ms rounds. Surviving events within each round show per-page invalidations of shared host mappings, USERPTR queue eviction, and restoration page-fetch retries returning `-EAGAIN` before resume. Sampled process maps place the affected addresses in shared `/dev/zero (deleted)` mappings; their individual runtime allocation owners are not yet established. The evidence identifies memory compaction as an invalidation source; it does not establish a complete queue-duration accounting or a mitigation. Per-page stack collection overran trace buffers in all three attempts, so the collector correctly rejects them as complete loss-free captures. The retained stacks remain affirmative evidence of the caller, and the reported event counts are incomplete.

The collector now bounds stack samples, streams events, records mapping/reason identities and restoration return codes, and validates trace loss and slow-round correlation. It runs inference as the invoking owner with that owner's resource limits, holds the GPU lease, and removes its private trace instance and uniquely named probes. It does not change power, VM, driver or firmware policy. Root remains necessary to attach probes; this agent's passwordless sudo is unavailable. A bounded-stack rerun (`userptr-20261016-182659-ckj8jjop`) reproduces 437/458/480-ms rounds and the same migration/restore pattern, but still overruns its per-CPU buffers. No further broad per-page capture is needed to identify the caller; these runs cannot support complete event counts or latency accounting. Evidence: `userptr-20261016-181736-pfr779xv/caller-analysis.json` and the later run's compact summaries in the evidence directory.

A native process-local THP-disable screen uses `PR_SET_THP_DISABLE` before ROCm initialization, with no profiler or host-wide policy changes. The ABBA P512/G1536 runs have maximum round times 420/474/397/414 ms (control/disabled/disabled/control); all 1,536 generated tokens match exactly. Disabling THP does not eliminate the pause and is rejected. The host has shmem THP disabled already, and this control does not prevent compaction from migrating ordinary pages. No THP change is retained in production. Evidence: `thp-process/analysis.json` in the evidence directory.

A maintainer-authorized, native host-policy ABBA (`compaction-ab-20261016-183909-_y95eunf`) confirms a mitigation for the reproduced plain P512/G1536 workload:

| Arm | `vm.compaction_proactiveness` | Maximum round, ms | Decode seconds | Rounds above 80 ms |
|---|---:|---:|---:|---:|
| Initial control | 20 | 433.892 | 39.7814 | 1 |
| Disabled 1 | 0 | 25.831 | 39.3681 | 0 |
| Disabled 2 | 0 | 25.857 | 39.3801 | 0 |
| Restored control | 20 | 392.141 | 39.7546 | 1 |

All 1,536 generated token IDs match across all four arms. Native median round time remains about 25.63 ms; the benefit is removal of the observed long pauses, with about 0.99% less decode time in this workload, not faster ordinary kernels. Global VM counters show approximately 27 million daemon migration-scanned pages per control and zero during each disabled arm. Setting the original nonzero value can itself wake compaction; three-second transition settling and its counters are recorded separately from each native run. The initial unchanged-policy control also reproduces the pause. These are two disabled runs at one workload, not a guarantee against every future stall, demand-driven compaction, or different contexts/concurrency.

The reviewed helper runs inference as the original owner with recovered resource limits, holds the GPU lease, and changes no GPU power policy. Local failure-injection checks cover normal restoration, interruption immediately after a successful write, failed inference and an intervening external policy change. The experiment restored the original value `20`; no persistent sysctl was applied. This host-wide workaround is superseded by the process-local mitigation below, so no persistent compaction-policy change is needed.

A process-local implementation keeps compaction at `20`: use `HSA_USERPTR_FOR_PAGED_MEM=0` with normal SVM behavior, replace explicitly registered host arenas with HIP host allocations, and read O_DIRECT artifact and disk-cache data into ordinary aligned RAM before copying into pinned transfer slots. Memory/address/graph-transfer and artifact byte qualifications pass. Native original/candidate ABBA at P512/G1536 retains exact tokens and maxima 26.41/25.82/26.12/26.10 ms, but VM counters show no compaction in any arm. This comparison is inconclusive for mitigation. Candidate peak device-wide GTT usage is about 1.27 GB; direct-I/O staging adds up to 512 MiB ordinary RAM. The frozen implementation and evidence are under `gtt-arena-candidate/`; temporary wall diagnostics are removed from source.

The maintainer then ran the reviewed controlled-compaction ABBA (`gtt-compaction-20261016-192211-qazi8q3b`), invoking one explicit host-memory compaction pass during each P512/G1536 decode. Proactive compaction stayed at `20` throughout:

| Arm | Maximum round, ms | Rounds above 80 ms |
|---|---:|---:|
| Initial control | 486.753 | 2 |
| Driver-backed host memory 1 | 25.733 | 0 |
| Driver-backed host memory 2 | 26.562 | 0 |
| Final control | 878.307 | 2 |

All pulses overlap measured decode and scan 12–14 million pages. The new-memory runs also scan about 79.5/77.9 million pages through background compaction over their full requests, and all 1,536 generated tokens match the controls exactly. This supports mitigation under explicit compaction stress while compaction remains enabled; it does not guarantee immunity to unrelated host stalls. The binaries have different round-diagnostic instrumentation, so this comparison establishes removal of the large pauses, not a small steady-throughput gain.

`DeviceContext` fixes the process-local policy before its first HIP call, including normal SVM addressing (`HSA_USE_SVM=1`); callers need no shell override. Pinned host buffers and arenas share the real driver GTT budget (about 62.43 GiB on this host), as well as system/cgroup resources. Allocation preflight checks current available GTT; actual HIP allocation remains authoritative under competing allocations. Requested RAM-KV capacity is never silently reduced or redirected to movable USERPTR memory. GTT errors are distinguished from automatic VRAM capacity errors. The original mmap/register arena and its bookkeeping are removed.

Default-startup qualification verifies actual GTT consumption without environment overrides, CPU/device pointer identity, exact captured transfers over repeated replays, exhausted-GTT admission, and exact artifact materialization. The 8-GiB RAM-cache cancellation/publish/evict/ restore qualification and MTP/DFlash checkpoint-state and continuation recovery pass. The full disk-cache qualification and real-artifact ordinary/MTP/DFlash cache-write integration, including disk restart and fork/cancel behavior, also pass. Disk reads retain O_DIRECT and use ordinary aligned page staging plus at most 4 MiB per raw-state read worker before copying into GTT. Page-read scratch remains allocated while the reader thread lives, including between restores; state-worker scratch and artifact-loading staging have temporary lifetimes. Native default-startup comparisons against the frozen pre-change control also retain exact tokens at C8/P8192 DFlash K7 (264 tokens across eight lanes) and C1/P65536 plain decode (33 tokens). Each lane retains the first prefill-produced token plus 32 decode tokens. All nine host tests, affected release builds, clang-tidy and changed-file formatting/lint checks pass. Source and probe changes received independent subagent review before GPU qualification.

A mount-specific disk-cache comparison uses C1 DFlash K7 append-frontier restores at 8K/64K, raw disk-cache state, 8-GiB RAM and disk budgets, and the same retained cache on `the local storage pool`. This is a mirrored NVMe ZFS mount (kernel 2.4.1, 128-KiB records, lz4 and primary caching enabled). A diagnostic USERPTR/direct-read control shares the current HIP host arena and artifact-loading code; it reproduces the former restore route rather than the exact former mmap/register build. Model loading is excluded. Direct/staged/staged/direct arms each make five restores; the first restore after each Engine start is separate, leaving eight repeated samples per route/context.

| Context | Direct restore hold, ms | GTT/staged restore hold, ms | Direct TTFT, ms | GTT/staged TTFT, ms |
|---|---:|---:|---:|---:|
| 8K | 72.374 | 78.484 | 103.237 | 109.189 |
| 64K | 336.045 | 336.455 | 373.426 | 374.100 |

These are medians: the observed combined-route cost is about 6 ms at 8K; the 0.7-ms 64K TTFT change is within run variation. Repeated 64K direct TTFT spans 366.6–617.9 ms versus 372.7–375.1 ms staged; these outliers were not traced and are not attributed to compaction. First-restore TTFT varies substantially in both routes and does not establish a reliable startup penalty. An auxiliary USERPTR/staged arm has repeated median TTFT 112.7/374.9 ms at 8K/64K, consistent with staging contributing the small 8K cost; its later ordering and different pool-read volume prevent isolated memcpy-cost attribution. All 50 restores require HostDisk, exact cold-reference tokens/frontiers, stable entry membership and zero fallback. The seed saves the append frontier without ladder-head images. Positive device-wide pool read counters confirm physical I/O occurred, but include other processes; these are warm/repeated ZFS measurements, not guaranteed cold SSD reads. Compaction stays at 20, clocks at auto, and no filesystem/cache policy is changed. Independent review covers the harness and interpretation. Evidence: `disk-latency-mount/report.txt`, `setup.json`, `summary.json` and per-arm records under the same campaign directory.

Commands, paired samples, numerical and resource evidence are retained under `profiles/bench/r9700-decode-fixes-20261016/`. Measurements use the same production artifact, fixed three-plane cache, physical PCI device and bulk graph environment as the initial profile, with `auto` checked before and after GPU work. No clock, voltage or power-policy changes were made. Temporary candidates and per-round timing diagnostics are not production code.

### Bulk graph submission with host arguments (2026-10-16 UTC)

Compose now selects `DEBUG_HIP_GRAPH_CLASSIC_PATH=0`, `ROC_ENABLE_LARGE_BAR=0`, `ROC_SKIP_KERNEL_ARG_COPY=0`, `HSA_DISABLE_FRAGMENT_ALLOCATOR=1`, and `DEBUG_HIP_GRAPH_BATCH_SIZE=16384`. On HIP 7.15.26333 / ROCr 1.21 this places immutable executable arguments in pinned host memory while retaining cached AQL packet replay. It removes the classic path's per-node command creation without restoring its predecessor's large device argument pools. `ROC_ENABLE_LARGE_BAR=0` controls the runtime allocation policy; it does not change the card's BAR configuration. Shared fixed-address phase images remain intact.

The checked-in `ninfer_r9700_graph_launch_bench` alternates two immutable graphs with distinct 512-byte by-value payloads and an output pointer per node. An independent integer oracle checks initialization, capture, warmup and every repetition. Seven repetitions each submit 100 graphs, with synchronization outside each submission-thread CPU interval. The host is a Ryzen 9 7950X3D; GPU power policy is `auto`, with no simultaneous build or GPU workload.

| Nodes per graph | Classic submission CPU, us | Bulk host-argument submission CPU, us | Reduction |
|---:|---:|---:|---:|
| 1 | 3.35 | 2.46 | 26.4% |
| 1000 | 1024.34 | 14.61 | 98.6% |
| 4096 | 4717.60 | 44.34 | 99.1% |

These are median calling-thread CPU times, not GPU execution time or token throughput. Synchronized wall time for 1000 nodes was 3825.81 versus 3696.64 us; for 4096 nodes it was 15001.6 versus 14964.9 us. Both include synchronization. The largest observed model graph has 1020 nodes, including 1008 kernel nodes. Packet batch caps 256/1024/4096/16384 passed the same exact oracle. The selected cap, 16384, won at 4096 nodes; 1024 was slightly faster at 1000 nodes (13.51 versus 14.61 us). This does not establish a unique winner at every graph size. Direct doorbell and alternate kernel-argument memory controls gave no reliable additional benefit in pinned-core forward/reverse controls and retain their defaults. This is a measured reduction, not a claim of zero remaining runtime overhead.

Normal argument copying is required. `ROC_SKIP_KERNEL_ARG_COPY=1` failed the 4096-node oracle and also produced a launch failure in a subsequent check; the normal-copy route passed at all three sizes with both host and device argument pools. The selected mode also passed all 19 persistent-phase cases and exact graph/eager generated-token checks for ordinary, MTP and DFlash inference at C2. No kernel arithmetic or model precision changed.

Raw scope-separated evidence is under `profiles/bench/r9700-graph-submit-20261016/qualified/`: `submission-matrix.json`, `submission-provenance.json`, `pinned-submission.json`, `phase-qual.json`, and `functional.json`. The provenance records the inherited normal-copy, host-argument environment as well as each matrix row's overrides.

The 2026-10-16 adaptive-K7 study measured the following C-specific GPU graph inventory. The 2026-10-17 README refresh above updates the C1 allowance and resolved capacities:

| C | Graph budget / observed, MiB | Shared KV tokens | Minimum physical/HIP free, MiB |
|---:|---:|---:|---:|
| 1 | 56 | 262144 | 9835.71 |
| 2 | 56 | 524288 | 2970.71 |
| 3 | 60 | 630144 | 73.90 |
| 4 | 60 | 622016 | 71.28 |
| 5 | 62 | 613952 | 68.13 |
| 6 | 62 | 605824 | 66.98 |
| 7 | 62 | 597696 | 66.19 |
| 8 | 62 | 589568 | 65.77 |

These capacities precede the physical-residual correction in the 2026-10-16 shared-GDN follow-up below; that correction supersedes the C6..8 capacity values while retaining this graph inventory.

All C1..8 cells exercise startup, fresh-Engine SSD restore with an exact cold-token check, and short/repeated requests, with all C lanes decode-ready. C6 and C8 additionally run near capacity at 100352 and 71680 prompt tokens per lane respectively. The minimum is sampled every 50 ms and checked at synchronized phase boundaries; it is not a proof against unsampled external allocations. The 64-MiB external headroom remains physically available. These are instrumented memory checks, not throughput measurements. `physical-manifest.json` records the commands, actual graph bytes, resolved KV and observed driver growth. The fixed startup inventory guard remains mandatory. Host argument pools consume pinned host RAM rather than device VRAM.

Uninstrumented whole-model comparisons use fixed DFlash K7, optimized proposal head, P2048, max context 262144, an 8192-token prompt, one warmup and three repetitions. Isolated decode measures 256 post-prefill decode tokens per lane; whole inference includes the first prefill token plus 128 decode tokens per lane. Both policies use automatic KV sizing. Every generated greedy token matches across policies at C1 and C6.

| Workload | Previous classic | Bulk host arguments | Change |
|---|---:|---:|---:|
| C1 isolated decode tok/s | 192.368 | 192.404 | +0.02% |
| C1 whole output tok/s | 39.275 | 39.267 | -0.02% |
| C6 isolated decode tok/s | 661.820 | 668.690 | +1.04% |
| C6 whole output tok/s | 45.737 | 47.282 | +3.38% |

C6 resolved KV capacities were 593856 / 593856 tokens (classic / bulk).

`model-performance.json` and the individual reports retain throughput dispersion and commands. C6 whole-inference standard deviations were 1.621 / 0.019 tok/s (classic / bulk), so the observed +3.38% whole-inference difference does not establish a repeatable speedup by itself. This compares submission policies at fixed K; production adaptive length can change its draft schedule with timing, so unchanged adaptive runs need not produce identical continuations. The prior adaptive-repeatability limitation remains in force. These results establish the submission-CPU reduction and measured model behavior, not a universal minimum launch cost.

A separate instrumented C6 fixed-K7 whole run (8192/128, max context 16384) measured CPU time strictly inside each real `hipGraphLaunch`, discarding the first two calls per executable. Across 32 / 32 retained calls (classic / bulk), weighted mean submission CPU was 767.89 / 25.89 us. These instrumented runs establish submission attribution, not throughput. `model-submit-profile.json` retains node counts, call counts and per-executable CPU ranges; the read-only interposer source is retained beside it.

The rebuilt Compose container passed health, image/argument verification, greedy HTTP and default sampling checks. It resolved 605824 KV tokens with graph observed/budget 62/62 MiB, 67.07 MiB available after startup and 67.04 MiB physically free after requests. Host cgroup peak was 18.29 GiB under the unchanged 24-GiB/no-swap limit. The container was then stopped and this session's GPU lease released; the queued owner immediately acquired it. `deployment.json` and `lock-released.json` record the checks and handoff.

### Previous classic graph residency and CPU submission (2026-10-16 UTC)

The following measurements describe the preceding classic-replay revision. Its memory results remain a historical control; the active Compose submission policy is described above.

That revision selected `DEBUG_HIP_GRAPH_CLASSIC_PATH=1`, `ROC_SKIP_KERNEL_ARG_COPY=1`, and `HSA_DISABLE_FRAGMENT_ALLOCATOR=1` on the selected HIP 7.15.26333 / ROCr 1.21 runtime, with other driver controls at their defaults. The calibrated preset remains `R9700Lq4`, C1..8, max context 262144, prefill chunk 2048, adaptive DFlash K7, optimized proposal head, no Vision, RAM cache 8192 MiB and SSD cache 25600 MiB.

Classic replay avoids the packet path's per-executable device argument pools. Direct argument capture uses runtime-owned argument storage and removes a redundant CPU copy. Persistent decode lowering also interns byte-identical run images into fixed-address 2-MiB arenas. Definitions retain those arenas after the startup index is destroyed. Their shared barrier words require the Engine's single ordered compute stream; replay adds no allocation, interning or graph update.

The previous 74-MiB value was a C6 observation, not a universal graph bound. With shared images, the complete immutable inventory (65*C executables) exactly matches the following per-C graph allowances. The low table applies only to all three settings and the qualified production preset. The same preset without those settings retains the prior packet-path table as a conservative bound; it is not a newly minimized measurement of that path. Other presets and qualification builds use the full conservative inventory bound. Requalify when kernels, inventory or the HIP toolchain change. The observed-allocation startup guard remains mandatory.

| C | Graph MiB, observed = budget | Late driver MiB, including SSD | Shared KV tokens | Long prompt qualification per lane | Minimum physical free MiB |
|---|---:|---:|---:|---:|---:|
| 1 | 56 | 8 | 262,144 | 65,536 | 9835.754 |
| 2 | 56 | 8 | 524,288 | 65,536 | 2970.883 |
| 3 | 60 | 6 | 630,144 | 65,536 | 74.203 |
| 4 | 60 | 6 | 622,016 | 65,536 | 71.723 |
| 5 | 62 | 4 | 613,952 | 65,536 | 68.695 |
| 6 | 62 | 4 | 605,824 | 100,352 | 67.668 |
| 7 | 62 | 4 | 597,696 | 65,536 | 67.016 |
| 8 | 62 | 4 | 589,568 | 71,680 | 66.715 |

All physical KV storage is allocated before the workload, and startup warms every captured B/K/target structural class at its largest frontier. The initial HIP-availability campaign restored an SSD prefix and ran 8K/long/8K prompts at every C, generating 129 tokens per lane. Long prompts used 64K at C1..5/C7 and near-capacity lengths at C6/C8, with all C lanes decode-ready.

Deployment exposed a limitation of that counter: HIP reported 70 MiB available at startup, while physical free VRAM was about 16 MiB after HTTP smoke requests. Those readings occur at different times and do not establish the exact source of the gap. An isolated fresh-HIP process also showed a different counter relationship, so no fixed offset or exact driver cause is assumed. Sizing now waits for RAM-cache registration and uses the smaller of HIP availability and the matching PCI device's graphics-inclusive amdgpu total-minus-used counter. Startup availability uses the same query; the required sysfs counters must be readable.

The table records corrected automatic capacities and physical runtime minima from a second, independent 50-ms sampler. Every C repeats startup, SSD restoration with its next greedy token exact against a cold run, and short requests. C6/C8 also repeat near-capacity long requests. The other long-prompt entries refer to the first campaign's larger physical pools; their graph, operator and lifetime qualification remains applicable at the reduced capacities. The late-driver column retains the HIP-measured growth used for its separate reservation. Capacity checks are instrumented and are not throughput evidence.

That qualification used separate reservations: 2 MiB startup driver residency, the C-specific late driver values above, SSD device storage (54 MiB with DFlash), and a 4-MiB arena allocation bound. The C6..8 main-stream allowances are superseded by the physical-residual correction below. None consumes the external 64-MiB headroom. At C6, graph allowance fell from 1624 to 62 MiB and shared KV grows from the prior deployment's 548736 to 605824 tokens after the physical-availability correction.

The exact-capacity C8 engine reproduced a workspace allocation failure with 442 MiB reported free. Allocating workspace first then reached an SSD staging allocation failure with 124 MiB reported free. Disabling ROCr's fragment allocator resolves the reproduced failures with the original allocation order and the same capacity; no ordering workaround or extra reserve is retained. The exact allocator failure mechanism is not established by this comparison.

The unprofiled launch probe alternates two 1000-node integer graphs with a 512-byte payload plus an output pointer per node. Five repetitions retain exact output and zero late VRAM growth. Median submission-thread CPU time falls from 1049.80 to 696.86 microseconds per graph (33.62% less). This is CPU submission work, not total process CPU or whole-model throughput.

Separate uninstrumented model runs use one warmup and three repetitions, without concurrent compilation. The baseline also uses classic replay and disables ROCr's fragment allocator; the candidate adds direct argument capture. C1 fixes K7; C6 uses the deployed adaptive policy and automatic KV capacity. Controlled C1 and C6 fixed-K7 runs have exact greedy token parity; C6 fixed-K7 also checks that both policies resolve the same physical KV capacity. Adaptive C6 continuations are not exactly repeatable even within the unchanged baseline: both policies produce the same two observed 129-token continuation variants across workloads, with the first divergent token at index 73. Their measured draft-length schedules differ, but the aggregate reports do not establish the causal transition. Adaptive results are throughput observations, not exact-token parity evidence. The table uses per-repetition medians; the raw reports retain means and deviations. One short C1 baseline repetition was slower, so its mean difference does not establish a speed gain.

| Measurement | Classic median | Direct-argument median | Observed change |
|---|---:|---:|---:|
| C1 fixed K7 pp128+tg512, decode tok/s | 90.119 | 90.085 | -0.04% |
| C1 fixed K7 pp8192+tg512, decode tok/s | 224.489 | 224.537 | +0.02% |
| C6 adaptive max K7 8192/256, isolated decode tok/s | 653.468 | 620.092 | -5.11% |
| C6 adaptive max K7 8192/128, whole output tok/s | 47.147 | 47.027 | -0.25% |
| C6 fixed K7 8192/256, isolated decode tok/s | 665.719 | 661.953 | -0.57% |
| C6 fixed K7 8192/128, whole output tok/s | 46.522 | 47.278 | +1.63% |

The CPU saving does not imply an equal inference-speed gain. Report means, deviations, commands, exact-token checks, per-C memory evidence and primary runtime source references are retained in `profiles/bench/r9700-classic-launch-20261016/`. The 19-case exact graph transformation checks pass in packet and classic modes, real-artifact C2 graph/eager parity passes for ordinary/MTP/ DFlash, planner checks cover complete and partial flag combinations, and independent review finds no ownership or synchronization blocker.

### Packet-path VRAM baseline (2026-10-15)

This historical control supplies the conservative packet-path bound retained after phase-image sharing. Compose now uses the qualified bulk host-argument configuration above. Availability and minimum-free figures in this historical control use HIP alone; they are not physical-headroom guarantees.

The production `R9700Lq4` artifact was measured independently at every C1..8 on one gfx1201 R9700 with the selected `/opt/rocm/core-10.0` toolchain (HIP 7.15.26333), `max_context=262144`, prefill chunk 2048, adaptive DFlash K7, optimized proposal head, no Vision, 8-GiB RAM cache, 25-GiB SSD cache, and GPU power mode `auto`.

The earlier startup-only calibration is superseded. Repeated `hipGraphExecUpdate` calls retain kernel-argument allocations on this HIP build: an independent two-graph, 1000-node reproducer measured 2/10/42/84/170 MiB retained after 1/8/32/64/128 updates, with exact buffer results. Thus the previous 42-MiB growth was NInfer-induced residency and was not a valid use of the 64-MiB external headroom. An allowance based only on startup cannot bound that update path.

Each B/K/context definition now owns an immutable executable: 65*C for this preset. Startup instantiates and uploads all of them, then warms the largest-frontier representative per family, exact B, captured K and target structural class. Serving directly selects the captured executable; it never updates, instantiates or uploads one. The measured graph reservation below has no arbitrary padding. Other selectors and qualification builds use the conservative full immutable-executable inventory bound; requalify the table after a graph/kernel/toolchain change.

NInfer separately reserves 2 MiB of startup driver residency preceding graph preparation (the first synchronous HIP clear of DFlash's 8-byte completion flag), plus its measured lazy kernel-driver residency. The main-stream component was 6/6/4/4/2/2/2/2 MiB at C1..8; the 2026-10-16 shared-GDN follow-up below corrects the physical residual at C6..8. Enabling SSD adds 2 MiB for the scatter kernel. These are NInfer-owned resources inside `B(M)`, in addition to the SSD's 52-MiB staging buffer and its 2-MiB descriptor allocation bound. The existing 4-MiB bound covers physical allocation rounding of the runtime arenas. Neither physical rounding nor lazy driver resources consume external headroom. HIP arena accounting rounds in 2-MiB units; physical-counter residency can differ, so logical reservation bytes need not equal physical residency byte for byte.

| C | Graph budget MiB | Late driver MiB, with SSD | Shared KV tokens | Long prompt per lane | Minimum observed free MiB |
|---|---:|---:|---:|---:|---:|
| 1 | 318 | 8 | 262,144 | 260,096 | 9676 |
| 2 | 578 | 8 | 524,288 | 260,096 | 2554 |
| 3 | 842 | 6 | 603,776 | 200,704 | 68 |
| 4 | 1102 | 6 | 585,408 | 145,408 | 66 |
| 5 | 1364 | 4 | 567,104 | 112,640 | 68 |
| 6 | 1624 | 4 | 548,736 | 90,112 | 68 |
| 7 | 1884 | 4 | 530,368 | 73,728 | 68 |
| 8 | 2144 | 4 | 512,000 | 63,488 | 70 |

For each C, a saved prefix was restored from SSD into a fresh Engine, and its next greedy token matched a cold run. The same Engine then ran an 8192-token prompt, a balanced long prompt near its shared physical capacity, and another 8192-token prompt. Every test generated 129 tokens per lane. Prefill-first scheduling held all C long requests decode-ready together; observed mapped pages covered at least all C prompts. The initial separate C1..8 checks used the container's mixed-forward `auto` scheduling. The explicit 2-MiB startup-driver correction was made after C1..4 long runs. C1..2 retained the same capped pools; C3..4 received a final startup/short-workload capacity recheck. Their long prompt lengths were unchanged and their observed mapped pages fit the final smaller pools. Their table minima are earlier long-run observations, conservative for those smaller pools, rather than new long-run measurements. No runtime graph updates or new executable instantiations occurred, and no measured cell consumed the 64-MiB external reserve. Allocation probes and periodic/boundary memory queries make these capacity checks ineligible as throughput measurements.

At C6, the original 2418-MiB graph allowance becomes 1624 MiB. With corrected SSD storage and driver accounting, shared KV capacity is 548,736 tokens, 28,928 above the original 519,808-token control. This replaces the invalid earlier claim of 569,664 tokens. C1 and C2 reach the `C*max_context` ceiling, explaining their unused VRAM.

The earlier instrument-free 1024-versus-2048 chunk sweep covered every C1..8 on the prior shared executable implementation, with 8192-token pure-prefill and whole-inference workloads, one warmup and three repetitions. C6 measured 3400.86 versus 3341.14 prefill tok/s and 47.405 versus 47.267 whole output tok/s for 2048 versus 1024; C3 favored 1024. Those timing results remain under `profiles/bench/r9700-vram-qualification-20261015/`, but its startup-only capacity qualification is explicitly superseded. Chunk 2048 remains the deployed choice; the current exact graph calibration does not admit unmeasured chunk 1024.

A separate uninstrumented C6 confirmation on the immutable executable implementation used the same 8192-token workloads, one warmup and three repetitions. Prefill measured 3395.44 tok/s versus the earlier 3400.86 (-0.16%); whole output measured 47.435 versus 47.405 tok/s (+0.06%). These observed differences do not establish a throughput improvement or regression. The comparison is recorded in `performance-comparison.json` beside the capacity evidence.

Five focused host planner/routing checks, SSD cache integration, and the immutable first-launch and 128-switch zero-growth GPU regression pass. Real-artifact C2 graph/eager checks retained exact greedy tokens for ordinary decoding and fixed-K7 DFlash. Changed-line static analysis and independent source/allocation review pass. Commands, environment, per-C memory observations, exact-token evidence and the consolidated qualification are under `profiles/bench/r9700-vram-runtime-budget-20261015/`.

The rebuilt Compose service was also checked at C6: health, image identity, literal `--max-context 262144`, resolved 548,736-token capacity, greedy generation and default-temperature generation passed. It reported 1608 MiB graph residency against the 1624-MiB measured bound, with 88 MiB free at startup. The per-C bound retains the maximum observed qualification residency rather than subtracting this deployment's 16-MiB unused graph allowance. `deployment.json` records the live metrics and request checks.

### Sustained memory bound

The native exact-device probe uses two 4 GiB buffers, 512 times larger than L2, and reports five trials of at least 0.25 seconds. Its aggregate read checksum covers every workgroup and matched the expected value exactly.

| Method | Best bus rate | Median bus rate | Advertised-peak fraction |
|---|---:|---:|---:|
| kernel uint4 read | `636.2 GB/s` | `636.0 GB/s` | `99.4%` |
| kernel uint4 write | `588.0 GB/s` | `587.9 GB/s` | `91.9%` |
| kernel uint4 copy | `549.0 GB/s` | `548.6 GB/s` | `85.8%` |
| HIP D2D copy | `543.8 GB/s` | `543.4 GB/s` | `85.0%` |

Copy bus rate counts both the read and write traffic. This is the hardware bandwidth bound, not a model or individual-Op throughput claim.

## Upstream sync qualification (2026-10-04, initial AMD base `687c84db`)

Native ROCm 10 Release on one R9700, `gfx1201` wave32, uniform A8/Q4, FP8-K/INT4-V G16 Text/MTP cache and private BF16 DFlash state. Every GPU command held the shared AMD GPU lock. The context qualifier verified the physical PCI identity and `auto` power mode before device construction and again after its oracle and timing work. These are complete Op edge timings, with warmed weights, alternating captured 64-edge graphs, ten retained trials and median HIP events; they are not cold-weight bandwidth or whole-inference gains.

| Context projection tokens | Full 6144 QKV rows (us) | Selected 2048 K/V rows (us) |
|---:|---:|---:|
| 1 | 57.95 | 56.56 |
| 8 | 21.99 | 14.40 |
| 16 | 24.75 | 15.68 |
| 32 | 31.98 | 22.53 |
| 64 | 48.80 | 27.16 |
| 128 | 94.01 | 62.95 |

The passive binder row view consumes the original Q4 codes/scales without repacking. All outputs passed the independent FP64 public-input Linear oracle with the native A8 error budget; supplementary selected-row equality with the full projection was exact at T1/4/8/9/16/17/32/64/65/128/129. The fused local K RMSNorm/RoPE/append oracle used complete FP64 normalization and rotation, with max-absolute error <=0.02 and relative RMS <0.004; it did not copy the kernel's private BF16 normalization staging. V, unselected lanes and rejected rows were exact, including wraparound and graph replay with changed device counts.

Norm/RoPE plus append became one kernel: B1/W8 improved 7.72 to 4.82 us, B4/W8 8.52 to 5.19 us, B8/W8 10.18 to 6.78 us, and the B1/W2048 prefill edge 72.69 to 62.08 us. This is not a universal geometry claim: the nonproduction B4/W64 append cell regressed 18.20 to 21.99 us. The current multi-request decode append width is at most eight; prefill appends one owner.

The proposed convolution finish/residual/RMSNorm fusion was rejected. Its parallel normalization passed the FP64 oracle, memcheck and racecheck, and reduced the 64-column reset-plus-edge from about 43 to 16 us, but changed late public greedy tokens at C8. The same context-KV port with only finish fusion disabled restored exact tokens. A fused version preserving the incumbent serial FP32 RMS reduction restored exact C8 tokens in all three repetitions but regressed the edge to about 130 us. Timings used alternating captured 100-edge graphs with a common D2D residual reset; the reset is included in both values. Intermediate 35/40/48/56-column parallel cells also lost to the existing two-kernel route. Neither candidate nor its extra normalized panel is retained. Upstream's NVIDIA three-to-one launch reduction does not apply to AMD's existing residual-fused finish.

The retained port on the initial AMD base was compared against that pulled snapshot using the explicit production artifact `/path/to/models/qwen3.8-27b-r9700-lq4/qwen3.8-27b-r9700-lq4.ninfer` and `ninfer_bench -pg 512,128 --spec dflash --draft-tokens 7 --lm-head-draft --max-ctx 4096 --warmup 1 -r 3 --retain-token-ids`, at fixed C1 and C8 with Device Graphs enabled. All three repetitions matched every generated greedy token and all speculative counters at both concurrencies. Decode-output throughput was 99.47 to 100.02 tok/s at C1 and 352.89 to 354.18 tok/s aggregate at C8; these small observed gains do not establish a broad end-to-end speedup. C1 prefill was 3208 to 3255 tok/s; concurrent prefill has no scalar throughput result in this report. Reported workspace and allocator peaks were unchanged. Summarized qualification and schema-24 reports are under `profiles/bench/r9700-upstream-sync-20261004/`.

Planned RAM/disk state selection passed exact kept/untouched GDN, hidden and DFlash cyclic byte checks for frontier, turn, response and checkpoint-drop reuse. The synthetic disk case decoded six state images when both sets were kept and three when only one was needed. Context, RAM/disk restore qualifiers passed memcheck; the independent disk integration test passed exact bytes, restart, identity, CRC and invalidation. Real selected-artifact DFlash and MTP recovery passed resident and RAM reuse, abort/restore, and decoded retries with matched state and continuation. The runtime planner passed C1..8 workspace/graph inventory checks; six affected host tests, affected Release targets, formatter, clang-tidy and diff whitespace checks passed.

### Push-time integration with AMD base `dc05db28`

Before publication, origin advanced through `38de9cd6` (producer-written Q4 images), `567d5b94` (single-launch column top-16) and `dc05db28` (benchmark record). The sync was rebased onto this base. Context still quantizes its shared activation image once, and every layer projects only the selected 2048 K/V rows from that image before the fused normalization/RoPE/append. The new producer-image qualifier compares eager and captured N2048 projection directly with the same independent FP64 public-input oracle and requires exact canonical-projection equality at all listed token widths. Subagent review cleared the merged image lifetime, row view and arena recipe before GPU work.

The combined Release build passed the context/image, Q4 producer, column top-16, convolution, RAM/disk restore and runtime-planner qualifiers, selected-artifact DFlash/MTP recovery, context memcheck, six affected host tests and scoped clang-tidy. A matched Device-Graph Engine A/B against `dc05db28`'s context/proposal schedule used the same explicit production artifact, pp512/tg128, fixed K7, shortlist head, context4096, one warmup and three repetitions. Every generated greedy token and every speculative counter matched at C1 and C8. Decode-output throughput was 100.38 to 100.71 tok/s at C1 and 356.97 to 357.43 tok/s aggregate at C8; these are small observations, not a broad speed claim. GPU work held the shared lock, with `auto` checked before and after the Engine batch. Results are under `profiles/bench/r9700-upstream-sync-push-20261004/`. The earlier rejected upstream fusion candidate remains excluded; origin's independently qualified producer/carry implementation is preserved.

## Current production (2026-09-28)

Artifact `qwen3.8-27b-r9700-lq4` (LQ4 Text and output head, 21 row-scaled FP8 protections); `compose.yaml` serving: DFlash `--draft-tokens 7 --adaptive-draft --lm-head-draft`, Device Graphs, prefill chunk 2048. Latest measurement of each phase; the dated sections below are the history behind them.

| Phase | Measurement | Result | Source section |
|---|---|---|---|
| Decode C1 | 12-prompt corpus, 768 new tokens, greedy + 2 p-less seeds | 90.6 tok/s per request (2.86 tok/round, 31.6 ms/round) | Adaptive K{3..7} default |
| Decode C4 | same corpus, 4 concurrent | 69.5 tok/s per request, 182 tok/s aggregate (2.71 tok/round, 39.1 ms/round) | Adaptive K{3..7} default |
| Prefill 2K | C1, 1,992-token prompt | ~3,000 tok/s, TTFT ~0.65 s | server logs, `profiles/bench/r9700-stall-probe-20260927/` |
| Prefill 32K | C1, code corpus | 2,790–2,850 tok/s | Long-context prefill attribution |
| Prefill 128K | C1, code corpus | 1,901 tok/s (69.0 s) | Long-context prefill attribution |
| Short append | 19-token follow-up turn on a 22K conversation, host-RAM reuse | TTFT 240 ms (was 818 ms) | Mid-row attention route |
| Decode C1..C4 | 13-prompt agent/code study corpus, p-less T1.5, 768 new tokens x 3 seeds | 90 / 148 / 181 / 204 tok/s aggregate | Batched DFlash drafting |

Decode depends on DFlash acceptance and therefore on the prompt mix; compare only within one harness. Prefill rows predate the 2026-09-28 output-head and attention GPTQ reconversions, which change weight values but not formats or routes, and all rows predate the 2026-09-30 reduction to 21 protections (C1 decode +2.3%, C4 unchanged, prefill +0.4%).

## Prefill follow-up attribution after bulk graph submission (2026-10-16)

Diagnostic on `experimental` after pulling `6f069b6c`, rebuilt Release `ninfer_bench`, production 21-protection `r9700-lq4` artifact, one R9700/gfx1201, ROCm 10, unchanged 300 W cap and `auto` power profile. The canonical host-argument bulk graph flags from the container were used. The shared GPU lease covered every inference process and was released after collection.

Workloads use `bench/fixtures/bench_corpus.ids`, chunk 2048 and mixed-forward `auto`: C1 ordinary prefill-only 8192 tokens (one sampled output, no speculation), and C6 fresh whole 8192-token prompts with fixed DFlash K7, optimized proposal head and 128 decode steps per lane (129 output tokens including the first prefill token). One warmup and three unprofiled repetitions, max context 262144 and automatic KV capacity, gave:

| Scope | Mean | Standard deviation |
|---|---:|---:|
| C1 8K prefill throughput | 3431.50 tok/s | 9.92 tok/s |
| C1 8K prefill service | 2.38716 s | 0.00690 s |
| C6 aggregate active prefill throughput | 3193.67 tok/s | 4.74 tok/s |
| C6 summed prefill service | 15.39048 s | 0.02285 s |
| C6 whole wave wall | 16.68891 s | 0.32627 s |

Service and decode costs overlap; they must not be added as a wall-time decomposition. These are current baselines, not an A/B speedup result.

Selected-region marker/kernel/copy traces capture one measured repetition after one warmup, excluding load, graph priming and warmup. Traces use max context 16384 and workload-sized KV (C6 timing uses 593856 KV tokens) to leave profiler memory. All traced output token IDs match all three unprofiled repetitions, including every C6 lane. Kernel shares below use summed dispatch durations, which can overlap; they are attribution, not exclusive wall shares:

| Kernel family | C1 prefill | C6 whole |
|---|---:|---:|
| LQ4/raw-FP8 prefill GEMMs | 80.93% | 72.78% |
| of which MLP gate/up and down | 56.71% | 50.49% |
| Chunked GDN recurrence | 5.70% | 5.25% |
| Dense prefill attention | 5.57% | 4.99% |
| Text FP8 activation preparation | 5.06% | 4.59% |
| GDN projection/conv scatter | 1.56% | 1.38% |

C6 projection durations include the verify columns sharing mixed forwards. Its initial ordinary owner uses 2048-token chunks; most subsequent prefill/mixed projection launches use width 1024. The dominant GEMM variants use 240 VGPRs; raw-FP8 and smaller-width variants use fewer. All captured prefill GEMMs have zero scratch; chunked GDN uses 256 VGPRs, 58880 bytes LDS and 8 bytes scratch. A late 100 ms telemetry sample during the C6 trace observed 41 samples above 90% GPU busy, averaging 297 W; this is a coarse trace-time observation, not proof of individual kernel saturation or a reason to change clocks/power.

The highest-value kernel target is the MLP GEMM family at N34816/K5120 (gate/up) and N5120/K17408 (down), particularly the real mixed width 1024 and ordinary width 2048. A serial estimate from the C6 trace puts a 10% reduction of MLP kernel duration near 5% of whole-wave latency; this is a prioritization bound, not an achieved or guaranteed speedup. Prior wider/narrower tiles and weight-staging experiments remain rejected as recorded below. A new challenger needs a specific improvement to issue/staging/epilogue efficiency and the owning FP64 oracle before timing. GDN register lifetime/scratch and moving reusable intra-chunk work off its serial recurrence are secondary hypotheses; its entire captured cost is only about 5%, and the previously rejected value-row split should not be repeated without a changed mechanism. Attention is a secondary 8K target; the historical 128K attention-heavy attribution remains a different workload.

The union of kernel intervals leaves 5.73% of C1 measured wall and 11.46% of C6 measured wall without a captured kernel active. C6 copy durations sum to 13.21 ms (0.08% of measured wall), so transfers alone cannot explain that interval, although copy duration is not additive wall time. The subsequent HIP API trace (`c6-whole8192-api-trace`) retained exact token parity. It recorded 15.508 s inside 756 `hipMemcpyAsync` calls, of which 96.38% overlapped active GPU kernels; this API duration must not be added to GPU time or interpreted as CPU execution cost. The 51087 `hipLaunchKernel` calls totalled 202.60 ms of API wall duration, 83.30% overlapping GPU kernels. The trace also records the profiler's system-memory interception-queue fallback. Its graph/API latencies and inactive intervals cannot replace unprofiled measurements or justify deleting required host-visible token/state synchronization. C6 ordinary chunk markers cover only the initial owner, so their gap cannot describe all mixed prefill service. Remaining inactive time is not established recoverable launch overhead.

Evidence: `profiles/rocprof/r9700-prefill-20261016/` contains exact commands, unprofiled reports, selected-region SQLite traces, token comparisons and standalone attribution scripts. No numerical, kernel, scheduling or container configuration was changed by this diagnostic. The maintained legacy whole-profile analyzer rejects benchmark schema 24 and C6; the retained standalone SQL analysis validates measured-region containment and uses explicit stage associations instead.

## FP8 activation preparation follow-up (2026-10-16)

The next bounded prefill experiment targets activation preparation on top of the shared-GDN implementation and upstream persistent-decode/deferred-GDN update. The production artifact remains `qwen3.8-27b/r9700-lq4`; the FP8-K/INT4-V/FP16-scale cache, scheduling, precision boundaries, clocks and 300-W power cap are unchanged. The R9700 runs in `auto`, with Release ROCm 10 / HIP 7.15 / AMDClang 23 and the canonical graph environment above.

For ordinary T>=256 rows of K5120 or K17408, the producer retains packed BF16 inputs across the row reduction instead of loading them twice. Sign-cleared BF16 magnitude bits admit one unsigned maximum that also detects nonfinite rows. The shared fused producer encoder likewise reduces unsigned FP32 magnitude bits. Both preserve the existing FP32 `maximum / 448` scale and zero-row policy. Native two-value saturated E4M3 conversion replaces two scalar conversions: ISA inspection confirms four packed conversion instructions per eight values, rather than eight packed instructions with duplicated inputs. Smaller/other plain rows keep the existing resident body. There is no added global workspace or changed activation image format.

Unprofiled owning-Op event measurements use seven groups of twenty launches after four warmups. The exact-code fixture has the same represented inputs before and after; baseline ranges below span its two surrounding invocations. These are producer latencies, not whole-inference gains.

| Plain activation shape | Baseline range | Selected producer |
|---|---:|---:|
| T1024, K5120 | 0.041526–0.042966 ms | 0.029520 ms |
| T2048, K5120 | 0.052462–0.052786 ms | 0.037320 ms |
| T1024, K17408 | 0.070962–0.071230 ms | 0.047480 ms |
| T2048, K17408 | 0.198428–0.198668 ms | 0.187270 ms |

The fused T2048 RMSNorm producer measured 0.065786 ms versus 0.079276–0.081022 ms, and GDN gated RMSNorm measured 0.095034 ms versus 0.108738–0.110659 ms. Attention-output gating was effectively neutral at this width. The new K5120/K17408 prefill specializations use 57/73 VGPRs, 40 bytes LDS and zero private scratch or spills. The memory-traffic lower bound drops from two BF16 reads plus one code write (five bytes/value) to one read plus one write (three); this does not establish an HBM-traffic saving because the second baseline read can hit cache.

Whole-inference comparison uses max context 262144, automatic KV, P2048, mixed `auto`, one warmup and three repetitions per invocation, without profiling or concurrent builds. C1 uses ordinary 8192-token prefill; C6 uses fixed DFlash K7, optimized proposal head and an 8192,128 fresh-prompt whole wave per lane. Two invocations per implementation yield six pooled repetitions:

| Scope | Baseline median | Selected median | Observed latency reduction |
|---|---:|---:|---:|
| C1 8192-token prefill wave | 2.375835 s | 2.352935 s | 0.96% |
| C6 8192,128 whole wave | 16.293090 s | 16.188519 s | 0.64% |

All generated tokens, C6 speculative statistics and automatic KV capacities match. C1 baseline/selected ranges were 2.371986–2.396011 / 2.350651–2.366054 s; C6 ranges were 16.258738–16.300633 / 16.159915–16.198480 s. These support a small improvement on these workloads, not a general maximum-performance claim. Paired conversion alone was not a reliable whole-inference gain. A separate balanced comparison removes paired conversion from the otherwise selected producer. Its pooled C1/C6 medians change from 2.370090/16.228088 s to 2.361603/16.187690 s (0.36/0.25% reduction); C1 includes a 2.493601-s candidate outlier. The independent C6 comparison also favors retaining the native paired conversion, without establishing a universal gain.

An isolated C6 decode regression check uses `-pg 8192,128 --isolate-prompt-decode` with the same fixed-K7 settings, balanced order, warmup and repetition count. Pooled decode-service medians were 1.473705 s baseline and 1.470841 s selected (about 0.19% lower), with identical tokens and speculative statistics. This establishes no observed decode regression on the tested workload.

Final-source no-cache residency checks repeat startup plus an adaptive-DFlash 8192,8 whole request per lane at C6/C7/C8. Physical free is sampled every 50 ms with the same HIP allocation probe used above. Minimum free is 65.938/65.152/66.359 MiB; shared KV remains 608576/600448/592256 tokens, observed graphs stay at 62 MiB, and workspace capacity stays 265043968 bytes. The existing 64-MiB external headroom and runtime reservations cover the additional kernel code without a budget change. These instrumented runs are memory checks, not throughput evidence.

Cache-enabled probes use the Compose-equivalent public Engine options (8192 MiB RAM, 25600 MiB SSD) with separate per-concurrency cache directories. C6/C8 retain 605760/589440 shared KV tokens and reach 68.523/68.934 MiB minimum free; observed graphs remain 62 MiB.

Final-source producer and persistent qualifiers pass normally, with guard-page memcheck and allocation-poison initcheck. The producer also passes LDS racecheck; clang-tidy reports zero diagnostics. Persistent racecheck is limited by the existing `PhaseCta::group_sync` protocol: it uses workgroup release/acquire atomics, while checker epochs advance only at hardware workgroup barriers. An instrumented exact-HEAD baseline reproduces all 15 reported source sites, hazard kinds and per-site counts (519120776 total) identically. Neither report names changed encoder code; all 23 numerical scenarios pass in both. This is the documented unsupported synchronization model, not a passing persistent racecheck result. No checker suppression or product synchronization change was made. Independent review found no material implementation, numerical-contract or evidence concern.

The alternative MLP codebook lookup interleaved nibble selectors before table lookup, preserving decoded bytes while shortening the codebook-result dependency. It passed the independent FP64 Linear/SiLU oracle at real T1024/T2048 gate/up and down shapes, but neither plain down nor actual fused SiLU gate/up showed a reliable timing benefit in balanced comparisons. It is discarded.

Numerical evidence includes the existing independent FP64 producer oracle and complete exact code/scale/status checks against the independent E4M3FN encoder at both plain widths. The latter cover zero, NaN, positive/negative infinity and T255/256/257 dispatch boundaries. Temporary baseline comparisons also matched complete fused producer images, with/without cache warming, and persistent-decode qualification passed. Evidence and exact commands are under `profiles/bench/r9700-prefill-followup-20261016/`.

## Shared GDN chunk preparation (2026-10-16)

The follow-up optimization above targets state-independent work inside GDN rather than changing precision, clocks, scheduling or graph submission. For normalized ordinary T>=256, a separate chunk-parallel kernel computes normalized FP16 keys, FP32 query scales and FP32 QK/KK products once per Q/K head, shared by its three value heads. Per-value-head controls, FP16 A/P casts, diagonal solves and FP32 recurrent state stay in the existing recurrence. T16..255 keeps the direct chunk route. Caller-owned transient workspace is explicit in the Op and runtime plan: 18.125 MiB at T2048, reused below the existing MLP workspace peak. This is GDN scratch, not an attention KV-cache plane.

On the same R9700, Release ROCm 10 toolchain, canonical graph flags and unchanged `auto`/300 W profile, independent public FP64 oracle checks passed for ordinary in-place/distinct state, partial chunks, T255/256/257 crossover, snapshot and replay. Maximum output absolute error was 1.114e-5, relative L2 0.001864 and state absolute error 1.527e-5. The same public qualifier's unprofiled event screen gave:

| Ordinary normalized width | Incumbent | Shared preparation | Latency reduction |
|---|---:|---:|---:|
| T1024 | 0.318171 ms | 0.297464 ms | 6.51% |
| T2048 | 0.636317 ms | 0.607608 ms | 4.52% |

The integrated qualifier measured 0.297719/0.607449 ms at these widths. The producer uses 51 VGPRs, 17408 bytes LDS and zero scratch; the prepared recurrence uses 256 VGPRs, 58880 bytes LDS and 12 bytes scratch. ISA inspection found the incumbent's 8-byte scratch was one offset save before the chunk loop and reload afterward, not repeated per-chunk traffic. A lane-contiguous vector-product workspace alternative passed the oracle but gave no timing benefit and was discarded.

Whole-inference admission uses the same workloads and artifact as the attribution section, max context 262144, automatic KV, one warmup and three repetitions per invocation, in incumbent/candidate/candidate/incumbent order. The shared lease covers all invocations, with no build during timing. Pooled medians of six repetitions per implementation are:

| Scope | Incumbent | Shared preparation | Observed latency reduction |
|---|---:|---:|---:|
| C1 ordinary 8192-token prefill service | 2.390364 s | 2.382705 s | 0.32% |
| C6 fixed-K7 8192,128 whole wave | 16.768307 s | 16.748495 s | 0.12% |

These small whole-inference differences have overlapping timing ranges; they establish no robust general speedup. The qualified improvement is at the GDN operator, with C6 whole inference nearly neutral. A 2.910451 s C1 incumbent outlier makes means unsuitable for this comparison. C6 summed prefill-service medians are 15.606773/15.812400 s, but include mixed decode work and are not exclusive prefill GPU time. Eleven repetitions have wave minus summed service near 0.936 s; the first incumbent alone has 1.548 s. Aggregate reports cannot determine the cause of that phase-placement difference, so it does not establish a prefill-kernel regression. All twelve repetitions per concurrency have exact generated-token parity; C6 speculative statistics also match exactly. Workspace capacity/allocator peak remain 265043968/256262144 bytes, and automatic KV capacity is unchanged by this implementation.

Residency admission also exposed a pre-existing physical-budget shortfall with RAM/SSD caching disabled. The incumbent and candidate both resolved C8 KV to 592448 tokens and fell to about 61.6 MiB free, despite configured 64-MiB external headroom. A C1..8 startup plus 8192,8 adaptive whole-inference sweep and physical/HIP phase snapshots separated this from the GDN workspace. At C6/C7/C8, measured aggregate physical consumption beyond logical arenas and the HIP graph observation was 8.744/9.980/about 10.83 MiB versus an 8-MiB reservation. HIP allocation rounding and physical residency differ; the snapshots do not establish an exact driver-allocation cause.

The main-stream runtime bound changes from 6/6/4/4/2/2/2/2 to 6/6/4/4/2/4/4/6 MiB at C1..8, retaining the separate 2-MiB startup component, 4-MiB allocation bound and 2-MiB SSD scatter component. The increases at C6/C7/C8 are the smallest 2-MiB driver-reservation steps covering the observed aggregate deficit. They apply independently of RAM-cache size; two cache configurations do not establish a safe RAM-dependent calibration. The 64-MiB external headroom and 56/56/60/60/ 62/62/62/62-MiB graph table are unchanged. The throughput A/B above predates this independent capacity correction; it isolates the GDN implementation at identical KV capacity.

Corrected no-cache capacity and physical minima are below. C1..5 retain their measured sweep entries; C6..8 are rebuilt reruns after the correction. Every cell uses max context 262144, adaptive DFlash K7, optimized proposal head, P2048/mixed `auto`, no warmup and one 8192,8 whole request per lane. Physical free is sampled every 50 ms over the complete subprocess lifetime; these instrumented checks are not throughput measurements.

| C | Shared KV tokens, no RAM/SSD cache | Minimum physical free MiB |
|---:|---:|---:|
| 1 | 262144 | 9904.613 |
| 2 | 524288 | 2783.805 |
| 3 | 632960 | 71.301 |
| 4 | 624896 | 67.059 |
| 5 | 616768 | 65.539 |
| 6 | 608576 | 66.008 |
| 7 | 600448 | 65.223 |
| 8 | 592256 | 66.434 |

The cache-enabled probe sets public Engine options to 8192 MiB RAM and 25600 MiB SSD, matching Compose. It is compiled from the current benchmark driver with only those Engine options changed; the stock benchmark has no cache-capacity CLI flags. Production C6 resolves 605760 shared KV tokens, with 68.578 MiB minimum physical free; C8 resolves 589440 tokens with 69.012 MiB minimum free. Both retain 62 MiB observed/budgeted graphs. All measured allocator peaks remain 256262144 bytes within 265043968 bytes of workspace. The corrected runtime planner and its explicit C1..8 reservation assertions pass. The C8 probe uses its own fresh cache directory: reusing C6's directory correctly rejects the different cyclic-lane fingerprint. These short physical-capacity reruns do not claim a new long-context throughput result.

Integration after rebasing onto the upstream decode work through `2cbd438c` passes the ordinary GDN FP64 qualifier, deferred fold/record qualifier, runtime planner and explicit reservation assertions. Rebuilt no-cache C6/C8 adaptive 8192,8 runs retain 65.926/66.344 MiB minimum physical free with unchanged 608576/592256 shared KV tokens and 62-MiB graphs. The prefill A/B above remains evidence against its stated `6f069b6c` baseline; it does not measure the separate upstream decode improvements. Integration records are under `profiles/bench/r9700-prefill-followup-20261016/` and the `merged-adaptive-capacity-c6/c8` records in the original evidence directory.

The dominant GEMM hypotheses were also tested against the independent FP64 oracle and whole C1 prefill. Moving weight decode/store earlier than stage 3 passed the oracle but increased mean latency from 2.40010 s to 2.41972–2.44761 s. A 512x128 sixteen-wave CTA passed the oracle but took 2.458/2.468 s versus a stable 2.376 s incumbent. Both were rejected; production GEMM code is unchanged. The HIP API trace above does not justify removing required token/state synchronization or treating overlapping API waits as recoverable CPU time.

The selected GDN route passed the required gfx1201 racecheck, memcheck and initcheck, the runtime planner qualifier, and focused clang-tidy with zero diagnostics. Evidence and exact commands are in `profiles/bench/r9700-prefill-opt-20261016/`; API attribution remains under `profiles/rocprof/r9700-prefill-20261016/`.

## Drafter leftovers after the persistent decode kernel (2026-10-03)

A whole-round C1 kernel trace (DFlash K7 `--lm-head-draft`) puts the drafter and selection at about 2.8 ms of the ~29.4 ms round, with every drafter projection at 620-640 GB/s. Two items outside the bandwidth-bound projections were examined:

- The DFlash2 selector's N256/K5120 hidden projection ran on the generic wmma32 route (16 single-wave CTAs walking K serially): 79 us per round for 0.7 MB. The small-batch route with a 16-wave K split takes 3.8 us (`cb6c7cf7`).
- The 131072-row draft head streams about 356 MB per round (about 600 us). Its rows are stored in corpus-frequency order with the 21 forced special tokens last, so a shorter head was emulated by masking rows [R, 131040) before top-16 (production shape, 11 prompts x 3 p-less T1.5 seeds, calibrated draft temperature, 1024 tokens). Acceptance against the full head: R=98304 0.905 (95% bootstrap 0.866-0.943, median 0.964), R=65536 0.821 (0.755-0.886, median 0.892). The byte saving would be about 0.5% / 1% of the round, so a shorter head loses; the full shortlist stays. Evidence: `profiles/bench/r9700-head-rows-20261003/`, `r9700-selector-proj-20261003/`.

Remaining drafter glue after this study: the standalone A8G64 quantize launches and the column top-k (both removed below), and the path-select chain (29 us, kept; see below).

## Drafter Q4 activation images (2026-10-04)

The drafter's Q4 projections each quantized their BF16 input in a separate launch. Duplicating those launches cost 0.21 ms per 29.1 ms C1 round (about 4.8 us per launch including its gap), which bounds what removing them can save. Producers now publish the exact A8G64 image straight into the serialized activation region and the projections read it (`ops::linear_q4_activation_image`): RMSNorm of layer 0, the grouped-conv prepare, a fused conv-finish + next RMSNorm (out-of-place residual), and the SiLU, plus one shared quantization of the context for the five context-append QKV projections. Launches per round fell 255 -> 212 at C1.

A first version published each image's status word from one extra status CTA that re-evaluated every value; it was 1.1% slower (status CTAs of 18-70 us; the SWA reduce variant also lost its parallel grid). The selected version lets the last finishing block publish the status word through two persistent completion words, and leaves SWA unchanged. Production shape (`-pg 512,512`, DFlash K7 `--lm-head-draft`, 3 ABAB passes x 3 reps): C1 127.21 -> 127.88 tok/s (+0.52%, 29.17 -> 29.01 ms per round, every pass); C4 277.15 -> 278.84 tok/s (+0.6%). Rounds and acceptance are identical to the previous build at both concurrencies. Evidence: `profiles/bench/r9700-drafter-image-20261004/`.

The DFlash2 column top-16 then ran as a 32-way split (33 us: every thread insertion-sorted its 16 logits) plus a merge launch (10 us). One 1024-thread CTA per column now takes the 16th best of its 32 wave maxima as a threshold that at least 16 logits meet, re-reads the column for the few logits at least that good, and ranks them exactly (an exclusion fallback covers more than 2048 ties); results are identical. C1 production shape: 29.027 -> 28.985 ms per round (+0.16%, 127.82 -> 128.00 tok/s, identical rounds and acceptance).

The path-select chain (29 us, one CTA serially staging rows and scoring each hop) was rewritten to score every hop's 16 x 16 predecessor/candidate transitions in parallel (bit-identical choices) and walk the chain over them; the global row gathers did not overlap and the kernel took 43 us, so the serial kernel stays. Its ceiling was about 20 us (0.07% of the round). The one-launch top-k measures 18 us (two passes over each 256 KiB column on one CU), not the ~5 us bandwidth floor.

## Online p-less draft-temperature calibration (2026-10-03)

Upstream `24d527d1` scores eight candidate draft temperatures on every p-less chain round from the verified target law and the recorded selector proposal, and drafts the next round at the best predicted temperature for its K ([model §8](maintainer/qwen3.8-27b-model.md#8-dflash2-block-diffusion-draft-model)). Production shape: C1 CLI, DFlash K7 adaptive `--lm-head-draft`, p-less T1.5, thinking on, 1024 tokens, 11 scenario prompts x 3 seeds, pinned 0.4 (the previous default) against calibrated, same binary, ABBA passes. Each configuration is deterministic per seed (the two passes are identical), so the comparison has 33 independent cases; per-case ratios scatter (0.81-1.20) because the sampled trajectories diverge:

| calibrated / pinned 0.4 | mean (95% bootstrap) | median |
|---|---:|---:|
| acceptance (tokens per round) | 1.033 (1.004-1.061) | 1.040 |
| decode tok/s | 1.024 (0.987-1.058) | 1.044 |
| ms per round | 1.000 | |

The scoring kernel does not change the round time (30.16 vs 30.18 ms). Evidence: `profiles/bench/r9700-pless-calibration-20261003/`.

## Parallel p-less tile choice and KV-tier sync check (2026-10-03)

Upstream `91371f42` made the p-less tile choice block-cooperative: a wave32 chunk scan gives each thread the prefix interval of its contiguous tile chunk, and the owning thread walks only its chunk, replacing thread 0's serial walks over the vocabulary tiles in the ordinary sampler and the speculative finalize kernel (residual masses and the admitted sum are block-parallel too). C1 CLI, DFlash K7 `--lm-head-draft`, seeded p-less T1.5, 512 tokens, 3 prompts x 2 seeds x 2 ABBA passes; the baseline is the same tree with only the three sampling headers reverted:

| | before | after |
|---|---:|---:|
| decode, mean of 12 runs | 91.4 tok/s | 102.1 tok/s |
| round, median | 32.15 ms | 29.41 ms |

Every prompt/seed pair runs the same number of rounds in both builds (same sampled trajectories), and the new build's run-to-run spread is much narrower.

Concurrent serve, same two builds: one server at `--max-concurrency 8`, DFlash K7 `--lm-head-draft` (fixed K, so seeded p-less trajectories pair across builds), T1.5, 512-token streaming requests over 8 rotating prompts, 4 waves per C in each of 2 ABBA passes. Aggregate decode = generated tokens / (last finish - first token):

| C | before (tok/s) | after (tok/s) | paired median | wins |
|---:|---:|---:|---:|---:|
| 1 | 91.8 | 99.5 | +8.4% | 8/8 |
| 2 | 157.5 | 168.7 | +8.1% | 8/8 |
| 4 | 242.5 | 260.3 | +8.0% | 7/8 |
| 8 | 376.5 | 397.6 | +6.2% | 8/8 |

With `--adaptive-draft` the live K follows round time, trajectories diverge, and two passes per build were too noisy to resolve C1 (C2/C4/C8 +14/+7/+6%).

The `4bf04efd` KV-tier port (RAM claims no longer wait for a spill of the same entry, startup checkpoint slab, two-phase disk eviction, request-local restore failures) was checked with the 2026-10-01 churn harness (C3 serve, Compose tiers, DFlash K7 adaptive, six ~12K-token conversations x three passes; `profiles/bench/r9700-upstream-sync-20261003/`), ABBA against the pre-sync build. All 18 greedy replies match in both builds and the 2026-10-01 run; cold 12K TTFT is about 4.0 s, RAM loads 125-507 ms, and the largest decode gap 0.89/1.51 s (sync) against 2.36/0.91 s (pre-sync), each one outlier. The pre-sync build already carried the fork's stall fixes that upstream ported in the same commit, so this run is a no-regression check, not a speedup claim. A first sync run during a concurrent 8-job host build showed 5-7 s cold TTFT and one 5.6 s RAM restore; neither recurred on a quiet host, nor in ABBA reruns of both builds under 8 CPU-spinning processes or a looping clean `-j 8` build (largest decode gap 0.34-0.37 s sync, 0.36-0.55 s pre-sync; RAM loads 122-274 ms; cold TTFT about 4.0 s; identical replies). The cause of that single event is not established.

## Adaptive draft picker and p-less draft temperature per K (2026-10-02)

Upstream `fd16c8ba` replaced its picker with learned per-k hop hazards plus one exploration round in 32, and scaled the p-less draft temperature down at k=6/7. Measured here before porting: production serving shape (`--temperature 1.5`, DFlash `--lm-head-draft`, Device Graphs), 12 manifest prompts x greedy + two p-less seeds x 768 tokens, paired per-request decode rate with 95% bootstrap, alternating arm order per server lifetime.

| | C1 | C4 |
|---|---|---|
| fixed K7 vs adaptive K7 (2 lifetimes) | 1.035 (1.018-1.052) | 1.043 (1.023-1.066) |
| fixed K7 vs adaptive K7 (4 more lifetimes) | 0.999 (0.997-1.001) | 1.001 (0.987-1.016) |
| adaptive K7 + exploration vs adaptive K7 | 0.993 (0.986-0.999) | 0.998 (0.983-1.013) |
| p-less only: K7 draft T 0.3 vs 0.4 | 0.991 (0.965-1.017) | 1.018 (0.996-1.041) |
| p-less only: K6 draft T 0.35 vs 0.4 | 0.989 (0.966-1.010) | 1.005 (0.984-1.025) |

Fixed K7 is the best fixed k for every prompt at both C, and adaptive K7 matches it, except that one of six adaptive server lifetimes locked K6 for its whole life (5.98 drafted per round, 30.5 vs 29.9 ms per round, ~6% slower): T(k) is a running mean refreshed only by rounds at k, so a high early K7 sample is never corrected. Upstream-style exploration removes the lock in a closed-loop model but costs 0.7% at C1 on every lifetime (explored rounds pull the locked k to ~6.3 drafted per round), more than the lock's expected cost, so it is not adopted; neither is the hazard model it serves. The scaled draft temperatures are within noise of 0.4, which was tuned with K7 adaptive. Evidence: `profiles/bench/r9700-sync-open-items-20261002/`.

## Persistent decode kernel (2026-10-02)

Every single-sequence (`B=1`) decode graph is lowered after capture (`ops::persistent_decode_lower`): each run of consecutive hosted launches (the LQ4 and row-scaled FP8 small-T projections, the FP8 activation producers, the GDN front, pair projection with convolution and record, and Q/K norm-RoPE) becomes one cooperative persistent kernel in an owned child graph, with two 384-thread blocks per WGP. Each launch is a phase of virtual CTAs running the launch's own kernel body, separated by an atomic grid barrier (1.36 us against ~3 us per graph kernel). Outputs are bitwise those of the kernel nodes. A C1 K7 round runs 578 phases in 25 persistent kernels; verify attention, the drafter and the sampling tail stay kernel nodes. `B>1` definitions keep their kernel nodes (below). The grid barrier needs all blocks co-resident. Occupancy is checked at lowering, and replay uses actual cooperative dispatch plus fixed prior-transfer event dependencies; occupancy alone is insufficient. The GPU is not shared with another resident persistent workload.

What made phases as fast as their kernels:

| Change | C1 K7 ms/round (lowered) |
|---|---|
| First correct lowering (out-of-line bodies) | 33.6 |
| Inlined bodies, global-address-space arguments (no FLAT), one-pass front | 32.6 |
| Next-phase weight stream at the barrier, one wave per waiting block (replaces warm CTAs) | 32.1 |
| GDN pair convolution one thread per (row, token) | 31.2 |
| Next-phase instruction prefetch at the barrier (`s_prefetch_inst`, 128 KiB) | 30.07 |

The instruction prefetch was worth ~1 ms per round: every phase runs a different body of a ~0.3 MB kernel while the weight stream evicts its code from L2 (6.5x the hosted kernels' instruction-cache misses before it). The barrier stream is time-bounded (it stops when the barrier opens) and rate-bounded: wider streams queued the latency-bound phases' own loads (16 / 32 / 64 / 128 / 384 threads: 32.04 / 31.50 / 32.15 / 33.30 / 33.8 ms before the instruction prefetch); non-temporal loads did not help. Not adopted: three blocks per WGP (168 VGPRs; the front spills), four-step projection load batches (+0.2 ms), wider prefetch for the pair after the front.

Same-session A/B, `ninfer_bench` P512/G256, C1, kernel-node graphs (lowering disabled in the same build) vs lowered, interleaved:

| Mode | Kernel nodes | Lowered | Accepted tokens |
|---|---|---|---|
| DFlash K7 `--lm-head-draft` | 30.33 / 30.37 ms/round | 30.04 / 30.10 ms/round (-1.0%) | 519 / 3.0843 both |
| MTP K3 | 78.32 / 77.22 tok/s | 78.59 / 78.60 tok/s | 480 / 2.6667 both |
| No draft | 38.27 / 37.76 tok/s | 38.22 / 37.86 tok/s | — |

Greedy token ids are identical with and without lowering (3 prompts x DFlash K7, MTP K3, no draft, 384 tokens) and so is seeded p-less sampling at fixed K7; with `--adaptive-draft` the live K follows measured round time, so a faster round may pick another K and realize another sample of the same distribution. Per-body bitwise identity: `persistent_decode_qual`.


The 2026-10-19 cache-pressure failure left 63 of 64 blocks waiting at the first phase barrier while block 63 never entered the kernel. A coherent entry-count probe (zero private segment, the original 238-VGPR resource profile) found exactly one incomplete image among 319 used images; every other image completed with matching block entry counts. Ordinary launch, classic graph submission, cooperative launch alone, and a 63-block grid each reproduced the stall. The evidence identifies a resident-grid forward-progress failure, not a proven firmware defect. Actual cooperative child-graph capture and ordering B1 replay after prior load/copy/disk-scatter work fix the submitted-work contract; B>1 and mixed rounds retain overlap. Startup reserves an additional 20 MiB of cooperative graph residency (C6 observed 80 MiB against its previous 62-MiB allowance).

A one-block-per-WGP grid was also tested without changing the compiler resource budget. Both it and a 64-block host-copy-fence control completed four concurrent approximately 93K–178K tool prompts over three turns, C6 adaptive K7, 12-GiB RAM/25-GiB SSD tiers, context 262144, with zero errors or stall snapshots: 531.4 and 529.9 seconds including startup, respectively. Computed prefill was 772553 versus 770504 tokens. These single full-fixture runs establish no material cache-pressure penalty; they are not a general throughput comparison.

The complete event-based 64-block schedule retains higher B1 decode throughput: C1 `pp8192+tg256`, adaptive optimized-head DFlash K7, context 16384, workload KV capacity, prefill2048, one warmup and three repetitions measured 186.31 ±0.31 output tok/s versus 172.49 ±0.09 for the 32-block grid. Accepted drafts (633), rounds (135), and per-position acceptance counts matched. The 64-block event schedule is selected. Both graph submission modes pass all 23 exact captured-versus-lowered scenarios. Target/toolchain: R9700 gfx1201, HIP 7.15.26333/clang23 under `/opt/rocm/core-10.0`; production R9700Lq4 artifact. Commands, benchmark reports, entry probes, and full-workload logs are under `profiles/diagnostics/r9700-stall-20261019/`.


Final event-based C6 qualification completed all 12 valid tool responses in each of the burst and thinking/automatic-tool/streaming history workloads, with zero errors or stall snapshots. Burst recorded 107 decode rounds/111 row-rounds; history recorded 120/171. Each had seven reused requests and six RAM restores; neither recorded an SSD restore. The full disk-cache suite separately passes exact restoration and deterministic independent-stream checks that a submitted-scatter join waits for held GPU work but does not wait for held future SSD reads, with both uncompressed and Zstd storage. Ordinary and native MTP generation, C8 DFlash startup/generation, and the focused host/static checks also pass. Independent Astra review approved the supported single-owner paths with no known related blocking issue remaining.

Production configuration (CLI, DFlash adaptive K7 `--lm-head-draft`, C1, 11 manifest scenario prompts x {greedy, two p-less T1.5 seeds}, 768 tokens, two ABBA passes, 66 pairs; `26dddc00` kernel-node graphs vs `5d018ccf` lowered, 95% bootstrap): round time 1.3% shorter (1.0131, 1.0115–1.0145; 31.89 -> 31.48 ms/round), decode rate 1.025 (1.012–1.045; 137.1 -> 140.2 tok/s). All 22 greedy pairs produce identical output; 18 of 44 seeded pairs differ because adaptive K follows the measured round time.

The GDN record runs one whole-block virtual CTA per (sequence, head) with three value rows per row slot (each token staged once; the launch's four 256-thread row tiles fit one per block), and a phase's partial last round spreads over all blocks (group-major virtual CTAs): the C1 record phase 14.5 -> 11.7 us and the round 29.79-29.88 -> 29.64-29.70 ms (fast-mode passes, same accepted tokens). At C1 a persistent kernel reaches its first phase 0.7 us after it starts and its first phase runs as fast as a steady one, so the remaining unhosted nodes (verify attention, the drafter, sampling; ~150 per round) cost only their ~0.25 ms of launch gaps.

Phase time is not the objective on its own: placing the GDN front's fold CTAs on WGPs of their own (blocks b and b + 32 share one) cut the front 19.0 -> 15.3 us but lengthened the pair projection after it 78.9 -> 85.4 us, whose weights the waiting blocks had been streaming at the barrier, and the round by 0.1 ms; not adopted. The large projections already stream at about 630-650 GB/s. Shortening the FP8 activation quantization (one register-cached pass, one reduction barrier) was likewise neutral end to end.

What helped instead: a phase that streams no weights (quantization, GDN front and record, RoPE) prefetches at its barrier the next weight-streaming phase's lines, so a projection's tail and the small phases after it stream the coming projection: C1 DFlash K7 29.64-29.73 -> 29.45-29.55 ms/round, MTP K3 33.61 -> 33.45-33.47 ms/round (same accepted tokens). With that, two prefetch waves per waiting block beat one: DFlash 29.21-29.23 ms/round (three waves 29.38, four 29.60), MTP K3 33.22-33.24.

Multi-sequence (`B>1`) lowering, not adopted. Lowered as-is, C4 and C8 DFlash K7 decoded 6.6% and 5.5% slower: a C4 verification is 32 columns, whose two-token-tile projections were not hosted, so a round became 47 persistent kernels of about two phases each. Hosting the two-tile LQ4/FP8 projections, the separately projected GDN convolution record and a per-head GDN record (all bitwise, `persistent_decode_qual` with batched GDN layers) gave 72 kernels / 577 phases per C4 round, still 2.7% slower (45.25 vs 44.03 ms/round); C2 decoded equally (35.42 vs 35.45) and C8 within its run-to-run noise (58.7-63.4 vs 57.7-62.2). Per-phase timestamps attributed the rest to wide phases: the kernel runs at most six waves per SIMD (its 233-VGPR budget), where the 32-column GDN front (1.8x), convolution record (1.6x) and two-tile SiLU pair lose occupancy against their own launches, while a barrier saves only ~0.6 us of the ~1.6 us launch gap and every further persistent kernel starts ~8 us slower than a steady phase.

Removing latency phases, not adopted (C1 DFlash K7, interleaved with HEAD, same accepted tokens). Running a phase twice bounded what removing one could save: about 10 us per extra gated-RMSNorm or activation quantization phase and 15 us per extra GDN record phase. Each was then tried, all bitwise against the captured launches (`persistent_decode_qual`):
- GDN record merged with its gated RMSNorm quantization (each head CTA normalizes its own vectors; the row maximum and nonfinite flag merge over the 48 head CTAs by atomics and an in-phase arrival count; 47 merged phases per round): -0.01 to -0.07 ms/round over six pass pairs, ~0.1%, not worth its merge protocol.
- The MLP activation quantization over all 64 blocks (one register-held vector per thread, the same merge): +0.12 to +0.22 ms/round. The row CTAs used 8 blocks; the other 56 spent the phase streaming the down projection at the barrier.
- Prefetching the record's initial-slot recurrent state (64 KiB per head) at the barrier before it, instead of the output projection's lines: +0.03 to +0.04 ms/round.

So the barrier-time stream already absorbs most of a short phase's cost.

Evidence: `profiles/bench/r9700-megakernel-20261002/` (`ab*`, `trace*`, `tokens/`, `corpus/`, `abB`/`abX`/`abY`/`abZ`, `b4trace/`, `abdup/`, `m1-*`..`m4-*`).

The 2026-10-01 feasibility bound (96 phases, empty phase 0.95-1.0 us against 3.3 us per graph kernel; ~1.3 ms per round before costs; `profiles/bench/r9700-decode-c1-20261001/persistent_barrier.hip`, `persistent_gemm_chain.hip`) held as an upper bound; projections are HBM-bound (the SiLU pair streams at ~628 GB/s), so shorter latency-bound phases mostly move bytes into the following projection rather than remove them.

## Deferred GDN replay fold (2026-10-01)

The eager ReplaySSM fold after each DFlash round streamed every GDN layer's 3 MB state in and out (0.52 ms at C1, ~570 GB/s) and its dirty lines landed on the next round's first kernels. A continuing chain row now defers it: the next round's verify forward folds layer L ahead of layer L's GDN front, as extra CTAs of the FP8 front kernel (four state tiles per thread, sharing the key normalization), and only lanes leaving the batch, tree rounds and state-reading Program entries fold eagerly. State and history are bitwise the all-layer fold (`gdn_replay_fold_qual`, standalone per-layer and front-fused), and accepted tokens match HEAD at C1 (519) and C4 (2055).

The fold is bandwidth-bound, so deferral cannot hide its traffic: the front grows from 9.0 to 22.4 us per layer (the fold's own ~10 us of state streaming plus the front), against the 0.52 ms eager launch it replaces. Preloading the per-column records moved nothing; four tiles per thread (77 -> 20 fold CTAs per row) only 23.1 -> 22.4 us. Same-session A/B against HEAD, P512/G256 K7: C1 30.39 -> 30.33 ms per round (three interleaved passes, within noise), C4 268.0 / 269.5 -> 271.4 / 274.7 tok/s (+1.3 / +1.9%). Evidence: `profiles/bench/r9700-decode-c1-20261001/deferred/`.

## Decode cache warming (2026-10-01)

C1 DFlash K7 profile (`ninfer_bench` P512/G256, kernel trace): 30.9 ms per round, 802 kernels, 29.3 ms kernel time. The round's weight bytes (~15.4 GB) stream in ~24 ms at the 636 GB/s read peak; the rest is projection ramp/tail (~2 ms), latency-bound kernels (~2.7 ms: quantize producers, GDN front and record, verify attention) and launch boundaries. An empty kernel costs 3.05 us inside a Device Graph on gfx1201, unchanged by `HIP_FORCE_DEV_KERNARG`, `ROC_USE_FGS_KERNARG`, `DEBUG_HIP_GRAPH_*` or direct-doorbell settings; a grid of about 2016-2048 waves costs ~34 us (empty 256-thread grids of 253..256 CTAs).

(Lowered `B=1` graphs drop these warm CTAs; see Persistent decode kernel.) Latency-bound kernels now carry warm CTAs past their work grid that touch one dword per 256-byte line of the next projection's head (`core/cache_warm.h`), so it starts from L2/Infinity Cache: the FP8 activation producers warm 2 MiB of their consumer (codes plus group codes), the GDN normalized front 4 MiB of the GDN pair projection, and the GDN recurrence (record and snapshot) 4 MiB of the output projection after its producer's share; prefill widths (T > 128) warm nothing. Results are bit-identical (the producer, front and recurrence qualifications run with warm CTAs; accepted tokens match). C1 K7, three interleaved passes: 30.87 -> 30.32 ms per round (99.9 -> 101.7 tok/s). A producer's warm only pays for the time it overlaps: 4 MiB stretched each producer by as much as it saved in the projection.

Rejected: running each activation producer inside its projection kernel (the first T CTAs produce the image; row CTAs issue their first weight batch, then wait on a self-clearing arrival counter). It removed 196 of 802 kernels per round and was bitwise identical, but a 128-thread producer CTA is slower than the row-wide producer kernel and the waiting row CTAs leave DRAM idle meanwhile: C1 K7 31.27 ms per round (vs 30.32), normalized gate/up T8 164.8 vs 161.8 us per layer, 171.6 with the waiting CTAs touching their own slice; a synthetic spin producer had suggested 3-5 us saved per pair. Also rejected: warming from a parallel graph branch (fork/join made a layer 4x slower); `hipExtAnyOrderLaunch` consumer/producer overlap, which works eagerly (a 5.6 us producer fully hidden) but is dropped by graph capture; gfx1201 reports no dynamic data-prefetch regions. Evidence: `profiles/bench/r9700-decode-c1-20261001/`.

## Prefill GEMM token spans: chunks above 2048 (2026-10-01)

One LQ4 / row-scaled FP8 prefill GEMM launch over more than ~2048 tokens lost its activation image from cache: its CTAs walk every token tile of a row block together, so each row block re-streamed the whole image from DRAM. Per token (`lq4_linear_qual --time-cell`, now allowed to T8192), gate/up N34816 K5120 cost 1.75 us at T2048, 2.75 at T2560, 4.36 at T4096 and 5.55 at T8192; N12288 0.62 -> 1.90; the narrow MLP down (N5120 K17408) stayed flat to T4096 and reached 1.49 at T8192. The `--time-cell` cap of T2048 had hidden it, and the CLI/serve/bench default chunk 4096 ran 38% slower than chunk 2048. The GEMM now launches 2048-token spans for wide outputs and 4096-token spans for narrow ones; per-token cost is flat to T8192 (gate/up T4096 17.9 -> 7.1 ms, T8192 45.4 -> 14.2 ms). Whole prefill, production DFlash adaptive K7, C1 (`profiles/bench/r9700-prefill-curve-20261001/`):

| Prompt / chunk | Before | After |
|---|---|---|
| 3072 / 4096 | 1185 ms (2,592 tok/s) | 876 ms (3,506 tok/s) |
| 4096 / 4096 | 1908 ms (2,147 tok/s) | 1174 ms (3,487 tok/s) |
| 8192 / 2048 | 2399 ms (3,414 tok/s) | 2402 ms (3,410 tok/s) |
| 8192 / 4096 | 3857 ms (2,124 tok/s) | 2405 ms (3,406 tok/s) |
| 8192 / 8192 | — | 2419 ms (3,387 tok/s) |

Chunk size no longer moves prefill: 32K 2,949 / 2,949 tok/s and 128K 1,916 / 1,915 tok/s at chunk 2048 / 4096 (8K within 1%). Chunk 2048 needs half the workspace (253 against 505 MiB), so it is now the default for the CLI, server, benchmark and PPL tools as well as compose. Below T2048 the curve keeps the 256-token tile steps (T160 74 ms against T128 50 ms and T256 88 ms); a separate remainder launch would re-stream the ~14 GB of weights and costs about as much as the padding it saves.

## C=1..8: verify-width kernels for C5..C8 (2026-10-01)

`kMaximumConcurrency` is 8. DFlash verify rounds at C5..C8 run the Text and drafter projections at T = B x W = 17..64 columns (B <= 8 lanes, W = 4..8), widths the C2..C4 routes never reached. A C8 contention trace (7 decode lanes, 8K owner, production DFlash adaptive K7, `prof.sh`) attributed the regressions, and each fix was qualified against its FP64 oracle before adoption (unprofiled interleaved medians, `profiles/bench/r9700-c8-20260930/`):

| Change | Before -> after |
|---|---|
| Drafter SWA: always split for long windows (direct above 48 columns) | T8xB8, 4096 window: 3.30 -> 0.36 ms per call |
| LQ4 mid-T loop drained every load per step; deep-pipelined variant at T33..64 (load ring, LDS-only barriers), attention and GDN pairs fused into one launch | attention pair T56 94 -> 66 us; GDN pair 103 -> 74 us; N5120 K17408 106 -> 88 us; K6144 43 -> 36 us; SiLU and head at T49..64 -3 to -4% |
| A8Q4 token-tile CTA (exact ceil(T/16) token fragments, LDS-shared activations, in-CTA split-K, two groups in flight) for the drafter shapes and draft head at T17..64 | N5120/K17408 T56 270 -> 115 us; N34816 T56 350 -> 193 us; draft head N131072 (was WMMA32) T56 2500 -> 709 us; C2..C4 widths T18..32 0.53-0.94x |

Short prompts and prefill tails (T65..128) took the latency-bound mid-T loop until the deep kernel was extended to eight token tiles (2026-10-01 sweep, every candidate oracle-checked): attention pair T128 171 -> 115 us, GDN pair (8 x 1, depth 4) 161 -> 120 us, N5120 K17408 166 -> 149 us, SiLU gate/up 294 -> 269 us. The prefill CTA stays slower below T129 (1.2-2.4x mid-T). Whole-inference prefill, production DFlash adaptive K7, C1, ten repetitions x two interleaved runs against be26d62b (`widet/short_ab.sh`):

| Prompt | 48 | 64 | 72 | 96 | 112 | 128 | 200 |
|---|---|---|---|---|---|---|---|
| Before (ms) | 34.8 | 36.3 | 43.5 | 47.1 | 52.7 | 57.3 | 81.3 |
| After (ms) | 34.8 | 36.3 | 38.8 | 42.5 | 47.7 | 52.0 | 81.6 |

Rejected: three- and four-tile small-T LQ4 (8-38% slower than mid-T at T>=48, L2 activation re-reads), and a fused wide GDN normalized front at T33..64 (113 us against 17 us for the three-launch composition).

Steady-state C8 decode round (rocprof, contention run): 62.1 -> 50.8 ms wall. Per call, the verify-width GEMMs now match their C4 cost (N5120 down/out 63.0 against 62.6 us, GDN pair 102 against 101 us); the SiLU gate/up takes 175 us at T64 against 171.5 us at T32 (C4), both 86-87% of its T1 streaming time (150 us); none of the candidate CTA shapes (4 x 2 / 8 x 1 / 2 x 2 / 4 x 1 row blocks x K splits, depth 2..4) was faster. Costs that grow with lanes are per-lane state traffic (GDN replay fold 1.7 -> 4.2 ms, record 34 -> 64 us, verify attention 20 -> 42 us per call at 3 -> 7 lanes).

Whole inference (`-pg 512,512`, DFlash adaptive K7, two runs): C4 268.6 / 269.3 -> 273.4 / 277.2 tok/s against 4a9382b5; C8 411 / 422 tok/s. 8K PPL mean NLL 1.8803156903 against 1.8803155915: the scorer evaluates the output head over 64-token slices, which now take the deep head route. ctest 91/91.

## FP8 protections reduced to 21 against the 5090 NVFP4 build (2026-09-30)

The admission bar is now "no worse than the 5090 NVFP4 build", measured directly: the frozen 5090 scorer (`r9700-nvfp4-multitext-pareto-20260922/bin/nvidia-ppl`, standard NVFP4 artifact and NVFP4 cache) on the same BF16-source cells. NVFP4 vs BF16, dNLL nats/token:

| Cell | NVFP4 | 26-protection production | 21 protections | 21 minus NVFP4 |
|---|---:|---:|---:|---:|
| 8K prefill | +0.0355 | +0.0129 | +0.0146 | -0.0209 +/- 0.0057 |
| 8K decode | +0.0159 | +0.0122 | +0.0134 | -0.0025 +/- 0.0048 |
| 32K prefill | +0.0227 | +0.0113 | +0.0116 | -0.0111 +/- 0.0029 |
| 32K decode | +0.0150 | +0.0099 | +0.0119 | -0.0031 +/- 0.0025 |

NVFP4's prefill is much further from BF16 than its decode, so the usable margin is the decode one, about 0.004-0.005 nats/token for the 26-protection artifact. On the three 4K fixture texts (prefill) NVFP4 is +0.044, production +0.015.

The 26 protections were the NVFP4 reference's BF16 set (15 in the split layout) plus 11 from an unqualified NVIDIA selective-FP8 experiment. Of those 11, attention query/key and gate/value at layers 27/31/51 stay FP8 (protected query/key is not demoted, and a split FP8/LQ4 pair loses the fused input-projection route); attention output 11 and MLP gate/up and down at layers 62/63 become GPTQ LQ4. File 17.02 -> 16.75 GB.

Paired against production: +0.0017 / +0.0012 / +0.0003 / +0.0019 +/- 0.001-0.002 (cells as above); new severe 4/3/14/15 within caps 5/5/17/17.

NIAH: standard and multikey greedy 20/20 each at 8K-128K, 5/5 each at 240K, multikey 5/5 at 260K (261,131-token prompts), sampled multikey 50/50 (8K) and 45/45 (32K-128K). Graph and eager greedy tokens are identical at C1..C4 P2048/G256 and C1 P20480, ordinary and DFlash; greedy DFlash differs from ordinary decode at C1 after a near tie in both artifacts (production at token 12 of the P20480 cell), not at C2..C4.

Speed, interleaved against production: ordinary decode 37.00 -> 37.62 tok/s (+1.7%), prefill +0.4%, DFlash round 32.15 -> 31.98 ms; production serving shape (p-less T1.5, K7 adaptive, 13-prompt study workload x 3 seeds) C1 90.8 -> 92.9 tok/s (+2.3%), C4 205.6 -> 205.7 tok/s. DFlash acceptance over ten fixed prompts 4.07 -> 4.25 tokens/round (greedy trajectories diverge). Removing the remaining non-query/key protections (attention output 3/7, GDN output 4: 44 MB) is below measurement resolution and was not pursued. Evidence: `profiles/ppl/r9700-nvfp4-budget-20260929/`, `profiles/ppl/r9700-lq4-p21-20260929/`, `profiles/bench/r9700-lq4-p21-20260929/`.

**3-bit MLP gate/up (rejected, PPL-costing).** A sign + 2-bit format (LQ4 book positions {1, 3, 5, 7}, the lowest-error 4-of-8 subset on sampled gate/up rows: relative L2 1.9x, calibrated output error ~3.6x the 4-bit codes) was measured exactly by encoding it as restricted LQ4 words (GPTQ, damping 0.1) on the 21-protection artifact. It would save ~22 MB per layer per step (~36 us per round at 620 GB/s; no 3-bit kernel was built). dNLL vs BF16, and minus NVFP4:

| gate/up layers at 3 bits | 8K prefill | 8K decode | 32K prefill | 32K decode | new severe over cap |
|---|---:|---:|---:|---:|---|
| none (21 protections) | +0.0146 (-0.021) | +0.0134 (-0.003) | +0.0116 (-0.011) | +0.0119 (-0.003) | none |
| 12 mid (0,1,28,34-36,38,40,48-51) | +0.0209 (-0.015) | +0.0184 (+0.003) | +0.0142 (-0.009) | +0.0142 (-0.001) | 32K prefill 22/17 |
| 12 late (52-63) | +0.0206 (-0.015) | +0.0191 (+0.003) | +0.0181 (-0.005) | +0.0184 (+0.003) | 3 of 4 cells |
| all 64 | +0.0302 (-0.005) | +0.0267 (+0.011) | +0.0266 (+0.004) | +0.0262 (+0.011) | all 4 cells |

Every subset puts decode behind NVFP4 or breaks a severe cap, so no 3-bit set is admissible under the NVFP4 bar. Per-layer calibrated error ranks the late layers cheapest, yet they carry about half of the all-layer NLL cost; the proxy does not predict model sensitivity. Evidence: `profiles/ppl/r9700-lq4-lut3-20260930/` (receipts with per-layer errors; the evaluation-only encoder option is kept there as a patch).

## Mixed prefill/decode frontier (2026-09-29)

C4 contention bench: 3 lanes decode with 6144-token outputs while the 4th lane runs three fresh 8,192-token prefills (`ninfer_bench --contention 8192,3`), with production DFlash (K7 adaptive, LM-head draft) and prefill chunk 2048. **Decode share** is the decode rounds/s during the prefills divided by the mean decode-only rate of 4 s windows before and after (acceptance drifts along the greedy corpus). Aggregate decode is that share times the median decode-only rate (215 tok/s). **Stall** is the longest decode gap inserted by one prefill step or mixed round. S is the owner's slice (the flag of these runs, `--prefill-slice`), and D is the slices' decode-round cadence. The current flags are `--mixed-forward N`, the forward width N = S + (C-1)·W (W=8 here), and `--mixed-forward-rounds D`.

Pareto frontier after the small-T prefill work below (sweep `fused6`; time-share rows from `ts3`; aggregate decode at the 222 tok/s decode-only reference; stall is the mixed round time):

| Policy | S | D | Decode share | Aggregate decode tok/s | Prefill tok/s | Stall |
|---|---:|---:|---:|---:|---:|---:|
| Prefill-first (default) | 0 | – | 0.016 | 4 | 3,319 | whole prompt (2.5 s) |
| Mixed | 2024 | 1 | 0.076 | 17 | 3,174 | ~0.52 s |
| Mixed | 1000 | 1 | 0.132 | 29 | 3,044 | ~0.30 s |
| Mixed | 488 | 1 | 0.215 | 48 | 2,736 | ~0.18 s |
| Mixed | 1000 | 2 | 0.221 | 49 | 2,705 | ~0.30 s |
| Mixed | 1000 | 4 | 0.358 | 79 | 2,255 | ~0.30 s |
| Mixed | 232 | 1 | 0.375 | 83 | 2,157 | ~0.11 s |
| Mixed | 488 | 4 | 0.527 | 117 | 1,670 | ~0.18 s |
| Mixed | 232 | 2 | 0.535 | 119 | 1,589 | ~0.11 s |
| Mixed | 104 | 1 | 0.537 | 119 | 1,442 | ~0.07 s |
| Mixed | 488 | 8 | 0.696 | 155 | 1,100 | ~0.18 s |
| Mixed | 232 | 4 | 0.709 | 157 | 1,029 | ~0.11 s |

**Score** (prefill tok/s ÷ 3,319 + decode share; time-sharing ≈ 1.0) from the D=1 sweep `fused7`. The other slices are the remaining 256-aligned widths: S + 24 verify columns = 768 / 1280 / 1536.

| S | 488 | 744 | 1000 | 1256 | 1512 | 2024 (`fused6`) |
|---|---:|---:|---:|---:|---:|---:|
| Decode share | 0.215 | 0.165 | 0.132 | 0.103 | 0.090 | 0.076 |
| Prefill tok/s | 2,736 | 2,871 | 3,040 | 3,049 | 3,116 | 3,174 |
| Score | 1.039 | 1.030 | **1.048** | 1.022 | 1.029 | 1.032 |

S=1000 at D=1 has the highest score (1.048 in `fused6` and again in `fused7`). D>1 always lowers the score. Its extra rounds are plain graph decode rounds at decode-only speed (39.7 ms against 39.4 ms), so they move the operating point along the time-share line.

The scores do not follow per-width prefill cost; they also depend on how the 8,192-token prompts split into slices (8 × 1000 + 192, but 11 × 744 + 8). Whole-model prefill runs at 3,414–3,483 tok/s at every 256-multiple width from 768 to 2048. Other widths lose 3–5% (T896 3,311 and T1152 3,279 tok/s).

Trace attribution (`prof/mixed232`) puts a mixed round's decode side at ~16–20 ms against a 39.4 ms decode round:
- drafter 6.2 ms;
- GDN fold 1.7 ms;
- verify head and sampling 1.7 ms;
- verify-column GDN record, per-lane attention and copies ~3.7 ms;
- the 24 GEMM columns ~3.6 ms.

Costs that decode rounds share barely change the score, because the decode-only reference improves with them. They still raise absolute decode tok/s. The mixed round is not host-bound: the submitting thread peaks at 19% of a core with the GPU 85–100% busy. Eager and graph dispatch cost the same per kernel, so capturing the mixed round as a graph gains nothing.

Later mixed-round changes (interleaved A/B, pinned binary, two rounds each; `fsb_*` baseline, `fsn_*` new):
- **Adopted: the first slice joins a mixed round.** Admission with decode-ready requests no longer runs the first slice as its own step, and the owner's KV commit moved from between the last layer and the tails to the end of the round. Each 8K prompt gets one more mixed round.

  | S | Score before (2 runs) | Score after (2 runs) |
  |---|---|---|
  | 1000 | 1.045 / 1.042 | 1.057 / 1.049 |
  | 488 | 1.039 / 1.033 | 1.043 / 1.044 |
  | 232 | 1.027 / 1.019 | 1.030 / 1.023 |

  The prefilling request's greedy tokens stay bit-identical. The co-running decode lane diverges at a near-tie (token 94–100), because one more of its rounds runs at mixed width. For prompts no longer than S, which were never mixed before, the decode stall of one standalone slice (~0.3 s at S=1000) is gone.
- **Rejected: the owner's DFlash context append on a second stream**, overlapping the next round's drafter. Greedy outputs were identical, but the whole contention window changed by −0.4% at S=1000 (noise), +0.5% at S=488 and +1.8% at S=232.

Later bitwise changes (pinned binaries; 8K PPL mean NLL 1.8781831196376255 unchanged; greedy pair checks at S=1000/488 identical; score A/B at C4 with R=6):
- **Adopted: split GDN input projections in mixed rounds.** The LQ4 value-z projection writes the value rows and the output gate as separate targets, removing the per-layer owner gate copy. Score at S=1000 rose by about 0.006.
- **Adopted: row-scaled protections on the fused prefill routes.** The 26 row-scaled E4M3 protections use the LQ4 prefill kernels' split, accumulate and SiLU-less MLP epilogues at T > 128. GDN output layer 4 keeps its standalone gated RMSNorm, because the fused producer's per-head reduction order is not bitwise. Prefill-first rose from 3,510–3,525 to 3,530–3,532 tok/s at T2048 and mixed prefill by ~0.6%. The score is unchanged (~1.061), because both sides of the ratio moved.
- **Adopted: three launch fusions.** The wide GDN control projection is split by 16-head group (grid ×3). The GDN input RMSNorm now comes from the fused norm-and-E4M3 producer's BF16 side output. A mixed round's owner direct scatter, history publish and verify record convolution run as one launch (119 VGPRs, 12 waves/SIMD, no scratch). All three are score-neutral within noise (1.063 against 1.064).
- **Adopted: batched verify attention.** The KV append, packed-decode attention and merge of all B verify sequences run as one launch each (grid z = sequence), instead of three per sequence per attention layer. Decode rose from 201.6 to 203.2 tok/s at C2 and from 239.2 to 241.5 tok/s at C4 (`-pg 512,512`, two runs). The score is unchanged (1.061–1.062 at S=1000).
- **Rejected: the owner's chunked recurrence concurrent with the verify ReplaySSM record** on a side stream (fork/join per GDN layer). The score fell from 1.064 to 1.035 at S=1000 and to 0.967 at S=232. Mixed prefill lost ~8 ms per round, ~170 µs per layer, far more than the 43 µs record. The same reason rules out moving the GDN replay fold or the verify attention onto a side lane under the owner's work.
- **Not pursued:** releasing the host on the egress event (~0.35 ms per round, about +0.0013 score), and folding the owner's DFlash context append into the next verify append (~0.7 ms, and it changes drafter bits). Both are below the ±0.002 score noise. What remains of a mixed round's decode side (drafter ~6.3 ms, verify tail 1.4 ms, round boundary ~0.4 ms) is also in a decode round, so speeding it up does not move the score.

**How this compares with other engines.** No engine or paper we found uses an additive score like this one. vLLM (V1 chunked prefill, decode-first) and SGLang ship a fixed per-forward token budget sized by GPU memory: 2048 on a 32 GB GPU, 8192 on H100-class GPUs. vLLM documents smaller budgets as better ITL and larger as better TTFT. TensorRT-LLM tunes `max_num_tokens` (default
8192) for throughput at equal latency. llama.cpp fills an `n_batch` of 2048 with decode tokens first. Sarathi-Serve picks the largest token budget whose batch time meets a P99 time-between- tokens SLO (512 strict, 2048 relaxed), and notes the tile-quantization cliff (257 tokens up to +32% over 256). DistServe measures goodput under TTFT and TPOT SLOs.

The score here measures only the co-batching gain over time-sharing, which scores 1.0. It does not choose a latency point. Its maximum at S=1000 (a 1024-column forward) is a hardware-efficiency optimum. S also sets the decode lanes' round time during a prefill: ~0.18 s at S=488, ~0.30 s at S=1000 and ~0.52 s at S=2024, against 39 ms for a decode-only round. The TTFT cost at S=1000 is ~8% (2.7 s against 2.5 s for an 8K prompt). The default (`--mixed-forward auto`) is this optimum: a 1024-column forward (bounded by the chunk) under DFlash with C > 1, so S = 1000 at C4 with W=8. Smaller draft windows were checked at C4 with adaptive draft and two passes each. With max K=3, S=1012 scored 1.075 and 1.063, against 1.066 and 1.055 at S=1000. With max K=5, S=1006 scored 1.059 and 1.059, against 1.057 and 1.057 at S=1000. The tile-filling S wins every pair, and the score stays above 1 (`profiles/bench/r9700-mixed-pd-20260929/kcheck.sh`). If a deployment has an inter-token latency target, the rule is the largest 256-multiple `--mixed-forward` whose mixed round meets it (512 for ~0.2 s). The flag only accepts multiples of
256. With no decode-ready request the owner runs prefill-first chunks regardless.

**Adopted: tile-filling mixed slices.** A mixed round's owner now takes the verify columns its round leaves unused, so the forward stays at S + 8·(C−1) = 1024 when fewer than C−1 requests decode or adaptive draft length shortens the verify panels. At C4, S=1000 (`--contention-lanes`, two interleaved runs, `tfo_*` fixed against `tfn_*` filled):

| Decoding lanes | Owner slice (fixed → filled) | Prefill tok/s | Score |
|---:|---|---|---|
| 1 | 1000 → 1016 | 3,130 / 3,140 → 3,163 / 3,163 | 1.057 / 1.060 → 1.069 / 1.068 |
| 2 | 1000 → 1008 | 3,091 / 3,091 → 3,100 / 3,101 | 1.059 / 1.058 → 1.061 / 1.062 |
| 3 | 1000 (unchanged) | 3,057 / 3,060 → 3,061 / 3,058 | 1.063 / 1.064 → 1.064 / 1.063 |

Decode share and decode tok/s are unchanged, TTFT drops by ~0.02 s with one decoding lane, and the greedy pair check stays identical.

**Final validation (2026-09-30, rebased onto upstream 0eab8185, 21-protection production artifact; pinned binaries, against an upstream-only build of 0eab8185).**
- Quals, all pass:
  - GDN (including the single mixed convolution launch);
  - FP8 producers (with the normalized side output);
  - projected controls, normalized front and pair convolution record;
  - LQ4 front and Linear, and row-scaled Linear (small and prefill T);
  - eager and fused attention;
  - dense prefill and mid-row attention;
  - the dense-verify route discriminator (sequence batch, default, long context);
  - full-attention leaf and KV;
  - engine boundary and engine cache/cancel.
- ctest: all 91 tests pass.
- 8K PPL mean NLL: 1.8803156 against upstream's 1.8802432. The K5120 RMSNorm row-CTA route at every width, which the fused normalized producers share, is qualified against the oracle but not bitwise upstream's token8/generic RMSNorm at 25+ rows.
- Greedy pair checks at S=1000/488: the prefilling request is identical to prefill-first. The co-running decode lane diverges at a near tie (token 102–107), because its rows run at the mixed width in mixed rounds.
- Acceptance-free speed against upstream, interleaved over three rounds:

  | Workload | Upstream | This build |
  |---|---:|---:|
  | Plain decode C1 (tok/s) | 37.9 | 37.9 |
  | Plain decode C4 (tok/s) | 122.7 | 125.2 |
  | Prefill-first T512 (tok/s) | 2,946 | 3,242 |
  | Prefill-first T1024 (tok/s) | 3,307 | 3,485 |
  | Prefill-first T2048 (tok/s) | 3,455 | 3,539 |

  DFlash `-pg` tok/s is not comparable across builds whose greedy text differs, because acceptance follows the text.
- C4 score (this build): S=1000 1.060 / 1.063, S=488 1.053 / 1.052, one decode lane at S=1000 1.070.
- Serve (C3, S=1008, i.e. `--mixed-forward 1024`, two streaming decodes while a 9,593-token prompt arrives): the long request answers correctly in 3.18 s, and each stream's largest gap is 0.33 s, against 2.99 s prefill-first. A 0.62 s gap in one run was a transient host stall; two reruns gave 0.33 s, as did the pre-sync build.
- Tool-grammar owners mix their final step (2026-09-30). Their first-token root mask was moved to the exchange's last row, so these steps no longer run alone. The serve A/B used the same server as above, with two streaming decodes and `tool_choice: required` requests of 307–9,868 prompt tokens, two passes each. All 20 calls were schema-valid and identical across the builds. Tool-request latency fell by 0.02–0.09 s at every size. The streams' largest gap fell from 0.188/0.263 s to 0.149/0.218 s at 307/539 tokens, and from 0.341–0.353 s to 0.304–0.331 s at 2.9K–9.9K tokens. Script: `profiles/bench/r9700-mixed-pd-20260929/serve_tool.sh`.

Remaining score options were assessed against a per-round cost model that reproduces the measured score (a 1000-slice round is worth ~1.07, the 192-token final slice ~0.93, a plain decode round 1.0). All of them are exhausted:
- **Slice size.** Merging the final remainder into the previous slice models −0.001, an even 9-way split is worse, and S=1024 (a 1048-column forward) is worse. Going from S=2024 to 1000 adds a forward worth ~1.19 of its cost; going below 1000 adds one worth <1.
- **Adaptive draft length.** Mixed forwards stay inside the 1024 tile at K3..K7. A tile-filling slice rule would help only low-acceptance production rounds; the bench score is unchanged. Lowering K in mixed rounds would raise the score while cutting real decode tokens, so it games the metric.
- **Scheduling.** The first and last slices are both mixed. The prefill-first baseline uses the faster 2048 chunk. Two decode rounds cannot share a forward, because each round's drafts depend on the previous round's acceptance.
- **Mixed-only kernels.** Nothing above ~0.5 ms per round remains that is not rejected above.

Repeat runs of one configuration span 1.055–1.068 at C4, so gains below ~0.004 are not measurable with this bench.

Score by concurrency (the new build; prefill-first rate of each C as the normalizer; D=1 unless noted; `c2`, `c3`, `fsn_1`). S + 8·(C−1) columns fill a 256-multiple forward:

| C | Decode-only tok/s | Best S | Score | Decode share / tok/s | Prefill tok/s |
|---|---:|---:|---:|---|---:|
| 2 | 135 | 1016 | 1.065 | 0.121 / 16 | 3,143 |
| 3 | 250 | 1008 | 1.054 | 0.131 / 33 | 3,077 |
| 4 | 239 | 1000 | 1.057 | 0.145 / 35 | 3,028 |

The other points of each C:

| C | S | D | Decode share | Decode tok/s | Prefill tok/s | Score |
|---|---:|---:|---:|---:|---:|---:|
| 2 | 504 | 1 | 0.198 | 27 | 2,850 | 1.054 |
| 2 | 1016 | 2 | 0.201 | 27 | 2,857 | 1.059 |
| 2 | 504 | 2 | 0.318 | 43 | 2,419 | 1.044 |
| 2 | 248 | 1 | 0.313 | 42 | 2,319 | 1.009 |
| 2 | 2040 | 1 | 0.069 | 9 | 2,992 | 0.967 |
| 3 | 2032 | 1 | 0.082 | 21 | 3,202 | 1.042 |
| 3 | 1008 | 2 | 0.214 | 53 | 2,782 | 1.048 |
| 3 | 496 | 1 | 0.212 | 53 | 2,765 | 1.041 |
| 3 | 240 | 1 | 0.355 | 89 | 2,293 | 1.043 |
| 3 | 496 | 2 | 0.337 | 84 | 2,322 | 1.033 |

C5..C8 (2026-10-01, after the C5..C8 verify-width kernels; build 83175b29; two runs per cell, five at C8 for 1024/2048; `profiles/bench/r9700-c8-20260930/score/`). Columns are the forward width N (`--mixed-forward N`, D=1); each C normalizes by its own prefill-first rate (3,339-3,362 tok/s). The 1024 decode share is 0.146 / 0.151 / 0.157 / 0.160 at C5 / C6 / C7 / C8:

| C | 768 | 1024 (auto) | 1280 | 1536 | 2048 |
|---|---:|---:|---:|---:|---:|
| 5 | 1.037 / 1.044 | **1.056 / 1.064** | 1.028 / 1.030 | 1.034 / 1.037 | 1.044 / 1.047 |
| 6 | 1.023 / 1.021 | **1.043 / 1.059** | 1.023 / 1.022 | 1.030 / 1.035 | 1.036 / 1.039 |
| 7 | 1.040 / 1.036 | **1.056 / 1.053** | 1.027 / 1.029 | 1.035 / 1.038 | 1.038 / 1.041 |
| 8 | 1.003 / 1.005 | **1.031** (5 runs, 1.028-1.035) | 1.005 / 1.022 | 1.019 / 1.013 | 1.032 (5 runs, 1.022-1.041) |

`auto` (1024) is the best width at C5..C7 and ties 2048 at C8, where 1024 also keeps the higher decode share (0.160 against 0.101) and half the decode stall. The C8 score is lower because its mixed round carries seven lanes of decode-side work (drafter, GDN replay fold and record, verify attention), so mixed prefill drops to ~2,910 tok/s against ~3,000-3,075 at C5..C7. C8 at the production context (32K, `--kv-capacity auto`, `--mixed-forward auto`) starts with the maximum 262,144-token KV capacity, a 265 MB workspace (mixed layout included), 1.12 GB of Device Graphs against a 2.32 GB allowance and 6.7 GB of planned slack.

C=1 has no decode lane during a prefill, so every policy reduces to prefill-first there.

Long context (C4, current build, `--contention L,R --contention-context L`: the 3 decode lanes hold L-token prompts while the owner prefills fresh L-token prompts; R=2 at 32K/64K, R=1 at 128K; `long_*`). Score normalizes by the same-context prefill-first rate:

| Context | Prefill-first tok/s (TTFT) | S=488 | S=1000 | S=1528 | S=2024 |
|---|---|---:|---:|---:|---:|
| 8K owner, 512 decode | 3,332 (2.5 s) | – | **1.061** | – | – |
| 32K | 2,923 (11.2 s) | 1.033 | **1.039** | 0.982 | 1.025 |
| 64K | 2,480 (26.4 s) | 1.025 | **1.030** | – | – |
| 128K | 1,905 (68.8 s) | 1.013 | **1.021** | 0.988 | 1.013 |

| Context | S=1000 decode share / TTFT | S=488 decode share / TTFT |
|---|---|---|
| 32K | 0.128 / 12.3 s | 0.230 / 14.0 s |
| 64K | 0.121 / 29.1 s | 0.218 / 32.8 s |
| 128K | 0.112 / 75.7 s | 0.205 / 85.1 s |

S=1000 stays best at every context, and D=1 remains the rule. The score falls with context, because each mixed round's owner slice carries the whole-context attention over the prompt so far, which lengthens the round the decode lanes wait on. It is also because the decode lanes' own verify attention grows. S=1528 (a 1552-column forward) is not 256-aligned and loses 3–5%.

The bench corpus has 65,536 tokens and 682 distinct IDs and wraps. Long prompts therefore repeat, which inflates DFlash acceptance: decode-only rates of 290–450 tok/s at long context are not representative of natural text. Shares and scores are ratios within one run and are less affected.

Dominated by these: every time-share point (2048-token chunks after D decode rounds):

| D | Decode share | Prefill tok/s |
|---:|---:|---:|
| 1 | 0.065 | 2,864 |
| 2 | 0.114 | 2,724 |
| 4 | 0.196 | 2,488 |
| 8 | 0.324 | 2,102 |
| 16 | 0.483 | 1,589 |
| 32 | 0.643 | 1,066 |

The mixed line now sits 4–10% above the time-share line P≈3,300·(1−share). The gain grows with decode share. Each mixed round fuses one decode round into a prefill step for ~12 ms of extra wall time, against 45 ms for a separate decode round. At S=232, a mixed round's 232 prompt tokens cost 63 ms over the decode-only round (~3,650 tok/s marginal), above the full-chunk rate.

The same sweep before the small-T prefill work (`fused4`/`fused5`) had these points:
- S=488: 0.204 / 2,568
- S=232: 0.343 / 1,980
- S=488, D=4: 0.499 / 1,620
- S=1000, D=4: 0.338 / 2,194

Full-chunk mixing (S=2024: 2024 prompt + 24 verify columns fill the 2048 chunk) at D=1/2/4/8/16 gave 0.069/3,038, 0.118/2,872, 0.205/2,598, 0.336/2,180 and 0.497/1,636 on that earlier build: 3–6% above time-share at each D, and equal to S=1000 at twice the stall. Fusion saves one decode round's non-Linear work per slice, so large slices stay close to the time-share line. The middle of the frontier needed 128–512-token prefill near 2048-chunk efficiency (next section).

Larger S with D>1 amortizes each mixed round's fixed excess (the owner's attention/GDN segments, verify rows and host-side slice work), which is why S=1000/488 with D=2..8 dominates small S at D=1 at equal share.

Mixed-round overheads removed during the campaign:
- A mid-T LQ4/raw-FP8 Linear route for T 33..128, 8 waves × (16-row block, K split). Whole-model T=48..128 prefill went from ~95 to 47–71 ms.
- K5120 RMSNorm takes the row-CTA route at every width. 25..127 rows had fallen to the generic kernel (10× slower), and ≥128 rows used the serial token8 chain (115 → 47 µs per call at T2048). This matches the fused normalized producers' reduction order at every width.
- The plain per-token E4M3 quantizer now runs on a resident 128×256 grid striding tokens. Grids above ~2048 waves paid a ~30 µs dispatch cliff: T64 dropped from 35 to 6.5 µs, and T2048/K5120 from 85 to 50 µs, which also speeds ordinary prefill.
- The GDN projection→conv direct scatter now serves any T and the combined [value,z] rows. It replaces 3 copies, the conv and 3 extracts: T40 dropped from 45 to 21 µs per layer, and P2048 from 0.53 to 0.20 ms.

In total, the mixed round at S=40 went from 72.8 to 61.9 ms. Evidence: `profiles/bench/r9700-mixed-pd-20260929/` (`frontier.py ts3 fused4 fused5 fused6`, kernel traces in `prof/`).

### Small-T prefill (2026-09-29)

Whole-model prefill step (C1, production DFlash, `ninfer_bench -p T`; ms, then tok/s):

| T | 64 | 128 | 256 | 384 | 512 | 768 | 1024 | 2048 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Before | 57.0 | 71.1 | 102.5 | 162.7 | 173.2 | 286.7 | 306.9 | 598.5 |
| After | 41.7 | 58.2 | 89.5 | 134.1 | 158.7 | 227.5 | 298.6 | 597.3 |
| tok/s | 1,534 | 2,201 | 2,859 | 2,863 | 3,225 | 3,376 | 3,430 | 3,429 |

Changes:
- **LQ4/raw-FP8 prefill Linear CTA shapes.** The CTA is (G×64 tokens)×(8/G×R rows), with 256×128, 256×64 and 512×64 selected by `prefill_shape`, and 16-token fragments past T skip their activation loads and WMMAs. Every shape publishes identical bits (one wave's same-order K accumulation per output).
  - T ≤ 256 with fewer than 96 128-row CTAs use 64-row CTAs: N5120/7168/4096 are 16–20% faster per launch.
  - Wide N uses 512-token tiles wherever they add no padding: T2048 588 → 583 ms.
  - Narrow N uses one 512-token tile at T 257..511 (T288 129 → 113 ms). Above that, 512-token tiles lose 15–35% per launch at T2048 in the model, because they double the re-reads of the large K17408 activation images, even though they win in an isolated hot-cache benchmark.
  - The rule was chosen by interleaved whole-model runs. A 256×32 shape was 40–75% slower everywhere and was rejected.
- **GDN prefill front for every width.** The formerly P2048-only leaf (norm, controls, split value/gate projection, conv direct scatter) now serves every ordinary prefill width. This removes 288 launches per step at T ≠ 2048, bitwise-identical.
- **A8Q4 M128 prefill CTA with predicated partial tiles, admitted above T32.** This serves the DFlash context projections of every prefill step and mixed slice. The N5120/K25600 feature projection went from 651 to 361 µs at T64 and from 3,650 to 971 µs at T512; the per-layer N6144/K5120 projection from 834 to 235 µs at T512. These had used the WMMA32 route except at exactly 1024/2048/4096/8192.

Measured and not adopted (interleaved whole-model `ninfer_bench -p`, bitwise-identical candidates):
- **Mid-T token split** (two token parts per narrow-N launch at T 49..128). It saved 5–16 µs per launch in isolation, but the whole step moved only −1.7% at T64 and ±0.4% at T 80..128. The mid-T kernel still beats every prefill shape at T ≤ 128 by 18–120%.
- **4-wave CTAs** (256×32 and 256×64 rows) for narrow N. At T ≤ 256 they do not beat 8-wave 256×64, so per-wave intensity rather than CTA balance limits the 5120-row launches. Whole steps moved −2.8% at T520 but +0.5–1% at T640/1032. A mixed round reaches T513+ only if S + (C−1)·W > 512; S=488 at C4 is exactly 512.
- **Launch gaps.** Eager and HIP-graph dispatch cost the same on this ROCm: 3.2 µs per empty kernel, ~1 µs on a busy stream, in a 1,000-kernel chain. Prefill graph capture therefore cannot recover the ~4.5 µs ramp and tail per boundary. `HIP_FORCE_DEV_KERNARG=1` changed nothing.
- **GDN chunked recurrence split over value rows** (two 4-wave CTAs per head, each recomputing the shared per-chunk k normalization, A/P and diagonal): 0.63 → 1.55 ms at T2048. At 58.9 KB of LDS only two CTAs fit per WGP. The kernel already uses 256 VGPRs (one spill), which leaves no room to prefetch the next chunk's loads in registers.

Remaining costs at T2048:
- **Linears:** about 470 ms, at the ~210 TFLOP/s power bound (T256 ~205 wide, ~155 for the 5120-row down/output launches).
- **GDN chunked recurrence:** 33 ms, latency-bound on 48 CTAs as above.
- **Memory-bound helpers near DRAM bandwidth:**
  - activation quantization: 15 ms at ~580 GB/s (92% of the read peak);
  - GDN conv scatter: 9 ms at ~440 GB/s;
  - GDN controls: 3 ms. Fusing the input RMSNorm with its quantizer would save at most ~2 ms.
- **Inter-kernel gaps:** about 9 ms per step, from 1,062 launches (see above).

Verification:
- `ninfer_r9700_lq4_linear_qual` and `ninfer_r9700_fp8_row_scaled_linear_qual` pass at T 33..128 (mid-T, including 65/80/113) and 130/200/300/640/1000/2048, with SiLU, residual, poison and split cases on every shape.
- `a8q4_shape_sweep_qual --prefill-cta-regression` passes, with the context shapes at T 33..1000 partial tiles.
- The RMSNorm production regression passes, with exact same-column invariance at T1..128, 255, 256 and 2048.
- PPL on `tools/ppl/corpus.ids` at 8K:
  - Chunks 2048 and 256 are bitwise-identical, at mean NLL 1.878183.
  - Chunk 384 gives 1.878221.
  - The morning's recorded baseline was 1.878510.

Serve check (C3, prefill chunk 2048): two greedy 700-token streams run while a 9,593-token prompt arrives.

| Config | Worst stream gap | Long request latency |
|---|---:|---:|
| Prefill-first | 3.08 s | 3.09 s |
| S=488, D=2 | 0.204 s | 4.51 s |
| S=232, D=1 | 0.204 s | 4.78 s |
| S=40, D=1 | 0.088 s | 9.35 s |

All four runs answered the needle correctly. In these runs the worst gap came at admission, where the owner's first slice ran as a separate step; that slice now joins a mixed round (see the frontier section). Later gaps are the mixed rounds (~0.12 s at S=232).

Correctness (`--pair-check 3000,64`, C2):
- The prefilling request's 64 greedy tokens are bit-identical to prefill-first at S=40/120/2040.
- The co-running decode lane's verify rows use the mixed width's Linear/norm routes (same Op oracles), so its greedy stream can flip at a near-tie (first difference at token 33–139 of 256).

## Decode pauses by incoming request and the Vision encoder (2026-09-30)

C3 serve, DFlash K7 adaptive, default `--mixed-forward auto` (1024), two streaming decodes while one request arrives (`profiles/bench/r9700-mixed-pd-20260929/probe.sh`, `tiers.sh`). Pause is the longest gap in either stream while the request is in flight; arrival adds one ordinary decode round (0.034 s) in every case.

| Incoming request | Longest pause | Request latency |
|---|---|---|
| 9.6K-token text prompt (8 runs) | 0.329-0.334 s | 3.17-3.22 s |
| Follow-up turn, prefix in VRAM / RAM tier / disk tier | 0.089 / 0.112 / 0.088 s | 0.25 / 0.51 / 2.02 s |
| 768x512 image (408 tokens), before / after | 0.347 / 0.205 s | 0.87-1.05 / 0.83-0.87 s |
| 1920x1080 image (2,064 tokens), before / after | 2.82 / 0.35 s | 3.9 / 1.79-1.89 s |

The RAM and disk restores (0.18 s copy, 1.7 s SSD read) run on the copy stream while decode continues. A single earlier 0.62 s gap did not recur in 8 instrumented runs; that run's whole prefill was 14% slower, which points to external contention rather than a scheduling path.

The 1080p pause was the Vision encode, which ran as one unbounded step. A kernel trace of one encode (`profiles/rocprof/vision-1080p-20260930*`, attribution only) put 2,172 of 2,443 ms in Vision attention (80 ms per block at P=8,256, ~4 TFLOP/s) and 78 ms in LayerNorm, whose row statistics ran serially on one thread. Changes:
- Vision attention computes S^T = K Q^T, so each lane owns one query: the online softmax is per-lane plus one half-wave shuffle, and the accumulator is already the B operand of O^T = V^T P^T, with no LDS transpose. K and a transposed V are staged per 32-key tile with 16-byte loads, 8 waves (128 queries) per block. Probabilities keep the BF16 high-plus-residual split: a single BF16 failed the FP64 oracle (rel L2 0.0047 > 0.0025). Qualifier event timing: P=8,256 84.2 -> 7.48 ms, P=1,536 3.6 -> 0.37 ms, three images P=5,196 12.8 -> 1.23 ms; error against the FP64 oracle unchanged (rel L2 0.00167). 16 waves only helped P=1,536.
- LayerNorm runs one wave per row (two-pass mean and centered variance): D=1152 x 8,256 rows 1.80 -> 0.082 ms against its FP64 oracle.
- While decode waits, the encode runs as its own prefill step (`encoded_only`), so the following text chunk waits for a decode round instead of adding to the same pause (0.735 -> 0.44 s).

- The Vision A8Q4 linears ran the one-wave WMMA32 route (one 16x16 tile per wave, no operand reuse). They now take the cooperative prefill CTA, exact against the FP64 oracle (`tools/r9700/vision_a8q4_linear_qual.hip`): the M128xN128 kernel for patch, qkv, projection and merger tuples above 32 patches, and the M64xN128 fallback for the MLP tuples (N4304, and K4304 padded to 4352) above 128 patches, which keep WMMA32 below. Per 1080p encode 160 -> 66 ms; for example qkv 1.42 -> 0.51 ms, fc2 2.02 -> 0.84 ms.
- Attention key tiles of 32 instead of 64: P=1,536 0.37 -> 0.31 ms, P=5,196 1.27 -> 1.22 ms, P=8,256 unchanged. Rejected: 128-key tiles (P=8,256 10.2 ms), 16 waves, and FP16 probabilities (one WMMA instead of two), which fail the oracle at rel L2 0.0045 with or without a 2^14 scale against subnormals: the error is rounding bias over many near-equal probabilities, which the BF16 residual removes.

The 1080p encode window fell from 2,443 to 318 ms: attention 200 ms (~70 TFLOP/s executed, including the residual pass), linears 67 ms, activation quantization, bias and GELU ~30 ms. Its decode pause (0.35 s) now matches a text mixed round (0.33 s), so the encode is not split further. Greedy image descriptions diverge from the previous build at near-ties and stay accurate, and a 128x128 image (64 patches) runs. The BF16-source Vision parity tool needs a torch environment that this host lacks.

## Decode stalls from KV-tier host copies (2026-10-01)

C3 serve with the Compose tiers (RAM 4 GiB, disk 32 GiB, context 32768), DFlash K7 adaptive: two streams decode while six ~12K-token conversations rotate through the third lane for three passes, so each turn captures the retained lane and later turns restore from RAM or disk (`profiles/bench/r9700-stalls-20261001/churn.sh`, worker-thread timers in a scratch build). Every capture, RAM restore and disk restore copied the ~150 MB GDN rewrite image (plus DFlash and ladder images) on the scheduler thread while the other lanes waited: ROCm executes `hipMemcpyAsync` between pinned host buffers as a blocking CPU memcpy (151 MB: 11 ms in the call), while a `hipLaunchHostFunc` copy returns in 0.02 ms and does not delay kernels on another stream. Those copies now run as copy-stream host callbacks:

| Worker-thread call while 2 lanes decode | Before (total / max) | After (total / max) |
|---|---|---|
| Lane capture (18 admissions) | 520 / 49 ms | 99 / 14 ms |
| RAM restore (4 hits) | 112 / 36 ms | 18 / 6 ms |
| Disk restore pump (9 hits) | 204 / 40 ms | none above 2 ms |

All 18 greedy replies are identical between the builds. A generation-recovery retry that restores from RAM or disk now also waits in copy-hold instead of blocking decode for the whole copy.

Two more scheduler stalls showed up under stress (same harness, `churn.sh ... cancel`, which drops every third driver request 0.25 s in; `TIERS` sets the tier sizes):
- A cancelled copy-hold admission drained synchronously: the disk restore cancel waited for its in-flight SSD reads, and the drain waited for the copy stream. With the Compose tiers each cancel blocked decode 1.2-1.4 s. The hold now stays parked behind a copy-stream fence and a non-blocking disk cancel, and drains once they settle; no drain took 2 ms or more afterwards.
- With `--kv-disk-compress zstd` and small tiers (RAM 1 GiB, disk 3 GiB), the disk worker ran zstd-1 over each ~150 MB GDN state blob holding the cache mutex, which restore pumping, claims and planning on the scheduler take: `pump_disk_restore` max 683 -> 14 ms and admission max 785 -> 64 ms after encoding with the mutex released.

A long-context run (4 conversations of ~26K tokens, past the first checkpoint mark) showed one 63 ms pinned checkpoint-head allocation at first use; the head pool then recycles. The median mixed-round pause stayed 0.32 s in every run.

## Mid-row attention route (2026-09-28)

Causal chunks of 9..127 rows (short appended turns and tool results, prompt tails with `P mod 2048` in 9..127) fell between the packed decode route (1..8 rows) and dense prefill (>= 128) onto the fused one-query-row-per-CTA kernel, which re-reads the whole cache for every row and head: 34 ms per layer for 19 rows at 22K context, 547 ms of a 19-token follow-up's 591 ms prefill. They now run the dense prefill tile split over context chunks with the packed route's FP32 merge (`fp8_int4_kv_attention_mid_rows`; 239 VGPRs, no spill; the production dense kernel is unchanged, interleaved A/B within noise). FP64 oracle: `dense_prefill_attention_qual` (rows 9..127, contexts to 262K, chunks that start inside the query rows, per-chunk NaN poisoning). Median per-layer times, G16:

| rows | 2K | 8K | 22K | 32K | 131K |
|---:|---:|---:|---:|---:|---:|
| 9 | 0.05 | 0.10 | 0.20 | 0.27 | 1.00 |
| 19 | 0.12 | 0.12 | 0.24 | 0.33 | 1.22 |
| 64 | 0.08 | 0.17 | 0.36 | 0.51 | 2.10 |
| 127 | 0.12 | 0.29 | 0.69 | 1.02 | 4.22 |

(ms; 19 rows at 22K was 34 ms.) End to end, a 19-token follow-up restored from host RAM: TTFT 418 / 429 / 818 ms -> 220 / 183 / 240 ms at 8K / 11K / 22K. C4 DFlash auto KV capacity is unchanged (550,528 tokens). Evidence: `profiles/bench/r9700-attn-midrows-20260928/`, `profiles/bench/r9700-decode-study-20260928/`.

## Mid-row split extended to 1023 rows (2026-09-30)

Idea port of upstream 9639c32f (split attention for mid-sized appends). A dense-prefill call launches 4 KV heads x ceil(rows/32) row tiles, so 128..~500-row calls (tool results, medium appended turns, prompt tails) fill only 16..64 of the 64 CUs' CTA slots while each CTA streams the whole context. Host-fixed 9..1023-row causal calls now take the mid-row split whenever the wave model splits the context: the chunk count minimizes `ceil(ctas*c/64) * (ceil(ctx/c) + 64)` over at most 64 chunks, at least 256 keys per chunk and at most 2048 row-chunk FP32 partials; 1024 rows and above, device-counted rows and calls the model does not split stay on dense prefill. The kernels are unchanged and 9..127-row timings are within noise. The partial buffer is at most 50.7 MB, inside the existing 530 MB workspace plan, so C4 DFlash auto KV capacity is unchanged by this route (560,448 tokens with the 21-protection weights). FP64 oracle: `dense_prefill_attention_qual` (rows 128..768 at their own length, 22,229 and 131,072 keys; 192 rows at a V-scale extreme; 384 rows at 262,144), `runtime_planner_qual` workspace coverage. Median per-layer ms, G16, dense -> split:

| rows | 8K | 22K | 65K | 131K |
|---:|---:|---:|---:|---:|
| 128 | 0.62 -> 0.28 | 1.64 -> 0.72 | 4.83 -> 2.11 | 9.56 -> 4.09 |
| 192 | 0.65 -> 0.42 | 1.76 -> 1.11 | 5.15 -> 3.07 | 9.82 -> 5.84 |
| 256 | 0.69 -> 0.53 | 1.85 -> 1.44 | 5.43 -> 4.10 | 10.23 -> 7.73 |
| 384 | 0.95 -> 0.82 | 2.57 -> 2.14 | 7.00 -> 5.79 | 13.35 -> 11.07 |
| 768 | 1.60 -> 1.55 | 4.33 -> 4.09 | 13.28 -> 10.97 | 26.54 -> 21.88 |

512 rows stay dense (the grid already fills whole waves). End to end (C1 DFlash serve, no thinking, follow-up turns on a resident 130K conversation, four turns each, pre-split binary vs this build): 217-token appends TTFT 341 -> 291 ms, 654-token appends 719 -> 648 ms (`e2e/append_ttft.py`, `e2e/results.txt`). Fixed 256- and 128-CTA targets were rejected (slower at 9..127 and at 192/384 rows respectively). Evidence: `profiles/bench/r9700-split-append-20260930/`.

## Decode step attribution (2026-09-29)

C1, production artifact, Device Graphs, `ninfer_bench` region traces (kernel durations) against unprofiled wall time. Plain T1 decode: 26.52 ms per step, 703 kernels, 25.38 ms kernel time. DFlash K7: 31.31 ms per round, 827 kernels, 30.03 ms kernel time.

| Part of a DFlash K7 round | ms | Note |
|---|---:|---|
| Target LQ4/FP8 Linears (W8 verify, head, GDN pair) | 24.3 | 570-635 GB/s; e.g. gate/up 95 MB in 154 us (620 GB/s), head 675 MB in 1.1 ms (635 GB/s) |
| Drafter Q4 Linears | 2.6 | shortlist head ~590 GB/s; narrow N6144/N4096 projections 470-530 GB/s |
| GDN state (record 48 x 13.5 us, replay fold 0.53 ms, normalized front 48 x 9 us) | 1.6 | record/front are latency-bound, fold streams state at ~570 GB/s |
| Activation quantization (~250 launches of 2.5-3.3 us) | 0.8 | per-token E4M3 scale needs a row-wide max |
| Other small kernels (attention, KV append, norms, DFlash control) | 0.8 | |
| Inter-kernel gaps | ~1.3 | evenly ~1.5 us per boundary, no host stalls |

Weight streaming is already at ~95% of the 636 GB/s read peak, so the old 500-555 GB/s figure is superseded. What is left over pure streaming is ~3-4 ms per round, spread over many small kernels and launch boundaries; the recoverable part is estimated at 3-5% of a round, each candidate (quantization/launch fusion, GDN record and front, narrow drafter projections) around 1%. Evidence: `profiles/bench/r9700-decode-bw-20260929/`.

The four candidates were then worked through; none yields an admissible gain:

- Narrow drafter projections: a per-shape `k_split` sweep (2 to 20 waves per tile, T8, DRAM-cold) finds no cell faster than production for any drafter shape. The N5120/K25600 feature projection's in-graph 177 us (115 us in the one round without a preceding fold, 123 us isolated) is the replay fold's dirty-line writeback landing on the next kernel.
- GDN normalized front: one CTA per (head, token) is bitwise equal but slower at T8, 12.9 vs 12.0 us per launch in a 48-launch graph (faster only at T4, 9.8 vs 10.3).
- GDN record: the recurrence costs ~0.1 us per token; the kernel is bound by streaming each layer's 3 MB state plus launch. Overlapping the other state stream, the 0.53 ms replay fold, with the drafter on a second stream (low or high priority) gives no reliable gain: C1 K7 31.48 vs 31.54 ms/round over four interleaved pairs, C4 13.98 vs 13.99 ms per lane-round.
- Activation quantization: every layer's quantizers are fused into their producers (norm, gated norm, attention gate) except the one after the SwiGLU gate/up, whose per-token E4M3 scale needs a grid-wide maximum. Quantizing inside the down projection instead would double its per-CTA L2 activation reads (BF16 instead of FP8, 278 KB per CTA at T8).

Two further per-round costs were found:

- The speculative tool-mask exchange (ids readback, a Device Graph host node for the grammar match, a 248 KB mask upload) cost ~0.1 ms/round without tools and 0.21 ms/round with six declared tools (server, identical outputs, 193.0 vs 194.3 tok/s with the exchange removed). HIP does not overlap it as a graph branch: a side-stream branch executes serially, `DEBUG_HIP_GRAPH_SEGMENT_SCHEDULING` makes rounds 2.3x slower, and a captured `hipEventWaitExternal` wait crashes in `hipStreamWaitEvent` (a manually added event-wait node works). It is now a host mailbox answered by a matcher thread during the target forward (`docs/maintainer/concurrent-inference-architecture.md`, Speculative grammar exchange), with null masks for all-permitted rows and one broadcast mask per row when its nodes share a grammar state; a first version that copied every node's mask recovered only 0.8 of the 1.4 tok/s. C1 K7, identical outputs: 192.9 -> 194.3 tok/s with six tools (the no-exchange ceiling), 188.0 -> 188.4 without; `ninfer_bench` P512/G256 31.33 -> 31.25 ms/round, greedy tokens identical. `tools/smoke/serve_features.py` (structured output, forced tools) passes at C1 and C2.
- The pinned-host Q4N16K16 token embedding cost 50 us (target) plus 16.5 us (drafter) per round: one row gather touches 320 interleaved 128-byte lines over PCIe. The Q4G64 embedding is now stored `row-split-k128-v1` (one contiguous 2,560-byte code run and 160-byte scale run per row; `transcode_embedding_rows.py` revised `r9700-lq4` losslessly, every other payload byte-identical and the embedding's inverse permutation exact). The gather reads one 16-byte code chunk per 32 features. `ninfer_r9700_eager_qual` product-shape and pinned-host (in place, staged, shifted MTP window) embedding checks pass exactly; greedy DFlash K7 tokens are identical to the tiled artifact. Kernel trace, P512/G128 C1 K7: target gather ~50 -> ~7 us, drafter gather 16.5 -> 4.8 us; unprofiled P512/G256 31.264 -> 31.234 ms/round (three interleaved pairs, loaded host).

## Batched DFlash drafting at C2..C4 (2026-09-28)

The DFlash2 drafter ran batched across requests only at `k == 4` (an upstream gate from when K4 was the production setting); every other chain K looped it per request, so C4 K5 spent ~6.4 ms per round drafting versus ~2 ms batched. All chain rounds now draft in one batched pass. The A8Q4 small-batch projections gained the T14/T16/T21 cells (C2 K6/K7, C3 K6, C4 K3; the general WMMA route before) and T4 for every drafter shape (C1 K3), and the fused LQ4 GDN pair projection/convolution admits W4 (C1 K3). The new cells pass `ninfer_r9700_a8q4_small_batch_projection_qual` (FP64, 169 cells) and `ninfer_r9700_gdn_lq4_front_qual` (bitwise against the composition). Greedy tokens match the unchanged-path reference at every C/K (C4 K6/K7 already differed from K4 before the change: their 28/32-row verify takes different target routes). `ninfer_bench` P512/G256, ms per round, before -> after:

| C | K3 | K4 | K5 | K6 | K7 |
|---|---:|---:|---:|---:|---:|
| 1 | 33.1 -> 30.6 | 30.8 | 31.1 | 31.2 | 31.5 |
| 2 | 43.5 -> 39.3 | 39.5 | 42.2 -> 40.2 | 42.9 -> 40.7 | 44.2 -> 41.9 |
| 3 | 51.4 -> 43.3 | 44.0 | 51.0 -> 46.9 | 52.2 -> 48.6 | 53.0 -> 49.5 |
| 4 | 59.3 -> 47.5 | 49.9 | 58.1 -> 52.6 | 60.1 -> 55.3 | 61.3 -> 57.0 |

On the study corpus (production adaptive K up to 7, same seeds) aggregate decode rises 137.6 -> 148.0 (C2), 172.2 -> 180.9 (C3) and 196.5 -> 204.4 tok/s (C4, two seed sets each), and C4 adaptive now reaches 3.62 tokens per round (was 3.32) and matches fixed K4. The fused GDN pair kernel stays batch-1: at C>1 the split projection is weight-bound at the same cost and the separate convolution/record is ~0.3 ms per round. Evidence: `profiles/bench/r9700-dflash-batched-20260928/`.

## Decode study (2026-09-28)

Production serving shape (DFlash p-less T1.5, adaptive K up to 7, draft temperature 0.4) on 13 prompts (code, structured, story, translation, AIME, logic, 6K OWUI tools, 11K/22K agent code) x 3 seeds x 768 tokens. At C1 adaptive stays at K7 (output identical to fixed K7): 106.8 tok/s vs fixed K5 101.4 and K3 80.7; draft temperature 0.4 and 0.6 tie (106.8 / 107.9), 0.8 gives 105.3, 1.0 gives 101.8. Aggregate throughput is 90 / 138 / 172 / 195 tok/s at C1..C4. Round time was lowest at K4 for every C (C4: K3 59.3, K4 49.9, K5 58.1, K7 61.3 ms) because only `k == 4` ran the DFlash drafter batched; fixed below. Long-context plain decode rises from 26.8 ms/step at 512 to 36.9 at 200K: ~510 GB/s of KV reads, 80% of the read peak, so attention headroom is at most ~1.5% at 32K.

## Long-context prefill attribution and exhausted mechanisms (2026-09-27)

Production `r9700-lq4` artifact, C1, chunk 2048, code corpus. 128K prefill measures 1,901 tok/s (68.96 s; kernel time 68.16 s, so host gaps are ~1%). Kernel shares (`profiles/rocprof/prefill-128k-lq4-trace-20260927/`, attribution only): dense prefill attention 47.2% (~105 useful TFLOP/s averaged over the growing context), LQ4 prefill GEMMs 44.3%, chunked GDN 3.2%, everything else below 1.5% each. At 240K attention is the majority.

Dense prefill attention ablations (`dense_prefill_attention_qual --time 2048 131072 16`, median of 10, oracle check disabled for the ablated builds; production 56.2 ms, 116 TFLOP/s):

| Removed | ms | Share |
|---|---:|---:|
| exp2 in the online Softmax | 56.3 | 0% |
| end-of-block barrier | 56.2 | 0% |
| FP8/INT4 -> BF16/FP16 conversion (raw K/V still loaded and stored) | 54.5 | 4.6% |
| all K/V staging (loads, conversion, LDS stores) | 50.5 | 10% |
| QK WMMAs | 33.7 | 40% |
| PV WMMAs | 37.4 | 33% |

The kernel is WMMA plus one LDS fragment per WMMA, which an isolated loop bounds near 140 TFLOP/s (180 with no LDS load, 155 with one load per two WMMAs). Measured out:

- **Paired-wave split** (two waves share a head, each owning half of D for both row tiles, so every K/V fragment feeds two WMMAs; partial scores and probabilities exchanged through LDS): a structure microbenchmark with the exact per-block load, exchange and barrier counts runs at ~120 TFLOP/s against ~126 for the current structure; the two extra barriers per block cost more than the halved fragment loads. Not built.
- **Staging**: the remaining 5.5% is exposed global-load latency. Hiding it needs a register prefetch of the next block (it spilled at 256 VGPRs; the budget is now 240) or a VMEM-to-LDS load, which gfx1201 lacks. Converting K/V once per chunk instead of once per CTA would need a transient BF16 copy of the whole context (~1 GB at 256K), i.e. a second cache representation.
- **Prefill chunk 4096** (`profiles/bench/r9700-longctx-chunk-20260927/`, alternating pairs): 32K 2847/2790 -> 2875/2862 tok/s (+1-2.5%, within run-to-run spread), 128K 1905/1906 -> 1862/1864 tok/s (-2.3%). The default stays 2048.

LQ4 GEMM repacking was measured by the research ports below (at most 3.3%, mixed) and the GEMM's redundant per-CTA decode cannot be shared without a cross-CTA or runtime-repacked weight path. No long-context prefill mechanism with a credible whole-prefill bound remains.

## Two-tile small-T FP8 Linear for T 17..32 (2026-09-28)

Above T16 (C3/C4 DFlash verify) every LQ4 Linear, the output head and the row-scaled FP8 projections fell to the prefill GEMM. Both small-T kernels now add a second 16-token WMMA tile on the same decoded weights; T > 32 keeps the prefill GEMM. FP64-oracle qualified on all production shapes at T17..32 (`lq4_linear_qual`, `fp8_row_scaled_linear_qual`, `attention_fused_qual`). `--time-cell`, median us:

| N x K | T16 | T24 before (prefill) | T24 two-tile |
|---|---:|---:|---:|
| 34816 x 5120 | 162 | 474 | 171 |
| 5120 x 17408 | 38 | 339 | 57 |
| 12288 x 5120 | 31 | 157 | 47 |
| 7168 x 5120 | 20 | 105 | 31 |
| 5120 x 6144 | 18 | 118 | 28 |

Evidence: `profiles/bench/r9700-lq4-small-t-20260928/`.

## Activation codec and residual rounding (rejected, 2026-09-28)

Each candidate was isolated in the checkpoint-direct BF16 scorer (BF16 weights, BF16 KV) by changing only the Linear-input or residual boundary. Results are paired against an unmodified run of the same harness (prefill schedule, `corpus.ids`), in dNLL nats/token:

| Boundary change | 8K | 32K |
|---|---:|---:|
| production per-token E4M3 activation (scale absmax/448) | +0.0034 +/- 0.0013 | +0.0026 +/- 0.0008 |
| E4M3 with one scale per 64 columns (G64) | | +0.0014 +/- 0.0006 |
| E4M3 G32 (G128 / G512) | -0.0002 +/- 0.0010 (+0.0022 / +0.0029) | |
| G32 on RMSNorm-fed inputs only / attention+GDN output inputs only / MLP down only | +0.0013 / +0.0013 / +0.0034 | |
| per-token scale chosen by minimum squared error (8 / 32 candidates) | +0.0017 / +0.0027 | |
| residual add rounded once, `BF16(x + y)` instead of `BF16(x + BF16(y))` | +0.0009 +/- 0.0005 | |

The activation codec accounts for about a quarter of production's BF16-source gap. The gain from finer scales comes from exactly encoding each group's largest (error-energy-dominant) element, not from subnormals: subnormals are under 0.8% of elements even per-token. Only 64-column groups fit the kernels, because each FP8 WMMA pairs two 32-column weight groups; G32 needs regrouped operands. The prefill GEMM can't hold a second accumulator set, so it must rescale its accumulators at every group boundary. A timed probe of that rescale (T2048, `--time-cell`, scale ratios of one) cost +19-24% for G64 and +53-70% for G32 on the gate/up, down and 12288-row projections; the WMMAs share the VALU issue with the rescale multiplies. G64's paired gain over per-token is 0.0012 +/- 0.0009 at 32K. Neither granularity is worth its prefill cost. Single rounding moves away from the BF16 formula, which itself rounds the Linear output before the residual add. None of these is adopted. Evidence: `profiles/ppl/r9700-activation-codec-ref-20260928/`.

Follow-up screens used the same harness:

| Boundary change | 8K | 32K |
|---|---:|---:|
| production Q4G64 token embedding (artifact bits) vs BF16 | -0.0008 +/- 0.0006 | +0.0002 +/- 0.0004 |
| LQ4 token embedding (production encoder) minus production Q4G64 | +0.0015 +/- 0.0007 | -0.0006 +/- 0.0004 |
| chunked-GDN FP16 operand emulation (validated at 5.3e-4 relative state error) vs FP32 | +0.0006 +/- 0.0005 | -0.0004 +/- 0.0003 |
| G64 activation scales for scored (decode) positions and head only, minus all per-token | -0.0026 +/- 0.0015 | -0.0015 +/- 0.0009 |

Embedding quantization and the chunked-GDN FP16 operands have no measurable cost, so neither change is worth making. Decode-only G64 keeps prompt positions per-token (the prefill GEMM is unchanged) and recovers about as much as G64 everywhere in the BF16-weight harness.

**Decode-only G64 on the production gate (rejected).** Implemented as grouped activation images for T <= 32. Every small-T FP8 consumer (T1 GEMV, small-T and two-tile WMMA, protected row-scaled small-T, the GDN fused projection) adds each 64-column step times its group scale. Every producer publishes max|group| / 448. The implementation passed the producer, Linear, row-scaled, attention-fused and GDN-front FP64 oracles. Against the same HEAD binary on `r9700-lq4` (`profiles/ppl/r9700-decode-g64-20260929/`, patch retained there):

| Cell | baseline dNLL vs BF16 | candidate | paired candidate - baseline | flips |
|---|---:|---:|---:|---:|
| prefill 8K (control) | +0.0129 | +0.0129 | byte-identical | 7.79% / 7.79% |
| decode 8K, graph | +0.0122 | +0.0121 | -0.0001 +/- 0.0018 | 7.40% / 7.30% |
| decode 32K, graph | +0.0099 | +0.0111 | +0.0012 +/- 0.0009 | 6.60% / 6.35% |

With LQ4 weights the reference's predicted gain (-0.0015 at 32K) does not appear, so the code was reverted without speed or NIAH runs.

## DFlash K6/K7 chains (2026-09-27)

Fixed K6/K7 (W7/W8) chains are supported (`--draft-tokens 6|7`). Their earlier ~4 ms (C1) / ~16 ms (C4) round penalty was route fallback, removed by 7-8-row A8Q4 small-batch drafter projections, W7/W8 GDN pair conv/record and a T<=32 normalized GDN front (all qualified against their FP64/exact oracles, `tools/r9700`). Production `qwen3.8-27b-r9700-lq4`, `--lm-head-draft`, Device Graphs, 12-prompt corpus x greedy + two p-less seeds, ABC/CBA order, paired per-request decode rate with 95% bootstrap interval:

| C | arm | tokens/round | ms/round | vs baseline |
|---|---|---|---|---|
| 1 | fixed K7 vs fixed K5 | 3.00 / 2.79 | 31.7 / 31.2 | 1.098 (1.059–1.142) |
| 4 | fixed K7 vs fixed K5 | 3.01 / 2.77 | 77.4 / 78.9 | 1.126 (1.099–1.154) |
| 1 | fixed K7 vs production adaptive K5 | 3.00 / 2.79 | 31.7 / 31.2 | 1.098 (1.040–1.160) |
| 4 | fixed K7 vs production adaptive K5 | 3.01 / 2.57 | 77.0 / 43.6 | 0.672 (0.653–0.691) |

At the time, C4 adaptive ran mostly K3 because every LQ4 target Linear above 16 tokens (C4 x W>=5) left the small-T route for the prefill kernel, nearly doubling the round; fixed K5/K7 paid that cliff. Adaptive `{3..7}` then measured 0.92x adaptive K5 at C1: each request's first round picked the lowest measured T(k) before any hop was observed, and deeper hops never earned credit, so it settled on K4. Both are fixed on 2026-09-28 (below). Evidence: `profiles/bench/r9700-w8-kernels-20260927/`.

### Adaptive K{3..7} production default (2026-09-28)

`compose.yaml` now runs `--draft-tokens 7 --adaptive-draft`: the two-tile kernels above plus a DFlash adaptive policy that continues the deepest observed hop acceptance into unseen hops (new hops start from it, Beta weight 4). Paired decode rate vs the `40c5eab8` server at adaptive K5, same harness, 95% bootstrap:

| | C1 | C4 |
|---|---|---|
| two-tile kernels, adaptive K5 | 1.000 | 1.225 (1.184–1.275) |
| + policy, adaptive K7 | 1.074 (1.024–1.123) | 1.251 (1.212–1.294) |
| C4 cohort tokens/s | | 156 -> 182 |

C1 uses the second pass only (the first baseline cell loaded the artifact before its 04:02 reconversion); C4 is n=66. Evidence: `profiles/bench/r9700-w8-kernels-20260927/overnight-economics/`.

## Rejected research ports (2026-09-27)

Candidates from the R9700 research campaign, measured and not adopted; their code is removed. Evidence: `profiles/bench/r9700-feature-ports-20260927/`.

| Candidate | Result |
|---|---|
| Fixed-budget W8 trees (K7) | root-sibling 0.87 (C1) / 0.88 (C4) of K5 chain; normalized best-first 0.97 / 1.01 (parity, never faster) |
| Exact LQ4 nibble repacking | complete-Op change 0.3% slower to 3.3% faster; word-at-a-time scheduling 35–62% slower at T1 |
| Grammar-mask overlap with target forward | default HIP serializes the graph; no measured benefit |
| 16-key dense-prefill attention | 22.5% slower at P2048/context 65536 |
| Conditional live-context verify split | some Ops 12–30% faster, but no matched whole-Engine gain |
| Shared GDN normalization / compact A/P+V64 prep | 13% / 2.83x slower complete Op |
| Post-idle keepwarm pulses | 1 ms pulses save 0.5 ms cached wake-up at ~12 W; 10 ms no benefit |
| DFlash committed-history copy drafting (output-identical) | 1.06–1.09x on verbatim-document and one-line-edit requests, but 0.980 (C1) / 0.975 (C4) paired decode rate on the general scenario corpus, where matches occur in ~4% of rounds; the neural drafter already accepts ~5 tokens/round on verbatim text, so copying saves only drafter compute. Not useful |

Horizon/tree economics: graphs on, 12 manifest scenario prompts x {greedy, two p-less seeds}, 768-token limit, ABC/CBA order over two passes, paired per-request decode rate against K5 chain. Outputs differ across verify widths, so this is a matched-workload, not token-identical, comparison.

## GPTQ rounding for LQ4 weights (2026-09-27)

Converter-only change (identical format, size and kernels, so speed is unchanged): GDN and MLP LQ4 projections are GPTQ-rounded against BF16-reference input second moments of 128 x 2048 calibration tokens from real opencode/OpenWebUI sessions plus code and news (recipe in `qwen3.8-27b-artifact.md`). Held-out layer output error falls 40-80% for GDN projections and 12-20% for MLP (after raising MLP-down damping to 0.3; it overfits at 0.01). Paired against the independently rounded artifact (`profiles/ppl/r9700-lq4-gptq-20260927/`):

| Evaluation | dNLL/token vs previous |
|---|---:|
| 4K prefill wiki / technical / code | +0.0047 / -0.0151 / -0.0146 (pooled -0.0083 +/- 0.0036) |
| 4K decode-schedule spans, pooled | -0.030 +/- 0.014 |
| 8K WikiText (4095 positions) | +0.0011 +/- 0.0036 |

Against BF16 at 8K: +0.0228 (previous +0.0218, within noise), top-1 agreement 91.06% (90.72%).

NIAH: standard and multikey 8K-128K greedy 20/20 each, sampled multikey 50/50 (8K) and 45/45 (32K-128K); at 240K (new fixtures, 239,969/241,006-token prompts, the opencode compaction point) standard and multikey greedy 5/5 each and sampled multikey 10/10, TTFT ~179 s (1350 tok/s prefill). Rejected: GPTQ on the attention projections too (8K multikey NIAH failed 5/10 sampled runs against 0/10); the news/llama.cpp-only calibration tied this one on PPL.

**BF16-source admission cells** (`profiles/ppl/r9700-admission-gptq-20260927/`, chunk 2048, G16, against the retained source-BF16 references; tiers from `tools/ppl/README.md`):

| Cell | dNLL vs BF16 | new severe (accuracy cap) | greedy flips | accuracy | capacity-speed |
|---|---:|---:|---:|---|---|
| 8K prefill | +0.0228 +/- 0.0039 | 2 (5) | 8.9% | fail (> 0.02) | pass |
| 8K decode, graph = eager | +0.0218 +/- 0.0040 | 2 (5) | 9.1% | fail (> 0.02) | pass |
| 32K prefill | +0.0185 +/- 0.0020 | 20 (17) | 8.4% | fail (severe) | pass |
| 32K decode, graph = eager | +0.0179 +/- 0.0020 | 16 (17) | 8.5% | pass | pass |

Same-route graph/eager decode is bit-identical at 8K and 32K (NLL and argmax), and whole-inference greedy tokens match graph/eager at C1..C4 (P2048/G256) and at P20480 for ordinary and DFlash.

**Output head as GPTQ LQ4 (promoted into `r9700-lq4`, 2026-09-28).** Measured as the evaluation identity `r9700-lq4-head-eval`, since folded into production. The Q4G64 target head re-encoded as LQ4 at the same 4.25 bits/weight, GPTQ-rounded against the final-norm second moment (damping 0.1; recipe in `qwen3.8-27b-artifact.md`). Same BF16-source cells (`profiles/ppl/r9700-head-eval-20260927/`):

| Cell | production dNLL vs BF16 | head dNLL vs BF16 | paired head - production | greedy flips |
|---|---:|---:|---:|---:|
| 8K prefill | +0.0228 | +0.0145 | -0.0083 +/- 0.0027 | 8.9% -> 7.0% |
| 8K decode | +0.0218 | +0.0135 | | 9.1% -> 7.3% |
| 32K prefill | +0.0185 | +0.0128 | -0.0057 +/- 0.0013 | 8.4% -> 7.8% |
| 32K decode | +0.0179 | +0.0128 | | 8.5% -> 7.7% |

New severe tokens are unchanged within noise (32K prefill 19 vs 20). Speed is unchanged (grouped A/B, `profiles/bench/r9700-head-eval-speed-20260927/aba/`: ordinary decode 36.95 vs 36.93 tok/s, DFlash round 31.41 vs 31.43 ms, prefill within noise). DFlash acceptance over ten 1024-token prompts x 512 greedy tokens (`.../accept2/`) is 3.72 vs 3.79 tokens/round: fewer rounds on 7 of 10 prompts, the total set by one low-acceptance fixture prompt (301 vs 245 rounds) where the greedy continuations diverge. NIAH (`profiles/bench/r9700-niah-lq4-head-20260928/`): standard and multikey greedy 20/20 each at 8K-128K and 5/5 each at 240K, sampled multikey 50/50 (8K) and 45/45 (32K-128K), identical to production.

**Attention gate/value and output GPTQ (promoted into `r9700-lq4`, 2026-09-28).** The 7 LQ4 attention gate/value and 13 output projections GPTQ-rounded (damping 0.1) instead of independently; query/key stays independent (GPTQ there failed 8K multikey NIAH). Paired against the head-promoted production artifact (`profiles/ppl/r9700-attn-vo-gptq-20260928/`):

| Cell | dNLL vs BF16 | paired - previous | new severe (cap) | greedy flips |
|---|---:|---:|---:|---:|
| 8K prefill | +0.0129 | -0.0017 +/- 0.0023 | 3 (5) | 7.8% |
| 8K decode | +0.0122 | -0.0013 +/- 0.0023 | 2 (5) | 7.4% |
| 32K prefill | +0.0113 | -0.0015 +/- 0.0012 | 13 (17), was 19 | 6.8% |
| 32K decode | +0.0099 | -0.0028 +/- 0.0012 | 13 (17) | 6.6% |

All four cells move the same way, and 32K prefill now passes its severe cap. NIAH (`profiles/bench/r9700-niah-attn-vo-gptq-20260928/`): standard and multikey greedy 20/20 each at 8K-128K and 5/5 each at 240K, sampled multikey 50/50 (8K) and 45/45 (32K-128K). Format, size and kernels are unchanged, so speed is too.

**GPTQ activation ordering (rejected, 2026-09-28).** Group-wise act-order (32-wide groups in descending diag(H) energy, columns within a group likewise, so each codebook search still sees one contiguous group) on every GPTQ role and the head lowered synthetic calibrated output error 2-6%, but against the attention-GPTQ production artifact it is worse in all four BF16-source cells: +0.0013 / +0.0043 / +0.0003 / +0.0038 nats/token (8K prefill, 8K decode, 32K prefill, 32K decode; `profiles/ppl/r9700-actorder-20260928/`). Not adopted; the code was removed.

**Protected FP8 projections as LQ4 (not selected, PPL-costing).** Re-encoding the 26 row-scaled FP8 protections as GPTQ LQ4 too (0.62 GB less per token) gives ordinary decode 36.78 -> 38.27 tok/s (+4.0%) and prefill 8K +0.6%, at +0.003 +/- 0.002 nats/token paired against the GPTQ artifact (8K +0.0022, 4K prefill +0.0036, 4K decode +0.0049); NIAH passes.

**Decode bound.** A DFlash P4096/G128 kernel profile of the GPTQ artifact puts the LQ4 small-T projections at 500-555 GB/s effective weight reads (gate/up 10.6 ms, accumulating projections 7.6 ms and GDN pair/convolution 4.3 ms per round), the drafter's A8Q4G64 at ~480 GB/s and the row FP8 projections at ~530 GB/s, against ~612 GB/s for a synthetic contiguous read: decode is within 10-20% of weight bandwidth, so further gains need fewer bytes per round. Dropping the two final byte permutes of the LQ4 decode (repacked nibble order, measured as a wrong-output ceiling) changes T1 by nothing, T6 by 0 to -3.8% and prefill GEMMs by -0.7 to -2.7%; not pursued.

**Measured or assessed out.** *KV values:* in the BF16-source cache-only scorer at 8K (FP8 keys, G16 values), replacing uniform INT4 levels with the LQ4 non-uniform magnitude shape (and a per-group scale search) changes NLL by -0.0003 +/- 0.0012 against G16 INT4; a six-way per-group scale search on the INT4 levels gives -0.0010 +/- 0.0009, half of the cache's whole +0.0019 cost at best. *GDN chunk-scan split:* precomputing the chunk-local U = T beta V, W = T beta Gamma K, P and the scaled Q/K operands would make the state scan independent per 16-row wave (384 CTAs instead of 48), but writes and rereads ~113 MB per layer per 2048 tokens (~0.38 ms of DRAM time against the fused kernel's 0.63 ms), so the estimated gain is ~1% of prefill; not implemented.

## Protected FP8 projections on the prefill GEMM (2026-09-27)

The 26 selective-cap FP8 projections ran T > 16 through hipBLASLt (MT128x128x32 solutions, 4.8% of 32K prefill kernel time). They now run on the LQ4 prefill GEMM with raw E4M3 staging (`fp8_row_scaled_prefill_linear`), with scales and per-token poison in the epilogue. Per call at T2048 (hipBLASLt from the 32K trace, ours from a scratch event-timed harness with random data): gate/up 4059 -> 3441 us, down 5120x17408 3063 -> 1717 us, attention input 7168x5120 910 -> 705 us, output 5120x6144 764 -> 600 us. The kernel sits at the same ~212-217 TFLOP/s as LQ4, which matches the recorded power bound at 300 W rather than a decode cost.

Qualification: the hipBLASLt-era FP8 gate/up qualifier's selective-protected and shared-context modes passed with zero BF16-step error against their FP64 oracles at T2047/T2048 (its default mode's A8Q4 probe failed identically on the previous commit). Against the BF16 reference at 8K (`tools/ppl/corpus.ids`, chunk 2048) dNLL is +0.02175 vs +0.02206 for hipBLASLt (paired -0.0003 +/- 0.0034), flips 380 vs 382; 4K code/wiki/technical PPL moves +0.33/+0.09/+0.08% (accumulation-order perturbation). Interleaved whole-prefill A/B (two pairs each): 32K +1.0% and +7.0% (noisy), 64K +0.8% and +1.4%.

hipBLASLt was then removed entirely (build dependency, `LinearExecution` library context, descriptors, heuristics and matmul workspace, and the hipBLASLt-specific qualifiers above); FP8 Linear qualification is now `ninfer_r9700_fp8_row_scaled_linear_qual` (small-T and prefill routes, FP64 oracle, per-token poison).

## Long-context prefill: six waves per SIMD and interleaved weight staging (2026-09-26, night)

A 32K prefill trace (`profiles/rocprof/prefill-32k-lq4-trace-20260926/`, attribution only) put 66% of kernel time in the LQ4 prefill GEMM and 18% in fused dense prefill attention. Both kernels sat just above the gfx1201 240-VGPR cliff (255 and 243 VGPRs, occupancy 5), so a WGP held one attention CTA or two GEMM CTAs and staging/barriers ran exposed.

- **Dense prefill attention** is capped at six waves per SIMD (`amdgpu_waves_per_eu(6)`, 240 VGPRs, no spill): each lane's causal context is re-read from LDS on masked blocks, and a nonfinite query poisons the denominator after the block loop. Two 12-wave CTAs now share a WGP. T2048 (`dense_prefill_attention_qual --time 2048 <context> 16 30`): 3.86 / 15.48 / 32.56 ms -> 3.50 / 13.97 / 27.96 ms at context 8K / 32K / 64K (~116 useful TFLOP/s at 64K). The split verification kernel shares the block arithmetic and is unchanged (234 VGPRs).
- **LQ4 prefill GEMM** is capped at 240 VGPRs (three CTAs per WGP instead of two) and decodes/stages the next slab before the last of the four 16-column chunks instead of after the last WMMA. T2048 (`ninfer_r9700_lq4_linear_qual --time`, random activations): gate/up 176 -> 207, down 180 -> 209, GDN v/z 175 -> 210, attention 188 -> 216, output 190 -> 216, GDN q/k 189 -> 216 TFLOP/s.

Neither change alters arithmetic order: the 4K code prefill PPL rerun reproduces the committed sum NLL bit-for-bit (1645.4861488342285, `profiles/ppl/r9700-cb4-final-20260926/`). Both kernels pass their FP64 oracles; the static attention profile now pins 240 VGPRs / occupancy 6.

Whole prefill, same session, same command (`ninfer_bench -p`, chunk 2048, C1, code corpus, `profiles/bench/r9700-long-prefill-20260926/`): 32K 2630 -> 2943 tok/s (+11.9%), 64K 2218 -> 2489 tok/s (+12.2%); attention alone accounted for +2.3% / +3.8%. The documented context ladder (DFlash backend loaded, max context 131200) reads 8K 3352, 32K 2894, 64K 2455 tok/s.

**Measured out or bounded.** Grouped LDS fragment prefetch in attention (PF 2/4/8: no change at six waves per SIMD), one-block-ahead page-table reads (no change), an LDS-only GEMM slab barrier (<1%), a split codebook read ahead of the decode (<1%). One LDS fragment per WMMA bounds this attention structure: an isolated FP16 WMMA loop reaches ~180 TFLOP/s but ~140 with one `ds_load_b128` per WMMA. The GEMM's decode/staging still costs ~20% (skipping it measured +24%); the isolated FP8 WMMA ceiling is ~340 TFLOP/s. The chunked GDN prefill (0.63 ms per 2048 tokens, ~4% of prefill) is latency-bound on one 8-wave CTA per value head; ablation attributes ~26% of it to the intra-chunk A/P products and ~13% to the q/k normalization loads, which a parallel precompute pass with a workspace would take off the serial chain (estimated 1-2% of prefill).

## LQ4 Text weights and per-token E4M3 activations (2026-09-26, evening)

The Text-layer Q4G64 projections of the selective-cap DFlash2 base were re-encoded from BF16 as `LQ4` (now `r9700-lq4`; codec in `tensor-formats.md`, recipe in `qwen3.8-27b-artifact.md`). Each 32-column group selects an eight-magnitude codebook whose values are exact E4M3, so every Linear runs FP8 x FP8 WMMA with only the FP32 row scale and the per-token activation scale applied after the K sum: prefill is a 256-token x 128-row GEMM that decodes weights into double-buffered LDS and loads activation fragments directly; T2..16 is a four-way K-split WMMA kernel; T1 is an FP8 dot4 GEMV. Epilogues split outputs, accumulate into the residual, or (MLP gate/up, stored interleaved) publish the SiLU-gated product directly. QK/PV attention and the typed cache are unchanged (no FP8 attention).

Interleaved A/B against the production Q4 selective-cap artifact, same binary, R9700 at the 300 W cap:

| Workload | Q4 selective-cap | LQ4 |
|---|---:|---:|
| prefill 4K tok/s | ~2380 | ~3050 (+28%) |
| prefill 8K / 32K / 64K (earlier LQ4 build) | | +36% / +18% / +16% |
| P4096/G128 DFlash ms/round | 31.37 | 31.43 |
| P4096 ordinary decode tok/s (512 tokens) | 36.70 | 36.92 |
| P65536 DFlash ms/round (earlier LQ4 build) | 34.87 | 35.49 |

Quality (`profiles/ppl/r9700-cb4-final-20260926/`, same windows): 4K prefill PPL wiki -0.33%, technical -1.48%, code -0.95% relative to Q4; 128-position decode-path spans +2.68%, +4.37%, +0.67% (short spans, mixed). Against BF16 at 8K, prefill dNLL +0.0224 vs +0.0354 for Q4 (paired -0.0131 +/- 0.0053), top-1 flips 378 vs 424, new severe 2 vs 8; decode path +0.0245 vs +0.0367 (paired -0.0123 +/- 0.0054), flips 382 vs 437, severe 3 vs 10. NIAH exact-answer 8K..128K x five positions passes 20/20 (standard) and 20/20 (multikey) (`profiles/bench/r9700-niah-cb4-20260926/`). Kernels are checked against an FP64 oracle of the decoded codebook (`ninfer_r9700_lq4_linear_qual`, T1..2048, pair, SiLU-pair, residual, per-token poison), the producers against theirs (`ninfer_r9700_fp8_producers_qual`), and the fused GDN front bitwise against its composition (`ninfer_r9700_gdn_lq4_front_qual`).

The per-token E4M3 image now carries one status word per token, plain-stored by the producing CTA; consumers poison only a flagged token's outputs. This removed the serial status CTA (40 us per SiLU-producer call) and every status memset; the SiLU producer itself is superseded by the SiLU-pair epilogue.

**Measured out.** Eight-byte group-code loads shared across four steps (10-40% slower than byte loads); small-T 8x2 and 2x4 K-split depths (4x2 best for verification; T1 took the GEMV); compile-time K specialization (no gain); accurate `expf`/division in the SiLU epilogue (3x code, 15-20% slower prefill; fast exp/reciprocal and branch-free BF16 rounding used instead).

## Decode launches, host stalls and packed T1 attention (2026-09-26, overnight)

Same setup (selective-cap Q4/FP8 DFlash2 artifact, code corpus, C1, draft 5). Each step was qualified before the next; ms/round is the comparable DFlash metric because some steps change the FP32 association and therefore the greedy trajectory.

| Step | P4096/G128 DFlash ms/round |
|---|---:|
| start of session | 33.39 |
| multi-CTA small-T A8G64 codecs with a status CTA (no reset launch) | 33.12 |
| fused GDN verification front (RMSNorm + a/b controls + codec, one kernel) | 32.73 |
| K6144 small-batch projections split over eight waves | 32.68 |
| GDN recorded convolution fused into the pair projection epilogue | 32.23 |
| pinned KV-resolution readback; no host wait for the replay fold | 31.91 |
| GDN front zeroes the gated output status word (no memset) | 32.03 (32.16 same run) |
| drafter SWA split partials in BF16 WMMA (72 -> 25 us per layer) | 31.78 |

Run-to-run host drift (cause not established) shifted later absolute values by about +0.2 ms (the session-start binary still measured 33.3); rows after it compare against the same run. P65536/G256: 37.10 -> 35.35 ms/round (session start vs end).

**Status CTA.** Small-T producers used one 1024-thread CTA (compute-bound on one WGP) or a status memset launch. They now spread one 16-byte vector per thread over many CTAs; one extra CTA rescans the rows for the nonfinite/overflow conditions (overflow exactly when |x| >= 8321040, checked over every finite FP32 value) and plain-stores the status word. T6: quantize 4.2 -> 2.6 us, normalized prepare 6.4 -> 4.8 us. The gated prepare keeps its memset: its status pass would repeat every SiLU/norm and measured 9.2 us.

**GDN front.** One CTA per control head normalizes all T rows with the eager row-CTA RMSNorm arithmetic, runs the unchanged control dot on the BF16 seam, and the token's owner CTA encodes it; CTA 0 sees every seam value and publishes the status. Bitwise the eager composition (new `ninfer_r9700_gdn_normalized_front_qual`).

**Pair + convolution.** For one sequence (W5..6) wave zero of each pair CTA holds all tokens of its 16 channels after the split combine, so the recorded causal convolution and the output-gate copy run in the epilogue (`gdn_pair_conv_record_bf16`, bitwise the composition; `ninfer_r9700_gdn_pair_conv_record_qual`).

**Host.** The segmented KV status/cursor readbacks landed in `std::array`s, so each copy into pageable memory was a staged synchronous copy (~140 us of idle GPU per round); they now use a pinned buffer. The commit tail no longer synchronizes before the next round unless a DFlash context append's ingress copy is pending. C4 (P2048/G96): 328.7 -> 332.2 tok/s.

**Packed decode attention for T1.** Ordinary T1 decode used FP8-Q/FP8-K WMMA scores, a separate Softmax and a vector PV (110 us per layer at 4K), and split-512 from 8K. Host-fixed 1..6-row decode now uses the packed split-context route that DFlash verification already used (BF16 Q, exact-BF16 FP8 K, FP16 probabilities, FP16 V/8, FP32 merge), so ordinary, MTP and DFlash share one attention arithmetic; split-512 remains for tree/device-count T4. The discriminator adds widths 1..4 (max |error| 2.4e-4 against the 2e-3 dense criterion); NIAH standard and multikey 8K..128K pass 20/20 each on ordinary decode. Further T1 work: the output projection with residual epilogue and the MLP (gate/up, fused SiLU codec, down) use the small-batch K-split kernels at T=1, the T1 normalized prepare shares the per-row small kernel (now the eager row-CTA RMSNorm reduction, so fused and unfused seams match bit for bit), and the fused GDN front covers T1.

| Ordinary decode, tok/s | 4K | 32K |
|---|---:|---:|
| session start | 33.30 | 23.47 |
| packed T1 attention | 34.56 | 31.56 |
| K-split T1 output projection, shared prepare | 35.68 | 33.74 |
| T1 MLP through the normalized small-batch route | 36.29 | — |
| fused T1 GDN front | 36.68 | — |

**Measured out.** Long-context packed attention stays near 400 GB/s at T1/64K (compute waves idle at the barrier, loaders waiting on global loads). A depth-2/3 register ring, 16-byte K/V loads, scalar page-table loads and 128 chunks changed nothing; page-granular (64-byte-row) K loads spilled. A synthetic read of the TokenFastest K plane in 16-key blocks tops out near 511 GB/s versus 612 GB/s for whole 64-byte rows, so the block-granular K access pattern is the limit. The prefill GDN convolution scatter was vectorized (987 -> 224 us per P2048 chunk) without an end-to-end change: it overlaps the side-stream output-gate projection.

## Attention and drafter boundary fusions (2026-09-26, day)

Same setup; every row is an A/B against the previous binary in the same session, and each step keeps the greedy trajectory (identical DFlash acceptance).

| Step | P4096/G128 DFlash tok/s |
|---|---:|
| start (e125d435) | ~143.8 |
| FP8 attention: RMSNorm into the E4M3 codec, one split-output QK/GV launch, gate into the codec, residual epilogue; T <= 16 FP8 status from one extra CTA | ~145.0 |
| Q4 attention: normalized split pair (T5/6), gate into the A8G64 codec with residual epilogue | ~145.1 |
| DFlash drafter Q/K RMSNorm + RoPE + V split in one Op (`dflash_qkv_norm_rope`) | ~145.5 |
| DFlash convolution finish accumulates into the residual | ~145.7 |
| six-row status pass of the T <= 6 normalized prepare | ~146.1 |

P65536/G256: 35.33 -> 34.93 ms/round. Ordinary decode and prefill are unchanged within noise. The attention fusions are bit-identical to the unfused chains (`ninfer_r9700_attention_fused_qual`); the drafter front is bit-identical at T <= 8 and within one rotation rounding above, where the eager combined RoPE kernel contracts differently (`ninfer_r9700_dflash_qkv_norm_rope_qual`).

**Measured out.** *MALL prefetch:* touching a 16 MB weight block costs about as long as reading it (33 us), a prefetched block is evicted by any intervening large projection, a forked Device Graph branch costs ~400 us, and extra prefetch CTAs inside the recurrence kernel added 11-15 us per layer in a synthetic layer sequence, so no form pays. *FP8 Q in packed decode attention:* a two-term (hi+lo) E4M3 query keeping BF16-equivalent precision was no faster; plain E4M3 Q saves 6-9% of the kernel (~1% of a 64K round) and was already rejected on real-tensor operator error. *GDN record kernel:* 2 or 8 row tiles instead of 4 were slower; it reads the 3.1 MB FP32 state at ~270 GB/s and is bound by the serial token transitions.

## Paired small-batch projections (2026-09-26)

Deeper per-wave prefetch (two or three G64 groups) did not move the narrow projections (N5120/K6144 72%, N12288 84% in isolation): each launch pays a fixed ramp and tail. The GDN query-key/value-z (N4096+N12288) and the Q4 attention (N7168+N7168) projections of one prepared activation now run as one launch (qualified against the public FP64 bound at T5/6/12). 4K DFlash 33.65 -> 33.36 ms/round; dispatches per round 1,232 -> 1,146.

## Fused Q/K normalization and RoPE (2026-09-26)

`ops::qk_norm_rope` replaces the two Q/K RMSNorms and RoPE of every full-attention layer (three launches, ~11 µs at T6) with one wave per head reading the projection planes directly; the tracing path keeps the separate Ops. FP64 criterion: one BF16 step per contributing operand across T1/6/130 and 1-D/MRoPE (relative L2 7.7e-4). P4096 DFlash 33.82 -> 33.65 ms/round; ordinary decode 32.7 -> 33.0 tok/s.

## DFlash verify attention: double-buffered 16-key blocks (2026-09-26, second batch)

Thread trace (`rocprofv3 --att`, available on gfx1201 with the installed decoder) of the 64K verify kernel: the compute waves spent 50% of their time at barriers waiting for the loaders' K/V conversion and 32% on LDS latency, WMMA 13%. Deeper register prefetch, 16-byte staging loads and page-table pipelining changed nothing, because staging could not overlap compute in one LDS buffer. Verification now stages 16-key blocks into two buffers (41.7 KB, still three CTAs per WGP): loaders convert block n+1 while the compute waves consume block n, with one barrier per block. The merge reads chunk origins/denominators once in parallel (same summation order).

| Context | Verify+merge before, µs (microbenchmark) | After |
|---|---:|---:|
| 4K | 51.6 | 39.5 |
| 16K | 136 | 111 |
| 64K | 384 | 302 |
| 128K | 703 | 544 |

In-model at 64K the verify kernel is 312 -> 248 µs and the merge 22 -> 10 µs per layer; P65536/G256 measures 37.2 ms/round (38.7 before). The softmax origin-raise schedule now advances per 16 keys, so greedy tokens can differ on near-ties; the FP64 leaf and route discriminators pass. Templating the shared block routine also changed the dense prefill kernel's code generation (same arithmetic, identical leaf-qualifier errors): 64K prefill attention fell from 12.4 to 9.2 s, and whole prefill measured 4K 2384, 32K 2086 (2004), 64K 1810 (1681) tok/s.

## DFlash decode: fewer launches, packed verify attention (2026-09-26)

Same setup. Round time is the comparable metric: the gated-RMSNorm seam change below alters the greedy trajectory of this sample (acceptance 3.88 -> 4.27), and prompts longer than the ~4.7K-token code corpus repeat it, which inflates acceptance at 32K/64K.

| Workload | Before, ms/round | After, ms/round |
|---|---:|---:|
| P4096/G128 | 34.93 | 33.94 |
| P15200/G128 | 36.2 | 34.93 |
| P32768/G128 | 37.14 | 36.20 |
| P65536/G128 | 40.46 | 38.66 |

Ordinary decode P4096/G256: 32.6 -> 32.7 tok/s. Dispatches per 4K round fell from 1,350 to 1,233 and status/zero fills from 101 to 70. **Launches.** The GDN output's gated RMSNorm now feeds the A8G64 codec directly at T1..8 (the prefill route's per-vector arithmetic, so its BF16 seam differs from the retired eager route; one thread per slot, 11.2 -> 3.8 µs with its status reset); the two Q4 attention input projections share one quantization and each is split into its two outputs by one copy kernel; the gate/up prepare also clears the fused-SiLU down status (one normalized-MLP Op, bit-identical to the two-Op composition); and the attention output buffer is zeroed only for padded columns. **Verify attention.** The (row, query head) pairs of a KV head's six query heads are packed into sixteen-lane tiles, so T6 needs three computing waves instead of six (62% of WMMA rows were padding); the other three waves are loaders that hold block n+1 in registers while block n is consumed. Per-pair arithmetic is unchanged; 64K W6 is 0.458 -> 0.312 ms per layer (packing alone 0.361), about 55% of KV bandwidth; chunk caps of 48 and 96 were no better than 64.

**Measured out.** Sampled drafter proposals under p-less T1.5 (the selector's softmax over its top-16 at draft temperature 1.0 or 1.5, with residual resampling on rejection): 7 prompt types x 3 seeds, 256 tokens, fixed K5. Greedy proposals accepted 2.59 tokens/round (52.3%), sampled 2.04 (T1.0) and 1.91 (T1.5), and greedy won in every category. Superseded 2026-09-28: a low draft temperature does beat greedy (0.4: +9.9% decode at p-less T1.5, upstream 5d6f6bf2 port, `docs/cli.md`); only drafting at the target temperature loses. Overlapping the GDN control projection with the input projection needs concurrent graph branches, which this ROCm serializes unless `DEBUG_HIP_GRAPH_SEGMENT_SCHEDULING=1` is set (a debug switch; it does not change the current single-branch graph).

## DFlash decode: K-split projections and latency-bound kernels (2026-09-25)

Same setup as the section below.

| Workload | Before, ms/round | After, ms/round | Decode tok/s before → after |
|---|---:|---:|---:|
| P4096/G128 (code) | 39.3 | 34.9 | 98.8 → 111.0 (identical acceptance) |
| P15200/G128 (code) | 40.6 | 36.2 | 126.0 → 141.3 |
| Ordinary decode P4096/G256 | — | — | 31.6–32.0 → 32.6 |

GPU busy time per 4K round fell from 36.7 to 32.1 ms. The small-batch A8Q4 projections (T2..24) now split K across 4 waves of one CTA (8 at N5120/K4096, 16 at N1280; N12288 keeps its route); one wave per 16-row tile left too few G64 payloads in flight on the narrow shapes. At T6: N34816/K5120 169→154 µs, N5120/K17408 94→79 µs, N5120/K6144 42→33 µs, N4096/K5120 29→21 µs, N1280/K5120 25→7.5 µs (85–94% of DRAM bandwidth for the large shapes). Only the FP32 association changes; the small-batch qualifier's FP64 bound still holds (89 cells, maximum budget use 0.888).

Latency-bound kernels: K5120 small-T RMSNorm keeps a row in registers, one 16-byte vector per thread (7.7→1.8 µs, also used by ordinary decode and T25..127 mixed decode/prefill units); the BF16 GDN control projection likewise (9.9→5.1 µs, still bitwise equal across T1..24); the GDN verify record splits value rows over four CTAs and stages every token's normalized q/k first (15→11 µs, bit-identical); the drafter's column top-16 keeps register lists with wave-shuffle merges (0.33→0.06 ms per round, exact); verification argmax and the row-scaled FP8 activation codec use 16-byte loads (146→30 µs; 29→2.5 µs at T6 x17408, also 6.0→3.3 ms per 4K prefill), both exact. Remaining per-round cost is mostly bandwidth-bound weight streaming, the LM heads, the GDN replay fold, and about 2.6 ms of launch gaps over ~1,350 dispatches.

## DFlash decode: split verify attention and small-width dispatch (2026-09-25)

Selective-cap DFlash-loaded artifact, C1, fixed five drafts, optimized proposal head, chunk2048, G16, ROCm 10. Round time is decode seconds divided by speculative rounds (acceptance varies with the generated text, so tok/s on one sample mixes speed with acceptance).

| Workload | Before, ms/round | After, ms/round | Decode tok/s before → after |
|---|---:|---:|---:|
| P4096/G128 (code) | 46.1 | 39.3 | 89.5 → 98.8 (33 rounds vs 31) |
| P15200/G128 (code) | 58.1 | 40.6 | 84.8 → 126.0 |
| Ordinary decode P4096/G256 | — | — | 29.9 → 31.6–32.0 tok/s |

**Verify attention.** W4..6 verification used an FP8-Q WMMA score plane, an FP32 softmax and a PV that walked the whole context serially per wave (0.26 ms at 4K and 1.82 ms at 32K for W6, 20–25x off KV bandwidth). It now runs the dense-prefill arithmetic split over context chunks (one CTA per KV head and chunk, at most 64 chunks of at least 256 keys) with a stable FP32 merge. Verification runs only the six row-tile-0 waves (192 threads, two staging items per thread), so three CTAs share a WGP and overlap one another's load phases, bit-identical to the twelve-wave form: W6 0.057 / 0.088 / 0.135 / 0.246 / 0.458 ms at 4K / 8K / 15.2K / 32K / 64K (4.5–8.8x the old route). Q stays BF16, which is more precise than the FP8-Q route; greedy DFlash output therefore no longer equals greedy ordinary decode bit for bit (the 4K sample diverges at a near-tie at token 8, the 15.2K sample is identical), while p-less sampling semantics are unchanged. A warp-specialized variant (three packed-head compute waves plus loader waves, raw FP8 K in a two-stage LDS ring) spilled 100–850 VGPRs in every form and was dropped; the six-wave CTA reaches about a third of KV bandwidth at 32K.

**NIAH on these builds.** Standard single-needle and a new multi-key variant (the target record among 32 same-form records, four with near-miss names; `make_niah_positions.py --multikey`) at 8K/32K/64K/128K x five depths, DFlash, no prefix reuse, thinking off: greedy 20/20 each; production sampling (p-less, temperature 1.5, adaptive DFlash, optimized head) 60/60 multi-key runs (three seeds) and 40/40 standard runs (two seeds). Evidence: `profiles/bench/r9700-niah-dflash-20260925/`, `profiles/bench/r9700-niah-pless-20260925/`.

**Small-width dispatch.** A8G64 activation quantization up to 8,192 vectors runs as one self-resetting 1024-thread CTA (no status memset launch); the GDN query-key/value-z pair shares one quantization at every width; the gate/up RMSNorm feeds the codec directly for all T >= 2 (a one-CTA prepare with a deterministic per-row reduction up to 12 rows); the DFlash drafter's SWA consumes eight context keys per step (0.315 → 0.179 ms at T6/W4096). Dispatches per 4K round fell from 1,935 to 1,560.

**Residual epilogue at verify widths.** The small-batch N5120 projections (attention and GDN output, MLP down) add their BF16 result to the residual in the GEMM epilogue, and the SiLU of the down input feeds the codec directly at every small-batch width (previously T2048 only): both are bit-exact to the composition, 40.6 → 39.6 ms per 4K round.

**FP8 projections.** The row-scaled E4M3 attention projections (N7168/K5120) ran through hipBLASLt at about 50% of bandwidth for T <= 16 (116 µs per call). A dedicated route (one 128-thread CTA per 16 rows, native FP8 WMMA, four ordered K quarters, in-kernel poisoning) takes about 70 µs; it serves verification and ordinary decode. PPL-4K prefill is unchanged; decode 6.647/23.019/3.235 versus 6.676/23.038/3.220 (mixed sign). Evidence: `profiles/bench/r9700-dflash-decode-20260925/`, `profiles/rocprof/r9700-dflash-decode-*-20260925/`.

## Prefill follow-up: producers, DFlash inventory, chunked GDN (2026-09-25)

Same setup as the base-prefill campaign below (selective-cap model, C1, chunk2048, dense G16, `auto`, ROCm 10). Whole 4K is P4096/G128 on the DFlash-loaded artifact (median of three after one warmup); 8K–64K are `context_ladder.py` screens with DFlash loaded and no speculative rounds.

| Build | 4K prefill | 8K | 32K | 64K |
|---|---:|---:|---:|---:|
| Base campaign (committed `276d0926`, rerun) | 2160.7 | 2023 | 1735 | 1516 |
| + producers, DFlash M128 inventory, chunked GDN | **2295.9** | **2197** | **1889** | **1613** |

**Producers (bit-exact or oracle-qualified, PPL-neutral).** The GDN a/b control projections use one split-K BF16 WMMA kernel for every T > 24 (57 µs at T2048 versus two 250 µs BF16 GEMMs); the public `normalized_linear` gained an A8 prefill route that normalizes straight into the A8G64 codec; the GDN output leaf owns its gated RMSNorm and feeds the codec directly (1.06 versus 1.17 ms per layer-chunk); the GDN query-key and value-z projections share one activation quantization (bit-exact). A gate/up GEMM with a SiLU+A8G64 epilogue was built bit-exact but rejected: its epilogue VALU serializes with WMMA in this issue-bound kernel (8.67 versus 8.26 ms for the MLP chain) while the standalone SiLU kernel is memory-bound at 0.33 ms.

**DFlash context projections.** The DFlash2 feature projection (5120x25600) and per-layer fused QKV (6144x5120) were outside the explicit M128 inventory and fell back to the wmma32 kernel (17.2 and 3.3 ms per chunk-call at 32K). Both now run the M128 kernel after the FP64 regression; DFlash-loaded 32K prefill rose from 1840 to 1860 tok/s.

**Chunked GDN prefill.** The per-token staged recurrence (1.3 ms per layer-chunk at about half the non-dual-issue FP32 VALU rate) is replaced by a chunked (WY) route: one workgroup per value head walks 64-token chunks with FP16 WMMA products and FP32 accumulation, keeps each wave's 16 state rows as S^T WMMA accumulators that feed the next chunk directly, solves only the four FP32 diagonal blocks of (I + A)^-1 and forms the corrected values by block forward substitution. It initially used no workspace (the 2026-10-16 shared-preparation update above adds transient workspace for T>=256). P2048 was about 0.7–0.87 ms per call. Output error versus FP64 stays at the BF16 floor (rel-RMS 1.73e-3 versus 1.66e-3 for the staged route); the FP32 state carries about 5e-4 relative error on long synthetic runs (no growth with length), inside the GDN Op criterion. Interleaved: 4K 2242→2306, 32K 1874→1930 tok/s (eval artifact); 4K prefill PPL 6.4610/9.3909/2.2548 versus 6.4534/9.4455/2.2497 (mixed sign, noise).

**Second batch.** The chunked GDN kernel was made branch-free on padded tokens and its load phase restructured (P2048 0.92→0.64 ms); the route now starts at T16. The two FP8 attention projections share one E4M3 activation quantization, and the attention output's FP32→BF16 cast and sigmoid gate run as one exact kernel. Interleaved: 4K 2309→2336, 32K 1924→1946 tok/s; PPL-4K unchanged. Under the 300 W cap the random-code gate/up GEMM runs at ~2.46 GHz, versus ~2.88 GHz for all-zero codes at the same power: the GEMM is power-bound, and only a higher power limit or a voltage offset (driver overdrive, disabled on this host) would raise its clock.

**Third batch.** The residual add moved into the M128 GEMM epilogue (bit-exact BF16(residual + BF16(projection))) for the GDN output, SiLU down and attention output projections; the P2048 GDN leaf normalizes its input once, publishing the BF16 rows for the control projection and quantizing them for the shared projections; and the value-z matrix is projected as separate value and output-gate halves, with the output-gate GEMM on a side stream that overlaps the convolution and the 48-CTA recurrence. Interleaved: 4K 2350→2380, 32K 1954→1980 tok/s; PPL-4K neutral. Prefill graph capture (~2 µs saved per dispatch, ~0.2%) and a double-buffered or register-prefetched dense attention were measured out: one 37.9 KB attention CTA already fills a CU, gfx1201 has no VMEM-to-LDS load, and a register prefetch spills at the 256-VGPR limit.

**Chunk size.** Chunk 4096 remains 9–13% slower than 2048 (4K 1954 vs 2199, 8K 1847 vs 2130, 32K 1671 vs 1838): the P2048-specialized fused routes do not apply and the GEMM is already compute-bound.

**Rejected formulations.** An IU8 GEMM (signed A8 x offset-binary W8, correction folded into the WMMA seed) is only 1.5–1.9% faster in a register/LDS-only inner-loop microbenchmark before its W4→W8 staging and larger LDS stage: the per-element two-sided G64 scale epilogue, not the high-nibble shift, bounds the A8xW4 loop. Production reaches about 85% of that inner-loop bound.

**FP8 QK, operator gate.** Plain E4M3 Q (scale 1.0) had kept PPL within noise, but FP64 attention on real post-RoPE Q/K/V of all 16 attention layers (wiki and code, 4K) shows added output error versus BF16 Q with p99 up to 38% and maximum 41% in layers 23–39 (227 of 768 layer-head-text cells above 10% p99; the entire INT4-V error is at most 15%). The Q range is benign (max |q| 16.4, no saturation); the 3-bit mantissa is the error, so power-of-two Q scales cannot fix it. A split Q (E4M3 high part plus E4M3 residual, two FP8 WMMAs per tile) adds at most 0.09% but is only 0–1.9% faster (32K/64K), because the plain route's gain came from halving WMMA work. Neither is selected. Evidence: `profiles/bench/r9700-fp8qk-gates-20260925/`.

NIAH (exact answer, 8K/32K/64K/128K x start/q25/mid/q75/end, greedy C1, no prefix reuse): the committed base-campaign build passes 20/20 and the follow-up build passes 20/20. Whole-4K decode is 30.2 vs 29.8 tok/s in this pair; a decode-focused interleaved A/B shows no difference (31.2/30.9 vs 31.1/30.9). Final comparison: `profiles/bench/r9700-prefill-followup-final-20260925/` (`run.sh`). Evidence: `profiles/bench/r9700-niah-baseline-20260925/`, `r9700-niah-n2-chunked-20260925/`, `r9700-n2-producers-20260925/`, `r9700-gdn-chunked-20260925/`.

## Base-prefill compute campaign (2026-09-25)

Same installed selective-cap model/companion, C1, chunk2048, dense G16, R9700 in `auto`, ROCm 10. 4K is P4096/G128 whole inference (median of three after one warmup, code text); 8K–64K are cycled-code prefill screens (warmup0/repetition1, loaded DFlash K5, no speculative rounds).

| Step | 4K prefill | 8K | 32K | 64K | Worst 4K prefill PPL vs 5090 NVFP4 |
|---|---:|---:|---:|---:|---:|
| Start: gate/up A4, score-panel attention | 1687.6 | 1407 | 951 | — | +1.56% |
| Uniform A8 + M128xN128 A8 GEMM | 1725.8 | 1512 | 970 | 678 | +0.008% |
| + fused dense prefill attention (v1) | 1971.9 | 1853 | 1591 | 1349 | — |
| + fused attention v2 | — | 1841 | 1643 | 1428 | — |
| + GEMM pipeline and staged GDN | **2173.8** | **1976** | **1765** | **1514** | **+0.056%** |

Ordinary decode is unchanged (30.1–30.2 tok/s). Current 4K prefill PPL (WikiText/technical/code) is 6.457054/9.445386/2.254297 versus NVFP4 6.481045/9.576387/2.253032; teacher-forced decode is 6.716401/23.151247/3.248450 (128 positions; paired differences from the first A8 build are within about one standard error). Evidence: `profiles/bench/r9700-a8-gateup-20260925/`, `r9700-fused-attention-20260925/`, `r9700-gdn-gemm-20260925/`.

**Uniform A8.** Gate/up A4 and A8 ran at nearly the same speed because neither Q4 GEMM is matrix-issue bound: on gfx1201 WMMA and VALU issue serialize (measured), and the per-G64-group two-sided scale epilogue costs about one VALU slot per WMMA cycle. The new A8 prefill kernel uses an eight-wave M128xN128 CTA with 2x4 fragments per wave, token tiles fastest in the grid, an exact magic-number I32-to-FP32 epilogue (high-nibble chain seeded with 0x04B40000), scale-first scalar-base U32 loads, a peeled final group and WMMA/VALU interleaving. It is bit-exact to the former A8 kernel (FP64 regression over six Text and two MTP shapes at T=1024–8192). Gate/up N34816/K5120/T2048 on worst-case random codes: former A8 6.2, shipped A4 5.4, new A8 5.39 ms. The default is now `NINFER_R9700_Q4_PREFILL_A4_FAMILIES=0`. Under this load the card sits at its 300 W limit near 2.44 GHz (a register-only WMMA loop holds 2.91 GHz), and all-zero data is 14% faster than random data, so the GEMM is power- and epilogue-bound at about 157 useful TOPS.

**Fused dense prefill attention.** The QK / maximum / FP32-score-plane / split-PV / merge stages and their 0.5–0.74 GB caller workspace are replaced by one kernel (see `docs/maintainer/softmax-attention.md`). Complete-Op at 2048 appended rows: 32K 142→20.8 ms, 64K 248→44.2 ms (~77 TFLOP/s at 32K). The dense FP64 qualifier passes through 262144 context, both FP16 scale extremes, graph replay and every poisoning case; the kernel uses 246/244 VGPRs and no scratch. 32K WikiText PPL versus the same build with score-panel attention: mean NLL −0.0021 (final 512 positions) and +0.0010 (last 16383; 9 new/4 resolved severe).

**GDN recurrence.** The P2048 sequential recurrence was barrier- and LDS-bound (~14% of VALU throughput). The staged route precomputes normalized FP32 q/k in lane order and exp(g), then runs four state rows per wave with no workgroup barriers, DPP row reductions and next-token prefetch. P2048 state and outputs are bit-exact to the former sidecar route; 2.85→1.39 ms per layer call.

Current 4K kernel service: Q4 A8 GEMM 72%, GDN recurrence 7%, FP8 hipBLASLt 3.7%, attention 2.9%, BF16 GDN controls 2.6%, activation quantization 4%, norms/residual/convolution about 4% (`profiles/rocprof/r9700-gdn-gemm-4k-20260925/`). At 32K the GEMM is about 58% and attention about 15% (`profiles/rocprof/r9700-fused-32k-20260925/` shows the v1 split).

**FP8 QK evaluation (not selected).** Casting represented BF16 Q once with the decode route's saturating E4M3 conversion and using FP8xFP8 WMMA against the stored FP8 keys (no K decode) (superseded by the operator gate in the follow-up section above) makes the attention Op 1.47–1.56x faster and whole prefill 8K 1955 (no gain), 32K 1864 (+5.6%), 64K 1671 (+10.4%). Model quality is unchanged within noise: 4K prefill PPL +0.002%/+0.12%/−0.08% (WikiText/technical/code), 32K mean NLL +0.0021±0.0042 (512 positions) and −0.0005±0.0007 (16383; 5 new/6 resolved severe). Operator error versus FP64 rises 10–50x, beyond the BF16 route's 2e-3 criterion, so adoption would need its own criterion as decode's FP8-Q route has. Evidence and binaries: `profiles/bench/r9700-fp8qk-eval-20260925/`. FP8 probabilities for PV (Sage-style) were not evaluated.

## Earlier matrix-PV crossover (2026-09-25)

Matrix PV now begins at2048 visible tokens with2 key splits, uses4 from4096, 16 from12288 and32 from32768. Shorter contexts retain scalar PV. Retuning the matrix kernel is necessary because the old scalar cutoff/split count was not its best schedule. No weight, Linear activation or cache format changes.

Seven rotating-order warm complete-Op medians,2048 queries, auto:

| Group/context | Previous scalar PV, ms | Selected matrix PV, ms |
|---|---:|---:|
|G16/2048|8.62439|6.00138|
|G16/4096|36.6397|26.5062|
|G16/8192|76.0053|49.9984|
|G32/2048|8.60507|5.97431|
|G32/8192|70.8930|49.2036|

Both128-query appended calls and8192-query initial calls also improve. A separate 8/16/32/64-split long-context sweep found only a2.4% complete-Op advantage for16 versus32 at32K, with16 losing at64K; retain the current long-context schedule. This is not evidence of a universal optimal split count.

G16/G32 FP64, graph replay, metadata/canary and new2K/4K boundary cases pass. The planner's C1–4 checks and envelope maxima immediately around both new cutoffs pass. Two/four-split specializations use eight native FP16 WMMA sites,94VGPR, 13816-byte LDS, wave32 WGP and no scratch/spills. The global maximum workspace envelope remains510.6796875MiB; a standalone initial2K call now needs480.5625MiB.

Matched32K WikiText, final512 scored positions: PPL6.506943806327→6.439215091077302, mean-NLL delta−0.0104632322, zero new severe positions. This passes the unchanged quality gate, but one window does not establish a broad quality improvement. Matched whole C1/chunk2048 screens improve8K1305.40→1490.14 tok/s (tail1048→1256) and32K930.54→967.01 tok/s (tail617→620,35.21→33.89s). Same installed model, cyclic code corpus, warmup0/repetition1 and auto power as the preceding comparisons. The64K confirmation improves661.61→672.05 tok/s,99.06→97.52s; tail remains405. Whole reports: `profiles/bench/r9700-early-wmma-contexts-20260925/`. The refreshed `local/ninfer-r9700:compose` image passes its server executable smoke check and the same32K GPU PPL case: NLL and argmax sidecars are byte-identical to the native build, with matching artifact identity and precision metadata. Command and outputs: `profiles/ppl/r9700-final-image-20260925/`. The server remains stopped; the previous runtime image is retained with the `-rollback` tag. Evidence: `profiles/bench/r9700-dense-pv-tiles-20260925/wmma-short-splits*`, `early-wmma-*`, and `profiles/ppl/r9700-early-wmma-20260925/`.

## Long-context follow-up exclusions and attribution (2026-09-25)

The final selected-region32K trace attributes33.607s of prefill/setup kernel service: all matrix PV25.98%, QK17.87%, Q4A8/Q4A4 projections17.84%/15.86%, GDN recurrence6.65%, score maximum5.14%, and PV merge1.00%. Command, trace and attribution: `profiles/rocprof/r9700-dense-final32k-20260925/`. The investigated staged-route changes no longer reveal another demonstrated material win. This is not a hardware ceiling: fused streaming QK/softmax/PV could avoid the score plane, but would be a distinct algorithm requiring new oracle and model qualification, not an admitted continuation of these kernels. The canceled128K whole-model run was not repeated;128K/262K evidence here is Op-only.

Before the earlier crossover, the selected-region32K trace attributed34.774s of prefill/setup kernel service: PV WMMA16-split19.14%, QK32x64 16.91%, short-context scalar PV6.69%, with Q4A8/Q4A4 projections17.23%/15.32%. This is attribution, not unprofiled throughput or proof of a hardware ceiling. The reproducible command and trace are in `profiles/rocprof/r9700-qk-pv-final32k-20260925/`.

Rejected: materialize FP32 probabilities in the maximum pass, then consume them in both PV feature blocks instead of computing each exponential twice. Complete outputs are bit-exact and pass the independent FP64 oracle, but seven rotating-order warm complete-Op medians regress185.221→200.569ms at32K and287.667→326.321ms at64K (G16,2048 appended query rows, auto). The extra score writes outweigh saved PV arithmetic. Production retains raw FP32 scores and computes probabilities in PV. Task-local source and logs: `profiles/bench/r9700-dense-pv-tiles-20260925/` (`probability_candidate.hip`, `probability-p32768.log`, `probability-p65536.log`); candidate0 is the incumbent and candidate1 materializes probabilities. The legacy log field `splits=1` is only a comparison label; actual partitions are32 in both cases.

Also rejected: wave-owned register probability operands with a shuffle-reduced FP32 denominator. This removes probability LDS staging (13816→4600 bytes) but changes coalesced score loads into strided per-query access and raises registers 94→105. FP64 passes (relative-L2 difference from incumbent about5e-8), but complete Op medians regress184.929→259.731ms at32K and289.164→420.357ms at64K. These observations reject this mapping; they do not establish individual stall causes. Source and logs are `register_candidate.hip` and `register-p*.log` in the same task directory.

Not promoted: remove only the duplicate FP32 probability LDS tile, and accumulate the already-rounded FP16 probability operands in FP32 for the denominator. This preserves coalesced score loads and reduces LDS13816→7672 bytes (93VGPR). FP64 passes, but complete-Op medians only improve185.301→184.049ms at32K and 288.522→285.029ms at64K. There is no demonstrated material whole-model benefit to justify changing another numerical boundary; no model-quality promotion was attempted. Production retains FP32 probabilities for the denominator. Source/logs: `half_den_candidate.hip`, `half-den-p*.log` in the task directory.

## Dense QK key/query reuse (2026-09-25)

For calls with at least512 query rows, production QK now uses32-query/64-key tiles, falling back to the revised16-query/32-key tile when score panels shrink below32 rows. Smaller whole calls retain16x16. Two sets of six GQA waves share decoded keys; each query fragment serves four K16 WMMA fragments. No operand format, score accumulation order, cache, workspace or model recipe changes.

Qualified complete-attention warm median A/B, G16/2048 appended queries, auto, seven rotating-order samples; candidate output is bit-exact to the incumbent:

| Context | Previous, ms | Selected, ms |
|---:|---:|---:|
|2048|9.62679|8.70563|
|8192|96.2533|73.3662|
|32768|272.262|184.782|
|65536|391.505|287.647|
|262144 (16-row panels)|1214.28|1165.31|

The wide route regresses the final narrow-panel shape, so it is not used there. Production and prototype complete-Op timings agree at32K (184.762/184.753ms). G16/G32 independent FP64, graph/metadata and131071/131072/131073 panel-boundary qualification passes. Native BF16 WMMA sites32/64, metadata VGPR41/65, assembler next-free VGPR97/145, LDS16480/32928, wave32 WGP and zero spills/scratch. The static checker distinguishes the two register fields, and the selected-P2048 hardware audit now resolves the exact32x64 symbol. Its17 affected tests pass.

Matched32K WikiText NLL and argmax sidecars are byte-identical to the previous PV-WMMA build (PPL6.506943806327). Whole C1/chunk2048 prefill825.07→930.54 tok/s, tail495→617,39.72→35.21s. The8K screen is1305.40 tok/s versus1240.21 before QK tuning. Same loaded model, cyclic speed corpus and warmup0/repetition1 method; these screens do not establish a performance ceiling. The64K confirmation improves576.82→661.61 tok/s (+14.70%), tail335→405, 113.62→99.06s. This is4.38× the original pre-fix64K average rate. Evidence: `profiles/bench/r9700-dense-pv-tiles-20260925/qk-*`, `profiles/bench/r9700-qk-tiles-contexts-20260925/`, and `profiles/ppl/r9700-qk-tiles-20260925/`.

## Long-context matrix-instruction PV (2026-09-25)

The next PV route replaces scalar split accumulation with native FP16 WMMA and FP32 accumulation:16 query rows,128 value features and six waves per block. Probabilities are formed in FP32, rounded to FP16 for the numerator, and summed in FP32 for the denominator. Represented signed-INT4 times stored-FP16-scale values are divided by8 before FP16 conversion and restored by8 in FP32 before the split merge, preventing finite operand overflow without changing the cache. At this checkpoint the12K/32K dispatch boundaries, split counts and workspace envelope were unchanged; the later earlier-crossover section above supersedes the short-context selection. This is an explicitly qualified private arithmetic change, not bit-exact equivalence or a Linear activation recipe change.

Layer0 mechanism: reduce PV's scalar multiply/accumulate issue cost. Direct BF16, high/low corrected BF16 and FP16 operands were compared at64/128/256 feature tiles; FP16/128 was faster and more accurate than direct BF16. Corrected BF16 was closer to the FP64 oracle but materially slower. Nonperiodic complete-Op A/B, including non-power-of-two FP16 scales, same warm rotating-order seven-sample method and2048 appended query rows:

| Context / group | Scalar split incumbent, ms | FP16 WMMA, ms |
|---|---:|---:|
|12288 / G16|129.323|104.423|
|65536 / G16|542.325|395.813|
|65536 / G32|542.736|393.043|
|131072 / G16|911.337|612.284|

The complete G16/G32 FP64/graph/metadata suite passes through262144 context, including non-power-of-two scales, minimum-subnormal and maximum-finite scales. ISA confirms eight native `v_wmma_f32_16x16x16_f16` instructions per loop body, 94 next-free VGPR,13816-byte LDS, compiler occupancy14, wave32 WGP and zero spills/scratch. Matched32K WikiText final512-position PPL6.510527→6.506944, mean-NLL delta −0.00055047, zero new severe positions; maximum absolute per-position change1.20241. The existing gate passes without a tolerance change; this is not a broad quality claim. The superseded scalar split implementation and comparison controls are absent from production. Evidence: `profiles/bench/r9700-dense-pv-tiles-20260925/wmma-*` and `profiles/ppl/r9700-wmma-pv-20260925/`. Matched whole32K (same C1/chunk2048/K5-loaded prefill-only workload) improves 659.21→825.07 tok/s (+25.16%), tail382→495,49.71→39.72s. This is an unprofiled warmup0/repetition1 screen supported by the qualified rotating-order Op results, not a ceiling claim. Reports: `profiles/bench/r9700-wmma-pv-contexts-20260925/`. Completed64K confirmation:418.77→576.82 tok/s (+37.74%), tail237→335, 156.50→113.62s. Relative to the original pre-fix baseline,32K is2.02× faster and64K is3.82× faster. The canceled128K whole-model run remains unmeasured.

## Long-context PV reuse and split tuning (2026-09-25)

The follow-up profile of `0c084300` attributes42.14% of32K prefill/setup GPU service to PV,20.87% to QK,3.54% to score maximum, and0.54% to partial merging: `profiles/rocprof/r9700-balanced-pv-final32k-20260925/`. The next bounded sweep crossed query tiles1/2/4/8/16 with16/32/64 key splits. Eight query rows recover V reuse while keeping enough independent blocks; emitted register use drops109→68, compiler occupancy12→16, with zero spills. G16/G32 LDS is5912/5656 bytes. Tiles1/2 and64-way splitting did not win.

The first follow-up (`e3e033d2`) selects eight query rows,16 splits at12288–32767 context and32 splits from32768. Complete attention includes unchanged QK/max, every panel and merge; seven rotating-order warm event samples, R9700/gfx1201/ROCm10, auto,2048 appended queries, fragmented pages and nonperiodic represented inputs:

| Context / group | Previous tile4/split16, ms | Selected, ms |
|---|---:|---:|
|12288 / G16|135.968|130.136|
|32768 / G16|360.188|345.588|
|65536 / G16|611.195|563.770|
|65536 / G32|606.189|562.194|
|131072 / G16|990.466|909.065|

The full G16/G32 FP64/graph/invalid-metadata qualification passes, including both dispatch boundaries and262144 context. The workspace envelope covers both split regimes, including small query counts; the default2048-row envelope is unchanged. Changing only query tiling is bit-exact;32 splits change FP32 association. Matched32K WikiText final512-position PPL is6.510527 versus6.437947 for `0c084300`, delta mean NLL+0.01121072 with zero new severe positions, within the existing accuracy gate. This is one matched window, not a broad quality claim. Raw evidence: `profiles/bench/r9700-dense-pv-tiles-20260925/joint-*` and `profiles/ppl/r9700-joint-pv-20260925/`.

Matched whole C1/chunk2048/max-context131200, same selective-cap model and cycled code IDs, loaded K5 DFlash without speculative rounds, warmup0/repetition1: 32K prefill629.06→659.21 tok/s, tail369→382,52.09→49.71s. This is a screening measurement supported by the rotating-order Op sweep, not a ceiling claim. Whole reports: `profiles/bench/r9700-joint-pv-contexts-20260925/`. The completed64K confirmation improves393.37→418.77 tok/s (+6.46%), tail216→237, 166.60→156.50s, with the same unchanged default workspace envelope.

## Long-context dense PV fix (2026-09-25)

The first fix (`0c084300`) uses four query rows per PV block and16 key splits at visible context>=12288, followed by an FP32 partial-sum merge. Shorter contexts retain the16-row unsplit route. This fixes the collapse in available parallel work as the bounded score buffer forces smaller query panels: at128K a full panel now launches512 PV blocks instead of8. It does not change weights, activation quantization, the three cache planes, QK arithmetic, or the softmax maximum. FP32 reduction association changes, so model outputs are not promised bit-exact to the preceding build.

Panels now divide the query's16-row tiles evenly, retaining the minimum panel count instead of leaving a near-empty final grid. At6144 context/2048 query rows, this changes672/672/672/32 rows to512/512/512/512, preserving unsplit output bit-for-bit and improving the measured complete Op54.98→53.50ms. At12288, the split route plus balancing improves163.49→135.36ms with nonperiodic input.

Matched C1/chunk2048/max-context131200/workload, installed selective-cap DFlash companion, K5 loaded but no speculative rounds, same cycled code IDs, R9700/auto, warmup0/repetition1; model loading excluded:

| Context | Average prefill before → after, tok/s | Trailing rate before → after, tok/s | Prefill seconds before → after |
|---:|---:|---:|---:|
|8192|1161.88 → 1240.21|952 → 962|7.05 → 6.61|
|16384|794.00 → 888.71|388 → 555|20.63 → 18.44|
|32768|409.00 → 629.06|177 → 369|80.12 → 52.09|
|65536|150.97 → 393.37|61 → 216|434.09 → 166.60|

These are single-run whole-request screens, not a statistical ceiling claim; 8K remains below the split dispatch threshold; only panel balancing changes there, so its small whole-run timing difference is not attributed entirely to the fix. Trailing rates retain the prorated one-second semantics below. The128K **whole-model** attempt remains canceled; it was not repeated. At the128K **operator shape**,2048 query rows with appended causal positions improve7478.94→993.55ms (7.53×), including QK, maximum, PV, merge and every panel. At32K the same G16 operator improves776.43→358.72ms; G32 with nonperiodic input improves770.85→358.61ms. These warm-operand medians use seven rotating candidate-order event samples, not profiler timing.

The independent represented-input FP64 qualification passes both groups through 262144, including12K boundary cases, partial panels, nonperiodic cache data, fragmented pages, active/inactive rows, malformed metadata, output/workspace guards and poisoned graph replay. Native leaf and C1–4 host workspace/graph planner checks also pass. Split PV uses109 VGPR and4264/4008-byte G16/G32 LDS; merge uses8 VGPR and4-byte LDS. Both retain wave32, native FP32 arithmetic and zero scratch/spills. No private candidate flag or alternate product route is exposed.

Matched32K WikiText PPL over the final512 scored positions changes 6.54289475→6.43794659 (−1.60400%; mean-NLL delta−0.01617005), with zero new severe positions. Worst absolute per-position NLL change is1.87492; the mean does not mean every score is improved or unchanged. This meets the existing accuracy tier on this window, not a general quality improvement, fresh5090 comparison or128K quality result. Caller-owned partial storage adds at most126.4921875MiB; matched benchmark workspace capacity rises608,387,072→741,023,744 bytes. There is no allocation inside the Op.

A focused32K trace of the first split implementation (16K crossover, before panel balancing) reduced PV service50.972→23.161s and final-chunk service11.657→5.612s. The final chunk's PV dispatches increased32→2048 blocks; merge was only0.230s, 0.44% of total prefill/setup GPU service. QK remained10.77s. This supports the parallelism mechanism, not a bandwidth-saturation claim. Retained attribution: `profiles/rocprof/r9700-dense-pv-selected32k-20260925/`; final unprofiled timing above uses the12K crossover and balanced panels.

Direct qualification, ISA/resources and sweep evidence: `profiles/bench/r9700-dense-pv-tiles-20260925/`. Whole results: `profiles/bench/r9700-balanced-split-pv-contexts-20260925/`. Matched quality: `profiles/ppl/r9700-dense-pv-baseline-20260925/` and `profiles/ppl/r9700-balanced-split-pv-20260925/`. Dense attention still has increasing causal work and materializes FP32 score panels. This fix does not establish bandwidth saturation or eliminate quadratic long-context cost; a fused streaming attention redesign is a separate possible follow-up, not an already measured win. XAttention remains unpromoted.

## Pre-fix dense prefill context scaling (2026-09-25)

The installed `r9700-q4-fp8-selective-cap-n16k16-dflash2-q4-eval` artifact, mixed gate/up A4 profile, C1, chunk2048, max-context131200, workload KV capacity, loaded DFlash K5/optimized head, device0/R9700 in`auto`: unprofiled pure-prefill screen, warmup0/repetition1, no speculative rounds. The4096-token code corpus is cycled to each requested length; this is timing evidence, not natural long-context quality or representative XAttention sparsity evidence.

| Prompt tokens | Average active prefill tok/s | Trailing prefill tok/s | Prefill seconds |
|---:|---:|---:|---:|
|8192|1161.88|952|7.05|
|16384|794.00|388|20.63|
|32768|409.00|177|80.12|
|65536|150.97|61|434.09|
|131072|not completed|not measured|aborted after >46 minutes elapsed|

Average excludes loading; trailing rate is the Engine's last-second window, with chunk tokens prorated over the window and rounded, not instantaneous kernel timing. The benchmark now retains each lane's tail rate and window in `reps[].prefill_tail_by_lane`; it does not sum unrelated lane windows. The server remains stopped for this investigation. The user stopped the128K run because its duration was unacceptable. Its elapsed time includes startup and is not a completed prefill duration or throughput result. Retain the incomplete log/command; do not rerun128K unchanged. This establishes an unacceptable performance case, not by itself numerical corruption or a hang.

Source inspection identifies a scaling constraint beyond the increasing number of attended keys. The dense FP32 score buffer is capped at384MiB. It forces smaller query panels, each followed by a separate PV launch:

| Visible context | Full panel rows | Panels per2048-token chunk | PV blocks per full panel |
|---:|---:|---:|---:|
|8192|512|4|128|
|16384|256|8|64|
|32768|128|16|32|
|65536|64|32|16|
|131072|32|64|8|

These are source-derived final-context launch shapes, not occupancy counters. On a64-CU R9700, the low block counts support a parallelism/latency-hiding hypothesis; the kernel also retains a serial16-key loop, barriers and120VGPR. This baseline motivated smaller query tiles and split-KV PV with an explicit partial-state merge; the subsequently qualified production fix is recorded above. Shrinking internal attention panels does not change the Linear activation profile.

A fresh32K/G16 selected-region trace confirms the dominant owner: dense PV is63.47% (50.972s), QK13.28% (10.663s), and score maximum2.32% (1.860s) of80.312s prefill/setup kernel service. Reported profiled prefill is80.798s; use the unprofiled table above for speed. The first2048-token chunk has1.196s kernel service/PV0.091s; the final chunk has11.657s/PV8.757s. Its256 PV dispatches (16 panels across16 full-attention layers) each launch32 blocks. This physically confirms the low-parallelism launch shape and PV dominance, not hardware occupancy, memory saturation, cache hit rate or a proven stall cause. Trace and attribution: `profiles/rocprof/r9700-prefill32k-20260925/`.

The complete commands/reports are retained at `profiles/bench/r9700-prefill-context-sweep-20260925/`. Reusable dense/XAttention speed, matched-token PPL, resume and trace commands are in `tools/bench/context_ladder.md`. Keep unprofiled speed and profiled attribution separate; reuse matching completed cells rather than repeating expensive128K controls. This setup does not resume the paused admission campaign. The runner's seven focused tests pass, the four retained speed cells collect without rerunning, and a real256-token prefill PPL smoke has127 finite aligned scores. Resuming that cell launches no inference. Smoke evidence lives in `profiles/ppl/r9700-context-ladder-smoke-20260925/`; it validates tooling, not long-context quality or a new comparison against5090.

## Current prefill precision and retained 5090 quality (2026-09-25)

The selected selective-cap model uses global Q4 A8 with the gate/up-only A4 override atT>128, as confirmed by the current build and owning dispatch. For large Text-prefill matrix calls:

| Route | Main Text matrices | Modeled Text-body matrix MAC share |
|---|---:|---:|
| Q4 weights / integer A4, gate/up layers0–61 |62|45.43%|
| Q4 weights / integer A8 |232|49.14%|
| Protected FP8 weights / FP8 GEMM operands |26|5.43%|

The count excludes96 small BF16 GDN A/B controls and the output head. MAC shares additionally exclude attention, recurrence and other non-matrix work; they are not runtime shares or fractions of every stored activation. Inter-Op hidden values remain BF16, GDN persistent state FP32, and the growing cache is FP8-K/INT4-V/FP16-scale. Dense prefill attention uses BF16-WMMA QK; short-context PV is FP32, while contexts>=12288 use FP16-WMMA PV with FP32 accumulation, denominator and merge (represented values are scaled by1/8 before FP16 conversion, then restored by8). These private attention operands do not alter the Linear mix. Attention is not integer-A4. Ordinary/small-verify Q4 calls remainA8. The smaller long-context attention panels are internal to the attention Op; they do not shrink the2048-token Linear calls or trigger the T<=128 A8 fallback. Body MACs follow the retained cap26 cost inventory (head format excluded), under `profiles/ppl/r9700-endpoint-precision-20260923/cap26-head-gate-up-cost.json`.

Retained delivered AMD recipe versus the frozen standard5090 NVFP4 reference:

| Text | AMD prefill PPL |5090 prefill PPL| Change |AMD decode PPL|5090 decode PPL|
|---|---:|---:|---:|---:|---:|
| WikiText |6.602872|6.481045|+1.88%|6.705888|6.864514|
| Technical |9.655792|9.576387|+0.83%|22.160924|22.304207|
| Code |2.282552|2.253032|+1.31%|3.216515|3.415608|

These are identical4096-token inputs: prefill scores2047 positions after2048 warmup tokens; teacher-forced decode scores128 after3967. No speculation. Prefill and decode columns score different spans and must not be compared as an activation-quality A/B. Integer A4 and NVIDIA NVFP4 also have different codebooks/scales, protections and cache formats. These retained results are neither fresh latest-binary PPL nor proof of8K–128K quality equivalence. The AMD delivery reports and six NLL sidecars remain under `profiles/bench/r9700-compact-mixed-delivery-20260923/`; the unchanged5090 reference is tracked in `tools/ppl/fixtures/nvfp4-5090-20260922/`.

## Post-fix long-context attribution (2026-09-25)

On `fc3d61a7`, the same installed selective-cap companion, C1, code corpus, optimized proposal head, fixed K5, max-context32768/workload, and power`auto`: a fresh unprofiled chunk4096 screen measured1598.94 prefill tok/s atP2048 and818.52 atP15200 (G32, warmup0/repetition1). The preceding chunk2048 P15200/G128 result was868.70. This single screen does not select a new chunk; it agrees with the earlier repeated2048/4096 comparison below. Keep2048. The historical1904.34 P2048 result used all-Q4, not this selected weight recipe. Standalone long-context prefill also agrees with the user's875.7tok/s server request: the observed gap is not evidence of Docker overhead.

A fresh selected-region marker/kernel trace atP15200/G128/chunk2048 separates graph0 prefill/setup from graph1 DFlash decode using recorded graph IDs:

| GPU kernel service owner | Prefill/setup share | Decode-graph share |
|---|---:|---:|
| Dense FP32 PV accumulation |37.23%|—|
| Dense BF16-WMMA QK |13.63%|—|
| Dense score maximum |2.25%|—|
| Batched DFlash vector PV |—|23.05%|
| Small-batch Q4 gate/up |—|21.53%|
| Small-batch Q4 down |—|11.52%|

These are attribution-only fractions, not unprofiled speed or bandwidth claims. Prefill/setup kernel service sums17.432s against17.613s reported prefill time. The first2048-token chunk has1.197s kernel service, versus3.204s for the last full2048-token chunk; dense PV alone grows90.5→1673.5ms across those chunks. The increasing attention extent, not just prompt length divided by a fixed throughput, is material. Interval logger rates are not individual chunk timings.

The strongest next kernel investigation is long-context dense PV: its serial 16-key tiles, shared-memory/barrier work and six-head FP32 accumulation are a concrete issue/synchronization hypothesis. The emitted route has120VGPR, 9208-byte LDS and no scratch; these do not prove occupancy or bandwidth saturation. Test a genuinely new long-context tiling/split-reduction mechanism at the public Op, including merge traffic and oracle error, before implementing a production change. Halving this owner would save about18.6% of profiled prefill GPU service, not establish a measured23% throughput gain. The separately owned verify PV is the next attention target; halving it would save about11.5% of decode graph kernel service. Preserve the prior rejected short-context head-partition, FP32-BLAS and high/low-BF16-WMMA PV findings; do not repeat them unchanged. The measured adaptive65.63 versus fixed-K5 83.75tok/s gap is also worth policy diagnosis, but this trace does not establish why adaptive stayed atK3 or justify forcingK5 for all prompts/concurrencies.

Commands, exact workload and offline graph-ID attribution are retained under `profiles/rocprof/r9700-long-context-followup-20260925/`. No PMC/sudo or power changes were needed. The idle Compose server was stopped for isolated capture, then automatically restarted and verified healthy on8001. No production kernel, sampling policy or chunk setting was changed by this investigation.

## Long-context DFlash attention production fix (2026-09-25)

The pre-fix DFlash selector stopped batched FP8-Q WMMA at 8,192 visible tokens. At and above that boundary, K3 (four verification rows) used split512 while K4/K5 used fused attention. Direct comparisons found no long-context speed crossover in favor of those incumbents at the tested points through the native 262,144-token limit. The following table is operator evidence, not a server throughput claim.

R9700/gfx1201, ROCm 10, device 0 on the measured R9700, power `auto`, G16, token-fastest FP8 K and feature-fastest INT4 V / FP16 scales, BF16 Q, 24 query heads / 4 KV heads / dimension 256. The table reports median direct-launcher attention latency in milliseconds after a 256 MiB eviction pass outside each timed interval (seven individual trials). It does not establish physical cache hit rates or reproduce a complete inference schedule.

| Visible context | K3: split512 → batched WMMA | K5: fused → batched WMMA |
|---:|---:|---:|
| 8,192 | 0.837 → 0.445 | 5.623 → 0.584 |
| 15,200 | 1.477 → 0.780 | 10.388 → 1.023 |
| 32,768 | 3.085 → 1.440 | 22.628 → 2.084 |
| 65,536 | 6.434 → 2.917 | 44.882 → 4.241 |
| 131,072 | 13.060 → 5.809 | 89.280 → 8.723 |
| 262,144 | 27.632 → 13.061 | 182.063 → 19.786 |

K4 also improves at all these points (15,200: 10.290 → 0.903 ms). Tiny contexts 6/16/32 favor fused attention; at 63/64 the advantage becomes small or begins to favor WMMA. Retain the existing lower cutoff rather than claim one universal route. Separate warm, repeated-operand trials support the same long-context direction; their durations differ materially from eviction-pass timings at large extents.

The existing independent represented-BF16 public FP64 oracle and private FP8-Q profile checks pass for all three widths at the listed points. Poisoned-output graph replay and output/workspace canaries pass. The inherited public criterion allows the independently computed FP8-Q quantization delta plus implementation error; it is not a model-quality bound. The unchanged kernel's retained assembly also passes the existing native FP8-WMMA/wave32/no-spill static checker (`tools/r9700/check_attention_parity_static.py tools/r9700/build/kv_op_qual.s`). An additional nonperiodic V fixture with nonzero head/feature-dependent means passes the same oracle and poisoned-replay checks at 15,200 and 262,144 tokens for all three widths. Public-oracle maximum absolute differences are at most 0.000863 and 0.000251 respectively, including the private FP8-Q quantization delta.

For the observed 15.2K adaptive request, inferred average K was approximately 3, not 5. K3's measured attention savings across 16 full-attention layers amount to about 11–12 ms; relative to its inferred 72 ms round, that suggests roughly 18–21% higher throughput **only if other costs and acceptance stay unchanged**. The much larger K4/K5 operator improvement cannot be applied to that K3 request. The saved approximately 105 tok/s benchmark used 4K, fixed K5, greedy sampling, the full proposal head and a code continuation; it is not a matched control for 15K chat with temperature 1.5, adaptive length and the optimized proposal head.

Production integration extends the G16 DFlash W4..6 selector and score-workspace envelope through262144 and gives it precedence over ordinary/MTP split512. W6's paired PV at4096..8191 has a separate graph executable topology from vector PV, with boundary profiles and corresponding startup allowances. Other formats, tree/device-count forms and ordinary/MTP calls retain their existing routes. The public leaf qualifier passes at15200/32768 for all three widths, including fragmented pages, compact table rows0..3, pending publication, independent FP64 public/profile checks, undersized scratch rejection, guards and poisoned graph replay. Public maximum absolute error is below0.000085 on these fixtures. Commands, fixtures and raw results are local under `profiles/bench/r9700-long-context-wmma-20260925/`.

Matched whole-Engine checks use the installed selective-cap N16/K16 DFlash2-Q4 artifact in `local_llm/models/qwen3.8-27b-r9700-q4-fp8-selective-cap/`, G16, C1, P15200/G128, chunk2048, capacity32768/workload, optimized proposal head, Device Graph, greedy decoding, cold requests (warmup0, one repetition), auto. The4096-token `r9700-compact-mixed-delivery-20260923/code.ids` corpus is cycled by the benchmark. These are not matched temperature1.5 chat measurements.

| Mode | Before decode tok/s | After decode tok/s |
|---|---:|---:|
| Ordinary |25.74|unchanged route|
| Fixed K3 |56.15|69.10|
| Fixed K4 |not measured|79.86|
| Fixed K5 |24.29|83.75|
| Adaptive K3..5 |56.06|65.63|

All four speculative modes exactly match ordinary generated token IDs. Fixed K5 acceptance changes from85.12%/5.12 tokens per round to80.95%/4.92 tokens per round; its improvement comes despite lower acceptance, from cheaper verification. Decode-phase time divided by rounds falls from210.8 to58.8 ms for K5 and from65.1 to52.9 ms for K3 (phase-level averages, not isolated kernel timings). Adaptive selected K3 throughout this short sample, so the fixed-K5 result must not be advertised as adaptive throughput. This change does not alter the adaptive policy. Production integration commands and reports are under `profiles/bench/r9700-long-context-production-20260925/`. Additional exact-token checks pass across both W6 graph boundaries (P4080/G64 and P8180/G64), in eager K5 atP15200/G128, and for fixed K5 and adaptive atC2/3/4 with P9000/G32 per lane. The full runtime planner and focused route discriminators also pass. These short concurrent checks establish correctness, not a steady-state concurrency throughput ranking. A second15,200-token `technical.ids` continuation (G64) also exactly matches ordinary token IDs with fixed K5:46.17 tok/s versus ordinary25.63. Its lower throughput than the code fixture reinforces that the83.75 result is workload dependent, not a universal server speed promise.

Deployment: `bash scripts/hot-patch.sh --image-only` rebuilt the three apps incrementally with 8 jobs; `docker compose up -d --no-build --wait server` started the healthy updated image on host port8001. The installed server binary matches the exported build. Chat/Responses/Anthropic/SSE smoke requests pass, including observed disk and RAM restoration. Preserve-thinking is enabled. A fresh15424-token synthetic storage-protocol chat with128 generated tokens, temperature1.5 and the unchanged adaptive/optimized-head defaults measured 866.8 prefill and51.5 decode tok/s,2.76 tokens/round,59.1% acceptance. This is an actual deployed-server check, not a before/after comparison to the user's different prompt. Logs and responses are in the production evidence directory. Whole-Engine and server validation cover the deployed32768-token capacity; the upper-context arithmetic/timing evidence through262144 is operator-level.

## Gate/up reuse, paired-feature PV and exact-tree GDN (2026-09-24)

Same installed cap26 Q4-head/gate-up-A4 weights and Q4 DFlash/BF16 codebooks, dense G16, auto, P4096/G128, chunk2048, warmup1/repetitions3. Three bounded candidates are retained after numerical and physical checks:

- Gate/up N34816/K5120 T18/20/24 shares weight payloads across two M tiles; cold complete-Linear latency falls31.4/30.1/28.3%. T15 control and all changed widths pass canonical original-input/represented-A8 FP64, exact codec/generic/ graph, poison/stale/zero/scaled fixtures and guards;12 corruptions are rejected. Native IU4,108/103/100VGPR, no LDS/private scratch.
- Paired-feature PV is selected only for W6/G16 feature-fast values/scales at 4096<=context<8192. Complete-attention cold savings are9–12% at the tested selected boundary/interior points. Broad W4/W5 pairing was slower and rejected. Boundary controls2048/4095/8192 retain the old route. Public/profile FP64, serial/graph and invalid-page guards pass; paired PV has118VGPR, no spills.
- GDN sequential record retains the original normalization tree using first-wave shuffles and explicitly materialized squares. Final ISA has separate square/add operations,60VGPR/1024LDS/no spills and20→3 emitted barrier signal/wait pairs. Warm Op savings are9–15%; cold C4 is mixed, not a universal latency win. Isolated whole comparisons resolve this: GDN alone raises C1K5 104.79→105.15tok/s and C4K4 202.73→203.51 aggregate, with separated three-repetition ranges. Retain this route only at actual compact batch1/4: fresh C2/C3 K4 tests showed a small regression/no benefit, so batch2/3 keeps the incumbent. All12 batch1..4/W4..6 oracle/record/graph cells and W2/W16 boundaries pass; both selected instruction/resource streams match their qualified controls.

| DFlash mode | Aggregate tok/s | Per-request tok/s |
|---|---:|---:|
| C1 K5 | 105.15 | 105.15 |
| C2 K4 | 146.25 | 73.12 |
| C2 K5 | 149.40 | 74.70 |
| C3 K4 | 191.05 | 63.68 |
| C3 K5 | 178.29 | 59.43 |
| C4 K4 | 203.26 | 50.82 |
| C4 K5 | 186.88 | 46.72 |
| C4 adaptive maxK5 | 201.04 | 50.26 |

C3K5 improves157.52→178.29 against a fresh whole control. C4K4/K5 improve from the previous pass's181.89/166.90; these latter baselines are retained, not fresh same-pass controls. K5 is the measured C2 preference; K4 remains preferable at C3/C4 on this workload. Adaptive is not a guaranteed winner. C1 retains its unchanged qualified instruction-body measurement; all concurrent modes were remeasured after batch-specific GDN selection. C2/C3 K4 are effectively unchanged against fresh145.92/191.02 controls, not claimed as material gains. All24 final benchmark repetitions and44 cold K3/adaptive graph/eager cases match ordinary tokens. GDN public FP64/record/snapshot checks and actual replay-fold ordinary-state-exact qualification pass. Four focused host and two static tests pass; CLI/server/PPL/bench are rebuilt. Weights/precision are unchanged. Ordinary decode and prefill were not remeasured; no absolute performance ceiling is claimed. Evidence/reproduction: `profiles/bench/r9700-remaining-candidates-20260924/`.

## Concurrent projections and paired QK (2026-09-23)

Follow-up investigation (2026-09-24, no production change): matched steady-graph attribution at C3K4/C4K4 shows gate/up180.50→324.83us/call while down stays 93.76→94.10us. This prioritizes T20 cross-tile weight reuse, not blanket projection replacement. Distinct secondary hypotheses are adjacent-feature INT4 PV reuse and exact-tree GDN normalization with fewer block barriers. No candidate has been timed or promoted in this investigation; C3/C4 acceptance also differs. Details: `profiles/bench/r9700-concurrent-overhead-20260923/remaining-investigation-20260924.md`.

Same cap26 Q4-head/gate-up-A4 weights, Q4 DFlash/BF16 codebooks, fixed G16 cache, dense P4096/G128, chunk2048, auto, warmup1/repetitions3. Concurrent projection coverage now includes45 new T10/12/15/18/20/24 cells across eight shapes. The independent original-input/represented-A8 FP64 criteria and exact generic/graph checks pass all48 affected cells;144 corruptions are rejected. Prior44 kernel instruction/resource streams remain exact. New nativeIU4 kernels use70–75VGPR, 22SGPR, no LDS/private scratch; complete-Op reductions range3.66–51.66%.

DFlash W4..6 QK now packs two query rows into separate eight-M-lane groups, masking unused heads and the odd W5 tail. FP8 operands and each dot's reduction order are unchanged. QK retains24VGPR/32SGPR and no LDS/private scratch; softmax/PV instruction streams remain exact. Complete attention at4K improves 8–17%; short-context latency is effectively unchanged. Public/profile FP64, serial/graph bit parity, guards and invalid-page metadata checks pass.

| DFlash mode | Fresh control aggregate tok/s | Final aggregate tok/s |
|---|---:|---:|
| C1 K5 | 101.26 | 102.73 |
| C2 K4 | 122.58 | 146.20 |
| C3 K4 | 160.20 | 190.96 |
| C4 K4 | 162.21 | 181.89 |
| C4 K5 | 154.05 | 166.90 |

C4 adaptive maxK5 reaches180.21 aggregate tok/s (45.05/request); no fresh adaptive pre-change control was collected in this pass. Fixed K4 remains the concurrent preference, and C3 has the highest aggregate rate on this workload. All18 final benchmark repetitions match same-concurrency ordinary tokens exactly.

C1's fresh control is the projection-only build (its C1 device bodies are unchanged); concurrent controls precede both optimizations. The projection-only C2/C3/C4 K4 results are144.27/186.82/178.75, so the attention contribution is measured separately. All44 cold K3/adaptive graph/eager Engine transition cases match ordinary tokens. Fixed-draft acceptance/round counts are checked separately from final-token parity. Six pending rows at K4 remain unobserved; append/storage are unchanged and their maximum-capacity regression passes.

The C4 runtime audit found less than0.9% uncovered inter-round time. Large intra-graph trace gaps are instrumentation-confounded, not proven hardware idle; the requested copy table was empty. Shared FP8 preparation is only0.261ms/round; FP8 matmul is4.45% of steady kernel service with no identified algorithm defect. Neither a host-copy rewrite nor a custom FP8 rewrite is justified by this capture. These are bounded decisions, not absolute-ceiling claims. Prefill was not tuned or remeasured; precision, weights and chunk2048 are unchanged. The semantic Linear dispatch can also serve a matching short prefill tail, with the same exact output. Evidence/reproduction: `profiles/bench/r9700-concurrent-overhead-20260923/`.

## Remaining projections and adaptive correctness (2026-09-23)

The selected cap26 Q4-head/gate-up-A4 model and Q4 DFlash companion are unchanged. Eight T5/T6 Linear cells now use the existing successor pipeline: N7168/K5120, N6144/K5120, N1280/K5120 and N5120/K4096. Independent original-input and represented-A8 FP64 bounds, exact generic/codec/graph checks and guards pass; the prior36 instruction/resource streams are unchanged. Cold complete-Op savings are17–50%; final whole C1 K5 rises98.0082→101.5431tok/s, all repetitions exact ordinary tokens. These are separate operator and whole-inference claims.

Final unprofiled P4096/G128, chunk2048, auto, one warmup/three repetitions:

| Decode mode | Aggregate tok/s | Per-request tok/s |
|---|---:|---:|
| C1 fixed K5 | 101.54 | 101.54 |
| C1 fixed K4 | 87.64 | 87.64 |
| C4 fixed K4 | 162.22 | 40.56 |
| C4 fixed K5 | 153.98 | 38.50 |
| C4 adaptive, maximum K5 | 160.90 | 40.22 |

All15 repetitions match same-C ordinary tokens. Fresh control C4 fixedK4 is 161.84: that fixed route is essentially unchanged. Prior-pass C1 K4/C4 K5 were 84.71/149.08; these are historical, not fresh paired controls. Ordinary decode was not changed or remeasured in this pass (previous C1 approximately30.16tok/s).

Adaptive DFlash distinguishes physical captured K from logical output allowance. Additional padded candidates require measured same-C cost and enough physical context. Logical ingress/publication masks and the existing minimum-width masked fallback remain; K3 is not banned. Selection occurs at the actual compact round boundary. Fresh C4 adaptive rises156.6669→160.8980 aggregate tok/s (40.2245 per request), mostly selecting K4 on this workload.

Expanded cold-start tests exposed and fixed a preexisting K3/W4 attention seam: W4 used BF16-Q fused arithmetic where ordinary W1 and W5/W6 used FP8-Q WMMA. Warmup could train adaptive selection away from K3 and conceal the mismatch. W4 now shares the same batched attention profile at context64..8191, with proper workspace/topology planning. The public BF16-Q FP64 oracle is unchanged; the existing explicit FP8-Q quantization/profile bound and separate arithmetic check pass at short/4K contexts, C1–4 metadata, graph replay and poisoned rows. All three renamed device instruction/resource streams are unchanged.

Forty-four real Engine cases now match ordinary tokens exactly: cold fixedK3, adaptive graph with two prompt-offset sets, and adaptive eager; cases cover short C1 output limits, near-context C4, and long unequal request budgets. The trace observes9 padded and4 target-only rows. A six-pending-row K5→K4 transition was not observed; append/storage mathematics are unchanged and the maximum-append workspace regression passes. Existing routes outside the admitted W4 domain, weights, MLP activation policy, fixed cache and prefill/PPL routes are unchanged.

Gate/up remains~25% of C1 traced kernel service, at~558GB/s useful-weight rate versus the636GB/s stream control. No new scheduler/issue mechanism justified another challenger; prior depth2 failures remain excluded. This is not proof of an absolute hardware ceiling. Prefill remains paused; chunk2048 is retained. Evidence: `profiles/bench/r9700-dflash-projections-tail-20260923/`.

## Compact prefill and concurrent decode optimization (2026-09-23)

Same selected cap26 Q4-head/gate-up-A4 weights and precision as the delivery below. The existing SiLU→BF16-round→A8-prepare→Q4-down fusion now admits BF16 gate/up output independently of its producer format, retaining exact T2048/down-A8 guards. All six mixed-profile NLL sidecars remain byte-exact. C1 codeP4096/G128 ordinary prefill1494.30→1519.18tok/s; K5 prefill1450.71→1472.19tok/s. Ordinary decode stays30.15tok/s; this is a materialization/launch saving, not lower precision.

Ordinary T2–4 Q4 MLP gate/up, down and N5120/K6144 output use successor loading through the semantic Linear owner. Independent FP64/public-input, exact codec, graph/poison/guard and exact generic-output checks pass; native IU4 WMMA,59–61VGPR, no LDS/private scratch. Cold complete-Op A/B improves gate/up10–14%, down39–43%, output22–29%. Concurrent K5 T12/18/24 MLP uses a16-row tiled pipeline: gate/up7–18% and down41–50% lower complete-Op latency. Full public BF16-input FP64 checks pass for all14 selected MLP cells, two dedicated down cells and five output cells; exact generic/eager/graph outputs and codec/poison/guards pass. Tiled kernels use70VGPR,22SGPR, native IU4, no LDS/private scratch; retained small-width instruction streams are identical. Whole medians, auto/G16/dense/chunk2048, warm1/reps3:

| C | Ordinary before, aggregate tok/s | Ordinary after, aggregate tok/s | Ordinary per request | K5 after, aggregate tok/s | K5 per request |
|---|---:|---:|---:|---:|---:|
| 1 | 30.15 | 30.15 | 30.15 | 96.28 | 96.28 |
| 2 | 38.15 | 49.98 | 24.99 | 119.66 | 59.83 |
| 3 | 53.90 | 71.04 | 23.68 | 133.66 | 44.55 |
| 4 | 65.79 | 86.03 | 21.51 | 140.39 | 35.10 |

All18 final C2–4 repetitions exactly match prior same-C greedy tokens; K5 matches ordinary at each C. Lanes use corpus offsets, not identical prompts. Per-request rates are aggregate throughput divided by C, not individual request latency. The tiled K5 step improves prior101.56/120.11/125.35 aggregate tok/s by17.8/11.3/12.0%. C1 rates reuse the unchanged fusion checkpoint; later changes select only C>1 extents and preserve C1 instructions. Final C2–4 ordinary prefill1506–1520tok/s, K5 prefill1467–1470tok/s; no new prefill arithmetic or precision change. Candidate safety limits are nonbinding with no CPU throttling. CLI/server/PPL/bench are rebuilt. This bounded pass does not establish a physical throughput ceiling. Evidence: `profiles/rocprof/r9700-compact-mixed-speed-20260923/`. Its `final-verified-summary.json` retains rates and exact prior-token checks; `baseline-corrected.json` supersedes the explicitly marked first helper summaries.

Numerical limitation found during the subsequent larger-batch experiment: the full-output public BF16-input oracle (rather than the prior sampled-row norm) rejects the unchanged generic N4096/K5120/T6 control at 2.0341% relative RMS against the 2% criterion. This is not a measured model-quality regression, but the historical complete small-batch qualification failed; do not infer universal A8 accuracy from the passing model sidecars or narrower MLP checks. The failing test and receipt are retained. The subsequent principled implementation-profile qualification below resolves the unexplained operator cutoff, not model quality.

### Four-token concurrent DFlash follow-up

The same successor pipeline now covers K4 MLP widths T10/15/20, corresponding to C2/3/4. Six new cells pass the unchanged full-output public BF16-input FP64 gate (maximum relative RMS1.96781%), represented-A8 oracle, exact codec/generic/graph output, poison and guards. All23 prior kernel instruction streams are unchanged; new cells use native IU4,70VGPR/22SGPR and no LDS/private scratch. Cold complete-Op gate/up latency falls7–18%, down44–50%. Same codeP4096/G128/chunk2048, auto, warm1/reps3: K4 aggregate decode98.32/132.17/142.61→122.14/160.87/161.66tok/s at C2/3/4 (per-request61.07/53.62/40.41). All nine selected repetitions exactly match prior same-C ordinary greedy tokens. K4 now beats the retained K5 rates on this concurrent sample; C1 routing is unchanged and its sample still favors K5. This is workload-specific selection, not a universal best draft length. Evidence: `profiles/bench/r9700-compact-followup-20260923/`.

The inherited N4096/T6 failure was localized separately: exact represented-A8 FP64 output itself differs2.01625% from the original BF16-input oracle, versus 2.03407% GPU error. Input quantization error is0.26831%; cancellation amplifies it at the output. Exact codec/generic checks pass. Arithmetic scheduling alone cannot repair the2% failure while retaining those codes/scales. One lower-input-MSE scale refinement worsened output error to2.18842% and was rejected. No numerical codec was changed. The subsequent numerical review keeps this result explicitly.

The v3 small-batch/down qualification retains the original full-output BF16-input FP64 oracle and adds an analytically derived A8-profile envelope: independently measured quantization error plus exact-integer/group-FP32-FMA/BF16-cast error. A separate per-output arithmetic bound prevents quantization allowance from concealing kernel defects; both bounds also apply per-token in L2. The derivation is in `tools/r9700/README.md`. All32 small-batch and two down cells pass, including exact codec/generic/eager/graph, poison recovery and guards; all102 deliberately corrupted outputs fail. Maximum public/arithmetic norm-budget fractions are 0.88150/0.45584. The historical2.03407% RMS remains recorded and still fails its old diagnostic screen. This changes no production arithmetic, weights, precision, PPL threshold or previously measured model quality. Evidence: `profiles/bench/r9700-a8-bound-dflash-20260923/`.

Matched chunk check on the frozen preceding build: C1 codeP4096 prefill 1473.39/1523.68/1459.01tok/s at chunks1024/2048/4096; WikiTextP8192 gives 1241.86/1197.63 at2048/4096. Each uses warm1/reps3, same weights, auto and G16. The2048/4096 runs generate identical tokens on both samples. Keep2048;4096 is 3.6–4.2% slower here, not a theoretically preferred size. No model precision changed.

A bounded high/low-BF16 WMMA PV prototype passed two independent complete-attention oracle cases and exhaustively reconstructed finite INT4×FP16 values, but complete attention regressed10.09→11.01ms for P2048/context2048 and42.89→43.98ms for P2048/context4096. Despite fewer arithmetic instructions and zero scratch spills, the added decomposition/data/synchronization cost defeated the mechanism. Rejected; no candidate remains in production and no model-quality claim follows.

Concurrent adaptive K5 initially failed before inference because smaller captured FP8 verification widths were absent from load-time preparation. Load and Program binding now share the full captured-width inventory, without changing workspace maxima or adding preparation during capture. Host contracts pass; repaired C4 adaptive measures154.82 aggregate tok/s and all three repetitions exactly match ordinary tokens. FixedK4 remains faster on this sample. The first adaptive repetition records130 rounds,517 drafted tokens and381 accepted (73.69%).

The remaining N5120/K6144 K5 output projections now use the same tiled successor pipeline at T12/18/24. All eight output cells pass unchanged full-output numerical, codec, graph/poison and guard criteria; all29 previous instruction streams are unchanged. New cells use70VGPR/22SGPR, nativeIU4, no LDS/private scratch. Matched cold complete-Op latency falls26–36%. Fresh C4 whole comparison improves fixedK5 140.488→144.777 aggregate tok/s (+3.05%,36.19 per request), with all six control/ candidate repetitions exact ordinary tokens and identical121 verification rounds. Candidate repetitions144.660–144.786 versus control140.172–140.533tok/s. The repaired-adaptive154.82 measurement precedes this last output extension; its result is not a fresh final-build adaptive timing. FixedK4's measured161.66 still leads; K4/C1 routes retain identical instructions. Final delivered binaries were rebuilt and host contracts passed. The subsequent A8 bound qualification above resolves the operator-criterion question without changing model quality.

### Draft projection and adaptive-tail follow-up

The compact C1 draft still used generic Linear for N5120/K17408 down and N5120/K25600 feature projection at T5/6. Transferring the existing successor pipeline to these four shape-owned public routes passes complete independent FP64/A8-profile qualification, exact generic/codec/eager/graph, poison and guards. The prior32 kernel instruction/resource streams remain exact. New kernels use nativeIU4,62/63VGPR,22SGPR and no LDS/private scratch. Cold complete-Op latency falls44.6–47.4%. Matched C1 codeP4096/G128 K5 improves95.9727→97.9506tok/s (2.06%); all three repetitions on each build match ordinary tokens and use25 request-lane speculative rounds. K4 improves83.1713→84.5524tok/s (1.66%), again exact ordinary tokens in all six repetitions and30 rounds each. Same weights, precision, auto, G16, chunk2048, warm1/reps3; no PPL arithmetic changes.

Fresh fixedK4/adaptive C4 baseline measures161.7119/156.5149 aggregate tok/s (40.4280/39.1287 per request), with exact ordinary tokens. Adaptive actual K3/K4/K5 request-lane histograms are3/123/4,2/129/1,2/129/1; these are not graph launch counts. Matched traces show31 steady C4/K4 rounds differ only0.41% in kernel service. Most of the difference is at the tail: adaptive uses slow K3/W4 routes as requests drain, while fixedK4 keeps masked/padded W5 execution. The current budget-clamp policy is intentional and tested; no numerical or steady-state chooser bug is demonstrated. Five redundant compact copies cost about8us/round, too small to explain the gap. Keep fixedK4 for this measured concurrent workload. Do not banK3 or change adaptive semantics from this one sample; physical-width/logical-budget separation is a future policy experiment. Evidence: `profiles/bench/r9700-a8-bound-dflash-20260923/`.

After removing the duplicate verify-only Linear API/kernel and rebuilding all deliverables, the final public four-cell qualification passes again. Its linked projection code object matches the qualified candidate exactly, and the previous target-down instruction/resource streams match the consolidated route. Final matched confirmation (same workload, median of three repetitions):

| Mode | Before, aggregate tok/s | Final, aggregate tok/s | Final per request |
|---|---:|---:|---:|
| C1 K5 |95.97|98.03|98.03|
| C1 K4 |83.17|84.71|84.71|
| C4 K5 |144.82|149.08|37.27|
| C4 K4 |161.71|161.74|40.43|
| C4 adaptive maxK5 |156.51|156.28|39.07|

C1 K5/K4 gains are2.15%/1.85%, C4 K5 gains2.94%; fixedC4 K4 is unchanged. Adaptive has no demonstrated gain (repetition ranges overlap); its tail limitation remains. All15 final repetitions exactly match same-C ordinary tokens. Final scope limits are nonbinding with zero recorded throttling. Focused execution-state capacity and benchmark host tests, plus two Python matrix tests, pass. Ordinary T1–4 decode and large-prefill routes are unchanged; no new speed or ceiling claim is made for them. `final-summary.json` binds the final confirmation.

## Selected compact mixed-profile delivery (2026-09-23)

The user selected the compact cap26 Q4 embedding/head model with gate/up-only A4 large prefill. Fresh builds now default to `NINFER_R9700_Q4_PREFILL_A4_FAMILIES=1`, global Q4 bits8, W8 bits8; delivered `build-r9700` was explicitly reconfigured and CLI/server/PPL/bench rebuilt. T>128 full-K N34816/K5120 Q4 uses the already-qualified fast A4 route; other Q4 calls, including ordinary decode and small DFlash verify, stay A8. No new kernel arithmetic or weight conversion was needed. Broader A4 profiles remain evaluators, not defaults.

Both artifacts and their creation receipts remain installed in `/path/to/models/qwen3.8-27b-r9700-q4-fp8-selective-cap/`: base15,793,065,984 bytes; canonical-Q4/BF16-codebook DFlash companion17,002,543,616 bytes. The adjacent README logs exact creation/build/run commands. These evaluation identities are not renamed, and the independent BF16-source production-admission status is unchanged.

All six final-build PPL NLL sidecars are byte-identical to the retained mixed profile: worst per-text prefill regression+1.880% versus NVFP4, decode-0.642%. The10-versus6 newly severe technical-prefill tradeoff remains; this is not a universal quality-equivalence claim.

Delivered companion, codeP4096/G128, C1, chunk2048, context4240, dense/G16, power`auto`, Device Graphs, one warmup/three measured repetitions (median rates):

| Mode | Prefill tok/s | Output decode tok/s |
|---|---:|---:|
| Ordinary |1494.30|30.15|
| DFlash K4 |1448.40|83.20|
| DFlash K5 |1450.71|96.05|
| DFlash adaptive, maxK5 |1459.47|92.53|

Every repetition in all modes produces exactly the same128 greedy tokens as ordinary decode. K5 was fastest on this sample; the prior short-chat uniform-A8 result favoredK4. Do not treat either fixed K or this small adaptive comparison as a universal optimum. The selected mixed profile changes prefilled state, so old uniform-A8 acceptance/speed numbers are not guarantees. The memory safety scopes recorded no high/max/OOM events and no CPU throttling; benchmark limits were nonbinding. All heavyweight work ran serially, with four build jobs (maximum14).

Checks: selected dispatch thresholds/padding/exclusions, FP8 execution-state contract, six exact quality sidecars, real companion ordinary/K4/K5/adaptive execution and repeated-token parity. Independent review passed. Uniform-control preparation now rejects the mixed-default build before snapshotting; configure family0 and complete the build to reproduce uniform controls. Evidence and replay script: `profiles/bench/r9700-compact-mixed-delivery-20260923/`. This delivers the selected recipe; it does not reopen paused XAttention or broad kernel sweeps.

## Endpoint precision and mixed-activation search (2026-09-23)

Completed 27 configurations / 162 finite, aligned scoring cells on the R9700, using the unchanged three-text 5090 reference and the same six scoring spans below. This is measured quality plus theoretical byte/work accounting: **no new speed measurements or promotion**. Six exact-copy artifacts replace embedding, output head, or both with W8 on the 26-protection and smaller 22-protection FP8 bases. The 26-protection variants were crossed with all six activation profiles; smaller-base A8 controls led to focused gate/up and gate/up+attention tests with Q4 or W8 head. Two gate/up W8-head controls used BF16 rather than A8 head activations.

Key comparisons (decimal GB, base-only artifacts without DFlash; worst per-text PPL change versus NVFP4, not a comparison between the different prefill/decode scoring spans):

| Base / endpoints / activation | File GB | Worst prefill | Worst decode | Logical weight stream GB/step |
|---|---:|---:|---:|---:|
| Retained cap26 / Q4 / A8 default |15.793|+0.126%|+1.814%|14.272|
| Retained cap26 / Q4 / gate-up A4 |15.793|+1.880%|-0.642%|14.272|
| cap26 / W8 head / gate-up A4 |16.468|+1.116%|-0.879%|14.947|
| cap26 / W8 head / gate-up+attention A4 |16.468|+1.670%|+2.090%|14.947|
| cap22 / Q4 / gate-up A4 |15.542|+2.679%|-0.098%|14.021|
| cap22 / W8 head / gate-up A4 |16.218|+2.156%|+0.197%|14.696|

Only cap26 W8-head/gate-up passes the preferred 2% screen among the new weight/profile combinations. Its BF16-head control also passes (+1.184% worst prefill); cap22's remains a near miss (+2.138%). All six decode NLL files are byte-identical between head A8/BF16 builds: ordinary T1 already uses BF16 activations. Batched PPL scores many head rows, unlike ordinary prefill's final logits. This sensitivity test is not evidence for faster A16 execution.

No embedding-only or both-endpoint cap26 profile passes. MLP-wide A4's best endpoint variant still has a 3.490% worst regression; all-projection A4's best is 5.294%. These results do not support extending broad A4 to the less-protected cap22 variants. Near misses remain evidence, not statistical proof of a quality cliff. The 2% screen is not the BF16-source production gate: cap26 W8-head/gate-up has new severe counts 5/10/5 prefill and 1/1/0 decode (Wiki/technical/code), versus retained Q4-endpoint/gate-up's 5/10/3 and 1/2/0. Average PPL does not erase that tradeoff.

Theoretical ranking and next optimization priorities, not additional authorized experiments:

1. Keep cap26 Q4 endpoints + gate/up A4 as the compact speed-oriented shortlist leader. Its modeled Q4 integer-product work is 75.98% of same-weight A8; gate/up+attention is 74.86%, MLP-wide 63.97%, and all eligible projections 50%, but the broader routes fail this screen. Optimize the qualified gate/up route before broadening A4 coverage.
2. Retain cap26 W8-head/gate-up as a PPL-oriented alternative: it improves all six aggregate PPLs over Q4-endpoint/gate-up, but costs 675,430,400 extra file/head bytes (644.14 MiB) and 4.73% more logical ordinary weight-stream bytes. It has the same body-work proxy, not a demonstrated speed advantage. Any future performance work must include the tiled W8 head's bulk A8 and small-T BF16 routes, and qualify DFlash verify/acceptance separately.
3. Retain cap22 W8-head/gate-up as a near-miss capacity/quality option, not a promoted winner. Revisit with broader held-out quality evidence if accepting its roughly 2.15% worst span would change the decision; do not rerun unchanged cells or relax the screen implicitly.

Embedding alone adds the same 644.14 MiB but just 2,720 extra logical bytes per token; it did not buy the hoped-for quality headroom. Stream accounting includes all Text layer weights, the complete head, final norm, and one embedding row, excluding KV, activations, cache effects, rereads and launch/reduction costs. Q4 product work excludes FP8 and head arithmetic. It cannot predict tok/s or establish a global optimum. At this search's closure the default was unchanged; the subsequent user-selected delivery above promotes only gate/up A4, not the W8-head alternative.

Evidence/commands: `profiles/ppl/r9700-endpoint-precision-20260923/`, `comparison.md/json`, and `tools/ppl/endpoint_precision.py`. Original compact controls remain in `profiles/ppl/r9700-fp8-capped-selection-20260923/`; immutable NVIDIA fixtures are unchanged. Verification includes exact payload readback, all six real-artifact bindings, five focused converter tests, the FP8 execution-state contract, and per-cell profile/identity/NLL checks. After a shared-host memory/I/O stall, resumed conversion/build/GPU phases were strictly serial; builds used four jobs, with 24 GiB job memory and zero job-swap limits. These constrained runs must not be used as speed evidence. The larger NVIDIA +328 MiB artifact remains unmeasured.

## NVIDIA-aligned precision follow-up (2026-09-23)

The bounded follow-up completed42 finite/aligned quality cells: four broader A4 activation profiles on selective-cap, and three smaller weight recipes with uniformA8. Same immutable 5090 reference,4096-token texts, chunk2048, prefill2047 and decode128 scored positions as below. The local admission criterion remains at most2% worse PPL on every text/schedule, with newly severe positions reported separately. No new candidate passed; none was timed or promoted. Default weights/execution and the frozen NVIDIA reference remain unchanged. This does not establish a global precision optimum or make2% a statistical quality cliff.

| Candidate | File GB | Worst prefill PPL regression | Worst decode PPL regression | New severe prefill / decode (Wiki, technical, code) |
|---|---:|---:|---:|---|
| Selective-cap, A4 all eligible projection shapes |15.793|6.418%|5.197%|4,9,2 /1,2,0|
| Selective-cap, A4 MLP gate/up+down |15.793|4.180%|5.256%|4,9,2 /1,2,0|
| Selective-cap, A4 attention input only |15.793|0.739%|3.719%|5,3,7 /1,2,0|
| Selective-cap, A4 gate/up+attention input |15.793|2.300%|2.419%|5,6,5 /1,4,0|
| Default-NVIDIA15 protection locations, uniformA8 |15.424|0.526%|5.911%|3,5,3 /1,3,0|
| Three output protections only, uniformA8 |15.217|0.833%|4.869%|4,6,5 /1,2,0|
| Selective-cap without late MLP protections, uniformA8 |15.542|0.336%|2.232%|5,4,5 /1,4,0|

All A4 changes apply only to full-K public Q4 Linear at T>128; decode/verify remainsA8. Decode quality nevertheless changes through prefilled KV/recurrent state. This is why prefill PPL alone cannot select the profile. Attention input is N7168/K5120, excluding GDN inputs. The existing gate/up-only mixed profile remains within2% with its previously measured prefill win, but retains its severe-position tradeoff; uniformA8 remains default.

Selective-cap already protects the same large Text projection locations as NVIDIA's+328MiB artifact:15 split counterparts of the default's9 BF16 projections, plus11 split counterparts of the larger variant's8 added FP8 projections. AMD stores all26 at FP8 and keeps endpointsQ4; NVIDIA retains the original BF16 protections and W8 endpoints. Identical protection locations do not imply equivalent integer-A4 and NVFP4 activation errors.

Scale payloads, excluding alignment/runtime KV/activation images:

| Artifact | Group scale MiB | Row scale MiB | BF16 DFlash codebooks MiB |
|---|---:|---:|---:|
| AMD selective-cap base |807.747 FP16|0.875 FP32|none|
| AMD selective-cap+Q4 DFlash |861.301 FP16|0.875 FP32|242.5|
| NVIDIA default+NVFP4 DFlash |1525.234 FP8 +211.895 FP16|none|242.5|
| NVIDIA+328MiB |1478.359 FP8 +211.895 FP16|0.244 BF16|242.5|

NVIDIA additionally has about2KiB of FP32 divisors. Codebooks are not scales; the larger artifact's344,311,808 extra bytes are overwhelmingly increased weight precision, not BF16 row scales. Its exact matched PPL remains unmeasured here because no NVIDIA GPU is visible.

Verification: exact converted payload readback; real-artifact binding/PPL for all three new recipes; host selector0..5 and FP8 workspace contracts; public-input FP64/independent codec, finite/poison and arena checks for all six A4 projection shapes atT2048 and threshold cases. The explicitlyA8 fused-down route now respects the selected down activation width. No kernel arithmetic was rewritten. Evidence/commands: `profiles/ppl/r9700-nv-aligned-precision-20260923/`; runner `tools/ppl/select_nv_aligned.py`. Rejected candidates and sidecars are retained.

## FP8-capped compact base selection (2026-09-23)

### Subsequent decode optimization checkpoint

On the new17.0025GB selective-cap canonical-Q4 DFlash companion, matched C1 short-chat P89/G128/chunk2048, maxctx1024, auto, Device Graph, warm1/reps3 measured **30.7514 ordinary, 73.7744 K4, and69.1459 K5 output tok/s**. All three repetitions in both speculative modes match the same-artifact ordinary generated tokens exactly. K4 has37 rounds/90 accepted/one fallback; K5 has39 rounds/88 accepted/one fallback. These rates exceed the requested30/60 targets for this workload, not every context or concurrency. Evidence: `profiles/bench/r9700-compact-decode-20260923/dflash-loadstage4-chat/`.

The matched base-only code P4096/G128 baseline20.2346 ordinary tok/s improves to29.7623 with local-Q4 normalization/residual fusion and ordered attention-PV load staging; all repeated output tokens remain exact. Further staging experiments are active, so this is a checkpoint, not a ceiling. Mixed recipes now select the existing fused Q4 operations per local format/shape; explicit ordinary-decode intent excludes speculativeT1 rather than testing unrelated matrix inventories. PV retains its exact serial FP32 FMA order and decoded INT4/FP16 values; page lookup is hoisted and independent reads staged before accumulation. Full represented-input FP64 plus supplementary exact serial checks pass at4096 and64–67, with final32-load staging checked across all16layouts at65; context3 fails an unrelated FP8-query attention comparison afterPVpasses and is retained without relaxing that criterion.

The matched **companion** at P4096/G128/chunk2048/context4240 subsequently measures 28.6544 ordinary,68.8281 K4 and83.2839 K5 outputtok/s, warm1/reps3 and exactordinarytokens throughout. K4 has29rounds/99acceptedtokens; K5 has24rounds/104acceptedtokens, nofallbacks. This is not the same resident artifact/capacity as the base-only29.7623 row. Additional graph frontier-update accounting fixes the reproduced75MiB-versus74MiB startup deficit; observed allocation checks remain unchanged. Evidence: `dflash-graph-updates-code/` under the package above. A measured-region graph-only trace attributes27.77% of K5 decode-kernel time to batchedPV. Sharing the ordinary ordered-load-staging body improves complete attention W5/W6 at4101/4102 from0.9007/0.9436ms to0.2818/0.3450ms. Independent completeFP64, all-row serial-bit parity, Device Graph exactness and guard checks pass at short and long context. ISA:111VGPR, 48SGPR, noLDS/scratch, occupancy12. Matched wholecompanion ordinary/K4/K5 now measures 29.8313/89.6352/104.8799 outputtok/s with identical tokens and acceptance in all3repetitions. Evidence: `dflash-attention-{long,short}-staged32/` and `dflash-staged32-code/` in the package. These are workload-specific C1 results, not a proven ceiling or a C2–4 admission.

Mixed-A4 cooperative single-bank gate/up preserves all six prior mixed-precision NLL sidecars byte-for-byte, but wholeprefill1376–1389tok/s remains belowuniformA8. ExactN34816/K5120/T2048 completequantize+matrix timing gives A4 7.4804ms versus A8 5.6684ms. The selected A4 ping/pong consumer reduces this to5.6628ms (paired A8 5.8518ms); wholeprefill wiki/technical/code=1498.1668/1492.7539/1488.6179tok/s. Public-input FP64 at129/2048, exactcodec/represented-code FP64, and all six retained mixed-precision NLL sidecars pass. ISA uses4signedIU4 instructions/group,73VGPR,13056LDS,no scratch,occupancy16. A8's original owning kernel and emitted instruction schedule are preserved. Wholeevidence: `speed-selective-cap-pingpong-original-a8_a8-{wiki,technical,code}/` in the package above. Retain uniformA8 as the default precision choice; the mixedrecipe still adds10versus6 newly severe technical-prefill positions despite passing the same2%NVFP4PPL screen. The fast A4 kernel is selected within the explicitly enabled mixed evaluator, not by a runtime flag.

Further DFlash row-sharing PV passes completeFP64/serial/graph checks and improves W5/W6 Op latency to0.2440/0.2831ms, but wholeK4/K5=89.4565/105.4945tok/s shows no material consistent gain over staged loads. It is rejected and removed; microbenchmark reuse alone is insufficient.

The final base-decode change stages two raw G64 groups in the N5120/K17408 projected-residual operation. Integer dot order, serial FP32 group FMAs, explicit BF16 delta and residual addition remain unchanged; K6144 keeps its incumbent. Full5120-output independent FP64, exact unfused and graph parity, guards and poison checks pass. Three cold weight copies with80MiB scrub measure0.1336815→0.1277205ms. ISA has32nativeIU4dot8 instructions,58VGPR,18SGPR, noLDS/scratch, occupancy16, and both groups' raw loads precede computation.

Balanced C1/P4096/G128/chunk2048/maxctx4224 base-only ABBA whole measurements give controls 29.8817/29.8317 versus selected30.1973/30.1730 outputtok/s; all12 measured runs retain every baseline token. Evidence: `residual-stage2-whole/` and `residual-down-{before,stage2}.json`. Depth4 remains numerically exact but only reaches0.1264000ms, below the0.2ms/token material saving bound over depth2; it is removed. Prior losing CTA and gate/up-prefetch sweeps are not repeated. These results establish the requested C1 targets on the stated workloads, not memory bandwidth saturation, a universal throughput guarantee, or an optimization ceiling.

The first final-linked confirmation measured29.4460tok/s with exact tokens. A follow-up balanced old/final/final/old comparison measured30.1642/30.1673/29.5064/29.6305. Both binaries show run-to-run variation, not a consistent final-build regression: their entire KV/GDN objects and named dot8/residual/GDN-pair/A8-prefill machine code and resource metadata are identical. Report approximately30tok/s at4K, with observed29.45–30.20 range, not a guaranteed30tok/s minimum on every run. All output-token checks still pass. Evidence: `ordinary-final-code/` and `final-linked-comparison/`; no failed or slow result was discarded.

Final-linked shortchat confirmation measures30.7130 ordinary,75.5710 K4 and70.7222 K5 outputtok/s, all three repetitions exactordinarytokens (`final-linked-chat/`). Both C1 targets are satisfied on this delivered build and workload. CLI, serving, PPL and benchmark rebuilds pass; uniformA8 and mixedA4 public dispatch, standaloneA4 regression and selected down Op qualification pass. No further material winner remains in the bounded candidates above.

The companion is produced with `compose_fp8_capped_dflash`, preserving every selected base payload and copying only the donor's66 DFlash objects (32Q4/34BF16). Both selector codebooks stay BF16. Payload readback, real1190-object strict binder, missing/wrong companion rejection, workspace and host routing checks pass. This changes neither the retained base quality selection nor the separate final BF16-source admission requirement. The original recipe-comparison rates below predate these kernel changes and remain historical comparison evidence.

### Recipe and quality selection

Original selection for subsequent optimization: **Q4/FP8 selective-cap, uniform Q4 A8 execution**. The selected local delivery above now uses gate/up-only A4 prefill with these same weights. The15,793,065,984-byte artifact is saved at `/path/to/models/qwen3.8-27b-r9700-q4-fp8-selective-cap/qwen3.8-27b-r9700-q4-fp8-selective-cap-n16k16-eval.ninfer`. It is10.7% smaller than the17.68 GB selective base and26.7% smaller than four-role. It does not contain a DFlash companion and does not replace final BF16-source production admission. Existing model files remain unchanged.

The recipe retains selective's26 protected projection positions, replaces its15 BF16 matrices with row-scaled FP8, and returns embedding/output head to Q4. Its other11 FP8 projections stay FP8. Everything else retains all-Q4 bytes or existing direct norm/control precision. This caps large promoted weight matrices, not norms, cache, GDN state or DFlash selector codebooks. The exact fixed inventory and converter are in `qwen3.8-27b-artifact.md` under maintainer docs.

The immutable5090 reference is Git-tracked in `tools/ppl/fixtures/nvfp4-5090-20260922/` (`0b00efa1`): three exact4096-token inputs and six NLL sidecars/reports. These are the same prefill2047-position and short-decode128-position spans as the comparison below, with identical cache and measurement settings. All24 new recipe quality cells are finite and aligned.

| FP8 promotions over all-Q4/A8 | File GB | Worst prefill PPL change vs NVFP4 | Worst short-decode PPL change | Prefill tok/s | Ordinary decode tok/s |
|---|---:|---:|---:|---:|---:|
| Early6 attention QK/GV pairs | 15.38 | +0.63% | +5.79% | not timed | not timed |
| All attention QK/GV pairs | 15.72 | -0.01% | +4.33% | 1442–1456 | 20.36–20.45 |
| All attention + GDN QK | 16.20 | +0.11% | +4.59% | 1423–1455 | 19.45–19.50 |
| Selective-cap26 projections | 15.79 | +0.13% | +1.81% | 1431–1446 | 20.18–20.20 |

Rates span per-text medians, C1/P4096/G128/chunk2048, ordinary graph, auto, warm1/r3. All nine admitted speed cells have exact repeated tokens. The early-attention candidate exceeds the5% secondary PPL screen and was not timed. Selective-cap alone meets the preferred2% screen in both schedules among the four new recipes. Its PPLs (WikiText/technical/code) are 6.455954/9.389574/2.255864 for prefill and6.680329/22.708779/3.239396 for short decode. Its newly severe positions versus NVFP4 are5/6/6 and1/3/0 respectively: aggregate PPL agreement is not the separate BF16-source severe-position production gate.

Relative to the old tiled selective companion's ordinary mode, selective-cap improves prefill about54% and decode about4%. It is not fastest on every axis: all-Q4/A8 still leads decode (~21.9 tok/s) and four-role leads prefill (~1627 tok/s). The benefit is a compact candidate with tighter measured PPL agreement, not proof of a global Pareto optimum or hardware ceiling.

The bounded mixed evaluator applies A4 only to Q4 MLP gate/up N34816/K5120 at T>128; decode/verify and other Q4 operations stay A8. Its same-artifact PPLs are 6.602872/9.655792/2.282552 prefill and6.705888/22.160924/3.216515 short decode: all within2% of NVFP4. Technical-prefill newly severe positions rise from6 to10, so this is not numerical equivalence. Mixed A4 measures900–901 prefill tok/s and20.18–20.19 decode tok/s with exact repeat tokens in all three speed cells. Uniform A8 remains the selected default; the generic A4 route needs a separately qualified cooperative prefill implementation before it can earn promotion on speed. This identifies an accuracy-tolerable optimization target, not a speed win.

Verification: complete artifact payload readback and strict binder rejection checks for all four recipes; FP8 execution-state and host planning contracts; unchanged all-Q4 NLL control; mixed real-shape public-input FP64 oracle within its A4 error bound plus exact quantized-reference checks at768 tile/shape-boundary outputs, full finite/poison and arena checks; both-build selector/report tests. The installed artifact/current default build reproduces every frozen code-sample NLL exactly. The mixed evaluator corrected an exact-span A4 workspace binding failure before any model evaluation. No kernel arithmetic was rewritten.

Evidence and exact commands: `profiles/ppl/r9700-fp8-capped-selection-20260923/`, including `quality-summary.json`, `comparison.md/json`, `mixed-quality-summary.json`, and `selection.json`. Drivers: `tools/ppl/select_fp8_capped.py`, `tools/convert/qwen3_8_27b_r9700/convert_fp8_capped.py`. The four-recipe integration is committed as `03240dd5`; the mixed evaluator is `0a439704`. Further kernel tuning and DFlash integration are separate follow-ups, not completed claims.

## Matched NVFP4 quality and AMD recipe comparison (2026-09-23)

The bounded comparison uses the user's standard 5090 artifact `/path/to/models/qwen3.8-nvfp4-flash2-nvfp4-bf16codebook-from-bf16/qwen3_8_27b_nvfp4_dflash_nvfp4.ninfer` and a frozen scorer from the frozen NVIDIA checkout at `dfc818ae32c4e0e1ded7b9252b87930338fb8455`. Its source, build, and model bytes were not changed. Three identical4096-token inputs cover WikiText, technical prose, and C++ source. Both embedded tokenizers and native tokenization match. Prefill PPL scores2047 next-token positions after2048 warmup tokens; a separate teacher-forced decode check scores the last128 positions. Compare recipes within each span, not prefill PPL against decode PPL. No speculation participates in either quality schedule.

AMD uses the fixed FP8-K/INT4-V/FP16-scale G16 dense cache; NVIDIA uses its standard NVFP4 cache without Sage/sparse attention. This is an end-to-end product comparison, not isolated activation quantization. A4/A8 refers to Q4 activation width; W8 activation configuration stays8. The A4 evaluator was corrected to exclude fused routes hardcoded to A8 (`3540705b`), and both activation profiles pass the owning N16/K16 public Q4 dispatch oracle. A4 has fewer optimized routes; its timings do not establish the hardware limit of four-bit arithmetic.

AMD speed uses C1, fresh P4096/G128, chunk2048, context4224, ordinary Device Graph execution, auto power, one warmup and three measured repeats per text. Model loading is excluded. Rates below span the three per-text medians, not pooled scorer runtime or short-context decode. The host-only graph profile-update allowance fix (`44013616`) enabled affected startup cells; original failures are retained and explicit retry receipts select corrected runs. Kernel math and successful earlier measurements are unchanged.

| Recipe / Q4 activation | File GB | Worst prefill PPL change vs NVFP4 | Worst short-decode PPL change | Prefill tok/s | Ordinary decode tok/s |
|---|---:|---:|---:|---:|---:|
| All-Q4 / A4 | 15.17 | +12.13% | +22.15% | 674–676 | 15.85–15.87 |
| All-Q4 / A8 | 15.17 | +0.74% | +2.81% | 1437–1440 | 21.90–21.92 |
| Source-MSE mixed Q4/W8 / A4 | 22.88 | +1.29% | +0.70% | 656–657 | 10.62 |
| Source-MSE mixed Q4/W8 / A8 | 22.88 | -0.99% | +0.83% | 951 | 11.15–11.18 |
| Selective / A8, old head | 17.68 | -0.48% | +4.67% | 929–941 | 18.07–18.09 |
| Four-role FP8/Q4 / A8 | 21.55 | -0.77% | +1.57% | 1625–1628 | 15.12 |
| Selective / A8, current tiled head | 18.89 | -0.48% | +4.67% | 928–939 | 19.41–19.42 |

Sizes are decimal file GB, not resident VRAM. Current selective includes its unused DFlash companion; the base portion is approximately17.68 GB. Selective changes28 all-Q4 matrices: two W8 embedding/head, fifteen BF16, eleven FP8. Mixed retains its available older RowSplit W8 head; four-role has a Q4 head. This compares available complete implementations, not intrinsic format performance. The tiled selective head preserves measured PPL exactly and improves ordinary decode about7.4% over its older counterpart.

Prefill PPL, lower is better:

| Recipe / activation | WikiText | Technical | C++ |
|---|---:|---:|---:|
| 5090 NVFP4 reference | 6.48105 | 9.57639 | 2.25303 |
| All-Q4 / A4 | 7.14944 | 10.73758 | 2.41774 |
| All-Q4 / A8 | 6.45516 | 9.59965 | 2.26963 |
| Mixed / A4 | 6.56223 | 9.53078 | 2.28199 |
| Mixed / A8 | 6.34720 | 9.16747 | 2.23071 |
| Selective / A8, either head | 6.41437 | 9.35286 | 2.24231 |
| Four-role / A8 | 6.41363 | 9.33412 | 2.23561 |

For a5% per-text PPL screen across both schedules, all-Q4/A8 is the practical light/fast ordinary-decode choice. With a2% screen across both, four-role has the best measured prefill and decode rates among eligible rows; mixed/A8 has the lowest prefill PPL but costs capacity and speed. These are diagnostic PPL screens, not the existing BF16-source severe-position admission gate. All-Q4/A4 is unattractive: worse PPL and slower than all-Q4/A8. No production artifact was replaced, and this does not establish a DFlash winner for a different base recipe.

Strict per-text prefill-quality/severe-position/speed dominance retains all five A8 rows; including short-decode severe positions retains all seven rows. Small sample-specific severe counts and timing differences therefore make the mathematical frontier less selective than the practical screens above. Neither frontier asserts statistical significance. Raw per-text newly severe positions and timing ranges are retained rather than hidden by an aggregate.

Exact commands, matched inputs, aligned NLLs, per-repeat speeds, binary identities, and reports: `profiles/ppl/r9700-nvfp4-multitext-pareto-20260922/`. The reproducible driver is `tools/ppl/compare_nvfp4.py`; `comparison.md` and `comparison.json` contain per-text matrices and strict measured frontiers. These three local samples do not replace BF16-source production admission, general capability testing, or DFlash acceptance/throughput qualification.

## Corrected concurrent decode qualification (2026-09-22)

The unchanged selective-protected tiled-head artifact named below passes the finite P89/G128 greedy workload at C1–4 with DFlash K4/K5 exactly matching fresh same-C ordinary tokens. All12 graph cells use warm1/r3; all9 C2–4 eager cells use warm0/r1. Repeated graph outputs and eager/graph tokens and speculative accounting agree exactly. R9700/gfx1201, ROCm10, fixed G16 cache, context1024, chunk4096 and power auto are unchanged.

| Concurrency | Ordinary aggregate tok/s | DFlash K4 aggregate tok/s | DFlash K5 aggregate tok/s |
|---|---:|---:|---:|
| 1 | 27.51 | 58.75 | 57.29 |
| 2 | 34.04 | 69.31 | 64.42 |
| 3 | 45.34 | 87.94 | 77.15 |
| 4 | 57.67 | 97.99 | 86.73 |

These are medians of three unprofiled graph repetitions; divide aggregate throughput by C for normalized per-request throughput, not measured request latency. Added lanes rotate the corpus. K4 wins at every measured C; C2 K4 now exceeds C1 aggregate throughput by18%. The correctness changes cost about8% C1 speculative speed versus the earlier measurement below. C4 speculative rates also decrease; the old faster numbers failed exact-greedy qualification.

The corrections canonicalize small-token K5120 RMSNorm, protected BF16 Linear, projected GDN controls, and accepted-token replay-fold normalization/dot association. Independent FP64 Op oracles pass; same-input checks establish cross-width equality and replay state matches ordinary snapshot state exactly. The artifact, output-head math and tolerances are unchanged, and target verification is still batched. C4 ordinary output changes from its old width-dependent arithmetic; shared ordinary lanes now match across C1–4. This finite workload is not universal context parity, terminal artifact/PPL selection, or proof of bandwidth saturation or exhausted speed headroom.

Evidence, exact launch commands and retained failed localization runs: `profiles/bench/r9700-dflash-concurrent-correctness-20260922/` (`run_final.py`, `summarize_final.py`, `final-summary.json`, per-cell reports). Output directories are create-only.

## Historical pre-correction bandwidth and concurrency follow-up (2026-09-22)

The earlier selected build was measured without kernel or recipe changes. Its numerical failures are resolved only by the subsequent corrected qualification above. The fresh native 4 GiB read-stream probe sustained **636.0 GB/s median** under `auto`, with exact checksum; write/copy medians were 588.0/548.6 GB/s. This is a streaming reference, not measured inference traffic. The installed gfx1201 counters do not reliably establish absolute GDDR6 byte rates.

Descriptor- and schedule-based C1 weight accounting gives the following conditional comparison. It excludes inactive tensors and full-table embedding/codebook scans, counts selected rows and known weight replays, and uses observed useful decode outputs per speculative round (including correction/bonus tokens), rather than nominal draft width.

| C1 mode | Measured decode tok/s | Modeled weight GB/round | Useful tokens/round | Weight-only stream reference tok/s |
|---|---:|---:|---:|---:|
| Ordinary | 27.53 | 15.482 | 1.000 | 41.08 |
| DFlash K4 | 63.76 | 17.066 | 3.368 | 125.53 |
| DFlash K5 | 62.15 | 17.113 | 3.459 | 128.57 |

These are **not achievable-speed predictions or proven ceilings**. The model assumes reuse within weight calls and omits some rereads, activation/intermediate traffic, state work, compute, synchronization and scheduling. Its logical weight rates of 426/323/307 GB/s must not be called physical bandwidth utilization. Neither saturation nor exhaustion of optimization headroom is established. The complete source-accounted assumptions and fresh C1 reports are retained in `profiles/bench/r9700-decode-bandwidth-concurrency-20260922/bandwidth/fresh-c1-accounting.json`; `run_stream.py` and `account.py` in that directory reproduce the respective measurements/model.

### C1–4 measurements

The admitted saved binary `r9700-bf16-controls-wave32-20260922/whole-inference/candidate-ninfer_bench` and the selected tiled-head artifact below were held fixed: greedy, Device Graph, G16 fixed cache, P89/G128, context1024, chunk4096, one warmup and three repetitions per cell under `auto`. Each cell ran in a separate process, serialized on the GPU. Values are mean aggregate decode tok/s, with normalized per-request throughput (`aggregate / C`) in parentheses, not measured individual request latency.

| Concurrency | Ordinary | DFlash K4 | DFlash K5 |
|---|---:|---:|---:|
| 1 | 27.53 (27.53) | 63.76 (63.76) | 62.15 (62.15) |
| 2 | 33.79 (16.90) | 62.25 (31.12) | 61.66 (30.83) |
| 3 | 44.67 (14.89) | 89.11 (29.70) | 82.08 (27.36) |
| 4 | 56.82 (14.21) | 108.92 (27.23) | 98.77 (24.69) |

Relative to each mode's fresh C1, aggregate ratios at C2/C3/C4 are ordinary 1.228/1.623/2.064, K4 0.976/1.398/1.708, and K5 0.992/1.321/1.589. C4 whole-output rates including prefill are 51.89/91.15/83.67 tok/s respectively. The stock benchmark rotates the corpus by lane index: added lanes have different P89 prompts, so these ratios describe this concurrent workload, not an isolated identical-request scaling test. Same-C modes use the same per-lane inputs. K4 acceptance rates for C1–4 are 60.40/56.91/60.76/64.16%; K5 rates are 50.28/47.87/51.59/56.04%. K4 has one summed fallback step at each C2–4 cell; K5 has none.

All three repetitions within every cell reproduce exact tokens and accounting. C1 K4/K5 also match ordinary greedy output. **C2–4 DFlash does not match same-C ordinary output on all lanes**: matching lanes are 0/2, 1/3, 1/4 for K4 and 0/2, 0/3, 0/4 for K5. These are throughput observations, not exact-greedy production admission. No numerical gate was relaxed and no kernel was changed in this measurement task.

Source dispatch establishes a plausible follow-up, not a measured causal attribution: several Q4 pipeline, BF16 staging/control and RMSNorm optimizations select only T5/T6, whereas batched verification uses T10/12, T15/18 and T20/24. K4 proposal batches requests; K5 proposal executes per sequence. These route differences prevent extrapolating C1 optimization coverage or ideal weight reuse to C2–4. Any follow-up must resolve the relevant exact-greedy discrepancy and measure the actual larger-batch owner before optimization or promotion.

Evidence and exact launch commands: `profiles/bench/r9700-decode-bandwidth-concurrency-20260922/` (`concurrency/run.py`, `concurrency/summary.json`, per-cell reports/receipts under `concurrency/results/`). The first ordinary process succeeded; a checker incorrectly expected `spec=mtp` instead of the reported draft-disabled `spec=none`. The corrected checker validated and reused that retained run; no failed GPU measurement was overwritten or repeated. C1 traffic accounting is not extended to the fallback-bearing C2–4 schedule as a measured physical traffic estimate.

### Measured C2 verification cost

Pre-correction C1/C2 K4 traces reproduce each mode's saved tokens and accounting. These are not timings of the corrected build above. Profiling measures kernel attribution, not unprofiled speed or physical memory utilization:

| Active batch | Physical graph launches | Kernel ms/graph | Target-layer ms/graph |
|---|---:|---:|---:|
| C1 | 38 | 47.122 | 38.585 |
| C2, two active lanes | 37 | 97.565 | 87.180 |
| C2, one-lane tail | 5 | 47.256 | 38.731 |

The C2 target layers take 2.26 times C1's cost, overwhelming the reuse elsewhere: target norm/head is only 2.328 versus 2.221 ms/graph, and drafting is 5.485 versus 4.218 ms. The tiny N48 BF16 control projections are a major regression: their T10 WMMA route totals 15.660 ms/full C2 graph, versus 0.761 ms for the C1 paired-wave32 controls. C2 also leaves the T5 Q4/down and normalization specializations. Generic N5120 Q4 trace rows include both down and another projection; they cannot all be attributed to down alone.

Across the complete run, graph cost per useful output increases from 13.989 to 15.024 ms. Lower acceptance (56.91% versus 60.40%) and five one-lane tails also limit sharing. The actual 79 lane evaluations are 37*2+5, consistent with 78 speculative rounds plus one fallback; lane-summed rounds are not graph counts. Prompt rotation confounds an isolated acceptance comparison. These findings explain the measured scaling, but do not resolve the separately observed numerical mismatch or authorize a route change.

Explicit databases, attribution, reconstruction scripts and independent topology checks are retained under `profiles/rocprof/r9700-dflash-c1-c2-k4-20260922/` (`analysis.md`, `comparison.json`).

## Selective-protected DFlash decode (2026-09-22)

The currently selected evaluation artifact is `/path/to/models/qwen3.8-27b-r9700-q4-selective-protected-dflash2-q4/qwen3.8-27b-r9700-q4-selective-protected-n16k16-dflash2-q4-head-n16k16-eval.ninfer` (18,887,772,672 bytes). It losslessly reorders only the W8 output-head codes and scales; 1,189 other payloads remain byte-identical to the original combined artifact. The original `qwen3.8-27b-r9700-q4-selective-protected-n16k16-dflash2-q4-eval.ninfer` is retained with its matched control binary. That companion preserved all 1,124 selective-protected base objects and added 32 Q4G64 matrices and 34 BF16 objects, retaining both selector codebooks. This is AMD Q4, not NVIDIA NVFP4, and does not establish cross-platform quality equivalence.

Matched ordinary/K4/K5 baselines use C1, context 1024, chunk 4096, G16, Device Graph, greedy sampling and one warmup. The 89-token chat prompt generates 128 decode tokens; ordinary decode measured 24.77555 tok/s. Raw corpus P128+G64 has much lower acceptance: 1.65789 output tokens per speculative round versus chat K4/K5's 3.36842/3.45946. It is not evidence by itself that companion quantization is defective.

The selected W8 wave-kernel change packs identical operand bytes into wide loads without changing scales, arithmetic, reduction order, or dispatch. Exact embedded gfx1201 ISA changes 32 byte-load sites to four B64 loads, 41 load waits to 11, and 83 VGPRs to 61; two signed IU8 WMMA sites and zero LDS/private storage remain. Independent original-BF16-input FP64 checks, exact A8 codec, tails, four-mod-eight code alignment and poisoned graph replay pass. All six complete output tensors are byte-identical to the retained incumbent. An initial qualifier initialization race was repaired before admission; numerical thresholds were unchanged.

Balanced whole-Engine A/B uses three fresh-process pairs per cell, both launch orders, one warmup and one measured repetition each, under `auto`. Every pair improves, each cell exceeds the predeclared 2% mean paired decode-throughput gain, and all 24 runs preserve exact tokens and speculative accounting:

| Workload | Draft width | Control decode tok/s | Selected decode tok/s | Mean paired gain |
|---|---|---:|---:|---:|
| Raw text | K4 | 17.50045 | 18.52400 | 5.85% |
| Raw text | K5 | 17.24277 | 18.38723 | 6.64% |
| Chat | K4 | 35.02727 | 37.96226 | 8.41% |
| Chat | K5 | 36.35116 | 39.00057 | 7.29% |

Chat K5 whole-output throughput, including prefill, is 36.06263 tok/s. These are workload-specific C1 results, not a 60 tok/s claim, C2–4 admission, or terminal artifact selection. Evidence and reproduction runners are under `profiles/bench/r9700-w8-packed-loads-20260922/`; matched baseline packages are `r9700-dflash-selective-baseline-20260922` and `r9700-dflash-selective-chat-20260922`.

The pre-change graph trace attributes 68.616 ms to target layers, 17.394 ms to the target final norm/head, and 5.155 ms to draft layers per graph. Its largest remaining target-layer families are Q4 gate/up, normalization, Q4 down and protected BF16 projections. These intercepted durations are attribution only; they must not be reported as post-change timing. Reattribute before choosing a subsequent kernel mechanism. Raw-text acceptance remains a separate limitation.

### Exact-order normalization admission and rejected refinements

The subsequent K5120 RMSNorm change selects the existing exact-order token8 kernel at exactly T5/T6, preserving the serial FP32 reduction order and all other shape predicates. Independent original-BF16 FP64 checks, exact incumbent outputs and poisoned graph replay pass. Balanced three-pair chat A/B under the same C1 workload admits this change:

| Width | Control decode tok/s | Selected decode tok/s | Mean paired gain | Selected whole-output tok/s |
|---|---:|---:|---:|---:|
| K4/W5 | 38.02294 | 41.27905 | 8.57% | 38.01150 |
| K5/W6 | 38.54031 | 42.17567 | 9.48% | 38.72952 |

Every chat pair wins and preserves exact token IDs and speculative accounting. Raw-text K4/K5 one-pair checks also pass, but are regression checks, not additional balanced performance admission. Evidence: `profiles/bench/r9700-dflash-rmsnorm-exact-20260922/summary.json`.

Two follow-up mechanisms are rejected. All three W8 scale-gather variants preserve exact outputs, but generic half-wave gathering loses its direct screen, and sustained fixed-T full-wave shuffle and explicit-readlane variants both lose at T5/T6. The packed-load baseline remains selected; lower ISA load/wait counts do not establish speed. Timings, protocol distinctions and retained reports are summarized in `profiles/bench/r9700-dflash-scale-gather-20260922/README.md`. Mapping the same RMSNorm row arithmetic to one 32-thread CTA per row also preserves exact outputs and stays within one BF16 step of the independent oracle, but takes approximately 0.070 ms versus token8's 0.050 ms; it is not promoted. Evidence: `profiles/bench/r9700-dflash-rmsnorm-exact-20260922/row-cta-screen.json`.

The Q4 gate/up scale-gather candidate subsequently retained exact parity but failed whole admission: chat K4/K5 mean paired gains were only 1.29%/1.20%, each with a losing pair. Its selector was removed. Evidence: `profiles/bench/r9700-dflash-gate-up-scale-gather-20260922/planner-retry/summary.json`.

The packed normalization epilogue has narrower admission. Chat K4/W5 improves 41.14269 → 42.60744 decode tok/s, a 3.56% mean paired gain with all three pairs winning. K5/W6's 1.56% mean gain includes a losing pair and is not admitted. Selection is therefore restricted to K5120 **rows5 only**; rows6 retain the exact scalar token8 epilogue. The combined candidate's top-level `NO_WIN` result is preserved rather than presented as an all-width win. Raw one-pair results remain regression-only evidence. See `profiles/bench/r9700-dflash-rmsnorm-packed-20260922/summary.json`.

Protected BF16 pair-load/K512 staging passes its original diagnostic at both qualified shapes, T5/T6 and aligned/two-byte-offset operands, with exact incumbent outputs and full-K independent FP64 checks. Aligned N7168/K5120 calls improve approximately 0.655/0.670 → 0.210/0.221 ms; N5120/K6144 improves 0.498/0.507 → 0.157/0.175 ms. Evidence: `profiles/bench/r9700-bf16-staging-20260922/screen/report.json`. Qualification through the actual selected generic dispatch now passes. The combined rows5-packed-plus-BF16 whole A/B passes all three balanced chat pairs at both widths, preserving exact tokens and speculative accounting: K4 40.78243 → 45.45008 tok/s (+11.50% mean paired gain), K5 42.15744 → 45.65818 tok/s (+8.30%). Whole-request rates are 41.36606 and 41.54923 tok/s. Raw-text one-pair regression checks improve to 22.30746/21.69262 tok/s; these are not repeated admission measurements. Evidence: `profiles/bench/r9700-bf16-staging-20260922/summary.json`.

### Lossless tiled head, coalesced ordinary decode and Q4 projections

An additional exact-order normalization next-block prefetch passes full numerical/graph/guard checks and emits real overlap at38VGPR with no spills. It improves T5/T6 direct medians 0.035264/0.050209→0.032304/0.046477ms, but129-call savings0.382/0.481ms miss the predeclared 2%round screen. It is not selected; its temporary code is removed. Retained evidence: `profiles/bench/r9700-rmsnorm-block-ahead-20260922/README.md`.

The selected bundle adds `r9700-w8g32-n16-k16-v1` output-head storage, a coalesced exact-reduction BF16 consumer for ordinary T1–3, and Q4 projection scale gathering at T5/T6 for N5120/K6144, N12288/K5120 and N4096/K5120. It retains the admitted BF16 staging and rows5-only packed normalization. The head permutation is inverse-byte-exact, with no requantization, duplicate resident head or runtime packing. Same recipe identity, BF16 codebooks and fixed cache are preserved. Real binding/old-layout rejection and public-consumer exact/FP64, graph and guard checks pass. The initial tiled ordinary consumer regressed C3 by 3.89% and was superseded; its failed evidence remains intact. The coalesced replacement preserves all 256 logical partial chains and the reduction tree; direct T1/T2/T3 warm medians improve from 5.826/11.626/17.448 ms to 2.143/4.258/6.369 ms, respectively.

Receipt-bound whole-Engine A/B against the retained BF16-staging control passes all 24 runs, with exact token streams, speculative accounting and matched configuration. Chat uses P89/G128, context 1024, chunk 4096, greedy sampling, Device Graph and `auto` power. C1 ordinary and K4/K5 each have three balanced fresh-process pairs; C3 ordinary and raw P128/G64 K4/K5 each have one regression pair. Gains below are for the complete bundle, not an isolated projection claim.

| Workload | Control decode tok/s | Selected decode tok/s | Mean paired decode gain | Selected whole-output tok/s |
|---|---:|---:|---:|---:|
| Chat ordinary C1 | 24.87909 | 27.46441 | 10.39% | 26.18621 |
| Chat ordinary C3 aggregate | 38.56344 | 45.08326 | 16.91% | 41.98362 |
| Chat K4 C1 | 45.89039 | 53.56033 | 16.72% | 48.09161 |
| Chat K5 C1 | 45.68180 | 52.43020 | 14.77% | 47.19157 |
| Raw K4 C1 (one pair) | 22.31616 | 26.08567 | 16.89% | 23.77438 |
| Raw K5 C1 (one pair) | 21.53918 | 24.87419 | 15.48% | 22.78313 |

Every chat speculative pair wins and each width exceeds the predeclared 2% mean gain; no ordinary or raw cell regresses by 2% in decode or whole throughput. Evidence: `profiles/bench/r9700-w8-tiled-head-20260922/whole-inference-combined/summary.json`. Original startup/ordinary failures remain in `whole-inference` and `whole-inference-retry`. The resumed optimization goal remains active: this is neither a 60 tok/s result nor terminal recipe/quality or full C1–4 performance admission. Separate C2/K4 and C4/K5 startup/correctness smoke passes exact lane tokens/accounting and graph allocation bounds in the same package's `concurrency-smoke/summary.json`; those short checks make no throughput-admission claim. Fresh matched K5 attribution in `profiles/rocprof/r9700-dflash-tiled-projections-chat-k5-20260922/attribution.json` measures 51.679ms target layers, 2.238ms target head and 4.688ms draft layers per graph. Remaining major owners are gate/up13.509ms, down9.740ms and scalar normalization6.141ms. These intercepted durations identify optimization targets, not unprofiled speedups.

### Down next-group pipeline admission

The subsequent down-only next-group pipeline is selected at N5120/K17408, T5/T6. Public-Op qualification passes exact outputs, the original represented-input FP64 oracle, graphs and guards. The matched whole A/B uses the admitted tiled-head/projection binary as control and the identical tiled artifact on both arms; only the down implementation changes. All 16 runs preserve exact streams, speculative accounting, artifact and configuration. Three balanced chat pairs per width exceed the 2% mean-gain gate with every pair winning:

| Width | Control decode tok/s | Selected decode tok/s | Mean paired decode gain | Selected whole-output tok/s |
|---|---:|---:|---:|---:|
| K4/W5 | 53.70092 | 57.31582 | 6.73% | 51.08892 |
| K5/W6 | 52.48043 | 55.87248 | 6.46% | 49.88149 |

Raw K4/K5 one-pair regression checks reach 27.86485/26.48399 decode tok/s and pass both decode/whole regression gates. Ordinary throughput is not remeasured: the exact T5/T6 predicate does not change its route. Earlier C2/C4 smoke predates this pipeline and is not new pipeline admission. Evidence: `profiles/bench/r9700-dflash-down-pipeline-20260922/whole-inference/summary.json`. The artifact/recipe is unchanged; these results do not establish 60 tok/s or terminal quality.

The pre-pipeline trace also bounds the draft-side frontier. Context append totals 0.908 ms; omitting unused Q rows in its five QKV projections could save at most about 0.262 ms. Proposal head plus selector totals 1.044 ms, and the already split SWA draft attention totals 0.139 ms. Device round-service work including postgraph folding totals about 0.672 ms. None identifies an independent roughly 1.3 ms/round (2%) mechanism; these are attribution bounds, not measured savings or proof that host overhead is absent. Do not reopen the historical serial draft-attention split-KV proposal against the current split SWA route. Normalization block-ahead qualification remains separate and unselected; the optimization frontier is still active.

### Verify projection pipeline-family admission

The next bundle extends one-group-ahead loading to Q4 gate/up N34816/K5120 and projections N5120/K6144 and N4096/K5120 at exactly T5/T6. N12288/K5120 retains its scale-gather route: its pipeline screen did not establish a useful gain. All selected routes pass the public represented-input FP64 oracle, exact incumbent outputs, poisoned graphs and guards. Linked gfx1201 ISA confirms successor-load overlap, ordered reductions and bounded final drains, with 62/63 VGPR, 22 SGPR and no LDS/private storage/spills. Public operator screens estimate 3.710/3.391 ms per round for gate/up and 1.569/1.666 ms for the narrowed projection family; these estimates are not separate whole-model claims.

The same-artifact whole comparison against the admitted down-pipeline binary passes all 16 runs with exact tokens and speculative accounting. Every balanced chat pair wins:

| Width | Control decode tok/s | Selected decode tok/s | Mean paired gain | Selected whole-output tok/s |
|---|---:|---:|---:|---:|
| K4/W5 | 57.42168 | 63.02158 | 9.75% | 55.59136 |
| K5/W6 | 55.93994 | 61.24468 | 9.48% | 54.10987 |

Raw-prompt single-pair regression checks reach 30.58844/29.02133 decode tok/s, with no decode or whole-output regression. These are C1 measurements; ordinary T1–4 and concurrent admission are not repeated for this T5/T6-only change. This is a bundle gain, not an isolated gate/up or projection whole-model gain. No quantization or artifact change is involved. Evidence: `profiles/bench/r9700-verify-pipeline-family-20260922/whole-inference/summary.json`. The subsequent paired BF16 control optimization is not included in these numbers.

That paired BF16 split-K16 candidate is rejected: its complete-Op FP64 oracle, native BF16 ISA, guards, own-graph checks and direct timing screen pass, but the first real-model chat K4 run changes output index 52 from token 413 to 3470. Overall 76/129 output positions differ, with 40 rather than 38 rounds and 88 rather than 90 accepted draft tokens. The campaign stops at this exact-token failure; no candidate speed is admitted and the unchanged T1-only control route is retained. Evidence: `profiles/bench/r9700-bf16-projected-control-splitk-20260922/whole-inference/rejection.json`.

### Exact-order paired GDN controls admission

The selected follow-up uses one wave32 per (head,token), computing the two BF16 projections with the incumbent's exact 160-FMA lane chains, shuffle reduction and BF16 rounding seams, followed by unchanged FP32 gating. Four-step blocked loads remove staged CTA barriers without reassociating either dot. Only T5/T6 changes; T1 retains its kernel, other widths retain their composition. The owning public Op requires no scratch, and the existing GDN device module is reused. Native code uses 25 VGPR/19 SGPR, with no LDS/private storage/spills. Public FP64, exact-output, graph, alignment and guard checks pass. The first scalar-load version missed three cold-cache material gates; the four-step version clears all eight warm/cold cells.

The strict same-artifact comparison against the admitted Q4 pipeline bundle passes all 16 runs with exact tokens and speculative accounting:

| Width | Matched control decode tok/s | Selected decode tok/s | Mean paired gain | Selected whole-output tok/s |
|---|---:|---:|---:|---:|
| K4/W5 | 62.43569 | 64.81318 | 3.83% | 57.02881 |
| K5/W6 | 60.68160 | 62.92418 | 3.71% | 55.64922 |

Every chat pair wins: K4 gains range 2.88–5.72%, K5 2.67–5.54%; the final control pair is slower at each width, so the mean is not a claim of a constant gain. Raw-prompt one-pair regression checks reach 31.45723/29.82590 decode tok/s and pass both decode/whole gates. Ordinary throughput is not retimed because T1 is unchanged; its latest measured C1 result remains 27.46441 tok/s. These DFlash results are C1, P89/G128 chat, context1024, chunk4096, greedy Device Graph, fixed selective-protected/Q4/BF16 artifact and typed production cache. Evidence: `profiles/bench/r9700-bf16-controls-wave32-20260922/whole-inference/summary.json`.

The bounded optimization pass has no remaining evidenced, untested material mechanism above its 2% round gate. This is not a claim of globally optimal kernels, saturated bandwidth, or optimality across other prompts, contexts, concurrency or recipes. Remaining costs include gate/up, exact-order normalization and down projections; their concrete challengers and rejections are retained. Two-group gate/up prefetch and its fixed-slot repair both fail the native overlap gate, so neither receives a GPU timing campaign. Further work needs a distinct mechanism that resolves an observed limit, not another unchanged sweep.

## Typed Text/MTP cache and attention

### Bounded-panel dense prefill (2026-09-21)

The all-Q4 N16/K16, G16, C1, spec-none P8192/chunk1024 benchmark improved from `184.8640144` to `1184.242683` prefill tok/s (`6.406x`). Three measured candidate durations were `6.870248486`, `6.901032133`, and `6.982186078 s`; median duration was `0.155599830` of the retained matched baseline. Each unprofiled cell used one warmup and three repetitions on the R9700 under `auto`. A fresh P2048/chunk4096 regression pair measured `1664.303599` versus `1659.247929 tok/s`, with candidate/baseline median duration `1.003513977`. This pair passes the 2% regression bound but does not reproduce the older `1904.339303 tok/s` measurement. Do not mix the historical value with this matched comparison.

The old route used tiled attention only for initial prefixes, falling back to serial causal attention on appended chunks. A selected-region 8K trace attributed `39087.580 ms` of `43838.812 ms` Text-prefill wall time to that fallback. The new route reuses the existing BF16-WMMA QK and FP32 softmax/PV arithmetic in bounded query panels; score/max storage stays at most 384.1875 MiB. G16/G32 independent FP64 checks cover initial/appended contexts, partial panels, all KV heads, device-active rows, fragmented pages, graph replay, and the 262144-token boundary. The typed leaf and all five ISA/resource checks pass; emitted resource counts and zero-spill status are unchanged.

Admission and complete input bindings are retained in `profiles/bench/r9700-chunked-attention-candidate-leaf-retry1-20260921/whole/result.json`. The original package retains passed raw qualification and the subsequently repaired public-leaf fixture failure. Deferred driver VRAM teardown interrupted the first whole sequence; its completed 8K result was retained, and only the missing P2048 pair was measured in `whole-completion`. These results admit the bounded-panel mechanism, not a terminal weight recipe, shared chunk, model-quality gate, or DFlash performance claim.

The subsequent C1 ordinary-prefill campaign completed 48 8K screen cells (chunks 1024/2048/4096/8192) and 24 32K finalist cells (2048/4096), crossing all-Q4, mixed Q4/W8, and four-role FP8/Q4 with G16/G32 and dense/XAttention B128/S16/tau900. Each cell used one warmup and three unprofiled repetitions under `auto`. Shared chunk **2048** wins the complete maximin-normalized objective (`0.9959696531` versus `0.8884210782` for 4096), with maximum workspace 608,387,072 versus 813,924,352 bytes. It wins eleven of twelve 8K profiles and all twelve 32K finalist pairs. Authority: `profiles/bench/prefill-chunk-selection-panel-attention-20260921.json`.

Representative matched C1/G16/chunk2048 prefill rates (tok/s):

| Recipe | 8K dense | 8K XAttention | 32K dense | 32K XAttention |
| --- | ---: | ---: | ---: | ---: |
| All-Q4 | 1189.0 | 1138.7 | 402.2 | 572.0 |
| Four-role FP8/Q4 | 1302.2 | 1260.5 | 414.6 | 608.8 |

XAttention is slower here at 8K but faster at 32K; these timing results do not admit its numerical quality or select a production recipe/group. No decode rounds or draft head execute in these measurements. Fresh BF16 references and the six matched quality campaigns have now completed at chunk2048; their results follow.

### Selected-chunk numerical quality (2026-09-21)

The independent BF16 source run and repeat match exactly in NLL/token sidecars at both lengths. BF16 PPL is `6.463887635` at8K and `5.632510588` at32K, scoring4095 and16383 positions. The same Python3.11/ROCm reference environment first passed the sampled FP64 full-span GDN oracle at4095/4096 rows. The selected artifacts and frozen dense/sparse G16/G32 scorers then completed all24 candidate cells against that reference.

| Recipe/profile | Assigned tier | G16 | G32 |
| --- | --- | --- | --- |
| All-Q4 dense | capacity-speed | pass | pass |
| All-Q4 XAttention | capacity-speed | pass | pass |
| Mixed Q4/W8 dense | accuracy | pass | pass |
| Mixed Q4/W8 XAttention | accuracy | fail at32K | fail at32K |
| Four-role FP8/Q4 dense | capacity-speed | pass | pass |
| Four-role FP8/Q4 XAttention | capacity-speed | pass | pass |

Mixed XAttention passes the mean-NLL limit but introduces19/18 new severe positions at32K, above its accuracy budget17. These are genuine measured exclusions, not missing evidence. Its passing dense controls remain eligible. No threshold or recipe tier is changed, and BF16 greedy-token differences remain diagnostic rather than an exact-quantized-model gate. Ten profiles proceed to capacity/whole eligibility; this is not final production selection. Evidence: `profiles/ppl/r9700-terminal-quality-receipt-bound-n16k16-20260921`, including the retained failed mixed-XAttention report and all aligned sidecars.

### Retained cache and attention evidence

The growing cache stores FP8 E4M3FN K, signed INT4 V, and FP16 V scales. A 32-point sweep covering G16/G32, every K/V/scale plane order, T=1..8, and 1K/4K/8K/32K contexts passed the independent layout and attention oracles after the coherent ROCm update.

The provisional latency leader is G16 with token-fastest K, feature-fastest V, and feature-fastest V scales: mean normalized latency `1.004041`, mean rank `2.188`, and 14 wins. This is not yet the permanent ABI: the retained real 8K comparison measured both G16 and G32 within their quality tier and favored G16 on NLL error against one historical BF16 realization. Current chunk2048 numerical eligibility is recorded above; matched capacity, phase, graph/eager, and whole-inference evidence still precede terminal selection. The older C=1..4 capacity matrices below remain historical. The current continuation retains all twelve profile capacity outcomes at C=1..4 and times only quality- and capacity-eligible profiles.

The production leaf uses two measured crossovers. At context 8,192 and above, ordinary T=1 and fixed-width T=4 select the three-stage split-512 leaf; T=4 includes causal, packed-tree, and device-active-row panels. At shorter contexts, T=1 selects wave32 FP8-Q/FP8-K WMMA from context 64, T=2 selects it from context 320, and remaining shapes retain represented-BF16-Q score streaming. The short-context WMMA path writes one caller-owned FP32 score panel, applies stable FP32 softmax, and keeps exact FP32-probability INT4-V accumulation vectorized because the tested WMMA PV formulation rounded information required by the represented contract.

A separately compiled score-streaming control isolates the private FP8-Q profile in PPL and other model-level comparisons. The complete 8K decode comparison scores 4,095 aligned positions: WMMA PPL is `6.538098` versus `6.541706` for streaming, mean signed ΔNLL is `-0.000552`, maximum absolute ΔNLL is `1.017672`, and 72 greedy choices differ. Neither route creates a new NLL-at-least-10 position relative to the other. WMMA reduces the complete scorer time from `882.169` to `718.316` seconds (`1.228x`) and measured within the accuracy tier against that retained source-BF16 realization at mean ΔNLL `+0.011989` with three new severe positions. This is private arithmetic, not an observable semantic cast: schedule comparison therefore gates finite aligned NLL under an explicit campaign bound and retains greedy flips diagnostically.

The short-context crossover is retained; scorer wall time supports this isolated A/B but does not replace the pending whole-inference benchmark objective. A paired pre-promotion MTP3 probe over the same 63 positions was bit-identical in both NLL and argmax sidecars (PPL `4.313991` in each build), when T=4 target verification still used score streaming. The cell JSON, raw sidecars, and machine-readable comparison are retained under `profiles/ppl/fp8-qk-wmma-crossover/` and `profiles/ppl/fp8-qk-wmma-crossover-8k/`.

The retained split-512 schema-v2 admission report passed the independent stored-byte FP64 oracle, invalid-input and workspace-boundary suites, fixed-address Device Graph replay, and an interleaved 20-sample timing matrix on the R9700 under `auto`. Across G16/G32, 8K/32K, ordinary T=1, and every active prefix of fixed-width T=4, the complete incumbent/split ratio ranges from `6.323x` to `17.646x`. The report SHA-256 is `bb4f0f2fba4773ab7c16e1f9c2e746fa9105f19ab4073fdffaa810108f2e4f1e`. The native-context caller-owned scratch is 37,847,040 bytes for T=1 and 151,388,160 bytes for T=4; it aliases across sequential layers and request slots. Promotion introduces distinct captured graph topologies; an MTP3 graph tracks its max+6 MTP-cache and max+4 Text T=4 leaves independently as each crosses 8K. The 32 C=1..4 capacity cells therefore remain pending fresh post-promotion measurement even though modeled Text-prefill scratch still dominates the global arena.

The earlier production P128..4096 initial-prefix leaf used the physically selected three-stage full-score GQA6 route. After the gated-RMSNorm, split-view SiLU, and K256 token8 RMSNorm promotions, a matched all-Q4/G16 P2048 C1 run under `auto`, with speculative execution disabled, measures `1,214.498278` prefill tok/s and `1.686293799 s` mean prefill with `0.001380049 s` standard deviation across three repetitions. The retained report is `profiles/bench/prefill-p2048-post-k256-rmsnorm-20260904.json`, SHA-256 `fe0598f603b9a3d3e5d25475b27c39da897ed0b4069566495053f04c1246b6b4`. This is retained as a historical low-context baseline and is superseded by the scalar-base production result below; it did not satisfy the explicit P2048 floor of `2,000` prefill tok/s (`1.024 s`). A matched current mixed-Q4/W8 diagnostic, using A8 activation coding, C1, chunk 4,096, and no speculative execution, is slower at `920.8914604` prefill tok/s and `2.223935366 s`. Its report is `profiles/bench/prefill-p2048-mixed-q4-w8-current-20260904.json`, SHA-256 `590f3e226d01a11269b79b959f7daf3800b3d392f7c6c8cea76c974bc1ec76df`. It is evidence that the current mixed recipe is not a shortcut to the dense P2048 gate, not a replacement for the all-Q4 authority.

The retained post-full-score trace at `profiles/rocprof/diagnostic-p2048-post-fullscore-causal-20260904/trace_kernel_stats.csv` (SHA-256 `afba65658aea8dd344b8c81b17730c8e35402b61868d7a1f39c5733e0a3e0c88`) captured both one warmup and one measured pass, so the production Q4 CTA calls and durations are two-pass totals rather than the decision attribution. The unique `ninfer_bench_measured` marker instead retains 2,249 wholly contained kernels with no boundary crossing. Its selected-region report is `profiles/rocprof/diagnostic-p2048-post-fullscore-causal-20260904/measured-selected-region-attribution.json`, SHA-256 `3993cb56bdc620f13e9ba223463f041120104d28cf91293dbe491d3bd3d976c7`. Independent kernel service is 1,729.283668 ms, with a 1,719.161926 ms timestamp union inside 1,753.456094 ms of marker wall. The remaining 34.294168 ms is kernel-inactive marker wall, not proven GPU idle because runtime/API activity was not captured.

On that measured-only basis, the production Q4 M64xN128 CTA accounts for 320 calls and 64.5355% of kernel service, ordinary GDN recurrence 48 calls and 8.4243%, dense full-score PV 16 calls and 5.0115%, dense full-score QK 16 calls and 4.8159%, the then-generic split-view SiLU 64 calls and 4.1999%, the then-incumbent gated RMSNorm 48 calls and 3.7309%, activation quantization 321 calls and 2.0451%, and the then-incumbent K256 RMSNorm 32 calls and 1.4637%. The global trace's 644 Q4 WMMA32 calls are not ordinary two-pass P2048 work: 640 layer calls belong to the startup T=1 ordinary code-warm execution and Device Graph capture, while four T=1 vocabulary-head calls cover those two startup executions plus warmup and measured prefill. Only the measured vocabulary head, at `2.585003 ms` or 0.1495% of measured kernel service, belongs to the measured P2048 pass; startup service is excluded from steady prefill attribution. Because the trace predates the gated-RMSNorm, split-view SiLU, and K256 RMSNorm promotions visible in the current executable, those rows explain the completed changes but do not establish current residual timings.

The initial prefill-chunk screen then found a metadata boundary missed by the admission fixture. A T=1 attention call at visible frontier 8,192 carried a device active-row pointer; the former row/context-only decision selected split-512, whose admitted contract rejects active-row metadata for T=1, and returned `hipErrorInvalidValue`. Workspace sizing and launch now use the same metadata-aware selector. Metadata-bearing T=1 retains the fused leaf, ordinary T=1 remains split at 8K and above, and fixed-width T=4 retains its admitted causal/tree/device-active-row split forms. The host routing test passes the exact 8,191/8,192 selector and workspace boundaries, and the direct 8K MTP reproduction completes in `profiles/bench/prefill-split512-failure-diagnostic-20260904.json`. This is regression evidence, not a prefill-chunk selection or a terminal throughput result.

The September 3, 2026 XAttention schema-v1 measurements are superseded diagnostics, not production rejection evidence. Their zero Q/K corpus made the estimator uniform and therefore forced `tau=0.900` to retain about 90% of pages; those builds also lacked retained executable/source identity and had regressed from paper B128 selection to B64.

The corrected candidate uses B128 query/keep blocks expanded to the B64 cache, counts mandatory causal blocks inside the threshold budget, includes ragged estimator planes/groups, and uses a concentrated represented timing fixture with explicit achieved keep fraction. The current S16 `tau=0.900` candidate uses a B16-query sparse FlashAttention-style consumer: native BF16 WMMA forms B16xB16 QK tiles over the ranker's packed logical keys, FP32 online softmax preserves increasing retained-key order, and each direct INT4-V/FP16-scale load is reused across 16 queries without a global score workspace.

Separate G16/G32 physical qualifiers passed the D256/Hq24/Hkv4 independent FP64 oracle at maximum absolute errors `1.6023e-8` and `1.6986e-8`, respectively, under the recorded `3e-4` absolute-or-relative tolerance. G16 measured `2.4200/8.8416 ms` sparse versus `69.1062/286.5128 ms` dense at 8K/32K (`28.56x/32.40x`), with `1.2393/4.0885 ms` consumer time. G32 measured `2.4008/8.7894 ms` versus `69.1753/287.6155 ms` (`28.81x/32.72x`), with `1.2454/4.0854 ms` consumer time. Both retained 14.06%/11.72% of pages. Schema-v4 reports have SHA-256 `697d26c1cb24a0cd059e92336a1f3494dac6cdc65a4c74aa80fef7af8d290519` (G16) and `dfcf0cd17801a83c97d978203e4763c089afc47e5bd15afe371ccceda1d6850e` (G32), with executable and source hashes beside them under `profiles/bench/r9700-xattention-b16-requal-s16-tau900-g{16,32}/`.

These concentrated-fixture results qualify the redesigned operator, not whole-model performance. Static-profile selection, the selected-profile retrieval gate, production-prefill chunk selection, and fresh whole-inference evidence remain open; no end-to-end admission follows from this isolated speedup.

A diagnostic selected-region trace of the compile-isolated S16/`tau=0.900` route at C=1, 32K context, and a 4,096-token prefill chunk is retained under `profiles/rocprof/xattention-s16-tau900-all-q4-g16-c1-32k-trace-20260904/`. The benchmark report SHA-256 is `a22837755d61d295ed00e160a0ee272b96c6bff87e90121ef9d11803f21a02d9`; the rocprof database SHA-256 is `83a42f5ce11d73f03354cf5ec2872b29fa8e1d53a45ae537d96f519be40357ae`. The derived schema-v1 attribution report is `profiles/rocprof/xattention-s16-tau900-all-q4-g16-c1-32k-trace-20260904/prefill-attribution.json` with SHA-256 `81a9efbb0115775d102203441849dabc8aebe699d8368b5f124fb8bf3e6cd6b4`. Its eight Text-prefill chunk ranges total `444.416 s`, matching the benchmark's `444.418 s` prefill measurement within `2.312 ms`. Kernel activity covers `444.068 s` of their union, leaving only `0.348 s` (`0.078%`) as host/idle gap. Independently summed kernel time is `443.849 s` in base Text prefill versus `0.318 s` in MTP-prefill; concurrent streams make those category totals non-additive. Base full attention accounts for `325.124 s`, post-mixer for `92.908 s`, and GDN for `25.817 s`. Within attention, the former serial sparse consumer is the dominant kernel at `302.770 s` over 128 dispatches (`68.13%` of prefill); the prefill-only A8Q4 linear totals `116.717 s`, and the ranker totals `15.353 s`. This rules out the MTP-prefill or host-gap explanation.

The runtime/qualifier comparison must normalize query work. The 32K qualifier's `32.450 ms` consumer result covers one 128-row query block, while the runtime chunk contains 32 such blocks: its comparable qualifier baseline is therefore `1.038 s` per layer. The final runtime chunk measures `4.146 s` per full-attention layer, about `3.99x` that baseline. If consumer time scaled only with kept pages, applying that ratio to the qualifier's `11.71875%` keep fraction would suggest roughly `46.8%` retention on this model input. That value is an inference, not an observed retention measurement, because fixed costs and the real keep distribution can also change scaling. The existing C=1 whole row and this trace remain diagnostic that motivated the B16-query redesign above. It does not predict that redesign's real model keep distribution or end-to-end speed. The production prefill chunk and fresh whole matrices must still be selected and measured before admission or whole-inference performance claims.

Selected complete target-leaf timings at context 257 are:

| Query rows | Time |
|---:|---:|
| 1, selected WMMA | about `0.104 ms` (`0.191 ms` streaming control) |
| 2, selected streaming | about `0.195 ms` (`0.196 ms` streaming control) |
| 9 | about `0.204 ms` |
| 17 | about `0.442 ms` |
| 128 | about `2.504 ms` |

At 4K context the focused raw-candidate trace attributes roughly `7.7 us` to parallel softmax and `818.8 us` to exact PV. QK/softmax/PV metadata is 24/23/15 VGPR, 0/76/0 bytes LDS, zero private scratch, wave32, occupancy 16.

The serialized boundary sweep measured T1 streaming/WMMA medians of `0.0376/0.0431 ms` at context 48, `0.0427/0.0443 ms` at 56, and `0.0479/0.0460 ms` at 64. T2 medians were `0.1696/0.1724 ms` at context 256, `0.1897/0.1851 ms` at 288, and `0.2099/0.1980 ms` at 320; the conservative T2 classifier starts at 320. At 4K, WMMA is `2.998x` and `1.499x` faster for T1/T2, while T3 remains a measurement tie and stays on streaming. The represented-input FP64 oracle, classifier, and timing record is retained in `profiles/bench/r9700-fp8-qk-wmma-crossover.json`.

Production-shaped suffix append plus selected attention was measured independently at T=1..8 and T=9/17/128 over 1K/4K/8K/32K. Append costs about 5--8 microseconds at narrow decode and 37--46 microseconds at T=128. Even an impossible zero-cost append bounds improvement to 0.80/0.20/0.10/ 0.03 percent at 1K/4K/8K/32K for decode and 0.40/0.10/0.05/0.01 percent at T=128. Ordered-pair timings showed no repeatable gain beyond clock variance.

One ordinary HIP kernel cannot globally order the independent T x Hkv append writers before the T x Hq attention readers. A cooperative launch cannot keep the T=128 grid resident, while reader-local encoding duplicates the GQA codec work sixfold and violates the transaction's single-writer/status ownership. Production therefore retains separate ordered append followed by the already fused QK, online FP32 softmax, and exact PV attention Op. Append/attention metadata is 13/23 VGPR, zero/52 bytes LDS, zero scratch, and occupancy 16.

## Linear

Standalone real-shape `[N,K]=[7168,5120]` prefill results:

| Represented weights | T16 | T32 | T64 | T128 |
|---|---:|---:|---:|---:|
| BF16 wave32 WMMA | `0.203 ms` | `0.267 ms` | `0.698 ms` | `2.418 ms` |
| provisional W8G32 | `1.119 ms` | `2.225 ms` | `4.371 ms` | `8.751 ms` |

Both routes pass direct independent FP64 formulas. W8G32 uses exact signed codes and FP16 scales with FP32 FMA because no available WMMA form preserves that codec. These measurements qualify the Op; they do not select the final model weight recipe.

The compile-time W8-A8 evaluator adds a native signed-INT8 WMMA route while preserving exact represented-BF16 execution below measured per-shape crossovers. Across the mixed artifact's 13 unique W8 shapes, A8 begins at T3 or T4 for nine shapes, T32 for `[34816,5120]`, and T64 for the three 1152-row Vision shapes; unknown shapes remain exact. At `[7168,5120]`, interleaved medians show 1.32x at T3, 1.53--1.75x at T4--8, and 3.37--5.55x at T16--128. These are operator results; matched real-model quality and whole-inference results are still required before selection.

The persistent Q4G64 evaluator additionally has separately compiled A4G64 and A8G64 activation profiles over identical artifact bytes. A8 is the production-style compile default; A4 requires an explicit evaluator build. A8 encodes a signed code into unsigned-low and signed-high nibble planes and uses two native IU4 WMMA sign modes with exact I32 recombination. Both profiles pass exact activation images and every independent FP64-formula output at `[7168,5120]`, T=1..8/16/32/64/128; four active lanes rotate across groups to cover all 64 packed K-lane/WMMA fragment positions. The timing harness measures five alternating forward/reverse route rounds, retains every sample, and reports the per-route median. The interleaved complete-Op medians are:

| Q4 activation profile | T1 | T8 | T16 | T32 | T64 | T128 |
|---|---:|---:|---:|---:|---:|---:|
| A4G64 | `0.05356 ms` | `0.07826 ms` | `0.09961 ms` | `0.11374 ms` | `0.13436 ms` | `0.24254 ms` |
| A8G64 two-plane | `0.05874 ms` | `0.08699 ms` | `0.11070 ms` | `0.12538 ms` | `0.15366 ms` | `0.30146 ms` |

A8 reduces the represented-BF16-times-decoded-Q4 maximum error from 0.212891 to 0.015625. Its WMMA kernel uses 64 VGPR, zero LDS/scratch, occupancy 16; activation quantization uses 10 VGPR, zero LDS/scratch. The machine-readable result is `profiles/bench/r9700-a8q4g64-linear-qualification.json`; direct compile-selected A4/A8 Tensor/WorkspaceArena results are retained beside it as `r9700-q4-tensor-dispatch-a4.json` and `r9700-q4-tensor-dispatch-a8.json`. Nonzero quantizer status is consumed asynchronously by both routes and poisons every output with exact BF16 NaN.

The selected wave32 quantizer was also measured across all 17 distinct all-Q4 artifact shapes, all six mixed-artifact Q4 shapes, and all 32 Q4 DFlash2 matrices at T=1..8/32/128. Five additional DFlash shapes extend the inventory to 22 unique shapes and 220 extents, including the K=25600 feature projection. Every extent matched the exact codec and independent sampled FP64 formula with zero BF16-step error. Count-weighted timing is effectively tied at the narrowest points: the selected route ranges from a 0.007% regression to a 2.034% gain for all-Q4, a 0.506% regression to a 1.401% gain for the mixed Q4 subset, and a 0.048% regression to a 1.436% gain for the DFlash matrices. The complete inventory, seven-trial interleaved samples, and medians are retained in `profiles/bench/r9700-a8q4g64-artifact-shape-sweep.json`. Physical public-wrapper checks also pass Q4 through grouped dynamic-convolution preparation and both selector chain/tree routes using their caller-owned nested Linear arena.

The earlier matched real-model 8K prefill scoring isolates activation width over identical artifact bytes. Every row scores the same 4,095 index-aligned tokens after a 4,096-token skip against the source-BF16 reference (PPL 6.460181); raw FP32 NLL and I32 argmax sidecars are retained.

| Stored matrix recipe | Q4 activation | PPL | mean-NLL delta | BF16-greedy flips | new NLL>=10 positions | score seconds |
|---|---:|---:|---:|---:|---:|---:|
| all Q4G64 | A4G64 | 7.397805 | +0.135526 | 800 | not separately attributed | 69.394 |
| all Q4G64 | A8G64 | 6.720524 | +0.039509 | 450 | 9 | 71.601 |
| 183 Q4G64 / 256 W8G32 | A4G64 | 6.765289 | +0.046148 | 420 | not separately attributed | 263.145 |
| 183 Q4G64 / 256 W8G32 | A8G64 | 6.550048 | +0.013815 | 237 | 2 | 264.720 |
| source-MSE 183 Q4G64 / 256 W8G32 | A8G64 | 6.544746 | +0.013005 | 234 | 3 | 264.980 |
| source-MSE 183 Q4G64 / 256 W8G32 | A8G64 Q4 + adaptive A8G32 W8 | 6.538677 | +0.012077 | 234 | 3 | 133.504 |

A8 is decisively useful: it reduces PPL by 0.677281 for all-Q4 and 0.215241 for the mixed artifact at only 2.207 and 1.575 seconds additional score time in these single runs. The source-MSE mixed adaptive-W8 profile is the Q4-containing accuracy leader in that retained comparison: +0.012077 mean NLL and three new severe positions measured within the accuracy tier's 0.02/five-position limits. Its represented-BF16 W8 control remains retained at +0.013005. The all-Q4+A8 profile also measured within the declared capacity-speed tier: +0.039509 is below `ln(1.05)` and nine new severe positions are below its eleven-position limit in that realization. At that point both remained capacity candidates with BF16-derived quality admission open; BF16-greedy choices are diagnostic. The adaptive W8 route is selected within the mixed profile: versus the identical stored artifact with represented-BF16 W8 activations, it improves mean NLL by 0.000928 and reduces this scorer's wall time by 1.98x. Scorer wall time is useful for this isolated A/B but is not substituted for the pending whole-inference Pareto matrix. Complete comparison data is retained at `profiles/ppl/8k-candidate-comparison.json` and `profiles/ppl/8k-candidate-comparison.md`; raw A8 results are under `profiles/ppl/q4g64-a8-real/`, `profiles/ppl/q4-w8-a8-real/`, and `profiles/ppl/q4-w8-mse-a8-real/`.

The final compile profile (A8 for Q4, adaptive A8 for W8, and the selected finite FP8-Q/K WMMA classifier) now has matched G16/G32 results at both 8K and 32K. Every FP32-NLL and I32-argmax sidecar is finite, complete, index-aligned with the independent BF16 source, and retained with its SHA-256:

| matrix recipe | cache | length | scored | PPL | mean-NLL delta | BF16-greedy flips | new severe / tier budget | quality tier |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| all Q4G64 + A8G64 | G16 | 8K | 4,095 | 6.720524 | +0.039509 | 450 | 9 / 11 | capacity-speed, one realization |
| all Q4G64 + A8G64 | G32 | 8K | 4,095 | 6.730933 | +0.041056 | 445 | 10 / 11 | capacity-speed, one realization |
| all Q4G64 + A8G64 | G16 | 32K | 16,383 | 5.892102 | +0.044561 | 1,673 | 41 / 41 | capacity-speed, one realization |
| all Q4G64 + A8G64 | G32 | 32K | 16,383 | 5.900777 | +0.046032 | 1,682 | 39 / 41 | capacity-speed, one realization |
| source-MSE Q4G64/W8G32 + adaptive A8 | G16 | 8K | 4,095 | 6.538677 | +0.012077 | 234 | 3 / 5 | accuracy, one realization |
| source-MSE Q4G64/W8G32 + adaptive A8 | G32 | 8K | 4,095 | 6.542475 | +0.012658 | 232 | 2 / 5 | accuracy, one realization |
| source-MSE Q4G64/W8G32 + adaptive A8 | G16 | 32K | 16,383 | 5.724998 | +0.015790 | 956 | 16 / 17 | accuracy, one realization |
| source-MSE Q4G64/W8G32 + adaptive A8 | G32 | 32K | 16,383 | 5.723254 | +0.015485 | 942 | 16 / 17 | accuracy, one realization |

Against that retained BF16 realization, all-Q4 measured outside the stricter accuracy tier but within its declared capacity-speed tier in every cell; 32K G16 is exactly at that realization's new-severe budget. The mixed source-MSE recipe measured within the accuracy tier in every cell, with one new-severe position of margin at 32K. Direct G16/G32 NLL differences are diagnostic and do not choose a group: three of four are within two paired standard errors, while all-Q4 at 32K favors G16 by 0.001471 mean NLL (2.18 standard errors). Under the former C=1..4 product cap, both recipes and both cache groups remained capacity candidates.

The table above is retained historical one-realization evidence, not current admission. The old source scorer was nondeterministic at both lengths; long-context tracing localized its first divergence to attention PV, and a later allocator-sensitive GDN route produced two stable 8K branches. The replacement v3 authority uses fixed-order PV and a single fused-recurrent GDN route, and binds the complete scorer/backend/environment boundary. Two fresh non-trace campaigns are byte-exact at both lengths. Their comparison is retained at `profiles/ppl/bf16-reference-deterministic-pv-gdn-v3-repeat-comparison-20260904.json` (SHA-256 `9b13916a2a3ca8b8e5d01352f98a8306414f4e13a79989c19b1d1248d5eb00de`): 8K NLL/argmax SHA-256 are `6ba4009ac23da9c831b475ce38fa98c377fd504ade81733de711be95e4092c0e` / `c9904bd5b09d00666aee8500023cb52cc6f22d5c369cc00ceba57ea04a6d80c1`, and 32K are `f5da949265ff566ebc368e3521cadc3f2ea7c687f1072ec993dbaec101ab1ab4` / `ec4e4d6b731d837f98b5f8bac7161ddacb9e55f0cf1842d68b58b9a2957a314b`.

The unchanged dense all-Q4 candidate sidecars were rebound offline to that exact authority in `profiles/ppl/xattention-dense-q4g64-v3-rebase-20260904/results.json` (SHA-256 `fc05174d70421f60351c0cd87c587d04f74eabd205e0d9431887a60c7c7bc4fa`). G16/G32 pass the capacity-speed tier: 8K mean-NLL deltas are `+0.0397932579` / `+0.0413409097` with 10/11 new severe positions of an 11-position budget; 32K deltas are `+0.0447215401` / `+0.0461926079` with 40/38 of 41. This closes only dense all-Q4 quality admission. The retained sparse sidecars require fresh scoring after the B16 consumer rebuild. The mixed-dense historical scorer binaries were overwritten, so its candidate provenance cannot be reopened for a valid offline rebase; mixed dense and sparse both require fresh acquisition using the selected prefill chunk.

The historical mixed dense schema-v6 campaign is fixed candidate-sidecar evidence at `profiles/ppl/xattention-dense-q4-w8-mse-20260903/results.json` (SHA-256 `7530b27e503baf8bf29c2fd83135c0e2395a9a04bd050a2872dd2a7b341b9220`). It binds mixed artifact SHA-256 `8fbadf14e355b1943ef9386a91ebafff0d852295a9adc0054bdd430291505ebd` and dense G16/G32 scorer SHA-256 `fb36b2fca7871fbcbdbbce32cb1da8d05965e2164fea0d2773dd8259d686b6d4` and `552e73200a351691d5451adccfa7dd91e9381631388e366d3c03c051338a967d`. Against its shared old BF16 realization, G16/G32 measured +0.011503/+0.012084 mean NLL with 4/3 new severe positions of a five-position budget at 8K and +0.015520/+0.015216 with 14/14 of 17 at 32K. These provisional pass flags are diagnostic only: the exact retained scorer bytes are no longer present, so the campaign cannot be rebound to the v3 authority and does not close current mixed quality admission.

The retained pre-split-promotion all-Q4 schema-v20 reports under schema-v13 manifests record exact C=1..4 capacity for dense and B128/S16/tau900 execution at the 4,096-token campaign-control chunk:

| execution | group | C1 | C2 | C3 | C4 | binding at C1/C2; C3/C4 |
|---|---:|---:|---:|---:|---:|---|
| dense | G16 | 262,144 | 524,288 | 570,304 | 558,080 | model context; device memory |
| dense | G32 | 262,144 | 524,288 | 593,152 | 580,416 | model context; device memory |
| B128/S16/tau900 | G16 | 262,144 | 524,288 | 553,280 | 541,056 | model context; device memory |
| B128/S16/tau900 | G32 | 262,144 | 524,288 | 575,424 | 562,688 | model context; device memory |

All sixteen historical cells are uncensored resolved effective maxima and exactly reproduce the corresponding C1..4 rows of their superseded C1..8 campaigns. The four manifests bind all-Q4 artifact SHA-256 `19d029a89c1ef1cf87420067555021a7c7b435c31a92bea7c64ccf42c03d80e9` and benchmark SHA-256 values `78b4f3ba85436badaaccb4695848406d4d8f5d4f4f1439235d8d3200758490fb` (dense G16), `5351c7c8fc4c7cd8fc136c2e6ab0107c65201d3fc0c4ab9fa4000f07e7587729` (dense G32), `ec85dfe59aa9fc8ca369f4ffb785bd33fda5959d2ac5889010643a8a01e4e775` (sparse G16), and `5780de5e08b5317c54e2212f4d3088dfeaad8bde290079c8adb42565b5e28e02` (sparse G32). The corresponding manifest SHA-256 values are `03265de925aa4be0b7a0584b60d819b80b3a1d8053e2cd0fa70eed37d104d63c`, `cb38eea0a67ceb9875aa04df5ea5c5be0c0f13881a8adf022c05ae86ee13460f`, `b75f8f4cf08c1ae28e7c51cef8b3ce11ad6eedf7605901a3b11798d50697a856`, and `74d448c2b2869dd3c37a33a26d7ce8e3f294acf221a7e934a6800b19437ee459` in the same dense-G16, dense-G32, sparse-G16, sparse-G32 order. The corresponding retained pre-promotion mixed schema-v20/schema-v13 matrices record dense and B128/S16/tau900 G16/G32 capacity at the same 4,096-token campaign-control chunk:

| execution | group | C1 | C2 | C3 | C4 | binding at C1; C2--C4 |
|---|---:|---:|---:|---:|---:|---|
| dense | G16 | 262,144 | 314,112 | 301,888 | 289,664 | model context; device memory |
| dense | G32 | 262,144 | 326,656 | 313,984 | 301,248 | model context; device memory |
| B128/S16/tau900 | G16 | 262,144 | 297,088 | 284,800 | 272,576 | model context; device memory |
| B128/S16/tau900 | G32 | 262,144 | 308,928 | 296,192 | 283,520 | model context; device memory |

Both dense manifests bind `r9700-q4-w8-mse-n16k16-eval` artifact SHA-256 `8fbadf14e355b1943ef9386a91ebafff0d852295a9adc0054bdd430291505ebd`. G16 binds benchmark SHA-256 `78b4f3ba85436badaaccb4695848406d4d8f5d4f4f1439235d8d3200758490fb` and manifest SHA-256 `cf87ab0c3cfd2e574c0c114652658973b53785af8b34026a077ad8bf5cbbed58`; G32 binds benchmark SHA-256 `5351c7c8fc4c7cd8fc136c2e6ab0107c65201d3fc0c4ab9fa4000f07e7587729` and manifest SHA-256 `93bcb29aaf2aceef9558cfd5221cb49b32688e691dd194c11f4bcf990a24995c`. Sparse G16 binds B128/S16/tau900 benchmark SHA-256 `ec85dfe59aa9fc8ca369f4ffb785bd33fda5959d2ac5889010643a8a01e4e775` and manifest SHA-256 `c4229e99a882c51158fe433a43cb44bc17b0c20c88cbaf352ecae178aec160b0`. Sparse G32 binds B128/S16/tau900 benchmark SHA-256 `5780de5e08b5317c54e2212f4d3088dfeaad8bde290079c8adb42565b5e28e02` and manifest SHA-256 `f12dfac84dc0d005e51e95142862dce1cc5abd020aeb7f98e1e354c76f937b03`. All sixteen mixed cells, and all 32 retained capacity cells across both recipes, were uncensored resolved effective maxima for their bound pre-promotion executables. The mixed dense cells exactly reproduce the corresponding historical C1..4 rows. The post-promotion capacity rerun, every whole-inference matrix, and terminal selection remain open.

The superseded mixed-recipe campaign's C=1..4 rows likewise resolved G16 to 262,144, 314,112, 301,888, and 289,664 tokens and G32 to 262,144, 326,656, 313,984, and 301,248 tokens. These values establish why the old C7/C8 exclusion no longer applies. Those old mixed manifests remain superseded history; the exact C=1..4 matrices above are retained controls and cannot become current selection inputs until rerun against the promoted executable and selected chunk.

The superseded all-Q4 schema-v12 manifests are `profiles/bench/pareto-capacity-markerfree-layout-all-q4-g16-20260903/manifest.json` (SHA-256 `be1d5727948512560f28a7e149547714f9b80dca96169646a2eef0310a883feb`) and `profiles/bench/pareto-capacity-markerfree-layout-all-q4-g32-20260903/manifest.json` (SHA-256 `c7e5020b03cd104440011fcc17095e3166defdc59045274192e6b930b027d922`). Both bind artifact `r9700-q4g64-n16k16-eval`, 15,172,829,184 bytes, SHA-256 `19d029a89c1ef1cf87420067555021a7c7b435c31a92bea7c64ccf42c03d80e9`. G16 binds benchmark SHA-256 `86763d6d3ac2ff8fc27a3815f3816aaba44a11acfaf4ebf54ca5d8991ddc7d65`; G32 binds `2c9bb5a64f0e25a307f3f2b783a527ed6b42d1671c6e826d7ca98545052e9b31`. These executables leave nested ROCTX ranges disabled in ordinary measurement. The prior `pareto-capacity-layout-*` campaigns, the completed G16 and interrupted G32 `pareto-capacity-unprofiled-layout-*` intermediate campaigns, and the schema-v11/schema-v18 `pareto-capacity-max-*` campaigns are all superseded for active selection. Their raw files remain unchanged as historical evidence. The durable exact comparison is `profiles/ppl/q4-a8-final-8k-32k-quality-comparison.json` with the concise companion `profiles/ppl/q4-a8-final-8k-32k-quality-comparison.md`. Matched whole-inference performance remains required to select a recipe, cache group, and execution profile.

Cache group and plane order are runtime-state build profiles rather than artifact metadata. G16 and G32 comparisons therefore use separately compiled evaluator/Engine binaries; the same explicit artifact is used for both when the experiment holds the weight recipe constant.

## DFlash2 and Vision

DFlash recipe-independent scheduling work and synthetic exact-shape operator qualification are now active after the terminal dense C1/P2048 prefill rejection; they do not wait for the terminal base or chunk. Physical companion, acceptance, capacity, and whole-inference campaigns begin only after schema-v7 `terminal_production_selection` fixes the base weight recipe, cache group, and execution profile and the receipt-bound shared chunk is selected. The companion is then produced directly from the real BF16 DFlash2 checkpoint; it does not automatically inherit the currently implemented all-Q4 evaluator recipe. Recipe selection first compares canonical Q4G64, source-MSE-refined Q4G64, and source-MSE-refined W8G32. Row-scaled E4M3 FP8 is conditional on exact-shape R9700 speed and DFlash quality evidence because it adds storage and currently lacks DFlash-owned prepared Linear execution. Every recipe keeps both selector codebooks and the private DFlash state in model-specified BF16.

The active proposal set contains exactly K4/W5 and K5/W6. K is the number of predicted drafts and W includes the target anchor, so these are respectively four predictions verified in a width-five one-block chain and five predictions verified in a width-six one-block chain. The former K1..11 C=1 shortlist and its derived broad frontier/capacity campaign are superseded; do not run or regenerate them. After a short recipe screen, each surviving recipe/K pair receives exact ordinary-output parity, proposal determinism, generated-quality, acceptance, whole-throughput, and C=1..4 capacity evidence. Missing capacity excludes only the exact recipe/K/concurrency cell when its failed command and logs are retained. Capacity decisions declare each candidate's eligible concurrency subset; it must include C1 to advance. Matched 8K/32K DFlash decode and retained-token fresh-prompt whole inference use same-build exact spec-none ordinary controls at every declared C.

The replacement decision authority must bind the base decision through each conversion report to the exact terminal winner, BF16 DFlash source, DFlash matrix recipe, companion artifact, executable, cache group, and K/W. It must reopen every retained report and recompute recipe eligibility, exact ordinary-output parity, exact repeated proposal/target determinism, and generated-quality evidence.

`prepare_selected_dflash.py` now prepares three recipe-separated, create-only CPU conversions using an explicit Python 3.11 environment and a fresh matching benchmark/planner build. Its schema-v2 plan validates the real terminal base and chunk before conversion. The schema-v5 `assemble_dflash_selection.py` recomputes the resulting evidence; historical fixed-Q4/K1..11 packages remain superseded.

Whole parity covers the complete requested generation including its first output token; isolated decode parity covers all 257 post-seed outputs, with seed equality inherited from deterministic whole-route parity. Before frontier ranking, each admitted cell must provide at least `1.02x` raw-mean speedup and a strictly positive two-standard-deviation conservative speedup over its matching ordinary route in both 8K/32K whole and decode measurements. C1 material screens run first; no full followup is licensed without a material K4/W5 C1 route. Missing measurements for a declared eligible C are failures, never inferred capacity exclusions. Capacity, acceptance, quality, and speed remain distinct gates.

The primary evaluation winner is the C1 frontier member maximizing worst normalized 8K/32K whole throughput, then capacity, then matched acceptance; canonical recipe/K/W order breaks only a complete tie. Separate per-concurrency frontiers retain all exclusions, including a primary winner that is unsupported or slower at C2..4. This is not dynamic recipe switching, an all-C production claim, or final production promotion.

The assembler's separate `admit` action requires an explicit recipe and K/W pair. It reopens schema-v5 evaluation and reconstructs all conversion, shortlist, capacity, C1-screen and followup inputs, then recomputes the raw gates. One chosen companion must qualify for every supported startup concurrency C1..4: complete capacity, exact ordinary-output parity, proposal determinism, generated-quality evidence, a material C1 win, and the existing matched 8K/32K whole/decode speed gates at all four C values. A candidate need not win every per-C frontier, but cells from different recipes or K/W pairs cannot be combined and the supported concurrency contract cannot be reduced. The distinct schema-v1 single-resident admission binds that explicit resident to the terminal base and all qualified evidence; schema-v5 stays evaluation-only. Its C1 screen is reused unchanged; only remaining capacity-eligible C2..4 run followups. Both original manifests are retained, with screen-owned proposal/generated-quality proof and per-stage raw ordinary-token parity. Final cutover revalidates this admission instead of accepting a bare evaluation. Admission does not rename/materialize a production artifact or satisfy the other final-cutover gates.

The first kernel hypothesis after prefill closes is a packed-W4 small-T route. Current A8Q4 WMMA uses a 16-token tile, so C1 K4/K5 proposal/head calls and W5/W6 DFlash/target calls leave most token lanes inactive. The retained complete 39-cell C1 screen admits only N34816/K5120 at T4/T5/T6: balanced-median candidate/incumbent ratios are `0.839747/0.879463/0.788788`, with conservative paired-ratio uppers `0.855786/0.895273/0.801884`; all other 36 shape/width cells remain on WMMA. The summary SHA-256 is `27cba659...b122`, independent review passed, and routing remains false.

The SHA-bound flattened screen then admits T8/T10/T12/T18/T20 at balanced-median ratios `0.821087/0.889587/0.884984/0.883982/0.926681` and rejects T15/T16/T24 at `1.139511/1.143556/1.041482`; its independently reproduced summary SHA-256 is `bc8f72b5...4feb`. Combined with C1, only T=`4,5,6,8,10,12,18,20` at N34816/K5120 are eligible. An off-by-default compile-selected candidate profile routes exactly those cells through packed dot8 for matched whole-DFlash A/B while the control build and all rejected/unlisted cells retain WMMA. The selector changes no caller-owned workspace or graph-stable addresses and is not production promotion.

The current N16 four-role canonical-Q4 control companion is materialized at `out/qwen3.8-27b-r9700-q4g64-f8e4m3-four-role-n16k16-dflash2-q4-eval.ninfer`, 22,763,026,944 bytes with SHA-256 `d8fc77c36cf17c92e96d67b9a6b5a1826a1ade4f59d59c003b2368fe981fc512`; its conversion report has SHA-256 `fb657164b9a9dc2987542ca4578b2bf75d79092a3efa8b7b1776520b4959f61f`. The matched control and candidate benchmark receipt is `profiles/bench/r9700-dflash-small-t-n16-matched-builds-20260906/build-receipt.json` (SHA-256 `cedb40ad2a995dceb339d81f530eddda17629bed211ae83033252e923a81b8ec`); it binds source commit `cd966d72ed18e1b5b7b57b664572b3c8aa1e02ce`, including the DFlash fixed-family Device Graph allowance correction from 40 MiB to 42 MiB (68 MiB total for the observed C1/K4/W5 topology), control binary SHA-256 `04122c7da1e262f0fc0ba5a3a2065b1e1750dadd64df3337038b54c3d85a1f75`, and candidate binary SHA-256 `6dbb09fcc75185ee1a09c0261c1a2dbbccbaa29021eb6f303f883b70350f469f`; normalized compile commands differ only in the compile-selected candidate. These are evaluation inputs, not a production recipe or routing decision, and the matched whole-DFlash A/B remains open.

The third bounded whole-A/B attempt cleared the graph-residency startup failure but stopped at functional qualification. Within each K4/W5 and K5/W6 role, both candidate runs are token-exact with both control runs, and every report is exact across its three repetitions. The fresh-prompt whole output is also token-exact with the fresh exact ordinary control. The isolated post-seed decode instead first differs from ordinary at output index 55 and then follows a different tail. This does not identify the small-T selector as the cause: the same isolated sequence is produced by candidate and control. Candidate timing from this attempt is inadmissible because the required isolated ordinary-parity gate did not pass. There is no speed, winner, or routing conclusion from attempt 3. The follow-up retained under `profiles/bench/r9700-dflash-p129-isolation-discriminator-20260906/results`, with result closure SHA-256 `fa8a7b8d7def4c59fc17370947373979ce920ef35fd1ca68f5315ad125267c56`, makes all four eager/graph ordinary/DFlash P128 seed authorities exact at token 24178 and reproduces each eager sequence exactly under Device Graph. Ordinary and DFlash append-versus-fresh execution both first differ at generated index 21 (128415 versus 96723). Fresh-P129 ordinary versus DFlash separately first differs at index 27 (95946 versus 98003); the isolated cross-mode pair differs at index 55 (112522 versus 100730). The result is functional classification only: graph capture is excluded, while shared append execution and DFlash target/accept/commit behavior remain separate localization owners. It provides no admissible timing or routing claim.

The follow-up exact layer-boundary capture is retained under `profiles/bench/r9700-qwen3-layer-boundary-traces-43e5e4cc-20260906/results`. Its unchanged raw closure is `bab590a7a40307a6f440a2bb41bd4c4135987eb7e67d8bac4ca5b58a38aa0d00`; the analysis-only zero-draft-report repair is bound by closure `b8096ffe54b6b51ead578c51a9357d96346073c64a37f27ef6fea91799569ec2`, and the summary SHA-256 is `ffaf7dc9a0ea43be5c81e617680d0b5b842c2323778b38e350da45d40990c3c0`. Both comparisons are exact through input, layer-zero mixer, and layer-zero MLP, then first differ after the layer-one GDN mixer. Ordinary W1 versus DFlash W5 differs in 1,219/5,120 represented BF16 residual values; fresh T129 column128 versus append T1 column0 differs in 142/5,120. The capture localizes the first visible divergence to one mixer but does not yet identify its primitive cause, establish a semantic failure threshold, or provide timing evidence.

The off-by-default K5120 rows5/6 RMSNorm selector passes its matched public numerical and graph gate in both selector states, but the focused P129 token experiment under `profiles/bench/r9700-dflash-rmsnorm-rows56-token-parity-20260906/results` rejects parity restoration. Its result closure SHA-256 is `03e10a9f92b64ac8f49de81796750fb28b3eb138399508fb0fec14444eb34337` and summary SHA-256 is `7fcd118d8a010c6162671a47700e27b743a2baf4f67b0af1d1b1ad2f6a2b6144`. Fresh ordinary and selector-off DFlash exactly reproduce their prior 28-token authorities and differ only at index 27 (95946 versus 98003). Selector-on first differs from both at index 21 (128415 versus 96723), so its index-27 token 96843 is on a different generated history and cannot establish repair of the prior index-27 mismatch. The selector remains off; these functional runs contain no admissible timing.

The selector-on internal follow-up under `profiles/bench/r9700-qwen3-layer1-gdn-detail-traces-e915a5e4-20260906/results` has result-closure SHA-256 `2a3a95672f323ac9fbcbf16a3914c4496baa8eb0e0cf8cb92da3ccdd58126907` and summary SHA-256 `93518da3aef2845594512d1800d8c08ad39352cdd15189257966f62f7728749b`. Target ordinary W1 and DFlash W5 are byte-exact for every captured layer-one GDN field from normalized h through final mixer residual x; their first visible residual difference moves to layer three, the first full-attention mixer, at 2,796/5,120 BF16 values. Text fresh T129 and append T1 are exact through h, controls, and q/k/v/z, then first differ in recurrence output o at 106/6,144 BF16 values with maximum absolute difference 0.000003814697265625. Gated normalization and the mixer residual inherit that difference. This excludes layer-one RMSNorm, control, and input projection/convolution for the selected Text column, but the trace does not distinguish an FP32 recurrence-state input mismatch from different wide-prefill/append recurrence arithmetic. It is functional evidence only.

The next exact target trace is retained under `profiles/bench/r9700-qwen3-layer3-attention-trace-617672eb-20260906/results`. Its immutable raw capture closure is `815307ab0a04a6e834eb19cc670c75e6d1f2c4c60c693f8861d0e7fb0b8eb79f`; an offline tuple-order repair changed no captured bytes, and the recovered summary and final result closure are `37fd48842066ce1150cccdad6ed00ad49e383759620f7ff96b0e1ee2c6b3013a` and `68ae75bc83ef4044148e594ca36cbc3eaa8a605c95be2aca3424eaad959ab3c2`. With small-T gate/up off and the qualified MLP-down, RMSNorm rows5/6, and GDN wave-normalization selectors on, target ordinary W1 and DFlash K4/W5 are byte-exact through the layer3 input, input RMSNorm, q/gate/k/v projections, q/k normalization, RoPE, causal visibility, and canonical cache positions 0--129. Their first difference is full-attention FP32 output element 0: bits 1052023983 versus 1052080823, with maximum absolute difference 0.02367246150970459. This localizes the first owner to full-attention route arithmetic; it is not an independent numerical oracle, timing evidence, or a routing decision.

The exact selector-one Text follow-up under `profiles/bench/r9700-fp8-e2-t129-text-parity-gate-20260906/results` retains four C1 eager arms for the same P129 history. All four emit `[96558,96917]`; fresh and append are byte-exact for both the layer-zero and layer-one FP32 recurrent states and for every captured layer-one GDN field. The first visible residual difference moves to layer-three post-mixer: 3,467/5,120 BF16 elements differ, first at hidden 0 (bits 48413 versus 48414), with the preceding boundary exact. This combined-build result is consistent with the e2 T129 algorithm removing the earlier layer-zero/layer-one difference, but does not isolate e2 causality from the other bound selectors. It is functional and timing-ineligible; whole Text parity remains blocked at the same first full-attention layer implicated by the target trace above.

A second packed-W4 candidate has passed standalone qualification through the production entry symbol for the exact DFlash MLP-down N5120/K17408, T5 cell. The retained summary is `profiles/bench/r9700-dflash-mlp-down-t5-production-symbol-qualification-20260906/summary.json` (SHA-256 `a10d6789b6df7914d32774bf5492e50be58645d6f6483a5904f28cab58a9381a`), with result closure `profiles/bench/r9700-dflash-mlp-down-t5-production-symbol-qualification-20260906/result.sha256` (SHA-256 `50e39c0ed6e5bf70c162a7e69d7fc819bddc2f458f06d090869f693f2ee555d2`). Its independent represented-input oracle reports zero BF16 steps for incumbent and candidate. Balanced medians are 0.207386 ms and 0.057680 ms respectively, a candidate/incumbent ratio of 0.278129 (about 3.60x faster). Both launch orders win; the paired-ratio two-standard-error upper bound is 0.276878 and the order-ratio delta is 0.006449. Independent review passed the exact source/binary/assembly/static bindings, hardware and power-state checks, raw-decision recomputation, and closure. The corresponding matched Engine builds are bound by `profiles/bench/r9700-dflash-mlp-down-t5-matched-builds-20260906/build-receipt.json` (SHA-256 `aa20148816786691eac0d7916201e13e3f489b8a9c78a9a24e1a309a210cf574`) at source commit `d1a9b6fb33a843363fba3ad46d6569071df1ab66`. Its control and candidate benchmark SHA-256 values are `db248587690acc76a2d4dd2e74a3a5fccd4d8558d3cb66a80082a36ff8c75949` and `1a6977420a4ddb460d0acea901ea10824ec90c798912d823f0b4c01fb24dc0f2`; the separate gate/up selector is off in both arms. The MLP-down selector remains off by default. This is standalone represented-input evidence only: it establishes neither generated-token parity nor whole-DFlash speed, recipe selection, or production routing. The corresponding T6 candidate failed its static resource gate before timing and remains on incumbent WMMA.

The owner trace under `profiles/rocprof/r9700-dflash-q4-c1-k4k5-owner-trace-plan-20260906` attributes 84.95%/85.27% of summed K4/W5 and K5/W6 decode service to target verification. Target Q4 Linear consumes 2707.578/2647.628 ms across 40/39 verification rounds, versus 497.823/472.893 ms for proposal/service Q4 Linear. K4/W5 and K5/W6 evidence SHA-256 are `dd8ffd1eb031b8c7c5004c0d45690d2d08aaa4c05eab5d82845c72a9cc397ad8` and `094139b10b607db827b6782dbe6b1a02dd2b5bee16771a35aa4211df19e11fe1`; asynchronous-drain repair SHA-256 is `95d652a9c0a2bce16d6e3e24d488b88dfe8b4b0735a2471347765118f9fc6fbc`. This identifies target N34816/K5120 gate/up only for that legacy all-Q4 execution, not for the current four-role companion. In the current artifact all 64 Text gate/up matrices are FP8 and bypass A8Q4; only the five Q4 DFlash proposal gate/up matrices use the selected small-T route per round. Direct-cell medians therefore predict only about 7.1 ms per K4 run and 5.3 ms per K5 run (roughly 0.15--0.19%), consistent with the near-neutral whole screen. The trace used a legacy RowSplit all-Q4/G32 artifact, eager P128/G64 execution, and profiler interception; its durations are attribution only, with no current-N16 performance, bandwidth/cache/stall, quality, acceptance, or whole-inference claim.

Independent represented-input oracle, exact gfx1201 ISA/resources, and current-companion whole-DFlash A/B remain required. The proposed `prepare_ragged_prefix` Wceil=12 compaction is not a live optimization: startup planning already sizes persistent features, round tensors, append positions, and workspace to the one resolved W5 or W6, and the Op writes that width directly. Its unreachable Wceil/copy branch is removed rather than generalized. Physical profiling, not static traffic alone, selects the next kernel.

Selected qualified Op results:

| Op/workload | Time | Numerical result |
|---|---:|---|
| SWA W4096 T3/B2 | `0.173 ms` | max abs `0.001246`, rel L2 `0.001619` |
| SWA W2048 T1 | `0.063 ms` | same complete oracle suite |
| grouped convolution BF16 T1/B1, D5120/G320 | `0.061 ms` | max abs `1.58101e-5` |
| grouped convolution W8 T2/B2 | `0.091 ms` | complete represented-input formula |
| Vision attention P194/S3 | about `0.087 ms` | rel L2 `0.001663` |
| Vision positional embedding P1024 | about `0.021 ms` | rel L2 `0.001633` |

The selected Vision attention route is resource-heavy (256 VGPR, 4 KiB LDS, 368 bytes private scratch) but measured faster than both its spill-free recompute candidate and scalar baseline. Resource counts alone are not a reason to replace a physically faster qualified implementation.

After the product request-lane cap moved to C/B=4, a focused physical gfx1201 rerun passed the runtime planner, GDN recurrence, target GDN composition, scalar schedule, sampling, DFlash KV append-prefix, SWA, MTP-round, and speculative-round qualifiers. The first SWA attempt exposed a stale qualifier-only B8 workspace-capacity probe; the production Op correctly rejected that out-of-contract request before launch. Changing the probe to the shared B4 maximum restored the intended active-domain coverage, and the complete rerun passed. The retained direct W2048 T16/B4 SWA point is `0.053 ms`, with maximum absolute oracle error `0.00125`. This is support-boundary correctness evidence, not model quality, capacity, whole-inference, or selection evidence.

## Speculative transitions

The retained pre-cap speculative-round measurements were approximately `2.01 ms` for chain K8/B8 and `0.59 ms` for mixed product-tree W12/B8 over 20 events at the 248320-token vocabulary. Those B=8 values are historical, not supported-product evidence. The active exact qualifier covers B=1..4; MTP next-round transformation passes exactly for K=1..5 and B=1..4. These measurements cover the complete state transition but not a model round.

## Reproduction

Build the current graph:

```sh
cmake -S . -B build-r9700 -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DNINFER_BUILD_APPS=ON \
  -DNINFER_BUILD_BENCHMARKS=ON
cmake --build build-r9700 --parallel
```

Focused physical targets include:

```text
ninfer_r9700_full_attention_qual
ninfer_r9700_swa_qual
ninfer_r9700_grouped_dynamic_conv_qual
ninfer_r9700_vision_attention_qual
ninfer_r9700_vision_pos_embed_qual
ninfer_r9700_speculative_round_qual
ninfer_r9700_runtime_planner_qual
```

The standalone Linear sweep/ISA recipe is `make -C tools/r9700 build/linear_op_qual linear-isa linear-resources`. The Q4 A4/A8 oracle, timing, ISA, and durable report recipe is `make -C tools/r9700 q4g64-linear q4g64-linear-isa q4g64-linear-resources`.

The fixed K6144 gated-RMSNorm prefill route assigns one token to each of eight wave32 waves while retaining the incumbent feature-ordered FP32 reduction and BF16 result. Its immutable pre-promotion report `profiles/bench/r9700-gated-rmsnorm-k6144-token8-ab-20260904-r3.json` (SHA-256 `31c50ee9f42edd61dae21266b64b099af0e5b3a94051eecb4f55fa783f5ed77f`) passed bit-exact incumbent and independent FP64 checks. Median P2048 time fell from 1.150442 to 0.155001 ms (7.4222x); every measured T>=128 point was non-regressing, and T64 won by 1.7874x. Production therefore selects token8 for K6144/T>=64 and retains the incumbent elsewhere. LLVM reports 22 VGPR, occupancy 16, four 128-bit input loads, and zero LDS/private/scratch.

The fixed K128 ordinary-GDN gated-RMSNorm route assigns one flattened value-head row to each wave32 wave and eight rows to each CTA only at `48*T` rows for T=1024/2048/4096/8192. Its immutable pre-promotion report `profiles/bench/r9700-gated-rmsnorm-k128-rows8-ab-20260904.json` (SHA-256 `bdada371d626a6f4398ac350f5aaebdf3318d1743fb507e32f626b3c062e2e5c`) passed incumbent BF16-bit parity, an independent FP64 complete-formula oracle, poison rewrite, input/output alias, and malformed-input checks and won every admitted extent. At P2048 it fell from 1.16636395 to 0.143720999 ms, a 0.123221397 candidate/incumbent ratio. LLVM reports 24 VGPR, occupancy 16, 128-bit reduction loads, and zero LDS/private/scratch. Other K128 row counts and feature widths retain the general route.

On the same four-role hybrid artifact and P2048/C1/chunk4096/no-spec workload, the production promotion improved the matched whole mean from 1.251199252 s / 1,636.830337 tok/s to 1.202762421 s / 1,702.749429 tok/s, a 1.040272x throughput gain. The post-promotion report is `profiles/bench/r9700-gated-rmsnorm-k128-rows8-production-p2048-c1-20260904.json`, SHA-256 `d1f67fb21427c5378e87afec4f4a273a34425223dd4c941d58e01fa3bc45a5aa`. This admits the eager kernel, but it does not promote the evaluation-only hybrid artifact or satisfy the 2,000 tok/s product floor.

The subsequent ordinary-GDN LDS-scope candidate narrowed synchronization only for CTA-private LDS and passed direct exact-parity and independent-oracle checks. Although its direct P2048 Op median improved from 3.113205 to 2.761526 ms, the matched whole run improved the accepted K128 baseline from 1.202762421 s / 1,702.749429 tok/s to 1.193522512 s / 1,715.930810 tok/s. The 9.239909 ms saving missed the predeclared 10 ms whole gate by 0.760091 ms, so the route is rejected and production retains the incumbent ordinary recurrence. The immutable terminal report is `profiles/bench/r9700-gdn-ordinary-lds-production-p2048-c1-20260904.json`, SHA-256 `0af07a8340f310b7f89a800bc90864377fb76ac5e45c2e22f8dd74d580a6a69f`.

The production Text-MLP boundary fuses SiLU-multiply with signed-A8G64 preparation only when the gate/up weight is row-scaled E4M3, the down weight is Q4G64, T=2,048, and the caller is one of the 64 main Text layers. It consumes the concatenated BF16 gate/up result, computes the FP32 SiLU-multiply, explicitly rounds that internal result to BF16, preserves the existing A8 scale/code/status semantics, and invokes the unchanged M64N128 Q4 down matrix. This removes one 142,606,336-byte BF16 write/read handoff and one launch per layer (9,126,805,504 logical bytes and 64 launches for P2048), without changing the graph-stable 608,387,072-byte workspace. Every other weight profile, token width, and layer retains the ordinary SiLU plus linear path.

The immutable direct report `profiles/bench/r9700-fused-silu-a8q4-down-p2048-ab-20260904.json` (SHA-256 `4d4143bc096d759c71c7d12cca2e3d68aa2a5c2bf0a4c74d2e972dfbc195f2b3`) passed full workspace/output bit parity, independent represented-input formula/codec probes, poison, tail/alias, and resource gates. Its stage median fell from 0.659238994 to 0.342199489 ms (ratio 0.519082598), and the complete MLP median fell from 3.926753998 to 3.644394517 ms (ratio 0.928093412). The matched hybrid whole report `profiles/bench/r9700-fused-silu-a8q4-down-production-p2048-c1-20260904.json` (SHA-256 `5daa98aafa34efcc5a55ee2eeca464ae94575c95ed60e6303cd9216ae253ab76`) improved the accepted K128 baseline from 1.202762421 s / 1,702.749429 tok/s to 1.178537057 s / 1,737.747715 tok/s: a 24.225364 ms saving, 0.97985855 elapsed ratio, and 1.02055953 throughput speedup. This clears the predeclared 10 ms whole gate and admits the exact fused boundary; it does not promote the evaluation-only hybrid artifact or satisfy the 2,000 tok/s product floor. The production static report `profiles/bench/r9700-fused-silu-a8q4-down-static-20260904.json` (SHA-256 `aea9061d6bb27fb9e0fec9508b242816cf5cf69118fe1f36b5616a3fcdbf5336`) binds the exact emitted symbol and confirms 12 VGPR, zero LDS/private/scratch, the required load/store and native-exp inventory, and no intermediate BF16 global traffic.

The exact ordinary Text P2048/C1 GDN scale-sidecar candidate passed its direct gate but failed its fixed whole-prefill gate. It combined the two represented-BF16 Q/K extracts with one incumbent-order FP32 inverse-norm sidecar per token and Q/K head, then retained the production 192-CTA/four-row-tile FP32-state recurrence. The provenance-complete direct report measured 2.252036095 ms versus 3.259594917 ms per call, or 108.0977325 ms over 48 calls, while remaining bit-exact with the incumbent and passing the complete FP64 state/output oracle. Preparation was 0.03379999846 versus 0.07719899714 ms and recurrence was 2.164277077 versus 2.893904924 ms. That report is `profiles/bench/r9700-gdn-scale-sidecar-p2048-ab-v2-20260904.json`, SHA-256 `20288276c005bf469f99d6a7dccddc497c5d49c6a2301b65dac746f71fc38304`. The clean isolated whole run measured 1.153886920 s / 1774.872034 tok/s, saving 24.650137 ms from the accepted 1.178537057 s baseline but missing the fixed 1.148537057 s / 30 ms gate by 5.349863 ms, with workspace unchanged at 608,387,072 bytes. Its report is `profiles/bench/r9700-gdn-scale-sidecar-production-p2048-c1-v2-20260904.json`, SHA-256 `110dbb4090dfa4a92ac993bd70ac4c21666b7a259abd17f63a4dbb913d9f6e95`. The candidate and its qualification surfaces were removed; production retains the general recurrence with no adjacent variant.

The exact P2048 MLP gate/up hipBLASLt algorithm check is terminal. All eight algorithms returned under the production zero-workspace preference passed the represented FP64 and eager/captured correctness gates. The best non-default result, rank 1 fingerprint `dde00100000000000000000000000000`, was effectively tied with production rank 0: matrix medians were 4.064851046 versus 4.061550617 ms (ratio 1.000812603), and complete medians were 4.186669827 versus 4.189990520 ms (ratio 0.999207470). Its projected 64-call saving was only 0.212524414 ms, far below the required 10 ms, while ranks 2 through 7 were slower. Production therefore retains rank 0 fingerprint `e0e00100000000000000000000000000`; no alternate fingerprint, workspace, capacity change, or whole run is admitted. The immutable terminal report is `profiles/bench/r9700-fp8-gate-up-algorithm-p2048-ab-20260904.json`, SHA-256 `5dd95d8f01d911e34e437175e028ed973ce025361ba61e9a269239f97c32de31`.

The ordinary-GDN output gated-RMSNorm-to-A8G64 fusion passed exact incumbent parity, an independent represented-FP64 formula/codec, poison and alias checks, and the gfx1201 static gate at 25 VGPR, 256 threads/eight waves, and zero LDS/private/scratch. Under `auto` at P2048, preparation improved from `0.239998996` to `0.196279004 ms`, but the candidate's 48-call aggregate was `9.421392202 ms`, above the predeclared `5.699527 ms` ceiling. Complete output improved from `1.421954989` to `1.358394980 ms`, only `3.050880432 ms` across 48 calls versus the required `5 ms`. The candidate is rejected and removed; production retains separate K128 gated-RMSNorm, A8G64 preparation, and Q4 output projection. The immutable report is `profiles/bench/r9700-gdn-output-gated-rmsnorm-a8-fusion-p2048-ab-20260904.json`, SHA-256 `8396c5e76b4166405acab70abce11f7f7137e264a1eb61eec3b25d0a098ba962`.

The subsequent fixed-P2048 ordinary-GDN projection/convolution direct-scatter candidate passed its direct gate: its per-layer median was 0.288159013 versus 0.559597015 ms (ratio 0.514940202), projecting 13.0290241 ms over the 48 GDN layers. The immutable direct report is `profiles/bench/r9700-gdn-prefill-projection-conv-direct-scatter-ab-20260904.json`, SHA-256 `fe852ad70f33f2e654cf31f2ef0da874c5355cdd491f9f97e027cb68142cf310`. At the isolated whole gate after the accepted MLP fusion, however, prefill improved only from 1.178537057 s / 1,737.747715 tok/s to 1.168570885 s / 1,752.573288 tok/s: 9.966172 ms and 1.00852852x, missing the predeclared 12 ms and 1.01x gates by 2.033828 ms and 0.00147148x. Workspace remained exactly 608,387,072 bytes. The immutable whole report is `profiles/bench/r9700-gdn-prefill-projection-conv-direct-scatter-production-p2048-c1-20260904.json`, SHA-256 `8ceaa1d948e0a76d29765a4255dd806a2bc3ffeba4b5ec009fc39f8b893e7c2f`. The candidate is rejected; production again uses the ordinary projection copies, causal convolution, and three extracts, and all candidate-only implementation and tooling surfaces are removed.

The dense full-score P2048/G16 16-wave PV head-partition candidate passed its represented FP64, incumbent-bit, poison/error, and static checks (70 VGPR, occupancy 16, 9,208-byte LDS, 512 threads, zero private/scratch), but missed both direct admission gates. Candidate versus incumbent medians were 6.84690714 versus 7.28727818 ms for PV (ratio 0.93956989) and 10.1469564 versus 10.8013163 ms for the complete Op (ratio 0.939418495), above the required 0.80 and 0.90 ratios. No whole run was admitted, production retains the eight-wave PV kernel, and this result does not authorize an adjacent mapping sweep. The immutable terminal report is `profiles/bench/r9700-dense-full-score-pv-w16-head-partition-ab-20260904.json`, SHA-256 `38ebd0c64352ccb0b27c1b60f92fa537865a92827901879f89b2c189c212feee`.

A capacity-preserving FP32 hipBLASLt PV replacement was also terminally rejected at P2048/G16. Four zero-workspace strided-batched calls (`batch_count=6`, `[M,N,K]=[256,2048,2048]`) preserved the existing interleaved query-head score/output planes and used one 8,388,608-byte decoded-V image. The in-process selected solution 140189 was GSU1/SK0 and passed the complete FP64 causal oracle, bitwise repeat, graph replay, invalid-frontier, fixed-domain, and no-clobber gates. Candidate versus incumbent medians were 4.524150848 versus 5.141388893 ms, only 9.875808716 ms direct saving over 16 calls; its old-bucket projection was 80.216934204 ms versus the required 61.161073 ms. Production is unchanged and no adjacent layout or algorithm sweep is admitted. The immutable terminal report is `profiles/bench/r9700-dense-fp32-gemm-pv-p2048-ab-20260904.json`, SHA-256 `b231a114f4b50684f2e2fdc3eb8a8ce08a9702053dc32e5ff9346a80132a8538`.

The dense full-score P2048 FP8-Q/Bk32 candidate also passed every numerical, liveness, and static gate (32 native FP8 and zero BF16 WMMAs, 63 VGPR, 8,192-byte LDS, occupancy 16, no spills), and was faster directly at 2.983591080 versus 3.442310095 ms per call (`0.8667409378x`). Across the 16 dense layers this is 47.73745728 ms and only 7.33950424 ms matched projected saving, short of the predeclared 32.439611 ms / 15 ms gate. No whole run was admitted. Production therefore retains the BF16-WMMA Bq16/Bk32 QK route, and the candidate-only implementation and tooling are removed without an adjacent sweep. The immutable design and terminal reports are `profiles/bench/r9700-dense-full-score-fp8-q-bk32-static-design-20260904.json` (SHA-256 `aed480c787313c280aefc83d8cdf943a6b81711872786677f905f3d4036cd521`) and `profiles/bench/r9700-dense-full-score-fp8-q-bk32-ab-20260904.json` (SHA-256 `ef0d6557a59145c3f632b96b24f41c922369d9f42fa48ab2c7b24d5b1c7e7f30`).

The fixed K256 query/key RMSNorm route likewise assigns one row to each of eight wave32 waves at T>=128 while retaining the exact feature-order FP32 reduction. Its immutable pre-promotion report `profiles/bench/r9700-rmsnorm-k256-token8-ab-20260904.json` (SHA-256 `87e4be62c5c7574273f7ee22eaf4858fa8442a496d738fa68a42d8b4452edbce`) passed exact-incumbent and independent FP64 checks and won every measured row. At 2048 rows it fell from 0.093921 to 0.041240 ms (2.2774x); at the real 49152-row query extent it fell from 1.101083 to 0.095321 ms (11.5513x). Smaller K256 rows and other widths retain the incumbent. LLVM reports 23 VGPR, occupancy 16, four 128-bit loads, and zero LDS/private/scratch.

The exact Text-MLP split view `[17408,T]` with element strides `[1,34816,34816*T,34816*T]` uses a two-dimensional feature-by-token SiLU-multiply grid at T>=128, removing generic coordinate decomposition without changing the FP32 formula or BF16 result. The immutable pre-promotion report `profiles/bench/r9700-silu-mul-split17408-2d-ab-20260904.json` (SHA-256 `a561f8b7981b836b2a5a582d093d67d087f2841efede95791aa5969a13b7bc10`) passed exact incumbent and independent FP64 checks and won every measured row. P2048 fell from 1.009159 to 0.374680 ms (2.6934x). Other layouts and T<128 retain the generic strided route; the selected gfx1201 kernel uses 9 VGPR, occupancy 16, and zero LDS/private/scratch.

The whole-product benchmark builds as `build-r9700/bench/ninfer_bench`; it requires a real accepted artifact. Run its `--help` output for the exact workload options.

## Missing final evidence

The complete 18-shard Qwen3.8-27B BF16 source and matched 8K/32K dual-A8 candidate sidecars are present. The localized deterministic scorer route is now mandatory in the implementation: hipBLAS/no rocBLAS atomics are fixed before PyTorch import, strict deterministic algorithms are enabled before backend construction, conflicts fail, and ordinary results bind the complete implementation/runtime provenance. Two fresh-process campaigns are byte-exact at 8K and 32K and therefore close the BF16 numerical authority; the unchanged dense all-Q4 sidecars have also passed offline gate recomputation against it. Its greedy argmax is diagnostic rather than an admission condition. The following remain explicitly unproven before Pareto classification:

- fresh current sparse all-Q4 and mixed dense/sparse quality acquisition against that authority;
- the selected integer weight recipe;
- G16 versus G32 under the qualified fixed plane orders;
- graph/eager model parity;
- DFlash2 acceptance and speed on the final companion artifact; existing MTP requires only regression preservation of its exact execution, cache, and state contracts, not further tuning;
- complete prefill/decode and C=1..4 throughput;
- whole-inference profiler attribution.

After the remaining candidate quality acquisition closes, the schema-v7 Pareto classifier retains every non-dominated candidate. A quality-eligible candidate is dominated only if another eligible candidate is no worse in mean-NLL delta, new-severe-position rate, resolved capacity, and every matched whole-inference speed cell, and is strictly better in at least one. Raw scorer seconds are not a throughput or dominance objective.

The Pareto frontier remains the audit record, but the product ships one artifact, cache, and execution profile. Quality is the admission gate. The classifier first retains one static-profile winner for each recipe, then applies the same global ordering to those recipe winners: normalize each required whole-inference cell against the best frontier result and maximize the candidate's minimum ratio; on an exact tie maximize minimum normalized resolved capacity, then minimize the worst fraction of the declared mean-NLL/severe-position quality budgets consumed. Canonical artifact and static-profile identity resolves only a complete measured tie. The deterministic decision uses retained per-cell means; raw repetition spread remains evidence rather than a noise tolerance. Schema v7 retains this as `terminal_production_selection` under `global_maximin_whole_then_capacity_then_quality_then_canonical_v1`, including the complete frontier, every recipe winner, the one terminal winner, and its compile-bound dense or B128/S16/tau900 attention identity. This maximin decision reflects whole-product performance without inventing workload weights or allowing an average to conceal a material regression.

`rocprofv3` runtime/kernel/memory traces and SQ busy/wave events are usable. On gfx1201, dispatch PMC cache collection additionally requires a stable power state. A focused real-model 8K G16 XAttention-consumer pass under `profile_standard` produced nonzero `GL2C_HIT`, `GL2C_MISS`, `TCP_REQ`, `TCP_REQ_MISS`, GL2 external read/write activity, and `SQ_WAVES`. Summing hardware dimensions gives a 98.371% GL2 hit ratio and 77.692% TCP/vector-GL0 hit ratio. The raw CSV is `profiles/rocprof/xattention-b16-all-q4-g16-c1-8k-cache-pmc-20260904/cache-pmc_counter_collection.csv` (SHA-256 `7bc45f6fc1342ec4d49a42e3a195ecbf9e02e3e59c219c1682eae730989a9cae`). Its 32 selected consumer dispatches use the expected 1,572,864-thread grid and total 4.5698 seconds of device time, with a 126.027 ms median. That is 2.76% of this pass's prefill duration, so the new consumer does not own the stable-profile whole duration. It is nevertheless about 3.16x slower at 4,096 real query rows than a linear extrapolation of the hot 128-row qualifier. The completed production-scale control below resolves that operator-scaling question; final matched `auto` whole-model attribution remains required.

The schema-v6 XAttention qualifier supplies the production-scale control at context 8,192/T=4,096/Hq24/Hkv4/D256. It rotates four address-distinct operand and workspace sets, reports rank and consumer stages separately, and uses signed nonzero INT4 values with exact FP16 scales varying by physical token, page, KV head, and value group behind a nonidentity page table. Its all-element FP64 softmax/PV check reconstructs the selected logical pages and therefore rejects payload or scale addressing failures that an all-zero timing fixture would conceal. Physical timing under `auto` passed exactly for both value groups. The current-tree refresh binds the updated dense control as well as the unchanged sparse implementation and independent oracles. At a measured 16.2502% keep fraction, G16 took `53.4330635 ms` (`26.5418129 ms` rank and `26.3109531 ms` consumer) and G32 took `56.6925545 ms` (`26.3639278 ms` rank and `30.2011223 ms` consumer). Both all-element production oracles reported exact zero relative-L2 and maximum-absolute error. The retained reports under `profiles/bench/r9700-xattention-current-tree-refresh-20260905/` have SHA-256 `205bac511ff61511a5e61233942beeead49829a417689a0ceaeafd4ef5427d11` (G16) and `ab5df29c798cdab9aae233da8433f84e2da8e36f8d3fde27a85f6e56d9bdcaaf` (G32). Schema v6 binds the executable, exact current source inventory, device/runtime identity, and power state, rechecks `auto` after numerical qualification and timing, and publishes only to a fresh path. This closes the stale dense-control source-hash gap and operator-scale qualification; it does not select G16/G32, a weight recipe, a prefill chunk, or a whole-model route.

The production-extent linear-Op gates also passed under `auto`. The A8Q4G64 cooperative CTA won all 32 qualified shape-by-token rows by `2.5261x` through `8.1587x`; the A8W8G32 CTA won all 16 qualified rows by `14.8205x` through `18.4695x`. Both independent BF16-step oracles reported zero maximum steps, including their exhaustive padded-tail cases. Their schema-v3 report SHA-256 values are `b3c78cab21dfc0f97521269245b8442009048aadcd9db5ffd947a4d7d3baf889` (Q4) and `1547f7d62fae4077f32f203b61831fc97b4f98bffcc930c6a29de7284f5c40c0` (W8). These results admit only the exact measured T=1,024/2,048/4,096/8,192 tuple predicates, which are now the production linear-Op dispatch boundary. Decode, partial chunks, Vision, DFlash, and unmeasured matrix shapes retain the incumbent one-wave path. The reports are immutable pre-promotion admission provenance; future decisions use rebuilt whole-model evidence rather than regenerating an admission report against changed production source. Matched model-prefill and whole-inference selection remain open.

The follow-up Q4 M64xN128 ping/pong staging challenger passed its exact oracle and had no regressing tuple, but it did not pass its predeclared `>=1.5x` admission gate. Weighted P2048 fell from `1126.778368` to `914.413888 ms`, a `0.8115295` challenger/incumbent ratio (`1.23224x`, `212.364480 ms` saved). The terminal report is `profiles/bench/r9700-a8q4-prefill-cta-pingpong-ab-20260904.json`, SHA-256 `06a2846e1ae5ba90989dcb401e578a1b96479a8e472b7aa0d5befdcbcf6281f7`. Production remains the selected M64xN128 persistent-N2 CTA at that operator-only decision point; this rejected staging-only result is not by itself a production performance claim.

The later matched whole-P2048 gate overrides that operator-only promotion decision without rewriting its evidence. The control report `profiles/bench/prefill-p2048-q4-pingpong-matched-control-20260904.json` (SHA-256 `66a4c832b67156e713fbe5d3de57598086ec4a264acf513b212d149e70ec1254`) identifies `m64n128-production` and measured `1219.355187 tok/s` in `1.679577815 s`. The challenger report `profiles/bench/prefill-p2048-q4-pingpong-whole-20260904.json` (SHA-256 `a187d24e1ed78154f40a71ec1243f1d64ef46fbb25d3f0a2b2af3937667bb244`) identifies ping/pong and measured `1348.188927 tok/s` in `1.519077416 s`. That is a `1.105657x` throughput improvement and `0.9044400` elapsed-time ratio, saving `0.160500399 s` at the whole-prefill scope. Ping/pong is now the sole production Q4 CTA for the exact eight-shape by P=1,024/2,048/4,096/8,192 predicate; the single-bank M64xN128 implementation remains only as an explicit regression/tail control.

The later source/emitted-compile-matched scalar-base/U32-voffset promotion preserves the same Q4G64/A8G64 representation, arithmetic, LDS topology, grid, ABI, and exact production predicate. Its actual-wrapper operator gate passed with a `21.057251555 ms` exact-call-weighted robust saving lower; the whole P2048 C1/chunk4096/spec-none gate reduced median total from `1097.650069` to `1075.475953 ms`, with a `20.306733144 ms` robust saving lower and exact token identity. The final production profile is `m64n128-pingpong-n16-k16-scalar-base-production`, with no build/runtime selector or duplicate full-tile path. Its `1075.438603 ms` prefill median is `1904.339303 tok/s`, which is the current dense authority but remains below the 2,000 tok/s floor. The immutable operator and whole reports are `profiles/bench/r9700-a8q4-n16k16-scalar-base-product-p2048-ab-20260905.json` and `profiles/bench/r9700-scalar-base-production-p2048-c1-ab-20260905.json`.

The next exact LDS-index-remap candidate was rejected statically without GPU execution. The selector-free canonical scalar-base object emits `88` VGPR; both algebraically equivalent remap forms emitted `92` VGPR. All three retained the same 17,152-byte LDS allocation, occupancy 16, zero private/scratch/spill storage, eight native IU4 instructions, and ten scalar-base/single-U32-voffset loads. The candidate therefore failed its explicit no-VGPR-regression gate before numerical or timing work. Both private remap forms and their temporary qualification surfaces were removed; the restored canonical `r9700_linear.hip` SHA-256 is `7d204bb10feeeb5988c26960ce992a6f487fd53afc162e89508ee3524b90b576`.

Complete dual-FMAC pairing preserved the canonical static resource contract (88 VGPR, 17,152-byte LDS, occupancy 16, zero scratch/spills, eight IU4 sites, and ten protected scalar-base/U32 loads) while changing the eligible instruction mix to eight dual FMACs and zero singles. The v1 and v2 operator reports were retained as inconclusive startup-transient evidence: `profiles/bench/r9700-q4-prefill-dual-fmac-p2048-ab-20260906.json` (SHA-256 `7c440f0b318c0886ad5e47a7629e91e992234637a96125b026943ad00edf9145`) and `profiles/bench/r9700-q4-prefill-dual-fmac-p2048-ab-v2-20260906.json` (SHA-256 `a60b1a7ae2795b6be0c75ed9da657c88e782092d84417a3dcd71d5884ead64d6`). After adding two balanced unscored intervals per server, v3 accepted every operator cell (robust upper ratios at most `0.9655278292`) and achieved a `12.6098960755 ms` call-weighted robust saving lower; its report is `profiles/bench/r9700-q4-prefill-dual-fmac-p2048-ab-v3-20260906.json` (SHA-256 `42c3847f7b52a8d9c436dcc957175439191decc057f51d2234c8d729676f6e1d`).

The terminal whole-P2048 gate rejected the candidate despite exact output parity and token identities passing. Control/candidate medians were `1072.2601595`/`1059.691983 ms`; ratio upper `0.9924224143` passed, but the robust saving lower was `8.1438331535 ms`, below the required `10 ms`. The authoritative report is `profiles/bench/r9700-q4-prefill-dual-fmac-whole-p2048-c1-full-v2-20260906/report.json` (SHA-256 `88da09557579e3d2e729ef1d7858c69fca5a29718c246ad9dc2403749cedfce1`). The selector, challenger, and timing tooling were removed, leaving canonical scalar-base production unchanged. The final explicitly scheduled two-entry register-load FIFO was rejected at its loaded-object resource gate. Its fixed G+1/G+2 payload slots retained both 8,576-byte LDS banks, 17,152-byte total LDS, a 512-thread maximum workgroup, zero private storage/spills, eight IU4 sites, and the canonical six dual plus four single FMAC sites, but the selected kernel used `97` VGPR: nine above the canonical `88`-VGPR gate and one above the absolute `96`-VGPR ceiling. No numerical or GPU timing work was admitted. The immutable report is `profiles/bench/r9700-q4-prefill-register-fifo-static-rejection-20260906.json` (SHA-256 `15b9369772015ae1cb079d17272f3c027d29fb2a7148236968e2e014bf4cd133`), with extracted inner gfx1201 ELF SHA-256 `24da41789f510c0f848b62ca2020b890ce0a33be96cea05ffbaeea12ea73aef7`. The selector and candidate were removed. Canonical P2048 therefore remains `1075.438603 ms` / `1904.339303 tok/s`.

The focused retained-production trace now refreshes attribution for that exact dense C1/P2048/G0/chunk4096/spec-none route. The validated evidence is `profiles/rocprof/r9700-retained-production-p2048-trace-plan-20260906/evidence.json` (SHA-256 `73b1f5ff024ec2b761411ab5f3e64eb7c9accdbd9fb0c93720a95ceb4c4857f0`). It binds the selector-free executable SHA-256 `a7c9303bd213fa3dbdb29ca0cee73addef1de8ab6a6e6b231509e25239776425`, four-role N16/K16 artifact SHA-256 `040c6e7ed29c856718a638c00181975710d987b7d5f49f4cafbdf68911f7e7d2`, and corpus SHA-256 `27e4f63c17efe3f89b5cf278b3b1a42a737316ed4044d7d0d1d52437059d1002`.

The separate unprofiled authority remains `1075.438603 ms` / `1904.339303 tok/s`; trace timing cannot replace it. The Text range was `1076.624083 ms`, with `1043.339444 ms` active-kernel wall union and `33.284639 ms` kernel-inactive wall. Independent service attribution places Q4 projection at `334.762518 ms` across 176 calls (`65.6891` useful TMAC/s), FP8 projection at `307.310638 ms` across 144 calls, dense attention at `147.247476 ms` across 48 QK/maximum/PV calls, and GDN recurrence at `139.397112 ms` across 96 calls. Ninety-three known short runtime fill/copy dispatches overlap Text without a stage marker, totaling only `0.357956 ms` (maximum `0.006760 ms`, `0.033248%` of Text wall); they are retained as unattributed and make stage attribution explicitly incomplete. The trace has no hardware counters and establishes neither profiled throughput, physical bandwidth/cache utilization, causal stalls or stall freedom, nor practical- ceiling closure.

The completed FP8 audit splits the `307.310638 ms` matrix service exactly: 64 MLP gate/up calls at `[T,N,K]=[2048,34816,5120]` take `255.024909 ms` (`183.234` useful TFLOP/s), 48 GDN query/key calls at `[2048,4096,5120]` take `23.726265 ms` (`173.781` useful TFLOP/s), and the 32 query/key plus gate/value calls across 16 full-attention layers at `[2048,7168,5120]` take `28.559464 ms` (`168.433` useful TFLOP/s). The first two roles share hipBLASLt solution `123104`/fingerprint `e0e001...` and the attention pair uses solution `123100`/fingerprint `dce001...`. Their fresh runtime symbols and resource envelopes exactly match the prior URI-bound loaded-ELF proofs in `profiles/rocprof/r9700-fp8-gate-up-linear-execution-proof-20260904/proof/proof.json` and `profiles/rocprof/r9700-fp8-attention-qk-gate-value-linear-execution-proof-20260904/proof/proof.json`: the gfx1201 kernels contain respectively 112 and 160 static `v_wmma_f32_16x16x16_fp8_fp8` sites, use 192 architectural VGPR and zero private/scratch, and use 25,088/12,544 bytes LDS. These are E4M3-by-E4M3 FP8 matrices with FP32 accumulation and BF16 publication; they are not the separate A8W8/IU8 CTA route.

Closing the unchanged 2,000 tok/s floor from `1075.438603 ms` requires at least `51.438603 ms` of whole-P2048 saving. Removing that amount from the FP8 matrices alone would require aggregate service no greater than `255.871638 ms`, or `217.542` useful TFLOP/s (`1.20103x` current). If the dominant gate/up family supplied it, that family would need `229.531` useful TFLOP/s (`1.25266x`). The two smaller families total only `52.285729 ms`; even at the retained `400.835`-TFLOP/s issue reference they can save at most `29.998424 ms`.

The installed gate/up catalog has already reduced 791 entries to ten supported zero-workspace solutions and timed all ten; the nearest alternate projects only `0.212524 ms` saving over 64 calls. The exact custom M128xN128 route regressed from `3.605496` to `7.640492 ms` per call.

The retained M128xN256 route was reopened after separating static logical (`97`), HIP runtime-reported (`101`), and descriptor allocation-rounded (`104`) VGPR. It passed the complete-FP64 oracle at all 2,192 probes, status/alias/canary and graph checks, exact FP8-ISA/resource checks, and launch-order stability, but terminally regressed from the solution-123104 incumbent median `3.499504089 ms` to `49.380744934 ms` (`14.7860151` useful TFLOP/s). Projected complete time is `49.48067909025 ms`, and projected 64-call saving is `-2936.39941408 ms`. The immutable decision is `profiles/bench/r9700-fp8-gate-up-m128n256-retained-reopen-20260906/attempt-4/decision.json` (SHA-256 `f2510b2587bb56a18c175e8d1fcc456c3cb7ad38890b9805acfd7b159c913e9b`).

Complete removal of the downstream SiLU/A8 stage is bounded near `20.76 ms`. Therefore no FP8 catalog, custom-matrix, adjacent-fusion, or counter-only experiment is admitted as a credible `>=51.439 ms` mechanism.

Canonical dense P2048 remains `1904.339303 tok/s`; the floor has not passed, practical-ceiling closure has not been proved, and choosing whether to retain or revise that floor is now a product-contract decision rather than authorization for another unbounded optimization sweep. The bounded prefill phase is closed and work transitions to recipe-independent DFlash2 scheduling and exact-shape operators. Do not rerun this object or an adjacent tile sweep. M128N256 may return only for a materially different source-level mechanism with independent numerical/static evidence and a concrete bound for at most `3.18103603125 ms` per matrix and at least `51.438603 ms` projected whole-P2048 saving.

This exhausts the explicit bounded candidate list. A targeted audit of the canonical loaded ISA and retained P2048 trace found no new mechanism outside the rejected topology and representation families with a credible `>=35.9 ms` whole-P2048 saving. That threshold is a `10.08%` reduction of the retained `356.276973 ms` Q4 service. Canonical ISA already issues successor loads ahead of its eight IU4 WMMAs at occupancy 16; the independent optimistic bounds for activation/scale-side cleanup (`19.045 ms`) and arithmetic-encoding cleanup (`<14.3 ms`) are each too small, and the concrete second-payload overlap schedule costs the nine VGPRs that failed this gate. Those overlapping bounds cannot be summed. Candidate exhaustion is not practical-ceiling evidence, so that closure remains open. Ordinary decode has separately reached its bounded roof. The P2048 floor/product-contract decision is now next; chunk/base selection and the selected companion remain blocked prerequisites for DFlash2.

The structural M64xN256 plus ping/pong follow-up also passed its exact/FP64 oracle and every tuple was nonregressing, but it likewise failed the fixed `>=1.5x` gate. Weighted P2048 fell from `1109.850230` to `920.208770 ms`, a `0.8291288` challenger/incumbent ratio (`1.20609x`, `189.641460 ms` saved). Its terminal report is `profiles/bench/r9700-a8q4-prefill-cta-m64n256-ab-20260904.json`, SHA-256 `e1d611ff2a74ebad9c300297c93ea1722a74c57d01c71a75836987fe7fe3ba3b`. At that operator-only decision point, production remained the single-bank M64xN128 route; the later matched whole-P2048 selection above supersedes that route with ping/pong.

The subsequent trace-selected M64xN256 32-wave experiment narrowed the candidate to the dominant Q4 MLP-down tuple `[N,K]=[5120,17408]`. Its 1,024-thread kernel retained the eight IU4 instructions and compiled to 87 VGPR, 25,856 bytes LDS, and zero private/scratch storage, but was slower than production at every qualified token extent. At T=1,024/2,048/4,096/8,192 the challenger/control ratios were `1.0146974/1.0120804/1.0203344/1.0149652`; T2048 measured `3.419854/3.379034 ms`. The terminal report is `profiles/bench/r9700-a8q4-m64n256-w32-mlp-down-ab-20260904.json`, SHA-256 `a41e2a2ad65809ade16b629e698244834d2c2da065335f8fab62869813aaef5b`. Both rejected M64xN256 implementations have been removed, production remains M64xN128 ping/pong, and the bounded stop rule forbids an adjacent N256 or cache-hint sweep.

The subsequent production-geometry next-G64 scale-prefetch experiment retained the M64xN128 ping/pong tile, eight native IU4 WMMAs, exact arithmetic, and 17,152-byte LDS while moving two scale halfword loads into the existing code-prefetch window. It passed static and numerical gates at 89 VGPR, occupancy 16, and zero scratch, but lost every MLP-down/GDN value-Z/GDN output cell at T=1,024/2,048/4,096/8,192 (`1.016370x`--`1.048674x`). Trace-call-weighted P2048 regressed from `375.835487` to `391.499994 ms` (`1.041679x`, `-15.664507 ms`). The immutable terminal report is `profiles/bench/r9700-a8q4-prefill-cta-scale-prefetch-ab-20260904.json`, SHA-256 `aaeeaf16a0040d5a23046d40fff532f34805124cdd3e5bbeb5d0a721962ad8ef`. The candidate was removed; production retains its post-compute scale loads.

The bounded residual-add to K5120 token8 RMSNorm prefill fusion likewise passed exact residual and output publication, complete independent-oracle, alias, rewrite, and static resource gates at 18 VGPR, occupancy 16, and zero LDS/private/scratch. Physical `auto` timing rejected it at every T=1,024/2,048/4,096/8,192 extent: challenger/control ratios were `2.98276377/2.94057178/2.26832342/1.89297712`. P2048 regressed from `0.188480005` to `0.554238975 ms` per pair, a `-0.365758955 ms` saving. The immutable terminal report is `profiles/bench/r9700-residual-rmsnorm-k5120-prefill-ab-20260904.json`, SHA-256 `6ed367d0438f0f7f8dbfe4bd01f7da126a6857c1953da8f1ed6e294eec20c134`; the retained static design is `profiles/bench/r9700-residual-rmsnorm-k5120-prefill-static-design-20260904.json`, SHA-256 `40dcc021e59e8c80272ed6c36c9ab4ef36e1bd6c2e9939c439671426a44eca73`. The candidate and its qualification tooling were removed. Production remains the separate residual-add plus selected K5120 RMSNorm pair, with no adjacent fusion variant authorized.

A separate exact-device ceiling probe under `auto` measured `801.669240` median issued-IU4 TOPS from an eight-chain register-resident native `v_wmma_i32_16x16x32_iu4` kernel and `633.264977 GB/s` from a 272-times-L2 packed-Q4-code/FP16-scale stream with the production G64 16:1 byte ratio. The streaming result is `98.9477%` of the nominal 640 GB/s bus rate. The retained report is `profiles/bench/r9700-q4-hardware-peak-20260904.json`, SHA-256 `72e4c91eb9177ff5969cce6c06febde251a4852e1e69bd317b77646ddcb7baa4`. Its raw HIP `multiProcessorCount` value is 32 scheduler units and does not supersede the architectural 64-CU hardware description.

For scale only, the production M64xN128 weighted P2048 aggregate's `101.082` tera-issued-operation- equivalents over `1.109850230 s` is `91.077154` issued TOPS, `11.3609%` of the isolated native-IU4 ceiling. The traffic model's `842.961535 GB` over that duration is `759.527288 GB/s`, or `119.938%` of the isolated stream rate. This apparent excess is not a physical bandwidth measurement: the model counts represented source requests, including requests that may be served or combined by cache, coalescing, and LDS reuse, while the physical transactions and their overlap are unknown. Consequently these ceilings show substantial unused native-IU4 issue capacity and a healthy near-nominal isolated memory path, but do not establish production HBM utilization, cache reuse, or stall freedom. Those production claims remain gated on a matched GL2C/TCP/SQ profile.

The corresponding complete direct-A8/packed-W4-to-IU8 P2048 pipeline was rejected statically before timing. Its fair M64xN128 expand-once ping-pong kernel used 25,344 LDS bytes, exact eight signed IU8 sites, and zero private/scratch bytes, but emitted 115 VGPR/occupancy 12. Preventing four-K16 operand hoisting and replacing tail-aware admitted-shape W loads reduced it only to 111 VGPR with occupancy still 12, missing the fixed <=96 VGPR/occupancy-16 gate. The best compact legal expansion idiom projected roughly 100 VGPR even with sequential publication, so the route is closed without GPU or whole-inference timing. The retained evidence is `profiles/bench/r9700-a8q4-direct-iu8-pingpong-static-rejection-20260905.json`, SHA-256 `d4f2c610b973935aa60b732c3ee39e9dc84a512781839458a5caadcca78f2a2c`; all qualifier source, checker, test, and build surfaces were removed.

The signed-A4G64 activation challenger halved its matrix instruction count and activation-code plane, but its complete quantize-plus-M64xN128 path failed operator admission. Weighted P2048 improved from `1121.674698` to `1006.050599 ms` (`1.11493x`), while the N1024 shape regressed at every measured extent by `1.21819x` through `1.35600x`. The retained report is `profiles/bench/r9700-a4q4-prefill-m64n128-vs-a8-production-ab-20260904.json`, SHA-256 `2ac938b230197384a476bd57ccfd39d72a1b73c03bee82c58bf3be9df7de5b1f`. Production remains A8; model-quality gates were not rerun because the operator gate failed and existing A4 quality evidence already rejects that private activation profile.

The matched `auto` ordinary C1 diagnostic used the all-Q4 G16 B128/S16/tau900 build, an 8,192-token prompt, a 4,096-token chunk, the Device Graph path, and 256 decoded tokens with speculative execution disabled. Across three repetitions it measured `239.5738442` prefill tok/s (`34.1947324 s`) and `8.354688852` decode output tok/s (`30.64198944 s`); whole output throughput was `3.963623176` tok/s over `64.84022237 s`. The raw report SHA-256 is `049f3e4712a65ba618f28b47d830a96019cd16dea8dd70a920e1385aca4f1c96` and its manifest SHA-256 is `a2ce50522910bd1442156383c85f359315c33bc5964578ba3ca28bf9685e104a`. The previously retained `19.24458338` tok/s result is the MTP3 speculative cell, not ordinary decode.

The current dense all-Q4 G16 ordinary baseline supersedes that `8.354688852` tok/s row for active optimization decisions. At C1/P8192+G256, spec-none, Device Graph, and one measured repetition, it reaches `15.13345066` decode output tok/s in `16.91616841 s`. This is a diagnostic baseline until a source-matched whole A/B supplies repeated candidate/control evidence. The report is `profiles/bench/ordinary-none-dense-all-q4-g16-c1-8k-current-20260905.json`, SHA-256 `829d19d4eff2364e0d782f0da3cfe7dfa7c3bee313614799f5aa53a333e17f7f`.

The corresponding one-round ordinary trace has a `78.193701 ms` ordinary host marker and 1,806 dispatches with the exact ordinary-range association. Their independent durations sum to `64.388840 ms`: Q4 WMMA contributes `42.301380 ms` across 321 calls, RMSNorm `13.642834 ms` across 161, split512 PV `4.276113 ms` across 16, and QK `1.030131 ms` across 16. The benchmark report is `profiles/rocprof/ordinary-decode-c1-one-round-trace-20260905/benchmark-report.json` (SHA-256 `7d9da725902c04e208525ae2d02dd33fc696ab452ad6673579a730f6e8f7eee6`), the ROCPD database is `profiles/rocprof/ordinary-decode-c1-one-round-trace-20260905/raw/ordinary-decode-c1-one-round_results.db` (SHA-256 `2cae81419c0cf360eb032537ca95a4030a1c80df1e9221dc5d6dfcd06f3aa0ff`), and the corrected analysis is `profiles/rocprof/ordinary-decode-c1-one-round-trace-20260905/analysis-v2.json` (SHA-256 `ca8937876bc16259aeaf793a1c6070c5b99dac74d71f2eca4b348ff0b2f7a9d1`). A dispatch-interval union means only the union of rocprofiler kernel start/end records. Graph dispatch intervals overlap, and the union is not GPU-active time, GPU wall time, utilization, or CU occupancy; it must not be subtracted from host-marker time to assign a physical idle fraction. It directs attention first to the source-matched Q4 Linear candidate and then RMSNorm, with whole A/B as the decision authority.

The native-dot8 T=1 Q4 Linear operator gate passed every selected tuple and reported a robust ordinary-call-weighted saving lower bound of `14.1025415618 ms` per token. Its report is `profiles/bench/r9700-a8q4-t1-native-dot8-all-text-20260905.json`, SHA-256 `300626f0d45b5b9bb8f6b420f44b7e5652e3a848738c636959f02f3448d2b5b0`. The seven full-K tuples were selected from ordinary decode; `[5120,17408]` and `[34816,5120]` also occur in DFlash2, so the canonical route is shape-owned by Linear rather than caller-specific. Its post-gate production regression uses nonuniform signed W4 codes in exact N16/K16 storage, nonuniform FP16 scales, dense varied activations, full-output WMMA parity, spread-row complete-K FP64 checks, status poisoning, full rewrites, and output canaries.

The source-matched four-pair C1/P8192+G256 ordinary Device Graph gate promoted native dot8. Median decode fell from `16.9262812925 s` (`15.1244089340 tok/s`) to `12.5301623450 s` (`20.4307009719 tok/s`). The robust candidate/control upper ratio was `0.7427031671`, the robust saving lower was `17.0000856399 ms/token`, and the prefill upper ratio was `1.0016298424`; exact generated tokens and configuration, environment, artifact, and workspace identities matched. The immutable report is `profiles/bench/r9700-a8q4-t1-dot8-whole-p8192-g256-full-20260905.json`, SHA-256 `19278ca8c8df5d417bbd5d36ce1689760e3cb6dbe606c3a597a6b2f0b847fee1`. There is no build or runtime selector. T=1 calls outside the seven exact full-K tuples retain WMMA. That promotion left ordinary decode open for the trace-selected RMSNorm candidate. MTP remains supported regression behavior and is not an optimization target.

A fresh selector-free production build retained exact 257 generated IDs for P8192+G256 and measured `20.45440879 tok/s` (`12.51563918 s` decode) in `profiles/bench/r9700-dot8-production-final-p8192-g256-c1-20260906.json`, SHA-256 `c341f1eeb2f5d5597272dbecc532982b008701c7ff6a22d946541f98c9c94b2c`. The extracted loaded gfx1201 code object has SHA-256 `6a4e9eed7804da2321e112a3c520f65eb3d6a1b88e8352c39f9b364a30033312` and contains exactly 16 `v_dot8_i32_iu4` instructions and no WMMA in the selected kernel, with 18 VGPR, zero LDS/private/spills, wave32, and a 256-thread maximum workgroup.

The subsequent ordinary-decode promotion is the parallel K5120 RMSNorm CTA at exactly rows 1 through 4. The immutable operator report `profiles/bench/r9700-rmsnorm-k5120-rows4-qualification-8192i-20260906.json` (SHA-256 `e58b2e56980fb083548f1c357c26fc98a8a5bae27b1723ccf712451eaf4b8103`) passed the independent complete-formula FP64 oracle, both gain modes, three epsilon values, ordinary/zero/mixed-magnitude inputs, Device Graph replay, invalid-input no-write checks, and every row-count timing gate. Its minimum robust ordinary-round saving lower bound was `11.7782198046 ms`. Production selects this route only for K5120 rows 1..4; rows 5..127 retain generic RMSNorm, rows >=128 retain the K5120 token8 prefill route, and other feature widths retain their existing routes.

The four-pair source-matched C1/P8192+G256 Device Graph gate reduced median decode from `12.521242570 s` (`20.44525522 tok/s`) to `9.467485694 s` (`27.03991411 tok/s`). The robust candidate/control upper ratio was `0.7598847464`, the robust saving lower was `11.72510638 ms/token`, and the robust prefill upper ratio was `1.0029548902`; generated tokens were exact in every pair. The immutable report is `profiles/bench/r9700-rmsnorm-rows4-whole-p8192-g256-full-20260906.json`, SHA-256 `3f5c7a678f29b09537b46ebf7692e6f8d9b422e08f9ffe656367815932b2d0f6`.

The next bounded grouped-PV experiment did not promote. Its reduced fixed-fixture mechanism report, `profiles/bench/r9700-split512-grouped-pv-qualification-v2-20260906.json` (SHA-256 `f690accdc47bd85096aa412aa415e3f55446e8d9b737b9928cef671b1054ea6d`), is screen-only. The product-integrated route nevertheless passed exact numerical, rejection, and fixed-address Device Graph qualification in `profiles/bench/r9700-split512-grouped-t1-direct-20260906.json` (SHA-256 `958b38efaced4b671222c32de93ce72264df6a58aa6fbb5797f0738e965ba65b`). The source-matched C1/P8192+G32 whole screen then measured a `1.0038023342` candidate/control decode ratio, `-0.141697625 ms/token` saving, and `1.0033795393` prefill ratio with exact generated tokens. The rejected report is `profiles/bench/r9700-split512-grouped-t1-whole-p8192-g32-screen-v2-20260906.json` (SHA-256 `bcd24621462dbb404ecc77b79a6b324f0c4d4b67a7e0afdcbacc69281cd1d9f2`). No full gate was run; the selector and challenger were removed.

The subsequent post-RMSNorm proxy analysis is `profiles/rocprof/r9700-post-rmsnorm-ordinary-c1-p8192-g256-proxy-plan-20260906/analysis.json` (SHA-256 `8784fe63db40d13163999077da8b62f049242807cfa3045a88d0412e7a0534ae`). It validates 256 exact ordinary regions, each with the terminal 37-tuple/1,806-dispatch Device Graph inventory. Whole-round ratio-of-sums proxies are 56.7040% GL2 hit, 52.0346% TCP hit, 97.5089% wait-any/wave cycles, 1.02066% issue-wait/wave cycles, 25.6970% occupancy, and 13.1582% VALU busy. More decisively, the 82,176 dot8 dispatches account for 96.6624% of GL2 read requests, 96.4485% of GL2 misses, 83.2272% of TA activity, and 72.0015% of wave cycles. Their symbol-local proxies are 51.0133% GL2 hit, 21.5963% TCP hit, 99.3061% wait-any/wave cycles, 0.03400% issue-wait/wave cycles, 28.7011% occupancy, and 6.08721% VALU busy. These overlapping counters do not establish physical HBM bandwidth, causal stalls, or stall freedom; the separate same-session 4-GiB stream probe reaches 635.9 GB/s but is not inference traffic.

The admitted weight-only non-temporal dot8 challenger preserved exact numerics and passed its static/resource gate, but its current-identity repeated-buffer direct report rejected it at `-8.26204 ms/token` weighted point saving and `-8.3750512166 ms/token` conservative lower saving. That report is `profiles/bench/r9700-dot8-weight-nt-direct-v2-20260906/report.json`, SHA-256 `63ed87fef150a885ed3375548635ba529d8e08cbcb9df0e932b24d248e65034e`.

Since repeated buffers do not exercise the proposed downstream cache-pollution benefit, the terminal decision used a source-matched three-pair, 12-process C1/P8192+G32 ordinary Device Graph screen. All three paired decode savings were negative (`-2.683606`, `-2.5499134531`, and `-2.6231869063 ms/token`); the robust decode ratio upper was `1.0773990082`, robust saving lower was `-2.8873396633 ms/token`, and robust prefill ratio upper was `1.0186573578`. The rejected report is `profiles/bench/r9700-dot8-weight-nt-whole-p8192-g32-screen-20260906.json`, SHA-256 `c6a1dea0f0e94d9dfcd461145c83d9161e35313e17e2fe9b0ca041a9b405e31c`. No G256 gate was run. The private selector, challenger, and temporary qualification tooling were removed while both reports and raw whole-screen evidence remain immutable. This terminal rejection does not infer physical bandwidth saturation or stall freedom.

The fresh selector-free build then measured `9.461402437 s` for 256 decode tokens (`27.05729956 tok/s`) with all 257 generated token IDs retained. The final report is `profiles/bench/r9700-rmsnorm-production-final-p8192-g256-c1-20260906.json` (SHA-256 `b05db0068a4f1c73ce9c2092443b42f9f48b0fdb80ff8b3335db60cd5bdca74b`) and its executable SHA-256 is `a7c9303bd213fa3dbdb29ca0cee73addef1de8ab6a6e6b231509e25239776425`. A selected-region trace of that executable records exactly 129 rows1..4 CTA dispatches and 32 legitimate generic RMSNorm dispatches, so none of the 129 K5120 ordinary-decode calls fell back. The trace database is `profiles/rocprof/r9700-rmsnorm-production-selected-trace-20260906/raw/rmsnorm-production-selected_results.db` (SHA-256 `8fe71be97e77c2651cb0c75fe203cedb13f200f8ac76082e4310bbfb855e5370`). The extracted loaded gfx1201 code object has SHA-256 `c3dcad45559a112f42f07b1d7e87fbd1d494d1d024083efc0498500ee678cd72`; the selected kernel uses 17 VGPR, 32 bytes LDS, wave32, occupancy 16, and zero scratch/spills. This selector-free `27.05729956 tok/s` result is the bounded practical ceiling used to close ordinary base-decode work. It is explicitly not an absolute hardware or physical ceiling and does not claim the maximum achievable throughput of the R9700.

Base-decode bandwidth work was subsequently reopened and admitted three exact projection fusions. At C1, the BF16 GDN control fusion's whole-model candidate/control decode-time ratios were `0.9876921`, `0.9870972`, and `0.9868352`; the all-Q4 attention input-projection fusion's ratios were `0.9887575`, `0.9873370`, and `0.9869812`. At C2, C3, and C4, the shared-quantize paired-Q4 route passed three whole-model pairs per concurrency with median ratios `0.9246617`, `0.9272089`, and `0.9341644`, and upper-two-standard-error bounds `0.9250822`, `0.9283767`, and `0.9342851`. Every process retained exact public tokens. The C2--C4 result closure is `profiles/bench/r9700-paired-projection-c2c4-whole-ab-20260919/results`, whose `result.sha256` digest is `26c4ae49e7a24ce97d1f56c891e841942507678aaab8a0df0ada315996e93cda`. These gates authorize selector-free production routing only for singleton BF16 GDN controls, singleton all-Q4 attention projection pairs, and all-Q4 projection pairs at exactly two through four tokens. Mixed weights and other token widths retain their prior routes. These unprofiled whole timings establish useful decode-speed gains, not physical HBM saturation or stall freedom, and the two separately measured C1 gains are not added to make a combined claim. The fresh selector-free composition run then reproduced exact retained tokens at every supported concurrency. Its C1 through C4 decode times were `9.0186748`, `14.7080029`, `16.4007540`, and `18.8960216 s`, corresponding to aggregate rates `28.3855`, `34.8110`, `46.8271`, and `54.1913 tok/s`. The retained-candidate timing ratios were `0.9865955`, `0.9997177`, `1.0002975`, and `1.0000448`. Evidence is `profiles/bench/r9700-three-route-production-confirmation-20260920/results`; the SHA-256 of its `result.sha256` closure is `9be2f67ff0f74f92e509bbfa6bb0909e7feb5e38993aed184d71ce1cd7033148`.

The subsequent projected-residual fusion passed the reviewed C1/P8192+G256 ordinary Device Graph A/B under `auto`, using the same Q4G64 artifact and fixed cache. In balanced execution order, control/candidate/candidate/control/control/candidate decode rates were `28.3689630`, `29.0009861`, `29.0069622`, `28.3912167`, `28.3819193`, and `28.9849878 tok/s`. Every process preserved the retained 257 public token IDs. The paired candidate/control decode-time ratios were `0.9782068408`, `0.9787724938`, and `0.9791937627`, with mean `0.9787243658`, upper two-standard-error bound `0.9792961944`, and median `0.9787724938`. Evidence is `profiles/bench/r9700-projected-residual-t1-whole-ab-20260920/attempt-1`; its `result.sha256` digest is `e7337ec8e9bc19d8519f52f06ffcb50359270fa47a4416ff859a4aee8afa9b79`. The independent result audit admitted promotion only for all-Q4/A8 base Text T1 at N5120/K6144 or K17408; mixed inventory, MTP, prefill, T>1, and A4 retain the incumbent composition. Selector-free implementation review and fresh linked public-Op qualification passed. The subsequent selector-free C1 confirmation retained all 257 public tokens and measured `28.9947`, `29.0153`, and `28.9945 tok/s`, a `29.0015 tok/s` mean. Its median decode time was `0.9788658` of the retained control and every run stayed within `1.01` of the admitted candidate median. Evidence is `profiles/bench/r9700-projected-residual-t1-production-confirmation-retry1-20260920/attempt-1`; its `result.sha256` digest is `f629374d5dba788ba93837ee06b74eab6be1650f95d05772694af848d37c9d5a`. Independent audit reported `SHIP`, closing this exact promotion. The result does not establish physical memory-bandwidth saturation or stall freedom.

The next normalized-linear direct screen fused the cooperative K5120 RMSNorm and exact A8G64 preparation while preserving the explicit BF16 normalization seam and unchanged native-IU4 N34816/K5120 consumer. Its complete public boundary improved `0.2406990→0.2347595 ms`; every one of three address-distinct allocations won, the paired-ratio mean plus two standard errors was `0.9790384`, and the 64-MLP-boundary estimate saved `0.3801284 ms/token`. The independent FP64 normalization-to-BF16-to-exact-A8-to-signed-Q4 oracle passed with maximum relative L2 `0.00187542`, zero BF16-step error, exact public-control parity, and captured graph recovery. Evidence is `profiles/bench/r9700-normalized-linear-t1-qualification-20260920/attempt-1`; the `result.sha256` digest is `01e1e0571d92541311195202df919d3750759f3bcedf7033944f604bcf2baf2c`. Independent audit admitted matched whole C1 A/B preparation; this direct result alone does not establish a whole-decode gain or bandwidth saturation.

The matched ordinary Device Graph C1/P8192+G256 whole A/B then retained all 257 public tokens in six balanced runs. Candidate/control decode-time ratios were `0.9895615`, `0.9896042`, and `0.9897497`; median `0.9896042` and mean plus two standard errors `0.9897524` passed the admission gate. Mean decode improved `29.0050→29.3087 tok/s`, about `1.05%`. Evidence is `profiles/bench/r9700-normalized-linear-t1-whole-ab-20260920/attempt-1`; its `result.sha256` digest is `2484b71a86314ec8bf4b116fe0b5caecb707d77747666a41dec5f2ee95ee32ee`. Independent audit admitted promotion review only for all-Q4/A8 ordinary base Text C1 MLP gate/up boundaries. Selector-free implementation review then passed, and fresh linked public-Op qualification at `profiles/bench/r9700-normalized-linear-t1-promoted-qualification-20260920/attempt-1` passed with `0.3647995 ms/token` direct saving. The selector-free production confirmation retained all 257 tokens and measured `29.3174`, `29.3157`, and `29.3116 tok/s` (mean `29.3149 tok/s`). Its median decode-time ratio to the retained control was `0.9893966`; every run was faster than the admitted candidate median. Evidence is `profiles/bench/r9700-normalized-linear-t1-production-confirmation-20260920/attempt-1`; its `result.sha256` digest is `f28d5a0e89bc8cfaa438d7c7ef86405c5eb486c650838c57bf086d9b40c2aebb`. Independent confirmation result audit reported `SHIP`, closing this scoped promotion. These timings do not establish physical bandwidth saturation or stall freedom.

The explicit next-group prefetch challenger for T1 N34816/K5120 gate/up was numerically and statically qualified, then rejected on complete normalized-linear boundary timing. On the R9700 in `auto`, three disjoint weight allocations and 24 balanced graph timing pairs measured `0.2361795→0.2369395 ms` (control→candidate), or `-0.0486398 ms/token` weighted saving over 64 calls. Every allocation was slower; paired ratio mean plus two standard errors was `1.0094922`. This fails the `0.2 ms/token` gate and does not justify a whole C1 A/B.

The independent FP64 oracle passed with zero BF16-step error and maximum relative L2 `0.001801662`; codec and public control parity were exact. Embedded native-IU4 ISA confirmed four successor B64 loads ahead of current compute, at 26 VGPR, 36 SGPR, occupancy 16 and zero LDS/scratch/spills. Correct instruction overlap did not produce a useful gain at the complete boundary. Independent review accepted the rejection; production routing was never changed, and temporary qualification ownership is removed. Sealed evidence is `profiles/bench/r9700-gate-up-prefetch-qualification-20260920/attempt-1`; `result.sha256` digest is `5ea1c61d7d02fdeb971c151ded538872cf46121c31d299686bf0b42390b68e15`, closure SHA-256 is `8e08b7a134601a6dddb5c16c5a74b7b19ed6f635bf435f50ee50b742c75ac4db`. Its scripts are retained provenance, not rerunnable commands.

A GDN projection/control heterogeneous grid has independently reviewed CPU/static feasibility (`SHIP`), with the complete contract and next qualification gate in the retired 2026-09 ledger (git history). The `112x256` grid assigns CTAs `0..63` to paired Q4 and `64..111` to control heads `0..47`; CTA-uniform branches confine the nine control barriers. Emitted gfx1201 code preserves native IU4 and the incumbent control reduction/BF16 seams, using 34 VGPR, 52 SGPR, 2048 bytes LDS, compiler occupancy 16, and zero scratch/spills.

Four vector B128 activation loads replace two scalar B256 loads, leaving an unresolved issue-cost risk. These static results establish neither numerical correctness nor performance. Retained Q4-pair/control timings are `0.102400/0.020920 ms` per layer; overlapping controls with the pair has an ideal `0.896 ms/token` saving after an estimated `0.002259 ms/layer` extra streaming cost across 48 layers. This is an unmeasured bound; the admission margin needs `4.167 us/layer` complete-boundary saving. A complete control+quantize+pair numerical and timing package must receive independent review before GPU work; no new physical command is admitted. Finish this bounded decision, then switch to recipe-independent DFlash optimization before another base-decode mechanism; normalization fusion is deferred. Final DFlash recipe/quality binding still follows numerical-accuracy/artifact selection.

The existing MTP shortlist head remains Q4G64 with A8G64 activations. MTP stays in exact-output, state, cache, row-view, and whole-route regression coverage, but a new shortlist-head trace, alternate head precision, acceptance campaign, or MTP performance optimization is not a final admission requirement. DFlash/DFlash2 is the preferred speculative path and the only speculative backend with remaining support and performance work. After final base/artifact dependencies, the C1 DFlash optimization target is at least 60 decode-output tok/s in matched whole inference. Admission first requires a material win over the current production base with exact public greedy-token parity; the target does not relax either gate. Historical MTP diagnostics can retain the theoretical `64*C` round minimum because their counter sums request lanes, but schema-v7 terminal tooling does not consume or validate those rows. Extra rounds remain diagnostic acceptance evidence; they do not trigger an MTP head-precision branch or alter base selection. Schema v7 carries no downstream-readiness status; selected-profile NIAH and DFlash admission remain separate final gates. The fail-closed NIAH entry point is `profiles/bench/post-terminal-niah-prepare-20260905/prepare.sh`; it remains blocked until the schema-v7 terminal selection exists and then schedules only the required 64K five-position ladder.

The `[131072,5120]` head payload is exactly 356,515,840 bytes (340 MiB) in Q4G64, 713,031,680 bytes (680 MiB) in W8G32, and 1,342,177,280 bytes (1,280 MiB) in BF16. W8 therefore adds 340 MiB and BF16 adds 940 MiB over Q4. At the complete Text+MTP cache costs of 28,288 bytes per token for G16 and 27,200 for G32, those deltas correspond before page/allocator rounding to about 12,603/13,107 aggregate cached tokens for W8 and 34,844/36,238 for BF16; divided evenly at C4, about 3,151/3,277 and 8,711/9,059 tokens per lane, respectively. The retained Q4/A8 shape sweep measured 2.337/2.265 ms at T=1/3, while the older W8 public-Linear bring-up observed 3.153/9.428 ms. These results use different qualifier harnesses and are only a reason to keep the comparison conditional; they are not a same-base direct A/B or a precision-selection result. No source-BF16 MTP acceptance authority currently exists: the checkpoint-direct BF16 scorer treats MTP as a non-executed comparison label, and the Python artifact reference consumes the artifact's encoded shortlist head.

This pass resolves the former known-zero cache-counter result, but it does not establish absolute bandwidth or final-route cache efficiency. The gfx1201 request-size buckets remain known-zero and base absolute counts may undercount, so no physical GL2/HBM byte rate is derived. The stable profile also pinned substantially lower clocks; its 49.43 tok/s benchmark timing is profiler control only, not production evidence. Ordinary performance timing stays under `auto`, and the same focused PMC pass must be repeated on the selected chunk/profile. ROCm Compute Profiler 3.8.0 still has no gfx1200/gfx1201 analysis configuration, and complete VALU/LDS/stall counters remain unavailable.
