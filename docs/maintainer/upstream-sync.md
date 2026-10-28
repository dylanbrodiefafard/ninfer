# Upstream synchronization record

Upstream is the RTX 5090/CUDA `dylanbrodiefafard/ninfer` repository’s `experimental` branch. It is reconciled into this R9700 repository by custom AMD ports, not merged ancestry; `AGENTS.md` states the sync rule.

## Baseline

Reconciled through upstream `5c2bd4b39a3cc645e400e842ca2b92968f920aec` (2026-10-27), with the 2026-10-28 AMD reconciliation below. Upstream rewrote the history that contained the former `7142d4280123d9df507f863c1ae3d16f50e6476e` baseline. This reconciliation maps the former dispositions onto the fetched history and advances the source baseline by behavior, without an ancestry merge. The next sync reviews upstream changes after `5c2bd4b3` against current AMD behavior.

## Rewritten-history reconciliation and cache failure fixes (2026-10-28)

The previous baseline's three source changes map to `05c7729d` (target-selected content cache identity), `2e66a610` (startup defaults and pending timeout), and `24545385` (actual disk capacity and idle defragmentation). Their existing AMD dispositions and verification remain authoritative below. Earlier rewritten counterparts include `cdb36597` (reasoning-close framing), `5798b00f` (preparation/disk telemetry), `f4878ff7` (grammar compilation), `06ee3376` (CUDA graph-update issue already removed by native immutable graph ownership), `ec2272e3` (excluded NVFP4 tail handling), `899e6b5d` (disposable cache writes), and `97147234`/`2bf403e5` (constraints and the excluded CUDA MTP fixture). This mapping preserves those decisions; it does not import NVIDIA execution or fabricate merged history.

Every commit reachable after `24545385` through the fetched `5c2bd4b3` tip is dispositioned here:

| Source | AMD disposition |
|---|---|
| `43d27d9f` | Mixed prefill/verify execution, context checkpoints, copy ownership and serving behavior already have native implementations (`68668568`, `d4886f7a` and subsequent native fixes). Exclude CUDA dependent launches, NVFP4/A4/GDN kernel tuning, CUDA private state routes, other targets and NVIDIA benchmarks. Retain the qualified gfx1201 implementations and their independent oracles. |
| `a5bc5a14` | Exclude CUDA NVFP4 LinearAdd fixture correction; no native consumer. |
| `b00a5c1d` | Native mixed-forward width and cadence controls are already delivered. Retain R9700's measured automatic width and startup-fixed cadence rather than importing the RTX 5090 4096-column default, CUDA tuning, alternate artifacts or NVIDIA performance claims. Its common serving/cache contracts have native equivalents. |
| `2bdb2ce8`, `fe16bedf`, `fce1d470` | Already ported and qualified in `7bcc26a6`: tokenizer heap merges, finite LayerNorm overflow handling and exact schema numeric source values. |
| `4af3c228` | Upstream sync documentation only; the native record remains authoritative. |
| `8b3bea53`, `bfee77bc` | Same remaining-work scheduling change carried on merged branches. Exclude the 4K tail cutoff and short-request prefill interruption: the current AMD contract has one prefill owner with startup-fixed slices/cadence. Preserve the native bounded FIFO scheduler, mixed-round ownership, and supported MTP behavior. CUDA-specific kernels and NVIDIA measurements are excluded. |
| `fedd87a4` | Already selectively ported (recorded below), including native raw-tool attribution. |
| `64f8f73e` | Exclude CUDA retained GDN-history test indexing; native raw-record/deferred-Fold state has its own oracle and qualifier. |
| `93ab32e6` | CUDA real-model Vision checkpoint fixture expectation; native checkpoints use the actual crossed chunk frontier and existing native Vision/context qualification. No production change. |
| `dcdab4f6` | CUDA cached-prefill interruption fixture draft expectation. The associated prefill-interruption policy is excluded above. |
| `106662c2` | Exclude CUDA W4A4 scale-plane readback correction; no native format consumer. |
| `c3bb9dfc` | Merge of the already dispositioned source branches; no separate behavior. |
| `cffce137` | C1..8 already delivered by native `be5848de` and subsequent width/oracle qualification. Preserve native kernels and performance evidence. |
| `fed58c48` | Upstream port of native `44411e91` system-prefix retention; already equivalent. |
| `c3dd2ddf` | Upstream port of native `58927804` RAM-backed windows/reasoning budgets; already equivalent, retaining the AMD cache layout and state ownership. |
| `686e69d4` | Port RAM-index revalidation after live-credit funding, including failed funding. Reselect and fund the actual winning image at the same window before claiming it. Host regressions cover total/longest eviction, larger replacement images, failed reservation and writable/disposable reads. |
| `5c2bd4b3` | Port optional capture/reclaim allocation containment, unfunded credit metadata failures, transactional window grants, retryable suspension/resume metadata failures, allocation-free suspended-window page floors, and fatal-worker availability published before tier cleanup. Add public Engine availability and health HTTP 200/503 behavior. Preserve HIP/GTT startup backing, AMD deferred GDN folds, checked failure fences and fixed three-plane cache semantics. |

The incident was an unclaimed RAM winner evicted by live-credit funding before its claim. At 2026-10-28 07:41:57 UTC, eight requests failed with `RAM cache entry id is unknown`; the worker stopped while unconditional HTTP health kept Docker healthy. Both missing upstream fixes are included. No weight, cache representation, kernel arithmetic or scheduler policy changes are part of this reconciliation.

Validation uses the R9700/gfx1201 Release build, the selected ROCm 10 toolchain and explicit production `qwen3.8-27b-r9700-lq4.ninfer` artifact. Host regressions cover stale-source reselection and failed-worker health/admission during held cleanup. GPU checks cover exact three-plane RAM ownership, metadata-fault credit allocation/relocation, source-preserving disk reclaim failures, optional capture bookkeeping failure, and cross-lane live continuation after capture/restore/window metadata faults. The live qualifier compares complete target token IDs, raw logprob bits, publication and reasoning policy against its uninterrupted control; immediate pool-count assertions protect failed state transitions. Suspension map-insertion failure is source-reviewed; its capture-metadata failure is explicitly injected without faulting HIP's internal allocator. All 64 host tests pass, including the new source-reselection and availability regressions. The RAM and disk GPU suites pass. All 18 live-allocation cases pass across ordinary, DFlash and MTP execution (three reasoning boundaries, greedy and seeded sampling); the DFlash recovery-KV qualifier also passes its optional-capture fault and committed state/retry checks. The warnings-as-errors build, formatting/lint hooks, whitespace checks and changed-line clang-tidy over 34 translation units pass with zero diagnostics. Astra reviewed the upstream dispositions, native implementation and regression coverage with no remaining findings. Evidence belongs under `profiles/server-diagnosis-20261028/`.

## Selective published-stop logprob port (2026-10-19)

Ported `fedd87a4ab81584b2d603004a2c3a85ba584d50c`: a stop token whose bytes are
published receives its output-channel attribution and therefore its token-logprob record.
Raw output covers both a terminating model stop and a model-stop token emitted with model
defaults disabled. Normal serving still withholds stop-token text and its logprob; this
change does not expose the terminal EOS score in ordinary HTTP responses. The native
Qwen3 frontend retains the existing reasoning/tool transitions and recovery policy.
The native port also attributes raw tool markup to content, including a published stop
inside an open call; raw output publishes these bytes without structured-tool filtering.

Verification: warnings-as-errors incremental Release build; three focused frontend and
recovery tests; affected frontend/test translation units pass clang-tidy with zero
diagnostics. The open/closed raw-tool regressions fail against the upstream-only port
and pass with the native extension. Astra reviewed the final source; formatting and
source whitespace checks pass. Evidence: `profiles/agent-quality/20261019/stop-logprob-*`.
No GPU arithmetic, sampler default or running container image is changed by this port.

The fetched upstream `experimental` tip was `c3bb9dfc`; its history does not
contain the recorded `7142d428` object. This is a selective source port, not a full sync:
the recorded reconciliation baseline is unchanged. A future full sync must first map the
previous dispositions onto the fetched history before advancing it.

## `edaad03a..7142d428` (3 upstream commits, reconciled 2026-10-19)

| Source | AMD disposition |
|---|---|
| `f899cba810fdd74599559bfc8078c508b11f07d7` | Port target-selected SHA-256 tensor-content cache identity; exclude file metadata, Text output/proposal heads and DFlash proposal-only paths. Include DFlash context K/V row slices and remove cyclic lane capacity from disk identity. Add native row-range hashing for row-scaled FP8, row-split integer, N16 Q4/W8 and tiled LQ4 planes. Preserve qualified native MTP and fingerprint its complete state-producing block; upstream's MTP disablement and NVIDIA attention-profile selectors are excluded. |
| `35c0d04d6b1baeb1a35a14d0759f13f3e43a74c1` | Port the five-minute pending-request timeout. Automatic KV sizing, DFlash K7/adaptive optimized proposal head and temperature 1.5 are already the native CLI/server defaults. Preserve explicit draft/head/adaptive choices and supported MTP; upstream's NVFP4-specific startup selector, other targets, RTX 5090 measurements and documentation are excluded. |
| `7142d4280123d9df507f863c1ae3d16f50e6476e` | Port actual pack-file capacity accounting, coalesced-hole reuse, stable pack store with independently checkpointed object maps, Put/Delete/Move replay, paced idle relocation and tail trimming. Preserve AMD fixed-codec payloads, pinned GTT buffers and ordinary-RAM direct-read staging. Integrate Engine idle permission with both generation and synchronous scoring. Expose storage bytes and defrag activity in metrics and request-log schema v25; update its maintained Python readers. |

Disk format is now v8; earlier directories require a fresh location. The stopped deployment's
incompatible v7 cache was identified for user-authorized cleanup on 2026-10-19. An initial
cleanup attempt was blocked; the deployment cache directory was subsequently observed empty
before the requested server launch. Profiling evidence remains. Deployment uses a 12 GiB RAM
cache setting. After storage recovered, the rebuilt container started healthy and a LiteLLM
request routed to the AMD deployment on port 8080, returned `OK`, and used no fallback. Native MTP
execution remains supported. The separate
Docker RAM-cache setting requested with this work is 12288 MiB (12 GiB).

Qualification uses the R9700/gfx1201 Release build and existing ROCm 10 toolchain. Focused host
checks cover storage-aware fingerprint mutations and copied artifacts, scheduler idle handoff
for generation/scoring, request logging, metrics and defaults (seven tests passed). The GPU cache suite checks exact
raw/Zstd restart bytes, FIFO capacity churn, admission during cancelled relocation, reader extent
leases and crash snapshots around durable Move publication. Python request-log reader checks
cover both serving-corpus and NIAH tools (27 tests passed). A real production-artifact C1 greedy
SSD restart restored 38 prefix tokens and produced the exact 16-token cold continuation. Local evidence is retained under
`profiles/upstream-sync/20261018/`. The changed-line static analysis covered 150 translation units; all seven findings were repaired
and the three affected translation units rechecked with zero diagnostics. No kernel or
inference-throughput change is claimed.

## Secondary-source review and selective ports (2026-10-18)

Reviewed all 204 commits in Neroued/ninfer’s `experimental` range
`e31bc99b13f517c8aae70b997b7c4a49b4dcdc5d..05327b22c1596ddca12635d8e04aea278f800724`
against current AMD behavior. This is a selective source review, not merged ancestry or a
replacement for the Dylan baseline above. Commit-by-commit local dispositions are retained in
`profiles/upstream-review/commit-dispositions.csv`.

| Source | AMD disposition |
|---|---|
| `81c8ce093b2c` (source precision) | Reject schema assertion numbers whose decimal source is changed by JSON parsing; preserve exact integer literals and represented-value identity above 2^53. Check raw HTTP bodies, CLI messages, and Engine schema strings; report source pointers. Broader numeric language expansion is deferred. |
| `91d568a6e065` (LayerNorm) | Retain ordinary FP32 arithmetic; scale and center finite BF16 extreme inputs when the original moments or variance-plus-epsilon overflow. Independent FP64 oracle at Vision D=1152 and the epsilon-overflow edge. |
| `75a890502e0c` (deferred cancellation) | Investigated, no blanket synchronization port: fixed startup-owned buffers and same-stream reuse already preserve lifetime. Added a pending-compute cancellation/reuse regression with exact seeded cold-lane and resident-peer controls at C6/K7/max-context 262144. |
| `a8e212aca` (tokenizer) | Ported per-encode scratch and adjacency/generation-checked heap merges, retaining the scalar short-word route. Independent official-tokenizer goldens cover ties, Unicode, and long words. |
| `0a8530d44141` (bounded-string masks) | Ported counted-repeat/dense mask compilation. Flattened complete JSON scalar/escape units so the fast path actually applies. Independent byte-language and scalar-prefix oracles cover all token bits and rollback. |
| `2e098d5bb` (sampling) | Native challenger rejected: greedy improved 1.84–2.77%, but default p-less slowed 0.3–0.9%, with B6/B7 paired intervals excluding zero. Existing vectorized argmax and sampler retained; independent eager/graph regression cases retained. |
| `4d6316eea8ef` (convolution) | Ported carried-window standalone tiled causal convolution. Independent FP64 convolution and exact final history, including aliasing, pass. Matched Release ABBA shows approximately 3–10% improvement at 128–8192 rows, with 64-row noise. Default LQ4 prefill already uses a different fused rolling route; no default Engine throughput claim. |
| `bc70d7c872df` (tool streaming) | Ported transactional provisional tool events through Engine, CLI, Chat, Responses, and Anthropic. Ambiguous delimiters, constrained arguments, cancellation, truncation, rejected recovery attempts, and multiple calls in one token preserve the publication contract. Responses completed events wait for accepted terminal success. |
| `2734a56e0` and dependencies (schemas/diagnostics) | Ported Unicode string bounds/pattern intersections, supported allOf/local-ref siblings, finite exact-one oneOf and provably disjoint alternatives, source-aware admission errors, compile-cache observations, committed constraint state, mask work/transfer metrics. Unsupported proof forms reject explicitly. |
| `8ffa31317` (choice/regex) | Ported Engine and CLI literal choices/full-match regex plus the Chat structured_outputs extension, sharing the existing constrained sampler. |
| `4126ecf40` (media) | Resolve CLI/example/reference media paths relative to the containing messages file. Preserve filesystem symlink/.. traversal semantics. |

The CPU grammar dependency is now one vendored `ninfer_token_grammar` target rather than a
network fetch plus six build-time patches. `third_party/xgrammar/README.ninfer.md` records the
source baseline, selected mask changes, licenses, and retained native guarantees. Canonical
Qwen framing, required/unique named arguments in any order, strict Unicode/escapes, raw tokenizer
bytes, and the existing bounded-number language remain supported.

Other reviewed correctness candidates required no missing AMD patch: production KV publication
callers already pass explicit streams (`064965c74563`); DFlash chunk bindings are refreshed
(`4201b5d2d`); native claims/copy holds preserve cache sources (`911d34db6`), and the faulty
upstream prefix-forest scoring algorithm is absent from bounded FIFO scheduling (`452c3f5fa`).
The Jinja trimming and Q8 D=2048 gather fixes have no corresponding native consumer.

Optional/deferred: combined tool-or-constrained-answer branches, broader bounded binary64 and
fractional integer bounds, tuple intersections, TOML, startup graph calibration changes,
Vision/KV-transfer/direct-host-loading experiments, and NVIDIA-specific kernels. Additional
checkpoints and hardware backends remain excluded by the product contract. These items are not
silently admitted by this port.

### Qualification and scope

Qualification uses the R9700/gfx1201, ROCm 10 / Clang 23, Release build, the fixed
FP8-K/INT4-V/FP16-scale cache, and the explicit production `qwen3.8-27b-r9700-lq4.ninfer`
artifact. No weights, clocks, power policy, production sampler, or cache format changed.
Independent reviewers covered numeric/state lifetime, schema language/source attribution, and
tokenizer/tool publication; actionable findings were repaired and rechecked.

Final Release build and all 62 host tests passed; the same 62 host tests passed with ASan/UBSan,
including the instrumented vendored grammar library. Static analysis reported zero diagnostics
across 147 affected translation units and the two subsequently corrected test files; pinned
formatting/lint/file gates and changed-document reference checks passed.

Schema composition includes 1,995 independent accept/reject comparisons
from an independent Draft 2020-12 validator with ECMAScript Unicode pattern semantics, plus
an explicit unsupported-proof admission case; tokenizer qualification compares 96 cases with
the official tokenizer and retains 28 goldens. Live HTTP checks passed with speculation
disabled, MTP, and default DFlash across Chat, Responses, and Anthropic, including incremental tools, constraints,
truncation, concurrent requests, disconnects, and readmission. The CLI literal-choice smoke
returned the required exact choice. LayerNorm, convolution/history, and sampling use independent
mathematical oracles; deferred cancellation uses the real production state geometry. Device
memcheck and initcheck passed all six selected eager/GDN/sampling executions; focused
racecheck passed all three qualifiers. The latter checks instrumented cross-wave LDS accesses,
not arbitrary global-memory races.

Tokenizer matched CPU ABBA on the Ryzen 9 7950X3D measured 1.05x ordinary prompts, 1.27x
identifier-heavy input, and 86x an 8192-byte word. Counted masks with the real Qwen vocabulary
reduced the measured initial mask from 1.1–235 ms to 10–17 us, while cold schema compilation
increased from 5.5–6.9 ms to 21–25 ms. Near-boundary masks improved; the already tiny completed
mask increased from 0.64 to 1.27 us. These are measured CPU/operator scopes, not Engine TTFT or
whole-inference speedups. The convolution window removes overlapping input reloads; the
selected tile-4 gfx1201 route uses 18 VGPRs, reports occupancy 16, and has no LDS, private, or
scratch allocation. Its independent oracle and public-route qualification passed. No unresolved
attribution question required a profiler capture to select it. Raw commands, workloads, paired
results, and limitations remain local
under `profiles/upstream-review/`.

### Remaining performance candidates (2026-10-18)

All R3/P5/P6/P7/P8 candidates were screened against R9700Lq4 and the viable routes
were implemented. P1–P4 retain their previously qualified dispositions, including rejection
of the slower P3 sampler. No weights, cache formats, clocks, power settings, or supported
features were changed.

| Area | Final selection and evidence |
|---|---|
| R3 startup graph allowance | Admit the measured C6/chunk1024/full-context/DFlash adaptive-K7 profile under the existing four-setting bulk-argument environment. Observed/reserved graph memory is 62/62 MiB instead of 62/1,998 MiB; auto-KV rises from 537,408 to 613,696 tokens (+14.2%). The near-capacity runtime retained a 261,824-token prefix plus five 69,632-token prefixes, then generated 256 tokens on every lane, including 211 actual K7 lane-rounds. Mixed execution, six ready decode lanes, 34 minimum free KV pages, and at least 80.24 MiB physical free VRAM passed. The 64 MiB headroom and observed-allocation guard remain. Canonical chunk2048 already reserves 72 MiB at C1 and 62 MiB at C6/C8; this adds no demonstrated default-profile capacity or throughput gain. |
| P5 Vision RoPE | Select aligned four-BF16 vectors only for paired Vision D72/H16 at T>=256; smaller, single-tensor and misaligned routes remain. Independent represented-input FP64 tests cover real interleaved QKV, 255/256/257 boundaries, T1536/8256, untouched V and graph replay. Physical captured public-Op ABBA confirms about 2x at T1536. ISA has 64-bit vector movement, 35 VGPR, 288 bytes LDS, zero scratch and reported occupancy 16. |
| P5 feature scatter | Select aligned eight-BF16 vectors for D5120/T>=128; smaller/misaligned/other-width inputs keep the scalar route. Independent whole-image exact-bit tests include NaN encodings, untouched columns, tails, alignment and graph replay. At 384 rows, captured public-Op latency falls about 29–33%. ISA has 128-bit movement, 11 VGPR and zero LDS/scratch. |
| P5 LayerNorm | Reject the general anchored CTA: FP64 and extreme-value checks passed, but it was about 89% slower at 1536 rows and also lost at 2048. A bounded tiny-image challenger saved only about 0.4 ms per full HTTP request and changed one fixture's greedy token. No LayerNorm change is promoted. |
| P6 host KV transfers | Long contiguous HeadMajor runs use one pitched transfer per head; short/fragmented ordering and exact plane-major RAM images remain. Nine independent 16-layer exact-image cases cover crossover, partial restore, offset and untouched storage; a five-page failed-restore case covers the async fence. Final plain/memcheck/initcheck pass. At 860 pages, public D2H improves 255.4 to 223.8 ms (1.14x) and H2D 296.4 to 227.5 ms (1.30x). Opt-in RAM tiers benefit; resident D2D already coalesces. |
| P7 materialization | Ordinary RAM remains the sole direct-I/O destination. Host ranges copy straight into mapped storage; host-only chunks skip pinned staging, whose allocation is planned across every ring-slot occupant. Existing event/read-reuse fences remain. A >576 MiB exact-byte fixture crosses the eight-slot ring in both placement directions; plain/memcheck/initcheck and all 675,430,400 production embedding bytes pass. Greedy IDs are unchanged with Device Graph enabled. This removes 640 MiB of redundant copies; production peak staging remains 1 GiB. Two ABBA startup groups are noisy, so no stable startup percentage or inference-speed gain is claimed. |
| P8 short Vision attention | Four waves/64 queries serve multiple uniform segments of length 4..64. Singleton, longer and nonuniform routes retain eight waves/128 queries, including unchanged descriptor/workspace ownership. All 47 independent FP64 cases and 38 focused device-check cases pass; racecheck sees 7,392 workgroups and 0 hazards. Matched public-Op graph ABBAABBA with sharp interleaved inputs gives18–23% lower latency at 16 segments. Complete output bits match the previous route in 22 performance cells. Four-wave ISA uses 126 VGPR, 12,032 bytes LDS, zero scratch and reported occupancy 10. |
| P8 remaining NVIDIA tuning | All 63 source entries have explicit dispositions. Existing AMD partitioning, fused projections, and Q4 WMMA/LDS staging cover transferable mechanisms. The faster BF16 GEMM fallback experiment is rejected as off-production because Vision matrices are Q4G64. CUDA instructions, formats and 2560-wide target constants do not justify another backend or arbitrary kernel campaign. |

Combined qualification includes the warnings-as-errors build, 62 host tests, 62 ASan/UBSan
host tests, focused clang-tidy, independent reviews, and owning numerical/exact and device
checks. Whole HTTP ABBA uses cache-free greedy one-token requests with five warmups and
30 measured requests per process, fixed 2048 context/KV, prefill 128, Vision enabled and
Device Graph disabled. Final chart TTFT medians were 301.35 to 298.97 ms; short-segment video
was 181.87 to 179.96 ms (16 segments of 16 patches). Outputs matched in every process. These small request gains do not
imply a default text-throughput improvement.

The existing whole-Vision diagnostic needed two repairs to run on production: exact
Vision/frontend admission for the LQ4 identity and n16k16 weight residency/dequantization.
Its tiny artifact regression checks signed code times stored scale exactly. The production
schedule passes all 471 artifact-reference comparisons. The diagnostic's older local Linear
and BF16-source criteria are not all green: the same 109 local Linear failures, five weight
criteria failures and three artifact/BF16-source failures occur on the baseline. There are
no new failure records; the baseline's three production/artifact failures disappear.
Thresholds were unchanged. The owning
changed Ops use independent represented-input oracles directly.

All GPU work holds the shared card lock, startup tests wait for asynchronous VRAM
reclamation, and timings are unprofiled. Screening and per-source dispositions remain in
`profiles/deferred-perf-20261018/`; implementation evidence, final reports and qualification
are in `profiles/production-perf-20261018/`. Losing challengers remain only in ignored
experiment files; there are no production forcing switches or alternate cache/weight paths.

## Ported features (`c450798c..e04fad37`)

- Sampling/API: full-domain p-less and epsilon sampling with the 1/1024 probability floor, suppression, masks and root-only cycle exclusion, configured argmax eligibility, DFlash p-less one-hot proposal, per-row position offsets; ordinary and speculative. HIP uses wave32/hipCUB with device-scope release/acquire completion counters and caller-owned workspace.
- Host runtime: reasoning/tool generation and recovery, chain/adaptive DFlash, Vision integration, scheduling and telemetry.
- Storage/startup: durable SSD prefix cache, RAM recovery/ownership, parallel loading and pinned initialization.
- Product/tooling: CLI/server contracts, xgrammar, Docker/build/test/evaluation tools.
- Earlier bounded correctness batch: prelaunch cancellation (`c075cc38` with `64840a95`'s postlaunch semantics), speculative tail invalidation (`0412f177`), committed-prefix statistics (`b2b1b7e4`), RAM readiness/ownership (`64840a95`).

## Kernel dispositions

- DFlash selector per-sequence projection: ported (numerical isolation; packed outputs equal separate C1 calls).
- DFlash selector parallel merge (sixteen block-wide rank reductions): applicable optimization, not ported; the AMD serial exact merge keeps the same ordering, and no R9700 benefit is shown.
- GDN replay/state overlays: already covered by the AMD raw-record replay and FP32 fold.
- GDN control/projection specializations (packed T5/T6, T to 20, FP8/projected convolution): AMD leaves retained; their workspace and precision contracts differ.
- Chunked GDN FP16 W/U staging: not ported; AMD owns an independently qualified chunked route.
- Attention/projection (NVFP4/U8 sparse prefill, FP8/NVFP4 linear routes): not applicable to the fixed FP8-K/INT4-V cache and the AMD weight formats.

CUDA paths, NVFP4 and additional model targets are excluded by the product contract.

## `e04fad37..34c7119b` (47 upstream commits, reconciled 2026-09-28)

- Ported: `5d366aeb` sampling-count rollback ordering, `909ff2d7` text-part tool results, `565161ad` Anthropic thinking signature, `610615cd` swscale alignment, `82fb7c5f` generated UTF-8 repair, `1c0b8f5f` tool-schema keyword relaxation, `0fe9b7ff` + `8f0082d3` literal client text encoding, `dfc818ae` unread disk scratch and grammar heap trim (its tokenizer blob packing is not ported: the AMD tokenizer tables differ and the saving is host memory only), `5b5d3caa` worker and disk-thread device binding, disk tier `cb6ec221`, `8c232d01`, `931617b2`, recovery on resident KV `d29841e0` + `550560fe` with merge `92388a91`'s literal-span adaptation (the executor probe test binds a probe device instead of a HIP device; the Program-level real test is `ninfer_r9700_recovery_kv_qual`; the CUDA test-only stream-order fix in `test_state_store.cpp` and the real prefix/disk/DFlash-cancel test edits have no AMD counterpart) as `059eb9f9`, `9f020ed7` HIP graph-update diagnostics (`91fc4b45`), `5d6f6bf2` DFlash2 p-less draft temperature (`1461d0ae` + `44f14b09`; R9700 default 0.4 measured +9.9% p-less T1.5 decode), `a7cbe5c9` Prometheus `/metrics` (`a13ddefd`; runtime sparse-attention knobs dropped, CUDA Graph names become Device Graph), `67753776` RAM KV spill on admission (`d0e58c0a`, plus exclusion of an attempt's own captures from reclaim targets, which upstream lacks).
- Ported with R9700 measurement: `724de290` causal-conv crossover, here T>=32 rather than T>8 (`bfbec72c`; the AMD serial/tiled crossover lies between 16 and 32 at C=10240); `f3c15618` headroom half (`4b4588f9`, `--kv-capacity-headroom`, default 64 MiB).
- Ported with an R9700 prompt route: `f3c15618` host-resident token embedding (`TensorPlacement::MappedHost`, `mapped_host_bytes` in the load summary, CLI and `server_start`). A direct port had kernels read every row from pinned memory; in the Q4G64 N16K16 tile layout a row touches one 8-byte word per 128-byte line, and that 16x PCIe read amplification cost 1.0% C1 8K prefill (3385 -> 3351 tok/s) with decode unchanged. The port keeps the in-place gather only for generated tokens (device ids in captured Device Graphs: decode, verify, drafter, MTP steps). Prompt rows are host-staged: the host gathers a window's distinct rows into a compact image of the same format (`ops::stage_embedding_rows`), the load stream copies it into a fixed persistent region, and the existing gather kernel reads it there; the next chunk's window is staged and copied while the current chunk runs, so only a prompt's first chunk stages synchronously. The materializer fills the pinned backing from the direct-I/O slots rather than from the page cache. Measured with the card on a chipset PCIe x4 link: greedy tokens identical (DFlash, MTP, 8K multi-chunk, vision); C1 8K prefill at chunk 2048 3389 -> 3385 tok/s (noise), at chunk 4096 3257 -> 3243 (-0.4%, the synchronous first chunk is half the prompt); decode unchanged; C4 DFlash auto capacity 525632 -> 550528 tokens.
- Not ported: `45bef20a` (report instead of fail on the device-wide graph memory delta). R9700 GPU jobs are serialized by the shared GPU lock, and the startup check is the only calibration guard for the graph allowance here. `482274b9` (tool-grammar mask overlap with DFlash2 C4/C6 verify on a second stream): this ROCm serializes concurrent graph branches (`docs/performance.md`), so the overlap cannot occur; its research documents are NVIDIA-specific.
- Ported with an R9700 copy path: `3e18ef63` (rewrite checkpoints in lane-owned pinned host memory; its graph-allowance change is not ported, the R9700-calibrated constants apply). A first direct port added 18-27 ms to checkpoint-capturing and 31-61 ms to checkpoint-restoring requests' TTFT: its GDN images moved with `hipMemcpy2DAsync`, whose ROCm host rect path pins the host range per call and copies with a shader or line-staged transfers. The port copies one contiguous range per layer and component, captures through the staging slot (device snapshot, then D2H on `copy_stream` behind later work), and restores from staging on device while it still holds the lane's image. Measured with the card on a chipset PCIe Gen4 x4 link, versus the pre-port build: checkpoint-capturing TTFT 149 -> 112-121 ms (DFlash) and 131 -> 105-111 ms (MTP), staging-hit restores unchanged, staging-miss restores +7-8 ms; C4 DFlash auto capacity 496192 -> 525632 tokens.
- Equivalent: `b71eebf3` (R9700-calibrated DFlash graph allowance), `c36f38c4` (host round preparation during the commit tail), `c7bc4cbe` (8 build jobs; `9e0795ac`), `34c7119b` (adaptive DFlash K locked at the cheapest k on warm requests; fixed by `a7880a27`, which credits unseen deeper hops with the deepest observed acceptance, measured 1.074x vs adaptive K5).
- Excluded (NVIDIA, NVFP4/TMA/tensor-core, or C > 4): `c62f7ead`, `c1da30a8`, `fda2972d`, `fccf6613`, `b83885e3`, `79033d70`, `4ca53273`, `6ab11dfa`, `43b49f9e`, `9b8fcce1`, `e800379e`, `41f331af`, `724dd314`, `91e41952`, `7485d137` (NVFP4 T6 decode attention); upstream docs `7e303491`, `c40469f3`, `f74915f9`.

Evidence: `profiles/bench/r9700-upstream-sync-20260928/`.

## `34c7119b..9639c32f` (14 upstream commits, reconciled 2026-09-30)

- Ported: `42d7fa75` multi-key NIAH (`1a6eb119`: 32K/240K/260K lengths, the 260K single-key fixtures, `run_niah_check --multikey`; all ten 260K fixtures regenerate byte-exact), `8aa4ce07` preserve-thinking replay at the generation opener (`b9dd0450`), `5342e7a9` p-less block verification (`d1f56ed8`; FP64 block oracle, greedy tokens unchanged, p-less accepted length within noise on R9700: C1 3.62 -> 3.57 tokens/round, C4 3.51 -> 3.51).
- Ported as an idea: `9639c32f` short-append attention (`2fa61af7`). Its decode-chunked small appends match the existing 1..8-row packed decode and 9..127-row mid-row routes; its grid.z key split for wider calls becomes the mid-row split for 128..1023 rows under a whole-wave cost model (128 rows at 131K: 9.56 -> 4.09 ms per layer). 1024-row calls fill whole waves and stay dense.
- Already equivalent: `c9a04602` (`d0e58c0a`, `23877a2e`), `db2b7153` (`455e45ef`), `1e4dc811` (`3c4b4ce6`), `9b7d1277` (`2a8b033f`), `37c63461` (the host tool-mask mailbox already runs plain rounds at the no-exchange ceiling), and the `94d97c11` DFlash2 draft-temperature contract.
- Excluded: `34ed7bc6` (NVFP4 A8 gate/up scratch), the `94d97c11` rows for the NVFP4-KV `--sage`, `--keep-frac` and `--xattn-tau` flags (XAttention exists here only as the compile-time qualification build with fixed tau, not a runtime flag; Sparge is not implemented; Sage is rejected by the precision policy), and the NVIDIA measurement records `f33ff218`, `163c6b38`, `ceeac05d`.

## `9639c32f..f7f70d89` (4 upstream commits, reconciled 2026-10-02)

- Ported: `ca08451c` `/metrics` buckets, labels and closed-label series storage (`cc613dc3`; keeps the HIP/Device Graph identity, `ninfer_device_memory_bytes`, the `/v1/score` route and the AMD API code set; `reservation_exceeded` stays absent because the R9700 executor never publishes it). `a1d289ec` closed preserve-off turns stored cut at their turn checkpoint, per-reason disk drop counters under the mutex, and the shared splice checkpoint-prefix fallback (`db1abce1`). The cut image's current-state copies use the copy-stream host callback behind the image fence instead of upstream's host wait and scheduler-thread memcpy. R9700 C1 serve, 4 two-turn chats of ~2.7K tokens: RAM 427 -> 286 MiB per entry, turn 2 appends with the resident checkpoint's reuse length and identical greedy tokens (DFlash k4, MTP k3, DFlash K7 adaptive with thinking, restored from RAM and from disk). The splice fallback is unreachable here (a trailing assistant turn is tokenized cold for its scoring boundary) and is kept as the one shared helper.
- Already equivalent from `fd16c8ba`: BF16 verify projections and the W8/Q4 heads at k=6/7 (every verify projection and head is one aggregate launch over all W*C columns up to 64), and the full-width drafter-context append (the AMD path has always appended at the storage width).
- Not ported after R9700 A/B (`docs/performance.md`, 2026-10-02): the `fd16c8ba` learned per-law/per-k hop-hazard picker with its 1-in-32 exploration, and the p-less draft-temperature scale at k=6/7. Adaptive K7 already matches the per-prompt best fixed K (K7 everywhere) at C1 and C4; exploration cost 0.7% at C1 against an intermittent K6 lock (one lifetime in six), and the scaled temperatures were within noise of 0.4. The unused DFlash `append` envelope that upstream widened to the verify width is removed here instead.
- Excluded: the `fd16c8ba` CUDA/NVFP4 kernel changes (W8 dynamic shared memory, NVFP4 W4A8 M48/K256, NVFP4 GDN record, GQA T=7/8 split rules). No action: `f7f70d89` relaxes an upstream real test the AMD tree does not carry (the cached-versus-fresh interleavings resume check).

Evidence: `profiles/bench/r9700-upstream-sync-20261002/`.

## `f7f70d89..574a8d91` (31 upstream commits, reconciled 2026-10-02)

Upstream's code-quality campaign (compiler-warning, clang-tidy, formatter, and lint gates, then compute-sanitizer) plus the races those tools found. The quality work reached `experimental` from the `quality/clang-tidy-backlog` branch, so `4051ebf0..26ba4203` are not descendants of `f7f70d89`; every non-merge commit in `f7f70d89..574a8d91` is dispositioned here.

- Ported races: `2e11477f` p-less reduction helpers (trailing barriers after every shared-slot result read, block-uniform inverse-CDF pick without the `red_idx[0]` round trip, register `scan_done` in the tile scan) and the accept-kernel half of `8dba51cc` (no `done_sh` read beside thread 0's write). The fork's helpers were the pre-fix upstream code; the new R9700 racecheck reported the same sites before the port (`sampling_device.h` slot-0 reuse, moments, residual, top-2 broadcast, and the compiler-hoisted `*found` load) and none after.
- Ported defects from the lint census `a9c82edd`: disk-reuse planning moved from a `const std::optional` (a copy of the restored host image per plan), and the eval CLI's missing `PackageNotFoundError` import. Its other behavioral hunks are absent here or already equivalent (q5 aliasing, the paged-KV scatter `__shared__` descriptor, the CRC32C `[[nodiscard]]` order, w8 pair planning, `tools/kdev`).
- Ported tooling with AMD equivalents (`docs/maintainer/code-quality.md`):
  - `4051ebf0`, `be07884e`, `f2e67b06`: `cmake/warnings.cmake`, warnings as errors in project C++
    (GCC 13) and HIP (ROCm clang) in place of the nvcc front-end flags; the HIP host pass does not
    report device-only helpers as unused. `NINFER_SANITIZE` instruments the host compilation of
    HIP translation units only (`-Xarch_host`), because gfx1201 device code cannot take ASan.
    The fork's warning census (97 project findings: shadowing, unhandled FP8-capped profile
    enumerators, dead fields and locals, sign compares, unchecked HIP event calls) is fixed.
  - `4051ebf0`, `fe50f67d`, `edde8e88`, `26ba4203`: `.clang-tidy` and `scripts/run-clang-tidy.py`
    on the ROCm toolchain's clang-tidy, reading the HIP compile database directly (no nvcc
    translation or builder re-exec, so `a7238ed3` has no counterpart); `bugprone-signed-bitwise`,
    new in LLVM 23 after upstream's 22.1.8 pin, is disabled.
  - `4051ebf0`, `ce01f249`: `.pre-commit-config.yaml` (clang-format selects `.hip` by extension),
    `ruff.toml`, `_typos.toml` with gfx12/ROCm vocabulary, and the auto-fixing `.githooks/pre-commit`.
  - `f8ea8cf9`, `574a8d91`: compute-sanitizer has no gfx1201 counterpart (ROCm ASan needs
    `xnack+`). `tools/r9700/gpu_check` and `scripts/gpu-check.sh` provide memcheck (guard pages),
    initcheck (poisoned allocations), and racecheck (LDS instrumentation of optimized device
    bitcode through the HIP compiler launcher) over the `gpucheck` CTest set (all 31 R9700
    qualifiers). Racecheck also found a fork-only race: `nll_from_logits_kernel` (the prefill
    NLL behind PPL and scoring) read the block maximum from `partial[0]` while wave 0 could
    already be storing its sum partial there; a barrier now separates them. After the fixes the
    set passes under memcheck, initcheck, and racecheck, and plainly (47 R9700 tests, 57 host).
  - `33cf6b17`, `07d638b4`: the builder's memory cap at three quarters of host RAM without swap,
    `--ulimit core=0`, and the update-then-restart rule.
  - `6c64f009`: the dead `.codex` clang-format hook is removed.
- Ported after the sync on 2026-10-02: `ce6ad2d5` tree-wide clang-format/ruff reformat with its `2695ffbe` blame entry (byte-pinned candidate qualifiers excluded), and `157161de` whole-tree clang-tidy as the gate: the 694 findings here are fixed or suppressed at the line, and the commit hook is no longer opt-in. Defects found on the way: `kv_ram_cache_qual` swallowed a waiter exception, and `bench-matrix` tests carried a never-executed assertion with an undefined name and the FP8 hybrid converter a stray expression.
- Not ported: the clang-tidy backlog fixes `9ff23c4c`, `d40226e0`, `196cc237`, `9b9b9b47`, `341822ba`, `1bd8f518` (style or hardening of upstream code; their defect hunks are covered above), `ce01f249`'s `--fast` tier and `kernel` label (the fork's test runner is host-only by default). Its master-merge process was subsequently adapted on 2026-10-09 in `merging-to-master.md` and `scripts/master-gates.sh`, and `f8ea8cf9`'s `--sanitizer` reduced cases (the R9700 qualifiers complete under racecheck at full scope). Absent here: `c1a37a5a` sparse-MoE scan race, `d5d01c3c` q5 epilogue and kdev guard, the `8dba51cc` small-T reduce kernel (the split-KV merges here use separate scalars and barriers), and the `test_attn_input_proj` sanitizer cases (`ninfer_r9700_target_variant_attention_projection_qual` is an empty entry point).

## `574a8d91..593c2d0c` (4 upstream commits, reconciled 2026-10-03)

- Ported: `91371f42` block-cooperative p-less tile choice. `sampling_block_choose_tile` (wave32 `__shfl_up` chunk scan, lowest owning thread, last nonempty tile for a goal past every interval) serves the ordinary p-less sampler and the speculative finalize kernel, whose residual masses and admitted sum are now block-parallel; the per-thread auxiliary-mass buffer is gone. Sampling and speculative-round qualifiers and racecheck pass. R9700 C1 DFlash K7 `--lm-head-draft`, seeded p-less T1.5, 512 tokens, 3 prompts x 2 seeds x 2 ABBA passes: decode 91.4 -> 102.1 tok/s mean, median round 32.15 -> 29.41 ms, identical round counts in every pair.
- Already equivalent: `91371f42`'s BF16 residual and DFlash conv verify aggregation at every width (`packed_route_tokens` makes every verify projection one aggregate launch). Excluded: its CUDA bench and SmallT kernel removal.
- Not applicable: `7eb0edcc` keeps a DFlash forced-K test off the 1-in-32 exploration rounds of the hop-hazard picker, which the fork declined (above).
- Ported with fork deltas: `4bf04efd` KV-tier rework, upstream's port of the fork's stall fixes (`7885114c`, `5187be80`, `14b7ac49`) plus: RAM blocks retire on fence and last I/O pin, so a RAM claim never waits for a spill of the same entry; a RAM restore waits for its entry's own capture copies in copy-hold; chunked (4 MiB) host-copy callbacks on a tier-owned host-copy stream; unlocked CRC, encode, compaction publication, reap, and tombstone I/O; two-phase eviction; serialized MANIFEST publication; claims that cancel an idle Extend/Refresh instead of waiting, with a stale generation becoming a `CacheRestoreFailure` cache miss; request-local restore failures parked like cancelled copy-holds; per-lane cached prefix hashes; and the startup-pinned checkpoint-image slab (`ckpt-pin=` / `ckpt-heads=`), which removes serving-time pinned allocation. Kept fork code: `PagedKVPublication`, the FP8-K/INT4-V RAM/disk fingerprints and intra-page order, mixed rounds and `decode_waiting`, `score_many`, `poll_idle`, the deferred GDN fold, the closed-turn cut store, and `synchronize_all_while_unwinding`. The disk-cache suite takes upstream's gate-held stall cases in place of timing windows, and the sleep-based payload stall hook is removed.
- `593c2d0c` records a 5090 A/B; the R9700 serve check is in `docs/performance.md`.

## `593c2d0c..48eb0b01` (4 upstream commits, reconciled 2026-10-03)

- Ported: `24d527d1` online p-less draft-temperature calibration with `d2b40485`'s round predicate. The accept Op's optional proposal-calibration output is a wave32 kernel after the multi-block accept pipeline (qualified against an FP64 oracle in `ninfer_r9700_speculative_round_qual`); `--dflash-p-less-draft-temperature` becomes an optional pin, the 27B prior is `clamp(0.8(T-1), 0.2, 1.25)` (0.4 at T=1.5, the previous R9700 default), and tool-mask batches bind the round's ingress configs. Fork delta: the graph-capture context carries the calibration predicate, so captured DFlash graphs contain the scoring kernel exactly when replays read it back; upstream captures without it, so its graph replays read calibration egress that no kernel wrote. The per-k temperature scale it removes was never ported here. R9700 production-shape A/B: acceptance +3.3%, decode +2.4% (`docs/performance.md`).
- Already equivalent: `48eb0b01` producer-side A8 activations. The R9700 LQ4 target already encodes projection inputs in their producers (`fp8_quantize_normalized`, `fp8_quantize_gated_rmsnorm`, `fp8_quantize_gated`), and at C1 the persistent decode kernel hosts the remaining quantize phases; MLP-down keeps a standalone quantizer in both trees. Its NVFP4 kernels are excluded.
- Not ported (pending a product decision): `055b333c` best-first packed-tree arm, learned arm picker, GDN tree-schedule record, and W<=16 verify aggregation. Upstream measured +10.3% C1 greedy but no gain under p-less with the retuned proposal temperature, the production sampler here; the port needs the per-k hazard picker the fork declined (2026-10-02), a HIP tree-select and tree-record kernel, and tree verify in the persistent C1 decode kernel. Its p-less default 0.8 is superseded by `24d527d1`, and the selector-lattice dump probe is offline tree-research tooling.

## `48eb0b01..945515ea` (3 upstream commits, reconciled 2026-10-04)

- `aa3252fadaa6c07cdd779634d295abb321634a55`: ported planned-state selection to RAM and disk restore. Turn/response checkpoint reuse skips current frontier GDN, hidden and DFlash cyclic state; requests dropping an unused rewrite checkpoint skip its host image and clear its metadata. Disk also skips the corresponding state decode jobs. Existing AMD context-head filtering, per-image fences and retained-block lifetime remain authoritative. The expanded RAM/disk qualifier checks exact kept and untouched bytes for all four reuse dispositions; frontier-plus-rewrite decodes six state images, checkpoint-only or frontier/drop decodes three.
- `442099cb99479bb62a4ee9d9960ceae92269a232`: ported selective DFlash context K/V projection as a passive 2048-row binder view of the packed 6144-row QKV weight, with native N2048/K5120 A8/Q4 routes, and a fused BF16 local K RMSNorm/RoPE/cache append. Existing AMD query normalization/RoPE was already fused. MTP selective K/V is already equivalent through its binder row views; the new MTP dual-RMS optimization is explicitly not ported: the current performance authority retains MTP for regression and directs new speculative optimization to DFlash, and upstream reports flat MTP Engine performance. CUDA/NVFP4 kernels, NVIDIA bench executables and exploratory experiment diaries are excluded; their applicable numerical/cache and timing checks are represented by the native context qualifier.
- `945515eafdbccdf3f80f19bce40c70f1ccf2a282`: explicitly not ported after native evaluation. AMD already fuses finish and residual. A native parallel finish/residual/RMSNorm candidate passed the independent FP64 oracle and memory/race checks and saved the 64-column Op edge, but changed late greedy tokens at C8. Disabling only that fusion restored exact token parity. Preserving the existing RMS reduction inside the fused kernel restored parity but slowed the 64-column reset-plus-edge from about 43 to 130 us. Neither candidate is admitted; the existing draft normalization schedule and arena ownership remain. CUDA launchers, NVIDIA benches and experiment diaries are excluded. The upstream three-to-one launch claim does not apply to AMD's existing two-kernel edge.

Verification and R9700 timing scope are recorded in `docs/performance.md`; upstream's 5090 numbers are not AMD evidence. This reconciliation changes no ancestry and creates no merge.

## `945515ea..f0d3922e` (1 upstream commit, reconciled 2026-10-04)

Source commit: `f0d3922e8a4dfedf57cd7011bc0eb72e0e1c8236` (`perf: optimize nvfp4 prefill and dflash gdn commits`). All changes are explicitly dispositioned; this sync changes no runtime.

- Excluded: the signed G16 A4 activation representation, RMSNorm/sigmoid/SwiGLU A4 producers, NVFP4 attention-input and residual consumers, NVFP4 tiled SwiGLU changes, and their target dispatch/workspace changes. These are CUDA/NVFP4 routes outside the fixed R9700 LQ4 weight and activation profiles. AMD's qualified producer-side A8 handoffs remain authoritative.
- Excluded: 64-row short Exact attention prefill tiles. The changed launcher dispatch and kernel operate on upstream's U8 NVFP4 cache, with CUDA dynamic shared memory and NVIDIA matrix instructions; they do not implement the AMD FP8-K/INT4-V/FP16-scale cache. Existing native short-append routes remain authoritative; the upstream tile threshold is not an AMD tuning result.
- Applicable optimization, not adopted: DFlash capacity-four accepted GDN history. Upstream retains FP32 normalized keys, decay and parent-derived innovations beside a dense checkpoint, appends short accepted paths, and materializes when retained plus accepted length is at least four. R9700 retains its qualified raw-record Fold and fused deferred-chain schedule. The upstream route requires history-aware verification and materialization/reset at dense consumers, snapshots, restores, retention and prefill. Its derived records must come from the same invocation's actual parent and commit an ancestor-complete path, unlike the raw-record Fold interface. A native replacement must qualify those ownership/provenance boundaries and AMD arithmetic, including persistent and mixed verification, before promotion. No such qualified gfx1201 replacement or R9700 performance evidence is established by this sync. Sequential history reconstruction can preserve the incumbent FP32 transition; this disposition claims neither numerical incompatibility nor an AMD slowdown.
- Excluded with the corresponding routes: A4/history layout and Op tests, the history-specific short-terminal resident/RAM continuation test, and updated CUDA attention fixtures. Upstream performance numbers, phase-seven research records and documentation describe those NVIDIA implementations; they are not imported as AMD behavior or evidence. The native raw-record and cache recovery qualifications remain unchanged.

Verification: accounted for all 66 changed files and checked the source commit's ancestry from `945515ea`; two independent subagent reviews checked the GDN arithmetic, provenance and state lifecycle disposition. Only this record and `AGENTS.md` changed, their baseline identifiers agree, and `git diff --check` passes. No build or GPU experiment is needed for this documentation-only reconciliation; no new correctness or performance claim is made. Future syncs review changes after `f0d3922e` against the retained AMD behavior, including the explicit non-adoption above.

## `f0d3922e..6e5b576e` (2 upstream commits, reconciled 2026-10-06)

- `63179e946f54b1f19a16fdfef961b5acc310987d`: ported token logprobs through a native gfx1201 Op, Engine request plans and round frames, committed output-channel records, exact token bytes, and OpenAI Chat Completions/Responses JSON and SSE. Each published token is scored against its target logits at temperature 1 over the token domain; sampling masks, penalties and draft distributions do not participate. Ranked alternatives use lower-id tie breaking. Disabled rows return before reading logits and leave their records untouched. Native integration preserves compact/adaptive panels, mixed prefill/decode and paged-KV transaction ownership. The shifted log-normalizer avoids FP32 cancellation for large equal logits; this additional represented-input case passes the independent FP64 oracle. Native qualification replaces the CUDA Op fixtures and benchmark. NVIDIA throughput claims and benches are excluded; the AMD test runner already routes qualifiers through CTest labels. Real packed-tree inference fixtures are excluded because this target requires chain width K+1; explicit column selection remains qualified at the Op boundary.
- `6e5b576e15d21de5470c3728d9fcc3cdc4e59f2a`: the capture defect was already fixed in AMD (`593c2d0c..48eb0b01` above). Ported the tied-prediction choice nearest the prior in log temperature, startup NaN egress poisoning/validation, and the last p-less chain draft temperature in SpeculativeStats, CLI and request log. The scoring launch is consistently the Program's startup property in capture and eager execution; tree bodies still omit it. Upstream CUDA performance results are excluded. Native captured calibration reports 0.8 at target p-less T=2 in the real-artifact check.

Verification: Release build with warnings as errors; native logprob FP64/exact oracle at 248320 physical rows / 248077 tokens, signed-zero/equal-logit ties, large equal logits, strided and skipped outputs, explicit columns, widths through 16 and batches through 8; real r9700-lq4 Engine checks for ordinary, MTP, DFlash chain/adaptive/eager/calibrated Device Graphs, C8 reporting/quiet rows, streaming channel parity and reporting-on/off unchanged greedy tokens. Teacher-forced ordinary decode agrees within 0.000004 nats; the prefill comparison changes geometry by including its target token and is checked behaviorally separately. Speculative and batched routes use behavioral NLL criteria for their qualified activation profiles, with greedy tokens additionally equal to rank zero of their own logits. Nine affected host tests pass; changed-line clang-tidy and final Op/test checks report no remaining diagnostics, and git diff --check passes. Evidence: `profiles/bench/r9700-upstream-sync-20261006/`. No throughput benefit is claimed, no unsupported inference route is enabled, and no ancestry merge, commit or deployment was created.

## `6e5b576e..d5c38cda` (2 upstream commits, reconciled 2026-10-09)

- `c7199023365cca8a4ca277e27e289219f12f49bc`: ported unified owning output constraints (JSON object, JSON Schema, XGrammar EBNF), CLI file options, Chat response_format and structured_outputs.grammar, and Responses text.format. Replaces AMD's schema-only option; required/named tool-call behavior remains supported. Response assertions are enforced and unsupported assertions fail admission; tools retain their documented relaxation policy. Native HIP TokenMaskExchange keeps the coherent mailbox/matcher thread, caller-owned planner storage, ordinary prefill roots and speculative chain columns. NVIDIA callback, CUDA graph and profiler paths are not imported. Renamed grammar/mask ownership to token grammar consistently. Ported the pinned build-copy compiler patches for JSON assertions, Unicode scalar lengths/escapes and canonical additional keys. Content preserves EBNF whitespace and literal special bytes, reasoning remains unconstrained, and caller stops can truncate a valid prefix with the existing finish reason. The FFmpeg current-buffer release was already equivalent on AMD. CUDA/NVFP4-only fixture identities are replaced by the production R9700 real-artifact C3 check; unsupported packed-tree inference remains excluded.
- `d5c38cda2e875aad37f49889c1f03eda1540bf20`: NVIDIA MTP RAM-reseed fixture assertion correction, explicitly excluded. The modified NVFP4 test and its seeded-k=4 assertion do not exist in AMD; no production behavior changes in this commit. AMD's adaptive MTP seed policy and native checks remain unchanged.

Verification: warnings-as-errors Release build; affected grammar/frontend, CLI, OpenAI/Responses, Anthropic, recovery, metrics and public-interface host checks. The real production artifact passes mixed constrained/unconstrained C3 generation with ordinary decode, MTP and captured DFlash chain, including exact EBNF, schema values, EOS, cached compilation, streaming/result and token-logprob attribution. Changed-line clang-tidy and git diff --check pass. Evidence: `profiles/bench/r9700-upstream-sync-20261009/`. No performance improvement is claimed; no commit, ancestry merge or deployment is created.

## `d5c38cda..17c504c9` (1 upstream commit, reconciled 2026-10-12)

- `17c504c98f263f29a29b1c0115925f02b9941a22`: ported disposable cache writes through `ExecutionOptions.cache_write` and the common OpenAI/Responses/Anthropic `ninfer` object. False reads preserve the original through an independent VRAM fork or a complete-original RAM migration, restore RAM without consuming it, and retain disposable continuation state ahead of ordinary eviction victims. Future-reuse captures and shutdown persistence reject disposable state; a later writable continuation promotes its resulting state. Checkpoint capture conflicts fail before enqueue/streaming. Source-invalid, stale-plan and destination restore failures have distinct source-preserving cleanup. Native ownership retains the bounded FIFO/head protection, request-local telemetry helpers and copy hold. Paged payload copies honor each of the FP8-K/INT4-V/FP16-scale planes' own physical order; native MTP publication and deferred GDN folds replace the upstream private state profile. Startup defines complete paged/cyclic payloads and fixed egress padding without changing valid frontiers. Transfer-test fences are held after DMA submission. Request-log schema 23 and every active consumer replace schema 22 consistently (upstream's 18-to-19 numbering belongs to its separate log contract).

Verification: warnings-as-errors Release build; eight affected host tests, including recovery restore/fork/stale-plan failure disposition and three serving schemas; 27 Python report-consumer checks; native exact-byte paged-copy/RAM tests with mixed plane orders, noncontiguous physical runs, independent destination mutation, restore accounting and later source consumption; disk-cache tests; and the production R9700 real Engine fixture at C1/C2 with ordinary, MTP and DFlash current/checkpoint forks, single-token MTP, original preservation, capacity rejection, disposable promotion, concurrent original/fork requests, disk cancellation and shutdown. The native page/RAM qualifier passes memcheck; recovery executor passes ASan/UBSan; changed-line clang-tidy and the final qualifier check report zero diagnostics. Upstream CUDA/NVFP4 test identities are replaced by native qualifiers and the explicit production artifact; CUDA stream helper and benchmark-only fixture changes are excluded. No performance claim, upstream ancestry merge, master promotion or deployment is part of this sync. Evidence: `profiles/bench/r9700-upstream-sync-20261012/`.


## `17c504c9..62687bd3` (4 upstream commits, reconciled 2026-10-16)

- `9edf680bac3598dcd857903a576c681b5bcbbac3`: excluded CUDA/NVFP4 prefill-tail fix and associated CUDA fixtures. AMD's native FP8-K/INT4-V attention already writes semantic zero for inactive query rows, including partial tiles; it does not use the affected NVFP4 kernel.
- `6c7e2ccee630dd39a5b00123c371a64587420662`: ported the pinned build-copy grammar patches for shared property rules, direct subset/reference transitions, one normalization after nested schema assembly, and exact tokenizer byte-prefix metadata. Completed-call validators use an independent cached byte-only compiler, with one joined cold-build worker under the frontend's compiler lock. Vocabulary-mask compilation uses sixteen host threads. Native tests retain AMD's canonical additional-key, assertion, Unicode, special-byte and speculative-state checks, and add finite-language/mask oracles, recursive schemas, cache isolation and concurrent cold preparation. The NVIDIA workload's throughput and memory figures are not AMD evidence.
- `a736c1c3cef8a55e74636287f3ed29d1181f88b2`: already resolved by AMD `eb63b0f8`, pulled before this reconciliation. Every captured profile owns an immutable executable and startup instantiates/uploads it once; the graph-update API and runtime update path are removed. Consequently workspace-address changes never update multidimensional memcpy nodes. The upstream raw-copy kernel, graph-update tests and CUDA timing are not imported into this different graph ownership model. Existing HIP pitched copies remain.
- `62687bd37f1dfb629384630ed8c07b5cc3ef9da1`: ported pre-preparation events and full synchronous preparation wall time for Chat, Responses and Anthropic; disk spill/compaction/ restore activity, pending-job count and active-session age through nonblocking snapshots, RuntimeStats, periodic logs, JSONL and both metrics encodings. Completed spill lifetime is labeled `spill_completed`, separately from request blocking time. AMD's host-to-device disk transfer time and device-page fields remain. The local request-log schema advances 23 to 24 in every active consumer; upstream's 19-to-20 identity belongs to its separate contract. Held-spill, log lifecycle/flush, timing and metrics regressions are adapted to native tests.

Verification: warnings-as-errors Release build; 11 focused host tests covering exact grammar languages/masks, frontend, recovery executor, public headers, three serving schemas, request logs and metrics; 27 Python corpus/NIAH report-consumer checks with Python 3.11; changed-line clang-tidy reports zero diagnostics. The real Qwen tokenizer probe with eight tools passes 1,736,525 mask-versus-byte-acceptance comparisons, including framing boundaries, unordered arguments and speculative rollback (read's `path` is named `filePath` in the probe fixture to match its declared test input). Cold compilation took 217 ms on this host; no A/B speedup or inference-throughput claim is made. Evidence: `profiles/bench/r9700-upstream-sync-20261015/`.

GPU verification completed on 2026-10-16 under the shared lock: the production R9700 C3 constrained-generation fixture passes ordinary decode, MTP and captured DFlash, including streaming/logprobs and cached preparation. The native disk-cache suite passes after adapting the telemetry test to the actual publication boundary: held preparation reports active work with zero session age, the installed session reports positive age, and completed spills leave idle work flags with lifetime counters preserved. The corrected test also passes clang-tidy with zero diagnostics. Final results are in `gpu.log` (real-model result) and `disk-recheck.log` (corrected disk-cache suite); the initial disk test's session-age assertion was a test-port error and required no production change. Source whitespace checks pass; unified-patch context lines retain their required leading space. The baseline advances to `62687bd3`; all four source commits are dispositioned above.

Before push, this sync was rebased onto AMD `69e2178c`; the combined warnings-as-errors Release build, 11 focused host tests and production-artifact constrained-generation GPU test pass (`rebase-build.log`, `rebase-host.log`, `rebase-gpu.log`). This reconciliation creates no upstream ancestry merge, master promotion or deployment.


## `62687bd3..edaad03a` (2 upstream commits, reconciled 2026-10-17)

- `81a12c69e09f91c6ca844b6f82575e903bbbafc4`: excluded exact NVFP4-to-FP8 CUDA codec expansion, eight-row A8 Down staging, NVFP4 T>=2 policy documentation, and their CUDA codec/LinearAdd tests and benchmarks. These do not implement the R9700 LQ4 format or native HIP projection routes. Query/single-chain-key prefetch changes only upstream's accepted-history GDN recorder, whose state representation was explicitly not adopted in `f0d3922e`; AMD retains its qualified raw-record/deferred-Fold path. Its history tests and replay benchmark follow that excluded route. No native speed benefit is established by these NVIDIA measurements. The kernel-iteration/tooling change admits bounded experiments addressing profiler-attributed instruction, address or synchronization overhead despite an unchanged byte roof: already equivalent to the native Layer 0 hypothesis and Layer 2 physical qualification rules. The SM120 classifier/recipe, CUDA admission constants, performance records and NVFP4 implementation documentation are excluded.
- `edaad03a737aa1a9e8288c47af3f07ada50ae905`: ported single structural reasoning-close framing through the Qwen3 frontend and native concurrent executor. Plain chat excludes the dedicated closing token once reasoning ends, including no-thinking admission and recovery re-prefill. Speculative rounds commit only through that transition before continuation samples with updated exclusions; ordinary/MTP/DFlash state ownership remains native. Tool grammars reserve closing-think markup in free prose, including byte-split forms. Schema-constrained argument values, response constraints, ordinary-token literal text and raw output retain their contents. Host regressions cover preview rollback, channel attribution, prefill/decode exclusion updates and speculative-prefix commit; the native C3 real-artifact fixture checks thinking/no-thinking and streamed logprobs with every supported backend. The spill preparation versus installed-session age assertions were already correct on AMD from the previous sync; ported the bounded retry for a temporarily unavailable nonblocking snapshot. Serving, concurrency and test documentation describe the resulting behavior.

Verification: warnings-as-errors Release build with the selected ROCm 10/gfx1201 toolchain; 11 focused host tests covering frontend, token/schema grammars, tokenizer, recovery/executor, public API, CLI and OpenAI/Responses/Anthropic schemas; native disk-cache suite; production `qwen3.8-27b/r9700-lq4` C3 constrained/plain generation with ordinary decode, MTP and captured DFlash, thinking/no-thinking, exact closing-token counts and streaming/logprob attribution. Both GPU checks passed under the shared lock (11.03 s disk cache, 32.53 s real Engine fixture). Changed-line clang-tidy over 40 translation units reports zero diagnostics; changed C++ files pass clang-format and source whitespace checks pass. The executor regression uses AMD's probe device and native prefill signature. Evidence: `profiles/bench/r9700-upstream-sync-20261017/`. No throughput claim, ancestry merge, master promotion or deployment is part of this sync. The baseline advances to `edaad03a`; both source commits are dispositioned above.
