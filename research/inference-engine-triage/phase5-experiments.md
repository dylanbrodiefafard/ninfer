# First repository port experiments

This experiment report evaluates the first three Phase4 mechanisms on
`experimental` at `593c2d0c95ce2e34ef137af70c94d0417b37a890`. Target:
Qwen3.8-27B/NVFP4, RTX5090 `sm_120a`, CUDA13.1, startup-fixed B1–6,
default NVFP4 target KV and BF16 local DFlash2 KV. Repository qualification is
in `phase4-qualification.md`; papers and the remaining experiments are outside
the original experiment goal. Publication was authorized separately after
qualification, as recorded below.

The scoped source questions are `dflash-context-kv-subset`, `dual-offset-rms`,
and `dflash-key-rope-append`. The Phase4 reference retains each contributing
repository, inspected development head and source receipt; starting mechanisms
include the context-row selection in `akumaburn/vllm-dflash2`, independent panel
normalization in `andyluo7/sglang`, and accepted K materialization in
`hukongyi/sglang`. These are native implementations of the qualified mechanisms,
not foreign builds or runtime weight repacking. Grouping all layer projections
and query-only norm/RoPE were not assigned to this finite experiment.

## Finite experiment ledger

| Experiment | Owner | Baseline/admission | Correctness | Measured outcome | Production decision |
|---|---|---|---|---|---|
| Selected context K/V weight rows | `phase5_kv_projection` | cold public Linear 20.480us at T3; kdev weight_replay: GO | full original NVFP4/W8/Q4 descriptors; independent decoded-weight FP64 dots, exact C1 packed parity, capture and guards pass | cold B1/W3 graph 20.480 → 14.304us; five distinct layers B6/W3 163.520 → 77.824us | retain; remove context full-QKV workspace |
| Independent dual MTP offset RMS | `phase5_dual_rms` | two-launch captured baseline measured; one-launch experiment admitted | independent FP64 panels, both alignment routes, broad gains, eager/replays pass; existing RMS/gated oracle suites pass | T3 graph 6.336 → 4.320us; T36 6.368 → 4.352us | retain in 5120-wide MTP stem |
| Plain K norm/RoPE/accepted cyclic append | `phase5_normalized_append` | three-launch baseline measured; composition admitted | independent FP64 norm/rotation and exact cache/state checks pass, including replacement/replay | final B6/W8 graph 6.432 → 4.416us; ring replacement 14.368 → 7.872us | retain one launch with native normalization reduction and BF16 rounding inside registers |

Each experiment finishes with a measured implementation decision, a substantiated
admission refusal, or a concrete execution prerequisite. A negative experiment
is a useful outcome; losing candidate code is removed. First qualify each closed
Op against its complete independent represented-input formula and exact state
oracle, then time the candidate and current composition. Private staging parity
is supplementary. Interpret measured Op/phase savings separately from Engine
throughput; no static byte reduction establishes a speedup.

## Ownership and execution

Agents own disjoint semantic implementation/test/bench files. The selected-row
agent owns the context projection call; the append agent owns its later norm,
RoPE and cache calls. The RMS agent owns only the MTP stem calls. The coordinator
owns shared CMake wiring, this ledger, all builds, numerical GPU tests and timing.
Candidates wait for a measured baseline and applicable admission gate.

The existing `ninfer-builder` mounts another checkout and is left untouched.
An isolated `ninfer-dylan3-builder` mounts this checkout and its ignored `build/`
directory, the existing model directory read-only and cached xgrammar source
read-only. It reuses the cached builder image. Its limits are 16GiB memory,
zero additional swap, four CPU cores and affinity28–31. Native builds use at most
four jobs and single-threaded nvcc optimization. GPU runs and builds are
serialized. Research Python/Git/network uses the existing capped wrapper.
No model/dependency downloads or foreign builds are needed.

## Required completion checks

- [x] All three baseline/admission decisions recorded.
- [x] All admitted candidates qualified against independent numerical/state oracles.
- [x] Each experiment has a measured outcome; retained production integration is coherent.
- [x] Relevant captured execution and supported real-artifact checks completed, or exact missing prerequisites stated.
- [x] Full C++ unit suite run against the final profile with the isolated builder and capped jobs.
- [x] Losing paths and temporary controls removed; final report and active references reconciled.

## Results and limitations

All three retained implementations reduce latency at their measured public Op
boundaries. Raw evidence is in ignored `profiles/bench/phase5/`. Alternating paired
CUDA-event samples use identical represented inputs, caches and stream. Normal
95% intervals describe sampled mean baseline-minus-new latency, not a guarantee
of Engine throughput. Representative mean savings:

| Boundary / condition | Mean saving, us | Sampled mean normal 95% interval, us |
|---|---:|---:|
| KV B1/W3, cold graph, 100 pairs | 7.1386 | [6.9351, 7.3420] |
| KV five distinct layers B6/W3, cold graph, 100 pairs | 84.9184 | [84.6576, 85.1792] |
| Dual RMS T3, graph, 200 pairs | 2.0608 | [1.9243, 2.1973] |
| Final append B6/W8/all, graph, 300 pairs | 2.1463 | [2.0821, 2.2106] |
| Final append W2048 ring replacement, graph, 300 pairs | 6.5041 | [6.3727, 6.6355] |

KV measurements also cover cold B1/B6 W1..8 and ingest128/512/2048, representative
warm-cache points, eager short appends and a five-distinct-weight layer boundary.
Dual RMS covers all distinct T from B1/2/6 times W1..6. The preliminary append
matrix covers B1/2/6, W3..8, counts zero/one/mixed/all and full ring replacement;
all sampled mean-saving intervals are positive. The final append arithmetic
profile is independently requalified across that correctness matrix, with paired
timing repeated at B6/W8/all and full ring replacement. The candidate grid for
KV preserves original 6144-row physical scale geometry while scheduling only
2048 K/V rows; the NVFP4 A16 route uses BF16 MMA, so the classifier's FP4 compute
ceiling is not an A16 throughput claim.

The first append arithmetic profile kept normalized K in FP32 registers until
the final rotated BF16 store. It passed the independent formula with smaller
pointwise error than the old two-round composition, but changed draft acceptance:
on pp512/tg128, DFlash2 k3 C1 ran56 versus57 rounds and C6 aggregate341 versus338
rounds. Against the repeated baseline, decode changed154.057 →157.194tok/s at C1
and653.355 →648.583tok/s at C6. Those mixed Engine outcomes do not establish a
universal gain. The retained profile uses the native warp-down sum, lane-zero
inverse RMS broadcast and BF16 normalization rounding inside registers, before
rotation. It passes the unchanged independent FP64 criterion and exact state
checks while preserving one launch and eliminating global normalized-K storage.
Its worst required pair tolerance is 0.00758645, within the predeclared 0.008.
Supplementary full-cache comparison still differs in one K word at B6/W8 and
16 at full ring replacement (zero V differences); it is not historical bitwise
parity. No alternate candidate route remains in production.

### Engine scope

RTX5090, CUDA13.1, default NVFP4 target KV, CUDA Graphs, the exact artifact below,
fixed three drafts, benchmark corpus pp512/tg128, one warm-up and three measured
repetitions. DFlash2 final-profile controls run baseline then new at C1, new then
baseline at C6; the MTP controls use the settled dual-RMS implementation. Means
and sample standard deviations are decode output tok/s:

| Mode | C | Baseline mean ± SD | Retained mean ± SD | Change | Rounds / accepted per repetition, old → new |
|---|---:|---:|---:|---:|---|
| DFlash2 | 1 | 154.000 ± 0.002 | 159.790 ± 0.033 | +3.76% | 57/70 → 55/72 |
| DFlash2 | 6 | 651.154 ± 0.583 | 654.173 ± 0.321 | +0.46% | 338/425 → 339/425 |
| MTP, W8 matrices | 1 | 143.611 ± 0.037 | 144.221 ± 0.022 | +0.43% | 56/71 → 56/71 |
| MTP, W8 matrices | 6 | 639.104 ± 0.304 | 639.699 ± 0.395 | +0.09% | 341/423 → 341/423 |

These are narrow corpus outcomes. DFlash acceptance still changes, so C1's gain
includes fewer speculative rounds and is not wholly a kernel-time gain. The C6
and MTP Engine differences are small, and earlier baseline repetitions drifted
by amounts comparable to the MTP gains. The robust retention evidence is the
qualified public-Op savings; no universal Engine throughput improvement or
unchanged generated trajectory is claimed. Real-model throughput here does not
cover long-window replacement or every context, width, and concurrency.

Reproduce the retained Engine route with `ninfer_bench --weights <exact-artifact>
--spec dflash --draft-tokens 3 --concurrency 1 --kv-dtype nvfp4 -pg 512,128
--warmup 1 --repetitions 3 -o json`; use concurrency6 for the aggregate case.
Canonical public-only Op benchmarks and their parameters are in `bench/README.md`.

The exact local artifact is
`/models/qwen3.8-nvfp4-flash2-nvfp4-bf16codebook-from-bf16/qwen3_8_27b_nvfp4_dflash_nvfp4.ninfer`.
Its identity is `qwen3.8-27b/nvfp4` and its DFlash2 matrices are NVFP4 with BF16
selector codebook. Header inspection found W8 MTP matrices in this artifact;
no local MTP-NVFP4 artifact was found. MTP product-route checks therefore use
this supported NVFP4 target with W8 MTP, and cannot establish a throughput
result specific to NVFP4 MTP matrices. No artifact is downloaded or regenerated.

The final full native gate ran inside the isolated builder with
`NINFER_DEV_JOBS=4 bash /src/scripts/run-unit-tests.sh --inner`: zero failures in
119 registered cases, 117 passed and two skipped, total CTest time1089.37s.
The skipped `ninfer_qwen3_6_27b_load_plan_test` and
`ninfer_qwen3_6_35b_a3b_dflash_load_plan_test` require their historical artifact
fixtures. The new Op tests and existing RMS, gated RMS and Linear profile suites
all ran. Compute-sanitizer memcheck passed dual RMS, selected-row NVFP4 projection
and fused append with zero errors. Racecheck passed dual RMS and fused append
with zero errors/warnings. KV memcheck exercises its five actual NVFP4 routes;
append checks include zero/mixed/full prefixes, captured replay and full ring
replacement. The final register-rounding append profile was separately rerun
under memcheck and racecheck, both with zero errors/hazards.

The final-profile focused real DFlash2 test passed k3 chain C1/C6 isolation, adaptive N5 widths,
terminal delivery/queue telemetry, RAM reseed and in-flight restore. It used the
exact artifact above and `NINFER_DFLASH_TEST_ONLY_K=3`, with unrelated Vision,
entitlement and p-less likelihood campaigns omitted. Cross-schedule target-only
greedy diagnostics remain expected; the C6 comparisons use the matching saved
C1 DFlash oracle, and actual target-margin checks ran. Canonical public-only
benchmarks were rebuilt and exercised after deleting comparison controls.

Baseline fixture uploads used the default stream while the tested Ops used a
nonblocking stream. Explicit setup fences fixed the exact-copy failures; timed
regions are unaffected. No production copy change was needed.

The dual RMS fixture exposed a limitation in the existing single-RMS gross-error
criterion: an ideal FP64 value 4.0781 cannot be stored in BF16 within the permitted
0.01477 error, since its closest BF16 value 4.0625 differs by 0.01560. The dual
experiment therefore declares, before candidate qualification, a gross bound
`1e-5 + 0.0040 * max_abs_reference`, derived from the BF16 nearest-rounding bound
`max_abs_reference / 256` plus FP32 reduction margin. Its relative-L2 bound remains
0.00185. The independent FP64 formula and broad unequal input/gain fixtures remain;
the existing single-RMS criterion is unchanged.

For fused norm/RoPE/append, the predeclared pointwise bound is
`0.008 * FP64_normalized_pair_norm + 1e-40`, covering the baseline's two BF16
roundings and FP32 arithmetic without making cancellation a relative-error
failure. Baseline worst required pair-relative tolerance was 0.00758645; exact
V copies, untouched slots, metadata, ring replacement and captured replays passed.

## Completion and cleanup

All three scoped experiments have a retained production decision, independent
qualification and measured evidence. Canonical public-only benchmarks were
rebuilt; the final append benchmark smoke passed after temporary comparison
controls were removed. Active benchmark, test and performance references are
consistent, local documentation links resolve, formatting checks pass and
`git diff --check` passes. No commit or push was made.

After the final gates, 187 owned generated executables were removed, including
the saved baseline and unused full-suite/benchmark binaries, reclaiming 16.504GiB.
The three retained Op tests/benchmarks, focused real test and Engine benchmark
remain available with build objects/configuration and product applications.
No research clone was created. Models and existing source/evidence are preserved.
The isolated builder is stopped after verification; its limits and mounts remain
available for reuse. The other checkout's builder was left untouched.

## Revalidation after upstream pull

The user requested a fresh pull, numerical/performance confirmation and a commit
message draft. `experimental` advanced to `48eb0b01` (four commits), including
producer-side A8 activations, tree/picker changes and p-less calibration. The
merge preserves every upstream A8 API and arithmetic body: the shared CTA RMS
body forwards `RmsOutput`, and dual RMS uses its BF16 specialization. Context
projection/append still consume BF16 panels. Direct independent oracle cases
now also cover W12/B1,B2,B6, including the tree route's 72-column packed shape,
for NVFP4/W8/Q4 projection and fused append. Criteria are unchanged.

- [x] Latest branch pulled; local changes restored and A8 integration reconciled.
- [x] Independent Op oracles and upstream exact A8 producer/consumer checks pass.
- [x] Fresh paired public-Op measurements confirm all three latency savings.
- [x] Focused sanitizer and real chain/tree/state checks pass.
- [x] Full C++ unit suite passes against the updated integration.
- [x] Temporary comparison sources/binaries removed and active results reconciled.

Every GPU run uses the shared host `flock /tmp/ninfer-gpu-queue.lock`; build jobs
remain capped at4, nvcc optimization at1, and the isolated builder at16GiB/four
cores/zero added swap. Other checkouts' queued tests are respected.

RTX5090, CUDA13.1, captured public Ops, 300 alternating pairs, cold256MiB
flush outside the timed interval where indicated:

| Boundary | Baseline → retained median, us | Mean saving, us | Sampled normal 95% interval, us |
|---|---:|---:|---:|
| KV B1/W3, cold | 20.480 → 14.336 | 6.8273 | [6.7112,6.9434] |
| KV five distinct parents B6/W3, cold | 165.888 → 77.824 | 86.7139 | [86.5595,86.8684] |
| Dual RMS T3 | 6.272 → 4.192 | 2.0141 | [1.9262,2.1019] |
| Dual RMS T36 | 6.432 → 4.416 | 2.0253 | [1.9403,2.1103] |
| Append B6/W8/all | 6.464 → 4.192 | 2.3186 | [2.2315,2.4057] |
| Append W2048/full ring | 14.528 → 8.064 | 6.6938 | [6.5871,6.8004] |

The same exact local artifact and default NVFP4 target KV are used for fresh
Engine comparisons, pp512/tg128, one warm-up and five measured repetitions.
Fixed-k3 baselines run before and after the candidate; the baseline column below
averages those two run means, distinguishing drift from candidate effects:

| Mode | C | Bracketed baseline mean → retained mean, tok/s | Change | Rounds / accepted per repetition, old → new |
|---|---:|---:|---:|---|
| DFlash2 k3 | 1 | 155.107 → 161.275 | +3.98% | 57/70 → 55/72 |
| DFlash2 k3 | 6 | 653.775 → 656.377 | +0.40% | 338/425 → 339/425 |
| MTP k3, W8 matrices | 1 | 145.357 → 145.194 | -0.11% | 56/71 → 56/71 |
| MTP k3, W8 matrices | 6 | 641.120 → 641.294 | +0.03% | 341/423 → 341/423 |

The latest adaptive DFlash2 C6 route (`--draft-tokens 7 --adaptive-draft
--lm-head-draft`) measures704.518 →713.959tok/s (+1.34%) with one baseline/candidate
pair. Average rounds change340.0 →336.2; the learned arm choice and acceptance
change, so this is a workload-specific outcome, not pure kernel attribution.
Fixed-k3 C1 also benefits from fewer rounds. MTP throughput differences are tiny
relative to baseline drift and do not establish an Engine gain. Its artifact's
W8 MTP-matrix limitation still applies. All three retention decisions remain
supported by direct numerical qualification and positive public-Op savings.

Current raw evidence: `profiles/bench/phase5-refresh/`, including
`refresh-summary.json` and `paired-ops.csv`. This revalidation supersedes the
original snapshot above for a commit-message benchmark claim.

Focused memcheck covers dual RMS, fused append, A8 activation producers and all
five selected-KV routes; racecheck covers dual RMS, fused append and A8 producers.
All report zero errors, with zero racecheck warnings or hazards. Real-artifact
k3 and k7 checks pass, including C6 chain isolation, k7/W12 C6 tree execution,
long-tree C1/C2 isolation, target-margin checks, adaptive execution and RAM
restore. The upstream state-reconstruction fixture currently exercises a
k4/W5 p-less **chain**, despite its `TREE_STATE_ONLY` selector name; its resumed
tokens match fresh reconstruction exactly. This is separate from the W12 tree
isolation evidence and is not a tree-reconstruction claim.

The full C++ gate (`scripts/run-unit-tests.sh --inner`, four build jobs in the
isolated GPU builder, under the shared host lock) completes in 1103.40 seconds:
121 registered cases, 119 passed, zero failures, two skipped legacy load-plan
cases whose Qwen3.6 artifacts are unavailable. Supported Qwen3.8 real-artifact
checks above ran separately. A8 activation and the three new Op tests pass in
the updated full-suite integration.

After verification, the temporary paired harness and its CMake target/object
directory, saved latest baseline, temporary runner and pre-pull backup were
removed. CMake reconfiguration passes with the measurement target removed.
Removing 187 unused generated executables plus the saved baseline and temporary
files reclaims 16.904GiB; the same eight focused test/benchmark executables listed
in `profiles/bench/phase5-refresh/cleanup.json` remain. The isolated builder is
stopped; the other checkout's builder is untouched. No clone or model download
was needed; this verification run ended before publication. The pre-existing
research README edit is preserved exactly, the pull's own stash backup is removed,
local documentation links resolve
and `git diff --check` passes.

## Publication integration

Before publication, `experimental` advanced to `aa3252fa`, which changes RAM/disk
restoration to retain only the state set required by the planned path. The
performance changes restore without conflicts. A four-job rebuild of the Engine,
benchmark, cache tests and supported DFlash real test passes. Under the shared
GPU lock, both updated RAM/disk cache tests pass, as do k7 C6 chain, W12 C6 tree,
long-tree isolation, adaptive RAM reseed/restore and exact k4/W5 chain-state
reconstruction. The measurements and full-suite results above were obtained
against `48eb0b01`; these additional checks qualify the cache-restore integration.
All staged-file commit-hook checks pass without source changes.

The performance commit includes the implementation, tests, canonical benchmarks,
their supporting references and this report. The broader inventory, remaining
triage material, documentation-index changes and pre-existing research README
edit remain outside it.
