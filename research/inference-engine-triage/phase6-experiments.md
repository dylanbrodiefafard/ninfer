# Repository experiments: next three mechanisms

Phase3 identified 49 promising repository originals; Phase4 grouped their
mechanisms into 25 qualification questions (24 proposed experiments, one
rejection). Phase5 implemented and qualified the first three. This finite next
batch addresses questions 4–6 against `442099cb` on `experimental`, after the
user requested a fresh pull and further work. The pull was already up to date.
The paper handoff remains separate; this batch does not reopen discovery.

Target: Qwen3.8-27B NVFP4, RTX5090 `sm_120a`, CUDA13.1, one resident model,
startup-fixed B1–6, default NVFP4 target KV and BF16 local DFlash2 KV. Existing
producer-side A8 must be included when pricing GDN readout; historical proposals
are not evidence that a mechanism remains missing.

| Question | Owner | First decisive gate | Decision |
|---|---|---|---|
| DFlash convolution finish/residual/plain RMS | `phase6_dflash_boundary` | Independent composite baseline, including observable BF16 residual and cancellation-conditioned normalization criterion; public Op timing before candidate | retain operator gain, Engine effectively flat; integration and full C++ gates pass |
| GDN recurrence/gated normalization | `phase6_gdn_readout` | Current BF16 and A8 complete-edge baseline; price loss of eight-way head parallelism and token-wide A8 scale reduction | reject: complete edge ~14% slower; provisional production path removed |
| Accepted-state replay overlap | `phase6_replay_overlap` | Current state/scratch dependency DAG and node-level fold/preparation trace; no stream/graph redesign without a positive measured overlap window | reject: Engine 1.22–1.91% slower; entire stream/event trial removed |

The root coordinator owns shared build wiring, this report, all builds and GPU
execution. Agents own disjoint semantic Op files and source analysis. CUDA
candidates start only after baseline qualification and admission. A measured win,
substantiated rejection or concrete prerequisite completes each question; a
proposal alone does not. Losing implementations and temporary controls are
removed. Publication was subsequently authorized on 2026-10-04; the retained
DFlash implementation and this bounded results report are the publication scope.

Every GPU run uses host `flock /tmp/ninfer-gpu-queue.lock`. The isolated builder
uses at most four build jobs, one nvcc optimization thread, 16GiB hard memory,
four CPU cores and no additional swap. Other builders and model files are
untouched. No foreign build, clone, model download or dependency upgrade is
needed. Only owned temporary binaries and traces are cleaned up.

## Completion checks

- [x] All three current-baseline/admission decisions recorded.
- [x] Admitted candidates qualified directly against independent mathematical/state oracles.
- [x] Each question has a substantiated result and coherent production decision.
- [x] Relevant capture, sanitizer and real-model checks completed for retained changes.
- [x] Full C++ gate completed for substantial retained implementation changes.
- [x] Losing paths and temporary controls removed; affected references reconciled.

Raw local evidence is collected under `profiles/bench/phase6/`. Performance
claims must name the measured Op, complete edge, round or Engine workload;
author speed claims and byte counts are not local speedups.

## Current admission evidence

DFlash's complete convolution-finish/residual/plain-RMS baseline is independently
qualified at B1,2,6 and W3,8,12 with random and cancellation fixtures. The oracle
evaluates the complete FP64 formula from represented BF16 inputs; the observable
residual boundary additionally has a tight independent plain-RMS check. Cancellation
requires propagating finish/storage error through the normalization condition,
rather than comparing a staged baseline against an unjustified fixed relative bound.
Existing Op criteria are unchanged. The candidate preserves the finish BF16 round
inside registers and publishes the BF16 residual before normalizing it.

The 512-thread public composite passed these checks and exact residual comparison
with the unfused baseline. In 300 alternating graph pairs, every tested shape had
a positive 95% interval for baseline minus composite. Representative warm-L2
captured complete-edge medians (reset excluded): B1/W3 5.952→5.504us;
B6/W8 7.552→5.760us; B6/W12 7.936→5.888us. The corresponding paired mean
differences were 0.829 [0.711,0.946], 1.610 [1.481,1.739], and
1.820 [1.657,1.983]us. These include graph dispatch/event timing and do not
establish an Engine speedup. Production caller/workspace integration is admitted.

GDN's current B6/W3 sequential record plus A8 gated-normalization complete edge
measured 15.232us, with recurrence 12.096us. A per-head fusion still needs a
token-wide A8 reduction and a BF16 handoff: both old and proposed edges have two
launches. One bounded dense B6/W3 experiment is admitted; B1 underfills the 170-SM
GPU. The full-head prototype compiled with 128 registers/thread, 36 shared bytes,
zero stack and zero local bytes, meeting the provisional two-resident-CTA resource
gate before numerical qualification and paired timing.

The temporary GDN baseline initially rejected the closest representable BF16
gated-normalization output: FP64 ideal 0.25489861496242611, nearest BF16
0.255859375, unavoidable error 0.00096076003757389339, versus an old gross
allowance 0.0008280829598663683. Before any candidate CUDA, the temporary
arithmetic-profile gross coefficient was derived from BF16 half-ULP plus
32 FP32 epsilons (0.003910064697265625), with absolute allowance 4.5e-5;
relative L2 remained 0.00185. No existing Op criterion was changed. The candidate's
complete oracle evaluates FP64 recurrence and gated RMS directly from represented
inputs, without copying its private BF16 recurrence staging. Its predeclared
head-wise propagation bounds use the existing recurrence L2/gross envelopes and
the same reverse-triangle/Jacobian normalization bound as the DFlash composition;
exact record/state and A8 checks remain separate from floating-point qualification.

The dense full-head candidate passed the independent complete FP64 oracle
(relative L2 0.002373; absolute norm error 0.04443 versus the predeclared
condition-aware cap 0.38672), exact raw records, unchanged FP32 state, and exact
A8 encoding from its represented BF16 output. It lost the admitted B6/W3 speed
gate: 300 alternating pairs measured baseline 14.624us versus candidate 16.672us
(median paired ratio 1.1397), with p95 15.392 versus 17.312us. Both graphs had two
nodes. The candidate reduced head parallelism without removing the token-wide
encoding launch or BF16 handoff. Reject this fusion; remove the provisional
normalized recurrence and standalone encoder APIs. No wider-shape or Engine
experiment is justified by this loss.

Accepted-state fold overlap has a measured dependency window. Nsight Systems
2026.1.3 node tracing of the committed baseline matched 55 B1/k3 folds to their
following graph launches and 37 B6/k7/W12 tree pairs. Median fold durations were
170.751us and 1196.379us, with exposed tails at following launch API entry of
169.190us and 1096.002us. Independent preparation/proposal before the first
target-state consumer lasted 1323.800us and 2996.308us. These trace windows admit
a separate fold stream and stable completion event, but are optimistic bounds,
not speedups; profiling inflates graph dispatch and concurrent draft work can
contend for memory bandwidth. Unprofiled Engine comparisons must decide retention.

## Isolated Engine decisions

Compared the committed baseline, DFlash fusion alone and DFlash fusion plus replay
overlap. The isolated control disables only creation of the replay executor, leaving
all DFlash fusion arithmetic and graph geometry identical. The control is a temporary
source/build comparison, not a retained runtime option. Each comparison uses an
old/new/old bracket, three measured repetitions per run after one warmup,
pp512/tg128, optimized DFlash proposal head and NVFP4 target KV. Baseline means
below average both bracket runs. All three bracket members have identical
acceptance lengths and round counts within each workload.

| Workload | Committed → DFlash fusion, output tokens/s | Fusion only → plus overlap, output tokens/s |
|---|---:|---:|
| C1, k3 chain | 170.812 → 170.982 (+0.10%) | 170.594 → 167.341 (−1.91%) |
| C6, k3 chain | 735.754 → 736.088 (+0.05%) | 731.144 → 720.541 (−1.45%) |
| C6, k7/W12 tree | 252.370 → 252.484 (+0.05%) | 251.672 → 248.601 (−1.22%) |

DFlash fusion has a repeatable independently measured operator-edge gain and no
material Engine regression in this sample, but the Engine deltas overlap measurement
variation. Retain the central closed Op and coherent shared normalized-buffer
integration; do not claim an end-to-end speedup. Workspace high-water values are
unchanged in the sampled Engine workloads because another phase sets the peak.

Reject the replay-overlap trial in all three admitted workloads. The trace window
did not translate into net throughput; it does not justify a narrower production
eligibility rule. The prototype's separate stream, borrowed event, graph wait,
state-transition joins, teardown drains and architecture changes are removed.
The startup smoke exposed the CUDA rule that an external wait flag is valid only
during capture; the corrected trial passed eager startup, captured chain decoding,
adaptive draft and RAM restore/reseed smoke checks before measurement. No retained
runtime change depends on this experiment. A later revisit needs a concrete way
to reduce contention/event cost or a newly measured workload, not the old optimistic
window alone.

The retained DFlash public suite qualified both unfused and composite outputs with
the strengthened per-column Jacobian bound, tight represented-residual RMS criterion,
guards, input preservation and eager/repeated capture at all documented shapes.
Focused B6/W12 memcheck reported zero errors; racecheck reported zero hazards,
errors or warnings. The production public benchmark measured B6/W8 7.552→5.568us
(three nodes → one, warm L2, residual reset excluded). These numerical checks are
independent of pairwise agreement.

Real-artifact qualification of the retained path also passed k7 chain and k7/W12
tree execution at C6, including overlapping requests, mixed short budgets and
long tree isolation. Adaptive {3,4,5}, terminal delivery/queue telemetry, RAM
reseed and in-flight RAM restore passed. Target-only greedy differences are
reported as cross-schedule diagnostics by the existing test; strict same-profile
isolation and state/reuse checks are the behavioral gates.

The full `scripts/run-unit-tests.sh --inner` gate ran in the capped builder under
the host GPU queue lock with four build jobs: 123 registered checks, 121 passed,
two unavailable legacy Qwen3.6 load-plan artifact checks skipped, zero failures,
1089.29s test time. The supported Qwen3.8 real artifact was checked separately
using the exact path recorded in the comparison commands.

The temporary Nsight container has been removed. Rejected stream/event and GDN/A8
production paths and all temporary CMake targets are absent. Saved comparison
executables, invalid sweep source, raw trace/export files and source-control backup
were deleted after recording bounded summaries: 1.141GiB of temporary file contents.
After the full gate, 186 unused generated executables were removed (19.287GiB
of file contents); 13 focused test/benchmark and product executables remain.
No model artifact, other builder or unowned clone was removed. Cleanup manifests
are `profiles/bench/phase6/temporary-cleanup.json` and `suite-cleanup.json`.

Scoped pinned pre-commit formatting/file/lint hooks pass, and `git diff --check`
passes. The changed-line clang-tidy command could not execute because the cached
builder lacks the required clang-tidy 22.1.8; no dependency/toolchain installation
was performed. Native compilation uses the repository's warnings-as-errors policy.
This is a tooling limitation, not a claimed clang-tidy pass. The user subsequently
requested commit and push of the retained implementation on 2026-10-04. The
recorded full C++ gate remains valid: publication changes the report only, with
no subsequent implementation changes. The two losing prototypes remain removed.
