# Remaining repository experiment results

The requested Phase6 publication is complete: `945515ea`
(`perf(dflash): fuse convolution finish residual and rmsnorm`) is pushed to
`origin/experimental`. Its full C++ gate passed 121 checks, two unavailable
legacy-artifact skips and zero failures. This publication integrates the five
retained Phase7 changes after completed qualification and independent cross-review.

Phase7 evaluates **all 18** remaining Phase4 `ready_to_implement` mechanisms.
The six previous experiments and the earlier feature-history rejection remain
closed; paper discovery and broader inventory are separate. Three disjoint
GPT-6.1-sol reviewer shards own activation, GDN/state and policy/frontend questions.
Root owns builds, GPU runs, integration and this report. `phase7-ledger.json`
tracks the finite count; evidence is in `evidence/phase7/*-outcomes.jsonl`.

## Completion requirements

- [x] All 18 current-native first gates evaluated and recorded.
- [x] Every admitted numerical/state candidate checked against an independent oracle.
- [x] Every question has an executed decision or concrete external prerequisite.
- [x] Losing candidates and temporary controls removed; retained implementations integrated.
- [x] Relevant focused and full C++ gates completed for retained substantive changes.
- [x] Owned temporary artifacts cleaned and the final ledger reconciled.

**Complete: 18/18 decisions closed, five retained implementations integrated and
qualified locally.** One question is already native; the other twelve have a
rejection or a concrete external prerequisite. Paper investigation and paused
Phase1 discovery remain separate.

## Execution and numerical contract

Target: Qwen3.8-27B NVFP4, RTX5090 `sm_120a`, CUDA13.1, one resident
model, startup-fixed C1–6. Engine comparisons use default NVFP4 target KV and
optimized DFlash head. The exact supplied artifact is
`/models/qwen3.8-nvfp4-flash2-nvfp4-bf16codebook-from-bf16/qwen3_8_27b_nvfp4_dflash_nvfp4.ninfer`;
its draft-local KV is BF16 and its MTP matrices are W8.

Every GPU invocation holds host `flock /tmp/ninfer-gpu-queue.lock`. The isolated
builder is capped at 16GiB memory, no additional swap, four CPU cores and four
build jobs, with one nvcc optimization thread. Host research helpers use the
existing capped runner. No foreign inference-engine builds, clones or model downloads were needed.
The standard full-gate bootstrap repaired builder packages and installed the
repository-pinned clang-tidy; final C++ execution uses the prepared container
directly to preserve 16GiB/four-core/four-job limits and benchmarks OFF.
Only owned temporary artifacts are cleaned.

Floating-point qualification uses independent complete FP64 formulas from
represented public inputs, exact signed stored-weight/scales, observable BF16
casts, codec/state boundaries and declared output criteria. Exact codecs,
records, padding and cache transformations use exact comparison. Pairwise
production agreement supplements the oracle; it does not replace it. Criteria
were not widened to admit candidates. Operator measurements do not establish
Engine gains. Rejections apply to the tested mappings and workloads, rather than
claiming every conceivable variant impossible.

## Decisions

| Phase4 question | Decision | Deciding evidence |
|---|---|---|
| Q7 SwiGLU→NVFP4 output | Retain exactly T1024/4096 | Complete math/codec/guards and permanent sanitizers pass; all four actual Engine prefill pairs improve at both selected lengths. |
| Q8 Gated RMS→A8 | Already implemented | Current verify producer uses the canonical token-wide 48-head A8 codec and packed projection consumer; prior independent oracle qualifies it. |
| Q9 RMS→NVFP4 output | Retain eligible prefill route | Independent complete RMS→cast→codec→projection passes; paired complete operator edge improves across practical lengths. |
| Q10 Sigmoid→NVFP4 output | Retain eligible prefill route | Nonzero-residual independent complete producer/codec/projection passes; practical-shape complete edge gains. |
| Q11 Adaptive NVFP4 scaling | Reject tested profiles | Activation profile has six quality regressions; qualified KV append+attention slows 9.664→24.672us at128 and12.544→67.840us at1024. Prototype removed. |
| Q12 Exact long-prefill partitions | Reject complete cost | Full FP64 math passes, but mandatory summary exceeds40.640us budget; complete fresh/nonzero compounds slow16.80%/39.53%. |
| Q13 Projection/conv/recurrence fusion | Reject classifier mappings | Per-head ownership adds20MiB Q/K code reads to remove≤2.8125MiB handoff atB6/W12; complete head group needs49,152FP32 values and only16–96CTAs across170SMs. |
| Q14 Causal-prefill tile choice | Retain short Exact Br64 | T≤256 unsplit native24-query-head route gains, full causal oracle and cached fragmented sanitizer cases pass. Longer tiles remain Br128. |
| Q15 Histogram Top16 | Reject real Engine cost | Exact scope and20 full public oracle cases pass, but four C6k3 Engine pairs average−.0206%, with identical acceptance. All trial changes removed. |
| Q16 GDN FP16 high/low MMA | Reject tested mapping | FP16 issue19.488us versus TF32K8 19.264us, plus2.688us packing; independent exponent-range counterexample. |
| Q17 Confidence-ranked prefixes | Reject current physical route | Heldout predictive signal exists, but unchanged physical width makes full same-q verification dominate; measured cost requires28.5% round removal. Observation hooks removed. |
| Q18 Committed-suffix proposals | Reject tested profiles | Lookup-only needs54.297% round removal; measured optimistic removable prefix≤13.977%; hybrid licensed yield falls17.84%. |
| Q19 MTP target anchor | External checkpoint prerequisite | Current trained first-target/previous-draft-hidden convention cannot accept foreign fixed-anchor semantics as a speed fix without a matching checkpoint/training result. |
| Q20 GDN accepted history | Retain fixed capacity4 | Actual48-layer API/state/stride/terminal/lifecycle oracles and bounded sanitizers pass; final C6k3 Engine gains2.39%, C1 is flat. |
| Q21 Tool closing cursor | Reject cost ceiling | Perfect removal saves<.23% publication CPU, before GPU inference; six exact fixtures pass. Timers removed. |
| Q22 Target-NLL draft predictor | Reject bounded shortening profile | Mixed conditional Brier changes;15 heldout fullC6 groups make zero decisions at the minimum captured arm. Broader exploration remains unqualified. |
| Q23 Activation-global-amax | Reject tested quality profile | Five downstream quality regressions despite aggregate normal SSE.9590x; no CUDA route admitted. |
| Q25 GDN dependent launch | Reject real Engine cost | Twelve math/state/raw/A8 cases pass; four actual Engine pairs average−.0364% with identical acceptance. PDL trial removed. |

## Retained activation and attention evidence

RMS→A4 preserves the represented BF16 producer result and native group16 codec,
then feeds the existing NVFP4 attention projection directly. At T1024, 300
alternating warm-L2 graph pairs measure120.064→114.688us (4.48%); T4096
511.456→484.864us. Full mathematical, exact-codec/padding/guard and eager/graph
checks pass T256/385/512/513/768/1024/4096. The target selects only its existing
AllowA4 nonverify attention panels (T256 orT>384), with caller-owned packed
storage and correctly scoped projection scratch.

Sigmoid→A4 similarly avoids an independent quantization launch before the
residual projection. At T1024, complete graph medians65.056→61.184us include
identical12.352us fixture reset; T4096 421.600→385.856us include117.728us reset.
T513/768/1024/4096 pass the independent nonzero-residual formula, exact codec,
immutable inputs and guards. Selection follows existing AllowA4 nonverify T>512.

SwiGLU→A4 encodes the existing BF16 shared result after the existing consumer
barrier; it adds no GEMM, repacking or barrier. Selected T1024 graph medians
484.992→483.040us, T4096 1874.176→1826.432us. The synthetic4096 eager fixture
slows1757.920→1811.904us, so retention uses actual Engine prefill evidence:
four alternating process pairs, C1/k7, `-p 1024,4096 --prefill-chunk 4096`,
three repetitions and one warmup, improve every pair. Mean prefill gains are
**+.7158% at1024 and+1.5625% at4096**. Only these exact lengths select packed output.

The partial513 standalone producer fails the unchanged primary oracle criterion
in both native and packed paths, with exactly identical BF16 output. Independent
input-codec control diagnoses the existing private A4 arithmetic floor. The
closed projection/residual composite, native codec/padding/guard checks pass;
this is explicitly not a qualified standalone513 producer claim. T768 math passes
but paired graph interval crosses zero, so it also keeps the ordinary route.

Combined RMS/sigmoid/short-Exact versus the published baseline has four positive
Engine prefill pairs at1024/4096, mean+.1779%/+.0957%. These small combined values
do not attribute an individual Engine gain to RMS or sigmoid; their strongest
claims remain the measured complete operator edges.

Selected Exact attention uses Br64 only for short T≤256 unsplit native24-query-head
prefill. Four bracketed public-Op pairs measure128/context0 22.048→16.208us,
256/context0 30.688→22.528us,128/prior512 55.296→42.656us and256/prior512
65.008→48.832us. At512/prior512 Br64 loses81.920→96.256us, so that route keeps
Br128. Independent complete causal attention covers fresh/partial/prior-context
and fragmented pages; selected partial129/prior512 passes memcheck and racecheck.
Optional Sage precision is a separate route and its default is unchanged.

The permanent `ninfer_a4_activation_test` passes normal, memcheck (zero errors)
and racecheck (zero hazards) with independent RMS385, sigmoid513, SwiGLU1024 and
partial513 closed-composite/codec/guard cases. Selected projection scratch and
packed outputs are caller-owned and address-stable under CUDA Graph replay.

## Histogram and controller findings

Histogram Top16 passes46 exact fixtures, including2049-way ties, signed zero,
nonfinite filtering, fewer than16 finite scores, all-nonfinite and the complete
BF16 bit domain. Both native and candidate pass20 complete public selector cases
with nonzero exact NVFP4 input and full FP64 projection/codebook/path/q mathematics
for greedy, stochastic and p-less behavior. Tail-shaped score fixtures improve
B6/k7 139.584→80.589us and140.051→76.269us, but uniform/flat/tied scores lose
through mandatory exact cold fallback. Actual C6/k3/W4 pp512/tg128 Engine pairs,
two repetitions and one warmup each, change−.1692/+.0437/−.0652/+.1081%,
with identical acceptance. This distribution-sensitive route is removed.

Chosen-proposal Markov confidence has useful prompt-heldout ranking signal:
raw same-budget shortened-prefix yield improves.850% adaptive and.591% fixed;
adaptive Brier.2401→.1874. All six main prompts and257 committed tokens match
across fixed/adaptive captures. This does not imply a physical speedup: unchanged
physical width still produces and verifies all rows, and full same-q verification
dominates shortened choices. Actual fullK7/W8 costs1.427times mostlyK3/W4;
ranked budget18 needs28.5% complete-round removal before confidence/ranking cost.
No ragged graph/scheduler is admitted by this bounded result. Auxiliary API and
runtime trace hooks are removed; bounded predictive findings remain in evidence.

Target-NLL uses licensed committed-sequence scores, not inferred wrong-context
rejected-draft likelihood. Conditional-hazard Brier improves two sequences and
regresses one. All15 heldout groups use captured arm3, its minimum, so zero policy
changes refuse this bounded shortening route but cannot qualify or disprove
larger-arm exploration.

## Accepted-history qualification

The candidate keeps at most four accepted transitions between dense checkpoint
writes, sharing FP32 keys across ratio-three value heads. Native recurrence
traversal emits provisional innovations; commit folds the licensed ancestor path,
and a shared per-lane count publishes only after all48 layer commits. Snapshot,
restore, ordinary mutation, prefix export and terminal retention materialize or
reset history at the required state boundary. No separate stream or event is added.

All12 isolated geometries pass complete48-layer/64-round FP64 recurrence,
queries/final state, exact raw records and conv3. Reachable cap4 cadence gains
are B1 W4/W8-chain/W12-tree7.652/2.114/5.087% and B6 23.609/14.339/9.126%.
Cap16 loses atB1 and costs more memory. Cap4 C6 ring costs36.21MiB plus
108.63MiB provisional planes atW12, with raw records retained. These isolated
cadence savings are **not** an Engine claim.

Six reachable actual-API geometries pass64-round independent FP64 query/state,
exact raw records/conv3, eager and graph replay, plus snapshot/abort/restore and
peer continuation. A separate C6/ceiling12 compact fixture composes legal row
subwindows0/2, four absolute slots5/1/4/0 and activeT4→8, checking all48-layer
outputs/states, exact records and unchanged inactive/staging slots. Memcheck
passes the48-layer compound; racecheck passes unfiltered bounded one-layer W4
chain/W12 tree record with direct FP64 output/raw-record checks, eager and graph.
The larger repeated-cadence racecheck exceeded the16GiB cap, so it is not a passed
check; the permanent test registrations separate global compound memcheck and
bounded shared-memory record racecheck.

Real Engine k7 chain/W12 tree C6 isolation, adaptive widths, terminal delivery,
RAM reseed/restore during concurrent decode pass. All six VRAM/RAM/disk cancellation
cases and C2 idle/full-RAM pressure checks pass. A short-terminal regression requires
the actual one-anchor zero-draft fallback and exact eight-token continuation from
resident versus RAM-restored state. Terminal lanes materialize inside the existing
commit tail before synchronization and retained publication; no new fence is added.
Heavy disk scratch uses the mounted source filesystem, not the container overlay.

After the terminal-boundary fix, two alternating final Engine pairs atC6/k3/W4,
pp512/tg128, two repetitions/one warmup, improve2.3245/2.4613%, mean**+2.39294%**,
with identical speculative statistics. Earlier fourpairs confirm+2.4618%; C1/k7/W8
is flat(mean−.00013%). Final C6/k7/W12 tree pairs improve.2416/.5070%, mean**+.37431%**, with
identical speculative statistics. These corpus-specific Engine results do not
establish a universal throughput gain.

A hard k4/W5 speculative-versus-ordinary64-token comparison fails at token21 in
both prehistory and candidate binaries, with exactly identical64-token speculative
outputs and identical ordinary outputs. This remains a known failing baseline,
not a passed test or an attributed history regression; its criterion is unchanged.
Direct numerical/state oracles and supported real route checks remain the primary
qualification.

## Verification and evidence

The full C++ unit suite completed with **126 passes, two unavailable legacy-artifact
skips and zero failures** (128 registered checks). It includes the permanent A4,
history, compact-stride and selected fragmented-attention oracle checks. Final
host index/optional-access cleanup was followed by a successful affected-target
rebuild, A4 and history oracle tests and the real short-terminal continuation
regression. Production arithmetic and numerical criteria were unchanged by that
cleanup. Changed-line clang-tidy 22.1.8 passed all 50 affected translation units;
the final test-index correction also passed its focused static check.

Retained activation and attention memcheck/racecheck gates pass; history global
memcheck and bounded unfiltered record racecheck pass. Final A4 and history
initcheck runs report zero errors. The oversized history racecheck and known
baseline hard-greedy failure remain explicit limitations above, not passed gates.
Active reference paths and the 18-question/disjoint 6/5/7 outcome counts were
reconciled; `git diff --check` passes.

Final independent cross-review also found no concrete commit blockers: three
subagents separately reviewed GDN Op/storage/tests, A4 Op/codecs/tests and
family/target integration authored by other reviewers; root reviewed the Exact
tile changes. The read-only scope and qualifications are recorded in
`evidence/phase7/final-cross-review.json`. Readiness applies to the five retained
implementations and their affected tests/docs, with the limitations above.

Publication checks also pass the pinned commit-stage format/lint/file-hygiene
gates. Formatter-only changes were followed by a successful affected-target
rebuild, A4/history mathematical checks, the real short-terminal continuation
regression and A4/history initcheck (zero errors).

Owned saved experiment executables and temporary benchmark build outputs were
removed, reclaiming **4.57GiB**, while retaining compact measurement summaries,
source harnesses and permanent test targets. The cleanup receipt is
`profiles/bench/phase7/owned-cleanup-summary.json`. The five retained Phase7 implementations and this report are published together;
the earlier Phase6 publication remains `945515ea`. Broader inventory and paper
investigation remain separate from this implementation publication.

Bounded measurement summaries and commands are under `profiles/bench/phase7/`:
`activation/*engine-gate-summary.json`, `policy_frontend/exact-br64-paired-summary.json`,
`policy_frontend/histogram-engine-gate-summary.json`,
`policy_frontend/confidence-physical-cost-gate.json` and
`gdn_state/pdl-engine-gate-summary.json`. Shard outcome files preserve the
question-specific classifier, mathematics and source provenance.
