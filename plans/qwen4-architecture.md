# Qwen4 architecture implementation plan

This plan defines how NInfer should add the Qwen4 architecture accurately and how that work is
qualified. The currently available architectural anchor is `Qwen/Qwen3.8-Flash-Next`, an official
Qwen4 preview whose checkpoint identifies itself as `qwen4_exp`. It is not evidence that every
future Qwen4 checkpoint has the same dimensions, components, tokenizer, or MTP design.

The first architecture implementation therefore has two levels:

1. a new `qwen4` family that owns the Text/Vision/speculative schedule, QSA/GR/PLE orchestration,
   state/workspace composition, and CUDA-Graph algorithms established by the preview, while
   semantically closed transformations remain central Ops; and
2. an exact target leaf for a named checkpoint whose configuration, tensor inventory, numeric
   profile, and frontend resources are fixed independently.

`qwen4` is the research and oracle target, instantiated today from the Qwen3.8-Flash-Next preview
checkpoint. For an exact Qwen4 target that contains the preview-style PLE table, the table is
explicitly an artifact-backed, host-resident random-access
tensor; “one resident model” means one model instance whose compute core is resident on the GPU,
not that the PLE table is copied to VRAM. The complete table is populated and OS-locked in RAM
before load succeeds, then read through an owned mapping, bounded CUDA-pinned staging, and
asynchronous H2D gathers as specified in section 3. This does not assume every future
Qwen4 checkpoint contains PLE. No currently audited preview profile qualifies for registration as a
runnable RTX 5090 product: its official BF16 payload is 359,999,963,128 bytes, and the audited
Unsloth UD-IQ1_S GGUF still has
43,735,298,560 non-PLE tensor bytes (40.73 GiB) after its 28,800,138,240-byte PLE tensor is assigned
to host storage. This exceeds the 32 GB GPU before state, KV, workspace, and graphs. Product
activation waits for a Qwen4 core quantization/profile that fits one RTX 5090 while retaining the
host PLE design. Running this preview requires either some CPU-executed core weights, a newly
qualified more-aggressive core quantization, expert streaming, or multi-GPU execution; none becomes
an NInfer product path implicitly.

For architecture bring-up only, this plan selects one explicit exception rather than treating the
whole GGUF as a product profile. An unregistered C=1 eager Text verifier may preserve the public
UD-IQ1_S represented weights, keep NVFP4-G16 QSA KV on the GPU, and host-map PLE plus the 48 routed
gate/up expert banks. Routed-down and every non-routed tensor remain device-resident. GPU routing
selects ten experts, then a bounded two-slot host/device pipeline gathers only those compressed
gate/up slices, one route rank at a time, for GPU execution. The verifier has a 4096-token ceiling
so it can cross the 2048-token QSA selection
budget without claiming native-context capacity. It is not an Engine/CLI/serve identity, CUDA Graph
or performance route, and does not authorize CPU execution or any other streamed weight.

No full checkpoint download is needed to execute this plan's research/design stage. The pinned
configuration, frontend resources, paper, and model implementation establish the architecture;
the index establishes tensor names, shard assignment, and total bytes. Exact shape/dtype inventory
requires reading every safetensors header, which can be done with bounded HTTP range reads without
downloading payloads. A full checkpoint becomes a deliberate conversion and qualification
prerequisite only after the runnable target identity is selected.

## 0. Native-checkpoint preparation and native precision qualification

### Completed goal: held-out resident-block precision qualification (2026-09-19)

Assigned after commit `014efc9b`. The previous goal closed its bounded experiments,
not every research-backed precision candidate. This follow-up uses authentic preview
components without requiring the complete ordinary compute core to fit the GPU.
No commit or push is authorized. Calibration and evaluation inputs must be disjoint;
existing failed recipes remain regression evidence, and criteria must not be widened.

- [x] Build diverse, longer real-input resident-block calibration and held-out panels;
  measure per-token and propagated error, recurrent state and routing/selection changes.
- [x] Evaluate original-source shared/GDN/QSA NVFP4 W4A16 and W4A4 candidates;
  qualify useful implementations or reject with independent mathematical/source-loss evidence.
- [x] Extend selective FP8/A8 component qualification beyond the current layer-0 Z and
  layer-0/3 shared panel where authentic weights and correctly propagated inputs permit.
- [x] Evaluate calibration-backed routed-expert/post-SwiGLU down and DFlash remedies
  with separate holdouts, retaining failed recipes as regression cases.
- [x] Broaden embedding/head, final GR-read and PLE-projection FP8 source-loss and
  composition evidence over representative inputs, distinguishing local from model quality.
- [x] Complete relevant exact/numerical/state checks, direct performance evidence for
  retained optimizations, active authority updates and the full C++ regression suite.

Public BF16 boundaries, protected controls, FP32 GDN state, p-less/epsilon sampling,
speculative transactions and eagerly locked host PLE remain unchanged unless a separately
qualified explicit design requires otherwise. Full future-model PPL, final precision defaults
and end-to-end speculative performance are not claims made by resident-component tests.

Progress: eight disjoint authored documents acquired (four calibration at137 tokens,
four heldout at257), reusing complete local source FP8 PLE for offline row extraction.
The first longer native panel exposed a local GDN1 output failure and one exact router
rank-order failure that the original33-token witness missed. Same-input GDN recurrence A/B:
private chunked profile relative L2=0.003612/maxabs=0.017770 versus unchanged gross
limit0.010346; FP32 recurrent profile relative L2=0.001684/maxabs=0.003739 passes.
Native FP32 recurrent dispatch passed all eight panels' original whole-output component checks;
the subsequent per-token and full-suite results are recorded below. Router diagnosis found FP32 dot-reduction collapse of two distinct source logits;
compensated native BF16 dots and canonical two-word ranking now pass the captured137-row,
decode, collapsed-score, true-tie and cancellation witnesses in eager/graph execution, plus
the complete MoE synthetic oracle suite. Three other137-token calibration documents pass
the unchanged native local/accumulated gates with the recurrent GDN profile. Corrected science
and four257-token held-out captures are complete. Seven documents pass the original
accumulated panel/gross screen; the original heldout_multilingual capture failed at layer2/3 token120. All local
whole-output Op gates and all same-input exact router orders passed. Final residual panel relative L2 is
0.7083–1.0452%; worst-token relative L2 is3.8017–5.3879%, including tails in otherwise passing
panels. At the multilingual witness, a0.5735% router-input difference changes the tenth expert
from369 to75; independent full-MoE error is21.8289%, falling to0.4728% when the reference
expert set is held fixed. This is propagated hard-selection sensitivity, not the fixed
same-input router bug. Before the compensated-QKV repair, native GDN layer1 T137 measured395.264us median through the complete
public Op (state restore excluded, RTX5090/CUDA13.1, three warmups/eleven samples); no
before/after speedup claim is made. Resident panels retain the explicit bounded `t+axis`
MRoPE profile, not literal source-framework full-text execution. Accumulated routing/residual sensitivity is measured
separately and is not hidden by modifying the mathematical oracle or tolerances. Captured
public inputs remain usable for same-input counterfactuals even when a propagated-chain screen
fails; they are not thereby qualified full-model calibration or quality evidence.

The finite calibration remedies are complete, with explicit rejection rather than
relaxed criteria: routed source/fitted A4 fails every held-out complete-MoE screen;
DFlash input-MSE improves worst-query hidden error from39.00% to34.11%, but all84
queries still fail2%. Endpoint/PLE source-loss checks are also complete: PLE value
and combined FP8 projections have token tails above2%, while row-FP8 head changes
two of12 sampled argmaxes. Full results and scope are in the existing precision and
speculative research references. New source-retained implementation candidates are
layer0 shared-up NVFP4/A16, row-FP8 Z2/A16/A8, and narrower layer-specific tensor-FP8/A8
roles. Shared-up NVFP4 and Z2 row-FP8 pass complete real-weight GPU oracles;
NVFP4 is a storage tradeoff, not a speedup. With compensated QKV, complete-GDN A8 saves5.56% for Z0 and6.03%
for Z2 on identical257-token inputs. The seven-weight selective recipe now contains
six tensor-FP8 roles plus row-FP8 Z2; conversion/binding rejects missing weights and
unqualified combinations. All-target C++ compilation and metadata admission pass;
32 relevant Python checks pass (one optional source-MTP fixture check skipped).
Strengthened GDN checks pass all257 tokens for Z0 and row-Z2 A8, with unchanged
convolution/recurrent-state gates. Mixed shared layer2 A16 exposed a separate local
rounding failure: token114/coordinate2392 is `-0.0849609375` against FP64
`-0.0856909956`, exceeding the original gross bound. Independent attribution reproduces
the failure from private per-expert down BF16 stores before the weighted sum; retaining
the down results until FP32 accumulation predicts `-0.08544921875`, within the bound.
The native A16 repair now passes the original257-token witness with unchanged
per-token, exact-route, probability and guard checks; the earlier shared-up rounding
hypothesis was ruled out. All eight selected shared A16/A8 held-out component checks
now pass; complete-MoE A8 latency reductions are modest0.39–0.77% on these same-input
pairs. Final BF16/NVFP4 shared-up controls also pass; the fitted NVFP4 option is
about0.8% slower but saves71.875% of that matrix's payload. The seven-weight
runtime integration passes A16/selective-A8 prefill and exact compact/recorded
graph policy-preservation checks. Selective-A8 whole33 and32+1 continuation also
pass. The final-route multilingual prefix exposed two GDN1 per-token gross
failures at tokens20/241, coordinate1883, under the strengthened local checks.
Attribution identifies QKV projection accumulation error crossing a declared BF16
store, amplified by one gated-normalization quantum; recurrent/norm/final-dot
arithmetic is not the cause. K16 MMA partials with compensated cross-partial sums
now pass the failed input and all four original held-out GDN1 panels. Complete
GDN T257 is630.752us versus559.104us for the failing profile. Compensated existing
SIMT execution restores T1 cost to96.256us versus94.144us, avoiding the padded-MMA
prototype's122.848us. Final mixed-projection and small-width checks pass;
the measured native-only dispatch uses SIMT through T20 and MMA from T21.
T21 complete-GDN falls from245.792 to161.824us with unchanged gates;
generic Linear dispatch is unchanged. Final seven-weight runtime integration and
selective-A8 whole33/32+1 continuation reruns pass with this profile, as does the
all-target build. Independent Astra review found no material issues.
The full-suite MoE witness exposed a race in the newly guarded test workspace:
default-stream initialization was not ordered before nonblocking-stream execution.
The preserved pre-fix binary reproduces the exact ID/probability failure; explicit
setup synchronization alone makes the focused suite pass, without numerical or
production changes. The final full-suite rerun was GPU-isolated because the
draft graph-budget check measures device-wide free memory; its initial concurrent
capture run invalidated that measurement. The isolated draft-runtime test passes.
Declared BF16 consumer boundaries and all criteria remain unchanged.
The earlier eight-panel baseline passed whole-output local
gates, not these subsequently added per-token GDN checks. The final repaired
multilingual four-block prefix passes all unchanged local output/token/state and
accumulated panel/gross checks. Final residual L2 is1.026423%, maxabs0.0126953125;
worst-token L2 is4.152011% at109, so this is not a per-token2% accumulated guarantee.
The previous token120 gross failure is resolved; hard expert-selection sensitivity
still exists. The original captures remain immutable inputs to the source study.
The GPU-isolated full C++ regression suite passes:126 passed,9 optional
fixture-dependent checks skipped,0 failures (538.94s), using
`NINFER_DEV_CONTAINER=ninfer-builder-qwen4-mixed NINFER_BUILD_VOLUME=ninfer-build-cache-qwen4-mixed NINFER_DEV_JOBS=8 ./scripts/run-unit-tests.sh`.
Real resident integration/continuation, the multilingual prefix,32 focused Python
checks and independent Astra review provide the additional evidence described above;
one optional Python source-MTP fixture check was skipped. No commit or push was made.
The earlier unrestricted shared0/3 A8 recipe
is superseded, not retained as an alternative native recipe.

### Completed goal: final feasible native mixed-precision readiness (2026-09-19)

The user assigned all six follow-up workstreams below. Preserve the completed, uncommitted
resident-component/early-PLE work. No commit or push is authorized by this goal. Weight
storage, GEMM activation precision, accumulation and persistent state remain independent
choices; the quantization-recommendations research is candidate authority, not blanket
default admission. A rejected candidate closes its experiment, not an unrelated implementation
obligation. Do not repeat a rejected experiment without a materially different hypothesis.

- [x] Integrate explicitly qualified native precision policies across conversion, binding,
  workspace planning, prefill, compact decode and speculative verification, retaining A16
  reference and protected controls. Distinguish bounded candidate execution from full-model
  quality admission; do not invent future checkpoint topology or precision inventories.
- [x] Optimize measured resident MoE grouping/reuse/launch/packing costs, including natural
  low-occupancy decode, under the kernel-iteration classifier and unchanged independent
  numerical criteria. Use resident component evidence, not streamed-expert throughput.
- [x] Evaluate actual-weight eight-bit vocabulary endpoints, selected GR-read/PLE projections
  and appropriate NVFP4 projection alternatives; complete justified support or record concrete
  rejection. Preserve A16 head reference, BF16 protected gates and FP32 recurrent state.
- [x] Investigate distinct actionable remedies for MTP propagated error, DFlash NVFP4 loss
  and NVFP4-KV accumulated sensitivity. Separate kernel correctness, represented-weight loss,
  activation quantization and propagated rounding; never widen criteria to admit a candidate.
- [x] Resolve whether materially distinct FP8-residual/QSA-A8 recipes have a supported path;
  complete FP8-KV runtime admission if its measured quality/capacity benefit warrants it,
  otherwise retain evidence-backed non-admission. Existing failed recipes stay rejected.
- [x] Qualify early PLE overlap under graph replay and larger prefill; acquire/qualify the
  complete authentic FP8 table with eager OS-locked RAM residency when local capacity permits.
  Preserve bounded pinned staging, immediate eligible fetching, exact decode, drain lifetime,
  graph ordering and no disk fallback or native ordinary-weight streaming.
- [x] Obtain finished-code Astra review; run relevant exact/numerical/state/graph/lifetime
  checks and full correct-container C++ unit suite. Update active authorities with actual
  decisions, material evidence and limitations; preserve p-less/epsilon sampling semantics.

External completion gates, not claims achievable with component fixtures: a fitting exact
checkpoint is required for full-model paired PPL/task and long-context quality, final per-layer
recipe/default selection, and end-to-end MTP/DFlash acceptance and throughput. These gates
must remain explicit and must not stop independent feasible work above.

Progress: fixed artifact-owned prefill recipe binding/planning/decoder wiring is implemented
and metadata admission plus seven converter tests pass. Real native33 A16/selective-A8 chains
pass the unchanged2% accumulated screen. Both routed-A4 recipes execute at least64 actual
expert occurrences and pass the existing complete-MoE oracle; this is not whole-chain quality
admission. All four policies preserve exact A16 compact/recorded graph output and state.
A16 remains mandatory for compact decode/verification and private drafts. Astra's
read-only precision review found no evidence-backed drop-in QSA A8 or FP8 residual remedy:
QSA's rejected path was already dynamically scaled, and residual loss is confirmed in the
independent formula. Those existing candidates stay rejected. Distinct K-only FP8/BF16-V
composition assessment is complete: aggregate losses pass but per-token drift and routing
changes do not justify FP8-K runtime admission. NVFP4 K-only fails; BF16 remains the reference.
The initial FP8 PLE RAM estimate omitted documented reclaimable clean ZFS ARC. Fresh conservative
capacity was75.39GB versus68.38GB required including16GiB reserve, enabling complete
pinned-source acquisition/conversion without host settings changes. Full FP8 residency now
passes: all12,500,061 pages present,51,200,249,856 bytes eagerly locked, all256 source
boundary rows decoded exactly, and teardown restored the original lock count. Other host swap
occupancy increased during this resource-intensive check; PLE itself remained locked and
the test container had no swap. Capacity admission is not a no-paging promise for other apps.
MTP private FP32 normalization, retaining original BF16 FC/add stores, passes the actual
captured24-row seed/two-draft local and propagated2% screens with unchanged routing and
criteria. Four-column weight reuse reduces its latency, but it remains slower than the former
BF16-normalization stem; this is a retained accuracy tradeoff, not a speed claim. Final retained
MTP trace records remain exact after the MoE changes. Resident MoE occupancy tiles and parallel
integer prefix scan pass the same oracles, with whole33 layer0 latency1.987→0.993ms and
natural one-token continuation163.616→145.600us. Optional row-FP8 endpoint/final-GR/PLE paths
pass independent represented-weight checks; FP8 endpoints additionally pass real C4 MTP/DFlash
graph/state/replay integration. Source-loss assessments are separate; uncalibrated NVFP4 head
is rejected (13.011% source-logit loss versus3.595% FP8). DFlash guessed protection recipes
remain rejected; BF16 source execution is retained. Both Astra reviews are clear.
PLE graph/W257 overlap passed exact output comparisons and independent trace review: all126
measured async rounds were ready before layer0 finished. All focused implementation and
complete-table checks are finished. The full correct-container C++ build and suite passed:
**135 selected, 126 passed, nine opt-in artifact skips, zero failures**, 592.23 seconds.
Command: `NINFER_DEV_CONTAINER=ninfer-builder-qwen4-mixed NINFER_BUILD_VOLUME=ninfer-build-cache-qwen4-mixed NINFER_DEV_JOBS=8 ./scripts/run-unit-tests.sh`.
The actual-weight projection, MTP target-input, FP8 endpoint draft, native precision-policy
and complete FP8 PLE residency checks were also run separately with their required artifacts.
Seven affected Python converter tests, kdev recipe/bound self-tests and affected Python
compilation pass. No numerical criterion was widened, and no commit or push was made.

### Completed goal: resident-component performance and precision (2026-09-19)

Base implementation is committed as `c307678c`. No new commit/push is authorized.
This goal uses existing authentic preview components; full-model PPL, acceptance and
throughput remain unavailable until an exact compute core fits the GPU.

- [x] Begin PLE row addressing/gather/H2D immediately when validated input IDs/history are
  known; overlap transfer with independent GPU work and wait only at decoder index1 (the
  second layer). Preserve complete eager RAM locking, bounded pinned storage, exact NVFP4/
  FP8 decode, graph ordering, accepted-prefix history and teardown/cancellation lifetime.
  Measure the consuming layer's remaining wait, not merely asynchronous API submission.
- [x] Profile authentic resident NVFP4/FP8 projection, MoE, GDN and QSA components separately
  from diagnostic streamed-expert overhead; fix measured worthwhile bottlenecks using the
  kernel-iteration classifier before any kernel speed change.
- [x] Localize remaining MTP propagated error and DFlash NVFP4 quality loss; evaluate targeted
  higher-precision exceptions against unchanged independent mathematical oracles.
  MTP stem attribution led to a same-schedule FP32/fused-add candidate with measured traffic
  reduction, but actual propagated seed error worsened 2.052%→2.130% with no latency win;
  candidate deleted and original arithmetic/gates restored. DFlash FC/KV, all-attention and
  MLP protections on authentic C8/16/24 features did not rescue the uncalibrated NVFP4 head.
  Neither the original 2% sensitivity screen nor draft quality/PPL is claimed resolved.
- [x] Evaluate selective A4/A8 against protected A16 with authentic weights/inputs, without
  converting component evidence into blanket activation/default or whole-model PPL admission.
  Resident native-33 whole/chunk and same-input EOS-65 measurements pass unchanged oracles;
  GDN Z-A8 saves 11.7%, while concentrated routed A4 is 2.446x faster but retains 9.29%
  represented-reference error under its existing 16% profile. QSA stays A16. Existing thirteen
  native FP8 Linear measurements are retained rather than repeated. Exact methodology,
  propagated-input/routing caveats and the next resident-MoE aggregation priority are in
  `docs/research/qwen4-native-fp8-source.md`; no new kernel/default admission follows.
- [x] Extend existing streamed/long-sequence diagnostics only for concrete remaining
  correctness/performance questions, or document why existing evidence resolves them.
- [x] Complete independent Astra review, relevant numerical/state/graph/lifetime checks,
  affected authorities and the full correct-container C++ regression gate.

Final verification: Astra reviewed the retained PLE implementation/tests, independently
recomputed all54 readiness margins from the CUDA trace, and cleared the retained precision/
timing changes. The full correct-container C++ build and suite passed: **134 selected,
126 passed, eight opt-in artifact skips, zero failures**, 533.61 seconds. Command:
`NINFER_DEV_CONTAINER=ninfer-builder-qwen4-mixed NINFER_BUILD_VOLUME=ninfer-build-cache-qwen4-mixed NINFER_DEV_JOBS=8 ./scripts/run-unit-tests.sh`.
Separately, the complete locked-table/real-prefix4 test, native33 A16/A8 and EOS65 component
oracles, actual-target restored MTP replay, and 22 Python reference checks pass. The restored
MTP trace matches its pre-experiment public outputs/state exactly. No precision gate was
widened: the MTP2% propagated screen and DFlash NVFP4 quality concerns remain explicit, not
misrepresented as model-quality admission. No commit or push was made in this goal.

No new full streamed-model trace is needed for this tranche. MTP and DFlash attribution reuse
the already captured authentic target-state panels, so another quantized/offloaded rollout would
not isolate the live precision question. Existing paged QSA independent tests cover selection at
2051/2115/4099/4101 and 262144 visible positions, ties, compact batches, masks, mapped pages,
BF16/NVFP4 represented caches and positions near 262143. PLE ordering is checked directly at
its new stream/graph boundary rather than inferred from plausible generated text. None of this
is full48 long-context/PPL qualification; that remains fitting-checkpoint admission work.

### Completed implementation goal: native architecture and Engine integration (2026-09-18)

The completed bounded tranche below is component evidence, not completion of the requested
implementation. The user explicitly requires the remaining feasible implementation. The full
preview's device footprint blocks full-checkpoint execution and PPL on this GPU; it does not
block writing and qualifying the missing native runtime. No commit or push is authorized here.

- [x] Bind and convert one complete exact native artifact: all 48 distinct source layers,
  endpoints, protected controls, complete locked PLE, source frontend assets, Vision and MTP,
  with explicitly optional checkpoint-specific DFlash and honest capacity admission.
- [x] Implement paged QSA including K/V, index keys and positions, long-context selection,
  compact batches and per-column causal visibility; preserve exact selection/tie semantics.
- [x] Implement compact batched GDN and PLE state transitions, masked prefixes and replay,
  using one layer-major decoder schedule rather than serial per-request model execution.
- [x] Implement full native embedding, 48-layer prefill/decode, final projection and sampling,
  with all ordinary compute weights GPU-resident and padded vocabulary rows excluded.
- [x] Own all 36 recurrent and 12 QSA states, PLE histories, continuation and sampling counts;
  implement prefix commit/retention/restore and address-stable graph execution.
- [x] Integrate native MTP and checkpoint-specific DFlash proposal/verification/replay with
  the shared p-less epsilon/floor law. Keep GPU proposal egress and locked-RAM PLE gathering
  outside captured target graphs; retain only accepted feature/cache prefixes.
- [x] Join source-correct Vision preprocessing/encoding/placement to Text and the existing
  Engine scheduler; share frontend ownership without copying another family's runtime.
- [x] Qualify independent numerical/codec/state oracles, realistic bounded source compositions,
  Engine admission and existing-product regressions; resolve actionable precision findings
  without relaxing criteria or asserting unavailable whole-model quality evidence.
- [x] Complete Astra finished-code review/fix loops and the full correct-container C++ suite;
  update affected ownership, model, artifact and operation authorities consistently.

Implementation qualification is deliberately distinct from full-artifact execution: the native
converter/binder is checked against every real source header and bounded real transforms, not an
unavailable complete conversion run. The shared production decoder body passes actual layers0–3
against the independent accumulated formula, compact B4 eager/graph exact state, and real65-row
chunked-prefill route comparison. Native MTP/DFlash shared owners pass actual-component C4,
shared-page admission, source-aligned reseeding and two distinct same-frontier checkpoint images.
All identified Astra findings are fixed and re-reviewed, including high-position QSA/DFlash
phase accuracy, unexecuted-prepare cancellation, and scoring against retained peer-page owners.
The full C++ regression gate passed in `ninfer-builder-qwen4-mixed` on RTX5090/CUDA13.1:
133 selected tests, 125 passed, eight explicitly artifact-dependent skips, zero failures,
543.36 seconds. The command was
`NINFER_DEV_CONTAINER=ninfer-builder-qwen4-mixed NINFER_BUILD_VOLUME=ninfer-build-cache-qwen4-mixed NINFER_DEV_JOBS=8 ./scripts/run-unit-tests.sh`.
Nine focused converter/metadata Python tests also pass. Full48 PPL, acceptance and throughput
still require a fitting exact checkpoint; none is inferred from bounded tests. This broader
architecture plan remains the admission roadmap for that external checkpoint, not an assertion
that its future model-level qualification has already passed. No commit or push was made.

### Completed bounded goal: native runtime, MTP and checkpoint-specific DFlash (2026-09-18)

The user requests all follow-up architecture work after the native mixed-precision tranche,
including a primary-source Hugging Face search for a compatible Flash-Next DFlash drafter.
The starting branch is `21b0b867`, rebased on experimental `e04fad37`. Shared sampling now
contains p-less epsilon slack and its unrelaxed `1/1024` probability floor; ordinary and
speculative paths must retain that single sampling law. No commit/push is requested for
this new goal.

- [x] Find and pin an exact Qwen4/Flash-Next-compatible DFlash/DFlash2 checkpoint, audit
  training target, feature taps, source mathematics, tokenizer/vocabulary and tensor inventory.
  Implement the actual checkpoint architecture, not a resized dense-27B drafter. Main draft
  matrices should use independently qualified NVFP4; preserve BF16 codebooks if the selected
  architecture actually has them, and protect source-required controls.
- [x] Freeze executable MTP recurrence, QSA-selection reuse, cache positions and observable
  casts from primary implementations; acquire the bounded authentic private block and
  independently qualify its NVFP4 conversion and higher-precision exceptions.
- [x] Implement complete bounded MTP draft execution, carried four-stream state, independent
  draft cache ownership and multi-step selection reuse. Qualify complete local formulas and
  state transitions, not just the stem. The separate propagated-chain/target quality admission
  gate remains failed/unqualified as recorded below; this check is not rollout admission.
- [x] Implement the compatible DFlash draft model and source-defined target feature capture,
  context/draft cache and position behavior, independently qualified logits/proposals and
  NVFP4 execution. Do not invent a codebook for a drafter that has none.
- [x] Implement provisional target/draft state, verification, rejection/partial/full acceptance
  and accepted-prefix fold across QSA, GDN, PLE/token history, positions and published tokens.
  Preserve common p-less/epsilon/floor, masks, correction/bonus and RNG semantics. Native target
  composition is qualified on actual layers 0–3; full48 target/drafter integration remains an
  external capacity/admission gate, not a claim made by this bounded transaction.
- [x] Complete feasible native artifact binding and fixed Qwen4 Text model scheduling beyond
  the diagnostic GGUF verifier; actual native layers 0–3 are bound and executed without runtime
  repacking. Keep complete product GPU residency and a complete exact inventory as full48
  admission conditions; do not substitute repeated layers for absent weights.
- [x] Complete source-defined Vision schedule/frontend and visual placement, using the
  qualified native BF16 block/merger and independent frontend/position references.
- [x] Integrate address-stable CUDA Graphs, startup-fixed C=1..4 request isolation, prefix
  retention and restore with the bounded native Qwen4 transaction/state ownership model.
- [x] Qualify full native PLE eager population/OS locking and bounded transfer lifetime using
  an actual chosen complete table if accessible within the local resource envelope; selected
  rows do not establish full-payload capacity. Never fall back to inference-time disk paging.
- [x] Evaluate actionable compression/precision alternatives without admitting the previously
  failed KV/residual profiles or weakening numerical gates. Real diagnostic-target DFlash
  features expose substantial NVFP4 weight loss; protecting only feature fusion reduces but
  does not remove it, so neither recipe is admitted as a quality default. Fitting-artifact
  memory/performance and final precision policies remain external model-level gates.
- [x] Complete independent codec/Op/state/sampling-law checks and bounded real-source
  compositions, full C++ regression suite, and Astra finished-code review/fix loops.

Current bounded implementation evidence (not whole-target admission):

- The exact PixelML five-layer DFlash v1 companion is acquired and converted. It has no
  DFlash2 codebook. Source BF16 and converted NVFP4/A16 schedules, learned-GR feature taps,
  accepted-context-only caches, shifted positions, all seven query outputs, shared authentic
  248320-row head, exact proposal selection, C=4 isolation and CUDA Graph replay pass focused
  checks. Same-input FP64 checks cover 604 boundaries. An avoidable BF16 attention split-partial
  rounding was corrected with FP32 D256 partials; D128 is unchanged. Five raw RMS gross screens
  remain explicitly failed, with independent proofs that their entire outputs are exactly the
  nearest BF16 values and those thresholds are unattainable in BF16. Public-boundary composition
  passes; the separate BF16 C65/K7 no-cast ideal 2% screen still fails at 2.4304%. NVFP4-versus-BF16
  draft quality and real-target acceptance are not qualified. Astra found no actionable issue in
  the finished slice. See `docs/research/qwen4-speculative-sources.md`.
- Native MTP binds the authentic private block and implements four-stream carry, complete
  frozen-domain selection, independent caches, discard/replay and target-hidden accepted replay.
  Actual-input local oracles and state/selection tests pass; an Astra-found historical-cache
  oracle input mismatch was fixed and the re-review is clear. The independently propagated
  chain retains seven failed 2% screens, including one changed expert membership; no rollout
  quality or acceptance-rate admission is inferred. Startup-captured draft graphs and four
  interleaved private owners sharing weights match independent eager outputs and complete
  caches exactly; Astra review is clear. See `docs/research/qwen4-mtp-execution.md`.
- Complete 27-block BF16 Vision and source-defined image/video preprocessing pass source
  references, graph and isolation checks. Fused projection+bias removed an erroneous extra BF16
  cast without widening the gate. Astra review is clear. This is not multimodal Engine registration.
- All 128 real NVFP4 PLE partitions now form a 28,800,138,752-byte canonical payload. The real
  resident-host materializer populated every page and eagerly locked 28,800,139,264 bytes;
  all partition boundaries matched an independent source-to-GPU decode oracle, with bounded
  staging and successful owner-lifetime/teardown tests. See `docs/research/qwen4-native-ple-source.md`.
- Qwen4's expanded 48/48-head replay and 36-layer accepted-prefix fold pass independent FP64
  recurrence, exact zero-accept/history, request-isolation, tree/path and CUDA Graph tests.
  The complete GDN layer emits independently checked raw replay records. PLE prefix-state
  recording/commit passes independent exact tail/history oracles at W1/7/16, including accepted
  EOS and zero-prefix no-op; Astra review is clear.
- Native binding and scheduling now run actual layers 0–3 with offline exact GDN permutations,
  resident native weights, independently qualified layer boundaries, whole/chunk continuation,
  C4 graphs and exact comparison with independent C1 owners. Astra review is clear. This is not
  a fabricated full model or full48 qualification. Native transaction publication/rollback,
  retained complete state, common sampling counts and typed graph ownership pass independent
  state/control checks and C1–4 graph/eager comparisons. A stale-output publication defect was
  fixed by requiring actual eager execution or validated typed graph launch; capture alone and
  repeated launches cannot publish. Astra's final transaction review is clear.
- A bounded header-only audit independently validates all 48 native compute inventories:
  295,968 tensors and 75,222,107,904 non-PLE payload bytes, before endpoints or runtime state.
  This alone exceeds the 5090's VRAM. Nine exact header ranges total 39,984,296 bytes; no
  full native weight payload was downloaded. Three parser regressions and Astra review pass.
- Full C++ regression on the correct `ninfer-builder-qwen4-mixed` container passes:
  126 standard CTests listed, 121 passed, five prerequisite skips, zero failures (505.17 s).
  The skipped Qwen4 endpoint, full PLE residency and transaction cases have separate successful
  real-fixture runs; the other two skips are existing target load-plan prerequisites. The
  Qwen4 Python suite passes 113 tests with two optional external-codec cross-check skips;
  the subsequently added three native-inventory tests also pass. New acquisition/reference
  Python modules compile. The subsequently added diagnostic feature and staged-round targets
  build and pass focused execution/default-skip checks; their finished Astra reviews are clear.
- Real diagnostic-target DFlash contexts 8/16/24 give BF16-versus-NVFP4 hidden relative L2
  0.33629/0.33771/0.36222, logit relative L2 0.13069/0.14516/0.16288, and 16/21 top-1 matches.
  Those differences are weight-precision diagnostics, not kernel-oracle failures or PPL.
  Independent FP64/public-BF16 C24 ablation changes hidden drift from 0.361924 to 0.315941
  when only feature fusion stays BF16; no new runtime profile is admitted.
- A complete staged diagnostic DFlash round uses its target's exact shared endpoints, seven
  real proposals, eight target verification columns and unchanged common p-less/epsilon
  acceptance. One draft is accepted in the observed round. Cancellation and partial/full
  publication reproduce all 157,147,144 bytes of continuation state exactly after accepted-input
  replay, with exact count rollback and next-anchor continuation. This proves a bounded full48
  integration, not native product scheduling, model acceptance quality, or performance.
- Actual diagnostic-target MTP handoff now pairs pre-final-GR `R_t` with the next token's
  embedding at private position `t`, as pinned source code requires. The 24-row seed, two
  feedback steps, shared endpoint sampling/count probe, graph/eager execution and actual
  target-hidden replay prefixes 0/1/2 pass. The independent same-input FP64 oracle reports
  zero failures; QSA relative L2 is at most 1.100%, MoE per-row at most 0.373%. A separate
  propagated-chain diagnostic still fails its unchanged 2% screen at seed token 13 (2.05172%);
  draft carry is at most 1.2367%. Legacy injection gross screens retain their independently
  proven nearest-BF16 limitations. Accepted replay inputs came from the DFlash round, so this
  does not claim MTP acceptance qualification. Focused build/GPU/default-skip and Astra review
  pass; see `docs/research/qwen4-mtp-execution.md`.

The feasible bounded implementation/qualification tranche is complete. No commit or push was
made. It does not declare the failed or unmeasured model-quality gates passed. Full native
48-layer execution, registered scheduling, native-target MTP/DFlash acceptance/PPL and final
activation/weight defaults need a fitting exact checkpoint (the audited preview core alone is
70.06 GiB), or a separately authorized change to the product's GPU-residency contract. A future
smaller target's shape/inventory cannot be inferred or fabricated from the preview.

Exact unreleased-model identity/topology, complete-model paired PPL/task/long-context quality,
end-to-end speeds and final registration need suitable actual artifacts. They remain requested
external qualification gates, not evidence obtainable by inventing a future checkpoint or
substituting component error for model quality. Continue independent implementation work while
one artifact-dependent gate is unavailable; report concrete blockers only after feasible paths
are exhausted.

### Completed goal: native NVFP4 mixed precision (2026-09-18)

The new goal supersedes the earlier tranche's completion claim as a statement of overall
native readiness. That tranche's checks remain valid bounded evidence; its rejected A8 and
FP8-residual profiles do not close the broader candidates. The design authority is
`docs/research/qwen4-quantization-recommendations.md`, supplemented by the native precision
source audit. Weight storage, temporary activation operands, accumulation and persistent
state are separate decisions. GGUF/IQ/K-quants remain historical diagnostic formats only,
never defaults for the future native model.

- [x] Separate native-facing precision/state contracts from historical verifier defaults;
  audit native protected-weight storage and actual source cast/control boundaries. Projection
  workspace profiles are explicit; native router/scalar controls remain BF16 and small
  effective-FP32 control views are documented source-faithful transformations, not GGUF defaults.
- [x] Implement and independently qualify BF16 QSA KV as the native reference. Actual-source
  widths 1/5/27/28/128/129 and both-cache selected-layer T5/T17 whole/chunk chains pass;
  exact append and independent attention oracles cover short/tiled/selection boundaries.
  Astra finished-code review found no material issue.
- [x] Assess FP8 KV separately, retaining NVFP4-G16 only as an explicitly selected
  experimental/diagnostic compression profile until quality qualification. Source-weight
  T129/128+1 decoded-consumer checks pass unchanged gates; K-only/V-only/both storage loss
  is measured independently. BF16 remains the reference; no runtime FP8 codec is admitted.
  Astra finished-work review found no material issue.
- [x] Pin/audit native NVFP4 PLE codes, block scales and per-shard multipliers; implement
  source-faithful artifact storage, RAM-resident packed-row staging and GPU BF16 decode
  alongside FP8 PLE. Preserve complete eager OS locking and bounded transfer lifetimes.
  Exact synthetic/authentic codec and complete native injection checks pass; Astra review
  found no material issue. Full-table capacity on unavailable native payloads remains deferred.
- [x] Independently evaluate FP8-weight/A16 and calibrated FP8-weight/A8 for shared experts,
  QSA and large GDN projections using authentic publisher calibration and bounded native-input
  qualification panels. Preserve
  NVFP4 expert scales and separate post-SwiGLU calibration; measure relevant dispatch widths.
  Authentic publisher scales are preserved, not independently reproduced calibration: six
  GDN policies and all eight shared policies pass bounded component gates; QSA is A16-only
  after rejecting all A8 candidates under unchanged screens. Astra finished-code review is clear.
- [x] Re-evaluate a research-grounded FP8 residual-storage profile separately from gate math;
  distinguish codec/Op correctness, quantization error and model-quality uncertainty. Repeated
  four-layer storage is rejected; no GPU FP8 residual codec/default is admitted. Other calibrated
  storage recipes and full-model quality are not inferred from this bounded assessment.
- [x] Qualify accepted native component compositions, state/selection and prefill/decode
  continuation; measure changed PLE consumer latency and relevant 5090 performance. The
  contiguous native first block (layers 0..3) passes synthetic and native-text whole/chunk
  checks with BF16 cache; source-calibrated FP8/A16 and selective guarded A8 also pass.
  PLE transfer/consumer timings are measured for both native table formats, and guarded
  A8 dispatch timings and boundary qualification are complete. Real-EOS expert-A4 tests
  pass unchanged whole/chunk gates; independent calibration attribution does not justify
  replacing the source scale with a guard (17/20 dots improve, three and worst-case worsen).
  Native token/hash-selected row panels are a distribution check, not full-model calibration.
- [x] Record concrete policies and deferred gates for protected roles, endpoints, vision,
  MTP and the actual future checkpoint, without inventing topology or copying dense-model
  sensitivity lists. The native-preparation matrix in `docs/maintainer/qwen4-artifact.md`
  separates source storage, admitted operands/state, rejected candidates and future gates.
- [x] Following the request for the most complete feasible architecture preparation, close
  bounded native endpoint gaps: audit/acquire the actual BF16 head and final GR, admit the
  exact BF16 Linear geometry, and qualify final read/head composition independently.
  All-vocabulary source checks and final-read composition pass unchanged gates; Astra is clear.
- [x] Close the source-defined Vision merger component gap with an independent oracle and
  a bounded native fixture; retain BF16/A16. Synthetic/native whole/chunk checks and Astra
  review pass. This does not admit a multimodal product target.
- [x] Close the remaining native BF16 Vision projection domains (patch, QKV, attention
  output, MLP up/down), including the real 4304 row/reduction tails. Qualify a bounded
  source patch/position/encoder-block fixture with existing surrounding Vision Ops and
  independent mathematical oracles; no runtime weight repacking or new generic Vision graph.
  Source P4/12/132 passes unchanged 2% gates (relative error 0.50–0.57%); exact synthetic
  tail and segment-isolation witnesses pass. Astra finished-code review is clear.
- [x] Qualify the source-defined two-projection/four-stream MTP stem as a bounded component,
  if its pinned formulas and source inventory suffice. Do not claim full speculative MTP
  admission without independent executable cache/selection/accept-fold traces. Native source
  and unequal-branch synthetic checks pass the complete FP64 oracle; Astra is clear.
- [x] Run focused and full C++ checks and resolve an independent finished-work **Astra**
  review-and-fix loop (user selection replaces Sol for this goal). On 2026-09-18,
  `./scripts/run-unit-tests.sh` in `ninfer-builder-qwen4-mixed` completed all 118
  standard CTests: 116 passed, two missing-artifact load-plan skips, zero failures
  (497.05 seconds). The 23 focused native Python tests, acquisition-tool compilation,
  native-source component/composition checks and final whitespace review pass.
  Astra's finished-code reviews, including guarded A8, Vision tails and the EOS-A4
  attribution, have no material open finding. No commit or push was made.

Numerical admission is mandatory for every changed route: one independent mathematical
FP32/FP64 oracle per floating Op, exact independent codec/layout oracles, explicitly audited
public casts/state boundaries, and criteria declared before measuring candidates. Exercise
real shapes and native weights, cancellation/outliers, codec rounding/saturation boundaries,
exact routing/selection where applicable, dispatch tails, and state/alias/lifetime transitions
that the change affects. Pairwise kernel parity and plausible text do not replace the oracle.
Diagnose failures; never silently widen a failed gate. Distinguish source weight loss,
activation approximation and cache compression from implementation error. These checks cannot
prove absence of every bug or substitute for future paired full-model quality measurements.

Full-model PPL/task/long-context quality, unavailable full-table payload capacity admission,
future target registration and final Pareto/default selection remain explicit deferred gates.
No commit or push is authorized by this goal.

This bounded native-preparation goal is complete. Its completed ledger does not claim
full-model execution, a full multimodal frontend, or speculative MTP state/acceptance
admission; the exact future target and those model-level gates remain separate work.

Current native PLE checkpoint: `NVFP4_PARTITION_F32M` preserves per-shard scale domains;
synthetic GPU gather/codec tests at T1/17/128/4096 and twelve authentic rows at T1/3/17
pass exact independent byte/codec checks. Artifact reader tests and eight focused Python
partition/source tests pass. Native PLE injection additionally passes actual-source full injection at T1/9/10/17/27/28/29
for both packed-table formats, with nonzero history, reset, chunked continuation and in-place
aliases against the independent complete formula. Selected-layer composition and finished
Astra review pass; contiguous four-layer BF16-cache composition also passes as described below.
Native resident MoE now
retains BF16 router/shared-gate weights without FP32 device expansion; focused synthetic and
actual-source A16/A4 regressions pass unchanged gates, with no material Astra finding.

Expanded-chain checkpoint: the contiguous native 0..3 BF16-cache T5/T17 cells pass, but
the experimental NVFP4-cache cells exceed the unchanged accumulated 2% gate at layer 3,
including T17 whole/chunk comparison. Local component gates pass. The state-codec/upstream
rounding decomposition is complete: same-input codes/scales are exact; upstream rounding
crosses codec decision boundaries. Do not describe that wider compressed chain as qualified
or relax its criterion. Native FP8 Linear's thirteen authentic projection fixtures, exact
activation packing, old FP8 regressions and source-calibrated dispatch tests pass, with
Astra review clear; closed GDN per-role A8 integration is also implemented and qualified below.

Further checkpoint: calibrated GDN has six independently passing per-role policies at
T65/64+1; simultaneous QKV/output A8 exceeds the predeclared gate and is not admitted.
Calibrated QSA A16 and shared-expert per-role integration are qualified. The repeated FP8 residual-store
experiment across native 0..3 reports 8.34% GPU drift and 8.37% independent formula storage
loss versus BF16, failing the unchanged accumulated screen; no runtime FP8 residual dtype
is enabled. A bounded native text panel now contains 33 tokenizer-selected BF16 token
embedding rows and 528 exact hash-selected FP8 PLE rows from the pinned NVIDIA source.
Its complete first-block and exact hash-continuation checks pass both whole33 and32+1,
for source BF16 and separately calibrated FP8/A16 projections under unchanged gates. It
is an input-distribution witness, not a full-model calibration or PPL corpus. A selective
GDN-Z/shared-expert guarded A8 composition passes the same whole/chunk criteria;
QSA calibrated projections are A16-only
after all four A8 role candidates failed the predeclared closed-component screens.

The first real-text selective-A8 attempt exposed a source-calibration mismatch for this panel: one GDN
input at token 15 is 49.75 while the static source scale only covers 16.875. Independent
clamp/rounding/projection attribution establishes clipping, not an implementation or scale-
direction defect. Its failed local/accumulated checks remain recorded. A guarded per-token
activation scale, floored by the unchanged source calibration, now passes exact packing,
component and real33 whole/32+1 tests against the same criteria. The isolated Z projection
error falls from 21.2924% to 0.667229%; it is a new private arithmetic profile, not a claim of
publisher bitwise equivalence or future-model PPL.
Packing-inclusive timing retains useful guarded A8 wins; K/V and shared gate/up keep
A16 through T24 and first permit A8 at T25. Focused boundary regressions and Astra
finished-code review pass; no source scale words or numerical gates were changed.

The additional architecture audit found no missing main Text mixer topology. It did find
that generic BF16 embedding gather already covers width 2560, but BF16 Linear did not admit
the preview head's `[248320,2560]` geometry; final GR already has an independent oracle.
The exact Vision merger had a Python oracle but no central closed Op; its new central Op
and native fixture now pass. Existing `mtp_fc` and `mtp_pack` do not implement the preview's
separately normalized token and four-stream stem; the new `gated_residual_stem` closes that
bounded component with an independent complete oracle.
The bounded tasks above supersede the earlier endpoint/merger/stem expansion deferral only;
future exact identity, full-model quality, multimodal product integration and speculative
transaction admission remain separate gates. No architecture is inferred for an unreleased model.

### Previous bounded tranche (2026-09-12 through 2026-09-18)

This tranche prepares Qwen4 NVFP4 execution before a smaller, fully resident checkpoint is
available. It does not register the oversized preview, replace the supported Qwen3.8-27B product,
or promise that future checkpoints inherit this preview's dimensions or precision recipe.
The active implementation goal now includes the native-format stages below, informed by
`docs/research/qwen4-quantization-recommendations.md` (2026-09-18). Their gates are not claims
that those stages are already implemented. Publisher quality results are research evidence,
not reproduced qualification results or authority to change protected state boundaries.

### 0.0 Active implementation ledger and precision decisions

- [x] Incorporate the research recommendations into the implementation order and gates.
- [x] Close the pending PLE residency implementation unit verification: 2026-09-18 baseline
  full suite passed 112 tests, with two missing-artifact skips. Full IQ4_NL table admission also
  passed in a temporary builder with unlimited memlock: 28,800,139,264 bytes OS-locked and
  26,538,652,160 device payload bytes loaded. Native changes require a new suite run; the larger
  authentic FP8 table's capacity gate remains distinct.
- [x] Establish exact native NVFP4 code/scale semantics and per-expert ownership, and acquire
  bounded real layers/rows with all required controls.
- [x] Implement missing BF16 component roles and native FP8 PLE decode/gather; retain packed
  FP8 in locked host RAM and transfer selected packed rows for GPU decoding to BF16.
- [x] Qualify native represented-weight A16 closed layers and continuation/state boundaries.
- [x] Qualify expert W4A4 independently, with separate post-SwiGLU down calibration; retain
  A16 dispatch when it is faster at small expert occurrence counts.
- [x] Evaluate selective W8A8 QSA/GDN/shared projection candidates after baseline fidelity,
  retaining only routes passing closed-layer and short-sequence gates.
- [x] Evaluate FP8 gated-residual storage separately from gate arithmetic, subject to concrete
  codec/state criteria and measured benefit; a rejected candidate also closes this experiment.
- [x] Complete focused and full-suite verification and independent finished-work Sol review.

Expert W4A4 is the first aggressive compute candidate. W8A8 has stronger direct preview
activation evidence than W4A4 for shared experts and large QSA/GDN projections, so evaluate it
next; W4A4 remains a later candidate, not a blanket default. Weight-only ablations do not qualify
A8 or A4 activation operands. Ordinary observable activations remain BF16; local GEMM operand
packing does not change persistent storage. Keep router/indexer decisions, GDN FP32 recurrence
and controls, PLE content gates, and hyperconnection gates protected. Hyperconnection read
projections remain BF16/A16 initially; eight-bit weights are a separate candidate, not permission
for A8. Vocabulary endpoints, vision and MTP expansion are deferred until an exact target
requires them, not additional tasks in this tranche.

The source-BF16 QSA cache is the quality reference. The authorized verifier's NVFP4-G16 cache
remains unchanged; kernel/state correctness there does not establish maximum-quality cache
compression. Projection W8A8 evidence does not qualify FP8 attention operands or KV storage.
FP8 residual storage must not lower gate computation or GDN recurrent precision. Compare
candidate compute against represented-input mathematical oracles, and keep source weight loss,
activation loss, and persistent-state compression loss distinct. Full-model paired PPL and
future-checkpoint default selection remain deferred until a suitable complete model is available.

The first FP8 residual-storage experiment is rejected, not enabled: actual layer-0/3 attention
and MLP GR weights at T=33, per-token/per-branch FP32 scale `maxabs/448`, E4M3FN nearest-even
storage, and BF16 decoded public values. Storage would be 50.0781% of BF16, but unchanged read
relative-L2 gate `0.006` sees `0.02887..0.03175`; unchanged write gate `0.0035` sees
`0.00408..0.00746`. These compare the same independent GR formula with original versus
decoded residual inputs while preserving gate arithmetic. This rejects that bounded storage
profile under existing criteria, not all possible FP8 residual strategies or the paper's
recipe. No production residual dtype or default changes, and no speed qualification follows.

### 0.1 Baseline and exact native source

Commit `42f0024b` provides independently qualified NVFP4 and row-scaled FP8 Linear/component
routes, complete synthetic GDN/GR/QSA/PLE/MoE fixtures, and a 23-matrix real-weight fixture.
That fixture was requantized from the lossy GGUF. It proves represented-kernel arithmetic on
those distributions, not compatibility with NVIDIA's packed checkpoint or source-BF16 quality.
The current full verifier still binds the exact GGUF-derived artifact, not a native NVIDIA model.

Use `nvidia/Qwen3.8-Flash-Next-NVFP4` revision
`fc694b54fb0174e0913e6adf86691ef85a4ead47` as the concrete native source. Its published recipe is:

| Component | Published storage/compute profile | Required decision |
|---|---|---|
| Main routed experts | NVFP4 W4A4, MSE-calibrated weights | Audit stored codes, block scales, global weight/input scales per projection/expert; do not assume one scalar per bank |
| Attention/GDN, shared experts and remaining main-model projections | BF16 | Add the exact missing BF16 component domains; preserve source protection before evaluating alternatives |
| PLE table | Per-tensor FP8 | Audit E4M3 variant, scale dtype/direction and exact table/shard layout; distinct from row-scaled Linear FP8 |
| MTP routed experts | 128-by-128 block-scaled FP8 | Not supported by row-scaled FP8 automatically; defer execution until MTP is separately admitted |

Bounded header/scalar reads confirmed that complete main layers 0 and 3 contain 6,166 tensors
each and respectively 1,570,383,296 and 1,557,359,104 bytes (~2.913 GiB together, excluding
workspace). Actual BF16 domains include GDN QKV/z/output, QSA Q/K/V/output and fused index-QK
`[640,2560]`, GR `[320,10240]`/`[10240,320]`, and shared MoE gate/up/down. Native composition now
admits these BF16 roles alongside NVFP4/row-FP8, with independent real-weight component tests.
This is implemented arithmetic support, not merely an artifact-name translation.

In layer 0, gate/up `weight_scale_2` has 218 distinct FP32 values across experts, and down has
281; layer 3 has 239 and 209. For example, layer-0 gate expert 0/511 multipliers are
`6.612142169615254e-05`/`3.106253643636592e-05`. Preserve the stored multiplication semantics:
conversion to a rounded reciprocal in the current divisor field is not exact represented-weight
equivalence. A native multiplier profile must have explicit codec/Op semantics and per-expert
ownership. Site-wide input-scale sharing is allowed only after validating equality; it held in
these sampled layers but is not an architectural invariant.

PLE headers contain 128 F8_E4M3 arrays `[2500012,160]`, totaling 51,200,245,760 bytes
(~47.684 GiB), plus one BF16 scalar `weight_scale=0.00019931793212890625`. A matching source
decode implementation audit now resolves scale direction, nibble ordering for experts, PLE
global-row/shard addressing and output casts. The exact producer is ModelOpt
`73d778422388f0e849ecb180375d34ac445711ca`, identified by the checkpoint's producer version.
`nvfp4_tensor.py` defines even-K low nibble and odd-K high nibble, with represented values
`E2M1(code) * E4M3FN(block_scale) * FP32(weight_scale_2)`. Exported `input_scale` is an activation
dequantization multiplier `amax/(6*448)`, not its reciprocal. Inference activation rounding and
underflow policy still require their own qualified compute profile.

For PLE, exporter `fp8_tensor.py` and the pinned SGLang consumer corroborate conversion to BF16
followed by BF16 scale multiplication and BF16 output. The 128 shards partition contiguous
global rows: `shard = global_row / 2500012`, `local_row = global_row % 2500012`. They are not
head partitions. Detailed pinned source references are in `docs/maintainer/qwen4-op-contracts.md`
section 7.1. Header facts alone were not used as proof of these formulas.

The recipe is a reference baseline, not a rule that sparse attention requires BF16 or that
every future PLE uses FP8. Main-model metadata/header inspection is the first binding gate.
The source index places main GDN layer 0 in `model-00001-of-00010.safetensors` and main QSA
layer 3 in `model-00002-of-00010.safetensors`; PLE is in `model-fp8-mtp-ple.safetensors`.
Filter exact main-model prefixes so MTP layer 0 is not mistaken for main layer 0.

### 0.2 Mandatory PLE residency prerequisite

Replace lazy page-cache dependence for PLE with an artifact-owned, eagerly populated, OS-locked
host tensor. A file-backed mapping is acceptable only while every PLE page is locked resident;
the source file is startup storage, never an inference-time row-fetch mechanism. Do not duplicate
the full table into a second allocation or CUDA-register the entire table. Keep bounded
CUDA-pinned staging for H2D rows and GPU-side decode/math.

Ownership and sequence: exact binder selects resident-host placement; artifact materialization
retains the mapping, locks the complete page-aligned payload before device upload/readiness,
and owns the lock through all consumers. Artifact owns the file-backed mapping/lock lifetime;
core retains raw CUDA staging/transfer ownership. Failure unwinds locks
and mappings and reports requested bytes plus the OS failure; there is no configurable lazy
fallback. Only PLE gains this requirement: the diagnostic routed-weight streaming exception
does not become a requirement to lock all 46 GB of mapped verifier payloads.

Readiness means synchronous lock/population succeeded, not an asynchronous warmup was queued.
`mlock2(MLOCK_ONFAULT)`, a successful `madvise`, or an observed warm page cache is insufficient.
Prove small-fixture resident/locked pages, byte identity, ownership after Reader destruction,
move/teardown cleanup, and fail-fast behavior with an insufficient per-process lock limit.
Run the full existing IQ4_NL PLE/Program route when resources permit; later repeat with the
complete authentic FP8 table. Never simulate full-table admission with only selected rows.

Resource admission is host as well as GPU: the existing table is 28,800,138,240 encoded bytes;
the preview FP8 table alone has 51,200,245,760 values before scale/alignment metadata. Check
RAM headroom, process/container memory limits, OS memory-lock limits, and staging/state budgets
before a large test. Do not kill services, disable swap globally, or force allocation under
unsafe pressure. If resources cannot support residency, the loader must fail and the real
capacity gate remains explicitly blocked pending maintainer action.

### 0.3 Native acquisition and representation gate

1. Read pinned config, quantization metadata, tensor index and bounded safetensors headers.
   Resolve source formula/scale semantics against the pinned ModelOpt export/decode implementation.
2. Extract selected main layers and all required scale/control tensors with verified HTTP byte
   ranges. Reject a server ignoring Range rather than accidentally downloading a whole shard.
   If full shards are necessary, estimate bytes and disk capacity first. Store prerequisites
   under `local_llm/models/qwen4-...`, never commit weights or cached activation panels.
3. Extend `tools/convert/qwen4`, `tools/reference/qwen4`, and `tools/parity/qwen4` for the exact
   native sample. Keep `.ninfer` as the C++ artifact; do not add a direct safetensors runtime.
   Preserve represented codes/scales exactly through offline layout transforms. No GGUF
   round-trip, lossy scale folding, or runtime repacking is allowed in this fidelity gate.
4. Audit whether the current complete-bank single-divisor layout can represent the actual
   per-expert values. If not, give the exact native bank a coherent stored scale representation
   and update converter, format/layout authority, binding, Op views, kernels and exact oracle
   together. Preserve source FP32 multipliers rather than rounding their reciprocals into the
   existing divisor representation. Do not silently normalize unequal global scales into
   quantized block scales.
5. Add BF16 projections and FP8 PLE only for the inspected roles/geometries. Per-tensor PLE,
   per-row Linear, and block-scaled MTP FP8 remain distinct registered meanings. Do not create
   speculative generic model graphs, arbitrary geometry registries or an unused MTP backend.

Gate: exact independent decode/layout checks over acquired objects, complete role inventory,
correct scale ownership and no unresolved representation loss. All bytes contributing to the
selected closed layer must be present, including router/shared/control weights.

Native expert A4 candidate gate, declared before its measurements: the A16 complete-MoE
criterion remains unchanged. The separately permitted A4 implementation uses the existing
Linear A4 profile's `0.16` relative-L2 and `0.16 * maximum_reference_magnitude` gross allowance,
with the MoE absolute floor `1/32768`, against the same complete ideal FP64 formula. This is an
engineering conformance envelope for four-bit activation arithmetic, not an acceptable PPL-loss
budget or a default-quality qualification. Three nonlinear projections may fail it; do not
increase it after seeing results. Report measured error, not just pass/fail, and retain source
calibration separately for gate, up and post-SwiGLU down. Exact routing remains protected.
The initial candidate reuses existing SM120 MMA tiles with GPU-only grouped occurrence packing
and scattering; small expert occurrence counts retain A16. No checkpoint/default activation
policy changes follow from this component experiment.

The initial opt-in source-bank profile grouped occurrences entirely on the GPU and used
W4A4 at 128 or more occurrences per expert, retaining A16 below that count. It reuses the
qualified SM120 MMA tiles and separately repacks gate, up and post-SwiGLU down inputs with their
stored per-expert multipliers. Synthetic complete-MoE tests through T=4096 and actual layer-0/3
expert-bank tests pass the declared oracle gates; standalone Linear A4 regression also passes.
On two deterministic BF16 input patterns, actual layer-0/3 A16 worst-token relative L2 is
`0.00316111` / `0.00364733`. At T=257 the active A4 groups reach `0.114184` / `0.0942673`.
With that initial 128-occurrence cutover, T=1/17/128/129 in these particular alternating-pattern
panels retained A16-sized expert groups; synthetic fixed-hot fixtures separately exercised the
127/128/129 cutover. The substantial A4
component error is explicitly not described as negligible accuracy loss.
These are component results, not full-model quality. Independent Sol source review of this
subset on 2026-09-18 found no substantive issues; final integration review remains open.

Initial complete resident-MoE CUDA-event measurements on RTX 5090 / CUDA 13.1.2, five timed
iterations after three warmups, source-format synthetic banks and NVFP4 shared projections:
T=128 fixed-hot routing took 390.982 us with opt-in A4 versus 1132.243 us A16. Rotating routes
(52 expert windows, small per-expert counts) took 4292.627 versus 4243.955 us: no A4 arithmetic
is selected there and dispatch adds overhead. T=512 fixed-hot A4 took 621.638 us versus
3798.893 us A16; rotating A4 took 4817.901 us versus 4765.933 us A16. These are public-Op
measurements including routing, packing and scatter,
not end-to-end decode/prefill speeds or an activation-policy default recommendation.
The benchmark is `ninfer_qwen4_sparse_moe_resident_bench --format nvfp4-expert-f32m
--policy a16|a4 --width T --iterations 5` in `ninfer-builder-qwen4-mixed`.

The final qualified private cutover is **32 occurrences per expert**, with A16Only still the
default. A bounded comparison of the existing MMA family's 32/64/128 cutovers chose 32;
there is one internal constant and no runtime tuning knob or alternate kernel family. Layer-0
`tools.kdev bound` admitted aggregation/weight-replay changes for these shapes; the calibrated
NVFP4 issue rate was `1.229813e11` MMA/s on 170 SMs. Five-iteration complete-Op measurements
with the same fixture, hardware and three warmups were:

| Workload | Cutover 128 (us) | Cutover 64 (us) | Final cutover 32 (us) |
|---|---:|---:|---:|
| T=32 fixed-hot | 376.038 | 376.960 | 231.571 |
| T=64 fixed-hot | 634.080 | 297.581 | 297.421 |
| T=128 fixed-hot | 390.982 | 391.194 | 390.688 |
| T=4096 rotating | 23862.317 | 5147.763 | 5146.675 |

Thus the balanced rotating prefill fixture improves about 4.64x without waiting for 128-token
expert groups. Small rotating T=32/64 remains A16 arithmetic and incurs about 46/54 us extra
dispatch/packing-launch overhead versus the initial 128 cutover (roughly 1.8%/1.3%); no claim
is made that every route distribution improves. Resident weights remain 1,423,607,820 bytes;
final workspace is 3,241,728/6,472,704/12,935,168/413,602,048 bytes at T=32/64/128/4096.
The synthetic complete oracle passes through T=4096 with new 31/32/33 witnesses. Actual
layer-0/3 tests pass unchanged criteria at T=63/64/65, explicitly counting groups of 31/32/33
occurrences and testing mixed A16/A4 execution. Their active A4 worst-token relative L2 remains
`0.114184`/`0.0942673`; unlike the initial profile, T=128/129 now activates A4 too. These
non-negligible implementation errors are not a checkpoint-quality endorsement or a default
activation-policy change.

### 0.4 Complete-layer A16 reference baseline and native W4A4 qualification

Build a bounded, unregistered layer harness at the actual preview shapes, resident on the GPU.
Start with one GDN+GR+MoE layer and one QSA+GR+MoE layer; include the PLE injection boundary
using authentic rows and then the fully resident table when capacity is available. Existing
Ops and tests own mathematics; the harness only owns exact binding, state and orchestration.

For the same stored NVFP4 weights, first qualify W4A16 as an activation-preserving diagnostic
profile. It is not the source checkpoint's advertised W4A4 arithmetic, nor a BF16-weight oracle.
Then implement/qualify the source W4A4 input scaling and compute profile. Evaluate both directly
against the same independent ideal FP64 formula from represented public inputs; their A/B is
additional evidence isolating activation quantization. Keep exact codec/transform checks exact.
Private production staging/reduction/casts never become the mathematical oracle.

Use deterministic ordinary/cancellation witnesses and bounded real activation panels. Panels
captured from the running GGUF verifier are useful public inputs, not native reference outputs.
Acquire matching BF16 source layers only if assessing weight-quantization loss; keep that result
separate from kernel error and A4-versus-A16 loss.

Predeclare per-Op output/state criteria and an accuracy manifest before A4 comparisons. Retain
existing criteria for existing represented-input profiles. If a candidate fails, preserve the
witness and diagnose routing/scale/rounding rather than widen tolerances to admit it. Explicit
near-zero absolute witnesses must remain separately labeled alongside strict normwise tests.
Record exact routing/selection structural invariants, top-k identities where the reference
margin determines them, and tie policy; quantify any A4-induced selection change rather than
mislabeling all valid top-k changes as a codec error.

Cover scalar decode, non-aligned prefill, one-shot versus chunked continuation, and the QSA
2048-token selection boundary within the existing 4096 verifier ceiling. Check output errors,
FP32 recurrence, BF16 convolution history, QSA KV/index contents/frontiers, PLE hashes/EOS
history, and residual branches at their specified semantic boundaries. The same-token sequence
must use the same external inputs and resets; compare each route to its oracle, not only to
another implementation.

### 0.5 Resident performance and selective precision expansion

Profile the complete layer first, without streamed expert weights masking device time. Use the
required `tools.kdev` admission/SM120 procedure before CUDA speed changes. Prioritize measured
NVFP4 MoE projection/occurrence-grouping bottlenecks; test realistic expert distributions as
well as fixed-hot cases. Distinguish total prompt width from per-expert token count.

Candidates include packed-weight replay reduction, gate/up fusion, bounded activation reuse,
and measured small-batch dispatch improvements. Retain only public-Op/layer gains that pass the
same mathematical gates and do not regress relevant decode/tail widths. Report arithmetic atom,
cache/routing conditions, transfer-inclusive versus resident scope, and timing statistic.

For PLE, measure startup loading separately from inference. Measure host gather, H2D completion,
and exposed layer-1 consumer wait—not only enqueue time. Exercise token/head order, repeated
rows, EOS and chunk boundaries with the table locked; cold storage is a startup test. Do not
evict locked pages to manufacture an inference benchmark. Add deduplication, transfer-stream
overlap or extra staging slots only when measured exposed latency justifies them and ownership
remains explicit. Keep model teardown ordered after all host fills/transfers/device consumers.

Only after source-profile fidelity, investigate additional FP8/NVFP4 attention/GDN/shared/GR/PLE
projections. Hold router/index selection and sensitive controls/state at their baseline precision
initially. Each accepted change needs the complete-layer gate and a short mixed-format sequence
gate; passing standalone Linear or copying Qwen3.8-27B cutovers is insufficient.

QSA/GDN selective A8 screening (2026-09-18, RTX 5090/CUDA 13.1): native layer-3 QSA matrices
were converted offline to nearest-even E4M3FN with BF16 row scales. The matrix weight-loss
relative L2 values were query/gate `0.0262857`, key `0.0263045`, value `0.025956`, output
`0.0263777`; these are weight errors, not activation/kernel errors. At T=17 with BF16 inputs
from seed 30303 uniform `[-0.2,0.2]`, the represented-FP8 A16 baseline passed the existing
complete-output and cache-state gates. Enabling A8 separately for query/gate or output failed
the unchanged complete-output criterion (`0.02` relative L2, `2.5e-4 + 0.02*max_reference`
gross bound); query/gate failed nine token checks, output all seventeen. Key-only and
value-only A8 failed the independently derived KV codec/state bounds. Indexer projections
stayed BF16. This experiment used exact decoded observed cache as the attention consumer input
plus independent cache-state bounds, not an entirely oracle-propagated sequence. All four
QSA A8 candidates and their public policy/temporary test switches were removed; QSA remains A16.

The native layer-0 GDN all-FP8 A16 baseline failed the existing near-exact packed-profile
convolution-history criterion: one witness was actual `0.1640625`, reference `0.1650390625`.
That is a private arithmetic/represented-history qualification limitation on real weights,
not proof of a wrong codec or source weight-loss result. No criterion was widened or baseline
retroactively labeled passing. Keeping QKV BF16 isolates Z/output experiments under the
already established BF16 history criterion. Output-only row-FP8 A16 passed with relative L2
`0.00165305`; output A8 failed at `0.0263308` against `0.009`, with maximum absolute error
`0.00639752` against `0.00155780`. Output A8 permission was removed.

**Rejected: GDN Z-only A8.** The GR-fed actual-source T=17 sequence fails the unchanged
local GDN gross-error criterion: maximum absolute error `0.00309234` exceeds `0.00289572`,
although relative L2 `0.005618` is below `0.009`. The candidate production permission, isolated
test switches and A8-only workspace change were removed. No criterion was widened. GDN remains
A16; the following isolated results are positive experimental evidence, not an admitted route.
Native layer-0 Z weights converted to row-FP8 have weight relative L2 `0.0263107`.
The complete represented-weight FP64 GDN oracle passed at T=1/12/13/17/65 with distinct state
and repeated-scalar continuation, unchanged output/history/recurrent criteria, exact rollback,
guards, and exact workspace high-water. At T=17 output relative L2 is `0.00166549` for A16 and
`0.00292297` for A8; at T=65 it is `0.00211623` and `0.00318382`, below the `0.009` gate.
T=65 recurrent error is unchanged at `0.000672458`, below `0.0065`.

Public complete-GDN timings on RTX 5090/CUDA 13.1, resident actual weights, three warmups,
twenty-call CUDA-event mean, including A8 packing, were:

| T | A16 Z microseconds | AllowA8 Z microseconds |
|---:|---:|---:|
| 1 | 66.3552 | 66.0480 |
| 12 | 102.144 | 102.451 |
| 13 | 112.128 | 101.890 |
| 17 | 130.355 | 113.051 |
| 65 | 259.635 | 241.766 |

T=1/12 uses the identical A16 dispatch; the small timing differences are run variation.
T>=13 used the existing qualified Linear A8 route in the now-removed candidate. These are
historical closed-GDN and time-continuation results; they do not establish multi-layer A8
sequence quality, full-model PPL, or a future checkpoint's qualified recipe.

Shared-expert W8A8 candidate outcome (2026-09-18): **rejected; shared projections remain
A16 and no shared-A8 public policy is retained**. On RTX 5090/CUDA 13.1, the acquired
native layer-0/3 BF16 shared gate/up/down matrices were independently converted offline to
row-scaled FP8 (BF16 row scale, nearest-even finite E4M3FN code). Gate-only, up-only,
down-only and all-three substitutions were evaluated at T=17/129 with the existing two
represented BF16 input patterns, protected router/scalar gate, and A16 routed experts.
The complete independent FP64 oracle decoded the candidate's actual FP8 weights; it did
not use source BF16 weights or A16 kernel outputs as the candidate arithmetic reference.
The unchanged complete-MoE criterion was relative L2 `2.5/255`, absolute floor `1/32768`,
and gross-error coefficient `2/255`.

| Shared FP8 A8 roles | Layer 0 worst token relative L2 | Layer 3 worst token relative L2 | Admission |
|---|---:|---:|---|
| gate only | 0.009465 | 0.018093 | rejected (including gross-error failures) |
| up only | 0.010868 | 0.016322 | rejected |
| down only | 0.012247 | 0.018611 | rejected |
| gate/up/down | 0.017227 | 0.030641 | rejected |

Weight quantization alone produced matrix relative L2 `0.025717..0.026149` and ideal
complete-MoE perturbations `0.009041..0.031953`; these are separate from the arithmetic
errors above and are not PPL estimates. The same candidate weights under A16 satisfied
the unchanged complete criterion except layer-0 gate-only: relative L2 `0.005393` passed
the aggregate threshold but a gross per-element error failed. That specific baseline
failure is an A16 private-arithmetic limitation for the new represented inputs, not
evidence of an A8-only failure or an artifact decoding error. Original native BF16-shared
baselines remained passing. No threshold was relaxed and the losing policy, experimental
routes and temporary candidate tests were removed; no speed admission or sequence-quality
claim is made for shared A8.

### 0.6 Integration gates, review, and smaller-model handoff

This bounded implementation tranche is complete (2026-09-18). Final verification on
RTX 5090 / CUDA 13.1.2 used `NINFER_DEV_CONTAINER=ninfer-builder-qwen4-mixed
NINFER_BUILD_VOLUME=ninfer-build-cache-qwen4-mixed NINFER_DEV_JOBS=8
./scripts/run-unit-tests.sh`: 112 passed, two missing-artifact load-plan skips,
zero failures, 452.74 seconds of test time. Separately configured real-source
component/sequence tests passed, including the original MoE regression inputs,
native FP8 PLE rows and active A4 residual composition below. The 11 focused Python
format/source tests, Python compilation and `git diff --check` passed. Independent
Sol review of the finished production changes, repairs and final sequence tests
returned clear; this was not merely a plan review. Full-model PPL, complete native
FP8-table capacity admission and future target registration remain deferred gates,
not claims established by the bounded tests. No commit or push was made.

The native T=17 layer-0→layer-3 A16 sequence exposed a shared-MoE precision bug:
layer-3 token 10/index 154 gave `0.0351562` against ideal `0.0348243` (whole),
and `0.0339355` against `0.0342919` (partitioned). An independent staging
counterfactual isolated the shared gate/up BF16 stores, not routed quantization.
The BF16 shared pair now uses the existing GEMV/small-T/MMA families with private
FP32 outputs through SwiGLU, then a distinct BF16 activation buffer; mixed-format
pairs and routed A16/A4 contracts remain unchanged. Synthetic witnesses cover the
27/28 projection-family boundary and 128/129 full/tail tiles. The two original
represented input panels (not golden outputs) are retained locally under
`models/qwen4-native-layers/qwen4-moe-layer3-original-{whole,partition}-input.bf16`
with source/seed/profile metadata; the opt-in native harness recomputes the full
FP64 oracle when present and explicitly reports missing regression fixtures.

The corrected native source-A16 selected-layer chain passes at T=5/17 in whole
and prefill(T−1)+decode1 schedules. A separate active-W4A4 MLP residual witness
uses native layer-0/3 GR read → MoE → GR injection, with one represented residual
token repeated to T=65. Actual routing counts 65 (whole) and 64/1 (partitioned)
prove the grouped A4 route is active before the A16 decode tail. The unchanged
accumulated criterion `{relative_l2=.02, gross_absolute=.005, gross_relative_to_max_reference=.02}`
passes: worst relative L2 `0.00327633`, maximum absolute error `0.005859375`.
This qualifies that short MLP residual composition, not A4 stateful attention/GDN,
general prompt quality or full-model PPL; residual addition can attenuate the
substantial standalone A4 MoE errors recorded in §0.3.

Final corrected native MoE verification passes both original T=17 regression
panels and all source-width witnesses. Layer-0/3 standalone A16 worst-token
relative L2 is now `0.00298893` / `0.00324256`; active A4 at T=257 remains
`0.114199` / `0.0942307`. These supersede the pre-private-precision-fix accuracy
snapshot above, without changing its recorded performance measurement scope.

Reuse a short resident block sequence for the A16 baseline, source W4A4 profile and combined
candidate placements. Check accumulated output/state error, routing/selection drift, lifetime,
and prefill-to-decode transitions. This is not full-model PPL, nor evidence that the oversized
preview is a registered Engine target. Engine graph/concurrency/MTP/frontend work remains behind
the exact fitting-target admission gate in the later phases of this plan.

At each implementation checkpoint run focused exact/numerical/real-layer tests, affected Python
tests, `git diff --check`, and the full `scripts/run-unit-tests.sh` gate after substantial work.
Require independent Sol review of finished implementation and repairs, not just this plan.
Commit and push only on the user's explicit request.

When a smaller checkpoint arrives: freeze its actual topology/precision/scales, admit host PLE
and GPU core/state budgets, adapt only concrete shapes/formats, bind its complete artifact,
enforce PLE residency before public Engine readiness, and run paired per-token NLL/PPL plus
task/long-context/state qualification. Re-evaluate combined A4 placements on that checkpoint;
do not inherit this preview's calibration or claim that a small layer harness proves model PPL.

## 1. Fixed authorities and evidence

Freeze these revisions in the eventual model and artifact references before implementation:

| Authority | Pin | Use |
|---|---|---|
| Qwen architecture report | arXiv `2608.30320v1`, 2026-08-31 | QSA, GR, GDN, PLE design equations and reported ablations |
| Official preview repository | commit `69885871a64393807d988b27b1b5e380e8f28526` | Qwen's statement that Flash-Next is a Qwen4 preview |
| Official BF16 checkpoint | `Qwen/Qwen3.8-Flash-Next` revision `de4b8e4d43b917e7706784d8bb445c9af86a3540` | Exact config, tensor index, frontend resources, and source weights |
| Transformers Qwen4Exp | commit `c119ec3cc37ab69642f39cca2de4187714002b08` | Checkpoint-consuming Text/Vision formulas and state behavior |
| vLLM Qwen4Exp MTP cross-check | commit `d6bce42983bc0b2095ad6422dbf1399e219ae572` | Executable evidence for MTP, not by itself a mathematical authority |
| SGLang Qwen4Exp open integration | PR 36497 head `78c5024e9d9f589dcb4deb7f4ba4fb23f7e85385` | Independent executable MTP/QSA-reuse evidence; open PR, not a stable release authority |
| Unsloth UD-IQ1_S GGUF diagnostic | revision `38bb39ee97821de2c9009abb7e93950eec396e66` | External quant-specific smoke/PPL evidence and concrete host-PLE layout; never the mathematical oracle |
| llama.cpp Qwen4Exp diagnostic | commit `9a4843cf2f1a3fc8e39f8148e92ee6bfe18e2db6` | External Text/PLE execution cross-check; its current dense core-attention fallback is not QSA performance evidence |

Primary source addresses are preserved in §12. Do not implement from a floating `main`, a model
card summary, a third-party quantization, or the paper alone. The paper intentionally omits exact
PLE hashing/injection, checkpoint MoE details, Vision, and most MTP execution semantics.

Before a real target leaf is admitted, write two active authorities analogous to the current model
and artifact references:

- `docs/maintainer/<target>-model.md`: exact formulas, dimensions, topology, frontend, state,
  positions, MTP, and observable numeric boundaries;
- `docs/maintainer/<target>-artifact.md`: exact identity, complete ordered inventory, source pins,
  conversion, formats/layouts, aliases, placement, and binding.

Paper/reference discrepancies must be settled in the model authority from checkpoint parity:

- GDN production code scales normalized queries by `1/sqrt(128)` while the paper's recurrence
  presentation does not show that factor.
- QSA production code divides the four-head ReLU score sum by `sqrt(128)` while paper Eq. 15 does
  not show it. This does not change top-k absent ties, but it is still part of an exposed score
  oracle if scores are materialized.
- Paper Eq. 16 uses `ceil(K/r)` while the exact checkpoint uses integer `K/r`; these coincide only
  because `K=2048` and `r=4`.
- "Layer 2" means one-based layer id 2 and zero-based decoder index 1.

## 2. Preview architecture baseline

These facts define the current `qwen4` verification profile. A future Qwen4 checkpoint copies none of
the constants implicitly; its exact config must be frozen again.

### 2.1 Global shape

| Field | Preview value |
|---|---:|
| Text hidden width | 2560 |
| decoder layers | 48 |
| layer schedule | 12 repetitions of GDN, GDN, GDN, QSA |
| QSA layers, zero-based | `3, 7, ..., 47` |
| vocabulary rows | 248320, untied embedding/output head |
| RMSNorm epsilon | `1e-6` |
| native position capacity | 262144 |
| RoPE theta | `1e7` |
| MRoPE | interleaved sections `[11,11,10]` |
| residual branches / GR rank | 4 / 320 |
| routed experts / selected | 512 / 10 |
| routed/shared expert width | 640 / 640 |
| MTP layers | 1 QSA decoder layer with separate attention/MoE GR modules, plus final GR read |

The official one-million-token recipe is optional static YaRN applied at deployment. It is not the
native checkpoint contract and must never silently alter short-context behavior.

### 2.2 Gated DeltaNet

The preview retains the Qwen3.6-style delta recurrence but changes geometry and uses a sigmoid
output gate:

```text
Q/K heads=16, Q/K head width=128
V heads=48, V head width=128
qkv projection rows=10240, z rows=6144, a rows=48, b rows=48
causal depthwise convolution width=4
persistent recurrent state=FP32 [48,128,128] per layer/request
```

For every V head, use its shared Q/K head, convolve Q/K/V, apply SiLU, L2-normalize Q/K, then apply
decay, delta correction, state update, and state read in the exact order frozen by the model
authority. The GDN internal output RMSNorm is ordinary learned scaling, not zero-centered; its
output is multiplied by `sigmoid(z)` before the output projection. GDN convolution history and the
FP32 recurrence are independent persistent state.

### 2.3 Qwen Sparse Attention

Core attention has 24 query heads, two KV heads, head width 256, and 64 rotated dimensions. The
query projection logically produces a query and equal-width sigmoid output gate; Q and K use
zero-centered RMSNorm and partial interleaved MRoPE. Core attention is causal sparse GQA with
logical softmax, followed by the sigmoid query gate and output projection. Pinned Transformers uses
an FP32 softmax as a cross-check implementation profile, not an implicit semantic cast.

The per-layer indexer is not XAttention/Sparge and is not an approximate flag on dense GQA:

```text
index queries=4 heads x 128, shared index key=1 head x 128
partial RoPE=64 dimensions
micro-block r=4, token budget K=2048, complete-block budget=512
```

It projects token queries and raw keys, normalizes queries, and applies query-position RoPE. For
each request/query, blocks are groups of four consecutive entries in the ordered request-local
visible-token index list; they are not unconditionally `floor(absolute_position/4)` groups. Only
complete visible blocks are averaged. The exact preview baseline performs the mean in FP32, casts
the pooled key to the represented raw-key dtype, then applies key RMSNorm and block-start RoPE.
Each query scores a complete block with the sum of four `ReLU(q dot k)` values, selects the highest
512 blocks or all available blocks, expands them into token indices, and appends the incomplete
causal tail of zero to three tokens. NInfer defines equal-score selection by lower logical block id
first so selector output is deterministic. Upstream parity is diagnostic at ties because
`torch.topk` does not promise stable tied-index order. The selected indices drive core attention.

The persistent QSA state is:

- paged core K/V for 12 layers;
- indexer state sufficient to reproduce exact raw-key block averages and the incomplete tail;
- positions required for block-start MRoPE and multimodal MRoPE.

An optimized store may seal completed four-token blocks and retain only the raw tail instead of all
raw index keys, but only after direct proof that accumulation, cast, block boundary, position, and
selection results match the declared state transition.

### 2.4 Gated Residual

The token embedding is copied into four 2560-wide branches. Every attention/GDN sublayer and every
MoE sublayer has a separate GR read/write module; a final read-only GR collapses the branches.

For concatenated branch state `R` and per-branch zero-centered RMSNorm `Rhat`:

```text
G = reshape(sigmoid(W_up(SiLU(W_down(vec(Rhat)) / 4))))
x = mean_branches(G * Rhat)
s = 2 * sigmoid(W_write(vec(Rhat)) / 4)
R'[i] = R[i] + s[i] * F(x)
```

There is no residual branch-mixing matrix and no static term. The two `/4` factors, the branch
mean, zero-centered group normalization, and scalar-per-branch write gate are semantic. FP8 branch
storage described in the paper is an implementation profile, not a default semantic cast; BF16 and
FP8 routes require separate direct qualification against the same oracle.

### 2.5 Sparse MoE

Every decoder and MTP layer computes a logical router softmax over 512 experts, selects top 10,
renormalizes selected weights, evaluates ten 640-wide routed SwiGLU experts, and sums their weighted
outputs. Pinned Transformers uses FP32 for its router softmax as a cross-check implementation
profile, not an implicit semantic cast. The layer independently evaluates one 640-wide shared
SwiGLU expert, multiplies it by a sigmoid scalar gate, and adds it to the routed result. There is no
inference token dropping or capacity factor. Logical expert id remains tied to router row and stored
bank row.

### 2.6 PLE n-gram injection

PLE runs once, before the attention GR at one-based layer 2. It uses eight bigram and eight trigram
hash heads, 160 values per head, producing width 2560. Each head has a distinct successive-prime
vocabulary just above 20,000,000; the concatenated padded table is exactly
`[320001536,160]` (51.20024576B values). Missing history and history following token 248044 use/reset
to token 248044.

Addressing uses exact signed-int64/uint64 wrap behavior, seed 1234, SplitMix64 constants
`0x9E3779B97F4A7C15`, `0xBF58476D1CE4E5B9`, and `0x94D049BB133111EB`, and
`base_seed = seed + 10007 * ple_layer_index`; the preview has one PLE module, so its only admitted
module index is zero. Odd position multipliers mix token ids with XOR;
each result is reduced by its head's prime and shifted by the cumulative head offset. The exact
integer specification belongs in the model reference and an independent exact oracle.

The lookup result is projected to four keys and one shared value. Separately group-normalized keys
and residual queries produce one scalar per branch:

```text
g = dot(key, query) / sqrt(2560)
g = sign(g) * sqrt(max(abs(g), 1e-6))
v_gated = sigmoid(g) * value
output = v_gated + SiLU(depthwise_dilated_conv(group_rmsnorm(v_gated)))
```

The depthwise convolution has kernel 4, dilation 3, and history length 9 over width 10240. PLE owns
two prior token ids and this convolution state independently of GDN. PLE values add to all four
branches before the layer's GR read.

### 2.7 Vision, frontend, and MTP

The preview is multimodal. Its Vision tower remains 27 layers at width 1152 with 16 heads, FFN
width 4304, spatial patch 16, temporal patch 2, merge 2, and no deep-stack outputs; its merger emits
width 2560. Reuse central Vision Ops where their semantic contract matches. Implement or extend
the semantically closed merger in `src/ops`, qualify its 2560 profile directly, and let the exact
leaf bind weights/configuration; it is not a fourth target execution-leaf family.

The exact tokenizer, chat template, generation config, multimodal resources, special ids, and
reasoning/tool grammar are artifact resources. Golden token vectors protect their observable
behavior. Do not approximate the template in C++.

The MTP checkpoint contains `fc_embedding`, `fc_hidden`, two pre-norms, one QSA decoder layer with
separate attention and MoE GR read/write modules, and a final read-only GR mixer, while sharing
token embeddings and the output head. Transformers deliberately ignores `mtp.*`; the paper states
one learned layer, multi-step rollout, and reuse of step-zero QSA indices, but does not fully define
the stem, two-stream recurrence, rollout, cache alignment, or transaction. Phase 0 triangulates the
pinned checkpoint inventory, pinned vLLM implementation, and a separately maintained serving
implementation, then freezes the complete stem, recurrence, index reuse, and accept/fold behavior
in the model authority before the selected identity is admitted or registered. If the independent
executable paths disagree, MTP and target registration remain gated rather than choosing whichever
output is convenient. There is no implicit MTP-free product profile; creating one would be a
separate exact artifact/product decision with MTP tensors validate-only and no MTP claim.

The two current implementations agree on the unusual MTP structure to freeze with golden traces:

- preserve the target's pre-final-GR four-stream hidden `[T,4,2560]` while ordinary target logits
  consume the final GR read `[T,2560]`;
- normalize/project the shared token embedding with `fc_embedding`; separately normalize the
  entire concatenated 10240-wide carried state, reshape it into four streams, project each stream
  with the shared `fc_hidden`, then add the projected embedding to every branch;
- run the one-layer QSA/attention-GR/MoE-GR decoder, use its final GR read for the shared LM head,
  and carry its pre-final-read four-stream state into the next draft step; and
- capture the target-aligned last row's expanded QSA selection at draft-extend, then reuse that
  per-request/per-layer selection for later top-1 chain steps; target verification does not reuse
  the draft selection.

Before declaring these semantic, capture cross-engine stem output, pre/post-final-GR state, logits,
all 2051 selected/tail index slots, four-step carried state, and reject/accept results. The SGLang
pin is an open integration branch, so agreement of source structure alone is insufficient.

## 3. Ownership and architecture

Create a new family and exact leaf rather than extending Qwen3.6:

```text
src/targets/qwen4/
    frontend definitions and owning prepared/output types
    semantic weight-view schemas
    Text/Vision/MTP schedules and QSA/GR/PLE orchestration
    state/workspace composition and graph algorithms

src/targets/<exact-qwen4-sku>/
    registered identity and weights profile
    fixed config and topology constants
    artifact binder and LoadedModel
    three target execution-leaf families
```

Before product admission, `src/targets/qwen4/verifier.{h,cpp}` and `program.cpp` are the deliberate
diagnostic exception: an unregistered C=1 eager Text package that binds only
`qwen4/verification` + `unsloth-ud-iq1-s-host-staged`, owns the fixed 4096-token
verification state/staging, and composes the admitted Ops into the exact 48-layer schedule. Its
Program accepts only numeric token/target ids, exposes logits, NLL, and state/routing diagnostics,
and has C=1 T=1 decode plus T=1..4096 chunked prefill for the opt-in real-artifact test; it has no
registry, Engine, CLI, serving, MTP, Vision, multi-request batching, or CUDA-Graph route. It is not
the eventual family runtime or an exact product-SKU leaf.

QSA, GR, PLE, and exact n-gram continuation state make the Program materially different from
`src/targets/qwen3_6`. Do not copy the Qwen3.6 Program into a leaf, introduce a family base class,
or route behavior by strings. The exact package instantiates the Qwen4 family algorithms using
populated semantic views and private leaf payloads.

Central ownership follows semantic effects:

| Component | Owner |
|---|---|
| QSA selector, sparse core attention, GR, n-gram addressing, PLE injection, extended GDN/MoE/Vision transformations | `include/ninfer/ops`, `src/ops` |
| physical QSA KV/index pages, fixed state arrays, raw transfers | `src/core` |
| `.ninfer` formats, mapped tensor payloads, descriptors, materialization | `src/artifact` |
| layer order, state instance selection, QSA frontier, PLE prefetch timing, MTP index reuse, transaction commit/rollback | `src/targets/qwen4` |
| exact constants, tensor binding, storage profile, load diagnostics | exact target leaf |
| public generated-token publication and Engine PIMPL | `src/runtime` |

Artifact materialization distinguishes ordinary mapped tensors from resident-host tensors.
PLE uses resident-host placement: an owned exact-span mapping is fully populated and eagerly
OS-locked before model load returns. `LoadedModel` retains its backing and lock until all
consumers drain. Reader destruction must not release resident pages, and distinct live owners
must not accidentally unlock each other's mappings. This does not turn mapped expert streaming
into a product execution lane. The target owns PLE row requests and prefetch timing; artifact/core
own memory lifetime, locking and raw transfer mechanisms.

The execution requirements are:

- reject insufficient lock permission or RAM at startup; never return a lazy/page-cache fallback;
- derive 16 row ids per token and gather from resident RAM into bounded preallocated CUDA-pinned
  staging, preserving token/head order; deduplication is a measured optimization, not a prerequisite;
- batch prompt rows and overlap host gathers with layer-0 compute; the existing prefill path also
  overlaps H2D on its transfer stream and waits before layer-1 consumption;
- distinguish `host_fill_done`, `copy_done` and `consumer_done` so neither the host source nor
  device destination is reused before its last consumer; do not prefetch an unknown decode token;
- drain all host fills, CUDA transfers and consumers before teardown, without mutating committed
  token/conv state; and
- report startup population/locked capacity separately from inference gather latency, H2D
  completion, bytes and exposed consumer stalls. Inference-time storage reads are not admitted.

Do not CUDA-pin or duplicate the entire PLE table, encode it as an opaque resource, or use an
external sidecar. OS locking the complete table is mandatory and is distinct from CUDA-pinning
the small staging buffers. Exact FP8 PLE requires an independently decoded, registered format;
the native checkpoint audit in section 0 determines its scale/layout, not the row-scaled Linear
profile. Phase 0 requires the complete resident host budget plus GPU core/state/staging budget.
Failure stops activation rather than moving PLE to VRAM, disk paging, or CPU floating-point work.

### 3.1 UD-IQ1_S host-staged verification profile

This diagnostic profile is deliberately narrower than product activation. The exact public GGUF
inventory splits as follows:

| Placement | Tensor set | Encoded bytes |
|---|---|---:|
| fully resident, OS-locked host mapping | PLE | 28,800,138,240 |
| host mapped | 48 routed gate/up expert banks | 17,196,646,400 |
| device resident | 48 routed-down expert banks | 22,649,241,600 |
| device resident | every non-routed Text tensor | 3,889,410,560 |
| **device-resident ordinary weights** | | **26,538,652,160 (24.72 GiB)** |

The routed banks contain 68 IQ1_S and 28 IQ2_XXS gate/up tensors plus 48 IQ4_NL down tensors; the
PLE table is the remaining IQ4_NL tensor. For C=1 decode, gathering the selected ten gate/up slices
transfers exactly 335,872,000 encoded bytes over all 48 layers per token. The reusable staging
pipeline uses two 844,800-byte slots, for 1,689,600 bytes of device storage and the same amount of
pinned host storage. Each slot holds one maximum-size IQ2_XXS gate/up rank pair; IQ1_S pairs use
640,000 bytes. This is smaller than staging routed-down
(442,368,000 bytes/token) or all routed weights (778,240,000 bytes/token), and it keeps every model
transformation on the GPU for implementation verification.

The verifier is eager because each layer's router result determines its host source spans. Each
layer completes one 40-byte route-id D2H dependency, queues the independent shared expert, then
gathers selected rank pairs into alternating pinned slots. A distinct transfer stream copies each
rank into its paired device slot while the compute stream consumes the preceding rank. Explicit
transfer-ready and consumer-complete events protect slot reuse, while accumulation remains in route
rank order. The next layer's route/ID barrier transitively closes both slots from the preceding
call. One pipeline/event set is bound to one compute stream; changing streams requires an external
drain and a distinct pipeline. No whole-round graph claim is made.

For chunked prefill, the GPU returns all `10*T` ids once per layer. Fixed caller-owned integer
scratch groups occurrences by ascending expert id; each unique gate/up pair is gathered once into
groups of at most 32 and the matrices plus occurrence list cross in one contiguous H2D per group.
Two maximum-size group slots overlap transfer and GPU consumption. PLE similarly gathers the
complete `16*T` row panel and overlaps its single H2D with layer 0. These are raw compressed-byte
host operations only: all decode, projection, recurrence, attention, and mixture arithmetic stays
on the GPU.

A separate one-layer public-Op benchmark measures the future device-resident placement without
changing this verifier. It owns complete device IQ1_S or IQ2_XXS gate/up banks and the IQ4_NL down
bank, keeps selected ids on device, and fuses the routed grids and shared projections while
preserving the explicit BF16 seams and rank-ordered FP32 accumulation. Its exact weight footprints
are 808,785,920 bytes for the IQ1_S/Q5_K profile and 914,078,720 bytes for IQ2_XXS/Q6_K. Rotating
route windows cover all 512 experts so the result is not an L2-only hot slice. The experiment
cannot establish full-model fit: adding all preview gate/up banks to the 26,538,652,160 resident
bytes would require 43,735,298,560 bytes before state and runtime storage.

GGUF remains converter input, never a runtime lane. The converter writes one `.ninfer` artifact and
preserves the represented GGML blocks under explicit registered numeric-format/layout identities.
The seven formats present in the complete Text artifact are Q8_0, Q4_K, Q5_K, Q6_K, IQ1_S,
IQ2_XXS, and IQ4_NL. Each requires an independent exact block decoder before its production decoder
or matrix route is admitted. This diagnostic can establish layer taps, logits, NLL, state behavior,
and paired PPL, but cannot qualify the eventual resident codec or end-to-end performance.

Preserving the GGUF representation also preserves llama.cpp conversion semantics: applicable
zero-centered norm weights are already effective gamma, GDN `ssm_a` is already `-exp(A_log)`, and
GDN V-side tensors are tiled so represented value head `h` consumes Q/K head `h%16`. The verifier
must consume those values directly. Source-checkpoint references retain `(1+w)`, `A_log`, and
grouped head order only when their represented input is the pinned BF16 source.

## 4. Required Op contracts and independent oracles

Write the contract and independent oracle before optimizing each production route. Oracles start
from represented public inputs, decode stored weights exactly, evaluate the complete logical
formula naively in FP32/FP64 (or exact integer arithmetic), and do not reproduce a production
kernel's staging casts, reduction tree, or workspace dtype.

| Op boundary | Contract and oracle evidence |
|---|---|
| grouped RMSNorm | Exact branch grouping, source `(1+w)` or represented GGUF gamma boundary, epsilon, output representation; FP64 sum-of-squares oracle |
| GR read and GR inject | Complete Eq. 30-34 read/output/state transition including `/4`, branch mean, and final read-only form; real width 4x2560 plus small diagnostic shapes |
| QSA index projection/selection | Projection, request-local visible-rank blocks, FP32 mean then represented-key cast, norm/RoPE, complete-block causality, block-start position, four ReLU dots, lower-block-id ties, expand/truncate/tail; tolerant scores and exact selected ids |
| QSA gated sparse GQA | Q/K norms, partial MRoPE, paged selected-id addressing, causal ideal softmax, sigmoid output gate, projection, and core KV effects; compare directly with naive gathered FP64 attention |
| n-gram ids | SplitMix multiplier generation, primes, offsets, int64 wrap, XOR, remainder, EOS reset, and continuation; exact oracle, including prefill versus token steps |
| PLE gather/decode | Exact selected rows and registered table codec; exact ids and codec-specific numeric comparison |
| PLE injection | Key/value projections, group norms, signed-sqrt gate, sigmoid value, dilated convolution, additive result, and final conv state; one-shot/chunked/T=1 oracle |
| GDN Qwen4 profile | Full artifact-layout recurrence oracle with tiled `h%16` Q/K mapping, represented `ssm_a`, sigmoid output gate, projection shapes, FP32 state, and all semantic cast/state boundaries |
| sparse MoE Qwen4 profile | Full 512-way ideal router softmax, lower-id tie rule, top-10 normalization, ten routed SwiGLUs, shared gated expert, merge, and epilogue; real geometry and T regimes |
| MTP stem/pack and layer Ops | Stem/two-stream formula, QSA index inputs/reuse, layer outputs, and represented state records after §2.7 is resolved |
| MTP state fold and acceptance Ops | Closed record/fold and acceptance primitives against independent oracles; family integration owns reject/partial/full transaction and runtime owns publication |
| Vision merger | Exact source layout, GELU/norm/linear formula and 2560-wide output at real patch geometry |

Likely contract headers are:

```text
include/ninfer/ops/qsa_indexer.h
include/ninfer/ops/qsa_attention.h
include/ninfer/ops/gated_residual.h
include/ninfer/ops/ngram_embedding.h
include/ninfer/ops/ple.h
```

Extend an existing Op only when its mathematical contract already matches and a finite new geometry
or activation profile is sufficient. In particular, requalify `gdn_input_proj`,
`gdn_gating_proj`, and sparse MoE for the real Qwen4 shape; do not call the current H=2048,
256-expert, top-8 sparse-MoE implementation generic. Existing XAttention/Sparge tests do not
qualify QSA.

Tests must cover realistic regression boundaries rather than source shape:

- QSA frontiers 0 through 5 and 2047 through 2053, padding/segmentation, fragmented pages, unequal
  batched lengths, exact 512-block saturation, tail inclusion, all-zero/equal-score lower-id ties,
  FP32-pool/cast boundaries, and multimodal positions;
- PLE initial history, EOS reset, chunk boundaries, dilation history, and T=1 continuation;
- GR eager/graph and BF16/qualified FP8 storage at real width;
- GDN one-shot versus partitioned prefill and T=1 recurrence with initial/final FP32 state;
- MoE routing ties, duplicate selections forbidden, selected-weight normalization, shared path,
  T=1, verification widths, and prefill widths;
- every persistent-state Op under request reset/isolation at startup-fixed batch 1 through 8;
- MTP rejection, partial acceptance, full acceptance, and a speculative span crossing a QSA
  four-token block boundary.

Upstream FP32 attention/router softmax is a cross-check implementation profile unless the exact
target authority deliberately declares an observable semantic cast. The ideal oracle evaluates
softmax in FP64 from represented public inputs; explicit persistent-state boundaries such as GDN
FP32 state remain semantic. Numeric criteria are not copied from Qwen3.6 or chosen after observing
a failure. Establish suite-owned named criteria from oracle error distributions across adversarial
and target-representative inputs, record both relative and gross-error criteria where either can
hide a failure, and allow distinct criteria for materially different arithmetic/quantization
profiles. Within one named profile, apply its criterion uniformly without per-case overrides.

## 5. State, caching, and transaction design

Extend the sequence-state substrate with typed target state, preserving one logical committed
frontier per request:

```text
Qwen4SequenceState
  target decode cursor + current anchor token
  per-pool allocations, entitlements, committed/reserved frontiers
  main QSA core KV pages
  QSA index pages/sealed blocks + raw tail + positions
  GDN convolution history + FP32 recurrence
  PLE two-token history + dilated-convolution history
  optional MTP QSA KV/index/fixed state + provisional frontier
  current/rewrite/ladder continuation hidden and typed checkpoint ledger
  prefix identity and target-versioned continuation metadata
```

GR branches are current-token activations, not context-growing cache. Prefix RAM/disk snapshots and
response retention must include the current anchor/cursor, continuation hidden, QSA positions and
index state, GDN and PLE state, every backend frontier, prefix identity, and target-versioned
metadata, and must version their physical images. Restoring core KV without every matching
continuation owner is not a valid prefix. Restore publication is atomic only after all images are
present, version/identity/frontiers validate together, and every allocation is reserved.

Capacity is a joint target-derived reservation vector over core QSA KV pages, QSA index pages,
fixed GDN/PLE state, and optional MTP pools/provisional lead. It preserves the existing shared
growing-capacity contract: C active requests receive fixed state and at least one page entitlement,
one request can reach `max_context`, and the sum of current entitlements cannot exceed each pool;
it does not promise `C * max_context` resident tokens.

Speculative execution records provisional QSA core KV, index raw tail/sealed blocks, GDN state
effects, PLE token/conv effects, and MTP state without mutating the committed slot. One fold applies
only the accepted prefix. A rejected or partially accepted proposal cannot leave a completed QSA
block, overwrite the committed raw tail, advance PLE token history, or alter GDN recurrence. MTP
step-zero selected QSA indices are copied into explicit graph-stable storage and reused only for the
target-aligned steps declared by the MTP authority.

Core KV compression and index-key precision are independent profiles. Qualify BF16 first, then
INT8/NVFP4 core KV and any compressed index representation directly against the same QSA oracle.
Do not assume the current default NVFP4 KV is fastest for gather-heavy QSA on `sm_120a`; both
correctness and end-to-end performance select the product profile.

## 6. Reference, conversion, and parity tools

Add exact, target-private tooling:

```text
tools/convert/<target>/
tools/reference/<target>/
tools/parity/<target>/
tests/targets/<target>/
```

The independent Python reference consumes the `.ninfer` artifact, not live Transformers modules.
It implements complete Text, PLE, QSA, GR, MoE, Vision, and eventually MTP formulas from the active
model authority, and exposes taps for inputs, selected QSA ids, router ids/weights, block outputs,
logits, and every persistent state transition. It decodes registered packed weights itself.

The converter validates the exact upstream config, every source tensor name/shape/dtype, all
frontend resources, the complete planned inventory, and every conversion recipe before opening the
output. It streams source shards and does not require the 360 GB BF16 model in RAM. A representative
verifier checks source-to-artifact tensor transforms and packed-weight decode. The final `.ninfer`
contains the complete product image; PLE and experts are not optional sidecars.

Whole-model parity uses pinned deterministic input ids on a machine capable of running the official
BF16 source. Capture compact reference evidence rather than the checkpoint: per-token NLL, top
logit ids/values, QSA selected ids, chosen layer taps, and initial/final persistent state. Then
compare the artifact-native reference and NInfer from the same represented inputs. Framework parity
is supplementary; the independent artifact-native formula is the mathematical oracle.

## 7. Perplexity and behavioral qualification

Perplexity is mandatory integration evidence and never substitutes for an Op oracle. Generalize
`Engine::score`, `apps/ppl`, and `tools/ppl` so corpus encoding uses the active target frontend
rather than directly including the Qwen3.6 tokenizer. Preserve per-token `.nllf32` sidecars for
localization and paired statistics.

The scoring contract and a committed manifest are fixed before results are generated:

```text
mean_nll = sum[-log p(token[t+1] | valid prefix through token[t])] / scored_tokens
ppl = exp(mean_nll)
```

The manifest records corpus revision/hash, deterministic sample selection algorithm, seed/ranges
and sample count, exact token ids or raw-text encoding contract, document reset/separator policy,
token count, scored positions, context/stride, schedule, numeric formats, graph mode, and hardware.
Use raw-text language-model scoring without a chat template or implicit BOS unless the corpus
contract explicitly contains them.

Qualification includes:

1. a small frozen WikiText-103 raw test slice for fast reproducibility;
2. deterministic C4 and Chinese/multilingual held-out slices for the release campaign;
3. 4K, 32K, and 128K cells, plus a 262144-native-context cell when capacity permits;
4. full-prefill and teacher-forced T=1 schedules, including legal prefill chunk partitions;
5. BF16 core KV first, then each product codec against BF16 within the same schedule;
6. CUDA Graph on by default, with a focused eager comparison;
7. spec-none and MTP target-verify after MTP is admitted;
8. per-token delta NLL, paired standard error, maximum NLL, non-finite count, and terrible-token
   localization, not mean PPL alone.

QSA is intrinsic model math, so dense attention is not its baseline and QSA is not an optional
scheme row. Alternative QSA kernels compare at the same cache/index dtype and selection semantics.
The paper's private "Uncheatable PPL" is not a reproducible gate. Define quantization limits as
degradation from the pinned BF16/reference or artifact-native baseline before running the release
campaign.

The same pre-campaign manifest declares per-cell maximum paired `delta_mean_nll` and its confidence
bound, maximum/gross token-NLL error, zero non-finite policy, exact retrieval dataset/scorer and
minimum score or maximum regression, byte/token-exact multimodal goldens where applicable, and a
distributional/reference criterion for MTP acceptance. Performance thresholds are recorded and
judged separately from semantic/numerical correctness.

PPL cannot by itself validate retrieval or speculation. Pin the exact official RULER and MRCR task
revisions, instances, prompts, and scorers in the release manifest and run them at supported
lengths, alongside exact selected-index diagnostics, normal structured stopping, multimodal
goldens, and MTP mean accepted length/distribution. Final-output plausibility or greedy identity
does not excuse an Op/state failure.

## 8. Low-level implementation sequence

Each phase ends at its stated gate. Later phases do not waive an earlier gate.

### Phase 0 — select the runnable identity and freeze semantics

Choose the exact Qwen4 checkpoint that will replace the current product identity or be the newly
declared sole product identity. Confirm all of its device-resident weights plus C fixed state, the
shared growing-pool minimum that gives every lane one page and one sequence `max_context` capacity,
QSA index pools, any MTP provisional lead, workspace, graph allowance, and 1 GiB headroom fit the
5090. When the exact target contains preview-style PLE, its mapped PLE is excluded from those device
weights; independently confirm its artifact-owned, fully populated OS-locked host mapping and
bounded CUDA-pinned staging fit host capacity and meet the startup/inference latency gate.
For each homogeneous pool, require the target-derived candidate `B(M)` to include all coupled
pools/fixed allocations and satisfy the current `M_min <= M <= M_max` shared-capacity admission
rules; do not budget or advertise `C * max_context`. If PLE is host-resident, also require artifact
storage, full resident/locked host budget, bounded pinned staging, and measured startup/inference latency from
§3. The currently audited BF16 and UD-IQ1_S profiles cannot be registered. A complete custom profile
is conditionally eligible only after it passes this Phase 0 gate; until then, continue reference/Op
development without registration. Artifact-backed experts or multi-GPU execution remain a separate
product-contract decision.

Read every source shard header, produce the model/artifact authorities, pin every upstream
resource, settle the paper/code discrepancies in §1, resolve the MTP stem/two-stream
recurrence/index reuse/cache alignment/accept-fold semantics from the independent executable
evidence, and define native versus optional YaRN behavior. Gate: no inferred formula, unresolved
tensor, or unowned persistent state remains in the selected target contract.

### Phase 1 — independent reference and exact codecs

Implement n-gram hashing, grouped zero-centered RMSNorm, GR, PLE, QSA, MoE, GDN, Vision merger, and
state transactions in target-private Python reference code. Add exact/tolerance tests against
small hand-computed fixtures and upstream BF16 taps. Gate: the reference explains each captured
tap and state transition without calling production CUDA or Transformers forward methods.

### Phase 2 — artifact inventory, formats, and conversion

Define the complete `.ninfer` inventory and source mapping. Add only the generic artifact changes
required by the selected profile, including mapped host tensor placement or FP8 only if actually
used. Implement streaming conversion, conversion reports, inventory/recipe tests, packed numeric
verification, and one real conversion. Gate: every upstream tensor is consumed or deliberately
rejected, every artifact object is bound or deliberately validate-only, and representative packed
values decode against source values under the declared criterion.

### Phase 3 — Op contracts, oracle tests, then CUDA

Land the semantic headers and oracle tests from §4 before production routes. Implement correct
eager BF16 routes at real shapes, then use `tools.kdev bound`, `mma`, public-Op sweep, and production
path qualification for `sm_120a`. Add benchmarks only for decisions that affect the product path.
Gate: every route, token regime, cache codec, and state transition passes the same independent
oracle; selector ids are exact.

### Phase 4 — Qwen4 family Program and eager Text execution

Create `src/targets/qwen4` and the exact leaf. Build fixed layouts, workspace recipes, QSA pools,
GDN/PLE state, typed model views, eager prefill, eager T=1 decode, scoring, and final GR/head. Update
the Engine internals that currently hard-code `targets::qwen3_6::PreparedPrompt` to use closed
per-package type erasure or a variant keyed by `ActiveTarget`; do not expose target types publicly.
Gate: artifact-native reference parity for layer taps, selected indices, logits/NLL, chunked
prefill, T=1 continuation, request reset, and C=1..4 isolation under the current product contract.

The unregistered verifier portion of this phase is implemented for C=1: startup-fixed T=1..4096
storage, T-wide embedding/GR/GDN/PLE/QSA/MoE execution, per-query causal QSA visibility, final-column
head projection, and reset/partition/EOS continuation tests are live. Product registration,
frontend ownership, multi-request C=1..4 scheduling, transactional MTP, and CUDA Graphs remain the
later gates described below and are not implied by the verifier.

### Phase 5 — Vision and frontend

Bind exact tokenizer/template/generation/media resources, instantiate the Qwen4 Vision schedule,
the 2560 merger, visual scatter, and four-row Text/three-axis MRoPE positions. Add golden frontend
and multimodal parity cases. Gate: CLI and serving prepare the same owning input through the public
Engine, with exact token and multimodal position parity.

### Phase 6 — MTP and transactional execution

Implement the Phase-0 MTP authority's stem, one-layer QSA/dual-GR/MoE model, index reuse, draft
rollout, target verification, state-record/fold primitives, and family transaction. Extend every
prefix/checkpoint/state transaction to QSA and PLE, with runtime retaining generated-token
publication policy. Gate: independent Op oracles plus integration coverage for rejection,
partial/full acceptance, selected-index reuse parity, target logits, and published-token behavior
at C=1..4.

### Phase 7 — CUDA Graphs, concurrency, and retention

Create address-stable graph families for the startup-fixed batch envelope and supported
speculative widths. Integrate exact-Qwen4 state with paged capacity, prefix retention, RAM/disk
snapshots, cancellation, and boundary-only scheduling. Gate: graph/eager numeric equivalence under
the Op criteria, maximal compact batches, no cross-request state leakage, and exact restore of all
continuation state.

### Phase 8 — real-model quality and performance

Run real artifact taps, the frozen PPL campaign, retrieval, multimodal behavior, and MTP acceptance.
Measure prefill, decode, TTFT, VRAM, fixed state, KV/index bytes per token, Graph memory, and—when
applicable—PLE cold/warm page faults, gather/H2D latency, and overlap on the 5090 with the selected
product codec. Profile end-to-end first and optimize only attributed dominant work. Gate: all
predeclared numerical/behavioral/performance limits pass and the target-derived shared-capacity
vector fits startup and every admitted request transition without hidden allocation, reduced
declared capacity, or fallback.

### Phase 9 — product switch and cleanup

Register the exact identity in `src/targets/registry`, update README/CLI/serve/performance/model-card
documentation and the AGENTS product contract, and remove the superseded project-owned target path
if Qwen4 replaces Qwen3.8-27B. Do not retain aliases or dual internal lanes for compatibility.
Run focused affected checks throughout and `./scripts/run-unit-tests.sh` as the final full-suite
gate. The target is complete only when the real artifact, Engine score/generate/serve paths, PPL,
state restore, concurrency, and performance qualification all pass.

## 9. Concrete repository surfaces

Expected affected surfaces, refined after the exact target is selected:

```text
include/ninfer/ops/{qsa_indexer,qsa_attention,gated_residual,ngram_embedding,ple}.h
src/ops/<matching implementations>
tests/ops/<matching oracle tests>
bench/ops/<only decision-bearing benchmarks>

src/targets/qwen4/{export,impl/frontend,impl/runtime,impl/state,impl/vision}
src/targets/<target>/{export,impl/config,impl/load,impl/package,impl/variant}
src/targets/registry.{h,cpp}
src/runtime/engine/{engine,concurrent_executor}.*

tools/convert/<target>/
tools/reference/<target>/
tools/parity/<target>/
tools/ppl/{run.py,schemes.py,README.md,corpus metadata}
apps/ppl/main.cpp
tests/targets/<target>/

docs/maintainer/<target>-model.md
docs/maintainer/<target>-artifact.md
docs/maintainer/{concurrent-inference-architecture,paged-kv-cache,artifact-container,
                 storage-layouts,tensor-formats}.md as affected
docs/{README,cli,serving,performance}.md as affected
README.md and AGENTS.md at product activation
```

`src/CMakeLists.txt` and `tests/CMakeLists.txt` receive explicit sources and meaningful test
targets. Avoid source-string scans, class-shape tests, empty scheme rows, and tests that merely
prove a constructor/getter exists.

## 10. Stop conditions and material risks

Stop target activation, while retaining already valid architecture/Op work, under any of these
conditions:

- no exact checkpoint/profile has device-resident weights that fit the 32 GB 5090, or a target with
  preview-style PLE cannot also satisfy its resident/locked host capacity and latency contract;
- Qwen4 final architecture differs materially from the preview and its new semantics are not yet
  authoritative;
- MTP formula or transaction behavior cannot be resolved from two agreeing executable references
  and checkpoint evidence;
- a required numeric codec lacks an independent exact decoder/oracle;
- the independent reference and production route disagree on selected QSA ids or committed state;
- a frozen PPL/retrieval/multimodal/speculative manifest gate shows unresolved quality loss;
- startup capacity succeeds only through hidden allocation, reduced context/concurrency, or an
  unadvertised fallback.

The response to a stop is to resolve the authority, choose a fitting official SKU, or explicitly
revise the product contract. It is not to weaken an oracle, omit PLE/MTP/Vision, silently use dense
attention, or advertise a partially runnable identity.

## 11. Completion definition

The architecture is implemented only when the selected exact target has:

- active model and artifact authorities with no preview-derived assumptions left implicit;
- complete `.ninfer` conversion/binding and exact frontend resources;
- independent oracles for every new or extended floating-point and exact transform;
- correct eager and graph prefill/decode for C=1..4;
- transactional QSA/GDN/MTP state and, when present, PLE state, prefix retention, and restore;
- Text, Vision, score, generate, CLI, OpenAI, and Anthropic behavior through the public Engine;
- passing real-artifact parity, paired per-token PPL, long-context retrieval, and all applicable
  multimodal and MTP qualification;
- measured 5090 capacity and end-to-end performance under the advertised codec/context; and
- the focused checks plus the full C++ unit-test suite passing.

## 12. Primary source addresses

- https://huggingface.co/nvidia/Qwen3.8-Flash-Next-NVFP4/tree/fc694b54fb0174e0913e6adf86691ef85a4ead47
- https://huggingface.co/nvidia/Qwen3.8-Flash-Next-NVFP4/raw/fc694b54fb0174e0913e6adf86691ef85a4ead47/hf_quant_config.json
- https://huggingface.co/nvidia/Qwen3.8-Flash-Next-NVFP4/raw/fc694b54fb0174e0913e6adf86691ef85a4ead47/model.safetensors.index.json
- https://arxiv.org/html/2608.30320v1
- https://github.com/QwenLM/Qwen3.8-Flash-Next/tree/69885871a64393807d988b27b1b5e380e8f28526
- https://huggingface.co/Qwen/Qwen3.8-Flash-Next/tree/de4b8e4d43b917e7706784d8bb445c9af86a3540
- https://github.com/huggingface/transformers/blob/c119ec3cc37ab69642f39cca2de4187714002b08/src/transformers/models/qwen4_exp/modeling_qwen4_exp.py
- https://github.com/huggingface/transformers/blob/c119ec3cc37ab69642f39cca2de4187714002b08/src/transformers/models/qwen4_exp/configuration_qwen4_exp.py
- https://github.com/huggingface/transformers/blob/c119ec3cc37ab69642f39cca2de4187714002b08/src/transformers/cache_utils.py
- https://github.com/vllm-project/vllm/tree/d6bce42983bc0b2095ad6422dbf1399e219ae572/vllm/models/qwen4_exp
- https://github.com/sgl-project/sglang/blob/78c5024e9d9f589dcb4deb7f4ba4fb23f7e85385/python/sglang/srt/models/qwen4_exp_mtp.py
- https://github.com/sgl-project/sglang/blob/78c5024e9d9f589dcb4deb7f4ba4fb23f7e85385/python/sglang/srt/models/qwen4_exp.py
- https://github.com/sgl-project/sglang/blob/78c5024e9d9f589dcb4deb7f4ba4fb23f7e85385/python/sglang/srt/speculative/eagle_worker_v2.py
- https://github.com/sgl-project/sglang/tree/78c5024e9d9f589dcb4deb7f4ba4fb23f7e85385/python/sglang/srt/layers/attention/qsa
- https://huggingface.co/unsloth/Qwen3.8-Flash-Next-GGUF/tree/38bb39ee97821de2c9009abb7e93950eec396e66
- https://huggingface.co/unsloth/Qwen3.8-Flash-Next-GGUF/tree/38bb39ee97821de2c9009abb7e93950eec396e66/UD-IQ1_S
- https://huggingface.co/unsloth/Qwen3.8-Flash-Next-GGUF/tree/38bb39ee97821de2c9009abb7e93950eec396e66/MTP
- https://github.com/ggml-org/llama.cpp/tree/9a4843cf2f1a3fc8e39f8148e92ee6bfe18e2db6
