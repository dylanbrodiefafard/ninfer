# Native calibrated FP8 projection source

Audited 2026-09-18. This source profile is distinct from the earlier row-scaled FP8
experiments. It is a bounded qualification input, not a future model's selected default.

- Producer: `senfu/Qwen3.8-Flash-Next-NVFP4`, revision
  `5d37b3b3711d8406174b96ff950c0aa16324b266`.
- Manifest: https://huggingface.co/senfu/Qwen3.8-Flash-Next-NVFP4/blob/5d37b3b3711d8406174b96ff950c0aa16324b266/quantization-manifest.json
- Environment: https://huggingface.co/senfu/Qwen3.8-Flash-Next-NVFP4/blob/5d37b3b3711d8406174b96ff950c0aa16324b266/conversion_environment.json
- Parent: `RadixArk/Qwen3.8-Flash-Next-NVFP4`, revision
  `7b719225242aacd3dbd3f9407468c2ee9a9d2594`.
- Recorded consumer: SGLang `99c9362e6685db579c469f6e0e566b08827b3477`.
- Consumer scale handling: https://github.com/sgl-project/sglang/blob/99c9362e6685db579c469f6e0e566b08827b3477/python/sglang/srt/layers/quantization/modelopt_quant.py
- Consumer activation packing: https://github.com/sgl-project/sglang/blob/99c9362e6685db579c469f6e0e566b08827b3477/python/sglang/srt/layers/quantization/fp8_utils.py

Each actual projection stores E4M3FN `[N,K]` weights, one FP32 `weight_scale` scalar,
and one FP32 `input_scale` scalar. Both are positive finite dequantization multipliers.
Weight reconstruction is E4M3 * weight_scale. Static A8 packs x / input_scale into
E4M3, then GEMM applies input_scale * weight_scale. A16 uses the represented weight
without activation packing. Do not invert the stored scalar or round it to BF16.

The pinned consumer may merge projections, requantize weights to a maximum weight scale,
and use the maximum activation scale for a fused module. Keeping the original separate
projection code/scalar pairs preserves the artifact, but is not an assertion of identical
private arithmetic to that fused consumer. Config aliases for `qkvz`, `qkv` and `gate_up`
are not extra stored tensors.

## Acquired bounded roles

`tools.parity.qwen4.native_fp8_projection_fixture` preserves 117,309,544 source bytes
across thirteen projections, using exact HTTP ranges without acquiring expert banks:

| Layer | Roles | Shapes |
|---|---|---|
| 0 | GDN QKV, Z, output | `[10240,2560]`, `[6144,2560]`, `[2560,6144]` |
| 3 | QSA Q, K, V, output | `[12288,2560]`, `[512,2560]`, `[512,2560]`, `[2560,6144]` |
| 0, 3 | shared gate, up, down | `[640,2560]`, `[640,2560]`, `[2560,640]` |

Actual source activation scales distinguish boundaries: layer-0 shared gate/up use
`0.01667131669819355`, down `0.02664620615541935`; layer-3 gate/up use
`0.015345982275903225`, down `0.0224609375`. GDN QKV/Z use `0.0376674123108387`,
output `0.005894252099096775`; QSA Q/K/V use `0.01506696455180645`, output
`0.0049874442629516125`. Exact scalar words and all weight scales are in the local fixture JSON.

The manifest names a private `act-amax-combined.json` calibration file and an unpublished
`AMCORE-543 make_candidate.py` producer. It does not provide the calibration corpus or
per-projection activation distributions. These scales are authentic publisher values,
not independently reproduced calibration or proof they suit a future checkpoint.

The optional `--audit-controls` mode compares every remaining BF16 layer-0/3 tensor to
the acquired NVIDIA source before those controls can be reused. It checks source words,
not approximate numeric similarity. It does not compare expert banks or claim whole-model
identity. Native represented-weight oracle tests remain separate from original-BF16
weight error and from publisher/full-model quality evidence.

Actual audit result: all 31 protected tensors across the two layers match the pinned
NVIDIA source byte-for-byte, covering 61,943,744 bytes. This permits reusing those specific
controls in bounded component tests, not assuming equality for other layers or expert banks.

Offline `--convert-source` writes `qwen4-fp8-projections.ninfer`, identity
`qwen4/native-fp8-projection-qualification` / `senfu-fp8-source`, retaining original full
source weight names. Each matrix uses `FP8_E4M3FN_TENSOR_F32M` / `tensor-calibrated-v1`.
Both converter roundtrip and artifact readback exactly preserved all thirteen code planes
and all 26 FP32 scalar words. Python compilation and nine acquisition/PLE-codec regression
tests passed. No GPU arithmetic qualification is implied by these checks.

Decoded FP32 weight reconstruction versus the corresponding local NVIDIA BF16 matrices
has relative L2 between `0.02636947597` and `0.02654217717` across these thirteen roles.
This is a weight-only comparison to that independently pinned source, not activation error,
PPL, or a claim that the unpublished producer used exactly those source revisions.

## Calibrated Linear qualification

On RTX 5090, CUDA 13.1, `sm_120a`, the source-calibrated Linear route preserves
the separate FP32 weight and input scalars and reuses the canonical FP8 GEMV,
small-token, BF16-operand MMA, and E4M3-operand MMA kernels. It does not expand
source scalars into rounded row scales. A16 avoids activation quantization; A8
performs source-floor guarded packing in caller-owned workspace. Its per-token FP32
scale is `max(source_input_scale, RN(maxabs(x_token)/448))`; codes use FP32 RN division
and finite-saturating E4M3 nearest-even conversion. The artifact scalar words never
change. In-range tokens preserve the former static code/scale words; guarding is a
new private arithmetic profile, not publisher execution bit parity.

`ninfer_linear_fp8_tensor_test` checks all seven actual projection geometries and
`--native-real` checks all thirteen source matrices. Both passed, including small
token widths, dispatch boundaries, full/tail MMA panels, graph replay, and a full
output oracle for the 512-row case. The source artifact also traverses generic
binding/materialization, with exact scalar-word readback. The independent naive
FP64 matrix oracle retains decoded coefficients in double, with no incidental
FP32 pre-rounding. Existing predeclared Linear criteria remain unchanged: A16
relative L2 `1/256`, absolute `1/256`, gross `2/256`; A8 relative L2 `.04`, absolute
`1/256`, gross `.06`. These are operator implementation criteria, not PPL budgets.

An independent enumerated E4M3 codec oracle passed exact code and scale checks
for signed zero, subnormals, midpoint ties/neighbors, both signs, and finite
saturation, using both unit and authentic source input scales. The existing
row-scaled A16/A8 suites and generic artifact reader passed after shared kernel
template/FP64-oracle changes; six Python tensor-FP8 codec cases passed.

These tests use deterministic represented BF16 activation panels, not captured
full-model activation distributions. They establish packed-format preservation
and bounded Linear arithmetic; they do not qualify a whole-block A8 recipe,
publisher calibration quality, future-model PPL, or engine throughput.

### Earlier static-profile resident-projection dispatch

The same fixture executable's `--bench-native` mode uses ten warmups and 100
CUDA-event-timed public Linear calls, including A8 packing. These are repeated
resident-matrix measurements (weights may hit L2), not cold whole-model or Engine
measurements. The matching FP8 MMA issue probe measured 506.811 TFLOP/s; the kdev
card for `[10240,2560],T=8` classified DRAM and permitted raising useful work per
weight pass. No new compute family or kernel schedule was introduced.

| Projection geometry | Earlier static A8 first T | A16 / A8 microseconds at T=32 | A16 / A8 microseconds at T=512 |
|---|---:|---:|---:|
| `[10240,2560]` | 9 | 45.05 / 22.56 | 145.50 / 71.66 |
| `[6144,2560]` | 13 | 45.04 / 22.56 | 112.59 / 55.32 |
| `[12288,2560]` | 9 | 75.52 / 22.57 | 180.32 / 86.07 |
| `[512,2560]` | 21 | 44.01 / 22.56 | 45.05 / 24.58 |
| `[2560,6144]` | 24 | 73.74 / 51.21 | 104.43 / 51.35 |
| `[640,2560]` | 21 | 44.61 / 22.55 | 45.04 / 24.56 |
| `[2560,640]` | 9 | 12.31 / 8.21 | 14.34 / 10.26 |

The earlier static profile had its own measured cutovers: copying the row-scaled
route's earlier cutovers regressed several small-token cases after including
static packing. For output projections, T=21 was effectively tied; the retained
T=24 boundary has a clear measured benefit. These cutovers select implementation
arithmetic only when the caller explicitly permits A8; they do not approve A8 for
a model role or substitute for model-distribution/PPL qualification.

## Closed GDN component qualification

The fixed native GDN component now admits the three source-calibrated projections.
Each role has an explicit A16/A8 policy; defaults remain A16. Its caller-owned
workspace includes temporary A8 packing. QKV/Z output, convolution history,
recurrent output, gated-normalization output, and final output retain their
declared BF16 boundaries; recurrence and protected controls remain FP32. The
native BF16 control words are losslessly widened in the existing test-owned view,
and source A_log is converted to the represented `-exp(A_log)` field.

The fixture preserves source scalar words and permutes only code rows/columns
with the same V-head mapping as the native BF16 fixture, consistently with the
protected controls. It does not reconstruct or requantize source weights.
`ninfer_gated_delta_net_layer_test --native-fp8-real` evaluates the existing complete
FP64 formula from public BF16 inputs, with nonzero initial states, T=65 and 64+1
continuation. Every policy is compared directly to that same mathematical oracle.

Before execution, A8 profile limits were declared as relative L2 `.04` for output
and convolution history, `.08` for recurrent state; gross-relative allowances are
`.06`, `.06`, `.12`, retaining existing absolute floors. The state budget covers
multiple Q/K/V perturbation paths in these bounded workloads; it is not an
analytical bound or a PPL budget. A16 output/state gates are unchanged; its
calibrated convolution-history check uses the existing BF16-history gate rather
than the near-exact periodic GGML fixture criterion.

T=65 deterministic-panel results, relative L2 against represented-FP8-weight ideal
(the later authentic-token counterexample below supersedes any model-distribution admission):

| Activation policy | Output | Convolution history | FP32 recurrence | Decision |
|---|---:|---:|---:|---|
| A16 | .002094 | .0000631 | .000664 | admitted |
| QKV A8 | .033599 | .028931 | .031222 | admitted |
| Z A8 | .003260 | .0000631 | .000664 | admitted |
| output A8 | .026427 | .0000631 | .000664 | admitted |
| QKV + Z A8 | .033643 | .028931 | .031222 | admitted |
| Z + output A8 | .026551 | .0000631 | .000664 | admitted |
| QKV + output A8 | .042755 | .028931 | .031222 | rejected |
| all A8 | .042748 | .028931 | .031222 | rejected |

The rejected combinations also failed under 64+1; their gates were not widened.
The public workspace/execution admission rejects simultaneous QKV/output A8.
Z may combine with either one. Actual dispatch at 65/64 activates each selected
A8 route, while the final single-token continuation uses A16. The admitted
partitioned cases have output relative L2 `.002094`–`.033555`. Same-weight
implementation comparisons are supplementary: e.g. QKV-A8 versus A16 output
relative L2 `.033645`, Z-A8 `.003297`, output-A8 `.026462`.

Separately, the complete ideal component with FP8 weights versus the corresponding
NVIDIA BF16-weight ideal has output relative L2 `.040729`, convolution history
`.026784`, and recurrent state `.030161`. This is weight-storage loss, not kernel
error or activation-quantization loss. These deterministic activation/state panels
and component continuations do not establish a full-chain or future-model PPL
budget; default A16 and independent model-distribution qualification remain necessary.

### Authentic token-panel rejection: source Z input scale clips an outlier

The 33-token source embedding → BF16 source GR0 panel rejects the proposed GDN0
Z-A8 profile despite its earlier deterministic-panel pass. The unchanged complete
GDN output screen reported index 39031 (token 15), actual `0.0927734` versus ideal
`0.0197132`; source-calibrated weights with all A16 operands passed the same chain.
The standalone `ninfer_qwen4_gdn_a8_diagnostic` reconstructs this exact input from
`qwen4-text-panel.ninfer` and `qwen4-layer-0.ninfer`, without a full checkpoint.

The exact stored input multiplier `0.0376674123108387` gives a finite E4M3 input
limit of `16.875000715255737`. Exactly one of 84,480 public BF16 inputs exceeds
that limit: token 15 has magnitude `49.75`. This is the exact source EOS token
`248044` between prose and code in `qwen4-text-panel.json`, not an invented large
activation. The publisher scale clips this native-panel input; this does not
establish that the publisher's full model or calibration is globally incorrect.
Independent FP64 Z projection over
all 6,144 output rows at this token separates clipping from rounding:

| Comparison | Relative L2 |
|---|---:|
| GPU A8 versus represented-weight ideal | .212924 |
| Clamp-only input versus ideal | .213346 |
| Exact FP8 quant-decode versus clamp-only | .00676815 |
| Exact FP8 quant-decode versus ideal | .212920 |
| GPU A8 versus exact FP8 quant-decode projection | .00164330 |

The diagnostic shares Linear's independent enumerated finite-saturating E4M3
nearest-even codec oracle. It does not feed packed intermediates into the complete
GDN oracle or relax any gate. The large loss on this panel is static-scale clipping,
not incorrect nibble/byte decode, scale direction, FP32 multiplier handling, or MMA
arithmetic. The old static Z-A8 profile is rejected for this native input distribution. QKV shares this
input multiplier and sees the same input outlier; its earlier random-panel pass
cannot establish native-text suitability either. The guarded private profile above
uses the stored multiplier only as a floor and is qualified separately against the
same complete mathematical oracle and unchanged gates. It does not rewrite authentic
source scales or change the artifact format. One short panel cannot select a
replacement producer calibration recipe or establish PPL safety.

With guarding, the same EOS token selects FP32 scale approximately `0.111049`.
Its GPU Z projection error falls to `.00667229` against the unchanged ideal;
the independently quantized guarded projection error is `.00643817`, and GPU
versus that explicit diagnostic projection is `.00165958`. The original static
clamp/rounding attribution above remains recorded rather than being relabeled a
packing defect. The standalone diagnostic enforces the unchanged Linear A8
`.04` relative, `1/256` absolute, `.06` gross-relative screen against the ideal.

Exact guarded packing tests pass at all input widths 640/2560/6144, covering the
stored-scale floor, all-zero tokens, in-range code/scale identity, guarded large
values, signed zeros, ties/subnormals and output guards. All seven synthetic
projection shapes and thirteen authentic matrices pass the unchanged Linear gates.
The existing complete GDN T65/64+1 and shared-MoE policy panels also pass unchanged;
their in-range results remain identical. These checks qualify this arithmetic
profile; full-chain token-panel and model-quality decisions remain separate.

The guarded real 33-token chain, selecting GDN Z-A8 and shared gate/up/down A8
with QSA A16, passed both whole33 and 32+1 under the unchanged component gates
and 2% accumulated screen. `ninfer_qwen4_native_text_a8_real_test` retains this
bounded regression; it is not a full-model PPL qualification.

An isolated RTX 5090 public-Linear CUDA-event sweep (ten warmups, 100 complete
packing-inclusive calls per policy) measured the guarded profile on all thirteen
actual matrices. At T32, representative A16/guarded-A8 times in microseconds were:

| Role | A16 | Guarded A8 |
|---|---:|---:|
| GDN QKV | 45.04 | 24.60 |
| GDN Z | 46.26 | 24.60 |
| QSA Q | 77.78 | 24.59 |
| QSA K | 45.04 | 24.59 |
| GDN output | 73.74 | 53.26 |
| shared gate | 45.05 | 24.60 |
| shared down | 12.31 | 10.26 |

The guarded reduction adds approximately 2 microseconds versus the earlier static
profile in this warm-resident measurement. QKV T9, Z T13, Q T9, output T24 and
shared-down T9 still have useful measured wins. K/V/shared gate/up at T21–24 are
effectively tied around 24.60 microseconds; at T25 A16 takes about 28.69 versus
24.60 for guarded A8. The guarded `[512,2560]` and `[640,2560]` routes therefore
retain A16 through T24 and first permit A8 at T25. Other cutoffs are unchanged.
The kdev existing-FP8 geometry cards classify both T25 points as DRAM and permit
aggregate-T dispatch work; actual tensor-calibrated public-Op timings above, not
the generic row-scale byte model, determine the cutoffs. No new MMA schedule is
introduced. These are public-Op timings, not cold-cache or model token/s.

## Distinct KV composition assessment and non-admission

The authentic33-token original-BF16 native prefix was captured at layer3, after all preceding
GR/GDN/MoE/PLE boundaries passed their existing oracles. An independent CPU counterfactual
then evaluates QSA through the final layer3 residual with FP8 K and BF16 V, using per-head
FP32 maxabs/448 scales, signed nearest-even E4M3 and explicit BF16 reconstructed operands.
The BF16 calculation is cross-checked against the existing independent QSA formula and
captured component results. This is not a GPU codec or a new runtime cache format.

FP8-K gives aggregate QSA/final residual relative L2 of1.868%/0.995% versus the independently
propagated BF16 reference. These are supplementary whole-panel screens, not the native QSA
Op's per-token, same-represented-input admission gate. Storage-only worst-token QSA drift is
2.054%; worst-token suffix drift is3.906%, with two router-membership changes. QSA's aggregate
gross-error margin is only0.000211. The theoretical K/V payload becomes75.391% of BF16,
excluding unchanged index keys, positions and page metadata.

This is insufficient evidence for FP8 runtime admission: retain BF16 as the numerical
reference, without claiming the candidate can never qualify. It needs a materially stronger
quality case and exact GPU codec/composition qualification before adding a runtime format.
NVFP4 K-only is not a rescue: aggregate QSA/final drift is6.655%/2.341%; NVFP4 both is
11.511%/2.795%. This separates represented-cache loss from the already-qualified diagnostic
NVFP4 codec/kernel behavior. No criterion was widened, and no new production KV path was added.
Report: `out/qwen4-native-kv-composition.json`.

The existing all-role QSA-A8 and FP8-residual candidates remain rejected: their actual dynamic
scaling/formula losses were already measured, so retrying the same recipe would not resolve
an implementation gap. Similarly, guessed DFlash BF16 protection subsets did not rescue the
uncalibrated NVFP4 drafter. Its source BF16 option remains available; calibrated draft quality
needs independent captured features/holdout evidence, not another permutation of one prompt.

## Original-BF16 weight-only endpoint and read candidates

The six finite row-FP8 candidates are distinct from the calibrated tensor-FP8 source above.
`tools.parity.qwen4.native_weight_candidates` uses original BF16 endpoint/PLE artifacts and
authentic captured MTP hidden/final-read inputs. The existing offline encoder stores each row
with E4M3FN codes and a BF16 scale; the independent oracle decodes signed codes with that exact
scale and evaluates the complete formula in FP64. GPU A16 Linear checks cover the full248320
head and final-GR/PLE projection shapes, T3, one-row splitting and graph replay. Complete
final-GR and PLE key-only/both output/history checks pass their existing numerical profiles.
The embedding kernel now supports exact width2560 alongside5120; independent exact decode,
tail rows, signed zero, subnormal scales and repeated graphs pass.

Storage loss is measured separately from kernel correctness. On the same authentic three-row
panel: row-FP8 head source-logit relative L2 is3.595%; final-GR both-FP8 source-read loss is
0.156%; PLE key-only injection/history loss is0.077%/0.122%, versus2.336%/2.241% for value
quantization. Selected token/mask embedding source loss is2.631%, not a whole-vocabulary
quality measurement. All three head argmax values agree, but that is not PPL or acceptance
evidence. Any reported softmax KL is over the physical248320-row head, not the masked248077
token domain or p-less/epsilon sampling distribution.

The more aggressive, explicitly uncalibrated NVFP4 head was also assessed using NInfer's
existing two-level maxabs recipe and independent signed E2M1/E4M3 decode. Source-logit loss
is13.011% (individual inputs8.989–15.367%), versus row-FP8's3.595%, despite identical three
argmaxes. This recipe is rejected for native head admission; no NVFP4 head default or
activation policy was inferred from the dense27B target. No indiscriminate four-bit GR
projection experiment is justified by the research, which specifically identifies those
weights as sensitive.

Reports: `out/qwen4-native-weight-assessment.json`,
`out/qwen4-native-head-nvfp4-assessment.json`. These optional represented-weight paths do not
choose the final smaller-checkpoint recipe. Protected controls and all activation defaults
remain unchanged.

## Closed shared-expert MoE qualification

The resident MoE accepts the source-calibrated shared gate/up/down matrices with
separate explicit A16/A8 policies and per-role source input multipliers. Routed
NVFP4 banks and BF16 router/shared-scalar storage are unchanged. Native router
ranking now retains a two-FP32 summation expansion, as specified in the MoE Op
contract; selected probabilities and the shared scalar gate remain FP32.
Each role uses caller-owned packing scratch; no global format
admission or implicit activation policy was added to other components.

`ninfer_qwen4_native_layer_real_test --native-shared-fp8` composes actual Senfu
shared projections with the audited NVIDIA routed banks and protected controls.
This is a precisely represented bounded component fixture, not an assertion that
every tensor equals the publisher's full checkpoint. The single complete FP64
MoE oracle decodes FP8 coefficients with their exact FP32 weight multiplier; its
ideal SwiGLU does not copy private BF16 staging. Exact top-ten IDs and the existing
probability criterion remain unchanged.

Before candidate execution, shared-A8 qualification was fixed at relative L2
`.04`, gross-relative `.06`, and absolute floor `1/32768`, matching the existing
Linear A8 envelope. Existing A16 and routed-A4 criteria were not changed.
All eight shared activation-policy subsets passed layers 0 and 3 at
T=1/8/9/24/25/65 and 64+1. The widths straddle down's T=9 and gate/up's guarded T=25
dispatch boundaries. Worst per-token represented-weight errors were:

| Shared policy, routed A16 | Layer 0 | Layer 3 |
|---|---:|---:|
| A16 | .003696 | .003580 |
| gate A8 | .008948 | .015997 |
| up A8 | .008621 | .014703 |
| gate + up A8 | .012285 | .022277 |
| down A8 | .020889 | .018209 |
| gate + down A8 | .020889 | .026286 |
| up + down A8 | .020889 | .022453 |
| all A8 | .020889 | .029973 |

All shared subsets also passed with active routed A4 at T65 and 64+1 using its
existing `.16` complete-Op criterion. Worst errors were `.115124` (layer 0) and
`.0992514` (layer 3); IDs confirm groups reach the 32-occurrence activation threshold.
Supplementary comparison against the same routed-A4/shared-A16 implementation
isolates shared activation changes: all-shared-A8 relative L2 `.015953` and
`.029212` respectively. Both implementations are independently checked against
the complete represented-input oracle; their mutual comparison is not the gate.

Separately, replacing only BF16 shared weights by represented FP8 weights in the
ideal complete MoE produces relative L2 `.01823`–`.01869` for layer 0 and
`.02958`–`.03334` for layer 3 across the two deterministic BF16 input patterns.
This weight-storage effect is distinct from kernel and activation errors. No
full-model PPL, text-distribution, future-model default, or speed claim follows.

## Resident component timing and selective activation assessment

The following 33-token timings describe the earlier thirteen-matrix experimental
recipe. The disjoint 137/257-token follow-up in
`qwen4-quantization-recommendations.md` supersedes its storage/A8 admission; current
test commands use the narrowed qualified role selection. Do not treat these historical
measurements as timings for that replacement recipe.

On 2026-09-19, RTX 5090 / CUDA 13.1 / `sm_120a`, the existing native first-four-layer
33-token text panel was measured with device-resident authentic weights. The test-owned
`NINFER_QWEN4_NATIVE_TIMING=1` option reports three warmups and eleven complete-Op
CUDA-event intervals, summarized by median/min/max. Loading, host mathematical oracles,
and GDN snapshot restoration are outside the intervals. QSA rewrites the same visible
append slots. Ordinary oracle calls still run after timing, with restored GDN input state
and unchanged output/state/selection criteria. These are warm-resident component numbers,
not cold-cache bandwidth measurements, Engine throughput, or host-streaming timings.

Run `ninfer_qwen4_native_sequence_real_test --native-text-fp8` and
`--native-text-a8`, with that timing option and `NINFER_QWEN4_NATIVE_LAYERS` set to the
existing source-fixture directory. Both whole 33 and 32+1 passed the existing complete
component oracles and accumulated 2% screen. Whole-33 medians in microseconds:

| Component | Calibrated A16 chain | Selective A8 chain |
|---|---:|---:|
| Layer 0 GDN, FP8 projections | 208.640 | 184.320 (Z only) |
| Layer 0 resident routed-NVFP4 MoE | 1987.390 | 1934.110 (shared gate/up/down only) |
| Layer 1 BF16 GDN | 223.232 | 223.232 |
| Layer 1 resident routed-NVFP4 MoE | 1592.160 | 1591.040 |
| Layer 2 BF16 GDN | 225.280 | 225.312 |
| Layer 2 resident routed-NVFP4 MoE | 1467.740 | 1460.540 |
| Layer 3 QSA, FP8 A16 projections and BF16 cache | 408.448 | 407.264 (still A16) |
| Layer 3 resident routed-NVFP4 MoE | 1034.720 | 973.568 (shared gate/up/down only) |

GDN0 sees the same input and saves 11.7% with its previously qualified Z-A8 route.
Later components see the corresponding propagated chain inputs, and some routing decisions
change; their table differences are not isolated shared-kernel speedups. Routed MoE remains
the dominant measured component. The one-token continuation retains A16 dispatch: calibrated
GDN0 measured 32.736 us in both policies. No protected QSA A8 profile was reintroduced.
The earlier thirteen-matrix packing-inclusive Linear sweep remains the projection-level
evidence; it was not repeated merely because the timing harness is available.

The same timing option on `ninfer_qwen4_native_a4_eos_diagnostic` measures a controlled
same-input routed comparison: actual source EOS token 15's MoE input repeated 65 times,
identical exact routing, BF16 shared weights, ten expert groups with 65 occurrences.
Complete resident MoE median was 783.680 us for A16 and 320.352 us for active A4,
a 2.446x speedup (59.1% less time). Whole 65 and 64+1 still passed their unchanged
complete FP64 oracle gates. Represented-reference relative L2 remained .00351252 for
A16 and .0928886 for A4 whole 65; the latter is within the explicit 16% implementation
profile, not a small-quality-loss or full-model PPL qualification. The concentrated EOS
panel does not predict how often a natural prompt reaches 32 occurrences per expert.

The measured priority is useful token aggregation within the existing resident expert
schedule, not another projection MMA family or blanket A4 admission. Existing kdev cards
for `[640,2560,T65]` NVFP4/A4 and `[6144,2560,T33]` FP8/A8 classify the projection
problems as DRAM-bound and permit aggregate-T investigation; they refuse an unmotivated
compute-family rewrite. Those generic projection floors do not model complete routed
MoE or tensor-calibrated packing. No new kernel, calibration, cutoff, or default precision
was changed from this measurement; any future routed-kernel change needs an exact-point
traffic/issue profile and its own unchanged-oracle check.

### Expanded held-out resident precision qualification

The disjoint source study is documented in `qwen4-quantization-recommendations.md`.
Its retained layer0 shared-up NVFP4/A16 candidate was checked against the complete
independent represented-weight FP64 MoE on the 257-token `heldout_algorithms`
actual input, with all other shared weights BF16 and routed experts A16. The
fitted candidate, max-abs control and original BF16 path all pass unchanged
per-token output/probability gates, exact expert order and buffer guards. No
new shared-projection kernel was needed.

On RTX5090/CUDA13.1, complete resident-MoE medians (three warmups, eleven samples;
loading and host oracle excluded), before the FP32 routed-down repair below,
are BF16 `3127.52 us`, fitted NVFP4 `3148.26 us`,
and max-abs NVFP4 `3149.57 us`. This is a storage tradeoff, **not a speedup**:
shared-up payload falls from3,276,800 to921,604 bytes (71.875% less), with about0.7%
more complete-Op time in this measurement. Conversion retains the fitted candidate
with its calibration provenance; max-abs remains the experimental control. Shared
A4 and mixed FP8+NVFP4 shared compositions are not admitted by this evidence.

On the same actual 257-token multilingual GDN inputs, source tensor-FP8 Z0 only
(QKV/output BF16) with compensated QKV measures `626.688 us` at A16 versus
`591.840 us` at guarded A8: 5.56% less complete-GDN time. Row-FP8 Z2 measures
`645.088 us` at A16 versus `606.176 us` at dynamic A8 (6.03% less).
These are independent same-input Op A/Bs, not propagated
chain comparisons. Both use the unchanged represented-weight A8 output criterion
and unchanged convolution/FP32 recurrent-state gates; Z precision does not alter
the recurrent update. Row-Z2 codes and BF16 row scales are both head-permuted
offline and exactly checked. Its payload is15,740,928 bytes versus31,457,280 BF16.
The finite row-Z A8 route reuses existing Linear kernels; row-QKV/output A8 and
simultaneous other-projection A8 are not admitted. No end-to-end speedup is claimed.

The strengthened checks compare every one of the257 output tokens directly to
the same independent complete-GDN oracle, not just a panel aggregate. Before the
compensated-QKV repair, the recorded Z0 A8
whole/worst-token relative L2 is0.5650%/0.7878%; row-Z2 A8 is1.0675%/1.4478%.
Both pass the unchanged4% implementation criterion and gross bound, along with
the unchanged convolution-history and FP32 recurrent-state gates. These
represented-weight implementation errors are separate from the2% original-source
quality screens used to select the candidates. Both A16/A8 Z0 and row-Z2 cases
also pass these unchanged per-token/state gates with the final compensated-QKV
implementation used for the timings above.

The expanded mixed-shared layer2 A16 check exposed an implementation failure on
multilingual token114, coordinate2392: GPU `-0.0849609375`, independent FP64
`-0.08569099564753313`, absolute error0.0007300581 versus unchanged gross limit
0.0007026038. Attribution reproduced the result from private BF16 routed-down
stores before weighted mixing. Retaining those results in FP32 removes that
premature rounding; changing only shared-up staging would not fix this witness.
The native NVFP4 A16 path now keeps FP32 down slots through its rank-ordered
weighted sum, while preserving public BF16 output, gate/up/SwiGLU arithmetic,
and all other expert-format/policy profiles. The original257-token mixed-shared
layer2 A16 witness passes the unchanged per-token gates, exact expert order,
probabilities and guards after the repair. Its complete-Op median is3089.18us
(3086.18–3096.42us), not a claim of a general throughput improvement.

All four selected shared policies now pass direct per-token complete-MoE oracles
with both A16 and their selected A8 activations after that repair. The panel for
each layer is its worst-token source-loss witness among the four held-out
documents, and each timing pair uses identical represented input and weights.

| Layer / held-out panel | Shared FP8 weights | A8 role | A16 / A8 complete MoE (us) |
|---|---|---|---:|
| 0 / reasoning | gate, up | up | 3102.40 / 3090.24 |
| 1 / algorithms | gate | gate | 2918.94 / 2896.45 |
| 2 / multilingual | gate | gate | 3089.18 / 3074.85 |
| 3 / algorithms | gate | gate | 2735.90 / 2721.02 |

The modest0.39–0.77% complete-Op reductions do not establish cold-cache or
whole-model speedups. Hardware and timing methodology are the same as above;
the A16 criterion remains `{2.5/255,1/32768,2/255}` and shared-A8 remains
`{.04,1/32768,.06}`, with exact expert order and unchanged probability/guard checks.
Results are in `out/qwen4-projection-study/shared-final-op-qualification.json`.
The final FP32-down profile also rechecks the original BF16 and fitted NVFP4
shared-up0 candidates on the same257-token algorithms input: both pass unchanged
gates, with complete-MoE medians3109.47us and3134.18us respectively. The fitted
storage option remains about0.8% slower at this point, not a speed optimization.

### Retained resident-MoE occupancy optimization

The follow-up profiled the complete resident layer0 MoE on the same33-token panel, then the
three attributed NVFP4 grouped kernels. Before changing code, `kdev recipe` classified the
scalar NVFP4/A16 point as `profile-required`; its dense single-matrix floor is not a complete
MoE roof. Nsight Systems attributed491/483/833us to gate/up/down and16.5us to the serial
512-expert prefix sum. Nsight Compute reported128 registers/thread, about33% active warps,
and5.61billion predicated-on FFMA instructions per projection. Sparse groups were doing a
sixteen-token tile's arithmetic even when only one or two occurrences were present.

The retained route dispatches disjoint occurrence-count ranges: counts1–2 use the existing
two-token NVFP4/A16 Linear schedule, and larger groups retain the sixteen-token reuse
schedule. Both skip the same eligible A4 groups when that explicit policy is enabled.
An exact512-thread prefix scan replaces the serial integer scan. No codec, represented
weights, router, accumulation criterion or activation cutoff changed. This is SM120 scalar
schedule work, not a new MMA family. The kdev scalar-card printer was fixed to handle the
already-classified `profile-required`/no-MMA case; classifier acceptance was not loosened.

The final whole33 plus32/1 execution passes the independent local and accumulated2% checks.
Warm-resident complete layer0 MoE medians are992.832us versus1987.30us before, approximately
2.00x faster; the natural one-token continuation is145.600us versus163.616us, about11.0%
less time. Layers1/2/3 whole33 measured931.680/883.584/774.048us. The earlier narrowed
one-layer harness measured901.088us for the first tile candidate; do not substitute that
number for the final four-layer workload. These are component, not Engine or streamed-model,
measurements on RTX5090/CUDA13.1.2, three warmups and11 CUDA-event samples per median.

Commands: `NINFER_QWEN4_NATIVE_TIMING=1` with the actual native fixture and
`ninfer_qwen4_native_sequence_real_test --native-text-fp8`. Attribution artifacts are
`profiles/nsys/qwen4-resident-moe/native33-baseline.nsys-rep` and
`profiles/ncu/qwen4-resident-moe/native33-baseline.csv`. Profiler-instrumented kernel durations
are not mixed with the unprofiled whole-Op timings above.
