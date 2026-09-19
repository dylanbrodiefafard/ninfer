# Qwen4 weight, activation and state precision recommendations

Research conducted 2026-09-17; recommendations recorded 2026-09-18.

This is a research recommendation for a **future smaller Qwen4 checkpoint**, using
Qwen3.8-Flash-Next as the available architectural and quantization evidence. It is not a
qualified artifact recipe, an implementation assignment, or a replacement for the model,
artifact and Op contracts in `docs/maintainer/qwen4-*.md`. Apply each row only if the future
checkpoint retains that component. Architecture implementation is separate work.

## Interpretation

- **A4/A8 describe temporary GEMM operands**, not the storage dtype of every intermediate.
  Ordinary activation outputs remain BF16 unless a separately qualified fused route chooses
  private intermediate arithmetic. Weight storage, accumulation and persistent state are
  separate decisions.
- **Candidate** means supported enough to evaluate, not already proven to preserve maximum
  quality on the future checkpoint. Its original unquantized execution is the quality reference.
- FP32 accumulation/reduction entries are recommended implementation profiles where stated;
  they do not redefine the mathematical oracle or assert that all reference arithmetic is FP32.
- NInfer's current Qwen3.8-27B qualifies A4 computation for particular matrix shapes and
  operators. Its cutovers and protected-layer list do not transfer automatically to Qwen4.
- Weight-only quantization results do not establish activation-quantization tolerance.
  NVFP4, GPTQ INT4 and MLX INT4 are different numerical profiles.
- A4 can be numerically usable without being the fastest route at a small token count.

## Weight and activation matrix

| Component | Recommended weight storage | Recommended activation/computation | Lower-precision alternative and evidence |
|---|---|---|---|
| Routed expert gate/up projections | **NVFP4** | **A4**, FP32 accumulation, BF16 output | Strongest direct Flash-Next W4A4 evidence; NInfer also qualifies these operations. Retain A16 routes where faster at small token counts. |
| Routed expert down projection | **NVFP4** | **A4**, independently calibrated post-SwiGLU input; FP32 accumulation | Supported by expert W4A4 recipes and NInfer. Do not reuse gate/up activation calibration blindly. |
| Expert activation: SiLU and multiplication | No weights | BF16 output; fused internal arithmetic separately qualified | No need to store the intermediate persistently in A4. Quantize at the down-GEMM boundary. |
| Expert output weighting and combination | Router coefficients retained at appropriate precision | FP32 weighted accumulation, BF16 output | Avoid low-bit partial-sum storage. |
| Routed-expert router | **BF16** | BF16 input/projection boundary; **FP32 softmax**, exact top-k indices | Keep protected. Reference behavior does not imply that router logits are produced or stored entirely in FP32. |
| Shared-expert gate/up projections | **FP8 candidate; BF16 reference** | **A8 candidate**, BF16 output | Direct community W8A8 evidence. W4A4 deserves testing, but NInfer's dense MLP results alone do not qualify it. |
| Shared-expert down projection | **FP8 candidate; BF16 reference** | **A8 with separate post-activation calibration** | Same distinction; calibrated W4A16 is another memory-oriented candidate. |
| Shared-expert scalar output gate | **BF16** | BF16 input; FP32 accumulation/nonlinearity where appropriate | Keep protected separately from the shared MLP. |
| QSA query projection | **FP8 candidate; BF16 reference** | **A8 candidate**, BF16 projected Q | W4A16 has limited evidence; W4A4 is a further experiment supported by analogy to NInfer. |
| QSA key projection | **FP8 candidate; BF16 reference** | **A8 candidate**, BF16 projected K | Qualify separately: key errors affect attention scores and cached history. |
| QSA value projection | **FP8 candidate; BF16 reference** | **A8 candidate**, BF16 projected V | Evaluate independently from Q/K rather than assuming equal sensitivity. |
| QSA output projection | **FP8 candidate; BF16 reference** | **A8 candidate**, BF16 residual boundary | A4 is worth testing; current NInfer retains some higher-precision output projections. |
| QSA Q/K normalization and positional rotation | BF16 norm parameters | BF16 represented outputs; FP32 normalization reductions | Preserve operation order and cast boundaries. |
| QSA core attention QK/PV | No additional projection weights | **BF16 operands initially; FP32 score/softmax reductions** | FP8 attention operands require separate qualification. W8A8 projection evidence does not establish this. |
| QSA indexer projections | **BF16** | BF16 projected queries/keys | Keep protected initially; changing block selection can discard relevant context. |
| QSA indexer pooling and scoring | BF16 norm parameters | **FP32 pooling/score reductions**, exact selected indices | Matches the important reference precision boundaries. |
| GDN Q/K/V projections | **FP8 candidate; BF16 reference** | **A8 candidate**, BF16 operator outputs | NInfer justifies investigating W4A4; Flash-Next evidence is stronger for W8A8. |
| GDN Z/output-gate projection | **FP8 candidate; BF16 reference** | **A8 candidate**, higher-precision gate evaluation | Qualify separately from Q/K/V despite possible fusion. |
| GDN output projection | **FP8 candidate; BF16 reference** | **A8 candidate**, BF16 residual output | A4 remains a candidate, not an established Qwen4 default. |
| GDN A/B control projections | **BF16** | BF16 inputs; **FP32 decay/update computation** | Keep protected. |
| GDN decay parameters and time-step bias | Preserve source information; FP32 runtime representation where used | **FP32 exponentials/softplus/control arithmetic** | Tiny storage savings do not justify aggressive quantization. |
| GDN convolution | **BF16** | BF16 history boundary; appropriate FP32 accumulation | Do not silently change the recurrent history representation. |
| GDN normalization and output gating | BF16 parameters | BF16 output; FP32 reductions/control evaluation | Separate from projection operand precision. |
| Token embedding | **Eight-bit weight candidate; BF16 reference** | **BF16 gathered output** | Weight-only evidence supports evaluating eight bits. A gather has no GEMM activation operand. |
| PLE n-gram embedding table | **FP8** | **BF16 gathered values** | Four-bit storage is a more aggressive capacity candidate; runtime dequantization placement matters. |
| PLE key/value projections | **BF16 initially** | **A16 initially** | Eight-bit weights are supported as a candidate; A8 needs separate activation evidence. |
| PLE content gate, norms and convolution | **BF16** | BF16 represented values; FP32 reductions/nonlinearities | Protect independently from the large lookup table. |
| Hyperconnection read projections | **BF16 initially** | **A16** | Eight-bit weights are a credible next step. Weight ablations specifically argue against indiscriminate four-bit storage; they do not qualify A8. |
| Hyperconnection write gates, mixing parameters and norms | **BF16** | BF16 represented values; FP32 reductions where appropriate | Keep protected. |
| LM head | **Eight-bit weight candidate; BF16 reference** | **A16**, FP32 accumulation where appropriate | W4A16 is a more aggressive demonstrated candidate; A4/A8 head inputs are not established here. |
| Vision encoder and multimodal merger | **BF16 initially** | **A16 initially** | NInfer's quantized vision paths are useful implementation experience, but text-only Flash-Next results do not qualify their quality. |
| MTP projections and experts | Preserve a validated checkpoint-specific mixed recipe | Preserve its validated activation policy | Evaluate draft acceptance and total throughput; do not inherit the 27B DFlash recipe automatically. |

## Persistent and cached values

| Stored value | Recommendation | Alternative |
|---|---|---|
| Ordinary hidden activations/residual outputs | **BF16** | Local A4/A8 GEMM packing, without changing the persistent boundary |
| Gated multi-stream residuals | **BF16 baseline** | **FP8 storage is a priority experiment supported by the paper**; gate/mixing computation stays higher precision |
| GDN recurrent matrix | **FP32** | BF16 only after specific long-sequence state/quality qualification |
| GDN convolution history | **BF16** | No established reason here to lower it |
| QSA main K/V cache | **BF16 baseline** | FP8 after long-context quality and kernel qualification; NVFP4 is not justified by the external quality evidence reviewed here |
| QSA indexer keys | **BF16** | Quantization requires separate block-selection evidence |
| Indexer partial pooling accumulators | **FP32** | Do not conflate these with the stored key dtype |
| PLE convolution/history values | **BF16** | Qualify separately from table compression |
| Sampling probabilities and sensitive reductions | **FP32** | Keep distinct from the stored logits dtype |
| Token IDs, hashes, positions and selected indices | **Exact integers** | No approximate quantization |

The sandbox's separately authorized diagnostic NVFP4 QSA-KV profile is not superseded by this
research recommendation. A diagnostic implementation and evidence of maximum-quality suitability
are different claims.

## Connection to current NInfer

The inspected Qwen3.8-27B NVFP4 recipe has 247 NVFP4 Text matrix parents, nine protected BF16
large projection parents, W8 vocabulary endpoints, BF16 GDN control weights and FP32 recurrent
state. Its public activations are generally BF16. Eligible projection/MLP routes quantize
operands to A4 locally; ordinary single-token NVFP4 linear decode uses A16. Short recurrent and
speculative routes have additional arithmetic/state-boundary rules.

That provides direct implementation evidence for A4 gate/up, post-SwiGLU down, attention and
large GDN projections. It supports testing those same *kinds* of boundaries in Qwen4. It does
not prove that a different checkpoint, shared expert, sparse selector or hyperconnection has
the same sensitivity. In particular:

- Expert W4A4 has both analogous NInfer experience and direct Flash-Next evidence.
- QSA/GDN/shared large projections retain W4A4 as a serious candidate, while W8A8 has stronger
  direct Flash-Next activation evidence.
- QSA core attention operands and cache codecs require separate qualification from its
  projection GEMMs.
- GDN state/control precision is not inferred from large-projection precision.
- No routed router, QSA indexer, PLE or Qwen4 hyperconnection exists in the current dense 27B
  target; their recommendations rely on the external evidence and numerical reasoning.

Relevant repository references: `docs/maintainer/qwen3.6-27b-artifact.md`,
`docs/maintainer/qwen3.8-27b-artifact.md`, `docs/maintainer/qwen3.6-27b-model.md`,
`src/ops/linear/nvfp4/nvfp4_config.h`, the NVFP4 Linear/SwiGLU/LinearAdd route planners,
and `src/targets/qwen3_6/impl/runtime/dflash_impl.h`.

## Evidence and limits

The evidence below was researched on 2026-09-17; publisher experiments were not reproduced.
Source branches and model cards can change. Different publishers' numbers are not a matched
Pareto comparison.

- **NVIDIA Flash-Next NVFP4:** main routed experts use W4A4 with MSE-calibrated scales while
  other main-model paths retain BF16. Its broad benchmark comparison supports this mixed
  compression point, not uniform W4A4 or guaranteed losslessness for a smaller model.
  https://huggingface.co/nvidia/Qwen3.8-Flash-Next-NVFP4
- **PipeNetwork weight ablations:** matched WikiText2 PPL is 4.4708 for BF16, 5.3914 for uniform
  INT4 and 4.5286 for mixed four/eight-bit. Restoring hyperconnection read projections alone
  to eight bits gives 4.9744; restoring the attention/GDN/shared/PLE projection group gives
  4.8969. This identifies weight sensitivity, not A8 activation tolerance. The mixed recipe
  is larger, so it is not strict domination on every memory/quality objective.
  https://huggingface.co/pipenetwork/Qwen3.8-Flash-Next-MLX-mixed-4_8bit
- **Senfu dense W8A8:** QSA and large GDN/shared projections use FP8 W8A8, with W4A16 head.
  The evaluation is narrower, uses two B200s, and compares against published baseline numbers
  rather than a paired rerun. Claimed weight-traffic savings are calculated, not measured
  throughput. Activation calibration is described through MoE-block input maxima without
  adequate per-projection characterization, especially for post-activation down inputs.
  https://huggingface.co/senfu/Qwen3.8-Flash-Next-NVFP4
- **GPTQ:** routed/shared experts and QSA projections have a reported INT4 weight-only point
  with small corpus-level perplexity changes. This does not qualify NVFP4 A4 computation.
  https://huggingface.co/btbtyler09/Qwen3.8-Flash-Next-GPTQ-4bit/raw/main/README.md
- **Architecture paper:** reports FP8 gated-residual storage with little quality loss, but
  does not establish FP8 gate math, FP8 recurrent state or a universally applicable storage
  codec. *On the Design of Qwen3.8-Next Architecture: Evaluation, Efficiency, and Training
  Stability*, arXiv 2608.30320v1.
  https://arxiv.org/html/2608.30320v1
- **Official configuration and reference:** FP32 GDN state, protected conversion exclusions,
  FP32 router softmax and FP32 indexer reductions inform the recommended control boundaries.
  https://huggingface.co/Qwen/Qwen3.8-Flash-Next-FP8/raw/main/config.json
  https://raw.githubusercontent.com/huggingface/transformers/main/src/transformers/models/qwen4_exp/modeling_qwen4_exp.py
- **PLE storage versus runtime representation:** the RadixArk recipe distinguishes FP8 PLE
  storage from BF16 dequantized use. Check where decoding occurs before making a residency
  or bandwidth claim.
  https://huggingface.co/RadixArk/Qwen3.8-Flash-Next-NVFP4

## Decision priorities

The recommended aggressive starting point is **expert W4A4**. For shared experts, QSA
projections and large GDN projections, **W8A8 is the next evidence-backed candidate**, with
W4A4 explicitly retained for evaluation. Protect router/indexer decisions, hyperconnection
gates and GDN state. Evaluate FP8 gated-residual storage separately from its computation.

There is no established maximum-quality/speed frontier for the future smaller model. Qualify
routes against independent mathematical oracles from represented inputs and decoded weights;
assess weight-quantization loss separately against the original checkpoint. Use paired held-out
quality measurements, long-context retrieval, recurrent behavior and termination where affected.
Measure actual prefill/decode throughput at relevant token counts and concurrency, including
packing overhead and speculative acceptance, rather than assuming lower bits are faster.

### Resident projection follow-up protocol

`tools/parity/qwen4/native_projection_candidates.py` separates `--fit-only` from
`--frozen-fit`: the first reads only calibration captures and freezes weight codes,
stored scales and A4 divisors; the second reads those artifacts without refitting.
The finite original-BF16 NVFP4 candidate retains the matrix FP32 divisor and chooses
each K16 E4M3 scale from factors `{1, .8, .9, 1.1, 1.2}` by calibration-input-energy
weighted squared weight error. It reports the existing max-abs recipe separately.
A4 divisors use the same finite factor set and calibration projection error; the
post-SwiGLU down operand is derived independently rather than reusing gate inputs.
This is a target-private experimental recipe, not a new canonical encoder.
The finite shared policies are NVFP4 gate-only, up-only, down-only or all three
weights, with A16 or A4 on the selected roles. Source tensor-FP8 uses weight masks
`1..7` and A8 only on subsets of each weight mask (26 policies, bits gate/up/down);
row-FP8 uses all three weights and A8 masks `0..7`. The tensor-FP8 mixed-weight
extension was requested after the first layer-0/1 all-weight evaluations finished
but before their results were inspected; it did not change the frozen fits or
the following criteria.

Before held-out evaluation, complete GDN and shared-contribution/full-MoE source
screens are fixed at relative L2 `0.02` and gross error
`0.005 + 0.02 * max(abs(reference))`, both for the panel and every token. Shared-path
relative error alone is diagnostic: its contribution is also measured against the
complete source MoE result. QSA retains its stricter per-token gross floor `2.5e-4`.
GDN convolution retains `{1/256, 0, 1/128}` and recurrent state retains
`{0.0065, 2e-5, 0.004}`, including every recorded transition. QSA BF16 K/V retains
the per-K16 `max(abs(reference))/256 + 1e-4` bound and protected selector membership.
Every held-out document, token tail and state check must pass. These source-loss
screens do not replace independent represented-weight GPU arithmetic qualification.

The additional exact Senfu fixture contains only layer-1/2 Z and shared gate/up/down
codes and their original FP32 weight/input multipliers. Its producer calibration
corpus is unknown; disjoint local calibration/holdout panels do not establish
non-overlap with that external corpus. Row-FP8 experiments remain separate from
this guarded tensor-FP8 A8 profile. Protected controls, BF16 public boundaries and
FP32 recurrent state are unchanged; none of these resident experiments admits a
complete model or establishes PPL or end-to-end performance.

The completed four-calibration/four-heldout study (137/257 tokens per document)
retains the following **source-loss candidates**, not automatic runtime admissions:

| Projection | Retained source policies across every held-out document |
|---|---|
| GDN Z0 | Original producer tensor-FP8, A16 or guarded A8 |
| GDN Z1 | Original producer tensor-FP8 A16; A8 fails gross error |
| GDN Z2 | Original-BF16 row-FP8 A16 or dynamic A8; producer tensor-FP8 fails even A16 |
| Shared layer 0 | Tensor-FP8 `(weight mask, A8 mask)`: `(1,0)`, `(2,0)`, `(2,2)`, `(3,0)`, `(3,2)`, `(4,0)`, `(6,0)`, `(6,2)`, `(7,0)` |
| Shared layer 1 | Tensor-FP8: `(1,0)`, `(1,1)`, `(2,0)`, `(2,2)`, `(4,0)`, `(5,0)` |
| Shared layer 2 | Tensor-FP8: `(1,0)`, `(1,1)`, `(2,0)`, `(3,0)`, `(4,0)`, `(5,0)` |
| Shared layer 3 | Tensor-FP8: `(1,0)`, `(1,1)`, `(2,0)`, `(3,0)` |
| Shared layer 0 up only | NVFP4 W4A16, both max-abs and fitted; fitted worst-panel/token L2 `0.6157% / 1.0849%` |

Mask bits are gate/up/down; unset weight bits retain original BF16, and A8 applies
only to the indicated FP8 roles. In particular, all-three-FP8/all-A8 shared layer 0
fails with worst-token L2 `2.9654%` and 114 gross-token failures; layer 3 fails at
`3.6124%`. All-three-FP8/A16 layer 3 also fails (`2.4684%`). These expanded-corpus
results supersede the earlier short-panel evidence for the unrestricted layer-0/3
shared A8 recipe. Layer-2 gate+up FP8/A16 is a narrow source-screen pass
(`1.9994%` worst token), not evidence of quality margin. No all-three row-FP8 shared
policy passes for layer 1 or 2. All tested calibrated NVFP4 GDN/QSA policies, all
other shared NVFP4 policies, and every tested producer tensor-FP8 QSA policy fail
the unchanged complete-output or state screen; A4 is not retained.

Reports are `out/qwen4-projection-study/layerN.fit.json`,
`layerN.attention.heldout.json` and `layerN.shared.mixed.heldout.json` for `N=0..3`.
The shared comparison uses independent same-actual-input complete FP64 MoE outputs
from `out/qwen4-routed-corpus-oracles`, adding only the mathematically evaluated
shared-path change. Seven original source-prefix captures passed; the multilingual
capture had an accumulated hard-router discontinuity before the QKV repair below.
Those initial GDN checks
used whole-panel output gates; they did not establish every-token arithmetic admission.
Its same-input counterfactual remains valid, but the corpus does not establish
unconditional propagated-chain admission. GPU arithmetic qualification, supported
format/shape binding and direct performance evidence remain separate gates for
the retained candidates.

Strengthening the unchanged GDN output criterion to every token exposed a BF16
QKV accumulation defect. With the final routed-down profile, layer-1 multilingual
tokens 20/241 at coordinate 1883 produced `-.37890625 / -.55078125` against the
independent FP64 results `-.3743690129 / -.5550180498`. The same mathematical suffix
conditioned on the actual represented QKV/Z produced `-.3795648420 / -.5498464972`;
normalization/final-dot effects were at most approximately `1.7e-6`. Source heads
37/38 had 14/12 raw-QKV disagreements through token 20, none explained by exact
FP64-to-FP32-to-BF16 double rounding; through token 241 the counts were 129/107,
with only 1/0 explained that way. Long FP32 projection accumulation, amplified by
later BF16 boundaries, was therefore the dominant cause. Native BF16 QKV now uses
short-K16 MMA partials with compensated accumulation, or compensated existing
small-token SIMT, and rounds the final high/low pair directly to BF16. Generic
Linear, Z, the declared BF16 ingress/history/replay boundary, and FP32 recurrent
state are unchanged. On RTX 5090/sm_120a with CUDA 13.1, all four original held-out
GDN1 panels and the new failing input pass the original whole/per-token/state gates.
The concrete regression is `--gdn-input 1 INPUT_F32` using
`out/qwen4-resident-captures/heldout_multilingual_fp32_down/layer1.attn_input.actual.f32`;
the exploratory intermediate-capture path was removed after attribution.

The qualified GDN1 complete-Op median is `630.752 us` at `T=257`, versus
`559.104 us` for the prior profile that fails those two token gates. At `T=1`,
compensated existing SIMT costs `96.256 us` versus the prior `94.144 us`; the
initial all-width MMA candidate cost `122.848 us` and was not retained for decode.
The native accuracy profile uses existing SIMT schedules through `T=20` and the
existing MMA schedule above that point, without changing generic Linear's cutoff.
The finite crossover checks pass the same complete oracle: `T=21` improves from
`245.792` to `161.824 us`, `T=24` from `278.528` to `167.968 us`, and `T=27` from
`311.328` to `178.208 us`. Actual-source checks also pass at `T=1,2,4,16,20,28`.
These are direct closed-GDN CUDA-event measurements, not Engine performance claims.

The final corrected four-block multilingual prefix (`T=257`, original BF16
projections, native W4A16 experts, BF16 KV and the bounded `t+axis` positions)
passes every unchanged local output/token/state and accumulated panel/gross gate.
Its capture is `out/qwen4-resident-captures/heldout_multilingual_accurate_qkv`.
Final residual relative L2 is `1.026423%`, maximum absolute error `0.0126953125`,
and worst-token relative L2 `4.152011%` at token109. The former token120 gross
failure is resolved, but this is **not** a per-token2% accumulated-chain guarantee:
upstream rounding still changes some selected expert sets. The original source
study inputs remain immutable represented-input counterfactuals, not regenerated
calibration data. Final seven-weight selective-A8 integration and whole33/32+1
continuation also pass; none of these checks establishes full-model PPL.

Final regression qualification: all-target build,32 focused Python checks,
independent Astra review, and the GPU-isolated C++ suite (126 passed,9 optional
fixture-dependent skips,0 failures) pass. One optional Python source-MTP check was
skipped. The C++ gate used `./scripts/run-unit-tests.sh` in
`ninfer-builder-qwen4-mixed` on RTX5090/CUDA13.1. The newly guarded MoE test needed
explicit ordering of default-stream initialization before nonblocking execution;
the preserved pre-fix binary reproduced the failure and the setup-only fix passed.
No numerical criterion was relaxed. Future checkpoint binding, full-model PPL,
long-context quality and end-to-end speculative acceptance remain model-level gates.

Row-FP8 Z2 A16 additionally passes the independent complete-GDN GPU oracle on
the worst-panel source-loss witness, the authentic multilingual `T=257` input.
The frozen source-order codes and BF16 row scales are permuted together into
native V-head order; all `15,740,928` bytes match the existing converter's
original-BF16 row encoder after native preparation. A16 requires no kernel change.
Output uses the unchanged `{.009, 2.5e-4, .005}` criterion,
convolution `{1/256, 0, 1/128}`, and final FP32 state `{.0065, 2e-5, .004}`.
The original-BF16 baseline passes the same checks. Before the compensated-QKV
repair, on RTX 5090/sm_120a, complete resident GDN CUDA-event medians over 11
samples after three warmups were
`573.440 us` for row-FP8 Z2 and `569.344 us` for BF16; state restoration is outside
the interval. This is a storage reduction of `15,716,352` bytes, not a measured
speed improvement. The bounded command is
`ninfer_qwen4_native_kv_assessment_capture --gdn-row-fp8-z2 INPUT_F32 CANDIDATE_NINFER`,
with `NINFER_QWEN4_NATIVE_TIMING=1`; its source baseline is `--gdn-input 2 INPUT_F32`.

The source-retained row-Z2 A8 case also passes the independent whole-GDN oracle
using the existing A8 output criterion `{.04, 2.5e-4, .06}` and unchanged
convolution/recurrent-state criteria, including every token. With the final
compensated-QKV profile, its complete-Op median is `606.176 us`
(`604.160..606.208 us`), versus row-A16 `645.088 us` (`643.072..653.312 us`):
`6.03%` lower, clearing the predeclared `5%` improvement gate with disjoint ranges.
This retains a finite closed-Op admission: row-FP8 **Z only** may use dynamic A8,
and QKV/output must remain A16; row-FP8 QKV/output A8 remains rejected. It reuses
the existing Linear A8 kernel and caller-owned packing scratch. The command adds
a final `1` to the row-Z2 invocation above. Z0-only producer tensor-FP8 likewise
passes A16 and A8 on the same multilingual panel, with complete-GDN medians
`626.688 us` and `591.840 us` respectively (`5.56%` lower); these checks leave QKV/output BF16,
unlike the superseded all-three-FP8 short-panel witness. Source-loss screens,
represented-weight arithmetic gates and measured performance remain distinct.
The earlier pre-compensation untimed reruns also enforced the same output criterion
on each of all 257 tokens. Those historical residual measurements were:
row-Z2 A8 whole/worst-token relative L2 `1.067533% / 1.447816%` and
maximum token gross-error/bound ratio `0.395384`; tensor-Z0 A8 has
`0.565048% / 0.787760%` and ratio `0.300223`. Row-Z2 convolution/recurrent-state
relative L2 is `0.007598% / 0.010828%`; tensor-Z0 is `0.011645% / 0.007184%`.
The final-profile timed reruns pass every unchanged output/token/state gate as well.
These arithmetic comparisons use the exact decoded candidate weights and
represented BF16 inputs, not original-BF16 source weights.

### Routed activation calibration and endpoint follow-up

The same disjoint corpus tests routed gate/up and independently derived
post-SwiGLU down activation multipliers. Each eligible expert fits only calibration
occurrences over factors `{1,.5,.75,1.25,1.5,2}`; eligibility requires at least eight
calibration occurrences and 32 occurrences in a held-out panel, matching the actual
A4 dispatch threshold. Every held-out expert contributes to the independent
same-input FP64 complete MoE, including original BF16 shared weights; smaller groups
remain ideal A16. This is activation source-loss evidence, not a GPU A4 oracle.
All 16 original A16 GPU document/layer outputs pass their unchanged local oracle;
all 16 source and fitted A4 complete-MoE source screens fail. Fitting makes only
small improvements, with worst-token fitted relative L2 by layer
`15.4902%, 10.0812%, 10.3098%, 13.1406%`. No fitted multipliers are promoted.
The original explicitly experimental routed-A4 arithmetic route remains separately
qualified against its represented-input oracle; these results do not admit it as
a small-loss default. Report: `out/qwen4-routed-corpus.json`.

Expanded weight-only FP8 evidence preserves BF16 gathered/public values and A16
GEMM inputs. Across the four held-out native panels, row-FP8 embedding source relative L2
is `2.638–2.645%` (worst token `2.774%`). PLE key-only output relative L2 is
`0.0708–0.1751%`, value-only `0.6693–0.9022%`, and both `0.6713–0.9170%`;
their worst-token values reach `1.3911%, 2.0976%, 2.2534%`, respectively. Small
panel averages therefore do not establish uniformly small token loss. These are
projection changes over unchanged exact FP8 table rows, not PLE table codec tests;
the initial BF16/A16 PLE projection recommendation is retained. Report:
`out/qwen4-ple-corpus-report.json`.

For full-depth endpoint inputs, the separately authorized diagnostic model captured
128 accepted tokens from each of eight independently reset documents. Its complete
28.8 GB PLE payload was eagerly OS-locked; a dedicated container recorded zero
swap use. This is the UD-IQ1_S diagnostic target, not native NVFP4 model execution.
On the four held-out documents, row-FP8 final GR-read weights produce panel
relative L2 `0.1382–0.2085%`, with worst-token `1.2254%`. The row-FP8 LM head
at positions 31, 63 and 127 has panel logit relative L2 `2.8282–3.5830%`, worst
token `4.4534%`, and unchanged argmax on 10/12 inputs. Softmax KL ranges
`0.00022429–0.00921138` over the 248077 logical vocabulary. Those counterfactual
logit statistics neither implement nor qualify p-less/epsilon sampling, PPL, or
speculative acceptance. Report: `out/qwen4-endpoint-corpus-report.json`.
