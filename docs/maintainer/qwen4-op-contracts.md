# Qwen4 Op contract design

This document fixes the semantic boundaries and qualification design for the Ops needed by the
Qwen4 family described in `plans/qwen4-architecture.md`. The unregistered verifier now has C=1,
T=1..4096 entries for GGML embedding/linear, gated residual, QSA, GDN, PLE, and sparse MoE while
retaining the specialized T=1 decode routes. Exact n-gram continuation supplies the T-wide PLE
rows. Sections 2-4 describe the live `qsa_state_append`, `qsa_index_select`,
`qsa_selected_attention`, and actual-artifact `qsa_verifier` composite. Sections 5-9 describe the
remaining live transformations, including both scalar `qwen4_sparse_moe` and grouped
`qwen4_sparse_moe_prefill`. Their concrete headers, CUDA implementations, and independent
qualification tests are present; remaining conceptual entries are implementation designs and are
not evidence that those Ops, kernels, target, or product routes exist. A concrete
header under `include/ninfer/ops/` is authoritative for its
represented inputs, formula, supported domain, outputs, effects, aliasing, and workspace, and any
disagreement here must be resolved.

The research profile is the official `Qwen/Qwen3.8-Flash-Next` checkpoint at revision
`de4b8e4d43b917e7706784d8bb445c9af86a3540`, interpreted by Transformers Qwen4Exp at commit
`c119ec3cc37ab69642f39cca2de4187714002b08`. A future runnable Qwen4 target must freeze its own
geometry and formats instead of inheriting preview constants implicitly.

### Native weight-format qualification

The existing GDN, QSA core, GR and PLE composite Ops also accept native BF16, NVFP4 and row-scaled
`FP8_E4M3FN_ROW_BF16S` projections at their exact preview geometries. Their complete-layer
arithmetic retains BF16 activations (`LinearPolicy::A16Only`), FP32 controls/recurrent state, and
the existing output criteria. QSA index projections remain BF16; native QSA K/V use the explicit
BF16 baseline, while the historical diagnostic profile explicitly retains NVFP4-G16; the
mapped PLE table remains a separate embedding format. GR down `[320,10240]` admits BF16 and FP8 but not
NVFP4 because its N=320 violates the registered block-scale layout. Explicit format arguments to
each composite's workspace query describe the actual profile; callers supply their weight formats
explicitly rather than inheriting a diagnostic GGUF profile.

The earlier row-scaled activation-FP8 candidates are not admitted. GDN Z-only A8 passed isolated native
layer-0 checks and improved complete-Op timings, but the GR-fed T=17 sequence failed the
unchanged local GDN gross-error gate: maximum absolute error `0.00309234` exceeded `0.00289572`
despite relative L2 `0.005618` passing `0.009`. Its candidate API was removed; QSA and GDN-output
A8 candidates also failed their existing gates. Positive isolated evidence does not override
the sequence counterexample. The active plan records representation loss and timing separately.

The separately audited source-calibrated `FP8_E4M3FN_TENSOR_F32M` GDN projections preserve
the publisher's exact FP32 weight/input scales and have their own qualification. QKV, Z and
output each permit A16 or explicit A8; A16 remains the default. QKV and output may not both
select A8, because their combined component output error exceeded the predeclared A8 gate.
Z can combine with either role. The complete component oracle always uses the public BF16
input and exact represented weights, never private quantized activations. A8 profile budgets
were declared before execution: output relative L2 `.04`, convolution history `.04`, and
FP32 recurrent state `.08`, with gross-relative allowances `.06`, `.06`, and `.12`.
The state allowance accounts for multiple Q/K/V perturbation paths in the bounded test;
it is not a universal nonlinear bound. Existing A16 gates remain unchanged. Persistent
storage is still BF16 convolution history and FP32 recurrence, including under A8.
Native T=65 and 64+1 component sequences qualify the admitted choices; that is not full-model
PPL or default-policy approval. Exact source evidence, weight-only loss, and per-role
arithmetic results are recorded in `../research/qwen4-native-fp8-source.md`.

Calibrated Linear A8 uses a guarded temporary per-token scale:
`max(stored_input_multiplier, RN(maxabs(public_input) / 448))`. The artifact's
FP32 calibration remains unchanged and is the floor; in-range inputs retain their
original scale and codes. Out-of-range tokens expand the temporary scale rather
than clip to the publisher's calibration range. This is a qualified private arithmetic
profile, not publisher bitwise equivalence. Exact packing tests cover source-floor
and expanded-scale cases independently of the ideal floating-point Op oracle.

The native BF16 projection baseline reuses the existing BF16 Linear decode, small-T and MMA
kernels at ten exact preview geometries. The 320- and 640-wide inputs use the existing 64-wide
MMA tile at every T because the vector decode phase does not represent those widths. This is
an activation-preserving correctness baseline, not a measured claim of optimal decode dispatch.
Its GDN convolution-history oracle rounds the independent FP64 dot directly to BF16, avoiding
an artificial FP32 double-rounding seam. Direct BF16-weight history uses a BF16-precision gate
(relative L2 `1/256`, gross error at most maximum reference magnitude divided by `128`):
legitimate FP32 reduction ordering can cross a BF16 midpoint. The existing packed-weight history,
complete-output and FP32 recurrent-state criteria are unchanged.
The signed hash-distributed BF16 fixture observes convolution-history relative L2 from
`2.6352e-5` (T=3) through `3.0609e-5` (T=65) to `6.1170e-5` (T=257), with maximum absolute
error `6.1035e-5` to `1.2207e-4`. These are BF16-midpoint crossings, not a reason to tune a
fixture-specific numerical threshold. Panel versus repeated decode and the 32+33 / 128+129
chunk continuations retain the existing output and FP32 state checks.

Native GDN QKV profiles now use the existing FP32 normalized recurrent route at
every panel width, including prefill. A longer authentic layer-1 input exposed a failure
in the private FP16 chunked recurrence: relative L2 `0.003612`, maximum absolute error
`0.017770` against the unchanged gross limit `0.010346`. On the same represented input,
the recurrent route gives `0.001684` / `0.003739` and passes. This changes private
arithmetic and its workspace, not the mathematical oracle, BF16 convolution history or
FP32 persistent state. The separate diagnostic GGML profile retains its qualified
chunked route. All eight137/257-token calibration/held-out documents passed the original
whole-output GDN/state gates. Before the further QKV accumulation repair below,
the complete layer-1 Op at T137 measured395.264us median on
RTX5090/CUDA13.1 (three warmups, eleven CUDA-event samples, state restore outside the
interval). This is historical profile cost, not a measured speedup. Propagated-chain
routing sensitivity is separate and remains explicitly reported in the active plan.

Stronger per-token checks exposed native BF16 QKV accumulation errors amplified by
the declared BF16 consumer boundaries. Native `[10240,2560]` QKV now uses compensated
existing SIMT schedules through T20 and K16 MMA partials with compensated
cross-partial sums from T21, rounding the accumulator pair directly to BF16.
This is a private accuracy profile, not a guarantee of exact projection rounding;
other Linear callers retain their existing arithmetic. The failed 257-token input
and all four original held-out layer-1 inputs pass the unchanged complete-output,
per-token and state criteria. Complete-GDN cost on the failed input rises from
559.104 to630.752us; T1 medians are94.144 versus96.256us with overlapping ranges.
Public BF16 boundaries and FP32 persistent state remain unchanged.

The native BF16/BF16 GR read now retains FP32 normalized values, projection accumulations and
SILU/up intermediates. Real source layer-0 MLP hyperconnection weights exposed a gross-error
failure hidden by the synthetic fixture: BF16 staging produced `-0.703125` versus the complete
FP64 formula's `-0.677012`. Independent temporary seam attribution reproduced the deviation
(`-0.703299` before final BF16 rounding). The fix preserves public BF16 outputs and write scales,
does not alter the oracle or numerical criteria, and leaves quantized/mixed GR profiles unchanged.
It is an accuracy route, not a claim of optimal native-BF16 GR throughput.

Opt-in native source fixtures also exercise layer-0 GDN, layer-3 QSA and both layers' attention/
MLP GR components with the actual stored BF16 weights. GDN's test-owned execution view explicitly
permutes every V-side role from source grouped heads to the admitted tiled-head convention and
folds `-exp(A_log)` to FP32; its internal norm is ordinary gamma. GR/QSA zero-centered norms are
converted to effective FP32 `1+w`. The source artifacts are unchanged. The same component oracles
evaluate the resulting represented public inputs; these are real-weight, synthetic-activation
component checks, not a full Transformer-block composition, source-framework bitwise comparison,
or end-to-end model quality result. QSA's short first-block case does not qualify multi-block
indexer cutoff margins or long-context cache quality.

The closed QSA component additionally accepts tensor-calibrated FP8 Q/K/V/O
matrices with separate FP32 source weight/input multipliers and A16 compute only.
The indexer stays BF16; norm reductions and rotary/cache boundaries are unchanged.
The A16 key projection retains FP32 accumulator output through normalization, using
a narrow `[512,2560]` internal output policy. This does not broaden public Linear's
BF16 output contract or automatically admit calibrated FP8 in other Ops.

`ninfer_qsa_test --native-fp8-real` qualifies the Senfu source layer-3 calibrated
Q/K/V/O matrices with the existing protected BF16 source indexer/norms and BF16 KV.
Its A16 source cases T=1/24/129 and 128+1 passed the unchanged component output and
state gates, with exact selector trace equality. Each candidate still uses the
same independent FP64 mathematical oracle; activation packing is not copied into
that oracle. Final A16-only verification at T=129 measured output relative L2
`0.00331408` whole / `0.00331413` chunked, maximum absolute error `0.00130981`, and
K/V state relative L2 `0.00165402`/`0.00166057`; focused QSA and calibrated-FP8
Linear regressions also passed. A temporary per-role A8 assessment at T=129/128+1 failed those same
gates: Q-only failed 68 token-output screens in each schedule; K-only failed
49,066/48,678 state-coordinate screens, and V-only 47,519/47,169. O-only also failed
output checks (for example `-0.0688477` versus `-0.0595095`); combined A8 failed
61,392/60,917 checks. The failed candidate policy API, execution CLI, and unused
FP32 A8 key epilogue were removed. There is no QSA activation-policy escape hatch;
the independently qualified generic Linear A8 infrastructure remains unchanged.
These failures do not establish universal A8 unsuitability, but none of these QSA
profiles is admitted by the current evidence, and no gate was widened.
These fixtures combine explicit source
components and seeded inputs; they are not producer-matched full-model PPL evidence.

The public Linear Op additionally qualifies optimized A4/A8 activation-compute routes for NVFP4
and row-scaled FP8 respectively; BF16 remains A16-only. Permission for those routes at a standalone Linear boundary is not automatically
permission to use them repeatedly inside a nonlinear/recurrent layer. Native resident MoE uses
device-selected complete banks and independently qualifies the complete ideal FP64 formula; its
private projection and activation storage is not reproduced in that oracle.

These are usable, independently callable GPU layer components, not a registered Qwen4 Engine
target or a promise about unannounced smaller-checkpoint dimensions. The host-staged full-preview
verifier continues to bind only its exact existing GGUF-derived artifact. Bounded real-weight
qualification and its source-quality limitations are documented in `qwen4-artifact.md`.

The native NVFP4 A16 prefill route decodes each packed tile into CTA-private BF16 operands and
uses BF16 Tensor Core MMA with FP32 accumulation. E2M1 codes times their E4M3 block scale are
exactly representable in BF16; the artifact's global divisor is applied in the FP32 epilogue.
There is no persistent decoded-weight copy or runtime repack. Shape-specific measured crossovers
retain the decode/small-T routes below T64, T128, or T256. The separately permitted A4 route uses
native FP4 MMA and remains distinct from this activation-preserving route.

On RTX 5090, CUDA 13.1, `sm_120a`, cold-cache public Linear measurements for `[10240,2560]`
(A16 policy, three warmups, eleven timed repetitions, median) changed from 221.184 to 77.184 us
at T129, 835.584 to 154.880 us at T512, and 6844.416 to 1091.584 us at T4096. These are
operator-level results with device-resident weights, not full-preview throughput or measurements
including host expert transfers. Independent FP64 Linear qualification covers all eleven native
NVFP4 geometries and the T63/64/65, T127/128/129, and T255/256/257 crossover boundaries.

The row-scaled FP8 A16 prefill route likewise reuses persistent weight tiles with BF16 MMA,
applying each stored BF16 row scale after FP32 accumulation. On the same hardware/toolchain,
cold-cache public Linear medians (three warmups, twenty timed repetitions) for `[10240,2560]`
changed from 359.392 to 79.360 us at T129, 1366.016 to 153.280 us at T512, and 10914.816 to
1095.680 us at T4096. Its separate A8 policy permits
activation quantization only at independently qualified dispatch widths; A16 remains the
complete-layer policy.

Native resident MoE qualification uses the complete ideal FP64 formula, not a reference that
copies its private projection/SwiGLU stores. For source BF16 shared gate/up pairs, the existing
BF16 GEMV/small-T/MMA families retain FP32 projection outputs through SiLU-times-up and cast
only the dedicated activation panel to BF16 before shared down. The panel does not alias either
FP32 projection buffer. Mixed/quantized shared pairs and routed expert profiles are unchanged.
Ordinary inputs and decorrelated packed-weight
witnesses retain the existing `2.5/255` relative-L2 and gross-error criteria. One retained
periodic all-NVFP4 input nearly
cancels: its two reference RMS values are `9.0021e-9` and `4.6865e-10`. An offline attribution
experiment reproducing the private BF16 seams explains relative errors of 18.9% and 29.1%, but
maximum absolute errors are only `2.7660e-9` and `2.9991e-10`. This explicitly labeled near-zero
witness checks finite outputs and the existing absolute/gross bound, without claiming a relative
accuracy guarantee for that ill-conditioned case. It does not replace the strict relative tests;
the private-seam attribution model is not retained as a correctness oracle. The full FP8 and
mixed-format witnesses keep the ordinary relative criterion.

The complete resident MoE benchmark, on the same 5090/toolchain, measures GPU-stream elapsed
time for one layer with all banks resident. At T1, means over ten iterations were 161.635 us
(NVFP4) and 159.824 us (FP8). At T512, means over three iterations for fixed-hot/rotating expert
selection were 3.794/4.790 ms for NVFP4 and 1.449/2.186 ms for FP8. Repeating the scalar public
Op for the same token panel took 82.852/86.980 ms and 81.484/88.961 ms respectively. Rotating
selection spans all 512 experts; resident weights occupy approximately 1.424 GB (NVFP4) or
2.531 GB (FP8). These are one-layer schedule measurements, not cold-cache medians or full-model
decode/prefill speeds. NVFP4 grouped execution reuses the Linear A16 SmallT arithmetic; FP8
grouped execution reuses its BF16 MMA body. Routing and all floating-point execution stay on GPU.

The source-faithful `NVFP4_EXPERT_F32M` bank adds exact per-expert FP32 weight and input
multipliers; its represented coefficient is signed E2M1 times stored E4M3FN block scale times
stored weight multiplier, without reciprocal rounding. Default MoE execution remains A16.
Explicit expert `AllowA4` requires that format for all three routed projections and selects
W4A4 only for groups with at least 32 occurrences. Gate/up and post-SwiGLU down each use their
own stored input calibration. GPU-only grouping, compact MMA tile metadata, local activation
packing and occurrence scatter use caller-owned workspace; routing stays protected and shared
projections retain their independently selected precision policy. The public header records the private packing profile. This does not claim
bit-identical NVIDIA inference arithmetic or qualify future-checkpoint PPL.

The explicit `ninfer_qwen4_native_a4_eos_diagnostic` recreates the real 33-token
BF16 source embedding → GR0 → GDN0 → inject → MLP-GR prefix on the GPU, preserves
EOS token 15, and repeats its actual MoE input 65 times. The local reference is reset
to that represented input, so this isolates MoE rather than accumulated prefix drift.
Selected experts are `411,331,109,152,421,435,148,431,351,466`; groups of 65 and 64
exercise actual A4, while the one-token tail uses the qualified A16 fallback.

This source input exposes a calibration-range limitation despite passing the existing
local criteria. Every selected gate/up expert's input multiplier is approximately
`0.00138928`, giving E2M1/E4M3 finite coverage `6*448*input_multiplier = 3.734375`.
The represented input maximum is `4.375`: one of 2560 coordinates exceeds coverage
by 17.155%. Down calibration is separate: its limit is `8.5625`, whereas the existing
independent FP64 gate/up/SwiGLU formula gives maxima `0.306854..2.82101`, with no
out-of-range coordinate. That down statistic is an ideal-intermediate diagnostic,
not a claim of identical private BF16/A4 staging.

On the 5090, A16 and active A4 both passed their unchanged complete FP64 oracle
criteria and exact routing, whole 65 and 64+1. Supplementary relative L2 against
the BF16-represented reference was `0.00351252`/`0.00351251` for A16 and
`0.0928886`/`0.0921723` for A4 (whole/chunked). The latter passes the existing
16% implementation envelope but is not a small PPL-loss claim. Finite-range
exceedance is established; the approximately 9.3% total error includes ordinary
A4 rounding. No kernel, codec, calibration value or gate was changed for this
diagnostic, and no codec bug was demonstrated.

The diagnostic's `--attribution-only` mode repeats that exact GPU prefix and A16
local check, then evaluates the selected gate/up FP64 dots using three independently
constructed inputs: static-scale E4M3/E2M1 quantize/decode, original input clamped
only to the source finite limit, and a hypothetical guarded quantize/decode with
FP32 multiplier `max(source_multiplier, maxabs(input)/(6*448))`. Nearest-code
enumeration uses RNE ties, explicit FP32 codec divisions/products, and exact decoded
coefficient multiplication in the existing FP64 dot. This is private profile
attribution, not a candidate GPU kernel or new admission oracle.

On the same EOS witness, activation relative L2 is `0.0942321` static,
`0.0234538` clamp-only, and `0.0951139` guarded. Selected gate/up dot relative L2
ranges are respectively `0.0613824..0.103095`, `0.0174037..0.0354197`, and
`0.0554796..0.110424`. The guard improves 17 of 20 projection relative-L2 values,
but worsens expert 435 up and both expert 466 projections; worst-case dot error
also increases. It removes finite-range clipping but changes rounding across
all blocks, slightly worsening total activation error. These errors are not
additive: clamp-only is an isolated counterfactual, not a percentage decomposition
of total A4 error. This single-input mixed result does not justify overriding source
calibration or admitting a guarded A4 production route. The bounded attribution
run passed its unchanged prefix/A16 checks on the 5090; no PPL or full guarded-MoE
quality claim follows.

Native source-calibrated FP8 shared gate/up/down now preserve each projection's exact FP32
weight and input scalars. Separate explicit shared A16/A8 policies include their packing in
caller-owned workspace. All eight shared-policy subsets passed the complete ideal FP64 MoE
oracle with actual layer-0/3 matrices, unchanged exact route IDs and probability criteria,
T=1/8/9/24/25/65 and 64+1 continuation. Shared A8 uses a predeclared `.04` relative L2 / `.06`
gross-relative profile with the existing absolute floor, not a new mathematical oracle.
All combinations with active routed A4 also passed its existing `.16` profile at T65 and
64+1. A16 remains the default; this does not qualify a future model's PPL. Router/shared
scalar source BF16 storage and FP32 arithmetic are unchanged. Source weight-only error and
activation/kernel error are reported separately in `../research/qwen4-native-fp8-source.md`.

The opt-in A4 complete-MoE criterion, declared before measurement, is relative L2 `0.16`,
absolute floor `1/32768`, and gross allowance `0.16 * max(abs(reference))` against the same
independent ideal FP64 oracle. It is an implementation conformance envelope, not a quality-loss
budget. Default A16 criteria are unchanged. Synthetic witnesses cover small/group-boundary/wide
panels through T=4096, including fixed-hot T=31/32/33 and authentic alternating panels at
T=63/64/65 with measured expert counts 31/32/33; authentic layer-0/3 banks preserve all 512 experts' codes, scales and
independent multipliers. The bounded native harness also checks exact routing and probability
criteria; it uses synthetic inputs, not full-model captured activations. Current measurements
and integration limitations are recorded in the active architecture plan.

## 1. Boundary rules and notation

The contracts follow `docs/maintainer/op-development.md`:

- an Op owns a closed tensor transformation and any local state transition caused by one call;
- the Qwen4 Program owns layer order, state-instance selection, request frontiers, MTP index reuse,
  prefix retention, and provisional-state commit or rollback;
- core owns physical pages and raw transfers; artifact owns stored formats and materialization;
- Ops receive non-owning typed views and never artifact names, target identities, or schedule
  phases; and
- every production route is compared directly with one independent oracle. A second CUDA route or
  a Transformers forward pass is only supplementary evidence.

Logical formulas below use row-major mathematical notation. Proposed NInfer tensor views retain
the repository convention that the feature dimension is fastest and are written `[D,W,C]`, where
`D` is feature width, `W` is the maximum token width of one call, and `C` is the compact request
batch. `valid_tokens[c]` gives each request's positive valid prefix. A dense `C=1` call may omit the
final extent. Contracts must not assign meaning to invalid suffix storage unless they explicitly
promise zero or preservation.

The byte-preserving GGUF conversion exposes descriptor shapes in mathematical order but does not
transpose contiguous payloads. GGUF stores K first, so a mathematical FP32/BF16 matrix `[N,K]`
must be viewed as contiguous NInfer Tensor shape `[K,N]`; kernels index `row*K+column`. In
particular, mathematical depthwise convolution weights `[10240,4]` have physical Tensor shape
`[4,10240]`, while mathematical GR inject weights `[4,10240]` have physical shape `[10240,4]`.

For request `c`, token column `t` names request-local logical token id `base[c] + t`. Physical page
slots are not logical ids. The caller supplies checked page views that map a logical id to storage.
Positions are explicit three-axis signed integer tuples so Text positions and multimodal MRoPE use
the same contracts. A one-axis Text position repeats the scalar on all three axes.

The preview constants used for real-shape qualification are:

| Symbol | Meaning | Preview value |
|---|---|---:|
| `H` | Text hidden width | 2560 |
| `B` | GR branches | 4 |
| `R` | GR bottleneck rank | 320 |
| `Hi` | index query heads | 4 |
| `Di` | index head width | 128 |
| `Dr` | index/core rotated width | 64 |
| `r` | QSA micro-block width | 4 |
| `K` | QSA token budget | 2048 |
| `Kb` | QSA complete-block budget, `K/r` | 512 |
| `Hq/Hkv/Dh` | core query heads / KV heads / head width | 24 / 2 / 256 |
| `E/Km/I` | routed experts / selected experts / expert width | 512 / 10 / 640 |
| `Nh/Dn` | n-gram heads / values per head | 16 / 160 |

Unless an exact target authority says otherwise, zero-centered RMSNorm is

```text
zc_rmsnorm(x, w, eps)[j]
    = x[j] * (mean_k(x[k]^2) + eps)^(-1/2) * (1 + w[j]).
```

The mean and normalization oracle use FP64. The production output is rounded only at an explicit
represented output or persistent-state boundary. Private accumulator and staging precision are not
semantic.

The selected UD-IQ1_S GGUF is a converted representation, not the source checkpoint
representation: llama.cpp conversion has already replaced every applicable zero-centered norm
weight by its effective `gamma = 1 + w`. Actual-artifact Ops below therefore multiply by their
represented FP32 gamma directly. Applying `1 + gamma` would apply the conversion twice. The same
boundary folds GDN `A_log` to represented `ssm_a = -exp(A_log)` and tiles V-side GDN heads;
section 8 fixes both consequences explicitly.

## 2. QSA index projection and state append

The represented-state transition is implemented by `qsa_state_append` in
`include/ninfer/ops/qsa.h`. It appends T-wide BF16 normalized/rotated core K, projected V, raw index
keys, and three-axis positions to the fixed C=1 research state. Native BF16 K/V are copied
bit-exactly into BF16 `[256,capacity,2]` planes. The explicitly selected diagnostic NVFP4-G16
profile retains its code/scale planes; no scales exist for BF16. `qsa_verifier` is the admitted
actual-artifact T=1..4096 projection-plus-append form: it
uses separate BF16 index-query `[512,2560]` and index-key `[128,2560]` weights and FP32 norm weights.
The multi-request conceptual entry below remains future design.

The projection and append are one closed Op because the raw index key is persistent state, while
selection is a separate Op because it is exact and can be independently qualified from represented
queries and keys. This split makes the state dtype and pooling cast explicit instead of making an
unobservable fused projection determine exact selected ids.

Conceptual entry:

```text
qsa_index_project_append(
    x, qk_weight, append_ids, position_ids, valid_tokens,
    old_index_state, new_index_state, raw_query, workspace, stream)
```

### 2.1 Formula and shapes

For each valid current token, decode the stored projection weight and compute

```text
u = W_qk x
raw_query[h,:] = u[h*Di : (h+1)*Di],                h in [0,Hi)
raw_key[:]      = u[Hi*Di : (Hi+1)*Di].
```

The preview weight is logically `[640,2560]`, with four 128-wide query heads followed by one
128-wide shared key. `x` is `[H,W,C]`; `raw_query` is `[Di,Hi,W,C]`; `append_ids` is I32 `[W,C]`;
and `position_ids` is I32 `[3,W,C]`. An invalid column has append id `-1`. A valid append id
identifies the request-local logical state row to receive the raw key and supplied position. The
valid ids for each request must name the contiguous reserved interval beginning at that state's old
frontier.

The preview represented query and raw-key state are BF16. Their casts are semantic because both are
public outputs of this entry and raw keys survive the call. A future compressed index-key profile
must be a separately registered state representation and be qualified directly against this same
logical projection. Packed weights are exact-decoded by the oracle before an FP64 dot product.

The BF16 state view requires four-byte-aligned K/V planes and absent scale storage. Its four
present planes are pairwise disjoint, and append copies the original BF16 bits without arithmetic.
The diagnostic NVFP4 state view requires four-byte-aligned K/V code planes for packed 32-bit stores,
byte-aligned scale planes, two-byte-aligned BF16 raw keys, and four-byte-aligned I32 positions. The
six state planes are pairwise disjoint. Append K/V inputs are 16-byte aligned for their vectorized
K16 loads; BF16 raw-key inputs are two-byte aligned and I32 position/id inputs are four-byte
aligned.

### 2.2 Effects, aliasing, and workspace

The Op writes every valid `raw_query` element and exactly the raw-key and position rows named by
`append_ids` in `new_index_state`. A distinct output state must already expose the old prefix by
the target's page mapping or copy-on-write transaction; the Op writes only the append interval and
leaves every other row unchanged. The in-place form may alias old and new state exactly. It does
not advance a frontier.
Invalid token suffixes of `raw_query` are exact zero so a captured `W`-wide consumer cannot read
stale queries. No input, weight plane, query output, or written state row may otherwise overlap.

The projection accepts caller-owned workspace and has one interval capacity query over positive
`W`; an A16 route may return zero. The query includes only projection scratch, not persistent index
pages or selector scratch. The call is graph-capturable for the declared `C,W` envelope.

## 3. QSA index selection

Implemented by `qsa_index_select` in `include/ninfer/ops/qsa.h` for C=1, BF16 raw queries/keys,
the preview geometry, and state capacity at most 4096 tokens.

Conceptual entry:

```text
qsa_index_select(
    raw_query, index_state, query_ids, visible_ids, visible_offsets,
    rope_parameters, query_norm_weight, key_norm_weight,
    selected_ids, selected_count, workspace, stream)
```

`query_ids` is I32 `[W,C]` and maps every valid raw-query row to its request-local logical token id;
invalid columns are `-1`. Valid query rows are enumerated in `(c,t)` order with `c` outermost.
`visible_offsets` is I32 `[Q+1]` for `Q=sum_c(valid_tokens[c])`; each adjacent pair selects one
slice of flat I32 `visible_ids`. The slice contains request-local logical token ids in strictly
increasing key-axis order, contains no duplicates, contains its query id, and contains only rows of
the same request visible to that query. The target builds this view from its causal or segmented
schedule. Padding columns and physical page order never appear in it.

### 3.1 Request-local visible-rank blocks

For a query with ordered list

```text
L = [ell_0, ell_1, ..., ell_(n-1)],
```

define `nb=floor(n/r)` complete blocks by rank in `L`, not by absolute token id:

```text
block b = [ell_(r*b), ..., ell_(r*b+r-1)],  0 <= b < nb
tail    = [ell_(r*nb), ..., ell_(n-1)].
```

Thus a block may contain non-consecutive logical ids after segmentation, masking, or packing. Its
logical block id is `b`, and its block-start position is the stored three-axis position of
`ell_(r*b)`. The query position is the stored position of `query_ids[t,c]`. Neither
`floor(ell/r)` nor `b*r` is a position.

For every complete block, load its four represented raw keys, promote each component to FP32,
compute the four-way mean in FP32, and cast the pooled vector once to the represented raw-key state
dtype. That cast is semantic. Apply RMSNorm multiplied directly by effective FP32 gamma to the
represented pooled key, then apply partial interleaved MRoPE to its first `Dr` components at the
block-start position. Apply the same norm and partial MRoPE to each represented raw query head at
the current query position.

For block `b`:

```text
score[b] = sum_(h=0..Hi-1) ReLU(dot(q_rope[h], k_rope[b])) / sqrt(Di).
```

The score oracle evaluates norm, RoPE, dots, ReLU, sum, and scale in FP64 after reproducing only the
declared FP32-pool-to-state-dtype cast. The score need not be a public output. The scale remains in
the formula even though a positive common factor does not change ideal non-tied ordering.

Sort complete blocks by `(score descending, logical block id ascending)`. Take the first
`min(Kb,nb)`, expand every selected block's four token ids in its original rank order, truncate the
expanded prefix to `K`, then append the entire incomplete tail in its original order. For the
preview `K%r=0`, truncation does nothing and the valid result length is at most `K+r-1=2051`.

`selected_ids` is I32 `[K+r-1,W,C]`, in ranked-block expansion order followed by tail order.
`selected_count` is I32 `[W,C]`. The suffix at and after `selected_count` is exact `-1`. Equal
scores, including all-zero queries or keys, select lower logical block ids first. This ordering is
an NInfer semantic promise; unstable `topk` ordering from an upstream framework is diagnostic only.

The fixed verifier implementation selects its sorting network only from the host-known flat
visible extent and selected-output capacity. Extents through 2051 can contain at most 512 complete
blocks, so they use the next power-of-two network over the bounded complete-block count. Larger
extents use the fixed 1024-block network. Device CSR contents do not control dispatch: malformed
columns still take the host-selected route and produce the same empty result.

### 3.2 Effects, validation, and workspace

Selection reads but does not mutate index state. It rejects an out-of-frontier id, a duplicate or
non-increasing visible id, a cross-request id, an invisible query position, non-positive `r/K`, or
`K%r!=0` for the initial Qwen4 contract. It writes all output and count storage, including invalid
suffix `-1` and invalid token-column count zero/suffix `-1`. Inputs and outputs are pairwise
non-overlapping.

The workspace query covers the declared `C`, `W`, maximum visible count, `Hi`, `Di`, `r`, and `K`.
It includes pooled keys, score/rank scratch, and any MRoPE staging selected by the route, but not
persistent raw-key pages. No scratch pointer survives the call. Captured replay may change actual
valid counts and visible lengths inside the declared maxima without changing semantics.

## 4. QSA gated sparse GQA

The inner cache-consuming attention is implemented by `qsa_selected_attention` in
`include/ninfer/ops/qsa.h`: the live C=1/T=1..4096 entry accepts already normalized/rotated BF16
`q [256,24,T]`, maps 24 query heads to two BF16 or diagnostic NVFP4-G16 KV heads in groups of 12, reads every
per-column selected K/V row from state, and emits BF16 `[256,24,T]`. Its I32 selected-id operand is
`[S,T]` for caller-known `S` in `[1,2051]`; selected counts are device I32 `[T]`. The fixed
`qsa_verifier` composes that entry with the T-wide state append and selector, actual Q5_K
core/output projections, converted-gamma norms, partial MRoPE, and the output gate. The diagnostic
CSR entry is distinct from the native compact paged overloads in section 4.3. The verifier validates the complete state view
and the pairwise separation of every caller-owned input, weight, state, output, and workspace range
synchronously before its first projection launch; a rejected call queues no work and mutates no
storage.

Conceptual entry:

```text
qsa_gated_attention(
    x, q_gate_weight, key_weight, value_weight, output_weight,
    selected_ids, selected_count, append_ids, valid_tokens,
    position_ids, rope_parameters, q_norm_weight, k_norm_weight,
    old_core_state, new_core_state, out, workspace, stream)
```

### 4.1 Projection and cache transition

The preview query parent is logically `[12288,2560]` in per-head
`[query[256],gate[256]]` order, repeated for 24 heads; it is not interpreted as one all-query half
followed by one all-gate half. Key and value weights are each `[512,2560]`, output weight is
`[2560,6144]`, and Q/K norm weights are separate FP32 `[256]` vectors. The live verifier stores all
four matrices as GGML Q5_K block rows and accepts BF16 `[2560]` input. For every valid token:

```text
(q_raw[h], gate[h]) = W_q_gate[h] x,       h in [0,Hq)
k_raw[j]             = W_k[j] x,           j in [0,Hkv)
v_raw[j]             = W_v[j] x.
q = partial_mrope(rmsnorm(q_raw) * q_gamma, position)
k = partial_mrope(rmsnorm(k_raw) * k_gamma, position).
```

The first 64 of 256 dimensions receive interleaved MRoPE; consecutive pairs are assigned to
temporal/height/width axes in the exact repeating `[11,11,10]` pair split, theta is `1e7`, and the
remainder is unchanged. Q and K norm
weights are distinct. Encode K and V to the declared core-cache format and write the logical rows
named by `append_ids` into `new_core_state`. A selected id, including a newly appended id in the
same call, reads the decoded value from `new_core_state`; it does not bypass the cache codec. This
makes BF16, INT8, and NVFP4 cache profiles comparable under one explicit semantic state boundary.
The logical state planes are `[Dh,Hkv,capacity]` per request for both K and V; their physical page
layout remains a core view rather than an Op parameter.

The distinct-state form writes only the append interval; its old prefix must already be mapped into
the output view by the target transaction. Old and new state may alias exactly. The Op does not
advance or commit a frontier. The target chooses committed or provisional state views.

### 4.2 Sparse attention and gate

Query head `h` uses KV head `floor(h/(Hq/Hkv))`; the preview group size is 12. For its exact valid
selected-id list `S`, compute

```text
logit[j] = dot(q[h], decode(K_cache[S[j],kv_head])) / sqrt(Dh)
p[j]     = exp(logit[j] - max(logit)) / sum_k exp(logit[k] - max(logit))
a[h]     = sum_j p[j] * decode(V_cache[S[j],kv_head])
g[h]     = sigmoid(gate[h])
out      = W_o concat_h(g[h] * a[h]).
```

The oracle uses FP64 ideal softmax from the represented public inputs and exact-decoded weights and
cache values. An FP32 production softmax is an implementation profile, not a semantic cast unless
the eventual exact target authority deliberately says otherwise. The listed-id order defines the
oracle reduction order for repeatability but production association remains private. Lists must
contain no duplicates, no `-1` inside their valid prefix, and only ids visible to the query. The
caller promises visibility because this entry deliberately does not receive the visibility CSR;
the selector already guarantees causality, and this Op does not infer it from numeric ids. Every
valid query must have at least one selected id.

T=1 retains two qualified routes: selected bounds at most 64 use one CTA per query head, while
larger bounds use grouped-query score/value tiles and deterministic tile-order finalization. T>1
uses one CTA per query head and token and needs no workspace proportional to T; its per-query CSR
selection is the causal boundary that prevents a query from observing K/V appended for later
columns. These are private implementation profiles and do not change the listed-id formula, FP32
softmax, cache codec, or public represented outputs.

#### Test-only FP8 KV assessment

`ninfer_qsa_test --native-fp8-kv-assessment`, with `NINFER_QWEN4_NATIVE_LAYERS`,
evaluates a hypothesis, not a paper-prescribed recipe or admitted runtime format:
independent K/V scales per token and KV head, FP32 `max(abs(row))/448` (zero rows use
scale one), E4M3FN RNE/saturate-finite codes, and explicit FP32 code-times-scale
reconstruction followed by BF16 representation. Its theoretical K/V payload is
`(256+4)/(256*2) = 50.78125%` of BF16, excluding allocation overhead and unrelated
index/state storage. Exact independent code vectors cover ties, signed zero,
subnormal underflow and saturation; zero-row reconstruction is exact.

The source layer-3 BF16 matrices and independent FP64 projection/norm/RoPE oracle
generate represented BF16 K/V and Q at T=129 with nonzero positions. K-only, V-only,
and combined compression are separate cells. Residual inputs are seeded synthetic
BF16 values, not captured text activations. Frozen causal selected IDs exclude
selector changes; Q, index keys, softmax mathematics and GDN are not quantized.
The existing BF16-cache attention kernel consumes independently decoded candidate
values and is checked directly against the same FP64 attention oracle, with its
unchanged criterion, both whole and 128+1. Separately, FP64 attention and gated
output-projection differences against the original BF16 cache report storage loss.
Neither oracle reads a GPU-mutated cache. This does not qualify a GPU FP8 codec,
FP8 runtime storage, long-context selection, model quality or PPL; storage-loss
measurements are an assessment rather than a deployment acceptance gate.

On the RTX 5090/CUDA 13.1 build, the assessment and focused QSA tests passed with
unchanged same-input kernel gates. FP64 storage-loss results for this bounded input:

| Candidate | Attention relative L2 / max absolute | Output projection relative L2 / max absolute |
|---|---|---|
| K only | 0.0169408 / 0.0232602 | 0.0195155 / 0.00727305 |
| V only | 0.0260962 / 0.0175780 | 0.0265606 / 0.00995216 |
| K and V | 0.0311334 / 0.0227258 | 0.0329898 / 0.0100114 |

Whole and 128+1 same-input attention passed independently; decoded cache bytes and
frozen selection remained exact. The combined approximately 3.3% projected relative
L2 is measurable storage error, not evidence of acceptable PPL. Keep BF16 as the
native reference; this result alone does not admit FP8 KV as its replacement.

`out` is BF16 `[H,W,C]` for the preview. Invalid token columns are exact zero. In the live verifier
the per-head represented BF16 attention is multiplied by sigmoid of its corresponding
represented BF16 raw gate, then the concatenated `[6144]` vector crosses the Q5_K output matrix.
The cache write and
the complete output are the only effects. Output may not alias `x`, selected ids, weights, or cache;
the only permitted state alias is exact old/new cache alias. Workspace covers projections,
softmax/reduction scratch, page addressing, and codec staging for the complete declared envelope.

### 4.3 Native compact paged execution

The `QsaPagedStateView` overloads share one P64 logical-to-physical block table across core K/V,
optional NVFP4-G16 scales, raw BF16 index keys, and three-axis I32 positions. Every plane is
page-major and caller-owned; paging changes neither represented values nor the selector formula.
Native capacity is bounded by the exact source context, 262144 tokens. Device controls select
distinct table rows for compact B=1..4, per-row valid lengths and append frontiers, and per-column
MRoPE coordinates. B=1 admits W=1..4096; B=2..4 admits W=1..16. Query j sees exactly logical
rows `[0,frontier+j]`, independently of the numerical MRoPE coordinates. The host maximum-visible
envelope bounds execution but never changes visibility. Mappings remain stable until the stream
drains; the runtime owns allocation, reservation, prefix retention, and frontier commit.

Selection forms complete four-token visible-rank blocks, retains the highest 512 with lower-rank
tie breaking, and appends the incomplete causal tail. A fixed 512-block streaming merge uses
8192 bytes per query, independent of context capacity; no capacity-sized shared sort is used.
Invalid columns have count zero and selected IDs -1, produce zero output, and cannot append state.
`qsa_verifier_selected` instead consumes an explicitly frozen, unique visible subset, preserving
the source MTP selection domain. It still performs the same projection and represented cache append.
Device controls and page tables are read on-stream and support fixed-shape CUDA Graph replay.

The native selector and shared core Q/K MRoPE implementation evaluate frequency, position product,
and trigonometric functions in FP64 before their FP32 rotation arithmetic. This is a qualified
private precision profile, not a new semantic cast: large source coordinates must not amplify
FP32 frequency/phase error into incorrect selection or persistent keys. The independent oracle
evaluates the full FP64 norm and rotation from represented inputs, followed only by the explicit
BF16/cache-codec state boundary. `ninfer_qsa_paged_test` checks exact selector IDs through context
262144, fragmented mappings and ties, independent FP64 attention, and complete sparse-matrix
composites with exact persistent K/V checks at low and near-ceiling source positions. Normal and
frozen-selection graph routes retain the unchanged 2% composite output criterion; this is kernel
and state evidence, not a model-quality or throughput claim.

## 5. Gated Residual read and inject

Implemented family: `include/ninfer/ops/gated_residual.h`. The family has three entries rather than a
boolean mode hidden in one call:

```text
gated_residual_read(R, norm_weight, down_weight, up_weight,
                    x, workspace, stream)
gated_residual_read_write(R, norm_weight, down_weight, up_weight, write_weight,
                          x, write_scale, workspace, stream)
gated_residual_inject(R, block_output, write_scale, R_out, stream)
```

The final model mixer uses `gated_residual_read`; attention and MoE sublayers use the read/write
form followed by inject. The target, not the Op, holds `R` and `write_scale` across the intervening
sublayer call.

### 5.1 Read formula

The implemented verifier admits C=1 and T=1..4096. For `R` `[H,B,T]`, normalize each branch and
token separately with its own slice of effective FP32 gamma (native source `1+w`, or
the diagnostic artifact's already converted scale):

```text
Rhat_i = rmsnorm(R_i, eps) * gamma_i
u      = SiLU(W_down vec(Rhat) / B)
G      = reshape(sigmoid(W_up u), [B,H])
x      = (1/B) * sum_i G_i * Rhat_i.
```

The converted preview weights are GGML Q8_0 `W_down [320,10240]`, GGML Q8_0
`W_up [10240,320]`, and FP32 gamma `[10240]` grouped as four independent 2560-wide rows. The
source unit offset is already folded into that gamma.
Division by four occurs after the down-projection and before SiLU; there is no second division
before `W_up` and no static or branch-mixing term.

The read/write form additionally computes

```text
write_scale = 2 * sigmoid(W_write vec(Rhat) / B),
```

with FP32 `W_write [4,10240]` and output `[B]`. The final read-only form has no write weight or
scale output. `x` and `write_scale` are explicit represented BF16 outputs for the preview, so their
final casts are semantic; internal norm/projection materializations are not.

The qualified write implementation assigns one 256-thread CTA to each of the four rows and uses
the common FP32 block reduction. On an RTX 5090 with CUDA 13.1, three warm public-Op measurements
changed `gated_residual_read_write` from 31.75/31.69/31.50 us to 29.70/29.65/29.64 us (6.4% at the
median of run medians). The unchanged read entry measured 25.86/25.85/25.85 us before and
25.95/26.03/26.04 us after; its 0.7% shift bounds measurement drift rather than a read-path change.
The supplementary public read/write-minus-read increment fell from about 5.84 us to 3.62 us.

#### Native FP8 residual-storage experiment

The opt-in `ninfer_gated_residual_test --fp8-residual-experiment` uses exact native layer-0 and
layer-3 attention/MLP GR weights (`NINFER_QWEN4_NATIVE_LAYERS`), 33 synthetic BF16 residual
tokens, one FP32 scale per token/2560-wide branch (`maxabs/448`), E4M3FN round-to-nearest-even
storage, and FP32 decode followed by the public BF16 boundary. Gates and all weights retain the
protected baseline. The same independent FP64 formula compares original BF16 residuals with the
decoded candidate, isolating storage error from GPU implementation error.

On 2026-09-18 the four read relative-L2 errors were `0.0297841`, `0.0306891`, `0.0288672`, and
`0.0317548` (layer-0 attention/MLP, then layer-3 attention/MLP), exceeding the unchanged `0.006`
read gate. Write-scale errors were `0.00407994`, `0.00516035`, `0.00595261`, and `0.00745576`,
exceeding the unchanged `0.0035` gate. Although payload plus scales would occupy `50.0781%` of
BF16 storage, this specific profile is rejected. It remains test-only; no persistent-state codec,
runtime route or default changes. This bounded synthetic-activation result does not reject all
calibrated FP8 residual schemes or establish model-level quality.

The subsequent `ninfer_qwen4_native_sequence_real_test --fp8-residual-experiment`
repeats this codec at residual stores throughout contiguous native compute layers 0..3,
including PLE at layer 1, using BF16 QSA cache and T17 whole/16+1 execution. Each GPU
consumer still receives explicit decoded BF16 values and is checked against its unchanged
same-input mathematical oracle; a separately propagated FP64 formula receives its own
codec round trips. This is decoded-consumer verification, not a GPU FP8 encoder qualification.
Against the BF16-storage baseline, final GPU output relative L2 is `0.0833825` (maximum
absolute error `0.21875`); independent formula storage loss is `0.0836543`. Router/selector
traces differ in 380 ordered ID slots, which is not a count of distinct expert-set changes.
Local component gates pass, but the unchanged accumulated and whole/chunk screens fail.
The assessment returns failure and the recipe remains rejected. Inputs are synthetic branch
residuals and explicit source PLE fixture ordinals, not token-derived activations or a
calibration corpus; this does not refute a different calibrated FP8 storage recipe.

### 5.2 Inject formula and effects

For block result `y` `[H,T]`:

```text
R_out[:,i,t] = R[:,i,t] + write_scale[i,t] * y[:,t].
```

Every element is written. `x` and `write_scale` may not overlap `R` or each other. Inject permits
`R_out` to alias `R` exactly, but forbids partial overlap and any overlap with `y` or
`write_scale`.

Read entries accept caller-owned fixed-profile workspace; inject requires no global scratch. FP8
residual storage, if selected later, is a distinct represented-state profile with explicit
decode/encode boundaries and its own criterion. It is not enabled by the BF16 contract.

### 5.3 Source-defined four-stream stem

`gated_residual_stem` is a closed BF16 Op for the preview's two-projection MTP stem,
not an MTP rollout, cache transaction, or registered target. It independently offset-RMS
normalizes the token embedding `[2560,T]` and the entire concatenated hidden state
`[10240,T]`, with epsilon `1e-6`. Only after the latter normalization does it view the
hidden state as four 2560-wide branches. The two independent BF16 `[2560,2560]`
matrices project embedding and hidden branches; the embedding projection is added to
each branch and the public result is BF16 `[2560,4,T]`. Existing `mtp_fc`/`mtp_pack`
contracts do not describe this formula and are not used.

The private implementation composes central RMSNorm and A16 BF16 Linear with a
broadcast-add launcher. Its normalized/projection BF16 workspaces are implementation
choices, not oracle casts. The complete independent FP64 oracle uses the original
represented inputs and weights, a single 10240-wide hidden mean square, and no private
staging casts. Admission criteria declared before measurement are relative L2 `0.01`
and maximum absolute error `0.005 + 0.02*maxabs(reference)`, with finite outputs.
All storage is disjoint; workspace is caller-owned and scoped, addresses remain stable
during CUDA Graph capture/replay, and there is no persistent state or hidden allocation.

On the RTX 5090/CUDA 13.1 builder, `ninfer_gated_residual_stem_test` and
`ninfer_qwen4_native_mtp_stem_real_test` pass. Synthetic exact-shape widths 1/7/27/28/33
cover embedding and four-branch GEMV/small-T/MMA transitions with deliberately unequal
branch variances. The separately acquired four-tensor NVIDIA BF16 source fixture
`qwen4-mtp-stem.ninfer` passes widths 1/17/28: relative L2 `0.002874..0.002884`, maximum
absolute error at most `0.018190`. Whole and `(T-1)+1` schedules each pass the same oracle;
two fixed-route graph replays are exact. Guarded output/workspace, rejected alias/short
workspace, preserved matrices and workspace-scope restoration also pass. This is
bounded stem correctness, not full MTP recurrence, selection-reuse, acceptance/fold or
model-quality qualification; those gates remain in the model authority.

## 6. Exact n-gram row addressing

Implemented family: `include/ninfer/ops/ngram_embedding.h`, with exact CUDA and C=1 host integer
routes qualified by `tests/ops/test_ngram_embedding.cpp`. The host entry prepares the same validated
constants once, advances an arbitrary token span from the oldest-to-newest two-token history, and
returns sixteen rows per token plus the advanced raw history. It performs no floating-point work or
embedding decode. The CUDA route remains independently exercised against the naive exact oracle;
the tests also compare one-shot, partitioned, and tokenwise host execution directly with CUDA
across continuation, EOS, and reset.

The UD-IQ1_S verifier owns its two-token host history, uploads the exact row ids and advanced
history for device diagnostics/state, and stages all T row panels on the transfer stream while
layer 0 executes. A transfer-ready event orders device decode before decoder layer 1 without a
row-id D2H transfer.

Conceptual entry:

```text
ngram_row_ids(
    input_ids, valid_tokens, old_history, config,
    row_ids, new_history, stream)
```

This is an exact integer state-transition Op. For the preview, `input_ids` is I32 `[W,C]`,
`old_history` and `new_history` are I32 `[2,C]` in oldest-to-newest order, and `row_ids` is I32
`[16,W,C]`. The wrapper proves the configured table and offsets fit nonnegative I32; a target that
does not fit must revise the represented id type rather than truncate it.

The exact preview contains one PLE module. Its only admitted zero-based `ple_layer_index` is zero;
the wrapper rejects every other value rather than exposing unqualified synthetic prime tables.

### 6.1 Prime table and multipliers

Let `V` be unigram vocabulary size, `N` the maximum n-gram size, `P` heads per n-gram order,
`layer` the zero-based PLE-module index, and `seed` the configured seed. Define unsigned 64-bit
wrap by `u64(x)=x mod 2^64` and:

```text
splitmix64(x):
    z = u64(x + 0x9E3779B97F4A7C15)
    z = u64((z xor (z >> 30)) * 0xBF58476D1CE4E5B9)
    z = u64((z xor (z >> 27)) * 0x94D049BB133111EB)
    return z xor (z >> 31)

multiplier_max = floor((2^63 - 1) / max(V,1))
half_bound     = max(1, floor(multiplier_max / 2))
base_seed      = seed + 10007 * layer
m[j]           = 2 * (splitmix64(u64(base_seed
                         + 0x9E3779B97F4A7C15 * (j+1))) mod half_bound) + 1,
                 j in [0,N).
```

Every multiplier is positive and odd. For global head `h=layer*(N-1)*P+local_head`, its modulus is
the `(h+1)`-th prime strictly after `vocab_base-1`. Within one PLE layer, offsets begin at zero and
are the exclusive prefix sum of that layer's head moduli. Padding the concatenated embedding table
does not create addressable rows. Output head order is increasing n-gram order, then local head:
all `P` bigram rows, then all `P` trigram rows, and so on.

### 6.2 History, EOS reset, and row formula

Process each request's valid columns in increasing order. For current token `a_0`, let `a_j` be the
token `j` positions earlier when at least `j` tokens have occurred since the most recent EOS
strictly before the current token; otherwise let `a_j=EOS`. The current EOS may therefore form an
n-gram with its preceding segment, while the following token sees reset history.

For n-gram order `n=2..N`:

```text
mixed_n = u64(a_0*m[0]) xor u64(a_1*m[1]) xor ... xor u64(a_(n-1)*m[n-1]).
```

Reinterpret `mixed_n` as signed two's-complement I64 and use Euclidean remainder with the positive
head prime, matching `torch.remainder`; the result is in `[0,prime)`. Add that head's offset. The
preview's multiplier bound and token domain keep each product below `2^63`, but the signed
reinterpretation/remainder rule remains part of the exact generic contract.

`new_history` is the last two raw tokens of `old_history || valid input`, left-padded with EOS when
the request has fewer than two tokens total. EOS values are retained; reset occurs while resolving
a lag, not by rewriting stored history. Invalid input columns produce row id `-1` and do not alter
history.

Old and new history may be disjoint or alias exactly. They must not overlap ids or input. The Op
advances no target frontier and commits nothing; speculative execution writes a provisional
history selected by the Program. No workspace is required.

## 7. PLE gather/decode and injection

Implemented family: `include/ninfer/ops/ple.h`. The admitted host-resident C=1 profile has scalar
`ple_iq4_nl_stage_rows` and T-wide `ple_iq4_nl_stage_rows_batch`; the latter copies exactly
`16*T` 90-byte IQ4_NL rows into fixed caller-owned pinned storage and enqueues one contiguous H2D.
`ple_iq4_nl_decode_rows` exact-decodes U8 `[90,16,T]` to BF16 `[160,16,T]` on the GPU.
`ple_inject` implements the complete T=1..4096 injection/state transition with BF16/native packed
or diagnostic Q8_0 projections. Required `PleNormFormat` distinguishes source zero-centered BF16
norm parameters from effective FP32 gamma; native gamma is evaluated as FP32 `1+w` without
rewriting the BF16 source. Convolution coefficients independently admit BF16 or FP32, with FP32
reductions/nonlinearities retained. Workspace queries require explicit projection formats.
Every column in W is valid; multi-request invalid
suffixes remain outside this verifier entry.

### 7.1 Gather/decode boundary

The logical gather is:

```text
E[(h*Dn+d),t,c] = decode(table[row_ids[h,t,c],d]),
```

with `E [Nh*Dn,W,C]`, head-major flattening, and preview shape `[2560,W,C]`. Row ids must be valid
non-padding rows and are never deduplicated semantically. Exact table codec and scale lookup belong
to the represented table view. BF16 compares after exact BF16 decode.

A device-resident table could expose `ngram_embedding_gather(table,row_ids,E,...)` directly. The
admitted host-resident C=1 path instead takes all `16*T` exact row ids at the round boundary and
copies their encoded 90-byte spans, in token/head order and without semantic deduplication, from the
artifact-owned mapping into one bounded pinned slot. One H2D transfer populates the paired stable
device slot; `ple_iq4_nl_decode_rows` then performs only exact GPU decode. File I/O, page-fault
policy, staging-ring ownership, and slot completion remain outside the Op. Neither entry mutates
the table or row ids. Output and all inputs are non-overlapping.

The native per-tensor E4M3FN preparation route is `ple_fp8_stage_rows_batch` followed by
`ple_fp8_decode_rows`. It copies 160 encoded bytes per selected row into U8 `[160,16,T]`,
then computes `BF16_RNE(E4M3FN(code) * BF16(scale_bits))` on the GPU. The exact stored positive
finite BF16 scalar is a multiplier; it is not the row-scaled Linear FP8 format. Finite FP8
values (including signed zero) convert exactly to BF16 before the source multiplication.
The public raw-code/scale contract is admitted without claiming a native checkpoint binder.
The artifact owner must eagerly populate and lock the complete encoded host table before use;
the Op neither duplicates nor expands it. Caller-owned pinned/device slots contain `2560*T`
bytes each and remain unreusable until the transfer and decode consumers finish.

Source provenance: NVIDIA checkpoint `fc694b54fb0174e0913e6adf86691ef85a4ead47` names producer
ModelOpt `0.46.0.dev281+g73d778422`. That exact exporter's FP8 decoder casts codes and scales
to the requested output dtype before multiplying. SGLang's pinned Qwen4 loader confirms
contiguous row shards: shard `i` begins at `i*ceil(padded_vocab/split_ngram_parts)`; this preview
has 128 shards of 2,500,012 rows, not one shard per hash head. Its BF16 gather output is multiplied
by the one BF16 table scale. Source addresses:

- https://github.com/NVIDIA/Model-Optimizer/blob/73d778422388f0e849ecb180375d34ac445711ca/modelopt/torch/quantization/qtensor/fp8_tensor.py
- https://github.com/sgl-project/sglang/blob/c46bf5e990bdd99e2c200214b04683022100e4df/python/sglang/srt/models/qwen4_exp.py

Exact tests use an independent binary FP8 formula over every finite code, the source scalar
`0x3951` and other normal/subnormal BF16 scales, repeated rows, non-default-stream staging,
token/head order, T=1/17/128/4096, chunk boundaries, and output guards. This is codec/selected-row
qualification, not proof that the complete 47.684 GiB FP8 table fits the current host budget.

### 7.2 PLE injection boundary

Native NVFP4 PLE uses `ple_nvfp4_stage_rows_batch` and `ple_nvfp4_decode_rows`.
The resident `NVFP4_PARTITION_F32M` payload preserves partition-major 90-byte rows
(80 adjacent-pair E2M1 code bytes and ten E4M3 block scales), then one FP32 multiplier
per partition. The pinned producer's multipliers differ across shards. Staging adds the
exact four multiplier bytes to each selected record: U8 `[94,16,T]`, with bounded
`1504*T` pinned and device bytes. GPU reconstruction is FP32 followed by BF16, preserving
signed zero. The first E2M1/E4M3 product is exactly representable. The full packed payload,
not just selected rows, must be populated and OS-locked before load succeeds. CPU gather
performs byte copies only. No IQ codec or whole-table BF16 expansion participates.
Producer/source cast details and bounded fixture provenance are recorded in
`../research/qwen4-native-ple-source.md`. Selected-row tests do not prove complete-table
memory capacity or quantization quality. Gather has no GEMM activation precision policy.

Conceptual entry:

```text
ple_inject(
    R, E, valid_tokens, key_weight, value_weight,
    key_norm_weight, query_norm_weight, conv_norm_weight, conv_weight,
    old_conv_state, new_conv_state, R_out, workspace, stream)
```

For `R [H,B,W,C]`, `E [2560,W,C]`, and positive lane prefixes `valid_tokens [C]`:

```text
Khat_i = rmsnorm((W_key E)_i, eps) * key_gamma_i
Qhat_i = rmsnorm(R_i, eps) * query_gamma_i
V      = W_value E
g_i    = dot(Khat_i,Qhat_i) / sqrt(H)
g_i    = sign(g_i) * sqrt(max(abs(g_i),1e-6))
G_i    = sigmoid(g_i) * V
N_i    = rmsnorm(G_i, eps) * conv_gamma_i.
```

The preview weights are `W_key [10240,2560]`, `W_value [2560,2560]`, three independent source
zero-centered BF16 norm vectors (or diagnostic effective FP32 gamma vectors) `[10240]`, and
mathematical depthwise `conv_weight [10240,4]`, viewed
physically as contiguous `[4,10240]` with `weight[c*4+j]`. The represented old
and new convolution-state views are `[10240,9,C]`. Flatten branch-major `G/N` to channel `c`. With
dilation three, kernel width four, and old history `N_old[-9..-1,c]`, define `N_all` as old history
followed by current valid `N` and:

```text
conv[t,c] = sum_(j=0..3) conv_weight[c,j] * N_all[t - 9 + 3*j,c]
P[t,c]    = G[t,c] + SiLU(conv[t,c])
R_out[t,c]= R[t,c] + P[t,c].
```

The indexing is cross-correlation, matching `Conv1d`: weight 0 multiplies lag 9 and weight 3
multiplies the current token. `new_conv_state` is the last nine represented `N` rows of old history
followed by valid current rows. The cast into the declared convolution-state dtype is semantic,
and convolution reads those represented values, including current rows. With fewer than nine total
rows, the missing oldest rows are exact zero.

`R_out` may alias `R` exactly. Old/new convolution state may be disjoint or alias exactly after the
old history has been consumed. No other overlap is valid. Invalid token suffixes do not change
state and are exact zero in `R_out`. The Op does not update n-gram token history; that is the
independent exact Op in section 6.

The workspace query includes projections and any convolution staging for the complete `C,W`
envelope, not persistent history. One-shot prefill, arbitrary legal chunk partitions, and repeated
`W=1` calls must produce the same represented outputs and final state under the profile criterion.

The opt-in `ninfer_qwen4_native_ple_real_test` reads exact native BF16 layer-1 projection,
norm and convolution parameters plus selected authentic FP8/NVFP4 table rows. The pinned
Transformers reference at `c119ec3cc37ab69642f39cca2de4187714002b08` uses zero-centered grouped
RMSNorm and source `[10240,1,4]` convolution coefficients: dropping the singleton axis preserves
channel-major order, not a transpose. Full-model BF16 framework materializations remain a
cross-check profile; the complete mathematical oracle does not copy private GEMM staging casts.
Only the represented gathered BF16 embedding and persistent BF16 history are cast boundaries.
Fixture row IDs are ordinals into extracted rows, not n-gram hashes of a model text sequence;
these tests do not establish full-table admission, full-model PPL, or producer-matched mixed
checkpoint quality.
On the RTX 5090 / CUDA 13.1 builder, the public gather/decode/inject chain passes actual-source
FP8 and NVFP4 table profiles at T=1,9,10,17,27,28,29, including the BF16 projection small-T/MMA
boundary, history rollover, nonzero initial history and reset, and exact permitted state/residual
aliasing. The complete FP64 output criterion is relative L2 `0.02`, absolute floor `1e-4`,
gross allowance `0.02 * max(abs(reference))`; BF16 history uses relative L2 `1/256`, floor
`1e-5`, gross allowance `max(abs(reference))/128`. Both reject non-finite results. Gathered
embedding bits and retained old-history regions compare exactly. Chunk-local oracles consume
the actual represented incoming history; separate accumulated checks independently propagate
the complete initial-state formula. These gates were fixed before measurements and were not
weakened. Synthetic format cases use the same generalized formula; the superseded oracle's
private projection casts and constant-row simplifications have been removed.

### 7.3 Explicit PLE accepted-prefix state

`ple_inject` may emit every represented normalized gated-value column as disjoint BF16
`[10240,W]` records. `ple_commit_prefix` appends a caller-selected prefix of those records to
the nine-column convolution history and the matching verified input IDs to the two-token raw
n-gram history. The histories retain their last nine/two values respectively. EOS is copied
unchanged; the hash Op, not commit, owns its following-token reset rule. Device counts and distinct
slots support startup-fixed C=1..4 and CUDA Graph replay. Zero count is a strict no-op; rejected
suffixes and other slots are untouched. The exact sequence-concatenation oracle checks full/partial/
zero prefixes, history rollover and slot permutations. Choosing the accepted count and publishing
licensed output tokens are target/runtime decisions, not part of this exact state transformation.

## 8. Qwen4 GDN profile

The native formats and per-role policies qualified above apply to the same complete formula;
the historical GGML profile below is diagnostic provenance, not the native storage policy.

Implemented first profile: `include/ninfer/ops/gated_delta_net_layer.h`. It is one semantically
closed C=1/T=1..4096 layer entry over the converted UD-IQ1_S checkpoint storage: qkv and z are GGML
Q5_K (Q6_K for layer 2), output is GGML Q6_K, and a/b/conv/ssm_a/dt_bias/norm are FP32.
It owns projection, width-four causal convolution and final convolution history, controls, the
recurrence call, sigmoid-gated learned RMSNorm, and output projection. Distinct state inputs remain
read-only and therefore expose the rollback boundary without assigning commit policy to the Op.

No new Gated DeltaNet recurrence formula is needed. The GGUF converter changes the source's
grouped V-head order to tiled order for llama.cpp broadcast: represented V/control head `h`
consumes represented Q/K head `h%16`. The complete layer entry explicitly expands the sixteen
Q/K heads to 48 in that tiled order, then reuses the existing `gated_delta_net` recurrence with
`Hq=Hv=48`, head width 128, normalized Q/K, scale `1/sqrt(128)`, and FP32 `[128,128,48]` state.
Its live artifact verifier profile has:

1. hidden width 2560 and separate GGML block-row projections qkv `[10240,2560]` and z
   `[6144,2560]`, plus FP32 controls a/b `[48,2560]`; qkv history is represented BF16
   `[10240,3]`, while Z bypasses convolution. At aligned prefill widths divisible by 256, the
   Q5_K qkv projection privately decodes one row into FP32 shared storage and replays it through
   the established sequential 16-token FP32 accumulators. At exactly T=512, the Q6_K output
   projection uses the same decoded-row ownership with four sequential windows per CTA. Both
   profiles change no represented boundary;
2. mathematical FP32 convolution weights `[10240,4]`, physically viewed as contiguous
   `[4,10240]` and indexed `weight[channel*4+tap]`, plus represented
   `ssm_a=-exp(A_log)` and `dt_bias`, consumed without a BF16 verifier lane;
3. ordinary learned scale (no unit offset) and `norm(y)*sigmoid(z)` independently for each
   128-wide value head, followed by output projection `[2560,6144]`; and
4. distinct or exact in-place BF16 convolution state and FP32 recurrence state.

The live C=1 entry accepts arbitrary T partitions, updates the three-column BF16 convolution history
sequentially, and composes the recurrence in 64-token tiles while preserving one FP32 final state.
Program owns which in-place state becomes committed.

For verification widths 2..16 the complete layer can also expose raw replay records: BF16
projected QKV, expanded post-convolution/pre-normalization K/V, and FP32 interleaved decay/update
controls. These are explicit public outputs, independently checked against the same complete FP64
formula. The ordinary provisional output and final state math do not change. Rollback callers
retain disjoint initial state and decide which recorded prefix to apply.

The central record primitive admits `(Hq,Hv)=(48,48)` with identity mapping over the already
expanded keys. Fold admits the actual 36-layer Qwen4 bank `(36,48,48,10240)` and an explicit
single-layer overload over real convolution/recurrence tensor pools. Both reuse the same recurrence
kernel, normalize raw recorded keys, retain FP32 recurrence, and copy accepted convolution history
exactly. Counts zero, partial/full prefixes, explicit paths, C=4 slot isolation and graph replay
pass independent FP64 state/exact-transform checks. The fold's host row controls are captured by
value: changing acceptance requires a new enqueue, not mutating captured host storage. This is Op
qualification; target transaction/publication ownership remains separate.

For exact post-expansion `Hq=Hv=48` prefill, measured private Q and K routes begin at 448 and 512
full-chunk tokens respectively. The Q route normalizes raw BF16 Q directly into the output CTA's
FP16 shared Q tile. The K route gives each prepare CTA unique ownership of one 64-row K tile,
normalizes the complete 128-wide rows into two existing shared panels, publishes the identical FP16
K workspace for W construction, state passing, and output, and consumes those shared bits for KKT.
Both routes use the same FP32 warp reduction and FP16 cast as materialized normalization. Shorter
or grouped-head profiles retain the corresponding materialized Q/K path. This changes no public
represented boundary or recurrence formula. The workspace interval query accounts for the
non-monotonic Q allocation crossover rather than assuming the maximum T always has the largest
allocation; K fusion retains its materialized workspace.

The oracle exact-decodes Q5_K/Q6_K bytes and covers projection, width-four causal convolution, Q/K
L2 normalization, control gates, every FP32 recurrence update, sigmoid output gating, and Q6_K
output projection. Complete-layer Q5_K T=3 and layer-2 Q6_K T=1 cells, including nonzero initial
state, compare directly to one sequential FP64 oracle from the represented inputs; a
distinct-output call proves the rollback boundary. Complete-layer T=64 and T=65 partition cells
compare chunked execution with the production one-shot route, including exact BF16 convolution
state and criterion-bound output and FP32 recurrence state. Recurrence-only T=447/448/449 and
T=511/512/513 cells at post-expansion `Hq=Hv=48,D=128` cover both private-route boundaries and
their first recurrent tails; a grouped `Hq=16,Hv=48,T=512` cell protects the K-fusion ownership
exclusion. The maximum-width real-geometry cell uses matching T=4096 represented
inputs for scalar T=1, one-shot, aligned 64x64, and `64+1+1986+1+2044` schedules. Every route's
complete BF16 output, selected cumulative prefixes, and final FP32 state compare directly to one
complete FP64 recurrence oracle; pairwise bit comparisons remain diagnostic implementation-profile
evidence.

The projection oracle also exercises the exact Q5_K `[10240,2560]` and Q6_K `[2560,6144]`
dispatch boundaries at T=511/512/513. It independently decodes the stored block scales, minimums,
low nibbles, and high bits into FP64 products from represented BF16 activations, compares every
BF16 output, and checks that input and weight storage remain unchanged.

## 9. Live C=1 sparse-MoE verifier

The current `sparse_moe` contract is fixed to H=2048, 256 experts, top 8, width 512, and an
AddResidual epilogue. `qwen4_sparse_moe` is a separate fixed verifier entry because the preview
uses H=2560, 512 experts, top 10, width 640, actual GGML artifact codecs, and a Store result because
GR owns the residual write:

```text
H=2560, E=512, Km=10, I=640, shared_I=640
logical router / shared gate [512,2560] / [1,2560]
routed gate/up banks      [512,640,2560] each
routed down bank          [512,2560,640]
shared gate/up            [640,2560] each
shared down               [2560,640].
```

For each token, exact-decode weights and compute all 512 router logits. The oracle forms an ideal
FP64 softmax, ranks `(probability descending, expert id ascending)`, selects ten distinct experts,
and renormalizes the ten selected probabilities to sum to one. Since softmax is monotone, a route
may select from logits, but its public result must match this ordering. For expert `e`:

```text
routed_e = W_down[e] (SiLU(W_gate[e] x) * (W_up[e] x))
routed   = sum_(rank=0..9) normalized_weight[rank] * routed_selected[rank]
shared   = sigmoid(w_shared_gate x)
           * W_shared_down(SiLU(W_shared_gate_proj x) * (W_shared_up x))
moe      = routed + shared.
```

Native resident MoE accepts the source BF16 router and shared scalar-gate pair directly;
neither weight needs an FP32 device copy. The existing FP32 pair remains the diagnostic
storage profile. BF16 loads widen exactly. The native router retains a two-FP32 summation
expansion through dot reduction and ranking; the shared scalar gate and diagnostic FP32
profile retain their FP32 dots. This avoids losing close expert order merely by rounding a
private logit to FP32; it is not a claim of exact real arithmetic for every possible input.
Neither the control storage profile nor the private reduction correction changes the closed
Op's ideal logits, exact top-k ordering or FP32 selected-coefficient output representation.
The pinned Transformers unfused router instead materializes BF16
logits before FP32 softmax and casts selected coefficients back to BF16. That framework
arithmetic profile can create additional ties; bit-identical framework routing is not the
closed ideal Op's contract. Source-weight fidelity must not be described as framework bit
parity. Native real-weight and synthetic BF16-control tests use the same complete independent
MoE oracle and unchanged route/output criteria as the FP32-storage profile.

The disjoint-corpus calibration-science input reproduced a native layer-3 token-2
ordering failure: ideal expert logits 412 and 88 differ by `4.90125e-7`, but the former
FP32 reduction collapsed them to a false tie. The retained correction canonicalizes
the two-word sum and compares `(high, low)` without FP64 dot products or comparisons.
The captured 137-row panel, decode witness, distinct logits that round to the same
FP32 value, and opposite-sign cancellation true ties pass the independent FP64
ordering/probability oracle in eager and graph execution. The complete MoE synthetic
oracle also retains a false-tie witness across its native decode/prefill widths.
On RTX 5090 / CUDA 13.1, isolated projection-plus-ranking CUDA-event medians over
three 100-call runs changed from `12.34` to `16.42` us at one token and `69.94` to
`202.79` us at 137 tokens. These are the measured cost of the correctness repair,
not whole-MoE or model latency claims; no routing or output gate was relaxed.

The `Store` epilogue writes `destination=moe`; the existing `AddResidual` epilogue remains for its
existing registered geometry. Logical expert id always selects the same router row and two stored
bank matrix spans. There is no capacity factor, token dropping, duplicate expert, decoded-value
gather, or runtime repack.

The live UD-IQ1_S verifier accepts a contiguous FP32 K-fastest router, mapped-host IQ1_S or
IQ2_XXS routed gate/up banks, device IQ4_NL routed down, a contiguous FP32 shared scalar gate,
Q5_K shared gate/up (or the layer-2 Q6_K pair), and Q8_0 shared down. The GPU computes routing and
exposes the ten ids and renormalized FP32 weights. The host copies those ids into the first 40 bytes
of caller-owned pinned staging, waits for that route only, and copies exact encoded gate/up matrix
pairs in route-rank order into alternating slots. A distinct transfer stream performs one H2D per
rank while the compute stream consumes the preceding rank; every decode, linear, SwiGLU, shared
gate, mixture accumulation, and output operation remains on the GPU.

An IQ1_S gate/up pair is 640,000 bytes. An IQ2_XXS pair is 844,800 bytes, the fixed per-slot maximum.
The live pipeline owns two slots, so both pinned and device staging are exactly 1,689,600 bytes; the
same 6,400,000 or 8,448,000 encoded bytes still cross H2D per layer. Transfer-ready and
consumer-complete events protect host and device reuse. The next call's route-ready -> IDs-ready
barrier also proves the preceding call's two slots complete before rank zero can overwrite them.
The pipeline binds its event set to one compute stream and rejects stream substitution, which is
required for this transitive cross-call proof.
The mapped banks are never pinned or copied in full. The entry owns no scheduling, model registry,
or Engine path.

`qwen4_sparse_moe_prefill` owns the same complete formula for C=1, T=1..4096. It receives all
`10*T` route ids in one D2H, validates and groups them using a fixed 335,876-byte caller-owned
pinned integer scratch span, and stages each unique expert once in ascending-id order. A call with
more than 32 unique experts starts with 16 so compute can begin before a full-slot transfer; later
groups contain at most 32. Two 27,197,440-byte pinned/device slots alternate. Each group stores its
exact gate/up pairs followed immediately by its rank-token occurrence list, so one contiguous H2D
publishes both.
GPU gather, a per-expert fused GGML gate/up projection and SwiGLU, routed down, scatter, shared
branch, and rank-order FP32 accumulation preserve the scalar represented boundaries. The fused
route accumulates gate and up separately in FP32, rounds each completed projection to BF16,
widens both represented values for the existing FP32 SiLU/product, and rounds the activation to
BF16. Per-expert occurrence count M=1 uses the scalar paired projection; M>1 uses
16-occurrence tiles. The intermediate
gate/up values are logical rounding boundaries rather than global workspace tensors. Duplicate
expert use across tokens does not duplicate staged matrix bytes, and the Op performs no allocation
or floating-point CPU work.
The host scratch is four-byte aligned and mutually disjoint from both mapped banks and the pinned
stage; the same fixed-compute-stream rule protects cross-call slot reuse.

The same semantic Op also has a device-resident GGML T=1..4096 verifier profile. Its routed gate and up
operands are complete rank-three IQ1_S or IQ2_XXS banks `[512,640,2560]`, and routed down is the
complete IQ4_NL bank `[512,2560,640]`. T below 256 repeats the exact scalar fused route because the
measured fixed grouping and launch overhead outweighs its reuse benefit at short widths. At T>=256,
a 16-token FP32 router tile reads
each router row once per tile, then device histogram/prefix/scatter stages only integer occurrence
indices by expert. Grouped GGML kernels reuse each exact packed row across as many as 16 occurrences
and BF16-store gate/up, SwiGLU, and down projection seams. Each down result lands in its unique
rank-token slot; the final kernel visits ranks zero through nine in order and accumulates in FP32.
The shared Q5_K/Q6_K and Q8_0 branch uses the existing T-wide aggregate kernels. The profile
performs no transfer, host synchronization, event operation, allocation, or runtime repack. The
mapped-host staged route and current Program placement remain separate and unchanged. The one-layer
placement benchmark owns full device banks and reports both a fixed hot route and rotating windows;
at T>=52 the rotating profile covers all 512 experts. It is not a claim that the 48-layer preview
artifact fits one RTX 5090 or authority to change the live verifier's placement.

The complete closed Op, not private router or expert stages, owns the oracle. Qualification includes
all-zero router logits selecting ids `0..9` with weights exactly `0.1` in the ideal oracle, a
nonuniform softmax with a tie at the tenth boundary, distinct raw logits whose nonmaximum FP32
exponentials all underflow, positive and negative shared-gate saturation, both routed codecs, both
shared gate/up codecs, exact host/device staged bytes, and actual T=1 shapes. Its naive FP64 oracle
decodes IQ1_S/IQ2_XXS, IQ4_NL, Q5_K/Q6_K, and Q8_0 independently and evaluates the complete ideal formula.
The complete verifier's private FP32 reduction profile is qualified with fixed normwise and finite
gross criteria rather than copied into its ideal oracle. The separately public fused gate/up
SwiGLU Op's independent FP64 oracle does apply its declared BF16 projection and activation seams.
The T-wide profiles additionally cover duplicate-token routes, both routed/shared codec pairs,
an association-independent M=17 packed gate/up witness that exact-checks both projection rounds
and the activation round across the 16-occurrence tile boundary, and a T=28 nonzero
70-unique-expert panel that forces 16/32/22 mapped-stage grouping and slot reuse.
Each expert receives four distinct represented inputs and route weights; expert ids are permuted so
all three groups own a highest- or second-highest-weight expert. The panel independently derives
routing, exact staged bytes and occurrence order, surviving slot contents, and each token's output
from the complete FP64 formula. Distinct IQ1_S and maximum-footprint IQ2_XXS panels are submitted
back-to-back without caller synchronization on the same streams, events, and slots; both complete
outputs are checked and the second panel's surviving slot bytes prove cross-call lifetime closure.
The resident grouped path is checked independently
at T=257 with four alternating nonzero represented inputs, two distinct route sets, same-expert
reuse across different inputs and permuted ranks, back-to-back workspace reuse, exact route
outputs, and per-token complete FP64 output criteria. A T=4096 all-zero-input run checks maximum
capacity, guards, and lower-id ties without weakening the nonzero oracle witness. Batched C lanes
remain future
registered-target work.

## Native BF16 Vision patch and encoder reference

Native BF16 Linear now admits patch `[1152,1536]`, QKV `[3456,1152]`, attention
output `[1152,1152]`, MLP up `[4304,1152]` and down `[1152,4304]`. These use the
unchanged canonical BF16 MMA schedule at every positive token width. The real 4304
dimension requires compile-time masked row/K tiles: invalid source transfers are
zero-filled and invalid output rows are not stored. Divisible instantiations retain
their prior load/store branches; no runtime weight repacking or padded artifact exists.
The kernel-iteration card refused tile-shape tuning at `[1152,4304,T128]`; no such
optimization was performed. This is correctness-required shape admission, not an
optimal-throughput or inference-speed claim.

`tools.parity.qwen4.native_endpoint_fixture --component vision-block` audits the pinned
NVIDIA headers and acquires 15 exact BF16 source tensors: patch weight/bias, position
table and one complete encoder block. Payload is 39,328,672 bytes, preserving source
shapes and bytes in `qwen4-vision-block.ninfer`, identity
`qwen4/native-vision-block-qualification/nvidia-bf16-source`. The flattened patch keeps
source `[3,2,16,16]` order. The fixture driver uses explicit 2x2 merge-major spatial
groups, 48x48 align-corners position interpolation, both affine LayerNorms, all four
projection biases, 72-wide Vision RoPE, segmented noncausal attention, tanh-GELU and
both residual additions. Existing public Ops execute every stage; this is not a second
Vision execution backend or a full image/video frontend.

The complete FP64 formula reuses the existing independent LayerNorm, RoPE and attention
oracles through test-only adapters, and the existing BF16 dot oracle. It propagates
ideal intermediates without copying private BF16 normalization, projection or activation
staging; only the documented position-interpolation/addition casts are explicit.
Every source Linear additionally passes its unchanged represented-input BF16 criterion.
The accumulated gate was declared before measurement: relative L2 `0.02`, maximum
absolute error `0.005 + 0.02*maxabs(reference)`, finite outputs.

RTX 5090/CUDA 13.1 results: complete native patch/position/block at P4/P12/P132 passes
with relative L2 `0.00497801/0.00571211/0.00573572` and maximum absolute error
`0.0369369/0.0427145/0.0616180`. Perturbing only the second packed segment leaves the
first segment bit-exact. Synthetic tests of all five geometries at T1/4/128/132 place
nonzero contributions solely in the final 16 K elements, exercise final output rows,
and match the independent oracle exactly with guarded outputs. Existing BF16 Linear,
LayerNorm, RoPE and Vision attention regression suites pass. This qualifies one native
encoder block and its source shapes, not all 27 blocks, full frontend integration,
image understanding or multimodal model quality.

## Native BF16 Vision patch merger

`vision_patch_merger` owns the complete preview merger formula, not a target-private
sequence of partially verified stages. Its BF16 input is `[1152,4,G]`, with each
merge-major group ordered top-left, top-right, bottom-left, bottom-right. It applies
learned per-patch LayerNorm (population variance, epsilon `1e-6`), concatenates four
normalized patches as width 4608, evaluates biased `[4608,4608]` projection, exact
erf GELU (not the Vision-block tanh approximation), then biased `[2560,4608]`
projection. The BF16 `[2560,G]` result is the only output. Source matrices, norm
parameters and biases remain BF16. The implementation reuses existing GPU LayerNorm,
native BF16 GEMV/small-T/MMA, bias and exact-GELU kernels; private intermediate BF16
casts are not inserted into its mathematical oracle.

The caller supplies non-overlapping storage and the queried workspace; the Op owns
no allocation, persistent state, frontend regrouping or Vision tower. Validation
precedes GPU writes. Two exact BF16 Linear geometries are admitted for the merger,
`[4608,4608]` and `[2560,4608]`, with their existing independent Linear oracle as
well as the single complete naive FP64 merger oracle. This is not admission of
additional Vision geometries by itself or a multimodal Engine target; the separate
patch/encoder reference qualification above owns those five additional geometries.

`tools/parity/qwen4/native_vision_merger_fixture.py` acquires exactly the six BF16
source merger tensors (66,079,232 bytes) from the pinned NVIDIA source. The local
`qwen4-vision-merger.ninfer` identity is
`qwen4/native-vision-merger-qualification/nvidia-bf16-source`; its JSON records
source revision, shards, shapes and exact payload sizes. No complete shard or
Vision checkpoint is downloaded.

`ninfer_vision_patch_merger_test` checks asymmetric synthetic patch groups and
nonzero affine/bias values, including constant patches. Its `--native-real` mode
uses the actual six tensors with seeded represented BF16 patch activations.
Both profiles passed G=1/5/28/129 and 128+1, guard checks and rejection without
input/output mutation on the RTX 5090. Every execution compares directly with
the same full FP64 formula under the predeclared, unchanged criterion
`{0.02,1e-4,0.02}`. Source G=129 relative L2 was `0.00333155` whole /
`0.00333153` chunked, maximum absolute error `0.0248062`; synthetic G=129 relative
L2 was `0.00459203`. This is merger mathematics and chunk-execution evidence,
not full-Vision, image-understanding, throughput or model-quality qualification.

## 10. Schedule-owned transactions are not Ops

The unregistered `src/targets/qwen4/program.cpp` verifier now composes these live C=1 Ops into the
48-layer eager Text schedule with one exact 4096-token frontier. `prefill_chunk` accepts 1..4096
numeric token ids, constructs a causal CSR slice for every query, processes the layer schedule over
the complete T panel, and projects only the final column through the untied Q4_K head. It overlaps
the mapped PLE panel with layer 0 and uses grouped two-slot expert staging in each layer. The
existing T=1 entry remains the teacher-forced NLL/decode path. Successful prefill commits the QSA,
GDN, PLE, frontier, and final continuation state only after both streams synchronize; post-enqueue
failure poisons the Program until reset, while pre-enqueue validation leaves it usable. It exposes
diagnostic views but is not a registered target or an Engine execution path.

The following remain in `src/targets/qwen4` even when they invoke the Ops above:

- choosing main versus MTP QSA index/core state and the logical append frontier;
- building request-local visible lists from causal, padding, segment, and speculative topology;
- capturing the target-aligned MTP selection and deciding which later draft steps reuse it;
- selecting committed or provisional n-gram, PLE convolution, GDN, QSA, and MTP state views;
- allocating state records and applying reject, partial-accept, or full-accept commit/rollback;
- deciding PLE row deduplication, host prefetch timing, staging-ring ownership, and cancellation;
- advancing the request cursor, publishing generated tokens, and retaining/restoring prefixes; and
- choosing eager or graph execution and graph envelope.

A record-copy or fold can become an Op only when its complete input record and deterministic output
state are explicit, as with existing GDN replay/fold primitives. “Commit Qwen4 draft” is not an Op:
it chooses accepted length, owners, lifetimes, and several state instances. No contract in this
document authorizes an MTP state-fold or acceptance Op before the Phase-0 MTP formula is frozen.

## 11. Independent oracle construction

Each eventual `tests/ops/` suite owns a naive host oracle that does not include production launcher,
kernel, codec-helper, target, or Transformers code.

### 11.1 Exact and floating paths

- N-gram tests implement prime search, SplitMix64, two's-complement wrap, Euclidean remainder,
  offsets, EOS reset, and state continuation directly in test code. Every row id and history value
  compares exactly.
- QSA selector tests begin with represented query/key-state values, reproduce the one declared
  FP32 mean and cast, then use FP64 norm/RoPE/scores. Selected ids, counts, `-1` suffix, and state
  writes compare exactly. Score comparison is needed only if a future header exposes scores.
- The QSA verifier's current-token NVFP4-G16 witness bounds each decoded K value analytically per
  16-value group: normal-range E4M3 scale rounding gives `scale <= (17/16)*(M/6)`, and nearest
  E2M1 rounding contributes at most one decoded scale because its largest adjacent-code gap is two.
  The comparison also adds the BF16 round-to-nearest `abs(x)/256` bound and `1e-4` for the FP32
  norm/RoPE evaluation before staging. V projection witnesses with exactly represented BF16 input
  need only the `(17/96)*M` codec bound.
- Floating Ops exact-decode every stored weight in test-owned code and evaluate the complete formula
  in FP64. BF16/FP8/NVFP4 output encoding is not copied into the oracle; represented production
  outputs are promoted and compared with the retained ideal result under a named profile criterion.
- Stateful oracles retain both every logical output and final state. Partition tests invoke the
  oracle once over the complete sequence, not by composing the production chunk route.
- Ideal attention and router softmax use a max-subtracted FP64 exponential. Upstream FP32 softmax is
  a cross-check profile only.

### 11.2 Required conformance cells

The opt-in `ninfer_qwen4_native_sequence_real_test`, with `NINFER_QWEN4_NATIVE_LAYERS`
pointing to the extracted source fixtures, composes all native compute layers 0 through 3:
each layer's attention GR, GDN or QSA, injection, MLP GR, resident MoE and injection, with
source PLE before layer 1's attention read. This is the contiguous first block, not an
end-to-end model. Default admission uses BF16 QSA cache; `--nvfp4-diagnostics` explicitly
selects the rejected wider compressed-cache assessment and retains its failing exit status.
Synthetic represented BF16 residuals at T=5 and T=17 run both as whole
panels and as T-1 prefill plus T=1 continuation at each public component. GDN starts from zero state;
QSA runs the BF16-cache baseline and a separately selected diagnostic NVFP4-G16-cache witness.
The BF16-cache sequence uses native FP8 PLE rows; the diagnostic-NVFP4-cache sequence uses
native NVFP4 PLE rows. Each is checked independently against its represented-input oracle,
not compared as a matched weight/cache quantization quality A/B. PLE row IDs are selected-fixture
ordinals, not hashes for an input text. Its native BF16 parameters and zero-centered norms remain
source-faithful; complete table residency is a separate qualification gate.

Test-only adapters reuse the existing component fixtures and their single FP64 mathematical
oracles. Local comparisons use represented actual public inputs and the original component
criteria. In particular, the local QSA output formula consumes the validated represented actual
KV state, whereas the accumulated comparison constructs its own ideal BF16 KV values or
diagnostic NVFP4 codes and scales according to the explicit test profile.
The accumulated reference independently propagates BF16 public read/write/output
boundaries, router decisions, recurrent state, and independently represented KV;
it never substitutes production output or production cache bytes for the propagated reference.
The accumulated gate, declared before measuring the chain, is relative L2 `0.02` plus a gross
pointwise cap `0.005 + 0.02 * max(abs(reference))`, with non-finite rejection. Public state and
route/codec checks retain their component gates; whole/chunk parity is supplementary evidence.
The earlier, noncontiguous layer-0 → PLE1 → layer-3 witness (omitting the remainder of layer 1
and all of layer 2) on 2026-09-18 RTX 5090 / CUDA 13.1 at T=17 passed every local/state criterion and the final
accumulated gate: final relative L2 `0.00686405`/`0.00668704` (whole/partitioned), maximum absolute
error `0.01953125`/`0.02734375` versus gross cap `0.0353125`. Whole-versus-partitioned output
relative L2 was `0.00294694`, maximum absolute error `0.03125`.
These are synthetic-activation operator/composition results, not PPL or
evidence that the complete native model fits on the GPU.

The additional `--native-text` fixture uses 33 real tokenizer IDs, original BF16 embeddings
repeated into four branches, and their exact hash-selected FP8 PLE rows. Independent integer
hash checks and every local/accumulated criterion passed for whole/32+1 on RTX 5090, CUDA
13.1. Same-input router indices remain exact; independently propagated inputs can cross a
top-k boundary, so such trace differences are reported separately and are not described as
bit-identical full-chain routing. The 2% accumulated criterion is unchanged. This panel is
not a calibration corpus or a substitute for full-model PPL.

The earlier `--native-text-fp8` panel passed whole/32+1 with the thirteen
source-calibrated FP8 projection matrices (GDN layer 0, QSA layer 3, shared experts in
layers 0/3), all with A16 operands and BF16 cache. QSA same-input output relative L2 is
`0.00414611`/`0.00414967`; BF16 K/V state relative L2 is approximately `0.00166`.
The accumulated comparison propagates the represented FP8 weights independently; this
establishes implementation correctness for that input panel, not equality to BF16 weights
or a measured full-model weight-quantization loss.

The earlier selective `--native-text-a8` panel used guarded A8 for GDN layer-0 Z and all
three shared-expert projections in layers 0/3, with QSA and other GDN projections
remaining A16. Whole33 and 32+1 passed the unchanged local/state and 2% accumulated
criteria. The original static-scale attempt failed: EOS token 15 produces a GDN
input maximum of 49.75 against source range 16.875. Independent attribution gives
static Z projection relative error 0.212924, dominated by clipping; the guarded
projection gives 0.00667229 against the ideal formula. QSA same-input output error
in the guarded chain is 0.00418435/0.00418893. Exact same-input routing checks
remain in place; independently propagated inputs still change some top-k choices.
This closes the reproduced calibration-clipping defect for the represented panel,
not future-checkpoint calibration or full-model quality qualification.

The expanded disjoint 137/257-token corpus supersedes that broad source recipe.
The current selective policy uses tensor-FP8 Z0 and row-FP8 Z2 with A8, shared layer0 FP8 gate/up
with **up only** A8, and shared layers1..3 FP8 gate with **gate only** A8. Other
shared projections and all QSA matrices stay BF16/A16. Z1 remains A16. The A16 text witness uses
the same narrowed weights without A8; historical timings above/below do not describe
the replacement recipe. Storage and activation policy admission are checked separately
by the exact native artifact binder. Source-loss evidence and rejected combinations
are recorded in `docs/research/qwen4-quantization-recommendations.md`; none is a
whole-model quality admission.

For resident native NVFP4 or expert-scaled NVFP4 routed-down weights with A16
expert execution, private per-rank down results remain FP32 through the existing
rank-ordered FP32 weighted sum. Only the public destination is stored as BF16.
The mainloop, gate/up and SwiGLU arithmetic are unchanged. This removes premature
per-expert down rounding exposed by a real mixed-FP8 shared-expert input; it does
not insert those private intermediates into the mathematical oracle. Workspace
planning and execution use the same format/policy predicate. The extra capacity is
51,200 bytes per token (12.55 MiB at T257); FP8/GGML down and the entire explicit
AllowA4 expert profile retain their existing BF16 rank storage, including A16
fallback groups within that A4 profile.

That T=17 witness exposed a premature private BF16 core-K projection cast: using the same
represented actual input, 170 decoded K values differed from the independent ideal cache, while
V matched exactly. An attribution-only counterfactual inserting that one cast reproduced the
actual cache exactly. Native BF16 core K now retains FP32 projection output through norm/RoPE;
the public BF16 append boundary, NVFP4 codec, indexer and other projection formats are unchanged.
The canonical BF16 GEMV, small-T and MMA arithmetic/schedules are reused with an FP32 output
policy, also used by the native shared-MoE gate/up accuracy route. No speed claim follows from
this precision fix. The corrected same-input witness matches the independent K/V cache exactly;
its complete output relative L2 is `0.00346`, versus `0.01715` before the fix. The complete-chain
maximum error fell from `0.04395`/`0.04492` to the passing values above without changing any gate.
The native QSA component additionally passes at T=`1,5,27,28,128,129`, covering the reused
GEMV/small-T crossover and partial/full/tail MMA output policies with exact source weights.
The BF16-cache component repeats these source-weight widths plus T=5 chunk continuation.
Its state criterion keeps the pre-existing BF16 projection/norm allowance and removes the
NVFP4 codec allowance; no output gate is relaxed. Exact BF16 append checks all stored and
untouched bits, including signed zeros/subnormals, permuted append IDs, an invalid suffix,
and whole versus 17-token chunks. The common independent FP64 attention formula directly checks
both T=1 short/tiled and T=2 routes at selected counts 0, 1, 63/64/65, 127/128/129,
2047/2048 and 2051. BF16 selector checks cover the existing exact 2047..2053 budget/tie cases,
score ordering, malformed visibility and the 4096-token capacity. This is numerical execution
evidence, not a long-context model-quality result or a BF16-cache performance claim.
Before adding PLE to the composition, the BF16 and diagnostic-NVFP4 selected-layer T=5/T=17
whole/chunk sequences both passed the
unchanged local and accumulated gates. The BF16 T=17 layer-3 token-14 routing set differs
between independently propagated reference inputs and actual inputs near a small cutoff margin
(`0.000832` whole / `0.001669` partitioned); routing on the same represented input remains exact.
This is reported numerical sensitivity, not a claim of identical end-to-end routing.

In the earlier selected-layer composition (layer 0, layer-1 PLE only, then layer 3),
the T=5/T=17 whole/chunk cells passed on the 5090
for FP8-table/BF16-KV and NVFP4-table/diagnostic-NVFP4-KV profiles, with unchanged local
and accumulated gates. Independently propagated reference inputs still expose layer-3
routing-set sensitivity: BF16-KV T=17 token 8, and diagnostic-NVFP4-KV tokens 0, 1, 3
at T=5 plus 6, 7, 11 at T=17 (token 6 only in the whole schedule). Routing on each
same represented input remains exact. These differently represented table/cache profiles
are separate conformance cells, not a paired quantization-quality comparison.

The stronger contiguous layers 0..3 diagnostic-NVFP4 composition does **not** pass
the unchanged accumulated 2% screen. At T=5, one failing residual element is
`0.222656` versus `0.255859`; at T=17 it is `0.742188` versus `0.78125`.
The T=17 whole/chunk screen also fails (`0.847656` versus `0.886719`). Local
component criteria still pass. `--nvfp4-diagnostics` retains this failed screen;
the compressed four-layer chain is not qualified by the earlier narrower pass.

QSA decomposition rules out a same-input codec bug in these witnesses: production
K/V codes and scales exactly match both independent encoding of the public BF16
QSA-state comparator and independent FP64 same-input projections followed by the
specified BF16 append/codec boundary. The comparator is supplementary evidence,
not a replacement mathematical oracle. Different upstream propagated inputs do
cross NVFP4 decision boundaries:

| Schedule | K differing codes / scales | V differing codes / scales | QSA local output relative L2 | Propagated-reference output relative L2 |
|---|---|---|---|---|
| T=5 | 38/2560 / 3/160 | 35/2560 / 5/160 | 0.00300882 | 0.0346991 |
| T=17 | 173/8704 / 13/544 | 153/8704 / 14/544 | 0.00322051 | 0.0401914 |
| T=17, 16+1 | 161/8704 / 10/544 | 143/8704 / 11/544 | 0.00321021 | 0.0395188 |

Local output compares GPU with the FP64 formula over the actual cache and same
represented input. Propagated output compares independent same-input and propagated
reference formulas through gating and the output projection. The measured failure
is accumulated-input sensitivity amplified by coarse KV codes/scales and subsequent
routing, not grounds to weaken the screen or silently change the oracle. BF16 KV
remains the native reference; local diagnostic NVFP4 codec/attention qualification
does not establish compressed-chain quality or PPL.

The same sequence test separately qualifies the narrower MLP residual composition
`GR read -> MoE AllowA4 -> GR inject` for native layers 0 and 3. A represented residual token is
repeated across T=65 so selected expert groups actually reach 65 occurrences (64 for the
prefill-plus-one comparison), exceeding the admitted 32-occurrence A4 cutoff; the single-token
tail uses the A16 fallback. Exact local routing checks and the unchanged A4 component criterion
remain active, and the accumulated criterion above is unchanged. On the same 5090, accumulated
relative L2 was `0.00327633`/`0.00325159` for layer 0 and `0.00165074`/`0.00163827` for layer 3
(whole/partitioned); maximum absolute errors were `0.005859375` and `0.00390625`, respectively.
This is active routed-W4A4 MLP-residual evidence, not a full stateful A4 model or PPL claim.

The minimum meaningful matrix is:

| Family | Required cells |
|---|---|
| QSA projection/composite | diagnostic verifier: separate BF16 512x2560/128x2560 index projections, Q5_K 12288x2560/512x2560/2560x6144 core/output projections, FP32 norms, C=1/T=1..4096. Native paged entry: compact B=1..4, exact per-column selection and persistent BF16/NVFP4 state, independent FP64 output oracle, normal/frozen graph execution, near-ceiling MRoPE coordinates; section 4.3 |
| QSA selector | visible counts 0..5 and 2047..2053; 512-block saturation; non-contiguous visible ids; unequal C lanes; fragmented pages; multimodal positions; all-zero and boundary ties; BF16 pool/cast witnesses |
| QSA attention | real 24/2/256 geometry; diagnostic causal CSR and native prefix-causal paged views; selected count 1, 2048, and 2051; nonuniform independent FP64 oracles; compact B=1..4, fragmented/remapped pages, represented BF16/NVFP4 cache reads, invalid suffixes and slot isolation |
| GR | live verifier: real 4x2560/R=320; FP32 norm/write and Q8_0 down/up; read-only/read-write/inject; C=1/T=1..4096; exact Q8_0 decode oracle; in-place inject. Future registered batched entry qualifies C=4,8 and graph envelopes |
| n-gram | exact vectors below; empty/short history; EOS as current and prior token; one-shot/chunk/T=1; C lane isolation; every admitted PLE-module index |
| PLE gather | first/last valid row, repeated and permuted ids, 16-head/token order, codec edges, and exact T=1/16/17/128/4096 decode against the independent IQ4_NL oracle |
| PLE inject | live verifier: real 4x2560 projections/state; zero and nonzero history; one-shot/legal chunks/T=1; dilation witness; in-place output; C=1. Future registered batched entry: C=4,8 isolation and invalid-suffix semantics |
| GDN | live verifier complete layer: source 16 Q/K heads expanded to 48 tiled artifact heads (`h%16`) at width 128; direct represented `ssm_a`; Q5_K and layer-2 Q6_K inputs; Q6_K output; direct FP64 oracle at Q5_K T=3 and layer-2 Q6_K T=1; pairwise one-shot/partition evidence at T=64/65; nonzero FP32 initial state; sigmoid output gate; distinct rollback and in-place continuation. Represented-input recurrence: post-expansion `Hq=Hv=48,D=128,T=4096` scalar, one-shot, aligned 64x64, and `64+1+1986+1+2044` schedules, each covering complete BF16 output, selected prefixes, and final FP32 state against one FP64 oracle. Future registered routes qualify multi-request snapshot/record/fold if used |
| sparse MoE | live verifier: 512/top10/I640; lower-id all-zero and tenth-boundary ties; selected normalization; IQ1_S/IQ2_XXS scalar and grouped T=1..4096 two-slot host staging plus complete device-resident bank profiles; IQ4_NL routed down; Q5_K/Q6_K shared gate/up; Q8_0 shared down; exact group bytes/occurrence order and FP64 Store output. Future registered entry qualifies C=4,8 |

Named numeric criteria must be fixed from adversarial and target-representative oracle-error
distributions before a failing candidate is judged. Reduction Ops require both a normwise bound and
a finite gross pointwise cap, and every criterion rejects non-finite output/state. Exact selector
ids and integer transforms have no tolerance.

### Native PLE transfer readiness measurement

`ninfer_ple_transfer_bench` measures the public native packed-row gather/H2D/decode path.
The table is synthetic, eagerly populated and OS-locked; changing random indices use a
512 MiB footprint exceeding the test CPU's 128 MiB aggregate L3. Only bounded transfer slots
are CUDA-pinned. One copy-stream event orders GPU decode on the consumer stream, and each
iteration drains decode-ready before reusing storage. This is an isolated no-overlap measurement,
not full-table capacity admission, full-model timing, or a guarantee against PLE bottlenecks.

RTX 5090, Ryzen 9 7950X3D, CUDA 13.1.2, 2026-09-18; 10 warmups and 200 measured samples per
format/width, no concurrent GPU workload. Public validation, CPU row gathering, enqueue,
H2D, GPU decode and final host synchronization are included in readiness latency:

| T | FP8 median / p95 readiness, µs | NVFP4 median / p95 readiness, µs |
|---|---:|---:|
| 1 | 9.060 / 16.810 | 8.990 / 13.881 |
| 128 | 86.451 / 104.230 | 62.520 / 77.150 |
| 4096 | 2292.607 / 2446.978 | 1713.316 / 1829.645 |

Transfer payloads are respectively `2560*T` and `1504*T` bytes (NVFP4 records carry their
partition multiplier). T4096 median submit-call times are 1894.296 / 1387.314 µs, indicating
that host gathering/enqueue is material at this width. `copy_ready_to_decode_ready` includes
consumer scheduling gaps and decode; it is not pure kernel time. No independent first-layer
compute runs here, so production overlap is neither measured nor presumed.

Reproduce using `/build/bench/ninfer_ple_transfer_bench --t T --repetitions 200 --table-mib 512`
in the existing builder image with the matching build volume and an adequate memlock allowance
(the measurement used 1 GiB in a temporary container). The ordinary builder's 8 MiB limit
correctly cannot admit this benchmark table. No global swap or unrelated workload changes
are necessary.

## 12. Known-answer fixtures

These constants are small, permanent witnesses for independent test implementations. They do not
replace generated adversarial cases or real-shape qualification.

### 12.1 Preview n-gram constants

For `V=248320`, `N=3`, `P=8`, `layer=0`, `seed=1234`, `vocab_base=20000000`, and `EOS=248044`:

```text
m = [23703573157769, 20109073645365, 8052911324071]

prime = [
  20000003, 20000023, 20000033, 20000047,
  20000059, 20000063, 20000069, 20000077,
  20000081, 20000093, 20000107, 20000147,
  20000153, 20000159, 20000161, 20000171
]

offset = [
          0,  20000003,  40000026,  60000059,
   80000106, 100000165, 120000228, 140000297,
  160000374, 180000455, 200000548, 220000655,
  240000802, 260000955, 280001114, 300001275
]

sum(prime) = 320001446
padded rows at multiple 128 = 320001536
```

With old history `[EOS,EOS]` and one-shot input `[10,20,30]`, the lag triples, sixteen row ids,
and final history are:

```text
[10,EOS,EOS] ->
  [6826666,27775725,51991156,74082527,82622748,119600976,135816374,152166807,
   174244281,190221032,211723794,232787707,243645790,275729718,280030017,303574322]

[20,10,EOS] ->
  [4810669,34962340,40038404,63145186,97237011,115267695,122313706,158375242,
   171840321,191688874,219996824,228806855,252506948,276566639,284666446,305764640]

[30,20,10] ->
  [9878115,26555603,54895210,62571545,80580723,119917398,128922427,147596134,
   168936175,195223391,219226064,233524685,246670267,279816194,297531600,306108296]

new_history = [20,30]
```

A following `T=1` call with input `[40]` must use lags `[40,30,20]`, return:

```text
[11251501,34567287,46225500,74547382,84537869,101201472,136196980,149524337,
 178516953,196243462,200258135,226017340,244881709,263746180,290034520,301475424]
```

and publish history `[30,40]`. This is the chunk-continuation witness.

For old history `[EOS,EOS]` and input `[7,EOS,9]`, the EOS-reset witness is:

```text
[7,EOS,EOS] ->
  [2927653,34980843,54748278,66612378,97814964,109013870,126560393,151352333,
   167888935,182235580,215170017,237467519,247510681,278779700,296141806,304994522]

[EOS,7,EOS] ->
  [10204458,27984170,41283776,68842151,85621153,118821647,129504214,158727320,
   162716417,183296409,205500418,223498012,243883332,265230110,285892800,310808036]

[9,EOS,EOS] ->
  [18043673,37626835,51159316,78294604,94015356,106720349,136526052,144330141,
   176817901,186368539,203707490,230017629,247662678,266533413,293096193,307951937]

new_history = [EOS,9]
```

The final row proves that the token after EOS does not see token 7. The new state retains the raw
last two tokens; it is not rewritten to `[EOS,EOS]`.

### 12.2 QSA visible-rank and saturation fixtures

For

```text
visible_ids = [2,5,9,10,21,22,30,31,45,46,80]
r = 4
```

the only legal partition is:

```text
block 0 = [2,5,9,10]
block 1 = [21,22,30,31]
tail    = [45,46,80].
```

Using absolute-id blocks such as `[8,9,10,11]` is wrong. If both block scores are equal, selection
order is block 0 then block 1 and the output ids are the eleven ids shown above.

With all-zero represented queries and raw keys, identity RoPE, zero norm weights, visible ids
`[0,1,...,n-1]`, and the preview `r=4,K=2048`, expected valid ids are:

| `n` | Complete blocks | Expected selected ids |
|---:|---:|---|
| 0 | 0 | empty |
| 1 | 0 | `[0]` |
| 2 | 0 | `[0,1]` |
| 3 | 0 | `[0,1,2]` |
| 4 | 1 | `[0,1,2,3]` |
| 5 | 1 | `[0,1,2,3,4]` |
| 2047 | 511 | `[0..2046]` |
| 2048 | 512 | `[0..2047]` |
| 2049 | 512 | `[0..2048]` |
| 2050 | 512 | `[0..2049]` |
| 2051 | 512 | `[0..2050]` |
| 2052 | 513 | `[0..2047]` |
| 2053 | 513 | `[0..2047,2052]` |

For each case, every slot after its listed valid prefix through zero-based slot 2050 is `-1`. The
`n=2052/2053` rows prove lower-block-id tie selection and that an incomplete tail is unconditional
even after block-budget saturation.

For one pooled-key component, represented BF16 values `[1,1,1,1.03125]` have exact FP32 mean
`1.0078125`, exactly halfway between adjacent BF16 values. Round-to-nearest-even produces BF16
`1.0`. A selector that accumulates/rounds pairwise in BF16 or normalizes before pooling can change
this witness and is nonconforming.

### 12.3 GR formula fixture

Use diagnostic `B=4,H=1,R=1,eps=1`, zero norm weights, branch state `[1,2,3,4]`,
`W_down=[1,1,1,1]`, `W_up=[1,1,1,1]^T`, `W_write=I4`, and represented block output
`y=2`. The read/write FP64 oracle gives:

```text
Rhat = [
  0.70710678118654746,
  0.89442719099991586,
  0.94868329805051377,
  0.97014250014533188
]
W_down*Rhat/4 = 0.88008994259557727
SiLU(...)      = 0.62208353107647130
G (each branch)= 0.65069226675032232
x               = 0.57266771969167718
ideal_write_scale = [
  1.0881588866984520,
  1.1113398688917147,
  1.1180326522035791,
  1.1206768385310935
]
represented_BF16_write_scale = [
  1.0859375,
  1.109375,
  1.1171875,
  1.1171875
]
inject_oracle_R_out = [
  3.171875,
  4.21875,
  5.234375,
  6.234375
]
```

The inject oracle begins from the represented BF16 scale, not the ideal read/write value. This
single fixture detects moving either `/4`, omitting the branch mean, replacing the unit-offset
norm, making the write gate channel-wise, bypassing the public scale boundary, or adding a
static/mixing term.

### 12.4 PLE dilation fixture

For one diagnostic channel, let old normalized gated-value history at lags `-9..-1` be
`[1,2,3,4,5,6,7,8,9]`, current values be `[10,11,12,13]`, and convolution weights be
`[1,10,100,1000]`. Before SiLU, the four outputs must be:

```text
t=0: 1*1 + 4*10 + 7*100 + 10*1000 = 10741
t=1: 2*1 + 5*10 + 8*100 + 11*1000 = 11852
t=2: 3*1 + 6*10 + 9*100 + 12*1000 = 12963
t=3: 4*1 + 7*10 + 10*100 + 13*1000 = 14074
```

This distinguishes cross-correlation weight order, dilation three, a nine-row history, and
same-call visibility of current represented state.

## 13. Admission order

Implementation should land in this order without empty declarations or fake routes:

1. independent exact n-gram and floating reference fixtures;
2. contract header for one complete boundary above plus its host oracle test;
3. correct eager BF16 route at preview geometry, qualified directly against that oracle;
4. graph replay and C=1..8 envelopes for the same route;
5. only then additional cache/weight/residual codecs, each exact-decoded and independently
   qualified; and
6. Qwen4 Program integration after every called entry and state transition exists.

Do not create a header until its first supported implementation and meaningful test can land in
the same change. Do not add placeholder Qwen4 target calls, dense-attention fallback under the QSA
name, approximate hashing, MTP transaction stubs, or routes that merely return a reference tensor.

## 14. Pinned sources

- Qwen architecture report, arXiv 2608.30320v1:
  `https://arxiv.org/html/2608.30320v1`
- Transformers Qwen4Exp model at the pinned commit:
  `https://github.com/huggingface/transformers/blob/c119ec3cc37ab69642f39cca2de4187714002b08/src/transformers/models/qwen4_exp/modeling_qwen4_exp.py`
- Transformers Qwen4Exp configuration at the pinned commit:
  `https://github.com/huggingface/transformers/blob/c119ec3cc37ab69642f39cca2de4187714002b08/src/transformers/models/qwen4_exp/configuration_qwen4_exp.py`
- Official preview checkpoint configuration at the pinned revision:
  `https://huggingface.co/Qwen/Qwen3.8-Flash-Next/blob/de4b8e4d43b917e7706784d8bb445c9af86a3540/config.json`
