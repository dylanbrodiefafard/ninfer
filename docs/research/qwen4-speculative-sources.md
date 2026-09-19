# Qwen4 speculative checkpoint authorities

Research date: 2026-09-18. This records inspected publisher artifacts and executable
sources, not an admission of a NInfer implementation. The architecture remains Qwen4;
upstream checkpoint names retain their actual `Qwen3.8-Flash-Next` spelling.

## Exact-target DFlash checkpoint found

`PixelML/Qwen3.8-Flash-Next-NVFP4-DFlash` is a published, trained companion of the
same NVIDIA checkpoint used by the native qualification work. It is **DeepSpec
DFlash**, not the dense-27B DFlash2 companion and not an extracted native MTP head.

| Authority | Pin |
| --- | --- |
| Drafter HF repository | `9cd660f9050c92fedc88cbe547bd53af0392abe1` |
| Target NVIDIA checkpoint | `fc694b54fb0174e0913e6adf86691ef85a4ead47` |
| Publisher executable source/serving overlay | `PixelML/deepspec-qwen38-flash-next`, `54b35f57af721ffa8d55f1c0ae79f24952a82fd6` |
| Publisher's serving vLLM base | `e962733e08d10f7ca65dac4df99e116460b8b174` |
| Training config's official BF16 target revision | `f5d08274bafd880402bd16f5e3e6c514136ec06c` |

HF API inventory and two HTTP 206 reads (8-byte length, then 6,136-byte header)
independently confirm 58 BF16 tensors, 498,106,880 elements, 996,213,760 payload
bytes, and a 996,219,904-byte `model.safetensors` file. Initial discovery used headers;
the implementation subsequently acquired the complete bounded drafter. The file contains neither token embeddings nor
output head: both must alias the exact target's BF16 weights. The publisher reports
byte-identical training and NVIDIA target embedding/head tensors; acquisition should
retain its exact target binding rather than assume compatibility with another model.

There is **no codebook, selector, Markov head, or confidence head** in this inventory.
`markov_rank=0` and `enable_confidence_head=false` agree with the header. A BF16
codebook cannot be added merely because the existing dense-27B DFlash2 path uses one.
The string NVFP4 in this repository's name describes its **target**, not its draft
weight storage; every published draft weight is BF16.

### Complete tensor inventory

Shapes below are source row-major `[out,in]`. The per-layer group occurs five times,
at `layers.0` through `layers.4`; there are no biases.

| Tensor suffix | Shape | Count |
| --- | --- | --- |
| `fc.weight` | `[2560,12800]` | 1 |
| `hidden_norm.weight`, `norm.weight` | `[2560]` | 2 |
| `input_layernorm.weight`, `post_attention_layernorm.weight` | `[2560]` | 10 |
| `self_attn.q_norm.weight`, `self_attn.k_norm.weight` | `[256]` | 10 |
| `self_attn.q_proj.weight` | `[6144,2560]` | 5 |
| `self_attn.k_proj.weight`, `self_attn.v_proj.weight` | `[512,2560]` | 10 |
| `self_attn.o_proj.weight` | `[2560,6144]` | 5 |
| `mlp.gate_proj.weight`, `mlp.up_proj.weight` | `[7680,2560]` | 10 |
| `mlp.down_proj.weight` | `[2560,7680]` | 5 |

Config: hidden 2560, intermediate 7680, five full-attention layers, 24 Q heads,
two KV heads, head dimension 256, plain RMSNorm epsilon `1e-6`, SiLU MLP,
full-head NeoX rotary with theta `1e7`, vocabulary 248320, mask embedding row
248077, trained block length seven. All of these are checked by the publisher's
strict serving adapter in addition to its shape/dtype inventory.

### Exact feature boundary: do not infer it from the card's prose

The trained tap labels are `[3,15,23,35,43]`. The exporter and serving overlay
actually capture **attention-GR read/block-input outputs of layers
`[4,16,24,36,44]`**, respectively. The next layer's learned contraction consumes
the fully combined four-stream residual after the labeled preceding layer.
In vLLM this is tuple slot 1 of `attn_hyper_connection.mix` or
`combine_and_mix`; the Transformers GR equivalent is `mixed_input`, tuple slot 0.

This is not the prior layer's attention output, MLP input, raw 10240-wide residual,
an arithmetic mean of streams, or the final model mixer. No extra final norm is
applied at capture. Concatenate these five 2560-wide values, in order, into 12800
features. The final target hidden state is a training distillation target, not a
sixth input. For this exact target PLE is at layer index one, so none of these
capture boundaries needs a special PLE bypass.

### Formula and position/cache authority

For the native Engine integration, the PixelML drafter's position coordinate is the causal
token ordinal, independently of target three-axis MRoPE. At pinned source revision
`54b35f57af721ffa8d55f1c0ae79f24952a82fd6`,
`deepspec/modeling/dspark/qwen3/modeling.py:392–394` constructs context positions with
`arange(seq_len)` and appends `common.py:create_position_ids`, which adds query offsets to the
anchor's token ordinal. Thus a media-expanded prompt still gives the drafter context columns
0…N−1 and anchor N; the target's `rope_delta` is not added, and repeated image T-axis coordinates
are not substituted. Target MRoPE already affects the captured target features. This source-formula
choice is not evidence of this text-trained candidate's multimodal proposal quality.

The source backbone and `common.py` provide an executable formula independent of
NInfer's existing DFlash2 implementation:

1. For each accepted context token, fuse the five taps using `fc`, then apply
   plain `hidden_norm`. This same fused context feeds all five draft layers.
2. At anchor position `a`, the query block has K embeddings: the real anchor
   token followed by K-1 copies of target embedding row 248077. Query rotary
   positions are `a .. a+K-1`; **query zero predicts token `a+1`**. There is no
   discarded bonus query and no K+1 query layout.
3. Each draft layer RMS-normalizes only its evolving query stream before Q/K/V
   projection. Context K/V are projected directly from the shared fused context,
   without this per-layer input norm. Q/K use learned plain per-head RMSNorm;
   V does not. Apply full 256-dimensional NeoX RoPE to Q/K at their absolute
   positions. Scale attention logits by `1/sqrt(256)`.
4. Attention is noncausal within a draft block. Every query sees all K draft
   positions and only target-context positions strictly before anchor `a`.
   Multiple training anchors cannot see each other's noise blocks. GQA maps
   each group of twelve Q heads to one KV head.
5. Add projected attention to the query residual, then plain post-attention
   RMSNorm, `down(silu(gate(x))*up(x))`, and another residual addition.
6. After five layers apply plain final `norm`, then the target's shared output
   head. All K outputs propose tokens, in increasing position order.

The reference evaluator updates each layer's draft cache with new context K/V and
temporary query K/V, then crops back to `start` (the anchor position) immediately
after the block. Thus persistent draft cache contains **accepted target-context
features only**, never previous noise-query states. After target verification,
append only accepted real-context features on the next draft invocation. A rejected
target suffix must not enter this cache. Target QSA/GDN/PLE state is separately
transactional; it is not the drafter's ordinary full-attention cache.

The publisher overlay changes both proposal sampling offset and scheduler extra
slots. Copying only its method allowlist, or reusing dense-27B's skip-query-zero
layout, silently implements a different drafter.

### What is established and what still needs NInfer qualification

The publisher pairs this exact head with the target and publishes source adapter
code and bounded eager serving evidence. That is enough architecture authority to
implement and independently test the drafter. It is **not** authority to claim our
NVFP4 conversion, single-5090 graph path, stochastic acceptance, long-context
behavior, or future smaller-target compatibility is already qualified. The
publisher itself limits its strongest serving checks to greedy, non-thinking,
short output workloads; its adapter rejects draft quantization and graph mode.
Those are its qualified implementation constraints, not restrictions of the
mathematical architecture. Its imperfect and nonreproducible target comparisons
must not become weakened numerical gates in NInfer.

The bounded acquisition is the exact ~0.996 GB safetensors file plus
pinned config/card/train config. No full target download is needed. Retain the
source BF16 artifact for independent comparisons; create a Qwen4-specific NVFP4
matrix artifact using NInfer's explicit exact codec contract, BF16 norm vectors,
and A16 activations initially. This is our new quantization recipe, not a claim
to reproduce publisher-calibrated NVFP4 draft weights. Weight conversion must
have an exact decode oracle and a separate BF16-source approximation report.

A useful pre-small-model proof sequence is:

- source BF16 and decoded-NVFP4 FP64 mathematical oracles for each complete
  projection/norm/attention/MLP boundary and the five-layer composition;
- all five real layers, asymmetric context and noise, nonzero absolute positions,
  K=1/4/5/7 and context-prefix boundaries, with no private production casts in
  the mathematical oracle;
- independent masks/positions/feature-tap tests that expose a one-token offset,
  wrong HC contraction, causal-noise mask, or incorrectly layer-normalized context;
- first-call and incremental cached execution versus recomputation from accepted
  context, including zero/partial/full acceptance and rejected-suffix poison;
- target sampling's existing p-less, epsilon, and probability-floor policy reused
  for verification/correction, not the publisher's greedy-only serving claim;
- separate upstream BF16 executable trace comparison as supplementary evidence,
  followed by target end-to-end acceptance/PPL/speed once the full target can run.

### Implemented acquisition, oracle and geometry qualification

`tools/parity/qwen4/native_dflash_fixture.py` acquired the pinned source, exports
BF16 and explicitly max-abs-encoded NVFP4 qualification artifacts, and refuses to
replace existing files. Exact source word checks cover all retained BF16 weights;
an independent explicit-index address oracle checks every NVFP4 code, E4M3 scale,
and FP32 divisor word. The existing encoder remains
`DFLASH2_NVFP4_MAXABS_TWOLEVEL_V2`; its algorithm is adopted explicitly, not
renamed as a publisher-calibrated Qwen4 recipe. The resulting files are under
`local_llm/models/qwen4-dflash`, with source names and provenance retained.

The input fixture adds only the exact source mask row 248077 and 29 unique
BF16 anchor rows already present in the native real-token panel. All target
embedding rows use the NVIDIA source pin. No full embedding payload is duplicated.

`tools/reference/qwen4/dflash.py` implements the independent FP64 formula and
accepted-context cache semantics, including exact packed-weight decoding.
`tools/parity/qwen4/native_dflash_goldens.py` emits raw little-endian FP64 planes
plus JSON offsets, not another product artifact format. Full five-layer goldens
exist for both weight representations at `(context,K)=(0,1),(3,4),(65,7)`, with
nonzero absolute positions, authentic anchor/mask embeddings, and explicitly
synthetic BF16 context features. They retain both no-intermediate-cast whole
formulas and separate supplementary traces rounded only at explicit public Op
outputs. The latter do not replace same-input Op oracles.

Python checks cover signed NVFP4 codes, scale addressing across tile boundaries,
exact divisor bits, anchor layout, noncausal/GQA masks, unoffset/context norms,
and five-layer cached-versus-recomputed/poisoned-suffix behavior. FP64 identity
criteria were fixed at `atol=rtol=1e-12`; exact transformations compare exactly.

The additional Linear geometries `[2560,12800]`, `[7680,2560]`, and `[2560,7680]`
are admitted for BF16 and NVFP4 A16 using the existing GEMV/SmallT/MMA schedules.
The NVFP4 geometries remain A16 even under permissive `AllowA4`. No schedule
optimization or speed claim is made. On RTX 5090 / CUDA 13.1, both complete
BF16-A16 and NVFP4-A16 Linear test executables passed, including new geometry
cases `T=1,7,33,129`, unchanged independent FP64 criteria, and input/output guards.

The conversion is **not yet a qualified draft-quality recipe**. Its 36 matrices
have BF16-source relative-L2 weight approximation errors 0.094641–0.095227.
On the synthetic-feature whole-formula probes, NVFP4-versus-BF16 final-hidden
differences are 0.37267, 0.72377, and 0.76814. Those are source-weight approximation
effects with no GPU kernel in the comparison, not evidence of a packing bug and
not a measurement of real-target acceptance. Real captured HC features and
acceptance tests are needed before defaulting to this quantization recipe.

The explicit BF16 Op-boundary reference itself differs from the no-cast ideal
whole formula by 0.01113/0.01727/0.02196 for source BF16 weights and
0.01123/0.01763/0.02032 for decoded NVFP4 weights. The `(65,7)` case thus crosses
the separately declared 2% whole-composition screen even without implementation
error. Preserve that screen and explain the result; do not increase it or use
pairwise parity as a substitute for independent same-input Op checks.

`tools/parity/qwen4/native_dflash_check.py` checks each captured GPU public output
directly against FP64 from its **actual preceding GPU public inputs**. It uses
the unchanged independent Linear, RMSNorm, RoPE, attention, SiLU-multiply and
residual-add criteria, with exact decoded NVFP4 coefficients. The whole-formula
and public-BF16 composition screens remain separate report entries. Its exit
status covers local Op checks only, not a quality-default or full-target admission.

The initial six-case native GPU traces contained 604 local Op observations.
The checker found an avoidable attention error despite all composed 2% public-
boundary screens passing: a BF16 partial numerator before the final attention
division changed one output from the independently computed 4.59367828 to 4.625.
The diagnostic separately isolates probability-operand rounding from the extra
partial cast; keeping that partial numerator unrounded gives final BF16 4.59375.
This private-profile diagnosis does not replace the mathematical oracle.

Three initial RMSNorm gross-screen failures are representability limits, not
arithmetic bugs. The GPU values are the nearest BF16 values to the FP64 formula;
the original `1e-5 + 3.4e-3 * max_abs(reference)` envelope excludes even those
nearest representable outputs. The checker retains each failed raw screen and
adds an independent nearest-BF16 proof (neighbor comparison in FP64, even-code
ties) without widening the gate or excusing avoidable error. Regression tests
distinguish these impossible envelopes from the avoidable attention failure.

After retaining D256 partial numerators in FP32, the same six-case, 604-observation
checker passes every attention, Linear, RoPE, SiLU-multiply, and residual-add
check. All RMSNorm relative-L2 checks also pass. Five raw RMSNorm gross-envelope
failures remain: BF16 `C0K1 InputNorm[4]`, `C3K4 PostNorm[4]`, `C65K7 KeyNorm[1]`,
and NVFP4 `C65K7 ContextKeyNorm[1]`, `C65K7 InputNorm[3]`. In each case the entire
GPU output plane exactly equals independently selected nearest-BF16 values;
one coordinate's minimum representable error exceeds the unchanged gross
envelope. The checker therefore still exits 1 and reports those raw failures
with their representability proofs, rather than claiming every numeric screen
passes. The resulting report is
`out/qwen4-dflash-traces/local-oracle-fp32-partial.json`.

All six public-BF16 composed screens pass after the fix. No-cast ideal versus
GPU relative-L2 is 0.010373/0.018874/0.024304 for BF16 and
0.010603/0.017751/0.019514 for NVFP4 at the three cases respectively. Only the
BF16 `(65,7)` case still exceeds the separate 2% ideal screen; its public-BF16
composition error is 0.016655. These results establish local implementation
accuracy subject to the explicitly impossible RMSNorm gross envelopes; they do
not qualify converted draft quality or whole-target speculative acceptance.

### Shared endpoint and finished implementation qualification

`DFlashProgram::draft` accepts independent anchor RoPE positions and logical context lengths,
prepares the authentic anchor/mask sequence, executes all five layers, and applies the actual
target BF16 head. The endpoint test materializes only addressed authentic embedding rows in a
vocabulary-sized poison-filled allocation; it does not claim a complete target embedding artifact.
For both BF16 and NVFP4 draft profiles, every output row of all seven queries is checked against
an independent FP64 head dot product from the actual represented final hidden input. Exact
argmax, including query zero, shifted positions, manual gather correspondence and complete
endpoint CUDA Graph replay pass. Separate model tests cover permuted four-request slots,
context lengths 0/1/3/65, chunked accepted context, immutable noise-query caches and rejected
NaN suffixes. The independent feature-tap test checks all 48 layer labels and request placement.

Astra reviewed the finished binder/schedule, source tap semantics, endpoint, cache behavior,
D256 attention partial fix and numerical checker without an actionable finding. This establishes
the stated component and schedule behavior, not full-target sampling/acceptance or draft quality.

### Actual diagnostic-target feature comparison

`ninfer_qwen4_dflash_target_features_test` is a two-phase opt-in diagnostic. `--capture`
loads the existing full 48-layer UD-IQ1_S verifier and sequentially consumes the first 24
tokens of the authentic 33-token text/code panel, capturing the trained attention-GR inputs
at layers 4/16/24/36/44. These are accepted teacher-forced prompt tokens, not sampled draft
acceptance. The target's entire 28,800,139,264-byte page-rounded PLE mapping was locked
before device loading; the temporary 48 GiB/no-swap, unlimited-memlock container drained and
unloaded the target before `--compare` loaded either drafter. No ordinary-weight residency
exception beyond the existing diagnostic profile was introduced.

Both drafter profiles receive the same captured represented BF16 features and identical
authentic NVIDIA BF16 anchor/mask embeddings and full vocabulary head. Thus this comparison
isolates draft weight quantization on actual target-derived features; the features themselves
come from the low-bit diagnostic target, not a full native BF16/NVFP4 target. Every draft
projection uses A16. Three contexts 8/16/24 each predict seven tokens. Results on RTX 5090 /
CUDA 13.1.2, 2026-09-18:

| Accepted context | Hidden relative L2 | Head-logit relative L2 | Top-1 agreement |
|---:|---:|---:|---:|
| 8 | 0.336290 | 0.130688 | 7/7 |
| 16 | 0.337712 | 0.145163 | 3/7 |
| 24 | 0.362217 | 0.162875 | 6/7 |

Top-1, KL and point label-NLL use the full **248,320-row head domain**, temperature one,
without an extra validity-prefix mask or conditioning. These are diagnostics, not the
engine's p-less/epsilon proposal distribution. BF16-to-NVFP4 KL ranges from 0.002110 to
0.738390 nats (mean 0.176603 across 21 queries). Mean label-NLL difference is +0.194942
nats on this tiny panel; it is not model PPL or an acceptance-rate estimate. The 16/21 top-1
agreement and remaining hidden/logit drift do not qualify this locally converted NVFP4
drafter as a quality default. No numerical gate is weakened to admit it.

The independent represented-input Op oracles remain the implementation correctness evidence;
this pairwise run is a separate quantization-quality diagnostic. Its 614,400-byte feature
panel, provenance and comparison report are in `out/qwen4-dflash-target/`. The executable
skips by default. Explicit runs require `NINFER_QWEN4_NATIVE_LAYERS`,
`NINFER_QWEN4_NATIVE_DFLASH`, `NINFER_QWEN4_DFLASH_TARGET_OUTPUT`, and, for capture,
`NINFER_QWEN4_WEIGHTS`. Reports refuse replacement, so retained panels can be compared again
without another full target load by choosing a fresh report destination/copying the panel.

### One complete staged diagnostic round

`ninfer_qwen4_dflash_staged_round_test` closes the bounded full-target/draft/sampler/state
composition check without requiring simultaneous target/drafter residency. Its first phase
validates the exact existing diagnostic artifact and uploads only its unchanged Q4_K token
embedding and Q4_K head; it does not substitute native BF16 endpoints. Existing GPU embedding
decode, NVFP4/A16 DFlash, GPU head projection and full-domain argmax produce seven proposals.
These endpoint/draft owners drain and unload before the full diagnostic target is loaded with
its eager locked PLE table. No new production binder, verifier API or streaming exception is
introduced. Every captured context feature is compared bitwise with fresh target execution.

The target sequentially verifies the old anchor and all seven actual proposals. The existing
shared `speculative_accept_greedy_drafts` uses the full 248,320 domain with temperature 0.6,
p-less enabled, unchanged epsilon 0.0625/support cap 1024, and seed 1939. On the 24-token
teacher-forced context, the natural proposals were `[830,478,271,1302,198,279,279]`; one draft
was accepted and the licensed sequence was `[830,248046]`. This is one observed round, not an
acceptance-rate or quality qualification.

The run passed with **zero failures**. It checks exact prefix occurrence-count rollback and
all **157,147,144 bytes** of continuation state: every GDN convolution and FP32 recurrence,
every QSA core KV/scale/index/position plane, PLE convolution and raw-token history, and
the four-stream residual. Cancellation, one-token publication, and complete two-token
publication each reset/replay the accepted **verified input** prefix and match the saved
pre-rejection state exactly. The licensed correction remains the next anchor, not a processed
input. Its next-token logits and complete resulting state match a fresh causal replay without
the rejected suffix. The tool deliberately uses expensive reset/replay, not a production
speculative scheduling implementation or performance path.

The same load exports small actual target four-stream MTP inputs for separate qualification:
24 prompt residuals with exact GPU-decoded shifted embeddings, eight verified residuals, original
positions, verified/eligible licensed IDs and committed count. These are `State::residual`,
never the final 2560-wide GR output. The initial anchor is teacher-forced; the newly licensed
correction is sampled. Files are `out/qwen4-dflash-target/qwen4-mtp-target-inputs.bin/.json`;
the round report is `qwen4-dflash-staged-round.json` in that directory. GPU/build focused checks
and the opt-in default skip pass; Astra's finished-code review found no actionable issue.

### Bounded protected-projection ablation

`tools/parity/qwen4/native_dflash_precision.py` reuses the independent FP64 formula on the
captured diagnostic-target context of 24 tokens, with seven noise queries, identical native BF16
anchor/mask embeddings, exact represented weight decode and explicit public BF16 Op outputs.
The source BF16 result is the reference. All-NVFP4 matrices produce hidden relative L2
`0.361924`; preserving only `fc.weight` (feature fusion) in BF16 changes it to `0.315941`.
That projection conditions accepted context in all five layers, so this is a bounded causal
test of a plausible protection choice, not an arbitrary matrix sweep.

Protection reduces this witness's drift but does not eliminate it. No new runtime/artifact
profile is admitted. These are weight-precision effects between different represented models,
not same-input kernel-oracle failures, calibrated quality estimates, PPL, or acceptance-rate
evidence. The one diagnostic-target context cannot determine a production Pareto choice.
The local report is `out/qwen4-dflash-target/qwen4-dflash-precision.json`.

The subsequent role-group experiment in the same tool uses accepted context prefixes
8, 16 and 24, each with the corresponding authentic BF16 anchor embedding and seven queries.
It keeps all non-matrix controls protected and changes only represented matrix weights.
The three prefixes overlap; they are not independent calibration or holdout examples.

| BF16 matrix protection over local NVFP4 candidate | Added payload bytes | C8 hidden relative L2 | C16 | C24 |
|---|---:|---:|---:|---:|
| None | 0 | 0.336064 | 0.336973 | 0.361924 |
| Feature fusion FC | 47,103,996 | 0.275892 | 0.300185 | 0.315941 |
| FC and all five layers' K/V projections | 65,945,556 | 0.286073 | 0.227439 | 0.249289 |
| FC and all attention projections | 292,044,716 | 0.209180 | 0.198716 | 0.192334 |
| All MLP projections | 423,935,940 | 0.247269 | 0.284524 | 0.274137 |

The exact byte differences compare stored BF16 and NVFP4 matrix payloads, excluding unchanged
controls and container framing. FP64 composition also records every layer's hidden drift.
Neither the small context-entry protection nor the much larger attention/MLP exceptions rescue
this candidate; error accumulates through both sublayers, and protection is not monotonic across
prefixes. This is evidence against admitting a guessed protected-role recipe, not against
well-calibrated NVFP4 generally. Keep the source BF16 baseline and the existing explicit
experimental NVFP4 opt-in; no production precision policy changes follow. The next useful
quality evidence is a genuinely calibrated/trained draft checkpoint and representative accepted
target features, not more uncalibrated exceptions selected on this single prompt.

CPU-only Python 3.11/PyTorch run, exact represented weights, unchanged independent FP64 formula
and explicit public BF16 boundaries; report:
`out/qwen4-dflash-target/qwen4-dflash-role-precision.json`.

### License provenance

The publisher declares the artifact under the Qwen Community License 1.0 and
NVIDIA Open Model License, separately from its MIT-licensed code. Its source
repository contains MIT `LICENSE` and a `NOTICE` identifying upstream projects;
some unrelated Eagle3/training components carry Apache-2.0 provenance. Preserve
the applicable notices if copying source. This records publisher declarations,
not a new legal interpretation or a claim the weights are MIT.

## Other discovery results

HF model API search for `Flash-Next` returned this PixelML DFlash checkpoint plus
several extracted MTP drafters. Searching `Qwen4` returned no additional matching
DFlash/DSpark drafter in the inspected result set. This is a dated search result,
not a claim that no other model can exist. The dense `z-lab/Qwen3.8-27B-DFlash2`
checkpoint remains a different target/topology and cannot supply missing weights.

`limpincat/flashnext-drafters` at
`39d7d235eb4748cd90d3ae575a2a2e54b49018c9` contains an `mtp-nvfp4/` tree:
four routed-expert shards, a BF16 shard, index, quantization config, and build
manifests. This is MTP, not DFlash. The subsequent exact inventory/source comparison,
A16-only execution profile, cache semantics and numerical limitations are documented in
`qwen4-mtp-execution.md`; its placeholder input scales do not authorize A4.

## Source addresses

- HF metadata: https://huggingface.co/api/models/PixelML/Qwen3.8-Flash-Next-NVFP4-DFlash
- Pinned artifact/config/card/train config: https://huggingface.co/PixelML/Qwen3.8-Flash-Next-NVFP4-DFlash/tree/9cd660f9050c92fedc88cbe547bd53af0392abe1
- Source tree: https://github.com/PixelML/deepspec-qwen38-flash-next/tree/54b35f57af721ffa8d55f1c0ae79f24952a82fd6
- Backbone: https://github.com/PixelML/deepspec-qwen38-flash-next/blob/54b35f57af721ffa8d55f1c0ae79f24952a82fd6/deepspec/modeling/dspark/qwen3/modeling.py
- Masks/noise/positions: https://github.com/PixelML/deepspec-qwen38-flash-next/blob/54b35f57af721ffa8d55f1c0ae79f24952a82fd6/deepspec/modeling/dspark/common.py
- Cached evaluator: https://github.com/PixelML/deepspec-qwen38-flash-next/blob/54b35f57af721ffa8d55f1c0ae79f24952a82fd6/deepspec/eval/dspark/draft_ops.py
- Exact trained feature capture: https://github.com/PixelML/deepspec-qwen38-flash-next/blob/54b35f57af721ffa8d55f1c0ae79f24952a82fd6/scripts/data/export_features_vllm.py
- Serving layout/binding patches: https://github.com/PixelML/deepspec-qwen38-flash-next/blob/54b35f57af721ffa8d55f1c0ae79f24952a82fd6/serving/adapter/adapter-overlay.patch
- Strict weight/config adapter: https://github.com/PixelML/deepspec-qwen38-flash-next/blob/54b35f57af721ffa8d55f1c0ae79f24952a82fd6/serving/plugin/dflash_epoch7.py
- Serving config override: https://github.com/PixelML/deepspec-qwen38-flash-next/blob/54b35f57af721ffa8d55f1c0ae79f24952a82fd6/serving/adapter/draft-config.json
- License/notice: https://github.com/PixelML/deepspec-qwen38-flash-next/tree/54b35f57af721ffa8d55f1c0ae79f24952a82fd6
- Alternative MTP lead: https://huggingface.co/limpincat/flashnext-drafters/tree/39d7d235eb4748cd90d3ae575a2a2e54b49018c9/mtp-nvfp4
