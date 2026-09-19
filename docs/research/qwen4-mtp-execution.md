# Qwen4 private MTP execution authority

Audited 2026-09-18. This is source/codec research, not a claim of full-model MTP
qualification. The private block fits independently on the RTX 5090; the complete
target does not need to be loaded to qualify its represented-input mathematics.

## Sources and provenance

| Authority | Exact revision | Use |
|---|---|---|
| NVIDIA Qwen3.8-Flash-Next-NVFP4 | `fc694b54fb0174e0913e6adf86691ef85a4ead47` | Exact MTP tensor headers and reference weights |
| limpincat/flashnext-drafters, `mtp-nvfp4/` | `39d7d235eb4748cd90d3ae575a2a2e54b49018c9` | Candidate represented NVFP4 expert matrices, not calibrated activations |
| vLLM original model authority | `d6bce42983bc0b2095ad6422dbf1399e219ae572` | Existing recurrence/selection authority |
| vLLM current audit | `a1bf8ac12d9f1537ff2d233f5ab3d1346fd8bd44` | Current MTP model, proposer, and QSA behavior |
| SGLang original integration | `78c5024e9d9f589dcb4deb7f4ba4fb23f7e85385` | Previous independent recurrence cross-check |
| SGLang current audit | `5e4b94b134e4dcc9a10fac465f94cdf576268646` | Current recurrence and divergent reuse-tail behavior |
| TokenSpeed current audit | `e09336bbfdc6dadf497907e8d4d139812381e7a7` | Third implementation linked from official Qwen README |

The initial audit used index JSON, bounded safetensors headers, scalar byte
ranges, and source code. Subsequent authorized qualification acquired only the
two private MTP namespaces described below. No checkpoint executable/custom code
was executed; no full target or PLE payload was downloaded.

## Actual weight inventories

The original official BF16 checkpoint contains 31 private tensors because its
gate/up and down expert banks are combined. That count does **not** describe the
NVIDIA quantized checkpoint.

NVIDIA's private namespace contains 3,101 tensors and 2,698,026,496 payload bytes:

- 1,536 routed matrices, each FP8 E4M3, totaling 2,516,582,400 code bytes;
- 1,536 BF16 `weight_scale_inv` arrays totaling 307,200 bytes, with logical
  128-by-128 blocks (`[5,20]` for gate/up, `[20,5]` for down);
- 29 other tensors, all BF16, totaling 181,136,896 bytes. These include both stem
  matrices, norms, three GR modules, QSA projections/indexer, router, shared expert,
  and shared-expert scalar gate.

The FP8 expert payload lives in `model-fp8-mtp-ple.safetensors`; the BF16 tensors
are in shards 9 and 10 of 10. Exact range acquisition must exclude the PLE payload.
Block-FP8 scales must not be reinterpreted as row scales or tensor scales.

The limpincat candidate has 6,173 tensors and 1,596,726,784 payload bytes across
four 128-expert files and one BF16 file. Its 29 non-expert tensors have the same
names/dimensions as NVIDIA's protected roles, including **BF16**, not NVFP4, stem
matrices. This is a predominantly NVFP4 private block by weight count; there is no
need to manufacture an NVFP4 stem to call its existing recipe faithful.

The candidate's explicit stored codec is:

```text
W[n,k] = E2M1_signed(code[n,k])
         * E4M3FN(weight_scale[n,floor(k/16)]) * F32(weight_scale_2)
code[n,2*j] is the low nibble; code[n,2*j+1] is the high nibble.
```

Gate/up have separate code/scale arrays but share one global multiplier per
expert; down has its own multiplier. Logical shapes are `[640,2560]` gate/up and
`[2560,640]` down. The exporter manifest describes an exhaustive E4M3 scale search
and a multiplier derived from `amax/(448*6)`.

**Its activation scales are explicitly placeholders.** Both the layer summary and
per-shard completion manifest say `input_scale_source: placeholder`; sampled
expert-zero gate/up/down scalar words are exactly F32 `1.0`. This permits an
exact represented-weight **W4A16** qualification, not an A4 calibration claim.
The MTP binding must prohibit `AllowA4` for this fixture rather than silently use
placeholder scales. Stored input scales should still be preserved/identified in
conversion provenance, not replaced with invented calibration.

The candidate build log names only a local `flashnext-ssdstream/mtp` source;
it does not pin an upstream checkpoint revision or publish calibration data.
Its config also contains `ple_layer_ids: [2]`, unlike this repository's existing
exact target authority. Do not import its full target config or infer provenance
from its ModelOpt producer label. Before source-faithful claims, compare all 29
BF16 controls byte-for-byte against the pinned NVIDIA source and measure selected
expert weight loss separately. Exact codec correctness can be established even
when ancestral provenance is incomplete.

### Acquired private fixture and exact verification

`tools/parity/qwen4/native_mtp_fixture.py` now acquires both exact private
namespaces and emits `qwen4-mtp-nvfp4.ninfer`, identity
`qwen4/native-mtp-qualification` / `limpincat-nvfp4-w4a16-source`. The local files
are under `/ssdpool2nvme/local_llm/models/qwen4-mtp` (container `/models/qwen4-mtp`).
The artifact has 29 BF16 controls/matrices plus three NVFP4 expert-bank tensors;
it does not register an Engine identity or imply a complete target artifact.

On 2026-09-18, all 29 candidate BF16 tensors compared **word-exactly equal** to
the pinned NVIDIA source. An independent scalar scale-address oracle checked
every code byte, scale byte, and F32 multiplier word across all 512 experts and
all three bank projections after conversion. All comparisons passed. This
verification does not call the production layout unswizzler.

Selected source-weight comparisons against NVIDIA's decoded block-FP8 matrices:

| Expert | Gate relative L2 | Up relative L2 | Down relative L2 |
|---|---:|---:|---:|
| 0 | 0.08546145 | 0.08553062 | 0.08548236 |
| 511 | 0.08553871 | 0.08554451 | 0.08555558 |

These are weight-only differences between two published representations, not
CUDA output tolerances, quality acceptance criteria, or proof that the original
publisher's unpublished conversion input is byte-identical to NVIDIA's. The
fixture JSON retains these comparisons and explicitly marks activation policy
`A16-only` and input scales `publisher-placeholder-not-calibration`. A consumer
must enforce that policy; the generic NVFP4 storage codec alone does not forbid A4.

The acquisition/conversion and `--verify-only` commands ran with the existing
Python 3.11 numerical-reference environment, CPU-only. `py_compile` and scoped
`git diff --check` passed. GPU execution evidence is recorded below.

## Recurrence that agrees across implementations

Let `R` have four 2560-wide branches and let `e` be the target-token-aligned
embedding. The hidden stem norm is one 10240-wide norm, not four branch norms:

```text
u = fc_embedding(offset_rmsnorm(e))
V = reshape(offset_rmsnorm(flatten(R)), [4,2560])
R0[j] = fc_hidden(V[j]) + u
R1 = MTP_QSA_GR_then_MoE_GR(R0)
logit_input = MTP_final_GR_read(R1)
logits = shared_target_output_head(logit_input)
next_carried_hidden = R1
```

Draft step zero consumes the target's **pre-final-GR** four-stream hidden. Later
draft steps consume the prior draft's pre-final-GR result, not a repeated final
read vector. The private layer has its own QSA core KV/cache and no PLE or GDN.
The target still owns its independent QSA/GDN/PLE state during verification.

Current vLLM defers the stem embedding add into the first GR combination while
the old vLLM and current SGLang materialize it before that GR. This is the same
logical formula, not a mandate to copy either implementation's internal cast
or reduction placement. Public BF16 outputs and specified persistent cache
boundaries must be named by each tested Op; naive FP32/FP64 oracles evaluate
the formula rather than replay private fused casts.

### Exact token/hidden/position alignment

At audited vLLM revision `a1bf8ac12d9f1537ff2d233f5ab3d1346fd8bd44`,
`llm_base_proposer.py::set_inputs_first_pass` shifts target IDs left, substitutes the
next anchor into each request's last row, and copies target positions and hidden rows
without shifting them. The separate GPU autoregressive `prepare_prefill_inputs` path
does the same, including the next known token for chunked prompt continuation. Its
sampling comment explicitly distinguishes the hidden's source position from the token
being predicted. These are input-control semantics, not a private arithmetic profile.

For a target row after input token `x_t`, use its four-stream pre-final-GR state `R_t`
with `embedding(x_{t+1})` at private RoPE position `t`. The result predicts `x_{t+2}`.
The full prompt seed therefore uses `R[0:N]`, embeddings for
`x[1:N] + [next_anchor]`, and unshifted original positions. Later private calls use
the previous private carry and the preceding proposal embedding while advancing
private position by one. The predicted token's preceding position, used by a
counter-based sampling caller, is one greater than this private position.

For an accepted verified-input prefix, replay **actual target** hidden row `j`
with embedding of licensed output `j`, retaining that hidden row's source position.
The final licensed output is the correction/bonus anchor. It is an embedding input
to the MTP stem, not an extra already-processed target input. Discard provisional
private drafts before this target-aligned extend. Cancellation replays no rows.

Pinned implementation addresses:

- https://raw.githubusercontent.com/vllm-project/vllm/a1bf8ac12d9f1537ff2d233f5ab3d1346fd8bd44/vllm/v1/spec_decode/llm_base_proposer.py (`set_inputs_first_pass`, input rotation and `_set_positions`).
- https://raw.githubusercontent.com/vllm-project/vllm/a1bf8ac12d9f1537ff2d233f5ab3d1346fd8bd44/vllm/v1/worker/gpu/spec_decode/autoregressive/speculator.py (`_prepare_prefill_inputs_kernel`, `_prefill`, and `_prepare_decode_inputs_kernel`).

Draft positions advance one token per step. RoPE coordinates and logical cache
positions must be represented independently: current SGLang explicitly notes
that speculative RoPE positions can advance independently of physical paged-KV
positions, and computes compression/selection from logical cache positions.
Never address a cache with the next RoPE coordinate merely because they coincide
in a short text-only fixture.

## Selection reuse is not currently cross-implementation identical

The existing model reference accurately records agreement on step-zero selection
and carried hidden, but its short reuse description is insufficient to freeze
all later-step attention domains.

**Pinned vLLM profile:** step zero runs the private indexer and the proposer
compacts each request's target-aligned last row. Later steps return that exact
frozen selection. The indexer still updates raw/compressed side cache before its
`skip_topk` return. Current vLLM packs 2051 possible token indices plus one valid
count column; the count column is not a token. Target verification computes its
own indices. Do not turn the packed width 2052 into 2052 attended tokens.

**Current SGLang profile:** `QSAMTPSharedSparseIndices` retains the captured
selection and captured length, then appends indices from that length through the
current logical position in an additional reserved tail. The Qwen4 model bypasses
the indexer entirely on reused draft decode rows. Accepted draft-extend rows
refresh the seed. Thus its later draft attention domain and provisional index
side-cache update policy differ from vLLM, despite agreeing on the stem and
four-stream recurrence.

The **same disagreement already exists at the previous pins**: SGLang `78c...`
lines 174–195 append the post-capture interval, while vLLM `d6b...` lines 352–355
return the unchanged step-zero buffer. This is not a recent regression explaining
away the old description.

An explicit logical-set formulation makes the missing authority precise. Let
`L` be the captured prefix length, `B` its selected complete four-token blocks,
`U=[4*floor(L/4),L)` its original incomplete-block tail, and `p>=L` the current
draft position. Then:

```text
frozen_seed = expand(B) union U
vLLM later-step domain = frozen_seed
SGLang later-step domain = frozen_seed union [L,p]
```

The appended interval is disjoint from the seed. Under the SGLang interpretation,
draft-created complete blocks stay attended without index reselection; replacing
the interval with only the *current* incomplete block would drop draft tokens at
compression boundaries and would not implement either published protocol.

The paper's Section 2.1 says the backbone and MTP use QSA, normal QSA includes its
incomplete-block tail, and MTP reuses top-k indices. It does not specify whether
new draft tokens are appended to the reused set. Calling that appended interval
mandatory would be an inference, not a proved official requirement. Do not choose
an incompatible profile implicitly. This disagreement does not prevent qualifying
the private block on explicit represented selection inputs, but full speculative
admission needs an authoritative selection policy plus execution evidence.

TokenSpeed supplies an additional independent implementation check. Its QSA
indexer continues K compression/draft staging, then returns the unchanged shared
top-k rows; its paged attention backend consumes those explicit indices without
appending the post-capture tail. This agrees with vLLM's frozen-domain profile.
The official Qwen README lists all three serving implementations but contains no
executable MTP golden or tail-policy specification. Two implementations agreeing
is useful evidence, not proof of the training-time attention domain.

## Feasible qualification without a fitting target

1. Acquire only the private source block and candidate, not target/PLE shards.
   All private matrices fit resident on the 5090. Shared embedding rows are
   selected by exact token IDs; the existing full BF16 output-head fixture can
   be reused for vocabulary logits.
2. Bind the candidate with W4A16 routed experts and source BF16 protected roles.
   Exact tests cover all code nibbles, block scales, multipliers and bank ordering.
   Compare complete block outputs and intermediate public state to independent
   represented-input mathematical oracles, including the private final GR.
3. Separately compare against exact decoded NVIDIA block-FP8 source weights and
   report weight-only loss. Source loss is not a CUDA implementation error and
   cannot be hidden by widening an Op's correctness envelope.
4. Produce small executable state traces at four steps around compression
   boundaries, including a nontrivial long-prefix selection (>2048 candidates),
   per-request compaction, and distinct RoPE/logical positions. Test the selected
   upstream profile and an independent state-transition model. Synthetic hidden
   inputs qualify the block/transition, not real target acceptance rates.
5. Test rejection at every draft position, partial/full acceptance, bonus-token
   publication, and recomputation of retained target-aligned draft state. Preserve
   target GDN, QSA, PLE convolution/hash histories and request positions at the
   same accepted frontier. Provisional draft state cannot overwrite committed
   target state or survive rejection through an index-cache alias.
6. Keep the repository's shared p-less/epsilon/probability-floor sampler. Neither
   upstream's greedy top-1 path nor its rejection-sampling default replaces that
   product contract. Sampling distribution/state tests are distinct from block
   arithmetic and from unavailable full-target acceptance/throughput/PPL tests.

## Implemented bounded execution and numerical evidence

The selected profile is explicitly named
`qwen4-mtp-vllm-tokenspeed-frozen-domain-w4a16`. `src/targets/qwen4/mtp.{h,cpp}`
binds the real private fixture and runs the complete stem/QSA/GR/MoE/final-GR block.
The original bounded schedule owned carry/cache/selection and rollback/replay state; current
native ownership is described in the paged migration section below. Both expose no A4 choice.
Logical cache IDs are distinct from three-axis RoPE coordinates.
Accepted target-aligned rows are recomputed from target hidden, not folded from
draft hidden. Shared embedding/head/sampling remain caller-owned.

On RTX 5090, CUDA 13.1, sm_120a, the opt-in native C++ test ran a five-token seed,
four draft steps crossing a compression boundary, exact seed-selection reuse,
discard and deterministic re-draft, and all accepted counts 0–4 with retained
target-hidden replay versus a fresh execution. Accepted-frontier comparisons
include every live core K/V row, raw index key, RoPE position, selected ID/count,
and carried hidden. Rejected first-draft validation preserves the seed. The
native artifact and all its matrices remained GPU-resident. The test inputs are
deterministic represented BF16 vectors, not captured full-target hidden states.

`tools/reference/qwen4/mtp.py` independently decodes the candidate NVFP4 weights
and evaluates complete FP64 formulas from actual public inputs at each GPU Op
boundary. For QSA this includes the preceding GPU record's represented live
K/V, raw index keys, positions, and frozen selected IDs/count. Historical K is
already normalized and rotated; only newly projected K rows undergo those
transforms. Historical cache rows and positions are checked for exact retention,
new appends are checked independently, and the next local invocation advances
from the actual recorded state, never an oracle-reconstructed history. The
separately propagated diagnostic retains its own independently evolved cache.
All local gates passed: stem relative L2 0.00284–0.00292, QSA
0.00353–0.00514, routed/shared MoE 0.00286–0.00363, and GR reads approximately
0.00162–0.00175. Actual-input router IDs, frozen domains and cache positions
matched exactly. Every historical K/V/raw-index row was retained exactly. New
BF16 core K and raw index rows were exact for this workload; newly appended V
maximum difference was 0.00390625 (relative L2 at most 0.0001931, measured on
the new rows alone rather than diluted by old exact rows).

Astra review identified the original local oracle's use of its reconstructed
historical cache as a represented-input mismatch. The corrected existing-trace
run passes all local gates and exact history/domain checks. Two CPU regression
tests additionally use non-unit already-transformed historical K, altered
historical V, a high-valued excluded draft row, and deliberate historical-state
and frozen-domain corruption. They verify that the local oracle consumes actual
historical state without re-normalizing/re-rotating it, excludes the new tail,
and detects mutation; both tests pass.

The injection oracle remains the ideal FP64 `R + scale*block`. All ten injection
results exactly equal its nearest-even BF16 output encoding. Four older empirical
gross-error screens failed even though every output word was correctly rounded:
for example, ideal −4.452419281005859 rounds to the actual −4.4375, error
0.014919281 versus the old 0.011862183 screen. The correct output-format criterion
is exact nearest-even BF16 encoding, not rounding the mathematical oracle before
measuring error or enlarging an observed-error envelope.

A separate independent chain propagates each public BF16 result through the
entire recurrence. Its original 2% screen is retained and reports **seven failed
diagnostic comparisons**, not a false all-green whole-chain claim. Seed carry
relative L2 is 0.00662–0.01559; draft carries are 0.01334, 0.03280, 0.10349 and
0.06557. Step 3 changes one of ten routed expert memberships; final-read relative
L2 peaks at 0.13255. Other steps retain membership, so this is also nonlinear
amplification of small local numerical differences, not solely top-k switching.
No kernel error was identified by the actual-input local gates. No PPL,
proposal-acceptance or full-target quality claim follows from these diagnostics.

The central explicit-selection QSA route also has a complete independent FP64
attention oracle over a fixed, non-contiguous seed domain and six additional
appends, covering BF16, native NVFP4, native FP8, and the existing diagnostic Q5
projection route. The seed payload/count is immutable while raw index/position
state advances, with intentionally distinct RoPE coordinates. Existing selected
attention tests independently cover nonuniform 2051-entry domains and exact
zero/tail/count boundaries. These compositional tests are not a captured
2051-token full-target MTP rollout.

### Startup graph and request isolation

The original bounded schedule (now the test-only paged fixture adapter) preallocates a stable
draft embedding input and captures its width-one frozen-domain GPU computation at construction.
Control staging, seed snapshot and host frontier advancement
remain outside capture. Ordinary draft calls launch the graph; traced calls use the same extracted
compute schedule eagerly. Neither route changes the source recurrence, cache format or sampling
policy. The existing rejection/replay checks now compare graph draft outputs against the traced
eager outputs exactly. Four independently owned request states sharing one immutable weight owner
are additionally interleaved in order `[3,1,0,2]` for two draft steps; every other request's live state
remains unchanged, and all final carries, K/V/index/position caches and frozen domains match
independent eager requests exactly. This is graph correctness/isolation evidence, not a fused C=4
kernel or a measured throughput claim. Target-aligned extend remains eager.

Focused commands (existing container/environment):

```sh
NINFER_QWEN4_MTP=/models/qwen4-mtp /build/tests/ninfer_qwen4_mtp_test --native-real --trace /src/out/qwen4-mtp-native-trace.json
python3.11 -m tools.reference.qwen4.mtp --source /ssdpool2nvme/local_llm/models/qwen4-mtp/qwen4-mtp-nvfp4-source.safetensors --trace out/qwen4-mtp-native-trace.json
```

### Actual full-target hidden, shared endpoints, and accepted target replay

`tests/targets/qwen4/test_mtp_target_inputs.cpp` consumes the bounded capture emitted by the
separate staged DFlash diagnostic. The full 48-layer UD-IQ1_S verifier captures 24 sequential
four-stream `State::residual()` rows and their shifted next-token embeddings, plus eight real
verification-input rows. It drains/unloads before this test loads the native MTP private block.
The test materializes **only** the same diagnostic artifact's original Q4_K embedding and head
objects on GPU, without repacking, CPU matrix execution, PLE loading or other weight streaming.
This protects shared-endpoint identity; it is not a future Qwen4 storage recommendation.

The fixed prompt's initial anchor is teacher-provided token 348. Captured next-token IDs,
decoded embedding words and original positions are checked exactly. The native W4A16 private
block consumes the entire 24-row seed, then predicts three greedy proposals with two recurrent
draft calls. Both calls preserve the selected seed domain; traced eager outputs and subsequent
startup-graph replay agree exactly. An isolated common sampler probe uses temperature 0.6,
p-less, the unchanged epsilon/support floor, and seed 1939 on each complete head output;
occurrence-count updates are exact. These stochastic probes do not replace the greedy draft
proposal policy or publish target outputs.

The natural DFlash verification round licensed `[830,248046]`, so the MTP caller can additionally
discard drafts and replay actual target-hidden prefixes 0, 1 and 2. Row `j` is paired with
embedding of licensed token `j` at source position `24+j`. Live cache, positions, selected
domain and replay output match a fresh target-aligned MTP schedule exactly. These target rows
are valid represented inputs, but are not verification of the MTP proposals. No MTP acceptance
rate, generation-quality, PPL, product scheduling or throughput claim follows.

On RTX 5090 / CUDA 13.1, the focused integration returned zero failures; the default CTest
correctly skips without explicit artifact/capture environments. The unchanged independent
`tools/reference/qwen4/mtp.py` oracle checked all public boundaries and actual preceding GPU
QSA state for the 24-row seed and two draft calls: **zero local Op or state failures**. Largest
QSA relative L2 was 1.100%; maximum per-row MoE relative L2 was 0.373%. Every injection store
was exactly nearest-even BF16 of ideal FP64 arithmetic. The independent propagated 2% diagnostic
still fails one seed carry at 2.052% relative L2; the original threshold remains unchanged.
Local correctness and propagated sensitivity are reported separately.

Focused commands, after the staged full-target capture has completed:

```sh
NINFER_QWEN4_WEIGHTS=/models/qwen4_ud_iq1_s_verify.ninfer NINFER_QWEN4_MTP=/models/qwen4-mtp NINFER_QWEN4_MTP_TARGET_OUTPUT=/src/out/qwen4-dflash-target /build/tests/ninfer_qwen4_mtp_target_inputs_test
python3.11 -m tools.reference.qwen4.mtp --source /ssdpool2nvme/local_llm/models/qwen4-mtp/qwen4-mtp-nvfp4-source.safetensors --trace out/qwen4-dflash-target/qwen4-mtp-target-trace.json
```

## Native paged compute migration qualification

The native runtime now uses one compact batched `MtpProgram` over borrowed BF16 paged QSA
views, not a complete private-cache program for each slot. `NativeDraftRuntime` owns the shared
page pool, carry/frozen-domain images, source-target carry for reseeding, and startup graphs.
The existing bounded mathematical fixtures use an explicitly test-owned one-page adapter to
preserve their trace schema; their old per-request harness is not the product state owner.

After this migration, the real five-row seed/four-step fixture passed exact frozen-domain,
graph/eager, rejection/replay, target-replay counts 0…4, and whole/chunk checks. The actual
24-row target-hidden panel was rerun through paged execution, with proposals `[830,271,2]`,
zero state/replay failures, and a fresh represented-input trace at
`out/qwen4-mtp-paged.dEP04j/qwen4-mtp-target-trace.json`. The unchanged independent FP64 oracle
again reported zero local Op/state failures; historical represented cache words, all new appends,
positions, selected IDs/counts, and exact BF16 injection stores passed. The separate propagated
2% sensitivity screen still has one failed seed carry comparison. This repeats numerical
qualification of the migrated compute path; shared-pool request ownership is tested separately.

## Primary source addresses

- https://arxiv.org/html/2608.30320v1
- https://github.com/QwenLM/Qwen3.8-Flash-Next/blob/69885871a64393807d988b27b1b5e380e8f28526/README.md
- https://huggingface.co/nvidia/Qwen3.8-Flash-Next-NVFP4/tree/fc694b54fb0174e0913e6adf86691ef85a4ead47
- https://huggingface.co/limpincat/flashnext-drafters/tree/39d7d235eb4748cd90d3ae575a2a2e54b49018c9/mtp-nvfp4
- https://github.com/vllm-project/vllm/blob/a1bf8ac12d9f1537ff2d233f5ab3d1346fd8bd44/vllm/models/qwen4_exp/nvidia/mtp.py
- https://github.com/vllm-project/vllm/blob/a1bf8ac12d9f1537ff2d233f5ab3d1346fd8bd44/vllm/models/qwen4_exp/nvidia/indexer_qsa.py
- https://github.com/vllm-project/vllm/blob/a1bf8ac12d9f1537ff2d233f5ab3d1346fd8bd44/vllm/models/qwen4_exp/nvidia/qsa.py
- https://github.com/vllm-project/vllm/blob/a1bf8ac12d9f1537ff2d233f5ab3d1346fd8bd44/vllm/v1/spec_decode/llm_base_proposer.py
- https://github.com/vllm-project/vllm/blob/a1bf8ac12d9f1537ff2d233f5ab3d1346fd8bd44/vllm/v1/worker/gpu/spec_decode/mtp/speculator.py
- https://github.com/sgl-project/sglang/blob/5e4b94b134e4dcc9a10fac465f94cdf576268646/python/sglang/srt/models/qwen4_exp_mtp.py
- https://github.com/sgl-project/sglang/blob/5e4b94b134e4dcc9a10fac465f94cdf576268646/python/sglang/srt/models/qwen4_exp.py
- https://github.com/sgl-project/sglang/blob/5e4b94b134e4dcc9a10fac465f94cdf576268646/python/sglang/srt/layers/attention/qwen_sparse_attn_backend.py
- https://github.com/sgl-project/sglang/blob/5e4b94b134e4dcc9a10fac465f94cdf576268646/python/sglang/srt/layers/attention/qsa/qsa_indexer.py
- https://github.com/lightseekorg/tokenspeed/blob/e09336bbfdc6dadf497907e8d4d139812381e7a7/python/tokenspeed/runtime/layers/attention/qsa/indexer.py
- https://github.com/lightseekorg/tokenspeed/blob/e09336bbfdc6dadf497907e8d4d139812381e7a7/python/tokenspeed/runtime/layers/attention/backends/paged/qsa.py
