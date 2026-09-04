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
not that the PLE table is copied to VRAM. The table is read through an owned mapping, bounded pinned
staging, and asynchronous H2D gathers as specified in section 3. This does not assume every future
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
Program accepts only numeric token/target ids and exposes logits, NLL, and state/routing diagnostics
for the opt-in real-artifact test; it has no registry, Engine, CLI, serving, MTP, Vision, batching,
or CUDA-Graph route. It is not the eventual family runtime or an exact product-SKU leaf.

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

The current artifact layer can place tensors only on device or validate them; its retained-host
path accepts resources, not tensors. Qwen4 therefore extends the generic artifact contract with an
owned mapped-tensor placement. `LoadedModel` owns that mapping and the bounded host-gather executor
until every host gather and CUDA transfer has drained; their lifetime is not tied to Reader
construction. This placement is limited to the exact PLE table declared by the target artifact
authority; it is not a general CPU-weight or fallback execution lane. The target owns deterministic
PLE row requests and prefetch timing; artifact/core own bounded raw gather and transfer mechanisms.
The concrete execution profile is:

- derive and deduplicate the 16 row ids per token while retaining output order;
- gather rows from the artifact-owned mapping/file into a fixed, preallocated pinned staging ring,
  with explicit cold-page advice/read-ahead rather than pinning the full table;
- batch row requests across the compact C=1..8 execution unit and scatter gathered values back to
  token/head order;
- overlap prompt PLE fetch with layer-0 compute and schedule next-token fetch at the preceding
  round boundary with three distinct completions: `host_fill_done` follows the CPU gather and page
  faults and must precede `cudaMemcpyAsync`; `copy_done` follows H2D and permits reuse of the pinned
  host slot; `consumer_done` follows the last graph/kernel read and alone permits reuse of the
  graph-stable device staging slot (or of a coupled host/device ring slot);
- make cancellation and model teardown drain every launched host fill, transfer, and consumer before
  releasing their respective slots, without mutating committed PLE token/conv state; and
- report cold and warm page-fault/read bytes, gather latency, H2D bytes, and overlap effectiveness.

Do not copy the PLE into pinned RAM, encode it as an opaque resource, or use an external sidecar.
Exact FP8 PLE storage also requires a registered E4M3FN-plus-scale format/layout and an exact
decode oracle; it cannot be inferred from a framework bug or silently cast. Phase 0 requires the
artifact storage, host mapping/page-cache budget, bounded pinned staging, and measured cold/warm
decode/TTFT to be compatible with the product. Failure of that gate stops activation; it does not
move PLE to VRAM or enable general CPU execution.

### 3.1 UD-IQ1_S host-staged verification profile

This diagnostic profile is deliberately narrower than product activation. The exact public GGUF
inventory splits as follows:

| Placement | Tensor set | Encoded bytes |
|---|---|---:|
| host mapped | PLE | 28,800,138,240 |
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
call. No whole-round graph claim is made.

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
weights; independently confirm its artifact-owned host mapping, page-cache envelope, and bounded
pinned staging fit host capacity and meet the cold/warm latency gate.
For each homogeneous pool, require the target-derived candidate `B(M)` to include all coupled
pools/fixed allocations and satisfy the current `M_min <= M <= M_max` shared-capacity admission
rules; do not budget or advertise `C * max_context`. If PLE is host-resident, also require artifact
storage, host mapping/page-cache budget, bounded pinned staging, and measured cold/warm latency from
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
prefill, T=1 continuation, request reset, and C=1..8 isolation.

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
at C=1..8.

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
  preview-style PLE cannot also satisfy its mapped-host capacity and latency contract;
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
- correct eager and graph prefill/decode for C=1..8;
- transactional QSA/GDN/MTP state and, when present, PLE state, prefix retention, and restore;
- Text, Vision, score, generate, CLI, OpenAI, and Anthropic behavior through the public Engine;
- passing real-artifact parity, paired per-token PPL, long-context retrieval, and all applicable
  multimodal and MTP qualification;
- measured 5090 capacity and end-to-end performance under the advertised codec/context; and
- the focused checks plus the full C++ unit-test suite passing.

## 12. Primary source addresses

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
