# Qwen3.6-27B Model Reference

This reference records the exact Text, MTP, Vision, multimodal-position, numeric, and persistent-state
semantics used by the registered target. Fixed dimensions are encoded in the target's
`impl/config.h`; artifact assignment and binding are defined in
[`qwen3.6-27b-artifact.md`](qwen3.6-27b-artifact.md).

## 1. Model identity

The target checkpoint is called Qwen3.6-27B in this project and uses the `qwen3_5` /
`qwen3_5_text` architecture names in Hugging Face implementations. Qwen3.8-27B shares this Text,
MTP, and Vision shape. It is a dense multimodal model with three runtime components:

- a 64-layer hybrid Text decoder;
- a one-layer MTP draft model;
- a 27-layer Vision transformer and patch merger.

Qwen3.8-27B NVFP4 may additionally bind an optional DFlash2 companion when `dflash/` objects are
present. Qwen3.6-27B files do not.

The implementation is fixed to this checkpoint shape. A different layer count, hidden size, head
layout, or Vision tower is a different model implementation rather than a runtime configuration.

## 2. Global dimensions

### 2.1 Text decoder

| Field | Value |
|---|---:|
| hidden size | 5120 |
| decoder layers | 64 |
| intermediate size | 17408 |
| output/embedding matrix rows | 248320 |
| tokenizer-addressable token IDs | 248077 (`0..248076`) |
| full-attention interval | 4 |
| full-attention layers | 16 |
| Gated DeltaNet layers | 48 |
| RMSNorm epsilon | `1e-6` |
| RoPE theta | `1e7` |
| checkpoint position capacity | 262144 |
| MTP layers | 1 |

Layer `i` is full attention exactly when `(i + 1) % 4 == 0`; all other layers are GDN. The model
has a dense SwiGLU MLP in every layer and does not contain MoE experts.

### 2.2 Full attention

| Field | Value |
|---|---:|
| query heads | 24 |
| KV heads | 4 |
| head dimension | 256 |
| Q width | 6144 |
| K/V width | 1024 each |
| rotated dimensions per head | 64 |
| Q heads per KV head | 6 |
| attention scale | `1/sqrt(256) = 0.0625` |

Each query projection also produces a 256-dimensional per-head output gate. Q and K use
zero-centered `(1+w)` RMSNorm before partial NeoX RoPE. The attention result is multiplied by
`sigmoid(gate)` before the output projection.

### 2.3 Gated DeltaNet

| Field | Value |
|---|---:|
| Q/K heads | 16 |
| Q/K head dimension | 128 |
| V heads | 48 |
| V head dimension | 128 |
| Q width | 2048 |
| K width | 2048 |
| V width | 6144 |
| Z output-gate width | 6144 |
| A/B gate width | 48 each |
| causal convolution width | 4 |
| recurrent state dtype | FP32 |
| delta-rule scale | `1/sqrt(128)` |

Every group of three V heads shares one Q/K head. GDN state size is fixed per layer and does not grow
with context length.

### 2.4 Vision

| Field | Value |
|---|---:|
| transformer depth | 27 |
| hidden size | 1152 |
| intermediate size | 4304 |
| attention heads | 16 |
| head dimension | 72 |
| spatial patch | 16 × 16 |
| temporal patch | 2 frames |
| flattened patch width | `3 × 2 × 16 × 16 = 1536` |
| spatial merge | 2 × 2 patches |
| merger input width | 4608 |
| merger output width | 5120 |
| position table | 48 × 48 |
| Vision RoPE theta | 10000 |

### 2.5 Runtime extents

`T` denotes the Text/MTP token extent supplied to an Op. It is any positive tensor extent that
fits the applicable storage or explicit state capacity. Decode (`T=1`), verification-sized calls,
and prefill chunks are workload points and private implementation routes, not different Op domains.
The configured prefill chunk controls target workspace and request decomposition; its default 4096
does not cap an Op's `T`.

Vision uses different axes. `P` is the aggregate raw-patch count and must be a positive multiple of
4 because of the 2x2 spatial merge; `V=P/4` is the aggregate merged-token count. The registered 27B
processor/implementation envelope is `4<=P<=131072` and `1<=V<=32768`, with the frontend's media,
attention-pair, and prompt budgets imposing any additional request-specific restriction. These
Vision columns are not Text token `T` and are not expanded beyond that envelope.

### 2.6 DFlash2 companion (Qwen3.8-27B)

Qwen3.8-27B may carry an optional DFlash2 draft companion. Qwen3.6-27B files do not. The values
below are from Hugging Face `z-lab/Qwen3.8-27B-DFlash2` revision
`50307d4c4cde6860d4eee73e2547cd786fe8e8a4` and pinned `z-lab/dflash` `model.py`
`95c8aeca5e4b4c4f9c0c967c05ab89fa3ed24f4c` (`DFlash2DraftModel`). This is not the 35B DFlash v1
companion, not `z-lab/Qwen3.6-27B-DFlash`, and not a DSpark drafter.

| Field | Value |
|---|---:|
| draft hidden layers | 5, all sliding |
| hidden size | 5120 |
| dense SwiGLU intermediate size | 17408 |
| query heads | 32 |
| KV heads | 8 |
| head dimension | 128 |
| Q width | 4096 |
| K/V width | 1024 each |
| attention scale | `1/sqrt(128)` |
| SWA window | 2048 |
| SWA inequality | `abs(q−k) < 2048` (both directions; `is_causal=false`) |
| RMSNorm epsilon | `1e-6` |
| RoPE theta | `1e7` (`rope_type=default`) |
| native total block length | 8 |
| parallel mask proposals per native block | 7 |
| mask token | 248070 (tokenizer-addressable) |
| target feature layers, zero-based | `5, 19, 33, 47, 61` |
| feature concat width | `5 × 5120 = 25600` |
| conv | kernel 2, group 16, two-tap dynamic |
| selector | rank 256, unsorted top-16 |
| full-context DFlash KV pool | none (`full_layers=0`) |

The companion is five dense Qwen3 decoder blocks at 27B width, not a copy of the target's hybrid
GDN/full-attention topology. It contains no embedding table, output head, Vision tower, GDN state,
or projection bias. It reuses the target input embedding and independent target `lm_head` (or the
`[131072,5120]` Q4 shortlist on the product `--lm-head-draft` path). Internal mask id 248070 is a
legal tokenizer id used only as the padded query token; it is not a sampled draft.

## 3. Shared decoder layer skeleton

Both Text layer types use a pre-norm residual structure:

```text
h = RMSNorm(x, input_norm, unit_offset=true)
x = x + mixer(h)
h = RMSNorm(x, post_attention_norm, unit_offset=true)
x = x + down_proj(SiLU(gate_proj(h)) * up_proj(h))
```

The registered `.ninfer` target stores some projection groups as fused logical views, and its
private kernels may fuse projection, activation, or residual work. Those are storage/execution
choices; the mathematical result remains the schedule above.

All ordinary layer norms and Q/K norms use zero-centered weights: the effective scale is `(1+w)`.
The GDN internal gated norm is a plain ones-initialized RMSNorm and does not use the unit offset.

## 4. Full-attention layer

For normalized input `h[5120,T]`:

```text
q, gate = q_gate_projection(h)        # [24,256,T] each
k       = k_projection(h)             # [4,256,T]
v       = v_projection(h)             # [4,256,T]

q = RMSNorm(q, q_norm, unit_offset=true)
k = RMSNorm(k, k_norm, unit_offset=true)
q, k = partial_mrope(q, k, rotary_dims=64)

a = causal_gqa(q, k, v, scale=1/sqrt(256), kv_cache)
a = a * sigmoid(gate)
x = x + o_projection(a)
```

Prefill appends all K/V columns and evaluates causal attention for the chunk. Decode appends one
column and attends over the resident prefix. KV storage may be BF16 or INT8-G64. The exact runtime
cache codec and the common ideal attention oracle are defined by the repository-internal
[`gqa_attention.h`](../../include/ninfer/ops/gqa_attention.h) contract. Both cache formats and their
optimized compute profiles are judged by that one oracle construction rather than by
implementation-mirroring references.

Text-only positions use the same scalar position for temporal, height, and width MRoPE sections.
Multimodal prefill supplies distinct three-axis positions. Only 64 of each 256-dimensional head are
rotated, divided across the model's interleaved MRoPE sections `[11,11,10]`.

For the registered RTX 5090 implementation, `attn_input_proj` consumes the physical
`query_key [7168,5120]` Q4 parent and `gate_value [7168,5120]` Q5 parent directly. At
`T=1..16`, one Q4 projection writes Q/K through a split epilogue and one Q5 projection writes
Gate/V through a split epilogue, so a full-attention layer issues exactly two input-projection
kernels. The `T>=17` route likewise evaluates the two homogeneous parents with grouped MMA
launches. This is an implementation profile; Q, Gate, K, and V remain the four logical Op outputs.

## 5. Gated DeltaNet layer

The normalized input produces Q, K, V, Z, and per-V-head A/B controls:

```text
q = in_q(h)       # [16,128,T]
k = in_k(h)       # [16,128,T]
v = in_v(h)       # [48,128,T]
z = in_z(h)       # [48,128,T]
a = in_a(h)       # [48,T]
b = in_b(h)       # [48,T]
```

Q/K/V are concatenated into `[10240,T]` and passed through the depthwise causal width-4 convolution
and SiLU. Q and K are then L2-normalized per head with epsilon `1e-6`. The production GDN Op always
receives raw BF16 convolution outputs. Its recurrent implementation keeps normalized values in FP32
registers; when it selects the chunked implementation, it privately materializes normalized BF16
q/k for the chunked body and recurrent tail. The decay and update controls `g` and `beta` are
observable FP32 values with the logical formula:

```text
g    = -exp(A_log) * softplus(a + dt_bias)
beta = sigmoid(b)
```

For `T=1..16`, the registered `gdn_input_proj` implementation keeps QK and V as two independent
projections but gives each launcher a row slice of the final `[10240,T]` tensor with leading
dimension 10240. It therefore allocates no concat workspace and performs no input-projection D2D
copy. The grouped `T>=17` route is unchanged. Public `linear` output remains contiguous-only; the
pitched row slices are private to this Op implementation.

Ordinary width-one decode invokes the shared `gdn_input_proj_conv_snapshot` contract with the
initial and destination selectors both set to the lane's current slot. This is an in-place state
update and does not retain a speculative trajectory. MTP target verification instead invokes
`gdn_input_proj_conv_record`: the Q4/Q5 leaf combines projection, causal convolution, SiLU, direct
q/k/v placement, and publication of the represented convolution column to the Program-owned
ReplaySSM record row while leaving persistent state unchanged. The recurrent stage likewise emits
raw key/value/gate records; after output resolution, one all-layer Fold applies only the committed
record prefix to the lane's current state.

The eliminated qkv intermediate is not a semantic cast boundary; each exact route uses its
directly oracle-qualified private precision and staging. The separate Z projection remains in the
target output-gate leaf.

For V head `j`, let `q` and `k` come from Q/K head `j // 3`. With recurrent state
`S[128,128]`, one token performs:

```text
k  = k / ||k||
q  = q / ||q||
S  = exp(g) * S
Sk = S @ k
u  = beta * (v - Sk)
S  = S + u outer k
o  = (S @ q) * (1/sqrt(128))
```

The CUDA recurrence uses an algebraically equivalent ordering appropriate to the kernel. Prefill
uses chunked parallel state passing for large T and recurrent/small-T paths where appropriate;
ordinary decode uses the width-one in-place path, while MTP verification records T=W transitions
for Fold and overlays T=1 snapshot arithmetic on scratch SSM for packed `out`, so each verify
column matches ordinary decode. Fold applies the same finite-precision recurrent transition.

The per-head output is normalized and gated before projection:

```text
on = gated_rmsnorm(o, gdn_norm, z)    # RMSNorm(o) * SiLU(z)
x  = x + out_projection(on)
```

For `C=max_concurrency`, the Program reserves `2C` complete all-layer GDN state slots when speculation
is off, and `2C+1` when MTP or DFlash is on:

- `[0,C)` is the current committed convolution history and FP32 recurrent state for each lane;
- `[C,2C)` is the corresponding turn checkpoint used by thinking-aware prefix reuse;
- slot `2C` (MTP or DFlash) is Engine-wide GDN storage: the hot turn-rollback occupant, borrowed by
  prefill context-checkpoint freeze (then reloaded). It is not a rewrite slot. Staging hidden is a
  separate `[5120,1]` BF16 tensor, not `[5120,2C]`. DFlash2 checkpoint heads snapshot cyclic through
  a matching 1-lane Engine-wide staging window (D2D live→staging on compute, D2H from staging on
  `copy_stream`) so suffix prefill can mutate live local; restore writes the host image back and
  sets `dflash_context_frontier` to `F`.

When MTP is enabled, a separate Program-owned ReplaySSM arena holds `C` physical record rows of
width `draft_window+1` for every GDN layer. Records are pending-round scratch, not sequence state or
additional checkpoint slots.

## 6. Text prefill and decode

### Prefill

1. gather quantized token embeddings into `[5120,T]`;
2. replace placeholder columns with Vision merger output when input is multimodal;
3. run the 64 decoder layers in aligned chunks while carrying KV/GDN state;
4. retain the hidden columns required by MTP preparation;
5. apply final `(1+w)` RMSNorm to the last required column;
6. evaluate the full `lm_head[248320,5120]`;
7. select the first generated token with greedy or configured sampling.

Chunking limits workspace. It does not reset positions or state between chunks.

### Ordinary decode

1. embed the current token;
2. run all 64 layers for one position;
3. apply final norm and full `lm_head`;
4. sample the next token;
5. commit updated KV/GDN state and position.

The eager and CUDA Graph record/replay paths execute the same model schedule.

## 7. MTP draft model

The checkpoint contains one MTP decoder layer. It is a small draft model conditioned on both the
target hidden state and the token that follows it; it is not another `lm_head` attached directly to
the main decoder.

The MTP stem is:

```text
e = RMSNorm(embed(token), pre_fc_norm_embedding, unit_offset=true)
h = RMSNorm(target_hidden, pre_fc_norm_hidden, unit_offset=true)
x = fc(concat(e, h))
attention_residual = RMSNorm(x, input_norm, unit_offset=true)
```

The single MTP layer is a full-attention layer with the same 24 Q heads, 4 KV heads, 256 head
dimension, Q/K norm, partial MRoPE, gated output, and 17408-wide SwiGLU MLP as a Text full-attention
layer. It has its own one-layer KV cache. A final MTP norm produces the draft hidden state.

MTP reuses the Text embedding and full `lm_head` semantics; the checkpoint does not contain a tied
MTP embedding/head pair. At proposal sites the runtime may instead use the optional
`[131072,5120]` Q4 shortlisted head and remap its row index to a real vocabulary id. Target
verification always uses the full `lm_head`.

## 8. DFlash2 block-diffusion draft model

DFlash2 replaces repeated autoregressive draft steps with one masked-block forward. Prefill
captures the five target residuals after completing layers `5, 19, 33, 47, 61` (Hugging Face
`hidden_states[layer_id+1]`), projects them with `fc` + `hidden_norm`, and writes cyclic K/V for
all five draft layers. Decode appends only newly committed target features. Rejected query K/V is
not context. There is no growing DFlash Full pool.

For a multimodal prompt, capture occurs after token embeddings and Vision merger embeddings have
been composed, so the companion receives the same selected target residuals as text-only prefill.
The companion keeps absolute positions for its own context/proposal model. Target verification has
a separate position panel and applies the request's saved `rope_delta`; cache writes remain at
absolute token positions while Text RoPE follows the post-Vision MRoPE sequence.

One propose block:

1. Query rows are the anchor embedding plus seven MASK embeddings (id **248070**) at positions
   `E .. E+7`. `input_embedding_scale` is 1.0.
2. For each of the five layers, from pinned `Qwen3DFlashDecoderLayer`:
   - `h = RMSNorm(residual, input_norm)`
   - `h, attn_k1 = attention_conv.prepare(h)`
   - Q = `q_proj(h)` then RMSNorm per head; K/V = `cat(proj(target_hidden), proj(h))` then K
     RMSNorm; V unnormalized
   - RoPE covers context-concat + query; Q uses the last `q_len` of cos/sin; K uses the full
     concat. Cached context K is stored already RoPE'd.
   - Symmetric SWA-2048 GQA, scale `1/sqrt(128)`, cyclic capacity 2048
   - `o_proj` then `attention_conv.finish` on the 5120-d residual stream; residual add
   - `h = RMSNorm(residual, post_attention_norm)`
   - `h, mlp_k1 = mlp_conv.prepare(h)`; SiLU-GLU MLP; `mlp_conv.finish`; residual add
3. Final RMSNorm → draft-head logits on the seven mask columns (not the anchor).
   Concurrent C>1 runs each compact row as a C=1-shaped propose (`T=width`, `B=1`) so those
   Linears, SWA, and the draft head use the sequential kernels rather than a `T=width*B`
   specialization. Eager execution also resolves SWA's direct/split route from that row's
   frontier rather than the batch maximum; graph replay retains the fixed profile envelope.
4. Path selector (`dflash2_path_select`): unsorted top-16 of those logits, then the Markov score
   `score = unary + ⟨pred_code(prev) ⊙ W_h h_t , succ_code(cand)⟩`. Greedy chooses the maximum.
   Sampling draws from the temperature-scaled 16-way distribution and retains that row as `q`,
   except under p-less: p-less temperature is a target-distribution parameter, so the selector
   stays greedy / one-hot `q` (same chain convention as MTP). Chain accept also ignores any
   recorded 16-way `q` under p-less and uses that one-hot convention, so a stale or temperature-2
   softmax shortlist cannot inflate `p/q` and lock onto copied n-grams.
   Selector RNG is keyed by request seed and absolute token position, independent of compact batch
   row. `--lm-head-draft` runs top-16 on the shortlist and gathers codebooks by token id.
5. The 27B target verifies the packed tree (product W=12 at k=7) or chain (W=k+1) in one
   causal forward for the compact batch. Concurrent C>1 keeps that forward packed (`B=batch`)
   so CUDA graphs capture one 27B verify rather than a serial host loop. Residual Linear and
   GDN-control normally panel at the C=1 width (`packed_route_tokens`); the qualified W=5
   C=2..4 residual Linear routes instead use one panel-bit-exact A16 T=10/15/20 launch; T=20
   retains the T=5 panel reduction profile. Fused
   NVFP4/BF16-control attention input and NVFP4 SwiGLU make the same qualified exception. GQA
   launches one B=1 decode per sequence so it does not take `MultiBatch=true`. ReplaySSM records
   are `layer(g, 0, B)`.
   Packed GDN conv-record keeps the T=1 reduction and BF16 history boundary: B=1 uses the fused
   T=1 GEMV+FP32 conv route, while B=2..4 uses one request-indexed SmallT launch. Packed GDN
   recurrent overlays T=1 snapshot `out` on scratch SSM. Greedy accepts the matching prefix.
   Truncated sampling uses Leviathan `min(1,p/q)` on every hop. Under p-less, hop 0 is
   Leviathan with one-hot `q` (tree: SpecInfer membership); later hops and the bonus are greedy
   argmax. ReplaySSM Fold commits the corresponding sequential prefix. The RTX 5090
   recommendation is k=4 (W=5, one SmallT GQA tile). Native k=7 uses W=8 chain or W=12 packed
   tree; chain W=8 on 24 Q heads is T>6, so B=1 causal verify uses the Prompt GQA route over
   the full visible KV. The maximum k=11 route performs two MASK blocks (7+4) before one W=12
   chain verify.

`GroupedDynamicCausalConv` is grouped size-16, kernel 2, left-padded (causal along the query
block): `prepare` before the sublayer on the pre-norm hidden, `finish` on that sublayer's output.

## 9. Speculative round semantics

For `k` configured draft tokens, the runtime prepares a candidate window, runs target verification,
and accepts only the prefix licensed by the target distribution.

In greedy mode, MTP and DFlash2 accept the longest draft prefix matching the target argmax. In
sampling mode both use chain rejection sampling against the represented target distribution. A bad
draft therefore reduces acceptance and throughput; it must not change the distribution of emitted
target tokens. Under p-less, DFlash2 chain accept uses the same one-hot `q` convention as MTP
(ignore any 16-way selector `q`) **only at hop 0**: p-less temperature is not a draft softmax, so
`p/q` from a 16-way shortlist would over-accept copied n-grams. Later hops, and the bonus after a
full accept, are greedy (accept iff the draft equals that packed column's p-less argmax). Packed
tree verify is the same split: hop 0 is SpecInfer membership from p-less(`P_LLM`); later hops walk
only the argmax child. DFlash2 differs by producing the whole candidate chain in one masked-block
forward. Packed GDN conv-record uses a T=1-reduction, BF16-history SmallT launch at B=2..4; packed
GDN recurrent overlays ordinary T=1 snapshot arithmetic on scratch SSM so those packed logits
match width-one decode. Fold still consumes the T=W records.

Target verification writes candidate KV into provisioned but unpublished extents. After the final
per-row output prefix is known, one all-layer Fold commits the accepted sequential prefix into the
lane's current state. The transaction trims rejected KV, commits continuation hidden and MTP or
DFlash cyclic state, and only then advances the authoritative frontier and publishes output. Near
context capacity, the Engine falls back to the one-token target path when a complete safe round
does not fit.

## 10. Vision preprocessing

The native processor accepts structured text/image/video message parts. For each media item it:

1. consumes media bytes already acquired by the CLI or serving layer;
2. decodes the image or samples video frames;
3. chooses dimensions aligned to the 32-pixel merge factor;
4. bicubic-resizes and normalizes RGB values;
5. packs channel-major `2 × 16 × 16` temporal-spatial patches into FP32 rows of width 1536;
6. expands the chat-template placeholders and records the token spans;
7. constructs Vision grids, timestamps, token types, and three-axis text positions;
8. computes `rope_delta` for subsequent Text decode positions.

Images repeat a frame to form the temporal pair. Videos are sampled at the configured rate and
packed in temporal pairs. Source bytes, decoded pixels, media count, raw patches, and Vision-token
budgets reject oversized media work before Vision execution. The computed attention-pair count is
diagnostic rather than an admission limit. The processor does not impose a separate prompt-token
ceiling; Engine `max_context` admits the complete rendered text-plus-media prompt.

## 11. Vision tower

Patch rows are converted to BF16 and projected into `[1152,P]`. The bilinearly interpolated learned
48×48 position value is cast to BF16 before the BF16 residual add, matching the checkpoint's
explicit position-embedding cast boundary.

Each of the 27 transformer blocks performs:

```text
h = LayerNorm(x)
q, k, v = qkv_projection(h) + bias
q, k = vision_rope(q, k, 2D patch positions)
a = segmented_flash_attention(q, k, v, cu_seqlens)
x = x + projection(a) + bias

h = LayerNorm(x)
h = GELU_tanh(fc1(h) + bias)
x = x + fc2(h) + bias
```

Attention is segmented by image or video frame grid using cumulative sequence lengths; unrelated
media/frame segments do not attend to each other.

The merger applies LayerNorm, groups each spatial 2×2 patch block into width 4608, then computes:

```text
visual = fc2(GELU_exact(fc1(merged) + bias)) + bias    # [5120,V]
```

The result replaces the matching placeholder embedding columns before Text prefill. Vision state is
not retained for autoregressive decode.

## 12. Multimodal positions

The family prepared prompt's `positions` value is axis-major `[3,T]` in temporal, height, width
order.

- ordinary text tokens advance all three axes together;
- image/video placeholder tokens receive positions derived from the merged Vision grid;
- following text resumes after the maximum multimodal position;
- `rope_delta = next_multimodal_position - token_count` converts later scalar decode indices into
  the correct MRoPE position.

The Text RoPE kernel consumes these positions during multimodal prefill and uses the saved
`rope_delta` during decode. This is why Vision can disappear after prefill while Text positions
remain consistent.

## 13. Precision and oracle boundaries

- activations are BF16 at public model/operator boundaries;
- ordinary and Q/K norm oracles evaluate their reductions in FP32/FP64 and compare the declared
  BF16 outputs; production reduction and staging are route-private choices;
- GDN `g`, `beta`, and recurrent state are FP32;
- the ideal GQA oracle evaluates dot products, stable softmax, and value reduction in FP64 from
  BF16 Q and logical cache values; the BF16 Op output is promoted to FP64 for comparison;
- low-bit weight storage changes representation, not the intended dequantized matrix;
- INT8-G64 KV stores FP16 scales and signed codes, and its ideal logical K/V values are their FP32
  decode;
- the target's INT8 attention path intentionally quantizes Q to Q8-G64 for production computation;
  this native compute profile does not replace BF16 Q in the common ideal oracle, and its delta is
  accepted through the separate named INT8-cache compute-profile criterion;
- the full target `lm_head` is used for prefill, verification, and ordinary decode regardless of
  draft-head mode.

These points define public representations and mathematical oracles, not a mandatory kernel
operation order. Private accumulator precision, Tensor Core operand staging, intermediate
materialization, workspace dtype, and reduction association are selected by each implementation
route and accepted against the Op's criterion for that implementation profile.

GQA numerical qualification covers both registered geometries, supported prompt and small-T
regimes, the maintained conformance matrix, and target-representative activation ranges. Its
BF16-cache and INT8-cache compute-profile criteria are explicitly named in the GQA conformance
suite; they are not claimed as pointwise bounds for every arbitrary or adversarial BF16 tensor. A1
append-and-attend and A3 cached-only attention are each checked directly against the common ideal
oracle. Equality between those different numerical paths is not a contract or acceptance test.

## 14. State inventory

Let `C=max_concurrency`.

| State | Shape basis | Lifetime |
|---|---|---|
| Text GQA KV | 16 layers × context × 4 heads × 256 | active sequence |
| MTP KV | 1 layer × context × 4 heads × 256 | active sequence when MTP enabled |
| GDN convolution history | 48 layers × 10240 × 3 × `2C` BF16, plus one staging slot when MTP or DFlash is on | Program lifetime; current, turn-checkpoint, and checkpoint staging slots |
| GDN recurrent matrices | 48 layers × 48 heads × 128 × 128 × `2C` FP32, plus one staging slot when MTP or DFlash is on | Program lifetime; current, turn-checkpoint, and checkpoint staging slots |
| ReplaySSM records | 48 layers × `C` rows × `dflash_verify_width` (`draft_window+1` for chain verify) convolution/key/value/gate columns | Program lifetime when MTP or DFlash enabled; one pending round |
| DFlash2 local K/V | current and rewrite: 5 layers × 2048 × 8 heads × 128 × 2 planes × `C` lanes; plus one 1-lane checkpoint staging window | Program lifetime when DFlash enabled |
| DFlash2 target features | prefill `[25600,P]` plus pending `[25600,dflash_verify_width,C]` BF16 | Program lifetime when DFlash enabled |
| Continuation hidden | current and turn-checkpoint `[5120,C]` BF16 stores; MTP/DFlash staging `[5120,1]` | Program lifetime |
| Text step buffers | token, positions, logits, verify/draft/sampling tensors | Program lifetime |
| Program scratch | Text/MTP/Vision phase temporaries | one phase in the shared workspace arena |
| Vision request transient | encoded Vision output `[8192,V]` | active prefix during request begin |

KV memory grows with configured context. The fixed GDN state pool depends on `C` and, when MTP or
DFlash is on, one extra staging slot; it is independent of the speculative window. Enabling MTP
or DFlash adds the separate ReplaySSM arena, whose capacity is `C*dflash_verify_width` record columns
per GDN layer. DFlash2 adds five cyclic windows of capacity 2048
(current and rewrite, `C` lanes each) plus one Engine-wide 1-lane staging window for context-checkpoint
D2H, and does not allocate a growing
DFlash Full pool.

The Program freezes its feature set and memory plan at startup. The Qwen3.6 family builds named
Text-prefill, ordinary-round, MTP-prefill, MTP-round, and Vision phase capacities from the
configured execution domains and reserves the maximum as one pure scratch arena. Sequential
phases and scoped child Ops reuse it. Prefill allocations use
`min(prefill_chunk,max_context)`; Vision is bounded by both the registered frontend geometry and
`max_context`.

Vision encoded output is not part of that arena: its separate request-transient allocation is
reserved at startup and only an active prefix is exposed to a request. A zero MTP draft window has no MTP weight view, MTP KV cache, or optimized proposal head. A DFlash
Engine without `dflash/` objects fails at bind. With Vision disabled, it has no
Vision weight view, Vision scratch phase, or request-transient allocation; media is rejected by the
matching Frontend. CUDA Graph driver allowance is budgeted separately from both arenas. The
complete artifact inventory is still validated before these resident views are published.

## 15. Implementation map

| Model concern | Source |
|---|---|
| exact dimensions/layer counts and family hybrid-layer mapping | `src/targets/qwen3_6_27b/impl/config.h`, `src/targets/qwen3_6/export/ninfer/targets/qwen3_6/hybrid_topology.h` |
| immutable Text/MTP/Vision/DFlash2 bindings | `src/targets/qwen3_6_27b/impl/load/` |
| split attention projection, staged GDN projection/control, dense post-mixer leaves, leaf workspace, and graph frontier ranges | `src/targets/qwen3_6_27b/impl/variant.h`, `impl/variant.cpp` |
| Text/MTP/Vision execution, planning, Program lifecycle, workspace composition, prefix/state transactions, and graph mechanics | `src/targets/qwen3_6/impl/runtime/` |
| tokenizer, template, multimodal processing, output decoder | `src/targets/qwen3_6/impl/frontend/` |
| DFlash2 SWA-2048, grouped dynamic conv, and path selector | `include/ninfer/ops/swa.h`, `include/ninfer/ops/grouped_dynamic_conv.h`, `include/ninfer/ops/dflash2_path_select.h` |
| Qwen3.8 NVFP4 DFlash2 conversion | [`qwen3.8-27b-artifact.md`](qwen3.8-27b-artifact.md), `tools/convert/qwen3_8_27b/convert_nvfp4.py` |
| growing GQA paged cache pools, allocations, and per-layer views | `src/core/paged_kv_cache.*` |
| GDN layout/views/reset/copy and Text/MTP/GDN composition | `src/targets/qwen3_6/export/ninfer/targets/qwen3_6/decoder_state.h`, `src/targets/qwen3_6/impl/state/decoder_state.cpp` |
| fixed all-layer GDN state pool, ReplaySSM record arena, and Fold contract | `src/core/linear_attention_state.*`, `src/core/gdn_replay_records.*`, `include/ninfer/ops/gdn_replay.h`, `src/ops/linear_attention/gated_delta_net/replay.cpp` |
| generated-round buffer schema, MTP alignment, and Vision control | `src/targets/qwen3_6/export/ninfer/targets/qwen3_6/`, `src/targets/qwen3_6/impl/state/round_state.cpp`, `src/targets/qwen3_6/impl/vision/control.cpp` |
| `.ninfer` tensor assignment and binding | [`qwen3.6-27b-artifact.md`](qwen3.6-27b-artifact.md), `tools/reference/qwen3_6_27b/bindings.py` |
| native `.ninfer` converter and verifier | `tools/convert/qwen3_6_27b` |
| artifact-native Python Text/Vision/MTP reference | `tools/reference/qwen3_6_27b` |

The Python reference is an independent executable implementation for model/artifact inspection and
diagnosis; it is not the per-Op mathematical oracle, does not prescribe private C++ kernel
precision, and does not define cross-runtime generated-token equality. Each Op is checked against
its own naive FP32/FP64 or exact oracle. The C++ target
implements the complete Text/Vision/MTP product over `.ninfer` through the closed Engine
architecture.

## 16. Family runtime and Variant structure

The 27B (`src/targets/qwen3_6_27b`) and 35B-A3B (`src/targets/qwen3_6_35b_a3b`) execution packages
are peer compile-time Variants of one identity-free Qwen3.6 family runtime
(`src/targets/qwen3_6`).

The family owns the shared `SequencePlan<Variant>`, `RequestPlan<Variant>`, and `Program<Variant>`
algorithms; frontend and output semantics; Text/Vision/speculative schedules; state transactions;
workspace composition; and CUDA Graph capture/replay mechanics. Each package separately owns its
registered artifact identities and bindings, immutable model view, dimensions/storage facts, three
closed execution-leaf families (attention projection, GDN projection/control, post-mixer), graph
frontier data, and Program instance bytes.

Invariants: no mutable state or device allocation is shared between Programs; neither package is
defined as a delta from the other; and there is no runtime family selection or target-dependent
branch inside family scheduling. All artifacts embed the same six frontend resources, and a
prepared prompt carries no exact-target tag.
