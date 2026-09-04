# Qwen4 Architecture Model Reference

This reference freezes the Qwen4 model mathematics and persistent state established from the
official `Qwen/Qwen3.8-Flash-Next` BF16 preview. The preview is source provenance for this Qwen4
architecture authority, not the repository identity. This is **not** a registered NInfer target,
an advertised model, or permission to weaken the one-resident-model RTX 5090 product contract.

No currently audited preview profile qualifies as a registered resident RTX 5090 target. The official BF16
tensor payload is 359,999,963,128 bytes. Excluding the entire 51,233,085,475-value PLE component
still leaves 128,766,895,984 checkpoint values; even an impossible uniform four-bit encoding of all
of them would occupy 64,383,447,992 bytes (59.96 GiB) before scales, state, KV, workspaces, CUDA
Graphs, and the required 1 GiB headroom. The audited UD-IQ1_S profile still has 43,735,298,560
non-PLE tensor bytes (40.73 GiB). No registered target, `.ninfer` product weights profile, or
fallback product lane is defined by this document. The selected UD-IQ1_S artifact has one
unregistered C=1 eager numeric-token Program solely for native architecture verification. A
complete custom profile converted from the
pinned BF16 source is conditionally eligible only after it fits one 32 GB RTX 5090 and passes Phase
0 of the implementation plan, or after a separately authorized change to the product contract.

The corresponding exact BF16 source inventory is
[`qwen4-artifact.md`](qwen4-artifact.md).

## 1. Pinned authorities and precedence

| Authority | Fixed revision | Role |
|---|---|---|
| Qwen architecture report | arXiv `2608.30320v1`, 2026-08-31 | Design equations and reported architecture rationale |
| official preview repository | `QwenLM/Qwen3.8-Flash-Next` commit `69885871a64393807d988b27b1b5e380e8f28526` | Official statement that this model previews Qwen4 |
| official BF16 checkpoint | `Qwen/Qwen3.8-Flash-Next` revision `de4b8e4d43b917e7706784d8bb445c9af86a3540` | Exact dimensions, weights, buffers, and frontend resources |
| Transformers Qwen4Exp | `huggingface/transformers` commit `c119ec3cc37ab69642f39cca2de4187714002b08` | Checkpoint-consuming Text/Vision formulas and eager cache behavior |
| vLLM Qwen4Exp | `vllm-project/vllm` commit `d6bce42983bc0b2095ad6422dbf1399e219ae572` | Independent serving implementation and MTP candidate semantics |
| SGLang Qwen4Exp | PR 36497 head `78c5024e9d9f589dcb4deb7f4ba4fb23f7e85385` | Independent MTP/QSA-reuse cross-check; open integration, not a released authority |

For the preview, checkpoint configuration and represented weights settle exact dimensions and
stored values. Pinned Transformers settles ordinary checkpoint-consuming Text/Vision execution
where the paper is silent. The paper settles the intended logical equations where code contains
only optimized decompositions. A future product target must freeze its own exact checkpoint facts;
none of the constants here is inherited merely because it is called Qwen4.

MTP is different: Transformers deliberately ignores `mtp.*`, and the paper does not completely
specify rollout and transaction behavior. Section 11 records only the structure on which the two
independent serving implementations agree. That structure remains a candidate contract until
golden traces settle every listed boundary.

## 2. Exact topology

| Field | Preview value |
|---|---:|
| Text hidden width | 2560 |
| decoder layers | 48 |
| layer pattern | 12 repetitions of GDN, GDN, GDN, QSA |
| QSA decoder indices, zero-based | `3,7,11,15,19,23,27,31,35,39,43,47` |
| vocabulary matrix rows | 248320 |
| token embedding / output head | independent, untied |
| residual branches | 4 |
| concatenated residual width | 10240 |
| GR low rank | 320 |
| routed experts / selected experts | 512 / 10 |
| routed/shared SwiGLU width | 640 / 640 |
| RMSNorm epsilon | `1e-6` |
| native position capacity | 262144 |
| Text RoPE theta | `1e7` |
| partial rotated width | 64 of 256 Q/K dimensions |
| MRoPE sections | interleaved `[11,11,10]` frequency pairs |
| PLE placement | one-based layer 2, zero-based decoder index 1 |
| MTP private layers | one QSA decoder layer |

The checkpoint's `layer_types` calls the QSA entries `full_attention`. Pinned Transformers
normalizes those entries to `qwen_sparse_attention`; they are not dense-attention layers in this
checkpoint. The optional one-million-token serving recipe uses static YaRN. Native checkpoint
behavior is 262144 positions with default RoPE; YaRN is never enabled implicitly and is not part of
this research profile.

The Text embedding is duplicated into four branches before decoder layer 0. PLE adds a four-branch
value before the attention GR read at decoder index 1. Every decoder layer then performs a distinct
attention/GDN GR read-write followed by a distinct MoE GR read-write. A final read-only GR reduces
the four branches to one 2560-wide stream for the output head.

## 3. Norms and represented arithmetic boundaries

For a vector `x` and stored checkpoint weight `w`, zero-centered RMSNorm is

```text
offset_rmsnorm(x,w) = (1 + w) * x / sqrt(mean(x^2) + 1e-6)
```

GR and PLE norms over concatenated residual streams take the mean independently over each
contiguous 2560-wide branch. The MTP `pre_fc_norm_hidden` exception takes one mean over its entire
10240-wide concatenated input before reshaping. QSA Q/K norms, QSA indexer Q/K norms, every GR
norm, the PLE norms, both MTP stem norms, and the final GR norm are zero-centered.

The GDN internal output norm is the sole plain Text/MTP RMSNorm:

```text
plain_rmsnorm(x,w) = w * x / sqrt(mean(x^2) + 1e-6)
```

Pinned Transformers accumulates both norm variances in FP32 and casts the normalized, weighted
result back to the represented input dtype. Its QSA attention and MoE router softmax also use FP32.
These are cross-check implementation profiles, not automatically observable semantic casts. The
independent oracle evaluates logical softmax and floating-point formulas in FP64 from represented
inputs. Explicit boundaries that are semantic include stored BF16 values, FP32 GDN recurrence
state, the QSA FP32 block mean followed by a cast to the raw-key representation, and exact I64 PLE
addressing.

## 4. Gated Residual

Let `R[j]` be branch `j`, `Rhat[j] = offset_rmsnorm(R[j],w_norm[j])`, and `vec` concatenate the
four branches in branch-major order. Each attention/GDN or MoE GR module computes:

```text
u = SiLU(W_down vec(Rhat) / 4)
G = reshape(sigmoid(W_up u), [4,2560])
x = sum_j G[j] * Rhat[j] / 4
s = 2 * sigmoid(W_write vec(Rhat) / 4)       # four scalars
R'[j] = R[j] + s[j] * F(x)
```

`W_down` is `[320,10240]`, `W_up` is `[10240,320]`, and `W_write` is `[4,10240]`.
There is no branch-mixing matrix, static read term, or static write term. The two divisions by four,
the branch mean, per-branch normalization, sigmoid read gate, and factor-two write gate are model
math.

The final GR has the same norm/read path and no `W_write`; it returns only `x`. The checkpoint
therefore contains three tensors for each final read and four for each decoder sublayer read-write.
GR branch values are current-token activations, not context-growing persistent cache. FP8 branch
storage discussed by the paper is an implementation profile that requires direct oracle
qualification; this BF16 source reference does not make it semantic.

## 5. Gated DeltaNet

There are 36 GDN layers. Each has 16 Q heads, 16 K heads, 48 V heads, Q/K/V head width 128, and a
causal depthwise convolution of width four. Every Q/K head is shared by three V heads.

For a 2560-wide GR read `x`, the bias-free projections are:

```text
qkv = W_qkv x   # [10240] = q[16,128] | k[16,128] | v[48,128]
z   = W_z x     # [48,128]
a   = W_a x     # [48]
b   = W_b x     # [48]
```

Only `qkv` passes through the independent channel-wise causal convolution and SiLU. Q and K are
then L2-normalized with epsilon `1e-6`; Q/K are repeated three times across V heads. For each head,
with FP32 persistent state `S[t-1]` shaped `[128,128]`:

```text
beta[t]  = sigmoid(b[t])
g[t]     = -exp(A_log) * softplus(a[t] + dt_bias)
alpha[t] = exp(g[t])
q'[t]    = l2norm(q[t]) / sqrt(128)
k'[t]    = l2norm(k[t])

S_decay  = alpha[t] * S[t-1]
e[t]     = v[t] - transpose(S_decay) * k'[t]
S[t]     = S_decay + beta[t] * outer(k'[t], e[t])
y[t]     = transpose(S[t]) * q'[t]
```

The `1/sqrt(128)` query factor is present in both pinned Transformers recurrence routes although
it is not displayed in the paper's recurrence equations. The 48 values of `A_log` and `dt_bias`
are stored as BF16 in this checkpoint; conversion to FP32 for the gate formula does not change
their represented source values.

That statement describes the pinned BF16 source. The selected external GGUF instead stores the
folded decay `ssm_a=-exp(A_log)` in FP32 and reorders all V-side tensors from grouped to tiled head
order. Its represented value head `h` therefore consumes Q/K head `h%16`. Applicable
zero-centered norm weights elsewhere in the GGUF are likewise stored as already-folded gamma.

Each 128-wide `y` is plain-RMS-normalized, multiplied by `sigmoid(z)` in FP32, cast to the output
representation, concatenated to width 6144, and projected to width 2560. GDN owns two independent
persistent records per request/layer:

- the last three raw projected Q/K/V columns, shape `[10240,3]`, in the selected activation
  representation; and
- recurrence state `[48,128,128]`, semantically FP32.

One-shot prefill, arbitrarily partitioned prefill, and repeated `T=1` execution must produce the
same logical state and outputs under the declared numerical criterion.

## 6. Qwen Sparse Attention

### 6.1 Core attention

There are 12 QSA layers with 24 query heads, two KV heads, and head width 256. Each KV head serves
12 query heads. The bias-free projections are:

```text
q_gate = W_q x                              # [24,512]
q, gate = split(q_gate, [24,256], [24,256])
k = W_k x                                  # [2,256]
v = W_v x                                  # [2,256]
q = offset_rmsnorm(q); k = offset_rmsnorm(k)
q,k = partial_interleaved_mrope(q,k, rotated_width=64)
a = causal_selected_gqa(q,k,v, scale=1/sqrt(256))
y = W_o (a * sigmoid(gate))
```

The gate half is neither normalized nor rotated. Cached core K is the normalized, rotated K;
cached V is the projected V. Q, raw K, and gate are transient.

For rotary frequency pair `i in 0..31`, `inv_freq[i] = 1e7^(-2*i/64)`. Axis selection follows
`T,H,W,T,H,W,...,T,H`: temporal owns pair indices `0,3,...,30`, height owns `1,4,...,31`, and
width owns `2,5,...,29`. The chosen axis position multiplies `inv_freq[i]`; the resulting 32 phases
are duplicated across the two 32-wide halves before the usual rotate-half operation. Remaining
head dimensions `64..255` pass through unchanged.

### 6.2 Index projection, compression, and selection

The same pre-attention GR read `x` is independently projected by `[640,2560]` into four query heads
and one shared key head of width 128. Index queries are zero-centered-RMS-normalized and receive
the query token's partial 64-wide MRoPE. Raw index keys are cached before normalization and before
RoPE.

For a request and query, start from its ordered request-local visible-token index list `V`. This is
important for padding, packed requests, and nontrivial position ids: a block is not defined merely
as `floor(absolute_position/4)`.

```text
B = floor(len(V) / 4)
block[b] = V[4*b : 4*b+4]                  # complete blocks only
k_pool[b] = cast_raw_key(mean_fp32(raw_k[block[b]]))
k_block[b] = partial_mrope(offset_rmsnorm(k_pool[b]), position(block[b][0]))
score[b] = sum_h ReLU(dot(q_index[h], k_block[b])) / sqrt(128)
chosen = highest min(512,B) complete blocks
indices = flatten(block[chosen]) followed by V[4*B:]
```

Thus the complete-block token budget is 2048, the complete-block budget is 512, and the always
included incomplete causal tail has zero to three tokens. Output storage has 2051 slots padded by
invalid ids. The paper's `ceil(K/r)` and the checkpoint consumer's integer `K/r` coincide here
because `2048` is divisible by four. The `/sqrt(128)` score factor is present in pinned
Transformers but omitted from paper Eq. 15; it does not change top-k except for represented
rounding/tie effects.

NInfer's exact selector contract resolves equal scores by lower request-local logical block id.
Pinned `torch.topk` does not promise that order, so upstream results at exact ties are diagnostic,
not the deterministic oracle. Selected ids, complete-block causality, tail ids, and invalid padding
are exact outputs; score values use a numerical criterion.

### 6.3 Persistent QSA state

For every QSA layer/request, the logical persistent state is:

- core K/V for every committed token, two heads × 256 values each;
- raw index key for every committed token, one head × 128 values;
- all three MRoPE position rows required to rotate block starts and future queries; and
- the committed logical token frontier and request-local visibility/segmentation identity.

An optimized implementation may replace old raw keys with sealed normalized/rotated four-token
block keys while retaining the zero-to-three-token raw tail, but only if direct tests prove the
FP32 mean, raw-key cast, normalization, block-start MRoPE, selection, and transaction result. Such
a representation is not licensed to average already-normalized or already-rotated keys.

## 7. Sparse MoE

Every one of the 48 Text layers, and the candidate MTP layer, has 512 routed experts and one shared
expert. All are 640-wide bias-free SwiGLUs:

```text
router_logits = W_router x                         # [512]
p = ideal_softmax(router_logits)
ids = top_10(p), with lower expert id winning ties
w = p[ids] / sum(p[ids])
routed = sum_i w[i] * W_down[ids[i]](
             SiLU(W_gate[ids[i]] x) * W_up[ids[i]] x)
shared = sigmoid(W_shared_gate x) * W_shared_down(
             SiLU(W_shared_gate_proj x) * W_shared_up x)
output = routed + shared
```

There is no inference capacity factor, token dropping, stochastic routing, or auxiliary-loss term
in the forward pass. Logical expert id is identical to router row and expert-bank row. Pinned
Transformers uses an FP32 router softmax, which is a cross-check profile rather than a semantic
cast unless a future compiled profile declares one.

## 8. PLE n-gram addressing

PLE exists only on zero-based decoder layer 1 and executes before that layer's attention GR read.
It has eight bigram and eight trigram heads, each returning 160 values. Concatenation produces
2560 values.

The 16 exact head moduli are:

```text
20000003, 20000023, 20000033, 20000047,
20000059, 20000063, 20000069, 20000077,
20000081, 20000093, 20000107, 20000147,
20000153, 20000159, 20000161, 20000171
```

Their exact cumulative offsets are:

```text
0, 20000003, 40000026, 60000059,
80000106, 100000165, 120000228, 140000297,
160000374, 180000455, 200000548, 220000655,
240000802, 260000955, 280001114, 300001275
```

The valid rows end at 320001445 inclusive. The embedding is padded to `[320001536,160]` and split
in the source checkpoint into 128 consecutive `[2500012,160]` tensors. The exact stored I64 layer
multipliers are:

```text
23703573157769, 20109073645365, 8052911324071
```

These values are generated with seed 1234, `base_seed = seed + 10007 * ple_layer_index`, where the
only PLE has index zero, and unsigned 64-bit SplitMix64:

```text
z = x + 0x9E3779B97F4A7C15
z = (z xor (z >> 30)) * 0xBF58476D1CE4E5B9
z = (z xor (z >> 27)) * 0x94D049BB133111EB
z = z xor (z >> 31)
```

The complete multiplier construction is:

```text
max_long = 2^63 - 1
multiplier_max = floor(max_long / 248320)
half_bound = floor(multiplier_max / 2)
x[i] = wrap_u64(base_seed + 0x9E3779B97F4A7C15 * (i + 1))
m[i] = 2 * (splitmix64(x[i]) mod half_bound) + 1,  i = 0,1,2
```

Each head modulus is the successive prime after 19,999,999: head `h` uses the `(h+1)`th such
prime. The explicit moduli and stored buffers above remain the conversion oracle.

Multiplier generation uses unsigned-64 SplitMix wrapping. Runtime token products and XOR use I64
two's-complement wrapping, and the final `remainder` is the nonnegative Euclidean remainder of that
signed I64 value by the positive modulus. Treating a negative mixed value as an unsigned integer
before reduction would select a different row and is incorrect. For current token `t0`, previous
token `t1`, and token two positions back `t2`:

```text
bigram_mixed  = wrap(t0*m0) xor wrap(t1*m1)
trigram_mixed = bigram_mixed xor wrap(t2*m2)
row[h] = signed_euclidean_remainder(mixed, modulus[h]) + offset[h]
```

Heads `0..7` use `bigram_mixed`; heads `8..15` use `trigram_mixed`. At the start of a request and
for unavailable history after a segment boundary, missing ids are 248044. Encountering token
248044 resets following n-gram history to that same id. Padded/inactive inputs are substituted with
248044. PLE therefore owns the prior two raw token ids per request; token ids must never be
reconstructed approximately from hidden values in NInfer.

NInfer represents input ids and the two-token persistent history as I32 because every supported
token is in `[0,248320)`. Hash products are promoted to unsigned 64-bit before wrapping, XOR, signed
I64 reinterpretation, and Euclidean remainder. The upstream Python/checkpoint convention of I64
token tensors and the stored I64 multiplier buffer therefore does not make the NInfer history an
I64 state allocation.

## 9. PLE injection

Let `e` be the concatenated 2560-wide lookup and `R` the incoming four-branch state:

```text
K = reshape(offset_group_rmsnorm(W_key e), [4,2560])
V = W_value e                                      # [2560]
Q = reshape(offset_group_rmsnorm(vec(R)), [4,2560])
g[j] = dot(K[j],Q[j]) / sqrt(2560)
g[j] = sign(g[j]) * sqrt(max(abs(g[j]),1e-6))
U[j] = sigmoid(g[j]) * V
C = SiLU(depthwise_dilated_conv(offset_group_rmsnorm(vec(U))))
PLE(R,e) = vec(U) + C
R = R + PLE(R,e)
```

The depthwise convolution has 10240 channels, kernel width four, dilation three, no bias, and nine
prior activation columns of persistent history. Its state is independent of the GDN convolution
and the two-token n-gram history. PLE persistent state at this layer is therefore exactly:

- two I32 raw token ids, initialized/reset with 248044; and
- `[10240,9]` represented normalized-gated-value convolution history.

## 10. Vision and multimodal positions

The preview Vision tower has 27 blocks, width 1152, 16 heads of width 72, GELU MLP width 4304,
patch size 16 × 16, temporal patch size two, and spatial merge size two. A flattened input patch is
`[3,2,16,16]`. The learned position table is `[2304,1152]`, interpreted as a 48 × 48 grid and
bilinearly interpolated with aligned corners. The bias-bearing 3D patch projection is applied
first, then the interpolated learned position vector is added in the patch activation dtype.
Vision RoPE uses theta 10000 on the 36 frequency pairs of a 72-wide head; its rotation is computed
in FP32 and cast back to the represented Q/K dtype. Packed image/video segments use independent
non-causal attention.

Each pre-norm Vision block is:

```text
q,k,v = split(W_qkv LayerNorm(x) + b_qkv)
q,k = RoPE(q,k)
x = x + W_o attention(q,k,v, scale=1/sqrt(72)) + b_o
x = x + W_2 GELU_tanh(W_1 LayerNorm(x) + b_1) + b_2
```

All four Vision-block linears shown above have their learned bias; omitting one is not a valid
profile. Vision LayerNorm has learned weight and bias and epsilon `1e-6`. The merger normalizes each
1152-wide patch token before arranging each 2 × 2 spatial group as width 4608, then applies
`Linear(4608,4608)`, exact GELU, and `Linear(4608,2560)`, both with bias. There are no deep-stack
outputs.

Text-only positions use four equal rows initially: one causal-mask row and three MRoPE rows. For
multimodal input, the frontend constructs the causal text-position row plus temporal, height, and
width MRoPE rows. Text QSA uses the three MRoPE rows for both core Q/K and index Q/block K; the
separate text row determines causal visibility. All four rows, not merely a scalar decode cursor,
are observable continuation state.

## 11. MTP: established structure and admission gate

The checkpoint contains one private QSA/MoE decoder layer, separate attention and MoE GR modules,
a separate final read-only GR, `fc_embedding`, `fc_hidden`, and two pre-projection norms. It shares
the main token embedding and untied output head. It contains no PLE tensors. The exact 31 private
tensors are inventoried in the artifact reference.

Pinned vLLM and the independently maintained SGLang integration agree on this candidate stem and
two-stream recurrence. For target token-aligned embedding `e` and carried four-stream hidden `R`:

```text
e' = fc_embedding(offset_rmsnorm(e))                 # [2560]
Rhat_flat = offset_rmsnorm(vec(R))                   # one 10240-wide norm
Rhat = reshape(Rhat_flat,[4,2560])
R0[j] = fc_hidden(Rhat[j]) + e'                      # shared fc_hidden
R1 = one QSA/attention-GR/MoE-GR decoder layer(R0)
h_logits = final_GR_read(R1)                         # [2560]
logits = shared_output_head(h_logits)
next_carried_state = R1                              # [4,2560], before final read
```

`pre_fc_norm_hidden` is not the branch-grouped GR norm: it takes one variance over the complete
10240-wide concatenated state. Reshaping into four streams occurs only after that normalization.

The target model likewise retains its pre-final-read four-stream state for draft step zero while
ordinary target logits consume the final GR read. Later draft steps carry the prior draft layer's
pre-final-read state. Both implementations select QSA indices for the target-aligned draft-extend
row and reuse that per-request selection for later top-1 chain steps. Target verification computes
its own indices. The expanded selection width is 2051: 2048 selected complete-block token slots
plus up to three tail slots.

This agreement is sufficient to implement oracle fixtures for the stem and carried state, but not
to register MTP as correct. Before admission, golden traces from an execution environment capable
of running the source must freeze and compare:

- target pre-final-GR state and final-read logits input;
- normalized/projected embedding and each normalized/projected hidden branch;
- MTP layer pre/post attention-GR and MoE-GR states;
- all 2051 selected/tail slots at draft extend and at every reused step;
- absolute positions, core KV/index-cache alignment, and carried state for at least four steps;
- proposal probabilities/tokens and target-verification probabilities; and
- rejection, partial acceptance, full acceptance, cache fold, and published-token results.

SGLang pin `78c...` is the head of an open integration and vLLM's MTP logic is not an independent
mathematical oracle. If their traces disagree, MTP remains disabled. There is no implicit
MTP-free product variant of this checkpoint: such a variant would be a different exact artifact
and product decision.

## 12. Complete logical persistent state

For one request, the architecture requires:

```text
decode cursor and current anchor token
four MRoPE/text position rows and request-local segmentation
for each of 12 Text QSA layers:
  committed core K/V, raw index keys or proven-equivalent sealed blocks + raw tail
for each of 36 Text GDN layers:
  three-column [10240,3] QKV convolution history
  FP32 [48,128,128] recurrence state
PLE at decoder index 1:
  two I32 raw token ids
  [10240,9] dilated-convolution history
current four-stream continuation hidden where speculation/retention requires it
when MTP is admitted:
  independent MTP QSA core KV/index state and positions
  four-stream draft carried state and graph-stable reused-selection buffer
  provisional frontier and typed checkpoint/fold ledger
```

GR branches within a currently executing token are activations. Growing QSA state, fixed GDN/PLE
state, positions, and MTP provisional state are persistent owners. A prefix snapshot is invalid if
it restores only core KV: it must restore every continuation owner at one matching committed
frontier. Speculation must record QSA block/tail, GDN, PLE, and MTP effects provisionally and fold
only the accepted prefix.

## 13. Oracle and quality requirements

This authority does not make framework parity the mathematical oracle. Independent target-private
reference code must consume represented artifact inputs and evaluate:

- exact signed-I64/unsigned-U64 n-gram addressing over I32 token ids and continuation;
- FP64 grouped zero-centered RMSNorm, GR, PLE, QSA scores/attention, MoE, and Vision formulas;
- the complete GDN recurrence with FP32 persistent-state boundaries; and
- MTP stem/state/fold only after Section 11's golden evidence closes it.

QSA selected ids and PLE row ids are exact. Floating-point outputs use named criteria fixed from
adversarial and real-shape error distributions before product qualification. Every eager/graph,
prefill/decode, compressed-cache, and quantized-weight route compares directly to the same oracle.

Whole-model qualification additionally requires artifact-native layer taps and per-token NLL,
paired perplexity on frozen raw-text corpora, long-context retrieval, multimodal goldens, MTP
acceptance distributions, request isolation for startup-fixed batch 1 through 8, and exact prefix
restore. Perplexity is integration evidence and never substitutes for an Op or state-transition
oracle.

## 14. Primary source addresses

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
