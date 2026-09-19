# Qwen4 Architecture Model Reference

This reference freezes the Qwen4 model mathematics and persistent state established from the
official `Qwen/Qwen3.8-Flash-Next` BF16 preview. The preview is source provenance for this Qwen4
architecture authority, not the repository identity. The exact native artifact identity is
`qwen4/native-preview` / `nvfp4-native`. Its Engine integration is architecture implementation,
not a claim that the oversized preview can run on the RTX 5090 or an advertised fitting model.

No currently audited preview profile qualifies as a runnable resident RTX 5090 target. The official BF16
tensor payload is 359,999,963,128 bytes. Excluding the entire 51,233,085,475-value PLE component
still leaves 128,766,895,984 checkpoint values; even an impossible uniform four-bit encoding of all
of them would occupy 64,383,447,992 bytes (59.96 GiB) before scales, state, KV, workspaces, CUDA
Graphs, and the required 1 GiB headroom. The audited UD-IQ1_S profile still has 43,735,298,560
non-PLE tensor bytes (40.73 GiB). The canonical native mixed-weight inventory requires
80,369,063,936 GPU payload bytes before runtime state and therefore fails device admission.
Native admission never selects a fallback product lane. The selected UD-IQ1_S artifact has one
unregistered C=1 eager numeric-token Program with T=1 decode and T=1..4096 chunked prefill solely
for native architecture verification. A
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
specify rollout and transaction behavior. Section 11 records the shared recurrence, the explicit
vLLM/TokenSpeed frozen-domain qualification profile, and SGLang's differing later-step tail.
Bounded private-block correctness does not establish full-target speculative admission.

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

The complete native BF16 preview Vision schedule and source pixel preparation are implemented
as an unregistered qualification route in `src/targets/qwen4/vision.{h,cpp}` and
`vision_frontend.{h,cpp}`. Biased projections use the closed central `linear_bias` Op, with no
intermediate BF16 cast before bias. The source inventory, numerical qualification, caller
lifetime contract, and remaining integration boundary are recorded in
`docs/research/qwen4-native-vision.md`. This route does not register a full Qwen4 Engine target.

Text-only positions use four equal rows initially: one causal-mask row and three MRoPE rows. For
multimodal input, the frontend constructs the causal text-position row plus temporal, height, and
width MRoPE rows. Text QSA uses the three MRoPE rows for both core Q/K and index Q/block K; the
separate text row determines causal visibility. All four rows, not merely a scalar decode cursor,
are observable continuation state.

## 11. MTP: native private block and explicit execution profile

The checkpoint contains one private QSA/MoE decoder layer, separate attention and MoE GR modules,
a separate final read-only GR, `fc_embedding`, `fc_hidden`, and two pre-projection norms. It shares
the main token embedding and untied output head. It contains no PLE tensors. The exact 31 private
tensors are inventoried in the artifact reference.

Pinned vLLM and the independently maintained SGLang integration agree on this stem and
four-stream recurrence. For target token-aligned embedding `e` and carried four-stream hidden `R`:

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

Here target alignment means **`R_t` paired with `embedding(x_{t+1})`**, while the private
RoPE/cache row retains source position `t`; its output predicts `x_{t+2}`. It does not
pair `R_t` with `x_t`, shift RoPE to `t+1`, or substitute the collapsed final-GR read.
For a prompt `x[0:N]`, seed rows use hidden `R[0:N]`, shifted embeddings
`x[1:N] + [next_anchor]`, and original positions `0..N-1`. The next anchor may be
teacher-provided in a declared diagnostic; otherwise it is selected by the target sampler.
Subsequent private draft calls advance their independent source positions by one and consume
the previous private carry plus the preceding proposal's embedding. After a verified input
prefix of length `L`, target replay pairs verified hidden row `j` with licensed output token
`j` at original verified position `N+j`; the final pair uses the correction/bonus new anchor.
Only those `L` verified inputs are retained, not the correction/bonus as an additional input.
The current pinned vLLM generic proposer and GPU autoregressive preparer independently perform
this left token shift while preserving target positions; exact references are in the MTP
execution research note.

`pre_fc_norm_hidden` is not the branch-grouped GR norm: it takes one variance over the complete
10240-wide concatenated state. Reshaping into four streams occurs only after that normalization.

The target model likewise retains its pre-final-read four-stream state for draft step zero while
ordinary target logits consume the final GR read. Later draft steps carry the prior draft layer's
pre-final-read state. Both implementations select QSA indices for the target-aligned draft-extend
row. They **do not agree on later-step tails**, including at the original pins. NInfer's bounded
native private-block implementation explicitly chooses the audited **vLLM + TokenSpeed frozen
complete-domain profile**: later steps consume the exact seed token IDs/count, while still
appending their private core K/V, raw index keys, and independent RoPE positions. They do not
append newly drafted tokens to the attention domain. SGLang instead appends the post-capture
draft interval. This selection is an execution profile, not a claim of training-time authority or
SGLang parity. Target verification always computes its own indices. The maximum attended count
is 2051 (2048 complete-block slots plus three seed-tail slots); vLLM's extra packed count column
does not increase that to 2052 tokens. Source pins and formulas are in
`docs/research/qwen4-mtp-execution.md`.

`LoadedMtp` binds the independently acquired `qwen4/native-mtp-qualification` private artifact:
three NVFP4 expert banks, plus 29 source BF16 protected tensors. Every BF16 word matches the
pinned NVIDIA source; every expert code, block scale and multiplier is independently checked.
The publisher explicitly marks expert input scales as placeholders, so `MtpProgram` enforces
**A16Only**, with no A4 policy switch. The canonical native artifact instead supplies the same
already-prepared semantic weight view from its one owner. `MtpProgram` is one compact batched
compute/scratch owner over externally owned BF16 paged QSA views and device controls; it owns no
request frontier, private cache allocation, carry, or graph. `NativeDraftRuntime` owns one shared
exclusive page pool, per-slot carry/domain/target-hidden images, fixed compact staging, and
startup graph shapes for C=1…4. There is no serialized complete per-slot MTP program. Integer
staging and carry/domain gathers occur outside capture; captured execution is the same batched
GPU schedule as the eager numerical path. Historical bounded C4 interleaving evidence below
concerns the earlier qualification harness, not proof of the new shared-pool owner.

The separate actual-component `test_native_draft_runtime.cpp` now checks the new owner with
native private weights and authentic shared BF16 endpoints: ragged compact C4 graph/eager
logits and every live paged/cache/seed word agree exactly, including frozen-domain discard,
retained last-key restoration after reseeding, and unbind/rebind. Independent Prompt and
Retained images at the same frontier but different anchor seeds each restore their exact
last cache row and complete seed state. Fresh whole-prefix versus
one-row reseeding has relative L2 0.00430…0.01449 within the unchanged 2% component-composition
criterion; this different-shape comparison is not asserted bit-identical. Local Op correctness
remains qualified separately against the independent represented-input oracle.

`NativeDraftRuntime::extend_mtp` consumes target-aligned hidden rows; `draft_mtp` consumes previous
private pre-final-GR carry. `discard_mtp` restores the seed carry/domain/frontier. Accepted rows
are replayed through `extend_mtp` using their
**target** hidden states, never committed from draft hidden states. Physical provisional rows
above the restored logical frontier are invisible and are overwritten by retained-row replay.

Real private-weight tests execute a five-row seed and four draft steps with distinct logical
cache/RoPE coordinates, immutable selection reuse, exact rejection/replay, and accepted counts
0–4. Independent FP64 formulas check each Op from its actual represented public input, including
the preceding GPU-recorded QSA cache/domain. Old transformed K is not normalized or rotated again;
historical rows are checked exactly and new appends independently. A second,
fully propagated oracle is a sensitivity diagnostic: its strict 2% chain screen fails at later
steps despite passing local Op criteria, including one changed top-10 expert at step 3. This is
not full-model numerical/PPL or acceptance-rate proof.

An additional staged diagnostic captures 24 actual full-target pre-final-GR rows, unloads that
target, and runs native MTP with its exact original shared Q4_K embedding/head fully resident on
GPU. This is diagnostic-checkpoint provenance, not a native NVFP4 full target or proposed future
weight recipe. The source-aligned seed, two frozen-domain draft steps, exact eager/graph replay,
common p-less/epsilon head sampling, and replay of actual verified target rows for retained
prefixes 0/1/2 pass. Those verified rows were captured from a DFlash proposal round; they test
MTP target-hidden replay, not acceptance of MTP proposals. Initial anchor is teacher-provided.
All represented-input FP64 local Op/state gates pass on these actual inputs. The independent
propagated 2% diagnostic retains one failed seed carry comparison (2.052% relative L2); no
threshold was changed. Registered integration, MTP acceptance quality, full-model PPL and
throughput still need a fitting exact supported target.

### 11.1 Exact PixelML DFlash companion

The acquired PixelML companion is a five-layer DFlash v1 model, not DFlash2 and not native MTP.
It consumes the attention-GR block inputs at main layers `[4,16,24,36,44]`, concatenated in that
order. These correspond to the publisher's previous-layer tap labels `[3,15,23,35,43]`;
arithmetic branch means or raw four-stream residuals are not interchangeable features.

`LoadedDFlash` binds the exact 58-tensor source BF16 or separately converted NVFP4/A16 artifact.
`DFlashProgram` borrows five GPU BF16 accepted-context paged views and owns only compact scratch
for one to four slots. `NativeDraftRuntime` owns their single shared exclusive page pool,
reservation/frontiers/checkpoints and startup draft graphs. The fused/normalized context is shared across draft layers; only the noise-query stream
receives each layer's input norm. All live queries attend accepted context strictly before the
anchor plus every live noise query. Noise K/V are never persisted. Anchor then six mask embeddings
produce seven predictions, starting at query zero. RoPE uses full 256-wide split-half rotation
with theta `1e7`, independently from QSA's interleaved MRoPE. Logical cache positions and anchor
RoPE positions are explicit distinct inputs at the component boundary. The exact native PixelML
schedule uses causal token ordinals for both context and anchor, not the target's three-axis
MRoPE coordinates or `rope_delta`, as required by its pinned training source. The target supplies the shared BF16 embedding/head;
there is no learned codebook or selector in this checkpoint.

The full-D256 rotary implementation uses FP64 frequency/phase/trigonometry with FP32 rotation
coefficients to preserve its qualified BF16 output profile through source position 262143.
Near-ceiling paired Q/K and single-K cases compare against the existing independent FP64 RoPE
formula and unchanged pair-norm error criterion. They reproduced key/output errors with FP32
phase evaluation and pass with the corrected profile; D128 routes are unchanged.

The actual native shared-owner integration also checks C4 graph/eager logits and all five live
cache layers exactly, proposal non-mutation, shorter Prompt-checkpoint restoration, independent
C1 correspondence, and shared P64 entitlement exhaustion/release. Its captured target-feature
panel and native BF16 endpoints are component evidence, not full native target PPL or a claim
that the separately measured NVFP4 drafter quantization loss is acceptable.

Source-specific capture, BF16/NVFP4 A16 execution, shared full-vocabulary endpoint, accepted
context append, C=4 isolation and CUDA Graph replay have bounded independent qualification.
Converted NVFP4-versus-BF16 draft quality, real-target acceptance and full speculative transaction
admission are separate gates. `docs/research/qwen4-speculative-sources.md` records source pins,
complete formulas and retained failed ideal-chain/RMS screening results.

A separate staged diagnostic run now exercises the complete existing 48-layer verification
target, the native NVFP4/A16 drafter, that diagnostic target's actual Q4_K embedding/head, and
the unchanged shared p-less/epsilon accept Op. Target and drafter lifetimes do not overlap.
One natural seven-proposal round accepted one draft; cancellation, proper-prefix publication,
full licensed publication, and correction continuation matched fresh causal execution bitwise
over all 157,147,144 bytes of GDN/QSA/PLE/residual state and exact occurrence counts. This is
tool-only reset/replay qualification, not registered native speculative scheduling or an
acceptance-rate benchmark. On separate actual target feature panels, converted NVFP4 versus
source BF16 drafter top-1 agreement was 16/21 with 33.6–36.2% hidden relative-L2 drift; these
different-weight results leave converted draft quality unadmitted despite correct kernels and
state plumbing. The research reference records both experiments and their distinct endpoint
representations and numerical criteria.

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

### 12.1 Verification Program prefill transaction

The unregistered verification Program accepts one dense C=1 token chunk of width 1..4096. It
computes all n-gram rows sequentially from the existing two-token history, broadcasts the decoded
embedding into four residual branches, and runs each of the 48 layers over the full token panel.
QSA appends all current K/V/index rows first, but every query receives a distinct causal CSR slice
ending at its own logical token, so later appended rows are never visible. GDN and PLE consume their
panels in causal order and leave the same final represented state as any legal partition of the same
token sequence under their declared Op/state criteria.

PLE row gathering and sparse-expert grouping are CPU integer/raw-byte operations required by the
host-resident artifact placement; all decode and floating-point model transformations execute on
the GPU. The PLE panel transfer overlaps decoder layer 0. Each MoE layer receives all route ids once,
stages each unique gate/up expert once in ascending expert groups, and preserves rank-order mixture
accumulation per token. Persistent buffers and both streams are startup-owned; no Op allocates while
executing a chunk.

Ordinary prefill computes the vocabulary head only for the final column and returns that logits
view plus the final hidden column and all PLE row ids. Teacher-forced scoring remains the T=1 path,
where a target id is paired with every input. A call validates token/frontier capacity before
enqueue. After enqueue, state is mutated in place; the Program publishes the new frontier and host
PLE history only after both streams synchronize successfully. A failure after enqueue poisons the
Program until `reset`, which drains both streams before restoring all state.

### 12.2 Owned native first-block qualification runtime

`LoadedNativeFirstBlock` binds actual NVIDIA layers 0–3 with source NVFP4 expert banks,
protected BF16 matrices, and the offline control/head-layout preparation defined in the artifact
reference. `NativeFirstBlock` runs exactly GDN0, PLE1/GDN1, GDN2 and QSA3, with each layer's
attention and MoE GR transitions. It never repeats those weights as a claimed 48-layer model.
Its four-stream inputs and decoded PLE embeddings are represented GPU values supplied by the
caller; token embedding, n-gram table/hash ownership, final GR/head and sampling remain separate.

One to four startup-fixed slots share immutable resident weights and ordered scratch but own
independent committed/provisional GDN and PLE state, BF16 QSA cache, controls, output hidden and
layer-boundary four-stream views (not DFlash attention-GR feature taps). `prepare` validates/stages input and logical/RoPE metadata outside
capture. Fixed-width `enqueue` has only GPU work: committed GDN/PLE state is read, provisional
state is written, and QSA appends only the logically provisional suffix. Visibility uses stable
capacity-bounded buffers and dynamic causal offsets, so advancing a frontier does not require a
different captured pointer/extent. Source metadata copies are not inside a CUDA Graph.

Widths 2–16 optionally expose all three GDN replay records and PLE convolution records. The
transaction owner folds only accepted input columns before publishing the accepted frontier;
rejected physical QSA suffix rows remain invisible. Full-width baseline commit explicitly copies
provisional GDN/PLE state. Retention restores the complete committed state before restoring its
logical frontier. `NativeFirstBlock` owns no token-publication or sampling decision.

The RTX 5090/CUDA 13.1 `ninfer_qwen4_native_compute_real_test` runs the actual four prepared
layers against the existing independent source-weight formulas at every layer boundary, with
the predeclared accumulated criterion `{relative_l2=.02, gross_absolute=.005,
gross_relative_to_max_reference=.02}`. It also compares a five-token panel with 4+1 continuation,
and exact C4 eager/graph output and complete state across two advancing frontiers. These focused
checks pass, including exact comparison of every interleaved C4 slot against a separate C1
owner's output and complete state. The graph test does not substitute its pairwise equality
for the source oracle.

This is a bounded native qualification runtime, not a registered Engine target, full native
48-layer execution, full-model PPL proof, or an inference throughput claim. The exact full preview
requires a separate complete inventory and capacity admission; missing layers cannot be replaced
by repetitions or ordinary-weight CPU/disk streaming under this GPU-resident profile.

### 12.3 Native first-block speculative transaction qualification

`src/targets/qwen4/transaction.{h,cpp}` owns a bounded native **first-four-layer** text
transaction around `NativeFirstBlock`: three GDN layers, layer-1 PLE, and one QSA layer.
This is not a 48-layer target, Engine registration, MTP/DFlash end-to-end rollout, or
acceptance/throughput qualification. Weights, represented input embeddings and PLE values
are supplied by the separately qualified native component owners.

One startup-fixed owner serves C=1..4 independent slots, verification width 1..16 and an
explicit cache capacity. Preparation stores `[old_anchor,drafts...]` and text T/H/W positions;
each coordinate advances by one per verified input, independently of cache addresses. Eager
execution or a typed captured transaction graph reads committed GDN/PLE state and writes
provisional state/records plus the append-only QSA tail. Captured graphs retain their exact
slot/width membership and stable output views. Capture alone cannot license a commit, stale
records cannot satisfy a new preparation, and repeated enqueue/launch/publication is rejected.

Resolution consumes the existing common `runtime::GeneratedRound` and `OutputDecision`.
It does not implement another sampler or acceptance policy. If the output decision publishes
N tokens, exactly N **verified input columns** are folded, and the Nth licensed output becomes
the new anchor. For A accepted drafts plus a correction/bonus, N=A+1 processes
`[old_anchor,drafts[:A]]`, never that correction/bonus. A shorter structured-output prefix
instead leaves its final licensed draft as the next anchor. Zero/rejected publication preserves
all committed state and logical sampling positions.

Three qualified per-layer GDN folds update FP32 recurrence and BF16 convolution history;
the exact PLE prefix Op updates its nine-column convolution and two raw token IDs, retaining
EOS unchanged. Width one copies the complete provisional GDN state and uses the represented
final PLE convolution column. Historical QSA KV/index/position rows remain unchanged; rejected
physical suffix rows stay invisible behind the committed frontier and are overwritten later.
Continuation hidden is the last accepted input's four-stream output, not a rejected tail.

Mutable occurrence counts are slot-owned. Common positive-temperature speculative acceptance
may update them provisionally; after consumer drain, the shared
`runtime::rollback_sampling_counts` removes every licensed-but-unpublished suffix occurrence
before any host publication. Its semantics were extracted unchanged from the existing Qwen3.6
runtime; that runtime calls the same helper. P-less epsilon/support-floor, penalty and
counter-based RNG laws remain in the existing central sampling/accept Ops. Seed/config and
publication policy remain caller-owned, while authoritative frontier/anchor override provisional
sampler scratch lengths/anchors at the next ingress.

One independently owned, startup-allocated retained snapshot per slot includes every GDN/PLE
state, visible QSA prefix, raw token history, continuation hidden, occurrence counts, frontier,
anchor positions and published-token ledger. Capture/restore and commit drain consumers before
host metadata becomes visible; execution allocates no buffers. Failures poison affected slots
until reset or a valid retained restore. Graph objects and borrowed views cannot outlive the
transaction/model/stream owners.

Focused real-weight qualification uses `ninfer_qwen4_native_transaction_test` with
`NINFER_QWEN4_NATIVE_LAYERS` and `NINFER_QWEN4_NATIVE_COMPUTE` set to the prepared fixtures.
It covers all counts 0..4, the width-16 full prefix, independent FP64 GDN recurrence and exact
record-tail/raw-history oracles, supplementary fresh accepted-prefix execution, retained
restoration, untouched historical QSA rows, and C=1,2,3,4 graph/eager request isolation across
advancing frontiers. A real common p-less accept invocation with repeated token IDs verifies
counts, cancellation, proper-prefix publication and logical positions; controlled logits isolate
transaction semantics rather than claim model acceptance quality. Missing fixtures skip by
default. The numerical GDN fold criterion remains relative L2 2.7e-3 with gross bound
1e-5 + 3.9e-3 times maximum reference magnitude; supplementary whole-composition comparisons
retain the existing 2% criterion. Exact state/control checks remain bitwise.

### 12.4 Native Engine control boundary

The native preview's concrete `EngineProgram` is separate from the dense-family Program:
`src/targets/qwen4/engine_program.{h,cpp}` binds the common `ConcurrentExecutor` publication
contract to `NativeRuntime`. The common scheduler, output sessions, tool grammar, p-less
epsilon/floor law, and licensed-suffix occurrence-count rollback remain shared. Native
runtime integration is not evidence that the full preview fits a 5090 or that full native
PPL has been measured.

`src/text/qwen` owns the shared tokenizer/template, prepared prompts, output semantics and
encoded-history cache. The explicit Qwen4 processor profile calls its source pixel frontend;
the dense profile retains its existing preparation behavior. The native artifact owns all six
frontend resources. The pinned NVIDIA tokenizer has **248077** valid IDs; the physical shared
head has **248320** rows. Registered sampling, proposal argmax and NLL normalization exclude
the padded head rows. The source generation configuration recommends temperature 1, top-k20,
top-p0.95; the product's common p-less override still takes precedence when enabled.

Input columns, output tokens and private seeds have distinct ownership. A licensed prefix of
length L commits exactly L verified target inputs `[old_anchor,drafts[:L-1]]`; its final
licensed output is the new, unconsumed anchor. Rejection/cancellation commits zero inputs and
rolls back every licensed occurrence count. MTP pairs each committed actual `R_t` with the
licensed next token at source position t. Exact-prefix reuse that samples a different anchor
replaces the private last seed at the same position, not the target prefix. DFlash receives only
accepted target feature rows. Neither backend defines an alternative acceptance law.

The Qwen4 target has two concrete complete checkpoint images: retained and prompt/rewrite.
They own recurrent, PLE, continuation, private-backend and frontend identity state over one
exclusive QSA page allocation. Append cancellation may restore the old retained prefix; cold
reset invalidates it before overwriting its KV rows. Prompt/rewrite capture splits prefill at
the exact requested frontier. The target has no host KV tiers or dense-family checkpoint
ladder: explicit nonempty ladder settings and host-tier options are startup errors, while
default options do not implicitly enable those capabilities. PLE's required locked-RAM table
is independent of this KV capability. Source QSA supports BF16 and NVFP4-G16 KV, not dense
Sage/tile-skip/XAttention modes.

Scoring uses the same exclusive shared page budget. Because it runs outside the generation
scheduler, the adapter first selects the minimum required LRU idle-retained victim set with
nonmutating admission queries, then evicts and reserves. A retained peer prefix cannot prevent
an otherwise admitted full-context score. The state-owner regression covers C2/context128 with
only128 pooled tokens and a retained64-token peer; failed trial admission leaves that peer intact.

`ninfer_qwen4_native_draft_runtime_test` exercises the actual native NVFP4 private MTP and DFlash
components with the authentic full BF16 shared head and exact addressed native embedding rows.
Unavailable fixture embedding rows are poisoned, not synthesized. Target carry/features come from
the separately identified diagnostic full48 capture: this is component/runtime integration, not
native-target quality or speculative acceptance evidence. C4 graph/eager logits and logical live
cache/seed state match exactly; retained seed-tail restore, draft discard, DFlash prompt restore,
request isolation, unbind/rebind and shared-page admission are checked. The supplementary MTP
whole-prefix versus one-row reseed comparison uses the existing `{.02,.005,.02}` composition
profile because those shapes select different A16 arithmetic routes (observed relative L2
0.0043–0.0145). Independent represented-input FP64 component oracles remain the mathematical
authority; same-route state transformations retain exact criteria.

Frontend source resources used for this boundary:

- `https://huggingface.co/nvidia/Qwen3.8-Flash-Next-NVFP4/resolve/fc694b54fb0174e0913e6adf86691ef85a4ead47/tokenizer_config.json`
- `https://huggingface.co/nvidia/Qwen3.8-Flash-Next-NVFP4/resolve/fc694b54fb0174e0913e6adf86691ef85a4ead47/generation_config.json`

### 12.5 Complete native runtime ownership

`NativeRuntime` binds one complete `NativeModelView` and executes all 48 distinct layers through
`enqueue_native_decoder`. GR, linear and expert operations flatten independent compact columns;
GDN, PLE and paged QSA retain explicit slot, live-prefix and frontier controls. Decode has one
layer-major batch for one to four requests, not a loop over complete per-request models.
Single-request prefill admits widths through 4096; multi-request verification admits widths
through 16. All ordinary weights and floating-point computation remain GPU-resident.
Full, single-request eager prefill retains the existing qualified chunked GDN route on its
host-proven slot; masked compact decode/verification uses device-selected recurrent state.
Both routes are covered by the shared startup workspace recipe.

The sole host model-data exception is the already eagerly populated and OS-locked PLE table.
Qualified integer n-gram addressing uses each slot's exact two-token raw history. Bounded pinned
byte gathers upload complete selected FP8 or NVFP4 row records; GPU decoding supplies represented
BF16 embeddings. FP8 table views exclude the trailing BF16 tensor multiplier from row bytes and
retain its exact bits separately. As soon as validated input IDs and accepted raw history
determine the rows, `NativePleFetch` gathers them and starts H2D plus GPU codec decode on its
own nonblocking stream, before page materialization and control/embedding work. Decoder index0
(the first layer) has no PLE dependency. Only index1 (the second layer) waits for the ready event
immediately before injection. Captured graphs retain an external event-wait node, reusing the
new event generation recorded before each launch; fetches themselves stay outside capture.
Both transfer and compute consumers drain before staging reuse or teardown, including a
prepared batch discarded without enqueue. This removes serialization; it does not guarantee
that transfer is hidden for every host-memory/workload configuration.
Token embeddings and optional source Vision features are
prepared before the compute graph, then broadcast into the four residual branches. Source image
tokens still participate in raw PLE hashing; replacing a visual embedding does not replace its
token ID. Final GR and the untied head produce physical 248320-row logits, while the shared
sampler restricts valid token IDs to 248077.

`NativeState` owns all 36 FP32 recurrent/BF16 convolution states, PLE history, carried residual,
and twelve P64 QSA layers in one exclusive shared page pool. QSA K/V, raw index keys and original
three-axis positions use the same page-group map. Each slot has committed and provisional fixed
state, plus two typed complete images: retained and prompt/rewrite. Only causal prefixes are
retained; later cache rows are unobservable. Growth reservations account for all active requests
before materialization. Prospective admission may consider an earlier restored prefix, but actual
reservation cannot truncate live state. Cancelling before enqueue returns mapped provisional
pages to the committed frontier without losing the growth entitlement.

Verification records every actual GDN and PLE input needed to fold exactly the accepted prefix.
Only accepted input columns advance raw history, continuation and state; a correction/bonus token
is the next anchor, not an extra committed input. Stream drain precedes host frontier publication.
The runtime joins these state transitions with `NativeDraftRuntime`'s independent paged backend
and exact retained seed state. Current/retained/prompt Vision feature images and placement metadata
are also owned, so a later media preparation cannot mutate a retained checkpoint. A failed GPU
transaction makes the runtime unusable until teardown rather than publishing reusable partial state.

Device arenas, pinned staging, state images and exact-batch ordinary/verification graph addresses
are fixed at startup. Uncaptured legal prefill widths execute eagerly over the same body. The
allocation recipe is shared with pre-load metadata sizing; CUDA-driver graph memory has a separate
conservative allowance checked against observed startup allocation, not a purported exact formula.
These mechanisms do not establish full-preview throughput on hardware that cannot admit its weights.

The real `ninfer_qwen4_native_compute_real_test` additionally executes the **same shared decoder
body** on actual layers 0–3. Its final represented output passes the independent accumulated source
formula criterion from section 12.2. Two advancing compact B4 rounds have exact eager/graph output
and defined real-prefix state equality. A real 65-column prefix additionally checks the chunked
GDN route against compact recurrence under the unchanged accumulated criterion; this pairwise
check supplements, not replaces, the independent component formulas. No absent layer is repeated
or fabricated. Full36-layer
state-owner tests separately qualify prefix folds, raw history, checkpoint restore and shared-page
admission for both BF16 and NVFP4 QSA storage. These are bounded numerical/integration evidence,
not full-checkpoint generation, PPL, speculative acceptance or performance measurements.

## 13. Oracle and quality requirements

This authority does not make framework parity the mathematical oracle. Independent target-private
reference code must consume represented artifact inputs and evaluate:

- exact signed-I64/unsigned-U64 n-gram addressing over I32 token ids and continuation;
- source-checkpoint grouped RMSNorm with `1 + w`, or actual-GGUF grouped RMSNorm with its
  already-folded gamma directly, followed by FP64 GR, PLE, QSA scores/attention, MoE, and Vision
  formulas;
- the complete GDN recurrence with FP32 persistent-state boundaries; and
- MTP stem, private-block recurrence and target-hidden replay under Section 11's explicit profile,
  separately from full-target speculative acceptance and quality qualification.

QSA selected ids and PLE row ids are exact. Floating-point outputs use named criteria fixed from
adversarial and real-shape error distributions before product qualification. Every eager/graph,
prefill/decode, compressed-cache, and quantized-weight route compares directly to the same oracle.

Whole-model qualification additionally requires artifact-native layer taps and per-token NLL,
paired perplexity on frozen raw-text corpora, long-context retrieval, multimodal goldens, MTP
acceptance distributions, request isolation for startup-fixed batch 1 through 4, and exact prefix
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
