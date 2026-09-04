# Qwen4 Op contract design

This document fixes the semantic boundaries and qualification design for the Ops needed by the
Qwen4 family described in `plans/qwen4-architecture.md`. Section 5 now has the live C=1/T=1
actual-artifact `gated_residual_read`, `gated_residual_read_write`, and `gated_residual_inject`
entries at the
preview geometry, Section 6 has a live exact `ngram_row_ids` entry, and Sections 2-4 have the live
C=1 QSA foundation entries `qsa_state_append`, `qsa_index_select`, and
`qsa_selected_attention`, plus the actual-artifact C=1/T=1 composite `qsa_verifier_token`.
Section 7 has the live C=1 PLE entries `ple_iq4_nl_stage_rows`,
`ple_iq4_nl_decode_rows`, and `ple_inject`, and Section 8 has the live C=1/T=1 GGML verifier profile
`gated_delta_net_layer`. Section 9 has the live C=1/T=1 actual-artifact
`qwen4_sparse_moe` verifier. Their concrete headers, CUDA implementations, and independent
qualification tests are present; remaining conceptual entries are implementation designs and are
not evidence that those Ops, kernels, target, or product routes exist. A concrete
header under `include/ninfer/ops/` is authoritative for its
represented inputs, formula, supported domain, outputs, effects, aliasing, and workspace, and any
disagreement here must be resolved.

The research profile is the official `Qwen/Qwen3.8-Flash-Next` checkpoint at revision
`de4b8e4d43b917e7706784d8bb445c9af86a3540`, interpreted by Transformers Qwen4Exp at commit
`c119ec3cc37ab69642f39cca2de4187714002b08`. A future runnable Qwen4 target must freeze its own
geometry and formats instead of inheriting preview constants implicitly.

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
`include/ninfer/ops/qsa.h`. It appends already-produced BF16 normalized/rotated core K, projected V,
raw index keys, and three-axis positions to the fixed C=1 research state. Core K/V are encoded as
NVFP4-G16. `qsa_verifier_token` is the admitted actual-artifact T=1 projection-plus-append form:
it uses separate BF16 index-query `[512,2560]` and index-key `[128,2560]` weights and FP32 norm
weights. The wider, batched conceptual entry below remains future design.

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
dtype. That cast is semantic. Apply RMSNorm multiplied directly by the converted GGUF gamma to the
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
`include/ninfer/ops/qsa.h`: the live C=1/T=1 entry accepts already normalized/rotated BF16
`q [256,24,1]`, maps 24 query heads to two NVFP4-G16 KV heads in groups of 12, decodes every
selected K/V row from state, and emits BF16 `[256,24,1]`. Its I32 selected-id operand has a
caller-known bound in `[1,2051]`; selected count is a device scalar in `[0,bound]`. The caller owns
one aligned 1,008,128-byte workspace and no pointer survives the call. The fixed
`qsa_verifier_token` composes that entry with the state append and selector, actual Q5_K
core/output projections, converted-gamma norms, partial MRoPE, and the output gate. The generalized
entry below remains a proposed future composite.

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

For selected bounds at most 64, the qualified live implementation uses one CTA per query head. At
larger bounds, grouped-query score and value tiles share each decoded NVFP4 K/V row across four
query heads, with deterministic tile-order finalization in the caller-owned workspace. This is a
private T=1 implementation profile: it does not change the listed-id formula, FP32 softmax, cache
codec, or public represented outputs.

`out` is BF16 `[H,W,C]` for the preview. Invalid token columns are exact zero. In the live T=1
verifier the per-head represented BF16 attention is multiplied by sigmoid of its corresponding
represented BF16 raw gate, then the concatenated `[6144]` vector crosses the Q5_K output matrix.
The cache write and
the complete output are the only effects. Output may not alias `x`, selected ids, weights, or cache;
the only permitted state alias is exact old/new cache alias. Workspace covers projections,
softmax/reduction scratch, page addressing, and codec staging for the complete declared envelope.

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

The implemented verifier admits exactly C=1/T=1. For `R` `[H,B]`, normalize each branch
separately with its own slice of converted GGUF gamma:

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

### 5.2 Inject formula and effects

For block result `y` `[H]`:

```text
R_out[:,i] = R[:,i] + write_scale[i] * y[:].
```

Every element is written. `x` and `write_scale` may not overlap `R` or each other. Inject permits
`R_out` to alias `R` exactly, but forbids partial overlap and any overlap with `y` or
`write_scale`.

Read entries accept caller-owned fixed-profile workspace; inject requires no global scratch. FP8
residual storage, if selected later, is a distinct represented-state profile with explicit
decode/encode boundaries and its own criterion. It is not enabled by the BF16 contract.

## 6. Exact n-gram row addressing

Implemented family: `include/ninfer/ops/ngram_embedding.h`, with exact CUDA and C=1/T=1 host
integer routes qualified by `tests/ops/test_ngram_embedding.cpp`. The host entry prepares the same
validated constants once, accepts one current token plus the oldest-to-newest two-token history,
and returns sixteen rows plus the advanced raw history. It performs no floating-point work or
embedding decode. The CUDA route remains independently exercised against the naive exact oracle;
the tests also compare every host step directly with CUDA across continuation, EOS, and reset.

The UD-IQ1_S verifier owns its two-token host history, uploads the exact row ids and advanced
history for device diagnostics/state, and starts the mapped-row gather after the layer-0 mixer is
queued. The row H2D stays on the execution stream, so decode and PLE injection remain ordered before
decoder layer 1 without a row-id D2H transfer or a PLE-specific stream synchronization.

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

Implemented family: `include/ninfer/ops/ple.h`. The admitted host-resident profile is the fixed
C=1 decode boundary: `ple_iq4_nl_stage_rows` copies exactly sixteen 90-byte IQ4_NL rows from the
mapped table into caller-owned pinned storage and enqueues one transfer into a stable device slot;
`ple_iq4_nl_decode_rows` performs exact device decode to BF16. `ple_inject` implements the complete
C=1 injection/state transition with Q8_0 key/value projections and FP32 norm/convolution controls.
Every column in its declared W is valid; the future batched composite with `valid_tokens` and exact
invalid suffixes is not implemented by this verifier entry.

### 7.1 Gather/decode boundary

The logical gather is:

```text
E[(h*Dn+d),t,c] = decode(table[row_ids[h,t,c],d]),
```

with `E [Nh*Dn,W,C]`, head-major flattening, and preview shape `[2560,W,C]`. Row ids must be valid
non-padding rows and are never deduplicated semantically. Exact table codec and scale lookup belong
to the represented table view. BF16 compares after exact BF16 decode; a future FP8/scaled format
requires a registered decoder and codec-specific criterion.

A device-resident table could expose `ngram_embedding_gather(table,row_ids,E,...)` directly. The
admitted host-resident C=1 path instead takes the sixteen exact row ids at the round boundary and
copies their encoded 90-byte spans, in head order and without semantic deduplication, from the
artifact-owned mapping into one bounded pinned slot. One H2D transfer populates the paired stable
device slot; `ple_iq4_nl_decode_rows` then performs only exact GPU decode. File I/O, page-fault
policy, staging-ring ownership, and slot completion remain outside the Op. Neither entry mutates
the table or row ids. Output and all inputs are non-overlapping. Only the mapped IQ4_NL entry needed
by the selected research artifact is currently admitted.

### 7.2 PLE injection boundary

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

The preview weights are `W_key [10240,2560]`, `W_value [2560,2560]`, three independent converted
GGUF gamma vectors `[10240]`, and mathematical depthwise `conv_weight [10240,4]`, viewed
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

## 8. Qwen4 GDN profile

Implemented first profile: `include/ninfer/ops/gated_delta_net_layer.h`. It is one semantically
closed C=1/T=1 layer entry over the converted UD-IQ1_S checkpoint storage: qkv and z are GGML
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
   `[10240,3]`, while Z bypasses convolution;
2. mathematical FP32 convolution weights `[10240,4]`, physically viewed as contiguous
   `[4,10240]` and indexed `weight[channel*4+tap]`, plus represented
   `ssm_a=-exp(A_log)` and `dt_bias`, consumed without a BF16 verifier lane;
3. ordinary learned scale (no unit offset) and `norm(y)*sigmoid(z)` independently for each
   128-wide value head, followed by output projection `[2560,6144]`; and
4. distinct or exact in-place BF16 convolution state and FP32 recurrence state.

Batched snapshot/replay/fold admission remains a future target-schedule tranche. It must reuse the
same formula and represented boundaries; a Qwen4 Program will own which distinct outputs become
committed. It is not implied by the C=1 verifier entry.

The oracle exact-decodes Q5_K/Q6_K bytes and covers projection, width-four causal convolution, Q/K
L2 normalization, control gates, every FP32 recurrence update, sigmoid output gating, and Q6_K
output projection. Repeated `T=1` continuation is compared to one sequential FP64 oracle from the
same represented initial state; a distinct-output call proves the rollback boundary. There is no
prefill entry in this verifier profile.

## 9. Live C=1/T=1 sparse-MoE verifier

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
The mapped banks are never pinned or copied in full. The entry owns no scheduling, model registry,
or Engine path.

The same semantic Op also has a device-resident verifier profile. Its routed gate and up operands
are complete rank-three IQ1_S or IQ2_XXS banks `[512,640,2560]`, and routed down is the complete
IQ4_NL bank `[512,2560,640]`. A resident-only deterministic bitonic route orders raw logits by
descending score and then ascending expert id. One two-dimensional grid computes all ten routed
gate/up pairs and their BF16 SwiGLU seam; one down/finish grid visits ranks in order, preserves the
BF16 routed-down seam, and accumulates in FP32. The shared Q5_K/Q6_K gate/up projections and SwiGLU
are fused, and a 513th router CTA computes the shared scalar gate. The profile performs no transfer,
host synchronization, event operation, or runtime repack. The staged route and kernels remain
separate and unchanged. The one-layer placement benchmark owns full device banks and reports both
a fixed hot route and rotating windows covering all 512 experts. It is not a claim that the
48-layer preview artifact fits one RTX 5090 or authority to change the live verifier's placement.

The complete closed Op, not private router or expert stages, owns the oracle. Qualification includes
all-zero router logits selecting ids `0..9` with weights exactly `0.1` in the ideal oracle, a
nonuniform softmax with a tie at the tenth boundary, distinct raw logits whose nonmaximum FP32
exponentials all underflow, positive and negative shared-gate saturation, both routed codecs, both
shared gate/up codecs, exact host/device staged bytes, and actual T=1 shapes. Its naive FP64 oracle
decodes IQ1_S/IQ2_XXS, IQ4_NL, Q5_K/Q6_K, and Q8_0 independently and evaluates the complete ideal formula.
The verifier's private BF16 projection/SwiGLU storage and FP32 routing/accumulation profile are
qualified with fixed normwise and finite gross criteria; they are not copied into the ideal oracle.
Prefill and batched C lanes remain future registered-target work rather than claims of this entry.

## 10. Schedule-owned transactions are not Ops

The unregistered `src/targets/qwen4/program.cpp` verifier now composes these live C=1 Ops into the
48-layer eager Text schedule with one exact 4096-token frontier. It resets continuation only at a
sequence boundary, refreshes the four residual branches for each numeric token, runs PLE before
zero-based layer 1, appends and selects QSA through the NVFP4-G16 cache, serializes the selected
expert route-id dependency before feeding the two-stream/two-slot staging pipeline, and finishes
with the read-only GR and untied Q4_K head. It exposes diagnostic views and GPU-computed NLL but is
not a registered target or an Engine execution path.

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

The minimum meaningful matrix is:

| Family | Required cells |
|---|---|
| QSA projection/composite | live verifier: separate BF16 512x2560/128x2560 index projections, Q5_K 12288x2560/512x2560/2560x6144 core/output projections, FP32 norms, C=1/T=1, exact selected ids, current-token NVFP4 round-trip, complete FP64 output oracle. Future registered batched entry qualifies C=4,8 and distinct-state forms |
| QSA selector | visible counts 0..5 and 2047..2053; 512-block saturation; non-contiguous visible ids; unequal C lanes; fragmented pages; multimodal positions; all-zero and boundary ties; BF16 pool/cast witnesses |
| QSA attention | live verifier: real 24/2/256 T=1 geometry; new-current NVFP4-G16 cache read; selected count 1, 2048, and 2051; nonuniform 2051-entry complete FP64 oracle; short and tiled routes. Future registered entry qualifies prefill, fragmented pages, and other cache codecs |
| GR | live verifier: real 4x2560/R=320; FP32 norm/write and Q8_0 down/up; read-only/read-write/inject; C=1/T=1; exact Q8_0 decode oracle; in-place inject. Future registered batched entry qualifies C=4,8 and graph envelopes |
| n-gram | exact vectors below; empty/short history; EOS as current and prior token; one-shot/chunk/T=1; C lane isolation; every admitted PLE-module index |
| PLE gather | first/last valid row, repeated and permuted ids, 16-head order, codec edges, direct versus staged-remap equality when both execution profiles exist |
| PLE inject | live verifier: real 4x2560 projections/state; zero and nonzero history; one-shot/legal chunks/T=1; dilation witness; in-place output; C=1. Future registered batched entry: C=4,8 isolation and invalid-suffix semantics |
| GDN | live verifier: source 16 Q/K heads expanded to 48 tiled artifact heads (`h%16`) at width 128; direct represented `ssm_a`; Q5_K and layer-2 Q6_K inputs; Q6_K output; repeated T=1; nonzero FP32 initial state; sigmoid output gate; distinct rollback and in-place continuation. Future Program routes qualify prefill and snapshot/record/fold if used |
| sparse MoE | live verifier: 512/top10/I640; lower-id all-zero and tenth-boundary ties; selected normalization; IQ1_S/IQ2_XXS two-slot host staging and complete device-resident bank profiles; IQ4_NL routed down; Q5_K/Q6_K shared gate/up; Q8_0 shared down; shared-gate extremes; exact 844,800-byte per-slot cap and two-slot lifetime; device-id bank indexing; T=1 Store result. Future registered entry qualifies batched/prefill routes |

Named numeric criteria must be fixed from adversarial and target-representative oracle-error
distributions before a failing candidate is judged. Reduction Ops require both a normwise bound and
a finite gross pointwise cap, and every criterion rejects non-finite output/state. Exact selector
ids and integer transforms have no tolerance.

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
