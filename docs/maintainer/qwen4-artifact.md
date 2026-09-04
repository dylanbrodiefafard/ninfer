# Qwen4 Verification Artifact and Preview Source Inventory

This reference freezes the exact official BF16 preview-checkpoint inventory used to implement the
Qwen4 architecture and the selected native verification artifact. The preview checkpoint name is
source provenance, not the repository identity. There is no registered `qwen4` product identity,
weights profile, or executable target, because
neither currently audited checkpoint profile can satisfy the one-resident-model 32 GB RTX 5090
product contract. The target-private `src/targets/qwen4` verifier package binds the selected
host-staged `.ninfer`, owns its C=1 verification state, and remains deliberately unregistered. A
future complete custom profile is conditionally eligible only after it passes the implementation
plan's Phase 0 admission gate.

For an exact Qwen4 target containing this preview-style PLE table, that residency statement excludes
the table. PLE is deliberately host-resident and artifact-mapped; the GPU-residency gate applies to
the non-PLE compute core, fixed device state, KV/index pools, workspaces, graphs, and headroom.
Section 3 of the implementation plan owns the mapped-PLE execution design; this reference does not
assume that every future Qwen4 checkpoint contains PLE.

The model mathematics and state are defined in
[`qwen4-model.md`](qwen4-model.md). Generic future `.ninfer` framing,
format, and layout choices remain governed by [`artifact-container.md`](artifact-container.md),
[`tensor-formats.md`](tensor-formats.md), and [`storage-layouts.md`](storage-layouts.md). A fitting
Qwen4 target must receive a separate artifact reference that fixes its complete compiled object
order, names, formats, layouts, placement, aliases, and transforms; this document deliberately
does not invent those decisions for an unrunnable BF16 preview.

## 1. Exact source identity

| Field | Value |
|---|---|
| repository | `Qwen/Qwen3.8-Flash-Next` |
| revision | `de4b8e4d43b917e7706784d8bb445c9af86a3540` |
| model type | `qwen4_exp` |
| architecture | `Qwen4ExpForConditionalGeneration` |
| source weight shards | `model-00001-of-00131.safetensors` through `model-00131-of-00131.safetensors` |
| index | `model.safetensors.index.json` |
| index SHA-256 | `99e815241ef03325536b0aaa4441deea45174c17fae31e10f0bb456410c590de` |
| tensors | 1658 |
| represented values | 179,999,981,459 |
| tensor payload bytes | 359,999,963,128 |
| tensor dtypes | 1655 BF16 tensors, three I64 tensors |

The official rounded description of a 125B main model plus 51B n-gram embeddings is not an exact
storage inventory. The checkpoint contains 180.000B represented values after the private MTP,
Vision, embeddings/head, and padded PLE table are included.

## 2. Header-only audit method and result

Every one of the 131 shard headers was read from the pinned revision using two bounded HTTP byte
ranges: bytes `0..7` for the little-endian safetensors header length, then exactly that JSON header.
No weight payload was downloaded for the tensor inventory. For every tensor, the audit checked:

1. the name occurs once across all shard headers;
2. the header shard equals the pinned index's `weight_map` entry;
3. `data_offsets[1] - data_offsets[0]` equals `product(shape) * dtype_bytes`;
4. all 1658 index names and all 1658 header names form the same set; and
5. the sum of all tensor spans equals the index `metadata.total_size`.

The 131 JSON headers total 228,712 bytes. Individual headers range from 120 to 39,296 bytes,
contain one to 349 tensors, and describe 1,040,116,640 to 3,510,236,608 payload bytes per shard.
The exact tensor-span sum is 359,999,963,128 bytes, identical to the index total.

For reproducibility, the canonical tensor audit fingerprint is SHA-256
`0cd9fc04a2034014e7e9bbe12ac9b09441a8410f25c788cb36c20d3e3b68dddc`. It is computed over
UTF-8 records sorted by tensor name, one per line:

```text
name<TAB>dtype<TAB>comma-separated-shape<TAB>payload-bytes<TAB>shard-name<LF>
```

The corresponding canonical 131-shard summary fingerprint is SHA-256
`e82da7de15aed97d005873f57b5687bd152ae8ec0acb70150b6705ea087f94a7`, computed in shard-name
order over:

```text
shard-name<TAB>header-bytes<TAB>tensor-count<TAB>payload-bytes<LF>
```

The three small I64 payloads were additionally read by exact byte range because their values are
semantic PLE constants rather than merely inventory facts. They match Section 7.

## 2.1 External UD-IQ1_S diagnostic inventory

The Unsloth `Qwen3.8-Flash-Next-GGUF` revision
`38bb39ee97821de2c9009abb7e93950eec396e66` supplies a useful external, quant-specific Text/PLE
smoke and PPL reference. It is not a C++ runtime artifact, an independent oracle, or a registered
target. The unregistered verifier converter is the only NInfer path that accepts these GGUF shards;
it validates and byte-copies their closed tensor inventory into one `.ninfer`, which the
target-private verifier binder consumes without adding a GGUF runtime lane. Its three UD-IQ1_S
shards are:

| Shard | Bytes | SHA-256 | Tensors |
|---|---:|---|---:|
| `00001-of-00003` | 10,946,624 | `88a1420825a9304063e882ada29d438263617f51ac8923d438d927496693bafd` | 0 |
| `00002-of-00003` | 49,990,818,368 | `3a62e35bbf9add4733bd1438ebd3a67649d5edd6cb0e72bb78e33c913992b2b6` | 595 |
| `00003-of-00003` | 22,544,696,352 | `0e25ceaeb89b8a80aa973c6c0c7448943682f7408c2855b2ebd016b7643a861a` | 629 |
| **total** | **72,546,461,344** | | **1,224** |

Byte preservation includes llama.cpp's semantic conversion transforms. Applicable source
zero-centered norm weights are already stored as effective `gamma = 1 + w`; `ssm_a` is already
`-exp(A_log)`; and GDN V-side qkv rows, z/a/b controls, decay/bias/conv channels, and output columns
are reordered from source grouped-head order to GGUF tiled-head order. The verifier Ops consume
those represented values directly: they do not add another unit offset, exponentiate `ssm_a`, or
apply a second head permutation.

Despite its name, this is a heterogeneous quant. Its PLE table is
`per_layer_token_embd.weight [160,320001536]` in GGUF order, encoded as IQ4_NL with an exact
28,800,138,240-byte payload. Current llama.cpp marks that tensor lazy and gathers its selected rows
from host storage; the containing second shard also holds ordinary model tensors and is not itself
one indivisible host allocation. After removing only the PLE table, the two tensor shards retain
43,735,298,560 bytes (40.73 GiB) of Text-core tensor payload. That core cannot be resident on a
32 GB GPU before runtime allocations.

The main three-shard artifact contains Text, QSA metadata/indexer weights, GDN, GR, MoE, and PLE.
Vision is outside those shards. The selected diagnostic companion is `mmproj-BF16.gguf`, exactly
907,542,944 bytes with SHA-256
`2e788f8c511d8093c7b43cb87b2fd7e14228340318057f8fb20c86df2efe2355`. It has 334 GGUF tensors
representing 448,931,056 source values: the official 333-tensor inventory becomes 334 because the
temporal patch kernel is split into two GGUF tensors. Exactly 110 tensors / 444,100,608 values stay
BF16 and 224 tensors / 4,830,448 values are losslessly widened to F32; the widened objects include
both split patch-embedding kernels and the full `[1152,2304]` positional-embedding matrix. The
alternative `mmproj-F16.gguf` is 904,004,000 bytes with SHA-256
`1f7b7f0b984cf065c604360c29c8098362ed61b290db0ff12c6f360bb1a8a980`; it is not selected because
it adds a lossy BF16-to-F16 boundary. MTP sidecars exist separately but are outside this selected
external diagnostic: pinned stock llama.cpp does not execute them, and no executable MTP fork is
made an evidence authority here.
Pinned stock llama.cpp can provide deterministic short-prompt and quant-specific PPL evidence for
the main Text/PLE graph with partial CPU/GPU offload. Its current Qwen4Exp path calculates QSA
selection but executes core attention through a dense fallback, so it is neither QSA performance
evidence nor a substitute for exact selected-index and sparse-attention oracles.

A more aggressive custom core quant could potentially fit once the PLE table is host-mapped, but
it is a distinct weights profile. It must be converted from the pinned BF16 source rather than
double-quantized from UD-IQ1_S, must register and exactly decode every selected codec, and must pass
the frozen Op-oracle and paired per-token PPL criteria before target admission.

### External execution smoke

On 2026-09-04, all three downloaded UD-IQ1_S shard SHA-256 values matched the table above. The
pinned llama.cpp commit was built for `sm_120a` with CUDA compiler 13.3.73 and run on the RTX 5090
with driver 580.173.02. Both runs used mmap loading, lazy PLE mode, FlashAttention, 16 host threads,
and no warmup:

- a 256-token-context deterministic Text run loaded all three shards and generated four tokens from
  `The capital of France is`; its diagnostic timing was 53.3 prompt tokens/s and 26.3 generated
  tokens/s; and
- a fixed 31-GPU-layer, one-chunk, 32-token-context PPL run over
  `tests/targets/qwen4/fixtures/external_ppl_corpus.txt`, whose UTF-8 SHA-256 is
  `f689ec6374bdfc238155b616101ed7792f3390f64e8d10a333b07a9a71e8b1ae`, scored 15 positions and
  completed with `PPL = 15.8057 +/- 11.62342`. Its exact command, tokenization, profile, aggregate
  result, and four print-only prefix NLLs are frozen in `external_ppl_manifest.json` beside the
  fixture.

A subsequent capacity/QSA baseline fixed llama.cpp to 31 GPU layers with fit disabled, C=1, an
8192-token context, and IQ4_NL K/V cache. It processed all 5536 tokens of repository `README.md`
at SHA-256 `747a6ac9d0a7788ed1a0df42117824614e13af7b2eac6f08ccdaacfe004ab6d9`, crossing the
2048-token QSA selection budget and completing prompt evaluation at 62.06 tokens/s. IQ4_NL is only
llama.cpp's same-rate external cache option; it is not NInfer NVFP4-G16 and this result is not
NVFP4 numerical evidence.

These results prove only that the external heterogeneous quant can be loaded and that its short
Text/PLE graph executes on the available host/GPU configuration. The tiny PPL sample has large
uncertainty and is not a quality threshold, production-performance claim, independent Op oracle,
or NInfer parity result. The pinned runner applies the selected-id mask through dense MHA, so the
long-context run exercises QSA selection semantics but cannot qualify sparse-attention performance
or replace the independent QSA oracle. The runs exercise neither the separate Vision companion nor
MTP.

The selected native bring-up profile instead keeps PLE and the 17,196,646,400-byte routed gate/up
banks mapped on the host, stages only their selected expert slices, and leaves 26,538,652,160 bytes
(24.72 GiB) of ordinary weights resident on the GPU. It is an unregistered C=1 eager Text verifier
with a 4096-token ceiling and NVFP4-G16 QSA KV, not an advertised artifact profile. Section 3.1 of
the implementation plan owns its exact staging and qualification boundary. The native artifact
identity is `qwen4/verification` with weights id
`unsloth-ud-iq1-s-host-staged`; `tools/convert/qwen4/convert.py` owns the closed
1,224-tensor conversion. The 2026-09-04 local conversion produced a 72,535,629,824-byte `.ninfer`
after all three source hashes, tensor names, shapes, formats, counts, and 72,535,436,800 encoded
tensor bytes passed preflight. A subsequent streaming SHA-256 comparison matched every one of the
1,224 artifact payloads to its exact source tensor span. This establishes storage provenance and
does not by itself establish model execution parity. The corresponding real-artifact
materialization test passed on the RTX
5090: 26,538,652,160 payload bytes occupied a 26,538,656,768-byte device arena while the
45,996,784,640-byte mapped set retained its mapping after the reader closed. The same process also
allocated the 157,126,664-byte fixed C=1 state budget, including all twelve 4096-token NVFP4-G16
QSA K/V/index states, and one 8,448,000-byte expert staging slot. This is direct capacity evidence
for beginning native verification, not a logits, PPL, latency, or product-admission result. The
target-private package test additionally populated all 48 strongly typed layer views and reset the
GDN/QSA/PLE state before exact-decoding one real Q4_K token row into all four BF16 residual
branches. The package now also contains a complete synchronous C=1 eager Text Program and a
separate opt-in four-token native test. On 2026-09-04, the RTX 5090 test executed frozen
teacher-forced pairs `48->16451`, `16451->17120`, `17120->22188`, and `22188->11988`, producing
finite NLLs `3.42502`, `14.3384`, `12.2828`, and `8.42543` (four-position PPL `15031.6`) and an
exact reset/replay match for logits, NLL bits, final hidden, PLE rows/state, both GR boundaries,
all GDN/QSA state, QSA ids/counts/current NVFP4 rows, and MoE ids/weights. The corresponding pinned
llama.cpp NLLs were `3.15870537`, `13.7408428`, `12.0528638`, and `8.32870839`; this short trace is
useful localization evidence, not numerical-identity evidence, an Op tolerance, or a
product-quality/PPL threshold,
because the native verifier deliberately uses BF16 represented intermediates and NVFP4-G16 KV
while that external diagnostic uses its own FP32 graph profile and IQ4_NL KV.
The opt-in native test enforces the manifest's deliberately coarse cross-representation integration
gate: every paired prefix NLL must differ by at most 1.0 nat, their mean absolute difference by at
most 0.5 nat, and neither route may be non-finite. One nat caps a frozen target-probability ratio at
`e`; the mean bound detects systematic Program drift. These are behavioral integration limits, not
Op tolerances or a claim that the two numeric profiles are identical.

## 3. Exact component totals

| Source component | Tensors | Values | Payload bytes |
|---|---:|---:|---:|
| token embedding | 1 | 635,699,200 | 1,271,398,400 |
| final Text GR read | 3 | 6,563,840 | 13,127,680 |
| 48 attention GR modules | 192 | 317,030,400 | 634,060,800 |
| 36 GDN modules | 324 | 2,086,510,464 | 4,173,020,928 |
| 12 QSA modules | 108 | 617,358,336 | 1,234,716,672 |
| 48 sparse-MoE modules | 336 | 121,094,922,240 | 242,189,844,480 |
| 48 MoE GR modules | 192 | 317,030,400 | 634,060,800 |
| PLE, including padded embedding | 137 | 51,233,085,475 | 102,466,171,160 |
| MTP private tensors | 31 | 2,607,150,848 | 5,214,301,696 |
| Vision tower and merger | 333 | 448,931,056 | 897,862,112 |
| independent output head | 1 | 635,699,200 | 1,271,398,400 |
| **total** | **1658** | **179,999,981,459** | **359,999,963,128** |

The three I64 PLE buffers account for 35 values and 280 bytes; every other represented value is
BF16. The PLE embedding itself is 51,200,245,760 BF16 values and 102,400,491,520 bytes.

## 4. Name grammar and expansion rules

The tables below are a complete exact inventory. Braced variables expand only over the declared
sets:

```text
all Text layers L = {0,1,...,47}
GDN layers       G = L minus Q
QSA layers       Q = {3,7,11,15,19,23,27,31,35,39,43,47}
Vision blocks    B = {0,1,...,26}
PLE shards       S = {0,1,...,127}
```

Every pattern expands once per member of its set. Decimal indices have no zero padding inside
tensor names. Shapes use safetensors axis order exactly. `Bytes each` is the exact source payload
span of one expanded name.

## 5. Text-global and Gated Residual tensors

### 5.1 Text-global

| Exact source name | Shape | Dtype | Bytes each |
|---|---:|---:|---:|
| `model.language_model.embed_tokens.weight` | `[248320,2560]` | BF16 | 1,271,398,400 |
| `model.language_model.hyper_connection_mixer.hc_norm.weight` | `[10240]` | BF16 | 20,480 |
| `model.language_model.hyper_connection_mixer.input_mix_weight_down.weight` | `[320,10240]` | BF16 | 6,553,600 |
| `model.language_model.hyper_connection_mixer.input_mix_weight_up.weight` | `[10240,320]` | BF16 | 6,553,600 |
| `lm_head.weight` | `[248320,2560]` | BF16 | 1,271,398,400 |

The embedding and output head are different stored tensors; there is no alias or tied-weight
source contract.

### 5.2 Per-sublayer GR tensors

For every `l in L`, expand every row below once with `kind = attn` and once with `kind = mlp`.
This yields eight tensors per decoder layer and 384 tensors total.

| Exact source-name pattern | Shape | Dtype | Bytes each |
|---|---:|---:|---:|
| `model.language_model.layers.{l}.{kind}_hyper_connection.hc_norm.weight` | `[10240]` | BF16 | 20,480 |
| `model.language_model.layers.{l}.{kind}_hyper_connection.input_mix_weight_down.weight` | `[320,10240]` | BF16 | 6,553,600 |
| `model.language_model.layers.{l}.{kind}_hyper_connection.input_mix_weight_up.weight` | `[10240,320]` | BF16 | 6,553,600 |
| `model.language_model.layers.{l}.{kind}_hyper_connection.block_inject_weight.weight` | `[4,10240]` | BF16 | 81,920 |

## 6. Per-layer mixer and MoE tensors

### 6.1 GDN, every `l in G`

| Exact source-name suffix under `model.language_model.layers.{l}.linear_attn.` | Shape | Dtype | Bytes each |
|---|---:|---:|---:|
| `A_log` | `[48]` | BF16 | 96 |
| `dt_bias` | `[48]` | BF16 | 96 |
| `conv1d.weight` | `[10240,1,4]` | BF16 | 81,920 |
| `in_proj_a.weight` | `[48,2560]` | BF16 | 245,760 |
| `in_proj_b.weight` | `[48,2560]` | BF16 | 245,760 |
| `in_proj_qkv.weight` | `[10240,2560]` | BF16 | 52,428,800 |
| `in_proj_z.weight` | `[6144,2560]` | BF16 | 31,457,280 |
| `norm.weight` | `[128]` | BF16 | 256 |
| `out_proj.weight` | `[2560,6144]` | BF16 | 31,457,280 |

There are nine names per GDN layer and 324 names total.

### 6.2 QSA, every `l in Q`

| Exact source-name suffix under `model.language_model.layers.{l}.self_attn.` | Shape | Dtype | Bytes each |
|---|---:|---:|---:|
| `indexer.index_qk_proj.weight` | `[640,2560]` | BF16 | 3,276,800 |
| `indexer.k_layernorm.weight` | `[128]` | BF16 | 256 |
| `indexer.q_layernorm.weight` | `[128]` | BF16 | 256 |
| `k_norm.weight` | `[256]` | BF16 | 512 |
| `k_proj.weight` | `[512,2560]` | BF16 | 2,621,440 |
| `o_proj.weight` | `[2560,6144]` | BF16 | 31,457,280 |
| `q_norm.weight` | `[256]` | BF16 | 512 |
| `q_proj.weight` | `[12288,2560]` | BF16 | 62,914,560 |
| `v_proj.weight` | `[512,2560]` | BF16 | 2,621,440 |

There are nine names per QSA layer and 108 names total. `q_proj` stores 24 consecutive 512-row
head groups; within each head group, 256 query rows are followed by that head's 256 output-gate
rows. It is not a global 6144-query-row half followed by a global 6144-gate-row half.
`index_qk_proj` stores four 128-wide query heads followed by one 128-wide key head.

### 6.3 Sparse MoE, every `l in L`

| Exact source-name suffix under `model.language_model.layers.{l}.mlp.` | Shape | Dtype | Bytes each |
|---|---:|---:|---:|
| `gate.weight` | `[512,2560]` | BF16 | 2,621,440 |
| `experts.gate_up_proj` | `[512,1280,2560]` | BF16 | 3,355,443,200 |
| `experts.down_proj` | `[512,2560,640]` | BF16 | 1,677,721,600 |
| `shared_expert.gate_proj.weight` | `[640,2560]` | BF16 | 3,276,800 |
| `shared_expert.up_proj.weight` | `[640,2560]` | BF16 | 3,276,800 |
| `shared_expert.down_proj.weight` | `[2560,640]` | BF16 | 3,276,800 |
| `shared_expert_gate.weight` | `[1,2560]` | BF16 | 5,120 |

There are seven names per decoder layer and 336 names total. Expert bank axis zero is logical
expert id; the source does not permute expert rows.

## 7. PLE tensors at zero-based layer 1

All names in this section begin with `model.language_model.layers.1.ple.`.

| Exact suffix | Count | Shape | Dtype | Total bytes |
|---|---:|---:|---:|---:|
| `conv1d.weight` | 1 | `[10240,1,4]` | BF16 | 81,920 |
| `key_proj.weight` | 1 | `[10240,2560]` | BF16 | 52,428,800 |
| `value_proj.weight` | 1 | `[2560,2560]` | BF16 | 13,107,200 |
| `norm_key.weight` | 1 | `[10240]` | BF16 | 20,480 |
| `norm_query.weight` | 1 | `[10240]` | BF16 | 20,480 |
| `norm_conv.weight` | 1 | `[10240]` | BF16 | 20,480 |
| `ple_embedding.layer_multipliers` | 1 | `[3]` | I64 | 24 |
| `ple_embedding.ngram_heads_offsets` | 1 | `[16]` | I64 | 128 |
| `ple_embedding.ngram_heads_vocab_sizes` | 1 | `[16]` | I64 | 128 |
| `ple_embedding.ngram_embedding.shard_{s}.weight`, every `s in S` | 128 | `[2500012,160]` | BF16 | 102,400,491,520 |

The 128 embedding shard tensors concatenate in numeric `s` order to `[320001536,160]`. They are
source-level pieces of one logical PLE table, not 128 independent semantic tables.

The exact I64 payload values are:

```text
layer_multipliers = [23703573157769, 20109073645365, 8052911324071]
ngram_heads_vocab_sizes = [
  20000003, 20000023, 20000033, 20000047,
  20000059, 20000063, 20000069, 20000077,
  20000081, 20000093, 20000107, 20000147,
  20000153, 20000159, 20000161, 20000171
]
ngram_heads_offsets = [
  0, 20000003, 40000026, 60000059,
  80000106, 100000165, 120000228, 140000297,
  160000374, 180000455, 200000548, 220000655,
  240000802, 260000955, 280001114, 300001275
]
```

## 8. MTP private tensors

The MTP namespace contains exactly 31 BF16 tensors and 5,214,301,696 bytes. It does not contain an
embedding or output head; candidate execution shares the main checkpoint tensors.

### 8.1 Stem and final read

| Exact source name | Shape | Bytes |
|---|---:|---:|
| `mtp.fc_embedding.weight` | `[2560,2560]` | 13,107,200 |
| `mtp.fc_hidden.weight` | `[2560,2560]` | 13,107,200 |
| `mtp.pre_fc_norm_embedding.weight` | `[2560]` | 5,120 |
| `mtp.pre_fc_norm_hidden.weight` | `[10240]` | 20,480 |
| `mtp.hyper_connection_mixer.hc_norm.weight` | `[10240]` | 20,480 |
| `mtp.hyper_connection_mixer.input_mix_weight_down.weight` | `[320,10240]` | 6,553,600 |
| `mtp.hyper_connection_mixer.input_mix_weight_up.weight` | `[10240,320]` | 6,553,600 |

### 8.2 MTP attention and MoE GR modules

For each `kind = attn` and `kind = mlp`, expand:

| Exact source-name pattern | Shape | Bytes each |
|---|---:|---:|
| `mtp.layers.0.{kind}_hyper_connection.hc_norm.weight` | `[10240]` | 20,480 |
| `mtp.layers.0.{kind}_hyper_connection.input_mix_weight_down.weight` | `[320,10240]` | 6,553,600 |
| `mtp.layers.0.{kind}_hyper_connection.input_mix_weight_up.weight` | `[10240,320]` | 6,553,600 |
| `mtp.layers.0.{kind}_hyper_connection.block_inject_weight.weight` | `[4,10240]` | 81,920 |

### 8.3 MTP QSA

The exact prefix is `mtp.layers.0.self_attn.`. Its nine suffixes, shapes, and byte spans are exactly
the QSA rows in Section 6.2.

### 8.4 MTP sparse MoE

The exact prefix is `mtp.layers.0.mlp.`. Its seven suffixes, shapes, and byte spans are exactly the
sparse-MoE rows in Section 6.3.

The 7 stem/final-read tensors, 8 GR tensors, 9 QSA tensors, and 7 MoE tensors sum to the exact 31
private tensors.

## 9. Vision tensors

### 9.1 Stem

| Exact source name | Shape | Dtype | Bytes each |
|---|---:|---:|---:|
| `model.visual.patch_embed.proj.weight` | `[1152,3,2,16,16]` | BF16 | 3,538,944 |
| `model.visual.patch_embed.proj.bias` | `[1152]` | BF16 | 2,304 |
| `model.visual.pos_embed.weight` | `[2304,1152]` | BF16 | 5,308,416 |

### 9.2 Transformer blocks, every `b in B`

| Exact source-name suffix under `model.visual.blocks.{b}.` | Shape | Dtype | Bytes each |
|---|---:|---:|---:|
| `norm1.weight` | `[1152]` | BF16 | 2,304 |
| `norm1.bias` | `[1152]` | BF16 | 2,304 |
| `attn.qkv.weight` | `[3456,1152]` | BF16 | 7,962,624 |
| `attn.qkv.bias` | `[3456]` | BF16 | 6,912 |
| `attn.proj.weight` | `[1152,1152]` | BF16 | 2,654,208 |
| `attn.proj.bias` | `[1152]` | BF16 | 2,304 |
| `norm2.weight` | `[1152]` | BF16 | 2,304 |
| `norm2.bias` | `[1152]` | BF16 | 2,304 |
| `mlp.linear_fc1.weight` | `[4304,1152]` | BF16 | 9,916,416 |
| `mlp.linear_fc1.bias` | `[4304]` | BF16 | 8,608 |
| `mlp.linear_fc2.weight` | `[1152,4304]` | BF16 | 9,916,416 |
| `mlp.linear_fc2.bias` | `[1152]` | BF16 | 2,304 |

There are 12 tensors per block and 324 block tensors total.

### 9.3 Merger

| Exact source name | Shape | Dtype | Bytes each |
|---|---:|---:|---:|
| `model.visual.merger.norm.weight` | `[1152]` | BF16 | 2,304 |
| `model.visual.merger.norm.bias` | `[1152]` | BF16 | 2,304 |
| `model.visual.merger.linear_fc1.weight` | `[4608,4608]` | BF16 | 42,467,328 |
| `model.visual.merger.linear_fc1.bias` | `[4608]` | BF16 | 9,216 |
| `model.visual.merger.linear_fc2.weight` | `[2560,4608]` | BF16 | 23,592,960 |
| `model.visual.merger.linear_fc2.bias` | `[2560]` | BF16 | 5,120 |

The 3 stem, 324 block, and 6 merger tensors sum to 333 Vision tensors.

## 10. Pinned metadata and frontend resources

The official repository contains both the self-contained tokenizer and legacy split vocabulary
files. The six resources in the first group are the exact candidates for a future complete
`.ninfer` frontend image; `vocab.json` and `merges.txt` are redundant inputs and must not silently
become a second tokenizer lane. `config.json` remains target/converter authority rather than an
opaque runtime resource.

| Source file | Bytes | SHA-256 | Future role |
|---|---:|---|---|
| `tokenizer.json` | 12,809,320 | `0997f410c57a1f4e53b09e4be8f4a172d90edd9564368fb0847030937229b9f3` | canonical tokenizer |
| `tokenizer_config.json` | 17,928 | `b11349aafa7cdc6a320767cf7ceb29ed82f7eda5d65e8e0819e76f0ce947bf27` | added tokens and tokenizer policy |
| `chat_template.jinja` | 8,952 | `c3cf9e34abf4f9e36c2d72165aa9c132d3e2a725b6c2586aaa3a8af9d7a81041` | exact message/tool/reasoning template |
| `generation_config.json` | 202 | `e70c136c1b78ddc1fb0905bac8e733a4dc448d4f852a5dd75143fffc70be550e` | sampling and stop defaults |
| `preprocessor_config.json` | 390 | `27225450ac9c6529872ee1924fcb0962ff5634834f817040f444118116f4e516` | image preprocessing |
| `video_preprocessor_config.json` | 385 | `7768af27c1fafa9cc9011c1dc20067e03f8915e03b63504550e11d5066986d13` | video preprocessing |
| `vocab.json` | 6,722,759 | `ce99b4cb2983d118806ce0a8b777a35b093e2000a503ebde25853284c9dfa003` | redundant source tokenizer form |
| `merges.txt` | 3,353,259 | `a9d356d7bdf1ef4949e3e748e95b8e10ad9d4e2e838eddc38a0a7b6b94d1db8d` | redundant source tokenizer form |
| `config.json` | 4,745 | `889658f2508e8c61d409b02e70e0d78d8d4452ec65aaafbe129805d213d2e74b` | exact target/converter configuration |

The tokenizer-addressable domain is `0..248076`; matrix rows `248077..248319` are reserved and not
ordinary frontend tokens. Exact added tokens `248044..248076` are:

```text
248044 <|endoftext|>       248045 <|im_start|>        248046 <|im_end|>
248047 <|object_ref_start|> 248048 <|object_ref_end|> 248049 <|box_start|>
248050 <|box_end|>         248051 <|quad_start|>      248052 <|quad_end|>
248053 <|vision_start|>    248054 <|vision_end|>      248055 <|vision_pad|>
248056 <|image_pad|>       248057 <|video_pad|>       248058 <tool_call>
248059 </tool_call>       248060 <|fim_prefix|>      248061 <|fim_middle|>
248062 <|fim_suffix|>      248063 <|fim_pad|>         248064 <|repo_name|>
248065 <|file_sep|>        248066 <tool_response>     248067 </tool_response>
248068 <think>             248069 </think>            248070 <|audio_start|>
248071 <|audio_end|>       248072 <tts_pad>            248073 <tts_text_bos>
248074 <tts_text_eod>      248075 <tts_text_bos_single> 248076 <|audio_pad|>
```

Tokenizer configuration sets no implicit BOS, `model_max_length = 262144`, tokenizer class
`Qwen2Tokenizer`, EOS token `<|im_end|>`, and pad token `<|endoftext|>`. Generation defaults are
BOS/pad 248044, stop ids `[248046,248044]`, sampling enabled, temperature 1.0, top-k 20, and top-p
0.95. Model configuration separately uses 248044 as its BOS/EOS and PLE segment-reset id.

The image resource fixes RGB mean/std `[0.5,0.5,0.5]`, patch 16, temporal patch two, merge two,
and pixel area range 65536 through 16777216. Video uses the same normalization and geometry with
area range 4096 through 25165824. The chat template is an observable artifact resource: it owns
vision placeholders, `xhigh`/`medium`/`low` reasoning instructions, thinking preservation, tool
schema/rendering, tool-response grouping, and generation-prompt framing. Implementations must use
golden token vectors rather than approximating it in C++.

## 11. Future conversion and admission boundary

This BF16 inventory establishes what a converter must account for, not how an unknown fitting
target should be encoded. For any selected Qwen4 target, conversion admission requires:

- rereading every source header at that target's fixed revision and rejecting every unknown,
  missing, duplicate, wrong-shape, or wrong-dtype tensor before output is opened;
- consuming every source tensor exactly once or naming a deliberate validate-only object;
- independently verifying packed decode against represented source values;
- retaining frontend resources byte-for-byte under one canonical tokenizer path;
- keeping logical expert id equal to router and expert-bank row;
- when the exact target contains PLE, defining it as one logical table even if its physical
  placement is mapped host storage; and
- proving device weights, fixed state, growing QSA pools, workspace, graph memory, and headroom fit
  the one-GPU product contract before registration.

The currently audited BF16 and UD-IQ1_S profiles fail the last condition. Registration remains
barred unless a complete custom profile passes Phase 0. Until then, the preview must not acquire a
placeholder `.ninfer` identity, partial conversion profile, optional omitted PLE/MTP/Vision objects,
host-streamed expert fallback, or multi-GPU path through this reference.

## 12. Primary source addresses

- https://huggingface.co/Qwen/Qwen3.8-Flash-Next/tree/de4b8e4d43b917e7706784d8bb445c9af86a3540
- https://huggingface.co/Qwen/Qwen3.8-Flash-Next/blob/de4b8e4d43b917e7706784d8bb445c9af86a3540/model.safetensors.index.json
- https://huggingface.co/unsloth/Qwen3.8-Flash-Next-GGUF/tree/38bb39ee97821de2c9009abb7e93950eec396e66
- https://huggingface.co/unsloth/Qwen3.8-Flash-Next-GGUF/tree/38bb39ee97821de2c9009abb7e93950eec396e66/UD-IQ1_S
- https://huggingface.co/unsloth/Qwen3.8-Flash-Next-GGUF/tree/38bb39ee97821de2c9009abb7e93950eec396e66/MTP
- https://github.com/ggml-org/llama.cpp/tree/9a4843cf2f1a3fc8e39f8148e92ee6bfe18e2db6
- https://github.com/huggingface/transformers/blob/c119ec3cc37ab69642f39cca2de4187714002b08/src/transformers/models/qwen4_exp/modeling_qwen4_exp.py
- https://github.com/huggingface/transformers/blob/c119ec3cc37ab69642f39cca2de4187714002b08/src/transformers/models/qwen4_exp/configuration_qwen4_exp.py
