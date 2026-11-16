# Qwen3.8-Flash-Next artifact contract

This document defines the experimental `qwen3.8-flash-next/f3-fp8-nvfp4` artifact: identity,
object names, order, shapes, numeric formats, storage layouts, fused row order, source
transforms, the routed-expert store, the dense-study payloads, conversion, validation, and
binding. Common framing is defined in
[`artifact-container.md`](artifact-container.md), numeric semantics in
[`tensor-formats.md`](tensor-formats.md), and byte layouts in
[`storage-layouts.md`](storage-layouts.md). The registered Engine target's execution design and
qualification status are recorded in the [target plan](qwen3.8-flash-next-plan.md).

## 1. Identity

```text
filename   = qwen3.8-flash-next-f3-fp8-nvfp4.ninfer
model_id   = qwen3.8-flash-next
weights_id = f3-fp8-nvfp4
target_key = qwen3_8_flash_next
```

The artifact contains the Text model only: six frontend resources, the token embedding, 48
decoder layers, the final hyper-connection, the output head, and the routed experts. Vision and
MTP tensors are not stored, and the 51.2B-parameter n-gram (PLE) table stays outside the artifact
(`ngram_table.int8.bin`, read by `src/targets/qwen3_8_flash_next/impl/ngram_table.*`); the
artifact carries its hash constants (Section 5.2) so they can be checked against the table
header. Every format is fixed by the weights profile. An optional complete DFlash2 companion
appends the objects of Section 10 under the same identity; it does not replace or requantize any
base-model object.

## 2. Fixed target facts

All matrix shapes are logical `[N,K] = [output rows, input columns]`; quantization groups run
along `K`.

| Fact | Value |
|---|---:|
| hidden width | 2560 |
| layers | 48: GDN, except QSA layers `3, 7, ..., 47` (every fourth) |
| vocabulary rows (embedding and head, untied) | 248320 |
| hyper-connection streams / rank | 4 / 320 (flat width 10240) |
| GDN key heads x width / value heads x width | 16 x 128 / 48 x 128 |
| GDN convolution channels / taps | 10240 / 4 |
| QSA query heads / KV heads / head width | 24 / 2 / 256 |
| QSA indexer query heads x width / key width | 4 x 128 / 128 |
| routed experts per layer / per token | 512 / 10 |
| expert and shared-expert intermediate width | 640 |
| PLE layer | 1 (zero-based; the checkpoint's `ple_layer_ids` is `[2]`) |
| n-gram heads / row width | 16 / 160 |

Layer numbers in object names are unpadded zero-based decimal integers.

## 3. Numeric formats and row order

### 3.1 Format assignment

| Role | Format | Layout |
|---|---|---|
| GDN input `query_key_value_z` `[16384,2560]` | `FP8_E4M3FN_ROW_BF16S` | `row-scale-v1` |
| GDN output `[2560,6144]` | `FP8_E4M3FN_ROW_BF16S` | `row-scale-v1` |
| attention input `query_key_gate_value` `[13312,2560]` | `NVFP4` | `blockscale-k16-m128x4-v1` |
| attention output `[2560,6144]` | `FP8_E4M3FN_ROW_BF16S` | `row-scale-v1` |
| output head `[248320,2560]` | `FP8_E4M3FN_ROW_BF16S` | `row-scale-v1` |
| hyper-connection down `[320,10240]` and up `[10240,320]` | `FP8_E4M3FN_ROW_BF16S` | `row-scale-v1` |
| shared-expert record, per layer | `E2M1G16_E4M3FN` `[1,2560,640]` | `moe-expert-tiles-v1` |
| shared-expert matrix multipliers, per layer | `FP32` `[3]` | `contiguous-le-v1` |
| routed experts with an NVFP4 study code | `E2M1G16_E4M3FN` | `moe-expert-tiles-v1` |
| routed experts with an F3 study code | `F3G32_E4M3FN` | `moe-expert-tiles-v1` |
| routed-expert matrix multipliers | `FP32` | `contiguous-le-v1` |
| expert codec map, F3 books, n-gram hash constants | `I32` | `contiguous-le-v1` |
| GDN `a_log`, `dt_bias` | `FP32` (exact widening of the BF16 source) | `contiguous-le-v1` |
| token embedding and every other tensor | `BF16` (exact source words) | `contiguous-le-v1` |

The BF16 tensors are the token embedding, the hyper-connection norms and write projections, the
PLE projections, norms and convolution, the GDN convolution, a|b projection and output norm, the
QSA query/key norms, indexer projections and indexer norms, and the MoE control matrix. Norm
weights keep their stored values: every norm is zero-centered (`1 + w`) except the GDN output
norm, a plain gamma; the consumers apply the offset.

### 3.2 Quantized payloads

Every quantized dense tensor is a payload of the dense-format study
(`/home/battlefront/ai/study/results/dense/REPORT.md`, recipe `fallback_mid`, with the
attention-input parents of `ATTN_PARENT.md` and the hyper-connection projections of `HC.md`),
ingested word for word (Section 3.4):

| Role | Study method |
|---|---|
| GDN input | `fp8_gptq` (qkv and z records) |
| GDN output, attention output, output head | `fp8_gptq` |
| attention input | one parent record per layer: MSE block scales, RTN q/gate and k, GPTQ v |
| hyper-connection down/up, every layer and the final read | `fp8_gptq` (`HC.md`) |
| shared expert | `nv_gptq_mse` (gate, up, down records) |

NVFP4 parents carry their weight divisor `d_w` and store no input divisor: the Flash-Next NVFP4
projections run A16 only and bind with `input_scale_divisor = 0` (Section 9).

### 3.3 Fused row order

- GDN `query_key_value_z`: `[query 2048, key 2048, value 6144, z 6144]` = `in_proj_qkv` rows then
  `in_proj_z` rows. Value heads stay in the checkpoint's grouped order (value head `h` uses
  query/key head `h / 3`), the order `gated_delta_net` expects; no head reordering is applied.
- GDN `a_b_projection`: `[a 48, b 48]`, the input of `gdn_gating_packed`.
- attention `query_key_gate_value`: `[query 6144, key 512, gate 6144, value 512]`. Each of the 24
  heads of the source `q_proj` stores `[query 256, gate 256]`; the converter separates the halves,
  so query row `256 h + d` is `q_proj` row `512 h + d` and gate row `256 h + d` is `q_proj` row
  `512 h + 256 + d`.
- indexer: `index_query` is `index_qk_proj` rows `[0,512)` (four heads of 128), `index_key` rows
  `[512,640)`.
- MoE `control` `[544,2560]`: router rows `0..511`, `shared_expert_gate` as row 512, positive-zero
  rows `513..543` (the `moe_route` contract).
- shared expert `shared_expert`: one `moe-expert-tiles-v1` record of gate `[640,2560]`, up
  `[640,2560]` and down `[2560,640]`, laid out exactly as a routed E2M1G16 record (2,764,800
  bytes), with `shared_matrix_scales` `[gate, up, down]`; the `moe_experts` `shared_record` and
  `shared_matrix_scales` inputs.
- GDN convolution: tap-major `[4,10240]` (`causal_conv1d_silu` reads `weight[tap * C + c]`).
- PLE convolution: taps fastest `[10240,4]` (`ple_inject` reads `conv[c * 4 + tap]`).

### 3.4 Dense-study payloads

The converter copies the study's quantized payloads without requantization. The study stores
one NPZ record per matrix at `quant/dense/{layer}/{hf_key}/{method}.npz` (the head at
`head/lm_head/`), fingerprinted in `results/dense/artifact_hashes.json`: `q` holds the codes
(E4M3FN `[M,N]`, or E2M1 nibbles `[M,N/2]` with the even column in the low nibble), `scale` the
BF16 row multipliers `[M,1]` (FP8) or E4M3FN group scales `[M,N/16]` (NVFP4), `global_scale`
1.0 (FP8) or the NVFP4 matrix multiplier, plus `method` and `damping`. The converter checks each
record's SHA-256 and size, its method against `summary.json` `recommended_recipe`, and its
damping against `damping_by_tensor`, then:

- FP8 parents concatenate the records' codes and BF16 row-scale words in parent row order (GDN
  input: `in_proj_qkv` then `in_proj_z`); the global scale must be exactly 1.0;
- the hyper-connection down/up projections are FP8 records under `quant/dense/hc/`
  (`{layer}/{attn,mlp}_hyper_connection.input_mix_weight_{down,up}.weight/fp8_gptq.npz` for the
  `mixer_hyper_connection` and `moe_hyper_connection` reads, and
  `final/hyper_connection_mixer.input_mix_weight_{down,up}.weight/fp8_gptq.npz`), whose
  fingerprints and damping are in `hc_validation.json` (checked against the calibration's
  selection in `hc_calibration.json`); codes and BF16 row multipliers are copied exactly;
- the shared expert's three `nv_gptq_mse` records become one E2M1G16 record (codes and E4M3FN
  scales exactly) and their FP32 global scales the three multipliers, word for word;
- the attention-input parent comes from one study record per QSA layer,
  `quant/dense/attn_parent/{layer}.npz` (specified in `results/dense/ATTN_PARENT.md`,
  fingerprinted in `attn_parent_validation.json`): an NVFP4 parent under this repository's
  convention, packed codes `q`, natural E4M3FN block scales `scale` (MSE-searched), and one FP32
  `weight_divisor` `d_w = 2688 / amax(q_proj, k_proj, v_proj)`, with RTN codes for q/gate and k
  and GPTQ codes for v. It stores rows in source order per part (`q_proj` with its per-head
  interleave, `k_proj`, `v_proj`); the converter checks its metadata (format, version, shape,
  parts, row offsets, rounding, v damping) and permutes whole code and scale rows into
  `[query, key, gate, value]` (Section 3.3), copying the divisor word unchanged.

## 4. Object order and frontend resources

Objects are written in this order:

1. the six frontend resources: `frontend/tokenizer.json`, `frontend/tokenizer_config.json`,
   `frontend/chat_template.jinja`, `frontend/generation_config.json`,
   `frontend/preprocessor_config.json`, `frontend/video_preprocessor_config.json` (raw source
   bytes; the shared Qwen3.6 frontend binds all six);
2. `text/token_embedding`;
3. layers `0..47` (Section 5);
4. `text/final_hyper_connection/{norm,down,up}`;
5. `text/output_head`;
6. the routed-expert objects (Section 6).

Readers bind by name. Object names contain no format spelling.

## 5. Layer inventory

Every layer `l` emits, in order: on layer 1 only the PLE objects (5.2); the mixer
hyper-connection (5.1, prefix `mixer_hyper_connection`); the GDN (5.3) or QSA (5.4) objects; the
MoE hyper-connection (5.1, prefix `moe_hyper_connection`); the MoE objects (5.5).

### 5.1 Hyper-connection

| Object `text/layers/{l}/{prefix}/...` | Shape | Format | Source `model.language_model.layers.{l}.{attn,mlp}_hyper_connection.` |
|---|---|---|---|
| `norm` | `[10240]` | `BF16` | `hc_norm.weight` |
| `down` | `[320,10240]` | `FP8_E4M3FN_ROW_BF16S` | `input_mix_weight_down.weight` |
| `up` | `[10240,320]` | `FP8_E4M3FN_ROW_BF16S` | `input_mix_weight_up.weight` |
| `write` | `[4,10240]` | `BF16` | `block_inject_weight.weight` |

`text/final_hyper_connection/{norm,down,up}` come from `model.language_model.hyper_connection_mixer`
and have no `write`.

### 5.2 PLE (layer 1)

| Object `text/layers/1/ple/...` | Shape | Format | Source `...layers.1.ple.` |
|---|---|---|---|
| `key_projection` | `[10240,2560]` | `BF16` | `key_proj.weight` |
| `value_projection` | `[2560,2560]` | `BF16` | `value_proj.weight` |
| `key_norm` | `[10240]` | `BF16` | `norm_key.weight` |
| `query_norm` | `[10240]` | `BF16` | `norm_query.weight` |
| `conv_norm` | `[10240]` | `BF16` | `norm_conv.weight` |
| `convolution` | `[10240,4]` | `BF16` | `conv1d.weight` `[10240,1,4]` |
| `ngram_hash_multipliers` | `[3,2]` | `I32` | `ple_embedding.layer_multipliers` (int64 `[3]`) |
| `ngram_head_offsets` | `[16,2]` | `I32` | `ple_embedding.ngram_heads_offsets` (int64 `[16]`) |
| `ngram_head_sizes` | `[16,2]` | `I32` | `ple_embedding.ngram_heads_vocab_sizes` (int64 `[16]`) |

The registry has no 64-bit integer format, so each int64 constant is stored as its exact
little-endian word pair: element `[i,0]` is the low and `[i,1]` the high 32 bits, which makes the
payload byte-identical to the source int64 array.

### 5.3 GDN layer

| Object `text/layers/{l}/gdn/...` | Shape | Format | Source `...layers.{l}.linear_attn.` |
|---|---|---|---|
| `query_key_value_z` | `[16384,2560]` | see 3.1 | `in_proj_qkv.weight`, `in_proj_z.weight` |
| `convolution` | `[4,10240]` | `BF16` | `conv1d.weight` |
| `a_b_projection` | `[96,2560]` | `BF16` | `in_proj_a.weight`, `in_proj_b.weight` |
| `a_log` | `[48]` | `FP32` | `A_log` |
| `dt_bias` | `[48]` | `FP32` | `dt_bias` |
| `norm` | `[128]` | `BF16` | `norm.weight` |
| `output` | `[2560,6144]` | see 3.1 | `out_proj.weight` |

### 5.4 QSA layer

| Object `text/layers/{l}/attention/...` | Shape | Format | Source `...layers.{l}.self_attn.` |
|---|---|---|---|
| `query_key_gate_value` | `[13312,2560]` | see 3.1 | `q_proj`, `k_proj`, `v_proj` weights |
| `query_norm` | `[256]` | `BF16` | `q_norm.weight` |
| `key_norm` | `[256]` | `BF16` | `k_norm.weight` |
| `index_query` | `[512,2560]` | `BF16` | `indexer.index_qk_proj.weight` |
| `index_key` | `[128,2560]` | `BF16` | `indexer.index_qk_proj.weight` |
| `index_query_norm` | `[128]` | `BF16` | `indexer.q_layernorm.weight` |
| `index_key_norm` | `[128]` | `BF16` | `indexer.k_layernorm.weight` |
| `output` | `[2560,6144]` | see 3.1 | `o_proj.weight` |

### 5.5 MoE

| Object `text/layers/{l}/moe/...` | Shape | Format | Source `...layers.{l}.mlp.` |
|---|---|---|---|
| `control` | `[544,2560]` | `BF16` | `gate.weight`, `shared_expert_gate.weight` |
| `shared_expert` | `[1,2560,640]` | `E2M1G16_E4M3FN` | `shared_expert.{gate,up,down}_proj.weight` (Section 3.4) |
| `shared_matrix_scales` | `[3]` | `FP32` | the same three matrices |

## 6. Routed experts

| Object | Shape | Format |
|---|---|---|
| `text/routed_experts/e2m1_records` | `[1583,2560,640]` | `E2M1G16_E4M3FN` / `moe-expert-tiles-v1` |
| `text/routed_experts/f3_records` | `[22993,2560,640]` | `F3G32_E4M3FN` / `moe-expert-tiles-v1` |
| `text/routed_experts/codecs` | `[48,512]` | `I32` |
| `text/routed_experts/f3_codebooks` | `[48,3]` | `I32` |
| `text/routed_experts/matrix_scales` | `[48,512,3]` | `FP32` |

- `codecs[l,e]` is expert `(l,e)`'s `ops::MoeExpertCodec`: 0 for an E2M1G16 (NVFP4) record, 1 for
  an F3G32 record. The record index of `(l,e)` within its codec's tensor is the number of
  experts with the same codec that precede it in layer-major `(l,e)` order, so both record
  tensors list their experts in that order.
- `f3_codebooks[l,m]` is the F3 book of layer `l`, matrix `m` (0 gate, 1 up, 2 down): eight E2M1
  words, entry `v` in bits `4v..4v+3` of the I32 word. Every book of the current store is
  `[-6,-4,-2,-0.5,0.5,2,4,6]`, word `0x76419CEF`.
- `matrix_scales[l,e,m]` is the FP32 multiplier of matrix `m` of expert `(l,e)`; a layer's
  `[512,3]` view starts at a 6,144-byte (16-byte aligned) offset. The shared expert's
  multipliers are the per-layer `shared_matrix_scales` (Section 5.5).
- Record bytes follow `moe-expert-tiles-v1`, the byte order `include/ninfer/ops/moe_experts.h`
  reads: 1,996,800 bytes per F3G32 and 2,764,800 per E2M1G16 record.

Source: the expert-format study's `mixed_mse_gptq` store
(`/home/battlefront/ai/study/quant/mixed_mse_gptq/experts.bin` and `manifest.json`; format in
`results/formats/REPORT.md`, "Exact storage", and `fnstudy/formats/{store,codecs}.py`): the 1,583
most-routed experts (by calibration route count) in GPTQ NVFP4 (E4M3 scale per 16), the other
22,993 in GPTQ F3 (3-bit indices into the layer/matrix book, MSE-searched E4M3 scale per 32), one
FP32 multiplier per matrix. The converter repacks every code, scale, multiplier, and book word
without changing it; the BF16 checkpoint's routed-expert tensors are never read.

## 7. Inventory summary

| Component | Objects |
|---|---:|
| frontend resources | 6 |
| embedding and head | 2 |
| hyper-connections (`2 x 48 x 4 + 3`) | 387 |
| PLE | 9 |
| GDN layers (`36 x 7`) | 252 |
| QSA layers (`12 x 8`) | 96 |
| MoE control, shared-expert record and multipliers (`48 x 3`) | 144 |
| routed experts | 5 |
| total | 901 (895 tensors, 6 resources) |

| Family | Payload bytes |
|---|---:|
| routed experts (records) | 50,289,100,800 |
| routed-expert tables | 393,792 |
| vocabulary (BF16 embedding, FP8 head) | 1,907,594,240 |
| GDN input | 1,511,129,088 |
| hyper-connections | 647,598,720 |
| GDN output | 566,415,360 |
| attention input | 230,031,408 |
| attention output | 188,805,120 |
| MoE control | 133,693,440 |
| shared experts (records and multipliers) | 132,710,976 |
| PLE | 65,679,640 |
| attention control (norms, indexer) | 39,340,032 |
| GDN control | 20,666,880 |
| frontend | 12,829,285 |
| total payload | 55,745,988,781 |

## 8. Conversion

```bash
python3 -m tools.convert.qwen3_8_flash_next.convert \
  --model /home/battlefront/ai/qwen3.8-flash-next-bf16 \
  --experts /home/battlefront/ai/study/quant/mixed_mse_gptq/experts.bin \
  --experts-manifest /home/battlefront/ai/study/quant/mixed_mse_gptq/manifest.json \
  --dense-records /home/battlefront/ai/study/quant/dense \
  --dense-results /home/battlefront/ai/study/results/dense \
  --out models/qwen3.8-flash-next-f3-fp8-nvfp4.ninfer
```

The interpreter needs PyTorch and safetensors. Inputs are explicit paths; the output must not
exist (an interrupted run leaves a partial file to delete). The converter validates the
checkpoint config, the store header, bitmap, size and manifest, every source tensor's presence
and dtype, and every dense-study record (fingerprint, method, damping, metadata) before creating
the output; it streams one dense tensor or expert record at a time (CPU only) and writes
`<out>.conversion.json` with the study recipe, source paths and digests, converter revision,
inventory and byte summaries, elapsed time, peak RSS, and the validation result.

After writing, the converter re-reads the artifact and validates it; `--validate-only` runs the
same validation on an existing `--out` and writes `<out>.validation.json`. Validation rebuilds
the expected inventory from the sources and requires an exact directory match, then checks every
object:

- resources and BF16/FP32/I32 tensors equal their sources bit for bit, with row partitions and
  the convolution transposes addressed by explicit index formulas;
- every routed-expert record decodes through the layout's index-formula decoder to exactly the
  codes and scales of its study record, and the codec map, books and multipliers equal the
  study's words;
- every object ingested from the dense study decodes to exactly the words of its
  re-read, fingerprint-checked records; for three rows per record the decoded values also equal
  an independent FP64 bit-field decode of the study payload, and their relative error against
  the BF16 checkpoint rows is reported. The attention-input parent's rows are checked through an
  explicit artifact-row to source-row formula, and its divisor word must equal the record's.

Python tests: `tests/artifact/test_expert_tiles.py` (layout against a scalar encoder written from
the byte definition, F3 unit known answer, real-shape round trips) and
`tests/targets/qwen3_8_flash_next/test_conversion.py` (study-record decode against a bit-level
stream writer, store addressing on a sparse full-size file, de-interleave, the attention-parent
row permutation, control packing, convolution and indexer transforms, int64 word pairs, the
inventory) and `tests/targets/qwen3_8_flash_next/test_study_dense.py` (synthetic records
in the study's NPZ format: FP8 row concatenation, the shared-expert record and multipliers,
fingerprint and damping rejection, the attention-input parent row permutation, the
hyper-connection records and their separate fingerprints).

## 9. Binding

The binder is `src/targets/qwen3_8_flash_next/impl/load/binder.{h,cpp}` (`LoadedModel`). It
binds every object of Sections 4 to 6 by name, format, layout and shape and rejects missing,
extra or mismatched objects.

- The BF16 token embedding is bound with `TensorPlacement::MappedHost`, in device-mapped pinned
  host memory, since each token reads one 5 KiB row.
- NVFP4 dense parents bind with their stored `d_w` and `input_scale_divisor = 0`: the artifact
  stores no activation divisor, and the Flash-Next NVFP4 projections accept that as A16-only
  weights, whose routes never quantize the activation.
- Every routed-expert record is copied into the device-mapped pinned host store
  (`ExpertHostStore`; record `i` of a codec tensor at `base + i * record_bytes`, 256-byte
  aligned) and addressed through the per-layer device expert tables; each layer's shared-expert
  record and its multipliers stay in device memory, as `moe_experts` `shared_record` and
  `shared_matrix_scales`.
- The n-gram hash constants are read on the host (`LoadedModel::ngram_constants`) for comparison
  with the table header; every other tensor is device-resident.
- Presence of `dflash/feature_projection` requires all 66 companion objects from Section 10.
  Partial companions, extra objects and mismatched descriptors are rejected. Companion views
  share the owning dense materialization's lifetime.

## 10. Optional DFlash2 companion

The append converter follows the [27B companion convention](qwen3.8-27b-artifact.md): one
`.ninfer` file with unchanged base object payloads and an appended `dflash/` namespace. Its
identity remains `qwen3.8-flash-next/f3-fp8-nvfp4`; the distinct filename is
`qwen3.8-flash-next-f3-fp8-nvfp4-dflash.ninfer`. The complete artifact contains 967 objects:
901 base objects and 66 companion tensors (32 NVFP4 matrices and 34 BF16 tensors).

The fixed drafter has five sliding-attention layers, hidden width 2560, intermediate width 7680,
32 query heads, eight KV heads, head width 128, sliding window 2048, RoPE theta 10000, RMSNorm
epsilon `1e-6`, block width eight and mask token 248077. Features are the plain four-stream means
after the target's MoE inject at zero-based layers `[5,15,25,35,45]`, concatenated in that order.
The target embedding is shared. These facts are checked against the checkpoint config.

Objects appear in the order below; the layer group repeats for `i=0..4`. Matrix shapes are
rows-first `[N,K]`. All matrices use `NVFP4/blockscale-k16-m128x4-v1`, with their stored positive
weight divisor and no activation divisor. Other tensors use `BF16/contiguous-le-v1`.

| Object under `dflash/` | Shape | Format |
|---|---|---|
| `feature_projection` | `[2560,12800]` | NVFP4 |
| `context_norm` | `[2560]` | BF16 |
| `layers/i/input_norm` | `[2560]` | BF16 |
| `layers/i/attention/query_key_value` | `[6144,2560]` | NVFP4 |
| `layers/i/attention/query_norm` | `[128]` | BF16 |
| `layers/i/attention/key_norm` | `[128]` | BF16 |
| `layers/i/attention/output` | `[2560,4096]` | NVFP4 |
| `layers/i/attention_conv/base_kernel` | `[2560,2,2]` | BF16 |
| `layers/i/attention_conv/kernel_projection` | `[640,2560]` | NVFP4 |
| `layers/i/post_attention_norm` | `[2560]` | BF16 |
| `layers/i/mlp/gate_up` | `[15360,2560]` | NVFP4 |
| `layers/i/mlp/down` | `[2560,7680]` | NVFP4 |
| `layers/i/mlp_conv/base_kernel` | `[2560,2,2]` | BF16 |
| `layers/i/mlp_conv/kernel_projection` | `[640,2560]` | NVFP4 |
| `final_norm` | `[2560]` | BF16 |
| `selector/hidden_projection` | `[256,2560]` | NVFP4 |
| `selector/predecessor_codebook` | `[256,248320]` | BF16 |
| `selector/successor_codebook` | `[256,248320]` | BF16 |

QKV concatenates checkpoint q, k and v rows: `[0,4096)`, `[4096,5120)` and `[5120,6144)`.
Gate/up concatenates gate then up. Convolution kernels preserve the checkpoint's `[2,2,2560]`
bytes and bind as Tensor `ne={2560,2,2}`: hidden is fastest, at offset
`d + 2560*(offset + 2*phase)`. Codebooks preserve checkpoint `[248320,256]` bytes and bind as
`ne={256,248320}`, with rank fastest. Those two direct layouts deliberately do not use the
ordinary rows-first-to-Tensor dimension reversal.

From the trusted repository root, using a Python 3.11 interpreter with PyTorch:

```bash
python3 -I -c 'import runpy, sys; from pathlib import Path; sys.path.insert(0, str(Path.cwd())); runpy.run_module("tools.convert.qwen3_8_flash_next.dflash", run_name="__main__")' \
  --base-artifact models/qwen3.8-flash-next-f3-fp8-nvfp4.ninfer \
  --dflash-model /home/battlefront/ai/qwen3.8-flash-next-dflash2 \
  --out models/qwen3.8-flash-next-f3-fp8-nvfp4-dflash.ninfer
```

The checkpoint is data only: config JSON and safetensors are read, never its Python code. The
converter validates all 81 source tensor descriptors, fuses the matrices, encodes NVFP4 on the
CPU, copies every base object payload unchanged, and writes `<out>.conversion.json`. Output must
not exist; a failed run can leave an incomplete file, which must be removed explicitly before
retrying. `tests/targets/qwen3_8_flash_next/test_dflash_conversion.py` checks append transport,
fused row order, direct kernel/codebook bytes and incompatible-input rejection.

The companion enables native chain-only DFlash2 serving with a full shared FP8 head and
adaptive K0..3. Conversion/binding tests are separate from the native same-column numerical-floor,
accepted-prefix, forced-picker and complete-state continuation qualifications in the
[Program plan](qwen3.8-flash-next-plan.md#native-dflash2-qualification-status).
The optimized proposal-head flag and explicit verification widths are unsupported.
