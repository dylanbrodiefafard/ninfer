# Qwen3.8-27B artifact contract

MTP tensors and conversion recipes remain artifact contracts. MTP execution is disabled;
ordinary decoding and DFlash2 are the available Engine routes.

This document defines the registered `qwen3.8-27b/nvfp4` artifact: identity, object names, order,
shapes, numeric formats, storage layouts, fused row order, aliases, frontend resources, storage
profiles, conversion entry points, and Engine binding. Common framing is defined in
[`artifact-container.md`](artifact-container.md), numeric semantics in
[`tensor-formats.md`](tensor-formats.md), byte packing in
[`storage-layouts.md`](storage-layouts.md), and model mathematics, dimensions, frontend
semantics, and state behavior in [`qwen3.6-27b-model.md`](qwen3.6-27b-model.md).

## 1. Identity

```text
filename   = qwen3_8_27b_nvfp4.ninfer
model_id   = qwen3.8-27b
weights_id = nvfp4
target_key = qwen3_8_27b
```

The artifact contains Text, the optimized MTP draft head, MTP, Vision, and six frontend
resources. A Qwen3.8-27B NVFP4 file may additionally append 66 DFlash2 objects under `dflash/`.
Those objects are inventory-conditional: files without them remain valid for ordinary decoding and
artifact validation. `--spec dflash` requires the appended
companion. MTP execution is disabled. The identity is read from the version-2 artifact
directory; filenames and object counts do not select the target or weight profile.

The first identity component is the base-model contract. The second selects the complete
weight-storage contract under it. The target key is a source/build identity and is not serialized.
Logical row views and aliases are not additional objects or JSON records.

The Python and C++ generic artifact registries recognize `NVFP4` and `blockscale-k16-m128x4-v1`,
and the C++ reader validates their descriptor geometry and payload range. The 27B package resolves
the complete identity to a closed target-private weights profile (Section 14), then its exact
binder consumes every object of the selected inventory. CLI, serving, benchmark, Text, Vision,
prefix reuse, and CUDA Graph execution use the public Engine and the registered `qwen3_8_27b`
target.

## 2. Fixed target facts

Sections 2 through 9 define the base inventory bound as `WeightsProfile::Nvfp4`: NVFP4 Text
matrices with W8 vocabulary endpoints, as in the published Ostfralla `qwen3_8_27b_nvfp4.ninfer`.
Section 6 admits an NVFP4 encoding of the MTP matrices, Section 10 defines the optional DFlash2
companion, and Section 11 defines the FP8 Text storage profiles under the same identity.

All matrix shapes use logical `[N,K] = [output rows,input columns]` notation. Quantization groups
along `K`.

| Fact | Value |
|---|---:|
| vocabulary matrix rows | 248320 |
| tokenizer-addressable IDs | 248077 (`0..248076`) |
| Text hidden width | 5120 |
| Text layers | 64 |
| Text MLP intermediate width | 17408 |
| full-attention layers | 16 |
| GDN layers | 48 |
| query heads / KV heads / head width | 24 / 4 / 256 |
| query / KV widths | 6144 / 1024 |
| GDN key heads × width | 16 × 128 = 2048 |
| GDN value heads × width | 48 × 128 = 6144 |
| GDN convolution channels / taps | 10240 / 4 |
| MTP layers | 1 full-attention dense-MLP layer |
| optimized draft-head rows | 131072 |
| Vision depth / hidden / intermediate width | 27 / 1152 / 4304 |
| Vision heads / patch input width | 16 / 1536 |
| Vision position rows | 2304 |
| Vision merger input / output | 4608 / 5120 |
| native context capacity | 262144 |

Full-attention Text layers are:

```text
3, 7, 11, 15, 19, 23, 27, 31, 35, 39, 43, 47, 51, 55, 59, 63
```

Every other Text layer is GDN. Layer numbers in object names are unpadded decimal integers.

The frontend token domain is `0..248076`. Both vocabulary matrices retain 248320 rows. The base
`tokenizer.json` carries added-token ids through 248069, and
`tokenizer_config.json::added_tokens_decoder` supplies ids `248070..248076`.

## 3. Numeric formats and physical row order

### 3.1 Format assignment

| Role | Format |
|---|---|
| token embedding | `W8G32_F16S` |
| full output head | `W8G32_F16S` |
| full-attention input `query_key_gate_value`, layers `3,7,11,15,19,23` | `BF16` |
| full-attention input `query_key_gate_value`, layers `27..63` | `NVFP4` |
| full-attention output, layers `3,7` | `BF16` |
| full-attention output, other full-attention layers | `NVFP4` |
| GDN input `query_key_value_z` | `NVFP4` |
| GDN output, layer 4 | `BF16` |
| GDN output, other GDN layers | `NVFP4` |
| Text MLP gate/up and down | `NVFP4` |
| site-level NVFP4 input divisors | `FP32` |
| Text norms, GDN convolution, and GDN A/B projections | `BF16` |
| GDN `A_log` and `dt_bias` | `FP32` |
| optimized MTP draft head | `Q4G64_F16S` |
| optimized MTP draft-head id map | `I32` |
| MTP matrices | `W8G32_F16S` (Section 6 also admits `NVFP4`) |
| MTP norms | `BF16` |
| Vision block input/expansion matrices | `Q4G64_F16S` |
| Vision block output/contraction matrices | `Q5G64_F16S` |
| Vision patch projection | `Q6G64_F16S` |
| Vision merger matrices | `W8G32_F16S` |
| all other Vision weights and biases | `BF16` |

There is no per-tensor override. Every `NVFP4` tensor uses `blockscale-k16-m128x4-v1`. Every
`Q4G64_F16S`, `Q5G64_F16S`, `Q6G64_F16S`, and `W8G32_F16S` tensor uses encoder profile
`MAXABS_F16_RECIP_RNE_V1` and storage layout `row-split-k128-v1`. Every direct tensor uses
`contiguous-le-v1`.

### 3.2 Fused row order

The following concatenations define physical output-row order:

- full-attention `query_key_gate_value`: `[query 6144,key 1024,output_gate 6144,value 1024]`;
- GDN `query_key_value_z`: `[query 2048,key 2048,value 6144,z 6144]`;
- Text and MTP MLP input: `[gate,up]`;
- MTP attention input: `[query,key,output_gate,value]`.

Within every source full-attention q-projection, each of the 24 heads stores
`[query_256,output_gate_256]`. The converter separates those per-head halves before constructing
the artifact parents.

Every GDN convolution is stored tap-major as `[4,10240]`.

## 4. Object namespace, order, and frontend resources

### 4.1 Namespace and writer order

- `text/` contains the embedding, 64 Text layers, final norm, full output head, and optimized MTP
  draft head.
- `mtp/` contains MTP-private tensors.
- `vision/` contains the Vision tower and merger.
- `frontend/` contains the six raw frontend resources.

Objects are written in this order:

1. the six frontend resources in Section 4.2;
2. `text/token_embedding`;
3. Text layers `0..63`, using the applicable object order in Sections 5.2 and 5.3;
4. `text/final_norm` and `text/output_head`;
5. `text/draft_head` and `text/draft_head_token_ids`;
6. the twelve MTP tensors in Section 6;
7. the Vision stem, blocks `0..26`, and merger in Section 7.

Readers bind by name. Object names contain no format spelling.

### 4.2 Frontend resources

The artifact contains exactly these `raw-bytes-v1` resources:

| Order | Object name | Source filename | Meaning |
|---:|---|---|---|
| 0 | `frontend/tokenizer.json` | `tokenizer.json` | base BPE vocabulary, merges, token bytes, and its added-token subset |
| 1 | `frontend/tokenizer_config.json` | `tokenizer_config.json` | complete added-token decoder, prefix, and special-token policy |
| 2 | `frontend/chat_template.jinja` | `chat_template.jinja` | registered Qwen template |
| 3 | `frontend/generation_config.json` | `generation_config.json` | default stop ids |
| 4 | `frontend/preprocessor_config.json` | `preprocessor_config.json` | image preprocessing limits and constants |
| 5 | `frontend/video_preprocessor_config.json` | `video_preprocessor_config.json` | video sampling and preprocessing |

The artifact retains these resource payloads byte-for-byte.

## 5. Text and optimized MTP draft inventory

### 5.1 Text-global objects

| Order | Object name | Shape | Format |
|---:|---|---|---|
| 0 | `text/token_embedding` | `[248320,5120]` | `W8G32_F16S` |
| after all layers | `text/final_norm` | `[5120]` | `BF16` |
| next | `text/output_head` | `[248320,5120]` | `W8G32_F16S` |

The embedding and output head are independent objects.

### 5.2 Full-attention Text layer

For every full-attention layer `l` in Section 2, emit these objects in order. A rank-zero `FP32`
input divisor follows its matrix parent only when that parent is `NVFP4` (Section 9.3):

| Order | Object-name pattern | Shape | Format |
|---:|---|---|---|
| 0 | `text/layers/{l}/input_norm` | `[5120]` | `BF16` |
| 1 | `text/layers/{l}/attention/query_key_gate_value` | `[14336,5120]` | `BF16` or `NVFP4` |
| 1a | `text/layers/{l}/attention/input_projection/input_scale_divisor` | `[]` | `FP32`, NVFP4 input only |
| 2 | `text/layers/{l}/attention/query_norm` | `[256]` | `BF16` |
| 3 | `text/layers/{l}/attention/key_norm` | `[256]` | `BF16` |
| 4 | `text/layers/{l}/attention/output` | `[5120,6144]` | `BF16` or `NVFP4` |
| 4a | `text/layers/{l}/attention/output_projection/input_scale_divisor` | `[]` | `FP32`, NVFP4 output only |
| 5 | `text/layers/{l}/post_attention_norm` | `[5120]` | `BF16` |
| 6 | `text/layers/{l}/mlp/gate_up` | `[34816,5120]` | `NVFP4` |
| 6a | `text/layers/{l}/mlp/gate_up_projection/input_scale_divisor` | `[]` | `FP32` |
| 7 | `text/layers/{l}/mlp/down` | `[5120,17408]` | `NVFP4` |
| 7a | `text/layers/{l}/mlp/down_projection/input_scale_divisor` | `[]` | `FP32` |

Layers `3,7` therefore contain 10 objects, layers `11,15,19,23` contain 11, and layers `27..63`
contain 12.

### 5.3 GDN Text layer

For every other layer `l` in `0..63`, emit these objects in order:

| Order | Object-name pattern | Shape | Format |
|---:|---|---|---|
| 0 | `text/layers/{l}/input_norm` | `[5120]` | `BF16` |
| 1 | `text/layers/{l}/gdn/a_log` | `[48]` | `FP32` |
| 2 | `text/layers/{l}/gdn/dt_bias` | `[48]` | `FP32` |
| 3 | `text/layers/{l}/gdn/convolution` | `[4,10240]` | `BF16` |
| 4 | `text/layers/{l}/gdn/a_projection` | `[48,5120]` | `BF16` |
| 5 | `text/layers/{l}/gdn/b_projection` | `[48,5120]` | `BF16` |
| 6 | `text/layers/{l}/gdn/query_key_value_z` | `[16384,5120]` | `NVFP4` |
| 6a | `text/layers/{l}/gdn/input_projection/input_scale_divisor` | `[]` | `FP32` |
| 7 | `text/layers/{l}/gdn/norm` | `[128]` | `BF16` |
| 8 | `text/layers/{l}/gdn/output` | `[5120,6144]` | `BF16` on layer 4, otherwise `NVFP4` |
| 8a | `text/layers/{l}/gdn/output_projection/input_scale_divisor` | `[]` | `FP32`, NVFP4 output only |
| 9 | `text/layers/{l}/post_attention_norm` | `[5120]` | `BF16` |
| 10 | `text/layers/{l}/mlp/gate_up` | `[34816,5120]` | `NVFP4` |
| 10a | `text/layers/{l}/mlp/gate_up_projection/input_scale_divisor` | `[]` | `FP32` |
| 11 | `text/layers/{l}/mlp/down` | `[5120,17408]` | `NVFP4` |
| 11a | `text/layers/{l}/mlp/down_projection/input_scale_divisor` | `[]` | `FP32` |

Layer 4 contains 15 objects and every other GDN layer contains 16.

### 5.4 Optimized MTP draft head

| Order | Object name | Shape | Format |
|---:|---|---|---|
| 0 | `text/draft_head` | `[131072,5120]` | `Q4G64_F16S` |
| 1 | `text/draft_head_token_ids` | `[131072]` | `I32` |

Row `i` of `text/draft_head` represents full-head row
`text/draft_head_token_ids[i]`. The ids are unique and lie in `0..248076`.

## 6. MTP inventory

The MTP module contains exactly twelve physical objects:

| Order | Object name | Shape | Format |
|---:|---|---|---|
| 0 | `mtp/input_projection` | `[5120,10240]` | `W8G32_F16S` |
| 1 | `mtp/embedding_norm` | `[5120]` | `BF16` |
| 2 | `mtp/hidden_norm` | `[5120]` | `BF16` |
| 3 | `mtp/layer/input_norm` | `[5120]` | `BF16` |
| 4 | `mtp/layer/attention/query_key_gate_value` | `[14336,5120]` | `W8G32_F16S` |
| 5 | `mtp/layer/attention/query_norm` | `[256]` | `BF16` |
| 6 | `mtp/layer/attention/key_norm` | `[256]` | `BF16` |
| 7 | `mtp/layer/attention/output` | `[5120,6144]` | `W8G32_F16S` |
| 8 | `mtp/layer/post_attention_norm` | `[5120]` | `BF16` |
| 9 | `mtp/layer/mlp/gate_up` | `[34816,5120]` | `W8G32_F16S` |
| 10 | `mtp/layer/mlp/down` | `[5120,17408]` | `W8G32_F16S` |
| 11 | `mtp/final_norm` | `[5120]` | `BF16` |

On this identity the five MTP matrix parents (`input_projection`, `query_key_gate_value`,
`output`, `gate_up`, and `down`) may instead use `NVFP4` `blockscale-k16-m128x4-v1` with the same
`[N,K]`; Section 12.3 defines the converter that writes them. The Engine binder accepts either
encoding. The vocabulary endpoints and the optimized draft head keep their W8 and Q4 formats.

MTP token embedding, full output head, and optimized proposal head are aliases listed in
Section 8.2.

## 7. Vision inventory

### 7.1 Vision stem

| Order | Object name | Shape | Format |
|---:|---|---|---|
| 0 | `vision/patch_embedding` | `[1152,1536]` | `Q6G64_F16S` |
| 1 | `vision/patch_embedding_bias` | `[1152]` | `BF16` |
| 2 | `vision/position_embedding` | `[2304,1152]` | `BF16` |

### 7.2 Vision transformer block

For every block `b` in `0..26`, emit these twelve objects:

| Order | Object-name pattern | Shape | Format |
|---:|---|---|---|
| 0 | `vision/layers/{b}/attention/qkv` | `[3456,1152]` | `Q4G64_F16S` |
| 1 | `vision/layers/{b}/attention/qkv_bias` | `[3456]` | `BF16` |
| 2 | `vision/layers/{b}/attention/output` | `[1152,1152]` | `Q5G64_F16S` |
| 3 | `vision/layers/{b}/attention/output_bias` | `[1152]` | `BF16` |
| 4 | `vision/layers/{b}/mlp/fc1` | `[4304,1152]` | `Q4G64_F16S` |
| 5 | `vision/layers/{b}/mlp/fc1_bias` | `[4304]` | `BF16` |
| 6 | `vision/layers/{b}/mlp/fc2` | `[1152,4304]` | `Q5G64_F16S` |
| 7 | `vision/layers/{b}/mlp/fc2_bias` | `[1152]` | `BF16` |
| 8 | `vision/layers/{b}/norm1/weight` | `[1152]` | `BF16` |
| 9 | `vision/layers/{b}/norm1/bias` | `[1152]` | `BF16` |
| 10 | `vision/layers/{b}/norm2/weight` | `[1152]` | `BF16` |
| 11 | `vision/layers/{b}/norm2/bias` | `[1152]` | `BF16` |

### 7.3 Vision merger

| Order | Object name | Shape | Format |
|---:|---|---|---|
| 0 | `vision/merger/fc1` | `[4608,4608]` | `W8G32_F16S` |
| 1 | `vision/merger/fc1_bias` | `[4608]` | `BF16` |
| 2 | `vision/merger/fc2` | `[5120,4608]` | `W8G32_F16S` |
| 3 | `vision/merger/fc2_bias` | `[5120]` | `BF16` |
| 4 | `vision/merger/norm/weight` | `[1152]` | `BF16` |
| 5 | `vision/merger/norm/bias` | `[1152]` | `BF16` |

No Vision deep-stack object exists.

## 8. Logical views and aliases

All entries in this section are views or aliases of the physical objects above. They are not extra
artifact objects.

### 8.1 Fused row views

| Parent object | Logical role | Stored row selection | Shape |
|---|---|---|---|
| `text/layers/{l}/attention/query_key_gate_value` | query | `[0,6144)` | `[6144,5120]` |
| same | key | `[6144,7168)` | `[1024,5120]` |
| same | output gate | `[7168,13312)` | `[6144,5120]` |
| same | value | `[13312,14336)` | `[1024,5120]` |
| `text/layers/{l}/gdn/query_key_value_z` | GDN query | `[0,2048)` | `[2048,5120]` |
| same | GDN key | `[2048,4096)` | `[2048,5120]` |
| same | GDN value | `[4096,10240)` | `[6144,5120]` |
| same | GDN z | `[10240,16384)` | `[6144,5120]` |
| `text/layers/{l}/mlp/gate_up` | MLP gate | `[0,17408)` | `[17408,5120]` |
| same | MLP up | `[17408,34816)` | `[17408,5120]` |
| `mtp/layer/attention/query_key_gate_value` | query | `[0,6144)` | `[6144,5120]` |
| same | key | `[6144,7168)` | `[1024,5120]` |
| same | output gate | `[7168,13312)` | `[6144,5120]` |
| same | value | `[13312,14336)` | `[1024,5120]` |
| `mtp/layer/mlp/gate_up` | MTP MLP gate | `[0,17408)` | `[17408,5120]` |
| same | MTP MLP up | `[17408,34816)` | `[17408,5120]` |

These views describe represented row roles. The fused Attention and GDN input Ops consume their
complete parents and do not construct row-view `Weight` values (Section 13).

### 8.2 Aliases

| Logical consumer role | Stored object or view |
|---|---|
| MTP token embedding | `text/token_embedding` |
| MTP full output head | `text/output_head` |
| MTP optimized proposal head | `text/draft_head` plus `text/draft_head_token_ids` |
| GDN channel-major convolution | transpose view of the stored `[4,10240]` convolution |

## 9. Inventory summary

### 9.1 Object counts

The counts in this section describe the base inventory with W8 MTP matrices and no DFlash2
companion.

| Component | Derivation | Tensor objects |
|---|---:|---:|
| Text globals | embedding + final norm + full head | 3 |
| full-attention Text layers | `2 × 10 + 4 × 11 + 10 × 12` | 184 |
| GDN Text layers | `15 + 47 × 16` | 767 |
| main Text excluding draft | `3 + 184 + 767` | 954 |
| optimized MTP draft head | weight + id map | 2 |
| MTP | fixed inventory | 12 |
| Vision stem | fixed inventory | 3 |
| Vision blocks | `27 × 12` | 324 |
| Vision merger | fixed inventory | 6 |
| Vision total | `3 + 324 + 6` | 333 |
| all tensors | `954 + 2 + 12 + 333` | 1301 |
| frontend resources | fixed inventory | 6 |
| complete artifact | `1301 + 6` | 1307 |

### 9.2 Numeric-format counts

| Format | Text | draft | MTP | Vision | Total |
|---|---:|---:|---:|---:|---:|
| `BF16` | 362 | 0 | 7 | 222 | 591 |
| `FP32` | 343 | 0 | 0 | 0 | 343 |
| `I32` | 0 | 1 | 0 | 0 | 1 |
| `Q4G64_F16S` | 0 | 1 | 0 | 54 | 55 |
| `Q5G64_F16S` | 0 | 0 | 0 | 54 | 54 |
| `Q6G64_F16S` | 0 | 0 | 0 | 1 | 1 |
| `W8G32_F16S` | 2 | 0 | 5 | 2 | 9 |
| `NVFP4` | 247 | 0 | 0 | 0 | 247 |
| total | 954 | 2 | 12 | 333 | 1301 |

The base inventory contains 935 direct tensors using `contiguous-le-v1`, 119 group-quantized tensors
using `row-split-k128-v1`, and 247 NVFP4 tensors using `blockscale-k16-m128x4-v1`.

### 9.3 NVFP4 parents and input divisors

The Text allocation produces 247 NVFP4 matrix parents and nine BF16 exception parents: the six
early `query_key_gate_value` parents, attention output layers 3 and 7, and GDN output layer 4. The
247 Text `FP32` tensors beyond the 96 GDN `A_log`/`dt_bias` controls are site-level input
divisors, one per NVFP4 parent:

| Object suffix under `text/layers/{l}/` | Layers | Count |
|---|---|---:|
| `attention/input_projection/input_scale_divisor` | NVFP4 attention-input layers | 10 |
| `attention/output_projection/input_scale_divisor` | NVFP4 attention-output layers | 14 |
| `gdn/input_projection/input_scale_divisor` | all GDN layers | 48 |
| `gdn/output_projection/input_scale_divisor` | GDN layers except 4 | 47 |
| `mlp/gate_up_projection/input_scale_divisor` | `0..63` | 64 |
| `mlp/down_projection/input_scale_divisor` | `0..63` | 64 |

Each scalar is inserted immediately after its matrix parent. Attention-input `d_x` maps to the
single `query_key_gate_value` parent and GDN-input `d_x` maps to the single
`query_key_value_z` parent. Every other site maps to its one named parent. Thus 247 scalars cover
all 247 NVFP4 parents one-to-one.

## 10. DFlash2 companion inventory

The optional DFlash2 companion on the NVFP4 identity is defined by
`tools/convert/qwen3_8_27b/inventory_dflash2.py` (66 tensors). W8 matrices use `W8G32_F16S`; the
Q4 sibling uses `Q4G64_F16S` for the same `[N,K]` set. The product NVFP4 sibling uses `NVFP4`
`blockscale-k16-m128x4-v1` for those matrices. Direct tensors (norms, conv `base_kernel`) stay
BF16. Default selector codebooks are BF16 NInfer `(256, 248320)` rank-fastest; `--dflash-codebook nvfp4`
stores them as NVFP4 weights of logical shape `[248320, 256]` (token = row, rank = K).
`base_kernel` is stored as `(5120, 2, 2)` in Hugging Face order without a permute.

| Object | Logical shape | Format (W8 sibling) |
|---|---|---|
| `dflash/feature_projection` | `[5120, 25600]` | `W8G32_F16S` |
| `dflash/context_norm` | `[5120]` | `BF16` |
| `dflash/layers/{0..4}/input_norm` | `[5120]` | `BF16` |
| `dflash/layers/{0..4}/attention/query_key_value` | `[6144, 5120]` | `W8G32_F16S` |
| `dflash/layers/{0..4}/attention/query_norm` | `[128]` | `BF16` |
| `dflash/layers/{0..4}/attention/key_norm` | `[128]` | `BF16` |
| `dflash/layers/{0..4}/attention/output` | `[5120, 4096]` | `W8G32_F16S` |
| `dflash/layers/{0..4}/attention_conv/base_kernel` | `[5120, 2, 2]` | `BF16` |
| `dflash/layers/{0..4}/attention_conv/kernel_projection` | `[1280, 5120]` | `W8G32_F16S` |
| `dflash/layers/{0..4}/post_attention_norm` | `[5120]` | `BF16` |
| `dflash/layers/{0..4}/mlp/gate_up` | `[34816, 5120]` | `W8G32_F16S` |
| `dflash/layers/{0..4}/mlp/down` | `[5120, 17408]` | `W8G32_F16S` |
| `dflash/layers/{0..4}/mlp_conv/base_kernel` | `[5120, 2, 2]` | `BF16` |
| `dflash/layers/{0..4}/mlp_conv/kernel_projection` | `[1280, 5120]` | `W8G32_F16S` |
| `dflash/final_norm` | `[5120]` | `BF16` |
| `dflash/selector/hidden_projection` | `[256, 5120]` | `W8G32_F16S` |
| `dflash/selector/{predecessor,successor}_codebook` | `[256, 248320]` | `BF16` |

Fused QKV row ranges are query `[0,4096)`, key `[4096,5120)`, value `[5120,6144)`. Gate/up on
`mlp/gate_up` are `[0,17408)` and `[17408,34816)`.

## 11. FP8 Text storage profiles

### 11.1 Retained-state compatibility

Different numerical tensor inventories under the same artifact identity must not exchange retained
state. KV disk format v8 binds retained-state tensor descriptors and selected encoded contents with
SHA-256; use a fresh cache directory when switching to a composed mixed or selective variant. Copies of identical
tensors remain compatible. Old cache directories are not deleted.

### 11.2 Mixed FP8/NVFP4 profile

The characterized neroued/Unsloth base uses the same `qwen3.8-27b/nvfp4`
identity with a distinct, validated Text inventory. Its vocabulary endpoints,
attention and GDN projections, and MLPs in layers 56–63 use
`FP8_E4M3FN_ROW_BF16S` / `row-scale-v1` (146 tensors). Layers 0–55 retain
NVFP4 MLP matrices and their input divisors. Each GDN A/B control is one BF16
`[96,5120]` parent bound as two contiguous `[48,5120]` views. Norms, Vision,
MTP and frontend resources retain the stored base values. Qualification used the
current 66-object `dflash/` companion with unchanged non-draft payloads. An old
`dflash2/` companion is not executed by a second runtime. The selective FP8 converter
(Section 12.1) preserves the supported draft already present in its input shell.

FP8 prefill permits the qualified A8 implementation profile; decode and verify
use A16. Short GDN convolution snapshot/record (W≤16) stays A16 even when A8 is
allowed: the short A8 record profile failed its independent output criterion.
FP8 storage is not an FP8 KV-cache format, and does not alter p-less sampling,
DFlash acceptance, grammar or recovery semantics.

### 11.3 Selective FP8 profile

The selective FP8 profile retains the base W8 vocabulary endpoints, separate BF16
GDN A/B controls, and all nine protected BF16 projection objects. Any of the base's
247 NVFP4 Text matrices may instead use `FP8_E4M3FN_ROW_BF16S` / `row-scale-v1`;
its paired NVFP4 input-divisor object must then be absent. The remaining objects
retain the base inventory. W8 endpoints plus at least one FP8 Text projection
select `SelectiveFp8Nvfp4`; full inventory validation still rejects incompatible
formats, missing objects, and unconsumed divisors. Startup reserves the maximum of
the existing NVFP4 and FP8 leaf workspaces; execution selects the qualified policy
from each bound weight. This permits isolated weight-quality experiments without
changing endpoints, norms, BF16 protections, draft matrices, or sampling semantics.
Artifact validity does not establish a selective recipe's quality benefit.
FP8 packed verification retains C=1-shaped residual and GDN projection/convolution
panels. Flattening GDN W×B would change the SmallT reduction and, at some widths,
introduce a BF16 projected materialization absent from C=1. Exact batched/serial
output and valid-record checks protect this arithmetic isolation in addition to
the independent mathematical output criterion.

## 12. Conversion

### 12.1 Selective FP8 conversion (328 MiB recipe)


Generate directly from the original BF16 checkpoint and an original NVFP4 `.ninfer`
shell; this CPU-only conversion does not need an intermediate FP8 model or bank:

```bash
python3 -m tools.convert.qwen3_8_27b.convert_selective_fp8 \
  --base-artifact /models/base_nvfp4_dflash.ninfer \
  --model /models/qwen3.8-27b-bf16 \
  --out /models/qwen3_8_27b_nvfp4_fp8_328mib_dflash.ninfer
```

Use the Python 3.11 conversion environment with PyTorch and safetensors. Paths above
are explicit input/output examples, not downloaded prerequisites. The output must
not already exist. A `.conversion.json` sidecar records the recipe and verifies all
written payloads. The original shell's draft (if any) is copied unchanged.

The fixed recipe promotes eight matrices, using zero-based layer indices:
attention output 11; fused attention Q/K/gate/V inputs 27, 31 and 51; MLP gate/up
and down at 62 and 63. FP8 codes use per-row BF16 multipliers with round-to-nearest-even;
the scale is represented before code rounding. All original BF16 protections,
W8 endpoints, norms, GDN controls and remaining NVFP4 matrices are preserved exactly.
The 328 MiB label describes added payload relative to the base, not total model size.
It is an empirically selected coding-quality/memory tradeoff, not a claim of global
perplexity optimality or elimination of reasoning loops.

### 12.2 DFlash2 append


DFlash2 is not quantized from the Qwen3.8-27B Text checkpoint. It is an append of the Hugging Face
DFlash2 companion onto an existing `qwen3.8-27b/nvfp4` `.ninfer`. The append does not rewrite
Text, MTP, or Vision bytes. The directory passed to `--dflash-model` must contain that companion's
`config.json` and `model.safetensors`.

Download a base artifact that has no `dflash/` objects (published Ostfralla NVFP4, or any other
`qwen3.8-27b/nvfp4` file):

```bash
hf download Ostfralla/Qwen3.8-27B-NVFP4-NInfer \
  qwen3_8_27b_nvfp4.ninfer \
  --local-dir models
```

Download the pinned DFlash2 companion:

```bash
hf download z-lab/Qwen3.8-27B-DFlash2 \
  --revision 50307d4c4cde6860d4eee73e2547cd786fe8e8a4 \
  --local-dir /path/to/Qwen3.8-27B-DFlash2
```

Product Engine path (NVFP4 matrices + BF16 selector codebooks):

```bash
python3 -m tools.convert.qwen3_8_27b.convert_nvfp4 \
  --base-artifact models/qwen3_8_27b_nvfp4.ninfer \
  --dflash-model /path/to/Qwen3.8-27B-DFlash2 \
  --dflash-format nvfp4 \
  --out out/qwen3_8_27b_nvfp4_dflash_nvfp4.ninfer \
  --device cuda
```

`--dflash-format w8` or `q4` writes a sibling that differs only in DFlash2 matrix `QType`.
`--dflash-codebook nvfp4` is optional, requires `--dflash-format nvfp4`, and is not the speed
path (~174 MiB saved, no tok/s win on k=5 greedy). The default codebook is BF16.
`--dflash-format nvfp4` uses the append-only encoder profile `DFLASH2_NVFP4_MAXABS_TWOLEVEL_V2`
(one positive FP32 `d_w` in the A16 Linear contract `W = e2m1 * e4m3 / d_w`, E4M3FN per-K16
scales, E2M1 codes). NVFP4 DFlash2 Linear geometries are A16-only. The converter pins Hugging Face
`z-lab/Qwen3.8-27B-DFlash2` revision `50307d4c4cde6860d4eee73e2547cd786fe8e8a4`. The algorithm pin
for `z-lab/dflash` `model.py` is `95c8aeca5e4b4c4f9c0c967c05ab89fa3ed24f4c` and is not a conversion
input.

### 12.3 NVFP4 MTP parents

Rewrite the five MTP Linear parents from `Qwen/Qwen3.8-27B` BF16 onto an existing
`qwen3.8-27b/nvfp4` shell (Ostfralla or any other file without those parents already
consumed as the BF16 source). Text, Vision, and heads are copied unchanged:

```bash
python3 -m tools.convert.qwen3_8_27b.convert_mtp_nvfp4 \
  --base-artifact /path/to/qwen3_8_27b_nvfp4.ninfer \
  --model /path/to/Qwen3.8-27B \
  --out /path/to/models/qwen3.8-nvfp4-mtp-nvfp4-from-bf16/qwen3_8_27b_nvfp4.ninfer \
  --device cuda
```

The BF16 checkout must include the MTP sources (`mtp.fc.weight` and the layer-0
attention/MLP weights). Hugging Face revision `1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0`
places them in `model-00018-of-00018.safetensors`.

## 13. NVFP4 weight and input-divisor ownership

Every NVFP4 parent uses `blockscale-k16-m128x4-v1` and contains its packed E2M1 code plane,
swizzled E4M3FN scale plane, and trailing FP32 weight divisor `d_w`. Its reconstruction and exact
payload geometry are defined in [`tensor-formats.md`](tensor-formats.md) and
[`storage-layouts.md`](storage-layouts.md).

Source `input_global_scale` is a serialized divisor `d_x`, not a multiplier. The artifact stores it
once per activation-quantization site as:

```text
shape   = []
format  = FP32
layout  = contiguous-le-v1
value   = bit-exact source FP32 word, finite and > 0
```

The source shape `[1]` becomes the rank-zero artifact scalar without changing its FP32 word. `d_x`
is calibration metadata for the private A4 activation-compute profile. For one finite
represented BF16 K-axis block of 16 input elements, that profile interprets it as:

```text
s_x       = E4M3FN(d_x * max_abs(block) / 6)
q_x       = E2M1(x * d_x / decode_e4m3fn(s_x))
decode(x) = decode_e2m1(q_x) * decode_e4m3fn(s_x) / d_x
```

`E4M3FN(y)` first saturates nonnegative finite `y` to `448`, then converts with
round-to-nearest-even. `E2M1(y)` is NVIDIA E2M1 round-to-nearest-even with finite saturation to
`[-6,6]`; ties choose the result whose retained significand bit is even, and input zero retains its
sign. The magnitude midpoints are `0.25,0.75,1.25,1.75,2.5,3.5,5.0`.

An all-zero block produces positive-zero `s_x` and positive-zero `q_x`. If a nonzero block scale
rounds to positive zero, every `q_x` in that block is also positive zero and no division by zero is
performed. The admitted execution input domain contains only finite represented BF16 activations.
The dynamic `s_x` is execution-local; the artifact persists neither `s_x`, `q_x`, `1 / d_x`, nor a
combined coefficient derived from `d_x` and `d_w`.

This activation quantization describes a private implementation profile, not the mathematical Op
oracle and not an observable intermediate. The independent correctness oracle starts from the
represented BF16 public activation, exact-decodes the persistent NVFP4 weight as
`E2M1 * E4M3FN / d_w`, and evaluates the complete logical Op with naive FP64 accumulation. It does
not quantize the activation, introduce a BF16 materialization of the decoded weight, or use `d_x`.
Activation quantization, staging, accumulation, and output rounding contribute only to the
production route's error against the named output criterion. Op precision tests therefore compare
only observable outputs with this oracle; they do not inspect `s_x`, `q_x`, use of `d_x`, or the
selected instruction path. Performance evidence separately qualifies the A4 route.

Section 9.3 lists the 247 site-level divisor objects and their parents.

The runtime `Weight` representation uses `QType::NVFP4`,
`QuantLayout::BlockScaleK16M128x4`, E4M3FN scale dtype, `weight_scale_divisor`, and
`input_scale_divisor`; both divisor fields default to the invalid sentinel `0.0F`. The generic
`materialized_weight` path deliberately rejects NVFP4 because only the target-private
binding plan can validate `d_w` and `d_x` and combine them into a consumable immutable weight.
Other `QType` values never read either divisor field.

A consumable NVFP4 `Weight` has `group_size = group = 16`, `qdata` at the E2M1 code plane,
`scales` at the swizzled E4M3FN scale plane, `qhigh = nullptr`, `high_plane_bytes = 0`, and
`payload_bytes` covering the complete parent payload including its trailing `d_w`. Before
`Binder::finish()` and the end of the `Reader` lifetime, target-private `plan_load`:

1. exact-read and validate `d_w` from every NVFP4 parent tail;
2. exact-read and validate the associated site-level `d_x`;
3. retain both host FP32 values in the target-private binding plan.

The site scalar is validation-only and receives no independent device allocation. During loaded
model construction, the binder writes the parent `d_w` and the site's `d_x` into every associated
immutable NVFP4 `Weight`; missing values must remain an error and must never default to `1.0`.

The fused Attention and GDN input Ops consume their complete NVFP4 parents. Their target-private
binding and call boundaries do not construct query/key/gate/value/z `Weight` row views and do not
consult the artifact directory during execution.

The `attn_input_proj`, `gdn_input_proj`, `linear`, `linear_add`, and `linear_swiglu` signatures
consume their registered immutable weights. Each NVFP4 leaf reads
`d_x` and `d_w` from its complete parent `Weight`; the Op wrapper derives that weight's
`1 / (d_x * d_w)` as a leaf-private kernel argument. That coefficient is not another artifact
object, another `Weight` field, or a new Op parameter.

The nine BF16 Text exceptions use the same fused semantic Op boundaries. The Op layer admits
their single-parent BF16 weights through `attn_input_proj` for the six early
`query_key_gate_value` parents and through `linear_add` for attention output layers 3 and 7 and GDN
output layer 4.

All NVFP4 Text parents and BF16 exceptions described above are bound and executable. Prefill and
ordinary decode pass `AllowA4` for NVFP4 weights. Speculative target verification passes `AllowA8`
to eligible ordinary and GDN projections at every verification width.
The BF16 exceptions retain `A16Only`. Each semantic Op resolves its qualified route from the
exact geometry and T. Vision uses its registered storage and execution paths; MTP tensors are
validation inputs only. The 247 site-level `d_x` scalars are consumed and validated but receive
no device allocation.

## 14. Engine binding

The registered mapping is:

```text
ArtifactIdentity(qwen3.8-27b, nvfp4)
    -> WeightsProfile::Nvfp4 (W8 token embedding)
       or WeightsProfile::SelectiveFp8Nvfp4 (W8 endpoints, selected FP8 Text matrices)
       or WeightsProfile::MixedFp8Nvfp4 (row-scaled FP8 token embedding)
    -> target qwen3_8_27b
```

`WeightsProfile::Nvfp4` binds the base inventory of Sections 2 through 9. The package resolves the
profile from the identity, the `text/token_embedding` descriptor, and, for W8 endpoints, the Text
projection descriptors; the binder then validates the complete selected inventory. Filenames and
tensor counts are not dispatch inputs, no binder is tried and caught on failure, and no artifact
string branch reaches request execution. The registry resolves this profile before sequence or
load planning and passes the same typed value to both. The sequence plan retains it for
profile-specific workspace sizing, the loaded model retains it for immutable payload construction,
and Program construction requires equality. MTP matrices on that identity bind as W8 or NVFP4. The registry constructs the 27B `LoadedModel`, `SequencePlan`, and
`Program`, and reports `qwen3_8_27b/qwen3.8-27b/nvfp4` in the load summary.

Every storage profile materializes `text/token_embedding` (1.26 GiB as W8) in pinned host memory
(`TensorPlacement::MappedHost`) rather than VRAM: the embedding is only row-gathered, and the gather
kernels read it through its unified device address. The load summary reports it as
`mapped_host_bytes`; the freed VRAM goes to the automatic KV pool. `text/output_head` stays in VRAM.

The opt-in live Engine tests (for example `ninfer_qwen3_6_27b_prefix_real_test` and
`ninfer_qwen3_6_27b_ram_real_test`) load this identity from `NINFER_QWEN3_8_27B_NVFP4_WEIGHTS`,
the path to `qwen3_8_27b_nvfp4.ninfer`.

`ninfer_qwen3_8_27b_dflash_real_test` loads a reconverted NVFP4+DFlash2 file from
`NINFER_QWEN3_8_27B_NVFP4_DFLASH_WEIGHTS`. Sequential C=1 DFlash tokens are saved and used
as the C>1 oracle: overlapping C=4 Graph DFlash2 greedy on four distinct prompts must
match C=1 DFlash of the same k and output budget. Comparisons against target-only T=1
decode are diagnostic: packed/T1 greedy flips do not by themselves establish row mixing.
Actual-verifier logit licensing checks candidate choices separately. Product k=4 and k=5
chain widths are T=5/T=6 SmallT, so they are not required to match MTP k=3 token-for-token.
NVFP4 GDN conv-record under A4 (W=5..16) or A8 (W=4..16) aggregates projection across B requests
without introducing a BF16 current-projection intermediate. Convolution consumes FP32 current
projection and BF16 three-tap history; saved records/history remain BF16. A8 uses row-scaled
E4M3 activations, exact expansion of the stored E2M1 codes and separately scaled K16 partials.
It neither changes the artifact nor requantizes its weights.
A8 W2–6 fuses projection with convolution and record publication through a CTA-local FP32
tile; it needs only activation-quantization workspace. Wider A8 and A4 record widths retain
the global FP32 intermediate.
Under A16, B=1 W=4 uses one fused SmallT pass; B=1 W=5/6 uses a grouped pass and separate
FP32 convolution. B=2..4 W=2/5 groups requests, including one W5/C3 group. W6/C2 and C4 use
pairs; W6/C3 stays request-indexed. Other B=1 widths use fused T1 GEMV+FP32 conv, and other
B>1 widths use request-indexed CTAs. Independent record tests cover dense/ragged/tree inputs,
carried history and every valid prefix; the A16 pairing change additionally matches legacy
T1 outputs exactly. Quantized projection checks use their own mathematical oracle, not A16
bit parity or the superseded BF16-compose record route.
Packed GDN recurrent then carries register-resident state through the width-one transition for `out`. DFlash C>1
propose isolates each compact row as a C=1-shaped forward (`T=width`, `B=1`) so draft
Linears, SWA, and the draft head use sequential kernels. Eager propose resolves SWA's
direct/split route from the row frontier rather than the batch maximum; graph replay keeps
its fixed profile envelope. Target verify stays packed `B=batch` so graphs capture one 27B
forward; ReplaySSM records are `layer(g, 0, B)` and the feature sink covers the compact
lanes. Batched verification follows the batch-size consistency rule in
[qwen3.6-27b-model.md](qwen3.6-27b-model.md) §8 item 5: precision never depends on the batch, while
rounding may. Packed Linear/GDN-control sites normally launch at the C=1 width via
`linear_packed_sequences` / `packed_route_tokens`, so C>1 never selects a different activation
precision (A16↔W4A4); keeping the C=1 kernel also keeps C=1 rounding, which is not required. The DFlash qkv, attention-output
and feature projections (BF16 activations on tensor cores) and the Q4 draft head reduce every
column in one T-independent order, so they aggregate packed drafts of any width in one pass.
Qualified NVFP4 and BF16-control attention-input, residual and MLP routes aggregate through
W8. The selected default uses A8
for both ordinary and GDN verification projections at every verification width (W2–W6), so a
request's precision profile depends on neither its draft width nor the batch. Prefill and ordinary
decode retain their text policy.
The eligible verification post-mixer fuses RMSNorm and A8 quantization before gate/up/SwiGLU,
preserving the BF16 normalized-input rounding boundary. The down projection stays separate.
The quality/default decision is recorded in the performance reference. A16 residual T20
retains the T5 panel reduction profile.
Target GQA stays `[D, heads, W, B]`. SmallT uses one batched launch with request-indexed partial
CTAs and a batched reduction without taking the generic `MultiBatch=true` route. For NVFP4 KV
windows up to 32K the split count is sized per batch to one resident wave, so attention rounding,
not precision, varies with the batch; p-less temperature2 samples that reduction. These projection profiles
share the existing NVFP4 attention Q codec. No DFlash arithmetic profile is required to match
ordinary T1 decode token-for-token; same-profile isolation and state/publication correctness
remain required.
