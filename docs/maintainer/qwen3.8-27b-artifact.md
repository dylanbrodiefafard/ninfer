# Qwen3.8-27B artifact contract

This document defines the registered `qwen3.8-27b/groupwise-int` and `qwen3.8-27b/nvfp4`
artifacts: identity, persistent inventory, conversion entry point, and Engine binding. Model
mathematics, dimensions, frontend semantics, and state behavior are defined by
[`qwen3.6-27b-model.md`](qwen3.6-27b-model.md).

## 1. Identity

```text
filename   = qwen3_8_27b.ninfer
model_id   = qwen3.8-27b
weights_id = groupwise-int
target_key = qwen3_8_27b
recipe_id  = qwen3_8_27b-v1
```

```text
filename   = qwen3_8_27b_nvfp4.ninfer
model_id   = qwen3.8-27b
weights_id = nvfp4
target_key = qwen3_8_27b
```

Each artifact contains Text, the optimized MTP draft head, MTP, Vision, and six frontend
resources. A Qwen3.8-27B NVFP4 file may additionally append 66 DFlash2 objects under `dflash/`.
Those objects are inventory-conditional: current MTP-only 3.8 files and all 3.6-27B files remain
valid. `--spec dflash` requires the appended companion; `--spec mtp` on a reconverted file still
works (DFlash tensors stay host-placed). The identity is read from the version-2 artifact
directory; filenames and object counts do not select the target or weight profile.

## 2. Persistent inventory

The artifact contains 1118 tensors and six resources, for 1124 objects in total. Tensor format
counts are:

| Format | Tensors |
|---|---:|
| `BF16` | 582 |
| `FP32` | 96 |
| `I32` | 1 |
| `Q4G64_F16S` | 183 |
| `Q5G64_F16S` | 246 |
| `Q6G64_F16S` | 1 |
| `W8G32_F16S` | 9 |

The two vocabulary matrices use `W8G32_F16S` with `row-split-k128-v1`:

| Object | Logical shape |
|---|---|
| `text/token_embedding` | `[248320,5120]` |
| `text/output_head` | `[248320,5120]` |

Text layers use the Q4/Q5/Q6 groupwise assignment, `text/draft_head` uses Q4, the Vision patch
projection uses Q6, and the registered MTP and Vision-merger matrices use W8. Direct tensors use
`contiguous-le-v1`; all quantized tensors use `row-split-k128-v1`. The complete ordered inventory,
logical row views, and aliases are defined by
`tools/convert/qwen3_8_27b/inventory.py`.

Optional DFlash2 on the NVFP4 identity is defined by
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

## 3. Conversion

The converter consumes the Qwen3.8-27B BF16 checkpoint and writes one complete artifact:

```bash
python3 -m tools.convert.qwen3_8_27b.convert \
  --model /path/to/Qwen3.8-27B \
  --out out/qwen3_8_27b.ninfer \
  --device cuda
```

Before opening the output, it validates the checkpoint configuration, source tensor shapes and
dtypes, frontend resources, conversion recipes, and complete object plan. It writes the conversion
report to `out/qwen3_8_27b.ninfer.conversion.json`.

The converter owns and pins the official Qwen3.8 six-resource frontend profile. Relative to the
Qwen3.6-27B profile, `tokenizer.json`, `tokenizer_config.json`, and `chat_template.jinja` have
Qwen3.8-specific bytes; `generation_config.json`, `preprocessor_config.json`, and
`video_preprocessor_config.json` are byte-identical.

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

## 4. Engine binding

The registered mapping is:

```text
ArtifactIdentity(qwen3.8-27b, groupwise-int)
    -> WeightsProfile::GroupwiseIntW8Endpoints
    -> target qwen3_8_27b

ArtifactIdentity(qwen3.8-27b, nvfp4)
    -> WeightsProfile::Nvfp4
    -> target qwen3_8_27b
```

The groupwise profile binds the embedding and output head as W8 and the Text body through the
groupwise binding. Workspace selection follows the groupwise execution routes. The NVFP4 identity
reuses the Qwen3.6-27B NVFP4 profile unchanged: that profile already stores the vocabulary
endpoints as W8, which is the only groupwise difference between the two models, so the object
layouts are identical. The registry constructs the 27B `LoadedModel`, `SequencePlan`, and
`Program`, and reports `qwen3_8_27b/qwen3.8-27b/groupwise-int` or
`qwen3_8_27b/qwen3.8-27b/nvfp4` in the load summary.

The opt-in 27B live Engine tests (`ninfer_qwen3_6_27b_prefix_real_test`,
`ninfer_qwen3_6_27b_ram_real_test`) load these identities when
`NINFER_QWEN3_6_27B_WEIGHTS` or `NINFER_QWEN3_6_27B_NVFP4_WEIGHTS` points at the Qwen3.8
`.ninfer` (`qwen3_8_27b.ninfer` or `qwen3_8_27b_nvfp4.ninfer`). The env names follow the shared
27B package; they are not restricted to a Qwen3.6 filename.

`ninfer_qwen3_8_27b_dflash_real_test` loads a reconverted NVFP4+DFlash2 file from
`NINFER_QWEN3_8_27B_NVFP4_DFLASH_WEIGHTS`. Overlapping C=3 Graph DFlash2 greedy on three
distinct prompts must match C=1 DFlash of the same k for product k=4 and k=5 chain.
Those chain widths are T=5/T=6 SmallT, so they are not required to match MTP k=3
token-for-token. Packed verify launches Linear/LinearAdd/SwiGLU at per-sequence T=W so
NVFP4 residual and output-head routes match C=1 rather than packed T=W×B. NVFP4 target
verify at `T>=4` uses W4A4 attention, so no DFlash path is required to match ordinary
`T=1` A16 decode.
