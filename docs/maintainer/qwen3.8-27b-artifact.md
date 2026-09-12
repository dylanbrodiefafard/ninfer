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

On the NVFP4 identity the five MTP parents may be either that registered W8 layout or
`NVFP4` `blockscale-k16-m128x4-v1` (same `[N,K]`). Heads stay W8/Q4. The Engine binder accepts
either encoding. `tools/convert/qwen3_8_27b/convert_mtp_nvfp4.py` quantizes those five parents
from `Qwen/Qwen3.8-27B` BF16 with encoder `DFLASH2_NVFP4_MAXABS_TWOLEVEL_V2` and leaves Text,
Vision, and heads unchanged.

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

### Selective FP8 conversion (328 MiB recipe)

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

### Mixed FP8/NVFP4 storage profile

Different files under the same artifact identity must not exchange retained numerical state.
KV disk format v6 binds the opened artifact's local file generation; use a fresh cache directory
when switching to a composed mixed or selective variant. Old cache directories are not deleted.

The characterized neroued/Unsloth base uses the same `qwen3.8-27b/nvfp4`
identity with a distinct, validated Text inventory. Its vocabulary endpoints,
attention and GDN projections, and MLPs in layers 56–63 use
`FP8_E4M3FN_ROW_BF16S` / `row-scale-v1` (146 tensors). Layers 0–55 retain
NVFP4 MLP matrices and their input divisors. Each GDN A/B control is one BF16
`[96,5120]` parent bound as two contiguous `[48,5120]` views. Norms, Vision,
MTP and frontend resources retain the stored base values. Qualification used the
current 66-object `dflash/` companion with unchanged non-draft payloads. An old
`dflash2/` companion is not executed by a second runtime. The selective converter
above preserves the supported draft already present in its input shell.

FP8 prefill permits the qualified A8 implementation profile; decode and verify
use A16. Short GDN convolution snapshot/record (W≤16) stays A16 even when A8 is
allowed: the short A8 record profile failed its independent output criterion.
FP8 storage is not an FP8 KV-cache format, and does not alter p-less sampling,
DFlash acceptance, grammar or recovery semantics.

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

### Source conversion

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

Rewrite the five MTP Linear parents from `Qwen/Qwen3.8-27B` BF16 onto an existing
`qwen3.8-27b/nvfp4` shell (Ostfralla or any other file without those parents already
consumed as the BF16 source). Text, Vision, and heads are copied unchanged:

```bash
python3 -m tools.convert.qwen3_8_27b.convert_mtp_nvfp4 \
  --base-artifact /path/to/qwen3_8_27b_nvfp4.ninfer \
  --model /path/to/Qwen3.8-27B \
  --out /ssdpool2nvme/local_llm/models/qwen3.8-nvfp4-mtp-nvfp4-from-bf16/qwen3_8_27b_nvfp4.ninfer \
  --device cuda
```

The BF16 checkout must include the MTP sources (`mtp.fc.weight` and the layer-0
attention/MLP weights). Hugging Face revision `1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0`
places them in `model-00018-of-00018.safetensors`.

## 4. Engine binding

The registered mapping is:

```text
ArtifactIdentity(qwen3.8-27b, groupwise-int)
    -> WeightsProfile::GroupwiseIntW8Endpoints
    -> target qwen3_8_27b

ArtifactIdentity(qwen3.8-27b, nvfp4)
    -> WeightsProfile::Nvfp4 (W8 token embedding)
       or WeightsProfile::SelectiveFp8Nvfp4 (W8 endpoints, selected FP8 Text matrices)
       or WeightsProfile::MixedFp8Nvfp4 (row-scaled FP8 token embedding)
    -> target qwen3_8_27b
```

The groupwise profile binds the embedding and output head as W8 and the Text body through the
groupwise binding. Workspace selection follows the groupwise execution routes. The NVFP4 identity
with W8 endpoints reuses the Qwen3.6-27B NVFP4 profile: that profile stores the vocabulary
endpoints as W8, which is the only groupwise difference between the two models, so the object
layouts are identical. The mixed profile validates the complete inventory described
above after selecting from the embedding descriptor; filenames and tensor counts
are not dispatch inputs. MTP matrices on that identity bind as W8 or NVFP4. The registry constructs the 27B `LoadedModel`, `SequencePlan`, and
`Program`, and reports `qwen3_8_27b/qwen3.8-27b/groupwise-int` or
`qwen3_8_27b/qwen3.8-27b/nvfp4` in the load summary.

The opt-in 27B live Engine tests (`ninfer_qwen3_6_27b_prefix_real_test`,
`ninfer_qwen3_6_27b_ram_real_test`) load these identities when
`NINFER_QWEN3_6_27B_WEIGHTS` or `NINFER_QWEN3_6_27B_NVFP4_WEIGHTS` points at the Qwen3.8
`.ninfer` (`qwen3_8_27b.ninfer` or `qwen3_8_27b_nvfp4.ninfer`). The env names follow the shared
27B package; they are not restricted to a Qwen3.6 filename.

`ninfer_qwen3_8_27b_dflash_real_test` loads a reconverted NVFP4+DFlash2 file from
`NINFER_QWEN3_8_27B_NVFP4_DFLASH_WEIGHTS`. Sequential C=1 DFlash tokens are saved and used
as the C>1 oracle: overlapping C=4 Graph DFlash2 greedy on four distinct prompts must
match C=1 DFlash of the same k. The same C=1 and C>1 strings are also compared against
target-only T=1 decode. A later packed/T=1 greedy flip is not treated as row mixing;
`NINFER_DFLASH_TEST_RELAX_ORACLE=1` continues past that T=1 mismatch. Product k=4 and k=5
chain widths are T=5/T=6 SmallT, so they are not required to match MTP k=3 token-for-token.
NVFP4 GDN conv-record keeps the ordinary-decode reduction and BF16 3-tap history. B=1 W=4 uses
one fused SmallT weight pass; B=1 W=5/6 uses one grouped SmallT weight pass and a separate
sequential FP32 convolution. Other B=1 widths use the fused T=1 GEMV+FP32 conv route. Qualified
B=2..4 W=2/5 shapes group requests per SmallT weight pass; other B>1 widths use request-indexed
CTAs. Every route retains explicit BF16 history roundtrips.
Flattening that site to `T=W×B` W4A4 compose flipped
greedy column 0 versus C=1; `run_nvfp4_batched_matches_serial_fused` guards exact q/k/v/z and
valid-record identity for dense, ragged, and tree-parent W=4/5 C=1..4 and W=6 C=1 shapes.
Packed GDN recurrent then overlays T=1 snapshot `out` on scratch SSM. DFlash C>1
propose isolates each compact row as a C=1-shaped forward (`T=width`, `B=1`) so draft
Linears, SWA, and the draft head use sequential kernels. Eager propose resolves SWA's
direct/split route from the row frontier rather than the batch maximum; graph replay keeps
its fixed profile envelope. Target verify stays packed `B=batch` so graphs capture one 27B
forward; ReplaySSM records are `layer(g, 0, B)` and the feature sink covers the compact
lanes. Packed Linear/GDN-control sites normally launch at the C=1 width via
`linear_packed_sequences` / `packed_route_tokens` so C>1 does not select a different
T-specialized kernel (NVFP4 SmallT warp count, Q4 draft-head SmallT vs MMA, or A16↔W4A4).
The qualified W=5 C=2..4 residual Linear, fused NVFP4/BF16-control attention-input, and NVFP4
SwiGLU routes are explicit exceptions: their A16 T=10/15/20 schedules are bit-identical to C
separate T=5 panels. In particular, residual T=20 retains the T=5 panel reduction profile rather
than using the generic T=20 reduction.
Target GQA stays `[D, heads, W, B]`. SmallT uses one batched launch with request-indexed partial
CTAs and a batched reduction, preserving each request's arithmetic without taking the generic
`MultiBatch=true` route; p-less T=2 samples that reduction. Packed GDN conv-record similarly keeps
the W-local reduction: qualified W=2/5 B=2..4 shapes group weight replay through a private FP32
projection and other widths remain request-indexed. NVFP4 target verify at `T>=4` uses W4A4
attention, so no DFlash path is required to match ordinary `T=1` A16 decode.
