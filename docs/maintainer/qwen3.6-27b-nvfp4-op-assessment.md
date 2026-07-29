# Qwen3.6-27B NVFP4 Artifact Fusion and Op Support Assessment

## 1. Scope and status

This document evaluates two decisions that must be fixed before implementing Qwen3.6-27B NVFP4
Op support:

1. whether same-site Text projection weights in the additive NVFP4 artifact should be physically
   fused so an NVFP4 execution leaf consumes one `Weight` rather than multiple weight pointers;
2. whether the 27B GDN input-projection Op should produce Q, K, V, and Z together, and what that
   change means for the released Q4/Q5 artifact, the additive NVFP4 artifact, existing kernels,
   tests, and later target integration.

This document records the assessment and selected design. The fused converter contract has now
been implemented, the fixed NVFP4 artifact has been regenerated and verified, and the existing
Q4/Q5 GDN Op/call boundary has been revised to produce Z. The active storage facts are in
[`qwen3.6-27b-artifact.md`](qwen3.6-27b-artifact.md).

This implementation status does not claim NVFP4 execution support: the NVFP4/BF16 Attention
leaves, NVFP4 GDN leaves, NVFP4 target binding, and Engine consumption remain separate work.

The selected design is:

- the unpublished NVFP4 recipe physically fuses the full-attention Q/K/Gate/V site into one
  parent;
- the unpublished NVFP4 recipe physically fuses the GDN Q/K/V/Z site into one parent;
- both 27B artifact routes expose Q/K/V/Z as outputs of `gdn_input_proj` and
  `gdn_input_proj_conv_snapshot`;
- the released Q4/Q5 artifact bytes and object inventory remain unchanged;
- the first unpublished NVFP4 recipe uses `recipe_id = qwen3_6_27b_nvfp4-v1` and the fixed
  filename; there is no predecessor, compatibility identity, or recipe-version upgrade;
- `linear` does not gain an NVFP4 `[6144,5120]` problem solely for the GDN Z projection.

Kernel launch count remains an implementation choice. A physical single parent removes an
unnecessary multi-pointer storage boundary, but the Op contract does not promise that every `T`
executes in exactly one CUDA launch.

## 2. Facts that permit physical fusion

All matrices use logical `[N,K] = [output rows,input columns]`.

At every NVFP4 site proposed for fusion, the current recipe already validates a common weight
divisor and input divisor across the complete source set:

- a full-attention input site groups the source Q, K, and V projections; Q supplies both its query
  and output-gate row halves;
- a GDN input site groups the source QKV projection and the source Z projection.

The NVFP4 fused parts therefore have bit-identical positive FP32 `d_w` and `d_x` values. The
converter can concatenate their existing packed E2M1 rows and natural E4M3FN scale rows without
decoding, requantizing, or selecting a new divisor. The six early BF16 attention sites have no
divisors; their fusion is an exact concatenation of base-checkpoint BF16 words.

Both proposed matrices satisfy `blockscale-k16-m128x4-v1`:

| Role | Shape | `N / 128` | `K / 64` |
|---|---:|---:|---:|
| full-attention Q/K/Gate/V | `[14336,5120]` | 112 | 80 |
| GDN Q/K/V/Z | `[16384,5120]` | 128 | 80 |

Every component row range is also 128-row aligned:

- Attention: `6144`, `1024`, `6144`, `1024`;
- GDN: `2048`, `2048`, `6144`, `6144`.

The scale-plane swizzle is defined over 128-row tiles. The converter must concatenate selected
packed-code rows and natural scale rows and then encode the complete fused parent. It must not
concatenate previously encoded object payloads, because an encoded payload has its own plane
offsets and trailing divisor.

## 3. Selected NVFP4 artifact contract

### 3.1 Full-attention input parent

Replace both current input parents:

```text
attention/query_key [7168,5120]
attention/gate_value [7168,5120]
```

with:

```text
attention/query_key_gate_value [14336,5120]
row order = [query 6144, key 1024, output_gate 6144, value 1024]
```

The exact format assignment remains site-sensitive:

| Layers | Format | Layout |
|---|---|---|
| `3,7,11,15,19,23` | `BF16` | `contiguous-le-v1` |
| `27,31,35,39,43,47,51,55,59,63` | `NVFP4` | `blockscale-k16-m128x4-v1` |

The BF16 parent is formed from base-checkpoint BF16 words in the same row order. The source
q-projection remains head-interleaved `[query_256,output_gate_256]`; the converter first separates
the per-head halves, then forms `[query,key,output_gate,value]`.

Each NVFP4 attention-input site retains one rank-zero
`attention/input_projection/input_scale_divisor` immediately after the fused parent.

Consequences:

- the NVFP4 and BF16 27B single-parent `attn_input_proj` forms each receive one `Weight`;
- the released Q4/Q5 artifact keeps its existing two-parent Q4/Q5 form because its two parents use
  different numeric formats;
- no runtime concatenation, temporary packed parent, or second NVFP4 weight pointer is required.

### 3.2 GDN input parent

Replace both current NVFP4 parents:

```text
gdn/query_key [4096,5120]
gdn/value_z [12288,5120]
```

with:

```text
gdn/query_key_value_z [16384,5120]
row order = [query 2048, key 2048, value 6144, z 6144]
format = NVFP4
layout = blockscale-k16-m128x4-v1
```

All 48 GDN layers use this parent. Each site retains one rank-zero
`gdn/input_projection/input_scale_divisor` immediately after the parent.

Consequences:

- an NVFP4 GDN input leaf receives one complete `Weight`;
- no NVFP4 row-view construction is required for GDN V or Z;
- GDN Z is produced by the GDN input Op rather than by a later generic `linear`;
- every persisted NVFP4 parent in the selected recipe maps to exactly one persisted input divisor.

### 3.3 Unchanged objects

The following assignments do not change:

- full-attention and GDN output projections;
- every Text MLP `gate_up` and `down` parent;
- Text direct tensors, embedding, heads, and draft-head objects;
- all MTP and Vision objects;
- all six frontend resources;
- all 247 input-divisor sites and their FP32 words.

### 3.4 Deterministic inventory delta

The two fusion changes remove one object from each of 16 full-attention input sites and one object
from each of 48 GDN input sites.

| Inventory fact | Pre-fusion development state | Selected fused recipe | Delta |
|---|---:|---:|---:|
| Text-core tensors | 1018 | 954 | -64 |
| all tensor objects | 1365 | 1301 | -64 |
| resource objects | 6 | 6 | 0 |
| complete objects | 1371 | 1307 | -64 |
| `BF16` tensors | 597 | 591 | -6 |
| `FP32` tensors | 343 | 343 | 0 |
| `I32` tensors | 1 | 1 | 0 |
| `Q4G64_F16S` tensors | 55 | 55 | 0 |
| `Q5G64_F16S` tensors | 54 | 54 | 0 |
| `Q6G64_F16S` tensors | 3 | 3 | 0 |
| `W8G32_F16S` tensors | 7 | 7 | 0 |
| `NVFP4` tensors | 305 | 247 | -58 |
| `contiguous-le-v1` tensors | 941 | 935 | -6 |
| `row-split-k128-v1` tensors | 119 | 119 | 0 |
| `blockscale-k16-m128x4-v1` tensors | 305 | 247 | -58 |
| input-divisor tensors | 247 | 247 | 0 |

The six fused BF16 attention objects continue to represent the same eighteen selected BF16 source
linears. The number of BF16 Text matrix objects verified against the base source changes from 111
to 105, while the number of represented selected BF16 source linears remains 117.

Exact payload-region and file byte counts must be recorded from the regenerated artifact. They
must not be predicted in the contract because directory JSON length, payload alignment, eliminated
per-parent divisors, and eliminated inter-object gaps all contribute.

### 3.5 Implemented converter and verifier changes

Implementation of this selected artifact contract affects only the additive NVFP4 path:

- `tools/convert/qwen3_6_27b/inventory_nvfp4.py`
  - replace the two attention specs with one exact fused spec;
  - replace the two GDN specs with one exact fused spec;
  - insert each input divisor after its single parent;
  - update closed object, format, and layout counts.
- `tools/convert/qwen3_6_27b/recipe_nvfp4.py`
  - materialize Attention in `[query,key,output_gate,value]` order;
  - materialize GDN in `[query,key,value,z]` order;
  - preserve the existing exact common-`d_w` and common-`d_x` checks;
  - update one-to-one input-divisor coverage and recipe counts.
- `tools/convert/qwen3_6_27b/convert_nvfp4.py`
  - materialize the new fused BF16 attention object from base recipes;
  - use the first unpublished identity `qwen3_6_27b_nvfp4-v1`;
  - retain the fixed filename and `model_id`;
  - replace the prior internal development output rather than introducing a second recipe lane.
- `tools/convert/qwen3_6_27b/verify_nvfp4.py`
  - verify the new directory and counts;
  - compare every fused NVFP4 packed-code, scale, and divisor word with its selected sources;
  - compare the fused BF16 attention words with the base sources in exact row order.
- `tests/targets/qwen3_6_27b/test_nvfp4_recipe.py`
  - protect the exact new object names, shapes, order, counts, row order, divisor association, and
    source-word preservation.

The released converter, inventory, recipe, verifier, and artifact for
`qwen3_6_27b.ninfer` remain unchanged. The replaced NVFP4 bytes were disposable
internal-development output. They receive no alias, fallback, object-count dispatch, second recipe
identity, or second filename.

## 4. Selected GDN Q/K/V/Z Op contract

### 4.1 Stateless projection

The 27B GDN input Op must produce both the convolution input and the independent output gate:

```text
q[:,t] = Wq * x[:,t]       # [2048,T]
k[:,t] = Wk * x[:,t]       # [2048,T]
v[:,t] = Wv * x[:,t]       # [6144,T]
z[:,t] = Wz * x[:,t]       # [6144,T]

qkv = concat(q,k,v)         # [10240,T]
```

Public tensors are:

```text
x   BF16 [5120,T]
qkv BF16 [10240,T]
z   BF16 [6144,T]
T > 0
```

The artifact-specific weight forms are:

```text
released Q4/Q5:
    query_key Q4G64_F16S [4096,5120]
    value_z   Q5G64_F16S [12288,5120]

additive NVFP4:
    query_key_value_z NVFP4 [16384,5120]
```

Both forms implement the same mathematical Op and write the same observable outputs. They are
format-specific overloads or admissions behind the same `gdn_input_proj` semantic boundary.

The independent oracle exact-decodes the represented weight rows, starts from the values
represented by BF16 `x`, and evaluates all four projections with naive FP64 accumulation. It
compares the stored BF16 `qkv` and `z` outputs directly with those ideal values. Q/K/V/Z do not
gain an intermediate semantic activation-quantization or decoded-weight materialization boundary.

### 4.2 Projection, convolution, and snapshot

The 27B snapshot Op must likewise produce Z:

```text
p[:,t] = concat(Wq*x[:,t], Wk*x[:,t], Wv*x[:,t])
q,k,v and convolution snapshots =
    causal_width4_silu_and_snapshot(p, old_conv_state, initial_slot)
z[:,t] = Wz*x[:,t]
```

Its exact tensors are:

```text
x            BF16 [5120,T]
conv_weight  BF16 [10240,4]
conv_states  BF16 [10240,3,Slots]
initial_slot device I32 scalar
query        BF16 [2048,T]
key          BF16 [2048,T]
value        BF16 [6144,T]
z            BF16 [6144,T]
T > 0
Slots >= T
initial_slot in [0,Slots)
```

Only Q/K/V participate in convolution and state publication. Z is a direct projection output and
must not update or depend on convolution state.

The snapshot oracle evaluates Q/K/V projection, convolution, SiLU, every published state word, and
the independent Z projection as one complete logical Op. The former projected QKV buffer is not an
observable cast boundary. Z remains an observable BF16 output.

### 4.3 API consequence

The repository already has a single-parent `gdn_input_proj` overload and a single-parent snapshot
overload that output Z for the 35B W8 form. Those semantic forms should admit the 27B single-parent
NVFP4 geometry.

The prior 27B two-parent Q4/Q5 forms changed from:

```text
gdn_input_proj(x, query_key, value, qkv)
gdn_input_proj_conv_snapshot(x, query_key, value, ..., query, key, value)
```

to:

```text
gdn_input_proj(x, query_key, value_z, qkv, z)
gdn_input_proj_conv_snapshot(
    x, query_key, value_z, ..., query, key, value, z)
```

The second Q5 argument is the complete `[12288,5120]` parent, not a `[6144,5120]` V row view.

No NVFP4 `[6144,5120]` registration is then needed in generic `linear`. `linear` remains a
supported Op for its existing problems; it simply does not own GDN Z for either selected 27B
execution form.

## 5. Artifact-specific GDN handling

| Concern | Released Q4/Q5 artifact | Selected NVFP4 artifact |
|---|---|---|
| persisted input weights | Q4 `query_key [4096,5120]` plus Q5 `value_z [12288,5120]` | one NVFP4 `query_key_value_z [16384,5120]` |
| artifact rewrite | none | required |
| reason for physical form | one tensor cannot contain both Q4 and Q5 formats | all rows share NVFP4 format, layout, `d_w`, and `d_x` |
| Op outputs | QKV and Z | QKV and Z |
| weight arguments to leaf | two `Weight` values | one `Weight` value |
| runtime V/Z row views | must be removed from the call boundary; leaf interprets the full Q5 parent | not needed |
| generic NVFP4 `linear` for Z | not applicable | not needed |

The released artifact already contains the complete fused Q5 `value_z` parent in the required
`[value,z]` row order. Its bytes do not change. Only its eventual runtime consumption changes.

## 6. Kernel impact assessment

### 6.1 Attention

The current Q4/Q5 attention input implementation is not a template for the new physical form: it
receives two differently encoded parents and currently launches format-homogeneous work
separately.

Required future leaves are:

- one-parent NVFP4 `[14336,5120]` to four BF16 outputs;
- one-parent BF16 `[14336,5120]` to four BF16 outputs.

These are new execution leaves. They do not require rewriting the existing two-parent Q4/Q5
kernels. A one-launch direct-output route is the primary small-`T` performance candidate, but
larger-`T` launch decomposition remains subject to measurement.

### 6.2 NVFP4 GDN

There is no existing 27B NVFP4 GDN kernel to preserve or rewrite. It must be implemented directly
against the single `[16384,5120]` parent:

- ordinary `gdn_input_proj` writes QKV and Z;
- `gdn_input_proj_conv_snapshot` applies convolution/Snapshot only to Q/K/V rows and writes Z
  directly.

The implementation must not recover the former two-parent design by constructing row-view
`Weight` arguments or by invoking generic `linear` for Z.

### 6.3 Existing Q4/Q5 GDN

The prior Q4/Q5 execution leaves were hard-coded to a `[6144,5120]` V-only Q5 input. They have been
extended while retaining the common Q4/Q5 decode and MMA mechanisms.

For ordinary `gdn_input_proj`:

- small-`T` Q4 projection remains `[4096,5120]`;
- small-`T` Q5 work expands from 6144 V rows to the full 12288 V/Z rows and splits the result
  directly between the V slice of `qkv` and the separate Z allocation;
- grouped larger-`T` work uses two logical jobs over the one Q5 parent to write V and Z directly;
- the plan problem and admission geometry changed from `value_rows=6144` to
  `value_z_rows=12288`, while the convolution-channel count remains 10240.

For `gdn_input_proj_conv_snapshot`:

- the Q4 Q/K epilogue is unchanged semantically;
- the Q5 leaf evaluates all 12288 rows;
- rows `[0,6144)` feed convolution, SiLU, V output, and snapshot state;
- rows `[6144,12288)` bypass convolution and write Z;
- composed routes can use the revised projection Op followed by the existing convolution/state
  operations;
- fused small-`T` snapshot routes use a revised split epilogue. No separate `linear` owns Z.

The 35B W8 single-parent Q/K/V/Z leaves already expose the intended output structure and remain
unchanged. Their dimensions, weight format, and kernels are not reused as 27B implementations.

### 6.4 Direct answer on kernel rewrite

| Route | Assessment |
|---|---|
| NVFP4 Attention | new leaf; no existing NVFP4 kernel to rewrite |
| BF16 Attention exception | new leaf |
| NVFP4 GDN projection | new leaf; implement single-parent Q/K/V/Z directly |
| NVFP4 GDN snapshot | new leaf; implement Q/K/V convolution plus direct Z |
| Q4/Q5 GDN projection | revised to consume the full Q5 parent and write QKV/Z |
| Q4/Q5 GDN snapshot | revised fused epilogues write convolution outputs and direct Z |
| W8 GDN | no semantic or kernel change |
| generic `linear` | no NVFP4 GDN-Z leaf |

## 7. Op verification impact

The Q4/Q5 GDN fixtures now use the complete `[12288,5120]` parent, verify QKV and Z independently,
and retain the existing route-boundary/workspace checks. Remaining NVFP4 Op work must add:

- `tests/ops/test_attn_input_proj.cpp`
  - add exact 27B single-parent NVFP4 and BF16 shapes;
  - independently verify Q, K, Gate, and V row ranges;
  - preserve existing two-parent Q4/Q5 and 35B W8 cases.
- `tests/ops/test_gdn_input_proj.cpp`
  - add the one-parent `[16384,5120]` NVFP4 form.
- `tests/ops/test_gdn_input_proj_conv_snapshot.cpp`
  - add the one-parent NVFP4 form;
  - continue checking that only snapshot slots `[0,T)` change.
- affected plan tests and workspace-capacity tests
  - register the complete weight geometry without changing the semantic `T > 0` domain;
  - cover every route boundary actually selected by the implementation.

New execution-leaf sources and any renamed Q4/Q5 sources must also be registered in
`src/CMakeLists.txt`; this is build ownership for the Op implementation, not target or Engine
integration.

All NVFP4 numerical tests use the existing independent oracle rule: exact-decode persistent
`E2M1 * E4M3FN / d_w`, use represented BF16 activations, and accumulate the complete logical
formula naively in FP64. Private A4 activation quantization and `d_x` do not enter the oracle.

## 8. Target/runtime impact

The Q4/Q5 call-boundary changes are implemented:

- the existing Q4/Q5 binder retains the complete `value_z` `Weight` for the GDN projection leaf
  rather than publishing V and Z row-view weights as the call boundary;
- the 27B target leaf passes Z to both revised GDN Ops;
- the one shared family schedule consumes Z from the revised GDN input Op and removes the later
  standalone `gdn_output_gate_projection` call; the 35B route's corresponding no-op leaf is also
  removed;
- family workspace composition retains the existing Z allocation because Z spans the
  complete GDN mix lifetime;
- the existing Q4/Q5 artifact bytes and inventory are unchanged.

Remaining NVFP4 target integration must bind the two new single-parent object names, attach
validated `d_w`/`d_x`, and populate the same semantic target payload without request-time artifact
selection, model-id branches, object-count dispatch, or fallback paths. It does not create a second
family schedule.

## 9. Current delivery boundary

Completed:

1. fused additive NVFP4 inventory, recipe, converter tests, and verifier;
2. regenerated `qwen3_6_27b_nvfp4.ninfer` under `qwen3_6_27b_nvfp4-v1`, followed by complete
   source verification and replacement of the prior development output;
3. GDN Op contracts and independent Q4/Q5 oracles extended to Z;
4. Q4/Q5 GDN projection and snapshot leaves revised and qualified;
5. existing Q4/Q5 target binding and shared family call boundary revised without changing the
   original artifact.

Not delivered here:

1. one-parent NVFP4/BF16 Attention leaves;
2. one-parent NVFP4 GDN leaves;
3. NVFP4 target binding and Engine consumption.

Artifact correctness and Op correctness are separate gates. Regenerating a valid fused artifact
does not claim executable Engine support. The existing Q4/Q5 artifact remains wired through the
target; the additive NVFP4 artifact does not.
