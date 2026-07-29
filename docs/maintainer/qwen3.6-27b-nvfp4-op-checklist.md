# Qwen3.6-27B NVFP4 Op Support Checklist

## 1. Purpose and boundary

This document is the active checklist for making the fixed
`qwen3_6_27b_nvfp4.ninfer` Text weights consumable by repository-internal semantic Ops. It starts
from the artifact contract in
[`qwen3.6-27b-artifact.md`](qwen3.6-27b-artifact.md) and lists only Op-level functional
requirements:

- the exact semantic Op that owns each changed Text-linear role;
- persistent weight format and complete parent shape;
- public input and output tensors and their dimensions;
- the format/problem admissions and workspace-query coverage that must be added;
- numerical and behavioral evidence required to mark an item complete.

This checklist does not specify CUDA kernels, instruction shapes, launch counts, schedules,
activation staging, or performance routes. It also does not cover artifact binding, target payloads,
Engine selection or loading, family scheduling, CLI, or serving. Those concerns cannot be used to
mark an Op item complete.

All non-Text payloads retain the original artifact's formats and already supported Op contracts.
MTP, Vision, embedding, output heads, Text norms, GDN control projections, convolution, recurrent
mixing, RoPE, attention, and pointwise operations therefore require no new Op admission for this
artifact and are outside this checklist.

## 2. Fixed facts

Matrix weights use logical `[N,K] = [output rows,input columns]`. Public activation and result
matrices use `[rows,T]`, where `T > 0`. All public activation, projection-result, and residual
tensors named below are contiguous `BF16`.

The artifact changes exactly these Text matrix roles:

| Artifact role | Persistent parent `[N,K]` | NVFP4 sites | BF16 exception sites | Semantic Op |
|---|---:|---:|---:|---|
| full-attention Q/K/Gate/V input | `[14336,5120]` | 10 | 6 | `attn_input_proj` |
| full-attention output | `[5120,6144]` | 14 | 2 | `linear_add` |
| GDN Q/K/V/Z input | `[16384,5120]` | 48 | 0 | `gdn_input_proj`; `gdn_input_proj_conv_snapshot` |
| GDN output | `[5120,6144]` | 47 | 1 | `linear_add` |
| Text MLP gate/up | `[34816,5120]` | 64 | 0 | `linear_swiglu` |
| Text MLP down | `[5120,17408]` | 64 | 0 | `linear_add` |
| total physical parents |  | 247 | 9 |  |

The layer assignments are fixed:

- attention input is BF16 on layers `3,7,11,15,19,23` and NVFP4 on
  `27,31,35,39,43,47,51,55,59,63`;
- attention output is BF16 on layers `3,7` and NVFP4 on the other 14 full-attention layers;
- every one of the 48 GDN input parents is NVFP4;
- GDN output is BF16 on layer `4` and NVFP4 on the other 47 GDN layers;
- both Text MLP parents are NVFP4 on all 64 layers.

At the Op boundary, an artifact `NVFP4` matrix is one `Weight` with:

```text
qtype                 = QType::NVFP4
layout                = QuantLayout::BlockScaleK16M128x4
group_size / group    = 16
scale_dtype           = DType::FP8_E4M3FN
weight_scale_divisor  = finite and > 0
input_scale_divisor   = finite and > 0
```

Its code plane, scale plane, logical and padded shapes, and complete-parent payload must be valid.
An artifact `BF16` matrix is one `Weight` with `QType::BF16_CTRL`,
`QuantLayout::Contiguous`, and the exact complete-parent shape. Divisor fields do not apply to the
BF16 form.

The Op oracle exact-decodes an NVFP4 weight as persistent
`E2M1 * E4M3FN / weight_scale_divisor`, starts from the values represented by public BF16 inputs,
and evaluates the complete logical formula independently with naive FP64 accumulation. Private
activation quantization and `input_scale_divisor` do not alter that formula or create an observable
intermediate.

The current Op baseline already provides `QType::NVFP4`, its block-scale layout identity, exact
decode fixtures, the revised 27B Q4/Q5 GDN Q/K/V/Z contracts, and single-parent W8 Attention/GDN
overloads. It does not yet admit any of the eight exact NVFP4/BF16 problems below, so every item
starts unchecked.

## 3. Required format/problem registrations

There are five semantic Op surfaces and eight exact format/problem registrations to complete. The
site counts describe artifact coverage; they are not runtime dispatch inputs.

| ID | Semantic Op | Weight format | Exact parent `[N,K]` | Artifact roles | Status |
|---|---|---|---:|---|---|
| A1 | `attn_input_proj` | `NVFP4` | `[14336,5120]` | attention input, 10 sites | [ ] |
| A2 | `attn_input_proj` | `BF16` → `BF16_CTRL` | `[14336,5120]` | attention input, 6 sites | [ ] |
| G1 | `gdn_input_proj` | `NVFP4` | `[16384,5120]` | GDN input, 48 sites | [ ] |
| G2 | `gdn_input_proj_conv_snapshot` | `NVFP4` | `[16384,5120]` | GDN verify input, same 48 sites | [ ] |
| M1 | `linear_swiglu` | `NVFP4` | `[34816,5120]` | MLP gate/up, 64 sites | [ ] |
| R1 | `linear_add` | `NVFP4` | `[5120,6144]` | attention output and GDN output, 61 sites | [ ] |
| R2 | `linear_add` | `NVFP4` | `[5120,17408]` | MLP down, 64 sites | [ ] |
| R3 | `linear_add` | `BF16` → `BF16_CTRL` | `[5120,6144]` | attention/GDN output exceptions, 3 sites | [ ] |

No generic `linear` registration is an artifact-level requirement for these changed Text weights.
This conclusion does not come from the number or format of weight parents: the matrix portion of
each projection is still a linear map. It follows from the observable output contract of each Op.

`linear` accepts one weight `[N,K]` and writes one contiguous BF16 result `Y [N,T]`. Dimension zero
is stored fastest, so a contiguous BF16 `Y [N,T]` has:

```text
Y.nb[0] = 2
Y.nb[1] = 2 * N
```

A row slice `Y[start:start+R,:]` retains `Y.nb[1]`. For `T > 1` and `R < N`, that view is not a
contiguous `[R,T]`, whose required column stride is `2 * R`. The slice is contiguous at `T=1`, but
every Text Op here admits every positive `T`, so that special case cannot define the contract.

The actual semantic boundaries are therefore:

- `attn_input_proj` writes four independently contiguous outputs rather than one packed result;
- `gdn_input_proj` writes independently contiguous QKV and Z outputs;
- `gdn_input_proj_conv_snapshot` additionally owns convolution and snapshot-state mutation;
- `linear_swiglu` exposes only the complete nonlinear result, not a BF16 gate/up projection;
- `linear_add` updates the BF16 residual in place and does not expose a BF16 projected tensor.

In particular, attention and GDN output projections update the residual through `linear_add`, the
MLP gate/up projection and SwiGLU form one `linear_swiglu` Op, and GDN Z comes from the complete GDN
input parent. No standalone NVFP4 `linear [6144,5120]` is required for GDN Z. An implementation may
privately compose an Op from other primitives, but that choice neither adds nor removes a
functional checklist item.

## 4. Attention input projection

### A1 — single-parent NVFP4 `attn_input_proj`

- [ ] Admit one complete NVFP4 `query_key_gate_value` weight `[14336,5120]`; do not expose four
  weight row views or multiple weight arguments.
- [ ] Accept `x BF16 [5120,T]`, with every positive `T`.
- [ ] Write four distinct contiguous BF16 outputs:

  ```text
  q    [6144,T]
  gate [6144,T]
  k    [1024,T]
  v    [1024,T]
  ```

- [ ] Interpret physical parent rows exactly as:

  ```text
  query       [0,6144)
  key         [6144,7168)
  output_gate [7168,13312)
  value       [13312,14336)
  ```

  The public argument order `q, gate, k, v` is deliberately different from the physical row order.
- [ ] Implement the four independent logical projections `q=Wq*x`, `k=Wk*x`,
  `gate=Wgate*x`, and `v=Wv*x`. There is no observable packed `[14336,T]` output.
- [ ] Reject an invalid NVFP4 layout, shape, plane geometry, or non-positive/non-finite divisor.
- [ ] Preserve the existing two-parent 27B Q4/Q5 and single-parent 35B W8 admissions and outputs.
- [ ] Add independent-oracle coverage for all four row ranges and verify that output allocations
  are neither exchanged nor overlapped.

The existing single-parent C++ signature already expresses this semantic form; A1 extends its exact
format/geometry admission rather than creating a second NVFP4-specific public name.

The matrix arithmetic is exactly equivalent to forming `Y = W*x` with
`Y BF16 [14336,T]` and taking the four physical row ranges above. That equivalence does not make
generic `linear` the same Op: for `T > 1`, those ranges are strided views of the packed `Y`, whereas
A1 promises four separately contiguous tensors. Weight row order determines the logical mapping;
output allocation order does not.

### A2 — single-parent BF16 `attn_input_proj`

- [ ] Admit one complete contiguous `BF16_CTRL` `query_key_gate_value` weight
  `[14336,5120]`.
- [ ] Use the same input, output, row-order, aliasing, and `T > 0` contract as A1.
- [ ] Evaluate each projection from the represented BF16 weight and activation values; no
  quantization divisor is accepted or used.
- [ ] Cover all four row ranges with the independent oracle and retain the existing Q4/Q5 and W8
  regressions.

A2 is required even though it contains no NVFP4 weight: without it, the six BF16 attention-input
parents in the NVFP4 artifact do not have an Op consumer. Its distinction from BF16 `linear` is the
same four-contiguous-output contract as A1, not different projection arithmetic.

## 5. GDN input projection

The complete GDN input parent has this physical row order:

```text
query [0,2048)
key   [2048,4096)
value [4096,10240)
z     [10240,16384)
```

Q, K, and V form the 10240 causal-convolution channels. Z is an independent output gate and never
participates in convolution state.

### G1 — single-parent NVFP4 `gdn_input_proj`

- [ ] Admit one complete NVFP4 `query_key_value_z` weight `[16384,5120]`.
- [ ] Accept `x BF16 [5120,T]`, with every positive `T`.
- [ ] Write distinct contiguous outputs:

  ```text
  qkv BF16 [10240,T]  # row order [query 2048, key 2048, value 6144]
  z   BF16 [6144,T]
  ```

- [ ] Evaluate all four projections from the same public `x`; Z is not a later generic Linear Op.
- [ ] Reject row-view substitutes, invalid NVFP4 metadata, and output aliasing.
- [ ] Preserve the current 27B two-parent Q4/Q5 form and the 35B single-parent W8 form.
- [ ] Add independent-oracle coverage that checks Q, K, V, and Z separately.

The existing single-parent overload already has the correct semantic argument list. G1 extends its
admitted weight format and the exact 27B geometry.

As with Attention, the matrix arithmetic can be written as `Y = W*x`, followed logically by
`qkv = Y[0:10240,:]` and `z = Y[10240:16384,:]`. For `T > 1`, both are strided row views of one
contiguous `Y [16384,T]`; G1 instead promises two independently contiguous outputs. This layout
effect, rather than a different dot-product formula, distinguishes G1 from generic `linear`.

### G2 — single-parent NVFP4 `gdn_input_proj_conv_snapshot`

- [ ] Admit the same complete NVFP4 parent `[16384,5120]` and `x BF16 [5120,T]`.
- [ ] Accept the existing 27B snapshot operands:

  ```text
  conv_weight BF16 [10240,4]
  conv_states BF16 [10240,3,Slots]
  initial_slot device I32 scalar
  T > 0
  Slots >= T
  initial_slot in [0,Slots)
  ```

- [ ] Write:

  ```text
  query BF16 [2048,T]
  key   BF16 [2048,T]
  value BF16 [6144,T]
  z     BF16 [6144,T]
  ```

- [ ] Apply the width-four causal convolution and SiLU only to the projected Q/K/V channels.
  Projected Z bypasses convolution and snapshot state.
- [ ] Publish the resulting width-three Q/K/V projection history to state slots `[0,T)` and leave
  all other slots unchanged.
- [ ] Ensure the workspace-capacity query covers this exact NVFP4 snapshot profile for every
  requested positive `T` interval. The capacity contract must remain sufficient for the existing
  Q4/Q5 and W8 profiles; no particular workspace layout is prescribed here.
- [ ] Extend the independent full-Op oracle to projection, convolution, SiLU, Z, and every changed
  snapshot word. Verify unchanged state slots exactly.
- [ ] Preserve the existing Q4/Q5 and W8 snapshot forms.

G2 has a stronger distinction from `linear` than G1: it defines convolution, SiLU, four separately
contiguous outputs, and mutation of an explicit snapshot state. A composition that first invokes
public `linear` would also make its BF16 packed projection an observable numerical boundary, while
G2 keeps projected Q/K/V private inside the complete formula.

## 6. Text MLP gate/up

### M1 — NVFP4 `linear_swiglu`

- [ ] Admit the exact NVFP4 `gate_up` parent `[34816,5120]` with row order
  `[gate 17408, up 17408]`.
- [ ] Accept `x BF16 [5120,T]` and write `out BF16 [17408,T]` for every positive `T`.
- [ ] Implement the complete logical formula:

  ```text
  gate[:,t] = Wgate * x[:,t]
  up[:,t]   = Wup * x[:,t]
  out[:,t]  = SiLU(gate[:,t]) * up[:,t]
  ```

- [ ] Keep projected gate/up private: a BF16 `[34816,T]` materialization is not an observable
  semantic boundary.
- [ ] Extend `linear_swiglu_workspace_capacity_bytes` to admit
  `(QType::NVFP4, gate_up_rows=34816, input_rows=5120)` for every valid positive `T` interval.
- [ ] Add an independent complete-formula oracle using exact NVFP4 decode and represented BF16
  input values.
- [ ] Preserve the existing 27B Q4 and 35B W8 registrations.

The two weight halves still represent ordinary linear maps, but their results are not outputs of
M1. M1 exposes only `SiLU(Wgate*x) * (Wup*x)`. Even though the existing `silu_mul` Op can read
strided gate/up views, composing public `linear` and `silu_mul` would first store a public BF16
`[34816,T]` projection and make that rounding seam part of the composed semantics. M1 deliberately
has no such observable intermediate.

## 7. Residual projections

All three registrations retain the existing `linear_add` semantic formula:

```text
residual[:,t] = residual[:,t] + W * x[:,t]
```

The input and in-place residual are contiguous BF16, distinct, and non-overlapping with the weight.
The oracle starts from the represented original BF16 residual and evaluates the projection and add
as one logical formula; a projected BF16 temporary is not an observable requirement.

Consequently, these registrations are not interchangeable with public `linear` followed by
`residual_add`. That composition exposes and rounds a BF16 projection before the add. `linear_add`
instead owns one in-place logical update and compares only the final BF16 residual with the
complete-formula oracle.

### R1 — NVFP4 `linear_add [5120,6144]`

- [ ] Admit an NVFP4 weight `[5120,6144]`.
- [ ] Accept `x BF16 [6144,T]` and update `residual BF16 [5120,T]` for every positive `T`.
- [ ] Use this one registration for both full-attention output (14 sites) and GDN output (47 sites);
  the Op has no model-role discriminator.
- [ ] Extend `linear_add_workspace_capacity_bytes` for
  `(QType::NVFP4, output_rows=5120, input_rows=6144)`.
- [ ] Add independent-oracle coverage of the complete in-place result.

### R2 — NVFP4 `linear_add [5120,17408]`

- [ ] Admit an NVFP4 weight `[5120,17408]`.
- [ ] Accept `x BF16 [17408,T]` and update `residual BF16 [5120,T]` for every positive `T`.
- [ ] Extend `linear_add_workspace_capacity_bytes` for
  `(QType::NVFP4, output_rows=5120, input_rows=17408)`.
- [ ] Add independent-oracle coverage of the complete in-place result.

### R3 — BF16 `linear_add [5120,6144]`

- [ ] Admit one contiguous `BF16_CTRL` weight `[5120,6144]`.
- [ ] Use the same `x [6144,T]`, residual `[5120,T]`, `T > 0`, aliasing, and complete-formula
  contract as R1.
- [ ] Extend `linear_add_workspace_capacity_bytes` for
  `(QType::BF16_CTRL, output_rows=5120, input_rows=6144)`.
- [ ] Add an independent BF16-weight oracle case.
- [ ] Preserve the existing Q5 and W8 registrations for both supported shapes.

R3 covers the two BF16 attention-output parents and the one BF16 GDN-output parent. There is no
BF16 MLP gate/up or down registration in this artifact.

## 8. Common completion criteria

The eight registrations remain incomplete until all applicable checks below pass:

- [ ] Public Op comments describe every newly admitted format, exact shape, row order, output, state
  effect, alias rule, and `T` domain without exposing a kernel schedule as semantics.
- [ ] Op wrappers validate the complete NVFP4 or BF16 `Weight` contract and reject unsupported
  shapes and malformed metadata.
- [ ] Workspace-capacity functions admit the new format/problems and are sufficient over every
  accepted `[min_tokens,max_tokens]` interval.
- [ ] Each production route selected by an admitted format/problem is compared directly with the
  same independent mathematical oracle.
- [ ] NVFP4 criteria account for exact persistent-weight decode and production error from any
  private activation-compute profile without inserting that profile into the oracle.
- [ ] BF16 exception routes are tested as first-class registrations, not treated as an NVFP4
  fallback.
- [ ] Existing 27B Q4/Q5 and peer-target W8 Op tests remain passing.
- [ ] Op tests cover invalid format/shape/divisor metadata, output placement, state effects where
  applicable, and representative real target shapes.

Completion of this checklist establishes Op-level support only. It does not establish that the
NVFP4 artifact can be selected, bound, loaded, or executed through Engine.
