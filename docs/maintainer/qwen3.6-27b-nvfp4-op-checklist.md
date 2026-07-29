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
overloads. A1, A2, G1, G2, R1, R2, and R3 are complete; partial implementation progress for the
remaining registration is recorded only where it changes the next required step.

## 3. Required format/problem registrations

There are five semantic Op surfaces and eight exact format/problem registrations tracked here. The
site counts describe artifact coverage; they are not runtime dispatch inputs.

| ID | Semantic Op | Weight format | Exact parent `[N,K]` | Artifact roles | Status |
|---|---|---|---:|---|---|
| A1 | `attn_input_proj` | `NVFP4` | `[14336,5120]` | attention input, 10 sites | [x] |
| A2 | `attn_input_proj` | `BF16` → `BF16_CTRL` | `[14336,5120]` | attention input, 6 sites | [x] |
| G1 | `gdn_input_proj` | `NVFP4` | `[16384,5120]` | GDN input, 48 sites | [x] |
| G2 | `gdn_input_proj_conv_snapshot` | `NVFP4` | `[16384,5120]` | GDN verify input, same 48 sites | [x] |
| M1 | `linear_swiglu` | `NVFP4` | `[34816,5120]` | MLP gate/up, 64 sites | [ ] |
| R1 | `linear_add` | `NVFP4` | `[5120,6144]` | attention output and GDN output, 61 sites | [x] |
| R2 | `linear_add` | `NVFP4` | `[5120,17408]` | MLP down, 64 sites | [x] |
| R3 | `linear_add` | `BF16` → `BF16_CTRL` | `[5120,6144]` | attention/GDN output exceptions, 3 sites | [x] |

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

当前实现已经覆盖每个正 `T`。`A16Only` 始终不量化 activation；`AllowA4` 在
`T<=16` 仍走 decode/Small-T A16，在 `T>16` 走具有 caller-owned workspace 的 W4A4
Tensor Core route。`16/17` 是 caller 精度许可的固定边界，不是格式 admission limit；
private tile、TMA 整 tile 条件和 kernel instance 同样不构成 public T 限制。

- [x] Admit one complete NVFP4 `query_key_gate_value` weight `[14336,5120]`; do not expose four
  weight row views or multiple weight arguments.
- [x] Accept `x BF16 [5120,T]`, with every positive `T`.
- [x] Write four distinct contiguous BF16 outputs:

  ```text
  q    [6144,T]
  gate [6144,T]
  k    [1024,T]
  v    [1024,T]
  ```

- [x] Interpret physical parent rows exactly as:

  ```text
  query       [0,6144)
  key         [6144,7168)
  output_gate [7168,13312)
  value       [13312,14336)
  ```

  The public argument order `q, gate, k, v` is deliberately different from the physical row order.
- [x] Implement the four independent logical projections `q=Wq*x`, `k=Wk*x`,
  `gate=Wgate*x`, and `v=Wv*x`. There is no observable packed `[14336,T]` output.
- [x] Reject an invalid NVFP4 layout, shape, plane geometry, or non-positive/non-finite divisor.
- [x] Preserve the existing two-parent 27B Q4/Q5 and single-parent 35B W8 admissions and outputs.
- [x] Add independent-oracle coverage for all four row ranges and verify that output allocations
  receive the intended semantic ranges.

The existing single-parent C++ signature already expresses this semantic form; A1 extends its exact
format/geometry admission rather than creating a second NVFP4-specific public name.

The permanent pure Linear surface and `attn_input_proj` use the same quantization and MMA compute
body, with compile-time output policies for contiguous Linear or direct Q/K/gate/V stores. The A4
route is deliberately two-stage: one kernel quantizes represented BF16 activation by token and K16
group to compact E2M1 codes plus E4M3 scales, then the GEMM kernel consumes native SM120 NVFP4
Tensor Cores. Fused repeated quantization was measured and removed.

For exact T, caller-owned workspace contains `2560*T` code bytes and `320*T` scale bytes, allocated
from the Op arena at 256-byte alignment. The capacity query and execution use the same allocation
recipe; allocation occurs before benchmark timing, while quantization, workspace traffic and GEMM
are all inside the timed public call. Neither Op allocates device memory or materializes an
observable packed parent output.

The primary `T=1024` route uses a non-RDC warp-specialized TMA `M256xN128xK128`, three-stage kernel
so its register reallocation remains effective in the production binary. Other qualified extents
use a small set of cp.async schedules; non-integral TMA tiles fall back instead of becoming invalid.
These are current RTX 5090 implementation facts, not mathematical or API requirements. Larger T
is secondary to the common `T=1024` workload and must not displace a faster `T=1024` schedule.

Linear A4 is checked directly against the exact-decoded-weight/naive-FP64 oracle at the `T=17`
precision boundary, a representative cp.async point, and the primary `T=1024` TMA point.
Attention checks all four final allocations against the same independent mathematical definition
at `T=17` and `T=1024`. Existing A16, Q4/Q5, W8 and BF16 suites remain qualified.

RTX 5090, CUDA 13.1, cold-cache measurements with 5 warmups and 30 samples produced:

| Profile | `T` | Median | Useful throughput | Dense FP4 peak |
|---|---:|---:|---:|---:|
| Linear, quantization + GEMM | 1024 | `152.576 us` | `985.24 TFLOP/s` | `58.79%` |
| Attention, quantization + direct four-output GEMM | 1024 | `152.544 us` | `985.45 TFLOP/s` | `58.80%` |

Quantization accounts for about `11.52 us`; the effective GEMM portion is therefore about
`141 us`, `1066 TFLOP/s`, or `63.6%` of dense FP4 peak. NCU on the TMA kernel reports no local
memory or stack spill, `58.89%` tensor-pipe utilization, `79.89%` L2-tag throughput and
`2.64` waves/SM. This evidence makes `T=1024` the production tuning anchor; larger-T wins alone do
not justify replacing its schedule.

The fixed `T<=16` A16 boundary is also performance-consistent: the measured full-call transition
from A16 at `T=16` to A4 at `T=17` reduces latency rather than introducing an upward step. It is
still a caller-approved precision boundary, not a benchmark-derived semantic choice.

The matrix arithmetic is exactly equivalent to forming `Y = W*x` with
`Y BF16 [14336,T]` and taking the four physical row ranges above. That equivalence does not make
generic `linear` the same Op: for `T > 1`, those ranges are strided views of the packed `Y`, whereas
A1 promises four separately contiguous tensors. Weight row order determines the logical mapping;
output allocation order does not.

### A2 — single-parent BF16 `attn_input_proj`

- [x] Admit one complete contiguous `BF16_CTRL` `query_key_gate_value` weight
  `[14336,5120]`.
- [x] Use the same input, output, row-order, aliasing, and `T > 0` contract as A1.
- [x] Evaluate each projection from the represented BF16 weight and activation values; no
  quantization divisor is accepted or used.
- [x] Cover all four row ranges with the independent oracle and retain the existing Q4/Q5 and W8
  regressions.

A2 is required even though it contains no NVFP4 weight: without it, the six BF16 attention-input
parents in the NVFP4 artifact do not have an Op consumer. Its distinction from BF16 `linear` is the
same four-contiguous-output contract as A1, not different projection arithmetic.

A2 is complete. Its private dispatch is monotone: decode at `T=1`, the qualified small-T family at
`T=2..22`, then one shared Linear-owned BF16 MMA body from `T=23` with an Attention-owned
four-output epilogue.
Full and predicated MMA tiles, the route boundary, and `T=1024` pass the independent
represented-BF16/FP64 oracle while the existing Q4/Q5 and W8 cases remain qualified.

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

G1 accepts every positive `T`. `A16Only` never quantizes activation; `AllowA4` resolves to A16 for
`T<=16` and to W4A4 for `T>16`. The policy-bearing entry uses caller-owned workspace sized by the
public capacity query, while the existing no-workspace overload remains the A16-only form for W8
and NVFP4.

- [x] Admit one complete NVFP4 `query_key_value_z` weight `[16384,5120]`.
- [x] Accept `x BF16 [5120,T]`, with every positive `T`.
- [x] Write distinct contiguous outputs:

  ```text
  qkv BF16 [10240,T]  # row order [query 2048, key 2048, value 6144]
  z   BF16 [6144,T]
  ```

- [x] Evaluate all four projections from the same public `x`; Z is not a later generic Linear Op.
- [x] Reject row-view substitutes, invalid NVFP4 metadata, and output aliasing.
- [x] Preserve the current 27B two-parent Q4/Q5 form and the 35B single-parent W8 form.
- [x] Add independent-oracle coverage that checks Q, K, V, and Z separately.

The existing single-parent overload already has the correct semantic argument list. G1 extends its
admitted weight format and the exact 27B geometry.

As with Attention, the matrix arithmetic can be written as `Y = W*x`, followed logically by
`qkv = Y[0:10240,:]` and `z = Y[10240:16384,:]`. For `T > 1`, both are strided row views of one
contiguous `Y [16384,T]`; G1 instead promises two independently contiguous outputs. This layout
effect, rather than a different dot-product formula, distinguishes G1 from generic `linear`.

The production path instantiates the same Linear-owned NVFP4 mainloops with a compile-time QKV/Z
output mapping; it neither calls public `linear` nor materializes/copies a packed parent output.
Pure Linear and G1 were qualified directly against the exact-decoded-weight/naive-FP64 oracle in
A16 decode/Small-T and W4A4 MMA/TMA regions. The A16 `T=1..16` public sweep has no unexplained
route step; once W4A4 MMA begins, tile- and instance-dependent steps are normal implementation
behavior rather than a semantic smoothness requirement.

### G2 — single-parent NVFP4 `gdn_input_proj_conv_snapshot`

- [x] Admit the same complete NVFP4 parent `[16384,5120]` and `x BF16 [5120,T]`.
- [x] Accept the existing 27B snapshot operands:

  ```text
  conv_weight BF16 [10240,4]
  conv_states BF16 [10240,3,Slots]
  initial_slot device I32 scalar
  T > 0
  Slots >= T
  initial_slot in [0,Slots)
  ```

- [x] Write:

  ```text
  query BF16 [2048,T]
  key   BF16 [2048,T]
  value BF16 [6144,T]
  z     BF16 [6144,T]
  ```

- [x] Apply the width-four causal convolution and SiLU only to the projected Q/K/V channels.
  Projected Z bypasses convolution and snapshot state.
- [x] Publish the resulting width-three Q/K/V projection history to state slots `[0,T)` and leave
  all other slots unchanged.
- [x] Ensure the workspace-capacity query covers this exact NVFP4 snapshot profile for every
  requested positive `T` interval. The capacity contract must remain sufficient for the existing
  Q4/Q5 and W8 profiles; no particular workspace layout is prescribed here.
- [x] Extend the independent full-Op oracle to projection, convolution, SiLU, Z, and representative
  changed snapshot words. Verify full-slot writes and unchanged state slots exactly.
- [x] Preserve the existing Q4/Q5 and W8 snapshot forms.

G2 has a stronger distinction from `linear` than G1: it defines convolution, SiLU, four separately
contiguous outputs, and mutation of an explicit snapshot state. Its packed projection remains a
private implementation value even when a production route composes the public Linear
implementation internally; the private BF16 rounding seam is qualified as part of the complete G2
profile and is not inserted into the mathematical oracle.

The policy-bearing single-parent form admits `A16Only` through `T=16` and `AllowA4` for every
positive `T`. `AllowA4` resolves to the same fused A16 kernel for `T<=16`. At `T>16`, G2 calls the
existing pure Linear W4A4 route into one private BF16 `[16384,T]` projection and follows it with one
post kernel; there is no NVFP4 A16 GEMM or repeated Small-T fallback for this Op.

Decode and Small-T instantiate the Linear-owned NVFP4 mainloops with a compile-time row-vector
finalizer. The first 10240 parent rows perform convolution, SiLU, direct Q/K/V placement and state
publication from the FP32 projection accumulator; the final 6144 rows write Z directly. These
routes require one launch and zero workspace.

For W4A4, capacity is the private projection plus the exact nested Linear activation workspace.
The post kernel assigns each 32-channel tile to one CTA. Warp 0 reads the selected initial history
before any state write, a block synchronization publishes it, and the CTA's warps then process
independent contiguous token tiles. This ownership is required when `initial_slot` aliases a slot
in `[0,T)`: a grid with separate token-tile CTAs would otherwise permit another CTA to overwrite
the initial slot before it was read.

The independent complete-formula cases cover fused `T=1`, the A16 boundary at `T=16`, the first
W4A4 point `T=17` with `initial_slot=0`, and the primary TMA workload `T=1024`. The same test retains
the existing Q4/Q5 and W8 routes, checks every output allocation and state effect, and matches
workspace execution high-water to the public capacity query.

RTX 5090, CUDA 13.1, cold-cache public-Op measurements with 10 warmups and 100 samples produced:

| `T` | Activation compute | Median |
|---:|---|---:|
| 1 | A16 fused | `38.144 us` |
| 4 | A16 fused | `44.224 us` |
| 8 | A16 fused | `58.016 us` |
| 16 | A16 fused | `93.472 us` |
| 17 | W4A4 Linear + post | `45.984 us` |
| 1024 | W4A4 Linear + post | `263.104 us` |

The `16/17` route change is an approved precision boundary and its latency step is normal kernel
behavior. Post tuning was anchored on `T=1024`; larger extents and individual boundary values do
not override that workload.

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

- [x] Admit an NVFP4 weight `[5120,6144]`.
- [x] Accept `x BF16 [6144,T]` and update `residual BF16 [5120,T]` for every positive `T`.
- [x] Use this one registration for both full-attention output (14 sites) and GDN output (47 sites);
  the Op has no model-role discriminator.
- [x] Extend `linear_add_workspace_capacity_bytes` for
  `(QType::NVFP4, output_rows=5120, input_rows=6144)`.
- [x] Add independent-oracle coverage of the complete in-place result.

### R2 — NVFP4 `linear_add [5120,17408]`

- [x] Admit an NVFP4 weight `[5120,17408]`.
- [x] Accept `x BF16 [17408,T]` and update `residual BF16 [5120,T]` for every positive `T`.
- [x] Extend `linear_add_workspace_capacity_bytes` for
  `(QType::NVFP4, output_rows=5120, input_rows=17408)`.
- [x] Add independent-oracle coverage of the complete in-place result.

R1 and R2 use the Linear-owned mainloops with a compile-time epilogue that adds the represented
input residual to the FP32 accumulator before the one final BF16 conversion. They do not expose a
rounded projection temporary. Both exact geometries pass the same complete-formula oracle in A16
decode/Small-T and W4A4 MMA/TMA regions; the existing Q5 and W8 registrations remain A16-only and
unchanged.

RTX 5090, CUDA 13.1, cold-cache public-Op measurements with 5 warmups and 30 samples produced:

| Op | `T=1` | `T=4` | `T=8` | `T=16` | `T=17` | `T=128` | `T=512` | `T=1024` |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| G1 | `37.888 us` | `42.080 us` | `56.320 us` | `93.536 us` | `37.824 us` | `40.032 us` | `107.552 us` | `177.152 us` |
| R1 | `17.664 us` | `21.760 us` | `29.984 us` | `42.272 us` | `25.888 us` | `31.872 us` | `47.648 us` | `83.168 us` |
| R2 | `38.144 us` | `44.288 us` | `60.576 us` | `89.184 us` | `54.368 us` | `58.336 us` | `109.568 us` | `202.944 us` |

At the primary `T=1024` point, G1, R1, and R2 respectively reach `969.78`, `774.63`, and
`899.44 TFLOP/s`, or `57.86%`, `46.22%`, and `53.67%` of the RTX 5090 dense FP4 reference.
Quantization and residual reads are included. Candidate scanning found no stable semantic-Op
schedule crossover relative to each geometry's tuned pure Linear winner, so production retains
the shared schedules rather than adding Op-local parameter tables.

### R3 — BF16 `linear_add [5120,6144]`

- [x] Admit one contiguous `BF16_CTRL` weight `[5120,6144]`.
- [x] Use the same `x [6144,T]`, residual `[5120,T]`, `T > 0`, aliasing, and complete-formula
  contract as R1.
- [x] Extend `linear_add_workspace_capacity_bytes` for
  `(QType::BF16_CTRL, output_rows=5120, input_rows=6144)`.
- [x] Add an independent BF16-weight oracle case.
- [x] Preserve the existing Q5 and W8 registrations for both supported shapes.

R3 covers the two BF16 attention-output parents and the one BF16 GDN-output parent. There is no
BF16 MLP gate/up or down registration in this artifact.

R3 reuses the BF16 Linear decode, exact-small-T, and MMA computation bodies and supplies the
in-place residual update as the Op-owned epilogue, with one final BF16 store and no workspace. The
measured production dispatch is decode at `T=1`, small-T for `2 <= T <= 26`, and MMA for `T >= 27`;
candidate overlap remains available in the Op benchmark so that the boundary is reproducible.

## 8. Common completion criteria

Each registration remains incomplete until all applicable checks below pass:

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
