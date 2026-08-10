# Aggregate-column Op transaction checklist

This is the task-local checklist for qualifying the existing column-independent public Ops that
future concurrent Decode will call with aggregate columns. It records execution status and
evidence; it does not redefine the architecture or any public Op contract. The governing
requirements remain [Concurrent decode operator requirements](concurrent-decode-operators.md),
[Op development](op-development.md), and the headers under [`include/ninfer/ops/`](../../include/ninfer/ops/).

Remove this checklist and its documentation-index entry after the qualification and
optimization work is complete. Integrate any durable contract change into the existing active
authority instead of retaining this file as a historical plan.

## Status and execution gate

- [x] Worktree is `/path/to/home/ninfer-aggregate-ops` on `perf/aggregate-column-ops`.
- [x] The inventory below is frozen from the completed caller/leaf/contract/benchmark review;
      active transactions consume only its static Op profiles.
- [x] Execution is organized as serial Op/profile transactions rather than one global test phase
      followed by one global optimization phase.
- [x] Every required transaction is closed or explicitly deferred by the user.

Only one transaction may be active. For ordinary closed Ops, one transaction covers that public Op
and all of its listed production profiles. `linear` is intentionally finer grained: each exact
format and `[N,K]` row below is a separate transaction. A transaction includes public-route
qualification, any implementation work required by the result, final verification, and cleanup.
After finding a route problem, continue in the same transaction through optimization and final
verification. When the user has authorized an active queue, commit each finished transaction
independently before starting the next; otherwise stop for review after the current transaction.

## Fixed scope

- Column-independent Ops retain their existing column extent `T`; concurrent execution presents
  only `T_total = B * W`.
- Do not add a request-batch dimension, batch descriptor, concurrent-only overload, or target key
  to any Op in this checklist.
- Every weight-bearing profile must consume all aggregate columns in one public Op call. A loop of
  request-local calls is not an aggregate route.
- Preserve every current `B=1` production case.
- Performance evidence must invoke the public `ninfer::ops` entry and let production dispatch
  select its implementation. Private launchers, forced candidates, and copied kernels are not
  evidence for this task.
- If an existing benchmark exposes route/candidate switches, only its automatic production mode
  can supply the final result; forced modes are development comparisons only.
- Do not load a model artifact or invoke target runtime, Engine, or DecodeRound benchmarks. Exact
  formats and shapes in this checklist are static Op profiles, not permission to cross the Op
  boundary.
- Attention, Linear Attention state transitions, sampling, speculative transactions, state
  selectors, scheduling, CUDA Graphs, and concurrent Engine work are outside this checklist.

### Aggregate domains

`B` is in `[1,8]`. `K` is the configured proposal length.

| Symbol | Production segment | Reachable aggregate extent |
|---|---|---|
| `O` | ordinary Decode or one MTP autoregressive proposal step | `T_total=B`, hence `1..8` |
| `M` | MTP target verify or alignment forward | `T_total=B*(K+1)`, `K=1..5`, maximum 48 |
| `D` | DFlash proposal block, target verify, or context append | `T_total=B*(K+1)`, `K=1..15`, maximum 128 |
| `P` | DFlash final proposal-head output | `T_total=B*K`, `K=1..15`, maximum 120 |

For legality, enumerate every distinct value produced by the applicable formula. For a public
workspace-capacity query, cover the inclusive closure from 1 through the profile's maximum. Timed
points need not repeat every equivalent factorization, but must include all `B=1` cases, the
maximum, and every observed production dispatch boundary or cliff.

## Per-transaction workflow

The leading checkbox on each Op section, or each `linear` table row, means the complete transaction
is closed. It must not be checked merely because the initial benchmark ran.

For the one active work unit:

1. Confirm its exact public entry, format, shape, policy, aggregate domain, and `B=1` cases.
2. Reuse the public benchmark when sufficient. Add only the minimum temporary public-Op capability
   needed for missing shape/T coverage and the relevant route-neutral useful-work metrics.
3. If the public result exposes a route problem, continue immediately with a task-local private
   launcher sweep, change production dispatch, and verify correctness and final public performance.
4. Keep the selected implementation behind production dispatch; remove losing candidates,
   comparison-only controls, and temporary paths. Only after the transaction is complete, add one
   concise result here and stop before the next work unit.

### Performance evidence

Use effective bandwidth/`READ_%` for memory-bound work and useful TFLOP/s for compute-intensive
work. Use `TC_%` only when route-specific evidence establishes the actual activation-compute
profile; `AllowA4` is permission and must not be treated as proof of A4 execution. Linear counts
`weight_bytes + 2*K*T + 2*N*T` bytes and `2*N*K*T` FLOPs. Measure one `T=1` benchmark point with
the same normal warmup/repetitions as other points, then calculate
`T * median(T=1) / median(T)`; never issue `T` sequential `T=1` launches as the comparison.
Use the Release `sm_120a` build on the RTX 5090, with enough warmup/repetitions to remove obvious
noise; discard runs taken under significant competing GPU load.

Current transaction: none. C10 is closed.

Current transaction status: `complete`.

## Transaction queue: weight-bearing and fused public Ops

Weight shapes below use `[N,K]`. `Q4`, `Q5`, `Q6`, `W8`, and `N4` abbreviate
`Q4G64_F16S`, `Q5G64_F16S`, `Q6G64_F16S`, `W8G32_F16S`, and NVFP4 respectively.

### W01 — `embedding`

- [x] Transaction closed for public [`embedding`](../../include/ninfer/ops/embedding.h).
- Production profiles:
  - 27B Q6 storage profile: Q6 `[248320,5120]`.
  - 27B W8 endpoint profile: W8 `[248320,5120]`.
  - 35B-A3B: W8 `[248320,2048]`.
- Production paths and domains:
  - 27B target and MTP: `O ∪ M`.
  - 35B-A3B target, MTP, and DFlash proposal: `O ∪ M ∪ D`.
- Qualification domains: all 27 distinct `O ∪ M` values through 48 for both D=5120 profiles;
  all 64 distinct `O ∪ M ∪ D` values through 128 for W8/D=2048, including every `B=1` case.
- Existing public benchmark: `ninfer_embedding_bench`.
- [x] Extend the Q6 public path from fixed `T=1/64` to an explicit aggregate token list.
- [x] Add the registered W8/D=5120 geometry; retain the existing W8/D=2048 path.
- [x] Make the retained benchmark public-only and report effective bandwidth against the sustained
      read reference.
- Workspace query: none.
- Ownership: in scope.

#### W01 qualification result — 2026-08-09

Release `sm_120a` public-Op correctness and full-domain sweeps passed on RTX 5090. There is no
workspace; source inspection confirms one aggregate kernel launch and no per-column host loop.

| Profile | Aggregate median over required domain | Maximum-T effective bandwidth / read utilization | Judgment |
|---|---:|---:|---|
| Q6 `[248320,5120]`, `T≤48` | `4.736..5.728 µs` | `119.4 GB/s / 7.13%` | 无需优化 |
| W8 `[248320,5120]`, `T≤48` | `5.568..5.728 µs` | `131.4 GB/s / 7.85%` | 无需优化 |
| W8 `[248320,2048]`, `T≤128` | `3.680..5.728 µs` | `140.2 GB/s / 8.38%` | 无需优化 |

The required extents are launch-limited sub-megabyte gathers, aggregate latency stays below 5.8 µs,
and the W8/D=2048 `T=6→7` dispatch boundary has no cliff. No production change is justified.

### W02 — `linear`

`linear` is not one transaction. Complete these rows serially; do not qualify or optimize several
formats/shapes as one batch.

| ID | Closed | Exact public profile | Production use | Domain / capacity interval |
|---|---|---|---|---|
| L01 | [x] | Q6 `[248320,5120]` | 27B groupwise full endpoint | `O ∪ M` / `[1,48]` |
| L02 | [x] | W8 `[248320,5120]` | 27B NVFP4-package full endpoint | `O ∪ M` / `[1,48]` |
| L03 | [x] | Q4 `[131072,5120]` | 27B optimized draft endpoint | `O` / `[1,8]` |
| L04 | [x] | W8 `[5120,10240]` | 27B MTP input projection | `O ∪ M` / `[1,48]` |
| L05 | [x] | W8 `[14336,5120]` | 27B MTP packed attention projection | `O ∪ M` / `[1,48]` |
| L06 | [x] | W8 `[5120,6144]` | 27B MTP attention output | `O ∪ M` / `[1,48]` |
| L07 | [x] | W8 `[34816,5120]` | 27B MTP gate/up projection | `O ∪ M` / `[1,48]` |
| L08 | [x] | W8 `[5120,17408]` | 27B MTP down projection | `O ∪ M` / `[1,48]` |
| L09 | [x] | Q6 `[248320,2048]` | 35B-A3B full endpoint, including DFlash | `O ∪ M ∪ D` / `[1,128]` |
| L10 | [x] | Q4 `[131072,2048]` | 35B-A3B optimized MTP/DFlash draft endpoint | `O ∪ P` / `[1,120]` |
| L11 | [x] | W8 `[2048,4096]` | 35B-A3B MTP input and attention output | `O ∪ M` / `[1,48]` |
| L12 | [x] | W8 `[2048,16384]` | 35B-A3B DFlash feature projection | `O ∪ D` / `[1,128]` |

L01 evidence: public correctness and `[1,48]` zero-workspace execution pass. `T=1` remains
`596.9 µs / READ 99.42%`; `T=48` improves from `1689.6` to `1044.6 µs`, with `TC 55.77%` and
calculated T=1 extrapolation `27.43x`; the former large route cliffs are removed.

L02 evidence: W8 `[248320,5120]` public correctness and `[1,48]` zero-workspace execution pass;
`T=1` remains `780.3 µs / READ 103.42%`, while `T=48` improves from `1341.1` to `1014.5 µs`
(`READ 80.95%`, `TC 57.43%`, calculated T=1 extrapolation `36.92x`) with no material route cliff.

L03 evidence: Q4 `[131072,5120]` public oracle and `[1,8]` zero-workspace execution pass; `T=1`
remains `220.8 µs / READ 96.50%`, while `T=8` improves from `869.9` to `259.7 µs`
(`READ 82.49%`, `TC 19.74%`, calculated T=1 extrapolation `6.80x`) with no remaining route cliff.

L04 evidence: W8 `[5120,10240]` public oracle and `[1,48]` zero-workspace execution pass; `T=1`
improves from `49.15` to `40.96 µs`, while `T=48` improves from `200.32` to `71.71 µs`
(`READ 47.62%`, `70.19 TFLOP/s / TC 33.50%`, calculated T=1 extrapolation `27.42x`); the former
route cliffs are removed.

L05 evidence: W8 `[14336,5120]` public oracle and `[1,48]` zero-workspace execution pass; `T=1`
improves from `60.80` to `52.83 µs`, while `T=48` improves from `163.42` to `102.02 µs`
(`READ 46.75%`, `69.07 TFLOP/s / TC 32.97%`, calculated T=1 extrapolation `24.86x`); the erroneous
T=5 and T=9 route cliffs are removed.

L06 evidence: W8 `[5120,6144]` public oracle and `[1,48]` zero-workspace execution pass; `T=1`
improves from `32.35` to `25.92 µs`, while `T=48` improves from `122.46` to `46.69 µs`
(`READ 44.14%`, `64.68 TFLOP/s / TC 30.87%`, calculated T=1 extrapolation `26.65x`); the former
route cliffs are removed.

L07 evidence: W8 `[34816,5120]` public oracle and `[1,48]` zero-workspace execution pass; `T=1`
improves from `126.56` to `118.40 µs`, while `T=48` improves from `323.07` to `190.08 µs`
(`READ 60.71%`, `90.03 TFLOP/s / TC 42.97%`, calculated T=1 extrapolation `29.90x`); the erroneous
T=5/T=9 cliffs and high-T latency plateau are removed.

L08 evidence: W8 `[5120,17408]` public oracle and `[1,48]` zero-workspace execution pass; `T=1`
improves from `79.46` to `61.47 µs`, while `T=48` improves from `335.49` to `118.37 µs`
(`READ 48.87%`, `72.29 TFLOP/s / TC 34.50%`, calculated T=1 extrapolation `24.93x`); the former
route cliffs and high-T latency plateau are removed.

L09 evidence: Q6 `[248320,2048]` public oracle and `[1,128]` zero-workspace execution pass; `T=1`
holds at `258.05→259.71 µs`, while `T=48/65/96` improve from `552.10/694.27/710.62` to
`398.27/514.08/610.34 µs`; `T=128` is `693.66 µs` (`187.69 TFLOP/s / TC 89.59%`, calculated T=1
extrapolation `47.92x`).

L10 evidence: Q4 `[131072,2048]` public oracle and `[1,120]` zero-workspace execution pass; `T=1`
holds at `99.94→99.68 µs`, while `T=2/48/80` improve from `355.97/366.34/376.45` to
`101.54/224.10/280.19 µs`; `T=120` improves to `368.19 µs` (`174.98 TFLOP/s / TC 83.52%`,
calculated T=1 extrapolation `32.49x`).

L11 evidence: W8 `[2048,4096]` public oracle and `[1,48]` zero-workspace execution pass; `T=1`
improves from `17.54` to `12.86 µs`, while `T=48` improves from `54.43` to `22.11 µs`
(`36.42 TFLOP/s / TC 17.38%`, calculated T=1 extrapolation `27.92x`).

L12 evidence: W8 `[2048,16384]` public oracle and `[1,128]` zero-workspace execution pass; `T=1`
holds at `30.24→30.30 µs`, while `T=2/33/65/96` improve from
`47.30/73.02/126.11/157.63` to `32.35/56.61/93.50/128.61 µs`; `T=128` is `173.70 µs`
(`49.45 TFLOP/s / TC 23.61%`, calculated T=1 extrapolation `22.33x`).

- Existing public benchmark: `ninfer_linear_bench`; explicit mode already admits exact registered
  format, shape, policy, T sweep, and capacity query. The compact suites are not the coverage
  authority.
- For every row, use the same A16/A4 permission as the production caller and close that row only
  after the full per-transaction workflow.
- Ownership: in scope.

### W03 — `attn_input_proj`

- [x] Transaction closed for every production overload of public
      [`attn_input_proj`](../../include/ninfer/ops/attn_input_proj.h).
- Production profiles:
  - 27B groupwise: Q4 query/key `[7168,5120]` plus Q5 gate/value `[7168,5120]`.
  - 27B N4 package: BF16 or N4 parent `[14336,5120]`; BF16 uses `A16Only`, while N4 uses
    production `AllowA4` and resolves to A16 for `T=1..3` and W4A4 from `T=4`.
  - 35B-A3B target/MTP: W8 parent `[9216,2048]`.
  - 35B-A3B DFlash: W8 query/key/value parent `[6144,2048]`.
- Production domains:
  - 27B target: `O ∪ M`;
  - 35B-A3B target: `O ∪ M ∪ D`;
  - MTP: `O ∪ M`;
  - DFlash proposal: `D`.
- Existing public benchmark: `ninfer_attn_input_proj_bench`; it covers all four forms and arbitrary
  token lists. Use only automatic production dispatch.
- [x] Query `[1,48]` for the 27B single-parent profile and `[1,128]` for the 35B-A3B target profile.
  The split groupwise and DFlash forms have no workspace.
- Ownership: projection is in scope; GQA is not.

W03 evidence: public correctness, required aggregate domains, and capacity intervals pass. Q4/Q5
`T=48` improves `253.5→124.3 µs`; W8 target removes the `T=17` cliff and reaches
`20.1/38.5/44.7 µs` at `T=18/64/128`. At maximum T, W8 companion is `40.5 µs`, BF16 is
`145.0 µs / READ 57.2%`, and N4 `AllowA4` is `34.2 µs / 206.2 TFLOP/s`; no further issue remains.

### W04 — `gdn_norm_gating_proj`

- [x] Transaction closed for public
      [`gdn_norm_gating_proj`](../../include/ninfer/ops/gdn_gating_proj.h), not the pre-normalized
      `gdn_gating_proj` substitute.
- Production profiles:
  - 27B two-weight BF16 control: two `[48,5120]` projections, `h[5120,T]`, and
    `g/beta[48,T]`.
  - 35B-A3B contiguous BF16 parent `[64,2048]`, `h[2048,T]`, and `g/beta[32,T]`.
- Production domains: 27B `O ∪ M`; 35B-A3B `O ∪ M ∪ D`.
- Existing public benchmark: `ninfer_gdn_gating_proj_bench` directly covers only the 35B-A3B
  `--norm-control --candidate auto` form.
- [x] Minimally permit the 27B two-weight public norm-control form and its aggregate token list.
- [x] Query `[1,48]` for 27B and `[1,128]` for 35B-A3B.
- Ownership: in scope.

W04 evidence: public correctness and both full aggregate domains/capacity intervals pass. The 27B
two-weight form is `11.9→17.7 µs` from `T=1→48`; the 35B form is `7.8 µs` at `T=1` and
`13.6 µs` at `T=128`. Its fused `T≤16` route remains faster than the composed comparison; no
production change is justified.

### W05 — `linear_add`

- [x] Transaction closed for public [`linear_add`](../../include/ninfer/ops/linear_add.h) and its
      capacity query.
- Production profiles:
  - 27B attention/GDN output: Q5, BF16, or N4 `[5120,6144]`.
  - 27B post-mixer down: Q5 or N4 `[5120,17408]`.
  - 35B-A3B attention/GDN/DFlash attention output: W8 `[2048,4096]`.
  - DFlash MLP down: W8 `[2048,6144]`.
- Production domains: 27B `O ∪ M`; 35B-A3B target `O ∪ M ∪ D`; DFlash proposal `D`.
- Existing public benchmarks:
  - `ninfer_bf16_linear_add_bench` for BF16 `[5120,6144]`;
  - `ninfer_nvfp4_linear_add_bench` for both N4 profiles;
  - `ninfer_w8_linear_add_bench` for both W8 profiles;
  - `ninfer_q5_linear_add_bench` for both Q5 profiles.
- [x] Add W8 `[2048,4096]` to a public production-dispatch harness.
- [x] Add Q5 `[5120,6144]` and `[5120,17408]`; a pure `linear` result is not evidence for the
      fused residual Op.
- [x] Query through 48 for 27B and through 128 for 35B-A3B/DFlash profiles.
- The two N4 profiles use production `AllowA4`; their W4A4 crossovers are respectively `T=7` and
  `T=8`, and the public capacity query must cover those selected routes.
- Ownership: in scope.

W05 evidence: all four public oracle tests and required zero-workspace capacity/domain sweeps pass.
At maximum aggregate T, Q5 `[5120,6144/17408]` is `67.6/179.8 µs`, W8
`[2048,4096/6144]` is `49.0/73.3 µs`, BF16 `[5120,6144]` is `48.7 µs / 1.32 TB/s`, and N4
`[5120,6144/17408]` is `26.2/52.8 µs`; the former materialization and route cliffs are removed.

### W06 — `linear_swiglu`

- [x] Transaction closed for public [`linear_swiglu`](../../include/ninfer/ops/linear_swiglu.h) and
      its capacity query.
- Production profiles:
  - 27B dense post-mixer Q4 or N4 `[34816,5120] -> [17408,T]`.
  - DFlash W8 `[12288,2048] -> [6144,T]`.
- Production domains: 27B `O ∪ M`; DFlash proposal `D`.
- Public benchmarks: `ninfer_q4_linear_swiglu_bench`, `ninfer_w8_linear_swiglu_bench`, and
  `ninfer_nvfp4_linear_swiglu_bench`.
- [x] Add the Q4 production profile to a public harness.
- [x] Preserve all 27B `B=1` cases, including MTP `T=2..6`.
- [x] Query and invoke N4 with production `AllowA4` over `[1,48]`; the resolver selects A16 for
      `T=1..4` and W4A4 from `T=5`.
- Ownership: in scope.

W06 evidence: all three public oracle tests and complete required capacity/domain sweeps pass. Q4
T24 falls from `265.2` to `151.1 µs` and reaches `65.4 TFLOP/s` at T48; W8 T33 falls from `48.8`
to `32.5 µs` and reaches `102.3 TFLOP/s` at T128; N4 T33/T48 fall from `140.6/142.7` to
`71.2/73.1 µs`, with `1.40 TB/s / 234.2 TFLOP/s` at T48.

### W07 — `sparse_moe`

- [x] Transaction closed for public [`sparse_moe`](../../include/ninfer/ops/sparse_moe.h) and its
      capacity query.
- Shared geometry: BF16 router/shared-gate `[257,2048]`, routed gate/up `[262144,2048]`, routed
  down `[524288,512]`, shared W8 gate/up `[1024,2048]`, and shared W8 down `[2048,512]`.
- Production codecs:
  - main 35B-A3B: routed Q4+Q5, with Q4+Q6 in layers 34, 38, and 39;
  - MTP: routed W8+W8.
- Production domains: main `O ∪ M ∪ D`; MTP `O ∪ M`.
- Existing public benchmark: `ninfer_sparse_moe_bench`, including all codecs, arbitrary T,
  capacity queries, and `trace-like`, `independent`, and `same` expert distributions.
- [x] Measure at least trace-like and independent aggregate routing; do not infer the concurrent
      result from a same-expert fixture alone.
- [x] Query `[1,128]` for main profiles and `[1,48]` for MTP.
- Ownership: in scope.

W07 evidence: public FP64-oracle tests, grouped/adaptive scheduling, and every required capacity/T
pass. Independent Q4+Q5 T45/T48 fall from `487.5/495.6` to `401.4/465.3 µs`; W8 T18/T19 fall
from `383.0/389.1` to `327.7/342.0 µs`. At maximum T, Q4+Q5 is `524.3/593.9 µs` and W8 is
`395.3/573.4 µs` for trace-like/independent routing, with calculated T=1 batching gains of
`7.0x/6.2x` and `4.7x/3.1x`, respectively.

### W08 — `linear_pair`

- [x] Transaction closed for public [`linear_pair`](../../include/ninfer/ops/linear_pair.h).
- Production profile in this task: DFlash context K/V, two adjacent W8 `[1024,2048]` row views of
  the same `[6144,2048]` parent.
- Production domain: `O ∪ D`.
- Existing public performance benchmark: none. A public semantic test exists.
- [x] Add a minimal public-only harness with the exact adjacent row-view layout and arbitrary T.
- Workspace query: none.
- Ownership: the DFlash `[1024,2048]` profile is in scope. The `[1024,5120]` profile is used only
  by MTP prefill and is excluded.

W08 evidence: the public FP64 oracle and public `T=1..128` sweep pass with no workspace. T1 remains
`7.78 µs`; T128 is `24.16 µs / 44.44 TFLOP/s / TC 21.2%`, a calculated T1 extrapolation of
`41.2x`. One pair call is already faster than one same-shape public W8 `linear` at T128
(`30.34 µs`), so no production route change is justified.

## Transaction queue: norm, pointwise, layout, and output public Ops

### C01 — `rmsnorm`

- [x] Transaction closed for public [`rmsnorm`](../../include/ninfer/ops/rmsnorm.h).
- Production profiles:
  - 27B unit-offset hidden `[5120,T]`, Q `[256,24,T]`, and K `[256,4,T]`;
  - 35B-A3B unit-offset hidden `[2048,T]`, Q `[256,16,T]`, and K `[256,2,T]`;
  - DFlash non-unit-offset hidden `[2048,T]`, Q `[128,32,T]`, and K `[128,8,T]`.
- Production domains follow the caller: target `O ∪ M` or `O ∪ M ∪ D`, MTP `O ∪ M`, DFlash
  context/proposal/final output `O ∪ D ∪ P`.
- Existing public benchmark: `ninfer_rmsnorm_bench`; it covers hidden27 and all listed 35B-A3B and
  DFlash geometries.
- [x] Add the missing 27B Q and K geometries.
- Workspace query: none. Ownership: in scope.

C01 evidence: the public FP64 oracle covers every listed geometry and all aggregate extents are
admitted without workspace or T-dependent dispatch. The 27B hidden/Q/K profiles remain
launch-limited at `9.05/9.14/9.23 µs` for T48 (`108.6/129.0/21.3 GB/s`); no route cliff or
per-column expansion is present, so no production change is justified.

### C02 — `gated_rmsnorm`

- [x] Transaction closed for public [`gated_rmsnorm`](../../include/ninfer/ops/gated_rmsnorm.h).
- Production profiles: 27B GDN `[128,48,T]`; 35B-A3B GDN `[128,32,T]`.
- Production domains: 27B `O ∪ M`; 35B-A3B `O ∪ M ∪ D`.
- Existing public benchmark: `ninfer_rmsnorm_bench` covers the 35B-A3B form.
- [x] Add the missing 27B gated geometry.
- Workspace query: none. Ownership: in scope.

C02 evidence: public FP64-oracle tests and both aggregate domains pass without workspace. The 27B
`[128,48,T]` profile is `9.26 µs / 191.1 GB/s` at T48 and the 35B `[128,32,T]` profile is
`9.24 µs / 340.3 GB/s` at T128; both use one fixed aggregate launch with no route cliff.

### C03 — `rope`

- [x] Transaction closed for public [`rope`](../../include/ninfer/ops/rope.h) after positions are
      already prepared.
- Production profiles:
  - 27B target/MTP pair: head 256, Q heads 24, K heads 4, rotary 64;
  - 35B-A3B target/MTP pair: head 256, Q heads 16, K heads 2, rotary 64;
  - DFlash proposal pair: head/rotary 128, Q heads 32, K heads 8;
  - DFlash context single-K: `[128,8,T]`.
- Existing public benchmark: `ninfer_rope_bench` covers both target pairs, DFlash pair, and
  arbitrary token lists.
- [x] Add the DFlash single-K form.
- Workspace query: none.
- Ownership: the Rope call is in scope; construction of flat per-request positions is not.

C03 evidence: public FP64-oracle coverage passes for all pair/single production geometries. The
added DFlash single-K `[128,8,T]` form uses one fixed aggregate launch and remains `9.09 µs` and
`57.7 GB/s` at T128, with no workspace, T restriction, or route cliff.

### C04 — `sigmoid_mul`

- [x] Transaction closed for public [`sigmoid_mul`](../../include/ninfer/ops/sigmoid_mul.h).
- Production profiles: 27B `[256,24,T]`; 35B-A3B `[256,16,T]`.
- Production domains: 27B `O ∪ M`; 35B-A3B `O ∪ M ∪ D`.
- Existing public benchmark: `ninfer_sigmoid_mul_bench`, which covers only `[256,16,T]`.
- [x] Add a selectable 24-head production geometry.
- Workspace query: none. Ownership: in scope.

C04 evidence: public FP64-oracle tests and both aggregate domains pass. The fixed single-launch
route reaches `9.00 µs / 196.5 GB/s` for 27B `[256,24,48]` and `9.00 µs / 349.4 GB/s` for 35B
`[256,16,128]`; no route cliff or per-column expansion is present.

### C05 — `silu_mul`

- [x] Transaction closed for public [`silu_mul`](../../include/ninfer/ops/silu_mul.h).
- Production profile: 27B MTP dense post-mixer `[17408,T]`.
- Production domain: `O ∪ M`.
- Existing public benchmark: `ninfer_silu_mul_bench`, with only decode `T=1` and prefill `T=4096`.
- [x] Add the aggregate token list while retaining one public call per point.
- Workspace query: none. Ownership: in scope.

C05 evidence: the public FP64 oracle and aggregate domain pass without workspace. The contiguous
`[17408,T]` production route is one aggregate launch, holds at `8.59 µs / 583.5 GB/s` for T48,
and has no T-dependent dispatch or cliff.

### C06 — `residual_add`

- [x] Transaction closed for public [`residual_add`](../../include/ninfer/ops/residual_add.h).
- Production profiles: MTP hidden D=5120 and D=2048; 27B also uses D=5120 after its MTP down
  projection.
- Production domain: `O ∪ M`.
- Existing public benchmark: `ninfer_residual_add_bench`, which is Vision-only D=1152 and
  restricts its second extent to a multiple of four.
- [x] Add D=2048/5120 with arbitrary aggregate T.
- Workspace query: none. Ownership: in scope.

C06 evidence: public FP64-oracle tests and both `[D,T≤48]` domains pass without workspace; the
old multiple-of-four restriction was benchmark-only. The fixed aggregate launch is `9.06 µs` at
T48 for both D values (`162.7 GB/s` at D5120 and `65.1 GB/s` at D2048), with no route cliff.

### C07 — `mtp_pack_fc_input`

- [x] Transaction closed for public [`mtp_pack_fc_input`](../../include/ninfer/ops/mtp_pack.h).
- Production profiles: 27B two `[5120,T]` inputs to `[10240,T]`; 35B-A3B two `[2048,T]` inputs to
  `[4096,T]`. The operation is an exact BF16 copy/remap.
- Production domain: `O ∪ M`.
- Existing public performance benchmark: none. A semantic public-Op test exists.
- [x] Add a minimal public-only benchmark for both D values and the aggregate token set.
- Workspace query: none. Ownership: in scope.

C07 evidence: bit-exact public tests and both `T≤48` domains pass without workspace. One aggregate
launch reaches `9.19 µs / 214.0 GB/s` for D5120 and `9.27 µs / 84.8 GB/s` for D2048 at T48;
there is no per-column expansion or route cliff.

### C08 — `mtp_split_attn_in`

- [x] Transaction closed for public [`mtp_split_attn_in`](../../include/ninfer/ops/mtp_pack.h).
- Production profile: 27B `[14336,T]` to Q/Gate `[256,24,T]` and K/V `[256,4,T]`, with exact
  BF16 copy/remap semantics.
- Production domain: `O ∪ M`.
- Existing public performance benchmark: none. A semantic public-Op test exists.
- [x] Reuse the minimal MTP transform benchmark without composing or duplicating production code.
- Workspace query: none. Ownership: in scope.

C08 evidence: bit-exact public tests and the complete `T≤48` domain pass without workspace. The
single aggregate remap launch is `9.09 µs / 302.7 GB/s` at T48, with no per-column expansion or
route cliff.

### C09 — `argmax`

- [x] Transaction closed for public [`argmax`](../../include/ninfer/ops/argmax.h).
- Production profiles:
  - full logits `[248320,T]` with 248077 valid rows;
  - optimized draft logits `[131072,T]` with 131072 valid rows.
- Production domains:
  - full: 27B `O ∪ M`; 35B-A3B `O ∪ M ∪ D`;
  - optimized draft: 27B `O`; 35B-A3B `O ∪ P`.
- Existing public benchmark: `ninfer_argmax_bench`, which limits full columns to 16 and shortlist
  columns to one.
- [x] Extend the existing allocations and public calls through 128 for full logits and 120 for
      shortlist logits.
- Workspace query: none. Ownership: in scope.

C09 evidence: exact public-oracle checks pass for both registered vocab profiles through full T128
and shortlist T120. Aggregate-aware block dispatch improves the public full T128 route from
`171.98 to 153.33 us` (`414.2 GB/s`) and shortlist T120 from `76.04 to 71.97 us` (`437.1 GB/s`),
while the T1 and B=1 routes remain normal and no per-column launch expansion occurs.

### C10 — `proposal_remap_token_ids`

- [x] Transaction closed for public
      [`proposal_remap_token_ids`](../../include/ninfer/ops/speculative_round.h).
- Production profile: in-place I32 `[T]` lookup through a distinct device map of count 131072.
- Production domains: MTP `O`; DFlash `P`.
- Existing public performance benchmark: none. A semantic public-Op test exists.
- [x] Add a minimal public-only count=131072 benchmark through T=120.
- Workspace query: none.
- Ownership: this stateless output transform is in scope; the surrounding speculative transaction
  is not.

C10 evidence: exact in-place public-oracle checks pass for the count=131072 map through T120. One
public call launches one aggregate kernel; T1 through T120 stays flat at `9.21-9.46 us`, confirming
a launch-limited transform with no per-token expansion.

## Transaction acceptance and cleanup

Do not update this document while a transaction is running. Benchmark output, candidate tables,
profiler reports, and command transcripts are working data, not checklist content. After the
transaction is finished, retain only one concise evidence statement containing the exact profile
and domain, correctness/capacity outcome, final public metric, material before/after result when
code changed, and judgment.

A transaction passes when the required extents and workspace contract are legal, the public Op
matches its independent oracle, one public call consumes the complete aggregate columns, every
applicable `B=1` case is preserved, and final performance has no unexplained route cliff, hidden
per-column expansion, or clearly inappropriate dispatch. Use effective GB/s/`READ_%` for a
memory-bound result and useful TFLOP/s for a compute-intensive result. `TC_%` requires an
established execution profile and must never be inferred from an `AllowA4` policy label. Use Nsys
or NCU only when launch topology, physical traffic, or instruction utilization would change the
route decision. There is no universal utilization cutoff: launch-limited small work is not a
failure merely because its percentage is low.

For Linear, the comparison is the calculated `T=1` linear extrapolation, never a timed sequence of
`T` separate launches. If legality, capacity, batching, route continuity, or final utilization
exposes a plausible route problem, continue immediately with task-local private-launcher
comparisons. Promote the winner into production dispatch, verify it through the same public Op and
oracle, and remove only losing candidates, forcing controls, duplicate benchmarks, and profiler
output files. Do not remove the selected production implementation merely because it began as a
candidate.

All work and evidence stop at the Op boundary: do not load a model artifact or invoke a target,
Engine, or DecodeRound path. Do not add concurrent Engine state, request metadata, selectors,
scheduling, CUDA Graph policy, or sequence-sensitive semantics. Do not commit unless the user asks.
After closing the current transaction, record its concise result; proceed to another row only when
the user has explicitly authorized the queue, and only after the prior transaction's independent
commit.

## Explicit handoff: not this worktree

Do not add these entries to the aggregate-column optimization list:

- `gqa_attention`, `gqa_attention_cached`, and `gqa_kv_append`;
- `gdn_input_proj_conv_snapshot` and `gated_delta_net_snapshot`;
- DFlash `prepare_masked_block`, `swa`, `bidirectional_gqa_attention`, and
  `kv_cache_append_prefix`;
- `sample`, speculative accept/select/commit, and `mtp_prepare_alignment_ids`;
- scalar, position, state-selector, scheduler, CUDA Graph, and Engine work;
- non-snapshot `gdn_input_proj`, `causal_conv1d_silu`, and `extract_bf16_columns`, which are
  prefill-only in the current production caller;
- Vision-only Ops and the MTP-prefill-only `[1024,5120]` `linear_pair` profile.

The important actual-path distinction is that Decode GDN calls the target-private
`gdn_input_projection_snapshot` leaf, not non-snapshot `gdn_input_proj`. The snapshot leaf and its
workspace/domain changes belong to the sequence-sensitive Linear Attention worktree.
