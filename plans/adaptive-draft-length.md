# Adaptive draft length (MTP + DFlash2)

Pick live draft count `K` from recent per-position accept and measured round time.
Identity: Qwen3.8-27B NVFP4, RTX 5090, `sm_120a`, `--kv-dtype nvfp4`, `--lm-head-draft`.

Artifacts:

- MTP: `/ssdpool2nvme/local_llm/models/qwen3.8-nvfp4-mtp-nvfp4-from-bf16/qwen3_8_27b_nvfp4.ninfer`
- DFlash2: `/ssdpool2nvme/local_llm/models/qwen3.8-nvfp4-flash2-nvfp4-bf16codebook-from-bf16/qwen3_8_27b_nvfp4_dflash_nvfp4.ninfer`

`--draft-tokens N` is the captured ceiling, not a frozen live `K`. Adaptive is opt-in until the
sweep beats static best `N` on both AIME and story; published MTP3 / DFlash k=7 numbers stay
reproducible without the flag.

## Nailed decisions

### CPU, not GPU

Every speculative round already returns to the host: graph launch → `device.synchronize()` →
egress D2H → output preview → Fold/resolve → next ingress H2D. Host fold+submit is ~0.4 ms vs a
~10–15 ms GPU round. Seam hiding and device tail-launch were already 1.000× / ~0.98× on this
identity. The policy is a few dozen FLOPs in that existing gap. Do not put it in a kernel, do not
overload `mtp_prepare_next_round` (that op stays budget/context clamp), and do not use CUDA Graph
conditionals or CLC.

### Evaluate every round; switch rarely

Update the EWMA from that round’s accepted prefix **every** round. Evaluate add/remove **every**
round. The check is free. Rate-limit **switches** with hysteresis plus a min dwell (same overlap
as the EWMA window), not by skipping rounds. Skipping 8–32 rounds delays thinking↔prose transitions
and does not save a measurable host gap.

### MTP ceiling is 5; do not lower it

Engine max is already 5. Static AIME already prefers MTP5 (195.9 tok/s, 3.06 tok/round) over MTP3
(184.7, 2.56). Lowering the ceiling throws away the high-accept regime adaptive is supposed to
reach. Capture `K ∈ {3,4,5}` as distinct graph topology classes (different AR step counts and
verify `T=K+1`). Add `K=2` only if the story/low-accept sweep beats `K=3`. Do not capture `K=1`
(extent 0 already falls back to one-token target progress).

Live MTP `K` steps by **one** inside `[k_min, N]`. Default `k_min=3` until `K=2` is kept.

### DFlash2 ceiling is not 5

`k≤5` with `W=k+1` is chain, one GQA SmallT tile (`T≤6`). `k=7` is packed-tree `W=12` (two tiles).
Measured AIME `(E,T)` puts tree last (147 tok/s vs chain k=5 193). Adaptive does **not** capture
tree. Frozen `--draft-tokens 7` without `--adaptive-draft` still runs k=7.

`--draft-tokens N` is still the storage ceiling:

- `N≥5` and `N≠6`: live `{3,4,5}` (chain W=4, W=5, W=6). Skip `K=6`. Tree `K=7` is not live.
- `N=7` (product default): storage W=12, live K ∈ `{3,4,5}` only. No tree hop.
- `N==6`: frozen `{N}`.
- `N==4`: frozen `{4}`.

Do not change the frozen DFlash product default away from 7 in this patch. Adaptive under
`--draft-tokens 7` never sits at 7.

### Graphs

K is a topology class, same as exact `B`: different AR/verify node counts cannot `cudaGraphExecUpdate`
across K. Context-frontier buckets still share an executable per `(B,K)` via update. Switching K
selects another pre-instantiated executable (like switching B), then fills `current_extents` /
`proposal_extents` as today.

MTP allowance today is 12 MiB × C (one class per B). Extra K classes multiply that; bump
`graph_allowance_bytes` from the captured inventory, do not guess.

Masking unused columns inside a max-K graph does not pick optimal K: linear/MoE/GDN still run at
captured T, and MTP still launches all `K-1` AR steps.

### Batching

One K per compact decode batch. Per-request EWMA. Mixed-K rows cannot split the batch.
`B=1` uses `max(row_k)` so C=1 policy is unchanged. `B≥2` picks the captured `K ≤ max(row_k)`
that maximizes `Σ_row E_row(K) / T_B(K)` with the C=2 T table (`kAdaptiveMtpC2T` /
`kAdaptiveDflashC2T`); ties keep the lower K.

### Policy

From a Kmax (or per-topology) run:

```text
s_i           = accepted_per_position[i] / rounds     # P(A ≥ i+1)
E[tokens | K] = 1 + sum_{i=0}^{K-1} s_i
tok/s(K)      ≈ E[K] / T(K)
```

`T(K)` is wall time of a graph captured at that K, NVFP4 KV, not a masked max-K replay.

Add `K → K+1` when `s_K > E[K] * (T(K+1)/T(K) − 1) * (1+δ_add)`.
Remove when `K−1` wins by a smaller `δ_remove`. Stay in the overlap.

- EWMA over ~32 rounds (overlap with min dwell ~8–16).
- Seed live K at the captured attractor, not ceiling `N`: MTP `4` if captured else nearest ≤4;
  DFlash `5` if captured else nearest ≤5. Frozen `|K|=1` seeds `N`.
- Ignore the first ~8 rounds (and any `pcur==0` fallback) before the first switch.
- `min(policy_K, budget_extent, ready_drafts)` — do not fight end-of-request clamp.
- MTP K increase may have `mtp_draft_count < K` for one round; the new graph proposes the extra
  drafts for the next round. Do not add a catch-up AR pass.

NVFP4 W4A4 starts at target `T≥4`, so MTP3–5 / DFlash k≥3 are a different quantization family than
MTP2. SmallT tile size is 6, so MTP stays one tile; DFlash k=7 is two tiles. Those jumps are why
T(K) is measured, not assumed linear.

## Measurement (before freezing δ and the capture set)

Same AIME seed `7632647173703958409` and story fixture as `docs/performance.md`, `--kv-dtype nvfp4`,
graphs on, C=1, then C=2/4 if concurrent is in scope.

1. Static MTP K=3,4,5 (and 2 if needed): `T(K)`, `s_i`, tok/s.
2. Static DFlash k=5 chain and k=7 tree: `(E,T)` at each. Keep k=4 only if it beats 5 on story.
3. Plug into add/remove inequalities; sweep EWMA window and `(δ_add, δ_remove)` vs static best.
4. Keep adaptive if it matches or beats static best on AIME **and** story (or wins the mix without
   giving back the AIME peak beyond a measured small delta). Revert capture of unused K.

Kill: no tok/s win vs static best K after hysteresis is tuned. Then do not ship the extra graphs.

## Implementation order

1. Capture MTP graphs for each kept K; fold K into `topology_class` with exact B; select at
   ingress fill. Host policy stub frozen at `N` (behavior = today).
2. Host EWMA + hysteresis; MTP adaptive behind the opt-in flag.
3. DFlash `{3,4,5}` chain (N≥5, N≠6). No tree k=7 in the adaptive live set.
4. Stats: report live K / histogram; `draft_window` remains the ceiling.
5. CLI/serving docs only after the sweep keeps a capture set.

## Remaining decisions (lock before LLD)

Sweep knobs that do **not** block LLD: EWMA window, δ_add/δ_remove, min dwell, warmup rounds,
whether MTP K=2 or DFlash k=4 joins the capture set.

These do:

### 1. Add direction without observing the next slot

At live K you only see `s_0..s_{K-1}`. `s_K` for the add test is not measured unless you are
already at K+1.

**Lock:** geometric continuation for unseen slots. Last-hot waives δ_add for 4→5 (and DFlash
4→5) when `T(k_hi) ≤ 1.20·T(live)`. Do not also require `score(k_hi) > score(live)`: that
reproduced r3 Python (held 4, 266 tok/s). Conditional accept
`p_i = s_i / s_{i-1}` (p_0 = s_0). Unobserved next slot uses geometric continuation
`ŝ_K = s_{K-1} * p̄` with `p̄` the more conservative of mean vs last observed conditional.
Remove uses live `s_{K-1}`. Seed at the captured attractor (MTP 4 / DFlash 5). Add-back is how
we climb after a drop.

DFlash adaptive is chain `{3,4,5}`. Do not continue any chain `s_i` into tree k=7; tree is not
a live class.

### 2. T(K) table shape

Round time grows with context; the *ratio* T(K+1)/T(K) may not. Online per-round timing is noisy
and couples to prefill/admit.

**Lock:** compiled C=1 T(K) from the NVFP4-KV sweep. Same table at every B and context for v1.
Revisit context buckets only if 8k vs 64k flips an add/remove inequality. Do not estimate T(K)
from masked max-K replays.

### 3. One Program, storage at ceiling, graphs slice

`draft_window` / `dflash_verify_width` stay the configured ceiling. Workspace, round frame, and
ReplaySSM arena stay sized for N (DFlash W=12 when N=7). Each kept K is a captured body with that
K’s k and verify width on a prefix of the same frame (`W=6` chain vs `W=12` tree).

**Lock:** no second Program, no runtime recapture, no per-request graph set.

### 4. Product surface

**Lock:** `EngineOptions.speculative.adaptive_draft` (CLI `--adaptive-draft`). Startup only —
graphs are a startup inventory. No per-request override. Family `Program` owns the policy;
variants only supply capture sets and T(K) tables. 35B MTP gets the same algorithm; 35B DFlash v1
is out of the first measurement (chain-only, different T(K)).

`SpeculativeStats.draft_window` remains N. Add a live-K histogram (or last K + rounds-at-K) after
the sweep keeps a set.

### 5. EWMA lifetime

**Lock:** per-request, reset at `install_sampling`. Do not persist across prefix-reuse turns. Do
not reset on reasoning→content (one sequence; extra state for a lag that hysteresis already
covers). Compact batch K = max of member live K at B=1. At B>=2, pick captured K maximizing
`Σ_row E_row(K) / T_B(K)` with compiled C=2 T(K); ties keep the lower K.

## Out of scope

GPU K decision, graph conditionals, device tail-launch, growing tree T/k inside one graph, k=6,
k=11 two-block, per-row K in one launch, default-on until the sweep lands, online T(K) profiling,
per-request adaptive flag, context-bucket T(K) in v1.

---

# Low-level design

## Ownership

| Piece | Owner |
|---|---|
| Capture-set + T(K) tables | Variant (`qwen3_6_27b` first; 35B MTP uses the same family hook with its own T(K) later) |
| Policy (EWMA, geometric add, hysteresis, dwell) | Family host header, no CUDA |
| Graph inventory, live K, ingress fill | Family `ProgramImplCore` |
| Flag | `SpeculativeOptions` → CLI / serve / bench |
| Stats | `SpeculativeStats` + existing lane copy-out |

Do not put policy in `src/ops`, `src/runtime` Engine PIMPL, or serve. Serve only forwards the startup flag.

## Capture set

```text
adaptive_draft_ks(backend, N):
  if !adaptive: return {N}
  MTP:    I = {3,4,5} ∩ [3..N]; return I.empty() ? {N} : I
  DFlash: if N==6: {N}
          if N>=5: {3,4,5}       # N is still the storage ceiling; live K ∈ {3,4,5}
          return {N}             # N<=4 frozen
  never include 1, 2, 6, 7, 8..11 as adaptive classes
```

`--adaptive-draft` with `--draft-tokens 1` or `2` is valid CLI but captures `{N}` only (no step). Same as frozen K.

`--adaptive-draft` without `--spec` is an error.

## Topology class

Today: `topology_class = planned.topology_class * C + (B-1)`.

With K, SequencePlan builds the expanded inventory **before** Program capture:

```text
k_index  = index of this K in captured_ks (0..|K|-1)
k_stride = C * (1 + max planned.topology_class over ALL captured k)
         # 27B MTP/DFlash: max=0 → k_stride=C
         # frozen k_index=0 → planned*C+(B-1), identical to today
topology_class = k_index * k_stride + planned.topology_class * C + (B-1)
```

27B MTP: call `mtp_graph_profiles(capacity, k)` per captured k (bucket ends depend on k).
DFlash: per-k `W(k)` for 64 vs 96 MiB classes. 35B DFlash planned is
`(chunked_target?2:0)|(split_swa?1:0)`; k=5 W=6 is never chunked; k=7 W=12 can be. `k_stride`
must use the max over the union.

`DecodeGraphProfile` gains `draft_k` and `verify_width`.
`select_graph_profile(family, B, frontier, k)` matches `batch_size`, frontier bucket, **and**
`profile.draft_k == k`.

`graph_allowance_bytes` is `graph_topology_allowance` over this expanded, already-folded list.
Do not wait until capture. Frozen `|K|=1` allowance equals today.

## Capture / run

Workspace, round frame, DFlash pending features, and ReplaySSM stay sized at ceiling `N` /
`dflash_verify_width(N)` (W=12 when N=7). Host ingress/egress stay packed at that ceiling:

```text
MTP current_drafts[row * N + j]
licensed / fold_path[row * W_ceil + j]
```

Do not re-pack host blobs to live k. `Tensor::slice` on dim 0 of `{N,C}` at C>1 is not
contiguous; `speculative_prepare_verify_inputs`, accept, DFlash propose/path_select require
contiguous `[k,B]` / `[W(k),B]`. Use the existing `copy_i32_panel` / workspace compact (already
used for k=7 propose W=8 on a W=12 frame). MTP AR `{B, N-1}` **may** slice dim 1 to `k-1`
(contiguous). Frozen `|K|=1` capture body is today's body — **no extra compact nodes**.

ReplaySSM physical width = `W_ceil`. `gated_delta_net_replay_record` requires contiguous
`record.ne[2] == q.ne[2]`. Adaptive K graphs:

1. Record into graph-stable **workspace** tensors shaped exactly `W(k)` (fit in the existing
   round workspace; addresses captured per executable).
2. `cudaMemcpy2D` (or equivalent) packs columns into persistent ReplaySSM at physical `W_ceil`.
3. Fold reads persistent ReplaySSM; `commit_columns` / `path[]` ≤ this-round `W(k)` ≤ `W_ceil`.

Frozen `|K|=1`: record directly into persistent ReplaySSM as today.

`PendingCandidate` stores this round's `round_k`, `verify_width`, `tree_verify`.
`resolve_pending_batch` uses those, **not** `Program::draft_window`. Chain: `path_length = -1`.
Tree: pack `fold_path` at `row * W_ceil + i`.

For each `k` in `captured_ks`, each `B=1..C`, each frontier profile:

- MTP: `capture_mtp_decode_batch(..., k, mtp_gqa_envelopes(planned.max, k, capacity), ...)`.
  Body uses `k` for AR steps and `T=k+1`. Compact verify/draft panels to contiguous `[k,B]` for
  Ops; `mtp_prepare_next_round` sees `verify_ids.ne[0] == k+1` so `next_extents ≤ k`.
- DFlash: `capture_dflash_decode_batch(..., k, W(k), ...)`. k=3 → chain W=4; k=4 → chain W=5; k=5 → chain W=6.
  `dflash_uses_tree_verify(k, W(k))` at capture k. `dflash_envelopes` append stays `{0,k+1}`.

Eager warm stays `B=1` at `captured_ks.back()` (ceiling). Without graphs, eager body uses live k
the same way; compact still required at C>1 when k < N.

## Live K state (`RequestControl`, not `SequenceState`)

Reset in `install_sampling` (new request). Prefix-reuse / RAM restore / VRAM retain install a new `RequestControl` and reset. Sequence KV/hidden/mtp_drafts are unchanged.

```text
struct AdaptiveDraftState {
  uint32_t live_k;
  uint32_t rounds_at_k;
  uint32_t observed;
  float s[5];                 // chain slots i=0..4 only; never index with drafted>5
  uint64_t rounds_hist[16];   // index = K
};
```

Constants (host-only, named, not preprocessor):

- `kAdaptiveEwma = 32` → `alpha = 2/(32+1)`
- `kAdaptiveWarmupRounds = 8`          # add
- `kAdaptiveFirstRemoveWarmup = 32`    # MTP first remove
- `kAdaptiveDflashFirstRemoveWarmup = 8`  # DFlash first remove (seed 5 is story's slower K)
- `kAdaptiveMinDwell = 8`
- `kAdaptiveDeltaAdd = 0.05`
- `kAdaptiveDeltaRemove = 0.02`  # must stay below T[4]/T[5]−1 ≈ 0.031 or 5→4 never fires
- `kAdaptiveDropSlotMax = 0.30`  # extra slot s[K-1] must be this dead to remove
- `kAdaptiveClimbSlotMin = 0.35` # last slot hot waives add hysteresis if score still improves
- `kAdaptiveHighP0MinK = 0.65`   # s[0] above this floors live K at 4 (story 4 vs dialogue 3)
- `kAdaptiveDropTo3Max = 0.08`   # DFlash 4→3 extra must be this dead (dialogue 0.014, not story 0.115)

Sweep may replace these numbers in one header. Policy tests pin the names, not magic literals.

Measured T(K) / campaign E, C=1 NVFP4 KV, Qwen3.8-27B, RTX 5090, seed `7632647173703958409`
(AIME 4096 + ninfer_bench pp2048+tg256). Policy tests may still inject their own table.

```text
MTP    T[3]=1.00  T[4]=1.10  T[5]=1.16     # AIME T = 13.53 / 14.83 / 15.64 ms
DFlash T[3]=0.91  T[4]=0.97  T[5]=1.00  T[7]=1.43
       # AIME T = 13.87 / 14.79 / 15.24 / 21.84 ms
E5=2.94  E7=3.21                           # AIME tokens/round
```

## Policy (pure function)

Header: `src/targets/qwen3_6/impl/runtime/adaptive_draft.h` (family, no Variant CUDA).

```text
next_k = adaptive_draft_next(
    cfg,              # captured_ks, T[], deltas, warmup, dwell, ewma
    state,            # in/out EWMA + dwell
    accepted,         # this round accepted draft count A
    drafted,          # pcur; 0 = fallback, do not update EWMA, do not switch
    budget_extent)    # already min(remaining-1, context)
```

Update (after warmup, `drafted>0`):

```text
for i in 0..min(drafted,5)-1:  s[i] = (1-α)s[i] + α * (A > i)
# never write s[i] for i >= drafted or i >= 5
```

Expected tokens at candidate `k` (k in captured_ks, k <= budget_extent unless budget is the only clamp):

```text
E[k] = 1 + sum_{i=0..k-1} ŝ_i
ŝ_i  = s[i] if s[i] was updated at least once
     else geometric: ŝ_i = ŝ_{i-1} * p̄
p̄    = mean of observed p_j = s[j]/s[j-1] for j with s[j-1]>ε, else campaign prior 0.5
```

Score `E[k]/T[k]`. Consider `{live_k-1, live_k, live_k+1}` intersected with `captured_ks`.

- Switch up if dwell and `observed >= warmup` and (`score(k_hi) > score(live) * (1+δ_add)` **or** last slot `s[K-1]` seen and `> climb_slot_min` and compiled `T(k_hi) ≤ 1.20·T(live)` **and not** a 3→4 last-hot). Last-slot-hot waives add hysteresis for 4→5 / DFlash, not a worse T, and not 3→4. Unseen tail `p̄ = min(mean conditional, last conditional)`.
- Switch down only if dwell, `observed >= first_remove_warmup` (32; add still at 8), extra slot `s[K-1]` is **seen and `< drop_slot_max`**, and `score(k_lo) > score(live) * (1+δ_remove)`. Unknown extra slot is not dead.
- After the neighbor pick: if `observed >= warmup` and `s[0] > high_p0_min_k` and 4 is captured, floor `live_k` at 4. The first boolean `s[0]=1` does not floor.
- Else keep live_k

Clamp: `min(next, budget_extent, captured max)`; if budget_extent is 0 the existing ordinary fallback applies (do not call policy).

Seed `live_k` at `adaptive_seed_k(captured, backend)` (MTP 4 / DFlash 5). First `warmup` counted rounds never **add**; first remove waits `first_remove_warmup`.

DFlash adaptive does not score k=7. Frozen k=7 remains a non-adaptive topology.

## When it runs

In `decode_mtp_batch` / `decode_dflash_batch`, **before** graph select and ingress fill:

1. `batch_k = 0`
2. For each row: `row_k = adaptive ? state.live_k : draft_window`, then
   `row_k = min(row_k, budget_extent, capacity-E-1)`. Ready MTP drafts do **not** cap graph K
   (a climb uses the new graph this round to propose extras; verify masks via `current_extents`).
3. `B=1`: `batch_k = max(row_k)`. `B≥2`: `adaptive_batch_k_sum_score` over captured `K ≤ max(row_k)`. Snap into `captured_ks`.
4. `select_graph_profile(..., batch_k)`
5. Fill per-row `current_extents` / `proposal_extents` as today with `extent = min(ready, batch_k, budget, cap)` — a low-accept row in a high-K batch still verifies at `batch_k` graph T but `valid_columns = extent+1`.
6. After egress D2H, **update** each row’s EWMA from that row’s `accepted_i` / `pcur`. Then `adaptive_draft_next` writes `live_k` for the **next** round. Graph this round already used `batch_k` from pre-round peek.

MTP increase: this round’s graph is the new `live_k`; `mtp_draft_count` may be smaller; `current_extents` is the ready count; `valid_columns` masks extra verify columns; the captured-k AR loop proposes `next_extents ≤ batch_k` for next round. No catch-up AR.

`mtp_prepare_next_round` still writes `next_extents` from budget/context with the **captured k** of this graph (not the host policy). Host `resolve_pending_batch` keeps storing `next_extents` into `mtp_draft_count`. Next round policy may choose a different graph; ready drafts are `mtp_draft_count`.

## Prefill / first decode

Prefill bridge still proposes `min(N, budget)` drafts (`mtp_proposal_extent` stays ceiling). First decode seeds `live_k=N`, so the first graph is the ceiling graph. Warmup then allows drops. DFlash has no stored drafts; first decode uses `live_k=N`.

## Concurrent / RAM / VRAM / SSD

Adaptive K does not change page accounting. Interactions that must keep working:

- **VRAM retain**: `retain_lane` clears `mtp_draft_count`. Next occupant `install_sampling` resets adaptive state. No K leak across requests on the same lane.
- **RAM second tier**: `capture_retained_lane` / `restore_ram_entry` restore SequenceState only. Adaptive state is not in the RAM image. Restore + new generate → fresh `live_k=N`.
- **SSD**: artifact mmap / NVMe is not a KV tier. No adaptive state on disk. Concurrent decode while another lane RAM-restores from host (copy_stream) must not read/write another lane’s `AdaptiveDraftState` (it is host-only per RequestControl).
- **Maximal batch**: one `batch_k`. A RAM-restore that joins at a boundary uses `live_k=N` until its own warmup; `batch_k = max` may pull a warmed neighbor up. That is accepted.
- **Cancel / stop / budget 1**: `pcur==0` or extent 0 → no EWMA update; existing fallback_steps path unchanged.
- **Vision+MTP**: adaptive allowed; DFlash+Vision still rejected at Engine construct.
- **CUDA graph off**: same policy, eager `k=batch_k`.

## Public types

```cpp
struct SpeculativeOptions {
  ...
  bool adaptive_draft = false;
};

struct SpeculativeStats {
  ...
  std::uint32_t live_draft_tokens = 0;          // last live K
  std::vector<std::uint64_t> rounds_per_draft;  // index = K, size N+1
};
```

`draft_window` remains ceiling N. Serving JSON adds `live_draft_tokens` and `rounds_per_draft` next to `accepted_per_position`. CLI stderr prints live K.

`validate_speculative_cli_options`: `--adaptive-draft` requires backend MTP or DFlash. Unchanged draft_tokens ranges.

## Files (expected)

- `include/ninfer/types.h` — flag + stats fields
- `src/product/speculative_options.h` — validate
- `apps/cli/options.cpp`, `src/serve/parse` / serve_options, bench support — `--adaptive-draft`
- `src/targets/qwen3_6/impl/runtime/adaptive_draft.h` — policy
- `src/targets/qwen3_6/impl/runtime/program.h` / `program_impl.h` — state, capture, select, decode
- `src/targets/qwen3_6/impl/runtime/layouts_impl.h` — graph allowance from expanded profiles; SequencePlan stores `captured_ks`
- `src/targets/qwen3_6_27b/impl/variant.cpp` — T(K) / E_campaign placeholders; `mtp_graph_profiles` already takes draft_window (call per k)
- `docs/cli.md`, `docs/serving.md` — flag; no performance.md rewrite until sweep
- Tests: host policy unit test (no GPU); graph inventory test in `test_runtime_mechanisms.cpp`; real Engine tests extend existing MTP/DFlash NVFP4 with `--adaptive-draft` and overlapping C=3; RAM/prefix tests with adaptive on

## Test matrix (LLD contract)

Host (no GPU): geometric add, remove, dwell block, warmup block, pcur==0 ignore, budget clamp,
DFlash `{3,4,5}` +1 chain, `batch_k = max(row_k)` at B=1, `adaptive_batch_k_sum_score` at B≥2.

Mechanisms: SequencePlan with adaptive captures more topology classes than frozen N; allowance ≥ frozen.

Real GPU (env artifacts): do **not** require adaptive greedy tokens == frozen-N greedy.
Different K changes the licensed bonus when a later draft would have matched. Require:
(1) policy unit tests, (2) adaptive Engine completes with the right backend, (3)
`sum(rounds_per_draft) == rounds`, `live_k` in captured set, (4) overlapping C=2/3 greedy
completes distinct prompts, (5) RAM restore / VRAM retain / prefix reuse reset adaptive to
`live_k=N` on the new request and do not leak K across lanes, (6) frozen `adaptive_draft=false`
still matches existing MTP k=3 / k=5 and DFlash k=5 / k=7 greedy oracles.

## Implementation order

1. Policy header + host tests (no GPU).
2. `SpeculativeOptions.adaptive_draft` plumbing + docs.
3. Capture/select by K; frozen path is `captured_ks={N}` (bit-identical topology to today when adaptive is off).
4. Wire policy into decode batches + stats.
5. DFlash `{3,4,5}` chain slice capture (W=4, W=5, W=6 under W_ceil).
6. Real Engine tests with artifacts.
