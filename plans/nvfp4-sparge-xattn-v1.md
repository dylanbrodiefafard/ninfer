# Handoff: Sparge off S3, onto exact NVFP4; then XAttention prefill v1

Written 2026-08-24. For the agent that picks this up after the in-flight GQA
launcher/kernel split. This is **not** the SageAttention3 speed campaign
(`ninfer/HANDOFF-nvfp4s3-attention.md`). Do not mix the two.

## Mission

Ship a **prefill-only** sparse-attention A/B on the **production exact NVFP4**
GQA kernels (no FP4-P). Two rankers share one skip-list iterator:

1. **nvfp4sp** — Sparge `keep_frac` + `k_mean` meansim (what lives on S3 today).
2. **nvfp4xattn** — XAttention antidiagonal scoring + mass threshold **τ**.

Then measure quality and kernel time against dense `kv-nvfp4`. Decision: keep
XAttention, keep Sparge as a control, or drop skip.

## Why this order (do not re-litigate)

- Production prefill/decode is exact NVFP4 (`gqa_attention_prefill_nvfp4.cuh`,
  BF16 PV). Sage3 (`nvfp4s3`) quantizes P; decode PPL already costs **+0.028 NLL**
  vs exact NVFP4 (`ninfer/sage-sparge-ppl-20260822.md`). Tile-skip on S3 compounds
  two approximations.
- Current Sparge is wired to `sage_pv` (`--keep-frac` requires `--sage`; `k_mean`
  is allocated only when `sage_pv`). That coupling is the bug.
- XAttention (arXiv 2503.16428) is prefill block-sparse. Its output is a
  **keep-list of key tiles**, which is already the S3 skip geometry (Bc=64 =
  `kPagedKVPageSize`). It does **not** use `keep_frac`. The knob is **τ**
  (cumulative predicted-attention mass). Paper default **τ = 0.9**; at 256k that
  kept ~7% of blocks. `keep_frac=0.9` would keep 90% of tiles. Do not alias them.
- FlexPrefill is not this v1. Vertical-slash is a different gather kernel.
- Decode skip is **out of v1**. At C=1 / 260k, GQA is ~32% of the decode step;
  DFlash/DSpark is the decode lever. S3 decode skip stays disabled or is stripped.
  Width > 1 (MTP/DFlash verify) must remain exact.

## Current state (as of this write)

| Piece | Where | Notes |
|---|---|---|
| Exact NVFP4 prefill | `src/ops/kernel/gqa_attention_prefill_nvfp4.cuh` | No skip. Dense tile loop. Production. |
| Exact NVFP4 decode | `src/ops/kernel/gqa_attention_decode_nvfp4.cuh` | No skip. Leave it. |
| S3 prefill + Sparge | `src/ops/kernel/gqa_attention_prefill_nvfp4s3.cuh` | `keep_frac<1` + `k_mean` ranks tiles; ExactT template vs approx. |
| S3 decode + Sparge | `src/ops/kernel/gqa_attention_decode_nvfp4s3.cuh` | T=1 only; `keep_frac<1` needs `k_mean`. |
| S3 TMA prefill | `src/ops/kernel/gqa_attention_nvfp4s3_tma.cuh` | Already refuses skip (`keep_frac` unused; launcher requires `keep_frac==1`). |
| Launchers (split in progress) | `gqa_attention_prefill.cu`, `gqa_attention_prefill_s3.cu`, `gqa_attention_decode.cu`, `gqa_attention_decode_s3.cu` | Dispatch: `dtype==U8 && sage_pv` → S3; else exact NVFP4. |
| `k_mean` plane | `PagedKVLayerView::k_mean_pages`, filled only on sage fill | Comment: "sage_pv only". |
| Engine knob | `NINFER_KEEP_FRAC`, `--keep-frac`, `TextContext::keep_frac_` | `--keep-frac <1` currently requires `--sage`. |
| PPL schemes | `tools/ppl/schemes.py` | `attn-topk` = `--sage --keep-frac 0.5`. |
| 8k PPL (sage+sparge) | `ninfer/sage-sparge-ppl-20260822.md` | keep 0.5 = +0.38% PPL prefill; keep 0.2 = +4.92%. Decode skip was off. |

Geometry: Qwen3.6/3.8-27B full-attn layers only — 16 of 64, `24Q/4KV/d=256`,
page 64. Skip never applies to GDN.

**Another agent is splitting kernels.** Do not re-merge `prefill.cu` /
`prefill_s3.cu`. Add sibling launchers the same way (`prefill_sp.cu`,
`prefill_xattn.cu` or one `prefill_sparse.cu` with a ranker enum). If the split
is mid-flight, wait for it to land or rebase onto those files; do not fight
the rename.

## Target architecture

```
                    ┌─────────────────────┐
  NVFP4 KV fill ───►│ exact NVFP4 prefill │  dense inner loop today
                    │  + skip-list walk   │◄── keep_tiles[n]
                    └─────────────────────┘
                              ▲
              ┌───────────────┴───────────────┐
              │                               │
     Sparge ranker (nvfp4sp)         XAttention ranker (nvfp4xattn)
     k_mean · q_mean, keep_frac      strided antidiag QK, τ
     sinks + recency + top-k         sinks + recency + mass≥τ
```

**Do not copy the entire MMA kernel twice** unless isolation is the only way to
A/B without touching production. Prefer:

- One exact-NVFP4 prefill kernel whose K-loop iterates `keep_list` (identity
  list when dense).
- Two rankers that write that list (device kernel or the top of the same CTA,
  matching today's S3 `proxy_scores` / `keep_list` smem pattern).

If you must fork for bisectability, the parent is
`gqa_attention_prefill_nvfp4.cuh`, **not** the S3 file. S3 has FP4-P, d-major V
scales, and a different numerical contract.

Suggested names (files, not user-facing flags):

| Name | Ranker | Compute |
|---|---|---|
| `nvfp4` | none (keep all) | exact NVFP4 QK + BF16 PV (production) |
| `nvfp4sp` | Sparge `keep_frac` + `k_mean` | same compute, skipped tiles |
| `nvfp4xattn` | XAttention τ + stride S | same compute, skipped tiles |
| `nvfp4s3` | none after this work | Sage3 FP4-P, exact tiles only |

## v1 knobs

### Sparge (`nvfp4sp`)

| Knob | Semantics | Expose |
|---|---|---|
| `keep_frac` ∈ (0, 1] | Fraction of Bc=64 key tiles kept (plus sinks/window rules already in S3). `1.0` = dense. | Yes. Default `1.0`. A/B: `0.5`. |
| `k_mean` plane | Per-page dequantized K mean, independent of `sage_pv`. | Internal. Allocate when `keep_frac<1` on NVFP4. |

Keep the existing S3 keep-set rule for the port (topk + ~20% sinks + ~40% recency
window, all scaled by `keep_frac`) so the first A/B is "same skip, different
compute" vs today's sage+sparge, then "same skip, exact NVFP4 vs dense NVFP4".

### XAttention (`nvfp4xattn`)

| Knob | Semantics | Expose |
|---|---|---|
| `xattn_tau` ∈ (0, 1] | Keep the smallest tile set whose softmax-normalized antidiagonal scores sum to ≥ τ. `1.0` = dense. | **Yes. This is the serve/PPL flag.** Default `1.0`. A/B: **0.9**. |
| `S` | Antidiagonal stride. Paper: 8 (quality) or 16 (faster rank). | Env/debug (`NINFER_XATTN_STRIDE`). v1 default **16**. Not a request flag. |
| `B` | Score/skip block size. | **No.** Bind to 64. |
| Per-head τ | Paper §2.3 DP. | **No** in v1. Uniform τ. |
| Min length | Skip ranking below this prompt length. | Internal, e.g. 8192. Short prompts stay dense. |

**Do not reuse `--keep-frac` as τ.** Add `--xattn-tau`. Mutually exclusive with
`keep_frac<1`.

Prefill / Prompt route only. `gqa_attention_cached` / SmallT / MTP verify stay
exact (`keep_frac` forced 1, `xattn_tau` ignored).

## Tasks

### 0. Sync with the kernel-split agent

- Confirm `gqa_attention_prefill.cu` + `gqa_attention_prefill_s3.cu` is the
  landed layout (already present when this was written).
- Do not reintroduce skip into the TMA S3 kernel. TMA stays exact-S3 only.

### 1. Make S3 exact-only

Strip tile-skip from S3 prefill and S3 decode:

- `keep_frac != 1.0f` + `sage_pv` → throw (or ignore with a one-time stderr, then
  throw — prefer throw so PPL cannot silently drop skip).
- Delete or dead-code the `approx` branch / `gqa_attention_decode_rank_nvfp4s3_kernel`
  once nothing calls it.
- `attn-sage` and `attn-tma` PPL cells must remain bit-identical to today at
  `keep_frac=1`.

Acceptance: `ninfer_gqa_attention_test` sage exact path green; sage+`GQA_KEEP_FRAC<1`
either gone or fails closed.

### 2. Allocate `k_mean` without `sage_pv`

Today `decoder_state.cpp` and the GQA fixture only attach `k_mean_pages` when
`sage_pv`. Sparge-on-exact-NVFP4 needs that plane whenever skip is on.

- Allocate `k_mean` for NVFP4 caches when `keep_frac<1` (and later when any
  ranker needs page stats).
- Fill it from the **exact NVFP4 fill** path (`gqa_attention_prefill_fill_nvfp4_kernel`
  or a sibling), dequantizing E2M1×UE4M3 K to the same layout S3 fill writes
  (`[4][64][kv_heads][pages]` — see `paged_kv_cache.h`).
- `sage_pv` remains "FP4-PV compute recipe", not "has k_mean".

### 3. nvfp4sp: skip-list on exact NVFP4 prefill

Port the S3 keep-list walk onto `gqa_attention_prefill_nvfp4.cuh` (or
`gqa_attention_prefill_nvfp4sp.cuh` if forked):

- Ranker: copy the meansim block from S3 prefill (`q_mean` of the 128-row Br
  tile · `k_mean` per kb).
- Inner loop: iterate `keep_list[0..k_keep_count)` instead of `kb=0..key_blocks`.
- Causal mask, online softmax, BF16 PV: unchanged from production NVFP4.
- Prompt launch in `gqa_attention_prefill.cu`: `dtype==U8 && !sage_pv && keep_frac<1`
  → sp kernel; `keep_frac==1` → current dense kernel (or the same kernel with
  identity keep_list — pick one and test both occupancy paths).

CLI:

- `--keep-frac <f>` allowed with `--kv-dtype nvfp4` **without** `--sage`.
- `--keep-frac <1` **rejected** with `--sage`.

PPL: change `attn-topk` to `("--keep-frac", "0.5")` (no `--sage`). Add
`attn-topk-0.2` only if you have GPU time; 0.2 is the known PPL cliff.

Oracle (`tests/ops/test_gqa_attention.cpp`):

- New skip cases on **exact NVFP4** (not sage criterion): device output vs
  FP64 softmax over **kept tiles only**, using the dumped keep set (same dump
  shape as `GqaS3PrefillDump::{keep_list,tile_count}` — rename to a shared
  skip dump if you touch it).
- Dense NVFP4 vs skip `keep_frac=1.0` must be bit-identical (or max-rel at the
  existing NVFP4 envelope). That proves the skip iterator is the identity.

### 4. nvfp4xattn: XAttention ranker, same skip iterator

New prefill ranker only. Paper algorithm 1, bound to this engine:

- `B = 64`. For each query tile (Br=128 may span two B-rows — score per CTA
  query tile; union keep-sets if you split).
- Stride `S=16`: reshape `Q[i::S]`, `K[i::S]` inside the tile, NVFP4 or BF16
  dequant QK (BF16 dequant of strided K is fine for v1 ranker; do not require
  a second NVFP4 MMA just to rank).
- Softmax the approximate scores; `find_blocks` until cumulative mass ≥ `τ`.
- Always union **sinks + last local tile** into the keep set (engine safety,
  even if the paper's scores would keep them).
- Write the same `keep_list` the sp kernel consumes. **Same MMA loop.**

CLI:

- `--xattn-tau <t>` in (0, 1]. Requires NVFP4, not sage. Conflicts with
  `keep_frac<1`.
- `NINFER_XATTN_STRIDE` = 8 or 16.
- Engage only if prompt length ≥ 8192 (constant ok).

PPL schemes:

```
attn-xattn     kv-nvfp4  --xattn-tau 0.9
attn-xattn-95  kv-nvfp4  --xattn-tau 0.95   # optional
```

Oracle:

- Dump keep_list; FP64 reference attends only those tiles; compare to device
  with the **exact NVFP4** envelope (not sage). Ranking bugs show up as a
  keep-set mismatch vs a host XAttention reference, not as MMA error.

### 5. A/B matrix (isolated GPU, not live `:8081`)

Project rule: **do not touch the live engine.** Isolated `ninfer-ppl` /
op benches only. User restarts serve if they want speed-suite.

**Quality (required before any speed claim)**

Same artifact as the perplexity campaign:
`qwen3.8-27b/nvfp4` (path in `plans/perplexity-campaign.md`).
`--skip half`, graphs on, prefill lane first.

| Cell | Flags | What it proves |
|---|---|---|
| `kv-nvfp4` | exact | Baseline |
| `attn-topk` | `--keep-frac 0.5` no sage | Sparge-on-exact vs dense |
| `attn-xattn` | `--xattn-tau 0.9` | XAttention vs dense |
| `attn-sage` | `--sage` keep all | S3 still exact; no skip regression |

Lengths: **8k then 32k** prefill. 8k is too easy (old keep 0.5 was +0.38% PPL).
Do not declare victory without 32k. If 32k is clean, 64k NIAH is the next gate
(not optional if you want this in serve).

Decode lane: all of the above must match `kv-nvfp4` decode NLL (skip must not
engage). If decode NLL moves, you leaked skip onto SmallT.

**Speed (after quality)**

- Isolated prefill bench: clone `bench/ops/gqa_attention_nvfp4s3_bench.cu` onto
  exact NVFP4 ± skip. Report median µs at ctx 8k/32k/64k/128k, Br=128, 4096-query
  chunk, Gqa27Geometry.
- Compare: dense nvfp4 vs nvfp4sp keep 0.5 vs nvfp4xattn τ=0.9.
- Ranker time vs MMA time separately if the ranker is a distinct launch.
- Do not quote the Sage3 64k 33257 µs number as the baseline; re-measure exact
  NVFP4 on the same binary.

Pass bar for v1 ship-as-experiment (not production default):

- Prefill 32k: ΔPPL vs `kv-nvfp4` **< 1%** (tighter than the old 0.1 NLL sage
  gate if mean NLL ~1.9; use both mean_nll and PPL).
- Non-finite NLL = fail.
- NIAH at the longest length you claim speedup: needle tile must be kept
  (if NIAH fails, raise τ or keep_frac; do not ship).
- Dense `keep_frac=1` / `tau=1` bit-identical to `kv-nvfp4`.

## Files you will likely touch

```
src/ops/kernel/gqa_attention_prefill_nvfp4.cuh          # skip-list loop, or leave dense
src/ops/kernel/gqa_attention_prefill_nvfp4sp.cuh        # if forked
src/ops/kernel/gqa_attention_prefill_nvfp4xattn.cuh     # ranker, or ranker.cuh shared
src/ops/kernel/gqa_attention_prefill_nvfp4s3.cuh        # strip skip
src/ops/kernel/gqa_attention_decode_nvfp4s3.cuh         # strip skip
src/ops/launcher/gqa_attention_prefill.cu
src/ops/launcher/gqa_attention_prefill_s3.cu
src/ops/launcher/gqa_attention.h
include/ninfer/ops/gqa_attention.h                      # keep_frac docs; add tau
src/core/paged_kv_cache.h                               # k_mean not sage-only
src/targets/qwen3_6/impl/state/decoder_state.cpp         # allocate k_mean
src/targets/qwen3_6/impl/runtime/text_context*.h         # tau + keep_frac routing
apps/ppl/main.cpp
tools/ppl/schemes.py
tests/ops/test_gqa_attention.cpp
bench/ops/gqa_attention_nvfp4_sparse_bench.cu           # new isolated A/B
```

## Do not

- Port skip onto Sage3 TMA or re-enable `NINFER_S3_TMA` with `keep_frac<1`.
- Stack Sage3 FP4-P and tile-skip in one recipe (`--sage --keep-frac` or
  `--sage --xattn-tau`).
- Implement FlexPrefill slash gather, DuoAttention streaming heads, Quest decode
  paging, or NSA/MoBA.
- Use `keep_frac` as XAttention τ.
- Engage skip on T>1 (MTP/DFlash verify) or on GDN layers.
- Restart/stop the live `:8081` engine.
- Claim e2e tok/s from kernel µs alone; GQA is 16/64 layers.

## Suggested commit slices (if asked to commit)

1. S3 exact-only + CLI reject sage+skip.
2. `k_mean` without `sage_pv` + nvfp4sp prefill + oracle + `attn-topk` scheme fix.
3. nvfp4xattn ranker + `--xattn-tau` + oracle + PPL scheme.
4. Isolated bench numbers in a short note under `profiles/bench/` (gitignored
  json ok; a few lines in this plan or BENCHMARK.md if the user wants it kept).

## References

- XAttention: https://arxiv.org/abs/2503.16428 — Algorithm 1, τ, S ∈ {8,16}, B=64.
- SpargeAttn: https://arxiv.org/abs/2502.18137 — meansim is the current S3 proxy.
- Prior quality: `ninfer/sage-sparge-ppl-20260822.md`.
- PPL how-to: `repo/plans/perplexity-campaign.md`, `repo/tools/ppl/README.md`.
- Fit notes: conversation 2026-08-24 (LServe / FlexPrefill / XAttention). XAttention
  is the first new ranker because it emits the keep-list this kernel already wants.
