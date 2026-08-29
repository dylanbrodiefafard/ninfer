# NVFP4 decode-band Linear mapping (RTX 5090 / sm_120a)

This is the active mapping argument for Qwen 3.8-27B serving Linear (ordinary decode,
MTP, DFlash2 chain verify). It is not a prefill document and not an e2e tok/s claim.
Procedure remains [`kernel-iteration.md`](kernel-iteration.md). Public measurement is
[`linear-benchmark.md`](linear-benchmark.md) (`ninfer_linear_bench` and the fused Linear
family benches).

Do not revive fused-into-MMA activation quantize, TMEM/`tcgen05`/2-SM, or a handwritten
`mma.sync` inner loop to chase decode tok/s. Those were measured or classified and lost.

## Serving T

- `C = max_concurrency`, `B` = compact batch this round (`1…C`). Saturated `C=2` ⇒ `B=2`.
- MTP width = `k+1`. MTP3 ⇒ 4; `k=5` ⇒ 6. DFlash2 chain `k≤5` ⇒ width 5 or 6.
- Verify Linear `T = width × B` (MLP / attn-in / lm_head pack the compact batch). Primary points: `T = 4,5,6,8,12` (and 18,24,36 secondary).
- Ordinary decode / MTP proposal AR: `T = B` (`1,2,3`). Cover them; they are not the tok/s point.

## Roofline

\[
Y_{T\times N}=X_{T\times K} W_{N\times K}^{\top},\quad
B_W=\tfrac{9}{16}NK,\quad
\mathrm{AI}=\tfrac{32}{9}T\approx 3.56\,T\ \mathrm{FLOP/byte}.
\]

Ridge at 1676 TFLOP/s and 1674.5 GB/s is \(\approx 1000\) FLOP/byte \(\Rightarrow T_{\mathrm{ridge}}\approx 280\).
Serving \(T\le 36\) is 10–100× below ridge. Reusing \(W\) across T, tensor-core issue rate, and
MMA instruction count do not move wall time. The only Linear floor is one pass over \(W\):

\[
t\ge t_{\mathrm{mem}}=B/1674.5\,\mathrm{GB/s}.
\]

Attn-in T=4: \(B_W=41.29\) MB, \(t_{\mathrm{mem}}=24.7\,\mu\mathrm{s}\). Production W4A4 public Op
\(\approx 32.8\,\mu\mathrm{s}\) (75% of the copy-probe roof). Same payload as memcpy; the kernel
is slower because of the **access pattern and grid**, not missing FLOPs.

Whole-decoder NVFP4 weights \(\approx 13.7\) GB \(\Rightarrow \approx 119\) tok/s with no speculation
(one token per full weight pass). 290 tok/s without spec is not a Linear-issue ceiling on this
HBM; it matches speculative Linear-only arithmetic (MTP3 × 119) if the rest of the round were free.

## What the current W4A4 decode kernel does wrong

Decode W4A4 is output-stationary tiled GEMM: grid \((N/64,\lceil T/32\rceil)\), `BlockK=256`,
3-stage `cp.async`, two `__syncthreads` per K-tile, `ldmatrix` + `m16n8k64`. That machine exists
so many T can reuse a \(W\) tile from smem. At serving T the reuse has value \(\approx 0\).
The T=1 GEMV path already streams the same \(W\) at the same ~75% roof.

Measured READ% vs 1674.5 tracks **how \(N\) is partitioned**, not useful FLOPs:

| Problem | BlockN | CTAs | vs 170 SMs | T=4 READ% |
|---|---:|---:|---|---:|
| mlp-up `[34816,5120]` | 128 | 272 | 1.6 waves | **~86%** (control) |
| attn-in `[14336,5120]` | 64 | 224 | 1.3 waves | **~75%** |
| mlp-down `[5120,17408]` | 64 | 80 | 0.47 waves | **~66–77%** |
| out-proj `[5120,6144]` | 64 | 80 | 0.47 waves | **~52–58%** |

mlp-up is the existence proof that ~86% is reachable in this family: enough CTAs **and**
16-byte NVFP4 scale-line fills (`BlockN≥128`).

### Residual underfill

80 CTAs on 170 SMs leave memory controllers idle. The Layer-0 classifier refuses `tile_shape` /
`occupancy` on DRAM because they do not change \(B_W\). They change **achieved** GB/s toward
the same floor. Residual at ~52–58% is not the floor.

### Scale-tile vs MMA tile

Persistent NVFP4 scales are 128 rows × k64 = 512-byte tiles (32 × 16-byte lines, 4 quartiles).
`BlockN=64` does `cp.async<8>` at stride 16: 50% of each 32-byte DRAM sector. `BlockN=128`
does `cp.async<16>` on the full line (mlp-up). A 64-wide MMA tile fights a 128-row scale layout.

### Fused quantize inside the N-partitioned MMA CTA

The extra bytes on this path are a 5-block BF16→NVFP4 quantize (~4 µs) plus a tiny workspace.
Fusing that quantize into each of 224 N-CTAs replicates the work 224×. NCU on attn-in T=4:

1. Naive fuse: 129 µs, DRAM 18%, **77% issue stalls at CTA barriers** (only 128/256 threads in
   the quantize loop).
2. Warp-interleaved tasks: 121 µs, barrier gone, **6.6 active threads/warp**, L1TEX scoreboard.
3. `cp.async` BF16 tiles into smem: 137–191 µs, **1 block/SM** (shared-memory limit), DRAM 12%.

Oracle passed. The idea is the wrong place to fuse given an N-partitioned grid. Deleted.

### MMA SASS

The legal atom is already inline PTX (`ldmatrix` + `mma.sync.kind::mxf4nvf4` `m16n8k64`).
Tensor-pipe busy is a few percent. Do not rewrite the MMA inner loop for decode.

## Target mapping (still one pass over \(W\))

Treat decode Linear as **output-partitioned streaming GEMV with T-wide A**, implemented inside
the existing W4A4 MMA family:

1. `BlockN=128` so scale fills are 16-byte lines.
2. `grid.z = SplitK` with `K_tiles % SplitK == 0` so CTA count \(\gtrsim 170\) without re-reading
   \(W\). Each CTA owns `(N-tile, K-slice)`. Partials are FP32, then a tiny reduce. Partial
   traffic is \(O(T N \times\mathrm{SplitK})\) bytes (~0.1 µs), not a second weight pass.
3. Quantize A **once** (existing 5-block kernel), not per N-CTA and not per K-tile.
4. Do not use this split on mlp-up decode (`N/128` already 272 CTAs). Leave TMA prefill alone.

Compile-time split (cap 4) so \(N/128 \times \mathrm{SplitK} \ge 170\) when \(K/256\) allows it:

| Geometry | N-tiles@128 | K-tiles@256 | SplitK | CTAs |
|---|---:|---:|---:|---:|
| attn-in 14336×5120 | 112 | 20 | 2 | 224 |
| gdn-in 16384×5120 | 128 | 20 | 2 | 256 |
| mlp-up 34816×5120 | 272 | 20 | 1 | 272 |
| out-proj 5120×6144 | 40 | 24 | 4 | 160 |
| mlp-down 5120×17408 | 40 | 68 | 4 | 160 |

`SplitK>1` cannot apply the fused epilogue (linear_add residual, SwiGLU pair) per slice:
reduce first, then epilogue. Workspace partials are capped at decode T (≤32), not prefill T.

## Measured (public `ninfer_linear_bench`, cold cache, RTX 5090)

Implemented inside production W4A4: `BlockN=128`, `grid.z=SplitK`, one weight pass, quantize
once, FP32 partials, then reduce. Oracle (`ninfer_linear_nvfp4_a4_test`) passed. Deleted from
production after these public-Op numbers; do not re-land without a new mechanism.

Baseline (HEAD decode W4A4): attn-in T=4 ≈ 32.8 µs (~75% READ), mlp-up T=4 ≈ 70 µs (~86%),
mlp-down T=4 ≈ 39 µs, out-proj T=4 A16 ≈ 18.4 µs / W4A4 T≥5 ≈ 18–20 µs.

| Variant | attn T=4 | gdn T=4 | mlp-up T=4 | out T=5 | down T=4 |
|---|---:|---:|---:|---:|---:|
| `BlockN=128` SplitK=2, 2-stage | 34.8 | 38.9 | ~71 | 20.5 | 41.0 |
| `BlockN=128` SplitK=1 (no K-split) | 34.8 | **36.9** | ~70 | **28.7** | **57.3** |
| `BlockN=128` 3-stage, split only residual | **32.8** | 36.9 | ~71 | 20.5 | 38.9–41 |
| Same + cooperative fused reduce | 32.8 | 36.9 | ~70 | 20.5 | 38.9 |
| Residual `BlockN=64` SplitK=2, 160 CTAs | 32.8 | 36.9 | ~70 | 20.5 | 40.9 |
| Decode `Cache::EvictFirst` at T≤4 | 34.8 | 38.9 | ~72 | 18.4 A16 | 41.0 |

Facts this mapping did not survive:

- Residual `BlockN=128` without split-K is a large loss (40 CTAs). Confirmed.
- Attn-in SplitK=2 vs SplitK=1 is the same 34.8 µs at 2-stage: restoring CTA count did not
  restore the 3-stage M32N64 wall time. The 2 µs is pipeline/tile, not occupancy.
- 3-stage `BlockN=128` on attn-in returns **parity** with HEAD, not a win (same 32.768 µs bucket).
- Split-K reduce (separate kernel ~2.8 µs NCU, or fused `this_grid().sync`) does not move
  out-proj off the 20.48 µs bucket. Occupancy gain is cancelled by shorter K-streams + partials.
- `EvictFirst` on this MMA path **regresses** T≤4 vs `cp.async.cg`.

The production decode grid is restored.

## Residual attribution (NCU, one public `linear()`, RTX 5090)

Production W4A4 is **two kernels**: `nvfp4_w4a4_quantize_kernel` then `nvfp4_w4a4_mma_kernel`.
No reduce, no pad memset. MMA DRAM read / \(B_W\) = **1.00** (one weight pass; extra
read is tens of KB). NCU GPU times are inflated vs `ninfer_linear_bench` median; shares
below scale the lighter capture (`dram__bytes_op_read`) to the public median.

| Point | Bench | Quantize (scaled) | MMA (scaled) | \(B_W\) | MMA read | MMA vs \(t_\mathrm{mem}\) | MMA grid | waves/SM |
|---|---:|---:|---:|---:|---:|---:|---|---:|
| out T=6 `[5120,6144]` | 18.43 | **~4.0 µs** (22%) | **~14.5 µs** | 16.89 MB | 17.76 MB | **73%** | 80×256, smem 1 CTA/SM | 0.47 |
| down T=4 `[5120,17408]` | 38.91 | **~3.9 µs** (10%) | **~35.1 µs** | 50.14 MB | 50.20 MB | **85%** | 80×256, smem 1 CTA/SM | 0.47 |
| attn T=4 `[14336,5120]` (control) | 32.77 | **~3.8 µs** (12%) | **~29.0 µs** | 41.29 MB | 41.31 MB | **85%** | 224×256, smem 2 CTA/SM | 0.66 |

Quantize is 5–17 CTAs, ~1% DRAM peak, ~4 µs of launch-bound work on tens of KB. It is
**not** the DRAM bound. Combined READ% vs \(t_\mathrm{mem}\) (58% out, 77% down, 75% attn)
puts that ~4 µs in the same median as MMA.

Same 80-CTA residual grid: **down MMA is already 85%** of the copy roof (long K, 68
K-tiles). **out MMA is 73%** (short K, 24 K-tiles). Grid underfill is not why down is
slow, and it is not the whole out story. Split-K shortened the stream that out is
already starving; that is why occupancy mapping lost on out.

Next public-Op win, in order, **measured and deleted** (cold cache, `ninfer_linear_bench`,
warmup 3 / repeat 20). Production remains 256-thread quantize + residual `M32N64S4`.

1. **Standalone quantize, more CTAs, still one kernel.** Decode T≤32 launched 32-wide
   CTAs (40–136 blocks) and 64-wide (20–68 blocks). Attn T=4 stayed **32.768 µs**, out T=6
   **18.432**, down T=4 **38.912**. NCU GPU time is ~4 µs **independent of task count**
   (attn 1280 vs down 4352); spreading the same one-task-per-thread work does not cut it.
   Do not fuse into the N-grid (224× replicate). PDL weight-prefetch cannot hide it: the
   independent W prologue is tens of KB, not 4 µs.

2. **Short-K residual pipe (out 73% → 85%), same 80 CTAs.** Pipeline depth vs HEAD S4:

   | Residual pipe | out T=6 | down T=4 |
   |---|---:|---:|
   | HEAD 4-stage, min 1 CTA/SM | **18.432** | **38.912** |
   | 3-stage, 4-stage smem pad | 20.480 | 40.960 |
   | 2-stage, 4-stage smem pad | 24.576 | 53.248 |
   | 2-stage, natural smem (no pad) | 24.576 | 53.248 |

   Fewer stages lose. Padding smem to keep 1 CTA/SM did not change the 2-stage numbers;
   the tax is outstanding `cp.async` depth on a 24/68-tile K-stream, not packing.
   Do not SplitK or decode TMA.

The ~4 µs quantize tax is a second kernel launch, not an occupancy hole. Cutting it
requires not launching that kernel (produce NVFP4 in a previous epilogue, or a route
that is not an N-replicated MMA prologue). Residual MMA already wants the 4-stage pipe.

## TMA N-slice (FlashInfer #2992 / b12x) — measured, deleted

Their decode win is vs a **128×128 TMA** baseline (N=1024 MoE, or N=512–1792 dense). Ours is
already M32 N64 cp.async. Porting the recipe onto residual did not beat HEAD:

| Variant | attn T=4 | out T=6 | down T=4 |
|---|---:|---:|---:|
| HEAD decode W4A4 (cp.async M32N64) | 32.8 | 18.4 | 38.9 |
| TMA M32N64, 80 CTAs, plus `memset` pad | 36.9 | 36.9 | 65.5 |
| TMA M32N32, 160 CTAs, no memset | 32.8 | **34.8** | **69.6** |
| cp.async M32N64 + 16-byte 128-row scale box | **34.8** | 18.4 | 38.9 |

`cudaMemsetAsync` of A to 32 rows taxed **every** W4A4 launch (~4 µs), including attn which
never used decode TMA. TMA M32N32 (4 MMA warps + 1 TMA warp, 160 CTAs) still lost ~2× on
residual: warp-specialized mbarrier at T=4–12 is worse load efficiency than 3-stage
`cp.async` at the same payload. Full 16-byte scale-line fills on BlockN=64 add extra scale
bytes and moved attn to the 34.8 µs bucket. Do not reland.

Published b12x µs that look faster (e.g. M=1 N=512 K=7168: 13.7 vs CUTLASS 24.2) are not
Qwen residual/down shapes. Closest published: M=1 N=5120 K=16384 b12x **50 µs**; our
mlp-down T=4 is **38.9 µs**.

## What not to do

- Fuse BF16 quantize into the MMA CTA on this N-partitioned grid.
- `BlockN=128` on residual **without** split-K (40 CTAs, worse underfill). Measured 28.7 / 57 µs.
- `BlockN=128` on attn-in **without** a 3-stage pipe (2-stage is 34.8 µs). 3-stage is only parity.
- Reland this SplitK mapping; it was measured and deleted.
- Reland decode TMA M32N64 / M32N32 or 16-byte 128-row scale overread on this MMA path.
- Split-K that walks all of K in every CTA (weight replay).
- TMEM / `tcgen05` / 2-SM / cluster-multicast TMA.
- Reland 32- or 64-thread standalone quantize; more CTAs of the same K16 tasks is
  public-Op parity.
- Residual 2-stage or 3-stage MMA vs HEAD 4-stage (out 24.6 / 20.5 vs 18.4; down
  53.2 / 41.0 vs 38.9), with or without 4-stage smem padding.
- Engine / serve A/B before the public Op bench wins.

## Gate

`split_k` is Layer-0 `MEASURE` on DRAM: allowed only if it does not re-read weights.
This mapping partitions K; NCU on the previous MMA kernel showed one weight pass and
residual DRAM well below 1674.5 GB/s from grid underfill. Qualify with
`ninfer_linear_bench` (then fused add / SwiGLU / attn_input_proj) at serving T, cold cache,
oracle first. **This mapping was implemented, oracled, benched, and deleted.** It did not
beat HEAD decode W4A4. See Measured above.
