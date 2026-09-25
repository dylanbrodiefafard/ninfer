# DFlash A8 verification follow-ups (plan, 2026-09-24)

Temporary plan: candidate speedups identified after the A8/A8 DFlash verification default and
its GDN-record/post-mixer fusions (`c1da30a8`). Each entry records the attribution that motivated
it, the intended change, its numerical consequence, and the measured outcome. Retained results
move to [performance.md](../performance.md); measured non-wins move to
[performance_enhancements.md](performance_enhancements.md); this file is then removed.

Target: `qwen3.8-27b/nvfp4` DFlash2 artifact
`/models/qwen3.8-nvfp4-flash2-nvfp4-bf16codebook-from-bf16/qwen3_8_27b_nvfp4_dflash_nvfp4.ninfer`,
RTX 5090 / `sm_120a` / CUDA 13.1, NVFP4 KV, CUDA Graphs, p-less T2, optimized proposal head.

## Motivating attribution

Source: `profiles/nsys/a8-fusion-before-c{1,4}` (Engine tg128, fixed k4; pre-fusion build, so
the post-mixer quantize/RMSNorm shares are slightly overstated). Bandwidth floors use 1674 GB/s
sustained read over NVFP4 codes plus E4M3 scales.

| Kernel | C1 avg µs (floor) | C4 avg µs (floor) | C1 / C4 GPU time |
|---|---:|---:|---:|
| A8 gate/up `[34816,5120]` | 65.9 (56.6) | 79.0 (56.6) | 28.8% / 24.9% |
| A8 MLP down `[5120,17408]` | 40.1 (28.3) | 46.5 (28.3) | 17.5% / 14.9% |
| A8 GDN input `[16384,5120]` | 31.8 (26.6) | 40.4 (26.6) | 10.4% / 9.4% |
| A8 residual out `[5120,6144]` | 14.8 (10.0) | 17.2 (10.0) | 6.2% / 5.4% |
| BF16 GDN gating a/b (96×5120) | 6.2 (0.6) | 16.2 (0.6) | 2.0% / 3.8% |
| Drafter A16 `nvfp4_small_t` (6144, 1280, 4096, 25600) | — | — | 2.6% / 4.6% |
| Standalone FP8 activation quantize (5120/6144/17408) | 1.3–2.2 | 1.4–2.3 | 2.6% / 2.3% |
| GDN recurrent fold (per round) | 165 | 713 | 1.1% / 3.6% |

GPU idle between kernels: 3.6% (C1) / 3.2% (C4); about 1085 kernels per C1 round.

## Candidates

| # | Candidate | Numerics | Status |
|---|---|---|---|
| 1 | GDN gating projection: parallelize tokens across the grid, same per-output K order | bit-exact | kept |
| 2 | N=5120 A8 projections: deeper pipeline and/or deterministic split-K | stages: exact; split-K: FP32 association changes | stages kept (M32 K512×3 added); in-CTA K split and M32 N16 grid lost; cross-CTA split-K not admitted |
| 3 | Swap-AB A8 MMA (weights on M=16, tokens on N=8) for T≤24 | per-output K16 order unchanged; exact if FMA order kept | kept for wide shapes |
| 4 | Drafter projections A16 → A8 | acceptance only; target distribution unchanged | not kept; acceptance unresolved |
| 5 | Remove standalone quantize launches (pre-mixer RMSNorm+A8 fusion, consumer-side quantize) and PDL weight prefetch on A8 kernels | exact if the same row scale/codes are produced | PDL lost; quantize fusion open (bounded ≤~1% per site) |
| 6 | Verification W2/W3 → A8 across C (keeps precision independent of C) | new PPL qualification required | done: A8 at every verify width (performance.md) |
| 7 | Defer GDN fold into the next round's overlay (one fewer state pass) | FP32 state transition must stay identical | closed: ≤~1.2% at C4 after 8, large state-transaction change |
| 8 | Chain GDN verify: register-resident record kernel instead of the scratch T=1 overlay | bit-exact (FP32 store/load identity, same transition) | kept |

Details:

1. `bf16_gdn_gating_proj_gemv_kernel` launches 96 CTAs (one logical row each) and loops all T
   tokens with a block reduction per token. Splitting tokens over `grid.y` keeps each output's
   thread-to-K mapping and reduction tree, so outputs stay bit-identical and batch-invariant.
2. After row partitioning, MLP-down/residual-out A8 launch 320 CTAs × 64 threads with two K256
   stages: about 8 KB of weight in flight per SM. More stages raise bytes in flight without
   changing arithmetic. Split-K (fixed-order cluster/DSMEM or two-pass reduction) changes FP32
   association and would invalidate exact NLL equality; it needs the paired PPL rerun.
3. With tokens on MMA M, T5 wastes 11/16 rows and T20 uses two m16 tiles. With weights on M and
   tokens on n8, T≤8 halves MMAs and scale FMAs, T17–24 cuts them 25%. At C4 gate/up costs
   79 µs versus 66 µs at T5 with identical bytes.
4. Drafter precision changes only acceptance. Its MLP already runs A4; its attention, conv
   kernel projections and context projection are A16-only `small_t`.
5. PDL infrastructure exists (`src/core/pdl.cuh`) but no NVFP4 kernel uses it; weights do not
   depend on the producer, so the first stages can be issued before `griddepcontrol.wait`.
6. `text_policy()` keeps request-local W2/W3 at A16 at every C. Adaptive and short fixed widths
   at C>1 therefore pay A16 cost that scales with aggregate T.
7. Fold reads and writes ~151 MB FP32 state per request per round and is already at its
   bandwidth floor; only removing a pass helps.

## Measurement protocol

- Engine: `tools.bench.run_serve_concurrency`, `long_decode_aime26_15`, 8192 completion tokens,
  `--mode dflash4` (fixed k4) and `--mode dflash5 --adaptive-draft`, C1 (KV 32768) and C4
  (KV 65536), steady full-batch aggregate decode tok/s. Bit-exact changes are checked by matching
  fixed-k4 response hashes.
- Op: public-Op cold-cache medians where a single kernel changes.
- Attribution: scoped nsys Engine traces before and after the retained set.

## Results

Engine cells are single 8192-token waves (same fixture/flags as above); changes are cumulative.
Fixed-k4 response hashes are identical across every row, so rows 1–3 are bit-exact to the
baseline. Adaptive C4 hashes differ between runs because the timed policy changes widths.
Evidence: `profiles/bench/dflash-a8-followups/`, traces `profiles/nsys/a8-followups-*`.

| Build | Fixed k4 C1 | Fixed k4 C4 | Adaptive C1 | Adaptive C4 |
|---|---:|---:|---:|---:|
| Baseline (`c1da30a8`) | 154.13 | 431.31 | 156.76 | 433.09 |
| + gating token split, A8 stage schedule | 158.59 | 438.97 | 162.77 | 440.29 |
| + SwapAB for wide A8 projections | 160.05 | 450.92 | 163.72 | 451.60 |
| `fccf6613` (A8 at every width) | 159.29 | 449.07 | 156.99 | 452.30 |
| + chain GDN record kernel (8) | 161.03 | 461.17 | 158.43 | 463.03 |
| + M32 N=5120 K512×3 (2) | 161.22 | 471.25 | 158.73 | 473.84 |

- **1, gating (kept):** C4 trace 58.5→26.2 ms of kernel time (16.5→6.7 µs per T20 call), C1
  6.3→5.0 µs. Bit-exact.
- **2, A8 pipeline (kept, stages only):** A8 now stages weights without the unused W4A4 activation
  buffers. That smaller footprint alone regressed M16 gate/up (public Linear T5 73.7→86 µs, more
  CTAs per SM); three stages restore it. Selected: N=5120 M16 K512×3, N=5120 M32 K256×3, other M16
  K256×3, other M32 K256×2. Public Linear (incl. quantize): MLP-down T5 57.3→41.0 µs, T20
  63.1→53.2; residual-out T5 26.6→20.5, T20 30.7→24.6. Engine C1 trace: MLP-down 40.3→34.6 µs,
  residual-out 14.9→13.9; the M32 Engine kernels did not move. K128×4 and BN32 wide variants lost.
  Split-K was not needed for this step and remains untried.
- **3, SwapAB (kept for wide shapes):** public-Op gain within timer resolution (T20 gate/up
  84→82 µs), but in Engine C4 gate/up 79.9→75.6 µs and GDN-in 39.0→37.0 µs. N=5120 with SwapAB
  (BN32/two warps) lost (MLP-down T20 53→78 µs) and is not used. Bit-exact.
- **4, drafter A8 (not kept):** Feature `[5120,25600]`, qkv `[6144,5120]` and attention output
  `[5120,4096]` via W4A8 at T≥4, batch-aggregated (MLP stays A4; conv kernel projections were not
  changed). C4 drafter projection time 1007→575 µs per round; C1 unchanged (qkv T5 19.7 µs both
  ways). Engine: fixed k4 C1 160.05→153.99, C4 450.92→462.76; adaptive C1 163.72→157.60, C4
  451.60→455.14. Per-request acceptance across the four seeds: fixed 31.0→31.2% (mean),
  adaptive 32.0→31.1%; seed 0 fell in both modes (32.0→30.2, 27.5→25.9). C1/C4 trajectories stay
  identical within each build, so an A8-only-when-batched drafter would break that invariance.
  Re-trying needs an acceptance measurement over many seeds or teacher-forced drafts.
- **5, PDL (not kept):** A8 kernels launched as programmatic dependents, prefetching all weight
  stages before `griddepcontrol.wait`. Traces grew (C1 span 1106→1128 ms, C4 1475→1497 ms) and the
  Engine regressed: fixed k4 C1 160.05→156.91, C4 450.92→442.97, hashes unchanged.
  **Quantize fusion (not attempted):** in the SwapAB C1 trace the standalone quantizers cost about
  308 µs per 13.9 ms round (17408: 141, 6144: 88, 5120: 79). The 5120 site needs pre-quantized
  inputs on the GDN-record and attention-input Ops; 17408/6144 need a cross-CTA row max (producer
  atomic max + consumer-side quantization from BF16 doubles L2 activation reads per CTA). Each is
  bounded at roughly 0.5–1% of round time.
- **2b, in-CTA K split (not kept):** extra warp groups take interleaved global K16 groups
  (T-independent association) and reduce in rank order. Public Linear for N=5120 was worse at
  every width: MLP-down ×2 45–61 µs versus 41–55; ×4 61–63 µs at T≤16. M32 N=5120 is therefore
  not limited by warps per SM. Cross-CTA split-K (cluster DSMEM) is untested.
- **6, W2/W3 A8 (done):** selected and qualified; see performance.md "DFlash A8 at every
  verification width". Original Op data: Public Linear, A16 → A8 including quantization, with the
  A8 T≥4 gate lifted temporarily:

  | Shape | T2 | T3 | T8 | T12 |
  |---|---|---|---|---|
  | gate/up `[34816,5120]` | 77.8→71.7 | 79.9→73.3 | 106.5→73.7 | 129.0→73.7 |
  | GDN-in `[16384,5120]` | 53.2→38.9 | 47.1→38.9 | 77.8→38.9 | 94.2→38.9 |
  | attn-in `[14336,5120]` | 32.8→34.8 | 34.8→34.8 | 53.2→34.8 | 63.5→34.8 |
  | MLP-down `[5120,17408]` | 36.9→40.4 | 41.0→40.6 | 59.4→41.0 | 69.6→43.0 |
  | residual-out `[5120,6144]` | 18.4→20.5 | 18.4→20.5 | 28.7→20.5 | 32.4→20.5 |

  A8 wins broadly at the C2–C4 aggregates of W2/W3 and roughly ties at C1.

- **2c, M32 N=5120 (kept K512×3):** NCU counters are admin-only on this host
  (`RmProfilingAdminOnly=1`); the launch shape is 160 CTAs × 4 warps and the public Op ran at 52%
  of sustained read. Public Linear A8 (incl. quantize), MLP-down / residual-out at T17–24:
  baseline 52.8 / 24.0 µs; N16 grid K512 83.3 / 34.1; N16 grid K256 55.3 / 25.9; N32 K512 44.6 /
  19.7 (kept). T≤16 unchanged. Grid underfill is therefore not the limiter; deeper per-stage K
  is. Cross-CTA split-K is `split_k` under a DRAM bound, which `tools.kdev` admits only if it cuts
  model bytes; it does not, and cluster>1 is outside the kdev legality envelope, so it is not
  pursued.
- **8, chain GDN record (kept):** the C4 swap trace showed `recurrent_overlay_kernel` at 22.5 µs
  per layer (1.08 ms per round, 5.8%) against 8.2 µs at C1. For chain verify the overlay stored
  and reloaded each row's head state in scratch after every column. The register-resident record
  kernel runs the same width-one transition; `ninfer_gated_delta_net_replay_record_test` already
  requires its `out` to equal the snapshot kernel bit for bit. The overlay is now tree-only.
  Engine hashes unchanged.
- **7, fold deferral (closed):** with chain verify reading state once, deferral would remove at
  most one state read per round (about a third of the 714 µs C4 fold, ~1.2%), add sequential
  steps to every record kernel, and move the committed-state frontier across rounds.

## Remaining follow-ups

- Drafter A8 with a proper acceptance study (candidate 4).
- Standalone-quantize removal (candidate 5 remainder).
- BF16 target attention layers (`bf16_small_t_inner` 14336×5120, ~92–108 µs, 6 per round) and the
  W8 verify LM head (~800 µs per round) are at their bandwidth floors; only format changes help.
