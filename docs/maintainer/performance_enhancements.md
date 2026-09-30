# Performance enhancements that did not move tok/s

Negative results for decode-speed work that was measured and should not be
re-tried without a new attribution. Published serving numbers stay in
[performance.md](../performance.md).

## Decode-round host/GPU seam (`qwen3.8-27b/nvfp4`, C=1)

Hypothesis: C=1 decode was leaving enough host time between CUDA Graph launches
that hiding or removing that seam would raise decode tok/s.

It does not. The GPU round is ~10–15 ms at 5k MTP3 and grows with context. Host
fold + submit is a few tenths of a millisecond. Overlap and device tail-launch
cannot show up in tok/s.

### Setup

| Item | Value |
|---|---|
| Identity | `qwen3.8-27b/nvfp4` (Osfralla MTP artifact) |
| GPU | RTX 5090, `sm_120a` |
| KV | `--kv-dtype int8` (this seam A/B). NVFP4 KV exists; new speed work uses `--kv-dtype nvfp4` |
| Bench | `ninfer_bench`, graphs on, C=1, chunk 4096, `--max-ctx 151000` |
| Workload | prompt 5k/20k/50k/100k/150k, generate 128, warmup 1, r=2 |
| MTP0 | no `--spec` |
| MTP3 | `--mtp-draft-tokens 3 --lm-head-draft` |
| Corpus | `profiles/bench/bench_corpus_160k.ids` (tiled greedy; not AIME quality) |

Reports: `profiles/bench/round-seam-baseline/`, `round-seam-opt1/`, `round-seam-opt2/`.

Nsys MTP3 5k/32 on the baseline path: `decode.mtp.wait` 13.65 ms (the GPU
round), `decode.fold` 0.20 ms, `decode.mtp.submit` 0.16 ms.

### Cluster Launch Control — do not implement for Engine scheduling

CLC (`clusterlaunchcontrol.try_cancel` / `query_cancel`) is intra-kernel tile
work-stealing. The host still launches one problem-sized grid. It does not
launch other kernels, replace CUDA Graphs, or replace Engine round scheduling.
Decode T=1..4 has too few tiles versus 170 SMs.

### Option 1 — hide the seam, host still launches

Device writeback of next ingress, stop-id truncate on device, MTP Fold inside
the graph, skip captured H2D on frozen membership, overlap next `cudaGraphLaunch`
with resolve/publish.

Decode tok/s vs baseline: **1.000–1.001×** at every cell. Prefill unchanged.
MTP3 accept counts identical to baseline.

### Option 2 — B=1 device graph tail-launch

Second B=1 definition: closed round without per-round D2H, device token ring,
scheduler kernel, `cudaGraphInstantiateFlagDeviceLaunch`. Host launches once and
waits the chain. Serving stayed on host-launched one-round graphs.

Decode tok/s vs baseline: **0.983–0.987×** (about 1.5% slower). Prefill
unchanged. MTP3 accept counts identical to baseline.

| prompt | MTP0 base | MTP0 opt1 | MTP0 opt2 | MTP3 base | MTP3 opt1 | MTP3 opt2 |
|---|---:|---:|---:|---:|---:|---:|
| 5k | 85.65 | 85.62 | 84.27 | 287.20 | 287.37 | 282.37 |
| 20k | 83.20 | 83.20 | 81.90 | 262.52 | 262.64 | 258.28 |
| 50k | 79.05 | 79.05 | 77.87 | 264.40 | 264.42 | 260.41 |
| 100k | 73.06 | 73.06 | 72.06 | 230.22 | 230.20 | 226.75 |
| 150k | 68.05 | 68.02 | 67.18 | 214.62 | 214.63 | 211.48 |

Values are `ninfer_bench` `decode_output_tok_s_mean`.

Option 2’s extra append/tail kernels and device-launch executable are a small
constant tax; they do not hide a seam that is already negligible against the
GPU round.

### Do not retry

Further host/GPU seam work, CLC-as-scheduler, or device tail-launch of the
decode graph will not raise C=1 decode tok/s on this identity until the GPU
round itself is shorter, or a profiler shows a host gap that is a material
fraction of that round. Prefill was never the target and did not move.

## DFlash2 packed-tree shapes and 4-warp smem GDN

See [dflash2-tree-speed.md](dflash2-tree-speed.md) for the historical W=12 BFS path, split
top-k select, and HBM 4-warp record. Product DFlash2 is chain-only, with adaptive `{3,4,5}`. Do not retry:

- local-edge N=8 pack (collapses to a chain)
- full beam W=16 (third GQA tile; slower than W=12)
- 4-warp shared-memory GDN record at W=12 (occupancy 1, wash vs 1-warp)

## Adaptive draft: width-factored acceptance (`qwen3.8-27b/nvfp4`, DFlash2)

Hypothesis: the adaptive picker pools per-hop acceptance `r_i` across draft widths, while DFlash2
drafts the whole block in parallel, so a width-5 block may accept less per hop than a width-4
block. On one AIME prompt at C=2 (p-less T1.5) it did (hops 1-3 cumulative 0.52/0.34/0.23 vs
0.56/0.41/0.30) and adaptive held k=5 at 4.7% below fixed k=4.

Tried: engine-global conditional acceptance `g(d,i)` per drafted width, a per-request level
`a_i = (Σ s + 2)/(Σ n·g + 2)`, scoring `r(d,i) = a_i g(d,i)`, and a 1-in-32 probe of the other
captured widths. In a host simulation with the AIME per-width acceptance the pooled picker ran
k=5 100% of the time and the width model k=4 97%.

It does not help on real mixed traffic. RTX 5090, DFlash2 artifact, batched chain drafting,
p-less T1.5, draft T0.4, NVFP4 KV, thinking, 3072 tokens/request, one server per point with a
warmup wave, then two measured waves rotating `aime26_15`, `code_python`, `story_en_mystery`,
`translation_zh_en`, `code_cuda`, `structured_csv` across slots. Mean aggregate tok/s:

| C | pooled (shipped) | width model | fixed k=4 | fixed k=5 |
|---:|---:|---:|---:|---:|
| 1 | 169.1 | 166.8 | 160.4 | 169.0 |
| 2 | 300.6 | 294.3 | 299.7 | 300.7 |
| 4 | 532.1 | 528.9 | 536.0 | 496.8 |
| 6 | 713.2 | 711.5 | 727.3 | 583.3 |

Across these prompts k=5 blocks accept more than k=4 (C=2 tokens/round 2.61/3.14 vs 2.47/3.04),
so the AIME drop does not generalise; the best fixed k is 5 at C≤2 and 4 at C≥4, and the pooled
picker already tracks it within 0-2%. The width model is 0.2-2% lower at every C (probe rounds at
a non-best k plus estimate noise), within wave-to-wave noise of 3-6%. Reverted. Reports:
`profiles/bench/adaptive-width-model-20260930/`.
