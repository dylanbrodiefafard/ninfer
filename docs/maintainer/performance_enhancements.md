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

See [dflash2-tree-speed.md](dflash2-tree-speed.md) for the kept W=12 BFS path, split
top-k select, and HBM 4-warp record. Do not retry:

- local-edge N=8 pack (collapses to a chain)
- full beam W=16 (third GQA tile; slower than W=12)
- 4-warp shared-memory GDN record at W=12 (occupancy 1, wash vs 1-warp)

