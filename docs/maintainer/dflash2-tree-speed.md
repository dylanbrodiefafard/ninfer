# DFlash2 verify speed

Investigation and A/B for Qwen3.8-27B NVFP4 DFlash2 on RTX 5090 (`sm_120a`). Published serving
numbers stay in [performance.md](../performance.md) until a new campaign is recorded there.
Negative host/GPU-seam results stay in [performance_enhancements.md](performance_enhancements.md).

Artifact: `/ssdpool2nvme/local_llm/ninfer-sandbox/out/qwen3_8_27b_nvfp4_dflash_w8.ninfer`.
New speed and Engine A/Bs use `--kv-dtype nvfp4`. The tok/s tables below were measured with INT8
group-64 unless a row says otherwise. Product companion is NVFP4 draft matrices + BF16 selector
codebooks; NVFP4 codebooks save ~174 MiB and are not a decode tok/s win. CUDA Graphs,
`--lm-head-draft`, AIME seed `7632647173703958409`. C=1 serve tok/s is `request_done` `decode=`
(not harness `steady_interval`). C>1 headline tok/s is aggregate `decode_tokens / wave_makespan`.

## Outcome

The packed-tree verify (beam-2 BFS, W=12) is a permanent product feature (`tree_verify=true`
for 27B) and the default route for the native draft window k=6,7; the chain (`W=k+1`) is used
for k≤5 and for k>7 (the spark two-block propose). The chain remains the speed recommendation:
k=4/W=5 is fastest on AIME, and at k=7 the tree (W=12) is slightly slower in tok/s than the
chain (W=8) (164.95 vs 166.82) with a slightly higher accept rate (32.88% vs 32.23%), so the
wider tree window is useful on workloads where it pays off. After fused batched NVFP4 GDN
conv-record, C>1 isolation holds and AIME C=3 k=4 reaches **324 aggregate tok/s** (NVFP4 KV).
The sections below retain the A/B evidence from the chain/tree speed investigation.

Tree GDN record uses 4-warp parent tiles in HBM when the ReplaySSM workspace is sized for it;
tests without that workspace keep the 1-warp shared-memory tile. Path/tree select scans the
shortlist with 32-way column splits, then scores each (parent, candidate) pair with the serial
rank-256 FMA (32 pairs in parallel for the tree walk). The 4-warp tree record keeps a 16 KiB
2-slot smem cache of the last two written parent tiles.

## Serve C=1 (stochastic)

| Config | AIME tok/s | accept | tok/round | Story tok/s | accept | tok/round |
|---|---:|---:|---:|---:|---:|---:|
| Published chain W=8 | 126.7 | 27.5% | 2.92 | 115.7 | 21.6% | 2.51 |
| Full beam W=16 | 117.5 | 34.5% | 3.42 | 98.9 | 25.7% | 2.80 |
| N=8 pack by local edge score | 110.7 | 26.8% | 2.87 | 101.1 | 22.9% | 2.60 |
| N=8 BFS prefix | 113.6 | 28.0% | 2.96 | 105.0 | 24.6% | 2.72 |
| N=12 BFS prefix, 1-warp GDN | 119.3 | 32.9% | 3.30 | 105.2 | 26.9% | 2.87 |
| N=12 BFS + 4-warp smem GDN | 119.0 | 32.9% | 3.30 | 104.9 | 26.9% | 2.87 |
| **N=12 BFS + split top-k + HBM 4-warp GDN** | 145.4 | 32.9% | 3.30 | 129.6 | 26.9% | 2.87 |
| **N=12 + walk smem + 2-slot GDN parent cache** | **149.7** | **32.9%** | **3.30** | **133.9** | **26.9%** | **2.87** |

Same AIME/story fixtures, 4096 / 1024 decode tokens, max-context 16384. Current reports:
`profiles/bench/qwen38_dflash2_walk_gdn_aime/`, `profiles/bench/qwen38_dflash2_walk_gdn_story/`.
Prior 145.4/129.6 reports: `profiles/bench/qwen38_dflash2_w12_opt_aime/`,
`profiles/bench/qwen38_dflash2_w12_opt_story/`.

Accept rate is unchanged versus the W=12 1-warp baseline: the kept kernels change only how
top-16, Markov dots, and GDN parent state are computed, not the SpecInfer walk. W=12 now beats published
chain tok/s because the previous tree tax was almost entirely the 1-block path/tree-select
kernel, not the extra verify width. Walk + 2-slot GDN cache add **+4.3 tok/s** AIME and
**+4.3 tok/s** story at the same accept (1.03× the 145.4 path, 1.18× published chain, 0.81× MTP3).

## Spark two-block (`--draft-tokens 11`)

Spark/vLLM DFlash2 is two sequential MASK blocks, then one target verify of the concatenated
chain. DGX Spark numbers (K=16, 124–135 tok/s) are on a slow box; 5090 should beat them. NInfer
caps verify at W=12 (two GQA tiles), so the practical split is **7+4 = 11 drafts**.

The historical `--draft-tokens 7` packed-tree experiment ran one block. `--draft-tokens 11` runs:
forward with 7 MASK columns, path-select a chain of 7, unmask those ids, second DFlash forward
with 4 new MASK columns, path-select the suffix (parent = last token of block 1), then chain
verify of W=12. SWA is already bidirectional. Default `--lm-head-draft` stays on.

AIME C=1 stochastic, seed `7632647173703958409`, 4096 decode, INT8 KV, presence penalty 0,
`request_done` decode tok/s. Post-origin bars. **k=11 “407/432 tok/s” was a GDN/KV commit bug**
(`path_length = committed` with `fold_path` all zeros on chain verify). Greedy k=11 matched T=1
after the fold was gated on `dflash_uses_tree_verify`.

| Config | tok/s | accept | tok/round | ~ms/round |
|---|---:|---:|---:|---:|
| MTP3 | 180.5 | 50.1% | 2.50 | 13.9 |
| DFlash tree k=7 `--lm-head-draft` | 157.5 | 35.8% | 3.50 | 22.2 |
| DFlash two-block k=11 **busted fold** | 432.8 | 69.6% | 8.64 | 20.0 |
| **DFlash two-block k=11 fixed** | **163.7** | **20.7%** | **3.28** | **20.0** |

Round time for k=11 did not change. The 432 tok/s was fake accept on corrupted GDN/KV. Fixed k=11
is ~k=7 tok/s with worse accept; MTP3 still wins AIME. k=7 stays the product default. k=11 is not
an AIME hammer once commit is sequential.

Story C=1 (`scenario_story_en_mystery`, 1024 decode, thinking off), measured on the busted fold:
MTP3 **163.8 tok/s** / 41.9% / 2.26; two-block k=11 **100.8 tok/s** / 7.8% / 1.85. Do not use k=11
as a prose default.

Reports: `profiles/bench/post_origin_dflash_c1c2_aime/` (k=7 and busted k=11),
`profiles/bench/dflash11_chain_fold_fix_aime/` (fixed k=11).

## Tree-shape A/B (before the kernel work)

Local-edge N=8 collapsed to a chain (one child per depth). BFS prefix keeps both children at
early depths. W=12 beats W=16 tok/s because GQA is two T=6 decode tiles (6+6) versus three
(6+6+4). `kSmallTChunkTokens = 6`. T=8 chain uses the prefill/small-t prompt route (one
launch/layer); T=12 tree uses two decode tiles.

Ignore one N=8 harness `steady_interval` of 614 tok/s (stats-window artifact).

## nsys: chain W=8 vs tree W=12 (pre-opt)

Same greedy CLI, 128 tok, 30 DFlash rounds, max-ctx 4096. Wall: tree decode 0.707 s / 179.6 tok/s;
chain 0.648 s / 195.9 tok/s; both 4.23 tok/round on that short prompt.

Per-round GPU delta tree − chain ≈ 2.0 ms. Family totals:

| Family | Tree | Chain | Δ / round (÷30) |
|---|---:|---:|---:|
| `dflash2_tree_select` / `dflash2_path_select` | 214 ms, 6.69 ms/call | 206 ms, 6.43 ms/call | +0.25 ms |
| GDN `recurrent_record` 1-warp parent vs 4-warp seq | 41.9 ms, 27.3 µs | 13.7 ms, 8.9 µs | +0.94 ms |
| GQA (tree: 2× T=6 decode tiles; chain: T=8 prefill kernel) | 14.9 ms | 8.9 ms | +0.20 ms |
| `gqa_kv_compact_path` | 5.9 ms | 0 | +0.20 ms |
| parent-indexed GDN conv | 7.3 vs 0.3 ms | | +0.23 ms |

Both select kernels were one block of 256. Vocab top-16 and the tid-0 merge of 256×16 lists
ran on a single SM. That kernel was ~28% of GPU time for **both** chain and tree, so it did
not explain the tree-vs-chain gap, but it capped absolute tok/s.

## What was implemented and measured

### 1. Split-column top-k for path/tree select (kept)

Top-16 of each draft column is independent of the Markov walk. Launch 32 split blocks per
column, merge 32 lists of 16, then walk scores on one block per batch. Scoring stays serial
FP32 so greedy tokens match the existing oracle tests.

Short greedy CLI (same 29-token primes prompt as the pre-opt nsys): decode **229.7 tok/s**
(0.553 s, 4.23 tok/round) vs 179.6 tok/s before.

Post-opt nsys (`profiles/nsys/dflash2-w12-opt.nsys-rep`), 32 calls:

| Kernel | Total | / call |
|---|---:|---:|
| `dflash2_column_topk_split_kernel` | 6.6 ms | 0.207 ms |
| `dflash2_column_topk_merge_kernel` | 1.8 ms | 0.057 ms |
| `dflash2_tree_select_kernel` (walk) | 26.8 ms | 0.838 ms |
| **select family** | **35.2 ms** | **1.10 ms** |

Select went from 6.69 ms/round to 1.10 ms/round (−5.6 ms). That is the tok/s move.

### 2. 4-warp HBM parent tiles for tree GDN record (kept)

4-warp **smem** tiles are 96 KiB at W=12 (occupancy 1) and measured equal to 1-warp smem
(119.0 vs 119.3 AIME). Do not retry smem 4-warp.

HBM tiles reuse one graph-stable work-arena buffer of `Hv × B × W × 128 × 128 × 4` bytes
(~37 MiB at C=1 W=12). Tree record is now the 4-warp sequential tile geometry.

Post-opt: 26.1 ms / 1536 calls / **17.0 µs** vs 27.3 µs 1-warp (−0.50 ms/round). Still slower
than sequential record (8.9 µs) because each column reloads parent state. Remaining GDN tax
is ~0.4 ms/round.

### 3. Skip KV/feature compact when `fold_path` is identity `0..m-1` (kept)

Host check before `gqa_kv_compact_path` / `gather_bf16_path`. Helps reject-all and left-spine
accepts. Compact was 5.9 ms pre-opt and 5.8 ms post-opt on the short greedy prompt (compact
still ran on most rounds). Not a tok/s lever.

### 4. Not implemented (low nsys signal or already rejected)

- Identity compact skip as a device kernel: the host check is enough.
- Growing T/k/shortlist, local-edge pack, Plan A (tree scores + chain verify), W=16.
- Host/GPU seam or CLC: see performance_enhancements.md.

## A/B after the 145 tok/s baseline

Kill rule for pack: stochastic AIME accept must rise ≥3 pp (32.9% → ≥36%). GPU opts are
kept if they do not drop accept and they raise `request_done` decode tok/s.

### Pack: d1=3 + hedge + depth-7 spine (reverted)

W=12, beam 2, same codebook. Path log-prob (log-softmax over 16 Markov scores), three
depth-1 children of the anchor, one child per live parent when quota allows, one reserved
column per remaining depth (shape 1+3+2+2+1+1+1+1). Report:
`profiles/bench/qwen38_dflash2_pack_aime/`.

| Config | AIME tok/s | accept | tok/round |
|---|---:|---:|---:|
| N=12 BFS prefix (baseline) | 145.4 | 32.9% | 3.30 |
| d1=3 / hedge / d7 spine | 141.6 | 31.1% | 3.18 |

Accept fell 1.8 pp. Extra depth-1 breadth traded away the two-wide beam at depths 4–5,
and the depth-7 spine did not pay for that on stochastic AIME. Reverted to BFS prefix.

### Walk: smem codebook + 32 serial (parent, cand) dots (kept)

`dflash2_tree_select` stages hidden (256), two parent pred rows, and 16 successor rows in
shared memory. 32 threads each run the existing rank-256 serial FMA for one pair. Path
select does the same for 16 chain candidates. Association is unchanged; greedy oracle tests
still pass.

Stacked with the 2-slot GDN cache below (not isolated). Combined serve C=1:

| Config | AIME tok/s | accept | tok/round | Story tok/s | accept | tok/round |
|---|---:|---:|---:|---:|---:|---:|
| 145.4 baseline | 145.4 | 32.9% | 3.30 | 129.6 | 26.9% | 2.87 |
| walk + 2-slot GDN cache | 149.7 | 32.9% | 3.30 | 133.9 | 26.9% | 2.87 |

Reports: `profiles/bench/qwen38_dflash2_walk_gdn_aime/`,
`profiles/bench/qwen38_dflash2_walk_gdn_story/`. The +4.3 tok/s matches the walk's predicted
round cut (~0.84 ms serial walk). GDN's share is not separated.

### GDN: 2-slot parent-tile smem cache (kept)

4-warp HBM record still writes all 12 columns to the work-arena tile buffer. After each
column it also keeps that tile in a 16 KiB 2-slot FIFO (`token & 1`). The next column
loads parent state from smem on a hit. This is not the rejected 96 KiB all-column smem
tile. Tree record vs sequential still matches in `ninfer_gated_delta_net_replay_record_test`.

### Native T=12 GQA (rejected)

INT8 small-t is templated on `TokenTile` with `RowTiles = (TokenTile * GroupSize + 15) / 16`
and `static_assert(RowTiles <= 3)`. 27B `GroupSize=6` makes T=6 → 3 row tiles (the maximum)
and T=12 → 5. A native T=12 tile needs a new producer/consumer warp split, not a chunk-size
bump. W=12 stays two T=6 decode tiles. Expected leftover remains ~0.20 ms/round.

## Leftover-kernel A/B (keep path = BFS-12 + 2-slot GDN)

Kill rule: raise `request_done` decode tok/s on both AIME and story without dropping accept.
AIME-only bumps are not a default (same rule as Markov T=0.2).

### GDN 4-slot parent cache (reverted)

Hypothesis: 2-slot even/odd (`token & 1`) misses when both children attach to one parent.
Four associative slots of the last four written tiles (32 KiB vs 16 KiB) should cover parent
distances 1–4.

AIME C=1: **149.8 tok/s, 32.9%, 3.30 tok/round** — wash vs 149.6. Occupancy from 32 KiB smem
ate the hit-rate win. Report: `profiles/bench/qwen38_dflash2_gdn4slot_aime/`. Keep 2-slot.

### DFS emit of the same BFS-12 nodes (reverted)

Same 12 BFS-prefix nodes, columns in DFS preorder (greedy child first) so a left-spine
accept is `fold_path = 0..m-1` (identity compact skip) and GDN parents on that spine are
`i-1`.

| Fixture | Keep (BFS) | DFS emit | Result |
|---|---|---|---|
| AIME 4096 | 149.6 tok/s, 32.9%, 3.30 | **154.6 tok/s, 34.7%, 3.43** | AIME-only |
| Story 1024 | 133.9 tok/s, 26.9%, 2.87 | **131.5 tok/s, 25.7%, 2.80** | Lose |

Reports: `profiles/bench/qwen38_dflash2_dfs_aime/`, `qwen38_dflash2_dfs_story/`. Accept
moved with column order (GQA T=6 tile grouping / child-scan order), so this is not a
constant-accept kernel win. Reverted to BFS emit.

### Keep-path nsys (128-tok greedy CLI)

`profiles/nsys/dflash2-w12-keep.nsys-rep`, 32 DFlash rounds, decode 238.9 tok/s / 4.23
tok/round. Per round:

| Family | Total | / round | Note |
|---|---:|---:|---|
| NVFP4 GEMM (whole capture) | 344 ms | | Draft+verify matmuls; not tree tax |
| GDN `recurrent_record` 4-warp | 27.4 ms | 0.86 ms | ~2× sequential (~0.43 ms leftover) |
| Select (split+merge+walk) | 10.9 ms | 0.34 ms | Walk 76 µs; split 207 µs |
| GQA INT8 T=6 decode (2 tiles) | 11.4 ms | 0.36 ms | Extra tile ~0.18 ms |
| Parent-indexed GDN conv | 7.0 ms | 0.22 ms | Register parent history already |
| `gqa_kv_compact_path` + gather | 6.5 ms | 0.20 ms | Host identity skip already kept |

Native T=12 would remove one T=6 tile (~0.18 ms) at the cost of a new `RowTiles=5` kernel.
Parent conv and compact are the same size. None of these is a tok/s lever on a ~22 ms
AIME round. Do not implement T=12, a faster compact kernel, or a sequential-only conv
bypass without a new nsys that shows they grew.

## Remaining tree tax (after the kept kernels)

On the 128-tok greedy nsys, select is no longer the round. Left versus chain:

- Tree GDN record still ~2× sequential (~0.4 ms/round); 4-slot smem and DFS emit both
  failed the keep rule.
- Parent-indexed conv (~0.22 ms/round).
- Extra GQA decode tile + compact (~0.4 ms/round combined).

Those do not dominate a ~20 ms AIME round at 150 tok/s. Matching 184.7 at ~22 ms/round needs
~4.1 tok/round (~45% of 7). Greedy beam-2 BFS-12 is **32.9%**.

## Candidate ceiling (parent head / sequential step)

`NINFER_DFLASH_CANDIDATE_STATS=1` with `--no-cuda-graph`, 256-token stochastic CLI (same
sampler as serve). 59 rounds, 60 first-reject hops (greedy-length 4.31 tok/round on this
short prompt, not AIME).

| Bucket (first reject of the round) | Share |
|---|---:|
| In packed tree (wrong parent) | 20.0% |
| In that depth's parallel **top-16** | 88.3% |
| In top-64 / top-256 | 93.3% |
| In the 131k draft head | 98.3% |
| Absent from draft head | 1.7% |

The parallel column already lists the missed token. Markov beam-2 does not attach it to the
walk node. That is drafter/target mismatch among 16 known candidates, not a missing
shortlist.

| Idea | Ceiling from this table | Already measured |
|---|---|---|
| Larger K / Markov over top-64 | 88% → 93% of rejects (5 pp) | K=32: +0.6 pp accept, revert |
| Parent-conditioned Markov head | Already the pack scorer | Reshapes of the same 16 all failed |
| One extra sequential DFlash step | Can only *must-help* the 12% not in top-16 (~0.12 tok/round if all hit). Any larger win needs sequential logits to re-rank the 88% that parallel already had. | Not implemented; 5-layer T=1 is ~0.5–1.5 ms on a ~22 ms AIME round (break-even ≈ +0.3 tok/round) |
| MTP token union into the tree | MTP sees target hidden (51.9% chain accept) but adds a full MTP propose on top of 5 DFlash layers | Not implemented; round would get longer |

`unmask_refine=1` (greedy first token unmasked, same 7 MASK columns) was A/B'd earlier:
greedy 128-tok **4.10 tok/round / 211 tok/s** vs keep **4.23 / 239**. That is not Spark
two-block. True two-block with new MASK columns is `--draft-tokens 11` above.

## Accept A/B (reverted to greedy Markov beam-2 BFS-12)

Hop mix on the kept pack, SpecInfer walk, AIME C=1 stochastic. A hop **hit** is the sampled
target token being a child of the current node. **miss_in_tree**: that token is somewhere in
the packed 12 columns but not as a child of the current node. **miss_absent**: it is not in
the packed tree. 512-token diagnostic on the kept K=16 pack: 74.5% hit / 6.8% wrong-branch /
18.7% absent (n=514 hops). Full 4096-token AIME is harder (~70% hit on the same pack).

Same AIME command as the 149.7 keep (seed `7632647173703958409`, presence penalty 0). Greedy
rebaseline on this binary: **149.6 tok/s, 32.9%, 3.30 tok/round**.

| Experiment | AIME tok/s | accept | tok/round | Hop hit / in-tree / absent | Result |
|---|---:|---:|---:|---|---|
| d1=3 + hedge, no spine (`1+3+2+2+2+2`) | 148.9 | 32.6% | 3.28 | | Revert |
| k=3 frontier BFS-12 (`1+3+3+3+2`) | 137.4 | 28.2% | 2.97 | | Revert |
| Shortlist K=32, same beam-2 pack | 147.8 | 33.5% | 3.34 | 70.1 / 8.6 / 21.3 | Revert |
| Rank children by unary, Markov parent | 136.5 | 27.8% | 2.95 | 66.0 / 11.6 / 22.4 | Revert |
| Sample 2 children from Markov softmax T=0.6 | 138.0 | 28.4% | 2.99 | 66.8 / 8.7 / 24.5 | Revert |
| Same, T=0.3 | 146.0 | 31.6% | 3.21 | 70.0 / 8.7 / 21.3 | Revert |
| Same, T=0.2 | 151.3 | 33.6% | 3.35 | 72.0 / 7.8 / 20.3 | AIME-only; story 131.5 / 25.7% vs 133.9 / 26.9% |
| Forced binary after depth 1 | 140.1 | 29.2% | 3.04 | 67.1 / 8.2 / 24.7 | Revert |

K=32 barely moves accept: Markov still attaches the same two children. Most `miss_absent`
is “in the shortlist, not one of the two packed children” or “not in that block column’s
top-16,” not “in top-16 but truncated by W=12.” W=16 already showed the BFS-truncation gap
is only +1.6 pp. Unary ranking and T=0.6 sampling both increase absent misses. Forcing both
spines drops the hot parent’s second child and also increases absent misses. T=0.2 is a
small AIME bump that regresses story, so it is not a product default.

The block shortlist is column-t of the parallel DFlash draft, not a parent-conditioned
draft distribution. Closing the remaining ~18–21% absent hops needs a different drafter
(or parent-conditioned candidates), not another greedy reshape of this beam.

Official DFlash2 `draft_sample_method: "probabilistic"` is wired for **chain**
propose (`W=k+1`): `dflash2_path_select` writes the 16-way selector q, and
`speculative_accept_greedy_drafts` runs Leviathan `min(1, p/q)` with residual
`max(0, p-q)`. Null q stays one-hot.
k=11 two-block already uses this q.

CUDA-graph capture used a dummy `SamplingConfig{}` (temperature 0) and passed those
host scalars as path-select kernel parameters, so replay always drafted greedy /
one-hot q while accept still saw the request temperature. Path-select now reads
the device ingress `SamplingConfig` the graph already copies each round.

Isolated CLI C=1, `--kv-dtype nvfp4`, graphs, `--lm-head-draft`, seed
`7632647173703958409`, presence penalty 0, artifact
`/models/qwen3_8_27b_nvfp4_dflash_nvfp4.ninfer`. AIME `long_decode_aime26_15`.
Temperature 0.6, top-p 0.95, top-k 20, min-p 0. CLI metric is `decode speed`
(`(generated-1) / decode_seconds`).

Native-block chain: `--spec dflash --draft-tokens 7 --lm-head-draft` (W=8). RTX 5090 speed path:
`--spec dflash --draft-tokens 4 --lm-head-draft` (W=5, one SmallT tile).

512-token prefix after the graph-q fix, sequential tree accept (historical; not
re-run 2026-08-29):

| Config | tokens | tok/s | accept | tok/round |
|---|---:|---:|---:|---:|
| tree k=7 W=12 | 512 | 139.39 | 33.33% | 3.32 |
| chain+q k=7 W=8 | 512 | 201.41 | 37.42% | 3.60 |

Full 4096 after W4A4 packed verify + sage V-fill hybrid (2026-08-29). Prior accept-pipeline
cutover row in parentheses:

| Config | tokens | tok/s | accept | tok/round | rounds |
|---|---:|---:|---:|---:|---:|
| chain+q k=4 W=5 | 4096 | **194.48** | 50.26% | 3.01 | 1360 |
| **tree k=7 W=12** | 4096 | **164.95** (was 166.51) | **32.88%** (was 29.87%) | **3.30** (was 3.09) | 1240 (was 1325) |
| chain+q k=7 W=8 | 4096 | 166.82 (was 143.34) | 32.23% (was 28.24%) | 3.25 (was 2.98) | 1258 (was 1376) |

k=4 W=5 is chain (`W=k+1`). It is the tok/s winner on this fixture because each round is a W=5
verify (one SmallT tile); tok/round is *lower* than k=7 (3.01 vs 3.30). Accept % is not comparable
across k (drafted = k × rounds). Among k=7 rows, tree wins accept by 0.65 pp but loses tok/s. A
matched story run also favored chain k=7 (**147.45 vs 144.54 tok/s**); k=4 reached **160.10
tok/s**. All three AIME rows hit the 4096 output limit. Logs:
`profiles/bench/qwen38_dflash2_nvfp4_aime_20260829/` and local
`dflash-story-{tree,chain,k4}.stderr` reports.

## Batched fused GDN (C>1 isolation, 2026-08-29)

NVFP4 C>1 verify no longer flattens GDN conv-record to `T=W×B` compose (W4A4 GEMM + BF16
conv). That path flipped greedy column 0 versus C=1 fused SmallT+FP32. The speed path is one
fused SmallT launch at `T=W` per row (`grid.x=B`). Op guard:
`run_nvfp4_batched_matches_serial_fused` (B=3 W=8, B=3 W=5, mixed valid). Engine isolation
`ninfer_qwen3_8_27b_dflash_real_test` k=7 C=3 Graph DFlash still matches saved C=1 DFlash.

Isolated CLI C=1, NVFP4 KV, same AIME command as the W4A4 table
(`/models/qwen3_8_27b_nvfp4_dflash_nvfp4.ninfer`, seed `7632647173703958409`, presence
penalty 0, `--lm-head-draft`, 4096 decode):

| Config | tokens | tok/s | accept | tok/round | rounds |
|---|---:|---:|---:|---:|---:|
| chain k=4 W=5 | 4096 | **162.18** | 46.57% | 2.86 | 1431 |
| chain k=7 W=8 | 4096 | 142.52 | 31.61% | 3.21 | 1275 |

C=1 on this tree is below the W4A4 snapshot above (194.48 / 166.82). k=4 accept also moved
(50.26% → 46.57%) at the same seed. k=4 still wins C=1 tok/s.

Serve C=1/2/3, same fixture, NVFP4 KV, `--kv-capacity auto`, presence penalty 0. Aggregate
tok/s is `decode_tokens / wave_makespan`. Per-request is mean `(completion-1)/decode_seconds`.
1-second full-batch steady intervals did not fire (stats window spanned the whole wave);
headline numbers use the request_done aggregate. Logs:
`profiles/bench/qwen38_dflash2_fused_batch_aime_20260829/`.

| Mode | C | Aggregate tok/s | Per-request tok/s | Accept | tok/round | Makespan (s) | vs C=1 |
|---|---:|---:|---:|---:|---:|---:|---:|
| chain k=4 W=5 | 1 | 162.1 | 162.6 | 46.6% | 2.86 | 25.26 | 1.00× |
| chain k=4 W=5 | 2 | 262.0 | 133.0 | 45.5% | 2.82 | 31.26 | 1.62× |
| chain k=4 W=5 | 3 | **324.4** | 112.1 | 47.3% | 2.89 | 37.87 | 2.00× |
| chain k=7 W=8 | 1 | 143.0 | 143.4 | 31.6% | 3.21 | 28.63 | 1.00× |
| chain k=7 W=8 | 2 | 230.8 | 121.3 | 29.7% | 3.08 | 35.49 | 1.61× |
| chain k=7 W=8 | 3 | 302.0 | 103.1 | 29.6% | 3.07 | 40.68 | 2.11× |

C=3 k=4 is **324 aggregate tok/s**. Isolation holds; per-request rate is the shared-SM cost of
the compact decode batch, not a quality loss.

Reports: `profiles/bench/qwen38_dflash2_nvfp4_aime_20260829/`,
`qwen38_dflash2_fused_batch_aime_20260829/`,
`qwen38_dflash2_d1h_aime/`, `qwen38_dflash2_k3_aime/`,
`qwen38_dflash2_k32_aime/`, `qwen38_dflash2_unary_aime/`,
`qwen38_dflash2_markov_sample_aime/`, `qwen38_dflash2_markov_t03_aime/`,
`qwen38_dflash2_markov_t02_clean_aime/`, `qwen38_dflash2_markov_t02_story/`,
`qwen38_dflash2_binary_aime/`, `qwen38_dflash2_greedy_rebase_aime/`.

## Do not retry without new attribution

- Local-edge N=8 pack (collapses to a chain).
- Full beam W=16 (third GQA tile; slower tok/s than W=12).
- d1=3 / hedge / d7-spine pack at W=12 (accept 31.1% vs 32.9% BFS prefix).
- d1=3 + per-parent hedge, no spine (32.6%).
- k=3 frontier BFS-12 (28.2%; shallower than k=2 at N=12).
- Shortlist K=32 (33.5% / 147.8 tok/s).
- Unary child ranking (27.8%).
- Markov softmax child sampling at T=0.6 / 0.3 (story also loses at T=0.2).
- Flatten NVFP4 GDN conv-record to `T=W×B` compose for C>1 (W4A4 GEMM + BF16 conv). It
  flips greedy column 0 versus C=1 fused SmallT+FP32. Keep one fused SmallT launch at `T=W`
  per row (`grid.x=B`).
- Forced binary after depth 1 (29.2%).
- Native T=12 INT8 GQA tile under the current `RowTiles<=3` kernel.
- GDN 4-slot parent-tile smem cache (32 KiB; wash vs 2-slot).
- DFS column order of the same BFS-12 nodes (AIME-only; story loses).
- Faster compact kernel / sequential-only parent conv (≤0.22 ms/round).
- 4-warp **shared-memory** GDN record at W=12 (occupancy 1, wash vs 1-warp).
- Host/GPU seam, CLC-as-scheduler, device tail-launch of the decode graph.
- Parallel rank-256 reduction inside a Markov score (greedy token flips). Parallel
  (parent, cand) pairs that each keep the serial 256-FMA are the walk above.
