# Single-GPU serving performance

Tested Git revisions:

- Concurrent MTP3 decode saturation for the three measured Qwen3.6 artifact profiles:
  `26da9df7c1b3d3c04ea7bbd730271aa01d00742a`;
- Refreshed Qwen3.6-35B-A3B and Qwen3.6-27B NVFP4 MTP3:
  `f4f21cc36bd1a83cbc046f668719d591dc9c1e2e`;
- Qwen3.6-35B-A3B stored MTP3 response audit:
  `b1a220f028aa750f75bceb3522ac00bbaab7e42d`;
- Qwen3.6-35B-A3B DFlash block=8 (`k=7`):
  `0dc94097e8ec5c5bcf59b9e13e9d1852f504eb61`;
- Qwen3.6-27B NVFP4 accuracy and MTP0:
  `b3d4d0f50b868711c62432bbd68e746217a2f49a`;
- Qwen3.6-27B groupwise-int MTP3: `5ea3242a206cdb0c4c1beaeb9d8a3048e6248423`;
- Qwen3.6-35B-A3B MTP0 and Qwen3.6-27B groupwise-int MTP0:
  `0795169393cab0f2c16246d4bac20dee735dc2a4`;
- Qwen3.8-27B NVFP4 EvalScope accuracy (INT8 and NVFP4 KV):
  `c0f4ec2cfe234b3e3988f79f0399d077de8178b6`.

## V4 SSD page-spill qualification

The 2026-09-02 page-only qualification used an RTX 5090, CUDA 13.1, the local ZFS mirror, NVFP4
KV geometry, incompressible deterministic data, and the exact 33,981-token workload: 531 logical
pages, 1,148,387,328 payload bytes, and 1,151,143,936 capacity-billed bytes per repetition. Each
three-repetition sample synchronized the pool before reading device counters and again after the
cache commit. Physical bytes are the sum of both NVMe leaf deltas divided by the explicitly
configured mirror replication factor of two; samples are accepted only when that normalized value
is within 3% of the cache's durable-extent accounting.

| Page batch | Host-visible commit GB/s | Post-pool-sync payload GB/s |
|---:|---:|---:|
| 1 | 3.716 | 2.357 |
| 4 | 4.119 | 2.507 |
| 8 | 3.471 | 2.244 |

Batch four is the production setting. Its clean aggregate was 6,920,171,520 leaf-device bytes,
3,460,085,760 mirror-normalized bytes, and 3,453,431,808 cache-accounted bytes. The dataset had
`sync=disabled`, so the host-visible commit number is not a crash-durability result. The post-sync
number includes an explicit pool sync and is the relevant physical write-through result for this
dataset, but no mirror-normalized v3 post-sync measurement survives for a same-scope comparison.
The historical v3 host-visible result was approximately 3.6–3.8 GB/s.

The same campaign measured pinned H2D at 28.94 GB/s, D2H at 28.67 GB/s, and packed-page
H2D-plus-scatter at 28.11 GB/s (97.1% of pinned H2D), so the page copy/scatter route is not the
remaining restore bottleneck. Direct-reader scaling at 1/4/8/16/32 readers was
3.85/11.99/15.63/17.64/17.57 GiB/s; production retains 16 readers. Warm 531-page restore
diagnostics varied from about 11.3 to 19.5 GB/s. They validate the route and concurrency but are
not cold-device claims because the available host could not evict ZFS ARC and the dataset uses
`primarycache=all`.

A follow-up V4-compatible qualification used the benchmark's exact 27B linear-attention geometry:
153,944,064 raw state bytes and a one-page prompt. One serial direct read plus serial CRC measured
2.617 GB/s. Four 4 MiB read/CRC chunks in flight measured 10.87–13.89 GB/s across the retained
runs (4.2–5.3×); more per-object workers contended, so production caps each large state object at
four while sharing the startup-fixed reader budget across independent objects. The exact 531-page
startup audit fell from a 259.1 ms mean to 65.6 ms (3.95×) by reading each page record once and
validating independent CRCs across that same bounded pool. This startup measurement was
warm/ARC-resident; it isolates validation overhead and is not a cold-device startup claim. Startup
still rejects corrupt live payloads before advertising an entry.

A second follow-up isolated CRC32C and state-worker dispatch. Alternating 1 GiB CRC sweeps on the
Ryzen 9 7950X3D measured the serial SSE4.2 chain at 14.0–14.3 GB/s. Three interleaved hardware
chains measured 22.5–23.1 GB/s at 1 MiB, 29.9–34.5 GB/s at the production 4 MiB chunk, and
28.8–32.2 GB/s at 256 MiB. The combine overhead lost below 512 KiB, so production retains the
serial path for smaller records and switches at 512 KiB. In the clean exact-state run this changed
restore from 13.06 to 14.87 GB/s (+13.8%), and the warm/ARC-resident 531-page startup audit mean
fell from 65.6 to 59.1 ms (-9.9%). Reusing the existing restore threads through a persistent
state-task queue instead regressed paired state restores by 3.4% and 7.1% (13.54→13.08 and
14.73→13.69 GB/s); that candidate was rejected and the transient four-worker state path remains.

Compaction of 312,246,272 retained bytes measured 2.57–2.65 GB/s against a 2.9–3.27 GB/s direct
write control. Increasing its transfer buffer from 1 MiB to 8 MiB regressed from 2.65 to 2.48 GB/s,
so the production buffer remains 1 MiB and no extra pipeline is retained. The final physical
batch-four rerun measured 3.944 GB/s host-visible commit and 2.448 GB/s post-pool-sync payload,
with every mirror-normalized sample within the 3% accounting bound. The accepted state and startup
changes did not alter page-record transfer layout; disk format v5 later changed only the startup
fingerprint from physical tensor geometry to the canonical logical-page schema. Contiguous page
reads, compaction, H2D/scatter, and durable writes remain at or close to the relevant measured
hardware/filesystem ceilings.

The serving measurements characterize the two measured Qwen3.6 model IDs independently on one
NVIDIA GeForce RTX 5090. They cover long-context prefill and baseline decode with speculative
decoding disabled, plus long-reasoning and cross-scenario decode with MTP and DFlash. The 27B
results report its `groupwise-int` and `nvfp4` weight profiles separately. The concurrent
decode-saturation campaign measures the same three Qwen3.6 artifact profiles at C=1, 2, 4, and 8.
A separate C=1 Qwen3.8-27B NVFP4 campaign below compares MTP0/3/5 with DFlash2 k=7 on the same
frozen AIME command (INT8 KV). A later NVFP4-KV DFlash2 campaign measures isolated CLI C=1 and
serve C=1/2/3 after fused batched GDN conv-record. Qwen3.8-27B NVFP4 accuracy uses
[Ostfralla/Qwen3.8-27B-NVFP4-NInfer](https://huggingface.co/Ostfralla/Qwen3.8-27B-NVFP4-NInfer)
with INT8 and NVFP4 KV.

The single-request corpus requests were submitted serially to a persistent `ninfer-serve` process
over the loopback OpenAI-compatible HTTP endpoint. Each reported corpus fixture used five fixed
seeds. Values are arithmetic mean ± sample standard deviation, and server warm-up completes before
the measured requests. The concurrent campaign has its own sustained-wave method below.

## Qwen3.8-27B NVFP4 host-overhead campaign

The 2026-09-01 non-kernel campaign fixed the supported target to
`qwen3.8-27b/nvfp4`, one RTX 5090, DFlash k=4, NVFP4 KV, CUDA Graphs, prefix reuse disabled,
`--prefill-chunk 4096`, and startup concurrency C=1, 2, 3, or 4. The artifact was
`out/qwen3_8_27b_nvfp4_dflash_nvfp4_codebook.ninfer`. The benchmark-reported toolchain was CUDA
13.3 compile/runtime with driver API 13.0. Greedy fixed waves used exact work counters and compared
output hashes as multisets because concurrent HTTP admission may permute requests across lanes.

The configuration gate selected DFlash k=4 at every C. Its external decode rates were 209.97,
328.37, 391.60, and 467.91 tokens/s for C=1-4. The following ratios are candidate/prior external
throughput; a value of 1.0 is neutral. Each row used the immediately preceding retained binary.

| Host change | C=1 | C=2 | C=3 | C=4 | Decision |
|---|---:|---:|---:|---:|---|
| Lock-free hot runtime counters and transition-only full snapshots | 0.9982 | 0.9980 | 1.0002 | 0.9982 | Retained: removes per-round locked lane/KV scans; no material regression |
| Periodic reporter no longer calls `memory_summary()` | 1.0000 | 0.9996 | 0.9996 | 0.9993 | Retained: removes reporter contention with the executor; throughput-neutral |
| Terminal-only publication for non-stream requests | 0.9971 | 1.0003 | 0.9988 | 1.0000 | Retained: removes per-round event lock/allocation/wakeup; exact outputs |
| Stable-decode dirty signal | 1.0027 | 0.9997 | 1.0012 | 1.0013 | Retained: skips unchanged queue/deadline/admission scans |
| Cached Program graph selection | 0.9973 | 0.9978 | 0.9998 | 1.0026 | Reverted: regressed C=1/2 and added invalidation state |
| Direct `(K,B,frontier)` graph-routing table | 0.9999 | 0.9989 | 0.9995 | 0.9999 | Reverted: O(1) lookup produced no end-to-end win |

Two fresh cumulative A/B passes against the preserved pre-campaign binary, in opposite binary
orders, produced retained/baseline throughput ratios of `0.9987/1.0014/0.9994/0.9995` and
`1.0030/1.0014/1.0013/1.0001` at C=1-4. Their per-C geometric means are
`1.0009/1.0014/1.0003/0.9998`: C=2's +0.14% result repeated, but no material uniform
end-to-end speedup is established. Exact work and output-hash multisets matched. The cumulative
host changes are retained for removing unnecessary executor-side contention and publication work,
not claimed as a general decode-speed improvement.

The terminal-only SSE control A/B preserved output hashes, event counts, and ordering. Mean
first-output changes for C=1-4 were +0.62, +0.49, +0.71, and -0.03 ms; worst event-gap changes were
below 0.9 ms. A 7,705-token prefill follow-up removed the full transition snapshot after the first
nonterminal 4,096-token chunk. A fresh C=4 paired repeat produced candidate/prior prefill-speed
ratios of 0.9991, 1.0003, 0.9997, and 1.0008 across the four serialized request positions and a
0.03% lower wave makespan, with exact work and output multisets.

Allocation and representation gates rejected final-string preallocation, stop-token lookup, and
stop-string matcher changes: representative final output growth copied only about 8 KiB per
2,048-token request, the artifact has exactly two default stop token IDs, and the fixed workload
has no stop strings. Consumer cancellation polling was retained because `CancellationView` has no
owning notification hook; replacing it would add cross-layer lifetime machinery without improving
the GPU launch path.

The graph-profile gate separately tested tighter DFlash context envelopes while preserving every
target-declared implementation-transition boundary. Relative to the existing profiles, maximum
spans of 2,048, 1,024, and 512 tokens produced C=1-4 external-throughput ratios of respectively
`1.0001/1.0002/0.9995/1.0004`, `1.0020/1.0010/0.9995/1.0006`, and
`1.0017/0.9974/0.9960/0.9965`. Exact work and output hashes matched. The 2,048 and 1,024 results
were neutral at measurement scale and 512 regressed C=2-4, so the existing transition-derived
profiles remain selected; startup graph update compatibility is not treated as evidence that a
profile is performance-optimal.

Primary reports are under `profiles/bench/host-overhead-*`. The baseline gate is
`host-overhead-mode-gate-baseline-20260831`; retained decode A/Bs end in
`memory-candidate-greedy-20260831`, `runtime-stats-causal-greedy-20260831`,
`terminal-only-candidate-greedy-20260831`, and `stable-decode-candidate-greedy-20260901`. The final
prefill repeat is `host-overhead-prefill-stats-{prior,candidate}-rerun-c4-thinking-greedy-20260901`.
The graph-routing baseline and profile sweep are under `profiles/bench/graph-routing-baseline-*`
and `profiles/bench/graph-profiles-span{2048,1024,512}-candidate-*`.
The fresh cumulative reports are
`host-overhead-cumulative-{baseline,candidate}{,-rerun}-greedy-20260901`.

## Single-request serving performance method

| Setting | Value |
|---|---|
| GPU | NVIDIA GeForce RTX 5090, 32 GiB |
| CUDA compile/runtime | 13.1 / 13.1 |
| CUDA driver API | 13.3 for NVFP4 and refreshed 35B MTP3; 13.1 for the remaining single-request campaigns |
| Request mode | One active request, `stream=false` |
| Maximum context | 262,144 tokens; 131,072 for refreshed NVFP4 MTP3 |
| Prefill chunk | 1,024 tokens |
| KV cache | INT8 group-64 |
| CUDA Graph | Enabled |
| Prefix reuse | Disabled |
| Sampling | Temperature 0.6, top-p 0.95, top-k 20, presence penalty 1.0 |
| Greedy profile | Exact argmax (`--sampling greedy` in the corpus runner) |
| MTP0 | no `--spec` |
| MTP3 | `--spec mtp --draft-tokens 3 --lm-head-draft` |
| DFlash block=8 | `--spec dflash --draft-tokens 7 --lm-head-draft` |

The MTP0 profile uses four Long NIAH prompts with approximately 8K, 64K, 128K, and 256K tokens.
Thinking is disabled and the output budget is 128 tokens. These runs measure prefill throughput,
server-internal time to first token, and baseline decode throughput at each context length. Content
scenarios are not repeated with MTP disabled because they do not change the baseline decode path.

The speculative-decode corpus contains three long-reasoning fixtures with thinking enabled and a
65,536-token output limit, followed by twelve fixtures covering code, story, translation, and
structured output. The cross-scenario fixtures disable thinking and use a 4,096-token output limit.
The tables report actual completion lengths rather than assuming that every request reaches its
limit.

Metrics are computed from the server's unrounded phase timings and speculative-decode counters:

```text
prefill_tok_s = prompt_tokens / prefill_seconds
server_ttft_ms = 1000 * (prepare_seconds + vision_seconds + prefill_seconds)
decode_tok_s = (completion_tokens - 1) / decode_seconds
spec_acceptance = accepted_tokens / drafted_tokens
spec_tokens_per_round = 1 + accepted_tokens / speculative_rounds
```

Decode throughput is a transport/execution measurement, not a correctness score. The response text,
finish reason, and fixture-level structural requirements are audited separately below. A request
that exhausts its output budget or enters a repetition loop remains useful as a sustained-decode
stress sample, but is not presented as a successfully completed task.

## Concurrent MTP3 decode saturation

The concurrent campaign uses the `long_decode_aime26_15` fixture with thinking enabled. The
rendered prompt is 293 tokens, and every request has an 8,192-token output budget. For each
concurrency C, the runner starts a fresh `ninfer-serve` process with `max_concurrency=C`, releases
C non-stream requests together using distinct fixed seeds, and waits for every HTTP response.
Startup and server warmup occur before the measured wave.

All points use an RTX 5090, CUDA 13.1 compile/runtime, CUDA driver API 13.3, stochastic sampling
(temperature 0.6, top-p 0.95, top-k 20, presence penalty 1.0), INT8 group-64 KV, a 1,024-token
prefill chunk, CUDA Graphs, prefix reuse disabled, and
`--spec mtp --draft-tokens 3 --lm-head-draft`. Each request has a 16,384-token context ceiling.
`--kv-capacity auto` resolved to exactly `C * 16,384` tokens at every point.

Saturated throughput uses only complete one-second server intervals satisfying all of the following:

- computed prefill tokens are zero;
- `running=C`, `prefilling=0`, and `decode_ready=C`;
- at least one decode round completed;
- every decode round had exactly C rows.

Ramp-up, prefill, and drain intervals are excluded. The reported aggregate rate is:

```text
steady_decode_tok_s = sum(committed_decode_tokens) / sum(interval_seconds)
```

Wave makespan starts when the client threads are released and ends after the last complete HTTP
response. MTP acceptance is aggregated over the complete wave. Each row below is one sustained
wave rather than a repeated-sample mean.

| Model profile | C | Steady (s) | Avg batch | Aggregate decode tok/s | MTP acceptance | Speedup vs. C1 | Wave makespan (s) |
|---|---:|---:|---:|---:|---:|---:|---:|
| Qwen3.6-27B `groupwise-int` | 1 | 43.01 | 1.00 | 185.8 | 68.2% | 1.00× | 44.23 |
| Qwen3.6-27B `groupwise-int` | 2 | 65.01 | 2.00 | 247.0 | 69.0% | 1.33× | 66.67 |
| Qwen3.6-27B `groupwise-int` | 4 | 102.02 | 4.00 | 309.5 | 68.4% | 1.67× | 107.49 |
| Qwen3.6-27B `groupwise-int` | 8 | 118.02 | 8.00 | 535.0 | 68.3% | 2.88× | 125.20 |
| Qwen3.6-27B `nvfp4` | 1 | 39.01 | 1.00 | 202.4 | 69.3% | 1.00× | 40.46 |
| Qwen3.6-27B `nvfp4` | 2 | 39.01 | 2.00 | 399.7 | 71.4% | 1.97× | 41.82 |
| Qwen3.6-27B `nvfp4` | 4 | 44.01 | 4.00 | 699.7 | 69.3% | 3.46× | 47.92 |
| Qwen3.6-27B `nvfp4` | 8 | 55.01 | 8.00 | 1,146.9 | 68.6% | 5.67× | 58.57 |
| Qwen3.6-35B-A3B `groupwise-int` | 1 | 12.00 | 1.00 | 593.0 | 67.2% | 1.00× | 13.75 |
| Qwen3.6-35B-A3B `groupwise-int` | 2 | 17.00 | 2.00 | 877.7 | 68.2% | 1.48× | 18.87 |
| Qwen3.6-35B-A3B `groupwise-int` | 4 | 26.01 | 4.00 | 1,166.0 | 69.8% | 1.97× | 28.43 |
| Qwen3.6-35B-A3B `groupwise-int` | 8 | 48.01 | 8.00 | 1,313.8 | 67.3% | 2.22× | 50.20 |

All 45 requests reached their output limit, producing 368,640 completion tokens. The campaign
contained 608 complete full-batch steady intervals and had no request, CUDA, or out-of-memory
failure. At C=8, available device memory after startup was 2.66 GiB for 27B groupwise-int,
2.18 GiB for 27B NVFP4, and 4.38 GiB for 35B-A3B.

## Qwen3.8-27B DFlash k=4 C<=4 retune

RTX 5090, CUDA Graph, NVFP4 KV, optimized proposal head, greedy
`long_decode_aime26_15`, 2,048 output tokens per request, max-context 16,384, and one compact
full-concurrency batch. The retained W=5 route aggregates the A16 attention-output, GDN-output,
MLP-down, fused SwiGLU, and supported fused attention-input projections across requests. Packed
GDN conv-record uses one request-indexed SmallT launch for C=2..4 instead of C serialized T=1
launches; C=1 retains the post-origin T=1 route.
Attention-input aggregation is limited to the NVFP4 and BF16-control routes used by this
artifact; Q4/Q5 and other verify widths retain their prior panel execution until separately
qualified.

| C | All panels tok/s | Residual aggregate tok/s | Projection aggregates tok/s | Concurrent GDN tok/s | vs prior | vs panels |
|---:|---:|---:|---:|---:|---:|---:|
| 2 | 161.3 | 173.6 | 187.5 | **219.9** | +17.3% | +36.3% |
| 3 | 163.6 | 177.8 | 193.3 | **229.6** | +18.8% | +40.4% |
| 4 | 164.4 | 173.5 | 186.8 | **220.6** | +18.1% | +34.2% |

The same shared target route improves matched MTP4 serving without changing acceptance:

| C | All panels tok/s | Residual aggregate tok/s | Projection aggregates tok/s | Concurrent GDN tok/s | vs prior | vs panels |
|---:|---:|---:|---:|---:|---:|---:|
| 2 | 156.7 | 167.9 | 178.4 | **212.1** | +18.9% | +35.3% |
| 3 | 162.3 | 177.2 | 190.9 | **229.5** | +20.2% | +41.4% |
| 4 | 164.8 | 175.2 | 188.1 | **225.3** | +19.8% | +36.7% |

The post-origin C=2..4 runs use the same artifact and prompts. All-panel, residual-only,
projection, and concurrent-GDN runs have identical output hashes,
speculative rounds, drafted-token counts, accepted-token counts, and acceptance. Relative to
five-token panels, the aggregated NVFP4 SwiGLU launches at T=10/15/20 measured
114.7/165.9/213.1 us instead of
174.1/233.5/294.9 us. NVFP4 attention-input measured 59.4/75.8/92.2 us instead of
73.7/102.4/131.1 us; the six BF16-control attention layers measured 108.5/124.9/137.2 us instead
of 202.8/301.1/399.4 us. The BF16 panel-preserving phase order is intentionally global at these
shapes: against the prior standalone packed route, T=10 is 1.9% slower, T=15 is 13.0% slower
(110.6 to 124.9 us), and T=20 is unchanged. That cost is dominated by eliminating two repeated
weight passes in the supported W=5 C=3 production route, which is 58.5% faster than its panels.
At the complete public GDN conv-record Op, W=5 C=2 cold Graph replay fell from 217.088 to
110.592 us (-49.1%); the superseded pair-within-CTA candidate measured 215.072 us and was removed.

Independent mathematical Op oracles cover every new T=10/15/20 route. The residual T=20
schedule retains the T=5 panel's 16-value-per-lane reduction; using the generic eight-value
schedule changed reduction association and failed the long C=4 identity gate. A true all-panels
versus combined C=4 probe was bit-identical across 512,000 sampled post-MLP BF16 values, 102,400
target-hidden BF16 values, 4,966,400 target-logit BF16 values, verifier IDs, cache positions,
argmax, and sampled live and ReplaySSM BF16/FP32 GDN state. Every checkpoint had zero mismatches,
zero relative L2 and maximum absolute error, and no nonfinite values. The GDN change adds direct
intermediate coverage at its public boundary: W=5 C=2/3/4 dense, ragged, and tree-parent q/k/v/z
outputs and valid conv-record values are bit-exact to independent C=1 T=1 launches. Sixty-three
decoded-NVFP4 FP64 projection/convolution checks had no nonfinite values; worst relative L2 was
0.00269457 under 0.00315 and worst maximum absolute error was 0.00380876 under its 0.00738378
gross limit. Four 2,048-token Graph
streams also matched exactly, including 681 rounds, 2,724 drafts, 1,366 accepts, and 50.1468%
DFlash acceptance. Separately, target-only decode PPL remained 6.414141594 through the GDN change:
all 2,047 scored FP32 NLL values were byte-identical, with 19 terrible tokens and no nonfinite
values. That PPL route does not exercise concurrent packed verification.

The subsequent GQA retune removes the remaining per-request attention loop. Concurrent NVFP4
SmallT verification now launches one request-indexed partial and one batched reduction per chunk;
the B=1 specialization and each request CTA's arithmetic remain unchanged. At public Graph Op scope
over W=2..5 and contexts 37/128/2,048/4,096, the mean speedups were 1.68x/1.97x/2.40x at
C=2/3/4. Representative W=5 latency fell from 22.528 to 14.016 us at C=2, from 32.768 to
14.336 us at C=3, and from 42.976 to 16.064 us at C=4 for context 37; at context 4,096 the
corresponding changes were 36.864 to 24.576 us, 53.248 to 34.816 us, and 69.632 to 40.960 us.
The existing 64-value reduction chunk remained best or tied at every sampled C=2/C=4, W=2/W=5,
context-128/context-4,096 point; 8/16/32-value alternatives were 11% to 105% slower.

Two matched 2,048-token fixed-wave serving A/Bs produced geometric-mean gains of 0.24% at C=2
and 1.33% at C=4; the uncontended pair improved C=2 from 219.50 to 221.02 aggregate tok/s and
C=4 from 220.86 to 224.54 tok/s. The complete first pass kept C=1 within 0.3% noise and measured
a 0.36% C=3 gain. Every lane retained the same output hash; speculative rounds, drafts, accepts,
and 50.1468% acceptance were exact at each concurrency. Independent mathematical-oracle coverage
includes dense, ragged, fragmented-page, permuted-table, W=2 C=2, and W=5 C=4 NVFP4 batches.
Packed real-artifact C=4 Graph isolation also retained exact C=1 outputs through the qualified
token window. A rebuilt-candidate NVFP4 decode PPL rerun remained 6.414141594 over 2,047 scored
tokens, and its complete FP32 NLL stream was byte-identical to the retained pre-change file.

A proposed fusion of the DFlash2 proposer MLP gate-up Linear and SiLU stages saved up to 2.0 us
in isolation but reduced production throughput in its matched pre-rebase C=2/3/4 campaign, so it
was removed.

The pre-rebase C=4 CUDA Graph node trace used to select the projection work attributed 32.5% of
kernel time to SwiGLU, 20.7% to GDN record, 17.6% to MLP down, and about 1% to DFlash top-k
selection. A fresh post-origin C=2 trace then exposed GDN request serialization as 36.1% of the
incremental round cost, motivating the concurrent route.

Reports: `profiles/bench/multi-request-kernel-post-origin-panel-control-20260904/`,
`profiles/bench/multi-request-kernel-post-origin-mtp4-panel-control-20260904/`,
`profiles/bench/multi-request-kernel-post-origin-residual-only-fixed-20260904/`, and
`profiles/bench/multi-request-kernel-post-origin-final-20260904/`. Concurrent-GDN results are in
`profiles/bench/multi-request-kernel-gdn-{final,mtp4-final}-20260904/`; its Op A/B is in
`profiles/bench/gdn-c2-retune-20260904/`. Accuracy report:
`profiles/ppl/multi-request-kernel-gdn-final-20260904.{json,nllf32}`. The GQA Op and serving A/Bs
are in `profiles/bench/multi-request-gqa-{baseline,candidate}-20260904/` and
`profiles/bench/multi-request-gqa-e2e-{baseline,candidate}-20260904/`; the uncontended repeat is
in `profiles/bench/multi-request-gqa-e2e-{baseline,candidate}-clean-20260904/`. The PPL rerun is
in `profiles/ppl/multi-request-gqa-candidate-20260904/`.

## Reproduction

Build `ninfer-serve` and prepare the registered `.ninfer` artifacts. The refreshed per-target
serving tables use:

```bash
python3 tools/bench/run_serve_concurrency.py \
  --serve build/apps/ninfer-serve \
  --artifact qwen3_6_35b_a3b=out/qwen3_6_35b_a3b.ninfer \
  --mode mtp3 --suite corpus-makespan --concurrency 1 \
  --max-context 262144 --kv-capacity auto \
  --output profiles/bench/concurrent_corpus_35b_mtp3_20260811

python3 tools/bench/run_serve_corpus.py \
  --serve build/apps/ninfer-serve \
  --artifact qwen3_6_27b=out/qwen3_6_27b.ninfer \
  --mode mtp3 \
  --output profiles/bench/serve_corpus_27b_mtp3_20260724

python3 tools/bench/run_serve_corpus.py \
  --serve build/apps/ninfer-serve \
  --artifact qwen3_6_27b=out/qwen3_6_27b_nvfp4.ninfer \
  --mode mtp0 --sampling stochastic \
  --output profiles/bench/serve_corpus_27b_nvfp4_w8_20260731

python3 tools/bench/run_serve_concurrency.py \
  --serve build/apps/ninfer-serve \
  --artifact qwen3_6_27b=out/qwen3_6_27b_nvfp4.ninfer \
  --mode mtp3 --suite corpus-makespan --concurrency 1 \
  --max-context 131072 --kv-capacity auto \
  --output profiles/bench/concurrent_corpus_27b_nvfp4_mtp3_20260811
```

The concurrent decode-saturation campaigns use:

```bash
python3 tools/bench/run_serve_concurrency.py \
  --serve build/apps/ninfer-serve \
  --artifact qwen3_6_27b=out/qwen3_6_27b.ninfer \
  --mode mtp3 --suite decode-saturation \
  --concurrency 1 --concurrency 2 --concurrency 3 --concurrency 4 \
  --decode-tokens 8192 --max-context 16384 --kv-capacity auto \
  --output profiles/bench/concurrent_decode_27b_mtp3_20260811

python3 tools/bench/run_serve_concurrency.py \
  --serve build/apps/ninfer-serve \
  --artifact qwen3_6_27b=out/qwen3_6_27b_nvfp4.ninfer \
  --mode mtp3 --suite decode-saturation \
  --concurrency 1 --concurrency 2 --concurrency 3 --concurrency 4 \
  --decode-tokens 8192 --max-context 16384 --kv-capacity auto \
  --output profiles/bench/concurrent_decode_27b_nvfp4_mtp3_20260811

python3 tools/bench/run_serve_concurrency.py \
  --serve build/apps/ninfer-serve \
  --artifact qwen3_6_35b_a3b=out/qwen3_6_35b_a3b.ninfer \
  --mode mtp3 --suite decode-saturation \
  --concurrency 1 --concurrency 2 --concurrency 3 --concurrency 4 \
  --decode-tokens 8192 --max-context 16384 --kv-capacity auto \
  --output profiles/bench/concurrent_decode_35b_mtp3_20260811
```

Use `--mode dflash7` for the corresponding DFlash block=8 campaign; add `--sampling greedy` for
the exact-argmax profile. Qwen3.8-27B NVFP4 DFlash2 uses the same flag on a reconverted artifact.
INT8-KV C=1 (the table below) and NVFP4-KV C=1/2/3 (the fused-GDN campaign after it):

```bash
python3 tools/bench/run_serve_concurrency.py \
  --serve build/apps/ninfer-serve \
  --artifact qwen3_8_27b=out/qwen3_8_27b_nvfp4_dflash_w8.ninfer \
  --mode dflash7 --mode mtp3 --mode mtp5 --mode mtp0 \
  --sampling stochastic \
  --temperature 0.6 --top-p 0.95 --top-k 20 --min-p 0 --presence-penalty 0 \
  --concurrency 1 --suite decode-saturation \
  --saturation-fixture long_decode_aime26_15 \
  --decode-tokens 4096 --max-context 16384 --kv-capacity 16384 \
  --output profiles/bench/qwen38_dflash2_c1_aime

python3 tools/bench/run_serve_concurrency.py \
  --serve build/apps/ninfer-serve \
  --artifact qwen3_8_27b=out/qwen3_8_27b_nvfp4_dflash_nvfp4.ninfer \
  --mode dflash4 --mode dflash7 \
  --sampling stochastic \
  --temperature 0.6 --top-p 0.95 --top-k 20 --min-p 0 --presence-penalty 0 \
  --concurrency 1 --concurrency 2 --concurrency 3 \
  --suite decode-saturation \
  --saturation-fixture long_decode_aime26_15 \
  --decode-tokens 4096 --max-context 16384 --kv-capacity auto \
  --kv-dtype nvfp4 --prefill-chunk 4096 \
  --output profiles/bench/qwen38_dflash2_fused_batch_aime_20260829
```

## Qwen3.8-27B NVFP4 C=1 decode

Same GPU, INT8 KV, graphs on, `--lm-head-draft`, seed `7632647173703958409`. The DFlash2 W8
companion is appended on `out/qwen3_8_27b_nvfp4_dflash_w8.ninfer`; MTP points load that same file
with DFlash host-placed. `long_decode_aime26_15` uses 4096 output tokens and max-context 16384.
Story is `scenario_story_en_mystery` at 1024 output tokens.

| Mode | Workload | Decode tok/s | Accept | Tokens/round |
|---|---|---:|---:|---:|
| MTP0 | AIME, stochastic | 86.0 | — | — |
| MTP3 | AIME, stochastic | 184.7 | 51.9% | 2.56 |
| MTP5 | AIME, stochastic | 195.9 | 41.1% | 3.06 |
| DFlash2 k=7 W8 | AIME, stochastic | 126.7 | 27.5% | 2.92 |
| MTP3 | AIME, greedy | 184.9 | 52.0% | 2.56 |
| DFlash2 k=7 W8 | AIME, greedy | 132.0 | 29.1% | 3.04 |
| MTP3 | story, stochastic | 163.5 | 41.9% | 2.26 |
| DFlash2 k=7 W8 | story, stochastic | 115.7 | 21.6% | 2.51 |

DFlash2 is a supported exclusive backend on this identity (`--spec dflash --draft-tokens 7
--lm-head-draft`; verify is the paper-accurate chain W=8). These first C=1 INT8-KV points beat
MTP0 (1.47× on stochastic AIME) and trail MTP3 (0.69×) and MTP5 (0.65×). Greedy AIME and story
show the same gap: DFlash2 accept is about 22–29% versus MTP3 about 42–52%. That accept gap is
the current speed target; it is not a reason to drop the backend.

Isolated CLI NVFP4-KV AIME (`long_decode_aime26_15`, 4096 tokens, seed `7632647173703958409`,
presence penalty 0, `--lm-head-draft`) after fused batched GDN conv-record (2026-08-29):

| Mode | Decode tok/s | Accept | Tokens/round | Rounds |
|---|---:|---:|---:|---:|
| chain k=4 W=5 | 162.18 | 46.57% | 2.86 | 1431 |
| chain k=7 W=8 | 142.52 | 31.61% | 3.21 | 1275 |

k=4 remains the RTX 5090 speed recommendation: cheaper W=5 verify, not more tokens per round.
An earlier same-day W4A4 packed-verify CLI snapshot was faster at C=1 (k=4 **194.48** /
k=7 **166.82**) with higher k=4 accept (50.26%). That snapshot is retained in
[dflash2-tree-speed.md](maintainer/dflash2-tree-speed.md); it predates the C>1 fused GDN path.

Serve C=1/2/3 on the same AIME fixture, NVFP4 KV, graphs, `--lm-head-draft`, presence penalty 0.
Headline tok/s is aggregate `decode_tokens / wave_makespan` (GPU throughput). Per-request
`(completion-1)/decode_seconds` is the isolation metric. All 12 serve requests hit the 4096 output
limit. Logs: `profiles/bench/qwen38_dflash2_fused_batch_aime_20260829/`.

| Mode | C | Aggregate tok/s | Per-request tok/s | Accept | Tokens/round | Makespan (s) | vs C=1 |
|---|---:|---:|---:|---:|---:|---:|---:|
| chain k=4 W=5 | 1 | 162.1 | 162.6 | 46.6% | 2.86 | 25.26 | 1.00× |
| chain k=4 W=5 | 2 | 262.0 | 133.0 | 45.5% | 2.82 | 31.26 | 1.62× |
| chain k=4 W=5 | 3 | 324.4 | 112.1 | 47.3% | 2.89 | 37.87 | 2.00× |
| chain k=7 W=8 | 1 | 143.0 | 143.4 | 31.6% | 3.21 | 28.63 | 1.00× |
| chain k=7 W=8 | 2 | 230.8 | 121.3 | 29.7% | 3.08 | 35.49 | 1.61× |
| chain k=7 W=8 | 3 | 302.0 | 103.1 | 29.6% | 3.07 | 40.68 | 2.11× |

C=3 k=4 is **324 aggregate tok/s** on this 27B NVFP4 DFlash2 path. Per-request rate falls as
the GPU is shared; isolation still matches C=1 DFlash. k=4 wins both C=1 and C=3 aggregate
on this fixture.

Omit `--mode` and supply the two measured Qwen3.6 groupwise-int artifacts to run the complete
published Qwen3.6 MTP0/MTP3 campaign:

```bash
python3 tools/bench/run_serve_corpus.py \
  --serve build/apps/ninfer-serve \
  --artifact qwen3_6_35b_a3b=out/qwen3_6_35b_a3b.ninfer \
  --artifact qwen3_6_27b=out/qwen3_6_27b.ninfer \
  --output profiles/bench/serve_corpus_20260720
```

For the 27B NVFP4 accuracy run, start the model service with:

```bash
build/apps/ninfer-serve out/qwen3_6_27b_nvfp4.ninfer \
  --host 127.0.0.1 --port 18080 \
  --max-context 262144 --prefill-chunk 1024 --kv-dtype int8 \
  --spec mtp --draft-tokens 3 --lm-head-draft
```

Then run the repository's full 27B reasoning suite in a separate shell:

```bash
PYTHONPATH=eval eval/.venv/bin/python -m ninfer_eval run \
  --config eval/configs/qwen3_6_27b_reasoning.yaml \
  --suite reasoning_full
```

## `qwen3_8_27b`

### EvalScope reasoning accuracy

The measured file is
[`Ostfralla/Qwen3.8-27B-NVFP4-NInfer`](https://huggingface.co/Ostfralla/Qwen3.8-27B-NVFP4-NInfer)
(`qwen3_8_27b_nvfp4.ninfer`, SHA-256
`eaf8ad124256d0a0c1ebbbca442ca58eee4f97ab34a60a0b4d57e2b41e2c56d2`). That `nvfp4` weights identity
was evaluated twice through NInfer's OpenAI-compatible serving route with thinking enabled, MTP=3,
and a 262,144-token context limit. The two runs differ only in `--kv-dtype`. EvalScope 1.9.0 used
0-shot prompts, rule-based scoring, and one sample per problem with temperature 0.6, top-p 0.95,
top-k 20, presence penalty 1.0, and seed 42. All 258 samples completed and were scored for each KV
codec.

| KV | AIME 2025 | AIME 2026 | GPQA-Diamond |
|---|---:|---:|---:|
| `int8` | 100.00% (30 / 30) | 96.67% (29 / 30) | 89.90% (178 / 198) |
| `nvfp4` | 93.33% (28 / 30) | 100.00% (30 / 30) | 92.42% (183 / 198) |

These are single-sample results under the stated evaluation profile, not pass@k scores. Each
benchmark remains independently reportable; no combined score is computed. Qwen3.8-27B
`groupwise-int` was not part of this campaign.

Download the measured artifact, start the model service with the matching `--kv-dtype`, and then
run the 3.8 reasoning suite:

```bash
hf download Ostfralla/Qwen3.8-27B-NVFP4-NInfer \
  qwen3_8_27b_nvfp4.ninfer \
  --local-dir models
```

```bash
build/apps/ninfer-serve models/qwen3_8_27b_nvfp4.ninfer \
  --host 127.0.0.1 --port 18080 \
  --max-context 262144 --prefill-chunk 1024 --kv-dtype int8 \
  --spec mtp --draft-tokens 3 --lm-head-draft
```

```bash
build/apps/ninfer-serve models/qwen3_8_27b_nvfp4.ninfer \
  --host 127.0.0.1 --port 18080 \
  --max-context 262144 --prefill-chunk 1024 --kv-dtype nvfp4 \
  --spec mtp --draft-tokens 3 --lm-head-draft
```

```bash
PYTHONPATH=eval eval/.venv/bin/python -m ninfer_eval run \
  --config eval/configs/qwen3_8_27b_reasoning.yaml \
  --suite reasoning_full
```

## `qwen3_6_35b_a3b`

### MTP0 context-length profile

| Prompt tokens | Samples | Prefill tok/s | Server TTFT (ms) | Decode tok/s |
|---:|---:|---:|---:|---:|
| 7,680 | 5 | 15,544.3 ± 242.4 | 500.2 ± 7.8 | 271.1 ± 3.6 |
| 64,512 | 5 | 10,809.0 ± 95.3 | 6,009.9 ± 52.6 | 242.9 ± 1.3 |
| 130,048 | 5 | 7,828.4 ± 34.1 | 16,693.3 ± 71.2 | 219.4 ± 1.6 |
| 260,096 | 5 | 5,157.1 ± 52.4 | 50,598.8 ± 519.7 | 188.2 ± 2.1 |

### MTP3 long-reasoning decode

| Fixture | Samples | Completion tokens | Decode tok/s | MTP acceptance | MTP tokens/round |
|---|---:|---:|---:|---:|---:|
| `long_decode_aime26_01` | 5 | 8,223.0 ± 2,224.1 | 726.2 ± 22.9 | 82.8% ± 3.4% | 3.48 ± 0.10 |
| `long_decode_aime26_15` | 5 | 65,536.0 ± 0.0 | 620.3 ± 8.1 | 72.7% ± 1.4% | 3.18 ± 0.04 |
| `long_decode_aime26_30` | 5 | 52,977.8 ± 11,849.6 | 671.9 ± 8.8 | 80.1% ± 2.7% | 3.40 ± 0.08 |

### MTP3 cross-scenario decode

Each category contains three fixtures and five seeds per fixture, for 15 samples.

| Category | Samples | Decode tok/s | MTP acceptance | MTP tokens/round |
|---|---:|---:|---:|---:|
| Code | 15 | 657.6 ± 34.3 | 70.3% ± 5.5% | 3.11 ± 0.16 |
| Story | 15 | 456.2 ± 36.6 | 38.0% ± 6.0% | 2.14 ± 0.18 |
| Translation | 15 | 649.7 ± 33.0 | 67.6% ± 5.1% | 3.03 ± 0.15 |
| Structured | 15 | 770.9 ± 29.3 | 89.1% ± 4.9% | 3.67 ± 0.15 |

### DFlash block=8 (`k=7`), stochastic sampling

The fixtures, five seeds, sampling parameters, and output limits are identical to MTP3. Different
speculative backends consume random values differently, so this is a fixed-workload comparison
rather than a token-identical paired-output comparison.

#### Long-reasoning decode

| Fixture | Samples | Completion tokens | Decode tok/s | DFlash acceptance | DFlash tokens/round |
|---|---:|---:|---:|---:|---:|
| `long_decode_aime26_01` | 5 | 8,495.4 ± 2,221.2 | 764.1 ± 55.6 | 65.2% ± 5.4% | 5.56 ± 0.38 |
| `long_decode_aime26_15` | 5 | 65,536.0 ± 0.0 | 584.0 ± 33.3 | 51.1% ± 3.7% | 4.58 ± 0.26 |
| `long_decode_aime26_30` | 5 | 53,330.4 ± 11,198.5 | 638.3 ± 15.8 | 56.4% ± 2.5% | 4.95 ± 0.17 |

#### Cross-scenario decode

| Category | Samples | Decode tok/s | DFlash acceptance | DFlash tokens/round |
|---|---:|---:|---:|---:|
| Code | 15 | 562.3 ± 36.2 | 43.0% ± 3.7% | 4.01 ± 0.26 |
| Story | 15 | 261.7 ± 51.1 | 12.1% ± 5.3% | 1.85 ± 0.37 |
| Translation | 15 | 490.8 ± 62.6 | 34.8% ± 6.3% | 3.44 ± 0.44 |
| Structured | 15 | 786.4 ± 124.7 | 66.5% ± 13.5% | 5.66 ± 0.94 |

#### Decode throughput versus MTP3

| Workload | MTP3 tok/s | DFlash tok/s | DFlash change |
|---|---:|---:|---:|
| `long_decode_aime26_01` | 726.2 | 764.1 | +5.2% |
| `long_decode_aime26_15` | 620.3 | 584.0 | -5.9% |
| `long_decode_aime26_30` | 671.9 | 638.3 | -5.0% |
| Code | 657.6 | 562.3 | -14.5% |
| Story | 456.2 | 261.7 | -42.6% |
| Translation | 649.7 | 490.8 | -24.5% |
| Structured | 770.9 | 786.4 | +2.0% |

### DFlash block=8 (`k=7`), greedy sampling

Greedy uses exact argmax; all other corpus and server settings remain unchanged. The five seeds
repeat the same deterministic generation path, so within-fixture standard deviation measures
runtime variation rather than output variation.

#### Long-reasoning decode

| Fixture | Samples | Completion tokens | Decode tok/s | DFlash acceptance | DFlash tokens/round |
|---|---:|---:|---:|---:|---:|
| `long_decode_aime26_01` | 5 | 6,692.0 ± 0.0 | 872.4 ± 3.3 | 74.4% ± 0.0% | 6.21 ± 0.00 |
| `long_decode_aime26_15` | 5 | 65,536.0 ± 0.0 | 651.6 ± 0.6 | 58.6% ± 0.0% | 5.10 ± 0.00 |
| `long_decode_aime26_30` | 5 | 65,536.0 ± 0.0 | 994.9 ± 3.4 † | 98.0% ± 0.0% | 7.86 ± 0.00 |

† The generation is a deterministic repetition loop, not a valid AIME response. The raw rate is
retained to describe what was measured, but is excluded from performance comparisons.

#### Cross-scenario decode

| Category | Samples | Decode tok/s | DFlash acceptance | DFlash tokens/round |
|---|---:|---:|---:|---:|
| Code | 15 | 599.8 ± 12.3 | 46.4% ± 1.4% | 4.25 ± 0.10 |
| Story | 15 | 291.5 ± 55.6 | 14.9% ± 5.7% | 2.04 ± 0.40 |
| Translation | 15 | 475.5 ± 50.6 | 33.0% ± 5.1% | 3.31 ± 0.36 |
| Structured | 15 | 869.0 ± 120.2 | 74.5% ± 13.1% | 6.21 ± 0.92 |

#### Decode throughput versus stochastic DFlash

| Workload | Stochastic tok/s | Greedy tok/s | Greedy change |
|---|---:|---:|---:|
| `long_decode_aime26_01` | 764.1 | 872.4 | +14.2% |
| `long_decode_aime26_15` | 584.0 | 651.6 | +11.6% |
| `long_decode_aime26_30` | 638.3 | 994.9 † | not comparable † |
| Code | 562.3 | 599.8 | +6.7% |
| Story | 261.7 | 291.5 | +11.4% |
| Translation | 490.8 | 475.5 | -3.1% |
| Structured | 786.4 | 869.0 | +10.5% |

### Speculative-decode output audit

The audit covers all 225 stored July responses from the 35B-A3B MTP3 stochastic-sampler, DFlash
stochastic-sampler, and DFlash greedy campaigns. It checks termination, exact repetition, and
fixture-specific mechanical constraints. AIME 1 was checked algebraically; the AIME 30 answer
(`393`) was checked by independent enumeration. This audit does not attempt to assign a subjective
quality score to prose or translations.

#### Long-reasoning answers

| Fixture | MTP3 stochastic sampler | DFlash stochastic sampler | DFlash greedy |
|---|---|---|---|
| `long_decode_aime26_01` | 5/5 correct, natural stop | 5/5 correct, natural stop | 5/5 correct, natural stop |
| `long_decode_aime26_15` | 0/5 answers; all reach 65,536-token limit | 0/5 answers; all reach 65,536-token limit | 0/5 answers; all reach 65,536-token limit |
| `long_decode_aime26_30` | 3/5 correct, 1 wrong, 1 no answer | 2/5 correct, 1 wrong, 2 no answer | 0/5 answers; all enter the same repetition loop |

The greedy AIME 30 response has an empty final-content field and fills its 65,536-token reasoning
budget. The exact line `Wait, $x_7 x_1 x_3$ is $x_7 x_1 x_3$.` occurs 2,406 times among 2,538
non-empty reasoning lines. Its 98.0% acceptance and 994.9 tok/s therefore characterize a highly
predictable pathological loop, not normal reasoning performance.

AIME 15 is also not a valid completion in any of the three campaigns: every sample exhausts the
budget without a boxed answer. Its output is long, non-convergent reasoning rather than the short
exact cycle seen in greedy AIME 30. The AIME 15 rates may be read only as sustained long-decode
throughput.

#### Cross-scenario outputs

| Category | MTP3 stochastic sampler | DFlash stochastic sampler | DFlash greedy |
|---|---|---|---|
| Code | 1/15 natural stops; 0/15 prompt-complete | 2/15 natural stops; 0/15 prompt-complete | 0/15 natural stops |
| Story | 9/15 natural stops; the nine Chinese outputs pass requested division and minimum length | 8/15 natural stops; the eight Chinese outputs pass requested division and minimum length | 10/15 natural stops; five Chinese dialogue outputs are under length |
| Translation | 15/15 natural stops; 15/15 pass structural checks | 15/15 natural stops; 15/15 pass structural checks | 15/15 natural stops; 15/15 pass structural checks |
| Structured | 0/15 satisfy the requested complete record/script contract | 0/15 satisfy the requested complete record/script contract | 0/15 satisfy the requested complete record/script contract |

The code prompts require complete runnable multi-file deliverables, but almost all outputs end at the
4,096-token limit. The three natural-stop exceptions also contain decisive contract failures: the
MTP3 CUDA response substitutes CUDA 12.8 and an older architecture list; the DFlash CUDA response
copies FP32 input into a half-sized 16-bit allocation and passes raw `unsigned short` values to BF16
intrinsics; and the DFlash Python response never writes its advertised JSONL event stream to the
configured log file. Code throughput is therefore a truncated-generation stress result, not
successful code-generation throughput.

All English mystery samples reach the output limit with an unfinished ending. The naturally stopped
Chinese stories have the requested chapter/act counts; the MTP3 and stochastic-DFlash samples also
meet their requested Chinese-character minima. Greedy's five dialogue stories contain 3,239 Chinese
characters each, below the requested 3,500. Story results are consequently a mixed normal/truncated
workload.

All translation outputs stop naturally. Each plain-document result preserves six sections and
provides at least twenty glossary entries; each Markdown result preserves heading levels, the
six-line table, all required inline identifiers, and the exact fenced JSON object. Translation is
the cleanest cross-scenario normal-completion comparison in this corpus.

The structured prompts intentionally exceed what these generations fit into 4,096 tokens. MTP3,
stochastic DFlash, and greedy DFlash produce only 49–60, 49–58, and 57 valid JSONL records,
respectively, versus the requested 160. Their complete-width CSV ranges are 122–139, 121–143, and
133 rows versus the requested 220. No SQL output satisfies all four tables, two views, at least 80
rows, and six final analytical queries. These high-acceptance results describe predictable partial
record generation only.

The exact-line and repeated-token scan found no other response with a short-cycle collapse comparable
to greedy AIME 30. Output-limit and prompt-compliance failures above remain material even when no
repetition loop is present.

## `qwen3_6_27b`

### EvalScope reasoning accuracy

Both weight profiles were evaluated through NInfer's OpenAI-compatible serving route with thinking
enabled, MTP=3, and a 262,144-token context limit. EvalScope 1.9.0 used 0-shot prompts, rule-based
scoring, and one sample per problem with temperature 0.6, top-p 0.95, top-k 20, presence penalty
1.0, and seed 42. All 258 samples completed and were scored for each profile.

| Weights ID | AIME 2025 | AIME 2026 | GPQA-Diamond |
|---|---:|---:|---:|
| `groupwise-int` | 86.67% (26 / 30) | 93.33% (28 / 30) | 86.87% (172 / 198) |
| `nvfp4` | 93.33% (28 / 30) | 93.33% (28 / 30) | 84.34% (167 / 198) |

These are single-sample results under the stated evaluation profile, not pass@k scores. Each
benchmark remains independently reportable; no combined score is computed.

### `groupwise-int`

#### MTP0 context-length profile

| Prompt tokens | Samples | Prefill tok/s | Server TTFT (ms) | Decode tok/s |
|---:|---:|---:|---:|---:|
| 7,680 | 5 | 3,218.1 ± 4.3 | 2,392.4 ± 3.0 | 77.6 ± 0.1 |
| 64,512 | 5 | 2,655.9 ± 2.9 | 24,335.7 ± 25.2 | 70.7 ± 0.1 |
| 130,048 | 5 | 2,185.3 ± 0.3 | 59,590.3 ± 8.9 | 64.5 ± 0.1 |
| 260,096 | 5 | 1,614.8 ± 0.6 | 161,221.8 ± 62.5 | 54.8 ± 0.1 |

#### MTP3 long-reasoning decode

| Fixture | Samples | Completion tokens | Decode tok/s | MTP acceptance | MTP tokens/round |
|---|---:|---:|---:|---:|---:|
| `long_decode_aime26_01` | 5 | 10,686.2 ± 553.8 | 175.4 ± 1.0 | 77.9% ± 0.9% | 3.34 ± 0.03 |
| `long_decode_aime26_15` | 5 | 61,604.2 ± 5,677.9 | 161.9 ± 2.8 | 73.4% ± 1.7% | 3.20 ± 0.05 |
| `long_decode_aime26_30` | 5 | 47,339.8 ± 9,162.2 | 172.2 ± 0.9 | 78.8% ± 0.8% | 3.36 ± 0.02 |

#### MTP3 cross-scenario decode

Each category contains three fixtures and five seeds per fixture, for 15 samples.

| Category | Samples | Decode tok/s | MTP acceptance | MTP tokens/round |
|---|---:|---:|---:|---:|
| Code | 15 | 167.0 ± 5.4 | 72.3% ± 3.5% | 3.17 ± 0.11 |
| Story | 15 | 112.6 ± 9.4 | 37.8% ± 5.9% | 2.13 ± 0.18 |
| Translation | 15 | 161.5 ± 11.3 | 68.3% ± 7.2% | 3.05 ± 0.22 |
| Structured | 15 | 193.0 ± 18.8 | 88.7% ± 11.7% | 3.66 ± 0.35 |

### `nvfp4`

The fixtures, seeds, sampling parameters, output limits, and runtime options are identical to the
groupwise-int serving campaign. Quantization can change sampled tokens, so the MTP3 results are a
fixed-workload comparison rather than a token-identical output comparison.

#### MTP0 context-length profile

| Prompt tokens | Samples | Prefill tok/s | Server TTFT (ms) | Decode tok/s |
|---:|---:|---:|---:|---:|
| 7,680 | 5 | 11,191.5 ± 70.2 | 692.5 ± 4.3 | 86.4 ± 0.5 |
| 64,512 | 5 | 6,298.5 ± 97.6 | 10,288.6 ± 159.3 | 78.0 ± 1.2 |
| 130,048 | 5 | 4,204.7 ± 14.1 | 31,012.5 ± 104.6 | 71.2 ± 0.2 |
| 260,096 | 5 | 2,510.6 ± 16.8 | 103,761.1 ± 698.8 | 59.9 ± 0.3 |

#### MTP3 long-reasoning decode

| Fixture | Samples | Completion tokens | Decode tok/s | MTP acceptance | MTP tokens/round |
|---|---:|---:|---:|---:|---:|
| `long_decode_aime26_01` | 5 | 12,053.4 ± 820.9 | 231.0 ± 3.0 | 80.2% ± 1.2% | 3.41 ± 0.04 |
| `long_decode_aime26_15` | 5 | 63,109.0 ± 5,426.9 | 213.1 ± 4.2 | 76.3% ± 2.0% | 3.29 ± 0.06 |
| `long_decode_aime26_30` | 5 | 57,166.4 ± 9,204.9 | 223.3 ± 1.8 | 81.1% ± 1.5% | 3.43 ± 0.04 |

#### MTP3 cross-scenario decode

Each category contains three fixtures and five seeds per fixture, for 15 samples.

| Category | Samples | Decode tok/s | MTP acceptance | MTP tokens/round |
|---|---:|---:|---:|---:|
| Code | 15 | 220.3 ± 8.2 | 74.2% ± 4.0% | 3.23 ± 0.12 |
| Story | 15 | 148.8 ± 11.6 | 39.2% ± 5.7% | 2.18 ± 0.17 |
| Translation | 15 | 213.6 ± 12.2 | 70.5% ± 6.0% | 3.12 ± 0.18 |
| Structured | 15 | 252.2 ± 16.3 | 89.8% ± 8.0% | 3.69 ± 0.24 |

The baseline and speculative-decode suites intentionally measure different supported workloads.
No per-scenario baseline/speculative speedup is reported.
