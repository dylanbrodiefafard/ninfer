# Qwen3.8-27B performance backlog

## Current non-kernel gate: GPU-gap attribution (2026-09-01)

The DFlash4 NVFP4 host-overhead campaign at C=1/2/3/4 removed avoidable telemetry and
terminal-publication work but showed no material uniform end-to-end decode gain. The next work is
therefore a **GPU-gap attribution gate**, not another speculative host cleanup:

1. Capture one warmed, steady DFlash4 decode wave at each supported C on the current artifact,
   NVFP4 KV, `--prefill-chunk 4096`, and fixed greedy workload.
2. Measure each GPU-idle interval between completed GPU work and the next CUDA submission, and
   assign it to one owner: executor scheduling/frame packing, graph selection/install, launch,
   D2H/output-state folding, or response publication.
3. Implement a CPU-side change only if a repeatable owned gap is large enough to move the
   end-to-end wave metric; benchmark it against the preserved cumulative baseline with exact work
   and output hashes.

Do not add an online graph autotuner, a new graph lookup cache, or per-round profiling output
without this attribution. The current transition-derived graph profiles and linear selection remain
the measured choice.

### Deferred correctness investigation: strict DFlash packed/T=1 oracle

After the GPU-gap item, investigate the current codebook artifact's strict target-only greedy
divergence in `ninfer_qwen3_8_27b_dflash_real_test` (DFlash k=4 chain) and the independent
DFlash k=7 C=3 row-isolation divergence from sequential C=1 DFlash. Relaxing the packed/T=1
target-only comparison does **not** relax row isolation, and the full relaxed suite still fails
that k=7 concurrent oracle. The host-overhead changes did not touch target math and their
pre/post benchmark output hashes match; nevertheless, do not treat relaxed testing as a
replacement for either strict oracle. Establish whether each mismatch is an accepted represented
precision boundary or a model/Op bug, then either fix it or state the supported numerical
criterion explicitly.

Working backlog for maximum single-GPU inference speed on the registered
`qwen3.8-27b` identities. Primary serving shape: **MTP**, `--lm-head-draft`,
INT8 KV, CUDA Graphs on, RTX 5090 / `sm_120a`.

This is not a second architecture spec. Contracts stay in
`docs/maintainer/qwen3.8-27b-artifact.md`, `qwen3.6-27b-model.md`,
`concurrent-inference-architecture.md`, and `op-development.md`. Remove or
rewrite this file when the backlog is exhausted or the product target changes.

Qwen3.8-27B shares the 27B execution package with Qwen3.6-27B. Almost every
item below is package work. The Qwen3.8-specific deltas are W8 embedding and
output-head on `groupwise-int`, Qwen3.8 tokenizer/template bytes, and the
registered sampling presets.

## 1. Decision: do not set `min-p=0` to make optimization easier

**No.** `min-p` is not a kernel fork and does not make the sampler, MTP graph,
or 27B body easier to optimize.

The sampler already treats `min_p <= 0` as disabled. With `min_p > 0` it only
drops a suffix of the **already truncated** candidate list (at most 20 ids)
after `exp`/`sum`:

```204:214:src/ops/kernel/sampling_device.cuh
        const float min_p_thresh = (cfg.min_p > 0.0f) ? cfg.min_p * e0 : -1.0f;
        const bool top_p_active  = (cfg.top_p < 1.0f);
        const float top_p_target = cfg.top_p * sum;
        // ...
            if (min_p_thresh >= 0.0f && prob[j] < min_p_thresh) { break; }
```

That does not change the 248k `sampling_partial_topk` grid, workspace, CUDA
Graph topology, or GEMM/attention launches. Cost vs a 27B MTP verify round is
noise.

What `min-p` *does* change is which token is drawn, therefore MTP accept length
and tokens/round. Turning it off “for speed work” mixes a quality/acceptance
shift into every A/B.

Registered Engine `min_p` for Qwen3.8 is already **0**. `--min-p 0.01` on the
live serve line is an override, not the package default.

Use one **frozen** sampling profile for all speed A/Bs and keep it for the
whole campaign:

| Knob | Recommendation |
|---|---|
| `min-p` | Leave at the registered 0, or keep 0.01 if that is the product serve line. Do not flip it between runs. |
| temperature / top-p / top-k | Freeze. Temperature moves MTP acceptance far more than `min-p`. |
| greedy | Allowed as a **kernel-timing** overlay (exact argmax, zero sampler variance). Not a substitute for the stochastic product line. |
| `--lm-head-draft` | Always on for the MTP product path. |
| `--spec mtp --draft-tokens` | Freeze k while measuring a kernel; sweep k as its own experiment (item B1). |

Do not add a sampler special-case, a second graph, or a “fast path” that is
only legal at `min_p=0`.

## 2. Target, baseline, and how to measure

### 2.1 Identities

| Artifact | Weights | Notes |
|---|---|---|
| `qwen3_8_27b_nvfp4.ninfer` | `nvfp4` | Same object layout as Qwen3.6-27B NVFP4. Current live serve line. |
| `qwen3_8_27b.ninfer` | `groupwise-int` | Qwen3.8 delta: W8 embedding and full output head. Slower body; same MTP host path. |

Run the campaign on the artifact that is actually served. Quote the other
profile only when a claim is about that profile.

Qwen3.8 is **not** in `docs/performance.md`. Do not cite 3.6 tables as 3.8
results. Use 3.6-27B MTP3 only as a prior: roughly 213–231 tok/s NVFP4 and
162–175 tok/s groupwise-int at C=1 on a 5090.

### 2.2 Product flags to pin

Typical live shape (adjust here if the serve line changes):

```text
--spec mtp --draft-tokens 3 --lm-head-draft
--kv-dtype int8 --kv-capacity auto --max-context 260000
--max-concurrency 2
```

CUDA Graphs stay on. `--no-cuda-graph` is a debug overlay, not a candidate
default.

### 2.3 Metrics

| Metric | Use |
|---|---|
| committed decode tok/s | primary, from server unrounded `decode_seconds` |
| MTP acceptance and tokens/round | required whenever k, sampling, or draft head changes |
| server TTFT (`prepare + vision + prefill`) | host / prefill / MTP bridge |
| C=2 decode-while-prefill aggregate tok/s | only for chunk-policy work |
| startup / `upload_seconds` | load path only |

An isolated op bench supports a kernel claim. It does not close an end-to-end
claim.

### 2.4 First profile (gate for everything else)

Warmed `ninfer-serve`, Qwen3.8 artifact, MTP3, C=1 then C=2. nsys one round:

1. target verify T=K+1 through 64 layers
2. sampler on those columns
3. MTP alignment forward
4. `draft_head` GEMM × K
5. MTP AR steps
6. D2H + CPU `OutputSession::preview`
7. `gdn_replay_fold` + second sync
8. `cudaGraphExecUpdate` when E crosses a graph bucket

Stop collecting once those slices can be ranked. Do not start a full ncu
campaign until a named kernel is the live decision.

## 3. Workstreams

Expected ranges are hypotheses until §2.4 exists for this artifact. “Body
kernels” are owned by the parallel decode/prefill campaign; they stay on the
list so this file is complete, but this backlog does not duplicate that work.

### A. 27B body kernels (owned elsewhere)

Do not pick these up here unless that campaign hands them back.

| ID | Item | Why it matters | Expected e2e | Difficulty |
|---|---|---|---|---|
| A1 | NVFP4 / groupwise decode GEMM, GQA, GDN recurrent | Most of an MTP verify at T=4 | large | high |
| A2 | Prefill GEMM / attention / GDN chunked | TTFT and C=2 decode stalls | large on prefill | high |
| A3 | W8 output head (Qwen3.8 groupwise-int) | Target-verify lm_head; 3.8-specific vs 3.6 groupwise | medium on that profile | high |
| A4 | Fused RMSNorm / RoPE / residual inside the graph | Launch tax already gone; math still separate | small unless nsys says otherwise | med |

### B. MTP product levers (this backlog, first)

| ID | Item | Why | Expected | Difficulty | Profile? |
|---|---|---|---|---|---|
| B1 | Sweep `--draft-tokens` 3 vs 4 vs 5 on Qwen3.8, keep `--lm-head-draft` | tok/s = tokens_per_round / ms_per_round. Extra AR is already in the graph. | 0–15%+ or a loss if acceptance drops | low | yes (corpus, not ncu) |
| B2 | Keep `--lm-head-draft`; never A/B the full 248k proposal head as a “speed” option | Full head is three 248k GEMMs per round | regression | n/a | no |
| B3 | C=2 `--prefill-chunk` 256 / 512 / 1024 | One prefill owner; decode waits a full chunk. ~1024 NVFP4 tokens can be several MTP rounds. | 5–20% of **decode-while-prefill** TPS; may hurt prefill tok/s | med | **yes** |
| B4 | Do not default DFlash | 27B package does not support it | n/a | n/a | n/a |

### C. MTP round structure (this backlog, after nsys)

MTP3 graph body is `mtp_decode_batch_body` in
`src/targets/qwen3_6/impl/runtime/mtp_impl.h`: H2D → verify/accept → MTP
alignment → propose ×3 → D2H. Fold is **after** CPU preview in
`program_impl.h`.

| ID | Item | Why | Expected | Difficulty | Profile? |
|---|---|---|---|---|---|
| C1 | Time Fold + CPU preview gap | 48 GDN layers, second `device.synchronize()`, stop strings need host detokenize | 1–5% if ≳0.3 ms on a ~15 ms round; else ignore | high to put Fold in the graph; low to measure | **yes** |
| C2 | Sampler on T=K+1 columns (248k vocab, ≤20 candidates, `min-p` irrelevant) | 4× ordinary decode sampler | 0–5% | med–high (oracle) | **yes** |
| C3 | `text/draft_head` Q4 GEMM `[131072,5120]` × K | MTP-only; not the 27B body campaign | few % to low tens of % if nsys names it | med | **yes** |
| C4 | MTP layer (attn + SwiGLU) alignment + AR | One-layer draft, width 4 then 1 | only if nsys names it | med | **yes** |
| C5 | `cudaGraphExecUpdate` at `{127,511,2047,…}` plus MTP3 cut near 1029 | 260k context / 50k gen **will** cross buckets | amortized 1–5% on long gens; hitches at boundaries | med | **yes** |
| C6 | Device-resident next-round drafts (skip D2H for the following ingress) | Architectural; host still needs tokens for stop policy | 0–3% | high | nsys D2H |

Do not recapture graphs at serving time. Do not fold “max-K then undo” into
GDN state.

### D. Host / TTFT (secondary for MTP tok/s)

| ID | Item | Metric | Expected | Difficulty |
|---|---|---|---|---|
| D1 | Tokenizer BPE (naive pair-merge) | long-prompt TTFT | 10–100+ ms at 8k if BPE dominates; prefill still owns 64k+ | med (bit-identical) |
| D2 | `encode_rendered_chat` double-encode on rewrite checkpoint | TTFT | small | low |
| D3 | Load-time `id → bytes` detokenize table | GPU-thread preview of up to K+1 tokens | &lt;1–3% tok/s at 27B MTP | low |
| D4 | MTP prefill bridge + eager AR propose on last chunk | TTFT | 1–5% of short-prompt TTFT | med |
| D5 | Eager 64-layer prefill wrappers + double sync on last chunk | prefill tok/s | &lt;1–5% | med |
| D6 | Vision FFmpeg + host bicubic + media mutex | vision TTFT | large on that slice | med (numeric resize contract) |
| D7 | HTTP SSE `nlohmann` dump | client inter-token latency, **not** Engine tok/s | 50–300 µs/token | low |
| D8 | KV-RAM pack / C=1 restore | TTFT on RAM hit (`--kv-ram-capacity`) | 0–15% of `kv_ram_load`; 0 tok/s on an in-flight MTP round | low–med |

### E. Load and toolchain (not tok/s)

| ID | Item | Expected | Notes |
|---|---|---|---|
| E1 | Clean CMake Release so `CMAKE_CUDA_FLAGS_RELEASE` is `-O3 -DNDEBUG` | &lt;1% graph-on decode; tiny prefill host | Docker does **not** add extra speed flags. Wipe a stale `build/` if the cache var is empty. |
| E2 | Stay on Release; no RelWithDebInfo `-lineinfo` in production | avoid debug metadata | already the advertised path |
| E3 | Do not add `--use_fast_math`, host `-ffast-math`, global `--maxrregcount`, RDC on NVFP4 TMA, RDC off on `ninfer_ops` | numeric / illegal insn | |
| E4 | Host `-march=native` / LTO / device `-dlto` | noise on GPU-bound MTP; `-dlto` can break `memory_evict` | only after e2e is bored |
| E5 | Artifact `O_DIRECT` + 4×64 MiB pinned H2D | 0–20% of **startup** | bind-mount `.ninfer` off overlayfs |
| E6 | Persistence mode / clocks / NUMA | process start, not warmed tok/s | not a CMake flag |

## 4. Order of work

1. **Gate:** §2.4 nsys on the served Qwen3.8 artifact, MTP k frozen, sampling frozen.
2. **B1** k=3/4/5 sweep on one long-reasoning fixture and one low-acceptance fixture (story-like).
3. **B3** only if C=2 overlapping prefill+decode is a real workload.
4. **C1–C5** in nsys rank order. Do not start Fold-in-graph or sampler fusion until the slice is ≥ a couple percent.
5. **A\*** stays with the body-kernel campaign. Hand C3/C4 to them if nsys says the draft stack is the round.
6. **D\*** when the goal is TTFT or streaming latency, not MTP tok/s.
7. **E1** whenever a local `build/` cache is empty; otherwise ignore toolchain.

Stop a line when the next experiment cannot change the live default.

## 5. Explicit non-goals

- Mixed prefill+decode in one traversal
- C ≫ 8, preemption, multi-GPU
- CPU sampling or logits D2H
- Serving-time graph capture
- Changing OpenAI/Anthropic wire JSON for speed
- Treating Docker as a faster runtime than local Release+Ninja
- Flipping `min-p`, temperature, or k inside a kernel A/B

## 6. Open questions (fill from §2.4)

- [x] Qwen3.8 NVFP4 MTP3 C=1 tok/s, acceptance, tokens/round vs the 3.6 proxy
- [x] Round ms split: verify / sampler / draft_head×K / MTP AR / Fold+preview
- [x] k=4 and k=5 on this checkpoint
- [x] C=2 chunk 256/512/1024 vs decode-round ms
- [x] Graph-update hitch (startup-range capture; 9 updates × ~0.24 ms)
- [ ] Whether the served line stays NVFP4 or also ships groupwise-int

Measured 2026-08-19, RTX 5090, `qwen3_8_27b_nvfp4.ninfer`, frozen sampling
(temp 0.6, min-p 0, presence 1.0), INT8 KV, graphs on, `--lm-head-draft`.

### Baseline (before speed work)

Short AIME, 4096 gen, max-context 16384:

| C | Decode tok/s | Prefill tok/s | MTP tok/round | Accept |
|---|---|---|---|---|
| 1 | 169.1 | 6270 | 2.34 | 44.7% |
| 2 | 163.4 | ~6300 | ~2.4 | ~46% |
| 3 | 152.7 | ~6280 | ~2.3–2.4 | ~44–46% |

High-context (thinking-on AIME after haystack; first NIAH run was invalid — 17-token stop):

| Wave | Prefill tok/s | Decode tok/s | TTFT | Accept |
|---|---|---|---|---|
| 50k C=1 | 7505 | 175.9 | 6.78 s | 55.6% |
| 50k C=2 | 7494 | 139.1 | 6.8 / 14.3 s | ~46% |
| 50k C=3 | 7461 | 127.8 | 6.8 / 14.3 / 22.0 s | ~48% |
| 100k C=1 | 5129 | 156.7 | 19.7 s | 52.3% |
| 100k C=2 | 5124 | 122.4 | 19.7 / 40.9 s | ~46% |
| 100k C=3 | 5122 | 114.4 | 19.7 / 41.0 / 62.5 s | ~50% |
| 150k C=1 | 3891 | 140.8 | 38.9 s | 49.5% |
| 150k C=2 | 3895 | 115.1 | 38.8 / 80.0 s | ~48% |

3.6-27B NVFP4 MTP3 proxy was ~213–231 tok/s at ~73–80% accept. Qwen3.8 is
slower because acceptance is ~45% on AIME, not because the round is much
longer.

### nsys gate (C=1, 256 gen after warmup)

GPU kernel share of the capture (includes short prefill):

| Slice | Share | Per-round | Action |
|---|---|---|---|
| NVFP4 body GEMM / GDN / residual (A1) | ~60%+ | most of ~14 ms | other campaign |
| W8 target lm_head T=K+1 (A3-class) | 5.5% | ~0.80 ms | other campaign |
| C3 `q4_rowsplit_gemv` draft_head ×K | 4.5% | ~0.64 ms | measured T=1 MMA; GEMV wins; do not ship |
| C1 `recurrent_fold` | 1.1% | ~0.17 ms | skip (&lt;0.3 ms) |
| C2 speculative sampler | 0.2% | ~0.03 ms | skip |
| C4 MTP pack/split/prepare | &lt;0.1% | noise | skip |
| C5 `cudaGraphExecUpdate` | 0.1% API | 9× ~0.24 ms | skip |
| C6 D2H inside `cudaMemcpyAsync` | mixed in 0.8% API | small | skip |

`cudaStreamSynchronize` is 77% of CUDA API time: host waiting on the graph,
not extra CPU work.

## 7. Per-item prefill / decode report

Deltas vs the §6 baseline. Decode primary is C=1 AIME 4096 unless noted.

| ID | What happened | Prefill tok/s | Decode tok/s | Notes |
|---|---|---|---|---|
| A1 | Not this campaign | — | — | Body GEMM/attn/GDN owns ~60%+ of GPU time |
| A2 | Not this campaign | — | — | Prefill GEMM/attn |
| A3 | Not this campaign | — | — | W8 verify head 5.5% of capture |
| A4 | Not this campaign | — | — | RMSNorm already in graph |
| B1 | Measured k=3/4/5 | 0% (same short prompt) | **k=4 AIME +2.9%** (173.7 vs 168.8); **k=5 −1.1%**; story k=4 **−3.6%**, k=5 **−5.7%** | Leave default k=3. k=4 wins only on high-accept AIME |
| B2 | Policy: keep `--lm-head-draft` | n/a | n/a | Full 248k proposal head not A/B'd |
| B3 | Measured chunk 256/512/1024 at 50k C=2 | **256: −33%** (4943 vs 7424); 512: −20% | **256: +4.6%** overlapping decode (145 vs 139); 512: +7.8% | Wall **worse** at 256 (+27%). Keep 1024 |
| B4 | Policy: no DFlash | n/a | n/a | 27B package does not support it |
| C1 | nsys only | 0% | 0% | Fold 0.17 ms/round; do not put in graph |
| C2 | nsys only | 0% | 0% | Sampler 0.2% of GPU |
| C3 | Measured T=1 MMA vs GEMV; **do not ship** | 0% | 0% (would be ~−0.2% if shipped) | 27B draft_head T=1: GEMV **233.5 µs** (91% HBM) vs MMA **243.7 µs**. 35B k=2048: tie ~106.5 µs. T=2 MMA already 245.8 µs — same roof. Serial K proposes stay serial. |
| C4 | nsys only | 0% | 0% | MTP layer kernels &lt;0.1% |
| C5 | nsys only | 0% | 0% | Update hitch ~0.24 ms at bucket crossings |
| C6 | nsys only | 0% | 0% | D2H not the round |
| D1+D2+D3 | Shipped together (heap BPE, one-pass checkpoint, id→bytes table) | **0%** (7505→7466, noise) | **0%** (169.1→168.9; 50k decode 175.9=175.9) | 50k **prepare 69→47 ms (−32%)**; TTFT unchanged (prefill owns it) |
| D4 | Measured last-chunk skip-inner-sync; **do not ship** | 0% | 0% | Short AIME TTFT 54.0→54.4 ms (noise). Last chunk already proposes MTP; extra sync is after GPU work. |
| D5 | Same measurement as D4 | 0% | 0% | Intermediate chunks must sync before `work_.reset()`. |
| D6 | N/A | — | — | Vision skipped this pass |
| D7 | Measured nlohmann vs concat; **do not ship** | 0% | 0% | Typical content chunk **1378→331 ns**. Unnoticeable vs a ~6 ms round. Hand-built JSON changes key order vs nlohmann `dump()`. |
| D8 | Measured pack vs pinned memcpy; **do not ship** | — | — | 27B GDN 28.75 vs 28.77 GB/s; fragmented 100 MiB KV 28.42 vs 28.74 GB/s. Restore 412 MiB in 14.3 ms at memcpy roof. |
| E1 | Shipped: `CMAKE_CUDA_FLAGS_RELEASE=-O3 -DNDEBUG` | 0% | 0% | Same A/B as D*; graph-on decode already optimized |
| E2 | Already Release | n/a | n/a | |
| E3 | Policy: do not add fast-math / maxrregcount | n/a | n/a | |
| E4 | Not shipped | — | — | GPU-bound MTP; `-dlto` risk |
| E5 | Not shipped | — | — | Startup only |
| E6 | Not shipped | — | — | Persistence/clocks; not warmed tok/s |

**Shipped code in this campaign:** tokenizer D1/D2/D3 and CMake E1. They do not
move warmed decode or prefill tok/s. The live product lever that actually
moved decode is **B1 k=4 on AIME only**; do not flip the default from k=3
without a quality call, because story acceptance drops.

**Largest remaining decode gap vs 3.6:** MTP accept ~45% vs ~75%. That is a
checkpoint/sampling issue, not a missing kernel in this backlog. Body GEMM
(A1) is still most of each round.
