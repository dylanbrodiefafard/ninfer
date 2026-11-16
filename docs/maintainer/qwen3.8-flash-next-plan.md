# Qwen3.8-Flash-Next experimental target plan

Temporary plan, started 2026-11-14 on branch `qwen3.8-flash-next`. It records the goal, the
measured baseline to beat, the budgets, the phase order, and per-item status. The target was
registered with the Engine on 2026-11-14 as the second supported identity (product contract in
`AGENTS.md`, CLI/serving/concurrency references updated together); the measurement against the
baseline below (phase 6) remains open. Remove this plan when that comparison is decided.

## Goal and constraints

- Model: `Qwen/Qwen3.8-Flash-Next` (180B: 120.8B routed experts, 51.2B n-gram table,
  about 4B dense, 4B MTP; 48 layers, hidden 2560). Text only; MTP not executed.
- Placement: the n-gram table on the ext4 boot SSD
  (`/home/battlefront/ai/qwen3.8-flash-next-ngram`); every routed expert in pinned host RAM
  with a dynamic VRAM cache; all computation on the GPU.
- Budgets: at most 50 GB of host RAM for weights; one RTX 5090 (32 GB) at PCIe Gen4 x16
  (the riser is rated Gen5; Gen5 may be enabled later).
- Quality bar: teacher-forced KL against BF16 no worse than the ISTA-DASLab GSQ-RCO IQ3_S
  GGUF on the same tokens, plus a short task check.
- Drafter: DFlash2 `incoai/Qwen3.8-Flash-Next-DFlash2` with NVFP4 matrices and a BF16
  selector codebook, as in the 27B production recipe.

## Baseline to beat

Strata 0.2.13 (engine 0.1.59, `ghcr.io/niko1221/strata:0.2.13-cuda13`), IQ3_S GSQ-RCO, this
host, 2026-11-14. The n-gram table was read from SSD (`--ple-io direct`), the container was
capped at 60 GB, context 64K, and the tuner was skipped. Its DFlash2 drafter is Q4_0 with
windows of up to 6 tokens, and it serves one request at a time.

| Phase | Strata |
|---|---|
| Prefill 1K tokens | 1.2-1.4K tok/s (10-12 CPU cores) |
| Prefill 4K | 2.8-3.2K tok/s |
| Prefill 13-16K | 4.9-6.5K tok/s |
| Prefill 35K | 6.2K tok/s |
| Decode, 1024 tokens | 172-216 tok/s, 41-50% draft acceptance (15-16 CPU cores) |
| Concurrent decode, C = 1/2/4, aggregate | 201 / 207 / 197 tok/s |

Raw results: `/home/battlefront/ai/bench/results/strata-iq3s.json`, produced by
`/home/battlefront/ai/bench/serve_bench.py`, which the ninfer server is measured with too.

## Physics that set the design

- One expert holds 3 x 640 x 2560 = 4.92M weights; 48 x 512 = 24,576 experts. At IQ3_S
  (3.44 bpw) one expert is about 2.1 MB and all of them about 50 GB.
- A prompt longer than a few hundred tokens routes to nearly every expert of every layer.
  Without a resident copy, each prefill chunk moves every non-resident expert across PCIe:
  about 24 GB at Strata's 12.7K-slot cache, about 1 s at Gen4. Short-prompt prefill is
  therefore bound by expert bytes and cache size; long-prompt prefill by GPU compute.
- Decode touches 10 experts per layer per token; the measured cache hit rate decides decode
  speed. A speculative verify of k tokens touches the union of their experts (1.75x-3x the
  single-token set for 2-4 rows), so batched verification and concurrent requests multiply
  misses unless the cache absorbs them.
- The n-gram path needs 16 table rows per token: one 4 KiB direct read each, about 130K reads
  for an 8K prompt, which the engine overlaps with the layer-0 work.

## Phases

| # | Item | Status |
|---|---|---|
| 0a | Strata baseline on this host | done (table above) |
| 0b | BF16 checkpoint, IQ3_S GGUF, DFlash2 drafter downloaded | done |
| 0c | 8-bit n-gram table (`int8_row`: FP16 row scale + 160 int8, 25 rows per 4 KiB page; row-relative error 0.65% vs BF16; its model-level cost is at the perturbation floor, see below), 52.4 GB at `/home/battlefront/ai/qwen3.8-flash-next-ngram/ngram_table.int8.bin`; 2000 random rows match an independent re-encode exactly | done |
| 0d | Quality harness: layer-streaming BF16 forward with KL/top-1/PPL per variant (BF16-self, IQ3_S, NVFP4-RTN, 8-bit n-gram); validated against the FP64 reference ops of branch `qwen4` and against llama.cpp `qwen4exp` | in progress (`/home/battlefront/ai/study/fnstudy`) |
| 0e | Expert format study (`/home/battlefront/ai/study/results/formats/REPORT.md`): `mixed_mse_gptq`, 50,289,399,424 bytes (IQ3_S experts 50,292,326,400): the 1,583 most-routed experts (22.4% of calibration routes) in GPTQ NVFP4 (E4M3 scale per 16), the other 22,993 in GPTQ F3, 3-bit indices into the E2M1 subset {±0.5, ±2, ±4, ±6} with an MSE-searched E4M3 scale per 32, one FP32 scale per matrix. F3 expands to native E2M1 nibbles and duplicated 16-wide scales, so both kinds feed the same FP4 MMA. Chosen best-of-13 on EVAL (KL 0.124 vs IQ3_S 0.141, paired ΔKL 95% [-0.035, -0.002]) and confirmed on a disjoint 27-sequence HOLDOUT (`results/formats/holdout.json`: KL 0.099 vs IQ3_S 0.109, paired ΔKL [-0.028, +0.004]; the F2/NVFP4 mix is worse there, 0.126). NVFP4-GPTQ alone reaches KL 0.073 at 67.9 GB | done (A16 evidence); task check at milestone M1; A8/A4 activation quality open |
| 1 | Target package `src/targets/qwen3_8_flash_next` with its own Program (the Qwen3.6 family runtime is a dense hidden-5120 Program and does not apply), `.ninfer` converter and FP64 reference under `tools/` | open |
| 2 | Ops ported from branch `qwen4` with their oracles: gated residual (HC), QSA index/select/attention, n-gram embedding and PLE, GDN at hidden 2560, sparse MoE; the routed MoE kernel follows upstream's exact-shape `h2560_i640_e512_k10` route, re-targeted to the chosen expert format | open |
| 3 | Expert residency: pinned host arena for every expert, VRAM slot cache seeded from a routing profile with decayed-frequency replacement, per-chunk streaming of non-resident experts during prefill, overlapped with compute | open |
| 4 | N-gram reader: `O_DIRECT` page reads through a bounded set-associative row cache, issued ahead of layer 1 | open |
| 5 | DFlash2 integration with batched verification | open |
| 6 | Measurement with `serve_bench.py` at C = 1/2/4/8 against the baseline above | open |

## Verified model facts

Checked against the BF16 checkpoint headers, `qwen4:tools/reference/qwen4/` and the study's
FP64 op validation (`/home/battlefront/ai/study/NOTES.md`):

- Dense tensors per GDN layer: `in_proj_qkv` [10240,2560] (q 2048, k 2048, v 6144), `in_proj_z`
  [6144,2560], `in_proj_a`/`in_proj_b` [48,2560], `conv1d` [10240,1,4], `A_log`/`dt_bias` [48],
  `norm` [128], `out_proj` [2560,6144]. QSA layers: `q_proj` [12288,2560] (per head q 256 then gate
  256, interleaved by head), `k_proj`/`v_proj` [512,2560], `q_norm`/`k_norm` [256], `o_proj`
  [2560,6144], `indexer.index_qk_proj` [640,2560] (query 512 then key 128), indexer norms [128].
  Every layer: `mlp.gate` [512,2560], shared expert gate/up [640,2560] and down [2560,640],
  `shared_expert_gate` [1,2560], and two hyper-connections (`hc_norm` [10240], down [320,10240],
  up [10240,320], `block_inject_weight` [4,10240]). Layer 1 adds PLE: `key_proj` [10240,2560],
  `value_proj` [2560,2560], `norm_key`/`norm_query`/`norm_conv` [10240], `conv1d` [10240,1,4].
  Model level: `embed_tokens` and `lm_head` [248320,2560] (untied) and a final
  `hyper_connection_mixer` without inject; there is no other final norm.
- GDN: grouped heads (value head h uses q/k head h/3), L2-normalized q/k with epsilon inside the
  square root and an extra 1/sqrt(128) on q, g = -exp(A_log) softplus(a + dt_bias), beta =
  sigmoid(b). Its output gate is **sigmoid(z)** (Qwen3.6-27B uses SiLU) and its output norm has a
  plain gamma; every other norm is zero-centered (1 + w).
- MoE: FP32 softmax over 512, top-10 with ties to the lower expert id, renormalized; gate rows
  precede up rows; shared expert scaled by sigmoid(shared_expert_gate x).
- The ISTA "IQ3_S" GGUF mixes six expert codecs (IQ2_S, IQ3_XXS, IQ3_S, IQ4_NL, IQ4_XS, Q2_0):
  3.33 bits per weight on average.

## Op port order

Branch `qwen4` (26 commits past merge-base `13f0868a`; HEAD is 162 past it) cannot be
cherry-picked: HEAD removed BF16 Linear and several GDN launchers, rewrote the NVFP4 registry,
reformatted the tree, and added a clang-tidy gate. Its code is re-typed onto HEAD in this order,
each step with its FP64 oracle from `qwen4:tools/reference/qwen4/`:

| Step | Content | Source | Status |
|---|---|---|---|
| 1 | Dense projections at hidden 2560, A16 (BF16 activations) at every T: NVFP4 Linear [2560,6144] and split-output NVFP4 `gdn_input_proj` [16384,2560] / `attn_input_proj` [13312,2560] (rows q 6144, k 512, gate 6144, v 512 after de-interleaving), FP8_E4M3FN_ROW_BF16S `gdn_input_proj` [16384,2560] and Linear [2560,6144] (the chosen recipe's GDN in/out and attention out), FP8 vocabulary head and embedding [248320,2560]. NVFP4 weights may carry no input divisor (A16-only). The shared expert runs inside the MoE Op | HEAD templates | done (oracle tests, memcheck/racecheck, clang-tidy, pi reviews). FP8 decode at T=1: GDN in 26.7 us (94% DRAM), [2560,6144] 12.3 us (76%); [2560,6144] is latency-bound at T=24..256 (split-K would close it) |
| 2 | N-gram row hashing: on the host inside the SSD reader (step 7), since the host issues the row reads; constants from the artifact (multipliers, head offsets and sizes), signed-I64 Euclidean remainder, a lag-k predecessor becomes EOS 248044 when any of the last k tokens is EOS | `qwen4:tools/reference/qwen4/ngram.py` | done: `NgramTable` (`src/targets/qwen3_8_flash_next/impl/ngram_table.*`) hashes on the host, reads each distinct missing page once with O_DIRECT on an I/O thread pool through an exact-LRU 8-way row cache; test matches the study's reference row ids (checkpoint constants and wrapping multipliers), the synthetic-table records, and page-read counts from an independent LRU model, and reuses the table after a failed read |
| 3 | `bf16_projection`: BF16 weights; FP32 output for the GDN a\|b gates [96,2560] (then `gdn_gating_packed`), the MoE control rows [544,2560], PLE key [10240,2560] and value [2560,2560]; BF16 output for the QSA indexer query [512,2560] and key [128,2560] (the converter splits `index_qk_proj`), which feed the BF16 QSA inputs; T=1 GEMV, T>=2 BF16 MMA | HEAD BF16 GEMV/MMA templates | done; oracle test and memcheck pass |
| 4 | Hyper-connection broadcast, read (with and without write gate), inject: FP32-internal SIMT kernels, token tiles of 8, K split 4 (`ops/hyper_connection`) | `qwen4` `gated_residual` | done; FP64-oracle test passes |
| 5 | GDN at hidden 2560 composes existing Ops: `gdn_input_proj` [16384,2560], `causal_conv1d_silu_split`, `bf16_projection` [96,2560] + `gdn_gating_packed`, the GDN core (same 16/48-head geometry as the 27B), `gated_rmsnorm` with the new sigmoid gate (`RmsNormGate::Sigmoid`), NVFP4 Linear [2560,6144] | HEAD | Ops done (sigmoid gate and packed gating tested); layer composition open; lane-batched (B<=12): `gdn_input_proj_conv_snapshot` (any W for B=1, W<=16 for B>1, in place at W=1) and `gdn_input_proj_conv_record` (chains W=1..16, trees W=2..16) admit the [16384,2560] NVFP4/FP8 parents under A16Only (one projection over all W*B columns, BF16 staging, slot-indexed convolution with split q/k/v), then the existing `gated_delta_net_snapshot` / `gated_delta_net_replay_record`; `gdn_replay_fold` admits the 36-layer pool; FP64-oracle, per-lane, memcheck/racecheck tests pass (`ninfer_lane_batched_state_bench`) |
| 6 | QSA state append, block index/select, selected attention (contiguous single-request state), then the layer composite; check the NVFP4 KV byte layout against HEAD's paged pool | `qwen4` `qsa*` | Ops done (`ops/qsa`: BF16 and NVFP4-G16 state, budgets 512/1024/2048 blocks, causal-prefix visibility, ascending block order; FP64-oracle test with an exact FP32-mean pooling boundary, memcheck, racecheck, clang-tidy; pi Sol review: no math or memory defect). Performance pass done (`ninfer_qsa_bench`, NVFP4 state, cold L2, RTX 5090): the state carries an FP32 sealed key per block (`block_keys` [128,C/4], C a multiple of 4), sealed by `qsa_state_append` (now taking the key norm weight) when a block's last slot is appended, so selection reads 512 B per block instead of rebuilding every key; streaming (W<32) and register-tiled (W>=32) FP32 scoring; radix select on an 8-CTA cluster with distributed-shared-memory histogram reductions for short chunks of rows over 8192 blocks; tensor-core attention (BF16 MMA, 12 heads of a KV group as rows, weights as BF16 hi+lo, each weight formed once), merge forms each split weight once. Decode W=1/W=12 at 256K, budget 2048: select 154/209 -> 49/66 us, attention 43/373 -> 23/64 us; prefill W=4096 at 256K: select 24.9 -> 8.0 ms, attention 78.2 -> 10.0 ms (82 TFLOP/s); no measured case regresses. FP64-oracle test extended (sealing across calls and reseal, new routes), memcheck, racecheck, clang-tidy, pi Sol reviews clean. Open: attention at W=1 stays latency-bound (16-23 us, two tiles per CTA); prefill scoring is FP32 SIMT at about 34 TFLOP/s (a tensor-core split-BF16 score would need a widened tie window); a BF16 sealed key would halve the 256K decode read but changes the score profile; then paged and batched-lane state. Paged lane-batched form done: the Ops take `QsaPagedState` (seven planes of a PageMajor `PagedKVPool` with 64-token pages, plus its block tables) and B rows of W columns with per-row lanes; the contiguous single-request form is removed. FP64-oracle tests over shuffled page tables, page-edge blocks and lanes with different frontiers, bitwise batched-vs-one-row checks, memcheck/racecheck pass. One lane against the contiguous form (NVFP4, budget 2048, cold L2): attention unchanged; select +2 us at 64K decode W=1/12, equal at 256K, +2% at W=4096; B=12 lanes at 256K: select 299/358 us, attention 72/209 us at W=1/4 |
| 7 | `ple_decode_rows` (int8_row records to BF16) and `ple_inject` (gate, conv-input norm, dilated convolution with carried BF16 history, inject), composed with `bf16_projection` key/value; SSD row reader in step 2 | `qwen4` `ple` reference | done; FP64-oracle test passes (decode rows exact; inject over chunked calls with carried history); `ple_inject` is lane-batched (B<=12 rows of W columns over per-row history slots [9,10240,S], in place) with a per-column `ple_inject_snapshot` (W<=16) for verify rollback; bitwise equal to one call per row |
| 8 | MoE routing: `bf16_projection` over the packed [544,2560] control matrix (router, shared-expert gate, zero padding) then `moe_route` (top-10, ties to the lower id, renormalized weights, shared scale), output layout as upstream's `ExpertRoutes`; token grouping, shared expert and merge belong to the routed-expert Op (step 10) | `qwen4` reference | routing done, tested |
| 9 | Expert slot cache and PCIe streaming (dynamic slot table; upstream's static windows are a reference, not the design) | new | open |
| 10 | Routed SwiGLU kernel for the chosen expert format, with `offsets[]` slot indirection as in upstream `include/ninfer/ops/moe.h` | new; needs 0e | open |

`qwen4`'s GDN layer kernels are not ported: they assume the tiled value-head order its GGUF
converter produced, while the BF16 checkpoint and HEAD's GDN core use grouped order.

## Execution design

The Qwen3.6 family runtime cannot be parameterized for this model: its layer loop adds every
block into one residual `[hidden,T]`, its leaf set is fixed to attention projection, GDN
projection and a dense post-mixer, its `ModelView` has fixed dense norms and MTP/DFlash/Vision
members, and family scheduling forbids target-dependent branches. The target package therefore
owns its Program (`src/targets/qwen3_8_flash_next/impl/runtime/`: `State`, `Forward`,
`Program`, `ExpertHostStore`/`ExpertSlotCache`/`ExpertResidencyPlanner`, `PleRowFeeder`), which
needs an explicit amendment of the AGENTS.md ownership table when the target is registered with
the Engine. It reuses `LinearAttentionStatePool` (same GDN geometry), `DecodeGraph`, the arenas
and layout builders, `ops::sample`, and the Qwen3.6 frontend if the artifact embeds its resources.

Per round: embedding, `hyper_connection_broadcast`, then per layer HC read, mixer (GDN or QSA),
HC inject, HC read, routing and `moe_experts`, HC inject; layer 1 runs `ple_inject` before its
mixer read; the head is a read-only HC read and the FP8 vocabulary head.

- Decode n-gram rows depend only on tokens the host holds at the round boundary, so
  `PleRowFeeder` gathers them between rounds for every lane of the round into a pinned row frame
  (one graph replay per round). Prefill gathers super-chunk k+1 on a host thread while super-chunk
  k computes. Each lane's n-gram history is its last two tokens, so rollback needs no state.
- Prefill runs layer-major over super-chunks (`ForwardOptions::super_chunk_tokens`): each layer
  runs over every sub-chunk (`max_chunk_tokens`, the Ops' width) of the super-chunk before the
  next layer, holding the super-chunk's four-stream residual (20 KiB per token) and n-gram rows
  (2.5 KiB per token). Per-layer GDN, convolution, QSA and PLE state advance sub-chunk by
  sub-chunk, so the logits are bit-identical to chunk-major prefill at the same sub-chunk width.
  A super-chunk of at least `window_min_tokens` admits nothing: each layer's non-resident expert
  records are placed (`moe_expert_cache_window`) and copied (`moe_expert_cache_fill`) once into
  one of two VRAM windows on the copy stream while the previous layer computes, and every
  sub-chunk's `moe_experts` reads them through its deferral. PCIe bytes per prompt therefore
  scale with the number of super-chunks, not sub-chunks, and the decode cache keeps its contents.
  The super-chunk is also the unit at which another lane's decode can interrupt a prefill.
- The device expert table is U64 `[48,512]` holding either a VRAM slot address or the mapped host
  address, so the routed kernel has no branch; it changes only on the device, through
  `moe_expert_cache_admit`, `_fill` and `_commit` (see "Expert residency design").
- Per-lane state: GDN pool slot (36 layers, about 113 MB per lane), PLE convolution history BF16
  `[9,10240]` (180 KiB) and sampling counts; QSA state lives in one paged pool shared by every
  lane. The n-gram row cache and the expert cache are Program-wide.

First milestone (M1) reached 2026-11-14 on the interim artifact `qwen3.8-flash-next/f3-nvfp4`
(NVFP4-RTN dense, NVFP4-RTN shared expert records, all experts in mapped host memory):
teacher-forced prefill of 4 x 512 HOLDOUT tokens (`tools/parity/qwen3_8_flash_next`) against the
study harness running the same recipe (`study/scripts/parity_reference.py`): mean KL(reference ||
engine) 0.0133 (p50 0.0008, p90 0.031, p99 0.20), top-1 agreement 96.8%, per sequence
0.005-0.020, no non-finite values; the recipe itself is 0.101 from BF16 and the perturbation
floor is about 0.027. Prefill versus decode (512-token prefill in 128-token chunks against 504
prefilled tokens plus 8 single-token steps, 4 sequences): top-1 agrees at 32 of 32 positions,
KL below 1e-3 at 26 and at most 0.022. A full run, load included, takes about 70 s. The parity
reference for the final artifact `f3-fp8-nvfp4` (study recipe `fallback_mid_parent_hc_gptq`, the
same 4 x 512 tokens) is `study/results/parity_reference_final_4x512.npz`: the recipe is 0.060 from
BF16 (top-1 93.8%) against 0.101 for the interim recipe. Target code:
`src/targets/qwen3_8_flash_next/impl/{config.h,model_view.h,load/binder.*,runtime/state.*,
runtime/ple_feeder.*,runtime/forward.*}`, driver `tools/parity/qwen3_8_flash_next/`.

M1 scope: one request, eager, BF16 QSA state, no speculation, experts read
through mapped host memory. Checked by teacher-forced logits on 4x512 tokens against the study
harness (`fnstudy/forward.py`, variants bf16 and nvfp4_rtn) and by prefill of N tokens against
prefill of N-k plus k decode steps. Then: B=1 decode graph, expert slot cache and prefill
streaming, PLE prefetch overlap, B<=12 batched forms (fused GDN projection with convolution
snapshot, slot-indexed `ple_inject`, batched QSA lanes), NVFP4 and paged QSA state, the chosen
expert codec, Engine integration, DFlash2.

## Engine integration design (2026-11-14)

The Engine's `ConcurrentExecutor<Instance>` (`src/runtime/engine/concurrent_executor.h`) needs
about 95 Program methods at compile time. Flash-Next implements the request lifecycle for real
(admission capacity, request/lane plans, per-chunk prefill, `decode_batch` over all lanes at
T=B, resolve, abort, suppression and typical-cycle hooks, reporting, `score`) and gives fixed
answers for mixed forward and context checkpoints, which it does not run. Shared RAM/SSD byte
transport and target-owned window/suspension semantics are described below.

- Batching: one forward over all lanes per round (concurrency contract §2.1); GDN through the
  lane-batched snapshot Ops (B up to 12), so Flash-Next admits C <= 12; per-lane PLE history;
  one multi-lane n-gram gather per round.
- QSA state lives in the paged pool (`PagedKVPool`, 64-token pages = 16 QSA blocks, 7 planes per
  QSA layer); the QSA Ops take block tables and B lanes. Shared capacity is the existing
  `--kv-capacity N`: NVFP4 QSA state is about 11.4 KiB per token, so 500K tokens is about 5.8 GB
  and 600K about 7.0 GB. Automatic capacity must leave VRAM for the expert cache (`max(L, C)`
  page groups by default); the expert cache takes what remains after construction.
- Frontend: reuse `qwen3_6::Frontend`; the Flash-Next chat template differs from the 27B
  ReasoningEffort template only by HF `{% generation %}` markers (byte-identical rendering) and
  its `tokenizer_config.json` has no `chat_template` field, so the frontend maps its digest to
  ReasoningEffort and treats the field as optional; the binder keeps the frontend resources.
- Registration: `ActiveTarget` and `Engine::Impl` become a two-alternative variant (27B,
  Flash-Next) dispatched with `std::visit`; a Flash-Next package header, a split
  `LoadedModel` load protocol, `EngineOptions::ngram_table_path` (`--ngram-table`) and
  `--qsa-select-budget`. Flash-Next supports optional native DFlash2 K0..3 and rejects vision,
  context checkpoints, INT8 KV and sparse or Sage attention. AGENTS.md's product contract and
  ownership table and the concurrency document are amended together when the target is
  registered. Status: done (2026-11-14). `EngineProgram`
  (`export/ninfer/targets/qwen3_8_flash_next/package.h`, mapping in its header) adapts the
  executor to the lane Program: per-lane prompt and sampled-token bookkeeping, one-shot token
  masks re-armed per sampled token, cycle-exit and suppressed tokens through `set_sampling`, top-20
  logprobs and QSA pages as the executor's main KV pages. Retained-lane support is described
  below, together with its shared host tiers. `LoadedModel` loads from the registry's opened `Reader` with load progress and keeps the
  frontend resources. Real Engine test: `ninfer_qwen3_8_flash_next_engine_real_test` (C=4 against
  solo, KV-capacity waits, prefill interruption, decode and mid-prefill cancellation) and
  `..._program_parity_real_test`; serve smoke `tools/smoke/serve_contract.py --text-only`.
  Verified 2026-11-15 (RTX 5090, f3-fp8-nvfp4, NVFP4 QSA state): Engine greedy output equals the
  Program's own for 64 tokens after a 17,000-token prompt; C=4 against solo passes the logprob
  criteria (equal first token, close logprobs, near-tie at the first divergence; batched rounds
  are not bitwise B=1); KV-capacity waits, prefill interruption and both cancellations are exact;
  the OpenAI/Responses/Anthropic smoke (streaming and a grammar-constrained tool call) passes at
  `--max-concurrency 4 --kv-capacity 500000` and at `12 / 600000`, where twelve concurrent chat
  requests decode in batches of twelve.
- Order: frontend acceptance; paged lane-batched QSA Ops; lane-batched Forward (`prefill_chunk`,
  `decode(lanes)`); sampling round (per-lane config, token masks, logprobs); Program contract and
  capacity curve; registry/Engine/options/docs; Engine tests (greedy parity, C=4 vs solo, small
  kv_capacity waits, cancellation, serve contract smoke); then serve_bench at C=1/2/4/8; decode
  graphs per batch size with frontier-bucketed QSA profiles, prefill streaming window, lane
  retention for multi-turn agents.
- Risks: prefill chunks stream
  most experts over PCIe (~1 s per wide chunk) and stall other lanes' decode without a mixed
  forward; QSA paging touches tuned kernels; NVFP4 QSA state needs its needle battery.

## Retained lanes and KV tiers (2026-11-15)

Retained lanes use the executor's existing single-owner cache contract. A completed ordinary
request keeps its Program lane, QSA page mapping, exact FP32 GDN recurrent/BF16 convolution
state, BF16 PLE history and committed tokens. The last sampled token is not appended, so the
current reusable frontier excludes it. Reuse resets sampling counts, masks and RNG configuration;
raw n-gram history is reconstructed exactly from the prefix's last two IDs. An exact hit samples
from the retained BF16 vocabulary distribution rather than re-executing the last token.

A startup-prefaulted pinned slab holds two exact continuation images per lane: the frontend's
typed rewrite boundary and the occupied frontier before a new suffix. Each image contains GDN,
PLE and tail logits. Captures happen at their exact execution boundary, before output advances
state; QSA remains a prefix of the same owned pages. The adapter processes any internal checkpoint
segments within one scheduling super-chunk, so they introduce no interruption boundary. No
arbitrary longest-common-prefix rollback or page-edge approximation is allowed. Cancellation of
suffix prefill restores a complete saved frontier; cancellation after decode retains the committed
input but never the unaccepted sampled output.

On completion the lane releases growth reservation and retains only mapped prefix pages. Shared
pool pressure and lane replacement prefer empty lanes, then disposable retention, then the least
recently admitted ordinary lane. `cache_write: false` forks pages and continuation into independent
allocations without consuming its source, captures no new checkpoints and cannot persist itself.
With RAM off, an original that cannot coexist with its fork causes bounded overload.

Implementation and focused real-artifact verification use `out/sol-lanes`
(Release, CUDA 13.1, `sm_120a`, RTX 5090 / PCIe Gen5). The focused `State` and `PleRowFeeder`
tests pass: cross-lane continuation transport is byte-identical, independent copies preserve
continuation bytes, shrinking unmaps released pages and claiming preserves surviving page IDs.
The seven changed translation units pass clang-tidy with zero diagnostics and build with
`-Werror`. Read-only review findings about stale mappings, checkpoint scheduling boundaries and
service projection were corrected before GPU qualification.

The `--reuse-parity` qualification passes the cold-route floor criterion. Two synthetic prompts
(seeds 73 and 901), each split at 63 / 64 / 65 / 4097 / 16385 with a 71-token suffix and four
teacher-forced decode steps, yield 50 checked distributions. An independent cold route starts
empty and shifts its execution boundary to the split without completing a request or calling
reuse/restore. Against default cold prefill, this route gives **45/50 top-1 agreement**, KL median
**0.00246848**, p95 **0.0113539**, maximum **0.0359098**. Direct retained/split and checkpoint
restore produce **byte-identical logits to this independently shifted cold route at all 50
positions**; continuation images also restore byte-identically. Thus the 63/65-token differences
are the qualified chunk-partition numerical profile, not an additional retention error. The gate
checks top-1/KL against that measured floor, not exact top-1 at every default-cold position.
The run takes 62.49 seconds on RTX 5090 / PCIe Gen5, NVFP4 KV, Release CUDA 13.1 `sm_120a`.

The Engine lifecycle checks pass: exact hits, cached-token counts and suffix-only compute,
output-tail exclusion, source-preserving disposable forks, disposable eviction, decode
cancellation, cancelled suffix rollback, four interleaved retained continuations and eviction
under shared-pool pressure. A cold/split diagnostic first differs at the layer-0 GDN mixer;
64-token partitioning is exact, while nonaligned partitioning follows the measured route floor.
No Ops change is required by this result.

One C=1 agent-transcript replay (`test_engine_real --conversation-bench`,
`NINFER_FLASH_NEXT_BENCH_TOKENS=/src/out/flashnext/long_tokens.i32`, same process, cold pass
followed by reuse pass, one output token per turn, NVFP4 KV, 55,000-token pool) gives:

| Turn | Prompt tokens | Reused tokens | Cold TTFT (s) | Reuse TTFT (s) |
|---|---|---|---|---|
| 1 | 8192 | 0 | 1.989 | 1.659 |
| 2 | 16384 | 8192 | 2.656 | 1.679 |
| 3 | 32768 | 16384 | 5.036 | 2.530 |
| 4 | 50000 | 32768 | 8.665 | 3.613 |

This is a single representative replay, not a numerical qualification or a statistically isolated
cache-warmth A/B.

The shared RAM and inclusive SSD tiers carry named Flash-Next continuation images rather than
relabeling PLE as family hidden/GDN state. The target owns the exact image layout: all GDN
convolution/recurrent slots, PLE history and tail logits, **116,139,008 bytes** per image. QSA
pages use the existing canonical page transport. Current, rewrite and occupy rollback images
restore byte-identically in RAM and after reopening raw and Zstd disk stores at real model
geometry, with disjoint source/destination page mappings. Disk format v9 binds the image byte
extent, artifact tensor contents, QSA selection budget and the external n-gram table's complete
SHA-256 identity. Disk-enabled startup reads those tensor/table bytes to establish identity.

Real Engine checks pass for RAM and SSD: a saved occupy frontier restores on a fresh lane,
exact-hit tokens and subsequent suffix/decode equal the direct retained route, and one-lane
disposable reuse secures the original without consuming it. Nonfinal system-prefix captures
include their own tail distribution and exact system-frontier hits agree with cold execution.
RAM-on requests use funded output windows and secured live continuation images, excluded from
prefix lookup and disk. Cross-lane live resume preserves all 16 sampled output IDs with presence
and frequency penalties, including both in-place and relocated growth of the parked credit;
windows of 4, 12 and 16 tokens exercise the same accounting used for the 8192-token product cap.
The parked counts offset stays fixed when its credit grows or relocates.

A six-identity transcript replay through C=4 with an 8192 MiB RAM arena, 220,000-token NVFP4 pool,
one output token per turn, and the same fixed transcript as above gives mean end-to-end TTFT
(including FIFO waiting). The six identities differ at the first token and each grows unchanged
through 8K to 50K. Every identity after turn 1 restores its previous complete prefix from RAM.
Cold runs precede reuse runs in one Engine; this is one representative replay, not an isolated
cache-warmth experiment. Total run wall time is 211.50 seconds.

| Turn | Tokens per identity | Total reused tokens | RAM hits | Cold mean TTFT (s) | Reuse mean TTFT (s) |
|---|---|---|---|---|---|
| 1 | 8192 | 0 | 0/6 | 6.09943 | 5.83806 |
| 2 | 16384 | 49152 | 6/6 | 8.89488 | 5.96833 |
| 3 | 32768 | 98304 | 6/6 | 17.5465 | 9.06513 |
| 4 | 50000 | 196608 | 6/6 | 30.2631 | 12.8225 |

## DFlash2 design (2026-11-14)

The Flash-Next drafter (`/home/battlefront/ai/qwen3.8-flash-next-dflash2`) is the 27B DFlash2
network at half width: 5 sliding-window layers (window 2048), hidden 2560, intermediate 7680,
32 query / 8 KV heads of 128, block 8 (K <= 7), mask token 248077, rope theta 1e4, target
layers [5,15,25,35,45], `fc` [2560,12800], the same 256-rank selector codebooks [248320,256]; it
uses the target's embedding and head. Its features are the plain mean of the four HC streams
after each target layer's MoE inject (not an HC read).

- Reused unchanged: path-select score/top-k/walk, masked/ragged prefix preparation, the
  speculative prepare/accept/select Ops and p-less calibration, SWA, rope, rmsnorm,
  `gated_delta_net_replay_record`. New admissions: NVFP4 A16 Linear at the drafter shapes,
  `grouped_dynamic_conv` at D=2560, the path-select projection [256,2560], FP8
  `gdn_input_proj_conv_record` [16384,2560], 36-layer GDN replay fold. New Ops: stream-mean
  feature capture fused into the MoE inject, PLE record/commit. The schedule is target-owned.
- Verify is chain-only (W = K+1; QSA stores slot = position, so tree siblings collide). QSA needs
  no rollback (rejected slots lie beyond the frontier and are overwritten; a block sealed through
  a rejected slot is resealed when its last slot is re-appended). Drafted tokens need n-gram rows
  before layer 1, so a round is graph A (context append, propose, drafts to host, verify
  embedding + layer 0), a host row fetch, then graph B (layers 1..47, head, accept, commits).
- Expert cost: replaying the split-LRU cache simulation with verify windows, expert pairs per
  step grow 1.64x at W=2, 2.69x at W=4, 4.37x at W=8; PCIe per committed token is 1.9-2.8 ms at
  W=4 (3 committed) and 1.6-3.2 ms at W=8 (4 committed) against 2.2 ms for plain decode, because
  rejected drafts route to their own experts. Expected gain about 1.0-1.5x at C=1, mainly from
  amortizing dense reads; an adaptive arm picker on measured round time is required, with smaller
  K at higher concurrency.
- Measured acceptance (study harness, `study/results/dflash/REPORT.md`, 837 teacher-forced
  rounds per arm on the mixed raw-document panel): conditional per-hop acceptance is 69-74% at
  every hop; mean committed length (greedy / sampled) on the recipe target with the NVFP4 drafter
  is 1.70 / 1.69 at K=1, 2.22 / 2.20 at K=2, 2.59 / 2.53 at K=3 and 3.44 / 3.30 at K=7. The
  recipe and the NVFP4 drafter each cost under 2% of committed length. By source (BF16 greedy,
  K=3), code reaches 3.35 and chat 2.20; the checkpoint README's 2.83-3.26 at K=3 are sampled
  completions of chat/math/code prompts, so serving acceptance on agent and code turns is
  expected above this panel. Replaying every verify column through the corrected split-pool
  simulator (strict capacity, fill on miss, 26.8 GB/s) and modeling dense time at the roof gives
  C=1 160 tok/s at K=2 against 147 autoregressive at 19.95 GB (190 against 165 at 23.4 GB),
  9-15%; at C=4 every speculative arm loses to autoregressive decode. Decision: DFlash2 is built
  after the decode kernel time and batched lanes, with an adaptive picker on measured round
  time over K in {0, 1, 2, 3} and K=0 as the C>=4 default; serving acceptance is measured in the
  engine before K is tuned further.

### Native DFlash2 qualification status

The append converter and optional binder support a complete 66-tensor companion (32 NVFP4
matrices, 34 BF16 tensors), preserving base payloads. See the
[artifact contract](qwen3.8-flash-next-artifact.md#10-optional-dflash2-companion). CLI/server startup
enables adaptive K0..3 when that companion is present. The native drafter uses its five layers,
independent committed-context projections and full shared FP8 head. Graph A appends pending
context, proposes, exports IDs and starts target layer zero; host n-gram reads overlap that layer.
Graph B finishes verification and acceptance. After Engine output preview, authoritative prefix
counts fold the 36 GDN layers and commit only published PLE/history/features/head state. This
avoids copying a full speculative target state before each round. A K0 root is a width-one chain
record: `gdn_input_proj_conv_record` and `gated_delta_net_replay_record` read the canonical GDN
pools read-only and write width-one records, with the width-one snapshot's arithmetic, and use
Decode-purpose sampling with deferred publication. No extra model column is evaluated and no
GDN state pool is copied (see [K0 width-one record](#k0-width-one-record-2026-11-16)).
Packed verification passes the same-column numerical-floor gate; cross-width greedy identity is not a
contract, just as [27B verification differs by K](../performance.md).

The target-owned chain-recording path uses read-only GDN convolution/recurrent sources and PLE
snapshots in independent storage. Its startup-fixed `ForwardOptions::max_verify_width` is zero
by default. `ninfer_qwen3_8_flash_next_engine_real_test --verify-parity` evaluates W=2,3,4 at C=1
against ordinary width-one decode, using the same represented weights and NVFP4 KV. Two
in-vocabulary deterministic prompts (seeds 73 and 901, lengths 1024 and 4097) each contribute 256
ordinary autoregressive steps; packed chains are teacher-forced on those sampled IDs, so
rejection cannot hide a target-distribution mismatch.

RTX 5090 / PCIe Gen5 / CUDA 13.1, Release `sm_120a`, companion artifact
`qwen3.8-flash-next-f3-fp8-nvfp4-dflash.ninfer`, per-lane context 8192, 11-lane diagnostic Program
with 90,112 shared NVFP4 KV tokens. Separate lanes hold independent width-one checkpoints for
B=W lane-batched decode and ordinary W-wide prefill. Both references consume exactly the same
teacher-forced columns as verification, never its records. The empirical floor is the envelope
of those valid routes: verification's mismatch count and KL median/p95/p99/maximum must not
exceed the respective measured-floor values (KL evaluation allowance `1e-12`). There is no
statistical confidence claim about these correlated fixed-panel observations.

| Width | Verify and W-prefill top-1 | B=W decode top-1 | Verify and W-prefill KL median / p95 / p99 / maximum |
|---|---:|---:|---|
| 2 | 255 / 256 | 254 / 256 | 0.000004559 / 0.003890 / 0.012943 / 0.145681 |
| 3 | 380 / 384 | 381 / 384 | 0.000014176 / 0.005183 / 0.034562 / 0.156775 |
| 4 | 506 / 512 | 509 / 512 | 0.000016349 / 0.007720 / 0.036512 / 0.190148 |

The same-width ordinary-prefill reference reproduces the verification statistics at every
reported quantile, including the tails. B=W decode alone is a narrower floor at W=3/4; that
comparison omits the accumulated within-window recurrent arithmetic.

The complete 116,139,008-byte canonical continuation is unchanged at both initial frontiers for
every width. All 512 subsequent ordinary sampled IDs equal the independent ordinary run after
provisional verification; the n-gram feeder's repeated-proposal and EOS tests also pass.

Exact contracts remain separate: rejected verification preserves canonical state byte-for-byte;
accepted commits equal non-speculative execution with the same width; picker-forced K equals
fixed K in output. Native Program tests cover fixed/forced K0..3, greedy, standard and p-less
sampling, exact token-count histograms, partial publication, cyclic wrap, and zero publication
after pending context and root-only fallback. The cyclic backup gathers 128-element head/slot
rows from physical `[128,2048,8,lanes]` storage, never a fictitious contiguous 1024-element slot.

The complete continuation is 116,139,008 bytes without speculation and 158,184,704 bytes with
DFlash2: cyclic KV, four pending feature columns and 256 bytes of explicit calibration metadata
extend the base image. RAM and reopened raw/Zstd SSD transport current/rewrite/rollback images
and QSA pages byte-for-byte. Engine tests cover fixed/adaptive streamed logprobs, caller stops,
cancellation, retained lanes, RAM/SSD reuse and source-preserving disposable turns. Live RAM
suspension preserves standard and p-less sampled IDs, calibration, counts and cumulative request
statistics through physical-lane changes and in-place/relocated credit growth. New requests
reset calibration and statistics; live Decoding restore preserves them. Context checkpoints
remain unsupported even when DFlash2 is active.

The native sampled study-panel check runs all 27 sequences and 31 frontiers each, K3, T=1,
top-k 20, top-p .95, with verifier-selected anchors and NVFP4 KV. Its 837 realized accept walks
reach/accept `837/590`, `590/403`, `403/283`: conditional acceptance **70.49%, 68.31%, 70.22%**,
mean L **2.52449**. The study's integrated case-c values are **70.7%, 67.5%, 71.9%**, L **2.527**.
These are descriptive comparisons, not equal draws or confidence bounds: the study integrates
accept uniforms and uses BF16 target KV/its own arithmetic profile. Native default NVFP4 KV and
represented artifact weights are used here. The study's speed estimates are not native results;
[native Engine/HTTP measurements](../performance.md#flash-next-native-dflash2-gen5-2026-11-15)
cover every fixed arm and adaptive selection at C=1,2,3,4,6,8, with Strata reference values and
separate RAM conversation TTFT. The base artifact is faster in the measured C6/C8 HTTP cases.

The drafter owns initialized proposal-residual and physical-head transfer panels (1,510,400
bytes per configured lane), separate from reused context scratch. CUDA 13.1 initcheck can flag
large captured memcpy sources despite an ordered producer; an exact Cast/crop reproducer
reports errors only in the large graph route, not eager execution. No sanitizer check is disabled
and no input or arithmetic workspace is zeroed to hide a read. Focused native initcheck at C1/C2
passed K3 with one-/64-token prefixes and K0 through sixteen outputs, before the K0 width-one
record route; that route removed the strided gate copy whose false initcheck flags had required a
persistent gate-record initialization, and the initialization with it. Ordinary/deferred B2 K0
state is compared byte-for-byte at two consecutive roots, including record reuse. Full native
Program memcheck reported zero errors; the longer initcheck panel was not repeated.

### Target-side decode performance experiments (2026-11-15)

No target runtime optimization from this panel is retained: the measured candidates did not
show a speedup in the priority screens. Pre-fill grouping still needs an Op-owner change; the
low-memory K0 fix is the [width-one record route](#k0-width-one-record-2026-11-16). Native
fixed-K verification-route tracing and exact cache-policy replay are retained as diagnostic
tools; production execution is unchanged by that opt-in instrumentation.

RTX 5090, PCIe Gen5 x16, 600 W, CUDA 13.1, Release `sm_120a`, container
`ninfer-builder-flashnext`, build `out/sol-lanes`. Public Engine `lanes-bench` uses
`qwen3.8-flash-next-f3-fp8-nvfp4-dflash.ninfer`, the external int8 n-gram table,
`out/flashnext/long_tokens.i32`, 4096-token prompts, 512 greedy outputs/request, NVFP4 KV,
automatic capacity and **fixed `--context 8192`**. The measured interval excludes initial prefill
and includes request drain. The grid runs C in ascending order in one process; priority screens
run only their listed C in fresh processes. Build/GPU locks, RAM and five-second idle checks apply.

The initial frozen K0 reference at C=1/2/3/4/5/6/8 was
120.69/140.48/177.68/198.11/169.54/188.38/194.56 tok/s. It differs in arm and Op generation
from the K3 integration measurements below and is not their before/after comparator.

| C | K3 with bf16/cache/HC updates, tok/s | K3 after moe_experts grid update, tok/s |
|---:|---:|---:|
| 1 | 181.82 | 179.59 |
| 2 | 252.76 | 255.90 |
| 3 | 269.41 | 275.75 |
| 4 | 281.52 | 289.91 |
| **5** | **216.60** | **241.44** |
| 6 | 227.95 | 235.83 |
| 8 | 207.05 | 227.15 |

Generated IDs match exactly between these columns at every C. The measured C5 gain from the
moe_experts integration is **11.47%**; C1 is **1.23% lower**, so this is not a universal speedup
claim. The separate fresh C5 K3 screens measured 250.69 with the frozen initial binary, 229.31
after bf16/cache integration, and 225.20 after HC integration. Aggregate per-lane round/fallback
counts changed from 787 to 845 to 882, so those earlier end-to-end differences cannot be attributed
to kernel latency alone. No other owner's changes were reverted.

- **Picker rejected.** Shared per-hop conditional acceptance, cumulative-product expected lengths,
  slow per-arm GPU/gather/constant-host timing, content-adaptive decay, 3% switching hysteresis
  and stale-only probes measured C5 **213.66**, versus **216.85** for the existing picker. The
  same initial Op generation's fixed K1/K2/K3 measured **212.50/233.11/250.69**. The candidate was
  removed; neither a within-1% result nor a complete post-integration best-arm grid is qualified.
  A K3 prefix is not an exact fixed-shorter-K counterfactual: the drafter's symmetric non-causal
  SWA sees every live proposal column, so changing width can change the proposals themselves.
- **Commit suffix rejected.** Captured selection/scatter with asynchronous return versus matched
  eager/synchronous control measured C5 K3 **224.75/225.20** and C1 K0 **115.78/116.02** tok/s.
  Generated IDs were identical and native Program exact-state cases passed. Full fold capture
  requires device lane/count inputs: capturing the current by-value rows would freeze publication.
- **Inline fill rejected.** T1 fill/publish followed by one nondeferred expert pass measured C1
  K0 **115.25**, versus **116.02** for the matched concurrent-fill control, with identical IDs.
  Dynamic hit-only skipping still needs an Op seam or graph conditional support.
- **Admission unchanged.** Actual W4 traces contain 512 native rounds at C1 and C5. Replay starts
  from exact owners, references, hands, frequency counts and halving progress, and reproduces every
  recorded counter bucket. C1 every-miss / full frequency filter / root-only every-miss costs
  **165.7/221.7/323.4 MB misses per round**; C5 existing full filter / root-only filter costs
  **2312.6/2405.5 MB**. Variants also lose over the first 128/256 rounds. No projected tok/s is
  used for this decision. `lanes-trace --spec dflash --draft-tokens 3` exports packed
  `[rounds,48,C,4,10]` routes and frequency snapshots; `replay_cache.py --root-only` changes only
  admission eligibility, preserving all-column lookups/protection. `--check-policy` cannot silently
  skip a requested counter check; borrowed slots remain protected.

K0 cannot simply use ordinary in-place decode while preserving zero-publication/rejection bytes;
the read-only width-one record route below removes its full-pool copies without doubling GDN
state. Grouping is private to moe_experts and cannot be moved ahead of the fill through its current
completed-fill event interface. The drafter's column top-16 has since been rebuilt; see
[Drafter column top-16](#drafter-column-top-16-2026-11-16). Details are in
`out/sol/target-perf-op-handoff.md` and the measured backlog dispositions.

Qualification after integration: BF16 reference KL **0.00452**, top-1 **98.24%**; split **32/32**
(KL median `3.46203e-5`, max `0.0091751`); graph/eager B2 **128/128** and B5 **320/320** columns
bit-identical; single-token prefill/decode **128/128** exact. Native Program **71.29 s**, Engine
**95.62 s**, and W2/3/4 same-column verify-floor qualification **74.23 s** all exit zero. The
verify result matches the independently executed same-width prefill at every reported KL quantile.
Builds are `-Werror`; focused clang-tidy reports zero diagnostics. Python 3.11 compilation, Ruff,
formatting and spelling checks pass. Failed runtime candidates were screened at C5/C1 rather than
expanded into full per-candidate grids; no new sanitizer or universal speedup claim is made.

### K0 width-one record (2026-11-16)

A K0 root used to emulate a read-only width-one record: per GDN layer it copied the whole
convolution and recurrent pools of every configured lane into private scratch, ran the in-place
snapshot Ops there and copied the projection, key, value and gates into width-two records. That
moved about 113 MB of recurrent state per configured lane per round, read and written, and the
snapshot wrote it once more (about 340 MB of traffic per lane).

The record Ops already share the snapshot's arithmetic: at the [16384,2560] parents the record and
snapshot run the same A16 projection over the W*B packed columns and the same convolution kernel,
and `gated_delta_net_replay_record` runs the snapshot's recurrent body without its state store.
The Op domains therefore admit width-one chains (no kernel changed): `gdn_input_proj_conv_record`
[16384,2560] T=1..16 for chains (trees T>=2), `gated_delta_net_replay_record` T=1..16 for chains
(trees and the 27B accepted-history profile keep T>=2), and `gdn_replay_fold` record width 1..16.
The target sends every verify width, including K0, through the Record branch; records have the
live width, the fold reads `verify_records(width)`, and the `root_*` scratch (3,268,608 bytes per
configured lane), its strided gate copy and the gate-record initialization that only served that
copy are removed. The canonical GDN pools are read-only until the authoritative fold, so
rejection and zero publication keep every canonical byte. The PLE history copy (184,320 bytes per
lane) and the sampling-count snapshot (993,280 bytes per lane) remain: the first needs a
read-only source seam in `ple_inject_snapshot`, the second restores truncated publication.

Verification (RTX 5090 Gen5, CUDA 13.1, Release `sm_120a`, build `out/opus-k0`, matched base
`out/opus-k0-base` = the same tree with the copy route restored; logs `out/opus-k0-logs`):
- Op tests: replay record now checks every case against the FP64 recurrence (W=1 dense B=1/12,
  masked B=5) and bit for bit against the same-shape snapshot; conv record W=1 B=1/5/12 against
  the FP64 oracle and bit for bit against the same-shape snapshot outputs and newest history tap,
  FP8 and NVFP4; fold W=1 B=1/5/12 with accepted and rejected roots. The fold test's FP64 oracle
  condition had never matched a registered geometry; it now runs whenever row 0 commits.
  memcheck (all three) and racecheck (record, fold) report zero errors.
- Real artifact: DFlash Program (including the two-row K0 root byte-identity against ordinary
  decode and rejection preserving the complete continuation), DFlash Engine, retained, RAM and
  SSD tiers pass; verify floor W=2/3/4 passes; lanes-check graph/eager B2 4096/4096 and B5
  10240/10240 bit-identical; BF16 parity log-probabilities and split results (32/32, max KL
  0.0091751) byte-identical to the base. `--dflash-live-tiers`, the default C=4 suite and
  `--program-parity` fail with the same first difference as before the change.
- Fixed K0 and fixed K3 lanes-bench tokens are identical to the base at C=1/4/5/6.

Public Engine lanes-bench, DFlash companion, 4096-token prompts, 512 greedy outputs per request,
`--context 8192`, NVFP4 KV, C=1,4,5,6 in one process; two repetitions in opposite build order.
ms/round is the decode interval over lane rounds / C.

| Arm | C | Base tok/s (rep 1 / 2) | After tok/s (rep 1 / 2) | ms/round (mean) | Mean change |
|---|---:|---|---|---|---:|
| K0 | 1 | 117.3 / 117.1 | 124.3 / 118.3 | 8.53 -> 8.25 | +3.5% |
| K0 | 4 | 241.2 / 228.2 | 244.0 / 240.1 | 17.06 -> 16.53 | +3.1% |
| **K0** | **5** | **217.9 / 203.7** | **225.7 / 211.4** | **23.75 -> 22.91** | **+3.7%** |
| K0 | 6 | 224.6 / 219.6 | 219.6 / 220.7 | 27.02 -> 27.26 | -0.9% |
| Adaptive | 1 | 165.3 / 158.8 | 170.4 / 160.9 | 11.60 -> 12.18 | +2.2% |
| Adaptive | 4 | 309.1 / 262.8 | 292.3 / 271.3 | 33.95 -> 36.94 | -1.5% |
| Adaptive | 5 | 253.9 / 235.4 | 254.2 / 224.9 | 48.57 -> 50.30 | -2.1% |
| Adaptive | 6 | 269.2 / 179.6 | 236.4 / 247.7 | 63.70 -> 56.86 | +7.9% |
| K3 | 1 | 182.0 / 179.9 | 172.2 / 179.8 | 14.19 -> 14.60 | -2.7% |
| K3 | 4 | 275.0 / 206.3 | 289.9 / 278.9 | 50.11 -> 41.55 | +18.2% |
| K3 | 5 | 223.4 / 198.1 | 238.3 / 223.1 | 68.98 -> 62.85 | +9.5% |
| K3 | 6 | 226.2 / 192.1 | 239.4 / 228.5 | 87.93 -> 78.14 | +11.8% |

Fixed K3 executes no changed code (identical tokens); its spread, including the disturbed second
base repetition, is the host's run-to-run noise, so no K3 regression or gain is claimed. K0 moved
in the same direction in both repetitions at C=1/4/5, in line with the removed traffic (about
0.95 ms of a 23 ms C=5 round); the C=6 difference is within the noise. Adaptive arm sequences
follow wall-clock picker timing, so their outputs and arm mixes differ and the adaptive rows are
not a controlled comparison.

Fully captured commit: not built. nsys of forced-K0 rounds (128 per run) puts the eager commit
window (commit-ingress upload, fold, selects, scatters) at C=1 / C=5 at 155 / 788 us median span
with 134 / 765 us GPU busy (fold 108 / 733 us, about 2.1 TB/s of state traffic); the 21 / 23 us
idle is the most a single captured commit graph could remove, about 0.2% / 0.1% of a round, below
run-to-run spread. A device-metadata `gdn_replay_fold` form would have no measurable benefit and
was not added. The fold itself is now the K0 root's main remaining cost over in-place decode.

### Drafter column top-16 (2026-11-16)

`dflash2_path_select` / `dflash2_tree_select` take an exact top-16 of every drafter logit column
before the Markov walk. Flash-Next has no shortlist head, so each column is the full V=248320;
the previous fixed 32-split route gave 7760 rows per CTA, kept a sorted private top-16 per thread
in local memory and merged serially, and cost ~213 us per C=1 K3 round (latency bound; the
byte floor of 3 columns is under 1 us). The 64/128-split variants of that route were measured
first: 64 splits cut the Flash-Next Op 3.6-4.4x but slowed the 27B shortlist Op 24-46%; 128
splits were slower than 64 everywhere. Neither was kept.

Current design (Op header and `src/ops/kernel/dflash2_path_select.cuh`): every BF16 logit maps
to a unique 64-bit key, the value's 16-bit order key (with -0 folded onto +0) above the
complemented row, plus a sign-of-zero bit; NaN/+-inf map to 0. Unsigned key order is the Op's
rank order, so lower-row tie-breaking and finite-value exclusion are properties of the key. A
split CTA (256 threads, 16 keys per thread from two coalesced 16-byte loads, 4096 rows, 61
splits at V=248320) takes the 16th-greatest per-thread maximum as a threshold (warp bitonic sort,
then a three-level bitonic merge of the eight warp lists that every warp computes), compacts the
at most 256 keys at or above it, and ranks them by counting. A one-CTA-per-column merge repeats
this over the split keys and decodes values bit-exactly (including -0) and rows (through
`logit_token_ids`), filling ranks without a finite logit with -inf and row = rank as before.
Two barriers per CTA; no sort network over the vocabulary. V is now bounded to 262144 rows.
Eight keys per thread (twice the split CTAs) measured equal or slower and was not kept.

Qualification: the column top-16 is checked bit for bit (values and ids) against an independent
host sort at V=248320 for C*K columns 1, 3, 8 and 36, the 27B shortlist V=131072 with its row map,
V=262144, odd V and a misaligned base (row-by-row loads), V=16/100, and adversarial columns
(cross-split ties, constant, NaN payloads and +-inf with +-max finite at split/column edges,
interleaved -0/+0 leaders, subnormals, fewer than 16 finite rows with and without the row map,
all non-finite, and per-thread plateaus that reach the 256-candidate bound). A public-Op case at
V=248320, T=3, B=5 checks the selector ids exactly and q against the FP64 softmax of the oracle
top-16. A sign-of-zero mutation fails the test. compute-sanitizer memcheck and racecheck: 0
errors/hazards. clang-tidy: 0 findings.

Public Op (`ninfer_dflash2_path_select_bench`, one captured Op call with hot logits, BF16 W_h;
includes the unchanged projection and walk), median us before -> after:

| Head | Shape | Before | After |
|---|---|---:|---:|
| Flash-Next V=248320 D=2560 | T=1 B=1 | 230.7 | 15.8 |
| | T=3 B=1 (C=1 K3) | 242.0 | 26.5 |
| | T=2 B=4 | 306.0 | 25.6 |
| | T=3 B=5 (C=5 K3) | 554.9 | 35.9 |
| | T=3 B=12 | 1090.0 | 51.7 |
| 27B shortlist V=131072 D=5120 | T=7 B=1 | 87.9 | 45.2 |
| | T=3 B=12 | 176.2 | 53.6 |
| 27B full head V=248320 D=5120 | T=7 B=1 | 317.5 | 49.5 |

nsys of the after bench: split + merge are 3.6 + 2.0 us at T=3 B=1 and 7.2 + 2.0 us at T=3 B=5.
The walk kernel (`dflash2_path_select_kernel`, ~16 us) is now the largest part of the Op.

Round level: fixed-K3 lanes-bench (DFlash companion, 4096-token prompts, 512 greedy outputs per
request, context 8192, NVFP4 KV, C=1 and C=5 in one process), matched builds `out/opus-topk-base`
/ `out/opus-topk-after` (same snapshot, only the Op files differ), four repetitions in alternating
order (logs `out/opus-topk-logs`). Generated tokens are identical in every pair. C=1: base
178.3/167.9/177.7/168.8, after 169.2/179.4/182.0/181.6 tok/s (mean +2.8%); C=5: base
218.5/181.7/220.8/175.1, after 223.9/224.5/225.3/227.5 tok/s (+2.6% against the mean of the two
undisturbed base runs). The expected saving is the Op delta, about 0.22 ms of a ~14.4 ms C=1 round
(1.5%) and 0.52 ms of a ~65 ms C=5 batch round (0.8%); both measured differences are within this
host's run-to-run spread, so only the direction is claimed at the round level.

### Narrow drafter NVFP4 A16 K-warp split (2026-11-16)

The five narrow Flash-Next drafter Linear geometries use sixteen-output-row CTAs with eight
K64 warps through T=16. Two shared staging buffers overlap the next K512 group with BF16 MMA;
E2M1 codes times stored E4M3 scales widen exactly to BF16, with FP32 partials and reduction.
Packed sequences use passes of at most sixteen same-reduction columns to preserve independent
C=1 outputs. T=1, wider T, gate/up and all 27B geometries retain their existing routes.
`tools.kdev recipe --idea profiled_new_family --candidate-atom bf16 --operand-terms 1`
admitted the measured narrow points. A single-buffer candidate did not improve and was discarded.

Public Linear cold-weight medians, microseconds (RTX 5090, PCIe Gen5, CUDA 13.1, sm_120a;
150 calls per point, baseline/after/after/baseline, mean of the two medians):

| N,K | T=4 before → after | T=15 before → after |
|---|---:|---:|
| 2560,12800 feature | 32.768 → 22.528 | 34.816 → 29.488 |
| 6144,2560 QKV | 12.288 → 11.264 | 16.384 → 16.384 |
| 2560,4096 attention output | 14.336 → 10.240 | 14.336 → 12.288 |
| 2560,7680 down | 21.504 → 15.968 | 22.528 → 18.432 |
| 640,2560 conv projection | 8.192 → 6.144 | 8.192 → 8.192 |

A T=2..16 cutoff sweep found no slower median in the selected domain. The complete NVFP4 A16
Linear suite passes against the independent naive FP64 dot-product oracle, including the 27B
geometries, masked tile boundaries, graph replay and exact packed-versus-panel checks. Focused
memcheck: 0 errors; racecheck: 0 hazards; clang-tidy: 0 findings; build uses `-Werror`.
Evidence: `out/sol/small-item2-buffered-ab.log`, `small-item2-cutoff.log` and
`small-item2-buffered-oracle.log`. These are Op-level gains; no round-throughput gain is claimed
or measured because the expected round saving is below the established lanes-bench spread.

## N-gram read budget

Random 4 KiB O_DIRECT reads of the 52 GB table on the boot SSD (SN850X, ext4) measured about
101K pages/s at 8 threads and 193K at 16 (178K at 32), 2026-11-14. A prompt needs at most 16
rows per token (fewer after the row cache), so a 4K-token prompt costs about 0.3 s of reads.
Rows depend only on token ids, so the reader fetches chunk k+1's rows while chunk k computes;
only the first chunk's reads (about 40 ms for 512 tokens) are exposed. Decode reads 16 rows per
token (well under 1 ms at queue depth 16) before layer 1 of the step.

## Expert residency design (step 9)

- **Always resident.** Everything every token uses stays in VRAM for the life of the model and
  never enters the expert cache: head, attention/GDN/QSA projections and controls,
  hyper-connections, PLE projections, the router and shared-expert gate, and all 48 shared
  experts (the token embedding is a per-token row gather and lives in mapped host memory).
  Shared experts are NVFP4 with GPTQ and MSE scales in the chosen recipe (133 MB, about
  0.08 ms of reads per decode token; round-to-nearest NVFP4 alone cost KL 0.050, GPTQ+MSE
  0.039 as a family). The routed kernel takes the shared expert as one NVFP4 record in the routed record layout
  (`shared_record` + `shared_matrix_scales`), not through the table.
- **Host store.** All 24,576 routed experts live in pinned host memory (at most 50 GB), one
  record per (layer, expert) holding gate/up and down codes and scales (NVFP4 2,764,800 bytes,
  F3 1,996,800 bytes), loaded once from the artifact into `HostPinnedArena` chunks and
  device-mapped (UVA); the FP32 matrix scales live in a resident table.
- **Device cache.** One VRAM pool of two record size classes, sized at startup from free memory
  after weights, KV and workspaces, plus a device table `expert_ptr[48][512]` (U64: the record's
  VRAM slot or mapped host address, codec tag in bit 0). The
  routed kernel reads each selected expert through its pointer without branching: a resident
  expert from its slot, a non-resident one **directly from mapped host memory over PCIe**. A decode or
  verify step therefore never waits for a host round trip on a miss; a miss costs its bytes at
  PCIe rate inside the kernel. Measured 2026-11-14 (pinned `cudaHostAllocMapped` buffer, random
  256-byte-aligned records of 3.3 KB, 8.3 KB and 2 MB, 85 to 1360 CTAs, 1 to 8 loads in flight
  per thread): every configuration reads at 26.8 GB/s, 93% of `cudaMemcpyAsync` (28.9 GB/s), so an
  F3 expert miss (2.0 MB) costs about 75 us and needs no special load scheduling.
- **Replacement: device-side CLOCK in two pools, filled on miss.** Simulated on recorded decode
  routing (`/home/battlefront/ai/study/scripts/simulate_expert_cache.py`, EVAL + HOLDOUT traces,
  43 x 2048 tokens, all 48 layers, 26.8 GB/s): recency beats frequency by far (one request, about
  20 GB: static CALIB top-K 66% hits, decayed frequency 87%, LRU 92-94%). An admitted miss crosses
  PCIe once, copied into its victim slot by `moe_expert_cache_fill`, which `moe_experts` then reads.
  The pools are one per record size (NVFP4, F3); giving NVFP4
  about 11% of the cache bytes (600-1,200 slots is flat-optimal at both sizes) beats both pinning
  every NVFP4 expert and one byte-sized LRU pool:

  | Expert cache | NVFP4 / F3 slots | C=1 hits, PCIe ms per token | C=4 hits, PCIe ms per step (4 tokens) |
  |---|---|---|---|
  | 19.95 GB | 900 / 8,744 | 93.9%, 2.20 | 90.4%, 12.6 |
  | 19.95 GB, every NVFP4 expert (pinning) | 1,583 / 7,799 | 92.8%, 2.57 | 88.4%, 15.0 |
  | 19.95 GB, one byte-sized LRU pool | - | 93.6%, 2.34 | 87.0%, 17.1 |
  | 23.4 GB | 900 / 10,472 | 95.8%, 1.54 | 93.5%, 8.55 |

  Shared KV capacity trades directly against the cache (NVFP4 QSA state about 10.1 KB per token).
  Same simulation, 11% rule, decode without speculation; aggregate assumes about 2.5-3.3 ms of
  per-step dense/attention/launch time on top of PCIe (an estimate until measured):

  | KV / agents | Expert cache | Hits | PCIe ms per step | Aggregate Gen4 / Gen5 (est.) |
  |---|---|---|---|---|
  | 500K / 1 | 21.8 GB | 95.0% | 1.8 | ~230 / ~290 tok/s |
  | 500K / 2 | 21.8 GB | 93.7% | 4.4 | ~280 / ~410 tok/s |
  | 500K / 4 | 21.8 GB | 92.2% | 10.2 | ~300 / ~490 tok/s |
  | 600K / 6 | 20.5 GB | 88.4% | 22.5 | ~230 / ~410 tok/s |

  Six distinct contexts thrash the cache on Gen4, so four agents is the useful limit there; the
  traces are 2K-token windows, so long agent sessions need their own measurement.

  `plan_expert_slots` applies the 11% rule. Victims are chosen on the device right after each
  layer's routing (`moe_expert_cache_admit`: CLOCK per pool, never evicting an expert the layer's
  current call routes to), `moe_expert_cache_fill` copies the admitted misses into their slots on a
  second stream at PCIe rate while `moe_experts` computes the resident experts, `moe_experts` then
  reads the admitted misses from their slots, and `moe_expert_cache_commit` publishes them, so no
  step waits for the host. Every cache byte buys hit
  rate: size the pools from free VRAM after weights, state and workspaces.
- **Prefill.** From about 300 tokens a chunk touches nearly every expert of a layer, so the
  non-resident experts of layer L+1 are copied into a double-buffered streaming window while
  layer L computes, and the table entries for that layer point into the window for the chunk.
  Below that width (verification, small batches) prefill uses the mapped-read path like decode.

## Routed-expert Op (step 10)

`ops::moe_experts` (`include/ninfer/ops/moe_experts.h` is the authority: signature, record byte
layout, precision and graph invariants). y = sum_j w_j down_e(silu(gate_e x) * up_e x)
+ shared_scale * shared(x) for H 2560, I 640, 512 routed experts, top-10 distinct ids per token, a
shared expert given as one VRAM NVFP4 record in the routed layout, T 1..8192; experts are read through a per-layer I64 [512] pointer table (VRAM
slot or mapped host record, codec in bit 0) with FP32 [512,3] matrix scales and per-layer F3
books.

- **Structure.** Four launches (six with a deferral): one-CTA stable grouping (shared-expert jobs
  first, then experts ascending with deferred experts last, jobs of at most C tokens, device-side
  job counts), gate|up with SwiGLU in the
  epilogue (32-row tiles of 16 gate + 16 up rows), down, and an ordered merge (slots in route
  order, shared last, one BF16 rounding; no float atomics). Records are 1200 chunks of 32 rows x
  K128; weights expand exactly to BF16 in registers one chunk ahead, activations stream through a
  cp.async pipeline, BF16 MMA with FP32 accumulation. Token tile C: 16 / 32 / 64 for T up to
  256 / 1024 / above. Cache fills are not part of this Op: with a `MoeExpertsDeferral`, the
  experts whose records `moe_expert_cache_fill` is still copying run after its ready event, from
  their slots, and the rest overlap the copy.
- **Launch structure.** Each matmul launch has min(work-item bound, one resident wave) CTAs
  (340 gate|up, 510 down at C=16); every (job, tile group) item is claimed from a per-launch
  device counter that the grouping kernel zeroes, so the cost follows the real job count and an
  empty deferred range is two near-empty launches (0.8 + 0.9 us) at every T. Claims (not static
  strides) keep the hardware scheduler's balance: a CTA stalled on a mapped record, or started
  late behind a concurrent fill's CTAs, takes fewer items. Down C=16 is held to three CTAs per SM
  by its launch bounds. Measured and rejected (no gain): programmatic dependent launches between
  the Op's kernels; issuing the grouping kernel's global reads before its first barrier.
- **Launch-structure A/B (2026-11-15, Gen5, `ninfer_moe_experts_bench --suite decode`, nsys
  span per layer, lanes x 4 tokens with recorded-trace expert reuse, a deferral with nothing
  deferred).** T = 1 / 4 / 8 / 16 / 20 / 48: 35.4 / 64.4 / 95.5 / 185.9 / 217.7 / 408.4 us
  before, 34.3 / 64.4 / 92.4 / 173.0 / 201.7 / 368.2 after; the empty deferred pair falls from
  2.1 / 3.9 / 6.4 / 11.3 / 13.9 / 31.2 to 1.7 us. Gate|up at T = 4 / 8 costs 3-3.5 us more
  with claims than with one CTA per item (a static-stride variant recovers it but loses 30% with
  mapped records or a concurrent fill); prefill at T=4096 in VRAM is 1.2% slower (2,983 to 3,018
  us), T=512 2.6% faster. Two findings for the caller: (1) resident gate|up time
  at T = 16/20 is PCIe: ten mapped records add 190-250 us at T = 16, their gate|up bytes at the
  48 GB/s mapped-read roof; (2) a concurrent `moe_expert_cache_fill` slows the single-CTA
  grouping kernel from 3 to 40-60 us and the resident gate|up from 108 to 410-450 us at T = 16,
  because the fill's mapped reads occupy every SM's memory pipeline. In the Program this costs
  nothing measurable (next item).
- **Miss schedule (2026-11-16): the layer is PCIe-bound, not schedule-bound.** A native C=5
  fixed-K3 lanes trace (base artifact with DFlash, 4096-token prompts, context 8192, 64 rounds,
  nsys with graph nodes traced, the same run's admit counters) moves 2,259 MB of missed records
  per round (10,469 NVFP4 + 57,901 F3 misses), and the 48 route-to-merge MoE spans of a round
  take 48.2 ms: 46.9 GB/s, the mapped-read roof (48 GB/s; 88% of the 53 GB/s copy engine). The
  rest of the 61.7 ms round is 13.5 ms of dense, attention and drafter work during which PCIe
  idles. About three quarters of the missed bytes are filter-rejected records that the matmuls
  read through mapped memory; the admitted fills (230 us median per layer) move about 11 of a
  layer's 47 MB. Schedules measured, none retained:
  - *Fill on few CTAs* (12 instead of one per SM). Op bench with a tenth of the routed experts
    filled and deferred: T = 1 / 4 / 8 / 16 / 20 / 48 take 57 / 135 / 174 / 430 / 483 / 983 us
    instead of 76 / 155 / 211 / 533 / 621 / 1202 us (the fill alone copies at 44-52 GB/s on 12
    CTAs; 4 CTAs are below the roof, 8-24 within 11%). With the Program's mix (4% of the routed
    experts admitted, 14% mapped) every grid from 6 to 170 CTAs and fill-then-compute land within
    2.5% of each other at T = 4..48 (T = 20: 924-948 us for 45.9 MB, 48-50 GB/s), because the
    mapped reads keep the link busy whatever the fill does. Program, paired per-layer spans at
    T = 20 against the 170-CTA control: 0.95 and 1.01 in two runs; lanes K3 C=4/5/6, best of two
    to four alternating runs, -1.9 / -1.3 / -3.1% ms per round, inside run-to-run noise (single
    runs moved up to 22%). It also slows the windowed 4096-token prefill's MoE from 3.1 to 18.3
    ms per layer: the window fill beside prefill GEMMs needs every SM. Generated tokens matched
    at every C.
  - *Write-through* (the admitted misses read through mapped memory by the matmul CTAs, which
    store the record into the slot; no fill kernel, no deferred pass): equal to plain mapped reads
    and to the 12-CTA fill within run-to-run variation (0-12% at T = 16..48); it adds an Op
    interface for no gain over the fill.
  - *Misses' work first* in each matmul launch: 10-36% slower at T = 16 / 20 / 48 (every CTA
    stalls on PCIe at once and the VRAM work runs after it instead of beside it); expert order,
    which interleaves the misses, is better.
  - *route + admit + group in one kernel* (grouping before the fill): grouping costs 52 us beside
    the fill but the link is busy during it; the three launches take about 15 us per layer, 0.7
    ms per C=5 round (1.2%), not worth an admit/grouping interface. *Skipping the fill and the
    empty deferred pair on hit-only layers* saves at most 1.7 us per such layer, and C >= 4
    rounds have almost none. *One CTA per item at T <= 4* (3 us per gate|up) only reaches C=1,
    where mapped records make the static grid lose (above).
  The remaining levers at C = 4-6 are bytes and overlap: fewer missed bytes (admission and
  cache capacity), copy-engine transfers (53 against 48 GB/s, but the addresses are known only
  on the device inside the graph), and moving transfers off the layer's critical path, either by
  prefetching layer l + 1's predicted experts during layer l (route-prediction accuracy on the
  recorded traces first) or by pipelining two lane groups so one group's dense work overlaps the
  other's transfers: on this trace a 2 + 3 split routes 11% more distinct experts per layer
  (123.3 against 111.3) against up to 13.5 ms per round of dense work to hide.
- **Precision.** A16 only for now. A8 for F3 experts in prefill (one G32 scale per
  `kind::f8f6f4` m16n8k32 step) is the planned compute lever once its study KL and the needle
  battery are measured; A4 is deferred because PCIe, not compute, bounds prefill while experts
  stream.
- **Measured (RTX 5090, cold L2, graph replay, median per layer, 32 NVFP4 + 480 F3 records).**
  FP64-oracle suite (T 1..8192, subnormal and narrow scales, graph replay with changed tables)
  passes at 0.37-0.50 of its bound; memcheck/racecheck clean; pi Sol and Fable
  found no kernel defect.

  | Placement, routing | T=1 | T=12 | T=512 | T=4096 |
  |---|---|---|---|---|
  | VRAM, uniform | 34.8 us (51% DRAM) | 180 us (79%) | 0.81 ms (78%, 68 TFLOP/s) | 3.04 ms (146 TFLOP/s) |
  | host (mapped), uniform | 0.77 ms (96% PCIe) | 8.5 ms (100%) | 39 ms (100%) | 78 ms (50%: records read twice) |
  | odd experts on host, skewed | - | 2.48 ms | 11.3 ms | 33.2 ms |

  Token tile 16 up to T=256 (every expert is a single
  job up to T=16, so decode and verify read each record once), 32 up to 1024, 64 above; only the
  column blocks a job fills are multiplied.
- **Next changes by payoff.** VRAM decode at T=1 is 34.8 us against a 17.8 us byte floor (1.7 ms
  per token over 48 layers): fold the grouping and merge launches into neighbours, give the BF16
  shared-expert CTAs lookahead, and relieve register pressure (214 registers in gate|up C=16);
  profiling needs ncu access from the builder container. Above T=16 an expert with more tokens
  than one tile reads its record once per job, so host-resident experts at large T need the
  prefill streaming window or a single-reader schedule. Prefill at 57% of the BF16 MMA roof is
  limited by activation re-reads across row-tile CTAs; then the FP32 contributions workspace
  (1.04 GB at T=8192) and A8 for F3 experts.

## Program measurements (2026-11-14, f3-fp8-nvfp4)

The target Program (`src/targets/qwen3_8_flash_next/impl/runtime/`: eager chunks, B=1 decode
graph, expert cache with admit, fill, commit and calibration warm start, BF16 paged QSA state of one
lane) on `models/qwen3.8-flash-next-f3-fp8-nvfp4.ninfer`, measured with
`ninfer_qwen3_8_flash_next_parity` (`tools/parity/qwen3_8_flash_next/`), RTX 5090, PCIe Gen4.

- Teacher-forced parity on `data/parity_4x512.npy` against the study's same-recipe reference
  (`results/parity_reference_final_4x512.npz`; the recipe itself is KL 0.060 from BF16), 2044
  positions: KL(reference || engine) mean 0.0045, p50 0.0002, p90 0.0104, p99 0.059, max 0.25;
  top-1 98.0%.
- Prefill against decode: 512 tokens prefilled in 128-token chunks against 504 tokens and 8
  single-token steps, 4 sequences: top-1 32/32, KL median 3.7e-5, max 0.011.
- Graph against eager: greedy decode through the decode graph reproduces eager decode for 2048 of
  2048 tokens at both budgets.
- The greedy token stream depends on the QSA state capacity (`--context`), so decode A/Bs fix it;
  at a fixed capacity runs are deterministic.

Bench on `data/holdout.npy` (chunk 2048; one untimed step, then 2048 greedy graph steps; hit
rate and PCIe bytes from the admit Op's device counters over the window; PCIe time at 26.8
GB/s). Each cell is first 256 steps / all 2048 steps:

| Budget | Prompt | Prefill tok/s | Decode tok/s | Replay ms/token | Hit rate | PCIe MB/token |
|---|---|---|---|---|---|---|
| max (about 27 GB) | 1K | 809 | 114.3 / 113.1 | 8.58 / 8.71 | 92.4 / 92.1% | 74 / 77 |
| | 4K | 1194 | 105.9 / 119.9 | 9.26 / 8.18 | 90.9 / 93.8% | 90 / 61 |
| | 16K | 1121 | 122.7 / 131.9 | 8.04 / 7.49 | 94.5 / 96.0% | 54 / 40 |
| 20 GiB (854 + 9,572) | 1K | 709 | 99.5 / 94.4 | 9.89 / 10.46 | 88.8 / 87.2% | 111 / 126 |
| | 4K | 1132 | 95.8 / 102.6 | 10.25 / 9.58 | 88.1 / 89.9% | 117 / 99 |
| | 16K | 1110 | 110.8 / 114.0 | 8.94 / 8.69 | 92.0 / 92.6% | 79 / 73 |

- **Decode cost model.** Every cell fits replay = 5.9-6.0 ms + miss bytes / 26.8 GB/s within
  0.1 ms: the non-PCIe compute is about 6.0 ms per token and each missed record costs its PCIe
  time. The hit rate is the largest decode lever: at C=1 one point is about 4.8 missed records,
  9.6 MB, 0.36 ms per token.
- **Fills.** `moe_expert_cache_fill` copies a layer's admitted misses at the mapped-read roof
  (26.6 GB/s for 3 records, 26.8 GB/s for 57) with one CTA per SM; four per SM measure the same.
  Against the in-kernel fill it replaced, prefill is 31-43% faster; moe_experts no longer has
  miss tails. Running the fill on a forked stream, with `moe_experts` computing the shared and
  resident experts first and the admitted misses from their slots after it, is neutral for decode
  against the same fill in line (all six 2048-step cells within -0.14..+0.10 ms) and 0.5-1% faster
  for prefill.
- **In-graph gaps.** The 0.83 ms first-kernel gap seen under nsys is the profiler's
  `cudaGraphLaunch` cost (about 1 ms on the host for the 1,167-node graph, against 1.7-4.7 us
  unprofiled in a microbench of the same node count with and without memcpy nodes); the real
  gaps are about 0.1 us x 970, 0.1 ms per token. A graph kernel costs at least about 0.62 us, so
  the 1,166 nodes cost about 0.7 ms of the 6.0 ms, which argues for launch fusion.
- **QSA capacity.** At 1K and 20 GiB, capacities of 1,281 / 16,768 / 65,536 tokens replay in
  10.00 / 10.05 / 10.04 ms. Capturing the decode graph once for the whole capacity costs at most
  0.05 ms, which rules out re-capturing per context band.
- **Hit-rate study** (`--mode trace`: 4K prompt, 2048 eager steps whose admits equal the graph's,
  counters per 128 steps, routes and the post-prefill pools dumped). An exact host model of
  `moe_expert_cache_admit` reproduces every bucket of the engine from the dumped post-prefill
  state, and from the warm-start layout it reproduces the engine with prefill admission disabled.
  The hit rate swings with the content, 0.87-0.99 per bucket at the larger budget and 0.81-0.98 at
  20 GiB, and the first 256 steps after this prompt are among the hardest. The three suspects:
  - prefill pollution costs 1.0-1.5 points in the first 128 steps on generated text and 3.3 on
    teacher-forced text, and 0.04-0.3 points over 2048 steps; it wears off within 256 steps;
  - generated against teacher-forced text: 94.0% against 94.4% at the larger budget. The
    engine's routes agree with the study panel's on 8.3 of 10 experts on the same text and give
    the same hit rate (94.7 / 94.6%, 90.7 / 90.6% at 20 GiB);
  - strict LRU beats CLOCK by 0.14-0.4 points (2.5-7% fewer miss bytes), and the 11% NVFP4 split
    is at the optimum (20 GiB: 400 / 600 / 854 / 1,200 / 1,583 NVFP4 slots give 120 / 105 / 98 /
    99 / 104 MB per step).
  The study panel replayed through the same strict CLOCK gives 93.5% at 20 GiB (the simulator
  predicted 93.9%) and 96.3% at the larger budget, so the gap to the bench is the text and the
  measurement window, not the policy: CLOCK and the 11% split stay.

## Prefill measurements (2026-11-14)

Layer-major super-chunk prefill with the streaming window, `--mode bench` on HOLDOUT followed by
CONFIRMATION tokens, larger expert budget, windows of one largest layer each (2 x 1.06 GB), the
tensor-core `hyper_connection_read`, n-gram rows uploaded per sub-chunk. Logits (all positions,
parity 4 x 512) are byte-identical to chunk-major prefill with and without windows. The stall
column is the time of one super-chunk, which a concurrent lane's decode waits behind.

| Sub / super-chunk | 16K prompt tok/s | 64K prompt tok/s | Activations + workspace + windows | Stall |
|---|---|---|---|---|
| 2048 chunk-major, no window (before) | 1,160 | 1,191 | 0.07 + 0.30 + 0 GB | 1.7 s |
| 2048 chunk-major, window | 1,753 | 1,803 | 0.07 + 0.30 + 2.13 GB | 1.1 s |
| 2048 / 8K | 4,016 | 4,477 | 0.21 + 0.30 + 2.13 GB | 1.8 s |
| 2048 / 16K | 4,798 | 5,929 | 0.40 + 0.30 + 2.13 GB | 2.8 s |
| 2048 / 32K | 4,779 | 6,305 | 0.78 + 0.30 + 2.13 GB | 5.2 s |
| 4096 / 8K | 4,285 | 4,717 | 0.23 + 0.61 + 2.13 GB | 1.7 s |
| **4096 / 16K (default)** | **5,205** | **6,217** | **0.42 + 0.61 + 2.13 GB** | **2.6-3.1 s** |
| 4096 / 32K | 5,191 | 6,688 | 0.80 + 0.61 + 2.13 GB | 4.9 s |
| 4096 / whole prompt (64K) | 5,162 | 6,303 | 1.56 + 0.61 + 2.13 GB | 10.4 s |

The 2048 and 4096 / 8K rows predate the per-sub-chunk row upload (it added 0-4%). The whole-prompt
super-chunk is slower than 32K because the n-gram gather of the first super-chunk (O_DIRECT page
reads, about 19 us per token) is mostly exposed: only layer 0 and the early sub-chunks of layer 1
overlap it. 16K is the default: within 7% of 32K at half the stall.

- **Threshold.** A windowed super-chunk costs about 1 s of copying whatever its width; the
  decode path (admit, fill) costs with the misses. At sub 4096 / super 32K, prompts of 512 / 640
  / 768 / 1024 / 1536 tokens prefill at 524 / 654 / 783 / 979 / 1,559 tok/s windowed against
  466-636 / 533 / 595 / 717 / 958 decode-path, so the crossover is at about 512 tokens:
  `window_min_tokens` = 512.
- **Window size** (sub 4096 / super 32K, 16K / 64K prompts): one layer per window 5,045 / 6,556
  tok/s; half a layer 4,532 / 5,670; a quarter 3,036 / 3,003 (the overflow is read through mapped
  memory once per job, about two jobs per record per sub-chunk). Halving the window returns 1 GB
  to the cache, +0.5 hit points (about 0.17 ms per decode token). The full window stays.
- **Decode after prefill** (first 256 steps after the 16K / 64K prompt): 93.4 / 90.6% hits with
  22.8 GB of slots, against 94.2 / 91.8% after the admitting chunk-major prefill with 25.5 GB.
  The windows and super-chunk activations cost about 3 GB of cache, about one hit point, 0.35
  ms per token; on one stream the experts the admitting prefill left were partly useful to the
  following decode. The isolation matters with several lanes.
- **Transient window.** The 2.1 GB of dedicated windows cost that hit point all the time,
  prefill or not. The windows now borrow cache slots per windowed super-chunk; see "Transient
  streaming window (2026-11-15)".

## Lane-batched decode (2026-11-15)

The Program (`runtime/program.*`) serves 1..12 lanes. Per-lane GDN slots, PLE history slots,
sampling counts and token masks sit beside one shared paged QSA pool: one `PagedKVPool` holds
all 12 QSA layers' planes, NVFP4G16 by default (BF16 selectable), with a block-table row per lane.
`open_lane` reserves and maps ceil(reserve / 64) pages, and `close_lane` unmaps the row and
returns them. `prefill_step` appends one super-chunk to one lane and synchronizes. The feeder
prefetches that lane's next super-chunk rows; another lane's prefill cancels the prefetch and
restores that lane's n-gram history. A decode round of B lanes is one graph replay of batch size
B, with graphs for 1..lanes captured at construction:
- GDN runs through `gdn_input_proj_conv_snapshot` and `gated_delta_net_snapshot` in place on
  each lane's slot.
- QSA runs over the lanes' block-table rows, and `ple_inject` over their history slots.
- One `moe_experts` call takes the B tokens, and its admit sees their union.
- `ops::sample` and `token_logprobs` (top 20) run over [0, 248077).

Measured with `ninfer_qwen3_8_flash_next_parity --mode lanes-check / lanes-bench` on
`f3-fp8-nvfp4`, RTX 5090 at PCIe Gen4. Prompts are 4,096-token slices of an agent coding
transcript (`out/flashnext/long_tokens.i32`), with calibration warm start.

**Verification.**

| Check | Result |
|---|---|
| B=1 parity, BF16 KV (4 x 512, against `parity_reference_final_4x512.npz`) | KL 0.0045, top-1 98.2% (unchanged) |
| B=1 parity, NVFP4 KV | KL 0.0168, top-1 96.0% (the NVFP4 QSA state's cost) |
| Split prefill vs decode | 32/32 top-1 for BF16 and NVFP4; KL median 3.5e-5 / 2.3e-5, max 0.011 / 0.083 |
| Single-token prefill route vs B=1 decode graph, 512 steps | bit-identical |
| Batched vs solo, B=2 | first tokens 2/2, KL mean 0.017, p50 1.4e-4, top-1 97.5% |
| Batched vs solo, B=4 | first tokens 4/4, KL mean 0.012, p50 1.3e-4, top-1 98.0% |
| Batched vs solo, B=12 | first tokens 12/12, KL mean 0.013, p50 2.9e-4, top-1 97.7% |
| Graph vs eager decode at B = 2, 4, 12 | every column bit-identical |
| B=2 decode with a third lane's 40K prefill interleaved every 8 rounds and a fourth lane's 1K prompt interrupting it | B=2 logits bit-identical; the 40K lane's first token and logits equal an uninterrupted prefill |

Batched vs solo is teacher-forced on the solo greedy tokens, 256 rounds per lane. The batched
routes (K=2560 snapshot projection, `moe_experts` at T=B, QSA split count) differ from the B=1
route only in rounding. The KL mean comes from a few routing-flip spikes; mean KL per 64 rounds
does not grow (B=12: 0.0049, 0.0049, 0.036, 0.0050). The unit tests
`ninfer_flash_next_state_test` (page accounting, admission, reuse, selector alignment) and
`ninfer_flash_next_ple_feeder_test` (two lanes, cancel restores history) pass.

**Aggregate decode** (2,048 greedy rounds after one untimed round; device counters; NVFP4 KV;
PCIe time at 26.8 GB/s; "other" is replay minus that PCIe time):

| KV pool | C | Expert cache | Aggregate tok/s | Per lane | Replay ms/round | Hit rate | Miss MB/round | PCIe ms | Other ms |
|---|---|---|---|---|---|---|---|---|---|
| auto | 1 | 24.6 GB (977 + 10,955) | 150.9 | 150.9 | 6.57 | 98.8% | 12 | 0.5 | 6.1 |
| auto | 2 | 24.4 GB | 119.1 | 59.6 | 16.6 | 90.3% | 188 | 7.0 | 9.6 |
| auto | 4 | 24.0 GB | 113.3 | 28.3 | 35.0 | 82.4% | 662 | 24.7 | 10.3 |
| auto | 6 | 23.6 GB | 102.0 | 17.0 | 58.5 | 76.9% | 1,273 | 47.5 | 11.0 |
| auto | 8 | 23.2 GB | 102.7 | 12.8 | 77.6 | 75.5% | 1,743 | 65.0 | 12.5 |
| auto | 12 | 22.4 GB | 95.9 | 8.0 | 124.7 | 70.3% | 2,965 | 110.6 | 14.1 |
| 500K | 4 | 18.5 GB (734 + 8,232) | 81.8 | 20.5 | 48.6 | 73.1% | 1,024 | 38.2 | 10.4 |
| 600K | 6 | 17.1 GB (679 + 7,608) | 68.7 | 11.4 | 87.1 | 63.6% | 2,028 | 75.7 | 11.4 |

The n-gram gather is 0.01-0.38 ms per round and the host gap about 0.07 ms. Solo runs of
prompts 1-3 give 144.5 / 123.8 / 118.3 tok/s at 98.0 / 95.1 / 93.8% hits.

- **Cost model.** Every row fits replay = other + miss bytes / 26.8 GB/s. The non-PCIe time grows
  only from 6.1 ms at C=1 to 10.3 ms at C=4 and 14.1 ms at C=12, so dense weights are amortized
  as designed (most of the growth is the extra resident expert reads). With the solo hit rates
  kept, C=4 would be about 4 / (10.3 + 5.3) ms = 256 tok/s.
- **What limits it.** The four solo prompts together miss 141 MB per four tokens. Run together
  they miss 662 MB, because the lanes route to nearly disjoint experts (about 39 distinct
  experts per layer per round at C=4) and compete for one CLOCK pool. Hit rates fall over the
  run at C >= 2 (C=4: 88.5% over the first 256 rounds, 82.4% over all), the signature of looping
  working sets just larger than the cache, where CLOCK keeps evicting what a lane needs next.
  Every KV gigabyte costs hit rate: C=4 drops from 82.4% at 24.0 GB to 73.1% at 18.5 GB. The
  plan's simulation (C=4: 90.4%, 12.6 ms per step at 20 GB) used mixed-panel traces with more
  shared experts than these single-domain agent lanes.
- **Against the baseline.** Strata reaches 197-207 tok/s aggregate at C = 1/2/4. Here aggregate
  decode falls from 151 at C=1 to 113 at C=4, so lane batching alone does not beat it on Gen4. The
  levers, by expected payoff:
  - a scan- and loop-resistant admission policy (admit on a repeated miss, or a frequency
    filter in front of CLOCK; a non-admitted miss costs the same PCIe bytes through mapped
    reads), to be decided first by replaying recorded C >= 2 routes through the simulator;
  - the transient streaming window (2.1 GB back to the cache);
  - PCIe Gen5, which halves the dominant term;
  - per-lane fairness in eviction.

## Expert admission policy (2026-11-15)

Lane-batched routes were recorded with `--mode lanes-trace`: the same four to twelve agent-transcript
prompts as the aggregate table, 2,048 eager rounds, bit-identical to the graphs. They are in
`out/flashnext/lanes_trace_auto/c{2,4,6,12}` and `lanes_trace_kv500k/c4`. Each trace was replayed
through `tools/parity/qwen3_8_flash_next/replay_cache.py`, an exact model of
`moe_expert_cache_admit` that reproduces the engine's counters bucket for bucket. Belady OPT with
bypass bounds what any policy could reach (`out/flashnext/scripts/opt.py`). The cost is miss MB
per round. The implied rate is C / (measured other time + miss bytes / 26.8 GB/s). A miss costs
the same bytes whether it is filled or read through mapped memory.

| Trace | CLOCK | GCLOCK 8 | GCLOCK 8 + TinyLFU 8 | CLOCK +2.13 GB | GCLOCK 8 + TinyLFU 8 +2.13 GB | OPT |
|---|---|---|---|---|---|---|
| C=1 (holdout trace) | 58.9 MB / 120.5 tok/s | 56.5 / 121.9 | 64.8 / 117.4 (filter off at C=1) | - | - | - |
| C=2 | 188.1 / 120.6 | 160.0 / 128.8 | 156.2 / 130.0 | 152.9 / 131.0 | 120.0 / 142.5 | 74.3 / 162.2 |
| C=4 | 662.6 / 114.1 | 586.6 / 124.2 | 558.0 / 128.4 (-15.8%) | 552.0 / 129.4 | 455.8 / 146.4 (-31%) | 241.8 / 206.8 |
| C=6 | 1,273 / 102.5 | 1,161 / 110.4 | 1,069 / 117.8 (-16%) | 1,075 / 117.3 | 893 / 135.3 | - |
| C=12 | 2,961 / 96.3 | 2,783 / 101.8 | 2,318 / 119.3 (-22%) | 2,472 / 112.9 | 1,952 / 138.0 | - |
| 500K, C=4 | 1,024 / 82.3 | 947 / 87.4 | 896 / 91.2 (-12.5%) | 869 / 93.4 | 751 / 104.1 | - |

Policies compared:
- **Admit on a second miss** within W rounds: 7% saved at C=4 (W = 4), worse with longer windows.
- **TinyLFU, exact per-expert counts:** 24,576 counters, so no sketch is needed. A miss is admitted
  only when its count exceeds the CLOCK victim owner's. The counts halve every A x (total slots)
  lookups, with A = 8 best (A = 1, 2, 4, 16, 32 and no aging all do worse).
- **GCLOCK:** reference counters capped at 8.
- **Per-lane quotas** were not tried. The best policy still sits far above OPT; that gap is the
  remaining lever.

**Adopted (implemented in the `moe_expert_cache` family):**
- GCLOCK with `reference_limit` 8, always. The parallel window CLOCK stays exact, with counters
  decremented per visit and a (K+1) N exhaustion bound.
- Exact frequency counts with halving every 8 x slots lookups.
- `MoeCacheAdmission::FrequencyFilter` on decode rounds of two or more lanes only. It costs about
  10% of miss bytes at C=1, where GCLOCK alone saves 4%.

The filter pairs the k-th miss with the k-th CLOCK candidate, as the sequential policy does, so
the window kernel stays exact.

The same Op family gained `moe_expert_cache_borrow` and `moe_expert_cache_return`: lent slots
have owner -2, are protected and come back empty. The exact oracle test covers them. They are the
basis of the transient streaming window ("Transient streaming window (2026-11-15)").

Gen4 baseline with the adopted policy (2026-11-15, the tree before the PCIe Gen5 switch):
`flock /tmp/ninfer-gpu-queue.lock docker exec ninfer-builder-flashnext bash
/src/out/flashnext/scripts/gen_baseline.sh gen4`. It writes `out/flashnext/lanes_bench_{auto,kv500k,kv600k}_gen4`
and `policy_{parity,split}_bf16_gen4`; the log is `out/flashnext/scripts/gen_baseline_gen4.log`. The
Gen5 rerun is the same command with tag `gen5`. B=1 parity is unchanged (KL 0.0045, top-1 98.2%)
and split is 32/32.

Measured miss bytes equal the replay's (C=4: 557.8 against 558.0 MB/round), and the cost model
holds. The other time is the replay time minus the PCIe time at 26.8 GB/s.

| KV pool | C | Cache | Aggregate tok/s (CLOCK before) | Per lane | Replay ms | Hit rate | Miss MB/round | PCIe ms | Other ms |
|---|---|---|---|---|---|---|---|---|---|
| auto | 1 | 24.6 GB | 152.2 (150.9) | 152.2 | 6.5 | 98.7% | 12 | 0.5 | 6.0 |
| auto | 2 | 24.4 GB | 127.7 (119.1) | 63.8 | 15.5 | 92.0% | 156 | 5.8 | 9.6 |
| auto | 4 | 24.0 GB | 128.3 (113.3) | 32.1 | 30.9 | 85.1% | 558 | 20.8 | 10.1 |
| auto | 6 | 23.6 GB | 119.1 (102.0) | 19.8 | 50.1 | 80.6% | 1,069 | 39.9 | 10.2 |
| auto | 8 | 23.2 GB | 120.6 (102.7) | 15.1 | 66.0 | 79.2% | 1,473 | 55.0 | 11.1 |
| auto | 12 | 22.4 GB | 121.5 (95.9) | 10.1 | 98.3 | 76.6% | 2,320 | 86.6 | 11.8 |
| 500K | 4 | 18.5 GB | 91.4 (81.8) | 22.9 | 43.5 | 76.4% | 896 | 33.4 | 10.1 |
| 600K | 6 | 17.1 GB | 79.6 (68.7) | 13.3 | 75.1 | 68.7% | 1,734 | 64.7 | 10.4 |

The baseline binary predates a null-pointer guard on the filter's counts read in
`admit_pool`. The guard does not change behavior, since the wrapper requires counts for the
filter; it was added for the static analyzer.

The follow-ups (transient window, the OPT gap, Gen5) are recorded in the sections below.

## PCIe Gen5 (2026-11-15)

The 5090 link was switched from Gen4 to Gen5 x16 on the Gen5-rated extender. Pinned `cudaMemcpy`
moves 52.9 GB/s host to device and 53.7 GB/s back (Gen4: 26.8); 120 s of alternating 1 GiB
copies (about 6.4 TB) left the NVML PCIe replay counter at 0. The identical baseline
(`out/flashnext/scripts/gen_baseline.sh gen5`, same tree, same policy) gives bit-identical
parity, split and per-round hit rates and miss bytes, so the rows below differ only in transfer
time; every row matches `other + miss bytes / 53 GB/s` within about 2%.

| KV pool | C | Gen4 aggregate tok/s | Gen5 aggregate tok/s | Speedup | Gen5 ms/round | Hit rate | Miss MB/round |
|---|---|---|---|---|---|---|---|
| auto | 1 | 152.2 | 157.1 | 1.03x | 6.37 | 98.7% | 12 |
| auto | 2 | 127.7 | 155.4 | 1.22x | 12.87 | 92.0% | 156 |
| auto | 3 | - | 170.1 | - | 17.63 | 87.4% | 362 |
| auto | 4 | 128.3 | 188.8 | 1.47x | 21.19 | 85.1% | 558 |
| auto | 6 | 119.1 | 191.9 | 1.61x | 31.26 | 80.6% | 1,069 |
| auto | 8 | 120.6 | 199.8 | 1.66x | 40.03 | 79.2% | 1,473 |
| auto | 12 | 121.5 | 209.1 | 1.72x | 57.39 | 76.6% | 2,320 |
| 500K | 4 | 91.4 | 144.2 | 1.58x | 27.74 | 76.4% | 896 |
| 600K | 6 | 79.6 | 134.9 | 1.69x | 44.49 | 68.7% | 1,734 |

The C=3 row was measured afterwards with a binary of the same tree
(`out/flashnext/lanes_bench_c3_gen5`); `gen_baseline.sh` covers C = 1, 2, 3, 4, 6, 8 and 12 from
here on. Prefill of the 4K lane prompts rises from 2.8-3.0K to 3.8-4.2K tok/s (1.4x). Host RAM reads at
about 51 GB/s with 8 AVX-512 threads, below the Gen5 link, so CPU-computed experts no longer
offer more miss bandwidth than the GPU path and are dropped as a lever.

## Transient streaming window (2026-11-15)

The two prefill streaming windows are no longer dedicated memory. For each windowed super-chunk,
`Forward::borrow_windows` reads the F3 CLOCK hand (one stream synchronization, outside any graph)
and lends 2 x 533 consecutive F3 slots from it, or from slot 0 when the run would wrap, through
`moe_expert_cache_borrow`. The windows are those slots' bytes; `moe_expert_cache_return` gives
them back empty on the stream after layer 47's `moe_experts`. A host-side failure inside the layer
loop (an n-gram read) waits for the enqueued window copies and returns the slots before
rethrowing. Decode rounds of other lanes between two super-chunks therefore run with the whole
pool, and each super-chunk borrows again from the current hand. The cache gains the 2.13 GB
(C=4: 954 + 10,691 to 1,038 + 11,635 slots). `Forward::device_bytes` no longer counts the windows;
`ProgramMemory::window_bytes` is the cache share they borrow, and the Engine's memory summary no
longer adds them to the workspace.

Verification: the Op's exact oracle test covers borrow and return, including a saturated lend,
under memcheck and racecheck. Prefill with borrowed windows gives byte-identical BF16 logits on
parity 4 x 512 (`policy_parity_bf16_gen5tw` against `_gen5`), identical split results, and graph
and eager decode agree on all 512 tokens after the 16K prompt.

A/B in one session, dedicated windows (`out/flashnext/parity_dedicated_windows`, the tree before
the change) against borrowed ones, `out/flashnext/scripts/window_ab.sh`, Gen5, NVFP4 KV:

| Workload | Prefill tok/s | Decode tok/s, first 256 rounds | Hit rate, first 256 | Decode tok/s, whole run | Hit rate, whole |
|---|---|---|---|---|---|
| C=1, 16K prompt (bench, 512 rounds) | 6,068 / 6,052 | 131.9 / 135.1 | 93.3 / 94.0% | 136.4 / 140.0 | 94.5 / 95.4% |
| C=1, 64K prompt (bench, 512 rounds) | 7,567 / 7,502 | 118.3 / 122.8 | 90.1 / 91.5% | 124.9 / 128.9 | 92.2 / 93.4% |
| C=1, 16K lane prompt (1,024 rounds) | 6,082 / 6,055 | 133.9 / 135.8 | 93.6 / 94.2% | 140.7 / 143.9 | 95.4 / 96.3% |
| C=4, 16K lane prompts (1,024 rounds) | 5,591 / 5,631 | 167.2 / 189.8 | 83.7 / 86.7% | 171.9 / 193.8 | 84.2 / 87.0% |
| C=1, 4K lane prompt (2,048 rounds, idle) | 4,100 / 3,898 | 136.2 / 138.0 | 94.0 / 94.6% | 157.1 / 157.8 | 98.7 / 98.9% |
| C=4, 4K lane prompts (2,048 rounds, idle) | 4,168 / 3,974 | 219.2 / 227.6 | 89.5 / 91.4% | 188.8 / 201.3 | 85.1 / 87.8% |

The 4K rows compare the coordinator's Gen5 baseline with this one. A back-to-back rerun of the 4K
prefills (`window_ab4k.sh`, two runs per binary) gave 3,883 / 3,918 against 3,839 / 3,958 tok/s at
C=1 and 4,082 / 4,175 against 4,171 / 4,209 at C=4, so the 4K prefill difference is run-to-run
noise: prefill is unchanged at every prompt length. Decode gains 2-3% at C=1 and 7-13% at C=4.

Gen5 baseline with borrowed windows (`gen_baseline.sh gen5tw`, outputs `*_gen5tw`). The measured miss
bytes equal the +2.13 GB replay within 0.5% (C=2 / 4 / 6 / 12: 120 / 456 / 893 / 1,954 against
120.0 / 455.8 / 892.7 / 1,952.4 MB/round):

| KV pool | C | Cache | Aggregate tok/s (dedicated windows) | ms/round | Hit rate | Miss MB/round | Prefill tok/s |
|---|---|---|---|---|---|---|---|
| auto | 1 | 26.7 GB | 157.8 (157.1) | 6.34 | 98.9% | 11 | 3,898 |
| auto | 2 | 26.5 GB | 162.2 (155.4) | 12.33 | 93.8% | 120 | 4,101 |
| auto | 3 | 26.3 GB | 186.1 (170.1) | 16.12 | 89.8% | 291 | 3,970 |
| auto | 4 | 26.1 GB | 201.3 (188.8) | 19.87 | 87.8% | 456 | 3,974 |
| auto | 6 | 25.7 GB | 204.7 (191.9) | 29.31 | 83.7% | 893 | 3,864 |
| auto | 8 | 25.3 GB | 218.8 (199.8) | 36.55 | 82.5% | 1,239 | 3,750 |
| auto | 12 | 24.5 GB | 226.1 (209.1) | 53.07 | 80.2% | 1,954 | 3,644 |
| 500K | 4 | 20.6 GB | 155.9 (144.2) | 25.66 | 80.1% | 751 | 3,842 |
| 600K | 6 | 19.2 GB | 141.6 (134.9) | 42.37 | 73.0% | 1,495 | 3,511 |

## Expert policy against OPT (2026-11-15)

Replayed on the same traces at the borrowed-window budget (+2.13 GB), against the adopted GCLOCK 8
+ TinyLFU 8. `out/flashnext/scripts/replay_score.py` evicts the exact lowest-scored resident
expert (a heap, not implementable as a parallel CLOCK; it bounds what any approximation of the
same score can reach). Every expert's score is its access count decayed with a half-life of H
rounds, and a miss replaces the victim only when its own score is higher. Variants:
- `lane`: per-lane scores, an expert valued at the maximum over lanes.
- `quota`: per-lane quotas. A resident expert belongs to the lane that last routed it, and a miss
  evicts the lowest-scored expert of the lane holding the most slots.
- `all`: admit every miss.

Miss MB per round:

| Policy | C=2 | C=4 | C=6 | C=12 |
|---|---|---|---|---|
| GCLOCK 8 + TinyLFU 8 (adopted) | 120.0 | 455.8 | 892.7 | 1,952 |
| Decayed frequency, H = 4 / 16 / 32 / 64 / 128 / 256 | - | 510 / 450 / 433 / 432 / 444 / 466 | - | - |
| Decayed frequency, H = 16, admit every miss | - | 450 | - | - |
| Decayed frequency, H = 32 | - | 433 (-5.0%) | 854 (-4.4%) | 1,888 (-3.3%) |
| Decayed frequency, H = 64 | 110.5 (-7.9%) | 432 (-5.3%) | - | - |
| Per-lane maximum score, H = 16 | - | 471 | - | - |
| Per-lane quotas, H = 64 | - | 445 (-2.5%) | - | - |
| Belady OPT with bypass | 59.4 | 196.6 | 381.3 | 870 |

Working sets explain the gap. Per lane, 3.7K distinct experts in 16 rounds, 7.9K in 64, and 13.0K
in 256; the four C=4 lanes together use 10.9K / 17.9K / 22.2K of the 24,576 experts against 12.7K
slots. Only 27% of a lane's accesses repeat from the previous round, and 70% within 16. OPT's
advantage is foresight over that long, flat reuse tail, which recency and frequency do not
predict. No variant reaches the 10% bar, even with exact eviction; the policy stays as adopted.

## Fill rate on Gen5 (2026-11-15)

Host-side nsys GPU metrics (`--gpu-metrics-set=gb20x` at 20 kHz, the device-wide PCIe RX counter)
were sampled over twelve 4K windowed lane prefills and 64 C=12 decode rounds
(`out/flashnext/scripts/fill_metrics_host.sh`, `profiles/nsys/flashnext_fill_metrics_c12`; the
container's nsys cannot import a capture with GPU metrics). The rates are relative to copy-engine
samples (cudaMemcpy, 53 GB/s):
- Zero-copy bursts in decode (fills and mapped expert reads) run at 96% of that rate.
- The windowed prefill's window fills run at 90%. They overlap the layer compute.

An isolated fill microbenchmark of the production kernel
(`out/flashnext/fillbench/fill_grid.cu`) bounds the kernel itself. At one CTA per SM it copies
47-49 GB/s for 16 to 512 F3 records and 45 GB/s for three, against 51-54 GB/s for cudaMemcpy of
the same bytes. Variants measured:
- Grid: two or four CTAs per SM measure the same or 1-3 GB/s less, except +4.8 GB/s at three
  records (6 MB, about 12 us per call).
- Loads in flight per thread: 2, 8 or 16 measure 44-48 GB/s.

The mapped-read roof on this link is therefore about 48 GB/s, 90% of the copy engine. The grid
(one CTA per SM, four loads in flight) is already at that roof and stays unchanged; the
`moe_expert_cache_fill_blocks` comment records the Gen5 measurement.

## KV pages as expert-cache slots (2026-11-15)

`ninfer-serve --max-concurrency 8 --max-context 65536` reserves 524K tokens of NVFP4 QSA state up
front, and the expert cache is sized from what is left, so served decode runs at the reduced-pool
rows of the Gen5 table (C=4: 155.9 tok/s with a 500K pool against 201.3 auto-sized; C=6: 141.6
against 204.7). Most of that pool holds no live KV most of the time. The cache therefore takes
the pool's unused pages as extra slots, and the KV pool takes them back on demand. A record is a
copy of an immutable host record, so taking pages back only evicts.

**Extent geometry.** The pool is PageMajor, one array per plane, so a run of consecutive page ids
is contiguous within each plane. An extent of E consecutive pages hosts, in every plane, as many
F3 records as fit in E x (plane page bytes). From `ops::qsa_paged_plane_specs`, an NVFP4 page is
746,496 bytes over the 12 QSA layers; per layer the planes are 3 x 16,384 (K codes, V codes,
index keys), 2 x 2,048 (scales), 768 (positions) and 8,192 (block keys) bytes per page.
BF16 pages are 1,876,992 bytes.

| KV format | Extent pages (tokens) | F3 slots per extent | Bytes used |
|---|---|---|---|
| NVFP4 | 122 (7,808) | 36 (the 16,384-byte planes) | 78.9% |
| NVFP4 | 244 (15,616) | 84 (two per 16,384-byte plane, one per block-key plane) | 92.1% |
| BF16 | 122 (7,808) | 108 | 94.2% |
| BF16 | 244 (15,616) | 228 | 99.4% |

The 244-page extent is preferred for NVFP4. Lending or reclaiming an extent costs one small
kernel and no copy at either size, because a reclaim only evicts and reused QSA pages need no
clearing (`State`). The larger extent's cost is granularity. Extents are taken back nearest the
low, never-lent pages first, so about one partly used extent sits at the KV frontier: on average
half an extent, 91 MB at 244 pages against 46 MB at 122 (the allocator prefers a lane's own
continuation and the first contiguous free run, so fragmented holes can place a lane elsewhere). Holes left by closed lanes return to the
cache only as whole aligned extents. The 13.2-point byte-use gain is worth more than that
granularity loss whenever more than about 0.7 GB of the pool is idle, which is the case this
design serves. The Op takes any geometry, so the target chooses.

**Op contract (`moe_expert_cache`, implemented).**
- `MoeExpertSlotPool` has F fixed slots (the contiguous array) and X flex slots at
  `flex_address[s - F]`. Slot order is CLOCK order and does not follow addresses.
- `MoeExpertCache::slot_index` I32 [512,L] is the inverse of the owners. The admit hit path reads
  the slot from it instead of dividing the address, and admit and borrow keep it exact.
- `moe_expert_cache_borrow` / `_return` take a `MoeCacheSlotRegion`. A Fixed run is a streaming
  window (contiguous bytes). A Flex run goes back to the KV pool. The host rejects a run outside
  its region, so the transient prefill window can never take KV pages.
- Pool views, counts and the frequency period stay fixed for the Program's life. Graphs bake them
  by value and never need re-capture: lending changes only owner state.
- Lending and reclaiming are ordinary borrows and returns. Lent slots are protected (owner -2), so
  CLOCK skips them, and `moe_experts` never reads them through the tables.

**Invariants for the target.**
- Every page id is in exactly one state: KV-free, KV-mapped, or cache extent.
- The Flex borrow of an extent is enqueued on the main stream before anything maps or writes its
  pages (`qsa_state_append`, prefix copies).
- The Flex return comes after the lane's block-table unmap and after every reader of those pages,
  on every stream.
- Flex slot index F + S x x + j, with S slots per extent, names extent x counted from the lowest
  page of the flex region. Pages are then reclaimed from the low end of the slot index first,
  while the warm start places the hottest records at the high end.
- The transient window uses only fixed slots. `Forward::borrow_windows` already sizes its run from
  `slots.bytes`.

Verification of the Op change (RTX 5090, CUDA 13.1). `ninfer_moe_expert_cache_test` compares the
complete state exactly against an independent sequential model after every admit, commit, borrow
and return, with the slot index derived from the model's owners. It covers:
- flex slots at shuffled addresses, with and without fixed slots, in both codec pools;
- 36- and 84-slot extents lent while a fixed run is lent;
- saturated and multi-window flex pools;
- a captured graph with an in-graph flex borrow and return;
- the host region checks.

That test, the fill, window and residency tests, and compute-sanitizer memcheck and racecheck all
report zero errors.

**Target integration (implemented).**
- Core: `PagedKVPool` has a third page state, withheld ([paged KV §5.5](paged-kv-cache.md#55-withheld-pages)).
  Entitlement and admission are unchanged and count withheld pages as capacity.
- `KvExpertExtents` (`runtime/kv_extents.h`) owns the geometry and the per-extent state. It is
  constructed after `State` and before `ExpertResidency`, withholds every extent while the pool is
  still idle, and hands the flex addresses to `ExpertResidency`. That constructor warm-starts all N
  slots, hottest at N - 1 (the top extent, reclaimed last).
- Every Program operation that maps pages calls `make_room` first: `open_lane` for its whole
  reservation, and `resize_lane` (used by `reuse_lane` and by the Engine adapter's retain, window
  grant and window shrink) for its growth. `make_room` borrows the lowest CacheHeld extents, the
  ones nearest the KV frontier, until the free set covers the pages. Every operation that releases
  pages (`close_lane`, `resize_lane`) calls `give_back`. It returns fully free extents, highest
  first, and always leaves one extent of free pages outside them. Without that spare, a lane that
  grows and shrinks across an extent boundary would take back and lend the same extent on every
  request, evicting its 84 records each time.
- `give_back` makes the compute stream wait for the copy stream before the first return. Host-tier
  transfers of QSA pages run there, and a shrink (retain, rollback) does not wait for them the way
  the adapter's `close` waits for `capture_done`. The borrow needs no join: every restore orders its
  copy stream after the compute stream (`order_copy_after_compute`) once the lane is open.
- Pages stay mapped eagerly: `State::open` and `State::resize` map the whole reservation, as before,
  and only unreserved pages are lent. Lazy materialization ahead of the frontier was considered and
  not adopted. A lane's unwritten reservation is its output budget: at most the request's
  `max_tokens` (`--default-max-tokens` 8192) with RAM off, and one window of at most 8192 tokens with
  RAM on. Completion (retain) shrinks the reservation to the frontier. In the HTTP workload below
  (1024-token outputs), reserved but unwritten pages are at most 8 x 1024 tokens, 1.6% of the
  8 x 64K pool. Lazy mapping
  would add a page-taking path, and with it `make_room`, to every Forward entry point that writes KV
  (prefill, decode, verify and score, which the parity tools and tests also call directly). It would
  also interleave decode-time growth of concurrent lanes in the free set. The RAM/SSD tiers and
  `copy_prefix` then copy one run per page per plane instead of one per plane.
- The prefill streaming windows borrow fixed slots only. `Forward::borrow_windows` sizes its run from
  `slots.bytes` (F). A hand among the flex slots, or a run that would pass F - 1, starts at slot 0, the
  first fixed slots the hand reaches after it wraps. The plan's alternative, F - count, would take
  the slots the hand passed most recently, the warmest of the fixed region.
- Memory reporting: `ProgramMemory::flex_cache_bytes` and `MemorySummary::expert_cache_flex_bytes`
  report the flex slots' record bytes, which lie inside the KV payload. `expert_cache_bytes` is the
  fixed slots' own allocation. The server prints both on its KV capacity line
  (`expert-cache=<fixed>+kv-lendable=<flex>`).

**Verification (2026-11-16, RTX 5090 Gen5, CUDA 13.1, Release `sm_120a`).** Before and after are the
same tree without and with this change (`out/opus-kvflex-base`, `out/opus-kvflex`; logs and scripts
in `out/opus-kvflex-logs`).
- Exact: `ninfer_kv_cache_test` (page states), `ninfer_flash_next_kv_extents_test` (geometry at NVFP4
  244/84 and BF16 122/108; CacheHeld/KvLent transitions, pool page states and the cache's device
  owners, tables and slot index against a model), `ninfer_flash_next_expert_residency_test` (flex
  warm start), State and KV-tier tests; compute-sanitizer memcheck on the extents and residency tests
  reports no errors.
- BF16 parity 4 x 512 with a 125-page pool, whose only extent is lent at startup and taken back when
  the lane opens, gives byte-identical log-probabilities to the tree before the change; the split
  results are identical.
- Real Engine modes pass: retained, RAM, SSD and live tiers, reuse parity, DFlash program and Engine
  cases, DFlash retained, RAM and SSD tiers. The default suite (C=4 against solo), `--program-parity`
  and `--dflash-live-tiers` fail with the same first difference on the tree before the change, so
  those failures predate it (the live-tier case uses a 32-page pool, below one extent).

Expert-cache misses per K3 verify round (`lanes-trace`, 4096-token prompts, 256 rounds, NVFP4 KV):

| QSA pool | C | Fixed cache | Flex slots (lent to KV at start) | Hit rate | Miss MB/round |
|---|---|---|---|---|---|
| 524,288 | 4 | 18.14 GiB | 5.15 GiB (0.16) | 76.5% -> 84.3% | 2,246 -> 1,546 |
| 524,288 | 5 | 17.77 GiB | 5.15 GiB (0.31) | 69.5% -> 78.5% | 3,664 -> 2,635 |
| 524,288 | 6 | 17.56 GiB | 5.15 GiB (0.31) | 68.7% -> 78.5% | 4,349 -> 3,055 |
| 500,032 | 4 | 18.41 GiB | 5.00 GiB (0.31) | 77.1% -> 84.3% | 2,193 -> 1,541 |
| 600,000 | 6 | 17.00 GiB | 5.94 GiB (0.31) | 67.2% -> 78.7% | 4,550 -> 3,031 |
| 40,960 (C x 8192) | 5 | 23.23 GiB | 0.31 GiB (0.16) | 80.3% -> 80.6% | 2,287 -> 2,259 |

Public Engine lanes-bench, DFlash K3, 4096-token prompts: 500K C=4 211.8 -> 253.4 tok/s, 600K C=6
178.1 -> 235.9 tok/s (1,024 outputs per request). At the fixed 8192-token context, C=5 measures 234.4,
180.7 and 239.7 tok/s before and 208.3, 245.1 and 239.6 after (512 outputs): identical tokens and
rounds, and run-to-run spread on the shared host exceeds any difference. The trace row above shows
the cache unchanged there, because a C x 8192 pool has two extents and one is reserved.

HTTP serving (`serve_bench.py`, `--max-concurrency 8 --max-context 65536`, auto KV, DFlash
companion, waves of C requests of 1,024 tokens, mean of two waves, aggregate tok/s). The server
reports `expert-cache=17.50 GiB+kv-lendable=5.15 GiB`:

| Client C | K3 before | K3 after | Adaptive before | Adaptive after |
|---:|---:|---:|---:|---:|
| 4 | 189.0 | 290.3 (+54%) | 206.1 | 309.0 (+50%) |
| 5 | 191.1 | 291.4 (+52%) | 209.7 | 318.4 (+52%) |
| 6 | 181.3 | 271.4 (+50%) | 207.5 | 298.6 (+44%) |

## Levers against the baseline

- Prefill below 4K: fewer expert bytes at equal quality (0e), a larger expert cache (NVFP4 KV,
  smaller workspaces), and streaming overlapped layer by layer; Gen5 would halve the transfer.
- Prefill above 4K: tensor-core expert GEMMs on the chosen format instead of IQ3_S grid
  decoding.
- Decode: drafter acceptance from the BF16 selector codebook and NVFP4 matrices instead of
  Q4_0 throughout; verification and concurrent requests batched in one round, which Strata
  does not do with DFlash2.

## Measured quality (study harness, EVAL 16x2048 tokens, KL against BF16)

| Variant | Mean KL | Top-1 agreement | PPL |
|---|---|---|---|
| BF16 (independent copy) | 0 | 100% | 4.845 |
| IQ3_S GSQ-RCO experts (3.33 bpw) | 0.141 | 88.8% | 4.880 |
| NVFP4 round-to-nearest experts (4.5 bpw) | 0.090 | 91.4% | 4.930 |
| int8 n-gram table, BF16 experts | 0.029 | 95.9% | 4.876 |
| n-gram rows + 0.01% / 0.1% Gaussian noise | 0.027 / 0.027 | 95.9% / 96.0% | 4.842 / 4.878 |
| n-gram rows through FP16 and back | 0.028 | 96.0% | 4.875 |
| n-gram rows through an identity transform | 0 | 100% | 4.845 |
| NVFP4 round-to-nearest shared experts, BF16 routed experts | 0.050 | 93.7% | 4.828 |
| HC down/up FP8 rows RTN / NVFP4 RTN | 0.039 / 0.132 | 95.1 / 89.5% | 4.808 / 5.094 |
| NVFP4 RTN dense, GDN in / GDN out / attention in / attention out (each alone) | 0.076 / 0.085 / 0.055 / 0.134 | 92.6 / 91.7 / 93.4 / 88.9% | 4.731 / 4.781 / 4.873 / 4.924 |
| NVFP4 RTN dense, all four families | 0.183 | 86.8% | 4.956 |
| FP8 row-scaled dense, all four families | 0.061 | 93.3% | 4.884 |
| FP8 row-scaled head / embedding | 0.029 / 0.033 | 95.2 / 95.6% | 4.877 / 4.849 |
| BF16 residual streams (block outputs and every update rounded to BF16), BF16 experts | 0.027 | 95.8% | 4.820 |

Any perturbation, however small, costs about 0.027 KL on this set: top-10-of-512 routing flips
cascade through 48 layers. That floor is included in every number above, the int8 n-gram table
costs nothing beyond it (kept), BF16 residual streams cost nothing beyond it either (the HC
Ops keep BF16 residuals), and KL differences below about 0.01 need paired per-sequence
intervals before they decide anything.

**Standard served NIAH KV check (2026-11-15).** The existing `out/flashnext-engine` server binary,
`f3-fp8-nvfp4`, RTX 5090 / PCIe Gen5 / CUDA 13.1, C=1, context and KV capacity 262144,
passed the standard 27B battery with both KV formats. Fixtures and exact-answer scoring use
`tools/bench/make_niah_positions.py` and `run_niah_check.py`, at start/q25/mid/q75/end;
thinking and prefix reuse are off. Multi-key means one target among 32 distractors.

| KV | Standard 8K/32K/64K/128K, greedy | Multi-key 8K/32K/64K/128K, greedy | Multi-key 8K/32K/64K/128K, p-less T=1.5 seeds 1-2 | Standard + multi-key 240K, greedy | Standard + multi-key 260K, greedy | Total |
|---|---|---|---|---|---|---|
| NVFP4 | 20/20 | 20/20 | 40/40 | 10/10 | 10/10 | 100/100 |
| BF16 | 20/20 | 20/20 | 40/40 | 10/10 | 10/10 | 100/100 |

The 260K prompts contain 260094 standard / 261131 multi-key tokens. The default selection
budget stays at 512 four-token blocks (2048 tokens plus the partial tail) at every length.
That standard panel had no failures, so its budget was not varied. **Keep NVFP4 KV as the default:** the short-panel
KL increase did not cause a retrieval failure here, while NVFP4 retains its memory advantage.
This qualifies the fixed-needle, B=1 battery on the preserved executable, not general task quality.
Standard greedy prefill medians over five positions at 128K/240K/260K were
19.364/35.565/39.013 s for NVFP4 and 19.415/36.757/40.138 s for BF16. Full per-position tables,
prompt files, telemetry, and timing ranges are in `out/sol/needle/REPORT.md` and its adjacent files.
No runtime code was changed or rebuilt for this check.

**Hard RULER-style follow-up (2026-11-15).** The same executable ran five layouts per suite
at 32K/128K/260K, greedy, thinking off, 192 output tokens, strict complete JSON answers with
no extras. The existing generator/scorer support six-value retrieval, four-query binding,
four interleaved six-variable assignment chains, global top-three word aggregation, and a
prose-style archive-code fact; answer metadata is outside the messages. Flash-Next used the
default 512 blocks. The 27B full-attention control used the specified production NVFP4-shell,
BF16-codebook DFlash2 artifact and the historical adaptive K5 baseline settings.

| Model / KV | Multi-value | Multi-query | Tracking 32K / 128K / 260K | Aggregation | Prose needle |
|---|---|---|---|---|---|
| Flash-Next / NVFP4 | 15/15 | 15/15 | 0/5 / 0/5 / 1/5 | 15/15 | 15/15 |
| Flash-Next / BF16 | 15/15 | 15/15 | 0/5 / 0/5 / 0/5 | 15/15 | 15/15 |
| 27B / NVFP4 | 14/15 | 15/15 | 0/5 / 0/5 / 0/5 | 15/15 | 15/15 |

The 27B multi-value miss was one omitted value at 260K/start; its standard battery passed
100/100. Replaying every failed Flash-Next case once at 1024 blocks recovered 1/14 NVFP4
and 0/15 BF16 cases. These are failure-selected probes, not full 1024-budget qualifications.
Tracking failures are not isolated to QSA or KV precision: all 15 control tracking answers
were valid, normally stopped JSON with wrong/missing/extra variables. Some Flash-Next failures
instead emitted invalid/duplicate answers or exhausted the output allowance. **Retain NVFP4
as the KV default:** BF16 did not fix any failed NVFP4 case. The standard needle success does
not establish general long-context reasoning quality. Full per-model/format/suite/length tables,
all failure prompts and answers, and the 354-request verification are in
`out/sol/needle/ruler/REPORT.md`; no runtime code was changed or rebuilt.

## Long-context selection budget

The checkpoint's QSA indexer selects 2048 tokens (512 blocks of 4) plus the current tail block.
Another engine's paired measurement on this model (peonist-ai/halogen-flash-server issue 57,
2026-11) found the default budget confabulates retrievals that a wider one finds: at 16K-32K
needle depths, 94/96 at 2048, 95/96 at 4096 and 96/96 at 8192 tokens. Their cost was prefill
-4.5% to -6.7% and decode about -2% at 4096, and prefill -19% and speculative decode about -10%
at 8192. Perplexity changed by less than 1%. Consequences here:

- The selection budget is a startup setting (2048 to 8192 tokens, whole blocks), not a kernel
  constant. The delivered default remains 512 blocks (2048 tokens): both KV formats passed
  the standard served 8K-260K NIAH battery. The hard tracking failure probes at 1024 blocks
  recovered only 1/14 NVFP4 and 0/15 BF16 cases and do not qualify a globally wider replacement.
- Teacher-forced KL on sequences of at most 2048 tokens never engages selection, so it cannot
  see quantization damage to block selection. The quality bar adds a needle retrieval battery at
  8K-32K depths for the KV format and the final engine, at each budget considered.
- The indexer projections and its key cache stay BF16.
- GPTQ is used only for the routed experts. The R9700 Qwen3.8-27B build
  (`/ssdpool2nvme/local_llm/ninfer-amd-r9700/docs/maintainer/qwen3.8-27b-artifact.md`) found
  GPTQ on attention query/key projections failing 8K multi-key needle retrieval in 5 of 10 runs,
  kept them independently rounded, and protected early attention layers in FP8/BF16; GPTQ fits
  the calibration distribution in ways teacher-forced KL at 2K cannot see. Here: QSA query/key
  and indexer projections are never GPTQ-rounded, attention-layer protection is decided by the
  needle battery, and the battery compares the GPTQ expert store against round-to-nearest
  experts and IQ3_S, not only the KV format and the final engine.
- The expert GPTQ used 32 calibration windows and damping 0.01 x mean diagonal; the R9700
  build chose 0.1 (0.3 for MLP down) on held-out data with 128 windows. A damping and
  calibration-size sweep on the HOLDOUT panel is a cheap quality lever.

## Open decisions

- Sampling: Flash-Next uses the shared `ops::sample` and speculative-round Ops, so with Engine
  integration it inherits the product default (p-less truncation at temperature 1.5, the
  thinking repetition exit, `--no-p-less-sampling` opt-out, DFlash2 p-less verification and
  draft-temperature calibration). The 1.5 default and the p-less constants were measured on the
  27B; the Flash-Next task check confirms or retunes the default temperature.

- Dense weights, decided 2026-11-14 by the dense-format study
  (`/home/battlefront/ai/study/results/dense/REPORT.md`, payloads `quant/dense`, 831 validated
  records): recipe `fallback_mid` — GDN in/out FP8 rows (GPTQ), attention in NVFP4 with MSE
  scales (q/gate and k round-to-nearest, v GPTQ), attention out FP8 rows (GPTQ), shared expert
  NVFP4 (GPTQ + MSE), head FP8 rows (GPTQ), embedding and PLE projections BF16, controls/HC/
  indexer/router BF16 (HC down/up FP8 rows GPTQ, below), per-matrix damping as stored. Whole-model KL against BF16 versus
  Strata's whole GGUF: EVAL 0.131 vs 0.147, HOLDOUT 0.107 vs 0.116 (paired interval crosses
  zero; reused panel), fresh confirmation 0.090 vs 0.104 (paired [-0.022, -0.006]). It reads
  4.80 GB per decode token (dense + shared + head + BF16 controls). Attention-in is stored as one
  NVFP4 parent per QSA layer with a single global divisor (the engine's format; the study's
  per-part globals could not be ingested exactly), regenerated by the study
  (`results/dense/ATTN_PARENT.md`, `quant/dense/attn_parent/`); with it the whole recipe measures
  EVAL 0.1288 vs whole GGUF 0.1468 (paired [-0.038, -0.002]) and confirmation 0.0883 vs 0.1043
  (paired [-0.027, -0.006]). The first EVAL-selected
  all-NVFP4 recipe (3.53 GB per token) failed HOLDOUT and is not used. The BF16 embedding is
  placed in mapped host memory (one 5 KB row per token over PCIe, as the R9700 build does) so
  it costs no VRAM.
- Always-read bytes per decode token in the final artifact (`qwen3.8-flash-next/f3-fp8-nvfp4`,
  55.75 GB): FP8 GDN input 1.51 GB and output 0.57 GB, FP8 hyper-connections 0.65 GB, FP8 head
  0.64 GB, NVFP4 attention input 0.23 GB, FP8 attention output 0.19 GB, NVFP4 shared experts
  0.13 GB, MoE control 0.13 GB, PLE projections 0.07 GB, plus small controls: about 4.17 GB,
  2.5 ms per step at the DRAM roof. The FP8 `hyper_connection_read` (16-read distinct-weight chain,
  Gen5) takes 8.19 us per read at T=1 (SIMT), and with the small-T Tensor Core route (admitted by
  `tools.kdev` `profiled_new_family`; up MMA from T=2, down MMA from T=3) 8.83 / 9.34 / 9.73 /
  9.86 / 10.37 / 13.70 / 15.36 us at T = 2 / 3 / 4 / 6 / 8 / 12 / 16, against 9.09 / 10.37 /
  11.65 / 13.57 / 17.02 / 19.20 / 21.38 us for SIMT; prefill uses the T >= 160 hi+lo BF16
  Tensor Core route (785 us per read at T=4096). HC down/up (`results/dense/HC.md`, records `quant/dense/hc/`, 194
  validated): alone with everything else BF16, FP8 RTN costs 0.039 and NVFP4 RTN 0.132
  (rejected); on top of the chosen recipe, FP8 GPTQ HC measures EVAL 0.1299 and confirmation
  0.0892 (paired vs whole GGUF [-0.025, -0.006]) against 0.1288 / 0.0883 with BF16 HC, so the
  recipe adopts FP8-row GPTQ HC down/up (hc_norm and block_inject stay BF16): 4.16 GB read per
  decode token instead of 4.80.
