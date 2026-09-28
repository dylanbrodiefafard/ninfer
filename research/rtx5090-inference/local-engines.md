# Local RTX 5090 engines and NInfer forks

Research cutoff: 2026-09-26. This is a source/code investigation, not an engine benchmark. All recommendations below are hypotheses for the current `qwen3.8-27b/nvfp4`, C=1..4, one-GPU resident-compute contract. URLs are preserved here rather than in chat. Stable source IDs start `LE-`. Clones are under `/tmp/ninfer-research/local-engines/` and are disposable evidence, not dependencies.

The reopened depth audit is in `fork-depth-audit.md`: it supersedes the initial incomplete fork enumeration with 469 accessible network entries, adds independently implemented engines Paddock, Quasar/q27 and PegaInfer, and adds concrete general-JSON and decision-scoring fork sources. Metadata enumeration is not code review.

## Findings that can change a NInfer decision

| Priority | Opportunity | What is actually new relative to this checkout | Evidence and next discriminating check |
|---|---|---|---|
| High | Broader graph-only programmatic dependent launch (PDL) for decode | NInfer has PDL primitives and uses them in selected Ops; the Wallawalla fork adds a graph-wide short-grid/streaming-consumer policy | LE-02 code and same-fork interleaved round-time comparison, ~0.8–3.4% lower at C1..4. Classify current Ops first; qualify only a representative accepted schedule. |
| High, workload-specific | Copy/ngram proposals before neural drafts | Current NInfer has n-gram PLE embedding math, **not** prompt-lookup speculative drafting | LE-02, LE-10 code; 8–11% of output from copy drafts on one fork agent workload. Measure actual code-edit/copy corpus; verify target sampler law, grammar and state rollback. |
| High, latency feature | Opt-in idle GPU keep-warm grace | No current `gpu_keep_warm` setting found; OMP runtime has one | LE-03 exact release source plus receipts: 155–157 ms prefill vs 253–304 ms after idle; +~71 W while held. Measure idle-to-first-token, not saturated decode. |
| High, useful feature | General JSON object/schema output | Tool-argument grammar exists, but `src/serve/openai_schema.cpp:448–457` accepts only response format text | LE-17 user workload explicitly skipped extraction. Add general schema contract only as separately authorized implementation; existing XGrammar plumbing is a starting point. |
| Medium | Durable explicit Responses lineage/checkpoints | Current local Responses history disappears on restart; disk KV tier alone does not preserve API IDs/history | LE-03 demonstrates saved complete state and restart/fork continuation. Current docs/serving.md:534 expressly describes process-local history. Target-specific admission still applies; Qwen4 disk/host KV prohibited. |
| Medium | Native small-M W4A4 producer/consumer pipeline | imp uses M32/N64/K256 native FP4 MMA and padded async ring. NInfer already has A4 plus A16 small-T routes, so this is a **shape-specific alternative**, not “add tensor cores” | LE-04 inspect actual verify shapes and classify with tools.kdev before any implementation. No evidence it beats NInfer. |
| Medium | Asymmetric or lower-bit rotated KV | EXL3 2–8 bit independent K/V and fork E8/Rice codecs offer capacity alternatives | LE-07/08/09. Must compare codec quality and full attention latency against current NVFP4. A smaller byte count is not a speed result. |
| Low/conditional | Wider/off/probing speculative controller | Bee controller has no-spec baseline probes, timing EWMA and hysteresis | LE-06. NInfer already optimizes E[Y]/T(k,C,L) over k3/4/5. Bee explicitly disables its adaptive controller on DFlash2 selector mode. |

Already present here: CUDA Graphs, compact batching, prefix reuse, native NVFP4, NVFP4 KV, GDN transaction/rollback, MTP, DFlash2, adaptive draft length, tool schema grammar, multimodal input, local Responses, Qwen3.8 host/disk KV tiers. None are claimed as new ideas simply because another README advertises them.

Historical README Qwen3.6/C8 numbers do not establish the current target's performance. Different quantized weights, sampling, output length, context, speculation, clocks and cache hit rate prevent ranking these engines by headline tok/s.

## NInfer forks

### LE-01 — Discovery scope and limits

Sources: `https://github.com/Neroued/ninfer`, `https://api.github.com/repos/Neroued/ninfer/forks?per_page=100&sort=stargazers&page=1` (pages 1–5 requested); access 2026-09-26. Repository metadata advertised 468 forks; five API responses yielded 282 entries, only **214 unique names**, with overlap and empty/short pages. This is an incomplete exposed listing, not an audit of all 468. Search discovery added forks outside that listing. Deduplicated reviewable inventory is `fork-inventory.tsv` (214 rows, inspection dispositions); raw metadata snapshot is `/tmp/ninfer-research/local-engines/ninfer-forks.json`. Search-discovered external repos absent from the API set remain documented below, not silently added to that214 count.

Read/code-screened relevant forks: Wallawalla47/ninfer-custom, alphastorm/omp-ninfer and alphastorm/ninfer, cometkim/ninfer, gzenz/ninfer, Doelfke/ninfer-yarn, co-l/ninfer, practicorecoza/ninfer-engine, headpiece747/ninfer-5090-windows, harryslimes/ninfer-fast. Other visible specialist forks: Don-Chad/ninfer-3090, geoffwatts/ninfer-v100, natpate/ninfer-windows, toddballinger/ninfer-5080, MirkoCovizzi/ninfer-rtx5090-mobile, ruwwww/ninfer-5060ti, wamansou/ninfer-tp2-1m, lynx-gt/ninfer-tp2-5060ti, ValerioDolci/ninfer-tp2, JCraigWasTaken/ninfer-gfx906, bratnieks/ninfer-amd, 2dameneko/ninfer-xx90-win, tmark00/ninfer, gdevsnack-ai-labs/ninfer-gb10. Their hardware/TP ports are outside the current product; numerical/storage ideas may still transfer, but port popularity is not sm120 performance evidence.

### LE-02 — Wallawalla47: directly relevant PDL and copy drafting

Source: `https://github.com/Wallawalla47/ninfer-custom`, inspected commit `600d8ac` (2026-09-26); README benchmarks describe September builds including `e36f7ee0`, `824e5976`, upstream comparator `bace20dc` plus Windows port `96da12bb`.

Code anchors:

- `src/core/pdl.cuh:38–99`: `graph_dependent`, `launch_consumer`, `enter`, `enter_streaming`.
- `src/models/qwen3_5/program/ngram_proposer.h:18–100`: bounded corpus with absolute positions, boundary sentinels and copy-match gating.
- `src/models/qwen3_5/program/decode.cpp:436` and `:753`: proposal selection and one-hot copy distribution construction.
- `src/models/qwen3_5/program/speculative/target_verification.cpp:10–47`: common target verification, ReplaySSM recording, acceptance and accepted hidden-state selection.
- `src/ops/kernel/speculative_round.cuh:44–66,178–244`: overlay copied proposals, p/q acceptance and positive residual distribution.
- `src/runtime/engine/model_instance.cpp:156–160`: wider frame planning; widths above15 require C1.

**PDL mechanism.** Only captured consumers gain dependent launches. Every consumer waits before reading dependent activation data. Short grids satisfying twice their CTA count <= `%nsmid` trigger downstream scheduling early; `%nsmid` bounds the SM identifier space and is not guaranteed to equal physical SM count (IDs can be noncontiguous), so the fork's “half the SMs” explanation is only a heuristic; weight-streaming consumers wait, then trigger only after the streaming loop. The purpose is to hide launch/immutable-weight prefetch while avoiding SM and bandwidth competition. This is more specific than switching PDL on everywhere.

README's alternated two-pass decode-saturation run with DFlash2 K7 and stochastic sampling reports C1 round time −3.4%/−1.6%, C2 −1.6%/−1.6%, C3 −0.9%/−1.0%, C4 −0.9%/−0.8%. Token throughput at C4 actually fell 557.8→552.0 because accepted outputs vary. Prefer round-time evidence, and do not map these magnitudes to today's NInfer. Current `src/core/pdl.cuh` lacks this `launch_consumer`/`enter_streaming` policy; existing PDL is used by selected q4/q5 GDN and sparse-MoE paths. Current `docs/maintainer/nvfp4-decode-linear.md:169–180` already rejects a particular PDL weight-prefetch attempt as unable to hide activation quantization. This fork's broader graph policy is a different hypothesis but must pass the same classifier, not bypass that negative result.

**Copy proposal semantics.** The fork indexes committed history and selected prompt/cross-request sources, respects text boundaries, requires a suffix match (default12), and leaves source-ending slack rather than copying through an ambiguous ending. It overlays copy rows on neural drafts for mixed batches. Copy distributions set q(copy)=1, all other candidate probabilities0. Target acceptance tests `u*q(d)<p(d)`; rejection draws normalized positive `p-q`. Thus inspected sparse path is not merely greedy token equality. For a deterministic proposal this mathematically recovers the target law if p is the actual complete target distribution and state updates are correct. Greedy mode has its own equality route.

**Qualification gap:** this is not a proof for NInfer's p-less sampler or tool grammar. The inspected fork's `include/ninfer/ops/sampling.h:27` fixes top-k to1..20 and `sampling_device.cuh:148` caps candidates to20. Its warp residual is consistent with that bounded old sampler, **not** current p-less/full-support sampling. Distribution support assumptions, target grammar masks at every tentative prefix, penalties, RNG ownership, mixed copy/neural C1..4, replay state, cancellation, and accepted-prefix publication must be verified against current contracts. Fork source uses old `src/models/qwen3_5` ownership, not current family/Op boundaries. Do not transplant the runtime wholesale.

The mixed agent benchmark reports 8–11% of output from ngrams and wall time 21.6→13.6 minutes, but also changes cache policy, prefill arithmetic, scheduling, and outputs. Cache hit fraction rises67.2→90.2%; this is not an isolated ngram speedup. A copy-heavy workload and a low-match control are the decisive experiments. Dynamic verification width can erase savings if a few copy lanes force all lanes through a much wider verifier.

Other fork features: persistent prefix cache on orderly shutdown, approximate faster INT8 prefill, Vision CPU offload, YaRN up to1M, smaller VRAM headroom, tolerant parsing. CPU Vision violates current compute contract; >native context needs explicit model/quality change; parser tolerance must not silently weaken schema. Current cache already has substantial retention machinery, so adopt no alternative on README claims alone. License: Apache-2.0 lineage, preserve notices and inspect per-file provenance before extraction.

### LE-03 — OMP NInfer: actual new interactive features

Sources: `https://github.com/alphastorm/omp-ninfer`, its `releases/v0.8.2/manifest.json`, `docs/measurements/2026-09-26-engine-keep-warm.json`, `docs/measurements/2026-09-26-idle-gpu-new-sessions.json`; runtime `https://github.com/alphastorm/ninfer/tree/v0.6.11-qwen38-5090-beta.1`, exact commit `32c21f73a7605f76480a6139de0488a14ed1aa48`. Release tag was fetched and inspected; default branch does **not** contain the keep-warm code.

Runtime anchors: `src/core/gpu_keep_warm.cu:14–42`, `src/runtime/engine/engine_core.h:83`, `include/ninfer/types.h:114`, `src/serve/serve_options.cpp:272`. One 32-thread kernel on a separate stream burns3.5ms per10ms during a bounded post-work grace, skips a launch if prior work remains, stops when work is pending. Runtime default0, shipped OMP profile60,000ms. Reported idle12–58s trials: prefill0.155–0.157s, TTFT0.173–0.181s with grace; prefill0.253–0.304s, TTFT0.316–0.366s without. Power99.5–102.7W versus29.3–29.8W idle. Claimed2.3W average is a replay estimate over that author's62-hour agent trace, not general. Source and receipt evidence are strong enough to motivate a local latency experiment, not claim a speedup already available here. **Receipt qualification:** measured artifact hash is the Qwen3.8 groupwise-int artifact (`eec39564…`), BF16 KV/MTP3 (`v081p-gw-bf16-c1024-mtp3`), Windows driver610.88 via WSL2/Docker, not current NVFP4/NVFP4-KV. Each arm first warms a ~30K-token system/tool prefix, then measures new two-message sessions reusing it. These milliseconds are incremental prefill after prefix reuse, not full30K prefill. Measurement source was`c6bd1674`; subsequent release adds a skip-if-spin-pending guard. This limits direct transfer but supports the clock-state mechanism.

Durability goes beyond current disk-KV caching: explicit Responses lineage, complete continuation save/restore after process death, sibling forks, transcript commit before provider state advance. Historical v0.4.0 restores109,589tokens with24.8s end-to-end first-touch restore **plus separate56.6s model reload**; subsecond server TTFT quoted elsewhere excludes those boundaries. v0.6.8 receipts include5.2GB session restore3.6–3.8s. Save/export moved outside the engine lock; README reports warm request during checkpoint traffic15.26→0.91s. Useful candidate if restart survival matters; explicit full contract and state-format work, not a small KV-cache flag. Current Qwen4 tier restrictions still prohibit applying disk/host KV retention to that family. Apache-2.0 NInfer runtime lineage; OMP integration's own files/license need checking separately before reuse.

### LE-07/08/09 — Capacity-focused and deployment forks

- **LE-07 cometkim/ninfer**, `024f6a9`,2026-09-16, `https://github.com/cometkim/ninfer`. README branch ladder: longer MTP, HQ-E8-Rice-2B KV, YaRN and DFlash2. `docs/maintainer/paged-kv-cache.md:282–315` and `src/ops/kv_cache/hq_e8_rice_codec.cuh`: fixed signed Hadamard, nearest2*E8 lattice, bounded Rice codes, norm/offset metadata, plus BF16 sink/recent residual window. D256 row described as64B codes+8B metadata per K/V,144B combined before side rows. This is a materially different codec from current NVFP4, requiring independent exact codec and numerical attention tests. Extension to1M is not evidence of quality there. Several DFlash/quant ideas are already present locally.
- **LE-08 gzenz/ninfer**, `e919931`,2026-09-24, `https://github.com/gzenz/ninfer`. README explicitly warns current version mixes caches under concurrent sessions. Additional host/device retention policies, CPU Vision and YaRN do not justify adoption for required C1..4. Doelfke/ninfer-yarn (`c6b3f61`,2026-09-24; `https://github.com/Doelfke/ninfer-yarn`) extends this line and advertises420K context; distinguish capacity from correct concurrent execution and long-context quality.
- **LE-09 co-l/ninfer**, `72c48f6`,2026-09-20, `https://github.com/co-l/ninfer`, linked `https://github.com/co-l/cache-pressure`. Focuses sustained agent sessions, dirty-cache/overflow and cost-ranked retention. Useful workload fixtures and evidence patterns, but no proven delta against today's checkpoint ladder/retention logic. `practicorecoza/ninfer-engine` (`3a0604d`,2026-08-25) offers rotated/E8 compressed KV and configurable Vision workspace. `headpiece747/ninfer-5090-windows` (`90575c2`,2026-09-21) provides native Windows packaging, v3 artifacts, QAT/full-NVFP4 profiles and four measured launchers. Current project Linux/v2/exact artifact contract prevents taking those launches as supported recipes. `harryslimes/ninfer-fast` inspected at`bace20d`2026-09-24; no distinct hot-path gain established from its README beyond altered defaults/upstream integration.

## Independent engines and llama.cpp branches

### LE-04 — imp, the closest architecture-oriented comparator

Source `https://github.com/kekzl/imp`, inspected`5bb076e`2026-09-26. Primary documentation `docs/BENCHMARKS.md`, `docs/PERF.md`; article `https://rfriedmann.de/blog/serving-30b-models-rtx-5090/` (dated2026-06-02, edited later). MIT license.

Best code lead: `src/quant/nvfp4_gemm_smallm_v2.cu:1–47,144–210`: M<=32, M32×N64×K256, one producer+four consumer warps,4–6stage15KiB/stage ring, native `mma.sync.mxf4nvf4`,144-byte nibble rows to avoid shared-bank conflicts, deterministic striped split-K reduction. Immutable weights may preload before dependent activations become ready. Code records a real mbarrier arrival bug fixed with `.noinc`; stealing the pipeline without the full barrier protocol is unsafe. NInfer's `src/ops/linear/nvfp4/nvfp4_small_t.cu/.cuh` has exact-shape A16 routes and other A4 routes; compare only classifier-approved shapes, including verification token counts and fused output costs.

`src/compute/attention_paged_nvfp4_multitok_gqa.cu` shares decoded K/V rows across Q heads and processes multiple tokens per warp. Current NInfer already groups Q heads/tokens and uses native MMA in `src/ops/kernel/gqa_attention_decode_nvfp4.cuh`; generic GQA reuse is **not** a missing feature. imp code is an implementation reference for load widths/reduction placement, not obvious superior math.

README reports Qwen3.8-27B NVFP4~102tok/s short-context with18.3GiB weights (2026-08-31). `docs/BENCHMARKS.md` reports Qwen3.8 NVFP4/NVFP4-KV @77K,C8 load-width change64.1→74.2 and sparse attention74.3→100.2; sparsity reduces NIAH10/10→8–9/10 and C8 is outside scope. Approximate sparsity therefore cannot be imported as an exact speed fix. GGUF comparison also converts weights to an NVFP4 decode copy and enables ngram speculation by default; same source GGUF does not imply same effective arithmetic. Synthetic prompts can loop into ngram hits, explicitly acknowledged by author. Article claims no other native NVFP4 engine are stale/overbroad; do not propagate them.

### LE-05 — llama.cpp mainline

Primary sources `https://github.com/ggml-org/llama.cpp/pull/21074` (merged2026-04-01), `https://github.com/ggml-org/llama.cpp/pull/22105` (DFlash), repository `https://github.com/ggml-org/llama.cpp`. PR21074's reported RTX5090 Qwen3.5-27B pp512 improves1026→2989tok/s while tg128 remains~63. It uses generic integer MMQ conversion, **not** native FP4×FP4 MMA; the author explicitly calls Blackwell-native MMA future work in that PR. Thus search headlines saying “NVFP4 landed” do not specify execution path. Current mainline evolves beyond that historical PR; do not use it as a current no-NVFP4 assertion.

Useful techniques are already represented in NInfer: paged/unified KV, graph replay, chunked prefill, prompt reuse, speculative checkpointing and constrained output. Current mainline/branches remain practical cross-checks for prompt-format and recurrent-state semantics. CPU offload, broad backend portability, arbitrary model graph and generic GGUF lane conflict with NInfer ownership/product scope.

### LE-06 — BeeLlama

Source `https://github.com/Anbeeld/beellama.cpp`, clone's exact revision in snapshot table below. Read `docs/beellama-args.md`, `docs/beellama-features.md`, `tools/server/server-adaptive-dm.h`, `common/speculative.cpp`. Features: DFlash/DDTree, asymmetric TurboQuant/TCQ KV, reasoning-loop guard, hidden-state capture/rings, and learned adaptive depth.

Controller anchors `server-adaptive-dm.h:141–176,269–335`: estimates expected accepted prefix as sum of survival probabilities, divides output by measured cycle time; EWMA alpha0.15, min3samples,5% switching margins and periodic no-spec baseline. This is useful as a comparison to current NInfer's E[Y]/T controller, not a missing adaptive feature. `common/speculative.cpp:1023,2542` specifically disables Bee adaptive draft-max for DFlash2 selector top-k. DDTree branch budgets create extra verifier work, so tokens/round is insufficient; correct stochastic acceptance and GDN tree state are required.

BeeLlama top-level license is MIT. TurboQuant/TCQ2–4bit offers capacity leads; asymmetric cache parsing does not prove every CUDA path is supported/fast. Default BF16/recurrent cast behavior must be checked independently. CPU hidden-state fallback paths are outside desired GPU resident ordinary execution. Reddit's3.26x DFlash claim is lead-only, not a NInfer comparison: `https://www.reddit.com/r/LocalLLaMA/comments/1u05t6u/benchmark_dflash_speculative_decoding_kv_cache/`.

### LE-10 — ik_llama.cpp

Source `https://github.com/ikawrakow/ik_llama.cpp`, README and `docs/parameters.md`, inspected `common/speculative.cpp:1098–1180,1297–1402`. Supports two-stage speculation (e.g.ngram-mod→MTP), DFlash/DSpark/MTP, suffix drafting and independent per-stage limits. Concrete point: a cheap high-confidence copy proposal can precede a learned proposer without a new model. NInfer lacks that proposer chain; implement exact target-specific strategy rather than importing generic string-driven framework. Hadamard K/V cache (`ggml/src/ggml-cuda/hadamard.cu/.cuh`) is another reference for rotation/layout. Broad quant catalog and CPU/MoE offload are not product requirements. No matched Qwen3.8-NVFP4 C1..4 5090 measurement establishing superiority was found.

### LE-11 — ExLlamaV3 / V2 / TabbyAPI

Sources `https://github.com/turboderp-org/exllamav3` (`12414d0`,2026-09-26), `https://github.com/turboderp-org/exllamav2`, `https://github.com/theroyallab/tabbyAPI`, `https://github.com/turboderp-org/exllamav3/releases`. V3 uses QTIP-derived EXL3 quantization, Qwen hybrid support, custom quantized cache and vendored Flash Linear Attention. Code `exllamav3/cache/quant.py:21–40` accepts independent K/V2–8bits; cache kernels under`exllamav3/exllamav3_ext/cache/`; GDN under`modules/gated_delta_net_fn/` and`vendor/fla/`. These are potential **codec/numerical** comparisons, not interchangeable NVFP4 weights. EXL3 lower-bitrate advantage must include quality and decode cost. V2 is a predecessor, not evidence of a new 5090-native FP4 path. TabbyAPI is a server/integration layer over ExLlama, not a separately comparable matrix engine.

Primary issue `https://github.com/turboderp-org/exllamav3/issues/317` reports Qwen3.8 long-prefill failure with5-bit V cache; a parser-supported codec is not enough qualification. No matched current-target timing found. MIT ExLlama license; review TabbyAPI separately before code extraction.

### LE-12 — FlashRT and HF kernels

Sources `https://github.com/flashrt-project/FlashRT`, `https://github.com/flashrt-project/FlashRT-HF-kernels` (`0823390`,2026-08-21). Runtime emphasizes VLA/multimodal static graphs and fused low-precision producers. Kernel packages are more useful than robot/video latency headlines.

- `flashrt-smallm-gemm/README.md:23–57`: draft SM120 NVFP4 W4A4 M1 matvec, CUTLASS-swizzled scales; smoke correctness K4096/12288, explicitly not fully benchmark-qualified.
- `fp8-prefill-attention-blackwell/README.md:14–16`: native FP8 causal GQA public contract is Q[S,32,128],KV[S,8,128],S multiple128>=256. That is not NInfer's D256 target shape; use as design reference, not drop-in speed figure.
- `gated-delta-attention/README.md:60–66,140–190`: one-launch sequential scan and FLA-style native MMA/WY paths. Some APIs carry BF16 state or cast FP32 scan state to BF16 at end; NInfer's persistent GDN FP32 boundary cannot be weakened just to reproduce those timings.
- `docs/kernel_fusion.md` catalogs residual/norm/quant and projection epilogues; NInfer already owns analogous fused Ops. Only a missing materialization in an actual profile would motivate extraction.

No current target A/B establishes superiority. Hub packaging is not a product requirement. FlashRT top-level license is Apache-2.0; HF-kernels top-level is MIT; per-file vendored dependency notices still apply.

## Breadth closure / lower-priority engines

| ID | Engine/source | Screen result for this task |
|---|---|---|
| LE-13 | `https://github.com/InternLM/lmdeploy` | TurboMind, W4/FP8/MXFP4, persistent batching and KV management. Concrete followup `https://github.com/InternLM/lmdeploy/pull/4510` adds `quant_policy=42` low-bit KV, examined in `kernels-speculation.md`. Primary README H800 MXFP4 advantage is not a5090 Qwen3.8-NVFP4 result. |
| LE-14 | `https://github.com/PaddlePaddle/FastDeploy`, `docs/supported_models.md` | Distributed PD, MTP/chunked-prefill and W4AFP8 features. Broad NVIDIA support does not prove exact checkpoint or native sm120 qualification; distributed mechanisms out of scope. |
| LE-15 | `https://github.com/aphrodite-engine/aphrodite-engine` (redirects dphnAI) | vLLM-derived PagedAttention, modern samplers and TurboQuant; reuse upstream kernel evidence, do not count it as independent CUDA speed proof. AGPL license requires review before copying. |
| LE-16 | `https://github.com/EricLBuehler/mistral.rs` | Independent Rust/Candle-oriented engine with quantization, paged attention, grammar/API features. Current front-page benchmark targets GB10/B200/H100, not matched5090. In-server tools/shell execution conflict with NInfer's inference-only tool contract. |
| LE-18 | `https://github.com/huggingface/text-generation-inference` | Primary repository archived2026-03-21, README maintenance mode. Historical serving ideas, low priority for new sm120 kernels. |
| LE-19 | `https://github.com/ollama/ollama`, `https://lmstudio.ai/docs/app`, `https://github.com/LostRuins/koboldcpp` | Product/runtime packaging and llama.cpp-derived routes; KoboldCpp is a real maintained fork with user-facing cache/sampling features, not merely a GUI. Do not count each product name as an independent low-level kernel result. Ollama has evolving own runtime components too, so blanket “only wrapper” is inaccurate. LM Studio CUDA runtime comparisons must identify actual backend/build. |

vLLM/Radiance/SGLang/TensorRT/FlashInfer/CUTLASS details are owned by the campaign's other reports, avoiding duplicated conclusions.

## LE-17 — Community observations and what they do not prove

`https://www.reddit.com/r/LocalLLaMA/comments/1w821fg/ninfer_vs_llamacpp_vs_vllm_quality_speed/` (access2026-09-26; search indexed~3weeks ago, rendered relative age inconsistent). Author's six-tier50-item-per-tier workload compares different quants/KV/concurrency/spec settings. NInfer extraction tier skipped because `json_object` returns400: confirmed in current source and a useful feature request. Claims “all speed is MTP” are not supported by a matched no-spec kernel comparison; vLLM speed excluded by author due to timing-boundary mismatch. Conditional retrieval100% is not unconditional admission success or proof of statistical equivalence. No public raw harness found through this post; poster link is an external generated artifact.

`https://www.reddit.com/r/LocalLLM/comments/1wip08q/finally_got_my_1x_5090_setup_dialed_in_for/` (access2026-09-26; rendered5h ago, search indexed~1week). Custom unnamed NInfer fork claims920tok/s decode **aggregate five lanes** +414tok/s background prefill; model NVFP4+9FP8 tensors, MTP4, overclock/power limit, rotated KV and NVMe spill. No source/config link in inspected post/comments. C5 exceeds current contract; screenshot is not a sustained matched benchmark. Useful idea is request-phase latency measurement under simultaneous ingress, already partly addressed by NInfer scheduling; not evidence that changing to that setup beats current C1..4.

`https://gist.github.com/PierpaoloPernici/f1d1382f8e357b4faffb1a9f584cc1df` is a discoverable same-model context/concurrency/tool benchmark lead. Primary Gist API and attached Markdown reports were inspected: created2026-08-17, updated2026-08-26. Most quality/concurrency numbers use groupwise-int+INT8 KV, while vLLM uses NVFP4+FP8. The later NVFP4 rerun covers only C1, not quality/C2/C4; NInfer was configured C2 during C4 traffic (queueing), one long C4 case lost a request, and headline client rates include prefill. The reported128K MTP ceiling is historical and contradicted by current code. Disposition: useful workload/schema diagnostics, no current speed ranking. Detailed feature lessons are in `community-features.md`.

## Research completion and provenance

All identified high-value local-engine avenues above were followed to primary source or marked with the exact evidence gap. Nothing was built, installed or benchmarked; no model was downloaded and no runtime implementation changed. Third-party timings are claims with stated boundaries. Exhaustive internet/fork completeness is impossible and explicitly not asserted; useful next work is controlled candidate qualification, not additional repost collection.

### Inspected clone snapshots

| Repository clone | Revision and commit date |
|---|---|
| Anbeeld-beellama.cpp | 0ba48c5 2026-09-25 |
| Doelfke-ninfer-yarn | c6b3f61 2026-09-24 |
| Wallawalla47-ninfer-custom | 600d8ac 2026-09-26 |
| alphastorm-ninfer | 82a1a22 2026-08-31 |
| alphastorm-omp-ninfer | 22d68fb 2026-09-27 |
| co-l-ninfer | 72c48f6 2026-09-20 |
| cometkim-ninfer | 024f6a9 2026-09-16 |
| flashrt-project-FlashRT | 839b159 2026-09-26 |
| flashrt-project-FlashRT-HF-kernels | 0823390 2026-08-21 |
| gzenz-ninfer | e919931 2026-09-24 |
| harryslimes-ninfer-fast | bace20d 2026-09-24 |
| headpiece747-ninfer-5090-windows | 90575c2 2026-09-21 |
| ikawrakow-ik_llama.cpp | cdf232c 2026-09-26 |
| kekzl-imp | 5bb076e 2026-09-26 |
| practicorecoza-ninfer-engine | 3a0604d 2026-08-25 |
| turboderp-org-exllamav3 | 12414d0 2026-09-26 |

Snapshot date normalization: alphastorm-omp-ninfer `22d68fb` has author and committer timestamp `2026-09-27T02:06:34+07:00`, which is **2026-09-26T19:06:34Z**. The apparent future date in `git log --format=%cs` is timezone formatting, not evidence retrieved beyond the UTC cutoff. Repository commit dates identify source history; retrieval was during this2026-09-26 campaign.
