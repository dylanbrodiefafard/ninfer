# Final backend, community and deployment source-graph expansion

Retrieved 2026-09-26. Research only: no weights, installed dependencies or GPU experiments. This pass follows issue comments, linked PRs, model cards, author projects and measured recipes rather than treating search-result titles as evidence. **C** means relevant source/patch inspected; **B** means publisher measurements; **D** means discovery/documentation only. The levels do not imply independent reproduction.

## FB01 — Poseidon SGLang: a genuinely additional exact-model stack

Sources: https://github.com/poseidonchan/5090-Qwen3.8-27B ; https://github.com/poseidonchan/5090-Qwen3.8-27B/blob/main/results/summary.json ; https://github.com/poseidonchan/5090-Qwen3.8-27B/blob/main/docs/gdn-state.md ; https://github.com/poseidonchan/5090-Qwen3.8-27B/blob/main/docs/quantization.md . **C+B.** Cloned and read the core patch, overlays, architecture, quantization and aggregate receipt.

Single 5090, custom 19.36 GiB mixed checkpoint, **W4A16** NVFP4 MLP/head, FP8 attention/GDN, target page-64 NVFP4 KV, draft FP8, active/cached BF16 GDN, MTP 3, quantized vision, ReplaySSM. August 30 aggregate: 128K cold prefill in 17.183 s; 260096-token prefill in 49.559 s; 128K+512 decode at 166.621 tok/s; three 128K+1K requests in 59.042 s total (50.811 output tok/s including prefill). C8 aggregate 603.339 is outside current capacity and must not be compared to C4. Predominantly deterministic synthetic token IDs; public runners currently cover smoke and prefill, while other gates have aggregate records rather than standalone reproduction scripts.

Native page-64 attention removes its old conversion path: author 128K prefill 36.134→17.183 s. However the patch loads hash-checked external `.so` variants (`nvfp4_attention_sm120_*`); this repository does not publish all native kernel implementation behind that claim. It is not evidence for beating NInfer's already native compressed prefill. An irregular 109658-token case exposed a tail fallback, useful shape-coverage evidence.

A material **negative** experiment held attention KV fixed and changed only restored recurrent-boundary representation: BF16 host restore had top-1 agreement 1.0 and zero reported log-probability errors in the tested 4K/32K cases; block-8 INT8 state at 32K produced top-1 agreement 0.423529, log-probability RMSE 4.748515 and maximum error 20.90394. Active recurrence remained BF16. This rebuts treating a short successful restore as permission to compress GDN history; it does not establish NVFP4 KV equivalence to FP8, which the author explicitly leaves unqualified.

`overlay/.../attention/qwen38_remappable_workspace.py` reserves a stable CUDA virtual address, unmaps FlashInfer scratch while serialized vision runs, then remaps **fresh physical pages** at the same address. Graph pointers survive, but it synchronizes the device and allocates at every boundary. Current NInfer already plans Text/MTP/Vision phases into the maximum-sized shared scratch arena (`docs/maintainer/qwen3.6-27b-model.md:624–637`). Importing VMM machinery would not create absent phase reuse and would introduce runtime allocation. Screened as a workaround for foreign allocator ownership, not a preferred local design.

`overlay/.../speculative/suffix_mtp_router.py` is an opt-in synchronous C1 suffix/MTP router, fixed MTP 3/verify 4, no rejection sampling/adaptive MTP. It validates ancestor chains rather than treating token 0 as padding and **always** performs draft-extend after verification. This reinforces the current proposal/commit design and contrasts with unsafe draft-sync skipping (FB04); no distinct copy-draft priority is established beyond earlier reports.

Quantization uses native-grid Hessian coordinate descent, not QuIP# despite using BlockLDLQ initialization. Local proxy wins were rejected for 64 gate/up pairs and LM head when downstream gates worsened. An early zero-centered RMSNorm fold used the wrong affine formula and caused long-context repetition; production removed it. This supports propagated long-context calibration and closed-loop admission, not swapping the current artifact or asserting lossless vision quantization. OCR proxy regression remains disclosed.

## FB02 — Prefix retention defaults: a real C2 failure, but different local ownership

Sources: https://github.com/vllm-project/vllm/issues/58303 ; https://github.com/vllm-project/vllm/pull/58549 ; https://github.com/syv-ai/HyperQwen ; https://github.com/syv-ai/HyperQwen/pull/184 . **C+B.** PR 58549 remains **open**.

The issue follows 52216→53595→55760/55861. The comments clarify that the last fixes went to **releases/v0.29.0**, not main. Main's default of 0 retains too few reachable checkpoints under EAGLE tail dropping; the release default of None retains every block and can exhaust the shared pool. There is no single universal current-main default.

The publisher uses the exact 27B model, W4A16 plus DFlash2/KVarN 4/2, a 5090D, two alternating approximately 135K conversations and a roughly 542K-token pool. Dense retention gave 0% reuse and 55–65 s of repeated prefill. An interval of 13056 tokens (6×2176-token blocks) gave 97.0/99.4% reuse and 3.4/1.4 s. The useful interval changes with pool size: the follow-up explicitly warns that the same 6× interval makes approximately 8K conversations regress from 81% hits to 0% on the release branch. The PR derives intervals after hybrid block-size resolution and distinguishes unset from explicit values; one block is effectively dense.

NInfer publishes current/rewrite, sparse prefill-ladder and turn-rollback checkpoints, rather than a recurrent snapshot at every KV block. `docs/maintainer/paged-kv-cache.md:762–790` specifies the default ladder marks and exclusively owned shared KV bundle. **Do not copy the 6× interval or infer the same local flaw.** Alternating independent C2/C4 conversations with restored-answer checks would expose capacity cliffs that single-session exact-repeat benchmarks miss. Metrics should distinguish matched KV from complete reusable GDN state. This is a qualification workload, not proof of a missing implementation.

## FB03 — Follow comments before attributing a cache or kernel defect

Sources: https://github.com/sgl-project/sglang/issues/41351 ; https://github.com/sgl-project/sglang/issues/41351#issuecomment-5847506820 ; https://github.com/ggml-org/llama.cpp/issues/27623 . **B+C for supplied minimal patch.**

SGLang 41351 initially suspected hybrid Radix/GDN checkpoint restoration because repeated branch log-probabilities differed. The follow-up localized the first difference **before recurrence**, at `model.layers.0.linear_attn.in_proj_qkvz`: identical BF16/quantized FP8 inputs and scales, cold M=248 versus warm M=120, and FlashInfer FP8 BMM differences of 0.03125/0.25 across TP2 ranks. Changing GDN checkpoints to FP32 did not fix it. A deterministic-mode fallback to existing scaled matrix multiplication yielded zero difference with real cache hits; reverting the patch restored the failure. This concerns a pinned ModelOpt FP8 SM120 TP2 workload, not a general FlashInfer defect. It is relevant to selective FP8 and cache/batch-shape numerical admission, without establishing a NInfer bug.

llama.cpp issue 27623 alleged a 25× long-context decode collapse on a 4080 Super. An early commenter proposed a CUDA header/runtime ABI mismatch, but the reporter verified the runtime and SM count. The reporter later discovered a **prefill-inclusive denominator**, rebuilt even the original commit and measured a healthy 32.4 tok/s at 91K. The issue closed as unrepeatable. Later comments about smaller Ada KV slowdowns describe a separate shape/hardware claim. The dramatic title is not established engine behavior.

## FB04 — Dynamic K=0 is not “turn speculation off for free”

Source: https://github.com/vllm-project/vllm/pull/53426 (stacked on 51575, companion 53420). **C+B**, open. Inspected early-return changes in autoregressive and DFlash speculators and architecture admission.

Skipping draft-state synchronization at dynamically resolved K=0 saves a forward pass, but recovery depends on the drafter. On two 5090s using TP2, the publisher reports Qwen3.6 MoE MTP savings of 4–11% in selected K=0 cells. EAGLE3 acceptance collapses nearly to zero on resume; MTP at K=8 drops 12–16%; tested Qwen3.8 FP8 plus DFlash2 new-request acceptance falls from 3.36 to 2.87. The 32K C4 DFlash tier achieves 168.9 aggregate tok/s versus 202.6 without speculation. These are not single-5090 NVFP4 C4 results.

NInfer owns its bounded adaptive policy and state commits. An upstream flag does not authorize dropping those transitions. Keep this as negative evidence for alternative scheduling: measure the full round and subsequent resumed acceptance, not the saved forward pass alone. Poseidon's suffix route explicitly keeps draft-extend current.

## FB05 — Reasoning controls: useful alternatives, substantial existing local overlap

Sources: https://github.com/vllm-project/vllm/pull/52677 ; https://github.com/sgl-project/sglang/pull/36750 . **C+B**, both open. Read API fields and GPU-thinking-state patch.

vLLM 52677 runs a bounded token-cycle check on the device, restricts it to the current thinking section, and forces the terminator or ramps a logit bias. State survives speculative rejection, and a forced multi-token terminator closes the section. Tests include GB10, not an isolated 5090 speed result. SGLang 36750 exposes a previously discarded OpenAI request field to existing strict-thinking grammar. PRO 6000 Flash-Next budgets of 16/64/512 yielded 17/65/513 reasoning tokens including the terminator. This fixes external-schema reachability rather than kernel speed.

Current `docs/cli.md:211–231` already defines p-less approximate-cycle exclusion on the first speculative hop and exact greedy behavior; serving also has repeated-tool/reasoning recovery. **Generic loop protection is not missing.** Forcing an early reasoning end changes sampling policy. An explicit thinking budget could be considered on user demand, with answer-quality and tool-continuation qualification. Emitting fewer reasoning tokens must not be presented as an unchanged-workload speed improvement.

## FB06 — SGLang quantized selector and capacity recipes, including superseded pins

Sources: https://github.com/sgl-project/sglang/pull/35496 ; https://github.com/sgl-project/sglang/pull/35786 ; https://github.com/sgl-project/sglang/issues/36452 ; https://github.com/sgl-project/sglang/blob/main/docs/cookbook/autoregressive/Qwen/Qwen3.8-27B.mdx . **C+B.** PR 35496 merged August 20; PR 35786 merged August 21 UTC.

PR 35496 admits quantized target heads through `quant_method.apply`, masks padded vocabulary with negative infinity rather than slicing packed weights or cropping strided logits, preserves contiguous top-k input, and includes the route in graph-folded selection. Greedy speculative and target-only runs can diverge at near ties: the publisher reports 95.5% versus 98.0% on 200 GSM8K questions. Successful serving is not a universal losslessness proof. NInfer already has typed quantized heads and device proposal handling.

PR 35786 matters because the author tested **the documented installation**. An old DFlash BF16 .90 pin failed with first-request OOM, and an old FP32 .945 setting failed during graph capture; pinned revision 1cf2b8c required retuning. One FP32 low-latency combination could not fit five recurrent slots plus 9216 KV tokens. This is historical: the current cookbook reports a 202-configuration GSM8K sweep on v0.5.19, explicitly **without rerunning throughput or acceptance**, and newer memory/chunk settings. New quality gates cannot be attached to old speed rows as a new matched study.

Issue 36452 reports dummy draft embedding/head allocations released only after KV autosizing, starving the pool by approximately 5 GiB. The author's 4090 with 48 GB increased capacity from 149806 to 305070 tokens. NInfer's startup binding/sharing does not allocate and discard random draft heads. This is a framework lifetime bug, not a new generic capacity technique, and the card is not a 5090.

## FB07 — Mainstream kernel PRs that mostly duplicate local work

| Primary source / status | Inspected mechanism and disposition |
|---|---|
| https://github.com/ggml-org/llama.cpp/pull/29187 — open | A 256-thread CUDA CTA per head/token/projection fuses the B/A dot product with softplus/sigmoid, supporting BF16/F16/F32/Q8_0 weights. PRO 6000 Max-Q exact-27B NVFP4 gains 16% at MTP 1 but slightly regresses at MTP 8. Local `Variant::gdn_norm_control_projection` already fuses norm, B/A and gating. |
| https://github.com/ggml-org/llama.cpp/pull/26048 — open | MMQ writeback applies an external scale and optional expert bias, removing graph nodes. PRO 6000 Qwen3.5 prefill gains 7.59% averaged across four ubatches, not one 5090. NInfer NVFP4 TMA already carries alpha and a typed epilogue; there is no equivalent separate weight-scale kernel to remove. |
| https://github.com/ggml-org/llama.cpp/pull/29247 — draft/open | The graph allocator removes storage for fused intermediates, mainly SYCL cast/QSA. Three-card Flash-Next scratch falls from 5696 to 3536 MiB, with modest speed gains. NInfer already plans explicit phase arenas; this does not justify a generic compiler redesign. |
| https://github.com/turboderp-org/exllamav3/pull/330 — closed, **not merged** | Native BF16 I/O, direct grouped projection output and a shared Hadamard transform under a cooperative residency gate. A historical 5090 TP2 prototype reduced K5/K6 projection time and gained 11.576% at C4; that is not the current PR head. The EXL3 codec differs, and local packed projections/native BF16 already remove most wrapper conversions. A schedule comparison, not an NVFP4 forecast. |

The independent kernel-agent report `final-technique-research.md` classifies GDN chunk PR29353, EXL one-warp BA369 and SGLang36865/38082. Material caveats: llama29353 remains unmerged and its K==1 state-snapshot gate disables the route with speculation; EXL369 is closed/unmerged with only +0.86% whole-inference gain and already-fused local BA; SGLang36865 broad L2 policy lost 0.76%, narrowly scoped policy gained 0.98%; 38082 disabled an existing native GEMV baseline, so the reported gain is not incremental to production.

## FB08 — TRT-LLM DFlash2 RFC: real 5090, wrong concurrency for ranking

Sources: https://github.com/NVIDIA/TensorRT-LLM/issues/18085 ; https://github.com/NVIDIA/TensorRT-LLM/issues/17911 . **B**, no submitted DFlash2 patch in the RFC.

Issue 18085 reports one 5090, greedy decoding, 15 concurrent requests of 512 output tokens, FP8 KV, maximum context 4096 and the median of five trials: 770.46 target-only→1126.77 DFlash2 tok/s (+46.25%). It uses an external SM120 FA2 build grafted onto an older TRT branch; long-context, non-greedy and cancellation behavior is not qualified. Selector changes gain 3.65% at C15 but lose at C8. Graph recapture takes 0.5–0.6 s, making frequent shape changes expensive. Draft quantization primarily helps capacity. This establishes neither C1..4 improvement nor an upstream merged feature. NInfer already owns stable graph addresses and resources.

Issue 17911 identifies missing head dimension 256 in SM120 FP8 context-FMHA code generation, causing large unfused scratch usage on Qwen3.6 MoE. It is a useful support-gate warning. NInfer owns compiled native head-256 paths rather than that generator; no new kernel port follows.

## FB09 — HF quantizer cards and structured-output failures

Sources: https://huggingface.co/kelnei/Qwen3.8-27B-NVFP4 ; https://github.com/kelnei/nvfp4-vllm ; https://huggingface.co/bernhardbrieger/Qwen3.8-27B-NVFP4 ; https://huggingface.co/Inferact/Qwen3.8-27B-NVFP4/discussions/2 ; https://huggingface.co/nvidia/Qwen3.8-27B-NVFP4/discussions/5 . **B/D**, cards and discussion contents retrieved without weights.

Kelnei's mixed recipe ranks eight FP8 MLP layer promotions by GPTQ Hessian loss and uses FP8 GDN/head weights. On PRO 6000 with vLLM 0.27.1, changing only the head from BF16 to FP8 at MTP depth 2 and C1 yields 81.1→92.6 tok/s, acceptance 56.4→54.7%, and a weight-only KL increase of 0.0005. This is useful evidence for selective precision, not an identical-artifact or 5090 result. The card's old statement that SGLang cannot quantize the head must be dated against PR 35496 and the current cookbook.

Bernhard's card documents ModelOpt 0.46.0, a 5090, a 20.59 decimal-GB checkpoint, BF16 sensitive families/MTP, and verified 192K capacity with FP8 KV. Absolute quality without a matched BF16 comparison remains limited evidence. Its claim that the 55.6 GB base model fits sequential single-model runs on 32 GB needs offload/distribution details; running models sequentially does not make one model's weights fit.

Inferact discussion 2 reports whitespace loops with vLLM 0.28, TP2 on two 5090s, MTP and xgrammar JSON schema: 12/30 normally, 11/30 without speculation and 7/30 without async scheduling. Unconstrained JSON succeeded in 15/15 cases, and official FP8 had zero failures in 60. `disable_any_whitespace` stopped runaway output but produced empty arrays in 29/30 cases. The failure is checkpoint/grammar-sensitive; calibration remains a hypothesized cause. This is useful admission evidence for any future general JSON-schema feature. Current tool-schema constraints and the p-less sampler differ. NVIDIA discussion 5 only reports 68→50 tok/s with MTP depths 1–3 and insufficient configuration; it remains anecdotal.

## FB10 — Community and recipe receipts that sharpen denominators

| Source | Followed evidence / disposition |
|---|---|
| https://huggingface.co/Qwen/Qwen3.8-27B/discussions/112 ; https://kgptalkie.com/tutorials/generative-ai/qwen-3-8-27b-settings-that-matter | 45 settings on one 5090 using Q4_K_M and llama b10448; median of three runs and reverse-order sweep. MTP 3 wins with F16 KV; MTP 2/3 tie with Q4 KV. The 100% easy quality suite is explicitly non-discriminating. Ollama default MTP/template choices explain much of the apparent runtime gap. Ngram-mod achieves 0.97× on novel prompts; repeated-prompt caching inflated an earlier gain. These are negative controls, not a current NVFP4 comparison. |
| https://www.reddit.com/r/Vllm/comments/1vy9cqt/ran_qwen3827b_on_a_single_5090_with_nvfp4_weights/ | This resolves to the already audited seanyourhighness overlay. The 616 aggregate tok/s C4 result uses 1536-token code output, thinking off and a repeated prompt cache. 262K is configured capacity, not occupied context for that throughput. It is not an independent replication. |
| https://github.com/kutaelee/qwen38-5090-128k-runtime-recipe | Pinned NInfer/artifact, FP8 KV, C1; 601 agent requests yield 171.05 output-weighted decode tok/s. Final task acceptance and cleanup remain pending. Useful agent telemetry, not a matched engine comparison. Native Windows llama versus WSL2 NInfer also changes the environment. Adapter metadata belongs to client protocol translation. |
| https://github.com/darksidewalker/qwen3.8-27b-sglang-dspark-blackwell | Pinned stock-nightly recipe: DFlash2 median 221 versus DSpark median 126/burst 323, with 80K configured context. Grafana plots use existing acceptance counters. Companion checkpoint and attention windows differ. Useful observability packaging, not a new engine or universal DFlash2 winner. |
| https://runpile.com/runs/run_FxKGqjdkjx14BwR- | Primary manifest/raw link: 5090 SGLang, 4096 input tokens and **one output token**, C8, one repetition, synthetic random IDs. The 3.7 output tok/s figure measures prefill/request throughput, not decode. The manifest cache label and HiCache launch flags require interpretation. |
| https://kickerai.com/getting-5x-more-out-of-qwen3-8-27b-on-vllm-a-debugging-story/ | A 48 GB Blackwell, not a 5090; vLLM 0.23 eager mode and 130 image/JSON requests. Moving from a serial client to 12 workers plus MTP 1 reduces the job from 18.2 to 3.4 minutes. Client concurrency dominates and 12 slots exceed this contract. Ordered pool.map results do not imply numerical identity. |
| https://github.com/turboderp-org/exllamav3/issues/385#issuecomment-5779329172 | The maintainer publishes actual 1.5.0 commands: 5090 PCIe gen5 x8, CUDA 13.3, Python 3.14.7 and Torch 2.13 cu132. This closes part of an earlier provenance gap, without adding throughput evidence. An author challenged another offload comparison for incomplete configuration; ordinary weight streaming remains outside scope. |

## FB11 — Wider SM120 labs: preserve negative results and geometry

Sources: https://github.com/loswald/rtxpro6000-bench ; https://github.com/loswald/rtxpro6000-bench/blob/main/box/lists/profiles.tsv ; https://github.com/brandonmmusic-max/sm120-kernels ; https://github.com/flashinfer-ai/flashinfer/pull/2786 . **B/C for retrieved recipes and code-patch descriptions**, no independent reproduction.

Loswald's exact-27B kernel comparison uses **four PRO 6000 replicas at C256–1024**, not one 5090 at C4. b12x/FlashInfer b12x W4A4 achieve approximately 5161/5182 node tok/s, automatic W4A16 approximately 1671 and FP8 approximately 3148. Backend selection can reverse a format headline at large M; this does not imply W4A4 beats Marlin at small M. The quality runner's 435-item inventory differs from the published 403-item rows. Profile notes identify old scoring errors, including grading reasoning as the answer and incorrect template/sampling settings. Fast distributed GLM layouts with degenerate output are not quality winners; a later 20-item check is not 403-item validation.

Brandon's K64 scale-layout fix, `EffBlk_SF=min(K/SFVectorSize,Blk_SF)`, targets CUTLASS grouped MoE GEMM within 99 KB shared memory on four PRO 6000s serving Qwen3 397B. Separate BF16, D128, noncausal attention kernels document double buffering, register P and ldmatrix; NInfer uses D256 causal compressed KV. Constant descriptors remove that implementation's prior per-call allocation, whereas NInfer already captures stable allocated descriptors. Selective block attention changes semantics. README claims about unavailable SM120 attention or undocumented fragment layouts require current kernel and PTX authority. The reported 251 TFLOPS for noncausal D128 does not establish a local gain.

## Query and lead ledger / bounded closure

| Channel / concrete search | Outcome |
|---|---|
| Updated GitHub issue/PR searches: vLLM 5090 since August (341 hits), SGLang SM120 (460), TRT 5090 (25), llama Qwen3.8 performance (167), EXL 5090 (16); up to 40 recent results per query | Selected relevant exact-model/phase issues and read linked PR metadata, files and comments. These are query hit counts, not code-audit counts. Status distinctions are retained. |
| vLLM 58303→58549→HyperQwen 184; SGLang 41351 comments; llama 27623 comments | Resolved a branch-default difference, a numerical causal reversal and a benchmark retraction. No unresolved causal claim is promoted. |
| HF exact-model and quantizer-author cards/discussions | Kelnei, Bernhard, NVIDIA MTP, Inferact whitespace and the 45-setting study. Known fork leads were routed to the fork agent. No weights downloaded. |
| Reddit LocalLLaMA/LocalLLM/Vllm searches for exact-model 5090 engines, failed DFlash and structured output | Resolved the 616 tok/s claim to an existing overlay; followed Poseidon, Darkside and Kutaelee receipts. Apple/AMD/platform branches were screened. Paiton R9700 was routed to the engine-inventory owner. |
| HN web search and Algolia recent “5090 inference” (30 hits) | Mostly hardware economics, Apple comparisons and known engine comments. No distinct matched 5090 kernel receipt. A missing hit does not prove absence. |
| Author projects and raw benchmark platforms | Followed Poseidon code/summary, EXL maintainer commands, Runpile manifest, Loswald profiles and Brandon kernel geometry. Qwentin matched existing TD06 rather than being counted again. |
| PureTensorAI cross-node part 4 | The orchestrator supplied the exact URL after named searches failed. The primary article rejects its headline as a repeating attractor rather than useful decode. |

Every discovered relevant concrete mechanism in this pass has an existing-local, conditional, incompatible, negative, source-limited or other-agent-owned disposition. The kernel agent's PR details are integrated in its report. This does not enumerate all private, unindexed or newly published material. Remaining uncertainties concern unmeasured local performance or unpublished sources/receipts; they are not assumed wins.

## FB12 — A distributed headline explicitly rejected by its author

Source: https://puretensor.ai/blog/distributed-inference-workstation-blackwell-part-4-cross-node-tp . **B/D.** The 1004.9 tok/s GLM result is a locked repeating attractor with accepted length 64/rate 1.00; the article itself rejects comparing this with meaningful decode. It is not evidence that distributed inference beats this native 27B target. Cross-node rank synchronization failures in that experiment do not prove speculation fundamentally impossible across nodes; a correct distributed design can coordinate verification and commits. This is a useful failed-experiment source and remains outside the single-GPU runtime contract.

## Independent cross-review of the companion final reports

Backend reviewer, 2026-09-26. No GPU experiments were run.

- **FT01, b12x:** independently inspected `_cute_kernels.py:1784–1812,1828–1845,1900–1908` and `_parallel_kernels.py:106–157`. The convergence route compares FP32 bit patterns across each warp's state slice, skips subsequent recurrence for a converged slice, and loads its precomputed final state. Unsplit keys and preplanned checkpoint storage are admission requirements. The exact-zero transfer/finite-state flags are separate safeguards. The report correctly restricts the certificate to reuse after recurrence-state equality; it does not certify the entire summary arithmetic against NInfer's oracle.
- **FT03, Kachua:** independently inspected `_apply_unit_lower_inverse` and its FP16 alternative. For selected chunks of 16/32, successive factors include the required powers through N^8/N^16. Nilpotence closes the real-arithmetic inverse exactly. `input_precision="tf32"` and BF16 contractions still require numerical qualification; the report correctly rejects the source comment's precision guarantee.
- **FT14, DDTree:** checked the primary paper's factorized marginal surrogate and local `dflash2_path_select.cuh:498–594` plus `speculative_round.cuh:962–995`. The local builder uses a two-node frontier with accumulated Markov scores, and verification already samples the target then follows a matching child. Budget allocation is a distinct hypothesis; missing tree verification is not. No local speed or acceptance gain follows from the paper's surrogate theorem.
- **FI01, BlackweLLM:** independently checked the August 23 projection note. Both reported rate changes and unchanged 226/255 acceptance/commit count match. The historical performance notes explicitly revisit earlier kernel-bound reasoning and identify state-copy/host orchestration overhead. This supports the report's scoped historical interpretation, not a new local optimization claim.
- **FI02, TensorSharp:** read the no-checkout clone with `git show HEAD`. The README gives the RTX 3080 Laptop, C1, greedy, MTP-off comparison and the dense-27B 1.07×/0.96×/0.95× ratios. The current linked comparison report explicitly has no overlapping TensorSharp/reference cells. The report accurately treats these as different records rather than proving the historical comparison false or independently reproducing it.
