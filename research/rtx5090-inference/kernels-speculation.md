# RTX 5090 kernels, attention, and speculative execution

**Depth-audit update (2026-09-26):** `technique-depth-audit.md` adds inspected SM120 GDN-prefill fusion, current grammar-mask serialization, feature-tap overlap, TMA weight-only eviction and fused quantization schedules. Its findings revise the original shortlist; this report alone is not a comprehensive technique-coverage claim.

Research cutoff: 2026-09-26. This is a research assessment, not a measured NInfer performance claim. External repositories were inspected without building or executing their code. Exact-target comparison is Qwen3.8-27B NVFP4, one RTX 5090, C=1..4, default NVFP4 KV. Numbers for other models, hardware, precision, contexts, sampling, and concurrency are evidence about a technique, not speedups over this checkout.

## What NInfer already has

The local implementation materially changes the recommendations one would make from external engine marketing:

- `src/ops/linear/nvfp4/nvfp4_config.h:343` already selects shape-specific A16/GEMV, SmallT, and W4A4 routes. T=1 stays GEMV; early W4A4 cutovers differ by projection. Exact-geometry A16 GEMM decodes a weight tile once per activation panel. `nvfp4_w4a4_tma.cu` already contains TMA candidates.
- `docs/maintainer/dflash2-tree-speed.md:525` records target projections as DRAM-bound and rejects per-projection A4 changes that altered recurrent draft decisions. It also records no matched Engine benefit from keeping SwiGLU weights in L2 and rejects proposal/target overlap because verification depends on draft IDs while the small independent preparation loses to contention.
- The same document, around line 551, attributes 29–31% of C1 round kernel time to target SwiGLU, 13–17% to GDN input, 15–16% to MLP-down, about 5% to target head, and about 1% to recurrent fold. Those three dominant projections already read one weight pass. Generic occupancy/TMA/cache changes cannot be assumed to remove a second pass that is absent.
- `src/targets/qwen3_6/impl/runtime/adaptive_draft.h` and `docs/maintainer/qwen3.6-27b-model.md:420` already implement online expected accepted tokens / measured round time, conditional per-hop acceptance, per-C timing, context-length slope, exploration bounds, and graph choices `{3,4,5}`. Recommending “adaptive draft length” alone is duplication.
- DFlash2 NVFP4 matrices, BF16 selector codebook, optimized top-k, MTP, graph isolation, NVFP4 KV, Sage3, XAttention and Sparge are present. The current supported DFlash2 product path is chain-only, maximum k=5; older tree experiments in historical documents are not a live tree product promise.
- `docs/performance.md` records that the additional selective FP8 328 MiB recipe costs roughly 2–3% throughput in its matched experiment while improving corpus quality. Do not mistake it for a speed optimization.

## Ranked actionable research opportunities

Ranking is potential upside, not immediate implementation readiness. Within the current exact-artifact contract, the immediately actionable research is the bounded prompt-lookup/suffix proposal screen and specific source/shape comparisons; training a companion requires compatible weights and a training decision, while replacing the target checkpoint requires an artifact-contract decision. Neither is implied by this research request.

For any companion, the distribution oracle is the target's complete conditional law over represented target logits at the exact tentative prefix, after the same penalties, temperature, truncation, grammar mask and stopping rules. Classical rejection must evaluate the true proposal probability and positive residual, or use the current target-only/p-less acceptance contract as specified. Matching a few greedy outputs is not evidence for stochastic law preservation. Training also requires source-compatible hidden-state extraction, an authorized training corpus and sufficient training hardware; none was provisioned in this campaign.

| Rank | Technique / evidence | Distinct opportunity here | Gate before implementation |
|---|---|---|---|
| 1 | Exact artifact + draft co-design and training on the actual served tool/chat template | Improve accepted tokens per target weight pass instead of trying to move the same weight bytes faster | Exact target/draft compatibility, real tool/code/chat/long-context held-out data, quality and C1..4 end-to-end evidence |
| 2 | FlashInfer B12x SM120 shape specialization and W4A16 Marlin-style schedules | Compare only classifier-admitted gaps such as wider verify/prefill panels or particular small-output matrices | Same decoded-weight oracle; include activation preparation, fused epilogue, scratch, and graph costs; no blanket backend replacement |
| 3 | ARCQuant / SharQ residual-corrected activation quantization | Potentially recover enough A4 quality to make otherwise rejected compute routes usable | Artifact/quantization work, byte/compute classifier, independent oracle and recurrent/speculation behavior; see quantization-quality.md |
| 4 | Prompt lookup / n-gram proposal as a workload-specific draft alternative | Potentially eliminate draft-model bandwidth on editing, repetition, and predictable structured outputs | Evidence it beats existing DFlash on that workload, bounded history, exact acceptance and state transactions; no extra model identity |
| 5 | Alternative KV codecs / query precision treatment | Capacity or long-context quality benefit, not an assumed C1 short-context speedup | End-to-end quality at positions/context lengths, packed-state oracle, actual metadata and transform costs |

## FlashInfer: inspect the SM120 backend, not the Blackwell label

Primary repository: https://github.com/flashinfer-ai/flashinfer

Inspected snapshot: `78c6e1fbfcb65043cdee92506c077259e7e878b7`, dated 2026-09-26; local clone `/tmp/ninfer-research/kernels/flashinfer`.

Concrete source: `flashinfer/gemm/kernels/dense_blockscaled_gemm_sm120_b12x.py` (especially `_select_default_mma_tiler_mn`, around lines 3017–3100). Its FP4 M=1 planner explicitly distinguishes `(64,128)`, `(64,64)`, and narrow-N `(64,32)` with swapped operands, rather than applying the datacenter kernel. Larger small-M problems favor a narrower 64-row tile to avoid wasting the M extent. The planner selects TMA and accounts for SM count. The source also has separate MXFP8 small-M choices; these must not be misreported as FP4 tile choices.

Transferable principle: choose compute and load layout for the exact M/N/K regime and combine width decisions with occupancy. NInfer already does this, so the useful artifact is a directly comparable external schedule at a demonstrated weak point. A public `mm_fp4` benchmark that omits FP4 activation quantization is not a complete Linear comparison.

Correctness history: issue https://github.com/flashinfer-ai/flashinfer/issues/3398 (opened 2026-05-24, now closed) reports silent M-dependent zeros on SM120 with FlashInfer 0.6.12, in both B12x and CUTLASS, after local compiler workarounds. It links fix #3497. This is historical evidence for shape-dependent qualification, not proof current FlashInfer remains broken. Related CUTLASS report https://github.com/NVIDIA/cutlass/issues/3096 concerns grouped MoE and mixed compilation paths; its multi-GPU 397B speed figures are inapplicable to the dense single-5090 target.

## Marlin and FLUTE

Original Marlin source and design: https://github.com/IST-DASLab/marlin (README and `marlin/marlin_cuda_kernel.cu`, read 2026-09-26). Its techniques include striped work distribution, asynchronous weight loads with streaming cache policy, prearranged offline weight/scales layout, overlapping dequantization with tensor-core work, and reduction scheduling that sustains bandwidth reuse across small batches. Its original results are FP16×INT4 on older GPUs; they do not establish an NVFP4 5090 win. NInfer already contains streaming hints, exact small-T schedules and artifact-owned layouts. The meaningful question is whether a specific W4A16 tensor-core schedule avoids duplicated work at a presently weak matrix geometry.

FLUTE: https://github.com/HanGuo97/flute (README inspected 2026-09-26). LUT quantization and packed lookup/dequantization are relevant to unusual codebook matrices, but its reported support/results center on earlier GPUs and different weight formats. NInfer's DFlash2 selector codebook is BF16 and numerically intentional. Switching it to LUT quantization changes the represented artifact and requires quality evidence; it is not an interchangeable NVFP4 GEMM implementation.

## FlashAttention and SageAttention

Current upstream SM120 code was read directly at https://github.com/Dao-AILab/flash-attention/blob/main/flash_attn/cute/flash_fwd_sm120.py on 2026-09-26. `FlashAttentionForwardSm120` subclasses the SM80 algorithm and enforces SM120 shared-memory capacity; it accepts FP16/BF16 and forces the SM80 instruction path while targeting the resident GPU. Therefore “FA4 cannot run on 5090” is too broad: the repository has an explicit SM120 route, but SM100's TMEM/tcgen05/WGMMA algorithm is not made available by changing an architecture flag. Source-level head-dimension/resource checks supersede stale secondary tables.

SageAttention3: https://arxiv.org/abs/2505.11594 (2025-05-16), https://github.com/thu-ml/SageAttention. Its FP4 attention paper reports 1038 TOPS and 5× over its FlashAttention baseline on RTX 5090; this is attention-kernel throughput, not a full LLM decode multiplier. NInfer already has Sage3 routes. In particular `src/ops/launcher/gqa_attention_decode_s3.cu` documents FP4-P as a quality floor with NLL and acceptance problems. Copying an FP4-P speed result would reopen a known numerical issue, not add a missing feature.

External experimental attention tuning: https://github.com/qu0b/fa4-sm120-research (RTX PRO 6000, 188 SMs). Useful candidate concepts include thread-count/tile changes under the 99 KiB shared-memory limit. Its larger card and BF16 attention are not a measured improvement over NInfer's quantized KV paths.

## KV compression

NVFP4 KV integration source: https://github.com/hikarioyama/vllm-nvfp4-kv-sm120/blob/main/docs/DESIGN.md (read 2026-09-26). The design eliminates scale scratch through explicit scale strides and in-kernel inverse V-scale swizzling; for head size 128 it accounts for 72 bytes per token/head/side, including scale bytes, versus 128 for FP8. It documents why naive repacking scratch erased the nominal capacity benefit. NInfer already owns its physical KV layout, so this is a layout/accounting check, not a reason to copy vLLM's interleaving.

TurboQuant: https://arxiv.org/abs/2504.19874 (2025-04-28). Random rotation plus scalar quantization and residual inner-product correction offers a distinct lower-bit KV representation; the paper reports near-neutral evaluated quality at 3.5 bits/channel. It is not a native NVFP4 drop-in: rotations, norms, metadata, random state, decode/attention math and cache compatibility matter. Lower payload bits do not guarantee lower 5090 latency. Its claimed quality does not establish quality for this Qwen artifact or recurrent/speculative workload. Prioritize only if capacity or observed long-context attention quality is the decision, and compare with current NVFP4 at equal total storage.

## Speculation: current features versus new work

DFlash upstream: https://github.com/z-lab/dflash (README read 2026-09-26), paper https://arxiv.org/abs/2602.06036 and DFlash2 announcement https://inco.ai/blog/dflash2/. The upstream now lists Qwen3.8-27B DFlash2; NInfer already supports this artifact family. Its MLX note recommends block size at most five for quantized execution due to verify-matmul efficiency. That is different hardware, but reinforces the principle that maximum acceptance length need not maximize speed. NInfer's k=4 versus k=7 history independently establishes that effect locally.

EAGLE3: https://github.com/SafeAILab/EAGLE and https://github.com/sgl-project/SpecForge. A trained checkpoint-specific drafter and its feature taps are required. EAGLE3 is not a free runtime switch for arbitrary Qwen weights. Its chief value is a competing draft-training recipe if a compatible exact-target artifact is available, not adding every speculative algorithm to Engine.

Adaptive scheduling primary sources: https://github.com/sgl-project/sglang/blob/main/docs/docs/advanced_features/adaptive_speculative_decoding.mdx; Nightjar https://arxiv.org/abs/2512.22420; Learning to Draft https://arxiv.org/abs/2603.01639. They motivate accounting for draft/verify time and request load. NInfer already optimizes that ratio with concurrency-aware time. A new controller is justified only by an identified decision failure (for example a bad timing model or inability to turn off unprofitable speculation), not by the word “adaptive.” Learned policies introduce training/robustness work that has no established need at fixed C1..4.

There is a narrower local reason to investigate: `plans/qwen3.8-27b-performance.md:213` records residual adaptive losses on mixed CUDA/Python traffic. Its linked historical log includes older widths, so those measurements are not current-source reproduction. Today's `adaptive_draft_ks` chooses the captured positive widths, rather than an automatically measured target-only arm. A fresh phase-change workload could decide whether bounded fallback to no speculation or faster change detection has value. This is a specific controller question, lower priority than existing numerical work, not permission to replace the current ratio-based policy with a generic bandit.

N-gram/prompt lookup primary code: https://github.com/vllm-project/vllm/blob/main/vllm/v1/spec_decode/ngram_proposer.py (verified on retrieval). No learned draft is required. Potential novelty here is saving drafter memory traffic on copying/editing tasks; it still needs target verification and GDN rollback. For a directly related NInfer fork's copy-draft proposal and acceptance audit, see the local-engine research report. Fork evidence is a stronger immediate lead than porting a generic suffix framework.

The suffix-GPU RFC https://github.com/vllm-project/vllm/issues/51788 describes a device-resident suffix index compatible with async scheduling, with results on L20 rather than 5090. It explicitly reports a high-concurrency crossover where no speculation wins, and identifies padded speculative slots contaminating acceptance denominators. Its transferable ideas are bounded draft-free proposal state and correct work accounting. There is no verified gain over NInfer DFlash2, and GPU index memory competes with KV capacity.

DSpark primary architecture documentation https://github.com/vllm-project/speculators/blob/main/docs/user_guide/algorithms/dspark.md describes Markov and confidence heads on a DFlash backbone. Crucially, the anchor predicts a next token by default; ordinary DFlash instead discards the anchor's output. The companion document `dflash.md` and current vLLM `vllm/v1/worker/gpu/spec_decode/dflash/speculator.py` explicitly encode this position difference. A mismatched importer can shift draft positions while producing plausible logits. DFlash2 already has inter-position selector dependencies, so DSpark's case against independent DFlash1 positions does not establish superiority here. SpecForge and Speculators are useful training/extraction tools if companion work is selected; distributed hidden-state connectors are not a runtime feature NInfer needs.

Jared Frost's 2026-09-18 exact-model practitioner report: https://jared-hpc.com/posts/nvfp4-qwen38-27b-rtx5090/. It describes a DSpark drafter trained with the same tool XML, tools array and reasoning preamble used at serving time. Reported acceptance improvement concentrates on agentic phases rather than the overall average. The transferable opportunity is serving-distribution alignment; NInfer already quantizes its DFlash draft. Do not import its 420 tok/s code result as a general decode target, and do not assume its custom target weights are compatible with the stock DFlash2 artifact. Full artifact-quality discussion is in `quantization-quality.md`.

## Rejected conclusions and measurement design

- No source inspected proves “engine X is faster than this checkout” under a matched exact artifact, default NVFP4 KV, sampling, context, C1..4 and output workload.
- More draft tokens, larger trees, and wider batching can increase verification cost faster than accepted tokens. Current product limits and previously rejected routes are material.
- Additional weight compression is an artifact decision. CPU/ordinary-weight streaming, multi-GPU results and sm100-only kernels do not answer this target's speed question.
- Any future kernel proposal first goes through `docs/maintainer/kernel-iteration.md`: actual production shape, byte/compute bound, allowed recipe, independent oracle, public Op timing and then matched Engine evidence. Existing tiny selector/control costs have a small Amdahl ceiling.
- For draft research, measure accepted tokens per round, complete-round GPU time, actual completion quality and per-phase output rate on held-out tool/code/chat/reasoning prompts. Keep greedy and stochastic regimes separate, include late context, and use aggregate complete-wave makespan at C>1.

## Final targeted gap screen

**Conversion reuse is substantially present.** Independent cross-review of the vLLM research found that `src/ops/kernel/gqa_attention_prefill_nvfp4.cuh:432` already dequantizes the current V tile into shared `v_bf16` before the PV MMA loop, with dedicated V worker warps. QK uses native NVFP4 MMA at line 361. Thus FlashInfer PR4346's one-time conversion per tile does not establish a new optimization for this primary NInfer path. A different consumer would need concrete evidence of repeated conversion before pursuing it.

**Mixed KV precision is quality/capacity research.** RateQuant, https://arxiv.org/abs/2605.06675 (v2 2026-06-26), warns that bit-allocation rankings depend on the actual quantizer's distortion curve; a policy fitted to another quantizer can worsen quality. It does not establish 5090 NVFP4 latency or this artifact's optimal head allocation. That makes a blind “more bits for keys” rule insufficient. Heterogeneous head codecs would complicate current layouts and consumer schedules; only a measured long-context capacity/quality limitation could justify that change.

**There is real TurboQuant integration code, but inspect its codec.** LMDeploy PR4510, https://github.com/InternLM/lmdeploy/pull/4510, merged 2026-04-16, adds `quant_policy=42`, Hadamard transforms, Lloyd-Max codebooks, and paged fill/read/attention paths. The patch's `lmdeploy/pytorch/kernels/cuda/turbo_quant.py` distinguishes 2-bit values and 3-bit keys with QJL correction; its “K4V2” shorthand is not a uniform four-bit scalar codec. Optional fast Hadamard versus matrix fallback materially changes overhead. No matched Qwen3.8/5090 result was established. The PR description incorrectly links arXiv 2510.17153 (HyperSearch); use the actual TurboQuant paper KS10 and do not treat the mismatched citation as evidence for its headline fidelity/compression claims.

**Draft-free does not remove grammar semantics.** ArcticInference issue183, https://github.com/snowflakedb/ArcticInference/issues/183 (2025-09-16, open at cutoff), reports suffix decoding pushing tokens after a guided grammar had terminated. This is evidence for preserving termination/grammar state at the acceptance boundary, not proof every suffix implementation is broken. A copied proposal can be arbitrary, but accepted tokens must obey the same target distribution and grammar state as ordinary generation. The fork audit is the decisive immediate evidence.

**Certified target-head pruning was screened without an admissible speed candidate.** A coarse shortlist followed by exact dot products does not certify omitted vocabulary scores. The retrieved exact-real certification example (`brian-naughton/certified-inference`) concerns a tiny model and formal certificates, not a demonstrated 5090 large-vocabulary fast head. No practical exact-target speed claim survived this search. A future candidate must bound every omitted logit (including relevant rounding/penalties) or fall back to the full head; full stochastic softmax/top-p imposes more than greedy argmax preservation. NInfer's approximate draft head is a different use because target acceptance can reject its proposals.

## Independent review of the leading NInfer fork candidates

Inspected Wallawalla47/ninfer-custom at `600d8ac`, using `/tmp/ninfer-research/local-engines/Wallawalla47-ninfer-custom` supplied by the local-engine research branch. Full benchmark/context discussion is in `local-engines.md`.

**Copy proposals:** `src/models/qwen3_5/program/decode.cpp:753` sets one-hot q for the copied token. `src/ops/kernel/speculative_round.cuh:178–244` actually tests p/q and samples from positive p−q at rejection; this is not merely an author claim. For deterministic draft d, accepting with p(d) and otherwise sampling p on tokens other than d recovers p. However, its `include/ninfer/ops/sampling.h:27` explicitly limits runtime top-k to 1..20, and `sampling_device.cuh:148` caps the candidate set. Its one-warp residual is consistent with that old bounded sampler, not evidence for current full-support/p-less or grammar behavior. Current NInfer needs the proposer idea wired into its existing semantic acceptance/state boundary. C1 can skip neural drafting on a match; its mixed C>1 path still runs the drafter and overlays copy rows, so “eliminates the draft bandwidth” does not apply to every batch.

**PDL:** `src/core/pdl.cuh` enables programmatic dependency only during graph capture, waits before consuming producer-dependent data, and delays downstream trigger until the streaming weight loop finishes. This is materially different from blindly launching two DRAM-bound projections together. One implementation caveat: its `sm_count()` reads PTX `%nsmid`, which is an upper bound on SM identifiers and can exceed the physical SM count; NVIDIA PTX explicitly says numbering need not be contiguous. The threshold therefore is not guaranteed to equal half the physical SMs. This is a performance-heuristic caveat, not a demonstrated numerical ordering failure. Timing claims still require matched work/round counts because the fork's output throughput and per-round improvements need not move together.

This avenue is complete as a research pass: primary code or primary papers were inspected for the concrete candidates, differences from existing NInfer were established, and unresolved performance/quality claims are explicitly future experiments rather than hidden implementation tasks.

## Source ledger

All accessed 2026-09-26. C = implementation inspected; P = primary paper or project documentation; M = author measurements not reproduced; I = primary issue/PR evidence. Stable IDs are local to this campaign branch.

| ID | Date / version | Level | Source |
|---|---|---|---|
| KS01 | 78c6e1fbfcb6, 2026-09-26 | C | https://github.com/flashinfer-ai/flashinfer/blob/main/flashinfer/gemm/kernels/dense_blockscaled_gemm_sm120_b12x.py |
| KS02 | opened 2026-05-24, closed at cutoff | I | https://github.com/flashinfer-ai/flashinfer/issues/3398 |
| KS03 | retrieved cutoff | I | https://github.com/NVIDIA/cutlass/issues/3096 |
| KS04 | retrieved master | C/P | https://github.com/IST-DASLab/marlin — original design, not a 5090 NVFP4 benchmark |
| KS05 | retrieved main | P | https://github.com/HanGuo97/flute |
| KS06 | retrieved main | C | https://github.com/Dao-AILab/flash-attention/blob/main/flash_attn/cute/flash_fwd_sm120.py |
| KS07 | 2025-05-16 | P | https://arxiv.org/abs/2505.11594 — SageAttention3 |
| KS08 | retrieved cutoff | M | https://github.com/qu0b/fa4-sm120-research |
| KS09 | retrieved main | P | https://github.com/hikarioyama/vllm-nvfp4-kv-sm120/blob/main/docs/DESIGN.md |
| KS10 | 2025-04-28 | P | https://arxiv.org/abs/2504.19874 — TurboQuant |
| KS11 | retrieved main | P | https://github.com/z-lab/dflash — current compatible model inventory |
| KS12 | retrieved main | P | https://github.com/SafeAILab/EAGLE/blob/main/README.md — training requirements |
| KS13 | retrieved main | P | https://github.com/sgl-project/sglang/blob/main/docs/docs/advanced_features/adaptive_speculative_decoding.mdx |
| KS14 | 2025-12 | P | https://arxiv.org/abs/2512.22420 — Nightjar |
| KS15 | 2026-03-02 | P | https://arxiv.org/abs/2603.01639 — Learning to Draft |
| KS16 | retrieved main | C | https://github.com/vllm-project/vllm/blob/main/vllm/v1/spec_decode/ngram_proposer.py |
| KS17 | RFC at cutoff | I/M | https://github.com/vllm-project/vllm/issues/51788 — L20 suffix-GPU, not 5090 |
| KS18 | retrieved main | P | https://github.com/vllm-project/speculators/blob/main/docs/user_guide/algorithms/dspark.md |
| KS19 | retrieved main | P/C | https://github.com/vllm-project/speculators/blob/main/docs/user_guide/algorithms/dflash.md and https://github.com/vllm-project/vllm/blob/main/vllm/v1/worker/gpu/spec_decode/dflash/speculator.py |
| KS20 | 2026-09-18 | M | https://jared-hpc.com/posts/nvfp4-qwen38-27b-rtx5090/ — further discussed as Q09 in quantization-quality.md |
| KS21 | 2026-06-26 v2 | P | https://arxiv.org/abs/2605.06675 — codec-specific mixed-precision KV allocation |
| KS22 | merged 2026-04-16 | C/I | https://github.com/InternLM/lmdeploy/pull/4510 — API patch inspected, TurboQuant integration |
| KS23 | opened 2025-09-16 | I | https://github.com/snowflakedb/ArcticInference/issues/183 — grammar termination failure report |
| KS24 | retrieved cutoff | P | https://github.com/brian-naughton/certified-inference — screened out as a fast large-model head candidate |
| KS25 | 600d8ac, 2026-09-26 | C | https://github.com/Wallawalla47/ninfer-custom — independent copy-proposal/PDL review |
| KS26 | PTX ISA8.8, retrieved cutoff | P | https://docs.nvidia.com/cuda/pdf/ptx_isa_8.8.pdf — `%nsmid` identifier-bound semantics |
