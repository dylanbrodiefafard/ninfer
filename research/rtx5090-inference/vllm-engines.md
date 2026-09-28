# vLLM, Radiance, SGLang and TensorRT-LLM research

> Follow-up depth audit: `competitive-depth-audit.md` adds SparkInfer, contradictory copy/PDL evidence, current PR patches, further Radiance descendants and corrected stale GDN claims. The original breadth closure below is a first-pass boundary, superseded by that expanded investigation.

Research date: 2026-09-26. Scope: techniques useful to NInfer's single RTX 5090, Qwen3.8-27B NVFP4, C=1..4 runtime. This is source research, not a reproduction of third-party measurements. URLs below preserve provenance; a shipped repository, an open PR and a performance claim are distinguished explicitly.

## Baseline and decision

NInfer already has CUDA Graphs, compact concurrency, NVFP4 KV, MTP/DFlash2, adaptive draft widths, fused GDN record/fold, weight replay across requests, an optimized draft proposal head, prefix retention and tool-schema constrained generation. Its current evidence is in `docs/performance.md`, particularly the September 5–6 C=1..4 and W=4/5/6 campaigns, not the older README Qwen3.6/C8 headline. September 11 selective-FP8 measurements report C1 target-only 83.15 tok/s and C2 DFlash5 373.63 aggregate tok/s on that particular fixed corpus. Other documented workloads produce very different rates. None of the sources examined establishes a controlled speed lead over current NInfer.

The strongest remaining leads from this branch are SM120 small-M GEMM tile selection, execution-context-sensitive attention performance, and precision-aware checkpoint/kernel selection. Several eye-catching features are already present in NInfer or violate its current contract.

## 1. vLLM: identify the actual arithmetic, not the quantization label

Primary sources:

- https://github.com/vllm-project/vllm/issues/47749 (July 2026, maintainer explanation)
- https://github.com/vllm-project/vllm/issues/45260 (SM120 architecture coverage tracker)
- https://github.com/vllm-project/vllm/releases

Issue 47749 resolves an important ambiguity: a weight-only NVFP4 layer without activation scales intentionally selects Marlin W4A16; W4A4 layers can select native CUTLASS FP4 on SM120. The misleading warning about absent hardware support does not prove native FP4 is unavailable. A checkpoint's name alone therefore cannot establish its instruction path, quality or speed. For NInfer, compare represented weights and activation precision per layer before interpreting another engine's token rate. NInfer's existing A16 decode/W4A4 prefill distinction is deliberate and already exploits this choice.

The coverage tracker is useful routing, not a benchmark: it separates dense GEMM, grouped MoE, FlashMLA and build targets. B200/SM100 support cannot be inferred to cover SM120. Qwen4 MoE investigations should inspect the exact grouped-kernel route separately from dense 27B results.

## 2. Direct single-5090 competitor: seanyourhighness NVFP4 DFlash2 overlay

Sources inspected:

- https://github.com/seanyourhighness/vllm-sm12x-nvfp4-dflash2
- https://github.com/seanyourhighness/vllm-sm12x-nvfp4-dflash2/blob/main/BENCHMARKS.md
- https://www.reddit.com/r/Vllm/comments/1vy9cqt/ran_qwen3827b_on_a_single_5090_with_nvfp4_weights/

The public recipe combines vLLM 0.27.1, FlashInfer 0.6.16.post3 with patches, ModelOpt NVFP4 target/draft, NVFP4 KV and K7 DFlash2. The Reddit report's 616 aggregate tok/s is C4, approximately 65 prompt tokens and 1,536-token predictable Python outputs; prefix reuse reduces repeated prefill. Its 262K maximum context and 325K pool are capacity figures, not the context occupied during that speed test. This is useful competitor evidence, not a 262K-context throughput measurement or a matched comparison to NInfer's AIME workloads. Optional CPU vision sidecar is outside NInfer's GPU-compute contract.

### 2.1 Masked XQA and isolated CUDA stream

Source: https://github.com/vllm-project/vllm/pull/53543 (open on retrieval; September 26 rebase request).
Code anchors: `vllm/v1/attention/backends/flashinfer.py`, `vllm/envs.py`; GPU test `test_flashinfer_xqa_nvfp4_spec_decode_with_baseline`.

The patch uses BF16/model-dtype query and output with NVFP4 KV, enables the packed causal-mask speculative XQA path, and optionally forks/joins an isolated CUDA stream during graph capture. Reported integrated call time falls 1.437→0.278 ms, while standalone replay of the same tensors is still 0.053–0.072 ms. The test geometry is Q24/KV4/D256, q_len=8 on RTX 5090. End-to-end numbers belong to the 0.27.1 backport, not the main-tree PR.

Applicability: a conditional diagnostic, not a general recommendation to add streams. First determine whether NInfer has a similar isolated-versus-integrated gap at identical geometry. Its own attention implementation and graph schedule may not exhibit the problem. Explicit dependencies, graph-capture lifetime and simultaneous-request isolation must remain correct; the PR does not establish the root hardware cause.

### 2.2 Runtime speculative width must propagate into GDN state metadata

Source: https://github.com/vllm-project/vllm/pull/53542 (open; August 24).
Code anchor: `vllm/v1/attention/backends/gdn_attn.py`, `GDNAttentionMetadataBuilder`.

A max-K7 graph could expose eight state columns even when runtime K3 refreshed only four. Narrowing the returned view yielded reported C8 212.17→298.91 aggregate tok/s on a 5090 backport. Later Intel-XPU comments identify an additional contiguous-buffer requirement under full capture. This is a useful concrete failure mode for adaptive K, but C8 lies outside NInfer's contract and NInfer already has width-specific replay/graph selection. Do not count the vLLM fix as an unimplemented NInfer feature.

### 2.3 Amortize NVFP4 cache conversion per attention tile

Source: https://github.com/flashinfer-ai/flashinfer/pull/4346 (open; August 4).
Code anchors: `include/flashinfer/attention/prefill.cuh`, `SharedStorage`, paged/ragged/single prefill dispatch; `tests/attention/test_batch_prefill_kernels.py`.

This patch repacks NVFP4 K/V into shared-memory BF16/FP16 once per tile, avoiding repeated conversion/scaling inside MMA iterations. Correct BF16-bit conversion and shared-memory accounting were material fixes. The PR's controlled result is 0.2999→0.2888 ms (3.7%) on RTX PRO 4000 Blackwell, B4/Q128/KV2048/Q32/KV8/D128, CUDA 13.0.88. The recipe's separate 10–12% claim is not this measurement and must not replace it.

Applicability after code cross-review: substantially already implemented in the primary NInfer dense NVFP4 prefill route. `src/ops/kernel/gqa_attention_prefill_nvfp4.cuh:432–451` uses dedicated V worker warps to decode each V tile into shared `v_bf16` before the PV MMA loop; QK at line 361 uses native NVFP4 MMA. No missing optimization was established here. Another consumer would need demonstrated repeated conversion before this becomes a live candidate. Long-context decode also already double-stages KV and rejected a four-warp variant; this source does not justify repeating that experiment.

## 3. Exact-model multi-GPU study: adrienbrault/qwen3.8-27b-rtx5090

Sources:

- https://github.com/adrienbrault/qwen3.8-27b-rtx5090
- https://github.com/adrienbrault/qwen3.8-27b-rtx5090/blob/main/bench/results/r183b-nvfp4-gemm-kernels.md
- https://github.com/adrienbrault/qwen3.8-27b-rtx5090/blob/main/scripts/r183b-kernels.sh
- https://github.com/adrienbrault/qwen3.8-27b-rtx5090/blob/main/bench/results/r204-upstream-picks-gate.md

R183b, September 4, walks the actual NVFP4 backend ladder by disabling individual kernel classes rather than globally forcing every mixed-precision layer to one backend. On **two** 5090s, Marlin changes dense-corpus perplexity delta from +0.744% to +0.207% against BF16, but loses 7.8% C8 and 17.3% C16 decode and adds 32% to 100K TTFT. Five fusion configurations did not establish a throughput win beyond replicate variation. Single-stream values were especially noisy because acceptance varied. The report is valuable for experimental design and W4A16/W4A4 quality tradeoffs; its rates are not single-GPU NInfer comparisons.

R204, September 6, tests FlashInfer's SM120 GDN prefill enablement against Triton/FLA on that same serving stack. Prefill across 2K–131K is essentially unchanged despite changed numerical behavior, so the new backend was not adopted. A faster-looking kernel name is not evidence that the operator controls request latency.

The repository's current default is TP2, C16, with changed NVIDIA weights and memory-clock overclocking. Its cached BF16 GDN state differs from NInfer's specified FP32 persistent state. Its multi-GPU all-reduce, host embedding offload, C64 saturation and disk-tier numbers must be filtered out of single-GPU algorithm recommendations.

## 4. Radiance identity and transferable mechanisms

Sources:

- https://github.com/GGZ14/vllm-mxfp4
- https://codeberg.org/ggz14/radiance-vllm-mxfp4
- https://github.com/GGZ14/vllm-mxfp4/blob/main/radiance_mxfp4_fp8.hip
- https://github.com/GGZ14/vllm-mxfp4/blob/main/radiance_gdn_lazy.py
- https://github.com/GGZ14/vllm-mxfp4/blob/main/radiance_verifyhead.py

The searched Radiance deployment is an AMD RDNA4/gfx1201 vLLM stack, historically distributed as `stilldeadcode/vllm-radiance`, with GGZ14's MXFP4 extension. It is not a CUDA RTX 5090 engine. Current source uses handwritten HIP W4A8 WMMA, separate prefill/decode tiles and a last-arrival split-K reduction. Its on-load NVFP4→MXFP4 conversion is lossy and directly conflicts with NInfer's no-runtime-weight-repacking rule; do not copy that behavior. The broad lesson is small-M specialization and activation-precision tradeoffs, which NInfer already practices, not a portable AMD speedup.

`radiance_gdn_lazy.py` stores candidate inputs and materializes only an accepted recurrent prefix, reducing vLLM's K7 state allocation from nine to three pages/request. **NInfer already implements the substantive mechanism.** Its family authority describes raw records plus an all-layer Fold, FP32 state, `2C+1` complete slots with speculation and C width-W record rows. Its turn checkpoints serve a different purpose from Radiance's vLLM page count, so those numbers cannot support a claimed NInfer capacity reduction.

`radiance_verifyhead.py` reuses an int2 coarse head and exact reranking for the **target** head, gated by sampling configuration. Its explanation calls greedy safe but relies on candidate recall rather than a mathematical bound proving that the true argmax/top-k is present. Exact scores inside an approximate shortlist do not make omitted candidates irrelevant. This is a correctness concern, not a suggested NInfer optimization. Draft-only approximation is a different case because target verification can reject it; NInfer already has an optimized proposal head. A target approximation would need an explicit changed semantic contract or a certified bound/fallback.

## 5. SGLang: useful upstream mechanisms, architecture-specific evidence

Sources:

- https://github.com/sgl-project/sglang/pull/26496 (merged June 4)
- https://github.com/flashinfer-ai/flashinfer/pull/3152 (CUTLASS small-N and SwapAB link from merged change)
- https://github.com/sgl-project/sglang/blob/main/docs/docs/advanced_features/quantized_kv_cache.mdx
- https://github.com/sgl-project/sglang/releases

The merged SM120 default change returns NVFP4 GEMM to FlashInfer CUTLASS after fixing a PDL issue, using SwapAB and smaller N tiles. It reports Qwen3.6-27B 40.82→48.05 tok/s, without enough workload detail for cross-engine ranking. FlashInfer PR3152 itself merged May 5; its reported 1.13–1.27× C1..4 gains are Qwen3-30B-A3B on DGX Spark, not RTX 5090. The mechanism is relevant to C1..4 narrow matrices; inspect the exact shape and register/shared-memory tradeoff rather than importing a backend wholesale.

Current quantized-KV documentation distinguishes SM100 direct NVFP4 prefill from SM120's automatic FP8 dequantization prefill workspace and architecture-specific XQA BF16 query/output path. Native NVFP4 decode exists in both recipes. Thus NVFP4 storage does not establish FP4 attention arithmetic or absence of a temporary workspace. NInfer's own codec/layout contract remains authoritative.

Recent releases include DFlash checkpoint-boundary tracking, lazy Mamba buffers, graph memory reuse, separate request/token graph capacities, and mask computation under overlap scheduling. These are concrete avenues for implementation comparison, not direct evidence of a 5090/Qwen3.8 speed gain. NInfer already has stable slot ownership, replay state and grammar integration; only a measured bottleneck justifies replacing them.

Qwen4-relevant warning: https://github.com/sgl-project/sglang/issues/36531 documents QSA backend selection and NVFP4 MoE/TMA failures on SM120, including a failed four-5090 configuration and PLE offload. A successful 96GB RTX PRO 6000 run does not imply 32GB admission. This reinforces the existing artifact-capacity and source-correctness contract.

## 6. TensorRT-LLM: kernel source versus product support

Sources:

- https://github.com/NVIDIA/TensorRT-LLM/blob/main/docs/source/supported-hardware.md
- https://nvidia.github.io/TensorRT-LLM/latest/features/quantization.html
- https://github.com/NVIDIA/TensorRT-LLM/discussions/8334

The current official supported-hardware file names B200/GB200/B300/GB300/DGX Spark, Hopper, L20/L40 and A100; it does **not** list RTX 5090. This is not proof that no subset runs: the community discussion links an RTX recipe. It does mean blanket supported-5090 claims are unjustified. The current quantization documentation describes active NVFP4 KV requiring offline ModelOpt quantization and separately compressed cold host/disk pages. Do not confuse storage-tier compression with active attention precision.

TRT-derived XQA kernels exposed through FlashInfer are directly relevant even when the full product/model stack is not. No matched exact-supported-NInfer-artifact, C1..4 single-5090 TensorRT-LLM result was established in this investigation. Multi-GPU disaggregation, high-concurrency scheduling and other-model MoE results are excluded as speed evidence.

## 7. Recipes and wrappers: useful operational evidence, not extra engines

- https://github.com/p4u/rtx5090-vllm — shell/UI/container recipes selecting vLLM and a separate Prism llama.cpp route. Its Qwen3.8 profile rates are far lower than NInfer's documented comparable-size target-only rates, but use different weights/settings and are not a controlled ranking. Its list is valuable for boot/capacity failure cases. The ternary checkpoint changes the model/forward contract and is not a drop-in optimization of NInfer's artifact.
- https://github.com/infernet-org/foundry — tuned Docker profiles and evaluation/monitoring around vLLM. Its headline 384 single / 1,228 C4 belongs to Qwen3.6-35B-A3B MoE. The dense Qwen3.8 Unsloth profile uses a substantially larger mixed-precision footprint and eager mode on 32GB. These are not dense-27B speed records. Prometheus latency, acceptance and GPU telemetry are potential product usability ideas, conditional on actual missing observability.

## Disposition and remaining decision questions

| Candidate | Disposition for NInfer | Decisive next evidence if selected for implementation |
|---|---|---|
| Small-N / SwapAB / exact narrow-M tile choice | Relevant kernel comparison | Classifier admission and operator oracle at NInfer's actual W,C matrix geometries, then matched Engine A/B |
| Per-tile NVFP4 K/V conversion reuse | Already present in primary dense prefill | Reopen only for a different consumer with demonstrated repeated conversion |
| Isolated attention stream | Conditional diagnostic | Same tensors, kernel, cache state and graph topology in isolated versus integrated runs |
| W4A16 versus W4A4 selective arithmetic | Quality/performance tradeoff; partly already implemented | Fixed artifact and represented-input oracle plus task quality at the affected phases |
| Lazy speculative GDN records | Already implemented | No new feature recommendation |
| Approximate target vocabulary shortlist | Reject as an exactness claim | Certified recall bound/fallback or explicit changed output semantics |
| Runtime-K metadata discipline | Existing architectural concern | Compare actual active widths and graph identity; not a generic rewrite |
| BF16/FP16 recurrent cache | Changes specified FP32 state | Requires explicit semantic redesign and long-state oracle/quality qualification |
| Radiance HIP, TP2/TP64, CPU vision, ordinary host embeddings | Outside current target or contract | Do not port as runtime architecture |

No source in this branch justifies a direct imported replacement of NInfer. The campaign identifies concrete techniques and rules out misleading comparisons; experiments listed above are proposed implementation gates, not unfinished research actions or performance claims.

## 8. Offload systems screened, without expanding the native contract

FreeToken (https://github.com/FlashML-org/FreeToken; paper https://arxiv.org/abs/2608.16157) implements bandwidth-adaptive CPU/GPU co-execution, double-buffered full-layer prefill streaming, an expert LRU cache and elastic KV/expert VRAM allocation. Its semantic anchor checkpoints for context edits are conceptually relevant to agentic reuse, but NInfer already owns thinking-aware turn checkpoints and replay. CPU computation and ordinary weight streaming are outside the native contract. The Qwen4 diagnostic exception is precisely bounded and does not license FreeToken's general policy.

KTransformers (https://github.com/kvcache-ai/ktransformers; `kt-kernel/README.md`) is explicitly CPU/GPU heterogeneous MoE execution, including AMX/AVX CPU experts. It answers an oversized-model capacity question rather than resident dense-27B speed. No CPU backend or generic heterogeneous scheduler is recommended.

SGLang HiCache (https://docs.sglang.io/docs/advanced_features/hicache_design) supplies a precise transfer lesson: distinguish GPU layer-first compute layout from page-oriented I/O, aggregate page-layer copies and overlap loading layer N+1 with computing layer N. Its best-effort/wait/timeout policies trade prefix restoration against recomputation latency. NInfer already measured packed-page copy/scatter at 97.1% of pinned H2D bandwidth and qualified reader/CRC pipelines. This is not an unfilled blanket cache feature. In particular, Qwen4 host/disk KV tiers are explicitly unsupported and must still fail admission; GPU prefix retention remains in scope.

## Source ledger

All entries retrieved 2026-09-26. Evidence levels: **C** inspected implementation; **P** primary project documentation/PR with author measurements (not independently reproduced); **R** community report; **L** local authority. IDs are stable within this report. A URL's presence does not mean every linked artifact was downloaded or every claim accepted.

| ID | Source / code | Level | Date or version | Finding / limitation |
|---|---|---|---|---|
| VE01 | vLLM issue 47749 | P | July 2026 | W4A16 dispatch intentional, warning misleading |
| VE02 | vLLM issue 45260 and releases | P | live cutoff | SM120 subsystem coverage map |
| VE03 | seanyourhighness repo README/BENCHMARKS | C/P | v0.27.1 overlay | exact-model single-5090 recipe, different artifact/workload |
| VE04 | Reddit 1vy9cqt | R | August 2026 | 616 C4 on short prompts, context capacity separate |
| VE05 | vLLM PR 53543, flashinfer backend | P | open cutoff | masked XQA + isolated stream, root cause unresolved |
| VE06 | vLLM PR 53542, GDN metadata | P | open cutoff | active width bug and reported C8 gain |
| VE07 | FlashInfer PR 4346, prefill.cuh | P | open cutoff | once-per-tile repack, 3.7% on PRO 4000 |
| VE08 | adrienbrault R183b report/script | C/P | 2026-09-04 | TP2 backend ladder and fidelity tradeoff |
| VE09 | adrienbrault R204 report | P | 2026-09-06 | GDN prefill backend neutral E2E |
| VE10 | Radiance HIP GEMM | C/P | retrieved main | AMD small-M schedule; not CUDA |
| VE11 | Radiance radiance_gdn_lazy.py | C | retrieved main | replay already substantially present in NInfer |
| VE12 | Radiance radiance_verifyhead.py | C | retrieved main | candidate omission not proven safe |
| VE13 | SGLang PR 26496 / FlashInfer 3152 | P | merged 2026-06-04 | SwapAB/small-N default selection |
| VE14 | SGLang quantized_kv_cache.mdx | P | retrieved main | SM100 versus SM120 routes differ |
| VE15 | SGLang releases / issue 36531 | P | live cutoff | cache/graph improvements; Qwen4 capacity/backend failures |
| VE16 | TRT-LLM supported-hardware.md | C/P | retrieved main | RTX 5090 absent from official list |
| VE17 | TRT-LLM quantization docs/discussion 8334 | P/R | live cutoff | active versus cold NVFP4 and community recipe |
| VE18 | p4u/rtx5090-vllm | C/P | retrieved main | recipe wrapper, changed ternary model excluded |
| VE19 | infernet-org/foundry | C/P | retrieved main | headline MoE not dense-27B; eager dense profile |
| VE20 | FreeToken repo/paper | P | Aug 2026 / cutoff | heterogeneous/streaming outside native contract |
| VE21 | KTransformers kt-kernel | P | cutoff | CPU expert execution excluded |
| VE22 | SGLang HiCache design | P | cutoff | page-layer transfer and wait policies, existing NInfer overlap |
| VE23 | NInfer performance.md and qwen3.6-27b-model.md | L | local checkout | current baseline, FP32 state and record/fold ownership |
