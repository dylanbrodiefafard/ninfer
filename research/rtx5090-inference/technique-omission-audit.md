# Technique omission audit

Retrieved 2026-09-26. This extends technique-depth-audit.md with additional primary papers, code and negative evidence. No GPU experiments, model downloads, installations or implementations were performed. Published results below remain publisher measurements. The search cannot establish an exhaustive census of private, unindexed or future work.

## New mechanisms and dispositions

### TO01 — DAMP quantizes recurrent state, not just weights

Primary: https://arxiv.org/abs/2608.27513 ; https://arxiv.org/html/2608.27513v1 (August 27). Paper-level evidence; no independently inspected implementation.

DAMP preserves 16 of 128 key channels at FP16 and combines INT8 remaining channels with Hadamard preprocessing. Its effective 9.875-bit state includes metadata; the paper reports 69.1% state-memory reduction, up to 2.01x kernel acceleration and 10.9% TPOT improvement. It studies Qwen3.6-35B-A3B and Kimi Linear, rather than this exact 27B artifact. Its large-batch state-memory motivation cannot be transferred to C=1..4. Uniform INT8/FP8 state can degrade reasoning and INT4/NVFP4 state is particularly damaging in the reported evaluations.

This is distinct from Minima's low-bit matrix arithmetic: NInfer's persistent GDN state is explicitly FP32. Adoption requires an explicit state-precision contract change, calibration and quality admission. It is not an allowed private transient precision optimization. Keep as a capacity/quality research branch, below exact prefill fusion and current-sampler-preserving schedule work.

### TO02 — Approximate triangular inversion is a different GDN prefill tradeoff

Primary: https://arxiv.org/abs/2606.06034 ; https://arxiv.org/html/2606.06034v1 (June 4), “When Good Enough Is Optimal: Multiplication-Only Matrix Inversion Approximation for Quantized Gated DeltaNet.” Paper evidence.

The NPU-focused Qwen3.5-4B study replaces triangular solves with truncated Neumann terms, structured masks and residual corrections; it reports up to 5x kernel and 20% layer improvement. Some parameter choices produce enormous perplexity or NaNs, whereas its selected configuration recovers the reported baseline. Strictly lower-triangular matrices admit a finite exact polynomial, but truncating that polynomial is not automatically exact.

NInfer `src/ops/linear_attention/gated_delta_net/chunked/prepare_wy_wu.cuh:224` performs block forward substitution, invoked for four diagonal blocks around line 714. This is a real alternative algorithm family, but its NPU timings and approximation do not establish an SM120 gain. Investigate the exact/non-CP FlashQLA fusion first. Numerical qualification must target the independent recurrence oracle and persistent-state error, not only short-answer agreement.

### TO03 — Discrete block verification versus an irrelevant diffusion result

Primary: https://arxiv.org/abs/2403.10444 (Block Verification Accelerates Speculative Decoding). Publisher reports roughly 5–8% acceleration. This changes acceptance coupling, not merely the already-parallel target forward pass.

Current NInfer `src/ops/kernel/speculative_round.cuh:73–84` uses no stochastic selector distribution for p-less: the proposal law is a point mass. For a deterministic candidate path, accepting its first k tokens cannot have probability greater than the target probability of that prefix while preserving the target law. Ordinary sequential acceptance already attains that product of conditional probabilities. This is a mathematical applicability limit, not a local performance result: do not presume stochastic block-verification gains for this deterministic proposal path. Non-p-less stochastic drafting would need its own analysis of actual p/q, grammar masks, residual sampling, EOS and rollback.

The newer https://arxiv.org/abs/2606.13426 (“Accelerating Speculative Diffusions via Block Verification,” June 11) is **not** supporting evidence for DFlash token verification. Its FreeDrafter evaluates continuous Gaussian image diffusion with Euler–Maruyama steps. It was screened out after reading the paper rather than carrying its headline speedup into language inference.

### TO04 — JetSpec parallel tree drafting requires a matching companion

Primary: https://github.com/hao-ai-lab/JetSpec ; https://arxiv.org/abs/2606.18394 . Repository/paper screen.

JetSpec builds causal parallel draft trees and publishes a Qwen3-8B companion checkpoint. Its B200 BF16 examples use greedy sampling and large trees (depth 20, width 7, budget 128); reported MATH throughput/acceptance does not qualify a 27B NVFP4 companion or stochastic p-less. NInfer already has tree verification and ReplaySSM, so the missing ingredient would be trained proposal quality and a useful tree-cost/acceptance tradeoff, not generic tree support. Capture-layer/representation compatibility, actual target sampling law and C=1..4 memory/verification costs remain gates. This extends the draft-training family already covered rather than displacing immediately actionable exact scheduling candidates.

### TO05 — AutoMegaKernel: inspect the denominator before adopting a persistent VM

Primary: https://arxiv.org/abs/2606.09682 ; https://github.com/RightNow-AI/AutoMegaKernel . Inspected source under `/tmp/ninfer-research/kernels/AutoMegaKernel`.

The advertised 1.19–1.23x 5090 comparison is W8A16 versus CUDA-graphed cuBLAS BF16, not identical arithmetic against an existing NVFP4 engine. Equal-BF16 results can be slower. Decode-position-zero and small/synthetic model experiments are not a long-context resident 27B result. The repository's hardware naming also includes “RTX 5090 Laptop (local)”; do not silently treat all receipts as the desktop 32GB target.

`vm/scheduler.cu` uses cooperative-grid synchronization; `vm/sync.cuh` implements instruction counters and acquire polling/backoff. A whole-model VM changes the project architecture and requires resident-block scheduling. Narrow producer/consumer scheduling remains worth understanding, but this paper does not override current NInfer classifier refusals or show that saving small intermediates offsets large weight traffic. No new priority assigned from its headline.

### TO06 — Exact image-feature memoization is narrower than approximate multimodal KV reuse

Primary: https://docs.vllm.ai/en/latest/features/cross_encoder_cache/ ; https://arxiv.org/abs/2606.23581 (Kamera, June 22); https://arxiv.org/abs/2512.12977 (VLCache); https://arxiv.org/abs/2602.01901 (QCache).

Exact encoder-output reuse is a plausible feature when the same image appears behind a changed text prefix or after prefix-retention misses. NInfer `vision_context_impl.h:362–390` binds a session to prepared prompt/plan/transient storage; lines 535–560 reuse an active item or aggregated batch within that session and otherwise call `context_.encode`. This establishes the inspected session behavior, not an assertion that every cache in the repository was absent. Exact prefix hits can already avoid re-encoding. Root's completeness-audit CA02 independently inspects a concrete QW3 cross-request image-feature cache.

Kamera's position-independent KV re-RoPE and low-rank conditioning patch, VLCache's heavy token reuse, and QCache's query/layer reuse change decoder computation. Same image bytes alone cannot prove equivalent decoder KV or GDN history under a different prefix. Screen these as optional approximate methods. A bounded exact feature cache would need artifact, preprocessing/grid and content equality in identity, ownership and capacity accounting; profile repeated-image misses before implementation. Distributed encoder-cache infrastructure is outside this single-resident product.

### TO07 — Exact selection does not imply exact vocabulary-head pruning

Primary: https://arxiv.org/abs/2609.08450 ; https://github.com/Tencent/hpc-ops (September 8). Paper-level mechanism screen.

HPC-Ops proposes sampled coarse boundaries followed by complete-row certification, exact FP32 refinement and underfill recovery for top-k selection. Its sparse-attention/indexer operator results (roughly 1.29–1.75x) are not a 5090 p-less result. The complete-score certification matters: this does not provide a bound that permits skipping arbitrary vocabulary projection rows. Qrita (https://arxiv.org/abs/2602.01518) similarly targets top-k/top-p rather than replacing p-less's global moments and actual threshold law. Neither warrants claiming exact output-head pruning.

### TO08 — New CUDA releases require qualification, not presumed speed

Primary: https://docs.nvidia.com/cuda/archive/13.3.0/cuda-toolkit-release-notes/index.html ; https://docs.nvidia.com/cuda/archive/13.2.2/cuda-toolkit-release-notes/index.html . Official release-note evidence.

The nested-reconvergence compiler register issue introduced in CUDA 12.8 is documented as fixed in 13.2 Update 2/13.3. This is a concrete reason to check affected code generation, not evidence that current NInfer hits the bug. cuBLAS release fixes/known issues include NVFP4 scaling and PDL-related dependencies; project-owned kernels do not automatically inherit either the performance or every library bug. No toolchain changes were made.

`setmaxnreg` is already used by NInfer in `src/ops/linear/nvfp4/nvfp4_w4a4_tma.cuh:223,258`; rediscovering the instruction in a recent microarchitecture article is not a new candidate. Architecture, toolchain acceptance and actual emitted instructions must be distinguished.

### TO09 — Autotuning evidence must match graphs and cache state

Primary: https://flashinfer.ai/2026/09/22/autotuner-v2.html (September 22). Author engineering article.

Autotuner v2 treats eager versus CUDA Graph execution and cold-cache preparation as material experimental conditions. Its L2 eviction work stays outside the timed GPU region and the default candidate remains in comparisons. This reinforces existing kdev qualification rather than creates a need for a general runtime tuner. A shape win from hot standalone replay is not necessarily a cold-weight model win; use deployed graph/cache conditions and keep tuning outside the request path unless separately justified.

### TO10 — Field-guide negatives prevent misleading low-bit recommendations

Primary: https://github.com/notwitcheer/sm120-field-guide ; https://github.com/notwitcheer/sm120-field-guide/blob/main/guide/quantization.md . Author notes, not independent reproduction.

The QuTLASS MXFP4 account reports large-M GEMM gains alongside much slower HF token throughput and increased memory (including apparent duplicate weight representations). This is evidence to account for rotation/quantization and integration costs, not proof that NVFP4 is generally slower. The regional 2-bit KV recall failure was a CPU-only, masked-GPU experiment on Qwen3-4B at about 5.7K context. It illustrates quality risk; it is not 5090 kernel throughput or evidence against the existing NVFP4 codec. Perplexity methodology comments do not replace this repository's supported evaluation authority.

### TO11 — Newly exposed imp branches: one conditional fusion and important negatives

Primary commits, inspected from https://github.com/kekzl/imp :

- https://github.com/kekzl/imp/commit/a90f50d96ab34e3ace4a4613a69c153d335a87f0 (September 25): GDN state-column splitting expands 48 heads to 192 CTAs and preloads state before the PDL wait. Reported alternating 27B decode runs improve around 97–98 tok/s. NInfer **already** splits state columns: `recurrent.cuh:13–15` has 16 columns/block and `recurrent.cu:26` uses 128/16 blocks/head, or 384 CTAs for 48 heads. Thus column splitting is not missing. Pre-dependency state fetch is a narrower potential mechanism, requiring proof that state is not produced by the predecessor and register/occupancy qualification; the commit itself reports a 255-register/stack-spill failed variant.
- https://github.com/kekzl/imp/commit/cbbea8d3391bf2dda56967b163b96a81a287aa15 (September 1): FP8 SSM prefill first overflowed because scaling followed FP16 materialization. The branch fixes that but explicitly records end-to-end benefit as refuted. Preserve this negative result rather than recommending FP8 from the branch name.
- https://github.com/kekzl/imp/commit/f11e98df9e969f3eaa3cc9bf2ec5f4b014db7242 (September 23): large scan gain concerns Mamba2/Nemotron register-state scanning, with separate gpt-oss RoPE optimization. It is not the Qwen GDN recurrence and its speedup is not transferable.
- https://github.com/kekzl/imp/commit/7f9005a3f9e9921680eac06cc5e6eaa5bace055e (August 26): gated norm emits NVFP4 activations for batched output projection, removing quantization launches/re-reads. NInfer `text_context_impl.h:1011–1014,1038–1043` materializes gated norm before `gdn_output_projection`; `qwen3_6_27b/impl/variant.cpp:477–496` dispatches linear-add. This is a concrete fusion boundary to classify **only for routes that actually consume quantized activations**: selective FP8 matrices and small-M BF16-input routes do not automatically benefit. The commit provides bit-identity tests but no isolated transferable end-to-end gain. A candidate must preserve the represented BF16/codec boundary where observable and be checked against the independent mathematical oracle, not merely another engine's unfused output.

## Search coverage and closure

Queries covered `Gated DeltaNet state quantization`, `Gated DeltaNet triangular inverse approximation`, `block verification speculative decoding`, `parallel tree drafting JetSpec`, `RTX 5090 megakernel`, `exact GPU sampling top k`, `vision encoder cache`, `multimodal KV reuse`, `CUDA 13.3 release notes`, `SM120 autotuning` and named primary leads from the other agents. They were followed into papers, official documentation or author code as cited above. Additional screens:

| Avenue | Disposition |
|---|---|
| Gated DeltaNet2 / changed gates | Architecture/checkpoint change, not a replacement kernel for the existing artifact |
| Apple Alloy/Prysm chunk algorithms | Other hardware; useful algorithm context, no SM120 result established |
| Trainium recurrence/inversion reports | Numerical-conditioning warning and platform mismatch, not a local speed result |
| FlashRT | Already inspected in local-engines; no distinct new mechanism established here |
| vLLM-Mach fused BA/conv/recurrence; MetaZenith normalized-conv prefill | Ecosystem agent owns the inspected code and qualified disposition; different overlap boundaries, not automatic add-on wins |
| KV virtualization/retrieval and new native engines | Root/local-agent reports own these; approximate retrieval distinguished from exact retention |
| FlashQLA non-CP | Prior report corrected: published receipt defaults auto-CP; no measured non-CP 339us claim remains |

These new sources broaden the technique map without establishing another measured same-artifact NInfer gain. Immediately actionable research remains phase attribution and exact-route comparisons from technique-depth-audit; approximate state/attention and new draft checkpoints have explicit additional admission gates. All material leads discovered in this pass now have an inspected, screened, duplicate or contract-gated disposition. This is a bounded research closure, not a promise of 100% Internet coverage.
