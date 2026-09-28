# Citation-graph expansion: kernels, state and speculative semantics

Retrieved 2026-09-26. This pass followed references, successor papers, author repositories and implementation PRs from the previous reports. It did not run GPU workloads, install software, download models or change the engine. “Exact” below always names its reference: preserving an actual sampler, a represented codec, or a specified recurrence is different from matching an unquantized model or a globally conditioned language distribution.

## Highest-value additions

### FT01 — b12x segment-parallel GDN with certified recurrence convergence

Primary code: https://github.com/local-inference-lab/b12x ; https://github.com/local-inference-lab/b12x/blob/main/docs/gdn-prefill.md . Inspected snapshot `a7d7d29`, September 24, under `/tmp/ninfer-research/kernels/b12x`.

**Lineage correction:** `local-inference-lab/sparkinfer` redirects to this b12x kernel library. It is distinct from `gittensor-ai-lab/sparkinfer`, the engine inspected in technique-depth-audit TD14 and competitive-depth-audit CD01. The previous TMA-cache observations cite the latter correctly; inspecting that clone did not constitute inspection of b12x.

The research-only GDN prefill API consumes post-convolution Q/K/V, keeps an FP32 state pool, and leaves output norm/gating external. Its segment-parallel route groups 128–1024 tokens into segments, internally retaining 16-token recurrence. Transfer and zero-initial-state summaries establish incoming states, followed by output correction. The interesting optimization is **certified convergence**: after 128 tokens, correction may reuse the remaining locally computed outputs only when every relevant FP32 state bit equals the saved local checkpoint. This is not FlashQLA's decay-threshold approximation.

Inspected `_shared/delta_prefill/_cute_kernels.py:1784–1810` compares `_f32_bits` and performs an all-lane vote; `_parallel_kernels.py:106–157` records exact-zero transfer and finite local-state conditions. The nonfinite safeguards matter because multiplying an infinite state by zero is not equivalent to discarding it. Scratch and convergence checkpoints are preplanned. Output reuse requires unsplit keys; the implementation refuses incompatible configurations near line 1907.

The certificate proves equality of the **remaining repeated local recurrence after equal state**, not blanket equality of all segment-summary arithmetic with NInfer's scalar oracle. BF16 matrix operands, centered transfer representation and changed reduction association still need numerical qualification. The repository reports oracle/graph coverage on Max-Q PRO 6000, with GB10/profile integration incomplete; no matched 5090 timing was established. This is a substantive longer-term comparator alongside non-CP FlashQLA, not a production-ready replacement or measured speedup.

Other b12x entries were screened: packed GDN decode has grouped request/state-index contracts and separate recovery/commit modules; NInfer already owns compact lanes and ReplaySSM. `gemm.blockscaled` distinguishes W4A16 from W4A4 and has split-K/shape policies rather than one universal kernel. Its Qwen Flash-Next QSA/PLE/HyperConnection references are explicitly not throughput-qualified and concern different model mathematics. Loader/GDS, RDMA, sparse MLA and MoE capacity features do not improve the current resident dense target merely by existing.

### FT02 — New llama.cpp prefill uses three-product BF16 reconstruction

Primary: https://github.com/ggml-org/llama.cpp/pull/29353 ; source `ggml/src/ggml-cuda/gated-delta-net-mma.cu` on that PR. Open September 24; still unmerged at retrieval. Files inspected via GitHub API under `/tmp/ninfer-research/final-technique`.

The kernel keeps FP32 recurrent state and splits operands into BF16 high/residual parts. `product` at lines 93–106 accumulates low×high, high×low and high×high; low×low is omitted. A 16-token block uses warp forward substitution to build its inverse, then tensor-core products for update/output. State stays in registers over chunks. This is a different useful precision/performance point from plain BF16 operands, not exact FP32 multiplication.

Publisher 5090 Qwen3.8-27B Q4_K_M prefill improves 3851.89→4301.52 tok/s at 2K and 3914.72→4322.87 at 4K, with F16 KV and 2048-token ubatches. Those are another engine/artifact's results. The PR acknowledges that speculative contexts currently disable the route even during prefill, due to its K==1 snapshot gate. Review also flags unsettled MMA swizzling. NInfer already has chunked tensor-core GDN; the transferable question is whether register-resident chunk fusion and two-component operands beat its existing WY/state/output schedule under its oracle and real shapes. The headline does not establish that comparison.

### FT03 — Kachua: finite Neumann solve is not truncated approximation

Discovery edge: https://arxiv.org/abs/2607.16831 → author references → https://github.com/romitjain/kachua-mlsys . Inspected clone `/tmp/ninfer-research/kernels/kachua`.

The MSInfer report explicitly distinguishes contest speedup against a simple reference from optimized FlashInfer; it also records a favorable local ratio reversing in official evaluation. Its references lead to Kachua, UW SyFI and LLM-CUDA, broadening the GDN algorithm comparison rather than furnishing a 5090 benchmark.

Kachua `gdn_prefill_qk4_v8_d128_k_last/solution/triton/kernel.py:275–377` applies the inverse of a strictly lower-triangular nilpotent matrix by repeated squaring through the finite required powers. This is algebraically exact in real arithmetic for its selected chunk size, unlike truncating an infinite-series approximation before nilpotence. Actual code uses `tl.dot(... input_precision="tf32")`, BF16 contractions elsewhere, and a compiler workaround. Its comment claiming approximately 19 TF32 mantissa bits must not be treated as a hardware precision guarantee: the actual instruction/profile and oracle decide error.

The dispatcher chooses chunk 16/32 and value tile size by sequence geometry and selects split-WY versus flat-WY versus direct recurrence. Its numerical notes report BF16 inversion drifting beyond contest tolerance on long sequences. B200-specific dispatch, Python output allocation and contest tolerance do not directly fit NInfer. This is a useful **exact finite algorithm candidate with precision gates**, separate from TO02's approximate NPU inversion. NInfer's current four 16×16 forward-substitution blocks are the comparison point, not an unchunked baseline.

Companion references screened: https://github.com/kamahori/mlsys-contest-syfi-agent-assisted (WY/WMMA and runtime interface); https://github.com/syhya/mlsys26-flashinfer-contest (shape dispatch and upstream Blackwell chunk path). Their runtime-FFI gains are largely inapplicable to an existing C++ graph engine; no independent 5090 result was inferred.

### FT04 — SGLang KDA: full-model cache interference reverses a kernel win

Primary: https://github.com/sgl-project/sglang/pull/36865 (merged September 2); originating https://github.com/BBuf/KDA-Pilot/pull/195 . PR metadata and source inspected.

The 1.319x geometric kernel headline concerns production shapes of Qwen3.5-4B/9B on PRO 6000, not all Qwen3.8 projections. The original Qwen3.8 policy retained 5.6–11 MiB weight-scale tensors per layer in L2. Broad isolated-kernel wins became a **0.76% full-model throughput regression**. Streaming the weights and scales and admitting only the DSpark M=9, K=17408, N=5120 down projection yielded a publisher 128.35→129.60 output tok/s (+0.98%), unchanged acceptance.

`qwen3x_nvfp4_gemm_sm120.py:1957–2052` retains activation/SFA and streams weight/SFB via distinct TMA cache policies. Production dispatch is deliberately narrower than the callable test API. This independently supports shape-specific cache-policy research and is negative evidence against preserving scales indiscriminately. It does not show NInfer lacks caching or that the fixed M=9 route corresponds to every C=1..4 speculative batch.

### FT05 — Skinny FP8 and one-warp B/A require the actual local route

Primary: https://github.com/sgl-project/sglang/pull/38082 (merged September 5); https://github.com/turboderp-org/exllamav3/pull/369 (closed, not merged).

SGLang's FP8 code uses skinny N=16 tiles with M/K variants 32/512, 64/256 and 32/256; `sm120_fp8_skinny_gemm.cuh:148–220,408–412` exposes the choices and cache-hint specialization. Its facade keeps streaming GEMV first for supported M=1 cases. The reported mixed-Qwen3.8 +1.396% C1 comparison disabled that competing native GEMV in the benchmark, so it is **not** incremental gain over its full production dispatcher. It also uses ModelOpt scaling, not automatically NInfer's row-scaled FP8 representation.

ExLlama's PR changes M1/K5120/N96 B/A from twelve eight-warp blocks to ninety-six one-warp blocks, reporting ~49% kernel reduction but ~0.86% complete-request throughput gain on 5090. NInfer combines RMSNorm and B/A/control work; `src/ops/gdn_gating_proj/bf16/bf16_gdn_gating_proj_kernels.cu` has several shape-specialized paths and reduction structures. A launch-only change to an isolated GEMV is not additive to that fusion. Preserve it as a shape/occupancy comparator if attribution identifies this op; no port recommendation without the local classifier.

## State, sampler and proposal citation branches

### FT06 — DASC compresses retained checkpoints by omission

Primary: https://arxiv.org/abs/2608.30386 ; https://arxiv.org/html/2608.30386v1 (August 31). Reference edge returns to ReplaySSM, already implemented locally.

DASC derives retention horizons from model weights and omits short-horizon state units from retained checkpoints. Restore zero-fills or refreshes them through a bounded suffix. Its KDA result reports 2.63x checkpoint compression and a fixed-memory-budget TTFT improvement; the paper also studies head-wise GDN. This differs from DAMP active-state quantization. Omission/refresh approximates the original checkpoint; decay is not an exact irrelevance proof. No author implementation was established in the inspected material. Current NInfer exact retention cannot silently substitute these states. Optional quality-gated capacity research only, especially since C1..4 state pressure differs from large serving systems.

### FT07 — SpecTr-GBV does not certify deterministic top-tree selection

Primary: https://arxiv.org/abs/2604.25925 ; https://arxiv.org/html/2604.25925v1 . Forward search from Block Verification; backward edges include SpecTr, Sequoia and SpecInfer.

The proposed optimal-transport block verifier assumes multiple **i.i.d. sampled draft sequences** and a specified conditional-coupling family. “Greedy block verification” is its algorithm name, not evidence that arbitrary greedy/top-k candidate generation meets those assumptions. DFlash's correlated block proposals and a deterministic selected tree cannot be substituted into its theorem without deriving their actual joint law.

Current p-less chain proposal q is a point mass (`speculative_round.cuh:73–84`). The single-path prefix-probability bound from TO03 remains relevant. A multi-branch tree can expand coverage, but costs target scoring and recurrent state handling; its proposal law and correction must be derived independently. No compatible drop-in verifier or matched SM120 benchmark was found here. The paper supplies a mathematical research direction, not evidence of an existing correctness defect.

### FT08 — Grammar-conditioned distribution differs from local mask semantics

Primary: https://arxiv.org/abs/2605.07698 ; https://arxiv.org/html/2605.07698v1 (May 8).

The paper distinguishes stepwise masked/renormalized sampling from conditioning the entire unmasked language model on eventual grammatical completion. Future-validity probabilities supply the missing global correction; exact evaluation is tractable only for restricted grammars in its experiments. The reported large distribution gap does not mean a speculative verifier preserving NInfer's actual stepwise masked sampler is incorrect.

For this campaign, “lossless grammar support” must mean equivalence to the actual supported masked+p-less target transitions, including valid-token filtering order and rollback. Switching to future-validity correction would change public sampling semantics. This clarification strengthens the existing tool-mask overlap recommendation: overlap execution while preserving the current law, rather than importing a different notion of grammar faithfulness.

### FT09 — Quantized-target draft training: evidence both for and against a presumed mismatch

Primary: https://arxiv.org/abs/2607.04244 ; https://github.com/nota-github/adaptfm-quant-dflash (July 5). Inspected README and serving/training organization.

This A10G/Qwen3.5-4B competition system pretrains a DFlash drafter on BF16 then adapts it to a QAD INT4 target, quantizes the drafter, and limits its attention window to 1024. The release requires at least two 80GB-class training GPUs, preferably eight. Its reported 6.978x combines target quantization, target distillation, drafting and serving changes; it is not a companion-only gain on unchanged NInfer. The separable relevant experiment is a matching draft's acceptance/cost under actual represented target states. Changing the target remains an artifact admission decision; reducing draft context may remain distribution-preserving only because the full target verifier corrects proposals.

Counterevidence: https://github.com/avifenesh/hqmtp/blob/main/results/VERDICT-nvfp4-final.md (July 10) explicitly retracts the claim that NVFP4 necessarily breaks co-trained MTP agreement for Qwen3.5-9B. Clean replay to 64K and sampled controls removed the apparent mismatch; prior evidence mixed pipelines, greedy loops and tiny samples. It leaves 27B replication open. Therefore our training recommendation cannot start from an assumed NVFP4 acceptance defect.

The later author study https://avifenesh.ai/research/small-vocabulary-mtp/ (August 5) favors a trimmed inherited draft head over a reduced student after out-of-distribution acceptance losses. It also corrects a top-64-only distillation-loss bug. Its measured engine recipe and preliminary memory/multi-seed limitations must not be conflated with the older README's acceptance-only student result. Target verification stays full-vocabulary; NInfer already has a restricted draft selector. New value is the negative evidence and evaluation design, not generic vocabulary trimming rediscovered.

### FT10 — Throughput-trained draft controllers are a conditional extension

Primary: https://arxiv.org/abs/2603.01639 ; https://github.com/zhzihao/Learning-to-Draft . Official author implementation.

LTD trains separate draft-depth and candidate-budget policies directly against draft/verify throughput, instead of maximizing acceptance alone. Its headline compares greedy EAGLE3 workloads; it does not establish p-less or current hybrid performance. The controller's training/inference cost and graph geometry matter. This supports a later data-driven controller only after inexpensive disable/fallback/change-point research already described locally; it does not override prior adaptive-draft regressions or justify a generic RL subsystem.

SpecQuant https://arxiv.org/abs/2609.21704 was screened as a different contract: routing among INT4/FP8/FP16 target variants by task complexity changes the served model and residency. EfficientRollout https://arxiv.org/abs/2606.18967 addresses an evolving RL policy with a quantized substitute; SubSpec https://github.com/vllm-project/vllm/issues/39427 targets CPU-offloaded models. Neither supplies a resident exact-artifact improvement here.

## Adjacent mechanisms and false-positive controls

### FT11 — Atlas GDN is a platform-gated fallback replacement

Primary: https://github.com/huggingface/transformers/pull/46423 (merged June 19); https://huggingface.co/kernels/Atlas-Inference/gdn ; https://github.com/Avarok-Cybersecurity/atlas . PR diff/card inspected, not a full Atlas source audit.

The Transformers change registers a compute-12.1-only Hub kernel for GB10 where the existing FLA/conv package fast paths were unavailable. It reuses the module projections/norm/output and replaces conv and recurrence cores. The card lists chunk2/3 and WY2/3/4 speculative kernels but ships SM121-only binaries. NInfer already has CUDA-native recurrence and verification; removing a pure-Torch fallback on another platform is not a comparable gain. The initial malformed HF API URL returned 404; the actual kernel card was accessible. No claim of missing source or general SM120 compatibility follows from that failed request.

### FT12 — Exact attention/storage and multimodal leads remain precisely scoped

Primary: https://www.qatq.org/ describes reversible compression of exported KV/migration tensors, not a fused compressed-attention consumer. Same-bit restore does not establish decode benefit; compression/decompression traffic and existing NVFP4 entropy need examination before a resident-cache hypothesis is promoted. Host/disk migration is not the target's required feature.

Primary: https://developer.nvidia.com/blog/when-to-use-encode-prefill-decode-disaggregation-to-accelerate-multimodal-model-serving/ corroborates repeated-media embedding reuse, while disaggregation itself is outside the current single-GPU design. Exact bounded image-feature reuse remains the narrower conditional opportunity from CA02/TO06.

Primary: https://arxiv.org/abs/2605.28115 (CIVIC) and https://arxiv.org/abs/2607.22200 (LayoutLite) compact visual token sequences; they trade numerical/model behavior for speed. They are not exact memoization or free text-prefill acceleration. Existing QCache/Kamera conclusions remain unchanged.

### FT13 — Exactness in an algorithm's name is not the serving contract

Primary: https://arxiv.org/abs/2602.04929 (TurboBoA) concerns an exact formulation of attention-aware quantization, not lossless original-model inference. https://arxiv.org/abs/2608.15383 (ExactMoE) explicitly defines exact as retaining selected experts/routing, not BF16 identity; routed MoE/offload does not match dense 27B.

Primary: https://github.com/jaimalleshk/AI-Models-Mono-Language-Pruning/blob/master/paper/paper.md conditions greedy byte equivalence on tokenizer closure and tested trajectories. It cannot justify pruning target vocabulary under p-less. https://github.com/lablup/mlxcel/issues/900 gates Gumbel-max to unfiltered sampling; this corroborates, rather than removes, the current FlashSampling applicability limit.

Primary: https://github.com/avifenesh/block-routed-swiglu records increased whole-block latency of 0.73–0.83%, not a speedup, and a failed capability hypothesis. It changes channel pairing/model mathematics and is screened out. This negative avoids treating a cheap architectural experiment as an inference optimization.

### FT14 — DDTree changes budget allocation, not the already implemented target walk

Primary: https://arxiv.org/abs/2604.12989 ; https://arxiv.org/html/2604.12989v1 ; https://github.com/liranringel/ddtree (April 14). Discovery edge: newly found Lucebox engine references; https://github.com/Luce-Org/lucebox-hub redirects to Luce-Org/lucebox.

DDTree best-first selection maximizes an acceptance surrogate formed from the product of per-position marginal probabilities under a fixed node budget. The paper explicitly distinguishes these marginals from path-conditioned target probabilities. Its verifier samples the target, follows a matching child, and stops at the first unmatched token; experiments use H200 GPUs at temperatures zero and one, not this 5090 hybrid.

NInfer already performs that p-less target-sample/child walk in `speculative_round.cuh:962–995`. Its DFlash2 builder in `dflash2_path_select.cuh:498–594` instead expands a fixed two-node frontier, bounds output width to 16, and scores Markov-codebook candidate pairs. The fresh lever is **allocation of a fixed verification budget across depths/branches**. DDTree's factorized-surrogate optimality does not transfer automatically to these Markov scores or establish better acceptance. A GPU-resident budget selector may merit a controlled comparison, preserving graph capacity, ancestor state and actual target-law sampling. This needs no claim that NInfer lacks trees and no new target checkpoint; it remains an unmeasured candidate.

## Discovery graph and evidence ledger

| ID | Entry path and inspected endpoint | Evidence / final disposition |
|---|---|---|
| FT01 | Sparkinfer name disambiguation → b12x docs → convergence/finite code | Source inspected; new algorithm comparator, research-only |
| FT02 | Backend PR search → llama29353 → CUDA products/inversion/dispatch | Source inspected; distinct precision/fusion route; unmerged and speculation gate |
| FT03 | MSInfer report references → Kachua author code; SyFI/LLM-CUDA references | Kachua source inspected; exact finite algebra with floating-point gates; B200 results |
| FT04 | SGLang release → PR36865 → KDA-Pilot provenance → TMA policy source | Source inspected; directly relevant cache-interference negative and narrow win |
| FT05 | PR38082/ExLlama369 → kernel geometry and benchmark denominator | Source inspected; shape-specific, no additive local gain established |
| FT06 | State-compression search → DASC → ReplaySSM reference | Primary paper; approximate retention, no inspected author implementation |
| FT07 | Block Verification forward search → SpecTr-GBV → theorem assumptions | Primary paper; i.i.d. assumptions do not certify current proposals |
| FT08 | Grammar+speculation search → future-validity theorem | Primary paper; clarifies oracle, different sampling product |
| FT09 | Quantized-draft search → AdaptFM code → hqmtp verdict/study | Author code/docs/negative ledger; conditional training, premise must be measured |
| FT10 | Speculation bibliography → LTD author repo; routing/offload papers | Primary screen; trained controller conditional, other residency contracts rejected |
| FT11 | New engine author channel → merged Transformers PR → actual HF kernel card | Integration/card inspected; SM121 fallback-replacement result |
| FT12 | Lossless-cache and vision search → publisher/author endpoints | Mechanism screen; exact restore not resident attention gain; token compression approximate |
| FT13 | Exact/sampling/vocabulary search → author claims and qualifications | Primary screens; prevent semantic/hardware denominator mistakes |
| FT14 | Lucebox references → DDTree paper → current tree builder/verifier | Paper and local source compared; budget allocation hypothesis, verification already present |

Representative query families included SM120/NVFP4 GEMV and small-M, GDN prefill optimization/state compression, exact KV compression, exact sampling/vocabulary pruning, grammar+speculation, quantized draft training and vision-encoder reuse. Backward citation mining used the GDN contest report and SpecTr-GBV/DASC references; forward discovery used names, algorithms and successor-paper searches. This is not a complete bibliometric citation database and does not assert every citing paper was read. Prior CUDA/PTX release and autotuning screens remain in TO08/TO09; no newly verified hardware facility superseded the existing SM120 limits.

## Consequence for priorities

No new source establishes a measured improvement over this checkout. The new concrete comparisons are b12x certified-convergence prefill, llama's reconstructed-BF16 register recurrence, Kachua finite inversion, and KDA's shape-specific cache policies. The grammar-overlap and exact image-cache hypotheses retain their existing profile gates. Companion training remains potential upside with substantial prerequisites; the hqmtp retraction makes measuring a mismatch before training even more important. Approximate state omission, global grammar conditioning, target-vocabulary pruning and target-routing changes remain explicitly outside an unchanged exact-target optimization.

## Independent cross-review of the companion final reports

Kernel/state reviewer, 2026-09-26:

- **Poseidon FB01:** independently read its `docs/gdn-state.md` and `results/summary.json`. INT8 checkpoint top-1 agreement 0.423529, log-prob RMSE 4.748515 and maximum absolute error 20.90394 match the source. Active recurrence remained BF16; only retained boundary representation changed. Requested a wording correction from “reproduced logits” to the actual reported top-1 agreement and log-prob errors: no published full-logit tensor comparison is established by that table. No inference about current NInfer FP32 state or NVFP4-vs-FP8 KV equality is justified. The summary's 50.811 aggregate output rate explicitly includes prefill and C8 is outside C1..4.
- **HyperQwen FB02:** local `docs/maintainer/paged-kv-cache.md:762–790` confirms current/rewrite, sparse prefill ladder and rollback checkpoints referring to one exclusively owned KV bundle. The report correctly refuses to transplant an every-block retention diagnosis or a fixed six-block interval. Alternating conversations are a useful qualification workload, not proof of a local capacity defect.
- **SGLang FB03:** independently read the full issue 41351 follow-up and minimal patch. Identical inputs/scales, cold/warm matrix-shape change, first difference before recurrence, failed FP32-checkpoint experiment, and patched/reverted outcomes are all faithfully represented. The fallback applies only under the deterministic mode in the supplied patch. This establishes a scoped publisher diagnosis, not universal numerical failure of FP8 or NInfer.
- **Independent report FI01:** the BlackweLLM August 23 note contains the reported 96.90→160.32 decode and 93.75→151.11 warm request rates with unchanged acceptance count. Local `variant.cpp:69–79,477–496` can split multi-request panels, but each panel still includes the verification width; this is not the external repeated-M1 baseline. The broad already-batched conclusion is correct without asserting identical layouts or kernels. The new b12x lineage is correctly separated.
- **Independent report FI02–FI03:** hardware, artifact and denominator qualifications do not overclaim local equivalence. TensorSharp uses a no-checkout clone; independent `git show HEAD:README.md` and `git show HEAD:docs/engine_comparison_report.md` inspection confirmed the RTX 3080 Laptop hardware, historical dense 1.07×/0.96×/0.95× ratios, and current detailed report with no overlapping reference cells. The report correctly distinguishes those snapshots. No additional numerical correction was identified. None of these checks ran GPU work.

Engine-discovery cross-review closure: independently inspected Knivesysl `src/forward_qwen.cu` GDN precision comments, CPU copy drafting, per-node state archive and `tools/serve_batched.py` sampler/verification contract; inspected Qwarz `hot_head.py` complete-group reconstruction and argmax proposal mapping plus `nvidia_mlp.py` selected donor loading. The engine report now distinguishes the 5-bpw artifact from its 6-bit head/4-bit MTP, frequency calibration from training, Knivesysl's own 64.7 tok/s denominator from vLLM's 72.1, inherited lineage, and the TF32/harness-error caveats. The b12x/SparkInfer alias duplicate was removed. Lucebox evidence is correctly labeled paper/local-applicability review rather than an external kernel-code audit. Final inventory has 126 unique canonical rows, no alias duplicated as a canonical row, and existing report-file targets. This is a lineage/evidence check, not 126 complete code audits.
