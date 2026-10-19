# Artifact quality, NVFP4 arithmetic, and draft compatibility

Research cutoff and access date for all sources: 2026-09-26. No models were downloaded, trained, converted or benchmarked in this research pass. Findings below are proposals for decisions, not implemented changes or verified NInfer speedups.

## Decision

The strongest distinct lead is not “turn on FP4.” It is to qualify an exact-target artifact and companion together, and to distinguish weight representation, activation arithmetic, and draft-training distribution. NInfer already has NVFP4 weights, selective FP8 protection, A16 short-panel execution, and a quantized DFlash2 companion. A new paper's speedup against BF16 or a heavily FP8 checkpoint therefore does not describe the remaining opportunity.

Two categories must remain separate:

1. A newly trained companion for the already supported target representation may improve speculation without introducing another model identity. It still needs explicit compatibility, payload inventory, math/state and acceptance qualification. A DSpark architecture is not identical to DFlash2 merely because both draft blocks.
2. Replacing the base with QUASAR/Minima/SparkInfer weights changes the represented checkpoint. The current default is an Ostfralla-derived artifact, so research relevance is not authorization to silently swap it. Such a choice needs an explicit artifact-contract decision, capacity/quality admission, consistent conversion/binding and numerical-state identity. Do not register arbitrary checkpoint targets under the research request.

## Existing baseline matters

`docs/maintainer/qwen3.8-27b-artifact.md:98` defines the selective FP8 recipe: eight matrices, namely attention output layer 11; fused attention inputs at 27, 31 and 51; MLP gate/up and down at 62 and 63. It adds 328 MiB, uses per-row BF16 scales/E4M3FN codes, and preserves BF16 controls, W8 endpoints and the draft. It does not make all GDN FP8.

`docs/performance.md:17` records matched WikiText PPL 7.254364 → 7.184892, a larger mixed publisher recipe at 7.095132, and roughly 2–3% throughput cost for the selected recipe. A broader 36-file coding check did not establish benefit from further proposed promotions. These are local results on specific represented inputs and cannot be compared numerically to another paper's differently chunked PPL. The active artifact authority also supports a distinct mixed FP8/NVFP4 inventory under the same identity, with numerical retained-state isolation.

`docs/maintainer/dflash2-tree-speed.md:525` documents why naive A4 conversion is insufficient: isolated and Engine speedups were rejected when real recurrent draft decisions changed. A new quality-preserving arithmetic route needs more than a tensor cosine similarity or plausible generated answer.

## Q01: Minima / Why Gated DeltaNet Survives 4-Bit Quantization

Primary paper: https://arxiv.org/html/2609.04098v1 (2026-09-03), checkpoint https://huggingface.co/minima-ai/mnma_qwen3.8_27b_nvfp4.

The study evaluates 496 transformer linears in NVFP4 W4A4 on Qwen3.8-27B, with embeddings, head, convolution and norms excluded. It attributes GDN robustness to block scaling, gate nonlinearities and the delta update's correction. Its mechanism experiments retain FP32 recurrent state; they do not justify quantizing that state. Hardware is RTX PRO 6000, with C32 decode and FP8 KV, not the NInfer C1..4/5090 contract. It reports a measurable perplexity residual despite task scores lacking separation.

Its most transferable concrete issue is fused calibration: combining projections with unequal global scales by taking only their maximum mis-scales weights unless local scales are adjusted. It repairs the checkpoint and checks kernel/reference GEMMs. A second result concerns calibrated FP8 KV scales, which is not an automatic improvement to a block-scaled NVFP4 cache.

**Decision relevance:** use the mechanism to motivate narrowly targeted artifact experiments, not remove all protections. NInfer already quantizes substantial GDN projection weight. First determine actual bytes and time in the remaining protected matrices; BF16 control gates are small, so their quantization cannot inherit the paper's whole-GDN memory gain. The paper's controlled failure is a useful import/fusion audit condition whenever changing converter scale policy. No current NInfer scale bug was demonstrated in this investigation.

## Q02/Q03: QUASAR QAT and exact-target checkpoint

Paper: https://arxiv.org/abs/2608.13966 (2026-08-14). Model card: https://huggingface.co/QUASAR-QAT/Qwen3.8-27B-QUASAR-NVFP4.

QUASAR changes training-time reconstruction using gradient-based saliency and clipping/dequantizer optimization while exporting standard low-bit formats. Its general paper is not an inference-kernel paper. The exact-target model card reports a 19.7 GB artifact, all 496 transformer linears NVFP4 W4A4, and one QAD epoch (2446 steps) against a frozen BF16 teacher. Reported GPQA-D is 90.91 versus BF16 91.41, and AIME26 is 100 for both, with 396/90 sampled responses respectively. These are publisher measurements, not proof of equivalence and not a paired comparison to the current NInfer artifact.

**Decision relevance:** the smallest coherent next decision is whether the trained exact-target weights improve held-out quality per byte versus the already qualified selective-FP8 artifact. Do not begin with retraining or a new engine. Inspect the head, embeddings, GDN controls, fused scale groups, MTP payload and required activation path before planning a conversion. W4A4 QAT does not guarantee that a changed A16 decode route, partial conversion, or the old DFlash2 companion has the same distribution. Keeping the target identity does not make its caches or draft automatically compatible.

## Q04: NVIDIA QAD

Primary report: https://arxiv.org/abs/2601.20088; implementation https://github.com/NVIDIA/Model-Optimizer/blob/main/examples/llm_qat/README.md; NVIDIA project report https://research.nvidia.com/labs/nemotron/nemotron-qad/ (2026-01-20).

QAD distills a low-precision student against a teacher to recover accuracy lost by quantization. The reported Nemotron-family success establishes that training can recover quality in practical exported NVFP4 formats; it does not establish this Qwen artifact's quality or a NInfer throughput gain. NVIDIA's newer Qwen3.6-35B-A3B example uses 500 iterations, but that is another model and workload.

**Decision relevance:** training is an option if an artifact gap is measured and existing public exact-target candidates do not solve it. It is not a prerequisite for this research campaign, and the reported GPU/training cost should not be assumed feasible on one 5090. Prefer evaluated published weights before initiating a new training campaign.

## Q05/Q06: ARCQuant

Paper: https://arxiv.org/abs/2601.07475 (v1 2026-01-12, v2 2026-07-04). Code https://github.com/actypedef/ARCQuant; inspected revision `4448db361c5c5f68a6a0a5297adf2ee2a3a41b66` (2026-09-21), local `/tmp/ninfer-research/kernels/ARCQuant`.

The method inserts selected activation quantization residuals as additional reduction channels, retaining an NVFP4 GEMM rather than a separate high-precision branch. The paper reports benefits on RTX 5090/PRO 6000 against FP16, not against NInfer's existing NVFP4 route. Code requires calibration-derived `reorder_indices` and `select_num`; `model/qLlamaLayer.py` exposes the reordered/augmented activation preparation and `kernels/src/{reorder,nvfp4,rmsnorm,down}.cu` supplies CUDA implementations. Repository documentation lists Qwen3 as unfinished and says vLLM efficiency scripts are a future release despite checking a vLLM integration item.

**Decision relevance:** inspect as an A4 quality-recovery candidate only where current A16 arithmetic is expensive enough and an augmented reduction is cheaper than the present route. Extra weight channels and activation preparation can lose at memory-bound C1. It changes layouts/artifact preparation and cannot be introduced as a hidden runtime weight-repacking step. The next decision test is one real weak projection with captured represented inputs: decoded-weight oracle, residual error relative to current A16, total bytes and complete Op cost. Reject early if the classifier shows no achievable end-to-end value.

## Q07/Q08: SharQ

Paper: https://arxiv.org/abs/2606.26587 (2026-06-25). Code https://github.com/actypedef/SharQ; inspected revision `3cc584fe48e32e59c8b4e37323470dcdd606bb11` (2026-07-01), local `/tmp/ninfer-research/kernels/SharQ`.

SharQ splits activations into an N:M sparse FP4 backbone and a dense residual measured after sparse quantization, sharing a weight payload with path-specific scale views. The paper claims training/calibration-free operation and RTX5090 improvement against FP16/FP8. Code has real and simulation modes; `kernels/include/sharq_blackwell_arch.h` explicitly selects `Sm120`, sparse 128×128×256 tiles and SM120 sparse TMA scheduling, so this is not an sm100-only paper implementation. `fused_sparse_prepare.cu`, `fused_rmsnorm_sparse_prepare.cu`, `sparse_nvfp4.cu` and `shared_weight_nvfp4.cu` are concrete inspection anchors. The README says model evaluation and benchmark/demo modes enable different fusion configurations, which must be reconciled before treating quality and speed as one result.

**Decision relevance:** potentially useful for compute-heavy prefill if it recovers quality without falling back to FP8. It is not compression of the existing target's weight bytes. Two paths plus metadata/preparation can add traffic at C1, and fusion may alter arithmetic. The classifier should compare full sparse+dense+prepare cost to current W4A4 prefill before any Engine integration. The exact Qwen3.8 hybrid is not demonstrated by the inspected repository.

## Q09: Jared Frost / SparkInfer artifact-and-draft co-design

Primary author report: https://jared-hpc.com/posts/nvfp4-qwen38-27b-rtx5090/ (2026-09-18). This is an engine/artifact author's benchmark, not independent replication.

The report compares per-tensor precision allocation on one 5090, then quantizes a DSpark companion and retrains it using the actual tool/chat formatting. It reports an overall acceptance-length gain from 2.761 to 2.904 and a larger relative gain for agentic traffic. The source explicitly distinguishes short-context benchmark rates from server rates. Its core lesson is to optimize on actual serving traces rather than infer chat/agent speed from highly predictable code generation.

**Decision relevance:** first use the current DFlash2 artifact to measure phase-specific acceptance on an authorized, held-out corpus rendered by NInfer's actual shared frontend. If the gap is template/distribution mismatch, a companion trained for the current target may be a smaller contract change than replacing the base. If the candidate requires the author's modified target, treat it as a checkpoint decision. This recommendation does not assume DSpark beats DFlash2; DFlash2 already models inter-position dependence through its selector.

## Ordered future decision tests

These are conditional experiments, not unchecked assigned implementation work.

1. **Artifact provenance/inventory screen:** compare candidate weight and activation conventions with current supported shell; identify real changed bytes and required semantic boundaries before downloading any large asset.
2. **Draft mismatch screen:** current target + current companion on frozen tool/code/chat/reasoning traces; isolate prefix/template, phase, context and sampling. Measure accepted tokens divided by whole-round time, not acceptance alone. If no material gap, stop draft-training work.
3. **Quality-per-byte decision:** for an admitted exact-target candidate, compare held-out NLL/behavior and long-context retrieval using the same represented inputs, KV codec, chunking and template. Keep selection data separate from validation. A few overlapping task intervals do not prove equality or remove a PPL cost.
4. **One-projection residual-quantization test:** only after the byte/compute classifier admits ARCQuant/SharQ at a measured bottleneck. No generic model wrapper or backend adoption is necessary for the experiment.
5. **Final usage test if any candidate survives:** C1..4 complete waves, mixed frontend/tool flow, MTP/DFlash acceptance, retained-state lifetime and NVFP4 default KV. Distinguish the candidate's changed model answers from engine arithmetic regressions.

## Source ledger

Evidence levels: P = primary paper; C = source code inspected; M = publisher model card or author measurement (unreplicated here); L = discovery lead only. All accessed 2026-09-26.

| ID | Date / revision | Level | Source and relevance |
|---|---|---|---|
| Q01 | 2026-09-03 v1 | P | https://arxiv.org/html/2609.04098v1 — exact-family GDN W4A4, scale-fusion failure, mechanism study |
| Q02 | 2026-08-14 v1 | P | https://arxiv.org/abs/2608.13966 — QUASAR training method |
| Q03 | live model card | M | https://huggingface.co/QUASAR-QAT/Qwen3.8-27B-QUASAR-NVFP4 — exact-target trained artifact and publisher evaluations |
| Q04 | 2026-01 report | P | https://arxiv.org/abs/2601.20088 — NVIDIA QAD; https://github.com/NVIDIA/Model-Optimizer/blob/main/examples/llm_qat/README.md — training workflow |
| Q05 | 2026-07-04 v2 | P | https://arxiv.org/abs/2601.07475 — ARCQuant method and hardware claims |
| Q06 | 4448db361c5c, 2026-09-21 | C | https://github.com/actypedef/ARCQuant — calibration, layouts and actual CUDA path |
| Q07 | 2026-06-25 v1 | P | https://arxiv.org/abs/2606.26587 — SharQ method and evaluation |
| Q08 | 3cc584fe48e32, 2026-07-01 | C | https://github.com/actypedef/SharQ — explicit SM120 sparse/dense implementation |
| Q09 | 2026-09-18 | M | https://jared-hpc.com/posts/nvfp4-qwen38-27b-rtx5090/ — exact-model author comparison and serving-aligned drafter |
| Q10 | accessed 2026-09-26 | L | https://huggingface.co/wallawalla47/Qwen3.8-27B-Quasar-NinferV3 — existing downstream conversion lead, not proof upstream compatibility |
| Q11 | accessed 2026-09-26 | L | https://huggingface.co/kybrcore/Qwen3.8-27B-QUASAR-NVFP4-NInfer — second downstream artifact lead, same caveat |
| Q12 | accessed 2026-09-26 | L | https://www.reddit.com/r/LocalLLaMA/comments/1vyie86/fully_quantized_nvfp4_qwen3827b_with_quasar_qad/ — discovery discussion; primary card/paper carry claims above |

Research conclusion: QUASAR and serving-aligned companion training are the most plausible exact-target followups. ARCQuant and SharQ are lower-confidence techniques to evaluate only against an identified arithmetic/quality bottleneck. None justifies unconditional replacement of the current artifact or removal of numerical protections.
