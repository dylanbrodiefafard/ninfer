# Final independent engine and evidence investigation

Cutoff: 2026-09-26. This report records the orchestrator's additional source inspection while three research agents followed independent engine, backend and technique queues. No external engine was built or benchmarked. Repository snapshots were inspected under `/tmp/ninfer-research`; no model weights were downloaded.

## FI01 — BlackweLLM: independent runtime, shared kernel library, useful negative evidence

Primary repository: https://github.com/jieen1/BlackweLLM

Snapshot inspected: `df71b8d`, authored 2026-09-04. This is an independent Python model/runtime with custom Triton/CuTe work and external kernel dependencies; the production path is not simply a vLLM launcher. Its kernel dependency formerly called SparkInfer is **local-inference-lab/b12x**, distinct from the independent **gittensor-ai-lab/SparkInfer** engine already inspected in the depth audit. Alias resolution matters: finding one does not constitute inspection of the other.

Sources:

- https://github.com/jieen1/BlackweLLM/blob/main/runtime/model/qwen36_model.py
- https://github.com/jieen1/BlackweLLM/blob/main/notes/2026-08-23-qwen38-modelopt-gdn-batched-projections.md
- https://github.com/jieen1/BlackweLLM/blob/main/notes/2026-08-19-qwen38-dspark-optimization-stop.md
- https://github.com/local-inference-lab/b12x
- https://github.com/local-inference-lab/sparkinfer

The project's Qwen27B measurements use an RTX PRO 6000 Blackwell Max-Q, FP8 KV, DFlash/DSpark and larger VRAM allocations than a 5090 can sustain. The README's multi-slot, long-context rates do not establish the same configuration fits 32 GB, nor a speed advantage over NInfer. Current launch instructions also include a different Flash-Next/MTP profile; do not combine its configuration with historical 27B rates.

The August 23 note reports a real and instructive difference: ModelOpt W4A4 GDN verify projections changed from repeated M=1 execution to one matrix over the verify window. At C1, K7, 128K prompt and 256 output tokens, author measurements rise from 96.90 to 160.32 decode tok/s and 93.75 to 151.11 warm end-to-end tok/s, with the same 226/255 accepted/committed count and completion digest. Source selection is format-dependent; the Unsloth mixed path retains its previous behavior. **This is already the broad execution principle of NInfer's batched verification, not an incremental 65% opportunity here.** Raw A/B IDs are named, but their original `/tmp` artifacts are not public in that note. Treat it as source-supported author measurement.

The August 19 note fixes four 131072-token requests, 256 output tokens, K7, FP8 KV and prefix/graph settings. It records slower graph-fused draft-context KV, neutral ragged tiers and RMS/FP8 fusion, and lower acceptance/throughput with a confidence threshold. Explicitly forcing FlashInfer produces small rate differences even though automatic selection already uses that same kernel family; the author correctly labels them noise. These results do not rule out different implementations or shapes, but prevent counting generic fusion, confidence gating or a renamed default as established gains.

The older optimization-stop trail is also worth following to its correction:

- https://github.com/jieen1/BlackweLLM/blob/main/notes/2026-08-03-stage4-kernel-levers-exhausted.md
- https://github.com/jieen1/BlackweLLM/blob/main/notes/2026-08-03-performance-gap-vs-historical.md
- https://github.com/jieen1/BlackweLLM/blob/main/notes/2026-08-03-mtp-round-profile.md

The apparent kernel-throughput explanation was superseded by direct profiling of uncaptured verify and host copies. Historical acceptance comparisons also had a double-counting correction. These are historical defects in that runtime, not current NInfer defects. The transferable lesson is concrete: inspect the final correction and full-round trace before deriving an optimization from an earlier explanatory note. NInfer already captures verification and owns device-resident speculative state; the historical fix is not a missing feature.

**Disposition:** retain as an independent engine and negative-evidence source; b12x's distinct prefill research is assessed in `final-technique-research.md`. No new top-ranked implementation follows from the large headline batching gain.

## FI02 — TensorSharp: broader runtime, different hardware and incomplete comparison snapshot

Primary sources:

- https://github.com/zhongkaifu/TensorSharp
- https://github.com/zhongkaifu/TensorSharp/blob/main/README.md
- https://github.com/zhongkaifu/TensorSharp/blob/main/docs/engine_comparison_report.md
- https://github.com/zhongkaifu/TensorSharp/tree/main/benchmarks/engine_comparison
- https://github.com/zhongkaifu/TensorSharp/blob/main/TensorSharp.Models/Models/Qwen35/Qwen35Model.GatedDeltaNet.cs
- https://tensorsharp.ai/
- https://tensorsharp.ai/overview.html
- https://tensorsharp.ai/models.html

Snapshot inspected: `a5dae07`, authored 2026-09-26. TensorSharp has a .NET host, native GGML execution and an experimental direct CUDA path. It is more than a protocol wrapper, but its backend families must remain separate when interpreting results. SciSharp/TensorSharp2 is a derivative lineage, not another independent kernel implementation.

The README's principal comparison is explicitly **RTX 3080 Laptop, 16 GB**, identical GGUFs, greedy, single stream, MTP off. Its dense Qwen3.6-27B IQ2_XXS CUDA row claims 1.07x decode, 0.96x prefill and 0.95x TTFT relative to llama.cpp; the 1.28x CUDA prefill row is the 35B-A3B MoE. None is a 5090/NVFP4 result. A secondary Chinese-language article discussing workstation choices led to this project; the primary source is the authority for its hardware and measurement.

At the inspected HEAD, the linked detailed report contains a later/different subset of runs and explicitly says there are **no overlapping TensorSharp/reference cells**. It compares TensorSharp CUDA with its own Direct CUDA for other model cells. Thus the linked current table does not reconstruct the historical README ratios. This is a provenance limitation, not evidence those earlier runs never happened.

The harness defines prefill throughput as prompt tokens divided by client TTFT, a useful user-visible proxy that includes work beyond the prefill kernels. Decode uses tokens after the first divided by first-to-last arrival time. Whitespace-normalized sequence similarity is not an independent mathematical oracle or a task-quality result.

The GDN source contains a fused whole-layer GGML graph path and native device recurrence, alongside backend-specific fallbacks. Its comments attribute important gains to eliminating managed per-layer round trips and keeping state resident. NInfer already uses those broad ownership/execution principles. This screen found no qualified, absent SM120/NVFP4 mechanism that justifies displacing the stronger candidates in the main report. The project's multi-platform, large-model and agent-host features are broader than NInfer's current contract.

**Disposition:** genuine runtime with multiple backend lineages; screened and retained, not a matched competitor measurement or a proposed framework transplant.

## FI03 — Article and workload claims: recover the actual denominator

Sources screened:

- https://www.tomshardware.com/tech-industry/artificial-intelligence/benchmarking-qwen-3-8-27b-on-rtx-5090-and-beyond-vram-capacity-alone-cant-overcome-severe-software-and-inference-engine-bottlenecks
- https://lcz.me/topic/1228
- https://zonetech.tw/blogs/single-vs-multi-gpu-enterprise-inference/
- https://waelmansour.com/blog/one-million-token-context-on-two-rtx-5090s/

Tom's Hardware's author tests are useful evidence for the recipes actually exercised. Their one-5090 vLLM run is constrained to 32K, while other reported long-context runs use two GPUs. This does not establish that a 5090 universally cannot execute longer context: artifact, KV codec, graph reserves and state policy differ. Do not turn a deployment's capacity failure into an architectural GPU limit. Its emphasis on whole-turn latency is relevant; its recipes are not a current NInfer comparison.

The Chinese workload post describes 1205 requests over roughly 13.7 hours, approximately 142.8 million prompt tokens and 2.05 million output tokens. The resulting 98.6% prompt-token share is **a count ratio, not measured GPU time spent prefilling**. Token counts alone cannot apportion inference cost, particularly when prefix reuse avoids recomputing prompt tokens. Long-prompt TTFT and queued concurrency observations are useful workload leads, not controlled kernel measurements. The associated implementation trails are handled in the engine/backend reports.

The million-token article uses two 5090s and a TP/extended-context fork. It belongs in the landscape, but tensor parallelism and altered context-extension semantics are outside the single-GPU target. Its statements about single-card capacity are recipe-specific, not universal bounds. The enterprise article is a discovery source; its TensorSharp performance references require the primary hardware qualifications above.

## FI04 — What the final pass can and cannot establish

The delegated reports cover systematic repository topics/search pages, author repositories, aliases and descendants, nondefault branches, current issues/PRs, papers and citations, Hugging Face cards, Reddit and multilingual source trails. The final engine inventory separates standalone runtimes, forks, shared backends, wrappers, kernel libraries and source-limited claims. The existing 469-repository fork inventory and selected 192 branch refs remain discovery records, not full audits of all those codebases.

Codeberg access was explicitly robots-blocked during this pass. Searches of additional hosting/language channels yielding no material hit do not prove no project exists. The original q36 source remains inaccessible; a mirror recovery is separately qualified in the engine report. Some projects publish binary kernels, local-only raw traces or benchmark summaries without matched controls. Those limits are retained rather than filled with inference.

Completion here means that the material leads discovered by this sustained campaign have a source-backed disposition and the recommendations account for the strongest positive and negative evidence. It cannot mean a certified census of the whole Internet, inaccessible private communities, every revision of every fork or future releases. Remaining local-performance questions are explicitly bounded experiments, not claims of measured gains.
