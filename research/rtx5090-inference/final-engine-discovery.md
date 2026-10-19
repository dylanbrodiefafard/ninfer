# Final engine discovery and lineage expansion

Evidence retrieved 2026-09-26 local time. This is an additional independent discovery pass, not a replacement for earlier source audits. No inference, weight download, installation, or GPU benchmark was performed. Reported speeds below remain author measurements. `engine-discovery.tsv` consolidates named engine families, forks, wrappers, and adjacent projects with explicit evidence levels; its row count is not a count of independently implemented or fully audited engines.

## Discovery coverage and stopping boundary

GitHub repository search covered `topic:rtx-5090` (98 results), `topic:sm120` (45), and all three pages of `topic:nvfp4` (236, pages 100/100/36). Additional description queries were `5090 inference` (50), `sm120 inference` (13), `"Qwen3.8" "engine"` (55), and `"inference engine" "Blackwell"` (16). Deduplication yielded 444 repository metadata records. These include irrelevant applications, kernels, datasets, and recipes; metadata enumeration is not code inspection. The 12 related-author inventories added 301 records, overlapping the search results: SRSWTI, HaberstrohSystems, QwarzEngine, nibor1896, MerkyorLynn, The8Darkness, avifenesh, signalnine, kekzl, guoqingbao, truespar, and Avarok-Cybersecurity.

Follow-up queries searched q36 references, CUDA-language 5090 engines, explicit sm_120a descriptions, and Chinese inference descriptions. HN searches used `5090 inference engine`, `sm120 inference`, and `Qwen3.8 inference engine`; the broad first query returned 138 matches, and both pages (100 plus 38) were screened. The remaining 38 were unrelated historical comments/jobs, not new engine leads. Algolia marks the hit count non-exhaustive and recomputed counts varied across requests; the original query parameters reproduced the 138-hit page set. Chinese/Japanese/Russian/Korean web queries and Korean curated lists led to Sharp and Lucebox primary sources below. GitHub API later rate-limited requests; raw source and selective git reads remained available. This is a documented search boundary, not an exhaustive indexed-web guarantee. The prior 469-fork metadata and 192-branch inventories remain distinct denominators.

Promising newly discovered engines were read from primary documentation, with selective source audits for Qwarz, knivesysl, Antelope, VoidInfer, Crow Nest, and the q36 mirror. Adjacent kernel research was transferred to the kernel agent and resolved in `final-technique-research.md`; BlackweLLM and TensorSharp were independently assigned to the root researcher. Remaining leads below have explicit off-contract, duplicate, early-project, or inaccessible dispositions; no promising identified engine is silently left as a future task.

## FE01 — Qwarz: ExLlamaV3 runtime with real hybrid-weight and draft-head work

Primary: https://github.com/QwarzEngine/qwarz ; inspected revision `166123f` (author date September 24). README, `server/worker.rs`, `src/qwasar_runtime/hot_head.py`, and `src/qwasar_runtime/rendezvous.py`.

Qwarz uses a Rust/SQLite supervisor and persistent ExLlamaV3 worker; it is neither an entirely independent kernel engine nor merely a GUI. The advertised desktop-5090 Qwen3.8 configuration combines an EXL3 **5 bpw artifact (6-bit head, 4-bit MTP)** with **192 donor NVFP4 MLP matrices**, assembled from two artifacts at load. That changes the checkpoint representation relative to the current NInfer artifact. Its six-step MTP proposer restricts only the draft head to 65,536 rows, arranged as 512 complete 128-token Hadamard groups; the target keeps the complete 248,320-row head. Source validates complete groups and maps back to original token IDs. This is a useful concrete proposal-only head-trimming implementation, not permission to shortlist the target distribution.

The author reports roughly 2.5 ms less verification-path time and 5–17% decode gains at 32K–128K in its own earlier revision. The map was frequency-calibrated on a 37-cell benchmark corpus with reported 99.6% coverage: coverage on that corpus does not establish unseen-workload acceptance. The GPU-resident draft path also costs roughly 2.5 GB for an embedding cache; trimming adds about 0.24 GiB. Current NInfer already has optimized draft-head paths; any incremental experiment must isolate map size, acceptance, and end-to-end cost rather than copy a headline. Source identity includes donor and map choices. No root LICENSE appeared in the inspected tree, so extraction permission is not established by public visibility.

## FE02 — knivesysl: newly discovered native CUDA derivative, workload-dependent copy speculation

Primary: https://github.com/SRSWTI/knivesysl ; revision `787abab` (September 26). `README.md`, `src/forward_qwen.cu` around lines 22732, 22914 and 27810 onward. Native approximately 29,000-line CUDA translation unit with Python ctypes orchestration, explicitly tuned for desktop GB202/170 SM. Advertised CUDA 13.3/driver 595 differs from the local toolchain. Supported representations include FP6 E2M3, NVFP4-all, and hybrid NVFP4 MLP/FP6 elsewhere. Its `.ksl` rename from `.tqf` is not a new quantizer; quality figures are explicitly inherited rather than independently qualified.

The published three-repeat median comparison uses a repetitive coding prompt, greedy 512-token generation, prefix caching disabled, frozen September 2 vLLM/SGLang references and September 3 knivesysl results. Plain knivesysl is often slower than vLLM. Ngram-16 improves an 8K/C1 case to 146.4 tok/s versus its own 64.7 plain and approximately 72 for plain vLLM, but at 128K/C1 falls to about 46 versus 54.5 plain and 57.8 vLLM; concurrent cases also regress. The long-context vLLM prompt is 130,496 tokens versus 131,072 elsewhere. These are neither a clean exact-artifact comparison nor evidence of a universal copy-draft gain.

Code searches CPU history using 4-to-2 token matches, grafts copy chains into the verification tree, samples target emissions, and commits the longest matching prefix. This is target-sampling/copy matching, not sparse p/q rejection sampling. Width-dependent floating-point differences still require qualification before claiming deterministic identity. The v2 path archives full FP32 recurrent state per speculative node: eight nodes across 48 layers cost about 1.2 GB, and up to 16 are supported. This is a memory tradeoff, not automatically an improvement over current replay. Current NInfer already provides replay and speculation; only measured proposal reuse and schedule differences are candidates. The project explicitly carries forward fork quantization/quality provenance; this is not a claim of independent kernel ancestry. Its default GDN TF32 path reports roughly 1.3e-2 core error versus 2.1e-5 for FP32. A 97.28% teacher-forcing agreement check compares its own wave-width controls, and an older approximately 3% result was retracted as a shifted-corpus harness bug. These do not establish exact recurrent mathematics. No root LICENSE was present; do not treat source visibility as permission to transplant it.

## FE03 — Antelope: native Rust/CUDA, but INT4 workstation evidence

Primary: https://github.com/HaberstrohSystems/antelope ; revision `d2d50af` (September 25), README and `kernels/sample.cu` around line 334 onward. The measured GPU is **RTX PRO 4000 Blackwell, 70 SM/24 GB**, not desktop 5090. This is C1 GPTQ/AutoRound INT4-group128 for Qwen3.5/3.6, not NVFP4. Its roughly 100 tok/s speculative versus 41 plain figure at 512 context is internal; the vLLM reference could not fit MTP and llama.cpp uses different Q4_K_M weights.

A target-head INT4 screen followed by INT8 candidate rescoring is called exact in portions of the project. Inspection finds a fixed safety margin and bounded histogram/candidate window. Exact rescoring of selected candidates does not prove that the globally winning candidate or all sampling probability mass survives screening. This is a numerical qualification gap, not an adoptable exact-target optimization. T16 layouts, PDL, and chunked GDN are interesting implementation choices already represented elsewhere in this campaign. MIT license applies to this inspected project.

## FE04 — VoidInfer: an unlisted NInfer descendant with useful negative evidence

Primary: https://github.com/The8Darkness/voidinfer ; revision `b7505d0` (September 22), `UPSTREAM_AUDIT.md`, `docs/current-status.md`, and `src/exl3/text_model.*`. Explicit NInfer ancestry exists despite a separate repository identity. The current qualified slice is Windows EXL3 greedy text C1 with bounded C2; positive-temperature, media, and several speculative routes have narrower or absent qualification. DFlash2 C1 R608 is harness-only, not the server default.

R612 segmented resident-prefix execution was rejected after **17.08–21.15% slowdown**. WMMA32 remains default-off pending quality; MTP remains unqualified/off. Historical OSCAR, NVFP4 and VeriCache experiments must not be substituted for the current EXL3 product. Its thousands of checks and frozen-reference rows do not prove arbitrary-model quality. Apache-2.0 plus third-party component terms. Disposition: preserve negative schedule evidence and lineage; no measured current-artifact speed lead.

## FE05 — Crow Nest: large Flash-Next streaming system, outside ordinary residency

Primary: https://github.com/nibor1896/crow-nest ; revision `07d9340` (September 25), `docs/architecture.md`, `docs/measurements.md`. Rust/CUDA with a roughly **104.7 GB CNQ4.5M Flash-Next artifact** on a 5090. Cold ordinary experts are streamed from about 46 GB of pinned host memory, alongside a bounded PLE cache. This does not fit the resident Qwen3.8 contract or the narrow native-Qwen4 exception. It is not validated by the diagnostic streaming exception either.

Author targets and measured service/decode rates use different workloads; do not flatten these into one speed number. The recorded six-seed/ten-task check is only 1/6 fully passing seeds (4 pass, 37 partial, 19 fail over 60 tasks). Byte identity to an installed build does not establish independent numerical or task quality. An attention split experiment was reverted for quality. Apache-2.0; disposition off-contract, with informative failed-experiment evidence.

## FE06 — q36 source recovery and name collisions

Original https://github.com/ambud/q36 remains inaccessible. A newly found mirror https://github.com/rochelleveritable865/q36 contains substantive CUDA code under Ambud-attributed commit `458eb01`; HEAD `b920bc6` changes the README and adds an unrelated downloadable ZIP. **No archive was fetched or run.** Git author metadata is not authenticated provenance, so this is a partially recovered attributed mirror, not verification that the original author endorses the current mirror.

`q36_cuda.cu` and `docs/ARCHITECTURE.md` expose MXFP4/W4A8, graph-wide PDL, and recurrent-prefix machinery. The recovered README concerns **Qwen3.6-35B MoE**, not the exact dense target; it reports 294 versus 280 tok/s against llama.cpp at short depth but 199.5 versus 208.3 at 90K. These mechanisms are already covered; it closes the earlier source-access gap without establishing a transferable gain. The recovered source states AGPL-3.0. https://github.com/Ninnix/q36 is the separate QuarkStar Vulkan/Metal/AMD project, not this CUDA engine.

## FE07 — lineage, deployment and early-project screening

All following addresses are primary repositories unless explicitly marked metadata-only. Their READMEs were screened, not full code audited.

| Repository | Disposition |
|---|---|
| https://github.com/azharul175706-ctrl/bw24 | Older Memra lineage: inspected commit `4f9ecba` credits Avi Fenesh and retains MIT copyright; targets 82-SM **5090 Laptop**, not desktop. Older licensing does not authorize later Memra code. |
| https://github.com/Resonance-Lab-AI/memrahh | Memra-named mirror, current README retains Avi/FSL identity; no independent engine claim. |
| https://github.com/MerkyorLynn/lynn-engine | Chinese first-party June 7 sunset retrospective: discontinued engine direction in favor of orchestration/weights; no current native win. |
| https://github.com/Avarok-Cybersecurity/atlas | Native Rust/CUDA, GB10 evidence, AGPL/commercial; hub GDN integration and negative draft research independently checked in final-technique-research.md. |
| https://github.com/harrrshall/qwenfast | PyTorch/Triton/FlashInfer Qwen3.8 FP8 on H200, not 5090 NVFP4. |
| https://github.com/infernet-org/foundry | vLLM deployment/profile system, not another native runtime. |
| https://github.com/AIdevsmartdata/chimere | Rust serving over customized ik_llama.cpp, 5060 Ti evidence; backend family already covered. |
| https://github.com/Danmoreng/gem16 | Native CUDA but Gemma checkpoint/16-GB specialization, no matched Qwen gain. |
| https://github.com/Onwcan/crucible | Native Rust/CUDA for own 120M model; throughput cannot extrapolate to 27B. |
| https://github.com/Praecise/praecise-engine | Acceleration layer whose initial backend is llama.cpp. |
| https://github.com/FILWYZ/nano-vllm-sm120-optimized | Chinese educational 5060-Laptop/Qwen3-0.6B fork; 8.23x is against its own SDPA compatibility baseline, not upstream vLLM. |
| https://github.com/lifeidle/qwen3.8-27b-5090-laptop-nvfp4-256k | llama.cpp laptop deployment recipe, not a new engine. |
| https://github.com/L1aoXingyu/qw38 | Source checkout explicitly bootstrap-only, no inference path. |
| https://github.com/Taf0711/trail | CUDA vector-add learning milestone; not a delivered inference engine. |
| https://github.com/zhaochengbiao005/qwen38-in-c | CPU C99/OpenMP, approximately 1 tok/s on Ryzen; not GPU engine. |
| https://github.com/Niko1221/Strata ; https://github.com/Apolog1ze-Dev/QwFNfer | Flash-Next ordinary-expert streaming, outside resident dense target. |
| https://github.com/Eden-Polaris/eden.cpp | Frozen llama.cpp fork, also Project-Glacie/eden.cpp lineage; multi-5060 recipe. |
| https://github.com/SRSWTI/shooting-brake | 118B across 5090 and two Intel B70 cards, outside single-GPU target. |
| https://github.com/hughmadden/mimo26f-afd | 5090 plus four DGX Sparks; comparison adds hardware, not equivalent single GPU. |
| https://github.com/SRSWTI/knivesysl-xe | Metadata identifies Intel SYCL port; main/master README unavailable. Closed as different hardware, not code-audited. |
| https://github.com/samihan342/Blackwell-MOE-Engine | Metadata lead; main/master README unavailable, API rate limited, and successful git ls-remote returned no HEAD or other refs. No delivered source tree or usable result was recoverable; likely empty repository, not an audited engine. |
| https://github.com/aaditagrawal/laya-inference-engine | 5070-Ti typed-decision single-forward workload, not autoregressive 27B generation. |
| https://github.com/xLLM-AI/xllm | Primary hardware table lists Ascend, Cambricon, Moore Threads, Hygon, MetaX and Iluvatar; not a demonstrated 5090 route. |
| https://github.com/guoqingbao/candle-vllm | Related Candle serving family; do not assume an exact alias of xinfer without ancestry proof. No new matched 5090 speed evidence. |
| https://github.com/SRSWTI/cradle-codec | KV transmission using HEVC/NVENC/10GbE. Prior tensor quantization remains lossy even with lossless video encoding; not resident decode acceleration. |
| https://github.com/2dameneko/ninfer-xx90-win | Windows/rotated-codec fork, README screening only; no independently established current win. |
| https://github.com/SirDebugALot/ninfer-3090 | Controlled ABBA 32K-prefill gain 785→1037 tok/s applies SM86/INT8 KV and different weights, not 5090 NVFP4. |

## FE08 — multilingual leads resolved to primary sources

A Korean AI Trends summary https://aitrends.kr/articles/101210 led to **https://github.com/mr-september/ninfer-sharp**. Its primary README corrects the summary: six listed effort values, not seven. This is a compiled prompt overlay and thinking-history rendering change, not a new compute engine. At matched medium effort the author reports 42.2% median fewer completion tokens and 22.6% less wall time over 24 tasks, with unchanged median decode TPS, INT8 KV and MTP3. Defaults also change from xhigh to medium, so unmatched default comparisons conflate interventions. Objective checks are not broad quality equivalence. Consider only an opt-in prompt/product feature, not a speedup preserving the same output distribution.

**https://github.com/Luce-Org/lucebox** is the current runtime identity; older **lucebox-hub** references and signalnine's hub mirror must not be counted as separate engines. `server/README.md` describes a C++/CUDA decode loop atop ggml, without libllama/PyTorch at runtime, Q4_K_M Qwen3.5/3.6 and DDTree budget 22. This is more than a llama.cpp UI. It also exposes a PFlash compression-proxy mode: prompt compression changes inputs and needs task-quality evaluation. The kernel agent traced DDTree, including the distinction between already-present target walking and a possible different frontier allocator, in final-technique-research.md.

**https://github.com/Eliovp-BV/paiton-vllm-plugin** resolves the R9700 Reddit lead https://www.reddit.com/r/ROCm/comments/1whv60r/ . It is a vLLM plugin/native-kernel integration for AMD, not a second Radiance alias. First-party benchmark records distinguish artifact, FP8 KV, 5-GiB cache and 8K window; newer 64K/experimental results are separate configurations. Its throughput is not a 5090 measurement. Primary article: https://eliovp.com/blog/paiton-qwen38-radeon-ai-pro-r9700 .

Korean Lucebox summaries and Japanese reposts led back to these known sources; TTS/CV-specific engines were excluded. Kairo's indexed description is a measurement/routing workbench, not evidence of a distinct inference kernel engine. These exclusions preserve breadth without falsely multiplying engine count.

## FE09 — inventory interpretation and extraction constraints

`engine-discovery.tsv` records primary repository identities and aliases only where evidence supports them. Recipe, wrapper, fork, independent runtime, kernel library and inaccessible-source rows remain different categories. In particular **gittensor-ai-lab/sparkinfer** is an engine, while **local-inference-lab/sparkinfer**, now **b12x**, is a kernel library; their performance and provenance must not be merged. BlackweLLM and TensorSharp receive separate root-owned source reports.

Most newly found mechanisms are already represented locally or need a different artifact. The strongest incremental leads from this pass are workload-gated copy proposals, quantified proposal-only head trimming, and specific negative schedule/quality results. None establishes a faster drop-in replacement for the current resident NVFP4 C1–4 engine. Missing licenses, AGPL, FSL and third-party terms matter before extraction; this report records source status, not a legal clearance.
