# Coverage audit and remaining observability limits

Retrieved2026-09-26. This third pass responds to an explicit request for absolutely comprehensive investigation. A100% census cannot be verified: private/deleted repositories, unindexed branches, inaccessible discussions and future publications have no enumerable boundary. The prior claim that no material lead remained was stronger than the search evidence supported. This pass found additional engines and records them instead of treating repetition of a search as proof of completeness.

## CA01 — Lumen: real native CUDA engine, different arithmetic

Sources: https://servelumen.com/ ; https://github.com/faisalmumtaz89/Lumen ; https://github.com/faisalmumtaz89/Lumen/blob/main/CHANGELOG.md . Inspected snapshot7b19010, September25, outside checkout. Native Rust orchestration and CUDA C/NVRTC, Qwen3.8-27B Q4_0/Q8_0/K-quants/BF16, single-stream focus. This is an independent engine, not a wrapper or closed product. The website's5090 benchmark protocol is greedy, no speculation,1024prompt/128generated, five warmed runs, clocks pinned. Format-tagged rows explicitly differ in weight bytes. No NVFP4 same-artifact comparison was established.

Inspected `crates/lumen-runtime/src/cuda/shaders/attention_decode.cu` and runtime split defaults. One CTA per KV head/chunk shares K/V across all grouped query heads, with compile-time group/head geometry; the long-context split count is bounded. The changelog reports Qwen3.8 Q4_0 ordinary decode82.59→85.56tok/s and records context-specific losses as well as wins. F32/F16 KV and its scalar arithmetic differ from current NInfer. NInfer already has grouped tiled compressed attention and split execution, so this corroborates reuse/split tuning rather than adds an absent algorithm. The source preserves FP64 reference comparisons and near-tie adjudication; that is more useful than treating greedy sequence equality as the only numerical criterion. Image-generation optimizations concern a different model/product and are screened out.

## CA02 — QW3/KVMem: distinct context-memory feature, not exact million-token attention

Sources: https://github.com/kvmem/kvmem-qw3 ; https://arxiv.org/abs/2609.04852 ; https://arxiv.org/html/2609.04852v1 . Inspected source1cf3b2f, September19. QW3 owns CUDA execution for the hybrid27B family. KVMem retains historical KV on GPU/RAM/NVMe and retrieves a query-dependent subset into the active context. The paper's consumer result uses a24GB5090 **Laptop**; addressable1M history is not1M simultaneously attended tokens or desktop5090 throughput.

`docs/KV_Memory_Paper.md:535–548` explicitly says selecting a different block set changes computation while DeltaNet still sees the full history. Identity-selection parity does not prove sparse-selection equivalence. Source documents de/re-RoPE and state/position handling, and real retrieval methods beyond LRU. This is a meaningful optional agent-memory product direction, but changes attention/context semantics; it cannot be silently adopted as an exact faster kernel. Qwen4's host/disk KV prohibition also remains in force. Existing Qwen3.8 prefix retention is a different feature from query-dependent historical block selection.

A narrower useful source is **reusable image embeddings**: `src/vision_gpu_frontend.cpp:628–700` looks up fingerprint plus image equality, retains encoded image features, deduplicates within requests, and separates host restores from GPU LRU promotion to avoid scan pollution. Current NInfer `vision_context_impl.h:490–562` encodes encountered items into transient request output, with within-session aggregate/active-item reuse. Exact prefix reuse may already skip the image span. A cross-request image-feature cache is therefore only worth investigating when identical images must be encoded again after changed prefixes or retention misses. Profile that workload first; preserve artifact, preprocessing/grid, image equality and ownership in cache identity. Do not import CPU vision execution or Qwen4 host-KV tiers. This is a conditional feature lead, not an established latency gap for ordinary prefix hits.

## CA03–CA06 — Additional primary engine screens

| Source | Established scope | Disposition |
|---|---|---|
| https://github.com/mudler/vllm.cpp | Independent C++ engine, Qwen3.6NVFP4 throughput receipts on **GB10**; documents build-only versus runtime gates and superseded llama.cpp denominators | Add to engine landscape. README says speed proven on GB10, not5090. Generic scheduler/compiler migration is not a target improvement; exact kernel lead required. Previously indexed STATUS path returned404; current README available |
| https://github.com/zml/zml ; https://zml.ai/posts/llmd/ | Zig/OpenXLA/MLIR runtime; July8 LLMD alpha, CUDA and other backends, paged/prefix serving, DFlash on Gemma | Native independent stack; primary article offers datacenter/other-platform results, no matched5090NVFP4 result established. Python-free serving and prefix cache already exist here |
| https://github.com/JustVugg/colibri | PureC frontier-MoE engine with disk expert streaming; surfaced through Japanese technical article | Distinct engine captured; ordinary-weight streaming/distributed oversized-model capacity does not satisfy current native resident27B contract |
| https://github.com/deepshnv/pipeshard-mlsys26-ae | MLSys client inference artifact, explicitly VRAM-constrained/offload-oriented5090 experiments | Capacity/offload source captured. No same-artifact resident NInfer improvement established; no binaries/weights downloaded |

The following primary source channel is also captured: https://github.com/notwitcheer/sm120-field-guide . It contains dated deployment, quantization, draft-training and measurement observations. The technique omission report screens relevant claims; compatibility anecdotes are not promoted into kernel results. A nominal5090 benchmark with24GB must be resolved as laptop/desktop before comparison, not silently grouped with the32GB target.

A curated-directory cross-check used https://github.com/moduvoice/awesome-rtx-5090 and followed the relevant engine/recipe entries to primary repositories. It added https://github.com/brontoguana/krasis : current Rust/CUDA hot path with GPU compute and host-RAM expert residency (not accurately dismissed as simply CPU math). Its MoE streaming/capacity contract remains different from resident dense27B; compact HQQ/KV profiles need independent quality admission. https://github.com/Ark0N/Qwen5090 is a multi-backend deployment/agent package, not another independent kernel result. The listed seanyourhighness vLLM recipe was already deeply covered. The directory's image/audio/scientific workflows do not become NInfer requirements merely by running on5090.

PipeShard's VLMOpt specifically offers CPU-resident vision weights streamed to GPU and tiled-Q/FlashAttention to avoid quadratic scratch; the former is excluded ordinary-weight streaming and the latter substantially overlaps existing tiled vision attention. Colibri also documents grammar-forced proposals and precision-sensitive MTP acceptance, but on a different GLM artifact with streaming economics; these reinforce existing speculation/grammar qualification rather than establish a new27B speedup.

## Search coverage matrix

This is a record of investigated avenues, not an unprovable assertion that each channel is exhausted.

| Avenue | How exercised / durable evidence | Limit |
|---|---|---|
| Direct engine discovery | Broad5090/sm120/sm120a/CUDA/nativeengine searches, exclusion queries removing known engines; primary repos followed | Search indexing and ranking are incomplete |
| NInfer lineage |469 accessible network entries; descendant enumeration; selected changes, tags and nondefault branches in source-channel-audit | Most forks metadata-only; all commits of all branches not audited |
| Mainstream backends and forks | vLLM/SGLang/TRT-LLM/llama/ExLlama plus release notes and actual PR diffs; ecosystem-omission-audit | Moving branches and unmerged work can change after retrieval |
| Community | Reddit LocalLLaMA/LocalLLM/Vllm/Qwen, HN, author gists/blogs, HF model cards/discussions; technical claims traced to primary material | Private Discord/closed forums/deleted posts unavailable; anecdotes remain anecdotes |
| Non-English discovery | Chinese/Russian engine queries; Japanese source discovery; primary code followed | Not every language/site indexed; translated mirrors not independent evidence |
| Papers and kernel tools | arXiv/OpenReview, author code, NVIDIA official docs, FlashInfer/FLA/TileLang/CUTLASS and new recurrence/speculation work | Paper results are workload-specific and not reproduced here |
| Execution phases | Startup/idle, host prompt work, dense/GDN prefill, decode, draft/verify, masks/sampling/head, longcontextKV, vision, continuation | New hypotheses require phase attribution on this checkout |
| Alternative contracts | Sparse/retrieved context, quantized state, lowbit artifacts, streaming/CPU/distributed backends explicitly screened | Captured does not mean admissible under current product |
| Evidence falsification | Negative results, current predicates, actual hardware, arithmetic, occupied context, concurrency denominators and harness bugs | Source audit cannot establish local end-to-end gains |

Representative orchestrator queries included `"RTX 5090" "inference engine"` with known-engine exclusions, `"sm120" inference engine github Qwen`, `"5090" "inference" "engine" "Rust" CUDA`, `"sm_120" "inference" engine CUDA github`, Chinese engine terms, and explicit ZML/MAX/Fastllm names. These are supplemented by each agent's query and source records, rather than a count of returned search hits.

## Interpretation

The campaign now has broader and more auditable coverage. Every promising lead actually discovered in this pass must receive an inspected, screened, duplicate, inaccessible or out-of-contract disposition before closure. Unknown sources may still exist. This report does not promise an unknowable100% completeness level, nor equate source count with depth. A future named source or release is a reason to reopen its relevant branch, without invalidating the qualified findings already established.
