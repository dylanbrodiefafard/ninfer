# Technique depth audit: additions and corrected priorities

Research date: 2026-09-26. This pass reopens the first campaign's technique coverage. It found genuine omissions, especially SM120 recurrent-prefill implementations and the current grammar scheduling dependency. Evidence below is inspected source or clearly labeled publisher measurement; nothing was built or benchmarked locally. Existing runtime/GPU work is untouched.

## What materially changes the shortlist

| Opportunity | New evidence | Decision priority |
|---|---|---|
| Overlap tentative grammar-mask production with target compute | Current source serializes CPU matching before target verification; TensorRT-LLM implements the independent dependency pattern | High for tool-heavy requests; first establish exposed callback time, then bounded schedule research |
| Fused SM120 GDN prefill | FlashQLA has exact 27B head geometry and actual RTX5090 operator receipts, plus a materially different fusion | High prefill-specific comparison; ahead of generic GEMM-library adoption |
| Per-feature-tap draft-context projection overlap | TokenSpeed splits the feature projection by captured layer; current NInfer gathers taps then projects | Medium conditional; dependent on real preparation cost and spare GPU resources |
| In-kernel activation quantization feeding persistent GEMM | Newly merged FlashInfer SM120 output-projection kernel has producer/consumer ready flags | Medium for prefill shapes only; large-M diffusion results do not establish small-M gain |
| LM-head/sampling fusion | FlashSampling and Grout provide concrete implementations, but current p-less law needs global moments and speculative verifier needs probabilities | Narrow greedy/ordinary route research, lower than preceding items |

These are ranked by decision value, not expected measured speedup. Copy proposals, bounded PDL and companion alignment remain relevant, but the previous largely decode-centered list underrepresented prefill and tool-heavy execution.

## TD-01 — GDN prefill: compare fusion, not “add chunking”

Current NInfer already uses a sophisticated chunked algorithm. `src/ops/linear_attention/gated_delta_net/common.h:8` fixes chunk size64. `chunked/launch.cu:73–137` invokes prepare-WY/WU, state passing, then output, with W/U, adjusted V and chunk-state workspace. `state_passing.cuh:350` loops over chunks, uses MMA and retains the final FP32 state (`:586+`). BF16 input normalization may create private FP16 operands; any replacement must be checked against represented-input mathematics and the public FP32 recurrent-state boundary, not copied FLA rounding.

FlashQLA's SM120 implementation is distinct: `flash_qla/ops/gated_delta_rule/chunk/__init__.py:60–123` computes gate cumsum and KKT solve before a fused forward; `blackwell_sm120/fused_fwd.py:659–662` requires chunk32. Its kernel has different warp roles and combines state/output work, allowing inference to omit intermediate h storage. It is not merely swapping Python frameworks. Primary publisher receipt `benchmark/benchmark_results_5090.txt:135–153` uses Hqk16/Hv48, D128, the 27B geometry, CUDA Graph timing, torch2.12.1/CUDA13.0, FLA0.5.2, FlashInfer0.6.17 and TileLang0.1.13:

| Tokens, one sequence | FlashQLA | FlashInfer | FLA |
|---|---:|---:|---:|
| 2,048 |143us|284us|239us|
| 4,096 |339us|430us|528us|
| 8,192 |607us|814us|1,068us|
| 32,768 |2,195us|2,895us|4,366us|

These are publisher operator receipts, not model prefill; no NInfer comparator exists. **The receipt does not record `--no-cp`.** `benchmark/bench_gated_delta_rule.py:266,284,556` defaults to automatic intra-card context parallelism and constructs that context before timing. Treat these as the published auto-selected route, not measured non-CP fusion performance; the proposed `auto_cp=False` comparison has no established timing in this receipt. Reconstructing which automatic split was selected cannot substitute for missing run flags. The receipt also includes cases where FlashQLA loses to FlashInfer, so general “2–3x” is unsuitable. The default NInfer4K and explicit8K chunk sizes make the middle rows more relevant than training-length headlines.

**Separate exact reformulation from decay truncation.** FlashQLA automatic intra-card context parallelism uses gate-derived warmup; `blackwell_sm120/cp_fwd.py:81–86` defaults threshold−10. A small forgotten-state contribution is not identically zero. Initial comparison should isolate non-CP fusion (`auto_cp=False`), then independently qualify any truncated warmup including near-zero gates, nonzero initial state, long continuation, tails and grouped heads. Issue35 reports huge errors for specific SM120 variable-length/padding configurations; source inspection does not resolve the report. It is a gate for the relevant route, not proof all fixed-length forward paths fail.

FlashInfer also has a real SM120 CuTe GDN path, not only B200's PR3001: `flashinfer/gdn_kernels/delta_rule_dsl/delta_rule_sm120.py` contains FP32 initial/final state options and HMMA inverse machinery; `gdn_prefill.py:1184–1205` dispatches it. PR3479 and TRT-LLM15634 corroborate its introduction. The MLSys contest B200 scores use different hardware and harness reference and must not replace this direct SM120 comparison.

**Next decision:** compare complete current GDN Op at exact geometry/T and initial-state semantics against non-CP fused schedules; then use the fraction of whole-prefill time attributable to GDN to bound end-to-end upside. This is more concrete than a general FLA dependency proposal.

## TD-02 — Grammar matching currently precedes all target verification

`src/targets/qwen3_6/impl/runtime/speculative_target_impl.h:24–28` invokes `frame.tool_masks->enqueue` on `execution.device.stream` before `card.target_verify_batch` at32+. `src/runtime/contract/tool_masks.cpp:75–99` enqueues draft-ID/parent/count D2H, a `cudaLaunchHostFunc` callback running grammar matching, then mask/config H2D on that same stream. The stream dependency therefore places host matching before the target transformer computation. This is direct source evidence of serialization, not a measured bottleneck. Non-grammar requests and tiny masks may make it immaterial.

TensorRT-LLM's primary guided/speculative technical article describes the more parallel dependency: mask generation can run while forward executes, with the mask-ready event required only before sampling; it also describes graph-captured callbacks and fixed buffers. This does not require a GPU grammar implementation or a changed grammar law.

A candidate NInfer schedule would preserve draft-ID readiness, generate all masks for tentative chain/tree prefixes independently, and join before the first operation consuming sampling config/masks. Confirm that no earlier target computation actually consumes those pointers. Preserve `OutputSession` ownership, preview-versus-commit separation, active-lane identity, fixed graph addresses, callback error propagation and lifetime through drain. Moving work to another stream without those dependencies would be incorrect. Current graph buffer choices already solve some of these concerns; a general overlap scheduler or request-policy change is unnecessary.

**Next decision:** inspect a tool-heavy CPU/GPU timeline for exposed callback duration at C1 and C4. If substantial, this bounded overlap is preferable to importing another grammar library. A speed claim requires whole rounds with masks and publication, not isolated matcher timing.

TokenSpeed provides a second inspected implementation: `python/tokenspeed/runtime/grammar/capturable_grammar.py:379–413` records a fork event, waits on a side stream, performs speculative-candidate D2H/build/H2D, records a bitmask event, and joins separately. Its module comments document previous-token pinned-buffer ownership. This corroborates the scheduling mechanism without asserting that TokenSpeed's entire sampling stack supports SM120.

## TD-03 — Draft feature taps: an earlier overlap point

TokenSpeed's August12 Qwen3.8 article describes splitting `fc` column-wise by target capture layer, projecting each emitted hidden-state slice on an auxiliary stream, accumulating, then normalizing/writing draft KV before the remaining target layers finish. Its deployment evidence is 2.4T MoE on B200/B300, not5090.

Current NInfer `text_context_impl.h:168–205` captures taps with scatter/copy. `dflash_impl.h:140–197` subsequently projects assembled features and normalizes them; lines1015–1036 prepare accepted/pending ragged prefixes and append draft context. This establishes a different overlap opportunity from blindly running the next draft before verification has accepted it.

The algebra permits a split linear reduction, but floating-point summation changes and persistent context writes require qualification. Per-tap partials must refer to the same tentative tokens, and rejected candidates must never become committed context. A single5090 may lose to contention or extra launches; the remaining target layers must provide useful slack. Inspect the exact DFlash2 artifact's feature projection/normalization order and measured preparation share before implementation. Training a companion with different tap layers is a separate artifact/quality decision, not a free runtime switch.

## TD-04 — New SM120 quantization/GEMM schedules

TileLang v0.1.13 (August3) added SM120 block-scaled NVFP4 MMA, PR2364, with package-pingpong lowering and a nonpersistent example. Its reported8192-cubed throughput is a large square GEMM receipt, not C1..4 inference. This is useful schedule/code-generation evidence for prefill; the earlier open support-request1592 is stale evidence of capability. No new runtime dependency is needed to inspect generated CUDA/PTX choices.

FlashInfer PR5521 (mergedSeptember25) implements MiniMax-H3 RMSNorm/AdaLN/quantization plus FC1/SwiGLU in two launches. Its interleaved gate/up rows keep paired outputs in register fragments; a producer folded into warp0 avoids a ninth warp and register spill. NInfer already has fused `LinearSwiGLU` and M256N128 three-stage TMA (`src/ops/linear_swiglu/nvfp4/nvfp4_linear_swiglu_w4a4_tma.cu:15`). Do not rank “fuse SwiGLU” as new.

PR5524 (also mergedSeptember25) is more interesting as a scheduling pattern: one persistent SM120 output-projection kernel lets spare producer warps quantize M tiles and publish ready flags, while TMA consumers and MMA proceed. NInfer `nvfp4_linear_swiglu_w4a4.cu:87` still launches activation quantization separately. The PR's M33K..110K, dimensions7168→5376, indexed gate and FP8/NVFP4 receipts are diffusion-specific; they establish neither small-M benefit nor a faster NInfer Op. Its host `cudaMallocAsync`/memset scratch is not a suitable direct transplant into NInfer's preplanned graph workspace. Compare the mathematical quantization recipe and global/block scales, then shape-level classifier and complete operator cost.

PR5545 remains open/experimental. It fuses packed-varlen SM120 Sage3-style FP4 QK and PV without Sage3's dense block-mean correction tensor, but reports roughly0.19 relative L2 on Gaussian inputs. Useful mechanism for vision's noncausal packed attention, not an admitted quality envelope. Current `src/ops/kernel/vision_attention.cuh:104+` is a separate BF16 flash kernel; text Sage3 support does not prove vision already uses FP4. Qwen vision geometry and image-task quality must gate any route, and changing visual token counts/downsampling is a semantic change rather than engine-only speed.

### More targeted than a generic L2 policy: TMA weight-only eviction

SparkInfer `kernels/csrc/cuda/fused/prefill_nvfp4_sm120.cu:35–129` overrides CUTLASS load to attach `TMA::CacheHintSm90::EVICT_FIRST` only to B and SFB; A/SFA remain normal. The stated mechanism is protecting repeatedly used activations from the single-pass weight stream. NInfer `src/ops/linear/nvfp4/nvfp4_w4a4_tma.cuh:184–193` emits TMA load without `.L2::cache_hint`, so this exact mechanism is absent from that helper. Earlier ordinary-load/window cache-policy failures do not alone reject a per-TMA policy. Conversely, a DRAM-bound classification does not guarantee a win: it matters only if L2 activation eviction materially adds reads/stalls. SparkInfer's comments report that evicting A instead lost61% and adding A evict-last was neutral; those are publisher controls, not local measurements. Its M128 argument must be recalculated for NInfer's M256 tiles and4K prefill, where weight reuse across M tiles changes the ideal policy. This is a narrower exact-shape candidate than a wholesale compiler change, conditional on the existing classifier.

## TD-05 — Output head and sampler: exact categorical is not exact p-less

FlashSampling fuses LM-head matmul with tile-local Gumbel maxima, then reduces candidates, eliminating full-logit writes. Its current May6 repository revision reports up to10% TPOT improvement on tested datacenter models; the March arXiv v1 says19%, so use version-specific claims. It does not provide a5090/NInfer result. The paper explicitly leaves top-k/top-p/masking integrations as future implementation, despite describing their mathematical extensions.

NInfer p-less is not plain temperature-softmax: `include/ninfer/ops/sampling.h:101–119` defines a support cut from global first/second probability moments, a support floor and typical-exclusion/fallback rules. Current `sampling.cuh:123–127,197–214,330+` computes moments then admitted mass/sample. Tile-local Gumbel maxima over raw logits do not implement that law. Exact p-less fusion would need global moments before admission and either retained logits or another projection/read pass; weigh that cost against saved traffic. Speculative rejection additionally needs target probabilities and residual sampling. Greedy fused head+argmax is narrower and mathematically straightforward, but must preserve masks, tie rules and any probability consumers.

Sorting-free FlashInfer top-k/top-p and Qrita are leads only for the non-p-less sampler. They are not replacements for the custom p-less path. Likewise vocabulary candidate pruning cannot be called exact unless it certifies every omitted logit cannot affect the actual global support/moments.

Grout (`23ae5a3`, July1) is a useful omitted native engine/compiler example, not an NVFP4 baseline. Its5090 receipt is FP16 Qwen3-4B batch1; CUDA13.2+ differs from current toolchain. `src/model.rs:2304–2312` has optional fused LM-head argmax guarded by greedy/no-logit conditions; default is false. It also fuses QK norm/RoPE/KV writes, concepts largely already present here. Its171tok/s cannot establish improvement for a27B quantized hybrid. cuTile-Rust's safety/compiler result is not evidence that rewriting NInfer in Rust accelerates it.

PegaInfer's inspected `docs/models/qwen3/dspark-integration.md` adds a real5090 DSpark comparison: Qwen3-4B, matched block7 DFlash, greedy, +3.6% geometric mean across C1/C4/C8; C1 aggregate−0.1%, C4+4.8%, accepted drafts2.52 versus2.30. Its rank256 Markov head is loaded while initial confidence scheduling is not. This confirms implementability but gives a much more restrained expectation than a universal DSpark acceleration claim. It is a different checkpoint; the anchor-first layout bug and checkpoint-specific tensor contract matter. No p-less qualification follows from a greedy receipt.

## TD-06 — Long-context and alternate-format implementations

Maharajahu's long-context repo (`5baa759`, September5) is unusually explicit evidence: its validated llama.cpp patch removes full Q4 KV→FP16 materialization during MTP verification, retains tile MMA, and reports53.76→80.53tok/s at189,935 populated tokens, three512-output repetitions. The “vision” row has a loaded projector but text input; real-image support is a separate smoke test. NInfer `gqa_attention_decode_nvfp4.cuh:48–62,156+` already stages compressed tiles and reads physical page IDs directly, so this is not an unimplemented49.8% gain. It does support checking actual long-context occupancy rather than only configured capacity. Current local prior evidence at131K already reports double-staged KV loads and66–68% DRAM throughput; generic pipeline proposals must address that baseline.

Qwentin (`95952dd`, August18) has real custom SM120 FP6 code, 4bit-K/E4M3-V, MTP tree verification and chunked prefill. Its FP6/alternate KV quality and speed cannot be transferred to the fixed NVFP4 artifact. `src/forward_qwen.cu:69–77` describes native FP6 `ldmatrix` unpack as an explicitly default-off probe, not a delivered default speedup. Lines79–87 similarly describe default-off per-tile readiness replacing a grid barrier. Both are useful implementation leads where operand format and dependency shape match; the exact artifact presently does not use FP6 matrices. Advertised teacher-forced top1 agreement is not task-quality parity. Prefix-cache, split-attention and one-read batched verify mostly duplicate existing NInfer capabilities.

Cache-aware admission that bypasses FIFO or lets another prefiller preempt a decode lane changes the scheduler contract. Within-contract opportunities are phase-aware chunk sizing and reuse-aware work estimates, but NInfer already has mixed-phase adaptive decomposition and published4K/8K tradeoffs. No new cache heuristic was found with sufficient exact-workload evidence to outrank the concrete dependency/fusion candidates above.

KVarN is an additional codec lead, not a measured NInfer acceleration: its primary repository describes Hadamard rotation, iterative row/column variance normalization and K4V2 presets, with FP16 dense-path compute and a fixed decode workspace that matters on tight single-GPU budgets. The showcased Qwen3-32B TP2 reasoning/capacity comparison is against FP16/TurboQuant, not existing NVFP4 KV. The normalization is a material algorithmic difference from simply changing bit counts; a future quality/capacity comparison must include rotation/normalization cost, metadata, partial-tile handling and recurrent long-generation quality. No matching5090/Qwen3.8 result was established in this screen; full backend code audit was not performed.

## Source ledger

All accessed2026-09-26. Code inspection does not imply a local performance reproduction.

| ID | Date/version | Evidence | Primary source |
|---|---|---|---|
| TD01 | FlashQLA cdfcb99, September26 | code + publisher5090 operator results | https://github.com/QwenLM/FlashQLA ; https://github.com/QwenLM/FlashQLA/blob/main/benchmark/benchmark_results_5090.txt |
| TD02 | August9 issue | unresolved user-reported numerical failure | https://github.com/QwenLM/FlashQLA/issues/35 |
| TD03 | current source / historical introduction | code + integration issue | https://github.com/flashinfer-ai/flashinfer/blob/main/flashinfer/gdn_kernels/delta_rule_dsl/delta_rule_sm120.py ; https://github.com/flashinfer-ai/flashinfer/pull/3479 ; https://github.com/NVIDIA/TensorRT-LLM/issues/15634 |
| TD04 | current primary technical guide | implementation dependency design, no5090 claim | https://github.com/NVIDIA/TensorRT-LLM/blob/main/docs/source/blogs/tech_blog/blog12_Combining_Guided_Decoding_and_Speculative_Decoding.md |
| TD05 | August12 | publisher architectural explanation, different model/hardware | https://lightseek.org/blog/tokenspeed-qwen3-8.html |
| TD06 | August3 release | merged compiler capability + largeGEMM claim | https://github.com/tile-ai/tilelang/discussions/2848 ; https://github.com/tile-ai/tilelang/pull/2364 |
| TD07 | mergedSeptember25 | source + publisher complete-Op diffusion receipts | https://github.com/flashinfer-ai/flashinfer/pull/5521 ; https://github.com/flashinfer-ai/flashinfer/pull/5524 |
| TD08 | openSeptember25 | experimental recipe + stated numerical error | https://github.com/flashinfer-ai/flashinfer/pull/5545 |
| TD09 | March16 paper / May6 README revision | theory + code + publisher datacenter tests | https://arxiv.org/abs/2603.15854 ; https://github.com/FlashSampling/FlashSampling |
| TD10 | March10,2025 | primary sorting-free algorithm | https://flashinfer.ai/2025/03/10/sampling.html |
| TD11 | June2026 paper / Grout July1 | code + publisherFP16 Qwen3-4B5090 receipt | https://arxiv.org/abs/2606.15991 ; https://github.com/huggingface/grout ; https://github.com/NVlabs/cutile-rs |
| TD12 | September5 | inspected patch + publisher matched llama.cpp receipts | https://github.com/Maharajahu/Qwen3.8-27B-RTX5090-CUDA-Kernels |
| TD13 | August18 | custom source + publisher alternate-format results | https://github.com/kacper-daftcode/qwentin |
| TD14 | inspected current source | per-TMA cache-hint code + publisher controls | https://github.com/gittensor-ai-lab/sparkinfer/blob/main/kernels/csrc/cuda/fused/prefill_nvfp4_sm120.cu |
| TD15 | TokenSpeed8bb4a47 | side-stream grammar dependency code | https://github.com/lightseekorg/tokenspeed/blob/main/python/tokenspeed/runtime/grammar/capturable_grammar.py |
| TD16 | inspected integration report | publisher5090 greedy different-checkpoint comparison | https://github.com/pegainfer-project/pegainfer/blob/main/docs/models/qwen3/dspark-integration.md |
| TD17 | June2026 / current README | primary method/backend description; no local code qualification | https://arxiv.org/abs/2606.03458 ; https://github.com/huawei-csl/KVarN |

## Scope of the revised conclusion

The first pass's assertion that no additional higher-priority admissible mechanism emerged was too broad: this pass found concrete current-source grammar serialization and a direct5090 recurrent-prefill comparator. These findings justify changing the research shortlist. They still do not establish a speedup over current NInfer. The material unanswered questions are measured exposure of mask callbacks, current GDN-prefill fraction, exact-state correctness of candidate fusion, and preparation/quantization cost at the real shapes. Resolving those requires a separately authorized implementation/performance task; this source-research pass does not consume GPU resources to answer them.
