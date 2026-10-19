# Independent research quality review

Reviewed 2026-09-26. Scope: synthesis priorities, baseline/source consistency, strongest fork mechanisms, primary CUDA semantics, and evidence labeling. This review changed no runtime code, built no candidates and ran no GPU jobs. It is not a correctness certification of entire external forks.

## Findings resolved in the reports

| Finding | Direct evidence | Resolution |
|---|---|---|
| Copy proposals really use rejection sampling, but an old sampler contract | Wallawalla47 clone `600d8ac`: `src/models/qwen3_5/program/decode.cpp:753` constructs one-hot q; `src/ops/kernel/speculative_round.cuh:178–244` uses p/q and positive p−q; its `include/ninfer/ops/sampling.h:27` and `sampling_device.cuh:148` cap top-k support at20 | The copy-proposal recommendation and LE-02 now explicitly exclude qualification of current p-less/grammar. The feature is a proposal strategy, not a transplantable sampler/runtime. |
| Copy drafts do not eliminate all neural draft work at C>1 | Same fork `program/decode.cpp:739` sets `drafter_runs = !any_ngram || lanes.size() > 1` and overlays copy rows | `kernels-speculation.md` records mixed-batch limitation; 8–11% copy contribution is not an isolated throughput gain. |
| PDL support is already present in current NInfer | `src/core/pdl.cuh`; selected uses in `src/ops/gdn_input_proj/q4_q5/` and `src/ops/sparse_moe/` | The PDL recommendation is only new qualified boundaries/policy, not “add PDL.” |
| Fork PDL threshold is not guaranteed to count physical SMs | Fork `src/core/pdl.cuh:78` reads `%nsmid`; official PTX describes an SM-identifier bound, possibly noncontiguous | LE-02 and kernel report call it a heuristic and make no ordering-bug claim. |
| Once-per-tile V conversion already exists | Current `src/ops/kernel/gqa_attention_prefill_nvfp4.cuh:432–451` fills shared BF16 V before PV; QK is native NVFP4 MMA at361 | FlashInfer PR4346 is no longer presented as an unimplemented optimization for this primary NInfer route. |
| Keep-warm receipts used a materially different profile | LE-03 receipt inspection: groupwise-int target, BF16 KV, MTP3, reused30K prefix | The keep-warm recommendation states incremental-prefill/profile limitation, not current NVFP4 speedup. |
| Drafter training and target replacement have different scope | Current artifact authority and inspected QUASAR/DSpark sources | Reports distinguish a compatible trained companion from replacement base weights; no automatic checkpoint substitution or training-resource assumption. |
| LMDeploy TurboQuant PR cites an unrelated paper | PR4510 API patch plus arXiv2510.17153 title check | Correct method source is KS10; PR headline fidelity is not inherited. Implementation is identified as K3+QJL/V2, with transformation costs. |

## Primary mechanism checks

**PDL:** NVIDIA's CUDA Programming Guide section4.5 explicitly requires every producer block to trigger (or exit), the consumer to synchronize before accessing dependent results, and warns that overlap is opportunistic rather than guaranteed. Graph capture is a supported route. The inspected fork helpers implement trigger/wait and graph-only launch decoration; source inspection does not prove every changed kernel path is safe. A future port must inspect all early returns and all dependency reads at each selected boundary. Primary source: https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/programmatic-dependent-launch.html (updated2026-09-10, accessed2026-09-26). `%nsmid` reference: https://docs.nvidia.com/cuda/pdf/ptx_isa_8.8.pdf.

**Current sampling/grammar:** `include/ninfer/ops/sampling.h:62–77` includes p-less mode and per-verification-column token-eligibility masks; lines101 onward define a law different from the fork's bounded legacy sampler. `src/runtime/contract/tool_masks.cpp:118–141` computes masks for each tentative chain/tree prefix from the output session. `src/text/qwen/tool_grammar.h:33` says preview never advances committed grammar; `frontend.cpp:1068` commits only at publication. The conditional copy-proposal recommendation therefore correctly requires preservation of the current target distribution, tentative-prefix masks, termination and committed state. Primary external failure corroboration is ArcticInference issue183, recorded as a user issue rather than proof all suffix implementations fail: https://github.com/snowflakedb/ArcticInference/issues/183.

**Distribution argument:** with deterministic proposed token d and q(d)=1, acceptance probability p(d) plus rejection distribution p(x)/(1−p(d)) for x≠d recovers p. This is only the one-step mathematical argument. It does not establish that a copied implementation computed the actual masked/current p, used valid state at each tentative prefix, or preserved the product's RNG/publication contract. The reports now maintain that distinction.

## Baseline and synthesis consistency

- Baseline local source/authority paths were checked: all18 concrete local paths extracted from baseline exist. Current kernel/quantization report anchors also exist; `src/models/qwen3_5/...` references are explicitly external-fork paths and resolve in the stated clone, not this checkout.
- The leading matrix candidates are comparison schedules, not a promise to beat already single-pass DRAM-bound projections. Their classifier/oracle/full-Op/Engine gates match local requirements.
- The priority order is defensible as decision value and feasibility, not measured expected gain. Companion training is placed after bounded code research; residual-corrected quantization is lower priority and expressly changes arithmetic/layout expectations.
- No inspected external result is presented as a matched speedup over current NInfer. Throughput denominators, hardware, context capacity versus occupancy and checkpoint differences remain explicit.
- The synthesis and baseline now qualify rejection of strict/required/named tool choices as the **Responses** surface (`src/serve/responses_schema.cpp:482,516,523`). Chat/Anthropic schemas parse Required/Named, so a blanket endpoint-independent rejection claim is too broad. General JSON response format remains separately unsupported.
- `sources.tsv` was parsed with Python's CSV reader and a real tab delimiter: every row has five columns, with no literal backslash-t delimiters. The ledger explicitly says URL inclusion is not verification; per-report evidence levels distinguish source code, paper, author measurements and discovery leads.

## First-pass closure, superseded by the depth audit

No unresolved material research claim blocks using the prioritized recommendations. The endpoint wording clarification is applied. Final assembly includes `community-features.md`, `fork-inventory.tsv`, the campaign record and source ledger; their presence and the updated source-ledger structure were checked. The synthesis owner performs the final navigation and whitespace check.

The initial targeted gap sweep was insufficient to support a comprehensive closure. The reopened `technique-depth-audit.md` identifies additional high-priority grammar-overlap and recurrent-prefill candidates, plus narrower TMA/cache and quantization schedules. Within the original narrower sweep: mixed-KV allocation is codec-specific quality/capacity work, GPU suffix methods repeat the copy-proposal avenue, certified vocabulary pruning had no relevant demonstrated fast exact-target route, and the remaining general frameworks largely duplicate existing mechanisms or require excluded offload/distributed/model contracts. Candidate GPU measurements are intentionally future decisions, not unfinished work in this source-research deliverable.

## Final depth cross-review

The local-engine reviewer independently confirmed the current same-stream grammar serialization and distinguished the general-JSON feature from existing rich chain/tree tool masks. The kernel reviewer checked SparkInfer TMA cache hints and transposed tile geometry against actual code, clarified route-level benchmark attribution, and confirmed that FlashInfer's split-KV gate fix is not a missing NInfer algorithm. The revised synthesis replaces the old copy/PDL ranking with phase-specific grammar, GDN-prefill and cache/tile research, with no claimed measured local gain. The four depth audits explicitly resolve or bound the newly discovered leads.

## Third-pass correction to closure language

Independent omission searches found additional native engines and current backend work after the prior cross-review. Review found no contradiction in the then-known claims, but could not establish absence of unknown sources. The completeness audit therefore separates source discovery, primary-code inspection, screening and accessibility limits explicitly. A positive cross-review is not a certificate of Internet completeness.

## Final sustained-pass review

The four final reports follow topic/author/alias discovery and citation/issue chains, and supersede earlier discovery-closure language. The source ledger is an address inventory; engine and fork inventories separately record discovery and inspection depth. They must not be presented as hundreds of independently benchmarked engines.

Material checks include:

- **Kernel-library identity:** local-inference-lab's SparkInfer redirect leads to b12x; gittensor-ai-lab's SparkInfer is a separate engine. The previous TMA observations still stand for their actual source, but do not constitute prior inspection of b12x.
- **Certified recurrence convergence:** the b12x optimization uses a bitwise FP32 state match, including signed zero, before reusing subsequent local outputs. Its segment-summary arithmetic still requires independent numerical qualification. “Exact certificate” must not be shortened to “entire algorithm exactly matches NInfer.”
- **DFlash tree allocation:** root inspection confirmed `src/ops/kernel/dflash2_path_select.cuh` uses `kFrontier=2`, `kExpand=16` and Markov pair scoring. Existing target-sample/child traversal is not missing; DDTree's distinct proposal is how to spend the node budget. Its factorized probability objective is not automatically valid for unnormalized local scores.
- **Foreign engine improvements already present locally:** BlackweLLM's repeated-M1-to-batched verify change and historical graph/state-copy fixes are not new NInfer recommendations. TensorSharp's managed/GGML synchronization savings likewise do not imply an absent local CUDA Graph mechanism.
- **Numerical and causal evidence:** independent review confirmed Poseidon's checkpoint error metrics and held-fixed attention cache; wording was narrowed to top-1/log-probability results rather than unsupported full-logit identity. SGLang's cache-hit discrepancy was traced before recurrence to shape-dependent FP8 projection arithmetic, with a scoped patch/revert control. HyperQwen's dense checkpoint policy is not NInfer's sparse ladder.
- **End-to-end negatives:** KDA's broad cache policy regressed despite kernel wins; BlackweLLM's graph/context and confidence experiments lost or were neutral; an upstream K=0 draft-sync shortcut damaged later acceptance. These qualify research priorities instead of being discarded from the survey.

No local runtime or numerical behavior changed. The recommendations remain hypotheses with explicit first decisive checks; none is a measured speedup over this checkout. Final source/reference/whitespace verification and the complete campaign status are recorded in `campaign.md`.
