# RTX 5090 inference engines: opportunities for NInfer

Research cutoff: **2026-09-26**. Target: NInfer Qwen3.8-27B NVFP4, one RTX 5090. Original local comparison: `qwen4` at `176e04cc`, C=1..4. Corrected branch: `experimental` at `d29841e0`, whose repository supports startup-fixed C=1..6. This report is a research recommendation, not an implementation contract or a new performance benchmark.

**Read [the experimental applicability review](experimental-review.md) first.** The external research remains relevant to the confirmed Qwen3.8-27B NVFP4 model; a broad restart is unnecessary. The top three leads survive, but grammar ownership/graph transfers need revision, TMA applies to A4 prefill rather than the newer A8 verification path, and the local performance baseline must be rebuilt. Experimental already includes shorter adaptive DFlash arms and several decode fusions, and records a negative A8 PDL experiment. The review supersedes local-baseline assumptions in the historical reports below. The old GPU queue remains cancelled.

The assembled campaign contains **420 deduplicated source addresses**, **126 classified repository identities**, **469 accessible fork-network records**, and **192 selected branch-head records** in 28 files including the branch review. These inventories overlap and measure different things: addresses are not independent studies, repositories are not all independent engines, and metadata enumeration is not code or performance verification. Inspection depth and dispositions are recorded alongside the sources.

## Findings that matter

The final sustained pass adds systematic topic/author/alias discovery and follows primary issue, citation and benchmark trails. Read the four `final-*.md` reports and `engine-discovery.tsv` for the current expansion. Earlier statements implying discovery was exhausted are superseded: every tracked material lead has a disposition, but unknown public/private work cannot be ruled out.

The strongest new leads are **overlapping tool-grammar generation with target verification, fused SM120 GDN prefill, and exact-shape TMA cache/tile schedules**. These emerged from the reopened depth audit and replace the first pass's excessive emphasis on copy drafting and generic launch overlap. Earlier draft-feature preparation, better allocation of the existing DFlash tree budget, and measured companion alignment remain conditional opportunities. Third-pass code inspection also identifies a Qwen27B prefill conv/QK-normalization/gate fusion boundary in MetaZenith; the ecosystem report distinguishes it from local Qwen4-only fusion. The final pass adds b12x certified-convergence prefill, reconstructed-BF16 recurrence and finite triangular inversion as concrete GDN comparators, plus cache-policy experiments where an isolated kernel win reversed end to end. No inspected source establishes a matched speed advantage over this checkout using the same artifact, sampling, KV, context, concurrency and workload. That is not proof NInfer is fastest.

NInfer already has much of what competing engines advertise: CUDA Graphs, compact batched decode, NVFP4 KV, quantized DFlash2, MTP, adaptive draft length, optimized proposal heads, recurrent-state-aware prefix reuse, Qwen3.8 RAM/SSD prefix tiers, and tool-schema constraints. Those are not new recommendations. Conversely, general JSON response constraints and restart-persistent Responses state are meaningful feature distinctions from the current server.

Radiance is primarily an AMD/HIP fork. Its transferable mechanisms need mathematical and scheduling review; its token rates are not 5090 results. Some “Blackwell” kernels explicitly exclude SM120. Several impressive exact-model reports use two 5090s or substantially different checkpoints. The detailed reports preserve these distinctions.

The final discovery pass recovered additional projects that ordinary engine-name searches missed:

| Source family | What was established | Consequence for NInfer |
|---|---|---|
| Knivesysl, BlackweLLM, Antelope, TensorSharp | Distinct runtimes with source to inspect; their target formats, hardware and backend ownership differ | Copy proposals, batching and recurrence schedules can be compared individually; none establishes a matched current-target speed win |
| Qwarz and Lucebox | Substantive execution/draft work over ExLlamaV3 and ggml respectively, beyond a user interface | Proposal-only head trimming and tree-budget allocation are concrete; donor weights and prompt compression change the comparison |
| VoidInfer and Sharp | Explicit NInfer descendants not safely discovered from fork-network metadata alone | Preserve negative EXL3 schedule results and separate prompt/thinking-effort changes from faster compute |
| Poseidon and HyperQwen | Exact-model backend modifications with useful state, cache and numerical evidence | Source-limited attention claims remain qualified; long-context restore and alternating-conversation tests expose important failure modes |
| b12x, formerly local-inference-lab/SparkInfer | Kernel library distinct from gittensor-ai-lab/SparkInfer | Newly inspected GDN algorithms, not a second name for the already-audited engine |
| Crow Nest, Paiton, Atlas and other adjacent projects | Ordinary-weight streaming, AMD execution, GB10-specific paths or other changed targets | Retained in the inventory with explicit boundaries, rather than silently omitted or presented as equivalent 5090 engines |

Full source and alias dispositions are in `final-engine-discovery.md`, `final-backend-research.md` and `final-independent-research.md`. A runtime can be independent at the scheduler/model layer while sharing a kernel backend; “engine count” alone is not a useful measure of novelty.

## Reading order and provenance

| File | Purpose |
|---|---|
| `experimental-review.md` | Current branch applicability, architectural changes, surviving candidates and superseded recommendations |
| `final-engine-discovery.md` | Systematic topic/author/alias discovery, new engines and source-access limits |
| `engine-discovery.tsv` | Canonical engine/backend inventory with aliases, evidence depth and dispositions |
| `final-backend-research.md` | Latest issue/PR chains, source-limited kernels, quality failures and benchmark corrections |
| `final-technique-research.md` | b12x, alternative GDN algorithms, tree budgets and paper-to-code citation paths |
| `final-independent-research.md` | BlackweLLM, TensorSharp and independent article/measurement review |
| `fork-depth-audit.md` | Expanded469-fork discovery, inspected feature forks and omitted native engines |
| `competitive-depth-audit.md` | SparkInfer, raw benchmark contradictions, current PR diffs and Radiance descendants |
| `technique-depth-audit.md` | New grammar/prefill/cache mechanisms and numerical/shape gates |
| `coverage-depth-audit.md` | Independent TokenSpeed inspection, secondary engine screens and coverage limits |
| `completeness-audit.md` | Third-pass coverage matrix, Lumen/QW3 and explicit limits on completeness |
| `source-channel-audit.md` | Additional native engines, nondefault fork branches and community discovery |
| `ecosystem-omission-audit.md` | Current backend releases, forks and overlooked deployment sources |
| `technique-omission-audit.md` | Additional papers, recurrence/speculation and toolchain evidence |
| `baseline.md` | Branch correction plus preserved original capabilities, measurements and rejected experiments |
| `local-engines.md` | NInfer forks and independent local engines, concrete code differences |
| `community-features.md` | Community claims, workload lessons and beneficial feature opportunities |
| `vllm-engines.md` | vLLM, Radiance, SGLang, TensorRT-LLM and deployment overlays |
| `kernels-speculation.md` | FlashInfer, Marlin, attention/KV and speculative methods |
| `quantization-quality.md` | QUASAR, QAD, GDN quantization, ARCQuant, SharQ and companion compatibility |
| `compiler-runtime.md` | Emmy, Mirage and why compiler/megakernel results do not directly transfer |
| `landscape-coverage.md` | Additional engine families, screened exclusions and research stopping boundary |
| `review-notes.md` | Independent checks of the leading candidates and resolved evidence problems |
| `campaign.md` | Scope, research coverage, completion and verification record |
| `sources.tsv` | Deduplicated source addresses and the reports that cite them |
| `branch-inventory.tsv` | 192 observed head refs across11 selected repositories, with inspection dispositions |
| `fork-inventory.tsv` | 469 accessible network fork names, URLs and descriptions, distinguishing metadata-only entries from inspected forks |

Each detailed report separates inspected code, primary author measurements, papers, issue/PR discussions, and community leads. The stable branch IDs (VE, KS, Q, CR, LC and local-engine IDs) identify evidence in those reports. Source URLs are preserved there and in the ledger; external results were **not independently benchmarked** during this campaign. Repository snapshots were inspected outside this checkout, without installing them or downloading model weights.

## Research priorities after the final pass

This ranking originated against `176e04cc`. The branch review above supplies its current disposition; it is not a measured performance ordering for experimental.

Ranked by relevance and decision value, not a forecast of percentage gains. Workload determines value: tool-heavy rounds, cold prefill and sustained decode have different bottlenecks. These are recommended future experiments, not uncompleted research assignments. Detailed evidence is in the final and depth reports; earlier reports retain historical discovery and corrected source context. Priority is an engineering judgment about relevance and decision value, not a measured ranking of expected speedups.

| Priority | Performance candidate | Evidence and boundary | First decisive check |
|---|---|---|---|
| 1 | Overlap grammar-mask production with target verification | NInfer enqueues D2H, CPU matching and H2D on the target stream before forward. TRT-LLM and TokenSpeed expose the independent fork/join pattern. Direct source difference, no measured NInfer bottleneck yet | Profile tool-heavy C1/C4 callback exposure; preserve candidate readiness, ownership, graph addresses and mask-ready join before sampling |
| 2 | GDN prefill: fusion, segment parallelism and recurrence arithmetic | FlashQLA has actual5090 Hqk16/Hv48,D128 receipts and a different chunk32 state/output fusion. Published auto-selected receipt at4K:339us versus FlashInfer430us/FLA528us; benchmark defaults to automatic context parallelism, so this is not established non-CP performance or a NInfer comparison | Compare complete GDN pipeline and FP32 state against the oracle, including the MetaZenith conv/QK-norm/gate preparation boundary; exclude decay-truncation approximation initially; resolve applicable varlen failures; compare b12x certified convergence, llama reconstructed-BF16 register recurrence and Kachua finite inversion as separately qualified alternatives; measure whole-prefill share |
| 3 | Selective TMA cache hints and shape-specific verify/prefill tiles | SparkInfer applies B/SFB evict-first while retaining normal A/SFA; current NInfer helper lacks per-TMA hint. Native swapped small-row schedules are additional candidates; SGLang KDA reports broad scale retention regressing full-model throughput by 0.76%, then a narrow M9 down-projection policy gaining 0.98% on PRO 6000 | Classifier-admitted exact geometry only; activation reuse and weight reuse across M tiles determine policy; include all preparation and complete Op cost |
| 4 | Earlier per-feature-tap draft-context projection | TokenSpeed partitions feature projection and starts partial work while later target layers run; NInfer currently assembles taps first | Establish preparation share and available slack; qualify changed reduction and tentative/committed context ownership; reject contention losses |
| 5 | Allocate the existing DFlash tree budget across depths/branches | DDTree supplies a best-first marginal-probability surrogate; local code already performs exact p-less target-sample/child walking but builds with a fixed two-node frontier. H200 paper, no local win | Preserve GPU residency, graph/node capacity and ancestor state; compare accepted output per complete round against current Markov-score builder. The paper's optimality does not automatically apply to local scores |
| 6 | Exact-target/template/workload companion alignment | Acceptance can dominate round economics; external receipts demonstrate strong corpus dependence | Measure acceptance by real request phase before changing or training a companion; hqmtp retracted an apparent NVFP4 mismatch after correcting its evaluation, so do not presume one; respect DFlash2/DSpark artifact differences |
| 7 | In-kernel quantization plus persistent GEMM | Recent FlashInfer SM120 code supplies a real producer/consumer schedule, but receipts are large diffusion shapes | Admit an actual NInfer prefill shape and quantization recipe; preplanned scratch, complete Op timing and oracle qualification |
| 8 | Bounded copy proposals and selected graph PDL | Controlled remesis exact-copy tests report4.82–5.35x on a pre-v3 C1 revision; these coexist with negative ordinary/editing cascade results. The copied fraction is not speedup; current p-less and grammar differ from legacy top-k qualification | Show missed profitable work versus existing DFlash; validate the full sampler/state law and complete-round benefit on both hit/miss workloads |
| 9 | Explicit idle keep-warm option | One fork shows lower post-idle incremental-prefill/TTFT with different weights/BF16KV and roughly71W extra power | Measure actual idle latency, power and interference; do not present as sustained decode speed |

The third pass also found a bounded immutable copy-source archive in remesis's nondefault branch. It can preserve proposal material across compaction without restoring omitted model context. Its C1/legacy top-k sampling and changed effective-seed behavior require explicit qualification; it is not current p-less/C1..4 evidence. Memra's workload-derived draft-vocabulary trimming is another conditional companion idea, while the broad optimized-head mechanism already exists locally.

### Feature opportunities

| Feature | Evidence and boundary |
|---|---|
| General JSON object/schema responses | `igorls/ninfer` has a concrete MTP-aware implementation; current tool-mask transactions are already richer in some respects. That fork disables DFlash2, so reuse endpoint/schema lessons without regressing current speculation. Its RTXPRO6000 speed numbers are not5090 evidence |
| Required/named tool choice on Responses | Current Responses rejects these choices; Chat Completions parses them separately. Audit forced-choice semantics per endpoint rather than claiming the entire server lacks tool selection |
| Restart-persistent Responses records | A fork provides durability; current records are process-local. Durable protocol history is distinct from resumable GPU state |
| Explicit branch/candidate scoring | `knoopx/ninfer` provides a decision-oriented reference. Candidate-normalized scores are not calibrated confidence over arbitrary language; define the intended API before adoption |

A further conditional feature is cross-request reuse of image embeddings after changed prefixes or retention misses (QW3 reference). Existing exact prefix hits may already avoid the work; measure that distinction first. Query-dependent historical KV retrieval (KVMem) is a separate context-memory product direction that changes attention semantics, not an exact faster attention replacement.

### Conditional artifact, quality and capacity research

DASC approximate retained-checkpoint omission and DAMP mixed-precision active recurrent state are additional research avenues, but change exact checkpoint restoration or our persistent FP32-state contract; neither is an ordinary private-intermediate precision choice. Exact-target QUASAR/Minima-style artifact quality, ARCQuant/SharQ arithmetic, alternative KV codecs including KVarN, and vision FP4 schedules remain useful research sources. They change represented formats, numerical profiles or quality/capacity tradeoffs and are not free engine accelerations. Compare full bytes, preparation, metadata and task/state quality. Fused head sampling is similarly constrained: plain categorical/Gumbel fusion does not implement NInfer's global-moment p-less law, and speculative verification needs probabilities.

## Findings that should prevent wasted work

- **Do not assume compressed recurrent checkpoints are harmless:** Poseidon reports severe 32K top-token disagreement and log-probability error from block-INT8 boundary state even with attention KV held fixed. DASC omission and DAMP active-state quantization need distinct quality contracts.
- **Do not assume a cache-hit discrepancy originates in cache restoration:** SGLang issue 41351 was localized to shape-dependent FP8 projection arithmetic before recurrence; changing checkpoint precision did not fix it.
- **Do not mistake a fast repetitive attractor for useful decoding:** one 1004.9 tok/s distributed headline accepted 64-token repeated proposals perfectly; the reporting article itself rejects it as a meaningful decode comparison.
- **Do not add generic adaptive drafting:** NInfer already optimizes expected accepted output over measured round time. A useful change must address a demonstrated decision failure, such as an unprofitable phase, rather than rename the existing objective.
- **Do not copy lazy GDN state as a new feature:** Radiance's input stash/replay is substantially present in NInfer's ReplaySSM. GDN state, KV, hidden state and draft state must remain one valid continuation.
- **Do not propose once-per-tile V conversion as missing:** independent source review found it already in NInfer's dense NVFP4 prefill. FlashInfer's similar patch is useful corroboration, not a new optimization here.
- **Do not infer a target-head guarantee from exact reranking:** an approximate shortlist can omit the true target argmax or probability mass. Radiance's target-head shortcut is not established as exact. Draft-only approximation has a different contract.
- **Do not equate NVFP4 storage with identical arithmetic:** W4A16, W4A4, mixed FP8, temporary dequantization and FP32 recurrent state materially change both cost and quality.
- **Do not port a generic megakernel runtime to remove a small host seam:** local seam and persistent SwiGLU/down experiments already have negative evidence. Mirage's inspected dense-linear dispatch also excludes CC120.
- **Do not infer 5090 compatibility from B200/SM100 results:** actual instruction sets, shared-memory limits, grouped-kernel support and model geometry decide compatibility.
- **Do not treat higher maximum context or total KV pool capacity as throughput at that context.** Short-prompt C4 output and a 262K admission limit are different measurements.
- **Do not import weight offload, CPU experts, tensor parallelism or preemptive large-scale serving.** Those solve different contracts. The Qwen4 diagnostic streaming exception is narrowly specified and does not change native Engine ownership.
- **Do not count wrapper projects as independent kernels.** Ollama/LM Studio-style products and tuned vLLM containers can offer useful deployment/features while sharing an underlying execution engine.

## How to compare a surviving candidate

Freeze the exact artifact and companion, tokenizer/template, actual represented arithmetic, NVFP4 KV, prompt IDs/media, sampling and stop behavior. Record whether any candidate intentionally changes the artifact or numerical profile. Use C1 and C4 plus any intermediate shape whose implementation changes; include short and long prompts and an actual tool/editing workload rather than only repetitive generation.

For each case report TTFT, complete-round time, accepted tokens per round, per-request decode rate, aggregate complete-wave throughput, memory and relevant quality/state checks. Distinguish warm startup, post-idle latency, prefix hits, cold/rewrite prefill and sustained decode. Preserve the real output/acceptance differences instead of attributing every token-rate change to a faster kernel. Validate ordinary and speculative grammar/state commits when those paths change.

For mathematical changes use the independent represented-input oracle, not another optimized engine as the oracle. A route that passes a random tensor tolerance but changes real recurrent verification needs further numerical investigation. For a performance claim, the complete public Op and Engine phase must improve at the scope being claimed. NInfer's existing classifier and prior negative results determine whether a candidate proceeds at all.

## Limits

This is broad discovery and source inspection as of the cutoff, not a census of every public/private engine or every fork revision. Search indexing and repository accessibility limit coverage. The469-row inventory is discovery coverage, not469 complete code audits. Current primary code, negative results and benchmark harnesses were inspected for consequential candidates; inaccessible evidence remains named in the depth reports. No external workload was reproduced and no current NInfer speedup is claimed. The remaining uncertainties are stated next to each candidate so a future implementation decision can choose one bounded experiment rather than repeat the whole survey.
