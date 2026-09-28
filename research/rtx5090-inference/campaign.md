# RTX 5090 inference research campaign

Research cutoff: 2026-09-26. Local baseline inspected: `176e04cc` (2026-09-19).

## Deliverable and scope

Produce an evidence-backed engine and technique survey, source ledger, and ranked recommendations for NInfer. This is research, not authorization to implement candidates, change the product contract, download checkpoints, or claim measured NInfer gains. Repository clones for inspection live outside the checkout under `/tmp/ninfer-research`.

Cover upstream engines, relevant forks, smaller specialized engines, kernel libraries, research papers, author articles, and community reports. Follow concrete new leads until the major engine families and plausible high-value techniques have dispositions. “Complete” means this bounded investigation is complete as of the cutoff, not that every Internet page or future engine has been examined. Search failures and inaccessible sources remain explicit limitations.

## Evidence rules

- Primary code, author reports and papers establish what exists; Reddit and articles supply discovery leads and workload anecdotes.
- Separate code inspection, author-measured results, independently reproduced results, and unverified claims. No external GPU benchmark is reproduced by this research campaign.
- Match hardware, model/artifact, activation precision, KV format, context, concurrency, speculative acceptance, sampling, phase, and throughput denominator before comparing speed.
- Distinguish an existing NInfer capability, an incremental improvement, a new feature, an experiment requiring qualification, and a contract-incompatible direction.
- Record original URLs in local research files. Avoid treating mirrors or search snippets as final technical authority when original sources are available.

## Research ledger

| Avenue | Owner | Status |
|---|---|---|
| Current NInfer capabilities, measurements and rejected experiments | Orchestrator | Complete; baseline.md |
| vLLM, Radiance, SGLang, TensorRT-LLM and deployment overlays | vllm_engines | Complete; vllm-engines.md |
| NInfer forks, llama.cpp family, ExLlama, imp, FlashRT and local engines | local_engines | Complete; local-engines.md and fork-inventory.tsv |
| NVFP4 kernels, attention/KV and speculative decoding | kernels_speculation | Complete; kernels-speculation.md |
| Compiler-generated kernels and persistent runtimes | vllm_engines follow-up | Complete; compiler-runtime.md |
| Quantization and quality preservation | kernels_speculation follow-up | Complete; quantization-quality.md |
| Community leads and useful serving features | local_engines follow-up | Complete; community-features.md |
| Additional engine-family and technique gap sweeps | vllm_engines / kernels_speculation | Complete; landscape-coverage.md and kernel report |
| Cross-review, source ledger, ranking and research closure | Orchestrator and independent reviewers | Complete; README.md, review-notes.md and sources.tsv |

## Local baseline cautions

The product target is Qwen3.8-27B NVFP4, one RTX 5090, startup-fixed C=1..4. README Qwen3.6/C=8 results are historical measurements, not a current matched baseline. The exact Qwen4 preview exception remains capacity/quality gated; ordinary-weight offload is not a production option.

Already delivered: CUDA Graphs, compact batched decode, chunked prefill, NVFP4 KV, MTP and DFlash2, optimized draft head, adaptive k=3/4/5 selection, recurrent-state-aware prefix checkpoints, Qwen3.8 host/SSD prefix tiers, tool-schema grammar enforcement including speculative positions, OpenAI Responses and Anthropic serving. General schema-constrained response output and restart durability need separate inspection; neither follows automatically from tool grammar or process-local continuation.

Relevant local evidence is in `docs/performance.md`, `docs/cli.md`, `docs/serving.md`, `docs/maintainer/qwen3.6-27b-model.md`, and `docs/maintainer/performance_enhancements.md`. Recent projection traces identify memory-bound weight traffic as the dominant decode cost. SmallT/GDN weight replay and batching have already been extensively optimized. Host seam hiding, device graph tail launch, several graph-routing changes, and several tree/tile changes have negative local evidence; a foreign engine speedup does not reopen them without materially new attribution.

## Verification record

This campaign changes research documentation only. The required `./scripts/run-unit-tests.sh` command was attempted on 2026-09-26, but its admission check refused to start: 15,359 MiB GPU memory free versus 20 GiB required. An existing `ninfer_qwen3_6_27b_cache_interleavings_real_test` in another builder container owned approximately 16.4 GiB. That process was left running. No C++ test result is claimed for this campaign. Source review, internal-reference checks and whitespace verification are the directly relevant checks for the research deliverable.

## First-pass limits and reopened investigation

The first pass produced a broad survey, but its stopping decision was premature. A user-requested depth audit immediately found omitted direct engines and relevant kernel changes. The first-pass claims of coverage closure are superseded by the depth ledger below. Proposed implementation experiments remain recommendations; completing this additional source investigation is assigned work.

The first-pass inventory contained214 names. The reopened enumeration recovered282 unique direct forks and469 accessible network repositories by following descendants; metadata-only entries remain explicitly distinguished from inspected code. This resolves the original pagination gap without claiming every repository was code-audited or every private fork was discoverable. Private repositories, unindexed work, inaccessible raw benchmark material and future releases cannot be exhaustively covered. Source inspection established no matched performance win over this checkout. No weights were downloaded, no candidate engine was installed, and no inference implementation or machine power setting was changed.

Documentation navigation, source/fork TSV structure and whitespace were checked. The research is indexed from `docs/README.md`; original addresses and evidence levels are preserved in the reports and source ledger. No commit was created.

## Depth audit ledger

| Avenue | Owner | Status |
|---|---|---|
| Fork enumeration, changed-code depth, omitted native engines | local_engines | Complete; fork-depth-audit.md |
| Competitive raw measurements, SparkInfer, upstream PRs and Radiance variants | vllm_engines | Complete; competitive-depth-audit.md |
| GDN prefill, kernel compilers, fused sampling and omitted mechanisms | kernels_speculation | Complete; technique-depth-audit.md |
| Independent search and evidence/priority correction | Orchestrator | Complete; coverage-depth-audit.md and revised README.md |
| Final source/reference checks and full chat report | Orchestrator | Complete; verified207-source ledger,469-fork inventory and17 research files; substantive chat report delivered with handoff |

## Revised closure evidence

The depth pass found and incorporated actual omissions instead of affirming the first pass. It added four detailed reports, expanded the fork network inventory, inspected native engine code and benchmark harnesses, checked current PR predicates, and recorded negative/stale evidence. Independent cross-review verified grammar serialization, SparkInfer TMA/tile details, and the distinction between FlashInfer split-KV gate fixes and NInfer's existing split execution. Final synthesis ranks phase-specific absent mechanisms over generic copy/PDL claims. No newly found material source lead remains without a disposition; measurement-dependent hypotheses are labeled future experiments rather than asserted gains.

Final report checks: all indexed report files exist; source TSV has207 unique addresses and consistent five-column rows; fork TSV has469 unique entries and consistent four-column rows; tracked and untracked documentation whitespace checks passed. Final review also separated FlashQLA auto-selected published timings from the proposed non-CP experiment, since the receipt does not establish that mode. The C++ suite admission limitation above remains; no implementation changed.

## Third-pass omission audit

The user requested a stronger completeness audit. New independent searches again found missed native engines and current backend changes, so previous discovery-closure language is superseded. This is a finite source investigation with explicit observability limits, not a100%Internet census. The orchestrator records phase/channel coverage and Lumen/QW3 in completeness-audit.md; delegated reports cover source channels/nondefault forks, backend ecosystem updates and techniques. Final verification and source-ledger regeneration follow after report cross-review.

| Avenue | Owner | Status |
|---|---|---|
| Native engines and branch/community omissions | local_engines | Complete; source-channel-audit.md |
| Backend ecosystem and release/PR omissions | vllm_engines | Complete; ecosystem-omission-audit.md |
| Papers, recurrence/speculation and toolchain omissions | kernels_speculation | Complete; technique-omission-audit.md |
| Independent engine screens and coverage matrix | Orchestrator | Complete; completeness-audit.md |

Third-pass cross-review confirmed the limited image-feature-cache opportunity, Lumen arithmetic/hardware qualifiers, KVMem's changed computation, and the MetaZenith27B fusion boundary versus Qwen4-only local normalization fusion. The latter still stores/reloads rawQ/K and uses different cast boundaries; no eliminated-all-traffic claim remains. Every promising lead discovered in this pass has a recorded disposition. Unknown sources and uninspected branch histories remain explicit limits, not silently declared complete. No runtime edits, GPU jobs, dependencies or model downloads were performed.

Third-pass verification:22 research files;280 deduplicated source addresses;469 accessible fork records;192 branch-head records across11 selected repositories. All indexed files exist; TSV row structure and tracked/untracked whitespace checks passed. The previous GPU unit-suite admission limitation remains unchanged because this pass edits documentation only. No commit created.

## Final sustained discovery and evidence pass

The user's further completeness request reopened discovery. Topic enumeration, author/lineage searches, multilingual reports and issue/citation chains found additional engines and consequential negative evidence. Earlier claims that no promising source remained apply only to the then-known queue; they cannot establish Internet-wide exhaustion. This pass explicitly distinguishes engine discovery, backend derivation, code inspection and performance reproduction.

| Avenue | Owner | Status |
|---|---|---|
| Repository topics, aliases, author lineages, independent engines and multilingual community trails | local_engines | Complete; final-engine-discovery.md and engine-discovery.tsv |
| Current backend issue/PR chains, model cards, recipe quality and raw benchmark provenance | vllm_engines | Complete; final-backend-research.md |
| Kernel-library identities, paper citation trails, recurrent-state and speculation mechanisms | kernels_speculation | Complete; final-technique-research.md |
| BlackweLLM, TensorSharp, article claims and independent synthesis | Orchestrator | Complete; final-independent-research.md |
| Cross-review, current recommendations, source inventory and document verification | All / Orchestrator | Complete; README.md, review-notes.md and all four inventories |

Known remaining uncertainties are not silently converted into positive results. Source-only projects, unpublished kernels, inaccessible originals and unmatched hardware remain labeled; no external benchmark is being reproduced. The final deliverable closes the tracked material leads and reports the access limits; it does not certify every Internet source.


## Final campaign closure and verification

All three delegated source investigations and independent cross-reviews are complete. This pass added four detailed reports and a canonical engine/backend inventory. The source ledger increased from 280 to **420 distinct addresses**; the consolidated inventory contains **126 repository identities**, alongside the existing **469 fork-network records** and **192 selected branch heads**. The complete research deliverable has **27 files**, all indexed from its main README. These are overlapping discovery/evidence denominators, not counts of independent engines or reproduced benchmarks.

Independent review corrected Qwarz's actual mixed head/MTP precision and calibration description, Knivesysl's plain-baseline denominator and inherited lineage, Poseidon's reported restore metrics, and a duplicate b12x/SparkInfer identity. It verified b12x's convergence predicate, DDTree's actual difference from local tree selection, Kachua's finite algebra versus floating-point profile, and the scoped SGLang causal diagnosis. The remaining HN result page was screened; a source-empty repository and inaccessible original/mirror provenance received explicit dispositions. No identified material source lead remains unclassified or awaiting an assigned investigation.

The strongest recommendations now distinguish grammar overlap, several qualified GDN prefill algorithm comparisons, selective TMA cache policies, earlier draft preparation and tree-budget allocation. Existing local optimizations, failed foreign experiments, contract-changing techniques, source-limited claims and other hardware are retained as evidence rather than promoted into new speed promises.

Verification passed: all 26 indexed companion artifacts exist; all TSV rows have their declared widths and unique keys; source and engine report references resolve; the canonical engine inventory has no duplicate alias identity; tracked and untracked documentation whitespace checks pass. No runtime implementation, dependency, model artifact, GPU setting or commit changed. External benchmarks were not reproduced. The earlier C++ suite admission failure remains the recorded limitation; this documentation-only expansion did not rerun GPU work.

This completes the finite research deliverable as of 2026-09-26. Private/unindexed projects, inaccessible sources, unpublished kernels or traces, uninspected histories and future work prevent any honest claim of 100% Internet coverage. Remaining candidate-specific numerical/performance questions are explicitly defined future experiments, not unfinished source research or established NInfer gains.
