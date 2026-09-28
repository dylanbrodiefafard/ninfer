# Applicability to experimental: Qwen3.8-27B NVFP4

Reviewed 2026-09-26 local time. Original local comparison: `qwen4` at `176e04cc`. Corrected implementation: `experimental` at `d29841e0`. This is a source and architecture review, not a fresh external-source search or GPU benchmark. The cancelled Qwen4 queue remains cancelled.

## Decision

**Keep the external research; replace the local implementation baseline and reassess priorities. A fresh broad research campaign is not justified by the branch correction.** The campaign explicitly targeted Qwen3.8-27B NVFP4 on RTX 5090 even though the checkout also contained Qwen4 work. Its engine inventories, primary-source investigations, SM120 exclusions, artifact/quality caveats and algorithm descriptions do not become invalid because of the checkout branch. The mistake affects claims about what NInfer already implements, exact code ownership, the selected arithmetic and performance attribution.

The user confirmed Qwen3.8-27B NVFP4. That identifies the model/format, not whether a run enables MTP or DFlash2, which companion artifact it loads, or whether it uses optional selective FP8 tensors. Those choices must be fixed before performance comparisons. The results below about DFlash verification apply when that backend is enabled.

## Material architectural differences

| Area | Current experimental evidence | Consequence |
|---|---|---|
| Verification precision | `src/targets/qwen3_6_27b/impl/variant.cpp:48–62`: text/prefill selects AllowA4; every NVFP4 verification width selects AllowA8. `docs/maintainer/dflash-a8-followups.md` records retained pipeline, SwapAB, panel-count, gating and chain GDN work | NVFP4 storage does not imply the old activation arithmetic. Old A4/SmallT decode attribution is not a current baseline. Separate A4 prefill from A8 verification |
| Frontend and grammar ownership | Frontend and exchange live under `src/targets/qwen3_6/impl/`, rather than Qwen4's shared `src/text/qwen` and runtime-contract exchange | Preserve the current family ownership. An overlap implementation should not port the Qwen4 ownership refactor |
| Grammar transfer/capture | `impl/runtime/tool_masks.cpp:74–87` stages candidate IDs with device-to-device 2D copy then fixed 1D device-to-host copy; header documents the driver's host-2D graph-update limitation | Old enqueue instrumentation/patches cannot be applied mechanically. The staging is deliberate, not redundant copy work to remove |
| Concurrency | `include/ninfer/types.h:22` exposes maximum 6; current architecture describes startup-fixed C1–6 | C1/C4 remains useful, but cannot qualify the complete experimental range. Include C6 and affected intermediate widths when implementing; this review reports repository behavior rather than silently changing the earlier C1–4 task scope |
| Adaptive DFlash | `impl/runtime/adaptive_draft.h:69–78,273–310`: captures k1–5 when adaptive DFlash is enabled with maximum 5; explicitly measures captured arms before selecting expected output/time | Shorter arms and direct measurements of discontinuous route costs are already present. Old k3–5-only descriptions and optimistic extrapolation assumptions are stale. MTP has its own arm rule |
| Rewrite retention | Current-state pool is C slots plus one engine staging slot; rewrite GDN/DFlash images are pinned host state. See concurrent architecture §4.3 and `elastic-concurrency.md` | Old device-slot counts, memory budgets and checkpoint-copy costs do not transfer. This saves device capacity but adds checkpoint transfers; it does not mean dynamic elastic lanes are implemented |
| Recovery | `concurrent-inference-architecture.md` recovery section and current executor preserve the admitted lane and reuse a ready prefix checkpoint before suffix prefill | Old always-cold retry assumptions are wrong. Grammar/cancellation changes must preserve the newer recovery and publication boundaries |

The common DeviceContext source is unchanged between the inspected branches. Adding a grammar stream would still require explicit ownership, graph fork/join, complete synchronization and teardown. Existing KV copy-stream ownership is not permission to reuse it for callbacks.

The optional selective-FP8 recipe is also already implemented; the documented 328 MiB recipe promotes eight matrices, including MLP down in layers 62/63 (`docs/maintainer/qwen3.8-27b-artifact.md:99–159`). Those tensors bypass NVFP4 TMA. The model label alone therefore cannot identify every executed projection route. Current A8 quality evidence is recorded in `docs/performance.md`; old exact-response expectations cannot be carried across an intentional arithmetic-profile change.

## Disposition of the research priorities

| Original lead | Disposition on experimental | Next useful evidence |
|---|---|---|
| Grammar-mask/verify overlap | **Still valid; implementation plan needs revision.** `speculative_target_impl.h:24–40` enqueues masks before target forward; `text_context_impl.h:759–762` consumes them after the head. The independent work window remains. Program's exchange-before-request declaration order still requires careful error/teardown draining | Measure actual completed tool calls with the selected companion; preserve fixed-pitch staging, graph replay and recovery. Join immediately before sampling, not before forward |
| GDN prefill fusion/alternative schedules | **Still valid.** Grouped 27B Hq16/Hv48,D128 remains chunk64 with BF16-to-FP16 normalization, preparation, state passing, output and recurrent tail. The state-passing kernel and independent FP64 oracle are unchanged. Removed Qwen4 Hq=Hv48 normalization fusions were not a 27B optimization | Rebuild and measure the current public Op. Attribute the production running profile before fusion. Naive fusion's repeated QK computation/parallelism loss remains a real concern; retain FP32 persistent state qualification |
| Selective TMA cache hints | **Still valid for A4 prefill, not the new A8 verify path.** Both TMA implementation files are unchanged; LinearAdd's renamed `nvfp4_linear_add_quantized.cu` retains the T>=1024, divisible-by-256 dispatch | Fresh same-shape Linear and LinearAdd T1024/T4096 baseline and oracle. Do not turn a prefill result into a decode claim |
| Earlier per-feature-tap draft projection | **Still conditional.** Current path still assembles target taps before projection | Measure projection share after the retained drafter changes. Account for concurrent resource contention and reduction/state semantics |
| Better DFlash tree-budget allocation | **Algorithm lead survives, low-priority experimental work.** Fixed-frontier builder remains; production still uses chain verification and has documented losing tree trials | Show a new acceptance/complete-round cost argument against current A8 chain verification. Existing tree code is not evidence that the deployed backend uses trees |
| Companion alignment | **Still conditional; fresh acceptance baseline required.** Target verification arithmetic and draft projection implementations changed | Measure actual task/template/phase acceptance before training, conversion or artifact substitution. Do not infer a mismatch from different engines' headline rates |
| In-kernel quantization/persistent GEMM | **Narrow and reassess.** Several fusion boundaries already changed; current follow-up plan has bounded residual quantization sites and prior negatives | Identify remaining standalone cost on the actual route. Keep existing numerical boundaries and avoid repeating retained fusions |
| Copy proposals / graph PDL | **Split the recommendation.** Copy proposals remain workload-conditional; experimental has direct negative A8 PDL results | Do not rerun generic PDL without new attribution that changes the argument. Compare copy hit and miss workloads against current drafting |
| Idle keep-warm | **Still conditional and branch-independent in principle** | Only investigate if post-idle latency is a real requirement; separate power/idle behavior from sustained decode |

The top three remain reasonable **measurement candidates**, not a proven speed ranking. Prefill-heavy and tool-heavy workloads may justify different orderings. No new speedup is claimed by this review.

For a changed grouped GDN maximum-width route, add direct Hq16/Hv48 T4096 qualification against the existing FP64 oracle. Experimental retains the grouped T3404, tail and partition checks but removes Qwen4-specific maximum-width tests. Test coverage from the old checkout must not be assumed present. The existing `--chunked-only --breakdown` remains a BF16-private attribution profile, not the production FP16 `--running` profile.

## Already addressed or no longer safe to reuse

- Generic adaptive drafting, adding DFlash k1/k2, or assuming shorter k bounds the time of longer k: already addressed by the current controller.
- Generic small-token SwapAB/panel-count and GDN gating parallelization proposals: current A8 follow-up evidence records retained implementations. External sources can still suggest a genuinely different shape/policy, not novelty by name alone.
- A8 prefetch PDL: the current branch records a loss, not an untested opportunity.
- Old C1–4-only capacity assertions, rewrite device-slot math, and cold-recovery costs: replace with current ownership and route evidence.
- Qwen4 PLE, preview-artifact streaming exceptions and Qwen4-specific frontend ownership: not reasons to modify the Qwen3.8 engine.
- The old queued binaries and partial results: Qwen4 provenance only. Even an unchanged kernel does not qualify the experimental Engine or its current dispatch. Do not resume that queue.

## Feature recommendations

The principal feature gaps survive this review. `src/serve/responses_schema.cpp:515–516,697–702` still rejects required Responses tool choice and non-text response formats; adjacent parsing rejects named selection. `docs/serving.md:548` still describes restart-lost process-local records. General JSON/schema responses, required/named Responses tool selection and optional record durability therefore remain possible explicit feature projects. Chat Completions must be evaluated separately; do not claim every endpoint lacks tool selection. Branch/candidate scoring and image-embedding reuse remain product/workload questions, not automatic performance wins.

## What to do next

1. Keep the existing external inventory and source reports. Read this review before their local-baseline claims; `baseline.md` preserves the original comparison for provenance.
2. Rebuild from experimental and record the actual artifact, backend, activation policies, KV format and concurrency. The previous GPU queue is cancelled and must stay disabled.
3. Obtain new grammar, GDN and A4-prefill baselines, then admit only candidates with a measured opportunity. Use current independent oracles and current cache/recovery integration boundaries.
4. Consult external primary sources again only for a specific implementation question or a genuinely new post-cutoff technique. A broad rediscovery of vLLM, Radiance, forks, Reddit and engine inventories would mostly duplicate work.

Verification: compared source and current architecture across both commits, independently reviewed runtime/grammar, GDN/speculation and linear/TMA paths. No GPU work was launched, no runtime code changed, and no external benchmark was reproduced for this review.

## Measured TMA follow-up (2026-09-27)

The selective B/SFB evict-first prototype was admitted, implemented, qualified
against both unchanged public Op oracles, and discarded after twelve accepted
atomic baseline/candidate pairs. At T4096, median Linear latency rose 2.27% and
LinearAdd rose 4.50%; all three pairs for each Op regressed. T1024 showed no
repeatable production benefit. Baseline source remains selected; this changes the
particular cache-hint lead from an untested candidate to a measured negative.
Other tile/cache policies and A8 verification were not tested by this experiment.
See [the stable performance result](../../docs/performance.md#selective-tma-weight-cache-hints-rejected-2026-09-27)
for method, numerical qualification and evidence boundaries. The earlier source
review above describes the pre-experiment admission decision.
