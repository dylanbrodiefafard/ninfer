# NInfer baseline for interpreting external claims

## Corrected experimental baseline

The current checkout is `experimental` at `d29841e0`; see `experimental-review.md` for the source comparison and candidate dispositions. It remains Qwen3.8-27B NVFP4 on one RTX 5090. Important differences from the original inspection below: A8 NVFP4 verification versus A4 prefill; startup-fixed maximum concurrency 6; adaptive DFlash arms 1–5 with measured-arm selection; family-owned frontend/mask exchange with fixed-pitch graph-compatible staging; host rewrite checkpoints and resident-prefix recovery. Dynamic elastic lanes are not implemented.

The external research and the three leading measurement candidates remain applicable. Several generic decode/fusion suggestions are already implemented, and generic A8 PDL has a local negative result. Current artifact inventory and actual backend must be pinned before remeasurement. No experimental GPU benchmark was performed for this review. Earlier queued results and binaries belong to Qwen4 and must not be used as current Engine evidence.

## Original inspection (historical)

Inspected 2026-09-26 at local commit `176e04cc` on **qwen4**, while targeting its Qwen3.8-27B NVFP4 implementation. The remaining sections preserve that original comparison. These are existing repository measurements and contracts, not current experimental benchmark runs.

## Supported workload

`AGENTS.md` defines one RTX 5090 (`sm_120a`), one resident Qwen3.8-27B NVFP4 model, startup-fixed C=1..4, compact decode batches, bounded FIFO ingress, and no preemption. Ordinary compute and ordinary weights stay GPU-resident. The audited Qwen4 preview exception does not make the oversized preview runnable, permit arbitrary future checkpoints, or authorize general weight offload. Qwen4 host/disk KV tiers fail admission; the Qwen3.8 prefix tiers are separately supported.

## What is already implemented

| Capability | Local authority | Implication for research |
|---|---|---|
| CUDA Graph decode, exact compact active batch, chunked prefill | `docs/maintainer/concurrent-inference-architecture.md` | Recommending these generically adds nothing. |
| NVFP4 KV, BF16 and INT8 alternatives | `docs/cli.md` | Compare KV methods against NVFP4, not only FP16. |
| MTP, DFlash2 NVFP4 draft, optimized proposal head | `docs/cli.md`, `docs/maintainer/qwen3.6-27b-model.md` | Additional draft schemes must improve accepted tokens per complete round. |
| Adaptive draft controller | `docs/cli.md:140`, `include/ninfer/types.h:116` | Already estimates `E[Y(k)] / T(k,C,L)` over k=3/4/5 with sticky switching; fixed k=4 remains fixed. |
| ReplaySSM candidate records and accepted-prefix fold | `docs/maintainer/qwen3.6-27b-model.md:247`, `docs/maintainer/replayssm-gdn.md` | Lazy speculative state is not automatically a new memory optimization. |
| Dense and approximate attention routes, XAttention | `docs/performance.md:352`, model reference | Any alternative needs quality and phase-specific comparison against the selected route. |
| Prefix reuse with GDN/hidden/draft state, ladder and turn checkpoints | `docs/serving.md:853`, concurrent architecture §6.4 | A token/KV prefix alone cannot restore a hybrid recurrent model. |
| RAM and SSD retained-prefix tiers for Qwen3.8 | `docs/cli.md:288`, `docs/serving.md:792` | Offload claims may describe an existing capacity feature; not ordinary-weight streaming. |
| Tool-envelope and argument-schema constraints during ordinary/speculative sampling | `docs/serving.md:340` | New grammar recommendations must identify a missing semantic feature or measured cost. |
| Responses Core, Chat Completions, Anthropic Messages, vision/video | `docs/serving.md`, README | API wrappers and media support are largely existing capabilities. |

Known feature distinctions: Responses `text.format` accepts text only (`docs/serving.md:281`); stored continuation is an in-process LRU lost at restart (`:534`); Responses strict tools and required/named tool choice are rejected (`:357`). Chat Completions separately parses required/named choices, so this is not a blanket all-endpoint limitation. These are possible explicit product extensions, not defects inferred from another server's feature list.

## Performance measurements must remain separate

| Existing campaign | Configuration and result | Interpretation |
|---|---|---|
| 2026-09-01 host campaign | DFlash4, NVFP4 KV, C1..4: 209.97 / 328.37 / 391.60 / 467.91 external decode tok/s | Workload-specific initial rates; cumulative host changes were essentially neutral. |
| 2026-09-05 kernel sweep | Greedy AIME, 2,048 generated tokens/lane, DFlash4 C1..4: 163.63 / 245.37 / 265.92 / 260.04 aggregate complete-wave tok/s | Different prompt, acceptance and denominator from many public “600+” claims. |
| Later GDN paired replay | Fixed 1,024-token waves, DFlash4 C4: 251.962→261.382 tok/s | +3.74% matched improvement, not a universal current rate. |
| 2026-09-06 W4/W6 grouping | DFlash3 C4 151.617→178.667; MTP3 C4 175.968→214.136 tok/s | Short 256/128-token waves; do not mix with 2,048-token wave results. |
| 2026-09-11 selective FP8 | 328 MiB additional payload; C2 DFlash5 381.33→373.63; target-only C1 84.83→83.15 tok/s | Quality/memory tradeoff costs about 2–3% on this corpus, not acceleration. Draft acceptance was 97%, unlike difficult chat. |

Source: named sections of `docs/performance.md`. Older README Qwen3.6, MoE and C8 figures answer different questions.

## Attribution and negative results

Recent C2/C4 node traces attribute about 43–46% of kernel time to NVFP4 SmallT and 26–27% to fused NVFP4 SwiGLU; GQA was below 1% in those short-context traces. A separate C1 trace attributes 28.8% to SwiGLU, 17.0% to GDN input and 14.9% to MLP down. These are distinct traces, not additive rows of one profile. Exact-shape classifier results identify the dominant projections as DRAM-bound (`docs/performance.md:528–794`).

Already investigated without retained speed benefit:

- Hiding the host round seam: 1.000–1.001×; device graph tail launch: 0.983–0.987× in the historical INT8-KV experiment. Reopen only with new material host-gap attribution.
- Graph-selection lookup and tighter context profiles: neutral or regressive in the later NVFP4 campaign.
- Persistent SwiGLU/down fusion: classifier rejected removal of only about 170–220 KiB activation traffic while streaming about 151 MiB weights per W5 layer.
- Tile/warp retuning of DRAM-bound projections, SwiGLU scale staging and a larger GQA tile: rejected or regressive at tested shapes.
- Packed DFlash trees: several attempts are recorded in `docs/maintainer/dflash2-tree-speed.md`; current production is chain verification. New tree evidence must change the cost/acceptance argument.

These findings do not prove every alternative impossible. They constrain which new mechanism would justify another experiment.

## Evidence needed for a proposed improvement

Use the existing `tools.kdev` recipe/bound/mma classifier before a kernel experiment. Qualify mathematical output from represented public inputs against the independent oracle, including actual weights where private precision is material. For speculative/recurrent changes, inspect state and acceptance transitions rather than relying on plausible final text. Measure Engine impact at C1 and C4 (and any changed intermediate shape), default NVFP4 KV, representative context and actual chat/tool template. Report TTFT, accepted output tokens per complete round, per-request latency and aggregate throughput with their denominators. External author benchmarks are leads until this matched comparison exists.
