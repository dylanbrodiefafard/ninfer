# Compiler and persistent-runtime screening

Research date: 2026-09-26. Primary repositories were shallow-cloned and code-read under `/tmp/ninfer-research`; no external build or GPU benchmark was run. This report completes the Emmy/Mirage research avenue and identifies conditional experiments, not an implementation assignment.

## Outcome

Emmy supplies credible **5090 prefill** evidence and inspectable precision/transport techniques, but no demonstrated NVFP4 Qwen3.8 speedup. Mirage's headline persistent-kernel speedup does not cover the current 5090 model contract; its inspected dense-linear dispatch explicitly rejects compute capability 120. Neither justifies replacing NInfer's runtime with a generic compiler or megakernel.

## Emmy: useful kernel techniques, different precision/model

Primary sources:

- CR01: https://github.com/cloudrift-ai/emmy — inspected implementation, current main.
- CR02: https://www.cloudrift.ai/blog/optimizing-gemma-4-12b-rtx — author article, dated 2026-08-01.
- CR03: https://github.com/cloudrift-ai/emmy/blob/main/recipes/gemma-4-12B-it/RESULTS.md — author measurement record, retrieved cutoff.
- CR04: https://github.com/cloudrift-ai/emmy/blob/main/emmy/compiler/ir/kernel/render.py — inspected CUDA emitter.
- CR05: https://github.com/cloudrift-ai/emmy/blob/main/emmy/compiler/pipeline/passes/lowering/kernel/_atom.py — inspected promotion cadence.

All retrieved 2026-09-26. CR01/04/05 are **code evidence**; CR02/03 are **primary reported measurements**, not independently reproduced.

The August article reports 1.30× geometric-mean speedup over eager across deployed Gemma4 FP16 kernel shapes, and lower TTFT in its vLLM integration. It explicitly says decode is slightly slower because memory-bound kernels gain little and plugin overhead remains. Its TMA transport removes per-thread address work; chunked FP16 tensor-core accumulation periodically promotes partials into FP32. These are separate mechanisms: a transport improvement and a precision/performance tradeoff. The article also says broad inter-operator fusion is not demonstrated there. Do not reinterpret it as a megakernel result or a NVFP4 decode win.

CR03 pins one RTX 5090, CUDA 13.0, vLLM 0.23.0, FP16 weights/KV, 16K context. Selected C1 4096/4096 serving is 54.5 output tok/s, TTFT 471 ms, TPOT 18.2 ms; the deployment uses different decode/prefill buckets across the full grid. Reported GSM8K is a 200-question check, not proof of equivalent quality on Qwen's recurrent trajectory. The current golden inventory was re-recorded after the publication inventory, so one cannot identify every published timing with today's generator output.

Concrete source anchors: `render.py` emits `emmy_mma_m16n8k16_f16_f16` and `emmy_mma_promote_f16acc`; `_atom.py` sets `_F16ACC_STEPS=4`, corresponding to K64 for k16 atoms in its global-memory-direct path. The operation accumulates short FP16 chains, converts partials, accumulates the shadow in FP32 and resets the partials. This is not exact FP32 accumulation and overflow/range behavior depends on represented operands. A BF16-input Op cannot simply switch to FP16 without qualifying that conversion against the independent oracle.

NInfer disposition: TMA and exact-shape specialization are **already present** in its NVFP4 candidates. Periodic promotion is a **conditional idea** for a measured compute-bound FP16-compatible Vision/other subproblem, not a high-priority dense-27B decode replacement. The accepted implementation must remain a direct `src/ops` route with explicit ownership, qualified at actual shapes. Importing Emmy's generic model graph, runtime search or plugin structure is **not recommended**. Current NInfer decode bottlenecks predominantly stream target weights; an FP16 arithmetic trick does not remove those bytes.

## Mirage Persistent Kernel: inspect the architecture gate

Primary sources:

- CR06: https://github.com/mirage-project/mirage — README's MPK claim, current code, retrieved 2026-09-26.
- CR07: https://github.com/mirage-project/mirage/blob/main/python/mirage/mpk/persistent_kernel.py — inspected build/linear dispatch.
- CR08: https://github.com/mirage-project/mirage/tree/main/include/mirage/persistent_kernel/tasks/blackwell — inspected SM100 task inventory.

CR06 is a **project claim**; CR07/08 are **code evidence**. The README advertises one persistent kernel combining compute and communication, with a broad 1.2–6.7× latency range. That number is not a single-5090 Qwen3.8 measurement.

The inspected build logic explicitly enables architecture macros for CC90 and CC100; other devices receive `-arch=native`, which alone does not add working task implementations. More decisively, `PersistentKernel.linear_layer` selects SM100 for `100 <= cc < 120`, Hopper for `90 <= cc < 100`, older linear for `80 <= cc < 90`, then asserts unsupported. The Blackwell task inventory contains `linear_fp8_sm100`, `mla_*_sm100` and related SM100 implementations. This is stronger evidence than guessing compatibility from the term Blackwell. A native compile flag is not an SM120 port.

Transferable concepts: task scheduling on resident blocks, finer producer/consumer overlap, and avoiding materialization **when a closed subgraph really has significant intermediate traffic**. A persistent schedule can also reduce repeated launch/scheduling work. These benefits have different denominators and must be evaluated separately. Holding many heterogeneous tasks resident can increase registers/shared memory, constrain occupancy and introduce global dependency cost.

NInfer disposition: **no direct compatible runtime path established**; do not port the general MPK runtime as a speed project. Its ordinary inference scheduler is deliberately compact and owned by Engine. If a specific high-traffic closed Op subgraph is identified, its mathematical computation can be fused under existing Op ownership without importing a graph framework. No new materializing subgraph was found here that overrides the local classifier's rejected SwiGLU/down fusion.

## Local negative results constrain both avenues

CR09: local `docs/maintainer/performance_enhancements.md`, read 2026-09-26 (**local measured authority**). C1 graph-round overlap gave 1.000–1.001× and device graph tail launch 0.983–0.987× on the documented INT8-KV campaign. That older dtype is a limitation on literal transfer to today's NVFP4 default, but the result directly rejects a launch-only story without fresh attribution. Host fold/submit was only a few tenths of a millisecond against 10–15 ms GPU rounds.

CR10: local `docs/performance.md`, W=4/6 follow-up (**local measured/classifier authority**). Persistent SwiGLU/down fusion was rejected because it would save approximately 170–220 KiB of activation traffic while streaming about 151 MiB weights per W5 layer. The existence of a megakernel compiler does not change that ratio. Existing attention already double-stages KV. These facts distinguish a potentially valuable memory-traffic fusion from a restatement of previously rejected launch elimination.

## Research disposition

| Idea | New versus existing | Decision |
|---|---|---|
| FP16 partial accumulation with FP32 promotion | Potential new precision profile at suitable shapes | Conditional, medium/low priority; FP64 represented-input oracle and model-state quality required |
| TMA and tile autotuning | Already available in NInfer's candidate space | Compare only an admitted weak shape; no generic replacement |
| Remove Python/plugin overhead | NInfer already C++ with CUDA Graphs | No feature gap |
| Persistent inter-op schedule | No compatible SM120 path established | Reject wholesale port; revisit only with new material traffic/latency attribution |
| Device launch / host seam removal | Already measured, neutral/regressed | Do not retry without a changed bottleneck |
| Generic graph/compiler runtime | Conflicts with target-specific ownership preference absent deliverable need | Do not introduce |

There is no defensible numerical estimate of NInfer speedup from these sources. Their role is to suggest bounded kernel experiments and prevent architecture-label and kernel-to-end-to-end extrapolation.
