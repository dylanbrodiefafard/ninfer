# Independent coverage depth audit

Retrieved 2026-09-26. The user-requested second pass found material omissions in the first pass; the initial closure was premature. This document records the orchestrator's independent branch alongside the three delegated depth audits. This is source research, not a measured NInfer performance campaign.

## TokenSpeed: omitted engine, two concrete dependency patterns

Primary sources:
- https://github.com/lightseekorg/tokenspeed
- https://lightseek.org/blog/tokenspeed-qwen3-8.html
- https://github.com/lightseekorg/tokenspeed/issues/947
- https://github.com/lightseekorg/tokenspeed/blob/main/python/tokenspeed/runtime/grammar/capturable_grammar.py
- https://github.com/lightseekorg/tokenspeed/blob/main/python/tokenspeed/runtime/execution/drafter/dflash.py

Inspected snapshot `8bb4a47` in `/tmp/ninfer-research/tokenspeed-depth`. The August12 article concerns a 2.4T MoE on datacenter multi-GPU systems, not the single5090 dense27B target. Its SM120 tracking issue947 is closed as inactive/not planned, but that is not proof that every engine path lacks SM120 support: GEMM dispatch includes an architecture12.0 FP8 route. Conversely, `tokenspeed-kernel/python/tokenspeed_kernel/ops/sampling/cute_dsl.py:90–105` excludes SM120 from its native sampler, and fused NVFP4 SwiGLU in GEMM CuTe DSL requires SM100-family hardware. No exact current5090/Qwen3.8-27B qualified route was established.

Two inspected mechanisms deserve transfer analysis without importing this runtime:

1. `capturable_grammar.py:379–413` forks a side stream after candidate readiness, performs D2H, CPU grammar matching and H2D, and joins when the mask is consumed. NInfer currently puts that host callback on the same stream before target verification. The kernel depth report verifies both sides and describes lifetime/commit constraints. This is a concrete scheduling difference, not a claim of measured exposed latency.
2. `dflash.py:908–995` builds per-capture feature projection slices and events and arms incremental projection on an auxiliary stream. It can start processing target feature taps before all target layers finish. The technique report compares NInfer's assembled-feature projection and explains state and arithmetic gates. Datacenter success does not imply spare bandwidth on one5090.

FlatKV's logical-history/physical-state distinction is useful architectural context, but NInfer already has fixed recurrent-state ownership, ReplaySSM and retained-prefix semantics. Importing a general allocator would not itself add a missing capability under C1..4.

## Secondary engine-family closure

These are primary-documentation screens, explicitly not full backend code audits. They broaden discovery without treating every runtime as a matched competitor.

| Source | What was established | Disposition |
|---|---|---|
| https://github.com/mit-han-lab/qserve (redirects to OmniServe) | QServe W4A8KV4 and LServe sparse long-context work; different quantization and sparse attention contracts | Useful historical mechanisms; no exact5090/NInfer improvement established; new artifact or approximation needs separate admission |
| https://github.com/mlc-ai/mlc-llm | TVM/TensorIR multi-platform compiler and serving stack | Compiler alternative, no matched Qwen3.8 NVFP4 result established in this screen; generic portability is outside the target's need |
| https://github.com/microsoft/onnxruntime-genai | Generative loop with CUDA/DirectML/TensorRT-RTX execution-provider routes | Feature/deployment reference; no native exact-target speed evidence established |
| https://docs.nvidia.com/deeplearning/tensorrt-rtx/latest/architecture/architecture-overview.html | Separate RTX runtime with on-device JIT; LLM use through ONNX Runtime GenAI/Windows ML, not native LLM deployment out of the box | Distinguish from TensorRT-LLM. Framework/export migration is not a bounded NInfer kernel improvement |

## Second-pass coverage and evidence stopping rule (superseded)

The second pass covered three independent dimensions:

- **Engine lineage:** expanded NInfer fork enumeration, changed-code forks, native competitors SparkInfer/q27/Paddock/PegaInfer/qwentin/Grout/TokenSpeed, upstream vLLM/SGLang/llama.cpp, Radiance descendants and generic framework screens.
- **Execution phase:** idle/startup, prompt preparation and prefix reuse, dense and recurrent prefill, ordinary decode, speculative preparation/verification, tool grammar, sampling/head, long-context KV, vision and serving continuation.
- **Evidence failure modes:** actual versus configured context, per-request versus aggregate throughput, identical artifact/arithmetic versus model-name match, current versus stale code, corpus dependence, harness environment bugs, approximate state/head methods, inaccessible evidence and negative results.

The resulting investigation is comprehensive for choosing NInfer follow-up work, not an assertion that all public/private engines or every fork commit have been audited. The469-row fork inventory is discovery coverage; its evidence-level field identifies the smaller inspected subset. Reddit and articles were discovery inputs, with consequential technical claims traced to primary sources where accessible. The q36 unavailable repository and New Guard inaccessible detailed report remain named limits; their unobserved details do not underpin top recommendations.

All material new leads discovered in those sweeps received an inspected or screened disposition in the four depth reports. The subsequent completeness audit nevertheless found additional engines; this was a statement about the then-known set, not evidence that discovery was exhausted. See completeness-audit.md and the three omission reports for the expanded coverage. Remaining uncertainty concerns whether proposed changes actually improve this checkout: current callback exposure, GDN phase share, exact-shape cache effects, sampler/state correctness and matched end-to-end results. Those require future targeted experiments; collecting more unrelated engine headlines cannot settle them. No weights, dependencies or candidate binaries were installed, and no GPU benchmark or hardware setting was changed.
