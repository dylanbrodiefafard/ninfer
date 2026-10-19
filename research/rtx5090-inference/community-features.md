# Community workload evidence and feature opportunities

Access date: 2026-09-26. This supplements `local-engines.md`; sources CF-01..05 are primary user reports or inspected implementation. External timings are not reproduced. No feature implementation is assigned by this research report.

## CF-01 — General JSON outputs are a real gap; tool grammar already exists

Community source: `https://www.reddit.com/r/LocalLLaMA/comments/1w821fg/ninfer_vs_llamacpp_vs_vllm_quality_speed/`. Author skipped structured extraction because NInfer rejected `response_format:json_object`. Current local source confirms `src/serve/openai_schema.cpp:448–457` only accepts `{type:text}`; `src/serve/responses_schema.cpp:693–703` only accepts `text.format` text. General JSON object/schema responses would benefit extraction and structured agent answers.

This does **not** mean NInfer lacks constrained decoding. `docs/serving.md:340–378` documents Engine-owned Qwen tool-envelope/argument grammar in ordinary and speculative target sampling, duplicate/required-key controls, final validation, and explicit rejection of unsupported schema assertions. Existing CPU XGrammar compile/match plus GPU sampling/state transaction is relevant infrastructure. Extending it must define reasoning-to-answer transitions, general JSON termination, supported schema subset, tool coexistence, token-limit incomplete output, sampling/penalty semantics, and speculative grammar rollback. A prompt that asks for JSON is not an implementation of a response-format contract.

**Tool choice is endpoint-specific.** Responses restricts choices to auto/none (`responses_schema.cpp:500–524`, docs/serving.md:283,358). Required/named Responses choices are an incremental feature. Chat Completions already parses required and named (`openai_schema.cpp:346–381`); translation filters declarations for named mode (`translate.cpp:85–102`). Do not describe all endpoints as rejecting them. Parser acceptance/filtering also does not by itself prove a guaranteed forced call; a future semantics investigation should check actual generation behavior before claiming strict forcing parity. This research does not assign such an implementation audit.

## CF-02 — Same-model Gist demonstrates benchmark and feature confounding

Source: `https://gist.github.com/PierpaoloPernici/f1d1382f8e357b4faffb1a9f584cc1df`; API `https://api.github.com/gists/f1d1382f8e357b4faffb1a9f584cc1df`, created2026-08-17, updated2026-08-26. Attached README and benchmark reports were read. Harness source `https://github.com/SeraphimSerapis/tool-eval-bench` is identified by the author as `v2.5.1.dev30+gded5b8f04`.

The original comparison used NInfer groupwise-int/INT8 KV and vLLM NVFP4/FP8 KV. Later NInfer NVFP4 update reran **C1 only**, with INT8 KV, not the quality/concurrency suite. Headline generation rates divide output tokens by wall time including prefill. NInfer server was limited to C2 when client sent C4, so that row includes queuing;32K×C4 lost one request. Warm NInfer run changed PP/TG sizes and cannot be paired directly. One-sample points and prefix reuse effects further limit speed conclusions. Reported old128K MTP ceiling is not current capability.

Useful feature evidence:69-scenario tool suite gives NInfer123/138 and custom-template vLLM134/138 after author's grader correction; structured-output category5/12 versus12/12; required-tool and schema-JSON scenarios failed/partially passed on the old setup. These are old-artifact/template/API results, not intrinsic model-quality scores or proof that today's grammar is deficient. The original full NVFP4 quality evaluation was **not** rerun. Preserve the real prompts/scenarios if designing future schema behavior; do not use the scores as a current regression claim.

The Gist itself reports conditional acceptance metrics inconsistently across sweeps and spec-bench. Do not turn summed per-position survival rates into tokens/round without confirming whether the bonus token and truncated draft length are counted. This reinforces the campaign's recommendation to measure committed tokens and complete rounds directly.

## CF-03 — High-throughput Reddit dashboard is a lead, not a reproducible winner

Source `https://www.reddit.com/r/LocalLLM/comments/1wip08q/finally_got_my_1x_5090_setup_dialed_in_for/`. Inspected body/comments claim920tok/s **aggregate five decoding lanes** (about184 each) while a sixth request prefills at414tok/s. Conditions: custom unnamed NInfer fork, NVFP4 plus9FP8 tensors, MTP4, overclock/power limit, rotated KV, NVMe session spill and turn checkpoints. No repository or reproducible config was supplied in inspected comments. Search publication-age and rendered age disagree, so only retrieval date is trusted here.

C5 and the modified artifact are outside the current matching contract. The useful workload question is decode latency while long new prompts enter, including tool idle periods and cache re-touch. Current NInfer already has chunked prefill/round scheduling, turn checkpoints and Qwen3.8 tiered retention. Nothing in this dashboard establishes a missing mechanism or better C1..4 throughput. Persistent preemptive lane swapping must not be smuggled in as an optimization of the bounded nonpreemptive Engine.

## CF-04 — Durable Responses are different from disk KV

Source `https://github.com/alphastorm/omp-ninfer`, with exact release/runtime anchors in LE-03. Current `docs/serving.md:534–558` says Responses history is process-local LRU and lost at restart. Current Qwen3.8 disk KV retention does not by itself preserve response IDs, typed Items, lineage, credentials or transaction publication across restart.

OMP persists explicit complete continuation state and supports restart plus sibling forks. Historical109,589token restore took24.8s including first-touch restore, with separate56.6s model load;0.778s server TTFT is **after restoration**, not restart time. Later5.2GB restore examples3.6–3.8s are different profiles. Distinct possible deliverables are (1) durable protocol history with recomputation when no reusable state exists and (2) durable complete model-state restore. Choosing one requires an explicit product contract; neither implies Qwen4 host/disk KV authority. Current exact-artifact/state/grammar identity must remain valid.

## CF-05 — Idle GPU keep-warm: precise scope and cost

Sources `https://github.com/alphastorm/omp-ninfer/blob/main/docs/measurements/2026-09-26-engine-keep-warm.json`, `https://github.com/alphastorm/ninfer/tree/v0.6.11-qwen38-5090-beta.1` (commit32c21f73). Release source was inspected; source file `src/core/gpu_keep_warm.cu` and Engine idle loop add one32-thread3.5ms spin every10ms on a separate stream. Runtime default0, released OMP profile60seconds; no machine clocks/settings were changed during this research.

Measured profile was **Qwen3.8 groupwise-int/BF16 KV/MTP3**, Windows driver610.88 through WSL2/Docker. Each arm first warmed a~30K system/tool prefix; numbers refer to incremental prefill of new two-message sessions after reuse. After12–58s idle, enabled prefill155–157ms/TTFT173–181ms versus253–304ms/316–366ms without. Power rose~71W while held; estimated2.3W average applies only to the author's62-hour trace.89role-corpus cases were byte-identical; this does not establish NVFP4 same-hardware results here. Candidate measurement usedc6bd1674; released code adds a skip-if-previous-spin-pending guard.

Useful bounded followup is idle-to-first-output latency/power under actual workload and GPU sharing, with no expectation of sustained decode gain. Replaying a whole model just to hold clocks would be a different and much more expensive mechanism. Keeping this opt-in preserves an explicit latency/power choice.

## Disposition

Community sources yielded concrete feature gaps and useful workload distinctions, not a new trusted speed leaderboard. General JSON response constraints and durable Responses are useful separate product choices. Copy speculation and targeted PDL are implementation candidates only after the numerical/state and classifier checks in `local-engines.md`. Existing adaptive drafting, tool grammar and prefix retention are not missing features.
