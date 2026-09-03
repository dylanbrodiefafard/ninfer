# Qwen3.8-27B NVFP4 active work

This file contains only current work for the supported `qwen3.8-27b/nvfp4` product on one RTX
5090. The default measurement profile is NVFP4 KV, CUDA Graphs enabled, prefill chunk 4096, and
startup-fixed C=1/2/3/4 unless a narrower experiment says otherwise. Completed campaigns belong in
the active documentation or git history, not in this backlog.

## Long-context chunk sensitivity

The directly submit-able 101,708-token reproducer and its observed behavior are recorded beside the
fixture:

- `plans/fixtures/qwen38-long-context-chunk-invariance-request.json`
- `plans/fixtures/qwen38-long-context-chunk-invariance-request.md`

### Established facts

- Before `972f3cde`, the default chunk-4096 DFlash2 path could select a registered stop token while
  the structured Qwen output session was still in reasoning. It reproduced on fresh prefill and
  response-checkpoint reuse.
- The runtime excludes registered model stop tokens from ordinary and speculative selection while
  reasoning is open, after `</think>` until non-whitespace answer content begins, and while a
  tools-enabled output has an ambiguous opener or unbalanced `<tool_call>` markers. Stops remain
  eligible after an established answer or complete tool call. A speculative round that crosses
  into a protected state and then selects a stop is discarded transactionally and retried with the
  registered model stops excluded. Caller stops and raw output retain their normal behavior.
- With stop suppression temporarily disabled, the current chunk-4096 route reproduced a registered
  stop after 32 reasoning tokens and emitted no answer on both fresh prefill and VRAM response-
  checkpoint reuse. The guard therefore masks a still-live model trajectory failure; it did not
  affect any prefill-stage localization.
- A target-model/no-spec diagnostic of the failing round measured `<|im_end|>` at 80.91% of the
  actual post-filter sampling distribution. The remaining support was ` Keep` 11.24%, ` Since`
  6.64%, and ` Need` 1.20%; the seeded draw was 0.52699. Premature selection was therefore the
  dominant model probability, not a low-probability sampling accident.
- At the exact failing DFlash2 verification column, the target distribution assigned 67.80% to
  `<|im_end|>`, 12.26% to ` Since`, 6.35% to newline, 5.57% to ` The`, 4.28% to ` Given`, and
  3.75% to ` Keep`; all other tokens were removed by the configured filters. This is the target
  distribution `p` at the selected speculative column. The diagnostic did not retain the proposal
  distribution `q`, so it does not establish the conditional `p-q` correction distribution when
  that correction path selected the token.
- The underlying token trajectory remains sensitive to legal long-prompt chunk boundaries. At the
  common 91,061-token frontier for chunk 4096 versus 1024, layer-0 input, control projection,
  projected Q/K/V, convolution output, and final BF16 convolution state were byte-identical. The
  first difference was the GDN recurrence: relative L2 was `3.68e-6` for its output and `3.52e-5`
  for FP32 state, comfortably inside the Op criteria `4.1e-3` and `2.7e-3`. Layer 1 then received
  different represented BF16 input, so its amplification was not a same-input Op comparison.
- A production-shape direct regression now compares C=6144, T=3404 against
  `1024+1024+1024+332` using the independent FP64 causal-convolution oracle and requires exact
  final BF16 state. It passes. The real-model trace and direct tests classify the dense split as
  qualified recurrence association sensitivity rather than an incorrect slot, convolution, or
  outer-prefill state transition.
- The independent GDN oracle was re-audited against the active model formula and the upstream
  recurrent definition: it consumes represented BF16 Q/K/V and FP32 gates/state, normalizes Q/K in
  FP64 with epsilon `1e-6`, applies decay before the delta correction, reads from the updated state,
  and retains the logical state/output in FP64. A production-geometry T=3404 diagnostic found the
  current chunked route and `1024+1024+1024+332` partition bit-identical, but both had output
  relative L2 `3.90949e-3` and maximum error `1.55941e-4` against the oracle. The relative criterion
  `4.1e-3` passed, while the gross limit `1.52500e-4` failed by 2.26%; final-state relative L2 was
  `2.02075e-3` against `2.7e-3`. This is evidence that the production route is close to the existing
  accuracy boundary, not that the oracle is too low precision.
- Canonical 1024-token micro-tiling inside the GDN Op was rejected. On the real 91,061-token fixture,
  chunk 4096 retained the exact same 57-token completion before and after the prototype, while the
  chunk-1024 control remained on a different 64-token reasoning trajectory. Prefill was 6460.2
  tok/s before and 6452.0 tok/s after, within run noise. The prototype therefore neither repaired
  trajectory sensitivity nor established a performance benefit and was removed.
- Exact layer-0 operands and incoming state were captured for the shared final T=947 prefill unit
  from both chunk routes. Q/K/V/g/beta were byte-identical. Incoming FP32 state differed by relative
  L2 `2.07975e-4`, but independent FP64 replay attenuated that to `8.08324e-6` in output and
  `1.86972e-5` in final state. The production chunked replay instead produced a `5.54177e-4` output
  difference and `3.56142e-5` final-state difference; the direct FP32 recurrent route produced
  `1.46484e-4` and `1.87039e-5`. Against the same FP64 oracle, chunked final-state error was
  `1.0671e-3`, versus `1.4e-6` for the recurrent route. The incoming drift is therefore not the
  dominant source at this frontier: the material amplification comes from the chunked route's
  BF16 intermediates and/or TF32 state-passing arithmetic. Separating those two implementation
  profiles is the next numerical experiment.
- A follow-up replay isolated the normalized-Q/K cast before any chunk W/U or TF32 state-passing
  work. The direct recurrent route with FP32-normalized Q/K had final-state error `1.40391e-6`
  against FP64; materializing the same normalized values to BF16 first raised it to `9.38114e-4`.
  The full chunked route was `1.0671e-3`, so normalized-Q/K BF16 staging accounts for about 88% of
  its observed state error on this real input. Output error rose from `1.58781e-3` to `1.7534e-3`
  under that cast, versus `1.88392e-3` for the full chunked route. Across the two incoming states,
  the recurrent final-state difference remained `1.8704e-5` with either normalization profile;
  the cast primarily introduces absolute trajectory error accumulated across prefill rather than
  amplifying the final unit's incoming-state difference. Promoting `h_chunk` alone is not useful
  for persistent-state correctness because it does not feed persistent state.
- The selected correction retains the public BF16 inputs and independent FP64 oracle while using
  FP16 for the private normalized Q/K and W/U/v_new/h_chunk chunk workspaces. Normalized Q/K are
  bounded and FP16 retains a 10-bit significand, matching TF32 operand precision, at the same two
  bytes per element as BF16. Extending that profile through the other private workspaces removes
  their BF16 rounding without changing workspace or CUDA Graph allocation bytes; width-one decode
  still bypasses the chunked route. At production geometry T=3404, the independent oracle measured
  output relative L2 `1.87873e-3` and maximum error `8.27174e-5`, passing limits `4.1e-3` and
  `1.52500e-4`. Final-state relative L2 was `4.20052e-4`, down 79.2% from the BF16-workspace
  same-input BF16-workspace baseline `2.02075e-3`, and maximum state error was `1.39668e-4`
  against the `9.02478e-4` gross limit. The same-input BF16 baseline reproduced output relative
  L2 `3.90949e-3` and its gross-bound failure. The full focused GDN suite passes, including raw-Q/K
  BF16, chunk/tail, distinct-state, and partitioned execution.
- A paired cold-L2 CUDA-Graph operator A/B alternated five baseline and five candidate binaries
  built from the same source state at each width (20 warmups and 200 samples per run). At T=1024
  the median-of-run medians was `163.872 us` for BF16 private staging and `165.888 us` for FP16, a
  1.23% cost. At T=4096 it was `718.848 us` versus `720.576 us`, a 0.24% cost. Workspace remained
  exactly
  `71,499,776` bytes at T=1024 and `285,999,104` bytes at T=4096, with five graph nodes in both
  profiles. At width one, five alternating 500-sample runs were exactly tied at a `6.144 us`
  median because both builds dispatch the unchanged recurrent kernel.
- Matched Engine A/Bs on the DFlash2 NVFP4 artifact used NVFP4 KV, CUDA Graphs, C=1, DFlash-5,
  and the optimized draft head (two warmups plus five measured repetitions). Candidate prefill
  throughput versus the BF16-private baseline was `10,767.5` versus `10,774.9 tok/s` at T=1024
  (-0.07%) and `11,621.1` versus `11,646.2 tok/s` at T=4096 (-0.22%). Removing redundant
  normalization of the recurrent tail raised the final T=3404 candidate to `10,255.3 tok/s`
  versus `10,222.7 tok/s` for the baseline (+0.32%). The 512-token decode result was `134.33` versus
  `133.59 tok/s` (+0.55%) with identical DFlash acceptance length `2.07287`, consistent with zero
  direct decode cost. Workspace capacity, per-case allocator peaks, CUDA Graph observed memory
  (`6,291,456` bytes), and all reported startup reservations were byte-identical.
- A matched XAttention-enabled Engine A/B on the same RTX 5090 used tau 0.9, NVFP4 KV, CUDA
  Graphs, C=1, T=32768, and prefill chunk 4096. Three order-balanced pairs each used two warmups
  and five measured repetitions after the GPU queue was clear. The FP16-private candidate's paired
  median throughput effect was +0.10%, with individual pairs from -0.32% to +0.34%; this establishes
  no measurable XAttention prefill slowdown. Median-of-run medians was 11,664.7 versus 11,635.4
  tok/s (+0.25%). Workspace allocator peak (`723,812,352` bytes) and observed CUDA Graph memory
  (`2,097,152` bytes) were byte-identical. The GDN correction remains active in the model's GDN
  layers while XAttention changes only the GQA attention layers.
- On the real 91,061-token fixture at prefill chunk 4096, the matched BF16-private baseline
  reproduced its known normal 57-token completion at `6,430.0 tok/s`. The FP16-private route
  completed normally in 50 tokens at `6,413.0 tok/s` (-0.27%). Its chunk-1024 control also closed
  reasoning, emitted an answer, and stopped normally in 46 tokens at `4,953.1 tok/s`; the prior
  BF16-private chunk-1024 control had remained in reasoning through the 64-token diagnostic limit.
  The two legal chunk sizes still take different qualified floating-point trajectories, but both
  now clear the behavioral failure frontier and neither selects a premature structured stop.
- The 27B control route previously dropped split-K at arbitrary power-of-two widths. Direct RTX 5090
  screening supports split8 through T=1792, split4 through T=3584, and split2 through T=4096. At
  T=3072 the control projection falls from 45.1 to 34.4 us, and at the real 3404-token tail from
  46.7 to 34.8 us; the finer routes are also closer to the independent FP64 oracle.
- Canonical chunk 128 is not a viable product workaround: it reduced prefill throughput by 75.8%
  and made TTFT 4.13 times slower than chunk 4096 on the reproducer.
- Exact token identity across chunk partitions is not the semantic target. The target is correct
  represented state transitions and valid structured output, with any approximation covered by an
  explicit numerical or behavioral criterion.

### Investigation design

Dense behavior and DFlash2 concurrent row isolation are classified. Continue with XAttention.
Tau 1.0 is the dense-identity gate. Tau 0.9 remains an explicitly approximate mode;
its measured long-context result was about +49.7% prefill throughput and -33.1% TTFT, but it needs
perplexity and long-context behavioral qualification before any default-policy decision.

Current XAttention numerical evidence on 2026-09-02:

- Tau 1.0 does not enter the sparse launcher; it selects the ordinary exact-NVFP4 dense kernel, so
  the identity gate has no separate arithmetic path.
- The focused GQA proof passes for both registered geometries. It checks the production keep-list
  against an independent paper inverse-reshape/mass oracle, includes adversarial inputs that
  distinguish the retired four-antidiagonal heuristic, and compares the retained-tile attention
  result directly with an independent FP64 softmax oracle under the NVFP4 criterion.
- On the isolated 27B T=4096 synthetic operator workload, tau 0.9 kept 90.7%, 90.2%, and 90.1% of
  visible pages at 32k, 64k, and 128k. Median operator time improved from 12.438 to 11.138 ms at
  32k, 25.494 to 22.750 ms at 64k, and 51.625 to 46.402 ms at 128k. Rank scratch was 120, 240,
  and 479 MB respectively. These random-input timings qualify the implementation cost only; they
  neither replace real-model keep density nor establish quality.
- The real-model CUDA-Graph prefill PPL comparison passed at 8k and 32k. Against dense NVFP4,
  tau 0.9 changed mean NLL by -0.00084 at 8k (paired 2-sigma 0.00171) and -0.00234 at 32k
  (paired 2-sigma 0.00488); both are unresolved inside paired noise, with no non-finite values.
- The first 32k XAttention cell exposed an integration defect before scoring: CUDA Graph startup
  consumed 70 MiB against the 12 MiB ordinary allowance. XAttention prefill is already eager and
  decode remains exact, so disabling decode graphs would be the wrong fix. XAttention-enabled
  plans now reserve the established 96 MiB large-topology allowance. Ordinary, MTP, and DFlash
  startup all pass with the new bound; this changes reserved VRAM, not execution work.
- The required 64k NIAH gate passed under tau 0.9, NVFP4 KV, CUDA Graphs, and MTP: the model returned
  the exact `ORCHID=493817; COLOR=COBALT` record from a 64,511-token prompt. Tau 0.9 remains an
  explicit approximate experiment rather than the product default; the isolated random-input
  keep density and scratch cost do not support changing the default policy on their own.

Final validation covers fresh prefill and response-checkpoint reuse at C=1/2/3/4, confirms that
reasoning cannot terminate on a registered stop token, confirms normal post-reasoning stopping, and
benchmarks TTFT/prefill throughput at the default chunk 4096. Exact token equality across legal chunk
partitions is required only if the localized operation's actual semantic contract requires it.

### Structured stop eligibility

The implementation, correction review, and maintainer approval are complete.

- `OutputSession` owns reasoning closure, first non-whitespace answer establishment, incremental
  tool-marker state, and speculative preview commit/discard. Tool-output eligibility is derived by
  ordinary and incremental prompt preparation from tool schemas, assistant tool-call history, or a
  Tool-role message; raw-token preparation remains unstructured and there is no public override.
- The executor suppresses registered model stops at committed state boundaries. A speculative round
  that enters a protected state and then selects a registered model stop is rejected independently
  of the caller's default-stop policy and retried with those stops suppressed.
- Rejection folds zero GDN columns, preserves the committed DFlash context frontier, invalidates the
  overwritten tail hidden value, reverses stochastic device token counts, and restores adaptive
  state plus every per-attempt speculative counter. Cancellation also invalidates an in-flight
  speculative tail before retention. Greedy penalty requests do not reverse counts because greedy
  selection never increments them.
- Greedy DFlash/MTP target verification receives the active sampling configuration, so target
  argmax honors the same suppression as proposal/acceptance.

Qualification on 2026-09-02:

- A clean full build and the full 104-entry CTest sweep had zero failures; 11 real-artifact tests
  skipped normally without their opt-in environment variables. The affected 13-test frontend/runtime,
  sampling, speculative, replay-fold, CLI, and serving-schema subset also passed independently.
- The relaxed real DFlash test passed with the local NVFP4 codebook artifact at fixed k=5 C=1,
  including adaptive N=7 concurrency, terminal/queue behavior, RAM reseed, and in-flight restore.
  The subsequent numerical campaign classified the k=5/k=7 C=1-versus-C=3 differences against
  independent Op/state oracles; exact amplified greedy identity is not the numerical criterion.
- The long fixture completed from fresh prefill and resident response-checkpoint reuse with a normal
  post-reasoning stop: 49 generated tokens, 125 reasoning characters, and 70 answer characters in
  both runs. This local `qwen3_8_27b_nvfp4_dflash_w8.ninfer` artifact rendered 91,061 prompt tokens;
  the fixture hash remained the recorded `fbe71d...e98`.
- The temporary exact-output Engine A/B used the 16-token prompt, 128 outputs per lane, greedy
  sampling, model-default termination off, NVFP4 KV, DFlash k=5 optimized head, graphs, one warmup
  pair, and eight alternating measured pairs at startup-fixed C=1/2/3/4. Median full-wave deltas
  were -0.020% C1, +0.008% C2, -0.126% C3, and +0.036% C4. The worst slowdown was 0.036%, with no
  systematic regression; the temporary harness and API toggle were removed.
- The requested final Sol review found no remaining correctness, API, or performance defects against
  the seven handoff corrections.

## GPU-gap attribution

The DFlash4 NVFP4 host-overhead campaign removed avoidable telemetry and terminal-publication work
but showed no material uniform end-to-end decode gain. Before another CPU cleanup, capture one
warmed steady DFlash4 decode wave at each C=1/2/3/4 and assign every repeatable GPU-idle interval to
executor scheduling/frame packing, graph selection/install, launch, D2H/output-state folding, or
response publication. Change a CPU path only when its owned gap is large enough to move the
end-to-end wave metric. Preserve exact work and output hashes in the A/B.

Do not add an online graph autotuner, another graph lookup cache, or per-round profiling output
without this attribution. The transition-derived graph profiles and current linear selection are
the measured choice.

## Smaller opportunities

- Cold or rewritten 150k-token prompts still spend about 21 ms in full-prompt BPE. Incremental
  turn-2+ encode is already shipped. Any new tokenizer design must beat both the plain-short-word
  and tool-heavy fixtures without token drift; do not revive the stack-128 rewrite.
- Adaptive draft is strong in aggregate but retains workload-specific losses, particularly the
  mixed CUDA/Python case. Its measurement history remains in `plans/adaptive-draft-bench.md`; it is
  lower priority than the numerical work and GPU-gap attribution.
