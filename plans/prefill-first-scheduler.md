# Prefill-first admission burst and checkpoint performance

## Deliverable

Replace the latency-oriented decode/prefill alternation with a throughput-first Engine schedule:

1. Continue an admitted request's prefill/finalization to completion.
2. While a lane, exact admission resources, and the current decode-debt budget remain available,
   admit the oldest eligible queued request and prefill it to completion.
3. When no further request can currently be admitted or the frozen burst budget is exhausted,
   launch one maximal decode batch containing every decode-ready lane.
4. Repeat; a decode completion that frees capacity may open another prefill admission burst.

The intended startup burst at concurrency `C` is sequential complete prefills followed by the
largest available exact-`B` CUDA Graph decode. Prefill remains single-owner and request-local;
this task does not introduce mixed or batched prefill.

## Semantic decisions

- Queue eligibility is frozen once per admission attempt from one bounded pending snapshot. New
  arrivals wait for the next completed GPU-unit or idle-control boundary.
- Once any decode-ready donor exists, the admission burst freezes a budget equal to the number of
  free lanes at that point. Every successful bind consumes one unit even if that request cancels or
  terminates on its prefill-produced token. Only a maximal decode refreshes the budget.
- A pending request that cannot currently fit does not stop a qualified existing backfill under
  the resource policy. If no queued request can be admitted, decode progresses.
- A copy-hold with unfinished RAM D2H/H2D is not runnable prefill. Existing decode-ready lanes may
  decode while the copy engine runs; once ready, admit-complete and prefill regain priority.
- Cancellation, timeout, output publication, state retention, and request resource accounting are
  still resolved only at legal boundaries. Scheduling must preserve lane/state/output isolation;
  exact token identity is qualified per target batch profile rather than across different `B`.
- Remove the mandatory decode round between admissions and the historical
  `previous_unit_was_decode` policy state. Do not retain a compatibility option for the old policy.
- Preserve exact maximal decode membership: every decode-ready lane participates, including all
  requests completed during the preceding prefill burst.

## Implementation boundaries

- Rewrite `ConcurrentExecutor::worker_loop` selection around directly runnable work rather than
  the previous-unit flag.
- Reconcile admission-turn, protected-head, and temporal-backfill logic with bounded admission
  bursts. Exact entitlement accounting remains authoritative; remove progress assumptions that
  exist only to force donor decode between admissions.
- Keep lane planning, one-prefill ownership, copy-hold ownership, and target Program math intact
  unless inspection identifies a direct requirement of the new scheduling contract.
- Update the active concurrent-inference architecture wherever it specifies alternation,
  decode-between-admission gates, progress proofs, or examples. Review user-facing serving and
  performance documentation for claims affected by longer decode pauses or burst batching.

## Low-level design

### Boundary transaction

`ConcurrentExecutor::worker_loop` remains the sole GPU execution and active-slot owner. Every loop
iteration begins at a legal completed-unit boundary and performs the existing transaction in this
order:

1. Under `execution_mutex_`, remove pending cancellations/timeouts visible at boundary entry.
2. Snapshot and resolve active cancellations, including rollback/retain and slot release.
3. Rebuild `RoundMembership` from every surviving `decode_ready` slot. The mapping remains compact,
   ascending by stable lane, and is immutable for any decode launched in this iteration.
4. Select exactly one next GPU unit, or make one bounded control/admission transition and iterate.

No change is made to `resolve_prefill_step`, `run_prefill_step`, `run_decode_round`, Program state
transactions, sampling, token publication, or per-lane storage. A final prefill continues to commit
its first generated token before setting `decode_ready`; it may also terminate a one-token request
and free the lane. Subsequent decode membership includes a surviving lane only at the next boundary.

### Runnable-work selector

Delete `previous_unit_was_decode`. Add worker-owned `decode_admission_budget`, absent when no donor
needs progress. Selection becomes state-derived:

```text
boundary maintenance
membership = all decode-ready lanes
if membership is empty:
    clear decode_admission_budget
else if decode_admission_budget is absent:
    freeze it to C - number_of_active_slots

if copy_hold exists:
    if its async RAM copies are not ready and membership is nonempty:
        run maximal DecodeRound(membership)       // overlap compute with copy engine
    else:
        advance admit-complete
        if admit-complete queues another unfinished copy and membership is nonempty:
            run the already-frozen maximal DecodeRound(membership) in this transaction
        if admit-complete starts/finalizes prefill:
            that is this iteration's GPU unit
    continue

if prefill owner exists:
    run its next PrefillChunk/finalization unit
    continue

if pending queue was nonempty at boundary entry
   and (membership is empty or decode_admission_budget > 0):
    progress = try_admit_one(one frozen pending snapshot)
    if progress is RanGpuUnit or CopyHold:
        if membership is nonempty:
            decrement decode_admission_budget       // bind consumed it permanently
        continue
    if progress is ControlProgress and membership is empty:
        continue                         // re-enter wait/selection with current queue

if membership is nonempty:
    run maximal DecodeRound(membership)
    clear decode_admission_budget
    continue

idle/wait
```

Each completed prefill clears `prefill_lane_`; the next boundary tries FIFO/protected admission
again if budget remains. The burst ends when `try_admit_one` finds no feasible head/backfill, the
queue snapshot is empty, all resources are occupied, or the frozen budget reaches zero. A terminal
prefill or cancellation may physically free its lane but does not refund budget, preventing a
continuous stream of one-token requests from starving older decode rows.

`try_admit_one` takes exactly one `pending_snapshot()` and advances through it. Removing a visible
cancelled, expired, or permanently invalid entry advances within that same finite vector; it never
resnapshots before returning. Requests appended after the snapshot are ineligible until the next
completed-unit or idle-control boundary. This makes one selection transaction finite under
continuous ingress.

### State and priority table

| Executor state | Decode membership | Next action |
|---|---:|---|
| `copy_hold_`, copies unfinished | nonempty | maximal decode while DMA overlaps |
| `copy_hold_`, copies unfinished | empty | wait copy, complete admission, start prefill |
| `copy_hold_`, capture D2H ready | either | evict victims; possibly queue restore H2D |
| `copy_hold_`, restore H2D newly pending | nonempty | maximal decode from frozen membership |
| `copy_hold_`, all copies ready | either | complete admission and start/finalize prefill |
| `prefill_lane_` | either | next prefill/finalization unit |
| no owner, feasible pending, burst budget positive | either | admit oldest qualified request |
| no owner, burst budget exhausted | nonempty | maximal decode and refresh next burst |
| no owner, pending blocked/empty | nonempty | maximal decode |
| no owner, pending empty | empty | idle/wait |

Copy-hold retains priority over admitting another request because it already owns a lane and Plan.
Decode during copy-hold is not a decode-over-prefill choice: no prefill unit is runnable until the
copy dependency is satisfied.

### Admission and protection

`try_admit_one` remains the single admission authority and continues to:

- prefer the FIFO head whenever an exact lane/resource plan is feasible;
- establish one frozen protection epoch when that head is blocked by active entitlements;
- admit only persistent-safe or temporal-credit-qualified later requests;
- enter drain when the protected head would fit absent temporal borrowers;
- account every admitted lane in the authoritative current resource set.

The old admission-turn predicate `(membership empty || previous unit was decode)` is removed.
Protection permits several admissions before donor progress, bounded by frozen decode debt:

1. When the first decode membership of a burst appears, the worker freezes at most `C-1` successful
   admissions before mandatory decode. Each successful bind spends one unit without refund.
2. One-prefill ownership makes each admitted prefill finite and prevents nested admission during a
   partial prefill.
3. Budget exhaustion or infeasibility makes every decode-ready active lane join the next maximal
   round, after which a later burst receives a new budget from then-free physical lanes.
4. A decode round commits positive finite output progress or terminates each row; finite declared
   output bounds eventually release lanes/resources.
5. Continuous ingress cannot replenish burst budget, alter a frozen protection epoch's donors or
   temporal credit, or bypass exact resource accounting. Thus every surviving donor decodes after
   at most `C-1` intervening successful admissions, including terminal-prefill requests.

Temporal credit remains conservative: consecutive temporal admissions subtract their complete
service projections before they run, while later boundary snapshots observe the prefill work each
has actually consumed. No safety calculation relies on wall-clock completion order.

### Cancellation, failure, and shutdown

- Active and queued cancellation remains visible at every chunk/round boundary, including between
  consecutive prefill chunks and between completed prefills in a burst.
- Cancelling the prefill owner clears transient ownership and rolls back/aborts through the existing
  Program contract before another admission is attempted.
- Cancelling a copy-hold drains its in-flight copies before releasing claims/victims, unchanged.
- A Program/state-integrity exception remains Engine-wide failure; scheduling does not attempt
  per-row recovery.
- Shutdown remains boundary-cooperative and fails active/queued work through `fail_all`.

### Lane, output, and numerical profile

Scheduling changes only which complete Program unit runs next. After admission binds a winning
Plan, it never changes:

- that request's stable lane, Plan, KV/GDN/speculative state ownership, or context positions;
- per-request sampling configuration, RNG state, generation budget, stop policy, or output session;
- the `lanes`/`budgets` pairing passed to `decode_batch` and `resolve_pending_batch`.

Admission timing may legitimately change the selected free/retained lane, prefix-reuse source/path,
reused-token count, RAM-copy timing, and first/subsequent exact-`B` numerical route. MTP packed
concurrent verification is not required to be token-identical to C=1; target-qualified floating
profiles can produce greedy flips without lane mixing. Therefore no universal isolated-versus-
concurrent token equality is claimed.

Isolation evidence instead uses distinct prompts, fixed options/seeds, prefix reuse disabled, and
the same exact-`B` target profile under controlled/reversed submission assignment where supported.
Each result must contain only its own licensed output and satisfy the existing target-specific
completion/numerical criteria.

### Documentation replacement

Revise `docs/maintainer/concurrent-inference-architecture.md` as the active authority:

- one-prefill ownership becomes contiguous complete-prefill ownership rather than chunk insertion;
- admission turns are allowed at every owner-free boundary;
- the alternating policy and latency-bound rationale are removed;
- admission/protection progress uses the fixed-capacity burst proof above;
- examples show multiple complete prefills followed by exact maximal decode;
- copy-hold overlap remains the sole decode-priority exception.

Review `docs/serving.md`, `docs/performance.md`, `docs/cli.md`, and `README.md` only for concrete
claims of interleaving or decode latency. Do not add general scheduler exposition outside the
maintainer authority.

### Implementation and evidence sequence

1. Capture a representative old-policy scheduler baseline before editing the selector.
2. Change the worker selector and active architecture together.
3. Build and run focused host checks plus the existing real concurrent/context-checkpoint routes.
4. Extend a real concurrent Engine fixture with a controlled long first prefill, then queued second
   request (`output_tokens=2`, reuse/default stops off), and assert the observable delta counters:
   one first decode round with two row-rounds. Use existing target-specific isolation criteria.
5. Exercise bounded donor progress with terminal-prefill admissions, protected backfill, RAM
   capture-D2H to restore-H2D, prefill cancellation, and copy-hold cancellation through the
   relevant real integration routes.
6. Measure new-policy admission burst, first decode `B`, aggregate wall time, and decode pause.
7. Run the context-checkpoint enabled/off campaign; optimize staging only if measured attribution
   supports a concrete change, then requalify restore correctness and performance.

## Correctness evidence

- Build the affected runtime/Engine targets.
- Run focused host tests covering admission/resource helpers and affected documentation checks.
- Run the controlled real concurrent route above and assert counter deltas establish exact first
  decode `B=2`, both outputs complete, and target-qualified lane outputs remain isolated.
- Exercise a long decode-ready donor against replenished one-token admissions and prove donor
  decode progress within the frozen budget; cover existing protected backfill and RAM copy-hold
  routes under multi-admission bursts.
- Exercise cancellation of prefill ownership and copy-hold ownership. Do not add tests for private
  source shape or the removed policy flag.

## Performance evidence

### Scheduler

- RTX 5090 / `sm_120a`, supported qwen3.8-27B NVFP4 artifact, `--kv-dtype nvfp4`.
- Compare old and new policies on concurrent cold prompts at representative `C` values, including
  `C=8` when capacity permits.
- Record aggregate wall time, aggregate prefill throughput, time until the full decode batch is
  formed, decode batch size, aggregate decode throughput, and the intentional maximum decode
  pause during the prefill burst.
- Use whole-Engine timing first. Use an Nsight Systems trace only if launch/synchronization gaps or
  actual batch formation cannot be established from existing metrics.

### Context checkpoints, after scheduler qualification

- A/B the same long prompt with the automatic context-checkpoint ladder enabled versus
  `--context-checkpoints off`, holding artifact, backend, prompt, prefill chunk, KV dtype, and
  output settings fixed.
- Cross multiple configured marks so the measurement includes repeated freezes; separately retain
  a no-mark/short control where useful.
- Record total and trailing-window prefill throughput plus per-mark boundary gaps. Inspect compute,
  D2D, D2H, event waits, and overlap on compute/copy streams when attribution is needed.
- Specifically test whether `fence_staging_copies()` serializes the next freeze on the previous
  host copy and whether rollback staging reload adds an avoidable H2D round trip.
- If checkpoint overhead is material and attributable, implement the strongest state-safe staging
  or synchronization correction, then repeat the same A/B. Preserve exact checkpoint restore
  behavior with the real context-checkpoint integration route.

## Coordination and exclusions

- Another active effort owns broad host-CPU work, including statistics publication and related
  serving/worker overhead. Preserve its workspace changes and do not duplicate that campaign.
- Cancellation handling currently does two things per boundary: reads at most `C <= 8` active
  atomic flags and scans the bounded pending queue for cancelled/expired entries. Keep semantic
  handling in this task; measure or report queue-scan cost only if relevant, and do not fold a
  general CPU optimization into this change.
- Do not retune model kernels, prefill chunk size, tokenizer, serving JSON, or unrelated CUDA Graph
  preparation in this task.

## Post-prefill review backlog

Do not start these until the prefill-first scheduler, its checkpoint follow-up, and their evidence
are complete. Reconcile ownership with `plans/host-cpu-perf.md` before changing code.

- Replace repeated pending-queue cancellation/deadline scans with event-driven cancellation plus
  deadline-ordered expiry, if boundary profiles show meaningful cost under a full outstanding
  queue. Preserve immediate active-lane cancellation checks at legal GPU boundaries.
- Consolidate or move boundary-time statistics publication and KV-RAM snapshot work off the GPU
  execution owner. This is already substantially owned by the host-CPU campaign; review its final
  result rather than producing a parallel implementation.
- Profile the remaining admission-boundary host gap after the other CPU campaign lands: request
  planning/hash work, protected-head/backfill selection, lane-plan invalidation, membership
  assembly, output preview/commit, and control H2D. Optimize only an attributed material portion.
- Review whether asynchronous RAM capture/restore copy-holds can be queued or pipelined more
  effectively when several retained lanes are displaced. The prefill task only gives a ready
  copy-hold immediate priority and uses decode while its transfer is unfinished.
- Revisit CUDA Graph preparation cost and graph-definition footprint for exact batch sizes
  `B=1..C`. The prefill task consumes the largest ready definition; it does not redesign startup
  capture or graph storage.
- Measure large-batch decode scaling separately from scheduler makespan. If an exact larger `B`
  ever reduces aggregate row/token throughput, investigate the decode operator or graph profile
  rather than weakening maximal membership in the scheduler.
- Consider true multi-request/batched prefill only as a separately designed target-runtime and Op
  project. It is not a scheduler extension: it changes activation/workspace geometry, ragged
  attention inputs, state commit, numerical qualification, and likely CUDA Graph ownership.

## Progress log

- 2026-08-31: traced the current 1:1 alternation to `previous_unit_was_decode` in
  `ConcurrentExecutor::worker_loop`; confirmed final prefill already marks its lane decode-ready
  and the next membership rebuild includes it in the maximal batch.
- 2026-08-31: established that decode-between-admission is a fairness/progress policy rather than
  a state-safety requirement. The new fixed-lane/resource-bounded admission burst replaces it.
- 2026-08-31: checkpoint inspection found an explicit `fence_staging_copies()` before each ladder
  freeze; benchmark attribution will determine whether it defeats intended copy overlap.
- 2026-08-31: identified active cancellation polling as `O(C)` atomic reads plus an `O(pending)`
  cancellation/deadline scan at every GPU boundary; broad CPU work remains with the other effort.
- 2026-08-31: recorded non-prefill follow-ups separately for review after scheduler and checkpoint
  work; none is allowed to enlarge the current deliverable.
- 2026-08-31: completed the low-level design: boundary transaction, state-derived runnable-work
  selector, copy-hold exception, admission/protection proof, cancellation/failure semantics, token
  isolation, documentation replacement, and evidence sequence are now explicit.
- 2026-08-31: `gpt-5.6-sol` design review found a terminal-prefill starvation counterexample,
  unbounded queue resnapshotting, incorrect universal token/reuse invariance, an omitted D2H-to-H2D
  copy-hold transition, and insufficient liveness evidence. Revised the design with non-refundable
  decode-admission budget, one finite queue snapshot per attempt, target-profile numerical criteria,
  explicit copy subphases, and observable real-Engine regressions.
- 2026-08-31: revised-design re-review by `gpt-5.6-sol` found no remaining blocking/high gaps.
  Implementation must reset decode-admission budget through every decode launch, including both
  copy-hold overlap paths; use one helper to keep those transitions consistent.
- 2026-09-01: implemented the finite pending snapshot, contiguous prefill priority, non-refundable
  decode-admission budget, and one decode helper covering ordinary/copy-hold launches. Replaced the
  active architecture's alternation/admission-turn contract and added a real MTP exact-B=2 first-decode
  regression using round/row counters. Affected targets compile; host admission/runtime mechanism
  tests pass. Preserved the old-policy serve binary at `/tmp/ninfer-serve-prefill-old` for GPU A/B.
- 2026-09-01: GPU execution is temporarily occupied by the separate host-CPU campaign; real model
  tests and old/new performance runs remain pending and must not disturb that process.
- 2026-09-01: real MTP NVFP4 suite passed for k=3, k=5, adaptive exact batches, and RAM restore in
  flight. The new two-long-prefill fixture observed exactly one decode round and two row-rounds.
  The first fixture attempt correctly produced two B=1 rounds because shared KV capacity covered
  only one 896-token request; corrected the test capacity to the configured three-lane entitlement.
- 2026-09-01: full real MTP context-checkpoint integration suite passed (`ok`), covering concurrent
  capture/restore, D2H/H2D copy-holds, cancellation, retention, and multi-lane execution.
- 2026-09-01: `gpt-5.6-sol` implementation review found no scheduler defect; it identified three
  evidence gaps and one documentation qualification. Added a scheduler-dedicated engine, fenced
  counter reads through the Engine execution lock, repeated the exact-B=2 wave in reverse order to
  verify per-request output ownership, added the terminal-prefill/non-refundable-budget liveness
  case, and qualified the final-donor cancellation rule. Restored the ordinary MTP fixture's
  original capacity so the scheduler entitlement is local to its tests.
- 2026-09-01: the augmented real MTP NVFP4 suite passed, including `ok MTP NVFP4 prefill-first
  scheduling`; it now establishes exact first decode B=2, partner-wave lane/output isolation, and
  the worker-owned C-1 admission bound before mandatory donor decode.
- 2026-09-01: final `gpt-5.6-sol` implementation review found no blocker/high/medium issue after
  replacing host waiter order with worker-owned decode-debt state and replacing the reversible
  two-prompt check with A/B, A/C, and B/C exact-B=2 partner invariance. The augmented real MTP
  suite passed again.
- 2026-09-01: old/new C=2 long-cold-prefill plus 128-token thinking-decode A/B on RTX 5090 formed
  average decode batch 1.32 under alternation versus exact 2.00 prefill-first. Request-wave makespan
  changed from 16.06 s to 15.90 s (1.0% faster); both requests finished together under the new
  policy instead of one receiving low-batch decode while the second prefilled.
- 2026-09-01: three-run 64k cold-prefill checkpoint A/B (warmup excluded, C=1, MTP3, NVFP4,
  4096-token chunks) measured 8461.1 tok/s and 7626.6 ms TTFT with three ladder freezes versus
  8641.6 tok/s and 7467.2 ms with checkpoints off: 2.1% prefill loss and +159.4 ms TTFT.
  Nsight Systems attributed the gap to nine per-mark pinned-image allocations: checkpoint-on added
  196 ms of `cudaMallocHost` time; asynchronous D2H totaled only 16.1 ms on the GPU.
- 2026-09-01: implemented a Program-owned high-water context-checkpoint host-image pool. Retired
  heads and their completion events are reused by exact-layout captures; no theoretical C-by-mark
  allocation is made. Repeating the identical A/B measured 8637.6 tok/s and 7470.8 ms TTFT, within
  0.05% and 3.6 ms of checkpoints-off. The first cold high-water request intentionally still pays
  the one-time pinning allocation. The final full real checkpoint lifecycle suite passed (`ok`),
  including concurrent capture/restore, RAM restore, cancellation, retention, and D2H/H2D
  copy-holds.
- 2026-09-01: SOL pool review found that head reuse tracked capture D2H but not every later H2D
  source use, allocation catches also covered event waits, rollback replacement retired the prior
  valid head too early, and one architecture paragraph named MTP but not DFlash. Re-recorded each
  head event after H2D consumption, moved event waits outside best-effort allocation catches,
  acquire-and-wait now precedes rollback retirement, and documented DFlash cyclic staging. Final
  re-review and post-fix real lifecycle rerun are pending shared-GPU availability.
- 2026-09-01: follow-up SOL review identified one remaining bound violation when rollback
  replacement acquired before retirement. Replacement now waits while the old head remains valid,
  then moves and reuses that exact image, keeping total ownership within `C * (marks + 1)` and
  avoiding a boundary allocation. Final SOL review confirms no blocker/high/medium issue remains.
  The final post-review full real checkpoint lifecycle suite passed (`ok`).
- 2026-09-01: extracted the worker decode-debt transitions into the production-used
  `DecodeAdmissionBurst` state machine and added deterministic coverage for frozen C-1 debt,
  terminal/cancelled admission non-refund, exhaustion, decode reset, empty-membership startup,
  and malformed membership rejection. Added deterministic checkpoint-pool policy coverage for
  exact-layout reuse (including MTP/DFlash separation) and the per-lane mark-plus-rollback
  high-water bound; real CUDA suites remain the proof of events, D2H/H2D ordering, and ownership.
- 2026-09-01: the expanded deterministic admission-policy and Qwen3.6 runtime-mechanism suites
  both pass; affected MTP NVFP4 scheduler integration was rebuilt after wiring the production
  worker through the debt state machine; its final-tree GPU rerun is recorded below.
- 2026-09-01: final-tree qualification reran both real GPU suites successfully: MTP NVFP4
  scheduler integration (k=3, k=5, prefill-first scheduling, adaptive batch/ram cases) and the
  complete Qwen3.6 context-checkpoint lifecycle suite (`ok`).
- 2026-09-02: rebased over `perf(engine): reduce concurrent host overhead`, resolving the shared
  worker-loop edit by retaining its `stable_decode_epoch` fast path and routing every normal and
  copy-hold decode launch through the new debt-reset helper. Rebasing again over the subsequent
  stop-token-suppression commit applied cleanly. All seven affected admission, runtime-mechanism,
  frontend, sampling, argmax, and speculative-round unit suites pass. On the combined tree, the
  real MTP suite passed k=3, k=5, and prefill-first scheduling before the later adaptive case hit
  an illegal address; a clean build of the unmodified remote tip reproduced that same adaptive
  failure. The combined-tree DFlash suite and clean remote tip also produced the same pre-existing
  chain C=3 parity failures. The complete combined-tree context-checkpoint lifecycle suite passed
  (`ok`). These upstream MTP/DFlash failures block a whole-suite green claim but do not implicate
  the conflict resolution; the rebased commit remains local and unpushed.

## Completion

Keep this file after the prefill implementation closes because it is the requested durable handoff
for the post-prefill review backlog. Remove it only after that backlog has been reviewed with the
maintainer and any selected work has moved into its active authority or implementation.
