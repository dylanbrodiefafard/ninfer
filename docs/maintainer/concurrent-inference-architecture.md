# NInfer Small-Scale Concurrent Inference Architecture

Current product execution supports ordinary decoding and DFlash2 only. MTP execution is
rejected at Engine admission. MTP algorithms and layouts below describe dormant private
machinery, not an available product backend.

This document defines the execution architecture by which NInfer supports a small number of
concurrent requests on a single GPU with a single model instance. A typical `max_concurrency` is
1-8.

The design goal is not to let multiple requests execute in turn, but to let all requests in the
decode phase form one true batched model execution: one model traversal, one CUDA Graph replay,
and one set of batched operators produce results for multiple requests at the same time.

This document defines only decisions that affect module boundaries, scheduling semantics, resource
ownership, or execution correctness. Protocol error formats, allocator implementation, kernel
organization, target-specific profile values, and test plans belong to the corresponding serving,
runtime, operator, and qualification documents.

---

## 1. Scope

### 1.1 Supported workload

- Single GPU, single resident model instance;
- `max_concurrency=C` is fixed at startup, and `C=1..8` is supported;
- `0..C` admitted requests at run time;
- Text and image/video prompts;
- ordinary decoding and engine-wide speculative decoding;
- streaming and non-streaming output;
- prefix reuse, plus an optional completed-bundle pinned-host RAM second tier;
- Different prompt lengths, context lengths, generation limits, and sampling configurations.

### Eight-lane execution geometry

Startup concurrency determines the lane pool, replay/history capacity, graph frontier, and
per-row metadata. Compact batch rows are transient: physical state-slot identities survive
batch growth, cancellation, and compaction. FIFO admission and one packed decode batch per
round boundary apply through C8.

Ordinary decode has at most eight rows. DFlash chain verification packs B sequences of width W
into T=B×W columns: K7 uses W8 and reaches T64; explicit W16 reaches T128. The target aggregate
projection/workspace budget is 128 columns, independent of the 48-column automatic-tree policy.
Existing tiled A8 attention, residual, and SwiGLU projections cover the aggregate budget.
The fused GDN projection/record tile remains limited to 48 columns; larger aggregates use the
materialized FP32 projection path. Packed gating similarly retains its 48-column aggregate
route and uses the per-sequence path above that boundary. These are dispatch boundaries rather
than request-concurrency limits.

Accepted-history and replay descriptors hold eight distinct state slots and preserve FP32
recurrent state, BF16 convolution history, accepted-prefix semantics, and stable graph addresses.
The shared KV budget follows actual allocations and request reservations; eight lanes do not
imply capacity for eight maximum-length contexts. The adaptive draft controller selects its
existing captured arms from measured throughput, without a concurrency-specific forced width.

### 1.2 Non-goals

- General request preemption and swapping. RAM-on settled KV-pressure suspension is bounded by
  §5.8. Spare-lane short-prefill interruption in §7.3 keeps its owner resident.
- Multi-request batched prefill;
- Multi-GPU or distributed inference;
- Priority, tenant QoS, or deadline-aware GPU scheduling;
- General continuous batching for tens to hundreds of requests;
- Dynamically capturing CUDA Graphs for new shapes during serving.

### 1.3 Required sequence-state substrate

The concurrent engine requires the lower layer to provide a concurrent-capable sequence-state
substrate, and it assumes that this contract is already satisfied. The substrate must provide:

- Shared, block-addressable growing-context storage that does not depend on a slot or on a
  per-request physically contiguous range;
- Fixed-size recurrent/backend state allocated per sequence;
- Unified reserve, commit, truncate, and release semantics for growing and fixed state;
- Pin, ownership-transfer, and eviction semantics for retained state;
- An opaque per-sequence state view that attention and other model operators can consume directly;
- State handles and a logical valid frontier that are stable during one GPU execution unit.

This document defines how the concurrency layer reserves, holds, batches, and transfers this
state. It does not define the substrate's page geometry, physical slab layout, allocator,
block-table representation, or page-aware operator implementation. The active contract of the
growing KV substrate is defined by [Paged KV Context Store](paged-kv-cache.md); this document
consumes only its allocation, reservation, and execution-view semantics.

---

## 2. Core invariants

### 2.1 Maximal batched decode

At any round boundary, if there are `B>0` decode-ready requests, the next decode round must
include all `B` requests. Splitting them into multiple request-local forwards is forbidden, and
so is executing them one after another in order to reuse the single-request path.

### 2.2 Boundary-only membership

A request joins or leaves active execution only at a boundary after a GPU round or prefill chunk
completes. While GPU work is in flight, slot binding, batch membership, and sequence-state
ownership do not change.

### 2.3 Fixed slots, shared state memory

`max_concurrency` fixes the number of control slots and does not divide KV/context memory evenly
among slots. KV cache, GDN state, and speculative-backend state are provided by the
Sequence-State Store of §1.3. A slot holds only an allocation handle and a reservation, and does
not own a static context partition.

The Engine configuration must be able to hold `C` active fixed-size sequence states, `C` rows of
execution metadata, and one shared executor workspace at the same time; otherwise that
`max_concurrency` is invalid. Retained state stays only in the fixed backing of a currently idle
lane and occupies its actually mapped growing state; a new admission can claim or evict it, so it
cannot lower the guarantee for `C` active sequences. Growing context is still shared capacity, so
there is no promise that any `C` long contexts can be admitted at the same time.

### 2.4 Admission guarantees completion capacity

With RAM off, admission reserves the complete prompt, declared output bound and transient growth.
With RAM on, admission funds the full prompt and a useful output window plus either a joint
completion claim or secured live-image credit (§5.8). A short-prefill interruption preserves the
owner's lane and reservations. Output limits are never reduced by a reservation window.

### 2.5 Executing and suspended prefill owners

At most one request executes prefill/finalization at a time. One additional text owner may
be suspended at a completed chunk boundary while a short request prefills on a spare lane.
Both retain their stable lane, KV/GDN/checkpoint allocations and request-local cursor; a
suspended owner is not decode-ready. No transient-bearing or media owner is suspended.
The suspended owner resumes before ordinary admission or recovery and must complete another
nonzero chunk before another short interruption. The spare-lane path performs no eviction,
active-state offload or host restore; otherwise ordinary admission waits for the owner.

The default remaining-work policy uses prefill-first below 4096 uncached tokens and
W4096/rounds12 at or above 4096. Remaining work is `prompt_tokens - cursor`, not total
conversation length or the best pre-admission reuse estimate. Section 7.3 defines cadence,
explicit tuning and the protected spare-lane admission exception.

### 2.6 Single GPU execution owner

One GPU executor serially submits all prefill chunks and decode rounds, and is the only modifier
of active slots, sequence state, and the next batch. CPU preparation, request ingress, and output
I/O may run in parallel, but they cannot directly advance model state.

The D2H/H2D transfers of the RAM second tier go through an independent copy engine
(`DeviceContext::copy_stream`). They are not a GPU scheduling unit and do not occupy the compute
owner. A copy may overlap with a legal compute unit of **another** lane; pages that the current
PrefillChunk / DecodeRound is reading or writing must not be a copy source or destination. The
MTP/DFlash prefill context-checkpoint freeze is the exception: the D2H of the staging GDN (slot
`C`, disjoint from the lane's current), the staging hidden, and the DFlash staging cyclic state
may overlap with the next prefill chunk of the **same** lane. During the freeze, `C` is borrowed
from the turn-rollback occupant, and the H2D after packing restores the rollback; `occupied=false`
must happen before the clobber D2D. `synchronize()` on `device.stream` drains only compute, not
copy_stream. Checkpoint host images (each lane's rewrite image and the MTP/DFlash
context-checkpoint head pool) are carved at startup from one prefaulted, registered pinned slab
and are owned by the Program; a head returns to the pool after it leaves the queue, and the
subsequent freeze, turn-rollback, and RAM/disk restore reuse the same image and completion event,
so `cudaMallocHost`/`cudaFreeHost` are not called during serving. The pool capacity is one
rollback head per lane, plus the number of ladder heads that all lanes can hold at the same time
within the shared KV capacity (marks above `--max-context` are unreachable); when the pool is
exhausted, an optional capture or restored head is skipped. Waiting for the previous DMA/host-copy
owner before reusing a head is always a stream wait (copy_stream or the compute stream waits on
the head fence); the scheduler thread does not host-wait.

### 2.7 Bounded ingress and output

Both the Engine and the HTTP server limit the number of generation request lifetimes to
`max_concurrency + max_pending_requests`. Each request also has a finite input size, a finite
effective output token bound, and an absolute pending deadline; media preparation is serialized by
a single permit. Sustained ingress can therefore only fill a finite set of request records and
cannot build an unbounded wait queue.

The total upper bound of owning results comes from "finite number of requests × finite output per
request"; result/event storage belongs to the respective request and continues to count toward
request lifetime capacity until the response consumer releases it.

### 2.8 Per-request semantic isolation

Sampling, RNG, stop conditions, generation limit, usage, and output state belong to the request.
They must not depend on which slot or compact batch row the request currently occupies, and they
must not affect other requests in the same batch.

### 2.9 Protected, resource-aware admission

Prepared request order is the fairness baseline for admission, but a FIFO head that temporarily
cannot obtain its full resource commitment must not unconditionally block all later requests. The
Scheduler may backfill the currently uncommitted lane/KV capacity with later requests, but it must
first establish a unique protection epoch for the oldest blocked head; later admission cannot
break the head's per-pool resource feasibility after frozen incumbent releases, and cannot extend
temporal borrowing indefinitely through sustained ingress.

Backfill changes only the admission order of waiting requests. It does not take resources that an
active request has committed but not yet materialized, does not produce partial admission, does
not preempt requests that are already admitted, and does not establish a second decode priority.
All backfilled decode-ready requests still enter the same maximal compact batch.

### 2.10 Constrained output and bounded recovery

The Qwen frontend owns compiled tool/response schemas and EBNF constraints, transactional grammar
state, and typed completed calls. `PromptOptions::output_constraint` constrains answer content
without injecting schema text into the prompt. Response schemas use JSON Schema defaults and
fixed property order; unsupported assertions fail preparation. Reasoning remains unconstrained
until `</think>`. Constrained content preserves literal special-token bytes and EBNF whitespace;
ignored model-stop bytes retain their ordinary publication policy. Tool definitions and response
constraints cannot be combined. Tool-call parsing is enabled only for tool grammars. Program owns planner-accounted device masks and pinned host exchange
storage. Ordinary sampling uses one mask per compact row; speculative verification uses
one mask per row and actual chain/tree node. CUDA Graph host nodes fork grammar state
after draft IDs/parents have arrived on the host, then upload masks for target sampling.
Proposal sampling does not advance the grammar. Only the accepted publication transaction
advances committed grammar; callback failures are surfaced before publication.

Every MTP and DFlash2 verify round selects its schedule from stable session mask-policy
presence at the synchronized round boundary, not from the matcher's current token
phase, draft length or startup concurrency. A compact batch with no grammar, replay or reasoning-budget mask row skips
the exchange entirely and samples from its ingress configs. A batch with any mask-policy
row forks the exchange from completed proposal production onto a dedicated
host-callback stream; target verification proceeds and joins only before target
sampling consumes masks/configs. Uploads cover only the batch's own rows. Fixed-pitch
device staging uses update-safe one-dimensional row copies; pinned host transfers keep
fixed addresses. DeviceContext's complete drain includes the callback stream, and
Program drains before destroying borrowed sessions, including an exception after a
fork but before its join. DFlash V1 keeps one exchanging schedule.

Runtime may withhold a completed repeated call, or pause persistent thinking after a
committed decode round, and rebuild that request on its existing
lane. This is neither preemption nor another admission. The conservative family detector
requires unchanged associated results and repeated reasoning in text-only history.
Reasoning-only retries require three non-overlapping occurrences of the same exact 256-token
passage within the current generated attempt. A rolling hash locates candidates, which
must pass exact token comparison; only disjoint occurrences increment the count. Different
separations are allowed, so changing a loop's period does not hide persistent repetition.
Repeated passage coverage must also contain at least 4,096 distinct redundant tokens;
the incremental interval union counts overlapping windows only once. This is evidence
of repeated work, not a generated-reasoning length limit.
The state resets on retry and cannot use original prompt tokens or another attempt.
There is no reasoning length limit. Runtime enables recovery for p-less, non-raw text-only
thinking input; reasoning retries additionally require positive temperature. The family
retains the owning original input. The retry splices onto the resident prompt, keeps
historical reasoning, closes the open think turn, and supplies an explicitly labeled
system notice or rejected-call feedback. It does not invent a call, result, or user turn.
Planning reserves service work for up to two retries, each within the original
context entitlement (consumed completion space can accommodate the notice). A
repair must fit the original resource commitment and preserve the remaining
output budget; otherwise recovery fails explicitly for that request.

At the committed round boundary the retry leaves the decode-ready set. It stays on its
admitted lane, closes the open think turn, and splices the notice or rejected-call
feedback onto the resident prompt, including historical reasoning. Only the failed
generation is omitted. A ready checkpoint that is a prefix is restored and the suffix is
prefilled. The retry does not take a new admission and does not trim at an arbitrary
token. Retry queue entries are bounded by admitted slots and checked against current lane
ownership. Other lanes still decode outside the exclusive retry prefill; a resident-miss
host restore holds its lane in copy-hold like an admission, so they keep decoding while the
RAM or disk image copies, and a failed host restore cold-prefills the same lane.

Calls remain unpublished until recovery accepts the terminal result. Already streamed prose
and reasoning cannot be retracted. All generated attempts consume the original token budget;
additional prefill tokens/time are reported separately, without changing original prompt
usage. At most two repair attempts are allowed. Exhaustion is a request error, not EOS
injection, release of a known repeated call, an engine shutdown, or a promise of useful
progress. Cancellation uses the normal boundary-owned lane cleanup.

---

#### Reasoning-budget ownership

OutputSession owns the cumulative reasoning-body token ledger and transactional preview masks.
A speculative path charges only accepted IDs, and mask construction forks each hypothesis from
the committed phase. At exhaustion the canonical close is an explicit token-domain policy;
its unmasked target logprob is retained separately from channel confidence records. Runtime
preserves this OutputSession through KV suspension and records the forced token against the
cumulative generated-token index. The production policy closes thinking without a clean retry;
explicit stops, cancellation and output/context ceilings remain authoritative. The internal
Retry comparator is available to focused policy qualification and has no request-visible selector.

Replay recovery withholds a proved initial answer replay and keeps generated IDs charged.
A repaired close cannot be replay evidence for another repair. Replacement prose is withheld
until it is accepted, and complete tool calls retain terminal publication. Exact repeated-tool
proof and canonical feedback use the complete ordered batch, so added novel calls remain valid.

## 3. Overall architecture

```text
 clients / CLI / OpenAI / Anthropic
                  │
                  ▼
┌──────────────── Server Frontend ─────────────────────┐
│ validate · bounded CPU preparation · ordered pending │
│ queue · protected-head resource backfill             │
│ finite count/per-request bytes/deadline · responses  │
└───────────────────┬──────────────────────────────────┘
                    │ admission at a boundary
                    ▼
┌──────────────────── GPU Executor ────────────────────┐
│ Boundary Coordinator · Scheduler · Slot Table[C]     │
│                         │                            │
│          ┌──────────────┴──────────────┐             │
│          ▼                             ▼             │
│  PrefillChunk(one sequence)   RoundMembership[B]    │
│          │                             │             │
│          │                    Target Batch Assembler │
│          │                             │             │
│          └───────────┬─────────────────┘             │
│                      ▼                               │
│       shared Model Runtime + DecodeBatchFrame[C]     │
│       weights · workspace · graph families           │
│                      │                               │
│                      ▼                               │
│               per-row round results                  │
└───────────────┬───────────────────┬──────────────────┘
                │                   │ committed output
                ▼                   ▼
┌─ Sequence-State Store prerequisite ─┐  Server Frontend
│ Text KV                            │  / async output
│ GDN / recurrent state             │
│ speculative-backend state         │
│ state entitlements                │
│ optional retained prefixes        │
└──────────────────────────────────────┘
```

The responsibilities of the components are as follows:

| Component | Responsibility |
|---|---|
| Server Frontend | Request validation, bounded CPU preparation/pending work, cancellation input, and response I/O |
| GPU Executor | Admission, boundary processing, state commits, and all GPU submission |
| Slot Table | Holds the stable control state of admitted requests |
| Scheduler | Performs protected-head admission at a boundary, and selects the next `PrefillChunk` or the complete active `DecodeRound` |
| Target Batch Assembler | Converts round membership into the typed controls and state selectors the target needs |
| Model Runtime | Holds the single resident model, shared execution memory, and graph assets, and executes the whole-batch schedule |
| DecodeBatchFrame | One address-stable round staging area with maximum capacity `C`; each round uses only the exact-`B` prefix |
| Sequence-State Store | Prerequisite storage substrate; owns per-sequence model state and capacity |

These are responsibility and ownership boundaries; they do not require C++ classes with the same
names.

All product entrypoints submit requests to the same GPU Executor; a caller or connection thread
does not drive the model loop directly, nor does it own a Model Runtime of its own. The `C`
concurrent requests share the same weights, workspace, round frame, and graph families, rather
than constructing `C` single-request Programs.

Blocking `Engine::generate` may be kept as a product facade, but its semantics are to submit an
owning request record and wait for that request to complete, not to run the complete generation
loop inside the caller thread. One Engine-owned GPU worker executes the global boundary loop;
concurrent callers wait for their own owning results, and, for streaming requests, for their
output queues. Delivery intent is fixed before queue membership: in both modes the GPU worker
appends committed output to the owning result, and only `Streaming` mode writes per-round output
events; `TerminalOnly` wakes the consumer only on completion or a recovery diagnostic. The GPU
worker does not call network writes or a potentially blocking `OutputSink`.

---

## 4. Request and slot model

### 4.1 Request lifecycle

```text
RECEIVED
   │ validation and bounded pending capacity
   ▼
WAITING
   │ bounded CPU preparation, then boundary admission:
   │ slot/lane + state entitlement
   ▼
PREFILL
   │ zero or more suffix chunks, then one finalization unit
   │ exact retained hit skips suffix work, not finalization
   ├── terminal ──────────────────────────────────► MODEL_FINISHED
   ▼
DECODE_READY
   │ participates in successive batched rounds
   ├────────────────────────────────────────────► DECODE_READY
   │ EOS / stop / limit
   ▼
MODEL_FINISHED
   │ response may continue draining
   ▼
RESPONSE_FINISHED
```

Cancellation or failure can terminate a request from any phase. `WAITING` includes both bounded
CPU preparation and waiting for GPU admission; these are server-side phases and do not need to be
expanded into more model-execution states.

`DECODE_READY` is a stable state: the request already owns the complete committed model frontier
and the single current token/anchor needed by the next decode round, and its sampling, RNG, and
published output are consistent with that frontier. Whether it belongs to the current in-flight
round is recorded by that round and is no longer modeled as a long-lived request state.

Intermediate prefill chunks only advance prompt state. The finalization unit performs target
output selection and establishes the decode anchor; if a terminal condition is hit immediately,
the request does not enter a decode batch. When the retained current frontier exactly covers the
complete prompt, suffix prefill may be skipped, but the next anchor is still produced from the
retained tail state through the same finalization unit, and under MTP the exact-hit
bridge/proposal is completed. On a hit on a saved rewrite checkpoint, the new suffix is prefilled
from that checkpoint. A KV match alone is not sufficient to enter `DECODE_READY`.

### 4.2 Slot

A slot is a stable control position leased to one admitted request. It stores or references:

- Request identity and lifecycle state;
- Sampling, RNG, stop, and generation-limit state;
- Per-request output session, usage, timing, and speculative statistics;
- A handle to the `SequenceState` in the shared pool;
- Prompt/decode progress required by the scheduler;
- Cancellation and terminal flags.

A slot does not own:

- A fixed proportion of context capacity;
- A permanent batch row;
- Shared model workspace;
- Response buffers after model completion;
- Logical ownership of the retained `SequenceState` after model completion.

After a request reaches `MODEL_FINISHED`, the model result is handed to the response path, and the
slot can be reused at the current or a later boundary. Network completion is not part of the slot
lifetime. The slot index is also not an external request identity; the cancellation or completion
of an old request cannot act on a later occupant.

The current runtime maps a control slot one-to-one to a physical execution lane, but the slot, the
external request identity, and the compact batch row remain three distinct identities.
Fixed-size device state can be reserved per lane and rebuilt when the lane is occupied by a new
request. After a request completes, the active slot is released immediately; if a prefix is
retained, the `SequenceState` can remain in the backing of that idle lane, but no active request
control exists any more and it does not count toward active concurrency.

### 4.3 SequenceState

`SequenceState` is one continuable model state defined by the target. An occupied slot has sole
write authority over it, and it contains or references at least:

- Main/backend KV allocations and their committed frontiers;
- Linear Attention and other fixed model-state allocations;
- The target decode cursor, including the current anchor and position/RoPE progress;
- Hidden/checkpoint state needed for continuation;
- Prefix identity and target-defined reusable checkpoints.

`SequenceState` does not contain stop/output/transport state, and does not own a batch row, round
activations, logits, shared workspace, or a graph. The current fixed-state backing is
lane-affine: ordinary consuming reuse stays on the same lane, and the planner searches all free
lanes for a reusable continuation and, once one is selected, establishes the new request control
on the same lane; when active capacity is needed, retained state on other free lanes may be
evicted first. The sampling, RNG, stop, and output state of a new request are always recreated.
Disposable forks copy into an independently owned destination lane under
the source-preserving transaction described in §6.4.

For Qwen3.6, the lane is the sole locator of Linear Attention state. With `C=max_concurrency`, the
shared pool always uses `[0,C)` as the current committed state of each lane; an MTP or DFlash
engine additionally reserves slot `C` as the Engine-wide GDN: the default hot occupant is
turn-rollback (the append occupy pins current+`tail_hidden` at the previously completed `E`
before suffix prefill), which is borrowed by the ladder freeze and then reinstalled; a rewrite
checkpoint capture borrows it and does not reinstall it (so a freeze within the same step is also
not reinstalled), and the subsequent rollback restore goes through the host image H2D. One slot
selects the convolution history and recurrent state of all GDN layers at once.

A rewrite checkpoint does not occupy a device slot: each lane owns one pinned host image (the
host-image layout of the GDN slot and the DFlash cyclic local lane), allocated at startup with the
Program and reused per lane afterwards. When a prefill chunk ends exactly on a checkpoint
frontier, MTP/DFlash D2D-copies current (and the DFlash local lane) into the Engine-wide staging
slot on the compute stream, and then copy_stream drains the staging into that image, overlapping
with subsequent prefill/decode; while the staging still holds that lane's current-generation
image, a rewrite restore does a D2D from staging back to current, and otherwise waits for the
image copy and then does an H2D. Another staging writer, or a RAM/SSD tier restore (which replaces
the image on the host), ends that staging hit. Without MTP/DFlash there is no staging slot, and
the capture does a D2H on the compute stream. Between the RAM/SSD tier and this image only host
copies are made. The checkpoint hidden `[hidden,1]` remains on the device. Per-lane device fixed
state therefore consists only of the current GDN and the DFlash current local state, and does not
double with rewrite checkpoints. A decode round does not maintain, in `SequenceState`, a state
selector that varies with the speculative position.

The Program initializes the text and speculative backend pools' physical KV payload planes and
every DFlash local K/V lane's complete padded payload once at startup, even when CUDA Graph capture
is disabled. Tier images and disposable forks copy whole pages and complete fixed images; unused
page tails and cyclic positions therefore contain defined bytes without extending valid frontiers
or the live attention window. Block tables and unrelated arena storage are outside this initialization.
The Program also initializes each complete ordinary, MTP, and DFlash decode egress frame once, so
full-frame host transfers include defined inactive entries and padding.

### 4.4 Batch row

A batch row exists only for one decode round. At each boundary, the Target Batch Assembler
re-establishes a compact mapping:

```text
batch row -> slot -> sequence-state handle
```

For example:

```text
slots:          [0] [1] [2] [3]
decode-ready:    Y   N   Y   Y

logical rows:   [0] [1] [2]
row -> slot:     0   2   3
logical B:       3
```

An empty slot does not enter the model batch, so slot reuse or holes do not require a different
execution model.

---
## 5. Request ingress and admission

### 5.1 Bounded ingress

Before expensive preparation, the Server Frontend validates request syntax, model capability, input
size, and the finite generation bound. Only requests that pass validation can obtain bounded CPU
preparation and pending capacity.

The host side sets finite limits on the following resources:

- the number of generation-request lifetimes, fixed at `C + max_pending_requests`;
- body/media bytes and media item count per request;
- CPU preparation work that holds request records at the same time;
- one serial media-input preparation permit;
- the longest time from a request obtaining lifetime capacity to its admission;
- the effective output token bound per request.

When the request lifetime count is full, a new generation request is rejected immediately as
overload. There is no additional unbounded overflow queue; the per-request input limits together with
the finite outstanding count bound host input memory, instead of a separate aggregate byte allocator.

CPU preparation produces an owning, immutable prompt representation. Prepared requests enter the
Engine ordered admission queue in preparation-completion order, so a request waiting for the media
permit does not hold the first position of the prepared queue. The queue timeout starts when a request
first obtains request lifetime capacity, so it includes media acquisition, prompt preparation, and
Engine admission waiting.

This order is a fairness baseline, not an absolute head-of-line barrier. Only the protected-head
backfill defined in §5.3–§5.6 can let a later request be admitted first; CPU preparation completion,
prefix reuse, or the request content itself does not create any other implicit priority.

### 5.2 Authoritative admission feasibility

The full-lifetime donor/backfill rules in §§5.2–5.7 apply with RAM off. RAM-on uses the same
target-authoritative pool vectors with the window and progress rules in §5.8.

The Scheduler does not re-derive whether a request can enter from token counts or GPU memory bytes. The
target/resource layer gives each prepared request an authoritative entitlement vector, already rounded
to the physical granularity of each pool. Logically it contains at least:

```text
E(request) = {
    active lane:                  1,
    Main Text page groups:       E_main,
    selected-backend page groups: E_backend,
    other request-time units:    target-defined, if any
}
```

Each dimension may be compared only with capacity of the same kind; Main/backend pages cannot be summed
by bytes or borrowed from one another, and tail-page slack of an allocation is not usable capacity. The
current fixed recurrent/backend backing is established for `C` copies and transfers ownership with the
physical lane, so it is expressed by the `active lane` dimension; the GPU memory bytes of fixed backing
are not counted again in the dynamic entitlement. Graph, workspace, round frame, and weights are
engine-fixed resources; owning result capacity is already obtained at ingress and is likewise not part
of this vector.

A candidate has current admission feasibility only if all of the following hold:

1. a control lane is available;
2. the request is valid for the fixed model, frontend, and selected execution backend;
3. after active-priority retained eviction, every entitlement dimension can commit the request's
   full-lifetime requirement;
4. Ordinary admission requires no other prefill owner or unfinished copy-hold. The only
   exception is the FIFO-head spare-lane short-prefill path in §7.3, which preserves the
   existing owner's lane and resources. Admission establishes suffix-prefill or exact-hit
   finalization ownership, or parks a copy-hold until RAM copies permit `start_prefill_lane`.

The selected backend's differences appear only in the typed entitlement given by the target; the
Scheduler does not establish different queue policies for ordinary, MTP, or DFlash. Even though startup
sizing guarantees that the backend does not become normal backpressure before the Main pool, admission
still consumes the full authoritative vector, and an early backend failure is a sizing/accounting
invariant violation.

Admission is one atomic boundary transaction: the prepared entry, the optional retained-state claim, the
slot/lane, and all fixed/growing state entitlements are either obtained together and published as an
admitted request, or the request-visible ownership is left unchanged. The feasibility probe and
protected-head shadow accounting allocate no pages, pin no retained entries, and give a queued request
no partial ownership; the same authoritative transaction still re-checks and commits at the end.

A request must first pass exclusive feasibility: with all evictable retained state evicted and no other
active request, it can occupy the Engine's admission exclusively. A request that cannot be held
exclusively is rejected permanently and cannot become a protected head; if there is no active request
and the queue head still cannot enter, the cause can only be a permanent rejection or a
resource-accounting failure, and the Engine must not wait idle.

### 5.3 Ordered queue and protected head

Each admission opportunity first checks the oldest prepared entry:

```text
oldest head is feasible now
    -> admit head

oldest head is permanently infeasible / cancelled / expired
    -> remove it, then reconsider the new oldest entry

oldest head is exclusively feasible but blocked by active lane/entitlement
    -> protect this head and consider qualified backfill
```

If the head's full lane/entitlement is already feasible and only the existing one-prefill-owner rule
prevents admission, it remains an ordinary waiting head, establishes no protection epoch, and does not
scan later candidates; at the next boundary with admission budget after the prefill owner clears, the
head is retried first. Protection is established only when there is no prefill owner and the head
genuinely needs an existing active request to release a lane or entitlement, so the frozen donor set is
non-empty.
An eligible short FIFO head may instead take the spare-lane exception in §7.3; it still
cannot bypass an earlier live head or establish a backfill protection epoch while interrupting.

Only the oldest head is protected at any one time. Protection is bound to a stable request generation
identity, not to a reclaimable lane, slot index, or compact batch row. It is only a Scheduler shadow
state and does not pre-claim real pages, a lane, or a retained continuation for the waiting request.

The protection lifecycle is:

```text
UNPROTECTED
    │ oldest head is transiently blocked at an admission boundary
    ▼
PROTECTED_OPEN
    │ protected resource opportunity has matured,
    │ but head is still blocked by a temporal borrower/prefill owner
    ▼
PROTECTED_DRAIN
```

`PROTECTED_OPEN` allows the backfill defined in §5.5. `PROTECTED_DRAIN` forbids all later/backfill
admission, advances only the already admitted bounded prefill/decode work, and keeps first-retrying the
protected head at every boundary that has admission budget. Successful admission of the head,
cancellation, queue timeout, or permanent rejection clears the current protection; the next oldest
request can then establish its own epoch, but does not inherit the old frontier, work credit, or shadow
ledger. Multiple waiting large requests do not establish nested reservations.

The queue deadline is not reset by protection or by being bypassed. Backfill only guarantees that, under
normal Engine progress, continuing later ingress does not cause unbounded starvation; it does not
promise that the protected request is admitted before its configured queue timeout.

### 5.4 Frozen release frontier and persistent-safe backfill

When protection is established, let:

- `K` be the per-dimension usable capacity;
- `H` be the protected head;
- `A` be the frozen incumbent identities already admitted at this moment;
- `D` be the minimal ordered prefix of `A` that is expected to complete earliest and whose release is
  sufficient for `H` to be admitted.

The Scheduler orders `A` by the remaining-service projection of §5.5 and adds donors one at a time until
the following relation holds in every dimension:

```text
sum(E(a), a in A without D) + E(H) <= K
```

`D` defines a set of incumbent completion/release events, not an externally promised wall-clock ETA.
Once established it can only shrink because of early incumbent completion/cancellation; later admitted
backfill can never become a donor, and cannot swap the frontier for a later active request. If the early
release of any other incumbent already makes `H` feasible, the Scheduler may admit `H` earlier than the
frozen frontier.

Let `P` be the backfill requests that are still active and classified as persistent-safe. They must
always satisfy the cumulative shadow invariant:

```text
sum(E(a), a in surviving members of A without D)
+ sum(E(p), p in P)
+ E(H)
<= K
```

A candidate can become a persistent-safe backfill only if it both `fits-now` and still satisfies this
per-dimension relation after being added. All still-surviving `P` must be accounted cumulatively;
multiple candidates cannot each be compared against the same initial surplus. When a persistent-safe
request completes it is removed from the ledger, and its surplus can be used by another qualified
request; but at any resource-release boundary that also has admission budget, `H` must be retried first
before the surplus is reconsidered for reuse; otherwise only the release/maturity is recorded. If decode
membership is still alive, the retry happens first after the next maximal decode; if the last donor has
left the queue, the debt is cleared and a retry can happen immediately.

This invariant guarantees that even if all `P` are still active when the donor frontier arrives, `H`
does not lose the lane, Main pages, or backend pages it needs because of them. It does not guarantee
that `H`'s wall time is unaffected; the bounded prefill of backfill and the larger decode batch still
change incumbent round latency.

In protection accounting, retained state is an evictable cache, not queued ownership. `E(H)` uses the
full entitlement of a cold admission; no matching lane/checkpoint is pinned for `H`. Retained state
produced after an incumbent or backfill completes must, when an active admission needs it, be evicted at
the same boundary under active-priority semantics, so its full active entitlement can be regarded as
releasable. Non-evictable engine-fixed occupancy belongs to the baseline outside `K` and cannot be
disguised as a donor release.

### 5.5 Service projection and bounded temporal borrowing

Total entitlement determines resource feasibility but cannot by itself represent residence time. The
Scheduler uses a finite, monotonic service projection expressed in uniform positive-integer
scheduling-work quanta to identify relatively short work:

```text
service work = known Text/Vision suffix-prefill and finalization work
             + effective remaining output reservation expressed as decode work
```

The output part uses the declared finite effective output bound; it does not predict prompt content,
reasoning difficulty, or the actual EOS. The prefill part uses the already bounded target
scheduling-unit profiles; a short output cannot offset arbitrarily long Text/Vision prefill. The current
Qwen3.6 profile uses the finite upper bound of externally scheduled prefill/finalization steps as the
prefill quanta and counts each effective remaining output token as one decode quantum; the known prefill
splits caused by prompt snapshots and Vision items are counted at planning time. The selected speculative
backend can influence the target's uniform work projection, but does not create MTP/DFlash policy
branches in the Scheduler.

The projection serves only the following two purposes: choosing the frozen incumbent donor order, and
constraining temporal borrowing. It is not a completion ETA, a memory-safety basis, or a fairness proof.
An active request can reach EOS early on the next round, and speculative acceptance and batch round
latency also vary; estimation errors must be absorbed by the protection state rather than by assuming
the prediction is accurate.

Let `W_frontier` be the projected remaining service distance for the frozen donors, advancing in
parallel, to reach the required release event when the epoch is established. Protection initializes a
positive, finite, non-replenished temporal credit with it:

```text
credit_initial = W_frontier
credit_after_admission = credit_before_admission - service_work(candidate)
```

The service work of every admissible request is at least one quantum; even an exact retained hit with no
suffix tokens still has finalization work. The credit also uses the same discrete unit, so each temporal
admission deducts at least one unit, and there is no sequence that decreases forever without being
exhausted. A candidate that cannot satisfy the persistent-safe invariant but whose current full
entitlement `fits-now` may temporarily borrow the critical capacity that `H` needs after the frontier
only if all of the following hold:

1. the candidate's full service work does not exceed the current projected remaining distance, and it is
   expected to complete before the frozen frontier;
2. that service work does not exceed the remaining temporal credit;
3. after admission the candidate's full work is deducted from the credit, and it is not refunded if the
   request completes early;
4. temporal admission stops immediately once the protected opportunity has matured.

Such a request is called a temporal backfill; its full active footprint still enters the current
authoritative resource accounting, but not the persistent-safe ledger, because the policy expects it to
release before the frontier. The work credit prevents a 32K long request and a 2K short request from
being treated as the same single bypass, and also limits the cumulative service work that later requests
can borrow within one epoch. Credit only decreases and never increases; a new arrival cannot extend the
epoch.

The Scheduler scans the bounded pending snapshot in prepared order and selects the oldest qualified
candidate. Qualification may skip an earlier entry that is unsafe or not short enough, but does not
permanently apply SJF, best-fit, or knapsack to the whole queue. If all requests declare the same huge
default output bound, the Scheduler has no reliable evidence for judging actual difficulty; temporal
qualification is then naturally conservative, and provable persistent-safe backfill can still be used;
it must not guess from prompt content.

### 5.6 Admission burst and progress rules

Ordinary admission requires:

```text
no prefill owner
and no copy-hold admission
and (no decode-ready request or decode-admission budget > 0)
```

Section 7.3 permits one short FIFO head to use a resource-feasible spare lane while a large
text owner is suspended. That bind spends the same budget and never evicts retained state.

A copy-hold, like a prefill owner, holds the admission/prefill opportunity exclusively: it is not a GPU
unit, but a second request must not be admitted before the D2H/H2D copies complete and
`start_prefill_lane` is called. A held lane is neither decode-ready nor a prefill owner.

When the first decode-ready membership appears, the Scheduler freezes this `decode-admission budget` to
the current number of free lanes. Each successful bind of a request deducts one budget, and it is not
refunded even if the request terminates at the first token of its final prefill, or is later cancelled or
fails. When the budget is zero, one `DecodeRound` containing all decode-ready rows must run first; after
that round completes, the next membership establishes a new budget from the free lanes at that time.
When there is no decode-ready request, no progress debt is needed, and the Scheduler can keep
processing finite pending snapshots.

One admission attempt successfully admits at most one request. Cleanup of cancellation, timeout, and
permanent-invalid entries does not count as admission; the Scheduler may continue looking for a new
head/candidate in the same frozen pending snapshot, but a candidate that fails under the same resource
state is not retried repeatedly at that boundary.

The order of each boundary opportunity is fixed as:

```text
1. resolve completed GPU unit
2. finish/cancel active requests and release-or-retain their state
3. remove visible pending cancellation, timeout and permanent failures
4. if admission is eligible, retry the protected head or the oldest FIFO head
5. on the same turn, if an open protected head remains blocked,
   admit at most one qualified backfill
6. choose and launch one next GPU unit
```

One admission attempt consumes only the one finite pending snapshot obtained within the boundary.
Cleaning up cancelled, expired, or permanent-invalid entries in the snapshot may continue to check the
new head/candidates of the same snapshot, but cannot take a new snapshot; requests that arrive after the
boundary are left to the next GPU-unit or idle-control boundary.

If exact current accounting already makes the protected head feasible, it is always admitted before any
later request. If the frozen donors have already released, or the current non-temporal shadow resources
are already sufficient to hold `H`, but `H` is still blocked by a temporal borrower or the existing
prefill owner, protection immediately enters `PROTECTED_DRAIN`. Drain does not preempt the borrower; it
forbids later/backfill admission, but the protected head is still retried first at every eligible
boundary. A persistent-safe borrower that is still in prefill may also briefly block head admission.
Its work is finite; the only nested admission is the bounded short-head
spare-lane exception, which must allow another nonzero owner chunk before repeating.

Completion/cancellation of a frontier member counts as a release only after its state is actually
released at a boundary. If that boundary has admission budget, the head is retried first immediately;
otherwise maturity is updated. As long as decode membership is still alive, the retry happens first
after the next maximal decode; if the last donor leaves the queue, the debt is cleared and a retry can
happen immediately. An active request's used/reserved transition does not change its entitlement; a
future reservation that has not yet been materialized is not tail surplus. Backfill completion retention
also cannot override head-first active-priority eviction.

Protection consumes the GPU progress policy:

- after a decode-ready donor first appears, one burst can successfully admit at most `C-1` more times,
  and each bind permanently deducts from this burst budget;
- each successful decode/speculative round of every non-terminal row commits at least one output token,
  otherwise it is directly terminal;
- each `PrefillChunk` advances a positive, finite amount of prompt work, or completes finalization;
- each request has a finite output bound, and each GPU unit has a bounded execution extent.

Therefore a surviving frozen donor takes part in one maximal decode after at most `C-1` intervening
admissions, even if all these admissions terminate at the prefill token and release their lanes
immediately. Frozen donors release within a finite number of rounds. The temporal admissions that can
happen before the frontier are also finite; a miss after the frontier matures enters drain, and the
already admitted borrowers release resources after eventual completion/cancellation. Continuing ingress
cannot roll the donor frontier backward, and cannot form a resource deadlock. This guarantee is eventual
progress, not zero wall-time interference or an admission ETA.

### 5.7 Waiting outcomes and sustained ingress

Admission outcomes are classified as:

| Condition | Outcome |
|---|---|
| request is invalid or exceeds a semantic limit | permanent rejection |
| cannot be held exclusively even after active-priority eviction | rejected for the current Engine configuration |
| request lifetime capacity is full | ingress rejects as overload |
| can be held exclusively, but the current slot/resource/prefill owner does not allow admission | protected waiting or ordinary waiting |
| a qualified later request uses the currently available backfill opportunity | admitted before the blocked head |
| queue timeout expires before admission | ends with queue-timeout |
| Engine is stopping or unavailable | rejected as unavailable |

Only an admitted request gets a GPU completion commitment. A queued request has only a bounded waiting
commitment, with no execution commitment or admission ETA. It waits until the first of: boundary
admission, cancellation/disconnect, queue timeout, or Engine shutdown/failure.

Under sustained ingress, once the finite queue is full, subsequent requests are rejected immediately. A
later arrival can backfill within an open protection epoch through the same persistent-safe/temporal
qualification, but cannot change the protected identity, increase the temporal credit, become a current
frontier donor, or increase the workload/resource commitment of any admitted request.

---

### 5.8 RAM-on output windows and live continuation

Enabling `--kv-ram-capacity` selects output windows of at most 8192 tokens; no separate toggle is
required. The full prompt and target-declared speculative headroom must fit before prefill. Eight
chats with 32768-token output limits initially reserve at most 65536 output tokens, plus prompt
and rounded backend headroom, rather than 262144 output tokens. Lifetime output/context limits
remain unchanged. Teacher-forced scoring reserves its complete extent because it has no scheduler
to extend windows.

At each settled boundary, the executor handles cancellation and transfer readiness, retries window
growth, then considers resume and ingress. Decode round budgets stop at both the funded window and
the lifetime limit. Exhausting a window blocks that lane without finishing the request; other lanes
continue. Target planning owns every rounded Main/backend resource vector and mapped-page floor.
Before suspension, unused future window entitlement may shrink to satisfy a concrete transaction,
retaining all represented pages and at least one useful next output token.

Each admitted identity has either secured contiguous RAM credit through its current window end or
a completion claim. The sum of completion members' immutable target-produced lifetime envelopes
fits every GPU pool and lane axis, including enabled native recovery suffix growth. These claims
prove future progress and do not allocate lifetime KV. Backed peers can evacuate so the cohort
finishes without another RAM allocation. A cohort member can become a victim only after funding
its image and exchanging its claim with a feasible progressing successor. Restore holds its
completion claim before transfer and keeps image backing until all readers finish. Queued
replacement claims remain held through capture and admission.

A single blocked lane waits while peers advance unless a funded queued replacement can use its
release. Two blocked lanes, no executable work, or the finite bypass limit opens sticky drain.
Pressure plans prove the complete post-transition resource vectors and select the minimum useful
victim set, favoring blocked lanes and lower transfer cost. Unblocked lanes are not suspended merely
because ingress waits. The first blocked or suspended identity freezes an incumbent epoch with at
most `C-1` fresh admissions, without refunds; only the oldest incumbent's completion or cancellation
resets it. Oldest suspended work precedes fresh ingress. A resumed victim must advance a useful
chunk before resuspension unless no executable work remains. Suspended and transfer-held identities
still count against `C + max_pending_requests` and are exempt from pre-admission queue timeout.

Capture occurs only after output/state commit and GDN history materialization. It preserves exact
KV pages/publications, GDN, hidden/rewrite/checkpoint state, DFlash cyclic state and pending features,
ledger/prefix identity, sampling counters and request controls. GPU ownership survives until the
copy fence settles. Restore binds independent pages on a free physical lane, resets destination GDN
history, rebinds lane-affine sampling pointers and continues without prefill, resampling or output
replay. The request's OutputSession, RNG, budget, recovery and event queues retain logical ownership.
Unrelated lanes can decode during transfers. Cancellation and shutdown fence transfers before
releasing image credits or device state.

Live images are private and never prefix-lookup entries, optional eviction victims or disk spills.
They share the configured RAM arena with completed/system prefixes, and take precedence over those
optional entries. Admission selects reuse before sizing its image credit. If reserving that credit
changes the RAM index, admission reselects the source and funds its actual image requirement at the
same output window before claiming a source or reducing the window. This also applies when a failed
reservation evicts optional entries. Credits include alignment, current/rewrite/checkpoint/backend
state and transfer readers; fragmented free space cannot promise an unallocatable image. Finite
backing may reduce residency when neither a completion claim nor a useful funded image fits. No live request falls
back to disk or re-prefill for pressure. Short-prefill interruptions secure the same claim or credit
and consume the same ingress fairness allowance as ordinary windowed admission.

`RuntimeStats` reports the window ceiling, growth-blocked and suspended counts, extensions and live
suspend/resume totals. `/metrics` and `/metrics.json` expose the same values. These counters describe
scheduling activity, not a throughput improvement.
## 6. Shared context and state memory

This section defines only the resource contract and ownership transitions by which the concurrent engine
uses the Sequence-State Store; the storage substrate itself is provided in advance by §1.3.

### 6.1 State ownership

Each admitted request obtains one sequence-state ownership on the selected free lane. Depending on the
selected model mode, that state contains:

- the Text KV cache;
- GDN convolution/recurrent state;
- speculative-backend persistent state;
- position and model-continuation metadata.

Growing KV pages come from shared pools and are not statically partitioned per lane, so different requests
can occupy very different amounts of context memory. Fixed recurrent/backend backing is bound to the
current lane; this is a physical placement and does not bind an external request or a compact row identity
to a lane.

### 6.2 Completion reservation

This section describes RAM-off reservation. RAM-on windows and completion claims follow §5.8.

Admission computes a target-specific entitlement for each request. The logical upper bound for Main Text is:

```text
prompt_tokens + effective_output_tokens - 1
```

It is rounded up to the page size. The DFlash Full backend uses the same entitlement; the MTP backend also
covers up to `K-1` provisional leading positions, and is truncated by the per-sequence `max_context`. Prefix
reuse changes only the already-materialized portion and the prefill work; it does not lower the entitlement
that a new request can ultimately reach.

These target facts, rounded by each pool's page geometry, form the authoritative admission vector of §5.2.
The same vector is used for the actual reservation, the protected frontier, and the persistent-safe shadow
ledger; the Scheduler does not maintain a second token or byte estimation formula.

Every class of shared state resource must always satisfy:

```text
active used capacity
+ active reserved-but-not-yet-used capacity
+ retained used capacity
<= total usable capacity
```

Although the physical KV pool of the selected speculative backend is separate from the Main Text pool, its
capacity is planned jointly by the same target-specific profile. Let `S=max_context`, `P` be the page size,
`L=ceil(S/P)`, `C=max_concurrency`, `M_min=max(L,C)`, and `M_max=C*L`. The Explicit policy derives
`M=ceil(K_main/P)` from the user's token capacity; the Automatic policy reserves headroom `R` after weights
are loaded and solves directly, from the reservation curve of the complete target physical layout, for the
largest `M` that fits in `F-R`. CLI/server use `R=1 GiB` and report the actual free memory after the full
startup. Main and DFlash each use `M` physical page groups, and the logical capacity of each allocation is
`L`; MTP uses `M + C*ceil((K_draft-1)/P)` physical groups, with the same logical capacity `L`, where
`K_draft` is the speculative draft window. The extra groups hold only the provisional lead of multiple
concurrent MTP rows and do not enlarge the request context. Startup requires `K_main>=S`, `M>=C`, and
`M<=C*L`; `K_main>=S` applies only to Explicit. After capacity resolution, all typed pools and the Graph
topology are fixed and are not expanded at request time. After active-priority retained eviction, any active
entitlement set that satisfies the advertised Main pool contract must also satisfy the backend reservation;
an earlier backend failure is a startup sizing or accounting invariant violation, not normal admission
backpressure.

Prefill/decode progress only converts the logical reservation the request already holds into used capacity
and creates no new resource requirement. Whether physical blocks are set aside in advance or bound on demand
belongs to the Sequence-State Store.

Prefix reuse atomically adjusts the retained entitlement on the same lane and converts it to an active
entitlement, without double charging. When a request ends, a retained entry keeps only the pages that are
already mapped and does not keep future growth entitlement; eviction releases the whole continuation before
active admission is retried.

Before starting a prefill chunk or a decode round, the Scheduler uses the request's remaining reservation and
its logical context/output limit to bound the maximum possible state growth of that unit. A request with no
legal room to grow becomes terminal directly and must not start a unit that could overrun.

This decouples concurrency from context length:

- one request can use most of the pool's capacity;
- two requests can use completely different amounts of capacity;
- when the capacity left by active reservations is insufficient, a new request waits;
- an active request never discovers late in its run that its declared output cannot be accommodated.

Therefore every request must have a finite maximum output bound. When the caller does not provide one, the
Engine uses the configured generation cap.

### 6.3 Fixed and growing state

Growing state such as the KV cache is charged by request context. Fixed-size recurrent/backend state is
allocated per admitted sequence. Even if the two come from different internal pools, both must be satisfied
in the same admission decision.

Growing KV state is carried by the [Paged KV Context Store](paged-kv-cache.md) in target-defined homogeneous
page-group pools. A sequence always holds a Main Text allocation; when the Engine selects MTP or DFlash, it
must also hold the corresponding backend allocation. Each pool has an independent frontier and reservation;
the physical context of a single request need not be contiguous. The concurrent engine operates only on the
bundle handle, the logical frontiers, and the reservation vector, and does not participate in page or
physical address management.

While GPU work is in flight, the meaning of the per-request state handle and view must stay stable. The
contracts for page size, slab layout, block table, and allocator belong to the Paged KV Context Store and are
not repeated in this concurrency document.

### 6.4 Prefix reuse

A retained prefix is a single-owner SequenceState detached from a finished request. It stays on its original
physical lane, but that lane's control slot is free to the scheduler. Retained state publishes only
checkpoints for which the target has saved the complete continuation state. The current Qwen3.6 retained
state can publish the current resume frontier, one typed rewrite checkpoint when valid, and, under MTP or
DFlash, the prefill context-checkpoint ladder heads on the live lane. A rewrite is marked `TurnClosure` or
`ResponseReplay` according to its purpose at capture time; the two kinds are mutually exclusive and reuse the
same physical payload, and additionally store the corresponding recurrent, hidden, speculative-backend, and
position state, without copying the KV payload. A ladder head freezes the current GDN at the end of the
committed prefill chunk and the hidden of that chunk's last column; the KV remains a prefix of the same
bundle.

The Qwen frontend publishes the desired checkpoint from the effective `preserve_thinking` semantics: `false`
selects the end of the first assistant opener after the last real user turn; `true` selects the end of the
complete deterministic generation prologue of this request, i.e. the current prompt frontier. Thinking
generation includes `<think>\n`, and non-thinking generation includes the complete empty thinking block.
Exception: when the latest assistant turn of a thinking request has no reasoning (discarded by the client),
the next-turn re-render diverges right after the opener (the effort template omits the empty wrapper; the
`<think>\n\n` of the toggle template is tokenized separately), so the checkpoint is placed at the generation
opener (before `<think>\n`), at the cost of one extra very short prefill unit. The boundary is first
produced as a byte offset, then independently tokenized and verified to be a complete prompt token prefix;
the schema adapter does not infer or rewrite these target-private semantics.

When admission succeeds on the same lane, it consumes the retained entry and transfers SequenceState
ownership to the new request:

- when the incoming prompt ends at the current resume frontier, suffix token processing is skipped, and the
  request enters `DECODE_READY` after a finalization unit produces the next anchor;
- when the incoming prompt fully contains the current resume frontier and has a following suffix, the suffix
  is prefilled from that frontier;
- when the incoming prompt ends exactly at a matching rewrite checkpoint, its complete continuation state is
  restored, and the same finalization unit produces the next anchor;
- when the incoming prompt matches a saved rewrite checkpoint and has a new suffix after it, after the atomic
  admission transaction commits, the growing allocations are truncated to the checkpoint frontier, the
  complete checkpoint is restored, and then the suffix is prefilled;
- when the incoming prompt matches a prefill context-checkpoint head at `F`, it takes
  `restore_context_checkpoint` or `restore_turn_rollback` (determined by the head kind):
  trim to `F`, load that head's GDN into current and its hidden into `tail_hidden`; for MTP set
  `mtp_kv_valid=F-1` and then take the existing bridge; for DFlash write that head's cyclic local back to
  that lane and set `dflash_context_frontier=F`.
  It is not `append_frontier`, and it does not read the rewrite hidden;
- a common prefix that ends at any other position without a checkpoint is treated as a cache miss.

Rewrite-checkpoint restore keeps the partial tail page that contains the checkpoint and releases the complete
pages after it. A KV page or token prefix match is not by itself a checkpoint; the current architecture does
not support arbitrary longest-common-prefix reuse.

Checkpoint kind is not a reuse compatibility bit. The Planner always first tries the current frontier by
token, position, media identity, and complete continuation state, and then takes the longest complete
checkpoint among the rewrite and that bundle's ladder heads; even if this request's `preserve_thinking`
selects a different desired kind, a matching older snapshot can still be restored, and `prefix_reuse_path`
reports the kind actually restored. The Planner takes the longest complete checkpoint among the rewrite and
the ladder/turn-rollback heads; a current matching `E` still beats append; a same-`F` rewrite beats
rollback/ladder. When `append_frontier` and `prompt_tokens > E`, turn-rollback is pinned at that `E`.
An exact hit is not pinned by default; if the request asks for `capture_context_checkpoint` and that `E` has
no head yet, it is written to the same slot. If the desired boundary lies in the suffix to be prefilled in
this request, the physical slot is overwritten when it is crossed; if it already lies before the selected
reuse frontier and there is no snapshot at that position, this request keeps the legal reuse and defers the
new checkpoint, rather than proactively doing a full reset just to build an auxiliary snapshot. A full reset
happens only when the incoming prefix matches no complete checkpoint.

A retained entry can be consumed by only one active request at a time. Multiple active requests do not share
the same writable sequence state, and no copy-on-write branching is used. `cache_write: false` uses independent copied state
and a protected cache source as described below. Request-local RNG, sampling, stop, generation
limit, and output state are always created by the new request.

Retained state occupies actual state-pool memory but not an active control slot, and does not keep future
growth reservation. Active admission has priority; before cache occupancy blocks an otherwise feasible
request, the retained entries on free lanes are evicted first. VRAM retained physical state stays on its
original lane, and the Planner selects the largest legal reuse among the free lanes. Ordinary consuming hits do not copy or migrate GPU page mappings; disposable forks
copy payloads into independently owned mappings. Equal reuse prefers a free lane with no retained
bundle; among equal-reuse dirty free lanes disposable state precedes ordinary state, then it covers the least-recently-admitted retained chat
(monotonic occupy tick assigned after the first non-throwing `advance_prefill` in
`start_prefill_lane`), then lowest lane index. An
optional host-RAM second tier snapshots a completed
bundle into pinned host memory at the admission site that is about to destroy it. A later hit
restores the full SequenceState onto the chosen free lane and then uses the same prefix-reuse
decision. Cache ownership is claimed only after the slot/lane and the full entitlement are already
satisfied.

Prefix lookup changes only the uncached prompt work and the prospective reuse plan, and does not by itself
grant queue priority. It may conservatively shorten the service projection of §5.5, but must still pass the
same protected-head qualification; whether or not it hits, the final active request enters the same
prefill/decode schedule and compact batch formation.

#### Disposable cache reads

`ExecutionOptions.cache_write` defaults to true. False selects a source-preserving fork, with
source identity and destination lane planned independently. Longest preservation-feasible reuse
wins, with direct VRAM, RAM, and disk tie preference; full-original RAM preservation is attempted
when direct VRAM admission cannot coexist with its source. Reclaim unrelated idle retention
before giving up. An infeasible longer hit does not exclude a shorter feasible hit or preserving
cold path. Active lanes keep ordinary FIFO/deadline/head-protection semantics.

The family Program composes the selected current/checkpoint state into independent main/backend
KV allocations, GDN state, hidden state, and DFlash cyclic state. Sampling and output control are
request-local. Source mutation, page reclamation, and image/staging reuse wait through its readers;
the single executor copy-hold owns initialization and cancellation. No page sharing or copy-on-write
is used. The fork's durable disk association is cleared before execution.

An ordinary original may be captured to RAM when its lane or pages must be freed. This capture
bypasses preserve-off rewrite cutting and preserves the complete original resume frontier plus
existing checkpoints. Before eviction, failed/deferred admission can release the source claim and
roll back that capture; after eviction, the saved original is committed and cancellation cannot
remove its only image. A fork releases successful RAM restores instead of consuming them and
keeps RAM-to-disk bookkeeping intact. Disk restore completion keeps the inclusive durable entry.
A disposable original cannot be offloaded. Permanently impossible preservation on an idle Engine
returns a request-local overload; temporary active/copy/reclaim pressure queues and replans.

False requests suppress future-reuse captures while retaining required execution transaction
state. Their valid terminal state carries disposable retention. Empty eligible lanes are preferred;
among retained victims disposable state precedes ordinary state, then admission recency and lane
index decide. Apply the same order to destination replacement and shared-page reclamation.
All capture entry points and both shutdown passes reject disposable persistence. A later true
continuation sets ordinary retention for its resulting state.

Restore failure disposition distinguishes invalid source, stale plan, and destination resource
failure. Only invalid source discards/quarantines the entry. Stale disk generation/claim races
release pins and replan; destination resource failures keep a healthy source and can cold-prefill.
Device-integrity failures remain Engine-wide. Request logs carry the write flag; saving an original
is billed to its triggering request, without counting it as a disposable checkpoint capture.

### 6.5 Host RAM second tier

`--kv-ram-capacity` enables a startup-fixed pinned-host budget that stores **already completed**
prefix bundles, plus any live ladder GDN heads moved into that same FIFO image at eviction. The
complete contiguous arena is pre-faulted in parallel, CUDA-registered, and validated concurrently
with artifact materialization before the target Program takes ownership. The default is `off`. It
does not change GPU pool capacity, active-set
accounting, or CUDA Graph addresses, and it does not move an in-flight request off the GPU.

MTP or DFlash prefill may freeze the current GDN into an Engine-wide staging slot during an in-flight
prefill (after the Program prefill step compute-syncs that chunk). That freeze borrows slot `C` from the
turn-rollback occupant: `occupied=false` before the clobber D2D, pack the ladder head, then H2D
rollback GDN and hidden back. DFlash also D2Ds that lane's cyclic local into a 1-lane Engine-wide
staging window before the same `d2d_done` fence; host-pack D2H reads that frozen window, not the live
local. The freeze is not a RAM capture; D2H of
staging GDN and cyclic onto the live-lane host log may overlap the next chunk on the same lane. KV pages of the
in-flight unit stay off `copy_stream`. `--no-prefix-reuse` disables ladder capture, turn-rollback
pin, and restore.

The host tier is an exclusive FIFO of chats that are not on a VRAM lane. A new capture appends at
the tail. Capture first rejects images that cannot fit around a claimed restore source; feasible
capacity pressure evicts the oldest unpinned host entry until the new image fits. If capture
cannot complete, `drops` increments and the incoming request proceeds while the VRAM bundle is
released. Host RAM is not a precondition for admission. A failed optional capture must not
remove that lane from the reclaimable victim set or return `Overloaded`: this applies both
to the selected dirty lane and to other free retained lanes needed for shared-pool capacity.

Capture and retained-lane eviction run only after `find_admission_lane` has already selected a
free lane for that request. A queued prompt that still cannot fit — because in-flight lanes occupy
enough shared pool pages that no free lane is feasible even after reclaiming every other free
retained bundle — waits and does not dump any GPU lane. A later smaller backfill candidate that
can fit may still admit and may capture other free retained lanes for itself.

The executor captures at each admission site that is about to destroy a retained bundle, not inside
`clear_lane` / `evict_retained_lane` / abort / destructor:

1. other free-lane retained entries the eviction loop is about to release;
2. a selected VRAM `FullReset` plan whose target lane is still retained (the usual
   `max_concurrency=1` load path);
3. a RAM restore that is about to cover a still-dirty target lane.

Capture queues D2H on `copy_stream` and **holds the source pages mapped** until `copies_ready`.
Admit is two-phase: bind records the request in its lane and the retained victims awaiting release
as copy-hold (including victims whose optional capture was dropped); other decode-ready lanes may
run a DecodeRound while that D2H (and later restore H2D) is in flight. Admit-complete waits with `cudaEventQuery` (and `EventSynchronize` on copy
only when membership is empty), then `evict_retained_lane` / `kv.reset()`, optional restore H2D
(a RAM restore starts only once its entry's own capture copies are ready, polled while others
decode),
`wait_kv_ram_copies_on_compute` immediately before this lane's `start_prefill_lane`, and harvest.
Harvest of D2H/H2D elapsed happens after that wait, not on an overlapping DecodeRound launch.
A cancellation while other lanes decode does not drain at once: `begin_copy_hold_cancel` stops
further disk restore work and records a fence on the copy stream behind the copies already queued,
the hold stays parked while DecodeRounds continue, and the drain below runs once
`copy_hold_cancel_settled` (or membership is empty), so it no longer waits on in-flight SSD reads.
A shutdown with such a hold parked finishes it as cancelled.
A `CacheRestoreFailure` before prefill consumes the prompt parks the hold the same way. Once it
settles, cleanup releases the claims. An invalid source is discarded and the request falls back
to cold prefill; a stale plan preserves the source and replans admission; an unavailable destination
preserves the source and falls back to cold prefill. Generation recovery keeps its lane and output
while restoring or cold-prefilling the retry. It refreshes a stale host lookup once, then cold-prefills
if that refreshed restore is also stale. Other request-local failures drain and fail only the
request; other exceptions remain Engine-fatal.
Allocation failure during optional retained-prefix capture, including RAM eviction-candidate
bookkeeping, drops that capture without failing inference. Disk-assisted RAM reclaim reports
`Failed` and counts `optional_alloc` when its metadata allocation fails; it preserves the resident
images so a later reclaim can retry. CUDA and state-integrity errors retain their fatal semantics.
Live suspension-credit reservation also returns unfunded on host metadata allocation failure,
retaining existing secured credits and live images. This allows the scheduler to reduce or defer
admission using the same resource policy as contiguous RAM exhaustion.
Window grants reserve host page-vector metadata before committing GPU entitlements; allocation
failure leaves the funded window unchanged. Suspension insertion and capture metadata failure
leave the resident continuation intact. Resume metadata failure fences transfer work and clears
only the partial destination, preserving the parked source for retry. Failed cleanup synchronization
remains fatal. Suspended-window planning reads stored image page floors without allocating host
image descriptors.



If the held request is cancelled or fails before admit-complete, drain waits for those copies,
harvests, releases unused RAM/disk claims, discards only sources classified invalid, and calls
`evict_retained_lane` on every selected victim not yet evicted. A failed restore releases those
victims the same way.
If capture succeeded, the completed D2H image is the only remaining copy. A later RAM hit
claims the matching host entry (pinned entries are invisible to later `plan_match`). `capture` and `unpack` record a start CUDA event before the copies and a done event
after them so other-lane decode can overlap the DMA. The pinned rewrite-checkpoint and ladder
images (GDN conv/recurrent, about 150 MB per image, plus DFlash cyclic) copy host-to-host through
`enqueue_host_copies`, a stream-ordered `cudaLaunchHostFunc` callback, because a host-to-host
`cudaMemcpyAsync` is synchronous with respect to the calling thread. Each image's `copies_done`
fence is stream-waited before and re-recorded after those copies; disk restore does the same on
its state stream. A RAM restore's installed ladder heads copy out of the entry on that host-copy
stream behind only their own fences: no stream joins them and the entry's copy fence, which gates
the lane's first prefill chunk, excludes them; a separate block fence keeps the entry's block
allocated until they land. Ordinary consume then erases that entry wherever it
sits in the FIFO and retires the host block, including after an incomplete first chunk; a throw
before consume releases the claim and leaves the host row in place. After consume the bundle lives
only in VRAM until a later spill recaptures it. Occupancy `used`/`entries` (human `kv-ram=` / `n=`)
count those live host residents, including a claimed-but-not-consumed pin, and exclude retired copy
blocks that still occupy the pin until reap. A later capture may still reap or evict while logged
occupancy looks low.

The planner picks the larger `reusable_prompt_tokens` between VRAM and RAM; equal reuse keeps
VRAM. VRAM `FullReset` and RAM restore both prefer a free lane with no retained bundle and cover a
dirty lane only when no empty lane is feasible. Among those equal-reuse dirty lanes, the victim is
disposable before ordinary retention, then the least-recently-admitted bundle and lowest lane
index. Page-reclaim eviction of other free retained lanes uses the same order and never captures the selected lane twice.
The admitted `RequestPlan` is the winner: RAM pass 1 keeps `evict_retained=false` even if the
target lane is dirty, and restore captures that lane's old bundle. `GenerationResult` uses
`prefix_reuse_source` for `none` / `vram_resident` / `host_ram` / `host_disk`; `prefix_reuse_path` still describes
only frontier/checkpoint semantics.

An opt-in system-prefix LRU lives in the existing RAM arena (`--kv-ram-system-prompts N`, default
0). The family frontend emits a verified leading-system token frontier independent of the first
user's content, including rendered tools and template instructions. A writable reusable request
splits prefill at that frontier, captures all continuation state, and orders later compute after the
entry's capture fence to protect active GDN/cyclic state and a partially filled last KV page.
Restoring such an entry retains it and touches recency rather than consuming it. Its immutable
prefix competes by the same longest-ready-prefix rules as ordinary RAM images. Count and arena-byte
limits bound retention; normal FIFO/disk eviction cannot select system entries, and system snapshots
are not durably spilled. Longer existing reuse is never sacrificed merely to capture an earlier
missing system prefix. See `paged-kv-cache.md` for ownership and eviction details.

`--kv-disk-capacity` adds an inclusive SSD third tier beside the RAM FIFO. Disk is considered in
both admission passes after the RAM strictly-greater check; equal reuse keeps VRAM then RAM. C=1
disk restore never enqueues RestoreRead until the VRAM victim is evicted. Orderly shutdown captures
retained VRAM, flushes not-durable RAM, and fsyncs before aborting occupied lanes. When
`EngineOptions.load_progress` is set, that callback reports `kv-disk copy active chats`,
`kv-disk save cache entries`, and `kv-disk finish disk writes` during shutdown.

One restore coordinator and the startup-fixed reader pool own disk restore. A claimed request pins
its immutable entry while per-extent reader leases protect locations as readers fill the bounded pinned/page-device window.
The coordinator may overlap validated page H2D/scatter with later SSD reads and may load independent
state objects in parallel, but it publishes no partially restored lane. Admit-complete waits for
the restore state owners and copy-stream event immediately before `start_prefill_lane`; cancellation
drains those owners and releases the claim without consuming the durable entry.

Disk request work has three scheduling classes. Restore reads yield only to an emergency spill needed to
make the selected admission feasible. Emergency spill excludes all restore payload I/O. Idle spill
uses separate staging and may duplex with restore, but every page job rechecks its spill epoch and
RAM residency before capture so eviction/cancellation cannot publish a stale bundle. Commit and
fsync preserve pack-namespace → map → entry → manifest order. Shutdown promotes unfinished idle
work as needed, waits for all payload jobs, publishes durable metadata, then synchronizes the store.

The Executor separately grants disk maintenance permission only when pending ingress, all request
lanes (including disposable requests), and copy holds are empty and tier copies are ready. It
revokes permission before admitting or executing request work. RAM write-behind does not grant
this permission. Idle defragmentation yields to every request I/O class, uses throttled copy
chunks into existing holes, and preserves old locations for readers across durable Move journal
publication. Shutdown cancels maintenance and waits for its location handoff or cancellation
before reporting the disk tier synchronized. The exact budgets and recovery order are defined in
`paged-kv-cache.md`.

---

## 7. Scheduling model

### 7.1 GPU scheduling units

The Scheduler submits only two kinds of GPU compute work:

```text
PrefillChunk(request)
DecodeRound(all decode-ready requests)
```

A complete request is not a scheduling unit. Ordinary decode and exclusive prefill execute on
`device.stream`. Eligible DFlash mixed rounds also use one startup-owned nonblocking owner
stream; matching-layer event joins preserve the single round boundary (§8.8).
Host-RAM D2H/H2D transfers run on `copy_stream`; they are not scheduling units, must not enter a
CUDA Graph, and must not alias the pages of the current compute unit.

The Model Runtime owns one address-stable shared workspace, which is not duplicated per request. Serial units reuse
the allocation; a mixed round uses disjoint bounded owner and verify views within that same
startup allocation (§8.8). A prefill owner may hold one request-transient lease across the Vision/Text phases and multiple chunks;
the lease is released after final prefill, cancellation, or failure. A copy-hold blocks further admission until its copies
permit `start_prefill_lane`. A transient-bearing owner cannot use the interruption exception;
a suspended text owner retains no transient lease.

### 7.2 Boundary processing

When the current GPU unit completes, or when a control event arrives while the GPU is idle, the GPU Executor performs one boundary:

```text
1. snapshot control events visible to this boundary
2. resolve the completed unit's per-request results
3. finish/cancel requests and release or retain their state
4. clean visible pending cancellation/timeout/permanent failures
5. if this boundary has admission eligibility, apply one head-first
   protected admission/backfill opportunity (bind may return copy-hold
   without launching a GPU unit; complete starts this lane's prefill
   only after the relevant copies are ready)
6. choose, prepare and launch one next GPU unit
```

A parked copy-hold that was cancelled or whose restore failed is checked first. Until its fenced copies settle, as long as the
membership is non-empty, only that membership's DecodeRound runs; generation recovery, admission, and prefill do not advance. After they settle
(or if the membership is empty), the lane is cancelled, or the failed entry is dropped and the cold fallback is taken. The remaining copy-holds are checked before the
admission gate: if the membership is non-empty and the copies are not ready, the
DecodeRound of the other lanes runs first (no harvest, no `EventSynchronize` on the copy); if the membership is empty, the copy is `EventSynchronize`d
and completed. After the capture D2H is ready, `admit_complete` may enqueue a restore H2D; if the frozen membership is non-empty,
the same boundary transaction immediately runs that membership's DecodeRound without reprocessing control events. Harvest
happens only after that request's `start_prefill_lane` wait.

An event that arrives after the boundary begins is left for the next boundary. This guarantees that a membership update is finite and that continuous arrivals cannot
delay the next launch indefinitely. The admission gate, the frozen pending snapshot, and the protection state are defined in §5.6;
a boundary that does not satisfy the gate only updates control/resource state. Cleaning up multiple invalid entries does not count as multiple admissions, but one
boundary publishes at most one newly admitted request.

### 7.3 Decode/prefill policy

The Engine and server default to `mixed_forward = 4096`, `mixed_forward_rounds = 12`,
and `mixed_forward_min_tokens = 4096`. Suffixes below the threshold run exclusively,
including checkpoint tails; zero threshold applies fixed cadence to every suffix and
disables short-prefill interruption.
Concurrency and speculative backend selection remain independent. A smaller explicit
`prefill_chunk` requires a compatible mixed width or `mixed_forward = 0`; single-request
CLI and perplexity execution explicitly select prefill-first. Benchmark defaults remain
prefill-first for control measurements. User tuning is documented in `docs/serving.md`.


The spare-lane exception checks the FIFO head's exact selected resident plans before
suspension. It never bypasses a live queued head, creates a protection epoch or borrows
backfill credit. Every successful short bind consumes the same frozen decode-admission budget
as ordinary admission. Cancelling a suspended owner releases only its state; cancelling the
short owner resumes the large one at the next legal boundary. Statistics count both staged
owners as prefilling requests. The single request-transient lease remains exclusive to the
executing owner.

The scheduling policy is:

```text
if a parked copy-hold was cancelled or its restore failed:
    if its fenced copies have not settled and requests are DECODE_READY:
        run one DecodeRound containing them (nothing else advances)
    else:
        cancel it, or drop the failed entry and fall back to cold prefill
else if an admitted copy-hold exists:
    if its copies are not ready and requests are DECODE_READY:
        run one DecodeRound containing them
    else:
        complete it and start its prefill
else if a suspended prefill exists and no current owner/copy-hold exists:
    restore its shared KV-row, RoPE, logprob and sampling-mask bindings; resume its cursor
else if a prefill owner exists:
    if its last chunk completed, interruption allowance exists, remaining >= threshold,
       and decode-admission debt allows another bind:
        consider only the live FIFO head from one bounded queue snapshot
        require a spare lane, no transient lease, direct resident feasibility and no eviction
        respect protected-head persistent safety; suspend the large owner and admit the short one
    if mixed_forward > 0 and remaining >= mixed_forward_min_tokens and requests are DECODE_READY:
        if the target can mix the owner's next chunk into DFlash verification:
            run mixed_forward_rounds - 1 ordinary DecodeRounds, then one DecodeRound
            containing the owner's next Shared-pace chunk
        else:
            run mixed_forward_rounds ordinary DecodeRounds, then the next Shared-pace chunk
    else:
        run the next Exclusive-pace PrefillChunk (at most prefill_chunk)
else if a generation-recovery retry is queued:
    start it (a host restore parks it as a copy-hold)
else if a pending request is admissible and decode-admission budget permits:
    admit it and start its prefill
else if one or more requests are DECODE_READY:
    run one DecodeRound containing all of them and refresh the next burst
else:
    remain idle
```

With prefill-first pacing (`mixed_forward = 0`, or every remaining suffix below the threshold),
three cold startup requests follow:

```text
Prefill[A, all chunks] -> Prefill[B, all chunks] -> Prefill[C, all chunks]
-> DecodeRound[A,B,C] -> DecodeRound[A,B,C] -> ...
```

When A is already decode-ready, B/C are also pending, and lanes/resources/budget are available, C is admitted immediately after B's complete prefill;
`DecodeRound[A,B,C]` runs only after both B and C are decode-ready. If B reaches a token terminal during prefill, its bind still consumes
budget, and continuing ingress cannot skip the mandatory donor decode.

The request plan projects prefill service work at the Shared-pace step width: `mixed_forward` does not exceed 4096
(the irregular-split extent), so no Shared/Exclusive alternation ever takes more steps than the projection.

The prefill chunk profile bounds the time of one non-cancellable GPU unit; the wall time of a complete admission burst may include up to
`C-1` complete request prefills. The specific token/media extent is a configuration qualified for the target and hardware,
and is not part of the scheduler semantics. Vision and other prefill GPU phases must themselves constitute a bounded unit, or must already be counted in that
chunk's latency bound; there is no unbounded prefill work outside the scheduler.

### 7.4 Joining and leaving decode

- A request joins the active decode set after final prefill/finalization completes. Exact retained hits skip
  suffix work; a short FIFO head may use the spare-lane exception, subject to the same resources
  and interruption allowance as other short prefills;
- A newly ready request first appears in the batch built at the next boundary;
- The protected-head policy affects only admission; once a backfill is decode-ready, it must not be excluded from the
  maximal batch in order to protect the waiting head;
- EOS, stop, or the generation limit removes the request after the completed round commits;
- The batch rows are re-compacted after every join/leave;
- An empty slot never produces an empty row.

### 7.5 Cancellation

Cancellation does not interrupt an in-flight GPU unit. The boundary observes cancellation once before launch, and a cancelled request does not
enter the next membership. After the unit returns and before any per-row output preview/commit, one row-aligned cancellation snapshot is taken again against the frozen membership;
this snapshot is unchanged for the whole resolve. A row cancelled at this point discards this unit's
not-yet-committed result and closes the response output per the serving contract. The already committed continuation
(KV, GDN current, rewrite checkpoint, ledger, and the context-checkpoint
heads at or before that frontier) is retained for later prefix reuse, at the same boundary as `OutputLimit`. An unfinished suffix prefill rolls back to the occupy
base: if a turn-rollback or ladder head exists there, that head is restored; otherwise, if a rewrite checkpoint is still held, it rolls back to
that checkpoint. After the rollback, the staged heads beyond it are dropped and the lane's staging occupancy is released, and then the state is retained. When the occupy base is 0 and
there is no rewrite, the whole `SequenceState` (including staged heads) is released. Ordinary in-flight decode has already overwritten
GDN current and the continuation hidden in place, so it cannot retain after rolling back only the ledger/KV frontier; that cancelled row releases the whole
`SequenceState`, consistent with §8.4. A cancellation observed between two units can still retain the committed state that has not yet been overwritten.
A speculative in-flight row folds with `commit_columns=0` and then retains the committed frontier from before the unit. Error/shutdown
still releases the whole `SequenceState`. Provisional writes do not affect the other rows of the same round.

Therefore, cancellation waits for at most one GPU unit whose membership is already fixed, plus one boundary processing.

The cancellation of a waiting protected head or backfill candidate is removed in the §7.2 control cleanup; when the protected head
leaves the queue, the epoch is atomically cleared, and the next-oldest entry does not inherit shadow state. The cancellation of a frozen donor shortens the frontier only after its active
state is actually released; if this boundary has no admission budget and the decode membership is still alive, the head retry is triggered after the
next maximal decode; if the last donor leaves, the retry can happen immediately.

---

## 8. Batched model execution

### 8.1 Runtime ownership

The resident Model Runtime is a model-instance object, not a request object. Within the Engine lifetime it uniquely owns:

- immutable model weights and the target schedule;
- the shared Sequence-State Store;
- one shared execution workspace;
- one `DecodeBatchFrame` of maximum capacity `C`;
- when the speculative backend is enabled, one all-layer ReplaySSM record arena of capacity `C` and width `draft_window+1`;
- DFlash's capacity-eight accepted-history planes, per-row provisional innovations, and per-lane retained length;
- startup-captured graph definitions and topology executables.

Each request's persistent state exists only in the slot control and the `SequenceState` currently owned by that slot. The Model Runtime does not store a
"current request", and does not duplicate the workspace, round buffers, or graph assets for each slot.

A decode round additionally creates two short-lived objects:

| Object | Location and lifetime | Content and ownership |
|---|---|---|
| `RoundMembership` | host; from batch build until that round's commit completes | immutable `B` and the `row -> slot` mapping; keeps the slots/state alive |
| `DecodeBatchFrame` | host/device stable storage; reused serially by all rounds | typed input controls, state selectors, round activations, and result staging; owns no sequence state |

`RoundMembership` does not upload request identity. The device schedule sees only the typed selectors required by the target/Op; completed results
are mapped back to the original slot by the host mapping. While GPU work is in flight, the membership, the related slot bindings, the state allocation, and
the frame ingress must not be modified.

### 8.2 DecodeBatchFrame

`DecodeBatchFrame` is planned once at Engine startup for `C`. Each exact-`B` schedule obtains only prefix views of the same backing,
and no separate set of round memory is allocated for each `B=1..C`. Stale rows beyond `B` are not read by the graph, so the
boundary does not need to zero the inactive tail.

The typed fields required by ordinary decode include at least:

```text
current_tokens[B]
cache_positions[1,B]
RoPE positions/deltas[per target contract, B]
Main KV table rows[B] + optional backend KV table rows[B]
lanes[B]
SamplingConfig[B] + logical sampling positions[B]
sampled_tokens[B]
```

`lanes[b]` is the stable execution lane of row `b`. Qwen3.6 uses it to select both the Linear Attention current state and the
continuation-hidden destination; speculative Fold also obtains the same lane from the frozen membership. The KV table row
remains separate, because it describes a paged allocation binding, not fixed-state ownership.

Hidden activations, mixer intermediates, and logits use exact-`B` views in the same shared frame/workspace.
`DecodeBatchFrame` is the runtime's logical collection of these typed regions, not a generic descriptor ABI passed to all Ops;
each Op still receives only its own semantic inputs. Host/device control staging may transfer the entire small SoA frame of fixed capacity `C` in one transfer; the model schedule reads only exact-`B` prefixes and submits no model work for the inactive tail.

Batch assembly handles different data differently:

| Data | Batching method |
|---|---|
| Small controls such as token, position, and sampling config | written per compact row into contiguous batch ingress |
| KV payload and block tables | kept in the shared paged pool; only the per-row table-row selector is written |
| Linear Attention / backend fixed state | kept in the shared state pool, located by stable lane or backend-defined row |
| Request stop, output, and external identity | kept only in the host slot; a registered EOS that must be masked during open reasoning enters the graph as a SamplingConfig row control |
| Activations, hidden, logits | produced in shared execution memory by one whole-batch schedule |

KV or recurrent state must not be gathered/copied to build a contiguous batch, and a device-pointer array must not be built for each row. The final hidden or other continuation image that the Target
needs to retain across rounds must be published to the respective `SequenceState` through typed
destination selectors within the same batched schedule, and not through a separate D2D copy per request at the boundary.

### 8.3 Round preparation

After the boundary selects a `DecodeRound`, the Target Batch Assembler prepares in the following order:

```text
all DECODE_READY slots
        │ compact once
        ▼
immutable RoundMembership[B]
        │
        ├─ materialize each row's already-reserved one-round state growth
        ├─ fill current token, exact positions and typed state selectors
        ├─ fill per-row sampling inputs and continuation destinations
        └─ select one whole-batch execution profile
        ▼
publish exact-B ingress -> launch
```

Every member of an ordinary round has one valid column, so there are no inactive rows or valid-column masks.
Materialization may update block tables for different allocations, but the pool/table-matrix base addresses remain unchanged. All
rows must be fully prepared before launch; it is not allowed to launch part of a batch first and then supplement state for another row.

Profile selection uses a bound over the whole batch, for example the largest visible-context frontier among all rows. Exact per-row
positions and context lengths remain device data. If a legal active set cannot be represented by a single profile, then that route
does not satisfy the concurrency contract, and the rows must not be split into cohorts.

The host may run a lightweight loop over the metadata for `B<=C`, but GPU ingress must be a batch-level publication; an
independent scalar copy, transition kernel, or model call must not be issued for each row.

### 8.4 Ordinary round state transaction

A `DECODE_READY` sequence always stores a target-defined committed cursor and a single current decode anchor. Taking
the Qwen3.6 ordinary transition as an example:

```text
before:
  target state valid for positions [0,p)
  current anchor token is at position p

execute:
  process the B anchors at their respective positions
  write each row's provisional KV/recurrent/continuation state
  sample one next anchor per row

commit for surviving row b:
  state frontier  := p[b] + 1
  current anchor  := sampled_tokens[b]
  anchor position := p[b] + 1
```

An Op writes only the physical state permitted for this round and does not own the committed frontier. The graph also does not advance a compact-row position or
selector in place into "next-round state"; the authoritative cursor is updated by `SequenceState` at the boundary commit, and the next round generates controls from the new
row mapping. This lets rows move freely, without vectorized scalar helpers or persistent row-local
state.

An ordinary round licenses exactly one token for each uncancelled row. EOS, stop, or the generation limit can make that token
a terminal token, but the model state and the output of the same row are not cut at different frontiers.
Cancellation is the only ordinary boundary outcome that can discard a whole row's provisional result; that `SequenceState` is then released.

Qwen3.6 ordinary GDN calls the existing width-1 Snapshot leaf with `initial_state_slots=lanes` and `snapshot_base_slots=lanes`.
After fully reading the row's initial checkpoint, the leaf overwrites the same current slot in place, so it produces no speculative
trajectory and needs no additional state slot.

### 8.5 Whole-model execution

When the logical batch size is `B`, the model schedule of one replay is:

```text
current_tokens[B]
  -> one batched embedding/input stage
  -> one traversal of all model layers
  -> publish B continuation images
  -> one batched lm_head
  -> one batched sampler
  -> sampled_tokens[B]
```

The ordinary column layout is equivalent to `[D,1,B]`, and column-independent projection/MoE/FFN routes consume the aggregate
`T=B` at once. Control code may iterate over lightweight row metadata, but must not call a layer, model
schedule, lm_head, or sampler separately for each request.

Per-request KV/context, recurrent state, sampling state, and output ownership remain independent; batched execution
merges the current activation columns and the weight-consuming work, and does not concatenate multiple sequences into one sequence.

### 8.6 Operator contract

| Operator family | Batched execution requirement |
|---|---|
| Linear, projection, dense FFN | one operator schedule consumes the aggregate row/token extent |
| MoE | route over the aggregate token set, then execute grouped experts over the whole set |
| Full attention | consume per-row context length and the substrate-provided KV view as a ragged batch |
| GDN / linear attention | execute one batched transition over independent per-row recurrent-state handles |
| `lm_head` | produce, at once, the logits of all active rows or verification positions |
| Sampler | consume per-row sampling configuration and RNG state, without splitting the batch |

Different context lengths, stop conditions, sampling values, or generation limits do not form different model batches. A request option that requires a
different model topology must be fixed at Engine startup, or else is rejected as unsupported.

The specific compact layout, typed extents, and selector ABI are defined by the corresponding `include/ninfer/ops/` semantic contracts;
this architecture does not duplicate the per-Op interface list.

### 8.7 Result resolution and commit

After graph replay, only the compact per-row result is returned, for example the ordinary sampled token; logits or per-layer state must not be returned.

Token logprobs follow the same rule. A request that set `OutputOptions::top_logprobs` raises its
row's `logprob_rows` flag in the round's control frame. Every decode schedule (ordinary, MTP, and
DFlash, captured or eager) ends its token decision with `ops::token_logprobs`, which scores each
produced token against the logits column that decided it (column `i` for a chain, node
`fold_path[i]` for a packed tree) and writes a fixed-size record per token slot into the result
frame: the token's log-probability and the 20 highest-ranked ids with theirs. The launch is part
of every definition, so no graph variant or post-replay launch exists; a row whose flag is clear
returns from the kernel before reading logits and its record slots are not written. The
prefill-sampled first token uses the same Op on the scalar step logits, before the MTP bridge
reuses them. The Engine turns the records of the committed token prefix into `TokenLogprob`
values per output channel; tokens that publish to no channel are dropped there.
The GPU Executor waits for one whole-round completion, then resolves each row independently through the frozen `RoundMembership`:

```text
row result
  -> locate its still-owned slot through RoundMembership
  -> apply cancellation / stop / output-limit policy
  -> commit the exact licensed prefix to SequenceState
  -> update request sampling, usage and owning output record
  -> mark DECODE_READY again or MODEL_FINISHED
```

For each row, the model-state frontier, target cursor, sampler/penalty progress, usage, and owning output record form
one logical transaction. A streaming output event may become visible to the response path only after all of this state is committed; a
terminal-only request produces no per-round event. A row finishing, being cancelled, or accepting a shorter
prefix in speculative mode does not change the commit result of the other rows.

A round sampled with model stop tokens excluded may commit only through the first completed
`</tool_call>`; `resolve_pending_batch` then folds that shorter prefix. A plain chat round also
commits through its reasoning close before sampling the continuation with the dedicated closing
token excluded. Tool grammars enforce the close boundary in their per-node matcher masks and reserve
closing-think markup only in free prose; constrained argument values retain literal markup.

When the MTP target GDN uses ReplaySSM, the GPU graph only reads the lane's current state and writes the
Program-owned raw records; it does not advance the committed GDN state. After the CPU output preview obtains the final commit length of each row,
`resolve_pending_batch` first executes one all-layer Fold with the original `B` rows, then completes the necessary hidden/backend correction,
and advances the host frontiers only after synchronization succeeds. A cancelled row participates in the original row mapping with `commit_columns=0`, the Fold is strictly a
no-op for that row, and then the row's already committed continuation is retained, rather than releasing the bundle. The Executor may commit the output preview only after this commit tail
succeeds, and, in streaming mode, publish the output event.

DFlash target GDN uses a Program-owned FP32 checkpoint plus capacity-eight accepted history.
The startup arena contains shared normalized keys, per-value-head innovations and decay, one
retained length per lane across all 48 layers, and separate provisional records at the captured
verify-width ceiling. Record reads the checkpoint and retained history, then derives innovations
from represented BF16 raw K/V and FP32 controls using the actual provisional parent. A commit
consumes only that same invocation's chain prefix or ancestor-complete selected tree path,
including the anchor. It appends history while the total length is below four; otherwise it
materializes the complete FP32 checkpoint and publishes length zero. Conv3 commits every round.
An abort commits zero and leaves the logical state unchanged. Arbitrary edited raw records or
records derived from another parent cannot use this compound profile. MTP retains native Fold.

All history work uses the main compute stream. All-layer state/ring writes finish before a
separate count publication; the next verify cannot overwrite provisional records until commit
finishes. Program mirrors the same validated accepted/path count on the host, including the
anchor, with the same flush-to-zero rule. An already dense slot requires no materialization
launch. Dense prefill, current-to-staging or host snapshots, and retained-prefix publication
materialize first. Restoring or zeroing a dense checkpoint resets the device length in stream
order. Lane discard clears host ownership; acquisition/reset or restore clears the device count
before reuse. The single staging slot and RAM/disk images always contain dense FP32 state.
Existing copy-stream completion events protect snapshot reads; history adds no stream or event.
All storage addresses are fixed before graph warmup; ordinary/MTP execution has no history owner.

After all rows have resolved, the `RoundMembership` is destroyed and the frame may be overwritten by the next unit. Slots that continue running are re-compacted at the next
boundary; no row identity is inherited from the current frame into the next frame.

### 8.8 B=1 and prefill

`B=1` uses exactly the same membership, frame, whole-model schedule, and commit transaction, and no separate
request-local ordinary decode path is kept.

Prefill has one sequence owner. An eligible DFlash mixed round evaluates that owner alongside
the decode rows while preserving separate state and activation storage. The Model Runtime owns
one nonblocking owner stream and two completion events for its lifetime. A primary-stream event
orders owner work after ingress. Each matching layer records both streams' completion events
before either cross-stream wait, so the two branches can overlap within a layer and both finish
before the next layer. The primary stream joins the owner tail before verify-head evaluation,
acceptance, state publication, or owner-context append. Failure cleanup drains outstanding owner
work before either branch's state or scratch can be released.

The owner and verify branches borrow disjoint bounded views of the same startup-planned workspace
allocation. Round scopes restore both cursors after stream completion, including failure cleanup;
the views retain their observed peaks. Public workspace accounting reports the greatest observed
allocation extent across the ordinary, verify, and offset owner views, and peak reset resets all
three views. No mixed round allocates a second workspace or changes graph addresses.

Exclusive prefill remains a single-sequence unit. The owner exclusively owns its `SequenceState`
and does not hold a persistent `DecodeBatchFrame` row. After final prefill establishes its complete
decode cursor, the request joins ordinary decode through batch assembly at the next boundary.

---

## 9. CUDA Graph model

### 9.1 Exact-B graph definitions

Because `C` is small and fixed, the Engine captures an exact definition for every logical batch size
at startup:

```text
Definition[family, B=1, profile]
Definition[family, B=2, profile]
...
Definition[family, B=C, profile]
```

`family` is fixed by the Engine startup mode as ordinary, MTP, or DFlash; all three modes use exact
`B=1..C`, and DFlash keeps no separate `B=1` concurrency route.

`B=1` is a first-class exact path. A single request never runs a graph permanently padded to `C` rows.
Each definition represents the complete decode round for its semantic family, exact `B`, and
whole-batch profile, and binds the same shared frame, workspace, and state-pool bases. It reads typed
controls and selectors, not request/slot identity.

Captured definitions and replayable executables do not correspond one to one. Exact `B` is the
structural key of an executable; within the same `B`, an executable is added only when the real CUDA
node topology differs:

```text
exact definitions[family,B,profile]
          │ exact B + target-declared profile topology class
          ▼
executable[family,B,topology class]
```

If different context profiles have an updatable node topology under the same exact `B`, they share one
executable, and the selected definition is installed when the profile changes. Different `B` values never
perform a cross-B graph update; this prevents batch-dependent operator routes, kernel parameters, and
launch shapes from triggering `cudaGraphExecUpdate` incompatibility. The number of resources is therefore
a finite set over exact `B`, not the full Cartesian product of `B × context profile`.

Captured adaptive panels, proposal packing, fused QKV splits, and GDN replay-record packing
use one raw device-copy kernel per panel. CUDA Graph updates cannot change multidimensional
memcpy parameters, including workspace addresses. The kernel copies exact row payload bytes
with independent pitches; profile changes can update addresses and extents without changing
copy-node topology or adding a launch per row.

Profiles are truncated by the configured context ceiling before capture, and only the topology classes
that are actually reachable instantiate an executable. A backend-specific proposal shape forms a topology
class under each exact `B` only when it really changes the CUDA node topology; no ordinary-tail or extra
`B=1` compatibility graph is generated.

Grammar-exchange presence is also an explicit profile/topology field. Every MTP and
DFlash2 `(k, B, planned topology)` has separately captured variants without and with
the exchange; their executables are distinct because the host branch changes CUDA node
topology. Startup graph accounting includes both. A round selects the variant from its
actual compact sessions; switching between ordinary, all-grammar and mixed batches never
captures a new graph or updates an executable across variants.

The startup preparation order for a graph family is fixed:

1. For each enabled semantic family, run one eager round with the first reachable profile at `B=1`, so
   that CUDA code and the library runtime finish lazy materialization before stream capture;
2. Capture all exact-`B`/profile definitions. Capture only records CUDA work; it does not execute a model
   round and does not carry production numerical qualification;
3. Each topology executable is instantiated from its first definition. The remaining definitions of that
   topology are verified for compatibility one by one through `cudaGraphExecUpdate`, and executable
   resource materialization is completed through `cudaGraphUpload`;
4. Each executable replays only its first definition once as a startup smoke. After the remaining
   profiles are traversed, the first installed definition is restored through update/upload, with no
   additional replay.

Startup therefore performs no eager warm-up for each definition, and does not treat the update of each
profile as real-model replay qualification. Complete operator and route correctness is the
responsibility of independent tests and the real-artifact integration route; production startup verifies
only the graph inventory, update compatibility, resource materialization, and one executability check per
executable.

### 9.2 Dynamic active set

When the active set changes, the runtime publishes a new frame ingress and selects the matching
pre-captured definition:

```text
[A]       -> Definition[B=1]
[A,B]     -> Definition[B=2]
[B]       -> Definition[B=1]
[B,C,D]   -> Definition[B=3]
```

The runtime first selects the exact-`B` executable; if it currently has a different context-profile
definition of the same `B` installed, it first performs a whole-graph update and then replays once. When
`B` changes, it directly selects another pre-instantiated executable. When the active identity changes
but `B/profile` does not, only typed data is updated, not the executable. A graph is not keyed by request
identity, slot subset, or slot bitmask, so join, leave, and slot reuse do not trigger capture.

Installed-definition state is modified only by the GPU Executor; update and replay lie on the same
serial execution lane. Concurrent callers cannot directly launch or update the same executable.

Paged-KV allocation, Linear Attention state ownership, and retained-state reuse change only block-table
content or selector values. A graph always binds the pool/table-matrix base, so these ownership
transitions do not form a graph key either.

### 9.3 Context and prefill profiles

The exact context length is a typed runtime control value. If the exact target needs a small number of
different kernel topologies for different context ranges, then besides `B`, a finite set of whole-batch
profiles fixed and captured at startup may be used. A profile is selected once by a batch-level bound,
for example the maximum row context-length bucket; the graph still executes ragged work with the exact
per-row lengths.

A context profile must not split the active set into cohorts, and must not be keyed by request identity
or active-slot combination. A profile that cannot simultaneously represent arbitrary legal per-row lengths
cannot serve as a concurrent decode route; a route that preserves whole-batch execution must be used.

Prefill is single-request work and may use a target-specific fixed execution profile. It does not change
the decode-batch graph model, and it must not trigger, during serving, a graph capture that would block
active decode.

### 9.4 Stable execution memory

All exact definitions reuse:

- one `DecodeBatchFrame` planned for `C`;
- one workspace sized to the maximum over all reachable phase/`B`/profile peaks;
- the same set of model/state pool bases;
- one set of address-stable pinned host ingress/result staging planned for `C`.

A graph-enabled exact-`B` definition contains one batch-level control upload at the start and one
batch-level result download at the end. Staging objects may be transferred at the fixed size `C`; only the
first `B` rows have semantics for the current round:

```text
fill pinned ingress rows [0,B)
        -> optional already-reserved KV page/table materialization
        -> one graph replay:
             one H2D typed-control frame
             whole-model batch schedule
             one D2H result frame
        -> one round completion wait
```

Cross-page materialization happens on the same execution lane before replay, does not change the captured
pool/table bases, and does not form a graph key. Graph-off mode submits these actions eagerly in the same
order.

Logits, hidden, workspace, or per-sequence state must not be duplicated for each `B`, profile, or captured
definition. There is no per-row CUDA submission in model/control ingress, forward, and result egress;
table publication when crossing pages belongs to state substrate materialization. No graph family is
captured, instantiated, or extended during serving. The startup graph allowance must count all reachable
exact-`B` definitions, plus one executable for each exact `B` and each actual topology class, and must not
reuse a reservation that covers only `B=1` definitions.

The temporary Paged-KV allocation used by capture/smoke materializes only one private physical page per
valid row, and publishes that page repeatedly into the temporary block-table row. Preparing one
eager/smoke run zeroes only the temporary pages, fixed-state lanes, and typed controls corresponding to
`[0,B)`; definition capture itself only resets the host allocator of the shared workspace. The whole
physical pool must not be zeroed according to the configured KV capacity, and the state of all `C` rows
must not be zeroed for each definition.

---

## 10. Speculative decoding

Speculative decoding is an Engine-level decode mode:

```text
speculative_backend = off | MTP | DFlash
```

All requests of the same Engine use the same backend and proposal window. The Scheduler still submits
only one `DecodeRound(all decode-ready requests)`.

Internally this round executes:

```text
all pending request anchors
  -> one batched proposal phase
  -> one batched target verification
  -> per-request accepted-prefix result
  -> per-request state commit
```

MTP advances proposal positions autoregressively, and each position executes as a batch over all active
requests. DFlash produces the proposal block through one batched block forward. The difference between the
two exists only inside the selected decode graph and does not change slots, admission, scheduling, or
active-batch formation.

MTP and DFlash keep their own proposal frame, proposal state, and graph topology; they share only the
non-owning target verify/accept view, the per-row accepted-prefix result, and the target state
transaction. The concurrency layer neither builds a virtual interface that hides backend differences nor
maintains two sets of membership or commit loops for the two backends.

Engine capability is fixed at startup. An MTP Engine can enable Vision at the same time; when MTP's
shifted input lands on a vision column, it uses the same Vision-composed embedding as the target, and the
prefix-reuse bridge follows the same input semantics as normal prefill. The Qwen3.8-27B NVFP4 DFlash2
companion itself stays text-only, but can be resident together with Vision: after the multimodal target
prefill completes embedding compose, it captures the same selected-layer hidden features for the DFlash
context. DFlash proposals use absolute positions; the target verifier uses an independent position panel
and adds the saved `rope_delta` row by row, so subsequent decode keeps MRoPE semantics. This combination
is not a request-level fallback.

Different requests can accept different numbers of proposals. Acceptance length is result metadata; it
must not be used to split verification, replay the model, or form an acceptance cohort. The target model
is always the output authority, and only accepted target/backend state can commit.

Qwen3.6's MTP and DFlash share the same target ReplaySSM transaction: target verify writes each layer's
convolution/key/value/gate records into a fixed arena per compact row, and the physical record row always
equals the batch row of this round; after the CPU obtains the final output prefix, one Fold uses the
frozen `lanes[b]` to commit row `b` to the current state of that lane. Rows must not be compacted or
reordered because of cancellation or different acceptance lengths. Records lie inside the CUDA Graph, and
Fold lies in the eager commit tail after the CPU decision; the next GPU unit must wait until the current
records have been consumed by Fold before it can overwrite the arena.

From the completion of target execution to the end of Fold, the request is Pending: the authoritative
execution/ledger frontiers, the ledger contents, and the prefix identity still stay at the round base;
licensed tokens and backend staging exist only as unpublished candidates. Fold, Text/backend KV trim,
continuation hidden, proposal continuation, and the host frontier must commit the same final prefix. A
continuing row commits all licensed outputs; a terminal row may commit a strict prefix because of
stop/EOS/output limit; a cancelled row commits zero columns and releases its sequence.

Each row's valid proposal extent is bounded by the remaining output/context capacity. Unused positions in
the fixed proposal window are masked and must not update request state or output.

If a row is not yet terminal but its valid proposal extent is zero, the same DecodeRound must perform
ordinary target progress for that row; an active row that does not advance must not be kept. EOS, stop,
output limit, or context limit truncate the effective committed extent before the per-row commit, and
model state and output commit only the same effective prefix.

---

## 11. Completion and response handling

Ingress establishes an owning record for the request and fixes `OutputDelivery` before queue membership;
the effective output bound is determined at admission. Each boundary appends the committed tokens to that
record, so neither an ordinary nor a multi-token speculative round depends on network progress to commit.
`Streaming` writes the compact deltas of that round into a request-owned event queue and wakes the
consumer at most once; `TerminalOnly` does not build per-round output events.
Both delivery modes can enqueue sparse, owning `RecoveryEvent` diagnostics and wake the consumer
before completion. `wait()` dispatches them through `OutputSink::recovery_event` on the caller
thread, never through model-output deltas. Cycle-exclusion notices use power-of-two count
cadence; retry transitions and terminal totals are always emitted when recovery was involved.

When the model completes, the GPU Executor finalizes the owning result record, releases the request's slot
and unused reservation, and either releases the sequence state or, per §6.4, transfers state that has
target-declared reusable checkpoints to the retained-prefix cache. After the above GPU
resources are reused, the response path can still continue to read the owning record.

Committed output is always written to request-owned result storage; streaming deltas and recovery
diagnostics go into separate event queues, which are drained by the caller thread waiting on that
request. Network writes, protocol serialization, and `OutputSink` callbacks do not run on the GPU
Executor. A slow caller does not block the GPU boundary loop; the backlog is still limited by that
request's finite output bound. Request lifetime capacity continues to be charged until the response
consumer releases the owning record, so the reuse of the model slot does not create an unbounded
completed-result backlog.

Model completion and response completion are therefore two independent lifetimes:

```text
model lifetime:     admission ───────────── model finished
response lifetime:  request received ───────────────── response finished
active slot/state:   admission ──────────── model finished
retained state:                                 optional ───── eviction/reuse
```

Runtime telemetry does not own executor state. The four hot monotonic counters are updated by a single
executor writer through lock-free publication; the scheduler and KV-RAM gauges are published only after an
actual state transition such as admission, prefill transition, completion, cancellation, timeout,
failure, or KV-RAM mutation. `runtime_stats()` returns a race-free, fieldwise-concurrent view, rather than
pretending to be a multi-field transaction with a single boundary identity; the transition publication
caused by a completing request must precede that request's terminal notification. The periodic serving
reporter reads only this published view and does not call `memory_summary()`, which would take execution
ownership; the latter is reserved for startup, CLI, and caller-requested complete arena diagnostics.

A GPU execution or state-integrity failure is an Engine-wide failure: admission stops, active and queued
requests end with unavailable/failure, and serving can resume only after the Engine is recreated. The
Scheduler does not guess that some row of a failed shared round can safely continue.
Worker failure closes admission before cache-tier cleanup begins. `Engine::available()` reads only
the admission state, without waiting for GPU execution or disk cleanup. Serving returns HTTP 503
with `{"status":"unavailable"}` from `/health` after worker failure or while stopping; a busy or
full healthy executor continues to return HTTP 200 with `{"status":"ok"}`.


---

## 12. End-to-end examples

### 12.1 Prefill burst joins an existing decode

This example uses prefill-first pacing. Shared large owners instead yield the decode cadence
in §7.3, and an eligible short FIFO head may use its spare-lane interruption.

```text
time ───────────────────────────────────────────────────────────►

GPU: Decode[A]
       │ boundary: admit B; freeze remaining burst budget
       ▼
     Prefill[B, all chunks]
       │ B becomes decode-ready; admit C with remaining budget
       ▼
     Prefill[C, all chunks]
       │ C becomes decode-ready; budget exhausted / slots full
       ▼
     Decode[A,B,C] -> Decode[A,B,C] -> ...
```

B/C do not run a separate decode forward, and do not insert an A decode between their own chunks; after
their respective final prefill, they join A's next logical batch together.

### 12.2 Two rows assembled and committed

Suppose A and B each occupy one of two slots, and hold completely different context/state allocations:

```text
                         row 0 (A)        row 1 (B)
current token               101              202
cache position               41              900
Text KV table row             5                1
stable lane                   0                2
sampling state lane           0                2
```

The Batch Assembler writes only the above controls/selectors into the frame. KV pages and Linear
Attention state stay in place in the shared pools. `Definition[B=2]` performs one complete model traversal
over the two rows and returns:

```text
sampled_tokens = [303, 404]
```

The boundary uses the frozen mapping to hand 303 to A and 404 to B. If A continues and B's 404 is a
terminal token, both rows first commit their own correct model/output frontier, and then B's slot/state is
released or retained. The next round re-forms the batch as A's `B=1` frame; A's KV row and stable lane do
not change because the compact batch row changed.

### 12.3 Active batch grows and shrinks

```text
ready requests          selected definition

[A]                     Definition[B=1]
[A,B]                   Definition[B=2]
[A,B,C]                 Definition[B=3]
[B,C]       A finished  Definition[B=2]
[B,C,D]     D joined    Definition[B=3]
```

Neither request identity nor the occupied slot affects graph identity.

### 12.4 Shared 128K context with two slots (RAM off)

Suppose the relevant shared context capacity is 128K units:

```text
request A: used prompt 50K + reserved output 30K = 80K commitment
request B: prompt       32K + reserved output 16K = 48K commitment
                                                       -----
                                                       128K
```

A and B can be admitted at the same time. As context grows, reserved capacity converts to used capacity,
so the `used + remaining reservation` of the two never exceeds 128K; it can reach the boundary but cannot
cross it.

If B needs `32K prompt + 32K output = 64K`, only 48K remains after A's commitment. B stays in the pending
queue and becomes the protected head, until A releases capacity, B's queue timeout expires, or another
terminal event occurs. NInfer does not admit B first and then decide which active request to truncate when
approaching 128K.

If there is also a later request C, it can backfill only after passing the full vector accounting of §5.
Explained on the Main capacity dimension alone, A is B's frozen donor; beyond the capacity that A's release
leaves for B, there is still 64K of shadow surplus. As long as the other dimensions such as lane/backend
also hold, this gives C a frontier-safe upper bound; but C must also satisfy the current `fits-now`, and
the real free space at this moment is only 48K, so C can currently obtain at most 48K. Even if C is still
active when A completes, the release boundary still admits B first. If C needs to borrow B's critical
capacity, it must also satisfy the temporal work credit, and a frontier miss triggers drain.

### 12.5 Persistent-safe and temporal backfill

The following example writes only the Main entitlement; the actual decision checks the lane and
selected-backend pages dimension by dimension at the same time. Suppose the capacity is 174K and `C>=4`:

```text
active A: 64K, projected to finish first
active B: 64K
free:     46K
head H:  100K, exclusively feasible but blocked now
```

After selecting `D={A}`, the frontier accounting is:

```text
B 64K + H 100K = 164K
safe surplus = 10K
```

An 8K candidate P can become a persistent-safe backfill:

```text
B 64K + P 8K + H 100K = 172K <= 174K
```

Even if P is still running when A releases, H retains its full entitlement. Another 20K candidate T fits
into the remaining capacity at present, but cannot join the persistent-safe ledger; it can serve as a
temporal backfill only if its complete prefill/decode service work is projected to finish before A's
release frontier and the epoch temporal credit is sufficient.

```text
Decode[A,B]
  -> admit/prefill P completely
  -> admit/prefill T completely if temporally qualified and burst budget remains
  -> Decode[A,B,P,T] ...
```

If A reaches EOS early while T has not yet finished, the Scheduler does not interpret this estimation error
as a new backfill opportunity:

```text
A releases
  -> retry H first
  -> H is still blocked by T
  -> PROTECTED_DRAIN
  -> no new admission; existing P/T and B continue
  -> T releases
  -> admit H
```

If T has already finished in time, or the early completion of another active request makes the exact
accounting sufficient, H is admitted directly at the corresponding boundary. Later arrivals never became
donors of H, and cannot restore temporal credit that has already been consumed.

---

## Related documents

- [Paged KV context storage](paged-kv-cache.md)
- [ReplaySSM GDN technical reference](replayssm-gdn.md)
- [Serving behavior](../serving.md)
- [Qwen3.6-27B model semantics](qwen3.6-27b-model.md)
- [Qwen3.6-35B-A3B model semantics](qwen3.6-35b-a3b-model.md)
