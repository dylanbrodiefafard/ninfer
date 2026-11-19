# Flash-Next R9700 expert helper

Temporary implementation plan for branch `qwen3.8-flash-next`. Design only; no helper
integration or performance win is qualified by this document. Fold the implemented decisions
and evidence into the [target plan](qwen3.8-flash-next-plan.md) and active references, then remove
this document. Proposed file and API names below are implementation assignments, not existing APIs.

## 1. Selected design and evidence

One optional gfx1201 R9700 owns an immutable, exclusive **decode-resident** subset of routed
experts. It consumes the artifact's original F3G32/E2M1G16 records and returns one FP32 weighted
partial per participating token. The 5090 retains the router, shared expert, all non-MoE work,
remaining expert cache/misses, proposal network and sampler. No CPU expert execution, peer access,
requantization, runtime repacking, helper prefill, ownership swaps or new Engine scheduler.
Prefill may temporarily copy helper-owned records into the existing 5090 streaming windows;
exclusive placement means persistent routed-expert residency and decode execution, not those
necessary local-prefill copies or the immutable host backing store.

Use CUDA-owned mapped pinned mailboxes registered with HIP, separate request/response cache
lines, and system-scope release/acquire. **Prequeue a HIP graph containing all 48 layer programs
before launching CUDA graph A**, rather than implementing another persistent expert engine.
There is no CPU per-layer dispatch. Small rounds use vectorized mapped transfers; T48 uses
prequeued DMA. The helper is idle between rounds, not a permanently spinning controller.

Evidence, all on this host, is the Strata study and phase reports under
`out/sol/dual-gpu/{report,phase0-report,phase0b-report}.md`. Phase 0b supersedes phase 0's ~250 µs
tax: an over-wide checker caused most of it. Best zero-expert exchange costs are:

| Packed target columns T | Path | Exchange median / p99 | 48-exchange cost |
|---:|---|---:|---:|
| 1 | Persistent vector mailbox probe | 16.832 / 18.560 µs | 0.791 / 0.817 ms |
| 4 | Same | 29.344 / 35.008 µs | 1.419 / 1.448 ms |
| 20 | Same | 124.576 / 128.544 µs | 5.983 / 6.016 ms |
| 48 | Prequeued DMA, captured CUDA probe | 220.512 / 226.144 µs | 11.009 / 11.212 ms |

The first three totals are quantiles of synthetic sums, **not model forwards**. Their persistent
transport costs are an optimistic basis for the selected prequeued small-T implementation;
extra HIP kernel boundaries must be measured. Corrected prequeued DMA at T1 actually measured
36.544 µs/exchange and 2.108 ms/forward. A persistent controller would require a different,
resident-worker expert implementation and global work protocol; duplicating the independently
qualified multi-launch HIP Op is not justified by that possible ~20 µs/layer saving. Revisit
this choice only if measured launch overhead prevents the required end-to-end win.

The 20 MiB R9700 read floor is 72.959 µs, 287.4 GB/s, not an expert-kernel timing or the card's
bandwidth ceiling. Phase 0's synthetic expert-read kernel is not a useful compute cost model.
No GPU experiment was run for this design.

## 2. Ownership, build and activation

| Owner | Files and responsibilities |
|---|---|
| Core | New `src/core/hip_device.{h,cpp}`, `expert_mailbox.{h,cpp,cu,hip}`: checked PCI selection, HIP device/stream/event/graph/device-buffer RAII; CUDA allocation/HIP registration and both aliases; bounded raw publish/acquire/copy/ack operations. No expert IDs, placement policy or model scheduling in core. |
| Ops | The concurrent Op owner's HIP expert contract/implementation under `include/ninfer/ops/` and `src/ops/`; CUDA expert partition/pack, local partial and final join contracts in that same ownership boundary. Cache eligibility belongs to `moe_expert_cache`. No mathematical kernels in the target or core. |
| Flash-Next Program | New `impl/runtime/expert_helper.{h,cpp}` and `expert_helper_profile.h`: placement, immutable helper banks, metadata upload, per-round 48-layer HIP graph construction/submission and lifecycle. `Program` owns it; `Forward` receives a non-owning view. `ExpertResidency` owns only the remaining CUDA cache. |
| EngineProgram | `impl/engine/engine_program.cpp::make_program_options`, startup diagnostics and request-error propagation. No HIP scheduling in `ConcurrentExecutor`, no helper lane state. |
| Product | `include/ninfer/types.h::EngineOptions`, `apps/cli/{options.h,options.cpp,main.cpp}`, `src/serve/{serve_options.h,serve_options.cpp,generation_service.cpp}`: one startup option, forwarded to Engine. |

The Op/CMake owner already working in this checkout owns the opt-in build option and HIP build
recipe. Agree one name, **`NINFER_ENABLE_HIP_EXPERT_HELPER`**, with that owner before integration;
if their delivered spelling differs, use that one consistently rather than adding an alias.
Default OFF means no HIP compiler/runtime discovery, HIP objects, ROCm link dependencies or
helper tests. Keep the top-level C/C++/CUDA compiler configuration intact; compile gfx1201 HIP
translation units in the option-owned separate target/toolchain invocation and link its narrow
host interface. Do not compile CUDA sources through HIP or introduce CUDA-name compatibility
headers. Preserve the existing default CUDA-only image and setup; the existing separate dual-GPU
builder is the opt-in environment, not a replacement container.

Conditional compilation is limited to build source lists, the Program/Forward helper members
and call seams, and enabled/disabled option validation. HIP headers/types never enter ordinary
CUDA Ops, public Engine headers or shared scheduling code. A plain internal descriptor contains
sized arrays, integer counts, opaque device aliases and ownership masks; CUDA and HIP translation
units use their own runtime headers. Enabled-build CUDA partial Ops need no HIP runtime types.
OFF compiles the current ordinary `moe_experts` call, not a fake helper, virtual backend or
no-op implementation. In an enabled binary **without the flag**, do not initialize HIP, allocate
mailboxes/profile tensors/scratch, change slot planning, or insert partition/publish/wait nodes.
The existing CUDA arithmetic and launch sequence remain intact.

Public startup flag: **`--expert-helper-pci 0000:13:00.0`**; owning Engine value
`std::optional<std::string> expert_helper_pci`. Parse a full PCI domain:bus:device.function,
normalize once, and reject malformed/empty values. The flag remains recognized in OFF builds
only to report "expert helper requires a build with NINFER_ENABLE_HIP_EXPERT_HELPER=ON";
that is validation, not an execution stub. `src/targets/registry.cpp::reject_flash_next_options`
rejects it on the 27B identity. Flash-Next `Package::validate_options` validates configuration;
core resolves the address and checks gfx1201, mapped-host/system-atomic support and available
memory. Missing device, unsupported device, failed registration or failed startup exchange
causes a contextual startup error, never silent CUDA-only fallback. No request-level toggle,
helper ordinal selection, automatic device discovery policy or HTTP JSON field.

## 3. Static placement: protect the primary hot core, cover its complement

### Input gap and profile ownership

The [artifact](qwen3.8-flash-next-artifact.md#6-routed-experts) has codec, codebook and matrix-scale
tables, **not route counts**. `ProgramOptions::calib_counts` exists for private diagnostics;
`EngineProgram::make_program_options` does not populate it and production `lanes-bench` rejects
`--calib-counts`. Do not pretend the codec map is a routing profile or require an undocumented
runtime sidecar.

Package the existing CALIB `[48,512]` nonnegative integer counts as a target-private constexpr
profile in `expert_helper_profile.h`, generated offline from the explicit study input
`/home/battlefront/ai/study/calib/routing_counts.npy`. A small target-private generation tool
checks shape, nonnegativity and integer range. Record the source/recipe in the generated header;
no installed Python dependency or absolute path is used at startup. The generator also accepts
served decode-route JSON v1 as an explicit offline input for this same compiled header (the
current header is generated from the served route profile of the final VM run); selecting
that profile requires generation and rebuild, never a runtime flag, sidecar or hot reload.
The artifact remains the
only product weight artifact, its directory and record bytes unchanged. Counts influence
performance only: the binder's actual codec map, record bytes, books and scales always determine
execution. Keep the profile out of OFF builds and do not change the flag-off cache warm start.

### Deterministic algorithm

1. In `Program::Program`, construct `State`, DFlash resources and `KvExpertExtents` as today.
   Account for enabled-only CUDA helper buffers in `Forward::device_bytes` **before**
   `program.cpp::slot_plan` subtracts future workspace and `kDeviceReserveBytes` from free VRAM.
   Let F be the resulting two **fixed** pool capacities. Flex F3 addresses remain the existing
   `extents_.flex_addresses()`; they are not guaranteed capacity when KV is fully reserved.
2. Use the existing 11% NVFP4 split to define a primary hot core P: for each codec take its F
   highest-count `(layer,expert)` pairs, ties by lower layer-major index, including zero-count
   pairs if necessary. Counts are global across layers, not equal per-layer quotas. Defining P
   from fixed capacity is intentionally conservative: `auto = C × max_context`, explicit KV,
   concurrency, DFlash workspace and helper overhead all influence it through real planning.
3. On the R9700 reserve **1 GiB** for runtime, Op scratch, staging and margin. Its record budget
   H is actual free VRAM minus that reserve, rounded down to 256 bytes. Fail if the measured
   non-record requirements exceed the reserve; never borrow primary KV capacity to cover them.
   For a free 32 GiB card, H is at most 31 GiB = 33.286 GB, not 31 decimal GB.
4. If the complement of P fits H, put **all of that complement** on the helper; leave spare
   helper memory unused. Do not move extra primary-hot experts merely to fill the card.
   Otherwise take pairs outside P in descending `count / record_bytes`, stable layer-major
   ties, admitting each whole record that fits and continuing past a record that does not.
   Compare integer ratios without overflow. The remaining tail stays host-backed for the
   primary cache. This favors useful removed bytes without putting the hottest core on the
   slower card. No split record and no balanced-layer fiction.
5. Freeze that ownership mask for the Program's lifetime. Upload helper-local pointers/scales
   and books; upload the same ownership bitset to CUDA. Plan/warm the CUDA cache against only
   the complement of helper ownership. When all of that complement fits fixed pools, provision
   every record of each codec, including zero-count ones; unused bytes need not be allocated.
   Otherwise use the existing two-pool split capped by remaining codec counts, GCLOCK and
   frequency filter. Flex slots retain fixed addresses/counts and ordinary borrow/return rules.

Helper experts are excluded from warm start, admit lookups/frequency/counters and decode
lookahead staging. Extend `moe_expert_cache_admit` with an optional per-layer eligibility bitset;
its routed set U is intersected with that bitset **before all hit, protection, frequency and
miss processing**. Extend `_stage` candidate eligibility likewise. Existing callers omit the
bitset and retain their exact semantics. `_fill`, `_commit` and `_defer` consume already-filtered
placements and need no new policy. `_window` intentionally remains unfiltered: local prefill
must access all routes. Ordinary small prefill also excludes helper experts from *persistent
admission*, but computes their unchanged IDs through mapped host records. No foreign pointer
or helper tag enters `expert_table`; nonresident helper entries remain valid CUDA host addresses.

No Strata swaps in v1. A static mask avoids transfers, double ownership, graph-bank replacement
and state races at runtime. KV pressure changes only local residency, never ownership; flex
reclaims cannot remove the helper's records. There is no primary per-layer bypass threshold in
v1: a helper-owned row always executes remotely. Changing that would require a different
placement/execution policy and must be earned by measurements, not hidden in a fallback.

### Capacity and hit-rate estimate

There are **50.2891 GB** of routed record bytes. With H = 31 GiB, only **17.0031 GB** must remain
on the primary, before slot-granularity loss. Thus this is potentially full two-card residency,
not simply a fractional improvement to the old hit rate. These are capacity examples, **not
claims that the historical reported budgets are all fixed rather than fixed+flex**:

| Guaranteed primary record budget | Helper complement | Primary CALIB assignment share |
|---:|---:|---:|
| 24.6 GB | 25.690 GB | 73.6% |
| 19.95 GB | 30.341 GB | 64.3% |
| 18.5 GB | 31.791 GB | 61.1% |
| 17.1 GB | 33.189 GB | 57.9% |

The last two columns come from a read-only CPU calculation using the above CALIB file, the
artifact's top-1583 NVFP4 assignment, the two exact record sizes and `plan_expert_slots`' 11%
rule. They estimate *assignment load*, not unique expert jobs, inference hit rates or helper
compute time. Actual F must be logged after enabled-only allocations; at 17.1 GB the margin is
only ~97 MB, so driver usage or workspace can invalidate full coverage.

If every non-helper record fits the fixed pools, the settled **5090-domain hit rate is 100%**
regardless of text; residual expert PCIe misses are zero. Warm all such records at startup.
Transient prefill-window borrowing can evict some; after window return, admits refill them, so
post-prefill startup/drain misses are not zero. No permanent pinning/refill sweep is added just
to hide this warmup cost. Define domain hit rate as hits / unique eligible `(layer,expert)`
lookups, and report helper assignments/unique jobs separately; counting helper routes as local
misses recreates Strata's PCIe-share bug. `ExpertCacheCounts` uses that eligible local domain.
The JSON-v1 route profile intentionally uses a different, global denominator: all unique routed
experts, with immutable helper residents counted as hits and only demanded local misses in
`miss_bytes`. It is placement/work telemetry, not a conditional primary-cache hit rate or a
measurement of all PCIe traffic. Keep the global profile/schema intact; distinguish these two
metrics in reports rather than treating its `hit_rate` as the 5090-domain rate.

For context, the old Gen4 C1 measurements ranged 87.2–96.0% hits and 40–126 MB/token;
C4 at 24.0/18.5 GB gave 82.4/73.1% and 662/1024 MB/round. Those are workload/policy-specific
baselines, not multipliers to apply after exclusion. When F+H cannot cover the store, e.g.
F ≈15 GB leaves at least ~2.0 GB outside both resident sets, historical aggregate hit rates
cannot determine the new conditional hit rate. Replay actual C1/C5 verify routes with the mask,
remaining pool split, frequency aging, prefill pollution and flex-reclaim timeline in
`tools/parity/qwen3_8_flash_next/replay_cache.py`. Report residual unique misses and bytes;
never assume independent helper/local hits or add percentages. A smaller working set may help
beyond directly removed misses, but no numeric claim is made without that replay.

## 4. Layer dataflow and arithmetic boundary

Integration point is `forward.cpp::Forward::moe`, immediately after its authoritative
`bf16_projection` → `moe_route`, before admit/fill. Leave `moe_route`'s logits, ranking,
renormalization and valid-column handling unchanged. The following Ops are enabled-only callers
of explicit semantic contracts, with reusable CUDA implementation machinery, not copied
mathematics in the target:

1. **Partition/pack Op** (`moe_expert_partition`, new contract/wrapper/launcher in `src/ops`).
   One 256-thread CTA scans at most 480 assignments and stably compacts participating tokens in
   original packed-token order. Outputs are the original-token → compact-token map (−1 for a
   token without helper routes), compact count N, BF16 activations and ten
   `(I32 expert_id, FP32 weight)` entries per compact token, written into stable compact CUDA
   VRAM panels owned by `ExpertHelper` and counted in `cuda_device_bytes`. Non-helper slots carry id −1 and weight 0, the inactive sentinel of this
   *new* helper contract, never of ordinary `moe_experts` or cache IDs. Preserve original
   route-slot order; do not renormalize either partition. The compact map is also the join's
   participation mask; no unused U16 helper-mask panel is allocated. Core's CUDA publish kernel
   claims the mailbox writer guard **before** copying these panels, then supplies the system
   fences and the request-header release. The Op never writes mapped storage. This keeps wire
   synchronization in core rather than introducing a second guarded partition implementation.
   For N compact tokens this adds a 5200 N-byte local VRAM read/write staging pair, not another
   PCIe transfer: the same 5200 N bytes are sent to mapped host memory. Maximum compact storage
   is 249,856 aligned bytes at T48; it is reused only after the layer's response/join.
   No host route readback or CUDA callback.
2. **Local branch.** Admit/stage see only eligible local experts. Existing admitted/fill-stream
   fork, resident work, mapped misses, deferred slot work and commit remain ordered as in
   `moe_expert_cache.h`. Add `moe_experts_local_partial`: reuse grouping/gate-up/down kernels,
   exclude helper assignments when building jobs, and emit an **FP32** `[2560,T]` sum of
   local routed contributions in original slot order followed by the shared contribution.
   The shared expert is always local. A zero local routed set still computes the shared expert.
   Ordinary `moe_experts` remains the complete BF16-output Op, including for all prefill.
3. **Helper branch.** HIP acquires the header, imports BF16 compact activations and metadata,
   groups its active assignments by expert, runs the separately qualified expert Op, and forms
   FP32 `[2560,N]` weighted partials in original route-slot order. No float atomics, shared
   expert, routing, n-gram data or intermediate gate/up traffic. Return FP32 even if internal
   matmul operands/intermediates follow the HIP Op's qualified private precision profile.
4. **Join.** After local expert work and cache commit, core's captured CUDA wait acquires the
   matching response once per layer, then vector-loads the partial into CUDA scratch. A
   `moe_expert_partial_join` Op scatters compact rows and computes
   `BF16(local_fp32[t,c] + helper_fp32[compact(t),c])`, rounding once at this observable MoE
   boundary. No intermediate local BF16 output/add, and no repeated system acquire by every
   output lane. Then the existing `hyper_connection_inject` and DFlash feature capture run.
   Waiting after local work overlaps all of its resident, deferred and mapped-miss execution
   with the helper; no per-expert join and no global device synchronization.

These contracts evaluate the original complete MoE formula. They change reduction association
(local slots + shared, then helper slots), not route weights or the mathematical operator.
Keep the default full MoE implementation's merge profile unchanged. The partial implementation
shares private grouping/matmul machinery with it, rather than maintaining a second CUDA expert
engine. New Op workspace planning must include both partial buffers and preserve work-arena
lifetimes through the final join.

Ordinary decode has T=C. Verify uses `t = row * W + column`, W=K+1, K=0..3 and C=1..12:
T≤48. Only this target verify computation uses the helper; DFlash's five-layer proposal network
and full FP8 head remain wholly CUDA. Preserve existing copied routes for padded columns and
compute their partition consistently; do not omit a helper contribution while local padded
columns still execute. Identical source rows may be deduplicated only as a separately verified
Op optimization, not an assumed capacity reduction. Rejected columns may perform helper work,
but cannot commit state. Mixed request-local extents, batch compaction, lane IDs and adaptive
step-down do not alter ownership or wire geometry.

`SubChunk::lane < 0` identifies the packed round route in the current `Forward`; do not select
helper execution merely because `streamed == false`, which also covers small prefill chunks.
For `run_super_chunk`, `enqueue_layers`, `enqueue_window`, `score`, all sub-chunk widths and
16,384-token interruption boundaries, execution stays local. Only the persistent admission mask
changes in helper-enabled prefill; normal layer-major window fill/deferral/return remains intact.

## 5. Mailbox, helper scheduling and failures

Allocate one fixed-capacity slot per layer, sized for `Tmax = lanes × (DFlash ? 4 : 1)`, not
per physical request lane. Each slot has disjoint input/output payloads and 128-byte-separated
producer/consumer header lines. CUDA is the only request writer; HIP is the only response writer.
A separate host-written cancellation line is not a second writer to either sequence flag.
Obtain CPU, CUDA and HIP aliases explicitly; a CUDA device pointer is not a HIP pointer.

Per compact token the logical input is **5120 BF16 activation bytes + 80 metadata bytes = 5200**;
the output is **10,240 FP32 bytes**. The original-token map stays in CUDA, so it is not transmitted.
Headers contain protocol version, layer, T, N, execution status and a 64-bit round sequence.
Qualify 64-bit system atomics on both devices in the startup protocol test; phase 0b's measured
flag round trip alone does not establish this extended packet protocol. Aligned payloads use
`uint4` transactions. At T48 all 48 payload slots total 35.6 MB plus headers;
at C1 K3 about 3.0 MB. These are small exchange allocations relative to 50.3 GB of pinned
experts, not a second expert arena. Compact CUDA source panels add at most 249,856 device bytes;
they are not another host exchange allocation. Initialize backing storage once so fixed-capacity DMA never
reads uninitialized padding. Do not register any `ExpertHostStore` chunk with HIP.

For every round, target `ExpertHelper::begin_round(T)` launches a pre-instantiated HIP graph:
48 repetitions of bounded wait/import → helper Op → export/publish, serialized on one helper
stream with reusable VRAM scratch. For T<48, import/export kernels copy populated prefixes
through the mapped aliases. At T48, use fixed-T H2D/D2H DMA into/from VRAM staging after the
request acquire and before response publication, matching the measured winning large-T route.
Initialize VRAM DMA staging as well as host backing once; inactive output rows must remain valid
transfer bytes even though CUDA never consumes them. T21–47 remain on the mapped path because
no measured crossover is established there. The Op consumes device N/job counts, so graph shape
does not depend on host routing results.
No device-side kernel launches, cross-runtime events, native uncancellable wait-value calls,
per-layer CPU thread dispatch or GPU-wide resident-worker barriers.

If N=0, CUDA publishes a sequence-stamped SKIP header only and casts its local FP32 partial
directly to the BF16 MoE output: **no CUDA wait, remote output read or helper-vector add**. HIP consumes/acknowledges SKIP, zeroes job
counts, and its Op kernels do no jobs. The mapped path transfers no token payload. Prequeued
T48 DMA nodes still move fixed initialized capacity even for SKIP; do not claim device-conditional
DMA or zero helper-side cost. Measure that queue overhead and the round-end drain explicitly.
Per-layer slots prevent consecutive skipped layers from overwriting unconsumed headers.
Join the HIP completion event before reusing any slot for the next round, including an all-SKIP
round; CUDA's skip alone does not grant host reuse. For sparse T48 traffic this fixed DMA route
is a qualification risk, not permission to add a CPU dispatcher silently.

Sequence equality, layer and shape checks reject stale/late responses; flags are not reset per
layer or graph replay. Write payload → writer system fences → group barrier → release sequence;
consumer leader acquire → group barrier → payload read. For DMA, publication follows completed
stream-ordered copies, with the system-visibility fence supplied by the export node. Abort/error
status uses the same ordering. Use atomics with the measured supported system scope, never
`volatile` as synchronization. At 64-bit sequence exhaustion, fail at an idle boundary rather
than wrap into a stale packet. Each round has at most one outstanding request per layer slot.

All device polling is bounded: a **5-second layer wait deadline**, cancellation checks and a
latched first-fatal-error record containing sequence/layer/side. One persistent host watchdog
observes response completion and progress at coarse intervals while armed for a round, with a **30-second round deadline**;
these are failure bounds, not performance targets. All 48 matching response sequences disarm
it even if the host has not called finish; a host stall after completion is not a deadline error.
Finish blocks on the HIP stream and then the CUDA stream without a polling sleep. Runtime errors
are latched as typed Host failures only after the watchdog is disarmed and the round marked
inactive; a failed runtime join also stops/joins the watchdog and retains enclosing storage until
process exit, rather than terminating or freeing possibly reachable aliases. HIP resource owners
bind their creation device for operations and destruction. HIP header/error details receive a
system fence immediately before their release publication. A failed source-seal wait poisons the
round and leaves N=0 instead of trapping the process; fixed DMA bytes never license failed jobs.
Driver wedges remain subject to the external whole-process deadline below.
On I/O exception between graph A and B,
request cancellation of unsubmitted work before draining; do not leave HIP waiting for layers
that CUDA will never publish. If both runtime joins succeed, no writer guard remains claimed,
and the only recorded state is host-requested cancellation, reset cancellation codes and
response seals at that idle boundary, retire the helper's pending execution and reuse its
storage. Sequence flags are never reset. Rethrow the original host exception under the existing
Program/request policy; no result from the discarded round is licensed. A fatal protocol,
deadline, helper or runtime error prevents this recovery. Fatal device causes may supersede
benign cancellation, including a bounded seal-wait deadline. If cleanup itself fails, retain
helper context with the original exception nested rather than hiding the root cause. Already-submitted normal request cancellation drains the round
and uses existing zero-publication semantics. On protocol timeout, the failed wait returns an
error and suppresses its copy; later helper jobs see cancellation/zero counts and terminate.
CUDA results must not be licensed/published or cached after any helper error, even if later
nodes execute to allow a bounded drain. Poison the Engine and fail outstanding requests with a
typed contextual error; do not reuse potentially advanced canonical state or retry on CUDA.

Normal teardown: stop submissions, complete/cancel the active round, join both graphs and fill
stream, destroy graph executables, unregister HIP, then free CUDA-owned pinned memory and
helper VRAM. Check asynchronous runtime errors as well as launch results. If a driver/runtime
is wedged and no longer honors bounded kernels or API calls, in-process recovery is not
promised: use an external whole-process deadline, as in phase 0. No destructor may free mailbox
memory still reachable by either GPU. Do not add a permanent busy wave, power-policy changes
or sysfs writes; first-round clock-ramp cost is measured separately from steady state.

## 6. Graphs, state and continuation tiers

`Forward::capture_decode_graphs` still builds graph pairs for each B; `capture_spec_graphs`
(in `speculative_impl.h`) still builds them for each K/B. Partition/publish and bounded wait/import
are ordinary captured CUDA kernels. Keep the existing graph A layer-zero / host PLE gather /
graph B layers-1..47 split. HIP is submitted before A; layer-one wait may span that CPU gather.
Round sequence is uploaded through stable ingress storage, not captured as a by-value constant.

Add round begin/finish/cancel to `Forward::decode`, `speculate`, and eager `verify` (for parity),
including their exception drains. After CUDA replay completion, check helper status and join
HIP before returning sampled rows or licensing speculative output. `Forward::~Forward` drains
before its non-owning helper reference becomes invalid. `Program` member order constructs the
helper after capacity/extents planning but before filtered `ExpertResidency` and `Forward`;
destruction reverses that order. If construction is split to break planning dependencies, use
one explicit startup layout value and one owning helper, not hidden allocations in Ops.

Mailbox aliases, ownership masks, helper banks, Op scratch, CUDA local/remote partial panels
and ingress storage live at fixed addresses for the Program lifetime. Include their sizes in
`Forward::device_bytes`, allocate routines and `Program::memory`; do not stash asynchronous
buffers in temporary `WorkspaceArena` scopes. Capture one HIP executable per actually supported
T/shape/transport, shared where T and bank views coincide, outside CUDA capture. Contents and
counts change; pointers and maximum extents do not. Ordinary cache mutation, flex borrow/return,
KV frontiers, request-lane membership and K/B selection need **no recapture**. Device, ownership,
capacity or buffer-address changes require an idle full Program reconstruction, not hot mutation.

The helper has **no per-request or continuation state**. Its immutable records are weight
placement; its activations, job lists and partials die at round completion. Do not add helper
fields to `State`, GDN/PLE snapshots, retained lanes, RAM/SSD images, live-suspension statistics,
raw n-gram history or DFlash pending features/calibration. Existing 116,139,008-byte base and
158,184,704-byte DFlash continuation payloads remain unchanged. Tier operations already occur
at settled boundaries; helper completion joins that boundary. DFlash acceptance and
`resolve_speculative`/`commit_verify` remain authoritative and unchanged.

Transporting an image produced with helper arithmetic is still an exact byte copy of that
image, not a promise of bitwise identity to an independently executed CUDA-only history. The
represented model/codec is unchanged, so helper PCI address, placement and cache residency do
not enter the disk model fingerprint. Cached tails, like current cross-batch/chunk tails, retain
the arithmetic of their producer. Cross-mode restored execution is checked numerically; same-mode
save/restore and branch source preservation remain exact transport/state contracts.

## 7. Startup, RAM and diagnostics

After loading `ExpertHostStore` once, initialize the requested HIP device and exchange resources,
calculate fixed-primary placement, then copy selected records from `ExpertHostStore::record`
through **two reusable 16 MiB HIP-owned pinned staging buffers**. CPU memcpy into staging,
HIP H2D copy, event before buffer reuse. Those staging buffers do not need CUDA registration.
Copy exact record spans with 256-byte-aligned destinations; transfer matrix-scale words and
per-layer books unchanged into separate helper metadata. Release startup staging after the fill;
never create a second 25–33 GB host copy or HIP-register the 4 GiB CUDA host chunks.

At the measured 6.472 GB/s host→R9700 rate, 25.7–33.2 GB costs **4.0–5.1 seconds bandwidth-only**,
plus CPU staging and allocation. This is an incremental startup estimate, not load-time evidence.
Validate actual bytes/counts, pointer ranges, supported codecs and geometry; verify the upload
with bounded per-record checksum readback compared to CPU checksums of the immutable spans.
This transport check supplements, not replaces, exact codec/oracle tests. Run one small
nonce-bearing request/response before graph capture/service readiness to catch unusable
registration/system visibility. An enabled runtime device/API failure must surface before serving.

The host has 124 GB; routed records already occupy 50.289 GB (~46.8 GiB), with embedding,
checkpoint slabs, RAM tier, OS and table page cache in addition. Incremental startup staging is
32 MiB and maximum exchange memory ~34 MiB, plus small metadata; helper records live in AMD
VRAM, not another pinned RAM arena. Preserve `EngineProgram::Impl`'s available-RAM check and
4 GiB floor, accounting for enabled-only allocations and configured RAM capacity. At C12 with
DFlash, three continuation slots alone are ~5.69 GB. Do not infer safety from physical RAM
minus experts or from phase 0's unloaded ~98 GiB availability. No whole-arena registration test
is needed for this design.

Log the helper PCI ID/name/gfx target, HIP/toolchain version, free/reserved/used VRAM, record
counts and bytes by codec and layer, CALIB route share, fixed/flex primary slots, remaining
codec counts, full-coverage feasibility at zero-flex, mailbox bytes/transport table, upload
seconds and validation result. Keep primary `expert-cache=<fixed>+kv-lendable=<flex>` separate
from helper VRAM; never add AMD bytes to CUDA free memory or KV capacity. `ProgramMemory` and
public memory reporting need an optional helper-device summary, absent/zero when disabled.

## 8. Numerical and lifecycle qualification

The oracle is the complete formula in `include/ninfer/ops/moe_experts.h`: represented BF16 x,
exact signed F3-book/E2M1 decoding, E4M3 block scales, stored FP32 matrix multipliers, FP32 route
weights and shared scale; gate/up shapes `[640,2560]`, down `[2560,640]`. Use one independent
naive FP64 implementation, not CUDA output as the HIP oracle. That oracle must **not** round
its SiLU/product to match a kernel's BF16 staging. Exact BF16 code×block-scale decode and
byte-preserving transfers have independent exact criteria. BF16 WMMA/SIMT reduction, down-input
rounding and intermediate precision are qualified implementation profiles, not new semantic
requirements. No FP8 staging assumption for arbitrary F3 code×scale products.

The Op owner qualifies both codecs and mixed books/scales at real geometry, T=1,4,20,48,
repeated/distinct experts and partial ownership including empty/all-helper routed sets. Check
helper weighted partials, CUDA local partials and the final BF16 join **directly** against their
respective restrictions of the same oracle. Start from existing `moe_experts` output-error
criteria; specify and justify any HIP-profile criterion from cancellation-aware absolute and
relative error, never choose a tolerance just to pass the observed maximum. Stable summation,
finite results, route-slot association and no intermediate BF16 merge are separately checked.

Program qualification extends `tools/parity/qwen3_8_flash_next/{parity.cpp,compare.py}` with the
helper option and matched teacher forcing. Run the existing 4×512 study-reference panel,
split-prefill/decode checks and long decode continuations, with NVFP4 KV as default; add BF16 KV
only to isolate the arithmetic change. Compare helper-on against CUDA-only at **identical C,
W, valid columns, context and forced input IDs**, reporting KL, top-1 and logit error. Use the
existing same-column numerical-floor approach: compare both modes to the independent study
reference and the measured valid-route envelope, and require no extra KL-tail/mismatch
regression beyond that declared envelope. Freeze the criterion before the candidate run. Do
not import the old mean KL 0.00452/0.01683 as a universal per-column tolerance or demand greedy
sequence identity across arithmetic profiles. Include the C1 W2/3/4 verification panel and
C5/C12 packed rounds; graph/eager in each identical enabled profile must agree exactly.

Exact state checks remain exact: zero/partial publication, rejected chains, K0 read-only records,
source-preserving disposable turns, retained/RAM/SSD/current/rewrite/occupy images and cross-lane
live suspension. Run affected Program/Engine real tests with helper enabled, including a decode
interleaved with another lane's >16K prefill and flex reclaim/return; compare full continuations
to an independently executed helper-enabled control, not to a different CUDA arithmetic history.

Repeat the existing server needle battery (standard/multi-key, greedy and p-less, 8K–260K)
and variable-tracking calibration fixtures at 32K/128K/260K. Needle baseline is 100/100;
tracking is not universally perfect: compare the matched scored cases and existing calibration
intervals, including its weak 128K cases. These test changed routed numerics at long context;
plausible text is not an Op or state oracle.

Cross-device nonce tests must cover repeated graph replay, SKIP→work→SKIP, changing T/K/B,
last-slot reuse, delayed/failed helper, missing CUDA graph B and cancellation teardown. Use
external process deadlines for injected driver/wait failures. CUDA compute-sanitizer memcheck
and racecheck cover new packing, shared staging, import/join and captured overlap kernels; add
production-dispatched cases to the sanitizer set. HIP is not instrumented by CUDA sanitizer:
report that limitation and use its independent oracle and exact stress/visibility checks.

## 9. Performance decision and falsifiers

Keep the target plan's historical model in its proper scope:
`replay ≈ 6.0 ms + miss_bytes / 26.8 GB/s` describes C1 Gen4 ordinary decode. At larger C/W,
use that workload's measured non-PCIe cost. This host's newer Gen5 native measurements use
~48 GB/s mapped expert reads or ~53 GB/s transfer modeling; do not charge Gen4 miss cost to a
Gen5 A/B. Phase 0b mailbox timings use the chipset-connected R9700 and remain the exchange basis.

For a changed layer use

`new_layer = dense_layer + split + max(local_experts_with_residual_misses, exchange + hip_experts) + join`.

Sum actual per-layer overlaps and host/graph-boundary costs. Do not subtract all helper bytes
and then assume its work is free, or add exchange to every local kernel despite overlap.
With full residency, residual miss bytes can approach zero after warmup, but helper compute
and synchronization remain a serial dependency at every joined layer.

| Recorded miss workload | Elimination opportunity at 26.8 / 53 GB/s |
|---|---:|
| C1 ordinary, 99–126 MB/token under pressure | 3.69–4.70 / 1.87–2.38 ms |
| C1 W4 trace, 165.7 MB/round | 6.18 / 3.13 ms |
| C4 under 500K KV, 1024 MB/round | 38.21 / 19.32 ms |
| C5 fixed K3, 2259 MB/round | 84.29 / 42.62 ms |

C1 K3's T4 exchange basis is ~1.42 ms per 48 layers, or ~0.79 MB of avoided expert miss bytes
per active layer at Gen4 / ~1.56 MB at Gen5 just to pay 29.344 µs. Helper expert time and overlap decide the
rest. Thus a **modest C1 win is plausible, not guaranteed**; at the observed low-miss C1
12 MB/token case (0.45/0.23 ms PCIe), communication alone may lose. For C5 K3, the ~6 ms T20
exchange basis is small against 43–84 ms of historical misses, making a substantial gain
plausible if helper kernels sustain useful small-expert bandwidth. C5 product default starts
K2, T15, not T20; its miss bytes and helper unique-record reads must be measured separately.
Do not turn the K3 example into a default throughput prediction. An illustrative 20 MB of helper
record reads/layer already takes ~70 µs at the measured 287 GB/s read floor, before unpack/math;
CALIB assignment shares alone cannot establish that volume or a speedup.

Required priority A/Bs, with the same source baseline plus only the helper integration and
separate binaries/build directories (no checkout/reset while another agent works):

- Public Engine `ninfer_qwen3_8_flash_next_parity --mode lanes-bench`: C1 **fixed K3**, then
  C5 **product default** (adaptive K≤3, initially K2). Use the explicitly named DFlash artifact,
  external int8 n-gram table and `out/flashnext/long_tokens.i32`; 4096-token prompts, 512 outputs,
  chunk 4096, super-chunk 16384, **fixed context 8192**, automatic KV and NVFP4 KV. Extend the
  existing Engine-only benchmark's options; do not route inference through private Forward.
  Include a matched C5 500K explicit-KV case to exercise fixed/flex pressure, not a broad sweep.
- HTTP: same `serve_bench.py` technical/story tasks, nonces, sampling and 1024 outputs as
  `docs/performance.md`'s native Flash-Next comparison. Compare client C1 K3 and C5 default,
  recording startup concurrency 8/context 65536/automatic capacity for comparability. A matched
  startup-C5 run may answer a separate budget question, not replace that control. Keep RAM/reuse
  settings identical and record actual K histograms, accepted length and committed tokens.
- At least three alternating paired runs after one warmup for the two priority cases; report
  median/range, decode/drain throughput and HTTP TTFT/ITL/wave throughput. Keep cold upload time
  separate. Fix workload/context/sampler, not probabilistic output hashes. Stop extending the
  panel when it answers whether the opt-in path improves the intended workload.

Instrument per-layer helper rows/unique experts, local hits/residual miss bytes, split/join,
helper import/compute/export and exposed CUDA wait; report mailbox prequeue/drain host time.
Use whole-inference attribution only if these fail to explain end-to-end results; kernel
profiling follows a demonstrated bottleneck. Any expert-kernel tuning follows `tools.kdev`
and the [kernel-iteration rules](kernel-iteration.md), owned by the Op agent.

Falsifiers: registration/visibility or bounded shutdown fails; same-column numeric/state gates
fail; full-coverage steady rounds still read helper records through the CUDA miss path; helper
compute plus exchange consistently exceeds the eliminated/overlapped work; C5's gain vanishes
in HTTP; or C1 K3 is materially slower outside paired-run variation. Do not claim a C1 win on
synthetic signaling or hide startup/drain/prefill regressions. A C5-only win can justify the
explicit opt-in deployment with that limitation documented; a failed priority screen is a
reason to stop and resolve the measured bottleneck, not enable the helper by default.

## 10. Contract updates and implementation packages

This plan does not itself amend the active product contract. Implementation must update these
existing authorities together:

- `AGENTS.md`: retain one NVIDIA primary, one resident model and C1–12 scheduling; explicitly
  permit one startup-selected gfx1201 expert-compute helper for this identity only. Amend the
  single-GPU/all-compute wording narrowly, core raw HIP/mailbox ownership, Op ownership and
  Flash-Next placement/scheduling ownership. This is not a general ROCm port or multi-GPU Engine.
- `README.md`, `docs/cli.md`, `docs/serving.md` and executable help: opt-in build/environment,
  `--expert-helper-pci`, supported identity/device, startup failure behavior, unchanged defaults,
  local-only prefill, RAM/startup costs, numerical profile and qualified performance scope.
  No OpenAI/Anthropic request/response schema extension or protocol compatibility alias.
- Target plan/artifact binding notes: compiled CALIB profile and exclusive decode placement;
  record layout and artifact inventory unchanged. `paged-kv-cache.md` and
  `concurrent-inference-architecture.md`: helper weights are not KV/state, and settled rounds
  include helper completion without changing interruption, reservation or suspension policy.
  `docs/performance.md` records only qualified Engine/HTTP results; code-quality/build notes
  document the optional toolchain and honest HIP sanitizer limitations.
- Tests: CLI/server valid/malformed/empty and OFF-build rejection, wrong-target rejection,
  missing PCI device/startup failure, enabled upload/exchange, exact partition/cache exclusion,
  all-local/all-helper/SKIP, numerical partial/join oracles, graph replay, lifecycle/continuation,
  prefill isolation and serving smoke. Retain OpenAI/Anthropic schema tests unchanged in meaning;
  add an enabled real smoke configuration, not helper JSON fields or source-string tests.

Ordered ownership packages, allowing parallel implementation without shared-file edits:

| Package / owner | Files owned and deliverable | Verification before handoff |
|---|---|---|
| 1. HIP expert + build owner (already active) | HIP Op contract, `src/ops` HIP implementation, its tests, and **all CMake files**. Own gfx1201 compilation/link recipe and opt-in test registration. Agree bank/active-assignment/N interfaces with integrator; no core/target scheduling in the Op. | Independent exact codec and FP64 Op oracle at real shapes; measured expert costs T1/4/20/48; Werror and applicable quality gates. |
| 2. Integrator, parallel with 1 | New `src/core/hip_device.*`, `expert_mailbox.*`; no CMake or `src/ops` edits. Provide a source-list request to owner 1. | CPU compile checks where applicable; dual-runtime nonce/sequence/SKIP/cancel/deadline/idle teardown and CUDA memcheck/racecheck after build wiring. |
| 3. Integrator, after interface agreement | New target `expert_helper.*`, compiled profile/tool, `expert_residency.*`, `program.*`: actual capacity plan, immutable upload, filtered warm start, helper lifetime and memory report. No edits to Op tests owned by 1. | Exact placement/byte budget from independent CPU model, upload transport check, full/no-flex and undersized-capacity cases, unchanged flag-off allocations. |
| 4. Op owner, while 3 proceeds | CUDA partition/local-partial/join and cache eligibility in contracts, wrappers and launchers; corresponding existing/new Op tests. All `src/ops`, `include/ninfer/ops` and Op-test changes stay with this owner. | Full/partial FP64 oracle and exact masks; filtered admit/frequency/stage model, unaffected default/window/fill/commit behavior; captured overlap memcheck/racecheck. |
| 5. Integrator, after 2–4 | `forward.{h,cpp}`, `verify_impl.h`, `speculative_impl.h`: stable storage, graph seams, prefill exception, begin/finish/cancel and publication error fence. Integrator owns target runtime files exclusively. | Same-column Program parity, C1/C5/C12 and K0–3, graph/eager, prefill interleave, flex transitions and exact state/tiers. |
| 6. Integrator, after 5 | Engine/CLI/serve option plumbing and diagnostics; parity benchmark option; active docs. Own product/target tests only **after** their file list is agreed with Op owner; CMake registration still goes through owner 1. | OFF and ON builds, ON-without-flag no HIP initialization and unchanged inference, parser/startup errors, Engine real lifecycle and OpenAI/Anthropic smoke. |
| 7. Integrator with Op-owner fixes only at measured bottlenecks | Priority Engine/HTTP A/Bs and active numerical/performance evidence; no speculative scheduler retuning. | Numerical panel, needle/tracking, paired speed results and limitations; affected fast units/clang-tidy/format/lint gates. Full merge gates only for an actual merge. |

Until handoff, the concurrent Op/CMake/test owner keeps those files exclusively. Resolve API
changes through that owner, not overlapping edits or a second helper Op. This design task writes
only this document; it does not execute any of the implementation or GPU-verification packages.

## 11. Second stage: both-card prefill, streaming load and a cold host residue

Starts after the decode qualification of §8 passes. It changes the product contract above:
with a helper, prefill also runs on both cards, and host RAM holds only the experts neither card
keeps. The CUDA-only configuration keeps its pinned 50.3 GB store and streaming windows; only
its load path changes (11.3).

### 11.1 Three-tier placement

Placement stays static for the Program's lifetime and is still computed after the KV, lane,
workspace and helper reservations. Counts come from the compiled profile, generated from the
served route-profile JSON.

- **P, pinned on the 5090:** the highest-count records filling every fixed slot of both codec
  pools (the 11% NVFP4 split is unchanged). Admission never evicts P.
- **H, on the R9700:** as in §3, taken outside P in descending `count / record_bytes` until the
  31 GiB budget is full.
- **T, the tail:** everything else. It lives in a pinned, device-mapped **host residue** that
  holds only T. On the 5090, T uses only flex slots, filled on demand and by the background
  refill of `flash-next-flex-reclaim`. When a T record is not resident, `moe_experts` reads it
  through its mapped host address, as it does today. NVFP4 records of T are always read through
  mapped memory, because flex slots are F3-only.

When P and H together cover all 24,576 records, T is empty: no host residue and no flex demand.
Small KV configurations reach this (§3, capacity table). At `--kv-capacity 1000000` with 8
lanes, the planning estimate is P about 12 GB, H 33.3 GB and T about 5 GB, about 2,400 records
and 0.14% of route assignments in the VM profile. The startup log reports the actual values.
`plan_expert_helper` computes the three tiers with exact byte totals per tier and codec
(`primary_*`, `helper_*`, `tail_*` of `ExpertHelperPlacement`); `local_eligibility` keeps its
meaning, P union T.

`ExpertResidency` owns a pinned fixed region (P, written once at load and never a victim) and the
flex pool (T only). Admission, staging and frequency state cover only T. The P and H masks are
uploaded once, next to the existing helper eligibility bitset.

### 11.2 Prefill on both cards

Prefill uses the grouped route of the helper expert Op (`kMoeHelperExpertsMaxTokens` = 16,384,
commit 64d3dfda on `flash-next-helper-prefill`). For each sub-chunk of a layer, the 5090
partitions helper-owned routes and sends the BF16 activations, IDs and weights over the
mailbox's DMA path. The R9700 returns a **BF16** partial for prefill (FP32 stays for decode). The
5090 joins it with its local partial.

- **Overlap.** Prefill is layer-major over the super-chunk (plan, "Prefill runs layer-major"). A
  sub-chunk's MoE output feeds only the next layer, so the helper work of sub-chunk s overlaps the
  5090's mixer and local experts for sub-chunks s+1 and later. Per layer, only the last
  sub-chunk's round trip is exposed. Link cost at 4,096 tokens is about 21 MB each way, about
  3.2 ms per direction at the measured x4 rate; the R9700 needs 1.83 ms per layer at that size.
- **Windows.** In helper mode, the streaming windows and their per-super-chunk copies (about 1 s
  for each windowed super-chunk) are removed. P is resident; H runs remotely; T is read through
  mapped memory. Prefill no longer evicts or borrows decode cache slots.
- **Numerical profile.** The BF16 return adds one BF16 rounding of the helper partial before the
  join. This is a declared implementation profile for prefill only, and it amends the §2
  "no intermediate BF16 merge" boundary of the criteria for this route. It is gated by the same
  frozen OFF-versus-ON method as §8's tonight gate: teacher-forced 4x512 panel, mean/p99 KL and
  top-1 against the same reference. Decode remains FP32-joined.
- **Interruption** stays at 16,384-token super-chunk boundaries. A helper prefill round
  completes or poisons like a decode round. There is no CUDA-only fallback.

### 11.3 Streaming load

No artifact page cache and no 50 GB temporary. `artifact::Reader::stream` reads a file range
sequentially with `O_DIRECT` through four reusable pinned bounce buffers of 64 MiB
(`PinnedReadBuffers`), one stream at a time (the host's single-stream `O_DIRECT` rate is 5.3 GB/s).
Unaligned range edges are read as whole 4096-byte blocks and trimmed, so a consumer sees exactly
the requested bytes, in order, in chunks that are a whole number of records. A buffer is refilled
only after the completion its consumer returned has run: a device copy completes on a CUDA event
recorded after its copies, a CPU copy on its future. The target decides each chunk's destination:
a P slot (CUDA H2D on the copy stream), an H bank (HIP H2D), or the host residue (CPU copy).
Peak extra host memory is 256 MiB plus the residue.

- Non-expert tensors (`artifact::materialize`) use the same reader for device tensors, the mapped
  host embedding and retained resources; only the small directory is read through the mapping.
- The CUDA-only configuration streams into its pinned store the same way
  (`fill_expert_store`: each chunk's records are copied by a worker thread while the reader fills
  the next buffer), which removes the page-cache copy of 55 GB but keeps the 50.3 GB store.
- Verification: the existing bounded per-record checksum readback, extended to P slots and the
  residue, against CPU checksums of the read file bytes. The reader itself is checked bytewise
  against a buffered `pread` of both record tensors (`ninfer_artifact_direct_stream_real_test`).
- Load time: measure the host's native sequential read rate of the artifact. The VM's 2-minute
  read is a virtio `cache=none` figure, not a host one.

### 11.4 Packages

| Package | Deliverable | Verification |
|---|---|---|
| 8. Placement v2 and residue | P/H/T placement, pinned P region, T-only admission/flex, residue store; AGENTS.md and plan contract wording | CPU placement oracle (exact masks/bytes at small KV and 1M/8 lanes); admission never evicts P; full-coverage case has no residue; decode exact versus §8's qualified helper build when T is empty |
| 9. Streaming loader | `O_DIRECT` bounce-buffer reader for all tensors, direct dispatch to P/H/residue/store | Exact record checksums on every destination; peak RSS and load time on the host |
| 10. Both-card prefill | Forward prefill rounds, mailbox prefill transport, BF16-return variant of the grouped Op (owner: Op/kernel owner), window removal in helper mode | FP64 Op oracle for the BF16-return route; frozen OFF-versus-ON panel gate; prefill/decode state exactness, interruption and tiers; prefill tok/s at 1.7K, 16K and 64K tokens, ON versus OFF |

Package 8 precedes 9 (the loader needs the masks). Package 10 can proceed in parallel with 9.
The idle-timeout eviction and background refill (branch `flash-next-flex-reclaim`) are separate
and apply to both configurations.
