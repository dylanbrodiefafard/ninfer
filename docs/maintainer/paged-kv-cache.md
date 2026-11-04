# FP8-K/INT4-V paged context store

This document defines growing key/value storage and publication for Qwen3.8-27B on R9700, including attention access, retained prefixes, host-RAM spill and SSD storage. The attention and Op references define kernel arithmetic; `concurrent-inference-architecture.md` defines scheduling.

## 1. Product contract

The Text and MTP growing caches have exactly three persistent planes:

- K: one OCP FP8 E4M3FN byte per feature;
- V: signed canonical INT4, two codes per byte; and
- V scale: one IEEE FP16 scale for each fixed feature group.

There is no K scale, K mean, homogeneous cache dtype, runtime cache-format selector, or alternate growing-cache path. Represented BF16 K/V inputs are quantized only when appended. Attention reads the stored representation directly; it never gathers the cache into a request-contiguous buffer.

The product is one resident model on one `gfx1201` wave32 device with startup-fixed concurrency `C=1..16` (`C=1..8` under DFlash). A single request may consume most of the shared physical page pool. Active, retained, and speculative provisional state use the same entitlement accounting.

The cache owns storage and physical mapping, not model scheduling. It does not interpret request priority, MTP acceptance, DFlash selection, or GDN state. The target runtime owns those semantics and changes the published cache frontier only through the transaction API.

## 2. Granularities and geometry

Allocation, publication and reuse have different boundaries:

| Granularity | Meaning | Contract |
|---|---|---|
| allocation | payload acquired or released together | one 64-token page group |
| publication | logical positions visible to consumers | one token |
| reuse | positions with complete continuation state | target checkpoint only |

A page boundary is neither an attention boundary nor a reusable-prefix boundary. A valid frontier may end at any token inside a page. Truncating cache publication does not prove that the rest of the model state can resume there; the target must also own a matching complete checkpoint.

For the Qwen3.8 Text geometry, `D=256`, `Hkv=4`, `Lfull=16`, page size `P=64`, and provisional value group `G=16`:

```text
bytes_per_token_per_layer = Hkv * (D + D/2 + 2*D/G)
                          = 4 * (256 + 128 + 32)
                          = 1664 bytes

Text bytes_per_token      = 16 * 1664 = 26624 bytes
Text page_group_bytes     = 64 * 26624 = 1.625 MiB
MTP page_group_bytes      = 64 * 1664  = 0.1015625 MiB
```

`G=16` is the current whole-attention timing leader, not a finalized public artifact decision. The real paired quality, capacity, and whole-inference Pareto gate must compare G16 and G32 before the losing branch is deleted.

## 3. Pools

Startup constructs a fixed set of typed pools:

```text
ordinary or DFlash Engine:
    Text FP8-K/INT4-V pool

MTP Engine:
    Text FP8-K/INT4-V pool
    MTP FP8-K/INT4-V pool
```

Text and MTP are physically separate because they have different layer counts and publication frontiers. Each pool nevertheless uses the same allocator semantics and current codec profile.

The selected DFlash2 target has no DFlash growing full-context layers. Its local and staging state is fixed BF16 cyclic storage with separate ownership; the rewrite checkpoint's cyclic lane lives in a lane-owned pinned host image with the rewrite GDN image. GDN convolution and recurrence state, Vision intermediates, and operator-transient query K/V are also outside this store.

## 4. Capacity resolution

Let:

```text
S      = EngineOptions.max_context
C      = EngineOptions.max_concurrency
P      = 64
L      = ceil(S/P) logical pages per allocation
M_min  = max(L,C)
M_max  = C*L
M      = resolved Text physical page-group count
```

An explicit capacity converts tokens to `ceil(tokens/P)` and must lie in `[M_min,M_max]`. It must also name at least `S` tokens. Automatic capacity evaluates the complete target layout after weights are resident:

```text
B(M)   = persistent state + workspace + request transient + Device Graph allowance
         + SSD restore device storage + physical allocation bounds
B_min  = B(M_min)
B_step = B(M_min+1) - B(M_min)
M      = min(M_max, M_min + floor((F-R-B_min)/B_step))
```

`F` is device memory available after weights and completed RAM-cache registration: the smaller of HIP availability and physical VRAM total minus usage from the matching PCI device's amdgpu sysfs counters. Required physical counters must be readable; HIP alone can overstate the graphics-inclusive reserve. Startup availability uses the same query. `R` is the configured sizing headroom (`--kv-capacity-headroom`, default 64 MiB). `R` is exclusively for external GPU users and display activity. NInfer-owned storage and its lazy driver allocations belong in `B(M)`; the runtime reservation includes both. Raise `R` when a desktop or another process needs more memory. Startup failures in automatic mode name the option. The same production layout builder supplies `B_min` and `B_step`; the common resolver does not duplicate model dimensions or bytes-per-token formulas.

Enabled SSD caching reserves its device restore window and per-pool scatter descriptors in `B(M)`. The production 16-reader window has 32 packed-page slots: G16 Text pages require 52 MiB of staging, plus a 2-MiB physical allocation bound for each scatter table (54 MiB with DFlash, 56 MiB with MTP). Host file-alignment padding belongs only to the pinned host I/O window. This reservation is independent of physical KV page count and preserves the affine curve; it does not consume the external headroom.

For the measured production `R9700Lq4` build at `S=262144`, adaptive DFlash K7 with the optimized proposal head, no Vision, and prefill chunk 2048, the target bounds the complete immutable Device Graph residency by its measured C1..8 inventory. This includes one executable for every captured profile and the lowered persistent decode definitions. Other startup selectors and qualification builds use the conservative immutable-executable inventory bound.

The target separately reserves 2 MiB for startup driver residency and a C-specific main-stream bound of 6/6/4/4/2/4/4/6 MiB at C1..8, plus 2 MiB for the SSD scatter kernel when enabled. The main-stream bound covers both lazy kernel residency and the measured aggregate physical residual beyond logical arenas, HIP graph accounting and the separate 4-MiB arena allocation bound. The C6/C7/C8 physical-counter correction is owned by NInfer, not external headroom; its attribution and qualification are recorded in the 2026-10-16 GDN follow-up in `docs/performance.md`. Physical and HIP graph deltas differ, so the residual is not claimed to identify an exact driver allocation. C9..16 (MTP or no speculation) reserves 32/32/32/34/34/34/34/34 MiB, measured on the MTP and ordinary routes (2026-11-04 C9..16 residency in `docs/performance.md`).

The low-residency calibration requires `DEBUG_HIP_GRAPH_CLASSIC_PATH=0`, `ROC_ENABLE_LARGE_BAR=0`, `ROC_SKIP_KERNEL_ARG_COPY=0`, and `HSA_DISABLE_FRAGMENT_ALLOCATOR=1`, as set by Compose. Immutable graph arguments reside in pinned host memory and cached packets replay in bulk; the same preset without those settings retains the conservative device-argument-pool reservation. Other presets use the full inventory bound.

Byte-identical persistent phase images share fixed-address arenas; immutable executables and the startup residency guard remain in place. Startup rejects graph residency above the planned bound. See `docs/performance.md` for the measured cells and capacity qualification.

The token embedding is materialized in pinned host memory, not in the device weights, so `F` includes its size (675,430,400 bytes for the production Q4G64 table). Persistent state includes the fixed device region that receives host-staged prompt embedding rows: one compact image for `min(prefill_chunk, S) + 1` ids, about 11 MiB for the Q4 table at chunk 4096 (`docs/maintainer/qwen3.8-27b-artifact.md`, token embedding placement).

The runtime reports configured `S`, resolved `M*P`, page counts, reservation bytes, headroom, and planned slack. Tail capacity created by page rounding is storage padding and never permits a sequence frontier beyond `S`.

The MTP pool has `M + C*ceil((K-1)/P)` physical page groups for startup-planned maximum draft window `K` (the configured K without adaptive drafting, otherwise the captured-set storage ceiling). The extra physical headroom supports provisional MTP growth; it does not increase any sequence's logical context ceiling. DFlash adds no growing pool.

### Historical all-Q4 capacity measurement

The superseded schema-v19 reports bind the qualified fixed physical plane candidate used by the earlier model campaign (token-fastest K, feature-fastest V, feature-fastest V-scale), the byte-identical all-Q4 artifact, and the compiled cache group. The strict automatic-capacity validator resolves:

| C | G16 tokens | G16 constraint | G32 tokens | G32 constraint |
|---:|---:|---|---:|---|
| 1 | 262,144 | model context | 262,144 | model context |
| 2 | 524,288 | model context | 524,288 | model context |
| 3 | 570,304 | device memory | 593,152 | device memory |
| 4 | 558,080 | device memory | 580,416 | device memory |
| 5 | 545,856 | device memory | 567,680 | device memory |
| 6 | 533,632 | device memory | 555,008 | device memory |
| 7 | 521,408 | device memory | 542,272 | device memory |
| 8 | 509,184 | device memory | 529,536 | device memory |

The retained historical schema-v12 directories are `profiles/bench/pareto-capacity-markerfree-layout-all-q4-g16-20260903/` and `profiles/bench/pareto-capacity-markerfree-layout-all-q4-g32-20260903/`. Their manifest SHA-256 values are `be1d5727948512560f28a7e149547714f9b80dca96169646a2eef0310a883feb` and `c7e5020b03cd104440011fcc17095e3166defdc59045274192e6b930b027d922`; both bind artifact SHA-256 `19d029a89c1ef1cf87420067555021a7c7b435c31a92bea7c64ccf42c03d80e9`. The respective benchmark SHA-256 values are `86763d6d3ac2ff8fc27a3815f3816aaba44a11acfaf4ebf54ca5d8991ddc7d65` and `2c9bb5a64f0e25a307f3f2b783a527ed6b42d1671c6e826d7ca98545052e9b31`.

These once superseded every earlier capacity generation, but are themselves historical after the C=1..4 product-cap migration and cannot enter the current selection. Fresh exact C=1..8 capacity matrices are required for both eligible recipes, both cache groups, and each candidate execution profile. The still earlier generations are the `pareto-capacity-layout-*` pair with always-on nested ROCTX markers, the completed G16 and interrupted G32 `pareto-capacity-unprofiled-layout-*` intermediate runs, and the schema-v11 manifest/schema-v18 report `pareto-capacity-max-*` pair. All old raw evidence remains unchanged as historical evidence; none is valid input to the current selection frontier.

## 5. Three-plane physical layout

Every layer has three required planes. A page-group ID identifies the same 64 logical positions across all three planes and all layers in that pool. There is one I32 block-table row per allocation, not a table per plane, layer, or head.

Each plane independently selects one of two closed physical orders:

- feature-fastest, page-major; or
- token-fastest, head-major.

The startup profile fixes all three orders. Requests and runtime modes cannot select layouts. The current provisional profile is:

| Plane | Physical order |
|---|---|
| K FP8 | token-fastest, head-major |
| V INT4 | feature-fastest, page-major |
| V-scale FP16 | feature-fastest, page-major |

The generic pool records plane extents, element type, alignment, slab order, and intra-page order. The typed wrapper validates exact K/V/scale geometry before exposing a layer view. Storage contains no optional fourth plane.

For a logical position `p`:

```text
logical_page = p / 64
token_offset = p % 64
physical_page = block_table[logical_page]
```

The plane layout then maps `(feature-or-group, token_offset, head, physical_page)` to bytes. The same physical page ID is used for K, V, and V scale. Page IDs are pool-local I32 values; no device pointer is stored in the table.

The layout sweep may instantiate all eight independent plane-order combinations and G16/G32 in qualification tools. Production startup instantiates only one compile-time profile, currently the provisional G16 profile above. After the real model gate, losing groups and layout-dispatch branches must be removed.

## 6. Allocation, entitlement, and mapping

A `PagedKVAllocation` owns:

- a page entitlement charged immediately against its pool;
- an ordered logical-to-physical page vector;
- at most one bound block-table row; and
- a mapping generation incremented whenever the mapping or row identity changes.

Reservation guarantees capacity but does not materialize pages. Materialization obtains pool-local page IDs, zeroes the new page groups, and publishes their IDs to the bound table row on the ordered stream. Mappings need not be physically contiguous. Duplicate physical IDs within a production allocation are invalid.

Entitlement, materialized-page count, and published token frontier are separate values. Shrinking publication need not immediately release a partial tail page. Trimming complete trailing pages invalidates old reads and returns their IDs to the pool. Recycled storage is not visible until a new owner has written and published it.

One sequence state owns independent Text and optional MTP allocations plus their publication objects. Ordinary reuse transfers retained ownership; disposable reads copy into independent pages. The current product does not implement page reference counts, copy-on-write fan-out or shared writable prefixes. Settled active continuations may suspend to secured RAM as described below; they never share writable pages with a new resident owner.

With RAM off, request entitlement covers its full effective output lifetime. Enabling RAM automatically funds the full prompt plus output windows of at most 8192 tokens, with target-reported physical headroom and independent per-pool rounding. The advertised output/context limit remains unchanged. A funded window can extend atomically; failed extension blocks that lane without terminal output while peers continue. Reserved pages and mapped pages remain distinct. DFlash verification writes and optional MTP provisional positions must fit before the next unit even when its accepted output is bounded by a smaller window.

## 7. Publication state and transactions

Each sequence/pool publication records:

- `valid_frontier`;
- health/poison state;
- whether a transaction is open; and
- a monotonically increasing transaction generation.

Only one mutation transaction may be open for a publication. Opening a transaction invalidates previous layer-read capabilities. The caller supplies fixed-address device status and position workspace; segmented MTP transactions also supply a cursor. Cache Ops do not allocate memory, stage hidden positions, or synchronize the host internally.

### 7.1 Ordinary append

An ordinary append represents a contiguous suffix beginning exactly at the closed frontier. One transaction launches every layer exactly once on one ordered HIP stream. Each layer consumes represented BF16 K/V, writes the three physical planes, and exposes only a same-stream pending read. Commit is legal only after every layer launched, the stream completed, and device status was zero. Commit then advances the frontier once for the whole pool.

### 7.2 Device-prefix and segmented MTP append

A device-prefix transaction captures a fixed maximum panel plus a device I32 active count. Zero is valid and produces no visible append. The host does not read the count between graph nodes.

One MTP round uses a segmented transaction spanning alignment and every autoregressive segment. A device base frontier initializes a fixed cursor and remains immutable for the lifetime of that transaction. Speculative acceptance advances a separate fixed-address frontier vector; it must not reuse or mutate the transaction base storage. Each segment validates and advances the private cursor on the ordered stream. A dynamic device table-row selector is permitted only within the allocation-owned row set. Eager and Device Graph routes use the same addresses and one post-round status/cursor resolution. Resolution publishes the complete Text verify suffix and only the licensed MTP suffix.

### 7.3 Failure

The append codec ORs failure bits for invalid positions/counts/addressing, nonfinite represented inputs, or unrepresentable FP16 V scales. A failed or abandoned launched transaction:

- does not advance the public frontier;
- closes every pending read;
- poisons only the affected sequence publication; and
- prevents stale written bytes from entering the readable domain.

Host validation failures before launch leave publication and bytes unchanged. Recovery uses a known-good checkpoint or destroys the sequence; it never edits the frontier around an unresolved device failure.

## 8. Read capabilities and attention

Attention cannot construct a cache read from raw pointers. It receives a non-forgeable `PagedKVLayerRead` containing the exact allocation, layer view, bound row, mapped-page count, mapping generation, transaction generation, and visible frontier.

A committed read is valid only while publication remains closed with the same generation and frontier. A pending read is created only after that layer's append launch and is valid only on the same ordered stream. This permits a decoder layer to attend newly written K/V before the all-layer transaction commits without exposing provisional state elsewhere.

Mapping materialize/trim/bind/unbind, a new transaction, commit, abort, compaction, or row rebinding invalidates older capabilities. Cross-stream pending reads and capabilities from another allocation are rejected before dispatch.

The attention leaf traverses logical positions, translates each page once per shared work unit, decodes FP8 K and signed INT4 V with FP16 scales, and evaluates causal or packed-tree visibility. Malformed device positions/table rows make only the affected output row conspicuously NaN. A cached-only read never writes cache storage.

## 9. Compaction and speculative rollback

Compaction is a transaction over every layer. It copies the exact packed three-plane representation, not dequantized values. The caller supplies a retained prefix and selected suffix path with unique, in-range positions. Commit publishes the compacted frontier only after all layers and device status pass.

When accepted tokens already occupy their final contiguous positions, zero-copy rollback uses publication truncation instead. Bytes beyond the retained frontier may remain in a tail page but are invalid until a later append fully overwrites the matching K code, V code, and V-scale groups.

MTP acceptance, cancellation, and prefix reuse therefore change cache visibility through a single transaction authority; direct frontier repair is prohibited.

## 10. Prefix retention, host RAM, and SSD storage

A reusable bundle includes every state component needed to continue, not just Text KV:

- Text allocation, publication, and exact physical mapping;
- optional MTP allocation/publication;
- GDN convolution and FP32 recurrence state;
- hidden/replay/checkpoint state; and
- fixed DFlash cyclic state when that backend is active.

The optional pinned-host tier holds an ordinary FIFO of completed conversation images and opt-in retained system-prefix LRU entries, with exclusive restore claims. Ordinary restore consumes the image; a disposable restore releases its claim and retains the original independently of the destination device lane. Capture is D2H on the copy stream and keeps source pages mapped until its event completes. Restore validates fixed codec and pool/state geometry before H2D, chooses a free lane/mapping, writes exact physical bytes, then orders compute after the copy event immediately before prefill. A later payload/checkpoint parse failure is fenced before the Program falls back to cold prefill; no incomplete restored state is published.

RAM and disk restore read only the state sets retained by the planned reuse path. Ordinary frontier reuse installs current GDN, DFlash cyclic state and tail hidden. Turn or response checkpoint reuse skips that frontier state and reads the rewrite image instead. For an exact prompt hit, prefill installs checkpoint hidden; a suffix prefill produces a new tail hidden. A request dropping an unused rewrite checkpoint skips its image and clears its metadata, so cancellation cannot select unread state. Context-head selection still installs only heads at or before the restored base. The retained KV planes and their copy/event lifetime rules are unchanged. Disk qualification observes six state decodes when both frontier and rewrite are kept, and three for checkpoint-only or frontier-with-drop reuse.

Resident and RAM reuse candidates must satisfy the selected backend's readiness requirements before longest-prefix ranking; an unusable longer candidate cannot hide a usable shorter one. An exact ordinary hit requires valid tail hidden state. Each staged checkpoint's lifetime event covers both its capture and subsequent H2D readers, so releasing the head waits for its last use, not just its initial D2H capture. Retired RAM storage transfers ownership only after its retirement record is allocated. Pinned-arena allocation failure leaves existing allocations/free space intact; releasing valid blocks requires no metadata allocation.

The RAM image version binds an FP8-K/INT4-V semantic fingerprint containing codec version, page size, physical/logical capacity, layer/head geometry, value group, and all plane orders. A mismatch rejects before any destination byte changes. Text and MTP fingerprints are independent.

`--kv-ram-system-prompts N` enables retained snapshots of the exact leading system-prefix frontier inside the same arena. The frontend marks the boundary before any first-user content, and the tokenizer verifies that its encoding is an exact prefix of the full prompt without changing input tokens. Prefill splits at the boundary; capture stores only its leading mapped KV pages and current GDN/backend state, with no stale later checkpoint. Ordinary and DFlash snapshots omit the tail hidden; their strict suffix prefill supplies it. MTP retains the last normalized Text hidden and only MTP validity through F−1: the bridge column at F−1 depends on lookahead token F and is regenerated from the new user token before suffix prefill. The copy stream waits for the producing compute, and subsequent compute waits for that entry's D2H event before mutating its partial page or recurrent/cyclic state. All claims, H2D readers and retired blocks retain the existing event lifetime rules.

A successful system restore releases its claim without retiring the source and touches LRU recency. Count replacement evicts the least recently restored unclaimed system entry; byte pressure first reclaims ordinary entries through their normal RAM/disk policy, then older system entries. Ordinary spill enumeration excludes system anchors, so they remain RAM-only and ordinary FIFO pressure cannot evict them. At most N indexed system entries are retained, with no capacity beyond the existing host arena. Optional capture failure preserves request progress. The planner continues longest-ready-prefix ranking across all retained classes. A missing system image behind an already longer selected frontier does not force a cold restart just to populate an auxiliary entry.

Host capacity is fixed by `--kv-ram-capacity`; `off` disables retained FIFO spill and active RAM suspension, but not live-lane checkpoint state. Optional completed-prefix captures that do not fit may be dropped. With RAM on, live image credits and suspended continuations share the configured arena and take precedence over optional completed/system entries. Live images are private continuation ownership, excluded from prefix lookup, FIFO eviction and disk spill.

Every resident outside the completion cohort has allocatable image credit through its granted window end; an already suspended request keeps its exact image. Credit includes required current/rewrite/checkpoint/backend state, packing and alignment, and bytes still held by transfers. Free-byte sums must not promise an image that fragmentation prevents allocating. The cohort's target-authoritative completion envelopes jointly fit every GPU pool and lane axis. Those envelopes include bounded native recovery suffix growth only for enabled retry paths; ordinary raw, disabled-recovery and public capped-Close requests pay no speculative recovery charge. Cohort membership is a future completion proof, not an upfront GPU reservation: members still acquire windows, and backed peers can evacuate so all members finish without extra RAM. A restore claims future cohort capacity before transfer commit and releases its existing image backing only after all restore readers finish. Active handoff first secures the departing member's image and the successor's future feasibility. A queued replacement that fits the cohort claims that future capacity through victim capture and physical admission without requiring a redundant RAM image credit; cancellation or failed admission releases the claim.

Unused future entitlement is reclaimed before suspension when it can satisfy a concrete admission or growth. Target planning retains mapped current/speculative pages and an immediately useful next output window; shrinking never drops current state or reduces lifetime output/context bounds. The scheduler searches target-produced window vectors, commits only the required slack release, and shrinks backed residents' arena credits accordingly. Physical page rounding and backend verification headroom remain target-owned. Suspended-window planning retains the captured mapped-page floor, including settled speculative tails, so a shrunken logical window cannot underfund exact restoration.

Only settled committed state is captured, including exact three-plane bytes/publications, GDN, backend cyclic state, hidden/anchor and required recovery checkpoints. Source GPU pages and lane remain owned until capture fences complete; resume binds free pages and potentially a different physical lane, restores the current frontier, then enters decode without prefill, resampling or output replay. Transfer failure preserves the source or secured image. A single blocked lane waits while peers execute unless a funded queued replacement can use it; two blocked lanes or no executable work triggers a minimal feasible victim plan. The detailed bounded fairness and completion-cohort lifecycle are owned by §5.8 of `concurrent-inference-architecture.md`. This path remains under qualification. The initial one-protected-request C8 experiment showed no throughput benefit because RAM credits limited residency to the same five requests as full reservation; the completion-cohort correction requires its own matched measurement.

Optional-cache occupancy reports indexed live cache images; retired in-flight buffers and unmaterialized live-image credits still consume allocator capacity and cannot be treated as free space merely because they are outside that occupancy figure.

Consume, a rollback discard, and capacity eviction erase a host entry without waiting and retire its block. A retired block is freed once its copy and block events complete and its last I/O pin drops (a disk spill still reading the image, or a disk worker's pending-copy snapshot borrowing its event), so a RAM restore claim neither cancels nor waits for a spill of the same entry. A RAM restore starts only once the entry's own capture copies are ready; the copy-hold polls that while other lanes decode. Retired blocks and idle-spill I/O pins do not count against the capture fit decision; a blocking reclaim waits for a victim's copies or a retired block's fence, and a non-blocking reclaim selects only victims whose copies have landed and otherwise defers the admission. Harvest bills only completed copies.

Rewrite-image and ladder-head copies between pinned host buffers run as 4 MiB host callbacks on the tier's startup host-copy stream, ordered only after each image's fence, so they overlap the entry's KV D2H/H2D. A middle-head hit installs only the entry's heads at or before the restored base into the Program's startup checkpoint pool; those copies wait for their own fences and are excluded from the entry's copy fence, which gates the lane's first prefill chunk, while a separate block fence keeps the retired block allocated until they land.

Disk restore decodes saved heads straight into pool heads handed in with the restore target, after each head's previous DMA or host-copy owner has finished.

Direct disk reads use ordinary aligned staging before copying into pinned GTT: one reusable page-record buffer per reader thread and at most 4 MiB per raw-state read worker. Header and CRC checks precede page publication; state chunks retain their logical CRC order. Page-read scratch remains allocated for reuse until its reader thread exits; state-worker scratch is released after that state read.

A closed preserve-off turn is captured cut at its turn checkpoint. The request stopped on a stop token or string without a tool call, and its `TurnClosure` checkpoint `F` is its own generation opener (frontend `generation_opener`; a checkpoint inside a tool loop sits at the loop's first opener and is not marked). The next preserve-off prompt re-renders that reply without reasoning and diverges after `F`, so no frontier past `F` is reachable. The RAM entry, and therefore its disk spill, stores the lane as of `F`: ledger, identity, Text and backend KV end at `F`; the current GDN state, DFlash cyclic lane and tail hidden are the checkpoint's; no rewrite set and no head past `F` is stored. The next turn appends at `F` with the reuse length `restore_turn_checkpoint` had, so each entry holds one state set. Such a request appending from a cut RAM/disk entry writes no automatic turn-rollback head; a resident append writes it as usual. The VRAM retained lane is unchanged. Cost: switching that conversation to preserve-on, or continuing the reply as an assistant prefill, re-prefills the reply.

RAM image version 6 binds both semantic and per-plane physical fingerprints. Capture explicitly returns captured, needs-eviction, or dropped; the caller owns spill/eviction decisions. Optional pinned allocation failure leaves the request able to proceed without retained cache capture. The startup pinned arena uses driver-backed HIP/GTT host memory and is zero-initialized; its preparation can overlap artifact materialization. Device startup fixes `HSA_USERPTR_FOR_PAGED_MEM=0` with normal SVM/unified addressing before HIP initialization, so these buffers do not use the movable USERPTR backing implicated in compaction stalls. RAM KV capacity, checkpoint slabs, mapped weights, transfer staging and runtime host buffers share the device's GTT budget as well as host-memory resources. Each allocation checks current GTT availability; HIP allocation remains authoritative under concurrent usage. Engine startup fails clearly when its requested capacity cannot be allocated; it never silently shrinks the RAM tier or falls back to registered pageable memory.

`--kv-disk-capacity` and `--kv-disk-location` enable an inclusive persistent SSD tier; a nonzero RAM tier is required. Each RAM capture requests a write-behind spill on the disk worker, so entries usually become durable while other requests decode.

When a capture needs RAM, the oldest unpinned disk-durable entry is evicted first. Without one, the in-flight write-behind spill, or else the oldest savable entry, is spilled first: synchronously when no other lane is decoding, otherwise at emergency priority on the disk worker while the admission stays queued, claims and captures nothing, and the other lanes keep decoding. The admission retries once that spill commits, so decoding never waits for a disk write and an entry is dropped unsaved only when the disk cannot store it; when disk bandwidth is the limit, admission (TTFT) waits for it. Entries captured earlier in the same admission attempt, which the deferral rolls back, are never the spill it waits on nor the unsaved drop, so the retried attempt cannot defer on its own capture. A disk restore whose setup would wait for another entry's window reads holds admission while other lanes keep decoding.

Disk version 8 stores raw or Zstd-compressed exact logical-page payloads plus complete checkpoint state. Packing respects each of the three planes' slab and intra-page order; scatter restores represented bytes without dequantization. Durable identity binds model, weights, artifact, fixed codec semantics and Vision-aware prefix identity, but excludes pool allocation capacity so a compatible restart may resize the pool. RAM images retain their stricter physical fingerprint; disk restore reconstructs the current physical mapping rather than importing old page IDs.

Ranking selects the longest complete usable frontier; ties keep VRAM before RAM before disk. Bounded reader/staging resources may overlap validated page copies with further reads, but a lane is not published until all state owners and the copy event complete. Cancellation drains I/O and releases the pinned entry generation without deleting its durable source.

Emergency spill needed for admission excludes restore payload reads; idle spill rechecks its epoch and RAM residency. A blocking admission waits only for its reclaim victim's spill (or a reclaim spill already at emergency priority), never for another entry's write-behind batches.

Capacity counts actual pack-file lengths, including holes and retired extents still leased by readers. Free extents are coalesced per segment and reused before append; capacity eviction must release physical room. Pack metadata checkpoints rotate independently of the stable pack store. Metadata and filesystem headroom remain outside the configured payload budget.

Only the Engine grants idle maintenance after all requests and copies settle, and it revokes permission before execution resumes. Write-behind permission alone does not permit relocation. The worker trims free tails and moves objects toward lower offsets in paced 1 MiB chunks, checking cancellation and foreground work between chunks. Index lookups and claims continue while payload I/O is unlocked.

Each relocation syncs its target before journaling a durable Move and handing off the runtime location. Old extents retain reader leases until their last read completes. Replay honors Put/Delete/Move records; a failed or cancelled move leaves the original location authoritative, and ambiguous journal rollback freezes reuse. Shutdown drains writes without launching a whole-store copy.

Artifact identity hashes only target-selected tensor descriptors and payload ranges with SHA-256, independent of inode, timestamps and file placement. Text state producers and Vision are included, Text output/proposal heads are excluded; native MTP includes its complete block, while DFlash includes only context feature/normalization and projected K/V dependencies. Disk identity excludes DFlash cyclic lane capacity as well as GPU pool capacity. RAM images keep their physical layout checks.

At most one spill session is installed: emergency preparation excludes the worker's idle preparation, and session teardown releases pins and resets the session in one critical section before unlinking its draft objects.

Spill preparation encodes state blobs (zstd and record CRC32C) and page-record headers with the index mutex released, then relocks only to reserve object IDs and the append range and to re-validate cancellation, a claim of the entry an Extend or Refresh would rewrite, and quarantine or eviction of its source; an abandoned preparation does not mark the RAM entry failed.

MANIFEST images are sequenced under the mutex and published by one writer (tmp write, fsync, rename, directory fsync); an image older than the last published one is dropped.

Capacity and quarantine eviction are two-phase: victims are selected and made unavailable under the mutex, their tombstones are made durable with the mutex released, and only a durably tombstoned victim drops its references; any other victim returns to service unless quarantined.

A disk claim never waits for an idle Extend or Refresh of its entry: it cancels the spill, whose commit abandons before its `meta.bin` rename or rolls it back. If that rollback fails, the newer generation is published and the claimed restore fails its committed-generation check as a `CacheRestoreFailure` cache miss, falling back to cold prefill.

Quarantine makes an entry unavailable and invalidates RAM tickets naming it at once, without waiting for its pins; the disk worker evicts it once they drop.

Stats observers and the scheduler's stats publication read the disk tier without blocking on its index lock and never see its counters step backwards. Durable publication orders pack namespace, map, entry and manifest before final synchronization. Orderly shutdown captures retained device state, flushes nondurable RAM and outstanding writes, then releases lanes. Disk caching adds no active-request disk swap or second growing-cache format; live RAM suspension remains separate from its optional prefix entries.

## 11. Fixed-concurrency and Device Graph rules

The runtime owns one startup-fixed block-table row per possible active lane and prepares exact batch definitions for `B=1..C`. Graph definitions bind stable pool, table, control, status, cursor, and workspace bases. Mapping content may change between replays, but the addresses do not.

Cross-page materialization happens before replay on the same execution lane. Serving never captures new graphs or allocates cache workspace. There is no per-row submission, per-page kernel launch, mapping-dependent recapture, or padding of `B=1` to maximum concurrency.

Graph capture records the same append/attention/state transitions used eagerly. It does not waive transaction commit, poisoning, or publication rules. Exact eager/graph state bytes are required at the semantic boundary.

## 12. Prohibited implementations

- Gather paged KV into a contiguous scratch cache before attention.
- Store per-plane, per-layer, or per-head block tables.
- Store 64-bit device pointers in block tables.
- Perform one independent page-table lookup per scalar lane when a workgroup can share it.
- Launch one kernel per physical page.
- Assume adjacent logical pages have adjacent physical IDs.
- Retain a continuous-cache fallback or runtime cache-format branch.
- Allocate, repack, or read device status back inside an Op.
- Publish a frontier before all layers and device status succeed.
- Reuse a read capability across a mapping/publication generation change.
- Make K/V/scale planes independently writable by multiple heads or transactions.

## 13. Correctness qualification

The mathematical oracle evaluates attention from logical positions and independently decoded represented cache values. It does not reproduce production page traversal, staging casts, or reduction order. The exact codec oracle independently derives every physical byte offset and packed code.

Required coverage for any changed cache/attention route includes:

- identity, contiguous-offset, and fragmented-permutation mappings;
- logical positions 0, 1, 31, 32, 63, 64, 65, 127, and 128 as applicable;
- append beginning mid-page and crossing one or more pages;
- G16 and G32 while group selection remains open;
- all eight independent plane-order combinations while layout selection remains open;
- exact FP8 saturation/signed-zero bytes, INT4 RNE ties, FP16 scale underflow/overflow;
- zero/full/partial device-prefix counts and segmented MTP alignment/AR append;
- same-stream pending read, committed read, cross-stream/stale-generation rejection;
- in-place tree compaction and zero-copy rollback;
- malformed/aliased mappings, invalid device row/position/count, nonfinite input, poison isolation;
- multiple allocations/table rows in one batched invocation;
- mapping changes between Device Graph replays; and
- exact host spill/restore plus semantic-fingerprint no-write rejection.

Floating output is compared to the independent FP64 formula using a criterion appropriate to the selected implementation profile. Codec, mapping, state, and packed-cache transformations are exact.

## 14. Performance qualification

Performance decisions use HIP-event timings on an otherwise idle R9700 after correctness passes. Short routes require repeated interleaved measurements and absolute latency; long routes use repeatable medians. Profiler data explains a measured route but does not select one by itself.

Representative coverage is:

- suffix append and complete attention at T=1..8 and T=9/17/128;
- context 1K, 4K, 8K, and 32K;
- G16/G32 and every plane-order candidate while selection is open;
- fragmented as well as identity mappings;
- ordinary and speculative fixed-width execution; and
- final whole-Engine prefill/decode at C=1..8 once a real artifact exists.

The 32-point cache-layout sweep passed the independent layout and attention oracles. The current G16/token-K/feature-V/feature-scale profile has normalized mean latency 1.004041, mean rank 2.188, and 14 wins. It is provisional because operator latency cannot replace model quality, resolved capacity, and whole-inference evidence.

If complete evidence leaves the G16 and G32 layouts for the selected byte-identical artifact non-dominated, production does not retain both. It selects the layout with the best worst-cell whole-inference throughput ratio after normalizing every required cell to its frontier best. Exact ties resolve by worst-cell normalized capacity, then worst quality-tier budget consumption, then canonical static cache/execution-profile identity when the measured objective vectors are identical. The operator sweep above is qualification evidence and is not a tie-breaker for the static product profile.

Measured suffix append costs about 5 to 8 microseconds at T=1..8 and 37 to 46 microseconds at T=128, while complete attention ranges from roughly 0.66 ms at 1K/T1 to 301 ms at 32K/T128. Even a physically impossible zero-cost append fusion is therefore below 0.8 percent at 1K and below 0.21 percent at longer contexts. A single ordinary HIP launch cannot globally order all independent append writers before all attention readers, and per-query-head encoding violates single-writer transaction ownership. Separate ordered append followed by the selected attention family is the qualified architecture.

Host-fixed 1..8-row-per-sequence decode over up to 16 sequences of a compact batch (ordinary T=1, MTP and DFlash chain verification; non-tree, contexts 64..262144) runs the packed decode route: the dense attention arithmetic split over context chunks with a stable FP32 merge and a bounded caller-owned partial workspace.

A uniform causal sequence batch (ordinary decode, MTP and chain verification, and a mixed round's verify part) keeps one append transaction per sequence but issues each layer's append for all of them as one codec launch, and, when every sequence takes the packed route, attends all of them with one split launch and one merge (grid z = sequence, each with its own table row, frontier, positions, output and workspace slice); every sequence's bytes equal its own launch. Tree verification and batches with a sequence off the packed route attend one sequence at a time after the shared append.

Host-fixed 9..127-row causal chunks (appended turns, tool results and prompt tails) run the mid-row route: the 32-row dense prefill tiles split over at most 64 context chunks into at most 2048 row-chunk FP32 partials (50.7 MB), merged by the same stable FP32 merge. The chunk count minimizes whole 64-CTA waves times keys per CTA plus a fixed overhead; host-fixed 128..1023-row calls take this route whenever that splits the context and dense prefill otherwise, as do 1024 rows and above and device-counted rows.

At context 8,192 and above, fixed-width T=4 with packed-tree or device-active-row metadata uses the three-stage split-512 leaf with caller-owned score/partial/merge storage; remaining shapes use fused QK/online-FP32-softmax/PV.

Planned attention workspace covers every narrower host-fixed row count at every smaller frontier, because planners size one call from the maximum row count and visible envelope, plus one packed slice per sequence of a batched launch.

Final admission additionally requires:

1. paired BF16-reference 8K and 32K PPL;
2. exact greedy-token comparison at every scored position;
3. eager/Device Graph state and output parity;
4. speculative acceptance/publication checks; and
5. complete prefill/decode and C=1..8 throughput with valid attribution.

The complete BF16 source checkpoint and supported Python environment are present, and the paired quality evidence is retained. Remaining final admission work is selection-dependent physical eager/Device Graph parity, speculative acceptance/publication, and attributed prefill/decode throughput across C=1..8 on the selected profile. No earlier device/backend result is product evidence.

## 15. Fixed contract versus tunable profile

Fixed architecture:

- FP8 E4M3FN K, signed INT4 V, FP16 V scale, three planes only;
- `P=64`, pool-local I32 page-group IDs, one allocation-owned block-table row;
- Text/MTP physical separation and target-derived capacity coupling;
- generation-bound read capabilities and transaction-only publication;
- no gather, fallback, hidden allocation, runtime repack, or runtime format selection; and
- fixed-address exact-B Device Graph execution.

Currently tunable until the real-model gate:

- G16 versus G32;
- the three plane orders;
- attention tile/split/vector/LDS staging and wave assignment; and
- append workgroup geometry, provided it preserves one writer per packed destination.

Changing page size, code semantics, scale type, pool grouping, page-ID model, publication ownership, or introducing another cache representation is an architecture change, not kernel tuning.

## Disposable cache reads

`ExecutionOptions.cache_write=false` forks a selected prefix into independent allocations and complete continuation state. It never shares writable pages, mutates source checkpoints, or consumes a RAM/disk original. Copy all three planes in their per-plane stored layouts, including partial final pages; preserve logical publication frontiers and initialize GDN, hidden state, position/media metadata and MTP/DFlash state at the selected frontier. Reader fences protect source allocations and images. A pending AMD GDN fold is materialized before copying current state.

An ordinary original may migrate into a full-original RAM image before its GPU bundle is released, bypassing closed-turn rewrite cutting while keeping existing checkpoints. A disposable original cannot be offloaded. Disposable completion retains only coherent current state in VRAM; no future-reuse checkpoint, RAM or disk capture is created. Disposable retention precedes ordinary retention for lane replacement and page reclamation. A later writable continuation promotes its own resulting state. Cancellation drains destination readers; a committed original migration remains cached. The concurrent architecture defines preservation-feasible admission and failure classification.
