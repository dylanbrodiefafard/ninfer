# NInfer Paged KV Context Store

Current product execution supports ordinary decoding and DFlash2 only. MTP execution is
rejected at Engine admission. MTP algorithms and layouts below describe dormant private
machinery, not an available product backend.

This document defines the growing KV storage architecture of NInfer for a single GPU, a single
resident model, and a small number of concurrent requests. It is the KV substrate on which the
[small-scale concurrent inference architecture](concurrent-inference-architecture.md) depends.

This document defines the KV pool layout, page ownership, capacity reservation, logical frontier,
prefix retention, and lifecycle. It also defines the device block-table representation, the
single-sequence and batched paged KV execution views, the state effects on the affected Ops, the
kernel addressing constraints, and the performance admission conditions. The concrete allocator
algorithm and the CUDA kernel code are not specified here.

---

## 1. Requirements

- Active requests with `max_concurrency=1..8` share every kind of growing KV capacity;
- A single request may use most of the capacity of the main KV pool; capacity is not split evenly
  per slot;
- The physical KV of different requests need not be contiguous;
- Active, retained, and speculative provisional KV use the same reservation accounting;
- Prefix state can be transferred from a retained owner to an active request without copying the
  KV payload;
- Once an active request is admitted, it has a completion capacity guarantee for its declared
  range of prefill, decode, and speculative temporary growth;
- During one GPU execution unit, page mappings and logical valid frontiers stay stable;
- BF16, INT8-G64, and the other fixed bytes-per-token layouts defined by a target use the same
  management semantics;
- The common allocator does not understand model semantics such as GQA, MHA, MLA, MTP, or DFlash;
- Single-sequence prefill/cached consumers and batched ordinary/MTP/DFlash decode consumers all
  consume paged KV directly, and do not require the growing KV of any sequence to be physically
  contiguous;
- Paged addressing must not introduce a gather-to-contiguous copy, and must pass performance
  admission on the same workload as the frozen baseline.

### 1.1 Non-goals

- general request preemption, swapping or cross-GPU storage; settled RAM-on pressure
  suspension follows §5.8 of `concurrent-inference-architecture.md`;
- sharing a writable prefix between active requests, page reference counting, or copy-on-write
  branching;
- arbitrary longest-common-prefix reuse;
- using a universal raw-byte allocator to dynamically repartition GPU memory among different KV
  layouts during serving;
- forcing the same kind of paging onto bounded cyclic KV or operator transient K/V;
- changing the pool layout or page size during serving;
- the scheduler, active-batch formation, or other non-KV model-execution policy;
- using the KV storage design to reorganize the Attention source directories, rename the Attention
  family, or enlarge the supported mathematical domain.

---

## 2. Three independent granularities

The KV design must distinguish three granularities that are independent of one another:

| Granularity | Meaning | Fixed value or constraint |
|---|---|---|
| allocation granularity | How many tokens of payload a homogeneous pool acquires or releases at a time | Every current growing pool uses `P=64` |
| valid-frontier granularity | Which logical KV positions may be read | 1 token |
| reusable-state granularity | Which prefix positions own a complete continuation state | Determined by the complete sequence-state checkpoint |

A page boundary is neither an Attention semantic boundary nor a prefix-cache hit boundary. A legal
frontier may lie at any token offset inside a page.

The Linear Attention state of the current Qwen3.6 target cannot be reconstructed from the KV prefix
alone. Prefix lookup may use only target-declared checkpoints that retain the complete continuation
state. The currently reusable positions are the current resume frontier and one typed rewrite
checkpoint when valid; the concrete claim/restore semantics are in §10. A shorter token prefix that
falls outside these two checkpoints is an arbitrary partial hit, which this design does not
support.

The Paged KV Store itself can truncate at any logical frontier; this does not mean the upper layer
can restore the whole model at an arbitrary position. A reusable position must be proven by a
complete target checkpoint; adding a new checkpoint does not require changing the page geometry.

---

## 3. KV pool set and capacity model

### 3.1 Homogeneous pools

At startup, an Engine establishes a fixed KV pool set from the exact target, the KV dtype, and the
speculative backend:

```text
ordinary Engine:
    Main Text KV Pool

MTP Engine:
    Main Text KV Pool
    MTP KV Pool

DFlash Engine (Qwen3.8-27B DFlash2):
    Main Text KV Pool
```

Qwen3.8-27B DFlash2 sets `full_layers=0`: that Engine has only the Main Text growing pool. Its
five SWA-2048 windows are DFlash local cyclic state, not a growing pool.

MTP and DFlash are engine-wide mutually exclusive backends, so an Engine contains at most two
growing KV pools. DFlash local cyclic KV is fixed state and is not part of this pool set.

These names describe how the target/runtime uses a pool. The common KV Store sees only a set of
immutable pool layouts and opaque pool handles, and does not dispatch on feature name, attention
type, or runtime string.

### 3.2 Capacity is a vector

The Engine's startup configuration provides:

```text
max_context                  per-sequence logical limit S
kv_capacity                  sizing policy: Explicit(K_main) or Automatic(R)
max_concurrency              upper bound on active sequences
selected speculative backend off, MTP, or DFlash
```

`max_concurrency` determines only the number of control lanes, block-table rows, and fixed
per-sequence resources; it does not split the KV capacity into equal shares.
`EngineOptions.max_context` constrains only the frontier of any single sequence;
`EngineOptions.kv_capacity` resolves to the number of shared physical page groups of the Main pool.
Backend capacity is not an independent user setting; it is derived uniquely from the resolved Main
capacity, the selected backend, `max_concurrency`, and the exact target frontier contract.

The planner normalizes the Main contract into a physical page capacity:

```text
L     = ceil(S / P_main)
M_min = max(L, max_concurrency)
M_max = max_concurrency * L
M     = resolved physical page groups in [M_min,M_max]
Main physical page groups = M
per-allocation logical page capacity = L
```

The Explicit policy uses `M=ceil(K_main/P_main)`. The Automatic policy carries a device headroom
`R` that must be preserved, and queries the remaining device memory `F` after the weights are
loaded and synchronized. For the CLI/server, `R` defaults to 32 MiB and can be set with
`--kv-capacity-headroom`; the Engine policy may specify this value explicitly. The exact target
uses the same complete physical candidate builder to obtain:

```text
B(M)     = persistent + workspace + request transient + CUDA Graph allowance
B_min    = B(M_min)
B_step   = B(M_min+1) - B(M_min)
M_auto   = min(M_max, M_min + floor((F-R-B_min)/B_step))
```

Automatic requires `F>=R+B_min`; when `M_min=M_max`, that single point is taken directly. When
`M_max` is not reached, the final `planned_slack=F-B(M)` lies in `[R,R+B_step)`, so device memory
is never squeezed down to a random remainder smaller than one page group. `B(M)` includes the
coupled growing pools of Main and the selected backend, its persistent part comes directly from the
production `LayoutBuilder`, and its workspace comes from the capacity recipe of the production
schedule; the common resolver maintains no model dimensions or bytes-per-token formulas and does
no allocation probing. The final plan then has the same builder generate and verify the reservation
curve for `M`.

For single-schedule graph families (ordinary, MTP, DFlash2), the CUDA Graph allowance is counted by
the number `n` of `(K, B, topology)` executables, including their reachable definitions. Ordinary
and MTP count `min(12n, 24+6n)` MiB: the measured total is a fixed warm-up part plus about 4 MiB
per executable (RTX 5090 serve benches: at most 46 MiB for 4 executables, 146 MiB for 30). DFlash2
counts `16+4.75n` MiB, covering the measured graph amount plus the driver memory that Instance
construction consumes outside the arena (about `2+2.3C` MiB, not inside the graph measurement
window): at 262144-token context with adaptive K7, the C=1..8 serve startups (12-88 executables)
measured 42-396 MiB for graphs, 46-416 MiB in total outside the arena, and the allowance is 17-27
MiB above each total, so Automatic capacity keeps the full `R` free after startup. All definitions
and executables are established at startup. The measured value is the device-wide free-memory
difference before and after preparation, which accounts for allocations/frees by other processes on
the same GPU, so it
is reported only as an observed/allowance figure and is not a startup failure condition; the
allowance calibration is checked by the real-artifact capacity test on an exclusive GPU. The
4096-token prefill and adaptive K switching use the already planned shared workspace, and add no
graph family or expansion of the KV pool.

`R` is sizing headroom that the capacity solver deliberately does not consume. The physical
footprint of the CUDA allocator, context, and modules is not entirely equal to the arena payload;
therefore, after the Instance and Graph are fully established and synchronized, the actual free
memory is queried again and reported together with the policy and the planned slack. The Engine
allocates no device memory after startup (in a serve measurement with C=6 and six requests of about
158k tokens, per-process usage is constant; after C=1..8 startup, an 18,400-token prefill and
decode grow it by at most 2 MiB, and a 260000-token prefill does not grow it), so the default `R`
is 32 MiB, covering only driver-side growth; when sharing the GPU with a desktop or other
processes, increase it with `--kv-capacity-headroom`. The startup failure message in Automatic mode
hints at this option.

The driver reserves local memory of per-thread stack size for every resident thread (about 0.25 MiB
per byte on the 170 SMs of the RTX 5090), and this is counted in the context's usage before the
weights are loaded. Before any allocation, `DeviceContext` sets `cudaLimitStackSize` to
`kDeviceStackBytesPerThread` (384 B, the largest compiled kernel stack frame) instead of the 1024 B
default, saving about 160 MiB. When a kernel that needs a larger frame is first launched, the
driver silently enlarges the reservation, bypassing the capacity plan;
`ninfer_device_stack_limit_test` uses `cuobjdump` to inspect every library that contains device
code, and fails if any kernel exceeds this value.

The Engine reports both the configured `max_context` and the resolved `kv_capacity=M*P_main`. The
rounding tail of the last physical page is only storage padding, and cannot let a sequence frontier
exceed `S`. Admission charges by page-group entitlement, so the sum of the page-rounded
entitlements of multiple active/retained requests cannot exceed `M`.

Internally, a target-derived capacity vector is used:

```text
KVCapacity = {
    main_page_groups,
    optional_backend_page_groups,
}
```

The Main per-allocation logical capacity is `L`, and its physical capacity is `M` page groups. The
logical capacity and the physical page-group count of the backend must be described separately:
the extra physical headroom of MTP must not be interpreted as a larger per-sequence logical
context. Backend sizing is derived from the fixed frontier relation of the selected backend; it
cannot be derived from the ratio of main/backend bytes-per-token, and cannot be configured
independently. Explicit requires raw `K_main >= S`; both policies require `M >= max_concurrency`
and `M <= max_concurrency*L`, which respectively guarantee that one request can reach `S`, that
every active lane can obtain at least one page, and that physical capacity which the lanes could
never use is not accepted.

### 3.3 Physical separation, coordinated capacity

Physical pool separation does not mean that Main and the backend may borrow capacity from each
other, but their instantaneous mapped extents also need not maintain a size relation. For a request
`r`, denote the reservation entitlements of Main and the backend as `E_main(r)` and
`E_backend(r)`, and the ranges of physical pages currently acquired as `M_main(r)` and
`M_backend(r)`. At runtime the following must hold separately:

```text
0 <= M_main(r)    <= E_main(r)
0 <= M_backend(r) <= E_backend(r)
```

There is no universally valid `M_backend(r) <= M_main(r)`. The two backends advance temporarily in
different ways:

- The autoregressive draft of MTP writes provisional backend KV before the target verify; within a
  round the Backend mapped extent can temporarily lead Main. The target verify only needs to
  materialize Main according to its own visible domain, and must not occupy extra Main pages merely
  to maintain a false size relation between mapped frontiers;
- DFlash2 has no backend growing pool: its proposal query K/V is transient workspace and its local
  K/V is fixed cyclic state.
  The committed stable KV frontier still satisfies that the backend does not lead Main; the MTP lead
  exists only in the provisional range mapped within a round, and must not be interpreted as
  committed state.

The Main and MTP pools of the registered targets both use `P=64`. Let
`C=max_concurrency`, and let the MTP draft window be `K_draft`. The startup capacity profile is
fixed as:

```text
speculative_backend = off:
    Main physical=M, logical-per-allocation=L
    no backend growing pool

speculative_backend = MTP:
    Main physical=M, logical-per-allocation=L
    MTP physical=M + C*ceil((K_draft-1)/P), logical-per-allocation=L

speculative_backend = DFlash:
    Main physical=M, logical-per-allocation=L
    no backend growing pool
```

The extra physical groups of MTP cover the `K_draft-1` provisional positions by which up to `C`
concurrent rows each exceed their Main entitlement; this does not increase the block-table width,
and does not allow any allocation to exceed `L` logical pages.

The two pools do not share physical pages; they only plan typed payloads separately for the same
number of logical 64-token groups. Because their materialization times differ, it is a normal state
for one pool to have free pages while the other pool has already materialized all of them. For
example, an MTP draft may cause the Backend to temporarily materialize one extra page, while DFlash
may cause Main to materialize several extra pages. Each pool only needs to obtain pages within its
own entitlement, and may not borrow, reinterpret, or dynamically reallocate the typed capacity of
the other pool.

Therefore, startup must complete the memory planning of all typed slabs, fixed state, workspace,
and driver allowance before establishing any pool. If Explicit cannot be honored, the Engine
configuration is rejected, and capacity is not silently reduced; Automatic resolves the maximum
legal `M` exactly once based on the memory remaining after the weights are loaded. After the final
pools are established, neither policy expands, reallocates, or rebuilds Graphs again. After
retained eviction, if a request set still satisfies the Main entitlement contract but cannot obtain
the necessary backend reservation, this is a sizing/accounting invariant violation; at runtime it
is handled as an Engine failure, and must not wait for backend pages to happen to be freed.

That one pool has materialized its entire entitlement while the other pool still has free pages
does not mean capacity sizing has failed. Free pages cannot be interpreted as another kind of
payload; they are typed slack of the coordinated profile, not external fragmentation. The exact
backend is fixed at startup, and the Engine should plan the typed slabs in one pass, rather than
introduce a variable-size raw allocator, runtime repartition, or compaction.

---

## 4. Pool layout and grouping rule

### 4.1 The grouping invariant

Only cache planes that satisfy all of the following conditions can form one homogeneous page-group
pool:

1. They use the same cache-ordinal domain;
2. They are managed by the same logical valid frontier;
3. They are reserved, materialized, truncated, retained, and released together;
4. They fit the same token page size `P`;
5. There is no meaningful need for independent ownership or independent capacity reuse.

Therefore the grouping boundary is determined by frontier and lifetime, not by "belonging to the
same request" or "belonging to the same model".

The current pool grouping is:

| Pool | Grouped planes | Pool frontier |
|---|---|---|
| Main Text | K, V, and optional code/scale planes of all target full-attention layers | target materialized KV frontier |
| MTP | K, V, and optional code/scale planes of the MTP layer | MTP KV frontier |

Text and MTP do not form one physical page group, because they can have different
valid/provisional frontiers, different retention readiness, and different consumer geometry.

### 4.2 Allocator-visible layout

Each pool layout gives at least:

- the page size `P`;
- the number of planes;
- for each plane, the element type, the logical page payload bytes, the physical strides, and the
  alignment;
- the pool page-group count;
- the target-owned plane index mapping.

The KV Store uses only these storage facts. Head count, head dimension, the GQA/MHA/MLA relation,
the quantization formula, the MTP/DFlash identity, and Attention semantics are all interpreted by
the target layout and the consuming Op. The allocator contains no attention-type variant.

Consumers use a unified logical coordinate `K/V[d,h,p]` for K/V planes. The physical axis order is
fixed by the homogeneous pool, and is not chosen by the allocator or by an individual request:

```text
Pool            K/V or code plane       INT8-G64 scale plane
Main Text/MTP   [D, P, Hkv, Nphysical]  [D/64, P, Hkv, Nphysical]
```

Main Text/MTP uses contiguous page-major order. For element bytes `E` and first-dimension extent `X`
(`D` for K/V/code, `D/64` for scale):

```text
nb[0] = E
nb[1] = X * nb[0]
nb[2] = P * nb[1]
nb[3] = Hkv * nb[2]
```

Slab alignment may exist between planes, but within a single plane no padding, axis permutation, or
third target-specific physical order is allowed. For a logical position `p`:

```text
b = floor(p / P)
o = p mod P
g = block_table[b]

Main/MTP:    K[d,h,p] = k_pages[d,o,h,g]
             V[d,h,p] = v_pages[d,o,h,g]
```

INT8 code uses the same formula; scale replaces `d` with the quant group `d/64`. K, V, code, and
scale do not keep their own page pointer tables, but use the same pool-local page-group ID `g`.

The common allocator receives the already determined closed plane order, bytes, strides, and
alignment, and does not derive head, codec, or Attention semantics from them. The production
wrapper accepts only the closed stride formula above that corresponds to its route. Kernels compute
the base at page/tile granularity, do not interpret an arbitrary layout in the hot loop, and no
runtime layout selector exists.

### 4.3 Closed physical orders

Main Text and MTP use page-major:

```text
plane
├── physical page 0: head 0, head 1, ... head H-1
├── physical page 1: head 0, head 1, ... head H-1
└── ...
```

A page-group ID selects one contiguous `P*Hkv` slice in each Main/MTP plane.

The allocator may return arbitrary page IDs, and consumer correctness and launch topology do not
depend on ID contiguity. Page-major is the only order for the registered growing pools; Main/MTP
Ops accept only page-major, and no arbitrary-stride or dual-layout execution path is provided.

### 4.4 Logical position domain

The page table uses the autoregressive cache ordinal, not the RoPE/MRoPE coordinate:

```text
logical block b covers cache positions [b*P, (b+1)*P)
```

Multimodal three-axis positions, `rope_delta`, and other position transformations do not change KV
slot ownership. They belong to the Attention input metadata.

### 4.5 Fixed bytes per logical position

For a cache family with `layers=L`, KV heads `H`, and head dimension `D`:

```text
BF16 bytes/token
    = 2(K,V) * L * H * D * 2

INT8-G64 bytes/token
    = 2(K,V) * L * H * D
    + 2(K,V) * L * H * (D/64) * sizeof(FP16 scale)

NVFP4-G16 bytes/token
    = 2(K,V) * L * H * (D/2)
    + 2(K,V) * L * H * (D/16) * sizeof(FP8 scale)
```

The logical page-group payload of a homogeneous pool is the sum of the bytes/token of all its
grouped planes multiplied by the `P` of that pool. The startup physical pool bytes are obtained from
the sum of the registered plane storage spans plus the slab/head alignment; alignment does not
change the allocator's page-group ID count. The page-group payloads of different pools need not be
equal.

---

## 5. Physical page-group pool

### 5.1 One allocation unit within one pool

A Main Text page group represents the payload of the same span of `P_main` positions across all Main
Text planes:

```text
Main Text page-group id = g

target layer 3 K plane   -> page[g]
target layer 3 V plane   -> page[g]
target layer 7 K plane   -> page[g]
target layer 7 V plane   -> page[g]
...
```

The MTP and DFlash pools each own an independent page-group ID namespace, free set, and block tables.
Page IDs with the same numeric value in different pools have no ownership relationship.

### 5.2 Plane slabs

The grouped planes within one pool are not required to form one contiguous blob. Each plane owns
Engine-lifetime-stable storage; a pool-local page-group ID `g` selects the corresponding logical page
group slices in each plane, following that pool's closed physical order:

```text
Main/MTP plane:    [page 0: all heads] [page 1: all heads] ...

page-group g      -> plane 0 slices selected by g
                  -> plane 1 slices selected by g
                  -> ...
```

A pool has exactly one page-group ID namespace, one free set, and one per-sequence block table. The
page-group payload in §7 is the total bytes of the reservation/accounting unit; the planes are not
required to be physically adjacent to each other.

Plane bases stay constant for the Engine lifetime. A consumer computes addresses from the plane base,
the registered strides, and `g`; it must not expand the block table into per-layer, per-head, or
per-plane device pointer arrays. The allocator may preferentially return contiguous page IDs to improve
locality, but kernel correctness and launch topology must not depend on that contiguity.

### 5.3 Why K/V and layers remain grouped inside a pool

K, V, code, scale, and full-attention layers within the same pool have the same frontier and lifetime.
Allocating them separately would not produce any effective resource that could be freed individually,
but it would introduce:

- an independent block table for each layer or plane;
- duplicated reservation counters;
- layer-by-layer partial allocation failure;
- duplicated bookkeeping for retained ownership.

Main Text layers are therefore grouped, but Main Text and MTP/DFlash are not grouped together.

### 5.4 Fragmentation and locality

All page groups within each pool are equivalent, so there is no variable-size external fragmentation
and no need for compaction. Allocation success depends only on the total number of free pages in that
pool, not on the largest contiguous hole.

For example, a Main Text pool has 128 Ki token-equivalents, `P=64`, for 2048 pages in total:

```text
A owns 48 Ki = 768 pages
B owns 48 Ki = 768 pages
free          = 512 pages
```

After A is released, its 768 old pages and the 512 tail pages may be separated by B's physical range.
A new request C needs 64 Ki, that is, 1024 pages:

```text
total free pages = 1280
needed pages     = 1024
```

A contiguous allocator would fail because the largest hole is only 768 pages; a paged allocator can map
C's first 768 logical pages onto A's old pages and take another 256 pages from the tail, so B does not
need to be moved.

Pages held by a retained allocation are not free capacity. If a retained A blocks active admission,
the Prefix Cache evicts A first; this is occupancy reclamation, not compaction.

The only payload slack is the last partially filled page of each pool allocation, at most `P-1`
positions. For `P=64`, the maximum tail-page slack of 8 active allocations in one pool is:

```text
8 * 63 = 504 token-equivalents
504 / 131072 = 0.38% of a 128 Ki pool
```

The allocator should preferentially assign contiguous page IDs to the same sequence, so that adjacent
logical pages are as contiguous as possible in the plane slab; however, a contiguous run is not an
admission condition. When no contiguous run can be found, any free pages are used, and the request must
not be rejected for that reason. Active pages are not relocated.

---

## 6. Per-sequence allocation bundle

An admitted sequence holds an allocation bundle that is independent of slot identity:

```text
SequenceKVBundle
├── Main Text PoolAllocation
└── MTP PoolAllocation (iff MTP is the selected backend)
```

Each `PoolAllocation` independently contains:

```text
PoolAllocation
├── ordered logical-block -> pool-local page-group mapping
├── mapped page-group count
├── total reserved entitlement
└── target-owned valid frontier
```

MTP and DFlash are mutually exclusive. When the Backend is disabled, the bundle contains only Main;
when a backend is enabled, every admitted sequence must own both a Main allocation and that backend's
allocation, and a request must not be downgraded to a bundle that lacks backend state.

### 6.1 Three extents per pool

Each pool allocation must distinguish:

| Extent | Ownership |
|---|---|
| reserved entitlement | admission/resource layer; guarantees pool-local pages that can be obtained in the future |
| mapped extent | KV Store; the logical range that is already bound to page IDs |
| valid frontier | target sequence state; the prefix that a consuming Op may currently read |

The mapped extent may temporarily be larger than the valid frontier, to hold provisional writes of a
prefill chunk or a speculative round. Mapped does not mean committed.

Different pool allocations may have different mapped extents and valid frontiers. The Store does not
compute the semantic relationship between them.

### 6.2 Slot execution views

An active slot owns, for each enabled pool, one row of fixed-address device block-table metadata, with
maximum width:

```text
ceil(max_pool_logical_extent[s] / P[s])
```

The rows of one pool form a fixed-address matrix:

```text
device_block_tables[s][slot][logical_block] -> pool-local page-group ID
```

Each row is contiguous, and its elements are I32 page IDs. The row width is determined by that pool's
maximum logical extent; the number of rows of a concurrent Engine is fixed at `max_concurrency`. The
matrix is execution metadata, not KV capacity, and does not own physical pages.

The `PoolAllocation` holds the authoritative ordered mapping; at lane admission the mapping is
installed into the corresponding row. Retained state does not occupy a bound table row; in the current
runtime the complete SequenceState is physically lane-affine, so reuse rebinds the same lane row rather
than relocating the fixed continuation state. The KV allocator itself still does not derive ownership
from a row. Unmapped tail entries have no consumer-visible meaning and may use an invalid sentinel in
debug/test builds, but production kernels do not add a page-ID bounds branch to every read.

A single-sequence prefill/cached Op obtains a `PagedKVLayerView` of one table row. A batched decode Op
obtains a `PagedKVBatchLayerView` with the shared plane bases and the complete table matrix, and then
selects the allocation for each compact batch row through a separate `table_rows[B]` selector. The
compact row, the control lane, and the block-table row therefore need not be the same, and no KV
payload gather is required.

### 6.3 Stable execution unit

Before a prefill chunk or decode round starts, the KV Store materializes, for every pool that may be
written, the maximum logical extent of this unit and publishes the complete mappings. While GPU work is
in flight:

- the block tables of all participating pools are neither extended nor rewritten;
- page payloads are not relocated;
- consuming operators use the valid-frontier inputs frozen for this unit;
- slot recycling and retained ownership transfer are forbidden.

All mapping updates, frontier commits, and page releases happen at a GPU boundary.

---

## 7. Fixed page size

Page size is a property of a homogeneous pool layout, not a global semantic of the whole KV Store.
Currently all registered growing pools are fixed at:

```text
P_main          = 64
P_mtp           = 64
```

The two pools have independent storage, page ID namespaces, and capacities, but use the same `P=64`.
`P` is not a request option and does not change dynamically during serving.

### 7.1 Design basis

The `P` of a pool is determined by storage and consumer locality:

- how many contiguous positions the consuming Attention's usual key tile covers;
- how a prefill chunk or proposal block spans pages;
- the frequency of page-table lookups, address discontinuities, and boundary handling;
- per-sequence block-table metadata;
- tail-page slack of active/retained allocations;
- whether a single-plane page is large enough to form contiguous, well-coalesced read/write ranges.

Prefix hit granularity does not take part in the choice of `P`. A legal retained frontier may lie at
any offset within a page; only whether the complete Linear Attention/backend state exists decides
whether that frontier can be reused.

### 7.2 Main Text geometry

The main decode Attention of the current target uses 32/64 keys as its main read scale, and the Text
prefill chunk is a multiple of 128. With `P_main=64` fixed:

- one page suffices to cover one aligned 64-key span;
- a 128-token aligned prefill interval spans only two pages;
- a 128 Ki context needs only 2048 entries, and a native 256 Ki context only 4096 entries;
- for the largest 27B Main Text BF16 pool, each page-group is 4 MiB, and the tail-page slack of a
  single allocation is below 4 MiB;
- block-table metadata is negligible relative to the KV payload, a 32-key tile does not cross a page,
  and a 64-key tile aligns with the page.

### 7.3 Backend page sizes

The context traversal of MTP full Attention uses the same `P_mtp=64` and page-major order as the
corresponding target full Attention. The MTP pool keeps independent pool ownership and capacity, and
shares only the page size, allocation, and block-table semantics with Main Text.

### 7.4 Current pool payloads at `P=64`

The table below counts only KV payload, excluding static slab alignment:

| Pool layout | KV storage | bytes/token | page-group payload | worst tail slack |
|---|---|---:|---:|---:|
| 27B Main Text | BF16 | 65536 | 4.0000 MiB | 3.9375 MiB |
| 27B Main Text | INT8-G64 | 33792 | 2.0625 MiB | 2.0303 MiB |
| 27B Main Text | NVFP4-G16 | 18432 | 1.1250 MiB | 1.1074 MiB |
| 27B MTP | BF16 | 4096 | 0.2500 MiB | 0.2461 MiB |
| 27B MTP | INT8-G64 | 2112 | 0.1289 MiB | 0.1269 MiB |
| 27B MTP | NVFP4-G16 | 1152 | 0.0703 MiB | 0.0692 MiB |

The 4 MiB page group of 27B Main Text BF16 is distributed across all full-attention planes. For a
single K or V plane of a single layer, the aggregate bytes per page ID are 128 KiB. A consumer always sees one pool-specific typed plane view, not a composite
payload that is a cross-model mechanism.

Assuming 32-bit page IDs, a single block-table row for a 128 Ki context is 8 KiB. Even with
`max_concurrency=4` and every slot holding both a main row and one backend row, all metadata totals
only 64 KiB.

---

## 8. Vector reservation and admission

With RAM off, admission reserves the maximum physical extent for the full effective output
lifetime. With RAM on, it reserves the full prompt plus at most 8192 output tokens and target-owned
speculative headroom. The executor extends windows at settled boundaries without reducing output
limits. Completion claims and secured live images guarantee progress as described in §5.8 of
`concurrent-inference-architecture.md`. Target computes the same typed vector for each window:

```text
KVReservation(request) = {
    main_total_pages,
    optional_backend_total_pages,
}
```

The computation includes:

- the main/backend positions of the prompt that are not yet materialized;
- full effective output with RAM off, or the funded output window with RAM on;
- the bounded provisional tail that the selected backend may touch;
- each pool's own page rounding;
- the mapped pages already held by the retained bundle;
- whether the request can or must establish a backend continuation.

For each pool `s` separately, the following is maintained:

```text
active_mapped[s]
+ active_reserved_but_unmapped[s]
+ retained_mapped[s]
<= total_page_groups[s]
```

Admission is an atomic transaction across all pools, fixed state, and output capacity. If any required
pool cannot complete its reservation, the request obtains neither a slot nor a partial allocation.

This per-request vector check does not mean that the backend is an independent admission dimension that
may normally be exhausted before the main contract. The startup `KVCapacityProfile` must satisfy the
active-set implication of §3.3: when a request is admissible with respect to slots, fixed state, and
advertised main capacity, the selected-backend reservation must also be satisfiable. Retained backend
pages are evicted by the active-priority policy before this judgment.

A reservation promises only the number of pool-local pages and does not select IDs in advance.
Prefill/decode progress converts reserved capacity into mapped pages without changing the total
entitlement. An admitted request therefore does not lose KV capacity it has already been promised
because other requests grow later.

A retained bundle has no future-growth reservation. At claim time, each existing allocation converts
from retained mapped to active mapped, and the differences for the incoming suffix, output, and
provisional tail are reserved separately.

The admission check of a boundary claim uses the mapped vector after checkpoint truncation; trailing
full pages may be converted to usable capacity within the same atomic transaction, but retained
ownership or mappings are not modified until the complete reservation vector has succeeded.

The speculative backend selected by the Engine is fixed for the Engine lifetime. When a backend is
enabled, every admitted sequence keeps the corresponding allocation and entitlement for its whole
active lifetime; a round that can execute only ordinary target progress because of the remaining
output/context does not constitute a backend capability downgrade, and the backend state must not be
released on that basis. On completion, either the complete, continuable bundle is retained as a whole,
or the entire bundle is released.

When admission is blocked by retained occupancy, the Prefix Cache selects and releases complete
retained entries and then re-executes atomic admission. The KV Store does not select eviction victims
itself. The host-RAM second tier holds no pool
pages: the pinned host image is an independent copy in logical-page order, and GPU pool page IDs
and mapped counts still describe device allocations only.

---

Live RAM continuations borrow secured contiguous arena credits. They are excluded from prefix
lookup, optional eviction and disk spill. RAM occupancy includes credits and live images once,
including transfer-held capacity. Suspension retains GPU pages until capture completion; resume
restores the exact mapped-page floor into independent pages and can select another physical lane.
Window shrink releases only unused entitlement, preserving mapped pages and speculative headroom.

## 9. KV lifecycle

### 9.1 Admission

```text
WAITING
  -> claim optional retained SequenceKVBundle
  -> reserve required pages in every pool
  -> bind bundle to active sequence state
  -> install pool mappings in an active slot execution view
```

Together with the fixed Linear Attention/backend state and the output capacity, these steps form the
atomic admission transaction of the concurrent architecture.

### 9.2 Prefill chunk

Before the chunk launches, the target determines which pools the chunk will write and the maximum
position in each. The KV Store materializes the required pages for each pool separately. Only after the
chunk succeeds does the target advance the corresponding pool frontiers.

If some layers inside a chunk have been written but the unit has not completed successfully, no new
frontier is published; a GPU/state-integrity failure is handled as an Engine-wide failure.

### 9.3 Ordinary decode

Main Text decode usually adds one materialized position. No allocation happens while the current tail
page still has space; when the `P_main` boundary is crossed, the next page is bound from the main
entitlement before the round launches.

Whether a backend pool grows in the same round is determined by the target decode mode and is not tied
to the main page ID.

### 9.4 Truncate and rollback

Each pool independently computes the number of pages to keep from its own valid/provisional frontier:

```text
needed_pages[s] = ceil(max(valid_frontier[s], provisional_frontier[s]) / P[s])
```

Pool-local pages lying entirely beyond that range are returned to the corresponding free set. The last
partially used page is kept; bytes after the frontier may be stale data, but must not enter the valid
range of the execution view.

### 9.5 Finish, retain, release

After a request reaches model completion, one of two mutually exclusive operations is performed:

```text
retain:
    active sequence transfers complete SequenceKVBundle ownership to Prefix Cache

release:
    every PoolAllocation returns mapped pages to its own pool
    all unused reservations are cancelled
```

When a speculative backend is enabled, only a complete bundle in which Main, the backend, and other
continuation state are at the same exact frontier may enter the Prefix Cache. Incomplete state releases
the whole bundle, and a target-only reusable entry must not be published.

The slot and the device table rows can then be reused. The lifetime of the network response does not
extend KV ownership. Cancellation releases or retains the bundle per §7.5 at the first GPU boundary
that observes it; it does not modify the mappings of an in-flight round.

---
## 10. Prefix reuse

### 10.1 Reusable checkpoints

A retained entry publishes the reusable checkpoints for which the target has proven that it owns the
complete continuation state. The current Qwen3.6 retained sequence contains:

- the current resume frontier;
- a typed rewrite checkpoint when one is valid, whose kind is `TurnClosure` or `ResponseReplay`;
- under MTP or DFlash, a prefill context-checkpoint ladder: at the end of each committed prefill
  chunk, if `valid_frontier` has reached the next mark (default 24576 / 36864 / 53248 / 77824 /
  102400 / 151552, replaceable with `--context-checkpoints off|a,b,c`) and the Vision items of that
  frontier are complete, the then-current GDN and the last-column hidden of that chunk are frozen.
  DFlash also freezes the lane's cyclic local K/V; restore writes it back into the cyclic buffer and
  sets `dflash_context_frontier=F`. The advertised frontier `F` is the end of that chunk, not the raw
  mark. The 4096 chunk is not split at the mark.
- under MTP or DFlash, a turn-rollback head: when `append_frontier` is occupied and
  `prompt_tokens > E`, the current GDN and `tail_hidden` are pinned at that completed `E` before the
  suffix prefill (DFlash pins the cyclic state as well). There is at most one per chat; the next
  append replaces it. An exact hit (`prompt_tokens == E`) does not write one by default; it is
  written into the same slot when `capture_context_checkpoint` is true and that `E` has no
  context-checkpoint head yet. Editing the last user message goes through `restore_turn_rollback`;
  regenerating from the same last user message still goes through the longer rewrite `TurnClosure`.

Host RAM capture (and therefore the disk spill that follows it) has an exception for a closed
preserve-off turn: the request ended with a stop token/string and no tool call, and its
`TurnClosure` checkpoint is this request's own generation opener (the frontend marks
`generation_opener`; a checkpoint inside a tool loop sits at the loop's first opener and is not
marked). The next preserve-off prompt then re-renders without that reply's reasoning and forks at
that opener, so the frontier after the checkpoint is no longer reachable. In this case the entry is
stored in the state rolled back to checkpoint `F`: the ledger/identity/Main Text and the backend KV
are cut to `F`, the current GDN, the DFlash cyclic lane and the tail hidden are taken from the
checkpoint, no rewrite copy is stored, and only heads whose frontier does not exceed `F` are kept.
The next turn therefore takes `append_frontier` at `F` (the same reuse length as
`restore_turn_checkpoint`), and each entry stores only one state. A request that carries its own
generation-opener `TurnClosure` checkpoint and appends from a cut RAM/disk entry does not
automatically write a turn-rollback head (the same as when it went through `restore_turn_checkpoint`
before); a resident append (for example a resend after cancellation) writes one as usual. The
retained lane in VRAM is unchanged. The cost: when the next turn uses preserve on, or continues that
reply as an assistant continuation, the reply has to be prefilled again.

Every checkpoint must describe all of the following:

- the frontier and page mapping reachable by the Main Text allocation;
- the selected backend's allocation, frontier and continuation state (absent when the backend is
  off);
- the Linear Attention state and the hidden state needed to continue execution;
- position/model-continuation metadata;
- a prefix identity consistent with the above state.

The current resume frontier, the rewrite checkpoint and the ladder heads refer to the same exclusively
owned KV bundle; a checkpoint is not a second KV allocation. On a live lane, the ladder GDN is the
host log of the SequenceState (`prefix_matches` against that lane's ledger), not a GDN-only row in
`KVRamCache::plan_match`. The reuse paths for an incoming prompt are:

- if the prompt ends exactly at the current resume frontier and the decode anchor is complete, it
  becomes decode-ready directly;
- if the prompt fully contains the current resume frontier and has a suffix, the suffix is prefilled
  from that frontier;
- if the prompt ends exactly at a matching rewrite checkpoint, the checkpoint is restored and
  finalization is completed from its hidden state;
- if the prompt matches a saved rewrite checkpoint and has a new suffix after it, the bundle is
  claimed, truncated to the checkpoint frontier, the checkpoint is restored, and then the suffix is
  prefilled;
- if the prompt matches a ladder head at `F`, it goes through `restore_context_checkpoint`: the
  head's GDN is loaded into current, the hidden into `tail_hidden` (the rewrite hidden is not
  written), KV is trimmed to `F`, heads with `frontier<=F` (including `F` itself) are kept, and
  heads with `frontier>F` and a resident rewrite with `rewrite.frontier>F` are dropped; DFlash also
  restores the cyclic state from that head and sets `dflash_context_frontier=F`;
- if the prompt matches the turn-rollback head at the previous completed `E`, it goes through
  `restore_turn_rollback`: the same complete checkpoint recipe (GDN to current, hidden to
  `tail_hidden`, MTP `mtp_kv_valid=E-1`, DFlash cyclic and `dflash_context_frontier=E`). A VRAM D2D
  copy `C->current` (staging slot) is used only when the staging identity is still that head;
  otherwise the host image is unpacked. A ladder freeze that borrows the staging slot must put the
  rollback head back;
- if the common prefix ends at any other position without a complete checkpoint, it is a cache miss.

The planner first tries the current frontier; otherwise it takes the longest complete checkpoint
among the rewrite checkpoint and the ladder / turn-rollback heads on that bundle; otherwise it does a
full reset. `TurnClosure` / `ResponseReplay` only record the capture purpose of a snapshot; they do
not change the mathematical content or the page ownership of the payload. As long as the full prefix
identity matches, the runtime may restore an existing snapshot across the current desired kind, and
reports the restore path by the actual snapshot kind. When the new desired frontier already lies
before the selected reuse frontier and there is no corresponding snapshot, page allocation is not
rolled back to force a recomputation; the existing snapshot is kept for this request and the new
capture is deferred.

The last case is not a page-size limitation. What is missing is the corresponding Linear
Attention/backend continuation state, not a KV page. When decode-only crosses a mark and prefill never
froze at the end of that chunk, there is no ladder head.

### 10.2 Non-page-aligned frontier

For example, Main Text `P=64` and retained frontier `F=1000`:

```text
mapped main pages = ceil(1000 / 64) = 16
last page valid offsets = [0, 40)
```

After the ownership transfer, execution can continue from logical position 1000. There is no need to
round the frontier down to 960, nor to copy or recompute the last page. The backend allocation is
described independently with its own `P` and frontier, and is not required to align with the main
tail page.

### 10.3 Exclusive bundle ownership

A retained bundle can be claimed by only one active request at a time. A pin means that the retained
entry holds the allocations in each pool, so that the corresponding IDs do not belong to the
respective free sets; no page refcount or special allocator role is needed.

When a rewrite checkpoint is selected, the claim transaction keeps the last partial page that contains
the checkpoint frontier and returns the complete pages after it to their pools; then each pool's exact
frontier and the fixed continuation state switch to that checkpoint together. This process does not
copy the retained KV payload.

Ordinary requests consume one retained bundle in place. `cache_write: false` forks a selected
retained prefix into independent allocations and complete continuation state. It does not share
writable pages, mutate source checkpoints, or consume a RAM/disk original. Copy main/backend KV
in their stored representations, including scale planes and partial final pages; preserve exact
logical frontiers and initialize GDN, hidden, position/media metadata, and MTP/DFlash state at the
same selected frontier. Transfer readers protect source allocations and images through fences.

If the source prevents admission, ordinary state may migrate into a full-original RAM image,
without the normal closed-turn rewrite cut, before its GPU bundle is released. Preserve existing
checkpoints without capturing new ones. An independently retained disposable source may only be
forked when another preserving destination is feasible; it is never offloaded.

Disposable completion retains only coherent current continuation state in VRAM, with no reuse
checkpoint captures and no RAM/disk persistence. It is the first eligible retained victim for
both lane replacement and page pressure. A later true request can promote its own result.
Insufficient permanent preservation capacity returns a bounded overload rather than destructively
claiming the source. Cancellation releases destination resources after reader fences; a committed
original migration remains cached. See the concurrent architecture for admission and cleanup.

### 10.4 Host RAM second tier

The optional pinned-host tier holds an ordinary exclusive FIFO of completed retained SequenceState bundles
that are not on a VRAM lane. Capture happens at the executor admission site that is about to
destroy that bundle and appends a fresh D2H image at the FIFO tail. A capture that does not fit
the host budget increments `drops` once and the incoming request still proceeds. Capture reports
`Captured`, `NeedsEviction`, or `Dropped`; only `NeedsEviction` permits the caller to spill/evict
residents and retry. An image larger than the entire arena is `Dropped` without evicting any
resident. D2H/H2D run on
`DeviceContext::copy_stream`. Rewrite-image and ladder-head copies between pinned host buffers
run as host callbacks (4 MiB chunks) on the tier's own startup host-copy stream, ordered only
after each image's fence, so they overlap the entry's KV/GDN D2H or H2D; that stream then joins
`copy_stream` before the entry's `copies_done`, which still covers every copy. Source GPU pages stay mapped until that entry's `copies_done` is
ready; unpack is not ordered behind block-table publish on the compute stream. Restore writes the
host image onto a new page mapping of the chosen free lane (an empty lane
first; a dirty lane only when none is empty, and among equal-reuse dirty lanes the
disposable bundle before ordinary bundles, then the least-recently-admitted bundle), records H2D completion on the same event, then
`device.stream` waits on that event only immediately before this lane's `start_prefill_lane`.
Capture and restore both require a selected
free lane; a queued request that cannot admit does not dump in-flight or other retained GPU
state. Consume, a rollback discard, and capacity eviction
(of the oldest unpinned resident) erase the host entry wherever it sits in the FIFO without
waiting and retire its block. A retired block is freed once its copy and block events have completed and its
last I/O pin has dropped: a disk spill still reading the image, or a disk worker's pending-copy
snapshot borrowing its event. A RAM restore claim therefore neither cancels nor waits for a spill
of the same entry; the spill commits from the pinned bytes. Harvest bills only completed copies;
a retired entry's never-harvested copies count only toward the lifetime totals.
Both then follow the same checkpoint decision as VRAM prefix reuse.
Before requesting capacity eviction, capture checks whether the image can fit a contiguous
arena interval after removing unclaimed residents. A claimed restore source divides the arena
into fixed gaps; if none can fit the image, capture drops once and preserves the other residents.
Retired blocks and idle-spill I/O pins are temporary and do not count against this decision. A
blocking reclaim waits for a victim's copies or a retired block's fence; a non-blocking reclaim
selects only victims whose copies have landed and otherwise defers the admission.
Logged `used`/`entries` (human `kv-ram=` / `n=`) are live host residents only. Serve and CLI also
print CUDA D2H `save=` and H2D `load=` elapsed harvested after the first non-throwing
`start_prefill_lane`. The host image packs logical page `i`
densely and neither stores nor sorts physical page IDs. A RAM hit still requires a complete
SequenceState proof; active pages are not moved because of RAM capture/restore. Capacity is fixed
in MiB by `--kv-ram-capacity`, default off. Construction pre-faults the complete contiguous host
arena in parallel and registers it with CUDA before startup succeeds; a failed host reservation or
CUDA registration fails Engine construction.
One long MTP or DFlash bundle with six ladder heads is about 6 GiB (Main+backend KV plus current/rewrite GDN plus
moved live heads, including DFlash cyclic). `--kv-ram-capacity off` still keeps the live-lane pinned GDN log for same-lane
rollback; other-lane restore after eviction remains a miss without the FIFO. Eviction of a retained
bundle is the existing KV dump plus those ladder heads **moved** (not aliased) into the same FIFO
entry. A middle-head hit unpacks that head's GDN+hidden (and DFlash cyclic) into current/`tail_hidden`, so
occupy does not unpack it again, and installs only the entry's heads at or before the restored
base (occupy drops later ones) into the Program's checkpoint pool. Those head copies run on the
RAM tier's host-copy stream behind only each head's fence; the entry's copy fence, which gates the
lane's first prefill chunk, does not include them, and a separate block fence keeps the retired
block allocated until they land. A RAM or disk restore reads only the state its planned path keeps: a turn- or response-checkpoint restore skips the entry's
frontier GDN, tail hidden and DFlash lane, which `start_prefill_lane` overwrites with the checkpoint
set (the disk tier neither reads nor decodes them), and a request that drops its rewrite checkpoint
skips the entry's rewrite set and installs no rewrite checkpoint. A RAM restore starts only once the entry's own capture copies
are ready; while other lanes decode, the copy-hold polls that instead of blocking. KV
`dst_extent` Main `pages_for_tokens(F)` / MTP `pages_for_tokens(F-1)`. Host RAM is not a second GPU
working set.

`--kv-ram-system-prompts N` enables retained snapshots of the exact leading system-prefix frontier
inside the same arena. The frontend marks the boundary before any first-user content, and the
tokenizer verifies that its encoding is an exact prefix of the full prompt without changing input
tokens. Prefill splits at the boundary; capture stores only its leading mapped KV pages and current
GDN/backend state, with no stale later checkpoint. Ordinary and DFlash snapshots omit the tail
hidden; their strict suffix prefill supplies it. The copy stream waits for the producing compute,
and subsequent compute waits for that entry's D2H event before mutating its partial page or
recurrent/cyclic state. All claims, H2D readers and retired blocks retain the existing event
lifetime rules.

A successful system restore releases its claim without retiring the source and touches LRU recency.
Count replacement evicts the least recently restored unclaimed system entry; byte pressure first
reclaims ordinary entries through their normal RAM/disk policy, then older system entries. Ordinary
spill enumeration excludes system anchors, so they remain RAM-only and ordinary FIFO pressure cannot
evict them. At most N indexed system entries are retained, with no capacity beyond the existing host
arena. Optional capture failure preserves request progress. The planner continues
longest-ready-prefix ranking across all retained classes. A missing system image behind an already
longer selected frontier does not force a cold restart just to populate an auxiliary entry.

### 10.5 Inclusive SSD third tier

`--kv-disk-capacity` enables an inclusive SSD store of completed bundles and requires
`--kv-ram-capacity > 0`. Each ordinary RAM capture requests a write-behind spill on the disk worker, so
entries usually become durable while other requests decode. When a capture needs RAM, the oldest
unpinned disk-durable entry is evicted first. Without one, the in-flight write-behind spill, or
else the oldest savable entry, is spilled first: synchronously when no other lane is decoding,
otherwise at emergency priority on the disk worker while the admission stays queued and the
other lanes keep decoding. The admission retries once that spill commits, so decoding never waits
for a disk write and an entry is dropped unsaved only when the disk cannot store it. The
admission's own earlier captures, which a deferral rolls back, are never that spill or drop
target; when only they remain, the capture is dropped and the admission proceeds. Equal reuse prefers VRAM, then RAM, then disk. A disk hit claims an
immutable entry, and each unlocked payload read acquires a per-extent lease. The hit restores
directly into the selected VRAM lane and never parks the restored bundle in the RAM FIFO.
Consuming the claim releases request ownership but does not delete the durable entry; a later VRAM or RAM hit likewise leaves the disk copy
available. C=1 does not submit restore reads until the selected retained victim has been captured
and its GPU pages released.

Disk format v8 stores objects in reusable, 4 KiB-aligned extents in per-kind pack files of at
most 1 GiB under `packs/1`. `PACKSET` selects an object-index epoch and reserves monotonically
increasing object IDs; its epoch does not identify a second payload copy. The immutable base
map and CRC-framed Put/Delete/Move journal resolve stable object IDs to extents. Record codecs and
CRC32C checks remain unchanged. Adjacent page writes use one `pwritev`; reused nonadjacent slots
are submitted as separate contiguous runs.

The existing `--kv-disk-capacity` MiB value bounds the sum of pack-file lengths, including
interior free extents and writes reserved before I/O starts. Extents are allocated from a
coalescing best-fit index before growing a pack. A completely free segment is truncated with
its slot unavailable to writers until truncation finishes. Index, entry, and filesystem metadata
is additional overhead. Metadata-only journal checkpoints bound the index: after the log exceeds
the greater of 64 KiB or twice the current object population's frame bytes, a quiescent worker
writes a new base and empty log, syncs their namespace, and atomically publishes the index epoch.
Old index files can then be removed without moving any payload. Captures that cannot fit a
free extent or bounded pack growth are optional misses, not inference failures. Reopening with
a smaller budget evicts ordinary and skipped entries until pack lengths fit; fragmentation may
require evicting extra entries to release a whole segment.

The publication order is segment bytes and segment namespace, object-map log, entry metadata,
then manifest. Each preceding stage is synchronized before the next one becomes durable. A crash
may therefore leave pack garbage or an unreachable map Put, but never a published entry that
depends on an undurable extent. Startup accepts those unreachable records, derives future append
frontiers from the durable pack namespace, and validates direct-I/O support only for live,
skipped, or held page/state objects. It memory-maps object maps, validates sorted extent ranges,
and checks each live page CRC once across the bounded reader pool. Torn map tails are truncated;
conflicting IDs, overlapping extents, corrupt live records, and unsupported fingerprints
invalidate the affected store or entry according to their ownership boundary.
Startup prepares index/FIFO or skipped-entry storage before acquiring object references.
Validation workers contain their own exceptions, so transient allocation failure rejects an
optional entry without terminating the process or deleting its durable image. If manifest
enumeration must restart after partial publication, the rebuild resets the index, FIFO, skipped
entries, and their reference counts together. Validation-cache cleanup releases its storage
without allocating.

Before the entry metadata rename, the spill worker prepares the replacement index node,
FIFO capacity, predecessor object lists, and uncertainty-hold capacity. Publication then updates
ownership without allocating; failed preparation drops the optional spill while preserving the
previous generation and releasing its pins. Spill page batches have a fixed maximum of eight
jobs, and idle-to-emergency promotion transfers its queue without allocation.
Spill preparation transfers the caller's RAM pin only after all jobs are queued, so a failed
queue allocation leaves exactly one caller-owned pin to release. The state-blob encode runs
with the index mutex released for both codecs, after the filesystem-space refusal it cannot change: it compresses (zstd) and seals each
record's codec header and record CRC32C, so the commit only reserves an object ID and extent
under the mutex; a cancelled worker prepare stops between 8 MiB compression chunks. A page
batch likewise builds its record headers and CRCs unlocked from the I/O-pinned RAM image, then
relocks to reserve IDs and the contiguous append range. An object-ID reservation that must
extend `PACKSET` writes it unlocked; only the installed spill session allocates IDs, and it
blocks index checkpoint publication. After relocking, preparation re-validates cancellation, a claim of the
entry an Extend or Refresh would rewrite, and quarantine or eviction of its source, and abandons
without marking the RAM entry failed so a retry re-plans. A commit's unlocked object sync and map append
use object records copied under the mutex. MANIFEST images are sequenced under the index mutex
and published by one writer at a time (tmp write, fsync, rename, directory fsync); an image older
than the last published one is dropped. Branch sharing reserves all
rollback bookkeeping before acquiring references; allocation failure cannot leave an untracked
shared reference.

The fingerprint describes each packed logical-page plane by dtype, plane order, logical
leading/head extents, and packed bytes per page. It deliberately excludes the resident GPU page
count, block-table row count and logical capacity, allocation alignment, and device strides. Those
are destination-pool properties rather than disk-record properties: restore scatters the canonical
packed bytes through the current pool geometry. A cache directory can therefore reopen after
`--vision` changes an auto-sized resident KV pool, while model/weights identity, KV dtype, page
size, logical plane schema, speculative backend, and persistent GDN/DFlash state geometry remain
strict compatibility boundaries. Vision media identity remains in each entry's exact identity
record, so changed image or video content cannot reuse a stored prefix.

The v8 fingerprint binds model/weights identity and a SHA-256 digest of target-selected tensor
names, shapes, numeric formats, storage layouts, encoded byte ranges, and their contents. The
artifact reader maps logical row selections to registered code/scale/divisor ranges and hashes
in canonical name/range order; dependency selection belongs to each target package. For 27B:

- Text embedding, layers, and final norm produce retained KV/GDN and tail/checkpoint hidden state.
  Output heads, proposal heads, and proposal token maps are excluded.
- Vision weights remain included even when Vision is disabled at startup: the directory can
  retain media entries across Vision on/off reopens.
- Stored MTP tensors are excluded because MTP execution is disabled.
- DFlash2 includes feature projection, context norm, per-layer key norms, and fused projection
  key/value rows. Its query, output, MLP, convolution, final-norm, and selector paths produce
  proposals, not retained context KV. Inactive backend weights are excluded.

For packed row selections, codes and corresponding stored scales are included; NVFP4 also
includes the shared weight divisor. Its row selections align to complete 128-row scale tiles.
Filesystem device/inode numbers, timestamps, pathname, total file length, directory formatting,
object order/placement, padding between objects, and frontend resources are absent. A copy or
repack of the same retained-state dependencies is compatible; changing a selected numerical
value or representation is not. Hashing reads the selected bytes only when disk caching is
enabled. Artifacts remain immutable while an Engine uses them. Frontend resources determine
prepared inputs and sampling defaults, not the numerical state for an already represented
prefix. Each entry independently matches token IDs, token types, MRoPE positions, and exact
media identity.

The attention profile (`sage_attn`, `keep_frac`, `xattn_tau`, and `xattn_min_len` when `xattn_tau <
1`) is a directory boundary because it changes the computation that produced retained state. KV
dtype, speculative backend, page size, logical plane schemas, GDN byte extents, per-lane DFlash
cyclic geometry, and snapshot version remain boundaries. Allocated DFlash lane count is excluded:
the disk image stores one lane and restore selects a destination lane. Sampling parameters, draft
token count, transport settings, RAM/disk budgets, compression choice, GPU resident capacity, and
request concurrency do not invalidate a directory unless they also change the recorded per-lane
geometry. Model/weights identity identifies the target's fixed mathematical contract; code changes
that invalidate persistent state must advance the snapshot version. Pre-v8 stores are rejected
rather than silently adopted or automatically deleted.

Exact-frontier reuse also requires a valid target tail-hidden image. In particular, speculative
DFlash cancellation can retain the committed KV/GDN/context frontier while invalidating that
image. An exact-prefix request must then use a valid checkpoint or recompute; a longer suffix
may still reuse the frontier because suffix prefill produces fresh hidden state. This eligibility
rule applies equally to VRAM, RAM, and disk candidates.

Spill refresh, extension, and branching compare the exact represented input before sharing
parent KV pages: token IDs, token types, all three MRoPE position axes, and complete media
identity and geometry must match. The shared prefix ends before any changed or incomplete
Vision item. A parent ticket or equal token IDs alone does not permit page reuse.
For MTP, the last backend slot also depends on the sampled ledger token at the execution
frontier. A same-frontier refresh preserves the full backend only when that token's identity
and the valid extent are unchanged; otherwise it rewrites the affected tail, including any
new page required when an exact-hit bridge extends validity after cancelled prefill.

Whole-store payload generations are not rewritten. Retirement requires a durable entry tombstone, zero live,
skipped, and uncertainty references, and drainage of the per-extent leases that unlocked
payload readers acquire with their copied locations. A CRC-framed Delete becomes durable
before its extent enters the free index. Replacement data is then written and synchronized,
its Put is synchronized, and only then may entry metadata publish it. Crash recovery replays
Put/Delete/Move frames before validating final extents for overlap, reconstructs free space from
those extents and actual segment lengths, and preserves the durable object-ID high-water mark.
An interrupted metadata checkpoint retains both index epochs until `PACKSET` publication is
durable; no reuse or new spill publication can rely on an uncertain epoch. No payload generation
is copied or retained for index checkpointing.

The Engine explicitly sets cache idleness only when every lane, pending ingress, and copy hold
is empty; a RAM capture's write-behind permission does not enable defragmentation. Admission
clears idleness before request execution, including disposable requests. After 250 ms idle,
maintenance drains reader-delayed retirement and truncates free tails. A single candidate pass
visits each kind in increasing address order, selecting source packs with at least 25% free bytes and
the earliest adequate earlier hole of the same kind, preserving contiguous page order when
the available hole can hold the run. The destination never grows a pack. Copies use a
1 MiB buffer and at least 16 ms per chunk (at most 62.5 MiB/s copied payload), with cancellation
checks between chunks and before sync/publication. Restore, prefetch, and spill work take
priority. Existing holes are coalesced without payload copying; fragmented space without an
adequate hole is deferred, and maintenance never evicts an entry to fund relocation.

A relocation holds its source object and verifies its stored-record CRC while copying. It syncs
the destination, durably appends a Move for the same object ID, then publishes the location.
Move preserves kind, extent/encoded/logical sizes, and CRC. Old locations retire independently
of IDs and remain occupied until their individual read leases drain; a later Move or Delete
cannot release an older reader's extent. Cancellation before publication releases the temporary
destination. Once Move is durable, the location handoff finishes even if a request arrives.
A failed journal append truncates and syncs its original boundary before later appends. If that
rollback also fails, journal/checkpoint/spill writes freeze and both possible locations stay
reserved until reopening resolves the journal. Startup needs only the final selected locations
because readers do not survive a restart.

Eviction uses durable tombstones so an uncertain
metadata publication cannot make a referenced object reusable. Capacity and quarantine
eviction are two-phase: victims are selected under the mutex (capacity eviction projects the
unique bytes released by FIFO victims so one round covers the shortfall) and made unavailable
to planning, claims, spill tickets, and RAM durability; their tombstones are written, synced
and renamed, and the tombstone directory fsynced once per round, with the mutex released.
Only a durably tombstoned victim then drops its object references and index position under
the mutex; a victim whose tombstone did not become durable keeps its references, has any
tombstone its attempt renamed removed, and returns to service unless it is quarantined. Capacity
eviction then moves on to the next FIFO candidate, and it waits for another thread's in-flight
round instead of evicting around victims whose bytes are about to be released. A tombstone is released only after a store fsync that
follows the unlocked scan finding its entry directory gone, and only when no entry, skipped
tree, or object reference still needs it. Before capacity eviction, admission
checks that the incoming incremental bytes plus unique extents protected by claims or I/O pins
(including a spill's parent) fit the budget. Known capacity shortfalls are rejected without
evicting unrelated residents. Shared protected extents are counted once. Refresh and extension reserve replacement bytes
without crediting the old image, which remains owned until publication.

Restore owns `2 * restore_io_threads` pinned page slots and equally sized device staging slots.
Readers issue aligned direct reads into the pinned slots, validate each packed record, enqueue one
dense H2D per logical page, and scatter it into the target's physical planes. Raw state objects
read directly into their final pinned owners using 4 MiB read/CRC chunks; saved checkpoint heads
decode straight into Program pool heads handed in with the restore target, only for the heads at
or before the restored base, after each head's previous DMA or host-copy owner has finished; each object uses at most
four readers and independent objects share the startup-fixed reader budget. Compressed state uses
bounded decoded owners. Contiguous CRC32C checks use the serial SSE4.2 dependency chain below
512 KiB and three interleaved hardware chains for larger payloads; incremental checks remain
serial.
The page window, state owners, destination page mappings, and object read lease remain alive until
the copy-stream completion event has been consumed. The spill side uses separate pinned scratch,
so an idle spill may overlap restore reads; emergency spill remains exclusive with restore I/O.
Restore readiness requires the completed join of page scatter and independent state H2D, not
only finished reads or empty page slots. Until that join completes, admission remains in
copy-hold so other active requests can continue decoding.
When admission uses prefetched pages, their queued reads become restore work and take priority
over idle spill. Leaving required reads behind idle spill would deadlock, because the active
restore itself prevents idle spill from advancing.
Cancellation before source invalidation drains dequeued reader claims as well as active page
and state transfers. A prefetched page retains its I/O pin from queue publication through reader
retirement, including the interval before a transfer slot is assigned and exception cleanup.
Prefix matching filters append and checkpoint candidates using the same backend-readiness
rules as resident reuse before ranking them. An unusable canceled tail cannot shadow a usable
checkpoint or another cache entry. Disk plans carry the selected committed generation; claim
checks it, so a same-prefix refresh requires replanning. A claim never waits for an idle Extend or
Refresh of its entry, not even for the commit's fsyncs: it cancels the spill, and the commit
abandons at its next check: before its `meta.bin` rename when the claim lands during the object
sync, otherwise by rolling the renamed `meta.bin` back, so the claimed generation stays published
and its objects untouched. Only if that rollback cannot be written, or the post-rename syncs fail,
is the newer generation published, with the replaced objects held until the entry is next
replaced or evicted; the claimed restore then fails its committed-generation check (at setup in
`load_host`/`restore_device`, or while running) as a `CacheRestoreFailure` cache miss, and the
request falls back to cold prefill.
MTP disk validation requires pages for the declared valid backend extent; readiness checks exclude
frontiers beyond that extent while preserving eligible earlier checkpoints.
Cache-owned CUDA event creation treats only `cudaErrorMemoryAllocation` as recoverable
allocation pressure; other CUDA statuses retain their execution-error classification. Checkpoint
head and staging events exist before DMA can borrow their buffers. RAM teardown and the bounded
admission/capture bookkeeping do not allocate.
Optional RAM/disk candidate-planning allocation failure skips that candidate at its current index
version and leaves ordinary lane/cold admission available.
An expected disk read/decode failure or RAM/disk restore metadata allocation failure before prefill
is a cache miss for the request. Drain its worker and CUDA owners, release the claim, and
quarantine the failed disk entry or discard the RAM image before recomputing
without reuse. Re-admission preserves request identity and FIFO ordering, rechecks the cold GPU
reservation, and does not apply the initial queue deadline to an already-admitted recovery.
The fallback is bounded to one cold-prefill retry, preserving the request's checkpoint-capture
settings. Invariant and execution failures are not
cache-miss signals. Quarantine marks the entry unavailable and invalidates RAM tickets naming it at once,
without waiting for claims, restores, or spills that pin it; a spill that extends, refreshes, or
branches from it abandons its commit. The disk worker then evicts it with normal
tombstone/refcount eviction once its pins drop; if tombstone persistence fails, the entry stays
unavailable in memory without dropping shared-object references. Disk-worker allocation and I/O
failures (an unopenable pack, a failed read) surface as the cache-miss signal; invariant
violations keep their type.

---

## 11. Speculative decoding

The MTP/DFlash runtimes do not manage their own device-memory allocator. Their growing KV is still
obtained from the target-defined backend pool in the common KV Store, which is physically separate from
the Main Text pool.

The KV procedure of one speculative round is:

```text
1. main/backend reservations were completed at admission
2. the round boundary materializes, per pool, the largest position this round can touch
3. target writes provisional Main Text KV
4. backend writes provisional MTP or DFlash KV
5. acceptance produces target-owned committed frontiers
6. rejected tails become logically unreachable in each pool separately
7. each pool independently releases the complete trailing pages that are no longer referenced
```

Different requests accepting different proposal lengths only changes their own bundle frontiers; it does
not create a new pool type. Rejected bytes may remain in a partial tail page; a later append must fully
overwrite the corresponding code and scale planes before they become valid again.

The KV Store does not understand proposal, verify, or acceptance, and does not derive the relationship
among the Main Text, MTP, and DFlash frontiers. The target submits the final frontier of each pool as the
transaction result.

---

## 12. KV resources outside growing pools

Not all K/V-shaped state grows with context:

| Resource | Storage class | Reason |
|---|---|---|
| target full-attention KV | Main Text paged pool | grows with every target materialized token |
| MTP persistent KV | MTP paged pool | independent MTP frontier |
| DFlash local sliding-window KV | fixed per-sequence KV state | capacity fixed at the window, does not grow with total context |
| DFlash boundary-local snapshot | fixed per-sequence KV state | fixed checkpoint payload |
| Vision/temporary query K/V | shared execution workspace | lives only within one operator/phase |

### 12.1 Fixed cyclic physical contract

The DFlash local sliding-window and the boundary-local snapshot use the same closed cyclic layout but own
independent storage. Context-checkpoint freeze has an additional Engine-wide 1-lane staging window (same
geometry, `lane_capacity=1`): pin/freeze first copies the live lane D2D into that window and then D2H from
it; a suffix only changes the live local state. For every local layer, the currently registered target
fixes:

```text
capacity        = 4096
padded_capacity = 4096
K/V dtype       = BF16
K/V shape       = [128, 4096, 8] = [D, padded_capacity, Hkv]

nb[0] = sizeof(BF16)
nb[1] = D * nb[0]
nb[2] = padded_capacity * nb[1]

physical_slot(p) = p mod capacity
K[d,h,p] = k[d, physical_slot(p), h]
V[d,h,p] = v[d, physical_slot(p), h]
```

Each active sequence holds five layers of local K/V and five layers of boundary-local K/V of the same
shape. Both sets of plane bases are stable for the lifetime of their fixed state unit; save/restore copies
the complete five-layer payload, and aliasing must not make the live local state and a checkpoint share
writable storage.

`CyclicKVCacheLayerView` carries only contiguous K/V tensors, the capacity, and the geometry; it carries no
block table, page ID, allocator handle, or mutable frontier. The caller defines the valid content with
absolute positions and a live interval. Cyclic storage does not enter growing pools, does not take part in
page reservation, does not use `PagedKVLayerView`, and does not change its modulo/window semantics because
of paged migration.

Fixed KV state is obtained from a fixed-state pool managed per admitted sequence. The Engine must always
be able to provide such fixed resources for `max_concurrency` active sequences. A currently retained entry
uses only the idle fixed unit of its original lane; a new admission may claim a retained entry, or first
evict it, and must not lower the active concurrency guarantee.

---

## 13. Correctness invariants

1. A page-group ID belongs to at most one active or retained allocation within its own pool;
2. pool grouping contains only planes that share frontier, lifetime, and page geometry;
3. a reserved-but-unmapped entitlement cannot be consumed by another request or retained entry of the same
   pool;
4. admission succeeds or fails atomically for the complete pool reservation vector;
5. all pool mappings within one execution unit remain unchanged;
6. an operator may read only positions inside the valid frontier of the corresponding pool;
7. before a frontier is published, the complete K/V and the necessary code/scale planes of that pool
   position must already be written;
8. page recycling does not require zeroing, but a new owner must not derive valid state from an old frontier
   or stale bytes;
9. page mapping uses the cache ordinal, not the RoPE/MRoPE coordinate;
10. a prefix hit must be proven by a complete SequenceState; a KV token match, page match, or RAM
    hash prefilter is not itself a hit; VRAM and RAM hits use the same checkpoint proof;
11. active pages are not moved by compaction, retained eviction, host-RAM capture/restore, or other
    request growth;
12. the legal maximum per-pool growth of an admitted request must always be covered by the reservation
    vector;
13. `KVCapacityProfile` must be derived uniquely from `max_context`, `kv_capacity`, `max_concurrency`, and
    the selected backend according to §3.2-§3.3; MTP physical headroom does not enlarge the per-allocation
    logical capacity, and the backend must not be exhausted before the Main entitlement contract;
14. a logical block of one pool allocation uses the same page-group ID in all grouped planes of that pool;
15. an active slot table row only mirrors the allocation mapping and owns neither page payload nor frontier;
16. growing-cache Ops consume only `PagedKVLayerView` or `PagedKVBatchLayerView` and obtain no allocator,
    request identity, or lifecycle authority;
17. the logical axes, the pool-specific closed physical order, the exact strides, and the storage bounds of
    K/V, code, and scale must be consistent with §4.2;
18. a page-group ID selects the same logical page group in all grouped planes, but these slices are not
    required to be physically adjacent;
19. DFlash local and boundary-local each own the complete fixed cyclic payload of §12.1; an absolute
    position selects a physical slot only through `p mod 4096`, and the two do not share writable backing
    storage;
20. MTP Vision prefill, as well as prefix reuse and ordinary tail rounds under any selected backend, must
    not degrade a sequence into one that lacks that backend's continuation state.

---

## 14. Edge cases

| Case | Required behavior |
|---|---|
| context ends in the middle of a page | advance only the exact frontier, keep the partial tail page of the corresponding pool |
| prefill chunk spans multiple pages | materialize the required pool pages per pool before launch; mappings within the unit are frozen |
| speculative tails differ | Main, MTP, or DFlash trim independently by their own frontiers |
| accepted length is zero | the target decides per-pool progress; the KV Store produces no special state |
| retained frontier is not page-aligned | reuse at the exact original offset, no rounding to a page boundary |
| prompt ends exactly at a saved rewrite checkpoint | after exclusive claim, restore the complete checkpoint and finish finalization from the checkpoint hidden |
| a saved rewrite checkpoint is hit and a suffix exists | after exclusive claim, truncate pages, restore the complete checkpoint, then prefill the suffix |
| only a shorter token prefix matches, but that position has no checkpoint | cache miss |
| retained occupancy blocks admission | Prefix Cache evicts a complete entry, then atomic admission is retried |
| after retained eviction, the request set satisfies the main contract but the backend reservation fails | startup sizing or accounting invariant violation; not a normal waiting condition |
| request cancellation | an unfinished prefill rolls back to the turn-rollback/ladder head on the occupy base, otherwise to the rewrite checkpoint, discards the staged heads after it, and retains; if the occupy base is 0 and there is no rewrite, the bundle (including staged heads) is released; an ordinary in-flight decode has already overwritten GDN/hidden, so the bundle is released, and only rolling back the ledger and caching afterwards is not allowed; a speculative in-flight decode discards the provisional result of this unit and folds with `commit_columns=0`, then retains the committed frontier; a decode/prefill frontier already committed between units is retained like OutputLimit, together with that frontier and the context-checkpoint heads before it |
| no free page when an admitted request materializes | reservation-accounting violation, treated as an Engine failure |
| one request uses most of the main pool capacity | legal, as long as the other per-pool entitlements still satisfy the invariants |

---

## 15. Consumer boundary

Paged KV storage serves both single-sequence prefill/cached routes and batched decode routes:

```text
shared typed page planes + fixed block-table matrix
                    │
          ┌─────────┴─────────┐
          ▼                   ▼
one selected table row   table_rows[B] selectors
single-sequence Op       batched decode Op
```

A storage view interprets only the physical planes and page translation. Per-row context lengths, valid
columns, and compact batch membership are expressed by the typed inputs of the consuming Op; the allocator,
frontier commit, and prefix ownership never enter the view.

### 15.1 `PagedKVLayerView`

A single-sequence growing KV consumer uses one non-owning typed view. The Tensor shape expresses both the
logical extent and the closed physical order of the owning pool. The logical fields are:

```text
PagedKVLayerView
├── k_pages           Tensor (route-closed physical axes)
├── v_pages           Tensor (route-closed physical axes)
├── k_scale_pages     optional Tensor (same pool order)
├── v_scale_pages     optional Tensor (same pool order)
├── block_table       I32 Tensor [Nlogical]
├── head_dim          D
├── num_kv_heads      Hkv
├── dtype             BF16 or I8
└── quant_group       0 or 64
```

`P` and `Nphysical` are given by the route's interpretation of the page tensors' shape, and the logical
capacity is `Nlogical*P`. All currently registered growing routes accept only `P=64`; the kernel treats it
as a compile-time fact instead of performing a runtime division in the inner loop. Main/MTP causal routes
accept only contiguous `[D,P,Hkv,Nphysical]` and scale planes of the same order. An Op does not accept a permuted, arbitrary-stride, or
other-pool-profile growing view.

Each layer view of the same pool references different plane slabs but the same active slot block-table row.
A view does not contain:

- an allocation, reservation, or free-list handle;
- request identity, slot identity, or batch row;
- a mapped extent or a valid/provisional frontier;
- prefix-cache ownership or backend mode;
- per-page K/V pointers.

These facts belong respectively to the KV Store, the target sequence state, or the scheduler. The consumer
wrapper validates the static range from the Tensor shape and the execution envelope; the caller guarantees
that the actually accessed positions are mapped and lie inside the valid domain declared by that Op.

### 15.2 `PagedKVBatchLayerView`

A batched growing KV consumer uses the same set of typed plane tensors and the complete block-table matrix:

```text
PagedKVBatchLayerView
├── k_pages / v_pages
├── optional k_scale_pages / v_scale_pages
├── block_tables       I32 Tensor [Nlogical,C]
├── head_dim / num_kv_heads
└── dtype / quant_group

Op inputs
└── table_rows         I32 Tensor [B]
```

`table_rows[b]` selects the table row of one active allocation for compact row `b`. The view does not embed
`B`, membership, context length, or valid-column metadata; one exact-`B` invocation may select any active
rows that do not conflict with each other. The plane bases and the table matrix base are stable for the
Engine lifetime, while the row selectors and the table content are round data.

### 15.3 Device address contract

For the current `P=64`, both kinds of route share the page translation:

```text
logical_block = position >> 6
page_offset   = position & 63
physical_page = block_table[logical_block]

Main/MTP element_address = plane_base
                         + d             * nb[0]
                         + page_offset   * nb[1]
                         + kv_head       * nb[2]
                         + physical_page * nb[3]

DFlash element_address  = plane_base
                         + d             * nb[0]
                         + page_offset   * nb[1]
                         + physical_page * nb[2]
                         + kv_head       * nb[3]
```

A batched consumer first uses `table_rows[b]` to select `block_tables[:,table_rows[b]]` and then performs
exactly the same logical-block translation; the batch axis does not change the pool layout or the page ID
domain.

INT8 scale uses the quant group `d/64` as its first-dimension coordinate and the scale Tensor's own `nb`.
Once a key tile has obtained `physical_page`, the K, V, code, and scale of the same pool all reuse that
page-group ID. The exact strides are determined uniquely for that pool by §4.2.

The formulas above are the Op contract; they do not require a production kernel to perform four generic
integer multiplications per element. The wrapper validates the route-closed strides; the CTA computes the
base at page/head granularity and broadcasts it, and the inner loop keeps using static row/vector offsets.
A production kernel does not check whether a page ID is a sentinel; mapping completeness is already
guaranteed by execution-unit materialization.

### 15.4 Pool-to-consumer binding

Before calling an Op, the target/runtime composes two kinds of views that do not own each other:

```text
pool.layer_planes(layer)
        +
selected allocation table row or table matrix
        =
PagedKVLayerView or PagedKVBatchLayerView
```

A Main Text consumer can bind only the Main Text pool, and an MTP consumer can bind only the MTP pool. A common Op does not receive a pool kind and does
not select a pool by target, layer role, or backend string.

---

## 16. Op contracts

Growing-cache parameters accept only a paged view; there is no parallel continuous overload, and no second
mathematical Op named `paged_attention`. Paging is a cache storage/addressing contract; the Attention
formula, visible domain, and per-sequence isolation are unchanged. The exact ABI is owned by the
corresponding `include/ninfer/ops/` header; this document defines only the storage/view boundary.

### 16.1 Affected entries

| Entry | Cache contract | State effect |
|---|---|---|
| `gqa_attention` | writable `PagedKVBatchLayerView` + `table_rows[B]` | appends valid K/V columns for `B` independent sequences and performs one ragged causal Attention |
| `gqa_attention_cached` | read-only `PagedKVLayerView` | reads only an already-populated paged cache |
| `gqa_kv_append` | writable `PagedKVLayerView` | writes all supplied rows, BF16 copy or INT8-G64 encode |
| `normalized_rope_kv_append` | batched `CyclicKVCacheLayerView` + lane selectors | DFlash local fixed window, not part of a growing pool |
| `swa` | batched `CyclicKVCacheLayerView` + lane selectors | DFlash local fixed window, not part of a growing pool |

These entries do not form a parallel storage-overload family. The workspace capacity query is determined by
the existing execution envelope and token interval; paging does not become a workspace route and does not
request any additional transient buffer proportional to context length.

A growing `KVCacheLayerView`, and any implementation that depends on a contiguous `[D,padded_context,Hkv]`,
is not part of the product. Fixed cyclic storage owns an independent layout/container, does not reuse the
paged pool, and has no continuous-growing backing abstraction.

### 16.2 Tensor domains

Main Text and MTP causal `gqa_attention` accept request-major batched tensors:

```text
Q/Out          BF16 [D,Hq,W,B]
K/V            BF16 [D,Hkv,W,B]
positions      I32  [W,B]
valid_columns  optional I32 [B]
table_rows     I32  [B]
```

`B=1` covers the single-sequence prompt/decode route; `B=2..4` covers batched decode/verify with `W=1..16`.
`valid_columns[b]` makes only the prefix `[0,Vb)` of each row take effect. `positions` is the KV logical
ordinal, not the RoPE/MRoPE coordinate. For query position `p` of row `b`, the causal visible domain is
still `[0,p]` of that allocation; a page boundary produces no mask boundary.

`gqa_attention_cached` and `gqa_kv_append` keep the single-sequence `[D,H,T]` tensor domain for the split
prefill path and do not form a second growing-cache storage contract.

### 16.3 Wrapper validation and caller promises

The wrapper must validate:

- the logical geometry of the K/V page tensors is consistent, `P=64`, and they exactly match the closed
  contiguous order of that route in §4.2;
- the physical page count, head geometry, dtype, and optional scale planes are consistent;
- the block table of a single view is a contiguous I32 `[Nlogical]`; the table matrix of a batch view is a
  contiguous I32 `[Nlogical,C]`, and the row selectors are a contiguous I32 `[B]`;
- a BF16 cache carries no scale planes, and the scale shape and strides of an INT8-G64 cache are complete;
- causal `max_visible_keys <= Nlogical*P`;
- DFlash `max_context <= Nlogical*P`;
- the input/output Tensor domain is consistent with the registered geometry of the current entry.

The wrapper does not read device positions, context lengths, commit counts, or page IDs. The caller
guarantees that:

- the execution envelope covers the exact device values of this invocation;
- the page tensors are established by the pool binding within the corresponding registered plane storage
  bounds;
- every logical block that may be accessed at most in this invocation has been materialized before launch;
- the read domain does not exceed the target-owned valid frontier;
- all K/V code/scale bytes of the writable rows will be completed before frontier publish;
- all selected block-table rows remain unchanged while the Op is in flight.

For a device-count prefix append, materialization is done by the maximum possible per-row write extent of
the envelope, rather than by guessing exact counts that are invisible to the host. The rejected tail of
each row is not written, and the Op itself does not advance the frontier.

---
## 17. Kernel addressing design

### 17.1 Addressing primitive

A kernel receives only the layer plane bases, one block-table row pointer, and the existing execution
inputs. Page translation should be a route-local inline device primitive; the current production
routes specialize at compile time on `P=64`, its closed physical order, the head geometry, and the
codec. Page/head bases are computed at CTA or tile granularity and shared; the stride formula must not
be left in place on every scalar/vector access.

Do not implement a generic runtime accessor that interprets an arbitrary page size, an arbitrary
layout, or a cache-kind variant in the inner loop. Narrow primitives such as shift/mask, page-base
calculation, and the INT8 codec may be shared, but BF16, INT8, causal prefill, causal small-T, and
DFlash context each keep their own independently optimizable kernel body.

### 17.2 Causal small-T BF16

The existing small-T partial/reduce decomposition is unchanged:

```text
partial grid: (KV head, split)
reduce grid:  (Q head, D chunk, query token)
```

No batch dimension is added. The addressing constraints are:

1. growing-cache addressing contains no flat cache index based on `padded_context`;
2. for each logical key tile, the page ID is obtained once and the page-local K/V bases for the
   current KV head are computed;
3. the page ID is shared within the CTA and is not looked up again by each vector lane;
4. the current K/V rows still participate in Attention directly from the input Tensor, and are also
   written to the corresponding page;
5. no split depends on another split's cache write;
6. the partial accumulator, softmax statistics, and reducer layout do not change because of paging.

The current BF16 key tile is 32 keys, so `P=64` contains exactly two tiles. The route planner must
avoid keeping a `ceil(window/active_splits)` partition that yields arbitrary `split_start` values. The
normal path allocates whole 32-key tiles; if a very short context splits a tile to increase
parallelism, the start/end of the sub-page span must also fall within the page-local valid range. No
vectorized global load may cross two physical pages.

### 17.3 Causal small-T INT8-G64

INT8 codes and FP16 scales use the same page ID. The translation order for one key tile is:

```text
logical tile -> physical page ID
             -> K/V code page bases
             -> K/V scale page bases
```

The 32/64-key implementation blocks both divide `P=64`. A route may continue to choose different warp
counts, key blocks, and dynamic shared-memory profiles for different T and context, but the split span
must be expressed in page-compatible tile units. The existing empirical spans such as 24 and 480 are
not semantics; if they produce tiles that cross pages, they must be re-chosen during paged route
tuning, and a per-element boundary branch must not be introduced to preserve the old heuristic.

The append/encode stage should use at least `(token, kv_head)` as the unit that shares page
translation. The four 64-d quant groups must not each reload the same block-table entry from global
memory. The codec formula and the final code/scale bits are unchanged.

### 17.4 Causal prefill and cached prompt route

Prefill keeps its existing two-stage effect:

```text
fill current K/V rows into cache
    -> Attention reads complete populated history
```

The fill kernel selects a page for each logical token. One execution unit may start in the middle of a
page and span any number of pages; prefill chunks, the retained frontier, and positions[0] are not
required to be page-aligned.

The Attention key loop continues to traverse from logical position 0 up to the upper bound visible to
the current query. The current BF16 and INT8 prefill key tiles are both 64, so one complete key tile
corresponds exactly to one physical page:

```text
logical block b -> block_table[b] -> one K page group and one V page group
```

Each tile's page ID is reused between the K stage and the V stage. The existing Q staging, online
softmax, MMA, and output epilogue do not change because of paging. The last partial tile continues to
zero out-of-bounds rows through the logical visibility mask, and must not read the uninitialized page
tail as valid keys.

The cached prompt route uses the same page-aware key traversal but does not perform the fill.

When the grid does not fill the GPU, the dense NVFP4 prompt splits each CTA's key tiles along grid.z:
split `z` traverses the contiguous tile interval `[z*ceil(n/Z), min(n,(z+1)*ceil(n/Z)))`, where `n` is
the number of causal tiles of that query block. Each tile is still one page, so the split boundaries
are naturally page-aligned, and the fill and visibility mask are unchanged. Each split writes FP32
unnormalized P·V, the log2-domain running max, and the exp2 sum; the merge kernel merges them stably
and writes the BF16 output.

### 17.5 Standalone append

Full append and device-count prefix append both write the final physical pages directly, without
building a contiguous staging cache.

- BF16 copy: one `(token, kv_head)` work unit looks up the page once and cooperatively copies the
  full D;
- INT8 encode: the same work unit reuses the page ID to write the code and the scale;
- sequential positions allow one CTA to process several tokens, but the page ID must be obtained
  again when crossing a page;
- the device-count prefix route first reads the valid count once, then schedules or masks only
  `[0,count)`;
- paged and cyclic append compile to different physical-address routes, with no runtime cache-kind
  branch kept on every store.

### 17.6 CUDA Graph behavior

The plane slab bases and each slot's block-table row pointer are stable for the lifetime of the Engine.
What changes across replays is the device values of the table content, positions, context length, and
commit count, not the kernel pointer arguments.

Before an execution unit that needs a new page:

```text
reserve entitlement already exists
    -> bind physical page ID
    -> update host allocation mapping
    -> publish corresponding device table entry
    -> launch/replay consumer on the same ordered stream
```

Graph capture is not keyed on page IDs, physical contiguity, or the retained owner. The mapping update
must precede the consumer, and the same row must not be rewritten while it is in flight; Ops and
kernels do not call the allocator internally, and do not wait for a host page fault.

Startup graph construction reserves one private page for each temporary row, and may write the same
page ID repeatedly into all logical entries of that row, to cover any reachable context envelope. Only
the eager code warm-up and one smoke run per executable actually access these pages; during
preparation, only the private pages corresponding to the current exact `B` are zeroed. Definition
capture and update/upload validation do not execute the consumer, so they must not scan or zero the
entire physical pool.

### 17.7 Prohibited implementations

The following implementations are not accepted even if they are functionally correct:

- gathering the paged KV into a request-contiguous buffer before every Attention;
- maintaining a separate logical page table for each layer, K/V, or scale;
- storing per-plane 64-bit pointers in the block table;
- having each scalar/vector lane independently perform the same page-table lookup;
- launching a separate CUDA kernel for each physical page;
- assuming that adjacent logical pages have adjacent physical IDs;
- falling back to a retained continuous-cache kernel at a page boundary;
- permanently padding an exact-`B` invocation to `max_concurrency` and reading inactive table rows.

---

## 18. Correctness qualification

Paging must not change the Attention mathematical result, the cache codec, or the state effects. The
existing independent Attention oracle continues to compute its result from logical positions and
represented cache values, and does not copy the production page traversal.

### 18.1 Required mapping patterns

Every affected Op executes the same logical case with at least the following three physical mappings:

| Pattern | Purpose |
|---|---|
| identity | `logical block b -> physical page b`, which eases the oracle and boundary diagnosis |
| contiguous offset | page IDs are contiguous but do not start at 0, excluding hidden base assumptions |
| fragmented permutation | logically adjacent pages map to non-adjacent IDs, proving there is no contiguity dependence |

All three must produce the same logical result and the same cache bits in the target rows.

### 18.2 Required cases

- positions/context cover tile/page boundaries such as `0, 1, 31, 32, 63, 64, 65, 127, 128`;
- prefill that starts in the middle of a page and spans one and several pages;
- the current retained frontier lies in the middle of a page and appending continues in the same tail
  page;
- a rewrite-checkpoint restore truncates a fragmented mapping, releases trailing pages, and continues
  from the exact checkpoint;
- BF16 append is bit-exact;
- INT8-G64 code and FP16 scale bits agree with the independent codec oracle;
- a cached-only route does not modify any cache plane;
- prefix append with count 0, before and after a page boundary, and the full count;
- rejected/provisional stale bytes do not enter the valid read domain;
- cyclic DFlash local behavior keeps its fixed-window semantics;
- updating the content of the same stable table row between graph replays, with the next replay
  reading the new mapping;
- several allocations in the same pool bound to different table rows, using different context
  lengths, valid columns, and row selectors in one batched invocation, still do not interfere with
  each other;
- after a page is released/recycled, a new owner cannot observe the old owner's stale bytes as valid
  state.

---

## 19. Performance qualification

### 19.1 Cost model

With `P=64`, one page-table entry is 4 bytes, while the K+V payload of a complete page read per KV
head is approximately:

| Format | K+V payload per KV head per page |
|---|---:|
| D256 BF16 | 64 KiB |
| D256 INT8-G64 incl. scales | 33 KiB |
| D128 BF16 | 32 KiB |

The page-table payload itself is therefore not the main bandwidth cost. What must be prevented, by
measurement, is repeated lookups, vector loads crossing pages, degraded TLB/cache locality, split
imbalance, and broken `cp.async`/MMA overlap of the existing routes. Performance qualification only
evaluates and adjusts route-local tile, split, lookup sharing, and staging; the storage layout is not a
route tuning variable.

### 19.2 Comparison boundary

The initial admission of paged KV uses a continuous production binary, saved separately before
deletion, as the reference. The reference and the candidate must use the same RTX 5090, toolchain,
build profile, artifact geometry, mathematical semantics, workload, dtype, context, cache condition,
and measurement procedure. The reference binary and its results are one-time comparison evidence only
and are not part of the product; neither the production tree nor the benchmarks contain a continuous
runtime route, control kernel, or compatibility switch. Later paged route changes use the currently
admitted paged production route as the reference.

The candidate may adjust tile, split, lookup sharing, and staging according to §17. When the cost of
page translation needs to be attributed, a controlled comparison with matching topology and split
policy may be added, but that comparison does not replace the production gate. Short kernels are
measured with reference/candidate pairs alternating, combining the repeat distribution with absolute
latency; long routes use the median throughput. A profiler is used only when the result cannot be
explained or exceeds the gate; no long-term benchmark research framework is built.

### 19.3 Representative coverage

The performance evidence covers actual routes, not the Cartesian product of all parameters:

- BF16/INT8-G64 causal decode for the registered target at `B=1,2,4,8`, `T=1`, and a representative
  verify small-T;
- one ordinary chunk and one long chunk of causal prefill/cached-only, including a non-aligned prefix
  base;
- standalone append for small-T and for one prefill-sized chunk;
- the identity mapping and one fixed fragmented permutation;
- production CUDA Graph replay, and the real-artifact request paths for ordinary, MTP, and DFlash.

§18 already covers the correctness of page/tile boundaries, so no performance case is built again for
each boundary. Only where an existing production route changes kernel topology at a context seam are
additional measurements taken on both sides of the seam.

### 19.4 Admission gates

For latency metrics, define:

```text
latency_regression = candidate_median / reference_median - 1
```

For throughput metrics, define:

```text
throughput_regression = 1 - candidate_median / reference_median
```

The following gates hold independently for each listed target, dtype, route, and workload case; they
must not be averaged across cases, and a speedup on another route must not offset a regression on the
current route. A threshold is the maximum acceptable regression, not the expected regression; the goal
is still to match or beat the reference. If measurement noise is so large that it cannot be determined
whether a threshold was crossed, the result is invalid, and the measurement conditions must be improved
or the number of valid repeats increased; it must not be treated as a pass.

A storage/addressing route is admitted only if it satisfies all of the following:

1. the median representative end-to-end decode throughput of the registered target regresses by no more
   than 1% relative to the baseline of the same batch profile;
2. the median of causal decode/cached Attention with context of at least 2K regresses by no more
   than 3%;
3. the median prefill Attention throughput regresses by no more than 3%;
4. standalone append shows no repeatable bandwidth regression of more than 5%;
5. CUDA Graph replay produces no additional host synchronization, per-page launch, or
   mapping-dependent recapture;
6. the entire §18 correctness matrix passes.

A single very small kernel at short context is easily affected by timer noise, so one relative
percentage point must not veto or mask a result; it is also constrained by the repeat distribution, the
absolute latency, and the end-to-end gate. Any regression that stably exceeds the above thresholds must
first be localized to lookup, TLB, split, staging, or reduction, and then the corresponding route
adjusted; admission must not be bypassed through a continuous fallback.

### 19.5 Current qualification record

The initial paged storage migration has been admitted on the RTX 5090, CUDA 13.1, `sm_120a`. The
frozen contiguous-KV reference below records only the `B=1` paging migration of that time and is not
current concurrent throughput data:

- the causal matrix covers two registered geometries, BF16/INT8-G64, append/cached, `T=1/4/16`,
  `L=2K/8K`, identity/fragmented mapping, and CUDA Graph cold-cache execution; critical cases were
  repeated with reference/candidate alternation, and no repeatable regression above 3% appeared;
- causal prefill, standalone append, and DFlash full-context passed the 3%, 5%, and 3% gates
  respectively;
- the correctness of identity, contiguous-offset, fragmented, page-boundary, multi-allocation,
  retention/recycle, and graph replay all passed;
- ordinary, MTP, DFlash, and prefix-reuse execution on the two real artifacts all passed.

Representative product-route results are as follows; higher throughput is better and lower round
latency is better:

| Route | Frozen reference | Paged result | Change |
|---|---:|---:|---:|
| 27B ordinary decode | 80.698 tok/s | 84.443 tok/s | +4.64% |
| 35B-A3B ordinary decode | 333.403 tok/s | 341.756 tok/s | +2.51% |
| 27B MTP round | 26.845 ms | 26.065 ms | -2.91% |
| 35B-A3B DFlash steady round | 10.312 ms | 10.005 ms | -2.97% |

---

## 20. Fixed decisions and tuning freedom

The following is the architecture contract that implementations must obey:

- `EngineOptions.max_context=S` is the per-sequence logical ceiling, and `EngineOptions.kv_capacity`
  is `Explicit(K_main)` or `Automatic(R)`; let `L=ceil(S/64)`, `M_min=max(L,max_concurrency)`,
  `M_max=max_concurrency*L`. Explicit takes `M=ceil(K_main/64)`; Automatic directly solves for the
  largest `M` in the interval from the complete target physical reservation curve and the free device
  memory after weight loading minus the headroom `R`; the CLI/server default for `R` is 32 MiB. The
  per-allocation logical capacity of Main is `L` and the physical capacity is `M`
  pages; for MTP the logical capacity is `L` and the physical capacity is
  `M + max_concurrency*ceil((K_draft-1)/64)` pages, where `K_draft` is the speculative draft window.
  Explicit is rejected if `K_main>=S` does not hold; any policy is rejected if the resolved `M` is
  not in `[M_min,M_max]` or if the minimum/runtime reservation cannot be accommodated;
- growing KV uses homogeneous pools, pool-local I32 page-group IDs, and allocation-owned ordered
  mappings;
- the page size of all registered growing pools is `P=64`;
- the K/V and code planes of Main Text/MTP are fixed as contiguous page-major `[D,P,Hkv,Nphysical]`,
  and the INT8-G64 scale planes are fixed as `[D/64,P,Hkv,Nphysical]`;
- the exact strides are uniquely determined by §4.2 for each homogeneous pool; neither the request nor
  the runtime mode selects the order;
- K/V/code/scale and the layers of the same pool share one page-group ID and one per-sequence block
  table;
- the device tables use startup-fixed `max_concurrency` rows; single-sequence Ops consume one row's
  `PagedKVLayerView`, and batched Ops consume the complete `PagedKVBatchLayerView` and independent
  row selectors;
- no gather-to-contiguous, no per-plane/per-head pointer tables, no continuous fallback, no
  arbitrary-layout runtime dispatch;
- DFlash local and boundary-local use the fixed contiguous cyclic layout of §12.1 and do not enter the
  growing pools;
- ownership, reservation, frontier, request identity, and slot identity do not enter the Op view.

The following belong to the route-specific implementation profile and may be measured and adjusted
without changing the storage contract:

- warps per CTA, split count, keys per split, and the 32/64-key route interval of the BF16/INT8 routes;
- how a block-table entry is broadcast in registers/shared memory;
- tokens/heads processed per CTA by the append kernel;
- the allocator's heuristic of preferring contiguous free IDs;
- route-private staging, workspace representation, and reduction decomposition.

Changing `P`, the pool grouping, the page ID model, or the closed pool orders is an architecture
contract change and is not kernel tuning. No implementation may quietly establish a third storage
contract inside a kernel.
