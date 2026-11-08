# Softmax Attention Organization and Migration Specification

This document defines the target semantics, public contracts, source organization, and migration
of existing implementations for NInfer Softmax Attention. It refines
[`op-development.md`](op-development.md) for the Attention domain. It describes the only target
state after the migration is complete; until the migration is complete, the `gqa_attention`,
`vision_attention`, `bidirectional_gqa_attention`, and `swa` symbols still visible in the source
are the current state to be replaced, not a second interface that must be kept compatible.

## 1. Design Conclusions

Softmax Attention and the existing Linear Attention are two parallel algorithm categories:

```text
src/ops/
├── softmax_attention/
└── linear_attention/
```

Within Softmax Attention, families are divided by complete mathematical transform and explicit
state effect, not by model, modality, Head topology, or CUDA algorithm:

```text
src/ops/softmax_attention/
├── dense/
└── sliding_window/
```

The current migration follows these conclusions:

1. MHA, MQA, and GQA are not three Ops and not three source directories. They are values of the
   Head geometry of a unified Q/K/V Attention.
2. Text causal cache Attention, packed Vision Attention, and DFlash context Attention all belong
   to the `dense` family, but their visible domains, physical inputs, and state effects differ, so
   each keeps an independent entry with a clear effect.
3. `vision` is a caller identity, not Attention semantics. Public symbols, file names, tests, and
   benchmarks no longer use `vision_attention`.
4. `gqa` describes only the mapping from Query Head to KV Head. Public symbols, file names, tests,
   and benchmarks no longer use `gqa_attention`.
5. The current SWA uses a cyclic KV cache, absolute positions, and a symmetric local visible
   domain; it is an independent `sliding_window` family, and its public full name is
   `sliding_window_attention`.
6. FlashAttention is a private implementation strategy of exact Softmax Attention, and
   PagedAttention is a KV storage and addressing strategy. Neither is a public semantic family.
7. MLA, structured sparse Attention, two-dimensional window Attention, and Deformable Attention
   create a peer family only when a registered target actually needs them; no empty directories,
   placeholder interfaces, or generic registration framework are created now.

## 2. Classification Axes

Attention-related terms must first be assigned to the following four mutually independent axes;
names from different axes must not be listed side by side as directories:

| Axis | Question it answers | Current examples | Effect on source organization |
|---|---|---|---|
| Mathematical family | What are the output formula, visible domain, and state transition | dense, sliding window | Determines the family under `softmax_attention/` |
| Head geometry | Which KV Head each Query Head uses | MHA, MQA, GQA | Ordinary values and validation rules; no family is created |
| Input/state representation | Whether K/V are ordinary Tensors, packed segments, a linear cache, a cyclic cache, or a latent cache | packed, causal cache, context+query, future MLA | Forms an entry with a clear effect, or a new family, within a family |
| Implementation strategy | How to tile, move, reduce, and address | flash, split-KV, paged, small-T | Exists only in private launchers, plans, and kernels |

The following classifications are mandatory:

- Self-attention and cross-attention use the same dense formula once Q/K/V are explicitly given;
  they do not become two Ops because Q/K/V come from different sources.
- Causal, non-causal, packed block-diagonal, context+query, and sliding window describe the visible
  domain or state effect; they must be stated explicitly in the entry contract and cannot be hidden
  in a caller convention.
- `prefill`, `decode`, `small_t`, `flash`, and `split_kv` are implementation or measurement terms
  and cannot become public semantic entries.
- Text, MTP, Vision, and DFlash are call sites or model scheduling concepts and must not appear in
  generic Attention Op names.

## 3. Unified Mathematical Semantics and Head Geometry

### 3.1 Head geometry

Explicit Q/K/V Softmax Attention expresses Head geometry with one narrow host value:

```cpp
struct AttentionHeadGeometry {
    std::int32_t head_dim;
    std::int32_t query_heads;
    std::int32_t kv_heads;
};
```

The current public representation keeps the Head dimension of Q, K, and V the same. A legal
geometry satisfies:

```text
D = head_dim > 0
Hq = query_heads > 0
Hkv = kv_heads > 0
Hq % Hkv == 0
group = Hq / Hkv
kv_head(h) = floor(h / group)
```

The logical Tensor shapes are:

```text
Q   [D, Hq,  Tq]
K   [D, Hkv, Tk]
V   [D, Hkv, Tk]
Out [D, Hq,  Tq]
```

The geometry gives semantics to the axes of the generic `Tensor` and serves workspace capacity
queries; an execution wrapper must verify that the geometry is consistent with the Q/K/V/Out/cache
views. It does not contain a target key, model role, execution phase, kernel selection, or device
resources.

MHA, MQA, and GQA are only the following three value regions:

```text
MHA: Hq == Hkv
MQA: Hkv == 1
GQA: 1 < Hkv < Hq
```

Therefore no `GqaGeometry`, `GqaExecutionEnvelope`, `gqa_*` launcher, or `gqa_*` kernel is
defined. If a real future target needs a non-uniform or non-contiguous Head mapping, a new explicit
mapping contract must be defined for that mathematical semantics; an arbitrary mapping Tensor must
not first be added to the current unified geometry.

### 3.2 Softmax formula

For Query token `i`, Query Head `h`, and the visible Key set `A(i)` defined by the entry:

```text
kh = floor(h / (Hq / Hkv))
score(i,h,j) = scale * dot(Q[:,h,i], K[:,kh,j]), j ∈ A(i)
p(i,h,:)     = stable_softmax(score(i,h,:))
ideal[:,h,i] = Σ(j ∈ A(i)) p(i,h,j) * V[:,kh,j]
```

The logical values after the public input storage boundary enter the same naive FP64 oracle. An
implementation may use online Softmax, Flash tiling, split-KV, private low-precision staging, or a
different reduction tree, but must not change the visible set, cache encoding, public output, or
state effect.

`scale` is an explicit semantic parameter. The currently registered domain may accept only
`1/sqrt(D)` for the corresponding geometry, but scale cannot be hidden in Vision, Text, or a
kernel name.

### 3.3 Currently registered geometry

The migration does not take the opportunity to widen the supported surface. Each entry accepts only
the finite domain that the current production path already supports:

| Use site | geometry | cache/layout | Entry after migration |
|---|---:|---|---|
| Qwen3.6-27B Text/MTP | `D256/Hq24/Hkv4` | BF16 or INT8-G64 linear cache | `causal_softmax_attention` |
| Qwen3.6-35B-A3B Text/MTP | `D256/Hq16/Hkv2` | BF16 or INT8-G64 linear cache | `causal_softmax_attention` |
| Qwen3.6 Vision | `D72/Hq16/Hkv16` | packed BF16 Q/K/V | `packed_softmax_attention` |
| Qwen3.6-35B-A3B DFlash full | `D128/Hq32/Hkv8` | read-only BF16 context + query K/V | `context_softmax_attention` |
| Qwen3.6-35B-A3B DFlash local | `D128/Hq32/Hkv8` | read-only BF16 cyclic context + query K/V | `sliding_window_attention` |

## 4. Public Contracts

The target state uses one shared value header and three public contract headers:

```text
include/ninfer/ops/attention_geometry.h
include/ninfer/ops/softmax_attention.h
include/ninfer/ops/sliding_window_attention.h
include/ninfer/ops/kv_cache_append.h
```

`attention_geometry.h` owns only `AttentionHeadGeometry` and its host-side legality rules, so that
the dense and sliding-window contracts need not include each other. `softmax_attention.h` owns the
formula, the entry-specific execution envelopes, and the workspace capacity queries of the dense
family. `sliding_window_attention.h` owns the independent visible domain and cyclic cache contract
of the local window. `kv_cache_append.h` owns the KV cache state transition that does not compute
Attention.

### 4.1 Ordinary dense Q/K/V

The ordinary single-segment Q/K/V entry takes the desired direct form:

```cpp
void softmax_attention(const Tensor& q, const Tensor& k, const Tensor& v,
                       AttentionHeadGeometry geometry, float scale,
                       WorkspaceArena& workspace, Tensor& out, cudaStream_t stream);
```

The formula and interface do not force `Tq` and `Tk` to be the same semantic axis, so a future dense
cross-attention does not need a new family. The current migration registers only the existing
`D72/Hq16/Hkv16`, `Tq=Tk` single-segment self-attention domain; it must not claim support for
arbitrary shapes or cross-attention merely because the formula is general.

The current migration may let the single-segment route of the packed implementation serve this
entry, without adding a second kernel.

### 4.2 Packed dense Q/K/V

The current Vision Attention migrates to a modality-independent packed entry:

```cpp
void packed_softmax_attention(const Tensor& q, const Tensor& k, const Tensor& v,
                              AttentionHeadGeometry geometry, float scale,
                              const Tensor& cu_seqlens, WorkspaceArena& workspace,
                              Tensor& out, cudaStream_t stream);

void packed_softmax_attention(const Tensor& q, const Tensor& k, const Tensor& v,
                              AttentionHeadGeometry geometry, float scale,
                              std::int32_t segment_length, Tensor& out,
                              cudaStream_t stream);
```

Each segment is an independent dense non-causal self-attention; different segments are not visible
to each other. The first entry accepts a strictly increasing device `cu_seqlens`; the second entry
represents equal-length contiguous segments. Its workspace query is renamed to:

```cpp
packed_softmax_attention_workspace_capacity_bytes(...)
```

After the migration, the contract uses `T` or `tokens` for the generic packed token count and does
not use `patches` as an Op-level axis name. Vision scheduling may still call that value `patches` in
its own code.

### 4.3 Causal cached dense Attention

The Text/MTP linear cache Attention keeps three behaviors with clear effects:

```cpp
struct CausalAttentionExecutionEnvelope {
    std::uint32_t min_visible_keys;
    std::uint32_t max_visible_keys;
};

void causal_softmax_attention(
    const Tensor& q, const Tensor& k, const Tensor& v, const Tensor& positions,
    AttentionHeadGeometry geometry, float scale, KVCacheLayerView cache,
    CausalAttentionExecutionEnvelope envelope, WorkspaceArena& workspace,
    Tensor& out, cudaStream_t stream);

void causal_softmax_attention_cached(
    const Tensor& q, const Tensor& positions, AttentionHeadGeometry geometry,
    float scale, const KVCacheLayerView& cache,
    CausalAttentionExecutionEnvelope envelope, WorkspaceArena& workspace,
    Tensor& out, cudaStream_t stream);
```

`causal_softmax_attention` first writes the current K/V into the cache rows given by the absolute
`positions`, and then lets the Query at position `p` see the already-filled `[0,p]`.
`causal_softmax_attention_cached` accepts no new K/V and does not modify the cache; it only reads
the same visible domain. The two share one mathematical oracle and one cache codec definition, but
must be qualified directly and separately.

The capacity query is:

```cpp
causal_softmax_attention_workspace_capacity_bytes(
    AttentionHeadGeometry geometry, DType cache_dtype,
    CausalAttentionExecutionEnvelope envelope,
    std::int32_t min_tokens, std::int32_t max_tokens);
```

`positions` determines the exact causal visible domain; the execution envelope only guarantees
graph-safe launch capacity and is not a mask parameter.

### 4.4 Context + query dense Attention

The true semantics of the current `bidirectional_gqa_attention` is: the Query sees both the
read-only persistent context and the complete query K/V segment. It migrates to:

```cpp
struct ContextAttentionExecutionEnvelope {
    std::uint32_t min_context;
    std::uint32_t max_context;
};

void context_softmax_attention(
    const Tensor& q, const Tensor& query_k, const Tensor& query_v,
    const Tensor& context_length, AttentionHeadGeometry geometry, float scale,
    const KVCacheLayerView& context, ContextAttentionExecutionEnvelope envelope,
    WorkspaceArena& workspace, Tensor& out, cudaStream_t stream);
```

All Query rows see `[0, context_length)` and the complete query segment; there is no causal
triangle. `context` and the query K/V remain two physical segments, with no requirement to
concatenate or copy them. The capacity query is
`context_softmax_attention_workspace_capacity_bytes(...)`.

`bidirectional` no longer appears in the name, because it describes only the visible domain of the
current entry; `GQA` no longer appears in the name, because `D128/Hq32/Hkv8` is already expressed by
the geometry.

### 4.5 Sliding-window Attention

SWA migrates to an independent header and family:

```cpp
struct SlidingWindowAttentionExecutionEnvelope {
    std::uint32_t min_context;
    std::uint32_t max_context;
};

void sliding_window_attention(
    const Tensor& q, const Tensor& query_k, const Tensor& query_v,
    const Tensor& positions, AttentionHeadGeometry geometry,
    std::uint32_t window, float scale,
    const CyclicKVCacheLayerView& context,
    SlidingWindowAttentionExecutionEnvelope envelope,
    WorkspaceArena& workspace, Tensor& out, cudaStream_t stream);
```

The only currently registered `window` is 4096. The explicit parameter makes
`abs(key_position-query_position) < window` part of the complete Op semantics; the wrapper still
accepts only values that have been implemented and qualified. The cyclic physical slot
`absolute_position % window` is the state layout contract of this entry.

The public symbol spells out `sliding_window_attention` in full. `SWA` may be used in paper
terminology, local variables, or profiler notes, but is no longer the primary name of a contract,
file, test target, or benchmark target.

### 4.6 KV cache append moves out of Attention

The current `gqa_kv_append` only writes the cache and optionally encodes INT8-G64; it does not
compute Attention. It is not a dense Attention entry, and migrates to `kv_cache_append`:

```cpp
void kv_cache_append(const Tensor& k, const Tensor& v,
                     const Tensor& positions, KVCacheLayerView cache,
                     cudaStream_t stream);
```

Together with the existing `kv_cache_append_prefix`, it forms one cache state-transition overload
group, and both migrate together to:

```text
include/ninfer/ops/kv_cache_append.h
src/ops/kv_cache/append/
```

`kv_cache_append` covers all input rows and performs a BF16 exact copy or INT8-G64 encode according
to the target cache dtype; `kv_cache_append_prefix` continues to accept a device `commit_count` and
writes only the admitted prefix. Neither belongs under `softmax_attention/` merely because
Attention calls it.

The fused append-and-attend entry of `causal_softmax_attention` must refer to the same cache codec
semantics and produce the same code/scale bits; it may own a fused implementation for performance
and need not go through the public `kv_cache_append`, which would add an extra launch.

### 4.7 Unified interfaces explicitly not introduced

The following designs are not introduced:

- `AttentionOptions`, a runtime `AttentionKind`, a string registry, or an Op base class;
- a catch-all entry that accepts an arbitrary dense mask Tensor;
- selecting semantics with booleans such as `is_vision`, `is_decode`, `is_gqa`, or `use_flash`;
- separate overloads for MHA, MQA, and GQA;
- putting linear, cyclic, paged, and latent caches into one `std::variant`;
- having a wrapper choose behavior from a target key, layer role, or Program phase.

When a new target needs a new complete visible domain or state representation, first write down the
closed formula and state effect, and then decide whether to extend an existing entry or create a
peer family.
## 5. Target source organization

After the migration, the directory layout is as follows. File names express a semantic entry or a real
private implementation route, not the calling model:

```text
src/ops/
├── softmax_attention/
│   ├── common/
│   │   ├── head_mapping.cuh
│   │   └── context_query.cuh
│   ├── dense/
│   │   ├── causal_cache/
│   │   │   ├── causal_softmax_attention.cpp
│   │   │   ├── launch.h
│   │   │   ├── small_t.cu
│   │   │   ├── small_t.cuh
│   │   │   ├── small_t_bf16.cuh
│   │   │   ├── small_t_i8.cuh
│   │   │   ├── prompt.cu
│   │   │   ├── prompt_common.cuh
│   │   │   ├── prompt_bf16.cuh
│   │   │   └── prompt_i8.cuh
│   │   ├── packed/
│   │   │   ├── packed_softmax_attention.cpp
│   │   │   ├── launch.h
│   │   │   ├── launch.cu
│   │   │   └── kernel.cuh
│   │   └── context/
│   │       ├── context_softmax_attention.cpp
│   │       ├── launch.h
│   │       ├── launch.cu
│   │       └── kernel.cuh
│   └── sliding_window/
│       ├── sliding_window_attention.cpp
│       ├── launch.h
│       ├── launch.cu
│       └── kernel.cuh
└── kv_cache/
    ├── codec.cuh
    └── append/
        ├── kv_cache_append.cpp
        ├── launch.h
        ├── launch.cu
        └── kernel.cuh
```

This is a structure of responsibility and ownership; it does not require creating empty files just to
make the tree line up:

- The entry `.cpp` owns contract validation, workspace scope, and finite dispatch.
- `launch.h` and `launch.cu` own the private plan, grid, block, shared memory, and launch errors.
- `.cuh` files own kernels and entry-local device computation.
- `softmax_attention/common/` accepts only narrow device primitives that at least two families
  actually share already.

Currently `bidirectional_gqa_attention.cuh` hosts the device body shared by linear-context and
cyclic-SWA. During migration this part should become `common/context_query.cuh`, which carries no
GQA/SWA identity and is called by the entry kernels of `dense/context` and `sliding_window`. The
`sliding_window/` code must not include the private kernel of `dense/context`, and a generic
Attention backend framework must not be created.

`head_mapping.cuh` implements only the zero-cost indexing derived from the facts validated in
`AttentionHeadGeometry`; it does no runtime registry or target selection.

The exact encode/decode device primitive of the INT8-G64 cache belongs to `kv_cache/codec.cuh`. Both
the standalone append and the fused causal append-and-attend implementation may reuse it; the codec
must not be duplicated in two families, and `kv_cache/append/` must not in turn include private
Attention files.

## 6. Migration of existing contracts and symbols

### 6.1 Public headers and symbols

| Existing item | Target item | Handling |
|---|---|---|
| (new shared type) | `AttentionHeadGeometry` in `include/ninfer/ops/attention_geometry.h` | Expresses only Head numeric facts; no family or registry is created |
| `include/ninfer/ops/gqa_attention.h` | `include/ninfer/ops/softmax_attention.h` | Delete the old header; keep no forwarding include |
| `GqaExecutionEnvelope` | `CausalAttentionExecutionEnvelope` | Rename and remove the GQA identity |
| `gqa_attention_workspace_capacity_bytes` | `causal_softmax_attention_workspace_capacity_bytes` | Add an explicit geometry |
| `gqa_attention` | `causal_softmax_attention` | Keep the append-and-attend state effect |
| `gqa_attention_cached` | `causal_softmax_attention_cached` | Keep the read-only cache effect |
| `gqa_kv_append` | `kv_cache_append` | Move out of the Attention family |
| `include/ninfer/ops/vision_attention.h` | `include/ninfer/ops/softmax_attention.h` | Remove the Vision identity |
| `vision_attention_workspace_capacity_bytes` | `packed_softmax_attention_workspace_capacity_bytes` | Axis names become the generic tokens/segments |
| `vision_attention(..., cu_seqlens, ...)` | `packed_softmax_attention(..., cu_seqlens, ...)` | Add the geometry and an explicit scale |
| `vision_attention(..., segment_length, ...)` | `packed_softmax_attention(..., segment_length, ...)` | Keep the equal-length segment overload |
| `include/ninfer/ops/bidirectional_gqa_attention.h` | `include/ninfer/ops/softmax_attention.h` | Delete the old header |
| `GqaContextExecutionEnvelope` | `ContextAttentionExecutionEnvelope` | Remove the GQA identity |
| `bidirectional_gqa_attention` | `context_softmax_attention` | The name expresses the context+query physical semantics |
| `bidirectional_gqa_attention_workspace_capacity_bytes` | `context_softmax_attention_workspace_capacity_bytes` | Same name as the entry |
| `include/ninfer/ops/swa.h` | `include/ninfer/ops/sliding_window_attention.h` | Delete the abbreviated header |
| `SwaContextExecutionEnvelope` | `SlidingWindowAttentionExecutionEnvelope` | Name with the full semantics |
| `swa` | `sliding_window_attention` | Keep the current window formula |
| `swa_workspace_capacity_bytes` | `sliding_window_attention_workspace_capacity_bytes` | Same name as the entry |
| `include/ninfer/ops/kv_cache_append_prefix.h` | `include/ninfer/ops/kv_cache_append.h` | Forms an overload group with the full append |

Project-owned interfaces do not preserve compatibility. In the end there must be no old function
aliases, old type aliases, forwarding headers, doubly registered CMake sources, or simultaneously
maintained old and new tests.

### 6.2 Source files

| Existing file group | Target directory |
|---|---|
| `src/ops/wrapper/gqa_attention.cpp` | `softmax_attention/dense/causal_cache/causal_softmax_attention.cpp`; the standalone append in it moves to `kv_cache/append/` |
| `src/ops/launcher/gqa_attention*` | `softmax_attention/dense/causal_cache/` |
| `src/ops/kernel/gqa_attention*` | The Attention kernels move to `softmax_attention/dense/causal_cache/`; the cache codec moves to `kv_cache/codec.cuh`; all symbols drop GQA |
| `src/ops/wrapper/vision_attention.cpp` | `softmax_attention/dense/packed/packed_softmax_attention.cpp` |
| `src/ops/launcher/vision_attention.*` | `softmax_attention/dense/packed/launch.*` |
| `src/ops/kernel/vision_attention.cuh` | `softmax_attention/dense/packed/kernel.cuh` |
| `src/ops/wrapper/bidirectional_gqa_attention.cpp` | `softmax_attention/dense/context/context_softmax_attention.cpp` |
| `src/ops/launcher/bidirectional_gqa_attention.*` | `softmax_attention/dense/context/launch.*` |
| `src/ops/wrapper/swa.cpp` | `softmax_attention/sliding_window/sliding_window_attention.cpp` |
| `src/ops/launcher/swa.*` | `softmax_attention/sliding_window/launch.*` |
| `src/ops/kernel/bidirectional_gqa_attention.cuh` | Split into `softmax_attention/common/context_query.cuh` and the context/sliding-window entry-local kernels |
| `src/ops/wrapper/kv_cache_append_prefix.cpp` and the corresponding launcher/kernel | `kv_cache/append/`, which also receives the standalone append separated from the causal cache |

Private C++/CUDA types migrate in step, according to their actual responsibilities:

| Existing private name | Target naming rule |
|---|---|
| `GqaAttentionRoute` | `CausalAttentionRoute` |
| `GqaAppendInput` / `GqaCachedInput` | `CausalAppendInput` / `CausalCachedInput` |
| `Gqa27Geometry` / `Gqa35Geometry` | `CausalD256H24Kv4` / `CausalD256H16Kv2`, named by their numeric facts |
| `BidirectionalGqaPlan` / `BidirectionalGqaRoute` | `ContextAttentionPlan` / `ContextAttentionRoute` |
| `SwaPlan` / `SwaRoute` | `SlidingWindowAttentionPlan` / `SlidingWindowAttentionRoute` |
| `VisionAttentionTile` | `PackedAttentionTile` |
| `kGqa*` / `kBidirectionalGqa*` / `kVisionAttention*` | Preferably derived from the validated geometry; where they must be fixed at compile time, named by entry, format, or tile facts |

The numeric-geometry-specific types are private compile-time dispatch material, not a new public Head
classification, and they do not enter target includes. Public calls always pass `AttentionHeadGeometry`.

`small_t`, `prompt`, `split_kv`, `bf16`, and `i8` may continue as private route/format names.
`decode` and `prefill`, if they only denote the implementation route for a token extent, should
preferably be changed during migration to `small_t`, `prompt`, or a concrete algorithm name, to avoid
fixing the Program phase into the implementation identity.

### 6.3 Callers

| Caller | Migration |
|---|---|
| `src/targets/qwen3_6/impl/runtime/text_context_impl.h` | Include `softmax_attention.h`; Text/MTP call `causal_softmax_attention` or `_cached`; standalone cache writes call `kv_cache_append` |
| `src/targets/qwen3_6/impl/runtime/layouts_impl.h` | Capacity queries use the geometry and the new entry names |
| `src/targets/qwen3_6/impl/runtime/vision_context_impl.h` | Include `softmax_attention.h`; call `packed_softmax_attention` without passing a Vision identity to the Op |
| `src/targets/qwen3_6/impl/runtime/dflash_impl.h` | The full route calls `context_softmax_attention`; the local route calls `sliding_window_attention` |
| `src/targets/qwen3_6/impl/runtime/workspace_recipe.h` | Target-local region names may keep their model meaning; Op capacity and shape use the unified geometry |

The family schedule still decides when to call the full, local, Vision, or MTP path; the Op wrapper
does not read the layer role or the target Variant.

### 6.4 Build ownership

`src/CMakeLists.txt` continues to list every `.cpp`/`.cu` explicitly. The migration must, in a single
target-state cutover:

1. add the new source paths;
2. update the single build owner;
3. delete the old horizontal source paths;
4. confirm that no kernel or launcher is compiled twice through both the old and new paths.

Do not switch to a recursive glob, and do not create a temporary second Attention library.

## 7. Test and benchmark migration

### 7.1 Qualification tests

The dense family shares one independent FP64 oracle and a Head mapping helper, but each complete
entry directly verifies its own visible domain, state effect, and registered implementation routes.
The suggested organization is:

```text
tests/ops/softmax_attention/
├── oracle.h
├── main.cpp
├── causal_cache.cpp
├── plain_and_packed.cpp
└── context.cpp

tests/ops/test_sliding_window_attention.cpp
tests/ops/test_kv_cache_append.cpp
```

The corresponding test targets:

```text
ninfer_softmax_attention_test
ninfer_sliding_window_attention_test
ninfer_kv_cache_append_test
```

Existing tests migrate as follows:

| Existing test | Target |
|---|---|
| `test_gqa_attention.cpp` | `softmax_attention/causal_cache.cpp`; the codec-only part moves to `test_kv_cache_append.cpp` |
| `test_vision_attention.cpp` | `softmax_attention/plain_and_packed.cpp` |
| `test_bidirectional_gqa_attention.cpp` | `softmax_attention/context.cpp` |
| `test_swa.cpp` | `test_sliding_window_attention.cpp` |
| `test_kv_cache_append_prefix.cpp` | Merged into the prefix cases of `test_kv_cache_append.cpp` |

The oracle must compute the Head mapping from the geometry and the entry's visible set, and must not
copy the group constants of the production kernels. It must cover:

- the four currently registered geometries;
- the BF16 and INT8-G64 causal cache profiles;
- append-and-attend, cached-only, and standalone append;
- the plain single-segment entry, and packed unequal-length and equal-length segments;
- the context+query visible domain with zero and nonzero context;
- the window boundary, where `distance=window-1` is included and `distance=window` is excluded;
- valid replay of the CUDA Graph execution envelope;
- cache code/scale bits, read-only inputs, and all declared state writes.

Different implementation profiles may have different named tolerances, but all are directly
accountable to the same ideal Attention oracle.

### 7.2 Benchmarks

The long-lived benchmarks are divided by public semantics:

| benchmark | Public entry in a single timed body |
|---|---|
| `ninfer_causal_softmax_attention_bench` | append-and-attend or cached-only |
| `ninfer_packed_softmax_attention_bench` | uniform plain segment or packed segments |
| `ninfer_context_softmax_attention_bench` | context-plus-query |
| `ninfer_sliding_window_attention_bench` | symmetric sliding window |
| `ninfer_kv_cache_append_bench` | full append or device-count prefix append |

Fixture construction, the public capacity query, cache conditioning, CUDA Graph capture/instantiate,
and synchronization are all outside the timed interval. The eager timed body calls the public Op
exactly once; inside the Graph, only that same public call is captured. A benchmark contains no
private header, launcher, candidate, route forcing, tile/split parameter, duplicated production
dispatch, or kernel-name regex. `--profile` encloses the complete public call, and every production
kernel that appears inside the call belongs to the measured result of that Op.

`attention_layer_bench` is not a closed public Op contract, and the original implementation composed
several internal stages and exposed private routes, so it is removed from the long-lived Op
benchmarks. The complete mixer and the target-level impact are measured by the public Engine or the
target round benchmark; private candidate comparisons may exist only in task-time temporary code and
are deleted once the route is chosen.

## 8. One-shot migration order

Implementation completes one consistent cutover in the following dependency order; this does not
require keeping releasable intermediate compatibility states:

1. In `attention_geometry.h` and `softmax_attention.h`, write the unified geometry, the public
   formulas, the dense entries, the envelopes, and the capacity contract.
2. Converge the standalone `gqa_kv_append` and the existing prefix append into `kv_cache_append.h` and
   `src/ops/kv_cache/append/`, and fix a single cache codec semantics.
3. Move the causal cache, packed, and context implementations into `softmax_attention/dense/`, first
   keeping numerics and route selection unchanged, then eliminating the GQA/Vision/bidirectional
   naming.
4. Extract the shared context+query device body into a neutral common primitive, move SWA into
   `sliding_window/`, and rename it completely.
5. Update the target callers, workspace capacity, explicit CMake sources, tests, and benchmarks.
6. Update the affected model references, the benchmark README, and the test commands.
7. Delete all old headers, sources, symbols, targets, and forwarding paths.
8. Run the focused build, the three Attention qualification sets, the KV append qualification, the
   relevant target execution, and the final public benchmarks.

If, while moving sources, an existing route is found to depend on an old include path, the private
ownership should be corrected; a forwarding header or a copied kernel must not serve as a transitional
end state.

## 9. Admission location for future Attention

Future names are first classified by their real semantics:

| Common name | Classification in this specification |
|---|---|
| MHA / MQA / GQA | `AttentionHeadGeometry`; no directory is created |
| self / cross attention | The Q/K/V sources and `Tq/Tk` values of ordinary `softmax_attention` |
| causal / bidirectional / prefix mask | An explicit visible-domain entry of the dense family; arbitrary masks are not added in advance |
| packed / variable length | An input-representation entry of the dense family |
| local / sliding window | `sliding_window`, or a separate window family once real two-dimensional semantics appear |
| global+local / block sparse / dilated sparse | A `structured_sparse/` peer family is created once actually registered |
| MLA | A `multi_head_latent/` peer family is created once actually registered; the latent cache and absorbed projections cannot be expanded into ordinary K/V and passed off as reuse |
| deformable attention | A separate family is created once actually registered, explicitly owning the reference point, sampling, and aggregation semantics |
| FlashAttention | A private kernel/route inside the dense/local family |
| PagedAttention | Cache storage/view and a private addressing implementation, not a mathematical family |
| Ring Attention | A multi-device execution strategy; not admitted by the current single-GPU product |

A new family must be driven by a registered target and a closed contract. Creating empty directories,
public enums, a runtime registry, a generic mask IR, or a backend interface merely for "extensibility"
is not extensibility but a framework the product does not need.

### 9.1 Terminology references

This specification uses the following primary works to distinguish mathematical semantics, Head
geometry, and implementation strategy:

- [Attention Is All You Need](https://arxiv.org/abs/1706.03762): scaled dot-product, multi-head,
  self, and encoder-decoder attention;
- [Fast Transformer Decoding: One Write-Head is All You Need](https://arxiv.org/abs/1911.02150)
  and [GQA](https://arxiv.org/abs/2305.13245): the KV Head sharing relation of MQA/GQA;
- [Longformer](https://arxiv.org/abs/2004.05150), [BigBird](https://arxiv.org/abs/2007.14062),
  and [Swin Transformer](https://arxiv.org/abs/2103.14030): local, global+local, structured sparse,
  and two-dimensional window visible domains;
- [FlashAttention](https://arxiv.org/abs/2205.14135) and
  [PagedAttention](https://arxiv.org/abs/2309.06180): IO-aware exact Attention and KV memory
  management;
- [DeepSeek-V2](https://arxiv.org/abs/2405.04434): the latent KV cache and absorbed projections of
  MLA;
- [Deformable DETR](https://arxiv.org/abs/2010.04159): reference-point sampling Attention.

## 10. Completion criteria

The migration is complete only when all of the following conditions hold:

- The target source tree, the shared geometry header, the three public contract headers, and the
  single CMake ownership are established;
- Production targets include only the new semantic headers;
- MHA/MQA/GQA exist only as geometry terminology;
- The Vision identity no longer appears in Attention Op contracts, source families, tests, or
  benchmark names;
- The public SWA name is completely migrated to `sliding_window_attention`;
- The standalone cache append no longer belongs to Attention;
- Old headers, old sources, old symbols, old test targets, old benchmark targets, and compatibility
  aliases are deleted;
- The independent qualification of dense, sliding-window, and KV append passes;
- The Text, Vision, MTP, DFlash, prefix reuse, and CUDA Graph paths of the two registered targets
  remain supported;
- Affected documents and commands use the new names, and no stale references remain apart from the
  migration mapping in this document;
- The relevant public benchmarks show that the migration caused no performance regression on the
  target paths, or give direct evidence for any intentional performance change.
