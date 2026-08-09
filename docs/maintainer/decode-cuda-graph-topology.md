# NInfer 单请求 Decode CUDA Graph topology

本文定义当前单请求 Engine 的 Decode CUDA Graph 架构。它覆盖 graph definition、graph executable、
execution profile 和真实 CUDA node topology 的 ownership 与运行关系。

当前 logical batch size 固定为 `B=1`。后续并发可以在这一 ownership model 上增加 batch profile，但不得
重新引入按 context frontier 复制完整 executable 的模型。

---

## 1. Scope and invariants

- 单 GPU、单 resident model instance、单 active request；
- prefill eager 执行，ordinary、MTP 和 DFlash decode 使用既有 round state machine；
- graph-on 与 graph-off 执行相同的 model schedule 和 state transaction；
- CUDA Graph 只在 Engine startup capture 和 instantiate；serving 期间不 capture 或 instantiate；
- 同一 execution profile 内，一个 decode round 只 replay 一次 full-round executable；
- context frontier 只选择 execution profile，不拥有独立 executable；
- 只有 CUDA node 数量、node type 或 dependency structure 不同，才形成新的 topology class；
- 一个 semantic graph family 内，每个 reachable topology class 只拥有一个 executable。

本架构不引入 batch 维度、inactive row、request admission、prefill graph、graph cache eviction 或通用 graph
compiler，也不为了统一 topology 改写现有 Attention/SWA kernel route。

---

## 2. Four distinct concepts

### 2.1 Semantic graph family

Semantic family 是 host schedule 实际执行的一种完整 model/state transaction：

| Engine configuration | Graph families |
|---|---|
| speculative off | ordinary |
| MTP | ordinary, ordinary-aligned, MTP |
| DFlash | ordinary, DFlash-initial, DFlash-steady |

这些 family 的 model calls 或 persistent-state transition 不同，不能仅通过修改 launch 参数互换。

### 2.2 Execution profile

Execution profile 是一个连续 frontier interval，以及该 interval 对应的 target-owned launch promises。它
决定：

- GQA、SWA 和 DFlash Attention 的 execution envelopes；
- kernel specialization、grid、block、split capacity 和 workspace upper bound；
- replay 前需要 materialize 的 Main/backend KV frontier；
- profile 所属的 topology class。

Exact request position、mask、commit count 和有效 context 仍由稳定 device controls 提供。Profile 不改变
数学语义或 request lifecycle。

### 2.3 Captured definition

每个 execution profile 保留一份 `cudaGraph_t` definition。Definition 在 Program 的稳定 IO、workspace、
Linear Attention state 和 KV block-table addresses 上 capture，保存该 profile 的完整 node parameters，并
作为 executable update 的 source；它本身不可 replay。

### 2.4 Topology-class executable

每个 reachable topology class 拥有一份 `cudaGraphExec_t`。同 class 的 definitions 具有相同 node count、
node types 和 dependency structure，因此可以依次安装到同一个 executable：

```text
profile 0 definition ─┐
profile 1 definition ─┼─> topology class A executable
profile 2 definition ─┘

profile 3 definition ─┐
profile 4 definition ─┴─> topology class B executable
```

Kernel function、arguments、grid、block 和 dynamic shared-memory configuration 可以在 CUDA 支持的范围内
通过 whole-graph executable update 改变；它们不单独产生 executable。

---

## 3. Ownership

```text
Program
  │
  ├─ ordinary graph family
  ├─ ordinary-aligned graph family       MTP only
  ├─ MTP graph family                    MTP only
  ├─ DFlash-initial graph family         DFlash only
  └─ DFlash-steady graph family          DFlash only

Graph family
  │
  ├─ ordered profiles
  │    ├─ frontier interval
  │    ├─ topology-class id
  │    └─ captured definition
  │
  └─ topology records
       ├─ topology-class id
       ├─ instantiated executable
       └─ currently installed profile
```

`src/core` 只拥有两种 CUDA RAII resource：

- graph definition：capture、move 和 destruction；
- graph executable：instantiate、whole-graph update、launch 和 destruction。

Core 不理解 frontier、MTP、DFlash 或 target route。

Qwen3.6 family runtime 拥有 graph family、profile lookup、installed-profile tracking、startup qualification 和
runtime replay。Exact target 负责产生有限的 execution profiles 与 topology-class ids；operator 不持有 graph
object。

---

## 4. Current topology inventory

### 4.1 Ordinary

Ordinary target decode 的 token width 固定为一。Frontier 变化只改变 GQA launch parameters，不改变整轮 node
topology：

```text
ordinary family -> one topology class
```

MTP Engine 的 ordinary-aligned round 额外推进一次 MTP companion state/KV，因此是独立 family；它自己的
全部 profiles 同样共享一个 topology class：

```text
ordinary-aligned family -> one topology class
```

### 4.2 MTP

MTP window 在 Engine startup 固定，当前 `K<=5`。Target verification、alignment batch 和各 AR step 的 token
extent 固定；frontier 只改变各 Attention call 的 parameters：

```text
MTP family -> one topology class
```

因此 MTP Engine 一共拥有三个 executables：ordinary、ordinary-aligned 和 MTP 各一个。

### 4.3 DFlash initial

DFlash-initial 只验证 prefill 已经生成的第一批 proposal。Target extent 为 `K+1`：

| Fixed `K` | Target GQA routes | Initial classes |
|---|---|---:|
| `K<=5` | small-T | 1 |
| `6<=K<=15` | prompt, chunked-small-T | 2 |

Prompt 与 chunked-small-T 的 node 数量不同，属于不同 topology classes。

### 4.4 DFlash steady

DFlash-steady 依次执行：

```text
append accepted target context
        -> DFlash proposal
        -> target verification
```

当前影响 node topology 的 route facts 是：

- local SWA：context envelope `<=96` 使用 direct，否则使用 split/reduce；
- target GQA：`K<=5` 固定 small-T；较大 `K` 在 prompt 与 chunked-small-T 间切换；
- split capacity、key-block specialization、grid 和 shared-memory 变化不增加 class。

Reachable classes 为：

| Fixed `K` | Steady classes | Count |
|---|---|---:|
| `K<=5` | `SWA-direct + target-small-T`, `SWA-split + target-small-T` | 2 |
| `6<=K<=15` | `direct + prompt`, `split + prompt`, `split + chunked` | 3 |

所以 executable upper bound 为：

| Engine configuration | Executables |
|---|---:|
| ordinary | 1 |
| MTP | 3 |
| DFlash, `K<=5` | 4 |
| DFlash, `6<=K<=15` | 6 |

`max_context` 未覆盖的 class 不 capture、instantiate 或计入 allowance。

---

## 5. Startup lifecycle

Program construction 按以下顺序建立 graph assets：

```text
plan target profiles
        ↓
warm every reachable operator route
        ↓
capture one definition per profile
        ↓
instantiate one executable per topology class
        ↓
update/replay every definition, short→long and long→short
        ↓
restore clean Program state
```

所有 definitions 使用同一 Program 的稳定地址。每个 class 选择第一份 definition instantiate，随后依次安装
该 class 的全部 definitions 并 replay。双向 cycle 同时确认 long-profile 和 later-short-profile update。

任何 capture、instantiate、update 或 replay 失败都会使 graph-enabled Engine construction 失败；不会建立
per-profile executable fallback，也不会把失败推迟到 serving。

Qualification 完成后，Program 清空 round controls、Linear Attention slots、token statistics 和 speculative
state，解除 temporary KV binding。第一个 request 看不到 graph preparation 使用过的状态。

---

## 6. Runtime dispatch

每轮 runtime dispatch 为：

```text
current semantic state
        -> choose graph family

current frontier E
        -> binary-search execution profile
        -> select profile's topology executable
        -> update executable if installed profile differs
        -> materialize profile KV upper bound
        -> one full-round graph launch
```

同一 profile 的 steady path 只执行 lookup、installed-profile comparison 和 `cudaGraphLaunch`。Profile crossing
时，在任何本轮 model work 之前，用 selected definition 更新 executable；不会 capture、instantiate 或创建新
class。

每个 family 独立记录 installed profile。因此 MTP/DFlash 转入 ordinary tail 时直接选择 ordinary family，
不会把 speculative executable 改造成 ordinary topology。

当前 B=1 round 在上一轮结束时已同步 host-visible token/count；同一 executable 不会被并发 update。未来并发
必须在此基础上重新定义 graph-exec sharing 与 update serialization，不能直接共享当前 mutable installed-profile
状态。

`use_cuda_graph=false` 时不建立 definitions 或 executables，直接执行完全相同的 schedule body。

---

## 7. Resource planning

Graph allowance 按 topology class 计算，而不是按 profile 数量计算。每个 class 只计其最大 execution profile 的
conservative bound；startup 的 `cudaMemGetInfo` check 覆盖 executable、retained definitions 和 qualification
触发的 driver/module allocations。

当前 bounds 为：

- ordinary / ordinary-aligned：每个 class `12 MiB`；
- MTP：每个 class 按最大 visible frontier 取 `12 MiB` 或 `82 MiB`；
- DFlash initial/steady：每个 class 按最大 visible frontier 取 `64 MiB` 或 `96 MiB`。

以 128K context 为例：

| Engine configuration | Planned graph allowance |
|---|---:|
| ordinary | 12 MiB |
| MTP, `K=5` | 106 MiB |
| DFlash, `K<=5` | 268 MiB |
| DFlash, `6<=K<=15` | 396 MiB |

Allowance 在 SequencePlan 中进入 device reservation，不能通过静默缩小 advertised context capacity 为 graph
让路。Program construction 后的实际 graph consumption 若超过 allowance，Engine construction 失败。

---

## 8. Boundary behavior

### New short request after a long request

Executable 可能仍安装 long profile。新 request 的第一轮选择 short profile并执行 reverse update，不重建
Program。

### Prefix reuse at a nonzero frontier

第一轮直接选择命中 frontier 的 profile，不从 profile zero 逐级推进。

### Exact profile boundary

Profile intervals contiguous、inclusive、无 gap 或 overlap ambiguity。一个 frontier 只属于一个 profile；KV
materialization 与 launch envelope 来自同一 profile。

### Ordinary/speculative switching

MTP/DFlash fixed window 因 output budget 或 context tail 不能继续时，由 request state machine 选择对应 ordinary
family。Graph layer只执行已选 family，不重新解释 speculative eligibility。

### Small maximum context

Profiles 截断到合法 maximum frontier。没有 reachable profile 的 topology class 不创建。

---

## 9. Validation contract

Graph topology 或 ownership 变化至少使用以下直接证据：

- startup 对每个 declared class 完成 definition update/replay cycle；
- ordinary、MTP、DFlash 使用真实 registered artifact 通过 public Engine route；
- MTP 覆盖 speculative 与 ordinary-aligned tail；
- DFlash 覆盖 initial、steady 与 ordinary tail；
- 代表性的 128K MTP/DFlash Engine 能在 planned allowance 内完成 construction；
- graph-on 与 graph-off 保持相同数学 schedule；
- steady profile throughput 不因 graph ownership 改变而显著回退。

性能比较必须在 GPU 占用条件相同的环境中进行。Profile transition 的一次性 update cost与 steady replay 分开
解释，不把环境竞争导致的吞吐变化归因于 graph architecture。
