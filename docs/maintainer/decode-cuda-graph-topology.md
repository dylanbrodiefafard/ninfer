# NInfer Decode CUDA Graph topology

本文定义当前 Engine 的 Decode CUDA Graph ownership、execution profile 和 executable 复用模型。
ordinary 与 MTP 已采用小规模 compact batch；DFlash 仍使用当前单请求 schedule。并发 request lifecycle、
MTP round transaction 与算子 batch contract 分别由
[小规模并发推理架构](concurrent-inference-architecture.md)、
[并发 MTP execution](concurrent-mtp-implementation.md)和
[并发 Decode 算子需求](concurrent-decode-operators.md)定义。

---

## 1. Scope and invariants

- 单 GPU、单 resident model instance；
- ordinary 与 MTP logical batch size 为本轮 compact membership 的精确 `B`，`1<=B<=C`；
- DFlash graph 当前固定 `B=1`；
- prefill eager 执行，DecodeRound 才进入 graph；
- graph-on 与 graph-off 执行相同的 model schedule 和 state transaction；
- CUDA Graph 只在 Engine startup capture 和 instantiate，serving 期间不创建 graph；
- 同一 execution profile 内，一个 DecodeRound 只 replay 一次 full-round executable；
- context frontier 只选择 execution profile，不拥有独立 executable；
- CUDA node 数量、node type 或 dependency structure 不同才形成新的 topology class；
- 一个 semantic graph family 内，每个 reachable topology class 只拥有一个 executable。

`B` 是 graph topology 的结构维度，不是 replay metadata。Request identity、row-to-lane mapping、exact
frontier、valid extent、sampling state 和 KV/state selectors 都是稳定 frame 中的 replay data。

---

## 2. Graph identity

### 2.1 Semantic graph family

Semantic family 是 host schedule 执行的一种完整 model/state transaction：

| Engine configuration | Graph families |
|---|---|
| speculative off | ordinary exact-`B` |
| MTP | MTP exact-`B` |
| DFlash | ordinary `B=1`, DFlash-initial `B=1`, DFlash-steady `B=1` |

MTP 没有额外的 ordinary 或 ordinary-aligned family。Current proposal extent 为零的 row 仍在统一 MTP
round 中完成一个 target step；zero、partial 和 full proposal extent 不改变 graph identity。

DFlash initial、steady 和 ordinary 的 model calls 与 persistent-state transition 不同，因此仍是独立
semantic families。

### 2.2 Execution profile

Execution profile 由以下事实组成：

```text
(semantic family, exact B, frontier interval, target-owned launch promises)
```

它决定 GQA/SWA/DFlash Attention execution envelope、kernel specialization、grid/block、workspace upper
bound、需要 materialize 的 KV frontier，以及所属 topology class。Exact request position、mask、accept/
commit count 和有效 context 不改变 profile 的数学语义。

### 2.3 Captured definition

每个 execution profile 保留一份 `cudaGraph_t` definition。Definition 在 Program 的稳定 IO frame、
workspace、Linear Attention state pool 和 Paged KV block-table addresses 上 capture，保存该 profile 的完整
node parameters，并作为 executable update source；definition 本身不 replay。

### 2.4 Topology-class executable

具有相同 node count、node types 和 dependency structure 的 definitions 共用一份 `cudaGraphExec_t`：

```text
(B=2, short frontier) definition ─┐
(B=2, long frontier) definition  ─┴─> B=2 topology executable

(B=3, short frontier) definition ─┐
(B=3, long frontier) definition  ─┴─> B=3 topology executable
```

Kernel function、arguments、grid、block 和 dynamic shared-memory configuration 可以通过 whole-graph
update 改变，不单独产生 executable。不同 `B` 不共用 executable，因为 tensor extents、copy bytes、grid
和部分 node parameters共同形成不同的结构实例；也不 capture active-slot subset，只有 `B=1..C`。

---

## 3. Ownership

```text
Program
  |
  |-- ordinary graph family             speculative off, B=1..C
  |-- MTP graph family                  MTP, B=1..C
  |-- DFlash-initial graph family       DFlash, B=1
  `-- DFlash-steady graph family        DFlash, B=1

Graph family
  |
  |-- ordered profiles
  |    |-- exact batch size
  |    |-- frontier interval
  |    |-- topology-class id
  |    `-- captured definition
  |
  `-- topology records
       |-- topology-class id
       |-- instantiated executable
       `-- currently installed profile
```

`src/core` 只拥有 graph definition/executable 的 CUDA RAII、capture、update、launch 与 destruction；不理解
frontier、batch、MTP、DFlash 或 target route。

Qwen3.6 family runtime 拥有 graph family、`(B, frontier)` lookup、installed-profile tracking、startup
qualification 和 runtime replay。Exact target 产生有限 execution profiles 与 topology-class ids。Operator
不持有 graph object，也不决定 graph family。

---

## 4. Current topology inventory

### 4.1 Ordinary concurrent decode

Ordinary target width固定为每 row 一个 token。每个 exact `B` capture 一套 definitions；frontier profiles
在相同 `B` 的 topology class 内通过 executable update 切换：

```text
ordinary family
  |-- B=1 topology class
  |-- B=2 topology class
  ...
  `-- B=C topology class
```

因此当前 ordinary executable 数量为 `C`，而不是 context profile 数量，也不是 slot subset 数量。

### 4.2 Concurrent MTP decode

Engine startup 固定 MTP window `K<=5`。每个 exact `B` definition包含一次 batched target verify、一次
batched acceptance/alignment，以及固定数量的 whole-batch proposal steps。Per-row `Pcur/Pnext`、acceptance
和 commit count只控制 valid prefix 与 state publication，不改变 schedule topology：

```text
MTP family
  |-- B=1 topology class
  |-- B=2 topology class
  ...
  `-- B=C topology class
```

当前 target 的 MTP frontier profiles 在每个 `B` 下共享一个 topology class，因此 executable 数量为 `C`。
以后若某个 target 的 frontier route确实改变 node dependency structure，它可以为同一 `B` 声明多个
topology classes；仍不按 exact frontier复制 executable。

### 4.3 DFlash `B=1`

DFlash-initial 验证 prefill 产生的第一批 proposal；DFlash-steady 执行 accepted target append、DFlash
proposal 和 target verification。SWA direct/split 与 target prompt/chunked route 会产生有限 topology
classes。DFlash 继续使用当前单请求 graph inventory，不因 ordinary/MTP 的 exact-`B` 迁移自动获得并发
schedule。

---

## 5. Startup lifecycle

Program construction 按以下顺序建立 graph assets：

```text
plan target frontier profiles
        |
cross product with reachable exact B
        |
warm every reachable operator route
        |
capture one definition per (B, profile)
        |
instantiate one executable per topology class
        |
update/replay every definition, short->long and long->short
        |
restore clean Program state
```

ordinary/MTP capture为每个 row 使用一张 private temporary Paged KV page，并在该 row 的临时 block table
中重复映射它。这样能够 capture任意 frontier envelope，而不为 graph construction保留 `C` 份完整 context。
Capture 完成后临时 allocation释放。

所有 definitions 使用同一 Program 的稳定地址。每个 class 选择第一份 definition instantiate，再依次安装
并 replay 同 class 的全部 definitions。双向 cycle确认 long-profile 和 later-short-profile update。

任何 capture、instantiate、update 或 replay 失败都会使 graph-enabled Engine construction失败；不会创建
per-profile executable fallback，也不会把失败推迟到 serving。Qualification 后 Program 清空 controls、
Linear Attention state、token counts 和 speculative state，并解除 temporary KV binding。

---

## 6. Runtime dispatch

每轮 runtime dispatch 为：

```text
round membership -> exact B
row frontiers    -> batch frontier envelope
        |
        v
select semantic family
        |
select (B, frontier) execution profile
        |
select profile's topology executable
        |
update executable when installed profile differs
        |
materialize required KV upper bound
        |
one full-round graph launch
```

Batch frontier envelope必须覆盖本轮所有 rows 的实际访问范围；exact row frontiers仍由 device controls提供。
同一 profile 的 steady path只执行 lookup、installed-profile comparison 和 `cudaGraphLaunch`。Profile crossing
在本轮 model work前更新 executable，不 capture 或 instantiate。

一个 Program 的 GPU execution unit串行提交，因此同一 executable不会被并发 update。Request 可以跨 round
改变 compact row，下一轮仅按新的 membership选择 `B` 与填写 selectors；graph不记忆 request identity。

`use_cuda_graph=false` 时不建立 definitions 或 executables，直接执行相同 schedule body。

---

## 7. Resource planning

Graph allowance 按 topology class 的最大 profile计算，而不是按 definition/profile 数量计算。Startup 的
`cudaMemGetInfo` check覆盖 executable、retained definitions以及 qualification触发的 driver/module allocation。

- ordinary concurrent：单个 exact-`B` class 的 conservative bound乘 `C`；
- MTP concurrent：target-owned MTP topology allowance乘 `C`；
- DFlash：ordinary `B=1` 加 initial/steady reachable classes。

Allowance 进入 device reservation，不通过静默缩小 advertised context capacity为 graph让路。Program
construction 后的实际 graph consumption若超过 allowance，Engine construction失败。

这种规划有意让 graph memory随 `C` 线性增长，但不随 `2^C` active-set组合或 frontier profile数增长。

---

## 8. Boundary behavior

### Batch membership changes

Request在 round boundary加入或退出后，下一轮直接选择新的 exact-`B` profile。没有 empty row、inactive
slot或 graph rebuild。

### Heterogeneous context frontiers

Profile选择使用覆盖所有 rows 的 batch frontier envelope；每行依旧读取自己的 position、Paged KV table row
和 valid frontier。较短 row不会因为 envelope较大而获得额外逻辑 context。

### New short request after a long request

若相同 `B` 的 executable仍安装 long profile，新一轮选择覆盖当前 batch的 profile并执行 reverse update；
不重建 Program。

### Prefix reuse at a nonzero frontier

第一轮直接选择覆盖命中 frontier的 profile，不从 profile zero逐级推进。

### MTP zero/partial proposals and terminal tail

`Pcur=0`、partial proposal 与 full proposal都进入同一 exact-`B` MTP graph。Terminal output policy在 replay
结束后提交 licensed prefix；不切换 ordinary graph，也不建立 ordinary-aligned graph。

### DFlash speculative tail

DFlash window不能继续时，单请求 state machine选择 ordinary `B=1` family。Graph layer只执行已经选择的
family，不重新解释 speculative eligibility。

---

## 9. Validation contract

Graph topology 或 ownership变化使用以下直接证据：

- startup 对每个 declared class完成 definition update/replay cycle；
- ordinary 与 MTP 的 `B=1..C` graph-on路径通过 public Engine route；
- MTP 的 zero/partial/full proposal extent使用同一 exact-`B` family；
- DFlash initial、steady 与 ordinary tail继续通过当前 `B=1` route；
- graph-on/off保持相同数学 schedule；
- graph allowance覆盖 configured `C` 的 reachable classes；
- steady round性能不因 graph ownership改变而显著回退。

性能比较必须保持 GPU占用条件一致。Profile transition的一次性 update cost与 steady replay分开解释，不把
环境竞争导致的吞吐变化归因于 graph architecture。
