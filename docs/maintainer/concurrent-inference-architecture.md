# NInfer 小规模并发推理架构

本文定义 NInfer 在单 GPU、单模型实例下支持少量并发请求的执行架构。典型
`max_concurrency` 为 2–8。

设计目标不是让多个请求轮流执行，而是让所有处于 decode 阶段的请求形成一次真正的 batched
model execution：一次 model traversal、一次 CUDA Graph replay 和一组 batched operators 同时为
多个请求产生结果。

本文只定义会影响模块边界、调度语义、资源所有权或执行正确性的决策。协议错误格式、allocator
实现、kernel 组织、target-specific profile 数值和测试计划分别属于对应的 serving、runtime、operator
和 qualification 文档。

---

## 1. Scope

### 1.1 Supported workload

- 单 GPU、单 resident model instance；
- 启动时固定 `max_concurrency=C`，典型 `C=2..8`；
- 运行时 `0..C` 个 admitted requests；
- Text 与 image/video prompt；
- ordinary decoding 与 engine-wide speculative decoding；
- streaming 与 non-streaming output；
- prefix reuse；
- 不同 prompt length、context length、generation limit 和 sampling configuration。

### 1.2 Non-goals

- request preemption、swap 或 pause/resume；
- 多请求 batched prefill 或 prefill/decode mixed forward；
- 多 GPU 或 distributed inference；
- priority、tenant QoS 或 deadline-aware GPU scheduling；
- 面向数十至数百请求的通用 continuous batching；
- serving 期间为新 shape 动态捕获 CUDA Graph。

### 1.3 Required sequence-state substrate

并发引擎要求下层提供 concurrent-capable sequence-state substrate，并以该 contract 已满足为前提。
该 substrate 必须提供：

- 不依赖 slot 或 per-request 物理连续区间的 shared, block-addressable growing-context storage；
- 按 sequence 分配的 fixed-size recurrent/backend state；
- 对 growing 和 fixed state 的统一 reserve、commit、truncate 和 release 语义；
- retained state 的 pin、ownership transfer 和 eviction 语义；
- 可由 attention 及其他 model operators 直接消费的 opaque per-sequence state view；
- 在一个 GPU execution unit 期间稳定的 state handles 和逻辑 valid frontier。

本文定义并发层如何预留、持有、组批和转移这些 state，不定义 substrate 的 page
geometry、物理 slab layout、allocator、block-table representation 或 page-aware operator implementation。

---

## 2. Core invariants

### 2.1 Maximal batched decode

在任一 round boundary，若有 `B>0` 个 decode-ready requests，下一次 decode round 必须包含全部
`B` 个请求。禁止把它们拆成多个 request-local forwards，也禁止为了复用单请求路径而依次执行。

### 2.2 Boundary-only membership

请求只在 GPU round 或 prefill chunk 完成后的 boundary 加入或退出 active execution。GPU work
in-flight 期间，slot binding、batch membership 和 sequence-state ownership 不变。

### 2.3 Fixed slots, shared state memory

`max_concurrency` 固定 control slots 的数量，不把 KV/context memory 平均分给 slots。KV cache、GDN
state 和 speculative-backend state 由 §1.3 的 Sequence-State Store 提供。Slot 只持有 allocation
handle 和 reservation，不拥有静态 context partition。

Engine configuration 必须能同时容纳 `C` 份 active fixed-size sequence state、`C` 行 execution
metadata 和一份 shared executor workspace；否则该 `max_concurrency` 无效。Retained state 可以使用
额外或当前空闲容量，但不能降低 `C` 个 active sequences 的 fixed-state guarantee。Growing
context 仍是 shared capacity，因此不承诺任意 `C` 个长上下文都能同时 admission。

### 2.4 Admission guarantees completion capacity

NInfer 不支持 preemption，因此 request 只有在其 prompt、声明的最大生成长度和必要的临时增长都能
获得完整资源承诺后才可 admission。已经 admitted 的 request 不会因为后来请求到达而被截断或逐出。

### 2.5 One prefill owner

同一时刻最多有一个 admitted request 执行 partial prefill。Prefill 以 bounded chunk 为单位，在
decode rounds 之间运行。其他等待 prefill 的请求仍留在 host queue，不占 slot 或 model state。

### 2.6 Single GPU execution owner

一个 GPU executor 串行提交所有 prefill chunks 和 decode rounds，并且是 active slot、sequence state
和下一轮 batch 的唯一修改者。CPU preparation、request ingress 和 output I/O 可以并行，但不能直接
推进模型状态。

### 2.7 Bounded ingress and output

等待请求数、输入字节、CPU preparation work，以及每请求和 aggregate response backlog 都有固定
上限。外部持续发送请求只会填满有限 queue，不能产生无界 request records、buffers 或 worker
threads。

### 2.8 Per-request semantic isolation

Sampling、RNG、stop conditions、generation limit、usage 和 output state 属于 request。它们不得依赖
request 当前位于哪个 slot 或 compact batch row，也不得影响同一 batch 的其他 requests。

---

## 3. Overall architecture

```text
 clients / CLI / OpenAI / Anthropic
                  │
                  ▼
┌──────────────── Server Frontend ─────────────────────┐
│ validate · bounded CPU preparation · pending FIFO    │
│ finite count/bytes/time · cancellation · responses   │
└───────────────────┬──────────────────────────────────┘
                    │ admission at a boundary
                    ▼
┌──────────────── GPU Executor ──────────────────┐
│ Slot Table                                     │
│ Scheduler                                      │
│ Batch Builder                                  │
│                                                │
│  PrefillChunk ─────┐                           │
│                    ├──> Model Executor          │
│  DecodeBatch[B] ───┘       │                   │
│                             ▼                   │
│                       round results             │
└───────────────┬───────────────────┬─────────────┘
                │                   │ committed output
                ▼                   ▼
┌─ Sequence-State Store prerequisite ─┐  Server Frontend
│ Text KV                            │  / async output
│ GDN / recurrent state             │
│ speculative-backend state         │
│ completion reservations           │
│ optional retained prefixes        │
└──────────────────────────────────────┘
```

各组件的职责如下：

| 组件 | 职责 |
|---|---|
| Server Frontend | 请求校验、有界 CPU preparation/pending work、取消输入和响应 I/O |
| GPU Executor | admission、调度、状态提交和全部 GPU submission |
| Slot Table | 保存 admitted requests 的稳定控制状态 |
| Batch Builder | 把全部 decode-ready requests 压紧为下一逻辑 batch |
| Model Executor | 使用选定 graph/profile 执行一次 whole-batch model schedule |
| Sequence-State Store | 前置 storage substrate；拥有 per-sequence model state 和容量 |

这些是责任和所有权边界，不要求采用同名的 C++ class。

---

## 4. Request and slot model

### 4.1 Request lifecycle

```text
RECEIVED
   │ validation and bounded pending capacity
   ▼
WAITING
   │ bounded CPU preparation, then admission:
   │ slot + state + completion reservation
   ├── complete prefix hit ─────────────────────► DECODE_READY
   ▼
PREFILL
   │ chunks; final chunk establishes decode anchor
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

Cancellation 或 failure 可以从任意阶段终止请求。`WAITING` 同时包含有界 CPU preparation 和等待 GPU
admission；二者是 server-side phases，不需要扩展成更多 model-execution states。

`DECODE_READY` 是稳定状态：request 已拥有完整 committed model frontier、下一次 decode round
需要的唯一 current token/anchor，且 sampling、RNG 和已发布输出与该 frontier 一致。是否属于
当前 in-flight round 由该 round 记录，不再建模成一个长期 request state。

Intermediate prefill chunks 只推进 prompt state。Final prefill 执行一次 target output selection，建立 decode
anchor；如果立即命中 terminal condition，请求不进入 decode batch。Complete prefix hit 必须同时
恢复等价 continuation frontier；单独的 KV match 不足以进入 `DECODE_READY`。

### 4.2 Slot

slot 是租给一个 admitted request 的稳定控制位置。它保存或引用：

- request identity 和 lifecycle state；
- sampling、RNG、stop 和 generation-limit state；
- shared pool 中 sequence state 的 handle；
- scheduler 所需的 prompt/decode progress；
- cancellation 和 terminal flags。

slot 不拥有：

- 固定比例的 context capacity；
- 永久 batch row；
- shared model workspace；
- model completion 后的 response buffers；
- slot release 后的 retained prefix state。

请求到达 `MODEL_FINISHED` 后，model result 移交 response path，slot 可以在当前或之后的 boundary
复用。Network completion 不属于 slot lifetime。slot index 也不是 external request identity；旧请求的
cancellation 或 completion 不能作用于之后的 occupant。

### 4.3 Batch row

batch row 只存在于一个 decode round。每个 boundary，Batch Builder 都重新建立 compact mapping：

```text
batch row -> slot -> sequence-state handle
```

例如：

```text
slots:          [0] [1] [2] [3]
decode-ready:    Y   N   Y   Y

logical rows:   [0] [1] [2]
row -> slot:     0   2   3
logical B:       3
```

empty slot 不进入 model batch，因此 slot reuse 或 hole 不需要不同的 execution model。

---

## 5. Request ingress and admission

### 5.1 Bounded ingress

在 expensive preparation 之前，Server Frontend 校验 request syntax、model capability、input size 和
有限 generation bound。通过校验的请求才能取得有界 CPU preparation 和 pending capacity。

host side 对以下资源设置有限上限：

- open 或 pending generation-request 数量；
- aggregate in-flight body 和 pending/input bytes；
- concurrent CPU preparation work；
- request 保持 unadmitted 的最长时间；
- per-request 和 aggregate response backlog。

若相关 count 或 byte limit 已满，新请求立即以 overload 拒绝。Engine 不因为连接存在就创建一条
per-request worker。

CPU preparation 产生 owning、immutable prompt representation。Prepared requests 按 preparation completion
顺序进入 admission FIFO，因此慢 media request 不阻塞已经完成 preparation 的请求。Queue timeout 从
请求首次取得 pending capacity 时开始，因而包含 preparation time。

### 5.2 Admission conditions

在 GPU boundary，Scheduler 检查 prepared FIFO 队首。只有同时满足以下条件才可 admission：

1. 存在 free control slot；
2. request 对 Engine 固定的 model 和 execution mode 合法；
3. 每类 state pool 都能承诺它的完整资源需求；
4. 若 request 需要 prefill，当前没有其他 prefill owner；
5. admitted lifetime 所需的 bounded host input 和 maximum owning-result capacity 可用。

Admission 是一次 atomic boundary transaction：prepared FIFO entry、optional retained-state claim、slot、
fixed/growing state reservation 和 host result capacity 要么全部取得并发布 admitted request，要么不改变
任何 request-visible ownership。不允许请求带着部分 reservation 回到 queue。

Prepared requests 之间使用 strict FIFO。暂时受阻的队首不能被绕过。Complete prefix hit 可以跳过
prefill，但不能绕过更早的 FIFO entry。

Admission 结果分为：

| 条件 | 结果 |
|---|---|
| request 非法或超过 semantic limit | 永久拒绝 |
| 即使没有其他 active request 也无法容纳 | 对当前 Engine configuration 拒绝 |
| pending queue 已满 | 以 overload 拒绝 |
| 独占时可容纳，但当前 slot 或资源不可用 | 保持 FIFO 等待 |
| admission 前 queue timeout 到期 | 以 queue-timeout 结束 |
| Engine 正在停止或不可用 | 以 unavailable 拒绝 |

只有 admitted request 获得 GPU completion commitment。Queued request 只有 bounded waiting commitment，
没有 execution commitment。

### 5.3 Waiting duration and sustained ingress

NInfer 不承诺 admission ETA。Queued request 等待到以下任一事件最先发生：

- 在 GPU boundary 成功 admission；
- cancellation 或 client disconnect；
- configured queue timeout；
- Engine shutdown 或 failure。

在 sustained ingress 下，有限 queue 填满后立即拒绝后续请求。已经 queued 或 admitted 的请求保持
FIFO position 和 resource commitment；later arrival 不能增加它们的 GPU workload 或 memory reservation。

---

## 6. Shared context and state memory

本节只定义 concurrent engine 使用 Sequence-State Store 的 resource contract 和 ownership
transitions；存储 substrate 本身由 §1.3 预先提供。

### 6.1 State ownership

每个 admitted request 从 Sequence-State Store 获得一份 sequence-state allocation。根据选定的
model mode，该 allocation 包含：

- Text KV cache；
- GDN convolution/recurrent state；
- speculative-backend persistent state；
- position 和 model-continuation metadata。

allocation 跟随 request，而不是 slot number。不同 request 可以占用差异很大的 context memory。

### 6.2 Completion reservation

Admission 为每个 request 计算 target-specific upper bound：

```text
uncached prompt
+ maximum permitted output
+ bounded speculative temporary growth
+ allocation rounding
```

每一类 shared state resource 都必须始终满足：

```text
active used capacity
+ active reserved-but-not-yet-used capacity
+ retained used capacity
<= total usable capacity
```

Prefill/decode 推进只把该 request 已有的 logical reservation 转换成 used capacity，不创建
新的资源要求。Physical blocks 是提前划归还是按需绑定，属于 Sequence-State Store。

Prefix reuse 把 retained used capacity 转为 active used capacity，不重复计费。Boundary restore 或
eviction 释放不再可达的 state，然后才重新计算 future reservation。

启动 prefill chunk 或 decode round 前，Scheduler 用 request 的 remaining reservation 和 logical
context/output limit 限制该 unit 最大可能的 state growth。没有合法增长空间的 request 直接 terminal，
不能启动可能越界的 unit。

这使 concurrency 与 context length 解耦：

- 一个 request 可以使用 pool 的大部分容量；
- 两个 request 可以使用完全不同的容量；
- active reservations 留下的容量不足时，新 request 等待；
- active request 不会在运行后期才发现声明的 output 无法容纳。

因此每个 request 都必须有有限 maximum output bound。Caller 未提供时，Engine 使用 configured
generation cap。

### 6.3 Fixed and growing state

KV cache 等 growing state 按 request context 计费。Fixed-size recurrent/backend state 按 admitted
sequence 分配。即使二者来自不同内部 pool，也必须在同一次 admission decision 中同时满足。

Growing state 由前置 substrate 以 block-addressable shared storage 承载，不要求单个 request 的
physical context 连续；model execution 通过 opaque state view 访问它。Concurrent engine 只操作
allocation handle、logical frontier 和 resource reservation，不参与 page 或物理地址管理。

GPU work in-flight 期间，per-request state handle 和 view 的含义必须稳定。Page size、slab
layout、virtual-address mechanism 和 allocator 属于 Sequence-State Store 的设计，不属于本并发架构。

### 6.4 Prefix reuse

Retained prefix 是从已结束 request 中分离出来的、单一 owner 的 SequenceState allocation。它保留当前
resume frontier，以及 prefill 期间显式保存的有限 reusable-boundary checkpoints。复用点只能是当前
frontier 或这些 checkpoints，不能是任意 longest token prefix。

Admission 成功时消费 retained entry，并把 SequenceState ownership 转移给新 slot：

- frontier reuse 直接从 retained frontier 继续；
- boundary reuse 把 KV logical length 截到 checkpoint，并恢复对应的 recurrent/backend state；
- complete prefix hit 直接进入 `DECODE_READY`，partial hit 只 prefill 未命中的 suffix。

一个 retained entry 同时只能被一个 active request 消费。多个 active requests 不共享同一份可写
sequence state，也不使用 copy-on-write branching。Request-local RNG、sampling、stop、generation
limit 和 output state 始终由新 request 创建。

Retained state 占用实际 state-pool memory，但不占 control slot，也不保留 future growth reservation。
Active admission 优先；cache occupancy 阻塞原本可行的 request 前，先驱逐 retained entries。只有在
slot 和完整 completion reservation 都已满足后才能转移 cache ownership。

Prefix lookup 只改变 uncached prompt work 和 memory requirement，不改变 scheduling 或 batch formation。

---

## 7. Scheduling model

### 7.1 GPU scheduling units

Scheduler 只提交两类 GPU work：

```text
PrefillChunk(request)
DecodeRound(all decode-ready requests)
```

完整 request 不是 scheduling unit。所有 GPU work 在一条 execution lane 上串行执行。

Model Executor 拥有一份地址稳定的 shared workspace，由串行的 GPU units 复用，不按 request
复制。Prefill owner 可在 Vision/Text phases 及多个 chunks 之间持有一份 request-transient lease；
final prefill、cancellation 或 failure 后释放。

### 7.2 Boundary processing

当前 GPU unit 完成，或 GPU idle 时收到 control event，GPU Executor 执行一次 boundary：

```text
1. commit the completed unit's per-request results
2. mark finished/cancelled requests and release their slots/state
3. admit FIFO requests while the admission rules allow it
4. choose and launch one next GPU unit
```

boundary 开始后到达的 event 留到下一 boundary。这保证一次 membership update 有限，持续 arrival 不能
无限延迟下一次 launch。

### 7.3 Decode/prefill policy

调度策略为：

```text
if the completed unit was a DecodeRound and a prefill owner exists:
    run one latency-bounded PrefillChunk
else if one or more requests are DECODE_READY:
    run one DecodeRound containing all of them
else if a prefill owner exists:
    run the next PrefillChunk
else:
    remain idle
```

因此 decode 和 prefill 同时持续 runnable 时：

```text
DecodeRound -> PrefillChunk -> DecodeRound -> PrefillChunk -> ...
```

没有 decode-ready request 时，prefill chunks 连续执行；没有 prefill owner 时，decode rounds 连续执行。

Prefill chunk profile 限制插入两个 decode rounds 之间的 GPU 时间。其具体 token/media extent 是经过
target 和 hardware qualification 的配置，不属于 scheduler semantic。Vision 和其他 prefill GPU phases
必须本身构成 bounded unit，或已被计入该 chunk 的 latency bound；不存在 scheduler 之外的
unbounded prefill work。

### 7.4 Joining and leaving decode

- request 在 final prefill 完成或 admitted complete prefix hit ready 后加入 active decode set；
- newly ready request 首次出现在下一个 boundary 构建的 batch；
- EOS、stop 或 generation limit 在 completed round commit 后移除 request；
- 每次 join/leave 后重新 compact batch rows；
- empty slot 永不产生 empty row。

### 7.5 Cancellation

Cancellation 不打断 in-flight GPU unit。第一个观察到 cancellation 的 boundary：

- 若 request 参加了刚完成的 unit，正常 commit 该 unit；
- 后续 unit 不再包含该 request；
- 释放它的 model state 和 slot；
- 按 serving contract 关闭 response output。

因此 cancellation 最多等待一个 membership 已固定的 GPU unit，再加一次 boundary processing。该策略
避免 partial-round rollback，并保持 shared round 中各 row 相互独立。

---

## 8. Batched model execution

### 8.1 Whole-model batch

logical batch size 为 `B` 时，一个 decode round 的执行形态为：

```text
B request descriptors
  -> one batched embedding/input stage
  -> one traversal of all model layers
  -> one batched lm_head
  -> one batched sampler
  -> B per-request round results
```

Control code 可以遍历 lightweight row metadata，但不能为每个 request 分别调用一次完整 layer 或 model
schedule。

Batch formation 只把 current token/proposal activations 和 row descriptors 压紧到 batch 维度。Per-request
KV/context、recurrent state、RNG 和 output ownership 保持独立，不拷贝、不拼接成一条
sequence；每个 row 通过 descriptor 引用自己的 state view。

### 8.2 Operator contract

| 算子族 | Batched execution 要求 |
|---|---|
| Linear、projection、dense FFN | 一个 operator schedule 消费 aggregate row/token extent |
| MoE | 对 aggregate token set routing，再对整个集合执行 grouped experts |
| Full attention | 以 ragged batch 消费 per-row context length 和 substrate-provided KV view |
| GDN / linear attention | 对独立的 per-row recurrent-state handles 执行一次 batched transition |
| `lm_head` | 一次产生全部 active rows 或 verification positions 的 logits |
| Sampler | 消费 per-row sampling configuration 和 RNG state，不拆分 batch |

不同 context length、stop condition、sampling value 或 generation limit 不形成不同 model batch。需要
不同 model topology 的 request option 必须在 Engine startup 时固定，否则作为 unsupported 拒绝。

### 8.3 Per-request commit

Model execution 产生 per-row provisional result。下一 boundary 对每行独立 commit：

- generated token 或 accepted token prefix；
- 与结果对应的 KV 和 recurrent-state progress；
- sampling/RNG 和 penalty-history progress；
- usage 和 output events；
- continue 或 terminal decision。

一行结束或接受较少 speculative tokens，不改变其他行的 commit result。

对每行而言，model-state progress、sampler/RNG progress、usage 和 owning output record 是一次逻辑
transaction。对应 output event 只有在这些 state 全部 commit 后才能对 response path 可见。

---

## 9. CUDA Graph model

### 9.1 Decode graphs by logical batch size

因为 `C` 较小且固定，Engine 为每个 logical batch size 捕获 exact decode graph：

```text
DecodeGraph[1]
DecodeGraph[2]
...
DecodeGraph[C]
```

`B=1` 是 first-class exact path。单请求不会运行永久 padding 到 `C` 行的 graph。
若 §9.3 需要有限 whole-batch profiles，则每个 `B` 对应这些预先捕获的 profile graphs；
上述记法省略 profile key。

每个 graph 表示 Engine 固定 model/decode mode 的一个完整 decode round。Graph 读取 stable descriptor，
其中包含当前 compact row-to-state mapping、exact per-row position、context length 和 request semantics。

### 9.2 Dynamic active set

active set 改变时，只更新 descriptor content 并选择匹配的预捕获 graph：

```text
[A]       -> DecodeGraph[1]
[A,B]     -> DecodeGraph[2]
[B]       -> DecodeGraph[1]
[B,C,D]   -> DecodeGraph[3]
```

Graph 不按 request identity、slot subset 或 slot bitmask 建 key。Join、leave 和 slot reuse 不触发
capture。

### 9.3 Context and prefill profiles

Exact context length 是 runtime descriptor value。若 exact target 对不同 context range 需要少量不同
kernel topology，则除 `B` 外可使用启动时固定并捕获的有限 whole-batch profiles。Profile
由 batch-level bound，例如 maximum row context-length bucket，选择一次；graph 仍以 exact per-row length 执行
ragged work。

Context profile 不能把 active set 分成 cohorts，也不能使用 request identity 或 active-slot combination 建
key。无法同时表示任意合法 per-row lengths 的 profile 不能作为 concurrent decode route；必须使用能
保持 whole-batch execution 的 route。

Prefill 是 single-request work，可以使用 target-specific fixed execution profile。它不改变 decode-batch
graph model，也不能在 serving 时触发会阻塞 active decode 的 graph capture。

---

## 10. Speculative decoding

Speculative decoding 是 Engine-level decode mode：

```text
speculative_backend = off | MTP | DFlash
```

同一 Engine 的全部 requests 使用相同 backend 和 proposal window。Scheduler 仍然只提交一个
`DecodeRound(all decode-ready requests)`。

该 round 内部执行：

```text
all pending request anchors
  -> one batched proposal phase
  -> one batched target verification
  -> per-request accepted-prefix result
  -> per-request state commit
```

MTP autoregressively 推进 proposal positions，每个 position 都在全部 active requests 上 batch 执行。
DFlash 通过一次 batched block forward 产生 proposal block。二者的区别只存在于选定的 decode graph
内部，不改变 slot、admission、scheduling 或 active-batch formation。

不同 request 可以接受不同数量的 proposals。Acceptance length 是 result metadata，不能据此拆分
verification、重放 model 或形成 acceptance cohort。Target model 始终是 output authority，只有 accepted
target/backend state 可以 commit。ReplaySSM 在同一 per-request state/commit interface 后工作。

每行的 valid proposal extent 受 remaining output/context capacity 限制。Fixed proposal window 中未使用
的位置被 mask，不能更新 request state 或 output。

若一行尚未 terminal 但 valid proposal extent 为零，同一 DecodeRound 必须对该行执行 ordinary
target progress，不能保留一个不前进的 active row。EOS、stop、output limit 或 context limit 在 per-row
commit 前截断 effective committed extent，model state 和 output 只提交同一有效前缀。

---

## 11. Completion and response handling

Admission 为 request 建立 owning result record，并以 maximum output bound 为它保留完整容量。每个
boundary 把 committed tokens 追加到该 record，因此 ordinary 或 multi-token speculative round 都不依赖
network progress 才能 commit。

Model completion 时，GPU Executor 终结 owning result record，释放 request 的 slot 和 unused reservation，
并释放 sequence state 或按 §6.4 把 exact reusable state 转移给 retained-prefix cache。上述 GPU
resource 被复用后，response path 仍可继续读取 owning record。

Committed output events 进入同时具有 per-request 和 aggregate bound 的 asynchronous transport queue。
Owning-result capacity 与 transport backlog 分开计费；即使 model slot 已复用，result capacity 仍计费到
response finished。

Network write、protocol serialization 和 user callback 永不在 GPU Executor 上运行。Client 过慢并超过
per-request transport bound 时，server 取消 active request，或关闭已 model-finished 的 response。Aggregate
result/transport capacity 已满时，new admission 等待，不能创建未计费 output；两种情况都不
阻塞 in-flight GPU round。

因此 model completion 和 response completion 是两个独立 lifetime：

```text
model lifetime:     admission ───────────── model finished
response lifetime:  request received ───────────────── response finished
active slot/state:   admission ──────────── model finished
retained state:                                 optional ───── eviction/reuse
```

GPU execution 或 state-integrity failure 属于 Engine-wide failure：停止 admission，active 和 queued
requests 以 unavailable/failure 结束，重新创建 Engine 后才能恢复 serving。Scheduler 不猜测 failed
shared round 中某一行可以安全继续。

---

## 12. End-to-end examples

### 12.1 New request joins an existing decode

```text
time ───────────────────────────────────────────────────────────►

GPU: Decode[A]
       │ boundary: admit B as the prefill owner
       ▼
     Prefill[B, chunk 0]
       ▼
     Decode[A]
       ▼
     Prefill[B, final chunk]
       │ B becomes decode-ready
       ▼
     Decode[A,B] -> Decode[A,B] -> ...
```

B 不执行单独的 decode forward，而是在 final prefill 后加入 A 的下一 logical batch。

### 12.2 Active batch grows and shrinks

```text
ready requests          selected graph

[A]                     DecodeGraph[1]
[A,B]                   DecodeGraph[2]
[A,B,C]                 DecodeGraph[3]
[B,C]       A finished  DecodeGraph[2]
[B,C,D]     D joined    DecodeGraph[3]
```

request identity 和 occupied slot 都不影响 graph identity。

### 12.3 Shared 128K context with two slots

假设相关 shared context capacity 为 128K units：

```text
request A: used prompt 50K + reserved output 30K = 80K commitment
request B: prompt       32K + reserved output 16K = 48K commitment
                                                       -----
                                                       128K
```

A、B 可以同时 admission。随着 context 增长，reserved capacity 转换为 used capacity，因此二者的
`used + remaining reservation` 始终不超过 128K；可以到达边界，但不能越过边界。

若 B 需要 `32K prompt + 32K output = 64K`，A 的 commitment 之后只剩 48K。B 保持在 pending
FIFO，直到 A 释放容量或 B 的 queue timeout 到期。NInfer 不会先 admission B，再在接近 128K 时决定
截断哪个 active request。

---

## Related documents

- [Serving behavior](../serving.md)
- [Qwen3.6-27B model semantics](qwen3.6-27b-model.md)
- [Qwen3.6-35B-A3B model semantics](qwen3.6-35b-a3b-model.md)
