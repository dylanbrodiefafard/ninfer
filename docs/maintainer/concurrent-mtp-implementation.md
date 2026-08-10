# NInfer 并发 MTP execution

本文定义 Qwen3.6 MTP 在小规模并发 Engine 中的稳定执行语义。通用 request、slot、scheduler、memory
ownership 与 compact batch 语义由[小规模并发推理架构](concurrent-inference-architecture.md)定义；Paged
KV 的物理资源契约由 [Paged KV context storage](paged-kv-cache.md)定义；算子层 batch 约束由
[并发 Decode 算子需求](concurrent-decode-operators.md)定义。本文只定义 MTP 如何闭合这些边界。

---

## 1. Product scope and invariants

Engine 启动时固定：

```text
C = max_concurrency, 1..8
K = MTP draft window, 1..5
```

运行时 active request 数量和 compact row membership 可以逐轮变化。并发 MTP 必须始终满足：

- 一个 DecodeRound 对全部 decode-ready rows 执行一次 batched target verification；
- 一次 batched MTP alignment 服务全部 rows；
- 每个 autoregressive proposal position 最多执行一次 whole-batch MTP step；
- linear、Attention、Linear Attention、lm/proposal head 和 sampling 均消费 aggregate batch，不在
  request loop 中重复 model execution；
- 一个 exact-`B` CUDA Graph replay 覆盖完整 MTP round；
- context、sampling、draft 数和 acceptance length 不形成 cohort；
- `B=1` 与 `B>1` 使用同一 Program transaction 和同一 MTP graph family；
- Text、Vision-prepared Text 和 prefix reuse 进入同一 lifecycle。

DFlash 使用相同 speculative Op 语义，但其 concurrent model schedule 不属于本文范围。

---

## 2. End-to-end control flow

```text
HTTP / CLI request
        |
        v
bounded pending queue
        |
        v
ConcurrentExecutor admission
        |  atomically acquires lane + Main KV + MTP KV entitlement
        v
staged prefill owner
        |  target chunk + matching MTP companion work
        v
DECODE_READY lanes
        |
        v
compact RoundMembership [lane_a, lane_b, ...]       B = membership.size()
        |
        v
Program::decode_batch(lanes, budgets)
        |  one exact-B MTP graph/eager body
        v
ragged licensed rows
        |
        v
output preview -> per-row commit count Q -> resolve_pending_batch
        |
        +---- continuing row -> next DECODE_READY boundary
        +---- terminal row   -> retained/released lane
        +---- cancelled row  -> release sequence resources
```

GPU execution unit 与 membership transaction 绑定。从 ingress build 开始，到该 membership 的所有 rows
完成 commit 或 abort 之前，Program 不启动下一 GPU unit。Compact row 只是本轮位置，不成为 persistent
request identity。

---

## 3. Ownership and persistent state

| Owner | State |
|---|---|
| Executor request | output session、generation budget、stop policy、timings、MTP statistics |
| Request lane | sampling config 与 token-count state、request control、transient prefill ownership |
| Sequence state | Main/MTP KV allocation、frontiers、Linear Attention committed slot、tail/boundary hidden、host drafts |
| Program | model、shared pools、workspace、fixed-capacity round frame、exact-`B` graph family |
| Round membership | immutable `row -> lane` mapping 和本轮 pending transaction |

每个 active sequence 维护：

```text
E    committed target execution frontier
S    committed output ledger frontier, S = E + 1 while decoding
Kt   Main KV valid frontier
Km   MTP KV valid frontier
H    committed target tail hidden
D[K] host-owned next-round draft tokens
P    valid draft count
```

正常 MTP decode boundary 要求：

```text
Kt = E
Km = E
ledger.size = S = E + 1
0 <= P <= K
```

Draft 只有几十 bytes，保存在 host `SequenceState`。它们随固定 ingress 上传，随固定 egress 下载，因此
sequence 在下一轮移动到任意 compact row 时不需要 device gather。Drafts 不是 reusable prefix state；request
结束、abort 或 prefix identity replacement 时立即失效。Main/MTP KV、tail hidden 和 boundary checkpoint
则可以随 resident sequence 保留。

---

## 4. Staged prefill

Concurrent executor 同一时刻只有一个 prefill owner。一个 MTP prefill scheduling unit 是不可拆分的：

```text
target chunk [a,a+n)
    -> target hidden h[a:a+n)
    -> shifted MTP inputs
    -> MTP prefix advances to a+n
    -> publish one scheduling boundary
```

MTP position `t` 消费 target hidden `h_t` 与下一 token 的 input：

```text
target hidden:  h[a]    h[a+1]  ... h[a+n-1]
MTP token:      x[a+1]  x[a+2]  ... x[a+n]
```

非 final chunk 的 `x[a+n]` 来自完整 prepared prompt 的 one-token lookahead。Final chunk 的最后一个
shifted input 是 target 刚产生的 first output anchor。Vision 列使用已经 composed 的 visual embedding；其
MTP RoPE position仍对应 target position `t`。因此 chunk boundary 不改变 MTP 数学语义，也不产生
device-to-host token 拼接。

Prefix reuse 采用以下 state transition：

| Reuse case | MTP action |
|---|---|
| Full reset | target 与 MTP 从零按 chunk 同步推进 |
| Append at `base` | MTP KV 回到 `base-1`，用 retained `h[base-1]` 与 suffix first input bridge |
| Boundary restore | 恢复 target boundary state/hidden，并执行相同 bridge |
| Exact full-prefix reuse | 从 retained tail hidden 产生 first anchor，再 bridge 并生成 initial drafts |

Final prefill unit同时产生 first licensed token、使 MTP prefix 与 target frontier 对齐，并根据剩余
output/context bound 生成初始 `P` 个 drafts。Executor 先按正常 output policy commit first token；若它已经
终止 request，drafts 直接丢弃。

---

## 5. Per-row round semantics

进入 round 时，第 `b` 行有：

```text
E[b]       committed execution frontier
R[b]       remaining output budget
Pcur[b]    current drafts available to verify, 0..K
Tcur[b]    target valid columns = Pcur[b] + 1
```

Target acceptance 在 device 上产生：

```text
A[b]       accepted drafts, 0..Pcur[b]
L[b]       licensed tokens = A[b] + 1
Enew[b]    E[b] + L[b]
Rnew[b]    R[b] - L[b]
Pnext[b]   drafts generated for the next round
```

Qwen3.6 MTP 的 extent bound 为：

```text
Pcur  <= min(K, R - 1, max_context - E - 1)
Pnext  = min(K,
             max(Rnew - 1, 0),
             max(max_context - Enew - 1, 0))
```

`Pcur` 在 launch 前确定，控制已经发生的 target verify。`Pnext` 依赖本轮 acceptance，只能在 device
accept 后计算，控制 alignment 后的 proposal generation。下一轮将保存的 `Pnext` 作为新的 `Pcur`。

Physical target width 固定为 `W=K+1`：

```text
row 0: Pcur=5 -> target valid columns 6
row 1: Pcur=2 -> target valid columns 3
row 2: Pcur=0 -> target valid columns 1
```

所有 rows 仍进入一个 target forward。Invalid tail 使用安全 id/position，所有 stateful Ops 对其 no-write，
结果不参与 acceptance。`Pcur=0` 也是统一 MTP graph 的正常 row；它产生一个 target token，并可根据新的
budget/context 继续生成 `Pnext`。不存在 zero/partial extent cohort。

---

## 6. Fixed-capacity MTP frame

Program 为 `C` 规划一份地址稳定的 MTP frame，并配套 pinned host ingress/egress。每轮只使用 exact-`B`
prefix，执行一次整块 H2D 和一次整块 D2H；KV/state payload 不进入 frame，只传 stable selectors。

### 6.1 Ingress

| Field | Logical shape | Meaning |
|---|---:|---|
| anchors | `[B]` | 当前 ledger tail token |
| frontiers / remaining budgets | `[B]` | `E/R`，accept 后 frontier 原位推进到 `Enew` |
| current extents | `[B]` | `Pcur` |
| current drafts | `[K,B]` | request-major current draft block |
| target valid columns | `[B]` | `Tcur=Pcur+1` |
| target RoPE positions | `[K+1,B]` | per-row Text/Vision-derived positions |
| Main/MTP KV table rows | `[B]` | shared paged-pool allocation selector |
| Linear state read/snapshot bases | `[B]` | target recurrent state selectors |
| continuation destinations | `[B]` | lane-owned tail-hidden slot |
| sampling configs | `[B]` | independent RNG/filter/penalty state |

Ingress 不包含 request id、stop condition、allocator object 或 raw KV pointer。

### 6.2 Device intermediates

```text
verify ids / target positions        [K+1,B] request-major
target logits                        [V,K+1,B]
target hidden                        [D,K+1,B]
accepted / licensed / next extents   [B]
alignment ids / hidden               [K+1,B]
selected target/alignment hidden     [D,B]
AR hidden                            [D,B]
```

`[K+1,B]` 的每行在物理上连续，因此 exact-`B` 是固定 frame 的连续 prefix。AR controls 和 next drafts
需要按 proposal step 取得连续 `[B]`，所以采用 step-major fixed-capacity storage：

```text
AR positions / rope / valid    physical [C, max(K-1,1)]
next drafts                    physical [C, K]
step s row b offset            s*C + b
```

当 `B<C` 时，完整 AR matrix 是带 pitch `C` 的 exact-`B` view；每个 step 的 `[B]` 仍连续。Transition Op
显式接收该 step stride，避免 repack，也避免把 inactive capacity rows带入 model execution。

### 6.3 Egress

| Field | Layout | Meaning |
|---|---:|---|
| licensed tokens | request-major `[K+1,B]` | 每行可供 output policy 检查的 prefix |
| licensed counts | `[B]` | `L` |
| accepted drafts | `[B]` | `A` 与 statistics metadata |
| next drafts | step-major fixed-`C` | 下一轮 candidates |
| next extents | `[B]` | `Pnext` |

Logits、hidden、KV 和 state snapshots 不下载。Full-licensed continuation hidden 在 graph 内 scatter 到
lane-owned store。

---

## 7. Batched GPU schedule

Exact-`B` MTP body 固定为：

```text
H2D fixed ingress
      |
prepare verify inputs [K+1,B]
      |
target forward, valid=Tcur[B]
      |
batched accept -> A[B], L[B], Enew[B], anchors'[B]
      |
select target hidden column A[B] -> continuation hidden [D,B]
      |
mtp_prepare_next_round -> alignment ids, Pnext, AR controls
      |
MTP alignment forward, valid=L[B]
      |
select alignment hidden column A[B] -> AR hidden [D,B]
      |
proposal head step 0 over B rows
      |
for s=0..K-2:
    one MTP AR model step over B rows, valid=(s+1 < Pnext[b])
    one batched proposal head
      |
scatter target continuation hidden + D2H fixed egress
```

Target valid extent 是 `Pcur+1`；alignment valid extent 是 `L=A+1`。后者只把 target 实际许可的 prefix
写入 MTP KV。Proposal step 0 直接消费 selected alignment hidden，不写 MTP KV；后续 `K-1` 个 AR steps
处理前一个 draft，并以 per-row `0|1` valid extent决定是否写 MTP KV。

Main KV 与 target Linear Attention snapshots 可以为 `Tcur` 写 provisional tail。MTP alignment/AR 也可以
在 committed frontier 之后写 provisional tail。Logical frontier 只在 host policy 决定 `Q` 后提交；未被
选择的 physical tail 下轮覆盖，不执行 rollback。

---

## 8. Operator contracts

Shared speculative Ops 使用单一 batched ABI：

- `speculative_prepare_verify_inputs`：按 `Pcur[B]` 构造安全的 fixed-width verify block；
- `speculative_accept_greedy_drafts`：逐行 sampling/accept，产生 licensed prefix、`A/L`，推进 device
  frontier，并更新该行 sampling token-count state；
- `speculative_select_accepted_hidden`：按逐行 selector 从 `[D,K+1,B]` 选出 `[D,B]`；
- `mtp_prepare_next_round`：根据 accept 后 frontier/budget 产生 alignment block、`Pnext` 与 pitched AR
  controls。

Target sequence-sensitive Ops 使用同一 `valid_columns[B]`、KV table rows 和 Linear state selectors。
GQA、GDN/Linear Attention 与 conv snapshot 都支持 `B=1..8` mixed-width invocation。特别地，B=1 的
fixed-width MTP graph 也会携带 valid extent；融合 GDN projection+conv+snapshot 路径直接 mask invalid
tail，只对有效 prefix发布 snapshot，不要求退回 composed execution。

Column-independent projection、MoE/FFN、norm 与 lm/proposal head 可以计算全部 safe aggregate columns；
stateful boundary与 accept extent确保无效结果不可见。

---

## 9. Output and commit transaction

GPU 返回每行 `L` 个 licensed tokens 后，Executor 独立执行 output preview，得到：

```text
0 <= Q[b] <= L[b]
continuing row: Q = L
terminal row:   1 <= Q <= L
cancelled row:  Q = 0 and abort
```

Program 在 GPU execution 后已为每行建立 `PendingCandidate`，记录 `base_E/Pcur/A/L/Pnext/frame_row`。
`resolve_pending_batch` 在相同 membership 上一次完成所有 rows：

- `Q=L` 且继续：提交 Main/MTP KV frontier `E+L`、对应 Linear snapshot、tail hidden 与 next drafts；
- terminal 且 `Q<L`：从仍存活的 target hidden frame 选择 column `Q-1`，batched scatter 修正所有此类
  rows，再提交精确 prefix；
- terminal `Q=L`：提交完整 licensed prefix并丢弃 next drafts；
- cancelled：丢弃整行 provisional state并释放 sequence resources。

Partial correction 是低频 boundary Op，不进入 steady graph。任一 shared GPU execution failure 使本次
Program execution整体失败，不尝试逐 row恢复未知 device state。

---

## 10. Context and fixed-state planning

MTP admission 同时预留两份 shared-pool entitlement：

```text
Main entitlement = prompt + maximum committable target context
MTP entitlement  = Main bound + at most K-1 provisional proposal positions
```

两份 entitlement 与 lane state 原子取得；任一 pool 不满足时 request 留在 pending queue。Main 与 MTP
cache geometry 不同，使用独立 Paged KV pools，但都按 request bound动态占用，不按 `C` 平分 context。
Program 每个 sequence 只持有 allocation handles，graph 使用 block-table row selector。

Target Linear Attention slots、tail/boundary hidden 和 sampling token counts 按 lane capacity 固定规划；它们
不随 context 增长。MTP drafter是 full-attention layer，不另建 Linear Attention state pool。

Prefill、verify、alignment 与 AR 前，Program materialize 本轮可能访问的 page upper bound。Commit 后仅更新
logical valid frontier；terminal/abort 释放未使用 entitlement，retained prefix 只保留实际有效资源。

---

## 11. CUDA Graph lifecycle

MTP decode 只有一个 semantic graph family：

```text
MtpGraph[B, context topology profile], B=1..C
```

同一 family覆盖 `Pcur/Pnext` 的 zero、partial 和 full values。Fixed `K`、exact `B` 与真实 kernel topology
属于 structural key；row identity、frontier、extent、acceptance、KV rows、positions 和 sampling config
都是 replay data。

每个 definition包含固定 ingress H2D、完整 §7 schedule 和固定 egress D2H。Context profiles 通过
whole-graph update 按 topology class共用 executable。Capture 使用每个 row 一张 private temporary Paged KV
page，并在该 row 的临时 block table 中重复映射它；这样可以覆盖任意 context envelope，而不为 graph
construction保留 `C` 份完整 context。Capture 完成后临时 allocation 全部释放。

Graph-off 路径调用完全相同的 MTP body。Startup capture/instantiate 必须覆盖所有 configured exact-`B`
profiles；serving 期间不 capture。Graph allowance 按 exact `B` 的 reachable topology executables规划，不按
slot subset或 active-set组合规划。

---

## 12. Engine integration boundary

`SpeculativeBackend::None` 与 `SpeculativeBackend::Mtp` 均由 `ConcurrentExecutor` 驱动，并共享：

- bounded ingress queue 与 pending timeout；
- lane admission和 retained-prefix eviction；
- single-owner chunked prefill 与 decode alternation；
- compact decode batch builder；
- ragged output preview/commit；
- cancellation、completion 与 slot recycling。

Program 的 concurrent entry 为：

```text
plan_request_for_lane
start_prefill_lane / advance_prefill_lane
decode_batch(lanes, budgets)
resolve_pending_batch(lanes, commit_counts, terminal, cancelled)
abort_lane
```

Program 不暴露另一套 single-request MTP round。`decode_batch` 对 MTP 始终选择统一 exact-`B` MTP body；
`B=1` 只是 compact membership 的一个正常大小。

---

## 13. Required observable behavior

该设计由以下直接行为约束：

- `B=1,2,8` 中 current/next extent、acceptance、context 和 sampling 可逐行不同且无 state 串扰；
- `Pcur=0` 的 row 仍产生一个 target token，并能在同一 graph 中建立后续 drafts；
- target、alignment 与每个 AR position 分别只有一次 whole-batch model schedule；
- Text、Vision 和 prefix bridge 都维护 `Km=E` 的 decode boundary invariant；
- terminal partial prefix提交精确的 KV/Linear state/tail hidden；
- graph-on/off 使用同一数学 transaction；
- `B=1` 不因 fixed-width masking失去原有融合 execution route；
- DFlash 的 B=1 behavior继续使用相同 shared speculative Op ABI。

测试只保护上述公开 Op 或 Engine observable contract，不为 private frame 字段、helper shape 或已删除路径
增加结构性测试。性能判断以 whole-round/whole-engine 数据为准；只有数据定位到具体 Op 时才建立临时
internal-launch benchmark。
