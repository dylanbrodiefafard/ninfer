# ReplaySSM GDN 引擎集成设计

本文定义 ReplaySSM 进入 Qwen3.6 Engine 后的状态布局、执行路径和提交事务。数学与数值要求见
[`replayssm-gdn.md`](replayssm-gdn.md)，已经实现的 record/fold 存储和 Op 合同见
[`replayssm-gdn-implementation.md`](replayssm-gdn-implementation.md)。本文以这些合同为输入，不重新
定义 record format 或 recurrent 数值路径。

集成完成后的核心结构是：

~~~text
MTP proposal ───┐
                ├─ target verify：读取 committed state，产生 outputs + raw records
DFlash proposal ┘
                                      |
                                      v
                           CPU 确定最终输出前缀
                                      |
                                      v
                    一次 all-layer Fold 提交 GDN state
                                      |
                                      v
                      提交其余 target/backend state
                                      |
                                      v
                                发布输出
~~~

MTP 和 DFlash 只在 proposal 生成及其私有 companion state 上不同。它们共用 target verify 的 Record
路径、最终 accepted-prefix 解释和 GDN Fold 提交事务。

---

## 1. 集成目标

本阶段完成以下结果：

1. speculative target verify 从逐位置完整 state snapshot 切换到 ReplaySSM Record；
2. CPU 确定最终接受前缀后，一次 Fold 更新本轮全部 rows 和全部 GDN layers；
3. MTP 与 DFlash 使用同一个 speculative target-state commit 路径；
4. linear-attention state pool 固定为每个 lane 两份完整 state，不再随 draft window 增长；
5. ordinary decode、prefill 和 turn checkpoint 继续使用同一个 state pool；
6. committed GDN state、KV frontier、continuation hidden、backend continuation representation 和输出
   保持同一逻辑提交边界；
7. CUDA Graph 捕获 Record，Fold 保持为 CPU 决策后的 eager kernel。

现有 Snapshot Ops 继续保留。Ordinary decode 使用它的单列形式作为原地 state update leaf；
speculative runtime 不再建立或选择 snapshot trajectory。

---

## 2. Round 语义

### 2.1 符号

| 符号 | 含义 |
|---|---|
| \(C\) | startup-fixed `max_concurrency`，当前为 1–8 |
| \(B\) | 本轮 compact decode batch 的 row 数，\(1\le B\le C\) |
| \(D\) | startup-fixed speculative draft window |
| \(T\) | record physical width，\(T=D+1\) |
| \(v_b\) | row \(b\) 本轮 target verify 的 valid input columns |
| \(p_b\) | target 许可的 output token 数 |
| \(m_b\) | output policy 最终接受并提交的 output token 数 |
| \(E_b\) | round 前 committed execution frontier |
| \(S_b\) | round 前 committed ledger frontier，始终为 \(E_b+1\) |

每个 speculative row 至少执行当前 anchor，因此

\[
1\le v_b\le T.
\]

若 target 接受了 \(A_b\) 个 drafts，则

\[
p_b=A_b+1,
\qquad
1\le p_b\le v_b.
\]

非取消 row 的最终提交满足

\[
1\le m_b\le p_b.
\]

继续运行的 row 必须提交全部 licensed outputs，即 \(m_b=p_b\)。Terminal row 可以因 EOS、stop 或
output limit 提交严格前缀。若 membership 内的请求在 GPU 执行期间被取消，则该 row 使用
\(m_b=0\)，随后释放整个 sequence state。

### 2.2 Transition 数量与 output 数量相同

Round 前的 current anchor 位于 position \(E_b\)，尚未作为 target input 执行。Verify inputs 是：

~~~text
column 0 = current anchor
column 1 = draft 0
...
column D = draft D-1
~~~

每执行一个 input transition，就产生一个 candidate output。最终接受前 \(m_b\) 个 outputs 时，需要提交
的恰好是前 \(m_b\) 个 input transitions。因此 Fold control 直接使用：

\[
\texttt{commit\_columns}[b]=m_b.
\]

提交后：

\[
E'_b=E_b+m_b,
\qquad
S'_b=S_b+m_b=E'_b+1.
\]

最后一个 accepted output 成为 position \(E'_b\) 上尚未执行的新 anchor，而 GDN state 已包含 positions
\([0,E'_b)\) 的 transitions。这与下一轮读取该 state 并执行新 anchor 的语义完全一致。

### 2.3 Pending 的含义

Target execution 完成后，licensed tokens 已经产生，target KV、candidate hidden、Replay records 以及
backend-private staging 也已物化，但最终 \(m_b\) 尚未确定。此时 request 进入 `Pending`：

- `execution_frontier` 和 `ledger_frontier` 仍表示 round 前 committed frontier；
- `PendingCandidate` 保存 `base_E`、`base_S` 和 `produced=p_b`；
- licensed token bytes 留在稳定的 host egress 中，直到本轮 resolve 结束；
- Replay records 留在 Program-owned arena 中；
- `RoundMembership` 保持 `row -> lane` 映射和相关 SequenceState 的所有权。

SequenceState 不需要先把 licensed suffix 写入 ledger 再在 terminal 时截断。Resolve 在 \(m_b\) 确定后只
追加最终接受的前缀。这样 `Pending` 期间的权威 host frontier 与仍未 Fold 的 committed GDN checkpoint
始终一致。

---

## 3. Linear-attention state 布局

### 3.1 两个固定 slot plane

Program 的 linear-attention state pool 固定包含 `2*C` 个 absolute slots：

| Absolute slot | 所有者与语义 |
|---|---|
| `[0,C)` | lane 的 current committed state |
| `[C,2C)` | lane 的 turn checkpoint |

映射为：

~~~cpp
current_state_slot(lane)         = lane;
turn_checkpoint_state_slot(lane) = max_concurrency + lane;
state_slot_count                 = 2 * max_concurrency;
~~~

一个 absolute slot 同时选择全部 GDN layers 的 BF16 causal-conv history 和 FP32 recurrent state。
`LinearAttentionStatePool` 仍是唯一 physical owner；ReplaySSM 不建立第二套 state pool。

### 3.2 为什么需要两份 state

一个 retained lane 可以同时拥有：

1. 当前生成 frontier 的 committed state；
2. prompt 内 assistant/turn boundary 的 checkpoint，供后续不同 suffix 恢复。

两者对应不同 position，且 boundary 恢复发生时 current state 仍然存在。因此每 lane 需要两个完整
state images。最多 \(C\) 个 lanes 都可能持有有效 boundary；在 startup-fixed capacity contract 下，动态
boundary pool 不能降低必须保证的最大容量。

### 3.3 SequenceState

`SequenceState::lane` 是 current/turn-checkpoint slot 的唯一定位信息。以下 snapshot-era 字段删除：

~~~text
linear_state_base
linear_state_capacity
current_linear_state_slot
~~~

状态操作固定为：

| 操作 | State effect |
|---|---|
| Full reset | zero `current_state_slot(lane)` |
| Prefill/append | 在 `current_state_slot(lane)` 原地推进 |
| 保存 turn checkpoint | copy `current_state_slot(lane) -> turn_checkpoint_state_slot(lane)` |
| 恢复 turn checkpoint | copy `turn_checkpoint_state_slot(lane) -> current_state_slot(lane)` |
| Ordinary decode | 在 `current_state_slot(lane)` 执行一个 transition |
| Speculative verify | 只读 `current_state_slot(lane)`，写 Replay records |
| Speculative resolve | Fold 到 `current_state_slot(lane)` |

`clear_lane` 使 turn checkpoint 的生命周期元数据失效；下一次 Full reset 在使用前清零 current slot。
无效 checkpoint 的旧 bytes 不具有语义，也不需要主动清零。

---

## 4. Replay record arena

### 4.1 Physical layout

当且仅当 startup backend 为 MTP 或 DFlash 时，Program persistent layout 规划一份
`GdnReplayRecordLayout`：

~~~text
record_capacity = C
width           = draft_window + 1
layers          = target GDN layer count
geometry        = exact target GDN geometry
~~~

四个 planes 沿用已经实现的合同：

| Plane | DType / shape |
|---|---|
| conv | BF16 `[Cp,T,C*L]` |
| key | BF16 `[128,Hq,T,C*L]` |
| value | BF16 `[128,Hv,T,C*L]` |
| gate | FP32 `[2,Hv,T,C*L]` |

Layer \(\ell\) 的 physical record row \(b\) 位于 outer index

\[
r(\ell,b)=\ell C+b.
\]

Arena 使用本次 Engine 配置的 exact \(T\)，不按 MTP/DFlash 的最大允许窗口预留。

### 4.2 Ownership 与 lifetime

Record arena 是 Program-owned transaction storage，位于 `PersistentLayout`，并在 Program 构造时绑定为
一份 `GdnReplayRecords`：

- 地址在全部 CUDA Graph definitions 和 replays 期间稳定；
- 与 shared execution workspace 同时存在；
- 从 target verify 开始存活，跨越 D2H、CPU output preview 和 Fold；
- 在本轮 resolve 完成后由下一轮复用；
- 不属于任何 request 或 SequenceState。

一个 Program 只需要一份 arena。GPU executor 串行提交 GPU units，并在当前
`RoundMembership` resolve 完成前不提交下一 round；因此 Program-owned frame 和 records 不会
提前被覆盖。`RoundMembership` 只保留本轮 `row -> lane` 关系，不拥有 record storage。
Exact-`B` graph 只写每层 records 的 physical rows `[0,B)`。

Ordinary backend 不分配这块 arena。

### 4.3 显存容量

一份完整 GDN state image 包括全部 layers 的 recurrent state 和 conv history：

| Target | One state image |
|---|---:|
| Qwen3.6-27B | 146.8125 MiB |
| Qwen3.6-35B-A3B | 61.40625 MiB |

在 `C=8` 的最大窗口配置下：

| Target/backend | 现有 state slots | 新 state + records | 节省 |
|---|---:|---:|---:|
| 27B MTP，`D=5,T=6` | 8,221.500 MiB | 2,430.84375 MiB | 5,790.65625 MiB |
| 35B-A3B MTP，`D=5,T=6` | 3,438.750 MiB | 1,022.2265625 MiB | 2,416.5234375 MiB |
| 35B-A3B DFlash，`D=15,T=16` | 8,351.250 MiB | 1,088.4375 MiB | 7,262.8125 MiB |

现有布局为每 lane `D+2=T+1` 个 state images；新布局为全 Program `2C` 个 state images，加一份
capacity-`C` record arena。上表不包含 weights、KV cache、ordinary workspace 和 CUDA Graph memory。

---

## 5. Row、lane、state slot 与 record row

这些标识承担不同职责：

| 标识 | Lifetime | 含义 |
|---|---|---|
| batch row `b` | 当前 round | compact activation/record row，`0 <= b < B` |
| lane | SequenceState lifetime | Program 内稳定的 continuation-state owner |
| linear state slot | Program lifetime | all-layer GDN state pool 中的 absolute slot |
| KV table row | KV allocation binding lifetime | paged KV block-table selector |
| record outer row | 当前 records 内容有效期 | `layer*C + b` |

本设计固定：

~~~text
linear_state_slot[b] = lanes[b]
record_row[b]        = b
~~~

`record_row[b]=b` 是 Op 由 `rows` span 中的位置定义的固定索引关系。Engine 不建立
`record_row` field、selector tensor 或第二份 row mapping。

Fold control 由同一个映射生成：

~~~cpp
fold_rows[b] = {
    .linear_state_slot = static_cast<std::int32_t>(lanes[b]),
    .commit_columns    = static_cast<std::int32_t>(accepted_tokens[b]),
};
~~~

例如本轮 compact rows 对应 lanes `{0,2}`：

~~~text
record row 0 -> state slot 0
record row 1 -> state slot 2
~~~

Records 因此始终按本轮 B 紧凑写入，persistent state 则始终按 lane 定位。Batch compact 后的 row 变化不会
迁移或复制 state。

### 5.1 Decode ingress

Ordinary、MTP 和 DFlash 的 ingress 都使用一个 I32 `lanes[C]` field。它同时提供：

- GDN initial state selector；
- continuation hidden destination selector；
- DFlash pending-feature/cyclic-state selector；
- Fold 的 host-side destination slot 来源。

以下重复 selectors 从 ingress、RoundState 和 TextContext 删除：

~~~text
linear_state_read_slots
linear_state_snapshot_base_slots
continuation_slots
~~~

`continuation_slots` 的语义由统一的 `lanes` 取代。KV table row 仍保持独立，因为 paged allocation 的
physical table-row binding 不是 linear-state slot contract。

Prefill 不使用 DecodeBatchFrame。RoundState 中的 scalar `linear_state_read_slot` 和
`linear_state_snapshot_base_slot` 同样删除；prefill 直接接收 host-known 的
current/turn-checkpoint absolute slots。

---

## 6. Text/GDN 执行路径

### 6.1 显式 state action

`TextPhase::Verify` 同时用于 ordinary decode 和 speculative target verify，但两者的 GDN state effect
不同。TextContext 内部增加明确的 state action：

~~~cpp
enum class GdnStateAction {
    UpdateInPlace,
    RecordForReplay,
};
~~~

调用组合固定为：

| Text execution | `TextPhase` | `GdnStateAction` |
|---|---|---|
| Prefill | `Prefill` | `UpdateInPlace` |
| Ordinary decode | `Verify` | `UpdateInPlace` |
| Speculative target verify | `Verify` | `RecordForReplay` |

Public schedule entry 决定 action；layer loop 不通过 nullable pointer 或 backend enum 猜测 state effect。
`RecordForReplay` 必须绑定 Program 的 `GdnReplayRecords`，`UpdateInPlace` 不消费 records。

### 6.2 Prefill

Prefill 始终在当前 lane slot 原地推进：

~~~text
read slot  = lane
write slot = lane
~~~

若 chunk 在请求的 turn-checkpoint frontier 结束，则在该 chunk 的 GDN work 之后执行一次 all-layer slot copy：

~~~text
lane -> C + lane
~~~

后续 chunk 继续在 `lane` 原地推进。RestoreTurnCheckpoint 在 prefill 开始前先执行
`C+lane -> lane`，此后进入相同的 prefill 路径。

TextContext 不再从 device scalar 回读 initial slot，也不再维护 request-local base/capacity group。Prefill
配置只传入由 lane 推导出的 current/turn-checkpoint slots。

### 6.3 Ordinary decode

Ordinary decode 处理 width 1。它继续复用现有 Snapshot Op leaf，但将 initial 和 destination selectors 都
绑定为 `lanes`：

~~~text
initial_state_slots = lanes
snapshot_base_slots = lanes
width               = 1
~~~

Snapshot Op 的现有合同允许 row 在完整加载自己的 initial state 后覆盖同一 slot。因此这条调用就是一个
batched、单 token、原地 GDN update，不产生 trajectory，也不需要额外 state slot。为同一数学和 kernel
路径再增加一个仅改名的 Update Op 不提供新的语义或性能能力。

### 6.4 Speculative target verify

对每个 GDN ordinal `gidx`：

1. 取得 `records.layer(gidx, B)`；
2. exact Variant 的 record projection wrapper 调用 `gdn_input_proj_conv_record`，以 `lanes` 选择初始
   conv state，并写 `layer.conv`；
3. `gated_delta_net_replay_record` 以同一 `lanes` 选择初始 recurrent state，写
   `layer.key/value/gate`，并产生正常 verify output；
4. source conv/recurrent state 保持只读；
5. invalid columns 仍由已有 valid-column contract mask，Fold 只消费最终接受前缀。

DFlash feature tap 继续从 target layer outputs 写入 `pending_features`，不改变 GDN Record 路径。

### 6.5 Exact Variant 与 workspace planning

两个 exact Variants 各增加：

- `gdn_input_projection_record(...)`；
- `gdn_input_projection_record_workspace_capacity_bytes(...)`。

它们按各自已经存在的 Q4/Q5、NVFP4 或 W8 profile 调用对应 record Op，不在 family runtime 中选择
weight-format route。

Workspace planner 删除含义模糊的 `snapshot` bool，并按三条实际执行路径计算：

| 路径 | Projection/conv workspace | Recurrent workspace |
|---|---|---|
| Prefill | existing prefill route | existing prefill route |
| Ordinary width-1 update | existing snapshot width-1 query | none |
| Speculative Record | record workspace query | none |

Fold 不使用 workspace。

---

## 7. MTP 与 DFlash 的公共 target verify

现有 `schedule::target_verify_accept` 是统一接入点。改造后的 `TargetVerifyFrameView` 包含：

- target ids、positions、valid columns 和 text KV rows；
- `lanes`；
- target hidden/logits/tokens；
- drafts、frontiers、licensed results 和 accepted-draft metadata；
- Program-owned `GdnReplayRecords` 的 non-owning reference；
- 可选 DFlash feature sink。

公共执行顺序是：

~~~text
prepare target verify inputs
  -> TextContext target forward with RecordForReplay
  -> target logits / argmax
  -> speculative accept
  -> select hidden at target-licensed prefix
  -> scatter provisional continuation hidden by lanes
~~~

MTP 在公共 target verify 之后继续完成 alignment 和下一轮 proposal。DFlash 在公共 target verify 之前完成
block proposal，并通过 feature sink 保存本轮 target features。两者不分别实现 GDN state selection 或
commit。

---

## 8. CUDA Graph 边界

### 8.1 Graph 内

Exact-`B` MTP/DFlash graph 包含：

- ingress H2D；
- backend proposal；
- target verify 的 projection/conv Record 与 recurrent Record；
- target accept 和 continuation-hidden selection；
- backend 后续计算；
- compact egress D2H。

Record arena 是固定地址的 graph output。不同 exact-`B` definitions 捕获同一 arena，各自只绑定
`records.layer(gidx,B)` 的 `[0,B)` rows。

Graph representative controls 使用 `lanes[row]=row`，并为对应 current slots 提供有效初始 state。Capture
不需要清零 record arena：每次真实 verify 会完整覆盖各 row 的 valid prefix，Fold 不读取 invalid tail。

### 8.2 Graph 外

CPU 必须先读取 licensed results，并经过 OutputSession preview 才能得到最终 \(m_b\)。因此 Fold 在 graph
外执行：

~~~text
graph completion + D2H synchronization
  -> CPU output preview
  -> eager Fold / commit tail
  -> commit-tail synchronization
~~~

这是每个 speculative round 的一次额外 all-layer Fold launch 和一次 commit-tail synchronization。Fold
不创建独立 CUDA Graph。

---

## 9. Speculative state commit transaction

### 9.1 Resolve 输入合同

Executor 把本轮 frozen membership 原样交给 `resolve_pending_batch`；`accepted/terminal/cancelled`
与该 membership 按 row 对齐。Speculative resolve 的输入合同是：

- lanes 数量与 `accepted/terminal/cancelled` row 数一致；
- 每个 lane 对应本轮的 `PendingKind::Speculative`；
- 非取消 row 满足 `1 <= accepted <= pending.produced`；
- continuing row 满足 `accepted == pending.produced`；
- cancelled row 使用 `accepted == 0`。

Target decode 已在建立 PendingCandidate 前验证
`produced <= target_valid_columns <= T`，因此最终有

\[
0\le m_b\le p_b\le v_b\le T,
\]

满足 Fold 的跨 Op 前置条件。

### 9.2 Fold rows

Fold rows 按原始 batch row 顺序构造，数量始终为 B：

~~~cpp
for (std::size_t b = 0; b < B; ++b) {
    rows[b].linear_state_slot = static_cast<std::int32_t>(lanes[b]);
    rows[b].commit_columns = cancelled[b]
        ? 0
        : static_cast<std::int32_t>(accepted[b]);
}
~~~

Rows 不筛选、压缩或重排。`gdn_replay_fold(records, all_layer_states, rows, stream)` 一次处理本轮全部 rows
和 layers。Cancelled row 的 `commit_columns=0` 是严格 no-op；该 row 随后释放。

### 9.3 GPU commit tail

DFlash graph 在 target verify 前已经把上一轮保留的 feature 区间
`[old_context_frontier,E_b)` append 到 companion context。Graph completion/D2H 同步后，Program 把
`dflash_context_frontier` 发布为 `E_b`。这只是把 DFlash context 补齐到本轮之前已经提交的
target frontier，不提交本轮 candidate suffix。此时 `pending_features[:,j,lane]` 保存本轮
position `E_b+j` 的 target features。

同一 stream 上按以下顺序 enqueue：

1. 一次 all-layer GDN Fold；
2. 若存在 terminal 且 `m_b<p_b` 的 row，则为原始 B 行构造一份 selector：partial row 使用
   `m_b-1`，其他 rows 使用原来的 `p_b-1`；随后对原始 B 行执行一次 batched hidden select/scatter；
3. 对非取消 terminal DFlash rows，把每个 lane 的 target-feature interval `[E_b,E_b+m_b)` append 到
   DFlash context；
4. 一次 `device.synchronize()`。

Hidden correction 必须在 DFlash flush 复用 decode ingress fields 之前入队；stream ordering 保证 correction
读取原 target frame 后，flush 才能覆盖相同 ingress backing。Fold 自身只消费 records、state 和 host-packed
row controls，不依赖 decode ingress。

DFlash continuing rows 不在 resolve 中 append 本轮 target features。其 context frontier 保持在 round 前的
\(E_b\)，下一 DFlash graph 先把已提交区间 `[context_frontier, execution_frontier)` append，再生成并覆盖
新的 pending features。Terminal row 没有下一 graph，所以 resolve 必须把 `[E_b,E_b+m_b)` flush 到
context，保证 retained sequence 的 backend frontier 与最终 target frontier 一致。

现有同步型 `flush_dflash_context_batch` 拆成 enqueue-only 的 GPU 部分和同步后的
host-frontier publish。GPU 部分显式接收每行 `start=E_b`、`count=m_b`，不从一个可能滞后的
host frontier 重新推导本轮 feature 起点。它把 terminal DFlash rows 单独组成自己的
compact append batch；该 batch 通过 lane selector 读取 `pending_features`，与 Fold 必须保持原始 B 行的合同
无关。它不在公共 commit tail 内自行同步。

~~~cpp
enqueue_dflash_context_append(lanes, starts, counts, stream);
~~~

`lanes/starts/counts` 只描述 terminal DFlash subset；对其中的 row `i`，`starts[i]=base_E`、
`counts[i]=accepted_tokens`。

### 9.4 Host commit

Commit-tail synchronization 成功后，对每个非取消 row 提交同一个 \(m_b\)：

1. 从稳定 host egress 追加前 `m_b` 个 licensed tokens 到 ledger；
2. 将 prefix identity 推进 `m_b`；
3. 设置 `execution_frontier = base_E + m_b`；
4. 设置 `ledger_frontier = base_S + m_b`；
5. 把 Text KV logical valid frontier 设置到 `base_E + m_b` 并 trim rejected suffix；
6. 设置 continuation hidden 为 valid；
7. 把 MTP KV logical valid frontier 设置到 `base_E+m_b` 并 trim rejected suffix；MTP continuing row
   发布 execution 产生的 next drafts，terminal row 清空 draft continuation；
8. DFlash continuing row 保持 graph completion 后已发布的 context frontier `base_E`；terminal row
   设置为刚刚 flush 完成的 `base_E+m_b`；
9. 将 lifecycle 置为 `Active` 或 retained `Complete`，清空 PendingCandidate。

Commit-tail GPU 执行与等待时间计入每个非取消 row 的 `decode_seconds`；否则接入
ReplaySSM 后的 decode throughput 会系统性遗漏 Fold 成本。

GDN state 不需要 host cursor 更新：current state 的 absolute slot 始终为 lane，Fold 已在同一 slot 中把
\(S_0\) 原地替换为 \(S_{m_b}\)。Boundary slot 保持不变。

Cancelled row 在 Fold no-op 后直接 `clear_lane`，不追加 ledger、不发布 continuation state。

OutputSession 的 preview 只产生尚未发布的决定。Executor 必须在 `resolve_pending_batch` 完成后才执行
`commit_preview`、budget commit 和 output event publication。因此外部永远不会观察到尚未完成 Fold 的
token。

### 9.5 单行入口

Speculative `B=1` 仍由 `resolve_pending_batch` 进入上述相同事务。现有
`resolve_pending_lane(lane, accepted_tokens, terminal)` 改名并收窄为
`resolve_prefill_lane(lane, terminal)`，只解析 final-prefill 的 `PendingKind::Begin`；final prefill 固定提交
它唯一许可的 token，不再接收多余的 accepted count。

一个脱离原始 batch 的 lane-only API 不具有 Replay record row identity，不能解析 speculative pending
round。这个边界保证所有 speculative rows，包括 `B=1`，都经过同一个 Fold transaction。

---

## 10. DFlash 与 MTP 私有状态

ReplaySSM 只替换 target GDN snapshot trajectory。公共提交事务仍按最终 \(m_b\) 协调其他状态：

| State | MTP | DFlash |
|---|---|---|
| Target GDN | 公共 Record/Fold | 公共 Record/Fold |
| Target Text KV | commit/trim 到 `base_E+m_b` | commit/trim 到 `base_E+m_b` |
| Target continuation hidden | full prefix 直接使用；partial terminal correction | 同左 |
| Proposal continuation | continuing 时发布 next MTP drafts | 每轮重新 block propose |
| Companion context | MTP KV 按 committed frontier 保留 | pending target features + explicit context frontier |

MTP alignment/next-proposal 保持在 target verify 之后，DFlash block proposal 保持在 target verify
之前；两者只在第 9.4 节的公共 host commit 中发布各自的 continuation state。它们不各自选择
GDN snapshot slot，也不各自推进 target committed frontier。

---

## 11. Prefix reuse

ReplaySSM 不改变 prefix 选择规则，只改变 GDN checkpoint 的 physical slot：

### 11.1 AppendAtFrontier

Retained sequence 的 current GDN state 已在 `slot=lane`，与 `execution_frontier` 对齐。Prefill suffix 直接在
该 slot 原地继续。

### 11.2 RestoreTurnCheckpoint

当 request plan 选择已保存的 turn checkpoint：

1. 恢复 Text/backend KV 的既有 checkpoint frontier；
2. copy GDN state `C+lane -> lane`；
3. 恢复 turn-checkpoint hidden；
4. 把 committed frontiers 设置到 checkpoint frontier；
5. 后续 suffix prefill 在 `lane` 原地继续。

Restore 后不交换 current/turn-checkpoint 的角色，source bytes 仍位于 `C+lane`。`KeepExisting` 保留
metadata 和 dedicated payload；`CaptureNew` 在旧 checkpoint 完成 restore 后使 metadata 失效，并仅在
完整 prefill 成功后发布新的 frontier。Replay Fold 始终只写 current slot。

### 11.3 Partial speculative terminal

若 stop 条件只接受 licensed batch 的前 \(m_b\) 个 outputs，Fold、KV trim、hidden correction 和 backend
flush 都提交相同的 `base_E+m_b`。因此 retained sequence 可以从这个真实终点继续 prefix reuse，不依赖
被拒绝的 target suffix。

---

## 12. 代码边界

### 12.1 Layout 与 state ownership

| 文件 | 修改 |
|---|---|
| `src/targets/qwen3_6/impl/runtime/linear_state_slots.h` | 定义 `2C` current/turn-checkpoint plane mapping，删除 snapshot-position helpers |
| `src/targets/qwen3_6/impl/runtime/layouts.h` | `PersistentLayout` 增加可选 `GdnReplayRecordLayout` |
| `src/targets/qwen3_6/impl/runtime/layouts_impl.h` | state pool 规划为 `2C`；spec backend 规划 exact-`C/T` records；workspace 按实际 GDN path 计算 |
| `src/targets/qwen3_6/impl/runtime/program.h` | Program 绑定 records；SequenceState 删除 base/capacity/current cursor；增加公共 speculative resolve helper |
| `src/targets/qwen3_6/impl/runtime/program_impl.h` | prefill slot lifecycle、graph representative、batch assembly 与 Fold commit transaction |
| `src/targets/qwen3_6/export/ninfer/targets/qwen3_6/runtime.h`、`src/targets/qwen3_6/impl/runtime/api_impl.h` | lane-only resolve 收窄为 final-prefill resolve；speculative B=1 继续使用 batch resolve |
| `src/runtime/engine/concurrent_executor.h` | final prefill 调用收窄后的 lane API；decode membership 以原始 row 顺序调用 batch resolve |

### 12.2 Round frame 与 schedule

| 文件 | 修改 |
|---|---|
| `src/targets/qwen3_6/export/ninfer/targets/qwen3_6/round_state.h` | 三种 ingress 统一 `lanes`；删除 read/snapshot selectors 和 prefill scalar selectors |
| `src/targets/qwen3_6/impl/state/round_state.cpp` | 绑定新的 ingress/round layout |
| `src/targets/qwen3_6/impl/runtime/schedule.h` | contexts 和 `TargetVerifyFrameView` 传递 lanes/records；prefill 传 current/turn-checkpoint slots |
| `src/targets/qwen3_6/impl/runtime/text_context.h` | 增加显式 GDN state action 和 Replay records binding |
| `src/targets/qwen3_6/impl/runtime/text_context_impl.h` | prefill 原地、ordinary width-1 原地、spec Record 三条路径 |
| `src/targets/qwen3_6/impl/runtime/text_prefill_impl.h` | 用 lane/current/turn-checkpoint slot 约定替换 group base/capacity |
| `src/targets/qwen3_6/impl/runtime/decode_impl.h` | ordinary decode 以统一 lanes 同时选择原地 GDN state 与 continuation hidden destination |
| `src/targets/qwen3_6/impl/runtime/speculative_target_impl.h` | 公共 target verify 切换到 Record |
| `src/targets/qwen3_6/impl/runtime/mtp_impl.h` | 删除 MTP snapshot selectors，传统一 lanes/records |
| `src/targets/qwen3_6/impl/runtime/dflash_impl.h` | 删除 DFlash snapshot selectors，传统一 lanes/records；保留 feature/context 语义 |

### 12.3 Exact targets

以下文件增加 record projection wrapper 和 workspace query：

- `src/targets/qwen3_6_27b/impl/variant.h/.cpp`；
- `src/targets/qwen3_6_35b_a3b/impl/variant.h/.cpp`。

Record/Fold Ops、`GdnReplayRecords` 和 all-layer state view 已经完成，不在集成阶段重新设计。

---

## 13. 实施顺序

1. 修改 `LinearStateSlots`、PersistentLayout 和 record binding，先建立最终 `2C + records` physical layout；
2. 统一 RoundState ingress 的 `lanes`，删除所有 snapshot selector storage；
3. 接通两个 exact Variant 的 record projection wrapper 和 workspace planning；
4. 改造 TextContext，使 speculative target verify 完整写入 all-layer records；
5. 让 MTP/DFlash 共用新的 target verify view，完成 eager 与 graph Record 路径；
6. 重构 Pending，使 speculative decode 在 resolve 前不推进 committed host frontiers；将 lane-only resolve
   收窄为 final-prefill resolve；
7. 实现一次 Fold、batched hidden correction、enqueue-only DFlash terminal flush 和一次同步组成的公共
   commit tail；
8. 切换 prefill/checkpoint/ordinary 到固定 current/turn-checkpoint slots，删除 SequenceState snapshot cursor；
9. 更新 graph representative preparation、memory accounting、tests 和 round benchmarks；
10. 更新并发架构文档中的 DecodeBatchFrame、speculative transaction 和 linear-state slot 描述。

每一步完成后都只保留最终合同，不保留 old/new runtime 双路径或 compatibility selector。

---

## 14. 正确性论证

### 14.1 Initial checkpoint identity

Record projection 和 recurrent Record 都使用 `initial_state_slots=lanes`。在 resolve 之前，同一个
`slot=lane` 没有被 speculative graph 修改。Fold row 的 destination 仍为该 lane。因此 verify 和 fold 从
逐 bit 相同的 conv/recurrent checkpoint 出发。

### 14.2 Transition input identity

Record 保存 verify recurrence 实际消费的 raw key/value/gate represented bits；conv record 保存 verify
causal convolution 使用的 BF16 projection column。Fold 读取同一 physical `(layer,b,t)` records。已有 Op
qualification 已验证 Record outputs 和 Fold state 对 snapshot trajectory 的 finite-precision clone。

### 14.3 Prefix identity

Output preview 返回的 `accepted_tokens=m_b` 同时用于：

- Fold `commit_columns`；
- Text KV logical frontier；
- ledger/prefix identity append；
- partial-terminal hidden selector `m_b-1`；
- terminal DFlash context flush count。

这些 consumers 由一次 `resolve_pending_batch` 构造，不存在独立的 GDN accepted-length 解释。

### 14.4 Destination identity

Record row 由 compact batch row决定，state destination 由 frozen `lanes[b]` 决定。Fold 的 public Op 固定
`rows[b] -> record row b`；Engine 不压缩 rows。于是 batch compaction 只改变本轮 row mapping，不改变
persistent state ownership。

### 14.5 Publication ordering

Fold、hidden correction 和必要的 DFlash flush 在同一 stream 完成并同步后，Program 才推进 committed host
frontier；executor 随后才调用 `OutputSession::commit_preview`。故任何已发布 token 都有与其一致的
committed target state，下一 round 也不会在旧 GDN checkpoint 上继续。

### 14.6 Record lifetime

Record producer、CPU preview 和 Fold 之间不调度可以覆盖 arena 的第二个 GPU unit。Fold 又在
`RoundMembership` 释放前完成并同步。因此 Fold 读取的仍是产生本轮 licensed outputs 的同一份
`(layer,b,t)` records，不需要 per-request record ownership。

---

## 15. 验证与测量

### 15.1 保留的 Op 数值证据

现有 ReplaySSM Op suites 继续负责：

- raw records exact bit copy；
- Record verify output 与 Snapshot output bit exact；
- Fold recurrent state 与 selected Snapshot state bit exact；
- Fold conv history exact；
- mixed commit lengths、non-contiguous absolute slots、`m=0` 和 rejected suffix；
- 完整 27B/35B-A3B geometries。

Engine 集成不建立一套旧 Program snapshot route 来重复这些数值测试。它验证 Op 已证明的 transition 被
正确映射和调度。

### 15.2 Runtime/layout tests

更新 family RoundState tests 和两个 exact-target layout/plan tests，验证：

- state pool slot count 对所有 backends 都是 `2*C`；
- ordinary backend 不含 record arena；
- MTP/DFlash record spec 精确等于配置的 `C` 和 `D+1`；
- current/turn-checkpoint slot mapping 和 copy/restore 语义。

### 15.3 Real Engine tests

在现有 27B MTP、35B-A3B MTP 和 35B-A3B DFlash real-artifact tests 上覆盖：

- graph 与 eager target verify；
- 多轮 continuing speculative decode；
- valid extent 为 1 的零草稿 speculative round，提交一列 Record；
- target-licensed batch 内的 partial terminal；
- terminal 后从精确 frontier 做 prefix reuse；
- turn checkpoint 保存、current state 继续推进、随后恢复；
- `B>1` compact batch 在某个 lane 完成后形成非连续 lanes，并让 surviving rows 具有不同 commit lengths。

最后一项直接保护 `record row b -> lanes[b]` 映射。Output、reused frontier 和后续 greedy continuation
必须与各 target 已有独立 baseline/oracle 要求一致。

### 15.4 Benchmark

现有 target round benchmarks 的主测量必须覆盖：

~~~text
decode graph/eager execution
  + CPU resolve preparation
  + Fold/commit tail
~~~

报告：

- target verify GPU latency；
- commit-tail GPU latency；
- target decode、OutputSession preview 和 `resolve_pending_batch` 的整轮 wall latency；
- licensed/committed tokens 与 batch size；
- Program sequence-memory bytes 的 old/new 对比。

因为 CPU output decision 位于两段 GPU work 之间，verify 和 commit-tail 分别用 CUDA events 测量；round
wall time覆盖完整事务。已有 `gdn_replay` Op benchmark 继续负责 Fold kernel 的独立性能，不在 target
benchmark 中重复 profiler 工作。

---

## 16. 完成后的稳定合同

集成完成后，Qwen3.6 runtime 保持以下不变量：

1. 每个 lane 的 current GDN state 永远位于 absolute slot `lane`；
2. 每个 lane 的 boundary GDN state 永远位于 absolute slot `C+lane`；
3. speculative target verify 不修改任何 persistent GDN state；
4. physical record row 等于本轮 compact batch row；
5. Fold destination 只来自 frozen `lanes[b]`；
6. Fold prefix length 等于 output policy 最终接受的 token 数；
7. MTP 与 DFlash 共享一次 all-layer Fold 和同一 target commit loop；
8. target execution/ledger frontier 只在 commit tail 成功后推进；
9. output 只在完整 state transaction 成功后发布；
10. state pool 容量固定为 `2*C`，与 speculative window 无关。
