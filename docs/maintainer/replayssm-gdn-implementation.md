# ReplaySSM GDN Op 实施方案

本文定义 ReplaySSM 的第一个实施阶段：冻结 record/state 存储约定，实现
replay-record 与 all-layer fold Ops，并在独立 test/benchmark 中完成数值和性能
验收。ReplaySSM 的数学定义见 [`replayssm-gdn.md`](replayssm-gdn.md)；本文只
约束实现。

实施顺序固定为：

~~~text
freeze storage ABI
        |
        v
implement record-producing Ops
        |
        v
implement one all-layer fold Op / kernel
        |
        v
qualify record -> fold against snapshot and an independent oracle
        |
        v
benchmark and tune the qualified kernels
~~~

这个阶段不改动 Program、decode schedule、CUDA Graph、slot lifecycle 或 speculative
backend。完成后得到一组已验证的存储合同和 Ops，下一阶段再根据它们与
当时的 NInfer runtime 制定接入计划。

---

## 1. 阶段产物

本阶段只交付六项内容：

1. 一份 fixed-capacity、all-layer ReplaySSM record layout 及 checked binder；
2. `LinearAttentionStatePool` 的 checked all-layer strided view，供 fold 按调用方给出的
   absolute slot 直接修改现有 state；
3. GDN input projection/causal-conv 的 record-producing form；
4. layer-local `gated_delta_net_replay_record` Op；
5. model-wide `gdn_replay_fold` Op，一次 kernel launch 更新全部指定 rows 和 layers；
6. 独立 correctness tests 与 benchmark。

现有 snapshot Ops 保留且不改变合同。它们是 record → fold 路径最直接的
finite-precision trajectory baseline。

---

## 2. 存储 ABI

### 2.1 符号与 represented values

| 符号 | 含义 |
|---|---|
| \(C\) | record row capacity，最大为 8 |
| \(B\) | 某次 replay-record 的实际 row 数，\(1\le B\le C\) |
| \(T\) | record 的 physical verify width；MTP 为 2–6，DFlash 为 2–16 |
| \(v_b\) | row \(b\) 本轮实际定义的 valid transition 数，\(1\le v_b\le T\) |
| \(m_b\) | row \(b\) 最终提交的 transition 数，\(0\le m_b\le v_b\) |
| \(L\) | GDN-state ordinal 数；\(\ell\in[0,L)\) 不是 transformer block index |
| \(H_q,H_v\) | q/k heads 和 value heads |
| \(K,V\) | key/value head dimension，当前都为 128 |
| \(C_p\) | causal-conv projection channels |
| \(S\) | supplied linear-attention state pool 的 absolute slot count |

`C` 和 `T` 都是 buffer construction-time facts。`C` 取 startup-fixed concurrency，允许
某轮只使用前 `B` rows；`T=draft_window+1`，是该 speculative verify profile 的 exact
physical width，并决定 record stride。MTP 的 draft window 为 1–5，DFlash 为 1–15，
所以 production ReplaySSM profile 不包含 `T=1`。Kernel 从 bound records 获得 runtime
`T`，并从 `valid_columns[b]` 获得该 row 的 \(v_b\)；`T` 不是 template argument。Records
backing 不在 forward 之间重新分配或改变 strides。

每个 token、layer 和 row 的 record 包含：

| 字段 | DType | 语义 |
|---|---|---|
| `conv` | BF16 | 写入 causal-conv history newest column 的 represented projection column |
| `key` | BF16 | recurrence 实际消费的 normalization-before raw key |
| `value` | BF16 | recurrence 实际消费的 raw value |
| `g` | FP32 | verify 产生的 log-decay gate |
| `beta` | FP32 | verify 产生的 correction gate |

`q` 只用于当轮 output readout，不进入 record。Record 是上述 represented values
的 lossless side copy，不从 hidden state 或 weights 重算。

### 2.2 Record arena 的 plane 布局

一份 `GdnReplayRecords` 由一段 caller-owned device backing 绑定。Backing 内有四个
256-byte-aligned regions：

| Region | DType | Tensor shape |
|---|---|---|
| `conv` | BF16 | `[Cp, T, C * L]` |
| `key` | BF16 | `[K, Hq, T, C * L]` |
| `value` | BF16 | `[V, Hv, T, C * L]` |
| `gate` | FP32 | `[2, Hv, T, C * L]` |

Tensor 的 dimension 0 最连续。Layer 和 record row 合并到最外层：

\[
r(\ell,b)=b+C\ell.
\]

各 plane 的 element offset 是：

\[
\begin{aligned}
o_{conv}(c,t,\ell,b)
  &=c+C_p\left(t+T\,r(\ell,b)\right),\\
o_{key}(d,h,t,\ell,b)
  &=d+K\left(h+H_q\left(t+T\,r(\ell,b)\right)\right),\\
o_{value}(d,h,t,\ell,b)
  &=d+V\left(h+H_v\left(t+T\,r(\ell,b)\right)\right),\\
o_{gate}(a,h,t,\ell,b)
  &=a+2\left(h+H_v\left(t+T\,r(\ell,b)\right)\right),
\end{aligned}
\]

其中 `gate[0,...]=g`，`gate[1,...]=beta`。Gate pair 在内存中相邻，Record 和
Fold 都使用一次 8-byte `uint2` bit-preserving store/load。

两个 registered `Cp`、`K*Hq` 和 `V*Hv` 的 BF16 row extents 都是 256 bytes 的
倍数，因此每个 layer/record-row 的 conv/key/value base 仍为 256-byte aligned，
`conv_record` 可直接作为 projection destination。Gate region 只要求每个 `{g,beta}`
pair 8-byte aligned；planner 和 wrapper 分别检查这两类 alignment。

这一顺序同时满足：

- 每层 verify 可直接取得 `[... ,T,B]` 的连续 slice；
- 单个 row/layer 的短 token sequence 连续，fold 可顺序消费；
- `key` 按 q/k head 存储一次，不为 grouped value heads 重复；
- all-layer fold 可从 `layer * C + b` 直接寻址；
- 不需要 device pointer table 或 per-layer allocation。

只有 `[0,B)` rows 和每行的 `[0,v_b)` 由本次 verify 定义。其他
rows/tails 无需初始化，fold 也不得读取它们。

### 2.3 Existing state pool 的 all-layer view

`LinearAttentionStatePool` 仍是 linear-attention state 的唯一 physical owner。Record →
fold 不创建第二套 state storage，也不为了 fold 重排现有 backing。Pool 中每层已有：

| State | DType | Layer-local tensor shape |
|---|---|---|
| causal-conv history | BF16 | `[Cp, 3, S]` |
| recurrent state | FP32 | `[K, V, Hv, S]` |

各层 geometry 相同，因此 conv layer bases 之间、recurrent layer bases 之间分别具有固定
byte stride。Pool 导出一个 checked non-owning view：

~~~cpp
struct LinearAttentionStateAllLayersView {
    Tensor conv_layer0;       // BF16 [Cp,3,S]
    Tensor recurrent_layer0;  // FP32 [K,V,Hv,S]
    std::int64_t conv_layer_stride_bytes;
    std::int64_t recurrent_layer_stride_bytes;
    LinearAttentionStatePoolSpec spec;
};
~~~

`spec.slot_count` 定义独立于 record capacity 的 \(S\)。一个绝对 linear-state slot
\(s\in[0,S)\) 同时选择全部 GDN layers 的 causal-conv history 和 recurrent state。Slot
不是 batch row；Op 也不把它解释为 lane、request-local offset 或 snapshot position。Fold
的 state addresses 固定为：

\[
\begin{aligned}
a_{conv}(c,w,\ell,s)
  &=base_{conv}+\ell\,stride_{conv}
    +2\left(c+C_p\left(w+3s\right)\right),\\
a_{state}(d_k,d_v,h,\ell,s)
  &=base_{state}+\ell\,stride_{state}
    +4\left(d_k+K\left(d_v+V\left(h+H_v\,s\right)\right)\right).
\end{aligned}
\]

`LinearAttentionStatePool::all_layers_view()` 从已绑定的 per-layer regions 验证并返回该
view；fold 不接收 device pointer table 或测试专用 state tensor。State pool 没有
verify-position 维度；它只暴露已经存在的 physical slots。`S` 的 production 数量、各 slot
承担的 lifecycle role，以及 runtime 如何为 batch row 选择 slot，都属于后续集成设计，
不是本阶段 storage 或 Op 合同的一部分。

本阶段只固定一条跨 Op 不变量：对于 row `b`，两个 record-producing Ops 的
`initial_state_slots[b]` 与 Fold control 中的 `linear_state_slot` 必须是同一个 absolute
slot。该值由调用方提供，Record/Fold 不从 batch row 推导，也不增加任何 offset。

### 2.4 Checked layout 与 binder

新增 target-neutral layout 和一个绑定后的 non-owning records object：

~~~cpp
struct GdnReplayRecordSpec {
    std::int32_t layers;
    std::int32_t record_capacity;
    std::int32_t width;
    std::int32_t conv_channels;
    std::int32_t qk_heads;
    std::int32_t value_heads;
    std::int32_t key_dim;
    std::int32_t value_dim;
};

struct GdnReplayRecordLayout {
    GdnReplayRecordSpec spec;
    TensorRegion conv;
    TensorRegion key;
    TensorRegion value;
    TensorRegion gate;
};

struct GdnReplayRecordLayer {
    Tensor conv;   // [Cp,T,B]
    Tensor key;    // [K,Hq,T,B]
    Tensor value;  // [V,Hv,T,B]
    Tensor gate;   // [2,Hv,T,B]
};

struct GdnReplayRecords {
    Tensor conv;
    Tensor key;
    Tensor value;
    Tensor gate;
    GdnReplayRecordSpec spec;

    [[nodiscard]] GdnReplayRecordLayer layer(
        std::int32_t layer, std::int32_t rows) const;
};
~~~

位置固定为 `src/core/gdn_replay_records.h/.cpp`。Planner 检查正整数、四维
Tensor 上限、byte-size overflow 和 256-byte region alignment；binder 只绑定 caller-owned
backing，不分配显存。

`GdnReplayRecords::layer(layer, rows)` 要求 `layer` 位于 `[0,L)`、`rows` 位于 `[1,C]`，
并返回：

- `conv [Cp,T,B]`；
- `key [K,Hq,T,B]`；
- `value [V,Hv,T,B]`；
- `gate [2,Hv,T,B]`。

这里 `B=rows`，返回的是该 layer 的 record rows `[0,B)`。Slice 只改变 tensor view，不
产生 copy，也不在 records object 中保存本轮 active row 数。All-layer Fold 直接接收完整的
capacity-sized `GdnReplayRecords`，并以 `rows.size()` 唯一定义本轮的 `B`。

### 2.5 容量

每个 layer/row/token 的字节数为

\[
P=2C_p+2KH_q+2VH_v+8H_v.
\]

因此 record arena 容量是

\[
M_{record}=LCTP,
\]

外加四个 region 的 alignment padding。`C=8` 时的目标尺寸为：

| Geometry | `T` | Capacity |
|---|---:|---:|
| `L=48,Hq=16,Hv=48,Cp=10240` | 6 | 81.84375 MiB |
| `L=30,Hq=16,Hv=32,Cp=8192` | 6 | 39.7265625 MiB |
| `L=30,Hq=16,Hv=32,Cp=8192` | 16 | 105.9375 MiB |

---

## 3. Op 合同

本阶段新增两个 ReplaySSM recurrent Ops，并为现有 GDN input-projection Op 增加
一个 record-producing form。所有 Ops 只消费 caller-owned tensors/workspace，不拥有
persistent allocation。

`include/ninfer/ops/` 中的 contract comments 写完整数学/索引转换、represented
inputs、semantic cast boundaries、effects、aliasing 和 workspace。第 4 节的 grid、template
mode、writer mapping 和 private arithmetic sequence 只属于实现，不进入 Op 合同或
`replayssm-gdn.md` 的数学公式。

### 3.1 `gdn_input_proj_conv_record`

该 form 加入 `include/ninfer/ops/gdn_input_proj.h`，为现有 two-parent 和
single-parent registered profiles 提供与 snapshot form 对称的 overloads。
Canonical two-parent contract 是：

~~~cpp
void gdn_input_proj_conv_record(
    const Tensor& x,
    const Weight& qk_weight,
    const Weight& value_z_weight,
    const Tensor& conv_weight,
    const Tensor& conv_states,
    const Tensor& valid_columns,
    const Tensor& initial_state_slots,
    Tensor& conv_record,
    Tensor& query,
    Tensor& key,
    Tensor& value,
    Tensor& z,
    WorkspaceArena& workspace,
    cudaStream_t stream);
~~~

Single-parent form 保持现有 `LinearPolicy` overload 和 A16 convenience overload。另增
`gdn_input_proj_conv_record_workspace_capacity_bytes` 的 two-parent/single-parent queries；
它们复用同一 route planner，但不让新 Op 依赖名称带 `snapshot` 的 capacity API。
本 form 的 ReplaySSM 执行域固定为 `B=1..8`、`T=2..16`。`valid_columns` 为空时
每行 \(v_b=T\)；否则它是 device I32 `[B]`，调用方保证每个 \(v_b\) 位于 `[1,T]`。

`conv_states` 是某一 GDN layer 的 supplied state-pool view `[Cp,3,S]`。
`initial_state_slots` 是 device I32 `[B]`；调用方保证每个值是第 2.3 节定义的
`[0,S)` absolute slot。Op 按该值直接寻址，不把它解释为 snapshot base，也不增加
per-request offset。

对每个 batch row，它：

1. 从 `initial_state_slots[b]` 读取 BF16 three-column conv history；
2. 执行与现有 snapshot form 相同的 projection、causal convolution、SiLU 和
   q/k/value/z split；
3. 对每个 valid column 写一列 BF16 `conv_record [Cp,T,B]`；
4. 保持输入 conv state 完全不变。

`conv_record[:,t,b]` 等于 snapshot form 在该 position 写入 newest history column 的
BF16 represented value。该 cast/storage boundary 是 record contract 的一部分。本 Op 内部仍然使用
各 projection route 现有的 private accumulator/staging path 继续下一列，record store 不得
成为新的反馈 cast boundary。

对已经以 BF16 materialization 为数值边界的 staged/composed routes，
`conv_record` 本身就是 projection 的 q/k/value destination；后续 causal-conv 直接
只读这个 plane。这些 routes 不再分配另一张 projected tensor，也不执行
projected → record copy。只有 projection 与 convolution 融合、且 convolution 消费未物化
FP32 accumulator 的 routes，才在 fused epilogue 中 side-copy `BF16_RNE(p)` 到 record。

`valid_columns` 与 snapshot form 保持相同语义：query/key/value 的 invalid tail 写 exact
zero，`z` 仍对全部 safe input columns 完成 projection。只有 valid prefix 的 conv
record 具有语义；invalid tail 可能被 materialized projection 写入，但 fold 不得读取。
Source conv state 的全部 bytes 保持不变。对于相同的 weight profile、`B/T` 和 compute
policy，Record 与 Snapshot 使用该 profile 的同一份 projection/conv route decision，因而
选择相同的 projection arithmetic。Q4/Q5、W8 和 NVFP4 各自保留独立 planner，不引入
跨格式的统一 route abstraction。Record workspace query 只计入所选 route 不可避免的
private scratch；
`conv_record` 是 caller-owned output，不计入 workspace。Inputs、outputs、conv record、conv state
和 live workspace 不得互相 overlap。

Wrapper 验证 tensor dtype、shape、contiguity、alignment、`B/T/S` 的 host-visible geometry
以及 aliasing。它不回读 `valid_columns` 或 `initial_state_slots`，因此其中每个 \(v_b\) 和
absolute slot 的取值范围属于上述调用前置条件。

### 3.2 `gated_delta_net_replay_record`

该 Op 加入 `include/ninfer/ops/gated_delta_net.h`。其 layer-local 合同为：

| Tensor | DType / shape |
|---|---|
| `q`, `k` | BF16 `[128,Hq,T,B]` |
| `v`, `out` | BF16 `[128,Hv,T,B]` |
| `g`, `beta` | FP32 `[Hv,T,B]` |
| `states` | FP32 `[128,128,Hv,S]` |
| `valid_columns` | empty or I32 `[B]` |
| `initial_state_slots` | I32 `[B]` |
| `key_record` | BF16 `[128,Hq,T,B]` |
| `value_record` | BF16 `[128,Hv,T,B]` |
| `gate_record` | FP32 `[2,Hv,T,B]` |

Public contract 是：

~~~cpp
void gated_delta_net_replay_record(
    const Tensor& q,
    const Tensor& k,
    const Tensor& v,
    const Tensor& g,
    const Tensor& beta,
    float scale,
    const Tensor& ssm_states,
    const Tensor& valid_columns,
    const Tensor& initial_state_slots,
    Tensor& key_record,
    Tensor& value_record,
    Tensor& gate_record,
    Tensor& out,
    cudaStream_t stream);
~~~

语义效果是：

- 从每行 selected initial state 顺序生成 GDN outputs；
- 在 valid prefix 中保存 raw `k/v/g/beta` represented bits；
- invalid output tail 写 exact BF16 zero，record tail 不变；
- `states` 全部只读，任何 slot 都不变。

`initial_state_slots[b]` 与 3.1 节使用同一个绝对 `linear_state_slot[b]`，因此 conv record
和 recurrent record 都从 batch row `b` 的同一 committed checkpoint 出发。后续 Fold 的
`rows[b].linear_state_slot` 必须是同一个值。与 3.1 节相同，调用方保证 device selector
位于 `[0,S)`；wrapper 不回读其元素。

`scale` 和 head grouping 与现有 batched snapshot contract 对齐。ReplaySSM profiles 固定
执行当前 targets 使用的 raw q/k normalization，不暴露可与 fold 配错的 runtime 开关。
本 Op 只注册 `B=1..8`、`T=2..16` 的 ReplaySSM 执行域。`valid_columns` 为空时
\(v_b=T\)；否则调用方保证 device tensor 中每个 \(v_b\) 位于 `[1,T]`。Record tensors 是
`GdnReplayRecords::layer(layer,B)` 返回值中除 `conv` 外的三个 slices；wrapper 检查它们与
inputs 的 B/T/head geometry 完全一致，但不回读 valid/slot selectors。该 Op 不需要
transient workspace。
Inputs、out、三个 record slices 和 state 必须 pairwise non-overlapping。

### 3.3 `gdn_replay_fold`

该 Op 拥有独立合同 `include/ninfer/ops/gdn_replay.h`。它一次调用处理本轮全部 active
batch rows 和全部 GDN layers。

每个 batch row 只提供其绝对 linear-state slot 和提交长度：

~~~cpp
struct GdnReplayFoldRow {
    std::int32_t linear_state_slot;
    std::int32_t commit_columns;
};
~~~

Public contract 是：

~~~cpp
void gdn_replay_fold(
    const GdnReplayRecords& records,
    LinearAttentionStateAllLayersView states,
    std::span<const GdnReplayFoldRow> rows,
    cudaStream_t stream);
~~~

- `B=rows.size()`，且 `1<=B<=records.spec.record_capacity`；
- `rows[b]` 固定对应 replay-record 的 batch row `b`，不允许选择或重排 record rows；
- `linear_state_slot` 是第 2.3 节定义的 `[0,S)` absolute pool slot，且必须等于两个
  record-producing Ops 为 row `b` 使用的 `initial_state_slots[b]`；
- `commit_columns` 是 \(m_b\)，即要顺序重放的 `[0,T]` prefix length。

目标验证先产生 licensed prefix \(p_b=accepted\_drafts_b+1\)。`commit_columns`
使用最终输出边界在 CPU 上解析出的 \(m_b\)，并满足
\(0\le m_b\le p_b\le v_b\)。只有最终边界保留全部 licensed outputs 时，才有
\(m_b=p_b=accepted\_drafts_b+1\)；在这种情况下，没有 draft 被接受时 \(m_b=1\)。独立地，
`commit_columns=0` 是正式的 no-op control：该 row 保持原 batch 索引，不被筛选、压缩或
重排，Fold 不读取它的 records，也不修改它的 recurrent state 或 conv history。

Wrapper 接收 1–8 个 host rows，并独立取得 `C=records.spec.record_capacity` 与
`S=states.spec.slot_count`。它检查 `B<=C`、`linear_state_slot` 位于 `[0,S)` 且互不重复，
以及每个 `commit_columns` 位于 `[0,T]`；同时从两个 checked objects 验证 dtype、
contiguity、layer count 和完整 geometry。`S` 与 `C` 不要求相等。Fold 对 raw key 固定执行
与 replay-record 相同的 normalization；
fold 不读 query，因此没有 query normalization 或 output scale。Row controls 按原索引复制到
fixed-size POD kernel argument，不分配 device metadata，也不需要 H2D metadata kernel。

调用方保证 \(m_b\le v_b\)，也就是 `rows[b].commit_columns` 不超过 row `b` 在
replay-record 时使用的 valid extent。Fold 没有保留 `valid_columns`，也不为验证这个跨 Op
不变量而回读 device metadata。

对每个 batch row `b` 和 layer，`m_b=0` 时 fold 不读取 records 或 state，也不产生任何
state store。`m_b>0` 时 fold：

1. 按第 2.3 节的 recurrent layer/slot address 加载 FP32 state；
2. 顺序消费 `r(layer,b)` 的前 `rows[b].commit_columns` 个
   `key/value/gate` records；
3. 把 final recurrent state 写回同一 `linear_state_slot`；
4. 用 `conv` record 把 three-column history 更新为
   `tail_3(old_history || conv_record[0:m_b])`。

Rejected tail 不得被读取。State input/output 完全 alias 是正式合同：每个 recurrent tile
在写回前全部位于 registers，conv history 也先加载需要的 columns 再原地写回。

Fold 不生成 token output，不读 q，不使用 workspace，不拥有任何 runtime commit
语义。它只实现由显式 tensors 和 row controls 完全决定的 state transformation。
四个 record planes 只读且彼此不 overlap，也不得与两个 state regions overlap。

### 3.4 Supported profiles

首个实现只为当前两个 geometry profiles 注册 all-layer fold：

| \(L\) | \(H_q\) | \(H_v\) | \(K/V\) | \(C_p\) |
|---:|---:|---:|---:|---:|
| 48 | 16 | 48 | 128/128 | 10,240 |
| 30 | 16 | 32 | 128/128 | 8,192 |

Record capacity 为 1–8，production ReplaySSM `T` 为 2–16，active row count 为 1–8。
MTP 只使用 `T=2..6`，DFlash 使用 `T=2..16`。Wrapper 按 geometry
选择编译期 launcher，kernel 内不按 model identity 分支。
`T<=16` 是 Replay record buffer 的 physical capacity bound，不是某个 private kernel
route 的性能阈值；单行 `valid` 可为 1，`commit_columns` 可为 0。

---

## 4. Kernel 改造

### 4.1 最终代码边界

Kernel 改造按现有代码的真实边界进行：

| 位置 | 改造结果 |
|---|---|
| `src/ops/linear_attention/gated_delta_net/recurrent.cuh` | 提取唯一 BF16 key-normalization、transition 和 readout helpers；定义 `RecurrentMode` 与 mode-specific accessors |
| `src/ops/linear_attention/gated_delta_net/recurrent.cu` | 保留现有 runtime-`T` general/direct/Snapshot launchers，并在同一 CUDA translation unit 增加 runtime-`T` Record 和 Fold entries |
| `src/ops/linear_attention/gated_delta_net/launch.h` | 声明 replay-record/fold private launchers 和有限 exact-geometry dispatch |
| `src/ops/gdn_input_proj/gdn_conv.cuh` | 取代 `gdn_conv_snapshot.cuh`；融合 projection routes 的 history/record publish sink 为编译期参数 |
| `src/ops/gdn_input_proj/gdn_projected_conv.cu/.h` | 归属 GDN-private materialized projected-conv launcher/kernel；直接 split 写 q/k/value，并以编译期参数控制 history publish |
| Q4/Q5、W8、NVFP4 GDN snapshot files | 各自复用原 projection 与 post-conv schedules，将 snapshot-only output object 中性化为同一 profile 的 Snapshot/Record forms 共用 |
| `src/CMakeLists.txt` | 纳入 record container 和 public wrappers；recurrent CUDA 实现继续由现有 `recurrent.cu` 唯一拥有 |

通用 `causal_conv1d` Op 保持现有合同与实现边界。q/k/value split、absolute-slot selectors
和 Snapshot/Record history publication 属于 `gdn_input_proj_conv_*` 的私有语义，不进入
`src/ops/kernel/causal_conv1d.cuh` 或它的 public launcher。

Public validation 分别留在 `src/ops/linear_attention/gated_delta_net/gated_delta_net.cpp`、
`src/ops/wrapper/gdn_input_proj.cpp` 和新增的
`src/ops/linear_attention/gated_delta_net/replay.cpp`。这些 wrappers 只验证合同并选择有限 launcher；任何 tensor allocation、
layout 推断和 runtime policy 都不进入 CUDA kernel。

### 4.2 提取 recurrent 数值主体

现有 BF16 recurrent block 的物理分工保持不变：

- `block=(32,4)`，一个 block 属于一个 value head 和一个 16-row state tile；
- `lane` 持有四个连续 key dimensions；
- 每个 warp 持有四个 value rows；
- 八个 state tiles 覆盖 128 个 value rows；
- 每个 thread 的 resident state 是 `float state[4][4]`。

首先把 `recurrent_bf16_kernel` 内的以下代码原样提取为 force-inlined helpers：

~~~cpp
struct RawQkLane {
    Bf16x4Pack bits;
    float value[4];
};

struct RawValueLane {
    __nv_bfloat16 bits;
    float value;
};

struct RawGatePair {
    uint2 bits;
    float g;
    float beta;
};

__device__ __forceinline__ RawQkLane load_raw_qk_lane(
    const __nv_bfloat16* base, int dqk_base);

template <bool Normalize>
__device__ __forceinline__ void normalize_qk_lane(
    float (&value)[4], int lane);

__device__ __forceinline__ RawValueLane load_value_lane(
    const __nv_bfloat16* base, int lane, int dv_base);

__device__ __forceinline__ void apply_gdn_transition(
    float (&state)[4][4],
    const float (&key)[4],
    float v_local,
    float g,
    float beta);
~~~

`load_raw_qk_lane` 保留当前 `load_vec<Bf16x4Pack>` 和 BF16→FP32 conversion 顺序；
`normalize_qk_lane` 保留 lane-local `sum += x*x`、warp reduction、lane-0
`rsqrtf(sum+1e-6)`、shuffle 和 multiply 顺序。Record mode 在两个 helpers 之间
side-copy raw pack；Fold 从 record 加载相同 pack 后走同样的 conversion 和
normalization helper。`load_value_lane` 对 `lane<4` 同时返回一个 raw BF16 和它的
FP32 conversion，其余 lanes 返回 zero；replay-record 直接使用返回的 raw field。

`apply_gdn_transition` 保留当前 kernel 的 code-shaped evaluation sequence：

~~~cpp
const float alpha = expf(g);

#pragma unroll
for (int r = 0; r < 4; ++r) {
    float partial = 0.0f;
    #pragma unroll
    for (int c = 0; c < 4; ++c)
        partial += state[r][c] * key[c];
    partial = warp_sum(partial);

    const float v_r = __shfl_sync(0xffffffff, v_local, r, 32);
    const float delta = beta * (v_r - alpha * partial);
    #pragma unroll
    for (int c = 0; c < 4; ++c)
        state[r][c] = alpha * state[r][c] + delta * key[c];
}
~~~

这里不引入 decayed-state temporary、不改成另一种矩阵表达式，也不手写新的 FMA/reduction。
Helper 使用 `__forceinline__`，最终每个 mode 中仍生成展开后的原有 tile arithmetic。数学资料
和 naive oracle 保持其独立数学定义，不随这段 production source 改写。

Query normalization 继续调用与 key 相同的 raw loader 和 normalization helper；readout 的四项 lane-local dot、
`warp_sum`、lane-to-row selection、scale 和 BF16 cast 提取为另一个 force-inlined helper。
Transition helper 不包含 q，因此 Fold mode 不会因共享代码而携带 readout work。

### 4.3 一个 recurrent body，三个 entry kernels

Snapshot/Record 是 layer-local grid，Fold 是 all-layer grid；它们不共享一个臃肿的
global signature。三者共享唯一的 force-inlined recurrent body。`T` 是 tensor/view
提供的运行时 scalar，row 的 `valid` 或 `commit_columns` 也是运行时值；二者都不是
template argument：

~~~cpp
enum class RecurrentMode {
    Snapshot,
    Record,
    Fold,
};

template <RecurrentMode Mode,
          bool NormalizeInputs,
          class Access,
          class Coordinates>
__device__ __forceinline__ void recurrent_bf16_body(
    const Access& access,
    const Coordinates& coord,
    std::int32_t width,
    std::int32_t valid);

template <bool NormalizeInputs, bool Batched, bool Masked>
__global__ void recurrent_snapshot_kernel(
    SnapshotAccess<Batched, Masked> access) {
    const auto coord = access.coordinates();
    recurrent_bf16_body<RecurrentMode::Snapshot, NormalizeInputs>(
        access, coord, access.width, access.active_columns(coord));
}

template <bool Masked>
__global__ void recurrent_record_kernel(RecordAccess<Masked> access) {
    const auto coord = access.coordinates();
    recurrent_bf16_body<RecurrentMode::Record, true>(
        access, coord, access.width, access.active_columns(coord));
}

template <class Geometry>
__global__ void recurrent_fold_kernel(
    const __grid_constant__ FoldAccess<Geometry> access) {
    const auto coord = access.coordinates();
    recurrent_bf16_body<RecurrentMode::Fold, true>(
        access, coord, access.width, access.active_columns(coord));
}
~~~

三个 `Access` values 各自只包含该 entry 所需的 pointers、strides 和 scalar arguments：

| Mode | Kernel arguments 中存在的 storage |
|---|---|
| `Snapshot` | q/k/v/g/beta、state、valid/initial/snapshot selectors、out |
| `Record` | q/k/v/g/beta、只读 state、valid/initial selectors、key/value/gate records、out |
| `Fold` | key/value/gate/conv records、recurrent/conv state layer-0 bases 与 layer strides、packed rows |

对应的 raw kernel arguments 至少固定为：

~~~cpp
template <bool Batched, bool Masked>
struct SnapshotAccess {
    const __nv_bfloat16 *q, *k, *v;
    const float *g, *beta;
    float* states;
    const std::int32_t *valid_columns, *initial_slots, *snapshot_bases;
    __nv_bfloat16* out;
    head_map heads;
    std::int32_t width;
    std::int64_t linear_state_slot_stride;
    float scale;
};

template <bool Masked>
struct RecordAccess {
    const __nv_bfloat16 *q, *k, *v;
    const float *g, *beta;
    const float* states;
    const std::int32_t *valid_columns, *initial_slots;
    __nv_bfloat16 *key_record, *value_record;
    std::uint32_t* gate_record_bits;
    __nv_bfloat16* out;
    head_map heads;
    std::int32_t width;
    std::int64_t linear_state_slot_stride;
    float scale;
};

struct alignas(8) GdnReplayFoldKernelRow {
    std::int32_t linear_state_slot;
    std::int32_t commit_columns;
};

struct alignas(16) GdnReplayFoldKernelRows {
    GdnReplayFoldKernelRow row[8];
};

template <class Geometry>
struct FoldAccess {
    const __nv_bfloat16 *key_record, *value_record, *conv_record;
    const std::uint32_t* gate_record_bits;
    float* recurrent_layer0;
    __nv_bfloat16* conv_layer0;
    std::int64_t recurrent_layer_stride;
    std::int64_t conv_layer_stride;
    std::int32_t record_capacity;
    std::int32_t width;
    GdnReplayFoldKernelRows rows;
};
~~~

`width` 由 wrapper 在完成 tensor/view shape 验证后传入。`Masked=false` 的 access
specialization 省略或完全不访问 `valid_columns`；Fold geometry
在类型中固定 `L/Hq/Hv/Cp`，其 argument 中不重复这些 scalars。Selectors 与 valid columns
仍是 device pointers；Fold rows 是 host 已解析、保持 batch-row 顺序的 by-value kernel
parameter。Fold entry 把包含 64-byte rows 的 aggregate 标为 `const __grid_constant__`，
shared body 只接收 const reference；row control 从 parameter/constant path 按需加载，不生成
per-thread aggregate copy。

`Access` 不是 runtime interface。它是一个 trivially-copyable POD，只提供下列
`__forceinline__` 地址函数：

~~~cpp
coordinates();                     // row/layer/head/state-tile
active_columns(coordinates);       // T、valid_columns[row] 或 commit_columns
state_read_ptr(coordinates, r, lane4);
key_ptr(coordinates, token, qk_head);
value_ptr(coordinates, token, value_head);
load_gate(coordinates, token);     // {raw uint2 bits, float g, float beta}

// 仅在相应 Mode 的 if constexpr 分支中存在
query_ptr(...);
store_key_record(...);
store_value_record(...);
store_gate_record(...);
store_snapshot_state(...);
store_final_state(...);
~~~

`SnapshotAccess` 和 `RecordAccess` 把 `blockIdx.y` 解码为 batch row；
`FoldAccess` 同样直接把 `blockIdx.y` 作为 batch row，再由 `blockIdx.z` 解码 layer
和 state tile。`key_ptr/value_ptr/load_gate` 因此可以有不同的 physical address
formula，但返回给 token body 的都是同样的 represented bits。这些 methods 中
不允许有 normalization、`expf`、dot product 或 state arithmetic。

所有 mode 共用 body 中从 state load 到 token loop 的同一段 source。Mode-specific
addressing 由 force-inlined accessors 完成；side effects 只位于 `if constexpr` 分支：

| Mode | q/readout | Raw record publish | Per-token state publish | Final state publish |
|---|---:|---:|---:|---:|
| `Snapshot` | yes | no | yes | no |
| `Record` | yes | yes | no | no |
| `Fold` | no | no | no | yes, when `commit_columns>0` |

`NormalizeInputs` 在 Snapshot/Record 中同时控制 q/k，在 Fold 中只控制 key。
Fold specialization 的 argument type 没有 q/out/scale fields；对应 q load、q normalization、
readout 和 stores 会在编译期完全消失。Record specialization 的 state pointer 是
`const float*`，且不存在任何 state output field，因此不能生成 state store。

`NormalizeInputs=false` 只为现有 Snapshot/direct-recurrent contracts 保留。Record 和
Fold entry templates 本身不接受 normalization 参数，它们在调用 shared body 时直接固定
`NormalizeInputs=true`，从类型层面排除两者选到不同 path。

编译期只固定会改变生成代码结构的有限选择：`Mode`、normalization、Snapshot/Record
的 batched/masked addressing，以及 Fold 的 registered geometry。Recurrent 和 GDN-private
materialized projected-conv 使用 runtime `T`。既有 fused projection route 若以 exact token
count 决定 MMA/epilogue schedule，则继续保留该 route 私有的 exact-`T` specialization。

### 4.4 Shared token loop 的准确顺序

Shared body 按下面的顺序组织：

~~~text
decode mode-specific row/layer/state-tile
Fold only: if commit_columns == 0, return without reading records or state
load selected FP32 state tile into registers
load raw key[0] once; optionally publish its raw BF16 pack; normalize it

for t in [0, valid):
    load raw g/beta bits
    load the BF16 value elements already owned by this tile
    Record only: publish raw value and gate bits

    apply_gdn_transition(state, key, raw_value, g, beta)

    if t + 1 < valid:
        load raw key[t+1]
        Record only: publish its raw BF16 pack
        normalize next key

    Snapshot/Record only:
        load and normalize q[t]
        execute existing readout helper
        store BF16 out[t]

    Snapshot only:
        publish the resident FP32 state tile to snapshot slot[t]

Snapshot/Record only:
    write exact BF16 zero to out[valid:T]

Fold only:
    store the final resident FP32 state tile once
    selected blocks publish final conv history
~~~

相应的 code-shaped skeleton 是：

~~~cpp
if constexpr (Mode == RecurrentMode::Fold) {
    if (valid == 0)
        return;
}

float state[4][4];
load_state_tile(access, coord, state);

RawQkLane key = load_raw_qk_lane(access.key_ptr(coord, 0), dqk_base);
if constexpr (Mode == RecurrentMode::Record)
    publish_key_once(access, coord, 0, key.bits);
normalize_qk_lane<NormalizeInputs>(key.value, lane);

for (std::int32_t token = 0; token < valid; ++token) {
    const RawGatePair gate = access.load_gate(coord, token);
    const RawValueLane value =
        load_value_lane(access.value_ptr(coord, token), lane, dv_base);
    if constexpr (Mode == RecurrentMode::Record) {
        publish_value_once(access, coord, token, value.bits);
        publish_gate_once(access, coord, token, gate.bits);
    }

    apply_gdn_transition(state, key.value, value.value, gate.g, gate.beta);

    if (token + 1 < valid) {
        key = load_raw_qk_lane(access.key_ptr(coord, token + 1), dqk_base);
        if constexpr (Mode == RecurrentMode::Record)
            publish_key_once(access, coord, token + 1, key.bits);
        normalize_qk_lane<NormalizeInputs>(key.value, lane);
    }

    if constexpr (Mode != RecurrentMode::Fold)
        readout_and_store(access, coord, token, state);
    if constexpr (Mode == RecurrentMode::Snapshot)
        store_snapshot_tile(access, coord, token, state);
}

if constexpr (Mode == RecurrentMode::Fold) {
    store_final_tile(access, coord, state);
    publish_final_conv_history(access, coord);
} else {
    zero_invalid_output_tail(access, coord, valid, width);
}
~~~

Snapshot/Record 的前置条件是 `0 < valid <= width`，其 masked `valid_columns` 位于 device
memory，元素范围由调用方保证。Fold 的前置条件是 `0 <= valid <= width`；`valid==0` 的
CTA-uniform 分支在任何 state 或 record address 被解引用前直接返回，因此该 row 没有 load、
store 或 conv-history publish。其余路径进入 token loop 前均满足 `valid>0`，所以读取
`key[0]` 是定义良好的。循环上界是 runtime `valid`，不做 exact-`T` 展开，也不需要在
循环中再次判断 rejected tail。

`RawGatePair` 在 Record 中分别从 `g` 和 `beta` 读取 FP32 bits，并在
unique-writer 分支中把它们组成 `uint2`；Fold 直接从 interleaved gate plane 读
`uint2` 并 bit-cast 回两个 FP32 registers。`apply_gdn_transition` 只接收这两个
registers，因此两种 physical load 不会泄漏进 transition arithmetic。

Next-key load 保持在 transition 之后、q readout 之前，与当前 recurrent kernel 一致。
Snapshot store 仍在 readout store 之后。Record stores 只复制已经为本次 transition 加载的
values，不插入第二次 normalization、gate computation 或 value conversion。

### 4.5 Replay-record 的唯一 writer 映射

设 `group=Hv/Hq`、`state_tile` 为当前 16-row tile。Record writers 固定如下：

| Field | Writer | 一次写入 |
|---|---|---|
| key | `state_tile==0 && warp==0 && h_v%group==0` 的全部 32 lanes | 每 lane 一个原始 `Bf16x4Pack`，合计一个 q/k head 的 128 elements |
| value | 每个 tile/warp 的 `lane<4` | 每 lane 一个原始 BF16，八个 tiles 合计一个 value head 的 128 elements |
| gate | `state_tile==0 && warp==0 && lane==0` | 一个按 `{g,beta}` 排列的 `uint2` bit copy |

Layer-local record offsets 与输入 offsets 相同：

~~~text
key   [d,h,t,b] = (((b*T + t)*Hq + h)*128 + d)
value [d,h,t,b] = (((b*T + t)*Hv + h)*128 + d)
gate  [a,h,t,b] = (((b*T + t)*Hv + h)*2 + a)
~~~

Key writer 在 `load_raw_qk_lane` 返回后、`normalize_qk_lane` 之前写入；value
writer 直接写 `RawValueLane::bits`；gate writer 用 integer bit casts 保存两个 FP32
payload。每个 element 只有一个 writer，且 token loop 只遍历 valid prefix，所以 invalid
record tail 没有 store。Out 的 invalid tail 仍由现有 tile-owned lanes 写 exact zero。

Fold 端沿用 transition body 的 per-thread gate load：每个 thread 对该 value head/token 的
同一 aligned address 做 `uint2` read，再用 `__uint_as_float` 形成 `g/beta` registers。Warp 内
是 uniform address transaction；gate record 保持 bit identity，也不引入 FP32 arithmetic
conversion。

### 4.6 Recurrent launcher dispatch

现有 `recurrent.cu` 继续拥有全部 BF16 recurrent entries。Launcher dispatch 与 `T`
无关；wrapper 验证 shape 后把 `T` 转为 `std::int32_t width`，随 `Access` 直接传给
kernel。有限的编译期 instances 只有：

- Snapshot：现有的 `NormalizeInputs × Batched × Masked` 组合，且
  `static_assert(!Masked || Batched)`；
- Record：`Masked` 两种组合，normalization 固定为 true；
- Fold：两个 registered geometry，normalization 固定为 true。

Snapshot 保持现有 public 执行域；Record/Fold 的 wrapper 在 runtime 验证 production
ReplaySSM physical width `T=2..16`。
三个 entries 与 shared helpers 位于同一 CUDA translation unit，使用同一 compiler
configuration；`src/CMakeLists.txt` 不增加新的 recurrent CUDA source 或专用 flags。

若 bit-exact test 失败，先在相同 represented inputs、相同 runtime `T` 下比较三个 mode
instances 的 transition instruction sequence，定位具体 FMA/reduction/conversion 差异，
再回到 shared source 修正该差异。

Snapshot 与 Record 都使用 `block=(32,4,1)`、`grid=(Hv,B,8)`；`blockIdx.y` 是
batch row，`blockIdx.z` 是 state tile。Fold 使用第 4.7 节单独定义的 all-layer grid。

现有 Snapshot entry 直接改写为调用 shared runtime-`T` body，不缩窄它的 public domain；
Record/Fold 调用同一 body。现有 direct BF16 recurrent entry 的 publication 合同保持不变，
但它的逐 token transition 同样调用第 4.2 节的 helper。这样 Snapshot baseline、Record 和
Fold 共用同一 transition source。

### 4.7 Fold 的 grid、row mapping 与寻址

Wrapper 取 `B=rows.size()` 并验证 `1<=B<=C` 后，把 `rows[b]` 原样复制到第 4.3 节的
fixed-size kernel POD 的 `row[b]`。不筛选、不压缩、不重排。Static assertions 固定
row size/alignment 和整个参数的 trivial-copy contract：

~~~cpp
static_assert(sizeof(GdnReplayFoldKernelRow) == 8);
static_assert(alignof(GdnReplayFoldKernelRow) == 8);
static_assert(sizeof(GdnReplayFoldKernelRows) == 64);
static_assert(alignof(GdnReplayFoldKernelRows) == 16);
static_assert(std::is_trivially_copyable_v<GdnReplayFoldKernelRows>);
~~~

Fold launch 固定覆盖全部 active rows：

~~~text
block  = (32, 4, 1)
grid.x = Hv
grid.y = B
grid.z = L * 8
~~~

索引直接为：

~~~text
b            = blockIdx.y
layer        = blockIdx.z >> 3
state_tile   = blockIdx.z & 7
value_head   = blockIdx.x
linear_state_slot = rows.row[b].linear_state_slot
commit       = rows.row[b].commit_columns
~~~

`commit==0` 时，该 row 的全部 blocks 在读取任何 state/record address 前通过第 4.4 节的
CTA-uniform 分支返回；`grid.y=B` 和原始 batch-row mapping 保持不变。

`grid.z` 在两个 profiles 中分别为 384 和 240，低于 CUDA limit。Kernel 从 parameter space
读取 batch row `b` 的 row control，然后形成地址：

~~~text
record_outer = layer * C + b
qk_head      = value_head / (Hv/Hq)

key_base   = key_record
           + ((record_outer*T + t)*Hq + qk_head) * 128
value_base = value_record
           + ((record_outer*T + t)*Hv + value_head) * 128
gate_base  = gate_record
           + ((record_outer*T + t)*Hv + value_head) * 2

state_base = recurrent_layer0
           + layer * recurrent_layer_stride
           + linear_state_slot * (Hv*128*128)
           + value_head * (128*128)

state_element(r,c) = state_base
                   + (state_tile*16 + warp*4 + r) * 128
                   + lane*4 + c
~~~

`recurrent_layer_stride` 以 FP32 elements 传入；conv stride 以 BF16 elements 传入。Wrapper
在 launch 前完成 byte-stride divisibility、profile geometry 和 address-range checks。

`commit>0` 时，每个 block 加载自己的 16×128 FP32 tile，按 `commit` 顺序调用 shared
token loop，最后写回同一 tile。一个被更新的 state element 因而只有一次 initial load 和
一次 final store；token loop 中没有 global state traffic。

### 4.8 Fold 内的 conv-history publish

Conv history 复用 Fold launch 中已经存在的 blocks。令

~~~text
tile_block = value_head * 8 + state_tile
tid        = threadIdx.y * 32 + threadIdx.x
channel    = tile_block * 128 + tid
~~~

两个 registered `Cp` 都是 128 的倍数，`Geometry` 对此做 `static_assert`。Kernel 用一个 CTA-uniform 分支
`tile_block < Geometry::Cp/128` 决定该 block 是否承担 conv work；进入分支后 128 个
threads 全部对应有效 channel，不需要 per-thread bounds predicate。两个 profiles 的覆盖情况为：

| Profile | Available recurrent blocks/layer/row | Conv blocks used |
|---|---:|---:|
| `Hv=48,Cp=10240` | `48*8=384` | 80 |
| `Hv=32,Cp=8192` | `32*8=256` | 64 |

该 channel 的 bases 是：

~~~text
history = conv_layer0
        + layer * conv_layer_stride
        + linear_state_slot * (3*Cp)
        + channel

record(t) = conv_record
          + ((layer*C + b)*T + t) * Cp
          + channel
~~~

Recurrent final-state stores 完成后，各参与 thread 独立执行 BF16 gather；两块 storage 不
alias，不需要 block barrier。令旧 history 为 `[h0,h1,h2]`、commit extent 为 `m`：

~~~text
m = 0:  no load, no store
m = 1:  load h1,h2,p[0]             -> store [h1,h2,p[0]]
m = 2:  load h2,p[0],p[1]           -> store [h2,p[0],p[1]]
m >= 3: load p[m-3],p[m-2],p[m-1]   -> store those three values
~~~

`m=0` 的 row 在进入 conv mapping 前已经返回。其余每个 channel 只做恰好需要的 1–3 个
old-state/record loads 和三个 BF16 stores。`m>=3` 不读旧 history，任何分支都不扫描
`[0,m)`，也不会地址计算到 rejected suffix。这样 recurrent fold 与 conv commit 仍是一次
all-layer launch。

### 4.9 Projection/conv 的两类数值路径

Projection/conv 有两种不能混为一谈的 production path：融合路径在 FP32
projection accumulator 上继续 convolution；materialized 路径先产生一张 BF16
projection plane。Record form 保留各路径原有的 cast boundary。

#### 4.9.1 Fused projection/conv

对 fused routes，把 `GdnConvSnapshotEpilogue` 改为一个带 compile-time sink 的
`GdnConvEpilogue`：

~~~cpp
struct SnapshotHistoryPublish {
    __nv_bfloat16* state_write;
    const std::int32_t* snapshot_base_slots;
    // publish(token,row,s1,s2,p_bits)
};

struct RecordColumnPublish {
    __nv_bfloat16* conv_record;
    // publish(token,row,ignored_s1,ignored_s2,p_bits)
};

template <class Publish>
struct GdnConvEpilogue {
    const __nv_bfloat16* conv_weight;
    const __nv_bfloat16* state_read;
    const std::int32_t* initial_slots;
    const std::int32_t* valid_columns;
    __nv_bfloat16 *query, *key, *value;
    Publish publish;
    // fixed row geometry and global row offset
};
~~~

Snapshot form 的 `state_read/state_write` 指向同一 state pool；Record form 只有 const
read pointer，`RecordColumnPublish` 中不存在 state-write pointer 或 snapshot selector。每个
projection row 的 token loop 保持为：

~~~text
p = projection accumulator in the route's existing private precision
conv = existing ordered four-tap FMA sequence over s0,s1,s2,p
output = BF16(SiLU(conv))
p_bits = BF16_RNE(p)
publish(token,row,s1,s2,p_bits)
s0 = s1; s1 = s2; s2 = p
~~~

`SnapshotHistoryPublish` 写 `[BF16(s1),BF16(s2),p_bits]`；`RecordColumnPublish` 只写
`p_bits` 到 `conv_record[(b*T+t)*Cp+row]`。Private feedback 始终是 FP32 `p`，不从
BF16 record 回读。Invalid column 只写 zero q/k/value output，不调用 publish。Projection
MMA/SIMT body、PDL order、accumulator reduction 和 z-output path 均不改变。

#### 4.9.2 Materialized projection/conv

对数值边界已经是 BF16 `[Cp,T,B]` 的 staged/composed routes，Record form 直接把
layer-local `conv_record` 当作 projection output：

~~~text
record_flat = view(conv_record, [Cp, T*B])
project(x_flat, ..., qkv=record_flat, z=z_flat)
gdn_projected_conv<NoHistoryPublish>(
    projected=conv_record, initial_state=const_state, q, k, value)
~~~

因此不存在 transient projected plane，也不存在 projected → record copy。Snapshot
form 则把同一 projection schedule 写入 `scratch.projected`，再调用该 route 的同一
GDN-private projected-conv body，并使用 `SnapshotHistoryPublish` instance。

Materialized routes 共用一个 four-tap/SiLU/output device body，并把 output split 与 history
side effect 分开。每条物理 route 的 kernel 保留自己的 launch schedule：

~~~cpp
struct GdnConvSplitOutput {
    __nv_bfloat16 *query, *key, *value;
    __device__ __forceinline__ void store(int row, int token, __nv_bfloat16 x) const;
};

struct NoHistoryPublish {
    __device__ __forceinline__ void publish(...) const {}
};

__device__ __forceinline__ __nv_bfloat16 gdn_conv4_silu(
    float w0, float w1, float w2, float w3,
    float s0, float s1, float s2, float p);

template <class HistoryPublish>
__device__ __forceinline__ void gdn_projected_conv_body(
    /* route-owned addressing and token work */, GdnConvSplitOutput output,
    HistoryPublish history);
~~~

同一 route 的 Snapshot 和 Record instances 共用 initial-history/weight loads、四项累加
顺序、SiLU、BF16 cast 和 q/k/value row split。Snapshot instance 在每个 valid token
发布 three-column history；Record instance 的 history call 在编译期消失，而它的
projection input 已经是 record。两种 instance 都直接写 q/k/value outputs，不再物化
convolved plane，也不再启动三个 `extract_bf16_columns` kernels。

Q4/Q5 staged `T=4` 保留现有 per-channel post schedule；NVFP4 W4A4 保留现有 tiled post
schedule；composed short-`T` routes 保留现有 small-`T` channel/token work mapping，但由
GDN-private kernel 直接 split output。Snapshot/Record 只改变 destination 与 history
publication，不改变同一 route 的 projection 或 post-conv schedule。Block shape、tile
常量和后续 tuning 仍是这些 route 的私有性能参数，不是 ReplaySSM 设计合同。

对 NVFP4 W4A4，现有 `Nvfp4GdnInputOutput` 已支持 qkv/z split store。Snapshot 将 qkv
指向 `scratch.projected`，Record 将 qkv 指向 `conv_record`，z 均直接写 public z output。
原来用于完整 parent-output scratch 的 snapshot-only post path 中性化为只读 Cp rows 的
projected-conv path，不再分配 `Cp+Zp` 的 BF16 parent plane。

#### 4.9.3 Workspace 结果

Storage binding 根据 public form 不同，但 arithmetic route 不同步分叉：

| Route | Snapshot private BF16 staging | Record private BF16 staging |
|---|---|---|
| fused projection/conv | none | none |
| materialized Q4/Q5, W8, or NVFP4 A16 | one `[Cp,T*B]` projected plane | none; `conv_record` is the plane |
| materialized NVFP4 W4A4 | one `[Cp,T*B]` projected plane + W4A4 quant workspace | W4A4 quant workspace only |

ReplaySSM execution domain 内迁移到 GDN-private projected-conv 的两种 form 都不再需要
convolved plane。Record workspace query 从各 profile planner 选定的 route 推导 underlying
projection workspace：Q4/Q5 和 W8 为 zero，NVFP4 A16 为 zero，NVFP4 AllowA4 只返回
W4A4 activation-quantization workspace。Batched route 以 `aggregate_columns=B*T` 调用对应
projection capacity recipe，与实际 flattened projection 完全一致。

### 4.10 每条 projection route 的具体改造

Q4/Q5、W8 和 NVFP4 各自把现有 snapshot-only planner 中性化。对于相同 profile、`B/T`
和 compute policy，该 profile 的 Snapshot 与 Record 先取得同一个 projection/conv schedule，
再绑定 form-specific destination 和 compile-time history sink：

| Profile/path | 现有主要文件 | Record instance |
|---|---|---|
| Q4/Q5 fused `T=2..3,5..6` | `q4_q5_gdn_input_conv_snapshot.cu` | Q4/Q5 epilogues 持有 `GdnConvEpilogue<RecordColumnPublish>`；PDL order 不变 |
| Q4/Q5 staged `T=4` | 同文件的 projected-conv post | projection 直接写 `conv_record`，post 读 record 并使用 `NoHistoryPublish` |
| Q4/Q5 composed `T=7..16` 或 batched | wrapper + GDN-private projected-conv launcher | flattened projection 直接写 record，split-output conv 读 record；无 projected/convolved scratch 或 extract kernels |
| W8 fused `T=2..16` | `w8_gdn_input_gemm_splitk.cu` | 既有 exact-`T` fused table 增加 Record instances，MMA schedule 不变 |
| W8 batched composed | wrapper + GDN-private projected-conv launcher | aggregate projection 直接写 record，再由 split-output conv 只读消费 |
| NVFP4 fused A16 compute (A16 `T=2..16`；AllowA4 `T=2..3`) | `nvfp4_gdn_snapshot_output.cuh`、small-T files | output object 改用 `RecordColumnPublish`，production schedule 不变 |
| NVFP4 AllowA4 materialized (`T>=4`) | `nvfp4_gdn_input_w4a4.cu`、`nvfp4_gdn_snapshot_post.cu` | W4A4 split output 把 qkv 写到 record、z 写到 public output；中性 post 只读 record |
| NVFP4 batched composed | wrapper + GDN-private projected-conv launcher | flatten 后的 A16/A4 projection 直接写 record，同一 split-output conv 只读消费 |

每个 profile 内，名称带 `Snapshot` 但已被两种 forms 共用的 planner、schedule、output
object 和 post kernel 改为中性 `Conv` 命名，并只保留该 profile 的一份 route catalog。
不增加跨 Q4/Q5、W8、NVFP4 的统一 planner。Form 不能让同一 profile/problem 选到不同
projection arithmetic，只能改变 destination/history side effect 和由此导出的 workspace
binding。

完成后，projection record 不存在独立 copy kernel。Fused path 在原 epilogue 中进行一次
BF16 side store，materialized path 则把 record 作为原本就必须存在的 BF16 destination。
Recurrent 仍由 Snapshot/Record/Fold modes 共用唯一 transition source definition。

---

## 5. 实施顺序

### 5.1 冻结 physical storage ABI

1. 为任意正 `spec.slot_count=S` 的 `LinearAttentionStatePool` 实现
   `LinearAttentionStateAllLayersView`，从已绑定 regions 推导并检查两个 layer strides，
   不改变 pool backing，也不约束 production slot lifecycle；
2. 固定绝对 `linear_state_slot` 到 `conv/recurrent(layer,slot)` 的 address mapping，并保持
   现有 slot copy/zero 和 layer-local Op behavior 不变；
3. 实现 `GdnReplayRecordSpec/Layout/GdnReplayRecords` 以及无状态的 layer slice；
4. 为四个 record planes 建立 overflow-safe bytes/offset 计算；
5. 添加 storage tests，固定 shape、offset、alignment、slice alias、`C` 与 `S` 的独立性、
   跨层 absolute-slot mapping 和三个目标 record 容量。

这些 tests 通过后才增加 Op contracts。此后任何 kernel 只能消费 checked views，不能
自行发明 layer/row/slot stride。

### 5.2 实现 conv record producer

1. 分别把 Q4/Q5、W8、NVFP4 的 fused `GdnConvSnapshotEpilogue`、profile-local planner 和
   post/output types 中性化，只实例化 `SnapshotHistoryPublish`，不建立跨格式 route
   abstraction；
2. 在 `src/ops/gdn_input_proj/` 提取 materialized projected-conv 的公共 device arithmetic
   与 GDN-private launchers；让 ReplaySSM execution domain 内现有 materialized Snapshot
   routes 保留各自物理 schedule、直接写 q/k/value，并移除 convolved scratch 和 extract
   launches；
3. 运行现有 Snapshot oracle tests，确认重构后的 fused/materialized routes 继续满足已有
   数值准入，并固定更新后的 snapshot workspace capacities；不要求修改前后的 Snapshot
   output 或 state bit exact；
4. 在 fused routes 实例化 `RecordColumnPublish`；在 materialized routes 把 projection
   destination 绑定到 `conv_record`，再实例化 `NoHistoryPublish`；
5. 按第 4.10 节逐条接通 Q4/Q5、W8、NVFP4 fused/staged/composed launchers，
   并确认 materialized routes 没有 projected → record copy；
6. 在 public contract 增加 record overloads 和 record-specific workspace queries，它们
   调用各 profile 的中性 planner，但按 Record storage binding 计算 capacity；
7. 通过每条 route 的 output/record/source-state/workspace tests 后，固定 conv
   producer 实现。

### 5.3 实现 recurrent replay-record

1. 提取第 4.2 节的 q/k loader、value loader、transition 和 readout helpers，并建立
   shared runtime-`T` body；让 existing recurrent/Snapshot entries 通过
   `SnapshotAccess` 调用该主体；
2. 运行现有 recurrent 与 Snapshot oracle tests，确认修改后的 Ops 继续满足已有数值准入；
   不要求修改前后的 Snapshot bit exact；
3. 在 `gated_delta_net` contract 中增加 replay-record Op，实现 `RecordAccess` 和
   第 4.5 节的 unique writers；
4. 通过 layer-local output、raw-record 和 source-state tests，确认 generated records
   可以作为 Fold inputs。

### 5.4 实现 all-layer fold

1. 写入 `gdn_replay_fold` contract、all-layer view validation、`rows[b]` 一一对应合同和
   fixed POD row-control packing；
2. 为两个 exact geometries 实现 `FoldAccess`，先完成第 4.7 节 recurrent
   layer/batch-row/absolute-linear-state-slot 寻址；
3. 实例化 shared body 的 Fold mode，并对 `m=0..v` 的每个 commit extent 对比 snapshot
   recurrent state；其中 `m=0` 直接对比未修改的 initial state；
4. 在同一 entry kernel 接入第 4.8 节 conv-history publish；
5. 完成 in-place state、mixed commit lengths、rejected-tail 和 full all-layer tests。

### 5.5 完成 pair qualification 与 tuning

1. 组合 conv record producer + recurrent replay-record + fold；
2. 与 snapshot trajectory 逐 state 对比；
3. 分别对独立数学 oracle 验证 verify output 和 final state；
4. 跑 exact-shape benchmark；
5. 只在 correctness 通过的 kernels 之间进行性能选择和 tuning。

---

## 6. Correctness 验收

### 6.1 两个独立参照

ReplaySSM 测试同时使用：

1. **Snapshot trajectory baseline**：从相同 initial state 执行现有 verify-and-store，
   `m=0` 时使用未修改的 initial recurrent/conv state，`m>0` 时取顺序执行 \(m\) 个
   valid transitions 后发布的 state；
2. **Naive mathematical oracle**：从 represented BF16/FP32 inputs 和 FP32 initial state
   独立计算完整 recurrence 和 causal convolution。

第一个参照验证 finite-precision clone，第二个参照防止 snapshot 和 ReplaySSM 共享
实现错误。Pairwise equality 不代替 oracle qualification。

### 6.2 Record producer 检查

- replay-record 与 snapshot 的 valid BF16 outputs bit exact；
- `gated_delta_net_replay_record` 的 invalid `out` tail 是 exact BF16 zero；
- valid prefix 的 key/value record 与 source BF16 bits 一致；
- valid prefix 的 gate record 与 source FP32 bits 一致；
- valid prefix 的 conv record 与 snapshot newest-column BF16 bits 一致；
- record conv form 的 query/key/value/z 与 snapshot form 遵循各自相同的 output contract；
- replay-record 前后全部 source state bytes 不变；
- recurrent record 的 invalid tail 不变；conv invalid tail 不作数值断言；
- inactive outer rows 和四个 planes 之外的 guard regions 不变。

### 6.3 Fold 检查

对每个 physical width \(T\)、valid extent \(v\in[1,T]\) 和
commit extent \(m\in[0,v]\)，比较：

~~~text
baseline: verify and store every valid position; use initial state for m=0,
          otherwise select the state after transition m
new path: verify and record, fold records[0:m] in place
~~~

覆盖：

- `B=1,2,4,8` 和同一 fold 中的 mixed commit lengths；
- batch rows 使用数值不同且非连续的 absolute linear-state slots，并覆盖 `S!=C`；
- 每个 layer 独立的 initial state/record pattern，检查跨 layer 寻址；
- grouped q/k heads；
- `m=0` 时用 poison 填充该 row 的全部 records，确认 recurrent state 和 conv history 的
  全部 bytes 不变；
- `m=1`、`v>=2` 时的 `m=2`、`v>=3` 时的 `m=3`，以及 `m=v`，覆盖
  conv-history gather 的全部分支；
- 用 poison 填充 rejected suffix，确认 fold 不读取；
- inactive linear-state slots 和 record guard regions 完全不变；
- 连续多轮 `record -> fold -> record -> fold`，每轮检查 committed state。

### 6.4 目标 geometry

测试分两层：

- layer-local tests 穷举长度、valid mask、record bits 和 state effects；
- 注册的 48-layer/30-layer 完整 geometry 使用可区分的 per-layer/per-row
  patterns，检查 caller-supplied absolute slots、guard、conv gather 和 all-layer 寻址。

Layer-local recurrent Record/Fold tests 覆盖 `T=2..16`、每个 `v=1..T` 和
`m=0..v`。Projection/conv record tests 覆盖第 4.10 节每条 route 及其 schedule/policy
边界。完整 geometry 至少覆盖 27B `T=2/6`、35B-A3B `T=2/6/16`，以及 `B=1` 与
`B=8` mixed-prefix cases。

对应的独立 suites 是：

- `tests/test_gdn_replay_records.cpp`：record layout/binder 和 all-layer state view；
- `tests/ops/test_gdn_input_proj_conv_record.cpp`：conv record producer；
- `tests/ops/test_gated_delta_net_replay_record.cpp`：recurrent outputs、raw records 和只读
  state effect；
- `tests/ops/test_gdn_replay_fold.cpp`：all-layer state transition 与完整 record → fold
  trajectory。

### 6.5 数值准入

第一目标是：

- verify output bit exact；
- record fields bit exact；
- fold recurrent state 与 selected snapshot（`m=0` 时为 initial state）逐 FP32 element
  bit exact；
- fold conv history bit exact。

如果不同 mode instances 仍产生 recurrent-state bit difference，必须先定位差异来自
哪个 instruction/reduction/cast boundary，不能直接放宽测试。只有在原因已明确、两条
路径都独立通过 oracle、且 multi-round drift 测试通过后，才接受“每个
finite FP32 recurrent-state element 最多 1 FP32 ULP”。Conv history 是 exact gather，
不接受误差放宽。

与 snapshot 的 bit/ULP 比较是 finite-precision trajectory criterion，不是独立数学
oracle 的 tolerance。Recurrent output/state 对独立 FP64 oracle 使用现有 GDN suite 的固定
criteria：

| Result | Relative L2 | Gross absolute | Gross relative to max reference |
|---|---:|---:|---:|
| BF16 output | `4.1e-3` | `5.0e-6` | `5.5e-3` |
| FP32 recurrent state | `2.7e-3` | `1.0e-5` | `3.9e-3` |

这些 criteria 不按 mode、route、`T/v/m` 或 geometry 改动。Oracle 为 finite 时，任何
non-finite production value 都直接失败。Record bits 和 conv-history gather 仍做 exact
comparison。

---

## 7. Benchmark 与 tuning

新增独立 `gdn_replay` benchmark executable，不与长序列 GDN benchmark 混合结果。

### 7.1 测量项

1. `gdn_input_proj_conv_snapshot` 与 `gdn_input_proj_conv_record`；
2. `gated_delta_net_snapshot` 与 `gated_delta_net_replay_record`；
3. model-wide `gdn_replay_fold`；
4. 完整 Op 路径：全部 layers 的两个 record producers 加一次 Fold，与同 geometry 的
   verify-and-snapshot 路径直接比较。

前三项分开报告 record producer 的增量、减少 full-state stores 带来的收益和 fold 自身
开销；第四项报告 ReplaySSM 在 Op 层面的总 GPU 开销，是判断这套实现是否合格的主比较。

### 7.2 Workloads

| Geometry | Projection profile | Compute policy | `T` | Batch |
|---|---|---|---|---|
| 48-layer, `Hq=16,Hv=48,Cp=10240` | Q4/Q5 | fixed A16 | 2、3、4、5、6 | 1、2、4、8 |
| 48-layer, `Hq=16,Hv=48,Cp=10240` | NVFP4 | `A16Only` | 2、3、4、5、6 | 1、2、4、8 |
| 48-layer, `Hq=16,Hv=48,Cp=10240` | NVFP4 | `AllowA4` | 2、3、4、5、6 | 1、2、4、8 |
| 30-layer, `Hq=16,Hv=32,Cp=8192` | W8 | `A16Only` | 2、6、16 | 1、2、4、8 |

48-layer workloads 覆盖全部 production MTP physical widths；其中 `T=4` 覆盖
Q4/Q5 staged route 以及 NVFP4 `AllowA4` 从 fused 到 materialized route 的边界。

每个 workload 同时测 dense `v=T` 和 rows 间不同的 deterministic mixed-`v`。Fold 的
commit patterns 包含：

- all rows `m=1`；
- dense case 的 all rows `m=T`；
- `m` 在 rows 之间变化且逐行满足 `0<=m<=v` 的 deterministic mixed pattern。

使用仓库现有 warm/cold-L2 规则和 CUDA-event 计时。Fold 额外报告一次普通 host
launch 的 wall time，因为它最终会作为独立 kernel 启动。

`bench/ops/gdn_replay_bench.cu` 的 fold timed body 必须恰好是一次 public
`gdn_replay_fold` 调用；row-control 构造、buffer 初始化和结果检查均在计时区间外。
两个 record-producer timed bodies 同样各只调用一次对应 public Op。
完整路径 timed body 对每个 layer 顺序调用 conv Record 和 recurrent Record，随后恰好调用
一次 `gdn_replay_fold`；snapshot baseline 对相同 layers、inputs、`T/v` 执行对应的两个
snapshot Ops。两边都不把输入初始化、row-control 构造或结果检查计入 GPU 时间，并同时
报告 CUDA-event GPU latency 与整段 host submission 的 wall time。
除 latency 外，报告 committed recurrent-state bytes/s 和每个 `(layer,row,committed transition)`
的时间，使 B 与 commit length 的 scaling 可以直接比较。Benchmark 只构造
Op contracts 所需的 exact tensors，不加载 artifact，不进入 target、Program 或 speculative
schedule。

### 7.3 Tuning 重点

- replay-record 的 record stores 是否增加 register pressure 或降低 occupancy；
- fold 的 state load/store throughput、token-loop issue 和 commit-length scaling；
- fused conv epilogue 的 BF16 side store 是否增加 projection critical path；
- 各 materialized split-output conv route，特别是 Q4/Q5 `T=4` 与 NVFP4
  `AllowA4` `T>=4` 的 occupancy、memory traffic，以及移除
  convolved scratch/extract launches 后的实际收益；
- `B=1` 与 `B=8` 时的并行度。

首先用 benchmark 确定限制因素。只有在需要区分 occupancy、memory throughput 或
warp stalls 时才使用 profiler。数值不合格的 kernel 不进入性能比较。

---

## 8. 阶段完成标准

当且仅当以下条件全部成立，Op 实施阶段完成：

- record layout/binder 与本文的 shape、offset、alignment 和 capacity 一致；
- 任意正 `LinearAttentionStatePool::spec.slot_count=S` 的 backing 可直接导出本文规定的
  all-layer strided state view，且 `S` 不与 record capacity `C` 绑定；record producers 与
  fold 对 batch row `b` 使用调用方提供的同一个 absolute slot；
- ReplaySSM 的 production `T=2..16` domain 内所有可选 GDN input-projection routes 都有
  record-producing form；
- fused projection routes 只增加一次 BF16 record side store，materialized routes 直接以
  `conv_record` 为 projection destination，没有 projected copy、convolved scratch 或 extract
  launches；
- replay-record 只读 source state，并保存 exact raw record bits；
- fold 通过一次 kernel launch 处理全部 active rows 和 layers，按 `commit_columns` 原地更新
  recurrent state 与 conv history；`commit_columns=0` 的 row 严格不变；
- snapshot、replay-record 和 fold 共用同一份 recurrent transition body；
- pairwise exactness 和独立 oracle 检查达到第 6.5 节的标准；
- exact target geometries 的 tests 通过；
- benchmark 已覆盖第 7.2 节的 shapes、独立 kernels 和完整 Record→Fold 路径，并保留
  最快的合格实现；
- 现有 snapshot Op 合同和测试保持有效；
- 没有 production runtime call site 切换到新 Ops。

这些产物是后续 Engine 集成设计的全部 ReplaySSM 输入；集成阶段不再重新
定义 record format 或 fold 数值路径。
