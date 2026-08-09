# Batched Linear Attention Snapshot 迁移计划

> 状态：实施期临时计划，2026-08-09。完成迁移并把最终 contract 固定到 public Op header 与
> `concurrent-decode-operators.md` 后删除本文，不保留为历史设计文档。

本文规划一次性迁移以下三个 stateful snapshot Ops：

- `causal_conv1d_silu_snapshot`；
- `gdn_input_proj_conv_snapshot`；
- `gated_delta_net_snapshot`。

三者共同实现一次 Linear Attention target traversal 中的同一 per-row frontier：卷积状态和 recurrent
state 从相同的 request state slot 开始，对同一 valid prefix 递推，并把每一列的 provisional state 写入
相同的 snapshot slot 序列。它们因此使用一套 batch semantics，但仍保持三个清晰的数学 Op boundary。

NVFP4 的 phase-independent numerical policy 和各 semantic Op 的 B=1 production resolver 已由
public Op contract 固定。本计划只把既有 resolver 扩展到 aggregate `B*W`，不重新讨论 activation
precision。

---

## 1. 交付结果

迁移完成后，三个 public snapshot Ops 必须同时满足：

1. 一个调用处理 `B=1..8` 个相互独立的 sequences；
2. ordinary decode、MTP target verify 和 DFlash target verify 使用同一 contract；
3. column storage 使用 request-major `[W,B]` 语义，物理上可零拷贝展平为 `B*W` columns；
4. 每行通过 device-resident state-slot selectors 直接访问 shared Linear Attention state pool；
5. mixed valid extents 不拆 cohort，不循环调用单请求 Op，也不 gather request-local state；
6. `B=1` 通过同一 public contract 运行，当前 Engine 原子切换后不保留旧 scalar snapshot overload；
7. 当前 single-request Engine 的 Text、MTP、DFlash、prefix continuation 和 CUDA Graph 行为不回退；
8. 最终 production dispatch 不保留实验 route、candidate forcing 或重复实现。

本计划只迁移 snapshot execution。以下内容不在本次范围内：

- ordinary/distinct-state `causal_conv1d_silu` 和 `gated_delta_net` 的 batched prefill；
- concurrent Engine、Batch Builder、round commit 或 state-slot allocation policy；
- `LinearAttentionStatePool` 的物理布局和容量管理；
- ReplaySSM 内部算法；
- GDN control、output projection、normalization 等 column-independent Ops 的 batch ABI；
- MTP 或 DFlash 的 proposal/acceptance 调度。

---

## 2. 当前实现事实

| Op | 当前状态语义 | 当前 production 角色 | 主要迁移点 |
|---|---|---|---|
| `causal_conv1d_silu_snapshot` | 一个 scalar initial slot、一个 scalar snapshot base | composed GDN snapshot 的基础 state transition | `[B]` selectors、valid prefix、row-isolated state writes |
| `gdn_input_proj_conv_snapshot` | projection 后对一条 sequence 做卷积和 snapshot | 两个 target 的 GDN input production leaf | aggregate weight work 与 per-row sequence transition 解耦 |
| `gated_delta_net_snapshot` | 一个 scalar initial slot、逐 token recurrent snapshots | Verify 阶段 recurrent production path | batch grid、per-row state selection、masked recurrence |

现有 shared storage 已满足新 Op contract 的物理前提：

```text
conv state:      [C, 3, Slots] BF16
recurrent state: [128, 128, Hv, Slots] FP32
```

本次不改变 state plane、slot stride 或 layer ownership。一个 slot 仍表示全部 Linear Attention layers 的
同一 logical frontier；Op 只解释调用方传入的 slot selectors，不分配、释放或提交 slot。

`gdn_input_proj_conv_snapshot` 当前覆盖三类实际 projection storage：

- Qwen3.6-27B groupwise Q4/Q5 split parents；
- Qwen3.6-27B NVFP4 single parent；
- Qwen3.6-35B-A3B W8 single parent。

这些格式的 projection route 不同，但 convolution/snapshot semantics 完全相同。

---

## 3. 统一 batch contract

### 3.1 Logical shapes

`W` 是一次 invocation 为每行保留的 column width，`B` 是 exact logical batch size：

```text
flat_column(b,j) = b * W + j
```

三个 Op 的新 logical shapes 为：

| Value | Shape |
|---|---|
| convolution input/output | `[C,W,B]` |
| fused GDN hidden input | `[D,W,B]` |
| fused GDN query/key/value/z | `[Dout,W,B]` |
| GDN recurrent q/k | `[128,Hqk,W,B]` |
| GDN recurrent v/out | `[128,Hv,W,B]` |
| GDN recurrent g/beta | `[Hv,W,B]` |
| initial state selectors | `I32[B]` |
| snapshot base selectors | `I32[B]` |
| optional valid columns | empty or `I32[B]` |

NInfer Tensor 的 dim 0 连续；因此 `[D,W,B]`、`[Dhead,H,W,B]` 和 `[Hv,W,B]` 都可以在不搬运
数据的情况下 view 为对应的 `B*W` aggregate-column layout。Stateful implementation 必须从 flat column
恢复 `(b,j)`，不能把 `B*W` 当成一条 sequence。

支持域固定为：

- `B=1..8`；
- `B=1` 保留各 Op 和 weight-policy 现有的 positive `W` domain；其中 convolution/recurrent 为任意
  positive `W`，fused projection继续遵守当前 format/policy domain；
- `B>1` 的当前 qualification domain 为 `W=1..16`，即 aggregate columns 最多 128；
- registered Qwen3.6 convolution/recurrent geometries和现有 weight formats；
- `valid_columns` 省略时所有 rows 均有 `W` 个有效 columns；提供时每个值位于 `[0,W]`。

### 3.2 Typed metadata

三个 Op 使用同一组独立参数，不引入通用 request descriptor 或新的 state manager：

```text
valid_columns[B]          optional; empty means dense W
initial_state_slots[B]    state image read by each row
snapshot_base_slots[B]    first provisional destination of each row
```

Dense/masked 是 call 和 CUDA Graph topology 的一部分。Wrapper 只根据 `valid_columns` 是否为空选择
specialization，不把 device values 拷回 host。`B` 和 `W` 来自 tensor shape。

为了避免三个 wrapper 对 domain、shape 和 optional metadata 产生不同解释，可以增加一个
`src/ops/linear_attention/` 下的窄 private validation helper。它只验证 host-visible tensor facts并返回
`{B,W,masked}`；不成为 public descriptor，不持有 pool，不包含 request identity，也不执行 device-side
selector validation。

### 3.3 Per-row transition

令：

```text
Vb   = valid_columns is empty ? W : valid_columns[b]
S0b  = state_pool[initial_state_slots[b]]
Dstb = snapshot_base_slots[b]
```

每个 snapshot Op 独立执行：

```text
state = S0b
for j in [0,Vb):
    output[...,j,b], state = transition(input[...,j,b], state)
    state_pool[Dstb+j] = state
for j in [Vb,W):
    state_pool is unchanged
    stateful output[...,j,b] = exact zero
```

Convolution 和 recurrent Op 对相同 invocation 使用相同的 `Vb`、initial slot 和 destination sequence。
它们不推进 committed frontier；round transaction 后续根据 accepted/committed extent 选择已写 snapshot。

`gdn_input_proj_conv_snapshot` 中 projection 是 column-independent，而 convolution 是 stateful：

- projection 一次消费全部 `B*W` columns；
- q/k/value 只发布 valid-prefix convolution outputs，invalid tail 写 exact BF16 zero；
- z bypasses convolution，仍对全部 `B*W` safe input columns定义正常 projection output；
- invalid z 不参与后续有效 recurrence 或 externally visible output。

该边界既避免为 z 增加无意义的 mask kernel，也保持 fused Op 的完整、确定输出 contract。

### 3.4 State selection and alias rules

调用方保证 selector values 满足：

- 每个 initial slot 位于 `[0,Slots)`；
- 每个 snapshot base 位于 `[0,Slots)`，且 `base+Vb<=Slots`；
- 不同 rows 的有效 destination intervals 不重叠；
- 任一 row 的 destination 不覆盖另一 row 的 initial state；
- 多个 rows 可以共享一个只读 initial slot；
- 一个 row 的 initial slot 可以位于自己的 destination interval，implementation 必须在首次写入前加载该
  row 所需的 initial state。

如果多个 rows 共享 initial slot，则该 slot 在本 invocation 内必须保持只读，不能同时成为任一 row 的
destination。Input/output、weights、metadata、state pool 和 caller workspace 遵守各 Op header 声明的
non-overlap contract。

Selectors 是 device-resident trusted execution metadata。Wrapper 不同步读取它们，不增加范围检查 kernel、
hash、canary 或 request-level defensive bookkeeping；range 和 ownership 由 target schedule 保证。

### 3.5 CUDA Graph contract

Capture-time topology 固定：

- exact `B`；
- exact `W`；
- dense 或 masked；
- target geometry、weight format、compute policy 和 private route。

Replay-time values包括 selectors、valid extents 和 activation。它们的 device addresses 必须稳定，数值可以在
round boundary 更新。三个 Op 不根据 selector values 改变 launch count或 route。

---

## 4. `causal_conv1d_silu_snapshot`

### 4.1 Public contract cutover

Snapshot overload 改为接收：

```text
x [C,W,B]
weight [C,4]
conv_states [C,3,Slots]
valid_columns empty or [B]
initial_state_slots [B]
snapshot_base_slots [B]
out [C,W,B]
```

普通 in-place 和 distinct-state overload 不变，它们继续服务 single-request prefill。

### 4.2 Execution organization

- Dense `B=1` 保留当前 decode、small-W 和 sequence kernels 的等价 private specialization；新增 metadata
  不能给现有 production path增加逐 token branch。
- `B>1` launcher 一次提交 batch kernel，grid 中包含 request row；host 和 wrapper 不循环 `B`。
- Small-W path 让一个 CTA 同时持有一个 request 的 channel tile 和该行全部 `W` columns，先加载 initial
  history，再写任何 snapshot。
- Long-W `B=1` 保留 sequence path；当前不为 `B>1,W>16` 建立未使用的 route。
- Masked specialization 只递推 `[0,Vb)`，invalid columns 写零且不写 snapshot。

State addressing 直接使用 shared pool base、slot stride 和 row selector，不生成 pointer arrays，不 gather
initial state，也不为 batch 分配 transient workspace。

---

## 5. `gdn_input_proj_conv_snapshot`

### 5.1 Semantic decomposition inside one public Op

该 public Op 仍然拥有完整的：

```text
packed projection -> width-4 causal convolution -> SiLU -> q/k/value split -> snapshots
                  \-> z projection
```

实现上必须区分两个 execution domains：

```text
projection domain:       aggregate columns [0,B*W)
state-transition domain: B independent rows, each [0,Vb)
```

这不是把 fused Op 拆成 target schedule；wrapper 可以选择 projection-epilogue fusion或由 public aggregate
projection加一个 batched convolution postprocess组成，二者都实现同一个 closed contract。

### 5.2 Production route structure

最终 dispatch 使用以下结构：

1. **Dense B=1。** 保留当前 Q4/Q5、W8 和 NVFP4 exact-W optimized routes；它们改为新 public
   contract 的 `B=1` specialization，而不是保留旧 overload。
2. **Fused aggregate tile。** 当已有 packed projection kernel 的 column tile能够一次覆盖 `B*W` 时，扩展
   snapshot epilogue：每个 flat column映射到 `(b,j)`，每个 row 使用自己的 state selectors 和 valid
   prefix。Projection weights 仍由一次 aggregate schedule服务全部 rows。
3. **Aggregate projection + batched postprocess。** 对更大的 `B*W`，一次 public aggregate projection 写
   `[channels+z,B*W]`，随后一次 batched convolution/split snapshot stage消费 `[channels,W,B]`。该 route
   是正式 production composition，不是逐 request fallback。

Route selection 只依赖 weight format、compute policy、`B`、`W` 和 dense/masked topology。最终 crossover
由同输入、同 cache condition 的实测固定；candidate launchers只允许存在于开发期 benchmark，不能进入
public benchmark 或最终 dispatch。

### 5.3 Format-specific work

#### Q4/Q5 split parents

- Q4 和 Q5 projection仍分别处理它们各自的 packed parent，并共同服务 aggregate columns；
- 当前 exact small-W epilogue改为 batch-aware，不能把 template column count直接作为一条 sequence 长度；
- staged/composed route使用 batched convolution snapshot基础实现；
- q/k/value state rows和 z rows继续保持 disjoint writes。

#### W8 single parent

- ordinary `W=1,B>1` 必须进入 aggregate MMA/GEMM-like route，不能提交 `B` 个 decode GEMV；
- existing exact-column fused epilogue在其覆盖范围内按 `(b,j)` 更新 state；
- 更大 aggregate extent复用 W8 aggregate projection route和 batched postprocess。

#### NVFP4 single parent

- exact small-column fused kernels使用相同 batch-aware snapshot output；
- aggregate projection使用既有的 phase-independent `AllowA4` semantics；
- production resolver以完整 batched snapshot Op、exact geometry和 aggregate `B*W` 为依据，在合格的
  aggregate A16/W4A4 routes中选择 winner，不能按 request切成多次 projection；
- 只有最终 resolver在某个 reachable aggregate extent选择 A16 时，才需要提供对应的一次 aggregate A16
  projection。不得预先把 aggregate A16 设为 batch迁移的固定依赖；
- 本任务只补测由 `B*W` 新引入的 route domain，不重复 B=1 activation-precision campaign。

### 5.4 Workspace

两个 snapshot workspace-capacity queries 增加 exact `batch_size` 和 inclusive `min_width/max_width`。
Capacity recipe使用：

```text
aggregate_columns = batch_size * width
```

并覆盖区间内所有实际 route boundary。Workspace是一个 aggregate allocation，不按 request复制 arena，也不
存放 persistent state。`B=1` 的 exact query必须返回与当前 production route相同的 high-water capacity。

---

## 6. `gated_delta_net_snapshot`

### 6.1 Public contract cutover

Snapshot overload 改为：

```text
q/k        [128,Hqk,W,B] BF16
v/out      [128,Hv,W,B] BF16
g/beta     [Hv,W,B] FP32
ssm_states [128,128,Hv,Slots] FP32
valid_columns empty or [B]
initial_state_slots [B]
snapshot_base_slots [B]
```

Scale、`normalize_qk`、head mapping和数值 contract不变。普通 running-state 和 distinct-state overloads
不增加 batch axis。

### 6.2 Execution organization

- 一次 recurrent launch的 grid 同时覆盖 value head、state tile和 batch row；不对 requests做 host loop。
- 每个 CTA只持有一个 row 的 recurrent state tile，按 `j=0..Vb-1` 顺序更新。
- Dense specialization不读取 valid metadata，也不在 inner recurrence中保留 mask branch。
- Masked specialization在 `Vb` 停止 recurrence，零写剩余 output columns，不写剩余 snapshots。
- `normalize_qk=true/false` 继续是 private compile-time specialization；batch metadata不改变归一化公式。
- State pool保持原 `[128,128,Hv,Slots]` layout，kernel用 row selector和现有 slot stride直接寻址。

Snapshot route继续不需要 caller workspace。ReplaySSM可以决定调用方提供哪些 snapshot destinations，但不
改变本 Op 的输入、输出或 slot-selection semantics。

---

## 7. 当前 B=1 Engine 的原子切换

算子实现完成后，当前 Engine只切换到新 contract 的 `B=1` case，不引入 concurrent scheduler：

1. `RoundState` 现有 I32 selector storage物理上已经是 `[1]`，直接作为 plural selector tensors；
2. 当前 Verify invocation 的所有 `W` columns均有效，因此传 empty `valid_columns`，进入 dense topology；
3. Qwen family `TextContext`、两个 Variant leaf signatures 和 workspace-capacity calls同步切换；
4. existing state pool、slot roles、prefix continuation和 commit selection不变；
5. workspace planning传 `batch_size=1`；
6. 删除旧 scalar snapshot signatures和只为旧 ABI存在的 launcher entry，不增加 compatibility overload。

未来 concurrent Engine只需提供 `[B]` metadata和 `[W,B]` activation views，不再修改这三个 Op contract。

---

## 8. 实施顺序

### Step 1 — 固定 baseline

在修改 performance path 前，使用现有 public benchmarks记录两种 target geometry和实际 weight formats 的
`B=1` snapshot latency：

- convolution snapshot：`W=1,6,16`；
- fused GDN input snapshot：各 production format 的现有 route boundaries；
- recurrent snapshot：`Hv=48` 和 `Hv=32` 的 `W=1,6,16`；
- complete 35B GDN layer：`W=1,6,16`。

Baseline只作为本任务的本地对照，不提交 raw CSV、不固定 repository hash，也不新增长期报告。

### Step 2 — 公共语义和 shared validation

- 一次修改三个 contract headers，固定 §3 的 shape、metadata、invalid-tail、state effects和 aliasing；
- 修改 snapshot workspace queries；
- 增加唯一的 private metadata validation helper；
- 不保留旧 public overload。

此时允许工作分支短暂处于未闭合状态；不为保持中间态可运行而增加 adapter或双路径。

### Step 3 — Batched convolution snapshot

- 先迁移 standalone snapshot wrapper、launcher和 kernels；
- 用独立 oracle闭合 dense、masked、B=1和B>1 state semantics；
- 扩展 public convolution benchmark；
- 该 Op成为后续 composed GDN input route 的唯一 convolution implementation。

### Step 4 — Batched fused GDN input snapshot

- 先建立所有 formats通用的 aggregate-projection + batched-postprocess正确路径；
- 再迁移 Q4/Q5、W8、NVFP4 已有 fused epilogues；
- 对实际重叠 domain实测 fused和composed route，固定唯一 production dispatch；
- 将最终 NVFP4 B=1 resolver扩展到 aggregate `B*W`；只在实测 winner需要时补 aggregate A16 route；
- 用 fused FP64 oracle直接验证完整 public Op，不用 production composition作为 oracle。

### Step 5 — Batched recurrent snapshot

- 迁移 recurrent wrapper、launcher和 kernel grid；
- 保持 B=1 dense specialization的现有 arithmetic path；
- 闭合 mixed valid extents、snapshot effects和row isolation；
- 扩展现有 public recurrent benchmark。

### Step 6 — B=1 target cutover

- 同步修改 Qwen family runtime和两个 Variant leaves；
- 更新 workspace layout queries；
- 通过新 contract重建 current decode graphs；
- 不加入任何 `B>1` Engine state或调度逻辑。

### Step 7 — 整体验收和清理

- 跑三个 public Op conformance suites；
- 跑一次 complete GDN layer performance comparison；
- 对两个真实 target做短 Text/MTP execution，并对支持的 checkpoint做一次 DFlash Text execution；
- 删除 temporary candidate bench、route forcing、旧 scalar helpers和未被 production dispatch引用的 kernels；
- 把最终稳定语义补入 active reference，然后删除本文。

---

## 9. Correctness qualification

只扩展现有三套 Op tests，不新增按实现文件或 private route划分的测试程序。

### 9.1 Shared batch cases

Standalone convolution和 recurrent snapshot使用以下紧凑矩阵：

| Case | Purpose |
|---|---|
| `B=1`, dense, existing route boundaries | 证明同一新 contract保持当前数学行为 |
| `B=8,W=1`, dense | maximum ordinary concurrency和独立 state slots |
| `B=3,W=6`, valid `{6,3,0}` | MTP-width mixed prefix、empty row、invalid zero和no-write |
| `B=2,W=16`, valid `{16,7}` | DFlash-width边界和较长 per-row transition |

Cases使用非平凡 initial/base selector排列，检查：

- 每行独立 oracle output；
- 每个有效 snapshot；
- invalid output exact zero；
- 所有未选 state slots bitwise unchanged；
- shared read-only initial slot和同-row initial/destination overlap各有一个合法 witness；
- metadata、inputs和weights保持不变。

### 9.2 Fused projection cases

现有 B=1 cases继续覆盖 Q4/Q5、W8、NVFP4 的全部 production arithmetic profiles和route boundaries。
每种 weight format增加最少的 B>1 witnesses：

- 一个 dense ordinary aggregate case；
- 一个 `B*W<=16` 的 mixed-valid fused-epilogue case；
- 一个刚跨入 large aggregate composed route 的 case。

不对 `B×W×format×route` 做完整 Cartesian product。Max `B`、max `W` 和 zero-valid semantics已由共享 state
Ops独立覆盖；fused suite只覆盖会改变 packed projection或epilogue实现的边界。完整 oracle独立解码 packed
weights，并直接计算 projection、convolution、SiLU、split、z和snapshots。

### 9.3 不增加的测试

不新增以下内容：

- private kernel/launcher名称或route枚举测试；
- source-string、文件组织、deleted overload或getter测试；
- selector hash/canary；
- 对所有非法 device selector values的同步 rejection测试；
- 逐 request production调用与batched production调用的pairwise parity作为唯一 oracle；
- 重复的 target-level fixtures。

---

## 10. Performance qualification

### 10.1 Measurement matrix

每个 Op只测会影响 production decision 的 profile：

```text
ordinary: B = 1,2,4,8; W = 1;  dense
MTP max:  B = 1,4,8;   W = 6;  dense + one mixed-valid case
DFlash:   B = 1,4,8;   W = 16; dense + one mixed-valid case
```

Convolution和 recurrent覆盖两个真实 geometry；fused GDN input覆盖 Q4/Q5、W8、NVFP4。现有 public
benchmarks增加 `B/W/valid profile`，timed body仍是一次 public Op call。只有在 route crossover不清楚时才写
临时 internal-launch bench，结论固定后立即删除。

### 10.2 Gates

必须同时满足：

1. `B=1` 每个当前 production profile相对修改前 baseline不回退超过 3%；超出时先确认同条件重测，仍存在
   就不能接受；
2. `B>1` timed body是一次 public Op invocation，source和trace中不存在 `B` 次 host dispatch；
3. 对相同 rows，batched latency低于顺序执行对应 `B` 个 B=1 calls；
4. `gdn_input_proj_conv_snapshot` 的 weight-bearing stage一次处理 aggregate columns，不通过内部 chunk-per-
   request重复读取 weights；
5. mixed-valid path保持一个 batch，不因 valid extent拆 cohort；
6. complete B=1 GDN layer无显著回退。

本阶段没有 concurrent Engine，因此不声称 end-to-end B>1 tok/s；该结论留给 ordinary batched decode闭合后
验证。这里证明的是 Op-level batch execution、weight sharing和state isolation。

---

## 11. 完成条件

只有以下条件全部满足，本迁移才算完成：

- 三个 public snapshot contracts使用统一 `[W,B]`、typed selectors和optional valid-prefix semantics；
- B1 Engine只走新 contract，没有旧 scalar ABI或compatibility path；
- state pool物理 layout和ownership保持不变，Op没有 allocator或commit authority；
- convolution、fused input和recurrent都在一个 invocation中处理全部 rows；
- 三类 packed projection format均拥有合格的 aggregate production route；
- independent-oracle conformance通过，invalid tail和untouched state effects完整验证；
- §10 gates通过；
- temporary benchmark/candidate/dead code清理完成；
- stable docs更新且本文删除。
