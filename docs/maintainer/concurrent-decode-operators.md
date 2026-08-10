# NInfer 并发 Decode 算子需求

本文定义 NInfer 小规模并发推理在 operator boundary 上必须满足的 contract。它是
[小规模并发推理架构](concurrent-inference-architecture.md)进入实现前的算子前置要求，覆盖 ordinary
decode、MTP 和 DFlash；不定义 scheduler、allocator、具体 C++ 类型、CUDA launch topology 或 kernel
实现。

并发 decode 的目标不是让 `B` 个 request 依次调用单请求算子，而是让一次 model traversal 中的每个
算子 schedule 同时消费全部 active requests。现有 Op 可以直接满足该要求时，不为形式统一增加新的
batch axis 或并行 API。

---

## 1. Scope and invariants

本文采用以下产品范围：

- 单 GPU、单 resident model instance；
- startup-fixed `max_concurrency=C`，当前 qualification domain 为 `C<=8`；
- prefill 仍为 single-request chunked prefill，不做 batched prefill；
- active decode batch 在每个 round boundary compact，logical batch size 为 `B`；
- speculative backend 是 engine-wide 的 `off | MTP | DFlash`；
- MTP proposal window 最大为 5，DFlash proposal window 最大为 15；
- Text、Vision-prepared Text、prefix reuse、heterogeneous context length 和 heterogeneous
  sampling configuration 均可进入同一 decode batch。

必须保持以下 invariant：

1. 一个 model layer 的同一逻辑 Op 不得由 host 对 `B` 个 requests 分别调用；
2. weight-bearing Op 必须消费 aggregate columns，使一次 schedule 中的 weight work 服务全部 rows；
3. request 的 KV、convolution state、recurrent state、sampling state 和 speculative result 保持隔离；
4. speculative window 的 invalid-tail columns 可以参与无状态计算，但不能写
   persistent state 或成为输出；
5. batch membership 在一个 GPU execution unit 内不变；
6. batch-aware Op 只消费 execution view 和 metadata，不取得 allocator、request lifecycle 或 commit
   authority；
7. `B=1` 是同一个 contract 的 first-class case，不保留另一套 concurrent-only 数学路径。

---

## 2. Batch semantic contract

### 2.1 Compact rows and topology constants

`B` 是当前 DecodeRound 中 decode-ready requests 的精确数量，范围为 `1..C`。Batch row 是一个
round-local execution identity：

```text
batch row -> request slot -> sequence resources
```

empty slot 不进入 batch，因此 exact-`B` graph 没有为了填满 `C` 而保留的 inactive row。Row
mapping 在一个 GPU execution unit 内不变，下一 round 可以重建；Op 不得把 row index 当作
request identity 或 persistent-state identity。

`W` 是某个 Op invocation/topology segment 为每行保留的 column width。`B` 和 `W` 由 tensor
shape 和 capture-time topology 确定，不是 replay 时更新的 device metadata：

| Phase | `W` | Aggregate columns `T_total` |
|---|---:|---:|
| ordinary target decode | 1 | `B` |
| MTP target verify / alignment forward | `K+1` | `B*(K+1)`，当前最多 48 |
| MTP autoregressive proposal step | 1 | `B` |
| DFlash proposal / target verify | `K+1` | `B*(K+1)`，当前最多 128 |

### 2.2 Canonical request-major layout

当前 workload 不需要通用的动态 token packing。Canonical logical shapes 为：

```text
token / position:  [W, B]
hidden:            [D, W, B]
Q/K/V:             [Dhead, H, W, B]
logits:            [V, W, B]

flat_column(b, j) = b * W + j
```

NInfer Tensor 的 dim 0 连续，因此可以将这些 storage 零拷贝 view 为 `[D,B*W]`、
`[Dhead,H,B*W]` 或 `[V,B*W]`。例如 `B=2,W=3` 的 flat order 是：

```text
[A0, A1, A2, B0, B1, B2]
```

同一 request 的 columns 连续，使 sequence-sensitive Op 可以按 `j=0..W-1` 顺序执行且
snapshot destination 可以使用 `snapshot_base[b]+j`。Column-independent Op 只看到
`T_total=B*W`，不接收 batch axis。

### 2.3 Typed per-row metadata and state selection

Replay 前更新的是 exact positions、typed extents 和 resource selectors 等小型 device data。不建立包含
KV、Linear Attention、sampling、MTP 和 DFlash 全部字段的通用 `BatchDescriptor`；每个 Op 只消费
自己需要的 typed metadata，例如：

```text
GQA:               cache_positions[W,B], valid_columns[B], kv_table_rows[B]
Linear Attention:  valid_columns[B], initial_state_slots[B], snapshot_base_slots[B]
Sampler:           sampling_configs[B], logical_positions[B]
Speculative:       current_proposal_extents[B], accepted_drafts[B], licensed_counts[B]
MTP continuation:  next_proposal_extents[B]
```

Op 不接收 request identity 或 request slot，也不推导 resource ownership。Homogeneous model state 的 pool
bases 只传一次，每行通过 stable integer selector 选择资源：

- GQA 消费 shared KV planes、完整 block-table matrix 和 `kv_table_rows[B]`，不接收
  `B` 份 single-sequence view、device-pointer array 或 gathered contiguous KV；
- Linear Attention 消费 shared state pool 和逐行 state-slot selectors；target 保证各行
  snapshot range 合法且不重叠；
- DFlash fixed state 通过 per-row state-unit selector 选择，不进入 growing paged-KV pool。

Sampling configuration 是小型异构 per-request data，使用连续的 `SamplingConfig[B]`。其 optional
token-count reference 保持 sampling contract 的 typed state，不为形式统一强制进入模型 state pool。

### 2.4 Execution, licensing, and commit extents

Speculative round 不存在一个可以被所有 Op 共用的“valid extent”。每行必须区分：

```text
current_proposal_extent[b]  本轮进入 target verification 的 draft 数，0..K
target_extent[b]            target verification columns，current_proposal_extent[b] + 1
accepted_drafts[b]          target 接受的 draft 数，0..current_proposal_extent[b]
licensed_count[b]           target 许可产生的 token 数，accepted_drafts[b] + 1
next_proposal_extent[b]     MTP 根据新 frontier/budget 为下一轮生成的 draft 数，0..K
commit_count[b]             最终提交的 licensed prefix，0..licensed_count[b]
```

ordinary round 等价于 `current_proposal_extent=0`、`target_extent=1`、`licensed_count=1`。持续运行的一行
必须提交全部 licensed tokens；terminal output policy 可以在 licensed prefix 中截断，cancellation
可以丢弃该行的 provisional result。

`current_proposal_extent` 在 launch 前已确定；`next_proposal_extent` 只能在 acceptance 得到
`licensed_count` 后计算。二者不得合并为一个字段：MTP alignment 消费本轮 target extent，而后续
autoregressive proposal steps 由 `next_proposal_extent` mask。下一轮把已生成的
`next_proposal_extent` 作为新的 `current_proposal_extent`。

每个 sequence-sensitive Op 只接收它的语义域所需的 extent。例如 target GQA 和 Linear
Attention 使用 `target_extent`，MTP 第 `j` 个 autoregressive proposal step 使用
`j < next_proposal_extent[b]`。Acceptance outcome 不能反过来改变已执行的 target extent。

Stateful target execution 可以为 `target_extent` 的每个 column 写入 provisional KV 和 snapshot，但不在 Op
内推进 committed frontier。Round transaction 根据 `commit_count` 选择对应的 prefix/frontier；未选中的
tail storage 不具有逻辑有效性。Proposal、acceptance、licensed 或 commit 长度不同都不得拆分 active
batch。

### 2.5 Valid prefix and invalid tail

任何 Op 的 valid columns 必须是每行的连续前缀 `[0,valid_extent[b])`，不支持行内 sparse
mask。对 `[valid_extent[b],W)` 内的 invalid tail：

- input preparation 必须填入合法的 dummy token/position，不留下越界值或未定义 storage；
- column-independent Op 可以执行全部 `B*W` columns；
- Attention、Linear Attention 和其他 stateful Op 不修改 persistent state，并为 invalid
  columns 写零 mixer output；
- sampler、accept、hidden selection 和 output transaction 忽略 invalid columns；
- invalid result 不得成为下一 round input 或 externally visible output。

一行 `current_proposal_extent=0` 时，其 target extent 仍为 1，因而可在同一 speculative DecodeRound
中完成 ordinary target progress。MTP 对 current/next proposal extent 的 zero、partial 和 full values
始终选择同一个 exact-`B` graph family；任何情况都不为 extent 建立 cohort。

---

## 3. Operator classification

并发迁移把 Op 分为三类：

| Class | Requirement |
|---|---|
| Sequence-sensitive | contract 必须显式理解多个独立 requests、typed state selectors 和所需 extent |
| Column-independent | 继续使用现有 `T` 维；一次调用消费 `T_total`，只扩展或确认 route/domain/performance |
| Single-request phase only | 第一版保持现状，不加入 batch contract |

“已有 positive-`T` contract”只证明 columnwise 数学可表达，不自动证明 production route 会共享 weight
work 或在 `T_total` 上达到合格性能。

---

## 4. Sequence-sensitive Ops

### 4.1 Main Text and MTP causal GQA

必须迁移：

- `gqa_attention`

一个调用必须同时处理 `B` 个独立 sequences。Logical inputs 为
`q[Dhead,Hq,W,B]`、`k/v[Dhead,Hkv,W,B]`、`positions[W,B]`，output 与 query 的 row
block 一一对应。每行可以具有不同的：

- context length 和 exact positions；
- Main 或 MTP paged-KV block-table row；
- 该 invocation 所需的 typed valid extent；
- batch-level execution envelope 内的 exact visible range。

Persistent pool planes 和完整 block-table matrix 只传入一次；Op 使用 `kv_table_rows[B]`
选择 block-table row。不得把 `[W,B]` 展平成一条 sequence，不得传入 `B` 份
single-sequence view 或 device-pointer array，不得 gather KV 到连续临时缓存，也不得在 wrapper
中循环调用 single-sequence Attention。

`valid_columns` 是可选的 typed execution metadata：省略表示所有 rows 的 `W` columns 都有效，提供
`I32[B]` 才启用逐行 valid-prefix masking。这个 dense/masked 选择属于 call/graph topology，不通过 host
回读 device value 推断。Masked row 的 invalid position slots 重复该行最后一个 valid position；空行填 0。
因此 ordinary `W=1` batch 使用 dense path，而异长 speculative target batch 仍在同一次 batched Op 中保持
cache no-write 与 exact-zero output 语义。

第一版不迁移：

- `gqa_kv_append`；
- `gqa_attention_cached`。

它们当前只服务 single-request prefill、MTP prefill 或 bridge。已迁移的 `gqa_attention`
在 single-request prefill 中使用同一 contract 的 `B=1,W=T` case；若以后引入 batched
prefill，再扩展其他 prefill-only entries，而不是提前增加未使用的并发路径。

### 4.2 Linear Attention state transition

Qwen3.6 Linear Attention 的 persistent transition 包含 convolution window 和 recurrent matrix，二者必须
以同一个 per-row frontier 执行。必须迁移：

- `causal_conv1d_silu_snapshot`；
- `gdn_input_proj_conv_snapshot`；
- `gated_delta_net_snapshot`。

Batch contract 必须接受等价于以下 per-row selectors 的信息：

```text
initial_state_slots[B]
snapshot_base_slots[B]
valid_columns[B]
```

每行从自己的 initial state 开始，按该行 `j=0..valid_columns[b]-1` 顺序递推，并把
column `j` 的 provisional state 写入 `snapshot_base_slots[b]+j`。在 target verification 中，
`valid_columns` 是§2.4 的 `target_extent`。一个 request 的 state 或 invalid-tail work 不能影响
另一行。

`gdn_input_proj_conv_snapshot` 是当前 target production leaf；standalone convolution snapshot 也必须拥有
一致的 batch state semantics，以支持该 leaf 的 composed route。普通/ distinct-state
`causal_conv1d_silu` 和 `gated_delta_net` 继续服务 single-request prefill，不在本阶段迁移。

该 contract 只定义独立 state transition，不把 GDN 算法写入 common state pool；未来其他 Linear
Attention 类型可以消费相同的 per-sequence state-selection 语义并拥有自己的数学 Op。

### 4.3 Sampling

必须迁移：

- `sample`；
- corresponding sampling workspace-capacity query。

一个调用为 ordinary decode 的每个 row 产生一个 token。Logical logits 为 `[V,B]`，
output 为 `[B]`，并逐行消费独立的：

- `SamplingConfig[B]` 中的 sampling parameters；
- seed 和 logical position；
- penalty-history/token-count state；
- output destination。

Greedy 和 stochastic rows 可以共存于同一 batch。任何一行的 RNG、penalty update 或 selected token 不得
依赖 compact row 的历史 occupant，也不得修改另一 request 的 state。

### 4.4 Shared speculative transaction

MTP 和 DFlash 共用的 target verification transaction 必须迁移：

- `speculative_prepare_verify_inputs`；
- `speculative_accept_greedy_drafts`；
- `speculative_select_accepted_hidden`。

逻辑数据域为：

```text
current token / length                 [B]
draft tokens                           [K, B]
target input ids / positions           [K+1, B]
target logits                          [V, K+1, B]
current proposal extent / target extent [B]
accepted drafts / licensed count       [B]
round output tokens                    [K+1, B]
request sampling/token-count state     per row
```

Prepare 只在每行 `current_proposal_extent` 内构造有效 draft，target 只在该行 `target_extent`
内建立 provisional state。Acceptance、correction/bonus sampling、penalty-history update 和 selected
hidden transition 逐行独立，并输出 `accepted_drafts` 和 `licensed_count`。Request statistics 由 runtime
根据这些 per-row result metadata 结算，不进入 Op persistent state。不同
proposal 或 acceptance lengths 不触发 target replay、cohort 或 per-request model call。最终
`commit_count` 属于 round transaction，不是 acceptance Op 的 resource-ownership authority。

### 4.5 MTP helpers

现有 scalar `mtp_prepare_alignment_ids` 由 batched `mtp_prepare_next_round` 原子替换。该 Op 根据每行
自己的 `accepted_drafts[b]` 构造 shifted alignment block，并根据 acceptance 后的新 frontier 与剩余
budget 计算 `next_proposal_extent[b]` 和 MTP AR 起始 position。MTP 第 `j` 个 autoregressive proposal
position 仍对全部 active rows 执行一次 aggregate model step，但只有
`j < next_proposal_extent[b]` 的行可以推进 MTP persistent state。

以下 Op 已是 column-independent transform，不增加 batch semantics：

- `mtp_pack_fc_input`；
- `mtp_split_attn_in`；
- `proposal_remap_token_ids`。

### 4.6 DFlash attention and context update

必须迁移：

- `prepare_masked_block`；
- `swa`；
- `bidirectional_gqa_attention`；
- paged `kv_cache_append_prefix`；
- cyclic `kv_cache_append_prefix`。

每行拥有独立的 anchor、proposal extent、context frontier、full-context paged table row、
local cyclic state 和 context-append count。DFlash block 内的 bidirectional visibility 只覆盖同一
request 的 query block，不能跨越相邻 flat blocks。

DFlash local cyclic KV 是 fixed per-sequence state。并发 storage 必须能为每个 active sequence 选择独立
state unit；它不进入 growing paged allocator。DFlash target feature capture 的 destination 也必须按
`[feature_rows,W,B]` 隔离，避免 requests 互相覆盖；这可以由 runtime copy/layout 完成，不要求新增独立
数学 Op。

Concurrent DFlash decode 使用统一 topology：

```text
append pending committed target features (per-row count may be zero)
    -> one batched DFlash proposal block
    -> one batched target verification
    -> per-row acceptance and licensed result
    -> boundary selects each row's commit prefix
```

并发 graph 不保留 request-local `initial` / `steady` 分组。Final prefill 建立 DFlash context 和 decode
anchor；proposal 在统一 decode round 中产生。不同时间加入的 requests 因而具有相同的 round entry
state，不需要 phase cohort。

### 4.7 Position and round-state transitions

当前 scalar-only control operations 不能直接表达并发 rows。必须提供 batched round semantics，覆盖：

- per-row cache position 和 RoPE position preparation；
- per-row `rope_delta`，包括 Vision-prepared Text + MTP；
- per-row frontier advance；
- Linear Attention committed snapshot selection；
- MTP position、accepted-draft/licensed count 和 speculative statistics result metadata。

不要求把每个现有 scalar helper 机械复制成 vector overload。上述变化应由少量有明确 transaction
ownership 的 rowwise preparation/commit Ops 承担；已经属于 speculative accept 的 state update 不再由
额外 scalar calls 重复执行。

`rope` 本身是 column-independent。只要输入已经是正确的 flat per-column positions，它不需要知道
request boundary。

---

## 5. Column-independent Ops

以下现有 Op 的数学 contract 已允许独立 columns。并发 schedule 必须对 aggregate `T_total` 调用一次，
但不增加显式 request batch axis：

| Family | Entries |
|---|---|
| Input | `embedding` |
| Normalization | `rmsnorm`, `gated_rmsnorm` |
| Position/elementwise | `rope`, `sigmoid_mul`, `silu_mul`, `residual_add` |
| Projection | `linear`, `linear_add`, `linear_pair`, `linear_swiglu` |
| Fused projection | `attn_input_proj`, non-snapshot `gdn_input_proj`, `gdn_norm_gating_proj` |
| Post-mixer | dense FFN entries above, `sparse_moe` |
| Output | output-head `linear`, `argmax`, `proposal_remap_token_ids` |
| MTP transforms | `mtp_pack_fc_input`, `mtp_split_attn_in` |

对这些 Op，“完成”同时要求：

1. wrapper 和 workspace-capacity query 接受实际 `T_total`；
2. registered 27B Groupwise/NVFP4、35B-A3B W8/MoE 和 companion routes 均有合法路径；
3. target 不按 request 重复调用；
4. weight-bearing implementation 在 `B=2..8` 上执行 aggregate weight work，而不是一个 launch 内隐藏
   `B` 份独立 GEMV；
5. `T_total>16` 时不因旧 single-request Verify profile 进入 unsupported 或明显不适合的 route；
6. `B=1` 的现有 production route 不发生显著回退。

其中 1、2、4、5、6 是 column-independent Op 自身的 qualification；3 属于后续 concurrent
schedule integration。Op qualification 只需证明公开入口一次消费完整 `T_total`，不得为证明第 3 项而加载
artifact 或调用 target、Program、Engine、DecodeRound。

Route 可以随 `T_total` 选择 small-T、GEMM、MoE small-token 或其他已 qualification 的实现。Decode phase
名称不应被解释为 `T=1`。

---

## 6. Single-request-only work

第一版不为以下路径增加 batch contract：

- Vision encoder、Vision Attention、patch/position embedding、scatter、merger；
- Text/Vision chunked prefill；
- prefill-only `gqa_kv_append` 和 `gqa_attention_cached`；
- non-snapshot convolution 和 Linear Attention recurrence；
- prefix-reuse copy/restore；
- allocator、page reservation、retained-state ownership；
- EOS、stop-string、usage、response publication 和 network I/O。

这些工作可以在 scheduler 的 single-request PrefillChunk 或 CPU boundary 中发生，不属于 batched
DecodeRound 的 model-execution invariant。

---

## 7. Workspace and CUDA Graph requirements

- shared executor workspace 按 startup configuration 的最大 aggregate extent 规划，不按 request 复制；
- ordinary、MTP verify/alignment、MTP AR 和 DFlash block 分别拥有与其固定 `W` 一致的 layout profile；
- workspace-capacity query 覆盖该 profile 的全部合法 `B=1..C`；
- graph capture 期间，typed batch metadata、compact inputs/outputs、pool planes 和 table matrix 地址稳定；
- `B`、`W` 和 tensor shapes 属于 captured topology；replay 之间只更新 compact row metadata、exact
  positions、typed extents 和 state selectors；
- context envelope 可以由 batch-level maximum 选择 graph/profile，但 Op 必须使用 exact per-row context；
- 一个 Op 内部可以由多个 kernels 组成，但 host schedule 不得按 request 展开这些 kernels；
- invalid-tail columns 的计算结果不得进入 persistent KV、Linear Attention state、sampling
  state 或 output。

Paged KV 的 page size、plane order、allocation和 ownership contract 不因本次 batch migration 改变。Batched
consumer 使用已有 fixed table matrix 和 per-row table selectors；allocator 仍不理解 Attention
或 batch。

---

## 8. Qualification requirements

### 8.1 Semantic evidence

Sequence-sensitive Op 需要覆盖代表性 `B=1,2,8`，并至少验证：

- 不同 context lengths、positions 和 physical page mappings；
- rows 之间 KV、conv、recurrent、RNG 和 statistics 无串扰；
- current/next proposal extents 为 full、partial 和 zero，且 accepted/licensed/commit frontiers 不同；
- compact row order 与 stable slot identity 不同；
- `B=1` 与原数学 contract 一致。

每个数学 Op 继续使用一个独立 oracle；batch oracle 是对每行独立数学结果的组合，不以旧 production
kernel 或逐请求 production calls 作为唯一 oracle。

### 8.2 Performance evidence

性能检查只覆盖能改变 production decision 的范围：

- ordinary `B=1..8`；
- MTP aggregate target/alignment extent 至 48，以及 AR step `B=1..8`；
- DFlash aggregate block/verify extent 至 128；
- registered short、medium 和 long context profiles；
- 两个 target 的实际 weight formats 和启用的 speculative backend。

Column-independent Op 的证据止于公开 Op 边界：使用 synthetic fixture 构造精确 format、shape 和
`T_total`，不加载 artifact，也不调用 target、Program、Engine 或 whole-round benchmark。最终判断使用与
瓶颈相符的 useful bandwidth/`READ_%` 或 useful FLOP/s/`TC_%`。Linear 只测一个带正常
warmup/repetition 的 `T=1` benchmark point，并计算 `T * median(T=1) / median(T)`；不得把顺序执行
`T` 次 `T=1` launch 当作 comparison workload。

Column-independent Op qualification 必须确认：

- 一个公开 Op call 消费完整 `T_total`，wrapper 不按 column/request 展开；
- Linear/lm_head/projection 的 weight traffic 能服务多个 columns，而不是隐藏 `B` 份独立 GEMV；
- 公开 `B=1` Op route 与修改前相比无显著回退。

后续明确包含 sequence-sensitive Op 或 concurrent Engine 的工作还必须确认 batch metadata 和 indirect
addressing 没有抵消主要收益，并在整轮层面确认 `B>1` 优于顺序执行相同 requests、满足 whole-batch
invariant。这些要求不是 column-independent Op qualification 的完成条件。

临时开发 benchmark 可以调用 private launcher 用于 route 选择；胜出实现进入 production dispatch 并由
公开 Op 复验。最终删除落选候选、实验开关和重复 benchmark，不删除已选中的 production 实现。

---

## 9. Implementation order

迁移按依赖关系闭合，不按 backend 平铺开发：

1. **固定 batch ABI 和 invalid-tail 语义。** 以§2 的 compact-row identity、request-major
   layout、typed metadata、frontier 和 invalid-tail contract 作为后续所有 Op 迁移的共同前提。
2. **审核 column-independent aggregate routes。** 对 ordinary、MTP 和 DFlash 实际会产生的
   `T_total` 检查 registry、workspace 和 production route，只修复实测证明的 domain 或性能 gap。
3. **迁移 batched causal GQA。** `gqa_attention` 一次消费全部 compact rows 及各自的
   paged-KV table row、exact positions 和 valid columns。
4. **迁移 batched Linear Attention snapshots。** 依次闭合
   `causal_conv1d_silu_snapshot`、`gdn_input_proj_conv_snapshot` 和
   `gated_delta_net_snapshot` 的 per-row state selection 和有效前缀更新。
5. **迁移 sampler 与 row transitions。** 闭合 heterogeneous sampling、position preparation、
   committed snapshot selection 和 ordinary round commit。
6. **闭合 ordinary batched decode。** 先在 public Engine path 上完成 ordinary whole-round
   correctness 和 performance qualification；这是第一个完整里程碑。
7. **接入 MTP。** 在已闭合的 target batch 上迁移 shared speculative
   prepare/accept/hidden selection、MTP alignment 和每个 batched autoregressive proposal step，然后完成
   MTP whole-round qualification。
8. **准备 DFlash fixed state units。** 在迁移 DFlash Op 之前，先使 local cyclic KV、
   boundary-local KV 及 per-sequence retained feature/state buffers 可以由 compact row 通过 stable unit index
   选择。这些 fixed state 不进入 growing paged-KV allocator，也不阻塞 ordinary 或 MTP 里程碑。
9. **接入 DFlash。** 最后迁移 masked block、SWA、full-context Attention、paged/cyclic
   context append，并复用已迁移的 shared speculative transaction，完成 DFlash whole-round
   qualification。

每个 sequence-sensitive Op 都做原子契约切换：当前 engine 同时改用新 contract 的 `B=1` case，不保留旧
production Op path。Host 只能按 semantic family、`B`、`W` 和 batch-level execution envelope
选择 route/profile，不能根据任意 per-row state 创建 cohort。普通 batched decode 未闭合前，不以
speculative helper 或 DFlash 的局部完成替代该 milestone。

---

## 10. Completion checklist

并发 decode 的 operator substrate 在以下条件全部满足后完成：

- [ ] ordinary round 的每个 layer 对全部 active rows 只执行一次 logical Op schedule；
- [ ] Main/MTP GQA 支持 shared pool/table matrix、per-row table selectors、positions 和 typed extents；
- [ ] convolution 与 recurrent snapshots 支持独立 per-row state selectors；
- [ ] sampler 支持 heterogeneous per-request configuration、RNG 和 penalty state；
- [ ] speculative prepare/accept/select 区分 proposal、target、accepted、licensed 和 commit frontiers；
- [ ] MTP 的 alignment 与每个 AR position 保持 whole-batch execution；
- [ ] DFlash 使用统一 batch round，并隔离 full/local context 与 feature state；
- [ ] Linear、MoE、lm_head 和其 fused variants 覆盖全部 aggregate extents；
- [ ] workspace 和 graph metadata 对 `B=1..C` 地址稳定且容量充分；
- [ ] `B=1` 无显著性能回退，`B>1` 不退化为 request-local execution。

---

## Related documents

- [Small-scale concurrent inference architecture](concurrent-inference-architecture.md)
- [Paged KV context storage](paged-kv-cache.md)
- [Op development and qualification](op-development.md)
