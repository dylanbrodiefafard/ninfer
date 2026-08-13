# Qwen3.6 历史 thinking 语义与跨用户轮次前缀复用

本文定义 NInfer 对 Qwen3.6 `preserve_thinking` 的产品语义、服务接口、frontend
渲染边界和 runtime prefix-reuse 行为。本文同时适用于 `qwen3_6_27b` 与
`qwen3_6_35b_a3b`；两者共享同一套 Qwen3.6 frontend 和 family runtime。

本文是一份实现约束，不是某个 PR 的代码审查。实现完成后，`docs/serving.md`、
`docs/maintainer/paged-kv-cache.md` 和
`docs/maintainer/concurrent-inference-architecture.md` 中受影响的公开或架构描述必须与本文一致。

本设计以已经集成 ReplaySSM 的当前 Qwen3.6 runtime 为实现基线。speculative target verify 不再保存
逐位置完整 GDN snapshot，而是写 Program-owned Replay records，并在 accepted-prefix 决策后 Fold 到
lane 的 current state。本设计不会恢复旧 snapshot trajectory，也不会把 Replay records 提升为跨请求
状态。

---

## 1. 目标与范围

本设计必须同时实现两种 Qwen3.6 官方支持的历史语义：

- `preserve_thinking=true`：历史 assistant thinking 继续出现在后续 prompt 中；
- `preserve_thinking=false`：当前用户轮次关闭后，该轮 assistant thinking 从后续 prompt 中移除。

两种模式都必须满足：

- 模型实际消费的 token、KV、GDN、MTP/DFlash 和 position state 完全一致；
- 只在拥有完整 continuation checkpoint 的位置复用 prefix；
- tool loop 不破坏下一次用户轮次所需的恢复点；
- 请求没有精确回传历史 reasoning 时，不从隐藏 cache 中擅自补写；
- 无法证明 checkpoint 等价时安全退化，不产生“token 看似匹配但状态不等价”的复用。

本设计不引入 arbitrary longest-common-prefix cache、多级会话 checkpoint、跨 sequence
共享或 copy-on-write KV，也不改变当前单 GPU、单 resident model、每 lane 一份 retained
sequence 的产品模型。

---

## 2. 必须先成立的语义模型

### 2.1 模型状态属于完整 token 前缀

设第一轮实际执行的 token 序列为：

```text
P + U1 + R1 + A1
```

其中：

- `P`：此前已经稳定的历史；
- `U1`：本轮真实用户问题；
- `R1`：模型产生的 thinking/reasoning；
- `A1`：可见回答、tool call 或其他 assistant 输出。

该轮结束时，resident state 对应的是完整前缀 `P + U1 + R1 + A1`。这里的 state
不只是 Main Text KV，还包括 GDN recurrent state、末尾 hidden、MTP 或 DFlash backend
state，以及继续执行所需的位置语义。

下一轮如果渲染为：

```text
P + U1 + A1 + U2
```

那么旧的 `A1` 状态不能继续使用。旧状态中的 `A1` 是在 `R1` 条件下计算得到的；删除
`R1` 后，必须从 `R1` 之前恢复，并重新计算 `A1 + U2`。只删除 reasoning 对应的 KV、
保留后面的 answer KV 或 recurrent state，在数学上不等价。

因此，本设计的首要不变量是：

> 只有 incoming prepared prompt 的完整 token/position/media identity 与某个已保存的完整
> continuation checkpoint 前缀一致时，才允许从该 checkpoint 继续执行。

### 2.2 `preserve_thinking` 是 prompt 语义，不是 cache hint

`preserve_thinking` 决定下一轮模型看见什么历史：

| 模式 | 下一轮历史 | 正常复用路径 |
|---|---|---|
| `true` | `P + U1 + R1 + A1 + U2` | 从 current execution frontier append |
| `false` | `P + U1 + A1 + U2` | 回退到 `R1` 之前并重新 prefill suffix |

cache 行为是该语义的结果，不能反过来把 `preserve_thinking` 解释成“尽量保留某些缓存”。

### 2.3 当前轮次与已关闭轮次

Qwen3.6 模板使用最后一个真实用户问题作为分界：

- assistant 消息位于最后一个真实用户问题之后时，属于当前开放轮次；
- 新的真实用户问题到达后，之前的开放轮次变成已关闭轮次；
- `preserve_thinking=false` 只保留当前开放轮次的 thinking；
- `preserve_thinking=true` 保留所有轮次的 thinking。

“真实用户问题”必须与当前 Qwen3.6 模板的 `last_query_index` 规则完全一致：

- 普通 `role=user` 消息是新的真实用户问题；
- `role=tool` 不是新的用户问题；
- 内容仅为 `<tool_response>...</tool_response>` wrapper 的兼容性 `role=user` 消息也不是；
- system、assistant 均不推进用户轮次。

该分类只能由 frontend 的一个共享 helper 计算。渲染器和 boundary 计算不得分别实现两份
近似逻辑。

### 2.4 ReplaySSM 后必须区分的三种状态

当前 runtime 中存在三个不同生命周期，不能都称为“保存 state”：

1. **current continuation**：与 `execution_frontier` 对齐，供当前 sequence 继续 append/decode；
2. **turn checkpoint**：与 turn rewrite frontier 对齐，跨请求保留，供未来删除 thinking 后恢复；
3. **speculative transaction scratch**：只属于一轮 frozen compact batch，从 verify 活到
   `resolve_pending_batch` 完成。

ReplaySSM records 属于第 3 类。它们按 compact batch row 而不是 sequence lane 寻址，既不进入
`SequenceState`，也不参与 checkpoint 的 `KeepExisting`、`CaptureNew`、`Drop`。只有 Fold、必要的
hidden correction、DFlash terminal context flush 和同步全部完成后，current continuation 才重新成为可由
下一个请求复用的 committed state。

---

## 3. 改造前实现为什么会在用户轮次失去复用

改造前的 family frontend 已经拥有 `PromptOptions::preserve_thinking`，模板也实现了：

```cpp
keep_thinking = options.preserve_thinking || (message_index > last_query_index);
```

但当时的 serving translation 固定写入 `preserve_thinking=false`。更重要的是，原有
`PromptIdentity::assistant_content_boundary` 表示本次 generation prompt 的 assistant opener，
而不是当前用户轮次中最早会在下一轮被重写的 assistant 位置。

在只有一次 assistant 输出的简单轮次中，这两个位置可以重合。tool loop 中则不同：

```text
U1
  A1(reasoning + tool call)   <- 下一轮最早发生变化的位置
  T1
  A2(reasoning + tool call)
  T2
  A3(final answer)            <- 最后一次 generation opener 附近
U2
```

每次 assistant/tool 循环都会提交新请求。原有请求启动路径会清空旧 boundary，并在新 prefill
中捕获最新 generation opener。到 `U2` 时，`preserve_thinking=false` 会删除 `A1`、`A2`、
`A3` 的 reasoning，token 序列从 `A1` 的 thinking 开始变化；此时保存的却是更晚的 boundary，
它本身已经不匹配，只能 FullReset。

所以完整修复包含两件独立的工作：

1. 将 `preserve_thinking` 作为明确的服务和请求语义暴露出来；
2. 让 `false` 模式保存并维持正确的“本轮重写 checkpoint”。

只完成第 1 项可以让 agent 部署通过 `true` 获得 append hit，但不能让默认 `false` 路径正确地
复用已经稳定的更早历史。

---

## 4. 最终产品决策

### 4.1 默认值与推荐场景

`ninfer-serve` 的默认值为：

```text
preserve_thinking = false
```

它与 Qwen3.6 模板默认语义一致。以下部署推荐在 server level 开启：

- coding agent；
- 多步 tool-use agent；
- 需要保留决策依据和跨轮一致性的长会话；
- 希望历史成为 append-only、优先降低跨用户轮次 TTFT 的场景。

推荐启动方式：

```bash
ninfer-serve model.ninfer --preserve-thinking
```

服务不得根据请求是否携带 `tools` 自动切换。自动切换会在同一历史中改变 prompt 语义，造成
不可预测的模型行为和 cache reset。

### 4.2 `enable_thinking` 与 `preserve_thinking` 相互独立

`enable_thinking` 控制本次 generation prompt 是否允许产生新 thinking；
`preserve_thinking` 控制如何渲染已经存在的历史 thinking。四种组合都有效：

| `enable_thinking` | `preserve_thinking` | 行为 |
|---:|---:|---|
| `true` | `false` | 本轮生成 thinking；轮次关闭后从历史移除 |
| `true` | `true` | 本轮生成 thinking；后续历史继续保留 |
| `false` | `false` | 本轮不生成 thinking；历史 closed-turn thinking 也不保留 |
| `false` | `true` | 本轮不生成 thinking；已有历史 thinking 仍保留 |

实现不得以其中一个值覆盖、推导或拒绝另一个值。

### 4.3 会话内应保持稳定，但切换必须安全

同一 conversation 推荐始终使用同一个 `preserve_thinking` 值。稳定模式是单 checkpoint
能够持续高效工作的前提。

显式切换并不是未定义行为：incoming token identity 仍然是最终权威。切换导致较早历史发生变化时，
当前 turn checkpoint 可能太晚，runtime 必须退化到更早的合法 checkpoint；当前只有一个 turn
checkpoint，因此通常是 FullReset。不得为了保留 cache hit 而继续使用旧语义的状态。

OpenAI Responses 的 `previous_response_id` 链应继承 parent response 已解析的值，避免子请求省略
字段时意外切换。子请求显式给出不同值时允许建立新的语义分支，但预期发生 restore 或 reset，并在
request log 中记录 semantic change。

### 4.4 为什么稳定模式只需要一份 checkpoint

单 checkpoint 的充分性可以按用户轮次归纳：

- 在稳定 `false` 模式下，所有更早的已关闭轮次已经按“不含 thinking”的形式存在于当前 resident
  prompt 中；新的真实用户问题只会关闭刚才的开放轮次。因此本次最早变化点一定是该开放轮次第一条
  assistant 的 thinking 开始处，更早历史不会再次改变。
- 在稳定 `true` 模式下，新的真实用户问题不会删除任何旧 thinking，incoming prompt 正常是 current
  frontier 的追加；新捕获的 checkpoint 只用于当前新开放轮次将来的 rewrite 或异常 history edit。
- 模式切换或更早 history edit 会破坏上述归纳前提，此时一份当前轮次 checkpoint 不保证足够，按
  exact identity 退化到 FullReset 正是预期行为。

因此，保存当前开放轮次最早的一个 checkpoint 既充分又最小。为每条 assistant 保存 checkpoint
不会减少正常 `false` rewrite 的重算范围，因为第一条 assistant thinking 之后的所有状态都依赖它。

---

## 5. 对外可观察行为

### 5.1 `preserve_thinking=true`

客户端完整回传 `reasoning_content` 时，新的 prompt 是旧执行前缀的追加：

```text
resident: P U1 R1 A1
incoming: P U1 R1 A1 U2 <assistant opener>
                         ^ append from execution frontier
```

正常结果：

- reuse path 为 `AppendAtFrontier`；
- 只 prefill `U2` 和新的 generation opener；
- 为 `U2` 捕获新的 turn checkpoint；
- 模型能够继续读取历史 reasoning。

这条路径依赖客户端或服务端 history store 原样保留 reasoning。flag 本身不能恢复请求中不存在的
文本。

### 5.2 `preserve_thinking=false`

新用户消息关闭上一轮后：

```text
resident: P U1 R1 A1
incoming: P U1    A1 U2 <assistant opener>
              ^ restore turn checkpoint, then re-prefill
```

正常结果：

- execution frontier 不匹配；
- reuse path 为 `RestoreTurnCheckpoint`；
- runtime 恢复到 `R1` 之前；
- 重新 prefill 去掉 reasoning 后的 `A1`、tool history、`U2` 和 generation opener；
- 捕获 `U2` 的新 turn checkpoint。

这不是与旧执行状态等价的 append；它是对新 prompt 的正确增量计算。其 prefill 成本与刚刚关闭的
一轮长度相关，而不再与全部历史长度相关。

### 5.3 同一轮 tool loop

在 `U1 -> A1 -> T1 -> A2 -> T2 -> ...` 中，所有 tool result 都属于 `U1` 的开放轮次。
每个后续请求通常从 execution frontier append，但必须保留 `A1` thinking 之前的同一份 turn
checkpoint。不得把它替换为 `A2` 或更晚 generation opener。

### 5.4 reasoning 没有被回传

对无状态 Chat Completions 或 Anthropic Messages，如果客户端只回传 answer/tool call 而没有回传
reasoning：

- prompt 以请求中明确给出的 history 为准；
- `preserve_thinking=true` 不允许从 resident cache 私下补回旧 reasoning；
- frontier 不匹配时只能从合法 turn checkpoint 重算，或者 FullReset；
- 日志可以说明此次请求开启了 preserve 但历史 reasoning 不完整，不能把它当作协议错误。

外部导入的历史本来就可能没有 reasoning，因此服务不能强制每个 assistant 消息都携带该字段。

---

## 6. Serving 和请求接口

### 6.1 内部表示与解析优先级

`GenerationRequest` 增加：

```cpp
std::optional<bool> preserve_thinking;
```

server 配置增加：

```cpp
bool preserve_thinking = false;
```

解析顺序为：

```text
request override
    > Responses parent inherited value
    > server --preserve-thinking default
    > false
```

普通 Chat Completions 和 Anthropic Messages 没有 server-side conversation parent，因而跳过第二层。
server 配置在进程生命周期内固定，最后一层只是类型和模板的明确默认值。

必须由一个 resolver 一次性得到本请求的 `ResolvedPromptSemantics`：

```cpp
struct ResolvedPromptSemantics {
    bool enable_thinking;
    bool preserve_thinking;
};
```

同一个 resolved value 同时传给 `PromptInput`、`PreparedRequest` 和 request logging。不得在
translation、preparation 和 logging 中分别调用 `value_or()`，避免以后出现渲染值与日志值不一致。

### 6.2 OpenAI Chat Completions

与 vLLM 一致的规范入口是：

```json
{
  "model": "qwen3.6-27b",
  "messages": [],
  "chat_template_kwargs": {
    "preserve_thinking": true
  }
}
```

同时接受窄兼容别名：

```json
{
  "preserve_thinking": true
}
```

规则如下：

- `chat_template_kwargs` 必须是 object；
- `preserve_thinking` 必须是 boolean 或 null；
- 两个入口都出现且值不同，返回 HTTP 400；
- 两个入口相同则接受；
- NInfer 不支持任意模板变量，unknown non-null kwargs 返回明确的 unsupported-option 错误，不能静默忽略。

### 6.3 OpenAI Responses

Responses 接受相同的 `chat_template_kwargs.preserve_thinking` 和顶层兼容别名。

`StoredResponse` 必须保存该 response 的 resolved `preserve_thinking`。使用
`previous_response_id` 时：

- child 未指定：继承 parent resolved value；
- child 显式指定相同值：正常继续；
- child 显式指定不同值：允许建立分支，标记 semantic change，token identity 决定 restore/reset；
- response store 已保存的 `reasoning_content` 继续参与 history flattening，因此 `true` 模式无需客户端
  再手工重发 parent reasoning。

如果 parent 已被驱逐或 `previous_response_id` 无效，请求按现有 Responses 错误语义失败，不能退回
server default 并生成另一段历史。

### 6.4 Anthropic Messages

Anthropic 原生：

```json
{"thinking": {"type": "enabled"}}
```

只映射 `enable_thinking`，不得隐式映射 `preserve_thinking`。NInfer 额外接受顶层 boolean
`preserve_thinking` 作为协议扩展；省略时使用 server default。

历史 `thinking` block 转换为 `reasoning_content`。`redacted_thinking` 没有可重新渲染的明文，不能用于
重建旧 token；此时仍遵守“请求 history 是权威”的规则并安全退化。

### 6.5 输出和使用方责任

- Chat Completions 继续把 reasoning 输出为 `message.reasoning_content` 和对应 stream delta；
- Responses store 继续保存 reasoning output item/history；
- Anthropic 继续使用 thinking content block；
- `preserve_thinking` 不改变 answer/reasoning 的输出分离格式；
- 无状态客户端若希望获得 `true` 的 append hit，必须把服务返回的 reasoning、content、tool calls 和
  tool results 按协议完整回传。

---

## 7. Frontend：计算真正的 turn rewrite boundary

### 7.1 边界的精确定义

`turn_rewrite_boundary` 是一个 token frontier，位于：

> 最后一个真实用户问题之后，第一条 assistant 消息的
> `<|im_start|>assistant\n` header 末尾、可选 `<think>` 块之前。

如果最后一个真实用户问题之后还没有历史 assistant 消息，则使用本次
`add_generation_prompt` 产生的 assistant header 末尾。

示意：

```text
<|im_start|>assistant\n | <think>\n...
                        ^ turn_rewrite_boundary
```

边界放在 header 之后，是因为 header 在 `preserve=true/false` 两种历史中都稳定；真正可能被删除的
第一个 token 是 `<think>` 块。这样可以复用最大且仍然严格等价的前缀。

该边界与 `enable_thinking` 无关。即使 generation prompt 通过一个空的闭合 think block 禁用新
thinking，下一次真实用户问题到达后，该 block 的渲染仍可能变化，因此仍需同一边界。

### 7.2 Renderer trace

`render_chat` 不再只返回 `std::string`，而是返回：

```cpp
struct RenderedChat {
    std::string text;
    std::optional<std::size_t> turn_rewrite_byte_offset;
};
```

渲染流程为：

1. 通过共享 helper 得到 `last_query_index`；
2. 渲染消息时，第一次遇到 `i > last_query_index && role == assistant`，在写完 assistant header
   后记录 byte offset；
3. 如果没有遇到，且 `add_generation_prompt=true`，在 generation assistant header 后记录；
4. 如果没有 generation prompt 且没有可重写 assistant，返回 `nullopt`；
5. 断言一份 rendered prompt 最多记录一个边界。

不能通过搜索第一个或最后一个 `<think>` 字符串推断位置；用户内容可能包含相同文本，tool loop 也有
多个合法 think block。

### 7.3 从 byte offset 得到 token frontier

text prompt 的处理顺序：

1. tokenize 完整 `RenderedChat::text`；
2. tokenize `[0, turn_rewrite_byte_offset)`；
3. prefix token 数即 `turn_rewrite_boundary`；
4. 精确断言 prefix encoding 等于完整 encoding 的同长度前缀；
5. 要求 serving generation prompt 下满足 `0 < boundary < prompt_tokens`。

该边界紧邻模板产生的换行，是当前 Qwen tokenizer pre-tokenizer 的 word boundary，因此独立编码 prefix
与完整编码具有相同 token 切分。第 4 步仍是必须的，以防未来 tokenizer/template 资源改变该性质。

multimodal prompt 中，media placeholder 会在 processor 内展开。processor 必须让 offset 与展开同步：

- placeholder replacement 完全位于 boundary 之前时，用 replacement 与 placeholder 的 byte 长度差
  调整 boundary；
- replacement 位于 boundary 之后时不调整；
- replacement 与 boundary 相交属于 frontend 逻辑错误；
- 展开后再执行上述完整 encoding、prefix encoding 和精确前缀断言。

这样只进行一次 media decode/preprocess；不能为了得到 boundary 再独立跑一遍 processor。

### 7.4 Prepared prompt identity

将：

```cpp
PromptIdentity::assistant_content_boundary
```

改名为：

```cpp
PromptIdentity::turn_rewrite_boundary
```

它描述本次 prompt 将来可能被关闭轮次语义重写的位置，不再描述“最新 generation opener”。对应的
`snapshot_boundary` 局部字段也应改用 turn-checkpoint 命名，避免实现再次退回旧含义。

`preserve_thinking` 不需要额外写入 resident prefix key。最终 rendered token、position、token type、
media digest 和 rope identity 才是复用权威；如果两个 flag 值恰好产生完全相同的 prepared identity，
从相同 checkpoint 继续就是等价的。resolved flag 仍需进入日志和 Responses conversation metadata。

---

## 8. Runtime：一份 turn checkpoint 的所有权与生命周期

### 8.1 Checkpoint 表示什么

每个 retained `SequenceState` 最多保存一份 `TurnCheckpoint`。它表示当前最后一个真实用户问题之后，
最早可能在下一次真实用户问题到达时发生重写的 token frontier。

设 checkpoint frontier 为 `B`，current execution frontier 为 `E`。当前 ReplaySSM runtime 中的物理
组成如下：

| State | Current continuation | Turn checkpoint at `B` | 仅属 speculative round |
|---|---|---|---|
| Main Text KV | 同一 paged allocation 中逻辑有效到 `E` | 同一 allocation 的 `[0,B)` prefix；不复制 KV | verify 产生但未提交的 trailing columns |
| Target GDN | fixed slot `lane` | fixed slot `C + lane` | Program-owned Replay records |
| Target hidden | `tail_hidden` | token `B-1` 的 `boundary_hidden` | target frame 中的 candidate/selected hidden |
| MTP | 当前 MTP KV 与 next drafts | 共享 MTP KV 中至少 `[0,B-1)` 的 bridge prefix | 当前 round frame、next-draft publication |
| DFlash | `local`、共享 `full` cache 和 current context frontier | `turn_checkpoint_local` 和共享 `full` cache 的 `[0,B)` prefix | `pending_features` 和本轮 append controls |
| 输入 identity | 当前 ledger、token/position/media identity | 同一 identity 的 `[0,B)` 精确前缀 | frozen compact row mapping |

`C` 是 startup-fixed maximum concurrency。Target GDN 的 current/turn-checkpoint slot 映射固定为
`lane`/`C+lane`，不会因 accepted token 数移动或交换角色。

MTP checkpoint 不保存 `mtp_drafts`。restore 只依赖 `boundary_hidden` 和至少覆盖到 `B-1` 的 MTP KV，
随后通过 `BeforeSuffix` bridge 重建 suffix 与新 proposals。DFlash 也不复制第二份 full paged KV；它只为
不可由 full prefix 单独恢复的 local cyclic cache 保存 `turn_checkpoint_local`。

实现必须把现有 `PrefixCheckpoint boundary`、`boundary_hidden`、
`dflash_boundary_valid/frontier` 统一收敛到 turn-checkpoint 语义。host metadata 只保留一个原子发布点：

```cpp
struct TurnCheckpoint {
    bool valid = false;
    std::uint32_t frontier = 0;
};
```

物理 payload 仍位于固定的 GDN slot、owning hidden tensor、共享 paged KV 和 DFlash
`turn_checkpoint_local`，不必嵌入该小型 host struct。`valid=true` 必须蕴含 Main Text prefix、GDN slot、
turn-checkpoint hidden，以及当前
startup-fixed speculative backend 所需的 MTP prefix 或 DFlash local/full prefix 全部完整并指向
`frontier`。因此删除 `hidden_valid`、`mtp_prefix_valid`、`dflash_boundary_valid` 和重复的
`dflash_boundary_frontier`；不能存在只发布了部分 component 的 checkpoint。

Host invariant 为：

```text
!valid  => frontier == 0
valid   => 0 < frontier
        && Text KV / prefix identity cover frontier
        && GDN turn slot / turn-checkpoint hidden belong to frontier
        && (MTP disabled || MTP KV covers frontier - 1)
        && (DFlash disabled || full prefix and turn_checkpoint_local belong to frontier)
```

Turn checkpoint 与 current execution frontier 引用同一份 exclusively owned KV bundle。checkpoint
不会复制 Main Text 或 backend KV；restore 时只 truncate trailing pages，并恢复 fixed continuation
state。Replay records 不属于该 bundle，不得增加到 `TurnCheckpoint` 或 `SequenceState`。

### 8.2 三种 checkpoint action

每个 `RequestPlan` 除 reuse path 外，还必须明确携带：

```cpp
enum class TurnCheckpointAction {
    Drop,
    KeepExisting,
    CaptureNew,
};
```

`RequestPlan` 同时携带 `std::optional<std::uint32_t> turn_checkpoint_capture_frontier`，它当且仅当 action
为 `CaptureNew` 时有值。`KeepExisting` 的 frontier 来自已经发布的 checkpoint，`Drop` 不携带 frontier。

含义为：

| Action | 使用条件 | Request start/finish 行为 |
|---|---|---|
| `KeepExisting` | desired boundary 与已有 checkpoint 相同且 prefix 精确匹配 | 保留 metadata 和 dedicated payload；tool loop 继续持有 |
| `CaptureNew` | desired boundary 位于本次要 prefill 的 suffix 内 | 先完成可能的旧 checkpoint restore；prefill 到新 boundary 时覆盖 payload，并在成功后发布 metadata |
| `Drop` | prompt 没有合法 desired boundary | 清除 atomic checkpoint metadata；stale payload 不再可达 |

当前实现“每次 request start 无条件清空 boundary”的行为必须移除。`KeepExisting` 是解决多步 tool loop
的关键；仅改变 frontend boundary 数值但仍无条件清空，第二次 append 后仍会丢失 checkpoint。

### 8.3 Reuse 与 checkpoint planning 顺序

planner 按以下顺序工作：

1. 要求 lane 不处于 `Pending`；上一 speculative round 必须已经完成 Fold、backend commit 和同步；
2. 依据 exact prefix identity 选择候选 `AppendAtFrontier`、`RestoreTurnCheckpoint` 或 `FullReset`；
3. 验证 MTP/DFlash 对候选路径所需的 backend continuation state，不完整则降级；
4. 根据 incoming `turn_rewrite_boundary` 选择 checkpoint action；
5. 如果 desired boundary 已位于 reuse base 之前、但没有可保持的同一 checkpoint，则选择能重新捕获它的
   更早合法路径；当前没有其他 checkpoint 时为 `FullReset`；
6. 在最终 reuse base 确定后生成 MTP bridge、Vision suffix、prefill split 和 service-work quanta。

等价伪代码：

```text
require sequence is not Pending
reuse = choose_exact_reuse(sequence, prompt)
reuse = validate_backend_or_reset(reuse)
base  = reuse.frontier_or_zero
B     = prompt.turn_rewrite_boundary

if B is null:
    checkpoint_action = Drop
else if sequence.turn_checkpoint is complete
     and sequence.turn_checkpoint.frontier == B
     and prompt exactly matches sequence identity through B
     and reuse did not invalidate that state:
    checkpoint_action = KeepExisting
else if B > base:
    checkpoint_action = CaptureNew(B)
else:
    reuse = FullReset
    base  = 0
    checkpoint_action = CaptureNew(B)

derive_backend_and_prefill_plan(reuse, base, checkpoint_action)
```

最后一个分支很少在稳定会话中触发，但它保证 runtime 不会为了本次 append hit 而发布一份缺少未来
恢复点的 retained sequence。

### 8.4 `KeepExisting`

`KeepExisting` 适用于同一用户轮次的 append 和从同一 turn checkpoint restore 两种情况：

- 不清空 checkpoint metadata；
- 不覆盖 GDN slot `C+lane` 或 `boundary_hidden`；
- 不覆盖 DFlash `turn_checkpoint_local`；
- 保证共享 MTP KV 中 checkpoint 所需的 prefix 仍可达，但不要求保存当前 `mtp_drafts`；
- suffix prefill/decode 只修改 GDN current slot `lane`、current backend state 和 boundary 之后的 KV；
- speculative verify 继续写 Program-owned Replay records，Fold 的 destination 只能是 `lane`；
- 不保存、复制或延长 Replay records 的生命周期；
- prefix identity 必须继续证明 checkpoint frontier 之前的输入未变。

如果 reuse path 本身是 `RestoreTurnCheckpoint`，restore 可以把 boundary state 复制到 current slot，但
dedicated turn-checkpoint slot 仍保持不变，DFlash restore 也只执行
`turn_checkpoint_local -> local`，因而后续仍能保存
同一个 checkpoint。

### 8.5 `CaptureNew`

`CaptureNew(B)` 要求 `base < B < prompt_tokens`。现有 prefill snapshot 机制可以继续使用，但其含义改为
turn checkpoint：

- reuse path 为 `RestoreTurnCheckpoint` 时，必须先用旧 checkpoint 完成 truncate/restore，再把旧
  checkpoint 标记为待替换；不能在 restore 消费它之前清空或覆盖 dedicated slots；
- 完成可能的 restore 后，将旧 host checkpoint 置为 invalid；request 的 staged state 独占尚未发布的
  `CaptureNew(B)`；
- prefill chunk 在 `B` 精确结束；
- Main Text KV 已处理到 `B`；
- prefill 使用 `GdnStateAction::UpdateInPlace`，running GDN state 从 `lane` 复制到 `C+lane`；
- 保存 token `B-1` 的 normalized hidden，供 MTP bridge 使用；
- MTP 不复制 drafts 或一份独立 cache；完整 MTP preparation 必须保证共享 MTP KV 中未来 restore 所需的
  `B-1` prefix 仍然可达；
- DFlash 必须先把 context append 到 `B`，再执行 `local -> turn_checkpoint_local`；共享 full paged KV 不复制；
- 后续 chunk 从 `B` 继续，不重置 current state；
- 只有整个 prompt prefill/finalization 和 backend 完整性检查成功后，才原子发布
  `TurnCheckpoint{true, B}`。

Capture 发生在 ordinary prefill，而不是 speculative verify，因此不产生也不消费 Replay records，不执行
Replay Fold。若 physical checkpoint payload 已被覆盖、但后续 prefill 失败，整个 sequence 必须清除且
不得重新暴露为 retained；不能回退发布半更新的新 checkpoint 或已经被覆盖的旧 checkpoint。

如果请求在捕获后、发布前失败，未完成的 sequence 不得作为 retained state 暴露。

### 8.6 `Drop`

`Drop` 将 host checkpoint 重置为 `{false, 0}`。它不要求释放 current frontier，也不影响当前请求的
正确执行；只是该 sequence 下一次不能从 turn boundary 恢复。固定的 GDN turn-checkpoint slot、
`turn_checkpoint_hidden` 和 DFlash `turn_checkpoint_local` 可以保留 stale bytes，后续
只有新的 `CaptureNew` 才能重新赋予它们含义。Drop 与 Replay record arena 无关。

正常 serving generation prompt 都应产生 desired boundary，因此 `Drop` 主要服务于 Engine API 中
`add_generation_prompt=false` 的输入或异常准备路径。

---

## 9. 各类模型状态的具体约束

### 9.1 Main Text KV

- checkpoint frontier 是 token granularity，不按 page boundary 向下取整；
- boundary 与 current frontier 共享 allocation；
- restore 保留包含 boundary 的部分页，释放后续完整页，并把 logical valid frontier 截断到 boundary；
- suffix prefill 可以覆盖 boundary 之后原来属于 reasoning/answer 的位置；
- page mapping 本身不证明 continuation state 完整，planner 仍只能使用 current frontier 和 turn
  checkpoint。

### 9.2 GDN / Linear Attention

Qwen3.6 的 recurrent state 是无法仅从 KV prefix 得到的关键状态。capture 必须在 boundary 处理完成后
复制 running state，restore 必须在 prefill 新 suffix 前复制回 current slot。

当前 physical mapping 固定为：

```text
current_state_slot(lane)  = lane
turn_checkpoint_state_slot(lane) = C + lane
```

ordinary prefill/decode 在 current slot 原地推进。speculative target verify 只读 current slot 并写
Replay records；accepted-prefix Fold 也只更新 current slot。任何 speculative round 都不得把
`C+lane` 作为 Fold destination，因此在 `KeepExisting` 期间运行任意数量的 MTP/DFlash rounds 都不会改变
turn checkpoint。

这也是不能退化成 arbitrary LCP reuse 的原因：即使 token LCP 比零长，没有相同位置的 GDN
checkpoint 仍不能继续。

### 9.3 MTP

MTP restore 必须同时满足：

- turn-checkpoint hidden 有效；
- MTP prefix KV 至少覆盖 bridge 所需的 `frontier - 1`；
- restore 后按 `BeforeSuffix` bridge 重新建立 suffix；
- exact frontier append 继续使用现有 append/exact-hit bridge 规则；
- `KeepExisting` 必须保证 checkpoint 的 MTP prefix 仍可达；
- `CaptureNew` 只有在该次 prompt 确实执行了完整 MTP preparation 后才能原子发布 checkpoint。

`mtp_drafts` 是 current frontier 的下一轮 proposal continuation，不是 turn checkpoint payload。新请求
开始时可以清除 drafts；Restore 必须从 turn-checkpoint hidden 与 `B-1` MTP KV 重建，而不能依赖捕获时或后来
decode 产生的 drafts。ReplaySSM records 同样只描述 target GDN transitions，不保存 MTP private cache 或
proposal state。

任一条件不成立时，不允许只复用 Main Text KV；必须对整个 reuse path 降级。

### 9.4 DFlash

DFlash 当前只支持 35B-A3B text-only 路径。turn checkpoint 必须绑定：

- Main Text frontier；
- 共享 DFlash full-context paged KV 中仍可达的同一 prefix；
- DFlash `turn_checkpoint_local` cyclic-cache copy；
- atomic `TurnCheckpoint.frontier`。

这些状态必须指向同一个 token frontier。`KeepExisting` 保留 dedicated boundary state；
`RestoreTurnCheckpoint` 先恢复它再 prefill suffix；`CaptureNew` 在新 boundary 更新；状态不完整时
FullReset。当前 `local`、`dflash_context_frontier` 和 `pending_features` 可以随 speculative rounds 推进，
但不得覆盖 `turn_checkpoint_local`。terminal resolve 必须先 flush 已接受的 target features 并同步，随后才能把
sequence 标记为 retained。

### 9.5 ReplaySSM transaction 边界

ReplaySSM 对 MTP 和 DFlash 共用同一条 target-state 提交规则：

1. verify 读取每个 lane 的 committed current GDN state，按 compact batch row 写 Program-owned records；
2. `Pending` 期间，host execution frontier 和 current GDN 仍停留在本轮 base；
3. CPU 得到最终 accepted prefix 后，Fold rows 以 `record row -> lane` 映射写回各 lane 的 current slot；
4. 同一事务完成 hidden correction、MTP continuation publication 或 DFlash terminal flush；
5. stream 同步后才推进 ledger/KV/backend frontiers 并发布 retained state。

Turn checkpoint 不参与上述事务。它既不是 Fold source，也不是 Fold destination；records、row mapping、
pending target features 和本轮 proposal frame 都不得写入 checkpoint metadata。planner 禁止在 lane 仍为
`Pending` 时开始下一个请求，因此 Restore/Keep/Capture 永远只面对已经提交完成的 current state。

### 9.6 Vision 与位置

Vision prompt 的复用继续以 token types、三轴 positions、media grid/span 和 content digest 为准。

- 已复用 boundary 之前的 media 不重新编码；
- boundary 之后出现的新 media 由现有 Vision suffix plan 处理；
- placeholder 展开后的 boundary token index 必须与最终 MRoPE positions 使用同一个 token 序列；
- media digest 或 position identity 在 boundary 前变化时禁止 restore；
- Vision 与 MTP 组合必须同时满足两者的 checkpoint/bridge 约束。

---

## 10. 典型请求时间线

### 10.1 `false`：多步 tool loop 后出现新用户问题

```text
Request 1: U1 + generation opener
  FullReset
  CaptureNew(B1)
  generate A1 reasoning + tool call

Request 2: U1 + A1 + T1 + generation opener
  AppendAtFrontier
  KeepExisting(B1)
  generate A2 reasoning + tool call

Request 3: U1 + A1 + T1 + A2 + T2 + generation opener
  AppendAtFrontier
  KeepExisting(B1)
  generate A3 final answer

Request 4: U1 + A1(no reasoning) + T1 + A2(no reasoning) + T2
           + A3(no reasoning) + U2 + generation opener
  frontier mismatch
  RestoreTurnCheckpoint(B1)
  prefill rewritten previous-turn suffix + U2
  CaptureNew(B2)
```

稳定历史 `P` 位于 `B1` 之前，因此 Request 4 的成本不随 `P` 的总长度线性增长。

### 10.2 `true`：新用户问题继续 append

```text
Request N resident:
  P + U1 + all reasoning + all tool history + A_final

Request N+1 incoming:
  same prefix + U2 + generation opener

  AppendAtFrontier
  CaptureNew(B2) while prefilling U2/opener
```

旧 `B1` 在开始新用户轮次后不再是“当前轮次”的 checkpoint，因此由 `B2` 替换。后续 `U2` 的 tool
loop 再使用 `KeepExisting(B2)`。

### 10.3 模式切换

从长期 `true` 切换到 `false` 时，incoming prompt 可能删除多个旧轮次的 reasoning，最早变化点早于
当前 turn checkpoint。正确行为通常是 FullReset。反向切换并补回过去 reasoning 时也相同。

如果恰好渲染出完全相同的 token/position/media identity，则 exact matching 可以命中；runtime 不需要
仅因 flag 名称不同而强制 reset。

---

## 11. 错误、退化与可观测性

### 11.1 安全退化顺序

复用只能按以下顺序选择：

```text
AppendAtFrontier
    -> RestoreTurnCheckpoint
    -> FullReset
```

不能选择任意 token LCP。以下情况预期 FullReset：

- 会话中切换模式并改写了当前 checkpoint 之前的历史；
- 用户编辑、删除或重排了更早消息；
- reasoning/content/tool serialization 未精确回传且变化早于 checkpoint；
- reused media、positions 或 rope identity 不一致；
- MTP/DFlash checkpoint 不完整；
- retained state 已被容量策略驱逐。

FullReset 是正确性路径，不是服务错误。

### 11.2 日志和指标

request start JSONL 和 console log 至少记录：

- resolved `enable_thinking`；
- resolved `preserve_thinking`；
- Responses continuation 是否发生 preserve semantic change。

完成日志或内部测试诊断应区分：

```text
full_reset
append_frontier
restore_turn_checkpoint
```

并继续报告 `reused_prompt_tokens`。仅有 cache percentage 无法区分 append 与正确的 turn rewrite。

日志不得记录 reasoning 原文；该选项只增加 boolean 和 reuse-path metadata。

### 11.3 性能预期

在请求 history 精确回传且模式稳定时：

- `true`：跨用户轮次通常复用 current execution frontier，prefill 只覆盖新增用户 suffix；
- `false`：跨用户轮次复用 turn checkpoint，prefill 只覆盖刚关闭的一轮和新增用户 suffix；
- 同一轮 tool loop：两种模式都从 current frontier append，同时保持同一 turn checkpoint；
- 更早稳定历史不再因为一个普通新用户问题而被重复 prefill。

这里不规定固定 TTFT 数值；验收依据是 reuse path、reused token frontier 和处理 suffix 的范围。

---

## 12. 代码落点

| 模块 | 文件 | 必需变更 |
|---|---|---|
| Engine input | `include/ninfer/types.h` | 保留 `PromptOptions::preserve_thinking=false` |
| Chat renderer | `src/targets/qwen3_6/impl/frontend/chat_template.{h,cpp}` | 共享 last-query classifier；返回 render trace；记录首个开放轮次 assistant boundary |
| Vision processor | `src/targets/qwen3_6/impl/frontend/processor.{h,cpp}` | placeholder 展开时同步 boundary byte offset；返回最终 trace |
| Family frontend | `src/targets/qwen3_6/impl/frontend/frontend.cpp` | 将 byte offset 转成精确 token frontier；执行 prefix assertion |
| Prepared prompt | `src/targets/qwen3_6/export/ninfer/targets/qwen3_6/prepared_prompt.h` | `assistant_content_boundary` 改为 `turn_rewrite_boundary` |
| Runtime plan types | `src/targets/qwen3_6/impl/runtime/program.h` | 引入 `TurnCheckpointAction` 和 atomic `{valid,frontier}` metadata，删除 component validity |
| Request planner | `src/targets/qwen3_6/impl/runtime/request_plan_impl.h` | 按 §8.3 选择 reuse path 和 checkpoint action |
| Runtime transaction | `src/targets/qwen3_6/impl/runtime/program_impl.h` | 实现 Keep/Capture/Drop；移除无条件 boundary clear；保持 pending resolve 先于 retained publication |
| GDN slot ownership | `src/targets/qwen3_6/impl/runtime/linear_state_slots.h` | 保持现有 `2C` layout，将 boundary helper 改为 turn-checkpoint 命名，不增加 slot 或 per-sequence records |
| Prefill schedules | `text_*`、MTP、DFlash schedule 路径 | 在精确 frontier 捕获 `C+lane`、hidden、MTP bridge prefix 和 DFlash `turn_checkpoint_local` |
| Replay transaction | `program_impl.h`、`speculative_target_impl.h` | 保持 records 为 Program-owned pending-round scratch；Fold 仅写 current slot |
| Common serving request | `src/serve/request.h` | 增加 optional request override |
| Server config | `src/serve/serve_options.{h,cpp}` | 增加 default false 和 `--preserve-thinking` |
| Semantic resolution | `src/serve/generation_service.*`、`translate.*` | 一次解析后同时供 prompt、prepared request、logging 使用 |
| OpenAI Chat | `src/serve/openai_schema.cpp` | 解析 kwargs 和顶层别名，校验冲突和类型 |
| Responses | `src/serve/responses_schema.*`、`response_store.*` | 解析字段；存储/继承 resolved conversation semantic |
| Anthropic | `src/serve/anthropic_schema.cpp` | 原生 thinking 只控制 enable；解析独立 preserve extension |
| Request logs | `src/serve/request_log.*` | 记录 resolved semantic、semantic change 和 reuse path |
| Active docs | `docs/serving.md`、paged/concurrent/ReplaySSM runtime 文档 | 更新公开接口和 checkpoint 权威描述；修正“restore 后旧 boundary 必然被消费”的旧生命周期 |

改名应一次完成，不保留旧 project-owned 字段别名或双重 boundary 路径。

本设计不修改 Replay record format、Replay Fold 数学、record arena layout 或 `2C` GDN state layout；这些
当前实现已经满足 turn checkpoint 所需的物理隔离。

---

## 13. 测试与验证

### 13.1 Frontend exact tests

两种 target 共享 frontend，因此一组 family frontend fixture 保护语义：

1. `false` 在新用户消息出现后删除所有已关闭轮次 reasoning；
2. `false` 仍保留最后真实用户问题之后的 reasoning；
3. `true` 保留所有历史 reasoning；
4. `role=tool` 和 bare tool-response wrapper 不推进真实用户轮次；
5. 多个 assistant/tool 循环始终返回第一条 assistant 的 boundary；
6. 新真实用户问题将 boundary 移到新的 generation opener；
7. `enable_thinking=false` 不改变 boundary 定义；
8. `reasoning_content` 与从 content 中解析 `<think>` 的路径得到相同语义；
9. boundary prefix encoding 与完整 token 序列精确匹配；
10. image/video placeholder 位于 boundary 前时，展开后的 token frontier、token types 和 positions 正确。

测试应断言实际 token IDs 和 boundary，不用 rendered substring plausibility 代替 token-level exactness。

### 13.2 Planner/state-machine tests

覆盖下列 transition：

| Existing state | Incoming prompt | Expected reuse | Expected checkpoint action |
|---|---|---|---|
| empty | first user prompt | FullReset | CaptureNew |
| same open turn frontier | appended tool result | AppendAtFrontier | KeepExisting |
| same checkpoint, changed suffix | rewritten suffix | RestoreTurnCheckpoint | KeepExisting 或 CaptureNew，取决于 desired boundary |
| new user, `false` | previous turn reasoning stripped | RestoreTurnCheckpoint | CaptureNew |
| new user, `true` | history exact append | AppendAtFrontier | CaptureNew |
| desired boundary before base but unavailable | no usable earlier checkpoint | FullReset | CaptureNew |
| no generation boundary | boundary-less prompt | 按 token identity | Drop |

额外断言：

- `KeepExisting` 请求启动后 checkpoint 仍有效；
- 两次以上 tool-loop append 不改变 checkpoint frontier；
- speculative Fold 前后 GDN `C+lane` 内容不变，Fold destination 始终为 current `lane`；
- checkpoint 不拥有 Replay record row，下一 compact batch 可以安全覆盖 Program-owned records；
- Capture 只有在 prefill 成功后发布；
- Restore 后 trailing KV pages 被释放，partial boundary page 保留；
- prefix identity 在 restore/capture 后与新 prompt 一致。

### 13.3 MTP、DFlash 与 Vision real-engine tests

使用明确的本地 artifact 分别覆盖：

- 27B + MTP：至少两次 assistant/tool continuation 后提交新 user；`false` 必须从最早 turn
  checkpoint restore，而不是 cache=0；capture 与 restore 之间必须实际完成 ReplaySSM Fold；
- 27B + MTP：完整回传 reasoning 的 `true` 新 user 必须 append at frontier；
- 35B-A3B + DFlash text-only：重复上述 false/true 两条路径，验证 `turn_checkpoint_local` 在 current context
  推进和 ReplaySSM Fold 后仍可恢复，并验证 full-context frontier；
- 27B Vision：首轮 user media 位于 checkpoint 前，后续 tool loop 保留 checkpoint；新 user media
  只出现在 suffix Vision plan 中。

并发 fixture 至少使用一次非连续 lane membership，验证 compact record row 的 Fold mapping 不会写入任何
lane 的 `C+lane`。现有单次/长距离 DFlash boundary restore 可以保留，但不能替代多次 tool-loop
`KeepExisting` 的验证。

真实模型输出不是 token oracle。测试应把第一轮实际返回的 reasoning/content/tool fields 原样构造进
下一请求，再检查 `reused_prompt_tokens`、reuse path 和 state transition。

仓库中的固定消息 fixture 和自管理 serve 测试可分别运行 MTP 与 DFlash：

```bash
python3 tools/smoke/serve_thinking_preservation.py \
  --artifact out/qwen3_6_27b.ninfer --backend mtp

python3 tools/smoke/serve_thinking_preservation.py \
  --artifact out/qwen3_6_35b_a3b.ninfer --backend dflash
```

脚本通过真实 HTTP 请求检查多次 `RestoreTurnCheckpoint` 后的冷启动输出等价性、已关闭轮次在
`false/true` 下的 prompt token 差异、ReplaySSM speculative rounds、Responses 继承与显式语义切换。

### 13.4 Serving schema tests

OpenAI Chat 和 Responses：

- kwargs `true/false/null/omitted`；
- 顶层别名 `true/false/null/omitted`；
- 两处同值接受、冲突返回 400；
- 非 boolean、非 object、unknown kwargs 返回稳定错误；
- request override 高于 server default；
- Responses child 省略时继承 parent，显式切换时记录 semantic change；
- response output/history 保留 reasoning round-trip。

Anthropic：

- `thinking.type` 只影响 enable；
- `preserve_thinking` 独立解析并覆盖 server default；
- historical thinking block 转成 reasoning；redacted block 不伪造文本。

request-log tests 保护 resolved booleans 和 reuse path，不检查日志排版中的无关空格。

### 13.5 文档与静态检查

实现完成时至少运行：

- affected schema/unit tests；
- Qwen3.6 frontend tests；
- 27B MTP、35B DFlash 和 Vision 中与本设计直接相关的 real integration tests；
- `git diff --check`；
- active documentation link/stale terminology review，确保不再把 checkpoint 描述为“最新 assistant
  generation boundary”。

---

## 14. 实施顺序

1. **Frontend authority**：统一真实用户分类，产出 render trace 和精确 token boundary，完成 exact tests。
2. **Runtime lifecycle**：在现有 ReplaySSM `2C + records` 布局上引入 Keep/Capture/Drop，修正 planner 与
   request transaction，完成 state-machine tests。
3. **Backend completion**：让 MTP、DFlash、Vision 路径遵守同一 turn checkpoint contract，运行 real
   integration tests。
4. **Serving surface**：加入 server default、三种协议解析、Responses 继承和集中 semantic resolver。
5. **Observability**：补齐 resolved semantic 与 reuse-path logging。
6. **Active docs**：更新 serving、paged KV 和 concurrent runtime 权威文档，删除旧术语。

不应先只发布 server flag、把默认 `false` 的 FullReset 留给以后处理。该 flag 与正确的 turn
checkpoint lifecycle 共同构成本设计的完整行为。

---

## 15. 完成标准

满足以下全部条件才算实现完成：

- 通用 server 默认 `false`，agent 部署可在 server 或 request level 选择 `true`；
- `enable_thinking` 与 `preserve_thinking` 的四种组合均按定义工作；
- `true` 且 history 精确回传时，新用户轮次从 execution frontier append；
- `false` 的多步 tool loop 在新用户轮次从该轮第一条 assistant 的 turn checkpoint restore；
- tool loop 中 checkpoint 不被后续 generation opener 覆盖或清除；
- Replay records 始终是 Program-owned 单轮 scratch；Fold 只修改 current slot，不能进入或覆盖 turn
  checkpoint；
- MTP drafts 和 DFlash pending features 不属于 turn checkpoint；
- missing reasoning、history edit、mode switch 和 backend checkpoint 缺失不会错误复用；
- Main KV、GDN、MTP、DFlash、Vision/position state 在同一 frontier 上保持一致；
- OpenAI Chat、Responses、Anthropic 的请求解析、输出 round-trip、日志和文档一致；
- focused unit/schema/real-engine tests 通过，且 `git diff --check` 无错误。

---

## 16. 非目标与明确拒绝的替代方案

- **不强制所有模型永久保留 thinking。** 当前实现服务于 Qwen3.6 的两种官方语义；未来 target 必须按
  自己的模板和训练契约定义行为。
- **不在 history 缺失时从 cache 注入 hidden thinking。** 这会让模型实际输入偏离请求内容，也无法在
  stateless API 中证明会话身份。
- **不保留 reasoning 后面的旧 answer state。** 删除 reasoning 后 answer hidden/KV 已不等价，必须重算。
- **不增加每条 assistant 一份 checkpoint。** 下一用户轮次最早从本轮第一条 assistant thinking 开始
  改变，一份最早 turn checkpoint 已经是正确且最小的恢复点。
- **不实现 arbitrary LCP。** Qwen3.6 recurrent/backend continuation state 只在显式 checkpoint 完整。
- **不根据 cache 命中率改写 prompt。** prompt semantic 由 request/server 配置决定，cache 只能服从它。
- **不把 Anthropic `thinking.type` 当成历史保留开关。** 生成新 thinking 与保留旧 thinking 是两个维度。

---

## 17. 外部行为参照

本设计的默认值和请求级模板参数与以下实现方向一致，同时保留 NInfer 自己的完整 state checkpoint
约束：

- Qwen3.6 官方 chat template 与模型说明：
  <https://huggingface.co/Qwen/Qwen3.6-27B/tree/main>
- llama.cpp 的 reasoning preservation/template kwargs：
  <https://github.com/ggml-org/llama.cpp>
- vLLM 的 `chat_template_kwargs` 请求接口：
  <https://github.com/vllm-project/vllm>

这些引擎的 API 形态是互操作参考；NInfer 是否可以复用某个 prefix，仍由本文定义的
KV/GDN/MTP/DFlash/position 完整状态不变量决定。
