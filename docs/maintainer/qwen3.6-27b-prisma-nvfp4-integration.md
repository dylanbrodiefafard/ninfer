# Qwen3.6-27B Prisma NVFP4 混合权重集成设计

本文是 Qwen3.6-27B 混合权重路径的实施期权威。它定义如何在不增加 target、不改变
Qwen3.6 family 调度和模型数学的前提下，将 PrismaSCOUT 的 Text NVFP4/BF16 分配与
NInfer 已有的 Q6 vocabulary、MTP、Vision 和 optimized draft head 组合成第二套封闭的
`.ninfer` artifact profile。

本文不是当前已交付 artifact 的格式声明。实现合入前，
[`qwen3.6-27b-artifact.md`](qwen3.6-27b-artifact.md)、
[`tensor-formats.md`](tensor-formats.md) 和
[`storage-layouts.md`](storage-layouts.md) 仍描述唯一可用的 27B artifact。实现完成后，
Section 11 指定的稳定合同吸收本文内容，本文及其文档索引随即删除。

本文不重新定义模型数学、容器 framing 或通用 Op 资格规则；对应权威仍为：

- [`qwen3.6-27b-model.md`](qwen3.6-27b-model.md)；
- [`artifact-container.md`](artifact-container.md)；
- [`op-development.md`](op-development.md)。

## 1. 已冻结的产品决策

### 1.1 一个 target，两套封闭 artifact profile

新权重继续使用：

```text
model_id  = qwen3.6-27b
target key = qwen3_6_27b
```

不增加 target、model id、Engine option、CLI 选项、运行时格式菜单或另一条 Program。
`SequencePlan<Variant>`、`RequestPlan<Variant>`、Text/Vision/MTP schedule、状态事务、
prefix reuse 和 CUDA Graph orchestration 保持不变。变化只发生在：

1. converter 选择和编码权重；
2. 27B binder 验证完整 artifact profile 并建立 immutable weight views；
3. 现有语义 Op 为其已登记的 27B shape 增加 NVFP4 或 BF16 execution leaf。

本设计使用以下说明性 profile 名称；它们不写入 `.ninfer` root JSON：

| Profile | 含义 |
|---|---|
| current mixed-lowbit | 当前 Q4/Q5/Q6/W8/BF16/FP32 artifact |
| Prisma Text hybrid | 本文定义的 Prisma Text + NInfer 其余组件 |

Binder 通过完整 object inventory 和 format/layout signature 一次性识别两者。它只接受
两个完整 profile，不接受逐 tensor 任意混合，也不在请求路径中重新判断 profile。

### 1.2 组件级配方

| 组件 | Prisma Text hybrid 的来源和格式 |
|---|---|
| Text 大矩阵 | 精确保留 PrismaSCOUT 选择的 NVFP4 或 BF16 |
| Text norm、GDN convolution 和控制参数 | 保留 Prisma passthrough words；继续执行当前 BF16/FP32 artifact transform |
| token embedding | 从 base BF16 生成当前 `Q6G64_F16S` |
| full output head | 从 base BF16 生成当前 `Q6G64_F16S` |
| optimized draft head | 当前 131072-row `Q4G64_F16S` head 和 `I32` token-id map |
| MTP | 当前五个 `W8G32_F16S` 矩阵和七个 BF16 tensor |
| Vision | 当前 Q4/Q5/Q6/W8/BF16 配方 |
| frontend resources | 当前六个 byte-exact 资源 |

Prisma checkpoint 中已量化的 MTP 和 Vision 不进入本 profile。Converter 也不先将这些
NVFP4 tensor 解码为 BF16，再量化为 NInfer 的 W8/Q4/Q5/Q6；MTP 和 Vision 直接从 base
BF16 checkpoint 走现有 canonical encoder。

Optimized draft head 是完整 artifact 的固定组成，不依赖 Engine 启动时是否启用 MTP 或
optimized proposal head。

### 1.3 NVFP4 是原生表示，不是新的量化输入

对于选中的 Text matrix，converter 必须保留 upstream 已生成的：

- packed E2M1 weight words；
- 每 16 个 K-axis weight 的 E4M3FN block-scale words；
- FP32 `weight_global_scale` serialized divisor；
- FP32 `input_global_scale` serialized divisor。

不得从 Prisma NVFP4 重建 BF16 后再运行 NInfer encoder，也不得重新求 block/global
scale。允许的转换只有可证明无损的物理重排、行切分、行拼接和 layout repack。

E4M3FN 只承担 NVFP4 block scale，不是独立 FP8 weight matrix。因此本 profile 增加一个
NVFP4 persistent weight format，不增加可被普通 `linear` 独立消费的 FP8 matrix format。

Upstream 数学背景见
[NVIDIA Transformer Engine NVFP4](https://docs.nvidia.com/deeplearning/transformer-engine/user-guide/features/low_precision_training/nvfp4/nvfp4.html)；
compressed-tensors loader 明确把 checkpoint 中的 global-scale 字段当作 divisor，并在
加载时取倒数，见
[vLLM `CompressedTensorsW4A4Fp4`](https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/layers/quantization/compressed_tensors/schemes/compressed_tensors_w4a4_nvfp4.py)。
NInfer 的独立 decoder 必须按该序列化约定建立 oracle，不能凭字段名称推断乘除方向。

### 1.4 不变项

本工作不改变：

- tokenizer、chat template、media preprocessing 和 output semantics；
- Text/Vision/MTP 数学、层数、shape、row-view 语义和执行顺序；
- BF16/INT8 KV、FP32 GDN state 和 speculative accept/commit 语义；
- public Engine、CLI 和 OpenAI/Anthropic schema；
- 35B-A3B target；
- runtime weight repack policy：完整模型加载后不做第二份权重重排或反量化缓存。

## 2. Source identity 和已有证据

### 2.1 Base checkpoint

当前 27B artifact 的 base source 继续是：

```text
Qwen/Qwen3.6-27B
revision 6a9e13bd6fc8f0983b9b99948120bc37f49c13e9
```

Embedding、full head、MTP、Vision、frontend resources 和 optimized draft-head rows 均从
这一 source 产生，沿用当前 artifact reference 的 source mapping 和 encoder。

### 2.2 Prisma checkpoint

选中的 mixed-precision source 是：

```text
rdtand/Qwen3.6-27B-PrismaSCOUT-Blackwell-NVFP4-BF16-vllm
local inspection:
/path/to/home/models/llm/qwen/Qwen3.6-27B/vllm-nvfp4-bf16
```

发布 converter 前必须把实际使用的 immutable Hugging Face revision 写入 converter
配置和 conversion report；本地目录名不能成为发布 provenance。

2026-07-24 的本地审计得到：

| 项目 | 结果 |
|---|---:|
| Safetensors shards | 4 |
| four shard files total | 20,171,471,784 bytes |
| indexed tensors | 2687 |
| Prisma Text NVFP4 source linears | 379 |
| Prisma Text BF16 source linears | 117 |
| MTP | present |
| Vision | present |

Checkpoint 的 `quantization_config` 是 compressed-tensors mixed precision：
`nvfp4-pack-quantized`、G16、E4M3FN scale、W4A4 input activation。其
[`README`](https://huggingface.co/rdtand/Qwen3.6-27B-PrismaSCOUT-Blackwell-NVFP4-BF16-vllm)
报告 5.31 bpp、held-out KL 0.0151 和同 harness 的文本任务结果。这些结果支持采用
Prisma Text allocation，但不自动验证本文重新组合后的 Vision、MTP 或 NInfer kernel。

### 2.3 已确认的可组合性

本地逐 word 比较确认 Prisma 中的：

```text
model.language_model.embed_tokens.weight
lm_head.weight
```

与 base BF16 checkpoint 相同。Hybrid profile 仍从 base source 生成两者的 Q6 artifact，
以保持当前 converter provenance。

所有需要合并为一个 NInfer parent 的 Prisma source linears，在本地 checkpoint 中具有
相同的 `weight_global_scale` 和 `input_global_scale`：

- full-attention q/k/v；
- GDN qkv/z；
- MLP gate/up。

Converter 必须对每一个实际 parent 重做 exact FP32-word equality 检查。任一组不相等时
转换失败；不得取 `max`、平均、选择其中一个 scale 或重新量化来掩盖不一致。

## 3. Text 的精确格式分配

Full-attention layers 是：

```text
3, 7, 11, 15, 19, 23, 27, 31, 35, 39, 43, 47, 51, 55, 59, 63
```

### 3.1 Full attention

| Artifact object | BF16 layers | NVFP4 layers |
|---|---|---|
| `attention/query_key` | `3,7,11,15,19,23` | `27,31,35,39,43,47,51,55,59,63` |
| `attention/gate_value` | `3,7,11,15,19,23` | `27,31,35,39,43,47,51,55,59,63` |
| `attention/output` | `3,7` | `11,15,19,23,27,31,35,39,43,47,51,55,59,63` |

`query_key` 和 `gate_value` 继续使用当前逻辑 row order。Q projection 的
`[query_256,output_gate_256]` per-head rows 在 packed rows 和对应 block-scale rows 上
重排，不解码 weight value。

### 3.2 GDN

对于所有 48 个 GDN layer：

| Artifact object | Format |
|---|---|
| `gdn/query_key` | NVFP4 |
| `gdn/value_z` | NVFP4 |
| `gdn/output` | NVFP4，只有 layer 4 为 BF16 |
| `gdn/a_projection`, `gdn/b_projection` | BF16 |
| `gdn/convolution`, `gdn/norm` | BF16 |
| `gdn/a_log`, `gdn/dt_bias` | 按当前 artifact transform 从 BF16 source 扩展为 FP32 |

`gdn/value_z` 继续是一个 `[12288,5120]` parent，row order 为 `[value,z]`。其两个 source
共享 global divisors 后，converter 对 packed row 和 block-scale row 做无损拼接。

### 3.3 MLP

所有 64 个 Text layer 均使用：

| Artifact object | Format |
|---|---|
| `mlp/gate_up` | NVFP4 |
| `mlp/down` | NVFP4 |

Gate/up source 的 packed rows、block-scale rows 和相等的 global divisors 组成当前
`[gate,up]` parent，不进行重新量化。

### 3.4 Artifact-level count

Prisma 的 379 个 NVFP4 source linears 在完成现有 row fusion 后成为 305 个 NVFP4
artifact matrix objects：

| 来源 | NVFP4 artifact objects |
|---|---:|
| 64 × MLP gate-up/down | 128 |
| 48 × GDN query-key/value-z | 96 |
| GDN output，除 layer 4 | 47 |
| late full-attention query-key/gate-value | 20 |
| full-attention output，除 layers 3/7 | 14 |
| total | 305 |

相对于 current profile，Text 中 128 个 Q4 和 192 个 Q5 objects 被精确替换为 305 个
NVFP4 objects 和 15 个 BF16 matrix exceptions。除 Section 5 的 activation-scale
companions 外，Text physical matrix/object count 不变。

## 4. 保持当前 NInfer 格式的组件

### 4.1 Vocabulary matrices

| Object | Shape | Format |
|---|---:|---|
| `text/token_embedding` | `[248320,5120]` | `Q6G64_F16S` |
| `text/output_head` | `[248320,5120]` | `Q6G64_F16S` |

二者继续使用 `MAXABS_F16_RECIP_RNE_V1` 和 `row-split-k128-v1`。它们不使用 Prisma 的
BF16 payload 作为 artifact payload，也不改成 NVFP4。

### 4.2 Optimized draft head

以下两个 objects 保持当前定义：

| Object | Shape | Format |
|---|---:|---|
| `text/draft_head` | `[131072,5120]` | `Q4G64_F16S` |
| `text/draft_head_token_ids` | `[131072]` | `I32` |

Ranking source、forced special-token 规则、row gather 和 Q4 encoder 均沿用当前 artifact
reference。Draft head 从 base BF16 full head 生成，不从 Prisma full head 或 NVFP4
matrix 派生。

### 4.3 MTP

MTP 保持当前十二个 objects：五个 `W8G32_F16S` matrices 和七个 BF16 tensors。MTP
embedding、full head 和 optimized head 继续别名到 Text objects。

Prisma checkpoint 自带的 MTP NVFP4/BF16 allocation 不进入 hybrid artifact，也不要求
MTP Op 增加 NVFP4 route。

### 4.4 Vision

Vision 保持当前 333-object recipe：

| Role | Format |
|---|---|
| patch projection | `Q6G64_F16S` |
| block qkv/fc1 | `Q4G64_F16S` |
| block output/fc2 | `Q5G64_F16S` |
| merger matrices | `W8G32_F16S` |
| all remaining weights and biases | `BF16` |

Prisma Vision NVFP4 tensors不进入本 profile，Vision schedule 和 Ops 不因本工作变化。

## 5. NVFP4 persistent representation

### 5.1 Planned numeric identity

稳定格式注册使用 canonical identity `NVFP4G16`，其逻辑表示至少包含：

```text
weight code          E2M1, one 4-bit word per logical weight
block scale          E4M3FN, one 8-bit word per 16 K-axis weights
weight global field  one FP32 serialized divisor per complete matrix
zero point           none
```

对 checkpoint 字段 `weight_global_scale = d_w`，独立 decoder 的重建必须与
compressed-tensors/vLLM 相同，即先按其 E2M1/E4M3FN 规则重建 block value，再应用
`1 / d_w`。最终准确公式、E2M1 nibble mapping、E4M3FN word classes、非有限值规则和
round-trip examples 在实现合入时进入 `tensor-formats.md`；本文不允许 production kernel
反过来定义 oracle。

`input_global_scale` 不是 weight tensor 数值的一部分。它是 NVFP4 W4A4 compute profile
的 FP32 calibration divisor，按 Section 5.2 作为独立 tensor 保存。

### 5.2 Activation-scale companion

每个 NVFP4 weight object `P` 后紧邻一个：

```text
P/input_global_scale
shape  []
format FP32
layout contiguous-le-v1
```

它保存 source `input_global_scale` 的 exact FP32 word。完成 row fusion 后，一个 parent
只保存一个 companion；Section 2.3 的 equality check 保证该选择无损。305 个 NVFP4
matrix objects 因而增加 305 个 FP32 scalar objects。

Hybrid profile 的 tensor-format count 是：

| Format | Count |
|---|---:|
| `BF16` | 597 |
| `FP32` | 401 |
| `I32` | 1 |
| `Q4G64_F16S` | 55 |
| `Q5G64_F16S` | 54 |
| `Q6G64_F16S` | 3 |
| `W8G32_F16S` | 7 |
| `NVFP4G16` | 305 |
| total tensors | 1423 |

完整 artifact 另含当前六个 frontend resources，共 1429 objects。

### 5.3 Physical layout decision

Safetensors 的 row-major packed bytes 不是自动注册的 `.ninfer` layout。NVFP4 layout
必须与 sm_120a production kernels 一起确定，并满足：

- loader 可将 artifact payload 直接放入最终 device storage；
- 没有 load-time 或 first-run runtime repack；
- packed E2M1、E4M3FN block scale 和 FP32 divisor 可被独立 exact decoder 恢复；
- 当前所有 fused parent 支持连续 row view，`gdn/value_z` 的 `value`/`z` view 不复制；
- encoded size、plane offsets、alignment、K padding 和 padding words 有封闭公式；
- converter 的 source-to-layout repack 是 bit-preserving，而非 numeric re-encode。

在 production kernel 和 microbenchmark 选出最终 layout 前，不以 source convenience 冻结
layout identity。实现合入时必须先在 `storage-layouts.md` 中登记最终 identity 和 byte
contract，再允许 converter 产生 hybrid artifact。

## 6. Converter 和 binder 合同

### 6.1 Converter

Converter 对一个 hybrid artifact 执行：

1. 固定并记录 base 和 Prisma 两个 immutable source revisions；
2. 验证 Prisma config、manifest、所有必需 source names、shape 和 dtype；
3. 根据 Section 3 的闭合集合选择 NVFP4/BF16，不从 regex 或运行时 config 推导新组合；
4. 对每个 NVFP4 parent 验证 weight/input global-divisor words 相等；
5. 只进行 packed-row/block-scale-row 的 split、gather、reorder 和 concatenate；
6. 从 base source 生成 Q6 embedding/head、Q4 draft、W8 MTP 和当前 Vision；
7. 生成完整 inventory 后再写 `.ninfer`，不产生缺组件的中间产品 artifact；
8. 在 conversion report 中记录 source identities、profile、format counts、component bytes
   和全部 exact validation 结果。

Converter 不提供“量化 Prisma NVFP4”“将任意层切换到 NVFP4”或“采用 Prisma MTP/Vision”
的选项。

### 6.2 Binder

27B binder 先验证公共 identity、resources、names 和 shapes，再匹配一个完整 profile：

```text
current mixed-lowbit signature
or
Prisma Text hybrid signature
```

匹配必须覆盖每个 tensor format/layout、所有 NVFP4 activation-scale companions、object
count 和无多余 objects。一个 artifact 同时含 current Q4/Q5 Text matrix 和不属于
Section 3 例外的 NVFP4/BF16 Text matrix 时直接拒绝。

Container root 不增加 `profile` 字段；format/layout descriptors 已足以无歧义判定。
Profile label 只用于 converter report、diagnostics 和 artifact reference。

### 6.3 Immutable model view

Materialization 后，现有 semantic `Weight` view 能表达：

- NVFP4 code/scale/global-divisor payload；
- 关联的 FP32 input-global-divisor；
- shape、row view 和已选择的 execution leaf。

Profile dispatch 在 Program construction/materialization 阶段完成。Family schedule 不读取
profile string，不在每 token 路径做 converter-style判断，也不持有两套 resident Text
weights。CUDA Graph 捕获所选 leaf 的实际 kernels。

## 7. 需要增加的 Op execution leaves

现有语义 Op 保持名称和 public 数学合同。只为下列 27B shapes 和 production token
extents 增加 finite support：

| Op | 新 route |
|---|---|
| `attn_input_proj` | NVFP4/NVFP4 late layers；BF16/BF16 early exceptions |
| `gdn_input_proj` | NVFP4 query-key 和 value row view |
| `gdn_input_proj_conv_snapshot` | 同上，保留 convolution/state transaction |
| `linear` | NVFP4 `z` row view；现有 BF16 exceptions |
| `linear_add` | NVFP4 attention/GDN output 和 MLP down；layer-4 BF16 GDN output |
| `linear_swiglu` | NVFP4 MLP gate-up |

Embedding、full/optimized heads、MTP 和 Vision 继续调用现有 routes。

每个 NVFP4 Op 的 public input/output 保持 BF16。W4A4 activation conversion、E4M3 scale、
Tensor Core operand form、accumulator、split policy 和 fusion 是 execution-leaf 私有实现；
它们不能改变 Op observable semantics。独立 oracle 从 represented BF16 input 和
`NVFP4G16` decoded weights 计算完整 FP32 formula，production W4A4 route 使用与其数值
profile相称的 tolerance。

资格验证必须覆盖实际产品点：

- ordinary decode `T=1`；
- MTP target verify 的小 `T`；
- Text prefill 的实际 chunk extents；
- full-attention、GDN 和 MLP 的所有不同 `[N,K]`；
- row view、workspace reuse、CUDA Graph capture/replay 和地址稳定性。

NVFP4 activation scratch 进入现有 workspace planner，不允许 kernel 隐式分配 device
memory。是否需要新的 fused leaf 由包含该 Op 的实际 schedule measurement 决定，不由
模型名称创建平行 Op。

## 8. Embedding 和 lm_head 的 Q6 决策

### 8.1 Reconstruction evidence

对同一 base BF16 tensor 均匀抽取 4096 rows，比较 current NInfer Q6 artifact decode 与
标准 G16 NVFP4 reference encode/decode，得到：

| Tensor | Format | relative RMSE | relative MAE | mean row cosine |
|---|---|---:|---:|---:|
| `lm_head` | NInfer Q6 | 2.5898% | 2.7520% | 0.999619 |
| `lm_head` | NVFP4 | 9.4801% | 9.0144% | 0.995525 |
| embedding | NInfer Q6 | 2.6343% | 2.7323% | 0.999664 |
| embedding | NVFP4 | 9.4657% | 9.0273% | 0.995512 |

该实验只比较 tensor reconstruction，不是端到端质量评测。但它足以说明，在这两个
具体 tensor 上，current Q6 对 base BF16 的重建精度显著高于标准 NVFP4。Hybrid profile
保持 Q6 因而不会相对 current NInfer artifact 引入新的 embedding/head quantization
loss；这不等于声称 Q6 对 BF16 无损。

NVIDIA 的公开 Qwen3.6-27B NVFP4 recipe 对 `lm_head` 使用 NVFP4、保留 embedding，
说明 NVFP4 head 是可运行的整体路线，而不是它比本项目 Q6 head 更精确的 isolated
证据；参见
[`nvidia/Qwen3.6-27B-NVFP4` config](https://huggingface.co/nvidia/Qwen3.6-27B-NVFP4/blob/main/config.json)。

### 8.2 Decode cost

Current RTX 5090 profile 中，Q6 full head 的平均 kernel 时间约为 `0.624 ms`，payload
约 `0.993 GB`。同 shape NVFP4 payload 约 `0.715 GB`，按当前有效带宽估算的下界约为
`0.399 ms`。这说明 head 不是免费项，但预期整轮 decode 收益约为 `0.5%..1.5%`，不足以
在缺少本 checkpoint 隔离质量证据时覆盖 Q6 的明显 reconstruction 优势。Current
measurement 来源见
[`text_tg32_summary.md`](../../profiles/nsys/linear-architecture-20260716/release/text_tg32_summary.md)。

因此首个 hybrid artifact 固定使用 Q6 full head。完成整模型集成后仍需重新测量 head
占比，但该测量不把 NVFP4 head 变成本工作的隐藏 fallback 或第三套 profile。

## 9. Capacity 预估

以下是根据当前 artifact 和本地 Prisma payload 得到的 tensor-byte 预算；最终 layout
alignment 和 JSON directory 会造成小幅变化，因此它不是 container size 合同：

| Component | Bytes |
|---|---:|
| Prisma Text body，排除 embedding/head | 14,505,134,040 |
| Q6 token embedding | 993,280,000 |
| Q6 full output head | 993,280,000 |
| current MTP | 451,267,584 |
| current Vision | 295,711,648 |
| optimized draft head | 357,040,128 |
| hybrid tensor payload estimate | 17,595,713,400 |

Current artifact 的 tensor payload 是 `17,482,342,304` bytes。Hybrid 估算只增加
`113,371,096` bytes；完整 `.ninfer` 约为 `17.609 GB`。Prisma source checkpoint 的
20.17 GB 不能直接作为 hybrid residency 估计，因为它保留 BF16 vocabulary matrices，
并采用不同的 MTP/Vision 配方。

## 10. 验收条件

### 10.1 Representation 和 conversion

- `NVFP4G16` 有独立 bit-level decoder，覆盖 E2M1、E4M3FN、global divisor 和边界 words；
- source-to-`.ninfer` 对 packed codes、block scales 和两个 FP32 divisors 做 exact
  comparison；
- fused parents 的 row transform、global-divisor equality 和 companion mapping 全部
  exact；
- Q6 vocabulary、Q4 draft、MTP、Vision 和 frontend resources 通过现有 canonical
  converter checks；
- conversion report 完整解释两个 source 和每个 component 的来源。

### 10.2 Artifact 和 binding

- current 1118-tensor profile 继续被接受；
- hybrid 1423-tensor profile 被完整接受；
- 任意 partial hybrid、额外 object、错误 exception layer、错误 scale companion 或
  format/layout mismatch 被拒绝；
- 两个 profile 产生同一 Qwen3.6 semantic model view，不产生第二个 target 或 Program。

### 10.3 Numerical execution

- Section 7 的每个 production route 直接对独立 oracle；
- BF16 exception routes 与其 BF16 oracle 对比；
- ordinary Text、prefix reuse、MTP proposal/verify/accept 和 GDN state transaction 在
  hybrid artifact 上通过；
- Vision request 使用 hybrid artifact 中保持不变的 current Vision weights 完成；
- CUDA Graph capture/replay 覆盖 ordinary 和启用 MTP 的实际 graph frontiers。

### 10.4 Performance 和行为

- 在 RTX 5090/sm_120a 上分别测量关键 NVFP4 Ops 的 decode、小-T verify 和 prefill；
- 以相同 Engine options 比较两个 artifact 的 end-to-end Text prefill/decode 和 MTP；
- 报告实际 device residency、workspace 峰值和 Q6 head 占比；
- 运行一组代表性 Text、tool-use 和 multimodal behavior checks，目的仅是发现 conversion
  或集成回归，不把它表述为新的独立量化质量研究。

没有本地 measurement 时，不发布“NVFP4 一定更快”或具体端到端加速比例。Upstream
quality evidence支持 recipe 选择，本地 checks 负责证明 NInfer 忠实实现了所选 artifact。

## 11. 文档切换

实现完成并通过 Section 10 后，在同一变更中执行：

1. 更新 `tensor-formats.md`：注册 `NVFP4G16` 的完整数值语义、oracle 和 evidence，
   从 explicit exclusions 中移除 NVFP4；
2. 更新 `storage-layouts.md`：登记最终 kernel-native layout、encoded-size 和 exact
   decode；
3. 更新 `artifact-container.md`：只扩展已登记 format/layout identity 表，不增加 root
   profile 字段；
4. 重写 `qwen3.6-27b-artifact.md` 的单-profile 声明，以公共 topology 加两套完整封闭
   format signatures、source mappings、counts 和 conversion rules；
5. 更新受影响 Op headers/tests；`op-development.md` 的通用规则无需因 NVFP4 重复；
6. 只有在完成端到端 measurement 后才更新 `docs/performance.md`；
7. 删除本文，并从 `docs/README.md` 的 active implementation design 列表移除链接。

`qwen3.6-27b-model.md` 不变，因为 numeric recipe 和 execution leaf 不改变模型数学。
