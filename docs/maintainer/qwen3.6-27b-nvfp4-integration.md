# Qwen3.6-27B NVFP4 artifact 与类型登记计划

本文负责生产一个新的 Qwen3.6-27B NVFP4 `.ninfer` artifact，并同步登记消费该
artifact 所需的公共持久类型与内部 runtime weight representation。交付范围是：

1. 在 Python 与 C++ artifact registry 同时登记 `NVFP4` numeric format 和
   `blockscale-k16-m128x4-v1` storage layout；
2. 在内部 `QType`、`QuantLayout`、scale dtype 和 `Weight` 中登记完整的 NVFP4
   representation；
3. 新增独立的 NVFP4 inventory、recipe、converter 和 verifier；
4. 从固定的 base BF16 source 与 NVFP4 source 生成
   `out/qwen3_6_27b_nvfp4.ninfer`；
5. 对新 artifact 做结构、字节和来源验证；
6. 更新与新 artifact 直接相关的 maintainer 文档。

本文不修改 27B target binder、target registry、family schedule、workspace policy、
Engine 或公共 API，不增加 NVFP4 Op dispatch、execution leaf 或 kernel。完成本文后，
C++ generic reader 已能识别并验证新 artifact 的持久格式和 byte geometry，内部
`Weight` 已能完整表示后续 binder 所需的数据；当前 27B binder 仍只接受原 artifact
合同，因此本阶段不宣称新 artifact 可以被 Engine 加载或执行。

## 1. 原 27B artifact 完全冻结

已经发布的原 artifact 保持：

```text
converter module = tools.convert.qwen3_6_27b.convert
recipe id         = qwen3_6_27b-v2
output name       = qwen3_6_27b.ninfer
model_id          = qwen3.6-27b
object count      = 1124
```

本计划不：

- 重新生成、覆盖、重命名或重新上传原 artifact；
- 修改原 artifact 的 object inventory、format、layout、offset 或 payload；
- 修改 `convert.py`、`inventory.py`、`recipe.py`、`verify.py` 或 `draft_head.py`
  的原有行为；
- 提升原 recipe id；
- 把原 artifact 转换成 NVFP4；
- 让原 converter 接受第二种 recipe。

本阶段也不执行 Hugging Face 上传、替换 repository file 或修改已有发布记录。

公共 Python artifact codec 增加 NVFP4 支持后，必须继续通过现有 artifact 和 27B
converter tests；这些改动不能改变任何原 format/layout 的 encoded-size、编码结果或
解析结果。

新的 converter 只允许写入 basename 为 `qwen3_6_27b_nvfp4.ninfer` 的路径，从工具边界
排除误覆盖 `qwen3_6_27b.ninfer`。

## 2. 新 artifact 的固定身份

新 artifact 使用：

```text
converter module = tools.convert.qwen3_6_27b.convert_nvfp4
recipe id         = qwen3_6_27b_nvfp4-v2
output name       = qwen3_6_27b_nvfp4.ninfer
model_id          = qwen3.6-27b
object count      = 1371
```

原 artifact 与新 artifact 的项目自有命名差异只增加 `nvfp4` token。新 artifact：

- 不注册新的 target；
- 不改变 `.ninfer` v1 framing；
- 不在 root directory 增加 `profile`、`quantization` 或 `nvfp4` 字段；
- 不在 object names 中加入 source 方案名称；
- 由 tensor format、layout 和完整 object inventory 自描述其存储签名。

生成的 `.ninfer` 文件只包含 `model_id`、object directory 和 payload，不记录 source
repository 名称，因此文件内不出现外部量化方案名称。

`model_id` 只选择现有 `qwen3_6_27b` Package，不编码 Text weight storage。后续 C++
集成在该 target 内部只使用以下 storage-signature identity：

```cpp
enum class Qwen27TextStorage {
    Q4Q5Groupwise,
    Nvfp4BlockScaled,
};
```

该值不序列化、不成为 Engine option，也不进入请求路径。Loader 在读取 directory 后
一次性选择它，再由对应 binder 验证完整的 1124-object 或 1371-object signature；
不得按 object count 识别，不得先运行一个 binder、捕获失败后再尝试另一个 binder。
固定的非消费式 discriminator 是 `text/layers/0/mlp/gate_up` 的 format/layout：

```text
Q4G64_F16S + row-split-k128-v1       -> Q4Q5Groupwise
NVFP4 + blockscale-k16-m128x4-v1     -> Nvfp4BlockScaled
```

discriminator 只选择闭合合同，随后仍须逐对象完成全部 name、kind、shape、format、
layout 和无多余对象验证。

后续 C++ 集成固定由通用 `Binder` 提供按 name 非消费式读取 tensor descriptor 的只读
接口；`qwen3_6_27b` target 使用该接口完成上述选择。公共 target registry 仍只读取
`model_id`，不包含 27B object name、format 或 layout 语义。

Target 必须先完成 `plan_load` 和 storage-signature 选择，再为所选 storage 计算
sequence/workspace plan。该顺序只在 Engine 构造期选择准确的 execution-leaf workspace
合同，不要求两套 storage 永久预留最大 workspace，也不把 storage branch 带入请求路径、
family schedule 或 CUDA Graph。

## 3. 两个 source 的所有权

### 3.1 Base BF16 source

Base source 固定为：

```text
repository = Qwen/Qwen3.6-27B
revision   = 6a9e13bd6fc8f0983b9b99948120bc37f49c13e9
```

它是以下全部 payload 的唯一来源：

- NVFP4 source 选择为 BF16 的 Text linears；
- Text norm、GDN convolution、GDN control 和其他 direct tensors；
- token embedding；
- full output head；
- optimized draft head 和 token-id map；
- MTP；
- Vision；
- 六个 frontend resources。

从 base source 产生的 Q4/Q5/Q6/W8 objects 继续使用现有
`MAXABS_F16_RECIP_RNE_V1` encoder 和现有 layout。

### 3.2 NVFP4 source

NVFP4 source 固定为：

```text
repository = rdtand/Qwen3.6-27B-PrismaSCOUT-Blackwell-NVFP4-BF16-vllm
revision   = 9b5389d4a1e207daab2d47732efea57d7e946dcf
```

这两个 repository/revision 是 recipe-owned provenance constants，不由任意本地目录名、
CLI 参数或运行时 metadata 推导。规范本地 source 是可信工作流输入；converter 验证下述
config、index、tensor 和逐 word 内容合同，但不声称从一个普通复制目录中密码学证明其
远程 Hugging Face revision。Conversion report 同时记录 recipe 中的固定 repository、
revision 和实际本地路径，且 converter 不提供覆盖 repository/revision 的选项。

该 source 只提供：

1. Text linear 的 NVFP4/BF16 选择；
2. 被选择为 NVFP4 的 Text linear 的 packed E2M1 words；
3. 对应的 E4M3FN G16 block-scale words；
4. 对应的 FP32 `weight_global_scale` serialized divisor；
5. 对应计算站点的 FP32 `input_global_scale` serialized divisor。

它的 MTP、Vision、embedding、lm_head 和其他 passthrough payload 不进入新 artifact。
Converter 不读取这些对象作为 materialization source。

### 3.3 Source 组合条件

NVFP4 source 中保持 BF16 的 117 个 Text linears 全部从 base source 写入 artifact。
Converter 必须先比较两个 source 中这 117 个 tensor 的 name、shape、dtype 和每个 BF16
word；任何差异都使转换失败。

需要融合为一个 artifact weight parent 的 NVFP4 source tensors 必须具有 bitwise
相同的 `weight_global_scale`。检查覆盖：

- NVFP4 full-attention q/k/v；
- GDN value/z；
- MLP gate/up。

`input_global_scale` 按激活量化计算站点合并，而不按 artifact weight parent 保存。
同一站点所消费的全部 source projections 必须具有 bitwise 相同的
`input_global_scale`：

- full-attention input 站点合并 q/k/v；
- GDN input 站点合并 qkv/z；
- MLP gate-up input 站点合并 gate/up；
- full-attention output、GDN output 和 MLP down 各自为单一 projection 站点。

两个 divisor 都不允许选取其中一个值、取最大值、平均、重新标定或重新量化。固定
source 已审计 122 组多 source 组合的两个 FP32 divisor，均无 word mismatch；全部
379 个 source `input_global_scale` 依照上述规则无损合并为 247 个语义站点。

## 4. Text 的固定格式分配

Full-attention layers 为：

```text
3, 7, 11, 15, 19, 23, 27, 31, 35, 39, 43, 47, 51, 55, 59, 63
```

### 4.1 Full attention

| Artifact object | BF16 layers | NVFP4 layers |
|---|---|---|
| `attention/query_key` | `3,7,11,15,19,23` | `27,31,35,39,43,47,51,55,59,63` |
| `attention/gate_value` | `3,7,11,15,19,23` | `27,31,35,39,43,47,51,55,59,63` |
| `attention/output` | `3,7` | `11,15,19,23,27,31,35,39,43,47,51,55,59,63` |

`query_key` 与 `gate_value` 保持原 artifact 的 logical shape、semantic row order 和
aliases。Q projection 的 `[query_256, output_gate_256]` per-head rows 分别 gather
到原有 query 与 output-gate regions；同一 gather 同时作用于 packed-code rows 和
block-scale rows。

BF16 exceptions 从 base q/k/v/o tensors 完成相同 gather 和 fusion，保存为
`BF16`、`contiguous-le-v1`。

### 4.2 GDN

| Artifact object | Format |
|---|---|
| `gdn/query_key` | `NVFP4` |
| `gdn/value_z` | `NVFP4` |
| `gdn/output` | layer 4 为 `BF16`，其余为 `NVFP4` |
| `gdn/a_projection` | `BF16` |
| `gdn/b_projection` | `BF16` |
| `gdn/convolution` | `BF16` |
| `gdn/norm` | `BF16` |
| `gdn/a_log` | base BF16 source 按原 transform 扩展为 `FP32` |
| `gdn/dt_bias` | base BF16 source 按原 transform 扩展为 `FP32` |

`gdn/query_key` 保持 `[query,key]` row order。`gdn/value_z` 保持 `[value,z]` row
order 和 `[12288,5120]` parent。

### 4.3 MLP

64 个 Text layers 全部固定为：

| Artifact object | Format |
|---|---|
| `mlp/gate_up` | `NVFP4` |
| `mlp/down` | `NVFP4` |

`mlp/gate_up` 保持 `[gate,up]` row order。

### 4.4 数量

379 个 NVFP4 source linears 按原 fusion 形成 305 个 NVFP4 artifact matrices：

| 来源 | Artifact matrices |
|---|---:|
| 64 × MLP gate-up/down | 128 |
| 48 × GDN query-key/value-z | 96 |
| GDN output，排除 layer 4 | 47 |
| late full-attention query-key/gate-value | 20 |
| full-attention output，排除 layers 3/7 | 14 |
| total | 305 |

15 个 BF16 Text matrix exceptions 为：

- 6 个 early full-attention `query_key`；
- 6 个 early full-attention `gate_value`；
- layers 3/7 的 2 个 full-attention `output`；
- layer 4 的 1 个 GDN `output`。

不存在逐 tensor 可配置的第三种分配。

## 5. 继续使用原 recipe 的组件

以下对象的 names、shapes、formats、layouts、object order、source mapping 和 encoder
与原 artifact 相同：

- `text/token_embedding`：`Q6G64_F16S`；
- `text/output_head`：`Q6G64_F16S`；
- `text/draft_head`：`Q4G64_F16S`；
- `text/draft_head_token_ids`：`I32`；
- 全部 MTP tensors；
- 全部 Vision tensors；
- 六个 frontend resources。

Draft shortlist 继续从固定 ranking 和 base vocabulary 产生。Vision image/video
patch embedding、MTP embedding/head 及所有相关权重只读取 base source。

## 6. NVFP4 权重与输入标定

### 6.1 `NVFP4` numeric format

Python artifact registry 新增且只新增格式名：

```text
NVFP4
```

每个 logical weight block 沿 K 轴包含 16 个 E2M1 code，共用一个 E4M3FN scale word。
E2M1 nibble 的 bit layout 固定为：

```text
bit 3     sign
bits 2:1 exponent
bit 0     mantissa
```

正数 magnitudes 按 code `0..7` 解码为：

```text
0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0
```

完整 E2M1 decode 为：

```text
decode_e2m1(code) = (-1)^bit3 * magnitude(code & 0x7)
```

因此 `0x0` 与 `0x8` 分别是正零和负零。Artifact decoder 必须接受并保留全部 16 个
nibble words，不得重新量化、canonicalize signed zero 或把 E2M1 当作二补码 integer。

E4M3FN word 的 bit layout 固定为：

```text
bit 7      sign
bits 6:3   exponent e
bits 2:0   fraction m
bias       7
```

其 exact decode 为：

```text
e == 0, m == 0 : signed zero
e == 0, m != 0 : (-1)^sign * m * 2^-9
1 <= e <= 14   : (-1)^sign * (1 + m/8) * 2^(e-7)
e == 15, m < 7 : (-1)^sign * (1 + m/8) * 2^8
e == 15, m == 7: NaN
```

E4M3FN 没有 infinity，最大有限 magnitude 是 `448`。NVFP4 weight block scale 只接受
sign bit 为零的有限 word，包括正零；负值、负零和两个 NaN words 均使转换失败。

packed byte 的 low nibble 对应较小 K coordinate，high nibble 对应下一 coordinate。

令：

```text
c[n,k] = E2M1 code
s[n,g] = E4M3FN word, g = floor(k / 16)
d_w    = serialized FP32 weight_global_scale divisor
```

权重的 exact logical decode 为：

```text
W[n,k] = decode_e2m1(c[n,k]) * decode_e4m3fn(s[n,floor(k/16)]) / d_w
```

Converter 原样保存 `d_w` 的 FP32 word；`d_w` 必须是有限正数。NVFP4 weight payload
只包含 E2M1 code plane、E4M3FN block-scale plane 和这个 weight divisor。

### 6.2 `InputScaleDivisor`

Source 字段虽然名为 `input_global_scale`，实际保存的是公式中的 FP32 serialized
divisor `d_x`，不是执行端使用的乘法因子 `1 / d_x`。27B inventory 把它登记为语义
role `InputScaleDivisor`，其持久表示固定为：

```text
shape    []
format   FP32
layout   contiguous-le-v1
payload  source FP32 word 的 little-endian 原样表示，共 4 bytes
validity finite && > 0
```

Source 的 `[1]` tensor 转成 rank-0 scalar `[]`，不得改变 FP32 word。令该值为 `d_x`。
对于一个 K-axis 16-element 输入激活 block，`x` 是进入 projection Op 的 represented
BF16 activation。独立 oracle 将这些 BF16 values 精确提升为 FP32 后计算绝对值、
block maximum、global/block scaling 和 E2M1 conversion；它不从更高精度的隐藏值开始。

对有限 BF16 block，量化语义为：

```text
s_x       = E4M3FN(d_x * max_abs(block) / 6)
q_x       = E2M1(x * d_x / decode_e4m3fn(s_x))
decode(x) = decode_e2m1(q_x) * decode_e4m3fn(s_x) / d_x
```

`E4M3FN(y)` 固定为先把非负有限 `y` 饱和到 `448`，再执行 round-to-nearest-even
转换。`E2M1(y)` 固定为 NVIDIA E2M1 `round-to-nearest-even, satfinite` 转换：先把
有限输入饱和到 `[-6,6]`，在相邻 representable values 的 midpoint 选择最低保留
significand bit 为偶数的结果，并保留输入零的 sign。Midpoints 包括 magnitudes
`0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0`。

全零 block 固定产生正零 `s_x` 和正零 `q_x`。若非零 block 的 scale conversion 仍
round 为正零，则该 block 的全部 `q_x` 也固定为正零；不得执行除零。NVFP4 execution
leaf 的 qualified input domain 要求 represented BF16 activation finite。

其中 `s_x` 是每次执行按 block 产生的动态 E4M3FN scale；`d_x` 是 source 提供的静态
激活标定值。`d_x` 不属于 weight 的数值解码，不进入 NVFP4 weight payload，也不是
运行时更新的状态。Artifact 不保存 `1 / d_x` 或由 `d_x`、`d_w` 推导的融合系数；
执行端构造 immutable model view 时只加载两个 divisor fields，Op leaf 按 Section 6.3
从它们派生融合系数。

247 个 scalar 的名称、存在条件和数量固定为：

| Object name | Layers | Count |
|---|---|---:|
| `text/layers/{l}/attention/input_projection/input_scale_divisor` | `27,31,35,39,43,47,51,55,59,63` | 10 |
| `text/layers/{l}/attention/output_projection/input_scale_divisor` | `11,15,19,23,27,31,35,39,43,47,51,55,59,63` | 14 |
| `text/layers/{l}/gdn/input_projection/input_scale_divisor` | 全部 48 个 GDN layers | 48 |
| `text/layers/{l}/gdn/output_projection/input_scale_divisor` | GDN layers，排除 layer 4 | 47 |
| `text/layers/{l}/mlp/gate_up_projection/input_scale_divisor` | `0..63` | 64 |
| `text/layers/{l}/mlp/down_projection/input_scale_divisor` | `0..63` | 64 |
| total | | 247 |

同一计算站点只存一个 scalar；它可以同时约束多个 fused weight parents，不能复制成
每个 NVFP4 matrix 一个字段。

### 6.3 Artifact 与 runtime `Weight` 的边界

Section 6.1 和 6.2 规定持久化所有权，不规定加载后的 C++ object 拆分。Artifact 中
`d_w` 留在 NVFP4 parent payload，`d_x` 留在独立的站点级 FP32 object；后续 binder
必须把两者合并进每一份关联的 immutable NVFP4 runtime `Weight` view：

```text
artifact NVFP4 parent  -- code, block scales, d_w --+
                                                   +--> runtime NVFP4 Weight
artifact site scalar   -- d_x ---------------------+
```

NVFP4 runtime weight view 除 code/scale geometry 外必须直接暴露：

```cpp
float weight_scale_divisor = 0.0F; // d_w，来自 NVFP4 parent payload
float input_scale_divisor  = 0.0F; // d_x，来自站点级 FP32 object
```

`weight_scale_divisor` 是 NVFP4 weight representation 本身要求的成员；
`input_scale_divisor` 是相对于该 representation 唯一增加的站点派生成员。其他
`QType` 不读取这两个字段。默认零值对 NVFP4 是有意设置的非法 sentinel；后续
constructor/binder 必须在创建可消费的 NVFP4 `Weight` 时写入两个有限正数，不能把
缺失标定静默解释为 `1.0`。

本计划在 C++ 内部类型系统登记：

```text
QType::NVFP4
QuantLayout::BlockScaleK16M128x4
DType::FP8_E4M3FN
```

新枚举值只追加，不重排现有枚举值。NVFP4 `Weight` 的固定表示为
`qtype = NVFP4`、`layout = BlockScaleK16M128x4`、`group_size = group = 16`、
`scale_dtype = FP8_E4M3FN`、`qdata` 指向 code plane、`scales` 指向 swizzled scale
plane、`qhigh = nullptr`、`high_plane_bytes = 0`，并携带上述两个 divisor values。
`payload` 与 `payload_bytes` 仍描述包括尾部 `d_w` word 在内的完整 device payload。

后续 `plan_load` 必须在 `Binder::finish()` 和 `Reader` 生命周期结束前：

1. 从每个 NVFP4 parent payload 尾部 exact-read 并验证 `d_w`；
2. exact-read 并验证对应 `InputScaleDivisor` 的 `d_x`；
3. 把两个 host-side FP32 values 存入 target-private `BindingPlan`。

`InputScaleDivisor` object 只需 `ValidateOnly`，不需要独立 device allocation。Target
在 `construct_loaded_model` 阶段从 `BindingPlan` 将各 parent 的 `d_w` 和同一站点的
`d_x` 填入关联的全部 NVFP4 runtime weights：

| Site scalar | Runtime weights receiving the same `d_x` |
|---|---|
| attention input projection | `query_key`, `gate_value` |
| attention output projection | `output` |
| GDN input projection | `query_key`, `value_z`；其 `value`、`z` row views 继承 |
| GDN output projection | `output` |
| MLP gate-up projection | `gate_up` |
| MLP down projection | `down` |

因此 247 个持久站点 scalars 映射到 305 个 NVFP4 artifact weight parents：

```text
247 sites
+ 10 attention-input second parents
+ 48 GDN-input second parents
= 305 NVFP4 parents
```

加载后的每一份 NVFP4 `Weight` 都携带 `d_x`，但 artifact 不复制这 58 个重复值。
NVFP4 row view 只接受 `row_begin` 和 `row_count` 均为 128 的倍数。对于 parent 的
logical `K`，view 地址固定为：

```text
qdata  += row_begin * K / 2
scales += (row_begin / 128) * (K / 64) * 512
```

`value_z` 和其他 row-view construction 必须连同两个 divisor、logical shape 和全部
NVFP4 metadata 一起复制；不得复用假设 FP16 scale 按 row 连续的现有 row-view 地址
公式，也不得在运行时重新查找 artifact directory。

这项合并保持现有 Op 调用接口。不得新增独立的 `Nvfp4InputQuant` 参数或让计算图传递
artifact handle。`attn_input_proj`、`gdn_input_proj`、snapshot、`linear`、
`linear_add` 和 `linear_swiglu` 的 NVFP4 execution leaves 直接从 `Weight` 读取 `d_x`
与 `d_w`。Op wrapper 在 NVFP4 dispatch 时从这两个 immutable fields 计算
`1 / (d_x * d_w)` 并作为 leaf-private kernel argument 使用；它既不成为 artifact
object、`Weight` 的第三个派生 scalar field，也不成为新增 Op 参数。CUDA Graph 捕获
实际 kernel argument，replay 不重新执行 storage selection。Fused weights 的共同
`d_x` 来自同一个 site scalar，生产执行路径不比较 `model_id`、storage signature 或
object inventory。

同一 artifact 的 15 个 BF16 exceptions 还要求现有签名的内部支持域增加：

- BF16/BF16 `attn_input_proj`，消费 early `query_key` 与 `gate_value`；
- BF16 `linear_add`，消费 layers 3/7 attention output 和 layer-4 GDN output。

这些都是既有 fused Op 的 execution leaves，不增加 Op 参数、平行 Op、family schedule
分支或模型级计算图分支。

本节在本计划中只落地公共持久类型和内部 `Weight` representation；target-private
divisor extraction、row-view construction、Op validation/dispatch 和 kernel argument
传递仍属于后续 C++ 执行集成。

## 7. `blockscale-k16-m128x4-v1` layout

Python artifact registry 新增：

```text
blockscale-k16-m128x4-v1
```

唯一合法组合为 rank-two `[N,K]` 的 `NVFP4` tensor，并且必须满足：

```text
N > 0
K > 0
N % 128 == 0
K % 64 == 0
object alignment = 256 bytes
```

该 layout 不登记任意 shape padding 语义；不满足上述整除条件的 tensor 直接拒绝。
定义：

```text
code_plane_bytes   = N * K / 2
scale_plane_offset = align_up(code_plane_bytes, 256)
scale_plane_bytes  = N * K / 16

weight_global_offset = scale_plane_offset + scale_plane_bytes
encoded_bytes         = weight_global_offset + 4
```

Payload 固定为：

```text
row-major packed E2M1 code plane
positive-zero padding to scale_plane_offset
128-row × 4-scale swizzled E4M3FN scale plane
little-endian FP32 serialized weight-global divisor
```

305 个 NVFP4 matrices 全部满足该闭合 shape domain。

### 7.1 Scale swizzle

令：

```text
g          = floor(k / 16)
row_tile   = floor(n / 128)
row_inner  = n % 128
scale_tile = floor(g / 4)
scale_lane = g % 4
K_tiles    = K / 64
```

Scale word 在 scale plane 内的 byte offset 固定为：

```text
(row_tile * K_tiles + scale_tile) * 512
+ (row_inner % 32) * 16
+ floor(row_inner / 32) * 4
+ scale_lane
```

Converter 从 source 的自然 `[N,K/16]` scale rows 离线执行 bit-preserving swizzle。
该 byte order 对齐 NVIDIA 的
[Blackwell block-scaled scale-factor layout](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/blackwell_functionality.html#scale-factor-layouts)。

### 7.2 Fused row boundaries

新 artifact 保持原 fused parent 和 row aliases。全部 row boundaries 都是 128 的倍数：

| Parent | Row regions |
|---|---|
| attention query-key | `6144 + 1024` |
| attention gate-value | `6144 + 1024` |
| GDN query-key | `2048 + 2048` |
| GDN value-z | `6144 + 6144` |
| MLP gate-up | `17408 + 17408` |

Converter 对 code rows 与 natural scale rows 执行同一 logical row
gather/concatenate，再对完成 fusion 的 scale matrix 执行一次 swizzle。它不解码再编码
E2M1 或 E4M3FN words。

## 8. 完整 inventory

新 artifact 的 tensor counts 固定为：

| Format | Count |
|---|---:|
| `BF16` | 597 |
| `FP32` | 343 |
| `I32` | 1 |
| `Q4G64_F16S` | 55 |
| `Q5G64_F16S` | 54 |
| `Q6G64_F16S` | 3 |
| `W8G32_F16S` | 7 |
| `NVFP4` | 305 |
| total tensors | 1365 |

加上六个 frontend resources，共 1371 objects。

247 个新增 FP32 tensors 是 Section 6.2 的 `InputScaleDivisor` scalars。15 个新增
BF16 matrices 是 Section 4 的 exceptions。原 Text 的 128 个 Q4 和 192 个 Q5
matrices 被 305 个 NVFP4 matrices 与 15 个 BF16 matrices 替代，matrix object 数量
不变；新旧 tensor 数量之差完全来自 247 个激活标定 scalars。

全部原有 objects 保持原 topology order 和 object names；247 个 scalars 插入其所
约束的最后一个 weight object 之后，不允许在 artifact 尾部集中追加：

| Scalar | 固定插入位置 |
|---|---|
| attention input projection | `attention/gate_value` 之后 |
| attention output projection | `attention/output` 之后 |
| GDN input projection | `gdn/value_z` 之后 |
| GDN output projection | `gdn/output` 之后 |
| MLP gate-up projection | `mlp/gate_up` 之后 |
| MLP down projection | `mlp/down` 之后 |

因此同一激活计算站点有一个明确的 directory object，weight payload 无需携带或复制
输入标定。这是持久 object inventory；它不阻止 binder 按 Section 6.3 把 divisor
直接填入多个 runtime `Weight` views。

## 9. 实现范围

### 9.1 公共 artifact 工具

Python 公共工具修改：

- `tools/artifact/numeric.py`：登记 `NVFP4`，提供 Section 6.1 的 E2M1/E4M3FN exact
  word decode 与 stored-word validity helpers，不把它伪装成现有 integer
  `QuantFormat`；
- `tools/artifact/layouts.py`：实现 Section 7 的 geometry、encode/decode 和 exact
  scale swizzle；
- `tools/artifact/container.py`：通过公共 registry 规划和验证新 payload size；
- `tools/artifact/inspect.py`：能够报告新 format/layout，不解释模型语义；
- `tests/artifact/`：只增加 Section 10.1 规定的 numeric/layout/container tests。

### 9.2 C++ 持久 registry 与内部 `Weight`

本计划同步修改以下 C++ 内部边界：

- `src/artifact/reader.h`、`reader.cpp`、`storage_layouts.cpp`：
  - 登记 `NumericFormat::NVFP4` 和
    `StorageLayout::BlockScaleK16M128x4V1`；
  - 实现 format/layout 的 exact parse 与 name round trip；
  - 新增 block-scale geometry，公开 code bytes、scale offset/bytes、
    weight-divisor offset 和 encoded bytes；
  - 只接受 Section 7 的唯一 format/layout 组合与 shape domain，并继续使用 256-byte
    object alignment；
- `src/core/dtype.h`、`dtype.cpp`：追加 `DType::FP8_E4M3FN`，其 element size 为
  1 byte；
- `src/core/tensor.h`：追加 `QType::NVFP4`、
  `QuantLayout::BlockScaleK16M128x4` 和 Section 6.3 的两个 divisor fields；
- `src/artifact/typed_binding.cpp`：登记 NVFP4 的 layout 与 qtype 映射，但现有
  `materialized_weight` 必须明确拒绝构造 NVFP4 `Weight`。原因不是格式未知，而是
  合法构造还需要 target-private `BindingPlan` 提供已经校验的 `d_w` 与 `d_x`；
  不得把 NVFP4 错误地落入 RowSplit 构造；
- closed-enum switch 因新增 `QType` 需要调整时，`NVFP4` 只能落入既有的
  unsupported-format error，不增加任何 Op dispatch、workspace 分支或 execution leaf。

C++ generic reader 的支持边界到 descriptor、geometry、payload range 和
materialization 为止。它不解释 E2M1/E4M3FN 数值、不提取 divisor、不选择 27B storage
signature，也不把 artifact 变成可执行模型。

本计划不修改：

- `src/targets/qwen3_6_27b` 的 binding plan、binder 或 `LoadedModel`；
- family runtime、计算图、CUDA Graph、workspace 或 target registry；
- `include/ninfer/engine.h`、`include/ninfer/types.h` 或任何产品 API；
- Op contract、Op wrapper 的 NVFP4 admission、kernel 参数或 CUDA 实现。

本计划不新增或修改任何 C++ test。新增 C++ 内容没有新的可执行消费路径；为 enum
成员、`Weight` 私有字段、format-name round trip 或尚未被 target 使用的 geometry
增加测试只会重复 Python 持久格式测试并锁定内部结构。C++ 侧只编译受影响 targets，
用于发现 enum switch、声明和类型布局引起的编译错误。

### 9.3 新增 target-private modules

新增：

```text
tools/convert/qwen3_6_27b/convert_nvfp4.py
tools/convert/qwen3_6_27b/inventory_nvfp4.py
tools/convert/qwen3_6_27b/recipe_nvfp4.py
tools/convert/qwen3_6_27b/verify_nvfp4.py
```

它们复用现有 27B 的 base component builders、frontend loader、draft shortlist 和
canonical encoders，但不改变原 modules 的入口或行为。

本机规范生成命令为：

```bash
python3 -m tools.convert.qwen3_6_27b.convert_nvfp4 \
  --model /path/to/home/models/llm/qwen/Qwen3.6-27B/base-hf-bf16 \
  --nvfp4-model /path/to/home/models/llm/qwen/Qwen3.6-27B/vllm-nvfp4-bf16 \
  --out out/qwen3_6_27b_nvfp4.ninfer
```

Converter 不提供逐 layer override，不接受 NVFP4 MTP/Vision/head，也不从 BF16 source
重新量化出 NVFP4。

### 9.4 Preflight 与写入

创建输出文件前必须完成：

1. 读取 recipe-owned 固定 repository/revision，并验证两个本地 source 的 config、
   tensor index、shard 集合和所需 tensor content contract；
2. 验证 closed Text allocation；
3. 验证 379 个 NVFP4 source linears 的四类 fields；
4. exact-compare 117 个 BF16 Text linears；
5. 验证 122 组 fused-source weight-divisor equality；
6. 验证 379 个 source input divisors 到 247 个计算站点的完整映射和组内 word
   equality，并验证这 247 个站点对全部 305 个 NVFP4 parents 的闭合覆盖；
7. 验证所有 logical NVFP4 words；
8. 计算完整 1371-object inventory、encoded sizes 和 offsets；
9. 从 base source 计算 draft shortlist 并读取六个 frontend resources；
10. 验证输出 basename 为 `qwen3_6_27b_nvfp4.ninfer`；
11. 全部成功后才创建输出文件。

每个 NVFP4 weight parent 的 materialization 固定为：

1. 读取 source packed-code rows、natural scale rows 和 FP32 weight divisor；
2. 对 code 与 scale 应用相同 logical gather/reorder；
3. 按原 fused row order concatenate；
4. 执行 scale swizzle；
5. 写只包含 code、block scale 和 weight divisor 的 NVFP4 payload；
6. exact-compare source words、fused logical words 和 layout decode words。

Inventory 到达 `input_scale_divisor` object 时，converter 读取该站点的 source
`input_global_scale`；多 source 站点先完成 bitwise equality check，再将其中完全相同
的 FP32 word 写成一个 rank-0 scalar。不得把它写入 NVFP4 payload，也不得按 weight
parent 重复写入。

## 10. 生成物和验证

本计划生成且只生成：

```text
out/qwen3_6_27b_nvfp4.ninfer
out/qwen3_6_27b_nvfp4.ninfer.conversion.json
```

Conversion report 使用：

```text
model_id  = qwen3.6-27b
target_key = qwen3_6_27b
recipe_id = qwen3_6_27b_nvfp4-v2
```

Report 分别记录 `base_source` 和 `nvfp4_source` 的路径、固定 revision、object counts、
format/layout counts、final bytes 和转换耗时。外部 repository 的真实名称只允许作为
report provenance value；项目自有 field、module、type 和 artifact name 只使用
`nvfp4`。Report 中的 repository/revision 是固定 recipe provenance，不表述为从普通
本地目录反向证明得到的身份。

### 10.1 新增 Python tests

本计划只新增以下 tests：

1. `tests/artifact/test_nvfp4_numeric.py`
   - 对全部 16 个 E2M1 words 验证 exact value 和 signed-zero bit；
   - 对全部 256 个 E4M3FN words 用独立 scalar oracle 验证 finite、NaN、signed-zero
     分类与 exact value；
   - 验证 weight scale 只接受非负 finite words，且 `d_w` 只接受 finite positive
     FP32 words；
   - 用少量手写 words 验证
     `decode_e2m1(code) * decode_e4m3fn(scale) / d_w`，专门防止 divisor 被误写成
     multiplier。
2. `tests/artifact/test_layouts.py`
   - 增加一个 `[128,64]` known-vector case，同时验证 `4612`-byte geometry、packed
     nibble 顺序、若干跨 128-row/4-scale 边界的已知 swizzle offsets、尾部 `d_w`
     word 和完整 logical-word round trip；
   - 增加一个参数化 contract-rejection case，只覆盖 rank、`N % 128`、`K % 64` 和
     format/layout mismatch 四类边界。
3. `tests/artifact/test_container.py`
   - 只在现有 `test_v1_round_trip_covers_every_registered_storage` fixture 中加入一个
     NVFP4 object，复用同一测试验证 writer planning、reader reopen、payload range 和
     inspector format count；不另建重复的 container test。
4. `tests/targets/qwen3_6_27b/test_nvfp4_recipe.py`
   - 从独立列出的 layer sets 验证 305 个 NVFP4 parents、15 个 BF16 exceptions、
     247 个 site scalars 和 1371-object inventory；
   - 验证 247 个 sites 对 305 个 parents 的映射闭合且每个 parent 恰好被覆盖一次；
   - 验证 target-private `InputScaleDivisor` 只接受 finite positive FP32 words；
   - 验证 converter 在任何 source I/O 和文件创建前拒绝非
     `qwen3_6_27b_nvfp4.ninfer` basename，直接保护原 artifact。

这些 tests 不 mock Hugging Face shard，不生成缩小版伪模型，也不复制完整 1371-object
JSON snapshot。

### 10.2 生成时验证

以下检查属于一次真实 artifact 生产的 acceptance verification，不再各自复制成 unit
test：

- `py_compile`；
- 运行现有 `tests/artifact/`、原 27B converter tests 和 Section 10.1 的新增 tests；
- 117 个 BF16 Text linears 与 base 的逐 word equality；
- 379 个 source linears 到 305 个 artifact parents 的 codes、scales 和 weight
  divisors equality；
- 122 组 fused-source weight-divisor equality；
- 379 个 source input divisors 到 247 个 `InputScaleDivisor` objects 的完整映射、
  组内 equality 和逐 word equality；
- 247 个 `InputScaleDivisor` objects 到 305 个 NVFP4 parents 的完整一对多映射，且
  每个 NVFP4 parent 恰好被一个站点覆盖；
- 1371-object inventory、directory、payload ranges 和 final file size；
- 新 artifact 重新打开后对全部 payload 执行 `verify_nvfp4.py`；
- 编译受影响 C++ targets，但不新增、修改或运行 C++ tests。

### 10.3 明确不增加的 tests

以下 tests 对本次交付没有有效保护，不增加：

- C++ enum 数值、format/layout name、`Weight` 字段默认值、struct size 或
  `materialized_weight` rejection tests：它们只锁定内部表示，没有新的 C++ consumer；
- 只断言 Python registry 常量、dataclass 类型、`__all__` 或 module/file 存在的 tests：
  Section 10.1 的 layout/container tests 已经覆盖 registry 是否真正可用；
- 动态 activation quantize/dequantize、E2M1 rounding midpoint、全零 activation block
  或 `d_x` kernel arithmetic tests：本计划没有实现 activation quantizer 或执行端；
- 只有 encode→decode round trip、没有 hand-authored byte offsets 或独立 oracle 的
  tests：encoder 与 decoder 采用同一个错误 swizzle 时仍会通过；
- 缩小版 mock checkpoint、mock shard index 或 mock converter end-to-end tests：它们
  不能证明固定真实 source 的 tensor、fusion 和 divisor 关系；
- 完整 1371-object JSON snapshot、逐 report field snapshot 或对同一 mapping 的多份
  count tests：真实 artifact verifier 和一个独立 semantic inventory test 已覆盖；
- future binder selection、NVFP4 row view、Op dispatch、workspace、CUDA Graph 或
  kernel tests：相应实现不在本计划。

本计划不运行 Engine integration、model inference、CUDA Graph、kernel correctness
或性能测试，因为 target binder 与执行端没有接入 NVFP4。

## 11. 文档改动

生成与验证完成后，同一变更更新：

1. `tensor-formats.md`：登记 `NVFP4` exact stored numeric contract，并明确 activation
   calibration 不属于 NVFP4 weight numeric representation；不登记27B object role；
2. `storage-layouts.md`：登记 `blockscale-k16-m128x4-v1` byte contract；
3. `artifact-container.md`：说明 `.ninfer` v1 framing 不变，并列出新增 format/layout；
4. `qwen3.6-27b-artifact.md`：分别记录原 1124-object artifact 与新 1371-object
   NVFP4 artifact，明确原 artifact 未更新，并在该 target-private reference 中登记
   `InputScaleDivisor` 的 rank-0 FP32 representation、有效性、247-object inventory
   和 Section 6.3 的持久站点到 runtime `Weight` 映射约束；
5. `docs/README.md`：指向更新后的稳定 artifact 文档。

这些稳定文档必须分别标记四个事实：Python producer 已支持；NVFP4 artifact 的持久
格式合同已经登记；C++ generic reader 已接受并验证该 format/layout identity 与
geometry；当前 27B binder 和 Engine consumer 尚不支持新 artifact。不得把 reader
支持改写成 Engine 可加载或可执行。不得更新 CLI、serving 或 performance 文档来宣称
NVFP4 可执行。

稳定合同进入上述文档后删除本文及其索引，不保留已完成的临时计划。

## 12. 完成条件

只有同时满足以下条件，本计划才完成：

1. 原 converter、recipe 和 `qwen3_6_27b.ninfer` 未修改、未覆盖；
2. 新 Python format/layout codec 通过 exact tests；
3. C++ generic reader 已登记 NVFP4 format/layout/geometry，受影响 C++ targets
   编译通过，且没有新增、修改或运行 C++ tests；
4. 内部 `DType`、`QType`、`QuantLayout` 和 `Weight` 已完整登记 Section 6.3 的
   representation，且所有现有 Op 对 NVFP4 仍明确拒绝；
5. 四个 `_nvfp4.py` modules 完成并通过 tests；
6. `out/qwen3_6_27b_nvfp4.ninfer` 从两个固定 source 成功生成；
7. 新 artifact 的 1371-object signature、247 个 input divisors 和所有 NVFP4
   logical words 通过独立验证；
8. conversion report 完整；
9. Section 11 的稳定文档完成更新；
10. 本计划没有修改 target binder、family runtime、Engine/API、Op dispatch 或
    CUDA kernel，新 artifact 仍不能被当前 Engine 加载或执行。
