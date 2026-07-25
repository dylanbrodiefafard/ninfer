# Linear Q4 直接路由重构设计

## 状态与范围

本文是 `linear` 顶层 type dispatch 和 Q4 production 路径的当前实现权威。Q4 是后续
Q5、Q6、W8 可以遵循的完整范例。本文定义当前架构、一次 `linear` 调用的
流水线、每层唯一职责、当前保留的 Q4 routes、文件所有权、注册新 shape/route 的方式，
以及已删除旧路径的边界。

本次范围只改变：

- `linear()` 的公共语义校验和按 `w.qtype` 的直接转发边界；
- pure Q4 Linear 的 route selection、host launcher 和 kernel 组织；
- 其他 Op 对旧 Q4 plan/launch 私有接口的耦合。

Q5、Q6 和 W8 的内部 route/plan 重构不在本次范围。它们可以暂时保留现有 type-private
实现，但不得继续要求 `linear.cpp` 拥有它们的格式细节。NVFP4 也不在本次实现范围。

## 1. 冻结目标

重构后的 production Q4 路径只保留：

1. 一个语义入口 `linear()`；
2. 一次公共语义校验；
3. 一个按 `w.qtype` 的 closed switch；
4. 一个纯函数式 Q4 route selector：

   ```text
   select_q4_launch(N, K, T, policy) -> Q4Launch
   ```

5. 六个当前 route 表实际引用的 host launchers；
6. 三个 Q4 `__global__` kernel 定义；
7. 被这些 launchers 实际实例化的 schedule 和边界模板。

以下概念不进入最终 production 路径：

- `Q4Problem`；
- 独立 production admission；
- `Q4SupportSpec`；
- `Q4RouteSpec` 之上的 candidate legality；
- `Q4ScheduleId` 和 `Q4KernelVariant` 组成的 runtime plan；
- `Q4Plan`；
- `q4_rowsplit_resolve_plan()`；
- `q4_rowsplit_execute_plan()`；
- `q4_rowsplit_launch_fixed()`；
- `q4_rowsplit_launch_candidate()`；
- candidate registry 和 forcing path。

不增加多 layout、多 group size、通用 quantized backend、runtime registry、字符串 dispatch、
base class 或 graph IR。Q4 在当前产品中就是一个固定的 `Q4G64_F16S` +
`row-split-k128-v1` 实现。

## 2. 必须严格区分的概念

### 2.1 Route

一个 production route 是：

```text
(N, K, T, LinearPolicy) -> 一个 host launcher
```

route 的结果不是 kernel id、schedule id 或 plan。最佳路线已经在本架构落地前确定；运行时
只返回并调用最终 launcher。

### 2.2 Host launcher

host launcher 是一个普通 host 函数。它拥有：

- 选中 kernel family 的具体 schedule；
- CUDA grid、block 和 dynamic shared-memory 配置；
- Full/Predicated 边界实例选择；
- 为满足 CUDA grid 上限所需的内部切分；
- kernel 参数组装和 launch-error 检查。

host launcher 是 route 表的终点，也是 kernel mechanics 的唯一 host 所有者。

### 2.3 Schedule

`R4W1Direct`、`R1W8Direct`、`R8C4`、`R8C8`、`R64C64` 和 `R64C128` 是物理
schedule。它们描述 CTA/warp/tile、pipeline 和 memory access 组织，不是独立语义 Op，
也不是 `__global__` kernel 数量。

### 2.4 Kernel

pure Q4 Linear 只拥有三个 `__global__` kernel 定义：

1. `q4_rowsplit_gemv_kernel`；
2. `q4_rowsplit_gemm_simt_kernel`；
3. `q4_rowsplit_gemm_mma_kernel`。

模板参数形成的 Full/Predicated、不同 tile 或不同 epilogue 实例仍是这三个 kernel
定义的实例，不作为新的 kernel 资产计数。

## 3. 总体架构

### 3.1 请求外的一次性边界

`.ninfer` reader、target binder 和 immutable weight-view construction 在 Program 建立前
一次性保证：

- format 是 `Q4G64_F16S`；
- layout 是 `row-split-k128-v1`；
- group size 固定为 64；
- scale 是 FP16 multiplier；
- logical/padded shape、plane offsets、payload bounds 和 alignment 正确；
- `qdata`/`scales` 指向最终 device storage；
- row view 保留同一合法 layout。

production `linear()` 不在每次调用时重新证明这些 artifact-format 事实。项目的 converter、
reader、binder、checked view construction 和格式/codec 测试共同保护这条前置合同。

### 3.2 一次 pure Q4 Linear 调用的流水线

```text
target/family schedule
        |
        v
ninfer::ops::linear(x, w, out, policy, ws, stream)
        |
        v
validate_linear_semantics(x, w, out, policy)
        |
        v
switch (w.qtype)
        |
        +---- Q4G64_F16S ----> q4_dispatch(...)
                                  |
                                  v
                 select_q4_launch(w.n, w.k, x.ne[1], policy)
                                  |
                                  v
                          Q4Launch function
                                  |
                                  v
                     selected host launcher
                                  |
                                  +-- boundary instance
                                  +-- grid/block/shared memory
                                  +-- grid-limit slicing
                                  |
                                  v
             one of the three Q4 __global__ kernel definitions
```

流水线中不存在回到 admission、重新枚举 candidates、构造 plan 或执行 plan 的支路。

## 4. 各层明确定义

### 4.1 L0：语义 API

权威接口仍位于 `include/ninfer/ops/linear.h`：

```cpp
void linear(const Tensor& x, const Weight& w, Tensor& out,
            LinearPolicy policy, WorkspaceArena& ws, cudaStream_t stream);
```

语义保持：

```text
x   [K,T] BF16
w   [N,K] registered persistent weight
out [N,T] BF16

out[n,t] = BF16(sum_k decode(w[n,k]) * BF16(x[k,t]))
```

`LinearPolicy` 是许可，不是 kernel 选择命令。无 policy overload 精确等价于
`LinearPolicy::A16Only`。

当前所有已注册的 Q4/Q5/Q6/W8 pure Linear kernels 都以 16-byte vector access 读取
activation，并要求 output 起始地址保持相同对齐。因此 public Op 合同还要求 `x.data`
和 `out.data` 16-byte aligned；这是所有当前可达 quantized Linear routes 的共同动态
operand 条件，不是 Q4 format 事实。

### 4.2 L1：公共语义校验

`linear.cpp` 中的 `validate_linear_semantics()` 只验证所有 weight formats 共享的调用
合同：

- `x` 和 `out` 是 BF16；
- 二者是 contiguous 2-D matrix views；
- `w.n > 0`、`w.k > 0`；
- `x[K,T]`、`w[N,K]`、`out[N,T]` 的逻辑 shape 一致；
- `T > 0`；
- 调用所需的 view/data pointers 有效；
- `x.data` 和 `out.data` 均为 16-byte aligned；
- `policy` 是一个已定义的 enum value。

aliasing 继续是 contract precondition；热路径不增加通用 device pointer-range overlap
扫描。

这项动态对齐只在 L1 检查一次。当前注册的 `(N,K)` 都使按列切出的 `x/out` views 继续
保持 16-byte alignment，因此后续 Q4 launcher 和 grid slices 不重复检查。`qdata`、
`qhigh`、`scales` 等持久 weight planes 的格式专属对齐仍由 artifact/binder 的一次性
合同保证。

这一层不验证：

- Q4 layout/group/scale dtype；
- Q4 payload byte formula；
- Q4 padding；
- Q4 plane alignment；
- shape 是否在 Q4 production registry；
- 某个 policy 是否被 Q4 支持；
- 任何 kernel/schedule 条件。

### 4.3 L2：QType dispatch

公共校验后只保留 direct closed switch：

```cpp
switch (w.qtype) {
case QType::Q4G64_F16S:
    detail::q4_dispatch(x, w, out, policy, ws, stream);
    return;
case QType::Q5G64_F16S:
    detail::q5_dispatch(x, w, out, policy, ws, stream);
    return;
case QType::Q6G64_F16S:
    detail::q6_dispatch(x, w, out, policy, ws, stream);
    return;
case QType::W8G32_F16S:
    detail::w8_dispatch(x, w, out, policy, ws, stream);
    return;
default:
    throw std::invalid_argument("linear: unsupported weight qtype");
}
```

`linear.cpp` 不 include 各格式的 plan header，不读取各格式 layout metadata，也不选择
schedule。Q5/Q6/W8 可以在各自 dispatcher 后暂时适配旧实现；这不改变顶层所有权。

### 4.4 L3：Q4 direct selector

Q4 dispatcher 只做两步：

```cpp
using Q4Launch = void (*)(const Tensor&, const Weight&, Tensor&,
                          WorkspaceArena&, cudaStream_t);

void q4_dispatch(const Tensor& x, const Weight& w, Tensor& out,
                 LinearPolicy policy, WorkspaceArena& ws, cudaStream_t stream) {
    const Q4Launch launch = select_q4_launch(w.n, w.k, x.ne[1], policy);
    launch(x, w, out, ws, stream);
}
```

`WorkspaceArena&` 是已有 `linear` execution-resource 合同的一部分。当前六个 A16 Q4
launchers 不分配 workspace；以后某个经过选择的 A8 route 需要 scratch 时，可以在同一
launcher 签名下使用它，不需要给 route 表增加 plan 或 workspace 字段。

`select_q4_launch()` 的输入严格限制为：

- `N`；
- `K`；
- `T`；
- `LinearPolicy`。

它不读取 `Weight` payload、layout、group、scale pointer、padding 或 device capability。
产品只编译 `sm_120a`，因此没有 runtime GPU dispatch。

selector 使用显式的 exact-shape switch 和 shape-local T 分支。当前 policy selector
精确定义为：

```cpp
Q4Launch select_q4_launch(std::int32_t n, std::int32_t k,
                          std::int32_t t, LinearPolicy policy) {
    switch (policy) {
    case LinearPolicy::A16Only:
    case LinearPolicy::AllowA8:
        return select_q4_a16_launch(n, k, t);
    case LinearPolicy::AllowA4:
        break;
    }
    throw std::invalid_argument("q4 linear: unsupported policy");
}
```

当前没有 A8 kernel，因此 `AllowA8` 只是允许使用 A8，并不要求存在 A8 route；它与
`A16Only` 进入同一个 A16 selector。不能为了 policy 对称性复制一份相同 route 表，也不
创建空的 `select_q4_allow_a8_launch()`。

### 4.4.1 `select_q4_a16_launch()` 的完整内部结构

`select_q4_a16_launch()` 不读取表、不遍历 route specs，也不先做 admission。它按 `K`
外层、`N` 内层定位 exact shape，再用按 T 递增的条件直接返回 host function pointer。
实现应与下列控制流等价：

```cpp
Q4Launch select_q4_a16_launch(std::int32_t n, std::int32_t k,
                              std::int32_t t) {
    if (t <= 0) {
        throw std::invalid_argument("q4 linear: unsupported shape or T");
    }

    switch (k) {
    case 5120:
        switch (n) {
        case 1024:
            if (t == 1)  return launch_q4_gemv_r1_w8_direct;
            if (t <= 15) return launch_q4_simt_r8_c4;
            if (t == 16) return launch_q4_simt_r8_c8;
            return launch_q4_mma_r64_c128;

        case 4096:
            if (t == 1)  return launch_q4_gemv_r1_w8_direct;
            if (t <= 4)  return launch_q4_simt_r8_c4;
            if (t <= 16) return launch_q4_simt_r8_c8;
            return launch_q4_mma_r64_c128;

        case 6144:
            if (t == 1)  return launch_q4_gemv_r1_w8_direct;
            if (t <= 7)  return launch_q4_simt_r8_c4;
            if (t <= 16) return launch_q4_simt_r8_c8;
            return launch_q4_mma_r64_c128;

        case 7168:
            if (t == 1)  return launch_q4_gemv_r1_w8_direct;
            if (t <= 7)  return launch_q4_simt_r8_c4;
            if (t == 8)  return launch_q4_simt_r8_c8;
            if (t <= 15) return launch_q4_simt_r8_c4;
            if (t == 16) return launch_q4_simt_r8_c8;
            return launch_q4_mma_r64_c128;

        case 34816:
            if (t == 1)  return launch_q4_gemv_r1_w8_direct;
            if (t <= 4)  return launch_q4_simt_r8_c4;
            if (t <= 16) return launch_q4_simt_r8_c8;
            return launch_q4_mma_r64_c128;

        case 131072:
            if (t == 1) return launch_q4_gemv_r4_w1_direct;
            return launch_q4_mma_r64_c128;
        }
        break;

    case 2048:
        if (n == 131072) {
            if (t == 1) return launch_q4_gemv_r4_w1_direct;
            return launch_q4_mma_r64_c128;
        }
        break;

    case 1152:
        if (t < 4 || t > 131072 || (t % 4) != 0) {
            break;
        }
        switch (n) {
        case 3456:
            if (t <= 36)  return launch_q4_simt_r8_c4;
            if (t <= 320) return launch_q4_mma_r64_c64;
            return launch_q4_mma_r64_c128;

        case 4304:
            if (t == 4)   return launch_q4_simt_r8_c4;
            if (t == 8)   return launch_q4_simt_r8_c8;
            if (t == 12)  return launch_q4_simt_r8_c4;
            if (t <= 24)  return launch_q4_simt_r8_c8;
            if (t <= 320) return launch_q4_mma_r64_c64;
            return launch_q4_mma_r64_c128;
        }
        break;
    }

    throw std::invalid_argument("q4 linear: unsupported shape or T");
}
```

这里有意不抽取通用 `Range`、`ShapeSpec` 或 packed-key catalog：

- route 数量很少，直接控制流就是 production registry；
- 每个返回值都是完整 host launcher，不需要二次 schedule-to-launch 映射；
- Text/MTP 的 `T > 0` 与 Vision 的 `4..131072, step 4` 在 shape 分支中一次表达；
- 找不到 shape 或 T 只有同一个 `invalid_argument` 失败出口；
- 不存在“admission 成功但 resolve/legality 失败”的中间状态。

新增 shape 时，在对应 `K` case 下新增一个 exact `N` case；新增边界时，在该 case 中按
T 递增插入条件。只有真正出现已注册 A8 route 后，`AllowA8` 才改为调用
`select_q4_allow_a8_launch()`；该函数只直接列出 A8 覆盖点，其余点回落到
`select_q4_a16_launch()`，同样不引入 plan 或候选枚举。

### 4.5 L4：保留的 host launchers

最终只保留 route 表实际引用的六个 pure Linear host launchers：

```text
launch_q4_gemv_r4_w1_direct
launch_q4_gemv_r1_w8_direct
launch_q4_simt_r8_c4
launch_q4_simt_r8_c8
launch_q4_mma_r64_c64
launch_q4_mma_r64_c128
```

每个 launcher 对外都是一个完整 `Q4Launch`。SIMT/MMA launcher 在内部根据本次 tile
边界选择 Full 或 Predicated 模板实例；调用者和 route selector 看不到 variant。

launcher 不再执行公共语义校验或固定 Q4 format 校验。它只执行已经选中路线的 mechanics：

- 从 views 取得 pointers、logical dimensions 和 strides；
- 判断 Full/Predicated 模板实例；
- 计算 grid/block/shared memory；
- 必要时按 CUDA `grid.y <= 65535` 切分 T；
- launch；
- 检查 launch error。

grid-limit slicing 不重新查询 route。所有 slices 保持原 route 的 schedule；Full route 的
切分容量必须是 tile columns 的整数倍，保证每个 slice 仍可使用 Full 实例，最后一个
slice 由原始 T 的整除条件保证。

Full/Predicated 不是只看 T 的宽泛“边界优化”。四个 tiled launchers 必须使用以下精确
谓词：

```cpp
template <std::int32_t TileCols>
bool q4_simt_use_full(std::int32_t n, std::int32_t k, std::int32_t t) {
    return (n % 8) == 0 && ((k / 64) % 16) == 0 && (t % TileCols) == 0;
}

template <std::int32_t TileCols>
bool q4_mma_use_full(std::int32_t n, std::int32_t t) {
    return (n % 64) == 0 && (t % TileCols) == 0;
}
```

对应关系是：

- `launch_q4_simt_r8_c4` 使用 `q4_simt_use_full<4>()`；
- `launch_q4_simt_r8_c8` 使用 `q4_simt_use_full<8>()`；
- `launch_q4_mma_r64_c64` 使用 `q4_mma_use_full<64>()`；
- `launch_q4_mma_r64_c128` 使用 `q4_mma_use_full<128>()`。

每个 launcher 在切分前基于完整 `(N,K,T)` 只计算一次 `use_full`，随后所有 slices 都调用
同一个 `kernel<..., true>` 或 `kernel<..., false>` 实例。这保持当前执行行为；不能对
每个 slice 重新选择 variant。

具体后果不能被 T 整除关系掩盖：

- Vision `K=1152` 时 `(K/64)=18`，其 SIMT routes 即使 T 可被 4/8 整除也始终使用
  Predicated；
- Vision `N=4304` 不能整除 64，其 MMA routes 即使 T 可被 64/128 整除也始终使用
  Predicated。

这些谓词是选中 host launcher 内部的 kernel-boundary 选择，不是新的 admission、
legality API 或 route 层。

### 4.6 L5：三个 kernel families

#### GEMV

`q4_rowsplit_gemv_kernel` 只服务 `T=1`。当前保留两种已测 schedule：

- `R4W1Direct`；
- `R1W8Direct`。

两者都 exact-decode Q4/FP16 scale，使用 BF16 activation 和 FP32 FMA/累加。

#### SIMT GEMM

`q4_rowsplit_gemm_simt_kernel` 服务 small T。当前保留：

- `R8C4`；
- `R8C8`。

它 exact-decode Q4/FP16 scale 到 FP32 weight values，以 BF16 activation 做 FP32 FMA。
Full/Predicated 只是边界模板，不是 route。

#### MMA GEMM

`q4_rowsplit_gemm_mma_kernel` 服务中大 T。当前保留：

- `R64C64`；
- `R64C128`。

它把 decoded Q4 weights 转为 BF16 Tensor Core operand，activation 保持 BF16，通过
`mma.sync ... f32.bf16.bf16.f32` 使用 FP32 accumulator。

### 4.7 L6：固定 storage/decode primitives

`q4_rowsplit_storage.cuh` 只保留：

- G64、32 code bytes/group、2 scale bytes/group 常量；
- SIMT decode atom；
- MMA BF16-pair decode atom。

它不拥有 route、admission、shape 或 policy。

## 5. 当前 Q4 production registry

以下表是重构后必须保持的已测 winner。host launcher 是 selector 的返回值；schedule
单独列出，二者不是同一个概念，也都不是 runtime enum。

### 5.1 Text/MTP extents

这些 shape 接受每个正整数 T：

| `(N,K)` | T | host launcher | schedule |
|---|---:|---|---|
| `(1024,5120)` | `1` | `launch_q4_gemv_r1_w8_direct` | `R1W8Direct` |
|  | `2..15` | `launch_q4_simt_r8_c4` | `R8C4` |
|  | `16` | `launch_q4_simt_r8_c8` | `R8C8` |
|  | `17+` | `launch_q4_mma_r64_c128` | `R64C128` |
| `(4096,5120)` | `1` | `launch_q4_gemv_r1_w8_direct` | `R1W8Direct` |
|  | `2..4` | `launch_q4_simt_r8_c4` | `R8C4` |
|  | `5..16` | `launch_q4_simt_r8_c8` | `R8C8` |
|  | `17+` | `launch_q4_mma_r64_c128` | `R64C128` |
| `(6144,5120)` | `1` | `launch_q4_gemv_r1_w8_direct` | `R1W8Direct` |
|  | `2..7` | `launch_q4_simt_r8_c4` | `R8C4` |
|  | `8..16` | `launch_q4_simt_r8_c8` | `R8C8` |
|  | `17+` | `launch_q4_mma_r64_c128` | `R64C128` |
| `(7168,5120)` | `1` | `launch_q4_gemv_r1_w8_direct` | `R1W8Direct` |
|  | `2..7` | `launch_q4_simt_r8_c4` | `R8C4` |
|  | `8` | `launch_q4_simt_r8_c8` | `R8C8` |
|  | `9..15` | `launch_q4_simt_r8_c4` | `R8C4` |
|  | `16` | `launch_q4_simt_r8_c8` | `R8C8` |
|  | `17+` | `launch_q4_mma_r64_c128` | `R64C128` |
| `(34816,5120)` | `1` | `launch_q4_gemv_r1_w8_direct` | `R1W8Direct` |
|  | `2..4` | `launch_q4_simt_r8_c4` | `R8C4` |
|  | `5..16` | `launch_q4_simt_r8_c8` | `R8C8` |
|  | `17+` | `launch_q4_mma_r64_c128` | `R64C128` |
| `(131072,5120)` | `1` | `launch_q4_gemv_r4_w1_direct` | `R4W1Direct` |
|  | `2+` | `launch_q4_mma_r64_c128` | `R64C128` |
| `(131072,2048)` | `1` | `launch_q4_gemv_r4_w1_direct` | `R4W1Direct` |
|  | `2+` | `launch_q4_mma_r64_c128` | `R64C128` |

### 5.2 Vision extents

这两个 shape 的列轴是 raw-patch `P`，只接受 `4..131072` 内的 4 的倍数：

| `(N,K)` | P | host launcher | schedule |
|---|---:|---|---|
| `(3456,1152)` | `4..36`，step 4 | `launch_q4_simt_r8_c4` | `R8C4` |
|  | `40..320`，step 4 | `launch_q4_mma_r64_c64` | `R64C64` |
|  | `324..131072`，step 4 | `launch_q4_mma_r64_c128` | `R64C128` |
| `(4304,1152)` | `4` | `launch_q4_simt_r8_c4` | `R8C4` |
|  | `8` | `launch_q4_simt_r8_c8` | `R8C8` |
|  | `12` | `launch_q4_simt_r8_c4` | `R8C4` |
|  | `16..24`，step 4 | `launch_q4_simt_r8_c8` | `R8C8` |
|  | `28..320`，step 4 | `launch_q4_mma_r64_c64` | `R64C64` |
|  | `324..131072`，step 4 | `launch_q4_mma_r64_c128` | `R64C128` |

selector 本身表达这些合法集合；不另建一份 admission domain。

## 6. Production 闭包

production 不保留 schedule enum、candidate enum、legality matrix 或 forcing path。

当前 production binary 中每个 pure Linear Q4 launcher 和 default-epilogue
kernel 实例都必须能从 `select_q4_a16_launch()` 的至少一个 case 到达。没有 case 引用的
launcher、schedule alias、模板实例和 dispatch 辅助代码直接删除。fused Op 的
custom-epilogue 实例由它自己的 route 负责，不计入 pure Linear registry。

本架构落地时没有重新运行 benchmark、重新测量边界或重新选择 winner。Section 5 和
Section 4.4.1 的映射是既有测量结果；测试证明当前实现执行这份映射并满足数值合同。

`bench/ops/linear_op_bench.cu` 已移除 Q4 fixed-candidate/forcing options、解析、调用和
旧 header include；通过 public `linear()` 的普通 Q4 benchmark mode 保留。benchmark
source 必须可构建，但 route 重构不以重新测量性能为验收条件。

## 7. 当前文件组织

```text
include/ninfer/ops/linear.h

src/ops/linear/
├── linear.cpp
└── q4/
    ├── q4_dispatch.h
    ├── q4_dispatch.cpp
    ├── q4_launch.h
    ├── q4_rowsplit_storage.cuh
    ├── q4_rowsplit_gemv.cuh
    ├── q4_rowsplit_gemv.cu
    ├── q4_rowsplit_gemm_simt.cuh
    ├── q4_rowsplit_gemm_simt.cu
    ├── q4_rowsplit_gemm_mma.cuh
    └── q4_rowsplit_gemm_mma.cu
```

职责：

- `q4_dispatch.h/.cpp`：`q4_dispatch` 和 direct selector；
- `q4_launch.h`：`Q4Launch` 和六个保留 launcher 的 implementation-private declarations；
- 三组 `.cu/.cuh`：host launch mechanics、schedule aliases、模板实例和三个 kernel；
- storage header：固定 codec/decode primitives。

已删除的旧文件：

```text
q4_rowsplit_plan.h
q4_rowsplit_plan.cpp
q4_rowsplit_launch.h
q4_rowsplit_launch.cpp
q4_rowsplit_kernels.h
```

kernel template 使用 kernel-private `bool Full` 表达 Full/Predicated，由各 host launcher 选择模板
实例；不为此保留旧 header，也不再向 dispatcher 暴露 variant。

## 8. 与 fused Ops 的边界和具体迁移

Q4 kernel templates 还被 `attn_input_proj`、`gdn_input_proj`、
`gdn_input_proj_conv_snapshot` 和 `linear_swiglu` 复用。实现保留 kernel template 级复用；
这些 fused Ops 已按以下边界与 pure Linear route 解耦。

共同规则：

1. fused Op 不调用 `select_q4_launch()`；它的 output topology、epilogue、pitch、workspace
   和 Q5 companion route 都不同，不是 pure Linear route。
2. fused Op 可以保留自身确有必要的 route/workspace plan，但该 plan 不再包含
   `Q4Problem`、`Q4Plan`、`Q4ScheduleId` 或 `Q4KernelVariant`。
3. fused host launcher 可以直接实例化相同的 Q4 `__global__` kernel template 和 schedule
   type。
4. Q4 SIMT/MMA kernel template 参数统一改为 kernel-private `bool Full`；Full/Predicated
   决定只存在于拥有该 launch 的 host 函数内部。
5. target/family schedule 只 include semantic Op headers，不接触这些私有 launchers。

### 8.1 Attention input projection

`Q4Q5AttnInputPlan` 自身的两条 fused routes 保留，但移除：

- `Q4Q5AttnInputSubplans::query_key` 中的 `Q4Plan`；
- `Q4Q5AttnInputPlan::grouped_variant`；
- 对 `q4_rowsplit_resolve_plan()` 的调用。

`ParentSplitFixed` 的 Q5 subplan 可以继续归该 fused plan 所有。Q4 split-output 部分改由
`q4_q5_attn_input_small_t_launch()` 根据原始 T 直接选择：

| T | Q4 split-output schedule |
|---:|---|
| `1` | `R1W8Direct` |
| `2..7` | `R8C4` |
| `8` | `R8C8` |
| `9..15` | `R8C4` |
| `16` | `R8C8` |

该 launcher 内部使用 Section 4.5 的 SIMT `use_full` 谓词实例化
`q4_rowsplit_gemm_simt_kernel<..., true/false, SplitOutput=true, ...>`；其调用签名不再
接收 Q4 plan/variant。

`GroupedHomogeneousPairMmaR64C128` launcher 的调用签名也移除 variant。它在切分前以
原始 `T % 128 == 0` 计算一次 fused-local `bool Full`，内部完成 grid slicing，并调用
`rowsplit_grouped_mma_kernel<..., true/false, ...>`。

### 8.2 GDN input projection

`Q4Q5GdnInputPlan` 的两条 fused routes 和 Q5 subplan 保留，但移除：

- `Q4Q5GdnInputSubplans::qk` 中的 `Q4Plan`；
- `Q4Q5GdnInputPlan::grouped_variant`；
- 对 `q4_rowsplit_resolve_plan()` 和 `q4_rowsplit_launch_fixed_pitched()` 的调用。

`IndependentDirectFixed` 在 GDN-owned pitched-output launcher 中根据原始 T 直接选择：

| T | Q4 pitched-output schedule |
|---:|---|
| `1` | `R1W8Direct` |
| `2..4` | `R8C4` |
| `5..16` | `R8C8` |

该 launcher 自己组装 pitched `out_ld`，选择 Section 4.5 定义的 `bool Full`，并直接
实例化 Q4 GEMV/SIMT kernel template；pure Linear 不保留 pitched launcher。

`GroupedMixedMmaR64C128` launcher 的签名移除 variant。它在切分前以原始
`T % 128 == 0` 计算一次 fused-local `bool Full`，随后完成 slicing 和 grouped kernel
launch。

### 8.3 GDN conv snapshot

`q4_q5_gdn_input_conv_snapshot_launch()` 已经拥有 exact-T switch 和 custom epilogue，不
需要新增 selector。迁移只做两件事：

- 删除它对 pure Linear Q4 enum/header 的依赖；
- 将当前 Q4 small-T
  `q4_rowsplit_gemm_simt_kernel<..., Q4KernelVariant::Predicated, ...>` 实例改为
  `q4_rowsplit_gemm_simt_kernel<..., false, ...>`。

其 T=1 Q4 GEMV 和 custom-epilogue 模板实例仍由该 fused launcher 所有。

### 8.4 LinearSwiGLU

`Q4LinearSwiGluPlan` 的三条 fused routes 和真实 workspace bytes 保留，但移除：

- `Q4LinearSwiGluPlan::variant`；
- `Q4LinearSwiGluPlan::materialized_projection`；
- `materialized_plan()`、`tiled_variant()` 和所有 `Q4Plan` 比较。

`Materialized` route 在 workspace 中分配 contiguous `gate_up` 后，直接调用 A16
convenience overload `linear(x, w, gate_up, ws, stream)`，再调用 `silu_mul()`；不再解析或
执行 private Q4 plan。

`MmaSplitHalfPairR32C128` launcher 的签名移除 variant，并接收完整 Tensor views。它在
切分前以原始 `T % 128 == 0` 计算一次 fused-local `bool Full`，自己完成 grid slicing，
随后实例化 `q4_linear_swiglu_mma_split_half_pair_kernel<..., true/false>`。`GemvPair`
route 不变。

因此，Q4 kernel code 仍可被多个语义 Op 复用，但 pure Linear route registry
只有 `select_q4_a16_launch()` 一个所有者。

## 9. 注册新 shape

新增一个 Q4 Linear shape 必须按以下顺序完成：

1. 在 exact target artifact/model view 中确认真实 `(N,K)`、列轴语义和调用 policy。
2. 确认它使用既有 `Q4G64_F16S`/`row-split-k128-v1`，不为同一格式重复添加 runtime
   metadata。
3. 在 `select_q4_a16_launch()` 对应 `K` case 下增加一个 exact `N` case，直接写入已经
   决定的 T-to-launcher 映射。
4. 若映射引用新的 route，先按 Section 10 增加完整 host launcher。
5. 为每个 route boundary `b` 覆盖 `b-1/b/b+1`；对 step domain 只覆盖合法相邻点，并
   覆盖该 shape 的列轴上下界和非法点。
6. 每个实际返回的 launcher 都直接通过公共 FP64 Linear oracle test。
7. 更新本文 Section 5 的 active registry；不存在第二份 support table。

新增 shape 不需要：

- 新 `Problem` type；
- 新 admission API；
- 新 plan；
- 新 execute layer；
- 新 backend object；
- 复制一套 Q4 format/layout 校验。

## 10. 注册或替换 route

新增 route 的最小 production 单位是一个完整 host launcher，而不是裸 kernel id：

1. 明确实际 activation compute（A16 或 A8）和 kernel schedule；
2. 把 Full/Predicated、grid slicing 和 launch geometry 全部封装进 launcher；
3. 在对应 compute selector 的 exact shape/T case 中直接返回该 launcher；
4. 为该 launcher 增加公共 FP64 oracle test，并增加 selector function-pointer test；
5. 删除不再被任何 route 引用的旧 launcher、schedule aliases 和模板实例。

一个新 route 不给 `Q4Launch` 增加 `actual_compute`、schedule、workspace bytes、name 或
variant 字段。它的 function identity 已经完整确定执行 leaf。

## 11. 验证要求

该路径的结构或行为修改只以 build 和 tests 验收，不要求运行 benchmark 或建立性能
对比门槛；route winner 的改变属于独立的测量任务。

### 11.1 语义

- policy-bearing 和 A16 convenience overload 都至少有一个 public Op 数值 case；
- 六个 retained host launchers 各有至少一个真实 production point 直接对独立 FP64
  oracle；
- 可合理执行的尺寸检查完整 output、guards、输入/weight preservation；
- 大 N/T case 使用 deterministic sampled-output FP64 oracle 和 guards，不为测试分配
  超出产品机器实际能力的大矩阵；
- `AllowA8` 当前与 A16 selector 相同，因此只需一个 public `AllowA8` 数值 case 证明
  policy 透传；不把相同数值测试按 policy 重复一遍；
- misaligned `x.data` 或 `out.data` 在 L1 失败；
- `T=0` 按新合同失败，不再保留旧 empty-call no-op；
- malformed Q4 group/layout/padding/weight-plane alignment 由 artifact/binder tests 保护，
  不再作为 `linear()` 的 runtime-validation test。

### 11.2 Route registry

private selector test 不分配 device tensor，只保护真实 production registry：

- 每个注册 shape/T/policy 返回预期 launcher；
- unsupported shape、T 和 policy 直接失败；
- 所有 Section 5 ranges 精确覆盖且无洞；
- Text/MTP 覆盖每个 seam、普通 prefill 点和大于默认 chunk 的代表点；
- Vision 覆盖 `P=4`、`P=131072`、4-step 约束、所有 seams 及相邻非法点；
- 每个保留 launcher 至少被一个 production point 引用。

它不恢复 candidate legality matrix，也不把 selector test 当作数值资格替代品。

### 11.3 路径与集成测试

- 用 launcher function pointer 断言代表点实际进入 Section 4.4.1 指定的 route；
- 为两个 GEMV、两个 SIMT、两个 MMA launchers 各覆盖至少一个数值 case；
- SIMT 和 MMA 各用可执行的 production points 覆盖 Full 与 Predicated；特别覆盖
  `K=1152` SIMT 必须 Predicated，以及 `N=4304` MMA 必须 Predicated；
- 不构造无法装入 RTX 5090 的真实大矩阵来触发 CUDA grid limit；单独对
  `for_each_token_slice()` 做纯 host 边界测试，覆盖 limit、limit+1 和多 slice；
- 运行受影响的 fused Op tests，证明它们已切断旧 `Q4Plan` 接口且结果正确；
- 在一个有代表性的 public Q4 route 上运行 CUDA Graph replay test。

## 12. 已落实的迁移边界

当前实现已经：

1. 在 public Linear 边界统一 BF16/shape/contiguous/non-null/16-byte `x/out` alignment
   校验，并按 QType 直接转发；
2. 以 `Q4Launch`、六个完整 host launchers 和 direct selector 取代 pure Q4
   problem/admission/legality/plan/execute/candidate 层；
3. 把 Full/Predicated 和 grid slicing 收入各自 host launcher；
4. 让 Q5/Q6/W8 通过 type-private dispatcher adapter 接收 policy 和 workspace，同时保留
   它们当前各自的内部实现；
5. 让 Attention input、GDN input、GDN conv snapshot 和 LinearSwiGLU 只保留
   fused-local route，不依赖 pure Q4 plan/enum/fixed-pitched launch；
6. 用 selector registry、public FP64 oracle、alignment、token-slice、CUDA Graph 和 fused
   Op tests 替换旧 Q4 candidate/plan/dispatch tests；
7. 从 CMake、benchmark source 和 production source 删除旧 Q4 plan/launch/candidate
   files、symbols 和 forcing options。

不保留旧接口 alias、compatibility wrapper 或两套并行 Q4 路径。

## 13. 完成条件

Q4 范例只有在以下条件全部满足时完成：

- production 调用链与 Section 3.2 完全一致；
- `linear.cpp` 不含 Q4 format/layout/route 细节；
- L1 统一检查所有当前 quantized Linear routes 共有的动态 `x/out` 16-byte alignment；
- selector 只读取 `(N,K,T,policy)`；
- production 没有 `Q4Problem`、admission、legality、`Q4Plan` 或 execute-plan；
- route 表只引用经过实测的六个 launchers；
- pure Q4 仍只有三个 `__global__` kernel 定义；
- Full/Predicated 使用 Section 4.5 的精确谓词，且和 grid slicing 一样对 selector 不可见；
- fused Ops 不依赖 pure Linear plan/enum/variant；
- selector tests 覆盖所有 registered shapes、policies 和 route boundaries；
- 可执行的完整/采样 oracle cases 覆盖六个 retained launchers 及 Full/Predicated；
- 相关 route、数值、fused Op 和 Graph replay tests 通过；
- 旧文件和旧 CMake/test entries 已删除；
- 可选 benchmark source 不再引用已删除的 Q4 fixed-candidate API，且本次没有运行
  benchmark。

完成后，Q5、Q6 和 W8 可以分别采用相同的“公共语义校验 → type dispatcher → direct
selector → 完整 host launcher → kernel”所有权模型，但不得通过为它们建立一个通用
backend framework 来复用本范例。
