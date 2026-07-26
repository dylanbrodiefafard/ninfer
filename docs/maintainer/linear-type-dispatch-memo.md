# Linear 直接路由架构与注册规则

## 状态与范围

本文是 `linear` 顶层 type dispatch 以及 Q4、Q5、Q6、W8 production 路径的当前实现
权威。本文定义当前架构、一次 `linear` 调用的流水线、每层唯一职责、各格式 direct
selector 与 host launcher 的所有权、注册新 shape/route 的方式，以及已删除旧路径的
边界。

当前实现已经统一：

- `linear()` 的公共语义校验和按 `w.qtype` 的直接转发边界；
- pure Q4/Q5/Q6/W8 Linear 的 route selection、host launcher 和 kernel 组织；
- fused Ops 对旧 pure Linear plan/launch/candidate 私有接口的耦合；
- BF16 未实现状态的表达。

`BF16_CTRL` 当前没有 pure Linear kernel 或注册 shape，因此公共 `linear()` 明确拒绝；
不保留空 selector、admission 或 plan 占位。NVFP4 不在当前实现范围。

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

ideal[n,t] = sum_k decode(w[n,k]) * represented_bf16(x[k,t])
out         = BF16 storage approximation of ideal
```

统一 CPU oracle 保留 `ideal` 的 double 累加结果；测试将 GPU 的 BF16 输出提升后直接与其比较。
最终输出存储舍入属于 A16/A8 compute profile 的验收判据，不在 oracle 内复刻。

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
schedule。Q4/Q5/Q6/W8 各自的 dispatcher 都直接调用 type-private selector，随后调用
返回的完整 host launcher。

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
Section 4.4.1 的映射是既有测量结果；selector 是该映射的 executable authority，public
数值测试只证明通过该入口可达的实现满足统一 Linear 数值合同。

`bench/ops/linear_bench.cu` 只通过 public `linear()` 测量 production route。它不 include
private dispatch/launcher/plan header，不提供 fixed candidate 或 forcing，也不复制
launcher tile、tail、weight replay 或 executed-FLOP 信息。单点、连续 T sweep 和
27B/35B suite 共用同一执行路径；NCU profile mode 只捕获一次 public Linear 调用。
吞吐参照固定使用 RTX 5090 的 `1792 GB/s` DRAM 和 `209.5 TFLOP/s` dense BF16 Tensor
Core 规格，不运行同进程 copy 或 Tensor Core peak probe。

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
5. 通过 public `linear()` 为每个 route boundary `b` 数值覆盖 `b-1/b/b+1`；对 step
   domain 只覆盖合法相邻点，并覆盖该 shape 的列轴上下界。
6. 每个实际返回的 launcher 都由相应 public case 直接对同一个 CPU FP64 GEMM oracle
   资格化；测试不复制 selector 或断言 function pointer。
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
4. 为该 launcher 增加通过 public `linear()` 执行的统一 CPU FP64 GEMM oracle case；
   若它建立新 route boundary，同时覆盖 `b-1/b/b+1`；
5. 删除不再被任何 route 引用的旧 launcher、schedule aliases 和模板实例。

一个新 route 不给 `Q4Launch` 增加 `actual_compute`、schedule、workspace bytes、name 或
variant 字段。它的 function identity 已经完整确定执行 leaf。

## 11. 验证要求

该路径的结构或行为修改只以 build 和 tests 验收，不要求运行 benchmark 或建立性能
对比门槛；route winner 的改变属于独立的测量任务。

### 11.1 语义

- policy-bearing 和 A16 convenience overload 都至少有一个 public Op 数值 case；
- 六个 retained host launchers 各有至少一个真实 production point 通过 public
  `linear()` 对同一个 CPU FP64 GEMM oracle；
- Q4 test fixture 从同一组 signed codes 和 stored FP16 scales 同时构造完整 GPU packed
  payload 与 CPU 使用的 logical float weight；
- oracle 只接收 float weight、BF16 activation 所表示的 float 值和 `(N,K,T)`，每个 dot
  使用 naive double accumulation；它不读取 Q4 payload，也不模拟 BF16/MMA、staging、
  reduction tree 或 BF16 output rounding；
- 可合理执行的尺寸检查完整 output、guards、输入/weight preservation；大 N/T case
  检查完整 output 写入和 guards，再把确定性采样行列交给同一个 GEMM；
- A16 suites 只使用 `A16Only`；未来实际接入 A8 route 时，由独立 A8 suite 使用
  `AllowA8` 和 A8 criterion，不让同一测试在 route 演进后静默改变 compute path；
- malformed Q4 group/layout/padding/weight-plane alignment 由 artifact/binder tests 保护，
  不再作为 `linear()` 的 runtime-validation test。

### 11.2 公共脚手架与数值判据

`tests/ops/linear/linear_test_common.{h,cpp}` 唯一拥有：

- Q4/Q5/Q6/W8 显式模拟权重生成器；
- packed payload 与被选 oracle rows 的 logical float materialization；
- `cpu_linear_gemm_fp64()`；
- output poison、完整写入、guards、确定性采样和 public call mechanics；
- `ActivationCompute -> ReductionCriterion` 集中映射，并复用公共 reduction 比较逻辑。

CPU GEMM 可以按输出行多线程并对 T 小块化以复用 weight load，但每个输出始终由一个
线程按 `k=0..K-1` 顺序 double 累加。它不使用 BLAS、production decoder、production
kernel、K 维树形归约或 fast-math。

每个 weight/activation suite 只能为整个 suite 选择一次命名 activation compute path。
A16 与未来 A8/A4 可以有不同容差，但同一路径内的 T、kernel、schedule、模板实例和
host launcher 不得改变容差；任何 suite 都不得定义第二个 oracle 或 per-case tolerance
literal。

### 11.3 Conformance matrix

Q4_A16 test 只 include public `linear()` contract 和上述 common：

- 覆盖 Section 5 的每个 registered geometry、每个可达 production route、Text/MTP
  `T=1` 和各 route boundary 的 `b-1/b/b+1`，以及 Vision 有限轴上下界和合法相邻点；
- 为两个 GEMV、两个 SIMT、两个 MMA launchers 各提供 public 数值 case，并通过真实
  production points 执行其 Full/Predicated mechanics；
- 至少各调用一次 policy-bearing 与 A16 convenience overload；
- 不 include private dispatch/launcher header，不调用 selector，不断言 launcher、
  schedule、kernel template instance 或 Full/Predicated identity。

## 12. 已落实的迁移边界

当前实现已经：

1. 在 public Linear 边界统一 BF16/shape/contiguous/non-null/16-byte `x/out` alignment
   校验，并按 QType 直接转发；
2. 以 `Q4Launch`、六个完整 host launchers 和 direct selector 取代 pure Q4
   problem/admission/legality/plan/execute/candidate 层；
3. 把 Full/Predicated 和 grid slicing 收入各自 host launcher；
4. 让 Q5/Q6/W8 与 Q4 一样通过 type-private direct selector 返回完整 host launcher，
   不再保留 pure Linear plan/execute/candidate 层；
5. 让 Attention input、GDN input、GDN conv snapshot 和 LinearSwiGLU 只保留
   fused-local route，不依赖 pure Q4 plan/enum/fixed-pitched launch；
6. 用 public `linear()` 和统一 CPU FP64 GEMM oracle 的 Q4/Q5/Q6/W8 A16 conformance
   matrices 替换旧 pure Linear candidate/plan/dispatch/selector tests；
7. 从 CMake、benchmark source 和 production source 删除 Q4/Q5/Q6/W8 旧
   plan/launch/candidate files、symbols 和 forcing options，并删除 BF16 空占位。

不保留旧接口 alias、compatibility wrapper 或任一格式的两套并行路径。

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
- Q4_A16 public 数值 cases 覆盖所有 registered geometries、A16 route boundaries 和六个
  retained launchers；
- 完整/采样 cases 全部使用 common 中唯一的 float-input/double-accumulation GEMM；
- pure Linear suite 不替代任何 fused Op 的独立数值资格；
- 旧文件和旧 CMake/test entries 已删除；
- 可选 benchmark source 不再引用已删除的 Q4 fixed-candidate API，且本次没有运行
  benchmark。

Q5、Q6 和 W8 已分别采用同一所有权模型；相似性止于层次和职责，不通过通用 backend
framework 共享 selector、route registry 或 runtime plan。

## 14. Q5/Q6/W8/BF16 的当前实现

### 14.1 统一调用形态

Q5、Q6、W8 分别定义自己的 function-pointer type：

```cpp
using Q5Launch = void (*)(const Tensor&, const Weight&, Tensor&,
                          WorkspaceArena&, cudaStream_t);
using Q6Launch = void (*)(const Tensor&, const Weight&, Tensor&,
                          WorkspaceArena&, cudaStream_t);
using W8Launch = void (*)(const Tensor&, const Weight&, Tensor&,
                          WorkspaceArena&, cudaStream_t);
```

三个 dispatcher 都只有以下行为：

```text
select_<type>_launch(w.n, w.k, x.ne[1], policy)
    -> 返回完整 host launcher
    -> launcher(x, w, out, ws, stream)
```

`A16Only` 与 `AllowA8` 当前都查询同一 A16 registry；`AllowA8` 是许可 A8，不是要求
A8。`AllowA4` 对这三种格式均失败。selector 只读取 `(N,K,T,policy)`，不读取 layout、
payload、padded K、device capability 或 workspace。

`WorkspaceArena&` 即使当前 route 不分配 scratch 也继续保留在 launcher 合同中。
Full/Predicated、CUDA grid 切分、exact-T switch 和复合 launch 都是选中 launcher 的
内部实现。

### 14.2 Q5 registry 与物理闭包

Q5 注册以下 exact shape：

- Text/MTP：`(1024,5120)`、`(6144,5120)`、`(7168,5120)`、
  `(5120,6144)`、`(5120,17408)`，接受每个正整数 T；
- Vision：`(1152,1152)`、`(1152,4304)`，只接受 `4..131072` 内 4 的倍数。

保留的七个完整 host launcher 是：

- `launch_q5_gemv_r16_s2_x`；
- `launch_q5_simt_r8_c4`、`launch_q5_simt_r8_c8`；
- `launch_q5_simt_split2_exact`、`launch_q5_simt_split4_exact`；
- `launch_q5_mma_r64_c64`、`launch_q5_mma_r64_c128`。

exact T-to-launcher 边界直接写在
`src/ops/linear/q5/q5_dispatch.cpp::select_q5_a16_launch()`。它是唯一 executable
registry；Q5_A16 test 通过 public Linear 和 common CPU FP64 GEMM 覆盖所有
registered geometries、routes 及边界，不复制 selector 或建立第二份 support/admission
table。

Q5 MMA launcher 在切片前按完整问题选择一次边界实例：

```cpp
Full = (w.n % 64) == 0 &&
       (T % TileCols) == 0 &&
       w.k == w.padded_shape[1] &&
       (w.k % 64) == 0;
```

Q5 pure Linear 保留五个 `__global__` kernel definition：

1. `q5_rowsplit_gemv_kernel`；
2. `q5_rowsplit_gemm_simt_split2_kernel`；
3. `q5_rowsplit_gemm_simt_split4_kernel`；
4. `q5_rowsplit_gemm_simt_kernel`；
5. `q5_rowsplit_gemm_mma_kernel`。

七种 schedule、exact-T 和 Full/Predicated 模板实例不是新增 kernel。

### 14.3 Q6 registry 与物理闭包

Q6 注册：

- `(248320,5120)`：每个正整数 T；
- `(248320,2048)`：每个正整数 T；
- `(1152,1536)`：`4..131072` 内 4 的倍数。

保留四个 host launchers：

- `launch_q6_simt_r8_c4`、`launch_q6_simt_r8_c8`；
- `launch_q6_mma_r64_c64`、`launch_q6_mma_r64_c128`。

exact 边界位于
`src/ops/linear/q6/q6_dispatch.cpp::select_q6_a16_launch()`。Q6_A16 test 通过 public
Linear 和 common CPU FP64 GEMM 覆盖其 routes 与边界，不复制 selector。SIMT 只有
kernel-private Predicated 实例；MMA 的 Full 判定与 Q5 相同，只把 `TileCols` 取为 64
或 128。

Q6 pure Linear 只有两个 `__global__` kernel definition：

1. `q6_rowsplit_gemm_simt_kernel`；
2. `q6_rowsplit_gemm_mma_kernel`。

Q6 没有独立 GEMV kernel；small-T 由 SIMT schedule 执行。

### 14.4 W8 registry

W8 Text/MTP 注册以下 exact shape，均接受每个正整数 T：

```text
(5120,10240)
(1024,5120)  (6144,5120)  (14336,5120)  (34816,5120)
(5120,6144)  (5120,17408)
(2048,4096)
(1024,2048)  (9216,2048)  (12288,2048)
```

Vision 注册：

```text
(2048,4608)  (4608,4608)  (5120,4608)
T = 1..32768
```

DFlash conditioning projection 注册：

```text
(N,K) = (2048,16384), T >= 1
```

前三类 shape 的 exact T-to-launcher seam 直接位于
`src/ops/linear/w8/w8_dispatch.cpp::select_w8_a16_launch()`。W8_A16 test 通过 public
Linear 和 common CPU FP64 GEMM 覆盖其 routes 与边界，不复制 selector。W8 不存在独立
runtime schedule enum；route 的结果就是 `W8Launch`。

W8 DFlash conditioning route 保留以下 launcher families：

- `T=1`：`launch_w8_decode_r4`；
- `T=2..32`：`launch_w8_exact_t_splitk`；
- `T=33..88`：`launch_w8_exact_t_composite`；
- `T=89..96/97..128/129..144`：medium split-K C96/C128/C144；
- 其余区间：13 个实际入选的 MMA schedules；
- 需要消除 MMA padding tail 的区间：八个独立
  `launch_w8_exact_mma_<R>_<C>` function identities。

八个 exact-MMA launchers 不是 `(schedule, tail_policy)` 组合值。其固定行为是：

1. `full_cols = floor(T / C) * C`；
2. prefix 使用对应 MMA schedule，并仅依据 `(N,K,full_cols)` 选择自己的
   Full/Predicated；
3. `tail=1` 使用 `DecodeR4`；
4. `tail=2..32` 使用 exact-T split-K；
5. `tail=33..65` 使用 exact-T composite。

当前 selector 中 exact route 只产生 `tail=1..65`。`launch_w8_exact_t_composite()` 按
32 列 exact chunks 执行；remainder 1 使用 composite-private `DecodeR16`，remainder
2..31 使用相应 exact-T 实例。`DecodeR8` 不在 pure Linear 闭包中。

W8 SIMT 边界条件为：

```cpp
Full = (w.n % 8) == 0 && (T % TileCols) == 0;
```

W8 MMA 边界条件为：

```cpp
Full = (w.n % TileRows) == 0 &&
       (T % TileCols) == 0 &&
       w.k == w.padded_shape[1] &&
       (w.k % 64) == 0;
```

homogeneous launcher 在任何 token slicing 之前只选择一次该实例。exact-tail 的 prefix
与 tail 是两个有意独立的物理阶段。

W8 pure Linear 保留五个 `__global__` kernel definition：

1. `w8_rowsplit_k16384_decode_kernel`；
2. `w8_rowsplit_exact_t_splitk_kernel`；
3. `w8_rowsplit_medium_t_splitk_kernel`；
4. `w8_rowsplit_gemm_simt_kernel`；
5. `w8_rowsplit_gemm_mma_kernel`。

pure W8 的物理闭包是 21 种 schedule：DecodeR4、composite-private DecodeR16、
exact-T split-K、medium split-K C96/C128/C144、SIMT R8C4/R8C8 和 13 种 MMA
schedule。`SplitKMma32PlusTail` 是 host composition 行为，不是第 22 个 schedule。
八个 exact-tail launchers 是 host function identities，不是新 schedule 或 kernel。

`w8_k2048_decode_kernel` 由 Attention/GDN 等 fused Ops 复用，不是 pure W8 Linear
production kernel。`W8Epilogue` 是 kernel-private compile-time epilogue 选择，定义在
W8 output kernel header 中，不属于 route API。

### 14.5 BF16

当前没有 pure BF16 Linear kernel、host launcher 或注册 shape：

- `linear()` 对 `BF16_CTRL` 返回统一 unsupported error；
- 不存在 `Bf16Problem`、`bf16_contiguous_admits()`、空 plan 或必然抛错的 private
  dispatcher。

以后只有在同时具备 exact registered shape、已选 host launcher、真实 kernel 和数值
测试时，才增加 `select_bf16_launch(N,K,T,policy) -> Bf16Launch`。

### 14.6 Fused Op 所有权

fused Op 可以复用同一格式的 kernel templates 和 schedule types，但拥有自己的语义
route、output topology、epilogue、workspace 与 composite launch。它们不嵌入或调用
pure Linear 的 plan、schedule enum、variant、candidate legality 或 selector。

当前具体边界是：

- Q5 LinearAdd 的 Materialized route 直接调用 public `linear()` 后执行
  `residual_add()`；两个 fused MMA launchers 自己选择 Full；
- Q4/Q5 Attention input 的 small-T fused launcher 内部直接选择 Q5 GEMV、
  Split4Exact 或 R8C4；
- Q4/Q5 GDN input 的 pitched-output launcher内部直接选择 Q5 GEMV、Split4Exact 或
  R8C8；pitched `out_ld` 不反向进入 pure Q5 API；
- Q4/Q5 GDN conv snapshot 直接实例化所需的 Q5 GEMV/SIMT templates；
- W8 Attention、GDN、LinearAdd 和 LinearSwiGLU 的 plan 只保留各自 schedule/workspace，
  Full/Predicated 在其 host launcher 内选择；
- W8 LinearPair 的 TwoSimt launcher 由 LinearPair 自己实例化 SIMT template，不能改成
  public `linear()`，因为 Pair 与 pure Linear 在部分 T 区间有不同 winner；
- LinearPair 的 homogeneous 与 exact routes 是不同 fused-local schedule identities；
  exact prefix 与 tail 分别决定物理边界实例。

benchmark-only、且其 parent/materialized shape 不在 pure W8 registry 的 control 已删除，
没有通过扩大 pure registry 来保留诊断入口。

### 14.7 已删除的层与文件

Q5、Q6、W8 pure Linear 均不再拥有：

- `<Type>Problem`、独立 admission 和 support/route spec；
- runtime `<Type>ScheduleId`、`<Type>KernelVariant` 和 `W8TailPolicy`；
- `<Type>Plan`、resolve-plan、execute-plan；
- fixed/candidate launch、candidate legality 和字符串 schedule API。

对应 `*_rowsplit_plan.{h,cpp}`、`*_rowsplit_launch.{h,cpp}`、
`*_rowsplit_kernels.h` 已从源码和 CMake 删除。BF16 的
`bf16_contiguous_plan.{h,cpp}` 同样删除。

### 14.8 注册新 shape 或 route

注册新 shape：

1. 在 exact target 中确认真实 `(N,K)`、T domain 和调用 policy；
2. 确认它使用该 type 已有固定 storage format；
3. 在 `select_<type>_a16_launch()` 的 exact `(K,N)` case 中写入已决定的
   T-to-launcher 映射；
4. 按 `op-development.md` Section 8.1，通过 public Linear 为新增 geometry、route 和
   `b-1/b/b+1` 边界增加 common CPU FP64 GEMM case，不复制 selector 映射。

注册新 route 的最小单位是完整 host launcher。它必须封装 schedule、Full/Predicated、
grid、slicing 和 exact/composite 行为，然后由 selector 直接返回 function pointer。
不得给 launch result 增加 schedule、variant、tail-policy、workspace bytes、name 或
`actual_compute` 字段。

### 14.9 验收边界

该重构只以 build 和 tests 验收，不运行 benchmark，也不重新选择 route winner。当前
保护包括：

- Q4_A16、Q5_A16、Q6_A16 和 W8_A16 public Linear 数值 conformance matrices；
- common 中 Q4/Q5/Q6/W8 显式 weight generators、唯一 CPU FP64 GEMM 和集中容差；
- Q5 LinearAdd、Q4/Q5/W8 Attention/GDN、W8 LinearAdd/LinearSwiGLU/LinearPair 等
  fused 路径保持独立所有权，不由 pure Linear suite 代替；
- pure Linear benchmark 的单点、连续 sweep、27B/35B suite 和单次 NCU capture 均通过
  public `linear()`，且没有 fixed schedule/variant forcing。
