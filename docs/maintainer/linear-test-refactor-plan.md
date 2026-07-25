# Linear 数值测试重构计划

## 1. 状态与目标

本文定义 pure `linear` Op 数值测试的重构方案。实施时删除现有 Linear 测试及其私有
selector 测试，从零建立按 weight format 和 activation compute profile 拆分的测试。
Q4_A16 是第一份实现，也是后续 Q5_A16、Q6_A16、W8_A16 以及未来低精度 activation
测试的范例。

本次重构只验证 public `linear()` 在受支持输入上的数值结果和 public effects。它不测试
私有 selector 的 function pointer、schedule、kernel 模板实例、Full/Predicated 分支或
host launcher identity。

本文是当前实施计划。重构完成后，稳定规则应合并进
`op-development.md`、`linear-type-dispatch-memo.md`、public Linear contract 和测试
说明，随后删除本文。

## 2. 唯一数值 oracle

所有 Linear 测试共享且只共享一个 CPU oracle：

```cpp
void cpu_linear_gemm_fp64(
    const float* weight,     // row-major [N,K]
    const float* activation, // column-major logical [K,T], x[t*K+k]
    double* output,          // column-major logical [N,T], out[t*N+n]
    std::int32_t n,
    std::int32_t k,
    std::int32_t t);
```

它计算：

```text
for each t
    for each n
        double sum = 0
        for k = 0 .. K-1
            sum += double(weight[n,k]) * double(activation[t,k])
        output[t,n] = sum
```

固定规则如下：

- activation 是 public BF16 输入实际表示的值，先精确提升为 `float`；
- 模拟权重先在测试侧形成合法持久化 payload，再反量化为逻辑 `float [N,K]`；
- CPU oracle 只接收上述两个逻辑 float 矩阵，不接收 `QType`、payload、scale、policy、
  route、schedule 或 launcher；
- 每个乘数先提升为 `double`，每个完整 dot product 使用 `double` 累加；
- oracle 输出保留为 `double`，不模拟 production 的 BF16/MMA operand rounding、activation
  quantization、staging cast、分块、split-K、reduction tree 或最终 BF16 写回；
- GPU BF16 输出提升为 `double` 后与 oracle 比较；public BF16 表示误差和 route 的内部
  近似统一由整个 activation compute path 的容差处理；
- Q4、Q5、Q6、W8、A16、A8 和 A4 不得定义第二个 oracle。

反量化属于测试输入构造，不属于 GEMM oracle。格式模拟器从自己生成的 signed code 和
存储 scale 同时构造：

1. GPU 使用的 packed payload；
2. CPU oracle 使用的逻辑 float 权重。

其中每个逻辑权重先从测试生成的 signed code 和实际存储的 scale bits 得到：

```text
weight_f32[n,k] =
    float(signed_code[n,k]) * float(fp16_decode(stored_scale_bits[n,group(k)]))
```

这个 float 值就是交给统一 GEMM 的权重输入；GEMM 内不再读取或解释量化格式。

格式模拟器不得调用 production CUDA decoder，也不得把某个 production kernel 的中间
表示作为逻辑权重。

## 3. CPU FP64 GEMM 的允许优化

oracle 不要求故意低效。允许优化计算组织，但不能改变每个 dot product 的数学来源和
朴素 K 归约：

- 按输出行把工作静态分配给多个 CPU 线程；
- 对 T 做小块处理，使同一权重值可服务多个互相独立的输出列；
- 编译器可对多个互相独立的输出 accumulator 做向量化；
- 预先分配输出、线程和小型 accumulator storage，避免内层动态分配；
- 大尺寸 sampled case 先形成采样行权重和采样列 activation，再调用同一个 GEMM。

推荐的内部循环形态是：

```text
parallel for disjoint row ranges
    for n in owned rows
        for t0 in blocks of T
            double acc[TBlock] = {}
            for k = 0 .. K-1
                double w = double(weight[n,k])
                for each t in this T block
                    acc[t] += w * double(activation[t,k])
            store acc
```

该结构对每个输出仍按 `k=0..K-1` 顺序累加。线程之间不共享 accumulator，因此结果不
依赖线程调度。

禁止以下“优化”：

- 沿 K 分段后再做树形、并行或 SIMD horizontal reduction；
- 使用 float accumulator；
- 调用 BLAS、cuBLAS、production kernel、production decoder 或另一条 GPU route；
- 根据 QType、T boundary、policy、launcher 或数值 profile 改变 oracle 算法；
- 为贴合某个 kernel 引入 BF16/TF32/INT8/NVFP4 中间舍入；
- 使用允许重排归约的 fast-math 编译选项。

## 4. 公共测试脚手架

新测试组织为：

```text
tests/ops/linear/
├── linear_test_common.h
├── linear_test_common.cpp
├── test_q4_a16.cpp
├── test_q5_a16.cpp
├── test_q6_a16.cpp
├── test_w8_a16.cpp
└── test_<weight>_<activation>.cpp
```

`linear_test_common.{h,cpp}` 是唯一公共脚手架，负责：

- Q4G64_F16S、Q5G64_F16S、Q6G64_F16S 和 W8G32_F16S 各自的测试权重生成；
- 从同一组测试 signed codes 和 stored scales 同时形成 GPU packed payload 与 oracle 使用的
  float 权重；
- BF16 activation 的生成、host 表示和 device payload；
- 唯一 `cpu_linear_gemm_fp64()`；
- device buffer、16-byte alignment、workspace 和 stream；
- output poison、前后 guard、完整写入和非有限值检查；
- 完整输出或确定性 structure-aware sampling；
- 调用两个 public `linear()` overload；
- GPU BF16 输出提升与统一数值判据；
- public contract 要求的 input/weight preservation。

脚手架不提供 selector、launcher 或 schedule 查询接口。

### 4.1 量化权重生成

`linear_test_common` 提供显式的格式入口：

```cpp
SimulatedLinearWeight make_q4g64_f16s_weight(
    std::int32_t n, std::int32_t k, std::uint32_t seed,
    std::span<const std::int32_t> oracle_rows);

SimulatedLinearWeight make_q5g64_f16s_weight(
    std::int32_t n, std::int32_t k, std::uint32_t seed,
    std::span<const std::int32_t> oracle_rows);

SimulatedLinearWeight make_q6g64_f16s_weight(
    std::int32_t n, std::int32_t k, std::uint32_t seed,
    std::span<const std::int32_t> oracle_rows);

SimulatedLinearWeight make_w8g32_f16s_weight(
    std::int32_t n, std::int32_t k, std::uint32_t seed,
    std::span<const std::int32_t> oracle_rows);
```

`SimulatedLinearWeight` 至少拥有：

```cpp
struct SimulatedLinearWeight {
    std::vector<std::uint8_t> packed_payload;
    std::vector<std::int32_t> oracle_rows;
    std::vector<float> oracle_weight; // [oracle_rows.size(), K]

    Weight device_weight(void* device_payload) const;
};
```

生成器按照对应持久化格式直接生成完整合法 payload。对于完整比较，
`oracle_rows` 表示全部 N 行；对于大 shape 的 sampled comparison，它只要求生成被采样
行的 dense float 权重，但 GPU payload 仍然完整。`device_weight()` 只把该格式的固定
metadata 和 device payload addresses 组装成 GPU 调用需要的 `Weight` view；它不参与
oracle 计算。

每个 generator 必须从同一份测试 codes 和实际写入 payload 的 scale bits 计算
`oracle_weight`。格式之间可以共享无数值策略的 bit packing、FP16 conversion 和 plane
offset primitives，但不得：

- 调用 production decoder 或 kernel；
- 让 test case 自行重新解释 payload；
- 为不同 route、schedule 或 activation profile 改变权重生成；
- 用一个隐式推断 layout/group/scale 的通用量化 backend 代替四个明确格式入口。

这样，weight-specific test 只选择格式生成器；权重输入构造、packed storage 和 oracle
float materialization 全部由公共脚手架唯一负责。

每个 `test_<weight>_<activation>.cpp` 只按 shape 声明 public conformance cases：

```cpp
struct Invocation {
    std::int32_t t;
    CallForm call_form;
    LinearPolicy policy;
};

struct ShapeCase {
    std::int32_t n;
    std::int32_t k;
    std::uint32_t seed;
    Comparison comparison;
    bool verify_input_preservation;
    std::span<const Invocation> invocations;
};
```

case 中不得出现 expected launcher、schedule、kernel、Full/Predicated 或 function
pointer。case 的意义只有：给定 public 输入和 policy，`linear()` 的输出是否符合唯一
oracle 及对应的统一数值判据。

## 5. 大尺寸 case

可合理承担完整 reference 的 case 比较全部输出。对完整 float 权重或完整 FP64 输出
明显过大的 production shape，采用以下统一流程：

1. GPU 仍接收完整合法 packed weight、完整 activation 和完整 output；
2. output 预填有限输入不可能产生的 BF16 NaN poison，并保留前后 guard；
3. 调用 public `linear()`；
4. 检查完整 output 已写入、值有限且 guards 未损坏；
5. 按固定规则选择代表性的行和列；
6. 只把这些行反量化为 dense float，并提取这些 activation 列；
7. 将得到的 `[Nsample,K] × [K,Tsample]` 交给同一个
   `cpu_linear_gemm_fp64()`；
8. 与 GPU 完整输出中的对应元素比较。

采样规则由公共脚手架唯一拥有。各类型测试不得另写 sampled dot oracle。

## 6. 统一容差

Linear 的容差和比较逻辑只允许定义在 `linear_test_common`。每个
`test_<weight>_<activation>.cpp` 必须为整个 suite 选择一个 activation compute path；
单个 T case 不得选择 tolerance。例如当前只有：

```cpp
enum class ActivationCompute {
    A16,
};
```

公共映射：

```cpp
LinearTolerance tolerance_for(ActivationCompute activation_compute);
```

满足以下规则：

- 所有 activation compute path 使用同一个 CPU FP64 GEMM oracle；
- activation compute path 只决定 acceptance criterion，不改变 reference；
- 容差数值不得出现在单个 case 或 weight-specific test 中；
- unexpected NaN/Inf 必须失败；
- GEMM 使用统一的 normwise criterion，并保留有限的 gross pointwise error 限制，防止
  局部严重错误被整体范数掩盖；
- A16 与 A8/A4 等真实 activation quantization 路径可以使用不同容差；
- 同一个 A16 路径内部选择的 GEMV、SIMT、MMA kernel、schedule、模板实例或 host
  launcher 不得产生不同容差；
- 不能因为某个实现失败而复制其数值行为或临时放宽单个 case。

新增 A8、A4 等 activation compute path 时，先在公共位置增加其枚举值和命名
criterion，再用既有 oracle 资格化对应 public cases；没有 production 实现的 path
不预留容差数值。

## 7. 测试拆分与覆盖

第一份测试为 `test_q4_a16.cpp`。它：

- 只 include public `ninfer/ops/linear.h` 和公共脚手架；
- 通过 public `linear()` 覆盖 Q4_A16 注册 geometry、可达 production route 和必要的
  T 边界；
- 至少各覆盖一次 policy-bearing overload 和 A16 convenience overload；
- 整个 suite 只选择一次 A16 criterion，不按 T、kernel、schedule 或 launcher 切换；
- 对合理尺寸做完整比较，对大尺寸按 Section 5 处理；
- 不测试 private selector 映射。

Q5_A16、Q6_A16 和 W8_A16 后续使用完全相同的脚手架，分别拥有自己的 case 文件。未来
Q4_A8 等实现落地后新增对应文件，不修改 oracle。当前 BF16 pure Linear 没有 production
实现，因此不建立伪造的 BF16 数值测试。

## 8. 旧测试处理

实施时删除所有旧 pure Linear 数值测试、当前未完成的 Q4 拆分代码和 Q4/Q5/Q6/W8
selector tests；不保留兼容 wrapper、旧 helper 或双轨 CMake target。

旧 `test_linear.cpp` 中混入的 fused Op 测试不进入新 Linear suite。仍需保护的 fused
语义必须由相应 public fused Op 自己的测试拥有；不得为了保留旧 monolith 而把 fused
route、plan 或 launcher 知识放入 Linear 脚手架。

private selector function pointer、token-to-launcher 表、schedule identity 和模板实例
不是数值合同，不建立替代测试。

## 9. 权威文档收口

实现本计划时同步修改：

- `docs/maintainer/op-development.md`：明确一个 Op 只有一个高精度 naive oracle，低精度
  实现只通过命名容差区别；
- `include/ninfer/ops/linear.h`：明确 Linear 的逻辑 float 权重、BF16 public 输入和统一
  高精度资格化方式；
- `docs/maintainer/linear-type-dispatch-memo.md`：删除把 selector、launcher 或
  exact-decode dot helper 当作 Linear oracle 的表述；
- `tests/README.md`：记录新的文件拆分、公共脚手架和运行方式。

未被当前文档索引、且仍描述已删除 `LinearPlan` 架构的
`linear-low-precision-compute-policy-memo.md` 应删除；其中仍有效的 policy 语义合并到
当前 Linear 权威文档。

## 10. 实施与验收

实施顺序：

1. 删除旧 Linear/selector tests 及旧 CMake entries；
2. 修订上述权威文档；
3. 从零实现 `linear_test_common`；
4. 从零实现 Q4_A16 conformance cases；
5. 增加新的 CMake test target；
6. 构建并运行 Q4_A16 test；
7. Q4 范例确认后，以相同脚手架增加其他 weight/activation 文件。

验收要求：

- Linear test tree 中只有一个 `cpu_linear_gemm_fp64()`；
- oracle 不接收 packed weight、QType 或任何 production route 信息；
- 所有 weight/activation case 都使用同一 oracle；
- 所有容差只在公共脚手架中定义；
- Q4_A16 test 只调用 public `linear()`；
- 没有 selector/function-pointer/source-scan 测试；
- 完整和 sampled case 都检查 GPU 完整输出写入及 guards；
- build、Q4_A16 test 和文档链接检查通过；
- 本次不运行 benchmark，也不以 benchmark 作为正确性门槛。
