# Linear benchmark 合同与预置 suite

## 状态与范围

本文是 `bench/ops/linear_bench.cu` 的当前权威，定义 pure Linear benchmark 的命令、
计时合同、指标、预置 suite 和扩展规则。benchmark 不改变 production Linear route，
也不重新选择任何 route winner。

benchmark 只测量：

```text
ninfer::ops::linear(x, w, out, policy, stream)
```

Q4/Q5/Q6/W8 LinearAdd、LinearSwiGLU、LinearPair 和其他 fused Ops 不属于这个
benchmark。它们继续由各自的 benchmark 独立测量。

当前只有 A16 pure Linear route。benchmark 不提供虚假的 BF16、A8 或 A4 选项；只有
相应 production route、数值资格和硬件规格参照同时存在后，才增加新的执行类型。

## 1. 使用场景

新 benchmark 服务四个具体需求。

### 1.1 单点性能

显式指定 weight type、`N`、`K` 和 `T`，测量一个 production Linear point：

```bash
./build/bench/ninfer_linear_bench \
  --qtype q4 --policy a16 \
  --n 4096 --k 5120 --t 8
```

数字 `(N,K)` 是主入口。benchmark 不复制 production selector 的完整 shape admission
表；不支持的 point 由 public `linear()` 及其 format selector 拒绝。

### 1.2 NCU 单点

profile mode 必须只接受一个 exact point：

```bash
ncu --profile-from-start off ... \
  ./build/bench/ninfer_linear_bench \
  --qtype q4 --policy a16 \
  --n 4096 --k 5120 --t 8 --profile
```

程序在 capture 外完成 allocation、payload initialization、warmup 和 L2 flush，然后：

```text
cudaProfilerStart()
    one public linear() call
    stream synchronize
cudaProfilerStop()
```

因此 NCU 可以看到该 host launcher 实际发出的一个或多个 kernel launch，而不会同时
采集 weight 初始化、copy roofline probe、Tensor Core probe、warmup 或重复测量。

这里必须继续区分：

- benchmark point 是一次 public Linear 调用；
- selector 返回的是 host launcher；
- host launcher 可以发出一个或多个 kernel launch；
- NCU 中看到的是具体 kernel 模板实例；
- benchmark 不把 schedule、模板实例或 host launcher 错称为 kernel。

### 1.3 小 T sweep

固定 `(qtype,N,K,policy)`，连续扫描一个 T 区间：

```bash
./build/bench/ninfer_linear_bench \
  --qtype q4 --policy a16 \
  --n 4096 --k 5120 \
  --sweep 1:32:1 \
  --csv-out profiles/bench/q4_4096x5120_t1_32.csv
```

Vision step domain 可以直接表达为：

```bash
--sweep 4:512:4
```

每个 T 输出相邻点的 median latency 变化。sweep 用于发现不合理耗时阶跃、观察已有
route seam，并为独立的 route 测量任务提供边界依据；它不设置自动性能 gate，也不在
benchmark 中复制 selector 或 candidate-legality 表。

### 1.4 典型模型 suite

为避免每次手写 `(qtype,N,K,T)`，benchmark 注册少量 27B/35B 代表性 point。一次
suite 调用顺序运行全部 point：

```bash
./build/bench/ninfer_linear_bench --suite qwen3_6_27b
./build/bench/ninfer_linear_bench --suite qwen3_6_35b_a3b
./build/bench/ninfer_linear_bench --suite all
```

无参数运行仍打印 usage，不隐式启动重型 suite。suite 是显式 convenience，不是
correctness matrix、完整 artifact inventory 或第二份 production admission registry。

## 2. 典型 suite

### 2.1 默认 T 集合

suite entry 使用两种固定采样类：

| T class | 默认 T | 用途 |
|---|---|---|
| `Continuous` | `1,16,128,1024` | 同时观察小 T 带宽阶段、route 过渡和大 T 计算阶段 |
| `VisionStep4` | `4,128,1024` | 对所有当前 Vision geometries 都合法的保守公共采样 |

这些点不是 production route boundary 的副本，也不尝试覆盖每个 selector case。完整
连续 seam 检查由显式 `--sweep` 完成。

suite 对每个 `(qtype,N,K)` 只生成一次 weight，并按该 entry 的最大 T 一次性分配
activation/output。不同 T 复用同一组 allocation，不重复构造大权重。

### 2.2 `qwen3_6_27b`

27B core suite 只登记当前模型中实际通过 pure public Linear 使用的主要 Text/MTP/head
几何：

| Label | QType | `(N,K)` | T class | 实际角色 |
|---|---|---:|---|---|
| `27b.output_head` | Q6 | `(248320,5120)` | Continuous | full output head |
| `27b.draft_head` | Q4 | `(131072,5120)` | Continuous | optimized proposal head |
| `27b.gdn_output_gate` | Q5 | `(6144,5120)` | Continuous | GDN Z row view |
| `27b.mtp_input` | W8 | `(5120,10240)` | Continuous | MTP input projection |
| `27b.mtp_attention` | W8 | `(14336,5120)` | Continuous | packed MTP Q/K/gate/V projection |
| `27b.mtp_gate_up` | W8 | `(34816,5120)` | Continuous | MTP gate/up |
| `27b.mtp_down` | W8 | `(5120,17408)` | Continuous | MTP down |

27B Vision suite 登记实际由 public Linear 调用的七个几何：

| Label | QType | `(N,K)` | T class | 实际角色 |
|---|---|---:|---|---|
| `27b.vision_patch` | Q6 | `(1152,1536)` | VisionStep4 | patch projection |
| `27b.vision_qkv` | Q4 | `(3456,1152)` | VisionStep4 | attention QKV |
| `27b.vision_attn_out` | Q5 | `(1152,1152)` | VisionStep4 | attention output |
| `27b.vision_fc1` | Q4 | `(4304,1152)` | VisionStep4 | MLP expansion |
| `27b.vision_fc2` | Q5 | `(1152,4304)` | VisionStep4 | MLP contraction |
| `27b.vision_merger_fc1` | W8 | `(4608,4608)` | VisionStep4 | merger expansion |
| `27b.vision_merger_fc2` | W8 | `(5120,4608)` | VisionStep4 | merger output |

基础 Text Attention、GDN input 和 MLP 的 packed/fused parents 不因存在于 artifact 就加入
pure Linear suite。它们当前由 Attention/GDN/LinearSwiGLU/LinearAdd 等独立语义 Op
拥有，应在相应 Op benchmark 中测量。

### 2.3 `qwen3_6_35b_a3b`

35B-A3B core suite 登记当前真正通过 pure public Linear 使用的主要 head、MTP 和
DFlash 几何：

| Label | QType | `(N,K)` | T class | 实际角色 |
|---|---|---:|---|---|
| `35b.output_head` | Q6 | `(248320,2048)` | Continuous | full output head |
| `35b.draft_head` | Q4 | `(131072,2048)` | Continuous | optimized proposal head |
| `35b.mtp_projection` | W8 | `(2048,4096)` | Continuous | MTP input/output projection geometry |
| `35b.dflash_feature` | W8 | `(2048,16384)` | Continuous | DFlash conditioning projection |

35B-A3B Vision backbone 与 27B 共享前六个几何，merger output 进入 2048 hidden：

| Label | QType | `(N,K)` | T class | 实际角色 |
|---|---|---:|---|---|
| `35b.vision_patch` | Q6 | `(1152,1536)` | VisionStep4 | patch projection |
| `35b.vision_qkv` | Q4 | `(3456,1152)` | VisionStep4 | attention QKV |
| `35b.vision_attn_out` | Q5 | `(1152,1152)` | VisionStep4 | attention output |
| `35b.vision_fc1` | Q4 | `(4304,1152)` | VisionStep4 | MLP expansion |
| `35b.vision_fc2` | Q5 | `(1152,4304)` | VisionStep4 | MLP contraction |
| `35b.vision_merger_fc1` | W8 | `(4608,4608)` | VisionStep4 | merger expansion |
| `35b.vision_merger_fc2` | W8 | `(2048,4608)` | VisionStep4 | merger output |

35B routed expert `[262144,2048]` gate/up 和 `[524288,512]` down 不加入 pure Linear
suite。它们属于 `sparse_moe` 的 execution contract，当前也不是 pure Linear selector
注册的 geometry。Attention/GDN/DFlash layer projections 同理继续由其 fused Op
benchmark 负责。

### 2.4 `all`

`all` 是两个 model suite 的 union。完全相同的
`(qtype,policy,N,K,T)` point 只执行一次；共享 Vision point 不因两个 target 都使用而
重复计时。结果可以保留多个 role label，但 timing row 只有一份。

## 3. 请求流水线

```text
CLI
 |
 +-- exact point: qtype + policy + N + K + T
 |
 +-- sweep:       qtype + policy + N + K + T range
 |
 +-- suite:       curated target entries + entry-local default T class
 |
 v
expand to BenchPoint(qtype, policy, N, K, T, labels)
 |
 v
group points by (qtype, policy, N, K)
 |
 v
allocate and initialize one packed weight
allocate x/out for max T in the group
 |
 v
for each T:
    bind [K,T] and [N,T] Tensor views
    warm up
    flush L2 outside timed interval
    time public ops::linear()
 |
 v
derive model bytes, useful FLOPs and fixed-spec ratios
 |
 v
compact table and optional CSV
```

suite expansion只提供便利 point；执行仍走和显式单点完全相同的 fixture、public API、
计时器和指标计算，没有 suite-specific dispatch。

## 4. 理论流量

设：

```text
K_pad = align_up(K, 128)
groups = N * K_pad / group_size
```

format weight bytes 是 kernel 需要消费的各存储平面之和，不包含 plane 之间只用于起始
地址对齐、不会被读取的 gap：

| QType | group | bytes/group | weight bytes |
|---|---:|---:|---:|
| Q4G64_F16S | 64 | `32 code + 2 scale` | `groups * 34` |
| Q5G64_F16S | 64 | `32 low + 8 high + 2 scale` | `groups * 42` |
| Q6G64_F16S | 64 | `32 low + 16 high + 2 scale` | `groups * 50` |
| W8G32_F16S | 32 | `32 code + 2 scale` | `groups * 34` |

一次 Linear 的理论最低流量为：

```text
model_bytes = weight_bytes + 2*K*T + 2*N*T
effective_GB/s = model_bytes / seconds / 1e9
dram_spec_pct = effective_GB/s / 1792 * 100
```

这是从 public representation 得到的 model floor，不是 profiler 观测的 physical DRAM
traffic。它不计：

- cache-line transaction 放大；
- 同一 weight tile 的 route-private 重读；
- workspace、split/reduce 或复合 launcher 的中间流量；
- cache hit 带来的 DRAM 流量减少。

这些问题需要 NCU 回答。benchmark 不再根据 launcher column tile 推导
`weight_replay_lower_bound_bytes`。

## 5. 理论计算量

所有 weight types 使用同一个数学 GEMM 工作量：

```text
useful_flops = 2*N*K*T
useful_TFLOP/s = useful_flops / seconds / 1e12
bf16_tc_spec_pct = useful_TFLOP/s / 209.5 * 100
```

不把 dequantization、bit decode、padding、split-K 重复工作或 tile rounding 加入
`useful_flops`，也不从 private schedule 推导 `executed_tflops`。这保证不同实现都用同一
数学工作量比较。

当前所有 suite point 都是 A16，因此 compute 参照使用 RTX 5090 dense BF16 Tensor
Core `209.5 TFLOP/s`。以后若增加真实 A8/A4 path，必须登记对应的固定硬件规格；不能
继续套用 BF16 peak，也不能用自建 probe 的实测值替代 datasheet/spec 参照。

统一 roofline 参考为：

```text
memory_floor_us = model_bytes / 1792 GB/s
compute_floor_us = useful_flops / 209.5 TFLOP/s
roofline_floor_us = max(memory_floor_us, compute_floor_us)
roofline_efficiency = roofline_floor_us / median_us
```

每个结果同时输出 `memory` 或 `compute` bound。不存在写死的“小 T/大 T”分界。

## 6. 计时合同

普通单点、sweep 和 suite 使用同一个 cold-cache 计时合同：

1. allocation、payload initialization 和首次 public call 不计时；
2. 每个 sample 前用固定 256 MiB buffer 驱逐 L2；
3. flush 完成在 CUDA event start 之前；
4. event 只包围一次完整 public Linear 调用；
5. primary result 使用 median，同时记录 min 和 p95；
6. 默认 warmup `3`、repeat `20`，允许显式覆盖；
7. 不再为每个 point 额外运行一套 warm-cache measurement。

如果以后确有独立 hot-cache 问题，应增加清楚命名的 opt-in mode，不能重新把 cold/warm
双倍工作设为所有命令的默认行为。

## 7. 输出

console header 固定打印：

```text
gpu=RTX 5090
dram_spec=1792 GB/s
bf16_dense_tc_spec=209.5 TFLOP/s
cache=cold
```

单行结果保留：

```text
label qtype policy N K T median_us min_us p95_us
model_GB effective_GB/s DRAM_%
useful_TFLOP/s BF16_TC_% bound roofline_%
```

sweep 额外输出相邻 T 的 `delta_%`。CSV 可以增加 weight/x/out byte breakdown、warmup、
repeat 和环境版本，但不恢复以下字段：

- measured stream-copy ceiling；
- measured Tensor Core ceiling；
- candidate；
- kernel variant；
- inferred weight replay；
- inferred executed FLOPs；
- duplicated launcher tile metadata。

benchmark 不需要声称选中了哪个 kernel。单点 NCU 可以看到真实 kernel 实例，production
selector 源码是 host launcher route 的唯一 executable authority。

## 8. 已完成的实现收口

旧 `linear_op_bench.cu` 已删除，并移除：

- `--all-targets` 隐式默认行为；
- LinearAdd、LinearSwiGLU、LinearPair 和 composed control；
- private fused plan headers；
- Q4/Q5/Q6/W8 launcher-name、tile、tail 和 MMA row-tile 镜像；
- stream-copy kernel、copy buffers、`--copy-repeat` 和 `--stream-ceiling-gbs`；
- Tensor Core peak probe；
- warm-cache second pass；
- per-shape default route-boundary T tables；
- `candidate_name` 和 `kernel_variant` 输出。

当前 `linear_bench.cu` 保留：

- deterministic packed Q4/Q5/Q6/W8 weight generation；
- BF16 activation/output allocation；
- public Linear invocation；
- cold-cache CUDA-event timing；
- single/sweep/suite/profile mode；
- compact console/CSV output。

构建 target 是 `ninfer_linear_bench`。旧 `ninfer_linear_op_bench` 不保留兼容别名。

## 9. 注册规则

### 9.1 新 production shape

新增 production Linear shape 不要求修改 benchmark：

1. selector 和 public numerical test 完成注册；
2. 立即可以用数字 `--n/--k/--t/--qtype` 测量；
3. 只有当它成为 27B/35B 的高频或架构代表性 geometry 时，才加入 model suite。

suite entry 只包含：

```text
label, qtype, policy, N, K, T class
```

不得携带 launcher、schedule、kernel、tile、Full/Predicated、workspace 或 route-boundary
metadata。

### 9.2 新 route

新增或替换 host launcher 不修改 benchmark。single、sweep 和 suite 始终调用 public
Linear，因此自然测量 selector 当前返回的 production route。

如果需要在 route 入选前比较多个候选，应建立该格式任务范围内的临时测量代码；不得把
generic candidate forcing、legality registry 或 plan 层重新引入长期 pure Linear
benchmark。

### 9.3 新 weight/activation compute type

只有同时完成以下事项才增加新的 benchmark type：

1. public policy 对应真实可达的 compute path；
2. packed-weight fixture 支持该 persistent format；
3. model-byte 公式明确；
4. 对应固定硬件 peak 明确；
5. public numerical suite 已按该 compute criterion 资格化。

`AllowA8` 只是许可，不能直接在输出中冒充实际 A8。当前所有预置 suite 必须显式使用
`A16Only`。

## 10. 当前验证

当前实现已验证：

1. `ninfer_linear_bench` Release target 可构建；
2. 27B suite 的 49 个 point 和 35B-A3B suite 的 37 个 point 均通过 public Linear
   执行；
3. `all` 合并为 68 个 unique point，共享 Vision exact point 只执行一次；
4. Text `T=1..3` sweep 和 Vision `T=4,8` step-4 sweep 均输出连续结果；
5. NCU `--profile-from-start off --launch-count 1` 只捕获一次 public Linear 调用产生的
   `q4_rowsplit_gemv_kernel` 实例；
6. Q4 `[4096,5120], T=1` 的 weight-byte 结果为 `11141120`，等于
   `4096*5120*(4/8+2/64)`；console 与 CSV 使用同一计算；
7. 输出固定引用 `1792 GB/s` 和 `209.5 TFLOP/s`，不存在实测 ceiling probe。

benchmark 不承担数值 correctness；Q4/Q5/Q6/W8 A16 correctness 继续由各自 public
Linear conformance suite 和统一 CPU FP64 GEMM oracle 负责。
