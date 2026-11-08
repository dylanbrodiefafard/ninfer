# Linear benchmark contract and preset suites

## Status and scope

This document is the current authority for `bench/ops/linear_bench.cu`. It defines the commands,
timing contract, metrics, preset suites, and extension rules of the pure Linear benchmark. The
benchmark does not change the production Linear route and does not reselect any route winner.

The benchmark measures only:

```text
ninfer::ops::linear(x, w, out, policy, workspace, stream)
```

This is a long-lived public benchmark. The measured call of single, sweep, suite, and profile
modes must go through the public entry point above. The fixture may construct inputs according
to the public weight format, but the benchmark must not include Linear-private
launcher/plan/dispatch headers, must not call `ninfer::ops::detail`, and must not provide
candidate, kernel, or route forcing options.

Q4/Q5/Q6/W8 LinearAdd, LinearSwiGLU, LinearPair, and other fused Ops do not belong to this
benchmark. They continue to be measured independently by their own benchmarks.

Q4/Q5/Q6/W8 and BF16_CTRL use the existing A16 route. The following NVFP4 exact problems support
both the A16 and A4 policies and serve as a permanent development surface; they are not added
to the model suites:

```text
[14336,5120]  attention input
[16384,5120]  GDN input
[34816,5120]  MLP gate/up development parent
[ 5120,6144]  attention/GDN output
[ 5120,17408] MLP down
```

`--policy a4` measures the complete public call; the production resolver selects an already
qualified A16 or A4 route from the exact geometry and T. The benchmark does not copy or infer
that private selection. The default prefill chunk `T=1024` is the primary performance point of
the AllowA4 surface; larger T is used only to confirm the positive-T contract and the
scalability of the route.

## 1. Use cases

The new benchmark serves four concrete needs.

### 1.1 Single point

Specify the weight type, `N`, `K`, and `T` explicitly to measure one production Linear point:

```bash
./build/bench/ninfer_linear_bench \
  --qtype q4 --policy a16 \
  --n 6144 --k 5120 --t 8
```

The numbers `(N,K)` are the primary entry. The benchmark does not copy the full shape admission
table of the production selector; unsupported points are rejected by the public `linear()` and
its format selector.

The command for a BF16 decode exact point is:

```bash
./build/bench/ninfer_linear_bench \
  --qtype bf16 --policy a16 \
  --n 14336 --k 5120 --t 1
```

The permanent NVFP4 A16 decode point is:

```bash
./build/bench/ninfer_linear_bench \
  --qtype nvfp4 --policy a16 \
  --n 14336 --k 5120 --t 1
```

Its primary W4A4 MMA point is:

```bash
./build/bench/ninfer_linear_bench \
  --qtype nvfp4 --policy a4 \
  --n 14336 --k 5120 --t 1024
```

The remaining four permanent NVFP4 problems use the same numeric entry, for example:

```bash
./build/bench/ninfer_linear_bench \
  --qtype nvfp4 --policy a4 \
  --n 16384 --k 5120 --t 1024

./build/bench/ninfer_linear_bench \
  --qtype nvfp4 --policy a4 \
  --n 34816 --k 5120 --t 1024

./build/bench/ninfer_linear_bench \
  --qtype nvfp4 --policy a4 \
  --n 5120 --k 6144 --t 1024

./build/bench/ninfer_linear_bench \
  --qtype nvfp4 --policy a4 \
  --n 5120 --k 17408 --t 1024
```

The workspace is allocated before the timed region according to the public capacity query; all
launches and traffic of activation quantization and the GEMM fall within one timed `linear()`
call. A GEMM-only result after pre-quantization is not a production metric of this benchmark.

### 1.2 NCU single point

Profile mode must accept exactly one exact point:

```bash
ncu --profile-from-start off ... \
  ./build/bench/ninfer_linear_bench \
  --qtype q4 --policy a16 \
  --n 6144 --k 5120 --t 8 --profile
```

The program completes allocation, payload initialization, warmup, and L2 flush outside the
capture, and then:

```text
cudaProfilerStart()
    one public linear() call
    stream synchronize
cudaProfilerStop()
```

NCU can therefore see the one or more kernel launches actually issued by that host launcher,
without also capturing weight initialization, the copy roofline probe, the Tensor Core probe,
warmup, or repeated measurements.

The following distinctions must continue to be maintained:

- a benchmark point is one public Linear call;
- the selector returns a host launcher;
- a host launcher may issue one or more kernel launches;
- NCU sees concrete kernel template instances;
- the benchmark does not mislabel a schedule, a template instance, or a host launcher as a
  kernel.

### 1.3 Small-T sweep

Fix `(qtype,N,K,policy)` and scan a contiguous range of T:

```bash
./build/bench/ninfer_linear_bench \
  --qtype q4 --policy a16 \
  --n 6144 --k 5120 \
  --sweep 1:32:1 \
  --csv-out profiles/bench/q4_6144x5120_t1_32.csv
```

The Vision step domain can be expressed directly as:

```bash
--sweep 4:512:4
```

Each T reports the change in median latency relative to the adjacent point. The sweep is used to
discover unreasonable latency steps, observe existing production route seams, and provide leads
for separate route measurement tasks; each point still calls only the public Linear. It does not
compare forced candidates, does not set an automatic performance gate, and does not copy the
selector, crossover, or candidate-legality tables into the benchmark.

### 1.4 Typical model suites

To avoid hand-writing `(qtype,N,K,T)` each time, the benchmark registers a small number of
representative 27B points. One suite invocation runs all points in order:

```bash
./build/bench/ninfer_linear_bench --suite qwen3_6_27b
```

Running with no arguments still prints usage and does not implicitly start a heavy suite. A
suite is an explicit convenience; it is not a correctness matrix, a complete artifact inventory,
or a second production admission registry.

## 2. Typical suites

### 2.1 Default T sets

Suite entries use two fixed sampling classes:

| T class | Default T | Purpose |
|---|---|---|
| `Continuous` | `1,16,128,1024` | Observes the small-T bandwidth phase, route transitions, and the large-T compute phase together |
| `VisionStep4` | `4,128,1024` | A conservative common sampling that is legal for all current Vision geometries |

These points are not copies of production route boundaries, and they do not try to cover every
selector case. The full continuous seam check is done by an explicit `--sweep`.

For each `(qtype,N,K)`, the suite generates the weight only once and allocates activation/output
once for the maximum T of that entry. Different T values reuse the same allocations and do not
reconstruct the large weight.

### 2.2 `qwen3_6_27b`

The 27B core suite registers only the major Text/MTP/head geometries that the current model
actually uses through pure public Linear:

| Label | QType | `(N,K)` | T class | Actual role |
|---|---|---:|---|---|
| `27b.draft_head` | Q4 | `(131072,5120)` | Continuous | optimized proposal head |
| `27b.mtp_input` | W8 | `(5120,10240)` | Continuous | MTP input projection |
| `27b.mtp_attention` | W8 | `(14336,5120)` | Continuous | packed MTP Q/K/gate/V projection |
| `27b.mtp_gate_up` | W8 | `(34816,5120)` | Continuous | MTP gate/up |
| `27b.mtp_down` | W8 | `(5120,17408)` | Continuous | MTP down |

The 27B Vision suite registers the seven geometries actually called through public Linear:

| Label | QType | `(N,K)` | T class | Actual role |
|---|---|---:|---|---|
| `27b.vision_patch` | Q6 | `(1152,1536)` | VisionStep4 | patch projection |
| `27b.vision_qkv` | Q4 | `(3456,1152)` | VisionStep4 | attention QKV |
| `27b.vision_attn_out` | Q5 | `(1152,1152)` | VisionStep4 | attention output |
| `27b.vision_fc1` | Q4 | `(4304,1152)` | VisionStep4 | MLP expansion |
| `27b.vision_fc2` | Q5 | `(1152,4304)` | VisionStep4 | MLP contraction |
| `27b.vision_merger_fc1` | W8 | `(4608,4608)` | VisionStep4 | merger expansion |
| `27b.vision_merger_fc2` | W8 | `(5120,4608)` | VisionStep4 | merger output |

The packed/fused parents of the base Text Attention, GDN input, and MLP are not added to the
pure Linear suite merely because they exist in the artifact. They are currently owned by
independent semantic Ops such as Attention/GDN/LinearSwiGLU/LinearAdd and should be measured in
the corresponding Op benchmarks.

## 3. Request pipeline

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
derive model bytes, useful FLOPs, fixed-spec ratios and measured-read ratio
 |
 v
compact table and optional CSV
```

Suite expansion only provides convenience points; execution still goes through exactly the same
fixture, public API, timer, and metric computation as an explicit single point, with no
suite-specific dispatch.

## 4. Theoretical traffic

Let:

```text
K_pad = align_up(K, 128)
groups = N * K_pad / group_size
```

The format weight bytes are the sum of the storage planes that the kernel needs to consume, and
do not include gaps between planes that exist only for start-address alignment and are never
read:

| QType | group | bytes/group | weight bytes |
|---|---:|---:|---:|
| Q4G64_F16S | 64 | `32 code + 2 scale` | `groups * 34` |
| Q5G64_F16S | 64 | `32 low + 8 high + 2 scale` | `groups * 42` |
| Q6G64_F16S | 64 | `32 low + 16 high + 2 scale` | `groups * 50` |
| W8G32_F16S | 32 | `32 code + 2 scale` | `groups * 34` |
| NVFP4 | 16 | `8 code + 1 E4M3 scale` | `groups * 9` |
| BF16_CTRL | — | direct BF16 | `2 * N * K` |

The theoretical minimum traffic of one Linear is:

```text
model_bytes = weight_bytes + 2*K*T + 2*N*T
effective_GB/s = model_bytes / seconds / 1e9
dram_spec_pct = effective_GB/s / 1792 * 100
read_ceiling_pct = effective_GB/s / 1674.5 * 100
```

This is a model floor derived from the public representation, not the physical DRAM traffic
observed by a profiler. It does not count:

- cache-line transaction amplification;
- route-private rereads of the same weight tile;
- intermediate traffic of workspace, split/reduce, or composite launchers;
- the DRAM traffic reduction due to cache hits.

These questions need NCU to answer. The benchmark no longer derives
`weight_replay_lower_bound_bytes` from the launcher column tile.

`1792 GB/s` remains the fixed hardware specification, used for the existing `DRAM_%` and the
fixed-spec roofline. The additional `READ_%` uses `1674.5 GB/s`, the 4 GiB `uint4` pure-read
result of `tools/hbm_bandwidth_probe.cu` on the RTX 5090, which represents the read-only
ceiling this machine has been measured to sustain. It exists only to give read-dominated Linear
calls an actually attainable utilization; it does not replace the fixed-spec metric and does not
change the `model_bytes` convention of one logical pass over the weight. The copy probe reads
and writes at the same time and is not the applicable ceiling for such GEMVs.

## 5. Mathematical work and route-neutral metrics

All weight types use the same mathematical GEMM work:

```text
useful_flops = 2*N*K*T
useful_TFLOP/s = useful_flops / seconds / 1e12
```

Dequantization, bit decode, padding, split-K repeated work, and tile rounding are not added to
`useful_flops`, and no `executed_tflops` is derived from the private schedule. This ensures
that different implementations are compared against the same mathematical work.

`AllowA4` is a permission, not the actual activation-compute profile; under the same policy,
different exact geometries and T values may select different routes. The long-lived public
benchmark therefore does not output `activation_compute`, `TC_%`, a compute roof, `bound`, or a
combined roofline, and does not guess the private resolver from the policy. The only fixed-spec
reference kept uniformly is the memory floor:

```text
memory_floor_us = model_bytes / 1792 GB/s
memory_floor_pct = memory_floor_us / median_us
```

## 6. Timing contract

Ordinary single-point, sweep, and suite modes use the same cold-cache timing contract:

1. allocation, payload initialization, and the first public call are not timed;
2. before each sample, L2 is evicted with a fixed 256 MiB buffer;
3. the flush completes before the CUDA event start;
4. the event brackets exactly one complete public Linear call;
5. the primary result is the median, and the min and p95 are also recorded;
6. the default is warmup `3` and repeat `20`, and explicit overrides are allowed;
7. no extra warm-cache measurement is run for each point.

If an independent hot-cache question genuinely arises later, a clearly named opt-in mode should
be added; the doubled cold/warm work must not again become the default behavior of every
command.

## 7. Output

The console header prints, fixed:

```text
gpu=RTX 5090
dram_spec=1792 GB/s
sustained_read=1674.5 GB/s
cache=cold
```

A single result row retains:

```text
label qtype policy N K T median_us min_us p95_us
model_GB effective_GB/s DRAM_% READ_%
useful_TFLOP/s memory_floor_pct
```

A sweep additionally outputs the `delta_%` relative to the adjacent T. The CSV may add the
weight/x/out byte breakdown, warmup, repeat, and environment versions, but does not restore the
following fields:

- measured stream-copy ceiling;
- measured Tensor Core ceiling;
- candidate;
- kernel variant;
- inferred weight replay;
- inferred executed FLOPs;
- duplicated launcher tile metadata.

The benchmark does not need to claim which kernel was selected. A single-point NCU run can see
the real kernel instance, and the production selector source is the only executable authority
for host launcher routes.

## 8. Registration rules

### 8.1 New production shape

Adding a production Linear shape does not require modifying the benchmark:

1. the selector and the public numerical test complete registration;
2. it can immediately be measured with the numeric `--n/--k/--t/--qtype` options;
3. it is added to a model suite only when it becomes a high-frequency or architecturally
   representative geometry of 27B.

A suite entry contains only:

```text
label, qtype, policy, N, K, T class
```

It must not carry launcher, schedule, kernel, tile, Full/Predicated, workspace, or
route-boundary metadata.

### 8.2 New route

Adding or replacing a host launcher does not modify the benchmark. Single, sweep, and suite
always call the public Linear, so they naturally measure the production route currently returned
by the selector.

Candidate selection follows the recommended workflow defined in
[`op-development.md`](op-development.md#7-performance-evidence):

1. within the scope of the development task for that format or route, write a temporary
   benchmark/sweep, calling the private launcher directly if necessary;
2. determine the fixed winner or crossover over the common legal domain of the candidates, with
   the same inputs and cache/timing conditions;
3. write the decision only into the production selector;
4. re-verify correctness and final performance through the public Linear;
5. delete the temporary sweep, the candidate forcing, and any private entry kept only for
   comparison.

Generic candidate forcing, private legal domains, crossover mirrors, launcher catalogs, or the
plan layer must not be reintroduced into the long-lived pure Linear benchmark. The public
`--sweep` only observes the overall behavior of the final selector and does not take on
candidate selection.

### 8.3 New weight/activation compute type

A new benchmark type is added only if all of the following are completed:

1. the public policy corresponds to a truly reachable production path;
2. the packed-weight fixture supports that persistent format;
3. the model-byte formula is explicit;
4. the public numerical suite has been qualified against the routes actually reachable under
   that policy.

A policy is only a permission, and the long-lived benchmark does not pass off the permission
itself as low-precision execution. All current preset suites explicitly use `A16Only`; NVFP4
AllowA4 is kept as explicit points at numeric geometries.

## 9. Current verification

The current implementation has been verified as follows:

1. the `ninfer_linear_bench` Release target builds;
2. the 41 points of the 27B suite all execute through the public Linear;
3. the Text `T=1..3` sweep and the Vision `T=4,8` step-4 sweep both output continuous results;
4. NCU `--profile-from-start off --launch-count 1` captures only the `q4_rowsplit_gemv_kernel`
   instance produced by one public Linear call;
5. the weight-byte result for Q4 `[6144,5120], T=1` is `16711680`, equal to
   `6144*5120*(4/8+2/64)`; the console and the CSV use the same computation;
6. the BF16 exact point above executes through the public Linear; the median of 500 cold-cache
   samples across repeated runs is `95.520–97.536 us`, i.e. `1505.5–1537.3 GB/s`;
7. relative to the `1674.5 GB/s` pure-read probe, this range is `89.91%–91.80%` of the
   attainable read bandwidth;
8. the DRAM read of the final single NCU capture is `146825728` bytes, while one logical pass
   over weight plus activation reads `146810880` bytes; the extra read is only `14848` bytes,
   with no weight replay;
9. the output retains the route-neutral useful TFLOP/s and memory-floor metrics, and does not
    infer the actual activation-compute profile from AllowA4;
10. the A4 Linear of all five NVFP4 exact problems passes the same exact-decode/naive-FP64
    oracle directly at `T=17`, the representative cp.async point, and the main `T=1024` TMA
    point;
11. on the RTX 5090 with CUDA 13.1 and a cold cache, the complete quantization + GEMM results
    for the five NVFP4 A4 `T=1024` problems are:

    | `[N,K]` | Median | Useful throughput | Dense FP4 peak |
    |---:|---:|---:|---:|
    | `[14336,5120]` | `152.576 us` | `985.24 TFLOP/s` | `58.79%` |
    | `[16384,5120]` | `174.784 us` | `982.92 TFLOP/s` | `58.65%` |
    | `[34816,5120]` | `390.112 us` | `935.81 TFLOP/s` | `55.84%` |
    | `[5120,6144]` | `72.992 us` | `882.62 TFLOP/s` | `52.66%` |
    | `[5120,17408]` | `197.888 us` | `922.42 TFLOP/s` | `55.04%` |

    The four previously registered rows retain their 5-warmup/30-sample measurements; the new
    `[34816,5120]` row uses 5 warmups and 40 samples.

The benchmark does not take on numerical correctness; each weight/activation-compute profile
remains the responsibility of the public Linear conformance suite and the unified CPU FP64 GEMM
oracle.
