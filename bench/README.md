# Benchmarks

`ninfer_bench` measures the complete public `ninfer::Engine` route against a `.ninfer` artifact.
The `bench/ops/` `ninfer_<op>_bench` executables measure central public Op contracts while leaving
implementation selection behind those contracts. Target benchmarks measure Program/model
composition. Correctness and model parity live outside this directory; development rules are in
[`../docs/maintainer/op-development.md`](../docs/maintainer/op-development.md).

## Build

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DNINFER_BUILD_BENCHMARKS=ON
cmake --build build --parallel --target ninfer_bench
```

## Product benchmark

The benchmark slices exact token counts from `bench/fixtures/bench_corpus.ids`, calls
`Engine::prepare_tokens()`, then calls `Engine::generate()` once for each repetition. It does not
have a private prefill/decode loop and does not call target implementation interfaces.

The matrix contains three independently measured test kinds:

- `pp{P}` prepares `P` tokens and requests one output token. This is the smallest request that runs
  the model; `prefill t/s` is `P / GenerationTimings.prefill_seconds`.
- `tg{G}` prepares a one-token seed outside the reported phase and requests `G+1` output tokens.
  The begin-round token belongs to prefill, leaving exactly `G` tokens in the reported decode
  phase.
- `pp{P}+tg{G}` uses the same `G+1` convention after a `P`-token prefill and reports both phase
  rates from the same generation call.

All benchmark requests use raw output, disable model-default stops, and disable prefix reuse. This
keeps the requested token count exact without adding another generation path. When CUDA Graph is
enabled and the matrix contains decode work, one ordinary public generation request primes the
decode graph before warmups and measured repetitions.

## CLI

```text
ninfer_bench --weights <artifact.ninfer>
          [--corpus <ids-path>]
          [-p, --n-prompt <list>]
          [-n, --n-gen <list>]
          [-pg, --prompt-gen <P,G;P,G...>]
          [-r, --repetitions <n>] [--warmup <n>]
          [--max-ctx <tokens>] [--prefill-chunk <tokens>]
          [--kv-dtype <bf16|int8>]
          [--mtp-draft-tokens <0..5>] [--lm-head-draft]
          [--device <id>] [--no-cuda-graph] [--profile-measured]
          [-o, --output <table|json|csv>] [--output-file <path>]
```

With no `-p`, `-n`, or `-pg`, the matrix is `pp512` and `tg128`.

Example:

```bash
./build/bench/ninfer_bench \
  --weights out/qwen3_6_27b.ninfer \
  -p 512,2048 -n 128 -pg '2048,128' -r 5 --warmup 1
```

`bf16` selects BF16 KV storage and `int8` selects INT8 group-64 KV storage. MTP is enabled with
`--mtp-draft-tokens`; `--lm-head-draft` selects the optimized proposal head. CUDA Graph decode is
enabled by default.

`--profile-measured` is a benchmark-only profiler boundary. It requires exactly one selected test
and `-r 1`, synchronizes after warmup, and brackets only the measured repetition with
`cudaProfilerStart/Stop`. Use it with an Nsight Systems `cudaProfilerApi` capture range so artifact
load, graph construction, and warmup do not enter topology counts.

## Linear Op benchmark

`ninfer_linear_bench` measures only the public pure `linear()` contract. It supports Q4, Q5, Q6,
W8, registered BF16 weights, and the registered NVFP4 problems. Existing formats use
`--policy a16`; NVFP4 additionally supports `--policy a4`, which lets the production resolver
select the qualified route for each exact geometry and T. LinearAdd, LinearSwiGLU,
LinearPair, Attention/GDN projections, and sparse MoE remain separate semantic Ops and are not
benchmark modes here.

This is a long-lived public benchmark: every timed and profiled point calls `ninfer::ops::linear`
and lets production dispatch choose the implementation. Candidate crossover work uses a
task-local temporary sweep, records the winner or boundary in production dispatch, and deletes the
temporary candidate controls afterward; private launchers and route forcing do not belong here.

Build the benchmark and measure one exact production point:

```bash
cmake --build build --parallel --target ninfer_linear_bench
./build/bench/ninfer_linear_bench \
  --qtype q4 --policy a16 --n 4096 --k 5120 --t 8
./build/bench/ninfer_linear_bench \
  --qtype nvfp4 --policy a4 --n 14336 --k 5120 --t 1024
```

A continuous small-T sweep reuses one packed weight and one maximum-T activation/output
allocation:

```bash
./build/bench/ninfer_linear_bench \
  --qtype q4 --policy a16 --n 4096 --k 5120 \
  --sweep 1:32:1 --csv-out profiles/bench/q4_4096x5120_t1_32.csv
./build/bench/ninfer_linear_bench \
  --qtype q4 --policy a16 --n 3456 --k 1152 \
  --sweep 4:512:4 --csv-out profiles/bench/q4_vision_qkv.csv
```

The registered suites run representative public Linear shapes for one or both exact products.
They are compact performance surveys, not copies of the production selector or numerical test
matrix:

```bash
./build/bench/ninfer_linear_bench --suite qwen3_6_27b
./build/bench/ninfer_linear_bench --suite qwen3_6_35b_a3b
./build/bench/ninfer_linear_bench --suite all
```

For NCU, `--profile` performs setup, warmup, and the L2 flush before enabling the profiler, then
captures exactly one public Linear call. That call may emit one or more production-selected kernel
launches:

```bash
ncu --profile-from-start off --set full \
  ./build/bench/ninfer_linear_bench \
  --qtype q4 --policy a16 --n 4096 --k 5120 --t 8 --profile
```

Every ordinary sample is cold-cache: a 256 MiB L2 eviction write completes before the timed
interval. Reported effective bandwidth uses the encoded weight planes once, one BF16 activation
read, and one BF16 output write. Reported FLOPs are the mathematical `2*N*K*T`; neither metric
copies route-private tile, replay, padding, split, schedule, host-launcher, or kernel-instance
behavior. The fixed RTX 5090 memory reference is `1792 GB/s` DRAM bandwidth. Because `AllowA4` is
a permission rather than an execution-profile label, the long-lived benchmark does not infer or
report private activation compute or Tensor Core utilization. `READ_%` additionally compares the same one-read model
bytes with the measured `1674.5 GB/s` pure-read ceiling from `tools/hbm_bandwidth_probe.cu`; it is
the practical utilization measure for read-dominated points. Physical traffic and instruction
utilization still require NCU.

## GDN control-projection Op benchmark

`ninfer_gdn_gating_proj_bench` measures the registered BF16 control projection. With `--35b
--norm-control`, it measures the complete 35B RMSNorm, normalized hidden output, A/B projection,
gate transform, and beta transform contract. `--candidate auto` uses production dispatch;
`--candidate composed` is the explicit RMSNorm-plus-control comparison. Every row reports the
selected route and transient workspace after a 256 MiB L2 flush.

```bash
cmake --build build --parallel --target ninfer_gdn_gating_proj_bench
./build/bench/ninfer_gdn_gating_proj_bench \
  --35b --norm-control --candidate auto \
  -p 1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16 --warmup 10 --repeat 200
./build/bench/ninfer_gdn_gating_proj_bench \
  --35b --norm-control --candidate composed \
  -p 1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16 --warmup 10 --repeat 200
```

## Gated DeltaNet Op benchmark

`ninfer_gated_delta_net_bench` measures the BF16 Gated DeltaNet contract with state/head dimension
128, batch 1, production Q/K normalization, and any positive, divisible
`value_heads >= qk_heads` mapping. Every measurement is a CUDA Graph replay preceded by a 256 MiB
L2 flush outside the timed interval.

`--running` measures the public running-state entry across recurrent-only, complete 64-token
chunks, and chunked-plus-recurrent-tail routes. `--snapshot` measures the 17-slot snapshot entry over
the production `T=1..16` range; `--qk-norm composed` retains the two-L2Norm comparison.
`--chunked-only` measures the private pre-normalized BF16 pipeline. Adding `--breakdown` reports the
end-to-end pipeline and isolated `prepare_wy_wu`, `state_passing`, and `output` stage timings.
`stage_share_pct` partitions the sum of isolated-stage medians. Each isolated stage receives its own
cold-L2 flush, so `relative_to_e2e_pct` is informative but is not an additive partition of the
pipeline latency.

`logical_bytes` and `logical_gbps` count each contract-visible tensor and state transfer once.
`traffic_bytes` and `traffic_gbps` instead sum one full tensor extent for every kernel input and
output in the selected implementation. This includes repeated consumption by different kernels,
the composed or public chunked Q/K-normalization intermediates, and every producer/consumer access
to chunked `g_cumsum`, W, U, `v_new`, and `h_chunk`. `intermediate_traffic_bytes` isolates those
normalization and chunked-workspace accesses. These deterministic byte counts describe
implementation-level tensor traffic; physical DRAM/L2 sectors and cache reuse still require NCU.

```bash
cmake --build build --parallel --target ninfer_gated_delta_net_bench ninfer_gdn_layer_bench
./build/bench/ninfer_gated_delta_net_bench \
  --running --value-heads 32 --sweep --warmup 20 --repeat 100 --csv
./build/bench/ninfer_gated_delta_net_bench \
  --snapshot --value-heads 32 --qk-norm fused --warmup 20 --repeat 100 --csv
./build/bench/ninfer_gated_delta_net_bench \
  --chunked-only --value-heads 32 --tokens 1024 --breakdown \
  --warmup 20 --repeat 100
./build/bench/ninfer_gdn_layer_bench \
  --t-sweep 1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16 \
  --route fused --norm-control fused --qk-norm fused --warmup 20 --repeat 500
```

## GDN input-projection Op benchmark

`ninfer_gdn_input_proj_bench` measures all registered public `gdn_input_proj` forms: the 27B
Q4/Q5 two-parent projection, the 35B W8 single-parent projection, and the 27B NVFP4 single-parent
projection under either admitted policy. Every timed and profiled point is exactly one public Op
call. Single-parent workspace is queried and allocated through the public capacity entry before
timing; the benchmark has no private launchers, route controls, or candidate mode.

Cold cache is the primary model-layer condition. Reported logical traffic counts encoded weights
once, BF16 input once, and QKV/Z outputs once. FLOPs describe the complete registered projection.
The benchmark reports caller policy rather than inferring a private activation-compute route.

```bash
cmake --build build --parallel --target ninfer_gdn_input_proj_bench
./build/bench/ninfer_gdn_input_proj_bench \
  --format all --tokens 1,2,4,8,12,16,32,64,128,256,512,1024 \
  --cache cold --warmup 5 --repeat 30 \
  --csv-out profiles/bench/gdn_input_proj.csv
./build/bench/ninfer_gdn_input_proj_bench \
  --format nvfp4 --nvfp4-policy a4 --tokens 1024 --cache cold --profile
```

## GDN input projection/convolution/snapshot Op benchmark

`ninfer_gdn_input_proj_conv_snapshot_bench` measures the public Qwen3.6-27B Q4/Q5 and NVFP4
`gdn_input_proj_conv_snapshot` forms. The timed body is exactly one complete public Op call;
the benchmark does not include private launchers, candidate selection, duplicated compositions, or
route labels. Its default `T=1..6` sweep is the production MTP verification interval. NVFP4 accepts
the public `a16` and `a4` policies; the reported profile names the caller policy, not a private
resolved route.

CUDA Graph replay is the default execution mode. The graph contains external timing event nodes
around the complete Op body, while L2 eviction stays outside the timed interval. Cold-cache results
model successive model layers with distinct weights and are the authoritative comparison; warm
results make launch and cache effects visible. The initial state occupies a slot disjoint from all
published snapshot slots, so repeated replay does not introduce a benchmark-only state reset.

```bash
cmake --build build --parallel --target ninfer_gdn_input_proj_conv_snapshot_bench
./build/bench/ninfer_gdn_input_proj_conv_snapshot_bench \
  --format q4q5 --sweep 1:6 --execution graph --cache both \
  --warmup 10 --repeat 100 \
  --csv-out profiles/bench/gdn_input_proj_conv_snapshot.csv
./build/bench/ninfer_gdn_input_proj_conv_snapshot_bench \
  --format nvfp4 --nvfp4-policy a4 --sweep 1:17 \
  --execution graph --cache cold --warmup 10 --repeat 100
```

`--execution eager|both` is available only to attribute launch behavior; it calls the same public
Op with the same operands and workspace.

## Softmax Attention Op benchmarks

The four Softmax Attention executables share one harness rule: fixture setup, public workspace
capacity queries, L2 conditioning, graph capture, and synchronization stay outside the timed
interval. An eager interval contains one public Op call; a captured graph contains that same one
public call. `--profile` likewise brackets one complete public call and requires one exact semantic
point. There are no private headers, launchers, route labels, candidate controls, tile sizes, split
counts, or kernel-name filters in these benchmarks.

`ninfer_causal_softmax_attention_bench` measures the two public causal-cache entries:
append-and-attend and cached-only. It covers the registered D256 H24/KV4 and H16/KV2 geometries
with BF16 and INT8-G64 KV storage. Production dispatch receives the caller-visible execution
envelope and owns all decode, prompt, Small-T, and split-KV choices.

Append-and-attend accepts `--batch 1,2,4,8`; each ordinary `--context L` point gives every row the
same context and all `W` columns are valid. One exact mixed profile uses `--row-contexts`,
`--valid-columns`, and `--table-rows`, each with exactly `B` entries. Cached-only remains B=1.
The timed call consumes the whole batch once; metadata copies and graph capture remain outside the
interval. Uniform full-width profiles use the dense public contract; exact partial profiles use
device-resident valid extents. Reported useful bytes/FLOPs sum only valid row work.

```bash
cmake --build build --parallel --target ninfer_causal_softmax_attention_bench
./build/bench/ninfer_causal_softmax_attention_bench \
  --entry both --geometry all --kv-dtype all --batch 1 \
  --tokens 1,2,4,6,8,12,16 --context 0,128,2048,8192 \
  --execution graph --cache cold --warmup 10 --repeat 61
./build/bench/ninfer_causal_softmax_attention_bench \
  --entry append --geometry d256-h16-kv2 --kv-dtype int8 \
  --batch 3 --tokens 6 --row-contexts 127,2047,63 \
  --valid-columns 6,3,0 --table-rows 2,0,1 \
  --execution graph --cache cold --warmup 10 --repeat 61
./build/bench/ninfer_causal_softmax_attention_bench \
  --entry cached --geometry d256-h16-kv2 --kv-dtype int8 \
  --tokens 16 --context 8192 --execution graph --cache cold --profile
```

`ninfer_context_softmax_attention_bench` measures the public read-only context-plus-query contract
at Q32/KV8/D128 with BF16 context storage. `T` is a complete non-causal query block and `L` is its
external context length.

```bash
cmake --build build --parallel --target ninfer_context_softmax_attention_bench
./build/bench/ninfer_context_softmax_attention_bench \
  --tokens 1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16 \
  --context 0,2048,8192,32768,131072,196608,262144 \
  --execution graph --cache cold --warmup 10 --repeat 61
```

`ninfer_sliding_window_attention_bench` measures the public Q32/KV8/D128 symmetric sliding-window
contract over the 4096-slot cyclic BF16 cache and a complete non-causal query block.

```bash
cmake --build build --parallel --target ninfer_sliding_window_attention_bench
./build/bench/ninfer_sliding_window_attention_bench \
  --tokens 1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16 \
  --context 0,32,64,96,128,4095,4096,8192,262144 \
  --execution graph --cache cold --warmup 10 --repeat 61
```

`ninfer_packed_softmax_attention_bench` measures both public dense Attention overloads: a uniform
plain-segment entry and a packed entry driven by cumulative segment lengths. Equal-length inputs
can compare both public entries; nonuniform inputs select only `packed`.

```bash
cmake --build build --parallel --target ninfer_packed_softmax_attention_bench
./build/bench/ninfer_packed_softmax_attention_bench \
  --entry both --segments 16 --length 256 \
  --execution graph --cache cold --warmup 10 --repeat 61
./build/bench/ninfer_packed_softmax_attention_bench \
  --entry packed --segment-lengths 128,256,384,512 \
  --execution graph --cache cold --warmup 10 --repeat 61
```

The bandwidth and FLOP fields are semantic useful-work models. They do not infer private scratch
traffic, launch decomposition, or selected implementation from kernel names; physical traffic and
instruction utilization require a profiler capture of the complete public call.

## KV cache append Op benchmark

`ninfer_kv_cache_append_bench` unifies the two public append contracts without combining them in
one timed body. `--mode full` calls full D256 KV publication for KV4/KV2 and BF16/INT8-G64 caches.
`--mode prefix` calls device-count prefix publication for BF16 D128/KV8 linear or 4096-slot cyclic
caches; `T` is the public envelope and `C` is the device commit count. Every measured interval or
captured graph contains exactly one selected public append call.

```bash
cmake --build build --parallel --target ninfer_kv_cache_append_bench
./build/bench/ninfer_kv_cache_append_bench \
  --mode full --full-geometry all --kv-dtype all --tokens 1,2,4,8,16 \
  --context 128 --execution graph --cache cold --warmup 10 --repeat 61
./build/bench/ninfer_kv_cache_append_bench \
  --mode prefix --tokens 1,2,4,8,16 --counts 0,1,2,4,8,16 \
  --layout all --execution graph --cache cold --warmup 10 --repeat 61
```

Prefix useful traffic is 8192 bytes per committed token; `C=0` still exercises the public
device-count contract and reports zero useful bytes.

## Masked-block preparation Op benchmark

`ninfer_prepare_masked_block_bench` measures the public exact I32 anchor/mask transform for every
registered `B=2..16`. Eager and graph modes call the same public contract; cold mode performs the
256 MiB L2 eviction before the timed interval.

```bash
cmake --build build --parallel --target ninfer_prepare_masked_block_bench
./build/bench/ninfer_prepare_masked_block_bench \
  --block-sizes 2,3,4,5,6,7,8,9,10,11,12,13,14,15,16 \
  --execution graph --cache both --warmup 20 --repeat 101
./build/bench/ninfer_prepare_masked_block_bench \
  --block-sizes 16 --execution graph --cache cold --profile
```

Useful traffic is `(2+2B)*4` bytes: two device scalar reads and two complete I32 output writes.

## W8 LinearSwiGLU Op benchmark

`ninfer_w8_linear_swiglu_bench` measures the registered W8 `[12288,2048] -> [6144,T]`
LinearSwiGLU profile. Production writes only the fused output and uses no workspace. The explicit
control runs the registered parent `linear` followed by `silu_mul`. Candidate mode retains the
decode, exact-T split-K Tensor Core, and tiled Tensor Core schedules used to tune every dispatch
seam. Every cold-cache sample follows a 256 MiB L2 flush.

```bash
cmake --build build --parallel --target ninfer_w8_linear_swiglu_bench
./build/bench/ninfer_w8_linear_swiglu_bench \
  --t-sweep 1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,32,64,96,128,256,512,896,1024 \
  --warmup 10 --repeat 50 --csv-out profiles/bench/w8_linear_swiglu.csv
./build/bench/ninfer_w8_linear_swiglu_bench \
  --profile --t-sweep 1024
```

## BF16 LinearAdd Op benchmark

`ninfer_bf16_linear_add_bench` measures the contiguous BF16 `[5120,6144]` projection with its
in-place BF16 residual epilogue. Production uses decode at `T=1`, exact-small-T at `T=2..26`, and
MMA from `T=27`; `--route all` retains the legal overlap needed to reproduce that crossover.
Every sample is cold-cache. Effective bandwidth counts the weight once, the activation once, and
the residual read plus write; its `READ_%` and `TC_%` use the benchmark's explicit RTX 5090 BF16
references.

```bash
cmake --build build --parallel --target ninfer_bf16_linear_add_bench
./build/bench/ninfer_bf16_linear_add_bench \
  --sweep 1:32:1 --route all --warmup 10 --repeat 50 \
  --csv-out profiles/bench/bf16_linear_add_t1_32.csv
./build/bench/ninfer_bf16_linear_add_bench \
  --t-sweep 1024,1536,2048 --route production --warmup 10 --repeat 50
./build/bench/ninfer_bf16_linear_add_bench \
  --t-sweep 1024 --route production --profile
```

## W8 LinearAdd Op benchmark

`ninfer_w8_linear_add_bench` measures the W8 `[2048,6144]` projection with its BF16 residual
epilogue. Production updates the residual in place and uses no workspace. The explicit control
runs the same-shape parent `linear` into a scratch allocation followed by `residual_add`.
Candidate mode retains the direct-decode, exact-T split-K Tensor Core, composite exact-T, and tiled
Tensor Core schedules used to tune every dispatch seam. Every cold-cache sample follows a 256 MiB
L2 flush.

```bash
cmake --build build --parallel --target ninfer_w8_linear_add_bench
./build/bench/ninfer_w8_linear_add_bench \
  --production-only \
  --t-sweep 1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,32,33,64,65,96,128,256,640,641,896,960,1024,1280,2048 \
  --warmup 10 --repeat 50 --csv-out profiles/bench/w8_linear_add.csv
./build/bench/ninfer_w8_linear_add_bench \
  --production-only --t-sweep "$(seq -s, 1 2048)" \
  --warmup 3 --repeat 15 --csv-out profiles/bench/w8_linear_add_all_t.csv
./build/bench/ninfer_w8_linear_add_bench \
  --profile --production-only --t-sweep 1024
```

## Attention input-projection Op benchmark

`ninfer_attn_input_proj_bench` measures every registered public `attn_input_proj()` weight/shape
contract: the 27B two-parent Q4/Q5 projection; the 35B W8 Q/K/gate/V and companion Q/K/V
projections; and the 27B BF16 and NVFP4 single-parent Q/K/gate/V projections. Fixture packing and
public workspace capacity queries happen before timing. Every sample and profiler range contains
exactly one public Op call, so production owns format-specific dispatch and launch decomposition.

```bash
cmake --build build --parallel --target ninfer_attn_input_proj_bench
./build/bench/ninfer_attn_input_proj_bench \
  --format all --tokens 1,2,4,8,12,16,32,64,128,256,512,1024 \
  --cache cold --warmup 10 --repeat 50 \
  --csv-out profiles/bench/attn_input_proj.csv
./build/bench/ninfer_attn_input_proj_bench \
  --format nvfp4 --nvfp4-policy a4 --tokens 1024 \
  --cache cold --warmup 10 --profile
```

The stateful GDN projection/convolution/snapshot contract remains in its own public Op benchmark;
it is not a mode of Attention input projection. End-to-end target measurement uses `ninfer_bench`.

## 35B sparse-MoE dFlash benchmark

`ninfer_sparse_moe_bench` measures the complete routed-plus-shared post-mixer Op for the 35B Text
Q4+Q5/Q6 profiles and the MTP W8+W8 profile. It is a long-lived public Op benchmark: fixture setup
uses `sparse_moe_workspace_capacity_bytes()`, and every eager or captured measurement calls only
`ninfer::ops::sparse_moe()`. Production dispatch exclusively owns decode, Small-T, prefill,
workspace views, launch decomposition, and schedule selection.

CUDA Graph replay is the default and authoritative execution mode. The benchmark eagerly
materializes the public call, captures it with stable tensor and workspace addresses, instantiates
the graph, and primes one replay before configured warmup. Timing-enabled external event nodes
surround the captured public call, so `graph_replay` reports the complete device-side SparseMoe
body while excluding fixture reset, L2 eviction, graph capture/instantiation/prime, host launch,
and host synchronization. `eager` uses the same public-call lambda and is only a comparison mode.

```bash
cmake --build build --parallel --target ninfer_sparse_moe_bench
./build/bench/ninfer_sparse_moe_bench \
  --codec q4-q5 --tokens 1 --execution graph --cache both \
  --distribution trace-like --warmup 20 --repeat 200
./build/bench/ninfer_sparse_moe_bench \
  --codec all --sweep 1:44:1 --execution graph --cache cold \
  --distribution trace-like --warmup 5 --repeat 50 \
  --csv-out profiles/bench/sparse_moe_public_graph.csv
```

Each cold sample flushes 256 MiB before the timed interval. `trace-like` is the primary
single-sequence verification distribution; `independent` and `same` bound zero and complete expert
overlap. `--execution eager|graph|both` and `--cache cold|warm|both` change only the harness around
the same public call. The CSV `timed_scope` is `full_sparse_moe_device_body`; it contains no private
route, candidate, grid, block, or launch-count fields.

## Target MTP round benchmark

`ninfer_qwen3_6_27b_mtp_round_bench` measures the registered target's native proposal and
verification round without introducing a second generation controller. It loads the same `.ninfer`
artifact through the target-private package facade, prepares a real prompt with that target's
Frontend, and reports draft/accept statistics for the target-owned MTP schedule:

```bash
cmake --build build --parallel --target ninfer_qwen3_6_27b_mtp_round_bench
./build/bench/ninfer_qwen3_6_27b_mtp_round_bench \
  --artifact out/qwen3_6_27b.ninfer
```

## 35B target-side speculative round benchmark

`ninfer_qwen3_6_35b_a3b_target_speculative_round_bench` measures the shared target verification,
greedy acceptance, accepted-hidden selection, and target-state publication transaction without
including MTP or dFlash proposal work. It uses the real 35B Text weights, target KV cache, and 17
GDN snapshot slots when `K=15`. The default sweep covers `K=1..15` in eager and CUDA Graph modes;
`--accepted-drafts A` forces one acceptance frontier for every selected K:

```bash
cmake --build build --parallel \
  --target ninfer_qwen3_6_35b_a3b_target_speculative_round_bench
./build/bench/ninfer_qwen3_6_35b_a3b_target_speculative_round_bench \
  --artifact out/qwen3_6_35b_a3b.ninfer \
  --context 128 --draft-tokens 7,15 --mode both
```

The reported `target_side_effective_tok_s` is `(A+1)/target-side latency`. It deliberately excludes
proposal generation and dFlash-specific context maintenance and is not an end-to-end speed claim.
`program_workspace_capacity_bytes` is the frozen capacity of the complete DFlash Program plan used
to supply the benchmark's shared arena; it is not a target-round scratch measurement and does not
change the timed scope.

## 35B complete DFlash round benchmark

`ninfer_qwen3_6_35b_a3b_dflash_round_bench` drives the production Program through consecutive
steady DFlash rounds. A measured round includes the previous confirmed feature-to-context append,
the six-layer proposal, target verify/accept, and host publication. It reports GPU and wall latency,
real acceptance, per-position acceptance, mean licensed length, and published tokens/s:

```bash
cmake --build build --parallel \
  --target ninfer_qwen3_6_35b_a3b_dflash_round_bench
./build/bench/ninfer_qwen3_6_35b_a3b_dflash_round_bench \
  --artifact out/qwen3_6_35b_a3b.ninfer \
  --context 4096 --draft-tokens 15 --proposal-head optimized
```

Run separate invocations for `K=1..15`, `full|optimized`, representative contexts, and
`--no-cuda-graph`. The target-side benchmark above supplies forced `A=0..K` transaction evidence;
this complete-round benchmark intentionally preserves the DFlash model's real greedy acceptance.

## Token-decision Op benchmarks

The G1 benchmark covers the Qwen3.6-35B full physical vocabulary with 248077 valid rows at
`C=1..16`, plus the 131072-row shortlist. Its `--control` route reads the same rotating payload and
uses the same launch grid without computing argmax, which provides the fixed-work comparison used
by the benchmark comparison. `--candidate-block` forces a tiled-atomic CTA geometry:

```bash
cmake --build build --parallel --target ninfer_argmax_bench ninfer_sampling_select_bench
./build/bench/ninfer_argmax_bench
./build/bench/ninfer_argmax_bench --control
./build/bench/ninfer_argmax_bench --candidate-block 128
```

The G2/G3 benchmark uses physical rows 248320, valid token domain 248077, optional occurrence
counts, and every MTP window `K=1..5`. With no arguments it runs the full greedy/stochastic matrix;
individual routes are suitable for Nsight Compute capture:

```bash
./build/bench/ninfer_sampling_select_bench --matrix
./build/bench/ninfer_sampling_select_bench --sample --mode stochastic --top-k 20
./build/bench/ninfer_sampling_select_bench --mtp --mode stochastic --mtp-k 5 --top-k 20
```

## 35B dFlash causal Attention qualification

The public causal benchmark covers exact verify widths `W=1..16`, both KV codecs, and the
append-and-attend and already-cached entries for the D256 H16/KV2 geometry. Batched target
qualification uses the append entry:

```bash
./build/bench/ninfer_causal_softmax_attention_bench \
  --entry append --geometry d256-h16-kv2 --kv-dtype bf16 \
  --batch 1,2,4,8 \
  --tokens 1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16 \
  --context 128,1024,8192 --execution graph --cache cold
./build/bench/ninfer_causal_softmax_attention_bench \
  --entry cached --geometry d256-h16-kv2 --kv-dtype int8 \
  --tokens 1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16 \
  --context 128,1024,8192 --execution graph --cache cold
```

The former `attention_layer` executable composed multiple implementation-level stages and exposed
private route controls, so it is not retained as a public Op benchmark. Complete mixer and target
effects are measured through the public Engine benchmark or the target round benchmarks.

## Pointwise Op benchmarks

The Section 5 benchmarks cover the complete Qwen3.6-35B pointwise matrix. Default invocation runs
all registered small, established, maximum-video, and maximum-image shapes. `--control` preserves
the selected kernel topology and payload while replacing the mathematical operation with minimal
bitwise work:

```bash
cmake --build build --parallel --target \
  ninfer_residual_add_bench ninfer_sigmoid_mul_bench \
  ninfer_gelu_bench ninfer_add_bias_bench

./build/bench/ninfer_residual_add_bench [--patches P] [--control]
./build/bench/ninfer_sigmoid_mul_bench \
  [--tokens T[,T...]] [--control | --candidate-block B]
./build/bench/ninfer_position_bench \
  [--tokens T[,T...]] [--candidate-block B] [--cold-graph] [--warmup N] [--repeat N]
./build/bench/ninfer_gelu_bench [--mode tanh|exact --columns C] [--control]
./build/bench/ninfer_add_bias_bench [--d D --columns C] [--control]
```

Aligned registered shapes use 16-byte BF16 packs in the cache-sized regime. GELU and AddBias
select their BF16x2 streaming routes for larger Vision items; odd or unaligned repository-internal
test shapes exercise the scalar fallbacks.

## Reports

Table, JSON, and CSV reports all identify the selected target, artifact, Engine configuration,
load summary, memory capacity, KV payload, workspace peak, phase throughput, and speculative
statistics. JSON schema version 10 records the public value objects directly:

- `load`: target, `weights_id`, load/upload time, file/H2D/staging bytes, tensor count, and resource
  count;
- `memory`: weights/sequence/workspace/request-transient arenas, planned context, KV storage,
  CUDA Graph allowance, and KV payload;
- each repetition's `timings`: prepare, Vision, prefill, decode, and total seconds;
- each repetition's `speculative`: window, rounds, drafted/accepted tokens, fallbacks, and per-position
acceptance.

Each test reports `workspace_peak_bytes` from the planned phase markers, including CUDA Graph
replay, and `workspace_allocator_peak_bytes` from host-side arena allocation activity. These are
intentionally separate: replay reuses captured addresses without advancing the host allocator.

`decode_output_tok_s` counts the requested `G` decode outputs. `decode_engine_tok_s` uses the
Program's speculative round statistics, so it also describes work performed by a final partially
committed speculative round. Reports also contain the command and machine information needed to
interpret a local measurement.

Raw reports and profiler captures remain local under `profiles/bench`, `profiles/ncu`, and
`profiles/nsys`.
