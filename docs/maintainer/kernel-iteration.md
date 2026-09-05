# Kernel iteration (layers 0–3)

This is the required procedure for CUDA speed work on NInfer. The control plane is
`python3 -m tools.kdev`. Fill the gate card before writing CUDA:

```bash
python3 -m tools.kdev recipe
python3 -m tools.kdev recipe --preset attn_in --t 1 --idea occupancy
python3 -m tools.kdev recipe --n 14336 --k 5120 --t 1024 --qtype nvfp4 --policy a4 \
    --measured-us 152.6 --idea tile_shape
```

Without geometry, `recipe` prints the procedure, idea catalog, and public-Op commands.
With a supported projection point it fills the eight-item gate card, prints the exact
public benchmark command for that point, and exits 2 on `STOP`.

Hardware: RTX 5090, `sm_120a`. Linear byte floors and timing live in
[`linear-benchmark.md`](linear-benchmark.md). Op admission, oracles, candidate-then-delete,
and public-Op measurement live in [`op-development.md`](op-development.md) §6–§7.

Do not use GPGPU-Sim, Accel-Sim, or an uncalibrated Python cycle model to pick a kernel.
Those are slower than silicon and not faithful enough for a 5% `sm_120` NVFP4 decision.

Registered Linear cards use the Linear GEMM model-byte floor (`ninfer::ops::linear`). Exact
GGML block-row qtypes are also admitted for `ninfer::ops::ggml_block_linear`, but only their
represented weight/input/output bytes and one-read memory lower bound are modeled. Their scalar
codec instruction cost and physical weight replay are not represented by the dense-FP4 roof, so
their bound is `profile-required`; compute-side ideas remain `measure` until an exact-point NCU
profile establishes physical traffic, instruction-pipe limits, occupancy, and spills. This
classifier does not apply to GQA or l2norm; those still use SM120 legality and the public-Op loop.

Decode-band NVFP4 Linear (serving T, CTA mapping, why fused quantize and MMA SASS lost):
[`nvfp4-decode-linear.md`](nvfp4-decode-linear.md).

## Layer 0 — bound (host, no GPU)

Name the public Op, exact `(N,K,T,qtype,policy)`, and prefill vs decode. Run `recipe` as
above, or the compact classifier:

```bash
python3 -m tools.kdev bound --preset attn_in --t 1 --qtype nvfp4 --idea occupancy
python3 -m tools.kdev bound --n 14336 --k 5120 --t 1024 --qtype nvfp4 --measured-us 152.6 --json
python3 -m tools.kdev bound --n 10240 --k 2560 --t 512 --qtype ggml_q5_k --idea aggregate_T
python3 -m tools.kdev bound --list-ideas
```

For registered Linear, the classifier uses the public model-byte floor, 1674.5 GB/s sustained
read, and 1676 TFLOP/s dense FP4. Its predicted time is
`max(t_mem, t_comp[, t_issue])`. GGML instead reports only the exact one-read lower bound
described below.

- **DRAM** (typical decode, `T ≲ 16`, ridge near `T ≈ 380` on 27B NVFP4 attn-in Linear): only ideas that
  cut extra bytes or raise useful work per weight pass (`weight_replay`, `aggregate_T`).
- **tensor-core** (typical prefill chunk `T=1024`): search tile, TMA, pipeline, and epilogue
  fusion inside the existing SM120 NVFP4 family.
- **profile-required** (GGML block rows): the displayed memory time is only the exact one-read
  representation lower bound. For `T>16`, `aggregate_T` and `weight_replay` may proceed to an A/B
  because they provably reduce packed-weight passes; at `T<=16` the live route already makes one
  pass. Other legal ideas first require exact-point NCU evidence and remain measurement tasks
  rather than classifier approvals.

If `verdict=refuse` or `sm120=ILLEGAL`, stop. Do not write CUDA.

## Layer 1 — SM120 legality and MMA issue roof

RTX 5090 is not B200. Legal: warp `mma.sync` (including `kind::mxf4nvf4`), single-CTA TMA,
accumulators in registers, shared memory ≤ 99 KiB. Illegal: `tcgen05`, TMEM, 2-SM MMA,
cluster-multicast TMA, SM100 128×128 TMEM tiles.

Calibrate `t_issue` once per machine, and again if `ops/common/mma.cuh` changes:

```bash
python3 -m tools.kdev mma
```

This writes `profiles/kdev/mma_issue.json`. Re-run bound or recipe; they pick up NVFP4
`mma_per_s`. The register-only issue roof is the compute floor that matters. 1676 TFLOP/s
is the datasheet fallback used when the probe file is absent.

## Layer 2 — parameter sweep on the GPU

Search parameters inside one family (tile M/N, K stages, warps, pipeline). Do not fork a new
algorithm per idea. Measure through the public Op when the contract is unchanged; a temporary
candidate sweep may call private launchers, then must keep one winner and delete the rest
([op-development.md](op-development.md) §7).

Linear is the bound's subject and is **not** a `kdev <op>`. Measure it through the public
bench at the exact point (`recipe` prints this command filled in):

```bash
./build/bench/ninfer_linear_bench --qtype nvfp4 --policy a16 --n 14336 --k 5120 --t 1
./build/bench/ninfer_linear_bench --qtype nvfp4 --policy a4  --n 14336 --k 5120 --t 1024
```

GGML block-row cards use the exact qtypes `ggml_q8_0`, `ggml_q4_k`, `ggml_q5_k`,
`ggml_q6_k`, `ggml_iq1_s`, `ggml_iq2_xxs`, and `ggml_iq4_nl`. They always have represented
BF16 input (`policy=a16`) and route to their public Op benchmark:

```bash
./build/bench/ninfer_ggml_block_linear_bench \
    --qtype ggml_q5_k --n 10240 --k 2560 --t 512
ncu --profile-from-start off --set full \
    ./build/bench/ninfer_ggml_block_linear_bench \
    --qtype ggml_q5_k --n 10240 --k 2560 --t 512 --profile --repetitions 1
```

Registered kdev ops (currently `l2norm`, `gqa_attention`):

```bash
python3 -m tools.kdev <op> --fast --bench
python3 -m tools.kdev <op> --fast --bench --profile
```

`--profile` / Linear `--profile` is one named NCU question (DRAM vs 1674.5 GB/s, tensor-pipe
busy, spills, DRAM bytes vs `model_bytes`). Not an open-ended report.

Correctness still gates performance: oracle first, then bench. Fast-but-wrong is invalid.

## Layer 3 — production path

Qualify the reachable production route against the independent oracle. Confirm the public Op
bench at the exact point. Engine / `ninfer_bench` / serve A/B only when the requested claim is
end-to-end tok/s.

If the candidate loses, delete it. Do not leave a second path.

## Gate card

`python3 -m tools.kdev recipe` with a supported projection point fills this card. Any `unknown`
means the task is measurement, not implementation. `STOP` / `MEASURE` / `GO` / `FILL` is the
last line.

1. Public Op, exact `(N,K,T,qtype,policy)`, prefill vs decode.
2. Registered Linear: `model_bytes`, useful FLOPs, AI, `t_mem`, `t_comp`[, `t_issue`]. GGML:
   exact representation bytes, useful FLOPs, and one-read memory lower bound; compute unmodeled.
3. Current measured µs and % of the matching registered-Linear roof (1674.5 GB/s or 1676
   TFLOP/s), or GGML one-read represented-byte throughput efficiency against 1674.5 GB/s.
4. Idea class. If it does not attack the bound, stop.
5. SM120 legality: MMA atom, smem, TMA, registers. TMEM / `tcgen05` ⇒ stop.
6. Parameters inside an existing family, or why a new family is required.
7. Microbench the public Op at that exact point. No Engine A/B until the Op wins.
8. If it loses, delete the candidate.
