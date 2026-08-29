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
With a Linear point it fills the eight-item gate card from the bound classifier, prints
the exact `ninfer_linear_bench` command for that point, and exits 2 on `STOP`.

Hardware: RTX 5090, `sm_120a`. Linear byte floors and timing live in
[`linear-benchmark.md`](linear-benchmark.md). Op admission, oracles, candidate-then-delete,
and public-Op measurement live in [`op-development.md`](op-development.md) §6–§7.

Do not use GPGPU-Sim, Accel-Sim, or an uncalibrated Python cycle model to pick a kernel.
Those are slower than silicon and not faithful enough for a 5% `sm_120` NVFP4 decision.

The bound classifier is the Linear GEMM model-byte floor (`ninfer::ops::linear`). It does
not apply to GQA or l2norm; those still use SM120 legality and the public-Op loop.

Decode-band NVFP4 Linear (serving T, CTA mapping, why fused quantize and MMA SASS lost):
[`nvfp4-decode-linear.md`](nvfp4-decode-linear.md).

## Layer 0 — bound (host, no GPU)

Name the public Op, exact `(N,K,T,qtype,policy)`, and prefill vs decode. Run `recipe` as
above, or the compact classifier:

```bash
python3 -m tools.kdev bound --preset attn_in --t 1 --qtype nvfp4 --idea occupancy
python3 -m tools.kdev bound --n 14336 --k 5120 --t 1024 --qtype nvfp4 --measured-us 152.6 --json
python3 -m tools.kdev bound --list-ideas
```

The classifier uses the public Linear model-byte floor, 1674.5 GB/s sustained read, and
1676 TFLOP/s dense FP4. Predicted time is `max(t_mem, t_comp[, t_issue])`.

- **DRAM** (typical decode, `T ≲ 16`, ridge near `T ≈ 380` on 27B NVFP4 attn-in Linear): only ideas that
  cut extra bytes or raise useful work per weight pass (`weight_replay`, `aggregate_T`).
- **tensor-core** (typical prefill chunk `T=1024`): search tile, TMA, pipeline, and epilogue
  fusion inside the existing SM120 NVFP4 family.

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

`python3 -m tools.kdev recipe` with a Linear point fills this card. Any `unknown` means the
task is measurement, not implementation. `STOP` / `MEASURE` / `GO` / `FILL` is the last line.

1. Public Op, exact `(N,K,T,qtype,policy)`, prefill vs decode.
2. `model_bytes`, `useful_flops`, AI, `t_mem`, `t_comp`[, `t_issue`]. Bound.
3. Current measured µs and % of the matching roof (1674.5 GB/s or 1676 TFLOP/s).
4. Idea class. If it does not attack the bound, stop.
5. SM120 legality: MMA atom, smem, TMA, registers. TMEM / `tcgen05` ⇒ stop.
6. Parameters inside an existing family, or why a new family is required.
7. Microbench the public Op at that exact point. No Engine A/B until the Op wins.
8. If it loses, delete the candidate.
