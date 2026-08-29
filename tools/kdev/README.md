# tools/kdev — kernel iteration control plane

Layer 0–3 procedure: [`docs/maintainer/kernel-iteration.md`](../../docs/maintainer/kernel-iteration.md).
Decode-band NVFP4 Linear mapping: [`docs/maintainer/nvfp4-decode-linear.md`](../../docs/maintainer/nvfp4-decode-linear.md).
Fill the gate card (blank procedure, or a Linear point that fills numbers + next commands):

```
python3 -m tools.kdev recipe
python3 -m tools.kdev recipe --preset attn_in --t 1 --idea occupancy
python3 -m tools.kdev recipe --n 14336 --k 5120 --t 1024 --qtype nvfp4 --policy a4 \
    --measured-us 152.6 --idea tile_shape
```

Exit 2 is `STOP` (refused idea or illegal SM120). Do not write CUDA.

## Layer 0 — bound (host)

The bound classifier is Linear GEMM only (`ninfer::ops::linear`). `recipe` with a point
runs it and prints the eight-item card; `bound` is the compact one-screen form.

```
python3 -m tools.kdev bound --preset attn_in --t 1 --idea occupancy
python3 -m tools.kdev bound --n 14336 --k 5120 --t 1024 --qtype nvfp4 --json
python3 -m tools.kdev bound --list-ideas
python3 -m tools.kdev bound --self-test
```

Exit 2 means refuse. Do not write CUDA.

## Layer 1 — MMA issue roof

```
python3 -m tools.kdev mma
```

Writes `profiles/kdev/mma_issue.json`. Bound and recipe read it for `t_issue`.

## Layers 2–3 — public Op at the exact point

Linear is not a kdev `<op>`. Measure it through the public bench (`recipe` prints the
filled command):

```
./build/bench/ninfer_linear_bench --qtype nvfp4 --policy a16 --n 14336 --k 5120 --t 1
./build/bench/ninfer_linear_bench --qtype nvfp4 --policy a4  --n 14336 --k 5120 --t 1024
```

Registered kdev ops:

```
python3 -m tools.kdev <op> [--fast|--full] [--bench] [--profile] [--san] [--json]
python3 -m tools.kdev.diff <op>
```

Every Op run, in the shared dev (builder) container (the host stays free of the CUDA
toolchain):

1. **Incremental build** of the op's test + bench targets (ccache + a persistent
   build tree with `BUILD_TESTING`/`NINFER_BUILD_BENCHMARKS` ON → seconds).
2. **Oracle** — the op's independent FP32/FP64 reference test, with
   `NINFER_OP_REPORT_STATS=1`. `--fast` runs the cheapest case (seconds);
   `--full` runs the representative matrix. The tracked KPI is the worst
   error-to-limit ratio.
3. **Bench** (`--bench`) — the op's CUDA-event median bench (µs + effective GB/s).
4. **Profile** (`--profile`) — one L2-cold `ncu` launch: SM/DRAM/issue SoL.
5. **Sanitizer** (`--san`) — `compute-sanitizer` memcheck (the memory axis the
   numeric oracle can't see).
6. **Verdict** — one JSON (single source of truth) + a JSONL trend row + a one-line
   human table. Perf numbers are only reported as valid when the oracle passed
   (no silent "fast-but-wrong").

Artifacts land in `profiles/kdev/<op>/` (gitignore-able): `<ts>.json`,
`trend.jsonl`, `ncu_<ts>.txt`, `last_dump.json`. MMA calibration is
`profiles/kdev/mma_issue.json`.

## Ops

Add an op to `registry.py` (test/bench targets + argv lists). Currently:
- `l2norm` — fast tier (one cheap case) + bench + **intermediate side-band**
  (`l2norm_dump` → `diff.py`).
- `gqa_attention` — oracle (full conformance) + NVFP4-S3 bench/profile + the
  **sage quality mode** (below). Its intermediate side-band is not ported yet;
  the port target is the base `gqa_attention_prefill_nvfp4s3.cuh` (the TMA file
  includes it, so a fix there is inherited by `..._tma.cuh`).

## Sage (nvfp4s3) quality mode — bug vs floor vs tile-skip

The SageAttention3 route has three loss hypotheses for any PPL regression:
the intrinsic **FP4 P-quant floor** (always on)
and a **kernel bug**. `--sage` separates them with the conformance test's own
oracles (`GQA_SAGE_ONLY=1` + `GQA_SAGE_FLOOR=1`), parsed per case:

```
python3 -m tools.kdev gqa_attention --sage [--fast] [--bench]
```

Per case it reports:
- `floor` = rel_L2(sage_ideal, ideal) — the unavoidable FP4-P cost (~0.053-0.059).
- `dev_vs_step_*` = rel_L2(kernel, step-exact FP64 emulation of the kernel's
  online-softmax + e2m1 loop). A correct kernel sits ~1e-4 from it (its
  residual is FP32 exp2/divide noise); O(0.01+) is a systematic deviation.
- class: `noise` (<1e-3) / `sub-floor-dev` (<max(0.02, floor)) / `>=floor-dev`.

Readings (2026-08-22, git@24c2368): prefill multi-tile A1 cases run at
0.018-0.030 (sub-floor systematic deviation); decode A3 cases run at
0.071-0.082 (>=1x floor). Single-tile A1 cases (keys<=64) are `no-signal`
(the step64 emulation is bit-identical to the closed form there, so its
residual is the expected independent-rounding distance, not a bug candidate).
`--keep-frac` is the exact-NVFP4 Sparge skip (`ninfer-ppl --kv-dtype nvfp4 --keep-frac`);
it is not a Sage3 lever and cannot be combined with `--sage`.

Stage-localization of a deviation: `S3_ORC_DUMP=1` (the test's env, not a
kdev flag yet) per-block-dumps the prefill kernel's P-quant stages (score,
S-arg, e2m1 codes) for the first tile; the P2 side-band port generalizes it
to any tile/block via `tools.kdev.diff`.

## Intermediate side-band (the "op-level values" without editing source)

The kernel exposes named intermediates through a null-guarded dump struct (zero
production cost — the production path passes `nullptr`). The test's `--dump`
mode writes the kernel + FP64-reference intermediates to JSON; `diff.py` reports
the **first divergent stage + index + both values** in one run:

```
$ python3 -m tools.kdev.diff l2norm
[kdev-diff] l2norm case='l2norm [128,16,1]' rows=16 (rel_tol=1e-04 abs_tol=1e-05)
  sumsq  clean (max rel 6.50e-08)
  inv_r  clean (max rel 5.51e-08)
  => no divergence (all stages within tolerance)
```

Porting it to a new op = (1) a `struct <Op>Dump` of named intermediates in the
op's public header, (2) a null-guarded `if (dump)` write in the kernel, (3) a
`<op>_dump` entry point, (4) a `--dump` mode in the test that emits JSON, (5) a
registry entry with `dump=True`. Proven end-to-end on `l2norm` (a injected
wrong-sum bug is localized to `sumsq@5` in a single run).

## Dev container

`./ninfer/dev-setup.sh` (idempotent, never touches the live `ninfer` serve
container) sets up the `ninfer-builder` dev home: ccache, the CUDA-compat strip
(the RTX 5090 host-driver fix), and the tests+bench build tree. Profilers
(ncu/nsys/cuda-gdb/compute-sanitizer) ship in the image.

## Process discipline baked in

Bound classifier → legal SM120 family → fast tests (tiny shapes) → profiler-guided
iteration → **tracked** measurements (`trend.jsonl`) → **rollback rails**
(the dev container is isolated from the live serve; git + the
`local/ninfer:5090-rollback-*` tag pattern) → **no silent fast-but-wrong** (the
oracle must pass before a bench number is reported).
