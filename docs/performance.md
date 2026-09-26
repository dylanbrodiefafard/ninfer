# Single-GPU serving performance

Tested Git revisions:

- Concurrent MTP3 decode saturation for the three measured Qwen3.6 artifact profiles:
  `26da9df7c1b3d3c04ea7bbd730271aa01d00742a`;
- Refreshed Qwen3.6-35B-A3B and Qwen3.6-27B NVFP4 MTP3:
  `f4f21cc36bd1a83cbc046f668719d591dc9c1e2e`;
- Qwen3.6-35B-A3B stored MTP3 response audit:
  `b1a220f028aa750f75bceb3522ac00bbaab7e42d`;
- Qwen3.6-35B-A3B DFlash block=8 (`k=7`):
  `0dc94097e8ec5c5bcf59b9e13e9d1852f504eb61`;
- Qwen3.6-27B NVFP4 accuracy and MTP0:
  `b3d4d0f50b868711c62432bbd68e746217a2f49a`;
- Qwen3.6-27B groupwise-int MTP3: `5ea3242a206cdb0c4c1beaeb9d8a3048e6248423`;
- Qwen3.6-35B-A3B MTP0 and Qwen3.6-27B groupwise-int MTP0:
  `0795169393cab0f2c16246d4bac20dee735dc2a4`;
- Qwen3.8-27B NVFP4 EvalScope accuracy (INT8 and NVFP4 KV):
  `c0f4ec2cfe234b3e3988f79f0399d077de8178b6`.

## Concurrency C=5/6 (2026-09-25)

`max_concurrency` admits 1–6. The C=5/6 verify aggregates (T=W×C up to 36) keep every
request's C=1 arithmetic, so C=1–4 kernels and outputs are unchanged:

- A8 MMA uses one M48 tile for T=33–48 (N=5120 K256×2 stages, wide K128×3), so W=6 C=6 reads
  weights once. The fused A8 GDN conv-record epilogue requires that single tile.
- Aggregate verify extent, fused RMSNorm+SwiGLU, GDN replay/record rows, and GDN gating extend to
  C=6. Gating aggregates every W=2..16 packed verify through T=36 on the T=1-reduction GEMV.
- The W8 verify LM head and the Q4 proposal head aggregate W=2..6 in passes of at most 32
  columns (both keep one K-split reduction per column through T=32). BF16 attention routes the
  W=5 C=5/6 aggregates (T=25/30) to SmallT, which keeps the T=5 panel reduction; the generic
  crossover sends T>22 to MMA. DFlash W=5 drafter projections, now including the feature
  projection, run one pass at C=5/6; only A16 T=20 keeps two T=10 groups.
- Adaptive draft measures every captured k once per batch size. Extrapolating an unmeasured
  arm's T from shorter arms is not a bound: before A8 covered W2/W3, a C=6 k=1 round (36 ms)
  cost more than k=4 (24 ms), k=3..5 were never probed, and C=6 locked k=1 (266 tok/s).

Every aggregate is bit-identical to its per-request panels in the Op tests (B=2..6, W=2..6), and
`ninfer_qwen3_8_27b_dflash_real_test` matches six overlapping greedy requests to their C=1
DFlash streams for k=1..5.

RTX 5090, CUDA 13.1, DFlash2 artifact, `long_decode_aime26_15`, 8192 completion tokens per
request, p-less T2, thinking, NVFP4 KV, max context 16384, KV capacity 16384×C, graphs,
optimized head. Single waves, steady full-batch aggregate decode tok/s:

| Build / mode | C1 | C2 | C3 | C4 | C5 | C6 |
|---|---:|---:|---:|---:|---:|---:|
| Before (`79033d70`), adaptive max5 | 159.1 | 291.9 | 399.5 | 474.3 | — | — |
| C≤6, adaptive max5 | 164.7 | 292.2 | 400.8 | 475.2 | **550.4** | **625.8** |
| Before (`79033d70`), fixed k4 | 160.8 | — | — | 471.1 | — | — |
| C≤6, fixed k4 | 160.6 | — | — | 472.5 | **556.0** | **619.9** |

Adaptive C2–C6 lock k=4 after one round per arm (C1 locks k=5). Aggregate throughput rises
+15.8% from C4 to C5 and +13.7% from C5 to C6; per-request decode is ~104 tok/s at C6. Fixed-k4
kernel time per round is 18.69/20.10/21.34 ms at C4/C5/C6: each added request costs ~1.3 ms
(GDN fold/record, attention, heads). Before the head/drafter/attention single-pass fixes, the
fifth request cost 2.80 ms because those three leaves split into extra weight passes at C=5.
Evidence: `profiles/bench/c6-{base2,fix}-*`, traces `profiles/nsys/c5{why,fix}-k4-c{4,5,6}`
and `profiles/nsys/c6-{k1-c4,k1-c6,k4-c6}` (pre-merge build, A16 W2/W3).

## DFlash GDN chain record and M32 A8 schedule (2026-09-25)

Two bit-exact changes, `qwen3.8-27b/nvfp4` DFlash2 on RTX 5090, NVFP4 KV:

- Chain verification publishes GDN replay records from the register-resident record kernel. The
  T=1 overlay kernel it replaces stored and reloaded each row's 64 KiB FP32 head state in scratch
  after every column, with a block barrier. FP32 store/load is exact and the per-column transition
  is the same width-one arithmetic. The overlay is now tree-only and chain verify reserves no
  overlay scratch. Overlay at C4 was 22.5 µs per layer, 1.08 ms per round.
- N=5120 A8 at M32 (T17–32) streams K512 over three stages instead of K256. Public Linear A8
  T20, including quantization: MLP-down 52.9→44.6 µs, residual-out 24.0→19.7 µs. M32 on the
  N16 grid (320 CTAs) was worse (83.3 µs), so grid underfill is not the limiter.

Engine, same fixture and flags as the section below, single waves, steady decode tok/s. Fixed-k4
and adaptive C1 response hashes match the previous build:

| Mode | C | `fccf6613` | Chain record | + M32 K512 |
|---|---:|---:|---:|---:|
| Fixed k4 | 1 | 159.29 | 161.03 | 161.22 |
| Fixed k4 | 4 | 449.07 | 461.17 | **471.25** |
| Adaptive max5 | 1 | 156.99 | 158.43 | 158.73 |
| Adaptive max5 | 4 | 452.30 | 463.03 | **473.84** |

## DFlash A8 at every verification width (2026-09-25)

Verification now uses the selected A8/A8 profile at W2 and W3 as well as W4–W6
(`kNvfp4FirstA8 = 2`), so a request's arithmetic no longer depends on its draft width. Ordinary
decode (T=1) and prefill keep their text policy. MTP target verification at k=1/2 follows the same
rule. A8 W2/W3 GDN records use the fused projection/convolution tile.

Paired actual-verifier scoring against `fda2972d` (W2/W3 A16), same corpora and method as the W4
study (C1, prefix 8, raw T1 probabilities, NVFP4 KV; 1024-token block bootstrap, 10000 draws):

| Width | Corpus | A16 PPL | A8 PPL | Change | Mean ΔNLL 95% interval |
|---|---|---:|---:|---:|---|
| W2 | WikiText, 32739 tokens | 5.164707 | 5.176074 | +0.220% | [0.00082, 0.00353] |
| W2 | Code references, 8545 tokens | 1.933908 | 1.937614 | +0.192% | [0.00049, 0.00317] |
| W3 | WikiText | 5.164409 | 5.176229 | +0.229% | [0.00085, 0.00358] |
| W3 | Code references | 1.933003 | 1.938109 | +0.264% | [0.00026, 0.00509] |

These match the approved W4 cost (+0.228% / +0.192%). W2/W3 C1/C4 score controls are exactly equal
on all four lanes over 256 tokens, and the greedy, p-less and stochastic 16-round licensing cells
pass at C1/C4 for both widths.

Engine, same fixture and flags as the section below, single waves, steady decode tok/s:

| Mode | C | `fda2972d` | A8 all widths |
|---|---:|---:|---:|
| Fixed k2 | 1 | 134.33 | 135.60 |
| Fixed k2 | 4 | 219.47 | **317.06** |
| Fixed k4 | 1 / 4 | 160.05 / 450.92 | 159.29 / 449.07 |
| Adaptive max5 | 1 | 163.62 | 156.99 |
| Adaptive max5 | 4 | 452.86 | 452.30 |

Fixed k4 hashes are unchanged (W5 is unaffected). Adaptive rarely selects k=1/2 on this workload
(3 of ~3450 rounds at C1, ~270 of ~14400 at C4); at C1 those few rounds change the seed-0
trajectory, and its acceptance (27.5→25.6%) accounts for the whole C1 difference: tokens per round
fall 3.9%, width choices and round cost are unchanged. Adaptive C4 acceptance is 32.1% vs 31.9%.
Fixed k2 accepts slightly less (C4 50.0→47.3%) yet is 44% faster at C4.
Evidence: `profiles/bench/dflash-a8-w23/`, `profiles/bench/dflash-a8-followups/{final,w23a8}-*`.

## DFlash A8 kernel schedule and GDN gating (2026-09-25)

Three bit-exact changes over `c1da30a8`; every fixed-k4 per-request response hash is unchanged.

- **GDN gating a/b projection:** tokens split across `grid.y` in tiles of four, reduced under one
  barrier with the T=1 thread-to-K mapping and shuffle tree. Engine C4 trace: 16.5→6.7 µs per
  T20 call; C1 6.3→5.0 µs.
- **A8 pipeline depth:** A8 stages only NVFP4 weights and FP8 activations (the W4A4 activation
  buffers are gone). N=5120 uses K512×3 at M16 and K256×3 at M32, other M16 schedules K256×3, and
  M32 N64 keeps K256×2. The smaller footprint alone slowed M16 gate/up (more CTAs per SM); three
  stages recover it. Public Linear including quantization: MLP-down T5 57.3→41.0 µs, T20
  63.1→53.2; residual-out T5 26.6→20.5, T20 30.7→24.6.
- **SwapAB for N≥14336 A8 projections:** weight rows on MMA M and eight-token panels on N, skipping
  panels past the valid tokens; per-output K16 FMA order is unchanged. Engine C4: gate/up
  79.9→75.6 µs, GDN input 39.0→37.0 µs.

RTX 5090/SM120a, CUDA 13.1, same DFlash2 artifact, `long_decode_aime26_15`, 8192 completion tokens,
p-less T2, thinking, NVFP4 KV, graphs, optimized head, no prefix reuse; KV 32768 at C1 and 65536
at C4. Single waves, steady full-batch aggregate decode tok/s:

| Build | Fixed k4 C1 | Fixed k4 C4 | Adaptive max5 C1 | Adaptive max5 C4 |
|---|---:|---:|---:|---:|
| `c1da30a8` | 154.13 | 431.31 | 156.76 | 433.09 |
| + gating, A8 stages | 158.59 | 438.97 | 162.77 | 440.29 |
| + SwapAB | **160.05** | **450.92** | **163.72** | **451.60** |

Total: fixed k4 +3.8% / +4.5%, adaptive +4.4% / +4.3% (C1 / C4). Adaptive C4 hashes differ because
timed width selection changes trajectories. Measured non-wins (drafter A8, PDL, in-CTA K split)
and open items are in the [DFlash A8 follow-up plan](maintainer/dflash-a8-followups.md).
Evidence: `profiles/bench/dflash-a8-followups/`, `profiles/nsys/a8-followups-{base,sched,swap}-c{1,4}`.

## DFlash A8 fusion qualification (2026-09-24)

Two measured fusions are retained over the preceding W4-enabled A8/A8 implementation:

- **GDN record, W4–6/C1–4:** A8 projection feeds convolution/record publication through a
  CTA-local FP32 tile. The tile aliases weight staging only after the final K-loop barrier;
  convolution reads it after a further CTA barrier. FP32 current projections and BF16 saved
  history are preserved. A4 and wider A8 record widths retain their existing global staging.
- **Verification post-mixer:** `rmsnorm_linear_swiglu` combines unit-offset RMSNorm and row-wise
  A8 quantization before the existing gate/up/SwiGLU kernel. The normalized input still rounds
  to BF16 in registers. The down projection remains separate. Family planning owns normalized
  scratch; the execution leaf selects the fused Op or the ordinary normalization composition.

### Attribution and public-Op results

RTX5090/SM120a, CUDA13.1, driver580.173.02. Nsight Systems traces of Engine `tg128`, fixed k4,
NVFP4 KV at C1/C4 put the separate GDN convolution kernel at0.7%/0.6% of GPU kernel time.
The 5120-channel quantizer alone accounts for1.1%/0.9%; wide RMSNorm is another2.7%/2.1%,
including uses beyond the selected post-mixer fusion. These traces support modest opportunity,
not a large theoretical Engine multiplier.

The GDN fusion removes an FP32 write/read of `2*T*10240*4` bytes and one launch. A first
candidate used a separate shared-memory tile and regressed W5/C4 to53.25µs: NCU showed
52KiB of static shared memory, restricting residency to one CTA/SM.
Reusing the completed weight staging removed that extra shared-memory allocation and retained
the winner. No weight re-encoding or K-reduction change was introduced.

Cold-cache public-Op medians,3 warmups/50 samples:

| Complete operation | W/C or T | Before µs | Fused µs |
|---|---|---:|---:|
| GDN projection + convolution/record | W5/C1 | 40.960 | 38.912 |
| GDN projection + convolution/record | W5/C4 | 49.152 | 43.008 |
| GDN projection + convolution/record | W6/C4 | 47.104 | 43.008 |
| RMSNorm + A8 gate/up/SwiGLU | T5 | 79.872 | 77.024 |
| RMSNorm + A8 gate/up/SwiGLU | T20 | 92.160 | 88.064 |
| RMSNorm + A8 gate/up/SwiGLU | T24 | 92.160 | 89.344 |

GDN timings use captured full-Op graph replay; normalized SwiGLU uses its public composed/fused
sequence benchmark. W5/C1 GDN transient capacity falls230420→25620bytes, and W5/C4
921680→102480bytes. The adaptive C4 `[2,6]` capacity remains327680bytes because W2's A16
route needs more than fused W6's122976bytes; `[3,6]` excludes that high-water mark.
These are Op scratch sizes, not total Engine memory savings.

### Numerical preservation

The GDN independent projection/convolution/history oracle passes chain/tree, ragged inputs,
every-prefix continuation and interval-capacity execution. The normalized SwiGLU check evaluates
FP64 normalization, the explicit normalized BF16 cast, decoded signed weights and full SwiGLU
formula. Its canonical A8 criterion remains unchanged. Independent FP8 encoding/dequantization
additionally separates codec distortion from arithmetic residual. The residual criterion uses
per-output conditioning/rounding bounds for the implementation's FP32 partials, BF16 gate/up
materialization and BF16 result; it does not round private intermediates in the oracle. Reusing
the old empirical A16 relative-L2 limit was inappropriate for these normalized fixtures: some
legal compound BF16 rounding residuals exceed0.0033. The supplementary bound explicitly
accounts for those stages and the global SiLU derivative bound1.1.

Fused/composed normalized outputs match exactly in tested T4–32 fixtures, including changed-input
graph replay. The standalone Op also passes its independent oracle at T1/T3, where the separate
LinearSwiGLU's A16 fallback is not a same-profile equality control. Production short verification
widths remain unchanged.

All18 before/GDN-only/both × W4/W5/W6 × C1/C4 score controls match exactly over256 unique
tokens. More importantly, **all32739 WikiText and8545 code-reference token NLLs are exactly
unchanged** between the previous implementation and both fusions. PPL remains5.1781168165 /
1.9363044071. These are zero-change fusion comparisons against A8/A8, not new A16-relative
quality measurements or generated-task accuracy claims.

### Engine speed

Same `long_decode_aime26_15` fixture,8192 completion tokens/request, fixed seeds, p-lessT2,
thinking enabled, optimized proposal head, graphs, NVFP4 KV and prefix reuse off. Context32768,
KV32768 at C1 and65536 at C4. Rates are steady full-batch aggregate decode tok/s.

| Mode / candidate | C | Before | After | Gain | Paired waves |
|---|---:|---:|---:|---:|---:|
| Fixed k4 / GDN alone | 1 | 151.75 | 152.41 | +0.44% | 3 |
| Fixed k4 / both | 1 | 151.75 | 153.66 | **+1.26%** | 3 |
| Fixed k4 / GDN alone | 4 | 424.72 | 428.69 | +0.94% | 3 |
| Fixed k4 / both | 4 | 424.72 | 430.44 | **+1.35%** | 3 |
| Adaptive max5 / both | 1 | 154.25 | 156.82 | **+1.67%** | 2 |
| Adaptive max5 / both | 4 | 426.97 | 433.36 | **+1.50%** | 2 |

All26 cells/65 requests completed with zero recovery attempts/cycle exclusions. Fixed-k4 response
hashes match across all profiles and waves. Adaptive C1 hashes match; adaptive C4 hashes differ,
so that comparison includes the timed adaptive policy's trajectory and is not a fixed-trajectory
kernel ratio. GDN-only C1 is a small effect: per-wave gains are0.03%,0.66%,0.62%. Both-fusion
fixed-k4 gains are positive in all waves (C1:0.71–1.77%; C4:1.28–1.44%). Whole-wave means
for both improve53.801→53.133s at fixed-k4/C1 and78.478→77.443s at fixed-k4/C4, separately
from the steady-rate metric. These are workload-specific measurements, not population estimates.

Local evidence: `profiles/bench/dflash-a8-fusion/` contains paired per-token scores,
`quality-report.json`, `speed-report.json`, command records and raw server logs;
`profiles/nsys/a8-fusion-before-c{1,4}` holds the scoped attribution traces.

Final validation: `NINFER_DEV_CONTAINER=ninfer-builder-dylan NINFER_DEV_JOBS=12
./scripts/run-unit-tests.sh` passed **105 tests, 2 artifact-dependent skips, 0 failures**
(400.03s test time). The real DFlash Engine test passed fixed/adaptive graph/eager concurrency,
Vision/MRoPE, terminal delivery and RAM continuation; the DFlash disk restore/continuation
case passed. The reported server configuration—context260000, automatic KV, C4, adaptive
max5, NVFP4 KV, optimized head, RAM32768MiB/disk100000MiB—also reached listening locally
with both fusions (KV553152tokens, graph usage94MiB/240MiB allowance), using a separate test
disk-cache directory. Read-only review closed with no remaining material findings after the
codec-residual and non-monotonic workspace-interval checks were added. Whitespace checks pass.

## DFlash W4 A8 and residual-projection tuning (2026-09-24)

The selected A8/A8 verification policy now covers **k=3/4/5 drafts (W4/W5/W6)**,
including adaptive DFlash with maximum draft5. Request-local W2/W3 stays A16 regardless of
concurrency. Prefill and ordinary decode retain their text policy. NVFP4 weights, FP32
current GDN convolution input, BF16 saved history and FP32 recurrent state are preserved.

### Diagnosis and selected schedules

On RTX5090, SM120a, CUDA13.1, Nsight Compute found the original A8 MLP-down
`[5120,17408]`, T5 kernel launched **80 CTAs for170 SMs**, with no spills and691.10GB/s
DRAM throughput versus1674.5GB/s sustained read. The attention-output `[5120,6144]`
kernel had the same80-CTA grid and35.79% DRAM throughput. The byte-floor classifier's
DRAM label was not evidence that either launch saturated memory. The evidence-gated
`grid_underfill` recipe admits output-row partitioning with the same K reduction and weight pass.

Retained A8 schedule: M16 for T≤16, M32 otherwise, K256 and two stages. The two N=5120
projections use **N16/two warps for M16 and N32/four warps for M32**. Other shapes retain
N64/four warps. Independent oracles qualify both the generic N16 scale layout and contiguous
N32 staging. Exact nibble expansion and per16 FP32 scale accumulation are unchanged.
The MLP-down T5 kernel now launches320 CTAs; profiled DRAM utilization rose39.20%→54.84%,
with no spills. Profiler duration fell72.70→51.97µs; profiler durations are not the public-Op medians.

Cold-cache public Linear medians,30 samples/3 warmups, including quantization:

| Projection | T | Before A8 µs | Selected A8 µs | A16 control µs |
|---|---:|---:|---:|---:|
| Attention output `[5120,6144]` | 5 | 32.384 | 26.624 | 22.528 |
| Attention output `[5120,6144]` | 24 | 34.816 | 30.720 | 73.344 |
| MLP down `[5120,17408]` | 5 | 65.536 | 57.376 | 47.104 |
| MLP down `[5120,17408]` | 24 | 81.536 | 61.440 | 163.456 |

This fixes part of the small-panel loss: **MLP-down is still slower than A16 at T4–6**.
The N32/two-warp and transposed output-major small-panel candidates lost to the retained
schedules and were removed. Broad tile/pipeline changes on the other shapes were not
admitted by the byte-floor gate; no separate codec/scale arithmetic speedup is claimed.
The remaining quantization, K16 instruction and scale-handling costs are included in the Op timings.

At W4, complete GDN record cold graph medians improve49.152→40.448µs at C1 and
116.736→40.960µs at C4 (A16→A8). Its caller-owned workspace grows from zero to184336
and737344bytes respectively; interval planning includes the new W4 route.

### Paired quality and Engine measurements

Same actual-verifier methodology and corpora as below, now scored at **W4**, with matched
before/after gold histories. WikiText has32739 scored tokens; six reset code-reference
documents have8545. These compare W4 A8 against W4 A16, not against the earlier W5 scores.

| Domain | W4 A16 PPL | W4 A8 PPL | Change |
|---|---:|---:|---:|
| WikiText | 5.163988 | 5.175765 | **+0.22808%** |
| Code references | 1.933908 | 1.937614 | **+0.19164%** |

Paired mean ΔNLL is0.00227816 /0.00191459nats, with conditional1024-token block-bootstrap
95% intervals `[0.00076750,0.00367605]` / `[0.00053510,0.00316991]` (10000 draws).
These are PPL diagnostics, not task-accuracy percentages or population-equivalence claims.
W3/W4/W5/W6 C1/C4 score controls match exactly within each profile over256 unique tokens;
before/after W3/W5/W6 are exact too. Six16-round actual-logit licensing cells at W4
(greedy/p-less/stochastic, C1/C4) pass. W4 Linear and fused Ops pass independent mathematical
oracles and codec-residual checks; GDN includes ragged/tree and every-prefix carried-history cases.

Same RTX5090 DFlash2 artifact and `long_decode_aime26_15` fixture as below, p-lessT2/thinking,
NVFP4 KV, graphs, optimized head, no prefix reuse,8192 completion tokens per request.
Before is the selected A8/A8 implementation prior to W4 and row-partition changes.
Steady full-batch aggregate decode throughput:

| Mode | C | Before tok/s | After tok/s | Gain | Alternating paired waves |
|---|---:|---:|---:|---:|---:|
| Fixed k3 | 1 | 145.75 | 151.58 | **4.00%** | 2 |
| Fixed k3 | 4 | 207.01 | 330.28 | **59.54%** | 2 |
| Adaptive max5 | 1 | 149.68 | 154.41 | **3.16%** | 2 |
| Adaptive max5 | 4 | 390.08 | 427.32 | **9.55%** | 2 |
| Fixed k4, schedule-only diagnostic | 1 | 144.68 | 151.51 | 4.72% | 1 |
| Fixed k4, schedule-only diagnostic | 4 | 391.62 | 424.58 | 8.42% | 1 |

All20 cells/50 requests completed with zero recovery attempts/cycle exclusions. Both fixed-k4
pairs have identical per-request response hashes. W4 arithmetic and adaptive choices can change
trajectories in the other comparisons; their rates are measured Engine tradeoffs, not isolated
kernel ratios. These workload-specific results do not imply adaptive always beats fixed draft widths.
Whole-wave makespans are retained separately from steady throughput in the report.

Local evidence: `profiles/bench/dflash-a8-w4/{before-ops.json,final-ops.json,quality-report.json,
speed-report.json}`, paired scorer TSVs, command records and raw server logs.

Final checks: `NINFER_DEV_CONTAINER=ninfer-builder-dylan NINFER_DEV_JOBS=12
./scripts/run-unit-tests.sh` rebuilt the normal binaries and passed **105 tests, 2
artifact-dependent skips, 0 failures** (482.54s test time). The explicit real DFlash Engine
test passed fixed k1–5, adaptive max5, graph/eager/concurrent execution, Vision/MRoPE,
terminal publication and RAM continuation; `disk_real --case dflash` passed disk restore
and continuation. W4 scorer checks with commit limit2 passed at C1/C4. Classifier/recipe
self-tests, Python compilation and whitespace checks passed. Read-only review found no
implementation blocker; its W4 codec-criterion and stale-contract findings were corrected.

## DFlash A8 and A16 GDN pairing qualification (2026-09-24)

**Selected default: ordinary A8 / GDN A8 verification projections.** Following qualification,
the user approved this measured PPL/throughput tradeoff. A8 operates over the
same NVFP4 artifact: row-scaled E4M3 activations, exact E2M1-code expansion, FP8 K16 MMA and
original per16 weight scales applied to FP32 partials before accumulation. It does not
requantize or persistently repack the weights. GDN convolution consumes FP32 current
projection; BF16 history/records and FP32 recurrent state are preserved.

Hardware and artifact are the RTX5090/SM120a/CUDA13.1 and exact DFlash2 NVFP4 artifact
identified in the A4 qualification below. Verification precision is selected from request-local
width: this initial qualification kept W2–4 A16 across concurrency and changed eligible W5/W6 verification
projections; prefill, ordinary decode and the existing attention Q codec stay common.

### Paired verifier PPL

Same actual-verifier gold-history scoring inputs and domain as the A4 study below: W5/C1,
prefix8 plus one unscored anchor, raw temperature1 probabilities over248077 valid vocabulary
rows, NVFP4 KV. Code consists of the same six independently reset reference files, not newly
generated model output. Old A/B/C results are retained as paired controls.

| Ordinary / GDN verification projections | WikiText,32739 tokens | Code references,8545 tokens |
|---|---:|---:|
| A16 / A16 (A) | 5.166109 | 1.931212 |
| A4 / A16 (B) | 5.232096 | 1.953540 |
| A4 / A4 (C) | 5.245608 | 1.959898 |
| A8 / A8 | **5.178117** | **1.936304** |
| A4 / A8 | 5.222942 | 1.954479 |
| A8 / A16 | 5.178623 | 1.936100 |

All-A8 raises PPL versus A by **0.23244% on WikiText and0.26368% on code**, compared with
C's approximately1.54% and1.49%. Its paired mean ΔNLL is0.00232171 and0.00263334nats,
respectively. Fixed-seed10000-draw/1024-token within-document block-bootstrap95% intervals
are `[0.00096601,0.00368203]` and `[-0.00022994,0.00506721]`. These are conditional diagnostics
on these corpora, not population guarantees or proofs of equivalence.

With ordinary projections already A8, adding GDN A8 changes PPL by−0.00978% on WikiText
and+0.01055% on code; both paired ΔNLL intervals include zero. With ordinary projections
A4, GDN A8 versus GDN A16 changes PPL by−0.17495% /+0.04808%, also with intervals spanning
zero. Most mean loss still comes from the ordinary A4 projections. These measurements do
not establish an improvement in generated-task success rates.

### Engine throughput

Same `long_decode_aime26_15` decode-saturation fixture and fixed benchmark seeds as the earlier
A/B/C controls: p-less temperature 2, thinking, optimized proposal head, CUDA Graphs, NVFP4 KV,
no prefix reuse, context 32768, KV capacity 32768 at C1 or 65536 at C4. Every request produced
8192 completion tokens (8191 decode tokens). All GPU runs were sequential.

Fixed k4, mean aggregate **steady full-batch decode tokens/s**:

| Ordinary / GDN projections | C1 | C4 | C4/C1 | New measurement waves |
|---|---:|---:|---:|---:|
| A8 / A8 | 144.60 | 392.61 | 2.72× | 3 |
| A4 / A8 | 161.77 | 480.13 | **2.97×** | 3 |
| A8 / A16 | 132.19 | 325.37 | 2.46× | 1 diagnostic wave |

All-A8 C4 waves were 393.959, 391.904 and 391.966 tok/s; GDN-only A8 waves were
480.494, 480.636 and 479.261 tok/s. Earlier same-workload controls below are A at
138.28/198.73, B at 148.75/380.37 and C at 158.50/498.88 tok/s (C1/C4). Thus all-A8
offers **4.6% higher C1 throughput and 97.6% higher C4 throughput** than A, with much smaller
PPL change than C. GDN-only A8
retains approximately 96% of C's C4 throughput while reaching approximately 3× C4/C1 scaling.
The arithmetic-profile comparisons include different generated trajectories and speculative
acceptance; they are measured Engine tradeoffs, not isolated kernel speedup ratios.

The separate **A16 W6 pairing** comparison fixes k5 and ordinary A4/GDN A16. Three alternating
old/new pairs measured **267.866→284.025 tok/s at C4 (+6.03%)**. Old waves were
268.045/267.777/267.777 and new waves 284.019/284.170/283.886. Per-request response hashes
match in every pair. Mean whole-wave makespan improves 122.478→115.317 s. These fixed-k5
figures must not be compared as the same workload as the fixed-k4 table above.

The new campaign has **20 completed cells /59 completed requests**, zero recovery attempts
and zero cycle exclusions. Whole-wave makespan remains distinct from the steady rate:
all-A8 C4 averages 84.783 s and GDN-only A8 C4 69.121 s. Commands, request accounting,
intervals and source server logs are retained in `speed-commands.json` and `speed-report.json`
under the local evidence directory below.

### Public-Op latency and the separate A16 batching control

Cold-cache medians,20 samples after3 warmups, complete public Op including activation
quantization. Principal Linear points at aggregate T24:

| Linear `[N,K]` | A16 µs | A4 µs | A8 µs |
|---|---:|---:|---:|
| Attention input `[14336,5120]` | 110.592 | 34.432 | 43.008 |
| GDN input `[16384,5120]` | 153.600 | 38.912 | 45.056 |
| MLP gate/up `[34816,5120]` | 260.128 | 69.632 | 86.016 |
| Output `[5120,6144]` | 73.728 | 18.432 | 34.816 |
| MLP down `[5120,17408]` | 163.840 | 40.960 | 81.920 |

A8 is faster than A16 at these aggregate points, but slower than A4. It is not universally
faster than A16 at C1: at T5 the residual/output examples are22.528→31.360µs and
47.104→65.536µs. The complete GDN record Op costs40.960µs with A8 at W5/W6 C1 and
49.152µs at C4, versus roughly38.912–40.960µs with A4.

For the **A16-only GDN batching change**, full C4 register panels lost and were removed.
The retained change pairs requests at W6/C2 and C4; W6/C3 stays request-indexed. In the
public GDN record Op, W6/C2 improves118.784→100.352µs and W6/C4 improves215.040→172.032µs
(about15.5% and20.0% lower latency). This does not change activation precision. Paired
W6/C4 verifier scoring over8192 unique tokens finds **zero changed token NLLs**, with exact
equality in all four replicated lanes; old/new PPL is6.00350. Replicated lanes do not count
as additional independent quality samples.

The initial A8 schedule used M16 through T16, otherwise M32, N64/K256, four warps and two
stages. Earlier K128/M32-only qualification schedules were replaced after public-Op timing;
their K16 partial accumulation order is unchanged. Final W4/W5/W6 C1/C4 score controls match
exactly, supporting the retained long-run PPL evidence. Independent mathematical/codec checks
cover all five Linear shapes, fused attention/add/SwiGLU and GDN record state transitions.
The all-A8 real Engine graph/eager/concurrency/RAM and DFlash HostDisk routes also pass.

After integration, `NINFER_DEV_CONTAINER=ninfer-builder-dylan NINFER_DEV_JOBS=12
./scripts/run-unit-tests.sh` passes **105 tests, with 2 artifact-dependent skips and no failures**
(392.61 s test time). The retained real-model activation fixture, using patterned weights,
gives maximum GDN group relative-L2 canonical/codec/residual errors of
0.036199/0.0359501/0.00254837 for A8, versus0.131225/0.131411/0.00269852 for A4. This is
operator evidence, not a substitute for the real-checkpoint PPL above. Seven Python report
tests, classifier/recipe self-checks and `git diff --check` also pass. These qualification checks
preceded the final selection of A8/A8; preserved profile binaries remain local comparison artifacts.

After selecting A8/A8, the same full unit command rebuilt the normal CLI/server and test binaries
and again passed **105 tests, 2 artifact-dependent skips, 0 failures** (383.70 s test time).
With `NINFER_QWEN3_8_27B_NVFP4_DFLASH_WEIGHTS` set to the artifact below, the rebuilt
`ninfer_qwen3_8_27b_dflash_real_test` and `ninfer_qwen3_6_27b_disk_real_test --case dflash`
both passed, covering graph/eager execution, fixed/adaptive concurrency, Vision/MRoPE,
terminal publication, RAM restore and disk continuation under the selected default.

Local evidence: `profiles/bench/dflash-a8-qualification/quality-report.json`,
`op-report.json`, `op-final-report.json`, paired TSVs and real-Engine logs. The canonical
oracle checks and codec-residual criteria establish numerical implementation correctness;
they do not replace the paired PPL evidence or select an acceptable quality budget.

## DFlash A4 verification qualification (2026-09-23)

**Historical A/B/C comparison.** The numerical and performance evidence below distinguishes
three arithmetic profiles. The subsequently qualified A8/A8 default is documented above.

| Profile | Eligible ordinary verification projections | GDN input projection |
|---|---|---|
| A | A16 | A16, FP32 current projection |
| B | A4 | A16, FP32 current projection |
| C | A4 | A4, FP32 current projection |

All preserve BF16 convolution history/records and FP32 recurrent state. C does not introduce
the extra BF16 current-projection rounding used by the older 503.95 tok/s diagnostic below.
Short request-local widths W2–4 retain A16; the measured A4 chain widths are W5/W6.
These profiles describe target projections: the existing NVFP4 attention Q codec is common
to all three, including A.

### Matched Engine throughput

RTX 5090, driver580.173.02, CUDA13.1, `sm_120a`, exact local artifact
`/models/qwen3.8-nvfp4-flash2-nvfp4-bf16codebook-from-bf16/qwen3_8_27b_nvfp4_dflash_nvfp4.ninfer`.
P-less temperature2, thinking, optimized proposal head, CUDA Graphs, NVFP4 KV and no prefix
reuse. Context32768; KV capacity32768 at C1 and65536 otherwise. Each request produced8192
completion tokens (8191 decode tokens). Three main-fixture waves alternate C/B/A, A/B/C,
C/B/A execution order with the benchmark's fixed seed set. All GPU work was sequential.

Mean aggregate **steady full-batch decode tokens/s**, `long_decode_aime26_15`:

| Policy | Profile | C1 | C2 | C3 | C4 | C4/C1 |
|---|---|---:|---:|---:|---:|---:|
| Fixed k4 | A | 138.28 | 184.70 | 200.23 | 198.73 | 1.44× |
| Fixed k4 | B | 148.75 | 257.49 | 330.69 | 380.37 | 2.56× |
| Fixed k4 | C | 158.50 | 288.01 | 405.75 | 498.88 | **3.15×** |
| Adaptive max5 | A | 144.67 | 190.14 | 207.72 | 222.50 | 1.54× |
| Adaptive max5 | B | 151.97 | 252.72 | 332.00 | 377.82 | 2.49× |
| Adaptive max5 | C | 156.87 | 283.78 | 384.86 | 481.73 | **3.07×** |

GDN A4's incremental C4 gain over B is **31.2% fixed /27.5% adaptive**; C1 gains are
6.6% /3.2%. Fixed C4 C waves are499.30,498.58,498.77tok/s. Adaptive C4 C waves are
474.60,495.44,475.15tok/s: a4.33% peak-to-trough spread relative to the mean, associated
with both round duration and token yield/histogram changes. This is not evidence that
adaptive universally beats fixed k4.

One second-fixture wave, `long_decode_aime26_30`, checks workload dependence:

| Policy | A C1/C4 | B C1/C4 | C C1/C4 |
|---|---:|---:|---:|
| Fixed k4 | 148.41 /211.39 | 163.93 /410.42 | 174.35 /508.85 |
| Adaptive max5 | 147.40 /227.75 | 156.29 /401.77 | 168.34 /524.46 |

There are84 complete cells and210 completed requests; recovery attempts and cycle exclusions
are zero. Selected steady intervals span45–158s/cell and cover90.1–98.2% of decode tokens.
Whole-wave makespan is a different metric: C fixed C4 on the main fixture averages67.75s,
about484 completion tokens/s, versus498.88 steady decode tokens/s. Adaptive probe/switch
work is included in observed rates/makespans but not separately timed. Memory records report
startup allocations/graph allowance, not an independently sampled whole-run GPU peak.

The command manifests, individual waves, makespans, round durations, yields, acceptance,
width histograms and memory records are in
`profiles/bench/dflash-a4-qualification/{phase-e-commands.json,phase-e-report.json,phase-e-wave*/}`.
The local controls are `/build/apps/ninfer-serve-phase-e-{a,b,c}` in `ninfer-builder-dylan`;
they were built from the same production source with explicit temporary arithmetic-policy
edits, then preserved before reverting those edits. The benchmark used Python3.11.16 at
`/opt/ninfer-python311/bin/python3.11` inside that builder. These are inference-level
comparisons: generated trajectories can differ across profiles, so gains are not isolated
kernel speedups at identical intermediate inputs. Read-only phase review reconciled every
cell with its server log and approved the measurement scope.

### Paired quality evidence

Actual target Verify and production replay commit, raw temperature1 full-domain logits,
identical reference tokens/commit boundaries, NVFP4 KV:

| Material | Unique scored tokens | A PPL | B PPL | C PPL | C/B change |
|---|---:|---:|---:|---:|---:|
| WikiText stream | 32739 | 5.1661087 | 5.2320963 | 5.2456080 | +0.2583% |
| Six local code-reference files | 8545 | 1.9312122 | 1.9535397 | 1.9598982 | +0.3255% |
| Six AIME problem statements | 690 | 2.3395555 | 2.3592293 | 2.3673919 | +0.3460% |

Conditional1024-token block-bootstrap95% intervals for **C−B mean NLL** are
[−0.0003091,+0.0056656] on WikiText and[−0.0028405,+0.0091058] on code. These do not establish
equivalence or broad-domain confidence. The original8192-token WikiText prefix showed
+1.0140% C/B PPL; it overlaps the extended sample. Code C−B p99 token delta is+0.8990 NLL
and maximum absolute delta3.1994, so the mean does not eliminate individual-token tails.
WikiText lacks retained original document boundaries; code files are independently reset,
but human authorship/training-held-out provenance is unverified. AIME input-text NLL is
**not worked-solution likelihood or solve accuracy**.

### Behavioral evidence and decision limit

Fixed k4, p-less temperature2, matched seeds/settings and the same artifact. With thinking
and8192 output tokens, every profile solved the same2/6 selected AIME questions; four math
cases and both code tasks exhausted their budgets without completing. They count as failures.

To obtain executable-code evidence, the same two local tasks were also run with thinking
disabled at seed12345. Its observed failures triggered a matched extension with seeds23456
and34567 declared before that extension, applied to every profile. This is an outcome-triggered
diagnostic extension, not a prospectively fixed independent benchmark. Initial failures were retained; code was
neither repaired nor regenerated until passing. Existing seed/visible/hidden tests validated
the reference solutions (94 LRU tests,100 limiter tests). Each generated solution passes only
if all tests pass within60s under Python3.11.

| Full code-task passes | A | B | C |
|---|---:|---:|---:|
| LRU cache, three seeds | 2/3 | 3/3 | 1/3 |
| Fair weighted limiter, three seeds | 0/3 | 1/3 | 0/3 |
| Combined | 2/6 | **4/6** | **1/6** |

C emitted syntax errors on two LRU samples and timed out in two limiter test runs. This is
a meaningful observed warning, but only two tasks/three seeds—not a population estimate or
proof that GDN A4 generally reduces coding accuracy. B's better result than A also demonstrates
the sensitivity of this small stochastic sample. Preserve both the thinking-budget failures
and this separate non-thinking diagnostic when judging the tradeoff. Complete responses,
answer keys, grading inputs and completed test logs are under
`profiles/bench/dflash-a4-qualification/behavior/`.
Timed-out jobs retain their tested modules and failure outcomes, but partial pytest output
was not retained by the timeout handler; no timeout was rerun merely to obtain a log.

Core qualification includes independent GDN/codec oracles, actual candidate-logit greedy and
p-less/top-k/top-p licensing, matched-budget C1/C4 isolation, graphs/eager, RAM/HostDisk
continuation and near-capacity width transitions. Cross-trajectory NLL and ordinary-A16
greedy margins remain reported diagnostics, not invented quality thresholds. Both the
additional average PPL cost and the small coding comparison must inform the user's final
GDN A16/A4 decision; the throughput result alone does not select the default.

## DFlash2 concurrent long-reasoning decode (2026-09-22)

RTX 5090, driver 580.173.02, CUDA 13.1, `sm_120a`; artifact
`/models/qwen3.8-nvfp4-flash2-nvfp4-bf16codebook-from-bf16/qwen3_8_27b_nvfp4_dflash_nvfp4.ninfer`.
The workload is the repository's `long_decode_aime26_15` fixture (335 prompt tokens), one
concurrent wave of C requests with the benchmark's fixed per-request seeds, thinking enabled,
and 8192 output tokens per request. Every request reached that output limit. Sampling is the
serving default p-less at temperature 2.0, with NVFP4 KV, CUDA Graphs, the optimized proposal
head, `--max-context 32768 --kv-capacity 65536`, and prefix reuse disabled.

These are **aggregate committed decode tokens per wall-clock second** from complete steady
one-second serving intervals at full occupancy, not sums of per-request GPU rates. Each cell
is one long wave, not a repeated-sample mean; different draft widths and adaptive decisions
can produce different continuations. The full-wave makespans were 83–164 seconds. There were
no generation-recovery retries in these waves.

| Configuration | C=2 tok/s | C=3 tok/s | C=4 tok/s |
|---|---:|---:|---:|
| Prior fixed DFlash k=4 | 186.15 | 200.58 | 198.93 |
| Prior adaptive DFlash, captured k=3/4/5 | 182.69 | 207.03 | 199.66 |
| Retained fixed DFlash k=2 | **197.91** | **209.93** | 217.74 |
| Retained adaptive DFlash, captured k=1/2/3/4/5 | 191.47 | 204.28 | **221.94** |

For this long AIME workload, fixed k=2 is the measured recommendation at C=2/3; adaptive
with `--draft-tokens 5` was fastest at C=4. Relative to prior fixed k=4, those configurations
improve throughput by 6.3%, 4.7%, and 11.6%. Adaptive alone is not a uniform win: its C=3
result is 1.3% below the prior adaptive wave. This is not evidence of a universal 250 tok/s
ceiling or a claim that every reasoning workload now exceeds it.

### Attribution and retained fix

All baseline steady intervals had average decode batch exactly C. A five-second C=4
CUDA-Graph node trace attributed 33.3% of kernel time to target NVFP4 SwiGLU, 22.0% to
MLP-down, 16.1% to GDN input, and 7.4% to attention/GDN output projections. GPU kernels
occupied approximately 98% of the capture. Attention, p-less selection, and HTTP/host
scheduling were not the dominant costs. A focused NCU check of the T=20 SwiGLU route found
20.4% DRAM throughput, 63.0% SM throughput, and zero local-memory spill sectors: the wider
A16/SIMT verification work is not saturating the card's DRAM bandwidth. At k=4 the long
waves accepted only about 31–32% of drafts, yielding about 2.25–2.27 tokens per row-round
while verifying five columns per request.

The missing short-width aggregation made reducing k unnecessarily expensive: W=2/3 MLP
and attention-input projections still launched separate request panels. The retained leaf
uses the already-qualified aggregate A16 Ops at these widths; NVFP4 attention/GDN residual
projections also aggregate W=2..4. It preserves the C=1 arithmetic policy and the existing
BF16/FP8 panel boundaries. Adaptive DFlash now captures k=1/2 as well, so it can select these
cheaper rounds. Fixed k=4 execution is unchanged.

Matched 2048-token-per-lane AIME waves isolate the aggregation change at fixed widths:

| Draft k | C=2 prior → retained | C=3 prior → retained | C=4 prior → retained |
|---|---:|---:|---:|
| 1 | 141.4 → 180.9 | 146.4 → 204.1 | 152.6 → 222.6 |
| 2 | 163.7 → 200.6 | 165.2 → 210.9 | 171.1 → 222.4 |
| 3 | 184.3 → 186.5 | 198.9 → 201.9 | 206.5 → 211.9 |

All nine pairs preserved per-seed response hashes and exact speculative rounds, drafts,
accepts, and fallback counts. Public SwiGLU at W=2 C=4 fell from 272.4 us for four panels
to 104.4 us for one aggregate. Attempting to share GDN weights across all four requests
instead of pairs regressed the public Op and was removed. The remaining scaling limit is
primarily target projection execution; this change does not remove that cost.

Validation: independent Op oracles for attention input, NVFP4 LinearAdd and SwiGLU;
real-artifact greedy C=4 isolation at k=1..5; adaptive short-budget and RAM-restore
continuations against cold recomputation; and automatic-capacity execution with a
259744-token prompt and forced `1,2,3,4,5,4,5,3,2,1` transitions at every C=1..4.
The full unit suite completed with 104 passes, two artifact-dependent skips, and no failures.
The expanded adaptive set reserves 240 MiB of graph allowance at C=4, versus the previous
144 MiB; automatic KV sizing accounts for the additional 96 MiB.

Reproduce the retained fixed-k=2 measurement inside the GPU builder:

```bash
python3 -m tools.bench.run_serve_concurrency \
  --serve /build/apps/ninfer-serve \
  --artifact qwen3_8_27b=/models/qwen3.8-nvfp4-flash2-nvfp4-bf16codebook-from-bf16/qwen3_8_27b_nvfp4_dflash_nvfp4.ninfer \
  --mode dflash2 --sampling p-less --suite decode-saturation \
  --concurrency 2 --concurrency 3 --concurrency 4 \
  --decode-tokens 8192 --saturation-thinking \
  --max-context 32768 --kv-capacity 65536 --kv-dtype nvfp4 \
  --output /src/profiles/bench/dflash-concurrency-fixed2 --port 18082
```

For adaptive, replace `--mode dflash2` with `--mode dflash5 --adaptive-draft`. Explicit
`--sampling p-less` matters: the benchmark's historical `stochastic` default selects the
top-p/top-k sampler. Reports are under `profiles/bench/dflash-concurrency-20260922-` with
suffixes `pless-baseline`, `adaptive-baseline`, `short-widths`, `short-aggregate`,
`fixed2-final`, and `adaptive-aggregate`; the node trace is
`profiles/nsys/dflash-concurrency-20260922-c4.nsys-rep`.

### Scaling follow-up: verification arithmetic policy

The preceding short-width fix does not explain or solve the dominant scaling loss. A matched
C=1 measurement and two isolated arithmetic-route controls identify it as a fork-specific
verification policy, rather than an unavoidable DFlash concurrency ceiling. The same artifact,
8192-token AIME waves, p-less temperature 2, NVFP4 KV, graphs, and optimized proposal head
were used. C=1 used a 32768-token KV capacity, within its one-request capacity ceiling;
C>1 used 65536. Actual request contexts stay below 9000 tokens.
Each point is one wave; generated continuations can differ across arithmetic profiles.

For context, the source repository's [published Qwen3.8 NVFP4 MTP3 saturation results](https://github.com/Neroued/ninfer/blob/9e163eee/docs/performance/qwen3.8-27b.md#decode-saturation)
are 143.8 / 267.6 / 461.1 tok/s at C=1/2/4, or 3.21x C=1 at C=4. These are historical
stochastic MTP3 measurements, not a fresh same-sampler DFlash A/B. Current upstream code
at `9e163eee` consumes the artifact's activation policy for the aggregate FFN, and permits
materialized batched NVFP4 GDN projection. It does not impose this fork's blanket A16
verification override.

Fork commit `f41e4fc3` introduced that override in `text_policy()` for p-less stability.
It selects A16 from the C=1 verify width even when the aggregate contains 10/15/20 columns,
and applies regardless of the request's sampler. Subsequent changes restored many aggregate
launches but retained scalar A16 arithmetic. Separately, `nvfp4_gdn_conv_resolve_plan()`
forces packed widths through SmallT A16 even when its caller permits A4; the recurrent
record path consumes an unmaterialized FP32 projection. These choices disproportionately
increase the cost of C>1 verification.

Public-Op medians, microseconds, for DFlash k=4 (W=5):

| Op / arithmetic | T=5 (C=1) | T=10 (C=2) | T=15 (C=3) | T=20 (C=4) |
|---|---:|---:|---:|---:|
| SwiGLU A16 | 90.11 | 114.72 | 167.58 | 213.02 |
| SwiGLU A4 | 71.68 | 71.68 | 71.68 | 71.68 |
| MLP-down LinearAdd A16 | 49.15 | 65.54 | 83.97 | 145.41 |
| MLP-down LinearAdd A4 | 40.96 | 40.96 | 40.96 | 40.96 |

The A4 timings include the existing activation quantization and use warp tensor-core MMA.
Increasing C quadruples useful columns without increasing their measured Op latency in this
band. The A16 route remains scalar FP32 accumulation over decoded NVFP4 weights.

Two diagnostic binaries isolate these policies on the current fork. The first permits A4 for
NVFP4 target verification in `text_policy()` while leaving GDN records unchanged. The second
also routes GDN records through the existing aggregate `gdn_input_proj()` and
`compose_record()` path, with planner-accounted projection workspace. That second control
introduces BF16 projection materialization before convolution as well as A4 activation
quantization. Neither diagnostic is a qualified production replacement.

| Fixed DFlash k=4 route | C=1 tok/s | C=2 tok/s | C=3 tok/s | C=4 tok/s | C4 / C1 |
|---|---:|---:|---:|---:|---:|
| Production A16 | 139.69 | 186.15 | 200.58 | 198.93 | 1.42x |
| Diagnostic A4 projections, existing GDN | 149.65 | 257.78 | 330.62 | 379.70 | 2.54x |
| Diagnostic A4 projections + composed GDN | 154.62 | — | — | 503.95 | 3.26x |

The final control measured only C=1/4, sufficient to resolve the remaining scaling question.
Complete steady batch-round times were respectively 16.28→45.44 ms, 15.04→23.41 ms, and
14.07→17.97 ms from C=1 to C=4. Production yields stayed almost identical at 2.274 versus
2.260 committed tokens per row-round, ruling out collapsing acceptance as the cause of its
poor scaling. The final control yielded 2.175/2.264 tokens per row-round: its round-cost
scaling alone supports about 3.13x throughput at equal yield. GPU scheduling and the DFlash
proposer implementation were held fixed.

Adaptive was tested with a maximum of **five** draft tokens. The qualified production path
measured 146.08 / 191.47 / 204.28 / 221.94 tok/s at C=1/2/3/4, only 1.52x scaling. Selecting
shorter widths reduces work but does not remove the arithmetic-route bottleneck.

**Production correction:** qualify a consistent batched tensor-core verification profile,
including GDN projection/convolution, against the independent numerical oracles and real
p-less/recurrent-state behavior. The two controls establish the performance cause and
available headroom; their acceptance rates and completed output budgets do not establish
accuracy equivalence. A blind removal of the precision guard is not a validated fix.
The diagnostic source edits were removed and the production server rebuilt. The earlier
short-width aggregation and adaptive-set changes remain the qualified implementation.

Reports: `profiles/bench/dflash-scaling-20260922-{c1-fixed4,c1-adaptive5,a4-projections,a4-gdn}`.
The C=2..4 production points reuse the preceding long-wave reports. Diagnostic executables
are `/build/apps/ninfer-serve-scaling-a4-projections` and
`/build/apps/ninfer-serve-scaling-a4-gdn` in the measurement builder; they are not CLI modes.

## DFlash2 automatic KV capacity

Qualified on RTX 5090, driver 580.173.02, CUDA 13.1, `sm_120a`, using
`qwen3.8-nvfp4-flash2-nvfp4-bf16codebook-from-bf16/qwen3_8_27b_nvfp4_dflash_nvfp4.ninfer`.
The server workload used `--max-context 260000 --kv-capacity auto --max-concurrency 4
--prefill-chunk 4096 --kv-dtype nvfp4 --spec dflash --draft-tokens 5 --adaptive-draft
--lm-head-draft`, temperature 1.5, 32768 MiB RAM cache, and 100000 MiB disk cache.

This qualification reserved 12 MiB per DFlash2 `(K, B, topology)` graph executable, or 144 MiB
for the three-width adaptive set used then, instead of inheriting the
autoregressive DFlash allowance of 1152 MiB. The current five-width set reserves 144 MiB
at C=4 (`min(12n, 24+6n)` MiB for `n` executables); the capacity numbers below describe the
earlier set.
With the same reported 13.26 GiB free after weights, server startup resolved 558656 tokens,
up from the reported 501312-token configuration: the 1008 MiB reduction buys exactly 896
64-token pages (57344 tokens). Measured graph usage was 54 MiB and free memory after startup
was 1.05 GiB. The 1 GiB sizing headroom and address-stable workspace/KV allocations are unchanged.
A separate cold single-request startup with fixed `K=5` used 4 MiB of its 12 MiB graph allowance
and completed generation, covering the smallest executable-count budget.

`ninfer_qwen3_8_27b_dflash_capacity_real_test` passed with automatic capacity, 68 full
4096-token prefill chunks, a 259744-token prompt, and forced live `K=3,4,5,4,5,3` transitions
at every batch size 1–4. Allocator-observed workspace peak stayed within 610.3 MiB; free device
memory changed from 1110 to 1108 MiB across the complete prefill/decode exercise. These forced
host selections test memory and graph transitions independently of timing-based adaptive policy.
The opt-in reproduction command is in `tests/README.md`.

## Selective FP8 328 MiB qualification (2026-09-11)

The eight-matrix recipe and CPU reproduction command are documented in
`docs/maintainer/qwen3.8-27b-artifact.md`. It preserves the original BF16 protections,
W8 endpoints, NVFP4 GDN matrices and draft. The additional payload is approximately
328 MiB, not the total artifact size (19,934,025,728 bytes with the tested DFlash draft).

Retained matched WikiText measurement: 8,192 fixed input IDs, first 4,096 unscored,
4,095 teacher-forced targets, prefill chunk 4,096, dense NVFP4 KV, no speculation:

| weight recipe | WikiText PPL |
|---|---:|
| original NVFP4 | 7.254364 |
| selective 61 MiB | 7.217649 |
| selective 285 MiB | 7.181243 |
| selective 328 MiB | 7.184892 |
| publisher mixed FP8/NVFP4 | 7.095132 |

These are corpus-specific measurements, not an overall quality score. Do not mix
them with earlier chunk512 results (~6.5–6.7). Coding PPL is lower because the token
distribution differs; absolute PPL values cannot be compared across corpora.

The coding selection used separate fixed selection and reserved windows. For the
328 MiB recipe, two NInfer selection windows scored 1.744490 and 1.625217; reserved
XGrammar C++ and Transformers Python windows scored 1.676018 and 2.088383. Corresponding
base scores were 1.767637, 1.654772, 1.676165 and 2.095829; publisher mixed scores were
1.744090, 1.624042, 1.666637 and 2.050691. The larger publisher profile is better on
the reserved pair, but adds approximately 3,022 MiB rather than 328 MiB.

Expanded frozen validation subsequently scored 36 files across six projects
(147,420 targets). The 328 MiB recipe's equal-project/file NLL aggregation gives
PPL 1.661752. Adding MLP down40 gives 1.660320; adding GDN output36 gives 1.662353;
adding both gives 1.661147. All paired baseline bootstrap intervals include zero,
including the corrected comparisons and whole-project bootstrap. These additions
are not established upgrades. Across-file PPL SD for the 328 MiB model is 0.451107;
that describes corpus heterogeneity, not seed noise or uncertainty in a paired
difference. The selected recipe is a practical memory/quality tradeoff, not a
proof of globally optimal layer allocation or absence of reasoning loops.

RTX 5090 / CUDA 13.1 Engine throughput, graphs and NVFP4 KV, one warmup and three
measured repeats, fixed benchmark corpus:

| workload | original NVFP4 tok/s | 328 MiB tok/s |
|---|---:|---:|
| C2 pp4096 aggregate | 11,827.52 | 11,467.78 |
| C2 DFlash5 pp4096+tg256 aggregate decode | 381.33 | 373.63 |
| C1 target-only tg256 | 84.83 | 83.15 |

The DFlash corpus gives 97% acceptance for both models; these rates do not predict
difficult coding-chat throughput. This measures a roughly 2–3% cost, not an FP8
speedup. All raw reports retain commands, per-repeat values and standard deviations
under `out/fp8-328-release/`; the older PPL reports and per-token losses remain under
`out/selective-fp8-{campaign,coding,remaining,validation}/` in the qualification checkout.
Redundant experiment model binaries were removed, not their measurement evidence.

Beyond synthetic Op oracles, 88 real-weight fused projection/residual/SwiGLU checks
used captured BF16 inputs and independent FP64 formulas without emulating private
activation quantization. All passed their unchanged precision-profile criteria;
64 output rows were sampled with full-output finite scans. Real DFlash C4 isolation,
adaptive drafting, RAM restoration and vision/XAttention integration also passed.
P-less likelihood, C2/C1 identity, eager mixed-frontier and turn-checkpoint reuse
during peer decode passed with the retained artifact. The full C++ suite passed
103 tests with two expected artifact-dependent skips; converter/codec Python tests
passed six cases.
These finite checks do not clear the separately recorded NVFP4 private-query
attention error or imply exact equality between sparse and dense attention.

## V4 SSD page-spill qualification

The 2026-09-02 page-only qualification used an RTX 5090, CUDA 13.1, the local ZFS mirror, NVFP4
KV geometry, incompressible deterministic data, and the exact 33,981-token workload: 531 logical
pages, 1,148,387,328 payload bytes, and 1,151,143,936 capacity-billed bytes per repetition. Each
three-repetition sample synchronized the pool before reading device counters and again after the
cache commit. Physical bytes are the sum of both NVMe leaf deltas divided by the explicitly
configured mirror replication factor of two; samples are accepted only when that normalized value
is within 3% of the cache's durable-extent accounting.

| Page batch | Host-visible commit GB/s | Post-pool-sync payload GB/s |
|---:|---:|---:|
| 1 | 3.716 | 2.357 |
| 4 | 4.119 | 2.507 |
| 8 | 3.471 | 2.244 |

Batch four is the production setting. Its clean aggregate was 6,920,171,520 leaf-device bytes,
3,460,085,760 mirror-normalized bytes, and 3,453,431,808 cache-accounted bytes. The dataset had
`sync=disabled`, so the host-visible commit number is not a crash-durability result. The post-sync
number includes an explicit pool sync and is the relevant physical write-through result for this
dataset, but no mirror-normalized v3 post-sync measurement survives for a same-scope comparison.
The historical v3 host-visible result was approximately 3.6–3.8 GB/s.

The same campaign measured pinned H2D at 28.94 GB/s, D2H at 28.67 GB/s, and packed-page
H2D-plus-scatter at 28.11 GB/s (97.1% of pinned H2D), so the page copy/scatter route is not the
remaining restore bottleneck. Direct-reader scaling at 1/4/8/16/32 readers was
3.85/11.99/15.63/17.64/17.57 GiB/s; production retains 16 readers. Warm 531-page restore
diagnostics varied from about 11.3 to 19.5 GB/s. They validate the route and concurrency but are
not cold-device claims because the available host could not evict ZFS ARC and the dataset uses
`primarycache=all`.

A follow-up V4-compatible qualification used the benchmark's exact 27B linear-attention geometry:
153,944,064 raw state bytes and a one-page prompt. One serial direct read plus serial CRC measured
2.617 GB/s. Four 4 MiB read/CRC chunks in flight measured 10.87–13.89 GB/s across the retained
runs (4.2–5.3×); more per-object workers contended, so production caps each large state object at
four while sharing the startup-fixed reader budget across independent objects. The exact 531-page
startup audit fell from a 259.1 ms mean to 65.6 ms (3.95×) by reading each page record once and
validating independent CRCs across that same bounded pool. This startup measurement was
warm/ARC-resident; it isolates validation overhead and is not a cold-device startup claim. Startup
still rejects corrupt live payloads before advertising an entry.

A second follow-up isolated CRC32C and state-worker dispatch. Alternating 1 GiB CRC sweeps on the
Ryzen 9 7950X3D measured the serial SSE4.2 chain at 14.0–14.3 GB/s. Three interleaved hardware
chains measured 22.5–23.1 GB/s at 1 MiB, 29.9–34.5 GB/s at the production 4 MiB chunk, and
28.8–32.2 GB/s at 256 MiB. The combine overhead lost below 512 KiB, so production retains the
serial path for smaller records and switches at 512 KiB. In the clean exact-state run this changed
restore from 13.06 to 14.87 GB/s (+13.8%), and the warm/ARC-resident 531-page startup audit mean
fell from 65.6 to 59.1 ms (-9.9%). Reusing the existing restore threads through a persistent
state-task queue instead regressed paired state restores by 3.4% and 7.1% (13.54→13.08 and
14.73→13.69 GB/s); that candidate was rejected and the transient four-worker state path remains.

Compaction of 312,246,272 retained bytes measured 2.57–2.65 GB/s against a 2.9–3.27 GB/s direct
write control. Increasing its transfer buffer from 1 MiB to 8 MiB regressed from 2.65 to 2.48 GB/s,
so the production buffer remains 1 MiB and no extra pipeline is retained. The final physical
batch-four rerun measured 3.944 GB/s host-visible commit and 2.448 GB/s post-pool-sync payload,
with every mirror-normalized sample within the 3% accounting bound. The accepted state and startup
changes did not alter page-record transfer layout; disk format v5 later changed only the startup
fingerprint from physical tensor geometry to the canonical logical-page schema. Contiguous page
reads, compaction, H2D/scatter, and durable writes remain at or close to the relevant measured
hardware/filesystem ceilings.

The serving measurements characterize the two measured Qwen3.6 model IDs independently on one
NVIDIA GeForce RTX 5090. They cover long-context prefill and baseline decode with speculative
decoding disabled, plus long-reasoning and cross-scenario decode with MTP and DFlash. The 27B
results report its `groupwise-int` and `nvfp4` weight profiles separately. The concurrent
decode-saturation campaign measures the same three Qwen3.6 artifact profiles at C=1, 2, 4, and 8.
A separate C=1 Qwen3.8-27B NVFP4 campaign below compares MTP0/3/5 with DFlash2 k=7 on the same
frozen AIME command (INT8 KV). A later NVFP4-KV DFlash2 campaign measures isolated CLI C=1 and
serve C=1/2/3 after fused batched GDN conv-record. Qwen3.8-27B NVFP4 accuracy uses
[Ostfralla/Qwen3.8-27B-NVFP4-NInfer](https://huggingface.co/Ostfralla/Qwen3.8-27B-NVFP4-NInfer)
with INT8 and NVFP4 KV.

The single-request corpus requests were submitted serially to a persistent `ninfer-serve` process
over the loopback OpenAI-compatible HTTP endpoint. Each reported corpus fixture used five fixed
seeds. Values are arithmetic mean ± sample standard deviation, and server warm-up completes before
the measured requests. The concurrent campaign has its own sustained-wave method below.

## Qwen3.8-27B NVFP4 host-overhead campaign

The 2026-09-01 non-kernel campaign fixed the supported target to
`qwen3.8-27b/nvfp4`, one RTX 5090, DFlash k=4, NVFP4 KV, CUDA Graphs, prefix reuse disabled,
`--prefill-chunk 4096`, and startup concurrency C=1, 2, 3, or 4. The artifact was
`out/qwen3_8_27b_nvfp4_dflash_nvfp4_codebook.ninfer`. The benchmark-reported toolchain was CUDA
13.3 compile/runtime with driver API 13.0. Greedy fixed waves used exact work counters and compared
output hashes as multisets because concurrent HTTP admission may permute requests across lanes.

The configuration gate selected DFlash k=4 at every C. Its external decode rates were 209.97,
328.37, 391.60, and 467.91 tokens/s for C=1-4. The following ratios are candidate/prior external
throughput; a value of 1.0 is neutral. Each row used the immediately preceding retained binary.

| Host change | C=1 | C=2 | C=3 | C=4 | Decision |
|---|---:|---:|---:|---:|---|
| Lock-free hot runtime counters and transition-only full snapshots | 0.9982 | 0.9980 | 1.0002 | 0.9982 | Retained: removes per-round locked lane/KV scans; no material regression |
| Periodic reporter no longer calls `memory_summary()` | 1.0000 | 0.9996 | 0.9996 | 0.9993 | Retained: removes reporter contention with the executor; throughput-neutral |
| Terminal-only publication for non-stream requests | 0.9971 | 1.0003 | 0.9988 | 1.0000 | Retained: removes per-round event lock/allocation/wakeup; exact outputs |
| Stable-decode dirty signal | 1.0027 | 0.9997 | 1.0012 | 1.0013 | Retained: skips unchanged queue/deadline/admission scans |
| Cached Program graph selection | 0.9973 | 0.9978 | 0.9998 | 1.0026 | Reverted: regressed C=1/2 and added invalidation state |
| Direct `(K,B,frontier)` graph-routing table | 0.9999 | 0.9989 | 0.9995 | 0.9999 | Reverted: O(1) lookup produced no end-to-end win |

Two fresh cumulative A/B passes against the preserved pre-campaign binary, in opposite binary
orders, produced retained/baseline throughput ratios of `0.9987/1.0014/0.9994/0.9995` and
`1.0030/1.0014/1.0013/1.0001` at C=1-4. Their per-C geometric means are
`1.0009/1.0014/1.0003/0.9998`: C=2's +0.14% result repeated, but no material uniform
end-to-end speedup is established. Exact work and output-hash multisets matched. The cumulative
host changes are retained for removing unnecessary executor-side contention and publication work,
not claimed as a general decode-speed improvement.

The terminal-only SSE control A/B preserved output hashes, event counts, and ordering. Mean
first-output changes for C=1-4 were +0.62, +0.49, +0.71, and -0.03 ms; worst event-gap changes were
below 0.9 ms. A 7,705-token prefill follow-up removed the full transition snapshot after the first
nonterminal 4,096-token chunk. A fresh C=4 paired repeat produced candidate/prior prefill-speed
ratios of 0.9991, 1.0003, 0.9997, and 1.0008 across the four serialized request positions and a
0.03% lower wave makespan, with exact work and output multisets.

Allocation and representation gates rejected final-string preallocation, stop-token lookup, and
stop-string matcher changes: representative final output growth copied only about 8 KiB per
2,048-token request, the artifact has exactly two default stop token IDs, and the fixed workload
has no stop strings. Consumer cancellation polling was retained because `CancellationView` has no
owning notification hook; replacing it would add cross-layer lifetime machinery without improving
the GPU launch path.

The graph-profile gate separately tested tighter DFlash context envelopes while preserving every
target-declared implementation-transition boundary. Relative to the existing profiles, maximum
spans of 2,048, 1,024, and 512 tokens produced C=1-4 external-throughput ratios of respectively
`1.0001/1.0002/0.9995/1.0004`, `1.0020/1.0010/0.9995/1.0006`, and
`1.0017/0.9974/0.9960/0.9965`. Exact work and output hashes matched. The 2,048 and 1,024 results
were neutral at measurement scale and 512 regressed C=2-4, so the existing transition-derived
profiles remain selected; startup graph update compatibility is not treated as evidence that a
profile is performance-optimal.

Primary reports are under `profiles/bench/host-overhead-*`. The baseline gate is
`host-overhead-mode-gate-baseline-20260831`; retained decode A/Bs end in
`memory-candidate-greedy-20260831`, `runtime-stats-causal-greedy-20260831`,
`terminal-only-candidate-greedy-20260831`, and `stable-decode-candidate-greedy-20260901`. The final
prefill repeat is `host-overhead-prefill-stats-{prior,candidate}-rerun-c4-thinking-greedy-20260901`.
The graph-routing baseline and profile sweep are under `profiles/bench/graph-routing-baseline-*`
and `profiles/bench/graph-profiles-span{2048,1024,512}-candidate-*`.
The fresh cumulative reports are
`host-overhead-cumulative-{baseline,candidate}{,-rerun}-greedy-20260901`.

## Qwen3.8-27B NVFP4 C=1-4 kernel campaign

The 2026-09-05 campaign used the Qwen3.8-27B DFlash2 artifact with NVFP4 matrices, BF16
selector codebook, NVFP4 KV, CUDA Graphs, the optimized proposal head, greedy sampling, a
16,384-token context ceiling, and 2,048 generated tokens per lane on one RTX 5090. The retained
changes group only classifier-qualified fixed shapes while preserving each request's C=1-width
reduction domain: W=5 GDN input/control, selected DFlash4 NVFP4/Q4 Linear calls, the W=5 W8
target vocabulary projection, and the Q4 DFlash proposal head at T=12/16. Unsupported shapes
continue through the existing per-request panels.

Against the preserved `40f6e4c5` binary, aggregate decode throughput measured from exact decode
tokens divided by complete-wave makespan changed as follows. Response hashes, speculative work,
and acceptance counts matched in every cell.

| Backend | C | Prior tok/s | Retained tok/s | Change |
|---|---:|---:|---:|---:|
| DFlash4 | 2 | 226.49 | 245.37 | +8.33% |
| DFlash4 | 3 | 226.96 | 265.92 | +17.17% |
| DFlash4 | 4 | 223.28 | 260.04 | +16.46% |
| MTP4 | 2 | 219.28 | 229.18 | +4.52% |
| MTP4 | 3 | 223.41 | 254.50 | +13.92% |
| MTP4 | 4 | 236.78 | 251.85 | +6.37% |

The final configuration sweep compared DFlash4 and MTP3/4/5 at every supported concurrency.
These are aggregate complete-wave rates, not per-lane rates.

| Configuration | C=1 | C=2 | C=3 | C=4 |
|---|---:|---:|---:|---:|
| DFlash4 | 163.63 | 245.37 | 265.92 | 260.04 |
| MTP3 | 154.05 | 203.12 | 213.72 | 218.73 |
| MTP4 | 149.26 | 229.18 | 254.50 | 251.85 |
| MTP5 | 139.48 | 174.71 | 179.00 | 181.39 |

DFlash4 was fastest at every C, by 6.22%, 7.06%, 4.49%, and 3.25% over the next configuration.
This supports DFlash4 as the explicit speed configuration for this measured greedy AIME
workload and artifact; the sweep did not cover other prompts, sampling policies, or DFlash draft
widths. It does not change the product default because speculative execution still requires an
explicitly compatible artifact and changes the numerical execution route. MTP5's C4 greedy
response differed from its C1-3 response, but the preserved prior binary produced exactly the
same two hashes, acceptance, and fallback counts; this is not a regression from the campaign.

Operator qualification measured the W=5 C3 GDN input-projection/conv-record Op at 147.456 to
116.736 us and the W8 target vocabulary projection at 849.920 us for T=5/10/15 and 855.712 us
for T=20. The GDN record route passed decoded-weight FP64 dense/ragged/tree oracle checks;
aggregated Linear and control routes passed their independent mathematical oracles and exact
packed-versus-panel checks. The final DFlash4 real-artifact C4 isolation passed in both Graph and
eager execution, and MTP4 passed a B4 partner/permutation isolation gate. Target-only C1,
speculation-disabled decode PPL was byte-identical across all 2,047 scored NLL values to the
pre-campaign reference (PPL 6.414141594, no non-finite values); that PPL run does not exercise
concurrent speculative verification.

CUDA 13.1's parallel split optimizer was also found to emit two materially different SASS
variants from identical CUDA input. Three clean `--split-compile 0` compilations of the GDN
snapshot translation unit alternated between 96- and 121-register decode kernels; three
`--split-compile 1` compilations were stable. Replacing only the slow snapshot object with the
stable object changed C=1 MTP4 from 144.59 to 150.64 tok/s (+4.19%) with exact output,
acceptance, and work counts. The build therefore defaults `NINFER_NVCC_THREADS` to 1. Ninja
continues to compile independent translation units concurrently, while an explicit value of 0
retains the faster but nondeterministic within-translation-unit build mode.

Reports are under `profiles/bench/multi-request-kernel-final-config-sweep-clean-20260905`,
`profiles/bench/multi-request-kernel-final-config-sweep-clean-mtp5-rerun-20260905`,
`profiles/bench/multi-request-kernel-last-commit-baseline-20260905`, and
`profiles/bench/multi-request-kernel-mtp5-baseline-isolation-20260905`, plus
`profiles/ppl/multi-request-kernel-final-20260905.json`.

## Qwen3.8-27B NVFP4 mixed-phase prefill campaign

The 2026-09-05 follow-up kept the preceding DFlash4/MTP4 decode implementation and targeted
prefill latency when one request enters while another request is decoding. Tests used the same
Qwen3.8-27B DFlash2 artifact, NVFP4 KV, CUDA Graphs, and one RTX 5090. The retained GDN prefill Op
writes the causal-convolution result directly to compact Q/K/V planes, eliminating a
`[10240,T]` intermediate and three device-to-device splits in each of 48 GDN layers. This campaign
evaluated an 8,192-token maximum. A remaining unit above 4,096 that is not aligned to the 128-token
kernel schedule is decomposed as a 4,096-token unit plus its tail; aligned extents retain the
single 8,192-token route. The current product default is 4,096 following the later 32k
memory/performance A/B below; 8,192 remains an explicit maximum-speed override.

The public causal-convolution Op benchmark at the production 10,240-channel partition measured
the direct split against the ordinary Op plus the three production-shaped copies. Representative
means were 108.10 to 59.36 us at T=2,048, 218.69 to 134.85 us at T=4,096 (-38.3%), and 441.25 to
266.40 us at T=8,192 (-39.6%). Whole-prefill profiling attributed the gain to eliminating the
copies: the replacement convolution kernel itself was slightly slower, while device-to-device
copy time fell from about 4.59 ms to 0.26 ms per profiled pass.

With the explicit 8,192 maximum and adaptive decomposition, public-Engine prefill results were:

| Startup C | 7,669-token owner ms | Aggregate tok/s | 8,192-token owner ms | Aggregate tok/s |
|---:|---:|---:|---:|---:|
| 1 | 708.831 | 10,819 | 695.170 | 11,784 |
| 2 | 710.758 | 21,580 | 697.484 | 23,490 |
| 3 | 711.069 | 32,356 | 698.983 | 35,160 |
| 4 | 712.920 | 43,029 | 700.311 | 46,791 |

The Engine serializes prefill ownership, so the table's aggregate column is C times the per-owner
rate rather than C simultaneous prefill kernels. Startup C=4 added only 0.58% per-owner latency at
7,669 tokens and 0.74% at 8,192 tokens relative to C=1. At 8,192 tokens, the complete retained
route improved the preceding default-4,096 binary from 725.237 to 695.170 ms at C=1 (+4.15%) and
729.699 to 700.311 ms at C=4 (+4.03%). The direct-store fusion contributes about 0.9% at this
length; the larger aligned unit supplies the remainder. For the irregular 7,669-token prompt,
adaptive decomposition improves the static-8,192 candidate from 727.047 to 707.508 ms (-2.69%).

The staggered C=2 HTTP workload started a 7,669-token newcomer while a 1,024-token DFlash4 donor
was decoding. Relative to the preceding default-4,096 binary, newcomer prefill changed from
733.857 to 726.006 ms (-1.07%) and TTFT from 743.222 to 734.647 ms (-1.15%). Mean donor
interruption changed from 741.624 to 740.254 ms; its median changed from 744.792 to 734.282 ms.
All donor and newcomer hashes matched exactly.

The new split route was checked directly against the independent complete FP64 causal-convolution
and SiLU oracle at the production Q/K/V partition for T=1, 7, 65, and 257. Final BF16 convolution
state and input/output guards matched exactly. The packed real-artifact DFlash4 C=4 Graph suite
passed chain, adaptive-length, terminal/queue, reseed, and in-flight restore isolation. Target-only
PPL retained all 2,047 NLL values byte-for-byte (PPL 6.414141594; no non-finite values). A matched
DFlash4 serve rerun preserved every output hash and the 50.1468% acceptance rate; C=1-4 decode
changes were -0.00%, -0.27%, -0.00%, and +0.15%. The uncontended matched MTP4 rerun likewise
preserved every output hash and the 46.5897% acceptance rate; C=1-4 decode changes were +0.89%,
+1.53%, +1.28%, and +0.36%.

Primary reports are under `profiles/bench/mixed-phase-goal-final-prefill-c*-20260905.json`,
`profiles/bench/mixed-phase-goal-c2-{baseline,adaptive-split-clean}-20260905`,
`profiles/bench/mixed-phase-goal-final-serve-dflash4-decode-20260905`,
`profiles/bench/mixed-phase-goal-final-serve-mtp4-decode-uncontended-20260905`, and
`profiles/ppl/mixed-phase-prefill-split-20260905.json`.

## Qwen3.8-27B NVFP4 XAttention prefill campaign

The 2026-09-05 XAttention follow-up used the production tau-0.9 profile, exact NVFP4 KV,
8,192-token chunks for performance runs, 4,096-token chunks for PPL, CUDA Graph decode, and one
RTX 5090. The selector now remains dense through the configured 8,192-token minimum instead of
ranking only the final query tile at the boundary. That boundary previously paid the rank prepass
while retaining every visible page. The keep-list finalizer also reserves only its live
score/id/mark planes: dynamic shared memory falls
from 53,248 to 36,864 bytes, admitting two CTAs per SM instead of one. Exact 64-, 128-, and
1,024-entry bitonic specializations avoid padding those ranks to the next larger retained case.

On the public T=4,096 GQA Op benchmark, the boundary change reduced the tau-0.9 8k cell from
2,501.9 to 2,309.5 us (-7.69%, +8.33% throughput). Order-balanced finalizer A/Bs improved the
32k, 64k, and 128k cells by about 0.25%, 0.42%, and 0.49%; the 1,024-entry sort supplied another
0.37% at 128k. Nsight measured the 64k finalizer at about 270 us after the shared-memory change,
down from about 439 us. The score GEMM remains the ranker's dominant component: at 64k it was
1.34 ms, 90.3% compute-utilized, and 95.3% L2-hit. Sweeps of four versus eight query blocks per
CTA, 64 versus 128 score columns, and a two-query-head fused CTA were 0.5% to 1.9% slower and were
not retained.

The Engine benchmark now measures each submitted request wave explicitly. For a pure-prefill test,
aggregate throughput is total prompt tokens divided by wave wall time; it also reports the
slowest lane's active prefill rate separately. This avoids multiplying a per-request active rate by
startup concurrency even though the Engine deliberately permits only one prefill owner. Corrected
matched 32k target-only results (no speculative backend) show nearly constant wave throughput from
C=1 through C=4 and a consistent tau-0.9 benefit:

| Startup C | Dense wave tok/s | XAttention wave tok/s | XAttention effect |
|---:|---:|---:|---:|
| 1 | 11,206 | 12,444 | +11.05% |
| 2 | 11,163 | 12,461 | +11.63% |
| 3 | 11,166 | 12,408 | +11.13% |
| 4 | 11,146 | 12,387 | +11.14% |

An order-balanced chunk-size A/B then reran the same 32k workload from the final binary. The table
reports complete-wave throughput and the effect of selecting 4,096 instead of 8,192:

| Attention | C | Chunk 4,096 tok/s | Chunk 8,192 tok/s | 4,096 effect |
|---|---:|---:|---:|---:|
| Dense | 1 | 11,177 | 11,284 | -0.95% |
| Dense | 2 | 11,121 | 11,160 | -0.35% |
| Dense | 3 | 11,077 | 11,145 | -0.61% |
| Dense | 4 | 11,071 | 11,141 | -0.63% |
| XAttention tau 0.9 | 1 | 12,376 | 12,489 | -0.91% |
| XAttention tau 0.9 | 2 | 12,297 | 12,409 | -0.90% |
| XAttention tau 0.9 | 3 | 12,287 | 12,394 | -0.87% |
| XAttention tau 0.9 | 4 | 12,269 | 12,373 | -0.84% |

Chunk 4,096 reduced the fixed workspace from 1,220.6 to 610.3 MiB and reduced the complete startup
reservation by 650 MiB at every C. It is therefore the memory-efficient long-context profile for
less than 1% measured 32k prefill cost, so 4,096 is the product default and 8,192 remains an
explicit maximum-speed override. A future startup-auto refinement may evaluate both fixed layouts
before KV sizing. Runtime switching cannot recover memory from the already reserved address-stable
arena.

The retained selector changes themselves are deliberately smaller than the dense-to-XAttention
effect. At 8k and C=4, an order-balanced five-repetition A/B against `db14d3df` reduced the
slowest lane's active prefill time by 0.18%. At 32k the corresponding active-rate effects were
+0.93% at C=2, +0.38% at C=3, and +0.03% at C=4. These active-time comparisons isolate the kernel
changes; they are not aggregate wave-throughput claims. Workspace capacity and keep counts above
the minimum were unchanged.

The independent XAttention proof passed for both registered GQA geometries. It compares the
production keep list with an independent inverse-reshape/mass oracle, distinguishes the retired
four-antidiagonal heuristic with planted inputs, and checks retained-tile attention directly
against an FP64 softmax oracle. A separate exact CUDA/CPU ordering oracle covers every retained
64- through 4,096-entry rank specialization, including ties, padding, and non-power-of-two live
counts. The new exact-minimum case selects dense as intended. At 8k,
dense-NVFP4 and tau-0.9 per-token NLL sidecars were byte-identical. At 32k, tau 0.9 added 0.000501
mean NLL over dense NVFP4, below the paired 1-sigma noise of 0.002492, with no non-finite values.
The 64k NIAH run returned the exact `ORCHID=493817; COLOR=COBALT` record. The forced-XAttention C=2
mixed Vision/text DFlash test also matched every sequential target token, covering packed request
isolation and decode composition. In a production HTTP C=2 stagger, a 32,856-token newcomer
arriving 0.5 seconds into a 512-token DFlash4 donor completed prefill at 10,911 tok/s while the
donor completed at 206.4 decode tok/s; both requests finished normally.

Primary reports are under `profiles/bench/xattn-goal-*`, including the matched
`xattn-goal-chunk-ab-*` chunk comparison, and
`profiles/ppl/xattn-goal-boundary-{8192,32768}-20260905`.

## Single-request serving performance method

| Setting | Value |
|---|---|
| GPU | NVIDIA GeForce RTX 5090, 32 GiB |
| CUDA compile/runtime | 13.1 / 13.1 |
| CUDA driver API | 13.3 for NVFP4 and refreshed 35B MTP3; 13.1 for the remaining single-request campaigns |
| Request mode | One active request, `stream=false` |
| Maximum context | 262,144 tokens; 131,072 for refreshed NVFP4 MTP3 |
| Prefill chunk | 1,024 tokens |
| KV cache | INT8 group-64 |
| CUDA Graph | Enabled |
| Prefix reuse | Disabled |
| Sampling | Temperature 0.6, top-p 0.95, top-k 20, presence penalty 1.0 |
| Greedy profile | Exact argmax (`--sampling greedy` in the corpus runner) |
| MTP0 | no `--spec` |
| MTP3 | `--spec mtp --draft-tokens 3 --lm-head-draft` |
| DFlash block=8 | `--spec dflash --draft-tokens 7 --lm-head-draft` |

The MTP0 profile uses four Long NIAH prompts with approximately 8K, 64K, 128K, and 256K tokens.
Thinking is disabled and the output budget is 128 tokens. These runs measure prefill throughput,
server-internal time to first token, and baseline decode throughput at each context length. Content
scenarios are not repeated with MTP disabled because they do not change the baseline decode path.

The speculative-decode corpus contains three long-reasoning fixtures with thinking enabled and a
65,536-token output limit, followed by twelve fixtures covering code, story, translation, and
structured output. The cross-scenario fixtures disable thinking and use a 4,096-token output limit.
The tables report actual completion lengths rather than assuming that every request reaches its
limit.

Metrics are computed from the server's unrounded phase timings and speculative-decode counters:

```text
prefill_tok_s = prompt_tokens / prefill_seconds
server_ttft_ms = 1000 * (prepare_seconds + vision_seconds + prefill_seconds)
decode_tok_s = (completion_tokens - 1) / decode_seconds
spec_acceptance = accepted_tokens / drafted_tokens
spec_tokens_per_round = 1 + accepted_tokens / speculative_rounds
```

Decode throughput is a transport/execution measurement, not a correctness score. The response text,
finish reason, and fixture-level structural requirements are audited separately below. A request
that exhausts its output budget or enters a repetition loop remains useful as a sustained-decode
stress sample, but is not presented as a successfully completed task.

## Concurrent MTP3 decode saturation

The concurrent campaign uses the `long_decode_aime26_15` fixture with thinking enabled. The
rendered prompt is 293 tokens, and every request has an 8,192-token output budget. For each
concurrency C, the runner starts a fresh `ninfer-serve` process with `max_concurrency=C`, releases
C non-stream requests together using distinct fixed seeds, and waits for every HTTP response.
Startup and server warmup occur before the measured wave.

All points use an RTX 5090, CUDA 13.1 compile/runtime, CUDA driver API 13.3, stochastic sampling
(temperature 0.6, top-p 0.95, top-k 20, presence penalty 1.0), INT8 group-64 KV, a 1,024-token
prefill chunk, CUDA Graphs, prefix reuse disabled, and
`--spec mtp --draft-tokens 3 --lm-head-draft`. Each request has a 16,384-token context ceiling.
`--kv-capacity auto` resolved to exactly `C * 16,384` tokens at every point.

Saturated throughput uses only complete one-second server intervals satisfying all of the following:

- computed prefill tokens are zero;
- `running=C`, `prefilling=0`, and `decode_ready=C`;
- at least one decode round completed;
- every decode round had exactly C rows.

Ramp-up, prefill, and drain intervals are excluded. The reported aggregate rate is:

```text
steady_decode_tok_s = sum(committed_decode_tokens) / sum(interval_seconds)
```

Wave makespan starts when the client threads are released and ends after the last complete HTTP
response. MTP acceptance is aggregated over the complete wave. Each row below is one sustained
wave rather than a repeated-sample mean.

| Model profile | C | Steady (s) | Avg batch | Aggregate decode tok/s | MTP acceptance | Speedup vs. C1 | Wave makespan (s) |
|---|---:|---:|---:|---:|---:|---:|---:|
| Qwen3.6-27B `groupwise-int` | 1 | 43.01 | 1.00 | 185.8 | 68.2% | 1.00× | 44.23 |
| Qwen3.6-27B `groupwise-int` | 2 | 65.01 | 2.00 | 247.0 | 69.0% | 1.33× | 66.67 |
| Qwen3.6-27B `groupwise-int` | 4 | 102.02 | 4.00 | 309.5 | 68.4% | 1.67× | 107.49 |
| Qwen3.6-27B `groupwise-int` | 8 | 118.02 | 8.00 | 535.0 | 68.3% | 2.88× | 125.20 |
| Qwen3.6-27B `nvfp4` | 1 | 39.01 | 1.00 | 202.4 | 69.3% | 1.00× | 40.46 |
| Qwen3.6-27B `nvfp4` | 2 | 39.01 | 2.00 | 399.7 | 71.4% | 1.97× | 41.82 |
| Qwen3.6-27B `nvfp4` | 4 | 44.01 | 4.00 | 699.7 | 69.3% | 3.46× | 47.92 |
| Qwen3.6-27B `nvfp4` | 8 | 55.01 | 8.00 | 1,146.9 | 68.6% | 5.67× | 58.57 |
| Qwen3.6-35B-A3B `groupwise-int` | 1 | 12.00 | 1.00 | 593.0 | 67.2% | 1.00× | 13.75 |
| Qwen3.6-35B-A3B `groupwise-int` | 2 | 17.00 | 2.00 | 877.7 | 68.2% | 1.48× | 18.87 |
| Qwen3.6-35B-A3B `groupwise-int` | 4 | 26.01 | 4.00 | 1,166.0 | 69.8% | 1.97× | 28.43 |
| Qwen3.6-35B-A3B `groupwise-int` | 8 | 48.01 | 8.00 | 1,313.8 | 67.3% | 2.22× | 50.20 |

All 45 requests reached their output limit, producing 368,640 completion tokens. The campaign
contained 608 complete full-batch steady intervals and had no request, CUDA, or out-of-memory
failure. At C=8, available device memory after startup was 2.66 GiB for 27B groupwise-int,
2.18 GiB for 27B NVFP4, and 4.38 GiB for 35B-A3B.

## Qwen3.8-27B DFlash k=4 C<=4 retune

RTX 5090, CUDA Graph, NVFP4 KV, optimized proposal head, greedy
`long_decode_aime26_15`, 2,048 output tokens per request, max-context 16,384, and one compact
full-concurrency batch. The retained W=5 route aggregates the A16 attention-output, GDN-output,
MLP-down, fused SwiGLU, and supported fused attention-input projections across requests. Packed
GDN conv-record uses grouped SmallT weight replay for the qualified W=5 C=1..4 shapes instead of
serial T=1 projection passes; C=1 uses one five-token weight panel and a separate sequential FP32
convolution.
Attention-input aggregation is limited to the NVFP4 and BF16-control routes used by this
artifact; Q4/Q5 and other verify widths retain their prior panel execution until separately
qualified.

| C | All panels tok/s | Residual aggregate tok/s | Projection aggregates tok/s | Concurrent GDN tok/s | vs prior | vs panels |
|---:|---:|---:|---:|---:|---:|---:|
| 2 | 161.3 | 173.6 | 187.5 | **219.9** | +17.3% | +36.3% |
| 3 | 163.6 | 177.8 | 193.3 | **229.6** | +18.8% | +40.4% |
| 4 | 164.4 | 173.5 | 186.8 | **220.6** | +18.1% | +34.2% |

The subsequent C=1 W=4..6 specialization processes each verification panel in one weight pass.
W=4 uses fused SmallT and needs no workspace: the complete public GDN record Op fell from 83.648
to 49.152 us (-41.2%) cold and from 63.040 to 28.480 us (-54.8%) warm. W=5/6 use grouped replay
and a separate sequential convolution. W=5 fell from 96.256 to 67.584 us (-29.8%) cold and from
77.568 to 42.784 us (-44.8%) warm; W=6 fell from 112.640 to 69.632 us (-38.2%) cold and from
92.000 to 47.040 us (-48.9%) warm. Across three matched 1,024-token fixtures, DFlash3 steady
decode improved by 11.9-13.3% (+12.6% geomean) and whole-wave throughput by 10.8-11.1% (+11.0%
geomean). Matched MTP3 improved from 157.18 to 173.22 tok/s (+10.2%) steady and 11.3% whole-wave;
the seeded p-less DFlash3 control improved from 150.97 to 177.97 tok/s (+17.9%) steady and 11.2%
whole-wave. DFlash4 steady decode
improved by 8.0-10.1% (+9.2% geomean) and whole-wave throughput by 9.5-9.6% (+9.6% geomean), while
DFlash5 improved by 6.9-13.0% (+10.5% geomean) and 12.3% geomean, respectively. A separate
2,048-token DFlash5 fixture improved from 139.59 to 156.16 tok/s (+11.9%); matched MTP5 improved
from 145.98 to 169.78 tok/s (+16.3%). Each A/B pair retained identical output hashes,
draft/accept counters, and acceptance.

The same shared target route improves matched MTP4 serving without changing acceptance:

| C | All panels tok/s | Residual aggregate tok/s | Projection aggregates tok/s | Concurrent GDN tok/s | vs prior | vs panels |
|---:|---:|---:|---:|---:|---:|---:|
| 2 | 156.7 | 167.9 | 178.4 | **212.1** | +18.9% | +35.3% |
| 3 | 162.3 | 177.2 | 190.9 | **229.5** | +20.2% | +41.4% |
| 4 | 164.8 | 175.2 | 188.1 | **225.3** | +19.8% | +36.7% |

The post-origin C=2..4 runs use the same artifact and prompts. All-panel, residual-only,
projection, and concurrent-GDN runs have identical output hashes,
speculative rounds, drafted-token counts, accepted-token counts, and acceptance. Relative to
five-token panels, the aggregated NVFP4 SwiGLU launches at T=10/15/20 measured
114.7/165.9/213.1 us instead of
174.1/233.5/294.9 us. NVFP4 attention-input measured 59.4/75.8/92.2 us instead of
73.7/102.4/131.1 us; the six BF16-control attention layers measured 108.5/124.9/137.2 us instead
of 202.8/301.1/399.4 us. The BF16 panel-preserving phase order is intentionally global at these
shapes: against the prior standalone packed route, T=10 is 1.9% slower, T=15 is 13.0% slower
(110.6 to 124.9 us), and T=20 is unchanged. That cost is dominated by eliminating two repeated
weight passes in the supported W=5 C=3 production route, which is 58.5% faster than its panels.
At the complete public GDN conv-record Op, W=5 C=2 cold Graph replay fell from 217.088 to
110.592 us (-49.1%); the superseded pair-within-CTA candidate measured 215.072 us and was removed.

Independent mathematical Op oracles cover every new T=10/15/20 route. The residual T=20
schedule retains the T=5 panel's 16-value-per-lane reduction; using the generic eight-value
schedule changed reduction association and failed the long C=4 identity gate. A true all-panels
versus combined C=4 probe was bit-identical across 512,000 sampled post-MLP BF16 values, 102,400
target-hidden BF16 values, 4,966,400 target-logit BF16 values, verifier IDs, cache positions,
argmax, and sampled live and ReplaySSM BF16/FP32 GDN state. Every checkpoint had zero mismatches,
zero relative L2 and maximum absolute error, and no nonfinite values. The GDN change adds direct
intermediate coverage at its public boundary: W=4/5 C=1..4 and W=6 C=1 dense, ragged, and
tree-parent q/k/v/z outputs and valid conv-record values are bit-exact to the former C=1 T=1
route. Independent decoded-NVFP4 FP64 projection/convolution checks had no nonfinite values;
worst relative L2 was
0.00269457 under 0.00315 and worst maximum absolute error was 0.00380876 under its 0.00738378
gross limit. Four 2,048-token Graph
streams also matched exactly, including 681 rounds, 2,724 drafts, 1,366 accepts, and 50.1468%
DFlash acceptance. A matched W=6 p-less stream also retained exact output, counters, and 39.5322%
acceptance while improving steady decode by 12.3%. Separately, target-only decode PPL remained
6.414141594 through the GDN change:
all 2,047 scored FP32 NLL values were byte-identical, with 19 terrible tokens and no nonfinite
values. That PPL route does not exercise concurrent packed verification.

The subsequent GQA retune removes the remaining per-request attention loop. Concurrent NVFP4
SmallT verification now launches one request-indexed partial and one batched reduction per chunk;
the B=1 specialization and each request CTA's arithmetic remain unchanged. At public Graph Op scope
over W=2..5 and contexts 37/128/2,048/4,096, the mean speedups were 1.68x/1.97x/2.40x at
C=2/3/4. Representative W=5 latency fell from 22.528 to 14.016 us at C=2, from 32.768 to
14.336 us at C=3, and from 42.976 to 16.064 us at C=4 for context 37; at context 4,096 the
corresponding changes were 36.864 to 24.576 us, 53.248 to 34.816 us, and 69.632 to 40.960 us.
The existing 64-value reduction chunk remained best or tied at every sampled C=2/C=4, W=2/W=5,
context-128/context-4,096 point; 8/16/32-value alternatives were 11% to 105% slower.

Two matched 2,048-token fixed-wave serving A/Bs produced geometric-mean gains of 0.24% at C=2
and 1.33% at C=4; the uncontended pair improved C=2 from 219.50 to 221.02 aggregate tok/s and
C=4 from 220.86 to 224.54 tok/s. The complete first pass kept C=1 within 0.3% noise and measured
a 0.36% C=3 gain. Every lane retained the same output hash; speculative rounds, drafts, accepts,
and 50.1468% acceptance were exact at each concurrency. Independent mathematical-oracle coverage
includes dense, ragged, fragmented-page, permuted-table, W=2 C=2, and W=5 C=4 NVFP4 batches.
Packed real-artifact C=4 Graph isolation also retained exact C=1 outputs through the qualified
token window. A rebuilt-candidate NVFP4 decode PPL rerun remained 6.414141594 over 2,047 scored
tokens, and its complete FP32 NLL stream was byte-identical to the retained pre-change file.

Fresh post-GQA CUDA Graph node traces then localized the remaining C<=4 projection cost. At C=2,
NVFP4 SmallT accounted for 42.97% of kernel time and fused NVFP4 SwiGLU for 26.29%; at C=4 the
shares were 45.85% and 26.64%. GQA had fallen below 1% in both traces. Exact-shape
`tools.kdev` classification found the W=2..5 Linear, SwiGLU, and GDN points DRAM-bound and refused
tile/warp retuning. The generic Linear and already-fused SwiGLU paths therefore remain unchanged.
The material opportunity was GDN record weight replay: W=2 and W=5 now project two requests per
SmallT weight pass, retain the W-local reduction profile, keep the channel projection in private
FP32 workspace, and perform the convolution in a separate 2.1 us kernel. W=3/4 retain their
request-indexed route because pairing was slower at those widths.

| Width | C | Prior GDN record us | Paired us | Change |
|---:|---:|---:|---:|---:|
| 2 | 2 | 59.392 | 51.200 | -13.8% |
| 2 | 3 | 73.728 | 69.632 | -5.6% |
| 2 | 4 | 89.824 | 69.632 | -22.5% |
| 5 | 2 | 112.640 | 88.064 | -21.8% |
| 5 | 3 | 155.648 | 145.408 | -6.6% |
| 5 | 4 | 200.672 | 151.552 | -24.5% |

The production C=2 trace reduced the full GDN record family from 806.014 to 625.437 ms across
7,440 record invocations (-22.4%); the paired projection used 230 registers/thread with no local
memory or spills. Matched 1,024-token-per-lane fixed waves show the end-to-end effect:

| Backend | C | Prior tok/s | Paired tok/s | Throughput | Makespan |
|---|---:|---:|---:|---:|---:|
| DFlash k=4 | 2 | 248.820 | 256.241 | +2.98% | -3.06% |
| DFlash k=4 | 4 | 251.962 | 261.382 | +3.74% | -3.62% |
| MTP4 | 2 | 238.822 | 245.388 | +2.75% | -3.26% |
| MTP4 | 4 | 255.958 | 265.386 | +3.68% | -3.95% |

DFlash acceptance remained exactly 59.2745% with 608/1,216 row-rounds at C=2/4; MTP acceptance
remained exactly 54.9219% with 640/1,280 row-rounds. Independent decoded-NVFP4 FP64 oracles cover
W=2 and W=5 at C=2/3/4, including dense, ragged, and tree-parent records; outputs are bit-exact to
the serial C=1 arithmetic. Real packed-artifact C=4 Graph isolation also matched C=1 through the
qualified window. The NVFP4 decode PPL rerun remained 6.414141594 over 2,047 scored tokens with
no nonfinite values, and every FP32 NLL was byte-identical to the post-GQA baseline.

A proposed fusion of the DFlash2 proposer MLP gate-up Linear and SiLU stages saved up to 2.0 us
in isolation but reduced production throughput in its matched pre-rebase C=2/3/4 campaign, so it
was removed.

A fresh C=1 node trace after the GDN projection work attributed 28.8% of GPU time to target
SwiGLU, 17.0% to GDN input projection, 14.9% to MLP-down, and 4.1% to the separate GDN replay
record and T=1 overlay kernels. Exact-shape `tools.kdev` classification refused further
tile/occupancy changes for the dominant DRAM-bound projections. Staging SwiGLU scales regressed
W=4/5/6 public-Op latency, and a 16-warp/Bc64 NVFP4 GQA tile regressed the W=5, 32,768-context
public Op from 45.056 to 57.344 us, so both were removed. The retained change publishes exact
replay key/value/gate records in the existing numerically qualified T=1 overlay kernel. The
production W=5 record-plus-overlay median fell from 14.24 to 8.54 us per GDN layer in the node
trace.

Matched C=1, 512-token, three-repetition runs with NVFP4 KV and unchanged speculative counters:

| Backend | Before tok/s | Fused replay tok/s | Change |
|---|---:|---:|---:|
| DFlash3 | 109.493 | 111.211 | +1.57% |
| DFlash4 | 110.948 | 112.848 | +1.71% |
| DFlash5 | 89.178 | 90.606 | +1.60% |
| MTP3 | 146.667 | 149.225 | +1.74% |
| MTP4 | 117.225 | 119.214 | +1.70% |
| MTP5 | 116.227 | 117.927 | +1.46% |

The public replay Op directly checks C=1 W=4/5/6 fused records bitwise against the no-workspace
record route, BF16 output against repeated T=1 snapshots, unchanged FP32 source state, and masked
B=3/4 isolation. DFlash Graph and eager real-artifact checks pass. MTP k=3/5 real-artifact checks
pass; the existing k=4/B=4 coverage assertion still fails because its short requests complete
without producing a four-row round. Target-only decode PPL remained 6.414141594, with every one
of 2,047 FP32 NLL values byte-identical and no nonfinite values.

### W=4/6 multi-request weight replay and remaining decode research

The 2026-09-06 follow-up used the same RTX 5090, Qwen3.8-27B NVFP4/BF16-codebook artifact,
NVFP4 KV, CUDA Graphs, 4,096-token prefill chunks, optimized proposal head, and greedy fixed
waves. W=4 now groups NVFP4 attention-input, fused SwiGLU, and MLP-down work across C=2..4;
W=6 groups MLP-down across C=2..4, fused SwiGLU at C=2/3, and NVFP4 attention-input at C=2.
The other W=4/6 attention-input and attention/GDN-output shapes remain panels: direct aggregate
kernels passed random mathematical-oracle tolerances but changed real recurrent verification
outputs. The W=6 C=3 MLP-down schedule retains the W=6 panel's 16-value reduction association.

Against the preserved `415b1ef8` binary, matched 256-token DFlash waves measured:

| Backend | C | Prior tok/s | Retained tok/s | Change |
|---|---:|---:|---:|---:|
| DFlash3 | 2 | 135.950 | 151.524 | +11.46% |
| DFlash3 | 3 | 148.171 | 169.750 | +14.56% |
| DFlash3 | 4 | 151.617 | 178.667 | +17.84% |
| DFlash5 | 2 | 117.164 | 122.894 | +4.89% |
| DFlash5 | 3 | 121.980 | 130.084 | +6.64% |
| DFlash5 | 4 | 122.483 | 128.309 | +4.76% |

Matched 128-token MTP waves independently confirmed the shared target-route gain:

| Backend | C | Prior tok/s | Retained tok/s | Change |
|---|---:|---:|---:|---:|
| MTP3 | 2 | 168.553 | 190.362 | +12.94% |
| MTP3 | 3 | 172.886 | 204.114 | +18.06% |
| MTP3 | 4 | 175.968 | 214.136 | +21.69% |
| MTP5 | 2 | 135.592 | 144.728 | +6.74% |
| MTP5 | 3 | 145.510 | 156.704 | +7.69% |
| MTP5 | 4 | 143.259 | 150.630 | +5.14% |

Every paired cell retained identical speculative rounds, drafted tokens, and accepted tokens.
Exact packed-versus-panel intermediate tests cover W=4/5/6 NVFP4 attention input, SwiGLU, and
both residual Linear geometries; Graph and eager DFlash real-artifact isolation pass for k=3/5,
and MTP Graph isolation passes through its existing short adaptive-k4 coverage assertion.

Three other research directions produced no retained code. At W=5 and contexts
32,768/65,536/131,072/196,608, the existing NVFP4-KV GQA path measured
45.024/67.552/118.784/165.888 us. Reducing it from eight to four warps regressed those points by
12-18%; the kernel already double-stages asynchronous KV loads, and NCU measured about 167.5 MB
of L2 traffic with 66-68% DRAM throughput at 131,072 tokens. Persistent SwiGLU-plus-down fusion
was rejected by the exact-shape classifier: its grid-wide dependency would remove only about
170-220 KiB of activation traffic while streaming about 151 MiB of weights per W=5 layer. The
fixed-width scheduling sweep selected W=5 for this DFlash corpus at C=1..3 (approximately tied
with W=4 at C=4) and W=4 for MTP at every C. Adaptive N=5 selected W=4; its measured E[Y]/round
time objective already includes weight streaming and all non-matrix round costs, so a separate
accepted-tokens-per-weight-byte objective was not retained. Backend-specific cold seeding did not
change the selected steady width and was removed.

The pre-rebase C=4 CUDA Graph node trace used to select the projection work attributed 32.5% of
kernel time to SwiGLU, 20.7% to GDN record, 17.6% to MLP down, and about 1% to DFlash top-k
selection. A fresh post-origin C=2 trace then exposed GDN request serialization as 36.1% of the
incremental round cost, motivating the concurrent route.

Reports: `profiles/bench/multi-request-kernel-post-origin-panel-control-20260904/`,
`profiles/bench/multi-request-kernel-post-origin-mtp4-panel-control-20260904/`,
`profiles/bench/multi-request-kernel-post-origin-residual-only-fixed-20260904/`, and
`profiles/bench/multi-request-kernel-post-origin-final-20260904/`. Concurrent-GDN results are in
`profiles/bench/multi-request-kernel-gdn-{final,mtp4-final}-20260904/`; its Op A/B is in
`profiles/bench/gdn-c2-retune-20260904/`. Accuracy report:
`profiles/ppl/multi-request-kernel-gdn-final-20260904.{json,nllf32}`. The GQA Op and serving A/Bs
are in `profiles/bench/multi-request-gqa-{baseline,candidate}-20260904/` and
`profiles/bench/multi-request-gqa-e2e-{baseline,candidate}-20260904/`; the uncontended repeat is
in `profiles/bench/multi-request-gqa-e2e-{baseline,candidate}-clean-20260904/`. The PPL rerun is
in `profiles/ppl/multi-request-gqa-candidate-20260904/`.
The C=1 W=5 Op A/B is in
`profiles/bench/c1-dflash4-gdn-b1-w5-{route-proxy-head95588f45,grouped-final}-20260905.csv`;
its serving A/B is in `profiles/bench/c1-gdn-w5-e2e-{before-95588f45,after}-20260905/`, and the
three-fixture confirmation is in
`profiles/bench/c1-gdn-w5-general1024-{before-95588f45,after}-20260905/`. The
retained NCU reports are in `profiles/ncu/c1-dflash4-gdn-record-w5-head95588f45-20260905.*` and
`profiles/ncu/c1-dflash4-gdn-record-w5-grouped-candidate-20260905.*`. The W=6 Op A/B is in
`profiles/bench/c1-dflash5-gdn-b1-w6-baseline-20260906.csv` and
`profiles/bench/c1-dflash45-gdn-b1-w5-w6-grouped-candidate-20260906.csv`; its NCU report is
`profiles/ncu/c1-dflash5-gdn-record-w6-grouped-candidate-20260905.*`. DFlash5 serving is in
`profiles/bench/c1-gdn-w6-dflash5-{general1024,aime15-2048}-{before-95588f45,after}-20260906/`,
and MTP5 is in `profiles/bench/c1-gdn-w6-mtp5-aime15-1024-{before-95588f45,after}-20260906/`.
The p-less A/B and adaptive-k5 smoke are in
`profiles/bench/c1-gdn-w6-dflash5-pless-512-{before-95588f45,after}-20260906/` and
`profiles/bench/c1-gdn-w6-dflash5-adaptive-k5-smoke-20260906/`.
The final target-only PPL rerun is `profiles/ppl/c1-gdn-w56-after-20260906.{json,nllf32}`.
The W=4 serialized, grouped-control, and retained fused-SmallT Op measurements are in
`profiles/bench/c1-draft3-gdn-b1-w4-baseline-20260906.csv`,
`profiles/bench/c1-draft345-gdn-b1-w4-w6-grouped-candidate-20260906.csv`, and
`profiles/bench/c1-draft345-gdn-b1-w4-fused-w5-w6-grouped-final-20260906.csv`; the retained NCU
report is `profiles/ncu/c1-draft3-gdn-record-w4-fused-smallt-final-20260906.*`. DFlash3 serving is
in `profiles/bench/c1-gdn-w4-dflash3-general1024-{before-95588f45,after}-20260906/`; matched MTP3
and p-less DFlash3 are in `profiles/bench/c1-gdn-w4-mtp3-aime15-1024-{before-95588f45,after}-20260906/`
and `profiles/bench/c1-gdn-w4-dflash3-pless-512-{before-95588f45,after}-20260906/`. Adaptive-k3
smokes are in `profiles/bench/c1-gdn-w4-dflash-adaptive-k3-smoke-{before-95588f45,after}-20260906/`.
The final PPL rerun is `profiles/ppl/c1-gdn-w456-after-20260906.{json,nllf32}`.
The fused replay A/B is in
`profiles/bench/c1-replay-fusion-{before,after}-{dflash3,dflash4,dflash5,mtp3,mtp4,mtp5}-tg512-20260906.json`;
its short/long node traces are in `profiles/nsys/c1-opportunity-*-20260906.*`, and the PPL result is
`profiles/ppl/c1-replay-fusion-after-20260906.{json,nllf32}`.
The post-GQA profiles are in `profiles/nsys/smallt-post-gqa-c{2,4}-nodes.nsys-rep`; isolated GDN
A/Bs are in `profiles/bench/smallt-goal-{baseline-isolated,candidate-paired}/`; the long serving
A/Bs are in `profiles/bench/smallt-goal-long-{baseline,paired}-20260905/`; and the PPL result is
`profiles/ppl/smallt-paired-20260905.{json,nllf32}`.

## Reproduction

Build `ninfer-serve` and prepare the registered `.ninfer` artifacts. The refreshed per-target
serving tables use:

```bash
python3 tools/bench/run_serve_concurrency.py \
  --serve build/apps/ninfer-serve \
  --artifact qwen3_6_35b_a3b=out/qwen3_6_35b_a3b.ninfer \
  --mode mtp3 --suite corpus-makespan --concurrency 1 \
  --max-context 262144 --kv-capacity auto \
  --output profiles/bench/concurrent_corpus_35b_mtp3_20260811

python3 tools/bench/run_serve_corpus.py \
  --serve build/apps/ninfer-serve \
  --artifact qwen3_6_27b=out/qwen3_6_27b.ninfer \
  --mode mtp3 \
  --output profiles/bench/serve_corpus_27b_mtp3_20260724

python3 tools/bench/run_serve_corpus.py \
  --serve build/apps/ninfer-serve \
  --artifact qwen3_6_27b=out/qwen3_6_27b_nvfp4.ninfer \
  --mode mtp0 --sampling stochastic \
  --output profiles/bench/serve_corpus_27b_nvfp4_w8_20260731

python3 tools/bench/run_serve_concurrency.py \
  --serve build/apps/ninfer-serve \
  --artifact qwen3_6_27b=out/qwen3_6_27b_nvfp4.ninfer \
  --mode mtp3 --suite corpus-makespan --concurrency 1 \
  --max-context 131072 --kv-capacity auto \
  --output profiles/bench/concurrent_corpus_27b_nvfp4_mtp3_20260811
```

The concurrent decode-saturation campaigns use:

```bash
python3 tools/bench/run_serve_concurrency.py \
  --serve build/apps/ninfer-serve \
  --artifact qwen3_6_27b=out/qwen3_6_27b.ninfer \
  --mode mtp3 --suite decode-saturation \
  --concurrency 1 --concurrency 2 --concurrency 3 --concurrency 4 \
  --decode-tokens 8192 --max-context 16384 --kv-capacity auto \
  --output profiles/bench/concurrent_decode_27b_mtp3_20260811

python3 tools/bench/run_serve_concurrency.py \
  --serve build/apps/ninfer-serve \
  --artifact qwen3_6_27b=out/qwen3_6_27b_nvfp4.ninfer \
  --mode mtp3 --suite decode-saturation \
  --concurrency 1 --concurrency 2 --concurrency 3 --concurrency 4 \
  --decode-tokens 8192 --max-context 16384 --kv-capacity auto \
  --output profiles/bench/concurrent_decode_27b_nvfp4_mtp3_20260811

python3 tools/bench/run_serve_concurrency.py \
  --serve build/apps/ninfer-serve \
  --artifact qwen3_6_35b_a3b=out/qwen3_6_35b_a3b.ninfer \
  --mode mtp3 --suite decode-saturation \
  --concurrency 1 --concurrency 2 --concurrency 3 --concurrency 4 \
  --decode-tokens 8192 --max-context 16384 --kv-capacity auto \
  --output profiles/bench/concurrent_decode_35b_mtp3_20260811
```

Use `--mode dflash7` for the corresponding DFlash block=8 campaign; add `--sampling greedy` for
the exact-argmax profile. Qwen3.8-27B NVFP4 DFlash2 uses the same flag on a reconverted artifact.
INT8-KV C=1 (the table below) and NVFP4-KV C=1/2/3 (the fused-GDN campaign after it):

```bash
python3 tools/bench/run_serve_concurrency.py \
  --serve build/apps/ninfer-serve \
  --artifact qwen3_8_27b=out/qwen3_8_27b_nvfp4_dflash_w8.ninfer \
  --mode dflash7 --mode mtp3 --mode mtp5 --mode mtp0 \
  --sampling stochastic \
  --temperature 0.6 --top-p 0.95 --top-k 20 --min-p 0 --presence-penalty 0 \
  --concurrency 1 --suite decode-saturation \
  --saturation-fixture long_decode_aime26_15 \
  --decode-tokens 4096 --max-context 16384 --kv-capacity 16384 \
  --output profiles/bench/qwen38_dflash2_c1_aime

python3 tools/bench/run_serve_concurrency.py \
  --serve build/apps/ninfer-serve \
  --artifact qwen3_8_27b=out/qwen3_8_27b_nvfp4_dflash_nvfp4.ninfer \
  --mode dflash4 --mode dflash7 \
  --sampling stochastic \
  --temperature 0.6 --top-p 0.95 --top-k 20 --min-p 0 --presence-penalty 0 \
  --concurrency 1 --concurrency 2 --concurrency 3 \
  --suite decode-saturation \
  --saturation-fixture long_decode_aime26_15 \
  --decode-tokens 4096 --max-context 16384 --kv-capacity auto \
  --kv-dtype nvfp4 --prefill-chunk 4096 \
  --output profiles/bench/qwen38_dflash2_fused_batch_aime_20260829
```

## Qwen3.8-27B NVFP4 C=1 decode

Current fixed-work Engine measurements use the production DFlash artifact, NVFP4 KV, CUDA Graphs,
4,096-token prefill chunks, the 65,536-token benchmark corpus, 512 generated tokens, and one active
request. DFlash results are five-run matched measurements; MTP results are the three-run
post-`a39c5c25` shape baseline.

| Mode | Draft k / verify W | Decode tok/s | Latest incremental gain |
|---|---:|---:|---:|
| DFlash | 3 / 4 | **112.52** | — |
| DFlash | 4 / 5 | **115.45** | **+0.42%** |
| DFlash | 5 / 6 | **92.17** | **+0.41%** |
| MTP | 3 / 4 | 150.83 | — |
| MTP | 4 / 5 | 117.46 | — |
| MTP | 5 / 6 | 116.35 | — |

The latest gain specializes the exact W=5/6 BF16 GDN control projection, reducing its register
count from 54 to 40 and its production latency by 22.0%/25.6%. Two opposite-order ten-run pairs,
which bracketed clock drift, independently averaged +0.29%/+0.44% at DFlash k=4/5. All matched
runs preserved exact speculative work. The preceding single-request GDN record change improved
k=4/5 by 1.18%/1.09%; its C=2/3/4 changes were noise-level. The selector rewrite is also not a
C=1 tradeoff: matched DFlash k=4 aggregate throughput rises
156.86 to 157.89 tok/s at C=2 (+0.65%) and 178.49 to 180.11 tok/s at C=4 (+0.91%). Rounds,
drafts, accepts, fallback counts, and accepted-token counts at every draft position are identical
for every matched pair.

Same GPU, INT8 KV, graphs on, `--lm-head-draft`, seed `7632647173703958409`. The DFlash2 W8
companion is appended on `out/qwen3_8_27b_nvfp4_dflash_w8.ninfer`; MTP points load that same file
with DFlash host-placed. `long_decode_aime26_15` uses 4096 output tokens and max-context 16384.
Story is `scenario_story_en_mystery` at 1024 output tokens.

| Mode | Workload | Decode tok/s | Accept | Tokens/round |
|---|---|---:|---:|---:|
| MTP0 | AIME, stochastic | 86.0 | — | — |
| MTP3 | AIME, stochastic | 184.7 | 51.9% | 2.56 |
| MTP5 | AIME, stochastic | 195.9 | 41.1% | 3.06 |
| DFlash2 k=7 W8 | AIME, stochastic | 126.7 | 27.5% | 2.92 |
| MTP3 | AIME, greedy | 184.9 | 52.0% | 2.56 |
| DFlash2 k=7 W8 | AIME, greedy | 132.0 | 29.1% | 3.04 |
| MTP3 | story, stochastic | 163.5 | 41.9% | 2.26 |
| DFlash2 k=7 W8 | story, stochastic | 115.7 | 21.6% | 2.51 |

DFlash2 is a supported exclusive backend on this identity (`--spec dflash --draft-tokens 4
--lm-head-draft`; verify is chain `W=k+1`, maximum k=5). Historical C=1 INT8-KV points below used
k=7 chain W=8. They beat MTP0 (1.47× on stochastic AIME) and trail MTP3 (0.69×) and MTP5 (0.65×).
Greedy AIME and story show the same gap: DFlash2 accept is about 22–29% versus MTP3 about 42–52%.
That accept gap is the current speed target; it is not a reason to drop the backend.

Isolated CLI NVFP4-KV AIME (`long_decode_aime26_15`, 4096 tokens, seed `7632647173703958409`,
presence penalty 0, `--lm-head-draft`) after fused batched GDN conv-record (2026-08-29):

| Mode | Decode tok/s | Accept | Tokens/round | Rounds |
|---|---:|---:|---:|---:|
| chain k=4 W=5 | 162.18 | 46.57% | 2.86 | 1431 |
| chain k=7 W=8 | 142.52 | 31.61% | 3.21 | 1275 |

k=4 remains the RTX 5090 speed recommendation: cheaper W=5 verify, not more tokens per round.
An earlier same-day W4A4 packed-verify CLI snapshot was faster at C=1 (k=4 **194.48** /
k=7 **166.82**) with higher k=4 accept (50.26%). That snapshot is retained in
[dflash2-tree-speed.md](maintainer/dflash2-tree-speed.md); it predates the C>1 fused GDN path.

Serve C=1/2/3 on the same AIME fixture, NVFP4 KV, graphs, `--lm-head-draft`, presence penalty 0.
Headline tok/s is aggregate `decode_tokens / wave_makespan` (GPU throughput). Per-request
`(completion-1)/decode_seconds` is the isolation metric. All 12 serve requests hit the 4096 output
limit. Logs: `profiles/bench/qwen38_dflash2_fused_batch_aime_20260829/`.

| Mode | C | Aggregate tok/s | Per-request tok/s | Accept | Tokens/round | Makespan (s) | vs C=1 |
|---|---:|---:|---:|---:|---:|---:|---:|
| chain k=4 W=5 | 1 | 162.1 | 162.6 | 46.6% | 2.86 | 25.26 | 1.00× |
| chain k=4 W=5 | 2 | 262.0 | 133.0 | 45.5% | 2.82 | 31.26 | 1.62× |
| chain k=4 W=5 | 3 | 324.4 | 112.1 | 47.3% | 2.89 | 37.87 | 2.00× |
| chain k=7 W=8 | 1 | 143.0 | 143.4 | 31.6% | 3.21 | 28.63 | 1.00× |
| chain k=7 W=8 | 2 | 230.8 | 121.3 | 29.7% | 3.08 | 35.49 | 1.61× |
| chain k=7 W=8 | 3 | 302.0 | 103.1 | 29.6% | 3.07 | 40.68 | 2.11× |

C=3 k=4 is **324 aggregate tok/s** on this 27B NVFP4 DFlash2 path. Per-request rate falls as
the GPU is shared; isolation still matches C=1 DFlash. k=4 wins both C=1 and C=3 aggregate
on this fixture.

Omit `--mode` and supply the two measured Qwen3.6 groupwise-int artifacts to run the complete
published Qwen3.6 MTP0/MTP3 campaign:

```bash
python3 tools/bench/run_serve_corpus.py \
  --serve build/apps/ninfer-serve \
  --artifact qwen3_6_35b_a3b=out/qwen3_6_35b_a3b.ninfer \
  --artifact qwen3_6_27b=out/qwen3_6_27b.ninfer \
  --output profiles/bench/serve_corpus_20260720
```

For the 27B NVFP4 accuracy run, start the model service with:

```bash
build/apps/ninfer-serve out/qwen3_6_27b_nvfp4.ninfer \
  --host 127.0.0.1 --port 18080 \
  --max-context 262144 --prefill-chunk 1024 --kv-dtype int8 \
  --spec mtp --draft-tokens 3 --lm-head-draft
```

Then run the repository's full 27B reasoning suite in a separate shell:

```bash
PYTHONPATH=eval eval/.venv/bin/python -m ninfer_eval run \
  --config eval/configs/qwen3_6_27b_reasoning.yaml \
  --suite reasoning_full
```

## `qwen3_8_27b`

### EvalScope reasoning accuracy

The measured file is
[`Ostfralla/Qwen3.8-27B-NVFP4-NInfer`](https://huggingface.co/Ostfralla/Qwen3.8-27B-NVFP4-NInfer)
(`qwen3_8_27b_nvfp4.ninfer`, SHA-256
`eaf8ad124256d0a0c1ebbbca442ca58eee4f97ab34a60a0b4d57e2b41e2c56d2`). That `nvfp4` weights identity
was evaluated twice through NInfer's OpenAI-compatible serving route with thinking enabled, MTP=3,
and a 262,144-token context limit. The two runs differ only in `--kv-dtype`. EvalScope 1.9.0 used
0-shot prompts, rule-based scoring, and one sample per problem with temperature 0.6, top-p 0.95,
top-k 20, presence penalty 1.0, and seed 42. All 258 samples completed and were scored for each KV
codec.

| KV | AIME 2025 | AIME 2026 | GPQA-Diamond |
|---|---:|---:|---:|
| `int8` | 100.00% (30 / 30) | 96.67% (29 / 30) | 89.90% (178 / 198) |
| `nvfp4` | 93.33% (28 / 30) | 100.00% (30 / 30) | 92.42% (183 / 198) |

These are single-sample results under the stated evaluation profile, not pass@k scores. Each
benchmark remains independently reportable; no combined score is computed. Qwen3.8-27B
`groupwise-int` was not part of this campaign.

Download the measured artifact, start the model service with the matching `--kv-dtype`, and then
run the 3.8 reasoning suite:

```bash
hf download Ostfralla/Qwen3.8-27B-NVFP4-NInfer \
  qwen3_8_27b_nvfp4.ninfer \
  --local-dir models
```

```bash
build/apps/ninfer-serve models/qwen3_8_27b_nvfp4.ninfer \
  --host 127.0.0.1 --port 18080 \
  --max-context 262144 --prefill-chunk 1024 --kv-dtype int8 \
  --spec mtp --draft-tokens 3 --lm-head-draft
```

```bash
build/apps/ninfer-serve models/qwen3_8_27b_nvfp4.ninfer \
  --host 127.0.0.1 --port 18080 \
  --max-context 262144 --prefill-chunk 1024 --kv-dtype nvfp4 \
  --spec mtp --draft-tokens 3 --lm-head-draft
```

```bash
PYTHONPATH=eval eval/.venv/bin/python -m ninfer_eval run \
  --config eval/configs/qwen3_8_27b_reasoning.yaml \
  --suite reasoning_full
```

## `qwen3_6_35b_a3b`

### MTP0 context-length profile

| Prompt tokens | Samples | Prefill tok/s | Server TTFT (ms) | Decode tok/s |
|---:|---:|---:|---:|---:|
| 7,680 | 5 | 15,544.3 ± 242.4 | 500.2 ± 7.8 | 271.1 ± 3.6 |
| 64,512 | 5 | 10,809.0 ± 95.3 | 6,009.9 ± 52.6 | 242.9 ± 1.3 |
| 130,048 | 5 | 7,828.4 ± 34.1 | 16,693.3 ± 71.2 | 219.4 ± 1.6 |
| 260,096 | 5 | 5,157.1 ± 52.4 | 50,598.8 ± 519.7 | 188.2 ± 2.1 |

### MTP3 long-reasoning decode

| Fixture | Samples | Completion tokens | Decode tok/s | MTP acceptance | MTP tokens/round |
|---|---:|---:|---:|---:|---:|
| `long_decode_aime26_01` | 5 | 8,223.0 ± 2,224.1 | 726.2 ± 22.9 | 82.8% ± 3.4% | 3.48 ± 0.10 |
| `long_decode_aime26_15` | 5 | 65,536.0 ± 0.0 | 620.3 ± 8.1 | 72.7% ± 1.4% | 3.18 ± 0.04 |
| `long_decode_aime26_30` | 5 | 52,977.8 ± 11,849.6 | 671.9 ± 8.8 | 80.1% ± 2.7% | 3.40 ± 0.08 |

### MTP3 cross-scenario decode

Each category contains three fixtures and five seeds per fixture, for 15 samples.

| Category | Samples | Decode tok/s | MTP acceptance | MTP tokens/round |
|---|---:|---:|---:|---:|
| Code | 15 | 657.6 ± 34.3 | 70.3% ± 5.5% | 3.11 ± 0.16 |
| Story | 15 | 456.2 ± 36.6 | 38.0% ± 6.0% | 2.14 ± 0.18 |
| Translation | 15 | 649.7 ± 33.0 | 67.6% ± 5.1% | 3.03 ± 0.15 |
| Structured | 15 | 770.9 ± 29.3 | 89.1% ± 4.9% | 3.67 ± 0.15 |

### DFlash block=8 (`k=7`), stochastic sampling

The fixtures, five seeds, sampling parameters, and output limits are identical to MTP3. Different
speculative backends consume random values differently, so this is a fixed-workload comparison
rather than a token-identical paired-output comparison.

#### Long-reasoning decode

| Fixture | Samples | Completion tokens | Decode tok/s | DFlash acceptance | DFlash tokens/round |
|---|---:|---:|---:|---:|---:|
| `long_decode_aime26_01` | 5 | 8,495.4 ± 2,221.2 | 764.1 ± 55.6 | 65.2% ± 5.4% | 5.56 ± 0.38 |
| `long_decode_aime26_15` | 5 | 65,536.0 ± 0.0 | 584.0 ± 33.3 | 51.1% ± 3.7% | 4.58 ± 0.26 |
| `long_decode_aime26_30` | 5 | 53,330.4 ± 11,198.5 | 638.3 ± 15.8 | 56.4% ± 2.5% | 4.95 ± 0.17 |

#### Cross-scenario decode

| Category | Samples | Decode tok/s | DFlash acceptance | DFlash tokens/round |
|---|---:|---:|---:|---:|
| Code | 15 | 562.3 ± 36.2 | 43.0% ± 3.7% | 4.01 ± 0.26 |
| Story | 15 | 261.7 ± 51.1 | 12.1% ± 5.3% | 1.85 ± 0.37 |
| Translation | 15 | 490.8 ± 62.6 | 34.8% ± 6.3% | 3.44 ± 0.44 |
| Structured | 15 | 786.4 ± 124.7 | 66.5% ± 13.5% | 5.66 ± 0.94 |

#### Decode throughput versus MTP3

| Workload | MTP3 tok/s | DFlash tok/s | DFlash change |
|---|---:|---:|---:|
| `long_decode_aime26_01` | 726.2 | 764.1 | +5.2% |
| `long_decode_aime26_15` | 620.3 | 584.0 | -5.9% |
| `long_decode_aime26_30` | 671.9 | 638.3 | -5.0% |
| Code | 657.6 | 562.3 | -14.5% |
| Story | 456.2 | 261.7 | -42.6% |
| Translation | 649.7 | 490.8 | -24.5% |
| Structured | 770.9 | 786.4 | +2.0% |

### DFlash block=8 (`k=7`), greedy sampling

Greedy uses exact argmax; all other corpus and server settings remain unchanged. The five seeds
repeat the same deterministic generation path, so within-fixture standard deviation measures
runtime variation rather than output variation.

#### Long-reasoning decode

| Fixture | Samples | Completion tokens | Decode tok/s | DFlash acceptance | DFlash tokens/round |
|---|---:|---:|---:|---:|---:|
| `long_decode_aime26_01` | 5 | 6,692.0 ± 0.0 | 872.4 ± 3.3 | 74.4% ± 0.0% | 6.21 ± 0.00 |
| `long_decode_aime26_15` | 5 | 65,536.0 ± 0.0 | 651.6 ± 0.6 | 58.6% ± 0.0% | 5.10 ± 0.00 |
| `long_decode_aime26_30` | 5 | 65,536.0 ± 0.0 | 994.9 ± 3.4 † | 98.0% ± 0.0% | 7.86 ± 0.00 |

† The generation is a deterministic repetition loop, not a valid AIME response. The raw rate is
retained to describe what was measured, but is excluded from performance comparisons.

#### Cross-scenario decode

| Category | Samples | Decode tok/s | DFlash acceptance | DFlash tokens/round |
|---|---:|---:|---:|---:|
| Code | 15 | 599.8 ± 12.3 | 46.4% ± 1.4% | 4.25 ± 0.10 |
| Story | 15 | 291.5 ± 55.6 | 14.9% ± 5.7% | 2.04 ± 0.40 |
| Translation | 15 | 475.5 ± 50.6 | 33.0% ± 5.1% | 3.31 ± 0.36 |
| Structured | 15 | 869.0 ± 120.2 | 74.5% ± 13.1% | 6.21 ± 0.92 |

#### Decode throughput versus stochastic DFlash

| Workload | Stochastic tok/s | Greedy tok/s | Greedy change |
|---|---:|---:|---:|
| `long_decode_aime26_01` | 764.1 | 872.4 | +14.2% |
| `long_decode_aime26_15` | 584.0 | 651.6 | +11.6% |
| `long_decode_aime26_30` | 638.3 | 994.9 † | not comparable † |
| Code | 562.3 | 599.8 | +6.7% |
| Story | 261.7 | 291.5 | +11.4% |
| Translation | 490.8 | 475.5 | -3.1% |
| Structured | 786.4 | 869.0 | +10.5% |

### Speculative-decode output audit

The audit covers all 225 stored July responses from the 35B-A3B MTP3 stochastic-sampler, DFlash
stochastic-sampler, and DFlash greedy campaigns. It checks termination, exact repetition, and
fixture-specific mechanical constraints. AIME 1 was checked algebraically; the AIME 30 answer
(`393`) was checked by independent enumeration. This audit does not attempt to assign a subjective
quality score to prose or translations.

#### Long-reasoning answers

| Fixture | MTP3 stochastic sampler | DFlash stochastic sampler | DFlash greedy |
|---|---|---|---|
| `long_decode_aime26_01` | 5/5 correct, natural stop | 5/5 correct, natural stop | 5/5 correct, natural stop |
| `long_decode_aime26_15` | 0/5 answers; all reach 65,536-token limit | 0/5 answers; all reach 65,536-token limit | 0/5 answers; all reach 65,536-token limit |
| `long_decode_aime26_30` | 3/5 correct, 1 wrong, 1 no answer | 2/5 correct, 1 wrong, 2 no answer | 0/5 answers; all enter the same repetition loop |

The greedy AIME 30 response has an empty final-content field and fills its 65,536-token reasoning
budget. The exact line `Wait, $x_7 x_1 x_3$ is $x_7 x_1 x_3$.` occurs 2,406 times among 2,538
non-empty reasoning lines. Its 98.0% acceptance and 994.9 tok/s therefore characterize a highly
predictable pathological loop, not normal reasoning performance.

AIME 15 is also not a valid completion in any of the three campaigns: every sample exhausts the
budget without a boxed answer. Its output is long, non-convergent reasoning rather than the short
exact cycle seen in greedy AIME 30. The AIME 15 rates may be read only as sustained long-decode
throughput.

#### Cross-scenario outputs

| Category | MTP3 stochastic sampler | DFlash stochastic sampler | DFlash greedy |
|---|---|---|---|
| Code | 1/15 natural stops; 0/15 prompt-complete | 2/15 natural stops; 0/15 prompt-complete | 0/15 natural stops |
| Story | 9/15 natural stops; the nine Chinese outputs pass requested division and minimum length | 8/15 natural stops; the eight Chinese outputs pass requested division and minimum length | 10/15 natural stops; five Chinese dialogue outputs are under length |
| Translation | 15/15 natural stops; 15/15 pass structural checks | 15/15 natural stops; 15/15 pass structural checks | 15/15 natural stops; 15/15 pass structural checks |
| Structured | 0/15 satisfy the requested complete record/script contract | 0/15 satisfy the requested complete record/script contract | 0/15 satisfy the requested complete record/script contract |

The code prompts require complete runnable multi-file deliverables, but almost all outputs end at the
4,096-token limit. The three natural-stop exceptions also contain decisive contract failures: the
MTP3 CUDA response substitutes CUDA 12.8 and an older architecture list; the DFlash CUDA response
copies FP32 input into a half-sized 16-bit allocation and passes raw `unsigned short` values to BF16
intrinsics; and the DFlash Python response never writes its advertised JSONL event stream to the
configured log file. Code throughput is therefore a truncated-generation stress result, not
successful code-generation throughput.

All English mystery samples reach the output limit with an unfinished ending. The naturally stopped
Chinese stories have the requested chapter/act counts; the MTP3 and stochastic-DFlash samples also
meet their requested Chinese-character minima. Greedy's five dialogue stories contain 3,239 Chinese
characters each, below the requested 3,500. Story results are consequently a mixed normal/truncated
workload.

All translation outputs stop naturally. Each plain-document result preserves six sections and
provides at least twenty glossary entries; each Markdown result preserves heading levels, the
six-line table, all required inline identifiers, and the exact fenced JSON object. Translation is
the cleanest cross-scenario normal-completion comparison in this corpus.

The structured prompts intentionally exceed what these generations fit into 4,096 tokens. MTP3,
stochastic DFlash, and greedy DFlash produce only 49–60, 49–58, and 57 valid JSONL records,
respectively, versus the requested 160. Their complete-width CSV ranges are 122–139, 121–143, and
133 rows versus the requested 220. No SQL output satisfies all four tables, two views, at least 80
rows, and six final analytical queries. These high-acceptance results describe predictable partial
record generation only.

The exact-line and repeated-token scan found no other response with a short-cycle collapse comparable
to greedy AIME 30. Output-limit and prompt-compliance failures above remain material even when no
repetition loop is present.

## `qwen3_6_27b`

### EvalScope reasoning accuracy

Both weight profiles were evaluated through NInfer's OpenAI-compatible serving route with thinking
enabled, MTP=3, and a 262,144-token context limit. EvalScope 1.9.0 used 0-shot prompts, rule-based
scoring, and one sample per problem with temperature 0.6, top-p 0.95, top-k 20, presence penalty
1.0, and seed 42. All 258 samples completed and were scored for each profile.

| Weights ID | AIME 2025 | AIME 2026 | GPQA-Diamond |
|---|---:|---:|---:|
| `groupwise-int` | 86.67% (26 / 30) | 93.33% (28 / 30) | 86.87% (172 / 198) |
| `nvfp4` | 93.33% (28 / 30) | 93.33% (28 / 30) | 84.34% (167 / 198) |

These are single-sample results under the stated evaluation profile, not pass@k scores. Each
benchmark remains independently reportable; no combined score is computed.

### `groupwise-int`

#### MTP0 context-length profile

| Prompt tokens | Samples | Prefill tok/s | Server TTFT (ms) | Decode tok/s |
|---:|---:|---:|---:|---:|
| 7,680 | 5 | 3,218.1 ± 4.3 | 2,392.4 ± 3.0 | 77.6 ± 0.1 |
| 64,512 | 5 | 2,655.9 ± 2.9 | 24,335.7 ± 25.2 | 70.7 ± 0.1 |
| 130,048 | 5 | 2,185.3 ± 0.3 | 59,590.3 ± 8.9 | 64.5 ± 0.1 |
| 260,096 | 5 | 1,614.8 ± 0.6 | 161,221.8 ± 62.5 | 54.8 ± 0.1 |

#### MTP3 long-reasoning decode

| Fixture | Samples | Completion tokens | Decode tok/s | MTP acceptance | MTP tokens/round |
|---|---:|---:|---:|---:|---:|
| `long_decode_aime26_01` | 5 | 10,686.2 ± 553.8 | 175.4 ± 1.0 | 77.9% ± 0.9% | 3.34 ± 0.03 |
| `long_decode_aime26_15` | 5 | 61,604.2 ± 5,677.9 | 161.9 ± 2.8 | 73.4% ± 1.7% | 3.20 ± 0.05 |
| `long_decode_aime26_30` | 5 | 47,339.8 ± 9,162.2 | 172.2 ± 0.9 | 78.8% ± 0.8% | 3.36 ± 0.02 |

#### MTP3 cross-scenario decode

Each category contains three fixtures and five seeds per fixture, for 15 samples.

| Category | Samples | Decode tok/s | MTP acceptance | MTP tokens/round |
|---|---:|---:|---:|---:|
| Code | 15 | 167.0 ± 5.4 | 72.3% ± 3.5% | 3.17 ± 0.11 |
| Story | 15 | 112.6 ± 9.4 | 37.8% ± 5.9% | 2.13 ± 0.18 |
| Translation | 15 | 161.5 ± 11.3 | 68.3% ± 7.2% | 3.05 ± 0.22 |
| Structured | 15 | 193.0 ± 18.8 | 88.7% ± 11.7% | 3.66 ± 0.35 |

### `nvfp4`

The fixtures, seeds, sampling parameters, output limits, and runtime options are identical to the
groupwise-int serving campaign. Quantization can change sampled tokens, so the MTP3 results are a
fixed-workload comparison rather than a token-identical output comparison.

#### MTP0 context-length profile

| Prompt tokens | Samples | Prefill tok/s | Server TTFT (ms) | Decode tok/s |
|---:|---:|---:|---:|---:|
| 7,680 | 5 | 11,191.5 ± 70.2 | 692.5 ± 4.3 | 86.4 ± 0.5 |
| 64,512 | 5 | 6,298.5 ± 97.6 | 10,288.6 ± 159.3 | 78.0 ± 1.2 |
| 130,048 | 5 | 4,204.7 ± 14.1 | 31,012.5 ± 104.6 | 71.2 ± 0.2 |
| 260,096 | 5 | 2,510.6 ± 16.8 | 103,761.1 ± 698.8 | 59.9 ± 0.3 |

#### MTP3 long-reasoning decode

| Fixture | Samples | Completion tokens | Decode tok/s | MTP acceptance | MTP tokens/round |
|---|---:|---:|---:|---:|---:|
| `long_decode_aime26_01` | 5 | 12,053.4 ± 820.9 | 231.0 ± 3.0 | 80.2% ± 1.2% | 3.41 ± 0.04 |
| `long_decode_aime26_15` | 5 | 63,109.0 ± 5,426.9 | 213.1 ± 4.2 | 76.3% ± 2.0% | 3.29 ± 0.06 |
| `long_decode_aime26_30` | 5 | 57,166.4 ± 9,204.9 | 223.3 ± 1.8 | 81.1% ± 1.5% | 3.43 ± 0.04 |

#### MTP3 cross-scenario decode

Each category contains three fixtures and five seeds per fixture, for 15 samples.

| Category | Samples | Decode tok/s | MTP acceptance | MTP tokens/round |
|---|---:|---:|---:|---:|
| Code | 15 | 220.3 ± 8.2 | 74.2% ± 4.0% | 3.23 ± 0.12 |
| Story | 15 | 148.8 ± 11.6 | 39.2% ± 5.7% | 2.18 ± 0.17 |
| Translation | 15 | 213.6 ± 12.2 | 70.5% ± 6.0% | 3.12 ± 0.18 |
| Structured | 15 | 252.2 ± 16.3 | 89.8% ± 8.0% | 3.69 ± 0.24 |

The baseline and speculative-decode suites intentionally measure different supported workloads.
No per-scenario baseline/speculative speedup is reported.
