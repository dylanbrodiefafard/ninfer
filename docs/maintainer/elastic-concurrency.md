# Elastic concurrency lanes (plan, 2026-09-25)

Temporary plan: investigation of letting the active lane count move between a configured
`C_min` and `C_max` so that unused lanes give their memory back to the shared KV pool. It records
the per-lane cost, what has been implemented to shrink it, and the one remaining open candidate.
The current contract (`max_concurrency` fixed at startup) is unchanged; this file is removed once
candidate B is implemented or dropped.

Target: `qwen3.8-27b/nvfp4`, RTX 5090 / `sm_120a` / CUDA 13.1, NVFP4 KV, CUDA Graphs.

## What a lane costs

The KV pool is shared: its size does not scale with `C`, and a lane owns no context partition
(concurrent-inference-architecture §2.3, §6.2). A lane costs only its fixed-size device state plus
graph and frame rows. The Main KV page group is 1,179,648 B per 64 tokens (18 KiB/token; MTP adds
one layer). Workspace is 610.3 MiB for every `C`.

| Item per lane | Before A/C | After A/C |
|---|---:|---:|
| GDN current slot (48 layers, FP32 recurrent + BF16 conv) | 146.8 MiB | 146.8 MiB |
| GDN rewrite-checkpoint slot | 146.8 MiB | pinned host |
| DFlash local cyclic K/V, current / rewrite | 40 + 40 MiB | 40 MiB (+ 40 MiB pinned host) |
| ReplaySSM records (width 5) + frame rows | ≈ 10 MiB | ≈ 10 MiB |
| Graph allowance, adaptive DFlash K1–5 | 60 MiB | ≈ 30 MiB at C=4 (`min(12n, 24+6n)`) |
| **DFlash adaptive, in KV tokens** | **≈ 25.4k** | **≈ 12.9k** |
| **MTP K4, in KV tokens** | **≈ 16.9k** | **≈ 9.1k** |

## A. Host-resident rewrite checkpoint — implemented

The per-lane rewrite (turn) checkpoint's GDN slot and DFlash rewrite local lane are a lane-owned
pinned host image allocated with the Program; the device GDN pool is `C` slots plus one
Engine-wide staging slot (now index `C`). Capture D2Hs the current slot on the compute stream when
a prefill chunk ends at the checkpoint frontier; rewrite restore H2Ds it back. RAM/SSD tiers copy
the image on the host.

Measured (`--max-context 262144 --kv-capacity auto --kv-dtype nvfp4`, 1 GiB headroom, serve startup
before → after, together with C):

| Mode | C=1 | C=4 |
|---|---:|---:|
| DFlash K5 adaptive (`--lm-head-draft`) | 262144 → 262144 (capped at `L`) | 556800 → 604800 tokens (+48000) |
| MTP K4 | 262144 → 262144 (capped at `L`) | 635392 → 666816 tokens (+31424) |

Costs: capture and restore move 146.8 MiB (+40 MiB DFlash) over PCIe at a measured 28.8 GB/s,
about 5.4 ms (+1.5 ms) serialized on the compute stream per checkpoint, instead of a 0.31 ms D2D.
A RAM/SSD tier capture or restore that carries a rewrite checkpoint does one extra 147–187 MiB host
memcpy on the executor thread (≈ 10–13 ms at 15 GB/s). The pinned images take `C × 187 MiB` host
RAM (DFlash), outside `--kv-ram-capacity`.

## C. Graph allowance matched to measurement — implemented

Single-schedule graph families charge `min(12n, 24+6n)` MiB for `n` executables instead of `12n`.
Measured startup usage stays inside it (adaptive DFlash C=4: 100 of 144 MiB; C=1: 24 of 54 MiB;
historical maximum 46 MiB at `n=4`, 146 MiB at `n=30`).

## B. Elastic lanes by VMM aliasing — open, not recommended now

Size control structures for `C_max` and back the fixed state of lanes `[C_min, C_max)` with
physical 2 MiB chunks mapped twice and permanently (`cuMemCreate` + two `cuMemMap`s): into the
lane's GDN current slot and into a tail page-ID range `R_e` of the Main KV code planes. Ownership
of `R_e` is purely logical, so CUDA Graphs, kernels, and addressing are unchanged. It needs
lane-major GDN slot layout with 2 MiB-aligned per-layer runs, VMM-backed recurrent tensors and
KV planes, an allocator that keeps `R_e` at the top of the ID space and out of the free set while
its lane is active, lane activation in the admission vector, page migration (D2D + block-table
republication) to evacuate `R_e`, and contract updates (AGENTS product contract, concurrent
inference §1.1/§2.3/§6.2/§9.4, paged-KV §3/§5.2).

Expected overhead: none at steady state (one Engine A/B would confirm VMM bandwidth); ≤ 1 ms per
lane activation when migration is needed. Expected payoff after A/C: an idle elastic lane returns
at most its 146.8 MiB current slot plus 40 MiB DFlash local, about 10k tokens, so a 3..6 elastic
engine gains ≈ 30k tokens (≈ 5%) over a static C=6 engine and only while ≤ 3 requests are active.
That does not justify the VMM, migration, and contract work today.

## Follow-ups

1. Re-evaluate B only if per-lane device state grows again or `C_max` rises well above 6.
2. The capture/restore D2H/H2D could overlap the next prefill chunk by staging through a device
   slot on `copy_stream`, at the cost of one extra Engine-wide device slot; not needed at the
   measured ≈ 7 ms per checkpoint.
