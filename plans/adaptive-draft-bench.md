# Adaptive draft measurement log

Do not commit until Jared says so. Inventory for the T(K) / E / tok/s sweep that fills
`src/targets/qwen3_6/impl/runtime/adaptive_draft.h`.

Identity: Qwen3.8-27B, RTX 5090, `sm_120a`. Graphs on, `--lm-head-draft`, `--kv-dtype nvfp4`,
`--no-prefix-reuse`. Seed `7632647173703958409`; C=2 adds `7968175640111700217`. Sampling:
temp 0.6, top-p 0.95, top-k 20, min-p 0, presence 0. Skip C=3 (DFlash k=5 identity parked in
`plans/dflash-k5-c3-identity.md`).

tok/s is **request_done** mean `(completion−1)/decode_seconds`. C=1 saturation-interval tok/s
is not used. `T_ms = 1000 × E / tok/s`. `s_i = accepted_per_position[i] / rounds`.
Adaptive `hist` is `rounds_per_draft` (index = K). C=2 tok/s is the mean of the two requests.

## Artifacts

| Role | Path |
|---|---|
| MTP NVFP4 | `/ssdpool2nvme/local_llm/models/qwen3.8-nvfp4-mtp-nvfp4-from-bf16/qwen3_8_27b_nvfp4.ninfer` |
| DFlash2 NVFP4 + BF16 codebook | `/ssdpool2nvme/local_llm/models/qwen3.8-nvfp4-flash2-nvfp4-bf16codebook-from-bf16/qwen3_8_27b_nvfp4_dflash_nvfp4.ninfer` |

AIME: `long_decode_aime26_15` plus `_01` / `_30`.
Story: `scenario_story_en_mystery`, `zh_dialogue` (low-accept), `zh_scifi` MTP-only (DFlash KV miss).
Code: `scenario_code_cuda` and `scenario_code_python`, 2048 output, thinking off.
`scenario_code_typescript` stopped at 200 tokens (`finish=stop_token`); decode-saturation
requires `output_limit`, so it was dropped.
Raw: `profiles/bench/adaptive-tk-20260827/`. Resume: `python3 tools/bench/run_adaptive_draft_sweep.py`.

## Need vs have

| Item | Status |
|---|---|
| Static MTP 3/4/5 AIME C=1 | **have** |
| Static MTP 2/3/4/5 story C=1 | **have** (k=2 loses to k=3; do not capture 2) |
| Static DFlash 3/4/5/7 AIME C=1 | **have** (T[3] measured, not interpolated) |
| Static DFlash 4/5/7 story C=1 | **have** (k=4 wins story) |
| Adaptive MTP N=5 AIME+story C=1 | **have** |
| Adaptive DFlash N=7 AIME+story C=1 | **rebench** (live `{3,4,5}`) |
| Static DFlash 3 story / dialogue / code / AIME_01/_30 / C=2 | **pending** |
| Static MTP 3/4/5 AIME+story C=2 | **have** |
| Static DFlash 4/5/7 AIME+story C=2 | **have** |
| Adaptive MTP N=5 + DFlash N=7 C=2 | MTP **have**; DFlash **rebench** `{3,4,5}` |
| Code CUDA+Python C=1 frozen MTP3/4/5 + DFlash4/5 | **have** |
| Code CUDA+Python C=1 adaptive | **have** |
| Code TypeScript saturation | skip (natural stop at 200 tok) |

## Compiled table (header)

```text
MTP    T[3]=1.00  T[4]=1.10  T[5]=1.16     # 13.54 / 14.82 / 15.65 ms AIME
DFlash T[3]=0.91  T[4]=0.97  T[5]=1.00  T[7]=1.43
       # 13.87 / 14.79 / 15.24 / 21.85 ms AIME
E5=2.94  E7=3.21
```

## C=1 static

### MTP AIME

| K | tok/s | E | acc | T ms | rel | s_i |
|---:|---:|---:|---:|---:|---:|---|
| 3 | 173.7 | 2.35 | 45.1% | 13.54 | 1.00 | 0.674, 0.427, 0.252 |
| 4 | 176.7 | 2.62 | 40.5% | 14.82 | 1.10 | 0.684, 0.451, 0.296, 0.187 |
| 5 | 175.2 | 2.74 | 34.9% | 15.65 | 1.16 | 0.707, 0.461, 0.276, 0.178, 0.119 |

Static best AIME: **MTP4 176.7**.

### MTP story

| K | tok/s | E | acc | T ms |
|---:|---:|---:|---:|---:|
| 2 | 158.0 | 2.08 | 54.1% | 13.17 |
| 3 | 168.0 | 2.26 | 42.0% | 13.44 |
| 4 | 148.9 | 2.19 | 29.8% | 14.73 |
| 5 | 158.2 | 2.45 | 29.0% | 15.47 |

Static best story: **MTP3 168.0**. k=2 does not join the capture set.

### DFlash2 AIME

| K | tok/s | E | acc | T ms | rel | s_i |
|---:|---:|---:|---:|---:|---:|---|
| 3 | 179.9 | 2.49 | 49.8% | 13.87 | 0.91 | 0.707, 0.478, 0.309 |
| 4 | 182.5 | 2.70 | 42.5% | 14.79 | 0.97 | 0.713, 0.480, 0.302, 0.205 |
| 5 | 192.9 | 2.94 | 38.8% | 15.24 | 1.00 | 0.724, 0.503, 0.347, 0.220, 0.145 |
| 7 | 147.0 | 3.21 | 31.6% | 21.85 | 1.43 | 0.815, 0.554, 0.369, 0.245, 0.152, 0.077, 0.000 |

Static best AIME: **k=5 192.9**. Product default k=7 is 147. Slot 7 never accepts (`s_6=0`).

### DFlash2 story

| K | tok/s | E | acc | T ms |
|---:|---:|---:|---:|---:|
| 4 | 170.5 | 2.50 | 37.6% | 14.67 |
| 5 | 163.5 | 2.46 | 29.3% | 15.04 |
| 7 | 136.3 | 2.91 | 27.4% | 21.39 |

Static best story: **k=4 170.5**. Adaptive N≥5 (N≠6) captures `{4,5}` chain only.

## Retune (chain `{4,5}`, seed MTP4 / DFlash5)

Live policy no longer includes tree k=7. DFlash N≥5 captures `{4,5}`. Seed is the score
attractor (MTP 4, DFlash 5), not ceiling N. δ_add=0.05, δ_remove=0.02 (0.04 cannot fire
5→4: T4/T5 savings is only 3%). Warmup 8. tok/s is the latest `server_start` request_done
mean. Point JSON `steady_interval` is not used.

## C=1 adaptive vs frozen MTP3 and DFlash4

| Workload | Adaptive | Frozen MTP3 | Frozen DFlash4 | vs frozen |
|---|---:|---:|---:|---|
| AIME MTP N=5 | **176.8** | 173.7 | — | **+3.1 (+1.8%)** |
| story MTP N=5 | 154.9 | **168.0** | — | **−13.1 (−7.8%)** |
| AIME DFlash N=7 | **189.5** | — | 182.5 | **+7.0 (+3.8%)** |
| story DFlash N=7 | 166.2 | — | **170.5** | **−4.3 (−2.5%)** |

AIME adaptive now beats both frozen baselines and matches MTP4 (176.7) / nearly DFlash5 (192.9).
Story MTP still loses: seed 4 is that workload’s valley (25 rounds at 4 before 4→3).

| Run | tok/s | E | live | hist | vs static best | vs product k=N |
|---|---:|---:|---:|---|---|---|
| MTP N=5 AIME | 176.8 | 2.43 | 3 | 3:1583, 4:77, 5:23 | **+0.1 vs MTP4 176.7** | +3.1 vs MTP3 173.7 |
| MTP N=5 story | 154.9 | 2.12 | 3 | 3:448, 4:25, 5:8 | −13.1 vs MTP3 168.0 | −13.1 vs MTP3 |
| DFlash N=7 AIME | 189.5 | 2.87 | 4 | 4:803, 5:623 | −3.4 vs k=5 192.9 | **+42.5 vs k=7 147.0** |
| DFlash N=7 story | 166.2 | 2.47 | 4 | 4:406, 5:8 | −4.3 vs k=4 170.5 | **+29.9 vs k=7 136.3** |

DFlash story drops tree-less: 8 warmup rounds at 5 then sits at 4. AIME DFlash starts at 5
(623 rounds) then falls to 4 for the rest — still +7 vs frozen k=4.

## C=2 (mean of two requests)

### Static

| Mode | AIME tok/s | Story tok/s |
|---|---:|---:|
| MTP3 | 154.1 | 144.2 |
| MTP4 | 149.4 | 136.2 |
| MTP5 | 137.9 | 124.1 |
| DFlash4 | 150.6 | 141.4 |
| DFlash5 | 143.1 | 129.1 |
| DFlash7 | 126.2 | 113.7 |

C=2 static best: MTP3, DFlash4. Higher K loses more at C=2 than at C=1 (W4A4 / aggregate T).

### Adaptive vs frozen MTP3 and DFlash4 (C=2)

| Workload | Adaptive | Frozen MTP3 | Frozen DFlash4 | vs frozen |
|---|---:|---:|---:|---|
| AIME MTP N=5 | 150.9 | **154.1** | — | **−3.2 (−2.1%)** |
| story MTP N=5 | 141.7 | **144.2** | — | **−2.5 (−1.7%)** |
| AIME DFlash N=7 | **156.3** | — | 150.6 | **+5.7 (+3.8%)** |
| story DFlash N=7 | 140.4 | — | **141.4** | **−1.0 (−0.7%)** |

| Run | tok/s | live | hist | vs C=2 static best | vs C=2 k=N |
|---|---:|---:|---|---|---|
| MTP N=5 AIME | 150.9 | 3,3 | 3:3272, 4:164, 5:16 | −3.2 vs MTP3 154.1 | +13.0 vs MTP5 137.9 |
| MTP N=5 story | 141.7 | 3,3 | 3:858, 4:50, 5:16 | −2.5 vs MTP3 144.2 | +17.6 vs MTP5 124.1 |
| DFlash N=7 AIME | 156.3 | 4,4 | 4:1971, 5:948 | **+5.7 vs k=4 150.6** | +30.1 vs k=7 126.2 |
| DFlash N=7 story | 140.4 | 4,4 | 4:827, 5:18 | −1.0 vs k=4 141.4 | +26.7 vs k=7 113.7 |

C=2 DFlash AIME mixing 4 and 5 beats frozen k=4. Story DFlash matches k=4 after an 8-round
warmup at 5. MTP still slightly behind MTP3 (batch-max + 4→3 lag).

## C=1 code (2048 tokens, thinking off)

Static best: CUDA MTP4 266.6 / DFlash4 272.0. Python MTP5 286.7 / DFlash5 285.4.

| Workload | Adaptive | Frozen MTP3 | Frozen DFlash4 | vs frozen |
|---|---:|---:|---:|---|
| CUDA MTP | 220.0 | **241.0** | — | **−8.7%** (MTP4 is 266.6) |
| Python MTP | **290.5** | 259.9 | — | **+11.8%** (MTP5 is 286.7) |
| CUDA DFlash | 259.0 | — | **272.0** | **−4.8%** |
| Python DFlash | **282.1** | — | 272.8 | **+3.4%** (DFlash5 is 285.4) |

| Run | tok/s | E | acc | live | hist |
|---|---:|---:|---:|---:|---|
| CUDA MTP adaptive | 220.0 | 3.01 | 66.6% | 3 | 3:673, 4:8 |
| CUDA DFlash adaptive | 259.0 | 3.97 | 59.5% | 4 | 4:1, 5:514 |
| Python MTP adaptive | 290.5 | 4.50 | 70.4% | 3† | 3:1, 4:11, 5:443 |
| Python DFlash adaptive | 282.1 | 4.32 | 66.5% | 4† | 4:1, 5:473 |

† last-K is end-of-request budget clamp; hist is the regime.

CUDA MTP is the new failure: seed 4, one dwell, then 4→3 for the rest, E collapses below frozen
MTP3. Python MTP climbs 4→5 and matches frozen MTP5. DFlash holds 5 on both (CUDA’s static
best is 4; Python’s is 5).

## Retune 2 (slot-gated remove, first-remove 32, last-slot climb, p0 floor)

Shipped in `adaptive_draft.h`. Host test `ninfer_qwen3_6_adaptive_draft_test` passed. Adaptive
point JSON deleted; frozen jobs skipped. Same seeds / fixtures / artifacts.

Policy deltas vs retune 1:

- Drop K→K−1 only if extra slot `s[K-1]` is **seen and &lt; 0.30**.
- Add if score hysteresis **or** last slot `s[K-1] > 0.35` and `score_hi > score_live`.
- First **remove** warmup 32; add still after 8.
- If `s[0] > 0.65` and 4 is captured, floor live K at 4.
- Still no DFlash 3 or 6. Still no tree k=7.

**DFlash3?** No. Frozen AIME k=3 is 179.9 &lt; k=4 182.5 &lt; k=5 192.9. Capturing 3 recreates the
MTP 4→3 overshoot on a chain whose static peak is 5.

**DFlash6?** No. k≤5 is chain (`W=k+1`, one SmallT tile). k=6 is packed-tree, same family as 7,
not a +1 chain step. N=6 stays frozen `{6}`.

### Why CUDA DFlash adaptive lost to both frozen 4 and frozen 5

It is not a 5→4 miss on a shared prefix. Adaptive hist `4:1, 5:514` (end-budget clamp to 4).
`E=3.97` and `acc=59.5%` **match frozen k=5 exactly**, including `s_i = 0.837, 0.703, 0.571,
0.474, 0.388`. Extra slot `s[4]=0.388` is above `drop_slot_max=0.30`, so 5→4 is correctly
refused. Frozen k=4 is a **different token path** (`acc=75.4%`, `E=4.01`, `s_i` all hotter).

| | tok/s | E | acc | hist |
|---|---:|---:|---:|---|
| Frozen DFlash4 | **272.0** | 4.01 | 75.4% | 4:510 |
| Frozen DFlash5 | 263.3 | 3.97 | 59.5% | 5:515 |
| Adaptive N=7 (live `{4,5}`, seed 5) | 258.6 | 3.97 | 59.5% | 4:1, 5:514 |

258.6 vs 263.3 is the two-graph tax (~1.8%). 258.6 vs 272.0 is seed-5 + hot extra slot keeping
the k=5 trajectory. Retune 2 does not (and should not) dump a 0.39 extra-slot.

### C=1 adaptive vs frozen MTP3 / DFlash4 (and vs static best)

| Workload | Adaptive | Frozen MTP3 | Frozen DFlash4 | vs that frozen | vs static best |
|---|---:|---:|---:|---|---|
| AIME MTP | **178.2** | 173.7 | — | **+2.6%** | **+0.8% vs MTP4 176.7** |
| story MTP | 157.9 | **168.0** | — | **−6.0%** | −6.0% (was −7.8%) |
| AIME DFlash | **190.3** | — | 182.5 | **+4.3%** | −1.3% vs k=5 192.9 |
| story DFlash | 159.3 | — | **170.5** | **−6.6%** | −6.6% (was −2.5%; 32-round seed-5 tax) |

| Run | tok/s | E | live | hist |
|---|---:|---:|---:|---|
| MTP N=5 AIME | 178.2 | 2.55 | 3 | 3:838, 4:745, 5:24 |
| MTP N=5 story | 157.9 | 2.24 | 3 | 3:278, 4:154, 5:24 |
| DFlash N=7 AIME | 190.3 | 2.87 | 4 | 4:803, 5:623 |
| DFlash N=7 story | 159.3 | 2.37 | 4 | 4:399, 5:32 |

AIME DFlash hist is identical to retune 1 (same seed, 5→4 still fires once `s[4]` dies). Story
DFlash now waits 32 rounds at seed 5 (`s[4]=0` immediately) before dropping; that is the −6.6%
vs k=4. Story MTP still ends at 3 (`s[0]=0.66`, p0 floor does not apply; extra slot of 4 is
dead). More dwell at 4 (154 vs 25) recovered 154.9 → 157.9, still well below MTP3.

### C=2

| Workload | Adaptive | Frozen MTP3 | Frozen DFlash4 | vs frozen |
|---|---:|---:|---:|---|
| AIME MTP | 150.4 | **154.1** | — | **−2.4%** |
| story MTP | 139.2 | **144.2** | — | **−3.5%** |
| AIME DFlash | **153.1** | — | 150.6 | **+1.7%** |
| story DFlash | 136.8 | — | **141.4** | **−3.3%** |

| Run | tok/s | live | hist (summed) |
|---|---:|---:|---|
| MTP N=5 AIME | 150.4 | 3,3 | 3:1392, 4:1812, 5:78 |
| MTP N=5 story | 139.2 | 3,3 | 3:488, 4:364, 5:48 |
| DFlash N=7 AIME | 153.1 | 4,5 | 4:734, 5:2125 |
| DFlash N=7 story | 136.8 | 4,4 | 4:797, 5:66 |

C=2 AIME DFlash still beats frozen k=4 by mixing, but more time at 5 (first-remove + last-hot)
cuts the margin from +3.8% to +1.7%.

### C=1 code

Static best: CUDA MTP4 **266.6** / DFlash4 **272.0**. Python MTP5 **286.7** / DFlash5 **285.4**.
Frozen CUDA MTP5 is 264.6.

| Workload | Adaptive | vs MTP3 / DFlash4 | vs static best | vs retune 1 |
|---|---:|---|---|---|
| CUDA MTP | 247.7 | **+2.8% vs 241.0** | −7.1% vs MTP4 266.6 | **220.0 → 247.7** (4→3 trap gone) |
| Python MTP | **289.6** | +11.4% vs 259.9 | **+1.0% vs MTP5 286.7** | 290.5 → 289.6 (noise) |
| CUDA DFlash | 258.6 | −4.9% vs 272.0 | −1.8% vs k=5 263.3 | 259.0 → 258.6 (unchanged k=5) |
| Python DFlash | **281.5** | +3.2% vs 272.8 | −1.4% vs k=5 285.4 | 282.1 → 281.5 (noise) |

| Run | tok/s | E | acc | live | hist |
|---|---:|---:|---:|---:|---|
| CUDA MTP adaptive | 247.7 | 3.80 | 59.9% | 3† | 3:9, 4:153, 5:376 |
| CUDA DFlash adaptive | 258.6 | 3.97 | 59.5% | 4† | 4:1, 5:514 |
| Python MTP adaptive | 289.6 | 4.50 | 70.4% | 3† | 3:1, 4:11, 5:443 |
| Python DFlash adaptive | 281.5 | 4.32 | 66.5% | 4† | 4:1, 5:473 |

† last-K is end-of-request budget clamp.

CUDA MTP extra slot at 4 is hot (`frozen s[3]=0.58`), so 4→3 no longer fires. Geometric add then
climbs 4→5 (predicted E5 from k=4 p̄). Realized mixed-K E=3.80 is below both frozen MTP4 (3.94)
and MTP5 (4.10), which is the remaining −7% vs static best. Python still seeds 4, climbs, holds 5.

## Read

1. Do **not** add DFlash 3 or 6. Frozen AIME already ranks 5 &gt; 4 &gt; 3. k=6 is tree, not chain.
2. CUDA DFlash vs frozen 4 **and** 5: same k=5 trajectory as frozen 5 (hot `s[4]=0.388`); −1.8%
   is two-graph tax. Frozen 4 is a different, luckier accept path, not a policy miss.
3. Slot-gated remove **fixed the CUDA MTP 4→3 trap** (220 → 248) and did not change CUDA/Python
   DFlash (still glued to seed 5).
4. First-remove 32 **taxes DFlash story** (seed 5 is that workload’s slower K): C=1 −2.5% → −6.6%
   vs k=4. AIME DFlash hist unchanged.
5. Capture/seed unchanged: MTP `{3,4,5}` seed 4; DFlash `{4,5}` seed 5. No tree.

## Extra fixtures before retune 3 (current policy)

`scenario_story_zh_scifi` C=1 adaptive, 1024, thinking off, same seed. Frozen static K not run.

| Run | tok/s | E | acc | live | hist |
|---|---:|---:|---:|---:|---|
| zh MTP adaptive | 142.0 | 1.95 | 30.9% | 3 | 3:484, 4:40 |
| zh DFlash adaptive | 115.7 | 1.72 | 17.9% | 4 | 4:562, 5:32 |

Harder than EN mystery (MTP 158 / DFlash 159). DFlash extra slot dies immediately (`s[4]=0`); first-remove 32 is a pure seed-5 tax, same pattern as EN.

## Retune 3 (Sol #1+#2, not #3)

Shipped together because they split by backend: C=1 MTP jobs isolate the climb change; C=1 DFlash jobs isolate first-remove. **Not** shipped: batch-max aggregate K (#3), unseen-slot shrinkage toward 0.5 (#4), capture-set shrink (#5).

- MTP first climb to K+1 always needs δ_add; last-hot waives only after `s[K]` (extra of K+1) has been seen.
- Unseen tail `p̄ = min(mean conditional, last conditional)`.
- DFlash `first_remove_warmup = 8`; MTP stays 32.

Host tests passed, including unseen-s[4] does not waive and DFlash constant = 8.

### C=1 vs retune 2

| Workload | r2 | r3 | hist r3 | vs static best |
|---|---:|---:|---|---|
| AIME MTP | 178.2 | 178.0 | 3:838, 4:745, 5:24 (same) | +0.7% vs MTP4 176.7 |
| story MTP | 157.9 | 158.4 | 3:278, 4:154, 5:24 (same) | −5.7% vs MTP3 168.0 |
| AIME DFlash | 190.3 | 189.6 | 4:803, 5:623 (same) | −1.7% vs k=5 192.9 |
| story DFlash | 159.3 | **166.5** | 4:406, **5:8** (was 5:32) | −2.3% vs k=4 170.5 |
| CUDA MTP | 247.7 | **256.7** | **4:525, 5:8** (was 5:376) | −3.7% vs MTP4 266.6 |
| Python MTP | **289.6** | 266.0 | **4:439, 5:72** (was 5:443) | **−7.2% vs MTP5 286.7** |
| CUDA DFlash | 258.6 | 259.0 | 4:1, 5:514 (same) | −1.6% vs k=5 263.3 |
| Python DFlash | 281.5 | 281.6 | 4:1, 5:473 (same) | −1.3% vs k=5 285.4 |
| zh MTP | 142.0 | 141.6 | 3:484, 4:40 (same) | — |
| zh DFlash | 115.7 | 115.2 | 4:588, **5:8** (was 5:32) | — |

**#2 worked:** EN story DFlash 159.3 → 166.5 (matches retune-1 166.2). C=2 story DFlash 136.8 → 140.5 (matches retune-1 140.4). AIME/code DFlash histograms unchanged, as predicted (they do not drop at round 8).

**#1 mixed:** CUDA MTP stayed at 4 (247.7 → 256.7). Python MTP mostly stayed at 4 (289.6 → 266.0), which **trips Sol’s kill** (Python −8.1% vs r2, −7.2% vs frozen MTP5). AIME/story MTP hists unchanged — those workloads were not climbing on last-hot.

### C=2

| Workload | r2 | r3 | note |
|---|---:|---:|---|
| AIME MTP | 150.4 | 151.3 | more time at 4; still vs MTP3 154.1 |
| story MTP | 139.2 | 139.3 | same hist |
| AIME DFlash | 153.1 | 152.8 | same hist; still +1.5% vs k=4 150.6 |
| story DFlash | 136.8 | **140.5** | 5:66 → 5:18; recovered retune-1 |

## Read (retune 3)

1. Do **not** combine #1+#2+#3. This pair was attributable. Batch-max is still untested.
2. DFlash first-remove 8 is a keep: story recovered, AIME DFlash untouched.
3. Conservative first 4→5 **helps CUDA and hurts Python** on the same knob. Frozen CUDA MTP4 266.6 vs MTP5 264.6; frozen Python MTP4 274.9 vs MTP5 286.7. One climb gate cannot serve both.
4. zh-scifi is a harder story (E≈1.7–2.0). MTP already lives at 3; DFlash 32→8 at seed 5 did not move tok/s because both K are poor.
5. Next if keeping both code peaks: climb to 5 only after a **direct** k=5 observation (short dwell at 5) or a Python-preserving geometric, not a global unseen-slot freeze.

## zh dropped; aime_01 + Python probe (retune 4)

zh-scifi removed from the sweep. TypeScript still cannot saturate. `long_decode_aime26_01` stops
early at 4096 (K-dependent `stop_token`); caps are **1536 MTP / 1280 DFlash** so every K hits
`output_limit`.

Python MTP r3 (266.0) was geometric `score_hi > live` blocking 4→5. Last-hot now **probes** K+1
when `s[K-1] > 0.35` and compiled `T(K+1) ≤ 1.20·T(K)` (no geometric gap). Conservative `p̄`
remains for δ_add.

| Code MTP | r3 (no probe) | now | frozen best |
|---|---:|---:|---:|
| CUDA | **256.7** (held 4) | 249.8 (4:52, 5:471) | MTP4 **266.6** |
| Python | 266.0 (held 4) | **286.6** (4:37, 5:424) | MTP5 **286.7** |

### aime_01 vs its own static best

| | Adaptive | Frozen best | vs |
|---|---:|---:|---|
| MTP 1536 | 248.9 (4:21, 5:377) | MTP4 **260.6** (MTP5 240.6) | **−4.5%** |
| DFlash 1280 | 283.0 (4:1, 5:295) | k=5 **287.0** (k=4 269.1) | **−1.4%** |

aime_01 is a high-accept AIME (unlike `_15`). MTP static best is **4**, same pattern as CUDA code:
probe glues adaptive to 5 and loses. DFlash static best is 5; adaptive matches it minus tax.

## Retune 5 (paired MTP 4/5 trial + seed 3)

Shipped in `adaptive_draft.h` / MTP `program_impl.h`. Host test
`ninfer_qwen3_6_adaptive_draft_test` passed. DFlash policy unchanged (not re-benched). C=2
aggregate `ΣE/T_B(K)` not in this drop.

Policy deltas vs retune 4:

- MTP seed **3** (story static best). `s[0]>0.75` floor to 4 only after `observed >= warmup` (8).
- MTP 4→5 is a **one-shot paired trial**: after a hot k=4 window, run k=5 for 32 rounds, compare
  independently observed `E5/T5` vs the k=4 EWMA `E4/T4`, keep the strictly higher score (no δ),
  never re-probe. DFlash still last-hot probes.
- C=2 still `max(row_k)`.

Sol kill: Python &lt; 283.8, AIME_15 &lt; 176.7, or CUDA/AIME_01 do not beat 249.8/248.9.

### C=1 MTP adaptive

| Workload | r5 | r4 | frozen best | vs best | vs r4 | hist |
|---|---:|---:|---:|---:|---:|---|
| AIME_15 | 175.0 | 178.2 | MTP4 **176.7** | **−1.0%** | −3.2 | 3:985, 4:631, 5:32 |
| story | 159.7 | 158.5 | MTP3 **168.0** | **−4.9%** | +1.2 | 3:302, 4:152 |
| CUDA | 250.3 | 249.8 | MTP4 **266.6** | **−6.1%** | +0.5 | 3:46, 4:223, 5:273 |
| Python | 276.8 | **286.6** | MTP5 **286.7** | **−3.5%** | −9.8 | 3:12, 4:25, 5:442 |

| Run | tok/s | E | acc | live | hist |
|---|---:|---:|---:|---:|---|
| AIME_15 | 175.0 | 2.48 | 43.7% | 3 | 3:985, 4:631, 5:32 |
| story | 159.7 | 2.25 | 37.9% | 3 | 3:302, 4:152 |
| CUDA | 250.3 | 3.78 | 63.0% | 3† | 3:46, 4:223, 5:273 |
| Python | 276.8 | 4.27 | 66.9% | 3† | 3:12, 4:25, 5:442 |

† last-K is end-of-request budget clamp.

AIME_15 ran one 32-round k=5 trial then mixed 3/4 (seed-3 + no p0 floor: `s[0]=0.677`). **Kill**
(175.0 &lt; 176.7). Story never entered the trial (`s[2]` not last-hot); 3↔4 mix, still below MTP3.
CUDA entered the trial and **kept 5** (273 rounds): continuation `E5/T5` beat the k=4 snapshot, so
the independent frozen-4 path is not recovered. 250.3 vs r4 249.8 is noise. Python climbed 3→4→5
and kept 5, but seed-3 + trial tax dropped E 4.50→4.27. **Kill** (276.8 &lt; 283.8).

### aime_01 MTP

Harness `output_limit` 1536 failed: `stop_token` at **1215**. Not saturation-comparable to frozen
MTP4 260.6 @ 1536.

| | tok/s | gen | finish | hist |
|---|---:|---:|---|---|
| r5 adaptive | 260.8 | 1215 | stop_token | 3:9, 4:271, **5:32** |
| r4 adaptive | 248.9 | 1536 | output_limit | 4:21, 5:377 |
| frozen MTP4 | 260.6 | 1536 | output_limit | 4 |

Trial **returned to 4** here (unlike CUDA code). That is the intended high-accept-but-4-wins
shape; the early EOS is a different sample path from seed 3.

### C=2 MTP adaptive (mean of two requests)

| Workload | r5 | prior adaptive | Frozen MTP3 | vs MTP3 |
|---|---:|---:|---:|---:|
| AIME | 151.1 (153.9 / 148.3) | ~150–151 | **154.1** | −1.9% |
| story | 138.0 (138.8 / 137.2) | 139.2 | **144.2** | −4.3% |

AIME hist summed 3:1093, 4:1963, 5:186 (one 32-round trial per request plus leftover). Story no
k=5 (3:291, 4:605).

## Retune 6 (revert r5 + no last-hot 3→4)

r5 killed Python and AIME_15. This drop restores r4 (MTP seed 4, last-hot 4→5, no paired trial)
and **only** blocks last-hot 3→4. Geometric δ_add can still 3→4. DFlash unchanged. C=2 still
`max(row_k)`. Kill: AIME_15 &lt; 176.7 or Python &lt; 283.8.

Host test passed. MTP NVFP4 GPU tests passed (seeded k=4 rounds; footer live_k may budget-clamp).

### C=1 MTP adaptive

| Workload | r6 | r4 | r5 | frozen best | vs best | vs r4 | hist |
|---|---:|---:|---:|---:|---:|---:|---|
| AIME_15 | 172.1 | **178.2** | 175.0 | MTP4 **176.7** | **−2.6%** | −6.1 | 3:1129, 4:457, 5:94 |
| story | 157.8 | 158.5 | 159.7 | MTP3 **168.0** | **−6.1%** | −0.7 | 3:278, 4:154, 5:24 |
| CUDA | 249.0 | 249.8 | 250.3 | MTP4 **266.6** | **−6.6%** | −0.8 | 3:9, 4:52, 5:471 |
| Python | 285.4 | **286.6** | 276.8 | MTP5 **286.7** | −0.5% | −1.2 | 3:2, 4:37, 5:424 |

AIME_15 **kill** (172.1 &lt; 176.7). Blocking last-hot 3→4 stuck more rounds at 3 after a drop
(1129 vs r4 838). Story hist is unchanged vs retune 2 (`3:278, 4:154, 5:24`): geometric δ_add
still re-climbs 3→4, so the last-hot block did not close the story leak. CUDA/Python hists match
r4 (4→5 last-hot intact). Python 285.4 is within noise of r4 286.6 and **above** 283.8.

### aime_01 MTP 1536

| | tok/s | gen | finish | hist |
|---|---:|---:|---|---|
| r6 | 249.3 | 1536 | output_limit | 4:21, 5:377 |
| r4 | 248.9 | 1536 | output_limit | 4:21, 5:377 |
| frozen MTP4 | **260.6** | 1536 | output_limit | 4 |

Identical to r4 (glued to 5). r5’s stop_token @ 1215 was seed-3 path, not this policy.

### C=2 MTP adaptive

| Workload | r6 | r5 | Frozen MTP3 | vs MTP3 |
|---|---:|---:|---:|---:|
| AIME | **153.5** (156.8 / 150.3) | 151.1 | **154.1** | −0.4% |
| story | **143.4** (143.9 / 142.8) | 138.0 | **144.2** | −0.6% |

C=2 is the closest to MTP3 yet. Story still has k=4/k=5 mix (`3:407, 4:412, 5:48` summed).

## r6 vs static best (superseded by r7)

MTP was retune 6. DFlash was still retune 4. **r4 remained the best C=1 MTP set.** r6 is not live.

| Workload | Adaptive | Static best | vs |
|---|---:|---:|---|
| AIME_15 MTP | 172.1 | MTP4 176.7 | **−2.6%** (kill) |
| AIME_15 DFlash | 189.6 | k=5 192.9 | −1.7% |
| story MTP | 157.8 | MTP3 168.0 | **−6.1%** |
| story DFlash | 166.5 | k=4 170.5 | −2.3% |
| CUDA MTP | 249.0 | MTP4 266.6 | **−6.6%** |
| Python MTP | 285.4 | MTP5 286.7 | −0.5% |
| CUDA DFlash | 259.0 | k=4 272.0 / k=5 263.3 | −4.8% / −1.6% |
| Python DFlash | 281.6 | k=5 285.4 | −1.3% |
| AIME_01 MTP | 249.3 | MTP4 260.6 | −4.3% |
| AIME_01 DFlash | 283.0 | k=5 287.0 | −1.4% |

## Retune 7 (r4 last-hot restored + C=2 ΣE/T_B)

r6 killed AIME_15. Live policy is r4 again (last-hot 3→4 restored, seed MTP 4 / DFlash 5, no
trial). C=1 still `max(row_k)`. C≥2 batch K is `adaptive_batch_k_sum_score`: maximize
`Σ_row E_row(K) / T_B(K)` with C=2 T tables; ties keep lower K; candidates `K ≤ max(row_k)`.
Kill: C=2 AIME ≥ 154.1 and story ≥ 144.2; C=1 must stay at r4.

### C=1 MTP (r4 restore)

| Workload | r7 | r4 | frozen best | hist |
|---|---:|---:|---:|---|
| AIME_15 | 174.7 | **178.2** | MTP4 176.7 | 3:855, 4:563, 5:210 |
| story | 157.8 | 158.5 | MTP3 168.0 | 3:278, 4:154, 5:24 |
| CUDA | 249.1 | 249.8 | MTP4 266.6 | 3:9, 4:52, 5:471 |
| Python | 285.0 | 286.6 | MTP5 286.7 | 3:2, 4:37, 5:424 |
| AIME_01 | 248.9 | 248.9 | MTP4 260.6 | 4:21, 5:377 |

CUDA/Python/story hists match r4. AIME_15 174.7 is below r4 178.2 (more k=5 than r4).

### C=2 aggregate (Sol #1)

| Workload | r7 aggregate | Frozen best | vs frozen |
|---|---:|---:|---|
| C=2 AIME MTP | **153.7** (154.2/153.2) hist 3:1616+1624 | MTP3 **154.1** | −0.4% |
| C=2 story MTP | 139.4 (140.1/138.8) hist 3:444+450 | MTP3 **144.2** | **−3.3%** (kill) |
| C=2 AIME DFlash | **155.3** (159.1/151.5) hist mostly 4 | DFlash4 **150.6** | **+3.1%** |
| C=2 story DFlash | 139.2 (141.9/136.5) hist mostly 4 | DFlash4 **141.4** | −1.6% |

MTP C=2 AIME sits at k=3 and matches MTP3. Story MTP missed 144.2. DFlash C=2 AIME beats frozen 4.

## New categories (translation + structured JSONL, 512 decode, thinking off)

Unused corpus categories. Both hit `output_limit` at 512 (trans EOS'd at 749 under 2048). Static
best is **K=5** on both (high-accept, Python-like). Seed-4 warmup then last-hot 4→5.

| Workload | Adaptive | Frozen best | vs | hist |
|---|---:|---:|---|---|
| trans MTP | 265.2 | MTP5 **268.3** | −1.2% | 3:1, 4:8, 5:116 |
| trans DFlash | 290.5 | k=5 **295.0** | −1.5% | 4:1, 5:114 |
| jsonl MTP | 337.7 | MTP5 **343.5** | −1.7% | 3:1, 4:8, 5:89 |
| jsonl DFlash | 366.8 | k=5 **372.3** | −1.5% | 4:1, 5:90 |

| Frozen | tok/s | E | acc |
|---|---:|---:|---|
| trans MTP3/4/5 | 243.2 / 254.3 / **268.3** | 2.28 / 2.76 / 3.15 | 76.2 / 69.3 / 63.1 |
| trans DFlash4/5 | 263.8 / **295.0** | 2.87 / 3.44 | 72.1 / 69.4 |
| jsonl MTP3/4/5 | 285.4 / 312.8 / **343.5** | 2.84 / 3.60 / 4.32 | 95.0 / 90.1 / 87.2 |
| jsonl DFlash4/5 | 302.3 / **372.3** | 3.44 / 4.62 | 86.1 / 93.1 |

DFlash adaptive `s_i` matches frozen k=5 exactly (trans extra slot 0.53, jsonl 0.857). The ~1.5%
gap is two-graph tax, same as CUDA DFlash. MTP gaps include 8 seed-4 rounds on a short 512-token
gen. Do not capture K=3 for these; static 5 wins by a lot.

## Sol #2 one-class tax (CUDA, adaptive `|K|=1` vs frozen same K)

Product capture is unchanged. MTP `--draft-tokens 3 --adaptive-draft` and DFlash `--draft-tokens 4
--adaptive-draft` are already singleton even with the flag.

| | Adaptive `{K}` | Frozen K | vs | E / acc / hist |
|---|---:|---:|---|---|
| CUDA MTP3 | 240.7 | **241.0** | **−0.12%** | 2.25 / 75.2% / 3:629 identical |
| CUDA DFlash4 | 271.8 | **272.0** | **−0.07%** | 3.01 / 75.4% / 4:510 identical |

`s_i` matches frozen byte-for-byte. Adaptive-body tax is ~0. CUDA DFlash N=7 adaptive 259.0 vs
frozen k=5 263.3 (~1.6%) is therefore two-graph inventory, not the EWMA/dwell body.

## Sol 5.6 remaining

1. C=2 `ΣE/T_B` — shipped. MTP story missed kill. Keep the helper; do not retune C=1 to chase C=2.
2. One-class `{K}` vs frozen K — done. Body tax ~0.
3. Stop C=1 EWMA retunes. CUDA-4 vs Python-5 is path-dependent.
4. Do not `drop_slot_max=0.40`.

## Swap trans/jsonl → AIME_30 + zh-dialogue

Trans/jsonl glued to K=5 (90%+ accept). Replaced with `long_decode_aime26_30` (thinking on, 4096)
and `scenario_story_zh_dialogue` (thinking off, 1024). MTP3 AIME_30 hit `output_limit` at 4096.

### AIME_30 (medium, ~180–195 tok/s)

Frozen MTP **3 wins** (193.5). Unlike AIME_15 (MTP4 176.7). Extra slot of 4 is 0.194 (drop-worthy).

| | Adaptive | Frozen best | vs | hist |
|---|---:|---:|---|---|
| MTP | 174.6 | MTP3 **193.5** (4=180.9, 5=177.2) | **−9.8%** | 3:870, 4:587, 5:173 |
| DFlash | **199.2** | k=4 **195.3** (k=5=191.1) | **+2.0%** | 4:252, 5:1088 |

MTP adaptive seed-4 / last-hot climbs into 4/5 on a workload whose static peak is 3. DFlash
adaptive beats frozen 4 while mostly at 5 (different path than frozen 5 at 191.1).

### zh-dialogue (low, ~100–145 tok/s)

Frozen MTP **3 wins** (143.2). Acc ~31%, same band as zh-scifi / harder than EN mystery.

| | Adaptive | Frozen best | vs | hist |
|---|---:|---:|---|---|
| MTP | **142.9** | MTP3 **143.2** (4=134.4, 5=128.3) | **−0.2%** | 3:471, 4:48 |
| DFlash | 103.9 | k=4 **110.5** (k=5=101.3) | −6.0% | 4:653, 5:8 |

MTP drops 4→3 after seed-4 warmup and stays; extra slots are dead (`s[3]≈0`). DFlash is the
seed-5 eight-round tax on a chain whose extra slot is 0.014.

## Extra low-accept (zh-scifi) + failed high-pin probes

`scenario_story_zh_scifi` C=1 MTP 1024, thinking off. DFlash 1024 hit `Paged KV materialize
extent is outside entitlement` (auto KV still 16384); skipped. en→zh MTP3 EOS at 621, acc 71%
(pin). SQL MTP3 1024 acc 77% (pin). Dropped both.

| Frozen MTP | tok/s | E | acc |
|---|---:|---:|---:|
| zh-scifi 3/4/5 | **144.3** / 137.4 / 131.5 | 0.94 / 1.03 / 1.03 | 31.5 / 25.7 / 20.7 |

r7 adaptive 142.0 (−1.6% vs MTP3), hist 3:484, 4:40. Same shape as dialogue.

## Retune 8 (last-hot requires score_hi > live) — KILLED, reverted

New signal, not a δ tweak: last-hot still waives δ_add but only if `score(k_hi) > score(live)`.
Intended to stop AIME_30 3→4 (extra-of-3 hot, E/T of 4 loses) without blocking AIME_15 geometric
4. Host test passed. Reverted immediately after the table: Python hist **identical to r3**.

| Workload | r8 | r7 | frozen best | vs frozen | hist |
|---|---:|---:|---:|---:|---|
| AIME_15 MTP | **178.4** | 174.7 | MTP4 176.7 | +1.0% | 3:838, 4:745, 5:24 (=r4) |
| story MTP | 158.5 | 157.8 | MTP3 168.0 | −5.7% | 3:278, 4:154, 5:24 |
| CUDA MTP | 248.0 | 249.1 | MTP4 266.6 | −7.0% | 3:9, 4:153, 5:376 |
| Python MTP | **266.2** | **285.0** | MTP5 286.7 | **−7.1% kill** | **4:439, 5:72** (=r3) |
| AIME_30 MTP | 181.5 | 174.6 | MTP3 193.5 | −6.2% | 3:842, 4:724, 5:15 |
| dialogue MTP | 143.1 | 142.9 | MTP3 143.2 | −0.1% | 3:471, 4:48 |
| zh-scifi MTP | 141.9 | 142.0 | MTP3 144.3 | −1.7% | 3:484, 4:40 |
| AIME_01 MTP | 249.9 | 248.9 | MTP4 260.6 | −4.1% | 4:60, 5:338 |

C=2 MTP AIME μ 158.3 vs MTP3 154.1; story 139.5 vs 144.2 (still miss). DFlash C=1 mostly unchanged
except AIME_30 199.2→189.7 (lost the r7 path win).

Live tree is r7 again (`last_hot_probe = extra_hot && t_sane`).

## DFlash capture `{3,4,5}` (this campaign)

AIME_01 DFlash k=3 EOS at 1257/1280 (`stop_token`); not a saturation point. C=2 T[3] measured
0.82 vs compiled 0.83; kept 0.83.

Frozen DFlash k=3 vs prior static best:

| Workload | k=3 | prior best | k=3 vs best |
|---|---:|---:|---|
| dialogue | **115.9** | k=4 110.5 | **+4.9%** |
| story | 160.1 | k=4 **170.5** | −6.1% |
| AIME_15 | 179.9 | k=5 **192.9** | −6.7% |
| AIME_30 | **200.1** | k=4 195.3 | **+2.5%** |
| CUDA | 227.4 | k=4 **272.0** | −16.4% |
| Python | 254.8 | k=5 **285.4** | −10.7% |
| C=2 AIME | **158.9** | k=4 150.6 | **+5.5%** |
| C=2 story | **142.6** | k=4 141.4 | **+0.8%** |

Adaptive `{3,4,5}` seed 5, p0 floor 0.65, DFlash `drop_to_3_max=0.08`:

| Workload | Adaptive | Frozen best | vs | hist |
|---|---:|---:|---|---|
| dialogue | 112.9 | k=3 **115.9** | −2.6% | 3:632, 4:8, 5:8 |
| story | 158.0 | k=4 **170.5** | −7.3% | 3:118, 4:317, 5:8 |
| AIME_15 | 184.5 | k=5 **192.9** | −4.3% | 3:175, 4:681, 5:623 |
| AIME_30 | **203.9** | k=3 200.1 | **+1.9%** | 3:40, 4:209, 5:1067 |
| CUDA | 259.0 | k=4 272.0 / k=5 263.3 | −4.8% / −1.6% | 3:1, 5:514 |
| Python | 281.4 | k=5 **285.4** | −1.4% | 3:1, 5:473 |
| C=2 AIME | 155.5 | k=3 **158.9** | −2.2% | 3:2897, 4:185, 5:147 |
| C=2 story | 141.2 | k=3 **142.6** | −1.0% | 3:900, 5:2 |

Dialogue sits at 3 after seed-5 warmup (the point of capturing 3). Story still leaks some
4→3 (p0=0.65 holds most rounds at 4; a p0-only trial hit 168.1). AIME spends more time at 4
than the first `{3,4,5}` run (193.7 hist 5:969). 3↔4 is not a flap; it is a one-way drop.

## Origin packed-tree pull (20260830)

Stash / pull / pop onto origin packed-tree DFlash (`tree_verify=true`, W=12 at k=6/7).
Adaptive live set stays `{3,4,5}`. Frozen tree k=6/7 rebench vs chain: T_ms ≈ 19.0 vs
chain k=5 ≈ 15.2 (rel T=1.25, was T[7]=1.43). High-accept CUDA/Python still lose
13%+; AIME k=7 147→173 still behind chain k=5 193. k=7 `s[6]=0`. Do not put tree in
the live set. `kAdaptiveDflashT[6]=T[7]=1.25`.

Fresh C=1/C=2 frozen 3/4/5 vs adaptive on the post-origin binary:
`profiles/bench/adaptive-tk-20260830/`. tok/s = request_done `(completion−1)/decode`.

| Workload | k=3 | k=4 | k=5 | Frozen best | Adaptive | vs | hist |
|---|---:|---:|---:|---|---:|---:|---|
| dialogue | **109.4** | 108.8 | 101.6 | k=3 | 110.8 | **+1.2%** | 3:628, 4:8, 5:8 |
| story | 169.3 | **171.3** | 167.4 | k=4 | 166.2 | −3.0% | 3:37, 4:359, 5:8 |
| AIME_15 | 186.8 | 188.2 | **192.3** | k=5 | 188.4 | −2.1% | 3:90, 4:311, 5:992 |
| AIME_30 | 194.7 | **205.3** | 204.2 | k=4 | 207.8 | **+1.2%** | 3:1, 4:31, 5:1221 |
| CUDA | 203.4 | **247.5** | 228.3 | k=4 | 227.7 | −8.0% | 3:1, 5:574 |
| Python | 221.8 | 245.4 | **277.2** | k=5 | 260.6 | −6.0% | 3:1, 4:9, 5:491 |
| C=2 AIME | **164.2** | 154.4 | 152.7 | k=3 | 158.2 | −3.7% | 3:2837, 4:212, 5:58 |
| C=2 story | **151.5** | 138.7 | 127.1 | k=3 | 147.3 | −2.8% | 3:844, 4:2, 5:2 |

Two high-accept lags, two causes:

1. CUDA glued to k=5. Frozen k=4 E=3.76 `s[4 last]=0.535`; k=5 E=3.56 `s[4]=0.282`.
   `extra_slot_dead` is true but `score(4) > score(5)×1.02` fails (T[5]/T[4]=1.03).
2. Python at the right K (hist 5:491) still −6%. Adaptive N=7 storage W=12 (tree
   default) even though live k=5 is chain W=6.

Low accept is not the tree/adaptive high-accept problem. Do not retune 3↔4 / p0
floor / `drop_to_3_max` / last_hot 4→5 (Python needs last_hot; r8 score gate
stuck Python at 4).

## r2 try (20260830)

Code:

- `5→4` drop: if `live_k==5` and extra slot dead, waive the score gate. Symmetric
  to last_hot climb. `4→3` still needs score + `drop_to_3_max=0.08`.
  Also: if `s[4]` is seen and dead, do not last-hot 4→5 (stops CUDA 4↔5 flap;
  Python `s[4]≈0.47` is not dead so last-hot still climbs).
- Adaptive DFlash `dflash_storage_verify_width`: ceil = max default W(k) over
  `captured_ks`. `{3,4,5}` → W=6. Frozen `--draft-tokens 7` unchanged (W=12).
  Workspace tree flag follows captured K, not N.

Expected: CUDA hist moves toward 4; Python tok/s closer to frozen 277; dialogue
unchanged (never sits at 5 with a dead extra that would then cascade in one
round — 5→4 still has to pass 4→3 separately). AIME_15 `s[4]=0.164` may drop
to 4 (~188 vs frozen k=5 192).

Frozen k=3/4/5 reused from `adaptive-tk-20260830/` (non-adaptive path unchanged).
Adaptive rerun: `profiles/bench/adaptive-tk-20260830-r2/`. tok/s = request_done
jsonl `(completion−1)/decode` (dialogue json `steady_interval` 117.4 is not used).

Startup fail on first adaptive serve: `DFlash decode state verify width does not
match the draft window`. Constructor required `N <= W-1`; adaptive N=7 with
W_ceil=6 is 7 > 5. Relaxed: frame is W-wide, N may exceed W-1 when captured K
is a chain subset.

| Workload | Frozen best | r1 adaptive | r2 | vs r1 | vs frozen | hist r2 |
|---|---|---:|---:|---:|---:|---|
| dialogue | k=3 109.4 | 110.8 | 110.6 | −0.2% | +1.1% | 3:628, 4:8, 5:8 |
| story | k=4 171.3 | 166.2 | 166.8 | +0.4% | −2.6% | 3:37, 4:359, 5:8 |
| AIME_15 | k=5 192.3 | 188.4 | 196.9 | **+4.5%** | **+2.4%** | 3:43, 4:980, 5:332 |
| AIME_30 | k=4 205.3 | 207.8 | 193.0 | −7.1% | −6.0% | 3:29, 4:1229, 5:123 |
| CUDA | k=4 247.5 | 227.7 | 211.6 | −7.1% | −14.5% | 3:16, 4:608, 5:8 |
| Python | k=5 277.2 | 260.6 | 246.7 | −5.3% | −11.0% | 3:12, 4:522, 5:8 |
| C=2 AIME | k=3 164.2 | 158.2 | 164.7 | **+4.1%** | **+0.3%** | mostly 3 |
| C=2 story | k=3 151.5 | 147.3 | 147.5 | +0.1% | −2.6% | mostly 3 |

r2 did not help overall. High-accept (CUDA/Python) got worse. Cause: DFlash
`first_remove=8`, so the 5→4 waiver fired after 8 Bernoulli samples of s[4].
Python true s[4]≈0.47 often looks dead in 8 rounds; EWMA then freezes because
k=4 rounds do not update s[4]; last-hot 4→5 is blocked on that stale bit.
CUDA hist 4:608 as intended, but tok/s 211 vs frozen k=4 247 (token stream
diverged after 8×k=5, s[3]=0.351 vs frozen 0.535). AIME_15 did drop 5→4 and
won; AIME_30 same drop lost.

W_ceil: workspace still 723812352 B (unchanged). Sequence 1171→1157 MiB.
Python never sat at k=5 so the W=12 vs W=6 verify path was not measured.

## r3 try (20260830)

5→4 waiver additionally requires `observed >= kAdaptiveFirstRemoveWarmup` (32),
not DFlash's 8-round first_remove. Last-hot block and W_ceil stay. Host test:
observed=8 CUDA-like slots hold 5; observed=32 drops.

Expected: Python stays at 5 (true s[4]≈0.47 after 32 rounds) so W_ceil=6 is
actually tested vs frozen 277. CUDA drops after ~32 rounds, stays at 4.
AIME_15/30 still likely drop once s[4] is dead with 32 samples.

Adaptive rerun: `profiles/bench/adaptive-tk-20260830-r4/`. tok/s = request_done jsonl.

| Workload | Frozen best | r1 | r4 | vs r1 | vs frozen | hist r4 |
|---|---|---:|---:|---:|---:|---|
| dialogue | 109.4 | 110.8 | 108.8 | −1.8% | −0.5% | 3:615, 4:8, 5:32 |
| story | 171.3 | 166.2 | 168.2 | +1.2% | −1.8% | 3:11, 4:356, 5:32 |
| AIME_15 | 192.3 | 188.4 | 196.9 | **+4.5%** | **+2.4%** | 3:43, 4:980, 5:332 |
| AIME_30 | 205.3 | 207.8 | 193.3 | −7.0% | −5.9% | 3:29, 4:1229, 5:123 |
| CUDA | 247.5 | 227.7 | 234.7 | **+3.1%** | −5.2% | 3:2, 4:536, 5:32 |
| Python | 277.2 | 260.6 | 248.9 | −4.5% | −10.2% | 4:496, 5:40 |
| C=2 AIME | 164.2 | 158.2 | 158.5 | +0.2% | −3.5% | mostly 3 |
| C=2 story | 151.5 | 147.3 | 149.3 | +1.3% | −1.5% | mostly 3 |

5→4 delay worked (CUDA/dialogue/story hist 5:32). Python stayed 40 rounds at 5 then
dropped; aggregate s[4]=0.013 so those 40 last-slot accepts were still cold, and
last-hot 4→5 was blocked on stale s[4] so it never resampled k=5. W_ceil at
sustained k=5 still untested. CUDA 234 is better than r1 228 (glued at 5).

## r5 try (20260830)

Revert the last-hot 4→5 block. 5→4 still waits 32 and waives score. Python at k=4
with hot s[3] can probe 5 again so s[4] can accumulate. CUDA may 4↔5 flap (8-round
dwell); r4-at-4 was 234 vs r1-at-5 228 so flap should not be a large loss.

Adaptive: `profiles/bench/adaptive-tk-20260830-r5/`. tok/s = request_done jsonl.

| Workload | Frozen best | r1 | r4 | r5 | vs r1 | vs frozen | hist r5 |
|---|---|---:|---:|---:|---:|---:|---|
| dialogue | 109.4 | 110.8 | 108.8 | 108.8 | −1.8% | −0.6% | 3:615, 4:8, 5:32 |
| story | 171.3 | 166.2 | 168.2 | 168.2 | +1.2% | −1.8% | 3:11, 4:356, 5:32 |
| AIME_15 | 192.3 | 188.4 | 196.9 | 187.3 | −0.6% | −2.6% | 3:44, 4:1095, 5:286 |
| AIME_30 | 205.3 | 207.8 | 193.3 | 210.1 | +1.1% | **+2.3%** | 3:42, 4:714, 5:509 |
| CUDA | 247.5 | 227.7 | 234.7 | 251.7 | **+10.5%** | **+1.7%** | 3:2, 4:169, 5:354 |
| Python | 277.2 | 260.6 | 248.9 | 262.1 | +0.6% | −5.5% | 4:132, 5:373 |
| C=2 AIME | 164.2 | 158.2 | 158.5 | 158.6 | +0.2% | −3.4% | mostly 3 |
| C=2 story | 151.5 | 147.3 | 149.3 | 148.1 | +0.5% | −2.3% | mostly 3 |

Keep r5 policy. CUDA is the one workload that was −8% vs frozen; it is now
slightly above frozen k=4 (hist mix 4/5, s[4]=0.307). Python mostly at 5
(262 vs r1 261) — W_ceil=6 vs old W=12 did **not** close the 277 gap. AIME_30
recovered (r4's all-4 193 was the miss). AIME_15 is 4-heavy again and a bit
behind r1. Low-accept dialogue/story/C=2 unchanged within noise.

Net vs r1: CUDA +10.5%, everything else ≈0±2%. Not a uniform win, but the
high-accept CUDA hole is closed and nothing else broke badly.

Do not restore the last-hot 4→5 block. Do not fire 5→4 before 32 observations.

## r6 try (20260830)

Two new C=1 DFlash benches, measured on **r5 binary first**, then the AIME_15 fix:

- `long_decode_aime26_01` at **1280** decode (output_limit; 4096 hits K-dependent stop).
- `scenario_structured_sql` at **2048**, thinking off.

Frozen k=3/4/5 for those go in `adaptive-tk-20260830/`. r5 adaptive for the new
jobs in `adaptive-tk-20260830-r5/`.

| Workload | k=3 | k=4 | k=5 | Frozen best | r5 | vs | hist r5 |
|---|---:|---:|---:|---|---:|---:|---|
| AIME_01 | 236.0 | 256.9 | **274.4** | k=5 | 297.5 | **+8.4%** | 3:2, 4:32, 5:244 |
| SQL | 235.6 | **276.4** | 272.9 | k=4 | 303.7 | **+9.9%** | 4:8, 5:426 |

AIME_01 `s[4]=0.429` and SQL `s[4]=0.441` are not dead, so the 5→4 waiver is not
their story. r5 already sits at 5 and beats frozen.

Opt: waive 5→4 only if extra of 5 is dead **and** `s[3]` is last-hot (`> 0.35`).
CUDA k=5 `s[3]=0.379` still drops; AIME_15 `s[3]=0.235` does not (score gate
like r1, stay at 5). Python `s[4]` not dead. 4→3 / last-hot 4→5 unchanged.

Adaptive r6: `profiles/bench/adaptive-tk-20260830-r6/`. tok/s = request_done jsonl.

| Workload | Frozen best | r5 | r6 | r6 vs r5 | r6 vs frozen | hist r6 |
|---|---|---:|---:|---:|---:|---|
| dialogue | k=3 109.4 | 108.8 | 108.8 | 0.0% | −0.5% | 3:615, 4:8, 5:32 |
| story | k=4 171.3 | 168.2 | 173.2 | +3.0% | **+1.1%** | 3:2, 4:281, 5:103 |
| AIME_15 | k=5 192.3 | 187.3 | 187.4 | +0.1% | −2.5% | 3:24, 4:983, 5:413 |
| AIME_30 | k=4 205.3 | 210.1 | 209.3 | −0.4% | **+2.0%** | 3:5, 4:377, 5:880 |
| AIME_01 | k=5 274.4 | 297.5 | 297.8 | +0.1% | **+8.5%** | 3:2, 4:32, 5:244 |
| CUDA | k=4 247.5 | 251.7 | 226.4 | **−10.1%** | −8.5% | 3:1, 4:135, 5:449 |
| Python | k=5 277.2 | 262.1 | 275.2 | **+5.0%** | −0.7% | 3:2, 4:90, 5:389 |
| SQL | k=4 276.4 | 303.7 | 303.6 | 0.0% | **+9.8%** | 4:8, 5:426 |
| C=2 AIME | k=3 164.2 | 158.6 | 164.4 | +3.6% | **+0.1%** | mostly 3 |
| C=2 story | k=3 151.5 | 148.1 | 149.2 | +0.7% | −1.5% | mostly 3 |

AIME_15 stayed 4-heavy. 32-round EWMA of `s[3]` is not a stable CUDA/AIME
split: first boolean 0/1 plus α=1/16.5 leaves CUDA true 0.379 and AIME 0.235
on the wrong side of 0.35 often. CUDA lost the r5 5→4 drop (226 ≈ r1 glued-at-5
228). Python/SQL/AIME_01 already had hot `s[4]` so the waiver was never their
path; Python 262→275 is more time at 5.

Do not keep r6 as the CUDA policy. r5 remains the better CUDA/AIME_15 pair
until a less noisy discriminator exists.

## r7 try (20260830)

Same CUDA/AIME split as r6 (`s[3] > 0.35` to waive 5→4) but wait **64** observations
so EWMA forgets the first 0/1 (`0.94^64 ≈ 0.02`). Score 5→4 still at 32 (neither
CUDA nor AIME_15 win that gate). Last-hot 4→5 unchanged (CUDA flap that made r5
251). AIME_15 should stay at 5; CUDA should drop after ~64 then mix 4/5.

Adaptive: `profiles/bench/adaptive-tk-20260830-r7/`. tok/s = request_done jsonl.
AIME_01 r7 used 1024 decode (1280 hit stop_token 13 tokens early).

| Workload | Frozen best | r5 | r6 | r7 | r7 vs r5 | r7 vs frozen | hist r7 |
|---|---|---:|---:|---:|---:|---:|---|
| dialogue | 109.4 | 108.8 | 108.8 | 108.7 | −0.1% | −0.6% | 3:615, 4:8, 5:32 |
| story | 171.3 | 168.2 | 173.2 | 173.0 | +2.9% | **+1.0%** | 3:2, 4:281, 5:103 |
| AIME_15 | 192.3 | 187.3 | 187.4 | 192.2 | **+2.6%** | **0.0%** | 3:10, 4:722, 5:645 |
| AIME_30 | 205.3 | 210.1 | 209.3 | 215.2 | +2.4% | **+4.8%** | 3:2, 4:142, 5:1080 |
| AIME_01 | 274.4 | 297.5 | 297.8 | 282.4 | −5.1% | **+2.9%** | 4:24, 5:209 |
| CUDA | 247.5 | 251.7 | 226.4 | 226.3 | −10.1% | −8.5% | 3:1, 4:135, 5:449 |
| Python | 277.2 | 262.1 | 275.2 | 256.5 | −2.1% | −7.5% | 3:1, 4:58, 5:456 |
| SQL | 276.4 | 303.7 | 303.6 | 303.5 | −0.1% | **+9.8%** | 4:8, 5:426 |
| C=2 AIME | 164.2 | 158.6 | 164.4 | 164.7 | +3.8% | **+0.3%** | mostly 3 |
| C=2 story | 151.5 | 148.1 | 149.2 | 149.3 | +0.8% | −1.5% | mostly 3 |

AIME_15 matched frozen k=5 (192.2, more time at 5). CUDA hist still mixes 4/5
but tok/s stayed at the r6 glued-5 number (226): extra 32 rounds at k=5 changed
the token stream, accepts went cold (`s=[0.80, 0.63, 0.50, 0.37, 0.20]` vs r5
`[0.84, 0.70, 0.57, 0.48, 0.31]`). Python also colder. r5 remains the CUDA
peak; r7 is the AIME_15 peak.

## r8 try (20260830)

CUDA r7 token stream went cold because 64 rounds at k=5 precede the 5→4 drop.
r6's `s[3] > 0.35` split is right in the limit but at 32 the first slot obs is 0 or 1.

r8: each new EWMA slot starts at `p̄=0.5` then updates (not the raw Bernoulli). Dead/hot
checks unbias that origin so `drop_slot_max=0.30` / `climb_slot_min=0.35` apply to the
implied rate (raw CUDA `s[4]≈0.31` at 32 would miss dead). After 32 rounds CUDA
`μ[3]≈0.38` still last-hot, AIME `μ[3]≈0.24` is not. Waiver warmup **32**. Keep s[3]
last-hot gate. Expected: CUDA drops early like r5 (hot stream, ~252); AIME_15 stays
at 5 like r7 (~192).

Adaptive: `profiles/bench/adaptive-tk-20260830-r8/`. SQL adaptive hit `stop_token` at 1964/2048
(runner aborted; jsonl kept). C=2 ran after that skip.

| Workload | frozen | r5 | r7 | r8 | vs r5 | vs r7 | vs frozen | hist |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| dialogue | 109.4 | 108.8 | 108.7 | 110.5 | +1.6% | +1.7% | **+1.0%** | 3:580, 4:8, 5:56 |
| story | 171.3 | 168.2 | 173.0 | 172.4 | +2.5% | −0.3% | +0.6% | 3:2, 4:281, 5:103 |
| AIME_15 | 192.3 | 187.3 | 192.2 | 195.5 | +4.4% | **+1.7%** | **+1.7%** | 3:26, 4:375, 5:949 |
| AIME_30 | 205.3 | 210.1 | 215.2 | 205.8 | −2.0% | −4.4% | +0.2% | 3:46, 4:558, 5:682 |
| AIME_01 | 274.4 | 297.5 | 282.4 | 310.4 | +4.3% | **+9.9%** | **+13.1%** | 3:2, 4:8, 5:203 |
| CUDA | 247.5 | 251.7 | 226.3 | 225.8 | −10.3% | −0.2% | −8.8% | 3:1, 4:135, 5:449 |
| Python | 277.2 | 262.1 | 256.5 | 259.6 | −1.0% | +1.2% | −6.4% | 3:1, 4:77, 5:429 |
| SQL | 276.4 | 303.7 | 303.5 | 279.0* | | | | 4:29, 5:421 |
| C=2 AIME | 164.2 | 158.6 | 164.7 | 164.3 | +3.6% | −0.2% | +0.1% | mostly 3 |
| C=2 story | 151.5 | 148.1 | 149.3 | 148.1 | 0.0% | −0.8% | −2.2% | mostly 3 |

\*SQL r8 `stop_token` at 1964; r5/r7 filled 2048. CUDA hist is identical to r7 (`3:1, 4:135, 5:449`) and
`s=[0.80, 0.63, 0.50, 0.37, 0.20]` is the same cold stream. Unbiasing the 0.5 prior did not
move the 5→4 drop earlier: the first 32 CUDA rounds still have extra-of-5 above 0.30, so the
waiver waits while k=5 cools the accept curve. AIME_15 is a new peak (195.5, more time at 5
than r7). r5 remains the CUDA peak; r8 is the AIME_15 / AIME_01 peak.

## r9 try (20260830)

r8's discriminator is fine; the wait is the CUDA killer. DFlash `first_remove` is already 8,
but 5→4 used `kAdaptiveFirstRemoveWarmup=32` and `kAdaptiveK5WaiveWarmup=32`.

r9: last-hot 5→4 waiver at **8**. Score 5→4 stays at 32 (noisy-dead extra must not dump Python).
Keep 0.5 prior + unbias + `s[3]>0.35`. Kill-trio bench only (CUDA / AIME_15 / Python / AIME_01).

Next if CUDA mixes 5-heavy after an early drop: block last-hot 4→5 while extra-of-5 is still dead.
Next if extra still isn't dead at 8: recent-window extra-dead, not lifetime EWMA.

Adaptive: `profiles/bench/adaptive-tk-20260830-r9/`. Full adaptive set (SQL `stop_token` at 1964).

| Workload | frozen | r5 | r8 | r9 | vs r5 | vs r8 | vs frozen | hist |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| dialogue | 109.4 | 108.8 | 110.5 | 110.5 | +1.6% | 0.0% | **+1.0%** | 3:580, 4:8, 5:56 |
| story | 171.3 | 168.2 | 172.4 | 168.3 | +0.1% | −2.4% | −1.8% | 3:15, 4:295, 5:88 |
| AIME_15 | 192.3 | 187.3 | 195.5 | 195.3 | **+4.3%** | −0.1% | **+1.6%** | 3:25, 4:629, 5:704 |
| AIME_30 | 205.3 | 210.1 | 205.8 | 199.7 | −5.0% | −3.0% | −2.7% | 3:7, 4:551, 5:767 |
| AIME_01 | 274.4 | 297.5 | 310.4 | 305.9 | +2.8% | −1.4% | **+11.5%** | 4:16, 5:200 |
| CUDA | 247.5 | 251.7 | 225.8 | **254.9** | **+1.3%** | **+12.9%** | **+3.0%** | 3:1, 4:135, 5:382 |
| Python | 277.2 | 262.1 | 259.6 | 249.3 | −4.9% | −4.0% | −10.1% | 4:92, 5:437 |
| SQL | 276.4 | 303.7 | 279.0* | 281.8* | | | | 4:29, 5:421 |
| C=2 AIME | 164.2 | 158.6 | 164.3 | 161.6 | +1.9% | −1.6% | −1.6% | mostly 3 |
| C=2 story | 151.5 | 148.1 | 148.1 | 142.6 | −3.7% | −3.7% | −5.9% | mostly 3 |

\*SQL r8/r9 `stop_token` at 1964; r5 filled 2048. CUDA recovered (254.9, hotter `s=[0.84, 0.72, 0.60, 0.48, 0.31]`). AIME_15 held the r8 peak. Python spent more time at 4 (92 vs r5 58 / r8 77). AIME_30 and C=2 story also lost vs frozen (more k=4 / k=3).

### Extra fixtures on r9 (20260830)

`thinking_logic_grid` 2048 thinking on, `scenario_story_zh_scifi` 1024, `scenario_translation_markdown` 1024. All three logic/markdown Ks hit `stop_token` (gen length depends on K).

| Workload | frozen 3/4/5 | r9 adaptive | hist r9 |
|---|---|---:|---|
| logic-grid | 236.6 / 261.2 / **267.0*** | **294.1*** | 4:16, 5:260 |
| zh-scifi | **125.8** / 117.2 / 114.6 | 124.4 | 3:451, 4:62, 5:54 |
| markdown | 192.4 / 196.7 / **209.1*** | 187.9* | 4:8, 5:298 |

Markdown is the high-100s/low-200s peer to AIME_15. zh-scifi is dialogue-band. logic-grid is Python-band (hot extra 0.51) and adaptive stayed at 5.

## r10 try (Python extra-dead streak) — KILLED, reverted

r9 Python hole: 8-round EWMA dip below 0.30. r10 required extra dead for 8 consecutive rounds before 5→4. `profiles/bench/adaptive-tk-20260830-r10/`.

| | r9 | r10 | vs r9 |
|---|---:|---:|---:|
| Python | 249.3 | **266.1** | +6.7% |
| CUDA | **254.9** | 238.7 | −6.4% |
| AIME_15 | **195.3** | 185.8 | −4.9% |
| AIME_30 | 199.7 | **216.2** | +8.3% |
| story | 168.3 | 168.6 | +0.2% |
| zh-scifi | 124.4 | 124.5 | 0% |

Python recovered toward r5 262 / frozen 277. CUDA and AIME_15 regressed. Reverted to r9.

## Sol 5.6 after r9

r9 recovered CUDA (254.9) and kept AIME_15 (195.3). Python 249 vs frozen 277 is the remaining
C=1 hole; r10's extra-dead streak closed it and broke CUDA/AIME_15. r9 is the keep.

## r11 try (20260831)

r11: forget `s[4]` on 5→4; per-slot `s_n` for unbias; first re-observe of slot 4 inits from
`s[3]·(s[3]/s[2])`. Host tests passed. Kill trio then stopped.

`profiles/bench/adaptive-tk-20260830-r11/`

| Workload | frozen | r9 | r11 | hist r11 |
|---|---:|---:|---:|---|
| CUDA | 247.5 | **254.9** | 226.3 | 3:1, 4:156, 5:427 |
| Python | 277.2 | 249.3 | 250.2 | 4:160, 5:369 |
| AIME_15 | 192.3 | 195.3 | 195.8 | 3:13, 4:510, 5:827 |
| AIME_30 | 205.3 | 199.7 | 201.6 | 3:18, 4:554, 5:737 |
| AIME_01 | 274.4 | 305.9 | 305.0 | 4:16, 5:200 |
| dialogue | 109.4 | 110.5 | 110.1 | 3:580, 4:8, 5:56 |
| story | 171.3 | 168.3 | 161.8 | 3:45, 4:358, 5:12 |

CUDA lost r9's hot mix (226, same 5-heavy shape as r8/r10). Python did not leave 4
(160 rounds vs r9 92). AIME_15 held. Reverted to r9.

## Doubled suite (20260831)

Original 13 adaptive rows plus 13 new: remaining text scenarios (TS, jsonl, csv, zh-en, en-zh)
and C=2 of CUDA/Python/dialogue/SQL/scifi/logic/md/TS. Frozen k=3/4/5 under
`profiles/bench/adaptive-tk-20260830/`. New-set r11 under `-r11/`. Full 26-row r12 under `-r12/`.

tok/s = mean over request_done of `(completion−1)/decode`. `*` = stop_token (jsonl kept).

### New C=1 frozen 3/4/5 + r11 + r12

| Workload | k=3 | k=4 | k=5 | best | r11 | r12 | hist r12 |
|---|---:|---:|---:|---:|---:|---:|---|
| TS | 238.9 | 252.5 | **299.4*** | 5 | 298.6* | 299.0* | 5:43 |
| jsonl | 274.3 | 315.2 | **363.5** | 5 | 363.7 | 362.9 | 5:362 |
| csv | 240.2 | 324.1 | **354.9** | 5 | 340.0 | 329.6 | 4:8, 5:391 |
| zh-en | 244.7 | 208.4 | **292.1*** | 5 | 293.1* | 293.2* | 5:185 |
| en-zh | 216.2 | 244.8 | **246.7*** | 5 | 256.3* | 252.4* | 4:26, 5:150 |

### New C=2 frozen 3/4/5 + r11 + r12

| Workload | k=3 | k=4 | k=5 | best | r11 | r12 | hist r12 |
|---|---:|---:|---:|---:|---:|---:|---|
| C=2 CUDA | 183.3 | 192.2 | **194.0** | 5 | 187.1 | 178.4 | 3:1216, 4:101, 5:66 |
| C=2 Python | 195.5 | 204.0 | **218.4** | 5 | 203.0 | 202.9 | 4:1035, 5:67 |
| C=2 dialogue | **100.1** | 88.3 | 80.3 | 3 | 98.5 | 94.7 | 3:1248 |
| C=2 SQL | 211.7 | 231.4 | **232.5** | 5 | 225.1 | 226.7 | 4:944, 5:43 |
| C=2 scifi | **109.9** | 94.4 | 88.2 | 3 | 106.5 | 105.8 | 3:1112 |
| C=2 logic | 213.1 | 214.6 | **230.1*** | 5 | 221.3 | 211.8* | 4:420, 5:87 |
| C=2 md | **166.1** | 161.1 | 162.7 | 3 | 167.0 | 161.9* | 4:464, 5:63 |
| C=2 TS | 212.5 | 217.2 | **239.0*** | 5 | 217.1 | 207.6* | 4:72, 5:36 |

## r12 try (20260831)

r12: first k=5 leg is r9 (lifetime EWMA extra-dead at 8). After 5→4, refuse another 5→4
iff this-visit extra hit rate `extra_hits/rounds_at_k > climb_slot_min` (0.35). Host tests passed.

| Workload | frozen best | r9 | r11 | r12 | vs frozen | hist r12 |
|---|---:|---:|---:|---:|---:|---|
| CUDA | 247.5 | **254.9** | 226.3 | 250.6 | +1.3% | 4:65, 5:463 |
| Python | 277.2 | 249.3 | 250.2 | 254.6 | −8.1% | 4:68, 5:449 |
| AIME_15 | 192.3 | **195.3** | 195.8 | 188.7 | −1.9% | 4:674, 5:707 |
| AIME_30 | 205.3 | 199.7 | 201.6 | **219.6** | +7.0% | 4:170, 5:1027 |
| AIME_01 | 274.4 | 305.9 | 305.0 | **306.1** | +11.6% | 4:16, 5:200 |
| dialogue | 109.4 | 110.5 | 110.1 | 110.5 | +0.9% | 3:580, 4:8, 5:56 |
| story | 171.3 | 168.3 | 161.8 | 168.2 | −1.8% | 4:295, 5:88 |
| SQL | 276.4 | 281.8* | | 281.0* | +1.7% | 4:29, 5:421 |
| logic | 267.0* | **294.1*** | | 276.8* | +3.7% | 4:8, 5:369 |
| zh-scifi | 125.8 | 124.4 | | 124.5 | −1.0% | 3:451 |
| markdown | 209.1* | 187.9* | | 188.0* | −10.1% | 4:8, 5:298 |
| C=2 AIME | 164.2 | 161.6 | | 164.2 | 0% | mostly 3 |
| C=2 story | 151.5 | 142.6 | | 148.2 | −2.2% | mostly 3 |

CUDA held the r9 band (not r11's 226). Python 4-rounds 68 vs r9 92 — small lift, hole remains.
AIME_15 188.7 is below r9 195.3.

## r9 vs r12 full 26-row suite (20260831)

Kill-trio is not the keep rule. r9 was run on the 13 new rows so both sit on the same 26 jobs.
tok/s = mean request_done `(completion−1)/decode`. `*` = stop_token.

| Workload | frozen best | r9 | r12 | r12/r9 |
|---|---:|---:|---:|---:|
| dialogue | 109.4 | 110.5 | 110.5 | 1.000 |
| story | 171.3 | 168.3 | 168.2 | 0.999 |
| AIME_15 | 192.3 | **195.3** | 188.7 | 0.967 |
| AIME_30 | 205.3 | 199.7 | **219.6** | 1.100 |
| AIME_01 | 274.4 | 305.9 | 306.1 | 1.001 |
| CUDA | 247.5 | **254.9** | 250.6 | 0.983 |
| Python | 277.2 | 249.3 | **254.6** | 1.021 |
| SQL | 276.4 | 281.8 | 281.0* | 0.997 |
| logic | 267.0 | **294.1** | 276.8* | 0.941 |
| zh-scifi | 125.8 | 124.4 | 124.5 | 1.001 |
| markdown | 209.1 | 187.9 | 188.0* | 1.000 |
| C=2 AIME | 164.2 | 161.6 | **164.2** | 1.016 |
| C=2 story | 151.5 | 142.6 | **148.2** | 1.039 |
| TS | 299.4 | 298.4 | 299.0* | 1.002 |
| jsonl | 363.5 | 363.5 | 362.9 | 0.998 |
| csv | 354.9 | 330.6 | 329.6 | 0.997 |
| zh-en | 292.1 | 293.3 | 293.2* | 1.000 |
| en-zh | 246.7 | 252.3 | 252.4* | 1.000 |
| C=2 CUDA | 194.0 | 182.2 | 178.4 | 0.979 |
| C=2 Python | 218.4 | 203.5 | 202.9 | 0.997 |
| C=2 dialogue | 100.1 | 96.7 | 94.7 | 0.980 |
| C=2 SQL | 232.5 | 221.8 | 226.7 | 1.022 |
| C=2 scifi | 109.9 | 104.6 | 105.8 | 1.012 |
| C=2 logic | 230.1 | 212.3 | 211.8* | 0.998 |
| C=2 md | 166.1 | 161.8 | 161.9* | 1.001 |
| C=2 TS | 239.0 | **232.5*** | 207.6* | 0.893 |

Aggregators: mean tok/s r9 **216.5** vs r12 215.7 (−0.4%). Geomean −0.3%. Mean vs frozen 98.2% vs 97.9%. Pairwise (>0.5%): 6–6–14 ties.

No hung decode. Only ≥10% pairwise gap is C=2 TS (both `stop_token` at 160–356 tokens). **r9 is the keep** on average.

## C>1 mixed draft-size campaign (20260831)

C≥2 decode uses one batch K: `adaptive_batch_k_sum_score` = argmax_K Σ_row E_row(K) / T_B(K),
K ≤ max(row live_k), ties keep lower K. Per-row live_k still exists; the round executes at
that one K. B=1 does not call sum-score.

tok/s = mean request_done `(completion−1)/decode`. Mixed same-budget jobs have
`avg_decode_batch` 1.3–2.0 (fast slot finishes first). Overlap reconstruction
(tokens × min(rounds)/rounds over first-finisher wall) **agrees with e2e on every
winner**, so contamination does not change the batch-K pick. `*` = stop_token.

Raw frozen: `profiles/bench/adaptive-tk-20260830-mixed/`.
Adaptive: `profiles/bench/adaptive-tk-20260830-mixed-adapt/`.

### How draft size is picked at C>1

1. Each row still has `live_k` from the r9 C=1 policy (or frozen N).
2. `row_ks[row] = min(live_k, budget, capacity)`.
3. Candidate K is captured ∩ [k_min, max(row_k)].
4. Batch K = argmax_K Σ_row E_row(K) / T_B(K). E_row is the EWMA chain on that
   slot. T_B is **batch-size dependent**, not the C=1 table.

min(want) / max(want) / majority are not the rule. Frozen mixed winners:

| mix | C=1 wants | emp K | min | max | old C2T T4=0.89 | mixE + measured T_B |
|---|---|---:|---:|---:|---:|---:|
| dlg+python | 3,5 | **5** | 3 | 5 | 4 | 5 |
| scifi+jsonl | 3,5 | **5** | 3 | 5 | 5 | 5 |
| dlg+story | 3,4 | **3** | 3 | 4 | 4 | 3 |
| cuda+python | 4,5 | **5** | 4 | 5 | 4 | 5 |
| sql+jsonl | 4,5 | **5** | 4 | 5 | 4 | 5 |
| story+jsonl | 4,5 | **5** | 4 | 5 | 4 | 5 |
| dlg+cuda | 3,4 | **3** | 3 | 4 | 4 | 3 |
| logic+dlg | 5,3 | **4** | 3 | 5 | 4 | 4 |
| md+python | 5,5 | **5** | 5 | 5 | 5 | 5 |
| mix3 dlg+story+py | 3,4,5 | **3** | 3 | 5 | 4 | 3 |
| mix3 scifi+cuda+jsonl | 3,4,5 | **5** | 3 | 5 | 4 | 5 |

Old table T[4]=0.89 over-picks k=4 (1/7 of the original C=2 mismatches).
ΣE/T with live mixed E and measured T_B matches **all** empirical winners,
including logic+dlg where the winner is the middle K.

C=1-E (not live mixed E) + measured C=2 T still misses cuda+py and dlg+cuda —
CUDA's C=1 E(4) vs E(5) does not match C=2 mixed E. Adaptive uses live EWMA, not
C=1 frozen E.

### Frozen mixed tok/s (e2e mean)

| job | k=3 | k=4 | k=5 | best | avgB k=3 |
|---|---:|---:|---:|---:|---:|
| mix2-dlg-py | 141.2 | 139.6 | **147.6** | 5 | 1.56 |
| mix2-sci-jsonl | 180.7 | 184.0 | **196.9** | 5 | 1.46 |
| mix2-dlg-story | **124.7** | 115.7 | 109.4 | 3 | 1.70 |
| mix2-cuda-py | 194.6 | 204.1 | **210.7** | 5 | 1.97 |
| mix2-sql-jsonl | 223.3 | 250.3 | **266.0** | 5 | 1.85 |
| mix2-story-jsonl | 197.4 | 210.4 | **220.8** | 5 | 1.60 |
| mix2-dlg-cuda | **136.7** | 134.1 | 131.9 | 3 | 1.61 |
| mix2-logic-dlg | 160.9 | **164.4** | 158.6 | 4 | 1.45 |
| mix2-md-py | 174.8* | 174.1 | **182.6*** | 5 | — |
| mix3-dlg-story-py | **125.4** | 118.6 | 119.3 | 3 | 2.18 |
| mix3-sci-cuda-jsonl | 156.2 | 160.0 | **168.6** | 5 | 2.08 |

### Homogeneous T_B(K) = ms(K)/ms(5)

C=2 (existing `adaptive-tk-20260830/c2-*-dflash/`): T[3]=0.82–0.83, T[4]=0.91–0.93.
Table was T[4]=0.89; updated to **0.93**.

| homo | C | T[3] | T[4] | tok/s 3/4/5 | best |
|---|---:|---:|---:|---|---:|
| dialogue | 3 | 0.85 | 0.97 | 83.9 / 74.4 / 70.2 | 3 |
| python | 3 | 0.85 | 0.98 | 170.3 / 172.4 / 191.0 | 5 |
| jsonl | 3 | 0.85 | 0.98 | 197.1 / 215.7 / 251.3 | 5 |
| story | 3 | 0.86 | 0.98 | 124.7 / 115.9 / 117.2 | 3 |
| cuda | 3 | 0.86 | 0.98 | 160.7 / 153.8 / 158.0 | 3 |
| dialogue | 4 | 0.85 | 0.94 | 72.4 / 66.0 / 63.4 | 3 |
| python | 4 | 0.85 | 0.95 | 149.5 / 149.1 / 160.3 | 5 |
| dialogue | 8 | 0.82 | 0.93 | 49.1 / 44.1 / 41.1 | 3 |

T[4] is **not** monotonic in B: C=2 0.93, C=3 **0.98**, C=4 0.94, C=8 0.93.
At C=3 k=4 is almost as expensive as k=5, so C=3 cuda/story/dialogue want k=3
even when C=1 wanted 4.

Shipped tables (DFlash, index=K):

```text
C=1  T[3]=0.91  T[4]=0.97  T[5]=1.00     # unchanged
C=2  T[3]=0.83  T[4]=0.93  T[5]=1.00     # was T[4]=0.89
C=3  T[3]=0.86  T[4]=0.98  T[5]=1.00
C≥4  T[3]=0.85  T[4]=0.94  T[5]=1.00     # C=4 measured; C=8 close
```

`adaptive_dflash_round_time(B)` selects C=2 / C=3 / C≥4. MTP C>1 still uses
`kAdaptiveMtpC2T` (not in this DFlash campaign).

Until every row has `observed < 8`, sum-score executes max(row_k) instead of
argmin T (unobserved p̄=0.5 otherwise sits at k=3 and never drafts extras).

### Confirm suite (2× mixed, 20260831)

11 new frozen k=3/4/5 mixes. mixE + measured T_B still **11/11**. Combined **22/22**.
min(want) 9/22, max(want) 12/22, old T[4]=0.89 8/22, C=1-E + T_B 14/22.

| job | wants | emp | min | max | mixE | k=3 | k=4 | k=5 | vs min | vs max |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| mix2-dlg-jsonl | 3,5 | **5** | 3 | 5 | 5 | 172.0 | 181.4 | **193.0** | +12.2% | 0 |
| mix2-sci-py | 3,5 | **3** | 3 | 5 | 3 | **152.5** | 148.4 | 142.5 | 0 | +7.0% |
| mix2-csv-scifi | 5,3 | **4** | 3 | 5 | 4 | 171.9 | **174.1** | 164.4 | +1.2% | +5.9% |
| mix2-sci-story | 3,4 | **3** | 3 | 4 | 3 | **131.9** | 117.9 | 113.0 | 0 | +11.9% |
| mix2-sci-cuda | 3,4 | **3** | 3 | 4 | 3 | **149.2** | 142.7 | 144.9 | 0 | +4.5% |
| mix2-dlg-sql | 3,4 | **3** | 3 | 4 | 3 | **160.3** | 158.9 | 153.1 | 0 | +0.9% |
| mix2-story-py | 4,5 | **5** | 4 | 5 | 5 | 165.7 | 164.9 | **170.7** | +3.5% | 0 |
| mix2-cuda-jsonl | 4,5 | **5** | 4 | 5 | 5 | 215.2 | 236.4 | **248.2** | +5.0% | 0 |
| mix2-sql-py | 4,5 | **5** | 4 | 5 | 5 | 199.7 | 208.7 | **223.4** | +7.1% | 0 |
| mix3-dlg-sql-jsonl | 3,4,5 | **5** | 3 | 5 | 5 | 158.3 | 161.8 | **183.7** | +16.1% | 0 |
| mix3-sci-story-py | 3,4,5 | **3** | 3 | 5 | 3 | **129.4** | 117.2 | 124.0 | 0 | +4.3% |

sci+python is 3-vs-5 that wants **k=3** (dlg+python wanted 5). csv+scifi wants **middle k=4**, second such case after logic+dialogue. ΣE/T_B with live mixed E is the rule that hits both.

Raw: same `profiles/bench/adaptive-tk-20260830-mixed/`.

### Adaptive mixed vs frozen best (keep: T_B + warmup-8 max-K)

r9 per-row live_k, DFlash N=7 live {3,4,5}. tok/s mean request_done.

| job | frozen best K | frozen | adaptive | vs best | executed 3/4/5 |
|---|---:|---:|---:|---:|---|
| mix2-dlg-py | 5 | 147.6 | 146.5 | 0.993 | 732 / 0 / 216 |
| mix2-sci-jsonl | 5 | 196.9 | 197.3 | 1.002 | 407 / 8 / 362 |
| mix2-dlg-story | 3 | 124.7 | 124.5 | 0.999 | 988 / 9 / 73 |
| mix2-cuda-py | 5 | 210.7 | 195.0 | **0.926** | 1168 / 9 / 104 |
| mix2-sql-jsonl | 5 | 266.0 | 265.7 | 0.999 | 2 / 0 / 813 |
| mix2-story-jsonl | 5 | 220.8 | 222.0 | 1.005 | 10 / 202 / 356 |
| mix2-dlg-cuda | 3 | 136.7 | 141.5 | 1.035 | 861 / 8 / 98 |
| mix2-logic-dlg | 4 | 164.4 | 160.6 | 0.977 | 726 / 0 / 198 |
| mix2-md-py | 5 | 182.6* | 188.0* | 1.030 | 380 / 14 / 141 |
| mix3-dlg-story-py | 3 | 125.4 | 124.8 | 0.995 | 1239 / 17 / 138 |
| mix3-sci-cuda-jsonl | 5 | 168.6 | 169.0 | 1.002 | 570 / 16 / 530 |

Mean vs frozen-best **0.997**, geomean 0.996. 9/11 within 2%.

cuda+python is the leftover: same C=2 high-accept undershoot as homogeneous
r9 (C=2 CUDA 182 vs frozen 194, C=2 Python 204 vs 218). After warmup the batch
still jumps to k=3 and extras freeze. logic+dlg is −2.3% (emp is middle k=4;
hist is 3 and 5, skipping 4).

### Try log

1. Frozen mixed k=3/4/5 on 7 C=2 pairs + 2 C=3 + 5 homo C=3. Then extra
   logic+dlg, md+py, homo C=4/C=8 dialogue, C=4 python.
2. Pickers: min/max/majority lose. Old T[4]=0.89 over-picks 4 (1/7 C=2).
   mixE + measured T_B is 11/11 including logic+dlg at middle k=4.
3. Shipped T_B: C=2 T[4] 0.89→0.93; new C=3 (0.86/0.98/1) and C≥4 (0.85/0.94/1).
   `program_impl.h` DFlash sum-score uses `adaptive_dflash_round_time(B)`.
4. Adaptive try A (T_B only, unobserved picks k=3): cuda-py −11% (hist all 3).
5. Adaptive try B (warmup-8 at max(row_k) + T_B): **keep**. Mean 0.997.
   cuda-py −7.4%.
6. Neighbor-of-max(row_k): dlg-story stuck at 4 (−5.4%); cuda-py stuck at 4
   (−7.9%). Reverted.
7. Hot-floor on live extra: cuda-py −11.6%. Reverted.
8. Warmup-32 + uncap + hot extra at any K: dlg-story/logic/mix3 `stop_token`
   aborts; cuda-py still −8%. Reverted to try B.

Do not retune r9 C=1 5→4 knobs for the cuda-py hole.

## Commands

```bash
python3 tools/bench/run_adaptive_draft_sweep.py
python3 tools/bench/run_adaptive_draft_mixed.py
python3 tools/bench/run_adaptive_draft_mixed.py adaptive
```

Existing C=1 AIME MTP 3/4/5 and DFlash 4/5/7 plus DFlash story were gathered before this driver
and live in `aime-mtp/`, `aime-dflash/`, `story-dflash/`.
