# Deferred: DFlash2 frozen k=5 C=3 ≠ C=1 greedy

**Ignore for now.** Frozen overlapping C=3 Graph DFlash2 at k=5 does not match C=1 of
the same k on Qwen3.8-27B NVFP4. This is a pre-existing frozen-path identity failure,
not an adaptive-draft regression. Do not block adaptive N=7 on it. Do not “fix” it by
rewriting the product contract unless a later task explicitly owns that change.

Identity: Qwen3.8-27B NVFP4 DFlash2, RTX 5090, `sm_120a`.
Artifact: `NINFER_QWEN3_8_27B_NVFP4_DFLASH_WEIGHTS` →
`/ssdpool2nvme/local_llm/models/qwen3.8-nvfp4-flash2-nvfp4-bf16codebook-from-bf16/qwen3_8_27b_nvfp4_dflash_nvfp4.ninfer`.

## Contract vs reality

`docs/maintainer/qwen3.8-27b-artifact.md` and commit `d8cb420f` (`feat(dflash2): nvfp4
matrices with bf16 codebooks, tree verify, and C=3 identity`) require overlapping C=3
Graph greedy to match C=1 DFlash of the **same** k for product **k=4 and k=5** chain
(T=5 / T=6 SmallT). That is not a requirement to match MTP k=3 or ordinary T=1 A16 decode.

| Case | C=3 vs C=1 same k |
|---|---|
| Frozen DFlash k=4 chain (T=5, W=5) | matches |
| Frozen DFlash k=5 chain (T=6, W=6) | **diverges** (HEAD and adaptive tree, same tokens) |
| Adaptive N=7 compact k=5 under `W_ceil=12` | matches (different launch: pack/scatter into W=12 pool) |

## Observed tokens (2026-08-27)

`ninfer_qwen3_8_27b_dflash_real_test` prompts A/B/C (the three 16-token “write a story”
fixtures). Overlapping submit, greedy lengths 19 / 13 / 7.

- Request A first diverge at generated index 14: C=3 `123066` vs C=1 `96719`.
- Request C first diverge at generated index 1: C=3 `96003` vs C=1 `95826`.
- Per-position / round counts were the same; this is token identity, not a hang.

Stashing all adaptive work and rebuilding HEAD reproduced the **same** k=5 mismatch.
Frozen k=5 never takes compact/pack (`k==N`). `gdn_mix` pack_replay only fires when
`width != replay_records.spec.width`; frozen k=5 has matching width.

## Current test stance

`tests/targets/qwen3_6_27b/test_engine_dflash_real.cpp`:

- k=4 C=3 still asserts token match vs C=1.
- k=5 C=3 asserts **completion + speculative stats only** (`match_c1=false`), with a
  comment that HEAD already diverges.

That weakening is a park, not a new contract. When this is fixed, restore
`run_k(5, "DFlash2 k=5 chain C=3", true)` and keep the artifact doc’s k=4/k=5 wording.

## Likely locus when resumed

Packed C>1 verify is supposed to pin NVFP4 Linear/LinearAdd/SwiGLU to the C=1 width so
aggregate `T=W×B` does not cross a W4A4 threshold C=1 does not (`packed_route_tokens`,
`pinned_route_policy` in the 27B variant). k=4 C=3 holding and k=5 failing points at
**T=6 SmallT** (one GQA tile, chain W=6) under overlapping C=3: GQA / GDN / valid-column
masking, not adaptive live K.

Do not confuse with:

- Adaptive compact k=5 under N=7 (`W_ceil=12`) — already C=1-identical in the live test.
- MTP NVFP4 k=5 C=3 — separate path; that test still requires C=1 match.
- W4A4 vs A16 at T=1 decode — explicitly out of the DFlash C=3 contract.

## Reproduce

```bash
NINFER_QWEN3_8_27B_NVFP4_DFLASH_WEIGHTS=/ssdpool2nvme/local_llm/models/qwen3.8-nvfp4-flash2-nvfp4-bf16codebook-from-bf16/qwen3_8_27b_nvfp4_dflash_nvfp4.ninfer \
  ./build/tests/ninfer_qwen3_8_27b_dflash_real_test
```

To re-assert the broken contract: temporarily pass `true` for k=5 `match_c1` (or run the
same overlapping C=3 vs sequential C=1 oracles on a frozen `--draft-tokens 5` Engine).
