# Perplexity campaign: Qwen3.8-27B NVFP4

One artifact: `qwen3.8-27b/nvfp4` at

`/ssdpool2nvme/local_llm/models/qwen3.8-nvfp4-Osfralla-mtp-ninfer/qwen3_8_27b_nvfp4.ninfer`

Scoring is `Engine::score` only. Generate and serve stay on their production graphs.

## What this is for

Mean PPL is the **integration** test for KV codecs and attention rows on a real
prompt. It is not an Op oracle. Sparse kernel bugs hide in the mean; those stay
in `tests/ops/test_gqa_attention.cpp` and `tests/ops/test_nll_from_logits.cpp`.

Codec Δ is INT8/NVFP4 vs BF16 **inside one schedule**. BF16 prefill vs BF16
decode is stack parity (Prompt vs SmallT plus GDN association), not a codec
result.

## Sweep

Default `python3 tools/ppl/run.py`:

1. **8k then 32k**, both `prefill` and `decode`, schemes `kv-bf16` → `kv-int8` →
   `kv-nvfp4`.
2. **CUDA graphs on** (Engine production default). `ninfer-ppl` used to force
   them off; that is no longer the case.
3. The decode lane runs under **MTP T=1 target-verify** by default
   (`--spec mtp --draft-tokens 3`, the production serve spec); `--no-mtp`
   reverts to spec-free decode. 8k extras (review): BF16 decode **mid-page
   skip 4128** (page size 64; default half-skip 4096 is page-aligned), BF16
   decode **`--skip 1`** on 257 tokens, BF16 decode **graphs off** vs the
   graphs-on 8k cell, and a **draft-4 probe** (`--spec mtp --draft-tokens 4`)
   unless the default draft length is already 4.
4. Sidecar `{cell}.nllf32` plus `max_nll` / `terrible_tokens` (nll ≥ 10,
   p ≤ ~4.5e-5). BF16 prefill vs decode reports mean/max `|Δnll|`.

`--tokens N` runs only that length. `--long` is 32k only. `--no-extras` /
`--no-mtp` drop the 8k extra cells.

MTP score is **not** a draft-accept-rate test: after the prefix prefill, drafts
are cleared so every scored step is T=1 target-verify through the MTP decode
graph. That still loads MTP weights and writes MTP KV with the selected cache
dtype. DFlash score is rejected.

## Oracles (ctest)

| Layer | Test | What it proves |
|---|---|---|
| NLL Op | `ninfer_nll_from_logits_test` | production vocab `[248320,248077]`, T=1 and T=64, vs FP64 logsumexp |
| GQA | `ninfer_gqa_attention_test` | BF16/INT8/NVFP4, both geometries, T=1 at page ±1 (`63/64/65`) |
| Score index | `ninfer_qwen3_6_score_index_test` | skip, chunk targets, terrible-token recording |

Do not put 8k/32k, WikiText fetch, or codec Δ gates in ctest. A new attention
kernel is admitted by extending the GQA oracle first, then adding a `schemes.py`
row at the same `kv_dtype` and schedule.

## Gates

Optional (`--gate kv-int8=…`) until the first BF16 8k cell exists. Without a
gate, a cell fails only on non-finite NLL (including the BF16 baseline).
`tokens_scored` must match across codecs in the same (length, schedule, spec)
group.

## Not in this matrix

Vision, C>1, prefix reuse, ChunkedSmallT, T=2..6 SmallT, sampled generate,
DFlash, full MTP draft-window accept. GDN can dilute a pure attention error;
that is why Op oracles exist.
