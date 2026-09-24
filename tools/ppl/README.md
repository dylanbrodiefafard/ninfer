# Perplexity runner

Teacher-forced next-token NLL for `qwen3.8-27b/nvfp4` across KV codecs. The KV-codec
runner `run.py` calls `Engine::score`. The separate verifier qualification tools described
below score the actual DFlash Verify/replay path.

The formula is always `mean_nll = sum_t nll(token_{t+1} | prefix)` over the scored
suffix, `ppl = exp(mean_nll)`. Two schedules select the kernel that produces those
logits:

| `--schedule` | What runs | GQA route |
|---|---|---|
| `prefill` (default) | chunked prompt forward of the full sequence | `prompt` (`T>6`) |
| `decode` | prefill the warmup prefix, then teacher-force the suffix at T=1 | `small_t` |

Default `--skip half` drops the first `n/2` positions from the mean (llama.cpp
second-half): those tokens still fill KV, but early positions barely use the cache
and dilute codec/kernel Δ. `--skip 0` scores every next-token except the last id.
Codec gates compare INT8/NVFP4 to BF16 **inside the same schedule**.

CUDA graphs default **on**, matching production decode. `--no-cuda-graph` is the
eager pair. Each cell writes `max_nll`, `terrible_tokens` (nll ≥ 10), and a
`{cell}.nllf32` sidecar so mean PPL cannot hide a handful of exploding tokens.

Decode-lane cells run under the production MTP spec by default: `--spec mtp
--draft-tokens 3` loads MTP and scores T=1 target-verify (drafts cleared after
the prefix). It is not a draft-accept-rate test. `--no-mtp` (or `--spec none`)
reverts to spec-free decode; `--draft-tokens N` overrides the draft length.
Prefill-lane cells are always spec-free for MTP. `--spec dflash` is admitted on
`--schedule prefill` only (target teacher-force with DFlash feature capture
loaded). DFlash decode score is rejected.

## Run

Build `ninfer-ppl`, then score. Default is 8k then 32k, both schedules, graphs on,
plus 8k extras (mid-page skip, short-context decode, graphs-off, MTP × KV dtypes).
BF16 KV always runs first inside each (length, schedule, spec) group.

```bash
cmake --build build --parallel --target ninfer-ppl

python3 tools/ppl/run.py \
  --weights /ssdpool2nvme/local_llm/models/qwen3.8-nvfp4-Osfralla-mtp-ninfer/qwen3_8_27b_nvfp4.ninfer
```

First run bakes WikiText-2 with `ninfer-ppl --encode` (the tokenizer inside the
`.ninfer` file). The script prints the output directory, which contains
`results.json` and `results.md`. `--tokens N` scores only that length; `--long`
is 32k only. `--no-extras` drops the 8k extra cells; `--no-mtp` runs the decode
lane spec-free (legacy behavior) and skips the MTP extras.

Decode is much slower (one T=1 step per scored token):

```bash
python3 tools/ppl/run.py --schedule decode
python3 tools/ppl/run.py --schedule prefill,decode --no-extras
```

Gates are optional until a BF16 baseline exists:

```bash
python3 tools/ppl/run.py --gate kv-int8=0.02 --gate kv-nvfp4=0.05
```

Without `--gate`, deltas are reported and the process still exits 0 unless a cell has
non-finite NLL. A non-finite BF16 baseline also fails.

## Noise columns

Each cell's `{cell}.nllf32` sidecar (per-token NLLs, float32 LE, written by
`ninfer-ppl` next to the cell JSON) feeds the noise stats in `results.json` /
`results.md`:

- `nll_std` / `nll_se` — per-token NLL std for the cell and its SE (std/√n).
- `delta_nll_se` — SE of the per-token **paired** Δnll vs the group's bf16
  baseline (same corpus positions, index-aligned; the paired mean equals the
  `Δ mean_nll` column exactly).
- `in_noise` — |Δ| ≤ 2·Δ1σ: the delta is not resolved above the per-token
  noise floor (rendered as `noise: yes` in the table).

The paired SE is far tighter than the SE of the two independent means — what
makes "wash or real?" answerable at the 1e-3 level. Keep the `.nllf32`
sidecars when copying results elsewhere; without them the noise columns are
blank.

## Add an attention scheme

When a new causal attention implementation is selectable on this identity:

1. Extend `tests/ops/test_gqa_attention.cpp` against the same oracle. PPL is not
   admission for a new attention kernel.
2. Add the Engine/CLI flag that selects it (`ninfer-ppl` plus `EngineOptions` if needed).
3. Register it in `tools/ppl/schemes.py`:

```python
SCHEMES["attn-flash"] = Scheme("attn-flash", "bf16", extra_args=("--attention", "flash"))
ORDER = ORDER + ("attn-flash",)
```

4. Keep `kv-bf16` as `BASELINE`. Compare the new scheme to the previous kernel at
   the **same** `kv_dtype` **and** schedule.
5. Do not add empty MLA/sparse rows. A rename of `gqa_attention` is not a new scheme.
6. Run both `--schedule prefill` and `--schedule decode` if the change can affect either
   route.

## DFlash verifier arithmetic qualification

`verify_campaign.py` runs the test-private `ninfer_qwen3_8_27b_verify_score_real_test`
profiles sequentially in `ninfer-builder-dylan`. It requires the explicitly named local
DFlash2 artifact and separately preserved A/B/C/D executables; it does not build profiles,
download text, or call `Engine::score`. Profile definitions and qualification results are in
the [performance reference](../../docs/performance.md#dflash-a4-verification-qualification-2026-09-23);
the selected A8/A8 default is documented in its preceding A8 section.
For controlled A/B/C builds, set `kNvfp4VerifyPolicy` / `kNvfp4GdnVerifyPolicy` in
`src/targets/qwen3_6_27b/impl/variant.cpp` to `A16Only` / `A16Only`, `AllowA4` / `A16Only`,
or `AllowA4` / `AllowA4`, respectively, then build and preserve the scorer executable under
the exact names used by `verify_campaign.py`. Restore both policies to `AllowA8` afterward.
Leave `kNvfp4TextPolicy` unchanged. D uses a preserved historical BF16-current-GDN diagnostic
binary; its superseded arithmetic is not a production selection.

```bash
/home/battlefront/.local/bin/python3.11 -m tools.ppl.verify_campaign matrix
/home/battlefront/.local/bin/python3.11 -m tools.ppl.verify_campaign cumulative
/home/battlefront/.local/bin/python3.11 -m tools.ppl.verify_campaign checkpoints --profiles A B C D
/home/battlefront/.local/bin/python3.11 -m tools.ppl.verify_campaign rounding --profiles D
/home/battlefront/.local/bin/python3.11 -m tools.ppl.verify_campaign fixtures
/home/battlefront/.local/bin/python3.11 -m tools.ppl.verify_campaign mixed --profiles C
/home/battlefront/.local/bin/python3.11 -m tools.ppl.verify_compare \
  --campaign-dir profiles/bench/dflash-a4-qualification --brief
```

Each successful cell records its command and writes per-token TSV plus stderr. Logit
captures contain four little-endian u32 fields (domain, lane, position, gold token), followed
by `domain` BF16 logits, repeated per checkpoint. They cover the full 248077-token logical
domain, excluding padded output-head rows. Scoring uses raw temperature-1 distributions and
NVFP4 KV; a common checkpoint is a fresh identical prefill followed by one verification
block, whereas a cumulative run retains each profile's evolving state.

`verify_compare.py baseline.tsv candidate.tsv --replicated-lanes` requires exact equality
of replicated lanes and counts lane zero only. The campaign's mixed-history checks instead
use stable document-offset IDs and compare each distinct history with its C=1 counterpart,
including reversed physical-lane order. `--logits` compares two binary captures using FP64
normalization, KL, total variation, gold probability, and top-token margins.

Reports use token-weighted NLL/PPL, paired deltas, and a fixed-seed contiguous-block bootstrap
(1024 tokens per block, 10000 draws). The intervals are conditional on this corpus and block
choice, not population-level confidence across domains. Missing cells and geometry
mismatches are explicit. A report is not an acceptance gate: no acceptable A4 PPL budget is
encoded. Curated fixture paragraphs are used once each, not tiled, and are diagnostics
rather than an additional held-out dataset.

For additional fixed local documents, supply a JSON list of objects with `name`, `domain`,
and an absolute UTF-8 source `path`. Names contain lowercase letters, digits, `-` or `_`.
The runner copies source text into the output directory, resets the model for each document,
and scores up to 8192 tokens after the eight-token prefill and supplied gold anchor. Keep
the output directory inside the checkout so the builder can read its `/src` counterpart.

```bash
/home/battlefront/.local/bin/python3.11 -m tools.ppl.verify_campaign documents \
  --documents profiles/bench/dflash-a4-qualification/local-documents/selection.json \
  --out profiles/bench/dflash-a4-qualification/local-documents
/home/battlefront/.local/bin/python3.11 -m tools.ppl.verify_compare \
  --documents-dir profiles/bench/dflash-a4-qualification/local-documents
```

The document report preserves distinct document IDs, checks matched gold tokens, and reports
individual documents plus token-weighted domain results. Bootstrap blocks and context trends
never cross document resets. Record source provenance and distinguish reference source code,
prompt-conditioned solutions, and problem-input text: their likelihoods support different
claims, and none alone establishes executable-test or solve accuracy.

Test-private partition diagnostics accept `NINFER_VERIFY_COMMIT_LIMIT=1` and
`NINFER_VERIFY_BF16_KV=1`; ordinary corpus scoring continues to default to NVFP4 KV.
`NINFER_VERIFY_RECORD_DUMP=<path>` captures replay inputs for C1 as little-endian u32
`{absolute_input_position, gdn_layer, plane, byte_count}` followed by raw plane bytes:
conv/key/value are BF16, gate is FP32. These controls localize state/partition differences.
Changing commit boundaries can change private attention partitions even with identical
represented inputs; current K/V are already quantized by the NVFP4 SmallT consumer.

`NINFER_VERIFY_DECODE_CHECK=greedy|p-less|stochastic` selects production decoding instead of
teacher forcing, using the supplied prefix as the prompt. Give a token limit of at least
`16*W+1` for its 16 fixed-chain rounds. It checks candidate-logit licensing and commit
invariants; see [tests](../../tests/README.md) for the precise scope. It emits no corpus PPL.
