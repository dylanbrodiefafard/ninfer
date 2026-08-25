# Perplexity runner

Teacher-forced next-token NLL for `qwen3.8-27b/nvfp4` across KV codecs. Generate and
serve are unchanged: this tool calls `Engine::score` only.

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
