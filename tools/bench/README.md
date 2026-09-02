# tools/bench

Offline helper for the `ninfer_bench` throughput tool. Correctness/parity tooling lives separately
under [`tools/parity`](../parity).

## Corpus baker

`ninfer_bench` benchmarks prefill at an exact length by slicing the first `P` token ids of a
committed corpus, so the corpus must be real, in-distribution text (not random tokens) and at
least as long as the largest prefill you want to run. `make_bench_corpus.py` bakes that corpus
offline with a local Hugging Face Qwen3.6 tokenizer.

Outputs (committed):

```text
bench/fixtures/bench_corpus.ids            whitespace-separated decimal token ids (exactly --tokens)
bench/fixtures/bench_corpus.manifest.json  tokenizer id, token count, and source description
```

Content sources:

- Built-in curated multi-domain prose (Chinese / English / code / math) — the default. It is
  encoded WITHOUT the chat template or special tokens, then tiled (paragraphs rotated each cycle)
  and truncated to exactly `--tokens`. Repetition only fills length; because prefill/decode
  throughput is token-count / bandwidth bound, it does not bias the numbers.
- `--source-text <file>` (repeatable) — tokenize your own long meaningful text instead, e.g. a
  downloaded public-domain book or a concatenated document set, for genuinely diverse very long
  content. The committed default is `~64k` tokens; raise `--tokens` and/or pass `--source-text`
  for more.

The binary slices `[0:P]`; the manifest is provenance only.

## Requirements

Install the tokenizer dependencies into the active Python environment:

```bash
pip install -r tools/bench/requirements.txt
```

The tokenizer is loaded locally only; the tool never downloads from the network. Pass
`--tokenizer-path` or set `NINFER_TOKENIZER_PATH`.

## Regenerate / check

```bash
# Regenerate the committed corpus from the built-in bank (default 65536 tokens).
python3 tools/bench/make_bench_corpus.py \
  --tokenizer-path /path/to/local/Qwen3.6-27B/tokenizer \
  --tokens 65536

# Bake from your own downloaded/assembled text instead (kept local; not committed).
python3 tools/bench/make_bench_corpus.py \
  --tokenizer-path /path/to/local/Qwen3.6-27B/tokenizer \
  --tokens 131072 --source-text /path/to/book.txt

# Check that the committed .ids and its descriptive manifest agree; no tokenizer or source needed.
python3 tools/bench/make_bench_corpus.py --check
```

`--tokens` is the exact committed corpus size and the ceiling on prefill length; increase it (and
optionally use `--source-text`) to benchmark longer prefills, memory permitting.

## NInfer performance matrix

`run_ninfer_bench_matrix.py` runs the layered public-Engine `ninfer_bench` matrix against the native
`.ninfer` artifact and stores its local reports under `profiles/bench/`. Its defaults are:

```text
artifact: out/qwen3_6_27b.ninfer
binary:   build/bench/ninfer_bench
corpus:   bench/fixtures/bench_corpus.ids
```

The matrix treats MTP `k=3` with the optimized proposal head as the primary path, keeps `k=0` and
`k=5` as controls, and sweeps `k=0..5` on representative context-decode cases. Decode-bearing cases
cover CUDA Graph and eager execution; prefill-only cases vary prompt length and prefill chunk.

```bash
# Configure the benchmark targets once; they are off in the default public build.
cmake -S . -B build -DNINFER_BUILD_BENCHMARKS=ON

# Inspect commands without running the model.
python3 tools/bench/run_ninfer_bench_matrix.py --preset core --dry-run

# Main run. Builds build/bench/ninfer_bench first, then writes JSON and summary.csv.
python3 tools/bench/run_ninfer_bench_matrix.py --preset core

# Longer run that adds 32k/64k prompt and context-decode points.
python3 tools/bench/run_ninfer_bench_matrix.py --preset full

# Run only the MTP draft-window sweep.
python3 tools/bench/run_ninfer_bench_matrix.py --preset full --suite mtp_sweep
```

Default outputs:

```text
profiles/bench/ninfer-<preset>-<timestamp>/
  commands.sh
  manifest.json
  json/<suite>/<case>.json
  logs/<suite>.<case>.stderr.txt
  summary.csv
  summary.json
```

Use `--resume` to skip completed JSON reports in an existing `--output-dir`, and `--preset smoke`
for a minimal script/runner check. `--no-build` uses the binary supplied by `--bench` without
building it.

Each raw report must be `ninfer_bench_report` schema v12. The flattened summary and schema-v3 matrix
manifest carry native names from the report: selected target, canonical `weights_id`, artifact,
load/read/upload/staging values, Engine memory arenas including request transient and CUDA Graph
allowance, per-test planned logical and allocator-observed workspace peaks, KV capacity and
payload, configured proposal head and graph mode, phase timings and throughput, and speculative
rounds/drafts/acceptance/fallbacks. The matrix manifest is descriptive and records the commands and
selected local inputs; it does not make repository state part of report validity.

`run_serve_corpus.py` runs both registered targets and both published MTP0/MTP3 suites when both
artifacts are supplied. Pass one `--artifact` to select a single target and `--mode mtp0` or
`--mode mtp3` to run only that suite. `--mode dflash7` runs the same decode corpus with DFlash
block=8 (`k=7`) and the optimized proposal head on 35B-A3B DFlash v1 or Qwen3.8-27B DFlash2.
It is rejected for `qwen3_6_27b`. Add
`--sampling greedy` to force exact argmax while retaining the same fixtures and repetition count.
Its schema-v5 result and flattened summaries retain the canonical `weights_id` received from the
schema-v9 serving startup record. The stochastic route pins its complete
temperature/top-p/top-k/min-p/presence/frequency profile explicitly, so model-default changes do
not alter the measurement method. `--sampling p-less` intentionally uses the product-default
p-less sampler and registered temperature (`2.0` for Qwen3.8); startup-log validation rejects a
point if the resolved process mode does not match the selected benchmark mode.

## Concurrent serving benchmark

`run_serve_concurrency.py` measures two separate concurrency properties through real loopback
Chat Completions requests:

- `decode-saturation` submits one long-decode wave and uses only complete one-second intervals in
  which every decode round has exactly the configured batch size. Ramp-up, prefill, and drain
  intervals are excluded.
- `corpus-makespan` shuffles the existing mode-specific corpus once with the fixed seed `20260811`,
  then runs that same order with exactly `N` persistent client workers. A worker submits the next
  request only after its current response completes, and makespan ends when the final response has
  been read. Request bodies are sent in shuffled-order sequence while response waits remain fully
  concurrent, removing client-thread arrival races without serializing inference.

Each concurrency point starts a fresh server because its execution graphs and memory plan are
startup-fixed. Prefix reuse is disabled, startup and warmup are outside both measurements, and the
runner writes per-point JSON, raw serving JSONL, and combined JSON/CSV/Markdown summaries.
The point report records the shuffle seed, dispatch method, shuffled position, and canonical corpus
position for every request.

```bash
python3 tools/bench/run_serve_concurrency.py \
  --artifact qwen3_6_27b=out/qwen3_6_27b_nvfp4.ninfer \
  --mode mtp3 --suite decode-saturation \
  --concurrency 1 --concurrency 2 --concurrency 4 \
  --decode-tokens 8192 \
  --output profiles/bench/concurrent-decode

python3 tools/bench/run_serve_concurrency.py \
  --artifact qwen3_6_27b=out/qwen3_6_27b_nvfp4.ninfer \
  --mode mtp3 --suite corpus-makespan \
  --concurrency 1 --concurrency 2 \
  --output profiles/bench/concurrent-corpus
```

Use `--kv-capacity auto` when the fixed corpus needs more shared KV than the default 262,144-token
pool. A point is intentionally not resumable: combining fragments from separate server processes
would not preserve either a steady interval or one continuous makespan.

## NVFP4 KV context sweep

`run_nvfp4_kv_context.py` drives `ninfer_bench` against `qwen3.8-27b/nvfp4` with MTP3 and the
optimized proposal head. It compares INT8-G64 and NVFP4 KV:

- C=1: combined `pp+tg` at 5k/20k/50k/100k/150k (prefill tok/s and decode tok/s)
- C=2: combined `pp+tg` at 20k/50k, two Engine lanes, aggregate decode tok/s

```bash
python3 tools/bench/run_nvfp4_kv_context.py \
  --weights out/qwen3_8_27b.ninfer \
  --bench build/bench/ninfer_bench
```

## Speed suite (live-serve scenario matrix)

`run_speed_suite.py` is the **fast-iteration** tool: it hits the *already-running* serve
(default `http://127.0.0.1:8081`) — no engine spawn, restart, or reconfiguration — and reports,
per scenario, prefill tok/s, decode tok/s, time-to-first-token, end-to-end wall, MTP draft
accept, and an output hash. Unlike the offline `ninfer_bench` matrix above, it measures the
production serve path and is the right instrument for watching a branch while you edit kernels.

Scenario matrix lives in [`speed_suite_cases.json`](speed_suite_cases.json); fixtures resolve
relative to the repo root (NIAH / chat-history from `examples/cli/messages/`, the OWUI capture and
room-puzzle prompt from `tools/bench/fixtures/`). Add your own realistic prompts there as inline
`prompt` text or committed JSON/`prompt_file` fixtures. Prefill is reported two ways:
`prefill_tok_s` (overall average: tokens / wall prefill time) and `prefill_tail_tok_s`
(steady-state rate over the trailing ≤1 s of prefill — the number your kernel work moves; for
long-context prefill the tail is slower than the average because attention cost grows with
depth). Both come from the server's `usage.prompt_tokens_details` block; the tail requires the
current engine build. Prefill rates there cover the computed (non-reused) suffix only — a cached
prefix is reported as `cached_tokens`, not counted into the rate.

```bash
# Full base set (small ± thinking, multi-turn, tools, 8k/64k NIAH, decode). ~2 min.
python3 tools/bench/run_speed_suite.py --label LABEL --runs 2 --warmup 1

# Fast subset while iterating:  --suite smoke,small,multiturn
# Long-context 128k point:      --include-slow
# Production prefix-reuse-on:   --no-cache-bust   (on by default so identical prompts refetch)
# Server-side TTFT only:        --no-stream
```

Outputs land under `profiles/bench/speed-suite/` (gitignored): a per-run JSON
(`speed-suite-LABEL.json`, per-case aggregates) and a publishable one-row-per-case CSV
(`speed-suite-LABEL.csv`) for the progress doc / cross-branch diff. Metrics read the server's
`usage.prompt_tokens_details` block, so no extra instrumentation is needed.

**Trials:** `--runs N` repeats each case N times *in one session* (mean/median/min/max reported).
For *independent* trials — cold vs warm start, thermal drift, minutes apart, or a different
branch/build — run the script N times with N `--label`s and aggregate the per-label JSONs; the
two are different claims, so keep them separate.

Baseline (RTX 5090, 2026-08-17, MTP3 + lm-head-draft, INT8 KV): prefill ~10.3–10.7k tok/s at
6–8k context but ~5.9k average / **~3.6k steady-state tail** at 64k (attention cost grows with
depth — the tail is the more actionable number, TTFT ~11s); decode ~162–180 thinking /
~138–262 no-thinking. The 64k NIAH case is in the base set precisely because the long-context
prefill regression it exposes is invisible at 8k.

### NIAH: position matrix (lengths x positions)

The NIAH (needle-in-a-haystack) fixtures form a **position matrix**: 6 lengths x 5 needle
positions = 30 cells, deterministic and pure, generated by
[`make_niah_positions.py`](make_niah_positions.py). A cell is a pure function of
`(length, position)` (re-running reproduces the identical fixture byte-for-byte, so committed
fixtures are reproducible). Every length is a prefix slice of one shared **master stream** —
the needle-free 256k master document (one continuous essay, no per-cell content drift) — with
a single `OFFICIAL RECORD` line spliced in as a standalone line at the requested depth.

- **Lengths**: 8k / 64k / 100k / 128k / 150k / 200k. **200k (≈200k tokens) is the new long
  end** — it fits the 262k context, unlike 256k (≈277k tokens) which overflows it. So **256k is
  the master stream** (a content source), not a ladder rung at the 262k context.
- **Positions**: start / q25 / mid / q75 / end (needle depth 0 / 0.25 / 0.5 / 0.75 / 1). The
  committed cells cover the full 8k and 200k ladders plus the mid spine for the middle lengths;
  the rest of the 30-cell matrix is on-demand through the generator.
- **Non-leaking question (defect fix)**: the question states the answer *form* (`ORCHID=<code>;
  COLOR=<color>`) but NOT the values, so a correct answer requires actually retrieving the
  needle. The old question echoed the exact answer, which made every NIAH gate vacuous (a model
  that ignored the haystack still passed); that is fixed in every committed fixture.

```bash
# Generate a cell / a whole ladder (deterministic, writes into examples/cli/messages/)
python3 tools/bench/make_niah_positions.py --length 200k
python3 tools/bench/make_niah_positions.py --length 8k --position all
# NIAH recall gate over a position matrix (live-serve recall gate)
python3 tools/bench/run_niah_check.py --lengths 8k,64k,200k --positions start,q25,mid,q75,end
```

**Gate semantics (read before interpreting matrix results).** The default check is the
*exact-format* needle `ORCHID=493817; COLOR=COBALT` — it passes only if the model both
retrieves the needle AND emits the exact answer form. Two levers: `run_niah_check.py
--needle 493817` is a *recall-only* gate (the value must appear in the answer, format
ignored — the pure recall signal), and `--runs N` repeats each cell for a per-cell pass
rate (e.g. 5/5). The console prints a `[FAIL] <case>: n/N runs failed to retrieve the
needle` line per failing case plus a total failure count on the final verdict; per-cell
retrieved/total and answer snippets are also in the JSON record. The default gate is the
light **mid-spine** (8k + 64k, fast); the matrix runs on demand.

Observed behavior (nvfp4s3 serve, RTX 5090, 2026-08-21 — a serve-dependent sampling
snapshot, not a spec): the exact-format gate passes stably at 8k/64k/150k/200k mid, but
mid cells at 100k–128k are sampling-flaky — 100k mid answered 0/10 exact (values always
present; the model varies the form, e.g. `ORCHID=493817, COBALT`), 128k mid 1/11 (mostly
format drift, occasionally truncated values `ORCHID=4938; COLOR=CO`). Non-monotonic in
length (150k/200k pass), so this is sampling variance over the strict needle, not a
monotonic long-context recall loss — but the occasional 128k value truncation is the one
signal worth tracking as the approximate-attention (sage / nvfp4s3) arc proceeds. For a
stable green gate use the default 8k+64k spine; treat matrix cells with `--runs`.

