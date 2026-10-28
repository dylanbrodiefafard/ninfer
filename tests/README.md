# Tests

The CTest graph is the native Radeon AI PRO R9700 product graph. It links the same `ninfer_r9700_core`, `ninfer_engine`, and `ninfer_serve` targets as the applications and registers the production-owned `tools/r9700` qualification executables. It does not compile an alternate device backend or duplicate production kernels inside tests.

## Organization

- `ops/test_fp8_int4_kv_oracle.cpp` is the independent exact host authority for FP8 E4M3FN keys,
  signed INT4 values, and FP16 value scales.
- `artifact/` covers generic container framing and the supported direct and row-split layouts.
- `targets/qwen3/` covers family-owned runtime mechanisms used by Qwen3.8-27B without
  requiring checkpoint resources.
- protocol tests cover the advertised OpenAI, Anthropic, Responses, request-log, Prometheus
  `/metrics` exposition, and server-option behavior.
- `tools/r9700/*_qual` owns physical gfx1201 operator, cache, transaction, runtime, and Engine
  qualification against independent exact or FP64 oracles.

Real-model token, perplexity and inference-performance checks run through the R9700 qualification workflow and require the complete BF16-derived product artifact.

## Public CPU checks

`.github/workflows/host-checks.yaml` runs file hygiene, the Python benchmark/report contracts and both Compose configuration checks on Ubuntu 24.04 with Python 3.11. It does not build HIP or claim GPU/model qualification. Run its Python tests without ROCm, model weights or third-party Python dependencies:

```sh
python3.11 -m unittest discover -s tests -p 'test_bench_matrix.py' -v
python3.11 -m unittest discover -s tests -p 'test_serve_corpus.py' -v
```

The workflow's `pre-commit run --all-files` uses the pinned tools in `.pre-commit-config.yaml`. Full native and GPU promotion gates remain separate.

## Build and run

Set private host inputs as described in `../docs/maintainer/local-development.md`. Source `.ninfer-local.env` before native tests so official-tokenizer sections receive `NINFER_OFFICIAL_TOKENIZER_DIR` and physical qualifiers receive the selected PCI identity. Run GPU tests under `flock --exclusive "$NINFER_GPU_LOCK"`; builds need no lock.

```bash
cmake -S . -B build-r9700 \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_HIP_ARCHITECTURES=gfx1201 \
  -DBUILD_TESTING=ON
cmake --build build-r9700 --parallel
flock --exclusive "$NINFER_GPU_LOCK" ctest --test-dir build-r9700 --output-on-failure
```

The tests expect the selected physical gfx1201 device. Absence of the supported R9700 is a test failure, not a request to select another backend. The self-contained artifact tests create only small temporary fixtures and need no checkpoint.

Run only the physical qualification set with:

```bash
ctest --test-dir build-r9700 -L r9700 --output-on-failure
```

The token-logprob real-model check uses the explicit production artifact:

```bash
flock --exclusive "$NINFER_GPU_LOCK" \
  env NINFER_R9700_WEIGHTS=/absolute/path/to/r9700-lq4.ninfer \
  ctest --test-dir build-r9700 -R '^ninfer_r9700_(token_logprobs_qual|engine_logprobs_real_test)$' \
    --output-on-failure
```

Run the retained Python container and report-consumer contracts with an explicitly selected Python interpreter that provides pytest, Torch, and safetensors. Fail the dependency check before starting the suite rather than falling back to the unqualified system interpreter:

```bash
TEST_PYTHON=/absolute/path/to/python-with-pytest-torch-and-safetensors
"${TEST_PYTHON}" -c 'import pytest, safetensors, torch'
NINFER_RUN_R9700_CODEC_TESTS=0 "${TEST_PYTHON}" -m pytest \
  tests/artifact \
  tests/test_bench_matrix.py \
  tests/test_serve_corpus.py \
  tools/convert/qwen3_8_27b_r9700/test_codec.py \
  tools/convert/qwen3_8_27b_r9700/test_vectorized_codec.py
```

The Python codec tests are independent converter/storage checks. They do not substitute for the physical operator qualifiers or the real-model perplexity and exact-token gates.

## Admission rule

A permanent test must protect supported observable behavior or a realistic regression: exact artifact framing, converter transforms, numerical operator output, state publication, Engine request behavior, or an external protocol. Source-shape scans, compatibility spellings, deleted formats, and performance thresholds do not belong in the permanent suite. Performance evidence is recorded by the benchmark and profiler workflow at the level of the claim.

## Constrained-output Engine integration

`ninfer_r9700_output_constraint_real_test` exercises JSON Schema, exact EBNF and an unconstrained request together at C3, with the fixed production cache, Device Graphs, ordinary decode, MTP and DFlash chain verification. It checks schema values, exact text, EOS, streaming/logprob attribution and a second cached-compilation pass. The plain request switches from no-thinking to thinking on the second pass and checks closing-token counts and streamed logprobs across the reasoning boundary. The frontend suite checks represented byte/escape syntax, Unicode lengths, special-token literals, reasoning transitions and speculative rollback. `ninfer_schema_grammar_test` checks finite JSON languages and next-token masks against explicit word sets, including both object orders. `ninfer_grammar_tokenizer_test` compares masks with byte-wise acceptance for duplicate, prefix-related, and Unicode token spellings. The token-grammar suite also checks shared-rule alternatives, recursive schemas, cache isolation and concurrent cold compilation with independent completed-call validators.

```bash
flock --exclusive "$NINFER_GPU_LOCK" \
  env NINFER_R9700_WEIGHTS=/path/to/models/qwen3.8-27b-r9700-lq4/qwen3.8-27b-r9700-lq4.ninfer \
  ctest --test-dir build-r9700 -R '^ninfer_r9700_output_constraint_real_test$' --output-on-failure
```

## Disposable cache-write Engine integration

`ninfer_r9700_engine_cache_write_real_test` checks `cache_write=false` source-preserving current/checkpoint reuse and writable continuation at C1/C2, ordinary/MTP/DFlash, capacity rejection, original RAM migration, disposable promotion, concurrent original/fork requests, disk cancellation and shutdown. The native RAM qualifier independently checks exact bytes for all three cache planes with mixed layouts and fragmented page mappings. Run the real fixture with the same explicit production artifact and GPU lease as the other R9700 Engine tests:

```bash
flock --exclusive "$NINFER_GPU_LOCK" \
  env NINFER_R9700_WEIGHTS=/path/to/models/qwen3.8-27b-r9700-lq4/qwen3.8-27b-r9700-lq4.ninfer \
  ctest --test-dir build-r9700 -R '^ninfer_r9700_engine_cache_write_real_test$' --output-on-failure
```

`ninfer_startup_watchdog_test` uses subprocesses to verify exit status 124 for stalled startup and warmup, including held logging locks and a full stderr pipe. Advancing load progress can outlast the inactivity interval; repeated/regressing counters cannot. Successful warmup disarms the guard before idle serving, and ordinary startup exceptions stop its thread. No GPU hang is injected.
