# Tests

The retained tests protect current `.ninfer`, numerical operator, target, runtime-transaction,
benchmark-report, and external protocol behavior. Repository verification principles are defined in
[`../AGENTS.md`](../AGENTS.md); Op contract and CUDA implementation guidance is in
[`../docs/maintainer/op-development.md`](../docs/maintainer/op-development.md).

## Organization

- `artifact/` — Python container, registered layout, quantization, and resource behavior;
- `ops/` — one identifiable qualification suite per semantic Op or closely related overload group,
  using independent numerical/state-transition oracles at real supported shapes;
- `ops/linear/` — weight/activation-profile-specific public Linear conformance tests plus their
  one shared input generator, FP64 GEMM oracle, tolerance registry, and output/effects mechanics;
- `ops/linear_add/`, `ops/linear_pair/`, `ops/linear_swiglu/` — fused-Op suites split by registered
  weight/activation profile, each evaluating its complete formula rather than composing production
  Ops;
- `targets/qwen3_6/` — shared tokenizer/template, multimodal preprocessing, MRoPE, prepared-prompt,
  stop/output decoding, hybrid topology, decoder/GDN and round-state layouts/views, shifted-MTP
  alignment, Vision control, and family runtime mechanisms;
- `targets/qwen3_6_27b/` — registered inventory, converter recipe, source verifier, artifact
  bindings, reference diagnostics, family Program/multimodal/MTP behavior, and the opt-in real-Engine
  prefix and RAM-tier tests;
- `targets/qwen3_6_35b_a3b/` — registered inventory/converter contracts, artifact-native diagnostic
  reference, MoE oracle, typed binding, selected-expert row access, 256K INT8 memory calculation,
  and the opt-in real public-Engine route;
- `targets/qwen4/` — research-only formula/continuation fixtures and opt-in native
  four-token Text execution for Qwen4-preview GR, PLE, exact n-gram addressing, QSA, sparse MoE,
  and GDN; these do not register or advertise the preview as an Engine target;
- `test_ninfer_artifact_reader.cpp` — C++ framing, directory, encoded-size, payload-span, and
  geometry behavior against a self-contained C++ fixture;
- `test_request_memory.cpp` — startup-frozen request-transient capacity, stable address,
  activation alignment, rejection, and peak semantics;
- `test_openai_schema.cpp`, `test_responses_schema.cpp`, `test_response_store.cpp`,
  `test_anthropic_schema.cpp`, and `test_tool_call_parser.cpp` — current protocol translation,
  Responses Item/state/SSE behavior, and incremental tool-call behavior;
- `test_request_log.cpp` and `test_http_error_handler.cpp` — generation lifecycle records,
  preparation rejections, protocol-shaped payload-limit errors, and application-error preservation;
- `test_ninfer_bench_support.cpp` — product benchmark CLI, timing boundary, and schema-v9 reports;
- `test_bench_matrix.py` — schema-v9 report consumption by the Python matrix summarizer;
- `test_serve_corpus.py` — serving request-log schema compatibility at the measurement consumer;
- device/tensor/arena tests — reusable lower-component behavior; KV tests cover the core physical
  container, family runtime tests cover dimension-driven GDN storage/view mechanics, and Op tests
  cover mathematical state transitions at their own boundary.

Tests are grouped by observable risk, not by mirroring every source file or class.
`ops/op_tester.h` and `ops/op_check.h` own only reusable device/guard and comparison mechanics.
Concrete numerical criteria remain named by the semantic Op suite; there are no cross-Op tolerance
presets.

`ops/quantized_weight.h` is the common packed-weight fixture for Q4/Q5/Q6/W8 and NVFP4 Op tests. It
owns deterministic payload generation, device `Weight` views, row views, and independent logical
weight decoding.

## Build and run

One command for the full C++ unit suite (GPU builder container, excluding opt-in
real-artifact Engine tests):

```bash
./scripts/run-unit-tests.sh
```

`./scripts/dev-setup.sh` starts `ninfer-builder` from this repository's Dockerfile
`build` stage when the container is not already running. Extra arguments go to CTest
(`./scripts/run-unit-tests.sh -R ninfer_sampling_test`). `--real` includes the
opt-in Engine tests and auto-finds exact `.ninfer` filenames in `models/`, `out/`,
`/models`, the builder's models mount, and sibling folders of that mount. Override
with environment variables or `models/weights.env`. `--print-weights` shows what
`--real` would use without running tests. `--python` also runs the host pytest suite
when that interpreter can import `pytest` and `torch`. The script exits before CTest
when the GPU has less than 20 GiB free and prints the processes holding VRAM.

Equivalent native commands:

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTING=ON
cmake --build build --parallel
ctest --test-dir build --output-on-failure -E '_real_test$'
```

GPU tests fail if there is no usable CUDA device. CTest skip (`77`) is only for
opt-in real-artifact Engine tests when the corresponding weights env is unset.
Frontend tests that need the official HF tokenizer skip those cases when
`NINFER_OFFICIAL_TOKENIZER_DIR` (or the maintainer checkout paths) is unset;
synthetic tokenizer coverage still runs. `ninfer_gqa_attention_test --full`
restores the complete Cartesian matrix; CTest uses the route-boundary unit set.

Run a focused target for a localized change:

```bash
cmake --build build --parallel --target ninfer_sampling_test
ctest --test-dir build -R ninfer_sampling_test --output-on-failure
```

Enable uniform floating-point error records when establishing or reviewing an Op criterion:

```bash
NINFER_OP_REPORT_STATS=1 \
  ctest --test-dir build -V -R '^ninfer_(rmsnorm|gqa_attention)_test$'
```

Every participating comparison emits one `OP_ERROR_STATS` record containing the stable case label,
actual error, active limit, and error-to-limit ratio. The switch changes reporting only; the same
statistics still drive the normal verdict. Passing tests remain quiet without it.

Linear tests are independently runnable by weight and activation-compute profile:

```bash
cmake --build build --parallel --target \
  ninfer_linear_q4_a16_test ninfer_linear_q5_a16_test \
  ninfer_linear_q6_a16_test ninfer_linear_w8_a16_test
ctest --test-dir build -R '^ninfer_linear_(q4|q5|q6|w8)_a16_test$' --output-on-failure
```

All Linear files use `ops/linear/linear_test_common.{h,cpp}` and the same
`ops/quantized_weight.h` fixture as the fused projection tests. The fixture produces the complete
packed GPU payload and exact-decodes the logical float rows used by the one
`cpu_linear_gemm_fp64()` reference. The reference performs naive double accumulation and never
reproduces a production route's activation quantization, staging, reduction tree, or BF16 output
rounding. Each activation compute path selects one centrally defined comparison tolerance for its
whole suite; private kernel, schedule, launcher, and T selection do not change it. Individual test
files call public `linear()` and contain no private selector, launcher, schedule, or kernel
assertions.

Run the native Python suites with the project Python environment:

```bash
python3 -m pytest \
  tests/artifact tests/targets/qwen3_6_27b tests/targets/qwen3_6_35b_a3b \
  tests/test_bench_matrix.py tests/test_serve_corpus.py
```

The Python binding tests use `NINFER_QWEN3_6_27B_ARTIFACT` when set, otherwise they look for
`out/qwen3_6_27b.ninfer`. They report a pytest skip when neither path provides the real
artifact. The 35B-A3B reference binding test follows the same rule with
`NINFER_QWEN3_6_35B_A3B_ARTIFACT` and `out/qwen3_6_35b_a3b.ninfer`. The remaining Python
target tests still run without either artifact.

The C++ prefix/MTP, RAM-tier, and three-tier disk integration tests are separately opt-in because
they load a full artifact and run the real Engine. Point `NINFER_QWEN3_6_27B_WEIGHTS` and/or
`NINFER_QWEN3_6_27B_NVFP4_WEIGHTS` at any Engine-loadable 27B-family `.ninfer` of that weight
profile; the artifact identity selects the target. Qwen3.6-27B and Qwen3.8-27B both work, so
`NINFER_QWEN3_6_27B_NVFP4_WEIGHTS` may be `out/qwen3_6_27b_nvfp4.ninfer` or
`qwen3_8_27b_nvfp4.ninfer`. `ninfer_qwen3_8_27b_mtp_nvfp4_real_test` additionally requires
`NINFER_QWEN3_8_27B_NVFP4_MTP_WEIGHTS` (BF16-sourced NVFP4 MTP). C=1 greedy MTP must emit 24
tokens with speculative rounds; overlapping C=2 and C=3 must complete the requested
lengths with MTP decode (packed MTP verify is not C=1-token-identical). k=3 and k=5.
The RAM-tier test covers capture sites 1–3, INT8 KV, MTP, oversize drop,
VRAM-wins-equal-reuse, longer-RAM-beats-shorter-VRAM, suffix prefill after RAM restore, RAM disabled, queued matcher,
`allow_prefix_reuse=false`, rewrite-checkpoint restore, dirty-lane checkpoint restore, cancel-after-consume, consume-then-VRAM,
overlapping `submit()` at `max_concurrency=2`, C=2/C=3 sequential FullReset onto an empty lane
keeping earlier chats in VRAM, C=2 continue-A refreshing recency so a later FullReset covers the
older dirty lane, C=2 FullReset covering the oldest dirty lane and RAM restore covering the LRU dirty
lane, exclusive
FIFO occupancy (RAM hit drops the restored chat from `used`/`entries`; later spill recaptures it),
one-entry spill drop of a dirty-lane occupant, Engine teardown after a RAM restore,
and the C=3 shared-pool analog (three 3-page chats, two 4-page continuations plus RAM suffix restore of the third, 2-page fits-now backfill, blocked 4-page tail, exact vs suffix reuse). `ninfer_qwen3_6_27b_disk_real_test` covers the SSD third tier on a real Engine: disk-without-RAM construction, restart `HostDisk`, inclusive disk after RAM consume, equal-length VRAM then RAM then disk, longer disk over shorter VRAM, suffix prefill after disk restore, C=1 dirty-lane and RAM-full disk hits, C=2 empty-lane and dirty-only disk hits, C=3 overlapping empty lanes, empty-lane disk vs two VRAM continues, triple overlapping `HostDisk` covering three dirty lanes, disk restore plus two MRU VRAM continues, duplicate disk submit, queued disk matcher behind a full batch, cancel-during-disk-restore, suffix disk with occupants, two HostDisk plus one MRU VRAM continue, disk admit after two in-flight VRAM continues, greedy vs DFlash directory fingerprint, and DFlash2 disk restore. `ninfer_admission_policy_test` locks the same 10-page leftover-2 / leftover-0 / no-lane arithmetic.
`ninfer_kv_ram_cache_perf_test` checks host pack/unpack bandwidth against pinned memcpy.
`ninfer_kv_disk_cache_perf_test` spills and restores a ~90 MiB 64-plane INT8 page image on the
repository `out/` NVMe (override with `NINFER_KV_DISK_PERF_DIR`), against a 40 MB/s floor and
buffered plus `O_DIRECT` sequential POSIX write/read baselines.
`ninfer_kv_ram_cache_opt_test` checks event
overlapped restore, fragmented vs contiguous PageMajor runs, and GDN/hidden RAM round-trips.
`ninfer_kv_ram_cache_test` includes `test_copy_compute_stream_overlap`: a 32 MiB hidden D2H/H2D
on `copy_stream` must still be in flight after a compute-only `cudaEventSynchronize` on
`device.stream`. It also checks that `unpack_device` without an intervening harvest still reports
both save and load elapsed, and that `consume` without harvest clears pending copy ids and folds
D2H elapsed into save. `ninfer_device_test` checks `order_copy_after_compute`.
`ninfer_kv_ram_cache_large_test` moves a 27B-shaped GDN slot (~147 MiB) and a
64-plane ~100 MiB INT8 KV image both ways, including a two-slot GDN plus KV restore, and repeats
that copy/compute overlap proof:

```bash
NINFER_QWEN3_6_27B_WEIGHTS=$PWD/out/qwen3_6_27b.ninfer \
  ctest --test-dir build -R ninfer_qwen3_6_27b_prefix_real_test --output-on-failure
NINFER_QWEN3_6_27B_NVFP4_WEIGHTS=/path/to/qwen3_8_27b_nvfp4.ninfer \
  ctest --test-dir build -R ninfer_qwen3_6_27b_ram_real_test --output-on-failure
NINFER_QWEN3_6_27B_NVFP4_WEIGHTS=/path/to/qwen3_8_27b_nvfp4.ninfer \
  ctest --test-dir build -R ninfer_qwen3_6_27b_disk_real_test --output-on-failure
NINFER_QWEN3_8_27B_NVFP4_DFLASH_WEIGHTS=/path/to/qwen3_8_27b_nvfp4_dflash.ninfer \
  ctest --test-dir build -R ninfer_qwen3_6_27b_disk_real_test --output-on-failure
NINFER_QWEN3_6_27B_NVFP4_WEIGHTS=/path/to/qwen3_8_27b_nvfp4.ninfer \
  ctest --test-dir build -R ninfer_qwen3_6_27b_context_checkpoint_real_test --output-on-failure
NINFER_QWEN3_8_27B_NVFP4_DFLASH_WEIGHTS=$PWD/out/qwen3_8_27b_nvfp4_dflash_w8.ninfer \
  ctest --test-dir build -R ninfer_qwen3_6_27b_context_checkpoint_real_test --output-on-failure
NINFER_QWEN3_8_27B_NVFP4_DFLASH_WEIGHTS=$PWD/out/qwen3_8_27b_nvfp4_dflash_w8.ninfer \
  ctest --test-dir build -R ninfer_qwen3_8_27b_dflash_real_test --output-on-failure
NINFER_QWEN3_8_27B_NVFP4_MTP_WEIGHTS=/ssdpool2nvme/local_llm/models/qwen3.8-nvfp4-mtp-nvfp4-from-bf16/qwen3_8_27b_nvfp4.ninfer \
  ctest --test-dir build -R ninfer_qwen3_8_27b_mtp_nvfp4_real_test --output-on-failure
NINFER_QWEN3_6_35B_A3B_WEIGHTS=$PWD/out/qwen3_6_35b_a3b.ninfer \
  ctest --test-dir build -R ninfer_qwen3_6_35b_a3b_ram_real_test --output-on-failure
```

The Qwen4 verification artifact has a separate unregistered test. Its weights come from the
Qwen3.8-Flash-Next preview checkpoint. The test executes four frozen
numeric token/target pairs through the complete C=1 eager Text schedule twice, checks exact
reset/replay and routing/state diagnostics, and prints per-token NLL plus their PPL. It does not use
the Engine or expose a runnable product identity:

```bash
NINFER_QWEN4_VERIFY_WEIGHTS=/path/to/qwen4_ud_iq1_s_verify.ninfer \
  ctest --test-dir build -R ninfer_qwen4_program_real_test --output-on-failure
```

`ninfer_qwen4_numerics_real_test` loads the same artifact once, obtains represented BF16 inputs at
real Program/GR boundaries, independently decodes its packed matrices, and checks complete FP64
formulas for layer-0 GR, layer-0 Q5_K and layer-2 Q6_K GDN, layer-3 QSA with NVFP4-G16 K/V,
layer-0 sparse MoE, and PLE. Exact checks own integer routing, codec-addressed stage bytes, retained
PLE history, and QSA metadata; numerical criteria own floating-point outputs and computed state:

```bash
NINFER_QWEN4_VERIFY_WEIGHTS=/path/to/qwen4_ud_iq1_s_verify.ninfer \
  ctest --test-dir build -R ninfer_qwen4_numerics_real_test --output-on-failure
```

The long-frontier companion keeps GR snapshots disabled and uses bounded streaming hashes rather
than retaining per-token logits. It executes the full 4096-token capacity twice, validates QSA
block/count/tail structure and current NVFP4 rows at positions 3-5, 2047-2053, and 4095, and keeps
an exact 1,307,372-byte probe transcript. Per-token output hashes are supplementary; the probe
transcript and full 157,147,144-byte continuation are compared byte-for-byte across reset/replay,
and the raw continuation is compared again after each rejected overflow. Through position 2050
every complete block fits; at positions 2051 and later it checks structural selection invariants
and exact reset/replay rather than claiming an independent top-block score oracle:

```bash
NINFER_QWEN4_VERIFY_WEIGHTS=/path/to/qwen4_ud_iq1_s_verify.ninfer \
  ctest --test-dir build -R ninfer_qwen4_program_long_real_test --output-on-failure
```

`--system-prepend` is applied on every request, including follow-ups, so the leading system
tokens stay in the reusable prefix. `ninfer_serve_system_prepend_real_test` checks VRAM reuse on
turn 2 and a host-RAM restore after an unrelated chat spills the first turn:

```bash
NINFER_QWEN3_6_27B_WEIGHTS=$PWD/out/qwen3_6_27b.ninfer \
  ctest --test-dir build -R ninfer_serve_system_prepend_real_test --output-on-failure
```

Run the peer 35B-A3B route independently:

```bash
NINFER_QWEN3_6_35B_A3B_WEIGHTS=$PWD/out/qwen3_6_35b_a3b.ninfer \
  ctest --test-dir build -R ninfer_qwen3_6_35b_a3b_real_test --output-on-failure
```

Without the corresponding variable CTest marks each C++ integration test as skipped. These tests
do not use another numerical/execution path's generated tokens as a golden.

The capability-evaluation coordinator has its own environment and unittest entry point:

```bash
PYTHONPATH=eval eval/.venv/bin/python -m unittest discover \
  -s eval/tests -p 'test_*.py'
```

Run the serving contract manually after starting a resident server in another terminal:

```bash
./build/apps/ninfer-serve out/qwen3_6_27b.ninfer \
  --host 127.0.0.1 --port 18080
```

```bash
python3 -m tools.smoke.serve_contract \
  --base-url http://127.0.0.1:18080 --model qwen3.6-27b
```

This smoke check is intentionally not a CTest: it needs the real artifact, a supported GPU, and a
server process that remains alive while the client exercises OpenAI Responses/Chat, Anthropic,
state, streaming, and multimodal requests.

The thinking-preservation fixture starts and stops its own server, submits a fixed two-step tool
history, compares restored and cold greedy output, compares stripped and preserved closed-turn
prompt lengths, and verifies turn/response rewrite-checkpoint reuse paths plus Responses
inheritance:

```bash
python3 tools/smoke/serve_thinking_preservation.py \
  --artifact out/qwen3_6_27b.ninfer --backend mtp

python3 tools/smoke/serve_thinking_preservation.py \
  --artifact out/qwen3_6_35b_a3b.ninfer --backend dflash
```

The shared messages are in
[`fixtures/serve/qwen3_6_thinking_preservation.json`](fixtures/serve/qwen3_6_thinking_preservation.json).

## What belongs here

A permanent test should protect one current risk, such as:

- exact registered artifact bytes, geometry, object binding, or conversion transform;
- a numerical operator contract with an independent oracle;
- family Frontend or Program frontier, prefix, MTP, or multimodal behavior;
- generated-token commit/stop/cancel consistency;
- public benchmark or OpenAI/Anthropic observable behavior;
- a reproduced supported bug.

Performance-only assertions belong in benchmarks and profiler review. Source scans,
implementation-shape assertions, trivial getters/configuration, retired command surfaces, and
broad additions without a concrete regression risk do not belong in the permanent suite.
