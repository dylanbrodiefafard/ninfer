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
  `test_anthropic_schema.cpp` — current protocol translation and Responses Item/state/SSE behavior;
- `targets/qwen3_6/test_tool_grammar.cpp` and `test_frontend.cpp` — declared-schema masks,
  speculative grammar transitions, orphan-closing-tag rejection across token boundaries,
  prohibition of tool envelopes inside reasoning and normal reasoning-to-call transitions,
  typed calls, and transactional publication preserving XML-valued arguments. The optional
  real-tokenizer probe compares masks with direct byte acceptance over the represented
  vocabulary at free-text and call-closing boundaries;
- `targets/qwen3_6/test_generation_recovery.cpp` — contextual duplicate detection, truthful
  reasoning-only retry notices, preservation of real messages/results, and legitimate
  changed reads/results and polling;
- `test_typical_cycle.cpp` — suffix-square boundaries, exact repeated-passage evidence,
  non-overlapping coverage, changing/large periods, and productive-length negative controls;
- `test_request_log.cpp` and `test_http_error_handler.cpp` — generation lifecycle records,
  live recovery stages and cumulative counts, preparation rejections, protocol-shaped
  payload-limit errors, and application-error preservation;
- `test_ninfer_bench_support.cpp` — product benchmark CLI, timing boundary, and schema-v14 reports;
- `test_bench_matrix.py` — schema-v14 report consumption by the Python matrix summarizer;
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

`ninfer_qwen3_6_27b_cache_admission_real_test` checks that a 1 MiB RAM budget cannot block
admission at C=1–4. It forces selected-lane and shared-pool victim capture drops, then verifies
overlapping dirty-lane replacements complete at C>1. It uses NVFP4 KV and
`NINFER_QWEN3_6_27B_NVFP4_WEIGHTS`.
The disk Engine suite also checks tiny-RAM capture drops during `HostDisk` admission at
C=1–4, retained sibling continuations, greedy replay parity, and request-local copy timing.

Cache interleaving coverage follows the supported ownership model: one executor mutates
admission/claim state, while CUDA copies and disk workers complete asynchronously. The bounded
explorations execute the real cache implementations:

| Test | Enumerated scope | Observable checks |
| --- | --- | --- |
| `ninfer_kv_ram_cache_test` | 175 states / 1,101 transitions: capacities of one, two, and three equal-size slots, two keys (including duplicates), one claim, and queued/harvested copy ownership | FIFO matching and eviction, claim release/consume, exact counters, copy retirement, exact restored KV |
| `ninfer_kv_disk_cache_test` | 320 schedules over two four-page entries: reader counts 1–16; prefetch dequeued, slot assigned, reading, or filled; promotion, cancellation, reclaim, or switch; injected in-flight read failures | Exact restored KV, obsolete work cannot affect a replacement restore, failure accounting, drained I/O pins |
| `ninfer_qwen3_6_27b_cache_interleavings_real_test` | Every cancellation subset at C=1–4, RAM-only and RAM+disk, for ordinary, MTP, and DFlash execution (60 subsets per backend); ordinary cancellation after the first publication, speculative cancellation after the second with a per-request backend-round witness | Actual RAM reuse, concurrent decode, canceled/completed terminal states, next-token continuation against fresh computation of the same represented history |

The Engine matrix uses NVFP4 KV and `--backend ordinary|mtp|dflash`. Ordinary/MTP use
`NINFER_QWEN3_6_27B_NVFP4_WEIGHTS`; DFlash uses
`NINFER_QWEN3_8_27B_NVFP4_DFLASH_WEIGHTS`. Speculative cases require actual backend execution.
MTP also retains eight C=2 first-publication cancellation cases that reproduce invalid-tail
images shadowing a later usable cache entry. The opt-in real CTests register all three backends.
Its cancellation
trigger is after host-claim consumption; it does not prove cancellation during Engine copy-hold.
Set `TMPDIR` to a writable NVMe scratch directory for the real cache tests: their temporary
stores intentionally exercise spill pressure, and container overlay storage can dominate runtime.
Completed peers must survive a RAM round trip. An ordinary in-flight cancellation discards its
overwritten recurrent state; a later request may miss or use an earlier valid checkpoint, but
must not restore that state under its former frontier. Multi-token greedy trajectories across
different batch/prefill arithmetic routes are not the cache oracle.
The lower-level suites control incomplete DMA and disk read/cancellation boundaries directly.
Two additional RAM schedules force a disk-style worker's pending-event snapshot before or after
an unrelated executor capture, then verify concurrent copy waits, drained pins, and exact KV bytes.
Sixteen disk schedules hold page or immediate-state DMA incomplete at reader counts 1, 2, 8,
and 16, then cancel or shut down before restoring another entry into the same destinations.
They check that CopyHold includes the independent state DMA, retirement drains both CUDA owners,
and replacement KV/hidden bytes and event-ticket cleanup are exact.
Four promotion schedules queue prefetch behind idle spill work, then demand or cancel its
restore with one or sixteen readers. They verify restore progress, resumed idle commits,
exact KV, and transfer of the queued-prefetch pin to ordinary restore ownership.
Sixteen pre-admission pressure schedules hold prefetch at dequeue, slot assignment, page read,
or completion with one, two, eight, or sixteen readers. A single executor then spills and evicts
a RAM victim, recaptures a retained image, and restores the claimed disk source with exact KV.
Disk corruption tests classify expected read failures separately from unexpected worker errors,
quarantine the failed source even when tombstone persistence fails, and verify that a shared-page
sibling restores exactly after durable invalidation and reopening the store.
Four prefetch allocation schedules fail the first or second queued page with one or sixteen
readers, then check exact demanded restore and zero leaked pins. Two additional setup failures
check that an unreturned restore ticket is drained and reclaimed before retry.
Six cancellation schedules pause dequeued prefetch before slot assignment with one, two, or
sixteen readers, including an injected worker exception. Cancellation must retire every reader
pin before strict source invalidation, then allow another exact restore.
An idle-candidate snapshot allocation failure must leave no pin or durable image and allow
the same worker to complete a later spill.
Six spill-queue allocation schedules fail after zero, one, or two successful enqueues, for idle and
emergency spills. An independent RAM I/O pin must survive cleanup, continue to prevent eviction,
and retire normally before a successful spill retry.
Two branch-spill schedules fail the actual next C++ allocation on the preparing thread,
preserve the parent's shared-object reference count,
retry successfully, and require all references to retire when both owning entries are evicted.
A two-gate RAM eviction schedule lets a disk-worker-style snapshot borrow the victim's CUDA
event during eviction's unlocked wait. Completing the victim's DMA must not release its entry
until the snapshot's second DMA finishes; the surviving image then restores exact KV.
Two RAM teardown schedules deny C++ allocations on the destroying executor while D2H or H2D
is incomplete, requiring DMA retirement and exact restored bytes. Three RAM event-exhaustion
cases cover capture timing before DMA, capture completion after gated D2H, and restore timing
before H2D; they require clean ownership and exact retry. A synthetic CUDA launch failure must
retain its execution-error classification. The checkpoint pool retirement test also fails initial
head-fence allocation, then retries before gated D2H and checks completed bytes at release. Six spill-batch schedules fail
page-record allocation after acquiring two, four, or eight jobs, for idle and emergency spills;
all payload claims must retire and a later eight-page spill must restore exactly. Idle-to-emergency
promotion also runs with caller allocation denied. Eight publication schedules fail metadata
preparation for create, refresh, extend, and branch, each with idle and emergency spill. They
require drained pins, an unchanged previous generation, successful retry, and exact restore after
reopening the store. Each retry denies real C++ allocations throughout committed-entry
installation and checks that even caught allocation attempts remain zero. Two failed-unlink
schedules deny allocation during locked/unlocked cleanup, require tombstoned entries to remain
unavailable, and verify retry/reopen.
Nine startup allocation schedules cover index/FIFO/node publication, skipped-entry ownership,
a spawned page-validation worker, manifest rebuild after partially loading normal or skipped
entries, and allocation-free validation-cache cleanup. They require exact reference ownership,
healthy sibling reuse, FIFO eviction, and later reopening of an entry skipped by transient
allocation pressure.
Backend-readiness regressions exercise seven RAM cases, nine disk ranking/reopen cases, and
four resident-planner cases: invalid current tails or unsupported longer frontiers must not hide
a usable alternative. Two same-prefix disk refresh schedules replace the selected generation before claim or while
claim waits for idle publication; both reject the stale plan and permit a fresh claim.
These bounds do not exhaust arbitrary OS/CUDA instruction schedules, unbounded request histories,
all prompt geometries, or every speculative backend combination. Passing them is evidence for
the stated transitions, not a proof that all possible interleavings are correct.

`ninfer_qwen3_6_27b_dflash_cache_cancel_real_test` checks exact-prefix continuation after
published DFlash cancellation through VRAM, RAM, and disk reopen. It uses
`NINFER_QWEN3_8_27B_NVFP4_DFLASH_WEIGHTS`; next-token output is compared with fresh computation
of the identical input. Planner unit cases separately check invalid-hidden rejection,
checkpoint fallback, valid-hidden append, and continued reuse when a suffix regenerates hidden.
Its C=2 RAM+disk case submits repeated exact prefixes, suffixes, changed prompts, and a
cancelled speculative continuation from an idle Engine under retained-lane capture pressure.
It requires actual HostRam and reopened HostDisk restores, DFlash execution, and matching
fresh next-token results. A 180-second watchdog for each admission lifecycle reports the
current phase and waiting/running/prefill/decode counters before exiting directly; shutdown
remains bounded by the CTest timeout. Use the NVMe `TMPDIR` described above for disk work.
A further three C=2 DFlash admissions use a 256 MiB RAM budget checked to fit only one image,
two occupied retained lanes, and a target already evicted to disk. Each HostDisk admission must
both capture a retained victim and evict a RAM image before its continuation is compared with
fresh computation. This exercises disk prefetch followed by synchronous RAM-pressure handling
while the request is still counted as waiting; it does not force the OS worker schedule.
Use `--case c2` to run just these C=2 lifecycle and RAM-pressure cases.
The disk Engine suite's `--case corrupt` selection truncates a cached page pack after startup
and requires transparent cold continuation with the same next token as fresh computation at
C=1–4, with C−1 healthy peers each decoding 128 tokens. The DFlash artifact environment runs
the same four cases with DFlash and checkpoint capture enabled. The failed entry is invalidated;
a later request must also succeed.
Its `--case metadata` selection checks 32 allocation-failure recoveries: C=1–4 at host
descriptor creation, partially published restore setup, post-copy checkpoint installation,
and lazy CUDA completion-event creation after restore DMA. The eight event cases also run via
`--case event`.
Descriptor/setup/event cases use ordinary and DFlash execution; checkpoint cases use MTP and DFlash
and require an actual captured checkpoint. Peers must finish and cold output must match fresh
computation after the injected failure.
The RAM Engine suite's `--case fallback` selection injects a one-shot restore metadata
allocation failure after H2D submission for a 128-token NVFP4 KV image. Sixteen cases combine
C=1–4, RAM-only or RAM+disk, and ordinary or DFlash execution. Each requires consumption of the
fault, discarded RAM ownership, cold continuation matching fresh computation, completion of
healthy decoding peers, and a successful later request. DFlash cases retain checkpoint capture
through recovery. These test transient cache allocation failure, not sustained process-wide OOM.
The RAM Engine suite's `--case planning` selection checks another 16 C=1–4 cases with
RAM-only/RAM+disk and ordinary/DFlash execution. A one-shot optional cache-lookup allocation
failure must preserve cold admission, healthy peers, and fresh next-token behavior. Candidate
version memoization prevents retrying the same failed lookup in a scheduler spin.
The existing RAM Engine suite accepts `--case mtp` to run only its MTP suffix-cache case;
that case checks the configured draft window and actual speculative execution without fixing
the adaptive policy to a particular first-round draft length.

`ninfer_kv_ram_cache_perf_test` checks host pack/unpack bandwidth against pinned memcpy.
`ninfer_kv_disk_cache_perf_test` spills and restores a ~90 MiB 64-plane INT8 page image on the
repository `out/` NVMe (override with `NINFER_KV_DISK_PERF_DIR`), against a 40 MB/s floor and
buffered plus `O_DIRECT` sequential POSIX write/read baselines.
`ninfer_kv_ram_cache_opt_test` checks event
overlapped restore, fragmented vs contiguous PageMajor runs, and GDN/hidden RAM round-trips.
`ninfer_kv_ram_cache_test` includes `test_copy_compute_stream_overlap`: callback-gated D2H/H2D
on `copy_stream` remain incomplete while a compute-only event on `device.stream` completes.
The same gates verify that eviction/consume fence unfinished copies before retiring and reusing
their pinned storage. It also checks that `unpack_device` without an intervening harvest still reports
both save and load elapsed, and that `consume` without harvest clears pending copy ids and folds
D2H elapsed into save. Post-DMA capture and restore metadata allocation faults use the same
gates to verify incomplete-copy ownership, safe cleanup, exact subsequent restores, and reclaimed
capacity. The arena suite checks fragmented retirement and full coalescing while preserving live
bytes. `ninfer_device_test` checks `order_copy_after_compute`.
The real checkpoint suite's `--case event-allocation` checks MTP or DFlash Program capture with
head-event allocation failure at a prefill checkpoint. The request must complete without a
capture and consume the injected failure; later capture and actual checkpoint restore must
succeed, with next-token output matching fresh computation. The default real checkpoint run
also includes this case.
The runtime-mechanism suite fills the checkpoint recycling pool, then verifies that dropping
an excess image waits for its gated D2H and releases the completed bytes without growing the pool.
Twelve copy-snapshot allocation schedules cover both snapshot buffers, D2H/H2D, and CPU wait,
compute-stream wait, or timing harvest. They require safe completion without leaked I/O pins,
exact destination bytes, and successful consumption and capacity reclamation.
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
than retaining per-token logits. It executes the full 4096-token scalar capacity twice, validates
QSA block/count/tail structure and current NVFP4 rows at positions 3-5, 2047-2053, and 4095, and
keeps an exact 1,307,372-byte probe transcript. It then executes and exactly replays both one-shot
T=4096 prefill and the `64+1+1986+1+2044` partition, crossing the GDN 64-token boundary and the QSA
2051-to-2052 selector switch. Every route validates the complete persistent-state inventory,
exact PLE addressing/history, and final-layer QSA selection structure. Scalar/prefill floating
deltas are reported only as localization evidence: their first persistent difference is the
layer-0 FP32 GDN recurrence after an exact 61,440-byte BF16 convolution state, and the independent
GDN Op test admits scalar T=1, one-shot, aligned 64x64, and the five-part schedule at the exact
post-expansion `Hq=Hv=48,D=128,T=4096` geometry against one complete FP64 oracle. The same oracle
directly covers T=447/448/449 around the fused-Q dispatch and first recurrent tail, while a
workspace-interval witness crosses the same allocation boundary. Per-token output hashes remain
supplementary; the scalar probe transcript and full 157,147,144-byte continuation are compared
byte-for-byte across reset/replay, and the raw continuation is compared again after each rejected
overflow. Through position 2050 every complete block fits; at positions 2051 and
later the test checks structural selection invariants and deterministic replay rather than
claiming an independent top-block score oracle:

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
