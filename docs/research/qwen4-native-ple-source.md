# Native NVFP4 PLE source and residency qualification

Audited 2026-09-18. This is codec/provenance and host-residency qualification, not a
future checkpoint recipe or model-quality qualification.

## Pinned producer

- Repository: `primitive-ai/Qwen3.8-Flash-Next-PLE-quant`
- Revision: `a0fa93f2b9ed5fcab0cf01b60e88ff1ce13a4763`
- Metadata: https://huggingface.co/primitive-ai/Qwen3.8-Flash-Next-PLE-quant/blob/a0fa93f2b9ed5fcab0cf01b60e88ff1ce13a4763/ples_nvfp4/META.json
- Decoder: https://huggingface.co/primitive-ai/Qwen3.8-Flash-Next-PLE-quant/blob/a0fa93f2b9ed5fcab0cf01b60e88ff1ce13a4763/worker_image_quant.py
- Behavioral measurements: https://huggingface.co/primitive-ai/Qwen3.8-Flash-Next-PLE-quant

The source has 128 contiguous shards, each containing 2,500,012 rows of width 160.
Each safetensors shard owns `weight_e2m1` U8 `[2500012,80]`, `weight_scale` E4M3FN
`[2500012,10]`, and its own `weight_scale_2` FP32 scalar. Global scale means per-shard,
not per-complete-table: exact inspected multiplier words are `0x37b30c31` for shards
0/1 and `0x379f3cf3` for shard 127. A converter must preserve each shard's word.

Even columns use the low nibble; odd columns use the high nibble. E2M1 magnitudes are
`0,.5,1,1.5,2,3,4,6`, with sign bit 3. Every group of 16 adjacent columns has an E4M3
multiplier. Source reconstruction is FP32(E2M1 * E4M3), followed by FP32 multiplication
by the shard scalar; gather casts the result to BF16. The first product is exactly
representable in FP32. Preserve the final FP32-then-BF16 boundary rather than imposing
a different direct-FP64-to-BF16 reconstruction.

The producer's measured NVFP4 table is 28.8 GB and has knowledge/tool scores 92.2/78.7,
versus BF16 92.2/79.2, with reported tool repeat spread +/-1.5. This supports a candidate,
not PPL equivalence. Its alternate FP8 table uses per-row FP32 scales, distinct from
NVIDIA's per-tensor FP8/BF16-scale table. Its default reclaimable mmap and CPU decode
are not the NInfer contract: NInfer requires complete eager OS-locked packed residency,
bounded packed-row staging, and GPU decoding.

## Bounded source fixture

`tools.parity.qwen4.native_nvfp4_ple_fixture` uses exact HTTP 206 ranges, rejecting ignored
or mismatched ranges before reading any payload. It selects shards 0, 1, 127 and local
rows 0, 2,500,011, 17, 123,456 from each. It preserves encoded source rows and scalar words,
plus independently decoded BF16 values and matching upstream BF16 rows. This is 12 rows,
not a full table. The independent scalar decoder does not call the producer decoder.

The comparison checkpoint is `Qwen/Qwen3.8-Flash-Next` revision
`de4b8e4d43b917e7706784d8bb445c9af86a3540`. The producer does not certify its original
BF16 revision, so the comparison is explicitly against this independently pinned release,
not proof of matching quantization provenance. Selected-row reconstruction error is emitted
in the local fixture report, never interpreted as model perplexity or a quality admission.

Source-only output remains safetensors; it is not a second runtime artifact lane. The native
`.ninfer` codec must keep row codes/scales and partition-specific scalar ownership intact.

The acquired twelve-row fixture's FP32 reconstruction relative L2 error against the pinned BF16 release is
`0.10132716476691395`; after the public BF16 gather rounding, relative L2 is
`0.10118661828178988`. Reports label these two precision boundaries separately.
All 1,920 scalar-decoded BF16 words exactly match an independent
execution of the producer's Torch expression. Offline `--convert-source` uses
`pack_nvfp4_partitions`, verifies exact code/scale/multiplier roundtrip, and writes only
`ple.rows` `[3,4,160]` in `NVFP4_PARTITION_F32M` / `partitioned-row-blockscale-k16-v1`.
The product artifact does not contain cached expected kernel outputs. Its identity is
`qwen4/native-ple-qualification` / `primitive-nvfp4-source-rows`.

Focused acquisition/scalar-codec tests: 9 passed; Python compilation and `git diff --check`
passed. This does not cover the GPU consumer, which has separate exact decode/gather tests.

## Native PLE injection weights

`tools.parity.qwen4.native_ple_component_fixture` acquires only layer-1 PLE math/control
payloads from NVIDIA revision `fc694b54fb0174e0913e6adf86691ef85a4ead47`, using bounded
ranges: 65,679,640 bytes total. The six BF16 tensors are key `[10240,2560]`, value
`[2560,2560]`, key/query/convolution norms `[10240]`, and convolution `[10240,1,4]`.
Three exact I64 addressing buffers (3 multipliers, 16 vocabulary sizes, 16 offsets) are
recorded as signed integers in the source JSON, without introducing an I64 runtime format.
The `.ninfer` fixture preserves the six BF16 payloads under original source names with
identity `qwen4/native-ple-component-qualification` / `nvidia-bf16-source`.

This allows the existing native layer-0 output to feed PLE injection before the layer-1
GR read, without acquiring another expert bank or the full table. It is not itself proof
that the combined native gather/injection/history path passes its independent oracle.

## Complete native NVFP4 table

`tools.parity.qwen4.native_ple_full_fixture` acquires only all 128 pinned producer PLE
shards and converts them into one `ple.table` `[128,2500012,160]` tensor with canonical
`NVFP4_PARTITION_F32M` / `partitioned-row-blockscale-k16-v1` storage. The payload is
28,800,138,752 bytes (26.8222 GiB); codes and E4M3 scale words are interleaved per row,
followed by all 128 original FP32 multiplier words. No quantization or numeric recasting
occurs. Artifact identity is `qwen4/native-ple-qualification` /
`primitive-nvfp4-complete-table`, not a registered Engine target.

Conversion feeds `ArtifactWriter` in 16,384-row chunks, never assembling the complete
table in RAM. Four bounded streaming download workers retain original safetensors;
neither source nor artifact files are overwritten. The focused streaming conversion
test covers a partial final chunk, two partitions, noncanonical source tensor ordering,
all code/scale values, and distinct exact scalar words against an independent byte-order
oracle. Together with the canonical partition codec tests: 4 passed.

Local acquisition command (Python 3.11 with the existing numerical-reference packages):

```sh
python3 -m tools.parity.qwen4.native_ple_full_fixture \
  --sources /models/qwen4-ple/primitive-a0fa93f2-nvfp4 \
  --out /models/qwen4-ple/qwen4-ple-nvfp4.ninfer --download
```

The separate `qwen4-ple-nvfp4-boundary-reference.ninfer` contains the first and last
row of every source partition. Its 40,960 BF16 words come from the independent scalar
formula above applied to original safetensors bytes, not the converted artifact or a
production decoder. It is not embedded in the runtime artifact.

`ninfer_qwen4_native_ple_residency_test` is explicitly opt-in through
`NINFER_QWEN4_FULL_PLE=/models/qwen4-ple`. Default CTest returns skip 77, so ordinary
tests never allocate/lock this table. The real test requires payload plus 16 GiB of
available RAM before loading, uses the actual artifact `ResidentHost` materializer,
checks every page with Linux `mincore` and exact `VmLck` accounting, destroys the
Reader and moves the materialization owner before use, and executes packed row
gathers plus GPU decode at T=16,3,1. Maximum CUDA-pinned input staging is 24,064 bytes.
Both ends of all 128 partitions must match the source BF16 words bit-for-bit; owner
teardown occurs after GPU drain and must restore the original locked-memory count.

Full acquisition and conversion are complete. The test builds and its default skip
passes. At the first full-run admission check, Linux `MemAvailable` was only 32.84 GB
after the ZFS filesystem's ARC warmed to 70.30 GB. The explicit 45.98 GB admission
threshold prevented locking; no global cache setting or unrelated service was changed.

The installed module is OpenZFS `2.4.1-1ubuntu5`; official matching source
https://raw.githubusercontent.com/openzfs/zfs/zfs-2.4.1/module/os/linux/zfs/arc_os.c
defines clean evictable MRU/MFU memory in `arc_evictable_memory` and reclaims it through
`arc_shrinker_scan`. Here `zfs_arc_pc_percent=0`. Linux `MemAvailable` omits that ARC
capacity. For this local qualification only, explicit
`NINFER_QWEN4_PLE_ZFS_ADMISSION=1` permits this conservative preflight:

```text
available = MemAvailable + min((mru_evictable_data + mfu_evictable_data) / 2,
                              max(ARC_size - ARC_minimum, 0))
required  = 28,800,138,752 + 16 GiB
```

This counts only half of directly reported clean evictable data, excludes metadata,
ghost and dirty bytes, preserves the ARC minimum, and retains the full 16 GiB reserve.
It changes no production admission behavior. Focused-run admission measured
`MemAvailable=39,025,590,272`, MRU clean data `28,890,492,928`, MFU clean data
`29,510,889,984`, ARC size `64,440,134,120` and minimum `4,189,336,448` bytes.
Discounted ARC was `29,200,691,456` bytes, so capacity was `68,226,281,728` bytes,
above the unchanged `45,980,007,936` requirement.

The temporary `local/ninfer-builder:5090` GPU container uses the existing build
volume and source/model mounts read-only, network disabled, and the same existing
builder `NVIDIA_DISABLE_REQUIRE=1` environment. Inspected cgroup `memory.max` is
`51,539,607,552` (48 GiB), `memory.swap.max=0`, and memlock is unlimited. No unrelated
container or global memory/cache setting is changed.

Complete-table result on RTX 5090 / CUDA 13.1.2: PASS. All **7,031,284** mapped pages
were resident; `VmLck` increased by exactly **28,800,139,264 bytes** including page
rounding. Reader destruction and materialization-owner movement preserved residency.
All 40,960 distinct boundary BF16 words matched the independent source oracle exactly;
T=3 and T=1 reordered/repeated gathers also matched. After stream drain, destroying the
owner restored the baseline `VmLck`. The temporary container exited successfully and
was automatically removed, releasing the table; original source shards and the
converted artifact remain available on disk for subsequent startup. This closes the
complete native NVFP4 PLE residency-capacity and bounded GPU gather/decode gate, not
model PPL, a registered future target, or an end-to-end performance claim.

## Early native fetching and consuming-layer readiness

`NativeRuntime::prepare` now submits `NativePleFetch` immediately after validated token IDs
and accepted raw history determine the row addresses, before page/control/embedding setup.
Host gathering copies represented bytes into bounded pinned storage; the owned nonblocking
stream performs H2D and exact GPU codec decode.
Decoder index0 is independent; index1 waits immediately before PLE injection. Capture uses
an external event-wait node; eager execution uses the ordinary wait flag (CUDA 13.1 rejects
the external flag outside capture). Both producers and consumers drain before reuse/teardown.
No ordinary model weight streams and no CPU floating-point inference were added.

The focused fetch test passes both sole exact codec oracles for C1–4, changed graph
generations, capture before the first event record, deliberately delayed transfers, maximum
4096 columns, cancellation/reuse and unconsumed teardown. Its artificial callback gate warms
codec modules first: lazy CUDA module loading can otherwise wait behind the test's deliberately
blocked producer. The unchanged PLE suite and actual native layers0–3 independent accumulated,
continuation and C4 graph/state checks also pass.

The optional `NINFER_QWEN4_PLE_OVERLAP=/models/qwen4-ple` section of
`ninfer_qwen4_native_compute_real_test` measures actual resident layers0–3 with the complete
NVFP4 table, **28,800,139,264 locked bytes**, bounded pinned staging and **NVFP4-G16 KV**.
It compares the old serialized transfer/decode with the new stream in the shared eager decoder
body, uses identical changing
random-access rows and checks final represented prefix outputs exactly. RTX5090/CUDA13.1.2,
two warmups and nine measured samples per route/shape, under Nsight CUDA/NVTX tracing:

| Width / compact batch | Async rows ready, median µs | Minimum readiness margin before layer0 ends, µs | Serialized / async prefix4, median µs |
|---|---:|---:|---:|
| 1 / 1 | 17.632 | 411.392 | 1762.18 / 1775.90 |
| 1 / 2 | 14.080 | 450.528 | 1905.06 / 1906.24 |
| 1 / 3 | 18.304 | 655.071 | 2755.10 / 2761.73 |
| 1 / 4 | 18.400 | 758.879 | 3127.36 / 3129.89 |
| 3 / 4 | 27.648 | 916.959 | 4134.50 / 4134.72 |
| 65 / 1 | 68.224 | 1700.191 | 7855.74 / 7836.35 |

Readiness includes host gather/submission and GPU transfer/decode, measured from the interval's
start event. NVTX layer ranges and CUDA launch correlations identify each layer's GPU kernels;
the readiness margin is final layer0 kernel end minus the PLE decode kernel end. PLE runs on
stream31 and decoder kernels on stream13 in this trace. In all 54 measured asynchronous rounds,
the rows were ready well before layer0 finished: **no PLE-data wait at layer2 was observed**.
The ordinary layer boundary gap was 1.28–2.82 µs by shape median. This removes the serialized
dependency but shows no material whole-prefix throughput improvement on this workload; the
four-layer compute time dominates. Graph replay ordering/state is qualified separately, not
benchmarked here. It is neither a full-model tok/s result nor a guarantee for
every future model, table size, host contention level or prefill width. Full FP8 table-capacity
performance remains unmeasured; its exact asynchronous codec/order tests pass.

Trace: `profiles/nsys/qwen4-ple-early-fetch/native-prefix4-matched.nsys-rep` and its SQLite
export. Collection uses `nsys profile --trace=cuda,nvtx --sample=none --cpuctxsw=none
--capture-range=cudaProfilerApi --capture-range-end=stop` around the test with the native
layer/compute fixture variables and the overlap opt-in above. A temporary 48GiB/no-swap,
unlimited-memlock container uses the existing builder image; no host settings, installed
dependencies or model downloads changed. The complete table unlocks at process teardown.

### Graph replay and larger-prefill follow-up

The same complete locked NVFP4 table was qualified again after the resident MoE changes,
now including graph replay and W257, with NVFP4-G16 QSA KV. The four actual native layers
use bounded diagnostic inputs and changing full-table row addresses; this is not a full
prompt/model throughput measurement. All eager/graph and serialized/asynchronous final
represented outputs match exactly. State/control allocations are address-stable, and the
captured external event wait uses the new producer generation on each replay.

RTX5090/CUDA13.1.2, two warmups plus nine measured rounds per shape/route:

| Width / batch | Async rows ready eager / graph, µs | Minimum margin before layer0 ends eager / graph, µs |
|---|---:|---:|
| 1 / 1 | 12.832 / 19.104 | 400.766 / 392.638 |
| 1 / 2 | 16.576 / 16.832 | 435.838 / 414.974 |
| 1 / 3 | 21.312 / 18.144 | 573.341 / 551.901 |
| 1 / 4 | 23.328 / 20.032 | 610.653 / 594.365 |
| 3 / 4 | 44.128 / 31.456 | 810.012 / 785.947 |
| 65 / 1 | 68.352 / 73.472 | 1694.423 / 1653.463 |
| 257 / 1 | 242.528 / 217.184 | 4526.409 / 4505.577 |

All126 measured asynchronous rounds were ready before layer0 finished. Independent Astra
trace review mapped graph replay kernels through original graph-node IDs to node creation
inside the layer0 range, then used actual GPU timestamps—not host NVTX duration. Producer
stream31 and consumer13 overlap correctly; no PLE-data stall at decoder index1 was observed.
The W257 graph whole-prefix medians were19.834ms serialized and19.788ms asynchronous; this
does not support a material whole-prefix speedup claim. Early fetching remains valuable as
the correct dependency schedule, without promising zero stalls for future compute speeds
or arbitrary host contention.

Trace: `profiles/nsys/qwen4-ple-final/native-prefix4-graph.nsys-rep` and its SQLite export;
same command as above plus `--cuda-graph-trace=node`. The temporary48GiB/no-swap/unlimited
memlock container was removed at exit and released the complete table.

### Complete original FP8 table residency

`tools.parity.qwen4.native_fp8_ple_full_fixture` acquired only the128 PLE code ranges and
original BF16 multiplier from pinned NVIDIA revision
`fc694b54fb0174e0913e6adf86691ef85a4ead47`. No ordinary main-model weights were downloaded.
All51,200,245,760 code bytes are retained; the exact scale footer is `0x3951`, giving a
51,200,245,762-byte payload. Bounded16MiB download/conversion chunks, manifest-bound resumable
source parts, finite-code checks and independent scalar boundary decoding were used. The
converted first/last row of each partition and footer agree exactly with original source bytes.

The initial MemAvailable-only estimate understated this host's available capacity by omitting
the already-documented clean ZFS ARC allowance. Fresh test admission measured43,847,999,488
MemAvailable bytes plus29,588,282,112 discounted clean ARC bytes, exceeding68,380,114,946
required bytes including the unchanged16GiB reserve. No host cache or swap setting changed.

Complete FP8 execution **passed** on RTX5090/CUDA13.1.2 in a temporary64GiB/no-swap,
unlimited-memlock container. All **12,500,061** payload pages were resident; `VmLck` increased
by exactly **51,200,249,856 bytes** including page rounding. Reader destruction and moving the
materialization owner preserved the mapping/lock. All40,960 independent source-boundary BF16
words match GPU decode exactly, including reordered/repeated T3/T1 gathers. Pinned staging is
bounded to40,960 bytes. After consumers drained, teardown restored the baseline locked-memory
count and the temporary container was removed.

This is a capacity, lifetime and exact transfer/decode qualification, not an FP8 full-model
throughput or PPL result. Host swap occupancy increased during the large residency run even
though the complete PLE payload was locked and the qualification container itself had no swap;
the conservative ARC capacity allowance is not a promise that other processes cannot page.
Production still requires successful eager locking and never falls back to disk-backed row
faulting. No global swap/cache manipulation was performed to manufacture a pass.

Files remain under `/ssdpool2nvme/local_llm/models/qwen4-ple/`: `qwen4-ple-fp8.ninfer`,
`qwen4-ple-fp8-boundary-reference.ninfer`, provenance `qwen4-ple-fp8.json`, and original
`nvidia-fc694b54-fp8/` source parts. Qualification command is
`NINFER_QWEN4_FULL_PLE=/models/qwen4-ple NINFER_QWEN4_PLE_ZFS_ADMISSION=1 /build/tests/ninfer_qwen4_native_ple_residency_test --fp8`
with the limits above and read-only model/build mounts. Both complete native PLE storage
formats now have real residency/transfer evidence; early-fetch performance numbers above
remain specifically the NVFP4 table measurement.
