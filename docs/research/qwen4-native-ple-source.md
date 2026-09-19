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
