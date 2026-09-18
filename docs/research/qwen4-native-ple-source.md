# Native NVFP4 PLE source and bounded qualification

Audited 2026-09-18. This is codec/provenance evidence, not a future checkpoint recipe or
complete-table residency/quality qualification.

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
