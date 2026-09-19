# Native Qwen4 Vision execution

The exact preview's complete BF16 Vision tower now has a target-owned execution schedule
in `src/targets/qwen4/vision.{h,cpp}`. This is an unregistered qualification route, not an
Engine model identity. It executes patch projection, source position interpolation, all
27 encoder blocks, and the closed merger into 2560-wide visual tokens. All weights stay
on the GPU. Workspace and control staging are startup-fixed; execution uses central Ops
and permits CUDA Graph capture. The caller drains execution before reconfiguration or
teardown. Captured input addresses remain the caller's responsibility.

`tools/parity/qwen4/native_vision_fixture.py` acquires only the 333 source BF16 tensors:
897,862,112 payload bytes. It checks the complete source inventory and exact shapes,
uses validated HTTP 206 ranges, and refuses to replace an existing artifact. The local
artifact is `models/qwen4-native-layers/qwen4-vision.ninfer` under the shared model root.

## Precision and qualification

The new closed `linear_bias` Op computes the complete biased projection before its BF16
output. The initial separate Linear→AddBias schedule inserted an additional BF16
rounding point not present in source `nn.Linear`. Its complete-tower test failed the
original gross-error gate despite every same-input block passing. The corrected
projection reuses the unchanged native BF16 MMA schedule with an FP32 bias epilogue;
the oracle computes the complete formula independently in FP64. There is no new tile
tuning or speed claim.

The composed FP64 reference follows public Op outputs as represented inputs to the next
Op. It does not copy private staging/reduction casts. The merger remains one closed ideal
formula without private casts. Full-tower input is a nonzero BF16 12-patch panel spanning
two independently attended images. The predeclared composition criterion is relative L2
0.02 and maximum error at most `0.005 + 0.02 * max(abs(reference))`.

RTX 5090, CUDA 13.1, `ninfer-builder-qwen4-mixed`, focused checks:

- Complete 27-block encoder: relative L2 0.005804, maximum error 32, gross bound 85.765.
- Complete merged output: relative L2 0.015751, maximum error 0.004622, gross bound 0.010946.
- All three individual visual tokens pass the same criterion; worst relative L2 0.017536.
- Graph replay, segment isolation, and packed-versus-separate first-image output are exact.
- Native closed-merger G=1/5/28/129 and 128+1: all existing gates pass; worst relative L2
  0.002651. The previous merger profile was approximately 0.00333.
- Biased-projection seven real geometries: independent complete-formula checks retain the
  native A16 relative L2 criterion 1/256 and gross criterion 1/256+(2/256)*max(reference),
  at T=1/5/128/129 with source/output guards and exact power-of-two column witnesses.

This panel establishes a runnable complete source-weight tower and catches accumulation,
layout, segment, and capture errors. It is not a multimodal benchmark, full-model PPL,
all-input numerical proof, or future-checkpoint default qualification.

## Pixel preparation and positions

`vision_frontend.{h,cpp}` supplies decoded-RGB and already-owned encoded-byte entrypoints.
It applies exact pinned image/video pixel bounds, RGB8 antialiased bicubic, source
normalization, temporal padding, `[channel,temporal,y,x]` patch flattening, 2x2 merge-major
ordering, and source video timestamps. The encoded-video route uses the existing decoder
with 2 fps, minimum 4 and maximum 768 frames. Four-row text/MRoPE positions and separate image/
video destination-column maps are prepared from explicit token-type runs and grids.
Tokenizer/template rendering and Engine integration are not provided by this component.

Shared pixel mathematics live in `src/media/prepare/pixels.{h,cpp}`. The existing Qwen3.6
frontend retains its previously qualified floating-coefficient arithmetic and target
resize policy. Qwen4's pinned RGB8 source uses double-constructed, quantized signed 16-bit
coefficients with integer half-up clipping after each pass. Using the previous floating
profile produced one-to-two RGB-code differences; the source integer profile closes that
gap without changing Qwen3.6's behavior. Actual FP32 normalization differs from the
independent reference by at most 5.914e-8, below the unchanged 1e-6 criterion.

Frontend reference cases cover an 83×121 image, three-frame 63×95 video with odd temporal
padding, and two-frame 17×25 video with small-image upscaling. Reference generation uses
PyTorch's CPU uint8 antialiased bicubic and independent tensor patch layout. The existing
Qwen3.6 frontend test passes; its two optional official-tokenizer subcases remain skipped
when that unrelated tokenizer is not supplied.

## Source authority

- NVIDIA artifact/processor revision `fc694b54fb0174e0913e6adf86691ef85a4ead47`:
  https://huggingface.co/nvidia/Qwen3.8-Flash-Next-NVFP4/tree/fc694b54fb0174e0913e6adf86691ef85a4ead47
- Transformer consumer revision `c119ec3cc37ab69642f39cca2de4187714002b08`, including
  Qwen4Exp Vision/model positions, `vision_utils`, Qwen2VL image processor and Qwen3VL
  video processor:
  https://github.com/huggingface/transformers/tree/c119ec3cc37ab69642f39cca2de4187714002b08/src/transformers
- RGB8 source coefficient/rounding implementation and reference executable PyTorch 2.11:
  https://github.com/pytorch/pytorch/blob/70d99e998b4955e0049d13a98d77ae1b14db1f45/aten/src/ATen/native/cpu/UpSampleKernel.cpp

Exact replay commands are the `--native-real` and `--frontend-real` modes of
`ninfer_qwen4_vision_test`, plus `ninfer_linear_bias_test` and the existing
`ninfer_vision_patch_merger_test --native-real`. Set `NINFER_QWEN4_NATIVE_LAYERS` to the
directory holding the exact Vision/source/reference artifacts. Optional
`NINFER_QWEN4_VISION_TRACE` writes patch-plus-27-block BF16 diagnostics; the independent
reference tool's `--trace` reports same-input versus propagated block errors.
