# Qwen3.8-27B NVFP4 Vision intermediate validation

This campaign checks every material numerical boundary in the production Vision schedule. The C++
tracer records represented preprocessing tensors, every patch-projection stage, all substeps of all
27 transformer blocks, and every merger stage. The Python driver then compares those 471 boundaries
per media item with the artifact-native implementation, compares preprocessing and MRoPE metadata
with the embedded Transformers processor, measures quantization drift against the source BF16 tower,
verifies all 222 direct Vision artifact tensors bit-exactly plus all 111 quantized matrices against
the source BF16 weights, and checks that the independent source schedule reaches the Hugging Face
final output. Quantized weights are reconstructed as exact FP32 signed-code × FP16-scale values;
whole-matrix, row, column, and every stored G64/G32 group are gated, including partial final K
groups. A second
artifact-native pass feeds every logical operation the captured C++ input from its immediately
preceding boundary, isolating local operator error from upstream propagation. Whole-tensor,
worst-token, and worst-feature criteria prevent large real shapes from diluting a dropped row or
strided-column defect. Fault-injection tests prove that isolated token, feature, and quantization-
group corruption cannot hide in aggregate metrics.

Build the diagnostic and run a real image campaign:

```bash
cmake --build build --target ninfer_qwen3_8_27b_vision_trace

python -m tools.parity.qwen3_6_27b.vision \
  --weights out/qwen3_8_27b_nvfp4.ninfer \
  --model-dir /path/to/qwen3.8-27b-bf16 \
  --messages examples/cli/messages/image_chart.json \
  --trace-exe build/tests/ninfer_qwen3_8_27b_vision_trace \
  --output profiles/bench/qwen3_8_vision_intermediates.json
```

The large artifact-reference activation dump is always temporary. The C++ trace is also temporary
unless `--trace-dir` is supplied. The JSON report retains every comparison metric and a summarized
pass/fail verdict without embedding model activations.
