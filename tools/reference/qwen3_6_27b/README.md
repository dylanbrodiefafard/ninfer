# Qwen3.8-27B Python Vision reference

This is the target-private, artifact-native Vision reference over a `qwen3.8-27b/nvfp4` `.ninfer`
artifact. It is independent of the C++ Engine implementation and has one consumer: the Vision
intermediate validation campaign in [`tools/parity/qwen3_6_27b/`](../../parity/qwen3_6_27b/README.md).

- `bindings.py`: `VisionArtifactBinding` validates the artifact identity, the six frontend
  resources, and the complete 333-tensor Vision inventory, then exposes typed Vision blocks. Text,
  MTP, draft-head, and DFlash2 objects are not inspected.
- `weights.py`: `WeightStore` decodes direct Vision controls once and streams each quantized
  matrix from the artifact mapping, dequantizing it on use.
- `vision.py`: `VisionEncoder` runs the fixed 27-block Vision schedule and the merger, reporting
  every Op boundary through an optional tap.
- `config.py` and `ops.py`: the fixed Vision dimensions and the BF16-boundary `linear` and
  `residual_add` operators. The remaining Vision operators, the frontend, and MRoPE prompt
  metadata live in `tools/reference/qwen3_6/common/`.

The frontend does not need the original Hugging Face checkpoint: `Frontend` materializes the
tokenizer, chat template, and image/video processor resources embedded in the artifact, then
delegates preprocessing to Transformers.

Install the dependencies from `requirements.txt`; the parity README gives the campaign command.
