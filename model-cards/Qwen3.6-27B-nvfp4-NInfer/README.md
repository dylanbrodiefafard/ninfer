---
library_name: ninfer
pipeline_tag: image-text-to-text
inference: false
license: apache-2.0
base_model:
  - Qwen/Qwen3.6-27B
  - rdtand/Qwen3.6-27B-PrismaSCOUT-Blackwell-NVFP4-BF16-vllm
base_model_relation: quantized
tags:
  - ninfer
  - qwen3.6
  - nvfp4
  - w4a4
  - blackwell
  - multimodal
  - conversational
  - cuda
  - rtx-5090
model-index:
  - name: Qwen3.6-27B-nvfp4-NInfer
    results:
      - task:
          type: text-generation
          name: Text Generation
        dataset:
          name: AIME 2025
          type: aime25
        metrics:
          - type: accuracy
            value: 93.33
            name: Accuracy (0-shot, rule)
        source:
          url: https://github.com/Neroued/ninfer/tree/master/eval
          name: NInfer EvalScope 1.9.0
      - task:
          type: text-generation
          name: Text Generation
        dataset:
          name: AIME 2026
          type: aime26
        metrics:
          - type: accuracy
            value: 93.33
            name: Accuracy (0-shot, rule)
        source:
          url: https://github.com/Neroued/ninfer/tree/master/eval
          name: NInfer EvalScope 1.9.0
      - task:
          type: text-generation
          name: Text Generation
        dataset:
          name: GPQA-Diamond
          type: gpqa_diamond
        metrics:
          - type: accuracy
            value: 84.34
            name: Accuracy (0-shot, rule)
        source:
          url: https://github.com/Neroued/ninfer/tree/master/eval
          name: NInfer EvalScope 1.9.0
---

# Qwen3.6-27B NVFP4 for NInfer

This model card is the version-controlled source for
[neroued/Qwen3.6-27B-nvfp4-NInfer](https://huggingface.co/neroued/Qwen3.6-27B-nvfp4-NInfer).

The repository contains the NVFP4 weight profile of
[Qwen3.6-27B](https://huggingface.co/Qwen/Qwen3.6-27B), converted from the fixed packed weights in
[rdtand/Qwen3.6-27B-PrismaSCOUT-Blackwell-NVFP4-BF16-vllm](https://huggingface.co/rdtand/Qwen3.6-27B-PrismaSCOUT-Blackwell-NVFP4-BF16-vllm)
to the native [NInfer](https://github.com/Neroued/ninfer) `.ninfer` artifact format. The artifact is
intended only for NInfer; it is not a Transformers checkpoint, Safetensors distribution, or GGUF
file.

This is a second weight profile for the existing `qwen3_6_27b` target, not a separate model target.
The version-2 artifact identity selects the NVFP4 binder and execution leaves. NInfer uses W4A4
Tensor Core MMA for prefill and A16 NVFP4 kernels for decode while retaining the same Text, Vision,
MTP, prefix-reuse, CLI, and serving paths as the groupwise-int profile.

## Artifact

| Field | Value |
|---|---|
| Filename | `qwen3_6_27b_nvfp4.ninfer` |
| Size | 18,324,064,000 bytes (17.07 GiB) |
| SHA-256 | `bce5f00d066c0f20f1317bf1fdcb458264cf95837c3b1f3fbec163694627893a` |
| Container version | 2 |
| NInfer model ID | `qwen3.6-27b` |
| NInfer weights ID | `nvfp4` |
| NInfer target key | `qwen3_6_27b` |
| Stored objects | 1,307 (1,301 tensors and 6 resources) |
| NVFP4 tensors | 247 |

The file contains the registered Text, Vision, MTP, proposal-head, tokenizer, chat-template,
generation, and media-processor objects required by NInfer. Text linears use the source repository's
fixed mixed NVFP4/BF16 allocation; Vision, MTP, and frontend resources retain their registered
NInfer formats.

Verify a downloaded file with:

```bash
printf '%s  %s\n' \
  'bce5f00d066c0f20f1317bf1fdcb458264cf95837c3b1f3fbec163694627893a' \
  'qwen3_6_27b_nvfp4.ninfer' | sha256sum --check
```

## Requirements

- [NInfer](https://github.com/Neroued/ninfer) revision
  [`a85109c`](https://github.com/Neroued/ninfer/commit/a85109c4ef16b0b4a218c61e4641cfd0d11320fe)
  or later, built from source;
- 64-bit Linux;
- NVIDIA GeForce RTX 5090 (`sm_120a`);
- CUDA Toolkit 13.1 or newer.

NInfer does not provide an install target or packaged binary. See the
[repository README](https://github.com/Neroued/ninfer#build) for source-build dependencies.

## Download and run

```bash
hf download neroued/Qwen3.6-27B-nvfp4-NInfer \
  qwen3_6_27b_nvfp4.ninfer \
  --local-dir models

./build/apps/ninfer models/qwen3_6_27b_nvfp4.ninfer \
  --prompt "Explain prefill and decode in three sentences." \
  --max-context 16384 \
  --max-new 256 \
  --spec mtp --draft-tokens 3 \
  --lm-head-draft
```

For images, videos, structured chat history, and HTTP serving, see the
[NInfer documentation](https://github.com/Neroued/ninfer/tree/master/docs).

## Supported use

The artifact supports:

- text generation in thinking and non-thinking modes;
- image, multi-image, video, and mixed multimodal messages;
- MTP speculative decoding with draft windows from one to five;
- BF16 and INT8 group-64 KV cache;
- CUDA Graph decode and compatible-prefix reuse;
- startup-bounded small-scale concurrent serving with true batched decode;
- the NInfer CLI;
- OpenAI Chat Completions and Anthropic Messages serving.

## Performance

The following single-GPU serving measurements were collected on an NVIDIA GeForce RTX 5090 with
CUDA 13.1 compile/runtime and CUDA driver API 13.3. Requests were submitted serially to a persistent
`ninfer-serve` process with CUDA Graph enabled, a 1,024-token prefill chunk, INT8 group-64 KV cache,
and prefix reuse disabled. Each value is the arithmetic mean ± sample standard deviation over five
fixed seeds; warm-up requests are excluded.

### Long-context baseline (MTP disabled)

| Prompt tokens | Prefill tok/s | Server TTFT (ms) | Decode tok/s |
|---:|---:|---:|---:|
| 7,680 | 11,191.5 ± 70.2 | 692.5 ± 4.3 | 86.4 ± 0.5 |
| 64,512 | 6,298.5 ± 97.6 | 10,288.6 ± 159.3 | 78.0 ± 1.2 |
| 130,048 | 4,204.7 ± 14.1 | 31,012.5 ± 104.6 | 71.2 ± 0.2 |
| 260,096 | 2,510.6 ± 16.8 | 103,761.1 ± 698.8 | 59.9 ± 0.3 |

At 7,680 prompt tokens this is 3.48× the prefill throughput of the published groupwise-int profile;
at 260,096 prompt tokens it is 1.55×.

### MTP=3 long-reasoning decode

Thinking was enabled and the output limit was 65,536 tokens.

| AIME 2026 fixture | Completion tokens | Decode tok/s | MTP acceptance | MTP tokens/round |
|---|---:|---:|---:|---:|
| Problem 1 | 11,717.0 ± 476.7 | 222.7 ± 3.4 | 80.8% ± 1.8% | 3.43 ± 0.06 |
| Problem 15 | 65,536.0 ± 0.0 | 201.6 ± 2.9 | 74.7% ± 1.5% | 3.24 ± 0.04 |
| Problem 30 | 46,439.2 ± 3,719.0 | 216.3 ± 1.5 | 80.8% ± 0.9% | 3.42 ± 0.03 |

### MTP=3 cross-scenario decode

Each category contains three fixtures and five seeds per fixture (15 samples). Thinking was
disabled and the output limit was 4,096 tokens.

| Category | Decode tok/s | MTP acceptance | MTP tokens/round |
|---|---:|---:|---:|
| Code | 211.7 ± 7.0 | 74.2% ± 3.6% | 3.23 ± 0.11 |
| Story | 144.5 ± 10.5 | 40.0% ± 5.4% | 2.20 ± 0.16 |
| Translation | 205.2 ± 13.9 | 70.4% ± 7.1% | 3.11 ± 0.21 |
| Structured output | 243.1 ± 15.3 | 90.2% ± 7.8% | 3.71 ± 0.24 |

See the
[full methodology and results](https://github.com/Neroued/ninfer/blob/master/docs/performance.md),
including metric definitions, comparison data, and the exact reproduction command.

## Evaluation

The artifact was evaluated through NInfer's OpenAI-compatible serving route with thinking enabled,
MTP=3, and a 262,144-token context limit. EvalScope 1.9.0 used 0-shot prompts, rule-based scoring,
and one sample per problem with temperature 0.6, top-p 0.95, top-k 20, presence penalty 1.0, and
seed 42. All 258 configured samples completed and were scored.

| Benchmark | Accuracy | Correct / total |
|---|---:|---:|
| AIME 2025 | 93.33% | 28 / 30 |
| AIME 2026 | 93.33% | 28 / 30 |
| GPQA-Diamond | 84.34% | 167 / 198 |

These are single-sample results under the stated NInfer evaluation profile, not pass@k scores.

## Limits

- The artifact is accepted only by NInfer revision `a85109c` or later and the matching registered
  target.
- NInfer executes on one RTX 5090 and one CUDA device, with a startup-fixed capacity of 1–8 active
  requests per Engine.
- It does not provide large-scale or preemptive continuous batching, priority/QoS scheduling,
  multi-GPU execution, CPU/GPU offload, or distributed serving.
- Context allocation is subject to GPU memory and the selected KV-cache type.
- NInfer does not execute generated tool calls.

## Provenance

| Field | Value |
|---|---|
| Base repository | `Qwen/Qwen3.6-27B` |
| Base revision | `6a9e13bd6fc8f0983b9b99948120bc37f49c13e9` |
| NVFP4 source repository | `rdtand/Qwen3.6-27B-PrismaSCOUT-Blackwell-NVFP4-BF16-vllm` |
| NVFP4 source revision | `9b5389d4a1e207daab2d47732efea57d7e946dcf` |
| Conversion recipe | `qwen3_6_27b_nvfp4-v1` |
| Converter repository | `https://github.com/Neroued/ninfer` |
| Converter revision | `a85109c4ef16b0b4a218c61e4641cfd0d11320fe` |
| Minimum runtime revision | `a85109c4ef16b0b4a218c61e4641cfd0d11320fe` |
| Ranking input SHA-256 | `c692dc76388132c910547589b4fb4a0503fbd6ad50aaac6a509bbcb192a8afa5` |

The artifact identity, summarized object inventory, and conversion provenance are published in
[`artifact-manifest.json`](https://huggingface.co/neroued/Qwen3.6-27B-nvfp4-NInfer/blob/main/artifact-manifest.json).
The exact storage contract is maintained in the
[Qwen3.6-27B artifact reference](https://github.com/Neroued/ninfer/blob/master/docs/maintainer/qwen3.6-27b-artifact.md).

## License

This NInfer artifact is distributed under the Apache License 2.0. The
[Qwen3.6-27B](https://huggingface.co/Qwen/Qwen3.6-27B) base repository and the
[NVFP4 source repository](https://huggingface.co/rdtand/Qwen3.6-27B-PrismaSCOUT-Blackwell-NVFP4-BF16-vllm)
are also licensed under Apache-2.0. Users remain responsible for complying with the license and
applicable laws.
