# NInfer AMD — Radeon AI PRO R9700

C++/HIP inference for **Qwen3.8-27B on one AMD Radeon AI PRO R9700**, with **1–16 concurrent requests** (1–8 with DFlash2).

An AMD port of [Neroued/NInfer](https://github.com/Neroued/ninfer), distributed under Apache-2.0. Model weights are separate; see `docs/acknowledgements.md` for credits.

## What it does

- **Works with chat clients.** Run from the terminal or connect a client that uses OpenAI Chat Completions, Responses or Anthropic Messages. Answers stream as they are generated, with reasoning kept separate from the answer.
- **Calls tools and returns structured answers.** Let the model choose a tool or require a specific one, with incremental tool-call streams. Constrain answers to JSON, a JSON Schema, an EBNF grammar, a full-match regex or literal choices.
- **Reads images and video.** Add `--vision` at startup to accept media alongside text.
- **DFlash2 decoding.** A small draft model proposes tokens for the main model to check together. Draft length adapts automatically, up to seven tokens. Use DFlash rather than MTP.
- **Reuses conversation history.** Cached prefixes avoid processing the same text again. Keep them in GPU memory, add a RAM cache, or save them to SSD for reuse after a restart.
- **Controls generation and scores text.** Adjust sampling, choose greedy decoding, inspect token probabilities or compare candidate answers. `ninfer-ppl` measures how well the model predicts text.
- **Reports server activity.** Prometheus metrics, JSON statistics and request logs report throughput, memory use, cache hits and draft acceptance.

## Built for the R9700

HIP kernels and Device Graphs are tuned for `gfx1201` (wave32). The engine batches up to sixteen requests with MTP or without speculation, and up to eight with DFlash, where it can process a new prompt while other requests generate. GPU memory is planned at startup, with workspace shared between execution phases.

**LQ4: 4-bit codebook weights decoded to FP8 for computation.** Most Text projections and the output head use LQ4; 21 sensitive projections stay in FP8. The DFlash-enabled artifact is **16.75 GB** on disk (decimal; this is not a VRAM requirement), with identity `qwen3.8-27b/r9700-lq4`. The paged KV cache stores FP8 keys, INT4 values and FP16 scales. DFlash keeps its own BF16 state.

For image/video inference, select the **17.42 GB** `qwen3.8-27b/r9700-lq4-vision-bf16` artifact and add `--vision`. It keeps the admitted Text weights byte-exact and uses original BF16 Vision matrices, adding about 640 MiB to storage and Vision-enabled weight memory. Vision-off loading has the same memory plan. Conversion and accuracy/performance evidence are in `docs/maintainer/qwen3.8-27b-artifact.md`.

This port combines NInfer features with AMD-specific kernels, weight/cache formats and memory tuning. Port decisions: `docs/maintainer/upstream-sync.md`.

## Context and capacity

Each request can use up to **262,144 tokens** for prompt and answer combined, subject to the shared GPU cache capacity below.

The following shared cache capacities were measured with automatic sizing, adaptive DFlash up to seven draft tokens, the optimized proposal head, chunk 2048, Vision off and Compose graph settings (HIP 7.15.26333; measured 2026-10-17):

| Concurrent requests (C) | Shared KV capacity (tokens) |
|---:|---:|
| 1 | 262,144 |
| 2 | 524,288 |
| 3 | 633,024 |
| 4 | 624,896 |
| 5 | 616,832 |
| 6 | 608,704 |
| 7 | 600,576 |
| 8 | 592,512 |

C9..16 runs MTP or no speculation. Shared KV tokens with MTP K5 adaptive, no RAM/SSD cache and Compose graph settings: 574,016 / 564,032 / 554,048 / 543,936 / 533,952 / 522,816 / 511,488 / 500,032 at C9..16; C16 without speculation resolves 608,704.

Capacity depends on available VRAM and enabled features; automatic sizing leaves 64 MiB headroom by default. RAM/SSD caches retain prefixes, not active-context offload. Test conditions and settings: `docs/performance.md`.

## Performance and quality

Measured 2026-10-17 on one R9700, Release build, Device Graphs, power `auto`, chunk 2048 and the optimized DFlash proposal head.

**Prompt processing (prefill), one request:**

| Prompt tokens | Average tokens/s |
|---:|---:|
| 8,192 | 3,375 |
| 32,768 | 2,903 |
| 65,536 | 2,477 |
| 131,072 | 1,907 |
| 245,760 | 1,354 |

**Answer generation with the default sampling settings:**

| Workload | Prompt tokens | C1 tokens/s | C6 aggregate tokens/s |
|---|---:|---:|---:|
| English story | 146 | 88 | 356 |
| Python package | 171 | 135 | 535 |
| Math reasoning | 346 | 126 | 491 |
| Document summary 8K | 8,192 | 93 | 340 |
| Document summary 32K | 32,779 | 97 | 300 |

P-less temperature 1.5, adaptive DFlash up to seven drafts, 512 output tokens per request (including reasoning). Means of three runs after one warmup.

Generation rates exclude prompt processing. Each prompt is cached before timing; C6 counts only intervals with all six requests decoding. Divide C6 by six for the average per active request.

Prompt content and sampling affect draft acceptance and speed. Prefill above uses a separate greedy token-corpus workload. Methodology and variation: `docs/performance.md`. Machine-readable results: `bench/results/readme-workloads-20261017.json`.

In every tested 8K/32K cell, the increase in prediction loss over the BF16 source was smaller than for the fixed 5090 NVFP4 reference. This compares next-token prediction loss, not answer quality or speed against another GPU. Standard and multikey needle retrieval passed through 240K; multikey also passed at 260K. Test conditions and limitations are in `docs/performance.md`.

## Build

On Ubuntu 24.04, after installing the supported ROCm 10 toolchain:

```sh
sudo apt-get install build-essential cmake ninja-build git patch pkg-config python3 curl \
  libavcodec-dev libavformat-dev libavutil-dev libswscale-dev \
  libcurl4-openssl-dev libzstd-dev

git clone https://github.com/jmander11/ninfer-amd-r9700.git
cd ninfer-amd-r9700
```

Requires 64-bit Linux, the R9700, ROCm 10 with `gfx1201` and rocprofiler SDK ROCTx, CMake ≥3.28, Ninja, C++20, Python 3, pkg-config, and FFmpeg, libcurl and zstd development libraries. CMake fetches a pinned XGrammar dependency. Python conversion/reference tools additionally need Python 3.11, PyTorch and safetensors.

```sh
cmake -S . -B build-r9700 -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DNINFER_BUILD_APPS=ON \
  -DNINFER_BUILD_BENCHMARKS=ON
cmake --build build-r9700 --parallel 8 \
  --target ninfer ninfer-serve ninfer-ppl ninfer_bench
```

Only `gfx1201` is supported; Ninja compile/link concurrency is capped at 8 jobs. Model artifacts are supplied separately. **A public model download is not yet available.** The commands below require the selected converted artifact; `docs/model-artifact.md` records its identity, checksum and conversion references.

## Run

Use the selected DFlash-enabled production `.ninfer` artifact:

```sh
build-r9700/apps/ninfer /path/to/qwen3.8-27b-r9700-lq4.ninfer \
  --prompt "Explain wave32 matrix instructions." \
  --max-context 8192 --max-new 256

build-r9700/apps/ninfer-serve /path/to/qwen3.8-27b-r9700-lq4.ninfer \
  --host 127.0.0.1 --port 8080 \
  --max-context 262144 --max-concurrency 6
```

CLI and server default to DFlash with the optimized draft head and adaptive drafting up to seven tokens. `--spec none` disables speculation; `--no-lm-head-draft` selects the full proposal head. Server KV capacity defaults to `auto`. Use `--draft-tokens N --no-adaptive-draft` for a fixed length.

P-less sampling uses temperature **1.5** by default; `--temperature` overrides it. It ignores top-p/top-k/min-p and penalties; `--no-p-less-sampling` enables those controls. Use `--greedy` for argmax. Vision and speculation settings are fixed at startup. Each executable's `--help` lists all options.

To check the server from a second terminal:

```sh
curl http://127.0.0.1:8080/health
curl http://127.0.0.1:8080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"qwen3.8-27b","messages":[{"role":"user","content":"Say hello."}],"max_tokens":128}'
```

For a localhost container, copy `.env.example` to `.env` and set your model, ROCm and self-contained Python paths, then:

```sh
mkdir -p .cache/ninfer
docker compose -f compose.yaml -f compose.local.yaml up -d --wait
```

This publishes `127.0.0.1:8080` and needs Compose ≥2.24.4. Defaults: C6, adaptive DFlash up to K7, temperature 1.5, 262,144-token context, 8 GiB RAM prefix cache and 25 GiB persistent disk cache. `docs/containers.md` covers prerequisites and the separate external-proxy deployment.

## Documentation

- `docs/cli.md` — input, sampling, media, speculation and prefix storage.
- `docs/serving.md` — HTTP schemas, tools, constraints, scoring, state and metrics.
- `docs/containers.md` — container prerequisites, deployment and verification.
- `docs/performance.md` — capacity, benchmarks, quality gates and limitations.
- `docs/README.md` — architecture, artifact formats, conversion and kernel development.
