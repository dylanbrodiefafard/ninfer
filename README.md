# NInfer — NVIDIA GeForce RTX 5090

Run **Qwen3.8-27B on one RTX 5090**, with up to **eight requests at once**.
NInfer is a C++/CUDA engine with kernels and CUDA Graphs tuned for `sm_120a`.
The recommended model combines [Ostfralla’s NVFP4 weights](https://huggingface.co/Ostfralla/Qwen3.8-27B-NVFP4-NInfer)
with a locally converted DFlash2 companion; setup is below.

## Features

- Terminal chat, OpenAI Chat Completions and Responses, and Anthropic Messages, with streaming
  and separate reasoning output.
- Tool calls, JSON Schema and EBNF constraints. Clients execute the tools.
- Images and video with `--vision`.
- DFlash2 speculative decoding with adaptive draft lengths.
- Conversation caching in GPU memory, RAM and SSD; `cache_write: false` for disposable turns.
- Sampling controls, token log probabilities, Prometheus metrics and JSON statistics.

Eight agents can work in parallel on one resident model. Additional requests wait in a bounded
FIFO queue; idle conversations can move to RAM and resume without processing their history
again. A twelve-session check verified cache reuse through six active lanes. Restore cost depends
on history size and host bandwidth; the throughput benchmarks below cover one, six, seven and eight requests.

Use **NVFP4 KV cache**, the default. W4A4 is used for prefill and some generation shapes;
generation also uses A16 and W4A8, with W4A8 used for most DFlash2 verification.
**MTP execution is disabled.** Existing MTP tensors may remain in artifacts.

Multi-GPU execution, active-context offload and preemptive scheduling are not supported.

## Performance and quality

Measured **2026-10-18 and 2026-10-21 on one RTX 5090**, Release build, CUDA Graphs, NVFP4 KV,
Vision off and a 600 W power cap. The DFlash2 artifact uses NVFP4 matrices and BF16 selector
codebooks, adaptive drafting up to seven tokens and the optimized proposal head.

The workloads match the AMD fork’s README: the same messages, thinking settings, prefill corpus,
warmup and repetition counts. Prefill uses this engine’s default 4,096-token chunk.

**Prompt processing (prefill), one request, 4,096-token chunks:**

| Prompt tokens | Average tokens/s |
|---:|---:|
| 8,192 | 15,385 |
| 32,768 | 12,294 |
| 65,536 | 9,652 |
| 131,072 | 6,747 |
| 245,760 | 4,433 |

Prefill uses a separate greedy token-corpus workload. Rates exclude model loading and request
preparation.

**Answer generation with p-less sampling at temperature 1.5:**

| Workload | Prompt tokens | 1 request, tokens/s | 6 requests, total tokens/s | 7 requests, total tokens/s | 8 requests, total tokens/s |
|---|---:|---:|---:|---:|---:|
| English story | 146 | 186 | 779 | 841 | 949 |
| Python package | 171 | 312 | 1,057* | — | — |
| Math reasoning | 346 | 287 | 970 | 1,033 | 1,150 |
| Document summary 8K | 8,192 | 228 | 815 | 892 | 981 |
| Document summary 32K | 32,779 | 226 | 718 | 803 | 856 |

Each request generates 512 tokens, including reasoning. Both tables report means of three runs
after one warmup. Generation rates exclude prompt processing: each prompt is cached before timing,
and each concurrent total counts only intervals with all C requests decoding. Divide by C for
the average per active request.

C6/C7/C8 use matched measurements from October 21. C1, prefill and the starred Python-package
C6 result are from October 18. Python-package retries on October 22 at C6, C7 and C8 all
triggered the benchmark's raw `<tool_call>` output guard; no fresh Python-package results
are reported.

Across the four matched workloads, C7 improves steady decode throughput by 6.5–11.9% over C6,
and C8 by 18.5–21.8%. Separate complete-wave measurements, including request initialization,
HTTP overhead and partial-batch tails, improve by 7.1–14.3% and 19.6–20.8%, respectively.

Prompt content and sampling affect draft acceptance and speed. These are throughput workloads;
the 512-token budget does not complete the long story and code tasks. Settings, variation and
reproduction commands are in the [performance guide](docs/performance.md#c7c8-aggregate-scaling-2026-10-21).
[Historical machine-readable results](bench/results/readme-benchmarks-20261018.json) preserve the
October 18 means, sample standard deviations and configuration; the performance guide records
the October 21 report locations.

**Historical accuracy: Ostfralla base artifact with MTP**

| KV cache | AIME 2025 | AIME 2026 | GPQA-Diamond |
|---|---:|---:|---:|
| NVFP4 | 93.33% (28/30) | 100.00% (30/30) | 92.42% (183/198) |

EvalScope 1.9.0, thinking enabled, MTP with three draft tokens, 262,144-token context, one sample
per problem, temperature 0.6, top-p 0.95, top-k 20, presence penalty 1.0 and seed 42. These are
single-sample scores, not pass@k or an evaluation of the DFlash2 speed configuration.

Separate exact-answer needle-retrieval tests passed both standard and multikey cases through
260K with NVFP4 KV. Test conditions, limitations, accuracy reproduction and historical results
are in [docs/performance.md](docs/performance.md).

## Build

Requires 64-bit Linux, an RTX 5090, a CUDA 13.1-compatible NVIDIA driver, CUDA Toolkit 13.1 or
newer, CMake ≥3.28 and a C++20 compiler. Only CUDA architecture `120a` is supported.

On Ubuntu 24.04, after installing the CUDA toolchain:

```bash
sudo apt-get install build-essential cmake ninja-build git patch pkg-config python3 \
  libavcodec-dev libavformat-dev libavutil-dev libswscale-dev \
  libcurl4-openssl-dev libssl-dev libzstd-dev

git clone --branch experimental https://github.com/dylanbrodiefafard/ninfer.git
cd ninfer

cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build --parallel 8
```

The executables are `build/apps/ninfer` and `build/apps/ninfer-serve`; run them from the build
tree. On other distributions, use FFmpeg development libraries (`libavformat`/`libavcodec` ≥60,
`libavutil` ≥58, `libswscale` ≥7), libcurl ≥7.85, OpenSSL (`libcrypto`) and zstd. CMake fetches pinned grammar dependencies.
Tests and benchmarks are opt-in targets.

## Prepare the recommended model

The conversion appends the [z-lab DFlash2 companion](https://huggingface.co/z-lab/Qwen3.8-27B-DFlash2)
as NVFP4 matrices with BF16 selector codebooks. Ostfralla’s main-model weights stay unchanged.

Download the base and pinned companion with the Hugging Face CLI:

```bash
hf download Ostfralla/Qwen3.8-27B-NVFP4-NInfer \
  qwen3_8_27b_nvfp4.ninfer --local-dir models

hf download z-lab/Qwen3.8-27B-DFlash2 \
  --revision 50307d4c4cde6860d4eee73e2547cd786fe8e8a4 \
  --local-dir models/Qwen3.8-27B-DFlash2
```

With Python 3.11, CUDA-enabled PyTorch, NumPy and safetensors, create the combined artifact:

```bash
python3.11 -m tools.convert.qwen3_8_27b.convert_nvfp4 \
  --base-artifact models/qwen3_8_27b_nvfp4.ninfer \
  --dflash-model models/Qwen3.8-27B-DFlash2 \
  --dflash-format nvfp4 --dflash-codebook bf16 \
  --out models/qwen3_8_27b_nvfp4_dflash_nvfp4.ninfer --device cuda
```

The combined artifact enables adaptive DFlash2 automatically, with up to seven draft tokens and
the optimized proposal head. The base alone runs without speculation. See the
[artifact guide](docs/maintainer/qwen3.8-27b-artifact.md) for conversion details.

## P-less and epsilon sampling

NInfer uses p-less sampling, which derives its cutoff from the model’s probability distribution,
together with an epsilon-sampling probability floor of 1/1024 and a small logit-precision margin. This replaces top-k, top-p, min-p and
presence/frequency penalties. The default temperature is **1.5**.

The working hypothesis is that sampling settings developed for BF16 may suit quantized models
less well because quantization changes the probability distribution. Limited local tests suggest
better answers, but this has not been established by a controlled evaluation.

Use `--temperature` to tune sampling, `--greedy` for argmax, or `--no-p-less-sampling` for
conventional filters and penalties. The [CLI guide](docs/cli.md) and `--help` list the controls.

Papers:

- [p-less Sampling: A Robust Hyperparameter-Free Approach for LLM Decoding](https://arxiv.org/abs/2509.23234) — Tan, Wu and Howard (2025).
- [Truncation Sampling as Language Model Desmoothing](https://aclanthology.org/2022.findings-emnlp.249/) — Hewitt, Manning and Liang (2022); introduces epsilon sampling.

### Loop recovery

Reasoning loops have been rare in local use; their frequency has not been systematically measured.
Recovery excludes the token continuing a detected reasoning cycle, preserves reasoning-end and
stop tokens, and leaves grammar-constrained tool-call output intact. Persistent repeated reasoning
or confirmed duplicate tool-call loops can trigger up to **two internal retries** within the
original output budget. Back-to-back identical calls are withheld when unchanged tool results
and repeated reasoning confirm an unproductive loop; legitimate repeats remain possible.

Retries preserve the prompt and real tool results, omit the failed attempt from the retry context,
and add feedback asking for progress. Already streamed text remains visible. If recovery fails after
two retries, the request returns an error. Long reasoning alone does not trigger recovery; internal
retries exclude media and raw output. See the [serving guide](docs/serving.md#function-tools) for the full conditions.

## Run

Generate an answer with the combined artifact:

```bash
build/apps/ninfer models/qwen3_8_27b_nvfp4_dflash_nvfp4.ninfer \
  --prompt "Explain prefill and decode in three sentences." \
  --max-context 16384 --max-new 256 --kv-dtype nvfp4
```

Answer text streams to stdout; reasoning, progress and statistics go to stderr. For chat history
or media, use `--messages FILE` instead of `--prompt`, and add `--vision` for images or video.
[Example inputs](examples/cli/) are included.

Start a server for eight active agents, with a RAM prefix cache for idle conversations:

```bash
build/apps/ninfer-serve models/qwen3_8_27b_nvfp4_dflash_nvfp4.ninfer \
  --host 127.0.0.1 --port 8080 \
  --max-context 262144 --max-concurrency 8 --kv-dtype nvfp4 \
  --kv-ram-capacity 8192 --pending-timeout-ms 900000
```

The RAM cache reserves 8 GiB of pinned host memory. The preparation and queue timeout is
15 minutes instead of the five-minute default, giving extra agents time to wait. Backend, Vision
and concurrency settings are fixed at startup. Use `--spec none` to disable speculation, or
`--spec dflash --draft-tokens N` to set a draft length explicitly.

From a second terminal:

```bash
curl http://127.0.0.1:8080/health
curl http://127.0.0.1:8080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"qwen3.8-27b","messages":[{"role":"user","content":"Say hello."}],"max_tokens":128}'
```

For OpenAI-compatible clients, set the base URL to `http://127.0.0.1:8080/v1` and the model to
`qwen3.8-27b`. Use `--model-id` to serve a different model alias.

## Context and caching

`--max-context` limits prompt and answer combined, up to **262,144 tokens per request**.
The default `--kv-capacity auto` sizes a shared token pool from free VRAM, leaving **64 MiB
headroom**. Active and retained histories share this pool; the server reports its capacity at
startup. Use `--kv-capacity N` for a fixed token budget.

Higher startup concurrency (`--max-concurrency C`) reserves more memory for request and draft
state, leaving less for the shared KV pool. Automatic capacity is capped at `C × --max-context`,
rounded to pages. The pool is shared rather than split into fixed per-request quotas; each request
still has its own `--max-context` ceiling.

Measured on the RTX 5090 with the recommended DFlash2 artifact, 262,144-token context,
4,096-token prefill chunks, adaptive drafting up to seven tokens, the optimized proposal head,
CUDA Graphs, NVFP4 KV and 64 MiB headroom. Vision and RAM/SSD tiers were off (2026-10-18).

| Startup concurrency (C) | Shared KV capacity (tokens, total) |
|---:|---:|
| 1 | 262,144 |
| 2 | 524,288 |
| 3 | 732,096 |
| 4 | 714,432 |
| 5 | 697,408 |
| 6 | 680,384 |

These are shared totals across active and retained histories. C1 and C2 reach the concurrency cap;
C3–C6 are limited by VRAM. Available memory and enabled features can change the result. See the
[capacity measurements](docs/performance.md#dflash2-automatic-kv-capacity) for commands and results.

The 64 MiB margin suits a **headless, dedicated GPU**. If the GPU also drives a desktop or runs
other workloads, increase `--kv-capacity-headroom MiB` to cover their peak memory use.
Shared-display and concurrent-load configurations have not been tested.

RAM and SSD caches retain idle history; they do not extend an active request beyond GPU capacity.
Both are off by default. Set `--kv-ram-capacity MiB` to fit available host RAM. Disk caching
requires the RAM tier; add `--kv-disk-capacity 25600 --kv-disk-location /path/to/ssd/ninfer-cache`
for 25 GiB of persistent storage. See the [cache guide](docs/maintainer/paged-kv-cache.md).

Keep system prefixes stable for persistent reuse. In harnesses such as OpenCode, remove an
unnecessary daily date from the initial prefix, or place it after stable history: a changed date
invalidates reuse after that point. NInfer does not remove dates itself.

## Docker

Requires Docker and the NVIDIA Container Toolkit on a Linux host with the supported GPU and
driver. Prepare the combined DFlash2 model in `models/`, then:

```bash
docker build --tag ninfer:local .
docker run --rm \
  --gpus '"device=0"' \
  --publish 127.0.0.1:8080:8080 \
  --volume "$PWD/models:/models:ro" \
  ninfer:local \
  ninfer-serve /models/qwen3_8_27b_nvfp4_dflash_nvfp4.ninfer \
  --host 0.0.0.0 --max-context 262144 --max-concurrency 6 --kv-dtype nvfp4
```

The same image can run `ninfer` for CLI requests.

## Documentation

- [CLI guide](docs/cli.md) — input, sampling, media, speculation and prefix storage.
- [HTTP serving](docs/serving.md) — APIs, tools, constraints, scoring, state and metrics.
- [Performance](docs/performance.md) — benchmarks, quality results, methods and limitations.
- [Documentation index](docs/README.md) — architecture, artifact formats and conversion.
- [Tests](tests/README.md) and [contributing](CONTRIBUTING.md) — development workflows.

## License

This fork of [Neroued/NInfer](https://github.com/Neroued/ninfer) is licensed under
[Apache-2.0](LICENSE). Model weights are supplied separately; their sources and conversion
provenance are in the artifact guides. Vendored dependencies retain their licenses in `third_party/`.
