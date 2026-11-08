# NInfer maintainer tools

`tools/` contains the project-owner workflows for artifact conversion and inspection, independent
Python references, numerical parity diagnostics, benchmark orchestration, and serving smoke checks.
These tools are not part of the public download-and-run path; normal users should start with the
[project README](../README.md).

Run commands from the repository root with a Python 3.11 environment containing the dependencies
for the selected tool.

## Task index

| Task | Location |
|---|---|
| Extend or re-encode a Qwen3.8-27B NVFP4 artifact | [`convert/qwen3_8_27b/`](convert/qwen3_8_27b/), over the shared family helpers in [`convert/qwen3_6_27b/`](convert/qwen3_6_27b/) |
| Inspect artifact metadata and objects | [`artifact/inspect.py`](artifact/inspect.py) |
| Artifact-native Python Vision reference | [`reference/qwen3_6_27b/`](reference/qwen3_6_27b/) |
| Compare 27B artifact/source Vision activations | [`parity/qwen3_6_27b/`](parity/qwen3_6_27b/README.md) |
| Kernel iteration (bound / MMA probe / Op loop) | [`kdev/`](kdev/README.md) |
| Run benchmark matrices | [`bench/`](bench/README.md) |
| Exercise a resident HTTP server | [`smoke/serve_contract.py`](smoke/serve_contract.py) |
| Exercise thinking preservation through a managed server | [`smoke/serve_thinking_preservation.py`](smoke/serve_thinking_preservation.py) |

## Artifact workflow

The converters read an existing `qwen3.8-27b/nvfp4` artifact plus official local source
checkpoints and write one complete `.ninfer` artifact. The paths below are placeholders for the
maintainer's local checkpoint checkouts:

```bash
# Optional DFlash2 append onto an existing qwen3.8-27b/nvfp4 file (not the
# Qwen3.8-27B Text checkpoint). NVFP4 draft matrices + BF16 selector codebooks.
hf download Ostfralla/Qwen3.8-27B-NVFP4-NInfer \
  qwen3_8_27b_nvfp4.ninfer \
  --local-dir models
hf download z-lab/Qwen3.8-27B-DFlash2 \
  --revision 50307d4c4cde6860d4eee73e2547cd786fe8e8a4 \
  --local-dir /path/to/Qwen3.8-27B-DFlash2
python3 -m tools.convert.qwen3_8_27b.convert_nvfp4 \
  --base-artifact models/qwen3_8_27b_nvfp4.ninfer \
  --dflash-model /path/to/Qwen3.8-27B-DFlash2 \
  --dflash-format nvfp4 \
  --out out/qwen3_8_27b_nvfp4_dflash_nvfp4.ninfer \
  --device cuda
```

Inspect the result:

```bash
python3 -m tools.artifact.inspect out/qwen3_8_27b_nvfp4_dflash_nvfp4.ninfer --objects
```

The selective FP8 and NVFP4 MTP converters, exact source revisions, inventories, formats, and
conversion recipes are recorded in
[`docs/maintainer/qwen3.8-27b-artifact.md`](../docs/maintainer/qwen3.8-27b-artifact.md). Published users download the completed artifacts from
Hugging Face instead of running these workflows.

## Python references and parity

`reference/qwen3_6_27b/` holds the artifact-native Python Vision implementation and its artifact
binding. It is an independent diagnostic reference, not an alternate public inference product or a
generated-token golden for the C++ engine. The [parity README](parity/qwen3_6_27b/README.md)
gives the direct artifact/source Vision comparison command.

## Benchmark orchestration

`tools/bench/run_ninfer_bench_matrix.py` builds and runs the public-Engine benchmark matrix and
writes ignored local reports below `profiles/bench/`:

```bash
python3 tools/bench/run_ninfer_bench_matrix.py --preset core --dry-run
python3 tools/bench/run_ninfer_bench_matrix.py --preset core
```

See [`tools/bench/README.md`](bench/README.md) and [`bench/README.md`](../bench/README.md) for the
orchestrator and executable contracts.

## Serving smoke

After starting `ninfer-serve` in another terminal:

```bash
python3 -m tools.smoke.serve_contract \
  --base-url http://127.0.0.1:18080 \
  --model qwen3.8-27b
```

The client exercises OpenAI, Anthropic, streaming, usage, multimodal, and tool-call response
surfaces against the resident process.

For typed rewrite-checkpoint and thinking-history behavior, the managed smoke script launches a
real server and consumes the repository fixture:

```bash
python3 tools/smoke/serve_thinking_preservation.py \
  --artifact out/qwen3_8_27b_nvfp4_dflash_nvfp4.ninfer --backend dflash
```
