# Code-quality gates

This is the authority for NInfer's static and dynamic quality tooling: what each gate catches,
where its configuration and version pin live, how to run it, and how a finding may be
suppressed. The expectations these gates serve are stated in [`AGENTS.md`](../../AGENTS.md)
("Code and comment quality"). Every tool is open source and version-pinned, so a finding is a
property of the code, not of the contributor's machine.

## Gates

| Gate | Catches | Configuration and pin | Run |
|---|---|---|---|
| Compiler diagnostics, `-Werror` | shadowing, missing virtual destructors, hidden overloads, unannotated fallthrough, format mismatches, float-to-double promotion, duplicated conditions/branches, member-initializer order, default-stream kernel launches, `this`-capturing extended lambdas | [`cmake/warnings.cmake`](../../cmake/warnings.cmake); toolchain pinned by the builder image (GCC 13, CUDA 13.1) | every build |
| clang-tidy 22 | use-after-move, unchecked `optional` access, dangling references, implicit widening of `int` index products in host code, slicing, exception escape from destructors and `noexcept`, empty catch, needless copies and no-op moves, path-sensitive analyzer findings | [`.clang-tidy`](../../.clang-tidy); version in [`scripts/run-clang-tidy.py`](../../scripts/run-clang-tidy.py) and the Dockerfile build stage | `./scripts/run-clang-tidy.py` (whole tree, zero findings) |
| clang-format 22 | C++/CUDA layout drift | [`.clang-format`](../../.clang-format); version in [`.pre-commit-config.yaml`](../../.pre-commit-config.yaml) | pre-commit |
| ruff (lint + format) | undefined names, late-binding closures, silent `zip` truncation, missing commas in string lists, unchecked `subprocess`, naive datetimes, dead code, import order | [`ruff.toml`](../../ruff.toml); version in `.pre-commit-config.yaml` | pre-commit |
| shellcheck | unquoted expansions, undeclared locals leaking to global scope, dead variables | version in `.pre-commit-config.yaml` | pre-commit |
| typos | misspelled identifiers, comments, and documentation | [`_typos.toml`](../../_typos.toml); version in `.pre-commit-config.yaml` | pre-commit |
| File hygiene | merge markers, accidentally committed large files or private keys, broken JSON/TOML/YAML, CRLF, trailing whitespace, missing final newline, shebang/executable-bit mismatch | `.pre-commit-config.yaml` | pre-commit |
| compute-sanitizer | device out-of-bounds and misaligned access, shared-memory races, illegal barrier use, reads of uninitialized device memory | CUDA 13.1 toolkit | `./scripts/run-unit-tests.sh --compute-sanitizer TOOL` |
| AddressSanitizer / UBSan | host heap/stack overflow, use-after-free, leaks, signed overflow, misaligned and invalid casts | `NINFER_SANITIZE` in `cmake/warnings.cmake` | separate tree, below |

No hosted CI runs these gates. They are enforced in two tiers:

| When | What runs | How |
|---|---|---|
| Every commit, on any branch | formatters, ruff, shellcheck, typos, file hygiene on the staged files | the commit hook, automatically |
| Often while working | the `-Werror` build, the fast unit tests, clang-tidy on changed lines | `./scripts/run-unit-tests.sh --fast`, `./scripts/run-clang-tidy.py --changed` |
| When the change requires them | the slow unit tests; the Engine tests for Engine, runtime, cache, speculative, or serving changes; sanitizers for changed kernels | `./scripts/run-unit-tests.sh`, `--real`, `--compute-sanitizer TOOL` |
| Every merge into `master` | every gate in this document, all passing on the merged commit | [Merging into master](merging-to-master.md) |

## Running the gates

Set up the commit hook once per checkout:

```bash
python3 -m pip install pre-commit==4.6.2   # the hook runner (Python 3.11); `prek` also works
git config core.hooksPath .githooks        # ./scripts/dev-setup.sh does this for you
```

The hook ([`.githooks/pre-commit`](../../.githooks/pre-commit)) runs the tools in
`.pre-commit-config.yaml` on the staged files. Formatters and `ruff --fix` rewrite files; the hook
stages those rewrites so they are part of the commit, runs the checks once more, and fails only
when a check still fails (a lint finding with no automatic fix, a misspelling, a shell error, a
broken JSON/TOML/YAML file). A file that also has unstaged edits is not re-staged, because that
would commit those edits too; stage or stash them and commit again. Committing with `--no-verify`
bypasses a gate and is not accepted. `pre-commit run --all-files` runs the same tools over the
whole tree.

clang-tidy is not in the commit hook: it needs the running builder and a built tree, and a widely
included header re-checks every includer. Run it on changed lines while working, and on the whole
tree before a merge into `master`.

Compiler diagnostics need no separate step: every project C++ and CUDA translation unit builds
with `NINFER_WARNINGS_AS_ERRORS=ON` by default. The policy applies only to NInfer targets;
third-party sources keep their upstream flags.

### Opt-in dual-card development container

Work on the Qwen3.8-Flash-Next second-GPU expert helper (RTX 5090 plus Radeon AI PRO R9700,
`gfx1201`) needs CUDA and HIP toolchains in one container. It has its own entry point, which
leaves the CUDA-only `scripts/dev-setup.sh`, the `Dockerfile` stages, the
`local/ninfer-builder:5090` image and the CMake defaults unchanged:

```bash
./scripts/dev-setup.sh            # builds the CUDA builder image it derives from
./scripts/dev-setup-dual-gpu.sh
```

It builds `docker/dual-gpu.Dockerfile` as `local/ninfer-builder:dual-gpu`, copying the host ROCm
tree (default `/opt/rocm/core-10.0`, override `NINFER_DUAL_ROCM_DIR`) into the image through the
named build context `rocm`, so the container toolchain does not change with later host-tree
updates. The HIP compiler is `/opt/rocm/core-10.0/lib/llvm/bin/clang++` with
`--offload-arch=gfx1201`. The `ninfer-builder-dual-gpu` container adds `/dev/kfd`, `/dev/dri`
and the host `render` group to the CUDA builder's devices, mounts the checkout at `/src` and its
own `ninfer-dual-gpu-build-cache` volume at `/build`, and mounts `NINFER_MODELS_DIR` at `/models`
and `NINFER_DATA_DIR` at `/data` read-only when they are set. The script header lists the other
overrides. An existing container keeps its original image, mounts and environment until it is
recreated.

The script requires `NINFER_DUAL_HELPER_ROCM_UUID`, the R9700's ROCm UUID: the `Uuid` of the
`gfx1201` agent in `rocminfo` (`GPU-` followed by hex digits; the host's iGPU has none). The
container sets `ROCR_VISIBLE_DEVICES` to it and `HIP_VISIBLE_DEVICES=0`, so HIP sees exactly the
R9700. The HIP runtime also treats `CUDA_VISIBLE_DEVICES` as a device filter, so without
`HIP_VISIBLE_DEVICES` a `CUDA_VISIBLE_DEVICES` naming the NVIDIA primary hides the R9700 and
helper startup fails. Keep those variables when running NInfer in that container.

Configure the optional feature explicitly in that container (the default remains OFF):

```bash
docker exec ninfer-builder-dual-gpu cmake -S /src -B /build/helper -G Ninja \
  -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTING=ON -DNINFER_WARNINGS_AS_ERRORS=ON \
  -DNINFER_ENABLE_HIP_EXPERT_HELPER=ON
docker exec ninfer-builder-dual-gpu cmake --build /build/helper --parallel 8
```

`--expert-helper-pci` is recognized by both product parsers. Host-only startup admission rejects
an OFF build and non-Flash-Next identity without loading weights or initializing the helper.
ON binaries without the option retain the ordinary CUDA schedule and allocate no helper
exchange/panels. The helper's startup diagnostics include selected PCI/name/gfx target,
HIP/toolchain, actual memory use, record inventory, compiled-profile share and checksum/nonce
validation. Enabled inference qualification and speed results remain pending.

The ON-only `ninfer_qwen3_8_flash_next_helper_real_test` runs the public Engine's helper smoke
with two repeated requests and consecutive decode rounds. Set
`NINFER_QWEN3_8_FLASH_NEXT_WEIGHTS`, `NINFER_QWEN3_8_FLASH_NEXT_NGRAM_TABLE` and
`NINFER_QWEN3_8_FLASH_NEXT_HELPER_PCI` explicitly; absent prerequisites skip with code 77. It
requires both devices and is not a CPU or R9700-only test. The target parity driver also accepts
`--expert-helper-pci`, including the public-Engine `lanes-bench` path; CUDA-only cache traces and
`--calib-counts` cannot be combined with helper mode.

CUDA memcheck/racecheck do not instrument HIP. Keep the independent HIP oracle, exact upload and
wire-visibility tests distinct from CUDA sanitizer evidence. The pinned clang-tidy 22 cannot
parse the ROCm-23 `__builtin_amdgcn_is_invocable` headers; matching ROCm clang-tidy analysis is
supplementary, not a pinned-gate pass. Do not suppress or disable checks to hide this limitation.

### Running clang-tidy

clang-tidy runs in the builder container against `/build/compile_commands.json`:

```bash
./scripts/run-clang-tidy.py                      # whole tree: the gate, zero findings
./scripts/run-clang-tidy.py --changed            # lines changed vs. origin/HEAD, while iterating
./scripts/run-clang-tidy.py --changed HEAD       # uncommitted changes only
./scripts/run-clang-tidy.py src/serve/foo.cpp    # named files; a header selects its includers
```

The runner rewrites nvcc commands into clang CUDA commands for `sm_120a`, so kernels, launchers, and
host code share one configuration. `--changed` analyzes every translation unit that is or includes a
changed file and reports only diagnostics on changed lines; it is the fast check while iterating.
The whole tree is clean, and the full run is the gate: it must report zero findings, because a
change can expose a path-sensitive analyzer finding on a line it did not touch. The two widening
checks apply to host translation units only: device index arithmetic is 32-bit by design and bounded
by each kernel's shape contract, which its host launcher validates under those checks. Header
changes are mapped to their includers through Ninja's dependency log, so build the tree first.

Runtime checkers are opt-in because they are slow:

```bash
./scripts/run-unit-tests.sh --compute-sanitizer memcheck  -L sanitizer   # the sanitizer set
./scripts/run-unit-tests.sh --compute-sanitizer initcheck -L sanitizer   # also synccheck
./scripts/run-unit-tests.sh --compute-sanitizer racecheck -L racecheck   # its racecheck part
./scripts/run-unit-tests.sh --compute-sanitizer racecheck -R ninfer_linear_nvfp4_a4_test

cmake -S . -B build-asan -G Ninja -DCMAKE_BUILD_TYPE=RelWithDebInfo -DBUILD_TESTING=ON \
  -DNINFER_SANITIZE=address,undefined
cmake --build build-asan
ASAN_OPTIONS=protect_shadow_gap=0 ctest --test-dir build-asan -E '_real_test$'
```

An instrumented tree reports compiler warnings without failing, because ASan instrumentation makes
GCC's flow-based warnings report paths that do not exist; the normal tree is the warning gate.
`--compute-sanitizer` changes only the CTest launcher (`NINFER_TEST_LAUNCHER`), so switching
tools does not recompile. CUDA's runtime maps memory inside ASan's shadow gap, hence
`protect_shadow_gap=0`. Run memcheck and racecheck for new or changed kernels with shared-memory
staging, asynchronous copies, or warp-level synchronization, and initcheck for new workspace or
arena consumers.

The `sanitizer` label is the set the compute-sanitizer gates run: the Ops the qwen3.8-27b NVFP4
DFlash2 flow dispatches. Suites whose full oracle sweep is too slow or too large under
compute-sanitizer contribute `--sanitizer` cases instead, one launch per route at real model
widths; registered as `<test>_sanitizer`, they also run in the ordinary unit suite, so the mode
cannot rot. `racecheck` labels the part of the set that is also quick under racecheck, which
tracks every shared-memory access and takes 5 to 50 minutes on each GEMM-heavy linear or
projection suite. Run a suite outside these labels by name when its kernels change. The rule for
keeping the set complete is in [Op development](op-development.md#64-sanitizer-scope).

## Suppressions

A finding is fixed unless the flagged code is correct and the reason can be stated. Suppress the
single instance, at the line, with the reason:

- clang-tidy: `// NOLINT(check-name): reason` or `// NOLINTNEXTLINE(check-name): reason`;
- GCC: `#pragma GCC diagnostic push` / `ignored "-W..."` / `pop` around the smallest region,
  with a comment;
- nvcc front end: `#pragma nv_diag_suppress <number>` / `nv_diag_default`, using the number the
  build prints;
- ruff: `# noqa: CODE  reason`; typos: an entry in `_typos.toml` for real vocabulary.

Bare `NOLINT`, `# noqa` without a code, file-wide suppressions, and removing a check or warning
to make a change pass are not accepted. A check that is wrong for the whole codebase is removed
in `.clang-tidy`, `cmake/warnings.cmake`, or `ruff.toml` with the reason recorded there.

## Changing a tool or its version

Bump the pin, run the tool over the whole tree, and commit the mechanical result separately from
behavioral changes; add a reformatting commit to [`.git-blame-ignore-revs`](../../.git-blame-ignore-revs)
(`git config blame.ignoreRevsFile .git-blame-ignore-revs`). clang-format and clang-tidy stay on
the same LLVM release. A new warning flag or check is admitted when it targets a defect class
this codebase can have and the tree is clean under it in the same change.

## Not adopted yet

- `-Wconversion`/`-Wsign-conversion`: ~1,200 findings, mostly deliberate kernel index
  arithmetic; `bugprone-implicit-widening-of-multiplication-result` covers the overflow class.
- A Python type checker (pyright/mypy): needs the tools' torch environment to resolve imports;
  adopt with that environment pinned.
- CMake formatting (gersemi) and include-what-you-use: style and build-time value only.
