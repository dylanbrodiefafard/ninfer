# Code-quality gates

This is the authority for NInfer's static and dynamic quality tooling: what each gate catches,
where its configuration and version pin live, how to run it, and how a finding may be
suppressed. The expectations these gates serve are stated in [`AGENTS.md`](../../AGENTS.md)
("Code and comment quality"). Every tool is open source and version-pinned, so a finding is a
property of the code, not of the contributor's machine.

## Gates

| Gate | Catches | Configuration and pin | Run |
|---|---|---|---|
| Compiler diagnostics, `-Werror` | shadowing, missing virtual destructors, hidden overloads, proven null dereference, unannotated fallthrough, format mismatches, float-to-double promotion, duplicated conditions/branches, member-initializer order, default-stream kernel launches, `this`-capturing extended lambdas | [`cmake/warnings.cmake`](../../cmake/warnings.cmake); toolchain pinned by the builder image (GCC 13, CUDA 13.1) | every build |
| clang-tidy 22 | use-after-move, unchecked `optional` access, dangling references, implicit widening of `int` index products, swapped arguments, slicing, exception escape from destructors and `noexcept`, empty catch, needless copies and no-op moves, path-sensitive analyzer findings | [`.clang-tidy`](../../.clang-tidy); version in [`scripts/run-clang-tidy.py`](../../scripts/run-clang-tidy.py) and the Dockerfile build stage | `./scripts/run-clang-tidy.py --changed` |
| clang-format 22 | C++/CUDA layout drift | [`.clang-format`](../../.clang-format); version in [`.pre-commit-config.yaml`](../../.pre-commit-config.yaml) | pre-commit |
| ruff (lint + format) | undefined names, late-binding closures, silent `zip` truncation, missing commas in string lists, unchecked `subprocess`, naive datetimes, dead code, import order | [`ruff.toml`](../../ruff.toml); version in `.pre-commit-config.yaml` | pre-commit |
| shellcheck | unquoted expansions, undeclared locals leaking to global scope, dead variables | version in `.pre-commit-config.yaml` | pre-commit |
| typos | misspelled identifiers, comments, and documentation | [`_typos.toml`](../../_typos.toml); version in `.pre-commit-config.yaml` | pre-commit |
| File hygiene | merge markers, accidentally committed large files or private keys, broken JSON/TOML/YAML, CRLF, trailing whitespace, missing final newline, shebang/executable-bit mismatch | `.pre-commit-config.yaml` | pre-commit |
| compute-sanitizer | device out-of-bounds and misaligned access, shared-memory races, illegal barrier use, reads of uninitialized device memory | CUDA 13.1 toolkit | `./scripts/run-unit-tests.sh --compute-sanitizer TOOL` |
| AddressSanitizer / UBSan | host heap/stack overflow, use-after-free, leaks, signed overflow, misaligned and invalid casts | `NINFER_SANITIZE` in `cmake/warnings.cmake` | separate tree, below |

CI ([`.github/workflows/quality.yml`](../../.github/workflows/quality.yml)) runs the pre-commit
gate on every push and pull request. Gates that need the CUDA toolchain or a GPU run in the
`ninfer-builder` container; the clean way to enforce them in CI is a self-hosted runner on the
RTX 5090 host that runs the build, `run-clang-tidy.py --changed`, and `run-unit-tests.sh`.

## Running the gates

Install the hook runner once (Python 3.11): `python3 -m pip install pre-commit==4.6.2` (or the
compatible, faster [`prek`](https://github.com/j178/prek)), then `pre-commit install`. The
commit hook runs the fast gates on staged files; `pre-commit run --all-files` checks the whole
tree exactly as CI does. Hooks that rewrite files (formatters, `--fix`) leave the rewrite
unstaged; review it and stage it.

Compiler diagnostics need no separate step: every project C++ and CUDA translation unit builds
with `NINFER_WARNINGS_AS_ERRORS=ON` by default. The policy applies only to NInfer targets;
third-party sources keep their upstream flags.

clang-tidy runs in the builder container against `/build/compile_commands.json`:

```bash
./scripts/run-clang-tidy.py --changed            # lines changed vs. origin/HEAD (the gate)
./scripts/run-clang-tidy.py src/serve/foo.cpp    # named files; a header selects its includers
./scripts/run-clang-tidy.py                      # whole tree: the existing-findings backlog
```

The runner rewrites nvcc commands into clang CUDA commands for `sm_120a`, so kernels, launchers,
and host code share one configuration. `--changed` analyzes every translation unit that is or
includes a changed file and reports only diagnostics on changed lines: new and modified code
meets the full check set, while findings in untouched code stay a visible backlog to burn down
by check family rather than a reason to weaken `.clang-tidy`. Header changes are mapped to their
includers through Ninja's dependency log, so build the tree first.

Runtime checkers are opt-in because they are slow:

```bash
./scripts/run-unit-tests.sh --compute-sanitizer memcheck  -R ninfer_arena_test
./scripts/run-unit-tests.sh --compute-sanitizer racecheck -R ninfer_gqa_attention_test
./scripts/run-unit-tests.sh --compute-sanitizer initcheck # also synccheck

cmake -S . -B build-asan -G Ninja -DCMAKE_BUILD_TYPE=RelWithDebInfo -DBUILD_TESTING=ON \
  -DNINFER_SANITIZE=address,undefined
cmake --build build-asan
ASAN_OPTIONS=protect_shadow_gap=0 ctest --test-dir build-asan -E '_real_test$'
```

`--compute-sanitizer` changes only the CTest launcher (`NINFER_TEST_LAUNCHER`), so switching
tools does not recompile. CUDA's runtime maps memory inside ASan's shadow gap, hence
`protect_shadow_gap=0`. Run memcheck and racecheck for new or changed kernels with shared-memory
staging, asynchronous copies, or warp-level synchronization, and initcheck for new workspace or
arena consumers.

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
this codebase can have and the existing tree is either clean under it or its backlog is burned
down in the same change.

## Not adopted yet

- `-Wconversion`/`-Wsign-conversion`: ~1,200 findings, mostly deliberate kernel index
  arithmetic; `bugprone-implicit-widening-of-multiplication-result` covers the overflow class.
- A Python type checker (pyright/mypy): needs the tools' torch environment to resolve imports;
  adopt with that environment pinned.
- CMake formatting (gersemi) and include-what-you-use: style and build-time value only.
