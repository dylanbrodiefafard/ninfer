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
./scripts/run-unit-tests.sh --compute-sanitizer memcheck  -R ninfer_arena_test
./scripts/run-unit-tests.sh --compute-sanitizer racecheck -R ninfer_gqa_attention_test
./scripts/run-unit-tests.sh --compute-sanitizer initcheck # also synccheck

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
