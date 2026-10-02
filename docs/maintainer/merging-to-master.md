# Merging into master

`master` holds only code that has passed every gate NInfer has. Working branches move faster:
the commit hook keeps them formatted and linted, and their authors run the fast unit tests often
(see [code-quality gates](code-quality.md)). This document is the authority for the step between
the two: what must run, and pass, on a commit before it is merged into `master`.

## Rule

Every gate below runs on the exact commit that will be merged, on the RTX 5090 host in the
`ninfer-builder` container, and every one passes. A gate that could not run is a failed gate:
the merge waits. Nothing is skipped because the change "does not touch" an area, and no gate is
weakened, disabled, or bypassed to get through; a finding is fixed on the branch and the gate is
run again.

The branch first takes the current `master` (merge `master` into the branch), so the gates run
on what `master` will become.

## Gates, in order

Cheap gates come first so a failure stops the run early.

| # | Gate | Command | Passes when |
|---|---|---|---|
| 1 | Formatters, linters, spelling, file hygiene | `pre-commit run --all-files` | every hook passes and no file is rewritten |
| 2 | Compiler warnings | `./scripts/dev-setup.sh`, then `docker exec ninfer-builder cmake --build /build` | the tree builds with tests and benchmarks on; `NINFER_WARNINGS_AS_ERRORS` is `ON` |
| 3 | clang-tidy, whole tree | `./scripts/run-clang-tidy.py` | `0 diagnostics` |
| 4 | Unit tests, all of them | `./scripts/run-unit-tests.sh` | every test passes, the tests labelled `slow` included |
| 5 | Engine tests on real artifacts | `./scripts/run-unit-tests.sh --real` | every Engine test passes with the supported artifacts bound |
| 6 | compute-sanitizer memcheck | `./scripts/run-unit-tests.sh --compute-sanitizer memcheck -L kernel` | every kernel test passes with `0 errors` |
| 7 | compute-sanitizer racecheck | `./scripts/run-unit-tests.sh --compute-sanitizer racecheck -L kernel` | every kernel test passes with `0 hazards` |
| 8 | compute-sanitizer initcheck | `./scripts/run-unit-tests.sh --compute-sanitizer initcheck -L kernel` | every kernel test passes with `0 errors` |
| 9 | AddressSanitizer and UBSan | the three commands below | every test passes with no sanitizer report |

Gate 9 uses its own tree, because an instrumented build is not the warning gate:

```bash
docker exec ninfer-builder cmake -S /src -B /build-asan -G Ninja \
  -DCMAKE_BUILD_TYPE=RelWithDebInfo -DBUILD_TESTING=ON -DNINFER_SANITIZE=address,undefined
docker exec ninfer-builder cmake --build /build-asan
docker exec -e ASAN_OPTIONS=protect_shadow_gap=0 ninfer-builder \
  ctest --test-dir /build-asan --output-on-failure -E '_real_test$'
```

Notes on scope:

- Gate 5 needs `qwen3_8_27b_nvfp4.ninfer` and its DFlash2 or MTP variant; an Engine test that
  skips because its supported artifact is missing has not run. `--print-weights` shows what is
  bound.
- Gates 6 to 8 run over the tests labelled `kernel`: every Op test. Host-only tests make no CUDA
  call, and compute-sanitizer reports that as an error, so they are not part of these gates.
- Gate 4 stops when the GPU has less than 20 GiB free; stop a resident `ninfer-serve` first.

## Cost and hazards

Plan for the sanitizer gates. Measured on the RTX 5090 host:

- the unit suite (gate 4) takes about 14 minutes; the whole-tree clang-tidy run (gate 3) checks
  several hundred translation units and takes longer;
- memcheck is a few times slower than a plain run for most tests, but
  `ninfer_speculative_round_test` did not finish in seven hours under it;
- racecheck is far slower: single linear tests take 25 to 50 minutes, and
  `ninfer_sampling_test` grew past 54 GB of host memory.

Run the sanitizer gates only in the builder created by `./scripts/dev-setup.sh`, which caps the
container's memory so a runaway run is killed instead of exhausting the host, and read the
warning about `ninfer_attn_input_proj_test` in [`tests/README.md`](../../tests/README.md) before
running gate 4 or gates 6 to 8 unattended.

## Recording the result

The merge commit, or the pull request it closes, states for each gate the command, the commit
it ran on, the host driver and toolchain, and the outcome. Summaries only: counts of tests,
diagnostics, and hazards, not logs.

## Merging

1. Merge the current `master` into the branch and resolve conflicts there.
2. Run gates 1 to 9 on that commit. Fix findings on the branch and repeat until every gate
   passes on a single commit.
3. Merge the branch into `master` with a merge commit that records the gate results, and push.
4. If `master` moved while the gates ran, start again from step 1.
