# AGENTS.md

## Working rules

Complete the user's explicit deliverable within the applicable product contract, choosing the
technically strongest coherent solution: optimize for architectural integrity, clear ownership,
functional and numerical correctness, and maximum relevant performance with direct code, and make
every affected implementation, test, tool, and active authority consistent with the selected
design. Never optimize for a small diff, low effort, short-term simplicity, backward
compatibility, or a superseded internal path: change size, difficulty, and compatibility are not
quality criteria. When work alternatives compete, prioritize in this order: product and
external-contract constraints, the explicit deliverable, correctness of supported behavior,
architecture and ownership, performance, and only the evidence needed to support the result. The
product and architecture described here are the current contract; a task may explicitly change
it, in which case update the affected implementation, tests, and active authorities consistently.

Correctness, performance, tests, profiling, documentation, provenance, cleanup, and tooling are
means to the requested outcome, not independent objectives; do not let supporting work replace,
delay, or materially enlarge the deliverable. Generality, defensive hardening, formal
completeness, broad compatibility, and test coverage are not goals by themselves; low
maintenance cost may distinguish otherwise equivalent designs but never justifies worse
architecture or performance. Prefer explicit target-specific implementation over framework-like
abstraction: do not add generic model graphs, family base classes, plugin discovery,
string-driven execution, hidden device allocation, runtime weight repacking, or placeholders for
hypothetical models or hardware unless an explicitly changed product contract requires them.

Before substantial work, determine the requested output and the behavior or decision it must
support; no separate planning artifact is required. Work is in scope only when it directly
contributes to the requested deliverable, is necessary to preserve an applicable product,
semantic, or external contract, resolves uncertainty that could materially change the result, or
checks a realistic regression introduced by the change. An architectural redesign, cross-cutting
refactor, or replacement of an existing path is in scope when necessary to deliver the strongest
solution; do not use scope control as a reason to ship an inferior patch, and do not expand into
unrelated audits, cleanup, hardening, compatibility work, benchmark campaigns, or documentation
projects: general preferences, future scenarios, and concerns outside the declared product model
do not create requirements. Handle incidental findings proportionally: address them when they
block the requested outcome or make it materially incorrect, include them when inseparable from
a coherent implementation, otherwise leave them unchanged and mention them only when useful to
the user. For analysis, review, or design work, the requested explanation or design artifact is
the deliverable, and experiments and code inspection serve only to resolve material questions;
for implementation work, implement the selected design completely across its affected
boundaries, remove the superseded project-owned path, and validate its supported observable
behavior; for diagnosis, establish the cause and supporting evidence without turning the task
into an unrequested fix or redesign.

Select evidence from the claim or decision it supports; the availability of a tool, test suite,
artifact, or profiler does not make its use necessary. Prefer representative evidence over
exhaustive evidence, and do not repeat an experiment unless the previous result is invalid or
inconclusive, or the new result could change a live decision. Verification must match the
semantic contract: exact comparison for exact formats and transformations, numerical or
behavioral criteria for floating-point and probabilistic work; do not substitute final-output
plausibility for verification of an operator or state transition. Record only the provenance
needed to interpret a material result: target, hardware/toolchain, workload or command, and
summarized outcome. Fixed hashes, clean worktrees, full command transcripts, raw profiler
inventories, byte-identical regeneration, and exact probabilistic outputs are not validity
requirements unless a concrete contract or the user requires them; this rule also governs
measurement reports in performance work.

Stop when the requested deliverable exists, applicable contracts are satisfied, material claims
have sufficient evidence, relevant checks pass or their limitations are stated clearly, and no
known in-scope issue prevents the result from being used; do not continue merely to eliminate
uncertainty, collect more metrics, complete a process loop, or investigate unrelated
observations. The final result leads with the deliverable, key decisions, relevant verification,
and material limitations; raw logs, experiment diaries, and intermediate artifacts are excluded
unless requested or themselves the deliverable.

When the user points out an error in the agent's execution or reasoning, never open with a
reflexive agreement such as "You're right...", "Indeed", "Exactly", or "Good question"; state
the specific mistake directly, explain its concrete effect when relevant, and say what has been
or will be changed, proportionately and without performative apology or praise.

## Current product contract

NInfer is a from-scratch C++/CUDA inference engine for maximum single-GPU inference performance,
compiled for `sm_120a` and tuned on an NVIDIA GeForce RTX 5090. The only supported identity is
`qwen3.8-27b/nvfp4`; its default artifact, when available, is the MTP-NVFP4 or DFlash2 (NVFP4
matrices, BF16 selector codebook) variant over the base Ostfralla NVFP4 shell. The workload is
one GPU, one resident model instance, and a startup-fixed one to eight active requests. The
Engine forms one compact decode batch per round boundary with bounded FIFO ingress and no
preemption. Large-scale or preemptive continuous batching, priority/QoS scheduling, additional
checkpoint targets, and retargeting to another execution platform are outside the current
product. This is a trusted local, single-owner project, and requirements from a different
workload, trust model, or deployment model are out of scope until the contract is explicitly
changed.

## Sources of truth

Read only the current authorities relevant to a live decision in the task; this list is a
routing map, not a mandatory reading list:

- `README.md` and executable `--help`: delivered capabilities and exact commands;
- `docs/README.md`: documentation map; `docs/cli.md`: CLI input, sampling, MTP, and runtime
  options; `docs/serving.md`: OpenAI/Anthropic HTTP behavior; `docs/performance.md`:
  performance methodology and results;
- `docs/maintainer/concurrent-inference-architecture.md`: request/slot lifecycle, scheduling,
  batched execution, CUDA Graph, and speculative-concurrency semantics;
- `docs/maintainer/paged-kv-cache.md`: KV capacity, page ownership and retention, physical
  layouts, and paged consumer contracts;
- `docs/maintainer/artifact-container.md`, `storage-layouts.md`, and `tensor-formats.md`:
  generic `.ninfer` contracts;
- `docs/maintainer/qwen3.6-27b-artifact.md`, `qwen3.8-27b-artifact.md`, and
  `qwen3.6-35b-a3b-artifact.md`: target inventories, conversion, and binding;
- `docs/maintainer/qwen3.6-27b-model.md` (also: family/Variant package structure) and
  `qwen3.6-35b-a3b-model.md`: model mathematics, dimensions, and state semantics;
- `docs/maintainer/op-development.md`: Op admission, contracts, ownership, qualification, and
  performance-evidence rules;
- `docs/maintainer/kernel-iteration.md`: Layer 0-3 CUDA speed procedure (`tools.kdev`
  bound/mma/Op sweep/production path);
- `include/ninfer/engine.h` and `include/ninfer/types.h`: in-tree C++ product interface.

## Product and ownership boundaries

These boundaries govern ordinary implementation work; an explicit architecture task may revise
them, updating the corresponding authorities and implementation together.

| Module | Owns | Must not own |
|---|---|---|
| `.ninfer` | the only C++ product artifact | `.qus` fallback, extension detection, compatibility shims, a second product lane |
| `include/ninfer/engine.h`, `include/ninfer/types.h` | the opaque Engine interface used by in-tree applications and owning host values; NInfer does not install or export a C++ SDK | |
| `include/ninfer/ops/` | repository-internal semantic Op contracts | |
| `src/core` | device primitives, tensors/views, checked layouts, arenas, graph RAII, physical KV-cache containers, raw transfer mechanisms | |
| `src/artifact` | generic `.ninfer` framing, descriptors, binding primitives, materialization | checkpoint execution semantics |
| `src/ops` | every semantically closed Op implementation, including fused, fixed-shape, and device-specialized paths; ownership follows the mathematical or state-transition contract, not the first model caller or demonstrated cross-target reuse | |
| `src/targets/qwen3_6` | Qwen3.6-family invariants shared by 27B and 35B-A3B: tokenizer/template and output semantics, media preprocessing and MRoPE prompt construction, owning prepared-prompt/output-session types, semantic weight-view schemas, passive Vision definitions, and the fixed planning/Program/Text/Vision/speculative/state/workspace/CUDA-Graph algorithms | target identity, registry entry, artifact binder, target leaf implementation, storage for a live Program instance |
| `src/targets/<package>` | registered checkpoint identities, storage profiles, binder, `LoadedModel`, configuration, populated family model-view values and private leaf payloads, diagnostics, graph frontier values, and exactly three execution-leaf families (attention projection, GDN projection/control, post-mixer); aliases and instantiates the family runtime types | a copied Program, Text/Vision/speculative schedule, workspace composition, state transaction, or graph-capture algorithm; leaf Ops remain in `src/ops` |
| `src/runtime` | common contracts, generated-token transaction/publication policy, public Engine PIMPL | model mathematics or target state |
| `src/media/decode` | consuming already-owned bytes | URL/path/data acquisition, which belongs to `src/product/media_acquire`, CLI, or serving and is not linked into a target |
| `src/product/prompt_input` | the shared product-side JSON/message-to-owning-input adapter | |
| `src/serve` | protocol translation and transport | |
| `tools/convert/<target>`, `tools/reference/<target>`, `tools/parity/<target>` | target-private conversion, correctness, and diagnostic implementations | |

CLI, server, and benchmark call only the public Engine for inference.

## Compatibility and document lifecycle

Project-owned C++ APIs, CLIs, Python tools, fixtures, reports, formats, and active documentation
do not preserve backward compatibility: when a task replaces project-owned behavior, remove the
obsolete aliases, fallbacks, transition branches, and tests in the affected contract instead of
maintaining two paths, without turning that into unrelated repository-wide cleanup. The
advertised OpenAI and Anthropic protocol surfaces are real external contracts; a change to
their behavior must update the affected schema tests and serving documentation together.
Integrate stable requirements into the existing active reference; use a temporary dated plan
only when active work genuinely needs one, remove completed or abandoned plans, and do not
create parallel `final`, `v2`, or `new-design` references.

## Numerical correctness

When a task changes numerical behavior or makes a numerical claim, identify the mathematical
oracle, represented public inputs, explicit semantic cast/quantization/state boundaries, output
criterion, and real model shapes relevant to that claim. If a route's private precision or
reduction profile matters to the evidence, describe it as an implementation profile rather than a
semantic requirement. Apply exact, tolerance-based, or behavioral comparison according to the
actual semantic contract.

Every floating-point Op has one independent naive FP32/FP64 mathematical oracle; exact transforms
and codecs have one independent exact oracle. The oracle evaluates the complete logical formula
from the represented public inputs and, for packed weights, decodes the signed code with the exact
stored scale. It does not copy a production kernel's staging casts, reduction tree, workspace dtype,
or another implementation's output.

The oracle does not prescribe a production arithmetic path. Unless an intermediate value is an
observable Op output, explicit Cast/quantize/dequantize result, registered codec value, or specified
persistent state, kernels may choose the natural intermediate precision, instruction operands,
reduction association, workspace representation, and kernel decomposition for their route; a
fused kernel need not reproduce an unfused BF16 materialization and may use a lower-precision
intermediate when that is the natural qualified implementation. Every production route is checked
directly against the same oracle with a criterion appropriate to its output and implementation
profile; pairwise implementation parity is supplementary evidence only.

Where relevant to the changed behavior, account for numeric-format decode, BF16 fusion order, FP32
GDN state, BF16/INT8/NVFP4 KV, MTP accept/commit state, arena lifetime, and CUDA Graph address
stability. This is a risk map, not a checklist for every numerical task.

## Performance work

CLI, serve, Engine A/Bs, and decode-speed work use the Engine default `--kv-dtype nvfp4` unless
the task is numerical identity, a long-context capacity comparison, or an explicit dtype A/B.
Pass `--kv-dtype bf16` when uncompressed KV is the contract.

Kernel speed work follows [`docs/maintainer/kernel-iteration.md`](docs/maintainer/kernel-iteration.md),
including the `tools.kdev` recipe/bound/mma procedure; do not implement an idea the classifier
refuses.

Define a performance claim at the level where it matters (operator, schedule, request phase, or
end-to-end inference) and measure that level directly; use whole-inference profiling when
end-to-end attribution remains unresolved, and kernel profiling only once a specific kernel is
identified.

## Tests and verification

Add or retain a test only when it protects supported observable behavior or a realistic
regression: numerical kernel/model correctness, `.ninfer` framing/binding, external
schema/report behavior, a small real integration route, GPU lifetime, or a reproduced bug. Do
not add tests for coverage, private file/class shape, getters/constructors, deleted
compatibility, source-string scans, hypothetical failures, or test ceremony, and do not replace
weak verification with low-value tests; state clearly when a relevant check could not run and
why.

Run a focused set of checks sufficient to support the changed behavior and its material claims.
Typical evidence (not a cumulative checklist): documentation changes get an affected
active-link/stale-reference review and `git diff --check`; C++ runtime/API changes get
affected explicit targets and meaningful tests; Python tooling gets `py_compile` and affected
Python tests; `.ninfer` reader/converter/binder changes get affected contract tests and a real
artifact when semantics require it; CUDA math gets an independent numerical oracle at relevant
shapes; memory/lifetime changes get the affected execution, with a sanitizer only for a
concrete lifetime risk; performance changes get measurement at the claimed scope, with
attribution tools only when needed; serving changes get affected OpenAI/Anthropic schema tests
and observable request/stream behavior.

After substantial work, and before a commit or push, run the full C++ unit-test suite with
`./scripts/run-unit-tests.sh`. That command builds the test targets in the `ninfer-builder`
GPU container and runs every CTest except the opt-in real-artifact Engine tests.
`--real` includes those Engine tests and binds `.ninfer` files from `models/`, `/models`, or `models/weights.env`. Focused checks remain the right evidence during the work;
the full suite is the gate that unrelated tests still pass.

## Local environment

Conventional project resources (not a per-task checklist): Python 3.11 via `python3`; the
supported artifact is `qwen3_8_27b_nvfp4.ninfer`, a local file placed by the maintainer (for
example in `models/`), with its download source in the README; local directories such as `out/`
are checkout-specific, not a convention; normal build in `build/`; profiler output in
`profiles/ncu/`, `profiles/nsys/`, `profiles/bench/`; hardware/toolchain is RTX 5090, `sm_120a`,
CUDA 13.1; speed and Engine A/B work uses `--kv-dtype nvfp4`. GPU tests and kernel iteration
run in the `ninfer-builder` container, started with `./scripts/dev-setup.sh`. That image is
the Dockerfile `build` stage (`docker build --target build --tag local/ninfer-builder:5090 .`),
not the runtime serve image. Use the selected Python 3.11 interpreter explicitly; do not
install or upgrade dependencies unless the task requires it. Never select an artifact by glob,
modification time, or an unqualified "latest" name; large artifacts, source checkpoints, and
profiler outputs are local prerequisites, not things to download or regenerate unless in scope.

## Commits

Create a commit only when the user requests one. Use Conventional Commit-style subjects with
concise lowercase types consistent with repository history: `feat`, `fix`, `perf`, `bench`,
`test`, `build`, `refactor`, `docs`, `chore`.