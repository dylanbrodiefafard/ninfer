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

## Completion discipline

Do not emit a final response while any user-assigned implementation task remains incomplete. The
live R9700 execution ledger's unchecked items are assigned work unless the user explicitly narrows
or cancels them. Continue autonomous in-scope work and use commentary only for progress updates.
Emit a final response only after every assigned task is complete and verified, or when an external
action, missing authority, or user decision genuinely blocks all further progress; state that
specific blocker plainly.

## Current product contract

NInfer is a from-scratch C++/HIP inference engine for maximum single-GPU inference performance,
compiled only for `gfx1201` and tuned on one AMD Radeon AI PRO R9700. The sole supported model is
Qwen3.8-27B. Its growing Text/MTP cache has exactly three planes: FP8 E4M3FN keys, signed INT4
values, and FP16 value scales; there is no cache dtype selector, K scale, alternate cache path, or
retained compatibility backend. The production Text weights are `qwen3.8-27b/r9700-lq4` (profile
`R9700Lq4`): GPTQ-rounded LQ4 Text projections and output head with 21 row-scaled FP8
protections (2026-09-30), admitted by the BF16-source PPL, NIAH, exact-token, and whole-inference
gates. The weight-precision bar is no worse than the 5090 NVFP4 build on every BF16-source cell
(measured with the frozen 5090 scorer); protected attention query/key projections are not demoted
below FP8. The selected image/video recipe is `qwen3.8-27b/r9700-lq4-vision-bf16` (profile
`R9700Lq4VisionBf16`): identical Text/MTP/DFlash2 payloads with exact original BF16 Vision
matrices. Vision-off binding validates those matrices without placing them on the GPU.
Other `-eval` identities remain evaluation and conversion-base artifacts. DFlash2 keeps its model-specified BF16 selector
codebook and private fixed BF16 state.

The workload is one R9700, one resident model instance, and a startup-fixed one to sixteen active
resident requests; DFlash2 admits one to eight. The Engine forms one compact decode batch per round boundary with bounded FIFO
ingress. With RAM enabled, output reservations grow in windows of at most 8192 tokens and settled
active continuations may suspend to secured RAM under KV pressure; RAM off retains full-lifetime
GPU entitlement. A completion cohort whose target-produced lifetime envelopes jointly fit GPU capacity and dynamic
window-end RAM credits for other residents guarantee a feasible evacuation path; live images are never optional cache or disk-spill victims. Total logical
requests remain bounded by concurrency plus pending capacity. One prefill owner at a time advances either prefill-first or in startup-fixed
slices, which under DFlash share a decode round's target forward (mixed round). Large-scale or
general preemptive continuous batching, priority/QoS scheduling, additional checkpoint targets, and
retargeting to another execution platform are outside the current product. This is a trusted local, single-owner project, and requirements from a different
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
  batched execution, Device Graph, and speculative-concurrency semantics;
- `docs/maintainer/paged-kv-cache.md`: KV capacity, page ownership and retention, physical
  layouts, and paged consumer contracts;
- `docs/maintainer/artifact-container.md`, `storage-layouts.md`, and `tensor-formats.md`:
  generic `.ninfer` contracts;
- `docs/maintainer/qwen3.8-27b-artifact.md` and `r9700-integer-artifact-candidate.md`: target
  inventory, conversion, and binding;
- `docs/maintainer/qwen3.8-27b-model.md` (also: family/Variant package structure): model
  mathematics, dimensions, and state semantics;
- `docs/maintainer/op-development.md`: Op admission, contracts, ownership, qualification, and
  performance-evidence rules;
- `docs/maintainer/kernel-iteration.md`: Layer 0-3 R9700 speed procedure (bound hypothesis,
  gfx1201 legality, qualified Op sweep, physical profiling, production path);
- `docs/maintainer/merging-to-master.md`: full promotion gates and experimental-to-master workflow;
- `docs/maintainer/code-quality.md`: compiler-warning, clang-tidy, formatter, and lint gates, and
  the gfx1201 device checks (memcheck, initcheck, racecheck) that replace compute-sanitizer;
- `include/ninfer/engine.h` and `include/ninfer/types.h`: in-tree C++ product interface.

## Upstream synchronization

Upstream `experimental` is reconciled through `5c2bd4b39a3cc645e400e842ca2b92968f920aec`
via custom AMD ports, not merged ancestry.
Unsupported NVIDIA paths were excluded; decisions are in `docs/maintainer/upstream-sync.md`.
For future syncs, review upstream changes after this baseline against current AMD behavior,
not `HEAD..upstream`; re-establish the baseline if upstream history was rewritten. Record
source commits and advance the baseline only after each change is ported, already equivalent,
or explicitly dispositioned. Do not fabricate ancestry with an `ours` merge or rewrite
published history merely to mark upstream merged.

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
| `src/targets/qwen3` | Qwen3-family invariants used by Qwen3.8-27B: tokenizer/template and output semantics, media preprocessing and MRoPE prompt construction, owning prepared-prompt/output-session types, semantic weight-view schemas, passive Vision definitions, and the fixed planning/Program/Text/Vision/speculative/state/workspace/Device-Graph algorithms | target identity, registry entry, artifact binder, target leaf implementation, storage for a live Program instance |
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
GDN state, the typed FP8-K/INT4-V/FP16-scale cache, private DFlash2 BF16 state, MTP accept/commit
state, arena lifetime, and Device Graph address stability. This is a risk map, not a checklist for
every numerical task.

## Performance work

CLI, serve, PPL, Engine A/Bs, and decode-speed work use the one fixed FP8-K/INT4-V cache. There is
no cache dtype flag or alternate production cache path. Independent BF16-reference runs are
separate executables/artifacts used for numerical comparison, not a mutable Engine option.

Kernel speed work follows [`docs/maintainer/kernel-iteration.md`](docs/maintainer/kernel-iteration.md),
including the `tools/r9700` oracle, ISA/resource, timing, and focused-profiler procedure; do not
promote an unqualified or physically slower challenger.

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
artifact when semantics require it; HIP math gets an independent numerical oracle at relevant
shapes; memory/lifetime changes get the affected execution, with a sanitizer only for a
concrete lifetime risk; performance changes get measurement at the claimed scope, with
attribution tools only when needed; serving changes get affected OpenAI/Anthropic schema tests
and observable request/stream behavior.

## Host prerequisites and workflow

Read `AGENTS.local.md` if present for private host instructions. Source
`.ninfer-local.env` before manual GPU commands; maintainer shell runners load it.
Neither file belongs in Git. The template `.ninfer-local.env.example` documents
settings; `docs/maintainer/local-development.md` explains setup.

Python tooling uses an explicitly selected Python 3.11 interpreter. The complete
18-shard Qwen3.8-27B BF16 source, selected converted artifact and reference
checkpoints are local prerequisites, never downloaded implicitly. Do not install
or upgrade dependencies unless the task requires it. Never select an artifact by
glob, modification time or an unqualified "latest" name.

Every GPU command must hold `flock --exclusive "$NINFER_GPU_LOCK" <command>` through
all child work. Choose the same lock for every checkout using the physical R9700;
never delete it or stop another owner's job. Builds need no GPU lock. Run one build
per agent with at most 8 jobs; reduce concurrency under memory or I/O pressure.

The normal HIP build is `build-r9700/`, with ROCm selected under `/opt/rocm`.
Generated evidence belongs under ignored `profiles/`. Product inference always uses
the fixed FP8-K/INT4-V/FP16-scale cache. `DeviceContext` sets
`HSA_USERPTR_FOR_PAGED_MEM=0` and `HSA_USE_SVM=1` before HIP initialization. Pinned
arenas use HIP/GTT backing and direct reads use ordinary-RAM staging. Runtime,
checkpoints and RAM prefixes share the driver GTT budget; an unavailable requested
allocation fails clearly. This policy changes no host compaction sysctl.

Prefer incremental CMake/Ninja builds. Compose builds before startup and reuses a
locked object cache; toolchain or dependency changes reset it. Use `docker compose
up -d --wait` for the base proxy deployment, or include `-f compose.yaml -f
compose.local.yaml` for localhost. Keep `COMPOSE_BAKE=false`. To use an existing
image, pass `--no-build --pull never server`. Run relevant tests separately.
The 24-GiB/no-swap limit applies to the server; the builder has no memory limit.
Deployment and cache paths belong in ignored `.env`, not shared instructions.
See `docs/containers.md`.

## Commits

Create a commit only when the user requests one. Use Conventional Commit-style subjects with
concise lowercase types consistent with repository history: `feat`, `fix`, `perf`, `bench`,
`test`, `build`, `refactor`, `docs`, `chore`.
