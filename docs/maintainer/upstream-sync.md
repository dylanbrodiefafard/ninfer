# Upstream sync log

Upstream is [Neroued/ninfer](https://github.com/Neroued/ninfer) (`git remote add upstream
https://github.com/Neroued/ninfer`), branch `dev`. This fork diverged heavily after the merge-base,
so upstream work is ported by hand, not merged.

## Review watermark

| Field | Value |
|---|---|
| Last upstream commit reviewed | `27df5a344726f68f5b65848fe157e15a44239a0c` (`feat(distributed): prepare qwen4exp model shards`) |
| Merge-base | `0c94153b` (2026-08-16) |
| Reviewed on | 2026-10-19 |

The next review starts from `git log 27df5a34..upstream/dev`; every commit up to and including
the watermark has been judged and needs no second pass.

## Review of e31bc99b..27df5a34 (2026-10-19)

Reviewed all 226 commits at the fetched `upstream/dev` snapshot. These are selective ports into
the fixed RTX 5090 product, with no upstream ancestry merge.

The secondary source is the sibling `ninfer-amd-r9700` checkout, repository
`jmander11/ninfer-amd-r9700`, through `702d2c86a`. Its sync commit `7bcc26a67` reviews the first
204 commits of this range through `05327b22`; `e4926cdac` records subsequent AMD performance
choices. Its active `docs/maintainer/upstream-sync.md` and local commit dispositions were used
to identify candidates, then checked against this fork's ownership and CUDA implementations.
AMD timings, HIP kernels, graph calibration constants, LQ4 weights, FP8-K/INT4-V storage, and
its additional artifact identities are not RTX 5090 evidence or product requirements.

### Selected ports and existing equivalents

| Source | RTX 5090 decision |
|---|---|
| `4201b5d2` | Not applicable: the Engine retains one exclusive prefill owner through finalization. No decode or admission can overwrite the initialized DFlash ingress between chunks; its stable lane and KV binding do not migrate. Upstream uses a different scheduling/state-slot model. |
| `a8e212ac`, AMD `7bcc26a67` | Ported per-encode BPE scratch and adjacency/generation-checked heap merges above 64 bytes. Kept this fork's packed emission tables and scalar short-word route. Equal ranks select the leftmost pair. Context migration/Jinja changes in the source commit are not imported. |
| `81c8ce09`, AMD `7bcc26a67` | Ported decimal source-preservation checks at raw HTTP, CLI messages, and Engine schema-string boundaries. Exact integer literals remain integers; inexact schema assertions reject before serialization with a source pointer. Broader number/tuple language expansion is not selected. |
| `91d568a6`, AMD `7bcc26a67` | Adapted the finite LayerNorm overflow finding to CUDA. Ordinary FP32 Welford routes remain; rows with overflowed moments or variance-plus-epsilon use an exceptional FP64 evaluation before BF16 publication. No AMD normalization or Spark dispatch is imported. |
| `75a89050` | No blanket cancellation fence: startup-owned buffers and same-compute-stream reuse preserve device lifetime. Startup/failure cleanup already fences before release. This retains the commit-tail overlap rather than imposing an unneeded wait. |
| `064965c7` | Production page publication already passes its caller stream; the source activation-store/default-stream path is absent. |
| `911d34db`, `452c3f5f` | Existing native copy holds preserve admission sources. Upstream's preemptive context forest, recovery scorer, and reclaim machinery are absent; the bounded FIFO scheduler and RAM/disk ownership remain. |
| `68c54356` | No Jinja trimming path: the native Qwen frontend owns its source boundaries. |
| `abb7f14f`, `08902f73`, `eb6696ae`, `41e50d0d` | Existing metrics, EBNF/JSON constraints, and constrained tool calls cover the corresponding product needs. No replacement runtime or grammar integration is imported. |
| `1cfdb4d6` | Native pair widening requires CUDA 13.2; the supported toolchain is CUDA 13.1. Existing exact codec expansion remains. |

Implementation commits: tokenizer `3db63dbb`, LayerNorm `f4b8aa25`, and schema precision
`12e3e78c`.

### Candidates not selected

The attention rewrites (`23b0997d` through `a012e2bc`, `7f6aafed`, `d44ab584`) use upstream's
own cache layouts, work partitions, and graph topology. This fork retains its qualified
NVFP4/INT8/BF16 attention and compact C<=6 verification routes. No comparative operator or
Engine speed claim is made, and no upstream schedule threshold is adopted without native
qualification. The same applies to matching BF16 Vision shapes in upstream's linear tuning;
matching dimensions alone do not establish a faster route.

AMD's counted-string masks (`0a8530d4`), dependency vendoring (`7e2973ee`), richer schema
composition/diagnostics (`2734a56e`), choices/regex (`8ffa3131`), provisional tool-argument events
(`bc70d7c8`), and messages-file-relative media paths (`4126ecf4`) are optional feature or
implementation changes, not fixes required to preserve the currently documented surfaces.
This sync retains the existing supported schema subset, whole-call tool publication, media
resolution, and patched pinned CPU grammar library. Their absence is an explicit selection,
not an assertion of feature parity with AMD or upstream.

AMD's sampling challenger regressed its default p-less path; those measurements do not select
a CUDA route. Carried convolution windows, vectorized Vision RoPE/scatter, pitched host KV
transfers, direct host loading, and graph allowance reductions depend on native layout and
measurement. This fork already has its own fused convolution, Vision, transfer, and graph
planning implementations; none of the AMD performance constants or throughput claims are
copied. Spark, multi-rank execution, new checkpoints, MoE offload, QSA/PLE/HC, and the upstream
preemptive scheduler are outside the product contract.

### Verification

Qualification used the RTX 5090, `sm_120a`, CUDA 13.1/GCC 13 Release builder and the explicit
`qwen3.8-nvfp4-flash2-nvfp4-bf16codebook-from-bf16/qwen3_8_27b_nvfp4_dflash_nvfp4.ninfer`
artifact under the existing models mount. Engine and HTTP checks used NVFP4 KV.

- Warnings-as-errors CLI/server and affected-test builds pass. The final fast suite passes
  132 non-real checks; two optional artifact load-plan checks skip. All nine final focused CTest checks pass, including C3 real-model constrained
  generation with ordinary decode and DFlash, frontend/tool grammar, three protocol schemas,
  raw JSON/CLI input, runtime mechanisms, and LayerNorm.
- Raw HTTP checks pass all 14 schema-number rejections: response schemas on Chat/Responses,
  tool schemas on all three protocols, and Responses/Anthropic tools with an extraneous
  `function` field, with streaming both enabled and disabled. Errors are HTTP 400
  before SSE and retain the assertion pointer. Valid constrained Chat generation passes in
  non-streaming and SSE modes after the rejections.
- Tokenizer correctness compares exact IDs against 28 frozen independent official-tokenizer
  goldens, including Unicode, rank ties, long words, and the short/heap boundary. The existing
  official tokenizer/frontend and literal-provenance checks also pass.
- LayerNorm compares BF16 output directly with the existing independent naive FP64 formula
  from represented BF16 inputs, weights, and bias at Vision D1152. Cases cover aligned warp
  and unaligned scalar routes, ordinary and near-constant data, finite extreme BF16 values,
  and finite variance whose addition to epsilon overflows FP32. The existing numerical
  criterion is unchanged: relative L2 1.9e-3, gross absolute 2e-4, and gross relative to maximum
  reference 2.7e-3. Moment-overflow fixtures use exactly representable nontrivial affine outputs to
  isolate normalization overflow from BF16 rounding ties. Memcheck reports zero errors and
  racecheck reports zero errors/warnings. LayerNorm is included in both sanitizer labels.
- Independent reviews of tokenizer invariants/benchmark scope, schema admission/error
  contracts, and LayerNorm mathematics/synchronization completed without unresolved findings.
  Active schema locations are selected by protocol, including flat tools carrying extraneous
  `function` metadata; unsupported Anthropic output formats keep their existing classification.
- Clang-tidy 22.1.8 reports zero diagnostics across 12 affected translation units. Pinned
  formatter/linter/file checks, changed-document reference review, and `git diff --check` pass.

Tokenizer-only timing used the Ryzen 9 7950X3D, GCC 13 `-O3 -std=c++20`, and the same official
Qwen3.8 tokenizer before/after the port. ABBA order used five samples per invocation, 100
encodes per sample for ordinary/identifier text and ten for the long word. Combined median
microseconds per encode were 75.15 -> 67.62 for 5,700 bytes of ordinary prose, 91.09 -> 87.79
for 6,600 bytes of identifiers, and 58,903 -> 694.5 for one 8,192-byte word (84.8x). Token counts
matched. These are CPU tokenizer measurements; no Engine TTFT, decode-throughput, or CUDA
speedup is claimed.

### Per-commit dispositions

The following table accounts for every commit in the reviewed range. “Retain” selects the
current implementation; “exclude” identifies work outside the product contract. The detailed
selected and cross-cutting decisions above govern mixed-purpose commits.

| Upstream | Item | Disposition |
|---|---|---|
| `23b0997d` | perf: reorganize and tune bf16 causal attention | Retain native attention layouts, dispatch, and graph contracts. |
| `4e8939d6` | perf: reorganize and tune fp8 causal attention | Retain native attention layouts, dispatch, and graph contracts. |
| `98ba2dac` | perf: stabilize bf16 attention graphs and live partitioning | Retain native attention layouts, dispatch, and graph contracts. |
| `5a15a166` | perf: unify fp8 attention graphs and query tiling | Retain native attention layouts, dispatch, and graph contracts. |
| `20a36378` | perf: organize and tune int8 causal attention | Retain native attention layouts, dispatch, and graph contracts. |
| `1192ad76` | perf: organize and tune nvfp4 causal attention | Retain native attention layouts, dispatch, and graph contracts. |
| `a637f28f` | perf: organize and tune k8v4 causal attention | Retain native attention layouts, dispatch, and graph contracts. |
| `1737ca11` | refactor: share causal attention primitives | Retain native attention layouts, dispatch, and graph contracts. |
| `a012e2bc` | fix: align attention qualification and engine graph planning | Retain native attention layouts, dispatch, and graph contracts. |
| `582c9a8f` | fix(bench): use varied inputs and restore mutable operands | Retain independent native benchmark/test fixtures. |
| `909fb087` | perf(linear): optimize fp8 34816x5120 with tma split-k | Retain qualified native routes; no unmeasured schedule import. |
| `344d69b8` | perf(linear): tune fp8 14336x5120 for mxfp8 | Retain qualified native routes; no unmeasured schedule import. |
| `7489500d` | perf(linear): tune fp8 16384x5120 decode and tma schedules | Retain qualified native routes; no unmeasured schedule import. |
| `b3f018ab` | perf(linear): tune fp8 5120x6144 tma schedules | Retain qualified native routes; no unmeasured schedule import. |
| `7ede9b44` | perf(linear): tune fp8 5120x17408 tma and split-k routes | Retain qualified native routes; no unmeasured schedule import. |
| `40bfe7dc` | perf(ops): tune fp8 fused projections with tma split-k | Retain qualified native routes; no unmeasured schedule import. |
| `7f6aafed` | perf(attention): optimize fp8 and k8v4 prefill with split-kv | Retain native attention layouts, dispatch, and graph contracts. |
| `c1c48a6a` | bench(serve): make kv cache dtype configurable | Native benchmarks already select KV dtype. |
| `d23835c1` | docs(perf): refresh qwen3.8 fp8 kv serving results | Retain native authorities/tooling; no product change. |
| `84cf93e4` | perf(linear): use native fp8 to bf16 conversion | Retain native codec routes; no FP8 weight-profile expansion. |
| `1cfdb4d6` | perf(linear): use native nvfp4 a16 decoding | Exclude CUDA 13.2-only expansion on CUDA 13.1. |
| `4201b5d2` | fix(runtime): bind dflash prefill controls per chunk | Retain exclusive prefill; no interleaved decode/state-slot migration. |
| `d44ab584` | perf(attention): extend grouped small prefill | Retain native attention layouts, dispatch, and graph contracts. |
| `a667efdd` | refactor(rope): derive launch capacity from device sm count | Retain native ownership and module organization. |
| `417eb3d6` | refactor(ops): derive norm and moe launches from device sm count | Exclude other hardware, model, or distributed execution. |
| `e621c7d6` | refactor(attention): derive launch plans from device sm count | Retain native ownership and module organization. |
| `064965c7` | refactor(kv): require explicit publication streams | Already explicit production publication streams. |
| `75a89050` | fix(runtime): wait for compute before request cleanup | Retain same-stream cancellation and existing failure fences. |
| `b9114396` | feat(runtime): replace context cache and add preemptive scheduling | Exclude upstream context/model runtime replacement. |
| `a8e212ac` | perf(engine): reduce preparation and context management overhead | Port tokenizer; retain native context/transfer ownership. |
| `f854788b` | bench: expand ttft workloads and recovery diagnostics | Retain native workload/recovery diagnostics. |
| `c772812b` | fix: align serving benchmarks and docs with runtime contracts | Retain native serving benchmark contract. |
| `abb7f14f` | feat(serve): expose prometheus runtime and request metrics | Retain existing metrics contract. |
| `68c54356` | fix(frontend): preserve cache boundaries through template trimming | No Jinja consumer. |
| `7e2973ee` | build: vendor xgrammar cpu core | Retain pinned patched CPU grammar dependency. |
| `911d34db` | fix(runtime): preserve cache sources across admission waits | Retain native copy-hold source ownership. |
| `a643abcd` | docs: streamline agent guidance and align references | Retain native authorities/tooling; no product change. |
| `08902f73` | feat: add gbnf constrained decoding across generation backends | Retain EBNF constraints; no GBNF syntax expansion. |
| `eb6696ae` | feat: add json mode and schema constrained decoding | Already supports JSON/schema constraints. |
| `41e50d0d` | feat: add constrained tool calling | Already supports constrained tools. |
| `624dcdd4` | fix(bench): share cold timing quantiles | Retain native timing/report contract. |
| `35e9b5c8` | feat(ops): support and tune q8 linear 2560x6144 | Retain qualified native routes; no unmeasured schedule import. |
| `04ded76f` | feat(ops): support and tune q8 linear 12288x2560 | Retain qualified native routes; no unmeasured schedule import. |
| `8f0aa859` | feat(ops): support and tune q8 linear 10240x2560 | Retain qualified native routes; no unmeasured schedule import. |
| `58ab3f22` | feat(ops): support and tune q8 linear 6144x2560 | Retain qualified native routes; no unmeasured schedule import. |
| `471924b0` | feat(ops): support and tune bf16 linear 48x2560 | Retain qualified native routes; no unmeasured schedule import. |
| `cd521ab7` | feat(ops): support and tune q8 linear 16384x2560 | Retain qualified native routes; no unmeasured schedule import. |
| `f58e32e0` | feat(ops): support and tune bf16 linear 96x2560 | Retain qualified native routes; no unmeasured schedule import. |
| `714c5149` | feat(ops): support and tune bf16 linear 1664x2560 | Retain qualified native routes; no unmeasured schedule import. |
| `bcdcbd45` | feat(ops): support and tune compact bf16 linear 324x10240 | Retain qualified native routes; no unmeasured schedule import. |
| `8beb4cdd` | feat(ops): support and tune bf16 linear 320x10240 | Retain qualified native routes; no unmeasured schedule import. |
| `16bdb491` | feat(ops): support and tune compact bf16 linear 10240x320 | Retain qualified native routes; no unmeasured schedule import. |
| `2553e26e` | feat(ops): support and tune bf16 linear 12800x2560 | Retain qualified native routes; no unmeasured schedule import. |
| `cf91c818` | feat(ops): support and tune full-vocabulary bf16 linear 248320x2560 | Retain qualified native routes; no unmeasured schedule import. |
| `07e1f8c3` | feat(ops): support and tune bf16 linear 13952x2560 | Retain qualified native routes; no unmeasured schedule import. |
| `d0f91cd9` | feat(ops): support and tune bf16 linear 2560x6144 | Retain qualified native routes; no unmeasured schedule import. |
| `070fa61a` | feat(ops): support and tune bf16 linear 2560x2560 | Retain qualified native routes; no unmeasured schedule import. |
| `7971ff18` | feat(ops): support and tune bf16 vision linear 1152x1536 | Retain qualified native routes; no unmeasured schedule import. |
| `c02a07b4` | feat(ops): support and tune bf16 vision linear 3456x1152 | Retain qualified native routes; no unmeasured schedule import. |
| `dd73c88e` | feat(ops): support and tune bf16 vision linear 1152x1152 | Retain qualified native routes; no unmeasured schedule import. |
| `9879d052` | feat(ops): support and tune compact bf16 vision linear 4304x1152 | Retain qualified native routes; no unmeasured schedule import. |
| `a64b5eea` | feat(ops): support and tune compact bf16 vision linear 1152x4304 | Retain qualified native routes; no unmeasured schedule import. |
| `b21780f3` | feat(ops): support and tune bf16 merger linear 4608x4608 | Retain qualified native routes; no unmeasured schedule import. |
| `15cba227` | feat(ops): support and tune bf16 merger linear 2560x4608 | Retain qualified native routes; no unmeasured schedule import. |
| `8ffa3131` | feat: add choice and regex constrained decoding | Do not expand choice/regex surface. |
| `2734a56e` | feat: extend constrained decoding composition and diagnostics | Retain documented schema subset and diagnostics. |
| `81c8ce09` | feat: support tuple schemas and bounded numbers | Port source precision; retain current admitted number/array language. |
| `84d1cf02` | docs: record qwen4exp support and linear plans | Retain native authorities/tooling; no product change. |
| `d8c0fa6f` | feat: add qwen4exp weight conversion and paged nf4 | Exclude other hardware, model, or distributed execution. |
| `dc8b58fb` | docs: define independent qwen4exp op tasks | Retain native authorities/tooling; no product change. |
| `30e3a92f` | docs: plan main workspace qwen4exp tasks | Retain native authorities/tooling; no product change. |
| `be9ad8a3` | feat: define ple shared op contracts | Exclude other hardware, model, or distributed execution. |
| `a0d27b22` | feat: support mixed row spans in paged kv | Exclude upstream context/model runtime replacement. |
| `f2620157` | feat(ops): add qwen4exp gdn shapes and reorganize gdn operators | Exclude other hardware, model, or distributed execution. |
| `fbd285b5` | docs: define expert weight cache and offload design | Retain native authorities/tooling; no product change. |
| `be12563a` | feat(ops): implement ple convolution and state extraction | Exclude other hardware, model, or distributed execution. |
| `d6a8de0e` | feat(ops): add qsa split input projections | Exclude other hardware, model, or distributed execution. |
| `1a9d676e` | feat(ops): add qsa tail extraction | Exclude other hardware, model, or distributed execution. |
| `30b489f6` | feat(ops): add bf16 qsa prepare | Exclude other hardware, model, or distributed execution. |
| `d02703e2` | docs(qsa): remove intermediate bf16 key rounding | Retain native authorities/tooling; no product change. |
| `fc4e3331` | feat(ops): support quantized kv in qsa prepare | Exclude other hardware, model, or distributed execution. |
| `e70fc17f` | feat(ops): add qsa indexer | Exclude other hardware, model, or distributed execution. |
| `47cdc08a` | feat(ops): add bf16 qsa sparse gqa | Exclude other hardware, model, or distributed execution. |
| `fbcd8504` | feat(ops): support quantized kv in qsa sparse gqa | Exclude other hardware, model, or distributed execution. |
| `6a4104fd` | feat: add sigmoid gated rmsnorm and explicitly name silu gate | Exclude other hardware, model, or distributed execution. |
| `f7d3090b` | feat: add 96x2560 gdn control and reorganize norm ops | Retain qualified native routes; no unmeasured schedule import. |
| `c6f47b52` | docs: sync rmsnorm and gdn control documentation | Retain native authorities/tooling; no product change. |
| `70a11f12` | feat: add optimized hc residual ops | Exclude other hardware, model, or distributed execution. |
| `5989682a` | feat: add optimized hc block and final mix | Exclude other hardware, model, or distributed execution. |
| `642d588f` | fix(tests): synchronize hc mix fixture initialization | Exclude other hardware, model, or distributed execution. |
| `f1ace85e` | docs: consolidate model and maintenance references | Retain native authorities/tooling; no product change. |
| `3124dffe` | refactor: streamline execution and op tooling | Retain native ownership and module organization. |
| `90d25edd` | refactor(ops): establish family organization and independent linear split | Retain native ownership and module organization. |
| `7ee63999` | refactor(ops): consolidate shared implementation primitives | Exclude other hardware, model, or distributed execution. |
| `20d3a13f` | refactor(ops): organize basic operator families | Retain native ownership and module organization. |
| `1844d861` | refactor(ops): organize fused operator families | Retain native ownership and module organization. |
| `4eb2c042` | refactor(ops): unify segmented linear projections | Retain native ownership and module organization. |
| `d53eb7ae` | refactor(ops): organize remaining linear variants | Retain native ownership and module organization. |
| `6733be72` | refactor(ops): organize rope and kv cache families | Retain native ownership and module organization. |
| `c012155c` | refactor(ops): organize softmax attention family | Retain native ownership and module organization. |
| `4366c7b6` | refactor(ops): organize linear attention family | Retain native ownership and module organization. |
| `8cc45088` | refactor(ops): organize sampling and sequence operations | Retain native ownership and module organization. |
| `e6585d15` | refactor(ops): organize sparse moe execution | Exclude other hardware, model, or distributed execution. |
| `ae54c1a9` | refactor(ops): finish family migration cleanup | Retain native ownership and module organization. |
| `6e772ee3` | docs(ops): consolidate refactoring guidance | Retain native authorities/tooling; no product change. |
| `6a42de35` | refactor(core): separate memory ownership and cuda execution | Retain native ownership and module organization. |
| `74671a4e` | feat(core): add unified cpu execution foundations | Exclude upstream context/model runtime replacement. |
| `7c4c5e3e` | feat(artifact): add nvfp4 banks and block-fp8 mtp import | Retain qualified native routes; no unmeasured schedule import. |
| `c18d72db` | feat(core): add expert weight sources and gpu replica cache | Exclude upstream context/model runtime replacement. |
| `610f79e9` | refactor(ops): rebuild moe around routed expert interfaces | Exclude other hardware, model, or distributed execution. |
| `8cf26bb6` | refactor(qwen3_5): separate dense mlp and moe execution | Exclude other hardware, model, or distributed execution. |
| `8e3dff94` | feat(ops): add cpu routed swiglu for q4 experts | Exclude upstream context/model runtime replacement. |
| `c7c36e88` | feat(moe): integrate eager offload with reusable expert windows | Exclude other hardware, model, or distributed execution. |
| `fddba033` | fix(artifact): load offloaded weights directly into host backing | Exclude other hardware, model, or distributed execution. |
| `c996baa8` | perf(moe): calibrate default 35b expert priorities | Exclude other hardware, model, or distributed execution. |
| `7ff78ee8` | perf(moe): reduce cpu offload scheduling overhead | Exclude other hardware, model, or distributed execution. |
| `545a171a` | refactor(moe): share execution and offload infrastructure | Exclude other hardware, model, or distributed execution. |
| `17bea0bc` | feat(moe): support cuda graphs for offloaded execution | Exclude other hardware, model, or distributed execution. |
| `19367582` | fix(runtime): calibrate cuda graph startup memory | Retain native graph calibration and capacity planning. |
| `8cea711b` | feat(ple): add nf4 disk reading foundations | Exclude other hardware, model, or distributed execution. |
| `333b4808` | feat(ple): add asynchronous embedding preparation | Exclude other hardware, model, or distributed execution. |
| `1b50c72d` | feat(ops): implement fused ple gate | Exclude other hardware, model, or distributed execution. |
| `569861a3` | perf(ops): specialize 10240-wide offset rmsnorm | Exclude upstream context/model runtime replacement. |
| `9cdbf221` | fix(convert): allow a8 for qwen4exp mtp experts | Exclude other hardware, model, or distributed execution. |
| `69c098bf` | feat(ops): implement nvfp4 moe | Exclude other hardware, model, or distributed execution. |
| `df4ff5fe` | feat(ops): implement block-fp8 moe | Exclude other hardware, model, or distributed execution. |
| `7e691b0b` | feat(ops): add cpu nvfp4 moe and organize moe kernels | Exclude other hardware, model, or distributed execution. |
| `5ac4d9f0` | refactor(ple): separate disk preparation from upload slots | Exclude other hardware, model, or distributed execution. |
| `0a8530d4` | fix(grammar): optimize length-bounded json string masks | Retain current grammar compiler/matcher. |
| `4de9e425` | fix(convert): use dedicated flash next chat template | Retain qualified native routes; no unmeasured schedule import. |
| `9cfd5646` | refactor(models): extract shared qwen integration foundations | Retain native ownership and module organization. |
| `48d8d185` | refactor(context): unify state images and continuation ownership | Retain native ownership and module organization. |
| `e9a7de36` | refactor(context): own kv prefixes and complete snapshots | Exclude other hardware, model, or distributed execution. |
| `ee5a5ffa` | refactor(context): own asynchronous migration and pool resources | Retain native ownership and module organization. |
| `4423e0ac` | refactor(context): finish state migration cleanup | Retain native ownership and module organization. |
| `088e8da4` | feat(qwen4-exp): add loading and execution resources | Exclude other hardware, model, or distributed execution. |
| `985121d4` | feat(qwen4-exp): integrate eager text execution | Exclude other hardware, model, or distributed execution. |
| `4f31ad4a` | chore(profiling): trace qwen4-exp execution costs | Retain native authorities/tooling; no product change. |
| `afe43480` | feat(qwen4-exp): integrate eager vision execution | Exclude other hardware, model, or distributed execution. |
| `439e1075` | feat(qwen4-exp): integrate eager mtp execution | Exclude other hardware, model, or distributed execution. |
| `3444964f` | feat(qwen4-exp): integrate cuda graph execution | Exclude other hardware, model, or distributed execution. |
| `f8a774f7` | perf(qwen4-exp): apply calibrated expert cache priorities | Exclude other hardware, model, or distributed execution. |
| `93fb1874` | chore: standardize agent temporary file storage | Retain native authorities/tooling; no product change. |
| `64d10fbf` | docs(qwen4-exp): consolidate model and execution references | Retain native authorities/tooling; no product change. |
| `8ed68325` | build: separate native rtx5090 and dgx spark targets | Exclude other hardware, model, or distributed execution. |
| `f3ae9a44` | feat(engine): support native dgx spark execution | Exclude other hardware, model, or distributed execution. |
| `066714cc` | refactor(platform): unify target capabilities and roofline references | Retain native ownership and module organization. |
| `91972187` | refactor(tests): consolidate linear family suites | Retain native ownership and module organization. |
| `35aede69` | refactor(build): share ops and core libraries | Retain native ownership and module organization. |
| `f30e661f` | refactor(tests): reorganize op qualification suites | Retain native ownership and module organization. |
| `c827ac62` | perf(linear): import initial spark bf16 routes | Exclude other hardware, model, or distributed execution. |
| `54d5619b` | feat(bench): import explicit linear token lists | Retain native benchmark token controls. |
| `573b97d2` | perf(linear): import spark bf16 96x2560 route | Exclude other hardware, model, or distributed execution. |
| `26023552` | perf(linear): import spark bf16 13952x2560 route | Exclude other hardware, model, or distributed execution. |
| `02f49c7a` | perf(linear): import spark bf16 13952x2560 split | Exclude other hardware, model, or distributed execution. |
| `d0f87935` | perf(linear): import spark bf16 2560x2560 route | Exclude other hardware, model, or distributed execution. |
| `8a3d1b3a` | perf(linear): import spark bf16 12800x2560 route | Exclude other hardware, model, or distributed execution. |
| `d15b1cd4` | perf(linear): import spark bf16 324x10240 route | Exclude other hardware, model, or distributed execution. |
| `468f9b92` | perf(linear): import spark bf16 320x10240 route | Exclude other hardware, model, or distributed execution. |
| `2c8d1490` | perf(linear): import spark bf16 10240x320 route | Exclude other hardware, model, or distributed execution. |
| `82ba758f` | perf(linear): import spark bf16 1152x1536 route | Exclude other hardware, model, or distributed execution. |
| `bddd76b9` | perf(linear): import spark bf16 3456x1152 route | Exclude other hardware, model, or distributed execution. |
| `7f47cf69` | perf(linear): import spark bf16 1152x1152 route | Exclude other hardware, model, or distributed execution. |
| `994d590c` | perf(linear): import spark bf16 4304x1152 route | Exclude other hardware, model, or distributed execution. |
| `caa4a0ca` | perf(linear): import spark bf16 1152x4304 route | Exclude other hardware, model, or distributed execution. |
| `443084cb` | perf(linear): import spark bf16 4608x4608 route | Exclude other hardware, model, or distributed execution. |
| `861a042e` | perf(linear): import spark bf16 2560x4608 route | Exclude other hardware, model, or distributed execution. |
| `bd651673` | perf(linear): import spark bf16 248320x2560 route | Exclude other hardware, model, or distributed execution. |
| `04172b86` | perf(linear): import spark q8 2560x6144 route | Exclude other hardware, model, or distributed execution. |
| `c593b927` | perf(linear): import spark q8 16384x2560 route | Exclude other hardware, model, or distributed execution. |
| `c4ff6b1d` | perf(linear): import spark q8 16384x2560 split route | Exclude other hardware, model, or distributed execution. |
| `f09f791b` | perf(linear): import spark q8 12288x2560 route | Exclude other hardware, model, or distributed execution. |
| `2f59d1c6` | perf(linear): import spark q8 12288x2560 split route | Exclude other hardware, model, or distributed execution. |
| `38ce3b49` | perf(gdn): import spark control projection routes | Exclude other hardware, model, or distributed execution. |
| `bbd1b09c` | perf(gdn): import spark input convolution routes | Exclude other hardware, model, or distributed execution. |
| `a12038ab` | perf(hc): import spark gated residual mix routes | Exclude other hardware, model, or distributed execution. |
| `3008b2b0` | perf(moe): import spark nvfp4 routed execution | Exclude other hardware, model, or distributed execution. |
| `3c750745` | perf(moe): import spark nvfp4 complete execution | Exclude other hardware, model, or distributed execution. |
| `c156c479` | perf(moe): import spark fp8 routed execution | Exclude other hardware, model, or distributed execution. |
| `0aa881d4` | perf(moe): import spark fp8 complete execution | Exclude other hardware, model, or distributed execution. |
| `e3b33a2b` | perf(moe): import spark router dispatch | Exclude other hardware, model, or distributed execution. |
| `551ebef0` | test(gdn): unify numerical criteria and benchmark traffic accounting | Retain native oracle suites and measurement contracts. |
| `22470b06` | perf(gdn): tune spark recurrent execution and short prefill | Exclude other hardware, model, or distributed execution. |
| `f3f66d53` | perf(hc): tune spark residual normalization dispatch | Exclude other hardware, model, or distributed execution. |
| `18ac2480` | test(rmsnorm): cover flash next shapes and bandwidth accounting | Retain native oracle suites and measurement contracts. |
| `4d6316ee` | perf(causal-conv): reuse token windows in spark split execution | Exclude other hardware, model, or distributed execution. |
| `e05c8618` | perf(ple): tune spark fused convolution and skip idle history copies | Exclude other hardware, model, or distributed execution. |
| `f05786bb` | perf(attention): tune spark uniform vision attention tiles | Exclude other hardware, model, or distributed execution. |
| `91d568a6` | perf(vision): tune spark normalization and axial rope execution | Port finite LayerNorm overflow handling; exclude Spark tuning. |
| `26edbb81` | perf(indexing): vectorize spark bf16 embedding and feature scatter | Exclude other hardware, model, or distributed execution. |
| `2e098d5b` | perf(sampling): tune spark token decisions and scoring reductions | Exclude other hardware, model, or distributed execution. |
| `132c81a2` | test(qsa): standardize qualification and benchmark metrics | Exclude other hardware, model, or distributed execution. |
| `438ae67c` | perf(qsa): tune spark indexer and sparse attention execution | Exclude other hardware, model, or distributed execution. |
| `bc70d7c8` | feat(tools): stream native tool call arguments | Retain whole-call tool publication contract. |
| `452c3f5f` | fix(cache): unify recovery scoring and incremental reclaim | Retain native cache/recovery; affected source forest absent. |
| `1408277a` | feat(qwen4-exp): compose resident main and draft graphs | Exclude other hardware, model, or distributed execution. |
| `598285a0` | docs: settle dual-spark execution and parallel strategy | Retain native authorities/tooling; no product change. |
| `1693a6e5` | docs(glm5next): define model quantization and spark op requirements | Retain native authorities/tooling; no product change. |
| `71cdd615` | refactor(runtime): separate native sessions and rank execution | Retain native ownership and module organization. |
| `0a239945` | docs: organize glm5next references and implementation designs | Retain native authorities/tooling; no product change. |
| `f8790a78` | build: vendor tomlplusplus 3.4.0 | No TOML configuration dependency. |
| `4126ecf4` | feat(config): unify cli and serve startup configuration | Retain current CLI/server configuration and media resolution. |
| `fd1baa04` | feat(distributed): add multi-rank execution foundations | Exclude other hardware, model, or distributed execution. |
| `05327b22` | docs: clarify ownership and documentation policies | Retain native authorities/tooling; no product change. |
| `95a651aa` | feat(moe): support expert subsets in complete execution | Exclude other hardware, model, or distributed execution. |
| `68a20575` | feat(linear): add spark parallel projection shapes | Exclude other hardware, model, or distributed execution. |
| `25bd3120` | feat(linear): add spark segmented projection shapes | Exclude other hardware, model, or distributed execution. |
| `69d48cca` | feat(gdn): add spark local control projection | Exclude other hardware, model, or distributed execution. |
| `ca6a8b90` | feat(conv): support local split convolution geometry | Exclude other hardware, model, or distributed execution. |
| `98df04ff` | feat(gdn): support local projection convolution state forms | Exclude other hardware, model, or distributed execution. |
| `895ff3ae` | perf(gdn): tune local spark recurrence dispatch | Exclude other hardware, model, or distributed execution. |
| `57cbc3c9` | feat(gdn): support local recurrence records | Exclude other hardware, model, or distributed execution. |
| `d537da48` | feat(gdn): add local replay fold geometry | Exclude other hardware, model, or distributed execution. |
| `b008a162` | test(norm): qualify local sigmoid rmsnorm geometry | Exclude other hardware, model, or distributed execution. |
| `ababd1b2` | feat(qsa): support local prepare head geometry | Exclude other hardware, model, or distributed execution. |
| `6d64ee2d` | feat(qsa): support local sparse attention geometry | Exclude other hardware, model, or distributed execution. |
| `8c9d12b3` | feat(ple): add sharded ngram embedding lookup | Exclude other hardware, model, or distributed execution. |
| `a70292fc` | feat(sampling): add partitioned vocabulary decisions | Exclude other hardware, model, or distributed execution. |
| `6ccfe2a9` | feat(reduction): add partitioned vocabulary logprobs | Exclude other hardware, model, or distributed execution. |
| `b6537dad` | feat(linear): add clamped swiglu projections on spark | Exclude other hardware, model, or distributed execution. |
| `a4e3f4be` | feat(moe): add complete clamped swiglu execution | Exclude other hardware, model, or distributed execution. |
| `148c159a` | feat(moe): add corrected sigmoid routing on spark | Exclude other hardware, model, or distributed execution. |
| `2616e08d` | feat(moe): add clamped local expert execution on spark | Exclude other hardware, model, or distributed execution. |
| `88105a00` | feat(moe): add complete sigmoid expert execution | Exclude other hardware, model, or distributed execution. |
| `757194af` | test(moe): correct near-zero error criteria | Exclude other hardware, model, or distributed execution. |
| `27df5a34` | feat(distributed): prepare qwen4exp model shards | Exclude other hardware, model, or distributed execution. |

## Verdicts for 0c94153b..e31bc99b

Ported (hand-rewritten; see the fork commit for evidence):

| Upstream | Item | Fork status |
|---|---|---|
| 4cece118 | replace malformed generated UTF-8 with U+FFFD instead of failing the request | 82fb7c5f |
| 780d5767, 8eaed538 | client/tool text keeps literal provenance (no control-token parsing inside content) | 0fe9b7ff |
| ee9d5192 | W4A4 TMA token-fastest CTA raster, activation-scale box fetched once per K-tile pair | 43b49f9e |
| 5f5fccab | W4A4 TMA partial last M tile | 9b8fcce1 |
| 1d8587bc | activation scales in tiles, one TMA request per stage | e800379e |
| 1d13c213 | bind the CUDA device on Engine worker and cache I/O threads | 5b5d3caa |
| b88c0f6f | settle pageable H2D uploads; stream-order sampling-count rollback | 5d366aeb |
| b2b96bae | aligned swscale destination buffers | 610615cd |
| 79c292bc | OpenAI tool messages accept text-part arrays | 909ff2d7 |
| e3aeaf8c | non-empty Anthropic thinking signature (message id) | 565161ad |

Measured before deciding (results in docs/performance.md and the commits):

| Upstream | Question | Outcome |
|---|---|---|
| 00369f63, 1c8f8acc, 0f0899c9 | fused SwiGLU TMA floor | ported on fork evidence: TMA from T=256 with partial tiles (41f331af) |
| 92bb06eb | short-prefill causal conv route | split conv serial kernel only to T=8 (724de290) |
| 5499799d | fused DFlash draft head + top-k | not ported: head + top-k + path select are ~3% of a C=1 round, fusion bound ~2% |
| DFlash2 acceptance gap | sampler vs drafter precision | sampler: fork at upstream's sampler accepts 39.1% at 207.9 tok/s; no W8 drafter port |
| 123bf1a1 | warmup kept out of prefix caches | not ported: one tiny "hi" entry, no correctness effect |
| ce09aee5, ceba8d40 | SSE keep-alive, TCP timeouts | not ported: FIN-based cancel works; keep-alive needs a second writer thread |
| fork-only | tool schemas with format/oneOf/allOf/... returned 400 | relaxed to a superset grammar (1c0b8f5f) |

Not ported, by design:

- Upstream's resource scheduler, host context cache, state-image/paged physical containers and
  their fixes (d6af046a..d4929686 range): the fork's RAM/SSD prefix caches are a different design
  and lack those structures.
- v3 artifact/weight decoupling (99f08f00, 168fdd81, 4cde7ad0, 04350ba9): multi-model generality
  outside the product contract.
- q4/q5/q6/q8/fp8/bf16 linear tuning and template unification, sparse MoE, KDA, groupwise-int
  evals: not on the `qwen3.8-27b/nvfp4` path.
- Upstream KV formats (int8 Hadamard, fp8, nvfp4-g16, k8v4, fp16 V) and the two-stage chunked GDN
  rewrite (0784e76f): different layouts; no evidence of a gain over the fork's own paths.
- Heuristic Qwen tool-parser rewrites (3b50962b, 0c5d570c, 719d56ef, 0e4cdf84): superseded by the
  fork's grammar-constrained tool decoding.
- Upstream DFlash2 ops and integration: the fork has its own DFlash2 with adaptive draft counts and
  C<=6 packed verify; upstream fixes 03177b91, f08597d6, 863aa8a5 are already covered.
- Approximate SiLU (05507ab0): reverted upstream (c4ae8a9c); never adopted.
- Logging (spdlog), llama.cpp timings, cpp-httplib 0.54 upgrade, perplexity evaluator, jinja
  engine import: the fork has its own equivalents or does not need them.
