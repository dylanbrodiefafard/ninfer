# Qwen4 weight, activation and state precision recommendations

Research conducted 2026-09-17; recommendations recorded 2026-09-18.

This is a research recommendation for a **future smaller Qwen4 checkpoint**, using
Qwen3.8-Flash-Next as the available architectural and quantization evidence. It is not a
qualified artifact recipe, an implementation assignment, or a replacement for the model,
artifact and Op contracts in `docs/maintainer/qwen4-*.md`. Apply each row only if the future
checkpoint retains that component. Architecture implementation is separate work.

## Interpretation

- **A4/A8 describe temporary GEMM operands**, not the storage dtype of every intermediate.
  Ordinary activation outputs remain BF16 unless a separately qualified fused route chooses
  private intermediate arithmetic. Weight storage, accumulation and persistent state are
  separate decisions.
- **Candidate** means supported enough to evaluate, not already proven to preserve maximum
  quality on the future checkpoint. Its original unquantized execution is the quality reference.
- FP32 accumulation/reduction entries are recommended implementation profiles where stated;
  they do not redefine the mathematical oracle or assert that all reference arithmetic is FP32.
- NInfer's current Qwen3.8-27B qualifies A4 computation for particular matrix shapes and
  operators. Its cutovers and protected-layer list do not transfer automatically to Qwen4.
- Weight-only quantization results do not establish activation-quantization tolerance.
  NVFP4, GPTQ INT4 and MLX INT4 are different numerical profiles.
- A4 can be numerically usable without being the fastest route at a small token count.

## Weight and activation matrix

| Component | Recommended weight storage | Recommended activation/computation | Lower-precision alternative and evidence |
|---|---|---|---|
| Routed expert gate/up projections | **NVFP4** | **A4**, FP32 accumulation, BF16 output | Strongest direct Flash-Next W4A4 evidence; NInfer also qualifies these operations. Retain A16 routes where faster at small token counts. |
| Routed expert down projection | **NVFP4** | **A4**, independently calibrated post-SwiGLU input; FP32 accumulation | Supported by expert W4A4 recipes and NInfer. Do not reuse gate/up activation calibration blindly. |
| Expert activation: SiLU and multiplication | No weights | BF16 output; fused internal arithmetic separately qualified | No need to store the intermediate persistently in A4. Quantize at the down-GEMM boundary. |
| Expert output weighting and combination | Router coefficients retained at appropriate precision | FP32 weighted accumulation, BF16 output | Avoid low-bit partial-sum storage. |
| Routed-expert router | **BF16** | BF16 input/projection boundary; **FP32 softmax**, exact top-k indices | Keep protected. Reference behavior does not imply that router logits are produced or stored entirely in FP32. |
| Shared-expert gate/up projections | **FP8 candidate; BF16 reference** | **A8 candidate**, BF16 output | Direct community W8A8 evidence. W4A4 deserves testing, but NInfer's dense MLP results alone do not qualify it. |
| Shared-expert down projection | **FP8 candidate; BF16 reference** | **A8 with separate post-activation calibration** | Same distinction; calibrated W4A16 is another memory-oriented candidate. |
| Shared-expert scalar output gate | **BF16** | BF16 input; FP32 accumulation/nonlinearity where appropriate | Keep protected separately from the shared MLP. |
| QSA query projection | **FP8 candidate; BF16 reference** | **A8 candidate**, BF16 projected Q | W4A16 has limited evidence; W4A4 is a further experiment supported by analogy to NInfer. |
| QSA key projection | **FP8 candidate; BF16 reference** | **A8 candidate**, BF16 projected K | Qualify separately: key errors affect attention scores and cached history. |
| QSA value projection | **FP8 candidate; BF16 reference** | **A8 candidate**, BF16 projected V | Evaluate independently from Q/K rather than assuming equal sensitivity. |
| QSA output projection | **FP8 candidate; BF16 reference** | **A8 candidate**, BF16 residual boundary | A4 is worth testing; current NInfer retains some higher-precision output projections. |
| QSA Q/K normalization and positional rotation | BF16 norm parameters | BF16 represented outputs; FP32 normalization reductions | Preserve operation order and cast boundaries. |
| QSA core attention QK/PV | No additional projection weights | **BF16 operands initially; FP32 score/softmax reductions** | FP8 attention operands require separate qualification. W8A8 projection evidence does not establish this. |
| QSA indexer projections | **BF16** | BF16 projected queries/keys | Keep protected initially; changing block selection can discard relevant context. |
| QSA indexer pooling and scoring | BF16 norm parameters | **FP32 pooling/score reductions**, exact selected indices | Matches the important reference precision boundaries. |
| GDN Q/K/V projections | **FP8 candidate; BF16 reference** | **A8 candidate**, BF16 operator outputs | NInfer justifies investigating W4A4; Flash-Next evidence is stronger for W8A8. |
| GDN Z/output-gate projection | **FP8 candidate; BF16 reference** | **A8 candidate**, higher-precision gate evaluation | Qualify separately from Q/K/V despite possible fusion. |
| GDN output projection | **FP8 candidate; BF16 reference** | **A8 candidate**, BF16 residual output | A4 remains a candidate, not an established Qwen4 default. |
| GDN A/B control projections | **BF16** | BF16 inputs; **FP32 decay/update computation** | Keep protected. |
| GDN decay parameters and time-step bias | Preserve source information; FP32 runtime representation where used | **FP32 exponentials/softplus/control arithmetic** | Tiny storage savings do not justify aggressive quantization. |
| GDN convolution | **BF16** | BF16 history boundary; appropriate FP32 accumulation | Do not silently change the recurrent history representation. |
| GDN normalization and output gating | BF16 parameters | BF16 output; FP32 reductions/control evaluation | Separate from projection operand precision. |
| Token embedding | **Eight-bit weight candidate; BF16 reference** | **BF16 gathered output** | Weight-only evidence supports evaluating eight bits. A gather has no GEMM activation operand. |
| PLE n-gram embedding table | **FP8** | **BF16 gathered values** | Four-bit storage is a more aggressive capacity candidate; runtime dequantization placement matters. |
| PLE key/value projections | **BF16 initially** | **A16 initially** | Eight-bit weights are supported as a candidate; A8 needs separate activation evidence. |
| PLE content gate, norms and convolution | **BF16** | BF16 represented values; FP32 reductions/nonlinearities | Protect independently from the large lookup table. |
| Hyperconnection read projections | **BF16 initially** | **A16** | Eight-bit weights are a credible next step. Weight ablations specifically argue against indiscriminate four-bit storage; they do not qualify A8. |
| Hyperconnection write gates, mixing parameters and norms | **BF16** | BF16 represented values; FP32 reductions where appropriate | Keep protected. |
| LM head | **Eight-bit weight candidate; BF16 reference** | **A16**, FP32 accumulation where appropriate | W4A16 is a more aggressive demonstrated candidate; A4/A8 head inputs are not established here. |
| Vision encoder and multimodal merger | **BF16 initially** | **A16 initially** | NInfer's quantized vision paths are useful implementation experience, but text-only Flash-Next results do not qualify their quality. |
| MTP projections and experts | Preserve a validated checkpoint-specific mixed recipe | Preserve its validated activation policy | Evaluate draft acceptance and total throughput; do not inherit the 27B DFlash recipe automatically. |

## Persistent and cached values

| Stored value | Recommendation | Alternative |
|---|---|---|
| Ordinary hidden activations/residual outputs | **BF16** | Local A4/A8 GEMM packing, without changing the persistent boundary |
| Gated multi-stream residuals | **BF16 baseline** | **FP8 storage is a priority experiment supported by the paper**; gate/mixing computation stays higher precision |
| GDN recurrent matrix | **FP32** | BF16 only after specific long-sequence state/quality qualification |
| GDN convolution history | **BF16** | No established reason here to lower it |
| QSA main K/V cache | **BF16 baseline** | FP8 after long-context quality and kernel qualification; NVFP4 is not justified by the external quality evidence reviewed here |
| QSA indexer keys | **BF16** | Quantization requires separate block-selection evidence |
| Indexer partial pooling accumulators | **FP32** | Do not conflate these with the stored key dtype |
| PLE convolution/history values | **BF16** | Qualify separately from table compression |
| Sampling probabilities and sensitive reductions | **FP32** | Keep distinct from the stored logits dtype |
| Token IDs, hashes, positions and selected indices | **Exact integers** | No approximate quantization |

The sandbox's separately authorized diagnostic NVFP4 QSA-KV profile is not superseded by this
research recommendation. A diagnostic implementation and evidence of maximum-quality suitability
are different claims.

## Connection to current NInfer

The inspected Qwen3.8-27B NVFP4 recipe has 247 NVFP4 Text matrix parents, nine protected BF16
large projection parents, W8 vocabulary endpoints, BF16 GDN control weights and FP32 recurrent
state. Its public activations are generally BF16. Eligible projection/MLP routes quantize
operands to A4 locally; ordinary single-token NVFP4 linear decode uses A16. Short recurrent and
speculative routes have additional arithmetic/state-boundary rules.

That provides direct implementation evidence for A4 gate/up, post-SwiGLU down, attention and
large GDN projections. It supports testing those same *kinds* of boundaries in Qwen4. It does
not prove that a different checkpoint, shared expert, sparse selector or hyperconnection has
the same sensitivity. In particular:

- Expert W4A4 has both analogous NInfer experience and direct Flash-Next evidence.
- QSA/GDN/shared large projections retain W4A4 as a serious candidate, while W8A8 has stronger
  direct Flash-Next activation evidence.
- QSA core attention operands and cache codecs require separate qualification from its
  projection GEMMs.
- GDN state/control precision is not inferred from large-projection precision.
- No routed router, QSA indexer, PLE or Qwen4 hyperconnection exists in the current dense 27B
  target; their recommendations rely on the external evidence and numerical reasoning.

Relevant repository references: `docs/maintainer/qwen3.6-27b-artifact.md`,
`docs/maintainer/qwen3.8-27b-artifact.md`, `docs/maintainer/qwen3.6-27b-model.md`,
`src/ops/linear/nvfp4/nvfp4_config.h`, the NVFP4 Linear/SwiGLU/LinearAdd route planners,
and `src/targets/qwen3_6/impl/runtime/dflash_impl.h`.

## Evidence and limits

The evidence below was researched on 2026-09-17; publisher experiments were not reproduced.
Source branches and model cards can change. Different publishers' numbers are not a matched
Pareto comparison.

- **NVIDIA Flash-Next NVFP4:** main routed experts use W4A4 with MSE-calibrated scales while
  other main-model paths retain BF16. Its broad benchmark comparison supports this mixed
  compression point, not uniform W4A4 or guaranteed losslessness for a smaller model.
  https://huggingface.co/nvidia/Qwen3.8-Flash-Next-NVFP4
- **PipeNetwork weight ablations:** matched WikiText2 PPL is 4.4708 for BF16, 5.3914 for uniform
  INT4 and 4.5286 for mixed four/eight-bit. Restoring hyperconnection read projections alone
  to eight bits gives 4.9744; restoring the attention/GDN/shared/PLE projection group gives
  4.8969. This identifies weight sensitivity, not A8 activation tolerance. The mixed recipe
  is larger, so it is not strict domination on every memory/quality objective.
  https://huggingface.co/pipenetwork/Qwen3.8-Flash-Next-MLX-mixed-4_8bit
- **Senfu dense W8A8:** QSA and large GDN/shared projections use FP8 W8A8, with W4A16 head.
  The evaluation is narrower, uses two B200s, and compares against published baseline numbers
  rather than a paired rerun. Claimed weight-traffic savings are calculated, not measured
  throughput. Activation calibration is described through MoE-block input maxima without
  adequate per-projection characterization, especially for post-activation down inputs.
  https://huggingface.co/senfu/Qwen3.8-Flash-Next-NVFP4
- **GPTQ:** routed/shared experts and QSA projections have a reported INT4 weight-only point
  with small corpus-level perplexity changes. This does not qualify NVFP4 A4 computation.
  https://huggingface.co/btbtyler09/Qwen3.8-Flash-Next-GPTQ-4bit/raw/main/README.md
- **Architecture paper:** reports FP8 gated-residual storage with little quality loss, but
  does not establish FP8 gate math, FP8 recurrent state or a universally applicable storage
  codec. *On the Design of Qwen3.8-Next Architecture: Evaluation, Efficiency, and Training
  Stability*, arXiv 2608.30320v1.
  https://arxiv.org/html/2608.30320v1
- **Official configuration and reference:** FP32 GDN state, protected conversion exclusions,
  FP32 router softmax and FP32 indexer reductions inform the recommended control boundaries.
  https://huggingface.co/Qwen/Qwen3.8-Flash-Next-FP8/raw/main/config.json
  https://raw.githubusercontent.com/huggingface/transformers/main/src/transformers/models/qwen4_exp/modeling_qwen4_exp.py
- **PLE storage versus runtime representation:** the RadixArk recipe distinguishes FP8 PLE
  storage from BF16 dequantized use. Check where decoding occurs before making a residency
  or bandwidth claim.
  https://huggingface.co/RadixArk/Qwen3.8-Flash-Next-NVFP4

## Decision priorities

The recommended aggressive starting point is **expert W4A4**. For shared experts, QSA
projections and large GDN projections, **W8A8 is the next evidence-backed candidate**, with
W4A4 explicitly retained for evaluation. Protect router/indexer decisions, hyperconnection
gates and GDN state. Evaluate FP8 gated-residual storage separately from its computation.

There is no established maximum-quality/speed frontier for the future smaller model. Qualify
routes against independent mathematical oracles from represented inputs and decoded weights;
assess weight-quantization loss separately against the original checkpoint. Use paired held-out
quality measurements, long-context retrieval, recurrent behavior and termination where affected.
Measure actual prefill/decode throughput at relevant token counts and concurrency, including
packing overhead and speculative acceptance, rather than assuming lower bits are faster.
