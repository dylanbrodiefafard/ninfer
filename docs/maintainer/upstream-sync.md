# Upstream sync log

Upstream is [Neroued/ninfer](https://github.com/Neroued/ninfer) (`git remote add upstream
https://github.com/Neroued/ninfer`), branch `dev`. This fork diverged heavily after the merge-base,
so upstream work is ported by hand, not merged.

## Review watermark

| Field | Value |
|---|---|
| Last upstream commit reviewed | `e31bc99b13f517c8aae70b997b7c4a49b4dcdc5d` (2026-09-26, `docs: align linear guidance and refresh q4 performance report`) |
| Merge-base | `0c94153b` (2026-08-16) |
| Reviewed on | 2026-09-26 |

The next review starts from `git log e31bc99b..upstream/dev`; every commit up to and including
the watermark has been judged and needs no second pass.

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
