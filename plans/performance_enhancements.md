# Performance enhancements

Living log of attempted host/runtime speedups. Published GPU tok/s and TTFT
live in `docs/performance.md`. Update this file when an attempt is measured
or abandoned so the next pass does not repeat it.

Measure product-path CPU with `NINFER_BENCH_ENCODE=1` on
`ninfer_qwen3_6_frontend_test` (Release `-O3`). That splits chat render, BPE
encode, full-sequence `Tokenizer::decode`, and per-token `decode_token_bytes`.
Do not use GPU tok/s to judge these diffs; generate is kernel-bound.

`Engine::prepare` re-renders and re-encodes the full prompt every turn.
Prefix reuse saves GPU KV, not this host encode.

## Product-path costs (current baseline)

RTX 5090 box, official Qwen3.6 tokenizer, two-sided HEAD vs experimental
chat-template recut. Token counts identical.

| case | tokens | render | encode | prepare |
|---|---:|---:|---:|---:|
| `plain_32k` | 32,807 | 0.016 ms | 3.96 ms | 4.0 ms |
| `plain_150k` | 150,049 | 0.148 ms | 21.2 ms | 21.4 ms |
| `tools_200_150k` (200 calls, 200 schemas, 1.5 KiB args) | 178,706 | 0.54 ms | 5.3 ms | 5.8 ms |
| `massive_tools` | 1,094,845 | 2.7 ms | 30.7 ms | 33.4 ms |

Generate-round host work (`decode_token_bytes` + output session) is
~0.05 µs/token and sits under `cudaSynchronize`. It is not a tok/s lever.

## Open

- **Full-prompt BPE of many short words.** Still the cold-prepare cost when
  there is no reusable prefix (first turn, rewritten history). A later attempt
  must beat the table above on `plain_150k` and `tools_200_150k` encode_ms
  without changing token identity. Do not revive the stack-128 rewrite.
  `test_official_tokenizer_merge` is necessary but not sufficient; compare
  token counts on the bench fixtures.

## Shipped

### Incremental host encode (turn-2+ prepare)

Engine PIMPL cache: longest memcmp prefix at a Tokenizer loop-pos, cap 16 /
48 MiB. `Frontend::prepare` stays cold. See
[`incremental-host-encode.md`](incremental-host-encode.md).

`NINFER_BENCH_PREPARE=1` on `ninfer_qwen3_6_incremental_encode_test`
(Release): cold `plain_150k` prepare 22.7 ms, second-turn append 1.5 ms,
hit flag true, ids match cold encode. Metric is second-turn `prepare`,
not GPU tok/s.

## Closed — do not repeat

### Chat-template `string_view` / `reserve` / single-text fast path

Dropped. Render of a 150k-token plain prompt went 0.148 → 0.020 ms; 200 tool
calls were unchanged (~0.55 ms). Encode did not move. The save is ~0.13 ms
on a 21 ms prepare, or nothing on the tool-call turn. Not worth the extra
fast path vs `rendered_content()`.

### `Tokenizer::decode` via `validate_utf8` (no codepoint vector)

Dropped from the product recut. Full-sequence detokenize on `massive_tools`
went ~61 → ~7 ms, but serving/CLI generate uses `decode_token_bytes`, which
did not change (~0.11 ms on `plain_32k`).

### BPE rewrite (stack-128 nodes, cached pair ranks, heap for n>128)

Reverted. Encode nearly doubled on tool fixtures (`large_tools` 1.23 → 2.12 ms,
`massive_tools` 32.6 → 64.4 ms). The bench words are short identifiers; every
word paid for a 128-node stack array. Any new BPE design has to win on short
words, not long unsplittable ones.

### NFC skip, `ids.reserve`, 8-byte `is_ascii`, output-delta `string_view`,
position `memcpy`, prefix-identity resize

Dropped. Encode stayed within noise of baseline. Isolated `OutputSession`
preview was ~0.05 µs/token either way.

## How to add an entry

State the product metric (usually `encode_ms` on `plain_150k` /
`tools_200_150k`), the before/after mean, whether token counts matched, and
keep or drop. Move the attempt to Closed if it is not shipping.
