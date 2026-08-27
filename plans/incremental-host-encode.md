# Incremental host encode

Cut `Engine::prepare` encode cost on turn 2+ by concatenating cached
committed-history token ids with a BPE of only the unmatched suffix.

Oracle: incremental `input_ids` and rewrite-checkpoint frontier must match
a cold `Tokenizer::encode` / `encode_rendered_chat` of the same rendered
text. Any doubt is a miss and a full encode. Silent drift is worse than
21 ms.

Downstream never sees this. Closed host-CPU attempts stay in
`plans/performance_enhancements.md`.

Do not implement until the lookup is **prefix search + memcmp**, not
SHA-256 of this turn’s committed text. Hashing the new committed blob
cannot hit on an append: turn 2’s committed string is `[U1,A1,U2]`, a
new key, so the 21 ms path would never be skipped.

## Isolation

Not a serving feature, CLI flag, `PromptInput` field, or Engine option.
`include/ninfer/engine.h` does not change.

Cache lives in `Engine::Impl` (`mutable`, mutex). CLI/HTTP/eval call
`Engine::prepare` as today.

`Frontend::prepare(PromptInput)` stays a cold encode (in-tree frontend
tests). Engine calls an **impl-header** `prepare_with_cache` that
`engine.cpp` includes. Do **not** put `EncodedHistoryCache` or a cache
argument on `src/targets/qwen3_6/export/ninfer/targets/qwen3_6/frontend.h`.

`prepare_tokens` never touches the cache. Destroying the Engine drops it.
Two Engines do not share.

Lock only for lookup / copy-out / insert. Copy committed **bytes and ids**
out under the lock; `encode(suffix)` and cold `encode(full)` run **outside**
the lock. Never return a reference into the LRU vector (use-after-evict).

## Why

Release `-O3`, official Qwen3.6 tokenizer, `NINFER_BENCH_ENCODE=1`:

| case | tokens | render | encode / prepare |
|---|---:|---:|---:|
| `plain_150k` | 150,049 | 0.15 ms | 21 ms |
| `tools_200_150k` | 178,706 | 0.54 ms | 5.3 / 5.8 ms |

Win is the **second** prepare of a long history plus a small append.

## Non-goals

- Concatenative BPE for arbitrary cuts.
- Cold first-encode / short-word BPE rewrite.
- Chat-template `string_view` recuts.
- GPU KV, prefix identity, Vision skip.
- Mid-text leftover-word streaming.
- User-visible knobs, metrics, serving docs.
- v1: Vision `Processor` path (media → always cold encode).

## Tokenizer fact

`Tokenizer::encode` with `parse_added_tokens` walks added specials
(leftmost match, lowest `token_index` on ties, advance by
`content.size()`, `best_len` ignored) and BPE+NFC’s only the gaps.
Legal splice `pos` values are exactly the loop positions: `0`, each
`match_pos`, each `pos` after a match, and `text.size()` after leftover
BPE.

A local “some added token starts or ends at `n`” probe is **not** that
walk (a special can straddle `n`; a local hit can be a token the real
loop never emits). Concat then disagrees with cold `encode(full)`.

**Loop-pos helper** lives on `Tokenizer` (same trie as encode): skip BPE,
scan from byte 0, require `n` to appear as `match_pos` or post-match
`pos` (or `n==0` or `n==size`). Phase 1 tests this against cold encode.
If a chat boundary disagrees, **stop**.

## Safety (0 wrong tokens)

SHA-256 is **not** identity. A collision with verify off is another
conversation’s ids. That is not 0 risk.

**Hit only after `memcmp` of stored committed bytes against the prefix
of `full` used to splice.** SHA-256 is a bucket / index only.

**Must all hold or miss:**

1. Text-only (no media). v1.
2. `parse_added_tokens` on.
3. Two renders, same messages, other options equal:
   `committed = render(..., add_generation_prompt=false).text`
   `full = render(..., caller add_generation_prompt).text`
   Require `full.starts_with(committed)`. Do not strip a guessed
   generation suffix off one string. If this invariant fails for a
   template, miss forever on that path (phase 1 stops the line).
4. Longest cached entry whose stored bytes are a prefix of `full`
   (memcmp), with `n = stored.size()`, and `n` is a Tokenizer loop-pos
   of `full`.
5. v1 rewrite: hit only if no checkpoint or
   `checkpoint.offset >= n`. Recompute frontier this turn; do not
   cache `RewriteCheckpointSpec`. Kind comes from this render.
6. On hit: `suffix = full.substr(n)`; `encode(suffix)` with added
   tokens; concat. Copy ids out first. If suffix encode throws,
   propagate; do not return partial ids.
7. Frontier: same rules as `encode_rendered_chat` today.
   `rel = checkpoint.offset - n`. Hit frontier is
   `committed_ids.size() + suffix_prefix_tokens`. Treat **`rel==0` as
   frontier = committed_ids.size()** (do not pass `prefix_byte_end==0`).
   `encode("")` itself does not throw; `encode_rendered_chat` throws on
   an empty prefix fallback. If `rel>0` and `prefix_tokens` missing or
   0, **miss and cold-encode `full`**. Kind from this render.

**Insert this turn’s committed** after both miss and hit, so turn 3+
still splices a short suffix. Never insert empty bytes (`n==0` prefixes
every `full`). On a verify mismatch, drop the bad entry and insert from
the **cold** ids; do not extend poisoned hit ids.

Proven ids without a second full encode: `n = committed.size()` is a
loop-pos of `full`; `encode(full.substr(n))` must be an exact id suffix
of the full ids (the generation opener is tiny). On a hit with
`hit.n <= n`, extend `hit.ids + encode(committed.substr(hit.n))`.
Do not insert a non-loop-pos cut. Do not slice `full` ids by guessing.

**Verify mode:** `NINFER_VERIFY_HOST_ENCODE=1` or debug/ASan: after a
would-be hit, cold-encode `full` and compare ids and frontier.
Mismatch → return cold ids, drop the entry. Allowed to abort in that
build. Verify does **not** replace memcmp. Default Release: verify off
only after phase 2 is green **with memcmp lookup**.

`enable_thinking` is **not** in a separate key. ThinkingToggle only
changes the generation suffix; committed bytes already cover
`tool_jsons`, effort, `preserve_thinking` (those rewrite the prefix).

## Cache record and size (picked)

Store committed UTF-8 **and** ids. Private `constexpr` on the cache
type, **not** `EngineOptions` / CLI / HTTP.

| Knob | Value |
|---|---|
| Entry cap | **16** |
| Max ids / entry | **262144** (`TokenId` × 4 = 1 MiB) |
| Max committed UTF-8 / entry | **2 MiB**; larger → skip insert |
| Worst-case RAM | 16 × (1 MiB ids + 2 MiB bytes) = **48 MiB** |
| Eviction | LRU on last hit or insert |
| Index | Linear memcmp of ≤16 entries; no SHA-256 |

Why 16: `kMaximumConcurrency` is 8; extra slots for `count_tokens` /
prepare replay and one extra history generation. Prefix search is
≤16 memcmps.

Why 2 MiB text: ~4 bytes/token of real chat at 256k is ~1 MiB; 2 MiB
is the insert ceiling.

Typical live use: one entry per distinct committed history among ≤8
requests, far under 48 MiB.

## Algorithm (Engine text prepare)

1. Two renders as above (or one render if `add_generation_prompt` is
   already false: `committed == full`).
2. Media → existing processor, no cache.
3. Scan cache for longest stored bytes that memcmp-match a prefix of
   `full` at a loop-pos. None → cold `encode_rendered_chat`; insert if
   proven slice.
4. Hit: encode suffix outside the lock; concat; frontier as above;
   missing `prefix_tokens` → cold full encode.
5. Verify mode.
6. Same `PreparedPrompt` shape as today.

`Engine::count_tokens` uses the same helper (ids only; frontier is
prepare-only). `count_tokens` text path today is `encode(rendered.text)`
with no checkpoint; ids must still match prepare.

## Required miss

- Media / v1 Vision.
- No cached prefix of `full` at a loop-pos.
- `full` does not start with this call’s `committed` when only
  `add_generation_prompt` differs (invariant failure).
- Checkpoint offset `< n`.
- `preserve_thinking=false` after an assistant `<think>`: earlier
  think block is stripped, old committed is not a prefix → miss.
- One tool then two consecutive tools: delayed `<|im_end|>` → miss.
- Empty committed; `prepare_tokens`; direct `Frontend::prepare`.
- Insert skip if committed UTF-8 > 2 MiB.

Replay (`committed == full`, including `add_generation_prompt=false`):
`n == full.size()` **is** a loop-pos. Empty suffix. Do not require an
added token to start at `n`.

ThinkingToggle `enable_thinking` flip, same messages: committed bytes
unchanged → **hit**, encode the new generation suffix only.

## Test plan

Oracle: `tokenizer.encode(full)` and, with a checkpoint,
`encode_rendered_chat` frontier. Official tokenizer. No LRU-shape tests.

**Hit must be observable.** Miss also equals cold encode, so id
equality alone does not prove a hit. Phase 2 reads a **test-only
thread-local** `HostEncodeObservation` from the impl header after
`EncodedHistoryPrepare` / `Engine::prepare`. `Frontend::prepare` does
not touch it. Tests fail on unexpected miss, not only on wrong ids.

### Phase 1 — splice helper, no Engine

| case | expect |
|---|---|
| split at loop-pos after `<\|im_end\|>` / `<\|im_end\|>\n` then gen opener | concat == cold full |
| late system append | concat == cold |
| cut mid user string or splitting `<\|im_end\|>` | refuse |
| `full` does not start with `committed` | refuse |
| CJK/NFC inside a gap, split after that turn’s `<\|im_end\|>` | concat == cold |
| `add_generation_prompt` false vs true, **both templates** | `full.starts_with(committed)`; suffix is only the opener |
| tool-group growth; think-strip (`preserve_thinking=false`) | refuse / not a legal prefix splice |

Stop if any legal chat boundary disagrees with cold encode.

### Phase 2 — Engine cache

Go through `Engine::prepare` / `count_tokens`. `Frontend::prepare` remains
cold.

**Hit (flag true, ids + frontier == cold)**

- `prepare(M)` then `prepare(M + user)` with `preserve_thinking=true`
  (or no assistant think block).
- Same `M` twice; `count_tokens` then `prepare`.
- Late system append.
- Generation suffix only; ThinkingToggle `enable_thinking` flip
  (committed hit, new suffix).
- Replay `add_generation_prompt=false` twice (empty suffix).
- TurnClosure frontier (checkpoint after `<\|im_start|>assistant\n`)
  and ResponseReplay (`preserve_thinking`, checkpoint after `<think>\n`).
  `<think>` / `</think>` are added tokens (248068 / 248069).

**Miss (flag false, ids still == cold)**

- `preserve_thinking=false`, assistant with `<think>`, then new user.
- `tool_jsons` or `reasoning_effort` change.
- One tool result, then two consecutive tools.
- Image/video; `prepare_tokens`.

**Isolation / concurrency**

- Only `Engine::prepare(PromptInput)` — no extra arguments.
- Two Engines: A does not affect B.
- 8 threads, distinct histories; 8 threads, identical history; all ==
  cold. Copy-out so eviction cannot UAF.

**Verify**

- `NINFER_VERIFY_HOST_ENCODE=1` on the Engine tests.
- Inject poisoned ids → return cold, drop entry.

### Phase 3 — measurement

`NINFER_BENCH_PREPARE=1`: Engine prepare `M` at `plain_150k`, then
`M + short user`. Keep if second prepare is render + suffix encode
(not ~21 ms) **and** ids match cold prepare **and** the hit flag is
true. Cold first prepare within noise of the baseline table. No GPU tok/s.

`tools_200_150k` second-turn: same oracle; time drops unless the append
rebuilds a tool group (miss is correct).

## Phases

1. Tokenizer loop-pos helper + phase-1 tests. Stop on disagreement.
2. Engine PIMPL cache (prefix + memcmp), impl-only prepare, phase-2
   tests, verify mode.
3. Bench gate. Verify off in default Release only after phase 2.

## Ownership

- `src/targets/qwen3_6`: loop-pos on Tokenizer, two-render committed/full,
  cache type in an **impl** header, `prepare_with_cache`.
- `src/runtime` Engine PIMPL: instance, mutex, copy-out, call
  `prepare_with_cache` from `prepare` / `count_tokens`.
- `src/serve`, CLI, `include/ninfer/*`, export `frontend.h`: no cache API.
- Ops, KV, prefix identity: untouched.

## Missing / not in v1

- Host skip of prefix Vision preprocess.
- Persist cache across process restart.
- Stats in `request_done`.
- Splice inside an open tool-response group.
- Public disable flag.
