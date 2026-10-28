# HTTP serving

`build-r9700/apps/ninfer-serve` loads one Qwen3.8-27B R9700 artifact and serves OpenAI- and Anthropic-compatible HTTP endpoints.

## Start the server

```bash
./build-r9700/apps/ninfer-serve /path/to/qwen3.8-27b-r9700-lq4.ninfer \
  --host 127.0.0.1 \
  --port 8080 \
  --max-context 16384 \
  --kv-capacity 32768 \
  --max-concurrency 2
```

When `--model-id` is omitted, the server advertises and accepts the loaded container's exact `identity.model_id`. An explicit `--model-id` remains a public HTTP alias override and does not select or alter the artifact.

Vision is disabled by default, so its weights and buffers are not allocated. Media generation and media token-count requests return HTTP 400 `vision_disabled`. Add `--vision` when the server must accept image or video input. Speculative weights and buffers are also selected at startup with `--spec none|mtp|dflash` and `--draft-tokens`. The default is DFlash with the optimized proposal head, adaptive drafting, and a maximum of seven draft tokens. `--spec none` omits speculative weights and state; `--no-lm-head-draft` selects the full proposal head. Qwen3.8-27B DFlash2 is available when `dflash/` is present and can be combined with `--vision`; the text-only companion consumes Vision-composed target hidden features. DFlash uses chain verification `W=k+1` for `k` in `1..7`. Choose draft lengths using end-to-end R9700 measurements.

`--adaptive-draft` picks live k in `{3..N}` (DFlash `--draft-tokens N>=5`; MTP `{3,4,5}`) after each round by locking `argmax E[Y(k)] / T(k,C,L)` from nested hop-survival `r_i` and online least-squares round time (shared slope, per-k intercept). An unmeasured k is probed at most once and dropped when dominated; switching k costs 1 ms. The policy keeps adapting during generation; see [adaptive drafting](maintainer/qwen3.8-27b-model.md). Frozen `--draft-tokens 4` stays `{4}`. `--spec dflash` defaults to seven drafts and adaptive length; `--no-adaptive-draft` selects a fixed length. Restart the server to enable a capability omitted at startup.

Server startup has a 180-second inactivity watchdog covering device initialization, model loading and warmup. Advancing load progress renews the deadline; repeated or regressing counters do not. A stalled startup exits with status 124 without HIP cleanup, allowing the process supervisor to retry instead of remaining indefinitely unhealthy after a GPU reset. The watchdog is disarmed after warmup and imposes no request, queue or idle timeout. It bounds recovery startup; it does not prevent a GPU/driver hang or guarantee that a damaged device can recover without intervention.

## Endpoints

| Method and path | Behavior |
|---|---|
| `GET /health` | inference availability: HTTP 200 with `{"status":"ok"}` when available; HTTP 503 with `{"status":"unavailable"}` after worker failure or while stopping |
| `GET /metrics` | Prometheus text exposition 0.0.4 of the process snapshot |
| `GET /metrics.json` | the same snapshot as JSON |
| `GET /v1/models` | configured OpenAI model alias |
| `GET /v1/models/{id}` | lookup of the configured alias |
| `POST /v1/chat/completions` | OpenAI-style chat generation |
| `POST /v1/responses` | OpenAI Responses Core generation, state, typed Items, and SSE |
| `POST /v1/responses/input_tokens` | Responses prompt-token count without generation |
| `GET /v1/responses/{id}` | retrieve a locally stored terminal Response |
| `DELETE /v1/responses/{id}` | delete a locally stored Response |
| `GET /v1/responses/{id}/input_items` | list that Response's normalized input Items |
| `POST /v1/messages` | Anthropic-style message generation |
| `POST /v1/messages/count_tokens` | checkpoint-native expanded input-token count |

## OpenAI Chat Completions

```bash
curl http://127.0.0.1:8080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "qwen3.8-27b",
    "messages": [
      {"role": "system", "content": "Answer concisely."},
      {"role": "user", "content": "What is speculative decoding?"}
    ],
    "max_tokens": 128
  }'
```

The endpoint supports:

- `system`, `developer`, `user`, `assistant`, and `tool` history;
- string content and ordered text, `image_url`, and `video_url` parts; tool messages accept a string or text parts only;
- `max_completion_tokens` and the legacy `max_tokens` spelling;
- `temperature`, `top_p`, `top_k`, presence/frequency penalties, and a nonnegative `seed` (top-p/top-k/penalties are ignored by default p-less; `--no-p-less-sampling` opts out);
- one stop string or an array of stop strings;
- `logprobs` and `top_logprobs` (0 to 20, which requires `logprobs: true`); see [Token log probabilities](#token-log-probabilities);
- non-streaming responses and server-sent event streams;
- `stream_options.include_usage`;
- function tools, tool choices, assistant tool-call history, and tool-result messages. Engine-owned constrained generation produces `tool_calls` only when the request includes current tool declarations and `tool_choice` is not `none`. Tool history alone does not authorize a new function. Markup on a tools-off request stays in `content` and ninfer-serve logs a warning.
- the top-level `reasoning_effort` field;
- the `enable_thinking` extension;
- `chat_template_kwargs.preserve_thinking` and the top-level `preserve_thinking` alias.
- the vendor `ninfer` object (`cache_write`, `capture_context_checkpoint`).

The request `model` is informational: NInfer runs one resident model, so it serves that model regardless of the identifier and echoes the requested value back (llama.cpp-compatible). The advertised alias is the artifact `identity.model_id` by default, or the explicit `--model-id` override. Reasoning is returned separately as `reasoning_content`; answer text remains in `content`.

Chat output allows one structural reasoning close, with the bounded replay-repair exception described below when generation recovery is enabled. Requests with thinking disabled already start in the content phase and cannot generate the dedicated closing token as prose. Plain requests exclude that token through sampling; a speculative round commits through the reasoning close before resampling its continuation. Tool grammars reject closing-think markup in free prose, including byte-split forms, while preserving literal tags inside schema-constrained arguments. Response JSON constraints and raw completion output retain their literal contents. This is a framing guard, not text replacement: ordinary-token literal examples are retained in unconstrained text.

For structured Qwen output, registered model stop tokens are excluded from selection while reasoning is open and after `</think>` until non-whitespace answer content begins. Tools-enabled output also excludes them for an ambiguous `<tool_call>` prefix and until its matching `</tool_call>` is complete; this applies independently to consecutive parallel calls. Model stop tokens become eligible after ordinary answer content or a complete tool call. If one speculative round crosses into a protected state before selecting a model stop token, NInfer rejects that uncommitted round and retries with the registered model stop tokens excluded. If a round sampled with those tokens excluded later completes a tool call, NInfer commits only through that close and lets the next round sample with model stops eligible, so extra-accepted tokens cannot immediately open another call.

Other request-provided stop conditions remain active, and raw-output requests do not apply this structured-output guard.

Message roles retain their input order through schema translation. The Qwen family frontend maps both `system` and `developer` to system-class ChatML blocks at their original positions; it does not move later instructions to the beginning of the conversation. A leading instruction keeps the artifact template's existing tool/reasoning-instruction composition.

Only chat-template markup encodes as control tokens. Client-supplied text (message content, reasoning history, tool-call names and arguments, tool results, and tool definitions) is literal: a file or tool result that spells `<|im_end|>`, `<tool_call>`, `<think>`, or a vision placeholder encodes as ordinary text and cannot open or close a turn, a tool call, or a media slot.

At startup, NInfer resolves prompt capabilities from the exact `frontend/chat_template.jinja` resource embedded in the loaded artifact. It does not infer them from the request's `model` field, the artifact identity, or a target profile. A recognized effort-capable template exposes `low`, `medium`, and `xhigh`; omitting effort uses that template's declared default (`xhigh` for the recognized effort-capable template). Set `--default-reasoning-effort low|medium|xhigh` to override omitted request effort; omission or `template` retains the artifact default. Startup rejects a configured effort unsupported by the loaded template. Explicit request effort takes precedence, including `none` to disable thinking. A server effort default applies only when thinking is enabled; it does not turn thinking back on after `--no-thinking` or an explicit request disabling it. Docker uses `NINFER_DEFAULT_REASONING_EFFORT`, default `low`. An explicit effort not exposed by the loaded template returns HTTP 400 with code `reasoning_effort_not_supported` before prompt preparation.

For Chat Completions, `reasoning_effort: "none"` disables thinking. `low`, `medium`, and `xhigh` select the corresponding template effort when available. The other OpenAI protocol values `minimal`, `high`, and `max` are parsed but rejected when the loaded template does not expose them. `enable_thinking` controls the same new-turn thinking switch; a contradictory combination with `reasoning_effort` returns `conflicting_template_option`.

`preserve_thinking` controls whether reasoning from closed assistant turns remains in later prompts. It defaults to the server setting, which is off unless `--preserve-thinking` is used. The supplied Docker Compose command also leaves it off; requests can enable it individually. If both OpenAI spellings are present they must carry the same boolean value. Unknown non-null `chat_template_kwargs` are rejected.

Chat Completions and Responses accept a strict vendor object:

```json
{ "ninfer": { "capture_context_checkpoint": true } }
```

Unknown keys inside `ninfer` return HTTP 400. `true` snapshots the chat’s current resume frontier `E` on an exact-hit / decode-only request into the one-slot turn-rollback head (the same slot automatic occupy-append already writes). A later `true` at a new `E` replaces that pin; ladder heads stay. If a ladder or rollback head already sits at this `E`, the request succeeds with `captured_tokens = 0` and does not fill the rollback slot. A brand-new chat (`E == 0`) is the same quiet no-op. `true` on a server started with `--spec none` or with `--no-prefix-reuse` returns HTTP 400 `context_checkpoint_unavailable` before the request is enqueued. Anthropic Messages does not accept this object. `POST /v1/responses/input_tokens` allows `ninfer` only when omitted or JSON `null`.

`ninfer.cache_write` is a strict boolean defaulting to `true`. Set it to `false` for a disposable turn that may read cached state without consuming or overwriting the original chat:

```json
{ "ninfer": { "cache_write": false } }
```

A VRAM hit copies the selected prefix and complete continuation state into an independent lane. A RAM/disk hit restores without consuming the source. If keeping the ordinary VRAM source prevents admission, the Engine attempts to save its full state and existing checkpoints to RAM before freeing its lane; this preserves the original, not the disposable result. All active lanes cause normal queueing. If an otherwise idle Engine cannot preserve the source using a separate lane, configured RAM capacity, or a preserving cold-prefill path, the request returns HTTP 429 `server_overloaded` with a capacity explanation. A disposable source itself is never saved to RAM to make another disposable turn possible.

The disposable result can remain in VRAM for an exact continuation. It is evicted before ordinary retention when a lane or shared pages are needed and never goes to RAM/disk, including at idle or shutdown. It captures no context ladder, turn-rollback, or turn/response rewrite checkpoint; existing checkpoints may be read, and execution snapshots required for speculation/recovery remain. Edited or rerendered follow-ups may therefore need prefill. A later `cache_write: true` continuation can make its own resulting state normally cacheable. Originals remain subject to ordinary cache eviction after initialization; this option does not permanently pin or purge them.

Combining `cache_write: false` with `capture_context_checkpoint: true` returns HTTP 400 `cache_write_conflict`, parameter `ninfer.capture_context_checkpoint`, before enqueue or streaming: `capture_context_checkpoint=true requires cache_write=true because it creates a retained context checkpoint.` Saving an original may increase RAM/disk capture counters and copy time, but the disposable request reports zero captured context-checkpoint tokens. Request-log schema 19 records `request.cache_write` on generation and preparation-rejection events.

Streaming begins with an assistant-role chunk, sends separate reasoning and content deltas, then a finish-reason chunk, an empty `choices` usage trailer, and `[DONE]`. The finish chunk and trailer both contain the same completed usage, including when `stream_options.include_usage` is false. Non-stream responses and each stream usage/finish chunk place the per-request stats in two OpenAI-standard details sub-objects (no top-level duplicates, no Ollama-compat aliases):

- `usage.prompt_tokens_details` — OpenAI-standard `cached_tokens` (prompt tokens served from prefix reuse, no recompute) plus engine stats under the `ninfer` namespace: `reuse_source` (`none` / `vram_resident` / `host_ram`), `prefix_reuse_path` (`full_reset` / `append_frontier` / `restore_turn_checkpoint` / `restore_response_checkpoint` / `restore_context_checkpoint` / `restore_turn_rollback`), `context_checkpoint` (`restored_tokens` / `captured_tokens`), `ttft_ms`, `prefill` (`tokens`, `ms`, `tok_s`, `ms_per_token`, `tail_tok_s`, `tail_window_s`) and `decode` (`tokens`, `ms`, `tok_s`, `ms_per_token`).

  `ttft_ms` is the same value as `[req] done` `ttft`: HTTP prepare plus engine time to the first generated token (admission wait, Vision encode, and prefill, including the first-token sample). It is not `prefill.ms`. Prefill `tokens` / rates cover the computed (non-reused) suffix only. `tail_tok_s` is that suffix's throughput over the trailing ≤1s of prefill GPU time and excludes Vision encode (which has its own `vision` phase in the request log).

  Decode `tokens` are `max(0, completion_tokens - 1)`: when a first generated token exists it is sampled during prefill (and is in `prefill.ms` / TTFT), so `tok_s = tokens / (ms / 1000)`. Clients that divide `usage.completion_tokens` by `decode.ms` will overstate decode tok/s (2× on a two-token completion). `decode.ms` is the GPU round time after host ingress and Device Graph install; it does not include CPU packing or graph-profile switching. Speculative GDN fold / compact after a round is still in `decode.ms`. Rate and millisecond fields are rounded to three decimal places.

  `cached_tokens` is the reused prefix length for any reuse path. `reuse_source` is where that prefix lived (`vram_resident` vs `host_ram`). `context_checkpoint.restored_tokens` is the absolute staged-checkpoint head frontier this request restored (the same length as `cached_tokens` on `restore_context_checkpoint` / `restore_turn_rollback`). It is 0 when the path is not one of those two. `context_checkpoint.captured_tokens` is the absolute advertised ladder freeze or turn-rollback pin this request wrote, 0 if it did not freeze or pin.
- `usage.completion_tokens_details` — OpenAI-standard keys: `reasoning_tokens` (thinking portion, 0 when thinking is off) and, when speculation is active, `accepted_prediction_tokens` / `rejected_prediction_tokens`.

When this request copied KV through a host tier, the `ninfer` namespace carries `kv_ram` with this request's HIP D2H/H2D `save_ms` / `load_ms` and/or `kv_disk` with its `save_ms`, SSD-to-host `load_ms`, and post-disk `h2d_ms`. Each object is omitted when the request did not copy through that tier. Process occupancy and lifetime capture/restore/eviction/drop counters are on [`GET /metrics`](#metrics), not on the per-request usage object.

### Token log probabilities

`logprobs: true` adds `choices[0].logprobs` to the response, and `top_logprobs: N` adds the `N` most likely alternatives at each position:

```json
{"content": [{"token": "42", "logprob": -0.0021, "bytes": [52, 50],
              "top_logprobs": [{"token": "42", "logprob": -0.0021, "bytes": [52, 50]},
                               {"token": "41", "logprob": -6.83, "bytes": [52, 49]}]}],
 "refusal": null}
```

- **Distribution.** Each value is the log-softmax of the target model's logits at temperature 1 over the whole vocabulary, at the position that produced the token. Sampling temperature, p-less truncation, penalties, and tool-grammar masks do not participate, so the values are comparable across requests and a token the sampler was steered to can have a low logprob. Speculative decoding does not change them: every emitted token is scored by the target model's verification of that position, never by the draft. Ranked alternatives are ordered by logit, lower token id first among equal logits.
- **Which tokens.** `content` lists the tokens of the answer text. Reasoning tokens, reasoning markers, tool-call markup of a tool-enabled request, and stop tokens are not listed.
- **Token alignment.** Entries follow tokens, not released text. In a stream each content chunk carries the tokens committed with it; a chunk may carry tokens with `content: ""` while their text is held for a stop-string or UTF-8 boundary, and text released later arrives with an empty list. A token trimmed from `content` by a stop string keeps its entry.
- **`token` and `bytes`.** `bytes` is the token's exact byte string. `token` is the same bytes as UTF-8 with each ill-formed sequence replaced by U+FFFD, because one token can hold part of a multi-byte character.

A request that asks for logprobs adds one scoring pass per decode round, shared by the requests batched in that round. With reporting disabled, the kernel exits before reading logits.

### Multimodal request

Start the server with `--vision` before sending media:

```bash
curl http://127.0.0.1:8080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "qwen3.8-27b",
    "messages": [{
      "role": "user",
      "content": [
        {"type": "image_url", "image_url": {"url": "https://example.com/image.png"}},
        {"type": "text", "text": "Describe this image."}
      ]
    }],
    "max_tokens": 128
  }'
```

OpenAI image and video sources may be HTTP(S) URLs or base64 data URLs.

Text and media prompts share the `--max-context` limit, checked after chat-template rendering and media-token expansion. There is no separate 32K multimodal prompt ceiling. Media preprocessing retains independent resource limits for source bytes, decoded pixels, media-item count, raw patches, and Vision tokens. `attention_pairs` remains preparation diagnostics and is not an admission limit.

An expanded prompt beyond `--max-context` returns HTTP 400 `context_length_exceeded`, including the prepared token count and configured context ceiling. A media preprocessing resource rejection returns HTTP 400 `media_budget_exceeded`. HTTP 413 `request_too_large` is reserved for a raw request body that exceeds `--max-request-mib` before JSON parsing; it is not used for model-context or media resource errors.

## OpenAI Responses Core

NInfer implements the Items and streaming events at the core of the OpenAI [Responses API](https://developers.openai.com/api/reference/resources/responses/overview). All registered artifact identities use the same adapter and Engine. OpenAI-hosted tools, durable cloud storage, background jobs, Conversations and compaction are not supported.

### Create a Response

```bash
curl http://127.0.0.1:8080/v1/responses \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "qwen3.8-27b",
    "instructions": "Answer concisely.",
    "input": "What is speculative decoding?",
    "max_output_tokens": 128,
    "store": true
  }'
```

The same endpoint works with OpenAI SDKs by replacing their base URL:

```python
from openai import OpenAI

client = OpenAI(base_url="http://127.0.0.1:8080/v1", api_key="local-secret")
response = client.responses.create(
    model="qwen3.8-27b",
    instructions="Answer concisely.",
    input="What is speculative decoding?",
    max_output_tokens=128,
)
print(response.output_text)  # SDK helper derived from response.output
```

`output_text` is an SDK convenience property. It is not emitted as a top-level wire field; the wire response contains typed `output` Items.

### Create request fields

| Field | NInfer Responses Core contract |
|---|---|
| `model` | required non-empty string; informational — served by the one resident model and echoed back |
| `input` | required string or non-empty typed Item array |
| `instructions` | optional string, inserted before the reconstructed conversation for this request only |
| `previous_response_id` | optional ID of a retained local Response |
| `max_output_tokens` | integer at least `16`; default is `--default-max-tokens` |
| `stream` | boolean; `true` selects Responses SSE rather than a JSON body |
| `store` | boolean, default `true`; controls local retrieval and continuation state |
| `temperature` | finite number in `[0,2]` |
| `top_p` | finite number in `[0,1]` |
| `metadata` | at most 16 string pairs; keys at most 64 characters and values at most 512 |
| `reasoning.effort` | `none` disables thinking; `low`, `medium`, or `xhigh` selects an effort exposed by the loaded chat template; `minimal`, `high`, and `max` return `reasoning_effort_not_supported` for the registered templates |
| `chat_template_kwargs.preserve_thinking` | optional boolean controlling whether closed-turn reasoning remains in reconstructed prompts |
| `preserve_thinking` | top-level alias for the same option; conflicting values are rejected |
| `text.format` | text, JSON object, or named JSON schema; see constrained output below |
| `tools` | flat Responses function definitions; see below |
| `tool_choice` | `auto`, `none`, `required`, or `{"type":"function","name":"..."}` |
| `parallel_tool_calls` | omitted or `true` |
| `truncation` | omitted or `disabled`; overlong input fails instead of silently dropping Items |
| `top_logprobs` | integer in `[0,20]`; a positive value enables token logprobs |
| `service_tier` | omitted, `auto`, or `default`; the response reports `default` |
| `background` | omitted or `false` |
| `include` | omitted, empty, or `["message.output_text.logprobs"]` to enable token logprobs |
| `stream_options` | omitted or `{"include_obfuscation":false}` |

Unknown top-level fields fail with `unknown_parameter`. Recognized but unsupported features fail with a field-specific 400 error instead of being silently ignored.

### Input Item contract

String `input` is normalized to one user `message` with an `input_text` part. Array input accepts:

| Item | Supported form |
|---|---|
| `message` | roles `user`, `assistant`, `system`, and `developer`; string content or typed content array |
| `input_text` | message content part containing string `text` |
| `output_text` | assistant-message replay part containing string `text` |
| `input_image` | user-message part with HTTP(S) or data-URI `image_url`; detail omitted or `auto`; requires server `--vision` |
| `input_video` | NInfer extension with HTTP(S) or data-URI `video_url`; requires server `--vision` |
| `reasoning` | raw replay Item with an empty `summary` and `reasoning_text` content parts |
| `function_call` | completed assistant call with optional `id`, and required `call_id`, `name`, and JSON-object string `arguments` |
| `function_call_output` | completed tool result with required `call_id` and string `output` |

Adjacent function-call Items are grouped into one assistant history turn. A reasoning Item attaches to the following assistant message or function call. Input Item IDs are preserved when supplied and generated otherwise; duplicate IDs fail.

System and developer message Items retain their positions in the input array. Top-level `instructions` is represented as a leading developer turn for the current request; target-specific role lowering occurs only in the Qwen family frontend.

`input_file`, `input_audio`, image `file_id`, non-`auto` image detail, reasoning summaries or encrypted reasoning, message `phase`, and other Item/content types are not supported. HTTP media URLs stored in a response chain are fetched again when that chain is continued; use data URIs when the historical media bytes must be immutable.

### Function tools

Responses function definitions are flat rather than Chat Completions' nested `function` object:

```json
{
  "type": "function",
  "name": "get_weather",
  "description": "Get current weather",
  "parameters": {
    "type": "object",
    "properties": {"city": {"type": "string"}},
    "required": ["city"]
  },
  "strict": false
}
```

NInfer renders these definitions in the Qwen prompt. The Engine constrains Qwen tool envelopes and supported argument schemas during ordinary and speculative target sampling, validates completed calls, and returns typed values for separate `function_call` output Items. This runs only when the request includes current tools and `tool_choice` is not `none`. If the model emits Qwen `<tool_call>` markup on a tools-off request, NInfer leaves it in the answer text and logs a warning; it does not invent a structured tool call. Each completed output has a protocol Item `id` (`fc_...`) and a distinct `call_id` (`call_...`). The client executes the function and sends a `function_call_output` Item in a later request. NInfer does not execute functions.

Tool schemas use the shared subset under constrained output below, including string pattern/Unicode-length conjunctions and supported multi-branch `allOf`. Selected narrowing assertions are deliberately relaxed, leaving the client responsible for them: `format` and `content*` are annotations; `oneOf` widens to `anyOf`; `not`, `propertyNames`, `uniqueItems`, `multipleOf`, `contains`/`minContains`/`maxContains`, `dependentRequired`/`dependentSchemas`, `if`/`then`/`else`, and `unevaluatedProperties`/`unevaluatedItems` are not enforced. The prompt retains the original schema. Other unsupported assertions, malformed schemas/patterns, and unresolved references fail preparation with HTTP 400 `invalid_tool_schema`. `strict:true`, hosted/MCP tools, and custom free-form tools remain unsupported.

When available, the OpenAI error `param` is a source JSON Pointer: Chat `/tools/N/function/parameters`, Responses `/tools/N/parameters`, or Anthropic input schema `/tools/N/input_schema`, followed by the schema-relative location. Named selection preserves the original index and tool name; references and composition retain source locations rather than generated definitions. Errors without a specific location retain `param: tools`. Anthropic includes the name and location in its own error message.

`tool_choice:"required"` requires at least one declared function call. A named Responses choice uses the flat object `{"type":"function","name":"weather"}` and restricts calls to that declared function. Both modes require current tools; unknown names are rejected. The shared token grammar enforces a call as the first non-whitespace content after optional thinking, with schema-valid arguments; filtering tool declarations alone is not the enforcement. Subsequent prose and calls remain allowed, and the normal function-call SSE events and completed Items are unchanged. Chat Completions and Anthropic required/named choices use the same enforcement. Output limits or cancellation can still interrupt a call and produce an incomplete response; a required choice does not manufacture a completed call.

Qwen argument framing uses exactly one LF before and after each parameter value. Declared arguments may appear in any order, independently of the order of `properties` in the supplied schema or its HTTP JSON normalization. The grammar tracks which named keys have been emitted: it cannot close before the required keys are present, repeat a named key, or bypass a named value's constraints through `additionalProperties`. Property-count bounds still apply.

Nested schema-defined JSON objects use the same order-independent rules, including through local references and supported alternatives. Completed-call validation rejects duplicate keys (including in nested JSON values); arbitrary additional key names are checked for duplication during this final validation, not by the finite named-key grammar. Nested JSON additional keys alongside named properties use canonical printable ASCII without quotes, backslashes, or escapes, preventing aliases of declared keys. Final validation rejects repeated arbitrary keys and never publishes calls that fail value or uniqueness checks.

These rules apply on ordinary and every speculative target position, with grammar state committed only for published tokens. They do not shorten reasoning, force an end-of-turn after a call, or change sampling/recovery settings.

Only those framing bytes are removed; indentation, trailing newlines and literal XML-looking text inside a string are preserved. For schemas admitting both a raw string and a JSON value, the lossless raw-string interpretation is preferred when the complete argument object satisfies the schema.

Grammar compilation shares property rules and builds reference-only subset transitions directly. Tokenizer prefix metadata follows byte prefixes, including duplicate and Unicode spellings. Completed-call validators consume UTF-8 bytes without model-vocabulary masks. The resident frontend caches both immutable grammars; matcher state stays private to each request. A joined worker builds byte validators alongside vocabulary masks on a cold preparation.

With declared tools, the free-text region after reasoning reserves the protocol prefixes `</invoke`, `</parameter`, `</function`, `</tool_call`, `<invoke`, `<parameter`, and `<function`. The prefix `<tool_call` must continue into a canonical declared call. These prefixes cannot be sampled as orphan prose before, between, or after calls; omitting `>` or appending whitespace/`= null` does not evade the constraint. Legitimate Qwen call frames and these literal strings inside schema-valid tool arguments remain allowed. The restriction is part of the grammar domain on ordinary and speculative target positions, not a post-response text scrubber or a recovery retry. While reasoning is open, `<tool_call` is also excluded: a real call must follow `</think>` rather than being rehearsed inside reasoning. Natural-language reasoning, XML tags outside these reserved prefixes, inter-call prose, and tools-off/raw output remain allowed. The engine never promotes reasoning markup into executable calls. Literal discussion of these reserved delimiters in tool-enabled answer prose must use an escaped representation rather than the exact strings. Likewise, literal discussion of the tool opener inside reasoning must escape that opener.

For text-only thinking requests, Engine can withhold an unproductive repeated call batch and retry internally before publishing it. The detector requires two consecutive identical cycles of one to four complete tool rounds, with the same ordered parsed call batches and unchanged ID-associated text results. Every phase must repeat substantial reasoning: the existing single-call/single-round case requires an identical 64-word passage; multi-round or multi-call cases require identical whole reasoning after whitespace normalization, with at least 64 words. The new proposal must repeat the entire next expected call batch and its reasoning. A batch containing a new call, changed results or arguments, an intervening user turn, media, incomplete rounds, or ordinary short polling do not meet this evidence. These observations do not prove the absence of external side effects or task progress. The detector does not infer tool semantics from tool names. A justified repeat with changed reasoning is allowed in multi-round/multi-call cases.

Recovery keeps the cached prompt, including real conversation content/results and historical reasoning. It closes the previous assistant prologue with its matching terminator and appends feedback explicitly stating that the rejected proposal was not executed; only the failed generation is omitted. A later retry appends another notice after the first. A ready checkpoint at the prompt frontier is restored and only the appended suffix is prefilled; when no resident, RAM, or disk checkpoint matches, the spliced prompt is prefilled from an empty KV image. It permits at most two retries within the original completion token budget and resource reservation. It neither executes tools nor forces EOS. Already streamed reasoning/prose remains visible; rejected calls are never published. A novel valid call is not a guarantee that the model has made useful progress.

Recovery-enabled requests also detect persistent generated reasoning independently of sampler and temperature; supported text-only thinking contexts can retry before a tool call exists. Three non-overlapping occurrences of an identical 256-token reasoning passage must appear within the current generated attempt. Their separation may differ; this catches multi-paragraph loops whose periods change after a one-token intervention. In addition, repeated passages must cover at least 4,096 distinct redundant tokens; overlapping windows cannot count the same tokens twice. This conservative threshold allows shorter loops to escape naturally without a recovery retry. It does not cap reasoning at 4,096 tokens. Hashes only locate candidates; exact generated-token comparison proves the match. Prompt tokens and previous attempts cannot supply occurrences. Two copies alone do not trigger a retry. It is not a reasoning-length timeout.

The retry keeps the cached prompt, including historical reasoning, the original task, and completed tool results. It closes the previous assistant prologue with its matching terminator and appends an explicitly labeled engine system notice; the failed attempt's generated tokens are not part of the retry prompt, and a later retry keeps the earlier notice. Only the appended suffix is prefilled after a checkpoint restore. It creates no assistant tool call, tool result, or user message. Already streamed reasoning remains visible, separated from the retry by a blank line, and charged to completion usage. Reasoning, framing replay and duplicate-tool recovery share the same maximum of two retries.

Reasoning evidence uses only contiguous reasoning-channel tokens observed before publication, and remains established when the same licensed round closes reasoning, produces an answer/tool, or reaches a natural stop or output/context boundary. The engine withholds that round's unpublished answer/tool deltas before retry or error; it does not replace previously published answer content. Genuine reasoning remains visible and charged. Standard and temperature-zero sampling use the same proof and bounded text-only thinking retry. Media and other inputs without a supported retry context fail explicitly on proved repetition; their prompts are not spliced. Raw output is excluded. Disabled thinking does not manufacture reasoning evidence, and repeated tool polling without substantive repeated reasoning remains allowed.

With generation recovery enabled, a thinking response can quarantine its initial answer while its generated token IDs exactly copy the preceding generated reasoning passage (at least 256 tokens). Genuine reasoning still streams normally. A divergent answer, an extension beyond the passage, or an uncorroborated terminal copy releases held text and its original logprobs unchanged. Only a complete exact copy licenses one additional dedicated `</think>` token: that consumed token confirms a framing replay, discards the held copy, and rearms quarantine. The extra close is not published or fed to ordinary tool grammar, but remains in generated IDs, KV state and completion usage. Another complete copy, or natural termination without a separate answer or selected tool after repair, requests an unpublished retry with a labeled engine notice. Caller stops and cancellation retain precedence. Repair and proof boundaries end the accepted speculative prefix; hypothesis masks license the extra close only at the matching boundary. Raw, schema-constrained and tool-argument contents are excluded. The original output budget and shared two-retry cap still apply; exhaustion is explicit. Intentionally requesting a long exact reasoning copy with an extra close can trigger this policy. After a confirmed framing replay triggers a retry, the new assistant continuation uses the model's native non-thinking prologue. The original input and historical reasoning stay intact; only the new generation mode changes. The frontend carries the already-corroborated immutable reasoning reference into these replacements, so a complete copy of that same passage triggers recovery without licensing another close. A subsequent closed-thinking prompt is ended with a turn terminator alone, preserving all earlier notices without inserting an extra reasoning close. All replacement answer/tool deltas and their logprobs remain withheld until ordinary terminal repetition/tool validation accepts the attempt; reasoning remains visible. A failed replacement is discarded and may retry only within the remaining shared budget. Explicit caller stops release held text; cancellation releases text while omitting provisional tool events. For an initially non-greedy p-less request, these replay-recovery attempts use standard sampling at temperature 1.0, top-k 20, top-p 0.95, min-p 0 and zero presence/frequency penalties, retaining the resolved request seed. Initial generation and standard/greedy request profiles are unchanged. The actual retry profile and `enable_thinking` mode are recorded on `retry_started` recovery events. A replacement that exhausts its output/context budget without non-whitespace answer text or a complete tool call fails explicitly; caller stops and cancellation retain precedence. This is framing recovery and publication validation, not a check that an answer or external task is correct.

Recovery-enabled requests also detect persistent repetition in free-text answer content, including when thinking is disabled. The exact generated-token evidence is the same three-copy/256-token/4,096-redundant-token criterion described above, restricted to contiguous answer-channel spans in the current attempt. Reasoning, tool arguments, and framing gaps cannot join spans. Raw and schema-constrained output are excluded, and monitoring ends once native tool publication selects a tool for this response, so literal closing markup inside arguments cannot re-enable it. Once established, repetition fails with `generation_recovery_exhausted` and a `repeated_content` recovery cause, including if natural EOS occurs in the same decode round. Already published prose is preserved; it is not retried or replaced, and no normal success terminator is sent after the error. Intentionally long exact repetitions can trigger this guard; `--no-generation-recovery` disables it. This detects repeated output, not factual errors, unfinished sentences, or whether an external task is complete.

Explicit caller string/token stops and cancellation terminate without a recovery retry. A natural model end-of-turn can still expose a confirmed duplicate call and trigger recovery.

If a confirmed repeat cannot be recovered within those bounds, the request fails with `generation_recovery_exhausted` (`server_error`, HTTP 500 before streaming headers, or the protocol's stream error after headers). The engine remains available. When recovery occurs, vendor usage at `prompt_tokens_details.ninfer.recovery` reports attempts, discarded calls, `discarded_reasoning_tokens` (omitted internally, not retracted), `framing_repairs`, `discarded_content_tokens`, additional prefill tokens/samples, and preparation/prefill milliseconds. Discarded replay tokens remain charged to completion usage; their text and logprobs are omitted together. Completion usage includes tokens spent on discarded attempts; original prompt usage is unchanged. Raw output does not use internal recovery.

For an unintervened diagnostic baseline, launch with `--no-generation-recovery`. This startup-only option disables repetition-driven token exclusions, internal reasoning/framing-replay/duplicate-tool retries, replay quarantine and repair, and the answer-repetition failure guard for every request. It leaves tool grammar, p-less sampling at all speculative target positions, and ordinary stopping/cancellation/budgets unchanged. Omit the flag to restore the default enabled policy. Disabled recovery can allow a loop to consume the entire output budget; it is not recommended as a general cure for loops. The startup log explicitly records when recovery is disabled.

Wire captures remain accurate with recovery enabled, but streamed reasoning can contain multiple attempts. Correlate them with the server's request-scoped recovery logs. A missing `prompt_tokens_details.ninfer.recovery` object does not establish zero cycle exclusions: that object reports discarded-generation retries, and framing repairs, whereas cycle exclusions are logged.

The server emits live `[req N] recovery` console records for streaming and non-streaming requests on all three HTTP surfaces. `retry_triggered` identifies `repeated_reasoning`, `reasoning_replay`, `repeated_content` or `duplicate_tool_call`; `replay_discarded` records held replay omission; `retry_started` marks the retry prefill, and `retry_prefill_complete` marks its completion. `exhausted` is a warning before the request error. `finished` reports the terminal outcome (including cancellation or output limits), not a claim that the task succeeded. Records include started attempt count, cumulative discarded reasoning/tool/content counts and framing repairs, generated tokens and remaining output budget. `cycle_exclusion` means one root-token exclusion was armed for sampling, not that a retry occurred or that it changed the counterfactual draw. To bound traffic these records appear at counts 1, 2, 4, 8, ...; terminal recovery records include the exact cumulative count. These diagnostics contain no prompt, reasoning, tool arguments, or token text, and never enter HTTP model-output chunks. Tool grammar is continuously enforced rather than a recovery trigger; its normal operation does not emit recovery events.

With `--kv-ram-system-prompts N`, the RAM cache can retain N distinct rendered system prefixes even when the first user message differs. The feature defaults to zero/off and shares `--kv-ram-capacity`; it does not enable disk caching. `/metrics.json` exposes `kv_ram.system_prompts` with `limit`, `entries`, `used_bytes`, `restores`, and `evictions`. Prometheus exposes the corresponding `ninfer_kv_ram_system_prompt_limit`, `_entries`, `_used_bytes`, `_restores_total`, and `_evictions_total` series. The per-request completion log includes `system_cached=Ntok` and structured `result.system_prompt_cache_hit_tokens`: the number restored specifically from a retained system-prefix entry for the initial prefill, not a sum across recovery attempts. A longer conversation-prefix hit can win instead and is reported in the existing `cache` count and `reuse_source`. These are cache observations, not a claim that every system prompt fits or that every restore improves latency.

### Response object and usage

A terminal wire response has `object: "response"`, one of `completed`, `incomplete`, or `cancelled` in `status`, and a typed `output` array. NInfer may emit:

- a `reasoning` Item containing raw `reasoning_text` and an empty summary;
- an assistant `message` containing an `output_text` part;
- one or more `function_call` Items.

When the request enables the logprob report, the `output_text` part carries `logprobs`: one `{token, logprob, bytes, top_logprobs}` object per output-text token, with the semantics of the Chat Completions [token log probabilities](#token-log-probabilities). The response echoes the request's `top_logprobs`.

Ordinary model/string stops produce `completed`. Output-token or context-capacity exhaustion produces `incomplete` with `incomplete_details.reason: "max_output_tokens"`. Errors accepted after an SSE response has started produce `response.failed`; validation and preparation errors remain normal HTTP error responses.

Usage is checkpoint-native:

```json
{
  "input_tokens": 42,
  "input_tokens_details": {"cached_tokens": 17},
  "output_tokens": 12,
  "output_tokens_details": {"reasoning_tokens": 5},
  "total_tokens": 54
}
```

`input_tokens` includes the chat template and expanded media tokens. `cached_tokens` is the exact resident prompt prefix reused by Engine. `output_tokens` is the count of accepted generated token IDs, including a withheld stop token when applicable. `reasoning_tokens` is counted in the Qwen output decoder while accepted tokens are still in the reasoning channel; it is not estimated by re-tokenizing decoded text.

### Responses streaming

Set `stream:true` for semantic Server-Sent Events. Every frame uses both the SSE event name and a matching JSON `type`, and every JSON event has a monotonically increasing `sequence_number`:

```text
event: response.output_text.delta
data: {"type":"response.output_text.delta","sequence_number":7,...}

```

The normal lifecycle is:

1. `response.created`, then `response.in_progress`;
2. `response.output_item.added` and `response.content_part.added`;
3. zero or more `response.reasoning_text.delta` or `response.output_text.delta` events;
4. matching `*.done`, `response.content_part.done`, and `response.output_item.done` events;
5. exactly one `response.completed`, `response.incomplete`, `response.cancelled`, or `response.failed` terminal event.

With the logprob report enabled, each `response.output_text.delta` carries the `logprobs` of the tokens committed with it (a delta may carry tokens and an empty `delta` while their text is held), and `response.output_text.done` carries the complete list.

Function arguments use `response.function_call_arguments.delta` and `.done`; IDs and output/content indices stay stable, and concatenated deltas equal each completed terminal Item. Declared names publish when their function header commits. An unconstrained first string parameter streams escaped JSON fragments; annotations such as `description` do not delay it. Constrained values, unions, later parameters, and ambiguous closing-markup suffixes wait for complete-call validation. Literal XML-looking string data and leading/trailing whitespace are preserved. Ordinary answer text continues streaming; Responses SSE does not emit Chat's `[DONE]`.

Chat publishes incremental `delta.tool_calls`; Anthropic publishes `tool_use` blocks and `input_json_delta`. All fragments and envelope-completion events are provisional until the terminal result. Responses defers accepted `.done` events until then; interrupted or cancelled calls become incomplete without completed-arguments events and never enter executable results or history. Anthropic block stops close framing only; consumers must await the terminal message before execution. Output exhaustion retains `length`/`max_tokens`, even after an earlier call completes. Duplicate-tool recovery buffers tool events until the accepted attempt is selected. Tool markup never leaks into prose. Long reasoning alone is not a loop or a reason to truncate output.

### Local response state and resources

`store` defaults to `true`. Stored Responses are bounded by an LRU store. By default it is process-local. Set `--response-store-location DIR` to retain terminal objects, normalized input Items, thinking policy and the shared conversation DAG across restarts. The directory is exclusively owned by one server process. This is local protocol-history persistence, not numerical checkpoint storage or OpenAI's cloud retention service.

Persistent mutations and LRU access order are committed through an atomic, synced manifest; immutable context nodes are shared between branches. A successful stored terminal response or delete is durable before its HTTP success is published. Streaming deltas alone do not create a stored entry. A disconnect after the terminal object has been stored can leave a retrievable ID even if the client did not receive its final SSE event. Unpublished payloads and unreachable ancestors are reclaimed after a commit or on startup. I/O publication failure returns an error and disables store access until restart rather than serving divergent memory/disk state. A missing or malformed committed payload fails startup rather than silently dropping history.

The existing record and memory-accounting limits apply after restart; lowering them evicts least-recently used public IDs on startup. They are not exact filesystem-byte quotas: JSON serialization and transient old/new payloads need additional disk space. The directory contains conversation and media source data in plaintext; use a dedicated location with appropriate local permissions. Owned media bytes are retained; external media URLs/paths are reacquired when a continuation is prepared, as with the process-local store.

`previous_response_id` reconstructs the complete stored input/output Item history before the new input. The current `instructions` value is placed first but is not saved into the continuation context, matching the Responses rule that previous top-level instructions do not carry forward. Function definitions are request configuration rather than conversation Items and must be sent again on tool-result turns. The reconstructed prompt follows the ordinary Engine path, so resident prefix reuse applies naturally. After restart the model can recompute this history or use an independently valid existing prefix cache. This store does not serialize GPU KV, GDN or DFlash state and makes no fast numerical-resume promise.

A stored Response also retains its resolved `preserve_thinking` value. A child which omits the field inherits the parent value. An explicit different value creates a new semantic branch; prompt rendering and identity still determine reuse. Changing the boolean alone never invalidates an exact current frontier or a complete matching rewrite checkpoint.

Resource behavior:

| Endpoint | Contract |
|---|---|
| `GET /v1/responses/{id}` | returns the stored terminal object, or 404 `response_not_found` |
| `DELETE /v1/responses/{id}` | removes public retrieval and returns `response.deleted`; descendant contexts already retained by other Responses remain usable |
| `GET /v1/responses/{id}/input_items` | returns normalized Items supplied to that request; supports `after`, `limit` `1..100` (default `20`), and `order` `asc|desc` (default `desc`) |
| `POST /v1/responses/{id}/cancel` | explicitly fails because background execution is unsupported |
| `POST /v1/responses/compact` | explicitly fails with `compaction_not_supported` |

`store:false` Responses cannot be retrieved or used as `previous_response_id`. LRU eviction and explicit deletion also make an ID unavailable. A single Response larger than the configured store capacity fails with `response_store_capacity_exceeded` rather than silently pretending it was stored.

### Candidate continuation scoring

`POST /v1/score` is a local, non-streaming scoring API:

```json
{"model":"qwen3.8-27b","messages":[{"role":"user","content":"Name a primary color."}],"candidates":["red","blue"]}
```

It accepts `model`, `messages`, 1–16 nonempty string `candidates`, and the existing `enable_thinking`, `preserve_thinking`, `reasoning_effort` and `chat_template_kwargs` controls. Context may include text and tool history; media, active tool declarations, sampling controls, output formats and generation options are rejected. The model field is informational, as on other endpoints. Each candidate is appended as the final assistant message to the same history; candidates never see one another.

Scores use the qualified teacher-forced prefill path and exactly the final-assistant boundary used by `ninfer-ppl --score-last-message`. The scored suffix includes template-generated reasoning framing, if any, and the assistant turn closure. Thus these are probabilities of **rendered assistant continuations**, not just bare strings. The response has `object:"candidate_scores"`, `scoring:"rendered_assistant"`, `schedule:"prefill"`, and a `data` array in input order. Each entry contains `index`, `text`, `prompt_tokens`, `scored_tokens`, `sum_nll`, `mean_nll`, `log_probability` (negative sum NLL), and `score_seconds`. `mean_nll` divides by `scored_tokens`; both quantities are exposed because candidates of different lengths can rank differently by sum and mean. No calibrated confidence or closed-set normalization is claimed. `usage` sums prompt and scored tokens across entries. Nonfinite or empty scores fail the entire request instead of producing a partial ranking.

Scoring requires an idle Engine and returns HTTP429 when it is busy. One execution reservation covers all candidates; normal generation cannot interleave with scoring resets. Requests arriving afterward may queue normally. Each prepared sequence must fit `--max-context`, and the aggregate token count must not exceed four times that limit. A client disconnect is checked between candidates; it does not interrupt a running prefill chunk. This API reuses existing scoring arithmetic, not computed shared-prefix state, and does not claim a branch-scoring speedup.

### Responses input token count

`POST /v1/responses/input_tokens` accepts exactly `model` and `input`, performs the same typed Item, template, and media expansion, and does not run generation:

```bash
curl http://127.0.0.1:8080/v1/responses/input_tokens \
  -H 'Content-Type: application/json' \
  -d '{"model":"qwen3.8-27b","input":"Count this prompt."}'
```

```json
{"object":"response.input_tokens","input_tokens":11}
```

Unsupported Create fields include Conversations, prompt templates, context management, hosted moderation, prompt-cache controls, safety/user identifiers, any `include` other than `message.output_text.logprobs`, background execution, compaction, files/audio, and OpenAI-hosted/MCP/custom tools. These are compatibility boundaries, not silently accepted placeholders.

## Constrained output

Chat Completions accepts the standard OpenAI `response_format`:

```json
{
  "response_format": {
    "type": "json_schema",
    "json_schema": {
      "name": "result",
      "strict": true,
      "schema": {
        "type": "object",
        "properties": {"answer": {"type": "string"}},
        "required": ["answer"],
        "additionalProperties": false
      }
    }
  }
}
```

Use `{"type":"json_object"}` for arbitrary JSON objects, or `{"type":"text"}` for unconstrained content. Responses accepts the same modes through `text.format`; its schema format is flattened: `{"type":"json_schema","name":"result","strict":true,"schema":{...}}`. The terminal Response echoes this format. Schema names contain 1–64 letters, digits, underscores, or hyphens. `strict` accepts boolean or null; every admitted schema assertion is enforced regardless of this flag.

The shared JSON triggers match the official [llama.cpp grammar guide](https://github.com/ggml-org/llama.cpp/blob/master/grammars/README.md), [vLLM structured-output contract](https://docs.vllm.ai/en/latest/features/structured_outputs/), and [SGLang structured-output contract](https://docs.sglang.io/docs/advanced_features/structured_outputs). Their raw grammar fields differ: llama.cpp uses `grammar` with GBNF, vLLM uses `structured_outputs.grammar`, and SGLang uses `ebnf`. NInfer selects the vLLM EBNF interface.

Chat Completions accepts exactly one vLLM-style `structured_outputs` field: `{"grammar":"root ::= \"yes\" | \"no\""}`, `{"regex":"[A-Z]+"}`, or `{"choice":["yes","no"]}`. Grammar uses XGrammar EBNF with start rule `root`; regex matches the complete answer; choices are literal strings and may include the empty string. This extension cannot accompany a constrained `response_format`. Responses `text.format` supports the JSON modes above. llama.cpp GBNF and SGLang `ebnf` are separate interfaces.

Constraints apply to content after the reasoning close; thinking remains free text. Include instructions for the desired response in the prompt: the schema is not injected into messages. Streaming uses the usual content deltas. Caller stop strings/tokens, cancellation, and token or context limits can leave incomplete constrained output; inspect the finish reason and validate truncated output before consuming it. Tool declarations and output constraints cannot be combined.

The JSON Schema subset admits primitive/container types and type unions; `properties`/`required`/`additionalProperties`; `items`/`prefixItems`; property/item/string count bounds; numeric bounds; `pattern`; `enum`/`const`/`anyOf`; and local JSON-pointer `$ref`/`$defs`/`definitions`. Type-specific assertions without `type` constrain only their relevant instance types. Counts must be nonnegative 32-bit integer literals. String lengths count decoded Unicode code points; patterns use search semantics unless anchored and may combine with length bounds.

Assertion siblings are conjoined. Supported `allOf` intersections combine types, numeric/count bounds, string patterns/lengths, object constraints, and homogeneous array items; finite values are filtered against the complete conjunction. Admitted `oneOf` forms include direct finite alternatives (overlaps retain only values accepted by exactly one branch), disjoint supported types such as string versus null, and object alternatives separated by a shared required property with disjoint finite values. Numeric interval separation alone does not qualify as a disjointness proof. Unsupported alternatives fail admission.

`prefixItems` is positional: shorter arrays remain valid subject to `minItems`; `false` forbids that and later positions, and `maxItems` may shorten the prefix. Tuple intersections remain unsupported. Local recursive value references are supported; external/anchor references, nested `$id` scopes, and nonproductive recursion are not. The default dialect is 2020-12; explicit 2020-12 and draft-07 identifiers are accepted within this subset. Draft-07 `$ref` assertion siblings and `prefixItems` are rejected rather than given another dialect's meaning. Unsupported answer assertions such as `uniqueItems`, `multipleOf`, `patternProperties`, and conditionals fail HTTP 400 `invalid_output_constraint`; answer schemas receive no tool relaxation.

JSON emission is compact. Answer properties follow schema order; tool parameters preserve required/unique named keys in any admitted order. Additional JSON keys alongside named properties use printable ASCII without quotes, backslashes, or escapes to prevent named-key aliases. A `minProperties` above one with arbitrary additional keys requires that many distinct required names; repeated keys cannot establish the count. Omitted `additionalProperties` and `items` remain permissive.

Bounded numbers use at most six fractional digits with inward-rounded bounds; intervals with no represented value fail admission. Integer bounds must be whole signed-64-bit values, including whole floating spellings such as `1.0`. Number bounds must survive the compiler's binary64 conversion; exact integer bounds that would round are rejected. Broader binary64 bounded-number emission, fractional integer bounds, and tuple intersections remain deferred. Numeric assertions and finite values must preserve their decimal value through JSON parsing/forwarding; inexact sources fail admission. Annotations and ordinary sampling fields are excluded.

Admission diagnostics retain source pointers through references/composition: OpenAI answer-schema `param` starts at `/response_format/json_schema/schema` for Chat or `/text/format/schema` for Responses. Tool errors use `invalid_tool_schema`; answer constraints use `invalid_output_constraint`. Exact integer literals remain available within the applicable literal and bound ranges.

The resident frontend shares a bounded compiled-grammar/tokenizer cache. Each request owns its matcher, and the existing packed GPU token mask constrains target sampling, including MTP and DFlash verification. Speculative matcher forks only advance on committed output, and mask construction overlaps target GPU execution. Unconstrained requests retain the mask-free route.

For active constrained Chat requests, including final streaming usage, `usage.prompt_tokens_details.ninfer.constraint` reports `complete`, `terminated`, `branch` (`none`, `reasoning`, `answer`, `tool_call`), `compile_cache` (`none`, `hit`, `built`, `waited`), `compile_seconds`, `compile_wait_seconds`, `mask_calls`, `mask_nodes`, `mask_seconds`, and `mask_upload_bytes`. Wait is included in compile elapsed time; mask time includes host forks, nodes count computed reachable states, and bytes count enqueued payloads. Recovery work is counted once; terminal state/cache describe the final attempt. Unconstrained requests omit the object. Completion and termination differ, so a caller stop or limit can leave an incomplete value.

The public Engine uses `PromptOptions.output_constraint`. Tool history remains valid input; literal tool markup in constrained content remains data.

The real-engine smoke client is `python3.11 -m tools.smoke.serve_features --model MODEL --base-url BASE --concurrency 1` (use concurrency4 for a C4 server). Run it against each qualified ordinary/MTP/DFlash configuration under the shared GPU lease. `--save-response FILE` and, after restarting with the same store directory, `--restore-response FILE` exercise durable retrieval and descendant continuation. The client does not start or stop servers.

## Anthropic Messages

```bash
curl http://127.0.0.1:8080/v1/messages \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "qwen3.8-27b",
    "max_tokens": 128,
    "messages": [
      {"role": "user", "content": "Explain prefix reuse in one sentence."}
    ]
  }'
```

The endpoint supports top-level system text, ordered mid-conversation system messages, user/assistant history, text and image blocks, thinking blocks, tool-use history, tool results, client-defined tools, non-streaming responses, and Anthropic SSE events. Mid-conversation system messages remain at their `messages` array position and are not merged into the top-level system instruction. A system section must follow a user/tool-result message and be final or immediately precede an assistant message; it cannot interrupt a tool-use/tool-result pair. Consecutive system messages remain separate ordered turns.

`thinking.type: "disabled"` disables thinking; other supported values enable it. Returned thinking blocks carry the message id as a non-empty opaque `signature` (a `signature_delta` just before the block's `content_block_stop` when streaming); echoed thinking text is lowered directly into history and the signature is not validated. The independent top-level `preserve_thinking` boolean controls closed-turn history and otherwise uses the server default.

Anthropic `output_config.effort` accepts the protocol values `low`, `medium`, `high`, `xhigh`, and `max`. The value is then checked against the loaded chat template in the same way as the OpenAI endpoints; the registered effort-capable template exposes `low`, `medium`, and `xhigh`. Combining an effort with `thinking.type: "disabled"` is rejected as contradictory.

Anthropic's `model` field is treated as a response label and does not select the loaded artifact.

`POST /v1/messages/count_tokens` uses the artifact's tokenizer, chat template, and media expansion without running GPU generation:

```bash
curl http://127.0.0.1:8080/v1/messages/count_tokens \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "qwen3.8-27b",
    "messages": [{"role": "user", "content": "Count this prompt."}]
  }'
```

## Authentication and CORS

Pass `--api-key VALUE` to require the same value as an OpenAI bearer token or Anthropic `x-api-key` header. `GET /health`, `GET /metrics`, `GET /metrics.json`, and CORS preflight requests remain unauthenticated. Binding a non-loopback `--host` therefore exposes occupancy, queue depth, and token totals to anyone who can reach the port.

```bash
curl http://127.0.0.1:8080/v1/models \
  -H 'Authorization: Bearer local-secret'
```

`--cors` adds permissive browser CORS headers. It is disabled by default.

## Server options

| Option | Meaning | Default |
|---|---|---:|
| `--host H` | listen address | `127.0.0.1` |
| `--port N` | listen port | `8080` |
| `--api-key KEY` | required bearer or `x-api-key` value | unset |
| `--model-id ID` | override the public OpenAI model alias | artifact `identity.model_id` |
| `--max-context N` | logical context ceiling of each sequence | `8192` |
| `--kv-capacity N\|auto` | explicit shared Main Text KV capacity, or maximize it from remaining GPU memory | `auto` |
| `--kv-capacity-headroom MiB` | device memory `--kv-capacity auto` leaves free; requires `auto` | `64` |
| `--kv-ram-capacity off\|N` | pinned host capacity in MiB for completed prefixes and live suspension; enables output windows up to 8192 tokens | `off` |
| `--kv-disk-capacity off\|N` | SSD KV prefix-cache unique-object capacity in MiB; `off` disables the tier | `off` |
| `--kv-disk-location PATH` | directory for the SSD page store; required iff `--kv-disk-capacity` is enabled | unset |
| `--kv-disk-compress off\|zstd` | zstd-1 on new GDN/hidden/cyclic writes; KV pages stay uncompressed | `off` |
| `--max-concurrency N` | maximum resident requests; logical lifetime count remains bounded by concurrency plus pending capacity; valid range `1..8` | `1` |
| `--max-pending-requests N` | additional requests allowed to wait for admission | `16` |
| `--pending-timeout-ms N` | maximum preparation-plus-admission wait | `300000` |
| `--prefill-chunk N` | text-prefill chunk | `2048` |
| `--mixed-forward auto\|N` | forward width (columns) of each prefill step while other requests are decoding; `0` runs each prompt's whole prefill first. With DFlash a text prompt's slice rides inside a decode round: one forward of `N` columns holds the verify columns of the decoding requests and the prompt tokens fill the rest, at least `N - (C-1)*W`. Other prompts run a separate step of `N` tokens. `N` must be a multiple of 256 (prefill loses 3–5% between multiples), at most `--prefill-chunk` and `--max-context`, and wider than `(C-1)*W`. `auto` is 1024 (bounded by those, rounded down to a multiple of 256) under DFlash with C > 1 and `0` otherwise; startup logs the resolved value. Smaller widths shorten each decode pause (about 0.18 s at 512 against 0.30 s at 1024) at some prefill throughput. See [prefill/decode frontier](performance.md#mixed-prefilldecode-frontier-2026-09-29) | `auto` |
| `--mixed-forward-rounds N` | decode rounds per prefill slice (`1` = every decode round carries one slice; larger values leave plain decode rounds between slices) | `1` |
| `--log-stats-interval-ms N` | aggregate throughput report interval; `0` disables it | `5000` |
| `--device N` | HIP device index | `0` |
| `--max-request-mib N` | body-size limit before JSON parsing | `384` |
| `--request-log-jsonl FILE` | append full-precision server/request records | disabled |
| `--response-store-location DIR` | optional directory for restart-persistent Responses history | unset (process-local) |
| `--response-store-max-records N` | maximum locally retained Responses objects | `1024` |
| `--response-store-max-mib N` | total local Response envelope/Item/context budget | `256` |
| `--spec none\|mtp\|dflash` | speculative backend | `dflash` |
| `--draft-tokens N` | MTP `1..5`, DFlash2 `1..7` | `7` with DFlash; required with MTP |
| `--adaptive-draft` | pick live draft K in `{3..N}` (DFlash `--draft-tokens N>=5`; MTP `{3,4,5}`) by locking `E[Y]/T(k,C,L)` (nested `r_i`; least-squares T; at most one probe of an unmeasured k; 1 ms switch cost). `--draft-tokens 4` stays `{4}` | on with DFlash; off with MTP |
| `--no-adaptive-draft` | use the fixed `--draft-tokens` length instead of adapting | off |
| `--dflash-verify-width N` | DFlash2 chain verify width `W=k+1`, `2..8` | auto |
| `--dflash-p-less-draft-temperature T` | Pins the DFlash2 draft temperature `0..2` for p-less requests: drafts are drawn from the 16-candidate path-select softmax at `T` and verified against that proposal, so output stays exactly the p-less target distribution. `0` drafts greedily. Unset, the engine calibrates it online per p-less `--temperature` and draft length: every chain round scores eight candidate temperatures against the verified target distribution and the next round uses the best ([model §8](maintainer/qwen3.8-27b-model.md#8-dflash2-block-diffusion-draft-model)) | calibrated |
| `--lm-head-draft` / `--no-lm-head-draft` | enable or disable the optimized proposal head | on with DFlash |
| `--default-reasoning-effort template\|low\|medium\|xhigh` | effort when omitted by a thinking request; explicit request effort takes precedence | `template` (Docker: `low`) |
| `--default-max-tokens N` | output limit when omitted by a request | `8192` |
| `--vision` | enable media input and load Vision GPU allocations | off |
| `--no-device-graph` | disable Device Graph decode | graphs on |
| `--no-prefix-reuse` | disable compatible-prefix caching | prefix reuse on |
| `--context-checkpoints off\|a,b,c` | disable the automatic prefill ladder, or replace the default marks. Custom lists require `--spec mtp` or `--spec dflash`. Marks at or above `--max-context` stay unused. Advertised freeze `F` is the committed chunk end at or past the mark. | default ladder |
| `--no-thinking` | disable thinking by default | thinking on |
| `--preserve-thinking` | preserve closed-turn assistant reasoning by default | off |
| `--system-prepend TEXT` | prepend TEXT to the leading system/developer instruction on every request; inserts a system turn if none exists | off |
| `--cors` | permissive browser CORS headers | off |
| `--temperature F` | process-level temperature override | unset |
| `--top-p F` | process-level top-p override | unset |
| `--top-k N` | process-level top-k override | unset |
| `--min-p F` | process-level min-p override | unset |
| `--presence-penalty F` | process-level presence-penalty override | unset |
| `--frequency-penalty F` | process-level frequency-penalty override | unset |
| `--seed N` | fixed seed when a request omits one | fresh random seed per request |
| `--greedy` | force exact argmax for all requests | off |
| `--no-p-less-sampling` | opt out of p-less and use top-p/top-k/min-p/penalties | p-less on |

The configured maximum `--draft-tokens` and adaptive policy are fixed for the server lifetime. Startup captures the required graph set; adaptive DFlash selects live K in `{3..N}` for maximum N>=5. `--no-adaptive-draft` uses the fixed configured length.

Engine selects sampling defaults from the loaded model and the request's resolved thinking mode. With the standard sampler (`--no-p-less-sampling`), Qwen3.8-27B uses temperature `2.0` and presence penalty `0` in both modes (`2.0/0.95/20/0/0` thinking, `2.0/0.80/20/0/0` non-thinking). Frequency penalty is `0` for all registered presets. Process flags override registered values, request fields override process flags, and `--greedy` finally forces temperature `0`. P-less is enabled by default with temperature `1.5` unless overridden; it keeps temperature and seed and ignores top-p, top-k, min-p, and presence/frequency penalties from both process flags and request bodies. Startup logs a one-time warning. Ignored request fields must still satisfy their normal input ranges before sampler resolution. `--no-p-less-sampling` opts into the registered production sampler. Combined with `--greedy`, p-less remains exact argmax. P-less membership is `p_v ≥ max(L·exp(-2ε/T), 1/M)` with `ε = 1/16` and `M = 1024`; L is the unperturbed collision probability, and an empty set falls back to the eligible mode. Under MTP or DFlash2, p-less applies at every hop (block verification over the chain with the recorded draft `q`; DFlash2 drafts are sampled at the calibrated or pinned `--dflash-p-less-draft-temperature`) and to the bonus. A thinking-cycle exclusion affects only the next token, not later hops in the same speculative round. There is no OpenAI or Anthropic schema field for this mode.

Run `./build-r9700/apps/ninfer-serve --help` for all options and defaults.

## Metrics

`GET /metrics` is Prometheus text exposition 0.0.4 (`text/plain; version=0.0.4`). `GET /metrics.json` renders the same snapshot as JSON: counter totals and histogram `count` / `sum`, not rates. Both read the public Engine `RuntimeStats` once per scrape plus the serve-owned counters; a scrape never queries the HIP device. Device-memory families are the startup measurements.

```bash
curl -s http://127.0.0.1:8080/metrics | head
curl -s http://127.0.0.1:8080/metrics.json | jq .scheduler,.gpu_kv,.kv_ram,.speculative,.recovery,.phases
```

```yaml
scrape_configs:
  - job_name: ninfer
    scrape_interval: 2s
    static_configs: [{ targets: ["127.0.0.1:8080"] }]
```

| Family group | Signals |
|---|---|
| identity | `ninfer_engine_info` (target, weights, model alias, `spec`, `kv_cache_format="fp8-k-int4-v"`, compile-bound `kv_value_group` and `xattention` profile, Device Graph, prefix reuse, Vision, auth/CORS) and `ninfer_build_info` (HIP compile/runtime/driver version, `gpu_arch`, `gpu_name`) |
| startup | context, prefill chunk, pending timeout, default output budget, draft/DFlash verify width, load seconds, arena capacities, `ninfer_device_graph_{allowance,observed}_bytes`, and `ninfer_device_memory_bytes{kind}` (`total`, `available_after_weights`, `runtime_reservation`, `kv_payload`, `available_after_startup`) as measured once at startup |
| scheduler | `ninfer_scheduler_{running,prefilling,decode_ready,waiting}_requests`, configured concurrency and FIFO depth, and `ninfer_engine_{computed_prefill_tokens,committed_decode_tokens,decode_rounds,decode_row_rounds}_total` |
| device KV | `ninfer_gpu_kv_pages{pool,state}` for `pool="main"` (FP8-K/INT4-V Text) and `pool="spec"` (the MTP paged pool; 0 without MTP, including DFlash2 whose private BF16 state is not paged) with `state` `capacity` / `entitled` / `mapped` / `free`, and `ninfer_gpu_kv_capacity_tokens{pool}` (64 tokens per page group) |
| live KV execution | `ninfer_kv_execution_window_tokens` (8192 with RAM on, zero with RAM off), `ninfer_kv_growth_blocked_requests`, `ninfer_kv_suspended_requests`, and absolute counters `ninfer_kv_window_extensions_total`, `ninfer_kv_live_suspensions_total`, `ninfer_kv_live_resumes_total`; JSON `kv_execution` contains `window_tokens`, `growth_blocked_requests`, `suspended_requests`, `window_extensions`, `suspensions`, `resumes` |
| host KV tiers | `ninfer_kv_{ram,disk}_{capacity_bytes,used_bytes,entries}`, `_{captures,restores,evictions,drops}_total`, `_{save,load}_seconds_total`, `ninfer_kv_disk_h2d_seconds_total`, and `ninfer_kv_cache_fallbacks_total` (failed host restores requeued for cold prefill) |
| prefix reuse | `ninfer_prefix_reuse_requests_total{path,source}`, `ninfer_prefix_cache_{hit_tokens_total{source},query_tokens_total}`, and context-checkpoint restored/captured tokens and capture requests |
| speculation | MTP/DFlash rounds, draft/accepted tokens, fallback steps, accepted tokens by draft position 0-14, rounds by live draft width 1-15, and the `ninfer_speculative_live_k` histogram |
| recovery | `ninfer_recovery_events_total{kind,cause}`, true `ninfer_recovery_cycle_exclusions_total`, `ninfer_recovery_framing_repairs_total`, `ninfer_recovery_discarded_content_tokens_total`, discarded reasoning tokens and tool calls, and the `ninfer_recovery_attempts` histogram |
| constraints | `ninfer_constraint_{requests,complete,terminated}_total`, `ninfer_constraint_branch_total{branch}`, `ninfer_constraint_compile_cache_total{result}`, `ninfer_constraint_{compile_seconds,compile_wait_seconds,mask_calls,mask_nodes,mask_seconds,mask_upload_bytes}_total`; JSON: `generation.constraint` |
| generation | `ninfer_generation_requests_total{protocol,result,stream,thinking,tools}`, TTFT / end-to-end / inter-token latency histograms, phase and per-request KV copy histograms, prompt/completion/reasoning/computed-prefill token histograms, prefill and decode tok/s histograms, finish reasons, tool calls, ignored tool markup, media requests, and token-count requests |
| HTTP | `ninfer_http_requests_total{method,protocol,route,status}`, `ninfer_http_request_duration_seconds{protocol,route}`, `ninfer_http_in_flight_requests`, `ninfer_api_errors_total{code}` over a closed code set, and Responses store records/bytes and caps |

Constraint totals combine terminal outcomes from all three protocols, including cancellation work. Rejected preparations and errors without terminal outcomes add no fabricated work. Scrapes read accumulated host values without building masks or synchronizing the GPU. Per-request constraint usage is Chat-only; Responses and Anthropic keep their protocol usage shapes.

The two `available_after_*` memory values are startup snapshots, not live free-memory counters. They use the smaller of HIP availability and graphics-inclusive physical free VRAM from the matching PCI device's amdgpu counters. The after-weights snapshot also follows RAM-cache registration.

| Question | Signal |
|---|---|
| In flight? | `ninfer_http_in_flight_requests` vs `ninfer_scheduler_running_requests` vs `ninfer_scheduler_waiting_requests` |
| Concurrent decode? | `rate(ninfer_engine_decode_row_rounds_total)/rate(ninfer_engine_decode_rounds_total)` > 1 |
| Device KV tight? | `ninfer_gpu_kv_pages{pool="main",state="entitled"} / ninfer_gpu_kv_pages{pool="main",state="capacity"}` |
| RAM cache working? | `ninfer_kv_ram_used_bytes` > 0, restores rising, `ninfer_prefix_reuse_requests_total{source="host_ram"}` |
| Disk cache working? | `ninfer_kv_disk_*` and `source="host_disk"`; `ninfer_kv_cache_fallbacks_total` staying flat |
| Prefix hit rate | `rate(ninfer_prefix_cache_hit_tokens_total)/rate(ninfer_prefix_cache_query_tokens_total)` |
| MTP/DFlash healthy? | `rate(ninfer_speculative_accepted_tokens_total)/rate(ninfer_speculative_draft_tokens_total)` |
| Why is TTFT high? | phase p95 of `queue`, `copy_hold`, `prefill`, `media_fetch`; one JSONL `request_done` for the joint view |
| Disk restore on the critical path? | `copy_hold` p95 and `ninfer_generation_kv_copy_seconds{tier="disk",op="h2d"}` |
| Recovery burning time? | `ninfer_generation_phase_seconds{phase="recovery"}` vs `{phase="decode"}` |

`ninfer_generation_phase_seconds{phase}` has the closed phases `prepare_cpu`, `media_wait`, `media_fetch`, `queue`, `copy_hold`, `vision`, `prefill`, `decode`, `recovery`, and `http_tail` (definitions under `request_done.timings_seconds` below); only positive phases are observed. Phase samples need not sum to end-to-end: recovery re-prefill, HIP copy time, and copy-hold wall overlap. `ninfer_generation_inter_token_latency_seconds` is decode seconds per decode-evaluated output token (TPOT); `ninfer_generation_output_tokens_per_second` is its inverse and matches NInfer tok/s reports. Latency, token, finish-reason, prefix, and speculative families observe successful generations only; `ninfer_generation_requests_total` also counts `rejected`, `error`, and `cancelled` attempts. `ninfer_recovery_events_total{kind="cycle_exclusion"}` counts published events, which are powers-of-two samples; the true exclusion count is `ninfer_recovery_cycle_exclusions_total`. An `exhausted` event's `cause` is its decision: `retry_budget`, `output_budget` (the repaired request cannot keep its admitted output budget or pages), `prologue`, or `lane_rebuild`, or `repeated_content` (answer text cannot be retried after publication), or `repeated_reasoning` (no supported unpublished retry context); the JSONL `recovery` event keeps the raw detail for the other cases. `ninfer_generation_media_requests_total` counts every generation attempt whose prompt had media, whatever its result.

Every label value is from a closed set, so series are bounded: unknown API error codes collapse to `code="other"`, unnamed ones to `code="unnamed"`, and unknown routes, methods, or statuses to `"other"`. HTTP series appear once their route has served a request. Histogram buckets are sized for this workload: wall clocks span 1 ms to 20 minutes, inter-token latency 2 ms to 250 ms, token counts 32 to 262144, decode rate 5 to 1000 tok/s, and prefill rate 250 to 30000 tok/s.

```promql
rate(ninfer_engine_committed_decode_tokens_total[1m])
histogram_quantile(0.95, rate(ninfer_generation_ttft_seconds_bucket[5m]))
histogram_quantile(0.95, rate(ninfer_generation_inter_token_latency_seconds_bucket[5m]))
histogram_quantile(0.95, rate(ninfer_generation_phase_seconds_bucket{phase="queue"}[5m]))
histogram_quantile(0.95, rate(ninfer_generation_phase_seconds_bucket{phase="copy_hold"}[5m]))
rate(ninfer_prefix_cache_hit_tokens_total[5m])
  / rate(ninfer_prefix_cache_query_tokens_total[5m])
rate(ninfer_speculative_accepted_tokens_total[5m])
  / rate(ninfer_speculative_draft_tokens_total[5m])
```

## Structured request log

`--request-log-jsonl FILE` enables the machine-readable measurement log. The server opens `FILE` in append mode and flushes every event, so successive model or MTP blocks may share one campaign file. The parent directory must already exist. Failure to open the file aborts startup; the log path is also rejected if it resolves to the model artifact.

```bash
./build-r9700/apps/ninfer-serve /path/to/qwen3.8-27b-r9700-lq4.ninfer \
  --request-log-jsonl profiles/bench/run/server.requests.jsonl
```

Every line is one `ninfer_serve_request_log` schema-v29 JSON object. All events carry `timestamp_unix_ms` and a process-unique `server_instance_id`; request IDs are monotonic only within that server instance.

| Event | Contents |
|---|---|
| `server_start` | target/weights identity and artifact (including `mapped_host_bytes`, the weights held in pinned host memory), resolved Engine, compile-bound Text-prefill/XAttention identity, registered thinking/non-thinking sampler defaults plus process overrides, thinking-history defaults, weights/sequence/workspace/request-transient arenas, KV sizing ledger, pinned-host KV RAM capacity/occupancy, Device Graph observed/allowance bytes, HIP/GPU environment, and redacted argv |
| `request_prepare` | parsed request ID/protocol, body size, message/tool counts, and handler parse/context-assembly time; emitted before synchronous preparation begins |
| `request_start` | completed synchronous `timings_seconds.synchronous_prepare`, protocol, resolved sampler and seed (including `p_less`), thinking modes, explicit reasoning effort and source field, Responses semantic-change flag, output budget, stream/message/tool shape |
| `request_rejected` | parsed request shape, media-item count, `phase: "prepare"`, and the exact HTTP status/type/code/parameter/message for a synchronous preparation rejection |
| `request_done` | finish reason, prompt/completion/cache/computed-prefill tokens, prefix reuse path, `reuse_source` (`none` / `vram_resident` / `host_ram` / `host_disk`), `context_checkpoint` (`restored_tokens` / `captured_tokens`), this request's unrounded phase clocks, `recovery` totals (including the true `cycle_exclusions`), complete speculative-decoding counters, `tool_call_count`, and `ignored_qwen_tool_call_names` (empty unless a tools-off completion contained parseable Qwen `<tool_call>` markup). Process KV occupancy and lifetime tier counters are not on this event |
| `recovery` | one line per published Engine recovery event: `request.id`, `kind`, raw `cause` (including the exhausted detail), attempts, cycle exclusions, discarded tool calls, reasoning tokens and content tokens, framing repairs, generated tokens, remaining tokens, and the complete resolved `sampling` profile on `retry_started` |
| `request_error` | the resolved request configuration and generation error message |
| `throughput` | interval token deltas and rates, scheduler occupancy including device KV pages (`gpu_kv_{main,spec}_{capacity,entitled,mapped,free}_pages`) and absolute `kv_cache_fallbacks`, interval `timings_seconds.kv_ram_save` / `kv_ram_load` / `kv_disk_save` / `kv_disk_load` / `kv_disk_h2d`, and decode-round batch statistics |

`server_start.engine.xattention_qualification` is always present. It is `false` in the ordinary product build. The OFF-by-default qualification build reports `true` plus the exact `xattention_profile`, `xattention_find_block`, `xattention_stride`, and `xattention_tau_permille` compiled into that executable; these are evidence fields, not runtime selectors. `server_start.engine.kv_value_group` likewise records the compile-bound G16/G32 Text and MTP value-cache group; it is not a runtime cache selector.

`request_done.timings_seconds` contains `prepare`, `prepare_cpu`, `media_wait`, `media_fetch`, `queued`, `copy_hold`, `ttft`, `vision`, `prefill`, `decode`, `total`, `recovery_prepare`, `recovery_prefill`, `kv_ram_save`, `kv_ram_load`, `kv_disk_save`, `kv_disk_load`, `kv_disk_h2d`, and `http_tail` as full-precision JSON numbers. They need not sum to `total`: recovery re-prefill, HIP copy time, and copy-hold wall overlap. `prepare_cpu` is `prepare` minus the media permit wait (`media_wait`) and media acquisition (`media_fetch`). `queued` is pending-FIFO wall from Engine submit until admission, including a requeue after a failed host restore, and excludes copy-hold. `copy_hold` is wall blocked on victim spill or RAM/disk restore until prefill starts. `http_tail` is the HTTP handler clock minus engine end-to-end, so it includes JSON parsing and the SSE or JSON body built after the Engine result.

`kv_ram_save` / `kv_ram_load` are HIP event elapsed for that request's RAM-tier FIFO D2H capture and H2D unpack (Main KV, optional MTP KV or DFlash cyclic state, and GDN images in the same copy span). They are not admission wait. Live-lane context-checkpoint freeze D2H and a VRAM-resident restore that unpacks already-pinned lane GDN are not included. Throughput events repeat those two keys as interval sums.

Its `speculative` object contains `backend`, `draft_window`, `rounds`, `drafted_tokens`, `accepted_tokens`, `fallback_steps`, `accepted_per_position`, and `p_less_draft_temperature` (the last p-less chain round’s draft temperature, or zero when no such round ran). Rates can be derived downstream from raw token counts and seconds instead of rounded stderr strings. When a stop cuts a licensed speculative round short, accepted-token counters include only the committed drafts; rounds and drafted-token counters still include the work performed.

The JSONL file contains no generated response text and never records an API-key value; `argv` replaces that value with `<redacted>`. Started request records (also carried by completion/error events) include `reasoning_effort` (null when omitted) and `reasoning_effort_param` (the source request field, empty when omitted). Submitted console lines include these fields only when the request explicitly supplies effort, including when it matches the default or is `none` to disable thinking. The separate `effective_reasoning_effort` field records the actual selected effort (null with thinking off or a template without effort support); it includes inherited server/template defaults. The startup JSONL server record includes `default_reasoning_effort` (null for the template default). The existing stderr summaries remain available for operators but are rounded and are not the aggregation source. Console lines use local `[YYYY-MM-DD HH:MM:SS.mmm] [level]` timestamps. A tools-off completion whose answer contains parseable Qwen `<tool_call>` markup also writes a `warning` line naming the ignored functions and the request's `tools` / `tool_choice` / `tool_history` shape; the same names are in `request_done.result.ignored_qwen_tool_call_names`. OpenAI Responses, OpenAI Chat, and Anthropic generation requests receive a request ID when they enter synchronous preparation. A `request_prepare` event precedes synchronous preparation. Successful preparation produces `request_start`; a preparation failure produces `request_rejected` without a matching start. Later generation failures produce `request_error`. Schema/model validation rejections before preparation and token-count-only calls are not measurement requests and do not receive request IDs.

By default the server also reports aggregate activity every five seconds. `prefill` counts prompt suffix tokens actually computed during the interval, excluding prefix-cache hits; `decode` counts tokens finally committed by decode rounds, excluding the first token produced by prefill. For MTP and DFlash this is the accepted committed output, not draft or rejected tokens. `avg_decode_batch` is decode row-rounds divided by decode rounds during the same interval. The `running`, `prefilling`, `decode_ready`, and `waiting` fields are the Engine scheduler snapshot at the end of the interval. Fully idle zero intervals are omitted. The JSONL `throughput` event keeps the raw token and round deltas as well as derived rates; downstream measurement should prefer those raw values.

Disk `work` identifies active spill preparation/session, idle defragmentation, and restore targets; `pending` counts queued disk jobs. An active spill session reports `spill_age` in seconds (zero while preparation has not installed a session). These are background cache activities, independent of scheduler request occupancy. Active disk work keeps the periodic report visible, including its transition to idle.

`spill_completed` is the sum of full lifetimes of spills completed during the interval, so it can exceed the interval and does not measure time blocking requests. The JSONL `timings_seconds.kv_disk_save` and cumulative metric `ninfer_kv_disk_save_seconds_total` have the same completed-spill lifetime semantics. JSONL scheduler fields and both metrics encodings expose `spill_active`, `defrag_active`, `storage_bytes`, `restore_active`, `pending_jobs`, and `spill_elapsed_seconds` (with a `kv_disk_` prefix in JSONL and `ninfer_kv_disk_` in Prometheus).

The preparation line is emitted after JSON/protocol parsing and, for Responses, context assembly; `parse` measures that handler work. It does not timestamp network arrival or body reception. The submitted line reports `prepare_sync`, the full `service.prepare` call including synchronous submission and tool-grammar compilation. JSONL records this as `timings_seconds.synchronous_prepare`; it is a separate wall clock from `request_done` phase accounting and must not be added to its phases. Disk snapshots use a nonblocking observer: if the disk index mutex is busy, the most recent snapshot is retained, so work flags and age may temporarily be stale. A gap with no preparation line can precede handler processing; these logs alone cannot attribute such a gap to a client or the network.

## Execution behavior

The server owns one resident Engine with a startup-fixed capacity of `1..8` active generation requests. At each decode boundary, every decode-ready request is compacted into one batch and processed by one model traversal and, when graphs are enabled, one exact-batch Device Graph replay. A request joins that batch only after its single-request prefill finishes; when it completes or is cancelled, the next boundary rebuilds the batch without an empty row.

`--max-pending-requests` bounds the requests waiting behind the active set. The total generation request lifetime capacity is `max_concurrency + max_pending_requests`, including requests still in CPU/media preparation and completed model results whose response has not yet been released. A full capacity returns HTTP 429 with code `server_overloaded`. The absolute `--pending-timeout-ms` deadline starts before preparation, covers media acquisition and Engine FIFO waiting, and returns HTTP 503 with code `request_queue_timeout` if admission does not occur in time. There is no admission ETA or unbounded overflow queue.

Input memory is bounded by the outstanding-request count and the per-request `--max-request-mib` limit. Media requests additionally share one preparation permit, so a waiting media request retains the same cancellation and timeout deadline. Model output is bounded by the same finite request count and each request's effective output-token limit; output callbacks and network serialization run outside the GPU executor and do not delay formation of the next batch.

`--max-context` and the resolved `--kv-capacity` are independent limits. The former is each sequence's logical ceiling; the latter sizes the shared Main Text KV pool used by all active requests and retained prefixes. Both are represented with 64-token pages internally, while a sequence can never cross the exact `--max-context` frontier. `--kv-capacity N` requests an explicit capacity; `--kv-capacity auto` chooses the largest legal capacity that fits the memory remaining after weights are loaded, leaving `--kv-capacity-headroom` MiB free (default 64; raise it when a desktop or another process shares the GPU). The token embedding is held in pinned host memory, not VRAM. Omitting `--kv-capacity` selects automatic sizing; `--max-context` remains the per-request ceiling. The shared pool is fixed at startup and is not divided evenly among request lanes.

`--kv-ram-capacity` is a separate pinned-host budget in MiB shared by completed prefix bundles and secured live continuation images; it does not change GPU pool sizing. Enabling it automatically uses output reservations of at most 8192 tokens with full prompt and backend headroom covered; RAM off retains full-lifetime GPU reservation. The client output/context limit is unchanged. One blocked resident waits while peers execute, unless a funded queued replacement can use its lane; two blocked lanes or no executable work triggers the smallest useful suspension plan. Capture/restore preserves committed state, output buffering, RNG and recovery, so the response may pause but never replays streamed content or reruns prefill. Suspended requests remain admitted and do not expire on a pre-admission queue timeout. Live images are not optional cache entries and never spill to disk. Window-end RAM credits for ordinary residents and a completion cohort preserve a feasible completion path. The cohort's target-produced lifetime envelopes jointly fit GPU capacity, including bounded native suffix growth only for enabled internal retry paths; those envelopes are future proofs rather than upfront allocations. Raw, disabled-recovery and public capped-Close requests acquire no speculative retry charge. Unused future window entitlement may be reclaimed before suspension while preserving current state and a useful next output window. Inadequate backing can still lower resident concurrency. Oldest suspended requests resume before fresh ingress, except an already selected replacement. A frozen pressure epoch allows at most `C-1` fresh bypass admissions, then drains until its oldest incumbent completes/cancels. A matched C8 DFlash raw-token completion workload measured 28.0% higher aggregate decode throughput and 21.9% shorter makespan with windowed reservations; see the windowed KV results in `docs/performance.md` for the workload, earlier fixed-horizon negative result and limits. This does not establish a universal or useful-action speedup. One long MTP chat with five context-checkpoint heads is about 6 GiB in that FIFO; `off` still keeps same-lane GDN rollback on the live pinned log, but other-lane restore after eviction needs the FIFO. `--kv-disk-capacity` is a third-tier SSD budget in actual pack-file bytes, including free holes and retired reader extents. It requires `--kv-ram-capacity > 0` and `--kv-disk-location`. Disk is inclusive of VRAM/RAM hits; equal reuse length prefers VRAM, then RAM, then disk. `--kv-disk-compress` is not part of the directory fingerprint and affects new GDN/hidden/cyclic writes only. When an admission's captures need RAM and no RAM entry is on disk yet, one is spilled before it is evicted: while other requests decode, the admission stays queued (`queued` phase, bounded by `--pending-timeout-ms`) until that spill commits instead of dropping the entry unsaved, so a disk-bandwidth-bound load trades TTFT for retained prefixes.

Runtime cache capacity or optional capture-allocation failures, including eviction-candidate bookkeeping, skip the capture and do not reject generation. Disk-assisted reclaim counts `optional_alloc` and retains its source when metadata allocation fails. Optional cache-lookup allocation failure leaves normal cold admission available.

If a disk cache read or optional RAM/disk restore metadata or HIP-event allocation fails before prefill, the Engine drains the partial restore, excludes that entry from reuse, and recomputes the prompt from an empty KV image. The same request returns through normal capacity admission; its original queue deadline does not expire this already-admitted recovery. This cache fallback does not produce `service_unavailable`.

Disk format v6 fingerprints canonical logical KV pages rather than the current GPU pool capacity, so one location can reopen across `--vision` on/off and automatic resident-capacity changes. Media content remains part of each entry identity. The fingerprint also binds the opened artifact's local file generation (device, inode, byte length, nanosecond modification/change times), not just its shared model/weights names. Replaced or modified artifacts require a fresh cache directory; even a byte-identical copy is conservatively a different file. Artifacts must remain immutable while an Engine is using them. Pre-v6 locations fail startup: select a new directory and retain the old one until its contents are no longer needed. KV dtype, speculative backend, and persistent-state incompatibilities also fail startup.

On stop, orderly shutdown copies active chats into the host cache, saves cache entries that are not yet on SSD, and finishes outstanding disk writes. The progress renderer prints `kv-disk` `copy active chats` / `save cache entries` / `finish disk writes` counts while shutdown runs.

Automatic sizing evaluates the complete target runtime layout for the chosen concurrency, fixed FP8-K/INT4-V growing-cache format, speculative backend, draft window, Vision setting, workspace, and Device Graph allowance. It uses a direct page-capacity calculation rather than allocation probing. Startup reports the policy, resolved capacity, runtime reservation, free memory after weights, automatic headroom, planned slack, actual free memory after complete startup, observed Graph memory, the pinned checkpoint-image slab (`ckpt-pin=`) and its context-checkpoint head count (`ckpt-heads=`), and pinned-host KV RAM (`kv-ram=`) and disk (`kv-disk=`) occupancy, sizes in MiB.

When `--kv-ram-capacity` is enabled, a post-warmup line reprints occupancy, periodic throughput lines print live host-resident `kv-ram=` used bytes plus `n=` / `restores=` / `evicts=` / `drops=` / `save=` / `load=`, plus device KV page entitlement (`gpu-kv=entitled/capacity` and `spec=` when a speculative pool exists) and non-zero `cache_fallbacks=`. Each `[req] done` line includes `reuse_source=` and this request's non-zero `kv_ram_save=` / `kv_ram_load=` / `kv_disk_save=` / `kv_disk_load=` / `kv_disk_h2d=` milliseconds; it does not print process occupancy. When `--kv-disk-capacity` is enabled, throughput lines also print `kv-disk=` occupancy, counters, and interval `h2d=`. `kv-ram=` / `n=` count chats still in the host FIFO, not chats already consumed after a restore onto a KV lane.

RAM `save=` / `load=` are HIP D2H/H2D elapsed for that request's admission or the throughput interval; the copies of a capture rolled back by a deferred or failed admission count only toward lifetime and interval totals.

Disk `save=` is spill-session wall harvested onto the request; disk `load=` is the host wall from the first live SSD read of that restore until the last page or state object has arrived in the pinned host window. Disk `h2d=` is the host wall from that last host arrival until the restore's page and state H2D complete (extra copy time after SSD is idle, not the overlapping first-to-last copy span).

`restores=` / `evicts=` / `drops=` are lifetime counters; lifetime capture counts stay in JSONL and on `GET /metrics`. A non-zero disk `drops=` is followed by its causes, for example `(spill_no_capacity:1)`, also recorded as `scheduler.kv_disk_drop_reasons` in `throughput` JSONL.

Exact RAM byte occupancy remains in `server_start` and `throughput` JSONL and on `GET /metrics`; set `NINFER_KV_RAM_LOG_BYTES=1` to print those byte values on the human lines. A new capture may still reap or evict while logged `kv-ram=` looks low. An explicit capacity is never silently reduced, and neither policy permits request-time pool growth.

Admission reserves the full prompt-plus-effective-output page entitlement, so an admitted request can finish within its declared bound. A later request waits in FIFO order when the remaining shared pages cannot satisfy its complete entitlement; the Engine never admits it and later truncates an older request to recover capacity. Startup rejects a KV pool smaller than one sequence, too small to provide one page per configured lane, or larger than all configured lanes could use.

Compatible resident prefixes are reused for both text and multimodal histories unless the server is started with `--no-prefix-reuse`, which also disables prefill context-checkpoint capture and the turn-rollback pin. `--context-checkpoints off` disables only the automatic ladder; occupy-append rollback and exact-hit `ninfer.capture_context_checkpoint` still pin when MTP/DFlash and prefix reuse are on. A multimodal hit requires matching token types, three-axis MRoPE positions, encoded-media digest, grid, and consumer spans; changing an earlier image or video therefore resets the prefix instead of reusing placeholder-token KV. Media wholly inside a matched prefix skips Vision execution, while new suffix media is encoded normally. The completion log reports the reused token count as `cache=`. When a prefill context-checkpoint head is restored or this request freezes one, the same line also carries `context_ckpt=restored:F` and/or `captured:F`, the absolute head frontiers.

The shared family runtime distinguishes `full_reset`, `append_frontier`, `restore_turn_checkpoint`, `restore_response_checkpoint`, `restore_context_checkpoint`, and `restore_turn_rollback`. Both rewrite checkpoint kinds include the recurrent, hidden, and selected speculative-backend continuation state required to recompute a rewritten suffix; matching KV tokens alone never authorize a partial hit.

MTP or DFlash prefill may also freeze current GDN at committed chunk ends that have reached a context mark (default 24576, 36864, 53248, 77824, 102400, 151552, or `--context-checkpoints a,b,c`); a later prompt that matches that prefix restores GDN into current and hidden into `tail_hidden` as `restore_context_checkpoint`. DFlash2 also restores that lane's cyclic local K/V and `dflash_context_frontier`.

Slot `C` is the Engine-wide GDN image for that freeze and for the turn-rollback pin: on `append_frontier` occupy with `E>0` and a real suffix (`prompt_tokens > E`), current GDN and `tail_hidden` are copied to `C` before suffix prefill so a later edit of the last user turn can restore that completed `E` as `restore_turn_rollback`. The same slot is written on an exact-hit / decode-only request (`prompt_tokens == E`) when `ninfer.capture_context_checkpoint` is true, unless a context-checkpoint head already sits at that `E` (skip, `captured_tokens = 0`; the rollback slot stays empty until a later `true` at a new `E`). A later exact-hit `true` replaces the one rollback pin and leaves ladder heads. Ladder freeze borrows `C` and reloads the rollback image afterward.

The rewrite checkpoints (`restore_turn_checkpoint`, `restore_response_checkpoint`) keep their GDN and DFlash state in the startup pinned checkpoint slab, about 187 MiB per lane with DFlash, rather than VRAM: capture snapshots the lane into staging on device and drains it to the host image on the copy stream behind later work, and restore copies staging back on device while it still holds that image, otherwise H2D from the image before the suffix prefill. Same last user regenerate still hits rewrite (`TurnClosure` is longer than rollback `E`).

With stable `preserve_thinking=true`, the auxiliary checkpoint rolls to the prompt frontier after the current response's complete deterministic generation prologue. For thinking generation this includes `<think>\n`; for non-thinking generation it includes the complete empty thinking block. Capturing that frontier does not split a tiny trailing prologue into a separate prefill unit, and a normalized response which no longer matches the raw generated tokens replays only that response and its suffix.

A closed turn with empty reasoning re-renders differently right after the assistant opener, so a thinking request whose latest assistant turn carries no reasoning (the client drops it) places the checkpoint at the generation opener instead, before `<think>\n`. That costs one short extra prefill unit, and its next turn replays the prologue, the response and the suffix rather than recomputing the whole prompt. A first turn keeps the prompt frontier, so a client that drops reasoning misses once, on its second turn. A client that echoes reasoning still reuses at least the previous prompt. Two shapes still miss on every turn: thinking off with `preserve_thinking=true` on the Qwen3.8 template (its history omits the empty generation wrapper), and a client that keeps reasoning inside a tool loop but strips it at the next user turn (the prompt diverges at the loop's first assistant turn).

Stable `false` keeps the first assistant opener in the open turn so a newly closed turn can be recomputed without its reasoning. When such a turn stops (stop token or string) without a tool call and its checkpoint is its own generation opener, the host RAM and disk tiers store it cut at that checkpoint: one state set and no reply KV per entry, and the next turn reuses it as `append_frontier` with the same reuse length. Switching that conversation to `preserve_thinking=true`, or continuing that reply as an assistant prefill, then re-prefills the reply.

`preserve_thinking` selects where the next checkpoint should live; it is not a cache-compatibility bit. An exact current frontier or matching complete checkpoint remains reusable across a mode change. If the newly desired boundary is already behind the selected reuse frontier and no snapshot exists there, the Engine keeps the valid hit and defers installing that new checkpoint rather than forcing an eager full reset. A later request that diverges before every retained checkpoint then resets normally. The JSONL completion record exposes the checkpoint actually restored as `prefix_reuse_path`, the reused length as `prefix_cache_hit_tokens`, and this request's absolute staged-checkpoint head frontiers as `context_checkpoint.restored_tokens` / `captured_tokens`. Changing reasoning effort changes rendered tokens and therefore does not reuse a prefix whose effort instruction differs.

An appended mid-conversation system message is an ordinary prompt suffix, so an unchanged prior history remains eligible for `append_frontier`. If the client modifies, removes, or moves a historical system message, the token prefix genuinely differs and a miss/reset is correct.

Speculative decoding is an engine option and does not change protocol output shapes, stop behavior, or usage accounting. If a stop truncates a multi-token MTP or DFlash round, the Engine commits the exact accepted target prefix so a following compatible turn can still reuse it. Output-limit and context-capacity finishes map to `length`/ `max_tokens`; ordinary model or string stops map to `stop`/ `end_turn`.

Function tools are rendered into the model prompt and generated calls are parsed into protocol responses. NInfer does not execute tools; its supported schema subset is enforced during decoding.

Prompt-token usage includes chat-template and expanded media tokens. Generated-token usage comes from accepted output token IDs, including a stop token whose decoded text may be withheld.

### Per-request reasoning limit

A positive uint32 cap is accepted through Chat Completions `thinking_token_budget`, Responses `ninfer.thinking_token_budget`, or Anthropic Messages `thinking.budget_tokens`. `ninfer-serve --default-thinking-token-budget N|off` sets the cap inherited when that exact request field is omitted; the built-in default is `off` (unlimited). Explicit null disables the server cap, and an explicit positive integer overrides it. Null `ninfer` or `thinking` wrappers do not opt out. Zero, fractional, negative and overflowing values are rejected; `4294967295` is supported.

The cap resolves after final thinking and reasoning effort. An inherited cap is inactive when thinking is disabled, while an explicit positive cap with disabled thinking is a 400 error. A positive supported effort can enable thinking and inherit the cap even when the server defaults thinking off. Anthropic adaptive thinking without a budget inherits the server cap. Reasoning effort itself is unchanged. The resolved effective cap is passed to the Engine and request log; server-start logs record the configured default. Docker uses `NINFER_DEFAULT_THINKING_TOKEN_BUDGET`, default `off` in Compose and `.env.example`; set it to `16384` for a deployment-wide 16K default. No public strategy selector is provided.

The policy charges newly committed reasoning body token IDs, excluding historical input and framing, cumulatively across retries. Caller stops, cancellation and total output/context ceilings retain precedence. A policy-forced native close consumes completion/context room and carries intervention provenance; it is not a naturally sampled close. The candidate still needs the authorized close-versus-clean-retry comparison before qualification; no progress or quality improvement is claimed.

For a request with a reasoning cap in the ordinary Close configuration, automatic clean retries
are disabled. Confirmed reasoning repetition keeps its analysis and continues toward natural or
capped close; cycle escape and framing repair remain active. Invalid replay continuations,
persistent answer repetition and proved duplicate-tool cycles fail explicitly rather than discard
the attempt and retry. The private Retry comparator remains separate and bounded.

Final usage adds `ninfer.reasoning_limit` when a cap is active on all three protocols, including final streaming usage: `budget_tokens`, `used_tokens`, `forced_closes`, `budget_retries`, and `limit_reached`. This policy count is distinct from standard reasoning-token usage. Request logs record the resolved cap (null for unlimited) and source parameter; completion logs and process metrics expose intervention accounting. Proxies may discard vendor usage fields, so resolved request diagnostics establish whether forwarding worked.

Reasoning-limit diagnostics also include `forced_tokens`, a separate framing provenance list with `generated_token_index`, `token`, `policy_forced=true`, and `target_logprob`. The index refers to the request's cumulative generated-token sequence, including discarded attempts. The logprob is the original unmasked target value when requested and available; otherwise it is null. These framing records do not appear as answer/reasoning channel logprobs and never report masked probability one as model confidence. Errors ending admitted requests retain the cumulative reasoning-limit ledger under `error.ninfer.reasoning_limit`, in request-error logs, and in process counters, so a failed replacement does not erase consumed reasoning or interventions.
