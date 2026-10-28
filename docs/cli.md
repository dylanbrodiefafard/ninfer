# NInfer CLI

`build-r9700/apps/ninfer` runs one request with the selected Qwen3.8-27B R9700 `.ninfer` artifact. Follow the [project README](../README.md) to build the engine and see [model-artifact.md](model-artifact.md) for the required weights.

## Text input

```bash
./build-r9700/apps/ninfer /path/to/qwen3.8-27b-r9700-lq4.ninfer \
  --prompt "Summarize the difference between prefill and decode." \
  --max-context 16384 \
  --max-new 256
```

Exactly one of `--prompt` and `--messages` is required.

Answer content is streamed to stdout. Reasoning, model loading (including the registered target and `weights_id`), timings, throughput, GPU memory, and speculative-decoding statistics are written to stderr, so stdout can be redirected independently:

```bash
./build-r9700/apps/ninfer /path/to/qwen3.8-27b-r9700-lq4.ninfer \
  --prompt "Return one sentence." --max-new 64 \
  > answer.txt 2> run.log
```

Thinking is enabled by default. If the chat template embedded in the loaded artifact exposes reasoning effort, `--reasoning-effort low|medium|xhigh` selects it; omitting the option uses the template's default. An artifact whose template does not expose effort rejects the option. Add `--no-thinking` for direct-response prompt rendering; it cannot be combined with `--reasoning-effort`. `--greedy` selects exact argmax decoding independently.

## Startup memory profile

Weights and buffers are selected when the Engine starts:

- the default loads DFlash2 and its optimized proposal head;
- `--spec none` omits speculative weights, state, and the optimized proposal head;
- `--spec mtp` loads only MTP; `--spec dflash` loads the Qwen3.8-27B DFlash2 companion when `dflash/` is present;
- `--no-lm-head-draft` uses the full proposal head, omitting the optimized head;
- Vision is disabled by default, omitting its weights, Vision scratch phase, and frozen request-transient allocation;
- `--vision` loads those allocations and enables image/video input.

The complete `.ninfer` inventory is validated even when some features are disabled. An Engine started without `--vision` rejects media; restart with the flag to enable it. Qwen3.8-27B DFlash2 can be combined with Vision when both are selected at startup. Use `--spec none` with Vision off for the smallest resident profile.

## Structured messages

`--messages` accepts either a non-empty JSON message array or an object containing `messages` and an optional `tools` array.

```json
[
  {
    "role": "system",
    "content": "Answer concisely."
  },
  {
    "role": "user",
    "content": [
      {
        "type": "image",
        "image": "../media/visual_chart.png"
      },
      {
        "type": "text",
        "text": "Describe the chart."
      }
    ]
  }
]
```

Local media paths resolve relative to the messages JSON file, independently of the working directory. Absolute paths, HTTP(S) URLs, and data URIs retain their meanings:

```bash
./build-r9700/apps/ninfer /path/to/qwen3.8-27b-r9700-lq4.ninfer \
  --messages examples/cli/messages/image_chart.json \
  --max-context 8192 \
  --max-new 128 \
  --vision
```

Supported roles are `system`, `developer`, `user`, `assistant`, and `tool`. System and developer messages retain their array positions; the Qwen family frontend renders both as system-class ChatML turns rather than moving later instructions to the beginning.

Message content may be a string or an ordered array containing:

| Content type | Source field | Accepted source |
|---|---|---|
| text | `text` | string |
| image / image_url | `image` or `image_url` | local path, HTTP(S) URL, or base64 data URI |
| video / video_url | `video` or `video_url` | local path, HTTP(S) URL, or base64 data URI |

`image_url` and `video_url` may be strings or objects containing a string `url`. Assistant history may include `reasoning_content` and `tool_calls`; a tool result uses role `tool` and `tool_call_id`.

Declare `tools` to enable schema-constrained tool calls. The CLI prints completed calls as a `{"tool_calls":[...]}` JSON object after streamed text. During generation, stderr emits `tool_call_event` JSON records (`start`, `arguments`, `completed`, `incomplete`) with stable call indices and IDs. Fragments and envelope completion are provisional until the terminal result; interrupted calls never enter executable results. An unconstrained first string parameter streams incrementally; other values wait for complete-call validation. Raw output remains raw token text. See `docs/serving.md` for schema and tool-choice support.

See [`examples/cli/`](../examples/cli/) for committed text, image, video, mixed-media, thinking, long-decode, and long-context inputs.

## Speculative decoding

The CLI defaults to DFlash2 with the optimized proposal head and adaptive drafting up to seven tokens. `--spec none` disables speculation. MTP supports one to five draft positions. DFlash2 supports Vision-composed target hidden features and target MRoPE verification positions. `--no-lm-head-draft` selects the full proposal head. `--lm-head-draft` enables the optimized head explicitly and requires a speculative backend:

```bash
./build-r9700/apps/ninfer /path/to/qwen3.8-27b-r9700-lq4.ninfer \
  --prompt "Write a short explanation of speculative decoding." \
  --max-context 16384 \
  --max-new 512 \
  --spec dflash
```

For Qwen3.8-27B DFlash2, the R9700 artifact must contain the appended `dflash/` objects. The native runtime uses chain verification with K in 1..7; `--dflash-verify-width` must equal K+1 when explicitly supplied.

```bash
./build-r9700/apps/ninfer /path/to/qwen3.8-27b-r9700-lq4.ninfer \
  --prompt "Write a short explanation of speculative decoding." \
  --max-context 16384 --max-new 512 \
  --spec dflash --draft-tokens 4 --lm-head-draft
```

MTP and DFlash cannot be enabled together. DFlash uses chain verification `W=k+1`, `k` in `1..7`. `--adaptive-draft` selects live K from `{3..N}` for `--draft-tokens N` (5..7; MTP stays `{3,4,5}`) using measured expected yield divided by round time; explicit `--draft-tokens 4` stays fixed. `--spec dflash` defaults to seven drafts with adaptive length enabled. Use `--no-adaptive-draft` for a fixed length (see `docs/performance.md`).

For DFlash, K describes the physical captured graph. Per-request output and context limits still clip the logical proposals and published tokens. Near an output limit, a larger padded graph may be selected only with measured cost at the current batch size and enough remaining physical context in every row; K3 remains eligible. Selection never adds graphs at runtime. R9700 speed recommendations require R9700 end-to-end measurements; NVIDIA timings do not transfer.

## Common options

| Option | Meaning | Default |
|---|---|---:|
| `--max-context N` | per-sequence logical context ceiling | `2048` |
| `--kv-capacity N\|auto` | explicit shared Main Text KV capacity, or maximize it from remaining GPU memory; omitted means `--max-context` | `2048` |
| `--kv-capacity-headroom MiB` | device memory `--kv-capacity auto` leaves free; requires `auto` | `64` |
| `--kv-ram-capacity off\|N` | pinned host capacity in MiB for completed prefixes and live suspension; enables output windows up to 8192 tokens | `off` |
| `--kv-ram-system-prompts N` | Retain up to N rendered system prefixes in the RAM budget; requires RAM capacity when nonzero | `0` |
| `--kv-disk-capacity off\|N` | SSD KV prefix-cache unique-object capacity in MiB; `off` disables the tier | `off` |
| `--kv-disk-location PATH` | directory for the SSD page store; required iff `--kv-disk-capacity` is enabled | unset |
| `--kv-disk-compress off\|zstd` | zstd-1 on new GDN/hidden/cyclic writes; KV pages stay uncompressed | `off` |
| `--prefill-chunk N` | positive text-prefill chunk, in multiples of 128 | `2048` |
| `--max-new N` | requested output-token limit | `128` |
| `--device N` | HIP device index | `0` |
| `--spec none\|mtp\|dflash` | speculative backend | `dflash` |
| `--draft-tokens N` | MTP `1..5`, DFlash2 `1..7` | `7` with DFlash; required with MTP |
| `--adaptive-draft` | pick live draft K in `{3..N}` (DFlash `--draft-tokens N>=5`; MTP `{3,4,5}`) by locking `E[Y]/T(k,C,L)` (nested `r_i`; least-squares T; at most one probe of an unmeasured k; 1 ms switch cost). `--draft-tokens 4` stays `{4}` | on with DFlash; off with MTP |
| `--no-adaptive-draft` | use the fixed `--draft-tokens` length instead of adapting | off |
| `--dflash-verify-width N` | DFlash2 chain verify width `W=k+1`, `2..8` | auto |
| `--dflash-p-less-draft-temperature T` | Pins the DFlash2 draft temperature `0..2` for p-less requests: drafts are drawn from the 16-candidate path-select softmax at `T` and verified against that proposal, so output stays exactly the p-less target distribution. `0` drafts greedily. Unset, the engine calibrates it online per p-less `--temperature` and draft length: every chain round scores eight candidate temperatures against the verified target distribution and the next round uses the best ([model §8](maintainer/qwen3.8-27b-model.md#8-dflash2-block-diffusion-draft-model)) | calibrated |
| `--lm-head-draft` / `--no-lm-head-draft` | enable or disable the optimized proposal head | on with DFlash |
| `--vision` | enable image/video input and load Vision GPU allocations | off |
| `--no-device-graph` | disable Device Graph decode | graphs on |
| `--context-checkpoints off\|a,b,c` | disable the automatic prefill ladder, or replace the default marks (24576, 36864, 53248, 77824, 102400, 151552). Custom lists require `--spec mtp` or `--spec dflash`. Marks at or above `--max-context` stay unused. Advertised freeze `F` is the committed chunk end at or past the mark, not the raw named size. | default ladder |
| `--capture-context-checkpoint` | pin the current resume frontier `E` on an exact-hit / decode-only request (the same one-slot turn-rollback head automatic occupy-append already writes). A fresh one-shot run has `E == 0`, so this is a no-op unless a retained lane already exists in the process. | off |
| `--no-thinking` | disable thinking in prompt rendering | thinking on |
| `--reasoning-effort low\|medium\|xhigh` | select an effort exposed by the loaded chat template | template default |
| `--greedy` | exact argmax decoding | off |
| `--no-p-less-sampling` | opt out of p-less and use top-p/top-k/min-p/penalties | p-less on |
| `--temperature F` | sampling temperature override | `1.5` with p-less; model/mode preset otherwise |
| `--top-p F` | nucleus-threshold override | registered model/mode default |
| `--top-k N` | top-k-threshold override | registered model/mode default |
| `--min-p F` | min-p-threshold override | registered model/mode default |
| `--presence-penalty F` | presence-penalty override | registered model/mode default |
| `--frequency-penalty F` | frequency-penalty override | registered model/mode default (`0`) |
| `--seed N` | sampling seed | `0` |

When a sampling flag is omitted, Engine selects the official general-task preset registered for the loaded model and the rendered prompt mode. The standard-sampler presets (selected with `--no-p-less-sampling`) are:

| Model | Prompt mode | Temperature | Top-p | Top-k | Min-p | Presence penalty |
|---|---|---:|---:|---:|---:|---:|
| Qwen3.8-27B | thinking | `2.0` | `0.95` | `20` | `0` | `0` |
| Qwen3.8-27B | non-thinking | `2.0` | `0.80` | `20` | `0` | `0` |

Frequency penalty is `0` in every registered preset. Qwen's separate precise-coding recommendation is task-specific and is therefore an explicit override rather than an inferred Engine default.

P-less is the default process/request truncation mode, with temperature `1.5` when no override is supplied. It keeps temperature and seed, ignores top-p, top-k, min-p, and presence/frequency penalties, and writes a one-time warning on stderr. Ignored parameters must still satisfy their normal input ranges. `--no-p-less-sampling` opts into the registered production sampler. Combined with `--greedy`, p-less remains exact argmax.

During thinking, p-less also exits a generated token-id square: the least period p in [32, 2048] such that the last 2p generated ids match with Hamming distance at most 2p/512 (so p<256 is exact identity). The continuation is excluded from the already-computed typical set V (renormalized V without that atom, or the in-domain runner-up when V is that singleton). This is not a `suppressed_tokens` member and does not rebuild L. It does not detect duplicate tool calls across requests and does not alter tool-call content. There is no CLI flag.

P-less membership is `p_v ≥ max(L·exp(-2ε/T), 1/M)` with `ε = 1/16` (first-order softmax perturbation of the logits) and `M = 1024`; L is the unperturbed collision probability, and an empty set falls back to the eligible mode.

Under MTP or DFlash2, p-less applies at every hop (block verification over the chain with the recorded draft `q`; DFlash2 drafts are sampled at the calibrated or pinned `--dflash-p-less-draft-temperature`, MTP drafts are one-hot) and to the bonus after a full accept. The cycle exclusion applies only to the first hop's next-token decision; later hops use their unmodified p-less candidate sets. Temperature zero remains greedy at every hop. The reasoning terminator (including split-token forms) and model stop tokens are never cycle exclusions. This policy does not impose a maximum reasoning length.

With declared tools, the answer's free-text grammar excludes orphan `</invoke>`, `</parameter>`, `</function>`, and `</tool_call>` strings outside valid call envelopes. Actual call framing and literal XML inside schema-valid arguments remain allowed. Reasoning excludes `<tool_call>` so a real call must follow `</think>`; ordinary reasoning and tools-off/raw output remain allowed. This is a sampling-domain constraint, not response-text deletion or an automatic retry.

With current declared tools, Engine separately withholds suspected duplicate-tool loops using repeated reasoning, identical calls and unchanged associated results. It can extend the cached context and retry at most twice, within the original output budget and resource reservation. Rejected calls are not printed or executed; already printed reasoning/prose is not retracted. Exhaustion is an explicit request error, not forced EOS or an engine shutdown.

With generation recovery enabled, persistent generated reasoning is detected independently of the sampler and temperature: three non-overlapping occurrences of the same exact 256-token reasoning passage must appear in the current generated attempt, and repeated passages must cover at least 4,096 distinct redundant tokens. Overlapping windows count those tokens only once. This is a repetition-evidence threshold, not a 4,096-token reasoning limit. The occurrences need not have a fixed separation, so changing words elsewhere in a multi-paragraph loop does not hide the repeated passage. Hashes locate candidates; exact token comparison confirms them. Two copies alone do not trigger a retry. Long reasoning without that repetition is not limited.

The retry keeps the cached prompt, including historical reasoning. It closes the previous assistant prologue with its matching terminator and appends the notice or rejected-call feedback; only the failed generation is omitted. A later retry appends another notice after the first; the earlier notice stays. A ready checkpoint at the prompt frontier is restored and only the appended suffix is prefilled. Original user content and actual tool results stay in that prompt, and an explicitly labeled engine system notice asks for concrete progress. No call or tool result is invented.

With generation recovery enabled, a thinking response can quarantine its initial answer while its generated token IDs exactly copy the preceding generated reasoning passage (at least 256 tokens). Genuine reasoning still streams normally. A divergent answer, an extension beyond the passage, or an uncorroborated terminal copy releases held text and its original logprobs unchanged. Only a complete exact copy licenses one additional dedicated `</think>` token: that consumed token confirms a framing replay, discards the held copy, and rearms quarantine. The extra close is not published or fed to ordinary tool grammar, but remains in generated IDs, KV state and completion usage. Another complete copy, or natural termination without a separate answer or selected tool after repair, requests an unpublished retry with a labeled engine notice. Caller stops and cancellation retain precedence. Repair and proof boundaries end the accepted speculative prefix; hypothesis masks license the extra close only at the matching boundary. Raw, schema-constrained and tool-argument contents are excluded. The original output budget and shared two-retry cap still apply; exhaustion is explicit. Intentionally requesting a long exact reasoning copy with an extra close can trigger this policy. After a confirmed framing replay triggers a retry, the new assistant continuation uses the model's native non-thinking prologue. The original input and historical reasoning stay intact; only the new generation mode changes. The frontend carries the already-corroborated immutable reasoning reference into these replacements, so a complete copy of that same passage triggers recovery without licensing another close. A subsequent closed-thinking prompt is ended with a turn terminator alone, preserving all earlier notices without inserting an extra reasoning close. All replacement answer/tool deltas and their logprobs remain withheld until ordinary terminal repetition/tool validation accepts the attempt; reasoning remains visible. A failed replacement is discarded and may retry only within the remaining shared budget. Explicit caller stops release held text; cancellation releases text while omitting provisional tool events. For an initially non-greedy p-less request, these replay-recovery attempts use standard sampling at temperature 1.0, top-k 20, top-p 0.95, min-p 0 and zero presence/frequency penalties, retaining the resolved request seed. Initial generation and standard/greedy request profiles are unchanged. The actual retry profile and `enable_thinking` mode are recorded on `retry_started` recovery events. A replacement that exhausts its output/context budget without non-whitespace answer text or a complete tool call fails explicitly; caller stops and cancellation retain precedence. This is framing recovery and publication validation, not a check that an answer or external task is correct.

With generation recovery enabled, free-text answers also fail explicitly on persistent exact repetition (three non-overlapping 256-token copies and at least 4,096 distinct redundant tokens). The detector does not join across reasoning/tool/framing gaps and excludes raw and schema-constrained output; monitoring ends when the response selects a tool. Published prose remains visible; it is not replaced by another attempt. The failure is `generation_recovery_exhausted`, including on natural EOS, while explicit caller stops and cancellation retain precedence. Deliberately repetitive long output may trigger the guard; `--no-generation-recovery` disables it. This is not a task-completion or factual-accuracy check.

Reasoning evidence is attributed to contiguous reasoning-channel tokens before publication and remains established if the same round closes reasoning or reaches a natural stop or budget boundary. That round's unpublished answer/tool output is withheld before retrying; it is not concatenated with the replacement. Text-only thinking requests can retry, including with standard or greedy sampling. A proved repeat without a supported retry context (for example, media) fails explicitly instead. Raw output is excluded. Reasoning, framing-replay and duplicate-tool recovery share the two-retry budget; duplicate-tool detection still requires substantive repeated reasoning and never treats empty-reasoning polling as proof. See the serving reference for scope and recovery usage fields.

Repeat `--stop-token-id`, `--stop`, or `--reasoning-stop` to add stop conditions. Use `--raw-output` to expose the frontend's raw output stream and `--print-token-ids` to include generated token IDs in diagnostics. During structured Qwen output, registered model stop tokens are excluded from sampling while reasoning is open and after the reasoning terminator until non-whitespace answer content begins. This prevents a response from ending inside reasoning or with an empty post-reasoning answer. For tools-enabled prompts, they are also excluded while a `<tool_call>` opener is ambiguous or a tool call is incomplete, and become eligible after the matching `</tool_call>`. A speculative round sampled with those tokens excluded commits only through the first completing `</tool_call>`; the next round can then stop. Other caller-added stop conditions remain active, and raw output does not apply this structured-output guard.

Run `./build-r9700/apps/ninfer --help` for all options and defaults.

Live-image credit metadata allocation failure follows the unfunded-capacity policy: admission may reduce its output window or defer. Failed suspension, resume, or window metadata allocation preserves the existing continuation and allows retry; a failed GPU synchronization remains fatal.

## Message-input scoring

`ninfer-ppl` accepts exactly one of `--ids` or `--messages`. Message input uses the same owning prompt adapter as generation; `--vision` enables image/video parts and `--thinking` controls the template. Set `--max-context` for message input. `--score-last-message` scores the final completed assistant payload, including its reasoning and turn closure, and requires preceding history. Its boundary comes from the actual rendered full-token sequence; an indivisible token crossing the header/payload boundary is scored whole. It cannot be combined with `--skip`. `--tokens` applies only to token-ID input. For example:

```bash
./build-r9700/apps/ninfer-ppl --weights /absolute/path/to/selected.ninfer \
  --messages tests/fixtures/frontend/ppl_reply_messages.json \
  --max-context 1024 --score-last-message --out-json /tmp/reply-score.json
```

## Context and memory

The registered model has a native context limit of 262,144 tokens. The practical allocation on one R9700 depends on the selected artifact, media workload, output budget, and enabled startup features. Growing Text/MTP KV always uses FP8 E4M3FN keys, signed INT4 values, and FP16 value scales; there is no runtime cache-format selector. The prepared prompt must fit `--max-context`; generation stops at the remaining context capacity when necessary.

`--kv-capacity N` controls the shared physical Main Text KV pool independently and is rounded up to the 64-token page size. `--kv-capacity auto` loads the selected weights, measures the remaining GPU memory, and directly chooses the largest legal page capacity for the complete enabled runtime layout. This includes the selected speculative backend, fixed sequence state, workspace, Vision request transient, and Device Graph allowance, while leaving `--kv-capacity-headroom` MiB (default 64) available for external GPU users and display activity. NInfer's own allocations and lazy driver residency are included in its runtime budget; they do not consume this reserve. Raise it when a desktop or another process needs more GPU memory. A startup failure in automatic mode names this option.

The token embedding is held in pinned host memory rather than VRAM (the load summary's `weight pinned host`), so it is not part of the device weights. It does not probe allocations or resize the pool at request time. The single-request CLI normally leaves the option omitted so it follows `--max-context`; the distinction matters primarily to a concurrent Engine or server.

Checkpoint images live in one driver-backed HIP/GTT pinned host slab allocated at startup outside `--kv-ram-capacity`: each lane's turn (rewrite) checkpoint image of the GDN state (146.8 MiB on Qwen3.8-27B, plus 40 MiB of DFlash local K/V), and with MTP or DFlash a pool of context-checkpoint heads (the same image plus the hidden row) for prefill-ladder and turn-rollback heads, including heads restored from the RAM or disk tier.

The pool holds one rollback head per lane plus the most ladder heads all lanes can hold at once: marks above `--max-context` are unreachable, and a lane holding k ladder heads keeps at least the k-th mark's tokens of the shared KV capacity. The Compose deployment (C=1, 32768-token max context, DFlash) reaches only the 24576 mark, so it holds 2 heads (0.37 GiB) plus one 0.18 GiB rewrite image; C=1 at a 262144-token max context with KV capacity for every mark holds 7 heads (1.3 GiB).

Serve prints the slab as `ckpt-pin=` and `ckpt-heads=` on the KV capacity line and the CLI as `checkpoint host pinned`. Serving never allocates or frees pinned checkpoint memory; while every head is owned, an optional capture or restored head is skipped. Startup fails with the required size when the slab would leave less than 4096 MiB of the host's available memory; `--context-checkpoints off` shrinks the pool to one turn-rollback head per lane, and without MTP or DFlash there is no pool.

With MTP or DFlash, capture copies the state into the Engine-wide device staging slot (a device-to-device copy) and drains it to the host image on the copy stream behind later work; a restore while staging still holds that image copies it back on the device, otherwise it costs one H2D before the suffix prefill.

`--kv-ram-capacity N` is a separate pinned-host budget in MiB shared by completed prefix bundles and secured live continuation images. Enabling it automatically funds the full prompt plus output windows of at most 8192 tokens; `off` retains full-lifetime GPU reservation. A window ending never reduces the requested output limit. Under KV pressure, one blocked lane waits while peers advance; a safely funded queued replacement, two blocked lanes, or no runnable work may trigger settled-boundary RAM suspension. Live images cannot be evicted as optional prefixes or spilled to disk, and resume continues without output replay. Window-end RAM credits for ordinary residents and a completion cohort whose target-produced lifetime envelopes jointly fit GPU capacity preserve a completion path within finite backing. Cohort members acquire only their current windows and need no own evacuation credit. Unused future window entitlement can be reclaimed without discarding current state or reducing the output limit. Inadequate backing can still reduce resident concurrency. Aggregate throughput qualification remains in progress. It is not a token capacity, does not enlarge the GPU pool, and defaults to `off`. `N` must be a positive decimal integer; `0` is rejected. Construction fails if the host pin cannot be allocated. The fixed host-memory policy uses driver-backed GTT allocations with normal unified addressing; it changes no system-wide compaction setting. The RAM tier shares the GTT budget with checkpoint slabs, mapped weights, transfer staging and runtime host buffers. Host RAM availability alone does not guarantee admission; startup reports the requested allocation and remaining GTT budget when that budget is insufficient. Requested capacity is never silently reduced.

`--kv-disk-capacity N` is a third-tier SSD budget in MiB of actual pack-file bytes, including free holes and retired reader extents. It requires `--kv-ram-capacity > 0` and `--kv-disk-location PATH`. Location without capacity is an error. `--kv-disk-compress zstd` compresses new GDN/hidden/cyclic blobs only; KV pages are never application-compressed. Disk is inclusive: a VRAM or RAM hit does not delete the committed SSD generation. Equal reuse prefers VRAM, then RAM, then disk.

Disk format v8 fingerprints canonical logical KV pages and persistent-state dimensions, excluding GPU pool and DFlash cyclic lane capacities. A compatible directory can reopen across `--vision` on/off and automatic capacity changes. Media content remains part of each entry identity. The artifact fingerprint is SHA-256 over selected tensor descriptors and payload ranges: Text state producers and Vision, the complete native MTP block when enabled, and DFlash context feature projection, normalization and K/V rows when enabled. Output/proposal-only weights and file location/inode/timestamps do not affect reuse; copied artifacts with identical selected tensors share a fingerprint. Artifacts remain immutable while an Engine uses them. Pre-v8 locations fail startup: select a fresh directory. Model/weights identity, fixed codec, speculative backend and persistent-state incompatibilities also fail startup.

The disk worker reuses coalesced holes before appending and trims free tails. It performs bounded idle defragmentation only with Engine idle permission, moving toward lower offsets in paced 1 MiB chunks and yielding when work arrives. A durable location journal and reader extent leases preserve restart and in-flight read correctness. Metadata and filesystem headroom are additional to the configured pack-file budget.

Orderly Engine shutdown copies active chats into the host cache, saves cache entries that are not yet on SSD, and finishes outstanding disk writes. When disk is enabled, the same stderr progress renderer prints `kv-disk` `copy active chats` / `save cache entries` / `finish disk writes` counts as shutdown runs.

`--kv-ram-system-prompts N` optionally retains up to N system-prefix snapshots in the same RAM budget. Zero disables it. A snapshot ends after the leading system/developer messages and template-generated tool/instruction block, before the first user message; that user's content need not match on reuse. Exact rendered tokens, positions and media identity determine a hit. Different tool definitions or system instructions can create a distinct entry. A longer matching conversation prefix still wins.

First-use prefill stops at the exact system boundary and copies Text/backend KV plus GDN and DFlash state. Subsequent compute waits for that capture before overwriting state or appending inside its last page. A system restore retains the source and updates its LRU recency. System snapshots are protected from ordinary FIFO/disk eviction and are not written to disk. New system captures may reclaim ordinary FIFO entries, then older system entries when their shared byte budget requires it. N is an upper bound: captures that cannot fit or would evict an in-flight source are skipped. Retired in-flight blocks still occupy the existing pin until their events complete. `--no-prefix-reuse` disables use and creation; `cache_write=false` prevents new snapshots. There is no separate RAM allocation or automatic disk fallback for these entries.

Ordinary conversation retention in host RAM is an exclusive FIFO: a bundle lives in VRAM or in this budget, not both. One long MTP or DFlash bundle with five context-checkpoint heads is about 6 GiB (Main KV, optional MTP KV or DFlash cyclic state, plus GDN checkpoint images); size the budget accordingly. `off` still captures live-lane GDN into the startup checkpoint pool so same-lane rollback works; other-lane restore after eviction remains a miss.

Startup still prints capacity plus `used`/`entries`. Serve `[req] done` and throughput lines print live host-resident `kv-ram=` used bytes plus `n=` / `restores=` / `evicts=` / `drops=` / `save=` / `load=`. When disk is enabled the same lines also print `kv-disk=` occupancy and counters. `kv-ram=` / `n=` exclude a chat after consume following a restore onto a KV lane; a later spill recaptures it as a new FIFO tail.

RAM `save=` / `load=` are HIP event elapsed for that request's RAM-tier D2H capture and H2D unpack of the FIFO bundle (Main+backend KV, current GDN/cyclic state, and hidden); rewrite-checkpoint and ladder GDN/cyclic images already live in pinned host memory and are copied host-to-host outside that span. Admission completes only after its copies land, so it bills them itself; a capture that a deferred or failed admission rolled back counts only toward the lifetime totals, never toward another request.

Disk `save=` is spill-session wall harvested onto the request; disk `load=` is the host wall from the first live SSD read of that restore until the last page or state object has arrived in the pinned host window (not H2D, and not a sum of overlapped SSD and copy clocks). Disk `h2d=` is the host wall from that last host arrival until the restore's page and state H2D complete (extra copy time after SSD is idle, not the overlapping first-to-last copy span). They are not admission wait, and they do not include live-lane context-checkpoint freeze D2H or a VRAM-resident restore that unpacks already-pinned lane GDN.

`restores=` / `evicts=` / `drops=` are lifetime counters on both lines. CLI `KV RAM events` prints lifetime captures/restores/evicts/drops plus that request's `save=` / `load=`. CLI `KV disk events` also prints `h2d=` (post-disk H2D wall).

The generation summary also prints `prefix reuse path`, `prefix reuse source`, and `context checkpoint` (`restored:F` / `captured:F` absolute ladder or turn-rollback head frontiers). Exact-hit `--capture-context-checkpoint` uses the same `captured:F` field. Exact byte values remain on the Engine API and in the JSONL request log; set `NINFER_KV_RAM_LOG_BYTES=1` to print those same byte counts on the human lines. A new capture may still need to reap or evict while logged occupancy looks low, because a just-consumed copy can occupy the pin until its HIP event completes.

At Engine startup NInfer reserves model weights, persistent sequence state, one phase-reused Program scratch arena, the maximum Vision request-transient buffer when Vision is enabled, and a separate Device Graph driver allowance. Scratch is the maximum of the enabled Text, MTP, DFlash, and Vision phases, not their sum. Its prefill bound uses `min(--prefill-chunk,--max-context)`. The request-transient buffer is also frozen at startup; a media request activates only the needed prefix and performs no project-owned device allocation or growth.

All weight, sequence, workspace, request-transient, and graph allocations are released when the Engine is destroyed.

## Constrained output

Use `--json-object` for arbitrary JSON objects, `--json-schema FILE` for a JSON Schema, `--grammar FILE` for XGrammar EBNF with a `root` rule, `--regex PATTERN` for a full-match regex, or repeated `--choice TEXT` for literal alternatives. Select one mode; choice alternatives may include the empty string.

```bash
build-r9700/apps/ninfer /path/to/qwen3.8-27b-r9700-lq4.ninfer --prompt 'Return a JSON object with an answer.' \
  --json-schema answer.schema.json --no-thinking --max-new 256
```

The constraint applies to answer content after reasoning, and supports Device Graphs, MTP, and DFlash through the same token masks used for tools. Schema assertions are enforced at decoding; unsupported assertions are rejected at preparation rather than relaxed. Describe the desired answer in the prompt; the schema is not added to the conversation. Tools and output constraints cannot be combined. Stops, cancellation, or output/context limits may truncate a valid prefix; complete JSON validity requires finishing the grammar. See the constrained-output contract in `docs/serving.md` for the supported schema subset and HTTP triggers.

The shared schema subset supports string pattern/Unicode-length conjunctions, supported `allOf` intersections, and finite-value filtering against sibling assertions. Tuple intersections, fractional integer bounds, and broader binary64 bounded-number emission remain unsupported. Bounded numbers use at most six fractional digits; inexact source numbers and unrepresentable bounds fail admission. See `docs/serving.md` for exact limits and source diagnostics.

For active constraints, the generation summary reports terminal branch and complete/terminated states, compile-cache outcome, compile/wait/mask host times, mask calls/nodes, and enqueued mask payload bytes. Wait is part of compile elapsed time. Recovery work is counted once; terminal state describes the final attempt. These counters do not measure GPU synchronization time.

`--thinking-token-budget N` sets a per-request positive uint32 reasoning-token cap; omission is unlimited. It requires thinking and parsed output. The cap counts newly committed reasoning body token IDs, excludes history and framing, and remains cumulative through retries. Total output/context limits and caller stops still apply. This default-off candidate remains unqualified until the authorized comparison and joint KV checks run.
