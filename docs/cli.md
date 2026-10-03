# NInfer CLI

`build/apps/ninfer` runs one request against one registered `.ninfer` artifact. Build NInfer and
download an artifact using the [project README](../README.md) before following this guide.

## Text input

```bash
./build/apps/ninfer models/qwen3_8_27b_nvfp4.ninfer \
  --prompt "Summarize the difference between prefill and decode." \
  --max-context 16384 \
  --max-new 256
```

Exactly one of `--prompt` and `--messages` is required.

Answer content is streamed to stdout. Reasoning, model loading (including the registered target and
canonical `weights_id`), timings, throughput, GPU memory, and speculative-decoding statistics are
written to stderr, so stdout can be redirected independently:

```bash
./build/apps/ninfer models/qwen3_8_27b_nvfp4.ninfer \
  --prompt "Return one sentence." --max-new 64 \
  > answer.txt 2> run.log
```

Thinking is enabled by default. If the chat template embedded in the loaded artifact exposes
reasoning effort, `--reasoning-effort low|medium|xhigh` selects it; omitting the option uses the
template's default. An artifact whose template does not expose effort rejects the option. Add
`--no-thinking` for direct-response prompt rendering; it cannot be combined with
`--reasoning-effort`. `--greedy` selects exact argmax decoding independently.

## Startup memory profile

GPU residency is frozen when the Engine starts:

- no `--spec` omits MTP/DFlash weights and state and the optimized proposal head;
- `--spec mtp` loads only MTP, while `--spec dflash` loads only the DFlash companion
  (35B-A3B DFlash v1, or Qwen3.8-27B DFlash2 when `dflash/` is present);
- a speculative backend with the full proposal head omits the optimized proposal head;
- Vision is disabled by default, omitting its weights, Vision scratch phase, and frozen
  request-transient allocation;
- `--vision` loads those allocations and enables image/video input.

The complete `.ninfer` inventory is still validated. These choices are not lazy loading: an
Engine without `--vision` rejects media and cannot enable Vision later. Qwen3.8-27B DFlash2 can be
combined with Vision when both are selected at startup; 35B-A3B DFlash v1 remains text-only. The
default speculative and Vision settings produce the smallest resident profile.

## Structured messages

`--messages` accepts either a non-empty JSON message array or an object containing `messages`
and an optional `tools` array.

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
        "image": "examples/cli/media/visual_chart.png"
      },
      {
        "type": "text",
        "text": "Describe the chart."
      }
    ]
  }
]
```

Run message files from the repository root when they contain repository-relative media paths:

```bash
./build/apps/ninfer models/qwen3_8_27b_nvfp4.ninfer \
  --messages examples/cli/messages/image_chart.json \
  --max-context 8192 \
  --max-new 128 \
  --vision
```

Supported roles are `system`, `developer`, `user`, `assistant`, and `tool`.
System and developer messages retain their array positions; the Qwen family frontend renders both
as system-class ChatML turns rather than moving later instructions to the beginning.

Message content may be a string or an ordered array containing:

| Content type | Source field | Accepted source |
|---|---|---|
| text | `text` | string |
| image / image_url | `image` or `image_url` | local path, HTTP(S) URL, or base64 data URI |
| video / video_url | `video` or `video_url` | local path, HTTP(S) URL, or base64 data URI |

`image_url` and `video_url` may be strings or objects containing a string `url`. Assistant
history may include `reasoning_content` and `tool_calls`; a tool result uses role `tool` and
`tool_call_id`.

Current `tools` declarations enable Engine-owned constrained tool generation. Complete,
schema-validated calls are returned separately from content; CLI prints them as a
`{"tool_calls":[...]}` JSON object after streamed text. An interrupted tool envelope is
not printed as a partial executable call. Raw output remains raw token text. The schema
subset and unsupported tool-choice modes are described in `docs/serving.md`.

See [`examples/cli/`](../examples/cli/) for committed text, image, video, mixed-media, thinking,
long-decode, and long-context inputs.

## Speculative decoding

Speculative decoding is disabled by default. Select MTP with one to five draft positions, 35B-A3B
text-only DFlash v1 with one to fifteen, or Qwen3.8-27B NVFP4 DFlash2 with one to eleven. The
Qwen3.8 companion is text-only internally, but a `--vision` Engine supplies it target hidden
features after Vision embedding composition and uses target MRoPE positions for verification.
`--lm-head-draft` selects the optimized proposal head and requires a selected backend:

```bash
./build/apps/ninfer models/qwen3_8_27b_nvfp4.ninfer \
  --prompt "Write a short explanation of speculative decoding." \
  --max-context 16384 \
  --max-new 512 \
  --spec mtp --draft-tokens 3 \
  --lm-head-draft
```

For 35B-A3B DFlash v1:

```bash
./build/apps/ninfer models/qwen3_6_35b_a3b.ninfer \
  --prompt "Write a short explanation of speculative decoding." \
  --max-context 16384 --max-new 512 \
  --spec dflash --draft-tokens 7 --lm-head-draft
```

For Qwen3.8-27B DFlash2, the NVFP4 artifact must contain the appended `dflash/` objects. A fixed
draft window verifies the chain `W=k+1` for `k` in `1..7`. On RTX 5090,
`--draft-tokens 7 --adaptive-draft` is the recommendation for one to two concurrent requests
under every sampler (at C>=4 under p-less, `--draft-tokens 5` measured 1.6-4.3% higher before the
tree arm, see [K7 measurements](performance.md#dflash2-k6k7-verify-2026-09-29)): besides the chain
arms `{3..7}` it captures a packed best-first draft tree of the full window (`W=12`) for batches
of up to four requests, and learns per batch size which arm maximizes expected tokens per second
([tree arm measurements](performance.md#dflash2-best-first-tree-arm-2026-10-03)). The picker
chooses the arm after each round by `argmax E[Y(arm)] / T(arm,C,L)` (nested hop survival from
engine-global hop hazards and a per-request content factor, exponentially forgetting
least-squares round time). That is a sticky policy, not a once-per-launch latch: see
[adaptive draft length](maintainer/qwen3.6-27b-model.md#81-adaptive-draft-length). Frozen
`--draft-tokens 4` stays `{4}` plus the tree arm when adaptive.

```bash
./build/apps/ninfer out/qwen3_8_27b_nvfp4_dflash_w8.ninfer \
  --prompt "Write a short explanation of speculative decoding." \
  --max-context 16384 --max-new 512 \
  --spec dflash --draft-tokens 4 --lm-head-draft
```

MTP and DFlash cannot be enabled together. `--spec dflash` on a 27B file without `dflash/` fails
at bind. Current 3.8 MTP-only files and all 3.6-27B files stay valid MTP artifacts. The published
[performance results](performance.md) use MTP with three draft tokens and DFlash with seven draft
tokens (block length eight), both with the optimized proposal head. Those DFlash k=7 figures are
historical chain W=8; product DFlash2 is chain `k≤7`. 35B DFlash v1 accepts up to fifteen draft
tokens; 3.8 DFlash2 accepts up to seven. The RTX 5090 tree investigation is in
[dflash2-tree-speed.md](maintainer/dflash2-tree-speed.md).

## Common options

| Option | Meaning | Default |
|---|---|---:|
| `--max-context N` | per-sequence logical context ceiling | `2048` |
| `--kv-capacity N\|auto` | explicit shared Main Text KV capacity, or maximize it from remaining GPU memory; omitted means `--max-context` | `2048` |
| `--kv-capacity-headroom MiB` | device memory `--kv-capacity auto` leaves free; requires `auto` | `64` |
| `--kv-ram-capacity off\|N` | pinned host KV prefix-cache capacity in MiB; `off` disables the tier | `off` |
| `--kv-disk-capacity off\|N` | SSD KV prefix-cache unique-object capacity in MiB; `off` disables the tier | `off` |
| `--kv-disk-location PATH` | directory for the SSD page store; required iff `--kv-disk-capacity` is enabled | unset |
| `--kv-disk-compress off\|zstd` | zstd-1 on new GDN/hidden/cyclic writes; KV pages stay uncompressed | `off` |
| `--prefill-chunk N` | positive text-prefill chunk, in multiples of 128 | `4096` |
| `--max-new N` | requested output-token limit | `128` |
| `--device N` | CUDA device index | `0` |
| `--kv-dtype bf16\|int8\|nvfp4` | KV-cache storage | `nvfp4` |
| `--sage` | Sage3 recipe on NVFP4 KV: V is stored with one scale per (dim, 16-key block), and prompt prefill runs PV as FP4 P × FP4 V (with SmoothQ); decode and verify keep BF16 P·V on the same cache. Requires `--kv-dtype nvfp4`; excludes `--keep-frac` / `--xattn-tau` | off |
| `--keep-frac F` | Sparge prompt-prefill tile skipping on NVFP4 KV: per 128-query tile, keep the top fraction `F` `(0,1]` of 64-key tiles by mean-Q·mean-K score, plus forced leading-sink and local-window tiles. `1` is dense; decode, verify and ≤6-token chunks stay dense | `1` |
| `--xattn-tau F` | XAttention prompt-prefill block skipping on NVFP4 KV: keep the 128-key blocks covering attention mass `F` `(0,1]` once more than 8,192 keys are visible. `1` is dense; exclusive with `--keep-frac` below `1`; decode and verify stay dense | `1` |
| `--spec mtp\|dflash` | speculative backend | off |
| `--draft-tokens N` | MTP `1..5`; 35B DFlash `1..15`; 3.8 DFlash2 `1..7` | unset |
| `--adaptive-draft` | pick the live draft arm by `E[Y]/T(arm,C,L)` (nested hop acceptance; DFlash learns engine-global hop hazards from one exploration round in 32, and tree depth hazards from every tree round; forgetting least-squares T; each captured arm measured once per batch size; 1 ms switch cost). DFlash with `--draft-tokens N` (N=5..7) captures chains `{3..N}`, `--draft-tokens 4` stays `{4}`; Qwen3.8 DFlash2 adds the `W=12` tree arm for batches of up to four. MTP captures `{3,4,5}` up to its configured limit | off |
| `--dflash-verify-width N` | pin the draft window's DFlash verify width (`W=k+1` chain, wider: packed tree, at most 16) for every batch size and disable the adaptive tree arm; chain-only targets require `W=k+1` | auto |
| `--dflash-p-less-draft-temperature T` | Pins the DFlash2 draft temperature `0..2` for p-less requests: drafts are drawn from the 16-candidate path-select softmax at `T` and verified against that proposal, so output stays exactly the p-less target distribution. `0` drafts greedily. Unset, the engine calibrates it online per p-less `--temperature` and draft length: every chain round scores eight candidate temperatures against the verified target distribution and the next round uses the best ([calibration](performance.md#dflash2-p-less-proposal-calibration-2026-10-03)) | calibrated |
| `--lm-head-draft` | optimized proposal head | off |
| `--vision` | enable image/video input and load Vision GPU allocations | off |
| `--no-cuda-graph` | disable CUDA Graph decode | graphs on |
| `--context-checkpoints off\|a,b,c` | disable the automatic prefill ladder, or replace the default marks (24576, 36864, 53248, 77824, 102400, 151552). Custom lists require `--spec mtp` or `--spec dflash`. Marks at or above `--max-context` stay unused. Advertised freeze `F` is the committed chunk end at or past the mark, not the raw named size. | default ladder |
| `--capture-context-checkpoint` | pin the current resume frontier `E` on an exact-hit / decode-only request (the same one-slot turn-rollback head automatic occupy-append already writes). A fresh one-shot run has `E == 0`, so this is a no-op unless a retained lane already exists in the process. | off |
| `--no-thinking` | disable thinking in prompt rendering | thinking on |
| `--reasoning-effort low\|medium\|xhigh` | select an effort exposed by the loaded chat template | template default |
| `--greedy` | exact argmax decoding | off |
| `--no-p-less-sampling` | opt out of p-less and use top-p/top-k/min-p/penalties | p-less on |
| `--temperature F` | sampling temperature override | registered model/mode default |
| `--top-p F` | nucleus-threshold override | registered model/mode default |
| `--top-k N` | top-k-threshold override | registered model/mode default |
| `--min-p F` | min-p-threshold override | registered model/mode default |
| `--presence-penalty F` | presence-penalty override | registered model/mode default |
| `--frequency-penalty F` | frequency-penalty override | registered model/mode default (`0`) |
| `--seed N` | sampling seed | `0` |

When a sampling flag is omitted, Engine selects the official general-task preset registered for
the loaded model and the rendered prompt mode. The current presets are:

| Model | Prompt mode | Temperature | Top-p | Top-k | Min-p | Presence penalty |
|---|---|---:|---:|---:|---:|---:|
| Qwen3.6-27B | thinking | `1.0` | `0.95` | `20` | `0` | `0` |
| Qwen3.6-27B | non-thinking | `0.7` | `0.80` | `20` | `0` | `1.5` |
| Qwen3.8-27B | thinking | `2.0` | `0.95` | `20` | `0` | `0` |
| Qwen3.8-27B | non-thinking | `2.0` | `0.80` | `20` | `0` | `0` |
| Qwen3.6-35B-A3B | thinking | `1.0` | `0.95` | `20` | `0` | `1.5` |
| Qwen3.6-35B-A3B | non-thinking | `0.7` | `0.80` | `20` | `0` | `1.5` |

Frequency penalty is `0` in every registered preset. Qwen's separate precise-coding recommendation
is task-specific and is therefore an explicit override rather than an inferred Engine default.

P-less is the default process/request truncation mode, with temperature `2.0` for Qwen3.8. It keeps
temperature and seed, ignores top-p, top-k, min-p, and presence/frequency penalties, and writes a
one-time warning on stderr. Ignored parameters must still satisfy their normal input ranges.
`--no-p-less-sampling` opts into the registered production sampler. Combined with `--greedy`,
p-less remains exact argmax. During thinking, p-less also exits a generated token-id
square: the least period p in [32, 2048] such that the last 2p generated ids match with
Hamming distance at most 2p/512 (so p<256 is exact identity). The continuation is excluded
from the already-computed typical set V (renormalized V without that atom, or the in-domain
runner-up when V is that singleton). This is not a `suppressed_tokens` member and does not
rebuild L. It does not detect duplicate tool calls across requests and does not alter tool-call
content. There is no CLI flag. P-less membership is `p_v ≥ max(L·exp(-2ε/T), 1/M)` with
`ε = 1/16` (first-order softmax perturbation of the logits) and `M = 1024`; L is the
unperturbed collision probability, and an empty set falls back to the eligible mode.
Under MTP or DFlash2,
p-less applies at every hop (block verification over the chain with the recorded draft `q`;
DFlash2 drafts are sampled at the calibrated or pinned `--dflash-p-less-draft-temperature`; MTP
drafts are one-hot) and to the bonus after a full
accept. The cycle exclusion applies only to the first hop's next-token decision; later hops use
their unmodified p-less candidate sets. Temperature zero remains greedy at every hop.
The reasoning terminator (including split-token forms) and model stop tokens are never
cycle exclusions. This policy does not impose a maximum reasoning length.

With declared tools, the answer's free-text grammar excludes orphan `</invoke>`,
`</parameter>`, `</function>`, and `</tool_call>` strings outside valid call envelopes.
Actual call framing and literal XML inside schema-valid arguments remain allowed.
Reasoning excludes `<tool_call>` so a real call must follow `</think>`; ordinary reasoning
and tools-off/raw output remain allowed. This is a sampling-domain constraint,
not response-text deletion or an automatic retry.

With current declared tools, Engine separately withholds suspected duplicate-tool loops
using repeated reasoning, identical calls and unchanged associated results. It can rebuild
the internal context and retry at most twice, within the original output budget and resource
reservation. Rejected calls are not printed or executed; already printed reasoning/prose is
not retracted. Exhaustion is an explicit request error, not forced EOS or an engine shutdown.
For text-only thinking requests, persistent reasoning can also trigger an internal retry:
three non-overlapping occurrences of the same exact 256-token reasoning passage must
appear in the current generated attempt, and repeated passages must cover at least 4,096
distinct redundant tokens. Overlapping windows count those tokens only once. This is a
repetition-evidence threshold, not a 4,096-token reasoning limit. The occurrences need not
have a fixed separation,
so changing words elsewhere in a multi-paragraph loop does not hide the repeated passage.
Hashes locate candidates; exact token comparison confirms them. Two copies alone do not
trigger a retry. Long reasoning without that repetition is not limited. The retry keeps
the cached prompt, including historical reasoning. It closes the open think turn and
appends the notice or rejected-call feedback. Only the failed generation is omitted. A
later retry appends another notice after the same close; the earlier notice stays. A
ready checkpoint that is a prefix is trimmed and the suffix is prefilled. Original user
content and actual tool results stay in that prompt, and an explicitly labeled engine
system notice asks for concrete progress. No call or tool result is invented. Reasoning
and duplicate-tool recovery share the two-retry budget.
Raw output and media inputs do not use these internal retries. See the serving reference
for the detector's conservative scope and recovery usage fields.

Repeat `--stop-token-id`, `--stop`, or `--reasoning-stop` to add stop conditions. Use
`--raw-output` to expose the frontend's raw output stream and `--print-token-ids` to include
generated token IDs in diagnostics. During structured Qwen output, registered model stop tokens
are excluded from sampling while reasoning is open and after the reasoning terminator until
non-whitespace answer content begins. This prevents a response from ending inside reasoning or
with an empty post-reasoning answer. For tools-enabled prompts, they are also excluded while a
`<tool_call>` opener is ambiguous or a tool call is incomplete, and become eligible after the
matching `</tool_call>`. A speculative round sampled with those tokens excluded commits only
through the first completing `</tool_call>`; the next round can then stop. Other caller-added stop
conditions remain active, and raw output does not apply this structured-output guard.

Run `./build/apps/ninfer --help` for the exact option contract.

## Context and memory

The registered model IDs have a native context limit of 262,144 tokens. The practical
allocation on one RTX 5090 depends on the selected artifact, media workload, output budget, and
KV-cache type.
Default KV storage is NVFP4. Use `--kv-dtype bf16` for uncompressed KV; `--kv-dtype int8`
remains a capacity alternative. The prepared prompt must fit
`--max-context`; generation stops at the remaining context capacity when necessary.
`--kv-capacity N` controls the shared physical Main Text KV pool independently and is rounded up to
the 64-token page size. `--kv-capacity auto` loads the selected weights, measures the remaining GPU
memory, and directly chooses the largest legal page capacity for the complete enabled runtime
layout. This includes the selected speculative backend, fixed sequence state, workspace, Vision
request transient, and CUDA Graph allowance, while leaving `--kv-capacity-headroom` MiB (default 64)
unallocated. The Engine allocates all device memory at startup, so the default only covers
driver-side growth such as lazy per-kernel local memory; raise it when a desktop or another process
uses the same GPU. A startup failure in automatic mode names this option. It does not probe allocations or resize the pool at request time. The single-request
CLI normally leaves the option omitted so it follows
`--max-context`; the distinction matters primarily to a concurrent Engine or server.
The startup log's `slack` includes the automatic `headroom`; they are not separate deductions.
`graphs` reports measured GPU usage / planned allowance; the measurement is a device-wide free-memory
delta, so other processes on the GPU can move it, and it does not fail startup. Ordinary, MTP, and DFlash2 budget
`min(12n, 24+6n)` MiB for `n` `(draft length, batch size, topology)` executables (144 MiB for
adaptive lengths 1/2/3/4/5 and four-way concurrency). All graphs are prepared at startup; changing adaptive draft length or
processing a full prefill chunk uses the already reserved runtime storage.
Checkpoint images live in one pinned host slab that startup allocates, prefaults, and registers
outside `--kv-ram-capacity`: each lane's turn (rewrite) checkpoint image of the GDN state
(146.8 MiB on Qwen3.8-27B, plus 40 MiB of DFlash local K/V), and with MTP or DFlash a pool of
context-checkpoint heads (the same image plus the hidden row) for prefill-ladder and
turn-rollback heads, including heads restored from the RAM or disk tier. The pool holds one
rollback head per lane plus the most ladder heads all lanes can hold at once: marks above
`--max-context` are unreachable, and a lane holding k ladder heads keeps at least the k-th mark's
tokens of the shared KV capacity. With the default marks, C=3 at a 262144-token max context and
at least 454656 KV tokens gives 21 heads (3.0 GiB MTP, 3.8 GiB DFlash, plus 0.4/0.5 GiB of
rewrite images); C=6 with every lane able to reach every mark doubles that to 42 heads
(6.0/7.7 GiB), while a 262144-token KV capacity bounds C=6 to 20 heads. Serve prints the slab as
`ckpt-pin=` and `ckpt-heads=` on the KV capacity line and the CLI as `checkpoint host pinned`.
Serving never allocates or frees pinned checkpoint memory; while every head is owned, an optional
capture or restored head is skipped. Startup fails with the required size when the slab would leave
less than 4096 MiB of the host's available memory; `--context-checkpoints off` shrinks the pool to
one turn-rollback head per lane, and without MTP or DFlash there is no pool. With MTP or
DFlash, capture copies the state into the Engine-wide device staging slot (a device-to-device copy) and drains
it to the host image on the copy stream behind later work; a restore while staging still holds that
image copies it back on the device, otherwise it costs one H2D before the suffix prefill.
`--kv-ram-capacity N` is a separate pinned-host budget in MiB for completed prefix bundles. It is
not a token capacity, does not enlarge the GPU pool, and defaults to `off`. `N` must be a positive
decimal integer; `0` is rejected. Construction fails if the host pin cannot be allocated.
`--kv-disk-capacity N` is a third-tier SSD budget in MiB of unique object bytes. It requires
`--kv-ram-capacity > 0` and `--kv-disk-location PATH`. Location without capacity is an error.
`--kv-disk-compress zstd` compresses new GDN/hidden/cyclic blobs only; KV pages are never
application-compressed. Disk is inclusive: a VRAM or RAM hit does not delete the committed SSD
generation. Equal reuse prefers VRAM, then RAM, then disk.
Disk format v6 fingerprints canonical logical KV pages rather than the current GPU pool capacity,
so one location can reopen across `--vision` on/off and automatic resident-capacity changes. Media
content remains part of each entry identity. The fingerprint also binds the opened artifact's
local file generation (device, inode, byte length, nanosecond modification/change times), not just
its shared model/weights names. Replaced or modified artifacts require a fresh cache directory;
even a byte-identical copy is conservatively a different file. Artifacts must remain immutable
while an Engine is using them. Pre-v6 locations fail startup: select a new directory and retain
the old one until its contents are no longer needed. KV dtype, speculative backend, and
persistent-state incompatibilities also fail startup.
Orderly Engine shutdown copies active chats into the host cache, saves cache entries that are not
yet on SSD, and finishes outstanding disk writes. When disk is enabled, the same stderr progress
renderer prints `kv-disk` `copy active chats` / `save cache entries` / `finish disk writes`
counts as shutdown runs.
Host RAM is an exclusive FIFO: a bundle lives in VRAM or in this budget, not both. One long MTP
or DFlash bundle with five context-checkpoint heads is about 6 GiB (Main+backend KV plus GDN and
DFlash cyclic heads); size the
budget accordingly. `off` still captures live-lane GDN into the startup checkpoint pool so
same-lane rollback works; other-lane restore after eviction remains a miss. Startup still
prints capacity plus `used`/`entries`. Serve `[req] done` and throughput lines print live
host-resident `kv-ram=` used bytes plus `n=` / `restores=` / `evicts=` / `drops=` / `save=` /
`load=`. When disk is enabled the same lines also print `kv-disk=` occupancy and counters. `kv-ram=` / `n=` exclude a chat after consume following a restore onto a KV lane; a later
spill recaptures it as a new FIFO tail. RAM `save=` / `load=` are CUDA event elapsed for that request's
RAM-tier D2H capture and H2D unpack of the FIFO bundle (Main+backend KV, rewrite GDN, captured
ladder GDN/cyclic images, and the restored matched head in the same copy span; restored ladder
heads installed for later reuse copy outside it). Admission completes only after its copies land,
so it bills them itself; a capture that a deferred or failed admission rolled back counts only
toward the lifetime totals, never toward another request. Disk `save=` is spill-session wall harvested onto the
request; disk `load=` is the host wall from the first live SSD read of that
restore until the last page or state object has arrived in the pinned host window (not H2D, and not a
sum of overlapped SSD and copy clocks). Disk `h2d=` is the host wall from that last host arrival until
the restore's page and state H2D complete (extra copy time after SSD is idle, not the overlapping
first-to-last copy span). They are not admission wait, and they do not include live-lane
context-checkpoint freeze D2H or a VRAM-resident restore that unpacks already-pinned lane GDN.
`restores=` / `evicts=` / `drops=` are lifetime counters on both lines.
CLI `KV RAM events` prints lifetime captures/restores/evicts/drops plus that request's `save=` /
`load=`. CLI `KV disk events` also prints `h2d=` (post-disk H2D wall). The generation summary also prints `prefix reuse path`, `prefix reuse source`, and
`context checkpoint` (`restored:F` / `captured:F` absolute ladder or turn-rollback head frontiers). Exact-hit `--capture-context-checkpoint` uses the same `captured:F` field. Exact byte values remain on the Engine API and in the JSONL request log; set
`NINFER_KV_RAM_LOG_BYTES=1` to print those same byte counts on the human lines. A new capture may
still need to reap or evict while logged occupancy looks low, because a just-consumed copy can
occupy the pin until its CUDA event completes.

At Engine startup NInfer reserves model weights, persistent sequence state, one phase-reused
Program scratch arena, the maximum Vision request-transient buffer when Vision is enabled, and a
separate CUDA Graph driver allowance. Scratch is the maximum of the enabled Text, MTP, DFlash, and
Vision phases, not their sum. Its prefill bound uses
`min(--prefill-chunk,--max-context)`. The request-transient buffer is also frozen at startup; a
media request activates only the needed prefix and performs no project-owned device allocation or
growth.

All weight, sequence, workspace, request-transient, and graph allocations are released when the
Engine is destroyed.
