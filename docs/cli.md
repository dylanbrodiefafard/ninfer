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
- `--spec mtp` loads only MTP, while `--spec dflash` loads only the text-only DFlash backend
  (35B-A3B DFlash v1, or Qwen3.8-27B DFlash2 when `dflash/` is present);
- a speculative backend with the full proposal head omits the optimized proposal head;
- Vision is disabled by default, omitting its weights, Vision scratch phase, and frozen
  request-transient allocation;
- `--vision` loads those allocations and enables image/video input.

The complete `.ninfer` inventory is still validated. These choices are not lazy loading: a
text-only Engine rejects media and cannot enable Vision later. DFlash and Vision are mutually
exclusive. The default speculative and Vision settings produce the smallest resident profile.

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

See [`examples/cli/`](../examples/cli/) for committed text, image, video, mixed-media, thinking,
long-decode, and long-context inputs.

## Speculative decoding

Speculative decoding is disabled by default. Select MTP with one to five draft positions, 35B-A3B
text-only DFlash v1 with one to fifteen, or Qwen3.8-27B NVFP4 text-only DFlash2 with one to seven.
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

For Qwen3.8-27B DFlash2, the NVFP4 artifact must contain the appended `dflash/` objects. Its
paper-accurate verifier is a chain with `W=k+1`; a width override, when supplied, must equal that
value. On RTX 5090, `k=4` (block length five) is the measured speed recommendation.

```bash
./build/apps/ninfer out/qwen3_8_27b_nvfp4_dflash_w8.ninfer \
  --prompt "Write a short explanation of speculative decoding." \
  --max-context 16384 --max-new 512 \
  --spec dflash --draft-tokens 4 --lm-head-draft
```

MTP and DFlash cannot be enabled together. `--spec dflash` on a 27B file without `dflash/` fails
at bind. Current 3.8 MTP-only files and all 3.6-27B files stay valid MTP artifacts. The published
[performance results](performance.md) use MTP with three draft tokens and DFlash with seven draft
tokens (block length eight), both with the optimized proposal head. 35B DFlash v1 accepts up to
fifteen draft tokens; 3.8 DFlash2 accepts up to eleven. Eleven runs Spark two-block (7 MASK + 4
MASK) chain verify. Published INT8-KV C=1 DFlash2 W8 numbers are in
[performance.md](performance.md). The RTX 5090 packed-tree investigation and chain cutover are in
[dflash2-tree-speed.md](maintainer/dflash2-tree-speed.md).

## Common options

| Option | Meaning | Default |
|---|---|---:|
| `--max-context N` | per-sequence logical context ceiling | `2048` |
| `--kv-capacity N\|auto` | explicit shared Main Text KV capacity, or maximize it from remaining GPU memory; omitted means `--max-context` | `2048` |
| `--kv-ram-capacity off\|N` | pinned host KV prefix-cache capacity in MiB; `off` disables the tier | `off` |
| `--kv-disk-capacity off\|N` | SSD KV prefix-cache unique-object capacity in MiB; `off` disables the tier | `off` |
| `--kv-disk-location PATH` | directory for the SSD page store; required iff `--kv-disk-capacity` is enabled | unset |
| `--kv-disk-compress off\|zstd` | zstd-1 on new GDN/hidden/cyclic writes; KV pages stay uncompressed | `off` |
| `--prefill-chunk N` | positive text-prefill chunk, in multiples of 128 | `4096` |
| `--max-new N` | requested output-token limit | `128` |
| `--device N` | CUDA device index | `0` |
| `--kv-dtype bf16\|int8\|nvfp4` | KV-cache storage | `nvfp4` |
| `--spec mtp\|dflash` | speculative backend | off |
| `--draft-tokens N` | MTP `1..5`; 35B DFlash `1..15`; 3.8 DFlash2 `1..11` | unset |
| `--adaptive-draft` | pick live draft K from host EWMA; requires `--spec mtp\|dflash` | off |
| `--dflash-verify-width N` | DFlash verify width `2..16`; chain-only targets require `W=k+1`. Qwen3.8 DFlash2 defaults to chain `W=k+1` for `k<=5` and packed-tree `W=12` for `k` in `{6,7}` | auto |
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
p-less remains exact argmax.

Repeat `--stop-token-id`, `--stop`, or `--reasoning-stop` to add stop conditions. Use
`--raw-output` to expose the frontend's raw output stream and `--print-token-ids` to include
generated token IDs in diagnostics. During structured Qwen output, registered model stop tokens
are excluded from sampling while reasoning is open and after the reasoning terminator until
non-whitespace answer content begins. This prevents a response from ending inside reasoning or
with an empty post-reasoning answer. For tools-enabled prompts, they are also excluded while a
`<tool_call>` opener is ambiguous or a tool call is incomplete, and become eligible after the
matching `</tool_call>`. Other caller-added stop conditions remain active, and raw output does not
apply this structured-output guard.

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
request transient, and CUDA Graph allowance, while leaving the default 1 GiB automatic headroom
unallocated. It does not probe allocations or resize the pool at request time. The single-request
CLI normally leaves the option omitted so it follows
`--max-context`; the distinction matters primarily to a concurrent Engine or server.
`--kv-ram-capacity N` is a separate pinned-host budget in MiB for completed prefix bundles. It is
not a token capacity, does not enlarge the GPU pool, and defaults to `off`. `N` must be a positive
decimal integer; `0` is rejected. Construction fails if the host pin cannot be allocated.
`--kv-disk-capacity N` is a third-tier SSD budget in MiB of unique object bytes. It requires
`--kv-ram-capacity > 0` and `--kv-disk-location PATH`. Location without capacity is an error.
`--kv-disk-compress zstd` compresses new GDN/hidden/cyclic blobs only; KV pages are never
application-compressed. Disk is inclusive: a VRAM or RAM hit does not delete the committed SSD
generation. Equal reuse prefers VRAM, then RAM, then disk.
Orderly Engine shutdown copies active chats into the host cache, saves cache entries that are not
yet on SSD, and finishes outstanding disk writes. When disk is enabled, the same stderr progress
renderer prints `kv-disk` `copy active chats` / `save cache entries` / `finish disk writes`
counts as shutdown runs.
Host RAM is an exclusive FIFO: a bundle lives in VRAM or in this budget, not both. One long MTP
or DFlash bundle with five context-checkpoint heads is about 6 GiB (Main+backend KV plus GDN and
DFlash cyclic heads); size the
budget accordingly. `off` still captures live-lane GDN to ordinary pinned buffers so same-lane
rollback works; other-lane restore after eviction remains a miss. Startup still
prints capacity plus `used`/`entries`. Serve `[req] done` and throughput lines print live
host-resident `kv-ram=` used bytes plus `n=` / `restores=` / `evicts=` / `drops=` / `save=` /
`load=`. When disk is enabled the same lines also print `kv-disk=` occupancy and counters. `kv-ram=` / `n=` exclude a chat after consume following a restore onto a KV lane; a later
spill recaptures it as a new FIFO tail. RAM `save=` / `load=` are CUDA event elapsed for that request's
RAM-tier D2H capture and H2D unpack of the FIFO bundle (Main+backend KV, rewrite GDN, and any ladder
GDN/cyclic images in the same copy span). Disk `save=` is spill-session wall harvested onto the
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
