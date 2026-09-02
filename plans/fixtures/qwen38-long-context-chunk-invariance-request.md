# Qwen3.8 long-context premature-EOS reproducer

`qwen38-long-context-chunk-invariance-request.json` is a directly submit-able OpenAI Chat
Completions request reconstructed from the affected Open WebUI conversation. It contains 108
messages and four tool schemas, renders to 101,708 prompt tokens with the recorded artifact, and
uses seed 12345. Its SHA-256 is:

```text
fbe71d99098cacd59f2590c19f8d4022497c581df17046582bc68cb6d0c64e98
```

Affected configuration:

- `qwen3.8-27b/nvfp4`, NVFP4 KV, prefill chunk 4096
- RTX 5090, startup concurrency 2
- DFlash2, five draft tokens, optimized draft LM head
- artifact `qwen3_8_27b_nvfp4_dflash_nvfp4.ninfer`

Before the fix, the request selected the registered EOS token after 38 completion tokens while the
reasoning block was still open, yielding reasoning text and an empty answer. The same failure class
was observed after only 3 and 5 tokens in the original Open WebUI conversation, on both fresh
prefill and response-checkpoint reuse.

The runtime fix excludes the registered model EOS IDs from greedy and stochastic selection while
the structured Qwen output session is in reasoning. This applies to ordinary and speculative
selection. After the accepted round containing `</think>` commits, the exclusions are cleared for
the next GPU round, allowing the answer to select EOS normally. Other stop conditions, raw output,
and non-thinking requests are unchanged.

With the fix, this fixture no longer selects EOS at token 38. At a 512-token output budget it closes
reasoning at token 85, emits a normal answer, and selects EOS at token 109. Fresh prefill,
response-checkpoint reuse, base/no-spec, and DFlash startup capacities C=1/2/3/4 were qualified.

The underlying model trajectory remains sensitive to legal long-prompt chunk boundaries; the EOS
guard prevents the invalid structured-output termination but does not make model calculations
chunk-invariant.

Submit the unchanged JSON fixture to a running server:

```bash
curl --silent --show-error \
  -H 'Content-Type: application/json' \
  --data-binary @plans/fixtures/qwen38-long-context-chunk-invariance-request.json \
  http://127.0.0.1:8002/v1/chat/completions
```
