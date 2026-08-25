#!/usr/bin/env bash
# Sage / sparge PPL cells (RTX 5090, qwen3_8_27b/nvfp4, 8k prefill/decode, skip half).
# Run inside the nvfp4-ppl-bench container:
#   docker exec nvfp4-ppl-bench bash /tmp/run_sage_cells.sh
# Raw PPL output per cell goes to /tmp/cells/<label>.log + <label>.json;
# bring the results back for analysis (no in-script math).
#
# PREFILL lane: tile-skip (keep_frac<1) engages on the Prompt route, so the
# sparsity quality cost is measured here.
# DECODE lane: scored tokens are T=1 SmallT steps (exact attention); sparge is
# skipped for decode by design — only codec (bf16 vs nvfp4 KV) + exact-sage cells.
set -euo pipefail

WEIGHTS=/models/qwen3_8_27b_nvfp4.ninfer
IDS=/tmp/corpus.ids

run_cell() {
  local label=$1 scheme=$2 kv_dtype=$3; shift 3
  local out=/tmp/cells/$label.log
  mkdir -p /tmp/cells
  echo "=== $label  ($scheme / $kv_dtype / $*)"
  echo "    raw output -> $out"
  /usr/local/bin/ninfer-ppl \
    --weights "$WEIGHTS" \
    --ids "$IDS" \
    --scheme "$scheme" \
    --kv-dtype "$kv_dtype" \
    --tokens 8192 \
    --skip half \
    --out-json "/tmp/cells/$label.json" \
    "$@" | tee "$out"
}

echo
echo "=== PREFILL lane (sparsity engages)"
# Baseline codec (reference for all deltas).
run_cell kv-bf16-baseline kv-bf16 bf16 --schedule prefill
# NVFP4 KV codec, exact attention, NO sage recipe — separates the KV quantization
# cost from the sage FP4-PV recipe cost (the sage rows below stack on this).
run_cell kv-nvfp4 kv-nvfp4 nvfp4 --schedule prefill
# Sage recipe, exact attention (keep_frac defaults to 1.0; no tile-skip).
run_cell attn-sage attn-sage nvfp4 --sage --schedule prefill
# Control: explicit --keep-frac 1.0 must match kv-nvfp4 exactly (dense).
run_cell attn-dense-keepfrac1.0 kv-nvfp4 nvfp4 --keep-frac 1.0 --schedule prefill
# Tile-skip on exact NVFP4 (no --sage).
run_cell attn-topk-keepfrac50 attn-topk nvfp4 --keep-frac 0.5 --schedule prefill
run_cell attn-topk-keepfrac20 attn-topk nvfp4 --keep-frac 0.2 --schedule prefill
run_cell attn-xattn-tau90 attn-xattn nvfp4 --xattn-tau 0.9 --schedule prefill

echo
echo "=== DECODE lane (sparsity skipped: codec + exact-sage only)"
# Codec decode cells: same codecs, T=1 teacher-forced (exact attention both dtypes).
run_cell kv-bf16-baseline-decode kv-bf16 bf16 --schedule decode
run_cell kv-nvfp4-decode kv-nvfp4 nvfp4 --schedule decode
# Sage recipe at keep_frac=1.0 (exact): isolates the sage PV recipe cost in decode.
run_cell attn-sage-decode attn-sage nvfp4 --sage --schedule decode

echo
echo "All cells done:"
ls /tmp/cells/*.json