#!/usr/bin/env bash
# Manual A/B entry point. Each immutable artifact uses its own v6 disk-cache directory.
set -euo pipefail
variant="${1:?usage: bash tools/debug/serve_selective_fp8.sh base|328 [serve options]}"
shift
repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
cd -- "$repo_root"
case "$variant" in
  base)
    artifact="${NINFER_QWEN3_8_27B_NVFP4_DFLASH_WEIGHTS:-$repo_root/models/qwen3_8_27b_nvfp4_dflash_nvfp4.ninfer}" ;;
  328)
    artifact="${NINFER_FP8_328_WEIGHTS:-$repo_root/models/qwen3_8_27b_nvfp4_fp8_328mib_dflash.ninfer}" ;;
  *) echo "unknown campaign variant: $variant" >&2; exit 2 ;;
esac
[[ -f "$artifact" && -x ./build/apps/ninfer-serve ]] || {
  echo 'Build ninfer-serve and retain the selected campaign artifact first.' >&2
  exit 1
}
mkdir -p out/selective-fp8-campaign/manual
./build/apps/ninfer-serve "$artifact" \
  --host 0.0.0.0 --port 18002 --model-id 3.6-27b \
  --max-context 260000 --kv-capacity auto --max-concurrency 2 \
  --preserve-thinking --kv-dtype nvfp4 --spec dflash --draft-tokens 5 \
  --lm-head-draft --temperature 1.5 --prefill-chunk 4096 \
  --pending-timeout-ms 900000 --default-max-tokens 32768 \
  --kv-ram-capacity 32768 --kv-disk-capacity 100000 \
  --kv-disk-location "${NINFER_KV_DISK_LOCATION:-$repo_root/out/cache/selective-fp8-$variant}" \
  --vision "$@" 2>&1 | tee -a "out/selective-fp8-campaign/manual/$variant-server.log"
