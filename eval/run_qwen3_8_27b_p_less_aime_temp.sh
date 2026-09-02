#!/usr/bin/env bash
set -euo pipefail

# Sequential production vs --p-less-sampling AIME25/AIME26 temperature sweep.
# Same client generation fields both times; the second process ignores truncation
# and penalties. Do not merge the two methods into one Engine.

repo_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
config="${repo_dir}/eval/configs/qwen3_8_27b_p_less_aime_temp.yaml"
eval_python="${repo_dir}/eval/.venv/bin/python"
compare_py="${repo_dir}/eval/compare_p_less_aime_temp.py"
log_dir="${repo_dir}/eval/server-logs"
runs_dir="${repo_dir}/eval/runs"

if [[ -x /build/apps/ninfer-serve ]]; then
    server_bin="/build/apps/ninfer-serve"
else
    server_bin="${repo_dir}/build/apps/ninfer-serve"
fi
if [[ -f /models/qwen3_8_27b_nvfp4.ninfer ]]; then
    artifact="/models/qwen3_8_27b_nvfp4.ninfer"
elif [[ -f "${repo_dir}/models/qwen3_8_27b_nvfp4.ninfer" ]]; then
    artifact="${repo_dir}/models/qwen3_8_27b_nvfp4.ninfer"
else
    artifact="${repo_dir}/out/qwen3_8_27b_nvfp4.ninfer"
fi

# Published 3.8 NVFP4 reasoning eval used 262144. C=1 + nvfp4 KV fits; fall back
# if this Engine refuses the reservation.
max_context="${NINFER_P_LESS_AIME_MAX_CONTEXT:-262144}"
fallback_context=252928

usage() {
    echo "usage: $0 [--plan]" >&2
}

plan_only=0
if [[ "${1:-}" == "--plan" ]]; then
    plan_only=1
    shift
fi
if [[ $# -ne 0 ]]; then
    usage
    exit 2
fi

if [[ "${plan_only}" -eq 1 ]]; then
    PYTHONPATH="${repo_dir}/eval" "${eval_python}" -m ninfer_eval plan \
        --config "${config}" --suite aime_temp --check-runtime
    exit 0
fi

for required_file in "${server_bin}" "${artifact}" "${config}" "${eval_python}" "${compare_py}"; do
    if [[ ! -f "${required_file}" ]]; then
        echo "missing required file: ${required_file}" >&2
        exit 1
    fi
done
if [[ ! -x "${server_bin}" || ! -x "${eval_python}" ]]; then
    echo "ninfer-serve and the evaluation Python must be executable" >&2
    exit 1
fi

health_ok() {
    "${eval_python}" -c 'import sys, urllib.request
try:
    urllib.request.urlopen("http://127.0.0.1:18080/health", timeout=2)
except Exception:
    sys.exit(1)
'
}

export LD_LIBRARY_PATH="/usr/lib/x86_64-linux-gnu:/usr/local/nvidia/lib64:/usr/local/cuda/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
cd -- "${repo_dir}"
mkdir -p -- "${log_dir}" "${runs_dir}"

server_pid=""
current_server_log=""
current_request_log=""
cleanup() {
    if [[ -n "${server_pid}" ]] && kill -0 "${server_pid}" 2>/dev/null; then
        kill -TERM "${server_pid}"
        wait "${server_pid}" || true
    fi
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

start_server() {
    local method="$1"
    local max_ctx="$2"
    local run_stamp="$3"
    local -a pless_args=()
    if [[ "${method}" == "p-less" ]]; then
        pless_args=(--p-less-sampling)
    fi

    current_server_log="${log_dir}/qwen3_8_27b_nvfp4-p-less-aime-${method}-c1-${run_stamp}.server.log"
    current_request_log="${log_dir}/qwen3_8_27b_nvfp4-p-less-aime-${method}-c1-${run_stamp}.requests.jsonl"

    if health_ok; then
        echo "port 18080 already has a healthy service; stop it before running this script" >&2
        exit 1
    fi

    "${server_bin}" "${artifact}" \
        --host 127.0.0.1 \
        --port 18080 \
        --model-id qwen3.8-27b \
        --max-context "${max_ctx}" \
        --kv-capacity auto \
        --max-concurrency 1 \
        --max-pending-requests 1 \
        --pending-timeout-ms 86400000 \
        --prefill-chunk 1024 \
        --kv-dtype nvfp4 \
        --spec mtp \
        --draft-tokens 3 \
        --lm-head-draft \
        "${pless_args[@]}" \
        --request-log-jsonl "${current_request_log}" \
        >"${current_server_log}" 2>&1 &
    server_pid=$!

    local ready=0
    local attempt
    for ((attempt = 1; attempt <= 180; ++attempt)); do
        if health_ok; then
            ready=1
            break
        fi
        if ! kill -0 "${server_pid}" 2>/dev/null; then
            wait "${server_pid}" || true
            echo "ninfer-serve exited before becoming ready; see ${current_server_log}" >&2
            return 1
        fi
        sleep 1
    done
    if [[ "${ready}" -ne 1 ]]; then
        echo "ninfer-serve did not become ready within 180 seconds; see ${current_server_log}" >&2
        return 1
    fi
}

run_method() {
    local method="$1"
    local run_stamp
    run_stamp="$(date -u +%Y%m%dT%H%M%SZ)"
    if ! start_server "${method}" "${max_context}" "${run_stamp}"; then
        cleanup
        server_pid=""
        if [[ "${max_context}" != "${fallback_context}" ]]; then
            echo "retrying ${method} with --max-context ${fallback_context}" >&2
            max_context="${fallback_context}"
            run_stamp="$(date -u +%Y%m%dT%H%M%SZ)"
            start_server "${method}" "${max_context}" "${run_stamp}"
        else
            exit 1
        fi
    fi
    local server_log="${current_server_log}"
    local request_log="${current_request_log}"

    echo "method: ${method}"
    echo "server log: ${server_log}"
    echo "request log: ${request_log}"

    local eval_log="${log_dir}/qwen3_8_27b_nvfp4-p-less-aime-${method}-c1-${run_stamp}.eval.log"
    set +e
    PYTHONPATH="${repo_dir}/eval" "${eval_python}" -m ninfer_eval run \
        --config "${config}" --suite aime_temp 2>&1 | tee "${eval_log}"
    local eval_status="${PIPESTATUS[0]}"
    set -e
    cleanup
    server_pid=""

    local run_dir
    run_dir="$(grep -E 'run directory: ' "${eval_log}" | tail -n 1 | sed -E 's/.*run directory: //' | tr -d '\r')"
    if [[ -z "${run_dir}" || ! -d "${run_dir}" ]]; then
        echo "could not locate eval run directory for ${method}; see ${eval_log}" >&2
        exit 1
    fi
    printf '%s\n' "{
  \"method\": \"${method}\",
  \"artifact\": \"${artifact}\",
  \"max_context\": ${max_context},
  \"kv_dtype\": \"nvfp4\",
  \"spec\": \"mtp\",
  \"draft_tokens\": 3,
  \"server_log\": \"${server_log}\",
  \"request_log\": \"${request_log}\",
  \"eval_log\": \"${eval_log}\"
}" >"${run_dir}/method.json"
    printf '%s' "${run_dir}" >"${log_dir}/${method}.run_dir"
    printf '%s' "${request_log}" >"${log_dir}/${method}.request_log"
    echo "tagged run directory: ${run_dir}"
    if [[ "${eval_status}" -ne 0 ]]; then
        echo "${method} evaluation exited ${eval_status}" >&2
        exit "${eval_status}"
    fi
}

echo "artifact: ${artifact}"
echo "server: ${server_bin}"

run_method production
run_method p-less

prod_dir="$(cat "${log_dir}/production.run_dir")"
prod_req="$(cat "${log_dir}/production.request_log")"
pless_dir="$(cat "${log_dir}/p-less.run_dir")"
pless_req="$(cat "${log_dir}/p-less.request_log")"

out_dir="${runs_dir}/p-less-aime-compare-$(date -u +%Y%m%dT%H%M%SZ)"
mkdir -p -- "${out_dir}"
PYTHONPATH="${repo_dir}/eval" "${eval_python}" "${compare_py}" \
    --production "${prod_dir}" \
    --p-less "${pless_dir}" \
    --production-requests "${prod_req}" \
    --p-less-requests "${pless_req}" \
    --out "${out_dir}"
echo "comparison: ${out_dir}/summary.md"
