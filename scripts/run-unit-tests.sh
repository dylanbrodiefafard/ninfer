#!/usr/bin/env bash
# Build and run the C++ unit-test suite.
#
# Default: every CTest except opt-in real-artifact Engine tests (`*_real_test`).
# Those load a full `.ninfer` and are documented separately in tests/README.md.
# Refuses to run when the GPU has less than 20 GiB free (likely OOM against a
# resident ninfer-serve).
#
# Usage:
#   ./scripts/run-unit-tests.sh                  # full unit suite
#   ./scripts/run-unit-tests.sh -R ninfer_sampling_test
#   ./scripts/run-unit-tests.sh --real           # include real-artifact tests
#   ./scripts/run-unit-tests.sh --print-weights  # show which .ninfer files were found
#   ./scripts/run-unit-tests.sh --python         # also run the host Python pytest suite
#
# --real auto-finds exact filenames in models/, out/, /models, the builder's
# models mount, and sibling folders of that mount. Override with env vars or
# models/weights.env (KEY=VALUE). Never picks a file by glob or mtime.
#
# Prefers the ninfer-builder GPU container (./scripts/dev-setup.sh). Extra
# arguments after flags are forwarded to ctest.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BUILDER="${NINFER_DEV_CONTAINER:-ninfer-builder}"
JOBS="${NINFER_DEV_JOBS:-$(nproc)}"
MIN_FREE_GIB="${NINFER_MIN_FREE_VRAM_GIB:-20}"
MIN_FREE_MIB=$((MIN_FREE_GIB * 1024))
INNER=0
INCLUDE_REAL=0
PRINT_WEIGHTS=0
RUN_PYTHON=0
CTEST_ARGS=()

WEIGHT_VARS=(
  NINFER_QWEN3_6_27B_NVFP4_WEIGHTS
  NINFER_QWEN3_6_27B_WEIGHTS
  NINFER_QWEN3_8_27B_NVFP4_DFLASH_WEIGHTS
  NINFER_QWEN3_8_27B_NVFP4_MTP_WEIGHTS
  NINFER_QWEN3_6_35B_A3B_WEIGHTS
  NINFER_QWEN4_VERIFY_WEIGHTS
)

while [[ $# -gt 0 ]]; do
  case "$1" in
    --inner) INNER=1; shift ;;
    --real) INCLUDE_REAL=1; shift ;;
    --print-weights) PRINT_WEIGHTS=1; shift ;;
    --python) RUN_PYTHON=1; shift ;;
    --help|-h)
      sed -n '2,21p' "$0"
      exit 0
      ;;
    --) shift; CTEST_ARGS+=("$@"); break ;;
    *) CTEST_ARGS+=("$@"); break ;;
  esac
done

in_container() {
  [[ -f /.dockerenv ]] || [[ -f /run/.containerenv ]]
}

builder_running() {
  command -v docker >/dev/null 2>&1 \
    && [[ "$(docker inspect -f '{{.State.Running}}' "$BUILDER" 2>/dev/null || true)" == "true" ]]
}

container_for_pid() {
  local pid="$1"
  local cg id name
  [[ -r "/proc/${pid}/cgroup" ]] || return 0
  cg="$(cat "/proc/${pid}/cgroup" 2>/dev/null || true)"
  id="$(grep -oE 'docker-[0-9a-f]{64}' <<<"$cg" | head -1 | sed 's/^docker-//')"
  if [[ -z "$id" ]]; then
    id="$(grep -oE '[0-9a-f]{64}' <<<"$cg" | head -1 || true)"
  fi
  [[ -n "$id" ]] || return 0
  command -v docker >/dev/null 2>&1 || return 0
  name="$(docker inspect --format '{{.Name}}' "$id" 2>/dev/null || true)"
  name="${name#/}"
  [[ -n "$name" ]] && printf '%s' "$name"
}

weight_filenames() {
  case "$1" in
    NINFER_QWEN3_6_27B_NVFP4_WEIGHTS)
      printf '%s\n' qwen3_8_27b_nvfp4.ninfer qwen3_6_27b_nvfp4.ninfer
      ;;
    NINFER_QWEN3_6_27B_WEIGHTS)
      printf '%s\n' qwen3_8_27b.ninfer qwen3_6_27b.ninfer
      ;;
    NINFER_QWEN3_8_27B_NVFP4_DFLASH_WEIGHTS)
      printf '%s\n' qwen3_8_27b_nvfp4_dflash_nvfp4.ninfer \
        qwen3_8_27b_nvfp4_dflash_w8.ninfer \
        qwen3_8_27b_nvfp4_dflash_nvfp4_codebook.ninfer
      ;;
    NINFER_QWEN3_8_27B_NVFP4_MTP_WEIGHTS)
      printf '%s\n' qwen3_8_27b_nvfp4_mtp.ninfer qwen3_8_27b_nvfp4_mtp_nvfp4.ninfer
      ;;
    NINFER_QWEN3_6_35B_A3B_WEIGHTS)
      printf '%s\n' qwen3_6_35b_a3b.ninfer
      ;;
    NINFER_QWEN4_VERIFY_WEIGHTS)
      printf '%s\n' qwen4_ud_iq1_s_verify.ninfer
      ;;
  esac
}

weight_tests() {
  case "$1" in
    NINFER_QWEN3_6_27B_NVFP4_WEIGHTS)
      echo "prefix, ram, checkpoint, serve-prepend"
      ;;
    NINFER_QWEN3_6_27B_WEIGHTS)
      echo "prefix, ram, checkpoint, serve-prepend, load-plan"
      ;;
    NINFER_QWEN3_8_27B_NVFP4_DFLASH_WEIGHTS)
      echo "ninfer_qwen3_8_27b_dflash_real_test"
      ;;
    NINFER_QWEN3_8_27B_NVFP4_MTP_WEIGHTS)
      echo "ninfer_qwen3_8_27b_mtp_nvfp4_real_test"
      ;;
    NINFER_QWEN3_6_35B_A3B_WEIGHTS)
      echo "35B real, ram, dflash, load-plan"
      ;;
    NINFER_QWEN4_VERIFY_WEIGHTS)
      echo "Qwen4 verifier artifact and program"
      ;;
  esac
}

models_mount_source() {
  local mount_line src dest
  while IFS= read -r mount_line; do
    [[ -n "$mount_line" ]] || continue
    src="${mount_line%%=*}"
    dest="${mount_line#*=}"
    if [[ "$dest" == /models ]]; then
      printf '%s' "$src"
      return 0
    fi
  done < <(builder_mounts)
  return 1
}

load_weights_env_file() {
  local file="$1"
  [[ -f "$file" ]] || return 0
  echo "Weights config: $file"
  local line key val
  while IFS= read -r line || [[ -n "$line" ]]; do
    line="${line%$'\r'}"
    [[ "$line" =~ ^[[:space:]]*# ]] && continue
    [[ "$line" =~ ^[[:space:]]*$ ]] && continue
    if [[ "$line" =~ ^([A-Za-z_][A-Za-z0-9_]*)=(.*)$ ]]; then
      key="${BASH_REMATCH[1]}"
      val="${BASH_REMATCH[2]}"
      val="${val#\"}"; val="${val%\"}"
      val="${val#\'}"; val="${val%\'}"
      case "$key" in
        NINFER_*_WEIGHTS|NINFER_WEIGHTS_DIRS|NINFER_MODELS_DIR|NINFER_WEIGHTS_ENV)
          if [[ -z "${!key:-}" ]]; then
            export "$key=$val"
          fi
          ;;
      esac
    fi
  done < "$file"
}

builder_mounts() {
  builder_running || return 0
  docker inspect -f '{{range .Mounts}}{{.Source}}={{.Destination}}{{println}}{{end}}' "$BUILDER" 2>/dev/null || true
}

add_search_dir() {
  local dir="$1"
  [[ -n "$dir" && -d "$dir" ]] || return 0
  local existing
  for existing in "${SEARCH_DIRS[@]+"${SEARCH_DIRS[@]}"}"; do
    if [[ "$existing" == "$dir" ]]; then
      return 0
    fi
  done
  SEARCH_DIRS+=("$dir")
}

collect_search_dirs() {
  SEARCH_DIRS=()
  local dir mount_line src dest parent sibling
  add_search_dir "${ROOT}/models"
  add_search_dir "${ROOT}/out"
  add_search_dir /models
  add_search_dir "${NINFER_MODELS_DIR:-}"
  if [[ -n "${NINFER_WEIGHTS_DIRS:-}" ]]; then
    local IFS=':'
    for dir in $NINFER_WEIGHTS_DIRS; do
      add_search_dir "$dir"
    done
  fi
  while IFS= read -r mount_line; do
    [[ -n "$mount_line" ]] || continue
    src="${mount_line%%=*}"
    dest="${mount_line#*=}"
    add_search_dir "$src"
    if [[ "$dest" == /models ]]; then
      parent="$(dirname "$src")"
      if [[ -d "$parent" ]]; then
        for sibling in "$parent"/*; do
          add_search_dir "$sibling"
        done
      fi
    fi
  done < <(builder_mounts)
}

find_named_file() {
  local name="$1"
  local dir
  for dir in "${SEARCH_DIRS[@]+"${SEARCH_DIRS[@]}"}"; do
    if [[ -f "${dir}/${name}" ]]; then
      realpath -m "${dir}/${name}"
      return 0
    fi
  done
  return 1
}

container_path_for() {
  local host="$1"
  [[ -n "$host" ]] || return 1
  if [[ "$host" == /src/* || "$host" == /models/* || "$host" == /build/* ]]; then
    printf '%s' "$host"
    return 0
  fi
  if [[ "$host" == "${ROOT}/"* ]]; then
    printf '/src/%s' "${host#"${ROOT}/"}"
    return 0
  fi
  local mount_line src dest
  local best_src="" best_dest=""
  while IFS= read -r mount_line; do
    [[ -n "$mount_line" ]] || continue
    src="${mount_line%%=*}"
    dest="${mount_line#*=}"
    if [[ "$host" == "${src}" || "$host" == "${src}/"* ]]; then
      if [[ ${#src} -gt ${#best_src} ]]; then
        best_src="$src"
        best_dest="$dest"
      fi
    fi
  done < <(builder_mounts)
  if [[ -n "$best_src" ]]; then
    if [[ "$host" == "$best_src" ]]; then
      printf '%s' "$best_dest"
    else
      printf '%s/%s' "$best_dest" "${host#"${best_src}/"}"
    fi
    return 0
  fi
  return 1
}

configure_weights() {
  load_weights_env_file "${NINFER_WEIGHTS_ENV:-}"
  load_weights_env_file "${ROOT}/models/weights.env"
  load_weights_env_file "${ROOT}/weights.env"
  collect_search_dirs

  WEIGHT_SOURCE=()
  WEIGHT_MISSING=()
  WEIGHT_UNMAPPED=()
  local var names name path mapped source
  for var in "${WEIGHT_VARS[@]}"; do
    path="${!var:-}"
    source=""
    if [[ -n "$path" ]]; then
      path="$(realpath -m "$path")"
      if [[ ! -f "$path" ]]; then
        local container_try="$path"
        if builder_running; then
          container_try="$(container_path_for "$path" || printf '%s' "$path")"
        fi
        if ! builder_running || ! docker exec "$BUILDER" test -f "$container_try"; then
          echo "Configured ${var} is not a file: $path" >&2
          echo "Fix the env var or models/weights.env and re-run." >&2
          exit 1
        fi
      fi
      source="env"
    else
      while IFS= read -r name; do
        [[ -n "$name" ]] || continue
        if path="$(find_named_file "$name")"; then
          source="found ${name}"
          export "${var}=${path}"
          break
        fi
      done < <(weight_filenames "$var")
    fi
    if [[ -z "${!var:-}" ]]; then
      WEIGHT_MISSING+=("$var")
      continue
    fi
    path="${!var}"
    if [[ "$INNER" -eq 0 ]] && builder_running; then
      if mapped="$(container_path_for "$path")"; then
        if [[ "$mapped" != "$path" ]]; then
          export "${var}=${mapped}"
        fi
        WEIGHT_SOURCE+=("${var}|${!var}|${source}")
      else
        WEIGHT_UNMAPPED+=("${var}|${path}|${source}")
        unset "$var"
      fi
    else
      WEIGHT_SOURCE+=("${var}|${path}|${source}")
    fi
  done
}

print_weights_report() {
  local entry var path source tests
  echo "=== real-artifact weights ==="
  if [[ ${#WEIGHT_SOURCE[@]} -gt 0 ]]; then
    for entry in "${WEIGHT_SOURCE[@]}"; do
      var="${entry%%|*}"
      rest="${entry#*|}"
      path="${rest%%|*}"
      source="${rest#*|}"
      echo "  ${var}"
      echo "    ${path}  (${source})"
    done
  fi
  if [[ ${#WEIGHT_UNMAPPED[@]} -gt 0 ]]; then
    echo
    echo "Found on the host, but not visible in ${BUILDER}:"
    local mount_src
    mount_src="$(models_mount_source || true)"
    for entry in "${WEIGHT_UNMAPPED[@]}"; do
      var="${entry%%|*}"
      rest="${entry#*|}"
      path="${rest%%|*}"
      echo "  ${var}=${path}"
      if [[ -n "$mount_src" ]]; then
        echo "    ln -s ${path} ${mount_src}/$(basename "$path")"
      fi
    done
    if [[ -z "$mount_src" ]]; then
      echo "  Recreate the builder with that folder mounted:"
      echo "    docker rm -f ${BUILDER}"
      echo "    NINFER_MODELS_DIR=/path/containing/the/file ./scripts/dev-setup.sh"
    fi
  fi
  if [[ ${#WEIGHT_MISSING[@]} -gt 0 ]]; then
    echo
    echo "Not found (those real tests will SKIP):"
    for var in "${WEIGHT_MISSING[@]}"; do
      echo "  ${var}  ($(weight_tests "$var"))"
      echo "    looked for: $(weight_filenames "$var" | paste -sd', ' - | sed 's/,/, /g')"
    done
    echo
    echo "To point at a file, create ${ROOT}/models/weights.env with:"
    for var in "${WEIGHT_MISSING[@]}"; do
      echo "  ${var}=/absolute/path.ninfer"
    done
    echo "Exact names also work in models/, out/, /models, or NINFER_MODELS_DIR."
  fi
}

require_real_weights() {
  if [[ -n "${NINFER_QWEN3_6_27B_NVFP4_WEIGHTS:-}" \
     || -n "${NINFER_QWEN3_6_27B_WEIGHTS:-}" \
     || -n "${NINFER_QWEN3_8_27B_NVFP4_DFLASH_WEIGHTS:-}" \
     || -n "${NINFER_QWEN4_VERIFY_WEIGHTS:-}" ]]; then
    return 0
  fi
  echo "Cannot run --real: no supported .ninfer artifact is visible." >&2
  echo "Place an exact supported artifact filename in models/ (or the builder /models mount)," >&2
  echo "or set its NINFER_*_WEIGHTS variable in models/weights.env." >&2
  exit 1
}

docker_weight_env_args() {
  local var
  for var in "${WEIGHT_VARS[@]}"; do
    if [[ -n "${!var:-}" ]]; then
      printf -- '-e\n%s=%s\n' "$var" "${!var}"
    fi
  done
}

print_gpu_consumers() {
  local line pid name mem cont
  local apps
  apps="$(nvidia-smi --query-compute-apps=pid,process_name,used_gpu_memory \
    --format=csv,noheader,nounits 2>/dev/null || true)"
  if [[ -z "${apps// /}" ]]; then
    echo "  (nvidia-smi reports no compute processes)" >&2
    return 0
  fi
  printf '  %-8s %-8s %s\n' "PID" "MiB" "PROCESS" >&2
  while IFS= read -r line; do
    [[ -n "$line" ]] || continue
    pid="$(cut -d, -f1 <<<"$line" | tr -d ' ')"
    name="$(cut -d, -f2 <<<"$line" | sed 's/^[[:space:]]*//;s/[[:space:]]*$//')"
    mem="$(cut -d, -f3 <<<"$line" | tr -d ' ')"
    printf '%s\t%s\t%s\n' "$mem" "$pid" "$name"
  done <<<"$apps" | sort -nr | while IFS=$'\t' read -r mem pid name; do
    cont="$(container_for_pid "$pid" || true)"
    if [[ -n "$cont" ]]; then
      printf '  %-8s %-8s %s  (docker stop %s)\n' "$pid" "$mem" "$name" "$cont" >&2
    else
      printf '  %-8s %-8s %s\n' "$pid" "$mem" "$name" >&2
    fi
  done
}

require_free_vram() {
  if [[ "${NINFER_VRAM_CHECKED:-0}" == "1" ]]; then
    return 0
  fi
  if ! command -v nvidia-smi >/dev/null 2>&1; then
    echo "Cannot run tests: nvidia-smi is not available (no GPU visibility)." >&2
    exit 1
  fi
  local total used free
  IFS=',' read -r total used free < <(
    nvidia-smi --query-gpu=memory.total,memory.used,memory.free \
      --format=csv,noheader,nounits | head -1
  )
  total="${total// /}"
  used="${used// /}"
  free="${free// /}"
  if [[ -z "$free" || ! "$free" =~ ^[0-9]+$ ]]; then
    echo "Cannot run tests: failed to read GPU memory from nvidia-smi." >&2
    nvidia-smi >&2 || true
    exit 1
  fi
  if (( free >= MIN_FREE_MIB )); then
    echo "GPU VRAM: ${free} MiB free (${MIN_FREE_GIB} GiB required)"
    return 0
  fi
  local free_gib used_gib
  free_gib="$(awk -v m="$free" 'BEGIN { printf "%.1f", m/1024 }')"
  used_gib="$(awk -v m="$used" 'BEGIN { printf "%.1f", m/1024 }')"
  echo "Cannot run tests: GPU has ${free} MiB (${free_gib} GiB) free; need at least ${MIN_FREE_GIB} GiB." >&2
  echo "Used ${used} MiB (${used_gib} GiB) of ${total} MiB. Free VRAM and re-run." >&2
  echo >&2
  nvidia-smi --query-gpu=index,name,memory.total,memory.used,memory.free --format=csv >&2
  echo >&2
  echo "Top processes consuming VRAM:" >&2
  print_gpu_consumers
  echo >&2
  echo "Typical fix: stop the resident serve container, then re-run:" >&2
  echo "  docker stop ninfer" >&2
  echo "  ./scripts/run-unit-tests.sh" >&2
  echo "Start it again afterwards with: docker start ninfer" >&2
  exit 1
}

run_python_suite() {
  local py="${NINFER_PYTHON:-python3}"
  if ! command -v "$py" >/dev/null 2>&1; then
    echo "Python suite skipped: ${py} not found"
    return 0
  fi
  if ! "$py" -c 'import pytest, torch' >/dev/null 2>&1; then
    echo "Python suite skipped: ${py} cannot import pytest and torch"
    return 0
  fi
  echo "=== Python pytest ==="
  (cd "$ROOT" && "$py" -m pytest \
    tests/artifact \
    tests/targets/qwen3_6_27b \
    tests/targets/qwen3_6_35b_a3b \
    tests/targets/qwen4 \
    tests/test_bench_matrix.py \
    tests/test_serve_corpus.py)
}

run_cpp_suite() {
  local src="$1"
  local build="$2"
  cd "$src"
  if [[ ! -f "${build}/CMakeCache.txt" ]] \
     || ! grep -q 'BUILD_TESTING:BOOL=ON' "${build}/CMakeCache.txt"; then
    echo "Configuring ${build} with BUILD_TESTING=ON..."
    cmake -S "$src" -B "$build" -G Ninja \
      -DCMAKE_BUILD_TYPE=Release \
      -DNINFER_BUILD_APPS=ON \
      -DBUILD_TESTING=ON
  fi
  echo "=== cmake --build ${build} ==="
  cmake --build "$build" --parallel "$JOBS"
  local ctest_cmd=(ctest --test-dir "$build" --output-on-failure)
  if [[ "$INCLUDE_REAL" -eq 0 ]]; then
    ctest_cmd+=(-E '_real_test$')
  fi
  if [[ ${#CTEST_ARGS[@]} -gt 0 ]]; then
    ctest_cmd+=("${CTEST_ARGS[@]}")
  fi
  echo "=== ${ctest_cmd[*]} ==="
  "${ctest_cmd[@]}"
}

run_inner() {
  local src="/src"
  local build="/build"
  if [[ ! -d "$src" ]]; then
    src="$ROOT"
  fi
  if [[ ! -f "${build}/CMakeCache.txt" ]]; then
    build="${src}/build"
  fi
  run_cpp_suite "$src" "$build"
}

bind_real_weights() {
  if [[ "$PRINT_WEIGHTS" -eq 1 || "$INCLUDE_REAL" -eq 1 ]]; then
    configure_weights
    if [[ "$INNER" -eq 0 ]]; then
      print_weights_report
    fi
  fi
  if [[ "$PRINT_WEIGHTS" -eq 1 ]]; then
    exit 0
  fi
  if [[ "$INCLUDE_REAL" -eq 1 ]]; then
    require_real_weights
  fi
}

if [[ "$INNER" -eq 1 ]] || in_container; then
  bind_real_weights
  require_free_vram
  run_inner
  exit 0
fi

bind_real_weights
require_free_vram

if command -v docker >/dev/null 2>&1 && "${ROOT}/scripts/dev-setup.sh"; then
  inner_args=(--inner)
  if [[ "$INCLUDE_REAL" -eq 1 ]]; then
    inner_args+=(--real)
  fi
  echo "=== docker exec ${BUILDER} ==="
  docker_cmd=(docker exec
    -e "NINFER_DEV_JOBS=${JOBS}"
    -e "NINFER_VRAM_CHECKED=1")
  if [[ "$INCLUDE_REAL" -eq 1 ]]; then
    mapfile -t weight_env_args < <(docker_weight_env_args)
    docker_cmd+=("${weight_env_args[@]+"${weight_env_args[@]}"}")
  fi
  docker_cmd+=(-w /src "$BUILDER" bash /src/scripts/run-unit-tests.sh)
  docker_cmd+=("${inner_args[@]}")
  docker_cmd+=("${CTEST_ARGS[@]+"${CTEST_ARGS[@]}"}")
  "${docker_cmd[@]}"
  if [[ "$RUN_PYTHON" -eq 1 ]]; then
    run_python_suite
  fi
  exit 0
fi

if [[ -f "${ROOT}/build/CTestTestfile.cmake" ]]; then
  run_cpp_suite "$ROOT" "${ROOT}/build"
  if [[ "$RUN_PYTHON" -eq 1 ]]; then
    run_python_suite
  fi
  exit 0
fi

echo "No GPU builder and no host test tree." >&2
echo "Start the committed builder with ./scripts/dev-setup.sh" >&2
echo "(Docker, NVIDIA Container Toolkit, RTX 5090), or configure a native tree:" >&2
echo "  cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTING=ON" >&2
exit 1
