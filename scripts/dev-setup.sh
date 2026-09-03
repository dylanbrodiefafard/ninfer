#!/usr/bin/env bash
# Idempotent setup of the NInfer GPU builder container.
#
# The image is this repository's Dockerfile `build` stage (CUDA 13.1 devel,
# sm_120a). The container bind-mounts the checkout at /src and keeps a persistent
# CMake tree at /build with BUILD_TESTING ON. It does not touch a live ninfer
# serve container.
#
# Usage (from the repository root, or via this script's directory):
#   ./scripts/dev-setup.sh
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
IMAGE="${NINFER_BUILDER_IMAGE:-local/ninfer-builder:5090}"
BUILDER="${NINFER_DEV_CONTAINER:-ninfer-builder}"
BUILD_VOL="${NINFER_BUILD_VOLUME:-ninfer-build-cache}"

if ! command -v docker >/dev/null 2>&1; then
  echo "docker is required to start ${BUILDER}" >&2
  exit 1
fi

if [[ "${NINFER_REBUILD_BUILDER:-0}" == "1" ]] || ! docker image inspect "$IMAGE" >/dev/null 2>&1; then
  echo "Building builder image ${IMAGE} from Dockerfile build stage..."
  docker build --target build --tag "$IMAGE" "$ROOT"
fi

docker volume create "$BUILD_VOL" >/dev/null

create_args=(
  --name "$BUILDER"
  --gpus all
  -v "${ROOT}:/src:rw"
  -v "${BUILD_VOL}:/build"
  -w /src
)
if [[ -n "${NINFER_MODELS_DIR:-}" ]]; then
  create_args+=(-v "${NINFER_MODELS_DIR}:/models:ro")
fi

if ! docker ps -a --format '{{.Names}}' | grep -qx "$BUILDER"; then
  echo "Creating builder container ${BUILDER}..."
  docker create "${create_args[@]}" "$IMAGE" sleep infinity >/dev/null
fi

if [[ "$(docker inspect -f '{{.State.Running}}' "$BUILDER")" != "true" ]]; then
  docker start "$BUILDER" >/dev/null
fi

# Repair long-lived containers created from an older builder image.
docker exec "$BUILDER" bash -lc \
  'command -v python3 >/dev/null 2>&1 || { apt-get update -qq && apt-get install -y -qq python3; }'
docker exec "$BUILDER" bash -lc \
  'command -v ccache >/dev/null 2>&1 || { apt-get update -qq && apt-get install -y -qq ccache; }'
docker exec "$BUILDER" bash -lc \
  'pkg-config --exists libzstd || { apt-get update -qq && apt-get install -y -qq libzstd-dev; }'

# Idempotent RTX 5090 host-driver fix (also applied in the Dockerfile build stage).
docker exec "$BUILDER" bash -lc \
  'rm -rf /usr/local/cuda/compat /usr/local/cuda-13.1/compat /usr/local/cuda-13/compat; rm -f /etc/ld.so.conf.d/*compat*.conf; ldconfig'

if ! docker exec "$BUILDER" test -f /build/CMakeCache.txt \
   || ! docker exec "$BUILDER" grep -q 'BUILD_TESTING:BOOL=ON' /build/CMakeCache.txt; then
  echo "Configuring /build with BUILD_TESTING=ON..."
  docker exec "$BUILDER" cmake -S /src -B /build -G Ninja \
    -DCMAKE_BUILD_TYPE=Release \
    -DNINFER_BUILD_APPS=ON \
    -DBUILD_TESTING=ON \
    -DNINFER_BUILD_BENCHMARKS="${NINFER_BUILD_BENCHMARKS:-ON}"
fi

echo "=== ninfer-builder ==="
echo "container : $BUILDER (running=$(docker inspect -f '{{.State.Running}}' "$BUILDER"))"
echo "image     : $IMAGE"
echo "repo      : ${ROOT} -> /src"
echo "build     : volume ${BUILD_VOL} -> /build"
echo "testing   : $(docker exec "$BUILDER" bash -lc 'grep -E "^BUILD_TESTING:BOOL=" /build/CMakeCache.txt')"
echo "gpu       : $(docker exec "$BUILDER" bash -lc 'nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1 || echo NO-GPU')"
echo
echo "Run unit tests with: ./scripts/run-unit-tests.sh"
