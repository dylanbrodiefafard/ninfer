#!/usr/bin/env bash
# Idempotent setup of the NInfer GPU builder container.
#
# The image is this repository's Dockerfile `build` stage (CUDA 13.1 devel,
# sm_120a). The container bind-mounts the checkout at /src and keeps a persistent
# CMake tree at /build with BUILD_TESTING ON. It does not touch a live ninfer
# serve container. ./scripts/hot-patch.sh uses this builder to inject incremental
# ninfer / ninfer-serve builds into a runtime image or serve container.
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

# Repair long-lived containers created from an older builder image. CMake
# FetchContent needs git; xgrammar patches need patch.
docker exec "$BUILDER" bash -lc '
pkgs=()
command -v git >/dev/null 2>&1 || pkgs+=(git)
command -v patch >/dev/null 2>&1 || pkgs+=(patch)
command -v python3 >/dev/null 2>&1 || pkgs+=(python3)
command -v ccache >/dev/null 2>&1 || pkgs+=(ccache)
pkg-config --exists libzstd || pkgs+=(libzstd-dev)
if ((${#pkgs[@]})); then
  echo "Installing missing builder packages: ${pkgs[*]}"
  apt-get update -qq
  apt-get install -y -qq "${pkgs[@]}"
fi
'

# Idempotent RTX 5090 host-driver fix (also applied in the Dockerfile build stage).
docker exec "$BUILDER" bash -lc \
  'rm -rf /usr/local/cuda/compat /usr/local/cuda-13.1/compat /usr/local/cuda-13/compat; rm -f /etc/ld.so.conf.d/*compat*.conf; ldconfig'

build_tree_ready=0
if docker exec "$BUILDER" bash -lc '
     test -f /build/CMakeCache.txt \
     && grep -q "BUILD_TESTING:BOOL=ON" /build/CMakeCache.txt \
     && test -f /build/build.ninja \
     && [[ ! /build/CMakeCache.txt -nt /build/build.ninja ]] \
     && test -f /build/_deps/ninfer_xgrammar_source-src/cpp/grammar.cc
   '; then
  build_tree_ready=1
fi

if [[ "$build_tree_ready" -eq 0 ]]; then
  echo "Configuring /build with BUILD_TESTING=ON..."
  # A failed configure can leave an empty FetchContent dir that blocks git clone.
  docker exec "$BUILDER" bash -lc '
    if [[ ! -f /build/_deps/ninfer_xgrammar_source-src/cpp/grammar.cc ]]; then
      rm -rf /build/_deps/ninfer_xgrammar_source-src \
             /build/_deps/ninfer_xgrammar_source-build \
             /build/_deps/ninfer_xgrammar_source-subbuild
    fi
  '
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
