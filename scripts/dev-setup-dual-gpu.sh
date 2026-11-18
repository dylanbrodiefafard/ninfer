#!/usr/bin/env bash
# Idempotent setup of the opt-in dual-card (RTX 5090 + Radeon AI PRO R9700) builder container.
#
# The image is docker/dual-gpu.Dockerfile: the CUDA builder image from ./scripts/dev-setup.sh
# plus a ROCm tree copied from the host through the named build context `rocm`. The container
# mounts the checkout at /src and keeps its own CMake volume at /build. It leaves the CUDA-only
# builder image, container and volume untouched; CUDA-only work keeps using ./scripts/dev-setup.sh.
#
# Usage (from the repository root, or via this script's directory):
#   ./scripts/dev-setup-dual-gpu.sh
# Environment:
#   NINFER_DUAL_ROCM_DIR        host ROCm tree (default /opt/rocm/core-10.0)
#   NINFER_DUAL_BUILDER_IMAGE   image tag (default local/ninfer-builder:dual-gpu)
#   NINFER_DUAL_DEV_CONTAINER   container name (default ninfer-builder-dual-gpu)
#   NINFER_DUAL_BUILD_VOLUME    build volume (default ninfer-dual-gpu-build-cache)
#   NINFER_REBUILD_DUAL_BUILDER=1 rebuilds the image
#   NINFER_MODELS_DIR           mounted read-only at /models when set
#   NINFER_DATA_DIR             mounted read-only at /data when set (for example an n-gram table)
#   NINFER_BUILDER_MEMORY_GIB   container memory limit (default three quarters of host RAM)
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
CUDA_IMAGE="${NINFER_BUILDER_IMAGE:-local/ninfer-builder:5090}"
IMAGE="${NINFER_DUAL_BUILDER_IMAGE:-local/ninfer-builder:dual-gpu}"
BUILDER="${NINFER_DUAL_DEV_CONTAINER:-ninfer-builder-dual-gpu}"
BUILD_VOL="${NINFER_DUAL_BUILD_VOLUME:-ninfer-dual-gpu-build-cache}"
ROCM_DIR="${NINFER_DUAL_ROCM_DIR:-/opt/rocm/core-10.0}"

for path in "$ROCM_DIR/lib/llvm/bin/clang++" /dev/kfd /dev/dri; do
  if [[ ! -e "$path" ]]; then
    echo "Missing dual-card prerequisite: $path" >&2
    exit 1
  fi
done
render_gid="$(getent group render | cut -d: -f3)"
if [[ -z "$render_gid" ]]; then
  echo 'The host render group is required for /dev/kfd and /dev/dri access.' >&2
  exit 1
fi
if ! docker image inspect "$CUDA_IMAGE" >/dev/null 2>&1; then
  echo "Build the CUDA builder image ${CUDA_IMAGE} first with ./scripts/dev-setup.sh." >&2
  exit 1
fi

if [[ "${NINFER_REBUILD_DUAL_BUILDER:-0}" == "1" ]] || ! docker image inspect "$IMAGE" >/dev/null 2>&1; then
  echo "Building dual-card builder image ${IMAGE}..."
  docker build --file "$ROOT/docker/dual-gpu.Dockerfile" --build-arg "BASE_IMAGE=${CUDA_IMAGE}" \
    --build-context "rocm=${ROCM_DIR}" --tag "$IMAGE" "$ROOT"
fi

docker volume create "$BUILD_VOL" >/dev/null

# The same hard memory limit as the CUDA builder; see scripts/dev-setup.sh.
host_mem_kib="$(awk '/^MemTotal:/ {print $2}' /proc/meminfo)"
MEMORY_GIB="${NINFER_BUILDER_MEMORY_GIB:-$((host_mem_kib * 3 / 4 / 1024 / 1024))}"
if ! [[ "$MEMORY_GIB" =~ ^[1-9][0-9]*$ ]] || ((MEMORY_GIB * 1024 * 1024 >= host_mem_kib)); then
  echo "NINFER_BUILDER_MEMORY_GIB=${MEMORY_GIB} must be a whole number of GiB below host RAM" >&2
  exit 1
fi

# NVIDIA devices come through CDI as in scripts/dev-setup.sh; the AMD card needs the KFD compute
# node and the DRM render nodes, which the render group may open.
create_args=(
  --name "$BUILDER"
  --device nvidia.com/gpu=all
  --device /dev/kfd
  --device /dev/dri
  --group-add "$render_gid"
  --memory "${MEMORY_GIB}g"
  --memory-swap "${MEMORY_GIB}g"
  --ulimit core=0
  -v "${ROOT}:/src:rw"
  -v "${BUILD_VOL}:/build"
  -w /src
)
if [[ -n "${NINFER_MODELS_DIR:-}" ]]; then
  create_args+=(-v "${NINFER_MODELS_DIR}:/models:ro")
fi
if [[ -n "${NINFER_DATA_DIR:-}" ]]; then
  create_args+=(-v "${NINFER_DATA_DIR}:/data:ro")
fi

if ! docker container inspect "$BUILDER" >/dev/null 2>&1; then
  echo "Creating dual-card builder container ${BUILDER}..."
  docker create "${create_args[@]}" "$IMAGE" sleep infinity >/dev/null
fi
if [[ "$(docker inspect -f '{{.State.Running}}' "$BUILDER")" != "true" ]]; then
  docker start "$BUILDER" >/dev/null
fi
printf 'Dual-card builder: %s\nImage: %s\nBuild volume: %s -> /build\n' "$BUILDER" "$IMAGE" "$BUILD_VOL"
