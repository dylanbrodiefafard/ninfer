#!/usr/bin/env bash
# Incremental rebuild of ninfer and ninfer-serve, then inject into a runtime
# image and/or serve container. Source-only edits; Dockerfile or base-image
# changes still need a full `docker build`.
#
# Uses ./scripts/dev-setup.sh (same ninfer-builder, /src bind-mount, /build
# volume). Builds only the two app targets. Does not start a stopped serve
# container.
#
# Usage (from the repository root, or via this script's directory):
#   ./scripts/hot-patch.sh                # build, patch image, patch+restart if running
#   ./scripts/hot-patch.sh --no-restart   # patch image/container, do not restart
#   ./scripts/hot-patch.sh --image-only   # patch the runtime image only
#   ./scripts/hot-patch.sh --export-only  # write out/hot-patch/ only
#
# Env:
#   NINFER_IMAGE          Runtime tag. Default: image of NINFER_CONTAINER if that
#                         container exists, else ninfer:local.
#   NINFER_CONTAINER      Serve container name (default: ninfer).
#   NINFER_DEV_CONTAINER  Builder name (default: ninfer-builder).
#   NINFER_DEV_JOBS       Ninja parallelism (default: nproc).
#   NINFER_HOT_OUT        Host export directory (default: <repo>/out/hot-patch).
#   NINFER_HOT_ROLLBACK   0 skips tagging ${NINFER_IMAGE}-rollback before overwrite.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BUILDER="${NINFER_DEV_CONTAINER:-ninfer-builder}"
JOBS="${NINFER_DEV_JOBS:-$(nproc)}"
CONTAINER="${NINFER_CONTAINER:-ninfer}"
OUT="${NINFER_HOT_OUT:-${ROOT}/out/hot-patch}"

RESTART=1
IMAGE_ONLY=0
EXPORT_ONLY=0

usage() {
  sed -n '2,23p' "$0"
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --no-restart) RESTART=0; shift ;;
    --image-only) IMAGE_ONLY=1; shift ;;
    --export-only) EXPORT_ONLY=1; shift ;;
    --help|-h) usage; exit 0 ;;
    *)
      echo "unknown argument: $1" >&2
      usage >&2
      exit 1
      ;;
  esac
done

if [[ "$IMAGE_ONLY" -eq 1 && "$EXPORT_ONLY" -eq 1 ]]; then
  echo "--image-only and --export-only cannot be combined" >&2
  exit 1
fi

if ! command -v docker >/dev/null 2>&1; then
  echo "docker is required for hot-patch" >&2
  exit 1
fi

container_exists() {
  docker inspect --type container "$1" >/dev/null 2>&1
}

container_running() {
  [[ "$(docker inspect --type container -f '{{.State.Running}}' "$1" 2>/dev/null || true)" == "true" ]]
}

image_exists() {
  docker image inspect "$1" >/dev/null 2>&1
}

builder_src_mount() {
  docker inspect --type container -f \
    '{{range .Mounts}}{{if eq .Destination "/src"}}{{.Source}}{{end}}{{end}}' "$BUILDER"
}

resolve_image() {
  if [[ -n "${NINFER_IMAGE:-}" ]]; then
    printf '%s' "$NINFER_IMAGE"
    return
  fi
  local from_container=""
  if container_exists "$CONTAINER"; then
    from_container="$(docker inspect --type container -f '{{.Config.Image}}' "$CONTAINER")"
    if [[ -n "$from_container" ]]; then
      printf '%s' "$from_container"
      return
    fi
  fi
  printf '%s' "ninfer:local"
}

copy_apps_into() {
  local dest="$1"
  docker cp "$OUT/ninfer" "${dest}:/usr/local/bin/ninfer"
  docker cp "$OUT/ninfer-serve" "${dest}:/usr/local/bin/ninfer-serve"
}

HOT_PATCH_CID=""
cleanup_hot_patch_cid() {
  if [[ -n "${HOT_PATCH_CID}" ]]; then
    docker rm -f "$HOT_PATCH_CID" >/dev/null 2>&1 || true
    HOT_PATCH_CID=""
  fi
}

commit_apps_into_image() {
  local img="$1"
  trap cleanup_hot_patch_cid EXIT
  HOT_PATCH_CID="$(docker create "$img")"
  copy_apps_into "$HOT_PATCH_CID"
  if [[ "${NINFER_HOT_ROLLBACK:-1}" == "1" ]]; then
    docker tag "$img" "${img}-rollback"
  fi
  docker commit -m "hot-patch $(date -u +%Y%m%dT%H%M%SZ)" "$HOT_PATCH_CID" "$img" >/dev/null
  docker rm -f "$HOT_PATCH_CID" >/dev/null
  HOT_PATCH_CID=""
  trap - EXIT
}

echo "=== builder (scripts/dev-setup.sh) ==="
"${ROOT}/scripts/dev-setup.sh"

if ! container_running "$BUILDER"; then
  echo "builder ${BUILDER} is not running" >&2
  exit 1
fi

src_mount="$(builder_src_mount)"
if [[ -z "$src_mount" ]]; then
  echo "${BUILDER} has no /src bind-mount; recreate it with ./scripts/dev-setup.sh" >&2
  exit 1
fi
if [[ "$(realpath -m "$src_mount")" != "$(realpath -m "$ROOT")" ]]; then
  echo "${BUILDER} bind-mounts ${src_mount} at /src; this checkout is ${ROOT}." >&2
  echo "Remove the container and re-run ./scripts/dev-setup.sh" >&2
  exit 1
fi

if docker exec "$BUILDER" test -f /build/build.ninja; then
  echo "Restatting /build ninja log for bind-mounted sources..."
  docker exec "$BUILDER" ninja -C /build -t restat
fi

echo "Incremental build ninfer ninfer-serve (-j${JOBS})..."
docker exec "$BUILDER" cmake --build /build --parallel "$JOBS" \
  --target ninfer --target ninfer-serve

mkdir -p "$OUT"
docker cp "${BUILDER}:/build/apps/ninfer" "$OUT/ninfer"
docker cp "${BUILDER}:/build/apps/ninfer-serve" "$OUT/ninfer-serve"
chmod +x "$OUT/ninfer" "$OUT/ninfer-serve"
echo "Exported ${OUT}/ninfer ($(du -h "$OUT/ninfer" | awk '{print $1}'))"
echo "Exported ${OUT}/ninfer-serve ($(du -h "$OUT/ninfer-serve" | awk '{print $1}'))"

if [[ "$EXPORT_ONLY" -eq 1 ]]; then
  echo "OK hot-patch (export-only)"
  exit 0
fi

IMAGE="$(resolve_image)"
patched_image=0
patched_container=0

if image_exists "$IMAGE"; then
  echo "Updating image ${IMAGE}..."
  commit_apps_into_image "$IMAGE"
  patched_image=1
  echo "Updated ${IMAGE}"
  if [[ "${NINFER_HOT_ROLLBACK:-1}" == "1" ]]; then
    echo "Previous image kept as ${IMAGE}-rollback"
  fi
elif [[ "$IMAGE_ONLY" -eq 1 ]]; then
  echo "Runtime image ${IMAGE} not found." >&2
  echo "Build it once with: docker build --tag ${IMAGE} ${ROOT}" >&2
  exit 1
else
  echo "Runtime image ${IMAGE} not found; not creating one from a live container."
  echo "Build it once with: docker build --tag ${IMAGE} ${ROOT}"
fi

if [[ "$IMAGE_ONLY" -eq 0 ]] && container_exists "$CONTAINER"; then
  echo "Copying binaries into container ${CONTAINER}..."
  copy_apps_into "$CONTAINER"
  patched_container=1
  if [[ "$patched_image" -eq 0 ]]; then
    echo "${CONTAINER} is patched; ${IMAGE} is not. A recreate from the image would lose this patch."
  fi
  if [[ "$RESTART" -eq 1 ]] && container_running "$CONTAINER"; then
    echo "Restarting ${CONTAINER} (weight load can take several minutes)..."
    docker restart "$CONTAINER" >/dev/null
    echo "Restarted ${CONTAINER}"
  elif container_running "$CONTAINER" && [[ "$RESTART" -eq 0 ]]; then
    echo "${CONTAINER} is still running the previous process; restart it to load the new binaries."
  else
    echo "${CONTAINER} is stopped; next start will use the new binaries."
  fi
elif [[ "$IMAGE_ONLY" -eq 0 ]]; then
  echo "No container named ${CONTAINER}; image/export only."
  echo "Set NINFER_CONTAINER or start a serve container from ${IMAGE}."
fi

if [[ "$patched_image" -eq 0 && "$patched_container" -eq 0 ]]; then
  echo "Nothing was patched. Use --export-only, build ${IMAGE}, or start ${CONTAINER}." >&2
  exit 1
fi

echo "OK hot-patch"
