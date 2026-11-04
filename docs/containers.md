# R9700 containers and test runner

## Compose: server with persistent prefix caching

`compose.yaml` builds this checkout and runs the selected DFlash artifact with temperature **1.5**, **12 GiB pinned RAM prefix cache**, **25 GiB disk cache**, chunk 2048, C6/adaptive K up to 7 and a 262144-token context limit. The cache options use MiB: `NINFER_RAM_CACHE_MIB=12288` and `NINFER_DISK_CACHE_MIB=25600`. Cache sizes are capacities, not upfront disk allocations.
Set `NINFER_SYSTEM_PROMPTS=N` to opt into retaining up to N system prefixes inside the RAM cache budget; the default is `0` (off). First-user content may differ. This does not allocate another cache or enable the disk tier. Startup must use an image containing `--kv-ram-system-prompts`.

A bind mount stores disk prefixes under `./.cache/ninfer/prefix` by default. Set `NINFER_DISK_CACHE_DIR` to choose another host cache directory; use a separate directory for each model/server deployment. This host directory survives container replacement and `docker compose down --volumes`. Disk cache is prefix reuse, not active-context offload. Disk format v8 requires a fresh directory when upgrading from earlier formats. RAM contents do not survive restart; durable disk entries can be restored for the same model fingerprint. When disabling disk cache with `NINFER_DISK_CACHE_MIB=off`, remove `--kv-disk-location /cache/prefix` from the local Compose command override: the server accepts a disk location only when disk capacity is enabled.

Docker defaults reasoning effort to `low` through `NINFER_DEFAULT_REASONING_EFFORT`. Set it to `medium` or `xhigh` to change the omitted-request effort, or `template` to use the artifact default. Explicit request effort still overrides it.

Requests may override temperature; p-less sampling remains enabled by default. Adaptive draft length and the optimized proposal head are enabled. Closed-turn assistant reasoning preservation is off by default; requests may enable `preserve_thinking` individually.

Pending requests have a 900000 ms (15-minute) preparation/admission timeout; the default output limit is 32768 tokens. The 262144-token context limit includes prompt and output and is a literal `--max-context` argument in `compose.yaml`.

`NINFER_DEFAULT_THINKING_TOKEN_BUDGET` controls the server's omitted-request reasoning cap (`off` by default, or a positive uint32 token count). For example, `16384` provides a 16K deployment default. An explicit request null opts out, a positive request cap overrides it, and inherited caps are inactive for requests with thinking disabled.

Device startup selects driver-backed HIP/GTT host memory before HIP initialization, using `HSA_USERPTR_FOR_PAGED_MEM=0` and normal SVM addressing. This fixed process policy needs no host sysctl changes. The RAM prefix cache, checkpoint slabs and runtime pinned buffers share the driver's GTT capacity; lower the requested RAM cache if startup reports insufficient GTT.

Compose selects bulk packet replay with `DEBUG_HIP_GRAPH_CLASSIC_PATH=0`, `ROC_ENABLE_LARGE_BAR=0`, `ROC_SKIP_KERNEL_ARG_COPY=0`, `HSA_DISABLE_FRAGMENT_ALLOCATOR=1`, and `DEBUG_HIP_GRAPH_BATCH_SIZE=16384` on the selected HIP 7.15 runtime. The runtime places immutable graph arguments in pinned host memory and submits cached dispatch packets in bulk. Normal argument copying is required: skipping it failed the large-graph exact-output check. Disabling ROCr's fragment allocator prevents the reproduced startup allocation failure at the qualified capacity. The low-residency budget requires the first four settings and the qualified production preset; the batch cap controls submission cost. Other settings retain the conservative inventory bound. See `docs/performance.md` for residency, capacity and speed evidence. Requalify when the HIP toolchain or captured kernels change.

Automatic sizing also reads the matching PCI device's amdgpu VRAM counters under `/sys` and uses the smaller of physical and HIP availability. The counters must remain readable inside the container so the external 64-MiB headroom is backed by physical availability.

### First run on localhost

Install Docker Engine with Buildx and Compose ≥2.24.4. Supply the ROCm 10 tree, a self-contained Python 3.11 distribution and the selected `.ninfer` artifact (see `model-artifact.md`). `.env.example` contains paths to edit, not downloads. A system Python virtual environment is not a self-contained distribution.

```sh
cp .env.example .env
# Edit NINFER_PYTHON_CONTEXT, NINFER_ROCM_CONTEXT and NINFER_MODELS_DIR.
# Set NINFER_MODEL relative to NINFER_MODELS_DIR.
mkdir -p .cache/ninfer
# If you changed NINFER_DISK_CACHE_DIR, create that directory instead.
docker compose -f compose.yaml -f compose.local.yaml config --quiet
docker compose -f compose.yaml -f compose.local.yaml up -d --wait
curl http://127.0.0.1:8080/health
```

`compose.local.yaml` replaces the service's proxy network with a private default network and publishes only `127.0.0.1:8080`. Use the same two `-f` arguments for later `logs`, `stop`, `down` and update commands. No external network is needed. The native README HTTP example works against this address.

Local release-prep validation (2026-10-17): both Compose configurations resolve successfully with edited paths. An isolated source checkout built all four README targets from scratch on Ubuntu 24.04 with the existing ROCm 10 prerequisites. The fresh CLI generated with default DFlash, and the fresh native server passed `/health` and the README Chat Completions request using the selected local artifact. This checks the native build/run instructions and Compose configuration; it does not establish a fresh container-image build or a clean-machine ROCm installation.

### Existing reverse-proxy deployment

The base `compose.yaml` preserves the maintainer deployment: no host port, external network `proxy`, service alias `ninfer-r9700`. Create the network if needed, then use the base file alone:

```sh
docker network inspect proxy >/dev/null 2>&1 || docker network create proxy
docker compose up -d --wait
docker compose logs -f server
```

Other containers on that network use `http://ninfer-r9700:8080/v1` for OpenAI clients. No API key is configured by default; joining the network does not create a public reverse-proxy route.

Buildx is required to build, not to run an already-built image. It is Docker's BuildKit frontend: it supplies build caching, multi-stage builds and the named local ROCm/Python contexts used here. Compose is the separate plugin that manages the service's devices, ports, mounts and lifecycle. Both must be installed for the complete workflow below.

The image compiles a snapshot of **local working-tree files** using CMake and Ninja and copies the resulting apps into the runtime image. `pull_policy: build` makes `docker compose up -d` build before starting, including when the image already exists. Unchanged builds reuse image layers; source edits reuse CMake/Ninja objects in a locked BuildKit cache and compile affected dependencies. The finished binaries are copied out of the cache into an ordinary image layer, so the runtime image is self-contained. Toolchain, Python or build-dependency changes automatically reset the object cache. The first build on a builder, or after cache pruning, is a full compile. Keep using the same Buildx builder to retain its cache. Source snapshots are synchronized by content, so restored timestamps cannot hide edits from Ninja. Configuration is regenerated with current source defaults while reusable object files remain cached.

Keep `COMPOSE_BAKE=false` from `.env.example` in the project's `.env`. On the maintainer host's Compose 2.37.1/Buildx 0.37.1 combination, Bake delegation built the changed image but started the existing container with its old image. Direct Compose BuildKit integration preserves the object cache and correctly recreates the container after a source change. The upstream report for this failure class is https://github.com/docker/compose/issues/12774.

A GPU hang can abort the process and trigger `restart: unless-stopped`. If the restarted process instead stalls during HIP initialization, model load or warmup, the server's 180-second inactivity watchdog exits with status 124 so Compose can retry. Docker health failures alone do not restart a running container. The watchdog bypasses normal HIP teardown, which can block in a damaged driver, and is disabled once warmup completes. A persistent device/driver failure can still require host intervention; retries do not establish a fix for the original GPU hang.

After changing or pulling source, run `docker compose up -d` (add `--wait` to wait for health). Compose recreates the server when the resulting image changes. Builds need no GPU lease and can overlap inference; lower `NINFER_BUILD_JOBS` if host memory or I/O is under pressure. To start an existing image without building, use `docker compose up -d --no-build --pull never server`. Tests run separately.

For a dedicated Buildx builder, create it once (this does not change Docker's globally selected builder):

```sh
docker buildx create --name ninfer-r9700-build --driver docker-container \
  --driver-opt cpu-period=100000,cpu-quota=1400000,default-load=true
export BUILDX_BUILDER=ninfer-r9700-build
docker compose up -d --wait
```

The compile-job default is 8, with no builder memory limit. The builder CPU quota allows 14 logical CPUs; it is not a Linux load-average cap. Compose's memory limit applies only to the running server. Reuse the builder for subsequent builds.

`.env` can override model path, C1..16 (C1..8 with DFlash), maximum draft count and cache capacities. Defaults are C6 and DFlash adaptive K up to 7, the recommended AMD draft setting. Adaptive selection measures yield and round time; a maximum of 7 does not force every round to draft seven tokens. See `docs/performance.md` for workload-specific measurements. The runtime is bounded to 24 GiB host RAM with no container swap; this limit does not count GPU VRAM. Cache/runtime allocations must fit that bound.

Docker references: https://docs.docker.com/reference/compose-file/build/ and https://docs.docker.com/reference/cli/docker/buildx/build/. Build policy: https://docs.docker.com/reference/compose-file/services/#pull_policy; object caching: https://docs.docker.com/reference/dockerfile/#run---mounttypecache; build/restart semantics: https://docs.docker.com/reference/cli/docker/compose/up/; builder limits: https://docs.docker.com/build/builders/drivers/docker-container/.

## Image prerequisites

The Dockerfile builds only `gfx1201` and the Qwen3.8-27B product. It copies the maintainer's self-contained ROCm 10 tree through BuildKit's `rocm` named context, and a self-contained Python 3.11 distribution through `python311`. No model or converted artifact is downloaded or copied into an image. Docker build installs Ubuntu build/runtime packages and CMake fetches the pinned CPU grammar dependency.

Docker Buildx is required for these named contexts. On this Ubuntu host with the Docker apt repository configured, install the missing plugin once:

```sh
sudo apt-get install docker-buildx-plugin
docker buildx version
```

```sh
docker buildx build --load --target runtime --tag local/ninfer-r9700:local \
  --build-context rocm=/opt/rocm/core-10.0 \
  --build-context python311=/absolute/path/to/self-contained-python-3.11 .
```

Image builds default to 8 compile jobs; `--build-arg NINFER_BUILD_JOBS=4` lowers this within the enforced range 1–8, which the CMake Ninja job pool also enforces. Each agent may run one build at a time; GPU jobs use the shared lock in `AGENTS.md`. These are native AMD images: no NVIDIA Container Toolkit or `--gpus all` is needed.

Both contexts must contain their `bin` and `lib` directories; do not provide a single executable, an external-symlink virtual environment, or the parent of a versioned ROCm installation. The runtime retains the same ROCm tree used to link the apps. The host provides the AMD kernel driver and `/dev/kfd` plus the R9700's DRM render node. The default render node is `/dev/dri/renderD128`; select the actual R9700 node with `NINFER_DRM_RENDER_NODE` if it differs.

## Builder

```sh
NINFER_PYTHON_CONTEXT=/absolute/path/to/self-contained-python-3.11 \
NINFER_MODELS_DIR=/absolute/path/to/local-models \
  bash scripts/dev-setup.sh
```

Defaults are `local/ninfer-r9700-builder:local`, container `ninfer-r9700-builder`, and volume `ninfer-r9700-build-cache`. Override them with `NINFER_BUILDER_IMAGE`, `NINFER_DEV_CONTAINER`, and `NINFER_BUILD_VOLUME`. `NINFER_ROCM_CONTEXT` defaults to `/opt/rocm/core-10.0`. If the Python context is omitted, setup resolves the installed `python3.11` executable and takes its distribution root. Setup mounts the checkout at `/src`, the build volume at `/build`, and the optional models directory read-only at `/models`. It checks existing container ownership of the checkout and never repurposes another builder. Set `NINFER_REBUILD_BUILDER=1` to rebuild the image; an existing container continues to use its original image until the user recreates it. No packages are installed into an existing container.

Setup configures the mounted build volume. Compile the applications explicitly:

```sh
docker exec ninfer-r9700-builder cmake --build /build --parallel 8 \
  --target ninfer ninfer-serve ninfer-ppl
docker exec ninfer-r9700-builder /build/apps/ninfer --help
```

This development container retains objects in its own `/build` volume and rebuilds affected dependencies after edits to the live `/src` checkout. Native `cmake --build build-r9700 --parallel 8 --target ninfer-serve` is incremental too. Compose uses its separate BuildKit object cache and updates the runtime image through `docker compose up -d`; development-container builds do not deploy apps.

For benchmarks, set `NINFER_BUILD_BENCHMARKS=ON` when running setup, then build the `ninfer_bench` target; its executable is `/build/bench/ninfer_bench`. The runtime image contains the three applications, not the benchmark or test tools.

## Tests

There is a registered CTest suite for host contracts and physical GPU Ops/runtime qualifiers, plus separate real-model and live-HTTP checks. No single command proves every model/context/precision combination. Run GPU tests only after coordinating with any resident GPU owner. On the shared maintainer host, hold the GPU lock specified in `AGENTS.md`; builds need no GPU lock.

```sh
docker compose stop server
# Existing native build, rebuilt first; all registered host and GPU CTests, serially:
bash scripts/run-unit-tests.sh --gpu -- --parallel 1
# Real-artifact RAM continuation/cancellation and recovery-on-resident-KV qualification:
bash scripts/run-unit-tests.sh --real /absolute/path/to/exact-model.ninfer -- --parallel 1
# Restart/persistence proof requires a fresh directory whose parent already exists:
build-r9700/src/ninfer_r9700_engine_cache_cancel_qual /absolute/path/to/exact-model.ninfer \
  --disk-dir /absolute/path/to/new-cache-qualification
# Start Compose only after the other tests finish, then exercise real HTTP routes:
docker compose up -d --no-build --pull never server
# Run the HTTP smoke from a Python 3.11 environment on the proxy network:
python3.11 tools/smoke/serve_cache.py --base-url http://ninfer-r9700:8080
```

The HTTP smoke sends Chat, Responses, Anthropic and streaming requests without a temperature override, checks generated output, requires both cache tiers enabled on `GET /metrics.json`, checks the `GET /metrics` exposition and generation count, and reports per-request copy times plus the scheduler, device-KV and tier snapshot. It does not prove disk restoration; the separate restart qualifier explicitly requires a disk hit and cold-output parity.

Earlier native and container integration checks exercised Chat, Responses, Anthropic and SSE, RAM prefix reuse and SSD restoration after restart. Historical local reports are under `profiles/bench/r9700-compose-validation-20260924/`; they used older cache sizes and do not establish current throughput. Current performance measurements are in `performance.md`.

The native runner uses an already configured `build-r9700` with `BUILD_TESTING=ON`; `NINFER_BUILD_DIR` changes that explicit path. `--builder` selects the dedicated builder and its `/build`. Host tests run by default. GPU tests must be requested explicitly, and the runner does not stop a resident server or other GPU owner.

```sh
bash scripts/run-unit-tests.sh -- --stop-on-failure
bash scripts/run-unit-tests.sh --builder --gpu -- -R ninfer_kv_disk_cache_test
bash scripts/run-unit-tests.sh --builder --real /models/exact-selected-artifact.ninfer
NINFER_PYTHON=/absolute/path/to/python3.11 bash scripts/run-unit-tests.sh --python
```

`--real` runs the public-Engine ordinary cancellation and RAM continuation qualifier with that exact artifact (one active request, greedy, 16-token outputs), then `ninfer_r9700_recovery_kv_qual ARTIFACT mtp`. That Program-level qualifier prefills only the generation-recovery suffix from a retained lane, restores a captured RAM image after three lane aborts, and requires two decoded retries to reproduce the undecoded control's committed GDN/hidden state and greedy continuation; run it again with `dflash` or `none` for the other backends. It does not infer a model filename or silently skip a missing artifact. Paths passed with `--builder` are container paths. `--gpu` enables registered physical tests; it does not select a cache dtype. All product tests use FP8 E4M3FN keys, signed INT4 values and FP16 value scales.

The same built qualifier has a separate explicit SSD restart mode:

```sh
build-r9700/src/ninfer_r9700_engine_cache_cancel_qual /absolute/path/to/selected.ninfer \
  --disk-dir /absolute/path/to/new-cache-qualification
```

It uses one lane, 16-token greedy requests, 1 GiB RAM and 1 GiB SSD cache, closes and reopens the public Engine, requires an SSD restore, and compares with cold output. The parent directory must exist; the named directory must not exist. Its cache files are retained on success or failure and are never removed by the test.

The physical-test preflight reads the selected DRM device's VRAM counters and requires 20 GiB free by default; `NINFER_MIN_FREE_VRAM_GIB` can set a different explicit requirement for a focused test. A build-directory lock prevents two instances of this runner sharing that build from using the GPU concurrently. CTest also uses the `r9700_device` resource lock. These locks do not coordinate unrelated benchmark processes: the maintainer still schedules the sole GPU.

`--python` additionally runs the artifact, benchmark-matrix and serving-corpus Python tests with Python 3.11. The selected interpreter must already provide `pytest` and `torch`; the runner does not install or upgrade dependencies. `NINFER_DEV_JOBS` controls development setup and test-runner build parallelism (default 8, enforced range 1–8). CTest arguments follow `--`.

## Runtime updates

Run `docker compose up -d` for both source edits and Dockerfile/toolchain/runtime dependency changes. Compose builds the image and replaces the service when needed; there is no separate app-patching deployment path. The build does not require a running server, a GPU device, or the development container. No container workflow changes the startup-fixed one-to-eight active-request contract.
