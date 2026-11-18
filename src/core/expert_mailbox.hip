#include "core/expert_mailbox.h"

#include <hip/hip_runtime.h>

#include <stdexcept>
#include <string>

namespace ninfer {
namespace {
__device__ std::uint64_t load(const std::uint64_t* p) {
    return __hip_atomic_load(p, __ATOMIC_ACQUIRE, __HIP_MEMORY_SCOPE_SYSTEM);
}

__device__ bool failed(ExpertMailboxView view) {
    return __hip_atomic_load(reinterpret_cast<std::uint32_t*>(&view.failure->code),
                             __ATOMIC_ACQUIRE, __HIP_MEMORY_SCOPE_SYSTEM) != 0 ||
           __hip_atomic_load(reinterpret_cast<std::uint32_t*>(&view.peer_failure->code),
                             __ATOMIC_ACQUIRE, __HIP_MEMORY_SCOPE_SYSTEM) != 0;
}

__device__ void fail(ExpertMailboxView view, std::uint32_t layer, ExpertWireError code) {
    // One HIP stream, one leader per serialized kernel; never RMW a CUDA/CPU-owned line.
    if (!failed(view)) {
        view.failure->side     = ExpertWireSide::Hip;
        view.failure->sequence = view.control->sequence;
        view.failure->layer    = layer;
        __threadfence_system();
        __hip_atomic_store(reinterpret_cast<std::uint32_t*>(&view.failure->code),
                           static_cast<std::uint32_t>(code), __ATOMIC_RELEASE,
                           __HIP_MEMORY_SCOPE_SYSTEM);
    }
}

__device__ bool cancelled(ExpertMailboxView view) {
    return load(&view.cancellation->sequence) == view.control->sequence || failed(view);
}

__device__ void copy_bytes(void* destination, const void* source, std::uint32_t bytes) {
    auto* dst       = static_cast<uint4*>(destination);
    const auto* src = static_cast<const uint4*>(source);
    for (std::uint32_t i = threadIdx.x; i < bytes / 16; i += blockDim.x) { dst[i] = src[i]; }
    auto* tail_dst       = static_cast<std::uint32_t*>(destination);
    const auto* tail_src = static_cast<const std::uint32_t*>(source);
    for (std::uint32_t i = (bytes / 16) * 4 + threadIdx.x; i < bytes / 4; i += blockDim.x) {
        tail_dst[i] = tail_src[i];
    }
}

// One 256-thread CTA per layer. Leader performs bounded system acquire, shared count and barrier
// transfer acquisition to all payload readers. No mapped payload access on SKIP/error. The wall
// clock has the constant frequency queried by the host (not the variable shader clock).
__global__ void import_packet(ExpertMailboxView view, std::uint32_t layer, void* x, void* ids,
                              void* weights, std::uint32_t* count, std::uint32_t tokens,
                              std::uint64_t deadline_ticks) {
    __shared__ std::uint32_t n;
    if (threadIdx.x == 0) {
        n                = 0;
        const auto start = static_cast<std::uint64_t>(wall_clock64());
        auto* header     = view.slots[layer].request;
        bool valid       = true;
        while (load(&header->sequence) != view.control->sequence) {
            if (cancelled(view)) {
                fail(view, layer, ExpertWireError::Cancelled);
                valid = false;
                break;
            }
            if (static_cast<std::uint64_t>(wall_clock64()) - start >= deadline_ticks) {
                fail(view, layer, ExpertWireError::Deadline);
                valid = false;
                break;
            }
            __builtin_amdgcn_s_sleep(1);
        }
        if (valid && !cancelled(view)) {
            if (header->version != kExpertWireVersion || header->layer != layer ||
                header->tokens != tokens || header->tokens != view.control->tokens ||
                header->compact_tokens > tokens ||
                (header->compact_tokens == 0 ? header->status != ExpertWireStatus::Skip
                                             : header->status != ExpertWireStatus::Work)) {
                fail(view, layer, ExpertWireError::Protocol);
            } else {
                n = header->compact_tokens;
            }
        }
        if (cancelled(view)) {
            fail(view, layer, ExpertWireError::Cancelled);
            // Failed T48 imports still have fixed DMA nodes. Seal the source against future
            // writers, joining an already-claimed writer before those DMA nodes may execute.
            __hip_atomic_store(&view.slots[layer].response->copy_guard, view.control->sequence,
                               __ATOMIC_SEQ_CST, __HIP_MEMORY_SCOPE_SYSTEM);
            __threadfence_system();
            const auto seal_start = static_cast<std::uint64_t>(wall_clock64());
            while (__hip_atomic_load(&header->copy_guard, __ATOMIC_SEQ_CST,
                                     __HIP_MEMORY_SCOPE_SYSTEM) != 0) {
                if (static_cast<std::uint64_t>(wall_clock64()) - seal_start >= deadline_ticks) {
                    // The round is poisoned and count stays zero. Never abort the process for
                    // a failed publisher; fixed DMA bytes cannot license any helper jobs.
                    fail(view, layer, ExpertWireError::Deadline);
                    break;
                }
                __builtin_amdgcn_s_sleep(1);
            }
            n = 0;
        }
        *count = n;
    }
    __syncthreads();
    if (tokens != 48) {
        copy_bytes(x, view.slots[layer].activations, n * kExpertActivationBytes);
        copy_bytes(ids, view.slots[layer].ids, n * 40);
        copy_bytes(weights, view.slots[layer].weights, n * 40);
    }
}

// One 256-thread CTA. Every mapped writer fences before the barrier and leader release; the DMA
// path has already completed its stream-ordered copy. Empty jobs acknowledge SKIP without copying.
__global__ void export_packet(ExpertMailboxView view, std::uint32_t layer, const void* partial,
                              const std::uint32_t* count, std::uint32_t tokens) {
    const auto n = *count;
    if (tokens != 48 && !failed(view)) {
        copy_bytes(view.slots[layer].partial, partial, n * kExpertPartialBytes);
    }
    __threadfence_system();
    __syncthreads();
    if (threadIdx.x == 0) {
        if (load(&view.cancellation->sequence) == view.control->sequence) {
            fail(view, layer, ExpertWireError::Cancelled);
        }
        auto* header           = view.slots[layer].response;
        header->version        = kExpertWireVersion;
        header->layer          = layer;
        header->tokens         = tokens;
        header->compact_tokens = n;
        header->status = failed(view) ? ExpertWireStatus::Failed
                                      : (n == 0 ? ExpertWireStatus::Skip : ExpertWireStatus::Work);
        __threadfence_system();
        __hip_atomic_store(&header->sequence, view.control->sequence, __ATOMIC_RELEASE,
                           __HIP_MEMORY_SCOPE_SYSTEM);
    }
}

// One CTA, independent lanes zip at most 480 metadata fields. No barriers/shared memory;
// each destination U64 is written exactly once. Inputs are U32 aligned, output U64 aligned.
__global__ void interleave(const std::uint32_t* first, const std::uint32_t* second,
                           std::uint64_t* pairs, const std::uint32_t* count) {
    for (std::uint32_t i = threadIdx.x; i < *count * 10; i += blockDim.x) {
        pairs[i] = first[i] | (static_cast<std::uint64_t>(second[i]) << 32U);
    }
}

// One CTA reduces an exact position-sensitive U64 checksum, with modular addition. Records are
// 256-byte aligned and their spans are multiples of 8. Shared tree has a barrier at every level.
__global__ void checksum(const std::uint64_t* data, std::size_t words, std::uint64_t* result) {
    __shared__ std::uint64_t sums[256];
    std::uint64_t sum = 0;
    for (std::size_t i = threadIdx.x; i < words; i += blockDim.x) {
        std::uint64_t v = data[i] ^ static_cast<std::uint64_t>(i);
        v               = (v ^ (v >> 30U)) * 0xbf58476d1ce4e5b9ULL;
        v               = (v ^ (v >> 27U)) * 0x94d049bb133111ebULL;
        sum += v ^ (v >> 31U);
    }
    sums[threadIdx.x] = sum;
    __syncthreads();
    for (unsigned step = 128; step != 0; step /= 2) {
        if (threadIdx.x < step) { sums[threadIdx.x] += sums[threadIdx.x + step]; }
        __syncthreads();
    }
    if (threadIdx.x == 0) { *result = sums[0]; }
}

void check(hipError_t status, const char* operation) {
    if (status != hipSuccess) {
        throw HipError(std::string("expert mailbox HIP ") + operation + ": " +
                       hipGetErrorString(status));
    }
}

void validate(std::uint32_t layer, std::uint32_t tokens, ExpertMailboxView view) {
    if (layer >= kExpertMailboxLayers || tokens == 0 || tokens > view.capacity) {
        throw std::invalid_argument("expert HIP mailbox shape out of range");
    }
}
} // namespace

void expert_mailbox_hip_import(ExpertMailboxView view, std::uint32_t layer, void* x, void* ids,
                               void* weights, std::uint32_t* count, std::uint32_t tokens,
                               const HipStream& stream) {
    stream.bind();
    validate(layer, tokens, view);
    int device = 0, khz = 0;
    check(hipGetDevice(&device), "current device");
    check(hipDeviceGetAttribute(&khz, hipDeviceAttributeWallClockRate, device), "wall clock rate");
    if (khz <= 0) { throw HipError("expert helper has no constant wall clock"); }
    const auto ticks = static_cast<std::uint64_t>(khz) * (kExpertLayerDeadlineNs / 1'000'000);
    import_packet<<<1, 256, 0, static_cast<hipStream_t>(stream.get())>>>(
        view, layer, x, ids, weights, count, tokens, ticks);
    check(hipGetLastError(), "import launch");
    if (tokens == 48) {
        hip_copy_to_device(x, view.slots[layer].host_activations, tokens * kExpertActivationBytes,
                           stream);
        hip_copy_to_device(ids, view.slots[layer].host_ids, static_cast<std::size_t>(tokens) * 40,
                           stream);
        hip_copy_to_device(weights, view.slots[layer].host_weights,
                           static_cast<std::size_t>(tokens) * 40, stream);
    }
}

void expert_mailbox_hip_export(ExpertMailboxView view, std::uint32_t layer, const void* partial,
                               const std::uint32_t* count, std::uint32_t tokens,
                               const HipStream& stream) {
    stream.bind();
    validate(layer, tokens, view);
    if (tokens == 48) {
        hip_copy_to_host(view.slots[layer].host_partial, partial, tokens * kExpertPartialBytes,
                         stream);
    }
    export_packet<<<1, 256, 0, static_cast<hipStream_t>(stream.get())>>>(view, layer, partial,
                                                                         count, tokens);
    check(hipGetLastError(), "export launch");
}

void expert_mailbox_hip_interleave(const void* first, const void* second, void* pairs,
                                   const std::uint32_t* count, const HipStream& stream) {
    stream.bind();
    interleave<<<1, 256, 0, static_cast<hipStream_t>(stream.get())>>>(
        static_cast<const std::uint32_t*>(first), static_cast<const std::uint32_t*>(second),
        static_cast<std::uint64_t*>(pairs), count);
    check(hipGetLastError(), "interleave launch");
}

void expert_mailbox_hip_checksum(const void* data, std::size_t bytes, std::uint64_t* result,
                                 const HipStream& stream) {
    stream.bind();
    if (bytes == 0 || bytes % 8 != 0 || reinterpret_cast<std::uintptr_t>(data) % 8 != 0) {
        throw std::invalid_argument("HIP checksum requires nonempty aligned U64 span");
    }
    checksum<<<1, 256, 0, static_cast<hipStream_t>(stream.get())>>>(
        static_cast<const std::uint64_t*>(data), bytes / 8, result);
    check(hipGetLastError(), "checksum launch");
}
} // namespace ninfer
