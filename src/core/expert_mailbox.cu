#include "core/expert_mailbox.cuh"
#include "core/expert_mailbox.h"

#include <stdexcept>
#include <string>

namespace ninfer {
namespace {
constexpr std::uint32_t kCopyThreads = 256;
// Each acquiring CTA copies up to four FP32 [2560] response columns: 2560 uint4 units, ten per
// thread, all loaded before any is stored.
constexpr std::uint32_t kAcquireColumns      = 4;
constexpr std::uint32_t kColumnUnits         = kExpertPartialBytes / 16;
constexpr std::uint32_t kAcquireUnitsPerLane = kAcquireColumns * kColumnUnits / kCopyThreads;
static_assert(kExpertPartialBytes % 16 == 0 && kExpertActivationBytes % 16 == 0);
static_assert(kAcquireColumns * kColumnUnits == kAcquireUnitsPerLane * kCopyThreads);

// One CTA of kCopyThreads copies `units` uint4 units: thread i owns units i + k * kCopyThreads.
// Every batch issues its kBatch loads before any of their stores, so a latency-bound source
// (mapped host memory, or VRAM feeding posted mapped stores) keeps kBatch x 4 KiB in flight per
// CTA instead of the 4 KiB of a load-store loop. Shape products <= 48 * 10240 bytes.
template <std::uint32_t kBatch>
__device__ void copy_units(uint4* __restrict__ destination, const uint4* __restrict__ source,
                           std::uint32_t units) {
    for (std::uint32_t base = threadIdx.x; base < units; base += kBatch * kCopyThreads) {
        uint4 staged[kBatch]{};
#pragma unroll
        for (std::uint32_t k = 0; k < kBatch; ++k) {
            const auto i = base + k * kCopyThreads;
            if (i < units) { staged[k] = source[i]; }
        }
#pragma unroll
        for (std::uint32_t k = 0; k < kBatch; ++k) {
            const auto i = base + k * kCopyThreads;
            if (i < units) { destination[i] = staged[k]; }
        }
    }
}

// uint4 units cover the aligned prefix; the final two I32/FP32 metadata elements of an odd
// compact count (40 N bytes) use scalar copies. Bases are 16-byte aligned.
__device__ void copy_bytes(void* destination, const void* source, std::uint32_t bytes) {
    copy_units<8>(static_cast<uint4*>(destination), static_cast<const uint4*>(source), bytes / 16);
    auto* tail_dst       = static_cast<std::uint32_t*>(destination);
    const auto* tail_src = static_cast<const std::uint32_t*>(source);
    for (std::uint32_t i = (bytes / 16) * 4 + threadIdx.x; i < bytes / 4; i += kCopyThreads) {
        tail_dst[i] = tail_src[i];
    }
}

// One CTA of kCopyThreads per slot: the request writer guard and header have one writer. No
// payload access for SKIP.
__global__ void __launch_bounds__(kCopyThreads)
    publish(ExpertMailboxView view, std::uint32_t layer, const void* x, const void* ids,
            const void* weights, const std::uint32_t* count) {
    if (!expert_wire_cuda::begin_request(view, layer)) { return; }
    const auto n = *count;
    if (n <= view.control->tokens) {
        copy_bytes(view.slots[layer].activations, x, n * kExpertActivationBytes);
        copy_bytes(view.slots[layer].ids, ids, n * 40);
        copy_bytes(view.slots[layer].weights, weights, n * 40);
    }
    expert_wire_cuda::publish_request(view, layer, n);
}

// Grid ceil(capacity / kAcquireColumns), block kCopyThreads: CTA b copies response columns
// [4 b, min(4 b + 4, N)) and exits without any mapped access when 4 b >= N (so N = 0 neither
// polls nor reads). Each active CTA's leader acquires the response sequence itself, and a shared
// flag plus barrier transfers that acquisition to its payload readers; CTA 0's leader alone
// records failures (acquire_response), the others only observe (observe_response). Consumers
// may not license any result on a latched error.
__global__ void __launch_bounds__(kCopyThreads)
    acquire(ExpertMailboxView view, std::uint32_t layer, const std::uint32_t* count,
            uint4* __restrict__ partial) {
    __shared__ bool valid;
    const auto n     = *count;
    const auto first = blockIdx.x * kAcquireColumns;
    if (first >= n) { return; }
    if (threadIdx.x == 0) {
        valid = blockIdx.x == 0 ? expert_wire_cuda::acquire_response(view, layer, n)
                                : expert_wire_cuda::observe_response(view, layer, n);
    }
    __syncthreads();
    if (!valid) { return; }
    const auto offset = first * kColumnUnits;
    copy_units<kAcquireUnitsPerLane>(partial + offset,
                                     static_cast<const uint4*>(view.slots[layer].partial) + offset,
                                     (min(n, first + kAcquireColumns) - first) * kColumnUnits);
}

void checked_launch() {
    const auto status = cudaGetLastError();
    if (status != cudaSuccess) {
        throw HipError(std::string("expert mailbox CUDA launch: ") + cudaGetErrorString(status));
    }
}

void check_layer(std::uint32_t layer) {
    if (layer >= kExpertMailboxLayers) {
        throw std::invalid_argument("expert mailbox layer exceeds 47");
    }
}
} // namespace

void expert_mailbox_cuda_publish(ExpertMailboxView view, std::uint32_t layer, const void* x,
                                 const void* ids, const void* weights, const std::uint32_t* count,
                                 void* stream) {
    check_layer(layer);
    publish<<<1, kCopyThreads, 0, static_cast<cudaStream_t>(stream)>>>(view, layer, x, ids, weights,
                                                                       count);
    checked_launch();
}

void expert_mailbox_cuda_acquire(ExpertMailboxView view, std::uint32_t layer,
                                 const std::uint32_t* count, void* partial, void* stream) {
    check_layer(layer);
    const auto blocks = (view.capacity + kAcquireColumns - 1) / kAcquireColumns;
    acquire<<<blocks, kCopyThreads, 0, static_cast<cudaStream_t>(stream)>>>(
        view, layer, count, static_cast<uint4*>(partial));
    checked_launch();
}
} // namespace ninfer
