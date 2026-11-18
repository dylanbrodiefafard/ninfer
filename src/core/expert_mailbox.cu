#include "core/expert_mailbox.cuh"
#include "core/expert_mailbox.h"

#include <stdexcept>
#include <string>

namespace ninfer {
namespace {
// One CTA, 256 lanes. uint4 covers aligned prefixes; the final two I32 metadata elements of an
// odd compact count use scalar copies. No payload access for SKIP. Shape products <= 48*10240.
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

__global__ void publish(ExpertMailboxView view, std::uint32_t layer, const void* x, const void* ids,
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

// One CTA, leader acquires exactly once. Shared valid plus the barrier transfers acquisition to
// all lanes. No remote read/wait when N=0; consumers may not license any result on latched error.
__global__ void acquire(ExpertMailboxView view, std::uint32_t layer, const std::uint32_t* count,
                        void* partial) {
    __shared__ bool valid;
    if (*count == 0) { return; }
    if (threadIdx.x == 0) { valid = expert_wire_cuda::acquire_response(view, layer, *count); }
    __syncthreads();
    if (valid) { copy_bytes(partial, view.slots[layer].partial, *count * kExpertPartialBytes); }
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
    publish<<<1, 256, 0, static_cast<cudaStream_t>(stream)>>>(view, layer, x, ids, weights, count);
    checked_launch();
}

void expert_mailbox_cuda_acquire(ExpertMailboxView view, std::uint32_t layer,
                                 const std::uint32_t* count, void* partial, void* stream) {
    check_layer(layer);
    acquire<<<1, 256, 0, static_cast<cudaStream_t>(stream)>>>(view, layer, count, partial);
    checked_launch();
}
} // namespace ninfer
