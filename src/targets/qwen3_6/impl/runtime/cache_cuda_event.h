#pragma once

#include "core/device.h"

#include <new>

namespace ninfer::targets::qwen3_6::detail {

// CUDA documents memory exhaustion separately from asynchronous launch failures
// for event creation. Optional cache ownership can recover only the former.
inline void check_cache_cuda_event_allocation(cudaError_t result) {
    if (result == cudaErrorMemoryAllocation) {
        const cudaError_t pending = cudaGetLastError();
        if (pending != cudaSuccess && pending != cudaErrorMemoryAllocation) {
            CUDA_CHECK(pending);
        }
        throw std::bad_alloc();
    }
    CUDA_CHECK(result);
}

inline void create_cache_cuda_event(cudaEvent_t* event, unsigned int flags) {
    // CUDA specifies a newly created output on success, without promising the
    // output value on failure. Publish ownership only after creation succeeds.
    cudaEvent_t candidate = nullptr;
    check_cache_cuda_event_allocation(cudaEventCreateWithFlags(&candidate, flags));
    *event = candidate;
}

} // namespace ninfer::targets::qwen3_6::detail
