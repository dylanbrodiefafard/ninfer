#include "core/expert_mailbox.h"
#include <hip/hip_runtime.h>

#include <string>

// One CTA of 256 independent lanes checks every transported word, including odd-count metadata
// tails, and writes a nonce-bearing output in place into the response payload (out is the slot's
// HIP partial alias), as the helper Op does. No shared state; bad is an AMD-local atomic counter.
__global__ void nonce(const void* x, const void* ids, const void* weights, void* out,
                      const std::uint32_t* count, std::uint32_t* bad) {
    const auto n      = *count;
    const auto* input = static_cast<const std::uint32_t*>(x);
    const auto seed   = n == 0 ? 0U : input[0];
    for (std::uint32_t i = threadIdx.x; i < n * 1280; i += blockDim.x) {
        if (input[i] != (seed ^ i)) { atomicAdd(bad, 1U); }
    }
    for (std::uint32_t i = threadIdx.x; i < n * 10; i += blockDim.x) {
        if (static_cast<const std::uint32_t*>(ids)[i] != (seed ^ i ^ 0x12340000U)) {
            atomicAdd(bad, 1U);
        }
        if (static_cast<const std::uint32_t*>(weights)[i] != (seed ^ i ^ 0x56780000U)) {
            atomicAdd(bad, 1U);
        }
    }
    for (std::uint32_t i = threadIdx.x; i < n * 2560; i += blockDim.x) {
        static_cast<std::uint32_t*>(out)[i] = seed ^ i ^ 0xa5a5a5a5U;
    }
}

// A single lane introduces a bounded helper delay without modifying transport flags.
__global__ void delay(std::uint64_t ticks) {
    const auto start = static_cast<std::uint64_t>(wall_clock64());
    while (static_cast<std::uint64_t>(wall_clock64()) - start < ticks) {
        __builtin_amdgcn_s_sleep(1);
    }
}

void mailbox_test_nonce(const void* x, const void* ids, const void* weights, void* out,
                        const std::uint32_t* count, std::uint32_t* bad,
                        const ninfer::HipStream& stream) {
    nonce<<<1, 256, 0, static_cast<hipStream_t>(stream.get())>>>(x, ids, weights, out, count, bad);
    const auto status = hipGetLastError();
    if (status != hipSuccess) {
        throw ninfer::HipError(std::string("nonce launch: ") + hipGetErrorString(status));
    }
}

// Single-writer injected runtime failure, before any real packet processing.
__global__ void inject_failure(ninfer::ExpertMailboxView view) {
    view.failure->side     = ninfer::ExpertWireSide::Hip;
    view.failure->sequence = view.control->sequence;
    view.failure->layer    = 0;
    __threadfence_system();
    __hip_atomic_store(reinterpret_cast<std::uint32_t*>(&view.failure->code),
                       static_cast<std::uint32_t>(ninfer::ExpertWireError::Helper),
                       __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_SYSTEM);
}

void mailbox_test_fail(ninfer::ExpertMailboxView view, const ninfer::HipStream& stream) {
    inject_failure<<<1, 1, 0, static_cast<hipStream_t>(stream.get())>>>(view);
    if (hipGetLastError() != hipSuccess) {
        throw ninfer::HipError("nonce failure injection launch failed");
    }
}

void mailbox_test_delay(const ninfer::HipStream& stream) {
    int device = 0, rate = 0;
    auto status = hipGetDevice(&device);
    if (status == hipSuccess) {
        status = hipDeviceGetAttribute(&rate, hipDeviceAttributeWallClockRate, device);
    }
    if (status != hipSuccess || rate <= 0) { throw ninfer::HipError("nonce clock query failed"); }
    delay<<<1, 1, 0, static_cast<hipStream_t>(stream.get())>>>(static_cast<std::uint64_t>(rate) *
                                                               20);
    status = hipGetLastError();
    if (status != hipSuccess) { throw ninfer::HipError("nonce delay launch failed"); }
}
