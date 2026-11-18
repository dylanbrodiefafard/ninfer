#pragma once

#include "core/expert_mailbox_protocol.h"

#include <cuda/atomic>
#include <cuda_runtime.h>

namespace ninfer::expert_wire_cuda {
using Sequence = cuda::atomic_ref<std::uint64_t, cuda::thread_scope_system>;
using Error    = cuda::atomic_ref<ExpertWireError, cuda::thread_scope_system>;

__device__ inline std::uint64_t now_ns() {
    std::uint64_t result;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(result));
    return result;
}

__device__ inline void fail(ExpertMailboxView view, std::uint32_t layer, ExpertWireError code) {
    // One CUDA stream, one leader per serialized kernel: this line has a single writer.
    // Peer errors license only drain, never a second write into the peer's error record.
    if (Error(view.failure->code).load(cuda::memory_order_acquire) == ExpertWireError::None &&
        Error(view.peer_failure->code).load(cuda::memory_order_acquire) == ExpertWireError::None) {
        view.failure->side     = ExpertWireSide::Cuda;
        view.failure->sequence = view.control->sequence;
        view.failure->layer    = layer;
        Error(view.failure->code).store(code, cuda::memory_order_release);
    }
}

__device__ inline bool cancelled(ExpertMailboxView view) {
    return Sequence(view.cancellation->sequence).load(cuda::memory_order_acquire) ==
               view.control->sequence ||
           Error(view.failure->code).load(cuda::memory_order_acquire) != ExpertWireError::None ||
           Error(view.peer_failure->code).load(cuda::memory_order_acquire) != ExpertWireError::None;
}

// CTA collective before any mapped payload store. Serializes an in-flight CUDA writer against
// the HIP failure seal, so unconditional fixed DMA cannot race a late failed publication.
__device__ inline bool begin_request(ExpertMailboxView view, std::uint32_t layer) {
    __shared__ bool writable;
    if (threadIdx.x == 0) {
        // Store→system fence→peer load is the two-single-writer exclusion handshake. If HIP
        // has already observed idle and sealed, CUDA must observe that seal before any store.
        Sequence(view.slots[layer].request->copy_guard)
            .store(view.control->sequence, cuda::memory_order_seq_cst);
        __threadfence_system();
        writable =
            Sequence(view.slots[layer].response->copy_guard).load(cuda::memory_order_seq_cst) !=
                view.control->sequence &&
            !cancelled(view);
        if (!writable) {
            Sequence(view.slots[layer].request->copy_guard).store(0, cuda::memory_order_release);
            fail(view, layer, ExpertWireError::Cancelled);
        }
    }
    __syncthreads();
    return writable;
}

// CTA collective, exactly one 256-thread block per slot. Called after every payload writer has
// finished its vector/scalar stores. Every writer fences; the barrier precedes leader release.
// N==0 transmits only a SKIP header, and must still be consumed by the prequeued helper graph.
__device__ inline void publish_request(ExpertMailboxView view, std::uint32_t layer,
                                       std::uint32_t count) {
    __threadfence_system();
    __syncthreads();
    if (threadIdx.x == 0) {
        auto* header           = view.slots[layer].request;
        header->version        = kExpertWireVersion;
        header->layer          = layer;
        header->tokens         = view.control->tokens;
        header->compact_tokens = count;
        const bool invalid     = count > view.control->tokens;
        if (invalid) { fail(view, layer, ExpertWireError::Protocol); }
        header->status = (invalid || cancelled(view))
                             ? ExpertWireStatus::Failed
                             : (count == 0 ? ExpertWireStatus::Skip : ExpertWireStatus::Work);
        Sequence(header->sequence).store(view.control->sequence, cuda::memory_order_release);
        Sequence(header->copy_guard).store(0, cuda::memory_order_release);
    }
}

// Leader only. Caller broadcasts the result with a CTA barrier before any payload reader runs.
__device__ inline bool acquire_response(ExpertMailboxView view, std::uint32_t layer,
                                        std::uint32_t count) {
    if (count > view.control->tokens) {
        fail(view, layer, ExpertWireError::Protocol);
        return false;
    }
    const std::uint64_t start = now_ns();
    auto* header              = view.slots[layer].response;
    while (Sequence(header->sequence).load(cuda::memory_order_acquire) != view.control->sequence) {
        if (cancelled(view)) {
            fail(view, layer, ExpertWireError::Cancelled);
            return false;
        }
        if (now_ns() - start >= kExpertLayerDeadlineNs) {
            fail(view, layer, ExpertWireError::Deadline);
            return false;
        }
        __nanosleep(64);
    }
    if (cancelled(view)) {
        fail(view, layer, ExpertWireError::Cancelled);
        return false;
    }
    if (header->version != kExpertWireVersion || header->layer != layer ||
        header->tokens != view.control->tokens || header->compact_tokens != count ||
        header->status != ExpertWireStatus::Work) {
        fail(view, layer, ExpertWireError::Protocol);
        return false;
    }
    return true;
}
} // namespace ninfer::expert_wire_cuda
