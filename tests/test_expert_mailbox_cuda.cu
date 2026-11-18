#include "core/expert_mailbox.cuh"

#include <cuda_runtime_api.h>

#include <cstdint>

namespace {
// One thread deliberately faults its isolated test process's CUDA context. No data is consumed;
// the host test checks typed asynchronous failure and teardown, outside sanitizer runs.
__global__ void asynchronous_error(std::uint32_t* destination) { *destination = 1; }

// One thread leaves an idle stream with a claimed writer, reproducing a missing publication
// without a CUDA runtime fault. HIP's bounded seal wait must report a fatal deadline.
__global__ void claim_writer(ninfer::ExpertMailboxView view) {
    ninfer::expert_wire_cuda::Sequence(view.slots[0].request->copy_guard)
        .store(view.control->sequence, cuda::memory_order_seq_cst);
}
} // namespace

cudaError_t mailbox_test_claim_writer(ninfer::ExpertMailboxView view, cudaStream_t stream) {
    claim_writer<<<1, 1, 0, stream>>>(view);
    return cudaGetLastError();
}

cudaError_t mailbox_test_cuda_error(cudaStream_t stream) {
    asynchronous_error<<<1, 1, 0, stream>>>(nullptr);
    return cudaGetLastError();
}
