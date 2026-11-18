#include <cuda_runtime_api.h>

#include <cstdint>

namespace {
// One thread deliberately faults its isolated test process's CUDA context. No data is consumed;
// the host test checks typed asynchronous failure and teardown, outside sanitizer runs.
__global__ void asynchronous_error(std::uint32_t* destination) { *destination = 1; }
} // namespace

cudaError_t mailbox_test_cuda_error(cudaStream_t stream) {
    asynchronous_error<<<1, 1, 0, stream>>>(nullptr);
    return cudaGetLastError();
}
