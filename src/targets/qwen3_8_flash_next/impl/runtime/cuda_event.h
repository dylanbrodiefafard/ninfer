#pragma once

#include "core/device.h"

#include <cuda_runtime.h>

namespace ninfer::targets::qwen3_8_flash_next {

// Owns one CUDA event created with `flags`; creation failures throw through CUDA_CHECK.
class CudaEvent {
public:
    explicit CudaEvent(unsigned int flags) { CUDA_CHECK(cudaEventCreateWithFlags(&event_, flags)); }

    ~CudaEvent() {
        if (event_ != nullptr) { static_cast<void>(cudaEventDestroy(event_)); }
    }

    CudaEvent(const CudaEvent&)            = delete;
    CudaEvent& operator=(const CudaEvent&) = delete;

    [[nodiscard]] cudaEvent_t get() const noexcept { return event_; }

private:
    cudaEvent_t event_ = nullptr;
};

// Owns one non-blocking CUDA stream; creation failures throw through CUDA_CHECK. Destruction
// synchronizes the stream first, so no enqueued work outlives its owner's buffers.
class CudaStream {
public:
    CudaStream() { CUDA_CHECK(cudaStreamCreateWithFlags(&stream_, cudaStreamNonBlocking)); }

    ~CudaStream() {
        if (stream_ != nullptr) {
            static_cast<void>(cudaStreamSynchronize(stream_));
            static_cast<void>(cudaStreamDestroy(stream_));
        }
    }

    CudaStream(const CudaStream&)            = delete;
    CudaStream& operator=(const CudaStream&) = delete;

    [[nodiscard]] cudaStream_t get() const noexcept { return stream_; }

private:
    cudaStream_t stream_ = nullptr;
};

} // namespace ninfer::targets::qwen3_8_flash_next
