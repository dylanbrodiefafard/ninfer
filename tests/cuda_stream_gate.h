#pragma once

#include "core/device.h"

#include <cuda_runtime.h>

#include <condition_variable>
#include <mutex>

namespace ninfer::test {

// Keep following work incomplete until release(), independently of DMA size or
// hardware speed. Destruction releases and joins the callback before its state dies.
struct StreamCopyGate {
    std::mutex mutex;
    std::condition_variable cv;
    bool released = false;
    bool finished = false;
    bool launched = false;
    static void CUDART_CB callback(void* pointer) {
        auto& gate = *static_cast<StreamCopyGate*>(pointer);
        std::unique_lock lock(gate.mutex);
        gate.cv.wait(lock, [&] { return gate.released; });
        gate.finished = true;
        gate.cv.notify_all();
    }
    void launch(cudaStream_t stream) {
        CUDA_CHECK(cudaLaunchHostFunc(stream, callback, this));
        launched = true;
    }
    void release() {
        std::lock_guard lock(mutex);
        released = true;
        cv.notify_all();
    }
    ~StreamCopyGate() {
        release();
        if (launched) {
            std::unique_lock lock(mutex);
            cv.wait(lock, [&] { return finished; });
        }
    }
};

} // namespace ninfer::test
