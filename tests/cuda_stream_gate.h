#pragma once

#include "core/device.h"
#include "targets/qwen3_6/impl/runtime/kv_ram_cache.h"

#include <cuda_runtime.h>

#include <condition_variable>
#include <mutex>

namespace ninfer::test {

// Keep following work incomplete until release(), independently of DMA size or
// hardware speed. Destruction releases and joins the callback before its state dies.
struct StreamCopyGate {
    std::mutex mutex;
    std::condition_variable cv;
    bool released                                     = false;
    bool finished                                     = false;
    bool launched                                     = false;
    targets::qwen3_6::detail::KVRamCache* armed_cache = nullptr;

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

    // Delay publication of copy completion, rather than copy submission: initcheck validates
    // D2H sources synchronously and cannot inspect a transfer behind a held stream callback.
    // The cache must outlive this gate until its one-shot callback has run.
    void arm(targets::qwen3_6::detail::KVRamCache& cache) {
        armed_cache = &cache;
        cache.test_before_copy_completion(
            [](void* state, cudaStream_t stream) {
                auto& gate       = *static_cast<StreamCopyGate*>(state);
                gate.armed_cache = nullptr;
                gate.launch(stream);
            },
            this);
    }

    void release() {
        std::lock_guard lock(mutex);
        released = true;
        cv.notify_all();
    }

    ~StreamCopyGate() {
        if (armed_cache != nullptr) { armed_cache->test_before_copy_completion(nullptr, nullptr); }
        release();
        if (launched) {
            std::unique_lock lock(mutex);
            cv.wait(lock, [&] { return finished; });
        }
    }
};

} // namespace ninfer::test
