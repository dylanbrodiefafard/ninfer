#include "core/device.h"
#include "core/arena.h"

#include <cuda_runtime.h>

#include <iostream>
#include <stdexcept>
#include <utility>

namespace {

int fail(const char* message) {
    std::cerr << message << '\n';
    return 1;
}

bool cuda_unavailable(cudaError_t err) {
    return err == cudaErrorNoDevice || err == cudaErrorInsufficientDriver;
}

int expect_throws_device(int device_id) {
    try {
        ninfer::DeviceContext invalid(device_id);
    } catch (const std::runtime_error&) { return 0; }
    std::cerr << "DeviceContext(" << device_id << ") did not throw\n";
    return 1;
}

int check_context(const ninfer::DeviceContext& ctx, const char* label) {
    int failures = 0;
    if (ctx.stream == nullptr) {
        std::cerr << label << " compute stream is null\n";
        ++failures;
    }
    if (ctx.load_stream == nullptr) {
        std::cerr << label << " load stream is null\n";
        ++failures;
    }
    if (ctx.copy_stream == nullptr) {
        std::cerr << label << " copy stream is null\n";
        ++failures;
    }
    if (ctx.sm() <= 0) {
        std::cerr << label << " sm is not positive\n";
        ++failures;
    }
    if (ctx.total_vram() == 0) {
        std::cerr << label << " total_vram is zero\n";
        ++failures;
    }
    return failures;
}

} // namespace

int main() {
    int count                   = 0;
    const cudaError_t count_err = cudaGetDeviceCount(&count);
    if (cuda_unavailable(count_err)) {
        std::cout << "SKIP: no usable CUDA device\n";
        return 77;
    }
    if (count_err != cudaSuccess) {
        std::cerr << "cudaGetDeviceCount failed: " << cudaGetErrorString(count_err) << '\n';
        return 1;
    }
    if (count == 0) {
        std::cout << "SKIP: no CUDA devices\n";
        return 77;
    }

    int failures = 0;

    ninfer::DeviceContext ctx(0);
    if (ctx.device != 0) {
        ++failures;
        std::cerr << "ctx.device expected 0, got " << ctx.device << '\n';
    }
    failures += check_context(ctx, "ctx");
    ctx.synchronize();

    const cudaStream_t original_stream = ctx.stream;
    const cudaStream_t original_copy   = ctx.copy_stream;
    ninfer::DeviceContext moved(std::move(ctx));
    if (ctx.stream != nullptr || ctx.load_stream != nullptr || ctx.copy_stream != nullptr) {
        ++failures;
        std::cerr << "move construction did not null source streams\n";
    }
    if (moved.stream != original_stream) {
        ++failures;
        std::cerr << "move construction did not transfer compute stream\n";
    }
    if (moved.copy_stream != original_copy) {
        ++failures;
        std::cerr << "move construction did not transfer copy stream\n";
    }
    failures += check_context(moved, "moved");

    {
        constexpr std::size_t kBytes = 32ULL << 20;
        ninfer::DeviceBuffer compute_buf(kBytes);
        ninfer::DeviceBuffer tiny(1);
        CUDA_CHECK(cudaMemsetAsync(tiny.p, 0, 1, moved.copy_stream));
        CUDA_CHECK(cudaStreamSynchronize(moved.copy_stream));
        for (int i = 0; i < 48; ++i) {
            CUDA_CHECK(cudaMemsetAsync(compute_buf.p, 0x5a, kBytes, moved.stream));
        }
        cudaEvent_t tail{};
        CUDA_CHECK(cudaEventCreateWithFlags(&tail, cudaEventDisableTiming));
        CUDA_CHECK(cudaEventRecord(tail, moved.stream));
        moved.order_copy_after_compute();
        unsigned char host = 0;
        CUDA_CHECK(cudaMemcpyAsync(&host, tiny.p, 1, cudaMemcpyDeviceToHost, moved.copy_stream));
        CUDA_CHECK(cudaStreamSynchronize(moved.copy_stream));
        const cudaError_t tail_ready = cudaEventQuery(tail);
        CUDA_CHECK(cudaEventDestroy(tail));
        if (tail_ready == cudaErrorNotReady) {
            ++failures;
            std::cerr << "order_copy_after_compute did not make copy_stream wait for queued "
                         "compute work\n";
        } else if (tail_ready != cudaSuccess) {
            ++failures;
            std::cerr << "order_copy_after_compute compute tail query failed: "
                      << cudaGetErrorString(tail_ready) << '\n';
        }
    }

    failures += expect_throws_device(count);

    ninfer::CudaEventTimer timer(moved);
    timer.start();
    moved.synchronize();
    const float elapsed_ms = timer.stop_ms();
    if (elapsed_ms < 0.0f) {
        ++failures;
        std::cerr << "timer elapsed time was negative\n";
    }

    return failures == 0 ? 0 : fail("device test failed");
}
