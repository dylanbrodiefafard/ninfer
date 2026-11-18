#include "runtime/expert_helper_device.h"
#include <cuda_runtime_api.h>

#include <chrono>
#include <cstdio>
#include <set>
#include <stdexcept>
#include <string>

using namespace ninfer;
using namespace ninfer::targets::qwen3_8_flash_next;

namespace {
void check(cudaError_t status) {
    if (status != cudaSuccess) { throw std::runtime_error(cudaGetErrorString(status)); }
}

struct Primary {
    cudaStream_t stream  = nullptr;
    std::uint32_t* count = nullptr;

    Primary() {
        check(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
        try {
            void* allocated = nullptr;
            check(cudaMalloc(&allocated, sizeof(std::uint32_t)));
            count = static_cast<std::uint32_t*>(allocated);
            check(cudaMemsetAsync(count, 0, sizeof(std::uint32_t), stream));
            check(cudaStreamSynchronize(stream));
        } catch (...) {
            if (count != nullptr && cudaFree(count) != cudaSuccess) { std::terminate(); }
            if (cudaStreamDestroy(stream) != cudaSuccess) { std::terminate(); }
            throw;
        }
    }

    ~Primary() {
        if (cudaStreamSynchronize(stream) != cudaSuccess || cudaFree(count) != cudaSuccess ||
            cudaStreamDestroy(stream) != cudaSuccess) {
            std::terminate();
        }
    }
};
} // namespace

int main() {
    try {
        HipDevice device("0000:13:00.0");
        Primary primary;
        HipStream hip;
        HipBuffer tables(48 * 512 * 8), scales(48 * 512 * 3 * 4);
        hip_zero(tables.data(), tables.size(), hip);
        hip_zero(scales.data(), scales.size(), hip);
        hip.synchronize();
        ExpertMailbox mailbox(48);
        std::array<ExpertHelperBank, 48> banks{};
        for (std::size_t l = 0; l < 48; ++l) {
            banks[l] = {.records       = static_cast<const std::uint64_t*>(tables.data()) + l * 512,
                        .matrix_scales = static_cast<const float*>(scales.data()) + l * 512 * 3,
                        .f3_books      = {0x76543210U, 0x76543210U, 0x76543210U}};
        }
        ExpertHelperExecution execution(mailbox.hip_view(), banks, 12, true, hip);
        std::set<std::uint32_t> shapes;
        for (std::uint32_t c = 1; c <= 12; ++c) {
            for (std::uint32_t w = 1; w <= 4; ++w) { shapes.insert(c * w); }
        }
        for (unsigned replay = 0; replay < 2; ++replay) {
            for (const auto t : shapes) {
                const auto start = std::chrono::steady_clock::now();
                static_cast<void>(mailbox.begin(t, primary.stream, hip));
                try {
                    execution.launch(t);
                    const auto queued = std::chrono::steady_clock::now();
                    for (std::uint32_t l = 0; l < 48; ++l) {
                        expert_mailbox_cuda_publish(mailbox.cuda_view(), l, nullptr, nullptr,
                                                    nullptr, primary.count, primary.stream);
                    }
                    const auto drain_start = std::chrono::steady_clock::now();
                    execution.join();
                    mailbox.finish();
                    const auto done = std::chrono::steady_clock::now();
                    if (t == 1 || t == 4 || t == 20 || t == 48) {
                        std::printf(
                            "helper all-SKIP T=%u replay=%u prequeue_us=%.3f drain_us=%.3f\n", t,
                            replay,
                            std::chrono::duration<double, std::micro>(queued - start).count(),
                            std::chrono::duration<double, std::micro>(done - drain_start).count());
                    }
                } catch (...) {
                    mailbox.cancel_and_drain();
                    throw;
                }
            }
        }
        static_cast<void>(mailbox.begin(48, primary.stream, hip));
        try {
            execution.launch(48);
            mailbox.cancel();
            execution.join();
            bool cancelled = false;
            try {
                mailbox.finish();
            } catch (const HipError&) { cancelled = true; }
            if (!cancelled) {
                throw std::runtime_error("helper graph cancellation was not reported");
            }
        } catch (...) {
            mailbox.cancel_and_drain();
            throw;
        }
        mailbox.cancel_and_drain(); // Recover the explicit cancelled finish at its idle boundary.
        execution.retire_joined_round();
        for (const auto tokens : {1U, 48U}) {
            bool original = false;
            try {
                try {
                    static_cast<void>(mailbox.begin(tokens, primary.stream, hip));
                    execution.launch(tokens);
                    expert_mailbox_cuda_publish(mailbox.cuda_view(), 0, nullptr, nullptr, nullptr,
                                                primary.count, primary.stream);
                    throw std::runtime_error("injected host gather failure");
                } catch (...) {
                    mailbox.cancel_and_drain();
                    if (mailbox.drained()) { execution.retire_joined_round(); }
                    throw;
                }
            } catch (const std::runtime_error& error) {
                original = std::string(error.what()) == "injected host gather failure";
            }
            if (!original || !mailbox.drained() ||
                mailbox.failure().code != ExpertWireError::None) {
                throw std::runtime_error(
                    "helper host-abort drain lost its cause or stayed poisoned");
            }
            static_cast<void>(mailbox.begin(tokens, primary.stream, hip));
            execution.launch(tokens);
            for (std::uint32_t layer = 0; layer < 48; ++layer) {
                expert_mailbox_cuda_publish(mailbox.cuda_view(), layer, nullptr, nullptr, nullptr,
                                            primary.count, primary.stream);
            }
            mailbox.finish();
            execution.retire_joined_round();
        }
        std::printf("production helper graph shapes=%zu replays=2 layers=48 all-SKIP drain=ok "
                    "cancellation/host-exception reuse=ok\n",
                    shapes.size());
        return 0;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "%s\n", e.what());
        return 1;
    }
}
