#include "core/expert_mailbox.h"

#include <cuda_runtime_api.h>

#include <array>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

using namespace ninfer;
void mailbox_test_nonce(const void*, const void*, const void*, const void*, void*,
                        const std::uint32_t*, std::uint32_t*, const HipStream&);
void mailbox_test_delay(const HipStream&);
void mailbox_test_fail(ExpertMailboxView, const HipStream&);
cudaError_t mailbox_test_cuda_error(cudaStream_t);
cudaError_t mailbox_test_claim_writer(ExpertMailboxView, cudaStream_t);

namespace {
void cuda_check(cudaError_t status) {
    if (status != cudaSuccess) { throw std::runtime_error(cudaGetErrorString(status)); }
}

void require(bool valid, const char* message) {
    if (!valid) { throw std::runtime_error(message); }
}

struct CudaStream {
    cudaStream_t value = nullptr;

    CudaStream() { cuda_check(cudaStreamCreateWithFlags(&value, cudaStreamNonBlocking)); }

    ~CudaStream() {
        if (cudaStreamSynchronize(value) != cudaSuccess ||
            cudaStreamDestroy(value) != cudaSuccess) {
            std::terminate();
        }
    }
};

struct CudaBuffer {
    void* data = nullptr;

    explicit CudaBuffer(std::size_t bytes) { cuda_check(cudaMalloc(&data, bytes)); }

    ~CudaBuffer() {
        if (cudaFree(data) != cudaSuccess) { std::terminate(); }
    }
};

struct CudaGraph {
    cudaGraph_t graph    = nullptr;
    cudaGraphExec_t exec = nullptr;

    ~CudaGraph() {
        if (exec != nullptr && cudaGraphExecDestroy(exec) != cudaSuccess) { std::terminate(); }
        if (graph != nullptr && cudaGraphDestroy(graph) != cudaSuccess) { std::terminate(); }
    }
};

struct Panels {
    CudaBuffer x{48 * 48 * kExpertActivationBytes}, ids{48 * 48 * 40}, weights{48 * 48 * 40},
        out{48 * 48 * kExpertPartialBytes}, count{4};
    HipBuffer hx{48 * kExpertActivationBytes}, hi{48 * 40}, hw{48 * 40}, pairs{48 * 80},
        ho{48 * kExpertPartialBytes}, hn{4}, bad{4};

    void initialize(const HipStream& stream) {
        for (auto* p : {&hx, &hi, &hw, &pairs, &ho, &hn, &bad}) {
            hip_zero(p->data(), p->size(), stream);
        }
        stream.synchronize();
    }

    void input(std::uint32_t n, std::uint32_t seed) {
        std::vector<std::uint32_t> data(48 * 48 * 1280);
        for (std::uint32_t l = 0; l < 48; ++l) {
            for (std::uint32_t i = 0; i < 48 * 1280; ++i) {
                data[l * 48 * 1280 + i] = seed ^ (l << 16) ^ i;
            }
        }
        cuda_check(cudaMemcpy(x.data, data.data(), data.size() * 4, cudaMemcpyHostToDevice));
        for (std::uint32_t l = 0; l < 48; ++l) {
            for (std::uint32_t i = 0; i < 48 * 10; ++i) {
                data[l * 48 * 10 + i] = seed ^ (l << 16) ^ i ^ 0x12340000U;
            }
        }
        cuda_check(cudaMemcpy(ids.data, data.data(), 48 * 48 * 40, cudaMemcpyHostToDevice));
        for (std::uint32_t l = 0; l < 48; ++l) {
            for (std::uint32_t i = 0; i < 48 * 10; ++i) {
                data[l * 48 * 10 + i] = seed ^ (l << 16) ^ i ^ 0x56780000U;
            }
        }
        cuda_check(cudaMemcpy(weights.data, data.data(), 48 * 48 * 40, cudaMemcpyHostToDevice));
        cuda_check(cudaMemcpy(count.data, &n, 4, cudaMemcpyHostToDevice));
        cuda_check(cudaMemset(out.data, 0xcd, 48 * 48 * kExpertPartialBytes));
    }

    void output(std::uint32_t n, std::uint32_t seed, const HipStream& stream,
                std::uint32_t layers = 1) {
        std::uint32_t bad_count = 0;
        hip_copy_to_host(&bad_count, bad.data(), 4, stream);
        stream.synchronize();
        require(bad_count == 0, "HIP input nonce mismatch");
        std::vector<std::uint32_t> data(48 * 48 * 2560);
        cuda_check(cudaMemcpy(data.data(), out.data, data.size() * 4, cudaMemcpyDeviceToHost));
        for (std::uint32_t l = 0; l < 48; ++l) {
            for (std::uint32_t i = 0; i < 48 * 2560; ++i) {
                require(data[l * 48 * 2560 + i] == (l < layers && i < n * 2560
                                                        ? seed ^ (l << 16) ^ i ^ 0xa5a5a5a5U
                                                        : 0xcdcdcdcdU),
                        "CUDA output nonce/untouched tail mismatch");
            }
        }
    }

    void helper(ExpertMailbox& box, std::uint32_t layer, std::uint32_t t, const HipStream& stream) {
        expert_mailbox_hip_import(box.hip_view(), layer, hx.data(), hi.data(), hw.data(),
                                  static_cast<std::uint32_t*>(hn.data()), t, stream);
        expert_mailbox_hip_interleave(hi.data(), hw.data(), pairs.data(),
                                      static_cast<std::uint32_t*>(hn.data()), stream);
        mailbox_test_nonce(hx.data(), hi.data(), hw.data(), pairs.data(), ho.data(),
                           static_cast<std::uint32_t*>(hn.data()),
                           static_cast<std::uint32_t*>(bad.data()), stream);
        expert_mailbox_hip_export(box.hip_view(), layer, ho.data(),
                                  static_cast<std::uint32_t*>(hn.data()), t, stream);
    }

    void primary(ExpertMailbox& box, std::uint32_t layer, cudaStream_t stream) {
        const auto layer_index = static_cast<std::size_t>(layer);
        expert_mailbox_cuda_publish(box.cuda_view(), layer,
                                    static_cast<std::byte*>(x.data) +
                                        layer_index * 48 * kExpertActivationBytes,
                                    static_cast<std::byte*>(ids.data) + layer_index * 48 * 40,
                                    static_cast<std::byte*>(weights.data) + layer_index * 48 * 40,
                                    static_cast<std::uint32_t*>(count.data), stream);
        expert_mailbox_cuda_acquire(
            box.cuda_view(), layer, static_cast<std::uint32_t*>(count.data),
            static_cast<std::byte*>(out.data) + layer_index * 48 * kExpertPartialBytes, stream);
    }
};

void replay() {
    CudaStream cuda;
    HipStream hip;
    Panels panels;
    panels.initialize(hip);
    ExpertMailbox box(48);
    qualify_expert_mailbox(box, cuda.value, hip);
    // Changing packed T reuses the same slots/ingress; graph shape is fixed within each replay.
    for (const auto t : {1U, 4U, 20U, 48U, 4U, 1U}) {
        HipGraph helper(hip);
        for (std::uint32_t l = 0; l < 48; ++l) { panels.helper(box, l, t, hip); }
        helper.instantiate();
        CudaGraph primary;
        cuda_check(cudaStreamBeginCapture(cuda.value, cudaStreamCaptureModeThreadLocal));
        for (std::uint32_t l = 0; l < 48; ++l) { panels.primary(box, l, cuda.value); }
        cuda_check(cudaStreamEndCapture(cuda.value, &primary.graph));
        cuda_check(cudaGraphInstantiate(&primary.exec, primary.graph, 0));
        for (std::uint32_t r = 0; r < 7; ++r) {
            const auto n    = r % 3 == 1 ? (t == 48 ? 47U : t) : 0U;
            const auto seed = 0xcab00000U + r;
            panels.input(n, seed);
            const auto sequence = box.begin(t, cuda.value, hip);
            require(sequence > (std::uint64_t{1} << 32), "64-bit sequence high word lost");
            helper.launch(hip);
            cuda_check(cudaGraphLaunch(primary.exec, cuda.value));
            box.finish();
            panels.output(n, seed, hip, 48);
        }
        std::printf("mailbox T=%u graph replays=7 layers=48 SKIP/work/last-slot=ok bytes=%zu\n", t,
                    box.bytes());
    }
}

void failure_case(const std::string& mode, std::uint32_t t = 1) {
    CudaStream cuda;
    HipStream hip;
    Panels panels;
    panels.initialize(hip);
    ExpertMailbox box(t);
    panels.input(mode == "invalid" ? t + 1 : 1, 0xabcdef01U);
    const auto start = std::chrono::steady_clock::now();
    static_cast<void>(box.begin(t, cuda.value, hip));
    if (mode == "failed") { mailbox_test_fail(box.hip_view(), hip); }
    if (mode == "delayed") { mailbox_test_delay(hip); }
    if (mode == "seal-timeout") {
        cuda_check(mailbox_test_claim_writer(box.cuda_view(), cuda.value));
        cuda_check(cudaStreamSynchronize(cuda.value));
        box.cancel();
    }
    if (mode != "cuda-deadline") { panels.helper(box, 0, t, hip); }
    if (mode == "late-cancel") {
        expert_mailbox_cuda_publish(box.cuda_view(), 0, panels.x.data, panels.ids.data,
                                    panels.weights.data,
                                    static_cast<std::uint32_t*>(panels.count.data), cuda.value);
        cuda_check(cudaStreamSynchronize(cuda.value));
        hip.synchronize(); // response exists, but CUDA has not acquired it
        box.cancel();
        expert_mailbox_cuda_acquire(box.cuda_view(), 0,
                                    static_cast<std::uint32_t*>(panels.count.data), panels.out.data,
                                    cuda.value);
    } else if (mode != "cancel" && mode != "hip-deadline" && mode != "seal-timeout") {
        panels.primary(box, 0, cuda.value);
    }
    if (mode == "cancel") { box.cancel(); }
    if (mode == "hip-deadline") {
        hip.synchronize(); // late CUDA must not write into a failed-acquire fixed DMA source
        panels.primary(box, 0, cuda.value);
    }
    bool threw = false;
    try {
        box.finish();
    } catch (const HipError&) { threw = true; }
    const bool delayed = mode == "delayed";
    require(threw != delayed, "failure/normal completion contract");
    if (delayed) {
        panels.output(1, 0xabcdef01U, hip);
    } else {
        require(box.failure().code != ExpertWireError::None, "failure was not latched");
        if (mode == "seal-timeout") {
            require(box.failure().code == ExpertWireError::Deadline,
                    "seal deadline was hidden by benign cancellation");
            box.cancel_and_drain();
            require(box.failure().code == ExpertWireError::Deadline,
                    "host abort erased a fatal seal deadline");
        }
        bool sticky = false;
        try {
            box.finish();
        } catch (const HipError&) { sticky = true; }
        require(sticky, "joined failure was licensed by a later finish");
        panels.output(0, 0, hip); // failed acquire must not overwrite any output byte
        bool poisoned = false;
        try {
            static_cast<void>(box.begin(1, cuda.value, hip));
        } catch (const HipError&) { poisoned = true; }
        require(poisoned, "failed mailbox reused");
    }
    const double elapsed =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
    require(elapsed < 8.0, "bounded failure took over eight seconds");
    std::printf("mailbox %s T=%u seconds=%.3f status=ok\n", mode.c_str(), t, elapsed);
}

void teardown() {
    CudaStream cuda;
    HipStream hip;
    Panels panels;
    panels.initialize(hip);
    panels.input(1, 0x12345678U);
    {
        ExpertMailbox box(48);
        HipGraph helper(hip);
        for (std::uint32_t l = 0; l < 48; ++l) { panels.helper(box, l, 48, hip); }
        helper.instantiate();
        static_cast<void>(box.begin(48, cuda.value, hip));
        helper.launch(hip);
        panels.primary(box, 0, cuda.value);
        cuda_check(cudaStreamSynchronize(cuda.value));
        box.cancel_and_drain(); // CUDA graph B never submits layers 1..47
    }
    panels.output(1, 0x12345678U, hip);
    {
        ExpertMailbox box(1);
        static_cast<void>(box.begin(1, cuda.value, hip));
        panels.helper(box, 0, 1, hip);
        // Active destructor must cancel and join before alias unregistration.
    }
    require(hip.ready(), "active teardown left HIP work running");
    std::puts("mailbox missing-graph-B and active cancellation teardown=ok");
}

void host_exception_reuse() {
    CudaStream cuda;
    HipStream hip;
    Panels panels;
    panels.initialize(hip);
    for (const auto tokens : {1U, 48U}) {
        ExpertMailbox box(tokens);
        HipGraph helper(hip);
        for (std::uint32_t layer = 0; layer < 48; ++layer) {
            panels.helper(box, layer, tokens, hip);
        }
        helper.instantiate();
        panels.input(1, 0x10203040U);
        bool original = false;
        try {
            try {
                static_cast<void>(box.begin(tokens, cuda.value, hip));
                helper.launch(hip);
                panels.primary(box, 0, cuda.value); // Graph A exists; graph B will not submit.
                throw std::runtime_error("injected host n-gram read failure");
            } catch (...) {
                box.cancel_and_drain();
                throw;
            }
        } catch (const std::runtime_error& error) {
            original = std::string(error.what()) == "injected host n-gram read failure";
        }
        require(original && box.drained() && box.failure().code == ExpertWireError::None,
                "host abort lost its cause or poisoned joined mailbox state");
        panels.input(1, 0x55667788U);
        static_cast<void>(box.begin(tokens, cuda.value, hip));
        helper.launch(hip);
        for (std::uint32_t layer = 0; layer < 48; ++layer) {
            panels.primary(box, layer, cuda.value);
        }
        box.finish();
        panels.output(1, 0x55667788U, hip, 48);
        std::printf("mailbox host exception/cancel/reuse T=%u nonce=ok\n", tokens);
    }
}

void late_finish() {
    CudaStream cuda;
    HipStream hip;
    Panels panels;
    panels.initialize(hip);
    ExpertMailbox box(1);
    panels.input(1, 0x31415926U);
    for (int round = 0; round < 2; ++round) {
        static_cast<void>(box.begin(1, cuda.value, hip));
        for (std::uint32_t layer = 0; layer < 48; ++layer) {
            panels.helper(box, layer, 1, hip);
            panels.primary(box, layer, cuda.value);
        }
        cuda_check(cudaStreamSynchronize(cuda.value));
        hip.synchronize();
        if (round == 0) { std::this_thread::sleep_for(std::chrono::seconds(31)); }
        box.finish();
        require(box.failure().code == ExpertWireError::None,
                "completed round poisoned by host stall");
        panels.output(1, 0x31415926U, hip, 48);
    }
    std::puts("mailbox late finish and persistent-watchdog rearm=ok");
}

void asynchronous_error() {
    // Intentional context poisoning runs in its own CTest process. CUDA handles cannot be
    // destroyed through the failed context; retain them until process exit, as production does
    // for allocations whose runtime join failed. HIP must still drain normally.
    cudaStream_t cuda = nullptr;
    cuda_check(cudaStreamCreateWithFlags(&cuda, cudaStreamNonBlocking));
    HipStream hip;
    {
        ExpertMailbox box(1);
        static_cast<void>(box.begin(1, cuda, hip));
        mailbox_test_delay(hip); // The CUDA fault must still join independently queued HIP work.
        const auto launch = mailbox_test_cuda_error(cuda);
        require(launch == cudaSuccess || launch == cudaErrorIllegalAddress,
                "unexpected fault-injection launch status");
        bool typed = false;
        try {
            box.finish();
        } catch (const HipError&) { typed = true; }
        require(typed && box.failure().code == ExpertWireError::Runtime &&
                    box.failure().side == ExpertWireSide::Host && !box.drained(),
                "asynchronous CUDA error did not become a joined-watchdog Host failure");
        box.cancel_and_drain();
        bool sticky = false;
        try {
            box.finish();
        } catch (const HipError&) { sticky = true; }
        require(sticky, "asynchronous failure was not sticky");
    }
    require(hip.ready(), "asynchronous CUDA error left HIP work pending");
    std::puts("mailbox asynchronous CUDA failure and non-terminating teardown=ok");
}

void round_deadline() {
    CudaStream cuda;
    HipStream hip;
    ExpertMailbox box(1);
    static_cast<void>(box.begin(1, cuda.value, hip));
    std::this_thread::sleep_for(std::chrono::seconds(31));
    bool deadline = false;
    try {
        box.finish();
    } catch (const HipError&) { deadline = box.failure().code == ExpertWireError::Deadline; }
    require(deadline, "round deadline measured from begin was not preserved");
    std::puts("mailbox activation-to-finish round deadline=ok");
}
} // namespace

int main(int argc, char** argv) {
    try {
        HipDevice device("0000:13:00.0");
        const auto info = device.info();
        std::printf("HIP %s %s %s runtime=%d\n", info.pci.c_str(), info.name.c_str(),
                    info.architecture.c_str(), info.runtime_version);
        if (argc == 2 && std::string(argv[1]) == "round-deadline") {
            round_deadline();
        } else if (argc == 2 && std::string(argv[1]) == "late-finish") {
            late_finish();
        } else if (argc == 2 && std::string(argv[1]) == "host-exception-reuse") {
            host_exception_reuse();
        } else if (argc == 2 && std::string(argv[1]) == "async-error") {
            asynchronous_error();
        } else if (argc == 2) {
            failure_case(argv[1]);
        } else {
            replay();
            for (const auto* mode : {"delayed", "failed", "cancel", "late-cancel", "invalid",
                                     "cuda-deadline", "hip-deadline", "seal-timeout"}) {
                failure_case(mode);
                failure_case(mode, 48);
            }
            teardown();
            host_exception_reuse();
            round_deadline();
            { ExpertMailbox idle(1); }
            std::puts("mailbox idle teardown=ok");
        }
        return EXIT_SUCCESS;
    } catch (const std::exception& error) {
        std::fprintf(stderr, "%s\n", error.what());
        return EXIT_FAILURE;
    }
}
