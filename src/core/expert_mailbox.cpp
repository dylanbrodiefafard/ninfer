#include "core/expert_mailbox.h"

#include <cuda_runtime_api.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstring>
#include <condition_variable>
#include <mutex>
#include <exception>
#include <limits>
#include <string>
#include <thread>
#include <vector>

namespace ninfer {
namespace {
void check(cudaError_t status, const char* operation) {
    if (status != cudaSuccess) {
        throw HipError(std::string("expert helper CUDA mailbox: ") + operation + ": " +
                       cudaGetErrorString(status));
    }
}

void check_failure(const ExpertWireFailure& error) {
    if (error.code != ExpertWireError::None) {
        throw HipError(
            "expert mailbox failure code=" + std::to_string(static_cast<unsigned>(error.code)) +
            " sequence=" + std::to_string(error.sequence) +
            " layer=" + std::to_string(error.layer) +
            " side=" + std::to_string(static_cast<unsigned>(error.side)));
    }
}

std::size_t stride(std::uint32_t capacity) {
    // Round each payload independently: IDs and weights for an odd N need not end at uint4.
    const auto aligned = [](std::size_t n) { return (n + 127) / 128 * 128; };
    return 256 + aligned(capacity * kExpertActivationBytes) +
           2 * aligned(std::size_t{capacity} * 40) + aligned(capacity * kExpertPartialBytes);
}

ExpertMailboxView view_at(void* base, std::uint32_t capacity) {
    auto* p = static_cast<std::byte*>(base);
    ExpertMailboxView result{};
    result.control      = reinterpret_cast<ExpertWireControl*>(p);
    result.cancellation = reinterpret_cast<ExpertWireCancellation*>(p + 128);
    result.failure      = reinterpret_cast<ExpertWireFailure*>(p + 256);
    result.peer_failure = reinterpret_cast<ExpertWireFailure*>(p + 384);
    result.capacity     = capacity;
    const auto aligned  = [](std::size_t n) { return (n + 127) / 128 * 128; };
    for (std::uint32_t i = 0; i < kExpertMailboxLayers; ++i) {
        auto* slot      = p + 512 + i * stride(capacity);
        result.slots[i] = {
            .request     = reinterpret_cast<ExpertWireHeader*>(slot),
            .response    = reinterpret_cast<ExpertWireHeader*>(slot + 128),
            .activations = slot + 256,
            .ids         = slot + 256 + aligned(capacity * kExpertActivationBytes),
            .weights     = slot + 256 + aligned(capacity * kExpertActivationBytes) +
                           aligned(std::size_t{capacity} * 40),
            .partial     = slot + 256 + aligned(capacity * kExpertActivationBytes) +
                           2 * aligned(std::size_t{capacity} * 40),
        };
    }
    return result;
}

struct CudaMappedMemory {
    void* host = nullptr;

    explicit CudaMappedMemory(std::size_t bytes) {
        check(cudaHostAlloc(&host, bytes, cudaHostAllocMapped | cudaHostAllocPortable), "allocate");
        std::memset(host, 0, bytes);
    }

    ~CudaMappedMemory() {
        if (cudaFreeHost(host) != cudaSuccess) { std::terminate(); }
    }
};
} // namespace

struct ExpertMailbox::Impl {
    std::uint32_t capacity;
    CudaMappedMemory storage;
    HipHostRegistration registration;
    ExpertMailboxView host, cuda, hip;
    cudaStream_t cuda_stream    = nullptr;
    const HipStream* hip_stream = nullptr;
    bool active                 = false;
    std::chrono::steady_clock::time_point round_deadline;
    std::atomic<bool> host_deadline{false};
    std::uint32_t host_deadline_layer = 0;
    std::jthread watchdog;
    // Nonzero high word makes the startup exchange qualify the complete 64-bit sequence.
    std::uint64_t sequence = std::uint64_t{1} << 32;

    Impl(std::uint32_t cap)
        : capacity(cap), storage(allocation_bytes(cap)),
          registration(storage.host, allocation_bytes(cap)), host(view_at(storage.host, cap)),
          hip(view_at(registration.alias(), cap)) {
        void* alias = nullptr;
        check(cudaHostGetDevicePointer(&alias, storage.host, 0), "CUDA alias");
        cuda = view_at(alias, cap);
        std::swap(hip.failure, hip.peer_failure);
        for (std::uint32_t i = 0; i < kExpertMailboxLayers; ++i) {
            for (auto* view : {&cuda, &hip}) {
                view->slots[i].host_activations = host.slots[i].activations;
                view->slots[i].host_ids         = host.slots[i].ids;
                view->slots[i].host_weights     = host.slots[i].weights;
                view->slots[i].host_partial     = host.slots[i].partial;
            }
        }
    }
};

std::size_t ExpertMailbox::allocation_bytes(std::uint32_t capacity) {
    if (capacity == 0 || capacity > kExpertMailboxMaxTokens) {
        throw std::invalid_argument("expert mailbox capacity must be in [1,48]");
    }
    return 512 + kExpertMailboxLayers * stride(capacity);
}

ExpertMailbox::ExpertMailbox(std::uint32_t capacity) : impl_(std::make_unique<Impl>(capacity)) {}

ExpertMailbox::~ExpertMailbox() { cancel_and_drain(); }

void ExpertMailbox::cancel_and_drain() noexcept {
    if (!impl_->active) { return; }
    cancel();
    try {
        finish();
    } catch (const HipError&) {
        // The error is preserved in failure(). Driver errors leaving work unjoined are fatal.
        if (impl_->active || (impl_->host.failure->code == ExpertWireError::None &&
                              impl_->host.peer_failure->code == ExpertWireError::None)) {
            std::terminate();
        }
    } catch (...) { std::terminate(); }
}

ExpertMailboxView ExpertMailbox::cuda_view() const noexcept { return impl_->cuda; }

ExpertMailboxView ExpertMailbox::hip_view() const noexcept { return impl_->hip; }

std::size_t ExpertMailbox::bytes() const noexcept {
    return 512 + kExpertMailboxLayers * stride(impl_->capacity);
}

std::uint64_t ExpertMailbox::begin(std::uint32_t tokens, void* cuda_stream,
                                   const HipStream& hip_stream) {
    if (impl_->active) { throw std::logic_error("expert mailbox round still active"); }
    if (tokens == 0 || tokens > impl_->capacity) {
        throw std::invalid_argument("expert mailbox round shape exceeds capacity");
    }
    if (impl_->sequence == std::numeric_limits<std::uint64_t>::max()) {
        throw HipError("expert mailbox sequence exhausted");
    }
    if (failure().code != ExpertWireError::None) { throw HipError("expert mailbox is poisoned"); }
    impl_->cuda_stream            = static_cast<cudaStream_t>(cuda_stream);
    impl_->hip_stream             = &hip_stream;
    impl_->host.control->tokens   = tokens;
    impl_->host.control->sequence = ++impl_->sequence;
    std::atomic_thread_fence(std::memory_order_release);
    impl_->active         = true;
    const auto deadline   = std::chrono::steady_clock::now() + std::chrono::seconds(30);
    impl_->round_deadline = deadline;
    impl_->watchdog       = std::jthread([this, deadline](const std::stop_token& stop) {
        std::mutex mutex;
        std::condition_variable_any wake;
        std::unique_lock lock(mutex);
        while (!stop.stop_requested()) {
            // Response sequences are single-writer completion/progress indicators. Details and
            // results remain unlicensed until finish() joins both runtime streams.
            std::uint32_t pending_layer = kExpertMailboxLayers - 1;
            for (std::uint32_t layer = 0; layer < kExpertMailboxLayers; ++layer) {
                if (std::atomic_ref<std::uint64_t>(impl_->host.slots[layer].response->sequence)
                        .load(std::memory_order_acquire) != impl_->sequence) {
                    pending_layer = layer;
                    break;
                }
            }
            const auto now = std::chrono::steady_clock::now();
            if (now >= deadline) {
                impl_->host_deadline_layer = pending_layer;
                impl_->host_deadline.store(true, std::memory_order_relaxed);
                std::atomic_ref<std::uint64_t>(impl_->host.cancellation->sequence)
                    .store(impl_->sequence, std::memory_order_release);
                return;
            }
            wake.wait_until(lock, stop, std::min(deadline, now + std::chrono::milliseconds(10)),
                            [] { return false; });
        }
    });
    return impl_->sequence;
}

void ExpertMailbox::cancel() noexcept {
    if (impl_->active) {
        std::atomic_ref<std::uint64_t>(impl_->host.cancellation->sequence)
            .store(impl_->sequence, std::memory_order_release);
    }
}

ExpertWireFailure ExpertMailbox::failure() const {
    // Read only at a joined boundary; each side release-publishes its code after the details.
    if (impl_->active) {
        throw std::logic_error("mailbox failure details require a joined boundary");
    }
    const auto& cuda = *impl_->host.failure;
    const auto& hip  = *impl_->host.peer_failure;
    if (cuda.code == ExpertWireError::None) { return hip; }
    if (hip.code == ExpertWireError::None) { return cuda; }
    // Simultaneous independent faults have no cross-device happens-before order. Choose the
    // earliest layer, CUDA on a tie; neither record can be overwritten by a cascading error.
    return hip.layer < cuda.layer ? hip : cuda;
}

void ExpertMailbox::finish() {
    if (!impl_->active) {
        check_failure(failure());
        return;
    }
    while (true) {
        const cudaError_t status = cudaStreamQuery(impl_->cuda_stream);
        if (status != cudaSuccess && status != cudaErrorNotReady) {
            cancel();
            // synchronize below must still join HIP before any owner can release aliases.
            impl_->hip_stream->synchronize();
            check(status, "asynchronous completion");
        }
        const bool hip_ready = impl_->hip_stream->ready();
        if (status == cudaSuccess && hip_ready) { break; }
        std::this_thread::sleep_for(std::chrono::milliseconds(2));
    }
    check(cudaStreamSynchronize(impl_->cuda_stream), "join CUDA round");
    impl_->hip_stream->synchronize();
    if (std::chrono::steady_clock::now() >= impl_->round_deadline) {
        impl_->host_deadline.store(true, std::memory_order_relaxed);
        cancel();
    }
    impl_->watchdog.request_stop();
    impl_->watchdog.join();
    impl_->active = false;
    if (failure().code == ExpertWireError::None &&
        std::atomic_ref<std::uint64_t>(impl_->host.cancellation->sequence)
                .load(std::memory_order_acquire) == impl_->sequence) {
        *impl_->host.failure =
            ExpertWireFailure{.code     = impl_->host_deadline.load(std::memory_order_relaxed)
                                              ? ExpertWireError::Deadline
                                              : ExpertWireError::Cancelled,
                              .side     = ExpertWireSide::Host,
                              .sequence = impl_->sequence,
                              .layer    = impl_->host_deadline_layer};
    }
    check_failure(failure());
}

void qualify_expert_mailbox(ExpertMailbox& mailbox, void* cuda_stream,
                            const HipStream& hip_stream) {
    struct DeviceAllocation {
        void* pointer = nullptr;

        explicit DeviceAllocation(std::size_t bytes) {
            check(cudaMalloc(&pointer, bytes), "nonce allocation");
        }

        ~DeviceAllocation() {
            if (cudaFree(pointer) != cudaSuccess) { std::terminate(); }
        }
    };

    constexpr std::uint32_t kNonce = 0xe3a97125U;
    std::vector<std::uint32_t> input(1280), metadata(10), output(2560);
    for (std::uint32_t i = 0; i < input.size(); ++i) { input[i] = kNonce ^ i; }
    for (std::uint32_t i = 0; i < metadata.size(); ++i) { metadata[i] = kNonce ^ (i << 8); }
    for (std::uint32_t i = 0; i < output.size(); ++i) { output[i] = ~kNonce ^ i; }
    DeviceAllocation x(kExpertActivationBytes), ids(40), weights(40), partial(kExpertPartialBytes),
        count(4);
    HipBuffer hx(kExpertActivationBytes), hi(40), hw(40), hp(kExpertPartialBytes), hn(4);
    const std::uint32_t n = 1;
    check(cudaMemcpy(x.pointer, input.data(), kExpertActivationBytes, cudaMemcpyHostToDevice),
          "nonce input");
    check(cudaMemcpy(ids.pointer, metadata.data(), 40, cudaMemcpyHostToDevice), "nonce ids");
    check(cudaMemcpy(weights.pointer, metadata.data(), 40, cudaMemcpyHostToDevice),
          "nonce weights");
    check(cudaMemcpy(count.pointer, &n, 4, cudaMemcpyHostToDevice), "nonce count");
    hip_copy_to_device(hp.data(), output.data(), kExpertPartialBytes, hip_stream);
    hip_stream.synchronize();
    try {
        static_cast<void>(mailbox.begin(1, cuda_stream, hip_stream));
        expert_mailbox_hip_import(mailbox.hip_view(), 0, hx.data(), hi.data(), hw.data(),
                                  static_cast<std::uint32_t*>(hn.data()), 1, hip_stream);
        expert_mailbox_hip_export(mailbox.hip_view(), 0, hp.data(),
                                  static_cast<std::uint32_t*>(hn.data()), 1, hip_stream);
        expert_mailbox_cuda_publish(mailbox.cuda_view(), 0, x.pointer, ids.pointer, weights.pointer,
                                    static_cast<std::uint32_t*>(count.pointer), cuda_stream);
        expert_mailbox_cuda_acquire(mailbox.cuda_view(), 0,
                                    static_cast<std::uint32_t*>(count.pointer), partial.pointer,
                                    cuda_stream);
        mailbox.finish();
    } catch (...) {
        mailbox.cancel_and_drain();
        throw;
    }
    std::vector<std::uint32_t> actual(2560);
    check(cudaMemcpy(actual.data(), partial.pointer, kExpertPartialBytes, cudaMemcpyDeviceToHost),
          "nonce response");
    if (actual != output) { throw HipError("expert helper startup nonce response mismatch"); }
    hip_copy_to_host(actual.data(), hx.data(), kExpertActivationBytes, hip_stream);
    hip_stream.synchronize();
    if (!std::equal(input.begin(), input.end(), actual.begin())) {
        throw HipError("expert helper startup nonce input mismatch");
    }
    for (const auto* plane : {&hi, &hw}) {
        hip_copy_to_host(actual.data(), plane->data(), 40, hip_stream);
        hip_stream.synchronize();
        if (!std::equal(metadata.begin(), metadata.end(), actual.begin())) {
            throw HipError("expert helper startup nonce metadata mismatch");
        }
    }
}
} // namespace ninfer
