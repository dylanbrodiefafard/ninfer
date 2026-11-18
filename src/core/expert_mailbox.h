#pragma once

#include "core/expert_mailbox_protocol.h"
#include "core/hip_device.h"

#include <cstddef>
#include <cstdint>
#include <memory>

namespace ninfer {
class ExpertMailbox {
public:
    // CUDA context and the selected HIP context must already exist. No expert-store registration.
    explicit ExpertMailbox(std::uint32_t capacity);
    ~ExpertMailbox();
    ExpertMailbox(const ExpertMailbox&)            = delete;
    ExpertMailbox& operator=(const ExpertMailbox&) = delete;
    [[nodiscard]] ExpertMailboxView cuda_view() const noexcept;
    [[nodiscard]] ExpertMailboxView hip_view() const noexcept;
    [[nodiscard]] std::size_t bytes() const noexcept;
    [[nodiscard]] static std::size_t allocation_bytes(std::uint32_t capacity);
    // Single host owner, idle boundary only; flags are never reset. Stream owners must outlive
    // this mailbox. GPU submissions follow begin; finish joins both before slot/ingress reuse.
    [[nodiscard]] std::uint64_t begin(std::uint32_t tokens, void* cuda_stream,
                                      const HipStream& hip_stream);
    void cancel() noexcept;
    // Exception/teardown path: cancel and join, consuming only an already-latched protocol error.
    // Failed runtime joins retain the backing until process exit, never free reachable memory.
    void cancel_and_drain() noexcept;
    // Coarse watchdog: 30 s until host cancellation, then drain bounded GPU polls. Throws HipError
    // with the latched sequence/layer/side after both join attempts, never licenses failed output.
    // Failure is sticky: later finish calls also fail, including after cancel_and_drain.
    void finish();
    [[nodiscard]] ExpertWireFailure failure() const;
    // True at idle/joined boundaries. After a failed drain, enclosing owners must also be retained.
    [[nodiscard]] bool drained() const noexcept;
private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

// Startup-only exact nonce exchange qualifying both aliases, payload directions and 64-bit
// sequence atomics before graph capture/readiness. Leaves the mailbox idle; failures poison it.
void qualify_expert_mailbox(ExpertMailbox& mailbox, void* cuda_stream, const HipStream& hip_stream);

// Capturable raw transfers. These own no expert semantics. Input/output are CUDA VRAM planes.
// acquire/copy skips N=0 without polling/reading a response. Publish uses current ingress sequence.
void expert_mailbox_cuda_publish(ExpertMailboxView view, std::uint32_t layer, const void* x,
                                 const void* ids, const void* weights, const std::uint32_t* count,
                                 void* stream);
void expert_mailbox_cuda_acquire(ExpertMailboxView view, std::uint32_t layer,
                                 const std::uint32_t* count, void* partial, void* stream);

// HIP import/export use caller-owned VRAM staging. count is always zero on SKIP/error; downstream
// Op kernels must consume it. T48 fixed-capacity DMA is ordered between acquire/publish kernels.
void expert_mailbox_hip_import(ExpertMailboxView view, std::uint32_t layer, void* x, void* ids,
                               void* weights, std::uint32_t* count, std::uint32_t tokens,
                               const HipStream& stream);
void expert_mailbox_hip_export(ExpertMailboxView view, std::uint32_t layer, const void* partial,
                               const std::uint32_t* count, std::uint32_t tokens,
                               const HipStream& stream);
// Raw metadata layout transform: zip two U32 planes of ten fields per compact row into U64
// entries (low word from first, high word from second). No ID/weight interpretation.
void expert_mailbox_hip_interleave(const void* first, const void* second, void* pairs,
                                   const std::uint32_t* count, const HipStream& stream);
// Exact bounded per-record byte checksum; result is one U64 per span, no expert mathematics.
void expert_mailbox_hip_checksum(const void* data, std::size_t bytes, std::uint64_t* result,
                                 const HipStream& stream);

} // namespace ninfer
