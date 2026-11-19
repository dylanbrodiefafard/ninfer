#pragma once

#include "core/expert_mailbox_protocol.h"
#include "core/hip_device.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>

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
    // Exception/teardown path: request cancellation and join. If both joins succeed and only
    // cancellation was latched, clear its device codes/seals at idle for reuse. Fatal errors
    // remain sticky; failed runtime joins retain backing until process exit. No output from
    // the discarded round is licensed.
    void cancel_and_drain() noexcept;
    // Coarse watchdog: 30 s until host cancellation, then drain bounded GPU polls. Throws HipError
    // with the latched sequence/layer/side after both join attempts, never licenses failed output.
    // A cancelled finish throws to prevent output licensing; cancel_and_drain may then recover
    // benign host cancellation. Protocol, deadline, helper and runtime failures stay sticky.
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
// Publish is one CTA and uses the current ingress sequence. Acquire spreads the response copy
// over ceil(capacity / 4) CTAs of four columns, each acquiring the sequence before it reads; it
// skips N=0 without polling/reading a response.
void expert_mailbox_cuda_publish(ExpertMailboxView view, std::uint32_t layer, const void* x,
                                 const void* ids, const void* weights, const std::uint32_t* count,
                                 void* stream);
void expert_mailbox_cuda_acquire(ExpertMailboxView view, std::uint32_t layer,
                                 const std::uint32_t* count, void* partial, void* stream);

// HIP side of a round, in stream order per layer L: expert_mailbox_hip_import(L, respond = L - 1
// for L > 0) acquires request L into caller-owned VRAM staging (x, ids and weights planes; count
// is always zero on SKIP/error, and downstream Op kernels must consume it), the helper Op writes
// rows [0, count) of response L in place through the slot's HIP partial alias, and the next
// import, or expert_mailbox_hip_respond after the round's last layer, publishes response L with
// a system-scope release. respond is exactly L - 1 (std::nullopt for L = 0), and count is the
// buffer the import of L - 1 wrote, unmodified since (the Op only reads it): the publication
// reports that count. T48 fixed-capacity request DMA is ordered after the import kernel.
void expert_mailbox_hip_import(ExpertMailboxView view, std::uint32_t layer,
                               std::optional<std::uint32_t> respond, void* x, void* ids,
                               void* weights, std::uint32_t* count, std::uint32_t tokens,
                               const HipStream& stream);
void expert_mailbox_hip_respond(ExpertMailboxView view, std::uint32_t layer,
                                const std::uint32_t* count, std::uint32_t tokens,
                                const HipStream& stream);
// Startup qualification's stand-in for the helper Op: writes rows [0, *count) of a VRAM partial
// into response `layer`'s payload through the HIP alias with plain stores, unpublished.
void expert_mailbox_hip_store_partial(ExpertMailboxView view, std::uint32_t layer,
                                      const void* partial, const std::uint32_t* count,
                                      std::uint32_t tokens, const HipStream& stream);
// Exact bounded per-record byte checksum; result is one U64 per span, no expert mathematics.
void expert_mailbox_hip_checksum(const void* data, std::size_t bytes, std::uint64_t* result,
                                 const HipStream& stream);

} // namespace ninfer
