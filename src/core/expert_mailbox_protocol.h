#pragma once

// Wire-only descriptor shared by CUDA and HIP translation units. No runtime headers, handles,
// host resource owners, expert IDs or placement policy enter this boundary.
#include <cstddef>
#include <cstdint>

namespace ninfer {
inline constexpr std::uint32_t kExpertWireVersion      = 1;
inline constexpr std::uint32_t kExpertMailboxLayers    = 48;
inline constexpr std::uint32_t kExpertMailboxMaxTokens = 48;
inline constexpr std::size_t kExpertActivationBytes    = 5120;
inline constexpr std::size_t kExpertMetadataBytes      = 80;
inline constexpr std::size_t kExpertPartialBytes       = 10240;
inline constexpr std::uint64_t kExpertLayerDeadlineNs  = 5'000'000'000ULL;

enum class ExpertWireStatus : std::uint32_t { Work = 1, Skip = 2, Failed = 3 };
enum class ExpertWireError : std::uint32_t { None, Cancelled, Deadline, Protocol, Helper, Runtime };
enum class ExpertWireSide : std::uint32_t { Cuda = 1, Hip = 2, Host = 3 };

// Each sequence has exactly one writer. Payload/status fields are protected by release/acquire
// of sequence; host never changes them while a round is active. Cancellation has its own line.
// After successful joins the host may clear benign cancellation codes and copy guards, never
// sequence flags. Each GPU owns its failure record while active; fatal causes supersede only
// cancellation, and are retained across rounds.
struct alignas(128) ExpertWireHeader {
    std::uint64_t sequence;
    std::uint32_t version, layer, tokens, compact_tokens;
    ExpertWireStatus status;
    // Raw DMA exclusion, not a packet flag. Request: CUDA's writer-active sequence (0 idle).
    // Response: HIP's failed-source seal sequence (0 unsealed). Separate single writers use
    // system sequentially-consistent stores/loads, not unsupported mapped-host atomic RMW.
    std::uint64_t copy_guard;
};

struct alignas(128) ExpertWireControl {
    std::uint64_t sequence;
    std::uint32_t tokens;
    std::uint32_t reserved;
};

struct alignas(128) ExpertWireCancellation {
    std::uint64_t sequence;
};

struct alignas(128) ExpertWireFailure {
    ExpertWireError code;
    ExpertWireSide side;
    std::uint64_t sequence;
    std::uint32_t layer;
};

static_assert(sizeof(ExpertWireHeader) == 128);
static_assert(sizeof(ExpertWireControl) == 128);
static_assert(sizeof(ExpertWireFailure) == 128);

// Runtime-neutral aliases: these addresses are valid only on the named runtime. Input planes
// are BF16[T,2560], I32[T,10], FP32[T,10]; output FP32[T,2560]. All bases are 16-byte aligned.
struct ExpertMailboxSlot {
    ExpertWireHeader* request;
    ExpertWireHeader* response;
    void* activations;
    void* ids;
    void* weights;
    void* partial;
    // CPU aliases are only for host-enqueued T48 request DMA, never kernel dereference.
    void* host_activations = nullptr;
    void* host_ids         = nullptr;
    void* host_weights     = nullptr;
};

struct ExpertMailboxView {
    ExpertWireControl* control;
    ExpertWireCancellation* cancellation;
    ExpertWireFailure* failure;
    ExpertWireFailure* peer_failure;
    ExpertMailboxSlot slots[kExpertMailboxLayers];
    std::uint32_t capacity;
};
} // namespace ninfer
