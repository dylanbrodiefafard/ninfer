#pragma once

#include "core/paged_kv_cache.h"
#include "targets/qwen3_6/impl/runtime/context_checkpoint.h"
#include "targets/qwen3_6/impl/runtime/prefix_identity.h"

#include "ninfer/types.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace ninfer::targets::qwen3_6::detail {

inline constexpr char kDiskFingerprintMagic[8] = {'N', 'I', 'D', 'K', 'F', 'P', '0', '1'};
inline constexpr char kDiskManifestMagic[8]    = {'N', 'I', 'D', 'K', 'M', 'N', '0', '1'};
inline constexpr char kDiskMetaMagic[8]        = {'N', 'I', 'D', 'K', 'M', 'T', '0', '1'};
inline constexpr char kDiskPageMagic[8]        = {'N', 'I', 'D', 'K', 'P', 'G', '0', '1'};
inline constexpr char kDiskTombstoneMagic[8]    = {'N', 'I', 'D', 'K', 'T', 'B', '0', '1'};
inline constexpr char kDiskPackSetMagic[8]      = {'N', 'I', 'D', 'K', 'P', 'S', '0', '1'};
inline constexpr char kDiskObjectMapMagic[8]    = {'N', 'I', 'D', 'K', 'O', 'M', '0', '1'};

inline constexpr std::uint32_t kDiskFormatVersion     = 5;
inline constexpr std::uint32_t kDiskPageHeaderBytes   = 16;
inline constexpr std::uint32_t kDiskPageIoAlignment   = 4096;
inline constexpr std::uint32_t kDiskCodecHeaderBytes  = 24;
inline constexpr std::uint32_t kDiskStatePayloadOffset = 4096;
inline constexpr std::uint32_t kDiskFingerprintPlaneBytes = 20;

enum class DiskCodec : std::uint8_t {
    Raw  = 0,
    Zstd = 1,
};

enum class DiskStateKind : std::uint8_t {
    CurrentGdn      = 1,
    RewriteGdn      = 2,
    RollbackGdn     = 3,
    LadderGdn       = 4,
    TailHidden      = 5,
    RewriteHidden   = 6,
    RollbackHidden  = 7,
    LadderHidden    = 8,
    DflashLocal     = 9,
    DflashRewrite   = 10,
    DflashRollback  = 11,
    DflashLadder    = 12,
};

enum class DiskObjectKind : std::uint8_t {
    Main     = 0,
    Backend  = 1,
    State    = 2,
    Ledger   = 3,
    Identity = 4,
};

struct DiskCodecHeader {
    DiskCodec codec                 = DiskCodec::Raw;
    DiskStateKind kind              = DiskStateKind::CurrentGdn;
    std::uint64_t uncompressed_bytes = 0;
    std::uint64_t compressed_bytes   = 0;
};

struct DiskCheckpointSlot {
    std::uint32_t frontier = 0;
    PrefixHash128 hash{};
    ContextCheckpointKind kind = ContextCheckpointKind::Ladder;
    std::uint64_t gdn_id       = 0;
    std::uint64_t hidden_id    = 0;
    std::uint64_t dflash_id    = 0;
};

// Describes one plane in the canonical packed logical-page image. Physical resident-page
// capacity, allocation alignment, and device strides are intentionally absent: none of them is
// present in a disk page record, and restore scatters the canonical bytes through the destination
// pool's current geometry.
struct DiskPlaneSchema {
    DType dtype                 = DType::BF16;
    PagedKVPlaneOrder order     = PagedKVPlaneOrder::PageMajor;
    std::int32_t leading_extent = 0;
    std::int32_t head_extent    = 0;
    std::uint64_t page_bytes    = 0;

    [[nodiscard]] bool operator==(const DiskPlaneSchema&) const = default;
};

struct DiskFingerprint {
    std::string model_id;
    std::string weights_id;
    KvCacheStorage kv_cache          = KvCacheStorage::Nvfp4;
    SpeculativeBackend speculative   = SpeculativeBackend::None;
    std::uint32_t page_size          = static_cast<std::uint32_t>(kPagedKVPageSize);
    std::uint64_t gdn_conv_bytes     = 0;
    std::uint64_t gdn_recurrent_bytes = 0;
    std::uint64_t cyclic_lane_bytes  = 0;
    std::uint32_t cyclic_layers      = 0;
    std::uint32_t cyclic_capacity    = 0;
    std::uint32_t cyclic_padded      = 0;
    std::int32_t cyclic_kv_heads     = 0;
    std::int32_t cyclic_head_dim     = 0;
    std::int32_t cyclic_lane_capacity = 0;
    std::uint32_t snapshot_version   = kDiskFormatVersion;
    std::vector<DiskPlaneSchema> text_planes;
    std::vector<DiskPlaneSchema> backend_planes;
};

struct DiskMeta {
    std::uint64_t entry_id            = 0;
    std::uint32_t execution_frontier  = 0;
    std::uint32_t ledger_frontier     = 0;
    std::int32_t rope_delta           = 0;
    std::uint32_t text_kv_valid       = 0;
    std::uint32_t mtp_kv_valid        = 0;
    std::uint32_t dflash_context_frontier = 0;
    bool tail_hidden_valid            = false;
    bool rewrite_valid                = false;
    RewriteCheckpointKind rewrite_kind = RewriteCheckpointKind::TurnClosure;
    bool hash_c_valid                 = false;
    std::uint32_t rewrite_frontier    = 0;
    PrefixHash128 hash_f{};
    PrefixHash128 hash_c{};
    std::uint64_t ledger_id           = 0;
    std::uint64_t identity_id         = 0;
    std::uint64_t current_gdn_id      = 0;
    std::uint64_t current_hidden_id   = 0;
    std::uint64_t current_cyclic_id   = 0;
    std::uint64_t rewrite_gdn_id      = 0;
    std::uint64_t rewrite_hidden_id   = 0;
    std::uint64_t rewrite_cyclic_id   = 0;
    DiskCheckpointSlot rollback;
    std::array<DiskCheckpointSlot, 2> ladders{};
    std::vector<std::uint64_t> main_page_ids;
    std::vector<std::uint64_t> backend_page_ids;
};

} // namespace ninfer::targets::qwen3_6::detail
