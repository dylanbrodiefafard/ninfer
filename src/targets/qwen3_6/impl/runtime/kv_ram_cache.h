#pragma once

#include "core/arena.h"
#include "core/cyclic_kv_cache.h"
#include "core/linear_attention_state.h"
#include "core/paged_kv_cache.h"
#include "targets/qwen3_6/impl/runtime/context_checkpoint.h"
#include "targets/qwen3_6/impl/runtime/kv_ram_snapshot.h"
#include "targets/qwen3_6/impl/runtime/prefix_identity.h"

#include "ninfer/types.h"

#include <array>
#include <atomic>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <mutex>
#include <optional>
#include <span>
#include <unordered_map>
#include <vector>

#include <cuda_runtime_api.h>

namespace ninfer::targets::qwen3_6::detail {

struct RamLadderHead {
    std::uint32_t frontier = 0;
    PrefixHash128 hash{};
    ContextCheckpointKind kind = ContextCheckpointKind::Ladder;
    const void* conv       = nullptr;
    const void* recurrent  = nullptr;
    const void* hidden     = nullptr;
    const void* dflash     = nullptr;
    std::size_t conv_bytes      = 0;
    std::size_t recurrent_bytes = 0;
    std::size_t hidden_bytes    = 0;
    std::size_t dflash_bytes    = 0;
};

struct RamLadderIndex {
    std::uint32_t frontier = 0;
    PrefixHash128 hash{};
    ContextCheckpointKind kind = ContextCheckpointKind::Ladder;
};

struct RamLadderImage {
    std::uint32_t frontier = 0;
    PrefixHash128 hash{};
    ContextCheckpointKind kind = ContextCheckpointKind::Ladder;
    const void* conv       = nullptr;
    const void* recurrent  = nullptr;
    const void* hidden     = nullptr;
    const void* dflash     = nullptr;
    std::size_t conv_bytes      = 0;
    std::size_t recurrent_bytes = 0;
    std::size_t hidden_bytes    = 0;
    std::size_t dflash_bytes    = 0;
};

struct RamCaptureSource {
    std::uint32_t execution_frontier      = 0;
    std::uint32_t ledger_frontier         = 0;
    std::int32_t rope_delta               = 0;
    std::uint32_t text_kv_valid           = 0;
    std::uint32_t mtp_kv_valid            = 0;
    std::uint32_t dflash_context_frontier = 0;
    bool tail_hidden_valid                = false;
    bool rewrite_valid                    = false;
    RewriteCheckpointKind rewrite_kind    = RewriteCheckpointKind::TurnClosure;
    std::uint32_t rewrite_frontier        = 0;

    std::span<const TokenId> ledger;
    const ResidentPrefixIdentity* identity = nullptr;
    PrefixHash128 hash_f{};
    PrefixHash128 hash_c{};
    bool hash_c_valid = false;

    const PagedKVAllocation* text      = nullptr;
    const PagedKVPool* text_pool       = nullptr;
    const PagedKVAllocation* backend   = nullptr;
    const PagedKVPool* backend_pool    = nullptr;

    const LinearAttentionStatePool* gdn = nullptr;
    std::int32_t gdn_current_slot       = -1;
    std::int32_t gdn_checkpoint_slot    = -1;

    const Tensor* tail_hidden                = nullptr;
    const Tensor* rewrite_checkpoint_hidden  = nullptr;

    std::vector<RamLadderHead> ladder_heads;

    std::uint64_t disk_entry_id = 0;

    const CyclicKVCache* dflash_local      = nullptr;
    const CyclicKVCache* dflash_checkpoint = nullptr;
    std::int32_t dflash_lane               = 0;

    cudaStream_t stream = nullptr;
};

struct RamRestoreTarget {
    std::uint32_t text_dst_pages    = 0;
    std::uint32_t backend_dst_pages = 0;
    PagedKVAllocation* text         = nullptr;
    PagedKVPool* text_pool          = nullptr;
    PagedKVAllocation* backend      = nullptr;
    PagedKVPool* backend_pool       = nullptr;

    LinearAttentionStatePool* gdn     = nullptr;
    std::int32_t gdn_current_slot     = -1;
    std::int32_t gdn_checkpoint_slot  = -1;

    Tensor* tail_hidden               = nullptr;
    Tensor* rewrite_checkpoint_hidden = nullptr;

    PrefixReusePath reuse    = PrefixReusePath::FullReset;
    std::uint32_t reuse_base = 0;

    CyclicKVCache* dflash_local      = nullptr;
    CyclicKVCache* dflash_checkpoint = nullptr;
    std::int32_t dflash_lane         = 0;

    cudaStream_t stream = nullptr;
};

struct RamRestoredHost {
    std::uint32_t execution_frontier      = 0;
    std::uint32_t ledger_frontier         = 0;
    std::int32_t rope_delta               = 0;
    std::uint32_t text_kv_valid           = 0;
    std::uint32_t mtp_kv_valid            = 0;
    std::uint32_t dflash_context_frontier = 0;
    bool tail_hidden_valid                = false;
    bool rewrite_valid                    = false;
    RewriteCheckpointKind rewrite_kind    = RewriteCheckpointKind::TurnClosure;
    std::uint32_t rewrite_frontier        = 0;
    bool backend_image_present            = false;
    std::vector<TokenId> ledger;
    ResidentPrefixIdentity identity;
    std::vector<RamLadderIndex> ladders;
    std::vector<RamLadderImage> ladder_images;
    std::uint64_t disk_entry_id = 0;
};

struct RamMatch {
    std::uint64_t entry_id     = 0;
    PrefixReusePath reuse      = PrefixReusePath::FullReset;
    std::uint32_t reuse_base   = 0;
};

class KVRamCache {
public:
    explicit KVRamCache(std::size_t capacity_bytes);
    ~KVRamCache();

    KVRamCache(const KVRamCache&)            = delete;
    KVRamCache& operator=(const KVRamCache&) = delete;
    KVRamCache(KVRamCache&&)                 = delete;
    KVRamCache& operator=(KVRamCache&&)      = delete;

    [[nodiscard]] std::optional<RamMatch> plan_match(const PreparedPromptData& prompt,
                                                     std::span<const PrefixHash128> hash_chain);

    void claim(std::uint64_t entry_id);
    void release(std::uint64_t entry_id);
    void consume(std::uint64_t entry_id);
    [[nodiscard]] bool is_claimed(std::uint64_t entry_id) const;

    std::optional<std::uint64_t> capture(const RamCaptureSource& source);
    [[nodiscard]] std::optional<std::uint64_t> peek_oldest_unpinned() const;
    [[nodiscard]] std::vector<std::uint64_t> fifo_ids() const;
    void pin_for_io(std::uint64_t entry_id);
    void unpin_for_io(std::uint64_t entry_id);
    bool evict_one_unpinned(std::uint64_t entry_id);
    void set_disk_entry_id(std::uint64_t entry_id, std::uint64_t disk_entry_id);
    [[nodiscard]] std::uint64_t disk_entry_id(std::uint64_t entry_id) const;
    [[nodiscard]] const void* host_block(std::uint64_t entry_id) const;
    [[nodiscard]] std::size_t host_bytes(std::uint64_t entry_id) const;

    struct HostKvView {
        std::uint32_t text_pages     = 0;
        std::uint32_t backend_pages  = 0;
        const std::uint8_t* text     = nullptr;
        const std::uint8_t* backend  = nullptr;
        const std::uint8_t* ledger   = nullptr;
        std::size_t ledger_bytes     = 0;
        const std::uint8_t* identity = nullptr;
        std::size_t identity_bytes   = 0;
        const std::uint8_t* gdn_conv_current = nullptr;
        const std::uint8_t* gdn_recurrent_current = nullptr;
        const std::uint8_t* gdn_conv_checkpoint = nullptr;
        const std::uint8_t* gdn_recurrent_checkpoint = nullptr;
        const std::uint8_t* tail_hidden = nullptr;
        const std::uint8_t* rewrite_hidden = nullptr;
        const std::uint8_t* dflash_local = nullptr;
        const std::uint8_t* dflash_rewrite = nullptr;
        std::size_t gdn_conv_bytes   = 0;
        std::size_t gdn_recurrent_bytes = 0;
        std::size_t hidden_bytes     = 0;
        std::size_t rewrite_hidden_bytes = 0;
        std::size_t cyclic_bytes     = 0;
        std::vector<RamLadderImage> ladder_images;
    };
    [[nodiscard]] HostKvView host_kv(std::uint64_t entry_id) const;
    RamRestoredHost unpack_device(std::uint64_t entry_id, const RamRestoreTarget& target);

    [[nodiscard]] RamRestoredHost load_host(std::uint64_t entry_id) const;

    [[nodiscard]] KvRamSnapshot snapshot() const noexcept;
    KvRamCopySeconds harvest_copy_seconds();
    [[nodiscard]] bool copies_ready(std::uint64_t entry_id) const;
    [[nodiscard]] bool pending_copies_ready() const;
    void wait_pending_copies_on_stream(cudaStream_t stream);
    void wait_pending_copies();
    [[nodiscard]] std::uint64_t index_version() const noexcept { return index_version_; }
    [[nodiscard]] std::uint64_t exact_comparisons() const noexcept { return exact_comparisons_; }
    void record_drop();

    void test_tamper_identity_digest(std::uint64_t entry_id, std::uint8_t byte);
    [[nodiscard]] std::size_t test_pending_copy_count() const noexcept;
    void test_fail_next_ticket_write() noexcept { fail_next_ticket_write_ = true; }
    void test_fail_next_capture() noexcept { fail_next_capture_ = true; }
    void test_fail_next_copy_sync() noexcept { fail_next_copy_sync_ = true; }
    void test_fail_next_retire() noexcept { fail_next_retire_ = true; }
    [[nodiscard]] std::uint32_t test_io_pins(std::uint64_t entry_id) const;
    void test_set_copy_sync_stall_ms(int ms);
    [[nodiscard]] bool test_copy_sync_entered() const;

private:
    enum class Section : std::uint8_t {
        Ledger = 0,
        Identity,
        TextKv,
        BackendKv,
        GdnConvCurrent,
        GdnConvCheckpoint,
        GdnRecurrentCurrent,
        GdnRecurrentCheckpoint,
        TailHidden,
        RewriteCheckpointHidden,
        DflashLocal,
        DflashRewriteCheckpoint,
        Count
    };

    struct Record {
        std::uint64_t id               = 0;
        PrefixHash128 hash_f{};
        PrefixHash128 hash_c{};
        bool hash_c_valid              = false;
        std::uint32_t execution_frontier = 0;
        std::uint32_t checkpoint_frontier = 0;
        bool checkpoint_valid          = false;
        PrefixReusePath checkpoint_path = PrefixReusePath::RestoreTurnCheckpoint;
        std::vector<RamLadderIndex> ladders;
        void* block                    = nullptr;
        std::size_t bytes              = 0;
        bool pinned                    = false;
        std::uint32_t io_pins          = 0;
        std::uint64_t disk_entry_id    = 0;
        bool copies_timed              = false;
        cudaEvent_t copies_start       = nullptr;
        cudaEvent_t copies_done        = nullptr;
    };

    struct Layout {
        std::size_t header_bytes                               = 0;
        std::array<std::size_t, static_cast<std::size_t>(Section::Count)> offset{};
        std::array<std::size_t, static_cast<std::size_t>(Section::Count)> length{};
        std::size_t entry_bytes                                = 0;
    };

    [[nodiscard]] Record& require(std::uint64_t entry_id);
    [[nodiscard]] const Record& require(std::uint64_t entry_id) const;
    void destroy_record(std::uint64_t entry_id, bool count_eviction,
                        std::unique_lock<std::mutex>& lock);
    void begin_copies(Record& record, cudaStream_t stream);
    void record_copies(Record& record, cudaStream_t stream);
    [[nodiscard]] bool copies_ready_locked(std::uint64_t entry_id) const;
    void wait_copies(Record& record);
    void wait_copies_on_stream(Record& record, cudaStream_t stream);
    void wait_event_unlocked(std::unique_lock<std::mutex>& lock, cudaEvent_t event,
                              std::uint64_t entry_id);
    void maybe_copy_sync_stall() const;
    double harvest_record(Record& record);
    [[nodiscard]] double copy_elapsed_seconds(const Record& record) const;
    void retire_record(Record& record);
    void reap_retired(bool block);
    void pin_pending_copy_events(std::vector<cudaEvent_t>& events, std::vector<std::uint64_t>& ids);
    void unpin_copy_events(const std::vector<std::uint64_t>& ids) noexcept;
    void drop_pending_save(std::uint64_t entry_id) noexcept;
    void drop_pending_id(std::uint64_t entry_id) noexcept;
    void bump_version() noexcept { ++index_version_; }

    struct RetiredCopy {
        void* block           = nullptr;
        cudaEvent_t copies_done = nullptr;
    };

    HostPinnedArena arena_;
    std::deque<std::uint64_t> fifo_;
    std::unordered_map<std::uint64_t, Record> records_;
    std::vector<RetiredCopy> retired_;
    std::vector<std::uint64_t> pending_save_ids_;
    std::optional<std::uint64_t> pending_load_id_;
    std::uint64_t next_id_           = 1;
    std::uint64_t index_version_     = 1;
    std::uint64_t captures_          = 0;
    std::uint64_t restores_          = 0;
    std::uint64_t evictions_         = 0;
    std::uint64_t drops_             = 0;
    std::uint64_t exact_comparisons_ = 0;
    double save_seconds_             = 0;
    double load_seconds_             = 0;
    double orphaned_save_seconds_    = 0;
    double orphaned_load_seconds_    = 0;
    mutable std::mutex io_mutex_;
    std::condition_variable io_cv_;
    bool fail_next_ticket_write_ = false;
    bool fail_next_capture_      = false;
    bool fail_next_copy_sync_   = false;
    bool fail_next_retire_      = false;
    std::atomic<int> copy_sync_stall_ms_{0};
    mutable std::atomic<bool> copy_sync_entered_{false};
};

} // namespace ninfer::targets::qwen3_6::detail
