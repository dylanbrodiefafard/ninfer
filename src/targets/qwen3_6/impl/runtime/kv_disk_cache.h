#pragma once

#include "core/cyclic_kv_cache.h"
#include "core/linear_attention_state.h"
#include "core/paged_kv_cache.h"
#include "targets/qwen3_6/impl/runtime/kv_disk_format.h"
#include "targets/qwen3_6/impl/runtime/kv_disk_snapshot.h"
#include "targets/qwen3_6/impl/runtime/kv_ram_cache.h"

#include "ninfer/types.h"

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <filesystem>
#include <functional>
#include <memory>
#include <mutex>
#include <new>
#include <optional>
#include <span>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include <cuda_runtime_api.h>

namespace ninfer::targets::qwen3_6::detail {

struct UninitializedBytes {
    UninitializedBytes() = default;
    ~UninitializedBytes() { std::free(data_); }
    UninitializedBytes(const UninitializedBytes& other) { assign(other); }
    UninitializedBytes& operator=(const UninitializedBytes& other) {
        if (this != &other) { assign(other); }
        return *this;
    }
    UninitializedBytes(UninitializedBytes&& other) noexcept
        : data_(std::exchange(other.data_, nullptr)), size_(std::exchange(other.size_, 0)) {}
    UninitializedBytes& operator=(UninitializedBytes&& other) noexcept {
        if (this != &other) {
            std::free(data_);
            data_ = std::exchange(other.data_, nullptr);
            size_ = std::exchange(other.size_, 0);
        }
        return *this;
    }
    void resize(std::size_t size) {
        if (size == size_) { return; }
        void* next = nullptr;
        if (size != 0 && ::posix_memalign(&next, kDiskPageIoAlignment, size) != 0) {
            throw std::bad_alloc();
        }
        std::free(data_);
        data_ = static_cast<std::uint8_t*>(next);
        size_ = size;
    }
    [[nodiscard]] bool empty() const noexcept { return size_ == 0; }
    [[nodiscard]] std::size_t size() const noexcept { return size_; }
    [[nodiscard]] std::uint8_t* data() noexcept { return data_; }
    [[nodiscard]] const std::uint8_t* data() const noexcept { return data_; }

private:
    void assign(const UninitializedBytes& other) {
        resize(other.size_);
        if (size_ != 0) { std::memcpy(data_, other.data_, size_); }
    }
    std::uint8_t* data_ = nullptr;
    std::size_t size_   = 0;
};

struct DiskMatch {
    std::uint64_t entry_id              = 0;
    PrefixReusePath reuse               = PrefixReusePath::FullReset;
    std::uint32_t reuse_base            = 0;
    PrefixHash128 hash_f{};
    std::uint32_t execution_frontier    = 0;
};

struct DiskLadderImage {
    std::uint32_t frontier = 0;
    PrefixHash128 hash{};
    ContextCheckpointKind kind = ContextCheckpointKind::Ladder;
    std::shared_ptr<PinnedHostBuffer> conv;
    std::shared_ptr<PinnedHostBuffer> recurrent;
    std::shared_ptr<PinnedHostBuffer> hidden;
    std::shared_ptr<PinnedHostBuffer> dflash;
};

struct DiskRestoredHost {
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
    std::vector<DiskLadderImage> ladder_images;
    std::uint64_t disk_entry_id           = 0;
};

struct DiskRestoreTarget {
    std::uint32_t text_dst_pages    = 0;
    std::uint32_t backend_dst_pages = 0;
    PagedKVAllocation* text         = nullptr;
    PagedKVPool* text_pool          = nullptr;
    PagedKVAllocation* backend      = nullptr;
    PagedKVPool* backend_pool       = nullptr;

    LinearAttentionStatePool* gdn    = nullptr;
    std::int32_t gdn_current_slot    = -1;
    std::int32_t gdn_checkpoint_slot = -1;

    Tensor* tail_hidden               = nullptr;
    Tensor* rewrite_checkpoint_hidden = nullptr;

    PrefixReusePath reuse    = PrefixReusePath::FullReset;
    std::uint32_t reuse_base = 0;

    CyclicKVCache* dflash_local      = nullptr;
    CyclicKVCache* dflash_checkpoint = nullptr;
    std::int32_t dflash_lane         = 0;

    cudaStream_t stream = nullptr;
};

enum class DiskFaultPoint : std::uint8_t {
    None,
    AfterRecordWrite,
    AfterPackSync,
    AfterMapAppend,
    AfterMapSync,
    AfterMetaTmpWrite,
    AfterMetaTmpSync,
    AfterMetaRename,
    AfterEntryDirSync,
    AfterEntryParentSync,
    AfterTombstoneTmpWrite,
    AfterTombstoneTmpSync,
    AfterTombstoneRename,
    AfterTombstoneDirSync,
    AfterPacksetTmpWrite,
    AfterPacksetTmpSync,
    AfterPacksetRename,
    AfterPacksetRootSync,
    CompactionAfterPackCopy,
    CompactionAfterPackSync,
    CompactionAfterBaseWrite,
    CompactionAfterBaseSync,
    CompactionAfterLogWrite,
    CompactionAfterLogSync,
    CompactionAfterPacksetTmpWrite,
    CompactionAfterPacksetTmpSync,
    CompactionAfterPacksetRename,
    CompactionRecoveryBeforeRootSync,
    CompactionAfterRootSync,
};

struct DiskOpenConfig {
    std::filesystem::path location;
    std::size_t capacity_bytes          = 0;
    KvDiskCompress compress             = KvDiskCompress::Off;
    std::uint32_t max_context           = 0;
    KVRamCache* ram                     = nullptr;
    DiskFingerprint fingerprint;
    const PagedKVPool* text_pool        = nullptr;
    const PagedKVPool* backend_pool     = nullptr;
    std::size_t logical_page_bytes      = 0;
    std::size_t gdn_staging_bytes       = 0;
    std::uint64_t hidden_bytes           = 0;
    std::uint32_t max_index_entries     = 1u << 18;
    std::uint32_t restore_window_slots = 2;
    std::uint32_t restore_io_threads    = 1;
    std::uint32_t pack_page_batch       = 4;
    DiskFaultPoint test_fault_point     = DiskFaultPoint::None;
};

class KVDiskCache {
public:
    explicit KVDiskCache(DiskOpenConfig config);
    ~KVDiskCache() noexcept;

    KVDiskCache(const KVDiskCache&)            = delete;
    KVDiskCache& operator=(const KVDiskCache&) = delete;
    KVDiskCache(KVDiskCache&&)                 = delete;
    KVDiskCache& operator=(KVDiskCache&&)      = delete;

    [[nodiscard]] std::optional<DiskMatch> plan_match(const PreparedPromptData& prompt,
                                                      std::span<const PrefixHash128> hash_chain);
    [[nodiscard]] std::optional<DiskRestoredHost> load_host(std::uint64_t entry_id) const;
    [[nodiscard]] bool populate_checkpoint_images(DiskRestoredHost& host);
    [[nodiscard]] DiskRestoredHost take_restore_checkpoints();

    bool claim(std::uint64_t entry_id);
    bool claim(std::uint64_t entry_id, PrefixHash128 expected_hash_f,
               std::uint32_t expected_frontier, std::uint32_t expected_reuse_base = 0,
               PrefixReusePath expected_reuse = PrefixReusePath::FullReset);
    void release(std::uint64_t entry_id);
    void consume(std::uint64_t entry_id);

    void note_ram_resident(std::uint64_t ram_id, std::uint64_t disk_ticket);
    void forget_ram_resident(std::uint64_t ram_id) noexcept;
    [[nodiscard]] bool ram_is_durable(std::uint64_t ram_id) const;

    void request_idle_spill();
    void cancel_idle_spill();
    void cancel_idle_of_ram(std::uint64_t ram_id);
    void begin_ram_idle_exclusion(std::uint64_t ram_id);
    void end_ram_idle_exclusion(std::uint64_t ram_id) noexcept;
    bool emergency_spill_ram(std::uint64_t ram_id);

    void prefetch_window(std::uint64_t entry_id, std::uint32_t text_dst_pages,
                         std::uint32_t backend_dst_pages);
    std::uint64_t restore_device(std::uint64_t entry_id, const DiskRestoreTarget& target);
    void pump_restore(cudaStream_t stream);
    void cancel_restore();

    [[nodiscard]] bool copies_ready() const;
    [[nodiscard]] bool restore_failed() const;
    void wait_copies();
    void wait_copies(std::uint64_t epoch);
    void wait_copies_on_stream(cudaStream_t stream);
    void wait_copies_on_stream(cudaStream_t stream, std::uint64_t epoch);
    void release_restore_ticket(std::uint64_t epoch);
    KvDiskCopySeconds harvest_copy_seconds();
    [[nodiscard]] KvDiskSnapshot snapshot() const noexcept;
    [[nodiscard]] std::uint64_t index_version() const noexcept {
        return index_version_.load(std::memory_order_relaxed);
    }

    void wait_idle_and_fsync();
    void flush_not_durable_ram(
        const std::function<void(std::uint64_t done, std::uint64_t total)>& on_progress = {});

    [[nodiscard]] std::uint64_t exact_comparisons() const noexcept { return exact_comparisons_; }
    [[nodiscard]] static std::uint32_t test_crc32c(std::span<const std::uint8_t> bytes) noexcept;
    void test_break_object(std::uint64_t object_id, DiskObjectKind kind);
    [[nodiscard]] std::uint64_t test_object_refcount(std::uint64_t object_id) const;
    [[nodiscard]] bool test_entry_in_index(std::uint64_t entry_id) const;
    [[nodiscard]] std::vector<std::uint64_t> test_main_page_ids(std::uint64_t entry_id) const;
    [[nodiscard]] std::vector<std::uint64_t> test_backend_page_ids(std::uint64_t entry_id) const;
    [[nodiscard]] DiskMeta test_load_meta(std::uint64_t entry_id) const;
    [[nodiscard]] std::size_t test_skipped_count() const;
    void test_arm_crash_before_meta();
    void test_arm_stall_after_meta_rename();
    void test_arm_fail_after_meta_rename();
    void test_arm_fail_rollback_meta();
    void test_arm_fail_after_rollback_rename();
    void test_arm_fail_tombstone();
    void test_arm_fail_object_write();
    void test_arm_fail_object_unlink();
    void test_arm_fail_entry_unlink();
    void test_arm_fail_prepare_spill();
    bool test_fifo_evict_one_unpersisted();
    [[nodiscard]] bool test_meta_renamed() const;
    [[nodiscard]] bool test_objects_fsynced_before_meta() const;
    [[nodiscard]] std::uint32_t test_disk_io_pins(std::uint64_t entry_id) const;
    void test_force_zstd_fail();
    void test_set_payload_io_stall_ms(int ms);
    [[nodiscard]] bool test_payload_io_entered() const;
    void test_set_manifest_io_stall_ms(int ms);
    [[nodiscard]] bool test_manifest_io_entered() const;
    void test_set_free_bytes_override(std::optional<std::uint64_t> bytes);
    void test_set_pack_position(DiskObjectKind kind, std::uint32_t segment,
                                std::uint64_t tail);
    void test_arm_partial_pwritev(std::size_t max_bytes);
    void test_arm_fault(DiskFaultPoint point);
    void test_arm_fault_sequence(DiskFaultPoint first, DiskFaultPoint second);
    [[nodiscard]] std::shared_ptr<void> test_hold_active_generation() const;
    void test_set_fsync_stall_ms(int ms);
    [[nodiscard]] bool test_fsync_entered() const;
    bool test_gc_skipped_one();
    bool test_fifo_evict_one();
    void test_arm_restore_job_barrier();
    [[nodiscard]] bool test_restore_job_dequeued() const;
    void test_release_restore_job_barrier();
    void test_arm_restore_state_barrier();
    [[nodiscard]] bool test_restore_state_entered() const;
    void test_release_restore_state_barrier();
    [[nodiscard]] bool test_has_ram_note(std::uint64_t ram_id) const;
    void test_arm_copy_lease_barrier();
    [[nodiscard]] bool test_copy_event_leased() const;
    void test_release_copy_lease_barrier();
    [[nodiscard]] cudaEvent_t test_copies_done() const;
    void test_arm_fail_page_read();
    void test_arm_fail_restore_state_setup();
    void test_arm_fail_after_checkpoint_prepare_start();
    void test_arm_fail_after_state_h2d_enqueue();
    void test_arm_direct_state_read_barrier();
    [[nodiscard]] bool test_direct_state_read_entered() const;
    void test_release_direct_state_read_barrier();
    void test_arm_fail_wait_copies();
    void test_set_state_decode_stall_ms(int ms);
    [[nodiscard]] int test_state_decode_count() const;
    [[nodiscard]] std::size_t test_retired_copy_events() const;
    void test_arm_wait_epoch_barrier();
    [[nodiscard]] bool test_wait_epoch_latched() const;
    void test_release_wait_epoch_barrier();
    [[nodiscard]] cudaEvent_t test_last_wait_copy_event() const;
    [[nodiscard]] std::uint32_t test_timing_harvests() const;
    [[nodiscard]] std::uint64_t test_committed_generation(std::uint64_t entry_id) const;
    [[nodiscard]] bool test_window_filled_for(std::uint64_t entry_id) const;
    void test_arm_page_read_barrier();
    [[nodiscard]] bool test_page_read_entered() const;
    void test_release_page_read_barrier();
    [[nodiscard]] std::uint32_t test_restore_state_inflight() const;
    [[nodiscard]] std::uint32_t test_window_inflight() const;
    [[nodiscard]] int test_page_read_count() const;
    void test_reset_restore_loop_idle_spins();
    [[nodiscard]] std::uint64_t test_restore_loop_idle_spins() const;
    void test_arm_payload_take_barrier();
    [[nodiscard]] bool test_payload_take_entered() const;
    void test_release_payload_take_barrier();
    void test_arm_slot_assign_barrier();
    [[nodiscard]] bool test_slot_assign_entered() const;
    void test_release_slot_assign_barrier();
    void test_arm_scatter_record_barrier(std::uint32_t logical_index);
    [[nodiscard]] bool test_scatter_record_entered() const;
    void test_release_scatter_record_barrier();
    [[nodiscard]] bool test_restore_read_queued(std::uint32_t pool, std::uint32_t logical) const;
    [[nodiscard]] bool test_reader_claim(std::uint32_t pool, std::uint32_t logical) const;
    [[nodiscard]] bool test_window_assigned(std::uint32_t pool, std::uint32_t logical) const;
    void test_arm_fail_after_payload_take();
    [[nodiscard]] std::uint32_t test_payload_io_inflight() const;
    [[nodiscard]] std::uint64_t test_prefetch_preempted_idle() const;
    [[nodiscard]] bool test_emergency_queued() const;
    void test_stop_io_threads();
    void test_set_stopping();

private:
    struct DecodedCheckpointState {
        std::shared_ptr<PinnedHostBuffer> first;
        std::shared_ptr<PinnedHostBuffer> second;
    };
    enum class JobKind : std::uint8_t {
        RestoreRead,
        EmergencySpillPage,
        EmergencyCommit,
        IdleSpillPage,
        IdleCommit,
        PrefetchWindow,
        Stop,
    };

    struct Job {
        JobKind kind                 = JobKind::Stop;
        std::uint64_t disk_entry_id  = 0;
        std::uint64_t ram_entry_id   = 0;
        std::uint64_t spill_epoch     = 0;
        std::uint64_t restore_epoch = 0;
        std::uint64_t claim_generation = 0;
        std::uint64_t committed_generation = 0;
        std::uint32_t pool           = 0;
        std::uint32_t logical_index  = 0;
        std::uint32_t window_slot    = 0;
        std::uint32_t text_extent    = 0;
        std::uint32_t backend_extent = 0;
    };

    struct IndexEntry {
        DiskMeta meta;
        std::vector<TokenId> ledger;
        ResidentPrefixIdentity identity;
        std::vector<std::uint64_t> uncertainty_ids;
        bool pinned       = false;
        std::uint32_t io_pins = 0;
        std::uint64_t claim_generation = 0;
        std::uint64_t committed_generation = 0;
    };

    struct RamNote {
        std::uint64_t ticket           = 0;
        bool durable                   = false;
        bool failed_this_generation    = false;
        std::uint64_t generation_stamp = 0;
    };

    struct WindowSlot {
        void* io                       = nullptr;
        void* host                     = nullptr;
        cudaEvent_t h2d_arrived        = nullptr;
        cudaEvent_t h2d_event          = nullptr;
        std::uint32_t pool           = 0;
        std::uint32_t logical_index  = 0;
        std::uint64_t object_id      = 0;
        std::uint64_t disk_entry_id  = 0;
        std::uint64_t epoch          = 0;
        std::uint64_t claim_generation = 0;
        std::uint64_t committed_generation = 0;
        bool filled                  = false;
        bool h2d_done                = false;
        bool assigned                = false;
    };

    struct EncodedStateBlob {
        DiskStateKind kind = DiskStateKind::CurrentGdn;
        DiskCodec codec = DiskCodec::Raw;
        const void* first = nullptr;
        std::size_t first_bytes = 0;
        const void* second = nullptr;
        std::size_t second_bytes = 0;
        std::vector<std::uint8_t> compressed;
        [[nodiscard]] std::size_t uncompressed_bytes() const noexcept {
            return first_bytes + second_bytes;
        }
        [[nodiscard]] std::size_t payload_bytes() const noexcept {
            return codec == DiskCodec::Zstd ? compressed.size() : uncompressed_bytes();
        }
    };

    struct SpillSession {
        std::uint64_t ram_id         = 0;
        std::uint64_t ticket         = 0;
        std::uint64_t parent_id      = 0;
        std::uint64_t child_id       = 0;
        bool emergency               = false;
        bool cancelled               = false;
        bool failed                  = false;
        bool committed               = false;
        bool meta_installed          = false;
        std::uint64_t epoch          = 0;
        enum class Action : std::uint8_t { Create, Extend, Branch, Refresh } action =
            Action::Create;
        std::uint32_t share_tokens   = 0;
        DiskMeta draft;
        std::vector<std::uint64_t> new_object_ids;
        std::vector<DiskObjectKind> new_object_kinds;
        std::uint32_t next_main      = 0;
        std::uint32_t next_backend   = 0;
        std::uint32_t main_pages     = 0;
        std::uint32_t backend_pages  = 0;
        KVRamCache::HostKvView image;
        RamRestoredHost host;
        std::vector<EncodedStateBlob> encoded_state;
        std::size_t next_encoded_state = 0;
        std::chrono::steady_clock::time_point started{};
    };

    struct PackDescriptor {
        int buffered = -1;
        int direct   = -1;
        ~PackDescriptor();
    };
    struct PackGeneration {
        std::uint64_t number = 0;
        std::filesystem::path root;
        mutable std::mutex descriptor_mu;
        mutable std::unordered_map<std::string, std::shared_ptr<PackDescriptor>> descriptors;
    };

    struct ObjectRef {
        struct Location {
            std::shared_ptr<PackGeneration> generation;
            std::uint32_t segment = 0;
            std::uint64_t offset = 0;
            std::uint64_t extent_bytes = 0;
            std::uint64_t stored_bytes = 0;
            std::uint64_t logical_bytes = 0;
            std::uint32_t record_crc32c = 0;
        } location;
        DiskObjectKind kind     = DiskObjectKind::Main;
        std::uint64_t bytes     = 0;
        std::uint32_t live_refs = 0;
        std::uint32_t skip_refs = 0;
        std::uint32_t hold_refs = 0;
    };

    struct StartupObjectValidation {
        DiskObjectKind kind = DiskObjectKind::Main;
        bool page = false;
        std::optional<DiskStateKind> state_kind;
    };

    struct SkippedTree {
        std::uint64_t entry_id = 0;
        std::filesystem::file_time_type mtime{};
        std::vector<std::pair<DiskObjectKind, std::uint64_t>> objects;
    };

    void io_loop();
    void restore_loop();
    void join_io_threads() noexcept;
    void stop_io_threads() noexcept;
    void enqueue(Job job, std::unique_lock<std::mutex>* lock = nullptr);
    [[nodiscard]] std::optional<Job> take_job(std::unique_lock<std::mutex>& lock);
    [[nodiscard]] std::optional<Job> take_restore_job(std::unique_lock<std::mutex>& lock);
    [[nodiscard]] bool prefetch_readable_locked() const noexcept;
    [[nodiscard]] bool restore_readers_busy_locked() const noexcept;
    [[nodiscard]] bool restore_or_prefetch_busy_locked() const noexcept;
    [[nodiscard]] bool restore_cancels_idle_locked() const noexcept;
    [[nodiscard]] bool emergency_payload_busy_locked() const noexcept;
    void note_reader_claim_locked(const Job& job);
    void drop_reader_claim(const Job& job) noexcept;
    [[nodiscard]] bool restore_job_is_live(const Job& job) const noexcept;
    [[nodiscard]] bool job_matches_index(const Job& job) const noexcept;
    [[nodiscard]] bool slot_matches_index(const WindowSlot& slot) const noexcept;
    [[nodiscard]] bool page_job_is_live_restore(const Job& job) const noexcept;
    [[nodiscard]] bool slot_is_live_restore_page(const WindowSlot& slot) const noexcept;
    [[nodiscard]] bool restore_state_is_live(std::uint64_t epoch, std::uint64_t entry,
                                              PrefixReusePath reuse,
                                              std::uint32_t reuse_base) const noexcept;
    [[nodiscard]] bool restore_state_is_live(std::uint64_t epoch, std::uint64_t entry,
                                              std::uint64_t committed_generation,
                                              std::uint64_t claim_generation,
                                              PrefixReusePath reuse,
                                              std::uint32_t reuse_base) const noexcept;
    void maybe_restore_job_barrier();
    void maybe_restore_state_barrier();
    void maybe_direct_state_read_barrier();
    void process_job(const Job& job);
    void cancel_idle_locked(std::unique_lock<std::mutex>& lock);
    void register_location();
    void unregister_location() noexcept;
    void enqueue_spill_jobs(SpillSession& session);
    void finish_restore_state_locked(std::unique_lock<std::mutex>& lock, cudaStream_t stream);
    void start_restore_state_h2d_locked(cudaStream_t stream);
    void pump_restore_locked(std::unique_lock<std::mutex>& lock, cudaStream_t stream);
    void wait_state_arena_idle(std::unique_lock<std::mutex>& lock);
    cudaEvent_t wait_restore_copy_event(std::uint64_t epoch = 0);
    void record_copies_done_locked(cudaStream_t stream);
    void record_restore_join_locked(cudaStream_t stream);
    void bill_post_disk_h2d();
    void fsync_new_objects(const SpillSession& session) const;
    [[nodiscard]] std::uint32_t window_slots() const noexcept {
        return static_cast<std::uint32_t>(window_.size());
    }
    void ensure_live_copies_event_locked();
    void retire_copies_event_locked();
    void release_copy_event(cudaEvent_t event);
    cudaEvent_t lease_copy_event_locked(std::uint64_t epoch);
    void finish_window_inflight(std::uint32_t slot, bool keep_slot) noexcept;
    void maybe_copy_lease_barrier();
    void maybe_wait_epoch_barrier();
    void maybe_page_read_barrier();
    void maybe_payload_take_barrier();
    void maybe_slot_assign_barrier(std::uint32_t logical_index);
    void maybe_scatter_record_barrier(std::uint32_t logical_index);
    void maybe_state_decode_stall();
    void harvest_retired_timing_locked();
    void try_harvest_copy_pair_locked(cudaEvent_t start, cudaEvent_t done, bool& timed);
    void note_live_host_load_locked(std::chrono::steady_clock::time_point started,
                                     std::chrono::steady_clock::time_point ended);
    void ensure_copies_start_locked();
    [[nodiscard]] bool populate_checkpoint_images(DiskRestoredHost& host, std::uint64_t epoch,
                                                  std::uint64_t entry, PrefixReusePath reuse,
                                                  std::uint32_t reuse_base, bool check_live);
    [[nodiscard]] bool populate_checkpoint_images(DiskRestoredHost& host, std::uint64_t epoch,
                                                  std::uint64_t entry,
                                                  std::uint64_t committed_generation,
                                                  std::uint64_t claim_generation,
                                                  PrefixReusePath reuse, std::uint32_t reuse_base,
                                                  bool check_live);
    void load_restore_state_locked(std::unique_lock<std::mutex>& lock);
    void reset_window_slot_keep_host(WindowSlot& slot) noexcept;
    void open_directory();
    void acquire_lock();
    void release_lock() noexcept;
    void load_or_write_fingerprint();
    void open_pack_store();
    void persist_packset(std::uint64_t reserved_next_id) const;
    void load_pack_map();
    void validate_live_direct_segments() const;
    void append_pack_map(const SpillSession& session) const;
    [[nodiscard]] bool compact_packs(std::unique_lock<std::mutex>& lock,
                                     std::uint64_t additional_bytes = 0);
    [[nodiscard]] std::uint64_t packed_allocated_bytes() const;
    [[nodiscard]] std::uint64_t packed_retained_bytes() const;
    [[nodiscard]] bool packs_need_compaction() const;
    void load_index();
    void rebuild_manifest();
    bool load_entry(std::uint64_t entry_id);
    void write_fingerprint_file() const;
    void write_manifest() const;
    void write_manifest(std::unique_lock<std::mutex>& lock) const;
    void fsync_path(const std::filesystem::path& path) const;
    void fsync_dir(const std::filesystem::path& dir) const;
    void fsync_object_namespaces() const;
    void fsync_store_dirs() const;
    void persist_eviction(std::unique_lock<std::mutex>& lock);
    void reclaim_durable_tombstones();
    void adopt_object_ids();
    void bump_next_id(std::uint64_t& next, std::uint64_t seen) const noexcept;
    [[nodiscard]] bool object_kind_conflict(DiskObjectKind kind, std::uint64_t id) const noexcept;
    [[nodiscard]] bool meta_cannot_restore(const DiskMeta& meta) const noexcept;
    [[nodiscard]] std::uint32_t backend_tokens_required(const DiskMeta& meta) const noexcept;
    void queue_unlink(const std::filesystem::path& path);
    [[nodiscard]] bool unlink_path(const std::filesystem::path& path);
    void write_entry_tombstone(
        std::uint64_t id, const std::vector<std::pair<DiskObjectKind, std::uint64_t>>& objects);
    void clear_entry_tombstone(std::uint64_t id) const;
    void queue_tombstone_object_unlinks();
    [[nodiscard]] bool tombstone_objects_remain(
        const std::vector<std::pair<DiskObjectKind, std::uint64_t>>& objects) const;
    void adopt_tombstone_ids();
    [[nodiscard]] bool entry_is_tombstoned(std::uint64_t id) const;
    [[nodiscard]] std::filesystem::path tombstone_path(std::uint64_t id) const;
    [[nodiscard]] std::uint32_t index_entry_limit() const noexcept;
    bool flush_queued_unlinks();
    bool flush_queued_unlinks(std::unique_lock<std::mutex>& lock);

    [[nodiscard]] std::filesystem::path pack_path(const PackGeneration& generation,
                                                   DiskObjectKind kind,
                                                   std::uint32_t segment) const;
    [[nodiscard]] std::filesystem::path pack_path(DiskObjectKind kind,
                                                   std::uint32_t segment) const;
    [[nodiscard]] std::shared_ptr<PackDescriptor> acquire_pack_descriptor(
        const std::shared_ptr<PackGeneration>& generation, DiskObjectKind kind,
        std::uint32_t segment, bool require_direct) const;
    void reap_retired_generations();
    [[nodiscard]] std::filesystem::path entry_dir(std::uint64_t id) const;
    [[nodiscard]] std::uint64_t allocate_object(DiskObjectKind kind, std::uint64_t bytes);
    void add_ref(std::uint64_t id);
    void add_skip_ref(std::uint64_t id);
    void add_hold(std::uint64_t id);
    void release_live_ref(std::uint64_t id);
    void release_hold(std::uint64_t id);
    void drop_ref(std::uint64_t id);
    void drop_skip_ref(std::uint64_t id);
    void unlink_unreferenced(std::uint64_t id);
    bool make_capacity(std::uint64_t needed, std::unique_lock<std::mutex>& lock);
    void fifo_evict_one();
    bool gc_skipped_one();

    void pin_disk(std::uint64_t entry_id);
    void unpin_disk(std::uint64_t entry_id);
    void invalidate_ram_notes_for_disk_entry(std::uint64_t disk_id);
    [[nodiscard]] IndexEntry& require(std::uint64_t entry_id);
    [[nodiscard]] const IndexEntry& require(std::uint64_t entry_id) const;
    void bump_version() noexcept {
        index_version_.fetch_add(1, std::memory_order_relaxed);
    }

    void promote_idle_spill_to_emergency();
    bool prepare_spill(std::uint64_t ram_id, bool emergency, std::unique_lock<std::mutex>& lock);
    void spill_one_page(SpillSession& session, std::uint32_t pool, std::uint32_t logical,
                        std::unique_lock<std::mutex>& lock);
    void spill_page_batch(SpillSession& session, std::span<const Job> jobs,
                          std::unique_lock<std::mutex>& lock);
    void commit_spill(SpillSession& session, std::unique_lock<std::mutex>& lock);
    void install_committed_entry(SpillSession& session, std::unique_lock<std::mutex>& lock,
                                 bool mark_durable = true, bool retain_replaced = false);
    void release_spill_pins(SpillSession& session, bool mark_failed);
    void drop_spill(SpillSession& session, std::unique_lock<std::mutex>& lock);
    void purge_spill_jobs(std::uint64_t epoch);
    [[nodiscard]] bool draft_ready(const SpillSession& session) const;
    void record_uncertainty(std::uint64_t entry_id, const std::vector<std::uint64_t>& ids);
    void reclaim_orphan_objects();
    void write_new_object(DiskObjectKind kind, std::uint64_t id, const void* data,
                          std::uint64_t bytes, std::unique_lock<std::mutex>* lock);
    void write_new_object_parts(DiskObjectKind kind, std::uint64_t id,
                               std::span<const std::pair<const void*, std::uint64_t>> parts,
                               std::uint64_t bytes, std::unique_lock<std::mutex>* lock);
    [[nodiscard]] std::size_t take_test_partial_pwritev_bytes() noexcept;
    void maybe_test_fault(DiskFaultPoint point) const;
    [[nodiscard]] EncodedStateBlob encode_state_blob(DiskStateKind kind, const void* a,
                                                     std::size_t na, const void* b,
                                                     std::size_t nb) const;
    std::uint64_t write_state_blob(const EncodedStateBlob& blob,
                                   std::unique_lock<std::mutex>& lock);
    std::uint64_t write_page_object(DiskObjectKind kind, const void* payload, std::size_t bytes,
                                    std::unique_lock<std::mutex>& lock);
    void persist_checkpoints(SpillSession& session, std::unique_lock<std::mutex>& lock);
    void write_meta_bin(std::uint64_t entry_id, const DiskMeta& meta, bool* renamed = nullptr);
    [[nodiscard]] bool idle_rewrite_of(std::uint64_t entry_id) const noexcept;
    [[nodiscard]] bool should_abandon_commit(const SpillSession& session) const noexcept;
    void maybe_payload_io_stall() const;
    void finish_payload_io(std::unique_lock<std::mutex>& lock);
    void close_restore_session_locked();
    void try_close_restore_session_locked();
    void discard_prefetch_queue();
    void purge_prefetch_of(std::uint64_t entry_id);
    void publish_page_job_failure(const Job& job);
    void bump_durable_generation() noexcept { ++durable_generation_; }
    std::uint32_t share_pages(SpeculativeBackend backend, bool main, std::uint32_t share_tokens)
        const;

    bool read_object(DiskObjectKind kind, std::uint64_t id, void* dst, std::uint64_t bytes);
    bool read_page_payload(DiskObjectKind kind, std::uint64_t id, void* io, void* dst,
                           std::size_t payload);
    bool decode_state(std::uint64_t id, DiskStateKind expected, void* dst, std::size_t bytes);

    struct StateDecodeJob {
        std::uint64_t id   = 0;
        DiskStateKind kind = DiskStateKind::TailHidden;
        void* dst          = nullptr;
        std::size_t bytes  = 0;
        bool ok            = true;
        void* dst2         = nullptr;
        std::size_t bytes2 = 0;
    };
    struct CheckpointDecodePlan {
        DiskMeta meta;
        std::unordered_map<std::uint64_t, DecodedCheckpointState> decoded;
        std::vector<StateDecodeJob> jobs;
        bool skip[3]{};
        std::size_t hidden_bytes[3]{};
    };
    bool decode_state_parts(std::uint64_t id, DiskStateKind expected, void* dst,
                            std::size_t bytes, void* dst2, std::size_t bytes2,
                            std::uint32_t read_workers = 1);
    bool decode_state_parallel(std::vector<StateDecodeJob>& jobs);
    [[nodiscard]] bool enqueue_unique_decode(
        std::unordered_map<std::uint64_t, DecodedCheckpointState>& decoded,
        std::vector<StateDecodeJob>& jobs, std::uint64_t id, DiskStateKind kind,
        std::size_t first_bytes, std::size_t second_bytes = 0);
    [[nodiscard]] bool collect_checkpoint_decode_jobs(
        const DiskMeta& meta,
        std::unordered_map<std::uint64_t, DecodedCheckpointState>& decoded,
        std::vector<StateDecodeJob>& jobs, bool skip[3], std::size_t hid_n[3]);
    [[nodiscard]] bool assemble_checkpoint_images(
        DiskRestoredHost& host, const DiskMeta& meta,
        std::unordered_map<std::uint64_t, DecodedCheckpointState>& decoded, const bool skip[3],
        const std::size_t hid_n[3]);
    [[nodiscard]] bool prepare_checkpoint_decode(
        DiskRestoredHost& host, CheckpointDecodePlan& plan, std::uint64_t epoch,
        std::uint64_t entry, std::uint64_t committed_generation,
        std::uint64_t claim_generation, PrefixReusePath reuse, std::uint32_t reuse_base,
        bool check_live);
    [[nodiscard]] bool finish_checkpoint_decode(
        DiskRestoredHost& host, CheckpointDecodePlan& plan, std::uint64_t epoch,
        std::uint64_t entry, std::uint64_t committed_generation,
        std::uint64_t claim_generation, PrefixReusePath reuse, std::uint32_t reuse_base,
        bool check_live);

    void fill_window_slot(std::uint32_t slot, DiskObjectKind kind, std::uint64_t object_id,
                          std::uint32_t pool, std::uint32_t logical);
    void reclaim_completed_h2d_slots();
    void wait_h2d_slots_locked();
    void h2d_ready_slots(cudaStream_t stream);
    void finish_restore_state();

    DiskOpenConfig config_;
    static constexpr std::uint64_t kPackSegmentBytes = 1ULL << 30;
    std::uint64_t pack_generation_ = 1;
    std::shared_ptr<PackGeneration> active_generation_;
    std::vector<std::shared_ptr<PackGeneration>> retired_generations_;
    bool packset_publication_pending_sync_ = false;
    std::uint64_t object_id_reservation_limit_ = 1;
    std::array<std::uint32_t, 5> pack_active_segment_{};
    std::array<std::uint64_t, 5> pack_active_tail_{};
    int lock_fd_ = -1;
    void* restore_window_allocation_                   = nullptr;
    void* restore_window_mem_                          = nullptr;
    std::size_t restore_window_bytes_                  = 0;
    std::size_t restore_window_stride_                 = 0;
    void* page_staging_                                = nullptr;
    std::size_t page_staging_bytes_                    = 0;
    std::size_t page_staging_stride_                   = 0;
    cudaStream_t page_scatter_stream_                  = nullptr;
    cudaStream_t scatter_test_gate_stream_             = nullptr;
    cudaEvent_t scatter_test_gate_                     = nullptr;
    PagedKVScatterPlane* text_scatter_planes_          = nullptr;
    std::size_t text_scatter_plane_count_              = 0;
    std::size_t text_scatter_max_plane_bytes_          = 0;
    std::size_t text_scatter_page_bytes_               = 0;
    PagedKVScatterPlane* backend_scatter_planes_       = nullptr;
    std::size_t backend_scatter_plane_count_           = 0;
    std::size_t backend_scatter_max_plane_bytes_       = 0;
    std::size_t backend_scatter_page_bytes_            = 0;
    struct ImmediateStateSlices {
        std::span<std::uint8_t> gdn_conv;
        std::span<std::uint8_t> gdn_rec;
        std::span<std::uint8_t> hidden;
        std::span<std::uint8_t> cyclic;
        std::span<std::uint8_t> rewrite_gdn_conv;
        std::span<std::uint8_t> rewrite_gdn_rec;
        std::span<std::uint8_t> rewrite_hidden;
        std::span<std::uint8_t> rewrite_cyclic;
    };
    void* restore_state_allocation_ = nullptr;
    void* restore_state_mem_ = nullptr;
    std::size_t restore_state_arena_bytes_ = 0;
    ImmediateStateSlices restore_state_capacity_{};
    ImmediateStateSlices restore_state_slices_{};
    cudaEvent_t state_arena_idle_ = nullptr;
    bool state_arena_h2d_pending_ = false;
    cudaStream_t state_h2d_stream_ = nullptr;
    std::vector<std::uint8_t> spill_scratch_;
    std::vector<std::uint8_t> spill_zstd_;
    std::vector<std::uint8_t> decode_zstd_;

    std::unordered_map<std::uint64_t, IndexEntry> entries_;
    std::deque<std::uint64_t> fifo_;
    std::unordered_map<std::uint64_t, ObjectRef> objects_;
    std::unordered_map<std::uint64_t, StartupObjectValidation> startup_validated_objects_;
    bool startup_loading_index_ = false;
    std::vector<SkippedTree> skipped_;
    std::unordered_map<std::uint64_t, RamNote> ram_notes_;
    std::uint64_t unique_bytes_     = 0;
    std::uint64_t next_object_id_   = 1;
    std::uint64_t next_entry_id_    = 1;
    std::uint64_t next_spill_epoch_ = 1;
    std::atomic<std::uint64_t> index_version_{1};
    std::uint64_t captures_         = 0;
    std::uint64_t restores_         = 0;
    std::uint64_t evictions_        = 0;
    std::uint64_t drops_            = 0;
    std::uint64_t exact_comparisons_ = 0;
    std::uint64_t durable_generation_ = 1;
    double save_seconds_            = 0;
    double load_seconds_            = 0;
    double pending_save_seconds_    = 0;
    double pending_h2d_seconds_     = 0;
    std::optional<std::chrono::steady_clock::time_point> load_host_start_;
    std::optional<std::chrono::steady_clock::time_point> load_host_end_;
    std::optional<std::uint64_t> test_free_bytes_override_;
    std::atomic<std::size_t> test_partial_pwritev_bytes_{0};
    mutable std::atomic<DiskFaultPoint> test_fault_point_{DiskFaultPoint::None};
    mutable std::atomic<DiskFaultPoint> test_next_fault_point_{DiskFaultPoint::None};

    mutable std::mutex mutex_;
    std::condition_variable cv_;
    std::condition_variable idle_cv_;
    std::thread io_thread_;
    std::vector<std::thread> restore_threads_{};
    std::uint32_t restore_io_threads_ = 1;
    bool stopping_ = false;
    std::deque<Job> restore_q_;
    std::deque<Job> emergency_q_;
    std::deque<Job> idle_q_;
    std::deque<Job> prefetch_q_;
    struct ReaderClaim {
        std::uint64_t disk_entry_id         = 0;
        std::uint64_t restore_epoch        = 0;
        std::uint64_t claim_generation       = 0;
        std::uint64_t committed_generation  = 0;
        std::uint32_t pool                 = 0;
        std::uint32_t logical_index        = 0;
    };
    std::vector<ReaderClaim> reader_claims_{};

    std::unordered_set<std::uint64_t> idle_cancel_rams_;
    bool idle_pinning_              = false;
    std::uint64_t idle_pinning_ram_ = 0;
    bool idle_cancel_all_           = false;
    std::uint64_t idle_cancel_epoch_ = 0;
    std::optional<SpillSession> spill_;
    std::optional<DiskRestoreTarget> restore_target_;
    std::uint64_t restore_entry_id_ = 0;
    bool restore_failed_            = false;
    bool restore_kv_done_           = false;
    bool restore_state_done_        = false;
    std::uint32_t restore_next_main_ = 0;
    std::uint32_t restore_next_backend_ = 0;
    std::uint32_t restore_h2d_main_      = 0;
    std::uint32_t restore_h2d_backend_   = 0;
    std::vector<WindowSlot> window_{};
    cudaEvent_t copies_start_ = nullptr;
    cudaEvent_t copies_done_  = nullptr;
    std::uint64_t copies_record_epoch_ = 0;
    std::uint32_t copies_done_waiters_ = 0;
    std::uint32_t copies_ticket_refs_   = 0;
    struct RetiredCopyEvent {
        cudaEvent_t event     = nullptr;
        cudaEvent_t start     = nullptr;
        std::uint64_t epoch   = 0;
        std::uint32_t waiters = 0;
        std::uint32_t ticket_refs = 0;
        bool timed            = false;
    };
    std::vector<RetiredCopyEvent> retired_copy_events_;
    bool copies_timed_        = false;
    bool copies_join_recorded_ = false;
    bool h2d_billed_          = false;
    bool restore_active_      = false;
    bool restore_state_loaded_ = false;
    bool restore_use_context_head_ = false;
    bool restore_unpack_rewrite_ = true;
    std::uint64_t restore_epoch_     = 1;
    std::uint64_t wait_ticket_epoch_  = 0;
    std::uint32_t window_inflight_   = 0;
    std::uint32_t payload_io_inflight_ = 0;
    std::uint32_t restore_state_inflight_ = 0;
    std::atomic<int> payload_io_stall_ms_{0};
    mutable std::atomic<bool> payload_io_entered_{false};
    std::atomic<int> manifest_io_stall_ms_{0};
    mutable std::atomic<bool> manifest_io_entered_{false};
    std::atomic<int> fsync_stall_ms_{0};
    mutable std::atomic<bool> fsync_entered_{false};
    std::atomic<bool> restore_job_barrier_armed_{false};
    std::atomic<bool> restore_job_dequeued_{false};
    std::atomic<bool> restore_job_continue_{true};
    std::atomic<bool> restore_state_barrier_armed_{false};
    std::atomic<bool> restore_state_entered_{false};
    std::atomic<bool> restore_state_continue_{true};
    std::atomic<bool> copy_lease_barrier_armed_{false};
    std::atomic<bool> copy_event_leased_{false};
    std::atomic<bool> copy_lease_continue_{true};
    std::atomic<bool> wait_epoch_barrier_armed_{false};
    std::atomic<bool> wait_epoch_latched_{false};
    std::atomic<bool> wait_epoch_continue_{true};
    std::atomic<bool> page_read_barrier_armed_{false};
    std::atomic<bool> page_read_entered_{false};
    std::atomic<bool> page_read_continue_{true};
    mutable std::atomic<int> page_read_count_{0};
    mutable std::atomic<std::uint64_t> restore_loop_idle_spins_{0};
    std::atomic<bool> payload_take_barrier_armed_{false};
    std::atomic<bool> payload_take_entered_{false};
    std::atomic<bool> payload_take_continue_{true};
    std::atomic<bool> slot_assign_barrier_armed_{false};
    std::atomic<bool> slot_assign_entered_{false};
    std::atomic<bool> slot_assign_continue_{true};
    std::atomic<bool> scatter_record_barrier_armed_{false};
    std::atomic<bool> scatter_record_entered_{false};
    std::atomic<bool> scatter_record_continue_{true};
    std::atomic<std::uint32_t> scatter_record_logical_{0};
    std::atomic<bool> fail_after_payload_take_{false};
    std::atomic<std::uint64_t> prefetch_preempted_idle_{0};
    cudaEvent_t last_wait_copy_event_ = nullptr;
    std::uint32_t timing_harvests_     = 0;
    std::atomic<int> state_decode_stall_ms_{0};
    mutable std::atomic<int> state_decode_count_{0};
    std::atomic<bool> fail_next_page_read_{false};
    std::atomic<bool> fail_next_restore_state_setup_{false};
    std::atomic<bool> fail_after_checkpoint_prepare_start_{false};
    std::atomic<bool> checkpoint_prepare_barrier_armed_{false};
    std::atomic<bool> checkpoint_prepare_barrier_entered_{false};
    std::atomic<bool> checkpoint_prepare_barrier_continue_{true};
    std::atomic<bool> fail_after_state_h2d_enqueue_{false};
    std::atomic<bool> direct_state_read_barrier_armed_{false};
    std::atomic<bool> direct_state_read_entered_{false};
    std::atomic<bool> direct_state_read_continue_{true};
    std::atomic<bool> fail_next_wait_copies_{false};
    std::vector<std::filesystem::path> pending_unlinks_;
    std::atomic<bool> meta_renamed_{false};
    std::atomic<bool> objects_dir_fsynced_{false};
    DiskRestoredHost restore_checkpoint_host_{};
    bool idle_requested_      = false;
    bool crash_before_meta_   = false;
    bool stall_after_meta_rename_ = false;
    bool fail_after_meta_rename_  = false;
    bool fail_rollback_meta_      = false;
    bool fail_after_rollback_rename_ = false;
    bool fail_tombstone_              = false;
    mutable bool fail_object_write_         = false;
    bool fail_object_unlink_                = false;
    bool fail_entry_unlink_                 = false;
    bool fail_prepare_spill_               = false;
    bool force_zstd_fail_     = false;
    std::filesystem::path canonical_location_;
    std::vector<std::uint64_t> branch_shared_ids_;
};

DiskFingerprint make_disk_fingerprint(std::string model_id, std::string weights_id,
                                      KvCacheStorage kv_cache, SpeculativeBackend speculative,
                                      const PagedKVPool& text, const PagedKVPool* backend,
                                      const LinearAttentionStatePool* gdn,
                                      const CyclicKVCache* cyclic);

} // namespace ninfer::targets::qwen3_6::detail
