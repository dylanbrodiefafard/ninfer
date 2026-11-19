#include "runtime/expert_residency.h"

#include <algorithm>
#include <cstring>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <string>

namespace ninfer::targets::qwen3_8_flash_next {
namespace {

using ops::MoeExpertCodec;

constexpr std::size_t kCodecs = ops::kMoeCacheCodecTags;

constexpr std::size_t tag_of(MoeExpertCodec codec) { return static_cast<std::size_t>(codec); }

// First slot of each pool's CLOCK range: past the fixed slots when they hold the pinned tier.
std::array<std::int32_t, kCodecs> clock_firsts(const ExpertSlotPlan& plan, bool pinned) {
    return pinned ? plan.slots : std::array<std::int32_t, kCodecs>{};
}

std::size_t record_bytes(MoeExpertCodec codec) {
    return static_cast<std::size_t>(ops::moe_expert_record_bytes(codec));
}

void check_cuda(cudaError_t status, const char* operation) {
    if (status != cudaSuccess) {
        throw std::runtime_error(std::string("expert residency: ") + operation + ": " +
                                 cudaGetErrorString(status));
    }
}

std::int32_t checked_layers(std::size_t entries) {
    if (entries == 0 || entries % kRoutedExperts != 0 ||
        entries / kRoutedExperts > static_cast<std::size_t>(kExpertLayers)) {
        throw std::invalid_argument(
            "ExpertHostStore: codecs must hold layers * 512 entries, 1 <= layers <= 48");
    }
    return static_cast<std::int32_t>(entries / kRoutedExperts);
}

constexpr std::size_t kMetadataAlignment = 256;

// One layer column of a table: 512 U64 entries.
constexpr std::size_t kTableAlignment = static_cast<std::size_t>(kRoutedExperts) * 8;

// Slots of the pool of codec tag `tag`: its fixed slots and, for F3, the flex slots.
std::size_t pool_slots(const ExpertSlotPlan& plan, std::size_t flex_slots, std::size_t tag) {
    return static_cast<std::size_t>(plan.slots[tag]) +
           (tag == tag_of(MoeExpertCodec::F3G32) ? flex_slots : 0);
}

// Capacity of ExpertResidency's metadata arena, including every alignment gap.
std::size_t metadata_capacity(std::int32_t layers, const ExpertSlotPlan& plan,
                              std::size_t flex_slots) {
    const std::size_t table_bytes =
        static_cast<std::size_t>(layers) * kRoutedExperts * sizeof(std::uint64_t);
    // One term per allocation of the constructor, each with its worst-case alignment gap: three
    // tables, the slot index (I32 per expert), the hands, the frequency counts (I32 per expert)
    // and lookups, the flex address table (U64 per flex slot), then each pool's owner (I32) and
    // reference (U8) arrays over its fixed and flex slots.
    const std::size_t index_bytes = table_bytes / 2;
    std::size_t bytes = 3 * (table_bytes + kTableAlignment) + (index_bytes + kMetadataAlignment) +
                        (kCodecs * sizeof(std::int32_t) + kMetadataAlignment) +
                        (index_bytes + kMetadataAlignment) +
                        (sizeof(std::int32_t) + kMetadataAlignment) +
                        (flex_slots * sizeof(std::uint64_t) + kMetadataAlignment);
    for (std::size_t tag = 0; tag < kCodecs; ++tag) {
        if (plan.slots[tag] < 0) {
            throw std::invalid_argument("ExpertResidency: negative slot count");
        }
        bytes +=
            pool_slots(plan, flex_slots, tag) * (sizeof(std::int32_t) + 1) + 2 * kMetadataAlignment;
    }
    return bytes;
}

Tensor table_view(const DeviceSpan& table, std::int32_t layers) {
    return Tensor(table.data, DType::I64, {kRoutedExperts, layers});
}

} // namespace

ExpertHostStore::ExpertHostStore(std::span<const MoeExpertCodec> codecs, std::size_t chunk_bytes)
    : layers_(checked_layers(codecs.size())), codecs_(codecs.begin(), codecs.end()) {
    for (const MoeExpertCodec codec : codecs_) {
        if (codec != MoeExpertCodec::Nvfp4G16 && codec != MoeExpertCodec::F3G32) {
            throw std::invalid_argument("ExpertHostStore: unknown expert codec");
        }
        ++codec_counts_[tag_of(codec)];
    }

    // Chunk layout: consecutive records while the chunk stays within chunk_bytes. Record sizes are
    // multiples of 256 and chunks start page aligned, so every record is 256-byte aligned.
    std::vector<std::size_t> offsets(codecs_.size());
    std::vector<std::size_t> chunk_of(codecs_.size());
    std::vector<std::size_t> chunk_sizes;
    std::size_t used = 0;
    for (std::size_t index = 0; index < codecs_.size(); ++index) {
        const std::size_t bytes = record_bytes(codecs_[index]);
        if (chunk_sizes.empty() || used + bytes > chunk_bytes) {
            if (!chunk_sizes.empty()) { chunk_sizes.back() = used; }
            chunk_sizes.push_back(0);
            used = 0;
        }
        offsets[index]  = used;
        chunk_of[index] = chunk_sizes.size() - 1;
        used += bytes;
    }
    chunk_sizes.back() = used;

    chunks_.reserve(chunk_sizes.size());
    std::vector<std::byte*> host_bases;
    std::vector<std::uint64_t> device_bases;
    for (const std::size_t size : chunk_sizes) {
        HostPinnedArena& chunk = chunks_.emplace_back(size);
        void* block = chunk.try_alloc(size, static_cast<std::size_t>(ops::kMoeRecordAlignment));
        if (block == nullptr) {
            throw std::runtime_error("ExpertHostStore: pinned chunk cannot hold its records");
        }
        // HostPinnedArena registers its memory with cudaHostRegisterDefault, which maps it on
        // unified-addressing devices; the device address is queried rather than assumed equal.
        void* device = nullptr;
        check_cuda(cudaHostGetDevicePointer(&device, block, 0), "mapped host address");
        host_bases.push_back(static_cast<std::byte*>(block));
        device_bases.push_back(reinterpret_cast<std::uint64_t>(device));
    }

    records_.resize(codecs_.size());
    device_addresses_.resize(codecs_.size());
    for (std::size_t index = 0; index < codecs_.size(); ++index) {
        records_[index] = host_bases[chunk_of[index]] + offsets[index];
        device_addresses_[index] =
            (device_bases[chunk_of[index]] + offsets[index]) | tag_of(codecs_[index]);
    }
}

std::size_t ExpertHostStore::index(std::int32_t layer, std::int32_t expert) const {
    if (layer < 0 || layer >= layers_ || expert < 0 || expert >= kRoutedExperts) {
        throw std::out_of_range("ExpertHostStore: (layer, expert) out of range");
    }
    return static_cast<std::size_t>(layer) * kRoutedExperts + static_cast<std::size_t>(expert);
}

void ExpertHostStore::write_record(std::int32_t layer, std::int32_t expert,
                                   std::span<const std::byte> record) {
    const std::size_t at = index(layer, expert);
    if (record.size() != record_bytes(codecs_[at])) {
        throw std::invalid_argument("ExpertHostStore: record size does not match its codec");
    }
    std::memcpy(records_[at], record.data(), record.size());
}

std::span<const std::byte> ExpertHostStore::record(std::int32_t layer, std::int32_t expert) const {
    const std::size_t at = index(layer, expert);
    return {records_[at], record_bytes(codecs_[at])};
}

MoeExpertCodec ExpertHostStore::codec(std::int32_t layer, std::int32_t expert) const {
    return codecs_[index(layer, expert)];
}

std::size_t ExpertSlotPlan::bytes() const noexcept {
    std::size_t total = 0;
    for (std::size_t tag = 0; tag < kCodecs; ++tag) {
        total +=
            static_cast<std::size_t>(slots[tag]) * record_bytes(static_cast<MoeExpertCodec>(tag));
    }
    return total;
}

ExpertSlotPlan plan_expert_slots(std::size_t budget_bytes,
                                 std::array<std::int32_t, ops::kMoeCacheCodecTags> expert_counts) {
    const auto fit = [&expert_counts](std::size_t bytes, MoeExpertCodec codec) {
        const auto limit = static_cast<std::size_t>(std::max(expert_counts[tag_of(codec)], 0));
        return static_cast<std::int32_t>(std::min(bytes / record_bytes(codec), limit));
    };
    const std::size_t nvfp4_record = record_bytes(MoeExpertCodec::Nvfp4G16);
    const std::size_t f3_record    = record_bytes(MoeExpertCodec::F3G32);
    ExpertSlotPlan plan;
    std::int32_t& nvfp4 = plan.slots[tag_of(MoeExpertCodec::Nvfp4G16)];
    std::int32_t& f3    = plan.slots[tag_of(MoeExpertCodec::F3G32)];
    nvfp4               = fit(budget_bytes / 100 * kNvfp4BudgetPercent, MoeExpertCodec::Nvfp4G16);
    f3 = fit(budget_bytes - static_cast<std::size_t>(nvfp4) * nvfp4_record, MoeExpertCodec::F3G32);
    nvfp4 = fit(budget_bytes - static_cast<std::size_t>(f3) * f3_record, MoeExpertCodec::Nvfp4G16);
    return plan;
}

ExpertResidency::ExpertResidency(const ExpertHostStore& store, const ExpertSlotPlan& plan,
                                 std::span<const std::uint64_t> flex_addresses,
                                 std::span<const std::uint64_t> calib_counts, cudaStream_t stream,
                                 const ExpertTiers& tiers)
    : layers_(store.layers()), plan_(plan),
      metadata_(metadata_capacity(layers_, plan_, flex_addresses.size())) {
    const std::size_t experts = static_cast<std::size_t>(layers_) * kRoutedExperts;
    if (!calib_counts.empty() && calib_counts.size() != experts) {
        throw std::invalid_argument("ExpertResidency: calib_counts must be empty or [L * 512]");
    }
    const bool tiered = !tiers.primary_slots.empty() || !tiers.admit_eligibility.empty();
    if (tiered &&
        (tiers.primary_slots.size() != experts || tiers.admit_eligibility.size() != experts / 32)) {
        throw std::invalid_argument(
            "ExpertResidency: tiers must give I32[L*512] primary slots and U32[L,16] eligibility");
    }
    const std::span<const std::uint32_t> admit_eligibility = tiers.admit_eligibility;
    const auto eligible                                    = [&](std::size_t entry) {
        return !tiered || (admit_eligibility[entry / 32] & (std::uint32_t{1} << (entry % 32))) != 0;
    };
    constexpr auto kFlexTag = tag_of(MoeExpertCodec::F3G32);
    if (flex_addresses.size() > static_cast<std::size_t>(std::numeric_limits<std::int32_t>::max() -
                                                         plan_.slots[kFlexTag])) {
        throw std::invalid_argument("ExpertResidency: the F3 pool exceeds 2^31 - 1 slots");
    }
    if (std::ranges::any_of(flex_addresses, [](std::uint64_t address) {
            return address == 0 || address % ops::kMoeRecordAlignment != 0;
        })) {
        throw std::invalid_argument("ExpertResidency: flex addresses must be nonzero and 256-byte "
                                    "aligned");
    }
    flex_slots_ = static_cast<std::int32_t>(flex_addresses.size());
    for (std::size_t entry = 0; entry < experts; ++entry) {
        // Bit 0 of a host address is the record's codec tag.
        if (eligible(entry) && (store.device_addresses()[entry] & 1U) == kFlexTag) {
            ++admittable_f3_records_;
        }
    }
    pinned_ = tiered;

    // Calibration warms fixed+flex slots in rank order; with tiers only flex slots take admitted
    // records, because the fixed slots hold the pinned records.
    const std::span<const std::uint64_t> host = store.device_addresses();
    std::array<std::vector<std::int32_t>, kCodecs> warm;
    if (!calib_counts.empty()) {
        std::vector<std::int32_t> order(experts);
        std::iota(order.begin(), order.end(), 0);
        std::stable_sort(order.begin(), order.end(), [&](std::int32_t a, std::int32_t b) {
            return calib_counts[static_cast<std::size_t>(a)] >
                   calib_counts[static_cast<std::size_t>(b)];
        });
        for (const std::int32_t code : order) {
            const auto entry = static_cast<std::size_t>(code);
            if (calib_counts[entry] == 0) { break; }
            if (!eligible(entry)) { continue; }
            const std::size_t tag           = host[entry] & 1U;
            std::vector<std::int32_t>& list = warm[tag];
            const std::size_t capacity = tiered ? (tag == kFlexTag ? flex_addresses.size() : 0)
                                                : pool_slots(plan_, flex_addresses.size(), tag);
            if (list.size() < capacity) { list.push_back(code); }
        }
    }
    // Pinned records: the pool slot of each, checked against the fixed capacity before any
    // allocation.
    std::array<std::int32_t, kCodecs> pinned_counts{};
    if (tiered) {
        std::array<std::vector<bool>, kCodecs> taken;
        for (std::size_t tag = 0; tag < kCodecs; ++tag) {
            taken[tag].assign(static_cast<std::size_t>(plan_.slots[tag]), false);
        }
        for (std::size_t entry = 0; entry < experts; ++entry) {
            const std::int32_t slot = tiers.primary_slots[entry];
            if (slot < 0) { continue; }
            const std::size_t tag = host[entry] & 1U;
            const auto position   = static_cast<std::size_t>(slot);
            if (position >= taken[tag].size() || taken[tag][position] ||
                (admit_eligibility[entry / 32] >> (entry % 32) & 1U) != 0) {
                throw std::invalid_argument(
                    "ExpertResidency: a pinned record needs its own fixed slot and no admission");
            }
            taken[tag][position] = true;
            ++pinned_counts[tag];
        }
        if (pinned_counts != plan_.slots) {
            throw std::invalid_argument(
                "ExpertResidency: the pinned records must fill every fixed slot");
        }
    }

    // Metadata layout: three 4096-byte aligned tables, then the slot index, the hands, the
    // frequency state, the flex address table and the pool arrays.
    const std::size_t table_bytes = experts * sizeof(std::uint64_t);
    expert_table_                 = metadata_.alloc_bytes(table_bytes, kTableAlignment);
    host_table_                   = metadata_.alloc_bytes(table_bytes, kTableAlignment);
    fill_table_                   = metadata_.alloc_bytes(table_bytes, kTableAlignment);
    slot_index_        = metadata_.alloc_bytes(experts * sizeof(std::int32_t), kMetadataAlignment);
    clock_hands_       = metadata_.alloc_bytes(kCodecs * sizeof(std::int32_t), kMetadataAlignment);
    frequency_counts_  = metadata_.alloc_bytes(experts * sizeof(std::int32_t), kMetadataAlignment);
    frequency_lookups_ = metadata_.alloc_bytes(sizeof(std::int32_t), kMetadataAlignment);
    if (!flex_addresses.empty()) {
        flex_address_ = metadata_.alloc_bytes(flex_addresses.size_bytes(), kMetadataAlignment);
    }

    for (std::size_t tag = 0; tag < kCodecs; ++tag) {
        const auto fixed        = static_cast<std::size_t>(plan_.slots[tag]);
        const std::size_t slots = pool_slots(plan_, flex_addresses.size(), tag);
        if (slots == 0) { continue; }
        if (fixed != 0) {
            slots_[tag] = DeviceBuffer(fixed * record_bytes(static_cast<MoeExpertCodec>(tag)));
        }
        owner_[tag]      = metadata_.alloc_bytes(slots * sizeof(std::int32_t), kMetadataAlignment);
        referenced_[tag] = metadata_.alloc_bytes(slots, kMetadataAlignment);
    }

    const bool any_admitted =
        !tiered || std::ranges::any_of(admit_eligibility, [](std::uint32_t w) { return w != 0; });
    refill_enabled_                    = any_admitted;
    constexpr std::size_t kColumnBytes = kRoutedExperts * sizeof(std::uint64_t);
    constexpr std::size_t kStateOffset = 2 * kColumnBytes;
    constexpr std::size_t kMaskOffset  = kStateOffset + kMetadataAlignment;
    refill_storage_                    = DeviceBuffer(kMaskOffset + admit_eligibility.size_bytes());
    auto* refill_base                  = static_cast<std::byte*>(refill_storage_.p);
    refill_source_                     = Tensor(refill_base, DType::I64, {kRoutedExperts});
    refill_destination_ = Tensor(refill_base + kColumnBytes, DType::I64, {kRoutedExperts});
    refill_state_       = Tensor(refill_base + kStateOffset, DType::I32, {2});
    if (tiered) {
        refill_eligible_ =
            Tensor(refill_base + kMaskOffset, DType::I32, {ops::kMoeExpertMaskWords, layers_});
        check_cuda(cudaMemcpyAsync(refill_eligible_.data, admit_eligibility.data(),
                                   admit_eligibility.size_bytes(), cudaMemcpyHostToDevice, stream),
                   "refill eligibility upload");
    }

    // Every allocation precedes the first transfer. Pageable sources are staged before
    // cudaMemcpyAsync returns and warm records live in the store, so unwinding after a failed
    // call frees no host buffer a queued copy still reads.
    std::vector<std::uint64_t> expert_table(host.begin(), host.end());
    std::vector<std::int32_t> slot_index(experts, ops::kMoeCacheNoSlot);
    const std::array<std::int32_t, kCodecs> hands = clock_firsts(plan_, pinned_);
    std::array<std::vector<std::int32_t>, kCodecs> owners;
    for (std::size_t tag = 0; tag < kCodecs; ++tag) {
        const auto fixed        = static_cast<std::size_t>(plan_.slots[tag]);
        const std::size_t slots = pool_slots(plan_, flex_addresses.size(), tag);
        if (slots == 0) { continue; }
        const std::size_t record = record_bytes(static_cast<MoeExpertCodec>(tag));
        // Fixed slots lie in this pool's allocation, flex slots at the addresses their owner lent.
        const auto slot_address = [&](std::size_t slot) -> std::uint64_t {
            return slot < fixed ? reinterpret_cast<std::uint64_t>(slots_[tag].p) + slot * record
                                : flex_addresses[slot - fixed];
        };
        // Empty slots first, then the warm experts from coldest to hottest.
        std::vector<std::int32_t>& owner = owners[tag];
        owner.assign(slots, ops::kMoeCacheEmptySlot);
        for (std::size_t entry = 0; tiered && entry < experts; ++entry) {
            const std::int32_t pinned_slot = tiers.primary_slots[entry];
            if (pinned_slot < 0 || (host[entry] & 1U) != tag) { continue; }
            const std::uint64_t address = slot_address(static_cast<std::size_t>(pinned_slot));
            owner[static_cast<std::size_t>(pinned_slot)] = ops::kMoeCacheBorrowedSlot;
            // NOLINTNEXTLINE(performance-no-int-to-ptr): a device address as the Op tables hold it.
            void* const destination = reinterpret_cast<void*>(address);
            const auto code         = static_cast<std::int32_t>(entry);
            check_cuda(
                cudaMemcpyAsync(destination,
                                store.record(code / kRoutedExperts, code % kRoutedExperts).data(),
                                record, cudaMemcpyHostToDevice, stream),
                "pinned record copy");
            expert_table[entry] = address | tag;
        }
        for (std::size_t rank = 0; rank < warm[tag].size(); ++rank) {
            const std::int32_t code                    = warm[tag][rank];
            const std::size_t slot                     = slots - 1 - rank;
            const std::uint64_t address                = slot_address(slot);
            owner[slot]                                = code;
            slot_index[static_cast<std::size_t>(code)] = static_cast<std::int32_t>(slot);
            // NOLINTNEXTLINE(performance-no-int-to-ptr): a device address as the Op tables hold it.
            void* const destination = reinterpret_cast<void*>(address);
            check_cuda(
                cudaMemcpyAsync(destination,
                                store.record(code / kRoutedExperts, code % kRoutedExperts).data(),
                                record, cudaMemcpyHostToDevice, stream),
                "warm record copy");
            expert_table[static_cast<std::size_t>(code)] = address | tag;
        }
        check_cuda(cudaMemcpyAsync(owner_[tag].data, owner.data(), owner_[tag].bytes,
                                   cudaMemcpyHostToDevice, stream),
                   "owner upload");
        check_cuda(cudaMemsetAsync(referenced_[tag].data, 0, referenced_[tag].bytes, stream),
                   "reference bit clear");
    }
    check_cuda(cudaMemcpyAsync(slot_index_.data, slot_index.data(), slot_index_.bytes,
                               cudaMemcpyHostToDevice, stream),
               "slot index upload");
    if (!flex_addresses.empty()) {
        check_cuda(cudaMemcpyAsync(flex_address_.data, flex_addresses.data(), flex_address_.bytes,
                                   cudaMemcpyHostToDevice, stream),
                   "flex address upload");
    }
    // Each hand starts at its range's first slot (a pool with an empty range never reads it).
    check_cuda(cudaMemcpyAsync(clock_hands_.data, hands.data(), clock_hands_.bytes,
                               cudaMemcpyHostToDevice, stream),
               "hand upload");
    check_cuda(cudaMemsetAsync(frequency_counts_.data, 0, frequency_counts_.bytes, stream),
               "frequency clear");
    check_cuda(cudaMemsetAsync(frequency_lookups_.data, 0, frequency_lookups_.bytes, stream),
               "frequency lookups clear");
    check_cuda(cudaMemcpyAsync(expert_table_.data, expert_table.data(), table_bytes,
                               cudaMemcpyHostToDevice, stream),
               "expert table upload");
    check_cuda(
        cudaMemcpyAsync(host_table_.data, host.data(), table_bytes, cudaMemcpyHostToDevice, stream),
        "host table upload");
    check_cuda(cudaMemsetAsync(fill_table_.data, 0, table_bytes, stream), "fill table clear");
    // The state is complete for work on any stream once the constructor returns.
    check_cuda(cudaStreamSynchronize(stream), "initial state synchronization");
}

ExpertResidency::~ExpertResidency() {
    // Also covers an exception between queuing a copy and recording its completion event.
    if (refill_copy_stream_) {
        // A sticky device fault is reported by execution; teardown cannot propagate it.
        static_cast<void>(cudaStreamSynchronize(*refill_copy_stream_));
    }
}

void ExpertResidency::request_refill() {
    if (!refill_enabled_ || flex_slots_ == 0) { return; }
    refill_requested_ = true;
    ++refill_generation_;
}

void ExpertResidency::join_refill(cudaStream_t stream) {
    if (!refill_pending_) { return; }
    CUDA_CHECK(cudaStreamWaitEvent(stream, refill_done_.get(), 0));
    Tensor table = expert_table();
    ops::moe_expert_cache_refill_publish(table, host_table(), cache(), refill_state_, stream);
    refill_pending_ = false;
}

bool ExpertResidency::service_refill(cudaStream_t stream, cudaStream_t copy_stream, bool allow) {
    if (refill_pending_) {
        const auto status = cudaEventQuery(refill_done_.get());
        if (status == cudaErrorNotReady) { return true; }
        CUDA_CHECK(status);
        // Feedback belongs to this completed selection. A subsequent give_back must not be
        // lost if this selection found no placement before those slots became available.
        if (*static_cast<const std::int32_t*>(refill_result_.data()) < 0 &&
            selected_generation_ == refill_generation_) {
            refill_requested_ = false;
        }
        join_refill(stream);
    }
    if (allow && refill_requested_) {
        ops::moe_expert_cache_refill_select(
            host_table(), cache(), refill_state_, refill_source_, refill_destination_, stream,
            refill_eligible_.data != nullptr ? &refill_eligible_ : nullptr);
        CUDA_CHECK(cudaEventRecord(refill_selected_.get(), stream));
        refill_copy_stream_ = copy_stream;
        CUDA_CHECK(cudaStreamWaitEvent(copy_stream, refill_selected_.get(), 0));
        ops::moe_expert_cache_fill(refill_source_, refill_destination_, 0, copy_stream,
                                   ops::kMoeStagingFillBlocks);
        CUDA_CHECK(cudaMemcpyAsync(refill_result_.data(), refill_state_.data, sizeof(std::int32_t),
                                   cudaMemcpyDeviceToHost, copy_stream));
        CUDA_CHECK(cudaEventRecord(refill_done_.get(), copy_stream));
        selected_generation_ = refill_generation_;
        refill_pending_      = true;
    }
    return refill_pending_ || (allow && refill_requested_);
}

ops::MoeExpertCache ExpertResidency::cache() {
    ops::MoeExpertCache cache;
    constexpr auto kFlexTag                             = tag_of(MoeExpertCodec::F3G32);
    const std::array<std::int32_t, kCodecs> clock_first = clock_firsts(plan_, pinned_);
    // Slots that can hold an admitted record: the CLOCK ranges.
    std::int64_t admittable_slots = 0;
    for (std::size_t tag = 0; tag < kCodecs; ++tag) {
        const auto slots = static_cast<std::int32_t>(
            pool_slots(plan_, static_cast<std::size_t>(flex_slots_), tag));
        admittable_slots += slots - clock_first[tag];
        if (slots == 0) { continue; }
        ops::MoeExpertSlotPool& view = cache.pools[tag];
        view.slots                   = {slots_[tag].p, slots_[tag].bytes};
        view.record_bytes            = record_bytes(static_cast<MoeExpertCodec>(tag));
        if (tag == kFlexTag && flex_slots_ != 0) {
            view.flex_address = Tensor(flex_address_.data, DType::I64, {flex_slots_});
        }
        view.owner       = Tensor(owner_[tag].data, DType::I32, {slots});
        view.referenced  = Tensor(referenced_[tag].data, DType::U8, {slots});
        view.clock_first = clock_first[tag];
    }
    cache.slot_index      = Tensor(slot_index_.data, DType::I32, {kRoutedExperts, layers_});
    cache.clock_hands     = Tensor(clock_hands_.data, DType::I32, {ops::kMoeCacheCodecTags});
    cache.reference_limit = kExpertReferenceLimit;
    cache.frequency       = ops::MoeExpertFrequency{
        .counts  = Tensor(frequency_counts_.data, DType::I32, {kRoutedExperts, layers_}),
        .lookups = Tensor(frequency_lookups_.data, DType::I32, {1}),
        // Without an admittable slot the counts decide nothing (no admit has a victim and refill
        // is off), so the longest period the Op accepts spares the admits a halving pass over
        // every count; the clamp keeps the narrowing exact for any pool size.
        .halving_lookups = admittable_slots == 0 ? ops::kMoeCacheMaxHalvingLookups
                                                 : static_cast<std::int32_t>(std::min<std::int64_t>(
                                                       ops::kMoeCacheMaxHalvingLookups,
                                                       kFrequencyHalvingSlots * admittable_slots)),
    };
    return cache;
}

Tensor ExpertResidency::expert_table() { return table_view(expert_table_, layers_); }

Tensor ExpertResidency::host_table() { return table_view(host_table_, layers_); }

Tensor ExpertResidency::fill_table() { return table_view(fill_table_, layers_); }

} // namespace ninfer::targets::qwen3_8_flash_next
