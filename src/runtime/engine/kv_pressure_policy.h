#pragma once

#include <algorithm>
#include <array>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <optional>
#include <span>
#include <stdexcept>
#include <vector>

namespace ninfer::runtime {

// Logical identity survives physical-lane release. suspend_ready means that the target has
// secured an allocatable image credit and the settled lane can be captured without rollback.
struct KvPressureRequest {
    std::uint64_t id               = 0;
    std::uint64_t queue_order      = 0;
    bool resident                  = false;
    bool growth_blocked            = false;
    bool completion_member         = false;
    bool suspend_ready             = false;
    bool useful_chunk_since_resume = true;
    std::size_t suspension_bytes   = 0;
    std::uint32_t suspension_count = 0;
};

enum class KvPressureReason : std::uint8_t { WaitingReplacement, Growth };

struct KvPressureVictims {
    // Bit i names resident[i] in the snapshot passed to ranked_victims. These are candidate
    // sets, not resource grants: the caller must prove exact complete post-transition fit.
    std::uint16_t mask               = 0;
    std::uint8_t count               = 0;
    std::uint8_t unblocked_count     = 0;
    std::size_t transfer_bytes       = 0;
    std::uint64_t suspension_count   = 0;
    std::uint64_t oldest_queue_order = 0;
};

struct KvPressureVictimOrder {
    std::array<KvPressureVictims, 255> sets{};
    std::size_t count = 0;

    [[nodiscard]] std::span<const KvPressureVictims> candidates() const noexcept {
        return {sets.data(), count};
    }
};

// Worker-local RAM-on policy. Resource geometry, image reservations, transfer ownership and
// the atomic completion-cohort handoff belong to the executor/target transaction.
class KvPressurePolicy {
public:
    KvPressurePolicy(std::uint32_t concurrency, std::size_t max_outstanding)
        : concurrency_(concurrency), max_outstanding_(max_outstanding) {
        if (concurrency == 0 || concurrency > 8 || max_outstanding < concurrency) {
            throw std::invalid_argument("invalid KV pressure request bounds");
        }
        incumbent_ids_.reserve(max_outstanding);
    }

    // Call only on settled boundary snapshots, including suspended and transfer-held admitted
    // identities. executable_work includes a legal prefill unit; outstanding transfer progress
    // should be awaited before the caller initiates another pressure transaction.
    void observe(std::span<const KvPressureRequest> admitted, bool executable_work) {
        if (admitted.size() > max_outstanding_) {
            throw std::logic_error("KV pressure logical admission bound exceeded");
        }
        if (oldest_incumbent_ &&
            std::none_of(admitted.begin(), admitted.end(), [this](const auto& request) {
                return request.id == *oldest_incumbent_;
            })) {
            clear_epoch();
        }
        const auto blocked   = std::count_if(admitted.begin(), admitted.end(), [](const auto& r) {
            return r.resident && r.growth_blocked;
        });
        const bool suspended = std::any_of(admitted.begin(), admitted.end(),
                                           [](const auto& r) { return !r.resident; });
        no_executable_work_  = !executable_work;
        pressure_needed_     = blocked >= 2 || (!executable_work && !admitted.empty());
        if (!oldest_incumbent_ && (blocked != 0 || suspended)) {
            const auto oldest = std::min_element(admitted.begin(), admitted.end(), older);
            oldest_incumbent_ = oldest->id;
            for (const auto& request : admitted) { incumbent_ids_.push_back(request.id); }
            bypass_remaining_ = concurrency_ - 1;
        }
        if (oldest_incumbent_ && (pressure_needed_ || bypass_remaining_ == 0)) { draining_ = true; }
    }

    [[nodiscard]] bool allows_ingress(std::size_t admitted_count,
                                      bool has_suspended) const noexcept {
        return admitted_count < max_outstanding_ && !has_suspended && !draining_ &&
               (!oldest_incumbent_ || bypass_remaining_ != 0);
    }

    // Debit only after an actual successful new admission, never on feasibility probes or
    // resume. Early completion does not refund credit; only oldest-incumbent departure resets.
    void record_ingress() {
        if (draining_ || (oldest_incumbent_ && bypass_remaining_ == 0)) {
            throw std::logic_error("KV pressure admission during drain");
        }
        if (oldest_incumbent_) {
            --bypass_remaining_;
            if (bypass_remaining_ == 0) { draining_ = true; }
        }
    }

    [[nodiscard]] bool draining() const noexcept { return draining_; }

    [[nodiscard]] bool pressure_needed() const noexcept { return pressure_needed_; }

    [[nodiscard]] std::uint32_t bypass_remaining() const noexcept { return bypass_remaining_; }

    [[nodiscard]] std::optional<std::uint64_t> oldest_incumbent() const noexcept {
        return oldest_incumbent_;
    }

    [[nodiscard]] std::span<const std::uint64_t> incumbent_ids() const noexcept {
        return incumbent_ids_;
    }

    [[nodiscard]] static std::optional<std::uint64_t>
    oldest_suspended(std::span<const KvPressureRequest> admitted) noexcept {
        const KvPressureRequest* oldest = nullptr;
        for (const auto& request : admitted) {
            if (!request.resident && (!oldest || older(request, *oldest))) { oldest = &request; }
        }
        return oldest ? std::optional(oldest->id) : std::nullopt;
    }

    [[nodiscard]] KvPressureVictimOrder ranked_victims(std::span<const KvPressureRequest> resident,
                                                       KvPressureReason reason) const {
        if (resident.size() > concurrency_) {
            throw std::logic_error("KV pressure resident bound exceeded");
        }
        KvPressureVictimOrder result;
        if (reason == KvPressureReason::Growth && !pressure_needed_ && !draining_) {
            return result;
        }
        if (reason == KvPressureReason::WaitingReplacement && draining_) { return result; }
        std::uint16_t eligible = 0;
        for (std::size_t i = 0; i < resident.size(); ++i) {
            const auto& r = resident[i];
            if (r.resident && !r.completion_member && r.suspend_ready &&
                (r.useful_chunk_since_resume || no_executable_work_) &&
                (reason != KvPressureReason::WaitingReplacement || r.growth_blocked)) {
                eligible |= static_cast<std::uint16_t>(1U << i);
            }
        }
        for (std::uint16_t mask = eligible; mask != 0;
             mask               = static_cast<std::uint16_t>((mask - 1U) & eligible)) {
            auto& set              = result.sets[result.count++];
            set.mask               = mask;
            set.count              = static_cast<std::uint8_t>(std::popcount(mask));
            set.oldest_queue_order = std::numeric_limits<std::uint64_t>::max();
            for (std::size_t i = 0; i < resident.size(); ++i) {
                if ((mask & (1U << i)) == 0) { continue; }
                const auto& r = resident[i];
                set.unblocked_count += static_cast<std::uint8_t>(!r.growth_blocked);
                set.transfer_bytes = saturated_add(set.transfer_bytes, r.suspension_bytes);
                set.suspension_count += r.suspension_count;
                set.oldest_queue_order = std::min(set.oldest_queue_order, r.queue_order);
            }
        }
        std::sort(result.sets.begin(), result.sets.begin() + result.count,
                  [](const auto& a, const auto& b) {
                      if (a.count != b.count) { return a.count < b.count; }
                      if (a.unblocked_count != b.unblocked_count) {
                          return a.unblocked_count < b.unblocked_count;
                      }
                      if (a.transfer_bytes != b.transfer_bytes) {
                          return a.transfer_bytes < b.transfer_bytes;
                      }
                      if (a.suspension_count != b.suspension_count) {
                          return a.suspension_count < b.suspension_count;
                      }
                      if (a.oldest_queue_order != b.oldest_queue_order) {
                          return a.oldest_queue_order > b.oldest_queue_order;
                      }
                      return a.mask < b.mask;
                  });
        return result;
    }

private:
    [[nodiscard]] static bool older(const KvPressureRequest& a,
                                    const KvPressureRequest& b) noexcept {
        return a.queue_order < b.queue_order || (a.queue_order == b.queue_order && a.id < b.id);
    }

    [[nodiscard]] static std::size_t saturated_add(std::size_t a, std::size_t b) noexcept {
        return b > std::numeric_limits<std::size_t>::max() - a
                   ? std::numeric_limits<std::size_t>::max()
                   : a + b;
    }

    void clear_epoch() noexcept {
        oldest_incumbent_.reset();
        incumbent_ids_.clear();
        bypass_remaining_ = 0;
        draining_         = false;
    }

    std::uint32_t concurrency_;
    std::size_t max_outstanding_;
    std::optional<std::uint64_t> oldest_incumbent_;
    std::vector<std::uint64_t> incumbent_ids_;
    std::uint32_t bypass_remaining_ = 0;
    bool draining_                  = false;
    bool pressure_needed_           = false;
    bool no_executable_work_        = false;
};

} // namespace ninfer::runtime
