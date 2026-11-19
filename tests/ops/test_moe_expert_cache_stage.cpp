// moe_expert_cache_stage against an exact host oracle of its contract: scores from the first k
// predicted ranks, non-resident candidates in descending score (ties to the lower expert), greedy
// whole-record placement within the byte budget, and the stage_ptr and source columns, over random
// predictions (including duplicate and out-of-range ids), residency, codec mixes, k, T and budgets
// from empty through exact-fit boundaries to larger than every candidate. The tables and the
// predictions stay untouched. A real-record case then runs the staging fill (12 CTAs) from mapped
// host records into the staging span, fills slots from the source column, as the layer's admitted
// misses would be filled, and merges the deferral (moe_expert_cache_defer), checking every byte
// and entry against the host records and addresses, eagerly and in a captured graph replayed with
// changed predictions and residency. moe_expert_cache_defer is also checked on random columns.

#include "core/arena.h"
#include "core/decode_graph.h"
#include "core/device.h"
#include "ninfer/ops/moe_expert_cache.h"
#include "ninfer/ops/moe_experts.h"
#include "ninfer/ops/moe_route.h"
#include "ops/op_tester.h"
#include "ops/sanitizer_scope.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <exception>
#include <iostream>
#include <memory>
#include <numeric>
#include <random>
#include <string>
#include <vector>

using namespace ninfer;
using namespace ninfer::test;

namespace {

constexpr std::int32_t kExperts = ops::kMoeExperts;
constexpr std::int32_t kTopK    = ops::kMoeTopK;
constexpr std::int32_t kLayers  = 3;
constexpr std::int32_t kLayer   = 2;
constexpr std::uint8_t kCanary  = 0xa5;
// Synthetic addresses for the placement cases, never dereferenced by the Op: host records 4 MiB
// apart, resident slots and the staging span in distinct ranges far from any device allocation.
constexpr std::uint64_t kHostBase    = 0x7f00'0000'0000ULL;
constexpr std::uint64_t kSlotBase    = 0x7e00'0000'0000ULL;
constexpr std::uint64_t kStagingBase = 0x7d00'0000'0000ULL;

std::uint64_t record_bytes(std::uint64_t host_entry) {
    return static_cast<std::uint64_t>(
        ops::moe_expert_record_bytes(static_cast<ops::MoeExpertCodec>(host_entry & 1U)));
}

struct Staged {
    std::vector<std::uint64_t> stage, source;
};

// The contract's outputs for column `layer`, written independently of the kernel's phases: exact
// integer scores, a full sort of the candidates, and the greedy walk over every candidate. With an
// eligibility set, only eligible experts are candidates.
Staged oracle(const std::vector<std::int32_t>& predicted, std::int32_t tokens, std::int32_t k,
              const std::vector<std::uint64_t>& expert, const std::vector<std::uint64_t>& host,
              std::int32_t layer, std::uint64_t base, std::uint64_t budget,
              const std::vector<bool>* eligible = nullptr) {
    std::vector<std::uint64_t> score(kExperts, 0);
    for (std::int32_t t = 0; t < tokens; ++t) {
        for (std::int32_t j = 0; j < k; ++j) {
            const std::int32_t id = predicted[static_cast<std::size_t>(t) * kTopK + j];
            if (id >= 0 && id < kExperts) {
                score[static_cast<std::size_t>(id)] += 360360U / static_cast<std::uint64_t>(4 + j);
            }
        }
    }
    const auto entry = [layer](std::int32_t e) {
        return static_cast<std::size_t>(layer) * kExperts + static_cast<std::size_t>(e);
    };
    std::vector<std::int32_t> candidates;
    for (std::int32_t e = 0; e < kExperts; ++e) {
        if (score[static_cast<std::size_t>(e)] > 0 && expert[entry(e)] == host[entry(e)] &&
            (eligible == nullptr || (*eligible)[static_cast<std::size_t>(e)])) {
            candidates.push_back(e);
        }
    }
    std::sort(candidates.begin(), candidates.end(), [&](std::int32_t a, std::int32_t b) {
        const auto sa = score[static_cast<std::size_t>(a)];
        const auto sb = score[static_cast<std::size_t>(b)];
        return sa != sb ? sa > sb : a < b;
    });
    Staged out{std::vector<std::uint64_t>(kExperts, 0), std::vector<std::uint64_t>(kExperts, 0)};
    std::uint64_t used = 0;
    for (const std::int32_t e : candidates) {
        const std::uint64_t size = record_bytes(host[entry(e)]);
        if (used + size <= budget) {
            out.stage[static_cast<std::size_t>(e)] = base + used;
            used += size;
        }
    }
    for (std::int32_t e = 0; e < kExperts; ++e) {
        const auto i  = static_cast<std::size_t>(e);
        out.source[i] = out.stage[i] != 0 ? out.stage[i] | (host[entry(e)] & 1U) : host[entry(e)];
    }
    return out;
}

struct Tables {
    std::vector<std::uint64_t> expert, host;
};

// Random tables over kLayers columns: each host entry a synthetic record with a random codec
// (nvfp4_share of codec 0), each expert resident with probability `resident`.
Tables random_tables(double nvfp4_share, double resident, std::mt19937& rng) {
    std::uniform_real_distribution<double> unit(0.0, 1.0);
    Tables t;
    const std::size_t entries = static_cast<std::size_t>(kExperts) * kLayers;
    for (std::size_t entry = 0; entry < entries; ++entry) {
        const std::uint64_t tag = unit(rng) < nvfp4_share ? 0 : 1;
        t.host.push_back((kHostBase + entry * (std::uint64_t{4} << 20)) | tag);
        t.expert.push_back(unit(rng) < resident ? (kSlotBase + entry * 4096) | tag : t.host.back());
    }
    return t;
}

// How a case draws its predicted ids: ten distinct ids per token from a small hot set (as routed
// tokens share experts), any ids with repeats inside a token, or distinct ids with some outside
// [0,512).
enum class Ids : std::uint8_t { DistinctHot, Repeats, OutOfRange };

std::vector<std::int32_t> random_predictions(std::int32_t tokens, Ids kind, std::mt19937& rng) {
    std::vector<std::int32_t> ids(static_cast<std::size_t>(tokens) * kTopK);
    std::uniform_int_distribution<std::int32_t> any(0, kExperts - 1);
    std::uniform_int_distribution<std::int32_t> hot(0, 63);
    std::uniform_int_distribution<std::int32_t> outside(0, 3);
    for (std::int32_t t = 0; t < tokens; ++t) {
        auto* row = ids.data() + static_cast<std::size_t>(t) * kTopK;
        if (kind == Ids::Repeats) {
            std::generate_n(row, kTopK, [&] { return hot(rng); });
            continue;
        }
        std::int32_t filled = 0;
        while (filled < kTopK) {
            const std::int32_t id = filled < 6 ? hot(rng) * 8 : any(rng);
            if (std::find(row, row + filled, id) == row + filled) { row[filled++] = id; }
        }
        if (kind == Ids::OutOfRange) {
            const std::int32_t pick = outside(rng);
            if (pick == 0) { row[pick] = -1; }
            if (pick == 1) { row[kTopK - 1] = kExperts; }
        }
    }
    return ids;
}

int placement_cases(cudaStream_t stream) {
    std::mt19937 rng(1234);
    int failures = 0;

    const struct {
        const char* name;
        double nvfp4_share;
        double resident;
        Ids ids;
    } mixes[] = {{"mixed", 0.12, 0.6, Ids::DistinctHot},
                 {"nothing resident", 0.3, 0.0, Ids::DistinctHot},
                 {"all resident", 0.2, 1.0, Ids::DistinctHot},
                 {"nvfp4 only", 1.0, 0.4, Ids::Repeats},
                 {"f3 only", 0.0, 0.5, Ids::OutOfRange}};

    const std::int32_t token_counts[] = {1, 4, 20, 96, ops::kMoeStageMaxTokens};
    const std::int32_t ks[]           = {1, 4, 10};
    for (const auto& mix : mixes) {
        const Tables t = random_tables(mix.nvfp4_share, mix.resident, rng);
        GuardedDeviceBuffer expert(t.expert.size() * sizeof(std::uint64_t));
        GuardedDeviceBuffer host(t.host.size() * sizeof(std::uint64_t));
        GuardedDeviceBuffer stage(static_cast<std::size_t>(kExperts) * sizeof(std::uint64_t));
        GuardedDeviceBuffer source(static_cast<std::size_t>(kExperts) * sizeof(std::uint64_t));
        expert.copy_from_host(t.expert.data(), expert.bytes());
        host.copy_from_host(t.host.data(), host.bytes());
        const Tensor expert_view(expert.data(), DType::I64, {kExperts, kLayers});
        const Tensor host_view(host.data(), DType::I64, {kExperts, kLayers});
        Tensor stage_view(stage.data(), DType::I64, {kExperts});
        Tensor source_view(source.data(), DType::I64, {kExperts});
        for (const std::int32_t tokens : token_counts) {
            const std::vector<std::int32_t> ids = random_predictions(tokens, mix.ids, rng);
            GuardedDeviceBuffer predicted(ids.size() * sizeof(std::int32_t));
            predicted.copy_from_host(ids.data(), predicted.bytes());
            const Tensor predicted_view(predicted.data(), DType::I32, {kTopK, tokens});
            // A random eligibility bitset (about 60% eligible), used on half of the calls.
            std::vector<bool> eligible(kExperts);
            std::vector<std::uint32_t> words(ops::kMoeExpertMaskWords, 0);
            std::uniform_real_distribution<double> unit(0.0, 1.0);
            for (std::size_t e = 0; e < eligible.size(); ++e) {
                eligible[e] = unit(rng) < 0.6;
                if (eligible[e]) { words[e / 32] |= 1U << (e % 32); }
            }
            GuardedDeviceBuffer eligible_storage(words.size() * sizeof(std::uint32_t));
            eligible_storage.copy_from_host(words.data(), eligible_storage.bytes());
            const Tensor eligible_view(eligible_storage.data(), DType::I32,
                                       {ops::kMoeExpertMaskWords});
            constexpr std::uint64_t kF3   = ops::kMoeF3RecordBytes;
            constexpr std::uint64_t kNv   = ops::kMoeNvfp4RecordBytes;
            const std::uint64_t budgets[] = {
                0,         kF3 - 1, kF3,       kNv - 1,    kNv,
                kF3 + kNv, 2 * kF3, 6'000'000, 15'000'000, std::uint64_t{1} << 31};
            for (const std::int32_t k : ks) {
                for (const std::uint64_t budget : budgets) {
                    const bool restricted   = unit(rng) < 0.5;
                    const std::string label = std::string("stage ") + mix.name + " T " +
                                              std::to_string(tokens) + " k " + std::to_string(k) +
                                              " budget " + std::to_string(budget) +
                                              (restricted ? " eligible" : "");
                    stage.fill(0x3c);
                    source.fill(0x3c);
                    // cudaMemset runs on the legacy stream, which does not order the Op's stream.
                    cuda_check(cudaDeviceSynchronize(), "cudaDeviceSynchronize");
                    // A synthetic staging span: the Op computes placements from it and never
                    // dereferences it.
                    const DeviceSpan span{
                        budget == 0 ? nullptr
                                    : reinterpret_cast<void*>( // NOLINT(performance-no-int-to-ptr)
                                          kStagingBase),
                        budget};
                    ops::moe_expert_cache_stage(predicted_view, k, expert_view, host_view, kLayer,
                                                span, budget, stage_view, source_view, stream,
                                                restricted ? &eligible_view : nullptr);
                    cuda_synchronize(stream);
                    const Staged want = oracle(ids, tokens, k, t.expert, t.host, kLayer,
                                               budget == 0 ? 0 : kStagingBase, budget,
                                               restricted ? &eligible : nullptr);
                    failures += verify_exact((label + " stage_ptr").c_str(),
                                             from_device<std::uint64_t>(stage.data(), kExperts),
                                             want.stage);
                    failures += verify_exact((label + " source").c_str(),
                                             from_device<std::uint64_t>(source.data(), kExperts),
                                             want.source);
                    failures += stage.verify_guards(label + " stage_ptr");
                    failures += source.verify_guards(label + " source");
                }
            }
            failures += verify_exact((std::string("stage predicted ") + mix.name).c_str(),
                                     from_device<std::int32_t>(predicted.data(), ids.size()), ids);
            failures += predicted.verify_guards(std::string("stage predicted ") + mix.name);
        }
        failures +=
            verify_exact((std::string("stage expert_ptr ") + mix.name).c_str(),
                         from_device<std::uint64_t>(expert.data(), t.expert.size()), t.expert);
        failures += verify_exact((std::string("stage host_ptr ") + mix.name).c_str(),
                                 from_device<std::uint64_t>(host.data(), t.host.size()), t.host);
        failures += expert.verify_guards(std::string("stage expert_ptr ") + mix.name);
        failures += host.verify_guards(std::string("stage host_ptr ") + mix.name);
    }
    return failures;
}

// moe_expert_cache_defer over random fill and stage columns: the fill entry where it is nonzero,
// else the stage entry; the inputs stay untouched.
int defer_cases(cudaStream_t stream) {
    std::mt19937 rng(4321);
    std::uniform_real_distribution<double> unit(0.0, 1.0);
    std::vector<std::uint64_t> fill(static_cast<std::size_t>(kExperts) * kLayers, 0);
    std::vector<std::uint64_t> stage(kExperts, 0);
    for (std::size_t i = 0; i < fill.size(); ++i) {
        if (unit(rng) < 0.3) { fill[i] = kSlotBase + i * 4096; }
    }
    for (std::size_t e = 0; e < stage.size(); ++e) {
        if (unit(rng) < 0.2) { stage[e] = kStagingBase + e * 4096; }
    }
    std::vector<std::uint64_t> want(kExperts);
    for (std::size_t e = 0; e < want.size(); ++e) {
        const std::uint64_t slot = fill[static_cast<std::size_t>(kLayer) * kExperts + e];
        want[e]                  = slot != 0 ? slot : stage[e];
    }
    GuardedDeviceBuffer fill_device(fill.size() * sizeof(std::uint64_t));
    GuardedDeviceBuffer stage_device(stage.size() * sizeof(std::uint64_t));
    GuardedDeviceBuffer deferred(static_cast<std::size_t>(kExperts) * sizeof(std::uint64_t));
    fill_device.copy_from_host(fill.data(), fill_device.bytes());
    stage_device.copy_from_host(stage.data(), stage_device.bytes());
    deferred.fill(0x3c);
    cuda_check(cudaDeviceSynchronize(), "cudaDeviceSynchronize");
    const Tensor fill_view(fill_device.data(), DType::I64, {kExperts, kLayers});
    const Tensor stage_view(stage_device.data(), DType::I64, {kExperts});
    Tensor deferred_view(deferred.data(), DType::I64, {kExperts});
    ops::moe_expert_cache_defer(fill_view, kLayer, stage_view, deferred_view, stream);
    cuda_synchronize(stream);
    int failures =
        verify_exact("defer", from_device<std::uint64_t>(deferred.data(), kExperts), want);
    failures += verify_exact("defer fill_ptr",
                             from_device<std::uint64_t>(fill_device.data(), fill.size()), fill);
    failures += verify_exact("defer stage_ptr",
                             from_device<std::uint64_t>(stage_device.data(), stage.size()), stage);
    failures += deferred.verify_guards("defer deferred");
    return failures;
}

// Real records for kRecords experts of kLayer (ids below 400) with alternating codecs, in pinned
// mapped host memory; every other entry is a resident decoy (a slot address no Op may read). Each
// token predicts the listed experts at ranks 0..4, in list or reversed order, then decoys; k = 4,
// so the fifth ranked expert is never a candidate, and the budget holds three of the four
// candidates. The layer's admitted misses are emulated by a fill column naming device slots for
// those of the first kAdmitted listed experts that are not resident.
class StagedRecords {
public:
    static constexpr int kRecords  = 5;
    static constexpr int kAdmitted = 3;
    static constexpr int kTokens   = 20;

    explicit StagedRecords(std::uint32_t seed) {
        std::mt19937 rng(seed);
        std::uniform_int_distribution<std::int32_t> pick(0, 399);
        while (static_cast<int>(experts_.size()) < kRecords) {
            const std::int32_t e = pick(rng);
            if (std::find(experts_.begin(), experts_.end(), e) == experts_.end()) {
                experts_.push_back(e);
            }
        }
        std::size_t host_bytes = 0;
        for (int i = 0; i < kRecords; ++i) {
            codecs_.push_back(i % 2 == 0 ? ops::MoeExpertCodec::F3G32
                                         : ops::MoeExpertCodec::Nvfp4G16);
            offsets_.push_back(host_bytes);
            host_bytes += size(i);
        }
        records_    = std::make_unique<PinnedHostBuffer>(host_bytes, 256);
        auto* bytes = static_cast<std::uint8_t*>(records_->data());
        for (std::size_t i = 0; i < host_bytes; ++i) {
            bytes[i] = static_cast<std::uint8_t>(rng());
        }
        // The four candidates less 256 bytes: the fourth in rank order does not fit.
        budget_ = offsets_[kRecords - 1] - ops::kMoeRecordAlignment;
        // One spare NVFP4 record of canary after the budget.
        staging_               = DeviceBuffer(budget_ + ops::kMoeNvfp4RecordBytes);
        std::size_t slot_bytes = 0;
        for (int i = 0; i < kAdmitted; ++i) {
            slot_offsets_.push_back(slot_bytes);
            slot_bytes += size(i);
        }
        slots_ = DeviceBuffer(slot_bytes);
        host_.assign(static_cast<std::size_t>(kExperts) * kLayers, 0);
        expert_.assign(host_.size(), 0);
        for (std::size_t entry = 0; entry < host_.size(); ++entry) {
            host_[entry]   = (kHostBase + entry * 4096) | 1U;
            expert_[entry] = (kSlotBase + entry * 4096) | 1U;
        }
        for (int i = 0; i < kRecords; ++i) {
            host_[entry(i)] = reinterpret_cast<std::uint64_t>(bytes + offsets_[i]) |
                              static_cast<std::uint64_t>(codecs_[i]);
        }
        host_device_ = DeviceBuffer(host_.size() * sizeof(std::uint64_t));
        host_device_.copy_from_host(host_.data(), host_device_.bytes);
        expert_device_ = DeviceBuffer(expert_.size() * sizeof(std::uint64_t));
        fill_          = DeviceBuffer(static_cast<std::size_t>(kExperts) * sizeof(std::uint64_t));
        predicted_ = DeviceBuffer(static_cast<std::size_t>(kTokens) * kTopK * sizeof(std::int32_t));
        stage_     = DeviceBuffer(static_cast<std::size_t>(kExperts) * sizeof(std::uint64_t));
        source_    = DeviceBuffer(static_cast<std::size_t>(kExperts) * sizeof(std::uint64_t));
        deferred_  = DeviceBuffer(static_cast<std::size_t>(kExperts) * sizeof(std::uint64_t));
        set_resident(0);
        set_order(false);
    }

    // Makes listed expert i resident (a decoy slot) for i in [0, count): it is neither staged nor,
    // being a hit, admitted.
    void set_resident(int count) {
        std::vector<std::uint64_t> fill(kExperts, 0);
        for (int i = 0; i < kRecords; ++i) {
            expert_[entry(i)] =
                i < count ? (kSlotBase + entry(i) * 4096) | static_cast<std::uint64_t>(codecs_[i])
                          : host_[entry(i)];
            if (i >= count && i < kAdmitted) {
                fill[static_cast<std::size_t>(experts_[static_cast<std::size_t>(i)])] =
                    reinterpret_cast<std::uint64_t>(slots_.p) + slot_offsets_[i];
            }
        }
        resident_ = count;
        expert_device_.copy_from_host(expert_.data(), expert_device_.bytes);
        fill_.copy_from_host(fill.data(), fill_.bytes);
    }

    // Every token predicts the listed experts at ranks 0..4 (reversed: the last one first), then
    // the resident decoys 495..491.
    void set_order(bool reversed) {
        std::vector<std::int32_t> ids(static_cast<std::size_t>(kTokens) * kTopK);
        for (int t = 0; t < kTokens; ++t) {
            for (int j = 0; j < kTopK; ++j) {
                const int i = reversed ? kRecords - 1 - j : j;
                ids[static_cast<std::size_t>(t) * kTopK + static_cast<std::size_t>(j)] =
                    j < kRecords ? experts_[static_cast<std::size_t>(i)] : 500 - j;
            }
        }
        reversed_ = reversed;
        predicted_.copy_from_host(ids.data(), predicted_.bytes);
    }

    void enqueue(cudaStream_t stream) {
        const Tensor expert(expert_device_.p, DType::I64, {kExperts, kLayers});
        const Tensor host(host_device_.p, DType::I64, {kExperts, kLayers});
        const Tensor predicted(predicted_.p, DType::I32, {kTopK, kTokens});
        Tensor stage(stage_.p, DType::I64, {kExperts});
        Tensor source(source_.p, DType::I64, {kExperts});
        ops::moe_expert_cache_stage(predicted, 4, expert, host, kLayer, {staging_.p, budget_},
                                    budget_, stage, source, stream);
        ops::moe_expert_cache_fill(host.slice(1, kLayer, 1), stage.view({kExperts, 1}), 0, stream,
                                   ops::kMoeStagingFillBlocks);
        const Tensor fill(fill_.p, DType::I64, {kExperts, 1});
        ops::moe_expert_cache_fill(source.view({kExperts, 1}), fill, 0, stream);
        Tensor deferred(deferred_.p, DType::I64, {kExperts});
        ops::moe_expert_cache_defer(fill, 0, stage, deferred, stream);
    }

    // cudaMemset runs on the legacy stream, which does not order the Ops' stream.
    void reset() {
        staging_.fill(kCanary);
        slots_.fill(kCanary);
        cuda_check(cudaDeviceSynchronize(), "cudaDeviceSynchronize");
    }

    // The staging span holds the staged records in rank order, then canary; the slot of each
    // admitted (non-resident) expert holds its record, staged or mapped, and the others canary.
    [[nodiscard]] int check(const std::string& label) const {
        const auto* bytes = static_cast<const std::uint8_t*>(records_->data());
        // The four candidates by score: predicted ranks 0..3, in list or reversed order.
        std::vector<int> ranked = {0, 1, 2, 3};
        if (reversed_) {
            std::transform(ranked.begin(), ranked.end(), ranked.begin(),
                           [](int j) { return kRecords - 1 - j; });
        }
        std::vector<std::uint8_t> expected(staging_.bytes, kCanary);
        // The deferral: an admitted (non-resident) expert's slot, else a staged one's span address.
        std::vector<std::uint64_t> deferred(kExperts, 0);
        std::size_t used = 0;
        for (const int i : ranked) {
            if (i < resident_ || used + size(i) > budget_) { continue; }
            std::copy_n(bytes + offsets_[i], size(i),
                        expected.begin() + static_cast<std::ptrdiff_t>(used));
            deferred[static_cast<std::size_t>(experts_[static_cast<std::size_t>(i)])] =
                reinterpret_cast<std::uint64_t>(staging_.p) + used;
            used += size(i);
        }
        for (int i = resident_; i < kAdmitted; ++i) {
            deferred[static_cast<std::size_t>(experts_[static_cast<std::size_t>(i)])] =
                reinterpret_cast<std::uint64_t>(slots_.p) + slot_offsets_[i];
        }
        std::vector<std::uint8_t> got(staging_.bytes);
        staging_.copy_to_host(got.data(), got.size());
        int failures = verify_exact((label + " staging").c_str(), got, expected);
        std::vector<std::uint8_t> slot_expected(slots_.bytes, kCanary);
        for (int i = 0; i < kAdmitted; ++i) {
            if (i < resident_) { continue; }
            std::copy_n(bytes + offsets_[i], size(i),
                        slot_expected.begin() + static_cast<std::ptrdiff_t>(slot_offsets_[i]));
        }
        std::vector<std::uint8_t> slot_got(slots_.bytes);
        slots_.copy_to_host(slot_got.data(), slot_got.size());
        failures += verify_exact((label + " promoted slots").c_str(), slot_got, slot_expected);
        failures += verify_exact((label + " deferral").c_str(),
                                 from_device<std::uint64_t>(deferred_.p, kExperts), deferred);
        return failures;
    }

private:
    [[nodiscard]] std::size_t size(int i) const {
        return static_cast<std::size_t>(ops::moe_expert_record_bytes(codecs_[i]));
    }

    [[nodiscard]] std::size_t entry(int i) const {
        return static_cast<std::size_t>(kLayer) * kExperts +
               static_cast<std::size_t>(experts_[static_cast<std::size_t>(i)]);
    }

    std::vector<std::int32_t> experts_;
    std::vector<ops::MoeExpertCodec> codecs_;
    std::vector<std::size_t> offsets_, slot_offsets_;
    std::size_t budget_ = 0;
    int resident_       = 0;
    bool reversed_      = false;
    std::unique_ptr<PinnedHostBuffer> records_;
    DeviceBuffer staging_, slots_, host_device_, expert_device_, predicted_, fill_, stage_, source_,
        deferred_;
    std::vector<std::uint64_t> host_, expert_;
};

int record_cases(cudaStream_t stream, bool graph) {
    StagedRecords records(17);
    records.reset();
    records.enqueue(stream);
    cuda_synchronize(stream);
    int failures = records.check("staged records, the last candidate over budget");
    if (!graph) { return failures; }
    DecodeGraphDefinition definition;
    DecodeGraphExecutable executable;
    definition.capture(stream, [&] { records.enqueue(stream); });
    executable.instantiate(definition);

    const struct {
        int resident;
        bool reversed;
    } replays[] = {{0, true}, {2, false}, {StagedRecords::kRecords, true}};

    for (const auto& replay : replays) {
        records.set_resident(replay.resident);
        records.set_order(replay.reversed);
        records.reset();
        executable.launch(stream);
        cuda_synchronize(stream);
        failures += records.check("staged graph replay, " + std::to_string(replay.resident) +
                                  " resident" + (replay.reversed ? ", reversed" : ""));
    }
    return failures;
}

} // namespace

int main(int argc, char** argv) {
    if (require_cuda() != 0) { return 1; }
    try {
        DeviceContext context;
        const bool sanitizer = sanitizer_scope(argc, argv);
        int failures =
            sanitizer ? 0 : placement_cases(context.stream) + defer_cases(context.stream);
        failures += record_cases(context.stream, !sanitizer);
        std::cout << (failures == 0 ? "OK" : "FAIL") << " moe_expert_cache_stage\n";
        return failures == 0 ? 0 : 1;
    } catch (const std::exception& error) {
        std::cerr << "moe_expert_cache_stage: " << error.what() << '\n';
        return 1;
    }
}
