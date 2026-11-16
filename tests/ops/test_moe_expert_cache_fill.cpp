// moe_expert_cache_fill: exact copies of device-mapped pinned host records of both codecs into
// their pending slots, only for the selected layer column, with every other slot byte untouched;
// commit afterwards publishes the slots (tagged expert-table entries) and clears the fills; an
// all-zero column copies nothing; and a captured graph replayed with changed fill entries copies
// the records the entries name at each replay.

#include "core/arena.h"
#include "core/decode_graph.h"
#include "core/device.h"
#include "ninfer/ops/moe_expert_cache.h"
#include "ninfer/ops/moe_experts.h"
#include "ops/op_tester.h"
#include "ops/sanitizer_scope.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <exception>
#include <iostream>
#include <memory>
#include <random>
#include <string>
#include <vector>

using namespace ninfer;
using namespace ninfer::test;

namespace {

constexpr std::int32_t kExperts = ops::kMoeExperts;
constexpr std::int32_t kLayers  = 3;
constexpr std::int32_t kLayer   = 1;
constexpr std::uint8_t kCanary  = 0xa5;

std::size_t record_bytes(ops::MoeExpertCodec codec) {
    return static_cast<std::size_t>(ops::moe_expert_record_bytes(codec));
}

// One pending fill: expert e of kLayer, its codec, and the slot index in the slot buffer.
struct Pending {
    std::int32_t expert;
    ops::MoeExpertCodec codec;
};

// Host records (pinned, device-mapped) and device slots for a list of pending fills. Slot i is
// separated from slot i+1 by a 256-byte canary gap, and the slot buffer starts and ends with one.
class Fixture {
public:
    explicit Fixture(const std::vector<Pending>& pending, std::uint32_t seed) : pending_(pending) {
        std::size_t host_bytes = 0;
        std::size_t slot_bytes = ops::kMoeRecordAlignment;
        for (const Pending& p : pending_) {
            host_offsets_.push_back(host_bytes);
            slot_offsets_.push_back(slot_bytes);
            host_bytes += record_bytes(p.codec);
            slot_bytes += record_bytes(p.codec) + ops::kMoeRecordAlignment;
        }
        host_ = std::make_unique<PinnedHostBuffer>(std::max<std::size_t>(host_bytes, 256), 256);
        std::mt19937 random(seed);
        auto* bytes = static_cast<std::uint8_t*>(host_->data());
        for (std::size_t i = 0; i < host_bytes; ++i) {
            bytes[i] = static_cast<std::uint8_t>(random());
        }
        slots_ = DeviceBuffer(slot_bytes);
        reset_slots();

        // Every entry names a host address: pending experts their record, the others a decoy
        // address that the Op must never read (the first record's).
        host_table_.assign(static_cast<std::size_t>(kExperts) * kLayers,
                           reinterpret_cast<std::uint64_t>(host_->data()));
        fill_table_.assign(host_table_.size(), 0);
        for (std::size_t i = 0; i < pending_.size(); ++i) {
            const std::size_t entry = static_cast<std::size_t>(kLayer) * kExperts +
                                      static_cast<std::size_t>(pending_[i].expert);
            host_table_[entry]      = reinterpret_cast<std::uint64_t>(bytes + host_offsets_[i]) |
                                      static_cast<std::uint64_t>(pending_[i].codec);
            fill_table_[entry]      = slot_address(i);
        }
        // Other columns hold pending-looking entries aimed at the slot buffer's leading canary gap:
        // a fill that read the wrong column would overwrite canary bytes.
        for (std::int32_t layer = 0; layer < kLayers; ++layer) {
            if (layer == kLayer) { continue; }
            fill_table_[static_cast<std::size_t>(layer) * kExperts + 3] =
                reinterpret_cast<std::uint64_t>(slots_.p);
        }
        host_device_ = DeviceBuffer(host_table_.size() * sizeof(std::uint64_t));
        fill_device_ = DeviceBuffer(fill_table_.size() * sizeof(std::uint64_t));
        upload_tables();
    }

    [[nodiscard]] std::uint64_t slot_address(std::size_t i) const {
        return reinterpret_cast<std::uint64_t>(slots_.p) + slot_offsets_.at(i);
    }

    void reset_slots() { slots_.fill(kCanary); }

    void upload_tables() {
        host_device_.copy_from_host(host_table_.data(), host_device_.bytes);
        fill_device_.copy_from_host(fill_table_.data(), fill_device_.bytes);
    }

    [[nodiscard]] Tensor host_tensor() const {
        return Tensor(host_device_.p, DType::I64, {kExperts, kLayers});
    }

    [[nodiscard]] Tensor fill_tensor() const {
        return Tensor(fill_device_.p, DType::I64, {kExperts, kLayers});
    }

    // Keeps only pending fill `kept` in kLayer's column.
    void keep_only(std::size_t kept) {
        for (std::size_t i = 0; i < pending_.size(); ++i) {
            if (i == kept) { continue; }
            fill_table_[static_cast<std::size_t>(kLayer) * kExperts +
                        static_cast<std::size_t>(pending_[i].expert)] = 0;
        }
        upload_tables();
    }

    // Clears kLayer's fill entries (no pending fill).
    void clear_column() {
        std::fill_n(fill_table_.begin() + static_cast<std::ptrdiff_t>(kLayer) * kExperts, kExperts,
                    0);
        upload_tables();
    }

    // Compares every slot with its host record (pending) or the canary (not pending), and every
    // gap with the canary; returns the failures.
    [[nodiscard]] int check_slots(const std::vector<bool>& filled, const std::string& label) const {
        std::vector<std::uint8_t> device(slots_.bytes);
        slots_.copy_to_host(device.data(), device.size());
        const auto* host = static_cast<const std::uint8_t*>(host_->data());
        std::vector<bool> covered(device.size(), false);
        int failures = 0;
        for (std::size_t i = 0; i < pending_.size(); ++i) {
            const std::size_t bytes = record_bytes(pending_[i].codec);
            const std::uint8_t* got = device.data() + slot_offsets_[i];
            const bool equal =
                filled.at(i)
                    ? std::memcmp(got, host + host_offsets_[i], bytes) == 0
                    : std::all_of(got, got + bytes, [](std::uint8_t b) { return b == kCanary; });
            if (!equal) {
                std::cerr << label << ": slot " << i << " (expert " << pending_[i].expert
                          << ") holds wrong bytes\n";
                ++failures;
            }
            std::fill_n(covered.begin() + static_cast<std::ptrdiff_t>(slot_offsets_[i]), bytes,
                        true);
        }
        for (std::size_t b = 0; b < device.size(); ++b) {
            if (!covered[b] && device[b] != kCanary) {
                std::cerr << label << ": gap byte " << b << " was written\n";
                ++failures;
                break;
            }
        }
        return failures;
    }

    // The tables must be unchanged by fill.
    [[nodiscard]] int check_tables_unchanged(const std::string& label) const {
        std::vector<std::uint64_t> host(host_table_.size());
        std::vector<std::uint64_t> fill(fill_table_.size());
        host_device_.copy_to_host(host.data(), host_device_.bytes);
        fill_device_.copy_to_host(fill.data(), fill_device_.bytes);
        const int failures =
            static_cast<int>(host != host_table_) + static_cast<int>(fill != fill_table_);
        if (failures != 0) { std::cerr << label << ": fill changed a table\n"; }
        return failures;
    }

    // Runs commit on an expert table whose pending entries hold their host address, as admit
    // leaves them, and checks the published entries and the cleared fill column.
    [[nodiscard]] int check_commit(cudaStream_t stream, const std::string& label) {
        DeviceBuffer expert_device(host_device_.bytes);
        expert_device.copy_from_host(host_table_.data(), expert_device.bytes);
        Tensor expert(expert_device.p, DType::I64, {kExperts, kLayers});
        Tensor fill = fill_tensor();
        ops::moe_expert_cache_commit(expert, fill, kLayer, stream);
        cuda_synchronize(stream);
        std::vector<std::uint64_t> published(host_table_.size());
        std::vector<std::uint64_t> fills(fill_table_.size());
        expert_device.copy_to_host(published.data(), expert_device.bytes);
        fill_device_.copy_to_host(fills.data(), fill_device_.bytes);
        int failures = 0;
        for (std::size_t entry = 0; entry < published.size(); ++entry) {
            const bool column  = entry / kExperts == static_cast<std::size_t>(kLayer);
            const bool pending = column && fill_table_[entry] != 0;
            const std::uint64_t expected_expert =
                pending ? fill_table_[entry] | (host_table_[entry] & 1U) : host_table_[entry];
            const std::uint64_t expected_fill = column ? 0 : fill_table_[entry];
            if (published[entry] != expected_expert || fills[entry] != expected_fill) {
                std::cerr << label << ": commit entry " << entry << " is wrong\n";
                ++failures;
                break;
            }
        }
        upload_tables();
        return failures;
    }

private:
    std::vector<Pending> pending_;
    std::vector<std::size_t> host_offsets_;
    std::vector<std::size_t> slot_offsets_;
    std::unique_ptr<PinnedHostBuffer> host_;
    DeviceBuffer slots_;
    std::vector<std::uint64_t> host_table_;
    std::vector<std::uint64_t> fill_table_;
    DeviceBuffer host_device_;
    DeviceBuffer fill_device_;
};

int run_eager(const std::vector<Pending>& pending, std::uint32_t seed, cudaStream_t stream,
              const std::string& label) {
    Fixture fixture(pending, seed);
    int failures = 0;
    ops::moe_expert_cache_fill(fixture.host_tensor(), fixture.fill_tensor(), kLayer, stream);
    cuda_synchronize(stream);
    failures += fixture.check_slots(std::vector<bool>(pending.size(), true), label);
    failures += fixture.check_tables_unchanged(label);
    failures += fixture.check_commit(stream, label);

    // No pending fill in the column: nothing is written.
    fixture.reset_slots();
    fixture.clear_column();
    ops::moe_expert_cache_fill(fixture.host_tensor(), fixture.fill_tensor(), kLayer, stream);
    cuda_synchronize(stream);
    failures += fixture.check_slots(std::vector<bool>(pending.size(), false), label + " empty");
    return failures;
}

// A graph captured once copies, at each replay, the records the current fill entries name.
int run_graph(cudaStream_t stream) {
    const std::vector<Pending> pending = {{2, ops::MoeExpertCodec::F3G32},
                                          {130, ops::MoeExpertCodec::Nvfp4G16},
                                          {511, ops::MoeExpertCodec::F3G32}};
    Fixture fixture(pending, 77);
    DecodeGraphDefinition definition;
    definition.capture(stream, [&] {
        ops::moe_expert_cache_fill(fixture.host_tensor(), fixture.fill_tensor(), kLayer, stream);
    });
    DecodeGraphExecutable graph;
    graph.instantiate(definition);
    int failures = 0;
    graph.launch(stream);
    cuda_synchronize(stream);
    failures += fixture.check_slots({true, true, true}, "graph replay 0");
    fixture.reset_slots();
    fixture.keep_only(1);
    graph.launch(stream);
    cuda_synchronize(stream);
    failures += fixture.check_slots({false, true, false}, "graph replay 1 (one pending)");
    fixture.reset_slots();
    fixture.clear_column();
    graph.launch(stream);
    cuda_synchronize(stream);
    failures += fixture.check_slots({false, false, false}, "graph replay 2 (empty)");
    return failures;
}

} // namespace

int main(int argc, char** argv) {
    if (require_cuda() != 0) { return 1; }
    try {
        DeviceContext context;
        int failures = 0;
        if (sanitizer_scope(argc, argv)) {
            failures +=
                run_eager({{5, ops::MoeExpertCodec::Nvfp4G16}, {6, ops::MoeExpertCodec::F3G32}}, 3,
                          context.stream, "sanitizer");
        } else {
            // A decode-sized miss set, a layer-wide set of both codecs (most of the column), and
            // the column's first and last entries.
            std::vector<Pending> wide;
            for (std::int32_t e = 0; e < kExperts; e += 9) {
                wide.push_back(
                    {e, e % 4 == 0 ? ops::MoeExpertCodec::Nvfp4G16 : ops::MoeExpertCodec::F3G32});
            }
            failures += run_eager({{17, ops::MoeExpertCodec::F3G32},
                                   {64, ops::MoeExpertCodec::Nvfp4G16},
                                   {400, ops::MoeExpertCodec::F3G32}},
                                  1, context.stream, "decode");
            failures += run_eager(wide, 2, context.stream, "wide");
            failures += run_eager(
                {{0, ops::MoeExpertCodec::Nvfp4G16}, {511, ops::MoeExpertCodec::Nvfp4G16}}, 4,
                context.stream, "ends");
            failures += run_graph(context.stream);
        }
        std::cout << (failures == 0 ? "OK" : "FAIL") << " moe_expert_cache_fill\n";
        return failures == 0 ? 0 : 1;
    } catch (const std::exception& error) {
        std::cerr << "moe_expert_cache_fill: " << error.what() << '\n';
        return 1;
    }
}
