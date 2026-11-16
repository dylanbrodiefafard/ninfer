// moe_expert_cache_window against an exact host oracle of its contract: placements of the
// non-resident records of one layer column (ascending expert, packed by record size, records past
// the window left at 0), over random residency and codec mixes and window sizes from empty through
// exact-fit boundaries to larger than the layer. Every other column and both tables stay
// untouched. A streamed case then feeds the placements to moe_expert_cache_fill as single-layer
// columns and checks the window bytes exactly against the mapped host records, eagerly and in a
// captured graph replayed with a changed residency.

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
// Synthetic addresses for the placement cases, never dereferenced by the Op: host records 4 MiB
// apart, resident slots and the window in distinct ranges far from any device allocation.
constexpr std::uint64_t kHostBase   = 0x7f00'0000'0000ULL;
constexpr std::uint64_t kSlotBase   = 0x7e00'0000'0000ULL;
constexpr std::uint64_t kWindowBase = 0x7d00'0000'0000ULL;

std::uint64_t record_bytes(std::uint64_t host_entry) {
    return static_cast<std::uint64_t>(
        ops::moe_expert_record_bytes(static_cast<ops::MoeExpertCodec>(host_entry & 1U)));
}

// The contract's placement of column `layer`.
std::vector<std::uint64_t> oracle(const std::vector<std::uint64_t>& expert,
                                  const std::vector<std::uint64_t>& host, std::int32_t layer,
                                  std::uint64_t base, std::uint64_t bytes) {
    std::vector<std::uint64_t> placed(kExperts, 0);
    std::uint64_t prefix = 0;
    for (std::int32_t e = 0; e < kExperts; ++e) {
        const auto entry = static_cast<std::size_t>(layer) * kExperts + static_cast<std::size_t>(e);
        if (expert[entry] != host[entry]) { continue; }
        const std::uint64_t size = record_bytes(host[entry]);
        if (prefix + size <= bytes) { placed[static_cast<std::size_t>(e)] = base + prefix; }
        prefix += size;
    }
    return placed;
}

// Random tables over kLayers columns: each host entry a synthetic record with a random codec
// (nvfp4_share of codec 0), each expert resident with probability `resident`.
struct Tables {
    std::vector<std::uint64_t> expert, host;
};

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

int placement_cases(cudaStream_t stream) {
    std::mt19937 rng(77);
    int failures = 0;

    const struct {
        const char* name;
        double nvfp4_share;
        double resident;
    } mixes[] = {{"mixed", 0.15, 0.5},
                 {"all streamed", 0.3, 0.0},
                 {"all resident", 0.2, 1.0},
                 {"nvfp4 only", 1.0, 0.4},
                 {"f3 only", 0.0, 0.6}};

    for (const auto& mix : mixes) {
        const Tables t = random_tables(mix.nvfp4_share, mix.resident, rng);
        GuardedDeviceBuffer expert(t.expert.size() * sizeof(std::uint64_t));
        GuardedDeviceBuffer host(t.host.size() * sizeof(std::uint64_t));
        GuardedDeviceBuffer placed(static_cast<std::size_t>(kExperts) * sizeof(std::uint64_t));
        expert.copy_from_host(t.expert.data(), expert.bytes());
        host.copy_from_host(t.host.data(), host.bytes());
        const Tensor expert_view(expert.data(), DType::I64, {kExperts, kLayers});
        const Tensor host_view(host.data(), DType::I64, {kExperts, kLayers});
        Tensor placed_view(placed.data(), DType::I64, {kExperts});

        // Window sizes: empty, one F3 record short of the first, the exact sum of the first k
        // streamed records and one byte less, half the layer, and larger than the whole layer.
        std::vector<std::uint64_t> sizes = {0, ops::kMoeF3RecordBytes - 1};
        std::uint64_t sum                = 0;
        int streamed                     = 0;
        for (std::int32_t e = 0; e < kExperts; ++e) {
            const auto entry = static_cast<std::size_t>(kLayer) * kExperts + e;
            if (t.expert[entry] != t.host[entry]) { continue; }
            sum += record_bytes(t.host[entry]);
            if (++streamed == 7) {
                sizes.push_back(sum);
                sizes.push_back(sum - 1);
            }
        }
        sizes.push_back(sum / 2);
        sizes.push_back(sum + (std::uint64_t{1} << 30));
        for (const std::uint64_t bytes : sizes) {
            const std::string label =
                std::string("window ") + mix.name + " bytes " + std::to_string(bytes);
            placed.fill(0x3c);
            // A synthetic window address: the Op computes placements from it and never
            // dereferences it.
            const DeviceSpan window{
                bytes == 0
                    ? nullptr
                    : reinterpret_cast<void*>(kWindowBase), // NOLINT(performance-no-int-to-ptr)
                bytes};
            ops::moe_expert_cache_window(expert_view, host_view, kLayer, window, placed_view,
                                         stream);
            cuda_synchronize(stream);
            failures +=
                verify_exact(label.c_str(), from_device<std::uint64_t>(placed.data(), kExperts),
                             oracle(t.expert, t.host, kLayer, bytes == 0 ? 0 : kWindowBase, bytes));
            failures +=
                verify_exact((label + " expert_ptr").c_str(),
                             from_device<std::uint64_t>(expert.data(), t.expert.size()), t.expert);
            failures +=
                verify_exact((label + " host_ptr").c_str(),
                             from_device<std::uint64_t>(host.data(), t.host.size()), t.host);
            failures += placed.verify_guards(label + " window_ptr");
            failures += expert.verify_guards(label + " expert_ptr");
            failures += host.verify_guards(label + " host_ptr");
        }
    }
    return failures;
}

// Real records: kStreamed experts of kLayer, alternating codecs, in pinned mapped host memory;
// every other entry is resident (a decoy slot address that neither Op may read). The window holds
// all but the last streamed record, so the last one stays mapped.
class Streamed {
public:
    static constexpr int kStreamed = 5;

    explicit Streamed(std::uint32_t seed) {
        std::mt19937 rng(seed);
        std::uniform_int_distribution<std::int32_t> pick(0, kExperts - 1);
        while (static_cast<int>(experts_.size()) < kStreamed) {
            const std::int32_t e = pick(rng);
            if (std::find(experts_.begin(), experts_.end(), e) == experts_.end()) {
                experts_.push_back(e);
            }
        }
        std::sort(experts_.begin(), experts_.end());
        std::size_t host_bytes = 0;
        for (int i = 0; i < kStreamed; ++i) {
            codecs_.push_back(i % 2 == 0 ? ops::MoeExpertCodec::F3G32
                                         : ops::MoeExpertCodec::Nvfp4G16);
            offsets_.push_back(host_bytes);
            host_bytes += static_cast<std::size_t>(ops::moe_expert_record_bytes(codecs_.back()));
            if (i + 1 < kStreamed) { window_bytes_ = host_bytes; }
        }
        records_    = std::make_unique<PinnedHostBuffer>(host_bytes, 256);
        auto* bytes = static_cast<std::uint8_t*>(records_->data());
        for (std::size_t i = 0; i < host_bytes; ++i) {
            bytes[i] = static_cast<std::uint8_t>(rng());
        }
        // The window has one record's spare bytes after the placed records, all canary.
        window_ = DeviceBuffer(window_bytes_ + ops::kMoeNvfp4RecordBytes);
        host_.assign(static_cast<std::size_t>(kExperts) * kLayers, 0);
        expert_.assign(host_.size(), 0);
        for (std::size_t entry = 0; entry < host_.size(); ++entry) {
            host_[entry]   = (kHostBase + entry * 4096) | 1U;
            expert_[entry] = (kSlotBase + entry * 4096) | 1U;
        }
        set_streamed(kStreamed);
        host_device_ = DeviceBuffer(host_.size() * sizeof(std::uint64_t));
        host_device_.copy_from_host(host_.data(), host_device_.bytes);
        placed_ = DeviceBuffer(static_cast<std::size_t>(kExperts) * sizeof(std::uint64_t));
    }

    // Streams the first `count` experts of the list (the others become resident decoys).
    void set_streamed(int count) {
        auto* bytes = static_cast<std::uint8_t*>(records_->data());
        for (int i = 0; i < kStreamed; ++i) {
            const auto entry = static_cast<std::size_t>(kLayer) * kExperts +
                               static_cast<std::size_t>(experts_[static_cast<std::size_t>(i)]);
            host_[entry]     = reinterpret_cast<std::uint64_t>(bytes + offsets_[i]) |
                               static_cast<std::uint64_t>(codecs_[i]);
            expert_[entry] =
                i < count ? host_[entry]
                          : (kSlotBase + entry * 4096) | static_cast<std::uint64_t>(codecs_[i]);
        }
        streamed_ = count;
        if (expert_device_.p == nullptr) {
            expert_device_ = DeviceBuffer(expert_.size() * sizeof(std::uint64_t));
        }
        expert_device_.copy_from_host(expert_.data(), expert_device_.bytes);
    }

    void enqueue(cudaStream_t stream) {
        const Tensor expert(expert_device_.p, DType::I64, {kExperts, kLayers});
        const Tensor host(host_device_.p, DType::I64, {kExperts, kLayers});
        Tensor placed(placed_.p, DType::I64, {kExperts});
        ops::moe_expert_cache_window(expert, host, kLayer, {window_.p, window_bytes_}, placed,
                                     stream);
        const Tensor host_column = host.slice(1, kLayer, 1);
        const Tensor placed_column(placed_.p, DType::I64, {kExperts, 1});
        ops::moe_expert_cache_fill(host_column, placed_column, 0, stream);
    }

    void reset_window() { window_.fill(kCanary); }

    // The window holds the placed records back to back, then canary bytes.
    [[nodiscard]] int check(const std::string& label) const {
        std::vector<std::uint8_t> got(window_.bytes);
        window_.copy_to_host(got.data(), got.size());
        std::vector<std::uint8_t> expected(window_.bytes, kCanary);
        const auto* bytes = static_cast<const std::uint8_t*>(records_->data());
        std::size_t at    = 0;
        for (int i = 0; i < streamed_; ++i) {
            const auto size = static_cast<std::size_t>(ops::moe_expert_record_bytes(codecs_[i]));
            if (at + size > window_bytes_) { break; }
            std::copy_n(bytes + offsets_[i], size,
                        expected.begin() + static_cast<std::ptrdiff_t>(at));
            at += size;
        }
        return verify_exact(label.c_str(), got, expected);
    }

private:
    std::vector<std::int32_t> experts_;
    std::vector<ops::MoeExpertCodec> codecs_;
    std::vector<std::size_t> offsets_;
    std::size_t window_bytes_ = 0;
    int streamed_             = 0;
    std::unique_ptr<PinnedHostBuffer> records_;
    DeviceBuffer window_, host_device_, expert_device_, placed_;
    std::vector<std::uint64_t> host_, expert_;
};

int streamed_cases(cudaStream_t stream, bool graph) {
    Streamed streamed(91);
    streamed.reset_window();
    streamed.enqueue(stream);
    cuda_synchronize(stream);
    int failures = streamed.check("window fill, last record mapped");
    if (!graph) { return failures; }
    DecodeGraphDefinition definition;
    DecodeGraphExecutable executable;
    definition.capture(stream, [&] { streamed.enqueue(stream); });
    executable.instantiate(definition);
    for (const int count : {Streamed::kStreamed, 2, 0}) {
        streamed.set_streamed(count);
        streamed.reset_window();
        executable.launch(stream);
        cuda_synchronize(stream);
        failures += streamed.check("window graph replay, " + std::to_string(count) + " streamed");
    }
    return failures;
}

} // namespace

int main(int argc, char** argv) {
    if (require_cuda() != 0) { return 1; }
    try {
        DeviceContext context;
        const bool sanitizer = sanitizer_scope(argc, argv);
        int failures         = sanitizer ? 0 : placement_cases(context.stream);
        failures += streamed_cases(context.stream, !sanitizer);
        std::cout << (failures == 0 ? "OK" : "FAIL") << " moe_expert_cache_window\n";
        return failures == 0 ? 0 : 1;
    } catch (const std::exception& error) {
        std::cerr << "moe_expert_cache_window: " << error.what() << '\n';
        return 1;
    }
}
