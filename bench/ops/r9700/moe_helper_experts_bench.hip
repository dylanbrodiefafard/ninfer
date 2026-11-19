// moe_helper_experts throughput on the R9700: per call, GB/s of distinct record bytes against a
// plain coalesced read of the same records (the measured roof for that byte set), and the
// TFLOP/s of the routed SwiGLU work (2 x (2 x 640 x 2560 + 2560 x 640) FLOP per active entry).
//
// One layer bank owns 512 distinct records (every id has its own bytes, so no two experts share
// cache lines). Each timed call uses one of kRouteSets route sets drawn with different expert
// popularity, rotating between calls, so neither the L2 nor the 64 MiB last-level cache carries a
// call's records into the next. Every call has max_tokens = T and N = T. Routing modes:
//   all           moe_route-like: each token's ten experts are the top ten of biased random
//                 logits, all owned by the helper (a decode round of helper-only experts).
//   concentrated  every token routes the same ten experts (T tokens per expert).
//   helper        prefill split: the helper owns ids not divisible by 3 (341 experts); each route
//                 slot is a helper route with probability 0.2, conditioned on at least one per
//                 token (about 2.2 active entries per token), drawn without repetition in the
//                 token by log-normal expert popularity (sigma 0.5); other slots are inactive.
//   skewed        helper with popularity sigma 1.5: a few experts receive hundreds of tokens.
//
//   ninfer_moe_helper_experts_bench [--codec f3|nvfp4|mixed]
//       [--routing all|concentrated|helper|skewed] [--output fp32|bf16] [--tokens T ...]
#include "ninfer/ops/moe_helper_experts.h"

#include <hip/hip_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <iostream>
#include <numeric>
#include <random>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using namespace ninfer::ops;

constexpr int kRouteSets  = 8;
constexpr int kIterations = 40; // timed calls per route set up to T = 256, fewer above

void hip_check(hipError_t status, const char* what) {
    if (status != hipSuccess) {
        throw std::runtime_error(std::string(what) + ": " + hipGetErrorString(status));
    }
}

// Owning device allocation of `count` elements.
template <class T>
class DeviceArray {
public:
    explicit DeviceArray(std::size_t count) {
        void* pointer = nullptr;
        hip_check(hipMalloc(&pointer, count * sizeof(T)), "hipMalloc");
        data_ = static_cast<T*>(pointer);
    }

    ~DeviceArray() { (void)hipFree(data_); }

    DeviceArray(const DeviceArray&)            = delete;
    DeviceArray& operator=(const DeviceArray&) = delete;

    [[nodiscard]] T* get() const noexcept { return data_; }

private:
    T* data_ = nullptr;
};

class Event {
public:
    Event() { hip_check(hipEventCreate(&event_), "hipEventCreate"); }

    ~Event() { (void)hipEventDestroy(event_); }

    Event(const Event&)            = delete;
    Event& operator=(const Event&) = delete;

    [[nodiscard]] hipEvent_t get() const noexcept { return event_; }

private:
    hipEvent_t event_ = nullptr;
};

// Streams `count` 16-byte units of each listed record once (grid-stride, coalesced) and folds
// them into one word, so the loads cannot be elided.
__global__ void read_records_kernel(const std::uint64_t* addresses, const std::uint64_t* units,
                                    int records, std::uint32_t* sink) {
    std::uint32_t fold = 0;
    for (int r = 0; r < records; ++r) {
        const auto* record =
            reinterpret_cast<const uint4*>(addresses[r]); // NOLINT(performance-no-int-to-ptr)
        for (std::uint64_t i = blockIdx.x * blockDim.x + threadIdx.x; i < units[r];
             i += static_cast<std::uint64_t>(gridDim.x) * blockDim.x) {
            const uint4 v = record[i];
            fold ^= v.x ^ v.y ^ v.z ^ v.w;
        }
    }
    if (fold == 0x9E3779B9U) { sink[0] = fold; }
}

struct RouteSet {
    std::vector<MoeHelperRoute> routes;
    std::vector<std::uint64_t> addresses; // distinct records
    std::vector<std::uint64_t> units;
    double bytes   = 0.0;
    double entries = 0.0; // active route entries
};

int main_impl(int argc, char** argv) {
    std::string codec_mode            = "f3";
    std::string routing               = "all";
    MoeHelperOutputDtype output_dtype = MoeHelperOutputDtype::Fp32;
    std::vector<int> extents;
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--codec" && i + 1 < argc) {
            codec_mode = argv[++i];
        } else if (arg == "--routing" && i + 1 < argc) {
            routing = argv[++i];
        } else if (arg == "--output" && i + 1 < argc) {
            const std::string dtype = argv[++i];
            if (dtype != "fp32" && dtype != "bf16") {
                throw std::invalid_argument("output must be fp32 or bf16");
            }
            output_dtype =
                dtype == "bf16" ? MoeHelperOutputDtype::Bf16 : MoeHelperOutputDtype::Fp32;
        } else if (arg == "--tokens" && i + 1 < argc) {
            extents.push_back(std::stoi(argv[++i]));
        } else {
            std::cerr << "usage: " << argv[0]
                      << " [--codec f3|nvfp4|mixed] [--routing all|concentrated|helper|skewed]"
                         " [--output fp32|bf16] [--tokens T ...]\n";
            return 2;
        }
    }
    if (routing != "all" && routing != "concentrated" && routing != "helper" &&
        routing != "skewed") {
        std::cerr << "unknown routing " << routing << '\n';
        return 2;
    }
    if (extents.empty()) { extents = {1, 4, 20, 48}; }

    int count = 0;
    hip_check(hipGetDeviceCount(&count), "hipGetDeviceCount");
    int device = -1;
    for (int d = 0; d < count && device < 0; ++d) {
        hipDeviceProp_t properties{};
        hip_check(hipGetDeviceProperties(&properties, d), "hipGetDeviceProperties");
        if (std::strncmp(properties.gcnArchName, "gfx1201", 7) == 0) { device = d; }
    }
    if (device < 0) { throw std::runtime_error("no gfx1201 device"); }
    hip_check(hipSetDevice(device), "hipSetDevice");

    // 512 distinct records with random bytes (any byte string is a valid code layout; the timing
    // does not depend on the values).
    std::vector<MoeExpertCodec> codecs(kMoeHelperExperts);
    std::vector<std::size_t> offsets(kMoeHelperExperts);
    std::size_t store_bytes = 0;
    for (int e = 0; e < kMoeHelperExperts; ++e) {
        codecs[e]  = codec_mode == "f3"      ? MoeExpertCodec::F3G32
                     : codec_mode == "nvfp4" ? MoeExpertCodec::Nvfp4G16
                     : e % 8 == 0            ? MoeExpertCodec::Nvfp4G16
                                             : MoeExpertCodec::F3G32;
        offsets[e] = store_bytes;
        store_bytes += static_cast<std::size_t>(moe_expert_record_bytes(codecs[e]));
    }
    const DeviceArray<std::uint8_t> store_buffer(store_bytes);
    std::uint8_t* store = store_buffer.get();
    {
        std::vector<std::uint32_t> random(store_bytes / 4);
        std::mt19937 rng(7U);
        for (auto& word : random) { word = rng(); }
        hip_check(hipMemcpy(store, random.data(), store_bytes, hipMemcpyHostToDevice), "upload");
    }
    std::vector<std::uint64_t> table(kMoeHelperExperts);
    for (int e = 0; e < kMoeHelperExperts; ++e) {
        table[e] = reinterpret_cast<std::uintptr_t>(store + offsets[e]) |
                   static_cast<std::uint64_t>(codecs[e]);
    }
    std::vector<float> scales(kMoeHelperExperts * 3, 1.0e-3F);
    const DeviceArray<std::uint64_t> table_buffer(table.size());
    const DeviceArray<float> scales_buffer(scales.size());
    std::uint64_t* table_device = table_buffer.get();
    float* scales_device        = scales_buffer.get();
    hip_check(hipMemcpy(table_device, table.data(), table.size() * 8, hipMemcpyHostToDevice), "t");
    hip_check(hipMemcpy(scales_device, scales.data(), scales.size() * 4, hipMemcpyHostToDevice),
              "s");

    const int max_extent = *std::max_element(extents.begin(), extents.end());
    const auto rows      = static_cast<std::size_t>(max_extent);
    const DeviceArray<std::uint16_t> x_buffer(rows * kMoeHidden);
    const DeviceArray<float> y_buffer(rows * kMoeHidden);
    const DeviceArray<std::int32_t> n_buffer(1);
    std::uint16_t* x = x_buffer.get();
    float* y         = y_buffer.get();
    std::int32_t* n  = n_buffer.get();
    {
        std::vector<std::uint16_t> host(static_cast<std::size_t>(max_extent) * kMoeHidden, 0x3F00);
        hip_check(hipMemcpy(x, host.data(), host.size() * 2, hipMemcpyHostToDevice), "x");
    }
    const DeviceArray<MoeHelperRoute> routes_buffer(kRouteSets * rows * kMoeHelperRoutesPerToken);
    MoeHelperRoute* routes = routes_buffer.get();
    const DeviceArray<std::uint64_t> addresses_buffer(kRouteSets * kMoeHelperExperts);
    const DeviceArray<std::uint64_t> units_buffer(kRouteSets * kMoeHelperExperts);
    const DeviceArray<std::uint32_t> sink_buffer(1);
    std::uint64_t* read_addresses = addresses_buffer.get();
    std::uint64_t* read_units     = units_buffer.get();
    std::uint32_t* sink           = sink_buffer.get();
    MoeHelperExpertsWorkspace workspace(max_extent);
    const Event start_event;
    const Event stop_event;
    hipEvent_t start = start_event.get();
    hipEvent_t stop  = stop_event.get();
    hipDeviceProp_t properties{};
    hip_check(hipGetDeviceProperties(&properties, device), "properties");
    const int read_grid = properties.multiProcessorCount * 16;

    std::printf("codec=%s routing=%s output=%s\n", codec_mode.c_str(), routing.c_str(),
                output_dtype == MoeHelperOutputDtype::Bf16 ? "bf16" : "fp32");
    std::printf("%5s %8s %9s %10s %10s %9s %9s %10s %7s %9s\n", "T", "entries", "distinct",
                "MB/call", "op_us", "TFLOP/s", "op_GB/s", "read_us", "of_roof", "N0_us");
    for (const int tokens : extents) {
        std::vector<RouteSet> sets(kRouteSets);
        std::mt19937 rng(1000U + static_cast<std::uint32_t>(tokens));
        std::normal_distribution<double> normal(0.0, 1.0);
        for (RouteSet& set : sets) {
            std::set<int> distinct;
            if (routing == "helper" || routing == "skewed") {
                const double sigma = routing == "helper" ? 0.5 : 1.5;
                std::vector<int> owned;
                std::vector<double> popularity;
                for (int e = 0; e < kMoeHelperExperts; ++e) {
                    if (e % 3 == 0) { continue; }
                    owned.push_back(e);
                    popularity.push_back(std::exp(sigma * normal(rng)));
                }
                std::discrete_distribution<int> pick(popularity.begin(), popularity.end());
                std::bernoulli_distribution helper_slot(0.2);
                for (int t = 0; t < tokens; ++t) {
                    std::vector<bool> slots(kMoeHelperRoutesPerToken);
                    do {
                        for (int j = 0; j < kMoeHelperRoutesPerToken; ++j) {
                            slots[static_cast<std::size_t>(j)] = helper_slot(rng);
                        }
                    } while (std::find(slots.begin(), slots.end(), true) == slots.end());
                    std::set<int> chosen;
                    for (int j = 0; j < kMoeHelperRoutesPerToken; ++j) {
                        if (!slots[static_cast<std::size_t>(j)]) {
                            set.routes.push_back({kMoeHelperInactiveExpert, 0.0F});
                            continue;
                        }
                        int expert = 0;
                        do {
                            expert = owned[static_cast<std::size_t>(pick(rng))];
                        } while (!chosen.insert(expert).second);
                        set.routes.push_back({expert, 0.1F});
                        distinct.insert(expert);
                        ++set.entries;
                    }
                }
            } else {
                const bool concentrated = routing == "concentrated";
                std::vector<double> bias(kMoeHelperExperts);
                for (auto& b : bias) { b = 0.45 * normal(rng); }
                for (int t = 0; t < tokens; ++t) {
                    std::vector<int> order(kMoeHelperExperts);
                    std::iota(order.begin(), order.end(), 0);
                    if (!concentrated || t == 0) {
                        std::vector<double> score(kMoeHelperExperts);
                        for (std::size_t e = 0; e < score.size(); ++e) {
                            score[e] = bias[e] + normal(rng);
                        }
                        std::partial_sort(order.begin(), order.begin() + 10, order.end(),
                                          [&](int a, int b) { return score[a] > score[b]; });
                    } else {
                        for (int j = 0; j < 10; ++j) {
                            order[j] = set.routes[static_cast<std::size_t>(j)].expert;
                        }
                    }
                    for (int j = 0; j < kMoeHelperRoutesPerToken; ++j) {
                        set.routes.push_back({order[j], 0.1F});
                        distinct.insert(order[j]);
                        ++set.entries;
                    }
                }
            }
            for (const int e : distinct) {
                const auto bytes = static_cast<std::uint64_t>(moe_expert_record_bytes(codecs[e]));
                set.addresses.push_back(reinterpret_cast<std::uintptr_t>(store + offsets[e]));
                set.units.push_back(bytes / 16);
                set.bytes += static_cast<double>(bytes);
            }
        }
        for (int s = 0; s < kRouteSets; ++s) {
            const auto& set = sets[static_cast<std::size_t>(s)];
            hip_check(hipMemcpy(routes + static_cast<std::size_t>(s) * max_extent *
                                             kMoeHelperRoutesPerToken,
                                set.routes.data(), set.routes.size() * sizeof(MoeHelperRoute),
                                hipMemcpyHostToDevice),
                      "routes");
            hip_check(hipMemcpy(read_addresses + s * kMoeHelperExperts, set.addresses.data(),
                                set.addresses.size() * 8, hipMemcpyHostToDevice),
                      "addresses");
            hip_check(hipMemcpy(read_units + s * kMoeHelperExperts, set.units.data(),
                                set.units.size() * 8, hipMemcpyHostToDevice),
                      "units");
        }
        hip_check(hipMemcpy(n, &tokens, 4, hipMemcpyHostToDevice), "n");
        double bytes    = 0.0;
        double distinct = 0.0;
        double entries  = 0.0;
        for (const auto& set : sets) {
            bytes += set.bytes;
            distinct += static_cast<double>(set.addresses.size());
            entries += set.entries;
        }
        bytes /= kRouteSets;
        distinct /= kRouteSets;
        entries /= kRouteSets;
        constexpr double kFlopPerEntry =
            2.0 * (2.0 * kMoeIntermediate * kMoeHidden + 1.0 * kMoeHidden * kMoeIntermediate);

        const auto op_call = [&](int s) {
            MoeHelperExpertsArgs args;
            args.tokens     = n;
            args.max_tokens = tokens;
            args.x          = x;
            args.routes =
                routes + static_cast<std::size_t>(s) * max_extent * kMoeHelperRoutesPerToken;
            args.bank.expert_table  = table_device;
            args.bank.matrix_scales = scales_device;
            args.bank.f3_books      = {0x76419CEFU, 0x76419CEFU, 0x76419CEFU};
            args.y                  = y;
            args.output_dtype       = output_dtype;
            moe_helper_experts(args, workspace, nullptr);
        };
        const auto read_call = [&](int s) {
            read_records_kernel<<<read_grid, 256, 0, nullptr>>>(
                read_addresses + s * kMoeHelperExperts, read_units + s * kMoeHelperExperts,
                static_cast<int>(sets[static_cast<std::size_t>(s)].addresses.size()), sink);
            hip_check(hipGetLastError(), "read kernel");
        };
        const int iterations = std::max(5, kIterations * 256 / std::max(tokens, 256));
        const auto time_us   = [&](const auto& call) {
            for (int s = 0; s < kRouteSets; ++s) { call(s); }
            hip_check(hipDeviceSynchronize(), "warmup");
            hip_check(hipEventRecord(start, nullptr), "record");
            for (int i = 0; i < iterations * kRouteSets; ++i) { call(i % kRouteSets); }
            hip_check(hipEventRecord(stop, nullptr), "record");
            hip_check(hipEventSynchronize(stop), "sync");
            float ms = 0.0F;
            hip_check(hipEventElapsedTime(&ms, start, stop), "elapsed");
            return 1000.0 * ms / (iterations * kRouteSets);
        };
        const double op_us   = time_us(op_call);
        const double read_us = time_us(read_call);
        // The same launch with N = 0: launch, prologue and exit without items.
        const std::int32_t zero = 0;
        hip_check(hipMemcpy(n, &zero, 4, hipMemcpyHostToDevice), "n");
        const double empty_us = time_us(op_call);
        std::printf("%5d %8.0f %9.1f %10.2f %10.1f %9.2f %9.1f %10.1f %6.0f%% %9.1f\n", tokens,
                    entries, distinct, bytes / 1e6, op_us, entries * kFlopPerEntry / op_us / 1e6,
                    bytes / op_us / 1e3, read_us, 100.0 * read_us / op_us, empty_us);
    }
    return 0;
}

} // namespace

int main(int argc, char** argv) {
    try {
        return main_impl(argc, argv);
    } catch (const std::exception& error) {
        std::cerr << "moe_helper_experts_bench: " << error.what() << '\n';
        return 1;
    }
}
