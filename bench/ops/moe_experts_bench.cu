// Public benchmark of one Qwen3.8-Flash-Next routed-expert layer (ninfer::ops::moe_experts).
//
// Every routed expert has its own record (32 of the 512 in NVFP4G16, the rest in F3G32, close to
// the chosen mix); the shared expert is an NVFP4G16 record in VRAM. One measurement is one CUDA
// Graph replay of the Op with the L2 flushed beforehand. Bytes count each distinct routed record
// once: host records as PCIe bytes, VRAM records plus the shared record as DRAM bytes.
//
// Two suites:
//   placement  ten distinct uniformly random experts per token, or skewed (per-expert bias, so
//              fewer distinct experts and some with more tokens than one tile), with every record
//              in VRAM, in device-mapped pinned host memory, or mixed (odd experts on the host);
//              no deferral. Each point runs the activation profiles --activations selects (A16,
//              A8, or both, the default).
//   decode     the decode and verification band as the Program runs it: L lanes of D consecutive
//              tokens with recorded-trace expert reuse (see make_lane_routes) and records in VRAM.
//              Each point runs five cache states: no deferral; a deferral with nothing deferred (a
//              layer whose misses admitted nothing); m deferred experts read from VRAM slots
//              after a ready event with no copy (the Op's own cost of admitted misses); the same m
//              experts copied into their slots by moe_expert_cache_fill on the forked stream
//              before ready (an admitted-miss layer as the Program runs it); and the same m
//              experts read through mapped host memory with nothing deferred (misses the admission
//              filter rejected). m is a tenth of the distinct routed experts, at least one, close
//              to the miss rate at four lanes.
#include "ninfer/ops/moe_expert_cache.h"
#include "ninfer/ops/moe_experts.h"

#include "core/device.h"
#include "ninfer_bench_common.h"

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <numeric>
#include <random>
#include <set>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

using namespace ninfer;
using namespace ninfer::bench;

namespace {

constexpr std::size_t kL2FlushBytes       = 256ULL << 20;
constexpr double kRtx5090SustainedReadGBs = 1674.5;
// Kernel reads of mapped pinned memory on the PCIe Gen5 link (the fill microbenchmark's roof).
constexpr double kPcieMappedReadGBs  = 48.0;
constexpr double kSharedExpertBytes  = static_cast<double>(ops::kMoeNvfp4RecordBytes);
constexpr double kFlopsPerAssignment = 2.0 * 3.0 * ops::kMoeHidden * ops::kMoeIntermediate;

ops::MoeExpertCodec codec_of(int expert) {
    return expert % 16 == 0 ? ops::MoeExpertCodec::Nvfp4G16 : ops::MoeExpertCodec::F3G32;
}

class PinnedHost {
public:
    explicit PinnedHost(std::size_t bytes) {
        CUDA_CHECK(cudaHostAlloc(&host_, bytes, cudaHostAllocMapped));
        CUDA_CHECK(cudaHostGetDevicePointer(&device_, host_, 0));
    }

    ~PinnedHost() { cudaFreeHost(host_); }

    PinnedHost(const PinnedHost&)            = delete;
    PinnedHost& operator=(const PinnedHost&) = delete;

    [[nodiscard]] void* host() const noexcept { return host_; }

    [[nodiscard]] void* device() const noexcept { return device_; }

private:
    void* host_   = nullptr;
    void* device_ = nullptr;
};

class Stream {
public:
    Stream() { CUDA_CHECK(cudaStreamCreateWithFlags(&stream_, cudaStreamNonBlocking)); }

    ~Stream() {
        cudaStreamSynchronize(stream_);
        cudaStreamDestroy(stream_);
    }

    Stream(const Stream&)            = delete;
    Stream& operator=(const Stream&) = delete;

    [[nodiscard]] cudaStream_t get() const noexcept { return stream_; }

private:
    cudaStream_t stream_ = nullptr;
};

class Event {
public:
    Event() { CUDA_CHECK(cudaEventCreateWithFlags(&event_, cudaEventDisableTiming)); }

    ~Event() { cudaEventDestroy(event_); }

    Event(const Event&)            = delete;
    Event& operator=(const Event&) = delete;

    [[nodiscard]] cudaEvent_t get() const noexcept { return event_; }

private:
    cudaEvent_t event_ = nullptr;
};

struct Records {
    std::vector<std::size_t> offsets;
    std::size_t bytes = 0;
};

Records record_layout() {
    Records records;
    for (int expert = 0; expert < ops::kMoeExperts; ++expert) {
        records.offsets.push_back(records.bytes);
        records.bytes += static_cast<std::size_t>(ops::moe_expert_record_bytes(codec_of(expert)));
    }
    return records;
}

// The shared inputs of every point: the record stores and the layer's scales.
struct Stores {
    Records records;
    void* device                      = nullptr;
    void* mapped                      = nullptr;
    const DeviceBuffer* scales        = nullptr;
    const DeviceBuffer* shared_record = nullptr;
    const DeviceBuffer* shared_scales = nullptr;
    DeviceBuffer* flush               = nullptr;
    int warmup                        = 3;
    int repeat                        = 20;
};

enum class Placement : std::uint8_t { Vram, Host, Mixed };
enum class Routing : std::uint8_t { Uniform, Skewed };

struct PlacementConfig {
    Placement placement;
    Routing routing;
};

bool host_resident(Placement placement, int expert) {
    return placement == Placement::Host || (placement == Placement::Mixed && expert % 2 == 1);
}

std::vector<std::int32_t> make_routes(int tokens, Routing routing, std::uint32_t seed) {
    std::mt19937 rng(seed);
    std::normal_distribution<double> normal(0.0, 1.0);
    std::vector<double> bias(ops::kMoeExperts, 0.0);
    if (routing == Routing::Skewed) {
        for (auto& b : bias) { b = 2.0 * normal(rng); }
    }
    std::vector<std::int32_t> ids;
    std::vector<int> order(ops::kMoeExperts);
    std::vector<double> logits(ops::kMoeExperts);
    for (int t = 0; t < tokens; ++t) {
        for (std::size_t e = 0; e < logits.size(); ++e) { logits[e] = bias[e] + 1.5 * normal(rng); }
        std::iota(order.begin(), order.end(), 0);
        std::partial_sort(
            order.begin(), order.begin() + ops::kMoeTopK, order.end(), [&](int a, int b) {
                return logits[static_cast<std::size_t>(a)] > logits[static_cast<std::size_t>(b)];
            });
        ids.insert(ids.end(), order.begin(), order.begin() + ops::kMoeTopK);
    }
    return ids;
}

// Routes of `lanes` lanes of `lane_tokens` consecutive tokens, lane-major. A lane's first token
// routes to ten distinct uniformly random experts; each later token keeps every expert of the
// previous token with probability 0.4 and fills the other slots with uniformly random experts it
// does not hold yet. On the recorded Flash-Next decode routes (four lanes, 2048 rounds, 48 layers),
// windows of 1 / 2 / 4 / 5 lanes x 4 consecutive tokens route to 27.3 / 54.5 / 102.5 / 125.5
// distinct experts per layer; this model gives 27.5 / 53.7 / 101.8 / 123.7.
std::vector<std::int32_t> make_lane_routes(int lanes, int lane_tokens, std::uint32_t seed) {
    constexpr double kKeep = 0.4;
    std::mt19937 rng(seed);
    std::uniform_int_distribution<int> expert_draw(0, ops::kMoeExperts - 1);
    std::bernoulli_distribution keep(kKeep);
    std::vector<std::int32_t> ids;
    for (int lane = 0; lane < lanes; ++lane) {
        std::vector<std::int32_t> previous;
        for (int t = 0; t < lane_tokens; ++t) {
            std::vector<std::int32_t> token;
            for (const std::int32_t expert : previous) {
                if (keep(rng)) { token.push_back(expert); }
            }
            while (token.size() < static_cast<std::size_t>(ops::kMoeTopK)) {
                const std::int32_t expert = expert_draw(rng);
                if (std::find(token.begin(), token.end(), expert) == token.end()) {
                    token.push_back(expert);
                }
            }
            ids.insert(ids.end(), token.begin(), token.end());
            previous = std::move(token);
        }
    }
    return ids;
}

// One layer call: routes, where each routed expert's record is read from, and the deferral.
struct Point {
    ops::MoeActivationProfile profile = ops::MoeActivationProfile::A16;
    std::vector<std::int32_t> ids;
    std::vector<bool> host;        // [512]: record read through mapped host memory
    std::vector<bool> deferred;    // [512]: record read from its VRAM slot after `ready`
    bool with_deferral    = false; // pass a deferral (all-zero slots when nothing is deferred)
    bool fill             = false; // the deferred slots are filled from host before `ready`
    std::string placement = "vram";
    std::string routing;
};

struct Timing {
    ColdTiming cold;
    std::size_t experts     = 0;
    std::size_t max_tokens  = 0; // tokens of the most popular expert
    std::size_t deferred    = 0;
    std::size_t host        = 0;
    double dram_bytes       = 0.0;
    double pcie_bytes       = 0.0;
    double tflop_per_second = 0.0;
};

Timing measure(const Point& point, const Stores& stores) {
    const auto tokens = static_cast<int>(point.ids.size() / ops::kMoeTopK);
    std::vector<float> weights(point.ids.size(), 0.1F);
    std::vector<float> shared_scale(static_cast<std::size_t>(tokens), 0.5F);
    std::vector<std::int64_t> table(ops::kMoeExperts);
    // A deferred expert's table entry names its host record, as a miss's does. Its slot is its VRAM
    // record, or with a fill a slot of its own that the fill writes.
    std::size_t slot_bytes = 0;
    for (int expert = 0; expert < ops::kMoeExperts; ++expert) {
        if (point.fill && point.deferred[static_cast<std::size_t>(expert)]) {
            slot_bytes += static_cast<std::size_t>(ops::moe_expert_record_bytes(codec_of(expert)));
        }
    }
    DeviceBuffer slot_store(std::max<std::size_t>(slot_bytes, 1));
    std::vector<std::int64_t> slots(ops::kMoeExperts, 0);
    std::vector<std::int64_t> host_table(ops::kMoeExperts);
    std::size_t slot_offset = 0;
    for (int expert = 0; expert < ops::kMoeExperts; ++expert) {
        const auto index      = static_cast<std::size_t>(expert);
        const std::size_t off = stores.records.offsets[index];
        const auto tag        = static_cast<std::uintptr_t>(codec_of(expert));
        const auto device     = reinterpret_cast<std::uintptr_t>(stores.device) + off;
        const auto mapped     = reinterpret_cast<std::uintptr_t>(stores.mapped) + off;
        const bool host       = point.host[index] || point.deferred[index];
        table[index]          = static_cast<std::int64_t>((host ? mapped : device) | tag);
        host_table[index]     = static_cast<std::int64_t>(mapped | tag);
        if (point.deferred[index]) {
            std::uintptr_t slot = device;
            if (point.fill) {
                slot = reinterpret_cast<std::uintptr_t>(slot_store.p) + slot_offset;
                slot_offset +=
                    static_cast<std::size_t>(ops::moe_expert_record_bytes(codec_of(expert)));
            }
            slots[index] = static_cast<std::int64_t>(slot);
        }
    }

    Timing result;
    const std::set<int> routed(point.ids.begin(), point.ids.end());
    result.experts    = routed.size();
    result.dram_bytes = kSharedExpertBytes;
    for (const int expert : routed) {
        const auto index  = static_cast<std::size_t>(expert);
        const auto record = static_cast<double>(ops::moe_expert_record_bytes(codec_of(expert)));
        // A filled record crosses PCIe once (the fill) and is then read from its VRAM slot.
        if (point.fill && point.deferred[index]) { result.pcie_bytes += record; }
        (point.host[index] ? result.pcie_bytes : result.dram_bytes) += record;
        result.host += point.host[index] ? 1 : 0;
        result.deferred += point.deferred[index] ? 1 : 0;
        result.max_tokens = std::max<std::size_t>(
            result.max_tokens,
            static_cast<std::size_t>(std::count(point.ids.begin(), point.ids.end(), expert)));
    }

    DeviceBuffer x = make_bf16(static_cast<std::size_t>(ops::kMoeHidden) * tokens);
    DeviceBuffer y(static_cast<std::size_t>(ops::kMoeHidden) * tokens * sizeof(std::uint16_t));
    DeviceBuffer ids_d(point.ids.size() * sizeof(std::int32_t));
    DeviceBuffer weights_d(weights.size() * sizeof(float));
    DeviceBuffer shared_d(shared_scale.size() * sizeof(float));
    DeviceBuffer table_d(table.size() * sizeof(std::int64_t));
    DeviceBuffer slots_d(slots.size() * sizeof(std::int64_t));
    DeviceBuffer host_table_d(host_table.size() * sizeof(std::int64_t));
    host_table_d.copy_from_host(host_table.data(), host_table_d.bytes);
    ids_d.copy_from_host(point.ids.data(), ids_d.bytes);
    weights_d.copy_from_host(weights.data(), weights_d.bytes);
    shared_d.copy_from_host(shared_scale.data(), shared_d.bytes);
    table_d.copy_from_host(table.data(), table_d.bytes);
    slots_d.copy_from_host(slots.data(), slots_d.bytes);
    WorkspaceArena workspace(ops::moe_experts_workspace_capacity_bytes(tokens, point.profile));

    ops::MoeExpertBank bank;
    bank.expert_table         = Tensor(table_d.p, DType::I64, {ops::kMoeExperts});
    bank.matrix_scales        = Tensor(stores.scales->p, DType::FP32, {3, ops::kMoeExperts});
    bank.f3_books             = {0xF7E6C491U, 0x91C4E6F7U, 0xE6F791C4U};
    bank.shared_record        = Tensor(stores.shared_record->p, DType::U8,
                                       {static_cast<std::int32_t>(ops::kMoeNvfp4RecordBytes)});
    bank.shared_matrix_scales = Tensor(stores.shared_scales->p, DType::FP32, {3});
    Tensor x_t(x.p, DType::BF16, {ops::kMoeHidden, tokens});
    Tensor ids_t(ids_d.p, DType::I32, {ops::kMoeTopK, tokens});
    Tensor weights_t(weights_d.p, DType::FP32, {ops::kMoeTopK, tokens});
    Tensor shared_t(shared_d.p, DType::FP32, {tokens});
    Tensor y_t(y.p, DType::BF16, {ops::kMoeHidden, tokens});

    const Stream stream;
    const Stream side;
    const Event fork;
    const Event ready;
    const ops::MoeExpertsDeferral deferral{Tensor(slots_d.p, DType::I64, {ops::kMoeExperts}),
                                           ready.get()};
    const Tensor host_t(host_table_d.p, DType::I64, {ops::kMoeExperts, 1});
    const Tensor fill_t(slots_d.p, DType::I64, {ops::kMoeExperts, 1});
    // With a deferral, `ready` is recorded on a stream forked from the Op's (the Program's fill
    // stream), after the fill when there is one.
    const auto call = [&](cudaStream_t s) {
        if (point.with_deferral) {
            CUDA_CHECK(cudaEventRecord(fork.get(), s));
            CUDA_CHECK(cudaStreamWaitEvent(side.get(), fork.get(), 0));
            if (point.fill) { ops::moe_expert_cache_fill(host_t, fill_t, 0, side.get()); }
            CUDA_CHECK(cudaEventRecord(ready.get(), side.get()));
        }
        ops::moe_experts(x_t, ids_t, weights_t, shared_t, bank, point.profile, y_t, workspace, s,
                         point.with_deferral ? &deferral : nullptr);
    };
    call(stream.get()); // configures kernel attributes outside capture
    CUDA_CHECK(cudaStreamSynchronize(stream.get()));
    {
        TimedGraph graph;
        graph.capture(stream.get(), call);
        result.cold =
            measure_cold_graph(graph, *stores.flush, stream.get(), stores.warmup, stores.repeat);
    }
    const double seconds    = result.cold.median_us * 1.0e-6;
    result.tflop_per_second = kFlopsPerAssignment * (ops::kMoeTopK + 1) * tokens / seconds / 1.0e12;
    return result;
}

void print(const Point& point, const std::string& state, const Timing& t) {
    const double seconds = t.cold.median_us * 1.0e-6;
    const auto tokens    = point.ids.size() / ops::kMoeTopK;
    std::printf(
        "%-3s %-5s %-9s %-10s T=%-5zu experts=%-3zu max/expert=%-4zu deferred=%-3zu "
        "host=%-3zu median=%9.1f us min=%9.1f us  DRAM %7.1f MB %7.1f GB/s (%5.1f%%)  PCIe "
        "%7.1f MB floor %7.1f us  %6.1f TFLOP/s\n",
        point.profile == ops::MoeActivationProfile::A8 ? "a8" : "a16", point.placement.c_str(),
        point.routing.c_str(), state.c_str(), tokens, t.experts, t.max_tokens, t.deferred, t.host,
        t.cold.median_us, t.cold.min_us, t.dram_bytes / 1.0e6, t.dram_bytes / seconds / 1.0e9,
        100.0 * t.dram_bytes / seconds / 1.0e9 / kRtx5090SustainedReadGBs, t.pcie_bytes / 1.0e6,
        t.pcie_bytes / (kPcieMappedReadGBs * 1.0e3), t.tflop_per_second);
}

void run_placement_point(int tokens, const PlacementConfig& config,
                         ops::MoeActivationProfile profile, const Stores& stores) {
    Point point;
    point.profile  = profile;
    point.ids      = make_routes(tokens, config.routing, 7U + static_cast<std::uint32_t>(tokens));
    point.host     = std::vector<bool>(ops::kMoeExperts, false);
    point.deferred = point.host;
    for (int expert = 0; expert < ops::kMoeExperts; ++expert) {
        point.host[static_cast<std::size_t>(expert)] = host_resident(config.placement, expert);
    }
    point.placement = config.placement == Placement::Vram   ? "vram"
                      : config.placement == Placement::Host ? "host"
                                                            : "mixed";
    point.routing   = config.routing == Routing::Skewed ? "skewed" : "uniform";
    print(point, "-", measure(point, stores));
}

// Lanes x consecutive tokens per lane of the decode suite's token extents: one lane without drafts
// (T = 1), then lanes verifying three draft tokens each.
std::pair<int, int> decode_shape(int tokens) {
    if (tokens >= 4 && tokens % 4 == 0 && tokens / 4 <= 12) { return {tokens / 4, 4}; }
    if (tokens >= 1 && tokens <= 12) { return {tokens, 1}; }
    throw std::invalid_argument("decode T is 1..12, or a multiple of 4 up to 48");
}

void run_decode_point(int tokens, const Stores& stores) {
    const auto [lanes, lane_tokens] = decode_shape(tokens);
    Point point;
    point.ids      = make_lane_routes(lanes, lane_tokens, 11U + static_cast<std::uint32_t>(tokens));
    point.routing  = std::to_string(lanes) + "x" + std::to_string(lane_tokens);
    point.host     = std::vector<bool>(ops::kMoeExperts, false);
    point.deferred = point.host;

    // The missed experts: a tenth of the distinct routed experts in a seeded random order.
    std::vector<int> routed;
    for (const int expert : std::set<int>(point.ids.begin(), point.ids.end())) {
        routed.push_back(expert);
    }
    std::mt19937 rng(13U + static_cast<std::uint32_t>(tokens));
    std::shuffle(routed.begin(), routed.end(), rng);
    const std::size_t misses = std::max<std::size_t>(1, routed.size() / 10);

    print(point, "none", measure(point, stores));
    point.with_deferral = true;
    print(point, "empty", measure(point, stores));
    for (std::size_t i = 0; i < misses; ++i) {
        point.deferred[static_cast<std::size_t>(routed[i])] = true;
    }
    print(point, "deferred", measure(point, stores));
    point.fill = true;
    print(point, "filled", measure(point, stores));
    point.fill     = false;
    point.host     = point.deferred;
    point.deferred = std::vector<bool>(ops::kMoeExperts, false);
    print(point, "mapped", measure(point, stores));
}

} // namespace

int main(int argc, char** argv) {
    try {
        enum class Suite : std::uint8_t { All, Placement, Decode };
        Suite suite                                     = Suite::All;
        std::vector<int> placement_tokens               = {1, 12, 512, 4096};
        std::vector<int> decode_tokens                  = {1, 4, 8, 16, 20, 48};
        std::vector<PlacementConfig> configs            = {{Placement::Vram, Routing::Uniform},
                                                           {Placement::Host, Routing::Uniform},
                                                           {Placement::Mixed, Routing::Skewed}};
        std::vector<ops::MoeActivationProfile> profiles = {ops::MoeActivationProfile::A16,
                                                           ops::MoeActivationProfile::A8};
        int warmup                                      = 3;
        int repeat                                      = 20;
        for (int i = 1; i < argc; ++i) {
            const std::string_view arg(argv[i]);
            if (arg == "--tokens" && i + 1 < argc) {
                const int tokens = parse_number<int>(argv[++i], "--tokens");
                placement_tokens = {tokens};
                decode_tokens    = {tokens};
            } else if (arg == "--repeat" && i + 1 < argc) {
                repeat = parse_number<int>(argv[++i], "--repeat");
            } else if (arg == "--suite" && i + 1 < argc) {
                const std::string_view name(argv[++i]);
                if (name == "placement") {
                    suite = Suite::Placement;
                } else if (name == "decode") {
                    suite = Suite::Decode;
                } else {
                    throw std::invalid_argument("--suite is placement or decode");
                }
            } else if (arg == "--activations" && i + 1 < argc) {
                const std::string_view name(argv[++i]);
                if (name == "a16") {
                    profiles = {ops::MoeActivationProfile::A16};
                } else if (name == "a8") {
                    profiles = {ops::MoeActivationProfile::A8};
                } else if (name != "both") {
                    throw std::invalid_argument("--activations is a16, a8 or both");
                }
            } else if (arg == "--config" && i + 1 < argc) {
                const std::string_view name(argv[++i]);
                const auto index = parse_number<std::size_t>(name, "--config");
                if (index >= configs.size()) { throw std::invalid_argument("--config is 0..2"); }
                configs = {configs[index]};
            } else {
                throw std::invalid_argument(
                    "usage: ninfer_moe_experts_bench [--suite placement|decode] [--tokens T] "
                    "[--config 0..2] [--activations a16|a8|both] [--repeat N] (placement configs: "
                    "0 vram uniform, 1 host uniform, 2 mixed skewed; activations apply to the "
                    "placement suite; decode T is 1..12 lanes x 1 token, or 4..48 as lanes x 4 "
                    "tokens)");
            }
        }
        if (suite != Suite::Placement) {
            for (const int tokens : decode_tokens) { static_cast<void>(decode_shape(tokens)); }
        }
        print_device_caps("moe_experts");
        Stores stores;
        stores.records          = record_layout();
        stores.warmup           = warmup;
        stores.repeat           = repeat;
        const std::size_t bytes = stores.records.bytes;
        // Record contents only need finite values: E4M3 scale and code bytes of 0x11 decode to
        // small normal numbers in both codecs.
        std::vector<std::uint8_t> pattern(bytes, 0x11);
        DeviceBuffer device_store(bytes);
        device_store.copy_from_host(pattern.data(), bytes);
        PinnedHost mapped_store(bytes);
        std::memcpy(mapped_store.host(), pattern.data(), bytes);
        stores.device = device_store.p;
        stores.mapped = mapped_store.device();
        std::vector<float> scales(static_cast<std::size_t>(3) * ops::kMoeExperts, 1.0e-3F);
        DeviceBuffer scales_d(scales.size() * sizeof(float));
        scales_d.copy_from_host(scales.data(), scales_d.bytes);
        DeviceBuffer shared_record(static_cast<std::size_t>(ops::kMoeNvfp4RecordBytes));
        shared_record.copy_from_host(pattern.data(), shared_record.bytes);
        DeviceBuffer shared_scales(3 * sizeof(float));
        shared_scales.copy_from_host(scales.data(), shared_scales.bytes);
        DeviceBuffer flush(kL2FlushBytes);
        stores.scales        = &scales_d;
        stores.shared_record = &shared_record;
        stores.shared_scales = &shared_scales;
        stores.flush         = &flush;
        if (suite != Suite::Decode) {
            for (const PlacementConfig& config : configs) {
                for (const int tokens : placement_tokens) {
                    for (const ops::MoeActivationProfile profile : profiles) {
                        run_placement_point(tokens, config, profile, stores);
                    }
                }
            }
        }
        if (suite != Suite::Placement) {
            for (const int tokens : decode_tokens) { run_decode_point(tokens, stores); }
        }
        return 0;
    } catch (const std::exception& error) {
        std::fprintf(stderr, "moe_experts_bench: %s\n", error.what());
        return 1;
    }
}
