// Public Qwen3.6-27B Q4/Q5 GDN projection/convolution/snapshot benchmark.
//
// The timed body is exactly one gdn_input_proj_conv_snapshot() public Op call.
// Production dispatch, kernel topology, and workspace use remain behind that contract.

#include "ninfer/ops/gdn_input_proj.h"

#include "core/device.h"
#include "ninfer_bench_common.h"
#include "quantized_weight.cuh"

#include <cuda_runtime.h>

#include <algorithm>
#include <cerrno>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <limits>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

using namespace ninfer;

namespace {

constexpr std::int32_t kHidden             = 5120;
constexpr std::int32_t kQueryRows          = 2048;
constexpr std::int32_t kKeyRows            = 2048;
constexpr std::int32_t kValueRows          = 6144;
constexpr std::int32_t kZRows              = 6144;
constexpr std::int32_t kQkRows             = kQueryRows + kKeyRows;
constexpr std::int32_t kValueZRows         = kValueRows + kZRows;
constexpr std::int32_t kChannels           = kQueryRows + kKeyRows + kValueRows;
constexpr std::uint64_t kDefaultFlushBytes = 256ULL << 20;

enum class Execution : std::uint8_t {
    Graph,
    Eager,
    Both,
};

enum class CacheMode : std::uint8_t {
    Cold,
    Warm,
    Both,
};

enum class CacheState : std::uint8_t {
    Cold,
    Warm,
};

struct TokenSweep {
    std::int32_t begin = 1;
    std::int32_t end   = 6;
    std::int32_t step  = 1;
};

struct Options {
    TokenSweep tokens;
    Execution execution       = Execution::Graph;
    CacheMode cache           = CacheMode::Both;
    int warmup                = 10;
    int repeat                = 100;
    std::uint64_t flush_bytes = kDefaultFlushBytes;
    std::string csv_out;
};

struct Stats {
    double median_us = 0.0;
    double min_us    = 0.0;
    double p95_us    = 0.0;
};

struct Result {
    std::int32_t tokens;
    Execution execution;
    CacheState cache;
    Stats stats;
    std::size_t workspace_bytes;
};

std::uint64_t parse_u64(std::string_view text, const char* label) {
    const std::string value(text);
    char* end                       = nullptr;
    errno                           = 0;
    const unsigned long long parsed = std::strtoull(value.c_str(), &end, 10);
    if (errno != 0 || end == value.c_str() || *end != '\0') {
        throw std::invalid_argument(std::string("invalid ") + label + ": " + value);
    }
    return static_cast<std::uint64_t>(parsed);
}

std::int32_t parse_positive_i32(std::string_view text, const char* label) {
    const std::uint64_t value = parse_u64(text, label);
    if (value == 0 ||
        value > static_cast<std::uint64_t>(std::numeric_limits<std::int32_t>::max())) {
        throw std::invalid_argument(std::string(label) + " must be in [1, INT32_MAX]");
    }
    return static_cast<std::int32_t>(value);
}

int parse_nonnegative_int(std::string_view text, const char* label) {
    const std::uint64_t value = parse_u64(text, label);
    if (value > static_cast<std::uint64_t>(std::numeric_limits<int>::max())) {
        throw std::invalid_argument(std::string(label) + " is too large");
    }
    return static_cast<int>(value);
}

TokenSweep parse_sweep(std::string_view text) {
    const std::string value(text);
    const std::size_t first = value.find(':');
    if (first == std::string::npos) {
        throw std::invalid_argument("--sweep must be START:END or START:END:STEP");
    }
    const std::size_t second = value.find(':', first + 1);
    if (second != std::string::npos && value.find(':', second + 1) != std::string::npos) {
        throw std::invalid_argument("--sweep has too many fields");
    }
    TokenSweep sweep;
    sweep.begin = parse_positive_i32(value.substr(0, first), "sweep start");
    sweep.end   = parse_positive_i32(value.substr(first + 1, second == std::string::npos
                                                                 ? std::string::npos
                                                                 : second - first - 1),
                                     "sweep end");
    if (second != std::string::npos) {
        sweep.step = parse_positive_i32(value.substr(second + 1), "sweep step");
    }
    if (sweep.begin > sweep.end) {
        throw std::invalid_argument("--sweep start must not exceed end");
    }
    return sweep;
}

Execution parse_execution(std::string_view value) {
    if (value == "graph") return Execution::Graph;
    if (value == "eager") return Execution::Eager;
    if (value == "both") return Execution::Both;
    throw std::invalid_argument("--execution must be graph, eager, or both");
}

CacheMode parse_cache(std::string_view value) {
    if (value == "cold") return CacheMode::Cold;
    if (value == "warm") return CacheMode::Warm;
    if (value == "both") return CacheMode::Both;
    throw std::invalid_argument("--cache must be cold, warm, or both");
}

void usage(const char* argv0) {
    std::fprintf(stderr,
                 "Usage: %s [options]\n\n"
                 "Public workload:\n"
                 "  --tokens T                    Exact token extent.\n"
                 "  --sweep START:END[:STEP]      Token sweep (default 1:6).\n\n"
                 "Measurement:\n"
                 "  --execution graph|eager|both  Default graph.\n"
                 "  --cache cold|warm|both        Default both; cold matches layer-to-layer use.\n"
                 "  --warmup N                    Warmup replays per point (default 10).\n"
                 "  --repeat N                    Measured samples per point (default 100).\n"
                 "  --flush-mib N                 L2 eviction storage (default 256 MiB).\n"
                 "  --csv-out PATH                Write result rows as CSV.\n"
                 "  -h, --help                    Show this text.\n",
                 argv0);
}

Options parse_options(int argc, char** argv) {
    Options options;
    bool have_tokens = false;
    bool have_sweep  = false;
    for (int index = 1; index < argc; ++index) {
        const std::string_view argument(argv[index]);
        const auto next = [&](const char* label) -> std::string_view {
            if (++index >= argc) { throw std::invalid_argument(std::string("missing ") + label); }
            return argv[index];
        };
        if (argument == "--tokens") {
            const std::int32_t tokens = parse_positive_i32(next("tokens"), "tokens");
            options.tokens            = {tokens, tokens, 1};
            have_tokens               = true;
        } else if (argument == "--sweep") {
            options.tokens = parse_sweep(next("sweep"));
            have_sweep     = true;
        } else if (argument == "--execution") {
            options.execution = parse_execution(next("execution"));
        } else if (argument == "--cache") {
            options.cache = parse_cache(next("cache"));
        } else if (argument == "--warmup") {
            options.warmup = parse_nonnegative_int(next("warmup"), "warmup");
        } else if (argument == "--repeat") {
            options.repeat = parse_nonnegative_int(next("repeat"), "repeat");
        } else if (argument == "--flush-mib") {
            const std::uint64_t mib = parse_u64(next("flush-mib"), "flush-mib");
            if (mib == 0 || mib > std::numeric_limits<std::uint64_t>::max() / (1ULL << 20)) {
                throw std::invalid_argument("--flush-mib is out of range");
            }
            options.flush_bytes = mib << 20;
        } else if (argument == "--csv-out") {
            options.csv_out = next("CSV output path");
        } else if (argument == "--help" || argument == "-h") {
            usage(argv[0]);
            std::exit(0);
        } else {
            throw std::invalid_argument("unknown argument: " + std::string(argument));
        }
    }
    if (have_tokens && have_sweep) {
        throw std::invalid_argument("--tokens and --sweep are mutually exclusive");
    }
    if (options.repeat <= 0) { throw std::invalid_argument("--repeat must be positive"); }
    if (options.flush_bytes > std::numeric_limits<std::size_t>::max()) {
        throw std::invalid_argument("flush buffer does not fit size_t");
    }
    return options;
}

std::vector<std::int32_t> selected_tokens(const TokenSweep& sweep) {
    std::vector<std::int32_t> result;
    for (std::int64_t tokens = sweep.begin; tokens <= sweep.end; tokens += sweep.step) {
        result.push_back(static_cast<std::int32_t>(tokens));
        if (tokens > static_cast<std::int64_t>(sweep.end) - sweep.step) { break; }
    }
    return result;
}

std::vector<Execution> selected_executions(Execution execution) {
    if (execution == Execution::Both) { return {Execution::Graph, Execution::Eager}; }
    return {execution};
}

std::vector<CacheState> selected_caches(CacheMode cache) {
    if (cache == CacheMode::Both) { return {CacheState::Cold, CacheState::Warm}; }
    return {cache == CacheMode::Cold ? CacheState::Cold : CacheState::Warm};
}

const char* execution_name(Execution execution) {
    switch (execution) {
    case Execution::Graph:
        return "graph_replay";
    case Execution::Eager:
        return "eager";
    case Execution::Both:
        break;
    }
    return "unknown";
}

const char* cache_name(CacheState cache) { return cache == CacheState::Cold ? "cold" : "warm"; }

class BenchmarkFixture {
public:
    explicit BenchmarkFixture(std::size_t flush_bytes)
        : qk_(bench::make_row_split_weight(QType::Q4G64_F16S, kQkRows, kHidden, kHidden,
                                           {0x53, 0x00, 0x3400})),
          value_z_(bench::make_row_split_weight(QType::Q5G64_F16S, kValueZRows, kHidden, kHidden,
                                                {0x53, 0x55, 0x3400})),
          conv_weight_(bench::make_bf16(static_cast<std::size_t>(kChannels) * 4)),
          flush_(flush_bytes) {
        CUDA_CHECK(cudaMemset(flush_.p, 0xa5, flush_.bytes));
        CUDA_CHECK(cudaDeviceSynchronize());
    }

    [[nodiscard]] const Weight& qk() const noexcept { return qk_.weight; }

    [[nodiscard]] const Weight& value_z() const noexcept { return value_z_.weight; }

    [[nodiscard]] Tensor conv_weight() const {
        return Tensor(conv_weight_.p, DType::BF16, {kChannels, 4});
    }

    void flush(cudaStream_t stream) {
        CUDA_CHECK(cudaMemsetAsync(flush_.p, 0xa5, flush_.bytes, stream));
    }

private:
    bench::PackedQuantizedWeight qk_;
    bench::PackedQuantizedWeight value_z_;
    DeviceBuffer conv_weight_;
    DeviceBuffer flush_;
};

class BenchmarkState {
public:
    BenchmarkState(BenchmarkFixture& fixture, std::int32_t tokens)
        : fixture_(fixture), slots_(tokens + 1),
          input_(bench::make_bf16(static_cast<std::size_t>(kHidden) * tokens)),
          states_(bench::make_bf16(static_cast<std::size_t>(kChannels) * 3 * slots_)),
          initial_slot_(sizeof(std::int32_t)), snapshot_base_slot_(sizeof(std::int32_t)),
          query_(static_cast<std::size_t>(kQueryRows) * tokens * 2),
          key_(static_cast<std::size_t>(kKeyRows) * tokens * 2),
          value_(static_cast<std::size_t>(kValueRows) * tokens * 2),
          z_(static_cast<std::size_t>(kZRows) * tokens * 2),
          workspace_bytes_(ops::gdn_input_proj_conv_snapshot_workspace_capacity_bytes(
              kQueryRows, kKeyRows, kValueRows, tokens, tokens)),
          workspace_(std::max<std::size_t>(1, workspace_bytes_)),
          x_(input_.p, DType::BF16, {kHidden, tokens}), conv_weight_(fixture.conv_weight()),
          conv_states_(states_.p, DType::BF16, {kChannels, 3, slots_}),
          initial_(initial_slot_.p, DType::I32, {1}),
          snapshot_base_(snapshot_base_slot_.p, DType::I32, {1}),
          query_tensor_(query_.p, DType::BF16, {kQueryRows, tokens}),
          key_tensor_(key_.p, DType::BF16, {kKeyRows, tokens}),
          value_tensor_(value_.p, DType::BF16, {kValueRows, tokens}),
          z_tensor_(z_.p, DType::BF16, {kZRows, tokens}) {
        const std::int32_t initial_slot          = tokens;
        constexpr std::int32_t kSnapshotBaseSlot = 0;
        initial_slot_.copy_from_host(&initial_slot, sizeof(initial_slot));
        snapshot_base_slot_.copy_from_host(&kSnapshotBaseSlot, sizeof(kSnapshotBaseSlot));
        CUDA_CHECK(cudaDeviceSynchronize());
    }

    [[nodiscard]] std::size_t workspace_bytes() const noexcept { return workspace_bytes_; }

    void prepare(CacheState cache, cudaStream_t stream) {
        if (cache == CacheState::Cold) { fixture_.flush(stream); }
    }

    void launch(cudaStream_t stream) {
        ops::gdn_input_proj_conv_snapshot(x_, fixture_.qk(), fixture_.value_z(), conv_weight_,
                                          conv_states_, initial_, snapshot_base_, query_tensor_,
                                          key_tensor_, value_tensor_, z_tensor_, workspace_,
                                          stream);
    }

private:
    BenchmarkFixture& fixture_;
    std::int32_t slots_;
    DeviceBuffer input_;
    DeviceBuffer states_;
    DeviceBuffer initial_slot_;
    DeviceBuffer snapshot_base_slot_;
    DeviceBuffer query_;
    DeviceBuffer key_;
    DeviceBuffer value_;
    DeviceBuffer z_;
    std::size_t workspace_bytes_;
    WorkspaceArena workspace_;
    Tensor x_;
    Tensor conv_weight_;
    Tensor conv_states_;
    Tensor initial_;
    Tensor snapshot_base_;
    Tensor query_tensor_;
    Tensor key_tensor_;
    Tensor value_tensor_;
    Tensor z_tensor_;
};

class BodyTimedGraph {
public:
    BodyTimedGraph() {
        CUDA_CHECK(cudaEventCreate(&body_start_));
        CUDA_CHECK(cudaEventCreate(&body_stop_));
        CUDA_CHECK(cudaEventCreateWithFlags(&completion_, cudaEventDisableTiming));
    }

    ~BodyTimedGraph() {
        if (exec_ != nullptr) { cudaGraphExecDestroy(exec_); }
        if (graph_ != nullptr) { cudaGraphDestroy(graph_); }
        if (body_start_ != nullptr) { cudaEventDestroy(body_start_); }
        if (body_stop_ != nullptr) { cudaEventDestroy(body_stop_); }
        if (completion_ != nullptr) { cudaEventDestroy(completion_); }
    }

    BodyTimedGraph(const BodyTimedGraph&)            = delete;
    BodyTimedGraph& operator=(const BodyTimedGraph&) = delete;

    template <class Launch>
    void capture(cudaStream_t stream, Launch&& launch) {
        CUDA_CHECK(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal));
        CUDA_CHECK(cudaEventRecordWithFlags(body_start_, stream, cudaEventRecordExternal));
        launch(stream);
        CUDA_CHECK(cudaEventRecordWithFlags(body_stop_, stream, cudaEventRecordExternal));
        CUDA_CHECK(cudaStreamEndCapture(stream, &graph_));
        CUDA_CHECK(cudaGraphInstantiate(&exec_, graph_, 0));
        std::size_t nodes = 0;
        CUDA_CHECK(cudaGraphGetNodes(graph_, nullptr, &nodes));
        if (nodes < 3) { throw std::runtime_error("GDN snapshot capture produced an empty graph"); }
    }

    void launch(cudaStream_t stream) const { CUDA_CHECK(cudaGraphLaunch(exec_, stream)); }

    double launch_body_timed(cudaStream_t stream) const {
        launch(stream);
        CUDA_CHECK(cudaEventRecord(completion_, stream));
        CUDA_CHECK(cudaEventSynchronize(completion_));
        float milliseconds = 0.0F;
        CUDA_CHECK(cudaEventElapsedTime(&milliseconds, body_start_, body_stop_));
        return static_cast<double>(milliseconds) * 1000.0;
    }

private:
    cudaGraph_t graph_      = nullptr;
    cudaGraphExec_t exec_   = nullptr;
    cudaEvent_t body_start_ = nullptr;
    cudaEvent_t body_stop_  = nullptr;
    cudaEvent_t completion_ = nullptr;
};

Stats summarize(std::vector<double> samples) {
    if (samples.empty()) { throw std::invalid_argument("cannot summarize an empty sample set"); }
    std::sort(samples.begin(), samples.end());
    const auto percentile = [&](double fraction) {
        const std::size_t index =
            std::min(samples.size() - 1,
                     static_cast<std::size_t>(fraction * static_cast<double>(samples.size() - 1)));
        return samples[index];
    };
    return {percentile(0.50), samples.front(), percentile(0.95)};
}

Stats measure_eager(BenchmarkState& state, CacheState cache, cudaStream_t stream, int warmup,
                    int repeat) {
    cudaEvent_t start = nullptr;
    cudaEvent_t stop  = nullptr;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    for (int index = 0; index < warmup; ++index) {
        state.prepare(cache, stream);
        state.launch(stream);
    }
    CUDA_CHECK(cudaStreamSynchronize(stream));

    std::vector<double> samples;
    samples.reserve(static_cast<std::size_t>(repeat));
    for (int index = 0; index < repeat; ++index) {
        state.prepare(cache, stream);
        CUDA_CHECK(cudaEventRecord(start, stream));
        state.launch(stream);
        CUDA_CHECK(cudaEventRecord(stop, stream));
        CUDA_CHECK(cudaEventSynchronize(stop));
        float milliseconds = 0.0F;
        CUDA_CHECK(cudaEventElapsedTime(&milliseconds, start, stop));
        samples.push_back(static_cast<double>(milliseconds) * 1000.0);
    }
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    return summarize(std::move(samples));
}

Stats measure_graph(BenchmarkState& state, const BodyTimedGraph& graph, CacheState cache,
                    cudaStream_t stream, int warmup, int repeat) {
    for (int index = 0; index < warmup; ++index) {
        state.prepare(cache, stream);
        graph.launch(stream);
    }
    CUDA_CHECK(cudaStreamSynchronize(stream));

    std::vector<double> samples;
    samples.reserve(static_cast<std::size_t>(repeat));
    for (int index = 0; index < repeat; ++index) {
        state.prepare(cache, stream);
        samples.push_back(graph.launch_body_timed(stream));
    }
    return summarize(std::move(samples));
}

std::vector<Result> run_point(BenchmarkFixture& fixture, std::int32_t tokens,
                              const Options& options, cudaStream_t stream) {
    BenchmarkState state(fixture, tokens);

    // Match production graph lifecycle: materialize once, capture, instantiate,
    // and prime one replay before the configured measurements.
    state.prepare(CacheState::Warm, stream);
    state.launch(stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));

    BodyTimedGraph graph;
    if (options.execution != Execution::Eager) {
        graph.capture(stream, [&](cudaStream_t launch_stream) { state.launch(launch_stream); });
        graph.launch(stream);
        CUDA_CHECK(cudaStreamSynchronize(stream));
    }

    std::vector<Result> results;
    for (Execution execution : selected_executions(options.execution)) {
        for (CacheState cache : selected_caches(options.cache)) {
            const Stats stats =
                execution == Execution::Graph
                    ? measure_graph(state, graph, cache, stream, options.warmup, options.repeat)
                    : measure_eager(state, cache, stream, options.warmup, options.repeat);
            results.push_back({tokens, execution, cache, stats, state.workspace_bytes()});
        }
    }
    return results;
}

void print_result(const Result& result) {
    std::printf("q4-q5 T=%-3d %-12s %-4s median=%8.3f us min=%8.3f us p95=%8.3f us "
                "workspace=%zu\n",
                result.tokens, execution_name(result.execution), cache_name(result.cache),
                result.stats.median_us, result.stats.min_us, result.stats.p95_us,
                result.workspace_bytes);
}

void write_csv(const std::string& path, const std::vector<Result>& results, const Options& options,
               const DeviceContext& context) {
    if (path.empty()) { return; }
    const std::filesystem::path output(path);
    if (!output.parent_path().empty()) {
        std::filesystem::create_directories(output.parent_path());
    }
    std::ofstream stream(output);
    if (!stream) { throw std::runtime_error("failed to open CSV output"); }
    int runtime = 0;
    CUDA_CHECK(cudaRuntimeGetVersion(&runtime));
    stream << "profile,tokens,execution,timed_scope,cache,median_us,min_us,p95_us,"
              "workspace_bytes,warmup,repeat,flush_bytes,build_type,gpu,cuda_runtime\n";
    for (const Result& result : results) {
        stream << "q4-q5," << result.tokens << ',' << execution_name(result.execution)
               << ",full_gdn_input_proj_conv_snapshot_device_body," << cache_name(result.cache)
               << ',' << result.stats.median_us << ',' << result.stats.min_us << ','
               << result.stats.p95_us << ',' << result.workspace_bytes << ',' << options.warmup
               << ',' << options.repeat << ',' << options.flush_bytes << ','
#ifdef NDEBUG
               << "Release"
#else
               << "Debug"
#endif
               << ',' << context.props.name << ',' << runtime << '\n';
    }
}

} // namespace

int main(int argc, char** argv) {
    try {
        const Options options = parse_options(argc, argv);
        DeviceContext context;
        std::printf("# op=gdn_input_proj_conv_snapshot profile=q4-q5 gpu=%s sm=%d execution=%s "
                    "timed_scope=full_public_op_device_body cold_flush_mib=%llu\n",
                    context.props.name, context.sm(),
                    options.execution == Execution::Both ? "both"
                                                         : execution_name(options.execution),
                    static_cast<unsigned long long>(options.flush_bytes >> 20));

        BenchmarkFixture fixture(static_cast<std::size_t>(options.flush_bytes));
        std::vector<Result> results;
        for (std::int32_t tokens : selected_tokens(options.tokens)) {
            std::vector<Result> point = run_point(fixture, tokens, options, context.stream);
            for (const Result& result : point) { print_result(result); }
            results.insert(results.end(), std::make_move_iterator(point.begin()),
                           std::make_move_iterator(point.end()));
        }
        write_csv(options.csv_out, results, options, context);
        return 0;
    } catch (const std::exception& error) {
        std::fprintf(stderr, "ninfer_gdn_input_proj_conv_snapshot_bench: %s\n", error.what());
        return 1;
    }
}
