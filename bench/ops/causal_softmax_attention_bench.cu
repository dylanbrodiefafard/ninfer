// Public-Op benchmark for causal cache Softmax Attention.
//
// The two benchmark entries map directly to the public append-and-attend and cached-only
// contracts. Decode, small-T, prompt, split-KV, and kernel selection remain private production
// implementation details and never enter this benchmark's dispatch or output schema.

#include "ninfer/ops/gqa_attention.h"

#include "core/device.h"
#include "core/kv_cache.h"
#include "ninfer_bench_common.h"

#include <cuda_profiler_api.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cerrno>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

using namespace ninfer;

namespace {

constexpr std::int32_t kHeadDim     = 256;
constexpr std::int32_t kKvGroup     = 64;
constexpr float kScale              = 0.0625F;
constexpr std::size_t kFlushBytes   = std::size_t{256} << 20;
constexpr double kDenseBf16TcTflops = 209.5;
constexpr double kRtx5090DramGBs    = 1792.0;

enum class Entry : std::uint8_t { Append, Cached, Both };
enum class GeometryChoice : std::uint8_t { H24Kv4, H16Kv2, All };
enum class KvChoice : std::uint8_t { Bf16, Int8, All };
enum class Execution : std::uint8_t { Eager, Graph, Both };
enum class CacheMode : std::uint8_t { Cold, Warm, Both };
enum class CacheState : std::uint8_t { Cold, Warm };

struct Geometry {
    const char* name;
    std::int32_t query_heads;
    std::int32_t kv_heads;
};

constexpr Geometry kH24Kv4{"d256-h24-kv4", 24, 4};
constexpr Geometry kH16Kv2{"d256-h16-kv2", 16, 2};

struct Options {
    Entry entry             = Entry::Both;
    GeometryChoice geometry = GeometryChoice::All;
    KvChoice kv             = KvChoice::All;
    Execution execution     = Execution::Graph;
    CacheMode cache         = CacheMode::Cold;
    std::vector<std::int32_t> tokens{1, 2, 4, 6, 8, 12, 16, 1024};
    std::vector<std::int32_t> contexts{0, 128, 2048, 8192};
    int warmup   = 5;
    int repeat   = 30;
    bool profile = false;
    std::string csv_out;
};

struct Result {
    Entry entry;
    Geometry geometry;
    DType kv_dtype;
    Execution execution;
    CacheState cache;
    std::int32_t tokens;
    std::int32_t context;
    std::size_t workspace_bytes;
    double logical_bytes;
    double useful_flops;
    bench::ColdTiming timing;
};

[[noreturn]] void usage(const char* message) {
    std::fprintf(stderr,
                 "error: %s\n"
                 "usage: ninfer_causal_softmax_attention_bench "
                 "[--entry append|cached|both] "
                 "[--geometry d256-h24-kv4|d256-h16-kv2|all] "
                 "[--kv-dtype bf16|int8|all] [--tokens T,...] [--context L,...] "
                 "[--execution eager|graph|both] [--cache cold|warm|both] "
                 "[--warmup N] [--repeat N] [--profile] [--csv-out PATH]\n",
                 message);
    std::exit(2);
}

std::int32_t parse_i32(std::string_view text, std::int32_t minimum, std::int32_t maximum,
                       const char* flag) {
    const std::string value(text);
    errno       = 0;
    char* end   = nullptr;
    long parsed = std::strtol(value.c_str(), &end, 10);
    if (errno != 0 || end == value.c_str() || *end != '\0' || parsed < minimum ||
        parsed > maximum) {
        usage(flag);
    }
    return static_cast<std::int32_t>(parsed);
}

std::vector<std::int32_t> parse_list(const char* text, std::int32_t minimum, std::int32_t maximum,
                                     const char* flag) {
    std::vector<std::int32_t> result;
    std::string_view remaining(text);
    while (!remaining.empty()) {
        const std::size_t comma     = remaining.find(',');
        const std::string_view item = remaining.substr(0, comma);
        if (item.empty()) { usage(flag); }
        result.push_back(parse_i32(item, minimum, maximum, flag));
        if (comma == std::string_view::npos) { break; }
        remaining.remove_prefix(comma + 1);
    }
    if (result.empty()) { usage(flag); }
    return result;
}

Options parse_options(int argc, char** argv) {
    Options options;
    for (int index = 1; index < argc; ++index) {
        const std::string_view argument(argv[index]);
        const auto next = [&](const char* flag) -> const char* {
            if (++index == argc) { usage(flag); }
            return argv[index];
        };
        if (argument == "--entry") {
            const std::string_view value(next("--entry requires a value"));
            if (value == "append")
                options.entry = Entry::Append;
            else if (value == "cached")
                options.entry = Entry::Cached;
            else if (value == "both")
                options.entry = Entry::Both;
            else
                usage("--entry expects append, cached, or both");
        } else if (argument == "--geometry") {
            const std::string_view value(next("--geometry requires a value"));
            if (value == "d256-h24-kv4")
                options.geometry = GeometryChoice::H24Kv4;
            else if (value == "d256-h16-kv2")
                options.geometry = GeometryChoice::H16Kv2;
            else if (value == "all")
                options.geometry = GeometryChoice::All;
            else
                usage("--geometry expects d256-h24-kv4, d256-h16-kv2, or all");
        } else if (argument == "--kv-dtype") {
            const std::string_view value(next("--kv-dtype requires a value"));
            if (value == "bf16")
                options.kv = KvChoice::Bf16;
            else if (value == "int8")
                options.kv = KvChoice::Int8;
            else if (value == "all")
                options.kv = KvChoice::All;
            else
                usage("--kv-dtype expects bf16, int8, or all");
        } else if (argument == "--tokens") {
            options.tokens = parse_list(next("--tokens requires a value"), 1, 262144, "--tokens");
        } else if (argument == "--context") {
            options.contexts =
                parse_list(next("--context requires a value"), 0, 262144, "--context");
        } else if (argument == "--execution") {
            const std::string_view value(next("--execution requires a value"));
            if (value == "eager")
                options.execution = Execution::Eager;
            else if (value == "graph")
                options.execution = Execution::Graph;
            else if (value == "both")
                options.execution = Execution::Both;
            else
                usage("--execution expects eager, graph, or both");
        } else if (argument == "--cache") {
            const std::string_view value(next("--cache requires a value"));
            if (value == "cold")
                options.cache = CacheMode::Cold;
            else if (value == "warm")
                options.cache = CacheMode::Warm;
            else if (value == "both")
                options.cache = CacheMode::Both;
            else
                usage("--cache expects cold, warm, or both");
        } else if (argument == "--warmup") {
            options.warmup = parse_i32(next("--warmup requires a value"), 0, 10000, "--warmup");
        } else if (argument == "--repeat") {
            options.repeat = parse_i32(next("--repeat requires a value"), 1, 10000, "--repeat");
        } else if (argument == "--profile") {
            options.profile = true;
        } else if (argument == "--csv-out") {
            options.csv_out = next("--csv-out requires a path");
        } else if (argument == "--help" || argument == "-h") {
            usage("help");
        } else {
            usage("unknown argument");
        }
    }
    for (const std::int32_t tokens : options.tokens) {
        for (const std::int32_t context : options.contexts) {
            if (context > std::numeric_limits<std::int32_t>::max() - tokens) {
                usage("context + tokens exceeds int32");
            }
        }
    }
    if (options.profile &&
        (options.entry == Entry::Both || options.geometry == GeometryChoice::All ||
         options.kv == KvChoice::All || options.tokens.size() != 1 ||
         options.contexts.size() != 1 || options.execution == Execution::Both ||
         options.cache == CacheMode::Both)) {
        usage("--profile requires one entry, geometry, dtype, T, context, execution, and cache");
    }
    return options;
}

std::int32_t align_context(std::int32_t visible) { return ((visible + 127) / 128) * 128; }

std::size_t cache_plane_bytes(const Geometry& geometry, DType dtype, std::int32_t padded) {
    return static_cast<std::size_t>(kHeadDim) * geometry.kv_heads * padded * dtype_size(dtype);
}

std::size_t scale_plane_bytes(const Geometry& geometry, std::int32_t padded) {
    return static_cast<std::size_t>(kHeadDim / kKvGroup) * geometry.kv_heads * padded *
           dtype_size(DType::FP16);
}

KVCacheLayerView make_cache_view(DeviceBuffer& k, DeviceBuffer& v, DeviceBuffer& k_scale,
                                 DeviceBuffer& v_scale, const Geometry& geometry, DType dtype,
                                 std::int32_t visible, std::int32_t padded) {
    const bool quantized = dtype == DType::I8;
    return {
        .k              = Tensor(k.p, dtype, {kHeadDim, padded, geometry.kv_heads}),
        .v              = Tensor(v.p, dtype, {kHeadDim, padded, geometry.kv_heads}),
        .k_scale        = quantized ? Tensor(k_scale.p, DType::FP16,
                                             {kHeadDim / kKvGroup, padded, geometry.kv_heads})
                                    : Tensor(),
        .v_scale        = quantized ? Tensor(v_scale.p, DType::FP16,
                                             {kHeadDim / kKvGroup, padded, geometry.kv_heads})
                                    : Tensor(),
        .max_context    = static_cast<std::uint32_t>(visible),
        .padded_context = static_cast<std::uint32_t>(padded),
        .num_kv_heads   = geometry.kv_heads,
        .head_dim       = kHeadDim,
        .dtype          = dtype,
        .quant_group    = quantized ? kKvGroup : 0,
    };
}

std::size_t workspace_capacity(const Geometry& geometry, DType dtype, std::int32_t tokens,
                               std::int32_t visible) {
    const ops::GqaExecutionEnvelope envelope{static_cast<std::uint32_t>(visible),
                                             static_cast<std::uint32_t>(visible)};
    return ops::gqa_attention_workspace_capacity_bytes(geometry.query_heads, dtype, envelope,
                                                       tokens, tokens);
}

class Case {
public:
    Case(Geometry geometry, DType dtype, std::int32_t tokens, std::int32_t context)
        : geometry_(geometry), dtype_(dtype), tokens_(tokens), context_(context),
          visible_(context + tokens), padded_(align_context(visible_)),
          q_(bench::make_bf16(static_cast<std::size_t>(kHeadDim) * geometry.query_heads * tokens)),
          k_(bench::make_bf16(static_cast<std::size_t>(kHeadDim) * geometry.kv_heads * tokens)),
          v_(bench::make_bf16(static_cast<std::size_t>(kHeadDim) * geometry.kv_heads * tokens)),
          positions_(static_cast<std::size_t>(tokens) * sizeof(std::int32_t)),
          cache_k_(bench::make_zeros(cache_plane_bytes(geometry, dtype, padded_))),
          cache_v_(bench::make_zeros(cache_plane_bytes(geometry, dtype, padded_))),
          cache_k_scale_(bench::make_zeros(dtype == DType::I8 ? scale_plane_bytes(geometry, padded_)
                                                              : std::size_t{1})),
          cache_v_scale_(bench::make_zeros(dtype == DType::I8 ? scale_plane_bytes(geometry, padded_)
                                                              : std::size_t{1})),
          output_(bench::make_zeros(static_cast<std::size_t>(kHeadDim) * geometry.query_heads *
                                    tokens * 2)),
          workspace_bytes_(workspace_capacity(geometry, dtype, tokens, visible_)),
          workspace_(std::max<std::size_t>(workspace_bytes_, 1)),
          q_tensor_(q_.p, DType::BF16, {kHeadDim, geometry.query_heads, tokens}),
          k_tensor_(k_.p, DType::BF16, {kHeadDim, geometry.kv_heads, tokens}),
          v_tensor_(v_.p, DType::BF16, {kHeadDim, geometry.kv_heads, tokens}),
          positions_tensor_(positions_.p, DType::I32, {tokens}),
          output_tensor_(output_.p, DType::BF16, {kHeadDim, geometry.query_heads, tokens}),
          cache_view_(make_cache_view(cache_k_, cache_v_, cache_k_scale_, cache_v_scale_, geometry,
                                      dtype, visible_, padded_)),
          envelope_{static_cast<std::uint32_t>(visible_), static_cast<std::uint32_t>(visible_)} {
        std::vector<std::int32_t> host_positions(static_cast<std::size_t>(tokens));
        for (std::int32_t token = 0; token < tokens; ++token) {
            host_positions[static_cast<std::size_t>(token)] = context + token;
        }
        CUDA_CHECK(cudaMemcpy(positions_.p, host_positions.data(), positions_.bytes,
                              cudaMemcpyHostToDevice));
    }

    void launch(Entry entry, cudaStream_t stream) {
        if (entry == Entry::Append) {
            ops::gqa_attention(q_tensor_, k_tensor_, v_tensor_, positions_tensor_, kScale,
                               cache_view_, envelope_, workspace_, output_tensor_, stream);
        } else {
            ops::gqa_attention_cached(q_tensor_, positions_tensor_, kScale, cache_view_, envelope_,
                                      workspace_, output_tensor_, stream);
        }
    }

    [[nodiscard]] std::size_t workspace_bytes() const noexcept { return workspace_bytes_; }

private:
    Geometry geometry_;
    DType dtype_;
    std::int32_t tokens_;
    std::int32_t context_;
    std::int32_t visible_;
    std::int32_t padded_;
    DeviceBuffer q_;
    DeviceBuffer k_;
    DeviceBuffer v_;
    DeviceBuffer positions_;
    DeviceBuffer cache_k_;
    DeviceBuffer cache_v_;
    DeviceBuffer cache_k_scale_;
    DeviceBuffer cache_v_scale_;
    DeviceBuffer output_;
    std::size_t workspace_bytes_;
    WorkspaceArena workspace_;
    Tensor q_tensor_;
    Tensor k_tensor_;
    Tensor v_tensor_;
    Tensor positions_tensor_;
    Tensor output_tensor_;
    KVCacheLayerView cache_view_;
    ops::GqaExecutionEnvelope envelope_;
};

const char* entry_name(Entry entry) { return entry == Entry::Append ? "append" : "cached"; }

const char* dtype_name(DType dtype) { return dtype == DType::BF16 ? "bf16" : "int8"; }

const char* execution_name(Execution execution) {
    return execution == Execution::Eager ? "eager" : "graph";
}

const char* cache_name(CacheState cache) { return cache == CacheState::Cold ? "cold" : "warm"; }

double cache_vector_bytes(DType dtype) {
    return dtype == DType::BF16
               ? static_cast<double>(kHeadDim * dtype_size(DType::BF16))
               : static_cast<double>(kHeadDim * dtype_size(DType::I8) +
                                     (kHeadDim / kKvGroup) * dtype_size(DType::FP16));
}

double causal_key_sum(std::int32_t tokens, std::int32_t context) {
    const double t = static_cast<double>(tokens);
    return t * context + t * static_cast<double>(tokens + 1) * 0.5;
}

double logical_bytes(Entry entry, const Geometry& geometry, DType dtype, std::int32_t tokens,
                     std::int32_t context) {
    const double q_and_output = 2.0 * kHeadDim * geometry.query_heads * tokens * 2.0;
    const double cache_reads =
        causal_key_sum(tokens, context) * geometry.kv_heads * 2.0 * cache_vector_bytes(dtype);
    if (entry == Entry::Cached) { return q_and_output + cache_reads; }
    const double input_kv     = 2.0 * kHeadDim * geometry.kv_heads * tokens * 2.0;
    const double cache_writes = 2.0 * geometry.kv_heads * tokens * cache_vector_bytes(dtype);
    return q_and_output + cache_reads + input_kv + cache_writes;
}

double useful_flops(const Geometry& geometry, std::int32_t tokens, std::int32_t context) {
    return 4.0 * kHeadDim * geometry.query_heads * causal_key_sum(tokens, context);
}

bench::ColdTiming measure(Case& data, Entry entry, Execution execution, CacheState cache,
                          bench::TimedGraph* graph, DeviceBuffer& flush, cudaStream_t stream,
                          int warmup, int repeat) {
    if (execution == Execution::Eager) {
        const auto launch = [&](cudaStream_t launch_stream) { data.launch(entry, launch_stream); };
        return cache == CacheState::Cold
                   ? bench::measure_cold_launch(launch, flush, stream, warmup, repeat)
                   : bench::measure_launch(launch, stream, warmup, repeat);
    }
    return cache == CacheState::Cold
               ? bench::measure_cold_graph(*graph, flush, stream, warmup, repeat)
               : bench::measure_graph(*graph, stream, warmup, repeat);
}

void report(const Result& result) {
    const double seconds = result.timing.median_us * 1.0e-6;
    const double gbps    = result.logical_bytes / seconds / 1.0e9;
    const double tflops  = result.useful_flops / seconds / 1.0e12;
    std::printf("entry=%-6s geometry=%-14s kv=%-4s execution=%-5s cache=%-4s T=%6d L=%7d "
                "workspace=%9zu median=%10.3f us min=%10.3f us p95=%10.3f us "
                "logical=%8.1f GB/s (%5.1f%% of %.0f) math=%7.2f TFLOP/s (%5.1f%% of %.1f)\n",
                entry_name(result.entry), result.geometry.name, dtype_name(result.kv_dtype),
                execution_name(result.execution), cache_name(result.cache), result.tokens,
                result.context, result.workspace_bytes, result.timing.median_us,
                result.timing.min_us, result.timing.p95_us, gbps, gbps / kRtx5090DramGBs * 100.0,
                kRtx5090DramGBs, tflops, tflops / kDenseBf16TcTflops * 100.0, kDenseBf16TcTflops);
}

void write_csv(const Options& options, const std::vector<Result>& results) {
    if (options.csv_out.empty()) { return; }
    const std::filesystem::path path(options.csv_out);
    if (!path.parent_path().empty()) { std::filesystem::create_directories(path.parent_path()); }
    std::ofstream output(path);
    if (!output) { throw std::runtime_error("failed to open CSV output"); }
    output << "entry,geometry,kv_dtype,execution,cache,T,context,workspace_bytes,logical_bytes,"
              "useful_flops,median_us,min_us,p95_us\n";
    for (const Result& result : results) {
        output << entry_name(result.entry) << ',' << result.geometry.name << ','
               << dtype_name(result.kv_dtype) << ',' << execution_name(result.execution) << ','
               << cache_name(result.cache) << ',' << result.tokens << ',' << result.context << ','
               << result.workspace_bytes << ',' << result.logical_bytes << ','
               << result.useful_flops << ',' << result.timing.median_us << ','
               << result.timing.min_us << ',' << result.timing.p95_us << '\n';
    }
}

void profile(Case& data, Entry entry, const Geometry& geometry, DType dtype, const Options& options,
             DeviceBuffer& flush, cudaStream_t stream) {
    const Execution execution = options.execution;
    const CacheState cache = options.cache == CacheMode::Cold ? CacheState::Cold : CacheState::Warm;
    bench::TimedGraph graph;
    if (execution == Execution::Graph) {
        data.launch(entry, stream);
        CUDA_CHECK(cudaStreamSynchronize(stream));
        graph.capture(stream,
                      [&](cudaStream_t launch_stream) { data.launch(entry, launch_stream); });
        for (int index = 0; index < options.warmup; ++index) { graph.launch(stream); }
    } else {
        for (int index = 0; index < options.warmup; ++index) { data.launch(entry, stream); }
    }
    CUDA_CHECK(cudaStreamSynchronize(stream));
    if (cache == CacheState::Cold) {
        bench::flush_l2(flush, stream);
        CUDA_CHECK(cudaStreamSynchronize(stream));
    }
    std::printf("PROFILE entry=%s geometry=%s kv=%s dispatch=public execution=%s cache=%s\n",
                entry_name(entry), geometry.name, dtype_name(dtype), execution_name(execution),
                cache_name(cache));
    std::fflush(stdout);
    CUDA_CHECK(cudaProfilerStart());
    if (execution == Execution::Graph)
        graph.launch(stream);
    else
        data.launch(entry, stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));
    CUDA_CHECK(cudaProfilerStop());
}

std::vector<Geometry> selected_geometries(GeometryChoice choice) {
    if (choice == GeometryChoice::H24Kv4) { return {kH24Kv4}; }
    if (choice == GeometryChoice::H16Kv2) { return {kH16Kv2}; }
    return {kH24Kv4, kH16Kv2};
}

std::vector<DType> selected_dtypes(KvChoice choice) {
    if (choice == KvChoice::Bf16) { return {DType::BF16}; }
    if (choice == KvChoice::Int8) { return {DType::I8}; }
    return {DType::BF16, DType::I8};
}

} // namespace

int main(int argc, char** argv) {
    try {
        int devices = 0;
        if (cudaGetDeviceCount(&devices) != cudaSuccess || devices == 0) {
            std::printf("SKIP: no usable CUDA device\n");
            return 0;
        }
        const Options options = parse_options(argc, argv);
        cudaStream_t stream   = nullptr;
        CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
        DeviceBuffer flush(kFlushBytes);
        const std::vector<Geometry> geometries = selected_geometries(options.geometry);
        const std::vector<DType> dtypes        = selected_dtypes(options.kv);

        if (options.profile) {
            const Entry entry       = options.entry;
            const Geometry geometry = geometries.front();
            const DType dtype       = dtypes.front();
            Case data(geometry, dtype, options.tokens.front(), options.contexts.front());
            profile(data, entry, geometry, dtype, options, flush, stream);
            CUDA_CHECK(cudaStreamDestroy(stream));
            return 0;
        }

        std::vector<Result> results;
        for (const Geometry& geometry : geometries) {
            for (const DType dtype : dtypes) {
                for (const std::int32_t context : options.contexts) {
                    for (const std::int32_t tokens : options.tokens) {
                        Case data(geometry, dtype, tokens, context);
                        for (const Entry entry : {Entry::Append, Entry::Cached}) {
                            if ((options.entry == Entry::Append && entry != Entry::Append) ||
                                (options.entry == Entry::Cached && entry != Entry::Cached)) {
                                continue;
                            }
                            bench::TimedGraph graph;
                            if (options.execution != Execution::Eager) {
                                data.launch(entry, stream);
                                CUDA_CHECK(cudaStreamSynchronize(stream));
                                graph.capture(stream, [&](cudaStream_t launch_stream) {
                                    data.launch(entry, launch_stream);
                                });
                            }
                            for (const Execution execution : {Execution::Eager, Execution::Graph}) {
                                if ((options.execution == Execution::Eager &&
                                     execution != Execution::Eager) ||
                                    (options.execution == Execution::Graph &&
                                     execution != Execution::Graph)) {
                                    continue;
                                }
                                for (const CacheState cache :
                                     {CacheState::Cold, CacheState::Warm}) {
                                    if ((options.cache == CacheMode::Cold &&
                                         cache != CacheState::Cold) ||
                                        (options.cache == CacheMode::Warm &&
                                         cache != CacheState::Warm)) {
                                        continue;
                                    }
                                    Result result{
                                        entry,
                                        geometry,
                                        dtype,
                                        execution,
                                        cache,
                                        tokens,
                                        context,
                                        data.workspace_bytes(),
                                        logical_bytes(entry, geometry, dtype, tokens, context),
                                        useful_flops(geometry, tokens, context),
                                        measure(data, entry, execution, cache, &graph, flush,
                                                stream, options.warmup, options.repeat)};
                                    report(result);
                                    results.push_back(result);
                                }
                            }
                        }
                    }
                }
            }
        }
        write_csv(options, results);
        CUDA_CHECK(cudaStreamDestroy(stream));
        return 0;
    } catch (const std::exception& error) {
        std::fprintf(stderr, "ninfer_causal_softmax_attention_bench: %s\n", error.what());
        return 1;
    }
}
