// Public-Op benchmark for qsa_index_select and qsa_selected_attention at the Qwen3.8-Flash-Next
// QSA geometry (24 query heads of width 256 over 2 KV heads, a 4x128 indexer over blocks of 4
// tokens). A paged state of B lanes of capacity C = context each, its 64-token pages in a
// shuffled physical order, is filled through lane-batched qsa_state_append calls; every lane's W
// queries occupy its last W slots [C-W, C), so every query sees a prefix of at least C-W+1 tokens
// (decode W=1 and W=12, prefill chunks W=512 and W=4096 at B=1; lane-batched decode and verify
// W=1 and W=4 at B=4 and 12). Attention consumes the selection the select Op produced, so its
// gather locality is the real one.
//
// Each Op call is captured in a CUDA graph and timed cold: a 256 MiB L2 flush precedes every
// replay. Bandwidth is the contract's compulsory bytes over the median time against the
// 1674.5 GB/s sustained-read roof:
//   select     the sealed FP32 key of every complete block of the longest prefix when some query
//              ranks, the index queries and the selected-id output;
//   attention  the K and V rows of every listed slot for both KV heads, per query (no dedupe across
//              queries, so prefill figures are an upper bound of DRAM traffic), q and out.
// TFLOP/s counts the useful work: select 2*4*128 per ranked (query, complete block); attention
// 4*256 per (query, query head, listed slot) for QK^T and PV.

#include "ninfer/ops/qsa.h"

#include "core/arena.h"
#include "core/device.h"
#include "core/tensor.h"
#include "ninfer_bench_common.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <string>
#include <string_view>
#include <vector>

using namespace ninfer;

namespace {

using ops::QsaKvFormat;
using ops::QsaSelectBudget;
using ops::QsaPagedState;

constexpr std::int32_t kHeadDim      = ops::kQsaHeadDim;
constexpr std::int32_t kKvHeads      = ops::kQsaKvHeads;
constexpr std::int32_t kQueryHeads   = ops::kQsaQueryHeads;
constexpr std::int32_t kIndexDim     = ops::kQsaIndexHeadDim;
constexpr std::int32_t kIndexHeads   = ops::kQsaIndexQueryHeads;
constexpr std::int32_t kBlockTokens  = ops::kQsaBlockTokens;
constexpr std::int32_t kPageTokens   = ops::kQsaPageTokens;
constexpr std::int32_t kAppendChunk  = 32768;
constexpr std::size_t kFlushBytes    = std::size_t{256} << 20;
constexpr double kDramRoofGBs        = 1674.5;
constexpr double kAutoRepeatBudgetUs = 300000.0;
constexpr int kAutoRepeatMinimum     = 3;
constexpr int kAutoRepeatMaximum     = 50;

struct Options {
    std::vector<std::int32_t> contexts{8192, 65536, 262144};
    std::vector<std::int32_t> widths{1, 12, 512, 4096};
    std::vector<std::int32_t> budgets{2048, 512};
    std::vector<std::int32_t> lanes{1};
    QsaKvFormat format = QsaKvFormat::NVFP4G16;
    bool select        = true;
    bool attention     = true;
    int warmup         = 2;
    int repeat         = 0; // 0: as many replays as fit kAutoRepeatBudgetUs, within [3,50]
};

[[noreturn]] void usage(const char* message) {
    std::fprintf(
        stderr,
        "error: %s\n"
        "usage: ninfer_qsa_bench [--context L,...] [--width W,...] [--budget 512|1024|2048,...]"
        " [--lanes B,...] [--kv nvfp4|bf16] [--op select|attention|all] [--warmup N]"
        " [--repeat N]\n",
        message);
    std::exit(2);
}

std::vector<std::int32_t> parse_list(std::string_view text, const char* flag) {
    std::vector<std::int32_t> values;
    while (!text.empty()) {
        const std::size_t comma = text.find(',');
        values.push_back(bench::parse_number<std::int32_t>(text.substr(0, comma), flag));
        if (comma == std::string_view::npos) { break; }
        text.remove_prefix(comma + 1);
    }
    if (values.empty()) { usage(flag); }
    return values;
}

Options parse_options(int argc, char** argv) {
    Options options;
    for (int index = 1; index < argc; ++index) {
        const std::string_view argument(argv[index]);
        const auto next = [&](const char* flag) -> std::string_view {
            if (++index == argc) { usage(flag); }
            return argv[index];
        };
        if (argument == "--context") {
            options.contexts = parse_list(next("--context requires a value"), "--context");
        } else if (argument == "--width") {
            options.widths = parse_list(next("--width requires a value"), "--width");
        } else if (argument == "--budget") {
            options.budgets = parse_list(next("--budget requires a value"), "--budget");
        } else if (argument == "--lanes") {
            options.lanes = parse_list(next("--lanes requires a value"), "--lanes");
        } else if (argument == "--kv") {
            const std::string_view value = next("--kv requires a value");
            if (value == "nvfp4") {
                options.format = QsaKvFormat::NVFP4G16;
            } else if (value == "bf16") {
                options.format = QsaKvFormat::BF16;
            } else {
                usage("--kv expects nvfp4 or bf16");
            }
        } else if (argument == "--op") {
            const std::string_view value = next("--op requires a value");
            options.select               = value == "select" || value == "all";
            options.attention            = value == "attention" || value == "all";
            if (!options.select && !options.attention) {
                usage("--op expects select, attention, or all");
            }
        } else if (argument == "--warmup") {
            options.warmup =
                bench::parse_number<int>(next("--warmup requires a value"), "--warmup");
        } else if (argument == "--repeat") {
            options.repeat =
                bench::parse_number<int>(next("--repeat requires a value"), "--repeat");
        } else {
            usage("unknown argument");
        }
    }
    for (const std::int32_t context : options.contexts) {
        if (context < kPageTokens || context > ops::kQsaMaximumCapacity ||
            context % kPageTokens != 0) {
            usage("--context values must be multiples of 64 in [64,262144]");
        }
    }
    for (const std::int32_t width : options.widths) {
        if (width < 1 || width > 65535) { usage("--width values must be in [1,65535]"); }
    }
    for (const std::int32_t lanes : options.lanes) {
        if (lanes < 1 || lanes > 12) { usage("--lanes values must be in [1,12]"); }
    }
    for (const std::int32_t budget : options.budgets) {
        if (budget != 512 && budget != 1024 && budget != 2048) {
            usage("--budget values must be 512, 1024, or 2048");
        }
    }
    if (options.warmup < 0 || options.repeat < 0) { usage("--warmup/--repeat must be >= 0"); }
    return options;
}

// Deterministic uniform BF16 values (SplitMix64).
class Generator {
public:
    explicit Generator(std::uint64_t seed) : state_(seed) {}

    // A value uniform in [-scale, scale).
    std::uint16_t next_bf16(float scale = 1.0F) {
        state_ += 0x9E3779B97F4A7C15ULL;
        std::uint64_t z = state_;
        z               = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
        z               = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
        z ^= z >> 31;
        const double unit = static_cast<double>(z >> 11) * 0x1.0p-53;
        return bench::f32_to_bf16(scale * static_cast<float>(2.0 * unit - 1.0));
    }

    // An index uniform in [0, bound).
    std::size_t next_index(std::size_t bound) {
        state_ += 0x9E3779B97F4A7C15ULL;
        std::uint64_t z = state_;
        z               = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
        z               = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
        z ^= z >> 31;
        return static_cast<std::size_t>(z % bound);
    }

    std::vector<std::uint16_t> bf16(std::size_t count) {
        std::vector<std::uint16_t> values(count);
        for (auto& value : values) { value = next_bf16(); }
        return values;
    }

private:
    std::uint64_t state_;
};

DeviceBuffer upload(const void* data, std::size_t bytes) {
    DeviceBuffer buffer(bytes);
    buffer.copy_from_host(data, bytes);
    return buffer;
}

template <typename T>
DeviceBuffer upload(const std::vector<T>& values) {
    return upload(values.data(), values.size() * sizeof(T));
}

// The paged state of `lanes` lanes of capacity C, its pages a shuffled permutation, filled
// through lane-batched qsa_state_append calls (row r appends lane r).
struct State {
    State(QsaKvFormat format_in, std::int32_t capacity_in, std::int32_t lanes_in,
          const DeviceBuffer& key_norm, Generator& generator)
        : format(format_in), capacity(capacity_in), lanes(lanes_in),
          logical_pages(capacity_in / kPageTokens), pages(lanes_in * logical_pages) {
        const auto pages_c   = static_cast<std::size_t>(pages);
        const auto page_c    = static_cast<std::size_t>(kPageTokens);
        const bool nvfp4     = format == QsaKvFormat::NVFP4G16;
        const std::size_t kv = (nvfp4 ? kHeadDim / 2 : kHeadDim * 2) * page_c * kKvHeads * pages_c;
        k                    = DeviceBuffer(kv);
        v                    = DeviceBuffer(kv);
        if (nvfp4) {
            k_scales =
                DeviceBuffer(static_cast<std::size_t>(kHeadDim / 16) * page_c * kKvHeads * pages_c);
            v_scales =
                DeviceBuffer(static_cast<std::size_t>(kHeadDim / 16) * page_c * kKvHeads * pages_c);
        }
        index_keys = DeviceBuffer(static_cast<std::size_t>(kIndexDim) * page_c * pages_c * 2);
        positions  = DeviceBuffer(3 * page_c * pages_c * sizeof(std::int32_t));
        block_keys = DeviceBuffer(page_c / kBlockTokens * kIndexDim * pages_c * sizeof(float));
        // Fisher-Yates shuffle of the physical pages with the deterministic generator.
        std::vector<std::int32_t> table(pages_c);
        for (std::size_t i = 0; i < pages_c; ++i) { table[i] = static_cast<std::int32_t>(i); }
        for (std::size_t i = pages_c - 1; i > 0; --i) {
            const std::size_t j = generator.next_index(i + 1);
            std::swap(table[i], table[j]);
        }
        block_tables = upload(table);
        const Tensor key_norm_t(key_norm.p, DType::BF16, {kIndexDim});

        // One chunk of K/V inputs is reused for every chunk; index keys differ per slot and lane.
        const std::int32_t chunk = std::min(capacity, kAppendChunk / lanes);
        const auto chunk_c       = static_cast<std::size_t>(chunk);
        const auto lanes_c       = static_cast<std::size_t>(lanes);
        const DeviceBuffer dk    = upload(generator.bf16(lanes_c * chunk_c * kKvHeads * kHeadDim));
        const DeviceBuffer dv    = upload(generator.bf16(lanes_c * chunk_c * kKvHeads * kHeadDim));
        DeviceBuffer dkeys(lanes_c * chunk_c * kIndexDim * 2);
        DeviceBuffer dpositions(lanes_c * 3 * chunk_c * sizeof(std::int32_t));
        DeviceBuffer dslots(lanes_c * chunk_c * sizeof(std::int32_t));
        std::vector<std::int32_t> host_lanes(lanes_c);
        for (std::size_t lane = 0; lane < lanes_c; ++lane) {
            host_lanes[lane] = static_cast<std::int32_t>(lane);
        }
        const DeviceBuffer dlanes = upload(host_lanes);
        std::vector<std::int32_t> host_positions(lanes_c * 3 * chunk_c);
        std::vector<std::int32_t> host_slots(lanes_c * chunk_c);
        QsaPagedState state = view();
        for (std::int32_t begin = 0; begin < capacity; begin += chunk) {
            const std::int32_t width = std::min(chunk, capacity - begin);
            const auto width_c       = static_cast<std::size_t>(width);
            const auto keys          = generator.bf16(lanes_c * width_c * kIndexDim);
            dkeys.copy_from_host(keys.data(), keys.size() * 2);
            for (std::size_t lane = 0; lane < lanes_c; ++lane) {
                for (std::int32_t t = 0; t < width; ++t) {
                    for (std::size_t axis = 0; axis < 3; ++axis) {
                        host_positions[(lane * 3 + axis) * width_c + static_cast<std::size_t>(t)] =
                            begin + t;
                    }
                    host_slots[lane * width_c + static_cast<std::size_t>(t)] = begin + t;
                }
            }
            dpositions.copy_from_host(host_positions.data(),
                                      lanes_c * 3 * width_c * sizeof(std::int32_t));
            dslots.copy_from_host(host_slots.data(), lanes_c * width_c * sizeof(std::int32_t));
            const Tensor k_t(dk.p, DType::BF16, {kHeadDim, kKvHeads, width, lanes});
            const Tensor v_t(dv.p, DType::BF16, {kHeadDim, kKvHeads, width, lanes});
            const Tensor keys_t(dkeys.p, DType::BF16, {kIndexDim, width, lanes});
            const Tensor positions_t(dpositions.p, DType::I32, {width, 3, lanes});
            const Tensor slots_t(dslots.p, DType::I32, {width, lanes});
            const Tensor lanes_t(dlanes.p, DType::I32, {lanes});
            ops::qsa_state_append(k_t, v_t, keys_t, positions_t, slots_t, lanes_t, key_norm_t,
                                  state, nullptr);
        }
        CUDA_CHECK(cudaDeviceSynchronize());
    }

    [[nodiscard]] QsaPagedState view() const {
        QsaPagedState state;
        state.format = format;
        if (format == QsaKvFormat::BF16) {
            state.k = Tensor(k.p, DType::BF16, {kHeadDim, kPageTokens, kKvHeads, pages});
            state.v = Tensor(v.p, DType::BF16, {kHeadDim, kPageTokens, kKvHeads, pages});
        } else {
            state.k        = Tensor(k.p, DType::U8, {kHeadDim / 2, kPageTokens, kKvHeads, pages});
            state.v        = Tensor(v.p, DType::U8, {kHeadDim / 2, kPageTokens, kKvHeads, pages});
            state.k_scales = Tensor(k_scales.p, DType::FP8_E4M3FN,
                                    {kHeadDim / 16, kPageTokens, kKvHeads, pages});
            state.v_scales = Tensor(v_scales.p, DType::FP8_E4M3FN,
                                    {kHeadDim / 16, kPageTokens, kKvHeads, pages});
        }
        state.index_keys = Tensor(index_keys.p, DType::BF16, {kIndexDim, kPageTokens, 1, pages});
        state.positions  = Tensor(positions.p, DType::I32, {1, kPageTokens, 3, pages});
        state.block_keys =
            Tensor(block_keys.p, DType::FP32,
                   {ops::kQsaPageBlocks * kIndexDim / kPageTokens, kPageTokens, 1, pages});
        state.block_tables = Tensor(block_tables.p, DType::I32, {logical_pages, lanes});
        return state;
    }

    // Bytes of one token's K and V rows of one KV head.
    [[nodiscard]] std::int32_t kv_row_bytes() const {
        return format == QsaKvFormat::BF16 ? 2 * kHeadDim * 2 : 2 * (kHeadDim / 2 + kHeadDim / 16);
    }

    QsaKvFormat format;
    std::int32_t capacity;
    std::int32_t lanes;
    std::int32_t logical_pages;
    std::int32_t pages;
    DeviceBuffer k, v, k_scales, v_scales, index_keys, positions, block_keys, block_tables;
};

// Bytes the selection must read per complete visible block under the state contract: the block's
// sealed FP32 key.
constexpr double kSelectBytesPerBlock = kIndexDim * 4.0;

struct Measurement {
    bench::ColdTiming timing;
    double bytes = 0.0;
    double flops = 0.0;
};

// Graph capture needs a non-default stream.
class Stream {
public:
    Stream() { CUDA_CHECK(cudaStreamCreateWithFlags(&stream_, cudaStreamNonBlocking)); }

    ~Stream() { cudaStreamDestroy(stream_); }

    Stream(const Stream&)            = delete;
    Stream& operator=(const Stream&) = delete;

    [[nodiscard]] cudaStream_t get() const noexcept { return stream_; }

private:
    cudaStream_t stream_ = nullptr;
};

int auto_repeat(const Options& options, const bench::TimedGraph& graph, DeviceBuffer& flush,
                cudaStream_t stream) {
    if (options.repeat > 0) { return options.repeat; }
    const bench::ColdTiming probe = bench::measure_cold_graph(graph, flush, stream, 0, 1);
    const auto fit                = static_cast<int>(kAutoRepeatBudgetUs / probe.median_us);
    return std::clamp(fit, kAutoRepeatMinimum, kAutoRepeatMaximum);
}

void print_row(const char* op, const State& state, std::int32_t budget, std::int32_t width,
               const Measurement& m) {
    const double seconds = m.timing.median_us * 1.0e-6;
    const double gbs     = m.bytes / seconds / 1.0e9;
    std::printf("%-9s %-5s budget=%4d C=%6d B=%2d W=%4d  median=%10.2f us  min=%10.2f  "
                "p95=%10.2f  %7.1f GB/s (%5.1f%% roof)  %6.2f TFLOP/s\n",
                op, state.format == QsaKvFormat::BF16 ? "bf16" : "nvfp4", budget, state.capacity,
                state.lanes, width, m.timing.median_us, m.timing.min_us, m.timing.p95_us, gbs,
                100.0 * gbs / kDramRoofGBs, m.flops / seconds / 1.0e12);
}

void run_case(const Options& options, const State& state, std::int32_t budget_blocks,
              std::int32_t width, const DeviceBuffer& query_norm, Generator& generator,
              DeviceBuffer& flush, cudaStream_t stream) {
    const auto budget          = static_cast<QsaSelectBudget>(budget_blocks);
    const std::int32_t context = state.capacity;
    const std::int32_t slots   = ops::qsa_selected_capacity(budget);
    const std::int32_t rows    = state.lanes;
    const std::int32_t queries = width * rows;
    const auto queries_c       = static_cast<std::size_t>(queries);

    std::vector<std::int32_t> host_slots(queries_c);
    std::vector<std::int32_t> host_lanes(static_cast<std::size_t>(rows));
    for (std::int32_t row = 0; row < rows; ++row) {
        host_lanes[static_cast<std::size_t>(row)] = row;
        for (std::int32_t t = 0; t < width; ++t) {
            host_slots[static_cast<std::size_t>(row) * width + static_cast<std::size_t>(t)] =
                context - width + t;
        }
    }
    const DeviceBuffer dslots = upload(host_slots);
    const DeviceBuffer dlanes = upload(host_lanes);
    const DeviceBuffer dindex = upload(generator.bf16(queries_c * kIndexHeads * kIndexDim));
    const DeviceBuffer dq     = upload(generator.bf16(queries_c * kQueryHeads * kHeadDim));
    DeviceBuffer dselected(static_cast<std::size_t>(slots) * queries_c * sizeof(std::int32_t));
    DeviceBuffer dcount(queries_c * sizeof(std::int32_t));
    DeviceBuffer dout(queries_c * kQueryHeads * kHeadDim * 2);

    const Tensor index_t(dindex.p, DType::BF16, {kIndexDim, kIndexHeads, width, rows});
    const Tensor slots_t(dslots.p, DType::I32, {width, rows});
    const Tensor lanes_t(dlanes.p, DType::I32, {rows});
    const Tensor qnorm_t(query_norm.p, DType::BF16, {kIndexDim});
    Tensor selected_t(dselected.p, DType::I32, {slots, width, rows});
    Tensor count_t(dcount.p, DType::I32, {width, rows});
    const Tensor q_t(dq.p, DType::BF16, {kHeadDim, kQueryHeads, width, rows});
    Tensor out_t(dout.p, DType::BF16, {kHeadDim, kQueryHeads, width, rows});
    const QsaPagedState view = state.view();

    WorkspaceArena workspace(std::max<std::size_t>(
        {ops::qsa_index_select_workspace_bytes(queries, context, budget),
         ops::qsa_selected_attention_workspace_bytes(queries, slots), std::size_t{256}}));
    const auto select = [&](cudaStream_t target) {
        ops::qsa_index_select(index_t, view, slots_t, lanes_t, qnorm_t, budget, context, selected_t,
                              count_t, workspace, target);
    };
    const auto attention = [&](cudaStream_t target) {
        ops::qsa_selected_attention(q_t, selected_t, count_t, lanes_t, view, out_t, workspace,
                                    target);
    };
    select(stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));
    std::vector<std::int32_t> counts(queries_c);
    dcount.copy_to_host(counts.data(), queries_c * sizeof(std::int32_t));

    if (options.select) {
        bench::TimedGraph graph;
        graph.capture(stream, select);
        Measurement m;
        const int repeat = auto_repeat(options, graph, flush, stream);
        m.timing         = bench::measure_cold_graph(graph, flush, stream, options.warmup, repeat);
        // Block keys are read only when some query ranks: up to the longest prefix's blocks of
        // every lane.
        std::int32_t ranked_blocks = 0;
        for (const std::int32_t slot : host_slots) {
            const std::int32_t complete = (slot + 1) / kBlockTokens;
            if (complete > budget_blocks) {
                ranked_blocks = std::max(ranked_blocks, complete);
                m.flops += 2.0 * kIndexHeads * kIndexDim * static_cast<double>(complete);
            }
        }
        m.bytes =
            static_cast<double>(rows) * static_cast<double>(ranked_blocks) * kSelectBytesPerBlock +
            static_cast<double>(queries_c) *
                (kIndexHeads * kIndexDim * 2.0 + 4.0 + static_cast<double>(slots) * 4.0 + 4.0);
        print_row("select", state, budget_blocks, width, m);
    }
    if (options.attention) {
        bench::TimedGraph graph;
        graph.capture(stream, attention);
        Measurement m;
        const int repeat = auto_repeat(options, graph, flush, stream);
        m.timing         = bench::measure_cold_graph(graph, flush, stream, options.warmup, repeat);
        double listed    = 0.0;
        for (const std::int32_t count : counts) { listed += static_cast<double>(count); }
        m.bytes = listed * kKvHeads * state.kv_row_bytes() +
                  static_cast<double>(queries_c) *
                      (2.0 * kQueryHeads * kHeadDim * 2.0 + static_cast<double>(slots) * 4.0 + 4.0);
        m.flops = listed * kQueryHeads * 4.0 * kHeadDim;
        print_row("attention", state, budget_blocks, width, m);
    }
}

} // namespace

int main(int argc, char** argv) {
    try {
        const Options options = parse_options(argc, argv);
        bench::print_device_caps("ninfer_qsa_bench");
        Generator generator(0x5153414265ULL);
        // Zero-centered norm weights of the magnitude the checkpoint carries.
        std::vector<std::uint16_t> norms(2 * kIndexDim);
        for (auto& bits : norms) { bits = generator.next_bf16(0.25F); }
        const DeviceBuffer query_norm = upload(norms.data(), kIndexDim * 2);
        const DeviceBuffer key_norm   = upload(norms.data() + kIndexDim, kIndexDim * 2);
        DeviceBuffer flush(kFlushBytes);
        const Stream stream;
        for (const std::int32_t lanes : options.lanes) {
            for (const std::int32_t context : options.contexts) {
                const State state(options.format, context, lanes, key_norm, generator);
                for (const std::int32_t budget : options.budgets) {
                    for (const std::int32_t width : options.widths) {
                        if (width > context || width * lanes > 65535) { continue; }
                        run_case(options, state, budget, width, query_norm, generator, flush,
                                 stream.get());
                    }
                }
            }
        }
        return 0;
    } catch (const std::exception& error) {
        std::fprintf(stderr, "ninfer_qsa_bench: %s\n", error.what());
        return 1;
    }
}
