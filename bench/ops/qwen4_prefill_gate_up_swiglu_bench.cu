#include "ninfer/ops/qwen4_sparse_moe.h"
#include "ninfer_bench_common.h"

#include <cuda_profiler_api.h>

#include <charconv>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

using namespace ninfer;

namespace {

constexpr std::int32_t kRows = ops::kQwen4SparseMoeIntermediate;
constexpr std::int32_t kColumns = ops::kQwen4SparseMoeHidden;

struct Format {
    const char* name;
    QType qtype;
    std::int32_t block_bytes;
};

constexpr Format kIq1{"ggml_iq1_s", QType::GGML_IQ1_S, 50};
constexpr Format kIq2{"ggml_iq2_xxs", QType::GGML_IQ2_XXS, 66};

struct Options {
    const Format* format = nullptr;
    std::int32_t tokens = 16;
    int warmup = 20;
    int repetitions = 200;
    bool profile = false;
};

[[noreturn]] void usage(const char* program, const char* error = nullptr) {
    if (error != nullptr) { std::fprintf(stderr, "%s\n", error); }
    std::fprintf(error == nullptr ? stdout : stderr,
                 "usage: %s --qtype <ggml_iq1_s|ggml_iq2_xxs> --t T "
                 "[--warmup N] [--repetitions N] [--profile]\n",
                 program);
    std::exit(error == nullptr ? 0 : 2);
}

int parse_int(std::string_view text, const char* label) {
    int value = 0;
    const auto result = std::from_chars(text.data(), text.data() + text.size(), value);
    if (result.ec != std::errc{} || result.ptr != text.data() + text.size()) {
        throw std::invalid_argument(std::string("invalid ") + label);
    }
    return value;
}

Options parse_options(int argc, char** argv) {
    Options options;
    for (int index = 1; index < argc; ++index) {
        const std::string_view argument(argv[index]);
        auto next = [&](const char* label) {
            if (++index >= argc) { throw std::invalid_argument(std::string("missing ") + label); }
            return std::string_view(argv[index]);
        };
        if (argument == "--qtype") {
            const std::string_view value = next("qtype");
            if (value == kIq1.name) {
                options.format = &kIq1;
            } else if (value == kIq2.name) {
                options.format = &kIq2;
            } else {
                throw std::invalid_argument("unsupported qtype");
            }
        } else if (argument == "--t") {
            options.tokens = parse_int(next("T"), "T");
        } else if (argument == "--warmup") {
            options.warmup = parse_int(next("warmup"), "warmup");
        } else if (argument == "--repetitions") {
            options.repetitions = parse_int(next("repetitions"), "repetitions");
        } else if (argument == "--profile") {
            options.profile = true;
        } else if (argument == "--help" || argument == "-h") {
            usage(argc > 0 ? argv[0] : "ninfer_qwen4_prefill_gate_up_swiglu_bench");
        } else {
            throw std::invalid_argument(std::string("unknown argument: ") + argv[index]);
        }
    }
    if (options.format == nullptr) { throw std::invalid_argument("--qtype is required"); }
    if (options.tokens <= 0 || options.tokens > 4096) {
        throw std::invalid_argument("T must be in [1,4096]");
    }
    if (options.warmup < 0 || options.repetitions <= 0) {
        throw std::invalid_argument("warmup must be nonnegative and repetitions positive");
    }
    return options;
}

void write_u16_le(std::uint8_t* destination, std::uint16_t value) {
    destination[0] = static_cast<std::uint8_t>(value & 0xffU);
    destination[1] = static_cast<std::uint8_t>(value >> 8U);
}

std::vector<std::uint8_t> make_payload(const Format& format, std::size_t bytes,
                                       std::uint32_t seed) {
    std::vector<std::uint8_t> payload(bytes);
    std::uint32_t state = seed;
    for (std::uint8_t& value : payload) {
        state = state * 1664525U + 1013904223U;
        value = static_cast<std::uint8_t>(state >> 24U);
    }
    for (std::size_t offset = 0; offset < payload.size(); offset += format.block_bytes) {
        write_u16_le(payload.data() + offset, 0x2000U);
    }
    return payload;
}

Weight make_weight(void* data, std::size_t bytes, const Format& format) {
    Weight weight{};
    weight.payload = data;
    weight.payload_bytes = bytes;
    weight.qdata = data;
    weight.qtype = format.qtype;
    weight.group_size = 256;
    weight.group = 256;
    weight.layout = QuantLayout::GgmlBlockRow;
    weight.ndim = 2;
    weight.n = kRows;
    weight.k = kColumns;
    weight.shape[0] = kRows;
    weight.shape[1] = kColumns;
    weight.padded_shape[0] = kRows;
    weight.padded_shape[1] = kColumns;
    return weight;
}

void print_timing(const char* cache, const char* route, const bench::ColdTiming& timing) {
    std::printf("cache=%s route=%s median_us=%.3f min_us=%.3f p95_us=%.3f\n",
                cache, route, timing.median_us, timing.min_us, timing.p95_us);
}

int run(const Options& options) {
    const Format& format = *options.format;
    const std::size_t row_bytes =
        static_cast<std::size_t>(kColumns / 256) * format.block_bytes;
    const std::size_t matrix_bytes = static_cast<std::size_t>(kRows) * row_bytes;
    const std::size_t input_elements =
        static_cast<std::size_t>(kColumns) * options.tokens;
    const std::size_t output_elements =
        static_cast<std::size_t>(kRows) * options.tokens;

    DeviceBuffer gate(matrix_bytes);
    DeviceBuffer up(matrix_bytes);
    const auto host_gate = make_payload(format, matrix_bytes, 0x91e10da5U);
    const auto host_up = make_payload(format, matrix_bytes, 0x4f1bbcdcU);
    gate.copy_from_host(host_gate.data(), host_gate.size());
    up.copy_from_host(host_up.data(), host_up.size());
    DeviceBuffer input = bench::make_bf16(input_elements);
    DeviceBuffer output(output_elements * sizeof(std::uint16_t));
    DeviceBuffer l2_flush(256U * 1024U * 1024U);

    Tensor x(input.p, DType::BF16, {kColumns, options.tokens});
    Tensor output_tensor(output.p, DType::BF16, {kRows, options.tokens});
    const Weight gate_weight = make_weight(gate.p, matrix_bytes, format);
    const Weight up_weight = make_weight(up.p, matrix_bytes, format);

    cudaStream_t stream = nullptr;
    CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    const auto launch = [&](cudaStream_t launch_stream) {
        ops::qwen4_sparse_moe_gate_up_swiglu(
            x, gate_weight, up_weight, output_tensor, launch_stream);
    };

    bench::print_device_caps("qwen4-prefill-gate-up-swiglu");
    std::printf("qtype=%s T=%d warmup=%d repetitions=%d\n",
                format.name, options.tokens, options.warmup, options.repetitions);
    if (options.profile) {
        bench::flush_l2(l2_flush, stream);
        CUDA_CHECK(cudaStreamSynchronize(stream));
        CUDA_CHECK(cudaProfilerStart());
        launch(stream);
        CUDA_CHECK(cudaStreamSynchronize(stream));
        CUDA_CHECK(cudaProfilerStop());
        std::printf("profile=true\n");
    } else {
        print_timing("warm", "production",
                     bench::measure_launch(launch, stream, options.warmup,
                                           options.repetitions));
        print_timing("cold", "production",
                     bench::measure_cold_launch(launch, l2_flush, stream,
                                                options.warmup, options.repetitions));
    }
    CUDA_CHECK(cudaStreamDestroy(stream));
    return 0;
}

} // namespace

int main(int argc, char** argv) {
    try {
        int device_count = 0;
        if (cudaGetDeviceCount(&device_count) != cudaSuccess || device_count == 0) {
            std::printf("SKIP: no usable CUDA device\n");
            return 0;
        }
        return run(parse_options(argc, argv));
    } catch (const std::exception& error) {
        usage(argc > 0 ? argv[0] : "ninfer_qwen4_prefill_gate_up_swiglu_bench",
              error.what());
    }
}
