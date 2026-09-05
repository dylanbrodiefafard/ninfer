#include "ninfer/ops/ggml_block_linear.h"
#include "ninfer_bench_common.h"

#include <cuda_profiler_api.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <charconv>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

using namespace ninfer;

namespace {

struct Format {
    const char* name;
    QType qtype;
    std::int32_t block_values;
    std::int32_t block_bytes;
    std::int32_t scale_offset;
    bool has_second_scale;
};

constexpr Format kFormats[] = {
    {"ggml_q8_0", QType::GGML_Q8_0, 32, 34, 0, false},
    {"ggml_q4_k", QType::GGML_Q4_K, 256, 144, 0, true},
    {"ggml_q5_k", QType::GGML_Q5_K, 256, 176, 0, true},
    {"ggml_q6_k", QType::GGML_Q6_K, 256, 210, 208, false},
    {"ggml_iq1_s", QType::GGML_IQ1_S, 256, 50, 0, false},
    {"ggml_iq2_xxs", QType::GGML_IQ2_XXS, 256, 66, 0, false},
    {"ggml_iq4_nl", QType::GGML_IQ4_NL, 32, 18, 0, false},
};

struct Options {
    const Format* format = nullptr;
    std::int32_t n = 0;
    std::int32_t k = 0;
    std::int32_t t = 0;
    int warmup = 3;
    int repetitions = 20;
    bool profile = false;
};

[[noreturn]] void usage(const char* program, const char* error = nullptr) {
    if (error != nullptr) { std::fprintf(stderr, "%s\n", error); }
    std::fprintf(
        error == nullptr ? stdout : stderr,
        "usage: %s --qtype <ggml_q8_0|ggml_q4_k|ggml_q5_k|ggml_q6_k|"
        "ggml_iq1_s|ggml_iq2_xxs|ggml_iq4_nl> --n N --k K --t T "
        "[--warmup N] [--repetitions N] [--profile]\n",
        program);
    std::exit(error == nullptr ? 0 : 2);
}

template <class Integer>
Integer parse_integer(std::string_view text, const char* name) {
    Integer value{};
    const char* begin = text.data();
    const char* end = begin + text.size();
    const auto result = std::from_chars(begin, end, value);
    if (result.ec != std::errc{} || result.ptr != end) {
        throw std::invalid_argument(std::string("invalid ") + name);
    }
    return value;
}

const Format& parse_format(std::string_view name) {
    for (const Format& format : kFormats) {
        if (name == format.name) { return format; }
    }
    throw std::invalid_argument("unsupported --qtype");
}

Options parse_options(int argc, char** argv) {
    Options options;
    bool have_n = false;
    bool have_k = false;
    bool have_t = false;
    for (int index = 1; index < argc; ++index) {
        const std::string_view argument(argv[index]);
        auto next = [&](const char* name) -> std::string_view {
            if (++index >= argc) { throw std::invalid_argument(std::string("missing ") + name); }
            return argv[index];
        };
        if (argument == "--qtype") {
            options.format = &parse_format(next("qtype"));
        } else if (argument == "--n") {
            options.n = parse_integer<std::int32_t>(next("N"), "N");
            have_n = true;
        } else if (argument == "--k") {
            options.k = parse_integer<std::int32_t>(next("K"), "K");
            have_k = true;
        } else if (argument == "--t") {
            options.t = parse_integer<std::int32_t>(next("T"), "T");
            have_t = true;
        } else if (argument == "--warmup") {
            options.warmup = parse_integer<int>(next("warmup"), "warmup");
        } else if (argument == "--repetitions") {
            options.repetitions =
                parse_integer<int>(next("repetitions"), "repetitions");
        } else if (argument == "--profile") {
            options.profile = true;
        } else if (argument == "--help" || argument == "-h") {
            usage(argc > 0 ? argv[0] : "ninfer_ggml_block_linear_bench");
        } else {
            throw std::invalid_argument(std::string("unknown argument: ") + argv[index]);
        }
    }
    if (options.format == nullptr || !have_n || !have_k || !have_t) {
        throw std::invalid_argument("--qtype, --n, --k, and --t are required");
    }
    if (options.n <= 0 || options.n > 248320 || options.k <= 0 || options.k > 10240 ||
        options.t <= 0 || options.t > 4096) {
        throw std::invalid_argument("N, K, and T must be within the public Op domain");
    }
    if (options.k % options.format->block_values != 0) {
        throw std::invalid_argument("K is not divisible by the selected GGML block width");
    }
    if (options.warmup < 0 || options.repetitions <= 0) {
        throw std::invalid_argument("warmup must be nonnegative and repetitions positive");
    }
    if (options.profile && options.repetitions != 1) {
        throw std::invalid_argument("--profile requires --repetitions 1");
    }
    return options;
}

std::uint64_t checked_mul(std::uint64_t lhs, std::uint64_t rhs, const char* label) {
    if (lhs != 0 && rhs > std::numeric_limits<std::uint64_t>::max() / lhs) {
        throw std::overflow_error(std::string(label) + " overflows");
    }
    return lhs * rhs;
}

std::uint64_t checked_add(std::uint64_t lhs, std::uint64_t rhs, const char* label) {
    if (rhs > std::numeric_limits<std::uint64_t>::max() - lhs) {
        throw std::overflow_error(std::string(label) + " overflows");
    }
    return lhs + rhs;
}

std::size_t checked_size(std::uint64_t bytes, const char* label) {
    if (bytes > std::numeric_limits<std::size_t>::max()) {
        throw std::overflow_error(std::string(label) + " does not fit size_t");
    }
    return static_cast<std::size_t>(bytes);
}

void write_u16_le(std::uint8_t* destination, std::uint16_t value) {
    destination[0] = static_cast<std::uint8_t>(value & 0xffU);
    destination[1] = static_cast<std::uint8_t>(value >> 8U);
}

std::vector<std::uint8_t> make_payload(const Format& format, std::uint64_t weight_bytes) {
    std::vector<std::uint8_t> payload(checked_size(weight_bytes, "weight payload"));
    std::uint32_t state = 0x91e10da5U ^ static_cast<std::uint32_t>(format.qtype);
    for (std::uint8_t& byte : payload) {
        state = state * 1664525U + 1013904223U;
        byte = static_cast<std::uint8_t>(state >> 24U);
    }
    for (std::size_t offset = 0; offset < payload.size(); offset += format.block_bytes) {
        // Exact finite FP16 scales. Remaining bytes are unrestricted packed code/control planes.
        write_u16_le(payload.data() + offset + format.scale_offset, 0x2000U);
        if (format.has_second_scale) { write_u16_le(payload.data() + offset + 2, 0x1800U); }
    }
    return payload;
}

Weight make_weight(const void* payload, std::uint64_t payload_bytes, const Format& format,
                   std::int32_t n, std::int32_t k) {
    Weight weight{};
    weight.payload = payload;
    weight.payload_bytes = payload_bytes;
    weight.qdata = payload;
    weight.qtype = format.qtype;
    weight.group_size = static_cast<std::uint32_t>(format.block_values);
    weight.group = format.block_values;
    weight.layout = QuantLayout::GgmlBlockRow;
    weight.ndim = 2;
    weight.n = n;
    weight.k = k;
    weight.shape[0] = n;
    weight.shape[1] = k;
    weight.padded_shape[0] = n;
    weight.padded_shape[1] = k;
    return weight;
}

struct Event {
    cudaEvent_t value = nullptr;
    Event() { CUDA_CHECK(cudaEventCreate(&value)); }
    ~Event() {
        if (value != nullptr) { cudaEventDestroy(value); }
    }
    Event(const Event&) = delete;
    Event& operator=(const Event&) = delete;
};

struct Stream {
    cudaStream_t value = nullptr;
    Stream() { CUDA_CHECK(cudaStreamCreateWithFlags(&value, cudaStreamNonBlocking)); }
    ~Stream() {
        if (value != nullptr) { cudaStreamDestroy(value); }
    }
    Stream(const Stream&) = delete;
    Stream& operator=(const Stream&) = delete;
};

struct Timing {
    double median_us;
    double min_us;
    double p95_us;
    double max_us;
};

Timing measure(const Options& options, const Tensor& x, const Weight& weight, Tensor& output,
               DeviceBuffer& l2_flush, cudaStream_t stream) {
    const auto launch = [&] { ops::ggml_block_linear(x, weight, output, stream); };
    for (int index = 0; index < options.warmup; ++index) {
        bench::flush_l2(l2_flush, stream);
        launch();
    }
    CUDA_CHECK(cudaStreamSynchronize(stream));

    Event start;
    Event stop;
    std::vector<double> samples;
    samples.reserve(static_cast<std::size_t>(options.repetitions));
    for (int index = 0; index < options.repetitions; ++index) {
        bench::flush_l2(l2_flush, stream);
        if (options.profile) {
            CUDA_CHECK(cudaStreamSynchronize(stream));
            CUDA_CHECK(cudaProfilerStart());
        }
        CUDA_CHECK(cudaEventRecord(start.value, stream));
        launch();
        CUDA_CHECK(cudaEventRecord(stop.value, stream));
        CUDA_CHECK(cudaEventSynchronize(stop.value));
        if (options.profile) { CUDA_CHECK(cudaProfilerStop()); }
        float milliseconds = 0.0F;
        CUDA_CHECK(cudaEventElapsedTime(&milliseconds, start.value, stop.value));
        samples.push_back(static_cast<double>(milliseconds) * 1000.0);
    }
    std::sort(samples.begin(), samples.end());
    const std::size_t p95_index = (95 * samples.size() + 99) / 100 - 1;
    return {samples[samples.size() / 2], samples.front(), samples[p95_index], samples.back()};
}

int run(const Options& options) {
    const Format& format = *options.format;
    const std::uint64_t blocks_per_row =
        static_cast<std::uint64_t>(options.k / format.block_values);
    const std::uint64_t row_bytes =
        checked_mul(blocks_per_row, static_cast<std::uint64_t>(format.block_bytes), "row bytes");
    const std::uint64_t weight_bytes =
        checked_mul(static_cast<std::uint64_t>(options.n), row_bytes, "weight bytes");
    const std::uint64_t input_bytes =
        checked_mul(checked_mul(static_cast<std::uint64_t>(options.k), options.t,
                                "input elements"),
                    sizeof(std::uint16_t), "input bytes");
    const std::uint64_t output_bytes =
        checked_mul(checked_mul(static_cast<std::uint64_t>(options.n), options.t,
                                "output elements"),
                    sizeof(std::uint16_t), "output bytes");
    const std::uint64_t model_bytes =
        checked_add(checked_add(weight_bytes, input_bytes, "model bytes"), output_bytes,
                    "model bytes");
    DeviceBuffer device_payload(checked_size(weight_bytes, "weight bytes"));
    {
        const std::vector<std::uint8_t> host_payload = make_payload(format, weight_bytes);
        device_payload.copy_from_host(host_payload.data(), host_payload.size());
    }
    DeviceBuffer input = bench::make_bf16(
        checked_size(input_bytes / sizeof(std::uint16_t), "input elements"));
    DeviceBuffer output(checked_size(output_bytes, "output bytes"));
    output.fill();
    DeviceBuffer l2_flush(256U * 1024U * 1024U);

    Tensor input_tensor(input.p, DType::BF16, {options.k, options.t});
    Tensor output_tensor(output.p, DType::BF16, {options.n, options.t});
    const Weight weight =
        make_weight(device_payload.p, weight_bytes, format, options.n, options.k);
    Stream stream;
    const Timing timing =
        measure(options, input_tensor, weight, output_tensor, l2_flush, stream.value);
    const double effective_gbs =
        static_cast<double>(model_bytes) / (timing.median_us * 1.0e3);

    bench::print_device_caps("ggml-block-linear");
    std::printf("qtype=%s N=%d K=%d T=%d warmup=%d repetitions=%d cache=cold profile=%s\n",
                format.name, options.n, options.k, options.t, options.warmup,
                options.repetitions, options.profile ? "true" : "false");
    std::printf("median_us=%.3f min_us=%.3f p95_us=%.3f max_us=%.3f\n", timing.median_us,
                timing.min_us, timing.p95_us, timing.max_us);
    std::printf("weight_bytes=%llu input_bytes=%llu output_bytes=%llu model_bytes=%llu\n",
                static_cast<unsigned long long>(weight_bytes),
                static_cast<unsigned long long>(input_bytes),
                static_cast<unsigned long long>(output_bytes),
                static_cast<unsigned long long>(model_bytes));
    std::printf("effective_model_GB_s=%.3f\n", effective_gbs);
    return 0;
}

} // namespace

int main(int argc, char** argv) {
    try {
        const Options options = parse_options(argc, argv);
        int device_count = 0;
        if (cudaGetDeviceCount(&device_count) != cudaSuccess || device_count == 0) {
            std::printf("SKIP: no usable CUDA device\n");
            return 0;
        }
        return run(options);
    } catch (const std::exception& error) {
        usage(argc > 0 ? argv[0] : "ninfer_ggml_block_linear_bench", error.what());
    }
}
