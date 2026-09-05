#include "ninfer/ops/qwen4_sparse_moe.h"

#include "core/arena.h"
#include "core/device.h"

#include <cuda_runtime.h>
#include <cuda_profiler_api.h>

#include <array>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string_view>
#include <utility>
#include <vector>

using namespace ninfer;

namespace {

constexpr std::int32_t kHidden = ops::kQwen4SparseMoeHidden;
constexpr std::int32_t kExperts = ops::kQwen4SparseMoeExperts;
constexpr std::int32_t kTopK = ops::kQwen4SparseMoeTopK;
constexpr std::int32_t kIntermediate = ops::kQwen4SparseMoeIntermediate;
constexpr std::int32_t kRouteWindows = 52;

struct FormatSpec {
    std::size_t values;
    std::size_t bytes;
};

FormatSpec format_spec(QType qtype) {
    switch (qtype) {
    case QType::GGML_Q8_0:
        return {32, 34};
    case QType::GGML_Q5_K:
        return {256, 176};
    case QType::GGML_Q6_K:
        return {256, 210};
    case QType::GGML_IQ1_S:
        return {256, 50};
    case QType::GGML_IQ2_XXS:
        return {256, 66};
    case QType::GGML_IQ4_NL:
        return {32, 18};
    default:
        throw std::invalid_argument("unsupported benchmark format");
    }
}

std::size_t matrix_bytes(QType qtype, std::int32_t rows, std::int32_t columns) {
    const FormatSpec spec = format_spec(qtype);
    return static_cast<std::size_t>(rows) * static_cast<std::size_t>(columns) / spec.values *
           spec.bytes;
}

void set_half_scale(std::uint8_t* bytes) {
    // Exact little-endian binary16 2^-10. Small finite scales keep arbitrary payloads finite.
    bytes[0] = 0x00;
    bytes[1] = 0x14;
}

void fill_valid_blocks(std::vector<std::uint8_t>& payload, QType qtype,
                       std::uint32_t seed) {
    std::uint32_t state = seed;
    for (std::uint8_t& byte : payload) {
        state ^= state << 13U;
        state ^= state >> 17U;
        state ^= state << 5U;
        byte = static_cast<std::uint8_t>(state);
    }
    const FormatSpec spec = format_spec(qtype);
    for (std::size_t offset = 0; offset < payload.size(); offset += spec.bytes) {
        auto* block = payload.data() + offset;
        if (qtype == QType::GGML_Q6_K) {
            set_half_scale(block + 208);
        } else {
            set_half_scale(block);
            if (qtype == QType::GGML_Q5_K) { set_half_scale(block + 2); }
        }
    }
}

Weight make_ggml_weight(void* data, std::size_t bytes, QType qtype, std::int32_t matrices,
                        std::int32_t rows, std::int32_t columns) {
    const FormatSpec spec = format_spec(qtype);
    Weight weight{};
    weight.payload = data;
    weight.payload_bytes = bytes;
    weight.qdata = data;
    weight.qtype = qtype;
    weight.group_size = spec.values;
    weight.group = static_cast<std::int32_t>(spec.values);
    weight.layout = QuantLayout::GgmlBlockRow;
    weight.n = rows;
    weight.k = columns;
    weight.ndim = matrices == 1 ? 2U : 3U;
    if (matrices == 1) {
        weight.shape[0] = rows;
        weight.shape[1] = columns;
        weight.padded_shape[0] = rows;
        weight.padded_shape[1] = columns;
    } else {
        weight.shape[0] = matrices;
        weight.shape[1] = rows;
        weight.shape[2] = columns;
        weight.padded_shape[0] = matrices;
        weight.padded_shape[1] = rows;
        weight.padded_shape[2] = columns;
    }
    return weight;
}

Weight make_router(void* data) {
    Weight weight{};
    weight.payload = data;
    weight.payload_bytes = static_cast<std::uint64_t>(kExperts) * kHidden * sizeof(float);
    weight.qdata = data;
    weight.qtype = QType::FP32_CTRL;
    weight.layout = QuantLayout::Contiguous;
    weight.n = kExperts;
    weight.k = kHidden;
    weight.ndim = 2;
    weight.shape[0] = kExperts;
    weight.shape[1] = kHidden;
    weight.padded_shape[0] = kExperts;
    weight.padded_shape[1] = kHidden;
    return weight;
}

struct Measurement {
    double wall_us = 0.0;
    double gpu_us = 0.0;
};

class Fixture {
public:
    Fixture(QType routed_qtype, QType shared_qtype, std::int32_t width)
        : routed_qtype_(routed_qtype), width_(width), shared_qtype_(shared_qtype),
          one_routed_(matrix_bytes(routed_qtype_, kIntermediate, kHidden)),
          routed_bank_bytes_(static_cast<std::size_t>(kExperts) * one_routed_),
          down_bank_bytes_(static_cast<std::size_t>(kExperts) *
                           matrix_bytes(QType::GGML_IQ4_NL, kHidden, kIntermediate)),
          host_gate_(routed_bank_bytes_), host_up_(routed_bank_bytes_),
          host_down_(down_bank_bytes_),
          host_shared_gate_proj_(matrix_bytes(shared_qtype_, kIntermediate, kHidden)),
          host_shared_up_(matrix_bytes(shared_qtype_, kIntermediate, kHidden)),
          host_shared_down_(matrix_bytes(QType::GGML_Q8_0, kHidden, kIntermediate)),
          device_inputs_(static_cast<std::size_t>(width_) * kHidden * sizeof(std::uint16_t)),
          device_router_(static_cast<std::size_t>(kExperts) * kHidden * sizeof(float)),
          device_gate_(routed_bank_bytes_), device_up_(routed_bank_bytes_),
          device_down_(down_bank_bytes_), device_shared_gate_(kHidden * sizeof(float)),
          device_shared_gate_proj_(host_shared_gate_proj_.size()),
          device_shared_up_(host_shared_up_.size()), device_shared_down_(host_shared_down_.size()),
          selected_ids_(static_cast<std::size_t>(width_) * kTopK * sizeof(std::int32_t)),
          selected_weights_(static_cast<std::size_t>(width_) * kTopK * sizeof(float)),
          destination_(static_cast<std::size_t>(width_) * kHidden * sizeof(std::uint16_t)),
          workspace_(ops::qwen4_sparse_moe_resident_workspace_capacity_bytes(width_)) {
        fill_valid_blocks(host_gate_, routed_qtype_, 0x12345678U);
        fill_valid_blocks(host_up_, routed_qtype_, 0x87654321U);
        fill_valid_blocks(host_down_, QType::GGML_IQ4_NL, 0x31415926U);
        fill_valid_blocks(host_shared_gate_proj_, shared_qtype_, 0x27182818U);
        fill_valid_blocks(host_shared_up_, shared_qtype_, 0x16180339U);
        fill_valid_blocks(host_shared_down_, QType::GGML_Q8_0, 0x42424242U);

        std::vector<float> router(static_cast<std::size_t>(kExperts) * kHidden, 0.0F);
        for (std::int32_t window = 0; window < kRouteWindows; ++window) {
            for (std::int32_t rank = 0; rank < kTopK; ++rank) {
                const std::int32_t expert = (window * kTopK + rank) % kExperts;
                router[static_cast<std::size_t>(expert) * kHidden + window] =
                    static_cast<float>(20 - rank);
            }
        }
        device_router_.copy_from_host(router.data(), device_router_.bytes);
        device_gate_.copy_from_host(host_gate_.data(), host_gate_.size());
        device_up_.copy_from_host(host_up_.data(), host_up_.size());
        device_down_.copy_from_host(host_down_.data(), host_down_.size());
        device_shared_gate_.fill(0);
        device_shared_gate_proj_.copy_from_host(host_shared_gate_proj_.data(),
                                                host_shared_gate_proj_.size());
        device_shared_up_.copy_from_host(host_shared_up_.data(), host_shared_up_.size());
        device_shared_down_.copy_from_host(host_shared_down_.data(), host_shared_down_.size());

        shared_gate_ = Tensor(device_shared_gate_.p, DType::FP32, {kHidden});
        selected_ids_view_ = Tensor(selected_ids_.p, DType::I32, {kTopK, width_});
        selected_weights_view_ = Tensor(selected_weights_.p, DType::FP32, {kTopK, width_});
        destination_view_ = Tensor(destination_.p, DType::BF16, {kHidden, width_});
        resident_weights_ = {
            .router = make_router(device_router_.p),
            .routed_gate = make_ggml_weight(device_gate_.p, device_gate_.bytes, routed_qtype_,
                                            kExperts, kIntermediate, kHidden),
            .routed_up = make_ggml_weight(device_up_.p, device_up_.bytes, routed_qtype_,
                                          kExperts, kIntermediate, kHidden),
            .routed_down = make_ggml_weight(device_down_.p, device_down_.bytes,
                                            QType::GGML_IQ4_NL, kExperts, kHidden,
                                            kIntermediate),
            .shared_gate = shared_gate_,
            .shared_gate_proj = make_ggml_weight(
                device_shared_gate_proj_.p, device_shared_gate_proj_.bytes, shared_qtype_, 1,
                kIntermediate, kHidden),
            .shared_up = make_ggml_weight(device_shared_up_.p, device_shared_up_.bytes,
                                          shared_qtype_, 1, kIntermediate, kHidden),
            .shared_down = make_ggml_weight(device_shared_down_.p, device_shared_down_.bytes,
                                            QType::GGML_Q8_0, 1, kHidden, kIntermediate),
        };
        CUDA_CHECK(cudaStreamCreateWithFlags(&stream_, cudaStreamNonBlocking));
    }

    ~Fixture() {
        if (stream_ != nullptr) {
            (void)cudaStreamSynchronize(stream_);
            (void)cudaStreamDestroy(stream_);
        }
    }

    Measurement measure(bool batched, bool rotating, std::int32_t iterations,
                        bool profile = false) {
        set_inputs(rotating);
        const std::int32_t warmup = 3;
        for (std::int32_t iteration = 0; iteration < warmup; ++iteration) {
            invoke(batched);
        }
        CUDA_CHECK(cudaStreamSynchronize(stream_));

        cudaEvent_t start = nullptr;
        cudaEvent_t end = nullptr;
        CUDA_CHECK(cudaEventCreate(&start));
        CUDA_CHECK(cudaEventCreate(&end));
        const auto wall_start = std::chrono::steady_clock::now();
        if (profile) { CUDA_CHECK(cudaProfilerStart()); }
        CUDA_CHECK(cudaEventRecord(start, stream_));
        for (std::int32_t iteration = 0; iteration < iterations; ++iteration) {
            invoke(batched);
        }
        CUDA_CHECK(cudaEventRecord(end, stream_));
        CUDA_CHECK(cudaStreamSynchronize(stream_));
        if (profile) { CUDA_CHECK(cudaProfilerStop()); }
        const auto wall_end = std::chrono::steady_clock::now();
        float gpu_ms = 0.0F;
        CUDA_CHECK(cudaEventElapsedTime(&gpu_ms, start, end));
        CUDA_CHECK(cudaEventDestroy(end));
        CUDA_CHECK(cudaEventDestroy(start));
        return {
            .wall_us = std::chrono::duration<double, std::micro>(wall_end - wall_start).count() /
                       iterations,
            .gpu_us = static_cast<double>(gpu_ms) * 1000.0 / iterations,
        };
    }

    std::size_t resident_bytes() const noexcept {
        return device_gate_.bytes + device_up_.bytes + device_down_.bytes +
               device_router_.bytes + device_shared_gate_.bytes +
               device_shared_gate_proj_.bytes + device_shared_up_.bytes +
               device_shared_down_.bytes;
    }

private:
    void set_inputs(bool rotating) {
        std::vector<std::uint16_t> inputs(
            static_cast<std::size_t>(width_) * kHidden, 0);
        for (std::int32_t token = 0; token < width_; ++token) {
            const std::int32_t window = rotating ? token % kRouteWindows : 0;
            inputs[static_cast<std::size_t>(token) * kHidden + window] = 0x3f80U;
        }
        device_inputs_.copy_from_host(inputs.data(), device_inputs_.bytes);
    }

    void invoke(bool batched) {
        WorkspaceArena workspace(DeviceSpan{workspace_.p, workspace_.bytes});
        if (batched) {
            Tensor x(device_inputs_.p, DType::BF16, {kHidden, width_});
            ops::qwen4_sparse_moe_resident(
                x, resident_weights_, selected_ids_view_, selected_weights_view_,
                destination_view_, workspace, stream_);
            return;
        }
        for (std::int32_t token = 0; token < width_; ++token) {
            auto* input = static_cast<std::byte*>(device_inputs_.p) +
                          static_cast<std::size_t>(token) * kHidden * sizeof(std::uint16_t);
            // Scalar public calls reuse one aligned result slot; this baseline measures the
            // unchanged T=1 execution repeated T times rather than materializing a panel.
            auto* ids = static_cast<std::byte*>(selected_ids_.p);
            auto* route_weights = static_cast<std::byte*>(selected_weights_.p);
            auto* output = static_cast<std::byte*>(destination_.p);
            Tensor x(input, DType::BF16, {kHidden});
            Tensor ids_view(ids, DType::I32, {kTopK});
            Tensor weights_view(route_weights, DType::FP32, {kTopK});
            Tensor output_view(output, DType::BF16, {kHidden});
            ops::qwen4_sparse_moe_resident(
                x, resident_weights_, ids_view, weights_view, output_view, workspace, stream_);
        }
    }

    QType routed_qtype_;
    std::int32_t width_;
    QType shared_qtype_;
    std::size_t one_routed_;
    std::size_t routed_bank_bytes_;
    std::size_t down_bank_bytes_;
    std::vector<std::uint8_t> host_gate_;
    std::vector<std::uint8_t> host_up_;
    std::vector<std::uint8_t> host_down_;
    std::vector<std::uint8_t> host_shared_gate_proj_;
    std::vector<std::uint8_t> host_shared_up_;
    std::vector<std::uint8_t> host_shared_down_;
    DeviceBuffer device_inputs_;
    DeviceBuffer device_router_;
    DeviceBuffer device_gate_;
    DeviceBuffer device_up_;
    DeviceBuffer device_down_;
    DeviceBuffer device_shared_gate_;
    DeviceBuffer device_shared_gate_proj_;
    DeviceBuffer device_shared_up_;
    DeviceBuffer device_shared_down_;
    DeviceBuffer selected_ids_;
    DeviceBuffer selected_weights_;
    DeviceBuffer destination_;
    DeviceBuffer workspace_;
    cudaStream_t stream_ = nullptr;
    Tensor shared_gate_;
    Tensor selected_ids_view_;
    Tensor selected_weights_view_;
    Tensor destination_view_;
    ops::Qwen4ResidentSparseMoeWeights resident_weights_;
};

struct Options {
    std::int32_t iterations = 10;
    std::int32_t width = 1;
    bool profile = false;
    bool iterations_explicit = false;
};

Options parse_options(int argc, char** argv) {
    Options options;
    for (int index = 1; index < argc; ++index) {
        const std::string_view option(argv[index]);
        if (option == "--profile") {
            options.profile = true;
            continue;
        }
        if ((option != "--iterations" && option != "--width") || index + 1 >= argc) {
            throw std::invalid_argument(
                "usage: qwen4 resident bench [--iterations N] [--width T] [--profile]");
        }
        const long parsed = std::strtol(argv[++index], nullptr, 10);
        if (option == "--iterations" && (parsed <= 0 || parsed > 100000)) {
            throw std::invalid_argument("iterations must be in [1,100000]");
        }
        if (option == "--width" &&
            (parsed <= 0 || parsed > ops::kQwen4SparseMoePrefillMaxWidth)) {
            throw std::invalid_argument("width must be in [1,4096]");
        }
        if (option == "--iterations") {
            options.iterations = static_cast<std::int32_t>(parsed);
            options.iterations_explicit = true;
        } else {
            options.width = static_cast<std::int32_t>(parsed);
        }
    }
    if (options.profile && options.iterations_explicit && options.iterations != 1) {
        throw std::invalid_argument("--profile requires exactly one iteration");
    }
    if (options.profile) { options.iterations = 1; }
    return options;
}

void report(QType qtype, QType shared_qtype, const char* workload, const char* placement,
            const Measurement& measurement, std::int32_t iterations,
            std::int32_t width, std::size_t resident_bytes,
            std::size_t workspace_bytes) {
    const char* format = qtype == QType::GGML_IQ1_S ? "iq1_s" : "iq2_xxs";
    const char* shared_format = shared_qtype == QType::GGML_Q5_K ? "q5_k" : "q6_k";
    std::cout << std::fixed << std::setprecision(3) << "format=" << format
              << " shared_format=" << shared_format
              << " workload=" << workload << " placement=" << placement
              << " width=" << width << " iterations=" << iterations
              << " wall_us=" << measurement.wall_us
              << " gpu_us=" << measurement.gpu_us << " resident_weight_bytes="
              << resident_bytes << " workspace_bytes=" << workspace_bytes
              << " input_tokens_per_second="
              << (1.0e6 * static_cast<double>(width) / measurement.gpu_us) << '\n';
}

} // namespace

int main(int argc, char** argv) {
    try {
        const Options options = parse_options(argc, argv);
        constexpr std::array<std::pair<QType, QType>, 3> profiles{{
            {QType::GGML_IQ1_S, QType::GGML_Q5_K},
            {QType::GGML_IQ2_XXS, QType::GGML_Q5_K},
            {QType::GGML_IQ2_XXS, QType::GGML_Q6_K},
        }};
        for (const auto [qtype, shared_qtype] : profiles) {
            if (options.profile && qtype != QType::GGML_IQ1_S) { continue; }
            Fixture fixture(qtype, shared_qtype, options.width);
            for (bool rotating : {false, true}) {
                if (options.profile && !rotating) { continue; }
                const char* workload = rotating ? "rotating" : "fixed_hot";
                Measurement scalar{};
                if (!options.profile) {
                    scalar = fixture.measure(false, rotating, options.iterations);
                }
                const Measurement resident = fixture.measure(
                    true, rotating, options.iterations, options.profile);
                const std::size_t workspace_bytes =
                    ops::qwen4_sparse_moe_resident_workspace_capacity_bytes(options.width);
                if (!options.profile) {
                    report(qtype, shared_qtype, workload, "scalar_repeat",
                           scalar, options.iterations, options.width,
                           fixture.resident_bytes(), workspace_bytes);
                }
                report(qtype, shared_qtype, workload, "resident", resident,
                       options.iterations, options.width, fixture.resident_bytes(),
                       workspace_bytes);
            }
        }
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "qwen4 resident sparse-MoE benchmark failed: " << error.what() << '\n';
        return 1;
    }
}
