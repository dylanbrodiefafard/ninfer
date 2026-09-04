#include "ninfer/ops/qwen4_sparse_moe.h"

#include "core/arena.h"
#include "core/device.h"

#include <cuda_runtime.h>

#include <array>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <span>
#include <stdexcept>
#include <string_view>
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

struct PipelineEvents {
    cudaStream_t transfer = nullptr;
    cudaEvent_t route_ready = nullptr;
    cudaEvent_t ids_ready = nullptr;
    std::array<cudaEvent_t, ops::kQwen4SparseMoePipelineSlots> transfer_ready{};
    std::array<cudaEvent_t, ops::kQwen4SparseMoePipelineSlots> consumer_complete{};

    PipelineEvents() {
        CUDA_CHECK(cudaStreamCreateWithFlags(&transfer, cudaStreamNonBlocking));
        CUDA_CHECK(cudaEventCreateWithFlags(&route_ready, cudaEventDisableTiming));
        CUDA_CHECK(cudaEventCreateWithFlags(
            &ids_ready, cudaEventDisableTiming | cudaEventBlockingSync));
        for (std::size_t slot = 0; slot < transfer_ready.size(); ++slot) {
            CUDA_CHECK(cudaEventCreateWithFlags(
                &transfer_ready[slot], cudaEventDisableTiming | cudaEventBlockingSync));
            CUDA_CHECK(cudaEventCreateWithFlags(&consumer_complete[slot],
                                                cudaEventDisableTiming));
        }
    }

    ~PipelineEvents() {
        if (transfer != nullptr) { (void)cudaStreamSynchronize(transfer); }
        for (cudaEvent_t event : consumer_complete) { (void)cudaEventDestroy(event); }
        for (cudaEvent_t event : transfer_ready) { (void)cudaEventDestroy(event); }
        (void)cudaEventDestroy(ids_ready);
        (void)cudaEventDestroy(route_ready);
        (void)cudaStreamDestroy(transfer);
    }
};

struct Measurement {
    double wall_us = 0.0;
    double gpu_us = 0.0;
};

class Fixture {
public:
    explicit Fixture(QType routed_qtype)
        : routed_qtype_(routed_qtype),
          shared_qtype_(routed_qtype == QType::GGML_IQ1_S ? QType::GGML_Q5_K
                                                          : QType::GGML_Q6_K),
          one_routed_(matrix_bytes(routed_qtype_, kIntermediate, kHidden)),
          routed_bank_bytes_(static_cast<std::size_t>(kExperts) * one_routed_),
          down_bank_bytes_(static_cast<std::size_t>(kExperts) *
                           matrix_bytes(QType::GGML_IQ4_NL, kHidden, kIntermediate)),
          host_gate_(routed_bank_bytes_), host_up_(routed_bank_bytes_),
          host_down_(down_bank_bytes_),
          host_shared_gate_proj_(matrix_bytes(shared_qtype_, kIntermediate, kHidden)),
          host_shared_up_(matrix_bytes(shared_qtype_, kIntermediate, kHidden)),
          host_shared_down_(matrix_bytes(QType::GGML_Q8_0, kHidden, kIntermediate)),
          device_inputs_(static_cast<std::size_t>(kRouteWindows) * kHidden * sizeof(std::uint16_t)),
          device_router_(static_cast<std::size_t>(kExperts) * kHidden * sizeof(float)),
          device_gate_(routed_bank_bytes_), device_up_(routed_bank_bytes_),
          device_down_(down_bank_bytes_), device_shared_gate_(kHidden * sizeof(float)),
          device_shared_gate_proj_(host_shared_gate_proj_.size()),
          device_shared_up_(host_shared_up_.size()), device_shared_down_(host_shared_down_.size()),
          device_stage_(ops::kQwen4SparseMoePipelineStageBytes),
          pinned_stage_(ops::kQwen4SparseMoePipelineStageBytes),
          selected_ids_(kTopK * sizeof(std::int32_t)),
          selected_weights_(kTopK * sizeof(float)), destination_(kHidden * sizeof(std::uint16_t)),
          workspace_(ops::qwen4_sparse_moe_workspace_capacity_bytes()) {
        fill_valid_blocks(host_gate_, routed_qtype_, 0x12345678U);
        fill_valid_blocks(host_up_, routed_qtype_, 0x87654321U);
        fill_valid_blocks(host_down_, QType::GGML_IQ4_NL, 0x31415926U);
        fill_valid_blocks(host_shared_gate_proj_, shared_qtype_, 0x27182818U);
        fill_valid_blocks(host_shared_up_, shared_qtype_, 0x16180339U);
        fill_valid_blocks(host_shared_down_, QType::GGML_Q8_0, 0x42424242U);

        std::vector<std::uint16_t> inputs(static_cast<std::size_t>(kRouteWindows) * kHidden, 0);
        std::vector<float> router(static_cast<std::size_t>(kExperts) * kHidden, 0.0F);
        for (std::int32_t window = 0; window < kRouteWindows; ++window) {
            inputs[static_cast<std::size_t>(window) * kHidden + window] = 0x3f80U;
            for (std::int32_t rank = 0; rank < kTopK; ++rank) {
                const std::int32_t expert = (window * kTopK + rank) % kExperts;
                router[static_cast<std::size_t>(expert) * kHidden + window] =
                    static_cast<float>(20 - rank);
            }
        }
        device_inputs_.copy_from_host(inputs.data(), device_inputs_.bytes);
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
        selected_ids_view_ = Tensor(selected_ids_.p, DType::I32, {kTopK});
        selected_weights_view_ = Tensor(selected_weights_.p, DType::FP32, {kTopK});
        destination_view_ = Tensor(destination_.p, DType::BF16, {kHidden});
        device_stage_view_ = Tensor(
            device_stage_.p, DType::U8,
            {static_cast<std::int32_t>(ops::kQwen4SparseMoePipelineStageBytes)});

        staged_weights_ = {
            .router = make_router(device_router_.p),
            .routed_gate_up = {
                .gate = std::span<const std::byte>(
                    reinterpret_cast<const std::byte*>(host_gate_.data()), host_gate_.size()),
                .up = std::span<const std::byte>(
                    reinterpret_cast<const std::byte*>(host_up_.data()), host_up_.size()),
                .qtype = routed_qtype_,
            },
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
        resident_weights_ = {
            .router = staged_weights_.router,
            .routed_gate = make_ggml_weight(device_gate_.p, device_gate_.bytes, routed_qtype_,
                                            kExperts, kIntermediate, kHidden),
            .routed_up = make_ggml_weight(device_up_.p, device_up_.bytes, routed_qtype_,
                                          kExperts, kIntermediate, kHidden),
            .routed_down = staged_weights_.routed_down,
            .shared_gate = staged_weights_.shared_gate,
            .shared_gate_proj = staged_weights_.shared_gate_proj,
            .shared_up = staged_weights_.shared_up,
            .shared_down = staged_weights_.shared_down,
        };
        pipeline_ = {
            .pinned_stage = pinned_stage_.data(),
            .pinned_stage_bytes = pinned_stage_.size(),
            .device_stage = device_stage_view_,
            .transfer_stream = events_.transfer,
            .route_ready = events_.route_ready,
            .ids_ready = events_.ids_ready,
            .transfer_ready = {events_.transfer_ready[0], events_.transfer_ready[1]},
            .consumer_complete = {events_.consumer_complete[0],
                                  events_.consumer_complete[1]},
        };
        CUDA_CHECK(cudaStreamCreateWithFlags(&stream_, cudaStreamNonBlocking));
    }

    ~Fixture() {
        if (stream_ != nullptr) {
            (void)cudaStreamSynchronize(stream_);
            (void)cudaStreamDestroy(stream_);
        }
    }

    Measurement measure(bool resident, bool rotating, std::int32_t iterations) {
        const std::int32_t warmup = rotating ? kRouteWindows : 16;
        for (std::int32_t iteration = 0; iteration < warmup; ++iteration) {
            invoke(resident, rotating ? iteration % kRouteWindows : 0);
        }
        CUDA_CHECK(cudaStreamSynchronize(stream_));

        cudaEvent_t start = nullptr;
        cudaEvent_t end = nullptr;
        CUDA_CHECK(cudaEventCreate(&start));
        CUDA_CHECK(cudaEventCreate(&end));
        const auto wall_start = std::chrono::steady_clock::now();
        CUDA_CHECK(cudaEventRecord(start, stream_));
        for (std::int32_t iteration = 0; iteration < iterations; ++iteration) {
            invoke(resident, rotating ? iteration % kRouteWindows : 0);
        }
        CUDA_CHECK(cudaEventRecord(end, stream_));
        CUDA_CHECK(cudaStreamSynchronize(stream_));
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
    void invoke(bool resident, std::int32_t window) {
        auto* input = static_cast<std::byte*>(device_inputs_.p) +
                      static_cast<std::size_t>(window) * kHidden * sizeof(std::uint16_t);
        Tensor x(input, DType::BF16, {kHidden});
        WorkspaceArena workspace(DeviceSpan{workspace_.p, workspace_.bytes});
        if (resident) {
            ops::qwen4_sparse_moe_resident(x, resident_weights_, selected_ids_view_,
                                           selected_weights_view_, destination_view_, workspace,
                                           stream_);
        } else {
            ops::qwen4_sparse_moe(x, staged_weights_, pipeline_, selected_ids_view_,
                                  selected_weights_view_, destination_view_, workspace, stream_);
        }
    }

    QType routed_qtype_;
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
    DeviceBuffer device_stage_;
    PinnedHostBuffer pinned_stage_;
    DeviceBuffer selected_ids_;
    DeviceBuffer selected_weights_;
    DeviceBuffer destination_;
    DeviceBuffer workspace_;
    PipelineEvents events_;
    cudaStream_t stream_ = nullptr;
    Tensor shared_gate_;
    Tensor selected_ids_view_;
    Tensor selected_weights_view_;
    Tensor destination_view_;
    Tensor device_stage_view_;
    ops::Qwen4SparseMoeWeights staged_weights_;
    ops::Qwen4ResidentSparseMoeWeights resident_weights_;
    ops::Qwen4SparseMoePipeline pipeline_;
};

std::int32_t parse_iterations(int argc, char** argv) {
    std::int32_t iterations = 104;
    for (int index = 1; index < argc; ++index) {
        if (std::string_view(argv[index]) != "--iterations" || index + 1 >= argc) {
            throw std::invalid_argument("usage: qwen4 resident bench [--iterations N]");
        }
        const long parsed = std::strtol(argv[++index], nullptr, 10);
        if (parsed <= 0 || parsed > 100000) {
            throw std::invalid_argument("iterations must be in [1,100000]");
        }
        iterations = static_cast<std::int32_t>(parsed);
    }
    return iterations;
}

void report(QType qtype, const char* workload, const char* placement,
            const Measurement& measurement, std::int32_t iterations,
            std::size_t resident_bytes) {
    const char* format = qtype == QType::GGML_IQ1_S ? "iq1_s" : "iq2_xxs";
    std::cout << std::fixed << std::setprecision(3) << "format=" << format
              << " workload=" << workload << " placement=" << placement
              << " iterations=" << iterations << " wall_us=" << measurement.wall_us
              << " gpu_us=" << measurement.gpu_us << " resident_weight_bytes="
              << resident_bytes << '\n';
}

} // namespace

int main(int argc, char** argv) {
    try {
        const std::int32_t iterations = parse_iterations(argc, argv);
        for (QType qtype : {QType::GGML_IQ1_S, QType::GGML_IQ2_XXS}) {
            Fixture fixture(qtype);
            for (bool rotating : {false, true}) {
                const char* workload = rotating ? "rotating" : "fixed_hot";
                const Measurement staged = fixture.measure(false, rotating, iterations);
                const Measurement resident = fixture.measure(true, rotating, iterations);
                report(qtype, workload, "staged", staged, iterations,
                       fixture.resident_bytes());
                report(qtype, workload, "resident", resident, iterations,
                       fixture.resident_bytes());
            }
        }
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "qwen4 resident sparse-MoE benchmark failed: " << error.what() << '\n';
        return 1;
    }
}
