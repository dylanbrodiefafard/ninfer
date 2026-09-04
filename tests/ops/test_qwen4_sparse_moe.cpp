#include "ninfer/ops/qwen4_sparse_moe.h"

#include "ops/ggml_block_linear/ggml_codebook_data.h"
#include "ops/op_tester.h"

#include <cuda_runtime.h>

#include <sys/mman.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <limits>
#include <numeric>
#include <random>
#include <span>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

using namespace ninfer;
using namespace ninfer::test;

namespace {

constexpr int kHidden = ops::kQwen4SparseMoeHidden;
constexpr int kExperts = ops::kQwen4SparseMoeExperts;
constexpr int kTopK = ops::kQwen4SparseMoeTopK;
constexpr int kIntermediate = ops::kQwen4SparseMoeIntermediate;

enum class RouteFixture {
    AllZero,
    Nonuniform,
    Underflow,
};

// A routed path crosses gate, up, SwiGLU-input, down, and final BF16 representations. The fixed
// normwise threshold is 2.5 BF16 relative-rounding units. The finite pointwise cap is two units at
// the output scale plus a 2^-15 cancellation floor. These are criteria for this implementation
// profile, not semantic per-element error guarantees or pairwise implementation tolerances.
constexpr ReductionCriterion kOutputCriterion{/*relative_l2=*/2.5 / 255.0,
                                               /*gross_absolute=*/1.0 / 32768.0,
                                               /*gross_relative_to_max_reference=*/2.0 / 255.0};
constexpr PointwiseCriterion kRouteWeightCriterion{
    /*absolute=*/128.0 * std::numeric_limits<float>::epsilon(),
    /*relative=*/128.0 * std::numeric_limits<float>::epsilon(),
};

struct FormatSpec {
    QType qtype;
    int values;
    int bytes;
};

FormatSpec format_spec(QType qtype) {
    switch (qtype) {
    case QType::GGML_Q8_0:
        return {qtype, 32, 34};
    case QType::GGML_Q5_K:
        return {qtype, 256, 176};
    case QType::GGML_Q6_K:
        return {qtype, 256, 210};
    case QType::GGML_IQ1_S:
        return {qtype, 256, 50};
    case QType::GGML_IQ2_XXS:
        return {qtype, 256, 66};
    case QType::GGML_IQ4_NL:
        return {qtype, 32, 18};
    default:
        throw std::invalid_argument("test: unsupported GGML format");
    }
}

std::size_t matrix_bytes(QType qtype, int rows, int columns) {
    const auto format = format_spec(qtype);
    return static_cast<std::size_t>(rows) * (columns / format.values) * format.bytes;
}

std::uint16_t read_u16(const std::uint8_t* bytes) {
    return static_cast<std::uint16_t>(bytes[0]) |
           (static_cast<std::uint16_t>(bytes[1]) << 8U);
}

std::uint32_t read_u32(const std::uint8_t* bytes) {
    return static_cast<std::uint32_t>(bytes[0]) |
           (static_cast<std::uint32_t>(bytes[1]) << 8U) |
           (static_cast<std::uint32_t>(bytes[2]) << 16U) |
           (static_cast<std::uint32_t>(bytes[3]) << 24U);
}

double half_to_double(std::uint16_t word) {
    const double sign = (word & 0x8000U) != 0 ? -1.0 : 1.0;
    const unsigned exponent = (word >> 10U) & 0x1fU;
    const unsigned fraction = word & 0x3ffU;
    if (exponent == 0) { return sign * std::ldexp(static_cast<double>(fraction), -24); }
    if (exponent == 31) {
        return fraction == 0 ? sign * std::numeric_limits<double>::infinity()
                             : std::numeric_limits<double>::quiet_NaN();
    }
    return sign * std::ldexp(static_cast<double>(1024U + fraction),
                             static_cast<int>(exponent) - 25);
}

int signed_byte(std::uint8_t value) {
    return value < 128U ? static_cast<int>(value) : static_cast<int>(value) - 256;
}

std::pair<int, int> scale_min(const std::uint8_t* table, int group) {
    if (group < 4) { return {table[group] & 63, table[group + 4] & 63}; }
    return {(table[group + 4] & 15) + 16 * (table[group - 4] >> 6U),
            (table[group + 4] >> 4U) + 16 * (table[group] >> 6U)};
}

double decode_value(QType qtype, const std::uint8_t* block, int index) {
    if (qtype == QType::GGML_Q8_0) {
        return half_to_double(read_u16(block)) * signed_byte(block[2 + index]);
    }
    if (qtype == QType::GGML_Q5_K) {
        const int group = index / 32;
        const int lane = index % 32;
        const auto [scale, minimum] = scale_min(block + 4, group);
        const int low = (block[48 + 32 * (group / 2) + lane] >> (4 * (group & 1))) & 15;
        const int high = (block[16 + lane] >> group) & 1;
        return half_to_double(read_u16(block)) * scale * (low + 16 * high) -
               half_to_double(read_u16(block + 2)) * minimum;
    }
    if (qtype == QType::GGML_Q6_K) {
        const int half = index / 128;
        const int within = index % 128;
        const int group = within / 32;
        const int lane = within % 32;
        const int low_word = block[64 * half + lane + 32 * (group & 1)];
        const int low = (low_word >> (4 * (group / 2))) & 15;
        const int high = (block[128 + 32 * half + lane] >> (2 * group)) & 3;
        return half_to_double(read_u16(block + 208)) * signed_byte(block[192 + index / 16]) *
               (low + 16 * high - 32);
    }
    if (qtype == QType::GGML_IQ1_S) {
        const int group = index / 32;
        const int lane = (index / 8) & 3;
        const int item = index & 7;
        const std::uint16_t control = read_u16(block + 34 + 2 * group);
        const int grid = block[2 + 4 * group + lane] |
                         (((control >> (3 * lane)) & 7U) << 8U);
        const int digit = (ops::detail::kIq1SPackedGrid[grid] >> (2 * item)) & 3U;
        const double delta = (control & 0x8000U) != 0 ? -0.125 : 0.125;
        const int multiplier = 2 * ((control >> 12U) & 7U) + 1;
        return half_to_double(read_u16(block)) * multiplier * (digit - 1 + delta);
    }
    if (qtype == QType::GGML_IQ2_XXS) {
        const int group = index / 32;
        const int lane = (index / 8) & 3;
        const int item = index & 7;
        const std::uint32_t grids = read_u32(block + 2 + 8 * group);
        const std::uint32_t signs_scales = read_u32(block + 6 + 8 * group);
        const int grid = (grids >> (8 * lane)) & 255U;
        int signs = (signs_scales >> (7 * lane)) & 127U;
        signs |= (__builtin_popcount(static_cast<unsigned>(signs)) & 1) << 7;
        const int digit = (ops::detail::kIq2XxsPackedGrid[grid] >> (2 * item)) & 3U;
        constexpr std::array<int, 3> magnitudes = {8, 25, 43};
        const double group_scale = half_to_double(read_u16(block)) *
                                   (0.5 + static_cast<double>(signs_scales >> 28U)) / 4.0;
        const int sign = (signs & (1 << item)) != 0 ? -1 : 1;
        return group_scale * magnitudes[static_cast<std::size_t>(digit)] * sign;
    }
    const std::uint8_t packed = block[2 + (index & 15)];
    const int code = index < 16 ? packed & 15U : packed >> 4U;
    return half_to_double(read_u16(block)) * ops::detail::kIq4NlGrid[code];
}

std::vector<double> linear(QType qtype, std::span<const std::uint8_t> matrix, int rows,
                           int columns, std::span<const double> input) {
    const FormatSpec format = format_spec(qtype);
    const std::size_t row_bytes = matrix_bytes(qtype, 1, columns);
    if (matrix.size() != static_cast<std::size_t>(rows) * row_bytes ||
        input.size() != static_cast<std::size_t>(columns)) {
        throw std::logic_error("test oracle received a malformed matrix");
    }
    std::vector<double> result(static_cast<std::size_t>(rows));
    for (int row = 0; row < rows; ++row) {
        const auto* row_data = matrix.data() + static_cast<std::size_t>(row) * row_bytes;
        double sum = 0.0;
        for (int column = 0; column < columns; ++column) {
            const auto* block = row_data + static_cast<std::size_t>(column / format.values) *
                                               static_cast<std::size_t>(format.bytes);
            sum += decode_value(qtype, block, column % format.values) * input[column];
        }
        result[static_cast<std::size_t>(row)] = sum;
    }
    return result;
}

double sigmoid(double value) {
    if (value >= 0.0) { return 1.0 / (1.0 + std::exp(-value)); }
    const double exponential = std::exp(value);
    return exponential / (1.0 + exponential);
}

double silu(double value) { return value * sigmoid(value); }

class AnonymousMapping {
public:
    explicit AnonymousMapping(std::size_t bytes) : bytes_(bytes) {
        data_ = mmap(nullptr, bytes_, PROT_READ | PROT_WRITE,
                     MAP_PRIVATE | MAP_ANONYMOUS | MAP_NORESERVE, -1, 0);
        if (data_ == MAP_FAILED) { throw std::runtime_error("mmap failed for routed bank"); }
    }

    ~AnonymousMapping() {
        if (data_ != MAP_FAILED) { (void)munmap(data_, bytes_); }
    }

    AnonymousMapping(const AnonymousMapping&) = delete;
    AnonymousMapping& operator=(const AnonymousMapping&) = delete;

    std::uint8_t* mutable_data() { return static_cast<std::uint8_t*>(data_); }
    const std::uint8_t* data() const { return static_cast<const std::uint8_t*>(data_); }
    std::size_t size() const { return bytes_; }

    void make_read_only() {
        if (mprotect(data_, bytes_, PROT_READ) != 0) {
            throw std::runtime_error("mprotect failed for routed bank");
        }
    }

private:
    void* data_ = MAP_FAILED;
    std::size_t bytes_ = 0;
};

void write_u16(std::uint8_t* destination, std::uint16_t value) {
    destination[0] = static_cast<std::uint8_t>(value);
    destination[1] = static_cast<std::uint8_t>(value >> 8U);
}

std::uint16_t scale_word(QType qtype, std::size_t block) {
    std::uint16_t magnitude = 0x1000U;
    if (qtype == QType::GGML_Q5_K || qtype == QType::GGML_IQ4_NL ||
        qtype == QType::GGML_Q8_0) {
        magnitude = 0x0800U;
    } else if (qtype == QType::GGML_Q6_K) {
        magnitude = 0x0400U;
    }
    return (block & 1U) == 0 ? magnitude : static_cast<std::uint16_t>(magnitude | 0x8000U);
}

std::vector<std::uint8_t> make_matrix(QType qtype, int rows, int columns, std::uint32_t seed) {
    const FormatSpec format = format_spec(qtype);
    std::vector<std::uint8_t> result(matrix_bytes(qtype, rows, columns));
    std::mt19937 generator(seed);
    std::uniform_int_distribution<int> bytes(0, 255);
    for (std::uint8_t& value : result) { value = static_cast<std::uint8_t>(bytes(generator)); }
    const std::size_t blocks = result.size() / static_cast<std::size_t>(format.bytes);
    for (std::size_t index = 0; index < blocks; ++index) {
        auto* block = result.data() + index * static_cast<std::size_t>(format.bytes);
        if (qtype == QType::GGML_Q6_K) {
            write_u16(block + 208, scale_word(qtype, index));
        } else {
            write_u16(block, scale_word(qtype, index));
        }
        if (qtype == QType::GGML_Q5_K) { write_u16(block + 2, 0x0400U); }
    }
    return result;
}

Weight make_ggml_weight(void* data, std::size_t bytes, QType qtype, int matrices, int rows,
                        int columns) {
    const FormatSpec format = format_spec(qtype);
    Weight weight{};
    weight.payload = data;
    weight.payload_bytes = bytes;
    weight.qdata = data;
    weight.qtype = qtype;
    weight.group_size = format.values;
    weight.group = format.values;
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

struct OracleRoute {
    std::array<int, kTopK> ids{};
    std::array<double, kTopK> weights{};
};

OracleRoute route_oracle(std::span<const float> router, std::span<const double> input) {
    std::vector<double> probabilities(kExperts);
    for (int expert = 0; expert < kExperts; ++expert) {
        double logit = 0.0;
        for (int column = 0; column < kHidden; ++column) {
            logit += static_cast<double>(
                         router[static_cast<std::size_t>(expert) * kHidden + column]) *
                     input[static_cast<std::size_t>(column)];
        }
        probabilities[static_cast<std::size_t>(expert)] = logit;
    }
    const double maximum = *std::max_element(probabilities.begin(), probabilities.end());
    double denominator = 0.0;
    for (double& value : probabilities) {
        value = std::exp(value - maximum);
        denominator += value;
    }
    for (double& value : probabilities) { value /= denominator; }

    std::vector<int> order(kExperts);
    std::iota(order.begin(), order.end(), 0);
    std::stable_sort(order.begin(), order.end(), [&](int left, int right) {
        if (probabilities[left] != probabilities[right]) {
            return probabilities[left] > probabilities[right];
        }
        return left < right;
    });
    OracleRoute route;
    double selected_sum = 0.0;
    for (int rank = 0; rank < kTopK; ++rank) {
        route.ids[rank] = order[rank];
        route.weights[rank] = probabilities[order[rank]];
        selected_sum += route.weights[rank];
    }
    for (double& weight : route.weights) { weight /= selected_sum; }
    return route;
}

std::span<const std::uint8_t> matrix_span(const AnonymousMapping& bank, int expert,
                                          std::size_t one_matrix) {
    return {bank.data() + static_cast<std::size_t>(expert) * one_matrix, one_matrix};
}

struct OracleInputs {
    QType routed_qtype;
    QType shared_qtype;
    std::span<const float> router;
    const AnonymousMapping& routed_gate;
    const AnonymousMapping& routed_up;
    const std::array<std::vector<std::uint8_t>, kTopK>& routed_down;
    std::span<const float> shared_gate;
    std::span<const std::uint8_t> shared_gate_proj;
    std::span<const std::uint8_t> shared_up;
    std::span<const std::uint8_t> shared_down;
};

std::vector<double> complete_oracle(std::span<const double> input, const OracleRoute& route,
                                    const OracleInputs& weights) {
    std::vector<double> output(kHidden, 0.0);
    const std::size_t routed_matrix = matrix_bytes(weights.routed_qtype, kIntermediate, kHidden);
    for (int rank = 0; rank < kTopK; ++rank) {
        const int expert = route.ids[rank];
        const auto gate = linear(weights.routed_qtype,
                                 matrix_span(weights.routed_gate, expert, routed_matrix),
                                 kIntermediate, kHidden, input);
        const auto up = linear(weights.routed_qtype,
                               matrix_span(weights.routed_up, expert, routed_matrix),
                               kIntermediate, kHidden, input);
        std::vector<double> activated(kIntermediate);
        for (int i = 0; i < kIntermediate; ++i) {
            activated[i] = silu(gate[i]) * up[i];
        }
        const auto down = linear(QType::GGML_IQ4_NL, weights.routed_down[rank], kHidden,
                                 kIntermediate, activated);
        for (int i = 0; i < kHidden; ++i) {
            output[i] += route.weights[rank] * down[i];
        }
    }

    const auto shared_gate_proj = linear(weights.shared_qtype, weights.shared_gate_proj,
                                         kIntermediate, kHidden, input);
    const auto shared_up = linear(weights.shared_qtype, weights.shared_up, kIntermediate,
                                  kHidden, input);
    std::vector<double> shared_activated(kIntermediate);
    for (int i = 0; i < kIntermediate; ++i) {
        shared_activated[i] = silu(shared_gate_proj[i]) * shared_up[i];
    }
    const auto shared = linear(QType::GGML_Q8_0, weights.shared_down, kHidden, kIntermediate,
                               shared_activated);
    double gate_logit = 0.0;
    for (int i = 0; i < kHidden; ++i) {
        gate_logit += static_cast<double>(weights.shared_gate[i]) * input[i];
    }
    const double gate = sigmoid(gate_logit);
    for (int i = 0; i < kHidden; ++i) { output[i] += gate * shared[i]; }
    return output;
}

template <class Function>
int expect_invalid(Function&& function, const char* label) {
    try {
        function();
    } catch (const std::invalid_argument&) {
        return 0;
    }
    std::cerr << label << " was accepted\n";
    return 1;
}

std::vector<float> make_input() {
    std::vector<float> input(kHidden);
    for (int i = 0; i < kHidden; ++i) {
        input[i] = static_cast<float>((i * 29) % 47 - 23) / 64.0F;
    }
    input[0] = 1.0F;
    round_to_bf16(input);
    return input;
}

int run_case(QType routed_qtype, QType shared_qtype, RouteFixture route_fixture,
             float shared_gate_extreme, cudaStream_t stream) {
    const auto input = make_input();
    std::vector<double> oracle_input(input.begin(), input.end());
    std::vector<float> router(static_cast<std::size_t>(kExperts) * kHidden, 0.0F);
    std::array<int, kTopK> intended_ids{};
    if (route_fixture == RouteFixture::AllZero) {
        std::iota(intended_ids.begin(), intended_ids.end(), 0);
    } else if (route_fixture == RouteFixture::Nonuniform) {
        intended_ids = {511, 257, 128, 64, 32, 16, 8, 4, 2, 0};
        for (int rank = 0; rank < 9; ++rank) {
            router[static_cast<std::size_t>(intended_ids[rank]) * kHidden] =
                static_cast<float>(12 - rank);
        }
        router[0] = 2.0F;
        router[static_cast<std::size_t>(1) * kHidden] = 2.0F;
    } else {
        // All non-maximum FP32 softmax values underflow to zero. Selection must still rank the
        // distinct raw logits rather than incorrectly treating the underflowed probabilities as
        // a lower-expert-id tie.
        intended_ids = {511, 500, 400, 300, 200, 100, 50, 25, 12, 6};
        for (int expert = 0; expert < kExperts; ++expert) {
            router[static_cast<std::size_t>(expert) * kHidden] = -300.0F;
        }
        router[static_cast<std::size_t>(intended_ids[0]) * kHidden] = 0.0F;
        for (int rank = 1; rank < kTopK; ++rank) {
            router[static_cast<std::size_t>(intended_ids[rank]) * kHidden] =
                static_cast<float>(-199 - rank);
        }
    }
    const OracleRoute route = route_oracle(router, oracle_input);
    if (route.ids != intended_ids) { throw std::logic_error("test router fixture is malformed"); }

    const std::size_t one_routed = matrix_bytes(routed_qtype, kIntermediate, kHidden);
    const std::size_t bank_bytes = static_cast<std::size_t>(kExperts) * one_routed;
    AnonymousMapping mapped_gate(bank_bytes);
    AnonymousMapping mapped_up(bank_bytes);
    for (int rank = 0; rank < kTopK; ++rank) {
        const auto gate = make_matrix(routed_qtype, kIntermediate, kHidden,
                                      0x1000U + static_cast<unsigned>(rank));
        const auto up = make_matrix(routed_qtype, kIntermediate, kHidden,
                                    0x2000U + static_cast<unsigned>(rank));
        std::memcpy(mapped_gate.mutable_data() +
                        static_cast<std::size_t>(route.ids[rank]) * one_routed,
                    gate.data(), gate.size());
        std::memcpy(mapped_up.mutable_data() +
                        static_cast<std::size_t>(route.ids[rank]) * one_routed,
                    up.data(), up.size());
    }
    mapped_gate.make_read_only();
    mapped_up.make_read_only();

    const std::size_t one_down = matrix_bytes(QType::GGML_IQ4_NL, kHidden, kIntermediate);
    const std::size_t down_bank_bytes = static_cast<std::size_t>(kExperts) * one_down;
    DeviceBuffer device_down(down_bank_bytes);
    device_down.fill(0);
    std::array<std::vector<std::uint8_t>, kTopK> down_matrices;
    for (int rank = 0; rank < kTopK; ++rank) {
        down_matrices[rank] = make_matrix(QType::GGML_IQ4_NL, kHidden, kIntermediate,
                                          0x3000U + static_cast<unsigned>(rank));
        device_down.copy_from_host(down_matrices[rank].data(), one_down,
                                   static_cast<std::size_t>(route.ids[rank]) * one_down);
    }

    const auto shared_gate_proj = make_matrix(shared_qtype, kIntermediate, kHidden, 0x4000U);
    const auto shared_up = make_matrix(shared_qtype, kIntermediate, kHidden, 0x5000U);
    const auto shared_down = make_matrix(QType::GGML_Q8_0, kHidden, kIntermediate, 0x6000U);
    std::vector<float> shared_gate(kHidden, 0.0F);
    shared_gate[0] = shared_gate_extreme;

    const OracleInputs oracle_weights{
        routed_qtype,
        shared_qtype,
        router,
        mapped_gate,
        mapped_up,
        down_matrices,
        shared_gate,
        shared_gate_proj,
        shared_up,
        shared_down,
    };
    const auto reference = complete_oracle(oracle_input, route, oracle_weights);

    DeviceBuffer device_x = to_device_bf16(input);
    DeviceBuffer device_router = to_device_f32(router);
    DeviceBuffer device_shared_gate = to_device_f32(shared_gate);
    DeviceBuffer device_shared_gate_proj = to_device(shared_gate_proj);
    DeviceBuffer device_shared_up = to_device(shared_up);
    DeviceBuffer device_shared_down = to_device(shared_down);
    GuardedDeviceBuffer device_stage(ops::kQwen4SparseMoeStageBytes);
    GuardedDeviceBuffer device_ids(kTopK * sizeof(std::int32_t));
    GuardedDeviceBuffer device_route_weights(kTopK * sizeof(float));
    GuardedDeviceBuffer device_destination(kHidden * sizeof(std::uint16_t));
    device_stage.fill(0xcd);
    device_ids.fill(0xcd);
    device_route_weights.fill(0xcd);
    device_destination.fill(0xcd);
    cuda_synchronize();

    constexpr std::size_t host_guard = 256;
    PinnedHostBuffer pinned(ops::kQwen4SparseMoeStageBytes + 2 * host_guard);
    auto* pinned_bytes = static_cast<std::uint8_t*>(pinned.data());
    std::memset(pinned_bytes, 0xa5, pinned.size());
    void* pinned_stage = pinned_bytes + host_guard;
    DeviceArena workspace(ops::qwen4_sparse_moe_workspace_capacity_bytes());

    Tensor x(device_x.p, DType::BF16, {kHidden});
    Tensor stage(device_stage.data(), DType::U8,
                 {static_cast<std::int32_t>(ops::kQwen4SparseMoeStageBytes)});
    Tensor ids(device_ids.data(), DType::I32, {kTopK});
    Tensor route_weights(device_route_weights.data(), DType::FP32, {kTopK});
    Tensor destination(device_destination.data(), DType::BF16, {kHidden});
    Tensor shared_gate_tensor(device_shared_gate.p, DType::FP32, {kHidden});
    ops::Qwen4SparseMoeWeights weights{
        .router = make_router(device_router.p),
        .routed_gate_up = {
            .gate = std::span<const std::byte>(
                reinterpret_cast<const std::byte*>(mapped_gate.data()), mapped_gate.size()),
            .up = std::span<const std::byte>(
                reinterpret_cast<const std::byte*>(mapped_up.data()), mapped_up.size()),
            .qtype = routed_qtype,
        },
        .routed_down = make_ggml_weight(device_down.p, down_bank_bytes,
                                        QType::GGML_IQ4_NL, kExperts, kHidden, kIntermediate),
        .shared_gate = shared_gate_tensor,
        .shared_gate_proj = make_ggml_weight(
            device_shared_gate_proj.p, shared_gate_proj.size(), shared_qtype, 1, kIntermediate,
            kHidden),
        .shared_up = make_ggml_weight(device_shared_up.p, shared_up.size(), shared_qtype, 1,
                                      kIntermediate, kHidden),
        .shared_down = make_ggml_weight(device_shared_down.p, shared_down.size(),
                                        QType::GGML_Q8_0, 1, kHidden, kIntermediate),
    };

    ops::qwen4_sparse_moe(x, weights, pinned_stage, ops::kQwen4SparseMoeStageBytes, stage, ids,
                          route_weights, destination, workspace, stream);
    cuda_synchronize(stream);

    int failures = 0;
    const auto actual_ids = from_device<std::int32_t>(device_ids.data(), kTopK);
    if (!std::equal(actual_ids.begin(), actual_ids.end(), route.ids.begin())) {
        std::cerr << "Qwen4 sparse MoE selected wrong expert ids\n";
        ++failures;
    }
    const auto actual_weight_f32 = from_device<float>(device_route_weights.data(), kTopK);
    std::vector<double> actual_weights(actual_weight_f32.begin(), actual_weight_f32.end());
    failures += verify_pointwise("Qwen4 sparse MoE selected weights", actual_weights,
                                 route.weights, kRouteWeightCriterion);
    const auto actual = from_device_bf16(device_destination.data(), kHidden);
    failures += verify_reduction("Qwen4 sparse MoE complete FP64 formula", actual, reference,
                                 kOutputCriterion);

    const std::size_t copied = ops::qwen4_sparse_moe_selected_stage_bytes(routed_qtype);
    std::vector<std::uint8_t> expected_stage(copied);
    for (int rank = 0; rank < kTopK; ++rank) {
        std::memcpy(expected_stage.data() + static_cast<std::size_t>(2 * rank) * one_routed,
                    mapped_gate.data() + static_cast<std::size_t>(route.ids[rank]) * one_routed,
                    one_routed);
        std::memcpy(expected_stage.data() + static_cast<std::size_t>(2 * rank + 1) * one_routed,
                    mapped_up.data() + static_cast<std::size_t>(route.ids[rank]) * one_routed,
                    one_routed);
    }
    if (std::memcmp(pinned_stage, expected_stage.data(), copied) != 0 ||
        !std::all_of(pinned_bytes, pinned_bytes + host_guard,
                     [](std::uint8_t value) { return value == 0xa5; }) ||
        !std::all_of(pinned_bytes + host_guard + ops::kQwen4SparseMoeStageBytes,
                     pinned_bytes + pinned.size(),
                     [](std::uint8_t value) { return value == 0xa5; })) {
        std::cerr << "Qwen4 sparse MoE pinned stage gather or guard mismatch\n";
        ++failures;
    }
    std::vector<std::uint8_t> actual_stage(copied);
    device_stage.copy_to_host(actual_stage.data(), copied);
    if (actual_stage != expected_stage) {
        std::cerr << "Qwen4 sparse MoE H2D stage changed source bytes\n";
        ++failures;
    }
    if (copied < ops::kQwen4SparseMoeStageBytes) {
        std::array<std::uint8_t, 256> unused{};
        device_stage.copy_to_host(unused.data(), unused.size(), copied);
        if (!std::all_of(unused.begin(), unused.end(),
                         [](std::uint8_t value) { return value == 0xcd; })) {
            std::cerr << "Qwen4 sparse MoE copied beyond the IQ1_S stage extent\n";
            ++failures;
        }
    }

    failures += device_stage.verify_guards("Qwen4 sparse MoE device stage");
    failures += device_ids.verify_guards("Qwen4 sparse MoE selected ids");
    failures += device_route_weights.verify_guards("Qwen4 sparse MoE selected weights");
    failures += device_destination.verify_guards("Qwen4 sparse MoE destination");

    auto invalid_bank = weights;
    invalid_bank.routed_gate_up.gate = invalid_bank.routed_gate_up.gate.first(
        invalid_bank.routed_gate_up.gate.size() - 1);
    failures += expect_invalid(
        [&] {
            ops::qwen4_sparse_moe(x, invalid_bank, pinned_stage,
                                  ops::kQwen4SparseMoeStageBytes, stage, ids, route_weights,
                                  destination, workspace, stream);
        },
        "short mapped routed bank");
    failures += expect_invalid(
        [&] {
            ops::qwen4_sparse_moe(x, weights, pinned_stage,
                                  ops::kQwen4SparseMoeStageBytes - 1, stage, ids, route_weights,
                                  destination, workspace, stream);
        },
        "short pinned stage");
    Tensor aliased_destination = x;
    failures += expect_invalid(
        [&] {
            ops::qwen4_sparse_moe(x, weights, pinned_stage,
                                  ops::kQwen4SparseMoeStageBytes, stage, ids, route_weights,
                                  aliased_destination, workspace, stream);
        },
        "aliased destination");
    auto mismatched_shared = weights;
    mismatched_shared.shared_up.qtype =
        shared_qtype == QType::GGML_Q5_K ? QType::GGML_Q6_K : QType::GGML_Q5_K;
    failures += expect_invalid(
        [&] {
            ops::qwen4_sparse_moe(x, mismatched_shared, pinned_stage,
                                  ops::kQwen4SparseMoeStageBytes, stage, ids, route_weights,
                                  destination, workspace, stream);
        },
        "mismatched shared gate/up formats");
    Tensor short_stage(device_stage.data(), DType::U8,
                       {static_cast<std::int32_t>(ops::kQwen4SparseMoeStageBytes - 1)});
    failures += expect_invalid(
        [&] {
            ops::qwen4_sparse_moe(x, weights, pinned_stage,
                                  ops::kQwen4SparseMoeStageBytes, short_stage, ids, route_weights,
                                  destination, workspace, stream);
        },
        "short device stage");
    DeviceArena short_workspace(ops::qwen4_sparse_moe_workspace_capacity_bytes() - 1);
    failures += expect_invalid(
        [&] {
            ops::qwen4_sparse_moe(x, weights, pinned_stage,
                                  ops::kQwen4SparseMoeStageBytes, stage, ids, route_weights,
                                  destination, short_workspace, stream);
        },
        "short workspace");
    return failures;
}

} // namespace

int main() {
    try {
        if (const int unavailable = require_cuda(); unavailable != 0) { return unavailable; }
        if (ops::qwen4_sparse_moe_selected_stage_bytes(QType::GGML_IQ1_S) != 6'400'000 ||
            ops::qwen4_sparse_moe_selected_stage_bytes(QType::GGML_IQ2_XXS) !=
                ops::kQwen4SparseMoeStageBytes) {
            throw std::runtime_error("Qwen4 sparse MoE stage-capacity proof is wrong");
        }
        int failures = expect_invalid(
            [] { (void)ops::qwen4_sparse_moe_selected_stage_bytes(QType::GGML_Q5_K); },
            "unsupported staged format");
        cudaStream_t stream = nullptr;
        cuda_check(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking), "cudaStreamCreate");
        failures += run_case(QType::GGML_IQ1_S, QType::GGML_Q5_K,
                             RouteFixture::AllZero, 80.0F, stream);
        failures += run_case(QType::GGML_IQ2_XXS, QType::GGML_Q6_K,
                             RouteFixture::Nonuniform, -80.0F, stream);
        failures += run_case(QType::GGML_IQ1_S, QType::GGML_Q5_K,
                             RouteFixture::Underflow, 0.0F, stream);
        cuda_check(cudaStreamDestroy(stream), "cudaStreamDestroy");
        return failures == 0 ? 0 : 1;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
