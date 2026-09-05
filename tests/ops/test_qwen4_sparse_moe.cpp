#include "ninfer/ops/qwen4_sparse_moe.h"

#include "ops/ggml_iq_oracle.h"
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

struct PipelineEvents {
    cudaStream_t transfer_stream = nullptr;
    cudaEvent_t route_ready = nullptr;
    cudaEvent_t ids_ready = nullptr;
    std::array<cudaEvent_t, ops::kQwen4SparseMoePipelineSlots> transfer_ready{};
    std::array<cudaEvent_t, ops::kQwen4SparseMoePipelineSlots> consumer_complete{};

    PipelineEvents() {
        cuda_check(cudaStreamCreateWithFlags(&transfer_stream, cudaStreamNonBlocking),
                   "cudaStreamCreate transfer");
        cuda_check(cudaEventCreateWithFlags(&route_ready, cudaEventDisableTiming),
                   "cudaEventCreate route_ready");
        cuda_check(cudaEventCreateWithFlags(
                       &ids_ready, cudaEventDisableTiming | cudaEventBlockingSync),
                   "cudaEventCreate ids_ready");
        for (std::size_t slot = 0; slot < transfer_ready.size(); ++slot) {
            cuda_check(cudaEventCreateWithFlags(
                           &transfer_ready[slot],
                           cudaEventDisableTiming | cudaEventBlockingSync),
                       "cudaEventCreate transfer_ready");
            cuda_check(cudaEventCreateWithFlags(&consumer_complete[slot],
                                                 cudaEventDisableTiming),
                       "cudaEventCreate consumer_complete");
        }
    }

    ~PipelineEvents() {
        if (transfer_stream != nullptr) { (void)cudaStreamSynchronize(transfer_stream); }
        for (cudaEvent_t event : consumer_complete) { (void)cudaEventDestroy(event); }
        for (cudaEvent_t event : transfer_ready) { (void)cudaEventDestroy(event); }
        (void)cudaEventDestroy(ids_ready);
        (void)cudaEventDestroy(route_ready);
        (void)cudaStreamDestroy(transfer_stream);
    }

    PipelineEvents(const PipelineEvents&) = delete;
    PipelineEvents& operator=(const PipelineEvents&) = delete;
};

// A routed path crosses gate, up, SwiGLU-input, down, and final BF16 representations. The fixed
// normwise threshold is 2.5 BF16 relative-rounding units. The finite pointwise cap is two units at
// the output scale plus a 2^-15 cancellation floor. These are criteria for this implementation
// profile, not semantic per-element error guarantees or pairwise implementation tolerances.
constexpr ReductionCriterion kOutputCriterion{/*relative_l2=*/2.5 / 255.0,
                                               /*gross_absolute=*/1.0 / 32768.0,
                                               /*gross_relative_to_max_reference=*/2.0 / 255.0};
// The standalone fused Op has two semantic BF16 projection boundaries and one BF16 Store. Its
// ideal FP64 oracle applies those boundaries without reproducing the production reduction order.
constexpr ReductionCriterion kGateUpSwiGluCriterion{
    /*relative_l2=*/2.5 / 255.0,
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
        const int digit = iq_oracle::iq1_grid(grid)[item];
        const double delta = (control & 0x8000U) != 0 ? -0.125 : 0.125;
        const int multiplier = 2 * ((control >> 12U) & 7U) + 1;
        return half_to_double(read_u16(block)) * multiplier * (digit + delta);
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
        const int magnitude = iq_oracle::iq2_grid(grid)[item];
        const double group_scale = half_to_double(read_u16(block)) *
                                   (0.5 + static_cast<double>(signs_scales >> 28U)) / 4.0;
        const int sign = (signs & (1 << item)) != 0 ? -1 : 1;
        return group_scale * magnitude * sign;
    }
    const std::uint8_t packed = block[2 + (index & 15)];
    const int code = index < 16 ? packed & 15U : packed >> 4U;
    return half_to_double(read_u16(block)) * iq_oracle::kIq4Nl[code];
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

std::uint64_t round_nonnegative_ties_to_even(double value) {
    const double lower_value = std::floor(value);
    auto lower = static_cast<std::uint64_t>(lower_value);
    const double remainder = value - lower_value;
    if (remainder > 0.5 || (remainder == 0.5 && (lower & 1U) != 0)) { ++lower; }
    return lower;
}

double round_fp64_to_bf16(double value) {
    if (value == 0.0 || !std::isfinite(value)) { return value; }
    const bool negative = std::signbit(value);
    const double magnitude = std::abs(value);
    std::uint16_t word = negative ? 0x8000U : 0U;
    if (magnitude < std::ldexp(1.0, -126)) {
        const auto significand = round_nonnegative_ties_to_even(std::ldexp(magnitude, 133));
        word = static_cast<std::uint16_t>(word | significand);
        return static_cast<double>(bf16_to_f32(word));
    }

    int exponent = 0;
    const double fraction = std::frexp(magnitude, &exponent);
    int unbiased_exponent = exponent - 1;
    auto significand = round_nonnegative_ties_to_even(std::ldexp(fraction, 8));
    if (significand == 256) {
        significand = 128;
        ++unbiased_exponent;
    }
    if (unbiased_exponent > 127) {
        word = static_cast<std::uint16_t>(word | 0x7f80U);
    } else {
        const auto exponent_field = static_cast<std::uint16_t>(unbiased_exponent + 127);
        word = static_cast<std::uint16_t>(
            word | static_cast<std::uint16_t>(exponent_field << 7U) |
            static_cast<std::uint16_t>(significand - 128));
    }
    return static_cast<double>(bf16_to_f32(word));
}

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
        if (qtype == QType::GGML_IQ1_S) {
            constexpr std::array<int, 4> grids = {0, 1, 2, 2047};
            for (int group = 0; group < 8; ++group) {
                std::uint16_t control = 0;
                for (int lane = 0; lane < 4; ++lane) {
                    const int grid = grids[(seed + index + group + lane) & 3U];
                    block[2 + 4 * group + lane] = static_cast<std::uint8_t>(grid);
                    control |= static_cast<std::uint16_t>(grid >> 8) << (3 * lane);
                }
                const std::uint16_t multiplier =
                    static_cast<std::uint16_t>((seed + index + group) & 7U);
                const std::uint16_t delta =
                    ((seed + index + group) & 1U) != 0 ? 0x8000U : 0U;
                write_u16(block + 34 + 2 * group,
                          static_cast<std::uint16_t>(control | (multiplier << 12U) | delta));
            }
        } else if (qtype == QType::GGML_IQ2_XXS) {
            constexpr std::array<int, 4> grids = {0, 1, 2, 255};
            for (int group = 0; group < 8; ++group) {
                std::uint32_t signs = 0;
                for (int lane = 0; lane < 4; ++lane) {
                    block[2 + 8 * group + lane] = static_cast<std::uint8_t>(
                        grids[(seed + index + group + lane) & 3U]);
                    const std::uint32_t sign =
                        (seed + 19U * index + 11U * group + 29U * lane) & 127U;
                    signs |= sign << (7 * lane);
                }
                const std::uint32_t scale =
                    static_cast<std::uint32_t>((seed + index + group) & 15U) << 28U;
                const std::uint32_t control = signs | scale;
                for (int byte = 0; byte < 4; ++byte) {
                    block[6 + 8 * group + byte] =
                        static_cast<std::uint8_t>(control >> (8 * byte));
                }
            }
        } else if (qtype == QType::GGML_IQ4_NL) {
            for (int lane = 0; lane < 16; ++lane) {
                const int low = static_cast<int>((seed + index + 5U * lane) & 15U);
                const int high = static_cast<int>((seed + 3U * index + 7U * lane) & 15U);
                block[2 + lane] = static_cast<std::uint8_t>((high << 4) | low);
            }
        }
    }
    return result;
}

std::vector<std::uint8_t> make_zero_scaled_matrix(QType qtype, int rows, int columns,
                                                  std::uint32_t seed) {
    std::vector<std::uint8_t> result = make_matrix(qtype, rows, columns, seed);
    const FormatSpec format = format_spec(qtype);
    const std::size_t blocks = result.size() / static_cast<std::size_t>(format.bytes);
    for (std::size_t index = 0; index < blocks; ++index) {
        auto* block = result.data() + index * static_cast<std::size_t>(format.bytes);
        if (qtype == QType::GGML_Q6_K) {
            write_u16(block + 208, 0);
        } else {
            write_u16(block, 0);
            if (qtype == QType::GGML_Q5_K) { write_u16(block + 2, 0); }
        }
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
    std::vector<double> logits(kExperts);
    for (int expert = 0; expert < kExperts; ++expert) {
        double logit = 0.0;
        for (int column = 0; column < kHidden; ++column) {
            logit += static_cast<double>(
                         router[static_cast<std::size_t>(expert) * kHidden + column]) *
                     input[static_cast<std::size_t>(column)];
        }
        logits[static_cast<std::size_t>(expert)] = logit;
    }
    std::vector<double> probabilities(kExperts);
    const double maximum = *std::max_element(logits.begin(), logits.end());
    double denominator = 0.0;
    for (int expert = 0; expert < kExperts; ++expert) {
        probabilities[expert] = std::exp(logits[expert] - maximum);
        denominator += probabilities[expert];
    }
    for (double& value : probabilities) { value /= denominator; }

    std::vector<int> order(kExperts);
    std::iota(order.begin(), order.end(), 0);
    std::stable_sort(order.begin(), order.end(), [&](int left, int right) {
        if (logits[left] != logits[right]) {
            return logits[left] > logits[right];
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
    std::array<int, kTopK> routed_down_experts;
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
        const auto down_entry = std::find(weights.routed_down_experts.begin(),
                                          weights.routed_down_experts.end(), expert);
        if (down_entry == weights.routed_down_experts.end()) {
            throw std::logic_error("oracle has no routed-down matrix for selected expert");
        }
        const auto down_index = static_cast<std::size_t>(
            std::distance(weights.routed_down_experts.begin(), down_entry));
        const auto down = linear(QType::GGML_IQ4_NL, weights.routed_down[down_index], kHidden,
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

int prefill_gate_up_dense_oracle_case(QType routed_qtype, cudaStream_t stream) {
    constexpr int maximum_test_width = 17;
    const auto gate = make_matrix(routed_qtype, kIntermediate, kHidden, 0x17U);
    const auto up = make_matrix(routed_qtype, kIntermediate, kHidden, 0x2bU);
    DeviceBuffer device_gate = to_device(gate);
    DeviceBuffer device_up = to_device(up);
    const Weight gate_weight = make_ggml_weight(
        device_gate.p, gate.size(), routed_qtype, 1, kIntermediate, kHidden);
    const Weight up_weight = make_ggml_weight(
        device_up.p, up.size(), routed_qtype, 1, kIntermediate, kHidden);

    std::vector<float> input(static_cast<std::size_t>(kHidden) * maximum_test_width);
    for (int token = 0; token < maximum_test_width; ++token) {
        for (int column = 0; column < kHidden; ++column) {
            input[static_cast<std::size_t>(token) * kHidden + column] =
                static_cast<float>((column * 29 + token * 17) % 61 - 30) / 128.0F;
        }
    }
    round_to_bf16(input);
    std::vector<double> reference;
    reference.reserve(static_cast<std::size_t>(kIntermediate) * maximum_test_width);
    for (int token = 0; token < maximum_test_width; ++token) {
        std::vector<double> represented(kHidden);
        for (int column = 0; column < kHidden; ++column) {
            represented[column] = input[static_cast<std::size_t>(token) * kHidden + column];
        }
        const auto gate_projection = linear(routed_qtype, gate, kIntermediate, kHidden,
                                            represented);
        const auto up_projection = linear(routed_qtype, up, kIntermediate, kHidden,
                                          represented);
        for (int row = 0; row < kIntermediate; ++row) {
            const double gate_bf16 = round_fp64_to_bf16(gate_projection[row]);
            const double up_bf16 = round_fp64_to_bf16(up_projection[row]);
            reference.push_back(round_fp64_to_bf16(silu(gate_bf16) * up_bf16));
        }
    }

    DeviceBuffer device_input = to_device_bf16(input);
    int failures = 0;
    for (int width : {1, 16, 17}) {
        GuardedDeviceBuffer device_output(
            static_cast<std::size_t>(kIntermediate) * width * sizeof(std::uint16_t));
        cuda_check(cudaDeviceSynchronize(),
                   "cudaDeviceSynchronize public gate/up SwiGLU guards");
        cuda_check(cudaMemsetAsync(device_output.data(), 0xa7, device_output.bytes(), stream),
                   "cudaMemsetAsync public gate/up SwiGLU output");
        Tensor x(device_input.p, DType::BF16, {kHidden, width});
        Tensor output(device_output.data(), DType::BF16, {kIntermediate, width});
        ops::qwen4_sparse_moe_gate_up_swiglu(
            x, gate_weight, up_weight, output, stream);
        cuda_synchronize(stream);
        const auto actual = from_device_bf16(
            device_output.data(), static_cast<std::size_t>(kIntermediate) * width);
        const std::string label =
            std::string(routed_qtype == QType::GGML_IQ1_S ? "Qwen4 IQ1" : "Qwen4 IQ2") +
            " public gate/up SwiGLU dense FP64 T=" + std::to_string(width);
        failures += verify_reduction(
            label.c_str(), actual,
            std::span<const double>(reference).first(
                static_cast<std::size_t>(kIntermediate) * width),
            kGateUpSwiGluCriterion);
        failures += device_output.verify_guards(label.c_str());
    }

    constexpr int maximum_width = ops::kQwen4SparseMoePrefillMaxWidth;
    DeviceBuffer zero_input(
        static_cast<std::size_t>(kHidden) * maximum_width * sizeof(std::uint16_t));
    GuardedDeviceBuffer maximum_output(
        static_cast<std::size_t>(kIntermediate) * maximum_width * sizeof(std::uint16_t));
    cuda_check(cudaDeviceSynchronize(),
               "cudaDeviceSynchronize public gate/up SwiGLU maximum guards");
    cuda_check(cudaMemsetAsync(zero_input.p, 0, zero_input.bytes, stream),
               "cudaMemsetAsync public gate/up SwiGLU maximum input");
    cuda_check(cudaMemsetAsync(maximum_output.data(), 0xa7, maximum_output.bytes(), stream),
               "cudaMemsetAsync public gate/up SwiGLU maximum output");
    Tensor maximum_x(zero_input.p, DType::BF16, {kHidden, maximum_width});
    Tensor maximum_out(maximum_output.data(), DType::BF16,
                       {kIntermediate, maximum_width});
    ops::qwen4_sparse_moe_gate_up_swiglu(
        maximum_x, gate_weight, up_weight, maximum_out, stream);
    cuda_synchronize(stream);
    failures += verify_exact(
        routed_qtype == QType::GGML_IQ1_S
            ? "Qwen4 IQ1 public gate/up SwiGLU maximum zero oracle"
            : "Qwen4 IQ2 public gate/up SwiGLU maximum zero oracle",
        from_device<std::uint16_t>(maximum_output.data(),
                                   static_cast<std::size_t>(kIntermediate) * maximum_width),
        std::vector<std::uint16_t>(static_cast<std::size_t>(kIntermediate) * maximum_width));
    failures += maximum_output.verify_guards(
        routed_qtype == QType::GGML_IQ1_S
            ? "Qwen4 IQ1 public gate/up SwiGLU maximum"
            : "Qwen4 IQ2 public gate/up SwiGLU maximum");

    Tensor wrong_shape(device_input.p, DType::BF16, {kHidden - 1, maximum_test_width});
    Tensor valid_output(maximum_output.data(), DType::BF16,
                        {kIntermediate, maximum_test_width});
    failures += expect_invalid(
        [&] {
            ops::qwen4_sparse_moe_gate_up_swiglu(
                wrong_shape, gate_weight, up_weight, valid_output, stream);
        },
        "Qwen4 public gate/up SwiGLU malformed input shape");
    Weight wrong_format = gate_weight;
    wrong_format.qtype = QType::GGML_Q8_0;
    failures += expect_invalid(
        [&] {
            ops::qwen4_sparse_moe_gate_up_swiglu(
                Tensor(device_input.p, DType::BF16, {kHidden, maximum_test_width}),
                wrong_format, up_weight, valid_output, stream);
        },
        "Qwen4 public gate/up SwiGLU malformed format");
    Tensor aliased_output(device_input.p, DType::BF16,
                          {kIntermediate, maximum_test_width});
    failures += expect_invalid(
        [&] {
            ops::qwen4_sparse_moe_gate_up_swiglu(
                Tensor(device_input.p, DType::BF16, {kHidden, maximum_test_width}),
                gate_weight, up_weight, aliased_output, stream);
        },
        "Qwen4 public gate/up SwiGLU destructive output alias");
    failures += expect_invalid(
        [&] {
            ops::qwen4_sparse_moe_gate_up_swiglu(
                Tensor(device_input.p, DType::BF16, {kHidden, maximum_test_width}),
                gate_weight, gate_weight, valid_output, stream);
        },
        "Qwen4 public gate/up SwiGLU aliased gate/up");
    return failures;
}

int prefill_gate_up_rounding_boundary_case(QType routed_qtype, cudaStream_t stream) {
    constexpr int width = 17;
    const std::uint32_t gate_seed = routed_qtype == QType::GGML_IQ1_S ? 3U : 0U;
    const std::uint32_t up_seed = 2U;
    const float active_input = routed_qtype == QType::GGML_IQ1_S ? 500.0F : 752.0F;
    const FormatSpec format = format_spec(routed_qtype);

    // One independently packed nonzero coefficient makes the FP32 dot product exact and
    // association-independent. The remaining blocks have zero scales. Tokens 15 and 16 straddle
    // the production 16-occurrence tile boundary.
    auto gate = make_zero_scaled_matrix(routed_qtype, kIntermediate, kHidden, gate_seed);
    auto up = make_zero_scaled_matrix(routed_qtype, kIntermediate, kHidden, up_seed);
    const auto gate_witness = make_matrix(routed_qtype, 1, kHidden, gate_seed);
    const auto up_witness = make_matrix(routed_qtype, 1, kHidden, up_seed);
    std::copy_n(gate_witness.begin(), format.bytes, gate.begin());
    std::copy_n(up_witness.begin(), format.bytes, up.begin());

    std::vector<float> input(static_cast<std::size_t>(kHidden) * width, 0.0F);
    constexpr std::array<int, 3> active_tokens{0, 15, 16};
    for (int token : active_tokens) {
        input[static_cast<std::size_t>(kHidden) * token] = active_input;
    }
    round_to_bf16(input);

    const double gate_exact = decode_value(routed_qtype, gate.data(), 0) * active_input;
    const double up_exact = decode_value(routed_qtype, up.data(), 0) * active_input;
    const float gate_total = static_cast<float>(gate_exact);
    const float up_total = static_cast<float>(up_exact);
    const float gate_bf16 = bf16_to_f32(f32_to_bf16(gate_total));
    const float up_bf16 = bf16_to_f32(f32_to_bf16(up_total));
    const std::uint16_t expected_word = f32_to_bf16(
        gate_bf16 / (1.0F + std::exp(-gate_bf16)) * up_bf16);
    const std::uint16_t unrounded_word =
        f32_to_bf16(gate_total / (1.0F + std::exp(-gate_total)) * up_total);
    if (expected_word == unrounded_word) {
        throw std::logic_error("SwiGLU fixture does not witness BF16 projection rounding");
    }

    DeviceBuffer device_x = to_device_bf16(input);
    DeviceBuffer device_gate = to_device(gate);
    DeviceBuffer device_up = to_device(up);
    GuardedDeviceBuffer device_activated(
        static_cast<std::size_t>(kIntermediate) * width * sizeof(std::uint16_t));
    cuda_check(cudaDeviceSynchronize(),
               "cudaDeviceSynchronize public gate/up SwiGLU witness guards");
    cuda_check(cudaMemsetAsync(
                   device_activated.data(), 0xcd, device_activated.bytes(), stream),
               "cudaMemsetAsync public gate/up SwiGLU witness output");
    Tensor x(device_x.p, DType::BF16, {kHidden, width});
    Tensor activated(device_activated.data(), DType::BF16, {kIntermediate, width});
    const Weight gate_weight = make_ggml_weight(
        device_gate.p, gate.size(), routed_qtype, 1, kIntermediate, kHidden);
    const Weight up_weight = make_ggml_weight(
        device_up.p, up.size(), routed_qtype, 1, kIntermediate, kHidden);
    ops::qwen4_sparse_moe_gate_up_swiglu(
        x, gate_weight, up_weight, activated, stream);
    cuda_synchronize(stream);

    const auto actual = from_device<std::uint16_t>(
        device_activated.data(), static_cast<std::size_t>(kIntermediate) * width);
    std::vector<std::uint16_t> expected(actual.size(), 0);
    for (int token : active_tokens) {
        expected[static_cast<std::size_t>(kIntermediate) * token] = expected_word;
    }
    int failures = verify_exact(
        routed_qtype == QType::GGML_IQ1_S
            ? "Qwen4 IQ1 prefill fused BF16 projection/SwiGLU boundaries"
            : "Qwen4 IQ2 prefill fused BF16 projection/SwiGLU boundaries",
        actual, expected);
    failures += device_activated.verify_guards(
        routed_qtype == QType::GGML_IQ1_S
            ? "Qwen4 IQ1 prefill fused activation"
            : "Qwen4 IQ2 prefill fused activation");
    return failures;
}

int run_case(QType routed_qtype, QType shared_qtype, RouteFixture route_fixture,
             float shared_gate_extreme, PipelineEvents& pipeline_events,
             cudaStream_t stream) {
    const auto input = make_input();
    std::vector<double> oracle_input(input.begin(), input.end());
    std::vector<float> router(static_cast<std::size_t>(kExperts) * kHidden, 0.0F);
    std::array<int, kTopK> intended_ids{};
    if (route_fixture == RouteFixture::AllZero) {
        std::iota(intended_ids.begin(), intended_ids.end(), 0);
    } else if (route_fixture == RouteFixture::Nonuniform) {
        // The selected set crosses both 256-expert halves. Experts 255 and 256 tie at the tenth
        // boundary, so the lower id must win even though the pair straddles the half boundary.
        intended_ids = {511, 400, 300, 257, 128, 64, 32, 16, 8, 255};
        for (int rank = 0; rank < 9; ++rank) {
            router[static_cast<std::size_t>(intended_ids[rank]) * kHidden] =
                static_cast<float>(12 - rank);
        }
        router[static_cast<std::size_t>(255) * kHidden] = 2.0F;
        router[static_cast<std::size_t>(256) * kHidden] = 2.0F;
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

    // A second, disjoint route exercises persistent slot ownership across Op calls. The second
    // invocation is submitted without synchronizing either stream after the first, so a rank-8/9
    // slot overwritten by the next call's rank 0/1 changes the first result.
    std::vector<float> next_router(static_cast<std::size_t>(kExperts) * kHidden, 0.0F);
    std::array<int, kTopK> next_intended_ids{};
    for (int rank = 0; rank < kTopK; ++rank) {
        next_intended_ids[rank] = 510 - rank;
        next_router[static_cast<std::size_t>(next_intended_ids[rank]) * kHidden] =
            static_cast<float>(20 - rank);
    }
    const OracleRoute next_route = route_oracle(next_router, oracle_input);
    if (next_route.ids != next_intended_ids) {
        throw std::logic_error("back-to-back test router fixture is malformed");
    }

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
        const auto next_gate = make_matrix(routed_qtype, kIntermediate, kHidden,
                                           0x7000U + static_cast<unsigned>(rank));
        const auto next_up = make_matrix(routed_qtype, kIntermediate, kHidden,
                                         0x8000U + static_cast<unsigned>(rank));
        std::memcpy(mapped_gate.mutable_data() +
                        static_cast<std::size_t>(next_route.ids[rank]) * one_routed,
                    next_gate.data(), next_gate.size());
        std::memcpy(mapped_up.mutable_data() +
                        static_cast<std::size_t>(next_route.ids[rank]) * one_routed,
                    next_up.data(), next_up.size());
    }
    mapped_gate.make_read_only();
    mapped_up.make_read_only();

    DeviceBuffer resident_gate(bank_bytes);
    DeviceBuffer resident_up(bank_bytes);
    resident_gate.copy_from_host(mapped_gate.data(), mapped_gate.size());
    resident_up.copy_from_host(mapped_up.data(), mapped_up.size());

    const std::size_t one_down = matrix_bytes(QType::GGML_IQ4_NL, kHidden, kIntermediate);
    const std::size_t down_bank_bytes = static_cast<std::size_t>(kExperts) * one_down;
    DeviceBuffer device_down(down_bank_bytes);
    device_down.fill(0);
    std::array<std::vector<std::uint8_t>, kTopK> down_matrices;
    std::array<std::vector<std::uint8_t>, kTopK> next_down_matrices;
    for (int rank = 0; rank < kTopK; ++rank) {
        down_matrices[rank] = make_matrix(QType::GGML_IQ4_NL, kHidden, kIntermediate,
                                          0x3000U + static_cast<unsigned>(rank));
        device_down.copy_from_host(down_matrices[rank].data(), one_down,
                                   static_cast<std::size_t>(route.ids[rank]) * one_down);
        next_down_matrices[rank] = make_matrix(
            QType::GGML_IQ4_NL, kHidden, kIntermediate,
            0x9000U + static_cast<unsigned>(rank));
        device_down.copy_from_host(next_down_matrices[rank].data(), one_down,
                                   static_cast<std::size_t>(next_route.ids[rank]) * one_down);
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
        route.ids,
        shared_gate,
        shared_gate_proj,
        shared_up,
        shared_down,
    };
    const auto reference = complete_oracle(oracle_input, route, oracle_weights);
    const OracleInputs next_oracle_weights{
        routed_qtype,
        shared_qtype,
        next_router,
        mapped_gate,
        mapped_up,
        next_down_matrices,
        next_route.ids,
        shared_gate,
        shared_gate_proj,
        shared_up,
        shared_down,
    };
    const auto next_reference = complete_oracle(oracle_input, next_route, next_oracle_weights);

    DeviceBuffer device_x = to_device_bf16(input);
    DeviceBuffer device_router = to_device_f32(router);
    DeviceBuffer next_device_router = to_device_f32(next_router);
    DeviceBuffer device_shared_gate = to_device_f32(shared_gate);
    DeviceBuffer device_shared_gate_proj = to_device(shared_gate_proj);
    DeviceBuffer device_shared_up = to_device(shared_up);
    DeviceBuffer device_shared_down = to_device(shared_down);
    GuardedDeviceBuffer device_stage(ops::kQwen4SparseMoePipelineStageBytes);
    GuardedDeviceBuffer device_ids(kTopK * sizeof(std::int32_t));
    GuardedDeviceBuffer device_route_weights(kTopK * sizeof(float));
    GuardedDeviceBuffer device_destination(kHidden * sizeof(std::uint16_t));
    GuardedDeviceBuffer next_device_ids(kTopK * sizeof(std::int32_t));
    GuardedDeviceBuffer next_device_route_weights(kTopK * sizeof(float));
    GuardedDeviceBuffer next_device_destination(kHidden * sizeof(std::uint16_t));
    GuardedDeviceBuffer resident_device_ids(kTopK * sizeof(std::int32_t));
    GuardedDeviceBuffer resident_device_route_weights(kTopK * sizeof(float));
    GuardedDeviceBuffer resident_device_destination(kHidden * sizeof(std::uint16_t));
    GuardedDeviceBuffer next_resident_device_ids(kTopK * sizeof(std::int32_t));
    GuardedDeviceBuffer next_resident_device_route_weights(kTopK * sizeof(float));
    GuardedDeviceBuffer next_resident_device_destination(kHidden * sizeof(std::uint16_t));
    device_stage.fill(0xcd);
    device_ids.fill(0xcd);
    device_route_weights.fill(0xcd);
    device_destination.fill(0xcd);
    next_device_ids.fill(0xcd);
    next_device_route_weights.fill(0xcd);
    next_device_destination.fill(0xcd);
    resident_device_ids.fill(0xcd);
    resident_device_route_weights.fill(0xcd);
    resident_device_destination.fill(0xcd);
    next_resident_device_ids.fill(0xcd);
    next_resident_device_route_weights.fill(0xcd);
    next_resident_device_destination.fill(0xcd);
    cuda_synchronize();

    constexpr std::size_t host_guard = 256;
    PinnedHostBuffer pinned(ops::kQwen4SparseMoePipelineStageBytes + 2 * host_guard);
    auto* pinned_bytes = static_cast<std::uint8_t*>(pinned.data());
    std::memset(pinned_bytes, 0xa5, pinned.size());
    void* pinned_stage = pinned_bytes + host_guard;
    DeviceArena workspace(ops::qwen4_sparse_moe_workspace_capacity_bytes());

    Tensor x(device_x.p, DType::BF16, {kHidden});
    Tensor stage(device_stage.data(), DType::U8,
                 {static_cast<std::int32_t>(ops::kQwen4SparseMoePipelineStageBytes)});
    Tensor ids(device_ids.data(), DType::I32, {kTopK});
    Tensor route_weights(device_route_weights.data(), DType::FP32, {kTopK});
    Tensor destination(device_destination.data(), DType::BF16, {kHidden});
    Tensor next_ids(next_device_ids.data(), DType::I32, {kTopK});
    Tensor next_route_weights(next_device_route_weights.data(), DType::FP32, {kTopK});
    Tensor next_destination(next_device_destination.data(), DType::BF16, {kHidden});
    Tensor resident_ids(resident_device_ids.data(), DType::I32, {kTopK});
    Tensor resident_route_weights(resident_device_route_weights.data(), DType::FP32, {kTopK});
    Tensor resident_destination(resident_device_destination.data(), DType::BF16, {kHidden});
    Tensor next_resident_ids(next_resident_device_ids.data(), DType::I32, {kTopK});
    Tensor next_resident_route_weights(next_resident_device_route_weights.data(), DType::FP32,
                                       {kTopK});
    Tensor next_resident_destination(next_resident_device_destination.data(), DType::BF16,
                                     {kHidden});
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
    auto next_weights = weights;
    next_weights.router = make_router(next_device_router.p);
    ops::Qwen4ResidentSparseMoeWeights resident_weights{
        .router = weights.router,
        .routed_gate = make_ggml_weight(resident_gate.p, resident_gate.bytes, routed_qtype,
                                        kExperts, kIntermediate, kHidden),
        .routed_up = make_ggml_weight(resident_up.p, resident_up.bytes, routed_qtype,
                                      kExperts, kIntermediate, kHidden),
        .routed_down = weights.routed_down,
        .shared_gate = weights.shared_gate,
        .shared_gate_proj = weights.shared_gate_proj,
        .shared_up = weights.shared_up,
        .shared_down = weights.shared_down,
    };
    auto next_resident_weights = resident_weights;
    next_resident_weights.router = make_router(next_device_router.p);
    ops::Qwen4SparseMoePipeline pipeline{
        .pinned_stage = pinned_stage,
        .pinned_stage_bytes = ops::kQwen4SparseMoePipelineStageBytes,
        .device_stage = stage,
        .transfer_stream = pipeline_events.transfer_stream,
        .compute_stream = stream,
        .route_ready = pipeline_events.route_ready,
        .ids_ready = pipeline_events.ids_ready,
        .transfer_ready = {pipeline_events.transfer_ready[0],
                           pipeline_events.transfer_ready[1]},
        .consumer_complete = {pipeline_events.consumer_complete[0],
                              pipeline_events.consumer_complete[1]},
    };

    ops::qwen4_sparse_moe(x, weights, pipeline, ids, route_weights, destination, workspace,
                          stream);
    ops::qwen4_sparse_moe(x, next_weights, pipeline, next_ids, next_route_weights,
                          next_destination, workspace, stream);
    ops::qwen4_sparse_moe_resident(x, resident_weights, resident_ids,
                                   resident_route_weights, resident_destination, workspace,
                                   stream);
    ops::qwen4_sparse_moe_resident(x, next_resident_weights, next_resident_ids,
                                   next_resident_route_weights, next_resident_destination,
                                   workspace, stream);
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
    const auto next_actual_ids = from_device<std::int32_t>(next_device_ids.data(), kTopK);
    if (!std::equal(next_actual_ids.begin(), next_actual_ids.end(), next_route.ids.begin())) {
        std::cerr << "Qwen4 sparse MoE back-to-back call selected wrong expert ids\n";
        ++failures;
    }
    const auto next_actual_weight_f32 =
        from_device<float>(next_device_route_weights.data(), kTopK);
    std::vector<double> next_actual_weights(next_actual_weight_f32.begin(),
                                            next_actual_weight_f32.end());
    failures += verify_pointwise("Qwen4 sparse MoE back-to-back selected weights",
                                 next_actual_weights, next_route.weights,
                                 kRouteWeightCriterion);
    const auto next_actual = from_device_bf16(next_device_destination.data(), kHidden);
    failures += verify_reduction("Qwen4 sparse MoE back-to-back complete FP64 formula",
                                 next_actual, next_reference, kOutputCriterion);
    const auto resident_actual_ids =
        from_device<std::int32_t>(resident_device_ids.data(), kTopK);
    if (!std::equal(resident_actual_ids.begin(), resident_actual_ids.end(), route.ids.begin())) {
        std::cerr << "Qwen4 resident sparse MoE selected wrong expert ids\n";
        ++failures;
    }
    const auto resident_actual_weight_f32 =
        from_device<float>(resident_device_route_weights.data(), kTopK);
    std::vector<double> resident_actual_weights(resident_actual_weight_f32.begin(),
                                                resident_actual_weight_f32.end());
    failures += verify_pointwise("Qwen4 resident sparse MoE selected weights",
                                 resident_actual_weights, route.weights,
                                 kRouteWeightCriterion);
    const auto resident_actual =
        from_device_bf16(resident_device_destination.data(), kHidden);
    failures += verify_reduction("Qwen4 resident sparse MoE complete FP64 formula",
                                 resident_actual, reference, kOutputCriterion);
    if (!std::equal(actual.begin(), actual.end(), resident_actual.begin())) {
        std::cerr << "Qwen4 resident fusion changed a routed BF16 seam\n";
        ++failures;
    }
    const auto next_resident_actual_ids =
        from_device<std::int32_t>(next_resident_device_ids.data(), kTopK);
    if (!std::equal(next_resident_actual_ids.begin(), next_resident_actual_ids.end(),
                    next_route.ids.begin())) {
        std::cerr << "Qwen4 resident sparse MoE back-to-back call selected wrong expert ids\n";
        ++failures;
    }
    const auto next_resident_actual_weight_f32 =
        from_device<float>(next_resident_device_route_weights.data(), kTopK);
    std::vector<double> next_resident_actual_weights(next_resident_actual_weight_f32.begin(),
                                                     next_resident_actual_weight_f32.end());
    failures += verify_pointwise("Qwen4 resident sparse MoE back-to-back selected weights",
                                 next_resident_actual_weights, next_route.weights,
                                 kRouteWeightCriterion);
    const auto next_resident_actual =
        from_device_bf16(next_resident_device_destination.data(), kHidden);
    failures += verify_reduction("Qwen4 resident sparse MoE back-to-back complete FP64 formula",
                                 next_resident_actual, next_reference, kOutputCriterion);
    if (!std::equal(next_actual.begin(), next_actual.end(), next_resident_actual.begin())) {
        std::cerr << "Qwen4 resident back-to-back fusion changed a routed BF16 seam\n";
        ++failures;
    }

    if (route_fixture != RouteFixture::Underflow) {
        constexpr int prefill_width = 17;
        std::vector<float> prefill_input;
        prefill_input.reserve(static_cast<std::size_t>(kHidden) * prefill_width);
        std::vector<double> prefill_reference;
        prefill_reference.reserve(static_cast<std::size_t>(kHidden) * prefill_width);
        std::vector<double> next_prefill_reference;
        next_prefill_reference.reserve(static_cast<std::size_t>(kHidden) * prefill_width);
        for (int token = 0; token < prefill_width; ++token) {
            prefill_input.insert(prefill_input.end(), input.begin(), input.end());
            prefill_reference.insert(prefill_reference.end(), reference.begin(), reference.end());
            next_prefill_reference.insert(next_prefill_reference.end(), next_reference.begin(),
                                          next_reference.end());
        }

        DeviceBuffer prefill_x = to_device_bf16(prefill_input);
        GuardedDeviceBuffer prefill_stage(ops::kQwen4SparseMoePrefillPipelineStageBytes);
        GuardedDeviceBuffer prefill_ids(static_cast<std::size_t>(kTopK) * prefill_width *
                                        sizeof(std::int32_t));
        GuardedDeviceBuffer prefill_weights(static_cast<std::size_t>(kTopK) * prefill_width *
                                            sizeof(float));
        GuardedDeviceBuffer prefill_output(static_cast<std::size_t>(kHidden) * prefill_width *
                                           sizeof(std::uint16_t));
        GuardedDeviceBuffer next_prefill_ids(static_cast<std::size_t>(kTopK) * prefill_width *
                                             sizeof(std::int32_t));
        GuardedDeviceBuffer next_prefill_weights(static_cast<std::size_t>(kTopK) * prefill_width *
                                                 sizeof(float));
        GuardedDeviceBuffer next_prefill_output(static_cast<std::size_t>(kHidden) * prefill_width *
                                                sizeof(std::uint16_t));
        prefill_stage.fill(0xcd);
        PinnedHostBuffer prefill_pinned(ops::kQwen4SparseMoePrefillPipelineStageBytes);
        PinnedHostBuffer prefill_host_scratch(
            ops::kQwen4SparseMoePrefillHostScratchBytes);
        std::memset(prefill_pinned.data(), 0xa5, prefill_pinned.size());
        DeviceArena prefill_workspace(
            ops::qwen4_sparse_moe_prefill_workspace_capacity_bytes(prefill_width));
        Tensor prefill_x_tensor(prefill_x.p, DType::BF16, {kHidden, prefill_width});
        Tensor prefill_stage_tensor(
            prefill_stage.data(), DType::U8,
            {static_cast<std::int32_t>(ops::kQwen4SparseMoePrefillPipelineStageBytes)});
        Tensor prefill_ids_tensor(prefill_ids.data(), DType::I32, {kTopK, prefill_width});
        Tensor prefill_weights_tensor(prefill_weights.data(), DType::FP32,
                                      {kTopK, prefill_width});
        Tensor prefill_output_tensor(prefill_output.data(), DType::BF16,
                                     {kHidden, prefill_width});
        Tensor next_prefill_ids_tensor(next_prefill_ids.data(), DType::I32,
                                       {kTopK, prefill_width});
        Tensor next_prefill_weights_tensor(next_prefill_weights.data(), DType::FP32,
                                           {kTopK, prefill_width});
        Tensor next_prefill_output_tensor(next_prefill_output.data(), DType::BF16,
                                          {kHidden, prefill_width});
        ops::Qwen4SparseMoePrefillPipeline prefill_pipeline{
            .pinned_stage = prefill_pinned.data(),
            .pinned_stage_bytes = prefill_pinned.size(),
            .host_scratch = prefill_host_scratch.data(),
            .host_scratch_bytes = prefill_host_scratch.size(),
            .device_stage = prefill_stage_tensor,
            .transfer_stream = pipeline_events.transfer_stream,
            .compute_stream = stream,
            .route_ready = pipeline_events.route_ready,
            .ids_ready = pipeline_events.ids_ready,
            .transfer_ready = {pipeline_events.transfer_ready[0],
                               pipeline_events.transfer_ready[1]},
            .consumer_complete = {pipeline_events.consumer_complete[0],
                                  pipeline_events.consumer_complete[1]},
        };
        ops::qwen4_sparse_moe_prefill(prefill_x_tensor, weights, prefill_pipeline,
                                      prefill_ids_tensor, prefill_weights_tensor,
                                      prefill_output_tensor, prefill_workspace, stream);
        ops::qwen4_sparse_moe_prefill(prefill_x_tensor, next_weights, prefill_pipeline,
                                      next_prefill_ids_tensor, next_prefill_weights_tensor,
                                      next_prefill_output_tensor, prefill_workspace, stream);
        cuda_synchronize(stream);

        std::vector<std::int32_t> expected_ids(static_cast<std::size_t>(kTopK) * prefill_width);
        std::vector<double> expected_weights(static_cast<std::size_t>(kTopK) * prefill_width);
        std::vector<std::int32_t> expected_next_ids(expected_ids.size());
        std::vector<double> expected_next_weights(expected_weights.size());
        for (int token = 0; token < prefill_width; ++token) {
            std::copy(route.ids.begin(), route.ids.end(),
                      expected_ids.begin() + static_cast<std::ptrdiff_t>(token * kTopK));
            std::copy(route.weights.begin(), route.weights.end(),
                      expected_weights.begin() + static_cast<std::ptrdiff_t>(token * kTopK));
            std::copy(next_route.ids.begin(), next_route.ids.end(),
                      expected_next_ids.begin() + static_cast<std::ptrdiff_t>(token * kTopK));
            std::copy(next_route.weights.begin(), next_route.weights.end(),
                      expected_next_weights.begin() + static_cast<std::ptrdiff_t>(token * kTopK));
        }
        failures += verify_exact(
            "Qwen4 sparse MoE prefill duplicate-token ids",
            from_device<std::int32_t>(prefill_ids.data(), expected_ids.size()), expected_ids);
        failures += verify_pointwise(
            "Qwen4 sparse MoE prefill duplicate-token weights",
            [&] {
                const auto values = from_device<float>(prefill_weights.data(),
                                                       expected_weights.size());
                return std::vector<double>(values.begin(), values.end());
            }(),
            expected_weights, kRouteWeightCriterion);
        const auto prefill_actual =
            from_device_bf16(prefill_output.data(), prefill_reference.size());
        failures += verify_reduction(
            "Qwen4 sparse MoE prefill complete FP64 formula",
            prefill_actual, prefill_reference, kOutputCriterion);
        for (int token : {15, 16}) {
            failures += verify_reduction(
                token == 15
                    ? "Qwen4 sparse MoE prefill occurrence column 15 FP64 formula"
                    : "Qwen4 sparse MoE prefill occurrence column 16 FP64 formula",
                std::span<const double>(prefill_actual).subspan(
                    static_cast<std::size_t>(token) * kHidden, kHidden),
                reference, kOutputCriterion);
        }
        failures += verify_exact(
            "Qwen4 sparse MoE prefill back-to-back ids",
            from_device<std::int32_t>(next_prefill_ids.data(), expected_next_ids.size()),
            expected_next_ids);
        failures += verify_pointwise(
            "Qwen4 sparse MoE prefill back-to-back weights",
            [&] {
                const auto values = from_device<float>(next_prefill_weights.data(),
                                                       expected_next_weights.size());
                return std::vector<double>(values.begin(), values.end());
            }(),
            expected_next_weights, kRouteWeightCriterion);
        failures += verify_reduction(
            "Qwen4 sparse MoE prefill back-to-back complete FP64 formula",
            from_device_bf16(next_prefill_output.data(), next_prefill_reference.size()),
            next_prefill_reference, kOutputCriterion);

        std::vector<int> sorted_experts(next_route.ids.begin(), next_route.ids.end());
        std::sort(sorted_experts.begin(), sorted_experts.end());
        const std::size_t rank_bytes = ops::qwen4_sparse_moe_rank_stage_bytes(routed_qtype);
        std::vector<std::uint8_t> staged_pair(rank_bytes);
        auto* pinned_prefill = static_cast<const std::uint8_t*>(prefill_pinned.data());
        for (std::size_t local = 0; local < sorted_experts.size(); ++local) {
            const int expert = sorted_experts[local];
            std::memcpy(staged_pair.data(), mapped_gate.data() +
                                                static_cast<std::size_t>(expert) * one_routed,
                        one_routed);
            std::memcpy(staged_pair.data() + one_routed,
                        mapped_up.data() + static_cast<std::size_t>(expert) * one_routed,
                        one_routed);
            if (std::memcmp(pinned_prefill + local * rank_bytes, staged_pair.data(),
                            rank_bytes) != 0) {
                std::cerr << "Qwen4 sparse MoE prefill unique staging bytes changed\n";
                ++failures;
            }
            const auto device_pair = from_device<std::uint8_t>(
                static_cast<const std::uint8_t*>(prefill_stage.data()) + local * rank_bytes,
                rank_bytes);
            failures += verify_exact("Qwen4 sparse MoE prefill device staging bytes",
                                     device_pair, staged_pair);
        }
        std::vector<std::int32_t> expected_occurrences;
        for (int expert : sorted_experts) {
            const auto iterator = std::find(next_route.ids.begin(), next_route.ids.end(), expert);
            const int rank = static_cast<int>(iterator - next_route.ids.begin());
            for (int token = 0; token < prefill_width; ++token) {
                expected_occurrences.push_back(rank + kTopK * token);
            }
        }
        const auto* pinned_occurrences = reinterpret_cast<const std::int32_t*>(
            pinned_prefill + sorted_experts.size() * rank_bytes);
        if (!std::equal(expected_occurrences.begin(), expected_occurrences.end(),
                        pinned_occurrences)) {
            std::cerr << "Qwen4 sparse MoE prefill occurrence grouping changed\n";
            ++failures;
        }
        const auto device_occurrences = from_device<std::int32_t>(
            static_cast<const std::uint8_t*>(prefill_stage.data()) +
                sorted_experts.size() * rank_bytes,
            expected_occurrences.size());
        failures += verify_exact("Qwen4 sparse MoE prefill device occurrence grouping",
                                 device_occurrences, expected_occurrences);
        failures += prefill_stage.verify_guards("Qwen4 sparse MoE prefill stage");
        failures += prefill_ids.verify_guards("Qwen4 sparse MoE prefill ids");
        failures += prefill_weights.verify_guards("Qwen4 sparse MoE prefill weights");
        failures += prefill_output.verify_guards("Qwen4 sparse MoE prefill destination");
        failures += next_prefill_ids.verify_guards(
            "Qwen4 sparse MoE prefill back-to-back ids");
        failures += next_prefill_weights.verify_guards(
            "Qwen4 sparse MoE prefill back-to-back weights");
        failures += next_prefill_output.verify_guards(
            "Qwen4 sparse MoE prefill back-to-back destination");

        Tensor bad_prefill_ids(prefill_ids.data(), DType::I32, {kTopK, 1});
        failures += expect_invalid(
            [&] {
                ops::qwen4_sparse_moe_prefill(
                    prefill_x_tensor, weights, prefill_pipeline, bad_prefill_ids,
                    prefill_weights_tensor, prefill_output_tensor, prefill_workspace, stream);
            },
            "mismatched prefill route width");
        auto short_prefill_pipeline = prefill_pipeline;
        --short_prefill_pipeline.pinned_stage_bytes;
        failures += expect_invalid(
            [&] {
                ops::qwen4_sparse_moe_prefill(
                    prefill_x_tensor, weights, short_prefill_pipeline, prefill_ids_tensor,
                    prefill_weights_tensor, prefill_output_tensor, prefill_workspace, stream);
            },
            "short prefill pinned stage");
        PinnedHostBuffer misaligned_scratch(
            ops::kQwen4SparseMoePrefillHostScratchBytes + 1);
        auto misaligned_prefill_pipeline = prefill_pipeline;
        misaligned_prefill_pipeline.host_scratch =
            static_cast<std::byte*>(misaligned_scratch.data()) + 1;
        failures += expect_invalid(
            [&] {
                ops::qwen4_sparse_moe_prefill(
                    prefill_x_tensor, weights, misaligned_prefill_pipeline,
                    prefill_ids_tensor, prefill_weights_tensor, prefill_output_tensor,
                    prefill_workspace, stream);
            },
            "misaligned prefill host scratch");

        const std::size_t prefill_required =
            ops::qwen4_sparse_moe_prefill_workspace_capacity_bytes(prefill_width);
        DeviceArena offset_prefill_workspace(prefill_required + 256);
        (void)offset_prefill_workspace.alloc_bytes(1, 1);
        ops::qwen4_sparse_moe_prefill(
            prefill_x_tensor, weights, prefill_pipeline, prefill_ids_tensor,
            prefill_weights_tensor, prefill_output_tensor, offset_prefill_workspace, stream);
        cuda_synchronize(stream);
        if (offset_prefill_workspace.used() != 1) {
            std::cerr << "Qwen4 sparse MoE prefill did not restore an offset workspace\n";
            ++failures;
        }
        DeviceArena short_offset_prefill_workspace(prefill_required + 255);
        (void)short_offset_prefill_workspace.alloc_bytes(1, 1);
        failures += expect_invalid(
            [&] {
                ops::qwen4_sparse_moe_prefill(
                    prefill_x_tensor, weights, prefill_pipeline, prefill_ids_tensor,
                    prefill_weights_tensor, prefill_output_tensor,
                    short_offset_prefill_workspace, stream);
            },
            "prefill workspace missing alignment padding");

        DeviceBuffer unaligned_workspace_backing(prefill_required + 256);
        DeviceArena unaligned_prefill_workspace(DeviceSpan{
            static_cast<std::byte*>(unaligned_workspace_backing.p) + 1,
            prefill_required + 255});
        (void)unaligned_prefill_workspace.alloc_bytes(1, 1);
        ops::qwen4_sparse_moe_prefill(
            prefill_x_tensor, weights, prefill_pipeline, prefill_ids_tensor,
            prefill_weights_tensor, prefill_output_tensor, unaligned_prefill_workspace,
            stream);
        cuda_synchronize(stream);
        if (unaligned_prefill_workspace.used() != 1) {
            std::cerr << "Qwen4 sparse MoE prefill did not restore an unaligned workspace\n";
            ++failures;
        }
        DeviceBuffer short_unaligned_workspace_backing(prefill_required + 255);
        DeviceArena short_unaligned_prefill_workspace(DeviceSpan{
            static_cast<std::byte*>(short_unaligned_workspace_backing.p) + 1,
            prefill_required + 254});
        (void)short_unaligned_prefill_workspace.alloc_bytes(1, 1);
        failures += expect_invalid(
            [&] {
                ops::qwen4_sparse_moe_prefill(
                    prefill_x_tensor, weights, prefill_pipeline, prefill_ids_tensor,
                    prefill_weights_tensor, prefill_output_tensor,
                    short_unaligned_prefill_workspace, stream);
            },
            "prefill workspace missing absolute-address alignment padding");

        constexpr int small_resident_width = 2;
        std::vector<float> small_resident_input;
        small_resident_input.reserve(static_cast<std::size_t>(kHidden) * small_resident_width);
        small_resident_input.insert(small_resident_input.end(), input.begin(), input.end());
        small_resident_input.insert(small_resident_input.end(), input.begin(), input.end());
        small_resident_input[static_cast<std::size_t>(kHidden) + 1] += 0.125F;
        round_to_bf16(small_resident_input);
        std::vector<double> second_small_oracle_input(
            small_resident_input.begin() + kHidden, small_resident_input.end());
        const OracleRoute second_small_route = route_oracle(router, second_small_oracle_input);
        if (second_small_route.ids != route.ids) {
            throw std::logic_error("small resident panel route fixture is malformed");
        }
        const auto second_small_reference = complete_oracle(
            second_small_oracle_input, second_small_route, oracle_weights);
        DeviceBuffer small_resident_x = to_device_bf16(small_resident_input);
        GuardedDeviceBuffer small_resident_ids(
            small_resident_width * kTopK * sizeof(std::int32_t));
        GuardedDeviceBuffer small_resident_weights(
            small_resident_width * kTopK * sizeof(float));
        GuardedDeviceBuffer small_resident_output(
            small_resident_width * kHidden * sizeof(std::uint16_t));
        DeviceArena small_resident_workspace(
            ops::qwen4_sparse_moe_resident_workspace_capacity_bytes(
                small_resident_width));
        Tensor small_resident_x_tensor(small_resident_x.p, DType::BF16,
                                       {kHidden, small_resident_width});
        Tensor small_resident_ids_tensor(small_resident_ids.data(), DType::I32,
                                         {kTopK, small_resident_width});
        Tensor small_resident_weights_tensor(small_resident_weights.data(), DType::FP32,
                                             {kTopK, small_resident_width});
        Tensor small_resident_output_tensor(small_resident_output.data(), DType::BF16,
                                            {kHidden, small_resident_width});
        ops::qwen4_sparse_moe_resident(
            small_resident_x_tensor, resident_weights, small_resident_ids_tensor,
            small_resident_weights_tensor, small_resident_output_tensor,
            small_resident_workspace, stream);
        cuda_synchronize(stream);
        std::vector<std::int32_t> small_expected_ids(2 * kTopK);
        std::copy(route.ids.begin(), route.ids.end(), small_expected_ids.begin());
        std::copy(second_small_route.ids.begin(), second_small_route.ids.end(),
                  small_expected_ids.begin() + kTopK);
        failures += verify_exact(
            "Qwen4 resident small-T distinct-input ids",
            from_device<std::int32_t>(small_resident_ids.data(), small_expected_ids.size()),
            small_expected_ids);
        std::vector<double> small_expected_weights(2 * kTopK);
        std::copy(route.weights.begin(), route.weights.end(), small_expected_weights.begin());
        std::copy(second_small_route.weights.begin(), second_small_route.weights.end(),
                  small_expected_weights.begin() + kTopK);
        const auto small_weight_f32 = from_device<float>(
            small_resident_weights.data(), small_expected_weights.size());
        failures += verify_pointwise(
            "Qwen4 resident small-T distinct-input weights",
            std::vector<double>(small_weight_f32.begin(), small_weight_f32.end()),
            small_expected_weights, kRouteWeightCriterion);
        const auto small_actual = from_device_bf16(
            small_resident_output.data(), small_resident_width * kHidden);
        failures += verify_reduction(
            "Qwen4 resident small-T first-token complete FP64 formula",
            std::span<const double>(small_actual).first(kHidden), reference,
            kOutputCriterion);
        failures += verify_reduction(
            "Qwen4 resident small-T second-token complete FP64 formula",
            std::span<const double>(small_actual).subspan(kHidden, kHidden),
            second_small_reference, kOutputCriterion);
        failures += small_resident_ids.verify_guards("Qwen4 resident small-T ids");
        failures += small_resident_weights.verify_guards("Qwen4 resident small-T weights");
        failures += small_resident_output.verify_guards("Qwen4 resident small-T output");

        // Qualify the device-grouped resident route directly against the independent formula.
        // Four represented inputs alternate: two select the same experts with different values,
        // one selects a disjoint expert set, and one selects the first set in reversed rank order.
        // This catches token/rank coupling, expert grouping, and result-scatter mistakes that
        // identical-token or fixed-rank panels cannot expose.
        constexpr int resident_width = 257;
        std::vector<float> resident_router(static_cast<std::size_t>(kExperts) * kHidden, 0.0F);
        for (int rank = 0; rank < kTopK; ++rank) {
            resident_router[static_cast<std::size_t>(route.ids[rank]) * kHidden] =
                static_cast<float>(20 - rank);
            resident_router[static_cast<std::size_t>(route.ids[rank]) * kHidden + 1] =
                static_cast<float>(20 - rank);
            resident_router[static_cast<std::size_t>(next_route.ids[rank]) * kHidden + 2] =
                static_cast<float>(20 - rank);
            resident_router[static_cast<std::size_t>(route.ids[kTopK - 1 - rank]) * kHidden +
                            3] = static_cast<float>(20 - rank);
        }
        std::array<std::vector<float>, 4> resident_inputs;
        resident_inputs[0].assign(kHidden, 0.0F);
        resident_inputs[1].assign(kHidden, 0.0F);
        resident_inputs[2].assign(kHidden, 0.0F);
        resident_inputs[3].assign(kHidden, 0.0F);
        resident_inputs[0][0] = 1.0F;
        resident_inputs[1][1] = 0.75F;
        resident_inputs[2][2] = 1.0F;
        resident_inputs[3][3] = 0.5F;
        for (auto& value : resident_inputs) { round_to_bf16(value); }
        std::array<std::vector<double>, 4> resident_oracle_inputs;
        std::array<OracleRoute, 4> resident_routes;
        for (int pattern = 0; pattern < 4; ++pattern) {
            resident_oracle_inputs[pattern].assign(resident_inputs[pattern].begin(),
                                                   resident_inputs[pattern].end());
            resident_routes[pattern] = route_oracle(resident_router,
                                                    resident_oracle_inputs[pattern]);
        }
        auto reverse_route_ids = route.ids;
        std::reverse(reverse_route_ids.begin(), reverse_route_ids.end());
        if (resident_routes[0].ids != route.ids || resident_routes[1].ids != route.ids ||
            resident_routes[2].ids != next_route.ids ||
            resident_routes[3].ids != reverse_route_ids) {
            throw std::logic_error("resident grouped route fixture is malformed");
        }
        const OracleInputs resident_oracle_weights{
            routed_qtype, shared_qtype, resident_router, mapped_gate, mapped_up,
            down_matrices, route.ids, shared_gate, shared_gate_proj, shared_up, shared_down};
        const OracleInputs resident_next_oracle_weights{
            routed_qtype, shared_qtype, resident_router, mapped_gate, mapped_up,
            next_down_matrices, next_route.ids, shared_gate, shared_gate_proj, shared_up,
            shared_down};
        std::array<std::vector<double>, 4> resident_references{
            complete_oracle(resident_oracle_inputs[0], resident_routes[0],
                            resident_oracle_weights),
            complete_oracle(resident_oracle_inputs[1], resident_routes[1],
                            resident_oracle_weights),
            complete_oracle(resident_oracle_inputs[2], resident_routes[2],
                            resident_next_oracle_weights),
            complete_oracle(resident_oracle_inputs[3], resident_routes[3],
                            resident_oracle_weights),
        };

        std::vector<float> resident_panel;
        resident_panel.reserve(static_cast<std::size_t>(resident_width) * kHidden);
        std::vector<std::int32_t> resident_expected_ids(
            static_cast<std::size_t>(resident_width) * kTopK);
        std::vector<double> resident_expected_weights(
            static_cast<std::size_t>(resident_width) * kTopK);
        for (int token = 0; token < resident_width; ++token) {
            const int pattern = token % 4;
            resident_panel.insert(resident_panel.end(), resident_inputs[pattern].begin(),
                                  resident_inputs[pattern].end());
            std::copy(resident_routes[pattern].ids.begin(), resident_routes[pattern].ids.end(),
                      resident_expected_ids.begin() + static_cast<std::ptrdiff_t>(token * kTopK));
            std::copy(resident_routes[pattern].weights.begin(),
                      resident_routes[pattern].weights.end(),
                      resident_expected_weights.begin() +
                          static_cast<std::ptrdiff_t>(token * kTopK));
        }
        DeviceBuffer resident_panel_device = to_device_bf16(resident_panel);
        DeviceBuffer resident_router_device = to_device_f32(resident_router);
        GuardedDeviceBuffer resident_panel_ids(
            static_cast<std::size_t>(resident_width) * kTopK * sizeof(std::int32_t));
        GuardedDeviceBuffer resident_panel_weights(
            static_cast<std::size_t>(resident_width) * kTopK * sizeof(float));
        GuardedDeviceBuffer resident_panel_output(
            static_cast<std::size_t>(resident_width) * kHidden * sizeof(std::uint16_t));
        GuardedDeviceBuffer resident_panel_second_output(
            static_cast<std::size_t>(resident_width) * kHidden * sizeof(std::uint16_t));
        DeviceArena resident_panel_workspace(
            ops::qwen4_sparse_moe_resident_workspace_capacity_bytes(resident_width));
        Tensor resident_panel_x(resident_panel_device.p, DType::BF16,
                                {kHidden, resident_width});
        Tensor resident_panel_ids_tensor(resident_panel_ids.data(), DType::I32,
                                          {kTopK, resident_width});
        Tensor resident_panel_weights_tensor(resident_panel_weights.data(), DType::FP32,
                                              {kTopK, resident_width});
        Tensor resident_panel_output_tensor(resident_panel_output.data(), DType::BF16,
                                             {kHidden, resident_width});
        Tensor resident_panel_second_output_tensor(
            resident_panel_second_output.data(), DType::BF16, {kHidden, resident_width});
        auto resident_panel_model_weights = resident_weights;
        resident_panel_model_weights.router = make_router(resident_router_device.p);
        ops::qwen4_sparse_moe_resident(
            resident_panel_x, resident_panel_model_weights, resident_panel_ids_tensor,
            resident_panel_weights_tensor, resident_panel_output_tensor,
            resident_panel_workspace, stream);
        ops::qwen4_sparse_moe_resident(
            resident_panel_x, resident_panel_model_weights, resident_panel_ids_tensor,
            resident_panel_weights_tensor, resident_panel_second_output_tensor,
            resident_panel_workspace, stream);
        cuda_synchronize(stream);

        failures += verify_exact(
            "Qwen4 resident grouped heterogeneous ids",
            from_device<std::int32_t>(resident_panel_ids.data(),
                                      resident_expected_ids.size()),
            resident_expected_ids);
        const auto resident_panel_weight_f32 = from_device<float>(
            resident_panel_weights.data(), resident_expected_weights.size());
        failures += verify_pointwise(
            "Qwen4 resident grouped heterogeneous weights",
            std::vector<double>(resident_panel_weight_f32.begin(),
                                resident_panel_weight_f32.end()),
            resident_expected_weights, kRouteWeightCriterion);
        const auto resident_panel_actual = from_device_bf16(
            resident_panel_output.data(), static_cast<std::size_t>(resident_width) * kHidden);
        const auto resident_panel_second_actual = from_device_bf16(
            resident_panel_second_output.data(),
            static_cast<std::size_t>(resident_width) * kHidden);
        if (resident_panel_actual != resident_panel_second_actual) {
            std::cerr << "Qwen4 resident grouped back-to-back output is nondeterministic\n";
            ++failures;
        }
        for (int token = 0; token < resident_width; ++token) {
            const int pattern = token % 4;
            const auto begin = static_cast<std::size_t>(token) * kHidden;
            failures += verify_reduction(
                "Qwen4 resident grouped per-token complete FP64 formula",
                std::span<const double>(resident_panel_actual).subspan(begin, kHidden),
                resident_references[pattern], kOutputCriterion);
            failures += verify_reduction(
                "Qwen4 resident grouped back-to-back complete FP64 formula",
                std::span<const double>(resident_panel_second_actual).subspan(begin, kHidden),
                resident_references[pattern], kOutputCriterion);
        }
        failures += resident_panel_ids.verify_guards(
            "Qwen4 resident grouped heterogeneous ids");
        failures += resident_panel_weights.verify_guards(
            "Qwen4 resident grouped heterogeneous weights");
        failures += resident_panel_output.verify_guards(
            "Qwen4 resident grouped heterogeneous output");
        failures += resident_panel_second_output.verify_guards(
            "Qwen4 resident grouped back-to-back output");

        DeviceArena short_resident_panel_workspace(
            ops::qwen4_sparse_moe_resident_workspace_capacity_bytes(resident_width) - 1);
        failures += expect_invalid(
            [&] {
                ops::qwen4_sparse_moe_resident(
                    resident_panel_x, resident_panel_model_weights, resident_panel_ids_tensor,
                    resident_panel_weights_tensor, resident_panel_output_tensor,
                    short_resident_panel_workspace, stream);
            },
            "short resident grouped workspace");
        Tensor wrong_resident_panel_ids(resident_panel_ids.data(), DType::I32,
                                        {kTopK, resident_width - 1});
        failures += expect_invalid(
            [&] {
                ops::qwen4_sparse_moe_resident(
                    resident_panel_x, resident_panel_model_weights,
                    wrong_resident_panel_ids, resident_panel_weights_tensor,
                    resident_panel_output_tensor, resident_panel_workspace, stream);
            },
            "mismatched resident grouped route width");

        if (routed_qtype == QType::GGML_IQ1_S &&
            route_fixture == RouteFixture::AllZero) {
            constexpr int maximum_width = ops::kQwen4SparseMoePrefillMaxWidth;
            DeviceBuffer maximum_x(static_cast<std::size_t>(maximum_width) * kHidden *
                                   sizeof(std::uint16_t));
            maximum_x.fill(0);
            GuardedDeviceBuffer maximum_ids(
                static_cast<std::size_t>(maximum_width) * kTopK * sizeof(std::int32_t));
            GuardedDeviceBuffer maximum_weights(
                static_cast<std::size_t>(maximum_width) * kTopK * sizeof(float));
            GuardedDeviceBuffer maximum_output(
                static_cast<std::size_t>(maximum_width) * kHidden * sizeof(std::uint16_t));
            DeviceArena maximum_workspace(
                ops::qwen4_sparse_moe_resident_workspace_capacity_bytes(maximum_width));
            Tensor maximum_x_tensor(maximum_x.p, DType::BF16,
                                    {kHidden, maximum_width});
            Tensor maximum_ids_tensor(maximum_ids.data(), DType::I32,
                                      {kTopK, maximum_width});
            Tensor maximum_weights_tensor(maximum_weights.data(), DType::FP32,
                                          {kTopK, maximum_width});
            Tensor maximum_output_tensor(maximum_output.data(), DType::BF16,
                                         {kHidden, maximum_width});
            ops::qwen4_sparse_moe_resident(
                maximum_x_tensor, resident_weights, maximum_ids_tensor,
                maximum_weights_tensor, maximum_output_tensor, maximum_workspace, stream);
            cuda_synchronize(stream);
            std::vector<std::int32_t> maximum_expected_ids(
                static_cast<std::size_t>(maximum_width) * kTopK);
            for (int token = 0; token < maximum_width; ++token) {
                std::iota(maximum_expected_ids.begin() +
                              static_cast<std::ptrdiff_t>(token * kTopK),
                          maximum_expected_ids.begin() +
                              static_cast<std::ptrdiff_t>((token + 1) * kTopK),
                          0);
            }
            failures += verify_exact(
                "Qwen4 resident maximum-width tie ids",
                from_device<std::int32_t>(maximum_ids.data(),
                                          maximum_expected_ids.size()),
                maximum_expected_ids);
            const auto maximum_actual = from_device<std::uint16_t>(
                maximum_output.data(), static_cast<std::size_t>(maximum_width) * kHidden);
            if (!std::all_of(maximum_actual.begin(), maximum_actual.end(),
                             [](std::uint16_t value) { return value == 0; })) {
                std::cerr << "Qwen4 resident maximum-width zero formula changed\n";
                ++failures;
            }
            failures += maximum_ids.verify_guards("Qwen4 resident maximum-width ids");
            failures += maximum_weights.verify_guards(
                "Qwen4 resident maximum-width weights");
            failures += maximum_output.verify_guards("Qwen4 resident maximum-width output");
        }
    }

    const std::size_t rank_bytes = ops::qwen4_sparse_moe_rank_stage_bytes(routed_qtype);
    std::vector<std::uint8_t> actual_stage(ops::kQwen4SparseMoePipelineStageBytes);
    device_stage.copy_to_host(actual_stage.data(), actual_stage.size());
    for (int slot = 0; slot < ops::kQwen4SparseMoePipelineSlots; ++slot) {
        const int rank = kTopK - ops::kQwen4SparseMoePipelineSlots + slot;
        std::vector<std::uint8_t> expected_pair(rank_bytes);
        std::memcpy(expected_pair.data(),
                    mapped_gate.data() +
                        static_cast<std::size_t>(next_route.ids[rank]) * one_routed,
                    one_routed);
        std::memcpy(expected_pair.data() + one_routed,
                    mapped_up.data() +
                        static_cast<std::size_t>(next_route.ids[rank]) * one_routed,
                    one_routed);
        const std::size_t slot_offset = static_cast<std::size_t>(slot) *
                                        ops::kQwen4SparseMoeRankStageCapacityBytes;
        if (std::memcmp(pinned_bytes + host_guard + slot_offset, expected_pair.data(),
                        rank_bytes) != 0 ||
            !std::equal(expected_pair.begin(), expected_pair.end(),
                        actual_stage.begin() + static_cast<std::ptrdiff_t>(slot_offset))) {
            std::cerr << "Qwen4 sparse MoE rank-pipeline bytes changed\n";
            ++failures;
        }
        if (!std::all_of(pinned_bytes + host_guard + slot_offset + rank_bytes,
                         pinned_bytes + host_guard + slot_offset +
                             ops::kQwen4SparseMoeRankStageCapacityBytes,
                         [](std::uint8_t value) { return value == 0xa5; }) ||
            !std::all_of(actual_stage.begin() +
                             static_cast<std::ptrdiff_t>(slot_offset + rank_bytes),
                         actual_stage.begin() + static_cast<std::ptrdiff_t>(
                             slot_offset + ops::kQwen4SparseMoeRankStageCapacityBytes),
                         [](std::uint8_t value) { return value == 0xcd; })) {
            std::cerr << "Qwen4 sparse MoE copied beyond one rank slot\n";
            ++failures;
        }
    }
    if (!std::all_of(pinned_bytes, pinned_bytes + host_guard,
                     [](std::uint8_t value) { return value == 0xa5; }) ||
        !std::all_of(pinned_bytes + host_guard + ops::kQwen4SparseMoePipelineStageBytes,
                     pinned_bytes + pinned.size(),
                     [](std::uint8_t value) { return value == 0xa5; })) {
        std::cerr << "Qwen4 sparse MoE pinned stage guard mismatch\n";
        ++failures;
    }

    failures += device_stage.verify_guards("Qwen4 sparse MoE device stage");
    failures += device_ids.verify_guards("Qwen4 sparse MoE selected ids");
    failures += device_route_weights.verify_guards("Qwen4 sparse MoE selected weights");
    failures += device_destination.verify_guards("Qwen4 sparse MoE destination");
    failures += next_device_ids.verify_guards("Qwen4 sparse MoE back-to-back selected ids");
    failures += next_device_route_weights.verify_guards(
        "Qwen4 sparse MoE back-to-back selected weights");
    failures += next_device_destination.verify_guards(
        "Qwen4 sparse MoE back-to-back destination");
    failures += resident_device_ids.verify_guards("Qwen4 resident sparse MoE selected ids");
    failures += resident_device_route_weights.verify_guards(
        "Qwen4 resident sparse MoE selected weights");
    failures += resident_device_destination.verify_guards(
        "Qwen4 resident sparse MoE destination");
    failures += next_resident_device_ids.verify_guards(
        "Qwen4 resident sparse MoE back-to-back selected ids");
    failures += next_resident_device_route_weights.verify_guards(
        "Qwen4 resident sparse MoE back-to-back selected weights");
    failures += next_resident_device_destination.verify_guards(
        "Qwen4 resident sparse MoE back-to-back destination");

    auto invalid_bank = weights;
    invalid_bank.routed_gate_up.gate = invalid_bank.routed_gate_up.gate.first(
        invalid_bank.routed_gate_up.gate.size() - 1);
    failures += expect_invalid(
        [&] {
            ops::qwen4_sparse_moe(x, invalid_bank, pipeline, ids, route_weights, destination,
                                  workspace, stream);
        },
        "short mapped routed bank");
    auto short_pinned_pipeline = pipeline;
    --short_pinned_pipeline.pinned_stage_bytes;
    failures += expect_invalid(
        [&] {
            ops::qwen4_sparse_moe(x, weights, short_pinned_pipeline, ids, route_weights,
                                  destination, workspace, stream);
        },
        "short pinned stage");
    Tensor aliased_destination = x;
    failures += expect_invalid(
        [&] {
            ops::qwen4_sparse_moe(x, weights, pipeline, ids, route_weights,
                                  aliased_destination, workspace, stream);
        },
        "aliased destination");
    auto mismatched_shared = weights;
    mismatched_shared.shared_up.qtype =
        shared_qtype == QType::GGML_Q5_K ? QType::GGML_Q6_K : QType::GGML_Q5_K;
    failures += expect_invalid(
        [&] {
            ops::qwen4_sparse_moe(x, mismatched_shared, pipeline, ids, route_weights,
                                  destination, workspace, stream);
        },
        "mismatched shared gate/up formats");
    auto short_resident_bank = resident_weights;
    --short_resident_bank.routed_gate.payload_bytes;
    failures += expect_invalid(
        [&] {
            ops::qwen4_sparse_moe_resident(
                x, short_resident_bank, resident_ids, resident_route_weights,
                resident_destination, workspace, stream);
        },
        "short resident routed bank");
    auto mismatched_resident = resident_weights;
    mismatched_resident.routed_up.qtype =
        routed_qtype == QType::GGML_IQ1_S ? QType::GGML_IQ2_XXS : QType::GGML_IQ1_S;
    failures += expect_invalid(
        [&] {
            ops::qwen4_sparse_moe_resident(
                x, mismatched_resident, resident_ids, resident_route_weights,
                resident_destination, workspace, stream);
        },
        "mismatched resident routed formats");
    Tensor short_stage(device_stage.data(), DType::U8,
                       {static_cast<std::int32_t>(ops::kQwen4SparseMoePipelineStageBytes - 1)});
    auto short_device_pipeline = pipeline;
    short_device_pipeline.device_stage = short_stage;
    failures += expect_invalid(
        [&] {
            ops::qwen4_sparse_moe(x, weights, short_device_pipeline, ids, route_weights,
                                  destination, workspace, stream);
        },
        "short device stage");
    auto same_stream_pipeline = pipeline;
    same_stream_pipeline.transfer_stream = stream;
    failures += expect_invalid(
        [&] {
            ops::qwen4_sparse_moe(x, weights, same_stream_pipeline, ids, route_weights,
                                  destination, workspace, stream);
        },
        "same compute and transfer stream");
    auto changed_compute_stream_pipeline = pipeline;
    changed_compute_stream_pipeline.compute_stream = pipeline_events.transfer_stream;
    failures += expect_invalid(
        [&] {
            ops::qwen4_sparse_moe(x, weights, changed_compute_stream_pipeline, ids,
                                  route_weights, destination, workspace, stream);
        },
        "changed pipeline compute stream");
    DeviceArena short_workspace(ops::qwen4_sparse_moe_workspace_capacity_bytes() - 1);
    failures += expect_invalid(
        [&] {
            ops::qwen4_sparse_moe(x, weights, pipeline, ids, route_weights, destination,
                                  short_workspace, stream);
        },
        "short workspace");
    return failures;
}

int prefill_partition_and_slot_reuse_case() {
    constexpr int pattern_count = 7;
    constexpr int amplitude_count = 4;
    constexpr int width = pattern_count * amplitude_count;
    constexpr int active_experts = pattern_count * kTopK;
    constexpr QType routed_qtype = QType::GGML_IQ1_S;
    const std::size_t one_routed = matrix_bytes(routed_qtype, kIntermediate, kHidden);
    const std::size_t bank_bytes = static_cast<std::size_t>(kExperts) * one_routed;
    AnonymousMapping mapped_gate(bank_bytes);
    AnonymousMapping mapped_up(bank_bytes);
    for (int expert = 0; expert < active_experts; ++expert) {
        const auto gate = make_matrix(routed_qtype, kIntermediate, kHidden,
                                      0xb000U + static_cast<std::uint32_t>(expert));
        const auto up = make_matrix(routed_qtype, kIntermediate, kHidden,
                                    0xc000U + static_cast<std::uint32_t>(expert));
        std::memcpy(mapped_gate.mutable_data() + static_cast<std::size_t>(expert) * one_routed,
                    gate.data(), one_routed);
        std::memcpy(mapped_up.mutable_data() + static_cast<std::size_t>(expert) * one_routed,
                    up.data(), one_routed);
    }
    mapped_gate.make_read_only();
    mapped_up.make_read_only();

    std::vector<float> input(static_cast<std::size_t>(kHidden) * width, 0.0F);
    std::vector<float> router(static_cast<std::size_t>(kExperts) * kHidden, 0.0F);
    std::vector<std::int32_t> expected_ids(static_cast<std::size_t>(kTopK) * width);
    std::vector<double> expected_weights(static_cast<std::size_t>(kTopK) * width);
    const auto expert_for = [=](int pattern, int rank) {
        return ((pattern * kTopK + rank) * 9 + 5) % active_experts;
    };
    std::array<std::array<OracleRoute, amplitude_count>, pattern_count> routes;
    constexpr std::array<float, amplitude_count> amplitudes{0.5F, 0.75F, 1.0F, 1.25F};
    std::array<bool, 3> high_weight_group{};
    for (int pattern = 0; pattern < pattern_count; ++pattern) {
        for (int rank = 0; rank < kTopK; ++rank) {
            const int expert = expert_for(pattern, rank);
            router[static_cast<std::size_t>(expert) * kHidden + pattern] = 20.0F - rank;
            if (rank < 2) {
                high_weight_group[expert < 16 ? 0 : (expert < 48 ? 1 : 2)] = true;
            }
        }
        for (int amplitude = 0; amplitude < amplitude_count; ++amplitude) {
            std::vector<double> pattern_input(kHidden, 0.0);
            pattern_input[pattern] = amplitudes[amplitude];
            routes[pattern][amplitude] = route_oracle(router, pattern_input);
            for (int rank = 0; rank < kTopK; ++rank) {
                if (routes[pattern][amplitude].ids[rank] != expert_for(pattern, rank)) {
                    throw std::logic_error("prefill partition router fixture is malformed");
                }
            }
        }
    }
    if (!std::all_of(high_weight_group.begin(), high_weight_group.end(),
                     [](bool covered) { return covered; })) {
        throw std::logic_error("prefill partition rank permutation is malformed");
    }
    for (int token = 0; token < width; ++token) {
        const int pattern = (5 * token + 3) % pattern_count;
        const int amplitude = token / pattern_count;
        input[pattern + static_cast<std::size_t>(kHidden) * token] = amplitudes[amplitude];
        for (int rank = 0; rank < kTopK; ++rank) {
            expected_ids[rank + kTopK * token] = routes[pattern][amplitude].ids[rank];
            expected_weights[rank + kTopK * token] =
                routes[pattern][amplitude].weights[rank];
        }
    }
    round_to_bf16(input);
    const std::size_t one_down =
        matrix_bytes(QType::GGML_IQ4_NL, kHidden, kIntermediate);
    const std::size_t down_bank_bytes = static_cast<std::size_t>(kExperts) * one_down;
    DeviceBuffer device_down(down_bank_bytes);
    device_down.fill(0);
    std::array<std::array<std::vector<std::uint8_t>, kTopK>, pattern_count> down_matrices;
    for (int pattern = 0; pattern < pattern_count; ++pattern) {
        for (int rank = 0; rank < kTopK; ++rank) {
            const int expert = routes[pattern][0].ids[rank];
            auto& matrix = down_matrices[pattern][rank];
            matrix = make_matrix(QType::GGML_IQ4_NL, kHidden, kIntermediate,
                                 0xa000U + static_cast<std::uint32_t>(expert));
            device_down.copy_from_host(matrix.data(), matrix.size(),
                                       static_cast<std::size_t>(expert) * one_down);
        }
    }
    const auto shared_gate_proj =
        make_matrix(QType::GGML_Q5_K, kIntermediate, kHidden, 0xd000U);
    const auto shared_up =
        make_matrix(QType::GGML_Q5_K, kIntermediate, kHidden, 0xe000U);
    const auto shared_down =
        make_matrix(QType::GGML_Q8_0, kHidden, kIntermediate, 0xf000U);
    std::vector<float> shared_gate(kHidden, 0.0F);
    for (int pattern = 0; pattern < pattern_count; ++pattern) {
        shared_gate[pattern] = static_cast<float>(pattern - 3) / 8.0F;
    }
    std::array<std::array<std::vector<double>, amplitude_count>, pattern_count>
        pattern_references;
    for (int pattern = 0; pattern < pattern_count; ++pattern) {
        for (int amplitude = 0; amplitude < amplitude_count; ++amplitude) {
            std::vector<double> pattern_input(kHidden, 0.0);
            pattern_input[pattern] = amplitudes[amplitude];
            const OracleInputs oracle_weights{
                routed_qtype,
                QType::GGML_Q5_K,
                router,
                mapped_gate,
                mapped_up,
                down_matrices[pattern],
                routes[pattern][amplitude].ids,
                shared_gate,
                shared_gate_proj,
                shared_up,
                shared_down,
            };
            pattern_references[pattern][amplitude] = complete_oracle(
                pattern_input, routes[pattern][amplitude], oracle_weights);
        }
    }

    // A second, independently encoded panel changes both the routed codec and every token's
    // selected expert set. It is submitted on the same event/slot resources without an
    // intervening caller synchronization, so its ids-ready barrier must close every consumer of
    // the first panel before either slot is overwritten.
    constexpr QType next_routed_qtype = QType::GGML_IQ2_XXS;
    const std::size_t next_one_routed =
        matrix_bytes(next_routed_qtype, kIntermediate, kHidden);
    const std::size_t next_rank_bytes = 2 * next_one_routed;
    const std::size_t next_bank_bytes =
        static_cast<std::size_t>(kExperts) * next_one_routed;
    AnonymousMapping next_mapped_gate(next_bank_bytes);
    AnonymousMapping next_mapped_up(next_bank_bytes);
    for (int expert = 0; expert < active_experts; ++expert) {
        const auto gate = make_matrix(next_routed_qtype, kIntermediate, kHidden,
                                      0x1b000U + static_cast<std::uint32_t>(expert));
        const auto up = make_matrix(next_routed_qtype, kIntermediate, kHidden,
                                    0x1c000U + static_cast<std::uint32_t>(expert));
        std::memcpy(next_mapped_gate.mutable_data() +
                        static_cast<std::size_t>(expert) * next_one_routed,
                    gate.data(), next_one_routed);
        std::memcpy(next_mapped_up.mutable_data() +
                        static_cast<std::size_t>(expert) * next_one_routed,
                    up.data(), next_one_routed);
    }
    next_mapped_gate.make_read_only();
    next_mapped_up.make_read_only();

    std::vector<float> next_input(static_cast<std::size_t>(kHidden) * width, 0.0F);
    std::vector<float> next_router(static_cast<std::size_t>(kExperts) * kHidden, 0.0F);
    std::vector<std::int32_t> next_expected_ids(static_cast<std::size_t>(kTopK) * width);
    std::vector<double> next_expected_weights(static_cast<std::size_t>(kTopK) * width);
    const auto next_expert_for = [=](int pattern, int rank) {
        return ((pattern * kTopK + rank) * 13 + 7) % active_experts;
    };
    std::array<std::array<OracleRoute, amplitude_count>, pattern_count> next_routes;
    std::array<bool, 3> next_high_weight_group{};
    for (int pattern = 0; pattern < pattern_count; ++pattern) {
        for (int rank = 0; rank < kTopK; ++rank) {
            const int expert = next_expert_for(pattern, rank);
            next_router[static_cast<std::size_t>(expert) * kHidden + pattern] = 20.0F - rank;
            if (rank < 2) {
                next_high_weight_group[expert < 16 ? 0 : (expert < 48 ? 1 : 2)] = true;
            }
        }
        for (int amplitude = 0; amplitude < amplitude_count; ++amplitude) {
            std::vector<double> pattern_input(kHidden, 0.0);
            pattern_input[pattern] = amplitudes[amplitude];
            next_routes[pattern][amplitude] = route_oracle(next_router, pattern_input);
            for (int rank = 0; rank < kTopK; ++rank) {
                if (next_routes[pattern][amplitude].ids[rank] !=
                    next_expert_for(pattern, rank)) {
                    throw std::logic_error("next prefill partition router fixture is malformed");
                }
            }
        }
    }
    if (!std::all_of(next_high_weight_group.begin(), next_high_weight_group.end(),
                     [](bool covered) { return covered; })) {
        throw std::logic_error("next prefill partition rank permutation is malformed");
    }
    for (int token = 0; token < width; ++token) {
        const int pattern = (5 * token + 3) % pattern_count;
        const int amplitude = amplitude_count - 1 - token / pattern_count;
        next_input[pattern + static_cast<std::size_t>(kHidden) * token] =
            amplitudes[amplitude];
        std::array<int, kTopK> first_set = routes[pattern][token / pattern_count].ids;
        std::array<int, kTopK> next_set = next_routes[pattern][amplitude].ids;
        std::sort(first_set.begin(), first_set.end());
        std::sort(next_set.begin(), next_set.end());
        if (first_set == next_set) {
            throw std::logic_error("back-to-back prefill route sets are not distinct");
        }
        for (int rank = 0; rank < kTopK; ++rank) {
            next_expected_ids[rank + kTopK * token] = next_routes[pattern][amplitude].ids[rank];
            next_expected_weights[rank + kTopK * token] =
                next_routes[pattern][amplitude].weights[rank];
        }
    }
    round_to_bf16(next_input);
    DeviceBuffer next_device_down(down_bank_bytes);
    next_device_down.fill(0);
    std::array<std::array<std::vector<std::uint8_t>, kTopK>, pattern_count>
        next_down_matrices;
    for (int pattern = 0; pattern < pattern_count; ++pattern) {
        for (int rank = 0; rank < kTopK; ++rank) {
            const int expert = next_routes[pattern][0].ids[rank];
            auto& matrix = next_down_matrices[pattern][rank];
            matrix = make_matrix(QType::GGML_IQ4_NL, kHidden, kIntermediate,
                                 0x1a000U + static_cast<std::uint32_t>(expert));
            next_device_down.copy_from_host(matrix.data(), matrix.size(),
                                            static_cast<std::size_t>(expert) * one_down);
        }
    }
    const auto next_shared_gate_proj =
        make_matrix(QType::GGML_Q5_K, kIntermediate, kHidden, 0x1d000U);
    const auto next_shared_up =
        make_matrix(QType::GGML_Q5_K, kIntermediate, kHidden, 0x1e000U);
    const auto next_shared_down =
        make_matrix(QType::GGML_Q8_0, kHidden, kIntermediate, 0x1f000U);
    std::vector<float> next_shared_gate(kHidden, 0.0F);
    for (int pattern = 0; pattern < pattern_count; ++pattern) {
        next_shared_gate[pattern] = static_cast<float>(3 - pattern) / 7.0F;
    }
    std::array<std::array<std::vector<double>, amplitude_count>, pattern_count>
        next_pattern_references;
    for (int pattern = 0; pattern < pattern_count; ++pattern) {
        for (int amplitude = 0; amplitude < amplitude_count; ++amplitude) {
            std::vector<double> pattern_input(kHidden, 0.0);
            pattern_input[pattern] = amplitudes[amplitude];
            const OracleInputs oracle_weights{
                next_routed_qtype,
                QType::GGML_Q5_K,
                next_router,
                next_mapped_gate,
                next_mapped_up,
                next_down_matrices[pattern],
                next_routes[pattern][amplitude].ids,
                next_shared_gate,
                next_shared_gate_proj,
                next_shared_up,
                next_shared_down,
            };
            next_pattern_references[pattern][amplitude] = complete_oracle(
                pattern_input, next_routes[pattern][amplitude], oracle_weights);
        }
    }
    DeviceBuffer device_x = to_device_bf16(input);
    DeviceBuffer device_router = to_device_f32(router);
    DeviceBuffer device_shared_gate = to_device_f32(shared_gate);
    DeviceBuffer device_shared_gate_proj = to_device(shared_gate_proj);
    DeviceBuffer device_shared_up = to_device(shared_up);
    DeviceBuffer device_shared_down = to_device(shared_down);
    DeviceBuffer next_device_x = to_device_bf16(next_input);
    DeviceBuffer next_device_router = to_device_f32(next_router);
    DeviceBuffer next_device_shared_gate = to_device_f32(next_shared_gate);
    DeviceBuffer next_device_shared_gate_proj = to_device(next_shared_gate_proj);
    DeviceBuffer next_device_shared_up = to_device(next_shared_up);
    DeviceBuffer next_device_shared_down = to_device(next_shared_down);
    GuardedDeviceBuffer stage(ops::kQwen4SparseMoePrefillPipelineStageBytes);
    GuardedDeviceBuffer ids(static_cast<std::size_t>(kTopK) * width * sizeof(std::int32_t));
    GuardedDeviceBuffer weights(static_cast<std::size_t>(kTopK) * width * sizeof(float));
    GuardedDeviceBuffer output(static_cast<std::size_t>(kHidden) * width *
                               sizeof(std::uint16_t));
    GuardedDeviceBuffer next_ids(static_cast<std::size_t>(kTopK) * width *
                                 sizeof(std::int32_t));
    GuardedDeviceBuffer next_weights(static_cast<std::size_t>(kTopK) * width * sizeof(float));
    GuardedDeviceBuffer next_output(static_cast<std::size_t>(kHidden) * width *
                                    sizeof(std::uint16_t));
    stage.fill(0xcd);
    ids.fill(0xcd);
    weights.fill(0xcd);
    output.fill(0xcd);
    next_ids.fill(0xcd);
    next_weights.fill(0xcd);
    next_output.fill(0xcd);
    PinnedHostBuffer pinned(ops::kQwen4SparseMoePrefillPipelineStageBytes);
    PinnedHostBuffer host_scratch(ops::kQwen4SparseMoePrefillHostScratchBytes);
    std::memset(pinned.data(), 0xa5, pinned.size());
    DeviceArena workspace(ops::qwen4_sparse_moe_prefill_workspace_capacity_bytes(width));
    DeviceArena next_workspace(ops::qwen4_sparse_moe_prefill_workspace_capacity_bytes(width));
    PipelineEvents events;
    cudaStream_t stream = nullptr;
    cuda_check(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking),
               "cudaStreamCreate prefill partition");

    Tensor x_tensor(device_x.p, DType::BF16, {kHidden, width});
    Tensor stage_tensor(
        stage.data(), DType::U8,
        {static_cast<std::int32_t>(ops::kQwen4SparseMoePrefillPipelineStageBytes)});
    Tensor ids_tensor(ids.data(), DType::I32, {kTopK, width});
    Tensor weights_tensor(weights.data(), DType::FP32, {kTopK, width});
    Tensor output_tensor(output.data(), DType::BF16, {kHidden, width});
    Tensor next_x_tensor(next_device_x.p, DType::BF16, {kHidden, width});
    Tensor next_ids_tensor(next_ids.data(), DType::I32, {kTopK, width});
    Tensor next_weights_tensor(next_weights.data(), DType::FP32, {kTopK, width});
    Tensor next_output_tensor(next_output.data(), DType::BF16, {kHidden, width});
    Tensor shared_gate_tensor(device_shared_gate.p, DType::FP32, {kHidden});
    Tensor next_shared_gate_tensor(next_device_shared_gate.p, DType::FP32, {kHidden});
    ops::Qwen4SparseMoeWeights model_weights{
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
            device_shared_gate_proj.p, shared_gate_proj.size(), QType::GGML_Q5_K, 1,
            kIntermediate, kHidden),
        .shared_up = make_ggml_weight(device_shared_up.p, shared_up.size(), QType::GGML_Q5_K, 1,
                                      kIntermediate, kHidden),
        .shared_down = make_ggml_weight(device_shared_down.p, shared_down.size(),
                                        QType::GGML_Q8_0, 1, kHidden, kIntermediate),
    };
    ops::Qwen4SparseMoeWeights next_model_weights{
        .router = make_router(next_device_router.p),
        .routed_gate_up = {
            .gate = std::span<const std::byte>(
                reinterpret_cast<const std::byte*>(next_mapped_gate.data()),
                next_mapped_gate.size()),
            .up = std::span<const std::byte>(
                reinterpret_cast<const std::byte*>(next_mapped_up.data()),
                next_mapped_up.size()),
            .qtype = next_routed_qtype,
        },
        .routed_down = make_ggml_weight(next_device_down.p, down_bank_bytes,
                                        QType::GGML_IQ4_NL, kExperts, kHidden, kIntermediate),
        .shared_gate = next_shared_gate_tensor,
        .shared_gate_proj = make_ggml_weight(
            next_device_shared_gate_proj.p, next_shared_gate_proj.size(), QType::GGML_Q5_K, 1,
            kIntermediate, kHidden),
        .shared_up = make_ggml_weight(next_device_shared_up.p, next_shared_up.size(),
                                      QType::GGML_Q5_K, 1, kIntermediate, kHidden),
        .shared_down = make_ggml_weight(next_device_shared_down.p, next_shared_down.size(),
                                        QType::GGML_Q8_0, 1, kHidden, kIntermediate),
    };
    ops::Qwen4SparseMoePrefillPipeline pipeline{
        .pinned_stage = pinned.data(),
        .pinned_stage_bytes = pinned.size(),
        .host_scratch = host_scratch.data(),
        .host_scratch_bytes = host_scratch.size(),
        .device_stage = stage_tensor,
        .transfer_stream = events.transfer_stream,
        .compute_stream = stream,
        .route_ready = events.route_ready,
        .ids_ready = events.ids_ready,
        .transfer_ready = {events.transfer_ready[0], events.transfer_ready[1]},
        .consumer_complete = {events.consumer_complete[0], events.consumer_complete[1]},
    };
    ops::qwen4_sparse_moe_prefill(x_tensor, model_weights, pipeline, ids_tensor,
                                  weights_tensor, output_tensor, workspace, stream);
    ops::qwen4_sparse_moe_prefill(next_x_tensor, next_model_weights, pipeline,
                                  next_ids_tensor, next_weights_tensor, next_output_tensor,
                                  next_workspace, stream);
    cuda_synchronize(stream);

    int failures = verify_exact("Qwen4 sparse MoE prefill partition ids",
                                from_device<std::int32_t>(ids.data(), expected_ids.size()),
                                expected_ids);
    const auto actual_weight_f32 = from_device<float>(weights.data(), expected_weights.size());
    failures += verify_pointwise(
        "Qwen4 sparse MoE prefill partition weights",
        std::vector<double>(actual_weight_f32.begin(), actual_weight_f32.end()),
        expected_weights, kRouteWeightCriterion);
    const auto actual_output =
        from_device_bf16(output.data(), static_cast<std::size_t>(kHidden) * width);
    for (int token = 0; token < width; ++token) {
        const int pattern = (5 * token + 3) % pattern_count;
        const int amplitude = token / pattern_count;
        failures += verify_reduction(
            "Qwen4 sparse MoE prefill multi-group per-token complete FP64 formula",
            std::span<const double>(actual_output).subspan(
                static_cast<std::size_t>(token) * kHidden, kHidden),
            pattern_references[pattern][amplitude], kOutputCriterion);
    }

    failures += verify_exact(
        "Qwen4 sparse MoE back-to-back IQ2 prefill partition ids",
        from_device<std::int32_t>(next_ids.data(), next_expected_ids.size()),
        next_expected_ids);
    const auto next_actual_weight_f32 =
        from_device<float>(next_weights.data(), next_expected_weights.size());
    failures += verify_pointwise(
        "Qwen4 sparse MoE back-to-back IQ2 prefill partition weights",
        std::vector<double>(next_actual_weight_f32.begin(), next_actual_weight_f32.end()),
        next_expected_weights, kRouteWeightCriterion);
    const auto next_actual_output =
        from_device_bf16(next_output.data(), static_cast<std::size_t>(kHidden) * width);
    for (int token = 0; token < width; ++token) {
        const int pattern = (5 * token + 3) % pattern_count;
        const int amplitude = amplitude_count - 1 - token / pattern_count;
        failures += verify_reduction(
            "Qwen4 sparse MoE back-to-back IQ2 per-token complete FP64 formula",
            std::span<const double>(next_actual_output).subspan(
                static_cast<std::size_t>(token) * kHidden, kHidden),
            next_pattern_references[pattern][amplitude], kOutputCriterion);
    }

    const auto* pinned_bytes = static_cast<const std::uint8_t*>(pinned.data());
    const auto* device_bytes = static_cast<const std::uint8_t*>(stage.data());
    for (int slot_index = 0; slot_index < 2; ++slot_index) {
        const int first_expert = slot_index == 0 ? 48 : 16;
        const int expert_count = slot_index == 0 ? 22 : 32;
        const std::size_t slot_offset = static_cast<std::size_t>(slot_index) *
                                        ops::kQwen4SparseMoePrefillSlotCapacityBytes;
        for (int local = 0; local < expert_count; ++local) {
            const int expert = first_expert + local;
            std::vector<std::uint8_t> expected_pair(next_rank_bytes);
            std::memcpy(expected_pair.data(),
                        next_mapped_gate.data() +
                            static_cast<std::size_t>(expert) * next_one_routed,
                        next_one_routed);
            std::memcpy(expected_pair.data() + next_one_routed,
                        next_mapped_up.data() +
                            static_cast<std::size_t>(expert) * next_one_routed,
                        next_one_routed);
            if (std::memcmp(pinned_bytes + slot_offset + static_cast<std::size_t>(local) *
                                                       next_rank_bytes,
                            expected_pair.data(), next_rank_bytes) != 0) {
                std::cerr << "Qwen4 sparse MoE IQ2 prefill partition host bytes changed\n";
                ++failures;
            }
            failures += verify_exact(
                "Qwen4 sparse MoE IQ2 prefill partition device bytes",
                from_device<std::uint8_t>(
                    device_bytes + slot_offset +
                        static_cast<std::size_t>(local) * next_rank_bytes,
                    next_rank_bytes),
                expected_pair);
        }
        std::vector<std::int32_t> expected_occurrences;
        for (int expert = first_expert; expert < first_expert + expert_count; ++expert) {
            for (std::size_t rank_token = 0; rank_token < next_expected_ids.size();
                 ++rank_token) {
                if (next_expected_ids[rank_token] == expert) {
                    expected_occurrences.push_back(static_cast<std::int32_t>(rank_token));
                }
            }
        }
        const std::size_t occurrence_offset =
            slot_offset + static_cast<std::size_t>(expert_count) * next_rank_bytes;
        const auto* host_occurrences = reinterpret_cast<const std::int32_t*>(
            pinned_bytes + occurrence_offset);
        if (!std::equal(expected_occurrences.begin(), expected_occurrences.end(),
                        host_occurrences)) {
            std::cerr << "Qwen4 sparse MoE IQ2 prefill partition host occurrences changed\n";
            ++failures;
        }
        failures += verify_exact(
            "Qwen4 sparse MoE IQ2 prefill partition device occurrences",
            from_device<std::int32_t>(device_bytes + occurrence_offset,
                                      expected_occurrences.size()),
            expected_occurrences);
    }
    failures += stage.verify_guards("Qwen4 sparse MoE prefill partition stage");
    failures += ids.verify_guards("Qwen4 sparse MoE prefill partition ids");
    failures += weights.verify_guards("Qwen4 sparse MoE prefill partition weights");
    failures += output.verify_guards("Qwen4 sparse MoE prefill partition output");
    failures += next_ids.verify_guards("Qwen4 sparse MoE back-to-back IQ2 partition ids");
    failures += next_weights.verify_guards(
        "Qwen4 sparse MoE back-to-back IQ2 partition weights");
    failures += next_output.verify_guards("Qwen4 sparse MoE back-to-back IQ2 partition output");
    std::vector<std::uint16_t> represented_input(input.size());
    std::transform(input.begin(), input.end(), represented_input.begin(), f32_to_bf16);
    failures += verify_exact(
        "Qwen4 sparse MoE prefill partition input unchanged",
        from_device<std::uint16_t>(device_x, represented_input.size()), represented_input);
    std::vector<std::uint16_t> next_represented_input(next_input.size());
    std::transform(next_input.begin(), next_input.end(), next_represented_input.begin(),
                   f32_to_bf16);
    failures += verify_exact(
        "Qwen4 sparse MoE back-to-back IQ2 partition input unchanged",
        from_device<std::uint16_t>(next_device_x, next_represented_input.size()),
        next_represented_input);
    cuda_check(cudaStreamSynchronize(events.transfer_stream),
               "cudaStreamSynchronize prefill partition transfer");
    cuda_check(cudaStreamDestroy(stream), "cudaStreamDestroy prefill partition");
    return failures;
}

} // namespace

int main() {
    try {
        if (const int unavailable = require_cuda(); unavailable != 0) { return unavailable; }
        if (ops::qwen4_sparse_moe_rank_stage_bytes(QType::GGML_IQ1_S) != 640'000 ||
            ops::qwen4_sparse_moe_rank_stage_bytes(QType::GGML_IQ2_XXS) != 844'800) {
            throw std::runtime_error("Qwen4 sparse MoE stage-capacity proof is wrong");
        }
        static_assert(ops::kQwen4SparseMoePrefillGroupMatrixCapacityBytes == 27'033'600);
        static_assert(ops::kQwen4SparseMoePrefillSlotCapacityBytes == 27'197'440);
        static_assert(ops::kQwen4SparseMoePrefillPipelineStageBytes == 54'394'880);
        static_assert(ops::kQwen4SparseMoePrefillHostScratchBytes == 335'876);
        int failures = expect_invalid(
            [] { (void)ops::qwen4_sparse_moe_rank_stage_bytes(QType::GGML_Q5_K); },
            "unsupported staged format");
        failures += expect_invalid(
            [] { (void)ops::qwen4_sparse_moe_prefill_workspace_capacity_bytes(0); },
            "zero prefill width");
        failures += expect_invalid(
            [] { (void)ops::qwen4_sparse_moe_prefill_workspace_capacity_bytes(4097); },
            "excessive prefill width");
        failures += expect_invalid(
            [] { (void)ops::qwen4_sparse_moe_resident_workspace_capacity_bytes(0); },
            "zero resident width");
        failures += expect_invalid(
            [] { (void)ops::qwen4_sparse_moe_resident_workspace_capacity_bytes(4097); },
            "excessive resident width");
        if (ops::qwen4_sparse_moe_prefill_workspace_capacity_bytes(4096) == 0) {
            throw std::runtime_error("Qwen4 sparse MoE maximum prefill workspace is empty");
        }
        cudaStream_t stream = nullptr;
        cuda_check(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking), "cudaStreamCreate");
        PipelineEvents pipeline_events;
        failures += prefill_gate_up_rounding_boundary_case(QType::GGML_IQ1_S, stream);
        failures += prefill_gate_up_rounding_boundary_case(QType::GGML_IQ2_XXS, stream);
        failures += prefill_gate_up_dense_oracle_case(QType::GGML_IQ1_S, stream);
        failures += prefill_gate_up_dense_oracle_case(QType::GGML_IQ2_XXS, stream);
        failures += run_case(QType::GGML_IQ1_S, QType::GGML_Q5_K,
                             RouteFixture::AllZero, 80.0F, pipeline_events, stream);
        failures += run_case(QType::GGML_IQ2_XXS, QType::GGML_Q6_K,
                             RouteFixture::Nonuniform, -80.0F, pipeline_events, stream);
        failures += run_case(QType::GGML_IQ1_S, QType::GGML_Q5_K,
                             RouteFixture::Underflow, 0.0F, pipeline_events, stream);
        failures += prefill_partition_and_slot_reuse_case();
        cuda_check(cudaStreamSynchronize(pipeline_events.transfer_stream),
                   "cudaStreamSynchronize transfer");
        cuda_check(cudaStreamDestroy(stream), "cudaStreamDestroy");
        return failures == 0 ? 0 : 1;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
