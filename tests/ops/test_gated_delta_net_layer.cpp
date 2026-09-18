#include "ninfer/ops/gated_delta_net_layer.h"

#include "ops/op_tester.h"
#include "ops/native_projection_fixture.h"
#include "targets/qwen4/native_bf16_fixture.h"
#include "targets/qwen4/native_sequence_components.h"
#include "ops/launcher/gated_delta_net_layer.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <limits>
#include <random>
#include <string>
#include <utility>
#include <vector>

using namespace ninfer;
using namespace ninfer::test;

namespace {

constexpr std::int32_t kHidden     = 2560;
constexpr std::int32_t kQkHeads    = 16;
constexpr std::int32_t kValueHeads = 48;
constexpr std::int32_t kHeadDim    = 128;
constexpr std::int32_t kQkRows     = kQkHeads * kHeadDim;
constexpr std::int32_t kValueRows  = kValueHeads * kHeadDim;
constexpr std::int32_t kQkvRows    = 2 * kQkRows + kValueRows;
constexpr std::int32_t kPrompt     = 3;

// The complete T=1 profile crosses represented BF16 projection/conv/norm boundaries and retains
// measured headroom while detecting any omitted normalization, head repeat, gate, decay, or scale.
constexpr ReductionCriterion kOutputCriterion{/*relative_l2=*/9.0e-3,
                                               /*gross_absolute=*/2.5e-4,
                                               /*gross_relative_to_max_reference=*/5.0e-3};
constexpr ReductionCriterion kStateCriterion{/*relative_l2=*/6.5e-3,
                                              /*gross_absolute=*/2.0e-5,
                                              /*gross_relative_to_max_reference=*/4.0e-3};
constexpr ReductionCriterion kConvStateCriterion{/*relative_l2=*/5.0e-5,
                                                  /*gross_absolute=*/5.0e-6,
                                                  /*gross_relative_to_max_reference=*/1.0e-5};
// Direct BF16 weights have non-periodic mantissas. FP32 reduction order can move a dot product
// across a BF16 midpoint, so the stored history is qualified at BF16 precision, not near-exact
// agreement with the ideal FP64 dot rounded to BF16. Keep existing packed-profile gates unchanged.
constexpr ReductionCriterion kBf16ConvStateCriterion{/*relative_l2=*/1.0 / 256.0,
    /*gross_absolute=*/0.0, /*gross_relative_to_max_reference=*/1.0 / 128.0};

// Declared before candidate execution. A8 is an approximation profile, not a new oracle:
// Linear's .04/.06 allowance also covers stored QKV; .08/.12 state permits the two
// normalized Q/K perturbation paths plus V in the bounded recurrence. These are
// qualification budgets, not universal nonlinear bounds or whole-model quality gates.
constexpr ReductionCriterion kA8OutputCriterion{.04, 2.5e-4, .06};
constexpr ReductionCriterion kA8StateCriterion{.08, 2.0e-5, .12};
constexpr ReductionCriterion kA8ConvCriterion{.04, 5.0e-6, .06};

const ReductionCriterion& conv_state_criterion(QType type) {
    return type == QType::BF16_CTRL ? kBf16ConvStateCriterion : kConvStateCriterion;
}

double sigmoid(double value) {
    if (value >= 0.0) { return 1.0 / (1.0 + std::exp(-value)); }
    const double exponential = std::exp(value);
    return exponential / (1.0 + exponential);
}

double softplus(double value) {
    if (value > 20.0) { return value; }
    if (value < -20.0) { return std::exp(value); }
    return std::log1p(std::exp(value));
}

double silu(double value) { return value * sigmoid(value); }

double represented_bf16(double value) {
    // Round the mathematical value directly at the declared BF16 boundary. A temporary
    // FP32 cast can double-round values adjacent to a BF16 midpoint (the hash-distributed
    // BF16 projection witness contains these), and is not part of the oracle formula.
    return qwen4_sequence::represented(value);
}

int bf16_state_rounding_witness() {
    // Real BF16 QKV fixture: row 8691, token 1. FP32 narrowing moves the exact
    // below-midpoint dot onto a tie, changing the subsequent BF16 state bit.
    for (const auto [input, expected] : std::array<std::pair<double, double>, 6>{
            std::pair{0x1.6affff64p-7, 0x1.6ap-7}, {0x1.6bp-7, 0x1.6cp-7},
            {0x1.6b00009cp-7, 0x1.6cp-7}, {-0x1.6affff64p-7, -0x1.6ap-7},
            {-0x1.6bp-7, -0x1.6cp-7}, {-0x1.6b00009cp-7, -0x1.6cp-7}}) {
        if (represented_bf16(input) != expected) {
            std::cerr << "FAIL direct BF16 state rounding midpoint witness\n";
            return 1;
        }
    }
    return 0;
}

int conv_source_layout_witness() {
    constexpr int channel = 7;
    constexpr int key_channel = kQkRows + 11;
    std::vector<float> projected(kQkvRows, 0.0F);
    std::vector<float> state(static_cast<std::size_t>(kQkvRows) * 3, 0.0F);
    std::vector<float> weight(static_cast<std::size_t>(kQkvRows) * 4, 0.0F);
    projected[channel] = 4.0F;
    state[channel] = 1.0F;
    state[kQkvRows + channel] = 2.0F;
    state[2 * kQkvRows + channel] = 3.0F;
    weight[4 * channel] = 1.0F;
    weight[4 * channel + 1] = 2.0F;
    weight[4 * channel + 2] = 4.0F;
    weight[4 * channel + 3] = 8.0F;
    projected[key_channel] = 4.0F;
    state[key_channel] = 1.0F;
    state[kQkvRows + key_channel] = 2.0F;
    state[2 * kQkvRows + key_channel] = 3.0F;
    weight[4 * key_channel] = 1.0F;
    weight[4 * key_channel + 1] = 2.0F;
    weight[4 * key_channel + 2] = 4.0F;
    weight[4 * key_channel + 3] = 8.0F;
    round_to_bf16(projected);
    round_to_bf16(state);
    auto d_projected = to_device_bf16(projected);
    auto d_state = to_device_bf16(state);
    auto d_weight = to_device_f32(weight);
    GuardedDeviceBuffer d_state_out(state.size() * sizeof(std::uint16_t));
    GuardedDeviceBuffer d_q(kValueRows * sizeof(std::uint16_t));
    GuardedDeviceBuffer d_k(kValueRows * sizeof(std::uint16_t));
    GuardedDeviceBuffer d_v(kValueRows * sizeof(std::uint16_t));
    d_state_out.fill(0);
    d_q.fill(0);
    d_k.fill(0);
    d_v.fill(0);
    Tensor projected_t(d_projected.p, DType::BF16, {kQkvRows});
    Tensor weight_t(d_weight.p, DType::FP32, {4, kQkvRows});
    Tensor state_t(d_state.p, DType::BF16, {kQkvRows, 3});
    Tensor state_out_t(d_state_out.data(), DType::BF16, {kQkvRows, 3});
    Tensor q_t(d_q.data(), DType::BF16, {kHeadDim, kValueHeads});
    Tensor k_t(d_k.data(), DType::BF16, {kHeadDim, kValueHeads});
    Tensor v_t(d_v.data(), DType::BF16, {kHeadDim, kValueHeads});
    ops::detail::gated_delta_net_layer_conv_launch(projected_t, weight_t, state_t, state_out_t,
                                                    q_t, k_t, v_t, nullptr);
    cuda_synchronize();
    const auto q_bits = from_device<std::uint16_t>(d_q.data(), kValueRows);
    const auto state_bits = from_device<std::uint16_t>(d_state_out.data(), state.size());
    // 1*1 + 2*2 + 4*3 + 8*4 = 49. A tap-major transpose reads zeros at this channel.
    int failures = q_bits[channel] == f32_to_bf16(49.0F) ? 0 : 1;
    failures += q_bits[kQkRows + channel] == f32_to_bf16(49.0F) ? 0 : 1;
    failures += q_bits[2 * kQkRows + channel] == f32_to_bf16(49.0F) ? 0 : 1;
    const auto k_bits = from_device<std::uint16_t>(d_k.data(), kValueRows);
    constexpr int key_index = key_channel - kQkRows;
    failures += k_bits[key_index] == f32_to_bf16(49.0F) ? 0 : 1;
    failures += k_bits[kQkRows + key_index] == f32_to_bf16(49.0F) ? 0 : 1;
    failures += k_bits[2 * kQkRows + key_index] == f32_to_bf16(49.0F) ? 0 : 1;
    failures += state_bits[channel] == f32_to_bf16(2.0F) ? 0 : 1;
    failures += state_bits[kQkvRows + channel] == f32_to_bf16(3.0F) ? 0 : 1;
    failures += state_bits[2 * kQkvRows + channel] == f32_to_bf16(4.0F) ? 0 : 1;
    if (failures != 0) { std::cerr << "FAIL GDN GGUF channel-major conv layout witness\n"; }
    failures += d_state_out.verify_guards("GDN layout state");
    failures += d_q.verify_guards("GDN layout q");
    failures += d_k.verify_guards("GDN layout k");
    failures += d_v.verify_guards("GDN layout v");
    return failures;
}

std::vector<std::uint16_t> bf16_bits(const std::vector<float>& values) {
    std::vector<std::uint16_t> bits(values.size());
    std::transform(values.begin(), values.end(), bits.begin(), f32_to_bf16);
    return bits;
}

std::uint16_t read_u16(const std::uint8_t* bytes) {
    return static_cast<std::uint16_t>(bytes[0]) |
           (static_cast<std::uint16_t>(bytes[1]) << 8U);
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

int signed_byte(std::uint8_t word) {
    return word < 128U ? static_cast<int>(word) : static_cast<int>(word) - 256;
}

std::pair<int, int> scale_min(const std::uint8_t* table, int group) {
    if (group < 4) { return {table[group] & 63, table[group + 4] & 63}; }
    return {(table[group + 4] & 15) + 16 * (table[group - 4] >> 6U),
            (table[group + 4] >> 4U) + 16 * (table[group] >> 6U)};
}

std::vector<double> decode_block(QType qtype, const std::uint8_t* block) {
    std::vector<double> values(256);
    if (qtype == QType::GGML_Q5_K) {
        const double d = half_to_double(read_u16(block));
        const double dmin = half_to_double(read_u16(block + 2));
        for (int group = 0; group < 8; ++group) {
            const auto [scale, minimum] = scale_min(block + 4, group);
            for (int lane = 0; lane < 32; ++lane) {
                const int packed = block[48 + 32 * (group / 2) + lane];
                int code = group % 2 == 0 ? packed & 15 : packed >> 4;
                code += 16 * ((block[16 + lane] >> group) & 1U);
                values[32 * group + lane] = d * scale * code - dmin * minimum;
            }
        }
        return values;
    }
    const double d = half_to_double(read_u16(block + 208));
    for (int half = 0; half < 2; ++half) {
        for (int group = 0; group < 4; ++group) {
            for (int lane = 0; lane < 32; ++lane) {
                const int index = 128 * half + 32 * group + lane;
                const int low_word = block[64 * half + lane + 32 * (group % 2)];
                const int low = group < 2 ? low_word & 15 : low_word >> 4;
                const int high = (block[128 + 32 * half + lane] >> (2 * group)) & 3;
                values[index] = d * signed_byte(block[192 + index / 16]) *
                                (low + 16 * high - 32);
            }
        }
    }
    return values;
}

void write_u16(std::uint8_t* bytes, std::uint16_t value) {
    bytes[0] = static_cast<std::uint8_t>(value & 0xffU);
    bytes[1] = static_cast<std::uint8_t>(value >> 8U);
}

std::vector<std::uint8_t> make_quant(QType qtype, std::int32_t rows, std::int32_t columns,
                                     std::uint32_t seed) {
    const std::size_t block_bytes = qtype == QType::GGML_Q5_K ? 176U : 210U;
    std::vector<std::uint8_t> bytes(static_cast<std::size_t>(rows) * (columns / 256) * block_bytes);
    std::mt19937 generator(seed);
    std::uniform_int_distribution<int> distribution(0, 255);
    for (auto& byte : bytes) { byte = static_cast<std::uint8_t>(distribution(generator)); }
    for (std::size_t offset = 0; offset < bytes.size(); offset += block_bytes) {
        auto* block = bytes.data() + offset;
        if (qtype == QType::GGML_Q5_K) {
            write_u16(block, 0x0080U);
            write_u16(block + 2, 0x0040U);
        } else {
            write_u16(block + 208, 0x0080U);
        }
    }
    return bytes;
}

Weight ggml_weight(void* data, std::uint64_t bytes, QType qtype, std::int32_t rows,
                   std::int32_t columns) {
    Weight weight{};
    weight.payload         = data;
    weight.payload_bytes   = bytes;
    weight.qdata           = data;
    weight.qtype           = qtype;
    weight.layout          = QuantLayout::GgmlBlockRow;
    weight.n               = rows;
    weight.k               = columns;
    weight.group           = 256;
    weight.group_size      = 256;
    weight.ndim            = 2;
    weight.shape[0]        = rows;
    weight.shape[1]        = columns;
    weight.padded_shape[0] = rows;
    weight.padded_shape[1] = columns;
    return weight;
}

struct Fixture {
    QType input_qtype;
    quantized_weight::PackedWeight native_qkv, native_z, native_output;
    std::vector<std::uint8_t> qkv;
    std::vector<std::uint8_t> z;
    std::vector<float> a;
    std::vector<float> b;
    std::vector<float> conv;
    std::vector<float> ssm_a;
    std::vector<float> dt_bias;
    std::vector<float> norm;
    std::vector<std::uint8_t> output;
    DeviceBuffer d_qkv;
    DeviceBuffer d_z;
    DeviceBuffer d_a;
    DeviceBuffer d_b;
    DeviceBuffer d_conv;
    DeviceBuffer d_ssm_a;
    DeviceBuffer d_dt_bias;
    DeviceBuffer d_norm;
    DeviceBuffer d_output;

    explicit Fixture(QType input_format)
        : input_qtype(input_format),
          qkv(make_quant(input_format, kQkvRows, kHidden, 8101U)),
          z(make_quant(input_format, kValueRows, kHidden, 8102U)),
          a(static_cast<std::size_t>(kValueHeads) * kHidden),
          b(static_cast<std::size_t>(kValueHeads) * kHidden),
          conv(static_cast<std::size_t>(kQkvRows) * 4), ssm_a(kValueHeads),
          dt_bias(kValueHeads), norm(kHeadDim),
          output(make_quant(QType::GGML_Q6_K, kHidden, kValueRows, 8109U)) {
        if (native_projection_format(input_format)) {
            native_qkv = native_projection_fixture(input_format, kQkvRows, kHidden, 8101U);
            native_z = native_projection_fixture(input_format, kValueRows, kHidden, 8102U);
            native_output = native_projection_fixture(input_format, kHidden, kValueRows, 8109U);
            qkv = native_qkv.payload;
            z = native_z.payload;
            output = native_output.payload;
        }
        fill_uniform(a, 8103U, -0.008F, 0.008F);
        fill_uniform(b, 8104U, -0.008F, 0.008F);
        fill_uniform(conv, 8105U, -0.20F, 0.20F);
        fill_uniform(ssm_a, 8106U, -0.25F, -0.02F);
        ssm_a[0] = -157.984F;
        ssm_a[17] = -0.0279F;
        fill_uniform(dt_bias, 8107U, -0.5F, 0.5F);
        fill_uniform(norm, 8108U, 0.5F, 1.5F);
        upload();
    }

    explicit Fixture(const std::string& path, int layer = 0) : input_qtype(QType::BF16_CTRL) {
        qwen4_native::Bf16Source source(path, layer);
        const std::string p = "linear_attn.";
        const auto source_qkv = source.bits(p + "in_proj_qkv.weight", {kQkvRows, kHidden});
        const auto source_z = source.bits(p + "in_proj_z.weight", {kValueRows, kHidden});
        const auto source_output = source.bits(p + "out_proj.weight", {kHidden, kValueRows});
        const auto source_a = source.values(p + "in_proj_a.weight", {kValueHeads, kHidden});
        const auto source_b = source.values(p + "in_proj_b.weight", {kValueHeads, kHidden});
        const auto source_conv = source.values(p + "conv1d.weight", {kQkvRows, 1, 4});
        const auto source_log = source.values(p + "A_log", {kValueHeads});
        const auto source_bias = source.values(p + "dt_bias", {kValueHeads});
        norm = source.values(p + "norm.weight", {kHeadDim}); // ordinary gamma, no +1
        auto projected_qkv = source_qkv;
        auto projected_z = source_z;
        auto projected_output = source_output;
        a.resize(source_a.size()); b.resize(source_b.size()); conv = source_conv;
        ssm_a.resize(kValueHeads); dt_bias.resize(kValueHeads);
        // Source associates three consecutive V heads with each Q/K head. The admitted Op
        // tiles Q/K heads; reorder every V-side role consistently, preserving all BF16 bits.
        for (int head = 0; head < kValueHeads; ++head) {
            const int source_head = (head % kQkHeads) * 3 + head / kQkHeads;
            ssm_a[head] = -std::exp(source_log[source_head]);
            dt_bias[head] = source_bias[source_head];
            std::copy_n(source_a.begin() + source_head * kHidden, kHidden, a.begin() + head * kHidden);
            std::copy_n(source_b.begin() + source_head * kHidden, kHidden, b.begin() + head * kHidden);
            for (int d = 0; d < kHeadDim; ++d) {
                const int row = head * kHeadDim + d, source_row = source_head * kHeadDim + d;
                std::copy_n(source_qkv.begin() + (2*kQkRows+source_row)*kHidden, kHidden,
                            projected_qkv.begin() + (2*kQkRows+row)*kHidden);
                std::copy_n(source_z.begin() + source_row*kHidden, kHidden,
                            projected_z.begin() + row*kHidden);
                std::copy_n(source_conv.begin() + (2*kQkRows+source_row)*4, 4,
                            conv.begin() + (2*kQkRows+row)*4);
                for (int output_row = 0; output_row < kHidden; ++output_row) {
                    projected_output[output_row*kValueRows+row] = source_output[output_row*kValueRows+source_row];
                }
            }
        }
        native_qkv = qwen4_native::bf16_matrix(projected_qkv, kQkvRows, kHidden);
        native_z = qwen4_native::bf16_matrix(projected_z, kValueRows, kHidden);
        native_output = qwen4_native::bf16_matrix(projected_output, kHidden, kValueRows);
        qkv = native_qkv.payload; z = native_z.payload; output = native_output.payload;
        upload();
    }

    void upload() {
        d_qkv = to_device(qkv);
        d_z = to_device(z);
        d_a = to_device_f32(a);
        d_b = to_device_f32(b);
        d_conv = to_device_f32(conv);
        d_ssm_a = to_device_f32(ssm_a);
        d_dt_bias = to_device_f32(dt_bias);
        d_norm = to_device_f32(norm);
        d_output = to_device(output);
    }

    void calibrated_projections(const std::string& path) {
        artifact::Reader reader(path);
        if (reader.identity() != artifact::ArtifactIdentity{
            "qwen4/native-fp8-projection-qualification", "senfu-fp8-source"}) {
            throw std::invalid_argument("wrong calibrated GDN source");
        }
        const auto replace = [&](quantized_weight::PackedWeight& target,
                                 const char* role, int rows, int columns, int permutation) {
            const std::string name = std::string("model.language_model.layers.0.linear_attn.") + role + ".weight";
            const auto* object = reader.find(name);
            const auto* descriptor = object ? std::get_if<artifact::TensorDescriptor>(object) : nullptr;
            if (!descriptor || descriptor->format != artifact::NumericFormat::FP8_E4M3FN_TENSOR_F32M ||
                descriptor->layout != artifact::StorageLayout::TensorCalibratedV1 ||
                descriptor->shape != std::vector<std::uint64_t>{static_cast<unsigned>(rows), static_cast<unsigned>(columns)}) {
                throw std::invalid_argument("invalid calibrated GDN source role");
            }
            const auto raw = reader.payload(*descriptor).data;
            target.payload.resize(raw.size());
            std::memcpy(target.payload.data(), raw.data(), raw.size());
            const auto source = target.payload;
            const auto source_channel = [](int row) {
                const int head = row / kHeadDim;
                return ((head % kQkHeads) * 3 + head / kQkHeads) * kHeadDim + row % kHeadDim;
            };
            for (int row = 0; row < rows; ++row) {
                for (int column = 0; column < columns; ++column) {
                    const int source_row = permutation == 0 && row >= 2*kQkRows
                        ? 2*kQkRows + source_channel(row-2*kQkRows)
                        : permutation == 1 ? source_channel(row) : row;
                    const int source_column = permutation == 2 ? source_channel(column) : column;
                    target.payload[static_cast<std::size_t>(row)*columns+column] =
                        source[static_cast<std::size_t>(source_row)*columns+source_column];
                }
            }
            target.code_plane_bytes = static_cast<std::uint64_t>(rows)*columns;
            target.scale_plane_offset = target.code_plane_bytes;
            target.scale_plane_bytes = 8;
            target.weight.qtype = QType::FP8_E4M3FN_TENSOR_F32M;
            target.weight.layout = QuantLayout::TensorCalibrated;
            target.weight.scale_dtype = DType::FP32;
            target.weight.payload_bytes = target.payload.size();
            target.weight.scale_ne[0] = 2;
            target.weight.scale_nb[0] = 4;
            for (int axis=1; axis<4; ++axis) { target.weight.scale_nb[axis] = 8; }
        };
        replace(native_qkv, "in_proj_qkv", kQkvRows, kHidden, 0);
        replace(native_z, "in_proj_z", kValueRows, kHidden, 1);
        replace(native_output, "out_proj", kHidden, kValueRows, 2);
        qkv = native_qkv.payload; z = native_z.payload; output = native_output.payload;
        upload();
    }

    ops::GatedDeltaNetLayerWeights views() {
        return {
            native_projection_format(input_qtype) ? native_qkv.device_weight(d_qkv.p)
                : ggml_weight(d_qkv.p, d_qkv.bytes, input_qtype, kQkvRows, kHidden),
            native_projection_format(input_qtype) ? native_z.device_weight(d_z.p)
                : ggml_weight(d_z.p, d_z.bytes, input_qtype, kValueRows, kHidden),
            Tensor(d_a.p, DType::FP32, {kHidden, kValueHeads}),
            Tensor(d_b.p, DType::FP32, {kHidden, kValueHeads}),
            Tensor(d_conv.p, DType::FP32, {4, kQkvRows}),
            Tensor(d_ssm_a.p, DType::FP32, {kValueHeads}),
            Tensor(d_dt_bias.p, DType::FP32, {kValueHeads}),
            Tensor(d_norm.p, DType::FP32, {kHeadDim}),
            native_projection_format(input_qtype) ? native_output.device_weight(d_output.p)
                : ggml_weight(d_output.p, d_output.bytes, QType::GGML_Q6_K, kHidden, kValueRows),
        };
    }
};

struct OracleResult {
    std::vector<double> output;
    std::vector<double> conv_state;
    std::vector<double> ssm_state;
};

std::vector<double> project(const std::vector<float>& weight, std::int32_t rows,
                            const std::vector<float>& input, std::int32_t tokens) {
    std::vector<double> result(static_cast<std::size_t>(rows) * tokens);
    for (std::int32_t token = 0; token < tokens; ++token) {
        const std::size_t input_base = static_cast<std::size_t>(token) * kHidden;
        for (std::int32_t row = 0; row < rows; ++row) {
            const std::size_t weight_base = static_cast<std::size_t>(row) * kHidden;
            double sum = 0.0;
            for (std::int32_t d = 0; d < kHidden; ++d) {
                sum += static_cast<double>(weight[weight_base + d]) * input[input_base + d];
            }
            result[static_cast<std::size_t>(token) * rows + row] = sum;
        }
    }
    return result;
}

std::vector<double> project_quant(const std::vector<std::uint8_t>& weight, QType qtype,
                                  std::int32_t rows, std::int32_t columns,
                                  const std::vector<double>& input) {
    const std::size_t block_bytes = qtype == QType::GGML_Q5_K ? 176U : 210U;
    const std::size_t row_bytes = static_cast<std::size_t>(columns / 256) * block_bytes;
    std::vector<double> result(rows);
    for (std::int32_t row = 0; row < rows; ++row) {
        double sum = 0.0;
        const auto* row_data = weight.data() + static_cast<std::size_t>(row) * row_bytes;
        for (std::int32_t block_index = 0; block_index < columns / 256; ++block_index) {
            const auto decoded = decode_block(qtype, row_data + block_index * block_bytes);
            for (std::int32_t item = 0; item < 256; ++item) {
                sum += decoded[item] * input[static_cast<std::size_t>(block_index) * 256 + item];
            }
        }
        result[row] = sum;
    }
    return result;
}

OracleResult oracle(const Fixture& fixture, const std::vector<float>& input,
                    const std::vector<float>& initial_conv,
                    const std::vector<float>& initial_ssm, std::int32_t tokens) {
    std::vector<double> raw(static_cast<std::size_t>(kQkvRows) * tokens);
    std::vector<double> z(static_cast<std::size_t>(kValueRows) * tokens);
    for (std::int32_t token = 0; token < tokens; ++token) {
        std::vector<double> token_input(kHidden);
        for (std::int32_t d = 0; d < kHidden; ++d) {
            token_input[d] = input[static_cast<std::size_t>(token) * kHidden + d];
        }
        const auto raw_token = native_projection_format(fixture.input_qtype)
            ? native_projection_oracle(fixture.native_qkv, token_input)
            : project_quant(fixture.qkv, fixture.input_qtype, kQkvRows, kHidden, token_input);
        const auto z_token = native_projection_format(fixture.input_qtype)
            ? native_projection_oracle(fixture.native_z, token_input)
            : project_quant(fixture.z, fixture.input_qtype, kValueRows, kHidden, token_input);
        for (std::int32_t row = 0; row < kQkvRows; ++row) {
            raw[static_cast<std::size_t>(token) * kQkvRows + row] =
                represented_bf16(raw_token[row]);
        }
        for (std::int32_t row = 0; row < kValueRows; ++row) {
            z[static_cast<std::size_t>(token) * kValueRows + row] =
                represented_bf16(z_token[row]);
        }
    }
    const std::vector<double> a   = project(fixture.a, kValueHeads, input, tokens);
    const std::vector<double> b   = project(fixture.b, kValueHeads, input, tokens);
    std::vector<double> q(static_cast<std::size_t>(kQkRows) * tokens);
    std::vector<double> k(static_cast<std::size_t>(kQkRows) * tokens);
    std::vector<double> v(static_cast<std::size_t>(kValueRows) * tokens);
    std::vector<double> conv_state(static_cast<std::size_t>(kQkvRows) * 3);
    for (std::int32_t channel = 0; channel < kQkvRows; ++channel) {
        double s0 = initial_conv[channel];
        double s1 = initial_conv[kQkvRows + channel];
        double s2 = initial_conv[2 * kQkvRows + channel];
        for (std::int32_t token = 0; token < tokens; ++token) {
            const double p = raw[static_cast<std::size_t>(token) * kQkvRows + channel];
            const std::size_t weight_base = static_cast<std::size_t>(channel) * 4;
            const double sum = static_cast<double>(fixture.conv[weight_base]) * s0 +
                               static_cast<double>(fixture.conv[weight_base + 1]) * s1 +
                               static_cast<double>(fixture.conv[weight_base + 2]) * s2 +
                               static_cast<double>(fixture.conv[weight_base + 3]) * p;
            const double value = represented_bf16(silu(sum));
            if (channel < kQkRows) {
                q[static_cast<std::size_t>(token) * kQkRows + channel] = value;
            } else if (channel < 2 * kQkRows) {
                k[static_cast<std::size_t>(token) * kQkRows + channel - kQkRows] = value;
            } else {
                v[static_cast<std::size_t>(token) * kValueRows + channel - 2 * kQkRows] = value;
            }
            s0 = s1;
            s1 = s2;
            s2 = p;
        }
        conv_state[channel]                = represented_bf16(s0);
        conv_state[kQkvRows + channel]     = represented_bf16(s1);
        conv_state[2 * kQkvRows + channel] = represented_bf16(s2);
    }

    std::vector<double> q_norm(q.size());
    std::vector<double> k_norm(k.size());
    for (std::int32_t token = 0; token < tokens; ++token) {
        for (std::int32_t head = 0; head < kQkHeads; ++head) {
            const std::size_t base = static_cast<std::size_t>(token * kQkHeads + head) * kHeadDim;
            double q_sum = 0.0;
            double k_sum = 0.0;
            for (std::int32_t d = 0; d < kHeadDim; ++d) {
                q_sum += q[base + d] * q[base + d];
                k_sum += k[base + d] * k[base + d];
            }
            const double q_inverse = 1.0 / std::sqrt(q_sum + 1.0e-6);
            const double k_inverse = 1.0 / std::sqrt(k_sum + 1.0e-6);
            for (std::int32_t d = 0; d < kHeadDim; ++d) {
                q_norm[base + d] = q[base + d] * q_inverse;
                k_norm[base + d] = k[base + d] * k_inverse;
            }
        }
    }

    std::vector<double> state(initial_ssm.begin(), initial_ssm.end());
    std::vector<double> recurrent(static_cast<std::size_t>(kValueRows) * tokens);
    std::vector<double> delta(kHeadDim);
    const double query_scale = 1.0 / std::sqrt(static_cast<double>(kHeadDim));
    for (std::int32_t head = 0; head < kValueHeads; ++head) {
        // GGUF stores V-side heads tiled by repeat group, so represented head h consumes h%16.
        // The random per-head state/control fixtures make floor(h/3) a strict wrong-layout path.
        const std::int32_t qk_head = head % kQkHeads;
        const std::size_t state_base = static_cast<std::size_t>(head) * kHeadDim * kHeadDim;
        for (std::int32_t token = 0; token < tokens; ++token) {
            const std::size_t qk_base =
                static_cast<std::size_t>(token * kQkHeads + qk_head) * kHeadDim;
            const std::size_t value_base =
                static_cast<std::size_t>(token * kValueHeads + head) * kHeadDim;
            const std::size_t control = static_cast<std::size_t>(token) * kValueHeads + head;
            const double g = static_cast<double>(static_cast<float>(
                static_cast<double>(fixture.ssm_a[head]) *
                softplus(a[control] + fixture.dt_bias[head])));
            const double beta =
                static_cast<double>(static_cast<float>(sigmoid(b[control])));
            const double alpha = std::exp(g);
            for (std::int32_t row = 0; row < kHeadDim; ++row) {
                double dot = 0.0;
                const std::size_t row_base = state_base + static_cast<std::size_t>(row) * kHeadDim;
                for (std::int32_t column = 0; column < kHeadDim; ++column) {
                    dot += state[row_base + column] * k_norm[qk_base + column];
                }
                delta[row] = beta * (v[value_base + row] - alpha * dot);
            }
            for (std::int32_t row = 0; row < kHeadDim; ++row) {
                const std::size_t row_base = state_base + static_cast<std::size_t>(row) * kHeadDim;
                for (std::int32_t column = 0; column < kHeadDim; ++column) {
                    state[row_base + column] = static_cast<double>(static_cast<float>(
                        alpha * state[row_base + column] +
                        delta[row] * k_norm[qk_base + column]));
                }
                double dot = 0.0;
                for (std::int32_t column = 0; column < kHeadDim; ++column) {
                    dot += state[row_base + column] * q_norm[qk_base + column];
                }
                recurrent[value_base + row] = represented_bf16(query_scale * dot);
            }
        }
    }

    std::vector<double> normalized_gated(recurrent.size());
    for (std::int32_t token = 0; token < tokens; ++token) {
        for (std::int32_t head = 0; head < kValueHeads; ++head) {
            const std::size_t base =
                static_cast<std::size_t>(token * kValueHeads + head) * kHeadDim;
            double sum = 0.0;
            for (std::int32_t d = 0; d < kHeadDim; ++d) {
                sum += recurrent[base + d] * recurrent[base + d];
            }
            const double inverse = 1.0 / std::sqrt(sum / kHeadDim + 1.0e-6);
            for (std::int32_t d = 0; d < kHeadDim; ++d) {
                normalized_gated[base + d] = represented_bf16(
                    recurrent[base + d] * inverse * fixture.norm[d] *
                    sigmoid(represented_bf16(z[base + d])));
            }
        }
    }
    std::vector<double> output(static_cast<std::size_t>(kHidden) * tokens);
    for (std::int32_t token = 0; token < tokens; ++token) {
        std::vector<double> token_input(kValueRows);
        for (std::int32_t d = 0; d < kValueRows; ++d) {
            token_input[d] = normalized_gated[static_cast<std::size_t>(token) * kValueRows + d];
        }
        const auto token_output = native_projection_format(fixture.input_qtype)
            ? native_projection_oracle(fixture.native_output, token_input)
            : project_quant(fixture.output, QType::GGML_Q6_K, kHidden, kValueRows, token_input);
        for (std::int32_t row = 0; row < kHidden; ++row) {
            output[static_cast<std::size_t>(token) * kHidden + row] = token_output[row];
        }
    }
    return {std::move(output), std::move(conv_state), std::move(state)};
}

int run_complete_case(Fixture& fixture, std::int32_t tokens, const char* label) {
    std::vector<float> input(static_cast<std::size_t>(kHidden) * tokens);
    std::vector<float> initial_conv(static_cast<std::size_t>(kQkvRows) * 3);
    std::vector<float> initial_ssm(static_cast<std::size_t>(kHeadDim) * kHeadDim * kValueHeads);
    fill_uniform(input, 8201U, -0.20F, 0.20F);
    fill_uniform(initial_conv, 8202U, -0.05F, 0.05F);
    fill_uniform(initial_ssm, 8203U, -0.002F, 0.002F);
    round_to_bf16(input);
    round_to_bf16(initial_conv);
    const OracleResult expected = oracle(fixture, input, initial_conv, initial_ssm, tokens);
    DeviceBuffer d_input      = to_device_bf16(input);
    DeviceBuffer d_conv_in    = to_device_bf16(initial_conv);
    DeviceBuffer d_ssm_in     = to_device_f32(initial_ssm);
    GuardedDeviceBuffer d_conv_out(initial_conv.size() * 2);
    GuardedDeviceBuffer d_ssm_out(initial_ssm.size() * sizeof(float));
    GuardedDeviceBuffer d_output(static_cast<std::size_t>(kHidden) * tokens * 2);
    d_conv_out.fill(0xff);
    d_ssm_out.fill(0xff);
    d_output.fill(0xff);
    Tensor x(d_input.p, DType::BF16, {kHidden, tokens});
    Tensor conv_in(d_conv_in.p, DType::BF16, {kQkvRows, 3});
    Tensor conv_out(d_conv_out.data(), DType::BF16, {kQkvRows, 3});
    Tensor ssm_in(d_ssm_in.p, DType::FP32, {kHeadDim, kHeadDim, kValueHeads});
    Tensor ssm_out(d_ssm_out.data(), DType::FP32, {kHeadDim, kHeadDim, kValueHeads});
    Tensor output(d_output.data(), DType::BF16, {kHidden, tokens});
    auto weights = fixture.views();
    const std::size_t full_bytes = ops::gated_delta_net_layer_workspace_capacity_bytes(
        tokens, weights.qkv.qtype, weights.z.qtype, weights.output.qtype);
    WorkspaceArena full_workspace(full_bytes);
    ops::gated_delta_net_layer(x, weights, conv_in, conv_out, ssm_in, ssm_out, output,
                               full_workspace, nullptr);
    cuda_synchronize();

    int failures = verify_reduction(std::string(label) + " panel output",
                                    from_device_bf16(d_output.data(), expected.output.size()),
                                    expected.output, kOutputCriterion);
    failures += verify_reduction(std::string(label) + " distinct conv state",
                                 from_device_bf16(d_conv_out.data(), initial_conv.size()),
                                 expected.conv_state, conv_state_criterion(fixture.input_qtype));
    std::vector<double> actual_ssm(initial_ssm.size());
    const std::vector<float> actual_ssm_f32 =
        from_device<float>(d_ssm_out.data(), initial_ssm.size());
    std::copy(actual_ssm_f32.begin(), actual_ssm_f32.end(), actual_ssm.begin());
    failures += verify_reduction(std::string(label) + " distinct SSM state", actual_ssm,
                                 expected.ssm_state,
                                 kStateCriterion);
    const std::string rollback_conv_label = std::string(label) + " rollback conv input";
    const std::string rollback_ssm_label = std::string(label) + " rollback SSM input";
    failures += verify_exact(rollback_conv_label.c_str(), from_device<std::uint16_t>(
                                 d_conv_in, initial_conv.size()), bf16_bits(initial_conv));
    failures += verify_exact(rollback_ssm_label.c_str(),
                             from_device<float>(d_ssm_in, initial_ssm.size()), initial_ssm);

    DeviceBuffer sequential_conv = to_device_bf16(initial_conv);
    DeviceBuffer sequential_ssm  = to_device_f32(initial_ssm);
    GuardedDeviceBuffer sequential_output(input.size() * 2);
    sequential_output.fill(0xff);
    Tensor sequential_conv_tensor(sequential_conv.p, DType::BF16, {kQkvRows, 3});
    Tensor sequential_ssm_tensor(sequential_ssm.p, DType::FP32,
                                 {kHeadDim, kHeadDim, kValueHeads});
    Tensor sequential_output_tensor(sequential_output.data(), DType::BF16, {kHidden, tokens});
    const std::size_t step_bytes = ops::gated_delta_net_layer_workspace_capacity_bytes(
        1, weights.qkv.qtype, weights.z.qtype, weights.output.qtype);
    WorkspaceArena step_workspace(step_bytes);
    for (std::int32_t token = 0; token < tokens; ++token) {
        Tensor x_step   = x.slice(1, token, 1);
        Tensor out_step = sequential_output_tensor.slice(1, token, 1);
        ops::gated_delta_net_layer(x_step, weights, sequential_conv_tensor,
                                   sequential_conv_tensor, sequential_ssm_tensor,
                                   sequential_ssm_tensor, out_step, step_workspace, nullptr);
    }
    cuda_synchronize();
    failures += verify_reduction(std::string(label) + " repeated T=1 output",
                                 from_device_bf16(sequential_output.data(), input.size()),
                                 expected.output, kOutputCriterion);
    const std::vector<double> sequential_ssm_actual = [&] {
        const std::vector<float> values =
            from_device<float>(sequential_ssm, initial_ssm.size());
        return std::vector<double>(values.begin(), values.end());
    }();
    failures += verify_reduction(std::string(label) + " repeated T=1 SSM state", sequential_ssm_actual,
                                 expected.ssm_state, kStateCriterion);
    failures += verify_reduction(std::string(label) + " repeated T=1 conv state",
                                 from_device_bf16(sequential_conv, initial_conv.size()),
                                 expected.conv_state, conv_state_criterion(fixture.input_qtype));
    failures += d_conv_out.verify_guards(std::string(label) + " conv state");
    failures += d_ssm_out.verify_guards(std::string(label) + " SSM state");
    failures += d_output.verify_guards(std::string(label) + " output");
    failures += sequential_output.verify_guards(std::string(label) + " repeated output");
    if (full_workspace.used() != 0 || full_workspace.peak_used() != full_bytes ||
        step_workspace.used() != 0 || step_workspace.peak_used() != step_bytes) {
        std::cerr << "FAIL " << label << " workspace query/high-water mismatch\n";
        ++failures;
    }
    return failures;
}

int run_partition_case(Fixture& fixture, std::int32_t tokens,
                       std::initializer_list<std::int32_t> chunks, const char* label) {
    std::int32_t total = 0;
    std::int32_t maximum_chunk = 0;
    for (const std::int32_t chunk : chunks) {
        total += chunk;
        maximum_chunk = std::max(maximum_chunk, chunk);
    }
    if (total != tokens || maximum_chunk <= 0) {
        std::cerr << "FAIL invalid GDN layer partition fixture\n";
        return 1;
    }

    std::vector<float> input(static_cast<std::size_t>(kHidden) * tokens);
    std::vector<float> initial_conv(static_cast<std::size_t>(kQkvRows) * 3);
    std::vector<float> initial_ssm(static_cast<std::size_t>(kHeadDim) * kHeadDim * kValueHeads);
    fill_uniform(input, 9100U + tokens, -0.20F, 0.20F);
    fill_uniform(initial_conv, 9200U + tokens, -0.05F, 0.05F);
    fill_uniform(initial_ssm, 9300U + tokens, -0.002F, 0.002F);
    round_to_bf16(input);
    round_to_bf16(initial_conv);

    DeviceBuffer d_input = to_device_bf16(input);
    DeviceBuffer full_conv = to_device_bf16(initial_conv);
    DeviceBuffer full_ssm = to_device_f32(initial_ssm);
    GuardedDeviceBuffer full_output(input.size() * sizeof(std::uint16_t));
    DeviceBuffer partition_conv = to_device_bf16(initial_conv);
    DeviceBuffer partition_ssm = to_device_f32(initial_ssm);
    GuardedDeviceBuffer partition_output(input.size() * sizeof(std::uint16_t));
    full_output.fill(0xff);
    partition_output.fill(0xff);

    Tensor x(d_input.p, DType::BF16, {kHidden, tokens});
    Tensor full_conv_t(full_conv.p, DType::BF16, {kQkvRows, 3});
    Tensor full_ssm_t(full_ssm.p, DType::FP32, {kHeadDim, kHeadDim, kValueHeads});
    Tensor full_output_t(full_output.data(), DType::BF16, {kHidden, tokens});
    Tensor partition_conv_t(partition_conv.p, DType::BF16, {kQkvRows, 3});
    Tensor partition_ssm_t(partition_ssm.p, DType::FP32, {kHeadDim, kHeadDim, kValueHeads});
    Tensor partition_output_t(partition_output.data(), DType::BF16, {kHidden, tokens});
    auto weights = fixture.views();

    const std::size_t full_bytes = ops::gated_delta_net_layer_workspace_capacity_bytes(
        tokens, weights.qkv.qtype, weights.z.qtype, weights.output.qtype);
    WorkspaceArena full_workspace(full_bytes);
    ops::gated_delta_net_layer(x, weights, full_conv_t, full_conv_t, full_ssm_t, full_ssm_t,
                               full_output_t, full_workspace, nullptr);

    const std::size_t partition_bytes =
        ops::gated_delta_net_layer_workspace_capacity_bytes(
            maximum_chunk, weights.qkv.qtype, weights.z.qtype, weights.output.qtype);
    WorkspaceArena partition_workspace(partition_bytes);
    std::int32_t offset = 0;
    for (const std::int32_t chunk : chunks) {
        Tensor x_chunk = x.slice(1, offset, chunk);
        Tensor output_chunk = partition_output_t.slice(1, offset, chunk);
        ops::gated_delta_net_layer(x_chunk, weights, partition_conv_t, partition_conv_t,
                                   partition_ssm_t, partition_ssm_t, output_chunk,
                                   partition_workspace, nullptr);
        offset += chunk;
    }
    cuda_synchronize();

    const std::string prefix = std::string(label) + " T=" + std::to_string(tokens);
    const auto full_output_values = from_device_bf16(full_output.data(), input.size());
    const auto partition_output_values = from_device_bf16(partition_output.data(), input.size());
    std::vector<double> full_reference(full_output_values.begin(), full_output_values.end());
    int failures = verify_reduction(prefix + " partition output", partition_output_values,
                                    full_reference, kOutputCriterion);
    if (native_projection_format(fixture.input_qtype)) {
        // Persistent convolution history is the last three represented QKV projections.
        // Qualify both reduction routes against that oracle, not against each other's bits.
        std::vector<double> expected_conv(initial_conv.begin(), initial_conv.end());
        for (int history = 0; history < 3; ++history) {
            const int token = tokens - 3 + history;
            if (token < 0) {
                std::copy_n(initial_conv.begin() + (tokens + history) * kQkvRows, kQkvRows,
                            expected_conv.begin() + history * kQkvRows);
                continue;
            }
            const auto begin = input.begin() + static_cast<std::size_t>(token) * kHidden;
            const std::vector<double> token_input(begin, begin + kHidden);
            const auto projected = native_projection_oracle(fixture.native_qkv, token_input);
            for (int row = 0; row < kQkvRows; ++row) {
                expected_conv[static_cast<std::size_t>(history) * kQkvRows + row] =
                    represented_bf16(projected[row]);
            }
        }
        failures += verify_reduction(prefix + " full conv state",
            from_device_bf16(full_conv, initial_conv.size()), expected_conv, conv_state_criterion(fixture.input_qtype));
        failures += verify_reduction(prefix + " partition conv state",
            from_device_bf16(partition_conv, initial_conv.size()), expected_conv, conv_state_criterion(fixture.input_qtype));
    } else {
        failures += verify_exact((prefix + " partition conv state").c_str(),
                                 from_device<std::uint16_t>(partition_conv, initial_conv.size()),
                                 from_device<std::uint16_t>(full_conv, initial_conv.size()));
    }
    const auto full_ssm_values = from_device<float>(full_ssm, initial_ssm.size());
    const auto partition_ssm_values = from_device<float>(partition_ssm, initial_ssm.size());
    failures += verify_reduction(prefix + " partition SSM state",
                                 std::vector<double>(partition_ssm_values.begin(),
                                                     partition_ssm_values.end()),
                                 std::vector<double>(full_ssm_values.begin(), full_ssm_values.end()),
                                 kStateCriterion);
    failures += full_output.verify_guards((prefix + " full output").c_str());
    failures += partition_output.verify_guards((prefix + " partition output").c_str());
    if (full_workspace.used() != 0 || full_workspace.peak_used() != full_bytes ||
        partition_workspace.used() != 0 ||
        partition_workspace.peak_used() != partition_bytes) {
        std::cerr << "FAIL " << prefix << " workspace query/high-water mismatch\n";
        ++failures;
    }
    return failures;
}

double relative_l2(const std::vector<double>& actual, const std::vector<double>& expected) {
    double error=0, norm=0;
    for (std::size_t i=0; i<actual.size(); ++i) {
        error += (actual[i]-expected[i])*(actual[i]-expected[i]);
        norm += expected[i]*expected[i];
    }
    return std::sqrt(error/std::max(norm,1e-300));
}

int calibrated_gdn(const std::string& root) {
    Fixture original(root + "/qwen4-layer-0.ninfer");
    Fixture fixture(root + "/qwen4-layer-0.ninfer");
    fixture.calibrated_projections(root + "/qwen4-fp8-projections.ninfer");
    constexpr int tokens=65;
    std::vector<float> input(kHidden*tokens), initial_conv(kQkvRows*3),
                       initial_ssm(kHeadDim*kHeadDim*kValueHeads);
    fill_uniform(input,8201U,-.20F,.20F); round_to_bf16(input);
    fill_uniform(initial_conv,8202U,-.05F,.05F); round_to_bf16(initial_conv);
    fill_uniform(initial_ssm,8203U,-.002F,.002F);
    const auto expected=oracle(fixture,input,initial_conv,initial_ssm,tokens);
    const auto original_expected=oracle(original,input,initial_conv,initial_ssm,tokens);
    std::cout << "GDN source weight-only ideal relL2 output=" << relative_l2(expected.output,original_expected.output)
              << " conv=" << relative_l2(expected.conv_state,original_expected.conv_state)
              << " state=" << relative_l2(expected.ssm_state,original_expected.ssm_state) << '\n';
    using Policy=ops::GatedDeltaNetProjectionPolicy;
    using LP=ops::LinearPolicy;
    const std::array<Policy,6> policies{{{}, {LP::AllowA8,LP::A16Only,LP::A16Only},
        {LP::A16Only,LP::AllowA8,LP::A16Only}, {LP::A16Only,LP::A16Only,LP::AllowA8},
        {LP::AllowA8,LP::AllowA8,LP::A16Only}, {LP::A16Only,LP::AllowA8,LP::AllowA8}}};
    constexpr std::array names{"A16","QKV-A8","Z-A8","OUT-A8","QKV-Z-A8","Z-OUT-A8"};
    OracleResult a16;
    int failures=0;
    auto dx=to_device_bf16(input);
    const auto weights=fixture.views();
    for(std::size_t route=0; route<policies.size(); ++route) {
        for(bool partitioned:{false,true}) {
            auto dc=to_device_bf16(initial_conv),ds=to_device_f32(initial_ssm);
            GuardedDeviceBuffer dy(input.size()*2);
            Tensor x(dx.p,DType::BF16,{kHidden,tokens}),y(dy.data(),DType::BF16,{kHidden,tokens}),
                conv(dc.p,DType::BF16,{kQkvRows,3}),ssm(ds.p,DType::FP32,{kHeadDim,kHeadDim,kValueHeads});
            const auto policy=policies[route];
            const auto capacity=ops::gated_delta_net_layer_workspace_capacity_bytes(
                partitioned?64:65,weights.qkv.qtype,weights.z.qtype,weights.output.qtype,policy);
            WorkspaceArena workspace(capacity);
            for(int start=0;start<tokens;) {
                const int count=partitioned && start==0?64:tokens-start;
                auto cx=x.slice(1,start,count),cy=y.slice(1,start,count);
                ops::gated_delta_net_layer(cx,weights,conv,conv,ssm,ssm,cy,workspace,nullptr,policy);
                start+=count;
            }
            cuda_synchronize();
            const auto state=from_device<float>(ds.p,initial_ssm.size());
            OracleResult actual{from_device_bf16(dy.data(),input.size()),
                from_device_bf16(dc.p,initial_conv.size()), {state.begin(),state.end()}};
            const std::string label=std::string("native calibrated GDN ")+names[route]+(partitioned?" 64+1":" T65");
            failures+=verify_reduction(label+" output",actual.output,expected.output,
                route==0?kOutputCriterion:kA8OutputCriterion);
            const bool qkv_a8=policy.qkv==LP::AllowA8;
            failures+=verify_reduction(label+" conv",actual.conv_state,expected.conv_state,
                qkv_a8?kA8ConvCriterion:kBf16ConvStateCriterion);
            failures+=verify_reduction(label+" state",actual.ssm_state,expected.ssm_state,
                qkv_a8?kA8StateCriterion:kStateCriterion);
            failures+=dy.verify_guards(label);
            if(workspace.used()!=0 || workspace.peak_used()>capacity ||
               capacity-workspace.peak_used()>=256) {
                std::cerr << "FAIL " << label << " workspace accounting\n"; ++failures;
            }
            std::cout << label << " oracle relL2 output=" << relative_l2(actual.output,expected.output)
                << " conv=" << relative_l2(actual.conv_state,expected.conv_state)
                << " state=" << relative_l2(actual.ssm_state,expected.ssm_state);
            if(route==0 && !partitioned) { a16=actual; }
            std::cout << " sameweightsA16 relL2 output=" << relative_l2(actual.output,a16.output)
                << " state=" << relative_l2(actual.ssm_state,a16.ssm_state) << '\n';
        }
    }
    return failures;
}

} // namespace

#ifdef NINFER_QWEN4_SEQUENCE_COMPONENTS
namespace ninfer::test::qwen4_sequence {
int gdn_a8_input_diagnostic(const std::string& root,const Result& input) {
    Fixture fixture(root+"/qwen4-layer-0.ninfer",0);
    fixture.calibrated_projections(root+"/qwen4-fp8-projections.ninfer");
    float multiplier;
    std::memcpy(&multiplier,fixture.native_z.payload.data()+fixture.native_z.scale_plane_offset+4,4);
    const double limit=448.*multiplier;
    std::size_t clipped_count=0,max_index=0;
    for(std::size_t i=0;i<input.actual.size();++i) {
        clipped_count+=std::abs(input.actual[i])>limit;
        if(std::abs(input.actual[i])>std::abs(input.actual[max_index])) { max_index=i; }
    }
    std::cout<<"GDN_Z_A8_INPUT input_scale="<<multiplier<<" finite_limit="<<limit
        <<" max_abs="<<std::abs(input.actual[max_index])<<" max_token="<<max_index/kHidden
        <<" clipped_count="<<clipped_count<<" elements="<<input.actual.size()<<'\n';
    const int tokens=input.actual.size()/kHidden;
    auto dx=to_device_bf16(input.actual);
    GuardedDeviceBuffer dy(std::size_t(tokens)*kValueRows*2);
    Tensor x(dx.p,DType::BF16,{kHidden,tokens}),y(dy.data(),DType::BF16,{kValueRows,tokens});
    WorkspaceArena workspace(ops::linear_workspace_capacity_bytes(QType::FP8_E4M3FN_TENSOR_F32M,
        kValueRows,kHidden,ops::LinearPolicy::AllowA8,tokens,tokens));
    ops::linear(x,fixture.native_z.device_weight(fixture.d_z.p),y,ops::LinearPolicy::AllowA8,workspace,nullptr);
    cuda_synchronize();
    const auto actual=from_device_bf16(dy.data(),std::size_t(tokens)*kValueRows);
    int failures=0;
    std::vector<int> selected{15,static_cast<int>(max_index/kHidden)};
    std::sort(selected.begin(),selected.end());
    selected.erase(std::unique(selected.begin(),selected.end()),selected.end());
    for(int token:selected) {
        std::vector<double> original(kHidden),clamped(kHidden),quantized(kHidden),guarded(kHidden);
        float token_max=0.F;
        for(int d=0;d<kHidden;++d) { token_max=std::max(token_max,std::abs(input.actual[token*kHidden+d])); }
        const float guarded_scale=std::max(multiplier,token_max/448.F);
        for(int d=0;d<kHidden;++d) {
            original[d]=input.actual[token*kHidden+d];
            clamped[d]=std::clamp(original[d],-limit,limit);
            const float divided=static_cast<float>(original[d])/multiplier;
            // Independent nearest-value enumeration, identical exact codec oracle as
            // the Linear pack suite; this diagnoses approximation, never changes GDN's oracle.
            quantized[d]=quantized_weight::detail::decode_e4m3fn(
                quantized_weight::detail::encode_e4m3fn(divided))*multiplier;
            guarded[d]=quantized_weight::detail::decode_e4m3fn(
                quantized_weight::detail::encode_e4m3fn(static_cast<float>(original[d])/guarded_scale))*guarded_scale;
        }
        const auto ideal=native_projection_oracle(fixture.native_z,original);
        const auto clamp_projection=native_projection_oracle(fixture.native_z,clamped);
        const auto quant_projection=native_projection_oracle(fixture.native_z,quantized);
        const auto guard_projection=native_projection_oracle(fixture.native_z,guarded);
        const std::vector<double> gpu(actual.begin()+token*kValueRows,actual.begin()+(token+1)*kValueRows);
        failures+=verify_reduction("native EOS guarded Z Linear",gpu,ideal,
            ReductionCriterion{.04,1./256,.06});
        const auto relative=[](const auto& a,const auto& b) {
            return compute_reduction_stats(a.data(),b.data(),a.size()).relative_l2;
        };
        std::cout<<"GDN_Z_A8_ATTRIBUTION token="<<token<<" GPU_vs_ideal="<<relative(gpu,ideal)
            <<" clamp_vs_ideal="<<relative(clamp_projection,ideal)
            <<" quant_vs_clamp="<<relative(quant_projection,clamp_projection)
            <<" quant_vs_ideal="<<relative(quant_projection,ideal)
            <<" GPU_vs_static_quant="<<relative(gpu,quant_projection)
            <<" guarded_scale="<<guarded_scale<<" guarded_vs_ideal="<<relative(guard_projection,ideal)
            <<" GPU_vs_guarded="<<relative(gpu,guard_projection)<<'\n';
    }
    return failures+dy.verify_guards("GDN Z diagnosis output");
}
static Result gdn_component(Fixture& fixture, const Result& input, bool partitioned,
                            ops::GatedDeltaNetProjectionPolicy policy = {}) {
    const int tokens=input.actual.size()/kHidden;
    std::vector<float> initial_conv(kQkvRows*3),initial_ssm(kHeadDim*kHeadDim*kValueHeads);
    const auto local=oracle(fixture,input.actual,initial_conv,initial_ssm,tokens);
    const auto propagated=oracle(fixture,input.reference,initial_conv,initial_ssm,tokens);
    auto dx=to_device_bf16(input.actual),dc=to_device_bf16(initial_conv),ds=to_device_f32(initial_ssm);
    GuardedDeviceBuffer dy(input.actual.size()*2);
    Tensor x(dx.p,DType::BF16,{kHidden,tokens}),y(dy.data(),DType::BF16,{kHidden,tokens}),
        conv(dc.p,DType::BF16,{kQkvRows,3}),ssm(ds.p,DType::FP32,{kHeadDim,kHeadDim,kValueHeads});
    const auto weights=fixture.views();
    WorkspaceArena workspace(ops::gated_delta_net_layer_workspace_capacity_bytes(tokens,
        weights.qkv.qtype,weights.z.qtype,weights.output.qtype,policy));
    for(int start=0;start<tokens;) {
        const int chunk=partitioned && start==0 ? std::max(1,tokens-1) : tokens-start;
        auto cx=x.slice(1,start,chunk),cy=y.slice(1,start,chunk);
        ops::gated_delta_net_layer(cx,weights,conv,conv,ssm,ssm,cy,workspace,nullptr,policy);
        start+=chunk;
    }
    cuda_synchronize();
    Result result{from_device_bf16(dy.data(),input.actual.size()),represented(propagated.output)};
    const bool a8=policy.qkv==ops::LinearPolicy::AllowA8 || policy.z==ops::LinearPolicy::AllowA8 ||
        policy.output==ops::LinearPolicy::AllowA8;
    const bool qkv_a8=policy.qkv==ops::LinearPolicy::AllowA8;
    result.failures=verify_reduction("sequence GDN local output",wide(result.actual),local.output,
        a8?kA8OutputCriterion:kOutputCriterion);
    result.failures+=verify_reduction("sequence GDN final conv",from_device_bf16(dc.p,initial_conv.size()),
        local.conv_state,qkv_a8?kA8ConvCriterion:kBf16ConvStateCriterion);
    const auto state=from_device<float>(ds.p,initial_ssm.size());
    result.failures+=verify_reduction("sequence GDN final SSM",wide(state),local.ssm_state,
        qkv_a8?kA8StateCriterion:kStateCriterion);
    result.failures+=dy.verify_guards("sequence GDN output");
    return result;
}
Result gdn(const std::string& path, int layer, const Result& input, bool partitioned) {
    Fixture fixture(path,layer);
    return gdn_component(fixture,input,partitioned);
}
Result gdn_calibrated(const std::string& root, const std::string& path, int layer,
                      const Result& input, bool partitioned, int mask) {
    if(layer!=0 || mask<0 || mask>7 || (mask&5)==5) {
        throw std::invalid_argument("unqualified calibrated GDN source/policy");
    }
    Fixture fixture(path,layer);
    fixture.calibrated_projections(root+"/qwen4-fp8-projections.ninfer");
    using P=ops::LinearPolicy;
    return gdn_component(fixture,input,partitioned,
        {mask&1?P::AllowA8:P::A16Only,mask&2?P::AllowA8:P::A16Only,mask&4?P::AllowA8:P::A16Only});
}
}
#else
int main(int argc, char** argv) {
    if (const int unavailable = require_cuda()) { return unavailable; }
    if (argc == 2 && std::string_view(argv[1]) == "--native-fp8-real") {
        const char* root=std::getenv("NINFER_QWEN4_NATIVE_LAYERS");
        if(!root) { return 77; }
        return calibrated_gdn(root)?1:0;
    }
    if (argc == 2 && std::string_view(argv[1]) == "--native-real") {
        const char* root = std::getenv("NINFER_QWEN4_NATIVE_LAYERS");
        if (!root) { return 77; }
        Fixture fixture(std::string(root) + "/qwen4-layer-0.ninfer");
        int failures = bf16_state_rounding_witness();
        failures += run_complete_case(fixture, 1, "native source GDN T1");
        failures += run_complete_case(fixture, 3, "native source GDN T3");
        failures += run_partition_case(fixture, 65, {32, 33}, "native source GDN continuation");
        std::cout << (failures ? "FAIL" : "PASS") << " native source GDN\n";
        return failures ? 1 : 0;
    }
    Fixture q5_fixture(QType::GGML_Q5_K);
    int failures = bf16_state_rounding_witness() + conv_source_layout_witness();
    failures += run_complete_case(q5_fixture, kPrompt, "GDN Q5_K/Q6_K");
    failures += run_partition_case(q5_fixture, 64, {31, 33}, "GDN chunk boundary");
    failures += run_partition_case(q5_fixture, 65, {64, 1}, "GDN chunk tail");
    Fixture q6_fixture(QType::GGML_Q6_K);
    failures += run_complete_case(q6_fixture, 1, "GDN layer-2 Q6_K/Q6_K");
    for (QType type : {QType::BF16_CTRL, QType::NVFP4, QType::FP8_E4M3FN_ROW_BF16S}) {
        Fixture native(type);
        failures += run_complete_case(native, 3, type == QType::BF16_CTRL ? "GDN BF16" :
            type == QType::NVFP4 ? "GDN NVFP4" : "GDN FP8");
        failures += run_partition_case(native, 65, {32, 33}, "GDN native partition");
        failures += run_partition_case(native, 257, {128, 129}, "GDN native wide partition");
        if (type == QType::NVFP4) {
            native.native_z = native_projection_fixture(QType::FP8_E4M3FN_ROW_BF16S,
                                                          kValueRows, kHidden, 8102U);
            native.d_z = to_device(native.native_z.payload);
            failures += run_complete_case(native, 3, "GDN mixed NVFP4/FP8");
        }
    }
    try {
        const std::size_t broad = ops::gated_delta_net_layer_workspace_capacity_bytes(
            4096, QType::GGML_Q5_K, QType::GGML_Q5_K, QType::GGML_Q6_K);
        if (broad <= ops::gated_delta_net_layer_workspace_capacity_bytes(
                65, QType::GGML_Q5_K, QType::GGML_Q5_K, QType::GGML_Q6_K)) {
            std::cerr << "FAIL GDN broad workspace capacity\n";
            ++failures;
        }
    } catch (const std::invalid_argument&) {
        std::cerr << "FAIL GDN rejected T=4096 capacity\n";
        ++failures;
    }
    try {
        (void)ops::gated_delta_net_layer_workspace_capacity_bytes(
            4097, QType::GGML_Q5_K, QType::GGML_Q5_K, QType::GGML_Q6_K);
        std::cerr << "FAIL GDN accepted T=4097 capacity\n";
        ++failures;
    } catch (const std::invalid_argument&) {}
    for (auto z : {ops::LinearPolicy::A16Only, ops::LinearPolicy::AllowA8}) {
        try {
            (void)ops::gated_delta_net_layer_workspace_capacity_bytes(65,
                QType::FP8_E4M3FN_TENSOR_F32M, QType::FP8_E4M3FN_TENSOR_F32M,
                QType::FP8_E4M3FN_TENSOR_F32M,
                {ops::LinearPolicy::AllowA8,z,ops::LinearPolicy::AllowA8});
            std::cerr << "FAIL GDN admitted rejected QKV/output A8 combination\n"; ++failures;
        } catch (const std::invalid_argument&) {}
    }
    for (auto type : {QType::BF16_CTRL, QType::FP8_E4M3FN_ROW_BF16S}) {
        try {
            (void)ops::gated_delta_net_layer_workspace_capacity_bytes(65,type,type,type,
                {ops::LinearPolicy::A16Only,ops::LinearPolicy::AllowA8,ops::LinearPolicy::A16Only});
            std::cerr << "FAIL GDN admitted A8 for an unqualified projection format\n"; ++failures;
        } catch (const std::invalid_argument&) {}
    }
    std::cout << (failures ? "FAIL" : "OK") << " gated_delta_net_layer\n";
    return failures == 0 ? 0 : 1;
}
#endif
