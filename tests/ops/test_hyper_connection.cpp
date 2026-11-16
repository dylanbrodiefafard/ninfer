#include "ninfer/ops/hyper_connection.h"

#include "core/decode_graph.h"
#include "core/device.h"
#include "ops/op_tester.h"

#include <cmath>
#include <cstdint>
#include <exception>
#include <iostream>
#include <random>
#include <vector>

using namespace ninfer;
using namespace ninfer::test;

namespace {

constexpr std::int32_t kHidden  = ops::kHyperConnectionHidden;
constexpr std::int32_t kStreams = ops::kHyperConnectionStreams;
constexpr std::int32_t kFlat    = ops::kHyperConnectionFlat;
constexpr std::int32_t kRank    = ops::kHyperConnectionRank;

// x and the injected residual are FP32 results rounded once to BF16 (relative 2^-8); the absolute
// term covers FP32 reassociation of near-cancelling stream sums. write_scale stays FP32: its
// error is the FP32 reassociation of a 10240-term dot product before the sigmoid.
constexpr PointwiseCriterion bf16_output_criterion() {
    return {/*absolute*/ 1.0e-4, /*relative*/ 4.05e-3};
}

constexpr PointwiseCriterion write_scale_criterion() {
    return {/*absolute*/ 1.0e-5, /*relative*/ 1.0e-4};
}

std::vector<std::uint16_t> encode_bf16(const std::vector<float>& values) {
    std::vector<std::uint16_t> bits(values.size());
    for (std::size_t i = 0; i < values.size(); ++i) { bits[i] = f32_to_bf16(values[i]); }
    return bits;
}

double sigmoid(double v) { return 1.0 / (1.0 + std::exp(-v)); }

// Independent exact E4M3FN decode: sign, 4-bit exponent (bias 7), 3-bit fraction; exponent 0 is
// subnormal (fraction * 2^-9). The NaN codes 0x7f/0xff never occur in a fixture.
double decode_e4m3fn(std::uint8_t code) {
    const int exponent = (code >> 3) & 15;
    const int fraction = code & 7;
    const double scaled =
        exponent == 0 ? std::ldexp(fraction, -9) : std::ldexp(1.0 + fraction / 8.0, exponent - 7);
    return (code & 0x80) != 0 ? -scaled : scaled;
}

// One FP8_E4M3FN_ROW_BF16S matrix [rows, inputs] in its row-scale-v1 payload: the code plane, then
// at the next 256-byte boundary one BF16 multiplier per row.
class Fp8Matrix {
public:
    // Codes are uniform over the 254 finite E4M3FN words (zeros and subnormals included). Each row
    // gets its own multiplier, which places the represented row near row_norm (times a factor in
    // [0.5, 1.5)); row 0 is a zero row: zero multiplier, signed-zero codes.
    Fp8Matrix(std::int32_t rows, std::int32_t inputs, std::uint32_t seed, double row_norm)
        : rows_(rows), inputs_(inputs), codes_(static_cast<std::size_t>(rows) * inputs),
          scales_(static_cast<std::size_t>(rows)), scale_offset_((codes_.size() + 255) / 256 * 256),
          payload_(scale_offset_ + scales_.size() * sizeof(std::uint16_t)) {
        std::mt19937 generator(seed);
        std::uniform_int_distribution<int> code_distribution(0, 253);
        std::uniform_real_distribution<double> spread(0.5, 1.5);
        for (std::int32_t row = 0; row < rows; ++row) {
            double square_sum = 0.0;
            for (std::int32_t k = 0; k < inputs; ++k) {
                // 0..253 skips 0x7f: codes 0x00..0x7e, then 0x80..0xfe.
                const int draw = code_distribution(generator);
                auto code      = static_cast<std::uint8_t>(draw < 127 ? draw : draw + 1);
                if (row == 0) { code &= 0x80; }
                codes_[index(row, k)] = code;
                square_sum += decode_e4m3fn(code) * decode_e4m3fn(code);
            }
            const double multiplier = row == 0 ? 0.0 : row_norm / std::sqrt(square_sum);
            scales_[static_cast<std::size_t>(row)] =
                f32_to_bf16(static_cast<float>(multiplier * spread(generator)));
        }
        payload_.copy_from_host(codes_.data(), codes_.size());
        payload_.copy_from_host(scales_.data(), scales_.size() * sizeof(std::uint16_t),
                                scale_offset_);
    }

    // Exact represented value: code times the stored BF16 multiplier.
    [[nodiscard]] double value(std::int32_t row, std::int32_t k) const {
        return decode_e4m3fn(codes_[index(row, k)]) *
               static_cast<double>(bf16_to_f32(scales_[static_cast<std::size_t>(row)]));
    }

    [[nodiscard]] Weight weight() {
        auto* base = static_cast<std::uint8_t*>(payload_.data());
        Weight result{};
        result.qtype            = QType::FP8_E4M3FN_ROW_BF16S;
        result.layout           = QuantLayout::RowScale;
        result.scale_dtype      = DType::BF16;
        result.payload          = base;
        result.payload_bytes    = payload_.bytes();
        result.qdata            = base;
        result.qhigh            = nullptr;
        result.scales           = base + scale_offset_;
        result.high_plane_bytes = 0;
        result.group_size       = static_cast<std::uint32_t>(inputs_);
        result.group            = inputs_;
        result.ndim             = 2;
        result.shape[0]         = rows_;
        result.shape[1]         = inputs_;
        result.padded_shape[0]  = rows_;
        result.padded_shape[1]  = inputs_;
        result.n                = rows_;
        result.k                = inputs_;
        result.scale_ne[0]      = rows_;
        result.scale_nb[0]      = 2;
        result.scale_nb[1]      = static_cast<std::int64_t>(rows_) * 2;
        result.scale_nb[2]      = result.scale_nb[1];
        result.scale_nb[3]      = result.scale_nb[1];
        return result;
    }

    [[nodiscard]] int verify_guards(const std::string& label) const {
        return payload_.verify_guards(label);
    }

private:
    [[nodiscard]] std::size_t index(std::int32_t row, std::int32_t k) const {
        return static_cast<std::size_t>(row) * inputs_ + static_cast<std::size_t>(k);
    }

    std::int32_t rows_;
    std::int32_t inputs_;
    std::vector<std::uint8_t> codes_;
    std::vector<std::uint16_t> scales_;
    std::size_t scale_offset_;
    GuardedDeviceBuffer payload_;
};

struct Weights {
    std::vector<float> norm;  // BF16-represented [10240]
    Fp8Matrix down;           // [320,10240]
    Fp8Matrix up;             // [10240,320]
    std::vector<float> write; // BF16-represented [4,10240]
};

struct Expected {
    std::vector<double> x;           // [T][2560]
    std::vector<double> write_scale; // [T][4]
};

// FP64 evaluation of the complete read from the represented inputs (see the Op contract): BF16
// residual, norm and write, and down/up decoded exactly as code times row multiplier.
Expected read_oracle(const std::vector<float>& residual, const Weights& w, std::int32_t tokens,
                     bool with_write) {
    Expected out{std::vector<double>(static_cast<std::size_t>(tokens) * kHidden),
                 std::vector<double>(static_cast<std::size_t>(tokens) * kStreams)};
    std::vector<double> down(static_cast<std::size_t>(kRank) * kFlat);
    std::vector<double> up(static_cast<std::size_t>(kFlat) * kRank);
    for (std::int32_t row = 0; row < kRank; ++row) {
        for (std::int32_t k = 0; k < kFlat; ++k) {
            down[std::size_t(row) * kFlat + k] = w.down.value(row, k);
        }
    }
    for (std::int32_t row = 0; row < kFlat; ++row) {
        for (std::int32_t k = 0; k < kRank; ++k) {
            up[std::size_t(row) * kRank + k] = w.up.value(row, k);
        }
    }
    std::vector<double> normalized(kFlat), u(kRank);
    for (std::int32_t t = 0; t < tokens; ++t) {
        const float* r = residual.data() + static_cast<std::size_t>(t) * kFlat;
        for (std::int32_t i = 0; i < kStreams; ++i) {
            double sum = 0.0;
            for (std::int32_t d = 0; d < kHidden; ++d) {
                sum += double(r[i * kHidden + d]) * r[i * kHidden + d];
            }
            const double inverse = 1.0 / std::sqrt(sum / kHidden + 1.0e-6);
            for (std::int32_t d = 0; d < kHidden; ++d) {
                const std::int32_t k = i * kHidden + d;
                normalized[k]        = double(r[k]) * inverse * (1.0 + double(w.norm[k]));
            }
        }
        for (std::int32_t row = 0; row < kRank; ++row) {
            double sum = 0.0;
            for (std::int32_t k = 0; k < kFlat; ++k) {
                sum += down[std::size_t(row) * kFlat + k] * normalized[k];
            }
            const double a = sum / 4.0;
            u[row]         = a * sigmoid(a);
        }
        for (std::int32_t d = 0; d < kHidden; ++d) {
            double mixed = 0.0;
            for (std::int32_t i = 0; i < kStreams; ++i) {
                const std::size_t row = std::size_t(i) * kHidden + d;
                double logit          = 0.0;
                for (std::int32_t k = 0; k < kRank; ++k) { logit += up[row * kRank + k] * u[k]; }
                mixed += sigmoid(logit) * normalized[row];
            }
            out.x[std::size_t(t) * kHidden + d] = mixed / 4.0;
        }
        if (with_write) {
            for (std::int32_t i = 0; i < kStreams; ++i) {
                double sum = 0.0;
                for (std::int32_t k = 0; k < kFlat; ++k) {
                    sum += double(w.write[std::size_t(i) * kFlat + k]) * normalized[k];
                }
                out.write_scale[std::size_t(t) * kStreams + i] = 2.0 * sigmoid(sum / 4.0);
            }
        }
    }
    return out;
}

GuardedDeviceBuffer upload(const std::vector<std::uint16_t>& bits) {
    GuardedDeviceBuffer buffer(bits.size() * sizeof(std::uint16_t));
    buffer.copy_from_host(bits.data(), buffer.bytes());
    return buffer;
}

int run_case(std::int32_t tokens, bool with_write, std::uint32_t seed) {
    const std::string label =
        "hyper_connection T=" + std::to_string(tokens) + (with_write ? " write" : " final");
    // Streams at distinct scales, as a deep residual's streams drift apart.
    std::vector<float> residual(static_cast<std::size_t>(tokens) * kFlat);
    fill_uniform(residual, seed, -1.0F, 1.0F);
    for (std::size_t k = 0; k < residual.size(); ++k) {
        const std::int32_t stream = static_cast<std::int32_t>((k / kHidden) % kStreams);
        residual[k] *= 0.25F * float(1 << (2 * stream));
    }
    // Row norms of the former uniform BF16 fixture (+-0.05 over 10240 and +-0.12 over 320 inputs),
    // which keep u and the gate logits in the sigmoid's sensitive range.
    Weights w{std::vector<float>(kFlat),
              Fp8Matrix(kRank, kFlat, seed + 2, 0.05 * std::sqrt(kFlat / 3.0)),
              Fp8Matrix(kFlat, kRank, seed + 3, 0.12 * std::sqrt(kRank / 3.0)),
              std::vector<float>(std::size_t(kStreams) * kFlat)};
    fill_uniform(w.norm, seed + 1, -0.3F, 0.6F);
    fill_uniform(w.write, seed + 4, -0.03F, 0.03F);
    for (auto* v : {&residual, &w.norm, &w.write}) { round_to_bf16(*v); }
    const Expected expected = read_oracle(residual, w, tokens, with_write);

    const auto residual_bits = encode_bf16(residual);
    auto device_residual     = upload(residual_bits);
    auto device_norm         = upload(encode_bf16(w.norm));
    auto device_write        = upload(encode_bf16(w.write));
    GuardedDeviceBuffer device_x(static_cast<std::size_t>(tokens) * kHidden *
                                 sizeof(std::uint16_t));
    GuardedDeviceBuffer device_scale(static_cast<std::size_t>(tokens) * kStreams * sizeof(float));

    Tensor residual_tensor(device_residual.data(), DType::BF16, {kHidden, kStreams, tokens});
    ops::HyperConnectionWeights weights{
        Tensor(device_norm.data(), DType::BF16, {kFlat}), w.down.weight(), w.up.weight(),
        with_write ? Tensor(device_write.data(), DType::BF16, {kFlat, kStreams}) : Tensor{}};
    Tensor x_tensor(device_x.data(), DType::BF16, {kHidden, tokens});
    Tensor scale_tensor(device_scale.data(), DType::FP32, {kStreams, tokens});
    WorkspaceArena workspace(ops::hyper_connection_read_workspace_bytes(tokens));
    ops::hyper_connection_read(residual_tensor, weights, x_tensor, scale_tensor, workspace,
                               nullptr);
    cuda_synchronize();

    int failures = verify_pointwise(
        label + " x", from_device_bf16(device_x.data(), static_cast<std::size_t>(x_tensor.numel())),
        expected.x, bf16_output_criterion());
    failures += verify_exact(
        (label + " residual unchanged").c_str(),
        from_device<std::uint16_t>(device_residual.data(), residual_bits.size()), residual_bits);
    failures += device_x.verify_guards(label + " x");
    failures += device_scale.verify_guards(label + " write_scale");
    failures += w.down.verify_guards(label + " down");
    failures += w.up.verify_guards(label + " up");
    if (!with_write) { return failures; }

    const auto scale_words =
        from_device<float>(device_scale.data(), static_cast<std::size_t>(scale_tensor.numel()));
    const std::vector<double> scales(scale_words.begin(), scale_words.end());
    failures += verify_pointwise(label + " write_scale", scales, expected.write_scale,
                                 write_scale_criterion());

    // In-place inject of a block output through the device write scales just produced.
    std::vector<float> block(static_cast<std::size_t>(tokens) * kHidden);
    fill_uniform(block, seed + 5, -3.0F, 3.0F);
    round_to_bf16(block);
    auto device_block = upload(encode_bf16(block));
    std::vector<double> injected(residual.size());
    for (std::int32_t t = 0; t < tokens; ++t) {
        for (std::int32_t i = 0; i < kStreams; ++i) {
            // The oracle consumes the represented FP32 write scales the Op published.
            const double scale = scales[std::size_t(t) * kStreams + i];
            for (std::int32_t d = 0; d < kHidden; ++d) {
                const std::size_t k = std::size_t(t) * kFlat + std::size_t(i) * kHidden + d;
                injected[k] =
                    double(residual[k]) + scale * double(block[std::size_t(t) * kHidden + d]);
            }
        }
    }
    Tensor block_tensor(device_block.data(), DType::BF16, {kHidden, tokens});
    ops::hyper_connection_inject(residual_tensor, block_tensor, scale_tensor, residual_tensor,
                                 nullptr);
    cuda_synchronize();
    failures += verify_pointwise(label + " inject in place",
                                 from_device_bf16(device_residual.data(), residual.size()),
                                 injected, bf16_output_criterion());
    failures += device_residual.verify_guards(label + " residual");
    if (tokens == 1 || tokens == 20 || tokens == 32) {
        // The injected BF16 residual is a public state boundary. Qualify a captured inject/read
        // edge against the same oracle, independently of inject's separately checked rounding.
        const auto represented = from_device_bf16(device_residual.data(), residual.size());
        const std::vector<float> next_residual(represented.begin(), represented.end());
        const Expected next = read_oracle(next_residual, w, tokens, true);
        device_residual.copy_from_host(residual_bits.data(), device_residual.bytes());
        // Pageable H2D copies may return before DMA completes; the graph uses a nonblocking stream.
        cuda_synchronize();
        DeviceContext context;
        DecodeGraphDefinition definition;
        definition.capture(context.stream, [&] {
            ops::hyper_connection_inject(residual_tensor, block_tensor, scale_tensor,
                                         residual_tensor, context.stream);
            ops::hyper_connection_read(residual_tensor, weights, x_tensor, scale_tensor, workspace,
                                       context.stream);
        });
        DecodeGraphExecutable graph;
        graph.instantiate(definition);
        graph.launch(context.stream);
        context.synchronize();
        failures += verify_pointwise(label + " captured inject/read x",
                                     from_device_bf16(device_x.data(), expected.x.size()), next.x,
                                     bf16_output_criterion());
        const auto next_scales = from_device<float>(device_scale.data(), next.write_scale.size());
        failures += verify_pointwise(label + " captured inject/read write_scale",
                                     std::vector<double>(next_scales.begin(), next_scales.end()),
                                     next.write_scale, write_scale_criterion());
    }
    return failures;
}

int run_broadcast(std::int32_t tokens, std::uint32_t seed) {
    std::vector<float> embedding(static_cast<std::size_t>(tokens) * kHidden);
    fill_uniform(embedding, seed, -4.0F, 4.0F);
    const auto bits       = encode_bf16(embedding);
    auto device_embedding = upload(bits);
    GuardedDeviceBuffer device_residual(static_cast<std::size_t>(tokens) * kFlat *
                                        sizeof(std::uint16_t));
    Tensor embedding_tensor(device_embedding.data(), DType::BF16, {kHidden, tokens});
    Tensor residual_tensor(device_residual.data(), DType::BF16, {kHidden, kStreams, tokens});
    ops::hyper_connection_broadcast(embedding_tensor, residual_tensor, nullptr);
    cuda_synchronize();
    std::vector<std::uint16_t> expected(static_cast<std::size_t>(tokens) * kFlat);
    for (std::int32_t t = 0; t < tokens; ++t) {
        for (std::int32_t i = 0; i < kStreams; ++i) {
            for (std::int32_t d = 0; d < kHidden; ++d) {
                expected[std::size_t(t) * kFlat + std::size_t(i) * kHidden + d] =
                    bits[std::size_t(t) * kHidden + d];
            }
        }
    }
    int failures =
        verify_exact("hyper_connection_broadcast",
                     from_device<std::uint16_t>(device_residual.data(), expected.size()), expected);
    failures += device_residual.verify_guards("hyper_connection_broadcast residual");
    return failures;
}

} // namespace

int main() {
    if (require_cuda() != 0) { return 1; }
    try {
        int failures = run_broadcast(5, 11U);
        // SIMT decode at T=1; the small-T Tensor Core route boundaries at T=2 (SIMT down, Tensor
        // Core up), T=3 (both on Tensor Cores), T=8/9 (one -> two 8-token MMA tiles, partial at
        // T=7, 12 and 13), T=8/9/10 (fused -> separately finished gate input), T=16/17 (second
        // 16-token tile) and T=31/32/33 (back to SIMT with 64 hidden indices per up CTA);
        // the prefill Tensor Core route boundary at T=159/160/161,
        // partial prefill tiles (T=523), and a width of two 2048-token chunks and a small-T
        // remainder (T=4100); with and without the write projection.
        failures += run_case(1, true, 101U);
        failures += run_case(1, false, 103U);
        failures += run_case(2, true, 107U);
        failures += run_case(3, false, 139U);
        failures += run_case(7, true, 109U);
        failures += run_case(8, true, 113U);
        failures += run_case(9, true, 127U);
        failures += run_case(10, true, 129U);
        failures += run_case(12, true, 131U);
        failures += run_case(12, false, 137U);
        failures += run_case(13, true, 211U);
        failures += run_case(13, false, 223U);
        failures += run_case(16, true, 217U);
        failures += run_case(17, false, 219U);
        failures += run_case(17, true, 221U);
        failures += run_case(20, true, 225U);
        failures += run_case(24, true, 231U);
        failures += run_case(31, true, 227U);
        failures += run_case(32, true, 229U);
        failures += run_case(33, false, 233U);
        failures += run_case(159, true, 239U);
        failures += run_case(160, true, 241U);
        failures += run_case(161, false, 251U);
        failures += run_case(523, true, 307U);
        failures += run_case(4100, true, 401U);
        std::cout << (failures == 0 ? "OK" : "FAIL") << " hyper_connection\n";
        return failures == 0 ? 0 : 1;
    } catch (const std::exception& error) {
        std::cerr << "hyper_connection: " << error.what() << '\n';
        return 1;
    }
}
