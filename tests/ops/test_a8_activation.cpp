// A8 activation producers and consumers (include/ninfer/ops/a8_activation.h).
//
// Producers: the published codes/scales must equal the independent exact A8 codec
// (fp8_activation_ref.h) of the BF16 values the corresponding BF16 Op writes, and any BF16 output
// they also write must be bit-identical to that Op's. Consumers: an A8 overload fed the exact codec
// of a BF16 activation must reproduce the qualified AllowA8 route on that activation bit for bit;
// that route is checked against its FP64 oracle by the Linear-family tests.

#include "ninfer/ops/a8_activation.h"
#include "ninfer/ops/attn_input_proj.h"
#include "ninfer/ops/gated_rmsnorm.h"
#include "ninfer/ops/gdn_gating_proj.h"
#include "ninfer/ops/linear_add.h"
#include "ninfer/ops/rmsnorm.h"
#include "ninfer/ops/sigmoid_mul.h"
#include "ops/fp8_activation_ref.h"
#include "ops/input_projection_test_common.h"
#include "ops/sanitizer_scope.h"

#include <cstdint>
#include <exception>
#include <iostream>
#include <span>
#include <string>
#include <vector>

namespace {

using namespace ninfer;
using namespace ninfer::test;
using input_projection::DevicePackedWeight;

constexpr float kEps                 = 1e-6F;
constexpr std::uint8_t kPoisonByte   = 0xA5;
constexpr std::int32_t kHidden       = 5120;
constexpr std::int32_t kGdnHeadDim   = 128;
constexpr std::int32_t kGdnHeads     = 48;
constexpr std::int32_t kGdnValueRows = kGdnHeadDim * kGdnHeads;
constexpr std::int32_t kQueryRows    = 6144;
constexpr std::int32_t kKvRows       = 1024;

// Device-owned [rows, columns] A8 activation, poisoned so unwritten codes/scales are visible.
class DeviceA8 {
public:
    DeviceA8(std::int32_t rows, std::int32_t columns)
        : rows_(rows), columns_(columns),
          codes_(static_cast<std::size_t>(rows) * static_cast<std::size_t>(columns)),
          scales_(static_cast<std::size_t>(columns) * sizeof(float)) {
        codes_.fill(kPoisonByte);
        scales_.fill(kPoisonByte);
    }

    ops::A8Activation view() {
        return {Tensor(codes_.p, DType::FP8_E4M3FN, {rows_, columns_}),
                Tensor(scales_.p, DType::FP32, {columns_})};
    }

    void upload(const Fp8ActivationReference& encoded) {
        codes_.copy_from_host(encoded.codes.data(), encoded.codes.size());
        scales_.copy_from_host(encoded.scales.data(), encoded.scales.size() * sizeof(float));
    }

    std::vector<std::uint8_t> codes() const {
        return from_device<std::uint8_t>(codes_.p, codes_.bytes);
    }

    std::vector<float> scales() const {
        return from_device<float>(scales_.p, static_cast<std::size_t>(columns_));
    }

private:
    std::int32_t rows_;
    std::int32_t columns_;
    DeviceBuffer codes_;
    DeviceBuffer scales_;
};

std::vector<float> bf16_values(std::span<const std::uint16_t> bits) {
    std::vector<float> values(bits.size());
    for (std::size_t i = 0; i < bits.size(); ++i) { values[i] = bf16_to_f32(bits[i]); }
    return values;
}

int verify_codec(const std::string& label, std::span<const std::uint16_t> bf16_bits,
                 std::int32_t rows, const DeviceA8& actual) {
    const std::vector<float> values      = bf16_values(bf16_bits);
    const Fp8ActivationReference encoded = fp8_activation_reference(values, rows);
    int failures = verify_exact((label + " codes").c_str(), actual.codes(), encoded.codes);
    failures += verify_exact((label + " scales").c_str(), actual.scales(), encoded.scales);
    return failures;
}

// BF16 activation with a deliberately all-zero column 0 (scale 0, zero codes).
std::vector<float> make_activation(std::int32_t rows, std::int32_t columns, std::uint32_t seed,
                                   float magnitude) {
    std::vector<float> values(static_cast<std::size_t>(rows) * columns);
    fill_uniform(values, seed, -magnitude, magnitude);
    round_to_bf16(values);
    std::fill_n(values.begin(), rows, 0.0F);
    return values;
}

std::vector<std::uint16_t> bf16_bits(const DeviceBuffer& buffer, std::size_t count) {
    return from_device<std::uint16_t>(buffer.p, count);
}

int run_rmsnorm_a8(std::int32_t tokens) {
    const auto x = make_activation(kHidden, tokens, 101U + tokens, 4.0F);
    std::vector<float> weight(kHidden);
    fill_uniform(weight, 7U, -0.5F, 0.5F);
    round_to_bf16(weight);
    const std::size_t count = static_cast<std::size_t>(kHidden) * tokens;
    DeviceBuffer dx = to_device_bf16(x), dw = to_device_bf16(weight);
    DeviceBuffer reference(count * 2), normalized(count * 2);
    normalized.fill(kPoisonByte);
    DeviceA8 only(kHidden, tokens), both(kHidden, tokens);
    const Tensor tx(dx.p, DType::BF16, {kHidden, tokens});
    const Tensor tw(dw.p, DType::BF16, {kHidden});
    Tensor treference(reference.p, DType::BF16, {kHidden, tokens});
    Tensor tnormalized(normalized.p, DType::BF16, {kHidden, tokens});
    ops::rmsnorm(tx, tw, kEps, true, treference, nullptr);
    ops::A8Activation only_view = only.view(), both_view = both.view();
    ops::rmsnorm_a8(tx, tw, kEps, nullptr, only_view, nullptr);
    ops::rmsnorm_a8(tx, tw, kEps, &tnormalized, both_view, nullptr);
    cuda_synchronize();

    const std::string label = "rmsnorm_a8 T=" + std::to_string(tokens);
    const auto expected     = bf16_bits(reference, count);
    int failures =
        verify_exact((label + " normalized").c_str(), bf16_bits(normalized, count), expected);
    failures += verify_codec(label + " A8-only", expected, kHidden, only);
    failures += verify_codec(label + " with BF16", expected, kHidden, both);
    return failures;
}

int run_gated_rmsnorm_a8(std::int32_t tokens) {
    const auto x = make_activation(kGdnValueRows, tokens, 211U + tokens, 2.0F);
    std::vector<float> z(static_cast<std::size_t>(kGdnValueRows) * tokens);
    fill_uniform(z, 307U + tokens, -3.0F, 3.0F);
    round_to_bf16(z);
    std::vector<float> weight(kGdnHeadDim);
    fill_uniform(weight, 11U, 0.5F, 1.5F);
    round_to_bf16(weight);
    const std::size_t count = static_cast<std::size_t>(kGdnValueRows) * tokens;
    DeviceBuffer dx = to_device_bf16(x), dz = to_device_bf16(z), dw = to_device_bf16(weight);
    DeviceBuffer reference(count * 2);
    DeviceA8 actual(kGdnValueRows, tokens);
    const Tensor tx(dx.p, DType::BF16, {kGdnHeadDim, kGdnHeads, tokens});
    const Tensor tz(dz.p, DType::BF16, {kGdnHeadDim, kGdnHeads, tokens});
    const Tensor tw(dw.p, DType::BF16, {kGdnHeadDim});
    Tensor treference(reference.p, DType::BF16, {kGdnHeadDim, kGdnHeads, tokens});
    ops::gated_rmsnorm(tx, tw, tz, kEps, treference, nullptr);
    ops::A8Activation view = actual.view();
    ops::gated_rmsnorm_a8(tx, tw, tz, kEps, view, nullptr);
    cuda_synchronize();
    return verify_codec("gated_rmsnorm_a8 T=" + std::to_string(tokens), bf16_bits(reference, count),
                        kGdnValueRows, actual);
}

int run_sigmoid_mul_a8(std::int32_t tokens) {
    const auto x = make_activation(kQueryRows, tokens, 401U + tokens, 3.0F);
    std::vector<float> gate(static_cast<std::size_t>(kQueryRows) * tokens);
    fill_uniform(gate, 503U + tokens, -6.0F, 6.0F);
    round_to_bf16(gate);
    const std::size_t count = static_cast<std::size_t>(kQueryRows) * tokens;
    DeviceBuffer dx = to_device_bf16(x), dgate = to_device_bf16(gate);
    DeviceBuffer reference = to_device_bf16(x);
    DeviceA8 actual(kQueryRows, tokens);
    const Tensor tgate(dgate.p, DType::BF16, {kQueryRows, tokens});
    const Tensor tx(dx.p, DType::BF16, {kQueryRows, tokens});
    Tensor treference(reference.p, DType::BF16, {kQueryRows, tokens});
    ops::sigmoid_mul(tgate, treference, nullptr);
    ops::A8Activation view = actual.view();
    ops::sigmoid_mul_a8(tgate, tx, view, nullptr);
    cuda_synchronize();

    const std::string label = "sigmoid_mul_a8 T=" + std::to_string(tokens);
    std::vector<std::uint16_t> input_bits(count);
    for (std::size_t i = 0; i < count; ++i) { input_bits[i] = f32_to_bf16(x[i]); }
    int failures =
        verify_exact((label + " input preserved").c_str(), bf16_bits(dx, count), input_bits);
    failures += verify_codec(label, bf16_bits(reference, count), kQueryRows, actual);
    return failures;
}

Weight bf16_control_weight(void* data, std::int32_t rows, std::int32_t hidden) {
    Weight weight{};
    weight.qtype           = QType::BF16_CTRL;
    weight.layout          = QuantLayout::Contiguous;
    weight.payload         = data;
    weight.payload_bytes   = static_cast<std::uint64_t>(rows) * hidden * sizeof(std::uint16_t);
    weight.qdata           = data;
    weight.ndim            = 2;
    weight.shape[0]        = rows;
    weight.shape[1]        = hidden;
    weight.padded_shape[0] = rows;
    weight.padded_shape[1] = hidden;
    weight.n               = rows;
    weight.k               = hidden;
    return weight;
}

// The A8-publishing GDN control forms against the plain forms (packed when width > 0).
int run_gdn_norm_gating_a8(std::int32_t tokens, std::int32_t width) {
    constexpr std::int32_t kHeads = 48;
    const auto x                  = make_activation(kHidden, tokens, 601U + tokens, 4.0F);
    std::vector<float> norm(kHidden), a(static_cast<std::size_t>(kHeads) * kHidden),
        b(static_cast<std::size_t>(kHeads) * kHidden), a_log(kHeads), dt_bias(kHeads);
    fill_uniform(norm, 13U, -0.5F, 0.5F);
    fill_uniform(a, 17U, -0.02F, 0.02F);
    fill_uniform(b, 19U, -0.02F, 0.02F);
    fill_uniform(a_log, 23U, -1.0F, 1.0F);
    fill_uniform(dt_bias, 29U, -1.0F, 1.0F);
    round_to_bf16(norm);
    round_to_bf16(a);
    round_to_bf16(b);
    const std::size_t h_count    = static_cast<std::size_t>(kHidden) * tokens;
    const std::size_t ctrl_count = static_cast<std::size_t>(kHeads) * tokens;
    DeviceBuffer dx = to_device_bf16(x), dnorm = to_device_bf16(norm), da = to_device_bf16(a),
                 db = to_device_bf16(b), dalog = to_device_f32(a_log), ddt = to_device_f32(dt_bias);
    DeviceBuffer plain_h(h_count * 2), plain_g(ctrl_count * 4), plain_beta(ctrl_count * 4);
    DeviceBuffer a8_h(h_count * 2), a8_g(ctrl_count * 4), a8_beta(ctrl_count * 4);
    DeviceA8 activation(kHidden, tokens);
    const Tensor tx(dx.p, DType::BF16, {kHidden, tokens});
    const Tensor tnorm(dnorm.p, DType::BF16, {kHidden});
    const Tensor talog(dalog.p, DType::FP32, {kHeads});
    const Tensor tdt(ddt.p, DType::FP32, {kHeads});
    const Weight wa = bf16_control_weight(da.p, kHeads, kHidden);
    const Weight wb = bf16_control_weight(db.p, kHeads, kHidden);
    Tensor ph(plain_h.p, DType::BF16, {kHidden, tokens});
    Tensor pg(plain_g.p, DType::FP32, {kHeads, tokens});
    Tensor pb(plain_beta.p, DType::FP32, {kHeads, tokens});
    Tensor ah(a8_h.p, DType::BF16, {kHidden, tokens});
    Tensor ag(a8_g.p, DType::FP32, {kHeads, tokens});
    Tensor ab(a8_beta.p, DType::FP32, {kHeads, tokens});
    ops::A8Activation view = activation.view();
    const std::size_t ws_bytes =
        width > 0
            ? ops::gdn_norm_gating_proj_packed_sequences_workspace_capacity_bytes(
                  width, tokens / width, tokens / width)
            : ops::gdn_norm_gating_proj_workspace_capacity_bytes(kHeads, kHidden, tokens, tokens);
    WorkspaceArena workspace(std::max<std::size_t>(256, ws_bytes));
    if (width > 0) {
        ops::gdn_norm_gating_proj_packed_sequences(tx, tnorm, kEps, wa, wb, talog, tdt, workspace,
                                                   ph, pg, pb, nullptr, width);
        ops::gdn_norm_gating_proj_packed_sequences(tx, tnorm, kEps, wa, wb, talog, tdt, workspace,
                                                   ah, view, ag, ab, nullptr, width);
    } else {
        ops::gdn_norm_gating_proj(tx, tnorm, kEps, wa, wb, talog, tdt, workspace, ph, pg, pb,
                                  nullptr);
        ops::gdn_norm_gating_proj(tx, tnorm, kEps, wa, wb, talog, tdt, workspace, ah, view, ag, ab,
                                  nullptr);
    }
    cuda_synchronize();

    const std::string label = "gdn_norm_gating_proj A8 T=" + std::to_string(tokens) +
                              (width > 0 ? " W=" + std::to_string(width) : std::string());
    const auto h_bits       = bf16_bits(plain_h, h_count);
    int failures = verify_exact((label + " h").c_str(), bf16_bits(a8_h, h_count), h_bits);
    failures += verify_exact((label + " g").c_str(), from_device<float>(a8_g.p, ctrl_count),
                             from_device<float>(plain_g.p, ctrl_count));
    failures += verify_exact((label + " beta").c_str(), from_device<float>(a8_beta.p, ctrl_count),
                             from_device<float>(plain_beta.p, ctrl_count));
    failures += verify_codec(label, h_bits, kHidden, activation);
    return failures;
}

quantized_weight::PatternedWeightOptions nvfp4_options() {
    quantized_weight::PatternedWeightOptions options;
    options.weight_scale_divisor = 0.125F;
    options.input_scale_divisor  = 3.5F;
    return options;
}

int run_linear_add_a8(const DevicePackedWeight& weight, std::int32_t tokens) {
    const std::int32_t k = weight.host.weight.k;
    const auto x         = input_projection::make_bf16_activation(k, tokens, 701U + tokens);
    std::vector<float> residual(static_cast<std::size_t>(kHidden) * tokens);
    fill_uniform(residual, 709U + tokens, -1.0F, 1.0F);
    round_to_bf16(residual);
    const std::size_t count = static_cast<std::size_t>(kHidden) * tokens;
    DeviceBuffer dx = to_device_bf16(x), policy_residual = to_device_bf16(residual),
                 a8_residual = to_device_bf16(residual);
    DeviceA8 activation(k, tokens);
    activation.upload(fp8_activation_reference(x, k));
    const Tensor tx(dx.p, DType::BF16, {k, tokens});
    Tensor tpolicy(policy_residual.p, DType::BF16, {kHidden, tokens});
    Tensor ta8(a8_residual.p, DType::BF16, {kHidden, tokens});
    WorkspaceArena workspace(std::max<std::size_t>(
        256, ops::linear_add_workspace_capacity_bytes(QType::NVFP4, kHidden, k,
                                                      ops::LinearPolicy::AllowA8, tokens, tokens)));
    ops::linear_add(tx, weight.view(), tpolicy, ops::LinearPolicy::AllowA8, workspace, nullptr);
    ops::linear_add(activation.view(), weight.view(), ta8, nullptr);
    cuda_synchronize();
    const std::string label =
        "linear_add A8 K=" + std::to_string(k) + " T=" + std::to_string(tokens);
    return verify_exact(label.c_str(), bf16_bits(a8_residual, count),
                        bf16_bits(policy_residual, count));
}

int run_attn_input_proj_a8(const DevicePackedWeight& weight, std::int32_t tokens) {
    const auto x    = input_projection::make_bf16_activation(kHidden, tokens, 801U + tokens);
    DeviceBuffer dx = to_device_bf16(x);
    DeviceA8 activation(kHidden, tokens);
    activation.upload(fp8_activation_reference(x, kHidden));
    const std::size_t q_count  = static_cast<std::size_t>(kQueryRows) * tokens;
    const std::size_t kv_count = static_cast<std::size_t>(kKvRows) * tokens;
    DeviceBuffer pq(q_count * 2), pgate(q_count * 2), pk(kv_count * 2), pv(kv_count * 2);
    DeviceBuffer aq(q_count * 2), agate(q_count * 2), ak(kv_count * 2), av(kv_count * 2);
    const Tensor tx(dx.p, DType::BF16, {kHidden, tokens});
    Tensor policy_q(pq.p, DType::BF16, {kQueryRows, tokens}),
        policy_gate(pgate.p, DType::BF16, {kQueryRows, tokens}),
        policy_k(pk.p, DType::BF16, {kKvRows, tokens}),
        policy_v(pv.p, DType::BF16, {kKvRows, tokens});
    Tensor a8_q(aq.p, DType::BF16, {kQueryRows, tokens}),
        a8_gate(agate.p, DType::BF16, {kQueryRows, tokens}),
        a8_k(ak.p, DType::BF16, {kKvRows, tokens}), a8_v(av.p, DType::BF16, {kKvRows, tokens});
    WorkspaceArena workspace(std::max<std::size_t>(
        256, ops::attn_input_proj_workspace_capacity_bytes(
                 QType::NVFP4, 14336, kHidden, ops::LinearPolicy::AllowA8, tokens, tokens)));
    ops::attn_input_proj(tx, weight.view(), policy_q, policy_gate, policy_k, policy_v,
                         ops::LinearPolicy::AllowA8, workspace, nullptr);
    ops::attn_input_proj(activation.view(), weight.view(), a8_q, a8_gate, a8_k, a8_v, nullptr);
    cuda_synchronize();
    const std::string label = "attn_input_proj A8 T=" + std::to_string(tokens);
    int failures =
        verify_exact((label + " q").c_str(), bf16_bits(aq, q_count), bf16_bits(pq, q_count));
    failures += verify_exact((label + " gate").c_str(), bf16_bits(agate, q_count),
                             bf16_bits(pgate, q_count));
    failures +=
        verify_exact((label + " k").c_str(), bf16_bits(ak, kv_count), bf16_bits(pk, kv_count));
    failures +=
        verify_exact((label + " v").c_str(), bf16_bits(av, kv_count), bf16_bits(pv, kv_count));
    return failures;
}

} // namespace

int main(int argc, char** argv) {
    try {
        if (const int status = require_cuda(); status != 0) { return status; }
        const bool sanitizer = sanitizer_scope(argc, argv);
        // Verify widths: C=1 k=4 (5), C=4 k=4 (20), and the W*B<=48 aggregate limit.
        const std::vector<std::int32_t> tokens = sanitizer
                                                     ? std::vector<std::int32_t>{5, 48}
                                                     : std::vector<std::int32_t>{1, 2, 5, 20, 48};
        int failures                           = 0;
        for (const std::int32_t t : tokens) {
            failures += run_rmsnorm_a8(t);
            failures += run_gated_rmsnorm_a8(t);
            failures += run_sigmoid_mul_a8(t);
        }
        failures += run_gdn_norm_gating_a8(5, 0);
        failures += run_gdn_norm_gating_a8(20, 5);
        if (!sanitizer) {
            failures += run_gdn_norm_gating_a8(48, 0);
            failures += run_gdn_norm_gating_a8(48, 8);
        }

        // The consumers launch the projections' existing W4A8 kernels, which the Linear-family
        // tests sanitize; the --sanitizer cases cover the producer kernels only.
        if (sanitizer) {
            std::cout << (failures == 0 ? "OK" : "FAIL") << " A8 activation producers\n";
            return failures == 0 ? 0 : 1;
        }
        const DevicePackedWeight residual6144(quantized_weight::make_patterned_weight(
            QType::NVFP4, kHidden, kGdnValueRows, 901U, nvfp4_options()));
        const DevicePackedWeight residual17408(quantized_weight::make_patterned_weight(
            QType::NVFP4, kHidden, 17408, 903U, nvfp4_options()));
        const DevicePackedWeight attention(quantized_weight::make_patterned_weight(
            QType::NVFP4, 14336, kHidden, 905U, nvfp4_options()));
        for (const std::int32_t t : tokens) {
            if (t < 2) { continue; }
            failures += run_linear_add_a8(residual6144, t);
            failures += run_linear_add_a8(residual17408, t);
            failures += run_attn_input_proj_a8(attention, t);
        }
        std::cout << (failures == 0 ? "OK" : "FAIL") << " A8 activation producers/consumers\n";
        return failures == 0 ? 0 : 1;
    } catch (const std::exception& error) {
        std::cerr << "FAIL: " << error.what() << '\n';
        return 1;
    }
}
