// A4 activation producers and consumers (include/ninfer/ops/a4_activation.h).
//
// Producers: the published codes and tiled scales must equal the independent exact NVFP4 codec
// (nvfp4_activation_ref.h) of the BF16 values the corresponding BF16 Op writes, and the padding
// tokens of the last 16-token fragment must carry zero scales. Consumers: an A4 overload fed the
// exact codec of a BF16 activation must reproduce the qualified AllowA4 route on that activation
// bit for bit; that route is checked against its FP64 oracle by the Linear-family tests.

#include "ninfer/ops/a4_activation.h"
#include "ninfer/ops/attn_input_proj.h"
#include "ninfer/ops/gdn_gating_proj.h"
#include "ninfer/ops/gdn_input_proj.h"
#include "ninfer/ops/gated_rmsnorm.h"
#include "ninfer/ops/linear_add.h"
#include "ninfer/ops/linear_swiglu.h"
#include "ninfer/ops/rmsnorm.h"
#include "ninfer/ops/sigmoid_mul.h"
#include "ops/input_projection_test_common.h"
#include "ops/nvfp4_activation_ref.h"
#include "ops/sanitizer_scope.h"

#include <algorithm>
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
constexpr float kInputDivisor        = 3.5F;
constexpr std::uint8_t kPoisonByte   = 0xA5;
constexpr std::int32_t kHidden       = 5120;
constexpr std::int32_t kQueryRows    = 6144;
constexpr std::int32_t kKvRows       = 1024;
constexpr std::int32_t kIntermediate = 17408;
constexpr std::int32_t kGdnHeadDim   = 128;
constexpr std::int32_t kGdnHeads     = 48;
constexpr std::int32_t kGdnValueRows = kGdnHeadDim * kGdnHeads;
constexpr std::int32_t kGdnQkvRows   = 2048 + 2048 + kGdnValueRows;
constexpr std::int32_t kGdnInputRows = kGdnQkvRows + kGdnValueRows;

std::int32_t fragment_tokens(std::int32_t tokens) { return (tokens + 15) / 16 * 16; }

std::size_t tiled_scale_offset(std::int32_t rows, std::int32_t token, std::int32_t group) {
    const std::int32_t tiles_per_row = rows / 16 / ops::kA4ScaleTileGroups;
    const std::size_t tile =
        static_cast<std::size_t>(token / ops::kA4ScaleTileTokens) * tiles_per_row +
        group / ops::kA4ScaleTileGroups;
    return tile * ops::kA4ScaleTileTokens * ops::kA4ScaleTileGroups +
           static_cast<std::size_t>(token % ops::kA4ScaleTileTokens) * ops::kA4ScaleTileGroups +
           group % ops::kA4ScaleTileGroups;
}

// Device-owned [rows, columns] A4 activation, poisoned so unwritten codes/scales are visible.
class DeviceA4 {
public:
    DeviceA4(std::int32_t rows, std::int32_t columns)
        : rows_(rows), columns_(columns),
          codes_(static_cast<std::size_t>(rows / 2) * static_cast<std::size_t>(columns)),
          scales_(static_cast<std::size_t>(ops::detail::a4_scale_plane_bytes(rows, columns))) {
        codes_.fill(kPoisonByte);
        scales_.fill(kPoisonByte);
    }

    ops::A4Activation view() {
        return {Tensor(codes_.p, DType::U8, {rows_ / 2, columns_}),
                Tensor(scales_.p, DType::U8, {static_cast<std::int32_t>(scales_.bytes)}),
                kInputDivisor};
    }

    // Writes the reference codec, scattering its row-major [token, group] scales into the plane
    // and zeroing the padding tokens of the last fragment.
    void upload(const Nvfp4ActivationReference& encoded) {
        codes_.copy_from_host(encoded.codes.data(), encoded.codes.size());
        std::vector<std::uint8_t> plane(scales_.bytes, kPoisonByte);
        const std::int32_t groups = rows_ / 16;
        for (std::int32_t token = 0; token < fragment_tokens(columns_); ++token) {
            for (std::int32_t group = 0; group < groups; ++group) {
                plane[tiled_scale_offset(rows_, token, group)] =
                    token < columns_
                        ? encoded.scales[static_cast<std::size_t>(token) * groups + group]
                        : 0;
            }
        }
        scales_.copy_from_host(plane.data(), plane.size());
    }

    std::vector<std::uint8_t> codes() const {
        return from_device<std::uint8_t>(codes_.p, codes_.bytes);
    }

    // The scales of the first round_up(columns, 16) tokens in row-major [token, group] order.
    std::vector<std::uint8_t> fragment_scales() const {
        const auto plane          = from_device<std::uint8_t>(scales_.p, scales_.bytes);
        const std::int32_t groups = rows_ / 16;
        std::vector<std::uint8_t> out(static_cast<std::size_t>(fragment_tokens(columns_)) * groups);
        for (std::int32_t token = 0; token < fragment_tokens(columns_); ++token) {
            for (std::int32_t group = 0; group < groups; ++group) {
                out[static_cast<std::size_t>(token) * groups + group] =
                    plane[tiled_scale_offset(rows_, token, group)];
            }
        }
        return out;
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

std::vector<std::uint16_t> bf16_bits(const DeviceBuffer& buffer, std::size_t count) {
    return from_device<std::uint16_t>(buffer.p, count);
}

int verify_codec(const std::string& label, std::span<const std::uint16_t> bf16_bits,
                 std::int32_t rows, std::int32_t tokens, const DeviceA4& actual) {
    const Nvfp4ActivationReference encoded =
        nvfp4_activation_reference(bf16_values(bf16_bits), kInputDivisor);
    std::vector<std::uint8_t> expected_scales = encoded.scales;
    expected_scales.resize(static_cast<std::size_t>(fragment_tokens(tokens)) * (rows / 16), 0);
    int failures = verify_exact((label + " codes").c_str(), actual.codes(), encoded.codes);
    failures +=
        verify_exact((label + " scales").c_str(), actual.fragment_scales(), expected_scales);
    return failures;
}

// BF16 activation with a deliberately all-zero column 0 (zero scales and codes).
std::vector<float> make_activation(std::int32_t rows, std::int32_t columns, std::uint32_t seed,
                                   float magnitude) {
    std::vector<float> values(static_cast<std::size_t>(rows) * columns);
    fill_uniform(values, seed, -magnitude, magnitude);
    round_to_bf16(values);
    std::fill_n(values.begin(), rows, 0.0F);
    return values;
}

int run_rmsnorm_a4(std::int32_t tokens) {
    const auto x = make_activation(kHidden, tokens, 101U + tokens, 4.0F);
    std::vector<float> weight(kHidden);
    fill_uniform(weight, 7U, -0.5F, 0.5F);
    round_to_bf16(weight);
    const std::size_t count = static_cast<std::size_t>(kHidden) * tokens;
    DeviceBuffer dx = to_device_bf16(x), dw = to_device_bf16(weight);
    DeviceBuffer reference(count * 2), normalized(count * 2);
    normalized.fill(kPoisonByte);
    DeviceA4 only(kHidden, tokens), both(kHidden, tokens);
    const Tensor tx(dx.p, DType::BF16, {kHidden, tokens});
    const Tensor tw(dw.p, DType::BF16, {kHidden});
    Tensor treference(reference.p, DType::BF16, {kHidden, tokens});
    Tensor tnormalized(normalized.p, DType::BF16, {kHidden, tokens});
    ops::rmsnorm(tx, tw, kEps, true, treference, nullptr);
    ops::A4Activation only_view = only.view(), both_view = both.view();
    ops::rmsnorm_a4(tx, tw, kEps, nullptr, only_view, nullptr);
    ops::rmsnorm_a4(tx, tw, kEps, &tnormalized, both_view, nullptr);
    cuda_synchronize();

    const std::string label = "rmsnorm_a4 T=" + std::to_string(tokens);
    const auto expected     = bf16_bits(reference, count);
    int failures =
        verify_exact((label + " normalized").c_str(), bf16_bits(normalized, count), expected);
    failures += verify_codec(label + " A4-only", expected, kHidden, tokens, only);
    failures += verify_codec(label + " with BF16", expected, kHidden, tokens, both);
    return failures;
}

int run_gated_rmsnorm_a4(std::int32_t tokens) {
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
    DeviceA4 actual(kGdnValueRows, tokens);
    const Tensor tx(dx.p, DType::BF16, {kGdnHeadDim, kGdnHeads, tokens});
    const Tensor tz(dz.p, DType::BF16, {kGdnHeadDim, kGdnHeads, tokens});
    const Tensor tw(dw.p, DType::BF16, {kGdnHeadDim});
    Tensor treference(reference.p, DType::BF16, {kGdnHeadDim, kGdnHeads, tokens});
    ops::gated_rmsnorm(tx, tw, tz, kEps, treference, nullptr);
    ops::A4Activation view = actual.view();
    ops::gated_rmsnorm_a4(tx, tw, tz, kEps, view, nullptr);
    cuda_synchronize();
    return verify_codec("gated_rmsnorm_a4 T=" + std::to_string(tokens), bf16_bits(reference, count),
                        kGdnValueRows, tokens, actual);
}

Weight bf16_control_weight(void* data, std::int32_t rows) {
    Weight weight{};
    weight.qtype           = QType::BF16_CTRL;
    weight.layout          = QuantLayout::Contiguous;
    weight.payload         = data;
    weight.payload_bytes   = static_cast<std::uint64_t>(rows) * kHidden * sizeof(std::uint16_t);
    weight.qdata           = data;
    weight.ndim            = 2;
    weight.shape[0]        = rows;
    weight.shape[1]        = kHidden;
    weight.padded_shape[0] = rows;
    weight.padded_shape[1] = kHidden;
    weight.n               = rows;
    weight.k               = kHidden;
    return weight;
}

// The A4 form must write h, g, and beta bit-identical to the plain two-weight form and publish
// exactly the A4 codec of that h.
int run_gdn_norm_gating_a4(std::int32_t tokens) {
    const auto x = make_activation(kHidden, tokens, 601U + tokens, 4.0F);
    std::vector<float> norm_weight(kHidden),
        a_weight(static_cast<std::size_t>(kGdnHeads) * kHidden), b_weight(a_weight.size()),
        a_log(kGdnHeads), dt_bias(kGdnHeads);
    fill_uniform(norm_weight, 13U, -0.5F, 0.5F);
    fill_uniform(a_weight, 17U, -0.05F, 0.05F);
    fill_uniform(b_weight, 19U, -0.05F, 0.05F);
    fill_uniform(a_log, 23U, -1.0F, 1.0F);
    fill_uniform(dt_bias, 29U, -1.0F, 1.0F);
    round_to_bf16(norm_weight);
    round_to_bf16(a_weight);
    round_to_bf16(b_weight);
    const std::size_t count         = static_cast<std::size_t>(kHidden) * tokens;
    const std::size_t control_count = static_cast<std::size_t>(kGdnHeads) * tokens;
    DeviceBuffer dx = to_device_bf16(x), dnorm = to_device_bf16(norm_weight),
                 da = to_device_bf16(a_weight), db = to_device_bf16(b_weight);
    DeviceBuffer dlog(a_log.size() * sizeof(float)), dbias(dt_bias.size() * sizeof(float));
    dlog.copy_from_host(a_log.data(), dlog.bytes);
    dbias.copy_from_host(dt_bias.data(), dbias.bytes);
    DeviceBuffer ref_h(count * 2), ref_g(control_count * 4), ref_beta(control_count * 4);
    DeviceBuffer a4_h(count * 2), a4_g(control_count * 4), a4_beta(control_count * 4);
    DeviceA4 actual(kHidden, tokens);
    const Tensor tx(dx.p, DType::BF16, {kHidden, tokens});
    const Tensor tnorm(dnorm.p, DType::BF16, {kHidden});
    const Tensor tlog(dlog.p, DType::FP32, {kGdnHeads});
    const Tensor tbias(dbias.p, DType::FP32, {kGdnHeads});
    const Weight wa = bf16_control_weight(da.p, kGdnHeads);
    const Weight wb = bf16_control_weight(db.p, kGdnHeads);
    Tensor trh(ref_h.p, DType::BF16, {kHidden, tokens}),
        tah(a4_h.p, DType::BF16, {kHidden, tokens});
    Tensor trg(ref_g.p, DType::FP32, {kGdnHeads, tokens}),
        tag(a4_g.p, DType::FP32, {kGdnHeads, tokens});
    Tensor trbeta(ref_beta.p, DType::FP32, {kGdnHeads, tokens}),
        tabeta(a4_beta.p, DType::FP32, {kGdnHeads, tokens});
    WorkspaceArena workspace(std::max<std::size_t>(
        256,
        ops::gdn_norm_gating_proj_workspace_capacity_bytes(kGdnHeads, kHidden, tokens, tokens)));
    ops::gdn_norm_gating_proj(tx, tnorm, kEps, wa, wb, tlog, tbias, workspace, trh, trg, trbeta,
                              nullptr);
    ops::A4Activation view = actual.view();
    ops::gdn_norm_gating_proj(tx, tnorm, kEps, wa, wb, tlog, tbias, workspace, tah, view, tag,
                              tabeta, nullptr);
    cuda_synchronize();

    const std::string label = "gdn_norm_gating_proj A4 T=" + std::to_string(tokens);
    const auto expected     = bf16_bits(ref_h, count);
    int failures = verify_exact((label + " h").c_str(), bf16_bits(a4_h, count), expected);
    failures +=
        verify_exact((label + " g").c_str(), from_device<std::uint32_t>(a4_g.p, control_count),
                     from_device<std::uint32_t>(ref_g.p, control_count));
    failures += verify_exact((label + " beta").c_str(),
                             from_device<std::uint32_t>(a4_beta.p, control_count),
                             from_device<std::uint32_t>(ref_beta.p, control_count));
    failures += verify_codec(label, expected, kHidden, tokens, actual);
    return failures;
}

int run_sigmoid_mul_a4(std::int32_t tokens) {
    const auto x = make_activation(kQueryRows, tokens, 401U + tokens, 3.0F);
    std::vector<float> gate(static_cast<std::size_t>(kQueryRows) * tokens);
    fill_uniform(gate, 503U + tokens, -6.0F, 6.0F);
    round_to_bf16(gate);
    const std::size_t count = static_cast<std::size_t>(kQueryRows) * tokens;
    DeviceBuffer dx = to_device_bf16(x), dgate = to_device_bf16(gate);
    DeviceBuffer reference = to_device_bf16(x);
    DeviceA4 actual(kQueryRows, tokens);
    const Tensor tgate(dgate.p, DType::BF16, {kQueryRows, tokens});
    const Tensor tx(dx.p, DType::BF16, {kQueryRows, tokens});
    Tensor treference(reference.p, DType::BF16, {kQueryRows, tokens});
    ops::sigmoid_mul(tgate, treference, nullptr);
    ops::A4Activation view = actual.view();
    ops::sigmoid_mul_a4(tgate, tx, view, nullptr);
    cuda_synchronize();

    const std::string label = "sigmoid_mul_a4 T=" + std::to_string(tokens);
    std::vector<std::uint16_t> input_bits(count);
    for (std::size_t i = 0; i < count; ++i) { input_bits[i] = f32_to_bf16(x[i]); }
    int failures =
        verify_exact((label + " input preserved").c_str(), bf16_bits(dx, count), input_bits);
    failures += verify_codec(label, bf16_bits(reference, count), kQueryRows, tokens, actual);
    return failures;
}

quantized_weight::PatternedWeightOptions nvfp4_options() {
    quantized_weight::PatternedWeightOptions options;
    options.weight_scale_divisor = 0.125F;
    options.input_scale_divisor  = kInputDivisor;
    return options;
}

int run_attn_input_proj_a4(const DevicePackedWeight& weight, std::int32_t tokens) {
    const auto x    = input_projection::make_bf16_activation(kHidden, tokens, 801U + tokens);
    DeviceBuffer dx = to_device_bf16(x);
    DeviceA4 activation(kHidden, tokens);
    activation.upload(nvfp4_activation_reference(x, kInputDivisor));
    const std::size_t q_count  = static_cast<std::size_t>(kQueryRows) * tokens;
    const std::size_t kv_count = static_cast<std::size_t>(kKvRows) * tokens;
    DeviceBuffer pq(q_count * 2), pgate(q_count * 2), pk(kv_count * 2), pv(kv_count * 2);
    DeviceBuffer aq(q_count * 2), agate(q_count * 2), ak(kv_count * 2), av(kv_count * 2);
    const Tensor tx(dx.p, DType::BF16, {kHidden, tokens});
    Tensor policy_q(pq.p, DType::BF16, {kQueryRows, tokens}),
        policy_gate(pgate.p, DType::BF16, {kQueryRows, tokens}),
        policy_k(pk.p, DType::BF16, {kKvRows, tokens}),
        policy_v(pv.p, DType::BF16, {kKvRows, tokens});
    Tensor a4_q(aq.p, DType::BF16, {kQueryRows, tokens}),
        a4_gate(agate.p, DType::BF16, {kQueryRows, tokens}),
        a4_k(ak.p, DType::BF16, {kKvRows, tokens}), a4_v(av.p, DType::BF16, {kKvRows, tokens});
    WorkspaceArena workspace(std::max<std::size_t>(
        256, ops::attn_input_proj_workspace_capacity_bytes(
                 QType::NVFP4, 14336, kHidden, ops::LinearPolicy::AllowA4, tokens, tokens)));
    ops::attn_input_proj(tx, weight.view(), policy_q, policy_gate, policy_k, policy_v,
                         ops::LinearPolicy::AllowA4, workspace, nullptr);
    ops::attn_input_proj(activation.view(), weight.view(), a4_q, a4_gate, a4_k, a4_v, nullptr);
    cuda_synchronize();
    const std::string label = "attn_input_proj A4 T=" + std::to_string(tokens);
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

int run_gdn_input_proj_a4(const DevicePackedWeight& weight, std::int32_t tokens) {
    const auto x    = input_projection::make_bf16_activation(kHidden, tokens, 831U + tokens);
    DeviceBuffer dx = to_device_bf16(x);
    DeviceA4 activation(kHidden, tokens);
    activation.upload(nvfp4_activation_reference(x, kInputDivisor));
    const std::size_t qkv_count = static_cast<std::size_t>(kGdnQkvRows) * tokens;
    const std::size_t z_count   = static_cast<std::size_t>(kGdnValueRows) * tokens;
    DeviceBuffer pqkv(qkv_count * 2), pz(z_count * 2), aqkv(qkv_count * 2), az(z_count * 2);
    const Tensor tx(dx.p, DType::BF16, {kHidden, tokens});
    Tensor policy_qkv(pqkv.p, DType::BF16, {kGdnQkvRows, tokens}),
        policy_z(pz.p, DType::BF16, {kGdnValueRows, tokens});
    Tensor a4_qkv(aqkv.p, DType::BF16, {kGdnQkvRows, tokens}),
        a4_z(az.p, DType::BF16, {kGdnValueRows, tokens});
    WorkspaceArena workspace(std::max<std::size_t>(
        256,
        ops::gdn_input_proj_workspace_capacity_bytes(QType::NVFP4, kGdnInputRows, kHidden,
                                                     ops::LinearPolicy::AllowA4, tokens, tokens)));
    ops::gdn_input_proj(tx, weight.view(), policy_qkv, policy_z, ops::LinearPolicy::AllowA4,
                        workspace, nullptr);
    ops::gdn_input_proj(activation.view(), weight.view(), a4_qkv, a4_z, nullptr);
    cuda_synchronize();
    const std::string label = "gdn_input_proj A4 T=" + std::to_string(tokens);
    int failures            = verify_exact((label + " qkv").c_str(), bf16_bits(aqkv, qkv_count),
                                           bf16_bits(pqkv, qkv_count));
    failures +=
        verify_exact((label + " z").c_str(), bf16_bits(az, z_count), bf16_bits(pz, z_count));
    return failures;
}

int run_linear_swiglu_a4(const DevicePackedWeight& weight, std::int32_t tokens) {
    const auto x    = input_projection::make_bf16_activation(kHidden, tokens, 851U + tokens);
    DeviceBuffer dx = to_device_bf16(x);
    DeviceA4 activation(kHidden, tokens);
    activation.upload(nvfp4_activation_reference(x, kInputDivisor));
    const std::size_t count = static_cast<std::size_t>(kIntermediate) * tokens;
    DeviceBuffer policy_out(count * 2), a4_out(count * 2);
    const Tensor tx(dx.p, DType::BF16, {kHidden, tokens});
    Tensor tpolicy(policy_out.p, DType::BF16, {kIntermediate, tokens});
    Tensor ta4(a4_out.p, DType::BF16, {kIntermediate, tokens});
    WorkspaceArena workspace(std::max<std::size_t>(
        256,
        ops::linear_swiglu_workspace_capacity_bytes(QType::NVFP4, 2 * kIntermediate, kHidden,
                                                    ops::LinearPolicy::AllowA4, tokens, tokens)));
    ops::linear_swiglu(tx, weight.view(), tpolicy, ops::LinearPolicy::AllowA4, workspace, nullptr);
    ops::linear_swiglu(activation.view(), weight.view(), ta4, nullptr);
    cuda_synchronize();
    return verify_exact(("linear_swiglu A4 T=" + std::to_string(tokens)).c_str(),
                        bf16_bits(a4_out, count), bf16_bits(policy_out, count));
}

// The A4-output LinearSwiGLU must publish exactly the A4 codec of the BF16 output the
// Tensor-output form writes for the same A4 input.
int run_linear_swiglu_a4_output(const DevicePackedWeight& weight, std::int32_t tokens) {
    const auto x = input_projection::make_bf16_activation(kHidden, tokens, 871U + tokens);
    DeviceA4 activation(kHidden, tokens);
    activation.upload(nvfp4_activation_reference(x, kInputDivisor));
    const std::size_t count = static_cast<std::size_t>(kIntermediate) * tokens;
    DeviceBuffer bf16_out(count * 2);
    Tensor tbf16(bf16_out.p, DType::BF16, {kIntermediate, tokens});
    DeviceA4 a4_out(kIntermediate, tokens);
    ops::A4Activation a4_view = a4_out.view();
    ops::linear_swiglu(activation.view(), weight.view(), tbf16, nullptr);
    ops::linear_swiglu(activation.view(), weight.view(), a4_view, nullptr);
    cuda_synchronize();
    return verify_codec("linear_swiglu A4 output T=" + std::to_string(tokens),
                        bf16_bits(bf16_out, count), kIntermediate, tokens, a4_out);
}

int run_linear_add_a4(const DevicePackedWeight& weight, std::int32_t tokens) {
    const std::int32_t k = weight.host.weight.k;
    const auto x         = input_projection::make_bf16_activation(k, tokens, 901U + tokens);
    std::vector<float> residual(static_cast<std::size_t>(kHidden) * tokens);
    fill_uniform(residual, 909U + tokens, -1.0F, 1.0F);
    round_to_bf16(residual);
    const std::size_t count = static_cast<std::size_t>(kHidden) * tokens;
    DeviceBuffer dx = to_device_bf16(x), policy_residual = to_device_bf16(residual),
                 a4_residual = to_device_bf16(residual);
    DeviceA4 activation(k, tokens);
    activation.upload(nvfp4_activation_reference(x, kInputDivisor));
    const Tensor tx(dx.p, DType::BF16, {k, tokens});
    Tensor tpolicy(policy_residual.p, DType::BF16, {kHidden, tokens});
    Tensor ta4(a4_residual.p, DType::BF16, {kHidden, tokens});
    WorkspaceArena workspace(std::max<std::size_t>(
        256, ops::linear_add_workspace_capacity_bytes(QType::NVFP4, kHidden, k,
                                                      ops::LinearPolicy::AllowA4, tokens, tokens)));
    ops::linear_add(tx, weight.view(), tpolicy, ops::LinearPolicy::AllowA4, workspace, nullptr);
    ops::linear_add(activation.view(), weight.view(), ta4, nullptr);
    cuda_synchronize();
    return verify_exact(
        ("linear_add A4 K=" + std::to_string(k) + " T=" + std::to_string(tokens)).c_str(),
        bf16_bits(a4_residual, count), bf16_bits(policy_residual, count));
}

} // namespace

int main(int argc, char** argv) {
    try {
        if (const int status = require_cuda(); status != 0) { return status; }
        const bool sanitizer = sanitizer_scope(argc, argv);
        // Prefill widths: the smallest W4A4 widths, a partial 16-token fragment, a partial and a
        // whole 256-token TMA tile, and several token tiles of the scale plane.
        const std::vector<std::int32_t> tokens =
            sanitizer ? std::vector<std::int32_t>{5, 100, 400}
                      : std::vector<std::int32_t>{1, 4, 5, 100, 160, 256, 400, 777, 1024};
        int failures = 0;
        for (const std::int32_t t : tokens) {
            failures += run_rmsnorm_a4(t);
            failures += run_gated_rmsnorm_a4(t);
            failures += run_gdn_norm_gating_a4(t);
            failures += run_sigmoid_mul_a4(t);
        }

        // The A4-output LinearSwiGLU epilogues are producers too: the MMA group epilogue at T=5
        // and T=100 and the TMA epilogue from T=400.
        const DevicePackedWeight gate_up(quantized_weight::make_patterned_weight(
            QType::NVFP4, 2 * kIntermediate, kHidden, 907U, nvfp4_options()));
        for (const std::int32_t t : tokens) {
            if (t >= ops::kA4MlpGateUpMinTokens) {
                failures += run_linear_swiglu_a4_output(gate_up, t);
            }
        }

        // The consumers launch the projections' existing W4A4 kernels, which the Linear-family
        // tests sanitize; the --sanitizer cases cover the producer kernels only.
        if (sanitizer) {
            std::cout << (failures == 0 ? "OK" : "FAIL") << " A4 activation producers\n";
            return failures == 0 ? 0 : 1;
        }
        const DevicePackedWeight attention(quantized_weight::make_patterned_weight(
            QType::NVFP4, 14336, kHidden, 905U, nvfp4_options()));
        const DevicePackedWeight residual6144(quantized_weight::make_patterned_weight(
            QType::NVFP4, kHidden, kQueryRows, 911U, nvfp4_options()));
        const DevicePackedWeight residual17408(quantized_weight::make_patterned_weight(
            QType::NVFP4, kHidden, kIntermediate, 913U, nvfp4_options()));
        const DevicePackedWeight gdn_input(quantized_weight::make_patterned_weight(
            QType::NVFP4, kGdnInputRows, kHidden, 915U, nvfp4_options()));
        // The GDN input projection additionally covers its first W4A4 width and its
        // M64N128S3 and M128N128Resident routes.
        for (const std::int32_t t : {ops::kA4GdnInputMinTokens, 48, 300, 600}) {
            failures += run_gdn_input_proj_a4(gdn_input, t);
        }
        for (const std::int32_t t : tokens) {
            if (t >= ops::kA4GdnInputMinTokens) { failures += run_gdn_input_proj_a4(gdn_input, t); }
            if (t >= ops::kA4Residual6144MinTokens) {
                failures += run_linear_add_a4(residual6144, t);
            }
            if (t >= ops::kA4Residual17408MinTokens) {
                failures += run_linear_add_a4(residual17408, t);
            }
            if (t >= ops::kA4AttnInputMinTokens) {
                failures += run_attn_input_proj_a4(attention, t);
            }
            if (t >= ops::kA4MlpGateUpMinTokens) { failures += run_linear_swiglu_a4(gate_up, t); }
        }
        std::cout << (failures == 0 ? "OK" : "FAIL") << " A4 activation producers/consumers\n";
        return failures == 0 ? 0 : 1;
    } catch (const std::exception& error) {
        std::cerr << "FAIL: " << error.what() << '\n';
        return 1;
    }
}
