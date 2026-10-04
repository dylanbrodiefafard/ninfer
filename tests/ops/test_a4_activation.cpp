// Independent complete FP64 activation/cast/codec/projection oracles at native
// partial tiled extents. Exact codec checks, padding and guards protect the handoff.
#include "ninfer/ops/rmsnorm.h"
#include "ninfer/ops/attn_input_proj.h"
#include "ninfer/ops/sigmoid_mul.h"
#include "ninfer/ops/linear_add.h"
#include "ninfer/ops/linear_swiglu.h"
#include "core/device.h"
#include "core/decode_graph.h"
#include "ops/op_tester.h"
#include "ops/nvfp4_activation_ref.h"
#include "ops/quantized_weight.h"
#include "ops/sanitizer_scope.h"
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <exception>
#include <limits>
#include <random>
#include <stdexcept>
#include <vector>
using namespace ninfer;

namespace {
struct FixtureWeight {
    DeviceBuffer storage;
    Weight weight;
    std::size_t scale_offset;
    std::size_t scale_bytes;
};

FixtureWeight make_fixture_weight(int rows, int columns) {
    auto host = ninfer::test::quantized_weight::make_patterned_weight(
        QType::NVFP4, rows, columns, 1097,
        {.weight_scale_divisor = 128.0F, .input_scale_divisor = 3.5F});
    DeviceBuffer storage(host.payload.size());
    auto weight = host.device_weight(storage.p);
    return {std::move(storage), weight, static_cast<std::size_t>(host.scale_plane_offset),
            static_cast<std::size_t>(host.scale_plane_bytes)};
}
} // namespace

namespace rms_fixture {
constexpr int K = 5120, N = 14336, Patterns = 8;
constexpr int T                = 385;
constexpr float Divisor        = 3.5F;
constexpr double WeightDivisor = 128;

std::size_t scale_offset(int token, int group) {
    const std::size_t tile         = static_cast<std::size_t>(token / 256) * 20 + group / 16;
    const std::size_t token_offset = static_cast<std::size_t>(token % 256) * 16;
    return tile * 4096 + token_offset + group % 16;
}

double decode4(int code) {
    constexpr double v[]{0, .5, 1, 1.5, 2, 3, 4, 6};
    return (code & 8) ? -v[code & 7] : v[code & 7];
}

double decode8(int code) {
    const int e = (code >> 3) & 15, m = code & 7;
    return e == 0 ? std::ldexp(double(m), -9) : std::ldexp(1 + double(m) / 8, e - 7);
}

float bf16(double x) {
    const bool negative    = std::signbit(x);
    const double magnitude = std::abs(x);
    const auto central     = ninfer::test::f32_to_bf16(static_cast<float>(magnitude));
    double distance        = std::numeric_limits<double>::infinity();
    std::uint16_t best     = central;
    for (int step = -1; step <= 1; ++step) {
        const int code = static_cast<int>(central) + step;
        if (code < 0 || code >= 0x7f80) continue;
        const double error =
            std::abs(magnitude - ninfer::test::bf16_to_f32(static_cast<std::uint16_t>(code)));
        if (error < distance || (error == distance && (code & 1) == 0)) {
            distance = error;
            best     = static_cast<std::uint16_t>(code);
        }
    }
    return ninfer::test::bf16_to_f32(static_cast<std::uint16_t>(best | (negative ? 0x8000 : 0)));
}

double represented(const ninfer::test::Nvfp4ActivationReference& codec, int index) {
    const int code = (codec.codes[index / 2] >> (4 * (index & 1))) & 15;
    return decode4(code) * decode8(codec.scales[index / 16]) / Divisor;
}

void require(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}

void checked(const char* name, const std::vector<double>& actual, const std::vector<double>& ideal,
             ninfer::test::ReductionCriterion tolerance) {
    if (ninfer::test::verify_reduction(name, actual, ideal, tolerance))
        throw std::runtime_error(name);
}

void run(cudaStream_t stream) {
    std::mt19937 rng(781);
    std::uniform_real_distribution<float> random(-1, 1);
    std::vector<std::uint16_t> host_x(static_cast<std::size_t>(K) * T), host_w(K);
    std::vector<float> xsmall(K * Patterns), w(K);
    for (int d = 0; d < K; ++d) {
        host_w[d] = ninfer::test::f32_to_bf16(random(rng) * .5F);
        w[d]      = ninfer::test::bf16_to_f32(host_w[d]);
    }
    for (int t = 0; t < Patterns; ++t)
        for (int d = 0; d < K; ++d) {
            float v = random(rng);
            if (t == 0) v = 0;
            if (t == 1) v *= 1e-4F;
            if (t == 2 && d % 257 == 0) v *= 100;
            host_x[t * K + d] = ninfer::test::f32_to_bf16(v);
            xsmall[t * K + d] = ninfer::test::bf16_to_f32(host_x[t * K + d]);
        }
    for (int t = Patterns; t < T; ++t)
        std::copy_n(host_x.data() + static_cast<std::size_t>(t % Patterns) * K, K,
                    host_x.data() + static_cast<std::size_t>(t) * K);
    std::vector<double> rms_ideal(K * Patterns);
    std::vector<float> rounded(static_cast<std::size_t>(K) * Patterns);
    for (int t = 0; t < Patterns; ++t) {
        double ss = 0;
        for (int d = 0; d < K; ++d) ss += double(xsmall[t * K + d]) * xsmall[t * K + d];
        const double inv = 1 / std::sqrt(ss / K + double(1e-6F));
        for (int d = 0; d < K; ++d) {
            rms_ideal[t * K + d] = xsmall[t * K + d] * inv * (1 + w[d]);
            rounded[t * K + d]   = bf16(rms_ideal[t * K + d]);
        }
    }
    const auto canonical = ninfer::test::nvfp4_activation_reference(rounded, Divisor);
    auto packed_weight   = make_fixture_weight(N, K);
    packed_weight.weight.weight_scale_divisor = WeightDivisor;
    std::vector<std::uint8_t> payload(packed_weight.storage.bytes), weight_codes(Patterns * K),
        weight_scales(Patterns * K / 16);
    for (auto& c : weight_codes) c = static_cast<std::uint8_t>(rng() & 15U);
    for (auto& s : weight_scales) s = static_cast<std::uint8_t>(32 + rng() % 25);
    for (int n = 0; n < N; ++n) {
        for (int d = 0; d < K; d += 2)
            payload[static_cast<std::size_t>(n) * K / 2 + d / 2] =
                weight_codes[(n % Patterns) * K + d] |
                (weight_codes[(n % Patterns) * K + d + 1] << 4);
        for (int g = 0; g < K / 16; ++g) {
            const std::size_t offset =
                packed_weight.scale_offset +
                (static_cast<std::size_t>(n / 128) * (K / 64) + g / 4) * 512 +
                static_cast<std::size_t>(n % 32) * 16 +
                static_cast<std::size_t>((n % 128) / 32) * 4 + g % 4;
            payload[offset] = weight_scales[(n % Patterns) * (K / 16) + g];
        }
    }
    const float wd = WeightDivisor;
    std::memcpy(payload.data() + packed_weight.scale_offset + packed_weight.scale_bytes, &wd,
                sizeof(wd));
    packed_weight.storage.copy_from_host(payload.data(), payload.size());
    DeviceBuffer dx(host_x.size() * 2), dw(host_w.size() * 2), dn(dx.bytes), db(dx.bytes),
        codes(static_cast<std::size_t>(K) * T / 2 + 512),
        scales(static_cast<std::size_t>(K) * ((static_cast<std::size_t>(T) + 255) / 256 * 256) /
                   16 +
               512);
    dx.copy_from_host(host_x.data(), dx.bytes);
    dw.copy_from_host(host_w.data(), dw.bytes);
    codes.fill(0xa5);
    scales.fill(0xa5);
    Tensor x(dx.p, DType::BF16, {K, T}), weight(dw.p, DType::BF16, {K}),
        normalized(dn.p, DType::BF16, {K, T}), baseline_normalized(db.p, DType::BF16, {K, T});
    ops::A4Activation a4{{codes.p, static_cast<std::size_t>(K) * T / 2},
                         {scales.p, static_cast<std::size_t>(K) *
                                        ((static_cast<std::size_t>(T) + 255) / 256 * 256) / 16},
                         K,
                         T,
                         Divisor};
    DeviceBuffer q(static_cast<std::size_t>(6144) * T * 2), gate(q.bytes),
        key(static_cast<std::size_t>(1024) * T * 2), value(key.bytes);
    Tensor tq(q.p, DType::BF16, {6144, T}), tg(gate.p, DType::BF16, {6144, T}),
        tk(key.p, DType::BF16, {1024, T}), tv(value.p, DType::BF16, {1024, T});
    WorkspaceArena workspace(ops::attn_input_proj_workspace_capacity_bytes(
        QType::NVFP4, N, K, ops::LinearPolicy::AllowA4, T, T));
    const auto baseline = [&](cudaStream_t s) {
        ops::rmsnorm(x, weight, 1e-6F, true, baseline_normalized, s);
        ops::attn_input_proj(baseline_normalized, packed_weight.weight, tq, tg, tk, tv,
                             ops::LinearPolicy::AllowA4, workspace, s);
    };
    const auto candidate = [&](cudaStream_t s) {
        ops::rmsnorm_a4(x, weight, 1e-6F, nullptr, a4, s);
        ops::attn_input_proj(a4, packed_weight.weight, tq, tg, tk, tv, s);
    };
    CUDA_CHECK(cudaDeviceSynchronize());
    ops::rmsnorm_a4(x, weight, 1e-6F, &normalized, a4, stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));
    const auto ny =
        ninfer::test::from_device_bf16(normalized.data, static_cast<std::size_t>(K) * T);
    std::vector<double> norm_ref(ny.size());
    for (int t = 0; t < T; ++t)
        std::copy_n(rms_ideal.data() + static_cast<std::size_t>(t % Patterns) * K, K,
                    norm_ref.data() + static_cast<std::size_t>(t) * K);
    checked("RMS primary FP64", ny, norm_ref, {.00185, 1e-5, .004});
    // Codec exactness on explicitly published BF16 is supplementary, not the full-chain oracle.
    const std::vector<float> supplementary(ny.begin(),
                                           ny.begin() + static_cast<std::size_t>(K) * Patterns);
    const auto actual_codec = ninfer::test::nvfp4_activation_reference(supplementary, Divisor);
    std::vector<std::uint8_t> got_codes(codes.bytes), got_scales(scales.bytes);
    CUDA_CHECK(cudaMemcpy(got_codes.data(), codes.p, codes.bytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(got_scales.data(), scales.p, scales.bytes, cudaMemcpyDeviceToHost));
    for (int t = 0; t < T; ++t) {
        for (int d = 0; d < K / 2; ++d)
            require(got_codes[static_cast<std::size_t>(t) * K / 2 + d] ==
                        actual_codec.codes[static_cast<std::size_t>(t % Patterns) * K / 2 + d],
                    "exact signed A4 code mismatch");
        for (int g = 0; g < K / 16; ++g)
            require(got_scales[scale_offset(t, g)] ==
                        actual_codec.scales[static_cast<std::size_t>(t % Patterns) * K / 16 + g],
                    "exact tiled scale mismatch");
    }
    const int padded_tokens = static_cast<int>((static_cast<std::size_t>(T) + 255) / 256 * 256);
    for (int t = T; t < padded_tokens; ++t)
        for (int g = 0; g < K / 16; ++g)
            require(got_scales[scale_offset(t, g)] == 0, "padded scale must be zero");
    for (std::size_t i = a4.codes.bytes; i < got_codes.size(); ++i)
        require(got_codes[i] == 0xa5, "code guard modified");
    for (std::size_t i = a4.scales.bytes; i < got_scales.size(); ++i)
        require(got_scales[i] == 0xa5, "scale guard modified");
    std::array<std::array<double, Patterns>, Patterns> ideals{}, published{};
    for (int t = 0; t < Patterns; ++t)
        for (int n = 0; n < Patterns; ++n)
            for (int d = 0; d < K; ++d) {
                const double stored = decode4(weight_codes[n * K + d]) *
                                      decode8(weight_scales[n * (K / 16) + d / 16]) / WeightDivisor;
                ideals[t][n] += represented(canonical, t * K + d) * stored;
                published[t][n] += represented(actual_codec, t * K + d) * stored;
            }
    auto check_output = [&](const Tensor& output, int rows, int offset, bool use_published) {
        const auto actual =
            ninfer::test::from_device_bf16(output.data, static_cast<std::size_t>(rows) * T);
        std::vector<double> ref(actual.size());
        for (int t = 0; t < T; ++t)
            for (int n = 0; n < rows; ++n)
                ref[static_cast<std::size_t>(t) * rows + n] =
                    (use_published ? published : ideals)[t % Patterns][(n + offset) % Patterns];
        checked(use_published ? "decoded published A4 projection"
                              : "complete FP64 RMS->cast->codec->projection",
                actual, ref, {.16, 1.0 / 256, .16});
    };
    baseline(stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));
    check_output(tq, 6144, 0, false);
    check_output(tk, 1024, 6144, false);
    check_output(tg, 6144, 7168, false);
    check_output(tv, 1024, 13312, false);
    candidate(stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));
    for (bool published_oracle : {false, true}) {
        check_output(tq, 6144, 0, published_oracle);
        check_output(tk, 1024, 6144, published_oracle);
        check_output(tg, 6144, 7168, published_oracle);
        check_output(tv, 1024, 13312, published_oracle);
    }
    DecodeGraphDefinition definition;
    definition.capture(stream, [&] { candidate(stream); });
    DecodeGraphExecutable executable;
    executable.instantiate(definition);
    executable.launch(stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));
    check_output(tq, 6144, 0, false);
}
} // namespace rms_fixture

namespace sigmoid_fixture {
constexpr int K = 6144, N = 5120, Patterns = 8;
constexpr int T                = 513;
constexpr float Divisor        = 3.5F;
constexpr double WeightDivisor = 128;

std::size_t scale_offset(int token, int group) {
    const std::size_t tile         = static_cast<std::size_t>(token / 256) * 24 + group / 16;
    const std::size_t token_offset = static_cast<std::size_t>(token % 256) * 16;
    return tile * 4096 + token_offset + group % 16;
}

double decode4(int code) {
    constexpr double v[]{0, .5, 1, 1.5, 2, 3, 4, 6};
    return (code & 8) ? -v[code & 7] : v[code & 7];
}

double decode8(int code) {
    const int e = (code >> 3) & 15, m = code & 7;
    return e == 0 ? std::ldexp(double(m), -9) : std::ldexp(1 + double(m) / 8, e - 7);
}

float bf16(double x) {
    const bool negative    = std::signbit(x);
    const double magnitude = std::abs(x);
    const auto central     = ninfer::test::f32_to_bf16(static_cast<float>(magnitude));
    double distance        = std::numeric_limits<double>::infinity();
    std::uint16_t best     = central;
    for (int step = -1; step <= 1; ++step) {
        const int code = static_cast<int>(central) + step;
        if (code < 0 || code >= 0x7f80) continue;
        const double error =
            std::abs(magnitude - ninfer::test::bf16_to_f32(static_cast<std::uint16_t>(code)));
        if (error < distance || (error == distance && (code & 1) == 0)) {
            distance = error;
            best     = static_cast<std::uint16_t>(code);
        }
    }
    return ninfer::test::bf16_to_f32(static_cast<std::uint16_t>(best | (negative ? 0x8000 : 0)));
}

double represented(const ninfer::test::Nvfp4ActivationReference& codec, int index) {
    const int code = (codec.codes[index / 2] >> (4 * (index & 1))) & 15;
    return decode4(code) * decode8(codec.scales[index / 16]) / Divisor;
}

void require(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}

void checked(const char* name, const std::vector<double>& actual, const std::vector<double>& ideal,
             ninfer::test::ReductionCriterion tolerance) {
    if (ninfer::test::verify_reduction(name, actual, ideal, tolerance))
        throw std::runtime_error(name);
}

void run(cudaStream_t stream) {
    std::mt19937 rng(1697);
    std::uniform_real_distribution<float> random(-1, 1);
    std::vector<std::uint16_t> host_x(static_cast<std::size_t>(K) * T), host_g(host_x.size()),
        initial(static_cast<std::size_t>(N) * T);
    std::vector<float> xsmall(K * Patterns), gsmall(xsmall.size());
    std::vector<double> producer_ideal(static_cast<std::size_t>(K) * Patterns);
    std::vector<float> rounded(static_cast<std::size_t>(K) * Patterns);
    for (int t = 0; t < Patterns; ++t)
        for (int d = 0; d < K; ++d) {
            float x = random(rng), g = random(rng) * 8;
            if (t < 3) g = 0;
            if (t == 0) x = 0;
            if (t == 1) x *= 1e-4F;
            if (t == 2 && d % 257 == 0) x *= 100;
            const int i       = t * K + d;
            host_x[i]         = ninfer::test::f32_to_bf16(x);
            host_g[i]         = ninfer::test::f32_to_bf16(g);
            xsmall[i]         = ninfer::test::bf16_to_f32(host_x[i]);
            gsmall[i]         = ninfer::test::bf16_to_f32(host_g[i]);
            producer_ideal[i] = double(xsmall[i]) / (1 + std::exp(-double(gsmall[i])));
            rounded[i]        = bf16(producer_ideal[i]);
        }
    for (int t = Patterns; t < T; ++t) {
        std::copy_n(host_x.data() + static_cast<std::size_t>(t % Patterns) * K, K,
                    host_x.data() + static_cast<std::size_t>(t) * K);
        std::copy_n(host_g.data() + static_cast<std::size_t>(t % Patterns) * K, K,
                    host_g.data() + static_cast<std::size_t>(t) * K);
    }
    for (auto& v : initial) v = ninfer::test::f32_to_bf16(random(rng) * .1F);
    const auto canonical = ninfer::test::nvfp4_activation_reference(rounded, Divisor);
    auto packed_weight   = make_fixture_weight(N, K);
    packed_weight.weight.weight_scale_divisor = WeightDivisor;
    std::vector<std::uint8_t> payload(packed_weight.storage.bytes), weight_codes(Patterns * K),
        weight_scales(Patterns * K / 16);
    for (auto& c : weight_codes) c = static_cast<std::uint8_t>(rng() & 15U);
    for (auto& s : weight_scales) s = static_cast<std::uint8_t>(32 + rng() % 25);
    for (int n = 0; n < N; ++n) {
        for (int d = 0; d < K; d += 2)
            payload[static_cast<std::size_t>(n) * K / 2 + d / 2] =
                weight_codes[(n % Patterns) * K + d] |
                (weight_codes[(n % Patterns) * K + d + 1] << 4);
        for (int g = 0; g < K / 16; ++g) {
            const std::size_t offset =
                packed_weight.scale_offset +
                (static_cast<std::size_t>(n / 128) * (K / 64) + g / 4) * 512 +
                static_cast<std::size_t>(n % 32) * 16 +
                static_cast<std::size_t>((n % 128) / 32) * 4 + g % 4;
            payload[offset] = weight_scales[(n % Patterns) * (K / 16) + g];
        }
    }
    const float wd = WeightDivisor;
    std::memcpy(payload.data() + packed_weight.scale_offset + packed_weight.scale_bytes, &wd,
                sizeof(wd));
    packed_weight.storage.copy_from_host(payload.data(), payload.size());
    DeviceBuffer dx(host_x.size() * 2), dg(host_g.size() * 2), scratch(dx.bytes),
        diagnostic(dx.bytes), residual(initial.size() * 2), residual_initial(residual.bytes),
        codes(static_cast<std::size_t>(K) * T / 2 + 512),
        scales(static_cast<std::size_t>(K) * ((static_cast<std::size_t>(T) + 255) / 256 * 256) /
                   16 +
               512);
    dx.copy_from_host(host_x.data(), dx.bytes);
    dg.copy_from_host(host_g.data(), dg.bytes);
    residual_initial.copy_from_host(initial.data(), residual_initial.bytes);
    codes.fill(0xa5);
    scales.fill(0xa5);
    Tensor x(dx.p, DType::BF16, {K, T}), gate(dg.p, DType::BF16, {K, T}),
        working(scratch.p, DType::BF16, {K, T}), normalized(diagnostic.p, DType::BF16, {K, T}),
        result(residual.p, DType::BF16, {N, T});
    ops::A4Activation a4{{codes.p, static_cast<std::size_t>(K) * T / 2},
                         {scales.p, static_cast<std::size_t>(K) *
                                        ((static_cast<std::size_t>(T) + 255) / 256 * 256) / 16},
                         K,
                         T,
                         Divisor};
    WorkspaceArena workspace(ops::linear_add_workspace_capacity_bytes(
        QType::NVFP4, N, K, ops::LinearPolicy::AllowA4, T, T));
    const auto reset = [&](cudaStream_t s) {
        CUDA_CHECK(cudaMemcpyAsync(working.data, x.data, dx.bytes, cudaMemcpyDeviceToDevice, s));
        CUDA_CHECK(cudaMemcpyAsync(result.data, residual_initial.p, residual.bytes,
                                   cudaMemcpyDeviceToDevice, s));
    };
    const auto baseline = [&](cudaStream_t s) {
        reset(s);
        ops::sigmoid_mul(gate, working, s);
        ops::linear_add(working, packed_weight.weight, result, ops::LinearPolicy::AllowA4,
                        workspace, s);
    };
    const auto candidate = [&](cudaStream_t s) {
        reset(s);
        ops::sigmoid_mul_a4(gate, working, nullptr, a4, s);
        ops::linear_add(a4, packed_weight.weight, result, s);
    };
    CUDA_CHECK(cudaDeviceSynchronize());
    ops::sigmoid_mul_a4(gate, x, &normalized, a4, stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));
    const auto ny =
        ninfer::test::from_device_bf16(normalized.data, static_cast<std::size_t>(K) * T);
    std::vector<double> producer_ref(ny.size());
    for (int t = 0; t < T; ++t)
        std::copy_n(producer_ideal.data() + static_cast<std::size_t>(t % Patterns) * K, K,
                    producer_ref.data() + static_cast<std::size_t>(t) * K);
    if (ninfer::test::verify_pointwise("sigmoid primary FP64", ny, producer_ref, {2e-5, .00405}))
        throw std::runtime_error("sigmoid existing mathematical criterion failed");
    const std::vector<float> supplementary(ny.begin(),
                                           ny.begin() + static_cast<std::size_t>(K) * Patterns);
    const auto actual_codec = ninfer::test::nvfp4_activation_reference(supplementary, Divisor);
    std::vector<std::uint8_t> got_codes(codes.bytes), got_scales(scales.bytes);
    codes.copy_to_host(got_codes.data(), got_codes.size());
    scales.copy_to_host(got_scales.data(), got_scales.size());
    for (int t = 0; t < T; ++t) {
        for (int d = 0; d < K / 2; ++d) {
            const auto got = got_codes[static_cast<std::size_t>(t) * K / 2 + d];
            require(got == actual_codec.codes[static_cast<std::size_t>(t % Patterns) * K / 2 + d],
                    "exact supplementary signed codes failed");
            if (t % Patterns < 3)
                require(got == canonical.codes[static_cast<std::size_t>(t % Patterns) * K / 2 + d],
                        "isolated represented-input zero/tiny/outlier exact codec failed");
        }
        for (int g = 0; g < K / 16; ++g) {
            const auto got = got_scales[scale_offset(t, g)];
            require(got == actual_codec.scales[static_cast<std::size_t>(t % Patterns) * K / 16 + g],
                    "exact tiled scales failed");
            if (t % Patterns < 3)
                require(got ==
                            canonical.scales[static_cast<std::size_t>(t % Patterns) * K / 16 + g],
                        "isolated represented-input exact scale failed");
        }
    }
    const int padded_tokens = static_cast<int>((static_cast<std::size_t>(T) + 255) / 256 * 256);
    for (int t = T; t < padded_tokens; ++t)
        for (int g = 0; g < K / 16; ++g)
            require(got_scales[scale_offset(t, g)] == 0, "padded scale must be zero");
    for (std::size_t i = a4.codes.bytes; i < got_codes.size(); ++i)
        require(got_codes[i] == 0xa5, "code guard changed");
    for (std::size_t i = a4.scales.bytes; i < got_scales.size(); ++i)
        require(got_scales[i] == 0xa5, "scale guard changed");
    std::array<std::array<double, Patterns>, Patterns> ideals{}, published{};
    for (int t = 0; t < Patterns; ++t)
        for (int n = 0; n < Patterns; ++n)
            for (int d = 0; d < K; ++d) {
                const double stored = decode4(weight_codes[n * K + d]) *
                                      decode8(weight_scales[n * (K / 16) + d / 16]) / WeightDivisor;
                ideals[t][n] += represented(canonical, t * K + d) * stored;
                published[t][n] += represented(actual_codec, t * K + d) * stored;
            }
    auto check_output = [&](bool use_published) {
        const auto actual = ninfer::test::from_device_bf16(result.data, initial.size());
        std::vector<double> ref(actual.size());
        for (int t = 0; t < T; ++t)
            for (int n = 0; n < N; ++n)
                ref[static_cast<std::size_t>(t) * N + n] =
                    (use_published ? published : ideals)[t % Patterns][n % Patterns] +
                    ninfer::test::bf16_to_f32(initial[static_cast<std::size_t>(t) * N + n]);
        checked(use_published ? "closed projection independent decoded A4"
                              : "full FP64 sigmoid->cast->codec->exactweights+residual",
                actual, ref, {.16, 1.0 / 256, .16});
    };
    baseline(stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));
    check_output(false);
    candidate(stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));
    check_output(false);
    check_output(true);
    std::vector<std::uint16_t> unchanged(host_x.size());
    scratch.copy_to_host(unchanged.data(), scratch.bytes);
    require(unchanged == host_x, "packed producer modified input");
    dg.copy_to_host(unchanged.data(), dg.bytes);
    require(unchanged == host_g, "packed producer modified gate");
    DecodeGraphDefinition definition;
    definition.capture(stream, [&] { candidate(stream); });
    DecodeGraphExecutable executable;
    executable.instantiate(definition);
    executable.launch(stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));
    check_output(false);
}
} // namespace sigmoid_fixture

namespace swiglu_fixture {
constexpr int Input = 5120, K = 17408, N = 5120, Patterns = 8;
int T                          = 1024;
constexpr float Divisor        = 3.5F;
constexpr double WeightDivisor = 256;

std::size_t scale_offset(int token, int group) {
    const std::size_t tile         = static_cast<std::size_t>(token / 256) * 68 + group / 16;
    const std::size_t token_offset = static_cast<std::size_t>(token % 256) * 16;
    return tile * 4096 + token_offset + group % 16;
}

double decode4(int code) {
    constexpr double v[]{0, .5, 1, 1.5, 2, 3, 4, 6};
    return (code & 8) ? -v[code & 7] : v[code & 7];
}

double decode8(int code) {
    const int e = (code >> 3) & 15, m = code & 7;
    return e == 0 ? std::ldexp(double(m), -9) : std::ldexp(1 + double(m) / 8, e - 7);
}

float bf16(double x) {
    const bool negative    = std::signbit(x);
    const double magnitude = std::abs(x);
    const auto central     = ninfer::test::f32_to_bf16(static_cast<float>(magnitude));
    double distance        = std::numeric_limits<double>::infinity();
    std::uint16_t best     = central;
    for (int step = -1; step <= 1; ++step) {
        const int code = static_cast<int>(central) + step;
        if (code < 0 || code >= 0x7f80) continue;
        const double error =
            std::abs(magnitude - ninfer::test::bf16_to_f32(static_cast<std::uint16_t>(code)));
        if (error < distance || (error == distance && (code & 1) == 0)) {
            distance = error;
            best     = static_cast<std::uint16_t>(code);
        }
    }
    return ninfer::test::bf16_to_f32(static_cast<std::uint16_t>(best | (negative ? 0x8000 : 0)));
}

double represented(const ninfer::test::Nvfp4ActivationReference& codec, int index) {
    const int code = (codec.codes[index / 2] >> (4 * (index & 1))) & 15;
    return decode4(code) * decode8(codec.scales[index / 16]) / Divisor;
}

void require(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}

void checked(const char* name, const std::vector<double>& actual, const std::vector<double>& ideal,
             ninfer::test::ReductionCriterion tolerance) {
    if (ninfer::test::verify_reduction(name, actual, ideal, tolerance))
        throw std::runtime_error(name);
}

struct WeightFixture {
    FixtureWeight packed;
    std::vector<std::uint8_t> codes, scales;
    int input_rows, patterns;
};

WeightFixture make_weight(int rows, int input, int patterns, std::mt19937& rng) {
    auto packed                        = make_fixture_weight(rows, input);
    packed.weight.weight_scale_divisor = WeightDivisor;
    std::vector<std::uint8_t> payload(packed.storage.bytes),
        codes(static_cast<std::size_t>(patterns) * input),
        scales(static_cast<std::size_t>(patterns) * input / 16);
    for (auto& c : codes) c = static_cast<std::uint8_t>(rng() & 15U);
    for (auto& s : scales) s = static_cast<std::uint8_t>(32 + rng() % 25);
    for (int n = 0; n < rows; ++n) {
        const int pattern = patterns == 16 ? (n / K) * 8 + n % 8 : n % 8;
        for (int d = 0; d < input; d += 2)
            payload[static_cast<std::size_t>(n) * input / 2 + d / 2] =
                codes[pattern * input + d] | (codes[pattern * input + d + 1] << 4);
        for (int g = 0; g < input / 16; ++g) {
            const std::size_t offset =
                packed.scale_offset +
                (static_cast<std::size_t>(n / 128) * (input / 64) + g / 4) * 512 +
                static_cast<std::size_t>(n % 32) * 16 +
                static_cast<std::size_t>((n % 128) / 32) * 4 + g % 4;
            payload[offset] = scales[pattern * (input / 16) + g];
        }
    }
    const float wd = WeightDivisor;
    std::memcpy(payload.data() + packed.scale_offset + packed.scale_bytes, &wd, sizeof(wd));
    packed.storage.copy_from_host(payload.data(), payload.size());
    return {std::move(packed), std::move(codes), std::move(scales), input, patterns};
}

double stored(const WeightFixture& w, int pattern, int column) {
    return decode4(w.codes[pattern * w.input_rows + column]) *
           decode8(w.scales[pattern * (w.input_rows / 16) + column / 16]) / WeightDivisor;
}

void run(cudaStream_t stream) {
    std::mt19937 rng(3197);
    std::uniform_real_distribution<float> random(-1, 1);
    std::vector<std::uint16_t> host_x(static_cast<std::size_t>(Input) * T),
        initial(static_cast<std::size_t>(N) * T);
    std::vector<float> xsmall(Input * Patterns);
    for (int t = 0; t < Patterns; ++t)
        for (int d = 0; d < Input; ++d) {
            float x = random(rng);
            if (t == 0) x = 0;
            if (t == 7 && d % 257 == 0) x *= 4;
            host_x[t * Input + d] = ninfer::test::f32_to_bf16(x);
            xsmall[t * Input + d] = ninfer::test::bf16_to_f32(host_x[t * Input + d]);
        }
    for (int t = Patterns; t < T; ++t)
        std::copy_n(host_x.data() + static_cast<std::size_t>(t % Patterns) * Input, Input,
                    host_x.data() + static_cast<std::size_t>(t) * Input);
    for (auto& v : initial) v = ninfer::test::f32_to_bf16(random(rng) * .1F);
    auto gu = make_weight(2 * K, Input, 16, rng), down = make_weight(N, K, 8, rng);
    std::array<std::array<double, Patterns>, Patterns> producer_patterns{};
    for (int t = 0; t < Patterns; ++t)
        for (int f = 0; f < Patterns; ++f) {
            double g = 0, u = 0;
            for (int d = 0; d < Input; ++d) {
                g += xsmall[t * Input + d] * stored(gu, f, d);
                u += xsmall[t * Input + d] * stored(gu, f + 8, d);
            }
            producer_patterns[t][f] = g / (1 + std::exp(-g)) * u;
        }
    std::vector<double> producer_ideal(static_cast<std::size_t>(K) * Patterns);
    std::vector<float> rounded(static_cast<std::size_t>(K) * Patterns);
    for (int t = 0; t < Patterns; ++t)
        for (int d = 0; d < K; ++d) {
            producer_ideal[t * K + d] = producer_patterns[t][d % Patterns];
            rounded[t * K + d]        = bf16(producer_ideal[t * K + d]);
        }
    const auto canonical = ninfer::test::nvfp4_activation_reference(rounded, Divisor);
    DeviceBuffer dx(host_x.size() * 2), diagnostic(static_cast<std::size_t>(K) * T * 2),
        baseline_bf16(diagnostic.bytes), residual(initial.size() * 2),
        residual_initial(residual.bytes), codes(static_cast<std::size_t>(K) * T / 2 + 512),
        scales(static_cast<std::size_t>(K) * ((static_cast<std::size_t>(T) + 255) / 256 * 256) /
                   16 +
               512);
    dx.copy_from_host(host_x.data(), dx.bytes);
    residual_initial.copy_from_host(initial.data(), residual_initial.bytes);
    codes.fill(0xa5);
    scales.fill(0xa5);
    Tensor x(dx.p, DType::BF16, {Input, T}), normalized(diagnostic.p, DType::BF16, {K, T}),
        intermediate(baseline_bf16.p, DType::BF16, {K, T}), result(residual.p, DType::BF16, {N, T});
    ops::A4Activation a4{{codes.p, static_cast<std::size_t>(K) * T / 2},
                         {scales.p, static_cast<std::size_t>(K) *
                                        ((static_cast<std::size_t>(T) + 255) / 256 * 256) / 16},
                         K,
                         T,
                         Divisor};
    const auto capacity =
        std::max(ops::linear_swiglu_workspace_capacity_bytes(QType::NVFP4, 2 * K, Input,
                                                             ops::LinearPolicy::AllowA4, T, T),
                 ops::linear_add_workspace_capacity_bytes(QType::NVFP4, N, K,
                                                          ops::LinearPolicy::AllowA4, T, T));
    WorkspaceArena workspace(capacity);
    const auto reset = [&](cudaStream_t s) {
        CUDA_CHECK(cudaMemcpyAsync(result.data, residual_initial.p, residual.bytes,
                                   cudaMemcpyDeviceToDevice, s));
    };
    const auto baseline = [&](cudaStream_t s) {
        reset(s);
        ops::linear_swiglu(x, gu.packed.weight, intermediate, ops::LinearPolicy::AllowA4, workspace,
                           s);
        ops::linear_add(intermediate, down.packed.weight, result, ops::LinearPolicy::AllowA4,
                        workspace, s);
    };
    const auto candidate = [&](cudaStream_t s) {
        reset(s);
        ops::linear_swiglu_a4(x, gu.packed.weight, nullptr, a4, workspace, s);
        ops::linear_add(a4, down.packed.weight, result, s);
    };
    CUDA_CHECK(cudaDeviceSynchronize());
    // Qualify the native baseline against the same complete canonical formula before the candidate.
    std::array<std::array<double, Patterns>, Patterns> ideals{};
    for (int t = 0; t < Patterns; ++t)
        for (int n = 0; n < Patterns; ++n)
            for (int d = 0; d < K; ++d)
                ideals[t][n] += represented(canonical, t * K + d) * stored(down, n, d);
    auto check_output = [&](const auto& projection, const char* name) {
        const auto actual = ninfer::test::from_device_bf16(result.data, initial.size());
        std::vector<double> ref(actual.size());
        for (int t = 0; t < T; ++t)
            for (int n = 0; n < N; ++n)
                ref[static_cast<std::size_t>(t) * N + n] =
                    projection[t % Patterns][n % Patterns] +
                    ninfer::test::bf16_to_f32(initial[static_cast<std::size_t>(t) * N + n]);
        checked(name, actual, ref, {.16, 1.0 / 256, .16});
    };
    baseline(stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));
    check_output(ideals, "native baseline full FP64 SwiGLU->cast->codec->projection+residual");
    ops::linear_swiglu_a4(x, gu.packed.weight, &normalized, a4, workspace, stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));
    const auto ny =
        ninfer::test::from_device_bf16(normalized.data, static_cast<std::size_t>(K) * T);
    std::vector<double> producer_ref(ny.size());
    for (int t = 0; t < T; ++t)
        std::copy_n(producer_ideal.data() + static_cast<std::size_t>(t % Patterns) * K, K,
                    producer_ref.data() + static_cast<std::size_t>(t) * K);
    if (T == 1024) {
        checked("SwiGLU primary FP64 unchanged A4 producer criterion", ny, producer_ref,
                {.16, .01, .16});
    } else {
        std::puts("T513 checks complete FP64 closed chain/exact codec/padding only; standalone "
                  "producer primary criterion remains unqualified due to existing input-A4 floor.");
    }
    const std::vector<float> supplementary(ny.begin(),
                                           ny.begin() + static_cast<std::size_t>(K) * Patterns);
    const auto actual_codec = ninfer::test::nvfp4_activation_reference(supplementary, Divisor);
    std::vector<std::uint8_t> got_codes(codes.bytes), got_scales(scales.bytes);
    codes.copy_to_host(got_codes.data(), got_codes.size());
    scales.copy_to_host(got_scales.data(), got_scales.size());
    for (int t = 0; t < T; ++t) {
        for (int d = 0; d < K / 2; ++d)
            require(got_codes[static_cast<std::size_t>(t) * K / 2 + d] ==
                        actual_codec.codes[static_cast<std::size_t>(t % Patterns) * K / 2 + d],
                    "exact supplementary signed codes failed");
        for (int g = 0; g < K / 16; ++g)
            require(got_scales[scale_offset(t, g)] ==
                        actual_codec.scales[static_cast<std::size_t>(t % Patterns) * K / 16 + g],
                    "exact tiled scales failed");
    }
    const int padded_tokens = static_cast<int>((static_cast<std::size_t>(T) + 255) / 256 * 256);
    for (int t = T; t < padded_tokens; ++t)
        for (int g = 0; g < K / 16; ++g)
            require(got_scales[scale_offset(t, g)] == 0, "padded scale must be zero");
    for (std::size_t i = a4.codes.bytes; i < got_codes.size(); ++i)
        require(got_codes[i] == 0xa5, "code guard changed");
    for (std::size_t i = a4.scales.bytes; i < got_scales.size(); ++i)
        require(got_scales[i] == 0xa5, "scale guard changed");
    std::array<std::array<double, Patterns>, Patterns> published{};
    for (int t = 0; t < Patterns; ++t)
        for (int n = 0; n < Patterns; ++n)
            for (int d = 0; d < K; ++d)
                published[t][n] += represented(actual_codec, t * K + d) * stored(down, n, d);
    candidate(stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));
    check_output(ideals, "full FP64 SwiGLU->cast->codec->projection+nonzeroresidual");
    check_output(published, "closed consumer independent exact signed A4/weight projection");
    DecodeGraphDefinition definition;
    definition.capture(stream, [&] { candidate(stream); });
    DecodeGraphExecutable executable;
    executable.instantiate(definition);
    executable.launch(stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));
    check_output(ideals, "graph full FP64 SwiGLU composite");
}
} // namespace swiglu_fixture

int main(int argc, char** argv) {
    try {
        if (argc != 1 && !ninfer::test::sanitizer_scope(argc, argv)) {
            throw std::invalid_argument("expected optional --sanitizer");
        }
        DeviceContext context;
        rms_fixture::run(context.stream);
        sigmoid_fixture::run(context.stream);
        swiglu_fixture::T = 1024;
        swiglu_fixture::run(context.stream);
        swiglu_fixture::T = 513;
        swiglu_fixture::run(context.stream);
        std::puts("A4 qualified full FP64 chains/exact codec/partial padding/guards/graph PASS; "
                  "T513 standalone SwiGLU producer profile not claimed");
        return 0;
    } catch (const std::exception& error) {
        std::fprintf(stderr, "%s\n", error.what());
        return 1;
    }
}
