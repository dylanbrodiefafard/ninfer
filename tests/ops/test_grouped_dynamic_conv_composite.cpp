// Independent qualification of the closed convolution/residual/plain-RMS composition.
#include "ninfer/ops/grouped_dynamic_conv.h"
#include "ninfer/ops/residual_add.h"
#include "ninfer/ops/rmsnorm.h"
#include "core/decode_graph.h"
#include "core/device.h"
#include "ops/sanitizer_scope.h"
#include "ops/op_tester.h"

#include <cmath>
#include <cstdio>
#include <stdexcept>
#include <vector>

namespace {
using namespace ninfer;
constexpr int kD         = 5120;
constexpr int kG         = 320;
constexpr float kEpsilon = 1.0e-6F;
// Complete composition criteria include finish-storage error in the unfused baseline.
constexpr test::ReductionCriterion kNormCriterion{1.85e-3, 1.0e-5, 3.4e-3};

std::vector<std::uint16_t> fixture(std::size_t count, unsigned seed, float scale) {
    std::vector<std::uint16_t> values(count);
    for (auto& value : values) {
        seed  = seed * 1664525U + 1013904223U;
        value = test::f32_to_bf16(scale * (static_cast<float>(seed >> 8) / 8388608.0F - 1.0F));
    }
    return values;
}

double represented(std::uint16_t bits) { return test::bf16_to_f32(bits); }

// Independent binary round-to-nearest-even at the specified observable residual seam.
// Fixtures stay inside the normal finite BF16 range; zero requires no exponent handling.
double residual_round(double value) {
    if (value == 0.0) { return value; }
    int exponent          = 0;
    const double fraction = std::frexp(value, &exponent);
    const double scaled   = std::ldexp(fraction, 8);
    const double lower    = std::floor(scaled);
    const double delta    = scaled - lower;
    const double rounded  = delta < 0.5                    ? lower
                            : delta > 0.5                  ? lower + 1.0
                            : std::fmod(lower, 2.0) == 0.0 ? lower
                                                           : lower + 1.0;
    return std::ldexp(rounded, exponent - 8);
}

test::GuardedDeviceBuffer upload(const std::vector<std::uint16_t>& values) {
    test::GuardedDeviceBuffer result(values.size() * sizeof(std::uint16_t));
    result.copy_from_host(values.data(), result.bytes());
    return result;
}

int run(int batch, int width, bool cancellation, bool composite = true) {
    const std::size_t columns = static_cast<std::size_t>(batch) * width;
    const std::size_t elems   = columns * kD;
    const auto hidden         = fixture(elems, 17U, 1.0F);
    const auto base           = fixture(4 * kD, 31U, 0.5F);
    const auto dynamic        = fixture(columns * 2 * kG, 47U, 0.25F);
    auto residual             = fixture(elems, 67U, 1.0F);
    auto gain                 = fixture(kD, 79U, cancellation ? 16.0F : 1.0F);
    gain[0]                   = test::f32_to_bf16(0.0F);
    gain[1]                   = test::f32_to_bf16(-1.0F);
    std::vector<double> conv(elems), residual_ideal(elems), residual_boundary(elems), norm(elems);
    for (std::size_t col = 0; col < columns; ++col) {
        for (int d = 0; d < kD; ++d) {
            const std::size_t i  = col * kD + d;
            const double current = represented(hidden[i]);
            const double previous =
                col % static_cast<std::size_t>(width) == 0 ? 0.0 : represented(hidden[i - kD]);
            conv[i] =
                (represented(base[2 * kD + d]) + represented(dynamic[col * 2 * kG + d / 16])) *
                    current +
                (represented(base[3 * kD + d]) + represented(dynamic[col * 2 * kG + kG + d / 16])) *
                    previous;
            if (cancellation) { residual[i] = test::f32_to_bf16(static_cast<float>(-conv[i])); }
            residual_ideal[i]    = represented(residual[i]) + conv[i];
            residual_boundary[i] = residual_round(residual_ideal[i]);
        }
        double squares = 0.0;
        for (int d = 0; d < kD; ++d) {
            squares += residual_boundary[col * kD + d] * residual_boundary[col * kD + d];
        }
        const double inverse = 1.0 / std::sqrt(squares / kD + static_cast<double>(kEpsilon));
        for (int d = 0; d < kD; ++d) {
            norm[col * kD + d] = residual_boundary[col * kD + d] * inverse * represented(gain[d]);
        }
    }
    auto h = upload(hidden);
    auto a = upload(base);
    auto f = upload(dynamic);
    auto r = upload(residual);
    auto w = upload(gain);
    test::GuardedDeviceBuffer y(elems * 2);
    test::GuardedDeviceBuffer finished(elems * 2);
    Tensor ht(h.data(), DType::BF16, {kD, width, batch});
    Tensor at(a.data(), DType::BF16, {kD, 2, 2});
    Tensor ft(f.data(), DType::BF16, {kG, 2, width, batch});
    Tensor rt(r.data(), DType::BF16, {kD, width * batch});
    Tensor wt(w.data(), DType::BF16, {kD});
    Tensor yt(y.data(), DType::BF16, {kD, width * batch});
    DeviceContext device;
    const auto body = [&] {
        Tensor residual_batch = rt.view({kD, width, batch});
        Tensor out_batch      = yt.view({kD, width, batch});
        if (composite) {
            ops::grouped_dynamic_conv_finish_residual_rmsnorm(ht, at, ft, residual_batch, wt,
                                                              kEpsilon, out_batch, device.stream);
        } else {
            Tensor finished_batch(finished.data(), DType::BF16, {kD, width, batch});
            ops::grouped_dynamic_conv_finish(ht, at, ft, finished_batch, device.stream);
            Tensor finished_flat = finished_batch.view({kD, width * batch});
            ops::residual_add(finished_flat, rt, device.stream);
            ops::rmsnorm(rt, wt, kEpsilon, false, yt, device.stream);
        }
    };
    // Fixture uploads use the default stream; DeviceContext compute is nonblocking.
    CUDA_CHECK(cudaDeviceSynchronize());
    body();
    CUDA_CHECK(cudaStreamSynchronize(device.stream));
    const auto eager_residual   = test::from_device<std::uint16_t>(r.data(), elems);
    const auto eager_normalized = test::from_device<std::uint16_t>(y.data(), elems);
    DecodeGraphDefinition definition;
    DecodeGraphExecutable executable;
    definition.capture(device.stream, body);
    executable.instantiate(definition);
    int graph_failures = 0;
    for (int replay = 0; replay < 2; ++replay) {
        r.copy_from_host(residual.data(), r.bytes());
        y.fill(0xff);
        CUDA_CHECK(cudaDeviceSynchronize());
        executable.launch(device.stream);
        device.synchronize();
        graph_failures +=
            test::verify_exact("composite graph residual",
                               test::from_device<std::uint16_t>(r.data(), elems), eager_residual);
        graph_failures +=
            test::verify_exact("composite graph normalized",
                               test::from_device<std::uint16_t>(y.data(), elems), eager_normalized);
    }
    const auto residual_got = test::from_device_bf16(r.data(), elems);
    const auto norm_got     = test::from_device_bf16(y.data(), elems);
    char label[96];
    std::snprintf(label, sizeof(label), "B=%d W=%d cancellation=%d composite=%d", batch, width,
                  cancellation, composite);
    int failures                  = graph_failures;
    double maximum_residual_error = 0.0;
    std::vector<double> envelope_squared(columns, 0.0);
    constexpr double u = 3.95e-3; // Existing residual BF16 rounding criterion.
    for (std::size_t i = 0; i < elems; ++i) {
        // Propagate the existing finish pointwise criterion through residual storage.
        const double finish_error = 1.0 / 256.0 + 4.0 / 256.0 * std::abs(conv[i]);
        const double allowance    = (1.0 + u) * finish_error + u * std::abs(residual_ideal[i]);
        const double error        = std::abs(residual_got[i] - residual_ideal[i]);
        maximum_residual_error    = std::max(maximum_residual_error, error);
        if (!std::isfinite(residual_got[i]) || error > allowance) { ++failures; }
        const double boundary_error = allowance + u * std::abs(residual_ideal[i]);
        envelope_squared[i / kD] += boundary_error * boundary_error;
    }
    double gain_max = 0.0;
    for (const auto bits : gain) { gain_max = std::max(gain_max, std::abs(represented(bits))); }
    // Along each column's ideal-to-published residual segment, the reverse triangle inequality
    // bounds the smallest radius by max(||ideal||-error_envelope,0). RMS's radial Jacobian has
    // spectral norm no larger than 1/sqrt(radius^2/D+epsilon). Degenerate/cancelled columns use
    // the epsilon-only bound; ordinary columns retain their represented signal conditioning.
    double complete_allowance_squared = 0.0;
    for (std::size_t col = 0; col < columns; ++col) {
        double residual_squared   = 0.0;
        double ideal_norm_squared = 0.0;
        for (int d = 0; d < kD; ++d) {
            const std::size_t i = col * kD + d;
            residual_squared += residual_boundary[i] * residual_boundary[i];
            ideal_norm_squared += norm[i] * norm[i];
        }
        const double error_envelope = std::sqrt(envelope_squared[col]);
        const double radius         = std::max(std::sqrt(residual_squared) - error_envelope, 0.0);
        const double lipschitz      = 1.0 / std::sqrt(radius * radius / kD + kEpsilon);
        const double propagated     = gain_max * error_envelope * lipschitz;
        const double cap = propagated + 0.004 * (std::sqrt(ideal_norm_squared) + propagated);
        complete_allowance_squared += cap * cap;
    }
    double actual_error_squared = 0.0;
    for (std::size_t i = 0; i < elems; ++i) {
        const double error = norm_got[i] - norm[i];
        actual_error_squared += error * error;
        if (!std::isfinite(norm_got[i])) { ++failures; }
    }
    const double complete_allowance = std::sqrt(complete_allowance_squared);
    if (std::sqrt(actual_error_squared) > complete_allowance) { ++failures; }
    // The observable residual is the represented public input to plain RMS. Qualify its
    // complete mathematical relation tightly, independently of the convolution's conditioning.
    std::vector<double> seam_norm(elems);
    for (std::size_t col = 0; col < columns; ++col) {
        double squares = 0.0;
        for (int d = 0; d < kD; ++d) {
            squares += residual_got[col * kD + d] * residual_got[col * kD + d];
        }
        const double inverse = 1.0 / std::sqrt(squares / kD + static_cast<double>(kEpsilon));
        for (int d = 0; d < kD; ++d) {
            seam_norm[col * kD + d] = residual_got[col * kD + d] * inverse * represented(gain[d]);
        }
    }
    failures += test::verify_reduction(label, norm_got, seam_norm, kNormCriterion);
    std::printf("boundary_qualification %s residual_max_abs=%.9g complete_norm_l2_error=%.9g "
                "analytic_l2_cap=%.9g failures=%d\n",
                label, maximum_residual_error, std::sqrt(actual_error_squared), complete_allowance,
                failures);
    failures +=
        test::verify_exact("composite hidden preserved",
                           test::from_device<std::uint16_t>(h.data(), hidden.size()), hidden);
    failures += test::verify_exact("composite base preserved",
                                   test::from_device<std::uint16_t>(a.data(), base.size()), base);
    failures +=
        test::verify_exact("composite dynamic preserved",
                           test::from_device<std::uint16_t>(f.data(), dynamic.size()), dynamic);
    failures += test::verify_exact("composite gain preserved",
                                   test::from_device<std::uint16_t>(w.data(), gain.size()), gain);
    for (const auto* buffer : {&h, &a, &f, &r, &w, &y, &finished}) {
        failures += buffer->verify_guards("composite guards");
    }

    if (composite && batch == 1 && width == 1 && !cancellation) {
        Tensor residual_batch = rt.view({kD, width, batch});
        Tensor output_batch   = yt.view({kD, width, batch});
        Tensor overlapping    = ht;
        overlapping.data      = static_cast<std::byte*>(ht.data) + 4;
        failures += test::expect_invalid_argument(
            [&] {
                ops::grouped_dynamic_conv_finish_residual_rmsnorm(
                    ht, at, ft, overlapping, wt, kEpsilon, output_batch, device.stream);
            },
            "composite accepted partially overlapping residual");
        failures += test::expect_invalid_argument(
            [&] {
                ops::grouped_dynamic_conv_finish_residual_rmsnorm(
                    ht, at, ft, residual_batch, wt, 0.0F, output_batch, device.stream);
            },
            "composite accepted zero epsilon");
    }
    return failures;
}
} // namespace

int main(int argc, char** argv) {
    try {
        if (test::sanitizer_scope(argc, argv)) { return run(6, 12, false); }
        int failures = 0;
        for (const int batch : {1, 2, 6, 8}) {
            for (const int width : {1, 3, 8, 12, 16}) {
                for (const bool composite : {false, true}) {
                    failures += run(batch, width, false, composite);
                    failures += run(batch, width, true, composite);
                }
            }
        }
        failures += run(1, 32, false, false);
        failures += run(1, 32, false, true);
        return failures == 0 ? 0 : 1;
    } catch (const std::exception& error) {
        std::fprintf(stderr, "FAIL: %s\n", error.what());
        return 1;
    }
}
