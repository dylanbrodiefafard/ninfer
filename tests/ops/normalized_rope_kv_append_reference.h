#pragma once

#include "ninfer/ops/normalized_rope_kv_append.h"
#include "ops/op_tester.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <string>
#include <vector>

namespace ninfer::test::normalized_append {

inline constexpr int kHeadDim   = 128;
inline constexpr int kHeads     = 8;
inline constexpr int kCapacity  = 2048;
inline constexpr int kLanes     = 6;
inline constexpr float kEpsilon = 1.0e-6F;
inline constexpr float kTheta   = 1.0e7F;

// The combined norm/rotation BF16 profile permits two unit-roundoff allowances, 2*2^-8,
// plus margin for FP32 norm and trigonometric arithmetic. Pair scaling remains meaningful at
// cancellation; the absolute floor covers only BF16 subnormal rounding. This output criterion
// does not prescribe an intermediate representation or cast.
inline constexpr double kCombinedPairRtol = 8.0e-3;
inline constexpr double kSubnormalAtol    = 1.0e-40;

enum class InputProfile { Varied, Tiny, Cancellation };

inline std::size_t input_index(int batch, int token, int head, int dim, int tokens) {
    return ((static_cast<std::size_t>(batch) * tokens + token) * kHeads + head) * kHeadDim + dim;
}

inline std::size_t cache_index(int lane, int position, int head, int dim) {
    return ((static_cast<std::size_t>(lane) * kHeads + head) * kCapacity +
            static_cast<std::size_t>(position % kCapacity)) *
               kHeadDim +
           dim;
}

struct Oracle {
    std::vector<double> value;
    std::vector<double> pair_scale;
};

// Complete formula from represented inputs: no production coefficient table, range reduction,
// reduction tree, output quantization, or intermediate BF16 materialization enters this oracle.
inline Oracle oracle(const std::vector<std::uint16_t>& raw, const std::vector<std::uint16_t>& gamma,
                     const std::vector<std::int32_t>& positions, int tokens, int batch,
                     float epsilon, float theta) {
    Oracle result{std::vector<double>(raw.size()), std::vector<double>(raw.size())};
    for (int b = 0; b < batch; ++b) {
        for (int t = 0; t < tokens; ++t) {
            for (int h = 0; h < kHeads; ++h) {
                double square_sum = 0.0;
                for (int d = 0; d < kHeadDim; ++d) {
                    const double x = bf16_to_f32(raw[input_index(b, t, h, d, tokens)]);
                    square_sum += x * x;
                }
                const double inv = 1.0 / std::sqrt(square_sum / kHeadDim + epsilon);
                for (int j = 0; j < kHeadDim / 2; ++j) {
                    const std::size_t lo = input_index(b, t, h, j, tokens);
                    const std::size_t hi = input_index(b, t, h, j + kHeadDim / 2, tokens);
                    const double x       = bf16_to_f32(raw[lo]) * inv * bf16_to_f32(gamma[j]);
                    const double y =
                        bf16_to_f32(raw[hi]) * inv * bf16_to_f32(gamma[j + kHeadDim / 2]);
                    const double phase = positions[static_cast<std::size_t>(b) * tokens + t] *
                                         std::pow(static_cast<double>(theta), -2.0 * j / kHeadDim);
                    const double c     = std::cos(phase);
                    const double s     = std::sin(phase);
                    result.value[lo]   = x * c - y * s;
                    result.value[hi]   = y * c + x * s;
                    result.pair_scale[lo] = std::hypot(x, y);
                    result.pair_scale[hi] = result.pair_scale[lo];
                }
            }
        }
    }
    return result;
}

class Fixture {
public:
    Fixture(int tokens, int batch, int first_position, InputProfile profile = InputProfile::Varied,
            float epsilon = kEpsilon, float theta = kTheta)
        : tokens_(tokens), batch_(batch), epsilon_(epsilon), theta_(theta),
          raw_(static_cast<std::size_t>(tokens) * batch * kHeads * kHeadDim), values_(raw_.size()),
          gamma_(kHeadDim), positions_(static_cast<std::size_t>(tokens) * batch), counts_(batch),
          lanes_(batch),
          initial_k_(static_cast<std::size_t>(kHeadDim) * kCapacity * kHeads * kLanes, 0x7fc1),
          initial_v_(initial_k_.size(), 0x7fc2), d_raw_(raw_.size() * 2),
          d_values_(values_.size() * 2), d_gamma_(gamma_.size() * 2),
          d_positions_(positions_.size() * 4), d_counts_(counts_.size() * 4),
          d_lanes_(lanes_.size() * 4), cache_k_(initial_k_.size() * 2),
          cache_v_(initial_v_.size() * 2) {
        for (int d = 0; d < kHeadDim; ++d) {
            gamma_[static_cast<std::size_t>(d)] =
                f32_to_bf16(0.25F + static_cast<float>(d % 17) / 8.0F);
        }
        for (std::size_t i = 0; i < raw_.size(); ++i) {
            const float varied =
                static_cast<float>(static_cast<int>((i * 37 + 11) % 251) - 125) / 31.0F;
            raw_[i] = f32_to_bf16(profile == InputProfile::Tiny ? varied * 1.0e-20F : varied);
            // V is an exact-copy payload, including BF16 infinities, NaNs, and signed zeros.
            values_[i] = static_cast<std::uint16_t>((i * 7919 + 17) & 0xffffu);
        }
        if (profile == InputProfile::Cancellation) {
            for (int b = 0; b < batch; ++b) {
                for (int t = 0; t < tokens; ++t) {
                    for (int h = 0; h < kHeads; ++h) {
                        raw_[input_index(b, t, h, 0, tokens)]  = f32_to_bf16(1.0F);
                        raw_[input_index(b, t, h, 64, tokens)] = f32_to_bf16(0.0859375F);
                    }
                }
            }
        }
        for (int b = 0; b < batch; ++b) {
            lanes_[static_cast<std::size_t>(b)]  = kLanes - b - 1;
            counts_[static_cast<std::size_t>(b)] = tokens;
            for (int t = 0; t < tokens; ++t) {
                positions_[static_cast<std::size_t>(b) * tokens + t] = first_position + b * 313 + t;
            }
        }
        d_raw_.copy_from_host(raw_.data(), d_raw_.bytes);
        d_values_.copy_from_host(values_.data(), d_values_.bytes);
        d_gamma_.copy_from_host(gamma_.data(), d_gamma_.bytes);
        d_positions_.copy_from_host(positions_.data(), d_positions_.bytes);
        reset(tokens, false);
    }

    void reset(int committed, bool rotate_lanes) {
        for (int b = 0; b < batch_; ++b) {
            counts_[static_cast<std::size_t>(b)] = committed == -1 ? b % (tokens_ + 1) : committed;
            lanes_[static_cast<std::size_t>(b)]  = rotate_lanes ? b : kLanes - b - 1;
        }
        d_counts_.copy_from_host(counts_.data(), d_counts_.bytes);
        d_lanes_.copy_from_host(lanes_.data(), d_lanes_.bytes);
        cache_k_.copy_from_host(initial_k_.data(), cache_k_.bytes());
        cache_v_.copy_from_host(initial_v_.data(), cache_v_.bytes());
        // Pageable H2D copies can return before DMA completes on the default stream. The
        // independent execution stream must observe every upload before timing or graph replay.
        cuda_synchronize(nullptr);
    }

    void launch(cudaStream_t stream, bool exact_envelope = false) {
        Tensor k(d_raw_.p, DType::BF16, {kHeadDim, kHeads, tokens_, batch_});
        Tensor v(d_values_.p, DType::BF16, {kHeadDim, kHeads, tokens_, batch_});
        Tensor gamma(d_gamma_.p, DType::BF16, {kHeadDim});
        Tensor positions(d_positions_.p, DType::I32, {tokens_, batch_});
        Tensor counts(d_counts_.p, DType::I32, {batch_});
        Tensor lanes(d_lanes_.p, DType::I32, {batch_});
        const auto max_count = static_cast<std::uint32_t>(tokens_);
        ops::normalized_rope_kv_append(k, v, gamma, positions, counts, lanes, epsilon_, theta_,
                                       {exact_envelope ? max_count : 0, max_count}, cache(),
                                       stream);
    }

    int verify(const std::string& label) const {
        const Oracle expected = oracle(raw_, gamma_, positions_, tokens_, batch_, epsilon_, theta_);
        const auto got_k      = from_device<std::uint16_t>(cache_k_.data(), initial_k_.size());
        const auto got_v      = from_device<std::uint16_t>(cache_v_.data(), initial_v_.size());
        auto expected_v       = initial_v_;
        std::vector<bool> written(initial_k_.size(), false);
        int failures              = 0;
        double required_pair_rtol = 0.0;
        for (int b = 0; b < batch_; ++b) {
            for (int t = 0; t < counts_[static_cast<std::size_t>(b)]; ++t) {
                for (int h = 0; h < kHeads; ++h) {
                    for (int d = 0; d < kHeadDim; ++d) {
                        const std::size_t src = input_index(b, t, h, d, tokens_);
                        const std::size_t dst = cache_index(
                            lanes_[static_cast<std::size_t>(b)],
                            positions_[static_cast<std::size_t>(b) * tokens_ + t], h, d);
                        written[dst]       = true;
                        expected_v[dst]    = values_[src];
                        const double value = bf16_to_f32(got_k[dst]);
                        const double error = std::abs(value - expected.value[src]);
                        const double scale = expected.pair_scale[src];
                        required_pair_rtol =
                            std::max(required_pair_rtol, error / std::max(scale, kSubnormalAtol));
                        if (!std::isfinite(value) ||
                            error > kCombinedPairRtol * scale + kSubnormalAtol) {
                            if (failures < 5) {
                                std::cerr << label << " K src=" << src << " got=" << value
                                          << " expected=" << expected.value[src] << '\n';
                            }
                            ++failures;
                        }
                    }
                }
            }
        }
        for (std::size_t i = 0; i < written.size(); ++i) {
            if (!written[i] && got_k[i] != initial_k_[i]) { ++failures; }
        }
        failures += verify_exact((label + " V exact").c_str(), got_v, expected_v);
        failures += verify_exact((label + " K input").c_str(),
                                 from_device<std::uint16_t>(d_raw_, raw_.size()), raw_);
        failures += verify_exact((label + " V input").c_str(),
                                 from_device<std::uint16_t>(d_values_, values_.size()), values_);
        failures += verify_exact((label + " gamma").c_str(),
                                 from_device<std::uint16_t>(d_gamma_, gamma_.size()), gamma_);
        failures +=
            verify_exact((label + " positions").c_str(),
                         from_device<std::int32_t>(d_positions_, positions_.size()), positions_);
        failures += verify_exact((label + " counts").c_str(),
                                 from_device<std::int32_t>(d_counts_, counts_.size()), counts_);
        failures += verify_exact((label + " lanes").c_str(),
                                 from_device<std::int32_t>(d_lanes_, lanes_.size()), lanes_);
        failures += cache_k_.verify_guards((label + " K guards").c_str());
        failures += cache_v_.verify_guards((label + " V guards").c_str());
        std::cout << label << " required_pair_rtol=" << required_pair_rtol
                  << " failures=" << failures << '\n';
        return failures;
    }

    [[nodiscard]] CyclicKVCacheLayerView cache() {
        return {.k = Tensor(cache_k_.data(), DType::BF16, {kHeadDim, kCapacity, kHeads, kLanes}),
                .v = Tensor(cache_v_.data(), DType::BF16, {kHeadDim, kCapacity, kHeads, kLanes}),
                .capacity        = kCapacity,
                .padded_capacity = kCapacity,
                .num_kv_heads    = kHeads,
                .head_dim        = kHeadDim,
                .lane_capacity   = kLanes};
    }

private:
    int tokens_;
    int batch_;
    float epsilon_;
    float theta_;
    std::vector<std::uint16_t> raw_, values_, gamma_;
    std::vector<std::int32_t> positions_, counts_, lanes_;
    std::vector<std::uint16_t> initial_k_, initial_v_;
    DeviceBuffer d_raw_, d_values_, d_gamma_, d_positions_, d_counts_, d_lanes_;
    GuardedDeviceBuffer cache_k_, cache_v_;
};

} // namespace ninfer::test::normalized_append
