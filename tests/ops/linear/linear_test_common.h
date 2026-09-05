#pragma once

#include "ninfer/ops/linear.h"
#include "ops/quantized_weight.h"

#include <cstdint>
#include <span>
#include <string_view>
#include <vector>

namespace ninfer::test::linear {

enum class ActivationCompute : std::uint8_t {
    A16,
    A4,
};

enum class CallForm : std::uint8_t {
    Policy,
    A16Convenience,
};

enum class Comparison : std::uint8_t {
    Full,
    Sampled,
};

struct Invocation {
    std::int32_t t;
    CallForm call_form       = CallForm::Policy;
    ops::LinearPolicy policy = ops::LinearPolicy::A16Only;
};

struct ShapeCase {
    std::int32_t n;
    std::int32_t k;
    std::uint32_t seed;
    Comparison comparison;
    bool verify_input_preservation;
    std::span<const Invocation> invocations;
};

using WeightGenerator = quantized_weight::PackedWeight (*)(std::int32_t, std::int32_t,
                                                           std::uint32_t);

quantized_weight::PackedWeight make_q4g64_f16s_weight(std::int32_t n, std::int32_t k,
                                                      std::uint32_t seed);
quantized_weight::PackedWeight make_q5g64_f16s_weight(std::int32_t n, std::int32_t k,
                                                      std::uint32_t seed);
quantized_weight::PackedWeight make_q6g64_f16s_weight(std::int32_t n, std::int32_t k,
                                                      std::uint32_t seed);
quantized_weight::PackedWeight make_w8g32_f16s_weight(std::int32_t n, std::int32_t k,
                                                      std::uint32_t seed);
quantized_weight::PackedWeight make_nvfp4_weight(std::int32_t n, std::int32_t k,
                                                 std::uint32_t seed);

void cpu_linear_gemm_fp64(const float* weight, const float* activation, double* output,
                          std::int32_t n, std::int32_t k, std::int32_t t);

bool cuda_available();

int run_shape(std::string_view label, ActivationCompute activation_compute,
              WeightGenerator generator, const ShapeCase& shape);

int run_packed_column0_matches_decode(std::string_view label, WeightGenerator generator,
                                      std::int32_t n, std::int32_t k, std::uint32_t seed,
                                      ops::LinearPolicy packed_policy,
                                      std::span<const std::int32_t> packed_widths);

int run_packed_sequences_matches_panels(std::string_view label, WeightGenerator generator,
                                        std::int32_t n, std::int32_t k, std::uint32_t seed,
                                        std::int32_t sequence_width,
                                        std::span<const std::int32_t> batch_sizes,
                                        ops::LinearPolicy policy = ops::LinearPolicy::A16Only,
                                        bool verify_convenience = true);

} // namespace ninfer::test::linear
