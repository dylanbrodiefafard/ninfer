#pragma once

#include "ninfer/ops/linear.h"

#include <cstdint>
#include <span>
#include <string_view>
#include <vector>

namespace ninfer::test::linear {

enum class ActivationCompute : std::uint8_t {
    A16,
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

struct SimulatedLinearWeight {
    QType qtype                      = QType::Q4G64_F16S;
    std::int32_t n                   = 0;
    std::int32_t k                   = 0;
    std::int32_t padded_k            = 0;
    std::int32_t group_size          = 0;
    std::uint64_t high_plane_offset  = 0;
    std::uint64_t high_plane_bytes   = 0;
    std::uint64_t scale_plane_offset = 0;
    std::vector<std::uint8_t> packed_payload;
    std::vector<std::int32_t> oracle_rows;
    std::vector<float> oracle_weight;

    Weight device_weight(void* device_payload) const;
};

using WeightGenerator = SimulatedLinearWeight (*)(std::int32_t, std::int32_t, std::uint32_t,
                                                  std::span<const std::int32_t>);

SimulatedLinearWeight make_q4g64_f16s_weight(std::int32_t n, std::int32_t k, std::uint32_t seed,
                                             std::span<const std::int32_t> oracle_rows);
SimulatedLinearWeight make_q5g64_f16s_weight(std::int32_t n, std::int32_t k, std::uint32_t seed,
                                             std::span<const std::int32_t> oracle_rows);
SimulatedLinearWeight make_q6g64_f16s_weight(std::int32_t n, std::int32_t k, std::uint32_t seed,
                                             std::span<const std::int32_t> oracle_rows);
SimulatedLinearWeight make_w8g32_f16s_weight(std::int32_t n, std::int32_t k, std::uint32_t seed,
                                             std::span<const std::int32_t> oracle_rows);

void cpu_linear_gemm_fp64(const float* weight, const float* activation, double* output,
                          std::int32_t n, std::int32_t k, std::int32_t t);

bool cuda_available();

int run_shape(std::string_view label, ActivationCompute activation_compute,
              WeightGenerator generator, const ShapeCase& shape);

} // namespace ninfer::test::linear
