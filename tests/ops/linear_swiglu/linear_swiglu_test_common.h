#pragma once

#include "core/tensor.h"

#include <cstdint>
#include <span>
#include <string_view>

namespace ninfer::test::linear_swiglu {

enum class ActivationCompute : std::uint8_t {
    A16,
    A4,
};

struct Profile {
    QType qtype;
    std::int32_t gate_up_rows;
    std::int32_t input_rows;
    std::int32_t output_rows;
    std::uint32_t seed;
    ActivationCompute activation_compute;
};

int run_profile(std::string_view label, const Profile& profile,
                std::span<const std::int32_t> token_cases);

// Packed width column 0 vs T=1 A16 decode on the same weights and token-0 activation.
int run_column0_matches_decode(std::string_view label, const Profile& profile,
                               std::span<const std::int32_t> packed_widths);

} // namespace ninfer::test::linear_swiglu
