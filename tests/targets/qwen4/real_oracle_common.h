#pragma once

#include "core/device.h"
#include "targets/qwen4/verifier.h"

#include <cuda_runtime.h>

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <span>
#include <utility>
#include <vector>

namespace ninfer::test::qwen4::real_oracle {

inline constexpr std::size_t kQ4BlockBytes  = 144;
inline constexpr std::size_t kQ4BlockValues = 256;
inline constexpr std::size_t kQ5BlockBytes  = 176;
inline constexpr std::size_t kQ5BlockValues = 256;
inline constexpr std::size_t kQ8BlockBytes  = 34;
inline constexpr std::size_t kQ8BlockValues = 32;

inline std::uint16_t read_u16(const std::uint8_t* bytes) {
    return static_cast<std::uint16_t>(bytes[0]) |
           (static_cast<std::uint16_t>(bytes[1]) << 8U);
}

inline double binary16_to_double(std::uint16_t word) {
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

inline int signed_i8(std::uint8_t value) {
    return value < 128U ? static_cast<int>(value) : static_cast<int>(value) - 256;
}

inline std::pair<int, int> qk_scale_min(const std::uint8_t* packed, int group) {
    if (group < 4) { return {packed[group] & 63, packed[group + 4] & 63}; }
    return {(packed[group + 4] & 15) | ((packed[group - 4] >> 6U) << 4U),
            (packed[group + 4] >> 4U) | ((packed[group] >> 6U) << 4U)};
}

inline double ggml_q4_k_value(const std::uint8_t* row, std::int32_t column) {
    const auto* block = row + static_cast<std::size_t>(column / kQ4BlockValues) * kQ4BlockBytes;
    const int item = column % static_cast<std::int32_t>(kQ4BlockValues);
    const int group = item / 32;
    const int lane = item % 32;
    const auto [scale, minimum] = qk_scale_min(block + 4, group);
    const int packed = block[16 + 32 * (group / 2) + lane];
    const int code = group % 2 == 0 ? packed & 15 : packed >> 4;
    return binary16_to_double(read_u16(block)) * scale * code -
           binary16_to_double(read_u16(block + 2)) * minimum;
}

inline double ggml_q5_k_value(const std::uint8_t* row, std::int32_t column) {
    const auto* block = row + static_cast<std::size_t>(column / kQ5BlockValues) * kQ5BlockBytes;
    const int item = column % static_cast<std::int32_t>(kQ5BlockValues);
    const int group = item / 32;
    const int lane = item % 32;
    const auto [scale, minimum] = qk_scale_min(block + 4, group);
    const int packed = block[48 + 32 * (group / 2) + lane];
    const int low = group % 2 == 0 ? packed & 15 : packed >> 4;
    const int high = (block[16 + lane] >> group) & 1U;
    const int code = low | (high << 4);
    return binary16_to_double(read_u16(block)) * scale * code -
           binary16_to_double(read_u16(block + 2)) * minimum;
}

inline double ggml_q8_0_value(const std::uint8_t* row, std::int32_t column) {
    const auto* block = row + static_cast<std::size_t>(column / kQ8BlockValues) * kQ8BlockBytes;
    return binary16_to_double(read_u16(block)) * signed_i8(block[2 + column % 32]);
}

inline std::vector<std::uint8_t> copy_device_bytes(const void* device, std::size_t bytes) {
    std::vector<std::uint8_t> result(bytes);
    if (bytes != 0) {
        CUDA_CHECK(cudaMemcpy(result.data(), device, bytes, cudaMemcpyDeviceToHost));
    }
    return result;
}

template <class T>
std::vector<T> copy_device_values(const void* device, std::size_t count) {
    std::vector<T> result(count);
    if (count != 0) {
        CUDA_CHECK(cudaMemcpy(result.data(), device, count * sizeof(T), cudaMemcpyDeviceToHost));
    }
    return result;
}

int run_gr_cell(const targets::qwen4::verifier::LoadedModel& model, DeviceContext& device);
int run_qsa_cell(const targets::qwen4::verifier::LoadedModel& model, DeviceContext& device);
int run_gdn_cell(const targets::qwen4::verifier::LoadedModel& model, DeviceContext& device);
int run_moe_cell(const targets::qwen4::verifier::LoadedModel& model, DeviceContext& device);
int run_ple_cell(const targets::qwen4::verifier::LoadedModel& model, DeviceContext& device);

} // namespace ninfer::test::qwen4::real_oracle
