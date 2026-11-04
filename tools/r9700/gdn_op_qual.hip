#include <hip/hip_runtime.h>

#include "ninfer/ops/causal_conv1d_silu.h"
#include "ninfer/ops/scatter.h"
#include "ninfer/types.h"
#include "ops/r9700/gdn/gdn_ops.h"

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <limits>
#include <numbers>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace {

constexpr std::uint32_t kChannels   = 10240;
constexpr std::uint32_t kKeyDim     = 128;
constexpr std::uint32_t kValueDim   = 128;
constexpr std::uint32_t kQkHeads    = 16;
constexpr std::uint32_t kValueHeads = 48;
constexpr float kEpsilon            = 1.0e-6F;
constexpr float kScale              = 0.08838834764831845F; // 1/sqrt(128)

[[noreturn]] void fail(const std::string& message) { throw std::runtime_error(message); }

void hip_check(hipError_t status, const char* expression, const char* file, int line) {
    if (status == hipSuccess) { return; }
    std::ostringstream message;
    message << file << ':' << line << ": " << expression << ": " << hipGetErrorString(status);
    fail(message.str());
}

#define HIP_CHECK(expression) hip_check((expression), #expression, __FILE__, __LINE__)

template <typename T>
class DeviceBuffer {
public:
    explicit DeviceBuffer(std::size_t count) : count_(count) {
        HIP_CHECK(hipMalloc(&data_, count * sizeof(T)));
    }

    DeviceBuffer(const DeviceBuffer&)            = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;

    ~DeviceBuffer() { (void)hipFree(data_); }

    [[nodiscard]] T* get() const noexcept { return data_; }

    [[nodiscard]] std::size_t size() const noexcept { return count_; }

private:
    T* data_           = nullptr;
    std::size_t count_ = 0;
};

std::uint16_t bf16_rne_bits(float value) {
    std::uint32_t word           = std::bit_cast<std::uint32_t>(value);
    const std::uint32_t absolute = word & 0x7fffffffU;
    if ((absolute & 0x7f800000U) != 0x7f800000U) {
        word += 0x7fffU + ((word >> 16U) & 1U);
    } else if ((word & 0xffffU) != 0U) {
        word |= 0x10000U;
    }
    return static_cast<std::uint16_t>(word >> 16U);
}

hip_bfloat16 bf16_from_bits(std::uint16_t bits) {
    hip_bfloat16 result;
    result.data = bits;
    return result;
}

std::uint16_t bf16_bits(hip_bfloat16 value) { return value.data; }

std::vector<hip_bfloat16> bf16_fixture(std::size_t count, std::uint32_t seed,
                                       float divisor = 31.0F) {
    std::vector<hip_bfloat16> result(count);
    for (std::size_t index = 0; index < count; ++index) {
        const auto wave = static_cast<std::int32_t>((index * 37U + seed * 13U) % 509U) - 254;
        result[index]   = bf16_from_bits(bf16_rne_bits(static_cast<float>(wave) / divisor));
    }
    return result;
}

std::vector<float> fp32_fixture(std::size_t count, std::uint32_t seed, float divisor = 53.0F) {
    std::vector<float> result(count);
    for (std::size_t index = 0; index < count; ++index) {
        const auto wave = static_cast<std::int32_t>((index * 41U + seed * 17U) % 431U) - 215;
        result[index]   = static_cast<float>(wave) / divisor;
    }
    return result;
}

struct DeviceIdentity {
    int ordinal = -1;
    std::string name;
    std::string architecture;
    int wavefront = 0;
    std::string pci_bus_id;
    int runtime_version = 0;
    int driver_version  = 0;
};

DeviceIdentity select_r9700(bool print_human = true) {
    int devices = 0;
    HIP_CHECK(hipGetDeviceCount(&devices));
    int selected = -1;
    for (int ordinal = 0; ordinal < devices; ++ordinal) {
        hipDeviceProp_t properties{};
        HIP_CHECK(hipGetDeviceProperties(&properties, ordinal));
        if (properties.integrated == 0 &&
            std::string_view(properties.name) == "AMD Radeon AI PRO R9700" &&
            std::string_view(properties.gcnArchName) == "gfx1201") {
            if (selected >= 0) { fail("expected one discrete gfx1201 R9700"); }
            selected = ordinal;
        }
    }
    if (selected < 0) { fail("no discrete gfx1201 R9700 found"); }
    HIP_CHECK(hipSetDevice(selected));
    hipDeviceProp_t properties{};
    HIP_CHECK(hipGetDeviceProperties(&properties, selected));
    if (properties.warpSize != 32) { fail("GDN qualification requires wave32"); }
    char pci_bus_id[32]{};
    int runtime_version = 0;
    int driver_version  = 0;
    HIP_CHECK(hipDeviceGetPCIBusId(pci_bus_id, sizeof(pci_bus_id), selected));
    HIP_CHECK(hipRuntimeGetVersion(&runtime_version));
    HIP_CHECK(hipDriverGetVersion(&driver_version));
    DeviceIdentity identity{
        selected,   properties.name, properties.gcnArchName, properties.warpSize,
        pci_bus_id, runtime_version, driver_version};
    if (print_human) {
        std::cout << "device: name=" << identity.name << " arch=" << identity.architecture
                  << " wave=" << identity.wavefront << " pci=" << identity.pci_bus_id
                  << " runtime=" << identity.runtime_version
                  << " driver=" << identity.driver_version
                  << " vram_mib=" << properties.totalGlobalMem / (1024U * 1024U) << '\n';
    }
    return identity;
}

template <typename T>
void copy_to_device(DeviceBuffer<T>& destination, const std::vector<T>& source) {
    if (destination.size() != source.size()) { fail("host/device extent mismatch"); }
    HIP_CHECK(hipMemcpy(destination.get(), source.data(), source.size() * sizeof(T),
                        hipMemcpyHostToDevice));
}

template <typename T>
std::vector<T> copy_to_host(const DeviceBuffer<T>& source) {
    std::vector<T> result(source.size());
    HIP_CHECK(
        hipMemcpy(result.data(), source.get(), result.size() * sizeof(T), hipMemcpyDeviceToHost));
    return result;
}

void require_bf16_exact(const std::vector<hip_bfloat16>& actual,
                        const std::vector<hip_bfloat16>& expected, const char* label) {
    if (actual.size() != expected.size()) { fail(std::string(label) + " extent mismatch"); }
    for (std::size_t index = 0; index < actual.size(); ++index) {
        if (bf16_bits(actual[index]) != bf16_bits(expected[index])) {
            std::ostringstream message;
            message << label << " BF16 mismatch at " << index;
            fail(message.str());
        }
    }
}

void require_bf16_scatter_exact(const std::vector<hip_bfloat16>& actual,
                                const std::vector<hip_bfloat16>& packed, std::size_t packed_rows,
                                std::size_t packed_offset, std::size_t rows, const char* label) {
    if (actual.size() * packed_rows != packed.size() * rows) {
        fail(std::string(label) + " extent mismatch");
    }
    const std::size_t tokens = actual.size() / rows;
    for (std::size_t token = 0; token < tokens; ++token) {
        for (std::size_t row = 0; row < rows; ++row) {
            if (bf16_bits(actual[token * rows + row]) !=
                bf16_bits(packed[token * packed_rows + packed_offset + row])) {
                std::ostringstream message;
                message << label << " BF16 mismatch at token=" << token << " row=" << row;
                fail(message.str());
            }
        }
    }
}

void require_fp32_exact(const std::vector<float>& actual, const std::vector<float>& expected,
                        const char* label) {
    if (actual.size() != expected.size()) { fail(std::string(label) + " extent mismatch"); }
    for (std::size_t index = 0; index < actual.size(); ++index) {
        if (std::bit_cast<std::uint32_t>(actual[index]) !=
            std::bit_cast<std::uint32_t>(expected[index])) {
            std::ostringstream message;
            message << label << " FP32 mismatch at " << index;
            fail(message.str());
        }
    }
}

template <typename Expected>
void require_close(const std::vector<float>& actual, const std::vector<Expected>& expected,
                   double absolute_limit, double relative_limit, const char* label) {
    if (actual.size() != expected.size()) { fail(std::string(label) + " extent mismatch"); }
    for (std::size_t index = 0; index < actual.size(); ++index) {
        const double observed  = actual[index];
        const double reference = static_cast<double>(expected[index]);
        const double absolute  = std::abs(observed - reference);
        const double relative  = absolute / std::max(1.0e-6, std::abs(reference));
        if (!std::isfinite(observed) || !std::isfinite(reference) ||
            (absolute > absolute_limit && relative > relative_limit)) {
            std::ostringstream message;
            message << label << " mismatch at " << index << " observed=" << observed
                    << " expected=" << reference;
            fail(message.str());
        }
    }
}

template <typename Expected>
void require_bf16_close(const std::vector<hip_bfloat16>& actual,
                        const std::vector<Expected>& expected, double absolute_limit,
                        double relative_limit, const char* label) {
    std::vector<float> promoted(actual.size());
    for (std::size_t index = 0; index < actual.size(); ++index) {
        promoted[index] = static_cast<float>(actual[index]);
    }
    require_close(promoted, expected, absolute_limit, relative_limit, label);
}

template <class Function>
void require_invalid_argument(Function&& function, const char* label) {
    try {
        function();
    } catch (const std::invalid_argument&) { return; }
    fail(std::string(label) + " did not reject its invalid public contract");
}

double softplus_oracle(double value) { return value > 20.0 ? value : std::log1p(std::exp(value)); }

void qualify_conv_and_controls(hipStream_t stream) {
    using namespace ninfer::ops::r9700::gdn;
    const std::vector<hip_bfloat16> weight =
        bf16_fixture(static_cast<std::size_t>(kChannels) * 4U, 3U);
    const std::vector<hip_bfloat16> initial_history =
        bf16_fixture(static_cast<std::size_t>(kChannels) * 3U, 7U);
    DeviceBuffer<hip_bfloat16> d_weight(weight.size());
    DeviceBuffer<hip_bfloat16> d_history_in(initial_history.size());
    DeviceBuffer<hip_bfloat16> d_history_out(initial_history.size());
    copy_to_device(d_weight, weight);
    constexpr std::array<std::uint32_t, 18> token_cases{1,  2,  3,  4,  5,  6,  7,   8,    17,
                                                        31, 32, 33, 63, 64, 65, 128, 1025, 2048};
    for (const std::uint32_t tokens : token_cases) {
        const std::size_t elements                 = static_cast<std::size_t>(tokens) * kChannels;
        const std::vector<hip_bfloat16> input      = bf16_fixture(elements, tokens + 11U);
        std::vector<hip_bfloat16> expected_history = initial_history;
        std::vector<double> expected_output(elements);
        for (std::uint32_t channel = 0; channel < kChannels; ++channel) {
            double h0 = static_cast<float>(expected_history[channel]);
            double h1 = static_cast<float>(expected_history[kChannels + channel]);
            double h2 = static_cast<float>(expected_history[2U * kChannels + channel]);
            for (std::uint32_t token = 0; token < tokens; ++token) {
                const double current = static_cast<float>(
                    input[static_cast<std::size_t>(token) * kChannels + channel]);
                const double sum = static_cast<float>(weight[channel]) * h0 +
                                   static_cast<float>(weight[kChannels + channel]) * h1 +
                                   static_cast<float>(weight[2U * kChannels + channel]) * h2 +
                                   static_cast<float>(weight[3U * kChannels + channel]) * current;
                expected_output[static_cast<std::size_t>(token) * kChannels + channel] =
                    sum / (1.0 + std::exp(-sum));
                h0 = h1;
                h1 = h2;
                h2 = current;
            }
            expected_history[channel] = bf16_from_bits(bf16_rne_bits(static_cast<float>(h0)));
            expected_history[kChannels + channel] =
                bf16_from_bits(bf16_rne_bits(static_cast<float>(h1)));
            expected_history[2U * kChannels + channel] =
                bf16_from_bits(bf16_rne_bits(static_cast<float>(h2)));
        }
        DeviceBuffer<hip_bfloat16> d_input(elements);
        DeviceBuffer<hip_bfloat16> d_output(elements);
        copy_to_device(d_input, input);
        copy_to_device(d_history_in, initial_history);
        ninfer::Tensor input_tensor(
            d_input.get(), ninfer::DType::BF16,
            {static_cast<std::int32_t>(kChannels), static_cast<std::int32_t>(tokens)});
        ninfer::Tensor weight_tensor(d_weight.get(), ninfer::DType::BF16,
                                     {static_cast<std::int32_t>(kChannels), 4});
        ninfer::Tensor history_in_tensor(d_history_in.get(), ninfer::DType::BF16,
                                         {static_cast<std::int32_t>(kChannels), 3});
        ninfer::Tensor history_out_tensor(d_history_out.get(), ninfer::DType::BF16,
                                          {static_cast<std::int32_t>(kChannels), 3});
        ninfer::Tensor output_tensor(
            d_output.get(), ninfer::DType::BF16,
            {static_cast<std::int32_t>(kChannels), static_cast<std::int32_t>(tokens)});
        ninfer::ops::causal_conv1d_silu(input_tensor, weight_tensor, history_in_tensor,
                                        history_out_tensor, output_tensor, stream);
        HIP_CHECK(hipStreamSynchronize(stream));
        const std::vector<hip_bfloat16> production_output = copy_to_host(d_output);
        require_bf16_close(production_output, expected_output, 3.0e-2, 3.0e-2,
                           "public causal conv disjoint FP64 oracle");
        require_bf16_exact(copy_to_host(d_history_out), expected_history,
                           "public causal conv disjoint history");
        require_bf16_exact(copy_to_host(d_history_in), initial_history,
                           "public causal conv immutable input history");

        copy_to_device(d_history_in, initial_history);
        HIP_CHECK(causal_conv1d_silu_incumbent_qualification(
            d_input.get(), d_weight.get(), d_history_in.get(), d_history_out.get(), d_output.get(),
            kChannels, tokens, stream));
        HIP_CHECK(hipStreamSynchronize(stream));
        const std::vector<hip_bfloat16> incumbent_output = copy_to_host(d_output);
        require_bf16_exact(incumbent_output, production_output,
                           "production causal conv incumbent parity");
        require_bf16_exact(copy_to_host(d_history_out), expected_history,
                           "incumbent causal conv disjoint history");

        copy_to_device(d_history_in, initial_history);
        ninfer::ops::causal_conv1d_silu(input_tensor, weight_tensor, history_in_tensor,
                                        output_tensor, stream);
        HIP_CHECK(hipStreamSynchronize(stream));
        require_bf16_close(copy_to_host(d_output), expected_output, 3.0e-2, 3.0e-2,
                           "public causal conv in-place FP64 oracle");
        require_bf16_exact(copy_to_host(d_history_in), expected_history,
                           "public causal conv in-place history");

        for (const std::uint32_t tile : {4U, 8U, 16U, 32U}) {
            copy_to_device(d_history_in, initial_history);
            HIP_CHECK(causal_conv1d_silu_prefill_qualification(
                d_input.get(), d_weight.get(), d_history_in.get(), d_history_out.get(),
                d_output.get(), kChannels, tokens, tile, stream));
            HIP_CHECK(hipStreamSynchronize(stream));
            require_bf16_exact(copy_to_host(d_output), incumbent_output,
                               "token-tiled causal conv incumbent parity");
            require_bf16_exact(copy_to_host(d_history_out), expected_history,
                               "token-tiled causal conv disjoint history");
            require_bf16_exact(copy_to_host(d_history_in), initial_history,
                               "token-tiled causal conv immutable input history");

            copy_to_device(d_history_in, initial_history);
            HIP_CHECK(causal_conv1d_silu_prefill_qualification(
                d_input.get(), d_weight.get(), d_history_in.get(), d_history_in.get(),
                d_output.get(), kChannels, tokens, tile, stream));
            HIP_CHECK(hipStreamSynchronize(stream));
            require_bf16_exact(copy_to_host(d_output), incumbent_output,
                               "token-tiled causal conv in-place incumbent parity");
            require_bf16_exact(copy_to_host(d_history_in), expected_history,
                               "token-tiled causal conv in-place history");
        }

        // Direct scatter from separate projection rows, with the value projection either
        // standalone (stride 6144) or the leading half of the [value,z] rows (stride 12288).
        for (const std::uint32_t value_stride : {6144U, 12288U}) {
            constexpr std::size_t kQueryKeyRows = 4096U;
            constexpr std::size_t kValueRows    = 6144U;
            const std::string label = "direct-scatter T" + std::to_string(tokens) + " stride " +
                                      std::to_string(value_stride);
            std::vector<hip_bfloat16> query_key(static_cast<std::size_t>(tokens) * kQueryKeyRows);
            std::vector<hip_bfloat16> projected_value =
                bf16_fixture(static_cast<std::size_t>(tokens) * value_stride, 449U);
            for (std::size_t token = 0; token < tokens; ++token) {
                std::copy_n(input.data() + token * kChannels, kQueryKeyRows,
                            query_key.data() + token * kQueryKeyRows);
                std::copy_n(input.data() + token * kChannels + kQueryKeyRows, kValueRows,
                            projected_value.data() + token * value_stride);
            }
            DeviceBuffer<hip_bfloat16> d_query_key(query_key.size());
            DeviceBuffer<hip_bfloat16> d_projected_value(projected_value.size());
            DeviceBuffer<hip_bfloat16> d_query(static_cast<std::size_t>(tokens) * 2048U);
            DeviceBuffer<hip_bfloat16> d_key(static_cast<std::size_t>(tokens) * 2048U);
            DeviceBuffer<hip_bfloat16> d_value(static_cast<std::size_t>(tokens) * kValueRows);
            copy_to_device(d_query_key, query_key);
            copy_to_device(d_projected_value, projected_value);
            for (const bool in_place : {true, false}) {
                copy_to_device(d_history_in, initial_history);
                hip_bfloat16* history_out = in_place ? d_history_in.get() : d_history_out.get();
                HIP_CHECK(projection_conv_prefill_direct_scatter_bf16(
                    d_query_key.get(), d_projected_value.get(), value_stride, d_weight.get(),
                    d_history_in.get(), history_out, d_query.get(), d_key.get(), d_value.get(),
                    tokens, stream));
                HIP_CHECK(hipStreamSynchronize(stream));
                require_bf16_scatter_exact(copy_to_host(d_query), incumbent_output, kChannels, 0U,
                                           2048U, (label + " query parity").c_str());
                require_bf16_scatter_exact(copy_to_host(d_key), incumbent_output, kChannels, 2048U,
                                           2048U, (label + " key parity").c_str());
                require_bf16_scatter_exact(copy_to_host(d_value), incumbent_output, kChannels,
                                           4096U, kValueRows, (label + " value parity").c_str());
                require_bf16_exact(in_place ? copy_to_host(d_history_in)
                                            : copy_to_host(d_history_out),
                                   expected_history, (label + " history").c_str());
                if (!in_place) {
                    require_bf16_exact(copy_to_host(d_history_in), initial_history,
                                       (label + " immutable input history").c_str());
                }
            }
            if (projection_conv_prefill_direct_scatter_bf16(
                    d_query_key.get(), d_projected_value.get(), value_stride, d_weight.get(),
                    d_history_in.get(), d_history_in.get(), d_query.get(), d_key.get(),
                    d_value.get(), 0U, stream) != hipErrorInvalidValue ||
                projection_conv_prefill_direct_scatter_bf16(
                    d_query_key.get(), d_projected_value.get(), 10240U, d_weight.get(),
                    d_history_in.get(), d_history_in.get(), d_query.get(), d_key.get(),
                    d_value.get(), tokens, stream) != hipErrorInvalidValue ||
                projection_conv_prefill_direct_scatter_bf16(
                    d_query_key.get(), d_projected_value.get(), value_stride, d_weight.get(),
                    d_history_in.get(), d_history_in.get(), d_query_key.get(), d_key.get(),
                    d_value.get(), tokens, stream) != hipErrorInvalidValue) {
                fail(label + " malformed or overlap contract was accepted");
            }
        }
    }

    DeviceBuffer<hip_bfloat16> d_contract_input(2U * kChannels);
    DeviceBuffer<hip_bfloat16> d_contract_output(kChannels);
    DeviceBuffer<hip_bfloat16> d_contract_history(4U * kChannels);
    ninfer::Tensor input_tensor(d_contract_input.get(), ninfer::DType::BF16,
                                {static_cast<std::int32_t>(kChannels), 1});
    ninfer::Tensor weight_tensor(d_weight.get(), ninfer::DType::BF16,
                                 {static_cast<std::int32_t>(kChannels), 4});
    ninfer::Tensor history_in_tensor(d_contract_history.get(), ninfer::DType::BF16,
                                     {static_cast<std::int32_t>(kChannels), 3});
    ninfer::Tensor history_out_tensor(d_history_out.get(), ninfer::DType::BF16,
                                      {static_cast<std::int32_t>(kChannels), 3});
    ninfer::Tensor output_tensor(d_contract_output.get(), ninfer::DType::BF16,
                                 {static_cast<std::int32_t>(kChannels), 1});
    ninfer::Tensor wrong_channels(d_contract_input.get(), ninfer::DType::BF16,
                                  {static_cast<std::int32_t>(kChannels - 1U), 1});
    require_invalid_argument(
        [&] {
            ninfer::ops::causal_conv1d_silu(wrong_channels, weight_tensor, history_in_tensor,
                                            history_out_tensor, output_tensor, stream);
        },
        "causal conv real geometry");
    ninfer::Tensor overlapping_output(d_contract_input.get() + 1, ninfer::DType::BF16,
                                      {static_cast<std::int32_t>(kChannels), 1});
    require_invalid_argument(
        [&] {
            ninfer::ops::causal_conv1d_silu(input_tensor, weight_tensor, history_in_tensor,
                                            history_out_tensor, overlapping_output, stream);
        },
        "causal conv input/output alias");
    ninfer::Tensor overlapping_history(d_contract_history.get() + 1, ninfer::DType::BF16,
                                       {static_cast<std::int32_t>(kChannels), 3});
    require_invalid_argument(
        [&] {
            ninfer::ops::causal_conv1d_silu(input_tensor, weight_tensor, history_in_tensor,
                                            overlapping_history, output_tensor, stream);
        },
        "causal conv partial history alias");
    require_invalid_argument(
        [&] {
            ninfer::ops::causal_conv1d_silu(input_tensor, weight_tensor, history_in_tensor,
                                            history_out_tensor, output_tensor, nullptr);
        },
        "causal conv stream");

    const std::vector<float> a_log   = fp32_fixture(kValueHeads, 19U, 83.0F);
    const std::vector<float> dt_bias = fp32_fixture(kValueHeads, 23U, 7.0F);
    DeviceBuffer<float> d_a_log(a_log.size());
    DeviceBuffer<float> d_dt_bias(dt_bias.size());
    copy_to_device(d_a_log, a_log);
    copy_to_device(d_dt_bias, dt_bias);
    for (std::uint32_t tokens = 1; tokens <= 8; ++tokens) {
        const std::size_t count           = static_cast<std::size_t>(tokens) * kValueHeads;
        std::vector<hip_bfloat16> a       = bf16_fixture(count, tokens + 29U, 3.0F);
        const std::vector<hip_bfloat16> b = bf16_fixture(count, tokens + 37U, 17.0F);
        a[0]                              = bf16_from_bits(bf16_rne_bits(24.0F));
        std::vector<double> g_expected(count);
        std::vector<double> beta_expected(count);
        for (std::size_t index = 0; index < count; ++index) {
            const std::uint32_t head = static_cast<std::uint32_t>(index % kValueHeads);
            const double av          = static_cast<float>(a[index]) + dt_bias[head];
            g_expected[index]        = -std::exp(a_log[head]) * softplus_oracle(av);
            const double bv          = static_cast<float>(b[index]);
            beta_expected[index]     = 1.0 / (1.0 + std::exp(-bv));
        }
        DeviceBuffer<hip_bfloat16> d_a(count);
        DeviceBuffer<hip_bfloat16> d_b(count);
        DeviceBuffer<float> d_g(count);
        DeviceBuffer<float> d_beta(count);
        copy_to_device(d_a, a);
        copy_to_device(d_b, b);
        HIP_CHECK(control_gates_bf16(d_a.get(), d_b.get(), d_a_log.get(), d_dt_bias.get(),
                                     d_g.get(), d_beta.get(), kValueHeads, tokens, stream));
        HIP_CHECK(hipStreamSynchronize(stream));
        require_close(copy_to_host(d_g), g_expected, 3.0e-4, 3.0e-5, "GDN g control FP64 oracle");
        require_close(copy_to_host(d_beta), beta_expected, 3.0e-6, 3.0e-5,
                      "GDN beta control FP64 oracle");
    }
    if (causal_conv1d_silu_bf16(nullptr, d_weight.get(), d_history_in.get(), d_history_out.get(),
                                d_history_out.get(), kChannels, 1,
                                stream) != hipErrorInvalidValue ||
        causal_conv1d_silu_prefill_qualification(
            d_contract_input.get(), d_weight.get(), d_history_in.get(), d_history_out.get(),
            d_contract_output.get(), kChannels, 1, 5, stream) != hipErrorInvalidValue ||
        control_gates_bf16(reinterpret_cast<hip_bfloat16*>(1), nullptr, d_a_log.get(),
                           d_dt_bias.get(), reinterpret_cast<float*>(1),
                           reinterpret_cast<float*>(1), kValueHeads, 1,
                           stream) != hipErrorInvalidValue) {
        fail("conv/control boundary did not reject malformed input");
    }
}

template <typename Launch>
float median_kernel_ms(hipStream_t stream, Launch&& launch) {
    hipEvent_t start = nullptr;
    hipEvent_t stop  = nullptr;
    HIP_CHECK(hipEventCreate(&start));
    HIP_CHECK(hipEventCreate(&stop));
    for (int warmup = 0; warmup < 3; ++warmup) { HIP_CHECK(launch()); }
    HIP_CHECK(hipStreamSynchronize(stream));
    std::array<float, 9> samples{};
    for (float& sample : samples) {
        HIP_CHECK(hipEventRecord(start, stream));
        HIP_CHECK(launch());
        HIP_CHECK(hipEventRecord(stop, stream));
        HIP_CHECK(hipEventSynchronize(stop));
        HIP_CHECK(hipEventElapsedTime(&sample, start, stop));
    }
    HIP_CHECK(hipEventDestroy(start));
    HIP_CHECK(hipEventDestroy(stop));
    std::sort(samples.begin(), samples.end());
    return samples[samples.size() / 2U];
}

void benchmark_conv_prefill_candidates(hipStream_t stream) {
    using namespace ninfer::ops::r9700::gdn;
    constexpr std::uint32_t maximum_tokens = 8192;
    const auto input =
        bf16_fixture(static_cast<std::size_t>(maximum_tokens) * kChannels, 401U, 127.0F);
    const auto weight  = bf16_fixture(static_cast<std::size_t>(kChannels) * 4U, 409U, 127.0F);
    const auto history = bf16_fixture(static_cast<std::size_t>(kChannels) * 3U, 419U, 127.0F);
    DeviceBuffer<hip_bfloat16> d_input(input.size());
    DeviceBuffer<hip_bfloat16> d_weight(weight.size());
    DeviceBuffer<hip_bfloat16> d_history_in(history.size());
    DeviceBuffer<hip_bfloat16> d_history_out(history.size());
    DeviceBuffer<hip_bfloat16> d_output(input.size());
    copy_to_device(d_input, input);
    copy_to_device(d_weight, weight);
    copy_to_device(d_history_in, history);

    std::cout << "route,tokens,tile,median_ms\n";
    constexpr std::array<std::uint32_t, 9> token_cases{16,   32,   64,   128, 256,
                                                       1024, 2048, 4096, 8192};
    for (const std::uint32_t tokens : token_cases) {
        const float incumbent_ms = median_kernel_ms(stream, [&] {
            return causal_conv1d_silu_incumbent_qualification(
                d_input.get(), d_weight.get(), d_history_in.get(), d_history_out.get(),
                d_output.get(), kChannels, tokens, stream);
        });
        std::cout << "serial," << tokens << ",0," << incumbent_ms << '\n';
        for (const std::uint32_t tile : {4U, 8U, 16U, 32U}) {
            const float candidate_ms = median_kernel_ms(stream, [&] {
                return causal_conv1d_silu_prefill_qualification(
                    d_input.get(), d_weight.get(), d_history_in.get(), d_history_out.get(),
                    d_output.get(), kChannels, tokens, tile, stream);
            });
            std::cout << "tiled," << tokens << ',' << tile << ',' << candidate_ms << '\n';
        }
    }
}

void benchmark_projection_conv_direct_scatter(hipStream_t stream, std::uint32_t tokens) {
    using namespace ninfer::ops::r9700::gdn;
    constexpr std::size_t query_key_rows = 4096U;
    constexpr std::size_t value_rows     = 6144U;
    const auto query_key                 = bf16_fixture(tokens * query_key_rows, 431U, 127.0F);
    const auto projected_value           = bf16_fixture(tokens * value_rows, 433U, 127.0F);
    const auto weight  = bf16_fixture(static_cast<std::size_t>(kChannels) * 4U, 439U, 127.0F);
    const auto history = bf16_fixture(static_cast<std::size_t>(kChannels) * 3U, 443U, 127.0F);
    DeviceBuffer<hip_bfloat16> d_query_key(query_key.size());
    DeviceBuffer<hip_bfloat16> d_projected_value(projected_value.size());
    DeviceBuffer<hip_bfloat16> d_weight(weight.size());
    DeviceBuffer<hip_bfloat16> d_history_in(history.size());
    DeviceBuffer<hip_bfloat16> d_history_out(history.size());
    DeviceBuffer<hip_bfloat16> d_qkv(static_cast<std::size_t>(tokens) * kChannels);
    DeviceBuffer<hip_bfloat16> d_qkv_c(static_cast<std::size_t>(tokens) * kChannels);
    DeviceBuffer<hip_bfloat16> d_query(static_cast<std::size_t>(tokens) * 2048U);
    DeviceBuffer<hip_bfloat16> d_key(static_cast<std::size_t>(tokens) * 2048U);
    DeviceBuffer<hip_bfloat16> d_value(static_cast<std::size_t>(tokens) * value_rows);
    copy_to_device(d_query_key, query_key);
    copy_to_device(d_projected_value, projected_value);
    copy_to_device(d_weight, weight);
    copy_to_device(d_history_in, history);

    ninfer::Tensor qkv_c_tensor(
        d_qkv_c.get(), ninfer::DType::BF16,
        {static_cast<std::int32_t>(kChannels), static_cast<std::int32_t>(tokens)});
    ninfer::Tensor query_tensor(d_query.get(), ninfer::DType::BF16,
                                {2048, static_cast<std::int32_t>(tokens)});
    ninfer::Tensor key_tensor(d_key.get(), ninfer::DType::BF16,
                              {2048, static_cast<std::int32_t>(tokens)});
    ninfer::Tensor value_tensor(
        d_value.get(), ninfer::DType::BF16,
        {static_cast<std::int32_t>(value_rows), static_cast<std::int32_t>(tokens)});

    const auto incumbent = [&]() {
        HIP_CHECK(hipMemcpy2DAsync(d_qkv.get(), kChannels * sizeof(hip_bfloat16), d_query_key.get(),
                                   query_key_rows * sizeof(hip_bfloat16),
                                   query_key_rows * sizeof(hip_bfloat16), tokens,
                                   hipMemcpyDeviceToDevice, stream));
        HIP_CHECK(hipMemcpy2DAsync(d_qkv.get() + query_key_rows, kChannels * sizeof(hip_bfloat16),
                                   d_projected_value.get(), value_rows * sizeof(hip_bfloat16),
                                   value_rows * sizeof(hip_bfloat16), tokens,
                                   hipMemcpyDeviceToDevice, stream));
        HIP_CHECK(causal_conv1d_silu_prefill_qualification(
            d_qkv.get(), d_weight.get(), d_history_in.get(), d_history_out.get(), d_qkv_c.get(),
            kChannels, tokens, 4U, stream));
        ninfer::ops::extract_bf16_columns(qkv_c_tensor, 0, query_tensor, stream);
        ninfer::ops::extract_bf16_columns(qkv_c_tensor, 2048, key_tensor, stream);
        ninfer::ops::extract_bf16_columns(qkv_c_tensor, 4096, value_tensor, stream);
        return hipGetLastError();
    };
    const auto challenger = [&]() {
        return projection_conv_prefill_direct_scatter_bf16(
            d_query_key.get(), d_projected_value.get(), value_rows, d_weight.get(),
            d_history_in.get(), d_history_out.get(), d_query.get(), d_key.get(), d_value.get(),
            tokens, stream);
    };
    const float incumbent_ms  = median_kernel_ms(stream, incumbent);
    const float challenger_ms = median_kernel_ms(stream, challenger);
    std::cout << "projection_conv T" << tokens << " incumbent_ms=" << incumbent_ms
              << " direct_scatter_ms=" << challenger_ms
              << " projected_48_call_saving_ms=" << 48.0F * (incumbent_ms - challenger_ms) << '\n';
}

void qualify_batched_projection_conv(hipStream_t stream) {
    using namespace ninfer::ops::r9700::gdn;
    constexpr std::uint32_t query_rows   = 2048;
    constexpr std::uint32_t key_rows     = 2048;
    constexpr std::uint32_t value_rows   = 6144;
    constexpr std::uint32_t value_z_rows = 2U * value_rows;
    constexpr std::uint32_t width        = 4;
    constexpr std::uint32_t batch        = ninfer::kMaximumConcurrency;
    constexpr std::uint32_t columns      = width * batch;
    constexpr std::uint32_t state_slots = batch + batch * width;

    const auto query_key =
        bf16_fixture(static_cast<std::size_t>(columns) * (query_rows + key_rows), 101U, 257.0F);
    const auto value_z =
        bf16_fixture(static_cast<std::size_t>(columns) * value_z_rows, 103U, 257.0F);
    const auto weight = bf16_fixture(static_cast<std::size_t>(kChannels) * 4U, 107U, 1024.0F);
    const auto initial_state =
        bf16_fixture(static_cast<std::size_t>(state_slots) * 3U * kChannels, 109U, 257.0F);
    // Every row's valid extent and tree shape, cycled across the whole compact batch.
    constexpr std::array<std::int32_t, 8> kValid{4, 2, 3, 1, 1, 4, 2, 3};
    constexpr std::array<std::array<std::int32_t, width>, 8> kParents{{{-1, 0, 0, 2},
                                                                       {-1, 0, 0, 0},
                                                                       {-1, 0, 0, 1},
                                                                       {-1, 0, 1, 2},
                                                                       {-1, 0, 1, 1},
                                                                       {-1, 0, 1, 2},
                                                                       {-1, 0, 0, 0},
                                                                       {-1, 0, 0, 1}}};
    std::vector<std::int32_t> valid, initial_slots, snapshot_bases, parents;
    for (std::uint32_t b = 0; b < batch; ++b) {
        valid.push_back(kValid[(b + b / 8U) % kValid.size()]);
        initial_slots.push_back(static_cast<std::int32_t>(b));
        snapshot_bases.push_back(static_cast<std::int32_t>(batch + b * width));
        const auto& row_parents = kParents[(b + 3U * (b / 8U)) % kParents.size()];
        parents.insert(parents.end(), row_parents.begin(), row_parents.end());
    }

    DeviceBuffer<hip_bfloat16> d_query_key(query_key.size());
    DeviceBuffer<hip_bfloat16> d_value_z(value_z.size());
    DeviceBuffer<hip_bfloat16> d_weight(weight.size());
    DeviceBuffer<hip_bfloat16> d_state(initial_state.size());
    DeviceBuffer<std::int32_t> d_valid(valid.size());
    DeviceBuffer<std::int32_t> d_initial(initial_slots.size());
    DeviceBuffer<std::int32_t> d_bases(snapshot_bases.size());
    DeviceBuffer<std::int32_t> d_parents(parents.size());
    DeviceBuffer<hip_bfloat16> d_query(static_cast<std::size_t>(columns) * query_rows);
    DeviceBuffer<hip_bfloat16> d_key(static_cast<std::size_t>(columns) * key_rows);
    DeviceBuffer<hip_bfloat16> d_value(static_cast<std::size_t>(columns) * value_rows);
    DeviceBuffer<hip_bfloat16> d_z(static_cast<std::size_t>(columns) * value_rows);
    DeviceBuffer<hip_bfloat16> d_record(static_cast<std::size_t>(columns) * kChannels);
    copy_to_device(d_query_key, query_key);
    copy_to_device(d_value_z, value_z);
    copy_to_device(d_weight, weight);
    copy_to_device(d_valid, valid);
    copy_to_device(d_initial, initial_slots);
    copy_to_device(d_bases, snapshot_bases);
    copy_to_device(d_parents, parents);

    const auto projected = [&](std::size_t column, std::uint32_t channel) {
        return channel < query_rows + key_rows
                   ? query_key[column * (query_rows + key_rows) + channel]
                   : value_z[column * value_z_rows + channel - query_rows - key_rows];
    };
    const auto output_index = [&](std::size_t column, std::uint32_t channel) {
        if (channel < query_rows) { return std::pair{0U, column * query_rows + channel}; }
        if (channel < query_rows + key_rows) {
            return std::pair{1U, column * key_rows + channel - query_rows};
        }
        return std::pair{2U, column * value_rows + channel - query_rows - key_rows};
    };
    const auto conv_value = [&](std::uint32_t channel, float h0, float h1, float h2,
                                float current) {
        float sum = std::fma(static_cast<float>(weight[channel]), h0, 0.0F);
        sum       = std::fma(static_cast<float>(weight[kChannels + channel]), h1, sum);
        sum       = std::fma(static_cast<float>(weight[2U * kChannels + channel]), h2, sum);
        sum       = std::fma(static_cast<float>(weight[3U * kChannels + channel]), current, sum);
        return static_cast<double>(sum) / (1.0 + std::exp(-static_cast<double>(sum)));
    };

    std::vector<double> expected_query(static_cast<std::size_t>(columns) * query_rows, 0.0);
    std::vector<double> expected_key(static_cast<std::size_t>(columns) * key_rows, 0.0);
    std::vector<double> expected_value(static_cast<std::size_t>(columns) * value_rows, 0.0);
    std::vector<hip_bfloat16> expected_z(static_cast<std::size_t>(columns) * value_rows);
    std::vector<hip_bfloat16> expected_snapshot = initial_state;
    for (std::uint32_t b = 0; b < batch; ++b) {
        for (std::uint32_t channel = 0; channel < kChannels; ++channel) {
            const std::size_t initial_base =
                static_cast<std::size_t>(initial_slots[b]) * 3U * kChannels;
            float h0 = static_cast<float>(initial_state[initial_base + channel]);
            float h1 = static_cast<float>(initial_state[initial_base + kChannels + channel]);
            float h2 = static_cast<float>(initial_state[initial_base + 2U * kChannels + channel]);
            for (std::uint32_t token = 0; token < width; ++token) {
                const std::size_t column = static_cast<std::size_t>(b) * width + token;
                if (channel < value_rows) {
                    expected_z[column * value_rows + channel] =
                        value_z[column * value_z_rows + value_rows + channel];
                }
                if (token >= static_cast<std::uint32_t>(valid[b])) { continue; }
                const float current       = static_cast<float>(projected(column, channel));
                const auto [plane, index] = output_index(column, channel);
                const double result       = conv_value(channel, h0, h1, h2, current);
                if (plane == 0U)
                    expected_query[index] = result;
                else if (plane == 1U)
                    expected_key[index] = result;
                else
                    expected_value[index] = result;
                h0 = h1;
                h1 = h2;
                h2 = current;
                const std::size_t base =
                    (static_cast<std::size_t>(snapshot_bases[b]) + token) * 3U * kChannels;
                expected_snapshot[base + channel]             = bf16_from_bits(bf16_rne_bits(h0));
                expected_snapshot[base + kChannels + channel] = bf16_from_bits(bf16_rne_bits(h1));
                expected_snapshot[base + 2U * kChannels + channel] =
                    bf16_from_bits(bf16_rne_bits(h2));
            }
        }
    }

    copy_to_device(d_state, initial_state);
    HIP_CHECK(projection_conv_snapshot_bf16(
        d_query_key.get(), d_value_z.get(), d_weight.get(), d_state.get(), d_valid.get(),
        d_initial.get(), d_bases.get(), d_query.get(), d_key.get(), d_value.get(), d_z.get(), width,
        batch, state_slots, stream));
    HIP_CHECK(hipStreamSynchronize(stream));
    require_bf16_close(copy_to_host(d_query), expected_query, 3.0e-2, 3.0e-2,
                       "batched snapshot query");
    require_bf16_close(copy_to_host(d_key), expected_key, 3.0e-2, 3.0e-2, "batched snapshot key");
    require_bf16_close(copy_to_host(d_value), expected_value, 3.0e-2, 3.0e-2,
                       "batched snapshot value");
    require_bf16_exact(copy_to_host(d_z), expected_z, "batched snapshot z");
    require_bf16_exact(copy_to_host(d_state), expected_snapshot, "batched snapshot states");

    std::fill(expected_query.begin(), expected_query.end(), 0.0);
    std::fill(expected_key.begin(), expected_key.end(), 0.0);
    std::fill(expected_value.begin(), expected_value.end(), 0.0);
    std::vector<hip_bfloat16> expected_record(static_cast<std::size_t>(columns) * kChannels,
                                              bf16_from_bits(0));
    for (std::uint32_t b = 0; b < batch; ++b) {
        for (std::uint32_t channel = 0; channel < kChannels; ++channel) {
            const std::size_t initial_base =
                static_cast<std::size_t>(initial_slots[b]) * 3U * kChannels;
            const float checkpoint0 = static_cast<float>(initial_state[initial_base + channel]);
            const float checkpoint1 =
                static_cast<float>(initial_state[initial_base + kChannels + channel]);
            const float checkpoint2 =
                static_cast<float>(initial_state[initial_base + 2U * kChannels + channel]);
            std::array<float, width> saved0{};
            std::array<float, width> saved1{};
            std::array<float, width> saved2{};
            for (std::uint32_t token = 0; token < width; ++token) {
                if (token >= static_cast<std::uint32_t>(valid[b])) { continue; }
                const std::size_t column       = static_cast<std::size_t>(b) * width + token;
                const std::int32_t parent      = parents[column];
                const float h0                 = parent < 0 ? checkpoint0 : saved0[parent];
                const float h1                 = parent < 0 ? checkpoint1 : saved1[parent];
                const float h2                 = parent < 0 ? checkpoint2 : saved2[parent];
                const hip_bfloat16 represented = projected(column, channel);
                expected_record[column * kChannels + channel] = represented;
                const float current                           = static_cast<float>(represented);
                const auto [plane, index]                     = output_index(column, channel);
                const double result = conv_value(channel, h0, h1, h2, current);
                if (plane == 0U)
                    expected_query[index] = result;
                else if (plane == 1U)
                    expected_key[index] = result;
                else
                    expected_value[index] = result;
                saved0[token] = h1;
                saved1[token] = h2;
                saved2[token] = current;
            }
        }
    }
    copy_to_device(d_state, initial_state);
    HIP_CHECK(projection_conv_record_bf16(
        d_query_key.get(), d_value_z.get(), d_weight.get(), d_state.get(), d_valid.get(),
        d_initial.get(), d_parents.get(), d_record.get(), d_query.get(), d_key.get(), d_value.get(),
        d_z.get(), width, batch, state_slots, stream));
    HIP_CHECK(hipStreamSynchronize(stream));
    require_bf16_close(copy_to_host(d_query), expected_query, 3.0e-2, 3.0e-2, "tree record query");
    require_bf16_close(copy_to_host(d_key), expected_key, 3.0e-2, 3.0e-2, "tree record key");
    require_bf16_close(copy_to_host(d_value), expected_value, 3.0e-2, 3.0e-2, "tree record value");
    require_bf16_exact(copy_to_host(d_z), expected_z, "tree record z");
    require_bf16_exact(copy_to_host(d_record), expected_record, "tree projection record");
    require_bf16_exact(copy_to_host(d_state), initial_state, "tree checkpoint immutability");

    if (projection_conv_snapshot_bf16(nullptr, d_value_z.get(), d_weight.get(), d_state.get(),
                                      d_valid.get(), d_initial.get(), d_bases.get(), d_query.get(),
                                      d_key.get(), d_value.get(), d_z.get(), width, batch,
                                      state_slots, stream) != hipErrorInvalidValue ||
        projection_conv_record_bf16(
            d_query_key.get(), d_value_z.get(), d_weight.get(), d_state.get(), d_valid.get(),
            d_initial.get(), nullptr, d_record.get(), d_query.get(), d_key.get(), d_value.get(),
            d_z.get(), 1, batch, state_slots, stream) != hipErrorInvalidValue) {
        fail("batched projection-convolution boundary did not reject malformed input");
    }
}

// A mixed round's single convolution launch: bitwise the owner's direct scatter followed by the
// verify batch's record over combined [value, z] rows, from identically poisoned outputs.
void qualify_mixed_projection_conv(hipStream_t stream) {
    using namespace ninfer::ops::r9700::gdn;
    constexpr std::uint32_t query_key_rows = 4096;
    constexpr std::uint32_t value_rows     = 6144;
    constexpr std::uint32_t state_slots    = 8;
    const auto weight = bf16_fixture(static_cast<std::size_t>(kChannels) * 4U, 211U, 1024.0F);
    const auto states =
        bf16_fixture(static_cast<std::size_t>(state_slots) * 3U * kChannels, 223U, 257.0F);
    const auto history = bf16_fixture(static_cast<std::size_t>(kChannels) * 3U, 227U, 257.0F);
    DeviceBuffer<hip_bfloat16> d_weight(weight.size());
    DeviceBuffer<hip_bfloat16> d_states(states.size());
    DeviceBuffer<hip_bfloat16> d_history(history.size());
    copy_to_device(d_weight, weight);
    copy_to_device(d_states, states);

    struct Case {
        std::uint32_t owner, width, batch;
        std::vector<std::int32_t> valid, slots;
    };

    const std::vector<Case> cases{{1U, 8U, 1U, {8}, {5}},
                                  {7U, 8U, 3U, {8, 5, 1}, {5, 6, 7}},
                                  {1000U, 8U, 4U, {8, 8, 3, 2}, {4, 5, 6, 7}},
                                  {2040U, 2U, 4U, {2, 1, 2, 2}, {1, 2, 3, 4}},
                                  {968U, 8U, 7U, {8, 8, 3, 8, 2, 8, 5}, {7, 1, 2, 3, 4, 5, 6}}};
    for (const Case& c : cases) {
        const std::uint32_t columns = c.owner + c.width * c.batch;
        const auto query_key = bf16_fixture(static_cast<std::size_t>(columns) * query_key_rows,
                                            229U + c.owner, 257.0F);
        const auto projected =
            bf16_fixture(static_cast<std::size_t>(columns) * value_rows, 233U + c.owner, 257.0F);
        DeviceBuffer<hip_bfloat16> d_query_key(query_key.size()), d_projected(projected.size());
        DeviceBuffer<hip_bfloat16> d_query(static_cast<std::size_t>(columns) * 2048U);
        DeviceBuffer<hip_bfloat16> d_key(static_cast<std::size_t>(columns) * 2048U);
        DeviceBuffer<hip_bfloat16> d_value(static_cast<std::size_t>(columns) * value_rows);
        DeviceBuffer<hip_bfloat16> d_record(static_cast<std::size_t>(c.width) * c.batch *
                                            kChannels);
        DeviceBuffer<std::int32_t> d_valid(c.batch), d_slots(c.batch);
        // The reference record reads the verify columns' combined [value, z] rows.
        const std::size_t verify_columns = static_cast<std::size_t>(c.width) * c.batch;
        std::vector<hip_bfloat16> value_z =
            bf16_fixture(verify_columns * 2U * value_rows, 239U + c.owner, 257.0F);
        for (std::size_t column = 0; column < verify_columns; ++column)
            std::copy_n(projected.data() + (c.owner + column) * value_rows, value_rows,
                        value_z.data() + column * 2U * value_rows);
        DeviceBuffer<hip_bfloat16> d_value_z(value_z.size()), d_z(verify_columns * value_rows);
        copy_to_device(d_value_z, value_z);
        copy_to_device(d_query_key, query_key);
        copy_to_device(d_projected, projected);
        copy_to_device(d_valid, c.valid);
        copy_to_device(d_slots, c.slots);
        const auto reset = [&] {
            copy_to_device(d_history, history);
            for (auto* buffer : {&d_query, &d_key, &d_value, &d_record})
                HIP_CHECK(hipMemsetAsync(buffer->get(), 0xff, buffer->size() * sizeof(hip_bfloat16),
                                         stream));
        };
        const auto at = [&](const DeviceBuffer<hip_bfloat16>& buffer, std::size_t rows) {
            return buffer.get() + rows * c.owner;
        };
        reset();
        HIP_CHECK(projection_conv_prefill_direct_scatter_bf16(
            d_query_key.get(), d_projected.get(), value_rows, d_weight.get(), d_history.get(),
            d_history.get(), d_query.get(), d_key.get(), d_value.get(), c.owner, stream));
        HIP_CHECK(projection_conv_record_bf16(
            at(d_query_key, query_key_rows), d_value_z.get(), d_weight.get(), d_states.get(),
            d_valid.get(), d_slots.get(), nullptr, d_record.get(), at(d_query, 2048U),
            at(d_key, 2048U), at(d_value, value_rows), d_z.get(), c.width, c.batch, state_slots,
            stream));
        HIP_CHECK(hipStreamSynchronize(stream));
        const auto query = copy_to_host(d_query), key = copy_to_host(d_key);
        const auto value = copy_to_host(d_value), record = copy_to_host(d_record);
        const auto published = copy_to_host(d_history);
        reset();
        HIP_CHECK(projection_conv_mixed_split_bf16(
            d_query_key.get(), d_projected.get(), d_weight.get(), d_history.get(), d_history.get(),
            d_query.get(), d_key.get(), d_value.get(), c.owner, d_states.get(), d_valid.get(),
            d_slots.get(), d_record.get(), c.width, c.batch, state_slots, stream));
        HIP_CHECK(hipStreamSynchronize(stream));
        const std::string label =
            "mixed conv owner " + std::to_string(c.owner) + " batch " + std::to_string(c.batch);
        require_bf16_exact(copy_to_host(d_query), query, (label + " query").c_str());
        require_bf16_exact(copy_to_host(d_key), key, (label + " key").c_str());
        require_bf16_exact(copy_to_host(d_value), value, (label + " value").c_str());
        require_bf16_exact(copy_to_host(d_record), record, (label + " record").c_str());
        require_bf16_exact(copy_to_host(d_history), published, (label + " history").c_str());
        require_bf16_exact(copy_to_host(d_states), states, (label + " state pool").c_str());
    }
}

template <typename Scalar>
struct RecurrentOracle {
    std::vector<Scalar> state;
    std::vector<Scalar> output;
};

template <typename Scalar>
RecurrentOracle<Scalar>
recurrent_oracle(const std::vector<hip_bfloat16>& q, const std::vector<hip_bfloat16>& k,
                 const std::vector<hip_bfloat16>& v, const std::vector<float>& g,
                 const std::vector<float>& beta, const std::vector<float>& initial_state,
                 std::uint32_t tokens) {
    RecurrentOracle<Scalar> result;
    result.state.assign(initial_state.begin(), initial_state.end());
    result.output.resize(static_cast<std::size_t>(tokens) * kValueHeads * kValueDim);
    std::vector<Scalar> q_normalized(static_cast<std::size_t>(kQkHeads) * kKeyDim);
    std::vector<Scalar> k_normalized(static_cast<std::size_t>(kQkHeads) * kKeyDim);
    for (std::uint32_t token = 0; token < tokens; ++token) {
        for (std::uint32_t head = 0; head < kQkHeads; ++head) {
            Scalar q_sum           = Scalar{0};
            Scalar k_sum           = Scalar{0};
            const std::size_t base = (static_cast<std::size_t>(token) * kQkHeads + head) * kKeyDim;
            for (std::uint32_t key = 0; key < kKeyDim; ++key) {
                const Scalar q_value = static_cast<float>(q[base + key]);
                const Scalar k_value = static_cast<float>(k[base + key]);
                q_normalized[static_cast<std::size_t>(head) * kKeyDim + key] = q_value;
                k_normalized[static_cast<std::size_t>(head) * kKeyDim + key] = k_value;
                q_sum += q_value * q_value;
                k_sum += k_value * k_value;
            }
            const Scalar q_inverse = Scalar{1} / std::sqrt(q_sum + static_cast<Scalar>(kEpsilon));
            const Scalar k_inverse = Scalar{1} / std::sqrt(k_sum + static_cast<Scalar>(kEpsilon));
            for (std::uint32_t key = 0; key < kKeyDim; ++key) {
                q_normalized[static_cast<std::size_t>(head) * kKeyDim + key] *= q_inverse;
                k_normalized[static_cast<std::size_t>(head) * kKeyDim + key] *= k_inverse;
            }
        }
        for (std::uint32_t value_head = 0; value_head < kValueHeads; ++value_head) {
            const std::uint32_t qk_head = value_head / (kValueHeads / kQkHeads);
            const Scalar alpha          = std::exp(
                static_cast<Scalar>(g[static_cast<std::size_t>(token) * kValueHeads + value_head]));
            const Scalar beta_value = static_cast<Scalar>(
                beta[static_cast<std::size_t>(token) * kValueHeads + value_head]);
            for (std::uint32_t value_row = 0; value_row < kValueDim; ++value_row) {
                const std::size_t state_base =
                    (static_cast<std::size_t>(value_head) * kValueDim + value_row) * kKeyDim;
                Scalar read = Scalar{0};
                for (std::uint32_t key = 0; key < kKeyDim; ++key) {
                    read += result.state[state_base + key] *
                            k_normalized[static_cast<std::size_t>(qk_head) * kKeyDim + key];
                }
                const Scalar value = static_cast<float>(
                    v[(static_cast<std::size_t>(token) * kValueHeads + value_head) * kValueDim +
                      value_row]);
                const Scalar delta = beta_value * (value - alpha * read);
                Scalar readout     = Scalar{0};
                for (std::uint32_t key = 0; key < kKeyDim; ++key) {
                    Scalar& element = result.state[state_base + key];
                    element =
                        alpha * element +
                        delta * k_normalized[static_cast<std::size_t>(qk_head) * kKeyDim + key];
                    readout +=
                        element * q_normalized[static_cast<std::size_t>(qk_head) * kKeyDim + key];
                }
                result.output[(static_cast<std::size_t>(token) * kValueHeads + value_head) *
                                  kValueDim +
                              value_row] = static_cast<Scalar>(kScale) * readout;
            }
        }
    }
    return result;
}

void qualify_recurrent_and_transactions(hipStream_t stream) {
    using namespace ninfer::ops::r9700::gdn;
    const std::size_t state_elements = static_cast<std::size_t>(kValueHeads) * kValueDim * kKeyDim;
    const std::vector<float> initial_state = fp32_fixture(state_elements, 43U, 4096.0F);
    DeviceBuffer<float> d_initial(state_elements);
    DeviceBuffer<float> d_verify(state_elements);
    DeviceBuffer<float> d_fold(state_elements);
    DeviceBuffer<float> d_prefix(state_elements);
    copy_to_device(d_initial, initial_state);
    for (std::uint32_t tokens = 1; tokens <= 8; ++tokens) {
        const std::size_t qk_elements = static_cast<std::size_t>(tokens) * kQkHeads * kKeyDim;
        const std::size_t value_elements =
            static_cast<std::size_t>(tokens) * kValueHeads * kValueDim;
        const std::size_t control_elements = static_cast<std::size_t>(tokens) * kValueHeads;
        const std::vector<hip_bfloat16> q  = bf16_fixture(qk_elements, tokens + 47U, 43.0F);
        const std::vector<hip_bfloat16> k  = bf16_fixture(qk_elements, tokens + 59U, 41.0F);
        const std::vector<hip_bfloat16> v  = bf16_fixture(value_elements, tokens + 61U, 29.0F);
        std::vector<float> g               = fp32_fixture(control_elements, tokens + 67U, 257.0F);
        std::vector<float> beta            = fp32_fixture(control_elements, tokens + 71U, 1024.0F);
        for (std::size_t index = 0; index < control_elements; ++index) {
            g[index]    = -std::abs(g[index]) - 0.05F;
            beta[index] = 1.0F / (1.0F + std::exp(-beta[index]));
        }
        const auto fp64 = recurrent_oracle<double>(q, k, v, g, beta, initial_state, tokens);
        const auto fp32 = recurrent_oracle<float>(q, k, v, g, beta, initial_state, tokens);
        DeviceBuffer<hip_bfloat16> d_q(qk_elements);
        DeviceBuffer<hip_bfloat16> d_k(qk_elements);
        DeviceBuffer<hip_bfloat16> d_v(value_elements);
        DeviceBuffer<float> d_g(control_elements);
        DeviceBuffer<float> d_beta(control_elements);
        DeviceBuffer<hip_bfloat16> d_output(value_elements);
        copy_to_device(d_q, q);
        copy_to_device(d_k, k);
        copy_to_device(d_v, v);
        copy_to_device(d_g, g);
        copy_to_device(d_beta, beta);
        HIP_CHECK(recurrent_bf16_fp32(d_q.get(), d_k.get(), d_v.get(), d_g.get(), d_beta.get(),
                                      d_initial.get(), d_verify.get(), d_output.get(), kKeyDim,
                                      kValueDim, kQkHeads, kValueHeads, tokens, kScale, kEpsilon,
                                      stream));
        HIP_CHECK(hipStreamSynchronize(stream));
        require_fp32_exact(copy_to_host(d_initial), initial_state,
                           "verify checkpoint immutability");
        const std::vector<float> verify_state         = copy_to_host(d_verify);
        const std::vector<hip_bfloat16> verify_output = copy_to_host(d_output);
        require_close(verify_state, fp64.state, 3.0e-3, 2.0e-3, "GDN FP64 state oracle");
        require_close(verify_state, fp32.state, 3.0e-3, 2.0e-3, "GDN FP32 state oracle");
        require_bf16_close(verify_output, fp64.output, 3.0e-2, 2.0e-2, "GDN FP64 output oracle");
        require_bf16_close(verify_output, fp32.output, 3.0e-2, 2.0e-2, "GDN FP32 output oracle");

        HIP_CHECK(copy_state_fp32(d_initial.get(), d_fold.get(), state_elements, stream));
        HIP_CHECK(recurrent_bf16_fp32(d_q.get(), d_k.get(), d_v.get(), d_g.get(), d_beta.get(),
                                      d_fold.get(), d_fold.get(), d_output.get(), kKeyDim,
                                      kValueDim, kQkHeads, kValueHeads, tokens, kScale, kEpsilon,
                                      stream));
        HIP_CHECK(hipStreamSynchronize(stream));
        require_fp32_exact(copy_to_host(d_fold), verify_state, "full-prefix transaction clone");

        const std::uint32_t prefix = tokens / 2U;
        HIP_CHECK(copy_state_fp32(d_initial.get(), d_fold.get(), state_elements, stream));
        HIP_CHECK(hipStreamSynchronize(stream));
        if (prefix == 0) {
            require_fp32_exact(copy_to_host(d_fold), initial_state, "zero-prefix transaction");
        } else {
            const auto prefix_oracle =
                recurrent_oracle<double>(q, k, v, g, beta, initial_state, prefix);
            HIP_CHECK(recurrent_bf16_fp32(d_q.get(), d_k.get(), d_v.get(), d_g.get(), d_beta.get(),
                                          d_initial.get(), d_prefix.get(), d_output.get(), kKeyDim,
                                          kValueDim, kQkHeads, kValueHeads, prefix, kScale,
                                          kEpsilon, stream));
            HIP_CHECK(copy_state_fp32(d_initial.get(), d_fold.get(), state_elements, stream));
            HIP_CHECK(recurrent_bf16_fp32(d_q.get(), d_k.get(), d_v.get(), d_g.get(), d_beta.get(),
                                          d_fold.get(), d_fold.get(), d_output.get(), kKeyDim,
                                          kValueDim, kQkHeads, kValueHeads, prefix, kScale,
                                          kEpsilon, stream));
            HIP_CHECK(hipStreamSynchronize(stream));
            const std::vector<float> prefix_state = copy_to_host(d_prefix);
            require_close(prefix_state, prefix_oracle.state, 3.0e-3, 2.0e-3,
                          "accepted-prefix FP64 state oracle");
            require_fp32_exact(copy_to_host(d_fold), prefix_state,
                               "accepted-prefix transaction clone");
        }
    }
    if (recurrent_bf16_fp32(reinterpret_cast<hip_bfloat16*>(1), reinterpret_cast<hip_bfloat16*>(1),
                            reinterpret_cast<hip_bfloat16*>(1), reinterpret_cast<float*>(1),
                            reinterpret_cast<float*>(1), d_initial.get(), d_verify.get(),
                            reinterpret_cast<hip_bfloat16*>(1), 257U, kValueDim, kQkHeads,
                            kValueHeads, 1, kScale, kEpsilon, stream) != hipErrorInvalidValue ||
        copy_state_fp32(d_initial.get(), d_initial.get(), state_elements, stream) !=
            hipErrorInvalidValue) {
        fail("recurrent transaction boundary did not reject malformed input");
    }
}

} // namespace

int main(int argc, char** argv) {
    try {
        bool prefill_benchmark        = false;
        bool direct_scatter_benchmark = false;
        bool device_info              = false;
        if (argc == 2 && std::string_view(argv[1]) == "--device-info") {
            device_info = true;
        } else if (argc == 2 && std::string_view(argv[1]) == "--prefill-benchmark") {
            prefill_benchmark = true;
        } else if (argc == 2 && std::string_view(argv[1]) == "--direct-scatter-benchmark") {
            direct_scatter_benchmark = true;
        } else if (argc != 1) {
            fail("usage: gdn_op_qual "
                 "[--device-info|--prefill-benchmark|--direct-scatter-benchmark]");
        }
        const DeviceIdentity identity = select_r9700(!device_info);
        if (device_info) {
            std::cout << "{\"ordinal\":" << identity.ordinal << ",\"name\":\"" << identity.name
                      << "\",\"architecture\":\"" << identity.architecture
                      << "\",\"wavefront\":" << identity.wavefront << ",\"pci_bus_id\":\""
                      << identity.pci_bus_id
                      << "\",\"runtime_version\":" << identity.runtime_version
                      << ",\"driver_version\":" << identity.driver_version << "}\n";
            return 0;
        }
        hipStream_t stream = nullptr;
        HIP_CHECK(hipStreamCreateWithFlags(&stream, hipStreamNonBlocking));
        qualify_conv_and_controls(stream);
        if (prefill_benchmark) {
            benchmark_conv_prefill_candidates(stream);
            HIP_CHECK(hipStreamDestroy(stream));
            std::cout << "r9700_gdn_causal_conv_prefill_qualification: PASS\n";
            return 0;
        }
        if (direct_scatter_benchmark) {
            for (const std::uint32_t tokens : {40U, 104U, 2048U})
                benchmark_projection_conv_direct_scatter(stream, tokens);
            HIP_CHECK(hipStreamDestroy(stream));
            std::cout << "r9700_gdn_projection_conv_direct_scatter: PASS\n";
            return 0;
        }
        qualify_batched_projection_conv(stream);
        qualify_mixed_projection_conv(stream);
        qualify_recurrent_and_transactions(stream);
        HIP_CHECK(hipStreamDestroy(stream));
        std::cout
            << "r9700_gdn_op_qualification: PASS conv=T1..8/17/31/32/33/63/64/65/128/1025/2048 "
               "B=1..8 tree/snapshot mixed C=10240 Hq=16 Hv=48 K=V=128\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "r9700_gdn_op_qualification: FAIL: " << error.what() << '\n';
        return 1;
    }
}
