#include "ninfer/ops/linear.h"
#include "ninfer/ops/mtp_fc.h"
#include "ninfer/ops/mtp_pack.h"
#include "ops/linear/linear_test_common.h"
#include "ops/op_tester.h"

#include <array>
#include <cstdint>
#include <iostream>
#include <string>
#include <vector>

namespace {

using ninfer::test::linear::make_nvfp4_weight;

void fill_halves(std::vector<std::uint16_t>& embedding, std::vector<std::uint16_t>& hidden,
                 std::int32_t tokens) {
    std::uint32_t state = 0x91u + static_cast<std::uint32_t>(tokens);
    for (std::size_t i = 0; i < embedding.size(); ++i) {
        state        = state * 1664525u + 1013904223u;
        embedding[i] = static_cast<std::uint16_t>((state >> 9) & 0x7fffu);
        hidden[i]    = static_cast<std::uint16_t>((state >> 17) & 0x7fffu);
    }
}

int run_fused_vs_pack_a16(std::int32_t tokens) {
    constexpr std::int32_t kN = 5120;
    constexpr std::int32_t kD = 5120;
    constexpr std::int32_t kK = 10240;
    auto host_weight          = make_nvfp4_weight(kN, kK, 823U);
    ninfer::DeviceBuffer device_weight(host_weight.payload.size());
    device_weight.copy_from_host(host_weight.payload.data(), device_weight.bytes);
    const ninfer::Weight weight = host_weight.device_weight(device_weight.p);

    const std::size_t half = static_cast<std::size_t>(kD) * static_cast<std::size_t>(tokens);
    std::vector<std::uint16_t> embedding(half, 0);
    std::vector<std::uint16_t> hidden(half, 0);
    fill_halves(embedding, hidden, tokens);

    ninfer::test::GuardedDeviceBuffer device_embedding(half * sizeof(std::uint16_t));
    ninfer::test::GuardedDeviceBuffer device_hidden(half * sizeof(std::uint16_t));
    ninfer::test::GuardedDeviceBuffer device_packed(2 * half * sizeof(std::uint16_t));
    ninfer::test::GuardedDeviceBuffer device_reference(static_cast<std::size_t>(kN) * tokens *
                                                       sizeof(std::uint16_t));
    ninfer::test::GuardedDeviceBuffer device_fused(static_cast<std::size_t>(kN) * tokens *
                                                   sizeof(std::uint16_t));
    device_embedding.copy_from_host(embedding.data(), half * sizeof(std::uint16_t));
    device_hidden.copy_from_host(hidden.data(), half * sizeof(std::uint16_t));
    device_reference.fill(0xcd);
    device_fused.fill(0xcd);

    ninfer::Tensor embedding_tensor(device_embedding.data(), ninfer::DType::BF16, {kD, tokens});
    ninfer::Tensor hidden_tensor(device_hidden.data(), ninfer::DType::BF16, {kD, tokens});
    ninfer::Tensor packed_tensor(device_packed.data(), ninfer::DType::BF16, {kK, tokens});
    ninfer::Tensor reference_tensor(device_reference.data(), ninfer::DType::BF16, {kN, tokens});
    ninfer::Tensor fused_tensor(device_fused.data(), ninfer::DType::BF16, {kN, tokens});

    ninfer::ops::mtp_pack_fc_input(embedding_tensor, hidden_tensor, packed_tensor, nullptr);
    ninfer::ops::linear(packed_tensor, weight, reference_tensor, nullptr);
    ninfer::ops::mtp_fc(embedding_tensor, hidden_tensor, weight, fused_tensor, nullptr);
    ninfer::test::cuda_synchronize();

    const std::string label = "mtp_fc vs pack+linear A16 T=" + std::to_string(tokens);
    int failures            = 0;
    failures += ninfer::test::verify_exact(
        label.c_str(),
        ninfer::test::from_device<std::uint16_t>(device_fused.data(),
                                                 static_cast<std::size_t>(kN) * tokens),
        ninfer::test::from_device<std::uint16_t>(device_reference.data(),
                                                 static_cast<std::size_t>(kN) * tokens));
    failures += device_embedding.verify_guards((label + " embedding").c_str());
    failures += device_hidden.verify_guards((label + " hidden").c_str());
    failures += device_fused.verify_guards((label + " fused").c_str());
    return failures;
}

int check_w4a4_workspace_gate() {
    const std::size_t below = ninfer::ops::linear_workspace_capacity_bytes(
        ninfer::QType::NVFP4, 5120, 10240, ninfer::ops::LinearPolicy::AllowA4, 1, 7);
    const std::size_t at = ninfer::ops::linear_workspace_capacity_bytes(
        ninfer::QType::NVFP4, 5120, 10240, ninfer::ops::LinearPolicy::AllowA4, 8, 8);
    int failures = 0;
    if (below != 0) {
        std::cerr << "MtpFc AllowA4 workspace for T<=7 must be A16 (zero)\n";
        ++failures;
    }
    if (at == 0) {
        std::cerr << "MtpFc AllowA4 workspace for T=8 must be W4A4 (nonzero)\n";
        ++failures;
    }
    return failures;
}

} // namespace

int main() {
    if (!ninfer::test::linear::cuda_available()) {
        std::cerr << "FAIL: no usable CUDA device\n";
        return 1;
    }
    int failures = 0;
    failures += check_w4a4_workspace_gate();
    for (const std::int32_t tokens : {1, 2, 3, 4, 5, 6, 7, 8}) {
        failures += run_fused_vs_pack_a16(tokens);
    }
    std::cout << (failures == 0 ? "OK" : "FAIL") << " mtp_fc\n";
    return failures == 0 ? 0 : 1;
}
