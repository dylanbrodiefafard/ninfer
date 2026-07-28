#include "ninfer/ops/mtp_round.h"
#include "ops/op_tester.h"

#include <cstdint>
#include <iostream>
#include <string>
#include <vector>

using namespace ninfer;
using namespace ninfer::test;

namespace {

std::vector<std::int32_t> alignment_oracle(const std::vector<std::int32_t>& verify_ids,
                                           std::int32_t token, std::int32_t accepted,
                                           const std::vector<std::int32_t>& initial) {
    const int k = static_cast<int>(verify_ids.size()) - 1;
    auto out    = initial;
    for (int i = 0; i < k; ++i) { out[static_cast<std::size_t>(i)] = verify_ids[i + 1]; }
    out[static_cast<std::size_t>(accepted)] = token;
    return out;
}

int run_case(int k, std::int32_t accepted) {
    std::vector<std::int32_t> verify_ids(static_cast<std::size_t>(k + 1));
    std::vector<std::int32_t> initial(static_cast<std::size_t>(k + 1));
    for (int i = 0; i <= k; ++i) {
        verify_ids[static_cast<std::size_t>(i)] = 1000 + 17 * i + k;
        initial[static_cast<std::size_t>(i)]    = -7000 - 13 * i;
    }
    const std::int32_t token_value = 90000 + 31 * k + accepted;
    const auto expected            = alignment_oracle(verify_ids, token_value, accepted, initial);

    DeviceBuffer d_verify   = to_device(verify_ids);
    DeviceBuffer d_token    = to_device<std::int32_t>({token_value});
    DeviceBuffer d_accepted = to_device<std::int32_t>({accepted});
    GuardedDeviceBuffer d_alignment(initial.size() * sizeof(std::int32_t));
    d_alignment.copy_from_host(initial.data(), d_alignment.bytes());

    Tensor verify(d_verify.p, DType::I32, {k + 1});
    Tensor token(d_token.p, DType::I32, {1});
    Tensor accepted_tensor(d_accepted.p, DType::I32, {1});
    Tensor alignment(d_alignment.data(), DType::I32, {k + 1});
    ops::mtp_prepare_alignment_ids(verify, token, accepted_tensor, alignment, nullptr);
    cuda_synchronize();

    const std::string label =
        "mtp alignment K=" + std::to_string(k) + " A=" + std::to_string(accepted);
    int failures =
        verify_exact((label + " output").c_str(),
                     from_device<std::int32_t>(d_alignment.data(), initial.size()), expected);
    failures += verify_exact((label + " verify unchanged").c_str(),
                             from_device<std::int32_t>(d_verify, verify_ids.size()), verify_ids);
    failures += verify_exact((label + " token unchanged").c_str(),
                             from_device<std::int32_t>(d_token, 1), {token_value});
    failures += verify_exact((label + " accepted unchanged").c_str(),
                             from_device<std::int32_t>(d_accepted, 1), {accepted});
    failures += d_alignment.verify_guards((label + " guards").c_str());
    return failures;
}

} // namespace

int main() {
    if (cuda_unavailable()) {
        std::cout << "mtp_round: SKIP (CUDA unavailable)\n";
        return 77;
    }

    int failures = 0;
    for (int k = 1; k <= 5; ++k) {
        failures += run_case(k, 0);
        if (k > 1) failures += run_case(k, k / 2);
        failures += run_case(k, k);
    }

    if (failures != 0) {
        std::cerr << "mtp_round failures=" << failures << '\n';
        return 1;
    }
    std::cout << "mtp_round: PASS\n";
    return 0;
}
