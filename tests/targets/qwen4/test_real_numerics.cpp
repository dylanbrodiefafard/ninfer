#include "targets/qwen4/real_oracle_common.h"

#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <memory>
#include <stdexcept>

namespace verifier = ninfer::targets::qwen4::verifier;
namespace real_oracle = ninfer::test::qwen4::real_oracle;

int main() {
    const char* configured = std::getenv("NINFER_QWEN4_VERIFY_WEIGHTS");
    if (configured == nullptr || *configured == '\0') {
        std::cout << "skip: NINFER_QWEN4_VERIFY_WEIGHTS is not set\n";
        return 77;
    }
    const std::filesystem::path path(configured);
    if (!std::filesystem::is_regular_file(path)) {
        std::cerr << "verifier artifact is not a regular file: " << path << '\n';
        return 1;
    }

    try {
        ninfer::DeviceContext device(0);
        const std::unique_ptr<verifier::LoadedModel> model =
            verifier::LoadedModel::load(path, device);
        int failures = real_oracle::run_gr_cell(*model, device);
        failures += real_oracle::run_qsa_cell(*model, device);
        failures += real_oracle::run_gdn_cell(*model, device);
        failures += real_oracle::run_moe_cell(*model, device);
        failures += real_oracle::run_ple_cell(*model, device);
        std::cout << (failures == 0 ? "OK" : "FAIL") << " qwen4_real_numerics\n";
        return failures == 0 ? 0 : 1;
    } catch (const std::exception& error) {
        std::cerr << "Qwen4 real numerical oracle failed: " << error.what() << '\n';
        return 1;
    }
}
