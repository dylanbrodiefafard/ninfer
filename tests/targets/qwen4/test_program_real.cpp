#include "targets/qwen4/verifier.h"

#include "core/device.h"

#include <cuda_fp8.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <memory>
#include <numeric>
#include <span>
#include <string>
#include <vector>

namespace verifier = ninfer::targets::qwen4::verifier;
namespace ops = ninfer::ops;

namespace {

constexpr std::int32_t kVocabulary = 248320;
constexpr std::int32_t kEos = verifier::kPleResetToken;
constexpr std::array<std::int32_t, 4> kInputs = {48, 16451, 17120, 22188};
constexpr std::array<std::int32_t, 4> kTargets = {16451, 17120, 22188, 11988};
// Pinned llama.cpp values and the declared cross-representation criterion are mirrored from
// fixtures/external_ppl_manifest.json. The external FP32/IQ4_NL route is integration evidence,
// not the mathematical oracle for NInfer's represented BF16/NVFP4-G16 profile.
constexpr std::array<double, 4> kExternalNll = {3.15870537, 13.7408428, 12.0528638,
                                                 8.32870839};
constexpr double kMaximumAbsoluteNllDelta = 1.0;
constexpr double kMaximumMeanAbsoluteNllDelta = 0.5;
constexpr std::array<std::uint64_t, 3> kMultiplier = {
    23703573157769ULL, 20109073645365ULL, 8052911324071ULL};
constexpr std::array<std::int32_t, 16> kPrime = {
    20000003, 20000023, 20000033, 20000047, 20000059, 20000063, 20000069, 20000077,
    20000081, 20000093, 20000107, 20000147, 20000153, 20000159, 20000161, 20000171};
constexpr std::array<std::int32_t, 16> kOffset = {
    0, 20000003, 40000026, 60000059, 80000106, 100000165, 120000228, 140000297,
    160000374, 180000455, 200000548, 220000655, 240000802, 260000955, 280001114,
    300001275};

template <class T>
std::vector<T> copy_tensor(const ninfer::Tensor& tensor) {
    std::vector<T> result(static_cast<std::size_t>(tensor.numel()));
    CUDA_CHECK(cudaMemcpy(result.data(), tensor.data, result.size() * sizeof(T),
                          cudaMemcpyDeviceToHost));
    return result;
}

void append_tensor_bytes(std::vector<std::uint8_t>& destination, const ninfer::Tensor& tensor) {
    const std::size_t offset = destination.size();
    destination.resize(offset + tensor.bytes());
    CUDA_CHECK(cudaMemcpy(destination.data() + offset, tensor.data, tensor.bytes(),
                          cudaMemcpyDeviceToHost));
}

template <class T, class Predicate>
void append_checked(std::vector<std::uint8_t>& destination, const ninfer::Tensor& tensor,
                    Predicate predicate, const char* label, int& failures) {
    const auto values = copy_tensor<T>(tensor);
    if (!std::all_of(values.begin(), values.end(), predicate)) {
        std::cerr << "non-finite continuation state in " << label << '\n';
        ++failures;
    }
    const auto* bytes = reinterpret_cast<const std::uint8_t*>(values.data());
    destination.insert(destination.end(), bytes, bytes + values.size() * sizeof(T));
}

bool finite_bf16(std::span<const std::uint16_t> values) {
    return std::all_of(values.begin(), values.end(), [](std::uint16_t bits) {
        return (bits & 0x7f80U) != 0x7f80U;
    });
}

std::int64_t signed_u64(std::uint64_t value) { return std::bit_cast<std::int64_t>(value); }

std::array<std::int32_t, 16> expected_ple_rows(
    std::int32_t current, const std::array<std::int32_t, 2>& history) {
    const std::int32_t lag1 = history[1];
    const std::int32_t lag2 = lag1 == kEos ? kEos : history[0];
    std::array<std::int32_t, 16> rows{};
    const std::uint64_t mixed2 = static_cast<std::uint64_t>(current) * kMultiplier[0] ^
                                 static_cast<std::uint64_t>(lag1) * kMultiplier[1];
    const std::uint64_t mixed3 = mixed2 ^ static_cast<std::uint64_t>(lag2) * kMultiplier[2];
    for (std::size_t head = 0; head < rows.size(); ++head) {
        const std::int64_t mixed = signed_u64(head < 8 ? mixed2 : mixed3);
        std::int64_t remainder = mixed % kPrime[head];
        if (remainder < 0) { remainder += kPrime[head]; }
        rows[head] = kOffset[head] + static_cast<std::int32_t>(remainder);
    }
    return rows;
}

struct TokenSnapshot {
    std::vector<std::uint16_t> logits;
    std::vector<std::uint16_t> final_hidden;
    std::vector<std::uint16_t> gr;
    std::vector<std::int32_t> ple_rows;
    std::vector<std::int32_t> qsa_ids;
    std::vector<std::int32_t> qsa_counts;
    std::vector<std::int32_t> router_ids;
    std::vector<std::uint32_t> router_weight_bits;
    std::uint32_t nll_bits = 0;

    bool operator==(const TokenSnapshot&) const = default;
};

std::vector<std::uint8_t> snapshot_continuation(const verifier::State& state, int& failures) {
    std::vector<std::uint8_t> snapshot;
    snapshot.reserve(verifier::kPersistentStateBytes + state.residual().bytes());
    for (std::size_t layer = 0; layer < verifier::kLayerCount; ++layer) {
        if (state.gdn()[layer]) {
            append_checked<std::uint16_t>(snapshot, state.gdn()[layer]->conv,
                                          [](std::uint16_t bits) {
                                              return (bits & 0x7f80U) != 0x7f80U;
                                          },
                                          "GDN convolution", failures);
            append_checked<float>(snapshot, state.gdn()[layer]->recurrence,
                                  [](float value) { return std::isfinite(value); },
                                  "GDN recurrence", failures);
        }
        if (state.qsa()[layer]) {
            const auto& qsa = *state.qsa()[layer];
            append_tensor_bytes(snapshot, qsa.k_codes);
            append_tensor_bytes(snapshot, qsa.v_codes);
            append_tensor_bytes(snapshot, qsa.k_scales);
            append_tensor_bytes(snapshot, qsa.v_scales);
            append_checked<std::uint16_t>(snapshot, qsa.raw_index_keys,
                                          [](std::uint16_t bits) {
                                              return (bits & 0x7f80U) != 0x7f80U;
                                          },
                                          "QSA raw index key", failures);
            append_tensor_bytes(snapshot, qsa.positions);
        }
    }
    append_checked<std::uint16_t>(snapshot, state.ple_conv_state(),
                                  [](std::uint16_t bits) {
                                      return (bits & 0x7f80U) != 0x7f80U;
                                  },
                                  "PLE convolution", failures);
    append_tensor_bytes(snapshot, state.ple_token_history());
    append_checked<std::uint16_t>(snapshot, state.residual(),
                                  [](std::uint16_t bits) {
                                      return (bits & 0x7f80U) != 0x7f80U;
                                  },
                                  "residual", failures);
    return snapshot;
}

double decode_e2m1(std::uint8_t nibble) {
    constexpr std::array<double, 8> magnitude = {0.0, 0.5, 1.0, 1.5,
                                                  2.0, 3.0, 4.0, 6.0};
    return (nibble & 8U) == 0 ? magnitude[nibble & 7U] : -magnitude[nibble & 7U];
}

int validate_current_qsa_cache(const verifier::State& state, std::int32_t token_index) {
    int failures = 0;
    for (std::size_t layer = 0; layer < verifier::kLayerCount; ++layer) {
        if (!state.qsa()[layer]) { continue; }
        const auto& qsa = *state.qsa()[layer];
        std::array<std::int32_t, 3> position{};
        const std::size_t position_offset =
            static_cast<std::size_t>(token_index) * 3 * sizeof(std::int32_t);
        CUDA_CHECK(cudaMemcpy(position.data(),
                              static_cast<const std::byte*>(qsa.positions.data) + position_offset,
                              sizeof(position), cudaMemcpyDeviceToHost));
        if (position != std::array<std::int32_t, 3>{token_index, token_index, token_index}) {
            std::cerr << "QSA position mismatch at layer " << layer << '\n';
            ++failures;
        }

        bool any_nonzero = false;
        std::array<std::uint16_t, 128> raw_key{};
        const std::size_t raw_key_offset =
            static_cast<std::size_t>(token_index) * raw_key.size() * sizeof(std::uint16_t);
        CUDA_CHECK(cudaMemcpy(raw_key.data(),
                              static_cast<const std::byte*>(qsa.raw_index_keys.data) +
                                  raw_key_offset,
                              sizeof(raw_key), cudaMemcpyDeviceToHost));
        if (!finite_bf16(raw_key)) {
            std::cerr << "non-finite QSA raw key at layer " << layer << '\n';
            ++failures;
        }
        for (int head = 0; head < 2; ++head) {
            std::array<std::uint8_t, 128> k_codes{};
            std::array<std::uint8_t, 128> v_codes{};
            std::array<std::uint8_t, 16> k_scales{};
            std::array<std::uint8_t, 16> v_scales{};
            const std::size_t code_offset = 128ULL *
                (static_cast<std::size_t>(token_index) + verifier::kQsaCapacity * head);
            const std::size_t scale_offset = 16ULL *
                (static_cast<std::size_t>(token_index) + verifier::kQsaCapacity * head);
            CUDA_CHECK(cudaMemcpy(k_codes.data(),
                                  static_cast<const std::byte*>(qsa.k_codes.data) + code_offset,
                                  k_codes.size(), cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(v_codes.data(),
                                  static_cast<const std::byte*>(qsa.v_codes.data) + code_offset,
                                  v_codes.size(), cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(k_scales.data(),
                                  static_cast<const std::byte*>(qsa.k_scales.data) + scale_offset,
                                  k_scales.size(), cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(v_scales.data(),
                                  static_cast<const std::byte*>(qsa.v_scales.data) + scale_offset,
                                  v_scales.size(), cudaMemcpyDeviceToHost));
            for (int group = 0; group < 16; ++group) {
                __nv_fp8_e4m3 k_scale_bits;
                __nv_fp8_e4m3 v_scale_bits;
                k_scale_bits.__x = k_scales[group];
                v_scale_bits.__x = v_scales[group];
                const float k_scale = static_cast<float>(k_scale_bits);
                const float v_scale = static_cast<float>(v_scale_bits);
                if (!std::isfinite(k_scale) || !std::isfinite(v_scale)) {
                    std::cerr << "non-finite QSA NVFP4 scale at layer " << layer << '\n';
                    ++failures;
                }
                for (int lane = 0; lane < 16; ++lane) {
                    const int dimension = group * 16 + lane;
                    const std::uint8_t k_packed = k_codes[dimension / 2];
                    const std::uint8_t v_packed = v_codes[dimension / 2];
                    const std::uint8_t k_code =
                        (dimension & 1) == 0 ? k_packed & 15U : k_packed >> 4U;
                    const std::uint8_t v_code =
                        (dimension & 1) == 0 ? v_packed & 15U : v_packed >> 4U;
                    const double decoded_k = decode_e2m1(k_code) * k_scale;
                    const double decoded_v = decode_e2m1(v_code) * v_scale;
                    if (!std::isfinite(decoded_k) || !std::isfinite(decoded_v)) {
                        std::cerr << "non-finite decoded QSA NVFP4 value at layer " << layer
                                  << '\n';
                        ++failures;
                    }
                    any_nonzero |= decoded_k != 0.0 || decoded_v != 0.0;
                }
            }
        }
        if (!any_nonzero || qsa.k_codes.dtype != ninfer::DType::U8 ||
            qsa.v_codes.dtype != ninfer::DType::U8 ||
            qsa.k_scales.dtype != ninfer::DType::FP8_E4M3FN ||
            qsa.v_scales.dtype != ninfer::DType::FP8_E4M3FN) {
            std::cerr << "QSA current NVFP4 row was not represented at layer " << layer << '\n';
            ++failures;
        }
    }
    return failures;
}

TokenSnapshot snapshot_and_validate(const verifier::TokenResultView& result,
                                    const verifier::State& state,
                                    const std::array<std::int32_t, 16>& expected_rows,
                                    const std::array<std::int32_t, 2>& expected_history,
                                    int& failures) {
    TokenSnapshot snapshot;
    snapshot.logits = copy_tensor<std::uint16_t>(result.logits);
    snapshot.final_hidden = copy_tensor<std::uint16_t>(result.final_hidden);
    snapshot.ple_rows = copy_tensor<std::int32_t>(result.ple_row_ids);
    const auto nll = copy_tensor<float>(result.nll);
    snapshot.nll_bits = std::bit_cast<std::uint32_t>(nll[0]);
    if (!finite_bf16(snapshot.logits) || !finite_bf16(snapshot.final_hidden) ||
        !std::isfinite(nll[0]) || nll[0] < 0.0F ||
        !std::equal(snapshot.ple_rows.begin(), snapshot.ple_rows.end(), expected_rows.begin())) {
        std::cerr << "non-finite output/NLL or wrong PLE rows at token " << result.token_index
                  << '\n';
        ++failures;
    }
    const auto history = copy_tensor<std::int32_t>(state.ple_token_history());
    if (!std::equal(history.begin(), history.end(), expected_history.begin())) {
        std::cerr << "PLE history mismatch at token " << result.token_index << '\n';
        ++failures;
    }

    for (const auto& qsa : result.qsa) {
        const auto ids = copy_tensor<std::int32_t>(qsa.selected_ids);
        const auto count = copy_tensor<std::int32_t>(qsa.selected_count);
        snapshot.qsa_counts.push_back(count[0]);
        snapshot.qsa_ids.insert(snapshot.qsa_ids.end(), ids.begin(), ids.end());
        if (count[0] != result.token_index + 1) {
            std::cerr << "QSA selected count mismatch at layer " << qsa.layer << '\n';
            ++failures;
        }
        for (std::int32_t i = 0; i < ops::kQsaSelectedCapacity; ++i) {
            const std::int32_t expected = i <= result.token_index ? i : -1;
            if (ids[static_cast<std::size_t>(i)] != expected) {
                std::cerr << "QSA selected id mismatch at layer " << qsa.layer << '\n';
                ++failures;
                break;
            }
        }
    }
    failures += validate_current_qsa_cache(state, result.token_index);

    for (const auto& router : result.routers) {
        const auto ids = copy_tensor<std::int32_t>(router.selected_ids);
        const auto weights = copy_tensor<float>(router.selected_weights);
        snapshot.router_ids.insert(snapshot.router_ids.end(), ids.begin(), ids.end());
        for (float value : weights) {
            snapshot.router_weight_bits.push_back(std::bit_cast<std::uint32_t>(value));
        }
        float sum = 0.0F;
        for (std::size_t rank = 0; rank < ids.size(); ++rank) {
            sum += weights[rank];
            if (ids[rank] < 0 || ids[rank] >= 512 || !std::isfinite(weights[rank]) ||
                weights[rank] <= 0.0F ||
                std::find(ids.begin(), ids.begin() + static_cast<std::ptrdiff_t>(rank),
                          ids[rank]) != ids.begin() + static_cast<std::ptrdiff_t>(rank) ||
                (rank > 0 && weights[rank] > weights[rank - 1]) ||
                (rank > 0 && weights[rank] == weights[rank - 1] && ids[rank] < ids[rank - 1])) {
                std::cerr << "invalid router diagnostics at layer " << router.layer << '\n';
                ++failures;
                break;
            }
        }
        if (std::abs(sum - 1.0F) > 2.0e-6F) {
            std::cerr << "router weights do not sum to one at layer " << router.layer << '\n';
            ++failures;
        }
    }

    for (const auto& gr : result.gr) {
        const auto attention = copy_tensor<std::uint16_t>(gr.attention_residual);
        const auto ffn = copy_tensor<std::uint16_t>(gr.ffn_residual);
        if (!finite_bf16(attention) || !finite_bf16(ffn)) {
            std::cerr << "non-finite GR boundary at layer " << gr.layer << '\n';
            ++failures;
        }
        snapshot.gr.insert(snapshot.gr.end(), attention.begin(), attention.end());
        snapshot.gr.insert(snapshot.gr.end(), ffn.begin(), ffn.end());
    }
    return snapshot;
}

} // namespace

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
        std::unique_ptr<verifier::LoadedModel> model = verifier::LoadedModel::load(path, device);
        verifier::Program program(*model, device.stream);
        std::array<std::vector<TokenSnapshot>, 2> runs;
        std::array<std::vector<float>, 2> nlls;
        std::vector<std::uint8_t> first_continuation;
        int failures = 0;
        for (std::size_t replay = 0; replay < runs.size(); ++replay) {
            program.reset();
            std::array<std::int32_t, 2> history = {kEos, kEos};
            for (std::size_t token = 0; token < kInputs.size(); ++token) {
                const auto expected_rows = expected_ple_rows(kInputs[token], history);
                history = {history[1], kInputs[token]};
                const auto result = program.execute_token(kInputs[token], kTargets[token]);
                if (result.token_index != static_cast<std::int32_t>(token) ||
                    program.frontier() != static_cast<std::int32_t>(token + 1) ||
                    result.qsa.size() != verifier::kQsaLayerCount ||
                    result.routers.size() != verifier::kLayerCount ||
                    result.gr.size() != verifier::kLayerCount) {
                    std::cerr << "Program frontier or diagnostic extent mismatch\n";
                    ++failures;
                }
                TokenSnapshot snapshot = snapshot_and_validate(
                    result, program.state(), expected_rows, history, failures);
                nlls[replay].push_back(std::bit_cast<float>(snapshot.nll_bits));
                runs[replay].push_back(std::move(snapshot));
            }
            const auto continuation = snapshot_continuation(program.state(), failures);
            if (replay == 0) {
                first_continuation = continuation;
            } else if (continuation != first_continuation) {
                std::cerr << "continuation state changed after deterministic reset/replay\n";
                ++failures;
            }
        }
        if (runs[0] != runs[1] || nlls[0] != nlls[1]) {
            std::cerr << "logits/NLL/diagnostics changed after deterministic reset/replay\n";
            ++failures;
        }

        double total_nll = 0.0;
        double total_absolute_delta = 0.0;
        double maximum_absolute_delta = 0.0;
        std::cout << "Qwen4 verifier teacher-forced NLL";
        for (std::size_t token = 0; token < kInputs.size(); ++token) {
            total_nll += nlls[0][token];
            const double delta = std::abs(static_cast<double>(nlls[0][token]) -
                                          kExternalNll[token]);
            total_absolute_delta += delta;
            maximum_absolute_delta = std::max(maximum_absolute_delta, delta);
            std::cout << " " << kInputs[token] << "->" << kTargets[token] << "="
                      << nlls[0][token];
        }
        const double ppl = std::exp(total_nll / static_cast<double>(kInputs.size()));
        const double mean_absolute_delta =
            total_absolute_delta / static_cast<double>(kInputs.size());
        std::cout << " PPL=" << ppl << " max_abs_delta_nll=" << maximum_absolute_delta
                  << " mean_abs_delta_nll=" << mean_absolute_delta << '\n';
        if (!std::isfinite(ppl) || !std::isfinite(maximum_absolute_delta) ||
            !std::isfinite(mean_absolute_delta)) {
            std::cerr << "teacher-forced PPL or paired NLL delta is not finite\n";
            ++failures;
        }
        if (maximum_absolute_delta > kMaximumAbsoluteNllDelta ||
            mean_absolute_delta > kMaximumMeanAbsoluteNllDelta) {
            std::cerr << "paired external integration criterion failed: max_abs_delta_nll="
                      << maximum_absolute_delta << " limit=" << kMaximumAbsoluteNllDelta
                      << " mean_abs_delta_nll=" << mean_absolute_delta
                      << " limit=" << kMaximumMeanAbsoluteNllDelta << '\n';
            ++failures;
        }
        std::cout << (failures == 0 ? "OK" : "FAIL")
                  << " qwen4_program_real tokens=" << kInputs.size() << '\n';
        return failures == 0 ? 0 : 1;
    } catch (const std::exception& error) {
        std::cerr << "Qwen4 Program real-artifact execution failed: " << error.what() << '\n';
        return 1;
    }
}
