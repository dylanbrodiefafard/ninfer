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
#include <limits>
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
constexpr std::array<std::int32_t, 5> kTeacherForcedTokens = {
    48, 16451, 17120, 22188, 11988};
constexpr auto kInputs = [] {
    std::array<std::int32_t, kTeacherForcedTokens.size() - 1> result{};
    std::copy_n(kTeacherForcedTokens.begin(), result.size(), result.begin());
    return result;
}();
constexpr auto kTargets = [] {
    std::array<std::int32_t, kTeacherForcedTokens.size() - 1> result{};
    std::copy_n(kTeacherForcedTokens.begin() + 1, result.size(), result.begin());
    return result;
}();
// The first four values from the pinned 601-token llama.cpp trace are mirrored from
// fixtures/external_ppl_manifest.json. This prefix remains useful localization evidence, but the
// complete trace has 33/600 deltas above one nat and is an unresolved integration discrepancy,
// not a production correctness claim or a mathematical oracle for BF16/NVFP4-G16 execution.
constexpr std::array<double, 4> kExternalPairedPrefixNll = {
    3.0563440376731457, 13.217154127288497, 12.015043390206641, 8.821618971613773};
constexpr double kPrefixMaximumAbsoluteNllDelta = 1.0;
constexpr double kPrefixMaximumMeanAbsoluteNllDelta = 0.5;
constexpr double kNllAbsoluteTolerance = 2.0e-3;
constexpr double kNllRelativeTolerance = 1.0e-5;
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

float bf16_to_f32(std::uint16_t bits) {
    return std::bit_cast<float>(static_cast<std::uint32_t>(bits) << 16U);
}

double represented_nll_oracle(std::span<const std::uint16_t> logits,
                              std::int32_t target) {
    if (logits.size() != kVocabulary || target < 0 || target >= kVocabulary) {
        throw std::logic_error("Program NLL oracle received malformed represented inputs");
    }
    double maximum = -std::numeric_limits<double>::infinity();
    for (std::uint16_t bits : logits) {
        maximum = std::max(maximum, static_cast<double>(bf16_to_f32(bits)));
    }
    double exponential_sum = 0.0;
    for (std::uint16_t bits : logits) {
        exponential_sum += std::exp(static_cast<double>(bf16_to_f32(bits)) - maximum);
    }
    return maximum + std::log(exponential_sum) -
           static_cast<double>(bf16_to_f32(logits[static_cast<std::size_t>(target)]));
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
                                    std::int32_t target,
                                    int& failures) {
    TokenSnapshot snapshot;
    snapshot.logits = copy_tensor<std::uint16_t>(result.logits);
    snapshot.final_hidden = copy_tensor<std::uint16_t>(result.final_hidden);
    snapshot.ple_rows = copy_tensor<std::int32_t>(result.ple_row_ids);
    const auto nll = copy_tensor<float>(result.nll);
    snapshot.nll_bits = std::bit_cast<std::uint32_t>(nll[0]);
    const double expected_nll = represented_nll_oracle(snapshot.logits, target);
    const double nll_tolerance =
        kNllAbsoluteTolerance + kNllRelativeTolerance * std::abs(expected_nll);
    if (!finite_bf16(snapshot.logits) || !finite_bf16(snapshot.final_hidden) ||
        !std::isfinite(nll[0]) || nll[0] < 0.0F ||
        !std::equal(snapshot.ple_rows.begin(), snapshot.ple_rows.end(), expected_rows.begin())) {
        std::cerr << "non-finite output/NLL or wrong PLE rows at token " << result.token_index
                  << '\n';
        ++failures;
    }
    if (std::abs(static_cast<double>(nll[0]) - expected_nll) > nll_tolerance) {
        std::cerr << "Program NLL does not match its represented BF16 logits at token "
                  << result.token_index << " actual=" << nll[0]
                  << " reference=" << expected_nll << " tolerance=" << nll_tolerance
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
        verifier::Program program(*model, device);
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
                    result, program.state(), expected_rows, history, kTargets[token],
                    failures);
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

        struct PrefillSnapshot {
            std::vector<std::uint16_t> logits;
            std::vector<std::uint16_t> hidden;
            std::vector<std::int32_t> rows;
            std::vector<std::uint8_t> state;
            std::vector<std::uint16_t> continuation_logits;
            std::uint32_t continuation_nll = 0;
        };
        const auto capture_prefill = [&](std::span<const std::int32_t> ids,
                                         bool partitioned) {
            program.reset();
            verifier::PrefillResultView result;
            if (partitioned) {
                (void)program.prefill_chunk(ids.first(1));
                result = program.prefill_chunk(ids.subspan(1));
            } else {
                result = program.prefill_chunk(ids);
            }
            if (result.begin_index != (partitioned ? 1 : 0) ||
                result.end_index != static_cast<std::int32_t>(ids.size() - 1) ||
                program.frontier() != static_cast<std::int32_t>(ids.size())) {
                std::cerr << "prefill frontier/result interval mismatch\n";
                ++failures;
            }
            PrefillSnapshot snapshot{
                .logits = copy_tensor<std::uint16_t>(result.logits),
                .hidden = copy_tensor<std::uint16_t>(result.final_hidden),
                .rows = copy_tensor<std::int32_t>(result.ple_row_ids),
            };
            if (!finite_bf16(snapshot.logits) || !finite_bf16(snapshot.hidden)) {
                std::cerr << "non-finite prefill final output\n";
                ++failures;
            }
            std::array<std::int32_t, 2> history{kEos, kEos};
            std::vector<std::int32_t> expected_rows;
            expected_rows.reserve(static_cast<std::size_t>(ops::kPleHeads) * ids.size());
            for (std::int32_t id : ids) {
                const auto rows = expected_ple_rows(id, history);
                expected_rows.insert(expected_rows.end(), rows.begin(), rows.end());
                history = {history[1], id};
            }
            const std::size_t row_begin =
                partitioned ? static_cast<std::size_t>(ops::kPleHeads) : 0;
            if (!std::equal(snapshot.rows.begin(), snapshot.rows.end(),
                            expected_rows.begin() + static_cast<std::ptrdiff_t>(row_begin))) {
                std::cerr << "prefill PLE row panel mismatch\n";
                ++failures;
            }
            const auto qsa_counts = copy_tensor<std::int32_t>(result.qsa_selected_count);
            const auto qsa_ids = copy_tensor<std::int32_t>(result.qsa_selected_ids);
            for (std::int32_t token = 0; token < result.qsa_selected_count.ne[0]; ++token) {
                const std::int32_t expected_count = result.begin_index + token + 1;
                if (qsa_counts[static_cast<std::size_t>(token)] != expected_count) {
                    std::cerr << "prefill causal QSA selected count mismatch\n";
                    ++failures;
                }
                for (std::int32_t selected = 0; selected < ops::kQsaSelectedCapacity;
                     ++selected) {
                    const std::int32_t expected = selected < expected_count ? selected : -1;
                    const std::size_t offset = static_cast<std::size_t>(selected) +
                                               static_cast<std::size_t>(
                                                   ops::kQsaSelectedCapacity) * token;
                    if (qsa_ids[offset] != expected) {
                        std::cerr << "prefill causal QSA selected id mismatch\n";
                        ++failures;
                        break;
                    }
                }
            }
            const auto device_history =
                copy_tensor<std::int32_t>(program.state().ple_token_history());
            if (!std::equal(device_history.begin(), device_history.end(), history.begin())) {
                std::cerr << "prefill committed PLE token history mismatch\n";
                ++failures;
            }
            for (std::int32_t position = 0;
                 position < static_cast<std::int32_t>(ids.size()); ++position) {
                failures += validate_current_qsa_cache(program.state(), position);
            }
            snapshot.state = snapshot_continuation(program.state(), failures);
            const auto continuation = program.execute_token(kTargets.back(), 1);
            snapshot.continuation_logits = copy_tensor<std::uint16_t>(continuation.logits);
            snapshot.continuation_nll =
                std::bit_cast<std::uint32_t>(copy_tensor<float>(continuation.nll)[0]);
            return snapshot;
        };

        const PrefillSnapshot one_shot = capture_prefill(kInputs, false);
        const PrefillSnapshot replay = capture_prefill(kInputs, false);
        if (one_shot.logits != replay.logits || one_shot.hidden != replay.hidden ||
            one_shot.rows != replay.rows || one_shot.state != replay.state ||
            one_shot.continuation_logits != replay.continuation_logits ||
            one_shot.continuation_nll != replay.continuation_nll) {
            std::cerr << "prefill changed after deterministic reset/replay\n";
            ++failures;
        }
        const PrefillSnapshot partitioned = capture_prefill(kInputs, true);
        if (partitioned.state != one_shot.state) {
            std::cerr << "prefill one-shot/partition represented state mismatch\n";
            ++failures;
        }
        if (partitioned.hidden != one_shot.hidden || partitioned.logits != one_shot.logits ||
            partitioned.continuation_logits != one_shot.continuation_logits) {
            std::cerr << "prefill one-shot/partition represented outputs mismatch\n";
            ++failures;
        }
        if (one_shot.state != first_continuation) {
            std::cerr << "prefill/scalar represented continuation state mismatch\n";
            ++failures;
        }
        if (one_shot.hidden != runs[0].back().final_hidden ||
            one_shot.logits != runs[0].back().logits) {
            std::cerr << "prefill/scalar represented outputs mismatch\n";
            ++failures;
        }

        constexpr std::array<std::int32_t, 3> eos_crossing{48, kEos, 17120};
        const PrefillSnapshot eos_result = capture_prefill(eos_crossing, false);
        if (eos_result.rows.size() !=
            static_cast<std::size_t>(ops::kPleHeads) * eos_crossing.size()) {
            std::cerr << "EOS-crossing PLE panel extent mismatch\n";
            ++failures;
        }
        program.reset();
        (void)program.prefill_chunk(std::span(eos_crossing).first(2));
        const auto eos_tail = program.prefill_chunk(std::span(eos_crossing).last(1));
        const auto expected_eos_tail_rows =
            expected_ple_rows(eos_crossing.back(), {eos_crossing.front(), kEos});
        if (copy_tensor<std::int32_t>(eos_tail.ple_row_ids) !=
            std::vector<std::int32_t>(expected_eos_tail_rows.begin(),
                                      expected_eos_tail_rows.end())) {
            std::cerr << "EOS-crossing split PLE row mismatch\n";
            ++failures;
        }
        const std::array<std::int32_t, 2> expected_eos_history{kEos, eos_crossing.back()};
        if (copy_tensor<std::int32_t>(program.state().ple_token_history()) !=
            std::vector<std::int32_t>(expected_eos_history.begin(),
                                      expected_eos_history.end())) {
            std::cerr << "EOS-crossing split PLE history mismatch\n";
            ++failures;
        }

        program.reset();
        const std::int32_t frontier_before_rejection = program.frontier();
        try {
            (void)program.prefill_chunk(std::span<const std::int32_t>{});
            std::cerr << "empty prefill was accepted\n";
            ++failures;
        } catch (const std::invalid_argument&) {
        }
        constexpr std::array<std::int32_t, 1> invalid_token{-1};
        try {
            (void)program.prefill_chunk(invalid_token);
            std::cerr << "invalid prefill token was accepted\n";
            ++failures;
        } catch (const std::invalid_argument&) {
        }
        constexpr std::array<std::int32_t, 1> valid_after_rejection{48};
        (void)program.prefill_chunk(valid_after_rejection);
        if (frontier_before_rejection != 0 || program.frontier() != 1) {
            std::cerr << "prefill rejection mutated frontier or poisoned Program\n";
            ++failures;
        }
        const auto state_before_capacity_rejection =
            snapshot_continuation(program.state(), failures);
        const std::vector<std::int32_t> excessive_at_nonzero_frontier(
            verifier::kMaximumPrefillChunk, 48);
        try {
            (void)program.prefill_chunk(excessive_at_nonzero_frontier);
            std::cerr << "nonzero-frontier excessive prefill was accepted\n";
            ++failures;
        } catch (const std::length_error&) {
        }
        if (program.frontier() != 1 ||
            snapshot_continuation(program.state(), failures) !=
                state_before_capacity_rejection) {
            std::cerr << "capacity rejection mutated state or frontier\n";
            ++failures;
        }

        double total_nll = 0.0;
        double total_absolute_delta = 0.0;
        double maximum_absolute_delta = 0.0;
        std::cout << "Qwen4 verifier teacher-forced NLL";
        for (std::size_t token = 0; token < kInputs.size(); ++token) {
            total_nll += nlls[0][token];
            const double delta = std::abs(static_cast<double>(nlls[0][token]) -
                                          kExternalPairedPrefixNll[token]);
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
        if (maximum_absolute_delta > kPrefixMaximumAbsoluteNllDelta ||
            mean_absolute_delta > kPrefixMaximumMeanAbsoluteNllDelta) {
            std::cerr << "paired external prefix localization criterion failed: max_abs_delta_nll="
                      << maximum_absolute_delta << " limit=" << kPrefixMaximumAbsoluteNllDelta
                      << " mean_abs_delta_nll=" << mean_absolute_delta
                      << " limit=" << kPrefixMaximumMeanAbsoluteNllDelta << '\n';
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
