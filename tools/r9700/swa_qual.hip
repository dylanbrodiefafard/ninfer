#include "core/arena.h"
#include "core/device.h"
#include "ninfer/ops/swa.h"
#include "ninfer/types.h"

#include <hip/hip_bfloat16.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

constexpr int kD       = 128;
constexpr int kQHeads  = 32;
constexpr int kKvHeads = 8;
constexpr int kGroup   = 4;
constexpr float kScale = 0.08838834764831844055F;

[[noreturn]] void fail(const std::string& message) { throw std::runtime_error(message); }

template <class T>
ninfer::DeviceBuffer upload(const std::vector<T>& host) {
    ninfer::DeviceBuffer result(host.size() * sizeof(T));
    result.copy_from_host(host.data(), result.size());
    return result;
}

std::size_t q_index(int d, int head, int token, int batch, int tokens) {
    return static_cast<std::size_t>(d) +
           static_cast<std::size_t>(kD) *
               (static_cast<std::size_t>(head) +
                static_cast<std::size_t>(kQHeads) *
                    (static_cast<std::size_t>(token) + static_cast<std::size_t>(tokens) * batch));
}

std::size_t query_kv_index(int d, int head, int token, int batch, int tokens) {
    return static_cast<std::size_t>(d) +
           static_cast<std::size_t>(kD) *
               (static_cast<std::size_t>(head) +
                static_cast<std::size_t>(kKvHeads) *
                    (static_cast<std::size_t>(token) + static_cast<std::size_t>(tokens) * batch));
}

std::size_t context_index(int d, int slot, int head, int lane, int padded) {
    return static_cast<std::size_t>(d) +
           static_cast<std::size_t>(kD) *
               (static_cast<std::size_t>(slot) +
                static_cast<std::size_t>(padded) *
                    (static_cast<std::size_t>(head) + static_cast<std::size_t>(kKvHeads) * lane));
}

hip_bfloat16 pattern(std::size_t index, float phase, float amplitude) {
    return hip_bfloat16(amplitude *
                        std::sin(static_cast<float>(index % 131071U) * 0.0137F + phase));
}

struct Result {
    double max_abs     = 0.0;
    double relative_l2 = 0.0;
    float milliseconds = 0.0F;
};

Result run_case(ninfer::DeviceContext& device, int window, int tokens,
                const std::vector<std::int32_t>& lengths, const std::vector<std::int32_t>& valid,
                const std::vector<std::int32_t>& lanes,
                ninfer::ops::SwaContextExecutionEnvelope envelope) {
    const int batch = static_cast<int>(lengths.size());
    if (batch != static_cast<int>(valid.size()) || batch != static_cast<int>(lanes.size())) {
        fail("SWA qualifier batch vectors disagree");
    }
    const int lane_capacity          = *std::max_element(lanes.begin(), lanes.end()) + 1;
    const int padded                 = window;
    const std::size_t q_elements     = static_cast<std::size_t>(kD) * kQHeads * tokens * batch;
    const std::size_t query_elements = static_cast<std::size_t>(kD) * kKvHeads * tokens * batch;
    const std::size_t context_elements =
        static_cast<std::size_t>(kD) * padded * kKvHeads * lane_capacity;
    std::vector<hip_bfloat16> q(q_elements);
    std::vector<hip_bfloat16> query_k(query_elements);
    std::vector<hip_bfloat16> query_v(query_elements);
    std::vector<hip_bfloat16> context_k(context_elements);
    std::vector<hip_bfloat16> context_v(context_elements);
    for (std::size_t i = 0; i < q.size(); ++i) q[i] = pattern(i, 0.19F, 0.31F);
    for (std::size_t i = 0; i < query_k.size(); ++i) {
        query_k[i] = pattern(i, 0.71F, 0.29F);
        query_v[i] = pattern(i, 1.23F, 0.67F);
    }
    for (std::size_t i = 0; i < context_k.size(); ++i) {
        context_k[i] = pattern(i, 1.83F, 0.27F);
        context_v[i] = pattern(i, 2.41F, 0.73F);
    }
    std::vector<std::int32_t> positions(static_cast<std::size_t>(tokens) * batch);
    for (int b = 0; b < batch; ++b) {
        for (int token = 0; token < tokens; ++token) {
            positions[static_cast<std::size_t>(token) + static_cast<std::size_t>(tokens) * b] =
                lengths[b] + token;
        }
    }

    // Only the committed window and live query prefix are represented inputs. Poison every
    // unused cache slot and inactive Q/K/V row so a load followed by zero multiplication cannot
    // silently pass the finite-input oracle cases. A slot used by any live row remains finite.
    const hip_bfloat16 poison(std::numeric_limits<float>::quiet_NaN());
    for (int b = 0; b < batch; ++b) {
        std::vector<bool> live_slots(static_cast<std::size_t>(window), false);
        if (valid[b] > 0) {
            for (int position = std::max(0, lengths[b] - (window - 1)); position < lengths[b];
                 ++position) {
                live_slots[static_cast<std::size_t>(position & (window - 1))] = true;
            }
        }
        for (int slot = 0; slot < window; ++slot) {
            if (live_slots[static_cast<std::size_t>(slot)]) continue;
            for (int head = 0; head < kKvHeads; ++head) {
                for (int d = 0; d < kD; ++d) {
                    const auto index = context_index(d, slot, head, lanes[b], padded);
                    context_k[index] = context_v[index] = poison;
                }
            }
        }
        for (int token = valid[b]; token < tokens; ++token) {
            for (int head = 0; head < kQHeads; ++head)
                for (int d = 0; d < kD; ++d) q[q_index(d, head, token, b, tokens)] = poison;
            for (int head = 0; head < kKvHeads; ++head)
                for (int d = 0; d < kD; ++d) {
                    const auto index = query_kv_index(d, head, token, b, tokens);
                    query_k[index] = query_v[index] = poison;
                }
        }
    }

    std::vector<double> expected(q_elements, 0.0);
    std::vector<double> scores(static_cast<std::size_t>(window - 1 + tokens));
    for (int b = 0; b < batch; ++b) {
        for (int token = 0; token < valid[b]; ++token) {
            const int query_position =
                positions[static_cast<std::size_t>(token) + static_cast<std::size_t>(tokens) * b];
            const int context_begin = std::max(0, query_position - (window - 1));
            const int context_count = lengths[b] - context_begin;
            const int key_count     = context_count + valid[b];
            for (int q_head = 0; q_head < kQHeads; ++q_head) {
                const int kv_head = q_head / kGroup;
                double maximum    = -std::numeric_limits<double>::infinity();
                for (int key = 0; key < key_count; ++key) {
                    double dot = 0.0;
                    for (int d = 0; d < kD; ++d) {
                        const double q_value =
                            static_cast<float>(q[q_index(d, q_head, token, b, tokens)]);
                        double k_value;
                        if (key < context_count) {
                            const int position = context_begin + key;
                            k_value            = static_cast<float>(context_k[context_index(
                                d, position & (window - 1), kv_head, lanes[b], padded)]);
                        } else {
                            k_value = static_cast<float>(query_k[query_kv_index(
                                d, kv_head, key - context_count, b, tokens)]);
                        }
                        dot = std::fma(q_value, k_value, dot);
                    }
                    scores[static_cast<std::size_t>(key)] = dot * static_cast<double>(kScale);
                    maximum = std::max(maximum, scores[static_cast<std::size_t>(key)]);
                }
                double denominator = 0.0;
                for (int key = 0; key < key_count; ++key) {
                    scores[static_cast<std::size_t>(key)] =
                        std::exp(scores[static_cast<std::size_t>(key)] - maximum);
                    denominator += scores[static_cast<std::size_t>(key)];
                }
                for (int d = 0; d < kD; ++d) {
                    double numerator = 0.0;
                    for (int key = 0; key < key_count; ++key) {
                        double value;
                        if (key < context_count) {
                            const int position = context_begin + key;
                            value              = static_cast<float>(context_v[context_index(
                                d, position & (window - 1), kv_head, lanes[b], padded)]);
                        } else {
                            value = static_cast<float>(query_v[query_kv_index(
                                d, kv_head, key - context_count, b, tokens)]);
                        }
                        numerator =
                            std::fma(scores[static_cast<std::size_t>(key)], value, numerator);
                    }
                    expected[q_index(d, q_head, token, b, tokens)] = numerator / denominator;
                }
            }
        }
    }

    auto d_q         = upload(q);
    auto d_query_k   = upload(query_k);
    auto d_query_v   = upload(query_v);
    auto d_context_k = upload(context_k);
    auto d_context_v = upload(context_v);
    auto d_positions = upload(positions);
    auto d_valid     = upload(valid);
    auto d_lanes     = upload(lanes);
    ninfer::DeviceBuffer d_output(q_elements * sizeof(hip_bfloat16));
    d_output.fill(0x7f);
    ninfer::Tensor q_tensor(d_q.data(), ninfer::DType::BF16, {kD, kQHeads, tokens, batch});
    ninfer::Tensor query_k_tensor(d_query_k.data(), ninfer::DType::BF16,
                                  {kD, kKvHeads, tokens, batch});
    ninfer::Tensor query_v_tensor(d_query_v.data(), ninfer::DType::BF16,
                                  {kD, kKvHeads, tokens, batch});
    ninfer::Tensor positions_tensor(d_positions.data(), ninfer::DType::I32, {tokens, batch});
    ninfer::Tensor valid_tensor(d_valid.data(), ninfer::DType::I32, {batch});
    ninfer::Tensor lanes_tensor(d_lanes.data(), ninfer::DType::I32, {batch});
    ninfer::Tensor output_tensor(d_output.data(), ninfer::DType::BF16,
                                 {kD, kQHeads, tokens, batch});
    const ninfer::CyclicKVCacheLayerView context{
        .k               = ninfer::Tensor(d_context_k.data(), ninfer::DType::BF16,
                                          {kD, padded, kKvHeads, lane_capacity}),
        .v               = ninfer::Tensor(d_context_v.data(), ninfer::DType::BF16,
                                          {kD, padded, kKvHeads, lane_capacity}),
        .capacity        = static_cast<std::uint32_t>(window),
        .padded_capacity = static_cast<std::uint32_t>(padded),
        .num_kv_heads    = kKvHeads,
        .head_dim        = kD,
        .lane_capacity   = lane_capacity,
    };
    const std::size_t workspace_bytes =
        ninfer::ops::swa_workspace_capacity_bytes(envelope, tokens, tokens, batch);
    ninfer::WorkspaceArena workspace(std::max<std::size_t>(1, workspace_bytes));
    ninfer::ops::swa(q_tensor, query_k_tensor, query_v_tensor, positions_tensor, valid_tensor,
                     lanes_tensor, kScale, context, envelope, workspace, output_tensor,
                     device.stream);
    device.synchronize();
    if (workspace.used() != 0 || workspace.peak_used() > workspace_bytes) {
        fail("SWA workspace query/execution high-water mismatch");
    }
    std::vector<hip_bfloat16> actual(q_elements);
    d_output.copy_to_host(actual.data(), d_output.size());
    for (int b = 0; b < batch; ++b)
        for (int token = valid[b]; token < tokens; ++token)
            for (int head = 0; head < kQHeads; ++head)
                for (int d = 0; d < kD; ++d)
                    if (static_cast<float>(actual[q_index(d, head, token, b, tokens)]) != 0.0F)
                        fail("SWA inactive output tail was not exactly zero");
    double squared_error     = 0.0;
    double squared_reference = 0.0;
    Result result;
    for (std::size_t index = 0; index < actual.size(); ++index) {
        const double got   = static_cast<float>(actual[index]);
        const double error = got - expected[index];
        result.max_abs     = std::max(result.max_abs, std::abs(error));
        squared_error += error * error;
        squared_reference += expected[index] * expected[index];
        if (!std::isfinite(got) || !std::isfinite(expected[index]) ||
            std::abs(error) > 0.0045 + 0.004 * std::abs(expected[index])) {
            fail("SWA represented-BF16/FP64 oracle mismatch");
        }
    }
    result.relative_l2 = std::sqrt(squared_error / std::max(1.0e-30, squared_reference));
    if (result.relative_l2 > 0.0045) fail("SWA relative-L2 oracle mismatch");

    constexpr int iterations = 20;
    ninfer::DeviceEventTimer timer(device);
    timer.start();
    for (int iteration = 0; iteration < iterations; ++iteration) {
        ninfer::ops::swa(q_tensor, query_k_tensor, query_v_tensor, positions_tensor, valid_tensor,
                         lanes_tensor, kScale, context, envelope, workspace, output_tensor,
                         device.stream);
    }
    result.milliseconds = timer.stop_ms() / iterations;
    return result;
}

} // namespace

int main() {
    try {
        ninfer::DeviceContext device;
        if (ninfer::ops::swa_workspace_capacity_bytes(
                {0, 96}, 1, 16, static_cast<std::int32_t>(ninfer::kDFlashMaximumConcurrency)) !=
            0) {
            fail("short-envelope SWA capacity must be zero");
        }
        const std::size_t interval_capacity =
            ninfer::ops::swa_workspace_capacity_bytes({0, 2048}, 1, 16, 1);
        for (int tokens = 1; tokens <= 16; ++tokens) {
            if (ninfer::ops::swa_workspace_capacity_bytes({0, 2048}, tokens, tokens, 1) >
                interval_capacity) {
                fail("SWA interval workspace capacity missed an admitted width");
            }
        }
        const Result w4096    = run_case(device, 4096, 3, {4098, 65}, {3, 2}, {1, 0}, {0, 8194});
        const Result w2048    = run_case(device, 2048, 1, {2048}, {1}, {0}, {0, 2048});
        const Result w2048_b2 = run_case(device, 2048, 3, {2048, 65}, {3, 2}, {1, 0}, {0, 2048});
        const Result direct   = run_case(device, 2048, 2, {65}, {2}, {0}, {0, 96});
        const Result saturated =
            run_case(device, 2048, 16, {65, 64, 63, 62}, {16, 12, 8, 4}, {3, 1, 2, 0}, {0, 2048});
        // Production C1 K5 drafter block: six columns over a full 4096-token window.
        const Result production = run_case(device, 4096, 6, {8190}, {6}, {0}, {0, 8194});
        // Production C8 K7/K6 drafter blocks: eight lanes of eight (seven) columns over full
        // 4096-token windows, lanes permuted.
        const Result c8_k7 =
            run_case(device, 4096, 8, {8190, 5000, 4097, 6000, 7777, 9000, 4500, 8000},
                     {8, 8, 8, 8, 8, 8, 8, 8}, {7, 0, 6, 1, 5, 2, 4, 3}, {0, 9000});
        const Result c8_k6 =
            run_case(device, 4096, 7, {8190, 5000, 4097, 6000, 7777, 9000, 4500, 8000},
                     {7, 7, 7, 7, 7, 7, 7, 7}, {7, 0, 6, 1, 5, 2, 4, 3}, {0, 9000});
        std::cout << "swa: PASS gfx1201 wave32 D128/Hq32/Hkv8 W4096 T3/B2 mixed-tail "
                     "W2048 T1 poisoned-unused max_abs="
                  << std::max({w4096.max_abs, w2048.max_abs, w2048_b2.max_abs, direct.max_abs,
                               saturated.max_abs, production.max_abs, c8_k7.max_abs, c8_k6.max_abs})
                  << " rel_l2="
                  << std::max({w4096.relative_l2, w2048.relative_l2, w2048_b2.relative_l2,
                               direct.relative_l2, saturated.relative_l2, production.relative_l2,
                               c8_k7.relative_l2, c8_k6.relative_l2})
                  << " ms4096=" << w4096.milliseconds << " ms2048=" << w2048.milliseconds
                  << " ms2048_t3b2=" << w2048_b2.milliseconds
                  << " ms2048_t16b4=" << saturated.milliseconds
                  << " ms4096_t6b1=" << production.milliseconds
                  << " ms4096_t8b8=" << c8_k7.milliseconds << " ms4096_t7b8=" << c8_k6.milliseconds
                  << '\n';
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "swa: FAIL: " << error.what() << '\n';
        return 1;
    }
}
