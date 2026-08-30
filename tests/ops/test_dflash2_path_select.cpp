#include "ninfer/ops/dflash2_path_select.h"

#include "ops/direct_bf16_weight.h"
#include "ops/input_projection_test_common.h"
#include "ops/op_tester.h"
#include "ops/quantized_weight.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <limits>
#include <string>
#include <vector>

using namespace ninfer;
using namespace ninfer::test;
using namespace ninfer::test::direct_bf16_weight;

namespace {

constexpr std::int32_t kHidden = ops::kDflash2PathSelectHidden;
constexpr std::int32_t kRank   = ops::kDflash2PathSelectRank;
constexpr std::int32_t kTopK   = ops::kDflash2PathSelectTopK;

std::vector<std::uint16_t> encode_bf16(const std::vector<float>& values) {
    std::vector<std::uint16_t> bits(values.size());
    for (std::size_t i = 0; i < values.size(); ++i) bits[i] = f32_to_bf16(values[i]);
    return bits;
}

bool logit_better(float value, int index, float best_value, int best_index) {
    return value > best_value || (value == best_value && index < best_index);
}

void insert_topk(std::vector<float>& vals, std::vector<int>& idxs, float value, int index) {
    if (index < 0 || !std::isfinite(value) ||
        !logit_better(value, index, vals.back(), idxs.back())) {
        return;
    }
    int pos = kTopK - 1;
    while (pos > 0 && logit_better(value, index, vals[pos - 1], idxs[pos - 1])) {
        vals[pos] = vals[pos - 1];
        idxs[pos] = idxs[pos - 1];
        --pos;
    }
    vals[pos] = value;
    idxs[pos] = index;
}

std::vector<std::int32_t> path_oracle(const std::vector<float>& logits,
                                      const std::vector<float>& hidden, const HostWeight& weight,
                                      const std::vector<float>& pred, const std::vector<float>& succ,
                                      const std::vector<std::int32_t>& anchors,
                                      std::int32_t vocab, std::int32_t tokens, std::int32_t batch,
                                      std::int32_t codebook_rows) {
    std::vector<std::int32_t> path(static_cast<std::size_t>(tokens) * batch);
    for (std::int32_t b = 0; b < batch; ++b) {
        int prev = anchors[static_cast<std::size_t>(b)];
        for (std::int32_t t = 0; t < tokens; ++t) {
            std::vector<float> top_val(static_cast<std::size_t>(kTopK),
                                       -std::numeric_limits<float>::infinity());
            std::vector<int> top_idx(static_cast<std::size_t>(kTopK), 0x7fffffff);
            const std::size_t logit_col =
                (static_cast<std::size_t>(b) * tokens + t) * static_cast<std::size_t>(vocab);
            for (std::int32_t v = 0; v < vocab; ++v) {
                insert_topk(top_val, top_idx, logits[logit_col + v], v);
            }

            std::vector<float> column(static_cast<std::size_t>(kHidden));
            const std::size_t hid_col =
                (static_cast<std::size_t>(b) * tokens + t) * static_cast<std::size_t>(kHidden);
            for (std::int32_t d = 0; d < kHidden; ++d) {
                column[static_cast<std::size_t>(d)] = hidden[hid_col + d];
            }
            std::vector<double> h(static_cast<std::size_t>(kRank));
            for (std::int32_t r = 0; r < kRank; ++r) {
                h[static_cast<std::size_t>(r)] = dot_fp64(weight, r, column);
            }

            int best_id    = top_idx[0];
            double best_sc = 0.0;
            for (std::int32_t c = 0; c < kTopK; ++c) {
                const int cand = top_idx[static_cast<std::size_t>(c)];
                double acc     = static_cast<double>(top_val[static_cast<std::size_t>(c)]);
                const std::size_t pred_row =
                    static_cast<std::size_t>(prev) * kRank;
                const std::size_t succ_row =
                    static_cast<std::size_t>(cand) * kRank;
                for (std::int32_t r = 0; r < kRank; ++r) {
                    acc += (static_cast<double>(pred[pred_row + r]) * h[static_cast<std::size_t>(r)]) *
                           static_cast<double>(succ[succ_row + r]);
                }
                if (c == 0 || acc > best_sc || (acc == best_sc && cand < best_id)) {
                    best_sc = acc;
                    best_id = cand;
                }
            }
            path[static_cast<std::size_t>(t) + static_cast<std::size_t>(b) * tokens] = best_id;
            prev = best_id;
            (void)codebook_rows;
        }
    }
    return path;
}

void call_path_select(const Tensor& logits, const Tensor& hidden, const Weight& weight,
                      const Tensor& pred, const Tensor& succ, const Tensor& anchors,
                      std::int32_t batch, float temperature, unsigned long long seed, Tensor& path,
                      WorkspaceArena& workspace, cudaStream_t stream,
                      const Tensor* logit_token_ids = nullptr, const Weight* pred_nvfp4 = nullptr,
                      const Weight* succ_nvfp4 = nullptr, Tensor* selector_ids = nullptr,
                      Tensor* selector_q = nullptr, bool force_greedy = false) {
    std::vector<ops::SamplingConfig> configs(static_cast<std::size_t>(batch));
    for (std::int32_t b = 0; b < batch; ++b) {
        configs[static_cast<std::size_t>(b)].temperature = temperature;
        configs[static_cast<std::size_t>(b)].seed        = seed;
    }
    GuardedDeviceBuffer device_configs(configs.size() * sizeof(ops::SamplingConfig));
    device_configs.copy_from_host(configs.data(), device_configs.bytes());
    std::vector<std::int32_t> logical_positions(static_cast<std::size_t>(batch), 100);
    GuardedDeviceBuffer device_positions(logical_positions.size() * sizeof(std::int32_t));
    device_positions.copy_from_host(logical_positions.data(), device_positions.bytes());
    Tensor positions(device_positions.data(), DType::I32, {batch});
    ops::dflash2_path_select(logits, hidden, weight, pred, succ, anchors, positions,
                             reinterpret_cast<const ops::SamplingConfig*>(device_configs.data()),
                             path, workspace, stream, logit_token_ids, pred_nvfp4, succ_nvfp4,
                             selector_ids, selector_q, 0, 0, force_greedy);
    cuda_synchronize();
}

int run_greedy_case(const char* label, std::int32_t vocab, std::int32_t tokens, std::int32_t batch,
                    std::uint32_t seed) {
    const std::int32_t codebook_rows = vocab;
    HostWeight host_weight           = make_patterned(kRank, kHidden, seed);
    DeviceWeight device_weight(std::move(host_weight));

    std::vector<float> logits(static_cast<std::size_t>(vocab) * tokens * batch);
    std::vector<float> hidden(static_cast<std::size_t>(kHidden) * tokens * batch);
    std::vector<float> pred(static_cast<std::size_t>(kRank) * codebook_rows);
    std::vector<float> succ(static_cast<std::size_t>(kRank) * codebook_rows);
    std::vector<std::int32_t> anchors(static_cast<std::size_t>(batch));
    fill_uniform(logits, seed + 3, -8.0f, 8.0f);
    fill_uniform(hidden, seed + 5, -1.0f, 1.0f);
    fill_uniform(pred, seed + 7, -0.5f, 0.5f);
    fill_uniform(succ, seed + 9, -0.5f, 0.5f);
    round_to_bf16(logits);
    round_to_bf16(hidden);
    round_to_bf16(pred);
    round_to_bf16(succ);
    for (std::int32_t b = 0; b < batch; ++b) {
        anchors[static_cast<std::size_t>(b)] = (17 + b * 13) % vocab;
    }

    const auto expected =
        path_oracle(logits, hidden, device_weight.host, pred, succ, anchors, vocab, tokens, batch,
                    codebook_rows);

    const auto logit_bits  = encode_bf16(logits);
    const auto hidden_bits = encode_bf16(hidden);
    const auto pred_bits   = encode_bf16(pred);
    const auto succ_bits   = encode_bf16(succ);
    GuardedDeviceBuffer device_logits(logit_bits.size() * sizeof(std::uint16_t));
    GuardedDeviceBuffer device_hidden(hidden_bits.size() * sizeof(std::uint16_t));
    GuardedDeviceBuffer device_pred(pred_bits.size() * sizeof(std::uint16_t));
    GuardedDeviceBuffer device_succ(succ_bits.size() * sizeof(std::uint16_t));
    GuardedDeviceBuffer device_anchors(anchors.size() * sizeof(std::int32_t));
    GuardedDeviceBuffer device_path(static_cast<std::size_t>(tokens) * batch * sizeof(std::int32_t));
    GuardedDeviceBuffer device_sel_ids(
        static_cast<std::size_t>(kTopK) * tokens * batch * sizeof(std::int32_t));
    GuardedDeviceBuffer device_sel_q(
        static_cast<std::size_t>(kTopK) * tokens * batch * sizeof(float));
    device_logits.copy_from_host(logit_bits.data(), device_logits.bytes());
    device_hidden.copy_from_host(hidden_bits.data(), device_hidden.bytes());
    device_pred.copy_from_host(pred_bits.data(), device_pred.bytes());
    device_succ.copy_from_host(succ_bits.data(), device_succ.bytes());
    device_anchors.copy_from_host(anchors.data(), device_anchors.bytes());
    device_path.fill(0xcd);
    device_sel_ids.fill(0xcd);
    device_sel_q.fill(0xcd);

    Tensor logits_t =
        batch == 1 ? Tensor(device_logits.data(), DType::BF16, {vocab, tokens})
                   : Tensor(device_logits.data(), DType::BF16, {vocab, tokens, batch});
    Tensor hidden_t =
        batch == 1 ? Tensor(device_hidden.data(), DType::BF16, {kHidden, tokens})
                   : Tensor(device_hidden.data(), DType::BF16, {kHidden, tokens, batch});
    Tensor pred_t(device_pred.data(), DType::BF16, {kRank, codebook_rows});
    Tensor succ_t(device_succ.data(), DType::BF16, {kRank, codebook_rows});
    Tensor anchors_t(device_anchors.data(), DType::I32, {batch});
    Tensor path_t = batch == 1 ? Tensor(device_path.data(), DType::I32, {tokens})
                               : Tensor(device_path.data(), DType::I32, {tokens, batch});
    Tensor sel_ids_t =
        batch == 1 ? Tensor(device_sel_ids.data(), DType::I32, {kTopK, tokens})
                   : Tensor(device_sel_ids.data(), DType::I32, {kTopK, tokens, batch});
    Tensor sel_q_t = batch == 1 ? Tensor(device_sel_q.data(), DType::FP32, {kTopK, tokens})
                                : Tensor(device_sel_q.data(), DType::FP32, {kTopK, tokens, batch});

    const std::size_t workspace_bytes = ops::dflash2_path_select_workspace_capacity_bytes(
        QType::BF16_CTRL, tokens, tokens, batch);
    WorkspaceArena workspace(std::max<std::size_t>(256, workspace_bytes));
    call_path_select(logits_t, hidden_t, device_weight.view(), pred_t, succ_t, anchors_t, batch,
                     0.6f, 0ull, path_t, workspace, nullptr, nullptr, nullptr, nullptr, &sel_ids_t,
                     &sel_q_t, true);
    cuda_synchronize();

    int failures = verify_exact(label, from_device<std::int32_t>(device_path.data(),
                                                               static_cast<std::size_t>(tokens) *
                                                                   batch),
                                expected);
    const auto got_ids =
        from_device<std::int32_t>(device_sel_ids.data(),
                                  static_cast<std::size_t>(kTopK) * tokens * batch);
    const auto got_q =
        from_device<float>(device_sel_q.data(), static_cast<std::size_t>(kTopK) * tokens * batch);
    for (std::int32_t b = 0; b < batch; ++b) {
        for (std::int32_t t = 0; t < tokens; ++t) {
            const std::size_t hop =
                (static_cast<std::size_t>(b) * tokens + t) * static_cast<std::size_t>(kTopK);
            const int chosen = expected[static_cast<std::size_t>(t) +
                                        static_cast<std::size_t>(b) * tokens];
            float qsum       = 0.0f;
            int hot          = 0;
            bool matched     = false;
            for (std::int32_t c = 0; c < kTopK; ++c) {
                const float qv = got_q[hop + static_cast<std::size_t>(c)];
                qsum += qv;
                if (qv == 1.0f) {
                    ++hot;
                    if (got_ids[hop + static_cast<std::size_t>(c)] == chosen) { matched = true; }
                } else if (qv != 0.0f) {
                    std::cerr << label << ": greedy selector_q not one-hot at t=" << t << "\n";
                    ++failures;
                }
            }
            if (hot != 1 || !matched || qsum != 1.0f) {
                std::cerr << label << ": greedy selector did not one-hot the path token t=" << t
                          << "\n";
                ++failures;
            }
        }
    }
    failures += device_path.verify_guards("dflash2_path_select path");
    failures += device_sel_ids.verify_guards("dflash2_path_select selector_ids");
    failures += device_sel_q.verify_guards("dflash2_path_select selector_q");
    failures += device_weight.verify_preserved("dflash2_path_select weight");
    return failures;
}

int run_stochastic_support_case() {
    constexpr std::int32_t vocab  = 32;
    constexpr std::int32_t tokens = 2;
    HostWeight host_weight        = make_patterned(kRank, kHidden, 9u);
    DeviceWeight device_weight(std::move(host_weight));

    std::vector<float> logits(static_cast<std::size_t>(vocab) * tokens, -12.0f);
    std::vector<float> hidden(static_cast<std::size_t>(kHidden) * tokens, 0.0f);
    std::vector<float> pred(static_cast<std::size_t>(kRank) * vocab, 0.0f);
    std::vector<float> succ(static_cast<std::size_t>(kRank) * vocab, 0.0f);
    for (std::int32_t t = 0; t < tokens; ++t) {
        for (std::int32_t c = 0; c < kTopK; ++c) {
            logits[static_cast<std::size_t>(t) * vocab + c] = 8.0f - 0.05f * static_cast<float>(c);
        }
    }
    round_to_bf16(logits);
    round_to_bf16(hidden);
    const std::int32_t anchor = 3;
    std::vector<std::int32_t> anchors{anchor};

    const auto logit_bits  = encode_bf16(logits);
    const auto hidden_bits = encode_bf16(hidden);
    const auto pred_bits   = encode_bf16(pred);
    const auto succ_bits   = encode_bf16(succ);
    GuardedDeviceBuffer device_logits(logit_bits.size() * sizeof(std::uint16_t));
    GuardedDeviceBuffer device_hidden(hidden_bits.size() * sizeof(std::uint16_t));
    GuardedDeviceBuffer device_pred(pred_bits.size() * sizeof(std::uint16_t));
    GuardedDeviceBuffer device_succ(succ_bits.size() * sizeof(std::uint16_t));
    GuardedDeviceBuffer device_anchors(sizeof(std::int32_t));
    GuardedDeviceBuffer device_path(static_cast<std::size_t>(tokens) * sizeof(std::int32_t));
    device_logits.copy_from_host(logit_bits.data(), device_logits.bytes());
    device_hidden.copy_from_host(hidden_bits.data(), device_hidden.bytes());
    device_pred.copy_from_host(pred_bits.data(), device_pred.bytes());
    device_succ.copy_from_host(succ_bits.data(), device_succ.bytes());
    device_anchors.copy_from_host(&anchor, sizeof(anchor));
    device_path.fill(0xcd);

    Tensor logits_t(device_logits.data(), DType::BF16, {vocab, tokens});
    Tensor hidden_t(device_hidden.data(), DType::BF16, {kHidden, tokens});
    Tensor pred_t(device_pred.data(), DType::BF16, {kRank, vocab});
    Tensor succ_t(device_succ.data(), DType::BF16, {kRank, vocab});
    Tensor anchors_t(device_anchors.data(), DType::I32, {1});
    Tensor path_t(device_path.data(), DType::I32, {tokens});
    WorkspaceArena workspace(std::max<std::size_t>(
        256, ops::dflash2_path_select_workspace_capacity_bytes(QType::BF16_CTRL, tokens, tokens, 1)));
    call_path_select(logits_t, hidden_t, device_weight.view(), pred_t, succ_t, anchors_t, 1, 0.6f,
                     42ull, path_t, workspace, nullptr);
    cuda_synchronize();

    const auto got = from_device<std::int32_t>(device_path.data(), static_cast<std::size_t>(tokens));
    for (std::int32_t t = 0; t < tokens; ++t) {
        if (got[static_cast<std::size_t>(t)] < 0 || got[static_cast<std::size_t>(t)] >= kTopK) {
            std::cerr << "dflash2_path_select stochastic: token outside peaked top-16\n";
            return 1;
        }
    }
    return 0;
}

int run_stochastic_batch_row_invariance_case() {
    constexpr std::int32_t vocab  = 32;
    constexpr std::int32_t tokens = 2;
    constexpr std::int32_t batch  = 2;
    HostWeight host_weight        = make_patterned(kRank, kHidden, 13u);
    DeviceWeight device_weight(std::move(host_weight));

    std::vector<float> logits(static_cast<std::size_t>(vocab) * tokens * batch, -12.0f);
    std::vector<float> hidden(static_cast<std::size_t>(kHidden) * tokens * batch, 0.0f);
    std::vector<float> pred(static_cast<std::size_t>(kRank) * vocab, 0.0f);
    std::vector<float> succ(static_cast<std::size_t>(kRank) * vocab, 0.0f);
    for (std::int32_t b = 0; b < batch; ++b) {
        for (std::int32_t t = 0; t < tokens; ++t) {
            const std::size_t base = static_cast<std::size_t>(b * tokens + t) * vocab;
            for (std::int32_t c = 0; c < kTopK; ++c) {
                logits[base + static_cast<std::size_t>(c)] =
                    8.0f - 0.05f * static_cast<float>(c);
            }
        }
    }
    round_to_bf16(logits);
    round_to_bf16(hidden);

    const auto logit_bits  = encode_bf16(logits);
    const auto hidden_bits = encode_bf16(hidden);
    const auto pred_bits   = encode_bf16(pred);
    const auto succ_bits   = encode_bf16(succ);
    GuardedDeviceBuffer device_logits(logit_bits.size() * sizeof(std::uint16_t));
    GuardedDeviceBuffer device_hidden(hidden_bits.size() * sizeof(std::uint16_t));
    GuardedDeviceBuffer device_pred(pred_bits.size() * sizeof(std::uint16_t));
    GuardedDeviceBuffer device_succ(succ_bits.size() * sizeof(std::uint16_t));
    GuardedDeviceBuffer device_anchors(static_cast<std::size_t>(batch) * sizeof(std::int32_t));
    GuardedDeviceBuffer device_path(static_cast<std::size_t>(tokens) * batch *
                                    sizeof(std::int32_t));
    const std::array<std::int32_t, batch> anchors{3, 3};
    device_logits.copy_from_host(logit_bits.data(), device_logits.bytes());
    device_hidden.copy_from_host(hidden_bits.data(), device_hidden.bytes());
    device_pred.copy_from_host(pred_bits.data(), device_pred.bytes());
    device_succ.copy_from_host(succ_bits.data(), device_succ.bytes());
    device_anchors.copy_from_host(anchors.data(), device_anchors.bytes());
    device_path.fill(0xcd);

    Tensor logits_t(device_logits.data(), DType::BF16, {vocab, tokens, batch});
    Tensor hidden_t(device_hidden.data(), DType::BF16, {kHidden, tokens, batch});
    Tensor pred_t(device_pred.data(), DType::BF16, {kRank, vocab});
    Tensor succ_t(device_succ.data(), DType::BF16, {kRank, vocab});
    Tensor anchors_t(device_anchors.data(), DType::I32, {batch});
    Tensor path_t(device_path.data(), DType::I32, {tokens, batch});
    WorkspaceArena workspace(std::max<std::size_t>(
        256, ops::dflash2_path_select_workspace_capacity_bytes(QType::BF16_CTRL, tokens, tokens,
                                                               batch)));
    call_path_select(logits_t, hidden_t, device_weight.view(), pred_t, succ_t, anchors_t, batch,
                     0.6f, 42ull, path_t, workspace, nullptr);

    const auto got = from_device<std::int32_t>(device_path.data(),
                                                static_cast<std::size_t>(tokens) * batch);
    if (!std::equal(got.begin(), got.begin() + tokens, got.begin() + tokens)) {
        std::cerr << "dflash2_path_select stochastic result depends on compact batch row\n";
        return 1;
    }
    return 0;
}

int run_stochastic_selector_q_case() {
    constexpr std::int32_t vocab  = 32;
    constexpr std::int32_t tokens = 2;
    HostWeight host_weight        = make_patterned(kRank, kHidden, 9u);
    DeviceWeight device_weight(std::move(host_weight));

    std::vector<float> logits(static_cast<std::size_t>(vocab) * tokens, -12.0f);
    std::vector<float> hidden(static_cast<std::size_t>(kHidden) * tokens, 0.0f);
    std::vector<float> pred(static_cast<std::size_t>(kRank) * vocab, 0.0f);
    std::vector<float> succ(static_cast<std::size_t>(kRank) * vocab, 0.0f);
    for (std::int32_t t = 0; t < tokens; ++t) {
        for (std::int32_t c = 0; c < kTopK; ++c) {
            logits[static_cast<std::size_t>(t) * vocab + c] = 8.0f - 0.05f * static_cast<float>(c);
        }
    }
    round_to_bf16(logits);
    round_to_bf16(hidden);
    const std::int32_t anchor = 3;

    const auto logit_bits  = encode_bf16(logits);
    const auto hidden_bits = encode_bf16(hidden);
    const auto pred_bits   = encode_bf16(pred);
    const auto succ_bits   = encode_bf16(succ);
    GuardedDeviceBuffer device_logits(logit_bits.size() * sizeof(std::uint16_t));
    GuardedDeviceBuffer device_hidden(hidden_bits.size() * sizeof(std::uint16_t));
    GuardedDeviceBuffer device_pred(pred_bits.size() * sizeof(std::uint16_t));
    GuardedDeviceBuffer device_succ(succ_bits.size() * sizeof(std::uint16_t));
    GuardedDeviceBuffer device_anchors(sizeof(std::int32_t));
    GuardedDeviceBuffer device_path(static_cast<std::size_t>(tokens) * sizeof(std::int32_t));
    GuardedDeviceBuffer device_sel_ids(static_cast<std::size_t>(kTopK) * tokens *
                                       sizeof(std::int32_t));
    GuardedDeviceBuffer device_sel_q(static_cast<std::size_t>(kTopK) * tokens * sizeof(float));
    device_logits.copy_from_host(logit_bits.data(), device_logits.bytes());
    device_hidden.copy_from_host(hidden_bits.data(), device_hidden.bytes());
    device_pred.copy_from_host(pred_bits.data(), device_pred.bytes());
    device_succ.copy_from_host(succ_bits.data(), device_succ.bytes());
    device_anchors.copy_from_host(&anchor, sizeof(anchor));
    device_path.fill(0xcd);
    device_sel_ids.fill(0xcd);
    device_sel_q.fill(0xcd);

    Tensor logits_t(device_logits.data(), DType::BF16, {vocab, tokens});
    Tensor hidden_t(device_hidden.data(), DType::BF16, {kHidden, tokens});
    Tensor pred_t(device_pred.data(), DType::BF16, {kRank, vocab});
    Tensor succ_t(device_succ.data(), DType::BF16, {kRank, vocab});
    Tensor anchors_t(device_anchors.data(), DType::I32, {1});
    Tensor path_t(device_path.data(), DType::I32, {tokens});
    Tensor sel_ids_t(device_sel_ids.data(), DType::I32, {kTopK, tokens});
    Tensor sel_q_t(device_sel_q.data(), DType::FP32, {kTopK, tokens});
    WorkspaceArena workspace(std::max<std::size_t>(
        256, ops::dflash2_path_select_workspace_capacity_bytes(QType::BF16_CTRL, tokens, tokens, 1)));
    call_path_select(logits_t, hidden_t, device_weight.view(), pred_t, succ_t, anchors_t, 1, 0.6f,
                     42ull, path_t, workspace, nullptr, nullptr, nullptr, nullptr, &sel_ids_t,
                     &sel_q_t);
    cuda_synchronize();

    const auto path =
        from_device<std::int32_t>(device_path.data(), static_cast<std::size_t>(tokens));
    const auto ids =
        from_device<std::int32_t>(device_sel_ids.data(), static_cast<std::size_t>(kTopK) * tokens);
    const auto q =
        from_device<float>(device_sel_q.data(), static_cast<std::size_t>(kTopK) * tokens);
    int failures = 0;
    for (std::int32_t t = 0; t < tokens; ++t) {
        const std::size_t hop = static_cast<std::size_t>(t) * kTopK;
        float qsum            = 0.0f;
        int hot               = 0;
        int mass_n            = 0;
        bool path_in_list     = false;
        float path_q          = 0.0f;
        for (std::int32_t c = 0; c < kTopK; ++c) {
            const float qv = q[hop + static_cast<std::size_t>(c)];
            qsum += qv;
            if (qv == 1.0f) { ++hot; }
            if (qv > 1.0e-6f) { ++mass_n; }
            if (ids[hop + static_cast<std::size_t>(c)] == path[static_cast<std::size_t>(t)]) {
                path_in_list = true;
                path_q       = qv;
            }
        }
        if (std::abs(qsum - 1.0f) > 1.0e-5f || hot == 1 || mass_n < 2 || !path_in_list ||
            path_q <= 0.0f) {
            std::cerr << "dflash2_path_select stochastic q: hop " << t << " qsum=" << qsum
                      << " hot=" << hot << " mass_n=" << mass_n << " path_q=" << path_q << "\n";
            ++failures;
        }
    }
    failures += device_sel_q.verify_guards("dflash2_path_select stochastic selector_q");
    return failures;
}

int run_shortlist_remap_case() {
    constexpr std::int32_t vocab          = 32;
    constexpr std::int32_t tokens         = 2;
    constexpr std::int32_t codebook_rows  = ops::kDflash2PathSelectCodebookRows;
    HostWeight host_weight                = make_patterned(kRank, kHidden, 21u);
    DeviceWeight device_weight(std::move(host_weight));

    std::vector<float> logits(static_cast<std::size_t>(vocab) * tokens, -12.0f);
    std::vector<float> hidden(static_cast<std::size_t>(kHidden) * tokens, 0.0f);
    for (std::int32_t t = 0; t < tokens; ++t) {
        logits[static_cast<std::size_t>(t) * vocab + (t + 1)] = 8.0f;
    }
    round_to_bf16(logits);
    round_to_bf16(hidden);
    std::vector<std::int32_t> token_ids(static_cast<std::size_t>(vocab));
    for (std::int32_t v = 0; v < vocab; ++v) { token_ids[static_cast<std::size_t>(v)] = 1000 + v; }
    const std::int32_t anchor = 1000;
    std::vector<std::int32_t> anchors{anchor};

    const auto logit_bits  = encode_bf16(logits);
    const auto hidden_bits = encode_bf16(hidden);
    GuardedDeviceBuffer device_logits(logit_bits.size() * sizeof(std::uint16_t));
    GuardedDeviceBuffer device_hidden(hidden_bits.size() * sizeof(std::uint16_t));
    GuardedDeviceBuffer device_pred(static_cast<std::size_t>(kRank) * codebook_rows *
                                    sizeof(std::uint16_t));
    GuardedDeviceBuffer device_succ(static_cast<std::size_t>(kRank) * codebook_rows *
                                    sizeof(std::uint16_t));
    GuardedDeviceBuffer device_ids(token_ids.size() * sizeof(std::int32_t));
    GuardedDeviceBuffer device_anchors(sizeof(std::int32_t));
    GuardedDeviceBuffer device_path(static_cast<std::size_t>(tokens) * sizeof(std::int32_t));
    device_logits.copy_from_host(logit_bits.data(), device_logits.bytes());
    device_hidden.copy_from_host(hidden_bits.data(), device_hidden.bytes());
    device_pred.fill(0);
    device_succ.fill(0);
    device_ids.copy_from_host(token_ids.data(), device_ids.bytes());
    device_anchors.copy_from_host(&anchor, sizeof(anchor));
    device_path.fill(0xcd);

    Tensor logits_t(device_logits.data(), DType::BF16, {vocab, tokens});
    Tensor hidden_t(device_hidden.data(), DType::BF16, {kHidden, tokens});
    Tensor pred_t(device_pred.data(), DType::BF16, {kRank, codebook_rows});
    Tensor succ_t(device_succ.data(), DType::BF16, {kRank, codebook_rows});
    Tensor ids_t(device_ids.data(), DType::I32, {vocab});
    Tensor anchors_t(device_anchors.data(), DType::I32, {1});
    Tensor path_t(device_path.data(), DType::I32, {tokens});
    WorkspaceArena workspace(std::max<std::size_t>(
        256, ops::dflash2_path_select_workspace_capacity_bytes(QType::BF16_CTRL, tokens, tokens, 1)));
    call_path_select(logits_t, hidden_t, device_weight.view(), pred_t, succ_t, anchors_t, 1, 0.0f,
                     0ull, path_t, workspace, nullptr, &ids_t);
    cuda_synchronize();

    const auto got = from_device<std::int32_t>(device_path.data(), static_cast<std::size_t>(tokens));
    if (got[0] != 1001 || got[1] != 1002) {
        std::cerr << "dflash2_path_select shortlist remap: expected 1001,1002 got " << got[0]
                  << "," << got[1] << "\n";
        return 1;
    }
    return 0;
}

int run_nan_logits_shortlist_case() {
    constexpr std::int32_t vocab         = 32;
    constexpr std::int32_t tokens        = 4;
    constexpr std::int32_t codebook_rows = ops::kDflash2PathSelectCodebookRows;
    HostWeight host_weight               = make_patterned(kRank, kHidden, 27u);
    DeviceWeight device_weight(std::move(host_weight));

    std::vector<float> logits(static_cast<std::size_t>(vocab) * tokens,
                              std::numeric_limits<float>::quiet_NaN());
    std::vector<float> hidden(static_cast<std::size_t>(kHidden) * tokens, 0.0f);
    round_to_bf16(logits);
    round_to_bf16(hidden);
    std::vector<std::int32_t> token_ids(static_cast<std::size_t>(vocab));
    for (std::int32_t v = 0; v < vocab; ++v) { token_ids[static_cast<std::size_t>(v)] = 1000 + v; }
    const std::int32_t anchor = 1000;
    std::vector<std::int32_t> anchors{anchor};

    const auto logit_bits  = encode_bf16(logits);
    const auto hidden_bits = encode_bf16(hidden);
    GuardedDeviceBuffer device_logits(logit_bits.size() * sizeof(std::uint16_t));
    GuardedDeviceBuffer device_hidden(hidden_bits.size() * sizeof(std::uint16_t));
    GuardedDeviceBuffer device_pred(static_cast<std::size_t>(kRank) * codebook_rows *
                                    sizeof(std::uint16_t));
    GuardedDeviceBuffer device_succ(static_cast<std::size_t>(kRank) * codebook_rows *
                                    sizeof(std::uint16_t));
    GuardedDeviceBuffer device_ids(token_ids.size() * sizeof(std::int32_t));
    GuardedDeviceBuffer device_anchors(sizeof(std::int32_t));
    GuardedDeviceBuffer device_path(static_cast<std::size_t>(tokens) * sizeof(std::int32_t));
    device_logits.copy_from_host(logit_bits.data(), device_logits.bytes());
    device_hidden.copy_from_host(hidden_bits.data(), device_hidden.bytes());
    device_pred.fill(0);
    device_succ.fill(0);
    device_ids.copy_from_host(token_ids.data(), device_ids.bytes());
    device_anchors.copy_from_host(&anchor, sizeof(anchor));
    device_path.fill(0xcd);

    Tensor logits_t(device_logits.data(), DType::BF16, {vocab, tokens});
    Tensor hidden_t(device_hidden.data(), DType::BF16, {kHidden, tokens});
    Tensor pred_t(device_pred.data(), DType::BF16, {kRank, codebook_rows});
    Tensor succ_t(device_succ.data(), DType::BF16, {kRank, codebook_rows});
    Tensor ids_t(device_ids.data(), DType::I32, {vocab});
    Tensor anchors_t(device_anchors.data(), DType::I32, {1});
    Tensor path_t(device_path.data(), DType::I32, {tokens});
    WorkspaceArena workspace(std::max<std::size_t>(
        256, ops::dflash2_path_select_workspace_capacity_bytes(QType::BF16_CTRL, tokens, tokens, 1)));
    call_path_select(logits_t, hidden_t, device_weight.view(), pred_t, succ_t, anchors_t, 1, 0.0f,
                     0ull, path_t, workspace, nullptr, &ids_t);
    cuda_synchronize();

    const auto got = from_device<std::int32_t>(device_path.data(), static_cast<std::size_t>(tokens));
    for (std::int32_t t = 0; t < tokens; ++t) {
        if (got[static_cast<std::size_t>(t)] < 1000 || got[static_cast<std::size_t>(t)] >= 1000 + vocab) {
            std::cerr << "dflash2_path_select NaN logits: path token out of shortlist map\n";
            return 1;
        }
    }
    return device_path.verify_guards("dflash2_path_select NaN logits shortlist");
}

int run_tree_layout_case(std::int32_t width) {
    constexpr std::int32_t vocab  = 32;
    constexpr std::int32_t tokens = 2;
    constexpr std::int32_t e      = 40;
    HostWeight host_weight        = make_patterned(kRank, kHidden, 21u);
    DeviceWeight device_weight(std::move(host_weight));

    std::vector<float> logits(static_cast<std::size_t>(vocab) * tokens, -12.0f);
    std::vector<float> hidden(static_cast<std::size_t>(kHidden) * tokens, 0.0f);
    std::vector<float> pred(static_cast<std::size_t>(kRank) * vocab, 0.0f);
    std::vector<float> succ(static_cast<std::size_t>(kRank) * vocab, 0.0f);
    for (std::int32_t t = 0; t < tokens; ++t) {
        for (std::int32_t c = 0; c < kTopK; ++c) {
            logits[static_cast<std::size_t>(t) * vocab + c] = 8.0f - 0.1f * static_cast<float>(c);
        }
    }
    round_to_bf16(logits);
    round_to_bf16(hidden);
    const std::int32_t anchor = 3;
    std::vector<std::int32_t> anchors{anchor};
    std::vector<std::int32_t> frontiers{e};

    const auto logit_bits  = encode_bf16(logits);
    const auto hidden_bits = encode_bf16(hidden);
    const auto pred_bits   = encode_bf16(pred);
    const auto succ_bits   = encode_bf16(succ);
    GuardedDeviceBuffer device_logits(logit_bits.size() * sizeof(std::uint16_t));
    GuardedDeviceBuffer device_hidden(hidden_bits.size() * sizeof(std::uint16_t));
    GuardedDeviceBuffer device_pred(pred_bits.size() * sizeof(std::uint16_t));
    GuardedDeviceBuffer device_succ(succ_bits.size() * sizeof(std::uint16_t));
    GuardedDeviceBuffer device_anchors(sizeof(std::int32_t));
    GuardedDeviceBuffer device_frontiers(sizeof(std::int32_t));
    GuardedDeviceBuffer device_ids(static_cast<std::size_t>(width) * sizeof(std::int32_t));
    GuardedDeviceBuffer device_parent(static_cast<std::size_t>(width) * sizeof(std::int32_t));
    GuardedDeviceBuffer device_cache(static_cast<std::size_t>(width) * sizeof(std::int32_t));
    GuardedDeviceBuffer device_rope(static_cast<std::size_t>(width) * sizeof(std::int32_t));
    GuardedDeviceBuffer device_mask(static_cast<std::size_t>(width) * sizeof(std::int32_t));
    GuardedDeviceBuffer device_valid(sizeof(std::int32_t));
    device_logits.copy_from_host(logit_bits.data(), device_logits.bytes());
    device_hidden.copy_from_host(hidden_bits.data(), device_hidden.bytes());
    device_pred.copy_from_host(pred_bits.data(), device_pred.bytes());
    device_succ.copy_from_host(succ_bits.data(), device_succ.bytes());
    device_anchors.copy_from_host(&anchor, sizeof(anchor));
    device_frontiers.copy_from_host(&e, sizeof(e));

    Tensor logits_t(device_logits.data(), DType::BF16, {vocab, tokens});
    Tensor hidden_t(device_hidden.data(), DType::BF16, {kHidden, tokens});
    Tensor pred_t(device_pred.data(), DType::BF16, {kRank, vocab});
    Tensor succ_t(device_succ.data(), DType::BF16, {kRank, vocab});
    Tensor anchors_t(device_anchors.data(), DType::I32, {1});
    Tensor frontiers_t(device_frontiers.data(), DType::I32, {1});
    Tensor ids_t(device_ids.data(), DType::I32, {width, 1});
    Tensor parent_t(device_parent.data(), DType::I32, {width, 1});
    Tensor cache_t(device_cache.data(), DType::I32, {width, 1});
    Tensor rope_t(device_rope.data(), DType::I32, {width, 1});
    Tensor mask_t(device_mask.data(), DType::I32, {width, 1});
    Tensor valid_t(device_valid.data(), DType::I32, {1});
    WorkspaceArena workspace(std::max<std::size_t>(
        256, ops::dflash2_path_select_workspace_capacity_bytes(QType::BF16_CTRL, tokens, tokens, 1)));
    ops::dflash2_tree_select(logits_t, hidden_t, device_weight.view(), pred_t, succ_t, anchors_t,
                             frontiers_t, ids_t, parent_t, cache_t, rope_t, mask_t, valid_t,
                             workspace, nullptr);
    cuda_synchronize();

    const auto ids    = from_device<std::int32_t>(device_ids.data(), width);
    const auto parent = from_device<std::int32_t>(device_parent.data(), width);
    const auto cache  = from_device<std::int32_t>(device_cache.data(), width);
    const auto rope   = from_device<std::int32_t>(device_rope.data(), width);
    const auto mask   = from_device<std::int32_t>(device_mask.data(), width);
    const auto valid  = from_device<std::int32_t>(device_valid.data(), 1);
    int failures      = 0;
    const int live    = valid[0];
    if (live != 1 + ops::kDflash2TreeFrontier * tokens) {
        std::cerr << "dflash2_tree_select: expected live " << (1 + 2 * tokens) << " got " << live
                  << "\n";
        ++failures;
    }
    if (ids[0] != anchor || parent[0] != -1 || cache[0] != e || rope[0] != e || mask[0] != 1) {
        std::cerr << "dflash2_tree_select: invalid packed root\n";
        ++failures;
    }
    for (int j = 0; j < live; ++j) {
        if (cache[static_cast<std::size_t>(j)] != e + j) {
            std::cerr << "dflash2_tree_select: cache slot " << j << "\n";
            ++failures;
        }
        if (j > 0 && (parent[static_cast<std::size_t>(j)] < 0 ||
                      parent[static_cast<std::size_t>(j)] >= j)) {
            std::cerr << "dflash2_tree_select: parent of " << j << " is not prefix-closed\n";
            ++failures;
        }
        if ((mask[static_cast<std::size_t>(j)] & (1 << j)) == 0) {
            std::cerr << "dflash2_tree_select: ancestor mask missing self bit " << j << "\n";
            ++failures;
        }
        if (j > 0) {
            const int p = parent[static_cast<std::size_t>(j)];
            if ((mask[static_cast<std::size_t>(j)] & mask[static_cast<std::size_t>(p)]) !=
                mask[static_cast<std::size_t>(p)]) {
                std::cerr << "dflash2_tree_select: ancestor mask not closed at " << j << "\n";
                ++failures;
            }
        }
    }
    failures += device_ids.verify_guards("dflash2_tree_select ids");
    return failures;
}

int run_tree_compact_case() {
    constexpr std::int32_t vocab  = 32;
    constexpr std::int32_t tokens = 6;
    constexpr std::int32_t width  = ops::kDflash2VerifyWidth;
    constexpr std::int32_t e      = 8;
    HostWeight host_weight        = make_patterned(kRank, kHidden, 23u);
    DeviceWeight device_weight(std::move(host_weight));

    std::vector<float> logits(static_cast<std::size_t>(vocab) * tokens, -12.0f);
    std::vector<float> hidden(static_cast<std::size_t>(kHidden) * tokens, 0.0f);
    std::vector<float> pred(static_cast<std::size_t>(kRank) * vocab, 0.0f);
    std::vector<float> succ(static_cast<std::size_t>(kRank) * vocab, 0.0f);
    for (std::int32_t t = 0; t < tokens; ++t) {
        for (std::int32_t c = 0; c < kTopK; ++c) {
            logits[static_cast<std::size_t>(t) * vocab + c] = 8.0f - 0.1f * static_cast<float>(c);
        }
    }
    round_to_bf16(logits);
    round_to_bf16(hidden);
    const std::int32_t anchor = 1;
    GuardedDeviceBuffer device_logits(vocab * tokens * sizeof(std::uint16_t));
    GuardedDeviceBuffer device_hidden(kHidden * tokens * sizeof(std::uint16_t));
    GuardedDeviceBuffer device_pred(kRank * vocab * sizeof(std::uint16_t));
    GuardedDeviceBuffer device_succ(kRank * vocab * sizeof(std::uint16_t));
    GuardedDeviceBuffer device_anchors(sizeof(std::int32_t));
    GuardedDeviceBuffer device_frontiers(sizeof(std::int32_t));
    GuardedDeviceBuffer device_ids(static_cast<std::size_t>(width) * sizeof(std::int32_t));
    GuardedDeviceBuffer device_parent(static_cast<std::size_t>(width) * sizeof(std::int32_t));
    GuardedDeviceBuffer device_cache(static_cast<std::size_t>(width) * sizeof(std::int32_t));
    GuardedDeviceBuffer device_rope(static_cast<std::size_t>(width) * sizeof(std::int32_t));
    GuardedDeviceBuffer device_mask(static_cast<std::size_t>(width) * sizeof(std::int32_t));
    GuardedDeviceBuffer device_valid(sizeof(std::int32_t));
    const auto logit_bits  = encode_bf16(logits);
    const auto hidden_bits = encode_bf16(hidden);
    const auto pred_bits   = encode_bf16(pred);
    const auto succ_bits   = encode_bf16(succ);
    device_logits.copy_from_host(logit_bits.data(), device_logits.bytes());
    device_hidden.copy_from_host(hidden_bits.data(), device_hidden.bytes());
    device_pred.copy_from_host(pred_bits.data(), device_pred.bytes());
    device_succ.copy_from_host(succ_bits.data(), device_succ.bytes());
    device_anchors.copy_from_host(&anchor, sizeof(anchor));
    device_frontiers.copy_from_host(&e, sizeof(e));

    Tensor logits_t(device_logits.data(), DType::BF16, {vocab, tokens});
    Tensor hidden_t(device_hidden.data(), DType::BF16, {kHidden, tokens});
    Tensor pred_t(device_pred.data(), DType::BF16, {kRank, vocab});
    Tensor succ_t(device_succ.data(), DType::BF16, {kRank, vocab});
    Tensor anchors_t(device_anchors.data(), DType::I32, {1});
    Tensor frontiers_t(device_frontiers.data(), DType::I32, {1});
    Tensor ids_t(device_ids.data(), DType::I32, {width, 1});
    Tensor parent_t(device_parent.data(), DType::I32, {width, 1});
    Tensor cache_t(device_cache.data(), DType::I32, {width, 1});
    Tensor rope_t(device_rope.data(), DType::I32, {width, 1});
    Tensor mask_t(device_mask.data(), DType::I32, {width, 1});
    Tensor valid_t(device_valid.data(), DType::I32, {1});
    WorkspaceArena workspace(std::max<std::size_t>(
        256, ops::dflash2_path_select_workspace_capacity_bytes(QType::BF16_CTRL, tokens, tokens, 1)));
    ops::dflash2_tree_select(logits_t, hidden_t, device_weight.view(), pred_t, succ_t, anchors_t,
                             frontiers_t, ids_t, parent_t, cache_t, rope_t, mask_t, valid_t,
                             workspace, nullptr);
    cuda_synchronize();

    const auto parent = from_device<std::int32_t>(device_parent.data(), width);
    const auto cache  = from_device<std::int32_t>(device_cache.data(), width);
    const auto valid  = from_device<std::int32_t>(device_valid.data(), 1);
    const int live    = valid[0];
    int failures      = 0;
    if (live != width) {
        std::cerr << "dflash2_tree_select compact: expected live " << width << " got " << live
                  << "\n";
        ++failures;
    }
    if (live >= 3 && (parent[1] != 0 || parent[2] != 0)) {
        std::cerr << "dflash2_tree_select compact: BFS prefix dropped a depth-1 sibling\n";
        ++failures;
    }
    for (int j = 1; j < live; ++j) {
        if (parent[static_cast<std::size_t>(j)] < 0 ||
            parent[static_cast<std::size_t>(j)] >= j ||
            cache[static_cast<std::size_t>(j)] != e + j) {
            std::cerr << "dflash2_tree_select compact: invalid packed column " << j << "\n";
            ++failures;
        }
    }
    return failures;
}

int run_nvfp4_codebook_greedy_case(const char* label, std::int32_t tokens, std::int32_t batch,
                                  std::uint32_t seed) {
    constexpr std::int32_t vocab         = 128;
    constexpr std::int32_t codebook_rows = 128;
    HostWeight host_weight               = make_patterned(kRank, kHidden, seed);
    DeviceWeight device_weight(std::move(host_weight));

    quantized_weight::PatternedWeightOptions options;
    options.weight_scale_divisor = 0.125F;
    options.input_scale_divisor  = 1.0F;
    input_projection::DevicePackedWeight pred_packed(
        quantized_weight::make_patterned_weight(QType::NVFP4, codebook_rows, kRank, seed + 6,
                                                options));
    input_projection::DevicePackedWeight succ_packed(
        quantized_weight::make_patterned_weight(QType::NVFP4, codebook_rows, kRank, seed + 8,
                                                options));

    std::vector<float> logits(static_cast<std::size_t>(vocab) * tokens * batch);
    std::vector<float> hidden(static_cast<std::size_t>(kHidden) * tokens * batch);
    std::vector<float> pred(static_cast<std::size_t>(kRank) * codebook_rows);
    std::vector<float> succ(static_cast<std::size_t>(kRank) * codebook_rows);
    std::vector<std::int32_t> anchors(static_cast<std::size_t>(batch));
    fill_uniform(logits, seed + 10, -8.0f, 8.0f);
    fill_uniform(hidden, seed + 12, -1.0f, 1.0f);
    round_to_bf16(logits);
    round_to_bf16(hidden);
    for (std::int32_t b = 0; b < batch; ++b) {
        anchors[static_cast<std::size_t>(b)] = (19 + b * 11) % vocab;
    }
    for (std::int32_t token = 0; token < codebook_rows; ++token) {
        for (std::int32_t rank = 0; rank < kRank; ++rank) {
            const std::size_t index =
                static_cast<std::size_t>(token) * kRank + static_cast<std::size_t>(rank);
            pred[index] = static_cast<float>(
                quantized_weight::logical_weight_fp64(pred_packed.host, token, rank));
            succ[index] = static_cast<float>(
                quantized_weight::logical_weight_fp64(succ_packed.host, token, rank));
        }
    }
    round_to_bf16(pred);
    round_to_bf16(succ);

    const auto expected =
        path_oracle(logits, hidden, device_weight.host, pred, succ, anchors, vocab, tokens, batch,
                    codebook_rows);

    const auto logit_bits  = encode_bf16(logits);
    const auto hidden_bits = encode_bf16(hidden);
    GuardedDeviceBuffer device_logits(logit_bits.size() * sizeof(std::uint16_t));
    GuardedDeviceBuffer device_hidden(hidden_bits.size() * sizeof(std::uint16_t));
    GuardedDeviceBuffer device_anchors(anchors.size() * sizeof(std::int32_t));
    GuardedDeviceBuffer device_path(static_cast<std::size_t>(tokens) * batch *
                                    sizeof(std::int32_t));
    device_logits.copy_from_host(logit_bits.data(), device_logits.bytes());
    device_hidden.copy_from_host(hidden_bits.data(), device_hidden.bytes());
    device_anchors.copy_from_host(anchors.data(), device_anchors.bytes());
    device_path.fill(0xcd);

    Tensor logits_t =
        batch == 1 ? Tensor(device_logits.data(), DType::BF16, {vocab, tokens})
                   : Tensor(device_logits.data(), DType::BF16, {vocab, tokens, batch});
    Tensor hidden_t =
        batch == 1 ? Tensor(device_hidden.data(), DType::BF16, {kHidden, tokens})
                   : Tensor(device_hidden.data(), DType::BF16, {kHidden, tokens, batch});
    Tensor dummy_code;
    Tensor anchors_t(device_anchors.data(), DType::I32, {batch});
    Tensor path_t = batch == 1 ? Tensor(device_path.data(), DType::I32, {tokens})
                               : Tensor(device_path.data(), DType::I32, {tokens, batch});
    Weight pred_w = pred_packed.view();
    Weight succ_w = succ_packed.view();

    const std::size_t workspace_bytes = ops::dflash2_path_select_workspace_capacity_bytes(
        QType::BF16_CTRL, tokens, tokens, batch);
    WorkspaceArena workspace(std::max<std::size_t>(256, workspace_bytes));
    call_path_select(logits_t, hidden_t, device_weight.view(), dummy_code, dummy_code, anchors_t,
                     batch, 0.0f, 0ull, path_t, workspace, nullptr, nullptr, &pred_w, &succ_w);
    cuda_synchronize();

    int failures = verify_exact(label,
                                from_device<std::int32_t>(device_path.data(),
                                                          static_cast<std::size_t>(tokens) * batch),
                                expected);
    failures += device_path.verify_guards("dflash2_path_select NVFP4 codebook path");
    failures += pred_packed.verify_preserved("dflash2_path_select NVFP4 pred codebook");
    failures += succ_packed.verify_preserved("dflash2_path_select NVFP4 succ codebook");
    return failures;
}

int run_mixed_temperature_batch_case() {
    constexpr std::int32_t vocab  = 32;
    constexpr std::int32_t tokens = 2;
    constexpr std::int32_t batch  = 2;
    HostWeight host_weight        = make_patterned(kRank, kHidden, 61u);
    DeviceWeight device_weight(std::move(host_weight));

    std::vector<float> logits(static_cast<std::size_t>(vocab) * tokens * batch, -12.0f);
    std::vector<float> hidden(static_cast<std::size_t>(kHidden) * tokens * batch, 0.0f);
    std::vector<float> pred(static_cast<std::size_t>(kRank) * vocab, 0.0f);
    std::vector<float> succ(static_cast<std::size_t>(kRank) * vocab, 0.0f);
    for (std::int32_t b = 0; b < batch; ++b) {
        for (std::int32_t t = 0; t < tokens; ++t) {
            const std::size_t col =
                (static_cast<std::size_t>(b) * tokens + t) * static_cast<std::size_t>(vocab);
            for (std::int32_t c = 0; c < kTopK; ++c) {
                logits[col + static_cast<std::size_t>(c)] =
                    8.0f - 0.05f * static_cast<float>(c) - 0.2f * static_cast<float>(b);
            }
        }
    }
    round_to_bf16(logits);
    round_to_bf16(hidden);
    round_to_bf16(pred);
    round_to_bf16(succ);
    std::vector<std::int32_t> anchors{3, 5};

    const auto expected =
        path_oracle(logits, hidden, device_weight.host, pred, succ, anchors, vocab, tokens, batch,
                    vocab);

    const auto logit_bits  = encode_bf16(logits);
    const auto hidden_bits = encode_bf16(hidden);
    const auto pred_bits   = encode_bf16(pred);
    const auto succ_bits   = encode_bf16(succ);
    GuardedDeviceBuffer device_logits(logit_bits.size() * sizeof(std::uint16_t));
    GuardedDeviceBuffer device_hidden(hidden_bits.size() * sizeof(std::uint16_t));
    GuardedDeviceBuffer device_pred(pred_bits.size() * sizeof(std::uint16_t));
    GuardedDeviceBuffer device_succ(succ_bits.size() * sizeof(std::uint16_t));
    GuardedDeviceBuffer device_anchors(anchors.size() * sizeof(std::int32_t));
    GuardedDeviceBuffer device_path(static_cast<std::size_t>(tokens) * batch *
                                    sizeof(std::int32_t));
    device_logits.copy_from_host(logit_bits.data(), device_logits.bytes());
    device_hidden.copy_from_host(hidden_bits.data(), device_hidden.bytes());
    device_pred.copy_from_host(pred_bits.data(), device_pred.bytes());
    device_succ.copy_from_host(succ_bits.data(), device_succ.bytes());
    device_anchors.copy_from_host(anchors.data(), device_anchors.bytes());
    device_path.fill(0xcd);

    Tensor logits_t(device_logits.data(), DType::BF16, {vocab, tokens, batch});
    Tensor hidden_t(device_hidden.data(), DType::BF16, {kHidden, tokens, batch});
    Tensor pred_t(device_pred.data(), DType::BF16, {kRank, vocab});
    Tensor succ_t(device_succ.data(), DType::BF16, {kRank, vocab});
    Tensor anchors_t(device_anchors.data(), DType::I32, {batch});
    const std::array<std::int32_t, 2> logical_positions{100, 200};
    GuardedDeviceBuffer device_positions(logical_positions.size() * sizeof(std::int32_t));
    device_positions.copy_from_host(logical_positions.data(), device_positions.bytes());
    Tensor positions_t(device_positions.data(), DType::I32, {batch});
    Tensor path_t(device_path.data(), DType::I32, {tokens, batch});
    std::vector<ops::SamplingConfig> configs(2);
    configs[0].temperature = 0.7f;
    configs[0].seed        = 42ull;
    configs[1].temperature = 0.0f;
    configs[1].seed        = 0ull;
    GuardedDeviceBuffer device_configs(configs.size() * sizeof(ops::SamplingConfig));
    device_configs.copy_from_host(configs.data(), device_configs.bytes());
    WorkspaceArena workspace(std::max<std::size_t>(
        256, ops::dflash2_path_select_workspace_capacity_bytes(QType::BF16_CTRL, tokens, tokens,
                                                               batch)));
    ops::dflash2_path_select(logits_t, hidden_t, device_weight.view(), pred_t, succ_t, anchors_t,
                              positions_t,
                              reinterpret_cast<const ops::SamplingConfig*>(device_configs.data()),
                             path_t, workspace, nullptr);
    cuda_synchronize();

    const auto got = from_device<std::int32_t>(device_path.data(),
                                               static_cast<std::size_t>(tokens) * batch);
    for (std::int32_t t = 0; t < tokens; ++t) {
        const std::int32_t sampled = got[static_cast<std::size_t>(t)];
        const std::int32_t greedy  = got[static_cast<std::size_t>(tokens + t)];
        if (sampled < 0 || sampled >= kTopK) {
            std::cerr << "dflash2_path_select mixed-temp: sampled row token outside top-16\n";
            return 1;
        }
        if (greedy != expected[static_cast<std::size_t>(tokens + t)]) {
            std::cerr << "dflash2_path_select mixed-temp: greedy row used another row's temperature\n";
            return 1;
        }
    }
    return device_path.verify_guards("dflash2_path_select mixed-temp path");
}

int run_nvfp4_projection_oracle() {
    constexpr std::int32_t vocab  = 64;
    constexpr std::int32_t tokens = 4;
    quantized_weight::PatternedWeightOptions options;
    options.weight_scale_divisor = 0.125F;
    options.input_scale_divisor  = 1.0F;
    input_projection::DevicePackedWeight device_weight(quantized_weight::make_patterned_weight(
        QType::NVFP4, kRank, kHidden, 41u, options));

    std::vector<float> logits(static_cast<std::size_t>(vocab) * tokens);
    std::vector<float> hidden(static_cast<std::size_t>(kHidden) * tokens);
    std::vector<float> pred(static_cast<std::size_t>(kRank) * vocab);
    std::vector<float> succ(static_cast<std::size_t>(kRank) * vocab);
    std::vector<std::int32_t> anchors{7};
    fill_uniform(logits, 43u, -8.0f, 8.0f);
    for (std::int32_t t = 0; t < tokens; ++t) {
        const std::size_t col = static_cast<std::size_t>(t) * vocab;
        for (std::int32_t v = 0; v < vocab; ++v) { logits[col + v] = -12.0f; }
        logits[col + static_cast<std::size_t>((t * 3 + 5) % vocab)] = 8.0f;
    }
    fill_uniform(hidden, 47u, -1.0f, 1.0f);
    fill_uniform(pred, 53u, -0.5f, 0.5f);
    fill_uniform(succ, 59u, -0.5f, 0.5f);
    round_to_bf16(logits);
    round_to_bf16(hidden);
    round_to_bf16(pred);
    round_to_bf16(succ);

    std::vector<std::int32_t> expected(static_cast<std::size_t>(tokens));
    int prev = anchors[0];
    for (std::int32_t t = 0; t < tokens; ++t) {
        std::vector<float> top_val(static_cast<std::size_t>(kTopK),
                                   -std::numeric_limits<float>::infinity());
        std::vector<int> top_idx(static_cast<std::size_t>(kTopK), 0x7fffffff);
        const std::size_t logit_col = static_cast<std::size_t>(t) * vocab;
        for (std::int32_t v = 0; v < vocab; ++v) {
            insert_topk(top_val, top_idx, logits[logit_col + v], v);
        }
        std::vector<float> column(static_cast<std::size_t>(kHidden));
        const std::size_t hid_col = static_cast<std::size_t>(t) * kHidden;
        for (std::int32_t d = 0; d < kHidden; ++d) {
            column[static_cast<std::size_t>(d)] = hidden[hid_col + d];
        }
        std::vector<double> h(static_cast<std::size_t>(kRank));
        for (std::int32_t r = 0; r < kRank; ++r) {
            h[static_cast<std::size_t>(r)] =
                quantized_weight::dot_fp64(device_weight.host, r, column.data(), kHidden);
        }
        int best_id    = top_idx[0];
        double best_sc = 0.0;
        for (std::int32_t c = 0; c < kTopK; ++c) {
            const int cand = top_idx[static_cast<std::size_t>(c)];
            double acc     = static_cast<double>(top_val[static_cast<std::size_t>(c)]);
            const std::size_t pred_row = static_cast<std::size_t>(prev) * kRank;
            const std::size_t succ_row = static_cast<std::size_t>(cand) * kRank;
            for (std::int32_t r = 0; r < kRank; ++r) {
                acc += (static_cast<double>(pred[pred_row + r]) * h[static_cast<std::size_t>(r)]) *
                       static_cast<double>(succ[succ_row + r]);
            }
            if (c == 0 || acc > best_sc || (acc == best_sc && cand < best_id)) {
                best_sc = acc;
                best_id = cand;
            }
        }
        expected[static_cast<std::size_t>(t)] = best_id;
        prev                                  = best_id;
    }

    const auto logit_bits  = encode_bf16(logits);
    const auto hidden_bits = encode_bf16(hidden);
    const auto pred_bits   = encode_bf16(pred);
    const auto succ_bits   = encode_bf16(succ);
    GuardedDeviceBuffer device_logits(logit_bits.size() * sizeof(std::uint16_t));
    GuardedDeviceBuffer device_hidden(hidden_bits.size() * sizeof(std::uint16_t));
    GuardedDeviceBuffer device_pred(pred_bits.size() * sizeof(std::uint16_t));
    GuardedDeviceBuffer device_succ(succ_bits.size() * sizeof(std::uint16_t));
    GuardedDeviceBuffer device_anchors(sizeof(std::int32_t));
    GuardedDeviceBuffer device_path(static_cast<std::size_t>(tokens) * sizeof(std::int32_t));
    device_logits.copy_from_host(logit_bits.data(), device_logits.bytes());
    device_hidden.copy_from_host(hidden_bits.data(), device_hidden.bytes());
    device_pred.copy_from_host(pred_bits.data(), device_pred.bytes());
    device_succ.copy_from_host(succ_bits.data(), device_succ.bytes());
    device_anchors.copy_from_host(anchors.data(), sizeof(std::int32_t));
    device_path.fill(0xcd);

    Tensor logits_t(device_logits.data(), DType::BF16, {vocab, tokens});
    Tensor hidden_t(device_hidden.data(), DType::BF16, {kHidden, tokens});
    Tensor pred_t(device_pred.data(), DType::BF16, {kRank, vocab});
    Tensor succ_t(device_succ.data(), DType::BF16, {kRank, vocab});
    Tensor anchors_t(device_anchors.data(), DType::I32, {1});
    Tensor path_t(device_path.data(), DType::I32, {tokens});
    const std::size_t workspace_bytes = ops::dflash2_path_select_workspace_capacity_bytes(
        QType::NVFP4, tokens, tokens, 1);
    WorkspaceArena workspace(std::max<std::size_t>(256, workspace_bytes));
    call_path_select(logits_t, hidden_t, device_weight.view(), pred_t, succ_t, anchors_t, 1, 0.0f,
                     0ull, path_t, workspace, nullptr);
    cuda_synchronize();

    int failures = verify_exact("dflash2_path_select NVFP4 projection greedy T=4 V=64",
                                from_device<std::int32_t>(device_path.data(), tokens), expected);
    failures += device_path.verify_guards("dflash2_path_select NVFP4 projection path");
    failures += device_weight.verify_preserved("dflash2_path_select NVFP4 projection");
    return failures;
}

std::uint8_t encode_e4m3fn_nonneg(float value) {
    if (!(value > 0.0f)) { return 0; }
    std::uint8_t best = 0;
    float best_err    = value;
    for (int word = 1; word <= 0x7e; ++word) {
        const float decoded =
            static_cast<float>(quantized_weight::detail::decode_e4m3fn(static_cast<std::uint8_t>(word)));
        const float err = std::fabs(decoded - value);
        if (err < best_err) {
            best     = static_cast<std::uint8_t>(word);
            best_err = err;
        }
    }
    return best;
}

std::uint8_t encode_e2m1_nibble(float value) {
    constexpr float kMags[8] = {0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f};
    const float ax           = std::fabs(value);
    int best                 = 0;
    float best_err           = std::fabs(kMags[0] - ax);
    for (int i = 1; i < 8; ++i) {
        const float err = std::fabs(kMags[i] - ax);
        if (err < best_err) {
            best     = i;
            best_err = err;
        }
    }
    return static_cast<std::uint8_t>(best | (value < 0.0f ? 0x08 : 0));
}

quantized_weight::PackedWeight pack_dflash2_nvfp4_codebook(const std::vector<float>& token_major,
                                                           std::int32_t rows) {
    constexpr float kE2M1Max = 6.0f;
    constexpr float kE4M3Max = 448.0f;
    if (rows <= 0 || (rows % 128) != 0 ||
        token_major.size() != static_cast<std::size_t>(rows) * kRank) {
        throw std::invalid_argument("DFlash2 NVFP4 codebook pack: expected [rows,256] token-major");
    }
    float amax = 0.0f;
    for (float value : token_major) { amax = std::max(amax, std::fabs(value)); }
    const float divisor = amax == 0.0f ? 1.0f : (kE2M1Max * kE4M3Max) / amax;

    quantized_weight::PackedWeight packed;
    packed.code_plane_bytes = static_cast<std::uint64_t>(rows) * kRank / 2;
    packed.scale_plane_offset =
        quantized_weight::detail::align_up_size(static_cast<std::size_t>(packed.code_plane_bytes), 256);
    packed.scale_plane_bytes     = static_cast<std::uint64_t>(rows) * kRank / 16;
    packed.weight_divisor_offset = packed.scale_plane_offset + packed.scale_plane_bytes;
    packed.payload.assign(static_cast<std::size_t>(packed.weight_divisor_offset) + 4U, 0);

    constexpr int kGroup = 16;
    const int k_tiles    = kRank / 64;
    for (std::int32_t row = 0; row < rows; ++row) {
        const std::int32_t row_tile  = row / 128;
        const std::int32_t row_inner = row % 128;
        for (int group = 0; group < kRank / kGroup; ++group) {
            float group_amax = 0.0f;
            for (int i = 0; i < kGroup; ++i) {
                group_amax = std::max(
                    group_amax,
                    std::fabs(token_major[static_cast<std::size_t>(row) * kRank + group * kGroup + i]));
            }
            const float scale_fp32 =
                std::min(kE4M3Max, std::max(0.0f, group_amax * divisor / kE2M1Max));
            const std::uint8_t scale_word = encode_e4m3fn_nonneg(scale_fp32);
            const float decoded =
                static_cast<float>(quantized_weight::detail::decode_e4m3fn(scale_word));
            const float safe              = decoded > 0.0f ? decoded : 1.0f;
            const int scale_tile          = group / 4;
            const int scale_lane          = group % 4;
            packed.payload[packed.scale_plane_offset +
                           static_cast<std::size_t>(row_tile * k_tiles + scale_tile) * 512U +
                           static_cast<std::size_t>(row_inner % 32) * 16U +
                           static_cast<std::size_t>(row_inner / 32) * 4U +
                           static_cast<std::size_t>(scale_lane)] = scale_word;
            for (int i = 0; i < kGroup; i += 2) {
                const float lo =
                    token_major[static_cast<std::size_t>(row) * kRank + group * kGroup + i] *
                    divisor / safe;
                const float hi =
                    token_major[static_cast<std::size_t>(row) * kRank + group * kGroup + i + 1] *
                    divisor / safe;
                packed.payload[static_cast<std::size_t>(row) * (kRank / 2) +
                               static_cast<std::size_t>(group * (kGroup / 2) + i / 2)] =
                    static_cast<std::uint8_t>(encode_e2m1_nibble(lo) |
                                              (encode_e2m1_nibble(hi) << 4));
            }
        }
    }
    quantized_weight::detail::store_u32_le(packed.payload, packed.weight_divisor_offset,
                                           quantized_weight::detail::float_bits(divisor));
    packed.weight.qtype                = QType::NVFP4;
    packed.weight.layout               = QuantLayout::BlockScaleK16M128x4;
    packed.weight.scale_dtype          = DType::FP8_E4M3FN;
    packed.weight.payload              = packed.payload.data();
    packed.weight.payload_bytes        = packed.payload.size();
    packed.weight.qdata                = packed.payload.data();
    packed.weight.scales               = packed.payload.data() + packed.scale_plane_offset;
    packed.weight.group_size           = 16;
    packed.weight.group                = 16;
    packed.weight.ndim                 = 2;
    packed.weight.shape[0]             = rows;
    packed.weight.shape[1]             = kRank;
    packed.weight.shape[2]             = 1;
    packed.weight.shape[3]             = 1;
    packed.weight.padded_shape[0]      = rows;
    packed.weight.padded_shape[1]      = kRank;
    packed.weight.padded_shape[2]      = 1;
    packed.weight.padded_shape[3]      = 1;
    packed.weight.n                    = rows;
    packed.weight.k                    = kRank;
    packed.weight.weight_scale_divisor = divisor;
    packed.weight.input_scale_divisor  = 1.0F;
    return packed;
}

int run_nvfp4_codebook_matches_bf16_accept(const char* label, std::int32_t tokens,
                                           std::int32_t batch, std::uint32_t seed) {
    constexpr std::int32_t vocab = 128;
    quantized_weight::PatternedWeightOptions options;
    options.weight_scale_divisor = 0.125F;
    options.input_scale_divisor  = 1.0F;
    input_projection::DevicePackedWeight projection(quantized_weight::make_patterned_weight(
        QType::NVFP4, kRank, kHidden, seed, options));

    std::vector<float> logits(static_cast<std::size_t>(vocab) * tokens * batch);
    std::vector<float> hidden(static_cast<std::size_t>(kHidden) * tokens * batch);
    std::vector<float> pred(static_cast<std::size_t>(kRank) * vocab);
    std::vector<float> succ(static_cast<std::size_t>(kRank) * vocab);
    std::vector<std::int32_t> anchors(static_cast<std::size_t>(batch));
    fill_uniform(logits, seed + 3, -4.0f, 4.0f);
    fill_uniform(hidden, seed + 5, -1.0f, 1.0f);
    fill_uniform(pred, seed + 7, -0.5f, 0.5f);
    fill_uniform(succ, seed + 9, -0.5f, 0.5f);
    round_to_bf16(logits);
    round_to_bf16(hidden);
    round_to_bf16(pred);
    round_to_bf16(succ);
    for (std::int32_t b = 0; b < batch; ++b) {
        anchors[static_cast<std::size_t>(b)] = (17 + b * 13) % vocab;
    }

    input_projection::DevicePackedWeight pred_q(pack_dflash2_nvfp4_codebook(pred, vocab));
    input_projection::DevicePackedWeight succ_q(pack_dflash2_nvfp4_codebook(succ, vocab));
    Weight pred_w = pred_q.view();
    Weight succ_w = succ_q.view();

    const auto logit_bits  = encode_bf16(logits);
    const auto hidden_bits = encode_bf16(hidden);
    const auto pred_bits   = encode_bf16(pred);
    const auto succ_bits   = encode_bf16(succ);
    GuardedDeviceBuffer device_logits(logit_bits.size() * sizeof(std::uint16_t));
    GuardedDeviceBuffer device_hidden(hidden_bits.size() * sizeof(std::uint16_t));
    GuardedDeviceBuffer device_pred(pred_bits.size() * sizeof(std::uint16_t));
    GuardedDeviceBuffer device_succ(succ_bits.size() * sizeof(std::uint16_t));
    GuardedDeviceBuffer device_anchors(anchors.size() * sizeof(std::int32_t));
    GuardedDeviceBuffer device_bf16(static_cast<std::size_t>(tokens) * batch * sizeof(std::int32_t));
    GuardedDeviceBuffer device_nvfp4(static_cast<std::size_t>(tokens) * batch *
                                     sizeof(std::int32_t));
    device_logits.copy_from_host(logit_bits.data(), device_logits.bytes());
    device_hidden.copy_from_host(hidden_bits.data(), device_hidden.bytes());
    device_pred.copy_from_host(pred_bits.data(), device_pred.bytes());
    device_succ.copy_from_host(succ_bits.data(), device_succ.bytes());
    device_anchors.copy_from_host(anchors.data(), device_anchors.bytes());
    device_bf16.fill(0xcd);
    device_nvfp4.fill(0xcd);

    Tensor logits_t =
        batch == 1 ? Tensor(device_logits.data(), DType::BF16, {vocab, tokens})
                   : Tensor(device_logits.data(), DType::BF16, {vocab, tokens, batch});
    Tensor hidden_t =
        batch == 1 ? Tensor(device_hidden.data(), DType::BF16, {kHidden, tokens})
                   : Tensor(device_hidden.data(), DType::BF16, {kHidden, tokens, batch});
    Tensor pred_t(device_pred.data(), DType::BF16, {kRank, vocab});
    Tensor succ_t(device_succ.data(), DType::BF16, {kRank, vocab});
    Tensor dummy;
    Tensor anchors_t(device_anchors.data(), DType::I32, {batch});
    Tensor bf16_path = batch == 1 ? Tensor(device_bf16.data(), DType::I32, {tokens})
                                  : Tensor(device_bf16.data(), DType::I32, {tokens, batch});
    Tensor nvfp4_path = batch == 1 ? Tensor(device_nvfp4.data(), DType::I32, {tokens})
                                   : Tensor(device_nvfp4.data(), DType::I32, {tokens, batch});
    const std::size_t workspace_bytes =
        ops::dflash2_path_select_workspace_capacity_bytes(QType::NVFP4, tokens, tokens, batch);
    WorkspaceArena workspace(std::max<std::size_t>(256, workspace_bytes));
    call_path_select(logits_t, hidden_t, projection.view(), pred_t, succ_t, anchors_t, batch, 0.0f,
                     0ull, bf16_path, workspace, nullptr);
    call_path_select(logits_t, hidden_t, projection.view(), dummy, dummy, anchors_t, batch, 0.0f,
                     0ull, nvfp4_path, workspace, nullptr, nullptr, &pred_w, &succ_w);
    cuda_synchronize();

    const std::size_t count = static_cast<std::size_t>(tokens) * batch;
    const auto bf16_path_ids  = from_device<std::int32_t>(device_bf16.data(), count);
    const auto nvfp4_path_ids = from_device<std::int32_t>(device_nvfp4.data(), count);
    std::size_t path_mismatch = 0;
    for (std::size_t i = 0; i < count; ++i) {
        if (nvfp4_path_ids[i] != bf16_path_ids[i]) { ++path_mismatch; }
        if (nvfp4_path_ids[i] < 0 || nvfp4_path_ids[i] >= vocab || bf16_path_ids[i] < 0 ||
            bf16_path_ids[i] >= vocab) {
            std::cerr << label << ": path token out of vocab\n";
            return 1;
        }
    }
    std::cerr << label << ": " << (count - path_mismatch) << '/' << count
              << " greedy path tokens match BF16 codebook\n";
    int failures = 0;
    failures += device_bf16.verify_guards("dflash2_path_select BF16 codebook path");
    failures += device_nvfp4.verify_guards("dflash2_path_select NVFP4 codebook path");
    if (batch != 1) { return failures; }

    constexpr std::int32_t width = 6;
    constexpr std::int32_t e     = 40;
    GuardedDeviceBuffer device_frontiers(sizeof(std::int32_t));
    GuardedDeviceBuffer device_ids_bf16(static_cast<std::size_t>(width) * sizeof(std::int32_t));
    GuardedDeviceBuffer device_ids_nvfp4(static_cast<std::size_t>(width) * sizeof(std::int32_t));
    GuardedDeviceBuffer device_parent_bf16(static_cast<std::size_t>(width) * sizeof(std::int32_t));
    GuardedDeviceBuffer device_parent_nvfp4(static_cast<std::size_t>(width) * sizeof(std::int32_t));
    GuardedDeviceBuffer device_cache(static_cast<std::size_t>(width) * sizeof(std::int32_t));
    GuardedDeviceBuffer device_rope(static_cast<std::size_t>(width) * sizeof(std::int32_t));
    GuardedDeviceBuffer device_mask(static_cast<std::size_t>(width) * sizeof(std::int32_t));
    GuardedDeviceBuffer device_valid_bf16(sizeof(std::int32_t));
    GuardedDeviceBuffer device_valid_nvfp4(sizeof(std::int32_t));
    device_frontiers.copy_from_host(&e, sizeof(e));
    Tensor frontiers_t(device_frontiers.data(), DType::I32, {1});
    Tensor ids_bf16_t(device_ids_bf16.data(), DType::I32, {width, 1});
    Tensor ids_nvfp4_t(device_ids_nvfp4.data(), DType::I32, {width, 1});
    Tensor parent_bf16_t(device_parent_bf16.data(), DType::I32, {width, 1});
    Tensor parent_nvfp4_t(device_parent_nvfp4.data(), DType::I32, {width, 1});
    Tensor cache_t(device_cache.data(), DType::I32, {width, 1});
    Tensor rope_t(device_rope.data(), DType::I32, {width, 1});
    Tensor mask_t(device_mask.data(), DType::I32, {width, 1});
    Tensor valid_bf16_t(device_valid_bf16.data(), DType::I32, {1});
    Tensor valid_nvfp4_t(device_valid_nvfp4.data(), DType::I32, {1});
    ops::dflash2_tree_select(logits_t, hidden_t, projection.view(), pred_t, succ_t, anchors_t,
                             frontiers_t, ids_bf16_t, parent_bf16_t, cache_t, rope_t, mask_t,
                             valid_bf16_t, workspace, nullptr);
    ops::dflash2_tree_select(logits_t, hidden_t, projection.view(), dummy, dummy, anchors_t,
                             frontiers_t, ids_nvfp4_t, parent_nvfp4_t, cache_t, rope_t, mask_t,
                             valid_nvfp4_t, workspace, nullptr, nullptr, &pred_w, &succ_w);
    cuda_synchronize();
    const auto tree_bf16  = from_device<std::int32_t>(device_ids_bf16.data(), width);
    const auto tree_nvfp4 = from_device<std::int32_t>(device_ids_nvfp4.data(), width);
    std::size_t tree_mismatch = 0;
    for (std::size_t i = 0; i < static_cast<std::size_t>(width); ++i) {
        if (tree_nvfp4[i] != tree_bf16[i]) { ++tree_mismatch; }
    }
    std::cerr << label << " tree W=6: " << (static_cast<std::size_t>(width) - tree_mismatch) << '/'
              << width << " packed ids match BF16 codebook\n";
    failures += device_ids_bf16.verify_guards("dflash2_tree_select BF16 codebook ids");
    failures += device_ids_nvfp4.verify_guards("dflash2_tree_select NVFP4 codebook ids");
    return failures;
}

// ---------------------------------------------------------------------------
// Content-level tree-select oracle.
//
// The structural tree tests above (run_tree_layout_case / run_tree_compact_case)
// use a zero codebook and a zero hidden, so every (parent, cand) pair scores
// identically and only the *structural* invariants (prefix-closed parent, mask
// closure, cache/rope slots, live count) are exercised. The content of the
// node selection -- which token actually lands in which tree node, via the real
// Markov score -- was never checked. This oracle rebuilds the same beam-2 BFS
// in FP64 on the host and compares the full packed tree.
//
// The inputs are deliberately well-separated so the top-2 frontier choice is
// unambiguous regardless of the kernel's FP32/BF16 accumulation order:
//   pred_code[token, r] = prime[token]          (unique per token, all r)
//   succ_code[token, r] = token + 1             (unique per token, all r)
//   W_h[r, 0] = 1, W_h[r, k>0] = 0  and  hidden[0] = 1, hidden[k>0] = 0
//     => hidden projection h[r] = 1 exactly for every r
// so the Markov score for (parent, cand) is exactly
//   unary[cand] + kRank * prime[parent] * (cand + 1),
// which is strictly separated across (parent, cand) pairs.

struct TreeContentOracle {
    std::vector<std::int32_t> ids;
    std::vector<std::int32_t> parent;
    std::vector<std::int32_t> cache;
    std::vector<std::int32_t> rope;
    std::vector<std::int32_t> mask;
    std::int32_t valid = 0;
};

TreeContentOracle tree_content_oracle(const std::vector<float>& logits,
                                       const std::vector<float>& pred,
                                       const std::vector<float>& succ,
                                       std::int32_t anchor, std::int32_t frontier,
                                       std::int32_t vocab, std::int32_t tokens,
                                       std::int32_t width) {
    const std::int32_t kExpand = ops::kDflash2TreeExpandWidth;
    // node_id[i] = packed column i's token id; node_parent[i] = parent packed column
    // (-1 for the root); node_score[i] = cumulative Markov score of the path.
    std::vector<std::int32_t> node_id;    // token id per packed column
    std::vector<std::int32_t> node_parent;
    std::vector<std::int32_t> node_depth;
    std::vector<double> node_score;
    node_id.push_back(anchor);
    node_parent.push_back(-1);
    node_depth.push_back(0);
    node_score.push_back(0.0);

    auto top16 = [&](std::int32_t t) {
        std::vector<float> tv(kTopK, -std::numeric_limits<float>::infinity());
        std::vector<int> ti(kTopK, 0x7fffffff);
        const std::size_t col = static_cast<std::size_t>(t) * static_cast<std::size_t>(vocab);
        for (std::int32_t v = 0; v < vocab; ++v) {
            insert_topk(tv, ti, logits[col + static_cast<std::size_t>(v)], v);
        }
        return std::pair<std::vector<float>, std::vector<int>>(std::move(tv), std::move(ti));
    };

    // h[r] for each draft column: here the test fixtures force h = 1 exactly, but the
    // oracle computes the Markov sum generically over the kRank ranks so it stays valid
    // if the fixture is relaxed.
    std::vector<std::vector<double>> hidden_proj(tokens, std::vector<double>(kRank, 1.0));

    std::vector<std::int32_t> frontier_nodes{0};
    for (std::int32_t t = 0; t < tokens; ++t) {
        const auto tv_ti = top16(t);
        const auto& top_val = tv_ti.first;
        const auto& top_idx = tv_ti.second;
        struct Pair {
            double joint;
            int cand;
            int pcol;
        };
        std::vector<Pair> pairs;
        pairs.reserve(frontier_nodes.size() * static_cast<std::size_t>(kTopK));
        for (int f : frontier_nodes) {
            const int pcol         = f;
            const double base     = node_score[pcol];
            const int parent_token = node_id[pcol];
            for (int c = 0; c < kTopK; ++c) {
                const int cand        = top_idx[static_cast<std::size_t>(c)];
                const double unary    = static_cast<double>(top_val[static_cast<std::size_t>(c)]);
                double markov         = 0.0;
                for (std::int32_t r = 0; r < kRank; ++r) {
                    markov += (static_cast<double>(
                                    pred[static_cast<std::size_t>(parent_token) * kRank + r]) *
                                hidden_proj[static_cast<std::size_t>(t)][static_cast<std::size_t>(r)]) *
                               static_cast<double>(succ[static_cast<std::size_t>(cand) * kRank + r]);
                }
                pairs.push_back({base + unary + markov, cand, pcol});
            }
        }
        // Top-2 by (joint desc, cand asc, pcol asc) -- matches the kernel tiebreak.
        std::sort(pairs.begin(), pairs.end(), [](const Pair& a, const Pair& b) {
            if (a.joint != b.joint) { return a.joint > b.joint; }
            if (a.cand != b.cand) { return a.cand < b.cand; }
            return a.pcol < b.pcol;
        });
        const int take = std::min<std::size_t>(2, pairs.size());
        std::vector<std::int32_t> next_frontier;
        for (int s = 0; s < take; ++s) {
            const auto& p = pairs[static_cast<std::size_t>(s)];
            if (static_cast<std::int32_t>(node_id.size()) >= kExpand) { break; }
            node_id.push_back(p.cand);
            node_parent.push_back(p.pcol);
            node_depth.push_back(t + 1);
            node_score.push_back(p.joint);
            next_frontier.push_back(static_cast<std::int32_t>(node_id.size() - 1));
        }
        if (!next_frontier.empty()) { frontier_nodes = std::move(next_frontier); }
    }

    const std::int32_t out_n =
        static_cast<std::int32_t>(std::min(node_id.size(), static_cast<std::size_t>(width)));
    TreeContentOracle out;
    out.ids.resize(width);
    out.parent.resize(width);
    out.cache.resize(width);
    out.rope.resize(width);
    out.mask.resize(width);
    const std::int32_t last = out_n > 0 ? out_n - 1 : 0;
    for (std::int32_t i = 0; i < width; ++i) {
        if (i < out_n) {
            out.ids[static_cast<std::size_t>(i)]    = node_id[static_cast<std::size_t>(i)];
            out.parent[static_cast<std::size_t>(i)] = i == 0 ? -1 : node_parent[static_cast<std::size_t>(i)];
            out.cache[static_cast<std::size_t>(i)]  = frontier + i;
            out.rope[static_cast<std::size_t>(i)]   = frontier + node_depth[static_cast<std::size_t>(i)];
            int m   = 0;
            int cur = i;
            while (cur >= 0) {
                m |= 1 << cur;
                cur = node_parent[static_cast<std::size_t>(cur)];
            }
            out.mask[static_cast<std::size_t>(i)] = m;
        } else {
            // Padding columns duplicate the last live node's id/parent/rope/mask, but keep
            // their own cache slot (e + col) -- matches the kernel's output convention.
            out.ids[static_cast<std::size_t>(i)]    = out.ids[static_cast<std::size_t>(last)];
            out.parent[static_cast<std::size_t>(i)] = out.parent[static_cast<std::size_t>(last)];
            out.cache[static_cast<std::size_t>(i)]  = frontier + i;
            out.rope[static_cast<std::size_t>(i)]   = out.rope[static_cast<std::size_t>(last)];
            out.mask[static_cast<std::size_t>(i)]   = out.mask[static_cast<std::size_t>(last)];
        }
    }
    out.valid = out_n;
    return out;
}

int run_tree_content_case(const char* label, std::int32_t tokens, std::int32_t width,
                          std::int32_t anchor, std::int32_t frontier) {
    constexpr std::int32_t vocab = 32;
    // The first 32 primes, so prime[token] is unique per token.
    const std::int32_t prime[vocab] = {
        2,  3,  5,  7,  11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53,
        59, 61, 67, 71, 73, 79, 83, 89, 97, 101, 103, 107, 109, 113, 127, 131,
    };
    std::vector<float> logits(static_cast<std::size_t>(vocab) * tokens, 0.0f);
    std::vector<float> hidden(static_cast<std::size_t>(kHidden) * tokens, 0.0f);
    std::vector<float> pred(static_cast<std::size_t>(kRank) * vocab, 0.0f);
    std::vector<float> succ(static_cast<std::size_t>(kRank) * vocab, 0.0f);
    for (std::int32_t t = 0; t < tokens; ++t) {
        const std::size_t col = static_cast<std::size_t>(t) * static_cast<std::size_t>(vocab);
        for (std::int32_t v = 0; v < vocab; ++v) { logits[col + static_cast<std::size_t>(v)] = static_cast<float>(v); }
        // hidden[0] = 1 for every draft column, rest zero -> h = 1 exactly.
        hidden[static_cast<std::size_t>(t) * kHidden + 0] = 1.0f;
    }
    for (std::int32_t token = 0; token < vocab; ++token) {
        for (std::int32_t r = 0; r < kRank; ++r) {
            pred[static_cast<std::size_t>(token) * kRank + static_cast<std::size_t>(r)] =
                static_cast<float>(prime[token]);
            succ[static_cast<std::size_t>(token) * kRank + static_cast<std::size_t>(r)] =
                static_cast<float>(token + 1);
        }
    }
    round_to_bf16(logits);
    round_to_bf16(hidden);
    round_to_bf16(pred);
    round_to_bf16(succ);

    // W_h = identity-ish: row r has W_h[r,0] = 1, rest 0 -> h[r] = hidden[0] = 1.
    HostWeight host_weight;
    host_weight.n = kRank;
    host_weight.k = kHidden;
    host_weight.bits.assign(static_cast<std::size_t>(kRank) * kHidden, 0);
    for (std::int32_t r = 0; r < kRank; ++r) {
        host_weight.bits[static_cast<std::size_t>(r) * kHidden + 0] = f32_to_bf16(1.0f);
    }
    DeviceWeight device_weight(std::move(host_weight));

    const TreeContentOracle expected =
        tree_content_oracle(logits, pred, succ, anchor, frontier, vocab, tokens, width);

    const auto logit_bits  = encode_bf16(logits);
    const auto hidden_bits = encode_bf16(hidden);
    const auto pred_bits   = encode_bf16(pred);
    const auto succ_bits   = encode_bf16(succ);
    GuardedDeviceBuffer device_logits(logit_bits.size() * sizeof(std::uint16_t));
    GuardedDeviceBuffer device_hidden(hidden_bits.size() * sizeof(std::uint16_t));
    GuardedDeviceBuffer device_pred(pred_bits.size() * sizeof(std::uint16_t));
    GuardedDeviceBuffer device_succ(succ_bits.size() * sizeof(std::uint16_t));
    GuardedDeviceBuffer device_anchors(sizeof(std::int32_t));
    GuardedDeviceBuffer device_frontiers(sizeof(std::int32_t));
    GuardedDeviceBuffer device_ids(static_cast<std::size_t>(width) * sizeof(std::int32_t));
    GuardedDeviceBuffer device_parent(static_cast<std::size_t>(width) * sizeof(std::int32_t));
    GuardedDeviceBuffer device_cache(static_cast<std::size_t>(width) * sizeof(std::int32_t));
    GuardedDeviceBuffer device_rope(static_cast<std::size_t>(width) * sizeof(std::int32_t));
    GuardedDeviceBuffer device_mask(static_cast<std::size_t>(width) * sizeof(std::int32_t));
    GuardedDeviceBuffer device_valid(sizeof(std::int32_t));
    device_logits.copy_from_host(logit_bits.data(), device_logits.bytes());
    device_hidden.copy_from_host(hidden_bits.data(), device_hidden.bytes());
    device_pred.copy_from_host(pred_bits.data(), device_pred.bytes());
    device_succ.copy_from_host(succ_bits.data(), device_succ.bytes());
    device_anchors.copy_from_host(&anchor, sizeof(anchor));
    device_frontiers.copy_from_host(&frontier, sizeof(frontier));

    Tensor logits_t(device_logits.data(), DType::BF16, {vocab, tokens});
    Tensor hidden_t(device_hidden.data(), DType::BF16, {kHidden, tokens});
    Tensor pred_t(device_pred.data(), DType::BF16, {kRank, vocab});
    Tensor succ_t(device_succ.data(), DType::BF16, {kRank, vocab});
    Tensor anchors_t(device_anchors.data(), DType::I32, {1});
    Tensor frontiers_t(device_frontiers.data(), DType::I32, {1});
    Tensor ids_t(device_ids.data(), DType::I32, {width, 1});
    Tensor parent_t(device_parent.data(), DType::I32, {width, 1});
    Tensor cache_t(device_cache.data(), DType::I32, {width, 1});
    Tensor rope_t(device_rope.data(), DType::I32, {width, 1});
    Tensor mask_t(device_mask.data(), DType::I32, {width, 1});
    Tensor valid_t(device_valid.data(), DType::I32, {1});
    WorkspaceArena workspace(std::max<std::size_t>(
        256, ops::dflash2_path_select_workspace_capacity_bytes(QType::BF16_CTRL, tokens, tokens, 1)));
    ops::dflash2_tree_select(logits_t, hidden_t, device_weight.view(), pred_t, succ_t, anchors_t,
                             frontiers_t, ids_t, parent_t, cache_t, rope_t, mask_t, valid_t,
                             workspace, nullptr);
    cuda_synchronize();

    const auto ids    = from_device<std::int32_t>(device_ids.data(), width);
    const auto parent = from_device<std::int32_t>(device_parent.data(), width);
    const auto cache  = from_device<std::int32_t>(device_cache.data(), width);
    const auto rope   = from_device<std::int32_t>(device_rope.data(), width);
    const auto mask   = from_device<std::int32_t>(device_mask.data(), width);
    const auto valid  = from_device<std::int32_t>(device_valid.data(), 1);
    int failures      = 0;
    if (valid[0] != expected.valid) {
        std::cerr << label << ": valid got " << valid[0] << " expected " << expected.valid << "\n";
        ++failures;
    }
    for (std::int32_t i = 0; i < width; ++i) {
        if (i < expected.valid &&
            (ids[static_cast<std::size_t>(i)] != expected.ids[static_cast<std::size_t>(i)] ||
             parent[static_cast<std::size_t>(i)] != expected.parent[static_cast<std::size_t>(i)])) {
            std::cerr << label << ": node " << i << " got id=" << ids[static_cast<std::size_t>(i)]
                      << " parent=" << parent[static_cast<std::size_t>(i)] << " expected id="
                      << expected.ids[static_cast<std::size_t>(i)]
                      << " parent=" << expected.parent[static_cast<std::size_t>(i)] << "\n";
            ++failures;
        }
        if (cache[static_cast<std::size_t>(i)] != expected.cache[static_cast<std::size_t>(i)] ||
            rope[static_cast<std::size_t>(i)] != expected.rope[static_cast<std::size_t>(i)] ||
            mask[static_cast<std::size_t>(i)] != expected.mask[static_cast<std::size_t>(i)]) {
            std::cerr << label << ": cache/rope/mask mismatch at node " << i << "\n";
            ++failures;
        }
    }
    failures += device_ids.verify_guards((std::string(label) + " ids").c_str());
    return failures;
}

} // namespace

int main() {
    if (cuda_unavailable()) {
        std::cerr << "FAIL: no usable CUDA device\n";
        return 1;
    }

    int failures = 0;
    failures +=
        run_greedy_case("dflash2_path_select forced-greedy refinement T=2 V=256", 256, 2, 1, 11u);
    failures += run_greedy_case("dflash2_path_select greedy T=1 V=64", 64, 1, 1, 13u);
    failures += run_greedy_case("dflash2_path_select greedy T=4 B=2 V=64", 64, 4, 2, 17u);
    failures += run_greedy_case("dflash2_path_select greedy T=5 B=2 V=64", 64, 5, 2, 19u);
    failures += run_greedy_case("dflash2_path_select greedy T=4 B=3 V=64", 64, 4, 3, 23u);
    failures += run_greedy_case("dflash2_path_select greedy T=5 B=3 V=64", 64, 5, 3, 29u);
    failures += run_stochastic_support_case();
    failures += run_stochastic_batch_row_invariance_case();
    failures += run_stochastic_selector_q_case();
    failures += run_mixed_temperature_batch_case();
    failures += run_shortlist_remap_case();
    failures += run_nan_logits_shortlist_case();
    failures += run_tree_layout_case(ops::kDflash2VerifyWidth);
    failures += run_tree_layout_case(6);
    failures += run_tree_compact_case();
    failures += run_tree_content_case(
        "dflash2_tree_select content T=4 W=12 anchor=5 e=16", 4, ops::kDflash2VerifyWidth, 5, 16);
    failures += run_tree_content_case(
        "dflash2_tree_select content T=7 W=12 anchor=9 e=24", 7, ops::kDflash2VerifyWidth, 9, 24);
    failures += run_nvfp4_codebook_greedy_case("dflash2_path_select NVFP4 codebook greedy T=2 V=128",
                                              2, 1, 23u);
    failures += run_nvfp4_codebook_greedy_case(
        "dflash2_path_select NVFP4 codebook greedy T=5 B=2 V=128", 5, 2, 29u);
    failures += run_nvfp4_codebook_greedy_case(
        "dflash2_path_select NVFP4 codebook greedy T=5 B=3 V=128", 5, 3, 31u);
    failures += run_nvfp4_projection_oracle();
    failures += run_nvfp4_codebook_matches_bf16_accept(
        "dflash2_path_select NVFP4 W_h codebook BF16 vs NVFP4 greedy T=5 V=128", 5, 1, 41u);
    failures += run_nvfp4_codebook_matches_bf16_accept(
        "dflash2_path_select NVFP4 W_h codebook BF16 vs NVFP4 greedy T=5 B=2 V=128", 5, 2, 43u);
    failures += run_nvfp4_codebook_matches_bf16_accept(
        "dflash2_path_select NVFP4 W_h codebook BF16 vs NVFP4 greedy T=5 B=3 V=128", 5, 3, 47u);
    std::cout << (failures ? "FAIL" : "OK") << " dflash2_path_select\n";
    return failures ? 1 : 0;
}
