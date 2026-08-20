#include "ninfer/ops/dflash2_path_select.h"

#include "ops/direct_bf16_weight.h"
#include "ops/op_tester.h"

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
    if (!logit_better(value, index, vals.back(), idxs.back())) { return; }
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
    device_logits.copy_from_host(logit_bits.data(), device_logits.bytes());
    device_hidden.copy_from_host(hidden_bits.data(), device_hidden.bytes());
    device_pred.copy_from_host(pred_bits.data(), device_pred.bytes());
    device_succ.copy_from_host(succ_bits.data(), device_succ.bytes());
    device_anchors.copy_from_host(anchors.data(), device_anchors.bytes());
    device_path.fill(0xcd);

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

    const std::size_t workspace_bytes = ops::dflash2_path_select_workspace_capacity_bytes(
        QType::BF16_CTRL, tokens, tokens, batch);
    WorkspaceArena workspace(std::max<std::size_t>(256, workspace_bytes));
    ops::dflash2_path_select(logits_t, hidden_t, device_weight.view(), pred_t, succ_t, anchors_t,
                             0.0f, 0ull, path_t, workspace, nullptr);
    cuda_synchronize();

    int failures = verify_exact(label, from_device<std::int32_t>(device_path.data(),
                                                               static_cast<std::size_t>(tokens) *
                                                                   batch),
                                expected);
    failures += device_path.verify_guards("dflash2_path_select path");
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
    ops::dflash2_path_select(logits_t, hidden_t, device_weight.view(), pred_t, succ_t, anchors_t,
                             0.6f, 42ull, path_t, workspace, nullptr);
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
    ops::dflash2_path_select(logits_t, hidden_t, device_weight.view(), pred_t, succ_t, anchors_t,
                             0.0f, 0ull, path_t, workspace, nullptr, &ids_t);
    cuda_synchronize();

    const auto got = from_device<std::int32_t>(device_path.data(), static_cast<std::size_t>(tokens));
    if (got[0] != 1001 || got[1] != 1002) {
        std::cerr << "dflash2_path_select shortlist remap: expected 1001,1002 got " << got[0]
                  << "," << got[1] << "\n";
        return 1;
    }
    return 0;
}

} // namespace

int main() {
    if (cuda_unavailable()) {
        std::cout << "SKIP: no usable CUDA device\n";
        return 77;
    }

    int failures = 0;
    failures += run_greedy_case("dflash2_path_select greedy T=2 V=256", 256, 2, 1, 11u);
    failures += run_greedy_case("dflash2_path_select greedy T=1 V=64", 64, 1, 1, 13u);
    failures += run_greedy_case("dflash2_path_select greedy T=4 B=2 V=64", 64, 4, 2, 17u);
    failures += run_stochastic_support_case();
    failures += run_shortlist_remap_case();
    std::cout << (failures ? "FAIL" : "OK") << " dflash2_path_select\n";
    return failures ? 1 : 0;
}
