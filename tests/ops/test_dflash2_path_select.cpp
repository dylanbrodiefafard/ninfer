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
                      const Weight* succ_nvfp4 = nullptr) {
    std::vector<float> temperatures(static_cast<std::size_t>(batch), temperature);
    std::vector<unsigned long long> seeds(static_cast<std::size_t>(batch), seed);
    ops::dflash2_path_select(logits, hidden, weight, pred, succ, anchors, temperatures.data(),
                             seeds.data(), path, workspace, stream, logit_token_ids, pred_nvfp4,
                             succ_nvfp4);
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
    call_path_select(logits_t, hidden_t, device_weight.view(), pred_t, succ_t, anchors_t, batch,
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
    Tensor path_t(device_path.data(), DType::I32, {tokens, batch});
    const float temperatures[2]              = {0.7f, 0.0f};
    const unsigned long long seeds[2]        = {42ull, 0ull};
    WorkspaceArena workspace(std::max<std::size_t>(
        256, ops::dflash2_path_select_workspace_capacity_bytes(QType::BF16_CTRL, tokens, tokens,
                                                               batch)));
    ops::dflash2_path_select(logits_t, hidden_t, device_weight.view(), pred_t, succ_t, anchors_t,
                             temperatures, seeds, path_t, workspace, nullptr);
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

} // namespace

int main() {
    if (cuda_unavailable()) {
        std::cerr << "FAIL: no usable CUDA device\n";
        return 1;
    }

    int failures = 0;
    failures += run_greedy_case("dflash2_path_select greedy T=2 V=256", 256, 2, 1, 11u);
    failures += run_greedy_case("dflash2_path_select greedy T=1 V=64", 64, 1, 1, 13u);
    failures += run_greedy_case("dflash2_path_select greedy T=4 B=2 V=64", 64, 4, 2, 17u);
    failures += run_greedy_case("dflash2_path_select greedy T=5 B=2 V=64", 64, 5, 2, 19u);
    failures += run_greedy_case("dflash2_path_select greedy T=4 B=3 V=64", 64, 4, 3, 23u);
    failures += run_greedy_case("dflash2_path_select greedy T=5 B=3 V=64", 64, 5, 3, 29u);
    failures += run_stochastic_support_case();
    failures += run_mixed_temperature_batch_case();
    failures += run_shortlist_remap_case();
    failures += run_nan_logits_shortlist_case();
    failures += run_tree_layout_case(ops::kDflash2VerifyWidth);
    failures += run_tree_layout_case(6);
    failures += run_tree_compact_case();
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
