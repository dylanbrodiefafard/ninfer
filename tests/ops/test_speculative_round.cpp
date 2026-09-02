#include "ninfer/ops/speculative_round.h"
#include "ops/op_tester.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

using namespace ninfer;
using namespace ninfer::test;

namespace {

template <typename T>
void initialize(GuardedDeviceBuffer& buffer, const std::vector<T>& values) {
    buffer.copy_from_host(values.data(), values.size() * sizeof(T));
}

template <typename T>
std::vector<T> read(const GuardedDeviceBuffer& buffer, std::size_t count) {
    return from_device<T>(buffer.data(), count);
}

DeviceBuffer device_config(const ops::SamplingConfig& config) {
    return to_device(std::vector<ops::SamplingConfig>{config});
}

struct VerifyInputsExpected {
    std::vector<std::int32_t> verify_ids;
    std::vector<std::int32_t> positions;
};

VerifyInputsExpected verify_inputs_oracle(std::int32_t token,
                                          const std::vector<std::int32_t>& drafts,
                                          std::int32_t length) {
    VerifyInputsExpected expected{
        .verify_ids = std::vector<std::int32_t>(drafts.size() + 1),
        .positions  = std::vector<std::int32_t>(drafts.size() + 1),
    };
    expected.verify_ids[0] = token;
    for (std::size_t i = 0; i < drafts.size(); ++i) expected.verify_ids[i + 1] = drafts[i];
    for (std::size_t i = 0; i < expected.positions.size(); ++i) {
        expected.positions[i] = length + static_cast<std::int32_t>(i);
    }
    return expected;
}

int prepare_verify_case(int k) {
    const std::int32_t token_value  = 70000 + k;
    const std::int32_t length_value = 1000 - k;
    std::vector<std::int32_t> drafts(static_cast<std::size_t>(k));
    for (int i = 0; i < k; ++i) drafts[static_cast<std::size_t>(i)] = 37 + 7919 * i;
    const auto expected = verify_inputs_oracle(token_value, drafts, length_value);

    DeviceBuffer d_token  = to_device<std::int32_t>({token_value});
    DeviceBuffer d_drafts = to_device(drafts);
    DeviceBuffer d_length = to_device<std::int32_t>({length_value});
    DeviceBuffer d_extent = to_device<std::int32_t>({k});
    GuardedDeviceBuffer d_verify(expected.verify_ids.size() * sizeof(std::int32_t));
    GuardedDeviceBuffer d_positions(expected.positions.size() * sizeof(std::int32_t));
    d_verify.fill(0xcd);
    d_positions.fill(0xef);

    Tensor token(d_token.p, DType::I32, {1});
    Tensor draft_tensor(d_drafts.p, DType::I32, {k});
    Tensor length(d_length.p, DType::I32, {1});
    Tensor extent(d_extent.p, DType::I32, {1});
    Tensor verify(d_verify.data(), DType::I32, {k + 1});
    Tensor positions(d_positions.data(), DType::I32, {k + 1});
    ops::speculative_prepare_verify_inputs(token, draft_tensor, length, extent, verify, positions,
                                           nullptr);
    cuda_synchronize();

    const std::string label = "speculative prepare K=" + std::to_string(k);
    int failures =
        verify_exact((label + " verify ids").c_str(),
                     read<std::int32_t>(d_verify, expected.verify_ids.size()), expected.verify_ids);
    failures += verify_exact((label + " positions").c_str(),
                             read<std::int32_t>(d_positions, expected.positions.size()),
                             expected.positions);
    failures += verify_exact((label + " token unchanged").c_str(),
                             from_device<std::int32_t>(d_token, 1), {token_value});
    failures += verify_exact((label + " drafts unchanged").c_str(),
                             from_device<std::int32_t>(d_drafts, drafts.size()), drafts);
    failures += verify_exact((label + " length unchanged").c_str(),
                             from_device<std::int32_t>(d_length, 1), {length_value});
    failures += d_verify.verify_guards((label + " verify guards").c_str());
    failures += d_positions.verify_guards((label + " positions guards").c_str());
    return failures;
}

struct AcceptExpected {
    std::vector<std::int32_t> sampled;
    std::int32_t num_sampled;
    std::int32_t accepted;
    std::int32_t length;
    std::int32_t token;
};

AcceptExpected accept_state_oracle(const std::vector<std::int32_t>& drafts, std::int32_t accepted,
                                   std::int32_t terminal_token, std::int32_t initial_length) {
    const int k = static_cast<int>(drafts.size());
    AcceptExpected expected{
        .sampled     = std::vector<std::int32_t>(static_cast<std::size_t>(k + 1), 0),
        .num_sampled = accepted + 1,
        .accepted    = accepted,
        .length      = initial_length + accepted + 1,
        .token       = terminal_token,
    };
    for (int i = 0; i < accepted; ++i) {
        expected.sampled[static_cast<std::size_t>(i)] = drafts[static_cast<std::size_t>(i)];
    }
    expected.sampled[static_cast<std::size_t>(accepted)] = terminal_token;
    return expected;
}

int execute_accept_case(const std::string& label, const std::vector<std::int32_t>& target_tokens,
                        const std::vector<std::uint16_t>& logits_bits, int physical_rows,
                        const std::vector<std::int32_t>& drafts, std::int32_t initial_length,
                        int token_domain, ops::SamplingConfig config,
                        const std::vector<std::int32_t>& initial_token_counts,
                        const AcceptExpected& expected,
                        const std::vector<std::int32_t>* selector_ids = nullptr,
                        const std::vector<float>* selector_q = nullptr) {
    const int k              = static_cast<int>(drafts.size());
    DeviceBuffer d_targets   = to_device(target_tokens);
    DeviceBuffer d_logits    = to_device(logits_bits);
    DeviceBuffer d_drafts    = to_device(drafts);
    DeviceBuffer d_counts    = to_device(initial_token_counts);
    config.token_counts      = static_cast<std::int32_t*>(d_counts.p);
    DeviceBuffer d_config    = device_config(config);
    const auto config_before = from_device<std::uint8_t>(d_config, sizeof(ops::SamplingConfig));

    GuardedDeviceBuffer d_length(sizeof(std::int32_t));
    GuardedDeviceBuffer d_token(sizeof(std::int32_t));
    GuardedDeviceBuffer d_sampled(static_cast<std::size_t>(k + 1) * sizeof(std::int32_t));
    GuardedDeviceBuffer d_num(sizeof(std::int32_t));
    GuardedDeviceBuffer d_accepted(sizeof(std::int32_t));
    DeviceBuffer d_extent = to_device<std::int32_t>({k});
    initialize(d_length, std::vector<std::int32_t>{initial_length});
    initialize(d_token, std::vector<std::int32_t>{-1234567});
    d_sampled.fill(0x9d);
    initialize(d_num, std::vector<std::int32_t>{-11});
    initialize(d_accepted, std::vector<std::int32_t>{-13});

    Tensor targets(d_targets.p, DType::I32, {k + 1});
    Tensor logits(d_logits.p, DType::BF16, {physical_rows, k + 1});
    Tensor draft_tensor(d_drafts.p, DType::I32, {k});
    Tensor extent(d_extent.p, DType::I32, {1});
    Tensor length(d_length.data(), DType::I32, {1});
    Tensor token(d_token.data(), DType::I32, {1});
    Tensor sampled(d_sampled.data(), DType::I32, {k + 1});
    Tensor num_sampled(d_num.data(), DType::I32, {1});
    Tensor accepted(d_accepted.data(), DType::I32, {1});
    DeviceBuffer d_sel_ids;
    DeviceBuffer d_sel_q;
    Tensor sel_ids_t;
    Tensor sel_q_t;
    const Tensor* sel_ids_arg = nullptr;
    const Tensor* sel_q_arg   = nullptr;
    if (selector_ids != nullptr && selector_q != nullptr) {
        const int cap = static_cast<int>(selector_ids->size() / static_cast<std::size_t>(k));
        d_sel_ids     = to_device(*selector_ids);
        d_sel_q       = to_device(*selector_q);
        sel_ids_t     = Tensor(d_sel_ids.p, DType::I32, {cap, k});
        sel_q_t       = Tensor(d_sel_q.p, DType::FP32, {cap, k});
        sel_ids_arg   = &sel_ids_t;
        sel_q_arg     = &sel_q_t;
    }
    const std::size_t workspace_bytes =
        ops::speculative_accept_greedy_drafts_workspace_capacity_bytes(token_domain, k, k, 1, 1);
    WorkspaceArena workspace(std::max<std::size_t>(256, workspace_bytes));
    ops::speculative_accept_greedy_drafts(
        targets, logits, draft_tensor, extent, length, token, sampled, num_sampled, accepted,
        token_domain, static_cast<const ops::SamplingConfig*>(d_config.p), workspace, nullptr,
        sel_ids_arg, sel_q_arg);
    cuda_synchronize();

    int failures = verify_exact((label + " sampled").c_str(), read<std::int32_t>(d_sampled, k + 1),
                                expected.sampled);
    failures += verify_exact((label + " num sampled").c_str(), read<std::int32_t>(d_num, 1),
                             {expected.num_sampled});
    failures += verify_exact((label + " accepted").c_str(), read<std::int32_t>(d_accepted, 1),
                             {expected.accepted});
    failures += verify_exact((label + " length").c_str(), read<std::int32_t>(d_length, 1),
                             {expected.length});
    failures +=
        verify_exact((label + " token").c_str(), read<std::int32_t>(d_token, 1), {expected.token});

    failures +=
        verify_exact((label + " target tokens unchanged").c_str(),
                     from_device<std::int32_t>(d_targets, target_tokens.size()), target_tokens);
    failures += verify_exact((label + " logits unchanged").c_str(),
                             from_device<std::uint16_t>(d_logits, logits_bits.size()), logits_bits);
    failures += verify_exact((label + " drafts unchanged").c_str(),
                             from_device<std::int32_t>(d_drafts, drafts.size()), drafts);
    failures += verify_exact((label + " config unchanged").c_str(),
                             from_device<std::uint8_t>(d_config, sizeof(ops::SamplingConfig)),
                             config_before);

    auto expected_counts = initial_token_counts;
    if (config.temperature > 0.0f) {
        for (int i = 0; i < expected.num_sampled; ++i) {
            ++expected_counts[static_cast<std::size_t>(
                expected.sampled[static_cast<std::size_t>(i)])];
        }
    }
    failures +=
        verify_exact((label + " token counts").c_str(),
                     from_device<std::int32_t>(d_counts, expected_counts.size()), expected_counts);

    failures += d_length.verify_guards((label + " length guards").c_str());
    failures += d_token.verify_guards((label + " token guards").c_str());
    failures += d_sampled.verify_guards((label + " sampled guards").c_str());
    failures += d_num.verify_guards((label + " num guards").c_str());
    failures += d_accepted.verify_guards((label + " accepted guards").c_str());
    if (workspace.used() != 0 || workspace.peak_used() != workspace_bytes) {
        std::cerr << label << ": workspace query/execution high-water mismatch\n";
        ++failures;
    }
    return failures;
}

int greedy_accept_case(int k, int accepted_count, int token_domain = 64) {
    std::vector<std::int32_t> targets(static_cast<std::size_t>(k + 1));
    std::vector<std::int32_t> drafts(static_cast<std::size_t>(k));
    for (int i = 0; i <= k; ++i) {
        targets[static_cast<std::size_t>(i)] = 3 + 2 * i;
        if (i < k) drafts[static_cast<std::size_t>(i)] = targets[static_cast<std::size_t>(i)];
    }
    if (accepted_count < k) {
        drafts[static_cast<std::size_t>(accepted_count)] =
            targets[static_cast<std::size_t>(accepted_count)] + 1;
    }
    const std::int32_t initial_length = 200 + k;
    const auto expected               = accept_state_oracle(
        drafts, accepted_count, targets[static_cast<std::size_t>(accepted_count)], initial_length);
    std::vector<std::uint16_t> logits(static_cast<std::size_t>(token_domain) * (k + 1));
    for (std::size_t i = 0; i < logits.size(); ++i) {
        logits[i] = static_cast<std::uint16_t>(0x3f00u + (i % 127u));
    }
    std::vector<std::int32_t> token_counts(token_domain);
    for (int i = 0; i < token_domain; ++i) token_counts[static_cast<std::size_t>(i)] = i % 5;
    return execute_accept_case("speculative greedy K=" + std::to_string(k) +
                                   " A=" + std::to_string(accepted_count),
                               targets, logits, token_domain, drafts, initial_length, token_domain,
                               ops::SamplingConfig{}, token_counts, expected);
}

int greedy_tree_extent_case(int extent, int expected_accepted) {
    constexpr int kWidth        = 4;
    constexpr int kTokenDomain  = 64;
    constexpr int kInitialLen   = 40;
    const std::vector<std::int32_t> parent{-1, 0, 0, 1};
    const std::vector<std::int32_t> verify_ids{7, 10, 11, 12};
    const std::vector<std::int32_t> targets{10, 99, 0, 0};
    std::vector<std::uint16_t> logits(static_cast<std::size_t>(kTokenDomain) * kWidth, 0x3f00u);
    std::vector<std::int32_t> token_counts(kTokenDomain, 0);

    DeviceBuffer d_targets   = to_device(targets);
    DeviceBuffer d_logits    = to_device(logits);
    DeviceBuffer d_ids       = to_device(verify_ids);
    DeviceBuffer d_parent    = to_device(parent);
    DeviceBuffer d_valid     = to_device<std::int32_t>({kWidth});
    DeviceBuffer d_extent    = to_device<std::int32_t>({extent});
    DeviceBuffer d_counts    = to_device(token_counts);
    ops::SamplingConfig config{};
    config.token_counts      = static_cast<std::int32_t*>(d_counts.p);
    DeviceBuffer d_config    = device_config(config);

    GuardedDeviceBuffer d_length(sizeof(std::int32_t));
    GuardedDeviceBuffer d_anchors(sizeof(std::int32_t));
    GuardedDeviceBuffer d_licensed(static_cast<std::size_t>(kWidth) * sizeof(std::int32_t));
    GuardedDeviceBuffer d_lic_count(sizeof(std::int32_t));
    GuardedDeviceBuffer d_accepted(sizeof(std::int32_t));
    GuardedDeviceBuffer d_column(sizeof(std::int32_t));
    GuardedDeviceBuffer d_path(static_cast<std::size_t>(kWidth) * sizeof(std::int32_t));
    initialize(d_length, std::vector<std::int32_t>{kInitialLen});
    initialize(d_anchors, std::vector<std::int32_t>{-1});
    d_licensed.fill(0x9d);
    initialize(d_lic_count, std::vector<std::int32_t>{-1});
    initialize(d_accepted, std::vector<std::int32_t>{-1});
    initialize(d_column, std::vector<std::int32_t>{-1});
    d_path.fill(0x9d);

    Tensor target_t(d_targets.p, DType::I32, {kWidth, 1});
    Tensor logits_t(d_logits.p, DType::BF16, {kTokenDomain, kWidth, 1});
    Tensor ids_t(d_ids.p, DType::I32, {kWidth, 1});
    Tensor parent_t(d_parent.p, DType::I32, {kWidth, 1});
    Tensor valid_t(d_valid.p, DType::I32, {1});
    Tensor extent_t(d_extent.p, DType::I32, {1});
    Tensor length_t(d_length.data(), DType::I32, {1});
    Tensor anchors_t(d_anchors.data(), DType::I32, {1});
    Tensor licensed_t(d_licensed.data(), DType::I32, {kWidth, 1});
    Tensor lic_count_t(d_lic_count.data(), DType::I32, {1});
    Tensor accepted_t(d_accepted.data(), DType::I32, {1});
    Tensor column_t(d_column.data(), DType::I32, {1});
    Tensor path_t(d_path.data(), DType::I32, {kWidth, 1});
    WorkspaceArena workspace(256);
    ops::speculative_accept_tree_drafts(
        target_t, logits_t, ids_t, parent_t, valid_t, extent_t, length_t, anchors_t, licensed_t,
        lic_count_t, accepted_t, column_t, path_t, kTokenDomain,
        static_cast<const ops::SamplingConfig*>(d_config.p), workspace, nullptr);
    cuda_synchronize();

    const std::string label =
        "tree greedy extent=" + std::to_string(extent) + " A=" + std::to_string(expected_accepted);
    const std::int32_t bonus = expected_accepted == 0 ? 10 : 99;
    std::vector<std::int32_t> want_lic(kWidth, 0);
    std::vector<std::int32_t> want_path(kWidth, 0);
    if (expected_accepted == 1) {
        want_lic[0]  = 10;
        want_lic[1]  = 99;
        want_path[0] = 0;
        want_path[1] = 1;
    } else {
        want_lic[0]  = 10;
        want_path[0] = 0;
    }
    int failures = verify_exact((label + " accepted").c_str(), read<std::int32_t>(d_accepted, 1),
                                {expected_accepted});
    failures += verify_exact((label + " licensed count").c_str(), read<std::int32_t>(d_lic_count, 1),
                             {expected_accepted + 1});
    failures +=
        verify_exact((label + " licensed").c_str(), read<std::int32_t>(d_licensed, kWidth), want_lic);
    failures += verify_exact((label + " path").c_str(), read<std::int32_t>(d_path, kWidth), want_path);
    failures += verify_exact((label + " column").c_str(), read<std::int32_t>(d_column, 1),
                             {expected_accepted});
    failures +=
        verify_exact((label + " bonus").c_str(), read<std::int32_t>(d_anchors, 1), {bonus});
    failures += verify_exact((label + " length").c_str(), read<std::int32_t>(d_length, 1),
                             {kInitialLen + expected_accepted + 1});
    return failures;
}

std::vector<std::uint16_t> peaked_column_logits(int physical_rows, int width,
                                                const std::vector<std::int32_t>& peak_by_col) {
    std::vector<float> logits(static_cast<std::size_t>(physical_rows) * static_cast<std::size_t>(width),
                              -20.0f);
    for (int col = 0; col < width; ++col) {
        const int tok = peak_by_col[static_cast<std::size_t>(col)];
        if (tok >= 0 && tok < physical_rows) {
            logits[static_cast<std::size_t>(col) * static_cast<std::size_t>(physical_rows) +
                   static_cast<std::size_t>(tok)] = 20.0f;
        }
    }
    round_to_bf16(logits);
    std::vector<std::uint16_t> bits(logits.size());
    for (std::size_t i = 0; i < logits.size(); ++i) { bits[i] = f32_to_bf16(logits[i]); }
    return bits;
}

int execute_tree_case(const std::string& label, int token_domain, int physical_rows, int width,
                      const std::vector<std::int32_t>& parent, const std::vector<std::int32_t>& ids,
                      const std::vector<std::int32_t>& targets, const std::vector<std::uint16_t>& logits,
                      int extent, int valid, const ops::SamplingConfig& config_in,
                      const std::vector<std::int32_t>& initial_counts,
                      const std::vector<std::int32_t>& want_lic, int want_accepted,
                      int want_column, const std::vector<std::int32_t>& want_path,
                      int initial_length = 40) {
    DeviceBuffer d_targets = to_device(targets);
    DeviceBuffer d_logits  = to_device(logits);
    DeviceBuffer d_ids     = to_device(ids);
    DeviceBuffer d_parent  = to_device(parent);
    DeviceBuffer d_valid   = to_device<std::int32_t>({valid});
    DeviceBuffer d_extent  = to_device<std::int32_t>({extent});
    DeviceBuffer d_counts  = to_device(initial_counts);
    ops::SamplingConfig config = config_in;
    config.token_counts        = static_cast<std::int32_t*>(d_counts.p);
    DeviceBuffer d_config      = device_config(config);

    GuardedDeviceBuffer d_length(sizeof(std::int32_t));
    GuardedDeviceBuffer d_anchors(sizeof(std::int32_t));
    GuardedDeviceBuffer d_licensed(static_cast<std::size_t>(width) * sizeof(std::int32_t));
    GuardedDeviceBuffer d_lic_count(sizeof(std::int32_t));
    GuardedDeviceBuffer d_accepted(sizeof(std::int32_t));
    GuardedDeviceBuffer d_column(sizeof(std::int32_t));
    GuardedDeviceBuffer d_path(static_cast<std::size_t>(width) * sizeof(std::int32_t));
    initialize(d_length, std::vector<std::int32_t>{initial_length});
    initialize(d_anchors, std::vector<std::int32_t>{-1});
    d_licensed.fill(0x9d);
    initialize(d_lic_count, std::vector<std::int32_t>{-1});
    initialize(d_accepted, std::vector<std::int32_t>{-1});
    initialize(d_column, std::vector<std::int32_t>{-1});
    d_path.fill(0x9d);

    Tensor target_t(d_targets.p, DType::I32, {width, 1});
    Tensor logits_t(d_logits.p, DType::BF16, {physical_rows, width, 1});
    Tensor ids_t(d_ids.p, DType::I32, {width, 1});
    Tensor parent_t(d_parent.p, DType::I32, {width, 1});
    Tensor valid_t(d_valid.p, DType::I32, {1});
    Tensor extent_t(d_extent.p, DType::I32, {1});
    Tensor length_t(d_length.data(), DType::I32, {1});
    Tensor anchors_t(d_anchors.data(), DType::I32, {1});
    Tensor licensed_t(d_licensed.data(), DType::I32, {width, 1});
    Tensor lic_count_t(d_lic_count.data(), DType::I32, {1});
    Tensor accepted_t(d_accepted.data(), DType::I32, {1});
    Tensor column_t(d_column.data(), DType::I32, {1});
    Tensor path_t(d_path.data(), DType::I32, {width, 1});
    const std::size_t workspace_bytes =
        ops::speculative_accept_tree_drafts_workspace_capacity_bytes(token_domain, width, width, 1,
                                                                     1);
    WorkspaceArena workspace(std::max<std::size_t>(256, workspace_bytes));
    ops::speculative_accept_tree_drafts(
        target_t, logits_t, ids_t, parent_t, valid_t, extent_t, length_t, anchors_t, licensed_t,
        lic_count_t, accepted_t, column_t, path_t, token_domain,
        static_cast<const ops::SamplingConfig*>(d_config.p), workspace, nullptr);
    cuda_synchronize();

    int failures = verify_exact((label + " accepted").c_str(), read<std::int32_t>(d_accepted, 1),
                                {want_accepted});
    failures += verify_exact((label + " licensed count").c_str(), read<std::int32_t>(d_lic_count, 1),
                             {want_accepted + 1});
    failures +=
        verify_exact((label + " licensed").c_str(), read<std::int32_t>(d_licensed, width), want_lic);
    failures += verify_exact((label + " path").c_str(), read<std::int32_t>(d_path, width), want_path);
    failures += verify_exact((label + " column").c_str(), read<std::int32_t>(d_column, 1),
                             {want_column});
    failures += verify_exact((label + " anchor").c_str(), read<std::int32_t>(d_anchors, 1),
                             {want_lic[static_cast<std::size_t>(want_accepted)]});
    failures += verify_exact((label + " length").c_str(), read<std::int32_t>(d_length, 1),
                             {initial_length + want_accepted + 1});
    auto expected_counts = initial_counts;
    if (config.temperature > 0.0f) {
        for (int i = 0; i <= want_accepted; ++i) {
            ++expected_counts[static_cast<std::size_t>(want_lic[static_cast<std::size_t>(i)])];
        }
    }
    failures += verify_exact((label + " token counts").c_str(),
                             from_device<std::int32_t>(d_counts, expected_counts.size()),
                             expected_counts);
    if (workspace.used() != 0 || workspace.peak_used() != workspace_bytes) {
        std::cerr << label << ": workspace query/execution high-water mismatch\n";
        ++failures;
    }
    return failures;
}

int greedy_tree_second_child_case(int token_domain) {
    constexpr int kWidth = 4;
    const std::vector<std::int32_t> parent{-1, 0, 0, 1};
    const std::vector<std::int32_t> verify_ids{7, 10, 11, 12};
    const std::vector<std::int32_t> targets{11, 99, 50, 0};
    const auto logits = peaked_column_logits(token_domain, kWidth, {11, 99, 50, 0});
    std::vector<std::int32_t> counts(static_cast<std::size_t>(token_domain), 0);
    std::vector<std::int32_t> want_lic{11, 50, 0, 0};
    std::vector<std::int32_t> want_path{0, 2, 0, 0};
    return execute_tree_case("tree greedy second child V=" + std::to_string(token_domain),
                             token_domain, token_domain, kWidth, parent, verify_ids, targets,
                             logits, 7, kWidth, ops::SamplingConfig{}, counts, want_lic, 1, 2,
                             want_path);
}

int tree_sampling_membership_cases(int token_domain) {
    constexpr int kWidth = 4;
    const std::vector<std::int32_t> parent{-1, 0, 0, 1};
    const std::vector<std::int32_t> verify_ids{7, 10, 11, 12};
    const std::vector<std::int32_t> targets{0, 0, 0, 0};
    ops::SamplingConfig cfg{};
    cfg.temperature = 1.0f;
    cfg.top_k       = 1;
    cfg.seed        = 1ull;
    std::vector<std::int32_t> counts(static_cast<std::size_t>(token_domain), 0);
    counts[10] = 3;
    counts[11] = 4;
    counts[5]  = 9;
    counts[50] = 1;
    const std::string tag = " V=" + std::to_string(token_domain);

    int failures = 0;
    {
        const auto logits = peaked_column_logits(token_domain, kWidth, {10, 50, 0, 0});
        std::vector<std::int32_t> want_lic{10, 50, 0, 0};
        std::vector<std::int32_t> want_path{0, 1, 0, 0};
        failures += execute_tree_case("tree sampling first child" + tag, token_domain, token_domain,
                                      kWidth, parent, verify_ids, targets, logits, 7, kWidth, cfg,
                                      counts, want_lic, 1, 1, want_path);
    }
    {
        const auto logits = peaked_column_logits(token_domain, kWidth, {11, 0, 50, 0});
        std::vector<std::int32_t> want_lic{11, 50, 0, 0};
        std::vector<std::int32_t> want_path{0, 2, 0, 0};
        failures += execute_tree_case("tree sampling second child" + tag, token_domain, token_domain,
                                      kWidth, parent, verify_ids, targets, logits, 7, kWidth, cfg,
                                      counts, want_lic, 1, 2, want_path);
    }
    {
        const auto logits = peaked_column_logits(token_domain, kWidth, {5, 0, 0, 0});
        std::vector<std::int32_t> want_lic{5, 0, 0, 0};
        std::vector<std::int32_t> want_path{0, 0, 0, 0};
        failures += execute_tree_case("tree sampling miss child" + tag, token_domain, token_domain,
                                      kWidth, parent, verify_ids, targets, logits, 7, kWidth, cfg,
                                      counts, want_lic, 0, 0, want_path);
    }
    {
        ops::SamplingConfig suppressed_cfg = cfg;
        suppressed_cfg.suppressed_token_count = 1;
        suppressed_cfg.suppressed_tokens[0]   = 11;
        const auto logits = peaked_column_logits(token_domain, kWidth, {11, 0, 0, 0});
        std::vector<std::int32_t> want_lic{0, 0, 0, 0};
        std::vector<std::int32_t> want_path{0, 0, 0, 0};
        failures += execute_tree_case("tree sampling rejects suppressed draft" + tag,
                                      token_domain, token_domain, kWidth, parent, verify_ids,
                                      targets, logits, 7, kWidth, suppressed_cfg, counts, want_lic,
                                      0, 0, want_path);
    }
    return failures;
}

int tree_sampling_presence_overlay_case(int token_domain) {
    constexpr int kWidth = 2;
    const std::vector<std::int32_t> parent{-1, 0};
    const std::vector<std::int32_t> verify_ids{0, 10};
    const std::vector<std::int32_t> targets{0, 0};
    std::vector<float> logits(static_cast<std::size_t>(token_domain) * kWidth, -20.0f);
    logits[10] = 20.0f;
    logits[static_cast<std::size_t>(token_domain) + 10] = 20.0f;
    logits[static_cast<std::size_t>(token_domain) + 20] = 20.0f;
    round_to_bf16(logits);
    std::vector<std::uint16_t> bits(logits.size());
    for (std::size_t i = 0; i < logits.size(); ++i) { bits[i] = f32_to_bf16(logits[i]); }
    ops::SamplingConfig cfg{};
    cfg.temperature      = 1.0f;
    cfg.top_k            = 1;
    cfg.presence_penalty = 100.0f;
    cfg.seed             = 1ull;
    std::vector<std::int32_t> counts(static_cast<std::size_t>(token_domain), 0);
    std::vector<std::int32_t> want_lic{10, 20};
    std::vector<std::int32_t> want_path{0, 1};
    return execute_tree_case("tree sampling presence overlay V=" + std::to_string(token_domain),
                             token_domain, token_domain, kWidth, parent, verify_ids, targets, bits,
                             1, kWidth, cfg, counts, want_lic, 1, 1, want_path);
}

int deterministic_sampling_case() {
    constexpr int physical_rows = 248320;
    constexpr int token_domain  = 248077;
    constexpr int k             = 5;
    constexpr int accepted      = 2;
    const std::vector<std::int32_t> drafts{17, 7919, 65537, 131071, 200003};
    std::vector<std::int32_t> targets(static_cast<std::size_t>(k + 1));
    for (int i = 0; i <= k; ++i) targets[static_cast<std::size_t>(i)] = 101 + i;
    constexpr std::int32_t correction = 150001;

    std::vector<float> logits(static_cast<std::size_t>(physical_rows) * (k + 1), -20.0f);
    for (int col = 0; col <= k; ++col) {
        const std::size_t base = static_cast<std::size_t>(col) * physical_rows;
        const int winner =
            col < accepted ? drafts[static_cast<std::size_t>(col)] : correction + col - accepted;
        logits[base + static_cast<std::size_t>(winner)] = 20.0f;
        logits[base + 13]                               = 30.0f;
        logits[base + token_domain]                     = 100.0f;
        logits[base + physical_rows - 1]                = 200.0f;
    }
    round_to_bf16(logits);
    std::vector<std::uint16_t> logits_bits(logits.size());
    for (std::size_t i = 0; i < logits.size(); ++i) logits_bits[i] = f32_to_bf16(logits[i]);

    const std::int32_t initial_length = 4093;
    const auto expected = accept_state_oracle(drafts, accepted, correction, initial_length);
    std::vector<std::int32_t> token_counts(token_domain, 0);
    token_counts[static_cast<std::size_t>(drafts[0])]  = 3;
    token_counts[static_cast<std::size_t>(drafts[1])]  = 5;
    token_counts[static_cast<std::size_t>(correction)] = 7;

    ops::SamplingConfig config{};
    config.temperature = 1.0f;
    config.top_k       = 1;
    config.top_p       = 0.9f;
    config.min_p       = 0.5f;
    config.seed        = 0x123456789abcdef0ull;
    config.suppressed_token_count = 1;
    config.suppressed_tokens[0]   = 13;
    return execute_accept_case("speculative sampling deterministic support", targets, logits_bits,
                               physical_rows, drafts, initial_length, token_domain, config,
                               token_counts, expected);
}

int suppressed_bonus_case() {
    constexpr int token_domain = 64;
    constexpr int k            = 1;
    const std::vector<std::int32_t> drafts{7};
    const std::vector<std::int32_t> targets{7, 8};
    std::vector<float> logits(static_cast<std::size_t>(token_domain) * (k + 1), -20.0f);
    logits[7]                                          = 20.0f;
    logits[9]                                          = 30.0f;
    logits[static_cast<std::size_t>(token_domain) + 8] = 20.0f;
    logits[static_cast<std::size_t>(token_domain) + 9] = 30.0f;
    round_to_bf16(logits);
    std::vector<std::uint16_t> logits_bits(logits.size());
    for (std::size_t i = 0; i < logits.size(); ++i) { logits_bits[i] = f32_to_bf16(logits[i]); }

    constexpr std::int32_t initial_length = 40;
    const auto expected                   = accept_state_oracle(drafts, 1, 8, initial_length);
    std::vector<std::int32_t> token_counts(token_domain, 0);
    ops::SamplingConfig config{};
    config.temperature            = 1.0f;
    config.top_k                  = 1;
    config.seed                   = 7ull;
    config.suppressed_token_count = 1;
    config.suppressed_tokens[0]   = 9;
    return execute_accept_case("speculative sampling suppresses EOS in accept and bonus", targets,
                               logits_bits, token_domain, drafts, initial_length, token_domain,
                               config, token_counts, expected);
}

int suppressed_draft_rejection_case(int token_domain) {
    constexpr int k = 1;
    const std::vector<std::int32_t> drafts{9};
    const std::vector<std::int32_t> targets{7, 8};
    std::vector<float> logits(static_cast<std::size_t>(token_domain) * (k + 1), -20.0f);
    logits[7]                                          = 20.0f;
    logits[9]                                          = 30.0f;
    logits[static_cast<std::size_t>(token_domain) + 8] = 20.0f;
    logits[static_cast<std::size_t>(token_domain) + 9] = 30.0f;
    round_to_bf16(logits);
    std::vector<std::uint16_t> logits_bits(logits.size());
    for (std::size_t i = 0; i < logits.size(); ++i) { logits_bits[i] = f32_to_bf16(logits[i]); }

    constexpr std::int32_t initial_length = 40;
    const auto expected                   = accept_state_oracle(drafts, 0, 7, initial_length);
    std::vector<std::int32_t> token_counts(token_domain, 0);
    ops::SamplingConfig config{};
    config.temperature            = 1.0f;
    config.top_k                  = 1;
    config.seed                   = 7ull;
    config.suppressed_token_count = 1;
    config.suppressed_tokens[0]   = 9;
    return execute_accept_case("speculative sampling rejects suppressed draft V=" +
                                   std::to_string(token_domain),
                               targets, logits_bits, token_domain, drafts, initial_length,
                               token_domain, config, token_counts, expected);
}

int p_less_deterministic_accept_case() {
    constexpr int physical_rows = 248320;
    constexpr int token_domain  = 248077;
    constexpr int k             = 5;
    constexpr int accepted      = 2;
    const std::vector<std::int32_t> drafts{17, 7919, 65537, 131071, 200003};
    std::vector<std::int32_t> targets(static_cast<std::size_t>(k + 1));
    for (int i = 0; i <= k; ++i) { targets[static_cast<std::size_t>(i)] = 101 + i; }
    constexpr std::int32_t correction = 150001;

    std::vector<float> logits(static_cast<std::size_t>(physical_rows) * (k + 1), -20.0f);
    for (int col = 0; col <= k; ++col) {
        const std::size_t base = static_cast<std::size_t>(col) * physical_rows;
        const int winner =
            col < accepted ? drafts[static_cast<std::size_t>(col)] : correction + col - accepted;
        logits[base + static_cast<std::size_t>(winner)] = 20.0f;
        logits[base + token_domain]                     = 100.0f;
        logits[base + physical_rows - 1]                = 200.0f;
    }
    round_to_bf16(logits);
    std::vector<std::uint16_t> logits_bits(logits.size());
    for (std::size_t i = 0; i < logits.size(); ++i) { logits_bits[i] = f32_to_bf16(logits[i]); }

    const std::int32_t initial_length = 4093;
    const auto expected = accept_state_oracle(drafts, accepted, correction, initial_length);
    std::vector<std::int32_t> token_counts(token_domain, 0);
    token_counts[static_cast<std::size_t>(drafts[0])]  = 3;
    token_counts[static_cast<std::size_t>(drafts[1])]  = 5;
    token_counts[static_cast<std::size_t>(correction)] = 7;

    ops::SamplingConfig config{};
    config.temperature = 1.0f;
    config.top_k       = 20;
    config.top_p       = 0.9f;
    config.p_less      = 1;
    config.seed        = 0x123456789abcdef0ull;
    return execute_accept_case("speculative p-less deterministic support", targets, logits_bits,
                               physical_rows, drafts, initial_length, token_domain, config,
                               token_counts, expected);
}

int p_less_tree_membership_cases(int token_domain) {
    constexpr int kWidth = 4;
    const std::vector<std::int32_t> parent{-1, 0, 0, 1};
    const std::vector<std::int32_t> verify_ids{7, 10, 11, 12};
    const std::vector<std::int32_t> targets{0, 0, 0, 0};
    ops::SamplingConfig cfg{};
    cfg.temperature = 1.0f;
    cfg.top_k       = 1;
    cfg.p_less      = 1;
    cfg.seed        = 1ull;
    std::vector<std::int32_t> counts(static_cast<std::size_t>(token_domain), 0);
    counts[10]            = 3;
    counts[11]            = 4;
    counts[5]             = 9;
    counts[50]            = 1;
    const std::string tag = " V=" + std::to_string(token_domain);

    int failures = 0;
    {
        const auto logits = peaked_column_logits(token_domain, kWidth, {10, 50, 0, 0});
        std::vector<std::int32_t> want_lic{10, 50, 0, 0};
        std::vector<std::int32_t> want_path{0, 1, 0, 0};
        failures += execute_tree_case("tree p-less first child" + tag, token_domain, token_domain,
                                      kWidth, parent, verify_ids, targets, logits, 7, kWidth, cfg,
                                      counts, want_lic, 1, 1, want_path);
    }
    {
        const auto logits = peaked_column_logits(token_domain, kWidth, {11, 0, 50, 0});
        std::vector<std::int32_t> want_lic{11, 50, 0, 0};
        std::vector<std::int32_t> want_path{0, 2, 0, 0};
        failures += execute_tree_case("tree p-less second child" + tag, token_domain, token_domain,
                                      kWidth, parent, verify_ids, targets, logits, 7, kWidth, cfg,
                                      counts, want_lic, 1, 2, want_path);
    }
    return failures;
}

int batched_sampling_workspace_stride_case() {
    constexpr int physical_rows = 257;
    constexpr int token_domain  = 257;
    constexpr int k             = 3;
    constexpr int batch         = 2;
    constexpr int columns       = k + 1;

    const std::vector<std::int32_t> drafts{10, 11, 12, 30, 31, 32};
    const std::vector<std::int32_t> winners{10, 20, 21, 22, 30, 31, 32, 33};
    std::vector<std::uint16_t> logits(static_cast<std::size_t>(physical_rows) * columns * batch,
                                      f32_to_bf16(-20.0f));
    for (int row = 0; row < batch; ++row) {
        for (int col = 0; col < columns; ++col) {
            const std::size_t base =
                (static_cast<std::size_t>(row) * columns + col) * physical_rows;
            logits[base + static_cast<std::size_t>(winners[row * columns + col])] =
                f32_to_bf16(20.0f);
        }
    }

    DeviceBuffer d_targets = to_device(winners);
    DeviceBuffer d_logits  = to_device(logits);
    DeviceBuffer d_drafts  = to_device(drafts);
    DeviceBuffer d_extents = to_device<std::int32_t>({k, k});
    DeviceBuffer d_lengths = to_device<std::int32_t>({100, 200});
    DeviceBuffer d_anchors = to_device<std::int32_t>({-1, -1});
    DeviceBuffer d_licensed(static_cast<std::size_t>(columns) * batch * sizeof(std::int32_t));
    DeviceBuffer d_counts(static_cast<std::size_t>(batch) * sizeof(std::int32_t));
    DeviceBuffer d_accepted(static_cast<std::size_t>(batch) * sizeof(std::int32_t));

    ops::SamplingConfig config{};
    config.temperature = 1.0f;
    config.top_k       = 1;
    const std::vector<ops::SamplingConfig> configs{config, config};
    DeviceBuffer d_configs = to_device(configs);

    Tensor targets(d_targets.p, DType::I32, {columns, batch});
    Tensor logits_tensor(d_logits.p, DType::BF16, {physical_rows, columns, batch});
    Tensor draft_tensor(d_drafts.p, DType::I32, {k, batch});
    Tensor extents(d_extents.p, DType::I32, {batch});
    Tensor lengths(d_lengths.p, DType::I32, {batch});
    Tensor anchors(d_anchors.p, DType::I32, {batch});
    Tensor licensed(d_licensed.p, DType::I32, {columns, batch});
    Tensor counts(d_counts.p, DType::I32, {batch});
    Tensor accepted(d_accepted.p, DType::I32, {batch});
    const std::size_t workspace_bytes =
        ops::speculative_accept_greedy_drafts_workspace_capacity_bytes(token_domain, k, k, batch,
                                                                       batch);
    WorkspaceArena workspace(workspace_bytes);
    ops::speculative_accept_greedy_drafts(
        targets, logits_tensor, draft_tensor, extents, lengths, anchors, licensed, counts, accepted,
        token_domain, static_cast<const ops::SamplingConfig*>(d_configs.p), workspace, nullptr);
    cuda_synchronize();

    int failures = verify_exact("speculative sampling B=2 licensed",
                                from_device<std::int32_t>(d_licensed, columns * batch),
                                {10, 20, 0, 0, 30, 31, 32, 33});
    failures += verify_exact("speculative sampling B=2 counts",
                             from_device<std::int32_t>(d_counts, batch), {2, 4});
    failures += verify_exact("speculative sampling B=2 accepted",
                             from_device<std::int32_t>(d_accepted, batch), {1, 3});
    failures += verify_exact("speculative sampling B=2 lengths",
                             from_device<std::int32_t>(d_lengths, batch), {102, 204});
    failures += verify_exact("speculative sampling B=2 anchors",
                             from_device<std::int32_t>(d_anchors, batch), {20, 33});
    return failures;
}

std::pair<std::vector<std::int32_t>, std::vector<float>>
make_chain_selector(const std::vector<std::int32_t>& drafts, float q_draft, int cap = 16) {
    const int k = static_cast<int>(drafts.size());
    std::vector<std::int32_t> ids(static_cast<std::size_t>(cap) * static_cast<std::size_t>(k));
    std::vector<float> q(static_cast<std::size_t>(cap) * static_cast<std::size_t>(k), 0.0f);
    for (int i = 0; i < k; ++i) {
        const std::size_t base                    = static_cast<std::size_t>(cap) * i;
        ids[base]                                 = drafts[static_cast<std::size_t>(i)];
        q[base]                                   = q_draft;
        for (int c = 1; c < cap; ++c) {
            ids[base + static_cast<std::size_t>(c)] = 2000 + c + 32 * i;
        }
    }
    return {ids, q};
}

int greedy_ignores_selector_case() {
    constexpr int k            = 5;
    constexpr int accepted     = 2;
    constexpr int token_domain = 64;
    std::vector<std::int32_t> targets(static_cast<std::size_t>(k + 1));
    std::vector<std::int32_t> drafts(static_cast<std::size_t>(k));
    for (int i = 0; i <= k; ++i) {
        targets[static_cast<std::size_t>(i)] = 3 + 2 * i;
        if (i < k) { drafts[static_cast<std::size_t>(i)] = targets[static_cast<std::size_t>(i)]; }
    }
    drafts[static_cast<std::size_t>(accepted)] =
        targets[static_cast<std::size_t>(accepted)] + 1;
    const std::int32_t initial_length = 200 + k;
    const auto expected               = accept_state_oracle(
        drafts, accepted, targets[static_cast<std::size_t>(accepted)], initial_length);
    std::vector<std::uint16_t> logits(static_cast<std::size_t>(token_domain) * (k + 1));
    for (std::size_t i = 0; i < logits.size(); ++i) {
        logits[i] = static_cast<std::uint16_t>(0x3f00u + (i % 127u));
    }
    std::vector<std::int32_t> token_counts(token_domain);
    for (int i = 0; i < token_domain; ++i) { token_counts[static_cast<std::size_t>(i)] = i % 5; }
    const auto [ids, q] = make_chain_selector(drafts, 0.0f);
    return execute_accept_case("speculative greedy ignores selector q", targets, logits,
                               token_domain, drafts, initial_length, token_domain,
                               ops::SamplingConfig{}, token_counts, expected, &ids, &q);
}

int onehot_selector_matches_null_case() {
    constexpr int physical_rows = 248320;
    constexpr int token_domain  = 248077;
    constexpr int k             = 5;
    constexpr int accepted      = 2;
    const std::vector<std::int32_t> drafts{17, 7919, 65537, 131071, 200003};
    std::vector<std::int32_t> targets(static_cast<std::size_t>(k + 1));
    for (int i = 0; i <= k; ++i) { targets[static_cast<std::size_t>(i)] = 101 + i; }
    constexpr std::int32_t correction = 150001;

    std::vector<float> logits(static_cast<std::size_t>(physical_rows) * (k + 1), -20.0f);
    for (int col = 0; col <= k; ++col) {
        const std::size_t base = static_cast<std::size_t>(col) * physical_rows;
        const int winner =
            col < accepted ? drafts[static_cast<std::size_t>(col)] : correction + col - accepted;
        logits[base + static_cast<std::size_t>(winner)] = 20.0f;
        logits[base + token_domain]                     = 100.0f;
        logits[base + physical_rows - 1]                = 200.0f;
    }
    round_to_bf16(logits);
    std::vector<std::uint16_t> logits_bits(logits.size());
    for (std::size_t i = 0; i < logits.size(); ++i) { logits_bits[i] = f32_to_bf16(logits[i]); }

    const std::int32_t initial_length = 4093;
    const auto expected = accept_state_oracle(drafts, accepted, correction, initial_length);
    std::vector<std::int32_t> token_counts(token_domain, 0);
    token_counts[static_cast<std::size_t>(drafts[0])]  = 3;
    token_counts[static_cast<std::size_t>(drafts[1])]  = 5;
    token_counts[static_cast<std::size_t>(correction)] = 7;

    ops::SamplingConfig config{};
    config.temperature = 1.0f;
    config.top_k       = 1;
    config.top_p       = 0.9f;
    config.min_p       = 0.5f;
    config.seed        = 0x123456789abcdef0ull;
    const auto [ids, q] = make_chain_selector(drafts, 1.0f);
    return execute_accept_case("speculative sampling one-hot selector q", targets, logits_bits,
                               physical_rows, drafts, initial_length, token_domain, config,
                               token_counts, expected, &ids, &q);
}

int zero_q_rejects_and_corrects_case(int token_domain) {
    constexpr int k = 1;
    const std::vector<std::int32_t> drafts{7};
    const std::vector<std::int32_t> targets{3, 11};
    std::vector<float> logits(static_cast<std::size_t>(token_domain) * (k + 1), -20.0f);
    logits[3]                                = 20.0f;
    logits[static_cast<std::size_t>(token_domain) + 11] = 20.0f;
    round_to_bf16(logits);
    std::vector<std::uint16_t> logits_bits(logits.size());
    for (std::size_t i = 0; i < logits.size(); ++i) { logits_bits[i] = f32_to_bf16(logits[i]); }
    const std::int32_t initial_length = 40;
    const auto expected               = accept_state_oracle(drafts, 0, 3, initial_length);
    std::vector<std::int32_t> token_counts(token_domain, 0);
    ops::SamplingConfig config{};
    config.temperature  = 1.0f;
    config.top_k        = 1;
    config.seed         = 1ull;
    const auto [ids, q] = make_chain_selector(drafts, 0.0f);
    return execute_accept_case("speculative sampling q(d)=0 rejects and corrects V=" +
                                   std::to_string(token_domain),
                               targets, logits_bits, token_domain, drafts, initial_length,
                               token_domain, config, token_counts, expected, &ids, &q);
}

int fractional_q_accepts_when_p_covers_q() {
    constexpr int token_domain = 64;
    constexpr int k            = 1;
    const std::vector<std::int32_t> drafts{7};
    const std::vector<std::int32_t> targets{7, 11};
    std::vector<float> logits(static_cast<std::size_t>(token_domain) * (k + 1), -20.0f);
    logits[7]                                          = 20.0f;
    logits[static_cast<std::size_t>(token_domain) + 11] = 20.0f;
    round_to_bf16(logits);
    std::vector<std::uint16_t> logits_bits(logits.size());
    for (std::size_t i = 0; i < logits.size(); ++i) { logits_bits[i] = f32_to_bf16(logits[i]); }
    const std::int32_t initial_length = 40;
    const auto expected               = accept_state_oracle(drafts, 1, 11, initial_length);
    std::vector<std::int32_t> token_counts(token_domain, 0);
    ops::SamplingConfig config{};
    config.temperature  = 1.0f;
    config.top_k        = 1;
    config.seed         = 7ull;
    const auto [ids, q] = make_chain_selector(drafts, 0.5f);
    return execute_accept_case("speculative sampling q=0.5 p=1 accepts", targets, logits_bits,
                               token_domain, drafts, initial_length, token_domain, config,
                               token_counts, expected, &ids, &q);
}

int fractional_q_rejects_when_p_is_zero() {
    constexpr int token_domain = 64;
    constexpr int k            = 1;
    const std::vector<std::int32_t> drafts{7};
    const std::vector<std::int32_t> targets{3, 11};
    std::vector<float> logits(static_cast<std::size_t>(token_domain) * (k + 1), -20.0f);
    logits[3]                                          = 20.0f;
    logits[static_cast<std::size_t>(token_domain) + 11] = 20.0f;
    round_to_bf16(logits);
    std::vector<std::uint16_t> logits_bits(logits.size());
    for (std::size_t i = 0; i < logits.size(); ++i) { logits_bits[i] = f32_to_bf16(logits[i]); }
    const std::int32_t initial_length = 40;
    const auto expected               = accept_state_oracle(drafts, 0, 3, initial_length);
    std::vector<std::int32_t> token_counts(token_domain, 0);
    ops::SamplingConfig config{};
    config.temperature  = 1.0f;
    config.top_k        = 1;
    config.seed         = 7ull;
    const auto [ids, q] = make_chain_selector(drafts, 0.5f);
    return execute_accept_case("speculative sampling q=0.5 p=0 rejects", targets, logits_bits,
                               token_domain, drafts, initial_length, token_domain, config,
                               token_counts, expected, &ids, &q);
}

int residual_p_minus_q_prefers_uncovered_mass() {
    constexpr int token_domain = 64;
    constexpr int k            = 1;
    const std::vector<std::int32_t> drafts{7};
    const std::vector<std::int32_t> targets{3, 11};
    std::vector<float> logits(static_cast<std::size_t>(token_domain) * (k + 1), -20.0f);
    logits[3]  = 20.0f;
    logits[11] = 20.0f;
    logits[static_cast<std::size_t>(token_domain) + 11] = 20.0f;
    round_to_bf16(logits);
    std::vector<std::uint16_t> logits_bits(logits.size());
    for (std::size_t i = 0; i < logits.size(); ++i) { logits_bits[i] = f32_to_bf16(logits[i]); }
    const std::int32_t initial_length = 40;
    const auto expected               = accept_state_oracle(drafts, 0, 11, initial_length);
    std::vector<std::int32_t> token_counts(token_domain, 0);
    ops::SamplingConfig config{};
    config.temperature = 1.0f;
    config.top_k       = 3;
    config.seed        = 7ull;
    constexpr int cap  = 16;
    std::vector<std::int32_t> ids(static_cast<std::size_t>(cap) * k, 0);
    std::vector<float> q(static_cast<std::size_t>(cap) * k, 0.0f);
    ids[0] = 7;
    q[0]   = 0.5f;
    ids[1] = 3;
    q[1]   = 0.5f;
    ids[2] = 11;
    q[2]   = 0.0f;
    return execute_accept_case("speculative sampling residual p-q picks uncovered token", targets,
                               logits_bits, token_domain, drafts, initial_length, token_domain,
                               config, token_counts, expected, &ids, &q);
}

int batched_selector_row_isolation_case() {
    constexpr int token_domain = 64;
    constexpr int k            = 1;
    constexpr int batch        = 2;
    constexpr int columns      = k + 1;
    constexpr int cap          = 16;
    const std::vector<std::int32_t> drafts{7, 9};
    const std::vector<std::int32_t> targets{7, 11, 3, 11};
    std::vector<float> logits(static_cast<std::size_t>(token_domain) * columns * batch, -20.0f);
    logits[7]                                                                = 20.0f;
    logits[static_cast<std::size_t>(token_domain) + 11]                       = 20.0f;
    logits[static_cast<std::size_t>(columns) * token_domain + 3]              = 20.0f;
    logits[static_cast<std::size_t>(columns) * token_domain + token_domain + 11] = 20.0f;
    round_to_bf16(logits);
    std::vector<std::uint16_t> logits_bits(logits.size());
    for (std::size_t i = 0; i < logits.size(); ++i) { logits_bits[i] = f32_to_bf16(logits[i]); }

    std::vector<std::int32_t> ids(static_cast<std::size_t>(cap) * k * batch, 0);
    std::vector<float> q(static_cast<std::size_t>(cap) * k * batch, 0.0f);
    ids[0]      = 7;
    q[0]        = 0.5f;
    ids[cap]    = 9;
    q[cap]      = 0.5f;

    DeviceBuffer d_targets = to_device(targets);
    DeviceBuffer d_logits  = to_device(logits_bits);
    DeviceBuffer d_drafts  = to_device(drafts);
    DeviceBuffer d_extents = to_device<std::int32_t>({k, k});
    DeviceBuffer d_lengths = to_device<std::int32_t>({40, 80});
    DeviceBuffer d_anchors = to_device<std::int32_t>({-1, -1});
    DeviceBuffer d_licensed(static_cast<std::size_t>(columns) * batch * sizeof(std::int32_t));
    DeviceBuffer d_counts(static_cast<std::size_t>(batch) * sizeof(std::int32_t));
    DeviceBuffer d_accepted(static_cast<std::size_t>(batch) * sizeof(std::int32_t));
    DeviceBuffer d_sel_ids = to_device(ids);
    DeviceBuffer d_sel_q   = to_device(q);
    std::vector<std::int32_t> token_counts(token_domain, 0);
    DeviceBuffer d_token_counts = to_device(token_counts);
    ops::SamplingConfig config{};
    config.temperature  = 1.0f;
    config.top_k        = 1;
    config.seed         = 7ull;
    config.token_counts = static_cast<std::int32_t*>(d_token_counts.p);
    const std::vector<ops::SamplingConfig> configs{config, config};
    DeviceBuffer d_configs = to_device(configs);

    Tensor targets_t(d_targets.p, DType::I32, {columns, batch});
    Tensor logits_t(d_logits.p, DType::BF16, {token_domain, columns, batch});
    Tensor draft_t(d_drafts.p, DType::I32, {k, batch});
    Tensor extents(d_extents.p, DType::I32, {batch});
    Tensor lengths(d_lengths.p, DType::I32, {batch});
    Tensor anchors(d_anchors.p, DType::I32, {batch});
    Tensor licensed(d_licensed.p, DType::I32, {columns, batch});
    Tensor lic_counts(d_counts.p, DType::I32, {batch});
    Tensor accepted(d_accepted.p, DType::I32, {batch});
    Tensor sel_ids_t(d_sel_ids.p, DType::I32, {cap, k, batch});
    Tensor sel_q_t(d_sel_q.p, DType::FP32, {cap, k, batch});
    const std::size_t workspace_bytes =
        ops::speculative_accept_greedy_drafts_workspace_capacity_bytes(token_domain, k, k, batch,
                                                                       batch);
    WorkspaceArena workspace(std::max<std::size_t>(256, workspace_bytes));
    ops::speculative_accept_greedy_drafts(
        targets_t, logits_t, draft_t, extents, lengths, anchors, licensed, lic_counts, accepted,
        token_domain, static_cast<const ops::SamplingConfig*>(d_configs.p), workspace, nullptr,
        &sel_ids_t, &sel_q_t);
    cuda_synchronize();

    int failures = verify_exact("batched selector licensed",
                                from_device<std::int32_t>(d_licensed, columns * batch),
                                {7, 11, 3, 0});
    failures += verify_exact("batched selector accepted",
                             from_device<std::int32_t>(d_accepted, batch), {1, 0});
    failures += verify_exact("batched selector counts", from_device<std::int32_t>(d_counts, batch),
                             {2, 1});
    failures += verify_exact("batched selector lengths",
                             from_device<std::int32_t>(d_lengths, batch), {42, 81});
    failures += verify_exact("batched selector anchors",
                             from_device<std::int32_t>(d_anchors, batch), {11, 3});
    return failures;
}

int select_hidden_case(int rows, int columns, int accepted_value) {
    std::vector<std::uint16_t> hidden(static_cast<std::size_t>(rows) * columns);
    for (int col = 0; col < columns; ++col) {
        for (int row = 0; row < rows; ++row) {
            hidden[static_cast<std::size_t>(col) * rows + row] =
                static_cast<std::uint16_t>(0x0100u + ((col * 257 + row * 13) & 0x7fffu));
        }
    }
    std::vector<std::uint16_t> expected(static_cast<std::size_t>(rows));
    std::copy_n(hidden.begin() + static_cast<std::ptrdiff_t>(accepted_value) * rows, rows,
                expected.begin());

    DeviceBuffer d_hidden   = to_device(hidden);
    DeviceBuffer d_accepted = to_device<std::int32_t>({accepted_value});
    GuardedDeviceBuffer d_out(static_cast<std::size_t>(rows) * sizeof(std::uint16_t));
    d_out.fill(0xcd);
    Tensor hidden_tensor(d_hidden.p, DType::BF16, {rows, columns});
    Tensor accepted(d_accepted.p, DType::I32, {1});
    Tensor out(d_out.data(), DType::BF16, {rows, 1});
    ops::speculative_select_accepted_hidden(hidden_tensor, accepted, out, nullptr);
    cuda_synchronize();

    const std::string label =
        "speculative select D=" + std::to_string(rows) + " A=" + std::to_string(accepted_value);
    int failures = verify_exact((label + " output").c_str(),
                                read<std::uint16_t>(d_out, expected.size()), expected);
    failures += verify_exact((label + " hidden unchanged").c_str(),
                             from_device<std::uint16_t>(d_hidden, hidden.size()), hidden);
    failures += verify_exact((label + " accepted unchanged").c_str(),
                             from_device<std::int32_t>(d_accepted, 1), {accepted_value});
    failures += d_out.verify_guards((label + " output guards").c_str());
    return failures;
}

int remap_case(int token_count) {
    constexpr int map_size = 131072;
    std::vector<std::int32_t> id_map(map_size);
    for (int i = 0; i < map_size; ++i) {
        const auto value                    = 65537u * static_cast<std::uint32_t>(i) + 17u;
        id_map[static_cast<std::size_t>(i)] = static_cast<std::int32_t>(value & (map_size - 1u));
    }
    std::vector<std::int32_t> proposals(static_cast<std::size_t>(token_count));
    for (int i = 0; i < token_count; ++i) {
        proposals[static_cast<std::size_t>(i)] =
            i == 0 ? 0 : (i == token_count - 1 ? map_size - 1 : (7919 * i) & (map_size - 1));
    }
    std::vector<std::int32_t> expected(proposals.size());
    for (std::size_t i = 0; i < proposals.size(); ++i) {
        expected[i] = id_map[static_cast<std::size_t>(proposals[i])];
    }

    DeviceBuffer d_map = to_device(id_map);
    GuardedDeviceBuffer d_proposals(proposals.size() * sizeof(std::int32_t));
    initialize(d_proposals, proposals);
    Tensor proposal_tensor(d_proposals.data(), DType::I32, {token_count});
    ops::proposal_remap_token_ids(proposal_tensor, static_cast<const std::int32_t*>(d_map.p),
                                  map_size, nullptr);
    cuda_synchronize();

    const std::string label = "proposal remap T=" + std::to_string(token_count);
    int failures            = verify_exact((label + " in-place output").c_str(),
                                           read<std::int32_t>(d_proposals, proposals.size()), expected);
    failures += verify_exact((label + " map unchanged").c_str(),
                             from_device<std::int32_t>(d_map, id_map.size()), id_map);
    failures += d_proposals.verify_guards((label + " guards").c_str());
    return failures;
}

} // namespace

int main() {
    if (cuda_unavailable()) {
        std::cerr << "FAIL: no usable CUDA device\n";
        return 1;
    }

    int failures = 0;
    const std::size_t k15 =
        ops::speculative_accept_greedy_drafts_workspace_capacity_bytes(257, 15, 15, 1, 1);
    if (k15 == 0 || k15 != ops::sampling_workspace_capacity_bytes(257, 16, 16) ||
        ops::speculative_accept_greedy_drafts_workspace_capacity_bytes(257, 16, 16, 1, 1) != 0 ||
        ops::speculative_accept_greedy_drafts_workspace_capacity_bytes(257, 1, 16, 1, 1) != k15 ||
        ops::speculative_accept_greedy_drafts_workspace_capacity_bytes(257, 15, 15, 1, 2) !=
            2 * k15) {
        std::cerr << "speculative accept workspace did not close over K+1 sampling columns\n";
        ++failures;
    }
    try {
        (void)ops::speculative_accept_greedy_drafts_workspace_capacity_bytes(257, 0, 15, 1, 1);
        std::cerr << "speculative accept workspace accepted an invalid draft interval\n";
        ++failures;
    } catch (const std::invalid_argument&) {}
    const std::size_t tree12 =
        ops::speculative_accept_tree_drafts_workspace_capacity_bytes(257, 12, 12, 1, 1);
    if (tree12 == 0 || tree12 != ops::sampling_workspace_capacity_bytes(257, 12, 12) ||
        ops::speculative_accept_tree_drafts_workspace_capacity_bytes(257, 12, 12, 1, 2) !=
            2 * tree12 ||
        ops::speculative_accept_tree_drafts_workspace_capacity_bytes(64, 4, 4, 1, 1) != 0) {
        std::cerr << "speculative tree accept workspace did not close over packed width\n";
        ++failures;
    }
    try {
        (void)ops::speculative_accept_tree_drafts_workspace_capacity_bytes(257, 1, 12, 1, 1);
        std::cerr << "speculative tree accept workspace accepted W<2\n";
        ++failures;
    } catch (const std::invalid_argument&) {}
    for (const int k : {1, 5, 15}) failures += prepare_verify_case(k);
    failures += greedy_accept_case(1, 0);
    failures += greedy_accept_case(5, 2);
    failures += greedy_accept_case(5, 5);
    failures += greedy_accept_case(15, 7, 257);
    failures += greedy_tree_extent_case(7, 1);
    failures += greedy_tree_extent_case(0, 0);
    failures += greedy_tree_second_child_case(64);
    failures += greedy_tree_second_child_case(257);
    failures += tree_sampling_membership_cases(64);
    failures += tree_sampling_membership_cases(257);
    failures += tree_sampling_presence_overlay_case(64);
    failures += tree_sampling_presence_overlay_case(257);
    failures += deterministic_sampling_case();
    failures += suppressed_bonus_case();
    failures += suppressed_draft_rejection_case(64);
    failures += suppressed_draft_rejection_case(257);
    failures += p_less_deterministic_accept_case();
    failures += p_less_tree_membership_cases(64);
    failures += p_less_tree_membership_cases(257);
    failures += batched_sampling_workspace_stride_case();
    failures += select_hidden_case(5120, 6, 0);
    failures += select_hidden_case(5120, 6, 5);
    failures += select_hidden_case(2048, 16, 7);
    failures += remap_case(1);
    failures += remap_case(15);
    failures += remap_case(120);
    failures += greedy_ignores_selector_case();
    failures += onehot_selector_matches_null_case();
    failures += zero_q_rejects_and_corrects_case(64);
    failures += zero_q_rejects_and_corrects_case(257);
    failures += fractional_q_accepts_when_p_covers_q();
    failures += fractional_q_rejects_when_p_is_zero();
    failures += residual_p_minus_q_prefers_uncovered_mass();
    failures += batched_selector_row_isolation_case();

    if (failures != 0) {
        std::cerr << "speculative_round failures=" << failures << '\n';
        return 1;
    }
    std::cout << "speculative_round: PASS\n";
    return 0;
}
