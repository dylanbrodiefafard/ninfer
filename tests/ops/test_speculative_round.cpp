#include "ninfer/ops/sampling.h"
#include "ninfer/ops/speculative_round.h"
#include "ops/op_tester.h"

#include <algorithm>
#include <cmath>
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

unsigned long long host_splitmix64(unsigned long long x) {
    x += 0x9E3779B97F4A7C15ull;
    x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ull;
    x = (x ^ (x >> 27)) * 0x94D049BB133111EBull;
    return x ^ (x >> 31);
}

float host_sampling_uniform(unsigned long long seed, int position, int purpose) {
    constexpr unsigned int sub = 0u;
    unsigned long long key     = seed;
    key                        = host_splitmix64(
        key ^ (static_cast<unsigned long long>(static_cast<unsigned int>(position)) *
               0xD1B54A32D192ED03ull));
    key = host_splitmix64(
        key ^ (static_cast<unsigned long long>(static_cast<unsigned int>(purpose)) << 21) ^
        (static_cast<unsigned long long>(sub) * 0x2545F4914F6CDD1Dull));
    return static_cast<float>(static_cast<unsigned int>(key >> 40)) * (1.0f / 16777216.0f);
}

unsigned long long seed_with_uniform_in(int position, int purpose, float lo, float hi) {
    for (unsigned long long seed = 1; seed < 200000ull; ++seed) {
        const float u = host_sampling_uniform(seed, position, purpose);
        if (u >= lo && u < hi) { return seed; }
    }
    throw std::logic_error("no sampling seed in the requested uniform interval");
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

int p_less_suppressed_bonus_case() {
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
    config.p_less                 = 1;
    config.seed                   = 7ull;
    config.suppressed_token_count = 1;
    config.suppressed_tokens[0]   = 9;
    return execute_accept_case("speculative p-less suppresses EOS in accept and bonus", targets,
                               logits_bits, token_domain, drafts, initial_length, token_domain,
                               config, token_counts, expected);
}

int p_less_suppressed_draft_rejection_case(int token_domain) {
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
    config.p_less                 = 1;
    config.seed                   = 7ull;
    config.suppressed_token_count = 1;
    config.suppressed_tokens[0]   = 9;
    return execute_accept_case("speculative p-less rejects suppressed draft V=" +
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

int p_less_suppressed_distractor_accept_case() {
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
        logits[base + 13]                               = 30.0f;
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
    config.temperature            = 1.0f;
    config.p_less                 = 1;
    config.seed                   = 0x123456789abcdef0ull;
    config.suppressed_token_count = 1;
    config.suppressed_tokens[0]   = 13;
    return execute_accept_case("speculative p-less suppresses distractor at real shape", targets,
                               logits_bits, physical_rows, drafts, initial_length, token_domain,
                               config, token_counts, expected);
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
    {
        ops::SamplingConfig suppressed_cfg        = cfg;
        suppressed_cfg.suppressed_token_count     = 1;
        suppressed_cfg.suppressed_tokens[0]       = 11;
        std::vector<float> logits_f(static_cast<std::size_t>(token_domain) * kWidth, -20.0f);
        logits_f[11] = 20.0f;
        logits_f[5]  = 19.0f;
        round_to_bf16(logits_f);
        std::vector<std::uint16_t> logits(logits_f.size());
        for (std::size_t i = 0; i < logits_f.size(); ++i) { logits[i] = f32_to_bf16(logits_f[i]); }
        std::vector<std::int32_t> want_lic{5, 0, 0, 0};
        std::vector<std::int32_t> want_path{0, 0, 0, 0};
        failures += execute_tree_case("tree p-less rejects suppressed draft" + tag, token_domain,
                                      token_domain, kWidth, parent, verify_ids, targets, logits, 7,
                                      kWidth, suppressed_cfg, counts, want_lic, 0, 0, want_path);
    }
    return failures;
}

int p_less_dflash2_product_tree_multiblock_case() {
    constexpr int physical_rows = 248320;
    constexpr int token_domain  = 248077;
    constexpr int kWidth        = 12;
    const std::vector<std::int32_t> parent{-1, 0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5};
    const std::vector<std::int32_t> verify_ids{
        7, 17, 7919, 65537, 131071, 200003, 240001, 29, 3001, 50021, 170003, 230003};
    constexpr std::int32_t correction = 150001;

    std::vector<float> logits(static_cast<std::size_t>(physical_rows) * kWidth, -20.0f);
    const auto set = [&](int col, int token, float value) {
        logits[static_cast<std::size_t>(col) * physical_rows + token] = value;
    };
    set(0, verify_ids[2], 20.0f);
    set(2, verify_ids[6], 20.0f);
    set(6, correction, 20.0f);
    for (int col = 0; col < kWidth; ++col) {
        set(col, token_domain, 100.0f);
        set(col, physical_rows - 1, 200.0f);
    }
    round_to_bf16(logits);
    std::vector<std::uint16_t> bits(logits.size());
    for (std::size_t i = 0; i < logits.size(); ++i) { bits[i] = f32_to_bf16(logits[i]); }

    ops::SamplingConfig cfg{};
    cfg.temperature       = 1.0f;
    cfg.top_k             = 1;
    cfg.top_p             = 0.1f;
    cfg.min_p             = 0.9f;
    cfg.presence_penalty  = 2.0f;
    cfg.frequency_penalty = 2.0f;
    cfg.p_less            = 1;
    cfg.seed              = 0x123456789abcdef0ull;

    std::vector<std::int32_t> counts(static_cast<std::size_t>(token_domain), 0);
    counts[static_cast<std::size_t>(verify_ids[2])] = 3;
    counts[static_cast<std::size_t>(verify_ids[6])] = 5;
    counts[static_cast<std::size_t>(correction)]    = 7;
    std::vector<std::int32_t> want_lic(kWidth, 0);
    want_lic[0] = verify_ids[2];
    want_lic[1] = verify_ids[6];
    want_lic[2] = correction;
    std::vector<std::int32_t> want_path(kWidth, 0);
    want_path[0] = 0;
    want_path[1] = 2;
    want_path[2] = 6;
    return execute_tree_case("DFlash2 p-less tree W=12 real token-domain", token_domain,
                             physical_rows, kWidth, parent, verify_ids,
                             std::vector<std::int32_t>(kWidth, 0), bits, 7, kWidth, cfg, counts,
                             want_lic, 2, 6, want_path, 4093);
}

// Live serve is DFlash2 tree k=7 / W=12 / p-less T=2 / C=2. Greedy C>1 isolation
// does not enter this kernel: p-less tree uses the multiblock mass-finalize walk
// with one workspace replica per compact row. If that stride is wrong, row 1
// samples row 0's columns (OpenCode: one concurrent story stays coherent, the
// other emits token salad).
int p_less_tree_batch_row_isolation_case(int physical_rows, int token_domain, const char* label) {
    constexpr int kWidth = 12;
    constexpr int kBatch = 2;
    const std::vector<std::int32_t> parent{-1, 0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5};
    const std::vector<std::int32_t> ids0{7, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59};
    const std::vector<std::int32_t> ids1{8, 18, 20, 24, 30, 32, 38, 42, 44, 48, 54, 60};
    constexpr int peak0[] = {19, 37, 61};
    constexpr int peak1[] = {20, 38, 62};
    for (int token : {peak0[0], peak0[1], peak0[2], peak1[0], peak1[1], peak1[2]}) {
        if (token < 0 || token >= token_domain) {
            std::cerr << label << ": peak token " << token << " outside token_domain\n";
            return 1;
        }
    }

    std::vector<std::int32_t> parent_batch;
    std::vector<std::int32_t> ids_batch;
    parent_batch.reserve(static_cast<std::size_t>(kWidth) * kBatch);
    ids_batch.reserve(static_cast<std::size_t>(kWidth) * kBatch);
    parent_batch.insert(parent_batch.end(), parent.begin(), parent.end());
    parent_batch.insert(parent_batch.end(), parent.begin(), parent.end());
    ids_batch.insert(ids_batch.end(), ids0.begin(), ids0.end());
    ids_batch.insert(ids_batch.end(), ids1.begin(), ids1.end());

    std::vector<float> logits_f(static_cast<std::size_t>(physical_rows) * kWidth * kBatch, -20.0f);
    const auto plant = [&](int row, int col, int token, float value) {
        const std::size_t base =
            (static_cast<std::size_t>(row) * kWidth + static_cast<std::size_t>(col)) *
            static_cast<std::size_t>(physical_rows);
        logits_f[base + static_cast<std::size_t>(token)] = value;
    };
    plant(0, 0, peak0[0], 20.0f);
    plant(0, 2, peak0[1], 20.0f);
    plant(0, 6, peak0[2], 20.0f);
    plant(1, 0, peak1[0], 20.0f);
    plant(1, 2, peak1[1], 20.0f);
    plant(1, 6, peak1[2], 20.0f);
    if (physical_rows > token_domain) {
        for (int row = 0; row < kBatch; ++row) {
            for (int col = 0; col < kWidth; ++col) {
                plant(row, col, token_domain, 100.0f);
                plant(row, col, physical_rows - 1, 200.0f);
            }
        }
    }
    round_to_bf16(logits_f);
    std::vector<std::uint16_t> bits(logits_f.size());
    for (std::size_t i = 0; i < logits_f.size(); ++i) { bits[i] = f32_to_bf16(logits_f[i]); }

    DeviceBuffer d_targets =
        to_device(std::vector<std::int32_t>(static_cast<std::size_t>(kWidth) * kBatch, 0));
    DeviceBuffer d_logits = to_device(bits);
    DeviceBuffer d_ids    = to_device(ids_batch);
    DeviceBuffer d_parent = to_device(parent_batch);
    DeviceBuffer d_valid  = to_device<std::int32_t>({kWidth, kWidth});
    DeviceBuffer d_extent = to_device<std::int32_t>({7, 7});

    GuardedDeviceBuffer d_counts0(static_cast<std::size_t>(token_domain) * sizeof(std::int32_t));
    GuardedDeviceBuffer d_counts1(static_cast<std::size_t>(token_domain) * sizeof(std::int32_t));
    d_counts0.fill(0);
    d_counts1.fill(0);
    ops::SamplingConfig cfg{};
    cfg.temperature = 2.0f;
    cfg.p_less      = 1;
    cfg.seed        = 0x123456789abcdef0ull;
    std::vector<ops::SamplingConfig> configs{cfg, cfg};
    configs[0].token_counts = static_cast<std::int32_t*>(d_counts0.data());
    configs[1].token_counts = static_cast<std::int32_t*>(d_counts1.data());
    DeviceBuffer d_configs  = to_device(configs);

    GuardedDeviceBuffer d_length(kBatch * sizeof(std::int32_t));
    GuardedDeviceBuffer d_anchors(kBatch * sizeof(std::int32_t));
    GuardedDeviceBuffer d_licensed(static_cast<std::size_t>(kWidth) * kBatch * sizeof(std::int32_t));
    GuardedDeviceBuffer d_lic_count(kBatch * sizeof(std::int32_t));
    GuardedDeviceBuffer d_accepted(kBatch * sizeof(std::int32_t));
    GuardedDeviceBuffer d_column(kBatch * sizeof(std::int32_t));
    GuardedDeviceBuffer d_path(static_cast<std::size_t>(kWidth) * kBatch * sizeof(std::int32_t));
    initialize(d_length, std::vector<std::int32_t>{40, 80});
    initialize(d_anchors, std::vector<std::int32_t>{-1, -1});
    d_licensed.fill(0x9d);
    initialize(d_lic_count, std::vector<std::int32_t>{-1, -1});
    initialize(d_accepted, std::vector<std::int32_t>{-1, -1});
    initialize(d_column, std::vector<std::int32_t>{-1, -1});
    d_path.fill(0x9d);

    Tensor target_t(d_targets.p, DType::I32, {kWidth, kBatch});
    Tensor logits_t(d_logits.p, DType::BF16, {physical_rows, kWidth, kBatch});
    Tensor ids_t(d_ids.p, DType::I32, {kWidth, kBatch});
    Tensor parent_t(d_parent.p, DType::I32, {kWidth, kBatch});
    Tensor valid_t(d_valid.p, DType::I32, {kBatch});
    Tensor extent_t(d_extent.p, DType::I32, {kBatch});
    Tensor length_t(d_length.data(), DType::I32, {kBatch});
    Tensor anchors_t(d_anchors.data(), DType::I32, {kBatch});
    Tensor licensed_t(d_licensed.data(), DType::I32, {kWidth, kBatch});
    Tensor lic_count_t(d_lic_count.data(), DType::I32, {kBatch});
    Tensor accepted_t(d_accepted.data(), DType::I32, {kBatch});
    Tensor column_t(d_column.data(), DType::I32, {kBatch});
    Tensor path_t(d_path.data(), DType::I32, {kWidth, kBatch});
    const std::size_t workspace_bytes =
        ops::speculative_accept_tree_drafts_workspace_capacity_bytes(token_domain, kWidth, kWidth,
                                                                     kBatch, kBatch);
    WorkspaceArena workspace(std::max<std::size_t>(256, workspace_bytes));
    ops::speculative_accept_tree_drafts(
        target_t, logits_t, ids_t, parent_t, valid_t, extent_t, length_t, anchors_t, licensed_t,
        lic_count_t, accepted_t, column_t, path_t, token_domain,
        static_cast<const ops::SamplingConfig*>(d_configs.p), workspace, nullptr);
    cuda_synchronize();

    std::vector<std::int32_t> want_lic(static_cast<std::size_t>(kWidth) * kBatch, 0);
    want_lic[0]                                      = peak0[0];
    want_lic[1]                                      = peak0[1];
    want_lic[2]                                      = peak0[2];
    want_lic[static_cast<std::size_t>(kWidth)]       = peak1[0];
    want_lic[static_cast<std::size_t>(kWidth) + 1]   = peak1[1];
    want_lic[static_cast<std::size_t>(kWidth) + 2]   = peak1[2];
    std::vector<std::int32_t> want_path(static_cast<std::size_t>(kWidth) * kBatch, 0);
    want_path[0]                                    = 0;
    want_path[1]                                    = 2;
    want_path[2]                                    = 6;
    want_path[static_cast<std::size_t>(kWidth)]     = 0;
    want_path[static_cast<std::size_t>(kWidth) + 1] = 2;
    want_path[static_cast<std::size_t>(kWidth) + 2] = 6;

    int failures = verify_exact((std::string(label) + " licensed").c_str(),
                                read<std::int32_t>(d_licensed, static_cast<std::size_t>(kWidth) * kBatch),
                                want_lic);
    failures += verify_exact((std::string(label) + " path").c_str(),
                             read<std::int32_t>(d_path, static_cast<std::size_t>(kWidth) * kBatch),
                             want_path);
    failures += verify_exact((std::string(label) + " accepted").c_str(),
                             read<std::int32_t>(d_accepted, kBatch), {2, 2});
    failures += verify_exact((std::string(label) + " licensed count").c_str(),
                             read<std::int32_t>(d_lic_count, kBatch), {3, 3});
    failures += verify_exact((std::string(label) + " column").c_str(),
                             read<std::int32_t>(d_column, kBatch), {6, 6});
    failures += verify_exact((std::string(label) + " anchors").c_str(),
                             read<std::int32_t>(d_anchors, kBatch), {peak0[2], peak1[2]});
    failures += verify_exact((std::string(label) + " lengths").c_str(),
                             read<std::int32_t>(d_length, kBatch), {43, 83});
    failures += d_licensed.verify_guards((std::string(label) + " licensed guards").c_str());
    failures += d_path.verify_guards((std::string(label) + " path guards").c_str());
    failures += d_counts0.verify_guards((std::string(label) + " counts0 guards").c_str());
    failures += d_counts1.verify_guards((std::string(label) + " counts1 guards").c_str());
    if (workspace.used() != 0 || workspace.peak_used() != workspace_bytes) {
        std::cerr << label << ": workspace query/execution high-water mismatch\n";
        ++failures;
    }
    return failures;
}

// Same B=2 layout as the peaked walk, but T=2 p-less support is two in-domain
// tokens per row (the live coding sampler). A compact-row mass-finalize race
// would leak the other row's survivors into licensed_tokens.
int p_less_tree_batch_flat_support_isolation_case(int physical_rows, int token_domain,
                                                  unsigned long long seeds, const char* label) {
    constexpr int kWidth = 4;
    constexpr int kBatch = 2;
    const std::vector<std::int32_t> parent{-1, 0, 0, 1};
    const std::vector<std::int32_t> ids0{7, 10, 11, 12};
    const std::vector<std::int32_t> ids1{8, 20, 21, 22};
    const int row0_a = 10;
    const int row0_b = 11;
    const int row1_a = 20;
    const int row1_b = 21;
    std::vector<std::int32_t> parent_batch;
    std::vector<std::int32_t> ids_batch;
    parent_batch.insert(parent_batch.end(), parent.begin(), parent.end());
    parent_batch.insert(parent_batch.end(), parent.begin(), parent.end());
    ids_batch.insert(ids_batch.end(), ids0.begin(), ids0.end());
    ids_batch.insert(ids_batch.end(), ids1.begin(), ids1.end());

    std::vector<float> logits_f(static_cast<std::size_t>(physical_rows) * kWidth * kBatch, -20.0f);
    const auto plant = [&](int row, int col, int token) {
        const std::size_t base =
            (static_cast<std::size_t>(row) * kWidth + static_cast<std::size_t>(col)) *
            static_cast<std::size_t>(physical_rows);
        logits_f[base + static_cast<std::size_t>(token)] = 20.0f;
    };
    for (int col = 0; col < kWidth; ++col) {
        plant(0, col, row0_a);
        plant(0, col, row0_b);
        plant(1, col, row1_a);
        plant(1, col, row1_b);
        if (physical_rows > token_domain) {
            const std::size_t b0 =
                static_cast<std::size_t>(col) * static_cast<std::size_t>(physical_rows);
            const std::size_t b1 =
                (static_cast<std::size_t>(kWidth) + static_cast<std::size_t>(col)) *
                static_cast<std::size_t>(physical_rows);
            logits_f[b0 + static_cast<std::size_t>(token_domain)]      = 100.0f;
            logits_f[b0 + static_cast<std::size_t>(physical_rows - 1)] = 200.0f;
            logits_f[b1 + static_cast<std::size_t>(token_domain)]      = 100.0f;
            logits_f[b1 + static_cast<std::size_t>(physical_rows - 1)] = 200.0f;
        }
    }
    round_to_bf16(logits_f);
    std::vector<std::uint16_t> bits(logits_f.size());
    for (std::size_t i = 0; i < logits_f.size(); ++i) { bits[i] = f32_to_bf16(logits_f[i]); }

    DeviceBuffer d_targets =
        to_device(std::vector<std::int32_t>(static_cast<std::size_t>(kWidth) * kBatch, 0));
    DeviceBuffer d_logits = to_device(bits);
    DeviceBuffer d_ids    = to_device(ids_batch);
    DeviceBuffer d_parent = to_device(parent_batch);
    DeviceBuffer d_valid  = to_device<std::int32_t>({kWidth, kWidth});
    DeviceBuffer d_extent = to_device<std::int32_t>({1, 1});
    DeviceBuffer d_length = to_device<std::int32_t>({40, 80});
    DeviceBuffer d_anchors = to_device<std::int32_t>({-1, -1});
    DeviceBuffer d_licensed(static_cast<std::size_t>(kWidth) * kBatch * sizeof(std::int32_t));
    DeviceBuffer d_lic_count(kBatch * sizeof(std::int32_t));
    DeviceBuffer d_accepted(kBatch * sizeof(std::int32_t));
    DeviceBuffer d_column(kBatch * sizeof(std::int32_t));
    DeviceBuffer d_path(static_cast<std::size_t>(kWidth) * kBatch * sizeof(std::int32_t));
    DeviceBuffer d_counts0 = to_device(std::vector<std::int32_t>(static_cast<std::size_t>(token_domain), 0));
    DeviceBuffer d_counts1 = to_device(std::vector<std::int32_t>(static_cast<std::size_t>(token_domain), 0));

    Tensor target_t(d_targets.p, DType::I32, {kWidth, kBatch});
    Tensor logits_t(d_logits.p, DType::BF16, {physical_rows, kWidth, kBatch});
    Tensor ids_t(d_ids.p, DType::I32, {kWidth, kBatch});
    Tensor parent_t(d_parent.p, DType::I32, {kWidth, kBatch});
    Tensor valid_t(d_valid.p, DType::I32, {kBatch});
    Tensor extent_t(d_extent.p, DType::I32, {kBatch});
    Tensor length_t(d_length.p, DType::I32, {kBatch});
    Tensor anchors_t(d_anchors.p, DType::I32, {kBatch});
    Tensor licensed_t(d_licensed.p, DType::I32, {kWidth, kBatch});
    Tensor lic_count_t(d_lic_count.p, DType::I32, {kBatch});
    Tensor accepted_t(d_accepted.p, DType::I32, {kBatch});
    Tensor column_t(d_column.p, DType::I32, {kBatch});
    Tensor path_t(d_path.p, DType::I32, {kWidth, kBatch});
    const std::size_t workspace_bytes =
        ops::speculative_accept_tree_drafts_workspace_capacity_bytes(token_domain, kWidth, kWidth,
                                                                     kBatch, kBatch);
    int failures = 0;
    for (unsigned long long seed = 1; seed <= seeds; ++seed) {
        ops::SamplingConfig cfg{};
        cfg.temperature = 2.0f;
        cfg.p_less      = 1;
        cfg.seed        = seed;
        std::vector<ops::SamplingConfig> configs{cfg, cfg};
        configs[0].token_counts = static_cast<std::int32_t*>(d_counts0.p);
        configs[1].token_counts = static_cast<std::int32_t*>(d_counts1.p);
        DeviceBuffer d_configs  = to_device(configs);
        WorkspaceArena workspace(std::max<std::size_t>(256, workspace_bytes));
        ops::speculative_accept_tree_drafts(
            target_t, logits_t, ids_t, parent_t, valid_t, extent_t, length_t, anchors_t,
            licensed_t, lic_count_t, accepted_t, column_t, path_t, token_domain,
            static_cast<const ops::SamplingConfig*>(d_configs.p), workspace, nullptr);
        cuda_synchronize();
        const auto licensed =
            from_device<std::int32_t>(d_licensed, static_cast<std::size_t>(kWidth) * kBatch);
        const auto counts = from_device<std::int32_t>(d_lic_count, kBatch);
        for (int row = 0; row < kBatch; ++row) {
            const int n = counts[static_cast<std::size_t>(row)];
            if (n < 1 || n > kWidth) {
                std::cerr << label << " seed=" << seed << " row=" << row
                          << " licensed_count=" << n << '\n';
                ++failures;
                continue;
            }
            for (int i = 0; i < n; ++i) {
                const int tok =
                    licensed[static_cast<std::size_t>(row) * kWidth + static_cast<std::size_t>(i)];
                const bool ok0 = tok == row0_a || tok == row0_b;
                const bool ok1 = tok == row1_a || tok == row1_b;
                if (row == 0 && !ok0) {
                    std::cerr << label << " seed=" << seed << " row 0 licensed " << tok
                              << " is outside {10,11}\n";
                    ++failures;
                }
                if (row == 1 && !ok1) {
                    std::cerr << label << " seed=" << seed << " row 1 licensed " << tok
                              << " is outside {20,21}\n";
                    ++failures;
                }
            }
        }
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

int p_less_ignores_selector_q_when_u_exceeds_p(int physical_rows, int token_domain, int uncovered,
                                               const char* label) {
    // Adaptive DFlash may still present a T=2 16-way softmax q (hot-patch missed the
    // path-select object, or a stale graph buffer). P-less Leviathan must use one-hot q
    // like MTP: u > p' rejects even when p'/q_16way > 1.
    constexpr int k     = 1;
    constexpr int draft = 7;
    const std::vector<std::int32_t> drafts{draft};
    const std::vector<std::int32_t> targets{3, 11};
    std::vector<float> logits(static_cast<std::size_t>(physical_rows) * (k + 1), -20.0f);
    logits[draft]                                              = 20.0f;
    logits[uncovered]                                          = 20.0f;
    logits[static_cast<std::int64_t>(physical_rows) + uncovered] = 20.0f;
    if (physical_rows > token_domain) {
        logits[token_domain]                                     = 100.0f;
        logits[physical_rows - 1]                                = 200.0f;
        logits[static_cast<std::int64_t>(physical_rows) + token_domain]      = 100.0f;
        logits[static_cast<std::int64_t>(physical_rows) + physical_rows - 1] = 200.0f;
    }
    round_to_bf16(logits);
    std::vector<std::uint16_t> logits_bits(logits.size());
    for (std::size_t i = 0; i < logits.size(); ++i) { logits_bits[i] = f32_to_bf16(logits[i]); }

    constexpr std::int32_t initial_length = 40;
    const auto expected = accept_state_oracle(drafts, 0, uncovered, initial_length);
    std::vector<std::int32_t> token_counts(token_domain, 0);
    ops::SamplingConfig config{};
    config.temperature = 2.0f;
    config.p_less      = 1;
    config.seed        = seed_with_uniform_in(initial_length + 1, ops::kSamplePurposeSpeculativeAccept,
                                              0.51f, 0.99f);
    constexpr int cap  = 16;
    std::vector<std::int32_t> ids(static_cast<std::size_t>(cap) * k, 0);
    std::vector<float> q(static_cast<std::size_t>(cap) * k, 1.0f / static_cast<float>(cap));
    ids[0] = draft;
    for (int c = 1; c < cap; ++c) { ids[static_cast<std::size_t>(c)] = 2000 + c; }
    return execute_accept_case(label, targets, logits_bits, physical_rows, drafts, initial_length,
                               token_domain, config, token_counts, expected, &ids, &q);
}

int p_less_fractional_q_residual_does_not_reemit_draft(int physical_rows, int token_domain,
                                                       int uncovered, const char* label) {
    // Adaptive DFlash chain writes a 16-way q. P-less V={draft, uncovered} with q(draft)
    // covering the draft's p' must reject and sample the uncovered residual token — never
    // fall back to the rejected draft (that loop is the observed punctuation collapse).
    constexpr int k     = 1;
    constexpr int draft = 7;
    const std::vector<std::int32_t> drafts{draft};
    const std::vector<std::int32_t> targets{3, 11};
    std::vector<float> logits(static_cast<std::size_t>(physical_rows) * (k + 1), -20.0f);
    logits[draft]                                              = 20.0f;
    logits[uncovered]                                          = 20.0f;
    logits[static_cast<std::size_t>(physical_rows) + uncovered] = 20.0f;
    if (physical_rows > token_domain) {
        logits[token_domain]                                     = 100.0f;
        logits[physical_rows - 1]                                = 200.0f;
        logits[static_cast<std::size_t>(physical_rows) + token_domain]      = 100.0f;
        logits[static_cast<std::size_t>(physical_rows) + physical_rows - 1] = 200.0f;
    }
    round_to_bf16(logits);
    std::vector<std::uint16_t> logits_bits(logits.size());
    for (std::size_t i = 0; i < logits.size(); ++i) { logits_bits[i] = f32_to_bf16(logits[i]); }

    constexpr std::int32_t initial_length = 40;
    const auto expected = accept_state_oracle(drafts, 0, uncovered, initial_length);
    std::vector<std::int32_t> token_counts(token_domain, 0);
    ops::SamplingConfig config{};
    config.temperature = 2.0f;
    config.p_less      = 1;
    config.seed        = seed_with_uniform_in(initial_length + 1, ops::kSamplePurposeSpeculativeAccept,
                                              0.625f, 1.0f);
    constexpr int cap  = 16;
    std::vector<std::int32_t> ids(static_cast<std::size_t>(cap) * k, 0);
    std::vector<float> q(static_cast<std::size_t>(cap) * k, 0.0f);
    ids[0] = draft;
    q[0]   = 0.8f;
    ids[1] = 2001;
    q[1]   = 0.2f;
    return execute_accept_case(label, targets, logits_bits, physical_rows, drafts, initial_length,
                               token_domain, config, token_counts, expected, &ids, &q);
}

bool p_less_token_suppressed(const ops::SamplingConfig& config, int token) {
    const int count = std::min(config.suppressed_token_count, ops::SamplingConfig::kMaximumSuppressedTokens);
    for (int i = 0; i < count; ++i) {
        if (config.suppressed_tokens[i] == token) { return true; }
    }
    return false;
}

std::vector<int> p_less_support_oracle(const std::vector<float>& logits, int physical_rows, int col,
                                       int token_domain, const ops::SamplingConfig& config) {
    const std::size_t base = static_cast<std::size_t>(col) * static_cast<std::size_t>(physical_rows);
    double max_scaled      = 0.0;
    bool have_max          = false;
    for (int token = 0; token < token_domain; ++token) {
        if (p_less_token_suppressed(config, token)) { continue; }
        const double scaled =
            static_cast<double>(logits[base + static_cast<std::size_t>(token)]) / config.temperature;
        if (!have_max || scaled > max_scaled) {
            max_scaled = scaled;
            have_max   = true;
        }
    }
    std::vector<double> weights(static_cast<std::size_t>(token_domain), 0.0);
    double total = 0.0;
    for (int token = 0; token < token_domain; ++token) {
        if (p_less_token_suppressed(config, token)) { continue; }
        const double w = std::exp(
            static_cast<double>(logits[base + static_cast<std::size_t>(token)]) / config.temperature -
            max_scaled);
        weights[static_cast<std::size_t>(token)] = w;
        total += w;
    }
    double collision = 0.0;
    for (int token = 0; token < token_domain; ++token) {
        if (p_less_token_suppressed(config, token) || !(total > 0.0)) { continue; }
        const double p = weights[static_cast<std::size_t>(token)] / total;
        collision += p * p;
    }
    std::vector<int> support;
    for (int token = 0; token < token_domain; ++token) {
        if (p_less_token_suppressed(config, token) || !(total > 0.0)) { continue; }
        const double p = weights[static_cast<std::size_t>(token)] / total;
        if (p >= collision) { support.push_back(token); }
    }
    return support;
}

bool p_less_support_contains(const std::vector<int>& support, int token) {
    return std::find(support.begin(), support.end(), token) != support.end();
}

struct ChainAcceptObserved {
    std::vector<std::int32_t> licensed;
    std::int32_t count    = 0;
    std::int32_t accepted = 0;
    std::int32_t length   = 0;
    std::int32_t anchor   = 0;
};

ChainAcceptObserved run_chain_accept(const std::vector<std::uint16_t>& logits_bits, int physical_rows,
                                     const std::vector<std::int32_t>& drafts,
                                     std::int32_t initial_length, int token_domain,
                                     ops::SamplingConfig config,
                                     const std::vector<std::int32_t>* selector_ids,
                                     const std::vector<float>* selector_q) {
    const int k            = static_cast<int>(drafts.size());
    const int cols         = k + 1;
    std::vector<std::int32_t> targets(static_cast<std::size_t>(cols), 0);
    DeviceBuffer d_targets = to_device(targets);
    DeviceBuffer d_logits  = to_device(logits_bits);
    DeviceBuffer d_drafts  = to_device(drafts);
    std::vector<std::int32_t> token_counts(static_cast<std::size_t>(token_domain), 0);
    DeviceBuffer d_counts  = to_device(token_counts);
    config.token_counts    = static_cast<std::int32_t*>(d_counts.p);
    DeviceBuffer d_config  = device_config(config);
    DeviceBuffer d_length  = to_device<std::int32_t>({initial_length});
    DeviceBuffer d_token   = to_device<std::int32_t>({-1});
    DeviceBuffer d_sampled = to_device(std::vector<std::int32_t>(static_cast<std::size_t>(cols), 0));
    DeviceBuffer d_num     = to_device<std::int32_t>({-11});
    DeviceBuffer d_accepted = to_device<std::int32_t>({-13});
    DeviceBuffer d_extent  = to_device<std::int32_t>({k});

    Tensor target_t(d_targets.p, DType::I32, {cols});
    Tensor logits_t(d_logits.p, DType::BF16, {physical_rows, cols});
    Tensor draft_t(d_drafts.p, DType::I32, {k});
    Tensor extent(d_extent.p, DType::I32, {1});
    Tensor length(d_length.p, DType::I32, {1});
    Tensor token(d_token.p, DType::I32, {1});
    Tensor sampled(d_sampled.p, DType::I32, {cols});
    Tensor num_sampled(d_num.p, DType::I32, {1});
    Tensor accepted(d_accepted.p, DType::I32, {1});
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
        target_t, logits_t, draft_t, extent, length, token, sampled, num_sampled, accepted,
        token_domain, static_cast<const ops::SamplingConfig*>(d_config.p), workspace, nullptr,
        sel_ids_arg, sel_q_arg);
    cuda_synchronize();

    ChainAcceptObserved out;
    out.licensed = from_device<std::int32_t>(d_sampled, static_cast<std::size_t>(cols));
    out.count    = from_device<std::int32_t>(d_num, 1)[0];
    out.accepted = from_device<std::int32_t>(d_accepted, 1)[0];
    out.length   = from_device<std::int32_t>(d_length, 1)[0];
    out.anchor   = from_device<std::int32_t>(d_token, 1)[0];
    return out;
}

int check_p_less_chain_invariants(const char* label, const ChainAcceptObserved& got,
                                  const std::vector<float>& logits, int physical_rows,
                                  const std::vector<std::int32_t>& drafts, int token_domain,
                                  const ops::SamplingConfig& config, std::int32_t initial_length) {
    const int k = static_cast<int>(drafts.size());
    int failures = 0;
    auto fail    = [&](const std::string& msg) {
        std::cerr << label << ": " << msg << '\n';
        ++failures;
    };
    if (got.count < 1 || got.count > k + 1 || got.accepted < 0 || got.accepted + 1 != got.count ||
        got.accepted > k) {
        fail("produced count/accepted is not a valid chain prefix");
        return failures;
    }
    if (got.length != initial_length + got.count) {
        fail("length did not advance by the produced count");
    }
    if (got.anchor != got.licensed[static_cast<std::size_t>(got.accepted)]) {
        fail("anchor is not the correction/bonus token");
    }
    for (int i = 0; i < got.count; ++i) {
        const int token = got.licensed[static_cast<std::size_t>(i)];
        if (token < 0 || token >= token_domain) {
            fail("licensed token " + std::to_string(token) + " is outside token_domain");
            continue;
        }
        if (p_less_token_suppressed(config, token)) {
            fail("licensed suppressed token " + std::to_string(token));
        }
        const auto support = p_less_support_oracle(logits, physical_rows, i, token_domain, config);
        if (support.empty() || !p_less_support_contains(support, token)) {
            fail("licensed token " + std::to_string(token) + " is outside p-less support at hop " +
                 std::to_string(i));
        }
    }
    for (int i = got.count; i <= k; ++i) {
        if (got.licensed[static_cast<std::size_t>(i)] != 0) {
            fail("unused licensed slot was not cleared");
        }
    }
    for (int i = 0; i < got.accepted; ++i) {
        if (got.licensed[static_cast<std::size_t>(i)] != drafts[static_cast<std::size_t>(i)]) {
            fail("accepted prefix does not match the draft chain");
        }
    }
    if (got.accepted < k) {
        const int rejected = drafts[static_cast<std::size_t>(got.accepted)];
        const auto support =
            p_less_support_oracle(logits, physical_rows, got.accepted, token_domain, config);
        if (support.size() > 1 &&
            got.licensed[static_cast<std::size_t>(got.accepted)] == rejected) {
            fail("rejected draft " + std::to_string(rejected) +
                 " was re-emitted as the correction");
        }
    }
    return failures;
}

std::vector<float> peaked_p_less_chain_logits(int physical_rows, int token_domain, int cols,
                                              const std::vector<int>& survivors, float peak) {
    std::vector<float> logits(static_cast<std::size_t>(physical_rows) * static_cast<std::size_t>(cols),
                              -20.0f);
    for (int col = 0; col < cols; ++col) {
        const std::size_t base = static_cast<std::size_t>(col) * static_cast<std::size_t>(physical_rows);
        for (int token : survivors) {
            if (token >= 0 && token < token_domain) {
                logits[base + static_cast<std::size_t>(token)] = peak;
            }
        }
        if (physical_rows > token_domain) {
            logits[base + static_cast<std::size_t>(token_domain)]      = 100.0f;
            logits[base + static_cast<std::size_t>(physical_rows - 1)] = 200.0f;
        }
    }
    round_to_bf16(logits);
    return logits;
}

std::vector<float> peaked_p_less_per_column_logits(int physical_rows, int token_domain,
                                                   const std::vector<std::vector<int>>& survivors) {
    const int cols = static_cast<int>(survivors.size());
    std::vector<float> logits(static_cast<std::size_t>(physical_rows) * static_cast<std::size_t>(cols),
                              -20.0f);
    for (int col = 0; col < cols; ++col) {
        const std::size_t base = static_cast<std::size_t>(col) * static_cast<std::size_t>(physical_rows);
        for (int token : survivors[static_cast<std::size_t>(col)]) {
            if (token >= 0 && token < token_domain) {
                logits[base + static_cast<std::size_t>(token)] = 20.0f;
            }
        }
        if (physical_rows > token_domain) {
            logits[base + static_cast<std::size_t>(token_domain)]      = 100.0f;
            logits[base + static_cast<std::size_t>(physical_rows - 1)] = 200.0f;
        }
    }
    round_to_bf16(logits);
    return logits;
}

std::pair<std::vector<std::int32_t>, std::vector<float>>
uniform_selector_q(const std::vector<std::int32_t>& drafts, int cap = 16) {
    const int k = static_cast<int>(drafts.size());
    std::vector<std::int32_t> ids(static_cast<std::size_t>(cap) * static_cast<std::size_t>(k), 0);
    std::vector<float> q(static_cast<std::size_t>(cap) * static_cast<std::size_t>(k),
                         1.0f / static_cast<float>(cap));
    for (int hop = 0; hop < k; ++hop) {
        ids[static_cast<std::size_t>(hop) * static_cast<std::size_t>(cap)] =
            drafts[static_cast<std::size_t>(hop)];
        for (int c = 1; c < cap; ++c) {
            ids[static_cast<std::size_t>(hop) * static_cast<std::size_t>(cap) +
                static_cast<std::size_t>(c)] = 2000 + hop * cap + c;
        }
    }
    return {ids, q};
}

int p_less_chain_token_invariants_case(int physical_rows, int token_domain, int k,
                                       bool stale_selector_q, unsigned long long seeds,
                                       const char* label) {
    const std::vector<int> survivors{7, 11, 19};
    std::vector<std::int32_t> drafts(static_cast<std::size_t>(k), 7);
    drafts[0] = 7;
    if (k > 1) { drafts[1] = 11; }
    if (k > 2) { drafts[2] = 19; }
    if (k > 3) { drafts[3] = 7; }
    if (k > 4) { drafts[4] = 11; }
    auto logits_f = peaked_p_less_chain_logits(physical_rows, token_domain, k + 1, survivors, 20.0f);
    std::vector<std::uint16_t> logits_bits(logits_f.size());
    for (std::size_t i = 0; i < logits_f.size(); ++i) { logits_bits[i] = f32_to_bf16(logits_f[i]); }

    ops::SamplingConfig config{};
    config.temperature = 2.0f;
    config.p_less      = 1;
    std::vector<std::int32_t> ids;
    std::vector<float> q;
    const std::vector<std::int32_t>* ids_arg = nullptr;
    const std::vector<float>* q_arg          = nullptr;
    if (stale_selector_q) {
        auto sel = uniform_selector_q(drafts);
        ids      = std::move(sel.first);
        q        = std::move(sel.second);
        ids_arg  = &ids;
        q_arg    = &q;
    }

    constexpr std::int32_t initial_length = 40;
    int failures                          = 0;
    int rejected_hops                     = 0;
    for (unsigned long long seed = 1; seed <= seeds; ++seed) {
        config.seed           = seed;
        const auto got        = run_chain_accept(logits_bits, physical_rows, drafts, initial_length,
                                                 token_domain, config, ids_arg, q_arg);
        failures += check_p_less_chain_invariants(label, got, logits_f, physical_rows, drafts,
                                                  token_domain, config, initial_length);
        if (got.accepted < k) { ++rejected_hops; }
    }
    if (rejected_hops == 0) {
        std::cerr << label << ": never rejected a draft; cannot check residual validity\n";
        ++failures;
    }
    return failures;
}

int p_less_out_of_domain_draft_never_licensed_case(int physical_rows, int token_domain,
                                                   unsigned long long seeds, const char* label) {
    constexpr int k = 5;
    const std::vector<int> survivors{3, 11};
    std::vector<std::int32_t> drafts(static_cast<std::size_t>(k), token_domain + 16);
    if (token_domain + 16 >= physical_rows) { drafts.assign(static_cast<std::size_t>(k), -1); }
    auto logits_f = peaked_p_less_chain_logits(physical_rows, token_domain, k + 1, survivors, 20.0f);
    std::vector<std::uint16_t> logits_bits(logits_f.size());
    for (std::size_t i = 0; i < logits_f.size(); ++i) { logits_bits[i] = f32_to_bf16(logits_f[i]); }

    ops::SamplingConfig config{};
    config.temperature = 2.0f;
    config.p_less      = 1;
    auto sel           = uniform_selector_q(drafts);
    constexpr std::int32_t initial_length = 40;
    int failures                          = 0;
    for (unsigned long long seed = 1; seed <= seeds; ++seed) {
        config.seed    = seed;
        const auto got = run_chain_accept(logits_bits, physical_rows, drafts, initial_length,
                                          token_domain, config, &sel.first, &sel.second);
        failures += check_p_less_chain_invariants(label, got, logits_f, physical_rows, drafts,
                                                  token_domain, config, initial_length);
        if (got.accepted != 0) {
            std::cerr << label << ": accepted an out-of-domain draft\n";
            ++failures;
        }
    }
    return failures;
}

int p_less_two_token_rejection_emits_other_survivor(int physical_rows, int token_domain,
                                                    unsigned long long seeds, const char* label) {
    constexpr int k     = 5;
    constexpr int draft = 7;
    constexpr int other = 11;
    std::vector<std::int32_t> drafts(static_cast<std::size_t>(k), draft);
    auto logits_f =
        peaked_p_less_chain_logits(physical_rows, token_domain, k + 1, {draft, other}, 20.0f);
    std::vector<std::uint16_t> logits_bits(logits_f.size());
    for (std::size_t i = 0; i < logits_f.size(); ++i) { logits_bits[i] = f32_to_bf16(logits_f[i]); }

    ops::SamplingConfig config{};
    config.temperature = 2.0f;
    config.p_less      = 1;
    auto sel           = uniform_selector_q(drafts);
    constexpr std::int32_t initial_length = 40;
    int failures                          = 0;
    int rejections                        = 0;
    for (unsigned long long seed = 1; seed <= seeds; ++seed) {
        config.seed    = seed;
        const auto got = run_chain_accept(logits_bits, physical_rows, drafts, initial_length,
                                          token_domain, config, &sel.first, &sel.second);
        failures += check_p_less_chain_invariants(label, got, logits_f, physical_rows, drafts,
                                                  token_domain, config, initial_length);
        if (got.accepted == 0) {
            ++rejections;
            if (got.licensed[0] != other) {
                std::cerr << label << ": hop-0 rejection emitted " << got.licensed[0]
                          << " instead of the other survivor\n";
                ++failures;
            }
        }
    }
    if (rejections == 0) {
        std::cerr << label << ": p-less never rejected the two-token hop-0 draft\n";
        ++failures;
    }
    return failures;
}

struct TreeAcceptObserved {
    std::vector<std::int32_t> licensed;
    std::vector<std::int32_t> path;
    std::int32_t count    = 0;
    std::int32_t accepted = 0;
    std::int32_t length   = 0;
    std::int32_t anchor   = 0;
    std::int32_t column   = 0;
};

TreeAcceptObserved run_tree_accept(const std::vector<std::uint16_t>& logits_bits, int physical_rows,
                                   int width, const std::vector<std::int32_t>& parent,
                                   const std::vector<std::int32_t>& verify_ids, int extent,
                                   int valid, std::int32_t initial_length, int token_domain,
                                   ops::SamplingConfig config, WorkspaceArena* workspace_ptr) {
    std::vector<std::int32_t> targets(static_cast<std::size_t>(width), 0);
    DeviceBuffer d_targets = to_device(targets);
    DeviceBuffer d_logits  = to_device(logits_bits);
    DeviceBuffer d_ids     = to_device(verify_ids);
    DeviceBuffer d_parent  = to_device(parent);
    DeviceBuffer d_valid   = to_device<std::int32_t>({valid});
    DeviceBuffer d_extent  = to_device<std::int32_t>({extent});
    std::vector<std::int32_t> token_counts(static_cast<std::size_t>(token_domain), 0);
    DeviceBuffer d_counts  = to_device(token_counts);
    config.token_counts    = static_cast<std::int32_t*>(d_counts.p);
    DeviceBuffer d_config  = device_config(config);
    DeviceBuffer d_length  = to_device<std::int32_t>({initial_length});
    DeviceBuffer d_anchors = to_device<std::int32_t>({-1});
    DeviceBuffer d_licensed =
        to_device(std::vector<std::int32_t>(static_cast<std::size_t>(width), 0));
    DeviceBuffer d_lic_count = to_device<std::int32_t>({-1});
    DeviceBuffer d_accepted  = to_device<std::int32_t>({-1});
    DeviceBuffer d_column    = to_device<std::int32_t>({-1});
    DeviceBuffer d_path = to_device(std::vector<std::int32_t>(static_cast<std::size_t>(width), 0));

    Tensor target_t(d_targets.p, DType::I32, {width, 1});
    Tensor logits_t(d_logits.p, DType::BF16, {physical_rows, width, 1});
    Tensor ids_t(d_ids.p, DType::I32, {width, 1});
    Tensor parent_t(d_parent.p, DType::I32, {width, 1});
    Tensor valid_t(d_valid.p, DType::I32, {1});
    Tensor extent_t(d_extent.p, DType::I32, {1});
    Tensor length_t(d_length.p, DType::I32, {1});
    Tensor anchors_t(d_anchors.p, DType::I32, {1});
    Tensor licensed_t(d_licensed.p, DType::I32, {width, 1});
    Tensor lic_count_t(d_lic_count.p, DType::I32, {1});
    Tensor accepted_t(d_accepted.p, DType::I32, {1});
    Tensor column_t(d_column.p, DType::I32, {1});
    Tensor path_t(d_path.p, DType::I32, {width, 1});
    const std::size_t workspace_bytes =
        ops::speculative_accept_tree_drafts_workspace_capacity_bytes(token_domain, width, width, 1,
                                                                     1);
    WorkspaceArena local_workspace(std::max<std::size_t>(256, workspace_bytes));
    WorkspaceArena& workspace = workspace_ptr != nullptr ? *workspace_ptr : local_workspace;
    ops::speculative_accept_tree_drafts(
        target_t, logits_t, ids_t, parent_t, valid_t, extent_t, length_t, anchors_t, licensed_t,
        lic_count_t, accepted_t, column_t, path_t, token_domain,
        static_cast<const ops::SamplingConfig*>(d_config.p), workspace, nullptr);
    cuda_synchronize();

    TreeAcceptObserved out;
    out.licensed = from_device<std::int32_t>(d_licensed, static_cast<std::size_t>(width));
    out.path     = from_device<std::int32_t>(d_path, static_cast<std::size_t>(width));
    out.count    = from_device<std::int32_t>(d_lic_count, 1)[0];
    out.accepted = from_device<std::int32_t>(d_accepted, 1)[0];
    out.length   = from_device<std::int32_t>(d_length, 1)[0];
    out.anchor   = from_device<std::int32_t>(d_anchors, 1)[0];
    out.column   = from_device<std::int32_t>(d_column, 1)[0];
    return out;
}

int check_p_less_tree_invariants(const char* label, const TreeAcceptObserved& got,
                                 const std::vector<float>& logits, int physical_rows, int width,
                                 const std::vector<std::int32_t>& parent,
                                 const std::vector<std::int32_t>& verify_ids, int extent, int valid,
                                 int token_domain, const ops::SamplingConfig& config,
                                 std::int32_t initial_length) {
    int failures = 0;
    auto fail    = [&](const std::string& msg) {
        std::cerr << label << ": " << msg << '\n';
        ++failures;
    };
    if (got.count < 1 || got.count > extent + 1 || got.accepted < 0 ||
        got.accepted + 1 != got.count || got.accepted > extent) {
        fail("produced count/accepted is not a valid tree prefix");
        return failures;
    }
    if (got.length != initial_length + got.count) {
        fail("length did not advance by the produced count");
    }
    if (got.anchor != got.licensed[static_cast<std::size_t>(got.accepted)]) {
        fail("anchor is not the correction/bonus token");
    }
    if (got.path[0] != 0) { fail("fold path does not start at the packed root"); }
    if (got.column != got.path[static_cast<std::size_t>(got.accepted)]) {
        fail("accepted_column is not the node that produced the correction");
    }
    for (int i = 0; i <= got.accepted; ++i) {
        const int node = got.path[static_cast<std::size_t>(i)];
        if (node < 0 || node >= valid) {
            fail("fold path node is outside valid_columns");
            continue;
        }
        if (i > 0) {
            const int prev = got.path[static_cast<std::size_t>(i - 1)];
            if (node <= prev) { fail("fold path is not strictly increasing"); }
            if (parent[static_cast<std::size_t>(node)] != prev) {
                fail("fold path is not parent-closed");
            }
        }
        const int token = got.licensed[static_cast<std::size_t>(i)];
        if (token < 0 || token >= token_domain) {
            fail("licensed token " + std::to_string(token) + " is outside token_domain");
            continue;
        }
        if (p_less_token_suppressed(config, token)) {
            fail("licensed suppressed token " + std::to_string(token));
        }
        const auto support =
            p_less_support_oracle(logits, physical_rows, node, token_domain, config);
        if (support.empty() || !p_less_support_contains(support, token)) {
            fail("licensed token " + std::to_string(token) +
                 " is outside p-less support at packed column " + std::to_string(node));
        }
        if (i < got.accepted) {
            const int child = got.path[static_cast<std::size_t>(i + 1)];
            if (verify_ids[static_cast<std::size_t>(child)] != token) {
                fail("accepted hop does not match the child verify id");
            }
        }
    }
    for (int i = got.count; i < width; ++i) {
        if (got.licensed[static_cast<std::size_t>(i)] != 0) {
            fail("unused licensed slot was not cleared");
        }
    }
    return failures;
}

int independent_p_less_sample(const std::vector<std::uint16_t>& packed_logits, int physical_rows,
                              int col, int token_domain, ops::SamplingConfig config,
                              std::int32_t position, std::int32_t purpose) {
    std::vector<std::uint16_t> col_bits(static_cast<std::size_t>(physical_rows));
    const std::size_t base = static_cast<std::size_t>(col) * static_cast<std::size_t>(physical_rows);
    for (int i = 0; i < physical_rows; ++i) {
        col_bits[static_cast<std::size_t>(i)] = packed_logits[base + static_cast<std::size_t>(i)];
    }
    DeviceBuffer d_logits = to_device(col_bits);
    DeviceBuffer d_out    = to_device<std::int32_t>({-1});
    DeviceBuffer d_pos    = to_device<std::int32_t>({position});
    std::vector<std::int32_t> counts(static_cast<std::size_t>(token_domain), 0);
    DeviceBuffer d_counts          = to_device(counts);
    config.token_counts            = static_cast<std::int32_t*>(d_counts.p);
    DeviceBuffer d_cfg             = device_config(config);
    Tensor logits_t(d_logits.p, DType::BF16, {physical_rows, 1});
    Tensor out_t(d_out.p, DType::I32, {1});
    Tensor pos_t(d_pos.p, DType::I32, {1});
    const std::size_t sample_bytes = ops::sampling_workspace_capacity_bytes(token_domain, 1, 1);
    WorkspaceArena sample_workspace(std::max<std::size_t>(256, sample_bytes));
    ops::sample(logits_t, out_t, token_domain, static_cast<const ops::SamplingConfig*>(d_cfg.p),
                pos_t, purpose, sample_workspace, nullptr);
    cuda_synchronize();
    return from_device<std::int32_t>(d_out, 1)[0];
}

int p_less_sample_matches_tree_correction_when_no_child(int physical_rows, int token_domain,
                                                         const char* label) {
    constexpr int kWidth = 4;
    const std::vector<std::int32_t> parent{-1, 0, 0, 1};
    const std::vector<std::int32_t> verify_ids{7, 50, 51, 52};
    const std::vector<int> survivors{7, 11, 19};
    auto logits_f =
        peaked_p_less_chain_logits(physical_rows, token_domain, kWidth, survivors, 20.0f);
    std::vector<std::uint16_t> logits_bits(logits_f.size());
    for (std::size_t i = 0; i < logits_f.size(); ++i) { logits_bits[i] = f32_to_bf16(logits_f[i]); }

    ops::SamplingConfig config{};
    config.temperature                    = 2.0f;
    config.p_less                         = 1;
    constexpr std::int32_t initial_length = 40;
    int failures                          = 0;
    for (unsigned long long seed = 1; seed <= 16ull; ++seed) {
        config.seed    = seed;
        const auto got = run_tree_accept(logits_bits, physical_rows, kWidth, parent, verify_ids, 3,
                                         kWidth, initial_length, token_domain, config, nullptr);
        failures += check_p_less_tree_invariants(label, got, logits_f, physical_rows, kWidth, parent,
                                                 verify_ids, 3, kWidth, token_domain, config,
                                                 initial_length);
        if (got.accepted != 0) {
            std::cerr << label << ": accepted a child outside p-less support\n";
            ++failures;
            continue;
        }

        const int independent = independent_p_less_sample(
            logits_bits, physical_rows, 0, token_domain, config, initial_length + 1,
            ops::kSamplePurposeSpeculativeAccept);
        if (got.licensed[0] != independent) {
            std::cerr << label << ": tree correction " << got.licensed[0]
                      << " != independent sample() " << independent << " seed=" << seed << '\n';
            ++failures;
        }
    }
    return failures;
}

int p_less_tree_hop0_matches_sample_with_live_children(int physical_rows, int token_domain,
                                                       unsigned long long seeds,
                                                       const char* label) {
    constexpr int kWidth = 4;
    const std::vector<std::int32_t> parent{-1, 0, 0, 1};
    const std::vector<std::int32_t> verify_ids{1, 7, 11, 19};
    const std::vector<int> survivors{7, 11, 19};
    auto logits_f =
        peaked_p_less_chain_logits(physical_rows, token_domain, kWidth, survivors, 20.0f);
    std::vector<std::uint16_t> logits_bits(logits_f.size());
    for (std::size_t i = 0; i < logits_f.size(); ++i) { logits_bits[i] = f32_to_bf16(logits_f[i]); }

    ops::SamplingConfig config{};
    config.temperature                    = 2.0f;
    config.p_less                         = 1;
    constexpr std::int32_t initial_length = 40;
    int failures                          = 0;
    int accepted_hops                     = 0;
    for (unsigned long long seed = 1; seed <= seeds; ++seed) {
        config.seed    = seed;
        const auto got = run_tree_accept(logits_bits, physical_rows, kWidth, parent, verify_ids, 3,
                                         kWidth, initial_length, token_domain, config, nullptr);
        failures += check_p_less_tree_invariants(label, got, logits_f, physical_rows, kWidth, parent,
                                                 verify_ids, 3, kWidth, token_domain, config,
                                                 initial_length);
        const int independent = independent_p_less_sample(
            logits_bits, physical_rows, 0, token_domain, config, initial_length + 1,
            ops::kSamplePurposeSpeculativeAccept);
        if (got.licensed[0] != independent) {
            std::cerr << label << ": hop 0 " << got.licensed[0] << " != sample() " << independent
                      << " seed=" << seed << '\n';
            ++failures;
        }
        if (got.accepted > 0) { ++accepted_hops; }
    }
    if (accepted_hops == 0) {
        std::cerr << label << ": never extra-accepted a hop-0 child\n";
        ++failures;
    }
    return failures;
}

int p_less_tree_support_spans_tiles_case(int physical_rows, int token_domain,
                                         unsigned long long seeds, const char* label) {
    constexpr int kWidth = 4;
    const std::vector<std::int32_t> parent{-1, 0, 0, 1};
    const int high = token_domain > 200000 ? 200003 : (token_domain > 1800 ? 1800 : token_domain - 1);
    const std::vector<int> survivors{7, 600, high};
    const std::vector<std::int32_t> verify_ids{1, 7, 600, high};
    auto logits_f =
        peaked_p_less_chain_logits(physical_rows, token_domain, kWidth, survivors, 20.0f);
    std::vector<std::uint16_t> logits_bits(logits_f.size());
    for (std::size_t i = 0; i < logits_f.size(); ++i) { logits_bits[i] = f32_to_bf16(logits_f[i]); }

    ops::SamplingConfig config{};
    config.temperature                    = 2.0f;
    config.p_less                         = 1;
    constexpr std::int32_t initial_length = 40;
    int failures                          = 0;
    int seen_tile0                        = 0;
    int seen_later                        = 0;
    for (unsigned long long seed = 1; seed <= seeds; ++seed) {
        config.seed    = seed;
        const auto got = run_tree_accept(logits_bits, physical_rows, kWidth, parent, verify_ids, 3,
                                         kWidth, initial_length, token_domain, config, nullptr);
        failures += check_p_less_tree_invariants(label, got, logits_f, physical_rows, kWidth, parent,
                                                 verify_ids, 3, kWidth, token_domain, config,
                                                 initial_length);
        const int independent = independent_p_less_sample(
            logits_bits, physical_rows, 0, token_domain, config, initial_length + 1,
            ops::kSamplePurposeSpeculativeAccept);
        if (got.licensed[0] != independent) {
            std::cerr << label << ": hop 0 " << got.licensed[0] << " != sample() " << independent
                      << " seed=" << seed << '\n';
            ++failures;
        }
        if (got.licensed[0] < 512) {
            ++seen_tile0;
        } else {
            ++seen_later;
        }
    }
    if (seen_later == 0) {
        std::cerr << label << ": every hop-0 draw stayed in tile 0 (0-511); saw " << seen_tile0
                  << " draws, never 600/" << high << '\n';
        ++failures;
    }
    return failures;
}

int p_less_tree_valid_columns_hides_later_children(int physical_rows, int token_domain,
                                                   const char* label) {
    constexpr int kWidth = 12;
    const std::vector<std::int32_t> parent{-1, 0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5};
    const std::vector<std::int32_t> verify_ids{1, 50, 51, 52, 53, 7, 11, 19, 7, 11, 19, 7};
    const std::vector<int> survivors{7, 11, 19};
    auto logits_f =
        peaked_p_less_chain_logits(physical_rows, token_domain, kWidth, survivors, 20.0f);
    std::vector<std::uint16_t> logits_bits(logits_f.size());
    for (std::size_t i = 0; i < logits_f.size(); ++i) { logits_bits[i] = f32_to_bf16(logits_f[i]); }

    ops::SamplingConfig config{};
    config.temperature                    = 2.0f;
    config.p_less                         = 1;
    constexpr std::int32_t initial_length = 40;
    constexpr int kValid                  = 4;
    int failures                          = 0;
    for (unsigned long long seed = 1; seed <= 16ull; ++seed) {
        config.seed    = seed;
        const auto got = run_tree_accept(logits_bits, physical_rows, kWidth, parent, verify_ids, 7,
                                         kValid, initial_length, token_domain, config, nullptr);
        failures += check_p_less_tree_invariants(label, got, logits_f, physical_rows, kWidth, parent,
                                                 verify_ids, 7, kValid, token_domain, config,
                                                 initial_length);
        if (got.accepted != 0) {
            std::cerr << label << ": accepted a child at column " << got.path[1]
                      << " outside valid_columns=" << kValid << '\n';
            ++failures;
        }
        for (int i = 0; i < got.count; ++i) {
            if (got.path[static_cast<std::size_t>(i)] >= kValid) {
                std::cerr << label << ": fold path visited packed column "
                          << got.path[static_cast<std::size_t>(i)] << " >= valid_columns\n";
                ++failures;
            }
        }
    }
    return failures;
}

int p_less_tree_column_local_support_case(int physical_rows, int token_domain,
                                          unsigned long long seeds, const char* label) {
    constexpr int kWidth = 12;
    const std::vector<std::int32_t> parent{-1, 0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5};
    const std::vector<std::int32_t> verify_ids{1, 600, 1200, 1800, 50, 51, 52, 53, 54, 55, 56, 57};
    std::vector<std::vector<int>> survivors(kWidth);
    survivors[0]  = {600, 1200};
    survivors[1]  = {1800};
    survivors[2]  = {7};
    survivors[3]  = {11};
    for (int col = 4; col < kWidth; ++col) { survivors[static_cast<std::size_t>(col)] = {19}; }
    auto logits_f = peaked_p_less_per_column_logits(physical_rows, token_domain, survivors);
    std::vector<std::uint16_t> logits_bits(logits_f.size());
    for (std::size_t i = 0; i < logits_f.size(); ++i) { logits_bits[i] = f32_to_bf16(logits_f[i]); }

    ops::SamplingConfig config{};
    config.temperature                    = 2.0f;
    config.p_less                         = 1;
    constexpr std::int32_t initial_length = 40;
    int failures                          = 0;
    int accepted_first                    = 0;
    for (unsigned long long seed = 1; seed <= seeds; ++seed) {
        config.seed    = seed;
        const auto got = run_tree_accept(logits_bits, physical_rows, kWidth, parent, verify_ids, 7,
                                         kWidth, initial_length, token_domain, config, nullptr);
        failures += check_p_less_tree_invariants(label, got, logits_f, physical_rows, kWidth, parent,
                                                 verify_ids, 7, kWidth, token_domain, config,
                                                 initial_length);
        const int hop0 = independent_p_less_sample(
            logits_bits, physical_rows, 0, token_domain, config, initial_length + 1,
            ops::kSamplePurposeSpeculativeAccept);
        if (got.licensed[0] != hop0) {
            std::cerr << label << ": hop 0 " << got.licensed[0] << " != sample() " << hop0
                      << " seed=" << seed << '\n';
            ++failures;
        }
        if (got.accepted >= 1) {
            ++accepted_first;
            if (got.licensed[0] == 600) {
                if (got.licensed[1] != 1800) {
                    std::cerr << label << ": after accepting child 600, hop 1 used another column's "
                              << got.licensed[1] << '\n';
                    ++failures;
                }
                const int hop1 = independent_p_less_sample(
                    logits_bits, physical_rows, 1, token_domain, config, initial_length + 2,
                    ops::kSamplePurposeSpeculativeAccept);
                if (got.licensed[1] != hop1) {
                    std::cerr << label << ": after 600, hop 1 " << got.licensed[1]
                              << " != sample(col 1) " << hop1 << " seed=" << seed << '\n';
                    ++failures;
                }
            }
        }
        if (got.licensed[0] < 512 &&
            !p_less_support_contains(p_less_support_oracle(logits_f, physical_rows, 0, token_domain,
                                                           config),
                                     got.licensed[0])) {
            std::cerr << label << ": hop 0 emitted tile-0 token " << got.licensed[0] << '\n';
            ++failures;
        }
    }
    if (accepted_first == 0) {
        std::cerr << label << ": never accepted the hop-0 child; column-local support untested\n";
        ++failures;
    }
    return failures;
}

int p_less_tree_dirty_workspace_replay_case(int physical_rows, int token_domain, const char* label) {
    constexpr int kWidth = 12;
    const std::vector<std::int32_t> parent{-1, 0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5};
    const std::vector<std::int32_t> verify_ids{1, 7, 11, 19, 7, 11, 19, 7, 11, 19, 7, 11};
    const std::vector<int> survivors{7, 11, 19};
    auto logits_f =
        peaked_p_less_chain_logits(physical_rows, token_domain, kWidth, survivors, 20.0f);
    std::vector<std::uint16_t> logits_bits(logits_f.size());
    for (std::size_t i = 0; i < logits_f.size(); ++i) { logits_bits[i] = f32_to_bf16(logits_f[i]); }

    ops::SamplingConfig config{};
    config.temperature = 2.0f;
    config.p_less      = 1;
    const std::size_t workspace_bytes =
        ops::speculative_accept_tree_drafts_workspace_capacity_bytes(token_domain, kWidth, kWidth,
                                                                     1, 1);
    WorkspaceArena workspace(std::max<std::size_t>(256, workspace_bytes));
    cuda_check(cudaMemset(workspace.base(), 0xFF, workspace.capacity()), "dirty tree workspace");

    int failures                          = 0;
    std::int32_t length                   = 40;
    for (unsigned long long seed = 1; seed <= 8ull; ++seed) {
        config.seed = seed;
        workspace.reset();
        const auto got = run_tree_accept(logits_bits, physical_rows, kWidth, parent, verify_ids, 7,
                                         kWidth, length, token_domain, config, &workspace);
        failures += check_p_less_tree_invariants(label, got, logits_f, physical_rows, kWidth, parent,
                                                 verify_ids, 7, kWidth, token_domain, config,
                                                 length);
        length = got.length;
    }
    return failures;
}

int p_less_chain_column_local_support_case(int physical_rows, int token_domain,
                                           unsigned long long seeds, const char* label) {
    constexpr int k = 5;
    std::vector<std::int32_t> drafts(static_cast<std::size_t>(k), 600);
    drafts[0] = 600;
    drafts[1] = 1800;
    std::vector<std::vector<int>> survivors(k + 1);
    survivors[0] = {600, 1200};
    survivors[1] = {1800};
    for (int col = 2; col <= k; ++col) { survivors[static_cast<std::size_t>(col)] = {7, 11}; }
    auto logits_f = peaked_p_less_per_column_logits(physical_rows, token_domain, survivors);
    std::vector<std::uint16_t> logits_bits(logits_f.size());
    for (std::size_t i = 0; i < logits_f.size(); ++i) { logits_bits[i] = f32_to_bf16(logits_f[i]); }

    ops::SamplingConfig config{};
    config.temperature                    = 2.0f;
    config.p_less                         = 1;
    constexpr std::int32_t initial_length = 40;
    int failures                          = 0;
    int accepted_first                    = 0;
    for (unsigned long long seed = 1; seed <= seeds; ++seed) {
        config.seed    = seed;
        const auto got = run_chain_accept(logits_bits, physical_rows, drafts, initial_length,
                                          token_domain, config, nullptr, nullptr);
        failures += check_p_less_chain_invariants(label, got, logits_f, physical_rows, drafts,
                                                  token_domain, config, initial_length);
        if (got.accepted >= 1) {
            ++accepted_first;
            if (got.licensed[1] != 1800) {
                std::cerr << label << ": hop 1 used column 0's support, emitted " << got.licensed[1]
                          << '\n';
                ++failures;
            }
        }
    }
    if (accepted_first == 0) {
        std::cerr << label << ": never accepted hop 0; column-local chain support untested\n";
        ++failures;
    }
    return failures;
}

int p_less_tree_later_hops_greedy_not_membership(int physical_rows, int token_domain,
                                                 const char* label) {
    constexpr int kWidth = 4;
    const std::vector<std::int32_t> parent{-1, 0, 1, 1};
    const std::vector<std::int32_t> verify_ids{1, 7, 11, 19};
    std::vector<std::vector<int>> survivors(kWidth);
    survivors[0] = {7};
    survivors[1] = {3, 11};
    survivors[2] = {19};
    survivors[3] = {19};
    auto logits_f = peaked_p_less_per_column_logits(physical_rows, token_domain, survivors);
    const std::size_t col1 = static_cast<std::size_t>(physical_rows);
    logits_f[col1 + 3]     = 21.0f;
    logits_f[col1 + 11]    = 20.0f;
    round_to_bf16(logits_f);
    std::vector<std::uint16_t> logits_bits(logits_f.size());
    for (std::size_t i = 0; i < logits_f.size(); ++i) { logits_bits[i] = f32_to_bf16(logits_f[i]); }

    ops::SamplingConfig config{};
    config.temperature                    = 2.0f;
    config.p_less                         = 1;
    constexpr std::int32_t initial_length = 40;
    int failures                          = 0;
    for (unsigned long long seed = 1; seed <= 8ull; ++seed) {
        config.seed    = seed;
        const auto got = run_tree_accept(logits_bits, physical_rows, kWidth, parent, verify_ids, 3,
                                         kWidth, initial_length, token_domain, config, nullptr);
        failures += check_p_less_tree_invariants(label, got, logits_f, physical_rows, kWidth, parent,
                                                 verify_ids, 3, kWidth, token_domain, config,
                                                 initial_length);
        if (got.licensed[0] != 7) {
            std::cerr << label << ": hop 0 " << got.licensed[0] << " != unique child 7 seed="
                      << seed << '\n';
            ++failures;
        }
        if (got.accepted != 1 || got.licensed[1] != 3) {
            std::cerr << label << ": later hop walked a non-argmax child, accepted=" << got.accepted
                      << " licensed[1]=" << got.licensed[1] << " seed=" << seed << '\n';
            ++failures;
        }
    }
    return failures;
}

int p_less_chain_later_hops_greedy_not_leviathan(int physical_rows, int token_domain,
                                                 const char* label) {
    constexpr int k = 5;
    std::vector<std::int32_t> drafts(static_cast<std::size_t>(k), 11);
    drafts[0] = 7;
    std::vector<std::vector<int>> survivors(k + 1);
    survivors[0] = {7};
    survivors[1] = {3, 11};
    for (int col = 2; col <= k; ++col) { survivors[static_cast<std::size_t>(col)] = {3, 11}; }
    auto logits_f = peaked_p_less_per_column_logits(physical_rows, token_domain, survivors);
    const std::size_t col1 = static_cast<std::size_t>(physical_rows);
    logits_f[col1 + 3]     = 21.0f;
    logits_f[col1 + 11]    = 20.0f;
    round_to_bf16(logits_f);
    std::vector<std::uint16_t> logits_bits(logits_f.size());
    for (std::size_t i = 0; i < logits_f.size(); ++i) { logits_bits[i] = f32_to_bf16(logits_f[i]); }

    ops::SamplingConfig config{};
    config.temperature                    = 2.0f;
    config.p_less                         = 1;
    constexpr std::int32_t initial_length = 40;
    int failures                          = 0;
    for (unsigned long long seed = 1; seed <= 8ull; ++seed) {
        config.seed    = seed;
        const auto got = run_chain_accept(logits_bits, physical_rows, drafts, initial_length,
                                         token_domain, config, nullptr, nullptr);
        failures += check_p_less_chain_invariants(label, got, logits_f, physical_rows, drafts,
                                                  token_domain, config, initial_length);
        if (got.licensed[0] != 7) {
            std::cerr << label << ": hop 0 " << got.licensed[0] << " != unique draft 7 seed="
                      << seed << '\n';
            ++failures;
        }
        if (got.accepted != 1 || got.licensed[1] != 3) {
            std::cerr << label << ": later hop Leviathan-accepted a non-argmax draft, accepted="
                      << got.accepted << " licensed[1]=" << got.licensed[1] << " seed=" << seed
                      << '\n';
            ++failures;
        }
    }
    return failures;
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
    failures += p_less_suppressed_bonus_case();
    failures += p_less_suppressed_draft_rejection_case(64);
    failures += p_less_suppressed_draft_rejection_case(257);
    failures += p_less_suppressed_distractor_accept_case();
    failures += p_less_tree_membership_cases(64);
    failures += p_less_tree_membership_cases(257);
    failures += p_less_dflash2_product_tree_multiblock_case();
    failures += p_less_tree_batch_row_isolation_case(
        257, 257, "p-less tree B=2 row isolation V=257");
    failures += p_less_tree_batch_row_isolation_case(
        248320, 248077, "DFlash2 p-less tree B=2 row isolation");
    failures += p_less_tree_batch_flat_support_isolation_case(
        257, 257, 48ull, "p-less tree B=2 flat-support isolation V=257");
    failures += p_less_tree_batch_flat_support_isolation_case(
        248320, 248077, 8ull, "DFlash2 p-less tree B=2 flat-support isolation");
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
    failures += p_less_fractional_q_residual_does_not_reemit_draft(
        64, 64, 11, "p-less fractional q residual does not re-emit draft V=64");
    failures += p_less_fractional_q_residual_does_not_reemit_draft(
        248320, 248077, 600, "DFlash2 p-less fractional q residual does not re-emit draft");
    failures += p_less_ignores_selector_q_when_u_exceeds_p(
        64, 64, 11, "p-less ignores 16-way q when u > p V=64");
    failures += p_less_ignores_selector_q_when_u_exceeds_p(
        248320, 248077, 600, "DFlash2 p-less ignores 16-way q when u > p");
    failures += p_less_chain_token_invariants_case(
        64, 64, 5, true, 48ull, "p-less adaptive-DFlash k=5 16-way q token invariants V=64");
    failures += p_less_chain_token_invariants_case(
        64, 64, 5, false, 32ull, "p-less adaptive-DFlash k=5 one-hot q token invariants V=64");
    failures += p_less_chain_token_invariants_case(
        248320, 248077, 5, true, 8ull,
        "DFlash2 p-less adaptive k=5 16-way q token invariants");
    failures += p_less_out_of_domain_draft_never_licensed_case(
        64, 64, 16ull, "p-less out-of-domain draft is never licensed V=64");
    failures += p_less_out_of_domain_draft_never_licensed_case(
        248320, 248077, 4ull, "DFlash2 p-less out-of-domain draft is never licensed");
    failures += p_less_two_token_rejection_emits_other_survivor(
        64, 64, 64ull, "p-less two-survivor rejection emits the other token V=64");
    failures += p_less_two_token_rejection_emits_other_survivor(
        248320, 248077, 8ull, "DFlash2 p-less two-survivor rejection emits the other token");
    failures += p_less_sample_matches_tree_correction_when_no_child(
        64, 64, "p-less tree correction matches sample() when no child V=64");
    failures += p_less_sample_matches_tree_correction_when_no_child(
        2048, 2048, "p-less tree correction matches sample() when no child V=2048");
    failures += p_less_tree_hop0_matches_sample_with_live_children(
        64, 64, 32ull, "p-less tree hop 0 matches sample() with live children V=64");
    failures += p_less_tree_hop0_matches_sample_with_live_children(
        2048, 2048, 24ull, "p-less tree hop 0 matches sample() with live children V=2048");
    failures += p_less_tree_hop0_matches_sample_with_live_children(
        248320, 248077, 8ull, "DFlash2 p-less tree hop 0 matches sample() with live children");
    failures += p_less_tree_support_spans_tiles_case(
        2048, 2048, 24ull, "p-less tree hop 0 draws from every support tile V=2048");
    failures += p_less_tree_support_spans_tiles_case(
        248320, 248077, 8ull, "DFlash2 p-less tree hop 0 draws from every support tile");
    failures += p_less_tree_valid_columns_hides_later_children(
        64, 64, "p-less tree valid_columns hides later packed children V=64");
    failures += p_less_tree_valid_columns_hides_later_children(
        2048, 2048, "p-less tree valid_columns hides later packed children V=2048");
    failures += p_less_tree_column_local_support_case(
        2048, 2048, 32ull, "p-less tree W=12 samples each packed column's support V=2048");
    failures += p_less_tree_column_local_support_case(
        248320, 248077, 4ull, "DFlash2 p-less tree W=12 samples each packed column's support");
    failures += p_less_chain_column_local_support_case(
        2048, 2048, 32ull, "p-less chain samples each verify column's support V=2048");
    failures += p_less_chain_column_local_support_case(
        248320, 248077, 4ull, "DFlash2 p-less chain samples each verify column's support");
    failures += p_less_tree_later_hops_greedy_not_membership(
        64, 64, "p-less tree later hops greedy V=64");
    failures += p_less_tree_later_hops_greedy_not_membership(
        2048, 2048, "p-less tree later hops greedy V=2048");
    failures += p_less_chain_later_hops_greedy_not_leviathan(
        64, 64, "p-less chain later hops greedy V=64");
    failures += p_less_chain_later_hops_greedy_not_leviathan(
        2048, 2048, "p-less chain later hops greedy V=2048");
    failures += p_less_tree_dirty_workspace_replay_case(
        2048, 2048, "p-less tree W=12 dirty-workspace replay V=2048");
    failures += p_less_tree_dirty_workspace_replay_case(
        248320, 248077, "DFlash2 p-less tree W=12 dirty-workspace replay");
    failures += batched_selector_row_isolation_case();

    if (failures != 0) {
        std::cerr << "speculative_round failures=" << failures << '\n';
        return 1;
    }
    std::cout << "speculative_round: PASS\n";
    return 0;
}
