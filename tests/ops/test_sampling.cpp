// Public-contract qualification for sample().
//
// The deterministic branch is checked exactly against an independent CPU
// argmax.  The stochastic branch is checked against one FP64 mathematical
// distribution oracle built from the BF16 values represented at the public
// input.  The test never reproduces the device RNG algorithm or uses another
// production path as a golden.
#include "ninfer/ops/sampling.h"
#include "ops/op_tester.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

using namespace ninfer;
using namespace ninfer::test;

namespace {

struct Candidate {
    double adjusted = 0.0;
    int token       = 0;
};

struct Distribution {
    std::vector<int> tokens;
    std::vector<double> probabilities;
};

struct RunResult {
    std::vector<int> tokens;
    std::vector<std::vector<int>> counts;
    int integrity_failures = 0;
};

bool same_config(const ops::SamplingConfig& a, const ops::SamplingConfig& b) {
    return a.temperature == b.temperature && a.top_k == b.top_k && a.top_p == b.top_p &&
           a.min_p == b.min_p && a.presence_penalty == b.presence_penalty &&
           a.frequency_penalty == b.frequency_penalty && a.p_less == b.p_less && a.seed == b.seed &&
           a.token_counts == b.token_counts &&
           a.suppressed_token_count == b.suppressed_token_count &&
           std::equal(std::begin(a.suppressed_tokens), std::end(a.suppressed_tokens),
                      std::begin(b.suppressed_tokens)) &&
           a.typical_exclude == b.typical_exclude &&
           a.allowed_token_words == b.allowed_token_words &&
           a.allowed_token_column_stride == b.allowed_token_column_stride;
}

std::vector<std::uint16_t> bf16_bits(const std::vector<float>& values) {
    std::vector<std::uint16_t> bits(values.size());
    for (std::size_t i = 0; i < values.size(); ++i) { bits[i] = f32_to_bf16(values[i]); }
    return bits;
}

std::vector<float> repeat_column(const std::vector<float>& column, int columns) {
    std::vector<float> logits(column.size() * static_cast<std::size_t>(columns));
    for (int t = 0; t < columns; ++t) {
        std::copy(column.begin(), column.end(),
                  logits.begin() + static_cast<std::ptrdiff_t>(t) * column.size());
    }
    return logits;
}

std::vector<int> greedy_oracle(const std::vector<float>& logits, int physical_rows,
                               int token_domain, int columns) {
    std::vector<int> expected(static_cast<std::size_t>(columns));
    for (int t = 0; t < columns; ++t) {
        const std::size_t base = static_cast<std::size_t>(t) * physical_rows;
        int best               = 0;
        for (int token = 1; token < token_domain; ++token) {
            if (logits[base + token] > logits[base + best]) { best = token; }
        }
        expected[static_cast<std::size_t>(t)] = best;
    }
    return expected;
}

Distribution distribution_oracle(const std::vector<float>& column, int token_domain,
                                 const ops::SamplingConfig& config,
                                 const std::vector<int>* counts = nullptr) {
    if (config.p_less != 0) {
        auto suppressed = [&](int token) {
            for (int i = 0; i < config.suppressed_token_count; ++i) {
                if (config.suppressed_tokens[i] == token) { return true; }
            }
            return false;
        };
        std::vector<double> logits(static_cast<std::size_t>(token_domain));
        double max_scaled = 0.0;
        bool have_max     = false;
        for (int token = 0; token < token_domain; ++token) {
            if (suppressed(token)) { continue; }
            logits[static_cast<std::size_t>(token)] =
                static_cast<double>(column[static_cast<std::size_t>(token)]);
            const double scaled = logits[static_cast<std::size_t>(token)] / config.temperature;
            if (!have_max || scaled > max_scaled) {
                max_scaled = scaled;
                have_max   = true;
            }
        }
        std::vector<double> weights(static_cast<std::size_t>(token_domain));
        double total = 0.0;
        for (int token = 0; token < token_domain; ++token) {
            if (suppressed(token)) { continue; }
            const double w =
                std::exp(logits[static_cast<std::size_t>(token)] / config.temperature - max_scaled);
            weights[static_cast<std::size_t>(token)] = w;
            total += w;
        }
        double collision = 0.0;
        for (int token = 0; token < token_domain; ++token) {
            if (suppressed(token)) { continue; }
            const double p = weights[static_cast<std::size_t>(token)] / total;
            collision += p * p;
        }
        const double cut = ops::p_less_membership_cut(collision, config.temperature);
        Distribution out;
        double kept = 0.0;
        for (int token = 0; token < token_domain; ++token) {
            if (suppressed(token)) { continue; }
            const double p = weights[static_cast<std::size_t>(token)] / total;
            if (p >= cut) {
                out.tokens.push_back(token);
                out.probabilities.push_back(p);
                kept += p;
            }
        }
        if (out.tokens.empty()) {
            int argmax = -1;
            for (int token = 0; token < token_domain; ++token) {
                if (suppressed(token)) { continue; }
                if (argmax < 0 || logits[static_cast<std::size_t>(token)] >
                                      logits[static_cast<std::size_t>(argmax)]) {
                    argmax = token;
                }
            }
            out.tokens        = {argmax};
            out.probabilities = {1.0};
        } else {
            for (double& probability : out.probabilities) { probability /= kept; }
        }
        if (config.typical_exclude >= 0) {
            const auto found = std::find(out.tokens.begin(), out.tokens.end(),
                                         config.typical_exclude);
            if (found != out.tokens.end()) {
                if (out.tokens.size() == 1) {
                    const int mode = out.tokens[0];
                    int runner     = -1;
                    float best     = 0.0f;
                    bool have      = false;
                    for (int token = 0; token < token_domain; ++token) {
                        if (suppressed(token) || token == mode) { continue; }
                        const float x = column[static_cast<std::size_t>(token)];
                        if (!have || x > best || (x == best && token < runner)) {
                            best   = x;
                            runner = token;
                            have   = true;
                        }
                    }
                    out.tokens        = {runner};
                    out.probabilities = {1.0};
                } else {
                    const std::size_t at =
                        static_cast<std::size_t>(found - out.tokens.begin());
                    out.tokens.erase(out.tokens.begin() + static_cast<std::ptrdiff_t>(at));
                    out.probabilities.erase(out.probabilities.begin() +
                                            static_cast<std::ptrdiff_t>(at));
                    double rest = 0.0;
                    for (double p : out.probabilities) { rest += p; }
                    for (double& p : out.probabilities) { p /= rest; }
                }
            }
        }
        return out;
    }
    std::vector<Candidate> candidates;
    candidates.reserve(static_cast<std::size_t>(token_domain));
    for (int token = 0; token < token_domain; ++token) {
        bool suppressed = false;
        for (int i = 0; i < config.suppressed_token_count; ++i) {
            suppressed |= config.suppressed_tokens[i] == token;
        }
        if (suppressed) { continue; }
        const int count = counts == nullptr ? 0 : (*counts)[static_cast<std::size_t>(token)];
        double adjusted = static_cast<double>(column[static_cast<std::size_t>(token)]);
        if (count > 0) { adjusted -= static_cast<double>(config.presence_penalty); }
        adjusted -= static_cast<double>(config.frequency_penalty) * static_cast<double>(count);
        candidates.push_back({adjusted, token});
    }
    std::sort(candidates.begin(), candidates.end(), [](const Candidate& a, const Candidate& b) {
        if (a.adjusted != b.adjusted) { return a.adjusted > b.adjusted; }
        return a.token < b.token;
    });

    int cap = 20;
    if (config.top_k > 0 && config.top_k < 20) { cap = config.top_k; }
    cap = std::min(cap, token_domain);
    candidates.resize(static_cast<std::size_t>(cap));

    std::vector<double> weights(static_cast<std::size_t>(cap));
    const double max_scaled = candidates.front().adjusted / config.temperature;
    double total_weight     = 0.0;
    for (int rank = 0; rank < cap; ++rank) {
        const double weight = std::exp(
            candidates[static_cast<std::size_t>(rank)].adjusted / config.temperature - max_scaled);
        weights[static_cast<std::size_t>(rank)] = weight;
        total_weight += weight;
    }

    const bool use_min_p     = config.min_p > 0.0f;
    const bool use_top_p     = config.top_p < 1.0f;
    const double min_weight  = static_cast<double>(config.min_p) * weights.front();
    const double top_p_limit = static_cast<double>(config.top_p) * total_weight;
    double cumulative        = 0.0;
    int support              = 0;
    for (int rank = 0; rank < cap; ++rank) {
        if (use_min_p && weights[static_cast<std::size_t>(rank)] < min_weight) { break; }
        cumulative += weights[static_cast<std::size_t>(rank)];
        support = rank + 1;
        if (use_top_p && cumulative >= top_p_limit) { break; }
    }
    support = std::max(support, 1);

    Distribution out;
    out.tokens.reserve(static_cast<std::size_t>(support));
    out.probabilities.reserve(static_cast<std::size_t>(support));
    double kept_weight = 0.0;
    for (int rank = 0; rank < support; ++rank) {
        kept_weight += weights[static_cast<std::size_t>(rank)];
    }
    for (int rank = 0; rank < support; ++rank) {
        out.tokens.push_back(candidates[static_cast<std::size_t>(rank)].token);
        out.probabilities.push_back(weights[static_cast<std::size_t>(rank)] / kept_weight);
    }
    return out;
}

RunResult run_batch(const std::vector<float>& logits, int physical_rows, int token_domain,
                    std::vector<ops::SamplingConfig> configs,
                    const std::vector<int>& logical_positions, int purpose,
                    const std::vector<std::vector<int>>& initial_counts = {}) {
    const int batch = static_cast<int>(configs.size());
    if (batch <= 0 || logical_positions.size() != configs.size() ||
        logits.size() != static_cast<std::size_t>(physical_rows) * configs.size() ||
        (!initial_counts.empty() && initial_counts.size() != configs.size())) {
        throw std::invalid_argument("invalid sample batch fixture");
    }
    const std::vector<std::uint16_t> input_bits = bf16_bits(logits);
    DeviceBuffer device_logits                  = to_device(input_bits);
    GuardedDeviceBuffer device_out(static_cast<std::size_t>(batch) * sizeof(std::int32_t));
    const std::vector<int> output_sentinel(static_cast<std::size_t>(batch), -777777);
    device_out.copy_from_host(output_sentinel.data(),
                              output_sentinel.size() * sizeof(std::int32_t));

    std::vector<std::unique_ptr<GuardedDeviceBuffer>> device_counts;
    if (!initial_counts.empty()) {
        device_counts.reserve(configs.size());
        for (std::size_t row = 0; row < configs.size(); ++row) {
            if (initial_counts[row].size() != static_cast<std::size_t>(token_domain)) {
                throw std::invalid_argument("invalid sample token-count fixture");
            }
            auto counts = std::make_unique<GuardedDeviceBuffer>(initial_counts[row].size() *
                                                                sizeof(std::int32_t));
            counts->copy_from_host(initial_counts[row].data(), counts->bytes());
            configs[row].token_counts = static_cast<std::int32_t*>(counts->data());
            device_counts.push_back(std::move(counts));
        }
    }
    const std::vector<ops::SamplingConfig> expected_configs = configs;
    DeviceBuffer device_configs                             = to_device(configs);
    DeviceBuffer device_positions                           = to_device(logical_positions);

    Tensor logits_tensor(device_logits.p, DType::BF16, {physical_rows, batch});
    Tensor out_tensor(device_out.data(), DType::I32, {batch});
    Tensor positions_tensor(device_positions.p, DType::I32, {batch});
    const std::size_t workspace_bytes =
        ops::sampling_workspace_capacity_bytes(token_domain, batch, batch);
    WorkspaceArena workspace(std::max<std::size_t>(256, workspace_bytes));
    ops::sample(logits_tensor, out_tensor, token_domain,
                static_cast<const ops::SamplingConfig*>(device_configs.p), positions_tensor,
                purpose, workspace, nullptr);
    cuda_synchronize();

    RunResult result;
    result.tokens = from_device<int>(device_out.data(), static_cast<std::size_t>(batch));
    result.integrity_failures += device_out.verify_guards("sample output");
    result.integrity_failures +=
        verify_exact("sample read-only logits",
                     from_device<std::uint16_t>(device_logits, input_bits.size()), input_bits);

    const std::vector<ops::SamplingConfig> actual_configs =
        from_device<ops::SamplingConfig>(device_configs, configs.size());
    for (std::size_t row = 0; row < configs.size(); ++row) {
        if (!same_config(actual_configs[row], expected_configs[row])) {
            std::cerr << "sample modified SamplingConfig row " << row << '\n';
            ++result.integrity_failures;
        }
    }
    result.integrity_failures += verify_exact(
        "sample read-only logical positions",
        from_device<std::int32_t>(device_positions, configs.size()), logical_positions);

    result.counts.reserve(device_counts.size());
    for (std::size_t row = 0; row < device_counts.size(); ++row) {
        result.counts.push_back(
            from_device<int>(device_counts[row]->data(), static_cast<std::size_t>(token_domain)));
        result.integrity_failures += device_counts[row]->verify_guards("sample token_counts");
    }
    if (workspace.used() != 0 || workspace.peak_used() != workspace_bytes) {
        std::cerr << "sample workspace query/execution high-water mismatch\n";
        ++result.integrity_failures;
    }
    return result;
}

RunResult run_homogeneous_batch(const std::vector<float>& logits, int physical_rows,
                                int token_domain, int batch, ops::SamplingConfig config,
                                int first_position, int purpose,
                                const std::vector<int>* initial_counts = nullptr) {
    std::vector<ops::SamplingConfig> configs(static_cast<std::size_t>(batch), config);
    std::vector<int> positions(static_cast<std::size_t>(batch));
    for (int row = 0; row < batch; ++row) {
        positions[static_cast<std::size_t>(row)] = first_position + row;
    }
    std::vector<std::vector<int>> row_counts;
    if (initial_counts != nullptr) {
        row_counts.assign(static_cast<std::size_t>(batch), *initial_counts);
    }
    return run_batch(logits, physical_rows, token_domain, std::move(configs), positions, purpose,
                     row_counts);
}

RunResult run_repeated(const std::vector<float>& column, int token_domain, int total, int batch,
                       ops::SamplingConfig config, int position, int purpose) {
    if (batch <= 0 || total <= 0 || total % batch != 0) {
        throw std::invalid_argument("repeated sample count must be a positive batch multiple");
    }
    const int physical_rows                     = static_cast<int>(column.size());
    const std::vector<float> logits             = repeat_column(column, batch);
    const std::vector<std::uint16_t> input_bits = bf16_bits(logits);
    DeviceBuffer device_logits                  = to_device(input_bits);
    GuardedDeviceBuffer collected(static_cast<std::size_t>(total) * sizeof(std::int32_t));
    std::vector<ops::SamplingConfig> configs(static_cast<std::size_t>(batch), config);
    const std::vector<ops::SamplingConfig> expected_configs = configs;
    DeviceBuffer device_configs                             = to_device(configs);
    std::vector<int> positions(static_cast<std::size_t>(total));
    for (int i = 0; i < total; ++i) { positions[static_cast<std::size_t>(i)] = position + i; }
    DeviceBuffer device_positions = to_device(positions);

    Tensor logits_tensor(device_logits.p, DType::BF16, {physical_rows, batch});
    const std::size_t workspace_bytes =
        ops::sampling_workspace_capacity_bytes(token_domain, batch, batch);
    WorkspaceArena workspace(std::max<std::size_t>(256, workspace_bytes));

    for (int produced = 0; produced < total; produced += batch) {
        auto* out = static_cast<std::int32_t*>(collected.data()) + produced;
        auto* pos = static_cast<std::int32_t*>(device_positions.p) + produced;
        Tensor out_tensor(out, DType::I32, {batch});
        Tensor positions_tensor(pos, DType::I32, {batch});
        ops::sample(logits_tensor, out_tensor, token_domain,
                    static_cast<const ops::SamplingConfig*>(device_configs.p), positions_tensor,
                    purpose, workspace, nullptr);
    }
    cuda_synchronize();

    RunResult result;
    result.tokens = from_device<int>(collected.data(), static_cast<std::size_t>(total));
    result.integrity_failures += collected.verify_guards("sample repeated output");
    result.integrity_failures +=
        verify_exact("sample repeated read-only logits",
                     from_device<std::uint16_t>(device_logits, input_bits.size()), input_bits);
    const std::vector<ops::SamplingConfig> actual_configs =
        from_device<ops::SamplingConfig>(device_configs, configs.size());
    for (std::size_t row = 0; row < configs.size(); ++row) {
        if (!same_config(actual_configs[row], expected_configs[row])) {
            std::cerr << "sample repeated modified SamplingConfig row " << row << '\n';
            ++result.integrity_failures;
        }
    }
    result.integrity_failures +=
        verify_exact("sample repeated read-only logical positions",
                     from_device<std::int32_t>(device_positions, positions.size()), positions);
    if (workspace.used() != 0 || workspace.peak_used() != workspace_bytes) {
        std::cerr << "sample repeated workspace query/execution high-water mismatch\n";
        ++result.integrity_failures;
    }
    return result;
}

int verify_distribution(const char* label, const std::vector<int>& samples,
                        const Distribution& expected) {
    std::vector<int> observed(expected.tokens.size(), 0);
    for (int token : samples) {
        const auto it = std::find(expected.tokens.begin(), expected.tokens.end(), token);
        if (it == expected.tokens.end()) {
            std::cerr << label << ": sampled token " << token << " outside oracle support\n";
            return 1;
        }
        ++observed[static_cast<std::size_t>(it - expected.tokens.begin())];
    }

    const double n              = static_cast<double>(samples.size());
    double max_standardized_gap = 0.0;
    for (std::size_t i = 0; i < expected.tokens.size(); ++i) {
        const double probability = expected.probabilities[i];
        const double frequency   = static_cast<double>(observed[i]) / n;
        const double sigma       = std::sqrt(probability * (1.0 - probability) / n);
        const double limit       = 7.0 * sigma + 2.0 / n;
        const double gap         = std::abs(frequency - probability);
        if (gap > limit) {
            std::cerr << label << ": token=" << expected.tokens[i] << " frequency=" << frequency
                      << " oracle=" << probability << " gap=" << gap << " limit=" << limit << '\n';
            return 1;
        }
        if (sigma > 0.0) { max_standardized_gap = std::max(max_standardized_gap, gap / sigma); }
    }
    std::cout << "    " << label << " FP64 distribution match (max z=" << max_standardized_gap
              << ")\n";
    return 0;
}

int greedy_contract() {
    constexpr int physical_rows = 248320;
    constexpr int token_domain  = 248077;
    constexpr int batch         = 8;
    std::vector<float> logits(static_cast<std::size_t>(physical_rows) * batch, -9.0f);
    for (int row = 0; row < batch; ++row) {
        const std::size_t base = static_cast<std::size_t>(row) * physical_rows;
        int first              = (17 + 7919 * row) % token_domain;
        int second             = token_domain - 1 - ((31 + 65537 * row) % token_domain);
        if (second == first) { second = (first + 1) % token_domain; }
        if (second < first) { std::swap(first, second); }
        logits[base + first]             = row == 0 ? -0.0f : 16.0f + row;
        logits[base + second]            = row == 0 ? 0.0f : 16.0f + row;
        logits[base + token_domain]      = 100.0f;
        logits[base + physical_rows - 1] = 200.0f;
    }
    round_to_bf16(logits);

    std::vector<int> counts(static_cast<std::size_t>(token_domain), 0);
    counts[17] = 9;
    ops::SamplingConfig config;
    config.temperature       = 0.0f;
    config.top_k             = 1;
    config.top_p             = 0.01f;
    config.min_p             = 0.99f;
    config.presence_penalty  = 100.0f;
    config.frequency_penalty = 100.0f;
    config.seed              = 12345;

    const RunResult result = run_homogeneous_batch(logits, physical_rows, token_domain, batch,
                                                   config, 77, ops::kSamplePurposeDecode, &counts);
    int failures           = result.integrity_failures;
    failures += verify_exact("sample greedy mathematical result", result.tokens,
                             greedy_oracle(logits, physical_rows, token_domain, batch));
    for (const std::vector<int>& row_counts : result.counts) {
        failures += verify_exact("sample greedy skips token_counts updates", row_counts, counts);
    }
    return failures;
}

int deterministic_stochastic_contract() {
    std::vector<float> column = {5.0f, 4.5f, 4.0f, 3.0f, -1.0f};
    round_to_bf16(column);
    int failures = 0;

    struct Case {
        const char* label;
        float presence;
        float frequency;
        std::vector<int> counts;
    };

    const Case cases[] = {
        {"sample positive-temperature presence penalty", 1.0f, 0.0f, {1, 0, 0, 0, 0}},
        {"sample positive-temperature frequency penalty", 0.0f, 0.5f, {2, 0, 0, 0, 0}},
        {"sample adjusted-logit tie break", 0.5f, 0.0f, {1, 0, 0, 0, 0}},
    };
    for (const Case& test_case : cases) {
        ops::SamplingConfig config;
        config.temperature       = 0.8f;
        config.top_k             = 1;
        config.presence_penalty  = test_case.presence;
        config.frequency_penalty = test_case.frequency;
        config.seed              = 9981;
        const Distribution oracle =
            distribution_oracle(column, static_cast<int>(column.size()), config, &test_case.counts);
        RunResult result = run_homogeneous_batch(column, static_cast<int>(column.size()),
                                                 static_cast<int>(column.size()), 1, config, 11,
                                                 ops::kSamplePurposeDecode, &test_case.counts);

        std::vector<int> expected_counts = test_case.counts;
        ++expected_counts[static_cast<std::size_t>(oracle.tokens.front())];
        failures += result.integrity_failures;
        failures += verify_exact(test_case.label, result.tokens, {oracle.tokens.front()});
        failures += verify_exact("sample increments only selected token", result.counts.front(),
                                 expected_counts);
    }
    return failures;
}

int heterogeneous_batch_contract() {
    constexpr int physical_rows = 260;
    constexpr int token_domain  = 257;
    constexpr int batch         = 4;
    std::vector<float> logits(static_cast<std::size_t>(physical_rows) * batch, -8.0f);
    const auto set = [&](int row, int token, float value) {
        logits[static_cast<std::size_t>(row) * physical_rows + token] = value;
    };
    set(0, 5, 4.0f);
    set(0, 7, 3.0f);
    set(1, 11, 4.0f);
    set(1, 12, 3.0f);
    set(2, 13, 5.0f);
    set(2, 17, 4.5f);
    set(3, 19, 2.0f);
    set(3, 23, 2.0f);
    for (int row = 0; row < batch; ++row) { set(row, physical_rows - 1, 100.0f); }
    round_to_bf16(logits);

    std::vector<ops::SamplingConfig> configs(batch);
    configs[0].temperature      = 0.0f;
    configs[1].temperature      = 0.7f;
    configs[1].top_k            = 1;
    configs[1].seed             = 101;
    configs[2].temperature      = 0.7f;
    configs[2].top_k            = 1;
    configs[2].presence_penalty = 1.0f;
    configs[2].seed             = 202;
    configs[3].temperature      = 0.0f;

    std::vector<std::vector<int>> counts(batch, std::vector<int>(token_domain, 0));
    counts[0][5]           = 9;
    counts[2][13]          = 1;
    const RunResult result = run_batch(logits, physical_rows, token_domain, std::move(configs),
                                       {7, 103, 999, 41}, ops::kSamplePurposeDecode, counts);

    int failures = result.integrity_failures;
    failures += verify_exact("sample heterogeneous batch tokens", result.tokens, {5, 11, 17, 19});
    std::vector<std::vector<int>> expected = counts;
    ++expected[1][11];
    ++expected[2][17];
    for (int row = 0; row < batch; ++row) {
        failures += verify_exact("sample heterogeneous batch isolated token counts",
                                 result.counts[static_cast<std::size_t>(row)],
                                 expected[static_cast<std::size_t>(row)]);
    }
    return failures;
}

int suppressed_token_contract() {
    constexpr int token_domain = 257;
    constexpr int batch        = 3;
    std::vector<float> logits(static_cast<std::size_t>(token_domain) * batch, -20.0f);
    for (int row = 0; row < batch; ++row) {
        const std::size_t base = static_cast<std::size_t>(row) * token_domain;
        logits[base + 11] = 8.0f;
        logits[base + 17] = 7.0f;
    }
    for (int token = 0; token < token_domain; ++token) {
        logits[static_cast<std::size_t>(2 * token_domain + token)] = -INFINITY;
    }
    round_to_bf16(logits);

    std::vector<ops::SamplingConfig> configs(batch);
    configs[0].temperature             = 0.0f;
    configs[0].suppressed_token_count  = 1;
    configs[0].suppressed_tokens[0]    = 11;
    configs[1].temperature             = 0.8f;
    configs[1].top_k                   = 1;
    configs[1].seed                    = 1234;
    configs[1].suppressed_token_count  = 1;
    configs[1].suppressed_tokens[0]    = 11;
    configs[2].temperature             = 0.0f;
    configs[2].suppressed_token_count  = 1;
    configs[2].suppressed_tokens[0]    = 0;

    const RunResult result = run_batch(logits, token_domain, token_domain, std::move(configs),
                                       {1, 2, 3}, ops::kSamplePurposeDecode);
    return result.integrity_failures +
           verify_exact("sample suppresses greedy and stochastic top token", result.tokens,
                        {17, 17, 1});
}

int filtered_distribution_contract() {
    std::vector<float> column = {3.0f, 2.7f,  2.7f,  2.1f,  1.5f,  0.7f,
                                 0.1f, -0.4f, -1.0f, -2.0f, -3.0f, -4.0f};
    round_to_bf16(column);
    ops::SamplingConfig config;
    config.temperature = 0.75f;
    config.top_k       = 6;
    config.top_p       = 0.86f;
    config.min_p       = 0.12f;
    config.seed        = 20260726;

    const Distribution oracle =
        distribution_oracle(column, static_cast<int>(column.size()), config);
    if (oracle.tokens.size() < 2 || oracle.tokens.size() >= 6) {
        std::cerr << "filtered distribution fixture did not exercise both filters\n";
        return 1;
    }

    constexpr int samples  = 16384;
    const RunResult result = run_repeated(column, static_cast<int>(column.size()), samples, 8,
                                          config, 400, ops::kSamplePurposeDecode);
    return result.integrity_failures +
           verify_distribution("sample top-k/top-p/min-p", result.tokens, oracle);
}

int capped_distribution_contract() {
    std::vector<float> column(24, 0.0f);
    for (int token = 0; token < 24; ++token) {
        column[static_cast<std::size_t>(token)] = 2.0f - 0.1f * token;
    }
    round_to_bf16(column);
    ops::SamplingConfig config;
    config.temperature = 1.1f;
    config.top_k       = 64;
    config.seed        = 884422;

    const Distribution oracle =
        distribution_oracle(column, static_cast<int>(column.size()), config);
    if (oracle.tokens.size() != 20) {
        std::cerr << "top-k cap oracle fixture has unexpected support\n";
        return 1;
    }

    constexpr int samples  = 16384;
    const RunResult result = run_repeated(column, static_cast<int>(column.size()), samples, 8,
                                          config, 900, ops::kSamplePurposePrefill);
    return result.integrity_failures +
           verify_distribution("sample top-k public cap", result.tokens, oracle);
}

int real_shape_distribution_contract() {
    constexpr int physical_rows = 248320;
    constexpr int token_domain  = 248077;
    std::vector<float> column(physical_rows, -20.0f);
    const int ids[]      = {17, 7919, 65537, 200003};
    const float logits[] = {3.0f, 2.0f, 1.0f, 0.0f};
    for (int i = 0; i < 4; ++i) { column[ids[i]] = logits[i]; }
    column[token_domain]      = 100.0f;
    column[physical_rows - 1] = 200.0f;
    round_to_bf16(column);

    ops::SamplingConfig config;
    config.temperature        = 1.0f;
    config.top_k              = 4;
    config.seed               = 7654321;
    const Distribution oracle = distribution_oracle(column, token_domain, config);

    constexpr int samples = 4096;
    RunResult result =
        run_repeated(column, token_domain, samples, 8, config, 2000, ops::kSamplePurposeDecode);
    return result.integrity_failures +
           verify_distribution("sample real token-domain B=8", result.tokens, oracle);
}

int rng_key_contract() {
    std::vector<float> column = {0.0f, 0.0f};
    round_to_bf16(column);
    ops::SamplingConfig config;
    config.temperature = 1.0f;
    config.top_k       = 2;
    config.seed        = 424242;

    constexpr int samples = 128;
    const RunResult baseline =
        run_repeated(column, 2, samples, 8, config, 100, ops::kSamplePurposeDecode);
    const RunResult repeat =
        run_repeated(column, 2, samples, 8, config, 100, ops::kSamplePurposeDecode);
    const RunResult rechunked =
        run_repeated(column, 2, samples, 1, config, 100, ops::kSamplePurposeDecode);
    const RunResult shifted =
        run_repeated(column, 2, samples, 8, config, 101, ops::kSamplePurposeDecode);
    const RunResult other_purpose =
        run_repeated(column, 2, samples, 8, config, 100, ops::kSamplePurposePrefill);
    ops::SamplingConfig other_seed_config = config;
    ++other_seed_config.seed;
    const RunResult other_seed =
        run_repeated(column, 2, samples, 8, other_seed_config, 100, ops::kSamplePurposeDecode);

    int failures = baseline.integrity_failures + repeat.integrity_failures +
                   rechunked.integrity_failures + shifted.integrity_failures +
                   other_purpose.integrity_failures + other_seed.integrity_failures;
    failures += verify_exact("sample identical counter key is reproducible", repeat.tokens,
                             baseline.tokens);
    failures += verify_exact("sample RNG does not depend on compact row", rechunked.tokens,
                             baseline.tokens);
    failures += verify_exact("sample position selects the corresponding counter",
                             std::vector<int>(shifted.tokens.begin(), shifted.tokens.end() - 1),
                             std::vector<int>(baseline.tokens.begin() + 1, baseline.tokens.end()));
    if (other_purpose.tokens == baseline.tokens) {
        std::cerr << "sample purpose did not separate the counter stream\n";
        ++failures;
    }
    if (other_seed.tokens == baseline.tokens) {
        std::cerr << "sample seed did not separate the counter stream\n";
        ++failures;
    }
    return failures;
}

int p_less_suppressed_token_contract() {
    constexpr int token_domain = 257;
    constexpr int batch        = 2;
    std::vector<float> logits(static_cast<std::size_t>(token_domain) * batch, -20.0f);
    for (int row = 0; row < batch; ++row) {
        const std::size_t base = static_cast<std::size_t>(row) * token_domain;
        logits[base + 11]      = 8.0f;
        logits[base + 17]      = 7.0f;
    }
    round_to_bf16(logits);

    std::vector<ops::SamplingConfig> configs(batch);
    configs[0].temperature            = 0.8f;
    configs[0].p_less                 = 1;
    configs[0].seed                   = 1234;
    configs[0].suppressed_token_count = 1;
    configs[0].suppressed_tokens[0]   = 11;
    configs[1].temperature            = 0.8f;
    configs[1].p_less                 = 1;
    configs[1].seed                   = 5678;
    configs[1].suppressed_token_count = 1;
    configs[1].suppressed_tokens[0]   = 11;

    const RunResult result = run_batch(logits, token_domain, token_domain, std::move(configs),
                                       {1, 2}, ops::kSamplePurposeDecode);
    return result.integrity_failures +
           verify_exact("sample p-less suppresses peaked token", result.tokens, {17, 17});
}

int p_less_suppressed_distribution_contract() {
    std::vector<float> column = {1.0f, 0.9f, 0.8f, 0.2f, -1.0f, -2.0f};
    round_to_bf16(column);
    ops::SamplingConfig config;
    config.temperature            = 0.9f;
    config.p_less                 = 1;
    config.seed                   = 314159;
    config.suppressed_token_count = 1;
    config.suppressed_tokens[0]   = 0;

    const Distribution oracle =
        distribution_oracle(column, static_cast<int>(column.size()), config);
    if (oracle.tokens.empty() ||
        std::find(oracle.tokens.begin(), oracle.tokens.end(), 0) != oracle.tokens.end()) {
        std::cerr << "p-less suppression oracle kept the excluded token or emptied support\n";
        return 1;
    }

    constexpr int samples  = 16384;
    const RunResult result = run_repeated(column, static_cast<int>(column.size()), samples, 8,
                                          config, 50, ops::kSamplePurposeDecode);
    return result.integrity_failures +
           verify_distribution("sample p-less suppressed small-vocab", result.tokens, oracle);
}

int p_less_suppressed_real_shape_contract() {
    constexpr int physical_rows = 248320;
    constexpr int token_domain  = 248077;
    std::vector<float> column(physical_rows, -20.0f);
    const int ids[]      = {3, 17, 7919, 65537, 200003};
    const float logits[] = {2.0f, 1.85f, 1.7f, 0.5f, 0.0f};
    for (int i = 0; i < 5; ++i) { column[ids[i]] = logits[i]; }
    column[token_domain]      = 100.0f;
    column[physical_rows - 1] = 200.0f;
    round_to_bf16(column);

    ops::SamplingConfig config;
    config.temperature            = 1.2f;
    config.p_less                 = 1;
    config.seed                   = 271828;
    config.suppressed_token_count = 1;
    config.suppressed_tokens[0]   = 3;
    const Distribution oracle     = distribution_oracle(column, token_domain, config);
    if (oracle.tokens.empty() ||
        std::find(oracle.tokens.begin(), oracle.tokens.end(), 3) != oracle.tokens.end()) {
        std::cerr << "p-less real-shape suppression oracle kept token 3 or emptied support\n";
        return 1;
    }

    constexpr int samples = 4096;
    RunResult result =
        run_repeated(column, token_domain, samples, 8, config, 3000, ops::kSamplePurposeDecode);
    return result.integrity_failures +
           verify_distribution("sample p-less suppressed real token-domain B=8", result.tokens,
                               oracle);
}

int p_less_distribution_contract() {
    std::vector<float> column = {1.0f, 0.9f, 0.8f, 0.2f, -1.0f, -2.0f};
    round_to_bf16(column);
    ops::SamplingConfig config;
    config.temperature       = 0.9f;
    config.top_k             = 1;
    config.top_p             = 0.1f;
    config.min_p             = 0.9f;
    config.presence_penalty  = 4.0f;
    config.frequency_penalty = 4.0f;
    config.p_less            = 1;
    config.seed              = 314159;

    const Distribution oracle =
        distribution_oracle(column, static_cast<int>(column.size()), config);
    if (oracle.tokens.size() < 2) {
        std::cerr << "p-less fixture did not keep a multi-token support\n";
        return 1;
    }

    constexpr int samples  = 16384;
    const RunResult result = run_repeated(column, static_cast<int>(column.size()), samples, 8,
                                          config, 50, ops::kSamplePurposeDecode);
    return result.integrity_failures +
           verify_distribution("sample p-less small-vocab", result.tokens, oracle);
}

int p_less_real_shape_contract() {
    constexpr int physical_rows = 248320;
    constexpr int token_domain  = 248077;
    std::vector<float> column(physical_rows, -20.0f);
    const int ids[]      = {3, 17, 7919, 65537, 200003};
    const float logits[] = {2.0f, 1.85f, 1.7f, 0.5f, 0.0f};
    for (int i = 0; i < 5; ++i) { column[ids[i]] = logits[i]; }
    column[token_domain]      = 100.0f;
    column[physical_rows - 1] = 200.0f;
    round_to_bf16(column);

    ops::SamplingConfig config;
    config.temperature = 1.2f;
    config.top_k       = 1;
    config.p_less      = 1;
    config.seed        = 271828;
    const Distribution oracle = distribution_oracle(column, token_domain, config);
    if (oracle.tokens.empty()) {
        std::cerr << "p-less real-shape oracle has empty support\n";
        return 1;
    }

    constexpr int samples = 4096;
    RunResult result =
        run_repeated(column, token_domain, samples, 8, config, 3000, ops::kSamplePurposeDecode);
    return result.integrity_failures +
           verify_distribution("sample p-less real token-domain B=8", result.tokens, oracle);
}

int p_less_heterogeneous_batch_contract() {
    constexpr int token_domain = 8;
    constexpr int batch        = 3;
    std::vector<float> logits(static_cast<std::size_t>(token_domain) * batch, -8.0f);
    logits[1]                                          = 5.0f;
    logits[static_cast<std::size_t>(token_domain) + 2] = 4.0f;
    logits[static_cast<std::size_t>(token_domain) + 3] = 3.5f;
    logits[static_cast<std::size_t>(token_domain) * 2 + 4] = 6.0f;
    round_to_bf16(logits);

    std::vector<ops::SamplingConfig> configs(batch);
    configs[0].temperature = 0.0f;
    configs[0].p_less      = 1;
    configs[1].temperature = 0.8f;
    configs[1].p_less      = 1;
    configs[1].seed        = 7;
    configs[2].temperature = 0.8f;
    configs[2].top_k       = 1;
    configs[2].seed        = 9;

    const RunResult result =
        run_batch(logits, token_domain, token_domain, configs, {1, 2, 3}, ops::kSamplePurposeDecode);
    int failures = result.integrity_failures;
    failures += verify_exact("p-less mixed batch greedy row",
                             std::vector<int>{result.tokens[0]}, {1});
    failures += verify_exact("p-less mixed batch truncated row",
                             std::vector<int>{result.tokens[2]}, {4});
    const Distribution p_less_oracle =
        distribution_oracle(std::vector<float>(logits.begin() + token_domain,
                                               logits.begin() + 2 * token_domain),
                            token_domain, configs[1]);
    const auto it =
        std::find(p_less_oracle.tokens.begin(), p_less_oracle.tokens.end(), result.tokens[1]);
    if (it == p_less_oracle.tokens.end()) {
        std::cerr << "p-less mixed batch stochastic row outside oracle support\n";
        ++failures;
    }
    return failures;
}

int p_less_multiblock_heterogeneous_batch_contract() {
    constexpr int physical_rows = 248320;
    constexpr int token_domain  = 248077;
    constexpr int batch         = 2;
    std::vector<float> logits(static_cast<std::size_t>(physical_rows) * batch, -20.0f);
    const auto set = [&](int row, int token, float value) {
        logits[static_cast<std::size_t>(row) * physical_rows + token] = value;
    };
    set(0, 3, 2.0f);
    set(0, 17, 1.85f);
    set(0, 7919, 1.7f);
    set(0, 65537, 0.5f);
    set(1, 200003, 4.0f);
    set(1, 65537, 3.5f);
    for (int row = 0; row < batch; ++row) {
        set(row, token_domain, 100.0f);
        set(row, physical_rows - 1, 200.0f);
    }
    round_to_bf16(logits);

    std::vector<ops::SamplingConfig> configs(batch);
    configs[0].temperature = 1.2f;
    configs[0].p_less      = 1;
    configs[0].seed        = 271828;
    configs[1].temperature = 0.8f;
    configs[1].top_k       = 1;
    configs[1].seed        = 314159;

    const RunResult result = run_batch(logits, physical_rows, token_domain, configs, {3000, 4000},
                                       ops::kSamplePurposeDecode);
    int failures = result.integrity_failures;
    failures += verify_exact("p-less multi-block mixed batch truncated row",
                             std::vector<int>{result.tokens[1]}, {200003});
    const Distribution p_less_oracle =
        distribution_oracle(std::vector<float>(logits.begin(), logits.begin() + physical_rows),
                            token_domain, configs[0]);
    if (std::find(p_less_oracle.tokens.begin(), p_less_oracle.tokens.end(), result.tokens[0]) ==
        p_less_oracle.tokens.end()) {
        std::cerr << "p-less multi-block mixed batch row outside oracle support\n";
        ++failures;
    }
    return failures;
}

int p_less_typical_exclude_identity_contract() {
    std::vector<float> column = {1.0f, 0.9f, 0.8f, 0.2f, -1.0f, -2.0f};
    round_to_bf16(column);
    ops::SamplingConfig baseline;
    baseline.temperature = 0.9f;
    baseline.p_less      = 1;
    baseline.seed        = 314159;
    const Distribution original =
        distribution_oracle(column, static_cast<int>(column.size()), baseline);
    if (original.tokens.size() < 2) {
        std::cerr << "typical-exclude identity fixture lost multi-token V\n";
        return 1;
    }
    ops::SamplingConfig excluded = baseline;
    excluded.typical_exclude     = 5;
    if (std::find(original.tokens.begin(), original.tokens.end(), 5) != original.tokens.end()) {
        std::cerr << "typical-exclude identity fixture put token 5 in V\n";
        return 1;
    }
    constexpr int samples = 4096;
    const RunResult a =
        run_repeated(column, static_cast<int>(column.size()), samples, 8, baseline, 50,
                     ops::kSamplePurposeDecode);
    const RunResult b =
        run_repeated(column, static_cast<int>(column.size()), samples, 8, excluded, 50,
                     ops::kSamplePurposeDecode);
    int failures = a.integrity_failures + b.integrity_failures;
    if (a.tokens != b.tokens) {
        std::cerr << "typical_exclude outside V changed the p-less draw stream\n";
        ++failures;
    }
    return failures;
}

int p_less_typical_exclude_multi_support_contract() {
    std::vector<float> column = {1.0f, 0.9f, 0.8f, 0.2f, -1.0f, -2.0f};
    round_to_bf16(column);
    ops::SamplingConfig baseline;
    baseline.temperature = 0.9f;
    baseline.p_less      = 1;
    baseline.seed        = 271828;
    const Distribution original =
        distribution_oracle(column, static_cast<int>(column.size()), baseline);
    if (original.tokens.size() < 2) {
        std::cerr << "typical-exclude multi-support fixture lost multi-token V\n";
        return 1;
    }
    const int drop               = original.tokens.front();
    ops::SamplingConfig excluded = baseline;
    excluded.typical_exclude     = drop;
    const Distribution oracle =
        distribution_oracle(column, static_cast<int>(column.size()), excluded);
    if (oracle.tokens.empty() ||
        std::find(oracle.tokens.begin(), oracle.tokens.end(), drop) != oracle.tokens.end()) {
        std::cerr << "typical-exclude multi-support oracle kept the continuation or emptied V\n";
        return 1;
    }
    for (int token : oracle.tokens) {
        if (std::find(original.tokens.begin(), original.tokens.end(), token) ==
            original.tokens.end()) {
            std::cerr << "typical-exclude multi-support admitted token " << token
                      << " outside original V\n";
            return 1;
        }
    }
    constexpr int samples  = 16384;
    const RunResult result = run_repeated(column, static_cast<int>(column.size()), samples, 8,
                                          excluded, 50, ops::kSamplePurposeDecode);
    return result.integrity_failures +
           verify_distribution("sample p-less typical_exclude multi-support", result.tokens,
                               oracle);
}

int p_less_typical_exclude_singleton_runner_up_contract() {
    std::vector<float> column(8, -20.0f);
    column[3] = 8.0f;
    column[5] = 1.0f;
    round_to_bf16(column);
    ops::SamplingConfig config;
    config.temperature     = 2.0f;
    config.p_less          = 1;
    config.seed            = 7;
    config.typical_exclude = 3;
    const Distribution original = [&] {
        ops::SamplingConfig base = config;
        base.typical_exclude     = -1;
        return distribution_oracle(column, static_cast<int>(column.size()), base);
    }();
    if (original.tokens != std::vector<int>{3}) {
        std::cerr << "typical-exclude singleton fixture did not collapse V to the mode\n";
        return 1;
    }
    const Distribution oracle =
        distribution_oracle(column, static_cast<int>(column.size()), config);
    if (oracle.tokens != std::vector<int>{5} || oracle.probabilities != std::vector<double>{1.0}) {
        std::cerr << "typical-exclude singleton oracle was not Dirac on the runner-up\n";
        return 1;
    }
    constexpr int samples  = 256;
    const RunResult result = run_repeated(column, static_cast<int>(column.size()), samples, 8,
                                          config, 11, ops::kSamplePurposeDecode);
    int failures           = result.integrity_failures;
    failures += verify_exact("sample p-less typical_exclude singleton runner-up", result.tokens,
                             std::vector<int>(samples, 5));
    return failures;
}

int p_less_typical_exclude_not_suppressed_rebuild_contract() {
    std::vector<float> column(8, -0.5f);
    column[0] = 6.0f;
    column[1] = 0.5f;
    column[2] = 0.4f;
    column[3] = 0.3f;
    column[4] = 0.2f;
    round_to_bf16(column);
    ops::SamplingConfig typical;
    typical.temperature     = 0.9f;
    typical.p_less          = 1;
    typical.seed            = 99;
    typical.typical_exclude = 0;
    ops::SamplingConfig suppressed;
    suppressed.temperature            = 0.9f;
    suppressed.p_less                 = 1;
    suppressed.seed                   = 99;
    suppressed.suppressed_token_count = 1;
    suppressed.suppressed_tokens[0]   = 0;
    ops::SamplingConfig baseline = typical;
    baseline.typical_exclude     = -1;
    const Distribution original =
        distribution_oracle(column, static_cast<int>(column.size()), baseline);
    const Distribution typical_oracle =
        distribution_oracle(column, static_cast<int>(column.size()), typical);
    const Distribution suppressed_oracle =
        distribution_oracle(column, static_cast<int>(column.size()), suppressed);
    if (original.tokens != std::vector<int>{0}) {
        std::cerr << "rebuild fixture did not collapse original V to the mode\n";
        return 1;
    }
    if (typical_oracle.tokens.size() != 1 || typical_oracle.tokens[0] == 0) {
        std::cerr << "typical_exclude of the mode was not Dirac on an in-domain runner-up\n";
        return 1;
    }
    if (suppressed_oracle.tokens.size() < 2 ||
        std::find(suppressed_oracle.tokens.begin(), suppressed_oracle.tokens.end(), 0) !=
            suppressed_oracle.tokens.end()) {
        std::cerr << "suppressing the mode did not rebuild a larger typical set\n";
        return 1;
    }
    if (typical_oracle.tokens == suppressed_oracle.tokens) {
        std::cerr << "typical_exclude and suppressed_tokens produced the same support\n";
        return 1;
    }
    constexpr int samples = 512;
    const RunResult typical_result =
        run_repeated(column, static_cast<int>(column.size()), samples, 8, typical, 4,
                     ops::kSamplePurposeDecode);
    int failures = typical_result.integrity_failures;
    failures += verify_exact("sample typical_exclude does not rebuild L", typical_result.tokens,
                             std::vector<int>(samples, typical_oracle.tokens[0]));
    const RunResult suppressed_result =
        run_repeated(column, static_cast<int>(column.size()), samples, 8, suppressed, 4,
                     ops::kSamplePurposeDecode);
    failures += suppressed_result.integrity_failures;
    failures += verify_distribution("sample suppressed mode rebuilds V", suppressed_result.tokens,
                                    suppressed_oracle);
    return failures;
}

int p_less_typical_exclude_greedy_ignores_contract() {
    constexpr int token_domain = 8;
    constexpr int batch        = 4;
    std::vector<float> column(token_domain, -4.0f);
    column[2] = 5.0f;
    column[6] = 4.0f;
    round_to_bf16(column);
    const std::vector<float> logits = repeat_column(column, batch);
    ops::SamplingConfig config;
    config.temperature     = 0.0f;
    config.p_less          = 1;
    config.typical_exclude = 2;
    const RunResult result =
        run_homogeneous_batch(logits, token_domain, token_domain, batch, config, 0,
                              ops::kSamplePurposeDecode);
    return result.integrity_failures +
           verify_exact("greedy ignores typical_exclude", result.tokens, std::vector<int>(batch, 2));
}

int p_less_typical_exclude_real_shape_singleton_contract() {
    constexpr int physical_rows = 248320;
    constexpr int token_domain  = 248077;
    std::vector<float> column(physical_rows, -20.0f);
    column[7919]              = 8.0f;
    column[65537]             = 1.5f;
    column[token_domain]      = 100.0f;
    column[physical_rows - 1] = 200.0f;
    round_to_bf16(column);
    ops::SamplingConfig config;
    config.temperature     = 2.0f;
    config.p_less          = 1;
    config.seed            = 4242;
    config.typical_exclude = 7919;
    const Distribution oracle = distribution_oracle(column, token_domain, config);
    if (oracle.tokens != std::vector<int>{65537}) {
        std::cerr << "real-shape typical_exclude singleton oracle was not the runner-up\n";
        return 1;
    }
    constexpr int samples = 256;
    const RunResult result =
        run_repeated(column, token_domain, samples, 8, config, 3000, ops::kSamplePurposeDecode);
    int failures = result.integrity_failures;
    failures += verify_exact("sample p-less typical_exclude real-shape runner-up", result.tokens,
                             std::vector<int>(samples, 65537));
    return failures;
}

int p_less_typical_exclude_real_shape_multi_contract() {
    constexpr int physical_rows = 248320;
    constexpr int token_domain  = 248077;
    std::vector<float> column(physical_rows, -20.0f);
    const int ids[]      = {3, 17, 7919, 65537, 200003};
    const float logits[] = {2.0f, 1.85f, 1.7f, 0.5f, 0.0f};
    for (int i = 0; i < 5; ++i) { column[ids[i]] = logits[i]; }
    column[token_domain]      = 100.0f;
    column[physical_rows - 1] = 200.0f;
    round_to_bf16(column);
    ops::SamplingConfig baseline;
    baseline.temperature = 1.2f;
    baseline.p_less      = 1;
    baseline.seed        = 271828;
    const Distribution original = distribution_oracle(column, token_domain, baseline);
    if (original.tokens.size() < 2) {
        std::cerr << "real-shape typical_exclude multi fixture lost multi-token V\n";
        return 1;
    }
    ops::SamplingConfig excluded = baseline;
    excluded.typical_exclude     = original.tokens.front();
    const Distribution oracle    = distribution_oracle(column, token_domain, excluded);
    constexpr int samples        = 4096;
    const RunResult result =
        run_repeated(column, token_domain, samples, 8, excluded, 3000, ops::kSamplePurposeDecode);
    int failures = result.integrity_failures;
    if (std::find(result.tokens.begin(), result.tokens.end(), excluded.typical_exclude) !=
        result.tokens.end()) {
        std::cerr << "real-shape typical_exclude multi-support emitted the continuation\n";
        ++failures;
    }
    failures += verify_distribution("sample p-less typical_exclude real-shape multi-support",
                                    result.tokens, oracle);
    return failures;
}

int workspace_route_boundary_contract() {
    constexpr int token_domain = 257;
    constexpr int batch        = 8;
    std::vector<float> logits(static_cast<std::size_t>(token_domain) * batch, 0.0f);
    const RunResult result =
        run_homogeneous_batch(logits, token_domain, token_domain, batch, ops::SamplingConfig{}, 0,
                              ops::kSamplePurposeDecode);
    int failures = result.integrity_failures;
    failures +=
        verify_exact("sample workspace route boundary", result.tokens, std::vector<int>(batch, 0));
    return failures;
}

int p_less_first_order_slack_contract() {
    ops::SamplingConfig config;
    config.temperature = 2.0f;
    config.p_less      = 1;
    config.seed        = 7;
    // Near-tie 2-mass: r < L but r ≥ L·exp(-2ε/T), so the slack admits both.
    std::vector<float> near(8, -20.0f);
    near[0] = 0.08f;
    near[1] = 0.0f;
    round_to_bf16(near);
    const Distribution near_oracle =
        distribution_oracle(near, static_cast<int>(near.size()), config);
    if (near_oracle.tokens.size() != 2) {
        std::cerr << "first-order slack did not admit the near-tie runner-up; |V|="
                  << near_oracle.tokens.size() << '\n';
        return 1;
    }
    // Peaked 2-mass stays a singleton: r is far below the relaxed cut.
    std::vector<float> peaked(8, -20.0f);
    peaked[0] = 4.0f;
    peaked[1] = 0.0f;
    round_to_bf16(peaked);
    const Distribution peaked_oracle =
        distribution_oracle(peaked, static_cast<int>(peaked.size()), config);
    if (peaked_oracle.tokens != std::vector<int>{0}) {
        std::cerr << "first-order slack admitted a far runner-up into a peaked V\n";
        return 1;
    }
    constexpr int samples = 256;
    const RunResult result =
        run_repeated(near, static_cast<int>(near.size()), samples, 8, config, 11,
                     ops::kSamplePurposeDecode);
    int failures = result.integrity_failures;
    failures += verify_distribution("sample p-less first-order slack near-tie", result.tokens,
                                    near_oracle);
    return failures;
}

double p_less_collision(const std::vector<float>& column, int token_domain, float temperature) {
    double max_scaled = 0.0;
    bool have_max     = false;
    for (int token = 0; token < token_domain; ++token) {
        const double scaled =
            static_cast<double>(column[static_cast<std::size_t>(token)]) / temperature;
        if (!have_max || scaled > max_scaled) {
            max_scaled = scaled;
            have_max   = true;
        }
    }
    double total = 0.0;
    std::vector<double> weights(static_cast<std::size_t>(token_domain));
    for (int token = 0; token < token_domain; ++token) {
        const double w = std::exp(
            static_cast<double>(column[static_cast<std::size_t>(token)]) / temperature - max_scaled);
        weights[static_cast<std::size_t>(token)] = w;
        total += w;
    }
    double collision = 0.0;
    for (int token = 0; token < token_domain; ++token) {
        const double p = weights[static_cast<std::size_t>(token)] / total;
        collision += p * p;
    }
    return collision;
}

int p_less_peaked_invariance_contract() {
    struct Shape {
        int physical_rows;
        int token_domain;
        int a;
        int b;
        const char* label;
    };
    const Shape shapes[] = {
        {257, 257, 3, 11, "sample p-less peaked invariance N=257"},
        {248320, 248077, 17, 7919, "sample p-less peaked invariance N=248077"},
    };
    const float temps[] = {0.6f, 1.5f, 2.0f};
    int failures        = 0;
    for (const Shape& shape : shapes) {
        for (float temperature : temps) {
            std::vector<float> column(static_cast<std::size_t>(shape.physical_rows), -20.0f);
            column[static_cast<std::size_t>(shape.a)] = 8.0f;
            column[static_cast<std::size_t>(shape.b)] = 7.0f;
            if (shape.physical_rows > shape.token_domain) {
                column[static_cast<std::size_t>(shape.token_domain)]           = 100.0f;
                column[static_cast<std::size_t>(shape.physical_rows - 1)] = 200.0f;
            }
            round_to_bf16(column);
            ops::SamplingConfig config;
            config.temperature = temperature;
            config.p_less      = 1;
            config.seed        = 17;
            const double collision = p_less_collision(column, shape.token_domain, temperature);
            const double n_eff     = 1.0 / collision;
            const double slack     = collision / ops::p_less_admission_scale(temperature);
            const double cut = ops::p_less_membership_cut(collision, temperature);
            if (!(n_eff < 0.5 * static_cast<double>(ops::kPLessMaxEffectiveSupport)) ||
                cut != slack) {
                std::cerr << shape.label << " T=" << temperature
                          << ": n_eff=" << n_eff << " was not idle vs M\n";
                return failures + 1;
            }
            const Distribution oracle =
                distribution_oracle(column, shape.token_domain, config);
            if (oracle.tokens.empty() ||
                std::find(oracle.tokens.begin(), oracle.tokens.end(), shape.a) ==
                    oracle.tokens.end()) {
                std::cerr << shape.label << " T=" << temperature
                          << ": peaked support lost the mode\n";
                return failures + 1;
            }
            const int samples = shape.token_domain > 1000 ? 1024 : 4096;
            const RunResult result =
                run_repeated(column, shape.token_domain, samples, 8, config, 50,
                             ops::kSamplePurposeDecode);
            failures += result.integrity_failures;
            failures += verify_distribution(
                (std::string(shape.label) + " T=" + std::to_string(temperature)).c_str(),
                result.tokens, oracle);
        }
    }
    return failures;
}

int p_less_small_vocab_uniform_contract() {
    constexpr int token_domain = 257;
    std::vector<float> column(token_domain, 0.0f);
    round_to_bf16(column);
    ops::SamplingConfig config;
    config.temperature = 2.0f;
    config.p_less      = 1;
    config.seed        = 19;
    const double collision = p_less_collision(column, token_domain, config.temperature);
    const double cut       = ops::p_less_membership_cut(collision, config.temperature);
    if (!(collision > 1.0 / static_cast<double>(ops::kPLessMaxEffectiveSupport)) ||
        cut != collision / ops::p_less_admission_scale(config.temperature)) {
        std::cerr << "small-vocab uniform floor was not idle\n";
        return 1;
    }
    const Distribution oracle = distribution_oracle(column, token_domain, config);
    if (oracle.tokens.size() != static_cast<std::size_t>(token_domain)) {
        std::cerr << "small-vocab uniform support size " << oracle.tokens.size()
                  << " expected " << token_domain << '\n';
        return 1;
    }
    constexpr int samples  = 16384;
    const RunResult result = run_repeated(column, token_domain, samples, 8, config, 50,
                                          ops::kSamplePurposeDecode);
    return result.integrity_failures +
           verify_distribution("sample p-less small-vocab uniform", result.tokens, oracle);
}

int p_less_soup_collapse_contract() {
    constexpr int physical_rows = 248320;
    constexpr int token_domain  = 248077;
    int failures                = 0;
    for (float temperature : {1.5f, 2.0f}) {
        std::vector<float> column(physical_rows, 0.0f);
        column[token_domain]      = 100.0f;
        column[physical_rows - 1] = 200.0f;
        round_to_bf16(column);
        ops::SamplingConfig config;
        config.temperature     = temperature;
        config.p_less          = 1;
        config.seed            = 23;
        const double collision = p_less_collision(column, token_domain, temperature);
        const double n_eff     = 1.0 / collision;
        if (!(n_eff > static_cast<double>(ops::kPLessMaxEffectiveSupport))) {
            std::cerr << "soup fixture T=" << temperature << " n_eff=" << n_eff
                      << " did not exceed M\n";
            return failures + 1;
        }
        const Distribution oracle = distribution_oracle(column, token_domain, config);
        if (oracle.tokens != std::vector<int>{0} || oracle.probabilities != std::vector<double>{1.0}) {
            std::cerr << "soup T=" << temperature << " oracle was not Dirac on min-argmax\n";
            return failures + 1;
        }
        constexpr int samples  = 256;
        const RunResult result = run_repeated(column, token_domain, samples, 8, config, 50,
                                              ops::kSamplePurposeDecode);
        failures += result.integrity_failures;
        failures += verify_exact(
            (std::string("sample p-less soup Dirac T=") + std::to_string(temperature)).c_str(),
            result.tokens, std::vector<int>(samples, 0));
    }
    return failures;
}

int p_less_floor_binds_with_head_contract() {
    constexpr int token_domain = 4096;
    constexpr int mode         = 13;
    std::vector<float> column(token_domain, 0.0f);
    column[mode] = 3.0f;
    round_to_bf16(column);
    ops::SamplingConfig config;
    config.temperature     = 2.0f;
    config.p_less          = 1;
    config.seed            = 29;
    const double collision = p_less_collision(column, token_domain, config.temperature);
    const double n_eff     = 1.0 / collision;
    if (!(n_eff > static_cast<double>(ops::kPLessMaxEffectiveSupport))) {
        std::cerr << "K=4096 head fixture n_eff=" << n_eff << " did not exceed M\n";
        return 1;
    }
    const Distribution oracle = distribution_oracle(column, token_domain, config);
    if (oracle.tokens.size() != 1 || oracle.tokens.front() != mode) {
        std::cerr << "K=4096 floor-bind support size " << oracle.tokens.size();
        if (!oracle.tokens.empty()) { std::cerr << " head=" << oracle.tokens.front(); }
        std::cerr << " expected Dirac on " << mode << '\n';
        return 1;
    }
    constexpr int samples  = 1024;
    const RunResult result = run_repeated(column, token_domain, samples, 8, config, 50,
                                          ops::kSamplePurposeDecode);
    return result.integrity_failures +
           verify_exact("sample p-less K=4096 floor-bind Dirac", result.tokens,
                        std::vector<int>(samples, mode));
}

int eligibility_masks(int domain, int physical) {
    const int words = (domain + 31) / 32;
    std::vector<std::uint32_t> masks(3 * words, 0);
    const std::vector<int> expected{31, 32, domain - 1};
    std::vector<float> logits(3 * physical, -20);
    for (int row = 0; row < 3; ++row) {
        masks[row * words + expected[row] / 32] |= 1u << (expected[row] % 32);
        masks[row * words] |= 1u << 8;
        logits[row * physical] = 100;
        logits[row * physical + 8] = 90;
        for (int v = domain; v < physical; ++v) { logits[row * physical + v] = 200; }
        if (domain % 32) { masks[(row + 1) * words - 1] |= ~0u << (domain % 32); }
    }
    auto device_masks = to_device(masks);
    std::vector<ops::SamplingConfig> configs(3);
    for (int row = 0; row < 3; ++row) {
        auto& cfg = configs[row];
        cfg.temperature = row == 2 ? 0 : 2;
        cfg.p_less = row == 0;
        cfg.allowed_token_words = static_cast<const std::uint32_t*>(device_masks.p) + row * words;
        cfg.suppressed_token_count = 1;
        cfg.suppressed_tokens[0] = 8;
    }
    const auto result = run_batch(logits, physical, domain, configs, {17, 18, 19},
                                  ops::kSamplePurposeDecode);
    int failures = result.integrity_failures;
    failures += verify_exact("mixed-mode eligibility masks", result.tokens, expected);
    failures += verify_exact("sample masks unchanged",
        from_device<std::uint32_t>(device_masks, masks.size()), masks);
    return failures;
}

int masked_p_less_distribution(int domain, int physical) {
    std::vector<std::uint32_t> mask((domain + 31) / 32, 0);
    mask[0] = 1u << 31;
    mask[(domain - 1) / 32] |= 1u << ((domain - 1) % 32);
    auto device_mask = to_device(mask);
    std::vector<float> logits(8 * physical, 0);
    for (int row = 0; row < 8; ++row) { logits[row * physical] = 100; }
    int left = 0, failures = 0;
    for (int group = 0; group < 8; ++group) {
        std::vector<ops::SamplingConfig> configs(8);
        for (int row = 0; row < 8; ++row) {
            configs[row].p_less = 1;
            configs[row].temperature = 2;
            configs[row].seed = group * 8 + row;
            configs[row].allowed_token_words = static_cast<const std::uint32_t*>(device_mask.p);
        }
        const auto result = run_batch(logits, physical, domain, configs,
            std::vector<int>(8, 19), ops::kSamplePurposeDecode);
        failures += result.integrity_failures;
        for (int token : result.tokens) {
            left += token == 31;
            failures += token != 31 && token != domain - 1;
        }
    }
    // Independent law: only two equal represented logits are eligible, hence
    // p=(1/2,1/2), collision=1/2, and both remain in the p-less support.
    if (left < 16 || left > 48) {
        std::cerr << "masked p-less collapsed a two-atom distribution: " << left << "/64\n";
        ++failures;
    }
    return failures;
}

} // namespace

int main() {
    if (cuda_unavailable()) {
        std::cerr << "FAIL: no usable CUDA device\n";
        return 1;
    }

    int failures            = 0;
    failures += eligibility_masks(64, 64);
    failures += eligibility_masks(1000, 1024);
    failures += eligibility_masks(248077, 248320);
    failures += masked_p_less_distribution(64, 64);
    failures += masked_p_less_distribution(248077, 248320);
    const std::size_t at_16 = ops::sampling_workspace_capacity_bytes(257, 16, 16);
    if (ops::sampling_workspace_capacity_bytes(256, 1, 16) != 0 || at_16 == 0 ||
        ops::sampling_workspace_capacity_bytes(257, 17, 17) != 0 ||
        ops::sampling_workspace_capacity_bytes(257, 1, 17) != at_16) {
        std::cerr << "sampling workspace route boundary contract failed\n";
        ++failures;
    }
    try {
        (void)ops::sampling_workspace_capacity_bytes(257, 0, 16);
        std::cerr << "sampling workspace accepted an invalid lane interval\n";
        ++failures;
    } catch (const std::invalid_argument&) {}
    failures += greedy_contract();
    failures += deterministic_stochastic_contract();
    failures += heterogeneous_batch_contract();
    failures += suppressed_token_contract();
    failures += filtered_distribution_contract();
    failures += capped_distribution_contract();
    failures += real_shape_distribution_contract();
    failures += rng_key_contract();
    failures += workspace_route_boundary_contract();
    failures += p_less_distribution_contract();
    failures += p_less_real_shape_contract();
    failures += p_less_heterogeneous_batch_contract();
    failures += p_less_multiblock_heterogeneous_batch_contract();
    failures += p_less_suppressed_token_contract();
    failures += p_less_suppressed_distribution_contract();
    failures += p_less_suppressed_real_shape_contract();
    failures += p_less_typical_exclude_identity_contract();
    failures += p_less_typical_exclude_multi_support_contract();
    failures += p_less_typical_exclude_singleton_runner_up_contract();
    failures += p_less_typical_exclude_not_suppressed_rebuild_contract();
    failures += p_less_typical_exclude_greedy_ignores_contract();
    failures += p_less_typical_exclude_real_shape_singleton_contract();
    failures += p_less_typical_exclude_real_shape_multi_contract();
    failures += p_less_first_order_slack_contract();
    failures += p_less_peaked_invariance_contract();
    failures += p_less_small_vocab_uniform_contract();
    failures += p_less_soup_collapse_contract();
    failures += p_less_floor_binds_with_head_contract();

    std::cout << (failures == 0 ? "OK" : "FAIL") << " sample public contract\n";
    return failures == 0 ? 0 : 1;
}
