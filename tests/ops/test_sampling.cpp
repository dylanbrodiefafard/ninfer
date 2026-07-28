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
    std::vector<int> counts;
    int integrity_failures = 0;
};

bool same_config(const ops::SamplingConfig& a, const ops::SamplingConfig& b) {
    return a.temperature == b.temperature && a.top_k == b.top_k && a.top_p == b.top_p &&
           a.min_p == b.min_p && a.presence_penalty == b.presence_penalty &&
           a.frequency_penalty == b.frequency_penalty && a.seed == b.seed &&
           a.token_counts == b.token_counts;
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
    std::vector<Candidate> candidates(static_cast<std::size_t>(token_domain));
    for (int token = 0; token < token_domain; ++token) {
        const int count = counts == nullptr ? 0 : (*counts)[static_cast<std::size_t>(token)];
        double adjusted = static_cast<double>(column[static_cast<std::size_t>(token)]);
        if (count > 0) { adjusted -= static_cast<double>(config.presence_penalty); }
        adjusted -= static_cast<double>(config.frequency_penalty) * static_cast<double>(count);
        candidates[static_cast<std::size_t>(token)] = {adjusted, token};
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

RunResult run_once(const std::vector<float>& logits, int physical_rows, int token_domain,
                   int columns, ops::SamplingConfig config, int position, int purpose,
                   const std::vector<int>* initial_counts = nullptr) {
    const std::vector<std::uint16_t> input_bits = bf16_bits(logits);
    DeviceBuffer device_logits                  = to_device(input_bits);
    GuardedDeviceBuffer device_out(static_cast<std::size_t>(columns) * sizeof(std::int32_t));
    const std::vector<int> output_sentinel(static_cast<std::size_t>(columns), -777777);
    device_out.copy_from_host(output_sentinel.data(),
                              output_sentinel.size() * sizeof(std::int32_t));

    std::unique_ptr<GuardedDeviceBuffer> device_counts;
    if (initial_counts != nullptr) {
        device_counts =
            std::make_unique<GuardedDeviceBuffer>(initial_counts->size() * sizeof(std::int32_t));
        device_counts->copy_from_host(initial_counts->data(),
                                      initial_counts->size() * sizeof(std::int32_t));
        config.token_counts = static_cast<std::int32_t*>(device_counts->data());
    }
    const ops::SamplingConfig expected_config = config;
    DeviceBuffer device_config                = to_device(std::vector<ops::SamplingConfig>{config});
    DeviceBuffer device_pos                   = to_device(std::vector<std::int32_t>{position});

    Tensor logits_tensor(device_logits.p, DType::BF16, {physical_rows, columns});
    Tensor out_tensor(device_out.data(), DType::I32, {columns});
    WorkspaceArena workspace(
        std::max<std::size_t>(256, ops::sampling_workspace_bytes(token_domain, columns)));
    ops::sample(logits_tensor, out_tensor, token_domain,
                static_cast<const ops::SamplingConfig*>(device_config.p),
                static_cast<const std::int32_t*>(device_pos.p), purpose, workspace, nullptr);
    cuda_synchronize();

    RunResult result;
    result.tokens = from_device<int>(device_out.data(), static_cast<std::size_t>(columns));
    result.integrity_failures += device_out.verify_guards("sample output");
    result.integrity_failures +=
        verify_exact("sample read-only logits",
                     from_device<std::uint16_t>(device_logits, input_bits.size()), input_bits);

    const ops::SamplingConfig actual_config =
        from_device<ops::SamplingConfig>(device_config, 1).front();
    if (!same_config(actual_config, expected_config)) {
        std::cerr << "sample modified SamplingConfig\n";
        ++result.integrity_failures;
    }
    const int actual_position = from_device<std::int32_t>(device_pos, 1).front();
    if (actual_position != position) {
        std::cerr << "sample modified pos_base: got=" << actual_position << " expected=" << position
                  << '\n';
        ++result.integrity_failures;
    }

    if (device_counts != nullptr) {
        result.counts =
            from_device<int>(device_counts->data(), static_cast<std::size_t>(token_domain));
        result.integrity_failures += device_counts->verify_guards("sample token_counts");
    }
    return result;
}

RunResult run_repeated(const std::vector<float>& column, int token_domain, int total,
                       int batch_columns, ops::SamplingConfig config, int position, int purpose) {
    const int physical_rows                     = static_cast<int>(column.size());
    const std::vector<float> logits             = repeat_column(column, batch_columns);
    const std::vector<std::uint16_t> input_bits = bf16_bits(logits);
    DeviceBuffer device_logits                  = to_device(input_bits);
    GuardedDeviceBuffer device_out(static_cast<std::size_t>(batch_columns) * sizeof(std::int32_t));
    DeviceBuffer collected(static_cast<std::size_t>(total) * sizeof(std::int32_t));
    DeviceBuffer device_config                = to_device(std::vector<ops::SamplingConfig>{config});
    DeviceBuffer device_pos                   = to_device(std::vector<std::int32_t>{position});
    const ops::SamplingConfig expected_config = config;

    Tensor logits_tensor(device_logits.p, DType::BF16, {physical_rows, batch_columns});
    Tensor out_tensor(device_out.data(), DType::I32, {batch_columns});
    WorkspaceArena workspace(
        std::max<std::size_t>(256, ops::sampling_workspace_bytes(token_domain, batch_columns)));

    int final_position = position;
    for (int produced = 0; produced < total; produced += batch_columns) {
        final_position = position + produced;
        device_pos.copy_from_host(&final_position, sizeof(final_position));
        ops::sample(logits_tensor, out_tensor, token_domain,
                    static_cast<const ops::SamplingConfig*>(device_config.p),
                    static_cast<const std::int32_t*>(device_pos.p), purpose, workspace, nullptr);
        cuda_check(cudaMemcpy(static_cast<std::int32_t*>(collected.p) + produced, device_out.data(),
                              static_cast<std::size_t>(batch_columns) * sizeof(std::int32_t),
                              cudaMemcpyDeviceToDevice),
                   "cudaMemcpy sampled batch");
    }
    cuda_synchronize();

    RunResult result;
    result.tokens = from_device_i32(collected, static_cast<std::size_t>(total));
    result.integrity_failures += device_out.verify_guards("sample repeated output");
    result.integrity_failures +=
        verify_exact("sample repeated read-only logits",
                     from_device<std::uint16_t>(device_logits, input_bits.size()), input_bits);
    const ops::SamplingConfig actual_config =
        from_device<ops::SamplingConfig>(device_config, 1).front();
    if (!same_config(actual_config, expected_config)) {
        std::cerr << "sample repeated modified SamplingConfig\n";
        ++result.integrity_failures;
    }
    if (from_device<std::int32_t>(device_pos, 1).front() != final_position) {
        std::cerr << "sample repeated modified pos_base\n";
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
    constexpr int columns       = 3;
    std::vector<float> logits(static_cast<std::size_t>(physical_rows) * columns, -9.0f);
    for (int column = 0; column < columns; ++column) {
        const std::size_t base = static_cast<std::size_t>(column) * physical_rows;
        int first              = (17 + 7919 * column) % token_domain;
        int second             = token_domain - 1 - ((31 + 65537 * column) % token_domain);
        if (second == first) { second = (first + 1) % token_domain; }
        if (second < first) { std::swap(first, second); }
        logits[base + first]             = column == 0 ? -0.0f : 16.0f + column;
        logits[base + second]            = column == 0 ? 0.0f : 16.0f + column;
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

    const RunResult result = run_once(logits, physical_rows, token_domain, columns, config, 77,
                                      ops::kSamplePurposeDecode, &counts);
    int failures           = result.integrity_failures;
    failures += verify_exact("sample greedy mathematical result", result.tokens,
                             greedy_oracle(logits, physical_rows, token_domain, columns));
    failures += verify_exact("sample greedy skips token_counts updates", result.counts, counts);
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
        RunResult result =
            run_once(column, static_cast<int>(column.size()), static_cast<int>(column.size()), 1,
                     config, 11, ops::kSamplePurposeDecode, &test_case.counts);

        std::vector<int> expected_counts = test_case.counts;
        ++expected_counts[static_cast<std::size_t>(oracle.tokens.front())];
        failures += result.integrity_failures;
        failures += verify_exact(test_case.label, result.tokens, {oracle.tokens.front()});
        failures +=
            verify_exact("sample increments only selected token", result.counts, expected_counts);
    }
    return failures;
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

    constexpr int samples           = 16384;
    const std::vector<float> logits = repeat_column(column, samples);
    const RunResult result =
        run_once(logits, static_cast<int>(column.size()), static_cast<int>(column.size()), samples,
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

    constexpr int samples = 16384;
    const RunResult result =
        run_once(repeat_column(column, samples), static_cast<int>(column.size()),
                 static_cast<int>(column.size()), samples, config, 900, ops::kSamplePurposePrefill);
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
        run_repeated(column, token_domain, samples, 1, config, 2000, ops::kSamplePurposeDecode);
    return result.integrity_failures +
           verify_distribution("sample real token-domain T=1", result.tokens, oracle);
}

int rng_key_contract() {
    std::vector<float> column = {0.0f, 0.0f};
    round_to_bf16(column);
    ops::SamplingConfig config;
    config.temperature = 1.0f;
    config.top_k       = 2;
    config.seed        = 424242;

    constexpr int columns    = 128;
    const RunResult baseline = run_once(repeat_column(column, columns), 2, 2, columns, config, 100,
                                        ops::kSamplePurposeDecode);
    const RunResult repeat   = run_once(repeat_column(column, columns), 2, 2, columns, config, 100,
                                        ops::kSamplePurposeDecode);
    const RunResult shifted  = run_once(repeat_column(column, columns - 1), 2, 2, columns - 1,
                                        config, 101, ops::kSamplePurposeDecode);
    const RunResult other_purpose = run_once(repeat_column(column, columns), 2, 2, columns, config,
                                             100, ops::kSamplePurposePrefill);
    ops::SamplingConfig other_seed_config = config;
    ++other_seed_config.seed;
    const RunResult other_seed = run_once(repeat_column(column, columns), 2, 2, columns,
                                          other_seed_config, 100, ops::kSamplePurposeDecode);

    int failures = baseline.integrity_failures + repeat.integrity_failures +
                   shifted.integrity_failures + other_purpose.integrity_failures +
                   other_seed.integrity_failures;
    failures += verify_exact("sample identical counter key is reproducible", repeat.tokens,
                             baseline.tokens);
    failures += verify_exact("sample position selects the corresponding counter", shifted.tokens,
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

} // namespace

int main() {
    if (cuda_unavailable()) {
        std::cout << "SKIP: no usable CUDA device\n";
        return 77;
    }

    int failures = 0;
    failures += greedy_contract();
    failures += deterministic_stochastic_contract();
    failures += filtered_distribution_contract();
    failures += capped_distribution_contract();
    failures += real_shape_distribution_contract();
    failures += rng_key_contract();

    std::cout << (failures == 0 ? "OK" : "FAIL") << " sample public contract\n";
    return failures == 0 ? 0 : 1;
}
