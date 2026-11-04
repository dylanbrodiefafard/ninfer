#include <hip/hip_bfloat16.h>
#include <hip/hip_runtime.h>

#include "core/arena.h"
#include "core/device.h"
#include "core/tensor.h"
#include "ninfer/ops/speculative_round.h"
#include "sampling_reference.h"
#include "ninfer/types.h"

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <limits>
#include <numeric>
#include <random>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace {

constexpr int kRealVocabulary = 248320;

[[noreturn]] void fail(const std::string& message) { throw std::runtime_error(message); }

void require(bool condition, const std::string& message) {
    if (!condition) { fail(message); }
}

template <typename T>
void require_equal(const std::vector<T>& actual, const std::vector<T>& expected,
                   const std::string& label) {
    if (actual == expected) { return; }
    std::size_t index = 0;
    while (index < actual.size() && index < expected.size() && actual[index] == expected[index]) {
        ++index;
    }
    std::string detail;
    if (index < actual.size() && index < expected.size()) {
        detail = " actual=" + std::to_string(static_cast<long long>(actual[index])) +
                 " expected=" + std::to_string(static_cast<long long>(expected[index]));
    }
    fail(label + " differs at index " + std::to_string(index) + detail);
}

std::uint16_t bf16_rne_bits(float value) {
    std::uint32_t word           = std::bit_cast<std::uint32_t>(value);
    const std::uint32_t absolute = word & 0x7fffffffU;
    if ((absolute & 0x7f800000U) != 0x7f800000U) {
        word += 0x7fffU + ((word >> 16U) & 1U);
    } else if ((word & 0xffffU) != 0U) {
        word |= 0x10000U;
    }
    return static_cast<std::uint16_t>(word >> 16U);
}

hip_bfloat16 bf16(float value) {
    hip_bfloat16 result;
    result.data = bf16_rne_bits(value);
    return result;
}

float represented(hip_bfloat16 value) {
    return std::bit_cast<float>(static_cast<std::uint32_t>(value.data) << 16U);
}

std::vector<std::uint16_t> bf16_bits(const std::vector<hip_bfloat16>& values) {
    std::vector<std::uint16_t> result(values.size());
    for (std::size_t index = 0; index < values.size(); ++index) {
        result[index] = values[index].data;
    }
    return result;
}

unsigned long long splitmix64(unsigned long long value) {
    value += 0x9E3779B97F4A7C15ULL;
    value = (value ^ (value >> 30)) * 0xBF58476D1CE4E5B9ULL;
    value = (value ^ (value >> 27)) * 0x94D049BB133111EBULL;
    return value ^ (value >> 31);
}

float uniform(unsigned long long seed, int position, int purpose, unsigned int sub = 0U) {
    unsigned long long key =
        splitmix64(seed ^ (static_cast<unsigned long long>(static_cast<unsigned int>(position)) *
                           0xD1B54A32D192ED03ULL));
    key = splitmix64(key ^
                     (static_cast<unsigned long long>(static_cast<unsigned int>(purpose)) << 21) ^
                     (static_cast<unsigned long long>(sub) * 0x2545F4914F6CDD1DULL));
    return static_cast<float>(static_cast<unsigned int>(key >> 40)) * (1.0F / 16777216.0F);
}

int select_r9700() {
    int count = 0;
    HIP_CHECK(hipGetDeviceCount(&count));
    int selected = -1;
    for (int device = 0; device < count; ++device) {
        hipDeviceProp_t properties{};
        HIP_CHECK(hipGetDeviceProperties(&properties, device));
        if (properties.integrated == 0 && std::string_view(properties.gcnArchName) == "gfx1201") {
            if (selected >= 0) { fail("expected exactly one discrete gfx1201 device"); }
            selected = device;
        }
    }
    if (selected < 0) { fail("no discrete gfx1201 R9700 found"); }
    return selected;
}

template <typename T>
std::vector<T> copy_from(const ninfer::DeviceBuffer& buffer, std::size_t count) {
    std::vector<T> result(count);
    buffer.copy_to_host(result.data(), result.size() * sizeof(T));
    return result;
}

struct Distribution {
    std::vector<int> ids;
    std::vector<double> probabilities;
};

ninfer::ops::SamplingConfig column_config(ninfer::ops::SamplingConfig config, int column) {
    if (config.allowed_token_words)
        config.allowed_token_words += column * config.allowed_token_column_stride;
    // Cycle exit applies only to the first generated position of this round. Later chain
    // hops and non-root tree nodes use the unmodified target p-less law (public Op contract).
    if (column > 0) config.typical_exclude = -1;
    return config;
}

Distribution distribution_oracle(const hip_bfloat16* logits, int token_domain,
                                 const ninfer::ops::SamplingConfig& config,
                                 const std::vector<int>* counts, const std::vector<int>& overlay) {
    if (config.p_less != 0 && config.temperature > 0) {
        std::vector<double> values(token_domain);
        for (int i = 0; i < token_domain; ++i) values[i] = represented(logits[i]);
        const auto result = sampling_reference::p_less(values, config);
        return {result.ids, result.probabilities};
    }
    std::vector<std::pair<float, int>> candidates;
    candidates.reserve(static_cast<std::size_t>(token_domain));
    for (int token = 0; token < token_domain; ++token) {
        if (!sampling_reference::eligible(token, config)) continue;
        float value = represented(logits[token]);
        if (config.presence_penalty != 0.0F || config.frequency_penalty != 0.0F) {
            int count = counts == nullptr ? 0 : (*counts)[static_cast<std::size_t>(token)];
            count += static_cast<int>(std::count(overlay.begin(), overlay.end(), token));
            if (count > 0) { value -= config.presence_penalty; }
            value -= config.frequency_penalty * static_cast<float>(count);
        }
        candidates.emplace_back(value, token);
    }
    const auto better = [](const auto& lhs, const auto& rhs) {
        return lhs.first > rhs.first || (lhs.first == rhs.first && lhs.second < rhs.second);
    };
    int cap = 20;
    if (config.top_k > 0 && config.top_k < cap) { cap = config.top_k; }
    cap = std::min(cap, static_cast<int>(candidates.size()));
    std::partial_sort(candidates.begin(), candidates.begin() + cap, candidates.end(), better);
    candidates.resize(static_cast<std::size_t>(cap));

    std::vector<double> weights(static_cast<std::size_t>(cap));
    const double maximum =
        static_cast<double>(candidates[0].first) / static_cast<double>(config.temperature);
    double total = 0.0;
    for (int index = 0; index < cap; ++index) {
        weights[static_cast<std::size_t>(index)] =
            std::exp(static_cast<double>(candidates[static_cast<std::size_t>(index)].first) /
                         static_cast<double>(config.temperature) -
                     maximum);
        total += weights[static_cast<std::size_t>(index)];
    }
    const double min_threshold =
        config.min_p > 0.0F ? static_cast<double>(config.min_p) * weights[0] : -1.0;
    const double top_p_target = static_cast<double>(config.top_p) * total;
    double cumulative         = 0.0;
    int support               = 0;
    for (int index = 0; index < cap; ++index) {
        if (min_threshold >= 0.0 && weights[static_cast<std::size_t>(index)] < min_threshold) {
            break;
        }
        cumulative += weights[static_cast<std::size_t>(index)];
        support = index + 1;
        if (config.top_p < 1.0F && cumulative >= top_p_target) { break; }
    }
    support              = std::max(1, support);
    double support_total = 0.0;
    for (int index = 0; index < support; ++index) {
        support_total += weights[static_cast<std::size_t>(index)];
    }
    Distribution result;
    result.ids.reserve(static_cast<std::size_t>(support));
    result.probabilities.reserve(static_cast<std::size_t>(support));
    for (int index = 0; index < support; ++index) {
        result.ids.push_back(candidates[static_cast<std::size_t>(index)].second);
        result.probabilities.push_back(weights[static_cast<std::size_t>(index)] / support_total);
    }
    return result;
}

int pick(const Distribution& distribution, int excluded, float draw) {
    double mass = 0.0;
    for (std::size_t index = 0; index < distribution.ids.size(); ++index) {
        if (distribution.ids[index] != excluded) { mass += distribution.probabilities[index]; }
    }
    if (!(mass > 0.0)) { return distribution.ids.front(); }
    const double target = static_cast<double>(draw) * mass;
    double cumulative   = 0.0;
    int selected        = distribution.ids.front();
    for (std::size_t index = 0; index < distribution.ids.size(); ++index) {
        if (distribution.ids[index] == excluded) { continue; }
        cumulative += distribution.probabilities[index];
        selected = distribution.ids[index];
        if (target < cumulative) { break; }
    }
    return selected;
}

float q_probability(const std::vector<int>* selector_ids, const std::vector<float>* selector_q,
                    int base, int selector_count, int token) {
    if (selector_ids == nullptr || selector_q == nullptr) { return 1.0F; }
    for (int index = 0; index < selector_count; ++index) {
        if ((*selector_ids)[static_cast<std::size_t>(base + index)] == token) {
            return (*selector_q)[static_cast<std::size_t>(base + index)];
        }
    }
    return 0.0F;
}

// Inverse CDF of max(0, weight p - q); weight = 1 is the Leviathan residual and p-less block
// verification passes its acceptance weight p_tau. A null selector is one-hot q at the draft.
int pick_residual(const Distribution& distribution, const std::vector<int>* selector_ids,
                  const std::vector<float>* selector_q, int base, int selector_count, int draft,
                  float draw, double weight = 1.0) {
    if (selector_ids == nullptr || selector_q == nullptr) {
        return pick(distribution, draft, draw);
    }
    std::vector<double> residual(distribution.ids.size());
    double mass = 0.0;
    for (std::size_t index = 0; index < distribution.ids.size(); ++index) {
        residual[index] = std::max(
            0.0, weight * distribution.probabilities[index] -
                     static_cast<double>(q_probability(selector_ids, selector_q, base,
                                                       selector_count, distribution.ids[index])));
        mass += residual[index];
    }
    if (!(mass > 0.0)) { return draft; }
    const double target = static_cast<double>(draw) * mass;
    double cumulative   = 0.0;
    int selected        = distribution.ids.front();
    for (std::size_t index = 0; index < distribution.ids.size(); ++index) {
        if (!(residual[index] > 0.0)) { continue; }
        cumulative += residual[index];
        selected = distribution.ids[index];
        if (target < cumulative) { break; }
    }
    return selected;
}

double probability_of(const Distribution& distribution, int token) {
    for (std::size_t index = 0; index < distribution.ids.size(); ++index) {
        if (distribution.ids[index] == token) { return distribution.probabilities[index]; }
    }
    return 0.0;
}

// Block verification (Sun et al. 2024, Algorithm 2) for one p-less chain row, evaluated in FP64
// from the independent p-less column laws p'_i and the recorded proposal laws q_i (one-hot at
// the draft when no selector is recorded): p_0 = 1, p_i = min(p_{i-1} p'_{i-1}(d)/q_{i-1}(d), 1),
// h_i = Z_i/(Z_i + 1 - p_i) with Z_i = sum_x max(p_i p'_i(x) - q_i(x), 0) for i < extent, and
// h_extent = p_extent. None of this depends on the seed.
struct BlockLaws {
    std::vector<Distribution> columns;
    std::vector<double> weight;
    std::vector<double> h;
};

BlockLaws p_less_block_laws(const hip_bfloat16* row_logits, int token_domain, int extent,
                            const ninfer::ops::SamplingConfig& config, const int* row_drafts,
                            const std::vector<int>* selector_ids,
                            const std::vector<float>* selector_q, int selector_row_base,
                            int selector_count) {
    BlockLaws laws;
    for (int column = 0; column <= extent; ++column) {
        laws.columns.push_back(
            distribution_oracle(row_logits + static_cast<std::size_t>(column) * token_domain,
                                token_domain, column_config(config, column), nullptr, {}));
    }
    const auto q_at = [&](int hop, int token) {
        if (selector_ids == nullptr || selector_q == nullptr) {
            return token == row_drafts[hop] ? 1.0 : 0.0;
        }
        return static_cast<double>(q_probability(selector_ids, selector_q,
                                                 selector_row_base + selector_count * hop,
                                                 selector_count, token));
    };
    laws.weight.assign(static_cast<std::size_t>(extent) + 1, 1.0);
    laws.h.assign(static_cast<std::size_t>(extent) + 1, 1.0);
    for (int i = 1; i <= extent; ++i) {
        const int draft = row_drafts[i - 1];
        const double qd = q_at(i - 1, draft);
        const double p =
            qd > 0.0
                ? std::min(
                      laws.weight[static_cast<std::size_t>(i - 1)] *
                          probability_of(laws.columns[static_cast<std::size_t>(i - 1)], draft) / qd,
                      1.0)
                : 0.0;
        laws.weight[static_cast<std::size_t>(i)] = p;
        double h                                 = p;
        if (i < extent) {
            // Tokens outside the p' support contribute max(-q, 0) = 0.
            const Distribution& column = laws.columns[static_cast<std::size_t>(i)];
            double z                   = 0.0;
            for (std::size_t index = 0; index < column.ids.size(); ++index) {
                z += std::max(p * column.probabilities[index] - q_at(i, column.ids[index]), 0.0);
            }
            const double denominator = z + 1.0 - p;
            h                        = denominator > 0.0 ? z / denominator : 1.0;
        }
        laws.h[static_cast<std::size_t>(i)] = h;
    }
    return laws;
}

// tau = max{i : eta_i < h_i}; eta_i is keyed by the round's first position with sub-key i.
int block_tau(const BlockLaws& laws, unsigned long long seed, int length) {
    int tau = 0;
    for (int i = 1; i < static_cast<int>(laws.h.size()); ++i) {
        const float eta =
            uniform(seed, length + 1, ninfer::ops::kSamplePurposeSpeculativeBlockAccept,
                    static_cast<unsigned int>(i));
        if (static_cast<double>(eta) < laws.h[static_cast<std::size_t>(i)]) { tau = i; }
    }
    return tau;
}

std::vector<hip_bfloat16> make_logits(int token_domain, int columns, int batch) {
    std::vector<hip_bfloat16> result(static_cast<std::size_t>(token_domain) * columns * batch);
    for (int row = 0; row < batch; ++row) {
        for (int column = 0; column < columns; ++column) {
            auto* base =
                result.data() + (static_cast<std::size_t>(row) * columns + column) * token_domain;
            for (int token = 0; token < token_domain; ++token) {
                const unsigned hash = static_cast<unsigned>(token) * 2654435761U ^
                                      static_cast<unsigned>(column * 1009 + row * 9176);
                base[token]         = bf16(-10.0F + static_cast<float>(hash % 2048U) / 512.0F);
            }
            for (int rank = 0; rank < std::min(24, token_domain); ++rank) {
                const int token = (13 + rank * 7919 + column * 101 + row * 503) % token_domain;
                base[token]     = bf16(9.0F - static_cast<float>(rank) * 0.1875F);
            }
        }
    }
    return result;
}

struct ChainExpected {
    std::vector<int> lengths;
    std::vector<int> anchors;
    std::vector<int> licensed;
    std::vector<int> licensed_counts;
    std::vector<int> accepted;
    std::vector<int> counts;
};

ChainExpected chain_oracle(int token_domain, int k, int batch, const std::vector<int>& targets,
                           const std::vector<hip_bfloat16>& logits, const std::vector<int>& drafts,
                           const std::vector<int>& extents, const std::vector<int>& initial_lengths,
                           const std::vector<ninfer::ops::SamplingConfig>& configs,
                           const std::vector<bool>& has_counts,
                           const std::vector<int>& initial_counts,
                           const std::vector<int>* selector_ids,
                           const std::vector<float>* selector_q, int selector_count) {
    const int columns = k + 1;
    ChainExpected result{initial_lengths,
                         std::vector<int>(static_cast<std::size_t>(batch), -1),
                         std::vector<int>(static_cast<std::size_t>(columns) * batch, 0),
                         std::vector<int>(static_cast<std::size_t>(batch)),
                         std::vector<int>(static_cast<std::size_t>(batch)),
                         initial_counts};
    for (int row = 0; row < batch; ++row) {
        const int extent   = std::clamp(extents[static_cast<std::size_t>(row)], 0, k);
        const auto& config = configs[static_cast<std::size_t>(row)];
        // The recorded selector q is the proposal law under every target sampler, p-less included.
        const auto* effective_ids = selector_ids;
        const auto* effective_q   = selector_q;
        int accepted              = 0;
        int terminal              = 0;
        if (!(config.temperature > 0.0F)) {
            while (accepted < extent &&
                   targets[static_cast<std::size_t>(row * columns + accepted)] ==
                       drafts[static_cast<std::size_t>(row * k + accepted)]) {
                ++accepted;
            }
            terminal = targets[static_cast<std::size_t>(row * columns + accepted)];
        } else {
            std::vector<int> row_counts(
                result.counts.begin() + static_cast<std::size_t>(row) * token_domain,
                result.counts.begin() + static_cast<std::size_t>(row + 1) * token_domain);
            const std::vector<int>* counts =
                has_counts[static_cast<std::size_t>(row)] ? &row_counts : nullptr;
            const int length = initial_lengths[static_cast<std::size_t>(row)];
            if (config.p_less != 0) {
                const int selector_row_base = selector_count * row * k;
                const BlockLaws laws        = p_less_block_laws(
                    logits.data() + static_cast<std::size_t>(row) * columns * token_domain,
                    token_domain, extent, config, drafts.data() + static_cast<std::size_t>(row) * k,
                    effective_ids, effective_q, selector_row_base, selector_count);
                accepted                   = block_tau(laws, config.seed, length);
                const Distribution& column = laws.columns[static_cast<std::size_t>(accepted)];
                if (accepted < extent) {
                    terminal =
                        pick_residual(column, effective_ids, effective_q,
                                      selector_row_base + selector_count * accepted, selector_count,
                                      drafts[static_cast<std::size_t>(row * k + accepted)],
                                      uniform(config.seed, length + accepted + 1,
                                              ninfer::ops::kSamplePurposeSpeculativeCorrection),
                                      laws.weight[static_cast<std::size_t>(accepted)]);
                } else {
                    terminal = pick(column, -1,
                                    uniform(config.seed, length + extent + 1,
                                            ninfer::ops::kSamplePurposeSpeculativeBonus));
                }
            }
            for (int column = 0; config.p_less == 0 && column <= extent; ++column) {
                std::vector<int> overlay;
                overlay.reserve(static_cast<std::size_t>(column));
                for (int index = 0; index < column; ++index) {
                    overlay.push_back(drafts[static_cast<std::size_t>(row * k + index)]);
                }
                const Distribution distribution = distribution_oracle(
                    logits.data() +
                        (static_cast<std::size_t>(row) * columns + column) * token_domain,
                    token_domain, column_config(config, column), counts, overlay);
                if (column < extent) {
                    const int draft = drafts[static_cast<std::size_t>(row * k + column)];
                    double p        = 0.0;
                    for (std::size_t index = 0; index < distribution.ids.size(); ++index) {
                        if (distribution.ids[index] == draft) {
                            p = distribution.probabilities[index];
                            break;
                        }
                    }
                    const int selector_base = selector_count * (column + row * k);
                    const float q    = q_probability(effective_ids, effective_q, selector_base,
                                                     selector_count, draft);
                    const float draw = uniform(
                        config.seed, initial_lengths[static_cast<std::size_t>(row)] + column + 1,
                        ninfer::ops::kSamplePurposeSpeculativeAccept);
                    if (q > 0.0F && draw < std::min(1.0, p / static_cast<double>(q))) {
                        accepted = column + 1;
                        continue;
                    }
                    terminal = pick_residual(
                        distribution, effective_ids, effective_q, selector_base, selector_count,
                        draft,
                        uniform(config.seed,
                                initial_lengths[static_cast<std::size_t>(row)] + column + 1,
                                ninfer::ops::kSamplePurposeSpeculativeCorrection));
                    break;
                }
                terminal = pick(distribution, -1,
                                uniform(config.seed,
                                        initial_lengths[static_cast<std::size_t>(row)] + extent + 1,
                                        ninfer::ops::kSamplePurposeSpeculativeBonus));
            }
        }
        for (int index = 0; index < accepted; ++index) {
            result.licensed[static_cast<std::size_t>(row * columns + index)] =
                drafts[static_cast<std::size_t>(row * k + index)];
        }
        result.licensed[static_cast<std::size_t>(row * columns + accepted)] = terminal;
        result.licensed_counts[static_cast<std::size_t>(row)]               = accepted + 1;
        result.accepted[static_cast<std::size_t>(row)]                      = accepted;
        result.anchors[static_cast<std::size_t>(row)]                       = terminal;
        result.lengths[static_cast<std::size_t>(row)] += accepted + 1;
        if (config.temperature > 0.0F && has_counts[static_cast<std::size_t>(row)]) {
            for (int index = 0; index <= accepted; ++index) {
                ++result.counts[static_cast<std::size_t>(row) * token_domain +
                                result.licensed[static_cast<std::size_t>(row * columns + index)]];
            }
        }
    }
    return result;
}

void run_chain_case(ninfer::DeviceContext& device, int token_domain, int k, int batch,
                    bool all_greedy, bool use_selector, bool p_less = false) {
    const int columns = k + 1;
    std::vector<int> drafts(static_cast<std::size_t>(k) * batch);
    std::vector<int> targets(static_cast<std::size_t>(columns) * batch);
    std::vector<int> extents(static_cast<std::size_t>(batch));
    std::vector<int> lengths(static_cast<std::size_t>(batch));
    for (int row = 0; row < batch; ++row) {
        extents[static_cast<std::size_t>(row)] = (row * 3 + k) % (k + 1);
        lengths[static_cast<std::size_t>(row)] = 1000 + row * 37;
        const int desired_accept = extents[static_cast<std::size_t>(row)] == 0
                                       ? 0
                                       : row % (extents[static_cast<std::size_t>(row)] + 1);
        for (int column = 0; column < k; ++column) {
            const int draft = (17 + row * 503 + column * 7919) % token_domain;
            drafts[static_cast<std::size_t>(row * k + column)] = draft;
            targets[static_cast<std::size_t>(row * columns + column)] =
                column < desired_accept ? draft : (draft + 1) % token_domain;
        }
        targets[static_cast<std::size_t>(row * columns + k)] = (101 + row * 31 + k) % token_domain;
    }
    std::vector<hip_bfloat16> logits = make_logits(token_domain, columns, batch);
    if (p_less) {
        std::fill(logits.begin(), logits.end(), bf16(-24));
        for (int row = 0; row < batch; ++row) {
            extents[row] = k;
            for (int column = 0; column < columns; ++column) {
                for (int token = 0; token < 64; ++token)
                    logits[(row * columns + column) * token_domain + token] = bf16(0);
                if (column < k) drafts[row * k + column] = 7 + column;
            }
        }
    }
    std::vector<int> counts(static_cast<std::size_t>(batch) * token_domain, 0);
    std::vector<ninfer::ops::SamplingConfig> configs(static_cast<std::size_t>(batch));
    std::vector<bool> has_counts(static_cast<std::size_t>(batch));
    const int mask_words = (token_domain + 31) / 32;
    std::vector<std::uint32_t> masks(static_cast<std::size_t>(batch) * columns * mask_words,
                                     ~std::uint32_t{0});
    ninfer::DeviceBuffer d_masks(masks.size() * sizeof(masks[0]));
    for (int row = 0; row < batch; ++row) {
        auto& config             = configs[static_cast<std::size_t>(row)];
        const bool greedy        = all_greedy || (batch > 1 && row == 0);
        config.temperature       = greedy ? 0.0F : 0.65F + 0.2F * static_cast<float>(row % 3);
        const int top_k_cases[]  = {1, 3, 7, 19, 20, 0, -1, 11};
        config.top_k             = top_k_cases[row % 8];
        config.top_p             = row % 3 == 0 ? 0.73F : 1.0F;
        config.min_p             = row % 4 == 2 ? 0.12F : 0.0F;
        config.presence_penalty  = greedy ? 0.0F : (row % 2 == 0 ? 0.4F : 0.0F);
        config.frequency_penalty = greedy ? 0.0F : (row % 3 == 1 ? 0.2F : 0.0F);
        config.seed = 0x123456789abcdef0ULL + static_cast<unsigned long long>(row * 1009 + k);
        if (p_less) {
            config.p_less                      = 1;
            config.temperature                 = 1;
            config.top_k                       = 1;
            config.top_p                       = 0.01F;
            config.min_p                       = 0.99F;
            config.presence_penalty            = 100;
            config.frequency_penalty           = 100;
            config.suppressed_token_count      = 2;
            config.suppressed_tokens[0]        = 0;
            config.suppressed_tokens[1]        = 1;
            config.typical_exclude             = 2;
            config.allowed_token_column_stride = mask_words;
            config.allowed_token_words         = masks.data() + row * columns * mask_words;
            for (int column = 0; column < columns; ++column)
                for (int token = 0; token < 64; ++token)
                    if ((token + column + row) % 5 == 0)
                        masks[(row * columns + column) * mask_words + token / 32] &=
                            ~(1U << (token % 32));
        }
        has_counts[static_cast<std::size_t>(row)] = !greedy && row % 4 != 3;
        counts[static_cast<std::size_t>(row) * token_domain + (13 + row * 503) % token_domain] =
            2 + row;
    }
    std::vector<int> selector_ids;
    std::vector<float> selector_q;
    constexpr int selector_count = 4;
    if (use_selector) {
        selector_ids.assign(static_cast<std::size_t>(selector_count) * k * batch, 0);
        selector_q.assign(selector_ids.size(), 0.0F);
        for (int row = 0; row < batch; ++row) {
            for (int column = 0; column < k; ++column) {
                const int base = selector_count * (column + row * k);
                selector_ids[static_cast<std::size_t>(base)] =
                    drafts[static_cast<std::size_t>(row * k + column)];
                selector_q[static_cast<std::size_t>(base)] =
                    row % 3 == 1 ? 0.0F : (row % 3 == 2 ? 0.35F : 0.8F);
                for (int candidate = 1; candidate < selector_count; ++candidate) {
                    selector_ids[static_cast<std::size_t>(base + candidate)] =
                        (13 + candidate * 7919 + column * 101 + row * 503) % token_domain;
                    selector_q[static_cast<std::size_t>(base + candidate)] =
                        0.2F / static_cast<float>(candidate);
                }
            }
        }
    }

    if (p_less) {
        // Force block-verification outcomes the exact comparison must distinguish: row 0 accepts
        // one draft and row 2 two, so the correction samples max(0, p_tau p' - q) with p_tau < 1.
        // Row 1's recorded q(d0) = 0 under the selector (p_1 = 0) and row 3 keep their natural
        // outcome.
        for (int row = 0; row < std::min(batch, 3); row += 2) {
            auto& config         = configs[static_cast<std::size_t>(row)];
            const int want       = row == 0 ? 1 : 2;
            const BlockLaws laws = p_less_block_laws(
                logits.data() + static_cast<std::size_t>(row) * columns * token_domain,
                token_domain, extents[static_cast<std::size_t>(row)], config,
                drafts.data() + static_cast<std::size_t>(row) * k,
                use_selector ? &selector_ids : nullptr, use_selector ? &selector_q : nullptr,
                selector_count * row * k, selector_count);
            int attempts = 0;
            while (block_tau(laws, config.seed, lengths[static_cast<std::size_t>(row)]) != want) {
                ++config.seed;
                require(++attempts < 10000000, "no seed reaches the requested block length");
            }
        }
    }

    ninfer::DeviceBuffer d_targets(targets.size() * sizeof(int));
    ninfer::DeviceBuffer d_logits(logits.size() * sizeof(hip_bfloat16));
    ninfer::DeviceBuffer d_drafts(drafts.size() * sizeof(int));
    ninfer::DeviceBuffer d_extents(extents.size() * sizeof(int));
    ninfer::DeviceBuffer d_lengths(lengths.size() * sizeof(int));
    ninfer::DeviceBuffer d_anchors(static_cast<std::size_t>(batch) * sizeof(int));
    ninfer::DeviceBuffer d_licensed(static_cast<std::size_t>(columns) * batch * sizeof(int));
    ninfer::DeviceBuffer d_licensed_counts(static_cast<std::size_t>(batch) * sizeof(int));
    ninfer::DeviceBuffer d_accepted(static_cast<std::size_t>(batch) * sizeof(int));
    ninfer::DeviceBuffer d_counts(counts.size() * sizeof(int));
    ninfer::DeviceBuffer d_configs(configs.size() * sizeof(configs[0]));
    ninfer::DeviceBuffer d_selector_ids(use_selector ? selector_ids.size() * sizeof(int) : 1);
    ninfer::DeviceBuffer d_selector_q(use_selector ? selector_q.size() * sizeof(float) : 1);
    d_targets.copy_from_host(targets.data(), targets.size() * sizeof(int));
    d_logits.copy_from_host(logits.data(), logits.size() * sizeof(hip_bfloat16));
    d_drafts.copy_from_host(drafts.data(), drafts.size() * sizeof(int));
    d_extents.copy_from_host(extents.data(), extents.size() * sizeof(int));
    d_lengths.copy_from_host(lengths.data(), lengths.size() * sizeof(int));
    d_counts.copy_from_host(counts.data(), counts.size() * sizeof(int));
    if (use_selector) {
        d_selector_ids.copy_from_host(selector_ids.data(), selector_ids.size() * sizeof(int));
        d_selector_q.copy_from_host(selector_q.data(), selector_q.size() * sizeof(float));
    }
    for (int row = 0; row < batch; ++row) {
        configs[static_cast<std::size_t>(row)].token_counts =
            has_counts[static_cast<std::size_t>(row)]
                ? static_cast<int*>(d_counts.data()) + static_cast<std::size_t>(row) * token_domain
                : nullptr;
    }
    const ChainExpected expected =
        chain_oracle(token_domain, k, batch, targets, logits, drafts, extents, lengths, configs,
                     has_counts, counts, use_selector ? &selector_ids : nullptr,
                     use_selector ? &selector_q : nullptr, use_selector ? selector_count : 0);
    d_masks.copy_from_host(masks.data(), masks.size() * sizeof(masks[0]));
    for (int row = 0; p_less && row < batch; ++row)
        configs[row].allowed_token_words =
            static_cast<std::uint32_t*>(d_masks.data()) + row * columns * mask_words;
    d_configs.copy_from_host(configs.data(), configs.size() * sizeof(configs[0]));

    ninfer::Tensor target_tensor(d_targets.data(), ninfer::DType::I32, {columns, batch});
    ninfer::Tensor logits_tensor(d_logits.data(), ninfer::DType::BF16,
                                 {token_domain, columns, batch});
    ninfer::Tensor draft_tensor(d_drafts.data(), ninfer::DType::I32, {k, batch});
    ninfer::Tensor extent_tensor(d_extents.data(), ninfer::DType::I32, {batch});
    ninfer::Tensor length_tensor(d_lengths.data(), ninfer::DType::I32, {batch});
    ninfer::Tensor anchor_tensor(d_anchors.data(), ninfer::DType::I32, {batch});
    ninfer::Tensor licensed_tensor(d_licensed.data(), ninfer::DType::I32, {columns, batch});
    ninfer::Tensor licensed_count_tensor(d_licensed_counts.data(), ninfer::DType::I32, {batch});
    ninfer::Tensor accepted_tensor(d_accepted.data(), ninfer::DType::I32, {batch});
    ninfer::Tensor selector_ids_tensor;
    ninfer::Tensor selector_q_tensor;
    if (use_selector) {
        selector_ids_tensor =
            ninfer::Tensor(d_selector_ids.data(), ninfer::DType::I32, {selector_count, k, batch});
        selector_q_tensor =
            ninfer::Tensor(d_selector_q.data(), ninfer::DType::FP32, {selector_count, k, batch});
    }
    const std::size_t workspace_bytes =
        ninfer::ops::speculative_accept_greedy_drafts_workspace_capacity_bytes(token_domain, k, k,
                                                                               batch, batch);
    ninfer::DeviceArena workspace(std::max<std::size_t>(workspace_bytes, 1));
    ninfer::ops::speculative_accept_greedy_drafts(
        target_tensor, logits_tensor, draft_tensor, extent_tensor, length_tensor, anchor_tensor,
        licensed_tensor, licensed_count_tensor, accepted_tensor, token_domain,
        static_cast<const ninfer::ops::SamplingConfig*>(d_configs.data()), workspace, device.stream,
        use_selector ? &selector_ids_tensor : nullptr, use_selector ? &selector_q_tensor : nullptr);
    device.synchronize();

    const std::string label = "chain V=" + std::to_string(token_domain) +
                              " K=" + std::to_string(k) + " B=" + std::to_string(batch);
    require_equal(copy_from<int>(d_lengths, batch), expected.lengths, label + " lengths");
    require_equal(copy_from<int>(d_anchors, batch), expected.anchors, label + " anchors");
    require_equal(copy_from<int>(d_licensed, static_cast<std::size_t>(columns) * batch),
                  expected.licensed, label + " licensed tokens");
    require_equal(copy_from<int>(d_licensed_counts, batch), expected.licensed_counts,
                  label + " licensed counts");
    require_equal(copy_from<int>(d_accepted, batch), expected.accepted, label + " accepted");
    require_equal(copy_from<int>(d_counts, counts.size()), expected.counts,
                  label + " token counts");
    require_equal(copy_from<int>(d_targets, targets.size()), targets, label + " target immutable");
    require_equal(copy_from<int>(d_drafts, drafts.size()), drafts, label + " drafts immutable");
    require_equal(copy_from<int>(d_extents, extents.size()), extents, label + " extents immutable");
    require_equal(copy_from<std::uint16_t>(d_logits, logits.size()), bf16_bits(logits),
                  label + " logits immutable");
    if (use_selector) {
        require_equal(copy_from<int>(d_selector_ids, selector_ids.size()), selector_ids,
                      label + " selector ids immutable");
        require_equal(copy_from<float>(d_selector_q, selector_q.size()), selector_q,
                      label + " selector probabilities immutable");
    }
    require(workspace.used() == 0 && workspace.peak_used() == workspace_bytes,
            label + " workspace high-water differs from sizing query");
    if (p_less) {
        hipGraph_t graph          = nullptr;
        hipGraphExec_t executable = nullptr;
        HIP_CHECK(hipStreamBeginCapture(device.stream, hipStreamCaptureModeGlobal));
        ninfer::ops::speculative_accept_greedy_drafts(
            target_tensor, logits_tensor, draft_tensor, extent_tensor, length_tensor, anchor_tensor,
            licensed_tensor, licensed_count_tensor, accepted_tensor, token_domain,
            static_cast<const ninfer::ops::SamplingConfig*>(d_configs.data()), workspace,
            device.stream, use_selector ? &selector_ids_tensor : nullptr,
            use_selector ? &selector_q_tensor : nullptr);
        HIP_CHECK(hipStreamEndCapture(device.stream, &graph));
        HIP_CHECK(hipGraphInstantiate(&executable, graph, nullptr, nullptr, 0));
        for (int replay = 0; replay < 2; ++replay) {
            d_lengths.copy_from_host(lengths.data(), lengths.size() * sizeof(int));
            d_counts.copy_from_host(counts.data(), counts.size() * sizeof(int));
            HIP_CHECK(hipMemsetAsync(d_licensed.data(), 0xff, columns * batch * sizeof(int),
                                     device.stream));
            HIP_CHECK(hipMemsetAsync(d_accepted.data(), 0xff, batch * sizeof(int), device.stream));
            HIP_CHECK(hipGraphLaunch(executable, device.stream));
            device.synchronize();
            require_equal(copy_from<int>(d_licensed, columns * batch), expected.licensed,
                          label + " graph licensed");
            require_equal(copy_from<int>(d_accepted, batch), expected.accepted,
                          label + " graph accepted");
            require_equal(copy_from<int>(d_lengths, batch), expected.lengths,
                          label + " graph lengths");
            require_equal(copy_from<int>(d_counts, counts.size()), expected.counts,
                          label + " graph counts");
        }
        HIP_CHECK(hipGraphExecDestroy(executable));
        HIP_CHECK(hipGraphDestroy(graph));
    }
}

// Proposal calibration against an FP64 oracle: per (temperature g, hop j, row b), the overlap
// sum_c min(p'_j(c), q_j(c)^(T_d/T_g) / Z) between the independent p-less target law of verify
// column j and the recorded proposal re-tempered to T_g. Row 0 is p-less with T_d = 0.6, a typical
// exclusion on a supported candidate (hop 0 only), and extent 2 < K so hop 2 is undrafted; row 1
// draws greedy drafts (draft_temperature 0), so every row-1 entry is -1.
void run_p_less_proposal_calibration(ninfer::DeviceContext& device) {
    constexpr int physical_rows = kRealVocabulary;
    constexpr int token_domain  = 248077;
    constexpr int k             = 3;
    constexpr int columns       = k + 1;
    constexpr int batch         = 2;
    constexpr int cap           = 16;
    constexpr int active        = 48;
    constexpr int grid          = ninfer::ops::kPLessProposalCalibrationTemperatureCount;
    constexpr float draft_temp  = 0.6F;
    constexpr float temperature = 1.5F;
    std::vector<hip_bfloat16> logits(static_cast<std::size_t>(physical_rows) * columns * batch,
                                     bf16(-20.0F));
    for (int row = 0; row < batch; ++row) {
        for (int column = 0; column < columns; ++column) {
            auto* base = logits.data() + static_cast<std::size_t>(row * columns + column) *
                                             static_cast<std::size_t>(physical_rows);
            for (int token = 0; token < active; ++token) {
                base[token] =
                    bf16(2.0F * std::sin((0.6F + 0.11F * static_cast<float>(column + row)) *
                                         static_cast<float>(token)) +
                         0.05F * static_cast<float>(token));
            }
            for (int token = token_domain; token < physical_rows; ++token) {
                base[token] = bf16(100.0F);
            }
        }
    }
    // Selector layout [C,K,B]: (b, h) owns entries [(b*K + h)*C, +C). Distinct ids per hop.
    std::vector<int> ids(static_cast<std::size_t>(cap) * k * batch);
    std::vector<float> q(ids.size());
    for (int row = 0; row < batch; ++row) {
        for (int hop = 0; hop < k; ++hop) {
            const std::size_t base = static_cast<std::size_t>(row * k + hop) * cap;
            double total           = 0.0;
            for (int c = 0; c < cap; ++c) {
                ids[base + c]  = (c * (5 + 2 * hop) + 1 + hop + row) % active;
                const double w = std::exp(0.9 * std::cos(0.7 * static_cast<double>(c + hop + row)));
                q[base + c]    = static_cast<float>(w);
                total += w;
            }
            for (int c = 0; c < cap; ++c) {
                q[base + c] = static_cast<float>(static_cast<double>(q[base + c]) / total);
            }
        }
    }
    const auto column_law = [&](int row, int column, const ninfer::ops::SamplingConfig& config) {
        std::vector<double> values(static_cast<std::size_t>(token_domain));
        const auto* base = logits.data() + static_cast<std::size_t>(row * columns + column) *
                                               static_cast<std::size_t>(physical_rows);
        for (int token = 0; token < token_domain; ++token) {
            values[static_cast<std::size_t>(token)] = represented(base[token]);
        }
        return sampling_reference::p_less(values, config);
    };
    std::vector<ninfer::ops::SamplingConfig> configs(batch);
    for (auto& config : configs) {
        config.temperature = temperature;
        config.p_less      = 1;
    }
    configs[0].draft_temperature = draft_temp;
    {
        const auto support = column_law(0, 0, configs[0]);
        for (int c = 0; c < cap && configs[0].typical_exclude < 0; ++c) {
            if (std::find(support.ids.begin(), support.ids.end(),
                          ids[static_cast<std::size_t>(c)]) != support.ids.end()) {
                configs[0].typical_exclude = ids[static_cast<std::size_t>(c)];
            }
        }
        require(support.ids.size() >= 2 && configs[0].typical_exclude >= 0,
                "p-less proposal calibration: fixture lacks an excludable supported candidate");
    }
    const std::vector<int> extents{2, k};
    const std::vector<int> targets(static_cast<std::size_t>(columns) * batch, 0);
    const std::vector<int> drafts(static_cast<std::size_t>(k) * batch, 0);
    const std::vector<int> lengths(batch, 10);
    ninfer::DeviceBuffer d_targets(targets.size() * sizeof(int));
    ninfer::DeviceBuffer d_logits(logits.size() * sizeof(hip_bfloat16));
    ninfer::DeviceBuffer d_drafts(drafts.size() * sizeof(int));
    ninfer::DeviceBuffer d_extents(extents.size() * sizeof(int));
    ninfer::DeviceBuffer d_lengths(lengths.size() * sizeof(int));
    ninfer::DeviceBuffer d_anchors(static_cast<std::size_t>(batch) * sizeof(int));
    ninfer::DeviceBuffer d_licensed(static_cast<std::size_t>(columns) * batch * sizeof(int));
    ninfer::DeviceBuffer d_licensed_counts(static_cast<std::size_t>(batch) * sizeof(int));
    ninfer::DeviceBuffer d_accepted(static_cast<std::size_t>(batch) * sizeof(int));
    ninfer::DeviceBuffer d_configs(configs.size() * sizeof(ninfer::ops::SamplingConfig));
    ninfer::DeviceBuffer d_selector_ids(ids.size() * sizeof(int));
    ninfer::DeviceBuffer d_selector_q(q.size() * sizeof(float));
    ninfer::DeviceBuffer d_calibration(static_cast<std::size_t>(grid) * k * batch * sizeof(float));
    d_targets.copy_from_host(targets.data(), targets.size() * sizeof(int));
    d_logits.copy_from_host(logits.data(), logits.size() * sizeof(hip_bfloat16));
    d_drafts.copy_from_host(drafts.data(), drafts.size() * sizeof(int));
    d_extents.copy_from_host(extents.data(), extents.size() * sizeof(int));
    d_lengths.copy_from_host(lengths.data(), lengths.size() * sizeof(int));
    d_configs.copy_from_host(configs.data(), configs.size() * sizeof(ninfer::ops::SamplingConfig));
    d_selector_ids.copy_from_host(ids.data(), ids.size() * sizeof(int));
    d_selector_q.copy_from_host(q.data(), q.size() * sizeof(float));
    const std::vector<float> poison(static_cast<std::size_t>(grid) * k * batch,
                                    std::numeric_limits<float>::quiet_NaN());
    d_calibration.copy_from_host(poison.data(), poison.size() * sizeof(float));

    ninfer::Tensor target_tensor(d_targets.data(), ninfer::DType::I32, {columns, batch});
    ninfer::Tensor logits_tensor(d_logits.data(), ninfer::DType::BF16,
                                 {physical_rows, columns, batch});
    ninfer::Tensor draft_tensor(d_drafts.data(), ninfer::DType::I32, {k, batch});
    ninfer::Tensor extent_tensor(d_extents.data(), ninfer::DType::I32, {batch});
    ninfer::Tensor length_tensor(d_lengths.data(), ninfer::DType::I32, {batch});
    ninfer::Tensor anchor_tensor(d_anchors.data(), ninfer::DType::I32, {batch});
    ninfer::Tensor licensed_tensor(d_licensed.data(), ninfer::DType::I32, {columns, batch});
    ninfer::Tensor licensed_count_tensor(d_licensed_counts.data(), ninfer::DType::I32, {batch});
    ninfer::Tensor accepted_tensor(d_accepted.data(), ninfer::DType::I32, {batch});
    ninfer::Tensor selector_ids_tensor(d_selector_ids.data(), ninfer::DType::I32, {cap, k, batch});
    ninfer::Tensor selector_q_tensor(d_selector_q.data(), ninfer::DType::FP32, {cap, k, batch});
    ninfer::Tensor calibration_tensor(d_calibration.data(), ninfer::DType::FP32, {grid, k, batch});
    ninfer::DeviceArena workspace(std::max<std::size_t>(
        ninfer::ops::speculative_accept_greedy_drafts_workspace_capacity_bytes(token_domain, k, k,
                                                                               batch, batch),
        1));
    ninfer::ops::speculative_accept_greedy_drafts(
        target_tensor, logits_tensor, draft_tensor, extent_tensor, length_tensor, anchor_tensor,
        licensed_tensor, licensed_count_tensor, accepted_tensor, token_domain,
        static_cast<const ninfer::ops::SamplingConfig*>(d_configs.data()), workspace, device.stream,
        &selector_ids_tensor, &selector_q_tensor, &calibration_tensor);
    device.synchronize();
    const std::vector<float> got =
        copy_from<float>(d_calibration, static_cast<std::size_t>(grid) * k * batch);

    double worst = 0.0;
    for (int row = 0; row < batch; ++row) {
        for (int hop = 0; hop < k; ++hop) {
            const bool scored = row == 0 && hop < extents[static_cast<std::size_t>(row)];
            std::vector<double> target(cap, 0.0);
            if (scored) {
                const auto law = column_law(
                    row, hop, column_config(configs[static_cast<std::size_t>(row)], hop));
                for (int c = 0; c < cap; ++c) {
                    const int id  = ids[static_cast<std::size_t>(row * k + hop) * cap + c];
                    const auto it = std::find(law.ids.begin(), law.ids.end(), id);
                    if (it != law.ids.end()) {
                        target[static_cast<std::size_t>(c)] =
                            law.probabilities[static_cast<std::size_t>(it - law.ids.begin())];
                    }
                }
            }
            for (int g = 0; g < grid; ++g) {
                const float value =
                    got[(static_cast<std::size_t>(row) * k + static_cast<std::size_t>(hop)) * grid +
                        static_cast<std::size_t>(g)];
                double expected = -1.0;
                if (scored) {
                    const double ratio =
                        draft_temp /
                        ninfer::ops::kPLessProposalCalibrationTemperatures[static_cast<std::size_t>(
                            g)];
                    std::vector<double> retempered(cap);
                    double z = 0.0;
                    for (int c = 0; c < cap; ++c) {
                        retempered[static_cast<std::size_t>(c)] =
                            std::pow(static_cast<double>(
                                         q[static_cast<std::size_t>(row * k + hop) * cap + c]),
                                     ratio);
                        z += retempered[static_cast<std::size_t>(c)];
                    }
                    expected = 0.0;
                    for (int c = 0; c < cap; ++c) {
                        expected += std::min(target[static_cast<std::size_t>(c)],
                                             retempered[static_cast<std::size_t>(c)] / z);
                    }
                }
                const double error = std::abs(static_cast<double>(value) - expected);
                worst              = std::max(worst, std::isfinite(error) ? error : 1.0e30);
                require(error <= 2.0e-4, "p-less proposal calibration row " + std::to_string(row) +
                                             " hop " + std::to_string(hop) + " grid " +
                                             std::to_string(g) + " got " + std::to_string(value) +
                                             " expected " + std::to_string(expected));
            }
        }
    }
    std::cout << "p-less proposal calibration: max |error|=" << worst << '\n';
}

// P-less block verification law over a two-draft chain with fractional selector q. Drafts are
// drawn on the host from two different 16-way q that leave real target mass uncovered. The first
// two emitted tokens (the second drawn from the oracle when a round emits one) must follow
// p'_0 x p'_1, and the mean accepted length must match an independent FP64 enumeration of
// Algorithm 2 over every draft pair. `chained` gives every column the same law and runs a second
// round per trial from where the first ended with the same seed: its accepted length must also
// match, which fails if a round reuses a uniform the previous round's acceptance conditioned on.
// physical_rows > token_domain places high logits outside the domain.
void run_p_less_block_statistics(ninfer::DeviceContext& device, int physical_rows, int token_domain,
                                 bool chained, float temperature = 1.5F, int trials = 60000) {
    constexpr int active    = 64;
    constexpr int k         = 2;
    constexpr int columns   = k + 1;
    constexpr int cap       = 16;
    constexpr int batch     = 4;
    const int launches      = trials / batch;
    const std::string label = "p-less block verification V=" + std::to_string(token_domain) +
                              (chained ? " chained" : "") +
                              (temperature < 1.0e-30F ? " low-positive-temperature" : "");

    std::vector<hip_bfloat16> column_logits(static_cast<std::size_t>(physical_rows) * columns,
                                            bf16(-20.0F));
    for (int column = 0; column < columns; ++column) {
        const float shift = chained ? 0.0F : 0.13F * static_cast<float>(column);
        auto* base        = column_logits.data() + static_cast<std::size_t>(column) * physical_rows;
        for (int token = 0; token < active; ++token) {
            base[token] = temperature < 1.0e-30F
                              ? bf16(8.0F)
                              : bf16(2.5F * std::sin((0.7F + shift) * static_cast<float>(token)) +
                                     0.04F * static_cast<float>(token));
        }
        for (int token = token_domain; token < physical_rows; ++token) {
            base[token] = bf16(100.0F);
        }
    }
    std::vector<hip_bfloat16> logits;
    logits.reserve(column_logits.size() * batch);
    for (int row = 0; row < batch; ++row) {
        logits.insert(logits.end(), column_logits.begin(), column_logits.end());
    }

    ninfer::ops::SamplingConfig config{};
    config.temperature = temperature;
    config.p_less      = 1;
    std::vector<std::vector<double>> target(
        columns, std::vector<double>(static_cast<std::size_t>(token_domain)));
    std::vector<std::vector<int>> support(columns);
    for (int column = 0; column < columns; ++column) {
        std::vector<double> values(static_cast<std::size_t>(token_domain));
        for (int token = 0; token < token_domain; ++token) {
            values[static_cast<std::size_t>(token)] = represented(
                column_logits[static_cast<std::size_t>(column) * physical_rows + token]);
        }
        const auto law                            = sampling_reference::p_less(values, config);
        support[static_cast<std::size_t>(column)] = law.ids;
        for (std::size_t index = 0; index < law.ids.size(); ++index) {
            target[static_cast<std::size_t>(column)][static_cast<std::size_t>(law.ids[index])] =
                law.probabilities[index];
        }
    }

    // Hop-major selector q [cap, k] with distinct ids, replicated for every row.
    std::vector<int> ids(static_cast<std::size_t>(cap) * k);
    std::vector<float> q(static_cast<std::size_t>(cap) * k);
    for (int hop = 0; hop < k; ++hop) {
        double total = 0.0;
        std::vector<double> weights(cap);
        for (int c = 0; c < cap; ++c) {
            const int token                              = (c * (5 + 2 * hop) + 3 + hop) % active;
            ids[static_cast<std::size_t>(hop * cap + c)] = token;
            weights[static_cast<std::size_t>(c)]         = std::exp(
                static_cast<double>(represented(
                    column_logits[static_cast<std::size_t>(hop) * physical_rows + token])) /
                    0.8 +
                0.3 * std::cos(static_cast<double>(c + hop)));
            total += weights[static_cast<std::size_t>(c)];
        }
        for (int c = 0; c < cap; ++c) {
            q[static_cast<std::size_t>(hop * cap + c)] =
                static_cast<float>(weights[static_cast<std::size_t>(c)] / total);
        }
    }
    const auto q_at = [&](int hop, int token) {
        for (int c = 0; c < cap; ++c) {
            if (ids[static_cast<std::size_t>(hop * cap + c)] == token) {
                return static_cast<double>(q[static_cast<std::size_t>(hop * cap + c)]);
            }
        }
        return 0.0;
    };
    double uncovered = 1.0;
    for (int c = 0; c < cap; ++c) {
        uncovered -= target[1][static_cast<std::size_t>(ids[static_cast<std::size_t>(cap + c)])];
    }

    // Independent enumeration of Algorithm 2 for K=2 over every draft pair:
    // E[tau] = 2 h_2 + (1 - h_2) h_1 and E[tau^2] = 4 h_2 + (1 - h_2) h_1.
    double mean_tau  = 0.0;
    double mean_tau2 = 0.0;
    for (int c0 = 0; c0 < cap; ++c0) {
        for (int c1 = 0; c1 < cap; ++c1) {
            const int d0    = ids[static_cast<std::size_t>(c0)];
            const int d1    = ids[static_cast<std::size_t>(cap + c1)];
            const double w  = static_cast<double>(q[static_cast<std::size_t>(c0)]) *
                              static_cast<double>(q[static_cast<std::size_t>(cap + c1)]);
            const double p1 = std::min(target[0][static_cast<std::size_t>(d0)] / q_at(0, d0), 1.0);
            double z1       = 0.0;
            for (int x = 0; x < token_domain; ++x) {
                z1 += std::max(p1 * target[1][static_cast<std::size_t>(x)] - q_at(1, x), 0.0);
            }
            const double denominator = z1 + 1.0 - p1;
            const double h1          = denominator > 0.0 ? z1 / denominator : 1.0;
            const double h2 =
                std::min(p1 * target[1][static_cast<std::size_t>(d1)] / q_at(1, d1), 1.0);
            mean_tau += w * (2.0 * h2 + (1.0 - h2) * h1);
            mean_tau2 += w * (4.0 * h2 + (1.0 - h2) * h1);
        }
    }
    const double sd_tau = std::sqrt(std::max(mean_tau2 - mean_tau * mean_tau, 1e-12));

    std::vector<int> selector_ids;
    std::vector<float> selector_q;
    for (int row = 0; row < batch; ++row) {
        selector_ids.insert(selector_ids.end(), ids.begin(), ids.end());
        selector_q.insert(selector_q.end(), q.begin(), q.end());
    }
    const std::vector<int> targets(static_cast<std::size_t>(columns) * batch, 0);
    const std::vector<int> extents(batch, k);
    const std::vector<int> start_lengths(batch, 40);
    ninfer::DeviceBuffer d_targets(targets.size() * sizeof(int));
    ninfer::DeviceBuffer d_logits(logits.size() * sizeof(hip_bfloat16));
    ninfer::DeviceBuffer d_drafts(static_cast<std::size_t>(k) * batch * sizeof(int));
    ninfer::DeviceBuffer d_extents(extents.size() * sizeof(int));
    ninfer::DeviceBuffer d_lengths(static_cast<std::size_t>(batch) * sizeof(int));
    ninfer::DeviceBuffer d_anchors(static_cast<std::size_t>(batch) * sizeof(int));
    ninfer::DeviceBuffer d_licensed(static_cast<std::size_t>(columns) * batch * sizeof(int));
    ninfer::DeviceBuffer d_licensed_counts(static_cast<std::size_t>(batch) * sizeof(int));
    ninfer::DeviceBuffer d_accepted(static_cast<std::size_t>(batch) * sizeof(int));
    ninfer::DeviceBuffer d_configs(static_cast<std::size_t>(batch) * sizeof(config));
    ninfer::DeviceBuffer d_selector_ids(selector_ids.size() * sizeof(int));
    ninfer::DeviceBuffer d_selector_q(selector_q.size() * sizeof(float));
    d_targets.copy_from_host(targets.data(), targets.size() * sizeof(int));
    d_logits.copy_from_host(logits.data(), logits.size() * sizeof(hip_bfloat16));
    d_extents.copy_from_host(extents.data(), extents.size() * sizeof(int));
    d_selector_ids.copy_from_host(selector_ids.data(), selector_ids.size() * sizeof(int));
    d_selector_q.copy_from_host(selector_q.data(), selector_q.size() * sizeof(float));

    ninfer::Tensor target_tensor(d_targets.data(), ninfer::DType::I32, {columns, batch});
    ninfer::Tensor logits_tensor(d_logits.data(), ninfer::DType::BF16,
                                 {physical_rows, columns, batch});
    ninfer::Tensor draft_tensor(d_drafts.data(), ninfer::DType::I32, {k, batch});
    ninfer::Tensor extent_tensor(d_extents.data(), ninfer::DType::I32, {batch});
    ninfer::Tensor length_tensor(d_lengths.data(), ninfer::DType::I32, {batch});
    ninfer::Tensor anchor_tensor(d_anchors.data(), ninfer::DType::I32, {batch});
    ninfer::Tensor licensed_tensor(d_licensed.data(), ninfer::DType::I32, {columns, batch});
    ninfer::Tensor licensed_count_tensor(d_licensed_counts.data(), ninfer::DType::I32, {batch});
    ninfer::Tensor accepted_tensor(d_accepted.data(), ninfer::DType::I32, {batch});
    ninfer::Tensor selector_ids_tensor(d_selector_ids.data(), ninfer::DType::I32, {cap, k, batch});
    ninfer::Tensor selector_q_tensor(d_selector_q.data(), ninfer::DType::FP32, {cap, k, batch});
    ninfer::DeviceArena workspace(std::max<std::size_t>(
        ninfer::ops::speculative_accept_greedy_drafts_workspace_capacity_bytes(token_domain, k, k,
                                                                               batch, batch),
        1));

    std::mt19937_64 rng(chained ? 0xc4a1ULL : 0xb10cULL);
    std::discrete_distribution<int> draw0(q.begin(), q.begin() + cap);
    std::discrete_distribution<int> draw1(q.begin() + cap, q.end());
    std::discrete_distribution<int> oracle1(target[1].begin(), target[1].end());
    std::vector<int> licensed;
    std::vector<int> accepted;
    const auto run_round = [&]() {
        std::vector<int> drafts(static_cast<std::size_t>(k) * batch);
        for (int row = 0; row < batch; ++row) {
            drafts[static_cast<std::size_t>(row * k)] = ids[static_cast<std::size_t>(draw0(rng))];
            drafts[static_cast<std::size_t>(row * k + 1)] =
                ids[static_cast<std::size_t>(cap + draw1(rng))];
        }
        d_drafts.copy_from_host(drafts.data(), drafts.size() * sizeof(int));
        ninfer::ops::speculative_accept_greedy_drafts(
            target_tensor, logits_tensor, draft_tensor, extent_tensor, length_tensor, anchor_tensor,
            licensed_tensor, licensed_count_tensor, accepted_tensor, token_domain,
            static_cast<const ninfer::ops::SamplingConfig*>(d_configs.data()), workspace,
            device.stream, &selector_ids_tensor, &selector_q_tensor);
        device.synchronize();
        licensed = copy_from<int>(d_licensed, static_cast<std::size_t>(columns) * batch);
        accepted = copy_from<int>(d_accepted, batch);
    };

    const auto top3 = [&](int column) {
        std::vector<int> order = support[static_cast<std::size_t>(column)];
        std::sort(order.begin(), order.end(), [&](int a, int b) {
            return target[static_cast<std::size_t>(column)][static_cast<std::size_t>(a)] >
                   target[static_cast<std::size_t>(column)][static_cast<std::size_t>(b)];
        });
        order.resize(std::min<std::size_t>(3, order.size()));
        return order;
    };
    const std::vector<int> top0 = top3(0);
    const std::vector<int> top1 = top3(1);
    const auto bucket           = [](const std::vector<int>& top, int token) {
        const auto it = std::find(top.begin(), top.end(), token);
        return it == top.end() ? static_cast<int>(top.size()) : static_cast<int>(it - top.begin());
    };
    const int b0 = static_cast<int>(top0.size()) + 1;
    const int b1 = static_cast<int>(top1.size()) + 1;
    std::vector<long long> observed(static_cast<std::size_t>(b0 * b1), 0);
    std::vector<long long> marginal0(static_cast<std::size_t>(token_domain), 0);
    std::vector<long long> marginal1(static_cast<std::size_t>(token_domain), 0);
    double tau_sum        = 0.0;
    double second_tau_sum = 0.0;
    std::vector<ninfer::ops::SamplingConfig> configs(batch, config);
    for (int launch = 0; launch < launches; ++launch) {
        for (int row = 0; row < batch; ++row) {
            configs[static_cast<std::size_t>(row)].seed =
                0x9e3779b97f4a7c15ULL * static_cast<unsigned long long>(launch * batch + row + 1);
        }
        d_configs.copy_from_host(configs.data(), configs.size() * sizeof(config));
        d_lengths.copy_from_host(start_lengths.data(), start_lengths.size() * sizeof(int));
        run_round();
        std::vector<int> first(static_cast<std::size_t>(batch) * 2);
        for (int row = 0; row < batch; ++row) {
            const int a = accepted[static_cast<std::size_t>(row)];
            require(a >= 0 && a <= k, label + ": accepted length out of range");
            tau_sum += a;
            first[static_cast<std::size_t>(row * 2)] =
                licensed[static_cast<std::size_t>(row * columns)];
            first[static_cast<std::size_t>(row * 2 + 1)] =
                a >= 1 ? licensed[static_cast<std::size_t>(row * columns + 1)] : oracle1(rng);
        }
        if (chained) {
            // Lengths now hold where each row's first round ended; same seeds, fresh drafts.
            run_round();
            for (int row = 0; row < batch; ++row) {
                second_tau_sum += accepted[static_cast<std::size_t>(row)];
            }
        }
        for (int row = 0; row < batch; ++row) {
            const int t0 = first[static_cast<std::size_t>(row * 2)];
            const int t1 = first[static_cast<std::size_t>(row * 2 + 1)];
            require(t0 >= 0 && t0 < token_domain && t1 >= 0 && t1 < token_domain,
                    label + ": emitted token out of range");
            require(target[0][static_cast<std::size_t>(t0)] > 0.0 &&
                        target[1][static_cast<std::size_t>(t1)] > 0.0,
                    label + ": emitted token outside the p-less support");
            ++marginal0[static_cast<std::size_t>(t0)];
            ++marginal1[static_cast<std::size_t>(t1)];
            ++observed[static_cast<std::size_t>(bucket(top0, t0) * b1 + bucket(top1, t1))];
        }
    }

    const auto chi2_marginal = [&](const std::vector<long long>& counts, int column, double& dof) {
        double chi2 = 0.0;
        for (const int token : support[static_cast<std::size_t>(column)]) {
            const double expected =
                target[static_cast<std::size_t>(column)][static_cast<std::size_t>(token)] * trials;
            const double diff =
                static_cast<double>(counts[static_cast<std::size_t>(token)]) - expected;
            chi2 += diff * diff / expected;
        }
        dof = std::max(1.0,
                       static_cast<double>(support[static_cast<std::size_t>(column)].size()) - 1.0);
        return chi2;
    };
    double dof0       = 0.0;
    double dof1       = 0.0;
    const double chi0 = chi2_marginal(marginal0, 0, dof0);
    const double chi1 = chi2_marginal(marginal1, 1, dof1);
    double chi_joint  = 0.0;
    for (int a = 0; a < b0; ++a) {
        double pa = 0.0;
        for (const int token : support[0]) {
            if (bucket(top0, token) == a) { pa += target[0][static_cast<std::size_t>(token)]; }
        }
        for (int b = 0; b < b1; ++b) {
            double pb = 0.0;
            for (const int token : support[1]) {
                if (bucket(top1, token) == b) { pb += target[1][static_cast<std::size_t>(token)]; }
            }
            const double expected = pa * pb * trials;
            if (!(expected > 0.0)) { continue; }
            const double diff =
                static_cast<double>(observed[static_cast<std::size_t>(a * b1 + b)]) - expected;
            chi_joint += diff * diff / expected;
        }
    }
    const double dof_joint     = static_cast<double>(b0 * b1 - 1);
    const auto limit           = [](double dof) { return dof + 6.0 * std::sqrt(2.0 * dof); };
    const double tau_tolerance = 5.0 * sd_tau / std::sqrt(static_cast<double>(trials));
    const double mean_observed = tau_sum / trials;
    const double second_mean   = second_tau_sum / trials;
    std::cout << label << ": chi2 t0=" << chi0 << " (" << limit(dof0) << ") t1=" << chi1 << " ("
              << limit(dof1) << ") joint=" << chi_joint << " (" << limit(dof_joint)
              << ") E[tau]=" << mean_observed
              << (chained ? " round2=" + std::to_string(second_mean) : std::string())
              << " exact=" << mean_tau << " tol=" << tau_tolerance << " uncovered=" << uncovered
              << '\n';
    require(uncovered > 0.05, label + ": selector q must leave target mass uncovered");
    require(chi0 < limit(dof0) && chi1 < limit(dof1) && chi_joint < limit(dof_joint),
            label + ": emitted tokens do not follow the p-less target law");
    require(std::abs(mean_observed - mean_tau) < tau_tolerance &&
                (!chained || std::abs(second_mean - mean_tau) < tau_tolerance),
            label + ": accepted length differs from the Algorithm 2 enumeration");
}

// Regression: this RNG key produces exactly zero. With target law concentrated on token 0
// and a one-hot proposal on token 1, no draft can be accepted, including when eta == 0.
// Expected publication follows the independent FP64 target support, not the block-tau helper.
void qualify_zero_acceptance(ninfer::DeviceContext& device) {
    constexpr unsigned long long seed = 13539153ULL;
    require(uniform(seed, 1, ninfer::ops::kSamplePurposeSpeculativeBlockAccept, 1) == 0.0F,
            "zero-acceptance fixture does not reach the RNG endpoint");
    for (const int tokens : {64, kRealVocabulary}) {
        for (const int k : {1, 7}) {
            const int columns = k + 1;
            std::vector<hip_bfloat16> logits(static_cast<std::size_t>(tokens) * columns, bf16(-24));
            for (int column = 0; column < columns; ++column) logits[column * tokens] = bf16(10);
            ninfer::ops::SamplingConfig config{};
            config.temperature = 1.5F;
            config.p_less      = 1;
            config.seed        = seed;
            std::vector<double> represented_logits(static_cast<std::size_t>(tokens), -24.0);
            represented_logits[0] = 10.0;
            const auto law        = sampling_reference::p_less(represented_logits, config);
            require(law.ids == std::vector<int>{0} && law.probabilities == std::vector<double>{1.0},
                    "zero-acceptance fixture must have a singleton independent target law");
            const std::vector<int> targets(static_cast<std::size_t>(columns), 0);
            const std::vector<int> drafts(static_cast<std::size_t>(k), 1);
            const std::vector<float> proposal(static_cast<std::size_t>(k), 1.0F);
            ninfer::DeviceBuffer d_logits(logits.size() * sizeof(hip_bfloat16));
            ninfer::DeviceBuffer d_targets(targets.size() * sizeof(int));
            ninfer::DeviceBuffer d_drafts(drafts.size() * sizeof(int));
            ninfer::DeviceBuffer d_selector_ids(drafts.size() * sizeof(int));
            ninfer::DeviceBuffer d_proposal(proposal.size() * sizeof(float));
            ninfer::DeviceBuffer d_config(sizeof(config));
            ninfer::DeviceBuffer d_extent(sizeof(int)), d_length(sizeof(int)),
                d_anchor(sizeof(int));
            ninfer::DeviceBuffer d_licensed(targets.size() * sizeof(int));
            ninfer::DeviceBuffer d_count(sizeof(int)), d_accepted(sizeof(int));
            d_logits.copy_from_host(logits.data(), logits.size() * sizeof(hip_bfloat16));
            d_targets.copy_from_host(targets.data(), targets.size() * sizeof(int));
            d_drafts.copy_from_host(drafts.data(), drafts.size() * sizeof(int));
            d_selector_ids.copy_from_host(drafts.data(), drafts.size() * sizeof(int));
            d_proposal.copy_from_host(proposal.data(), proposal.size() * sizeof(float));
            d_config.copy_from_host(&config, sizeof(config));
            d_extent.copy_from_host(&k, sizeof(k));
            ninfer::Tensor target(d_targets.data(), ninfer::DType::I32, {columns, 1});
            ninfer::Tensor logit(d_logits.data(), ninfer::DType::BF16, {tokens, columns, 1});
            ninfer::Tensor draft(d_drafts.data(), ninfer::DType::I32, {k, 1});
            ninfer::Tensor extent(d_extent.data(), ninfer::DType::I32, {1});
            ninfer::Tensor length(d_length.data(), ninfer::DType::I32, {1});
            ninfer::Tensor anchor(d_anchor.data(), ninfer::DType::I32, {1});
            ninfer::Tensor licensed(d_licensed.data(), ninfer::DType::I32, {columns, 1});
            ninfer::Tensor count(d_count.data(), ninfer::DType::I32, {1});
            ninfer::Tensor accepted(d_accepted.data(), ninfer::DType::I32, {1});
            ninfer::Tensor selector_ids(d_selector_ids.data(), ninfer::DType::I32, {1, k, 1});
            ninfer::Tensor selector_q(d_proposal.data(), ninfer::DType::FP32, {1, k, 1});
            const auto bytes =
                ninfer::ops::speculative_accept_greedy_drafts_workspace_capacity_bytes(tokens, k, k,
                                                                                       1, 1);
            ninfer::DeviceArena workspace(std::max<std::size_t>(bytes, 1));
            for (const bool selector : {false, true}) {
                const int initial_length = 0;
                d_length.copy_from_host(&initial_length, sizeof(initial_length));
                ninfer::ops::speculative_accept_greedy_drafts(
                    target, logit, draft, extent, length, anchor, licensed, count, accepted, tokens,
                    static_cast<const ninfer::ops::SamplingConfig*>(d_config.data()), workspace,
                    device.stream, selector ? &selector_ids : nullptr,
                    selector ? &selector_q : nullptr);
                device.synchronize();
                require_equal(copy_from<int>(d_accepted, 1), std::vector<int>{0},
                              "zero-probability draft acceptance");
                require_equal(copy_from<int>(d_count, 1), std::vector<int>{1},
                              "zero-acceptance licensed count");
                require_equal(copy_from<int>(d_licensed, 1), law.ids,
                              "zero-acceptance emitted token");
                require_equal(copy_from<int>(d_anchor, 1), law.ids, "zero-acceptance anchor");
                require_equal(copy_from<int>(d_length, 1), std::vector<int>{1},
                              "zero-acceptance committed length");
            }
        }
    }
    std::cout << "p-less zero-probability acceptance: small/product vocabulary, K1/K7, "
                 "proposal laws PASS\n";
}

// Independent regression for the semantic empty-support Dirac law. At column0 p'=(.5,.5),
// column1's flat vocabulary is removed by the1/1024 floor and becomes Dirac0. With one-hot
// drafts0,0: h1=0, h2=.5. The fixed RNG draws reject both and the correction is exactly token1.
void qualify_empty_p_less_support(ninfer::DeviceContext& device) {
    constexpr int k = 2, columns = 3;
    constexpr unsigned long long seed = 8ULL;
    require(uniform(seed, 1, ninfer::ops::kSamplePurposeSpeculativeBlockAccept, 1) < 0.5F &&
                uniform(seed, 1, ninfer::ops::kSamplePurposeSpeculativeBlockAccept, 2) > 0.5F,
            "empty-support fixture RNG changed");
    for (const int tokens : {127, 2048, 248077}) {
        const bool excluded_singleton = tokens == 127;
        std::vector<hip_bfloat16> logits(static_cast<std::size_t>(tokens) * columns, bf16(-128));
        logits[0] = logits[1] = bf16(0);
        std::fill(logits.begin() + tokens, logits.begin() + 2 * tokens, bf16(0));
        logits[2 * tokens + 2] = bf16(10);
        if (excluded_singleton) {
            logits[0] = bf16(10);
            std::fill(logits.begin() + tokens, logits.begin() + 2 * tokens, bf16(-128));
            logits[tokens] = bf16(10);
        }
        ninfer::ops::SamplingConfig config{};
        config.temperature     = 1.5F;
        config.p_less          = 1;
        config.seed            = seed;
        config.typical_exclude = excluded_singleton ? 0 : -1;
        std::vector<double> first(static_cast<std::size_t>(tokens), -128.0);
        first[0] = first[1] = 0;
        if (excluded_singleton) first[0] = 10;
        const auto p0                = sampling_reference::p_less(first, config);
        auto later_config            = config;
        later_config.typical_exclude = -1;
        std::vector<double> second(static_cast<std::size_t>(tokens),
                                   excluded_singleton ? -128.0 : 0.0);
        if (excluded_singleton) second[0] = 10;
        const auto p1 = sampling_reference::p_less(second, later_config);
        const std::vector<int> expected_first =
            excluded_singleton ? std::vector<int>{1} : std::vector<int>{0, 1};
        const std::vector<double> expected_probs =
            excluded_singleton ? std::vector<double>{1.0} : std::vector<double>{0.5, 0.5};
        require(p0.ids == expected_first && p0.probabilities == expected_probs &&
                    p1.ids == std::vector<int>{0} && p1.probabilities == std::vector<double>{1.0},
                "empty-support fixture does not have the independent target laws");
        const std::vector<int> targets{excluded_singleton ? 1 : 0, 0, 2};
        const std::vector<int> drafts{excluded_singleton ? 1 : 0, 0};
        const std::vector<float> proposal{1.0F, 1.0F};
        ninfer::DeviceBuffer d_logits(logits.size() * sizeof(hip_bfloat16));
        ninfer::DeviceBuffer d_targets(targets.size() * sizeof(int)),
            d_drafts(drafts.size() * sizeof(int));
        ninfer::DeviceBuffer d_ids(drafts.size() * sizeof(int)),
            d_q(proposal.size() * sizeof(float));
        ninfer::DeviceBuffer d_config(sizeof(config)), d_extent(sizeof(int)), d_length(sizeof(int));
        ninfer::DeviceBuffer d_anchor(sizeof(int)), d_licensed(columns * sizeof(int));
        ninfer::DeviceBuffer d_count(sizeof(int)), d_accepted(sizeof(int));
        d_logits.copy_from_host(logits.data(), logits.size() * sizeof(hip_bfloat16));
        d_targets.copy_from_host(targets.data(), targets.size() * sizeof(int));
        d_drafts.copy_from_host(drafts.data(), drafts.size() * sizeof(int));
        d_ids.copy_from_host(drafts.data(), drafts.size() * sizeof(int));
        d_q.copy_from_host(proposal.data(), proposal.size() * sizeof(float));
        d_config.copy_from_host(&config, sizeof(config));
        d_extent.copy_from_host(&k, sizeof(k));
        ninfer::Tensor target(d_targets.data(), ninfer::DType::I32, {columns, 1});
        ninfer::Tensor logit(d_logits.data(), ninfer::DType::BF16, {tokens, columns, 1});
        ninfer::Tensor draft(d_drafts.data(), ninfer::DType::I32, {k, 1});
        ninfer::Tensor extent(d_extent.data(), ninfer::DType::I32, {1});
        ninfer::Tensor length(d_length.data(), ninfer::DType::I32, {1});
        ninfer::Tensor anchor(d_anchor.data(), ninfer::DType::I32, {1});
        ninfer::Tensor licensed(d_licensed.data(), ninfer::DType::I32, {columns, 1});
        ninfer::Tensor count(d_count.data(), ninfer::DType::I32, {1});
        ninfer::Tensor accepted(d_accepted.data(), ninfer::DType::I32, {1});
        ninfer::Tensor selector_ids(d_ids.data(), ninfer::DType::I32, {1, k, 1});
        ninfer::Tensor selector_q(d_q.data(), ninfer::DType::FP32, {1, k, 1});
        const auto bytes = ninfer::ops::speculative_accept_greedy_drafts_workspace_capacity_bytes(
            tokens, k, k, 1, 1);
        ninfer::DeviceArena workspace(std::max<std::size_t>(bytes, 1));
        for (const bool selector : {false, true}) {
            const int initial_length = 0;
            d_length.copy_from_host(&initial_length, sizeof(initial_length));
            ninfer::ops::speculative_accept_greedy_drafts(
                target, logit, draft, extent, length, anchor, licensed, count, accepted, tokens,
                static_cast<const ninfer::ops::SamplingConfig*>(d_config.data()), workspace,
                device.stream, selector ? &selector_ids : nullptr,
                selector ? &selector_q : nullptr);
            device.synchronize();
            const int expected_accept = excluded_singleton ? 2 : 0;
            const int expected_count  = expected_accept + 1;
            const std::vector<int> expected_ids =
                excluded_singleton ? std::vector<int>{1, 0, 2} : std::vector<int>{1};
            require_equal(copy_from<int>(d_accepted, 1), std::vector<int>{expected_accept},
                          "empty-support Dirac accepted length");
            require_equal(copy_from<int>(d_count, 1), std::vector<int>{expected_count},
                          "empty-support Dirac licensed count");
            require_equal(copy_from<int>(d_licensed, expected_count), expected_ids,
                          "empty-support Dirac licensed IDs");
            require_equal(copy_from<int>(d_anchor, 1), std::vector<int>{expected_ids.back()},
                          "empty-support Dirac anchor");
            require_equal(copy_from<int>(d_length, 1), std::vector<int>{expected_count},
                          "empty-support Dirac committed length");
        }
    }
    std::cout << "p-less empty-support Dirac verification: V127/V2048/V248077, implicit/explicit q "
                 "PASS\n";
}

void qualify_nonfinite_status(ninfer::DeviceContext& device) {
    constexpr int k = 2, columns = k + 1, batch = 3;
    for (const int tokens : {127, kRealVocabulary}) {
        const int physical = tokens + 3, words = (tokens + 31) / 32;
        const std::vector<int> extents{0, 1, 2}, lengths(batch, 40);
        std::vector<int> targets(columns * batch, 3), drafts(k * batch, 3);
        std::vector<std::uint32_t> masks(words, 0U);
        masks[0] = 1U | (1U << 3) | (1U << 7);
        std::vector<hip_bfloat16> finite(static_cast<std::size_t>(physical) * columns * batch,
                                         bf16(NAN));
        for (int row = 0; row < batch; ++row)
            for (int column = 0; column <= extents[row]; ++column) {
                const auto base  = static_cast<std::size_t>(row * columns + column) * physical;
                finite[base + 3] = bf16(8.0F);
                finite[base + 7] = bf16(-24.0F);
            }
        ninfer::DeviceBuffer dl(finite.size() * sizeof(finite[0]));
        ninfer::DeviceBuffer dm(masks.size() * sizeof(masks[0]));
        ninfer::DeviceBuffer ds(columns * batch * sizeof(std::uint32_t));
        ninfer::DeviceBuffer dc(batch * sizeof(ninfer::ops::SamplingConfig));
        ninfer::DeviceBuffer dt(targets.size() * sizeof(int)), dd(drafts.size() * sizeof(int));
        ninfer::DeviceBuffer de(batch * sizeof(int)), dlength(batch * sizeof(int));
        ninfer::DeviceBuffer danchor(batch * sizeof(int)), dlicensed(columns * batch * sizeof(int));
        ninfer::DeviceBuffer dcounts(batch * sizeof(int)), daccepted(batch * sizeof(int));
        const std::vector<int> valid_columns{1, 2, 3};
        const std::vector<int> parents{-1, 0, 1, -1, 0, 1, -1, 0, 1};
        ninfer::DeviceBuffer dvalid(batch * sizeof(int)), dparents(parents.size() * sizeof(int));
        ninfer::DeviceBuffer dverify(targets.size() * sizeof(int));
        ninfer::DeviceBuffer daccepted_column(batch * sizeof(int)),
            dfold(columns * batch * sizeof(int));
        dvalid.copy_from_host(valid_columns.data(), dvalid.size());
        dparents.copy_from_host(parents.data(), dparents.size());
        dverify.copy_from_host(targets.data(), dverify.size());
        dm.copy_from_host(masks.data(), dm.size());
        dt.copy_from_host(targets.data(), dt.size());
        dd.copy_from_host(drafts.data(), dd.size());
        de.copy_from_host(extents.data(), de.size());
        ninfer::Tensor target(dt.data(), ninfer::DType::I32, {columns, batch});
        ninfer::Tensor logits(dl.data(), ninfer::DType::BF16, {physical, columns, batch});
        ninfer::Tensor draft(dd.data(), ninfer::DType::I32, {k, batch});
        ninfer::Tensor extent(de.data(), ninfer::DType::I32, {batch});
        ninfer::Tensor length(dlength.data(), ninfer::DType::I32, {batch});
        ninfer::Tensor anchor(danchor.data(), ninfer::DType::I32, {batch});
        ninfer::Tensor licensed(dlicensed.data(), ninfer::DType::I32, {columns, batch});
        ninfer::Tensor counts(dcounts.data(), ninfer::DType::I32, {batch});
        ninfer::Tensor accepted(daccepted.data(), ninfer::DType::I32, {batch});
        ninfer::Tensor valid(dvalid.data(), ninfer::DType::I32, {batch});
        ninfer::Tensor parent(dparents.data(), ninfer::DType::I32, {columns, batch});
        ninfer::Tensor verify(dverify.data(), ninfer::DType::I32, {columns, batch});
        ninfer::Tensor accepted_column(daccepted_column.data(), ninfer::DType::I32, {batch});
        ninfer::Tensor fold(dfold.data(), ninfer::DType::I32, {columns, batch});
        ninfer::DeviceArena workspace(
            std::max({std::size_t{1},
                      ninfer::ops::speculative_accept_greedy_drafts_workspace_capacity_bytes(
                          tokens, k, k, batch, batch),
                      ninfer::ops::speculative_accept_tree_drafts_workspace_capacity_bytes(
                          tokens, columns, columns, batch, batch)}));
        for (const bool p_less : {false, true}) {
            for (const bool tree : {false, true}) {
                std::vector<ninfer::ops::SamplingConfig> configs(batch);
                for (int row = 0; row < batch; ++row) {
                    configs[row].temperature            = 1.0F;
                    configs[row].p_less                 = p_less;
                    configs[row].top_k                  = 1;
                    configs[row].suppressed_token_count = 1;
                    configs[row].suppressed_tokens[0]   = 0;
                    configs[row].allowed_token_words    = static_cast<std::uint32_t*>(dm.data());
                    configs[row].nonfinite_logits =
                        static_cast<std::uint32_t*>(ds.data()) + row * columns;
                }
                dc.copy_from_host(configs.data(), dc.size());
                const auto launch = [&] {
                    if (tree) {
                        ninfer::ops::speculative_accept_tree_drafts(
                            target, logits, verify, parent, valid, extent, length, anchor, licensed,
                            counts, accepted, accepted_column, fold, tokens,
                            static_cast<const ninfer::ops::SamplingConfig*>(dc.data()), workspace,
                            device.stream);
                    } else {
                        ninfer::ops::speculative_accept_greedy_drafts(
                            target, logits, draft, extent, length, anchor, licensed, counts,
                            accepted, tokens,
                            static_cast<const ninfer::ops::SamplingConfig*>(dc.data()), workspace,
                            device.stream);
                    }
                };
                hipGraph_t graph          = nullptr;
                hipGraphExec_t executable = nullptr;
                HIP_CHECK(hipStreamBeginCapture(device.stream, hipStreamCaptureModeGlobal));
                launch();
                HIP_CHECK(hipStreamEndCapture(device.stream, &graph));
                HIP_CHECK(hipGraphInstantiate(&executable, graph, nullptr, nullptr, 0));
                for (const float poison : {NAN, INFINITY, -INFINITY}) {
                    for (const bool dirty : {true, false}) {
                        auto input = finite;
                        if (dirty)
                            for (int row = 0; row < batch; ++row) {
                                const auto base =
                                    static_cast<std::size_t>(row * columns + extents[row]) *
                                    physical;
                                input[base + 3] = input[base + 7] = bf16(poison);
                            }
                        dl.copy_from_host(input.data(), dl.size());
                        dlength.copy_from_host(lengths.data(), dlength.size());
                        HIP_CHECK(hipMemsetAsync(ds.data(), 0, ds.size(), device.stream));
                        HIP_CHECK(hipGraphLaunch(executable, device.stream));
                        device.synchronize();
                        const auto status = copy_from<std::uint32_t>(ds, columns * batch);
                        for (int row = 0; row < batch; ++row)
                            for (int column = 0; column < columns; ++column)
                                require(
                                    status[row * columns + column] ==
                                        static_cast<std::uint32_t>(dirty && column == extents[row]),
                                    "speculative nonfinite status mixed column/tail or stale "
                                    "replay");
                        const auto ids = copy_from<int>(dlicensed, columns * batch);
                        const auto n   = copy_from<int>(dcounts, batch);
                        for (int row = 0; row < batch; ++row) {
                            require(n[row] >= 1 && n[row] <= columns,
                                    "invalid reported licensed count");
                            for (int column = 0; column < n[row]; ++column)
                                require(ids[row * columns + column] >= 0 &&
                                            ids[row * columns + column] < tokens,
                                        "reported nonfinite speculative logits yielded unsafe id");
                        }
                    }
                }
                HIP_CHECK(hipGraphExecDestroy(executable));
                HIP_CHECK(hipGraphDestroy(graph));
            }
        }
    }
    std::cout
        << "speculative nonfinite status: chain/tree column offsets/inactive tails/replay PASS\n";
}

void qualify_prepare(ninfer::DeviceContext& device) {
    for (int k = 1; k <= 8; ++k) {
        for (int batch = 1; batch <= static_cast<int>(ninfer::kMaximumConcurrency); ++batch) {
            const int columns = k + 1;
            std::vector<int> anchors(static_cast<std::size_t>(batch));
            std::vector<int> positions(static_cast<std::size_t>(batch));
            std::vector<int> extents(static_cast<std::size_t>(batch));
            std::vector<int> drafts(static_cast<std::size_t>(k) * batch);
            std::vector<int> expected_ids(static_cast<std::size_t>(columns) * batch);
            std::vector<int> expected_positions(expected_ids.size());
            for (int row = 0; row < batch; ++row) {
                anchors[static_cast<std::size_t>(row)]   = 70000 + row;
                positions[static_cast<std::size_t>(row)] = 1000 + row * 37;
                extents[static_cast<std::size_t>(row)]   = (row * 3 + k) % (k + 1);
                for (int column = 0; column < k; ++column) {
                    drafts[static_cast<std::size_t>(row * k + column)] = 17 + row * 101 + column;
                }
                for (int column = 0; column < columns; ++column) {
                    expected_ids[static_cast<std::size_t>(row * columns + column)] =
                        column == 0 ? anchors[static_cast<std::size_t>(row)]
                                    : (column <= extents[static_cast<std::size_t>(row)]
                                           ? drafts[static_cast<std::size_t>(row * k + column - 1)]
                                           : anchors[static_cast<std::size_t>(row)]);
                    expected_positions[static_cast<std::size_t>(row * columns + column)] =
                        positions[static_cast<std::size_t>(row)] +
                        std::min(column, extents[static_cast<std::size_t>(row)]);
                }
            }
            ninfer::DeviceBuffer d_anchors(anchors.size() * sizeof(int));
            ninfer::DeviceBuffer d_positions(positions.size() * sizeof(int));
            ninfer::DeviceBuffer d_extents(extents.size() * sizeof(int));
            ninfer::DeviceBuffer d_drafts(drafts.size() * sizeof(int));
            ninfer::DeviceBuffer d_ids(expected_ids.size() * sizeof(int));
            ninfer::DeviceBuffer d_output_positions(expected_positions.size() * sizeof(int));
            d_anchors.copy_from_host(anchors.data(), anchors.size() * sizeof(int));
            d_positions.copy_from_host(positions.data(), positions.size() * sizeof(int));
            d_extents.copy_from_host(extents.data(), extents.size() * sizeof(int));
            d_drafts.copy_from_host(drafts.data(), drafts.size() * sizeof(int));
            ninfer::Tensor anchors_tensor(d_anchors.data(), ninfer::DType::I32, {batch});
            ninfer::Tensor positions_tensor(d_positions.data(), ninfer::DType::I32, {batch});
            ninfer::Tensor extents_tensor(d_extents.data(), ninfer::DType::I32, {batch});
            ninfer::Tensor drafts_tensor(d_drafts.data(), ninfer::DType::I32, {k, batch});
            ninfer::Tensor ids_tensor(d_ids.data(), ninfer::DType::I32, {columns, batch});
            ninfer::Tensor output_positions_tensor(d_output_positions.data(), ninfer::DType::I32,
                                                   {columns, batch});
            ninfer::ops::speculative_prepare_verify_inputs(
                anchors_tensor, drafts_tensor, positions_tensor, extents_tensor, ids_tensor,
                output_positions_tensor, device.stream);
            device.synchronize();
            require_equal(copy_from<int>(d_ids, expected_ids.size()), expected_ids,
                          "prepare ids K=" + std::to_string(k) + " B=" + std::to_string(batch));
            require_equal(
                copy_from<int>(d_output_positions, expected_positions.size()), expected_positions,
                "prepare positions K=" + std::to_string(k) + " B=" + std::to_string(batch));
            std::vector<int> sentinel(expected_ids.size(), 0x55555555);
            d_ids.copy_from_host_async(sentinel.data(), sentinel.size() * sizeof(int),
                                       device.stream);
            ninfer::ops::speculative_prepare_verify_ids(anchors_tensor, drafts_tensor,
                                                        extents_tensor, ids_tensor, device.stream);
            device.synchronize();
            require_equal(copy_from<int>(d_ids, expected_ids.size()), expected_ids,
                          "prepare ids-only K=" + std::to_string(k) +
                              " B=" + std::to_string(batch));
            require_equal(copy_from<int>(d_anchors, anchors.size()), anchors,
                          "prepare anchors immutable");
            require_equal(copy_from<int>(d_drafts, drafts.size()), drafts,
                          "prepare drafts immutable");
        }
    }
}

struct TreeExpected {
    std::vector<int> lengths;
    std::vector<int> anchors;
    std::vector<int> licensed;
    std::vector<int> licensed_counts;
    std::vector<int> accepted;
    std::vector<int> accepted_column;
    std::vector<int> path;
    std::vector<int> counts;
};

int child_for(const std::vector<int>& parents, const std::vector<int>& ids, int row, int width,
              int valid, int node, int token) {
    for (int child = node + 1; child < valid; ++child) {
        if (parents[static_cast<std::size_t>(row * width + child)] == node &&
            ids[static_cast<std::size_t>(row * width + child)] == token) {
            return child;
        }
    }
    return -1;
}

TreeExpected tree_oracle(int token_domain, int width, int batch, const std::vector<int>& targets,
                         const std::vector<hip_bfloat16>& logits, const std::vector<int>& ids,
                         const std::vector<int>& parents, const std::vector<int>& valid_columns,
                         const std::vector<int>& extents, const std::vector<int>& initial_lengths,
                         const std::vector<ninfer::ops::SamplingConfig>& configs,
                         const std::vector<bool>& has_counts,
                         const std::vector<int>& initial_counts) {
    TreeExpected result{initial_lengths,
                        std::vector<int>(batch, -1),
                        std::vector<int>(static_cast<std::size_t>(width) * batch, 0),
                        std::vector<int>(batch),
                        std::vector<int>(batch),
                        std::vector<int>(batch),
                        std::vector<int>(static_cast<std::size_t>(width) * batch, 0),
                        initial_counts};
    for (int row = 0; row < batch; ++row) {
        const int valid    = std::clamp(valid_columns[static_cast<std::size_t>(row)], 1, width);
        const int extent   = std::clamp(extents[static_cast<std::size_t>(row)], 0, width - 1);
        const auto& config = configs[static_cast<std::size_t>(row)];
        int node           = 0;
        int accepted       = 0;
        int terminal       = 0;
        result.path[static_cast<std::size_t>(row * width)] = 0;
        if (!(config.temperature > 0.0F)) {
            while (accepted < extent) {
                const int predicted = targets[static_cast<std::size_t>(row * width + node)];
                const int child     = child_for(parents, ids, row, width, valid, node, predicted);
                if (child < 0) {
                    terminal = predicted;
                    break;
                }
                result.licensed[static_cast<std::size_t>(row * width + accepted)] = predicted;
                node                                                              = child;
                ++accepted;
                result.path[static_cast<std::size_t>(row * width + accepted)] = node;
            }
            if (accepted == extent) {
                terminal = targets[static_cast<std::size_t>(row * width + node)];
            }
        } else {
            std::vector<int> row_counts(
                result.counts.begin() + static_cast<std::size_t>(row) * token_domain,
                result.counts.begin() + static_cast<std::size_t>(row + 1) * token_domain);
            const std::vector<int>* counts =
                has_counts[static_cast<std::size_t>(row)] ? &row_counts : nullptr;
            std::vector<int> overlay;
            for (int step = 0; step <= extent; ++step) {
                const Distribution distribution = distribution_oracle(
                    logits.data() + (static_cast<std::size_t>(row) * width + node) * token_domain,
                    token_domain, column_config(config, node), counts, overlay);
                const int sampled =
                    pick(distribution, -1,
                         uniform(config.seed,
                                 initial_lengths[static_cast<std::size_t>(row)] + accepted + 1,
                                 ninfer::ops::kSamplePurposeSpeculativeAccept));
                const int child =
                    step < extent ? child_for(parents, ids, row, width, valid, node, sampled) : -1;
                if (child >= 0) {
                    result.licensed[static_cast<std::size_t>(row * width + accepted)] = sampled;
                    overlay.push_back(sampled);
                    node = child;
                    ++accepted;
                    result.path[static_cast<std::size_t>(row * width + accepted)] = node;
                } else {
                    terminal = sampled;
                    break;
                }
            }
        }
        result.licensed[static_cast<std::size_t>(row * width + accepted)] = terminal;
        result.licensed_counts[static_cast<std::size_t>(row)]             = accepted + 1;
        result.accepted[static_cast<std::size_t>(row)]                    = accepted;
        result.accepted_column[static_cast<std::size_t>(row)]             = node;
        result.anchors[static_cast<std::size_t>(row)]                     = terminal;
        result.lengths[static_cast<std::size_t>(row)] += accepted + 1;
        if (config.temperature > 0.0F && has_counts[static_cast<std::size_t>(row)]) {
            for (int index = 0; index <= accepted; ++index) {
                ++result.counts[static_cast<std::size_t>(row) * token_domain +
                                result.licensed[static_cast<std::size_t>(row * width + index)]];
            }
        }
    }
    return result;
}

void run_tree_case(ninfer::DeviceContext& device, int token_domain, int width, int batch,
                   bool all_greedy, bool p_less = false) {
    std::vector<int> ids(static_cast<std::size_t>(width) * batch);
    std::vector<int> parents(ids.size());
    std::vector<int> targets(ids.size());
    std::vector<int> valid(static_cast<std::size_t>(batch));
    std::vector<int> extents(static_cast<std::size_t>(batch));
    std::vector<int> lengths(static_cast<std::size_t>(batch));
    for (int row = 0; row < batch; ++row) {
        valid[static_cast<std::size_t>(row)]   = std::max(2, width - row % 3);
        extents[static_cast<std::size_t>(row)] = std::min(width - 1, (row * 5 + width) % width);
        lengths[static_cast<std::size_t>(row)] = 2048 + row * 53;
        for (int column = 0; column < width; ++column) {
            parents[static_cast<std::size_t>(row * width + column)] =
                column == 0 ? -1 : (column - 1) / 2;
            ids[static_cast<std::size_t>(row * width + column)] =
                (17 + row * 503 + column * 7919) % token_domain;
            const int first_child = column * 2 + 1;
            targets[static_cast<std::size_t>(row * width + column)] =
                first_child < valid[static_cast<std::size_t>(row)]
                    ? ids[static_cast<std::size_t>(row * width + first_child)]
                    : (101 + row * 31 + column) % token_domain;
        }
    }
    std::vector<hip_bfloat16> logits = make_logits(token_domain, width, batch);
    // Give stochastic rows a deterministic mixture of child-membership and correction columns;
    // the oracle still evaluates filtering, penalties, and RNG independently.
    for (int row = 0; row < batch; ++row) {
        for (int column = 0; column < width; ++column) {
            const int first_child = column * 2 + 1;
            const int peak = first_child < valid[static_cast<std::size_t>(row)] && column % 2 == 0
                                 ? ids[static_cast<std::size_t>(row * width + first_child)]
                                 : (211 + row * 71 + column * 13) % token_domain;
            logits[(static_cast<std::size_t>(row) * width + column) * token_domain + peak] =
                bf16(14.0F);
        }
    }
    std::vector<int> counts(static_cast<std::size_t>(batch) * token_domain, 0);
    std::vector<ninfer::ops::SamplingConfig> configs(static_cast<std::size_t>(batch));
    std::vector<bool> has_counts(static_cast<std::size_t>(batch));
    for (int row = 0; row < batch; ++row) {
        auto& config             = configs[static_cast<std::size_t>(row)];
        const bool greedy        = all_greedy || (batch > 1 && row == 0);
        config.temperature       = greedy ? 0.0F : 0.75F + 0.1F * static_cast<float>(row % 3);
        config.top_k             = greedy ? 0 : (row % 3 == 0 ? 1 : 7);
        config.top_p             = row % 2 == 0 ? 0.85F : 1.0F;
        config.min_p             = row % 3 == 2 ? 0.05F : 0.0F;
        config.presence_penalty  = greedy ? 0.0F : 0.2F;
        config.frequency_penalty = greedy ? 0.0F : 0.1F;
        config.seed = 0x5bd1e995ULL + static_cast<unsigned long long>(row * 997 + width);
        if (p_less) {
            config.p_less            = 1;
            config.temperature       = 1;
            config.top_k             = 1;
            config.top_p             = 0.01F;
            config.min_p             = 0.99F;
            config.presence_penalty  = 100;
            config.frequency_penalty = 100;
        }
        has_counts[static_cast<std::size_t>(row)]              = !greedy && row % 4 != 3;
        counts[static_cast<std::size_t>(row) * token_domain +
               ids[static_cast<std::size_t>(row * width + 1)]] = 2;
    }

    const int mask_words = (token_domain + 31) / 32;
    std::vector<std::uint32_t> masks(static_cast<std::size_t>(batch) * width * mask_words, 0);
    ninfer::DeviceBuffer d_masks(masks.size() * sizeof(masks[0]));
    if (p_less) {
        // Each node has its own eligible singleton; a depth-indexed rather than node-indexed
        // mask will produce the wrong licensed path once the traversal reaches column 3.
        for (int row = 0; row < batch; ++row) {
            extents[row]                             = width - 1;
            valid[row]                               = width;
            configs[row].allowed_token_column_stride = mask_words;
            configs[row].allowed_token_words         = masks.data() + row * width * mask_words;
            for (int node = 0; node < width; ++node) {
                const int child = 2 * node + 1;
                const int token =
                    child < width ? ids[row * width + child] : (101 + node) % token_domain;
                masks[(row * width + node) * mask_words + token / 32] |= 1U << (token % 32);
            }
        }
    }

    ninfer::DeviceBuffer d_targets(targets.size() * sizeof(int));
    ninfer::DeviceBuffer d_logits(logits.size() * sizeof(hip_bfloat16));
    ninfer::DeviceBuffer d_ids(ids.size() * sizeof(int));
    ninfer::DeviceBuffer d_parents(parents.size() * sizeof(int));
    ninfer::DeviceBuffer d_valid(valid.size() * sizeof(int));
    ninfer::DeviceBuffer d_extents(extents.size() * sizeof(int));
    ninfer::DeviceBuffer d_lengths(lengths.size() * sizeof(int));
    ninfer::DeviceBuffer d_anchors(static_cast<std::size_t>(batch) * sizeof(int));
    ninfer::DeviceBuffer d_licensed(ids.size() * sizeof(int));
    ninfer::DeviceBuffer d_licensed_counts(static_cast<std::size_t>(batch) * sizeof(int));
    ninfer::DeviceBuffer d_accepted(static_cast<std::size_t>(batch) * sizeof(int));
    ninfer::DeviceBuffer d_accepted_column(static_cast<std::size_t>(batch) * sizeof(int));
    ninfer::DeviceBuffer d_path(ids.size() * sizeof(int));
    ninfer::DeviceBuffer d_counts(counts.size() * sizeof(int));
    ninfer::DeviceBuffer d_configs(configs.size() * sizeof(configs[0]));
    d_targets.copy_from_host(targets.data(), targets.size() * sizeof(int));
    d_logits.copy_from_host(logits.data(), logits.size() * sizeof(hip_bfloat16));
    d_ids.copy_from_host(ids.data(), ids.size() * sizeof(int));
    d_parents.copy_from_host(parents.data(), parents.size() * sizeof(int));
    d_valid.copy_from_host(valid.data(), valid.size() * sizeof(int));
    d_extents.copy_from_host(extents.data(), extents.size() * sizeof(int));
    d_lengths.copy_from_host(lengths.data(), lengths.size() * sizeof(int));
    d_counts.copy_from_host(counts.data(), counts.size() * sizeof(int));
    for (int row = 0; row < batch; ++row) {
        configs[static_cast<std::size_t>(row)].token_counts =
            has_counts[static_cast<std::size_t>(row)]
                ? static_cast<int*>(d_counts.data()) + static_cast<std::size_t>(row) * token_domain
                : nullptr;
    }
    const TreeExpected expected =
        tree_oracle(token_domain, width, batch, targets, logits, ids, parents, valid, extents,
                    lengths, configs, has_counts, counts);
    d_masks.copy_from_host(masks.data(), masks.size() * sizeof(masks[0]));
    for (int row = 0; p_less && row < batch; ++row)
        configs[row].allowed_token_words =
            static_cast<std::uint32_t*>(d_masks.data()) + row * width * mask_words;
    d_configs.copy_from_host(configs.data(), configs.size() * sizeof(configs[0]));

    ninfer::Tensor target_tensor(d_targets.data(), ninfer::DType::I32, {width, batch});
    ninfer::Tensor logits_tensor(d_logits.data(), ninfer::DType::BF16,
                                 {token_domain, width, batch});
    ninfer::Tensor ids_tensor(d_ids.data(), ninfer::DType::I32, {width, batch});
    ninfer::Tensor parents_tensor(d_parents.data(), ninfer::DType::I32, {width, batch});
    ninfer::Tensor valid_tensor(d_valid.data(), ninfer::DType::I32, {batch});
    ninfer::Tensor extents_tensor(d_extents.data(), ninfer::DType::I32, {batch});
    ninfer::Tensor lengths_tensor(d_lengths.data(), ninfer::DType::I32, {batch});
    ninfer::Tensor anchors_tensor(d_anchors.data(), ninfer::DType::I32, {batch});
    ninfer::Tensor licensed_tensor(d_licensed.data(), ninfer::DType::I32, {width, batch});
    ninfer::Tensor licensed_counts_tensor(d_licensed_counts.data(), ninfer::DType::I32, {batch});
    ninfer::Tensor accepted_tensor(d_accepted.data(), ninfer::DType::I32, {batch});
    ninfer::Tensor accepted_column_tensor(d_accepted_column.data(), ninfer::DType::I32, {batch});
    ninfer::Tensor path_tensor(d_path.data(), ninfer::DType::I32, {width, batch});
    const std::size_t workspace_bytes =
        ninfer::ops::speculative_accept_tree_drafts_workspace_capacity_bytes(token_domain, width,
                                                                             width, batch, batch);
    ninfer::DeviceArena workspace(std::max<std::size_t>(workspace_bytes, 1));
    ninfer::ops::speculative_accept_tree_drafts(
        target_tensor, logits_tensor, ids_tensor, parents_tensor, valid_tensor, extents_tensor,
        lengths_tensor, anchors_tensor, licensed_tensor, licensed_counts_tensor, accepted_tensor,
        accepted_column_tensor, path_tensor, token_domain,
        static_cast<const ninfer::ops::SamplingConfig*>(d_configs.data()), workspace,
        device.stream);
    device.synchronize();

    const std::string label = "tree V=" + std::to_string(token_domain) +
                              " W=" + std::to_string(width) + " B=" + std::to_string(batch);
    require_equal(copy_from<int>(d_lengths, batch), expected.lengths, label + " lengths");
    require_equal(copy_from<int>(d_anchors, batch), expected.anchors, label + " anchors");
    require_equal(copy_from<int>(d_licensed, ids.size()), expected.licensed, label + " licensed");
    require_equal(copy_from<int>(d_licensed_counts, batch), expected.licensed_counts,
                  label + " licensed counts");
    require_equal(copy_from<int>(d_accepted, batch), expected.accepted, label + " accepted");
    require_equal(copy_from<int>(d_accepted_column, batch), expected.accepted_column,
                  label + " accepted columns");
    require_equal(copy_from<int>(d_path, ids.size()), expected.path, label + " fold path");
    require_equal(copy_from<int>(d_counts, counts.size()), expected.counts,
                  label + " token counts");
    require_equal(copy_from<int>(d_ids, ids.size()), ids, label + " ids immutable");
    require_equal(copy_from<int>(d_parents, parents.size()), parents, label + " parents immutable");
    require_equal(copy_from<int>(d_targets, targets.size()), targets, label + " targets immutable");
    require_equal(copy_from<int>(d_valid, valid.size()), valid, label + " valid immutable");
    require_equal(copy_from<int>(d_extents, extents.size()), extents, label + " extents immutable");
    require_equal(copy_from<std::uint16_t>(d_logits, logits.size()), bf16_bits(logits),
                  label + " logits immutable");
    require(workspace.used() == 0 && workspace.peak_used() == workspace_bytes,
            label + " workspace high-water differs from sizing query");
    if (p_less) {
        hipGraph_t graph          = nullptr;
        hipGraphExec_t executable = nullptr;
        HIP_CHECK(hipStreamBeginCapture(device.stream, hipStreamCaptureModeGlobal));
        ninfer::ops::speculative_accept_tree_drafts(
            target_tensor, logits_tensor, ids_tensor, parents_tensor, valid_tensor, extents_tensor,
            lengths_tensor, anchors_tensor, licensed_tensor, licensed_counts_tensor,
            accepted_tensor, accepted_column_tensor, path_tensor, token_domain,
            static_cast<const ninfer::ops::SamplingConfig*>(d_configs.data()), workspace,
            device.stream);
        HIP_CHECK(hipStreamEndCapture(device.stream, &graph));
        HIP_CHECK(hipGraphInstantiate(&executable, graph, nullptr, nullptr, 0));
        for (int replay = 0; replay < 2; ++replay) {
            d_lengths.copy_from_host(lengths.data(), lengths.size() * sizeof(int));
            d_counts.copy_from_host(counts.data(), counts.size() * sizeof(int));
            HIP_CHECK(hipMemsetAsync(d_licensed.data(), 0xff, width * batch * sizeof(int),
                                     device.stream));
            HIP_CHECK(
                hipMemsetAsync(d_path.data(), 0xff, width * batch * sizeof(int), device.stream));
            HIP_CHECK(hipGraphLaunch(executable, device.stream));
            device.synchronize();
            require_equal(copy_from<int>(d_licensed, width * batch), expected.licensed,
                          label + " graph licensed");
            require_equal(copy_from<int>(d_path, width * batch), expected.path,
                          label + " graph path");
            require_equal(copy_from<int>(d_lengths, batch), expected.lengths,
                          label + " graph lengths");
            require_equal(copy_from<int>(d_counts, counts.size()), expected.counts,
                          label + " graph counts");
        }
        HIP_CHECK(hipGraphExecDestroy(executable));
        HIP_CHECK(hipGraphDestroy(graph));
    }
    if (token_domain == kRealVocabulary && !p_less) {
        for (int warmup = 0; warmup < 3; ++warmup) {
            ninfer::ops::speculative_accept_tree_drafts(
                target_tensor, logits_tensor, ids_tensor, parents_tensor, valid_tensor,
                extents_tensor, lengths_tensor, anchors_tensor, licensed_tensor,
                licensed_counts_tensor, accepted_tensor, accepted_column_tensor, path_tensor,
                token_domain, static_cast<const ninfer::ops::SamplingConfig*>(d_configs.data()),
                workspace, device.stream);
        }
        constexpr int repeats = 20;
        ninfer::DeviceEventTimer timer(device);
        timer.start();
        for (int repeat = 0; repeat < repeats; ++repeat) {
            ninfer::ops::speculative_accept_tree_drafts(
                target_tensor, logits_tensor, ids_tensor, parents_tensor, valid_tensor,
                extents_tensor, lengths_tensor, anchors_tensor, licensed_tensor,
                licensed_counts_tensor, accepted_tensor, accepted_column_tensor, path_tensor,
                token_domain, static_cast<const ninfer::ops::SamplingConfig*>(d_configs.data()),
                workspace, device.stream);
        }
        timer.record_stop();
        device.synchronize();
        std::cout << "timing: tree V=" << token_domain << " W=" << width << " B=" << batch
                  << " ms=" << timer.elapsed_ms() / static_cast<float>(repeats)
                  << " workspace_bytes=" << workspace_bytes << '\n';
    }
}

void qualify_hidden_and_remap(ninfer::DeviceContext& device) {
    constexpr int rows    = 257;
    constexpr int columns = 16;
    constexpr int batch   = static_cast<int>(ninfer::kMaximumConcurrency);
    std::vector<hip_bfloat16> hidden(static_cast<std::size_t>(rows) * columns * batch);
    for (std::size_t index = 0; index < hidden.size(); ++index) {
        hidden[index].data = static_cast<std::uint16_t>((index * 7919U + 17U) & 0xffffU);
    }
    std::vector<int> selectors(batch);
    for (int row = 0; row < batch; ++row) {
        selectors[static_cast<std::size_t>(row)] = (row * 2 + row / 8) % columns;
    }
    std::vector<hip_bfloat16> expected(static_cast<std::size_t>(rows) * batch);
    for (int row = 0; row < batch; ++row) {
        std::copy_n(hidden.begin() +
                        (static_cast<std::size_t>(row) * columns + selectors[row]) * rows,
                    rows, expected.begin() + static_cast<std::size_t>(row) * rows);
    }
    ninfer::DeviceBuffer d_hidden(hidden.size() * sizeof(hip_bfloat16));
    ninfer::DeviceBuffer d_selectors(selectors.size() * sizeof(int));
    ninfer::DeviceBuffer d_output(expected.size() * sizeof(hip_bfloat16));
    d_hidden.copy_from_host(hidden.data(), hidden.size() * sizeof(hip_bfloat16));
    d_selectors.copy_from_host(selectors.data(), selectors.size() * sizeof(int));
    ninfer::Tensor hidden_tensor(d_hidden.data(), ninfer::DType::BF16, {rows, columns, batch});
    ninfer::Tensor selectors_tensor(d_selectors.data(), ninfer::DType::I32, {batch});
    ninfer::Tensor output_tensor(d_output.data(), ninfer::DType::BF16, {rows, batch});
    ninfer::ops::speculative_select_accepted_hidden(hidden_tensor, selectors_tensor, output_tensor,
                                                    device.stream);
    device.synchronize();
    const auto actual_bits = copy_from<std::uint16_t>(d_output, expected.size());
    std::vector<std::uint16_t> expected_bits(expected.size());
    for (std::size_t index = 0; index < expected.size(); ++index) {
        expected_bits[index] = expected[index].data;
    }
    require_equal(actual_bits, expected_bits, "accepted hidden exact BF16 bits");

    constexpr int map_count = 4096;
    std::vector<int> map(map_count);
    for (int index = 0; index < map_count; ++index) {
        map[static_cast<std::size_t>(index)] = (index * 2053 + 17) & (map_count - 1);
    }
    for (const int token_count : {1, 8, 16, 257}) {
        std::vector<int> tokens(static_cast<std::size_t>(token_count));
        for (int index = 0; index < token_count; ++index) {
            tokens[static_cast<std::size_t>(index)] = (index * 7919) & (map_count - 1);
        }
        std::vector<int> remapped(tokens.size());
        for (std::size_t index = 0; index < tokens.size(); ++index) {
            remapped[index] = map[static_cast<std::size_t>(tokens[index])];
        }
        ninfer::DeviceBuffer d_map(map.size() * sizeof(int));
        ninfer::DeviceBuffer d_tokens(tokens.size() * sizeof(int));
        d_map.copy_from_host(map.data(), map.size() * sizeof(int));
        d_tokens.copy_from_host(tokens.data(), tokens.size() * sizeof(int));
        ninfer::Tensor token_tensor(d_tokens.data(), ninfer::DType::I32, {token_count});
        ninfer::ops::proposal_remap_token_ids(token_tensor, static_cast<const int*>(d_map.data()),
                                              map_count, device.stream);
        device.synchronize();
        require_equal(copy_from<int>(d_tokens, tokens.size()), remapped,
                      "proposal remap T=" + std::to_string(token_count));
        require_equal(copy_from<int>(d_map, map.size()), map, "proposal map immutable");
    }
}

void qualify_graph_replay(ninfer::DeviceContext& device) {
    constexpr int token_domain = 1024;
    constexpr int k            = 2;
    constexpr int columns      = 3;
    std::vector<int> targets{7, 11, 13};
    std::vector<int> drafts{7, 11};
    std::vector<int> extents{2};
    std::vector<int> lengths{100};
    std::vector<hip_bfloat16> logits(static_cast<std::size_t>(token_domain) * columns,
                                     bf16(-20.0F));
    logits[7]                                               = bf16(20.0F);
    logits[static_cast<std::size_t>(token_domain) + 11]     = bf16(20.0F);
    logits[static_cast<std::size_t>(token_domain) * 2 + 13] = bf16(20.0F);
    ninfer::ops::SamplingConfig config{};
    config.temperature = 1.0F;
    config.top_k       = 1;
    config.seed        = 1234;
    ninfer::DeviceBuffer d_targets(targets.size() * sizeof(int));
    ninfer::DeviceBuffer d_drafts(drafts.size() * sizeof(int));
    ninfer::DeviceBuffer d_extents(sizeof(int));
    ninfer::DeviceBuffer d_lengths(sizeof(int));
    ninfer::DeviceBuffer d_anchors(sizeof(int));
    ninfer::DeviceBuffer d_licensed(columns * sizeof(int));
    ninfer::DeviceBuffer d_licensed_counts(sizeof(int));
    ninfer::DeviceBuffer d_accepted(sizeof(int));
    ninfer::DeviceBuffer d_logits(logits.size() * sizeof(hip_bfloat16));
    ninfer::DeviceBuffer d_config(sizeof(config));
    d_targets.copy_from_host(targets.data(), targets.size() * sizeof(int));
    d_drafts.copy_from_host(drafts.data(), drafts.size() * sizeof(int));
    d_extents.copy_from_host(extents.data(), sizeof(int));
    d_lengths.copy_from_host(lengths.data(), sizeof(int));
    d_logits.copy_from_host(logits.data(), logits.size() * sizeof(hip_bfloat16));
    d_config.copy_from_host(&config, sizeof(config));
    ninfer::Tensor target_tensor(d_targets.data(), ninfer::DType::I32, {columns});
    ninfer::Tensor logits_tensor(d_logits.data(), ninfer::DType::BF16, {token_domain, columns});
    ninfer::Tensor draft_tensor(d_drafts.data(), ninfer::DType::I32, {k});
    ninfer::Tensor extent_tensor(d_extents.data(), ninfer::DType::I32, {1});
    ninfer::Tensor length_tensor(d_lengths.data(), ninfer::DType::I32, {1});
    ninfer::Tensor anchor_tensor(d_anchors.data(), ninfer::DType::I32, {1});
    ninfer::Tensor licensed_tensor(d_licensed.data(), ninfer::DType::I32, {columns});
    ninfer::Tensor licensed_counts_tensor(d_licensed_counts.data(), ninfer::DType::I32, {1});
    ninfer::Tensor accepted_tensor(d_accepted.data(), ninfer::DType::I32, {1});
    const std::size_t bytes =
        ninfer::ops::speculative_accept_greedy_drafts_workspace_capacity_bytes(token_domain, k, k,
                                                                               1, 1);
    ninfer::DeviceArena workspace(bytes);

    hipGraph_t graph          = nullptr;
    hipGraphExec_t executable = nullptr;
    HIP_CHECK(hipStreamBeginCapture(device.stream, hipStreamCaptureModeGlobal));
    ninfer::ops::speculative_accept_greedy_drafts(
        target_tensor, logits_tensor, draft_tensor, extent_tensor, length_tensor, anchor_tensor,
        licensed_tensor, licensed_counts_tensor, accepted_tensor, token_domain,
        static_cast<const ninfer::ops::SamplingConfig*>(d_config.data()), workspace, device.stream);
    HIP_CHECK(hipStreamEndCapture(device.stream, &graph));
    HIP_CHECK(hipGraphInstantiate(&executable, graph, nullptr, nullptr, 0));
    for (int replay = 0; replay < 3; ++replay) {
        const int reset_length = 100 + replay * 17;
        d_lengths.copy_from_host(&reset_length, sizeof(int));
        HIP_CHECK(hipGraphLaunch(executable, device.stream));
        device.synchronize();
        require_equal(copy_from<int>(d_licensed, columns), std::vector<int>({7, 11, 13}),
                      "graph replay licensed tokens");
        require_equal(copy_from<int>(d_lengths, 1), std::vector<int>({reset_length + 3}),
                      "graph replay dynamic state");
    }
    HIP_CHECK(hipGraphExecDestroy(executable));
    HIP_CHECK(hipGraphDestroy(graph));
}

void qualify_validation(ninfer::DeviceContext& device) {
    try {
        (void)ninfer::ops::speculative_accept_greedy_drafts_workspace_capacity_bytes(
            1, 0, 8, 1, static_cast<int>(ninfer::kMaximumConcurrency));
        fail("chain workspace accepted K=0");
        // NOLINTNEXTLINE(bugprone-empty-catch): expected rejection is the pass path
    } catch (const std::invalid_argument&) {}
    try {
        (void)ninfer::ops::speculative_accept_tree_drafts_workspace_capacity_bytes(
            1, 1, 16, 1, static_cast<int>(ninfer::kMaximumConcurrency));
        fail("tree workspace accepted W=1");
        // NOLINTNEXTLINE(bugprone-empty-catch): expected rejection is the pass path
    } catch (const std::invalid_argument&) {}
    try {
        (void)ninfer::ops::speculative_accept_tree_drafts_workspace_capacity_bytes(
            1, 2, 16, 1, static_cast<int>(ninfer::kMaximumConcurrency) + 1);
        fail("tree workspace accepted B above product concurrency");
        // NOLINTNEXTLINE(bugprone-empty-catch): expected rejection is the pass path
    } catch (const std::invalid_argument&) {}

    ninfer::DeviceBuffer d_i32(64);
    ninfer::Tensor vector(d_i32.data(), ninfer::DType::I32, {1});
    ninfer::Tensor matrix(d_i32.data(), ninfer::DType::I32, {2});
    try {
        ninfer::ops::speculative_prepare_verify_ids(vector, vector, vector, matrix, nullptr);
        fail("prepare accepted a null stream");
        // NOLINTNEXTLINE(bugprone-empty-catch): expected rejection is the pass path
    } catch (const std::invalid_argument&) {}
    try {
        ninfer::ops::speculative_prepare_verify_ids(vector, vector, vector, matrix, device.stream);
        fail("prepare accepted overlapping tensors");
        // NOLINTNEXTLINE(bugprone-empty-catch): expected rejection is the pass path
    } catch (const std::invalid_argument&) {}

    constexpr int token_domain = 1024;
    ninfer::DeviceBuffer d_target(2 * sizeof(int));
    ninfer::DeviceBuffer d_logits(2 * token_domain * sizeof(hip_bfloat16));
    ninfer::DeviceBuffer d_draft(sizeof(int));
    ninfer::DeviceBuffer d_extent(sizeof(int));
    ninfer::DeviceBuffer d_length(sizeof(int));
    ninfer::DeviceBuffer d_anchor(sizeof(int));
    ninfer::DeviceBuffer d_licensed(2 * sizeof(int));
    ninfer::DeviceBuffer d_licensed_count(sizeof(int));
    ninfer::DeviceBuffer d_accepted(sizeof(int));
    ninfer::DeviceBuffer d_config(sizeof(ninfer::ops::SamplingConfig));
    ninfer::DeviceBuffer d_selector_ids(sizeof(int));
    const std::vector<int> targets{7, 11};
    const std::vector<int> drafts{7};
    const std::vector<int> one{1};
    const std::vector<hip_bfloat16> logits(2 * token_domain, bf16(0.0F));
    ninfer::ops::SamplingConfig config{};
    config.temperature = 1.0F;
    config.top_k       = 1;
    d_target.copy_from_host(targets.data(), targets.size() * sizeof(int));
    d_logits.copy_from_host(logits.data(), logits.size() * sizeof(hip_bfloat16));
    d_draft.copy_from_host(drafts.data(), sizeof(int));
    d_extent.copy_from_host(one.data(), sizeof(int));
    d_length.copy_from_host(one.data(), sizeof(int));
    d_config.copy_from_host(&config, sizeof(config));
    ninfer::Tensor target(d_target.data(), ninfer::DType::I32, {2});
    ninfer::Tensor logits_tensor(d_logits.data(), ninfer::DType::BF16, {token_domain, 2});
    ninfer::Tensor draft(d_draft.data(), ninfer::DType::I32, {1});
    ninfer::Tensor extent(d_extent.data(), ninfer::DType::I32, {1});
    ninfer::Tensor length(d_length.data(), ninfer::DType::I32, {1});
    ninfer::Tensor anchor(d_anchor.data(), ninfer::DType::I32, {1});
    ninfer::Tensor licensed(d_licensed.data(), ninfer::DType::I32, {2});
    ninfer::Tensor licensed_count(d_licensed_count.data(), ninfer::DType::I32, {1});
    ninfer::Tensor accepted(d_accepted.data(), ninfer::DType::I32, {1});
    ninfer::Tensor selector_ids(d_selector_ids.data(), ninfer::DType::I32, {1, 1, 1});
    const std::size_t needed =
        ninfer::ops::speculative_accept_greedy_drafts_workspace_capacity_bytes(token_domain, 1, 1,
                                                                               1, 1);
    ninfer::DeviceArena workspace(needed);
    try {
        ninfer::ops::speculative_accept_greedy_drafts(
            target, logits_tensor, draft, extent, length, anchor, licensed, licensed_count,
            accepted, token_domain, nullptr, workspace, device.stream);
        fail("chain acceptance accepted a null config pointer");
        // NOLINTNEXTLINE(bugprone-empty-catch): expected rejection is the pass path
    } catch (const std::invalid_argument&) {}
    try {
        ninfer::ops::speculative_accept_greedy_drafts(
            target, logits_tensor, draft, extent, length, anchor, licensed, licensed_count,
            accepted, token_domain,
            static_cast<const ninfer::ops::SamplingConfig*>(d_config.data()), workspace,
            device.stream, &selector_ids, nullptr);
        fail("chain acceptance accepted an unpaired selector");
        // NOLINTNEXTLINE(bugprone-empty-catch): expected rejection is the pass path
    } catch (const std::invalid_argument&) {}
    try {
        ninfer::DeviceArena undersized(needed - 1);
        ninfer::ops::speculative_accept_greedy_drafts(
            target, logits_tensor, draft, extent, length, anchor, licensed, licensed_count,
            accepted, token_domain,
            static_cast<const ninfer::ops::SamplingConfig*>(d_config.data()), undersized,
            device.stream);
        fail("chain acceptance accepted undersized workspace");
        // NOLINTNEXTLINE(bugprone-empty-catch): expected rejection is the pass path
    } catch (const std::bad_alloc&) {}
    try {
        ninfer::Tensor aliased_output(d_target.data(), ninfer::DType::I32, {2});
        ninfer::ops::speculative_accept_greedy_drafts(
            target, logits_tensor, draft, extent, length, anchor, aliased_output, licensed_count,
            accepted, token_domain,
            static_cast<const ninfer::ops::SamplingConfig*>(d_config.data()), workspace,
            device.stream);
        fail("chain acceptance accepted output/input aliasing");
        // NOLINTNEXTLINE(bugprone-empty-catch): expected rejection is the pass path
    } catch (const std::invalid_argument&) {}

    try {
        ninfer::Tensor tokens(d_target.data(), ninfer::DType::I32, {2});
        ninfer::ops::proposal_remap_token_ids(tokens, static_cast<const int*>(d_target.data()), 2,
                                              device.stream);
        fail("proposal remap accepted an overlapping id map");
        // NOLINTNEXTLINE(bugprone-empty-catch): expected rejection is the pass path
    } catch (const std::invalid_argument&) {}
}

void benchmark(ninfer::DeviceContext& device) {
    constexpr int token_domain       = kRealVocabulary;
    constexpr int k                  = 8;
    constexpr int columns            = k + 1;
    constexpr int batch              = static_cast<int>(ninfer::kMaximumConcurrency);
    std::vector<hip_bfloat16> logits = make_logits(token_domain, columns, batch);
    std::vector<int> targets(static_cast<std::size_t>(columns) * batch, 0);
    std::vector<int> drafts(static_cast<std::size_t>(k) * batch, 0);
    std::vector<int> extents(batch, k);
    std::vector<int> lengths(batch, 8192);
    std::vector<ninfer::ops::SamplingConfig> configs(batch);
    for (int row = 0; row < batch; ++row) {
        configs[static_cast<std::size_t>(row)].temperature = 0.8F;
        configs[static_cast<std::size_t>(row)].top_k       = 20;
        configs[static_cast<std::size_t>(row)].top_p       = 0.9F;
        configs[static_cast<std::size_t>(row)].min_p       = 0.05F;
        configs[static_cast<std::size_t>(row)].seed = static_cast<unsigned long long>(row + 1);
    }
    ninfer::DeviceBuffer d_targets(targets.size() * sizeof(int));
    ninfer::DeviceBuffer d_logits(logits.size() * sizeof(hip_bfloat16));
    ninfer::DeviceBuffer d_drafts(drafts.size() * sizeof(int));
    ninfer::DeviceBuffer d_extents(extents.size() * sizeof(int));
    ninfer::DeviceBuffer d_lengths(lengths.size() * sizeof(int));
    ninfer::DeviceBuffer d_anchors(batch * sizeof(int));
    ninfer::DeviceBuffer d_licensed(static_cast<std::size_t>(columns) * batch * sizeof(int));
    ninfer::DeviceBuffer d_licensed_counts(batch * sizeof(int));
    ninfer::DeviceBuffer d_accepted(batch * sizeof(int));
    ninfer::DeviceBuffer d_configs(configs.size() * sizeof(configs[0]));
    d_targets.copy_from_host(targets.data(), targets.size() * sizeof(int));
    d_logits.copy_from_host(logits.data(), logits.size() * sizeof(hip_bfloat16));
    d_drafts.copy_from_host(drafts.data(), drafts.size() * sizeof(int));
    d_extents.copy_from_host(extents.data(), extents.size() * sizeof(int));
    d_lengths.copy_from_host(lengths.data(), lengths.size() * sizeof(int));
    d_configs.copy_from_host(configs.data(), configs.size() * sizeof(configs[0]));
    ninfer::Tensor target_tensor(d_targets.data(), ninfer::DType::I32, {columns, batch});
    ninfer::Tensor logits_tensor(d_logits.data(), ninfer::DType::BF16,
                                 {token_domain, columns, batch});
    ninfer::Tensor draft_tensor(d_drafts.data(), ninfer::DType::I32, {k, batch});
    ninfer::Tensor extent_tensor(d_extents.data(), ninfer::DType::I32, {batch});
    ninfer::Tensor length_tensor(d_lengths.data(), ninfer::DType::I32, {batch});
    ninfer::Tensor anchor_tensor(d_anchors.data(), ninfer::DType::I32, {batch});
    ninfer::Tensor licensed_tensor(d_licensed.data(), ninfer::DType::I32, {columns, batch});
    ninfer::Tensor licensed_counts_tensor(d_licensed_counts.data(), ninfer::DType::I32, {batch});
    ninfer::Tensor accepted_tensor(d_accepted.data(), ninfer::DType::I32, {batch});
    const std::size_t bytes =
        ninfer::ops::speculative_accept_greedy_drafts_workspace_capacity_bytes(token_domain, k, k,
                                                                               batch, batch);
    ninfer::DeviceArena workspace(bytes);
    for (int warmup = 0; warmup < 3; ++warmup) {
        ninfer::ops::speculative_accept_greedy_drafts(
            target_tensor, logits_tensor, draft_tensor, extent_tensor, length_tensor, anchor_tensor,
            licensed_tensor, licensed_counts_tensor, accepted_tensor, token_domain,
            static_cast<const ninfer::ops::SamplingConfig*>(d_configs.data()), workspace,
            device.stream);
    }
    constexpr int repeats = 20;
    ninfer::DeviceEventTimer timer(device);
    timer.start();
    for (int repeat = 0; repeat < repeats; ++repeat) {
        ninfer::ops::speculative_accept_greedy_drafts(
            target_tensor, logits_tensor, draft_tensor, extent_tensor, length_tensor, anchor_tensor,
            licensed_tensor, licensed_counts_tensor, accepted_tensor, token_domain,
            static_cast<const ninfer::ops::SamplingConfig*>(d_configs.data()), workspace,
            device.stream);
    }
    timer.record_stop();
    device.synchronize();
    std::cout << "timing: chain V=" << token_domain << " K=" << k << " B=" << batch
              << " ms=" << timer.elapsed_ms() / static_cast<float>(repeats)
              << " workspace_bytes=" << bytes << '\n';
}

} // namespace

int main() {
    try {
        ninfer::DeviceContext device(select_r9700());
        std::cout << "device: arch=" << device.props.gcnArchName
                  << " wave=" << device.props.warpSize << '\n';
        qualify_prepare(device);
        qualify_nonfinite_status(device);
        qualify_zero_acceptance(device);
        qualify_empty_p_less_support(device);
        for (int k = 1; k <= 8; ++k) {
            for (int batch = 1; batch <= static_cast<int>(ninfer::kMaximumConcurrency); ++batch) {
                run_chain_case(device, 64, k, batch, true, false);
            }
        }
        run_chain_case(device, 257, 8, static_cast<int>(ninfer::kMaximumConcurrency), false, true);
        run_chain_case(device, 4099, 8, static_cast<int>(ninfer::kMaximumConcurrency), false, true);
        run_chain_case(device, kRealVocabulary, 8, static_cast<int>(ninfer::kMaximumConcurrency),
                       false, true);
        for (int tokens : {127, 257, kRealVocabulary}) {
            run_chain_case(device, tokens, 5, 4, false, true, true);
            run_chain_case(device, tokens, 4, 4, false, false, true);
        }
        // Both verify routes: single-workgroup (V=64) and multi-block finalize (product V).
        for (const bool chained : {false, true}) {
            run_p_less_block_statistics(device, 64, 64, chained);
            run_p_less_block_statistics(device, kRealVocabulary, 248077, chained);
        }
        // Low positive temperature must preserve the flat 64-atom law through verification
        // moments, acceptance, correction, and bonus draws on both sampling routes.
        run_p_less_block_statistics(device, 64, 64, false, 1.0e-40F, 1024);
        run_p_less_block_statistics(device, kRealVocabulary, 248077, false, 1.0e-40F, 1024);
        run_p_less_proposal_calibration(device);
        for (int width = 2; width <= 16; ++width) {
            for (int batch = 1; batch <= static_cast<int>(ninfer::kMaximumConcurrency); ++batch) {
                run_tree_case(device, 64, width, batch, true);
            }
        }
        run_tree_case(device, 257, 16, static_cast<int>(ninfer::kMaximumConcurrency), false);
        run_tree_case(device, 4099, 16, static_cast<int>(ninfer::kMaximumConcurrency), false);
        run_tree_case(device, kRealVocabulary, 12, static_cast<int>(ninfer::kMaximumConcurrency),
                      false);
        for (int tokens : {127, 257, kRealVocabulary})
            run_tree_case(device, tokens, 6, 4, false, true);
        qualify_hidden_and_remap(device);
        qualify_graph_replay(device);
        qualify_validation(device);
        benchmark(device);
        std::cout << "speculative_round: PASS\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "speculative_round: FAIL: " << error.what() << '\n';
        return 1;
    }
}
