#include "runtime/contract/sampling.h"
#include "core/device.h"
#include "ninfer/ops/sampling.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace ninfer::runtime {
namespace {

void validate(const ResolvedSamplingParameters& sampling) {
    if (!std::isfinite(sampling.temperature) || !std::isfinite(sampling.top_p) ||
        !std::isfinite(sampling.min_p) || !std::isfinite(sampling.presence_penalty) ||
        !std::isfinite(sampling.frequency_penalty)) {
        throw std::invalid_argument("sampling parameters must be finite");
    }
    if (sampling.temperature < 0.0F || sampling.temperature > 2.0F) {
        throw std::invalid_argument("temperature must be in [0,2]");
    }
    if (sampling.top_k < 0) { throw std::invalid_argument("top_k must be nonnegative"); }
    if (sampling.top_p < 0.0F || sampling.top_p > 1.0F) {
        throw std::invalid_argument("top_p must be in [0,1]");
    }
    if (sampling.min_p < 0.0F || sampling.min_p > 1.0F) {
        throw std::invalid_argument("min_p must be in [0,1]");
    }
    if (sampling.presence_penalty < -2.0F || sampling.presence_penalty > 2.0F) {
        throw std::invalid_argument("presence_penalty must be in [-2,2]");
    }
    if (sampling.frequency_penalty < -2.0F || sampling.frequency_penalty > 2.0F) {
        throw std::invalid_argument("frequency_penalty must be in [-2,2]");
    }
}

} // namespace

void rollback_sampling_counts(const ops::SamplingConfig& sampling,
                              std::span<const TokenId> tokens) {
    if (sampling.temperature <= 0.0F || sampling.token_counts == nullptr) { return; }
    for (std::size_t index = 0; index < tokens.size(); ++index) {
        if (std::find(tokens.begin(), tokens.begin() + static_cast<std::ptrdiff_t>(index),
                      tokens[index]) !=
            tokens.begin() + static_cast<std::ptrdiff_t>(index)) {
            continue;
        }
        const auto occurrences = static_cast<std::int32_t>(
            std::count(tokens.begin() + static_cast<std::ptrdiff_t>(index), tokens.end(),
                       tokens[index]));
        std::int32_t count = 0;
        CUDA_CHECK(cudaMemcpy(&count, sampling.token_counts + tokens[index], sizeof(count),
                              cudaMemcpyDeviceToHost));
        if (count < occurrences) {
            throw std::logic_error("rejected sampling token count underflow");
        }
        count -= occurrences;
        CUDA_CHECK(cudaMemcpy(sampling.token_counts + tokens[index], &count, sizeof(count),
                              cudaMemcpyHostToDevice));
    }
}

ResolvedSamplingParameters resolve_sampling(const ModelSamplingDefaults& defaults,
                                            SamplingMode mode, const SamplingOverrides& overrides) {
    const SamplingPreset& preset = defaults.for_mode(mode);
    ResolvedSamplingParameters resolved{
        .temperature       = overrides.temperature.value_or(preset.temperature),
        .top_k             = overrides.top_k.value_or(preset.top_k),
        .top_p             = overrides.top_p.value_or(preset.top_p),
        .min_p             = overrides.min_p.value_or(preset.min_p),
        .presence_penalty  = overrides.presence_penalty.value_or(preset.presence_penalty),
        .frequency_penalty = overrides.frequency_penalty.value_or(preset.frequency_penalty),
        .seed              = overrides.seed.value_or(0),
        .p_less            = overrides.p_less,
    };
    validate(resolved);
    if (resolved.p_less) {
        resolved.top_k             = 0;
        resolved.top_p             = 1.0F;
        resolved.min_p             = 0.0F;
        resolved.presence_penalty  = 0.0F;
        resolved.frequency_penalty = 0.0F;
    }
    return resolved;
}

} // namespace ninfer::runtime
