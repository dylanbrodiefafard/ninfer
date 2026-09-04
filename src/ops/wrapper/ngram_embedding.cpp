#include "ninfer/ops/ngram_embedding.h"

#include "ops/launcher/ngram_embedding.h"

#include <array>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>

namespace ninfer::ops {
namespace {

constexpr std::uint64_t kGolden = 0x9E3779B97F4A7C15ULL;

std::uint64_t splitmix64(std::uint64_t x) {
    std::uint64_t z = x + kGolden;
    z               = (z ^ (z >> 30U)) * 0xBF58476D1CE4E5B9ULL;
    z               = (z ^ (z >> 27U)) * 0x94D049BB133111EBULL;
    return z ^ (z >> 31U);
}

bool is_prime(std::int64_t value) {
    if (value < 2) { return false; }
    if ((value & 1) == 0) { return value == 2; }
    for (std::int64_t divisor = 3; divisor <= value / divisor; divisor += 2) {
        if (value % divisor == 0) { return false; }
    }
    return true;
}

PreparedNgramRowConfig prepare_config(const NgramRowConfig& config) {
    if (config.vocabulary_size <= 0) {
        throw std::invalid_argument("ngram_row_ids: vocabulary_size must be positive");
    }
    if (config.eos_token_id < 0 || config.eos_token_id >= config.vocabulary_size) {
        throw std::invalid_argument("ngram_row_ids: eos_token_id is outside the vocabulary");
    }
    if (config.ple_layer_index != 0) {
        throw std::invalid_argument("ngram_row_ids: preview ple_layer_index must be zero");
    }
    if (config.vocab_base <= 0) {
        throw std::invalid_argument("ngram_row_ids: vocab_base must be positive");
    }

    PreparedNgramRowConfig prepared{};
    prepared.vocabulary_size = config.vocabulary_size;
    prepared.eos_token_id = config.eos_token_id;
    const std::uint64_t multiplier_max =
        static_cast<std::uint64_t>(std::numeric_limits<std::int64_t>::max()) /
        static_cast<std::uint64_t>(config.vocabulary_size);
    const std::uint64_t half_bound = multiplier_max / 2U > 0 ? multiplier_max / 2U : 1U;
    const std::uint64_t base_seed =
        config.seed + 10007ULL * static_cast<std::uint64_t>(config.ple_layer_index);
    for (std::uint64_t j = 0; j < 3; ++j) {
        prepared.multiplier[j] =
            2U * (splitmix64(base_seed + kGolden * (j + 1U)) % half_bound) + 1U;
    }

    const std::int64_t first_global_head =
        static_cast<std::int64_t>(config.ple_layer_index) * 16;
    std::int64_t candidate               = config.vocab_base;
    std::int64_t found                   = 0;
    std::int64_t offset                  = 0;
    while (found < first_global_head + 16) {
        if (candidate > std::numeric_limits<std::int32_t>::max()) {
            throw std::invalid_argument("ngram_row_ids: head prime exceeds positive I32");
        }
        if (is_prime(candidate)) {
            if (found >= first_global_head) {
                const std::int32_t local = static_cast<std::int32_t>(found - first_global_head);
                if (offset > std::numeric_limits<std::int32_t>::max()) {
                    throw std::invalid_argument("ngram_row_ids: row offset exceeds positive I32");
                }
                prepared.prime[local]  = static_cast<std::int32_t>(candidate);
                prepared.offset[local] = static_cast<std::int32_t>(offset);
                offset += candidate;
            }
            ++found;
        }
        ++candidate;
    }
    if (offset > static_cast<std::int64_t>(std::numeric_limits<std::int32_t>::max()) + 1) {
        throw std::invalid_argument("ngram_row_ids: table row count exceeds I32 addressing");
    }
    return prepared;
}

std::int32_t euclidean_remainder(std::uint64_t bits, std::int32_t modulus) {
    const std::int64_t signed_value = std::bit_cast<std::int64_t>(bits);
    std::int64_t result = signed_value % static_cast<std::int64_t>(modulus);
    if (result < 0) { result += modulus; }
    return static_cast<std::int32_t>(result);
}

void require_i32_contiguous(const Tensor& tensor, const char* name) {
    if (tensor.dtype != DType::I32 || !tensor.is_contiguous() || tensor.data == nullptr) {
        throw std::invalid_argument(std::string("ngram_row_ids: ") + name +
                                    " must be non-null contiguous I32");
    }
}

void require_shape(const Tensor& tensor, std::int32_t n0, std::int32_t n1, std::int32_t n2,
                   const char* name) {
    if (tensor.ne[0] != n0 || tensor.ne[1] != n1 || tensor.ne[2] != n2 || tensor.ne[3] != 1) {
        throw std::invalid_argument(std::string("ngram_row_ids: invalid shape for ") + name);
    }
}

struct ByteRange {
    std::uintptr_t begin;
    std::uintptr_t end;
};

ByteRange byte_range(const Tensor& tensor) {
    const auto begin = reinterpret_cast<std::uintptr_t>(tensor.data);
    const auto bytes = tensor.bytes();
    if (bytes > std::numeric_limits<std::uintptr_t>::max() - begin) {
        throw std::invalid_argument("ngram_row_ids: tensor address range overflows");
    }
    return {begin, begin + bytes};
}

bool overlaps(const Tensor& lhs, const Tensor& rhs) {
    const ByteRange a = byte_range(lhs);
    const ByteRange b = byte_range(rhs);
    return a.begin < b.end && b.begin < a.end;
}

void require_aliasing(const std::array<const Tensor*, 5>& tensors) {
    for (std::size_t i = 0; i < tensors.size(); ++i) {
        for (std::size_t j = i + 1; j < tensors.size(); ++j) {
            if (!overlaps(*tensors[i], *tensors[j])) { continue; }
            const bool history_pair = i == 2 && j == 4;
            const bool exact_alias = history_pair && tensors[i]->data == tensors[j]->data &&
                                     tensors[i]->bytes() == tensors[j]->bytes();
            if (!exact_alias) {
                throw std::invalid_argument("ngram_row_ids: tensor storage overlaps outside the "
                                            "exact old/new history alias");
            }
        }
    }
}

} // namespace

PreparedNgramRowConfig prepare_ngram_row_config(const NgramRowConfig& config) {
    return prepare_config(config);
}

NgramRowHostStep ngram_row_ids_host_step(
    std::int32_t input_id, const std::array<std::int32_t, 2>& old_history,
    const PreparedNgramRowConfig& config) {
    const auto valid_token = [&](std::int32_t token) {
        return token >= 0 && token < config.vocabulary_size;
    };
    if (config.vocabulary_size <= 0 || !valid_token(config.eos_token_id) ||
        !valid_token(input_id) || !valid_token(old_history[0]) ||
        !valid_token(old_history[1])) {
        throw std::invalid_argument("ngram_row_ids_host_step: token is outside the vocabulary");
    }
    const std::int32_t lag1 = old_history[1];
    const std::int32_t lag2 = lag1 == config.eos_token_id ? config.eos_token_id
                                                          : old_history[0];
    const std::uint64_t mixed2 =
        static_cast<std::uint64_t>(input_id) * config.multiplier[0] ^
        static_cast<std::uint64_t>(lag1) * config.multiplier[1];
    const std::uint64_t mixed3 =
        mixed2 ^ static_cast<std::uint64_t>(lag2) * config.multiplier[2];

    NgramRowHostStep result{};
    for (std::size_t head = 0; head < result.row_ids.size(); ++head) {
        if (config.prime[head] <= 0 || config.offset[head] < 0) {
            throw std::invalid_argument("ngram_row_ids_host_step: invalid prepared constants");
        }
        result.row_ids[head] = config.offset[head] +
                               euclidean_remainder(head < 8 ? mixed2 : mixed3,
                                                   config.prime[head]);
    }
    result.new_history = {lag1, input_id};
    return result;
}

void ngram_row_ids(const Tensor& input_ids, const Tensor& valid_tokens,
                   const Tensor& old_history, const NgramRowConfig& config, Tensor& row_ids,
                   Tensor& new_history, cudaStream_t stream) {
    require_i32_contiguous(input_ids, "input_ids");
    require_i32_contiguous(valid_tokens, "valid_tokens");
    require_i32_contiguous(old_history, "old_history");
    require_i32_contiguous(row_ids, "row_ids");
    require_i32_contiguous(new_history, "new_history");

    const std::int32_t width    = input_ids.ne[0];
    const std::int32_t requests = input_ids.ne[1];
    if (width <= 0 || requests <= 0) {
        throw std::invalid_argument("ngram_row_ids: W and C must be positive");
    }
    require_shape(input_ids, width, requests, 1, "input_ids");
    require_shape(valid_tokens, requests, 1, 1, "valid_tokens");
    require_shape(old_history, 2, requests, 1, "old_history");
    require_shape(row_ids, 16, width, requests, "row_ids");
    require_shape(new_history, 2, requests, 1, "new_history");
    require_aliasing({&input_ids, &valid_tokens, &old_history, &row_ids, &new_history});

    const PreparedNgramRowConfig prepared = prepare_config(config);
    detail::ngram_row_ids_launch(input_ids, valid_tokens, old_history, prepared, row_ids,
                                 new_history, stream);
}

} // namespace ninfer::ops
