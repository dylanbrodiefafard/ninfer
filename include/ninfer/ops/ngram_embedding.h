#pragma once

#include "core/tensor.h"

#include <cuda_runtime.h>

#include <array>
#include <cstdint>

namespace ninfer::ops {

struct NgramRowConfig {
    std::int32_t vocabulary_size = 0;
    std::int32_t eos_token_id    = 0;
    std::int32_t ple_layer_index = 0;
    std::uint64_t seed           = 0;
    std::int32_t vocab_base      = 0;
};

/** Host-prepared exact integer constants shared by the CUDA and C=1 host addressing routes. */
struct PreparedNgramRowConfig {
    std::int32_t vocabulary_size;
    std::int32_t eos_token_id;
    std::uint64_t multiplier[3];
    std::int32_t prime[16];
    std::int32_t offset[16];
};

/** Exact result of one C=1/T=1 host addressing step. */
struct NgramRowHostStep {
    std::array<std::int32_t, 16> row_ids;
    std::array<std::int32_t, 2> new_history;
};

/** Validate and prepare the integer constants for repeated host or CUDA n-gram addressing. */
[[nodiscard]] PreparedNgramRowConfig prepare_ngram_row_config(const NgramRowConfig& config);

/**
 * Exact C=1/T=1 host n-gram addressing route.
 *
 * This performs only integer hashing, Euclidean remainder, and raw token-history advancement.
 * It exists for artifact-backed row addressing; it does not decode or execute floating-point
 * model work. old_history is oldest-to-newest. The current EOS is retained and resets the missing
 * second predecessor only on the following step.
 */
[[nodiscard]] NgramRowHostStep
ngram_row_ids_host_step(std::int32_t input_id,
                        const std::array<std::int32_t, 2>& old_history,
                        const PreparedNgramRowConfig& config);

/**
 * Op: ngram_row_ids
 *
 * Math / indexing:
 *   For each request, process the valid input prefix in token order. For the current token a0,
 *   a1 is the preceding raw token and a2 is the token before a1 unless a1 is EOS; a missing or
 *   reset lag is EOS. The current EOS participates normally and resets only the following token.
 *   For n=2,3, mixed_n is the XOR of the first n unsigned-64 wrapped products a_j*m[j]. The
 *   eight rows for each order are Euclidean-remainder(mixed_n reinterpreted as signed I64,
 *   prime[head]) plus that head's layer-local exclusive prime-prefix offset.
 *
 *   splitmix64 and the odd multipliers are exactly those in
 *   docs/maintainer/qwen4-op-contracts.md section 6. Each head prime is selected from the global
 *   increasing sequence of primes strictly after vocab_base-1, beginning at global head
 *   ple_layer_index*16. The preview has one PLE module, so the only admitted module index is zero.
 *   Head order is eight bigram heads followed by eight trigram heads.
 *
 * Logical shapes:
 *   Contiguous I32 input_ids [W,C], valid_tokens [C], old_history/new_history [2,C] in
 *   oldest-to-newest order, and row_ids [16,W,C]. W and C are positive. Each valid_tokens[c] is
 *   in [0,W]. Input/history token values are in [0,vocabulary_size); EOS is in that interval.
 *
 * Supported domain:
 *   The preview geometry N=3 and P=8. vocabulary_size and vocab_base are positive;
 *   ple_layer_index is exactly zero, the preview's sole zero-based PLE-module index. The generated
 *   primes, their module-local exclusive prefix sums, and all row ids must fit nonnegative I32.
 *
 * Numeric:
 *   Exact I32 output and state. Unsigned arithmetic wraps modulo 2^64. The hash is reinterpreted
 *   as signed two's-complement I64 before Euclidean remainder by each positive prime.
 *
 * Effects:
 *   Writes all row_ids: valid columns receive rows and invalid suffix columns receive -1.
 *   new_history receives the last two raw tokens of old_history followed by the valid input,
 *   retaining EOS and left-padding missing history with EOS. old_history and new_history may be
 *   disjoint or alias exactly; every other pair of tensor storage regions must be disjoint.
 *   No frontier is advanced and no state is committed by the Op.
 *
 * Workspace:
 *   None. The call is asynchronous on stream and is graph-capturable for fixed tensor shapes and
 *   configuration.
 */
void ngram_row_ids(const Tensor& input_ids, const Tensor& valid_tokens,
                   const Tensor& old_history, const NgramRowConfig& config, Tensor& row_ids,
                   Tensor& new_history, cudaStream_t stream);

} // namespace ninfer::ops
