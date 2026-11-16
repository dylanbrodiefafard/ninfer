#pragma once

#include <cuda_runtime.h>

#include <cstdint>

namespace ninfer::ops::detail {

// The launchers trust the wrapper's validation; pointers are 16-byte aligned device addresses
// (rows 2-byte aligned) and every launch covers 1..65535 tokens (the grid's y extent).

void ple_decode_rows_launch(const std::uint8_t* rows, std::uint16_t* embedding, std::int32_t tokens,
                            cudaStream_t stream);

// One ple_inject launch sequence over columns [chunk_begin, chunk_begin + chunk_columns) of every
// row of a [*, width, batch] packed call. Packed column (j, b) is element b * width + j; launch
// token t covers row t / chunk_columns, column chunk_begin + t % chunk_columns. valid_columns is
// null (every column valid) or device I32 [batch].
struct PleInjectChunk {
    std::int32_t width                = 0;
    std::int32_t batch                = 0;
    std::int32_t chunk_begin          = 0;
    std::int32_t chunk_columns        = 0;
    const std::int32_t* valid_columns = nullptr;
    // Device I32 [batch]: the history slot each row reads, and the slot the final history is
    // written to (ple_inject, the same selector) or the base of its per-column snapshots.
    const std::int32_t* initial_slots = nullptr;
    const std::int32_t* publish_slots = nullptr;
};

enum class PleStatePublication : std::uint8_t {
    Final,
    EveryColumn,
};

// conv_input is BF16 workspace [batch * chunk_columns][10240] and gate FP32 workspace
// [batch * chunk_columns][4]. conv_states is BF16 [9,10240,slots].
void ple_inject_launch(std::uint16_t* residual, const float* key, const float* value,
                       const std::uint16_t* norm_key, const std::uint16_t* norm_query,
                       const std::uint16_t* norm_conv, const std::uint16_t* conv_weight,
                       std::uint16_t* conv_states, std::uint16_t* conv_input, float* gate,
                       const PleInjectChunk& chunk, PleStatePublication publication,
                       cudaStream_t stream);

} // namespace ninfer::ops::detail
