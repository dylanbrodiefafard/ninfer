#pragma once

#include "ops/common/math.h"
#include "ops/linear_attention/gated_delta_net/common.h"

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cstdint>
#include <cstdio>

#define NINFER_GATED_DELTA_NET_PROPAGATE(expr)                                                     \
    do {                                                                                           \
        const cudaError_t ninfer_gated_delta_net_error = (expr);                                   \
        if (ninfer_gated_delta_net_error != cudaSuccess) { return ninfer_gated_delta_net_error; }  \
    } while (0)

namespace ninfer::ops::detail::gated_delta_net::chunked {

inline constexpr std::int64_t kWorkspaceAlign = 256;

struct workspace_layout {
    std::int64_t g_cumsum_off   = 0;
    std::int64_t g_cumsum_bytes = 0;
    std::int64_t W_off          = 0;
    std::int64_t W_bytes        = 0;
    std::int64_t U_off          = 0;
    std::int64_t U_bytes        = 0;
    std::int64_t v_new_off      = 0;
    std::int64_t v_new_bytes    = 0;
    std::int64_t h_chunk_off    = 0;
    std::int64_t h_chunk_bytes  = 0;
    std::int64_t total_bytes    = 0;
};

inline std::int64_t reserve(std::int64_t& cursor, std::int64_t bytes) {
    if (bytes == 0) { return cursor; }
    const std::int64_t off = cursor;
    cursor                 = ninfer::ops::align_up<kWorkspaceAlign>(off + bytes);
    return off;
}

inline workspace_layout compute_workspace_layout(std::int64_t S, std::int64_t H_qk,
                                                 std::int64_t H_v, std::int64_t L, std::int64_t B) {
    (void)H_qk;
    constexpr std::int64_t f    = static_cast<std::int64_t>(sizeof(float));
    constexpr std::int64_t bf16 = static_cast<std::int64_t>(sizeof(__nv_bfloat16));
    const std::int64_t T        = L;
    const std::int64_t NT       = div_up(T, static_cast<std::int64_t>(kChunkTokens));

    const std::int64_t per_token_g   = B * T * H_v;
    const std::int64_t per_token_S   = B * T * H_v * S;
    const std::int64_t per_chunk_SxS = B * NT * H_v * S * S;

    workspace_layout w{};
    w.g_cumsum_bytes = per_token_g * f;
    w.W_bytes        = per_token_S * bf16;
    w.U_bytes        = per_token_S * bf16;
    w.v_new_bytes    = per_token_S * bf16;
    w.h_chunk_bytes  = per_chunk_SxS * bf16;

    std::int64_t cur = 0;
    w.g_cumsum_off   = reserve(cur, w.g_cumsum_bytes);
    w.W_off          = reserve(cur, w.W_bytes);
    w.U_off          = reserve(cur, w.U_bytes);
    w.v_new_off      = reserve(cur, w.v_new_bytes);
    w.h_chunk_off    = reserve(cur, w.h_chunk_bytes);
    w.total_bytes    = cur;
    return w;
}

inline std::int64_t workspace_bytes(std::int64_t S, std::int64_t H_qk, std::int64_t H_v,
                                    std::int64_t L, std::int64_t B) {
    return compute_workspace_layout(S, H_qk, H_v, L, B).total_bytes;
}

struct prepare_wy_wu_config {
    std::int64_t S    = 0;
    std::int64_t H_qk = 0;
    std::int64_t H_v  = 0;
    std::int64_t L    = 0;
    std::int64_t B    = 0;

    const __nv_bfloat16* k = nullptr;
    const __nv_bfloat16* v = nullptr;
    const float* g_in      = nullptr;
    const float* beta      = nullptr;

    __nv_bfloat16* W    = nullptr;
    __nv_bfloat16* U    = nullptr;
    float* g_cumsum_out = nullptr;

    std::int64_t k_stride_t_floats = 0;
    std::int64_t v_stride_t_floats = 0;

    cudaStream_t stream = nullptr;
};

struct state_passing_config {
    std::int64_t S    = 0;
    std::int64_t H_qk = 0;
    std::int64_t H_v  = 0;
    std::int64_t L    = 0;
    std::int64_t B    = 0;

    const __nv_bfloat16* W = nullptr;
    const __nv_bfloat16* U = nullptr;
    const __nv_bfloat16* k = nullptr;
    const float* g_cumsum  = nullptr;
    const float* state_in  = nullptr;

    __nv_bfloat16* v_new   = nullptr;
    __nv_bfloat16* h_chunk = nullptr;
    float* state_out       = nullptr;

    std::int64_t k_stride_t_floats = 0;

    cudaStream_t stream = nullptr;
};

struct chunk_output_config {
    std::int64_t S    = 0;
    std::int64_t H_qk = 0;
    std::int64_t H_v  = 0;
    std::int64_t L    = 0;
    std::int64_t B    = 0;

    const __nv_bfloat16* q       = nullptr;
    const __nv_bfloat16* k       = nullptr;
    const __nv_bfloat16* v_new   = nullptr;
    const float* g_cumsum        = nullptr;
    const __nv_bfloat16* h_chunk = nullptr;

    __nv_bfloat16* attn_out = nullptr;

    std::int64_t q_stride_t_floats = 0;
    std::int64_t k_stride_t_floats = 0;
    float scale                    = 0.0f;

    cudaStream_t stream = nullptr;
};

struct stage_validator {
    const char* name;
    std::int64_t S;
    std::int64_t H_qk;
    std::int64_t H_v;
    std::int64_t T;
    std::int64_t B;
    bool require_h_qk = true;

    cudaError_t check_shape() const {
        const bool bad_shape =
            S <= 0 || H_v <= 0 || T <= 0 || B <= 0 || (require_h_qk && H_qk <= 0);
        if (bad_shape) {
            std::fprintf(stderr, "%s: invalid shape (S=%lld H_qk=%lld H_v=%lld T=%lld B=%lld)\n",
                         name, static_cast<long long>(S), static_cast<long long>(H_qk),
                         static_cast<long long>(H_v), static_cast<long long>(T),
                         static_cast<long long>(B));
            return cudaErrorInvalidValue;
        }
        if (!is_supported_head_dim(S)) {
            std::fprintf(stderr, "%s: unsupported S=%lld (allowed: 16, 32, 64, 128)\n", name,
                         static_cast<long long>(S));
            return cudaErrorInvalidValue;
        }
        if (require_h_qk && !are_head_counts_valid(H_qk, H_v)) {
            std::fprintf(stderr,
                         "%s: invalid head counts H_qk=%lld H_v=%lld "
                         "(need H_qk >= 1, H_v >= H_qk, H_v %% H_qk == 0)\n",
                         name, static_cast<long long>(H_qk), static_cast<long long>(H_v));
            return cudaErrorInvalidValue;
        }
        return cudaSuccess;
    }

    cudaError_t check_full_chunks() const {
        if ((T % kChunkTokens) != 0) {
            std::fprintf(stderr,
                         "%s: Gated DeltaNet chunked path requires T to be a multiple of %d; "
                         "route tail tokens through AR instead (T=%lld)\n",
                         name, kChunkTokens, static_cast<long long>(T));
            return cudaErrorInvalidValue;
        }
        return cudaSuccess;
    }

    cudaError_t check_grid(std::int64_t grid_x, std::int64_t grid_y,
                           std::int64_t grid_z = 1) const {
        if (grid_x > static_cast<std::int64_t>(0xffffffff)) {
            std::fprintf(stderr, "%s: grid.x too large (%lld)\n", name,
                         static_cast<long long>(grid_x));
            return cudaErrorInvalidConfiguration;
        }
        if (grid_y > static_cast<std::int64_t>(0xffff)) {
            std::fprintf(stderr, "%s: grid.y too large (%lld)\n", name,
                         static_cast<long long>(grid_y));
            return cudaErrorInvalidConfiguration;
        }
        if (grid_z > static_cast<std::int64_t>(0xffff)) {
            std::fprintf(stderr, "%s: grid.z too large (%lld)\n", name,
                         static_cast<long long>(grid_z));
            return cudaErrorInvalidConfiguration;
        }
        return cudaSuccess;
    }
};

cudaError_t launch_prepare_wy_wu(const prepare_wy_wu_config& cfg);
cudaError_t launch_state_passing(const state_passing_config& cfg);
cudaError_t launch_output(const chunk_output_config& cfg);

} // namespace ninfer::ops::detail::gated_delta_net::chunked
