#include "ops/linear_attention/gated_delta_net/chunked/launch.h"
#include "ops/linear_attention/gated_delta_net/chunked/output.cuh"

namespace ninfer::ops::detail::gated_delta_net::chunked {
namespace {

namespace kernel = output;

constexpr std::int64_t kRtx5090SmCount = 170;
constexpr std::int64_t kCtasPerSm      = 4;
constexpr std::int64_t kTargetCtas     = kRtx5090SmCount * kCtasPerSm;

template <bool PRIVATE_F16, bool NORMALIZE_Q, bool MULTI_JOB>
cudaError_t launch_fixed(const chunk_output_config& cfg, dim3 grid, head_map qk_map, int chunks) {
    constexpr int smem_bytes = kernel::kernel_dims::SMEM_BYTES;

    cudaError_t err = cudaFuncSetAttribute(kernel::output_kernel<PRIVATE_F16, NORMALIZE_Q, MULTI_JOB>,
                                           cudaFuncAttributeMaxDynamicSharedMemorySize, smem_bytes);
    if (err != cudaSuccess) { return err; }

    const dim3 block(kernel::THREADS, 1, 1);

    kernel::output_kernel<PRIVATE_F16, NORMALIZE_Q, MULTI_JOB>
        <<<grid, block, smem_bytes, cfg.stream>>>(
        cfg.q, cfg.k, cfg.v_new, cfg.g_cumsum, cfg.h_chunk, cfg.attn_out, qk_map, cfg.scale,
        chunks);
    return cudaGetLastError();
}

} // namespace

cudaError_t launch_output(const chunk_output_config& cfg) {
    stage_validator v{"launch_output", cfg.H_qk, cfg.H_v, cfg.L};
    NINFER_GATED_DELTA_NET_PROPAGATE(v.check_shape());
    NINFER_GATED_DELTA_NET_PROPAGATE(v.check_full_chunks());
    if (cfg.q == nullptr || cfg.k == nullptr || cfg.v_new == nullptr || cfg.g_cumsum == nullptr ||
        cfg.h_chunk == nullptr || cfg.attn_out == nullptr) {
        return cudaErrorInvalidValue;
    }
    if (cfg.normalize_q && (!cfg.private_fp16 || cfg.H_qk != cfg.H_v)) {
        return cudaErrorInvalidValue;
    }

    const auto qk_map     = head_map::of((int)cfg.H_qk, (int)cfg.H_v);
    const std::int64_t NT = cfg.L / BT;

    // Keep at most one resident RTX 5090 wave and distribute chunks evenly
    // across it. Small grids retain one logical job per CTA.
    const std::int64_t logical_jobs   = NT * cfg.H_v;
    const std::int64_t jobs_per_block = (logical_jobs + kTargetCtas - 1) / kTargetCtas;
    const std::int64_t grid_chunks    = (NT + jobs_per_block - 1) / jobs_per_block;
    NINFER_GATED_DELTA_NET_PROPAGATE(v.check_grid(grid_chunks, cfg.H_v));

    const dim3 grid(static_cast<unsigned>(grid_chunks), static_cast<unsigned>(cfg.H_v), 1);
    if (jobs_per_block == 1) {
        if (cfg.normalize_q) {
            return launch_fixed<true, true, false>(cfg, grid, qk_map, static_cast<int>(NT));
        }
        return cfg.private_fp16
                   ? launch_fixed<true, false, false>(cfg, grid, qk_map, static_cast<int>(NT))
                   : launch_fixed<false, false, false>(cfg, grid, qk_map, static_cast<int>(NT));
    }
    if (cfg.normalize_q) {
        return launch_fixed<true, true, true>(cfg, grid, qk_map, static_cast<int>(NT));
    }
    return cfg.private_fp16
               ? launch_fixed<true, false, true>(cfg, grid, qk_map, static_cast<int>(NT))
               : launch_fixed<false, false, true>(cfg, grid, qk_map, static_cast<int>(NT));
}

} // namespace ninfer::ops::detail::gated_delta_net::chunked
