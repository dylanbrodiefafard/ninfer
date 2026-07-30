#include "ops/linear_attention/gated_delta_net/chunked/launch.h"
#include "ops/linear_attention/gated_delta_net/chunked/prepare_wy_wu.cuh"

namespace ninfer::ops::detail::gated_delta_net::chunked {
namespace {

namespace kernel = prepare_wy_wu;

cudaError_t launch_fixed(const prepare_wy_wu_config& cfg, dim3 grid, dim3 block, head_map qk_map) {
    constexpr int smem_bytes = kernel::kernel_dims::SMEM_FLOATS * static_cast<int>(sizeof(float));

    cudaError_t err = cudaFuncSetAttribute(kernel::prepare_wy_wu_kernel,
                                           cudaFuncAttributeMaxDynamicSharedMemorySize, smem_bytes);
    if (err != cudaSuccess) { return err; }

    kernel::prepare_wy_wu_kernel<<<grid, block, smem_bytes, cfg.stream>>>(
        cfg.k, cfg.v, cfg.g_in, cfg.beta, cfg.W, cfg.U, cfg.g_cumsum_out, qk_map);
    return cudaGetLastError();
}

} // namespace

cudaError_t launch_prepare_wy_wu(const prepare_wy_wu_config& cfg) {
    stage_validator v{"launch_prepare_wy_wu", cfg.H_qk, cfg.H_v, cfg.L};
    NINFER_GATED_DELTA_NET_PROPAGATE(v.check_shape());
    NINFER_GATED_DELTA_NET_PROPAGATE(v.check_full_chunks());
    if (cfg.k == nullptr || cfg.v == nullptr || cfg.g_in == nullptr || cfg.beta == nullptr ||
        cfg.W == nullptr || cfg.U == nullptr || cfg.g_cumsum_out == nullptr) {
        return cudaErrorInvalidValue;
    }

    const auto qk_map     = head_map::of((int)cfg.H_qk, (int)cfg.H_v);
    const std::int64_t NT = cfg.L / BT;
    NINFER_GATED_DELTA_NET_PROPAGATE(v.check_grid(NT, cfg.H_v));

    const dim3 grid(static_cast<unsigned>(NT), static_cast<unsigned>(cfg.H_v), 1);
    const dim3 block(kernel::THREADS, 1, 1);
    return launch_fixed(cfg, grid, block, qk_map);
}

} // namespace ninfer::ops::detail::gated_delta_net::chunked
