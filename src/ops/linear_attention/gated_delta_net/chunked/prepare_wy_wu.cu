#include "ops/linear_attention/gated_delta_net/chunked/launch.h"
#include "ops/linear_attention/gated_delta_net/chunked/prepare_wy_wu.cuh"

namespace ninfer::ops::detail::gated_delta_net::chunked {
namespace {

namespace kernel = prepare_wy_wu;

template <int S>
cudaError_t launch_typed(const prepare_wy_wu_config& cfg, dim3 grid, dim3 block, head_map qk_map) {
    constexpr int smem_bytes = kernel::kernel_dims<S>::SMEM_FLOATS * (int)sizeof(float);

    cudaError_t err = cudaFuncSetAttribute(kernel::prepare_wy_wu_kernel<S>,
                                           cudaFuncAttributeMaxDynamicSharedMemorySize, smem_bytes);
    if (err != cudaSuccess) return err;

    // 0 means "use packed default" -- materialise here so the kernel only
    // ever sees a real stride.
    const int64_t k_stride_t =
        (cfg.k_stride_t_floats != 0) ? cfg.k_stride_t_floats : (int64_t)cfg.H_qk * cfg.S;
    const int64_t v_stride_t =
        (cfg.v_stride_t_floats != 0) ? cfg.v_stride_t_floats : (int64_t)cfg.H_v * cfg.S;

    kernel::prepare_wy_wu_kernel<S><<<grid, block, smem_bytes, cfg.stream>>>(
        cfg.k, cfg.v, cfg.g_in, cfg.beta, cfg.W, cfg.U, cfg.g_cumsum_out, cfg.L, cfg.H_v, qk_map,
        k_stride_t, v_stride_t);
    return cudaGetLastError();
}

} // namespace

cudaError_t launch_prepare_wy_wu(const prepare_wy_wu_config& cfg) {
    stage_validator v{"launch_prepare_wy_wu", cfg.S, cfg.H_qk, cfg.H_v, cfg.L, cfg.B};
    NINFER_GATED_DELTA_NET_PROPAGATE(v.check_shape());
    NINFER_GATED_DELTA_NET_PROPAGATE(v.check_full_chunks());
    if (cfg.k == nullptr || cfg.v == nullptr || cfg.g_in == nullptr || cfg.beta == nullptr ||
        cfg.W == nullptr || cfg.U == nullptr || cfg.g_cumsum_out == nullptr) {
        return cudaErrorInvalidValue;
    }

    const auto qk_map = head_map::of((int)cfg.H_qk, (int)cfg.H_v);
    const int64_t NT  = div_up(cfg.L, static_cast<int64_t>(BT));
    const int64_t bh  = cfg.B * cfg.H_v;
    NINFER_GATED_DELTA_NET_PROPAGATE(v.check_grid(NT, bh));

    const dim3 grid((unsigned)NT, (unsigned)bh, 1);
    const dim3 block(kernel::THREADS, 1, 1);

    switch (cfg.S) {
    case 16:
        return launch_typed<16>(cfg, grid, block, qk_map);
    case 32:
        return launch_typed<32>(cfg, grid, block, qk_map);
    case 64:
        return launch_typed<64>(cfg, grid, block, qk_map);
    case 128:
        return launch_typed<128>(cfg, grid, block, qk_map);
    default:
        return cudaErrorInvalidValue; // check_shape already filtered
    }
}

} // namespace ninfer::ops::detail::gated_delta_net::chunked
