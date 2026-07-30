#include "ops/linear_attention/gated_delta_net/chunked/launch.h"
#include "ops/linear_attention/gated_delta_net/chunked/output.cuh"

namespace ninfer::ops::detail::gated_delta_net::chunked {
namespace {

namespace kernel = output;

template <int S>
cudaError_t launch_typed(const chunk_output_config& cfg, dim3 grid, head_map qk_map, int NT,
                         float scale) {
    constexpr int smem_bytes = kernel::kernel_dims<S>::SMEM_FLOATS * (int)sizeof(float);

    cudaError_t err = cudaFuncSetAttribute(kernel::output_kernel<S>,
                                           cudaFuncAttributeMaxDynamicSharedMemorySize, smem_bytes);
    if (err != cudaSuccess) return err;

    const dim3 block(kernel::THREADS, 1, 1);

    const int64_t q_stride_t =
        (cfg.q_stride_t_floats != 0) ? cfg.q_stride_t_floats : (int64_t)cfg.H_qk * cfg.S;
    const int64_t k_stride_t =
        (cfg.k_stride_t_floats != 0) ? cfg.k_stride_t_floats : (int64_t)cfg.H_qk * cfg.S;

    kernel::output_kernel<S><<<grid, block, smem_bytes, cfg.stream>>>(
        cfg.q, cfg.k, cfg.v_new, cfg.g_cumsum, cfg.h_chunk, cfg.attn_out, cfg.L, cfg.H_v, qk_map,
        q_stride_t, k_stride_t, NT, scale);
    return cudaGetLastError();
}

} // namespace

cudaError_t launch_output(const chunk_output_config& cfg) {
    stage_validator v{"launch_output", cfg.S, cfg.H_qk, cfg.H_v, cfg.L, cfg.B};
    NINFER_GATED_DELTA_NET_PROPAGATE(v.check_shape());
    NINFER_GATED_DELTA_NET_PROPAGATE(v.check_full_chunks());
    if (cfg.q == nullptr || cfg.v_new == nullptr || cfg.g_cumsum == nullptr ||
        cfg.h_chunk == nullptr || cfg.attn_out == nullptr) {
        return cudaErrorInvalidValue;
    }
    if (cfg.k == nullptr) return cudaErrorInvalidValue;

    const auto qk_map = head_map::of((int)cfg.H_qk, (int)cfg.H_v);
    const int64_t NT  = div_up(cfg.L, static_cast<int64_t>(BT));
    const int64_t bh  = cfg.B * cfg.H_v;
    const float scale = cfg.scale;
    NINFER_GATED_DELTA_NET_PROPAGATE(v.check_grid(NT, bh));

    const dim3 grid((unsigned)NT, (unsigned)bh, 1);

    switch (cfg.S) {
    case 16:
        return launch_typed<16>(cfg, grid, qk_map, (int)NT, scale);
    case 32:
        return launch_typed<32>(cfg, grid, qk_map, (int)NT, scale);
    case 64:
        return launch_typed<64>(cfg, grid, qk_map, (int)NT, scale);
    case 128:
        return launch_typed<128>(cfg, grid, qk_map, (int)NT, scale);
    default:
        return cudaErrorInvalidValue; // check_shape already filtered
    }
}

} // namespace ninfer::ops::detail::gated_delta_net::chunked
