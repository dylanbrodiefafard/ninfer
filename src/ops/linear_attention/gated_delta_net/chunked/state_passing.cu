#include "ops/linear_attention/gated_delta_net/chunked/launch.h"
#include "ops/linear_attention/gated_delta_net/chunked/state_passing.cuh"

namespace ninfer::ops::detail::gated_delta_net::chunked {
namespace {

namespace kernel = state_passing;

template <int S>
cudaError_t launch_typed(const state_passing_config& cfg, head_map qk_map, int NT) {
    using D                  = kernel::kernel_dims<S>;
    constexpr int smem_bytes = kernel::smem_layout<S>::SMEM_FLOATS * (int)sizeof(float);

    cudaError_t err = cudaFuncSetAttribute(kernel::state_passing_kernel<S>,
                                           cudaFuncAttributeMaxDynamicSharedMemorySize, smem_bytes);
    if (err != cudaSuccess) return err;

    const dim3 grid((unsigned)(cfg.B * cfg.H_v * D::D_STRIPS), 1, 1);
    const dim3 block(D::THREADS, 1, 1);

    const int64_t k_stride_t =
        (cfg.k_stride_t_floats != 0) ? cfg.k_stride_t_floats : (int64_t)cfg.H_qk * cfg.S;

    kernel::state_passing_kernel<S><<<grid, block, smem_bytes, cfg.stream>>>(
        cfg.W, cfg.U, cfg.k, cfg.g_cumsum, cfg.state_in, cfg.v_new, cfg.h_chunk, cfg.state_out,
        cfg.L, cfg.H_v, qk_map, k_stride_t, NT);
    return cudaGetLastError();
}

} // namespace

cudaError_t launch_state_passing(const state_passing_config& cfg) {
    stage_validator v{"launch_state_passing", cfg.S, cfg.H_qk, cfg.H_v, cfg.L, cfg.B};
    NINFER_GATED_DELTA_NET_PROPAGATE(v.check_shape());
    NINFER_GATED_DELTA_NET_PROPAGATE(v.check_full_chunks());
    if (cfg.W == nullptr || cfg.U == nullptr || cfg.k == nullptr || cfg.g_cumsum == nullptr ||
        cfg.state_in == nullptr || cfg.v_new == nullptr || cfg.h_chunk == nullptr ||
        cfg.state_out == nullptr) {
        return cudaErrorInvalidValue;
    }

    const auto qk_map = head_map::of((int)cfg.H_qk, (int)cfg.H_v);
    const int64_t NT  = div_up(cfg.L, static_cast<int64_t>(BT));

    // grid_x = B * H_v * D_STRIPS depends on S; check inside each case.
    auto check_grid_for = [&](int d_strips) -> cudaError_t {
        return v.check_grid(cfg.B * cfg.H_v * d_strips, /*grid_y=*/1, /*grid_z=*/1);
    };

    switch (cfg.S) {
    case 16:
        NINFER_GATED_DELTA_NET_PROPAGATE(check_grid_for(kernel::kernel_dims<16>::D_STRIPS));
        return launch_typed<16>(cfg, qk_map, (int)NT);
    case 32:
        NINFER_GATED_DELTA_NET_PROPAGATE(check_grid_for(kernel::kernel_dims<32>::D_STRIPS));
        return launch_typed<32>(cfg, qk_map, (int)NT);
    case 64:
        NINFER_GATED_DELTA_NET_PROPAGATE(check_grid_for(kernel::kernel_dims<64>::D_STRIPS));
        return launch_typed<64>(cfg, qk_map, (int)NT);
    case 128:
        NINFER_GATED_DELTA_NET_PROPAGATE(check_grid_for(kernel::kernel_dims<128>::D_STRIPS));
        return launch_typed<128>(cfg, qk_map, (int)NT);
    default:
        return cudaErrorInvalidValue; // check_shape already filtered
    }
}

} // namespace ninfer::ops::detail::gated_delta_net::chunked
