#include "ops/gdn_input_proj/nvfp4/nvfp4_gdn_snapshot_plan.h"

#include "core/device.h"
#include "ops/gdn_input_proj/nvfp4/nvfp4_gdn_snapshot_output.cuh"
#include "ops/linear/nvfp4/nvfp4_config.h"
#include "ops/linear/nvfp4/nvfp4_small_t.cuh"

#include <array>
#include <cstddef>
#include <utility>

namespace ninfer::ops::detail {
namespace {

using Launch = void (*)(const Tensor&, const Weight&, const Tensor&, Tensor&, const Tensor&,
                        const Tensor&, const Tensor&, Tensor&, Tensor&, Tensor&, Tensor&,
                        cudaStream_t);

template <int ActiveTokens>
void launch_exact(const Tensor& x, const Weight& weight, const Tensor& conv_weight,
                  Tensor& conv_states, const Tensor& valid_columns, const Tensor& initial_slot,
                  const Tensor& snapshot_base_slot, Tensor& query, Tensor& key, Tensor& value,
                  Tensor& z, cudaStream_t stream) {
    using Geometry = Nvfp4GdnInputGeometry;
    using Schedule = typename Nvfp4LinearSmallTProductionSchedule<Geometry, ActiveTokens>::Type;
    static_assert(Schedule::kTokenTile == ActiveTokens);

    constexpr int kBlocks = Geometry::kOutputRows / Schedule::kRowsPerCta;
    const float inverse   = 1.0F / weight.weight_scale_divisor;
    nvfp4_small_t_kernel<Geometry, ActiveTokens, Schedule, Nvfp4IdentityEpilogue,
                         Nvfp4GdnSnapshotOutput<ActiveTokens>, Nvfp4SmallTFinalization::RowVector>
        <<<kBlocks, Schedule::kThreads, 0, stream>>>(
            static_cast<const __nv_bfloat16*>(x.data),
            static_cast<const std::uint8_t*>(weight.qdata),
            static_cast<const std::uint8_t*>(weight.scales), inverse, Nvfp4IdentityEpilogue{},
            make_nvfp4_gdn_snapshot_output<ActiveTokens>(conv_weight, conv_states, valid_columns,
                                                         initial_slot, snapshot_base_slot, query,
                                                         key, value, z));
    CUDA_CHECK(cudaGetLastError());
}

template <std::size_t... Offsets>
constexpr auto make_launchers(std::index_sequence<Offsets...>) {
    return std::array<Launch, sizeof...(Offsets)>{
        &launch_exact<kNvfp4FirstSmallT + static_cast<int>(Offsets)>...};
}

constexpr auto kLaunchers = make_launchers(std::make_index_sequence<16 - kNvfp4FirstSmallT + 1>{});

} // namespace

void nvfp4_gdn_snapshot_small_t_launch(const Tensor& x, const Weight& weight,
                                       const Tensor& conv_weight, Tensor& conv_states,
                                       const Tensor& valid_columns, const Tensor& initial_slot,
                                       const Tensor& snapshot_base_slot, Tensor& query, Tensor& key,
                                       Tensor& value, Tensor& z, cudaStream_t stream) {
    const std::size_t index = static_cast<std::size_t>(x.ne[1] - kNvfp4FirstSmallT);
    kLaunchers[index](x, weight, conv_weight, conv_states, valid_columns, initial_slot,
                      snapshot_base_slot, query, key, value, z, stream);
}

} // namespace ninfer::ops::detail
