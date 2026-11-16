#include "ops/attn_input_proj/nvfp4/nvfp4_attn_input_plan.h"

#include "core/device.h"
#include "ops/attn_input_proj/nvfp4/nvfp4_attn_input_output.cuh"
#include "ops/linear/nvfp4/nvfp4_config.h"
#include "ops/linear/nvfp4/nvfp4_small_t.cuh"

#include <array>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <utility>

namespace ninfer::ops::detail {
namespace {

using Launch = void (*)(const Tensor&, const Weight&, Tensor&, Tensor&, Tensor&, Tensor&,
                        cudaStream_t);

// RTX 5090 cold-cache winner for the four-output epilogue at the A16 widths below the W4A4
// cutover: eight-warp CTAs with the activation columns staged in shared memory.
template <int ActiveTokens>
struct Nvfp4AttentionSmallTProductionSchedule {
    static_assert(ActiveTokens >= kNvfp4FirstSmallT);
    static_assert(ActiveTokens < kNvfp4FirstW4a4AttnInput);
    using Type =
        Nvfp4SmallTSchedule<8, 1, 2, 16, ActiveTokens, 1, Nvfp4SmallTActivationAccess::SharedPhase,
                            Nvfp4ScaleAccess::Direct, Nvfp4CodeCache::Default, 1,
                            Nvfp4SmallTBlockOrder::RowsContiguous, 1>;
};

template <int ActiveTokens>
void launch_exact(const Tensor& x, const Weight& weight, Tensor& q, Tensor& gate, Tensor& k,
                  Tensor& v, cudaStream_t stream) {
    using Geometry            = Nvfp4AttnInputGeometry;
    using Schedule            = typename Nvfp4AttentionSmallTProductionSchedule<ActiveTokens>::Type;
    constexpr int kTokenTiles = (ActiveTokens + Schedule::kTokenTile - 1) / Schedule::kTokenTile;
    constexpr int kBlocks     = (Geometry::kOutputRows / Schedule::kRowsPerCta) * kTokenTiles;

    const Nvfp4AttentionInput5120Output output{
        static_cast<__nv_bfloat16*>(q.data),
        static_cast<__nv_bfloat16*>(k.data),
        static_cast<__nv_bfloat16*>(gate.data),
        static_cast<__nv_bfloat16*>(v.data),
    };
    const float inverse_weight_divisor = 1.0F / weight.weight_scale_divisor;
    nvfp4_small_t_kernel<Geometry, ActiveTokens, Schedule>
        <<<kBlocks, Schedule::kThreads, 0, stream>>>(
            Nvfp4PackedActivation<Geometry>{static_cast<const __nv_bfloat16*>(x.data)},
            static_cast<const std::uint8_t*>(weight.qdata),
            static_cast<const std::uint8_t*>(weight.scales), inverse_weight_divisor,
            Nvfp4IdentityEpilogue{}, output);
    CUDA_CHECK(cudaGetLastError());
}

template <std::size_t... Offsets>
constexpr auto make_launchers(std::index_sequence<Offsets...>) {
    return std::array<Launch, sizeof...(Offsets)>{
        &launch_exact<kNvfp4FirstSmallT + static_cast<int>(Offsets)>...};
}

constexpr auto kLaunchers =
    make_launchers(std::make_index_sequence<kNvfp4FirstW4a4AttnInput - kNvfp4FirstSmallT>{});

} // namespace

void nvfp4_attn_input_small_t_launch(const Tensor& x, const Weight& weight, Tensor& q, Tensor& gate,
                                     Tensor& k, Tensor& v, cudaStream_t stream) {
    const std::int32_t tokens = x.ne[1];
    if (tokens < kNvfp4FirstSmallT || tokens >= kNvfp4FirstW4a4AttnInput) {
        throw std::invalid_argument(
            "nvfp4 attn_input_proj SmallT: T is at or above the W4A4 cutover");
    }
    kLaunchers[static_cast<std::size_t>(tokens - kNvfp4FirstSmallT)](x, weight, q, gate, k, v,
                                                                     stream);
}

} // namespace ninfer::ops::detail
