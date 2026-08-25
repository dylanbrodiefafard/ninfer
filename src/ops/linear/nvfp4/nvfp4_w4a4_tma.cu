#include "ops/linear/nvfp4/nvfp4_w4a4_tma_launch.h"

#include "core/device.h"
#include "ops/gdn_input_proj/nvfp4/nvfp4_gdn_input_output.cuh"
#include "ops/linear/nvfp4/nvfp4_config.h"
#include "ops/linear/nvfp4/nvfp4_w4a4_mma.cuh"
#include "ops/linear/nvfp4/nvfp4_w4a4_tma.cuh"
#include "ops/linear_add/nvfp4/nvfp4_linear_add_epilogue.cuh"

#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>

namespace ninfer::ops::detail {
namespace {

using TmaM256N128   = Nvfp4W4a4TmaSchedule<256, 3, 1>;
using TmaM256N128S2 = Nvfp4W4a4TmaSchedule<256, 2, 1>;
// A/B candidates selected via NINFER_NVFP4_TMA_SCHEDULE (prod default = TmaM256N128):
using TmaM128N128S2O1 = Nvfp4W4a4TmaSchedule<128, 2, 1>;
using TmaM128N128S3O1 = Nvfp4W4a4TmaSchedule<128, 3, 1>;
using TmaM128N128S4O1 = Nvfp4W4a4TmaSchedule<128, 4, 1>;
using TmaM128N128S2O2 = Nvfp4W4a4TmaSchedule<128, 2, 2>;

// Cached A/B selection of the TMA W4A4 schedule. 0 = production (TmaM256N128). Non-default
// values are for the isolated A/B bench only; production always lands on 0.
inline int nvfp4_tma_schedule_sel() {
    static const int sel = [] {
        const char* e = std::getenv("NINFER_NVFP4_TMA_SCHEDULE");
        if (e == nullptr) { return 0; }
        if (std::strcmp(e, "m256s3") == 0) { return 0; }
        if (std::strcmp(e, "m256s2") == 0) { return 1; }
        if (std::strcmp(e, "m128s2o1") == 0) { return 2; }
        if (std::strcmp(e, "m128s3o1") == 0) { return 3; }
        if (std::strcmp(e, "m128s4o1") == 0) { return 4; }
        if (std::strcmp(e, "m128s2o2") == 0) { return 5; }
        return 0;
    }();
    return sel;
}

constexpr std::int32_t kQueryRows  = 6144;
constexpr std::int32_t kKeyRows    = 1024;
constexpr std::int32_t kGateRows   = 6144;
constexpr std::int32_t kKeyBegin   = kQueryRows;
constexpr std::int32_t kGateBegin  = kKeyBegin + kKeyRows;
constexpr std::int32_t kValueBegin = kGateBegin + kGateRows;

struct AttentionOutput {
    __nv_bfloat16* query;
    __nv_bfloat16* key;
    __nv_bfloat16* gate;
    __nv_bfloat16* value;

    __device__ __forceinline__ __nv_bfloat16* destination(std::int32_t parent_row,
                                                          std::int32_t token) const {
        if (parent_row < kKeyBegin) {
            return query + static_cast<std::int64_t>(token) * kQueryRows + parent_row;
        }
        if (parent_row < kGateBegin) {
            return key + static_cast<std::int64_t>(token) * kKeyRows + parent_row - kKeyBegin;
        }
        if (parent_row < kValueBegin) {
            return gate + static_cast<std::int64_t>(token) * kGateRows + parent_row - kGateBegin;
        }
        return value + static_cast<std::int64_t>(token) * kKeyRows + parent_row - kValueBegin;
    }

    __device__ __forceinline__ void store_vector(std::int32_t parent_row, std::int32_t token,
                                                 uint4 values) const {
        store_vec(destination(parent_row, token), values);
    }
};

static_assert((kQueryRows % TmaM256N128::kBlockN) == 0);
static_assert((kKeyRows % TmaM256N128::kBlockN) == 0);
static_assert((kGateRows % TmaM256N128::kBlockN) == 0);

template <class Geometry, class Schedule, class Epilogue, class Output>
void launch_tma(const std::uint8_t* activation_codes, const std::uint8_t* activation_scales,
                const std::uint8_t* weight_codes, const std::uint8_t* weight_scales,
                std::int32_t tokens, float alpha, Epilogue epilogue, Output output,
                cudaStream_t stream) {
    const Nvfp4W4a4TmaDescriptors descriptors =
        make_nvfp4_w4a4_tma_descriptors<Geometry, Schedule::kBlockM>(
            activation_codes, activation_scales, weight_codes, weight_scales, tokens);
    constexpr std::size_t kSharedBytes = sizeof(Nvfp4W4a4TmaSharedStorage<Schedule>);
    static const bool kConfigured      = [] {
        CUDA_CHECK(cudaFuncSetAttribute(nvfp4_w4a4_tma_kernel<Geometry, Schedule, Epilogue, Output>,
                                             cudaFuncAttributeMaxDynamicSharedMemorySize,
                                             static_cast<int>(kSharedBytes)));
        return true;
    }();
    (void)kConfigured;

    const dim3 grid(Geometry::kOutputRows / Schedule::kBlockN, tokens / Schedule::kBlockM);
    nvfp4_w4a4_tma_kernel<Geometry, Schedule>
        <<<grid, Schedule::kThreads, kSharedBytes, stream>>>(descriptors, alpha, epilogue, output);
    CUDA_CHECK(cudaGetLastError());
}

template <class Geometry>
void launch_linear(const std::uint8_t* activation_codes, const std::uint8_t* activation_scales,
                   const std::uint8_t* weight_codes, const std::uint8_t* weight_scales,
                   __nv_bfloat16* output, std::int32_t tokens, float alpha, cudaStream_t stream) {
    switch (nvfp4_tma_schedule_sel()) {
    case 1:
        launch_tma<Geometry, TmaM256N128S2>(activation_codes, activation_scales, weight_codes,
                                            weight_scales, tokens, alpha, Nvfp4IdentityEpilogue{},
                                            Nvfp4ContiguousOutput{output, Geometry::kOutputRows},
                                            stream);
        return;
    case 2:
        launch_tma<Geometry, TmaM128N128S2O1>(activation_codes, activation_scales, weight_codes,
                                               weight_scales, tokens, alpha, Nvfp4IdentityEpilogue{},
                                               Nvfp4ContiguousOutput{output, Geometry::kOutputRows},
                                               stream);
        return;
    case 3:
        launch_tma<Geometry, TmaM128N128S3O1>(activation_codes, activation_scales, weight_codes,
                                               weight_scales, tokens, alpha, Nvfp4IdentityEpilogue{},
                                               Nvfp4ContiguousOutput{output, Geometry::kOutputRows},
                                               stream);
        return;
    case 4:
        launch_tma<Geometry, TmaM128N128S4O1>(activation_codes, activation_scales, weight_codes,
                                               weight_scales, tokens, alpha, Nvfp4IdentityEpilogue{},
                                               Nvfp4ContiguousOutput{output, Geometry::kOutputRows},
                                               stream);
        return;
    case 5:
        launch_tma<Geometry, TmaM128N128S2O2>(activation_codes, activation_scales, weight_codes,
                                               weight_scales, tokens, alpha, Nvfp4IdentityEpilogue{},
                                               Nvfp4ContiguousOutput{output, Geometry::kOutputRows},
                                               stream);
        return;
    default:
        launch_tma<Geometry, TmaM256N128>(activation_codes, activation_scales, weight_codes,
                                          weight_scales, tokens, alpha, Nvfp4IdentityEpilogue{},
                                          Nvfp4ContiguousOutput{output, Geometry::kOutputRows}, stream);
        return;
    }
}

} // namespace

void launch_nvfp4_w4a4_tma_linear(Nvfp4Problem problem, const std::uint8_t* activation_codes,
                                  const std::uint8_t* activation_scales,
                                  const std::uint8_t* weight_codes,
                                  const std::uint8_t* weight_scales, __nv_bfloat16* output,
                                  std::int32_t tokens, float alpha, cudaStream_t stream) {
    switch (problem) {
    case Nvfp4Problem::AttnInput:
        launch_linear<Nvfp4AttnInputGeometry>(activation_codes, activation_scales, weight_codes,
                                              weight_scales, output, tokens, alpha, stream);
        return;
    case Nvfp4Problem::GdnInput:
        launch_linear<Nvfp4GdnInputGeometry>(activation_codes, activation_scales, weight_codes,
                                             weight_scales, output, tokens, alpha, stream);
        return;
    case Nvfp4Problem::MlpGateUp:
        launch_tma<Nvfp4MlpGateUpGeometry, TmaM256N128S2>(
            activation_codes, activation_scales, weight_codes, weight_scales, tokens, alpha,
            Nvfp4IdentityEpilogue{},
            Nvfp4ContiguousOutput{output, Nvfp4MlpGateUpGeometry::kOutputRows}, stream);
        return;
    case Nvfp4Problem::Residual6144:
        launch_linear<Nvfp4Residual6144Geometry>(activation_codes, activation_scales, weight_codes,
                                                 weight_scales, output, tokens, alpha, stream);
        return;
    case Nvfp4Problem::Residual17408:
        launch_linear<Nvfp4Residual17408Geometry>(activation_codes, activation_scales, weight_codes,
                                                  weight_scales, output, tokens, alpha, stream);
        return;
    case Nvfp4Problem::DflashFeature:
    case Nvfp4Problem::DflashQkv:
    case Nvfp4Problem::DflashAttnOut:
    case Nvfp4Problem::DflashConvProj:
    case Nvfp4Problem::DflashSelector:
        return;
    }
}

void launch_nvfp4_w4a4_tma_attention(const std::uint8_t* activation_codes,
                                     const std::uint8_t* activation_scales,
                                     const std::uint8_t* weight_codes,
                                     const std::uint8_t* weight_scales, __nv_bfloat16* query,
                                     __nv_bfloat16* gate, __nv_bfloat16* key, __nv_bfloat16* value,
                                     std::int32_t tokens, float alpha, cudaStream_t stream) {
    launch_tma<Nvfp4AttnInputGeometry, TmaM256N128>(
        activation_codes, activation_scales, weight_codes, weight_scales, tokens, alpha,
        Nvfp4IdentityEpilogue{}, AttentionOutput{query, key, gate, value}, stream);
}

void launch_nvfp4_w4a4_tma_gdn(const std::uint8_t* activation_codes,
                               const std::uint8_t* activation_scales,
                               const std::uint8_t* weight_codes, const std::uint8_t* weight_scales,
                               __nv_bfloat16* qkv, __nv_bfloat16* z, std::int32_t tokens,
                               float alpha, cudaStream_t stream) {
    launch_tma<Nvfp4GdnInputGeometry, TmaM256N128>(
        activation_codes, activation_scales, weight_codes, weight_scales, tokens, alpha,
        Nvfp4IdentityEpilogue{}, Nvfp4GdnInputOutput{qkv, z}, stream);
}

template <class Geometry>
void launch_linear_add(const std::uint8_t* activation_codes, const std::uint8_t* activation_scales,
                       const std::uint8_t* weight_codes, const std::uint8_t* weight_scales,
                       __nv_bfloat16* residual, std::int32_t tokens, float alpha,
                       cudaStream_t stream) {
    launch_tma<Geometry, TmaM256N128>(
        activation_codes, activation_scales, weight_codes, weight_scales, tokens, alpha,
        Nvfp4AddResidualEpilogue{residual, Geometry::kOutputRows},
        Nvfp4ContiguousOutput{residual, Geometry::kOutputRows}, stream);
}

void launch_nvfp4_w4a4_tma_linear_add(Nvfp4Problem problem, const std::uint8_t* activation_codes,
                                      const std::uint8_t* activation_scales,
                                      const std::uint8_t* weight_codes,
                                      const std::uint8_t* weight_scales, __nv_bfloat16* residual,
                                      std::int32_t tokens, float alpha, cudaStream_t stream) {
    switch (problem) {
    case Nvfp4Problem::Residual6144:
        launch_linear_add<Nvfp4Residual6144Geometry>(activation_codes, activation_scales,
                                                     weight_codes, weight_scales, residual, tokens,
                                                     alpha, stream);
        return;
    case Nvfp4Problem::Residual17408:
        launch_linear_add<Nvfp4Residual17408Geometry>(activation_codes, activation_scales,
                                                      weight_codes, weight_scales, residual, tokens,
                                                      alpha, stream);
        return;
    case Nvfp4Problem::AttnInput:
    case Nvfp4Problem::GdnInput:
    case Nvfp4Problem::MlpGateUp:
    case Nvfp4Problem::DflashFeature:
    case Nvfp4Problem::DflashQkv:
    case Nvfp4Problem::DflashAttnOut:
    case Nvfp4Problem::DflashConvProj:
    case Nvfp4Problem::DflashSelector:
        return;
    }
}

} // namespace ninfer::ops::detail
