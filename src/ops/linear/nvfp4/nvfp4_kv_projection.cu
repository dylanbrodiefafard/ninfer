#include "ops/linear/nvfp4/nvfp4_kv_projection.h"

#include "core/device.h"
#include "ops/linear/nvfp4/nvfp4_config.h"
#include "ops/linear/nvfp4/nvfp4_gemv.cuh"
#include "ops/linear/nvfp4/nvfp4_w4a8_mma.cuh"

#include <cuda_bf16.h>

#include <algorithm>
#include <cstdint>

namespace ninfer::ops::detail {
namespace {

constexpr int kFirstRow   = 4096;
constexpr int kKvRows     = 1024;
constexpr int kOutputRows = 2 * kKvRows;

struct KvRows {
    static constexpr bool kContiguous = true;

    __device__ __forceinline__ int weight_row(int row_begin, int local_row) const {
        return kFirstRow + row_begin + local_row;
    }
};

// Every 16/32-row MMA tile and every 128-row GEMV scale tile stays within one 1024-row output.
// The original 6144-row weight geometry controls strides; only the output grid is shortened.
template <int RowOffset>
struct KvOutput {
    __nv_bfloat16* key;
    __nv_bfloat16* value;

    __device__ __forceinline__ __nv_bfloat16* address(int row, int token) const {
        const auto selected = row - RowOffset;
        return (selected < kKvRows ? key : value) + static_cast<std::int64_t>(token) * kKvRows +
               selected % kKvRows;
    }

    __device__ __forceinline__ void store(int row, int token, float number) const {
        *address(row, token) = __float2bfloat16_rn(number);
    }

    __device__ __forceinline__ void store_vector(int row, int token, uint4 numbers) const {
        store_vec(address(row, token), numbers);
    }
};

} // namespace

void launch_nvfp4_kv_projection(const Tensor& x, const Weight& weight, Tensor& key, Tensor& value,
                                cudaStream_t stream, std::int32_t sequence_width) {
    using Geometry = Nvfp4DflashQkvGeometry;
    if (sequence_width == 1) {
        using Schedule        = typename Nvfp4LinearDecodeProductionSchedule<Geometry>::Type;
        constexpr int kBlocks = kOutputRows / Schedule::kRowsPerCta;
        // Each panel keeps the original GEMV warp/lane mapping, staging and synchronization.
        // Its M128 tile starts 4096 rows into the original planes; output stores remove that
        // offset.
        for (int token = 0; token < x.ne[1]; ++token) {
            nvfp4_gemv_kernel<Geometry, Schedule, Nvfp4IdentityEpilogue, KvOutput<kFirstRow>,
                              Nvfp4PackedActivation<Geometry>, kFirstRow>
                <<<kBlocks, Schedule::kThreads, 0, stream>>>(
                    Nvfp4PackedActivation<Geometry>{static_cast<const __nv_bfloat16*>(x.data) +
                                                    static_cast<std::int64_t>(token) *
                                                        Geometry::kInputRows},
                    static_cast<const std::uint8_t*>(weight.qdata),
                    static_cast<const std::uint8_t*>(weight.scales),
                    1.0F / weight.weight_scale_divisor, Nvfp4IdentityEpilogue{},
                    KvOutput<kFirstRow>{static_cast<__nv_bfloat16*>(key.data) +
                                            static_cast<std::int64_t>(token) * kKvRows,
                                        static_cast<__nv_bfloat16*>(value.data) +
                                            static_cast<std::int64_t>(token) * kKvRows});
            CUDA_CHECK(cudaGetLastError());
        }
        return;
    }

    // CUDA grid.y admits 65535 CTAs. Beyond that envelope, split contiguous columns without
    // changing any output's complete ascending-K reduction or reusing a weight row's address.
    constexpr std::int64_t kColumnChunk = 48LL * 65535;
    for (std::int64_t first = 0; first < x.ne[1]; first += kColumnChunk) {
        const auto columns = static_cast<std::int32_t>(
            std::min(kColumnChunk, static_cast<std::int64_t>(x.ne[1]) - first));
        const Nvfp4Bf16Activation activation{static_cast<const __nv_bfloat16*>(x.data) +
                                             first * Geometry::kInputRows};
        const KvOutput<0> output{static_cast<__nv_bfloat16*>(key.data) + first * kKvRows,
                                 static_cast<__nv_bfloat16*>(value.data) + first * kKvRows};
        // The existing A16 family owns CTA/warp mapping, shared staging, cp.async waits and
        // output barriers. KvRows changes only original row addressing; OutputRows trims grid.x.
        launch_nvfp4_w4a8_mma<Geometry, Nvfp4IdentityEpilogue, KvOutput<0>, KvRows, false,
                              Nvfp4Bf16Activation, kOutputRows>(
            weight, columns, activation, Nvfp4IdentityEpilogue{}, output, stream);
    }
}

} // namespace ninfer::ops::detail
