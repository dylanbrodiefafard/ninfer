#pragma once

// Host-side pitched copies for adaptive draft panels. Dim-0 prefixes of {W,C} at C>1 are
// memcpy sources/destinations only — never Op consumers (LLD Capture/run).

#include "core/device.h"
#include "core/gdn_replay_records.h"
#include "core/tensor.h"

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

namespace ninfer::targets::qwen3_6 {

inline void copy_i32_panel(ninfer::Tensor dst, const ninfer::Tensor& src, cudaStream_t stream) {
    if (dst.data == src.data) { return; }
    CUDA_CHECK(cudaMemcpy2DAsync(
        dst.data, static_cast<std::size_t>(dst.nb[1]), src.data,
        static_cast<std::size_t>(src.nb[1]),
        static_cast<std::size_t>(src.ne[0]) * sizeof(std::int32_t),
        static_cast<std::size_t>(src.ne[1]), cudaMemcpyDeviceToDevice, stream));
}

// {rows, W, B}: copy min(src.ne[1], dst.ne[1]) width columns using nb[2] as the batch pitch.
// Wide→narrow (W_ceil pending → live W(k)) and narrow→wide (compact verify → frame) both work.
inline void copy_strided_width_panel(ninfer::Tensor dst, const ninfer::Tensor& src,
                                     cudaStream_t stream) {
    if (dst.data == src.data) { return; }
    const std::size_t elem = ninfer::dtype_size(src.dtype);
    const auto live_w =
        src.ne[1] < dst.ne[1] ? src.ne[1] : dst.ne[1];
    const std::size_t row  = static_cast<std::size_t>(src.ne[0]) *
                            static_cast<std::size_t>(live_w) * elem;
    CUDA_CHECK(cudaMemcpy2DAsync(dst.data, static_cast<std::size_t>(dst.nb[2]), src.data,
                                 static_cast<std::size_t>(src.nb[2]), row,
                                 static_cast<std::size_t>(src.ne[2]), cudaMemcpyDeviceToDevice,
                                 stream));
}

inline void pack_replay_record_layer(const ninfer::GdnReplayRecordLayer& dst,
                                     const ninfer::GdnReplayRecordLayer& src,
                                     cudaStream_t stream) {
    const auto pack4 = [&](const ninfer::Tensor& d, const ninfer::Tensor& s, std::size_t elem) {
        const std::size_t row = static_cast<std::size_t>(s.ne[0]) *
                                static_cast<std::size_t>(s.ne[1]) *
                                static_cast<std::size_t>(s.ne[2]) * elem;
        CUDA_CHECK(cudaMemcpy2DAsync(d.data, static_cast<std::size_t>(d.nb[3]), s.data,
                                     static_cast<std::size_t>(s.nb[3]), row,
                                     static_cast<std::size_t>(s.ne[3]), cudaMemcpyDeviceToDevice,
                                     stream));
    };
    const std::size_t conv_row =
        static_cast<std::size_t>(src.conv.ne[0]) * static_cast<std::size_t>(src.conv.ne[1]) *
        ninfer::dtype_size(ninfer::DType::BF16);
    CUDA_CHECK(cudaMemcpy2DAsync(dst.conv.data, static_cast<std::size_t>(dst.conv.nb[2]),
                                 src.conv.data, static_cast<std::size_t>(src.conv.nb[2]), conv_row,
                                 static_cast<std::size_t>(src.conv.ne[2]),
                                 cudaMemcpyDeviceToDevice, stream));
    pack4(dst.key, src.key, ninfer::dtype_size(ninfer::DType::BF16));
    pack4(dst.value, src.value, ninfer::dtype_size(ninfer::DType::BF16));
    pack4(dst.gate, src.gate, ninfer::dtype_size(ninfer::DType::FP32));
}

} // namespace ninfer::targets::qwen3_6
