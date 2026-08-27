#pragma once

#include "core/tensor.h"

#include <cuda_runtime.h>

namespace ninfer::ops::detail {

void launch_nvfp4_decode(const Tensor& x, const Weight& weight, Tensor& out, cudaStream_t stream);
void launch_nvfp4_small_t(const Tensor& x, const Weight& weight, Tensor& out, cudaStream_t stream);
void launch_nvfp4_decode_splitk(const Tensor& embedding, const Tensor& hidden, const Weight& weight,
                                Tensor& out, cudaStream_t stream);
void launch_nvfp4_small_t_splitk(const Tensor& embedding, const Tensor& hidden,
                                 const Weight& weight, Tensor& out, cudaStream_t stream);
void launch_nvfp4_mtp_fc_splitk(const Tensor& embedding, const Tensor& hidden, const Weight& weight,
                                Tensor& out, cudaStream_t stream);

} // namespace ninfer::ops::detail
