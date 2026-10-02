#pragma once

#include "core/tensor.h"
#include "ninfer/ops/linear.h"

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

namespace ninfer::ops::detail {

[[nodiscard]] std::size_t nvfp4_linear_workspace_capacity_bytes(std::int32_t output_rows,
                                                                std::int32_t input_rows,
                                                                LinearPolicy policy,
                                                                std::int32_t min_tokens,
                                                                std::int32_t max_tokens);

// DFlash projections whose A16 route is the tensor-core kernel: every output keeps one ascending
// K order at any T, so packed requests of width >=2 share one weight pass.
bool is_nvfp4_dflash_mma_aggregate_problem(std::int32_t output_rows, std::int32_t input_rows,
                                           LinearPolicy policy);

// DFlash drafter gate-up/down whose width-W request panel already takes W4A4: activation
// quantization and each output's reduction are column-local, so packed requests share one pass.
bool is_nvfp4_dflash_w4a4_aggregate_problem(std::int32_t output_rows, std::int32_t input_rows,
                                            LinearPolicy policy, std::int32_t sequence_width);

// DFlash conv projection (A16 SmallT), aggregated only for W=5 packed requests.
bool is_nvfp4_dflash_conv_w5_aggregate_problem(std::int32_t output_rows, std::int32_t input_rows,
                                               LinearPolicy policy);

void nvfp4_dispatch(const Tensor& x, const Weight& weight, Tensor& out, LinearPolicy policy,
                    WorkspaceArena* workspace, cudaStream_t stream);

} // namespace ninfer::ops::detail
