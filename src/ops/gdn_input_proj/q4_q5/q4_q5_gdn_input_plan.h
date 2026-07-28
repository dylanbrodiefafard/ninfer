#pragma once

#include "core/tensor.h"

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

namespace ninfer::ops::detail {

enum class Q4Q5GdnInputScheduleId {
    IndependentDirectFixed,
    GroupedMixedMmaR64C128,
};

struct Q4Q5GdnInputProblem {
    std::int32_t input_rows;
    std::int32_t qk_rows;
    std::int32_t value_rows;
    std::int32_t output_rows;
    std::int32_t padded_k;
    std::int32_t cols;
};

struct Q4Q5GdnInputPlan {
    Q4Q5GdnInputScheduleId schedule;
};

const char* q4_q5_gdn_input_schedule_name(Q4Q5GdnInputScheduleId schedule) noexcept;

bool q4_q5_gdn_input_admits(const Q4Q5GdnInputProblem& problem) noexcept;
Q4Q5GdnInputPlan q4_q5_gdn_input_resolve_plan(const Q4Q5GdnInputProblem& problem);

void q4_q5_gdn_input_execute_plan(const Q4Q5GdnInputPlan& plan, const Tensor& x,
                                  const Weight& qk_weight, const Weight& v_weight, Tensor& qkv,
                                  cudaStream_t stream);
void q4_q5_gdn_input_dispatch(const Tensor& x, const Weight& qk_weight, const Weight& v_weight,
                              Tensor& qkv, cudaStream_t stream);

} // namespace ninfer::ops::detail
