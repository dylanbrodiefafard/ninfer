#include "ops/launcher/qwen4_sparse_moe.h"

#include "core/device.h"
#include "ops/common/math.cuh"

#include <cuda_bf16.h>
#include <math_constants.h>

#include <cstdint>

namespace ninfer::ops::detail {
namespace {

constexpr int kBlock = 256;

__global__ void router_logits_kernel(const __nv_bfloat16* x, const float* weight,
                                     float* logits) {
    const int expert = static_cast<int>(blockIdx.x);
    float sum = 0.0F;
    const auto* row = weight + static_cast<std::int64_t>(expert) * kQwen4SparseMoeHidden;
    for (int column = static_cast<int>(threadIdx.x); column < kQwen4SparseMoeHidden;
         column += blockDim.x) {
        sum = fmaf(row[column], __bfloat162float(x[column]), sum);
    }
    __shared__ float partial[kBlock];
    partial[threadIdx.x] = sum;
    __syncthreads();
    for (int stride = kBlock / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) { partial[threadIdx.x] += partial[threadIdx.x + stride]; }
        __syncthreads();
    }
    if (threadIdx.x == 0) { logits[expert] = partial[0]; }
}

__global__ void route_kernel(float* logits, std::int32_t* selected_ids,
                             float* selected_weights) {
    static_assert(kBlock * 2 == kQwen4SparseMoeExperts);
    __shared__ float ranking_logits[kQwen4SparseMoeExperts];
    __shared__ float probabilities[kQwen4SparseMoeExperts];
    __shared__ float candidate_logit[kBlock];
    __shared__ std::int32_t candidate_id[kBlock];
    __shared__ float maximum;
    __shared__ float denominator;

    const int lane = static_cast<int>(threadIdx.x);
    const int peer = lane + kBlock;
    ranking_logits[lane] = logits[lane];
    ranking_logits[peer] = logits[peer];
    __syncthreads();
    if (lane == 0) {
        float value = ranking_logits[0];
        for (int expert = 1; expert < kQwen4SparseMoeExperts; ++expert) {
            value = fmaxf(value, ranking_logits[expert]);
        }
        maximum = value;
    }
    __syncthreads();
    probabilities[lane] = expf(logits[lane] - maximum);
    probabilities[peer] = expf(logits[peer] - maximum);
    __syncthreads();
    if (lane == 0) {
        // Preserve the expert-order FP32 sum used by the original implementation. Parallel work
        // is limited to independent exp/normalize values and order-independent max/top-k choices.
        float sum = 0.0F;
        for (int expert = 0; expert < kQwen4SparseMoeExperts; ++expert) {
            sum += probabilities[expert];
        }
        denominator = sum;
    }
    __syncthreads();
    probabilities[lane] /= denominator;
    probabilities[peer] /= denominator;
    __syncthreads();

    for (int rank = 0; rank < kQwen4SparseMoeTopK; ++rank) {
        const float lane_logit = ranking_logits[lane];
        const float peer_logit = ranking_logits[peer];
        if (peer_logit > lane_logit) {
            candidate_logit[lane] = peer_logit;
            candidate_id[lane] = peer;
        } else {
            // Equal raw logits select the lower expert id, including the all-tie fixture.
            candidate_logit[lane] = lane_logit;
            candidate_id[lane] = lane;
        }
        __syncthreads();
        for (int stride = kBlock / 2; stride > 0; stride >>= 1) {
            if (lane < stride) {
                const float other_logit = candidate_logit[lane + stride];
                const std::int32_t other_id = candidate_id[lane + stride];
                if (other_logit > candidate_logit[lane] ||
                    (other_logit == candidate_logit[lane] &&
                     other_id < candidate_id[lane])) {
                    candidate_logit[lane] = other_logit;
                    candidate_id[lane] = other_id;
                }
            }
            __syncthreads();
        }
        if (lane == 0) {
            selected_ids[rank] = candidate_id[0];
            selected_weights[rank] = probabilities[candidate_id[0]];
            ranking_logits[candidate_id[0]] = -CUDART_INF_F;
        }
        __syncthreads();
    }

    if (lane == 0) {
        float selected_sum = 0.0F;
        for (int rank = 0; rank < kQwen4SparseMoeTopK; ++rank) {
            selected_sum += selected_weights[rank];
        }
        for (int rank = 0; rank < kQwen4SparseMoeTopK; ++rank) {
            selected_weights[rank] /= selected_sum;
        }
    }
}

__global__ void swiglu_kernel(const __nv_bfloat16* gate, const __nv_bfloat16* up,
                              __nv_bfloat16* activated) {
    const int index = static_cast<int>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= kQwen4SparseMoeIntermediate) { return; }
    activated[index] = __float2bfloat16_rn(
        silu(__bfloat162float(gate[index])) * __bfloat162float(up[index]));
}

__global__ void zero_routed_kernel(float* routed) {
    const int index = static_cast<int>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index < kQwen4SparseMoeHidden) { routed[index] = 0.0F; }
}

__global__ void accumulate_kernel(const __nv_bfloat16* expert, const float* selected_weights,
                                  int rank, float* routed) {
    const int index = static_cast<int>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= kQwen4SparseMoeHidden) { return; }
    routed[index] =
        fmaf(selected_weights[rank], __bfloat162float(expert[index]), routed[index]);
}

__global__ void shared_gate_kernel(const __nv_bfloat16* x, const float* weight, float* output) {
    float sum = 0.0F;
    for (int column = static_cast<int>(threadIdx.x); column < kQwen4SparseMoeHidden;
         column += blockDim.x) {
        sum = fmaf(weight[column], __bfloat162float(x[column]), sum);
    }
    __shared__ float partial[kBlock];
    partial[threadIdx.x] = sum;
    __syncthreads();
    for (int stride = kBlock / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) { partial[threadIdx.x] += partial[threadIdx.x + stride]; }
        __syncthreads();
    }
    if (threadIdx.x == 0) { output[0] = sigmoid(partial[0]); }
}

__global__ void finish_kernel(const float* routed, const __nv_bfloat16* shared,
                              const float* shared_gate, __nv_bfloat16* destination) {
    const int index = static_cast<int>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= kQwen4SparseMoeHidden) { return; }
    destination[index] = __float2bfloat16_rn(
        fmaf(shared_gate[0], __bfloat162float(shared[index]), routed[index]));
}

} // namespace

void qwen4_sparse_moe_route_launch(const Tensor& x, const Weight& router, Tensor& logits,
                                   Tensor& selected_ids, Tensor& selected_weights,
                                   cudaStream_t stream) {
    router_logits_kernel<<<kQwen4SparseMoeExperts, kBlock, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(x.data), static_cast<const float*>(router.qdata),
        static_cast<float*>(logits.data));
    CUDA_CHECK(cudaGetLastError());
    route_kernel<<<1, kBlock, 0, stream>>>(static_cast<float*>(logits.data),
                                           static_cast<std::int32_t*>(selected_ids.data),
                                           static_cast<float*>(selected_weights.data));
    CUDA_CHECK(cudaGetLastError());
}

void qwen4_sparse_moe_swiglu_launch(const Tensor& gate, const Tensor& up, Tensor& activated,
                                    cudaStream_t stream) {
    swiglu_kernel<<<(kQwen4SparseMoeIntermediate + kBlock - 1) / kBlock, kBlock, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(gate.data),
        static_cast<const __nv_bfloat16*>(up.data),
        static_cast<__nv_bfloat16*>(activated.data));
    CUDA_CHECK(cudaGetLastError());
}

void qwen4_sparse_moe_zero_routed_launch(Tensor& routed, cudaStream_t stream) {
    zero_routed_kernel<<<(kQwen4SparseMoeHidden + kBlock - 1) / kBlock, kBlock, 0, stream>>>(
        static_cast<float*>(routed.data));
    CUDA_CHECK(cudaGetLastError());
}

void qwen4_sparse_moe_accumulate_launch(const Tensor& expert, const Tensor& selected_weights,
                                        std::int32_t rank, Tensor& routed,
                                        cudaStream_t stream) {
    accumulate_kernel<<<(kQwen4SparseMoeHidden + kBlock - 1) / kBlock, kBlock, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(expert.data),
        static_cast<const float*>(selected_weights.data), rank, static_cast<float*>(routed.data));
    CUDA_CHECK(cudaGetLastError());
}

void qwen4_sparse_moe_shared_gate_launch(const Tensor& x, const Tensor& shared_gate,
                                         Tensor& gate_value, cudaStream_t stream) {
    shared_gate_kernel<<<1, kBlock, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(x.data),
        static_cast<const float*>(shared_gate.data), static_cast<float*>(gate_value.data));
    CUDA_CHECK(cudaGetLastError());
}

void qwen4_sparse_moe_finish_launch(const Tensor& routed, const Tensor& shared,
                                    const Tensor& shared_gate_value, Tensor& destination,
                                    cudaStream_t stream) {
    finish_kernel<<<(kQwen4SparseMoeHidden + kBlock - 1) / kBlock, kBlock, 0, stream>>>(
        static_cast<const float*>(routed.data), static_cast<const __nv_bfloat16*>(shared.data),
        static_cast<const float*>(shared_gate_value.data),
        static_cast<__nv_bfloat16*>(destination.data));
    CUDA_CHECK(cudaGetLastError());
}

} // namespace ninfer::ops::detail
