#include "ops/launcher/qwen4_sparse_moe.h"

#include "core/device.h"
#include "ops/common/math.cuh"
#include "ops/kernel/ggml_block_linear.cuh"

#include <cuda_bf16.h>
#include <math_constants.h>

#include <cstdint>
#include <stdexcept>

namespace ninfer::ops::detail {
namespace {

constexpr int kBlock = 256;

template <QType Format>
__global__ void indexed_gate_up_swiglu_kernel(
    const __nv_bfloat16* x, const std::uint8_t* gate_bank, const std::uint8_t* up_bank,
    const std::int32_t* selected_ids, __nv_bfloat16* activated, std::int32_t rows,
    std::int32_t columns, std::uint64_t row_bytes) {
    using Decoder = GgmlDecoder<Format>;
    const std::uint64_t matrix_bytes = static_cast<std::uint64_t>(rows) * row_bytes;
    const std::int32_t rank = static_cast<std::int32_t>(blockIdx.y);
    const std::int32_t expert = selected_ids[rank];
    const std::uint64_t row_offset = static_cast<std::uint64_t>(expert) * matrix_bytes +
                                     static_cast<std::uint64_t>(blockIdx.x) * row_bytes;
    const auto* gate_row = gate_bank + row_offset;
    const auto* up_row = up_bank + row_offset;
    float gate_partial = 0.0F;
    float up_partial = 0.0F;
    for (std::int32_t column = static_cast<std::int32_t>(threadIdx.x); column < columns;
         column += static_cast<std::int32_t>(blockDim.x)) {
        const std::int32_t block_index = column / Decoder::block_values;
        const std::int32_t block_item = column % Decoder::block_values;
        const std::uint64_t block_offset =
            static_cast<std::uint64_t>(block_index) * Decoder::block_bytes;
        const float input = __bfloat162float(x[column]);
        gate_partial =
            fmaf(Decoder::value(gate_row + block_offset, block_item), input, gate_partial);
        up_partial = fmaf(Decoder::value(up_row + block_offset, block_item), input, up_partial);
    }

    constexpr unsigned full_warp = 0xffffffffU;
    for (int offset = 16; offset > 0; offset >>= 1) {
        gate_partial += __shfl_down_sync(full_warp, gate_partial, offset);
        up_partial += __shfl_down_sync(full_warp, up_partial, offset);
    }
    __shared__ float gate_warp_sums[8];
    __shared__ float up_warp_sums[8];
    const int lane = static_cast<int>(threadIdx.x) & 31;
    const int warp = static_cast<int>(threadIdx.x) >> 5;
    if (lane == 0) {
        gate_warp_sums[warp] = gate_partial;
        up_warp_sums[warp] = up_partial;
    }
    __syncthreads();
    if (warp == 0) {
        float gate_total = lane < 8 ? gate_warp_sums[lane] : 0.0F;
        float up_total = lane < 8 ? up_warp_sums[lane] : 0.0F;
        for (int offset = 16; offset > 0; offset >>= 1) {
            gate_total += __shfl_down_sync(full_warp, gate_total, offset);
            up_total += __shfl_down_sync(full_warp, up_total, offset);
        }
        if (lane == 0) {
            // The unfused public profile stores both projections to BF16 before SwiGLU.
            const __nv_bfloat16 gate_bf16 = __float2bfloat16_rn(gate_total);
            const __nv_bfloat16 up_bf16 = __float2bfloat16_rn(up_total);
            activated[static_cast<std::uint64_t>(rank) * rows + blockIdx.x] =
                __float2bfloat16_rn(
                silu(__bfloat162float(gate_bf16)) * __bfloat162float(up_bf16));
        }
    }
}

template <QType Format>
void launch_indexed_gate_up_swiglu(const Tensor& x, const Weight& gate_bank,
                                   const Weight& up_bank, const Tensor& selected_ids,
                                   Tensor& activated, cudaStream_t stream) {
    constexpr std::uint64_t block_values = GgmlDecoder<Format>::block_values;
    constexpr std::uint64_t block_bytes = GgmlDecoder<Format>::block_bytes;
    const std::uint64_t row_bytes =
        static_cast<std::uint64_t>(gate_bank.k) / block_values * block_bytes;
    const dim3 grid(static_cast<unsigned>(gate_bank.n),
                    static_cast<unsigned>(kQwen4SparseMoeTopK));
    indexed_gate_up_swiglu_kernel<Format><<<grid, kBlock, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(x.data),
        static_cast<const std::uint8_t*>(gate_bank.qdata),
        static_cast<const std::uint8_t*>(up_bank.qdata),
        static_cast<const std::int32_t*>(selected_ids.data),
        static_cast<__nv_bfloat16*>(activated.data), gate_bank.n, gate_bank.k, row_bytes);
}

template <QType Format>
__global__ void shared_gate_up_swiglu_kernel(
    const __nv_bfloat16* x, const std::uint8_t* gate, const std::uint8_t* up,
    __nv_bfloat16* activated, std::int32_t columns, std::uint64_t row_bytes) {
    using Decoder = GgmlDecoder<Format>;
    const auto* gate_row = gate + static_cast<std::uint64_t>(blockIdx.x) * row_bytes;
    const auto* up_row = up + static_cast<std::uint64_t>(blockIdx.x) * row_bytes;
    float gate_partial = 0.0F;
    float up_partial = 0.0F;
    for (std::int32_t column = static_cast<std::int32_t>(threadIdx.x); column < columns;
         column += static_cast<std::int32_t>(blockDim.x)) {
        const std::int32_t block_index = column / Decoder::block_values;
        const std::int32_t block_item = column % Decoder::block_values;
        const std::uint64_t block_offset =
            static_cast<std::uint64_t>(block_index) * Decoder::block_bytes;
        const float input = __bfloat162float(x[column]);
        gate_partial =
            fmaf(Decoder::value(gate_row + block_offset, block_item), input, gate_partial);
        up_partial = fmaf(Decoder::value(up_row + block_offset, block_item), input, up_partial);
    }

    constexpr unsigned full_warp = 0xffffffffU;
    for (int offset = 16; offset > 0; offset >>= 1) {
        gate_partial += __shfl_down_sync(full_warp, gate_partial, offset);
        up_partial += __shfl_down_sync(full_warp, up_partial, offset);
    }
    __shared__ float gate_warp_sums[8];
    __shared__ float up_warp_sums[8];
    const int lane = static_cast<int>(threadIdx.x) & 31;
    const int warp = static_cast<int>(threadIdx.x) >> 5;
    if (lane == 0) {
        gate_warp_sums[warp] = gate_partial;
        up_warp_sums[warp] = up_partial;
    }
    __syncthreads();
    if (warp == 0) {
        float gate_total = lane < 8 ? gate_warp_sums[lane] : 0.0F;
        float up_total = lane < 8 ? up_warp_sums[lane] : 0.0F;
        for (int offset = 16; offset > 0; offset >>= 1) {
            gate_total += __shfl_down_sync(full_warp, gate_total, offset);
            up_total += __shfl_down_sync(full_warp, up_total, offset);
        }
        if (lane == 0) {
            const __nv_bfloat16 gate_bf16 = __float2bfloat16_rn(gate_total);
            const __nv_bfloat16 up_bf16 = __float2bfloat16_rn(up_total);
            activated[blockIdx.x] = __float2bfloat16_rn(
                silu(__bfloat162float(gate_bf16)) * __bfloat162float(up_bf16));
        }
    }
}

template <QType Format>
void launch_shared_gate_up_swiglu(const Tensor& x, const Weight& gate, const Weight& up,
                                  Tensor& activated, cudaStream_t stream) {
    constexpr std::uint64_t block_values = GgmlDecoder<Format>::block_values;
    constexpr std::uint64_t block_bytes = GgmlDecoder<Format>::block_bytes;
    const std::uint64_t row_bytes =
        static_cast<std::uint64_t>(gate.k) / block_values * block_bytes;
    shared_gate_up_swiglu_kernel<Format><<<gate.n, kBlock, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(x.data),
        static_cast<const std::uint8_t*>(gate.qdata),
        static_cast<const std::uint8_t*>(up.qdata),
        static_cast<__nv_bfloat16*>(activated.data), gate.k, row_bytes);
}

template <QType Format>
__global__ void indexed_down_finish_kernel(
    const __nv_bfloat16* activated, const std::uint8_t* bank,
    const std::int32_t* selected_ids, const float* selected_weights,
    const __nv_bfloat16* shared, const float* shared_gate, __nv_bfloat16* destination,
    std::int32_t rows, std::int32_t columns, std::uint64_t row_bytes) {
    using Decoder = GgmlDecoder<Format>;
    const std::uint64_t matrix_bytes = static_cast<std::uint64_t>(rows) * row_bytes;
    constexpr unsigned full_warp = 0xffffffffU;
    __shared__ float warp_sums[8];
    const int lane = static_cast<int>(threadIdx.x) & 31;
    const int warp = static_cast<int>(threadIdx.x) >> 5;
    float routed = 0.0F;
    for (std::int32_t rank = 0; rank < kQwen4SparseMoeTopK; ++rank) {
        const std::int32_t expert = selected_ids[rank];
        const auto* row = bank + static_cast<std::uint64_t>(expert) * matrix_bytes +
                          static_cast<std::uint64_t>(blockIdx.x) * row_bytes;
        const auto* x = activated + static_cast<std::uint64_t>(rank) * columns;
        float partial = 0.0F;
        for (std::int32_t column = static_cast<std::int32_t>(threadIdx.x); column < columns;
             column += static_cast<std::int32_t>(blockDim.x)) {
            const std::int32_t block_index = column / Decoder::block_values;
            const std::int32_t block_item = column % Decoder::block_values;
            const float weight = Decoder::value(
                row + static_cast<std::uint64_t>(block_index) * Decoder::block_bytes,
                block_item);
            partial = fmaf(weight, __bfloat162float(x[column]), partial);
        }
        for (int offset = 16; offset > 0; offset >>= 1) {
            partial += __shfl_down_sync(full_warp, partial, offset);
        }
        if (lane == 0) { warp_sums[warp] = partial; }
        __syncthreads();
        if (warp == 0) {
            float total = lane < 8 ? warp_sums[lane] : 0.0F;
            for (int offset = 16; offset > 0; offset >>= 1) {
                total += __shfl_down_sync(full_warp, total, offset);
            }
            if (lane == 0) {
                // Preserve the projected-expert BF16 Store before rank-ordered FP32 accumulation.
                const __nv_bfloat16 expert_bf16 = __float2bfloat16_rn(total);
                routed = fmaf(selected_weights[rank], __bfloat162float(expert_bf16), routed);
            }
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        destination[blockIdx.x] = __float2bfloat16_rn(
            fmaf(shared_gate[0], __bfloat162float(shared[blockIdx.x]), routed));
    }
}

template <QType Format>
void launch_indexed_down_finish(const Tensor& activated, const Weight& expert_bank,
                                const Tensor& selected_ids, const Tensor& selected_weights,
                                const Tensor& shared, const Tensor& shared_gate,
                                Tensor& destination, cudaStream_t stream) {
    constexpr std::uint64_t block_values = GgmlDecoder<Format>::block_values;
    constexpr std::uint64_t block_bytes = GgmlDecoder<Format>::block_bytes;
    const std::uint64_t row_bytes =
        static_cast<std::uint64_t>(expert_bank.k) / block_values * block_bytes;
    indexed_down_finish_kernel<Format><<<expert_bank.n, kBlock, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(activated.data),
        static_cast<const std::uint8_t*>(expert_bank.qdata),
        static_cast<const std::int32_t*>(selected_ids.data),
        static_cast<const float*>(selected_weights.data),
        static_cast<const __nv_bfloat16*>(shared.data),
        static_cast<const float*>(shared_gate.data),
        static_cast<__nv_bfloat16*>(destination.data), expert_bank.n, expert_bank.k,
        row_bytes);
}

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

__global__ void resident_router_logits_kernel(const __nv_bfloat16* x, const float* weight,
                                              const float* shared_gate, float* logits,
                                              float* shared_gate_value) {
    const int expert = static_cast<int>(blockIdx.x);
    float sum = 0.0F;
    const auto* row = expert < kQwen4SparseMoeExperts
                          ? weight + static_cast<std::int64_t>(expert) * kQwen4SparseMoeHidden
                          : shared_gate;
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
    if (threadIdx.x == 0) {
        if (expert < kQwen4SparseMoeExperts) {
            logits[expert] = partial[0];
        } else {
            shared_gate_value[0] = sigmoid(partial[0]);
        }
    }
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

__global__ void resident_route_kernel(float* logits, std::int32_t* selected_ids,
                                      float* selected_weights) {
    static_assert(kQwen4SparseMoeExperts == 512);
    __shared__ float ranking_logits[kQwen4SparseMoeExperts];
    __shared__ float probabilities[kQwen4SparseMoeExperts];
    __shared__ std::int32_t ranking_ids[kQwen4SparseMoeExperts];
    __shared__ float maximum;
    __shared__ float denominator;

    const int expert = static_cast<int>(threadIdx.x);
    ranking_logits[expert] = logits[expert];
    ranking_ids[expert] = expert;
    __syncthreads();
    if (expert == 0) {
        float value = ranking_logits[0];
        for (int id = 1; id < kQwen4SparseMoeExperts; ++id) {
            value = fmaxf(value, ranking_logits[id]);
        }
        maximum = value;
    }
    __syncthreads();
    probabilities[expert] = expf(logits[expert] - maximum);
    __syncthreads();
    if (expert == 0) {
        // Preserve the expert-order FP32 sum used by the original implementation. Parallel work
        // is limited to independent exp/normalize values and an exact raw-logit ordering network.
        float sum = 0.0F;
        for (int id = 0; id < kQwen4SparseMoeExperts; ++id) {
            sum += probabilities[id];
        }
        denominator = sum;
    }
    __syncthreads();
    probabilities[expert] /= denominator;
    __syncthreads();

    for (int width = 2; width <= kQwen4SparseMoeExperts; width <<= 1) {
        for (int stride = width >> 1; stride > 0; stride >>= 1) {
            const int peer = expert ^ stride;
            if (peer > expert) {
                const float left_logit = ranking_logits[expert];
                const float right_logit = ranking_logits[peer];
                const std::int32_t left_id = ranking_ids[expert];
                const std::int32_t right_id = ranking_ids[peer];
                const bool left_better =
                    left_logit > right_logit ||
                    (left_logit == right_logit && left_id < right_id);
                const bool descending = (expert & width) == 0;
                if (left_better != descending) {
                    ranking_logits[expert] = right_logit;
                    ranking_logits[peer] = left_logit;
                    ranking_ids[expert] = right_id;
                    ranking_ids[peer] = left_id;
                }
            }
            __syncthreads();
        }
    }

    if (expert < kQwen4SparseMoeTopK) {
        selected_ids[expert] = ranking_ids[expert];
        selected_weights[expert] = probabilities[ranking_ids[expert]];
    }
    __syncthreads();
    if (expert == 0) {
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

void qwen4_sparse_moe_resident_route_launch(
    const Tensor& x, const Weight& router, const Tensor& shared_gate, Tensor& logits,
    Tensor& selected_ids, Tensor& selected_weights, Tensor& shared_gate_value,
    cudaStream_t stream) {
    resident_router_logits_kernel<<<kQwen4SparseMoeExperts + 1, kBlock, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(x.data), static_cast<const float*>(router.qdata),
        static_cast<const float*>(shared_gate.data), static_cast<float*>(logits.data),
        static_cast<float*>(shared_gate_value.data));
    CUDA_CHECK(cudaGetLastError());
    resident_route_kernel<<<1, kQwen4SparseMoeExperts, 0, stream>>>(
        static_cast<float*>(logits.data), static_cast<std::int32_t*>(selected_ids.data),
        static_cast<float*>(selected_weights.data));
    CUDA_CHECK(cudaGetLastError());
}

void qwen4_sparse_moe_shared_gate_up_swiglu_launch(
    const Tensor& x, const Weight& gate, const Weight& up, Tensor& activated,
    cudaStream_t stream) {
    switch (gate.qtype) {
    case QType::GGML_Q5_K:
        launch_shared_gate_up_swiglu<QType::GGML_Q5_K>(x, gate, up, activated, stream);
        break;
    case QType::GGML_Q6_K:
        launch_shared_gate_up_swiglu<QType::GGML_Q6_K>(x, gate, up, activated, stream);
        break;
    default:
        throw std::invalid_argument("Qwen4 resident shared SwiGLU received unsupported format");
    }
    CUDA_CHECK(cudaGetLastError());
}

void qwen4_sparse_moe_indexed_gate_up_swiglu_launch(
    const Tensor& x, const Weight& gate_bank, const Weight& up_bank,
    const Tensor& selected_ids, Tensor& activated, cudaStream_t stream) {
    switch (gate_bank.qtype) {
    case QType::GGML_IQ1_S:
        launch_indexed_gate_up_swiglu<QType::GGML_IQ1_S>(
            x, gate_bank, up_bank, selected_ids, activated, stream);
        break;
    case QType::GGML_IQ2_XXS:
        launch_indexed_gate_up_swiglu<QType::GGML_IQ2_XXS>(
            x, gate_bank, up_bank, selected_ids, activated, stream);
        break;
    default:
        throw std::invalid_argument(
            "Qwen4 resident gate/up SwiGLU received unsupported format");
    }
    CUDA_CHECK(cudaGetLastError());
}

void qwen4_sparse_moe_indexed_down_finish_launch(
    const Tensor& activated, const Weight& expert_bank, const Tensor& selected_ids,
    const Tensor& selected_weights, const Tensor& shared, const Tensor& shared_gate,
    Tensor& destination, cudaStream_t stream) {
    if (expert_bank.qtype != QType::GGML_IQ4_NL) {
        throw std::invalid_argument(
            "Qwen4 resident down accumulation received unsupported format");
    }
    launch_indexed_down_finish<QType::GGML_IQ4_NL>(
        activated, expert_bank, selected_ids, selected_weights, shared, shared_gate,
        destination, stream);
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
