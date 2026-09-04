#pragma once

#include "core/arena.h"
#include "core/tensor.h"

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

namespace ninfer::ops {

/** Persistent and learned views for the fixed Qwen4-preview Gated DeltaNet layer. */
struct GatedDeltaNetLayerWeights {
    Weight qkv;       // GGML Q5_K or Q6_K [10240,2560], q[2048] | k[2048] | v[6144]
    Weight z;         // same GGML input format as qkv, [6144,2560]
    Tensor a;         // FP32 physical [2560,48], mathematical [48,2560]
    Tensor b;         // FP32 physical [2560,48], mathematical [48,2560]
    Tensor conv;      // FP32 physical [4,10240], mathematical [10240,4], oldest..current taps
    Tensor ssm_a;     // FP32 [48], converted -exp(A_log) decay multiplier
    Tensor dt_bias;   // FP32 [48]
    Tensor norm;      // FP32 [128], ordinary learned scale (no unit offset)
    Weight output;    // GGML Q6_K [2560,6144]
};

/**
 * Returns caller-owned transient capacity for the registered H=2560, Hq=16, Hv=48, Dh=128,
 * C=1/T=1 verifier profile.
 */
[[nodiscard]] std::size_t gated_delta_net_layer_workspace_capacity_bytes();

/**
 * Op: gated_delta_net_layer
 *
 * Applies the complete Qwen4-preview GDN sublayer from represented BF16 input and converted
 * UD-IQ1_S artifact weights. It
 * projects independent qkv/z/a/b branches; applies the width-four causal depthwise convolution
 * and SiLU only to qkv; L2-normalizes Q/K with epsilon 1e-6 and repeats each of 16 Q/K heads over
 * three tiled value-head groups; forms beta=sigmoid(b) and
 * g=ssm_a*softplus(a+dt_bias), where represented ssm_a is the converted -exp(A_log), and evaluates
 * the FP32 48-head DeltaNet recurrence with query scale 1/sqrt(128). Each 128-wide recurrent output
 * is plain-RMS-normalized with epsilon 1e-6,
 * multiplied by sigmoid(z), represented as BF16, and projected to BF16 width 2560.
 *
 * x/out are contiguous BF16 [2560]. conv_state_in/out are BF16 [10240,3], storing the last three
 * raw projected qkv columns in oldest-to-newest order. ssm_state_in/out are FP32 [128,128,48].
 * State outputs may exactly alias their corresponding inputs or be disjoint; partial overlap is
 * forbidden. Distinct inputs are read-only, which is the rollback boundary used by a caller. The
 * entry admits only C=1/T=1; repeated calls implement continuation without owning scheduling.
 *
 * The mathematical oracle exact-decodes represented inputs/weights and evaluates each closed
 * formula naively in FP64. It applies the declared consumer representations between formulas:
 * decoded qkv and z projection outputs are BF16, controls and each recurrent transition are FP32,
 * recurrent output is BF16 before gated norm, and gated-norm output is BF16 before the Q6_K output
 * projection. The GGUF converter stores V-side qkv/z/a/b/ssm_a/dt/conv/output in tiled order, so
 * value head h consumes Q/K head h%16. BF16 convolution state,
 * FP32 recurrent state, and BF16 out are persistent/final observable boundaries. All non-state
 * operands, output, and live workspace are pairwise non-overlapping. Execution is asynchronous on
 * stream.
 */
void gated_delta_net_layer(const Tensor& x, const GatedDeltaNetLayerWeights& weights,
                           const Tensor& conv_state_in, Tensor& conv_state_out,
                           const Tensor& ssm_state_in, Tensor& ssm_state_out, Tensor& out,
                           WorkspaceArena& workspace, cudaStream_t stream);

} // namespace ninfer::ops
