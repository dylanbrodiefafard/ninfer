#include "ninfer/ops/gated_delta_net.h"

#include "ninfer/ops/l2norm.h"

#include "core/device.h"
#include "core/layout.h"
#include "ops/common/math.h"
#include "ops/linear_attention/gated_delta_net/common.h"
#include "ops/linear_attention/gated_delta_net/launch.h"

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>

namespace ninfer::ops {
namespace {

struct Geometry {
    std::int32_t head_dim;
    std::int32_t qk_heads;
    std::int32_t value_heads;
    std::int32_t tokens;
};

void require_dtype(const Tensor& t, DType dtype, const char* name) {
    if (t.dtype != dtype) { throw std::invalid_argument(std::string("gated_delta_net: ") + name); }
}

void require_shape(const Tensor& t, std::int32_t n0, std::int32_t n1, std::int32_t n2,
                   std::int32_t n3, const char* name) {
    if (t.ne[0] != n0 || t.ne[1] != n1 || t.ne[2] != n2 || t.ne[3] != n3) {
        throw std::invalid_argument(std::string("gated_delta_net: invalid shape for ") + name);
    }
}

void require_contiguous_nonnull(const Tensor& t, const char* name) {
    if (!t.is_contiguous()) {
        throw std::invalid_argument(std::string("gated_delta_net: ") + name +
                                    " must be contiguous");
    }
    if (t.data == nullptr) {
        throw std::invalid_argument(std::string("gated_delta_net: ") + name +
                                    " data must be non-null");
    }
}

Geometry require_geometry(const Tensor& q, const Tensor& v) {
    const Geometry geometry{q.ne[0], q.ne[1], v.ne[1], q.ne[2]};
    if (!detail::gated_delta_net::is_supported_head_dim(geometry.head_dim)) {
        throw std::invalid_argument("gated_delta_net: head dimension must be 16, 32, 64, or 128");
    }
    if (!detail::gated_delta_net::are_head_counts_valid(geometry.qk_heads, geometry.value_heads)) {
        throw std::invalid_argument(
            "gated_delta_net: value heads must be at least q/k heads and divisible by them");
    }
    if (geometry.tokens <= 0) {
        throw std::invalid_argument("gated_delta_net: T must be positive");
    }
    return geometry;
}

void require_scale(float scale, std::int32_t head_dim) {
    const float expected_scale = 1.0f / std::sqrt(static_cast<float>(head_dim));
    if (!std::isfinite(scale) || scale <= 0.0f || std::abs(scale - expected_scale) > 1.0e-6f) {
        throw std::invalid_argument("gated_delta_net: scale must be 1/sqrt(head_dim)");
    }
}

Geometry validate_recurrent(const Tensor& q, const Tensor& k, const Tensor& v, const Tensor& g,
                            const Tensor& beta, float scale, const Tensor& ssm_state,
                            const Tensor& out) {
    require_dtype(q, DType::BF16, "q must be BF16");
    require_dtype(k, DType::BF16, "k must be BF16");
    require_dtype(v, DType::BF16, "v must be BF16");
    require_dtype(out, DType::BF16, "out must be BF16");
    require_dtype(g, DType::FP32, "g must be FP32");
    require_dtype(beta, DType::FP32, "beta must be FP32");
    require_dtype(ssm_state, DType::FP32, "ssm_state must be FP32");

    const Geometry geometry = require_geometry(q, v);
    require_shape(q, geometry.head_dim, geometry.qk_heads, geometry.tokens, 1, "q");
    require_shape(k, geometry.head_dim, geometry.qk_heads, geometry.tokens, 1, "k");
    require_shape(v, geometry.head_dim, geometry.value_heads, geometry.tokens, 1, "v");
    require_shape(out, geometry.head_dim, geometry.value_heads, geometry.tokens, 1, "out");
    require_shape(g, geometry.value_heads, geometry.tokens, 1, 1, "g");
    require_shape(beta, geometry.value_heads, geometry.tokens, 1, 1, "beta");
    require_shape(ssm_state, geometry.head_dim, geometry.head_dim, geometry.value_heads, 1,
                  "ssm_state");

    require_contiguous_nonnull(q, "q");
    require_contiguous_nonnull(k, "k");
    require_contiguous_nonnull(v, "v");
    require_contiguous_nonnull(g, "g");
    require_contiguous_nonnull(beta, "beta");
    require_contiguous_nonnull(ssm_state, "ssm_state");
    require_contiguous_nonnull(out, "out");

    require_scale(scale, geometry.head_dim);
    return geometry;
}

Geometry validate_recurrent_snapshot(const Tensor& q, const Tensor& k, const Tensor& v,
                                     const Tensor& g, const Tensor& beta, float scale,
                                     const Tensor& ssm_states, const Tensor& initial_slot,
                                     const Tensor& out) {
    require_dtype(q, DType::BF16, "q must be BF16");
    require_dtype(k, DType::BF16, "k must be BF16");
    require_dtype(v, DType::BF16, "v must be BF16");
    require_dtype(out, DType::BF16, "out must be BF16");
    require_dtype(g, DType::FP32, "g must be FP32");
    require_dtype(beta, DType::FP32, "beta must be FP32");
    require_dtype(ssm_states, DType::FP32, "ssm_states must be FP32");
    require_dtype(initial_slot, DType::I32, "initial_slot must be I32");

    const Geometry geometry = require_geometry(q, v);
    require_shape(q, geometry.head_dim, geometry.qk_heads, geometry.tokens, 1, "q");
    require_shape(k, geometry.head_dim, geometry.qk_heads, geometry.tokens, 1, "k");
    require_shape(v, geometry.head_dim, geometry.value_heads, geometry.tokens, 1, "v");
    require_shape(out, geometry.head_dim, geometry.value_heads, geometry.tokens, 1, "out");
    require_shape(g, geometry.value_heads, geometry.tokens, 1, 1, "g");
    require_shape(beta, geometry.value_heads, geometry.tokens, 1, 1, "beta");
    if (ssm_states.ne[0] != geometry.head_dim || ssm_states.ne[1] != geometry.head_dim ||
        ssm_states.ne[2] != geometry.value_heads || ssm_states.ne[3] < geometry.tokens) {
        throw std::invalid_argument("gated_delta_net: invalid shape for ssm_states snapshot");
    }
    require_shape(initial_slot, 1, 1, 1, 1, "initial_slot");

    require_contiguous_nonnull(q, "q");
    require_contiguous_nonnull(k, "k");
    require_contiguous_nonnull(v, "v");
    require_contiguous_nonnull(g, "g");
    require_contiguous_nonnull(beta, "beta");
    require_contiguous_nonnull(ssm_states, "ssm_states");
    require_contiguous_nonnull(initial_slot, "initial_slot");
    require_contiguous_nonnull(out, "out");

    require_scale(scale, geometry.head_dim);
    return geometry;
}

void validate_chunked(const Tensor& q, const Tensor& k, const Tensor& v, const Tensor& g,
                      const Tensor& beta, float scale, const Tensor& ssm_state_in,
                      const Tensor& ssm_state_out, const Tensor& out) {
    // ssm_state_out carries the running-state contract validated by validate_recurrent;
    // ssm_state_in is an equally-shaped read view (may alias ssm_state_out for in-place).
    const Geometry geometry = validate_recurrent(q, k, v, g, beta, scale, ssm_state_out, out);
    require_dtype(ssm_state_in, DType::FP32, "ssm_state_in must be FP32");
    require_shape(ssm_state_in, geometry.head_dim, geometry.head_dim, geometry.value_heads, 1,
                  "ssm_state_in");
    require_contiguous_nonnull(ssm_state_in, "ssm_state_in");
}

struct ChunkedWorkspace {
    Tensor normalized_q;
    Tensor normalized_k;
    DeviceSpan stage;
};

template <class Allocator>
ChunkedWorkspace allocate_chunked_workspace(Allocator& allocator, std::int32_t head_dim,
                                            std::int32_t qk_heads, std::int32_t value_heads,
                                            std::int32_t tokens, bool normalize_qk) {
    ChunkedWorkspace out;
    const std::int32_t full =
        (tokens / detail::gated_delta_net::kChunkTokens) * detail::gated_delta_net::kChunkTokens;
    if (full == 0) { return out; }
    if (normalize_qk) {
        out.normalized_q = allocator.alloc(DType::BF16, {head_dim, qk_heads, tokens});
        out.normalized_k = allocator.alloc(DType::BF16, {head_dim, qk_heads, tokens});
    }
    out.stage = allocator.alloc_bytes(
        detail::gated_delta_net::chunked_workspace_bytes(head_dim, qk_heads, value_heads, full));
    return out;
}

} // namespace

std::size_t gated_delta_net_workspace_capacity_bytes(std::int32_t head_dim, std::int32_t qk_heads,
                                                     std::int32_t value_heads, bool normalize_qk,
                                                     std::int32_t min_tokens,
                                                     std::int32_t max_tokens) {
    if (!detail::gated_delta_net::is_supported_head_dim(head_dim) ||
        !detail::gated_delta_net::are_head_counts_valid(qk_heads, value_heads) || min_tokens <= 0 ||
        max_tokens < min_tokens) {
        throw std::invalid_argument("gated_delta_net workspace: invalid profile or interval");
    }
    WorkspaceLayoutBuilder layout;
    (void)allocate_chunked_workspace(layout, head_dim, qk_heads, value_heads, max_tokens,
                                     normalize_qk);
    return layout.peak_bytes(1);
}

void gated_delta_net(const Tensor& q, const Tensor& k, const Tensor& v, const Tensor& g,
                     const Tensor& beta, float scale, bool normalize_qk, WorkspaceArena& ws,
                     Tensor& ssm_state, Tensor& out, cudaStream_t stream) {
    if (q.ne[2] != 1) {
        gated_delta_net(q, k, v, g, beta, scale, normalize_qk, ws, ssm_state, ssm_state, out,
                        stream);
        return;
    }
    validate_recurrent(q, k, v, g, beta, scale, ssm_state, out);

    (void)ws;
    detail::gated_delta_net::launch_recurrent(q, k, v, g, beta, scale, normalize_qk, ssm_state, out,
                                              stream);
}

void gated_delta_net_snapshot(const Tensor& q, const Tensor& k, const Tensor& v, const Tensor& g,
                              const Tensor& beta, float scale, bool normalize_qk,
                              Tensor& ssm_states, const Tensor& initial_slot, Tensor& out,
                              cudaStream_t stream) {
    validate_recurrent_snapshot(q, k, v, g, beta, scale, ssm_states, initial_slot, out);

    detail::gated_delta_net::launch_recurrent_snapshot(q, k, v, g, beta, scale, normalize_qk,
                                                       ssm_states, initial_slot, out, stream);
}

void gated_delta_net(const Tensor& q, const Tensor& k, const Tensor& v, const Tensor& g,
                     const Tensor& beta, float scale, bool normalize_qk, WorkspaceArena& ws,
                     const Tensor& ssm_state_in, Tensor& ssm_state_out, Tensor& out,
                     cudaStream_t stream) {
    validate_chunked(q, k, v, g, beta, scale, ssm_state_in, ssm_state_out, out);

    auto scratch_scope   = ws.scope();
    const std::int32_t T = q.ne[2];
    const std::int32_t T_full =
        (T / detail::gated_delta_net::kChunkTokens) * detail::gated_delta_net::kChunkTokens;
    ChunkedWorkspace scratch =
        allocate_chunked_workspace(ws, q.ne[0], q.ne[1], v.ne[1], T, normalize_qk);
    Tensor q_compute         = q;
    Tensor k_compute         = k;
    bool recurrent_normalize = normalize_qk;
    if (normalize_qk && T_full > 0) {
        q_compute = scratch.normalized_q;
        k_compute = scratch.normalized_k;
        l2norm(q, 1.0e-6f, q_compute, stream);
        l2norm(k, 1.0e-6f, k_compute, stream);
        recurrent_normalize = false;
    }
    if (T_full > 0) {
        Tensor q_full    = q_compute.slice(2, 0, T_full);
        Tensor k_full    = k_compute.slice(2, 0, T_full);
        Tensor v_full    = v.slice(2, 0, T_full);
        Tensor g_full    = g.slice(1, 0, T_full);
        Tensor beta_full = beta.slice(1, 0, T_full);
        Tensor out_full  = out.slice(2, 0, T_full);
        detail::gated_delta_net::launch_chunked(q_full, k_full, v_full, g_full, beta_full, scale,
                                                ssm_state_in, ssm_state_out, out_full,
                                                scratch.stage.data, scratch.stage.bytes, stream);
    }

    const std::int32_t tail = T - T_full;
    if (tail > 0) {
        Tensor q_tail    = q_compute.slice(2, T_full, tail);
        Tensor k_tail    = k_compute.slice(2, T_full, tail);
        Tensor v_tail    = v.slice(2, T_full, tail);
        Tensor g_tail    = g.slice(1, T_full, tail);
        Tensor beta_tail = beta.slice(1, T_full, tail);
        Tensor out_tail  = out.slice(2, T_full, tail);
        // After full chunks the running state lives in ssm_state_out; a tail-only run (no full
        // chunks) reads the caller-provided ssm_state_in. Either way the tail publishes to
        // ssm_state_out.
        const Tensor& tail_in = (T_full > 0) ? ssm_state_out : ssm_state_in;
        detail::gated_delta_net::launch_recurrent_inout(q_tail, k_tail, v_tail, g_tail, beta_tail,
                                                        scale, recurrent_normalize, tail_in,
                                                        ssm_state_out, out_tail, stream);
    }
}

} // namespace ninfer::ops
