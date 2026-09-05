#include "ninfer/ops/gated_delta_net.h"

#include "core/device.h"
#include "core/layout.h"
#include "ops/common/math.h"
#include "ops/linear_attention/gated_delta_net/common.h"
#include "ops/linear_attention/gated_delta_net/launch.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <span>
#include <stdexcept>
#include <string>

namespace ninfer::ops {
namespace {

struct Geometry {
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

void require_alignment(const Tensor& tensor, std::uintptr_t alignment, const char* name) {
    if ((reinterpret_cast<std::uintptr_t>(tensor.data) & (alignment - 1)) != 0) {
        throw std::invalid_argument(std::string("gated_delta_net: ") + name +
                                    " has insufficient alignment");
    }
}

struct AddressRange {
    std::uintptr_t begin;
    std::uintptr_t end;
    const char* name;
};

AddressRange address_range(const void* pointer, std::size_t bytes, const char* name) {
    const std::uintptr_t begin = reinterpret_cast<std::uintptr_t>(pointer);
    if (pointer == nullptr || bytes == 0 ||
        bytes > std::numeric_limits<std::uintptr_t>::max() - begin) {
        throw std::invalid_argument(std::string("gated_delta_net: invalid address range for ") +
                                    name);
    }
    return {begin, begin + bytes, name};
}

AddressRange address_range(const Tensor& tensor, const char* name) {
    return address_range(tensor.data, tensor.bytes(), name);
}

bool overlaps(const AddressRange& left, const AddressRange& right) {
    return left.begin < right.end && right.begin < left.end;
}

void require_disjoint(std::span<const AddressRange> ranges) {
    for (std::size_t i = 0; i < ranges.size(); ++i) {
        for (std::size_t j = i + 1; j < ranges.size(); ++j) {
            if (overlaps(ranges[i], ranges[j])) {
                throw std::invalid_argument(std::string("gated_delta_net: ") + ranges[i].name +
                                            " overlaps " + ranges[j].name);
            }
        }
    }
}

bool exact_alias(const Tensor& left, const Tensor& right) {
    return left.data == right.data && left.bytes() == right.bytes();
}

void validate_recurrent_storage(const Tensor& q, const Tensor& k, const Tensor& v,
                                const Tensor& g, const Tensor& beta,
                                const Tensor& ssm_state_in, const Tensor& ssm_state_out,
                                const Tensor& out,
                                std::span<const AddressRange> scratch_ranges = {}) {
    require_alignment(q, 16, "q");
    require_alignment(k, 16, "k");
    require_alignment(v, 16, "v");
    require_alignment(g, alignof(float), "g");
    require_alignment(beta, alignof(float), "beta");
    require_alignment(ssm_state_in, 16, "ssm_state_in");
    require_alignment(ssm_state_out, 16, "ssm_state_out");
    require_alignment(out, alignof(std::uint32_t), "out");

    std::array<AddressRange, 11> ranges{};
    std::size_t count = 0;
    ranges[count++]   = address_range(q, "q");
    ranges[count++]   = address_range(k, "k");
    ranges[count++]   = address_range(v, "v");
    ranges[count++]   = address_range(g, "g");
    ranges[count++]   = address_range(beta, "beta");
    ranges[count++]   = address_range(ssm_state_in, "ssm_state_in");
    ranges[count++]   = address_range(out, "out");
    if (!exact_alias(ssm_state_in, ssm_state_out)) {
        ranges[count++] = address_range(ssm_state_out, "ssm_state_out");
    }
    for (const AddressRange& range : scratch_ranges) { ranges[count++] = range; }
    require_disjoint(std::span<const AddressRange>(ranges.data(), count));
}

Geometry require_geometry(const Tensor& q, const Tensor& v) {
    const Geometry geometry{q.ne[1], v.ne[1], q.ne[2]};
    if (q.ne[0] != detail::gated_delta_net::kStateDim) {
        throw std::invalid_argument("gated_delta_net: state/head dimension must be 128");
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

void require_scale(float scale) {
    const float expected_scale =
        1.0f / std::sqrt(static_cast<float>(detail::gated_delta_net::kStateDim));
    if (!std::isfinite(scale) || scale <= 0.0f || std::abs(scale - expected_scale) > 1.0e-6f) {
        throw std::invalid_argument("gated_delta_net: scale must be 1/sqrt(128)");
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
    require_shape(q, detail::gated_delta_net::kStateDim, geometry.qk_heads, geometry.tokens, 1,
                  "q");
    require_shape(k, detail::gated_delta_net::kStateDim, geometry.qk_heads, geometry.tokens, 1,
                  "k");
    require_shape(v, detail::gated_delta_net::kStateDim, geometry.value_heads, geometry.tokens, 1,
                  "v");
    require_shape(out, detail::gated_delta_net::kStateDim, geometry.value_heads, geometry.tokens, 1,
                  "out");
    require_shape(g, geometry.value_heads, geometry.tokens, 1, 1, "g");
    require_shape(beta, geometry.value_heads, geometry.tokens, 1, 1, "beta");
    require_shape(ssm_state, detail::gated_delta_net::kStateDim, detail::gated_delta_net::kStateDim,
                  geometry.value_heads, 1, "ssm_state");

    require_contiguous_nonnull(q, "q");
    require_contiguous_nonnull(k, "k");
    require_contiguous_nonnull(v, "v");
    require_contiguous_nonnull(g, "g");
    require_contiguous_nonnull(beta, "beta");
    require_contiguous_nonnull(ssm_state, "ssm_state");
    require_contiguous_nonnull(out, "out");

    require_scale(scale);
    return geometry;
}

Geometry validate_recurrent_snapshot(const Tensor& q, const Tensor& k, const Tensor& v,
                                     const Tensor& g, const Tensor& beta, float scale,
                                     const Tensor& ssm_states, const Tensor& valid_columns,
                                     const Tensor& initial_state_slots,
                                     const Tensor& snapshot_base_slots, const Tensor& out) {
    constexpr std::int32_t kMaximumBatch = 8;
    constexpr std::int32_t kMaximumWidth = 16;
    const bool masked                    = valid_columns.data != nullptr;
    require_dtype(q, DType::BF16, "q must be BF16");
    require_dtype(k, DType::BF16, "k must be BF16");
    require_dtype(v, DType::BF16, "v must be BF16");
    require_dtype(out, DType::BF16, "out must be BF16");
    require_dtype(g, DType::FP32, "g must be FP32");
    require_dtype(beta, DType::FP32, "beta must be FP32");
    require_dtype(ssm_states, DType::FP32, "ssm_states must be FP32");
    if (masked) { require_dtype(valid_columns, DType::I32, "valid_columns must be I32"); }
    require_dtype(initial_state_slots, DType::I32, "initial_state_slots must be I32");
    require_dtype(snapshot_base_slots, DType::I32, "snapshot_base_slots must be I32");

    const Geometry geometry  = require_geometry(q, v);
    const std::int32_t batch = q.ne[3];
    if (batch <= 0 || batch > kMaximumBatch || (batch > 1 && geometry.tokens > kMaximumWidth)) {
        throw std::invalid_argument("gated_delta_net: unsupported snapshot B/W domain");
    }
    require_shape(q, detail::gated_delta_net::kStateDim, geometry.qk_heads, geometry.tokens, batch,
                  "q");
    require_shape(k, detail::gated_delta_net::kStateDim, geometry.qk_heads, geometry.tokens, batch,
                  "k");
    require_shape(v, detail::gated_delta_net::kStateDim, geometry.value_heads, geometry.tokens,
                  batch, "v");
    require_shape(out, detail::gated_delta_net::kStateDim, geometry.value_heads, geometry.tokens,
                  batch, "out");
    require_shape(g, geometry.value_heads, geometry.tokens, batch, 1, "g");
    require_shape(beta, geometry.value_heads, geometry.tokens, batch, 1, "beta");
    if (ssm_states.ne[0] != detail::gated_delta_net::kStateDim ||
        ssm_states.ne[1] != detail::gated_delta_net::kStateDim ||
        ssm_states.ne[2] != geometry.value_heads || ssm_states.ne[3] < geometry.tokens * batch) {
        throw std::invalid_argument("gated_delta_net: invalid shape for ssm_states snapshot");
    }
    if (masked) { require_shape(valid_columns, batch, 1, 1, 1, "valid_columns"); }
    require_shape(initial_state_slots, batch, 1, 1, 1, "initial_state_slots");
    require_shape(snapshot_base_slots, batch, 1, 1, 1, "snapshot_base_slots");

    require_contiguous_nonnull(q, "q");
    require_contiguous_nonnull(k, "k");
    require_contiguous_nonnull(v, "v");
    require_contiguous_nonnull(g, "g");
    require_contiguous_nonnull(beta, "beta");
    require_contiguous_nonnull(ssm_states, "ssm_states");
    if (masked) { require_contiguous_nonnull(valid_columns, "valid_columns"); }
    require_contiguous_nonnull(initial_state_slots, "initial_state_slots");
    require_contiguous_nonnull(snapshot_base_slots, "snapshot_base_slots");
    require_contiguous_nonnull(out, "out");

    require_scale(scale);
    return geometry;
}

void validate_chunked(const Tensor& q, const Tensor& k, const Tensor& v, const Tensor& g,
                      const Tensor& beta, float scale, const Tensor& ssm_state_in,
                      const Tensor& ssm_state_out, const Tensor& out) {
    // ssm_state_out carries the running-state contract validated by validate_recurrent;
    // ssm_state_in is an equally-shaped read view (may alias ssm_state_out for in-place).
    const Geometry geometry = validate_recurrent(q, k, v, g, beta, scale, ssm_state_out, out);
    require_dtype(ssm_state_in, DType::FP32, "ssm_state_in must be FP32");
    require_shape(ssm_state_in, detail::gated_delta_net::kStateDim,
                  detail::gated_delta_net::kStateDim, geometry.value_heads, 1, "ssm_state_in");
    require_contiguous_nonnull(ssm_state_in, "ssm_state_in");
}

struct ChunkedWorkspace {
    Tensor normalized_q;
    Tensor normalized_k;
    DeviceSpan stage;
};

template <class Allocator>
ChunkedWorkspace allocate_chunked_workspace(Allocator& allocator, std::int32_t qk_heads,
                                            std::int32_t value_heads, std::int32_t tokens,
                                            bool normalize_qk) {
    ChunkedWorkspace out;
    const std::int32_t full =
        (tokens / detail::gated_delta_net::kChunkSize) * detail::gated_delta_net::kChunkSize;
    if (full == 0) { return out; }
    if (normalize_qk) {
        // In the exact Qwen4 geometry, each chunk-output CTA owns one distinct Q head and
        // normalizes it directly into its resident shared-memory tile. Retain materialized Q for
        // every other profile.
        if (!detail::gated_delta_net::uses_fused_q_normalization(qk_heads, value_heads, tokens,
                                                                 normalize_qk)) {
            out.normalized_q = allocator.alloc(
                DType::FP16, {detail::gated_delta_net::kStateDim, qk_heads, tokens});
        }
        out.normalized_k =
            allocator.alloc(DType::FP16, {detail::gated_delta_net::kStateDim, qk_heads, tokens});
    }
    out.stage =
        allocator.alloc_bytes(detail::gated_delta_net::chunked_workspace_bytes(value_heads, full));
    return out;
}

} // namespace

std::size_t gated_delta_net_workspace_capacity_bytes(std::int32_t qk_heads,
                                                     std::int32_t value_heads, bool normalize_qk,
                                                     std::int32_t min_tokens,
                                                     std::int32_t max_tokens) {
    if (!detail::gated_delta_net::are_head_counts_valid(qk_heads, value_heads) || min_tokens <= 0 ||
        max_tokens < min_tokens) {
        throw std::invalid_argument("gated_delta_net workspace: invalid profile or interval");
    }
    WorkspaceLayoutBuilder layout;
    (void)allocate_chunked_workspace(layout, qk_heads, value_heads, max_tokens, normalize_qk);
    std::size_t peak = layout.peak_bytes(1);
    // The fused route drops normalized-Q storage at its measured crossover, so workspace is not
    // monotonic at that boundary. Preserve the interval query's maximum-capacity contract.
    if (!detail::gated_delta_net::uses_fused_q_normalization(
            qk_heads, value_heads, min_tokens, normalize_qk) &&
        detail::gated_delta_net::uses_fused_q_normalization(qk_heads, value_heads, max_tokens,
                                                            normalize_qk)) {
        WorkspaceLayoutBuilder before_fusion;
        (void)allocate_chunked_workspace(before_fusion, qk_heads, value_heads,
                                         detail::gated_delta_net::kFusedQNormalizationMinTokens - 1,
                                         normalize_qk);
        peak = std::max(peak, before_fusion.peak_bytes(1));
    }
    return peak;
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
    validate_recurrent_storage(q, k, v, g, beta, ssm_state, ssm_state, out);

    (void)ws;
    detail::gated_delta_net::launch_recurrent(q, k, v, g, beta, scale, normalize_qk, ssm_state, out,
                                              stream);
}

void gated_delta_net_snapshot(const Tensor& q, const Tensor& k, const Tensor& v, const Tensor& g,
                              const Tensor& beta, float scale, bool normalize_qk,
                              Tensor& ssm_states, const Tensor& valid_columns,
                              const Tensor& initial_state_slots, const Tensor& snapshot_base_slots,
                              Tensor& out, cudaStream_t stream) {
    validate_recurrent_snapshot(q, k, v, g, beta, scale, ssm_states, valid_columns,
                                initial_state_slots, snapshot_base_slots, out);

    detail::gated_delta_net::launch_recurrent_snapshot(
        q, k, v, g, beta, scale, normalize_qk, ssm_states, valid_columns, initial_state_slots,
        snapshot_base_slots, out, stream);
}

void gated_delta_net(const Tensor& q, const Tensor& k, const Tensor& v, const Tensor& g,
                     const Tensor& beta, float scale, bool normalize_qk, WorkspaceArena& ws,
                     const Tensor& ssm_state_in, Tensor& ssm_state_out, Tensor& out,
                     cudaStream_t stream) {
    validate_chunked(q, k, v, g, beta, scale, ssm_state_in, ssm_state_out, out);
    validate_recurrent_storage(q, k, v, g, beta, ssm_state_in, ssm_state_out, out);

    auto scratch_scope   = ws.scope();
    const std::int32_t T = q.ne[2];
    const std::int32_t T_full =
        (T / detail::gated_delta_net::kChunkSize) * detail::gated_delta_net::kChunkSize;
    const bool fuse_q_normalization =
        detail::gated_delta_net::uses_fused_q_normalization(q.ne[1], v.ne[1], T, normalize_qk);
    const bool fuse_k_normalization =
        detail::gated_delta_net::uses_fused_k_normalization(q.ne[1], v.ne[1], T, normalize_qk);
    ChunkedWorkspace scratch = allocate_chunked_workspace(ws, q.ne[1], v.ne[1], T, normalize_qk);
    std::array<AddressRange, 3> scratch_ranges{};
    std::size_t scratch_count = 0;
    if (scratch.normalized_q.data != nullptr) {
        scratch_ranges[scratch_count++] = address_range(scratch.normalized_q, "normalized_q");
    }
    if (scratch.normalized_k.data != nullptr) {
        scratch_ranges[scratch_count++] = address_range(scratch.normalized_k, "normalized_k");
    }
    if (scratch.stage.data != nullptr) {
        scratch_ranges[scratch_count++] =
            address_range(scratch.stage.data, scratch.stage.bytes, "chunked_workspace");
    }
    validate_recurrent_storage(
        q, k, v, g, beta, ssm_state_in, ssm_state_out, out,
        std::span<const AddressRange>(scratch_ranges.data(), scratch_count));
    if (normalize_qk && T_full > 0) {
        Tensor k_source     = k.slice(2, 0, T_full);
        Tensor k_normalized = scratch.normalized_k.slice(2, 0, T_full);
        if (!fuse_q_normalization) {
            Tensor q_source     = q.slice(2, 0, T_full);
            Tensor q_normalized = scratch.normalized_q.slice(2, 0, T_full);
            detail::gated_delta_net::launch_normalize_fp16(q_source, q_normalized, stream);
        }
        if (!fuse_k_normalization) {
            detail::gated_delta_net::launch_normalize_fp16(k_source, k_normalized, stream);
        }
    }
    if (T_full > 0) {
        Tensor q_full = normalize_qk && !fuse_q_normalization
                            ? scratch.normalized_q.slice(2, 0, T_full)
                            : q.slice(2, 0, T_full);
        Tensor k_full =
            normalize_qk ? scratch.normalized_k.slice(2, 0, T_full) : k.slice(2, 0, T_full);
        Tensor v_full    = v.slice(2, 0, T_full);
        Tensor g_full    = g.slice(1, 0, T_full);
        Tensor beta_full = beta.slice(1, 0, T_full);
        Tensor out_full   = out.slice(2, 0, T_full);
        Tensor raw_k_full = k.slice(2, 0, T_full);
        detail::gated_delta_net::launch_chunked(
            q_full, raw_k_full, k_full, v_full, g_full, beta_full, scale, fuse_q_normalization,
            fuse_k_normalization, ssm_state_in, ssm_state_out, out_full, scratch.stage.data,
            scratch.stage.bytes, stream);
    }

    const std::int32_t tail = T - T_full;
    if (tail > 0) {
        Tensor q_tail    = q.slice(2, T_full, tail);
        Tensor k_tail    = k.slice(2, T_full, tail);
        Tensor v_tail    = v.slice(2, T_full, tail);
        Tensor g_tail    = g.slice(1, T_full, tail);
        Tensor beta_tail = beta.slice(1, T_full, tail);
        Tensor out_tail  = out.slice(2, T_full, tail);
        // After full chunks the running state lives in ssm_state_out; a tail-only run (no full
        // chunks) reads the caller-provided ssm_state_in. Either way the tail publishes to
        // ssm_state_out.
        const Tensor& tail_in = (T_full > 0) ? ssm_state_out : ssm_state_in;
        detail::gated_delta_net::launch_recurrent_inout(q_tail, k_tail, v_tail, g_tail, beta_tail,
                                                        scale, normalize_qk, tail_in, ssm_state_out,
                                                        out_tail, stream);
    }
}

} // namespace ninfer::ops
