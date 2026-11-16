// ninfer::ops - hyper_connection wrapper: implements the public api, validates parameters,
// partitions tokens into workspace-bounded chunks, and dispatches to the launcher.
// Host-compiled; never includes the kernel header.
#include "ninfer/ops/hyper_connection.h"

#include "ops/launcher/hyper_connection.h"
#include "ops/linear/fp8/fp8_format.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>

namespace ninfer::ops {
namespace {

// Reads of at least kHyperConnectionMmaMinTokens tokens run in chunks of up to
// kHyperConnectionMmaChunkTokens on the prefill Tensor Core route (6496 workspace bytes per token,
// about 12.7 MiB per chunk); shorter reads, and a remainder below the route threshold, run in
// chunks of up to kHyperConnectionSimtChunkTokens on the decode kernels (SIMT, or small-T Tensor
// Core kernels sharing the partial/square-sum workspace layout). Partials and square sums use
// 5200 bytes per token; separately finished gate inputs add 1296 bytes per token.
constexpr std::size_t kWorkspaceAlign = 256;

constexpr std::size_t partials_bytes(std::int32_t tokens) {
    return static_cast<std::size_t>(detail::kHyperConnectionKSplits) * tokens *
           detail::kHyperConnectionProjectedRows * sizeof(float);
}

constexpr std::size_t square_sums_bytes(std::int32_t tokens) {
    return static_cast<std::size_t>(detail::kHyperConnectionKSplits) * tokens * sizeof(float);
}

constexpr std::size_t mma_u_bytes(std::int32_t tokens) {
    return static_cast<std::size_t>(tokens) * kHyperConnectionRank * sizeof(std::uint16_t);
}

constexpr std::size_t mma_inverse_rms_bytes(std::int32_t tokens) {
    return static_cast<std::size_t>(tokens) * kHyperConnectionStreams * sizeof(float);
}

constexpr std::size_t align_up(std::size_t bytes) {
    return (bytes + kWorkspaceAlign - 1) / kWorkspaceAlign * kWorkspaceAlign;
}

bool aligned16(const void* pointer) {
    return (reinterpret_cast<std::uintptr_t>(pointer) % 16) == 0;
}

void require(bool condition, const char* op, const char* what) {
    if (!condition) { throw std::invalid_argument(std::string(op) + ": " + what); }
}

void require_tensor(const Tensor& t, DType dtype, std::int32_t ne0, std::int32_t ne1,
                    std::int32_t ne2, const char* op, const char* what) {
    require(t.dtype == dtype && t.ne[0] == ne0 && t.ne[1] == ne1 && t.ne[2] == ne2 &&
                t.ne[3] == 1 && t.is_contiguous() && t.data != nullptr && aligned16(t.data),
            op, what);
}

const std::uint16_t* bf16_data(const Tensor& t) {
    return static_cast<const std::uint16_t*>(t.data);
}

bool overlaps(const void* a, std::size_t a_bytes, const void* b, std::size_t b_bytes) {
    const auto a_begin = reinterpret_cast<std::uintptr_t>(a);
    const auto b_begin = reinterpret_cast<std::uintptr_t>(b);
    return a_begin < b_begin + b_bytes && b_begin < a_begin + a_bytes;
}

bool overlaps(const Tensor& a, const Tensor& b) {
    return overlaps(a.data, a.bytes(), b.data, b.bytes());
}

bool overlaps(const Tensor& a, const Weight& b) {
    return overlaps(a.data, a.bytes(), b.payload, static_cast<std::size_t>(b.payload_bytes));
}

// Validates one FP8_E4M3FN_ROW_BF16S RowScale weight [rows, inputs] and returns its planes.
detail::HyperConnectionFp8Planes fp8_planes(const Weight& weight, std::int32_t rows,
                                            std::int32_t inputs, const char* op, const char* what) {
    require(weight.qtype == QType::FP8_E4M3FN_ROW_BF16S && weight.n == rows && weight.k == inputs,
            op, what);
    (void)detail::validate_fp8_weight(weight, op);
    return {static_cast<const std::uint8_t*>(weight.qdata),
            static_cast<const std::uint16_t*>(weight.scales)};
}

} // namespace

void hyper_connection_broadcast(const Tensor& embedding, Tensor& residual, cudaStream_t stream) {
    constexpr const char* kOp = "hyper_connection_broadcast";
    const std::int32_t tokens = embedding.ne[1];
    require(tokens >= 1, kOp, "T must be positive");
    require_tensor(embedding, DType::BF16, kHyperConnectionHidden, tokens, 1, kOp,
                   "embedding must be aligned contiguous BF16 [2560,T]");
    require_tensor(residual, DType::BF16, kHyperConnectionHidden, kHyperConnectionStreams, tokens,
                   kOp, "residual must be aligned contiguous BF16 [2560,4,T]");
    require(!overlaps(embedding, residual), kOp, "embedding and residual must be disjoint");
    detail::hyper_connection_broadcast_launch(
        bf16_data(embedding), static_cast<std::uint16_t*>(residual.data), tokens, stream);
}

std::size_t hyper_connection_read_workspace_bytes(std::int32_t max_tokens) {
    if (max_tokens < 1) {
        throw std::invalid_argument("hyper_connection_read_workspace_bytes: T must be positive");
    }
    const bool mma_route = max_tokens >= detail::kHyperConnectionMmaMinTokens;
    const std::int32_t chunk =
        std::min(max_tokens, mma_route ? detail::kHyperConnectionMmaChunkTokens
                                       : detail::kHyperConnectionSimtChunkTokens);
    std::size_t bytes = align_up(partials_bytes(chunk)) + align_up(square_sums_bytes(chunk));
    if (max_tokens >= detail::kHyperConnectionFinishMinTokens) {
        bytes += 2 * align_up(mma_u_bytes(chunk)) + align_up(mma_inverse_rms_bytes(chunk));
    }
    return bytes;
}

void hyper_connection_read(const Tensor& residual, const HyperConnectionWeights& weights, Tensor& x,
                           Tensor& write_scale, WorkspaceArena& workspace, cudaStream_t stream) {
    constexpr const char* kOp = "hyper_connection_read";
    const std::int32_t tokens = residual.ne[2];
    require(tokens >= 1, kOp, "T must be positive");
    require_tensor(residual, DType::BF16, kHyperConnectionHidden, kHyperConnectionStreams, tokens,
                   kOp, "residual must be aligned contiguous BF16 [2560,4,T]");
    require_tensor(weights.norm, DType::BF16, kHyperConnectionFlat, 1, 1, kOp,
                   "norm must be aligned contiguous BF16 [10240]");
    const detail::HyperConnectionFp8Planes down =
        fp8_planes(weights.down, kHyperConnectionRank, kHyperConnectionFlat, kOp,
                   "down must be an FP8_E4M3FN_ROW_BF16S RowScale weight [320,10240]");
    const detail::HyperConnectionFp8Planes up =
        fp8_planes(weights.up, kHyperConnectionFlat, kHyperConnectionRank, kOp,
                   "up must be an FP8_E4M3FN_ROW_BF16S RowScale weight [10240,320]");
    require_tensor(x, DType::BF16, kHyperConnectionHidden, tokens, 1, kOp,
                   "x must be aligned contiguous BF16 [2560,T]");
    require(!overlaps(x, residual), kOp, "x must be disjoint from residual");
    const bool has_write = weights.write.data != nullptr;
    if (has_write) {
        require_tensor(weights.write, DType::BF16, kHyperConnectionFlat, kHyperConnectionStreams, 1,
                       kOp, "write must be aligned contiguous BF16 [4,10240]");
        require_tensor(write_scale, DType::FP32, kHyperConnectionStreams, tokens, 1, kOp,
                       "write_scale must be aligned contiguous FP32 [4,T]");
        require(!overlaps(write_scale, residual) && !overlaps(write_scale, x), kOp,
                "write_scale must be disjoint from residual and x");
    }

    require(!overlaps(x, weights.norm) && !overlaps(x, weights.down) && !overlaps(x, weights.up) &&
                (!has_write || !overlaps(x, weights.write)),
            kOp, "x must be disjoint from the weights");
    if (has_write) {
        require(!overlaps(write_scale, weights.norm) && !overlaps(write_scale, weights.down) &&
                    !overlaps(write_scale, weights.up) && !overlaps(write_scale, weights.write),
                kOp, "write_scale must be disjoint from the weights");
    }

    auto scope           = workspace.scope();
    const bool mma_route = tokens >= detail::kHyperConnectionMmaMinTokens;
    const std::int32_t chunk =
        std::min(tokens, mma_route ? detail::kHyperConnectionMmaChunkTokens
                                   : detail::kHyperConnectionSimtChunkTokens);
    const DeviceSpan partials_span = workspace.alloc_bytes(partials_bytes(chunk), kWorkspaceAlign);
    const DeviceSpan square_sums_span =
        workspace.alloc_bytes(square_sums_bytes(chunk), kWorkspaceAlign);
    DeviceSpan u_hi_span{};
    DeviceSpan u_lo_span{};
    DeviceSpan inverse_rms_span{};
    if (mma_route || (tokens >= detail::kHyperConnectionFinishMinTokens &&
                      tokens <= detail::kHyperConnectionSmallMaxTokens)) {
        u_hi_span        = workspace.alloc_bytes(mma_u_bytes(chunk), kWorkspaceAlign);
        u_lo_span        = workspace.alloc_bytes(mma_u_bytes(chunk), kWorkspaceAlign);
        inverse_rms_span = workspace.alloc_bytes(mma_inverse_rms_bytes(chunk), kWorkspaceAlign);
    }
    for (const DeviceSpan& span :
         {partials_span, square_sums_span, u_hi_span, u_lo_span, inverse_rms_span}) {
        const auto touches = [&](const Tensor& t) {
            return overlaps(span.data, span.bytes, t.data, t.bytes());
        };
        const auto touches_weight = [&](const Weight& w) {
            return overlaps(span.data, span.bytes, w.payload,
                            static_cast<std::size_t>(w.payload_bytes));
        };
        require(!touches(residual) && !touches(x) && !touches(weights.norm) &&
                    !touches_weight(weights.down) && !touches_weight(weights.up) &&
                    (!has_write || (!touches(weights.write) && !touches(write_scale))),
                kOp, "workspace must be disjoint from the inputs and outputs");
    }
    auto* partials    = static_cast<float*>(partials_span.data);
    auto* square_sums = static_cast<float*>(square_sums_span.data);
    const detail::HyperConnectionMmaWorkspace mma{static_cast<std::uint16_t*>(u_hi_span.data),
                                                  static_cast<std::uint16_t*>(u_lo_span.data),
                                                  static_cast<float*>(inverse_rms_span.data)};
    auto* x_data     = static_cast<std::uint16_t*>(x.data);
    auto* write_data = has_write ? static_cast<float*>(write_scale.data) : nullptr;
    for (std::int32_t begin = 0; begin < tokens; begin += chunk) {
        const std::int32_t count = std::min(chunk, tokens - begin);
        detail::hyper_connection_read_launch(
            bf16_data(residual) + static_cast<std::int64_t>(begin) * kHyperConnectionFlat,
            bf16_data(weights.norm), down, up, has_write ? bf16_data(weights.write) : nullptr,
            x_data + static_cast<std::int64_t>(begin) * kHyperConnectionHidden,
            has_write ? write_data + static_cast<std::int64_t>(begin) * kHyperConnectionStreams
                      : nullptr,
            partials, square_sums, mma, count, stream);
    }
}

void hyper_connection_inject(const Tensor& residual, const Tensor& block_output,
                             const Tensor& write_scale, Tensor& residual_out, cudaStream_t stream) {
    constexpr const char* kOp = "hyper_connection_inject";
    const std::int32_t tokens = residual.ne[2];
    require(tokens >= 1, kOp, "T must be positive");
    require_tensor(residual, DType::BF16, kHyperConnectionHidden, kHyperConnectionStreams, tokens,
                   kOp, "residual must be aligned contiguous BF16 [2560,4,T]");
    require_tensor(residual_out, DType::BF16, kHyperConnectionHidden, kHyperConnectionStreams,
                   tokens, kOp, "residual_out must be aligned contiguous BF16 [2560,4,T]");
    require_tensor(block_output, DType::BF16, kHyperConnectionHidden, tokens, 1, kOp,
                   "block_output must be aligned contiguous BF16 [2560,T]");
    require_tensor(write_scale, DType::FP32, kHyperConnectionStreams, tokens, 1, kOp,
                   "write_scale must be aligned contiguous FP32 [4,T]");
    require(residual.data == residual_out.data || !overlaps(residual, residual_out), kOp,
            "residual and residual_out must alias exactly or be disjoint");
    require(!overlaps(block_output, residual_out) && !overlaps(write_scale, residual_out), kOp,
            "block_output and write_scale must be disjoint from residual_out");
    detail::hyper_connection_inject_launch(
        bf16_data(residual), bf16_data(block_output), static_cast<const float*>(write_scale.data),
        static_cast<std::uint16_t*>(residual_out.data), tokens, stream);
}

} // namespace ninfer::ops
