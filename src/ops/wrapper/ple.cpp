// ninfer::ops - PLE wrapper: implements the public api, validates parameters, partitions columns
// into workspace-bounded chunks, and dispatches to the launcher. Host-compiled; never includes
// the kernel header.
#include "ninfer/ops/ple.h"

#include "ops/launcher/ple.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <stdexcept>
#include <string>

namespace ninfer::ops {
namespace {

// Tokens per inject launch sequence: 20 KiB of convolution input and 16 bytes of gates per token,
// about 20 MiB per chunk. A chunk covers the same column range of every row; chunks run in order
// and carry each row's convolution history between them through its state slot.
constexpr std::int32_t kChunkTokens   = 1024;
constexpr std::size_t kWorkspaceAlign = 256;

constexpr std::size_t align_up(std::size_t bytes) {
    return (bytes + kWorkspaceAlign - 1) / kWorkspaceAlign * kWorkspaceAlign;
}

constexpr std::size_t conv_input_bytes(std::int32_t tokens) {
    return static_cast<std::size_t>(tokens) * kPleChannels * sizeof(std::uint16_t);
}

constexpr std::size_t gate_bytes(std::int32_t tokens) {
    return static_cast<std::size_t>(tokens) * kPleStreams * sizeof(float);
}

void require(bool condition, const char* op, const char* what) {
    if (!condition) { throw std::invalid_argument(std::string(op) + ": " + what); }
}

bool shaped(const Tensor& t, DType dtype, std::int32_t ne0, std::int32_t ne1, std::int32_t ne2,
            std::size_t alignment) {
    return t.dtype == dtype && t.ne[0] == ne0 && t.ne[1] == ne1 && t.ne[2] == ne2 && t.ne[3] == 1 &&
           t.is_contiguous() && t.data != nullptr &&
           (reinterpret_cast<std::uintptr_t>(t.data) % alignment) == 0;
}

bool overlaps(const Tensor& a, const Tensor& b) {
    const auto a_begin = reinterpret_cast<std::uintptr_t>(a.data);
    const auto b_begin = reinterpret_cast<std::uintptr_t>(b.data);
    return a_begin < b_begin + b.bytes() && b_begin < a_begin + a.bytes();
}

const std::uint16_t* bf16(const Tensor& t) { return static_cast<const std::uint16_t*>(t.data); }

} // namespace

void ple_decode_rows(const Tensor& rows, Tensor& embedding, cudaStream_t stream) {
    constexpr const char* kOp = "ple_decode_rows";
    const std::int32_t tokens = rows.ne[2];
    require(tokens >= 1 && tokens <= 65535, kOp, "T must be in [1,65535]");
    require(shaped(rows, DType::U8, kPleInt8RowBytes, kPleRowsPerToken, tokens, 2), kOp,
            "rows must be 2-byte aligned contiguous U8 [162,16,T]");
    require(shaped(embedding, DType::BF16, kPleEmbedding, tokens, 1, 16), kOp,
            "embedding must be aligned contiguous BF16 [2560,T]");
    require(!overlaps(rows, embedding), kOp, "rows and embedding must be disjoint");
    detail::ple_decode_rows_launch(static_cast<const std::uint8_t*>(rows.data),
                                   static_cast<std::uint16_t*>(embedding.data), tokens, stream);
}

std::size_t ple_inject_workspace_bytes(std::int32_t max_tokens) {
    if (max_tokens < 1) {
        throw std::invalid_argument("ple_inject_workspace_bytes: T must be positive");
    }
    const std::int32_t chunk = std::min(max_tokens, kChunkTokens);
    return align_up(conv_input_bytes(chunk)) + align_up(gate_bytes(chunk));
}

namespace {

constexpr std::int32_t kMaximumBatch         = 12;
constexpr std::int32_t kMaximumSnapshotWidth = 16;

void require_selector(const Tensor& selector, std::int32_t batch, const char* op,
                      const char* what) {
    require(shaped(selector, DType::I32, batch, 1, 1, 4), op, what);
}

// Validates a packed [*, W, B] call and runs it in chunks of whole rows' column ranges.
void inject(Tensor& residual, const Tensor& key_projection, const Tensor& value_projection,
            const PleWeights& weights, Tensor& conv_states, const Tensor& valid_columns,
            const Tensor& initial_state_slots, const Tensor& publish_slots,
            detail::PleStatePublication publication, WorkspaceArena& workspace, cudaStream_t stream,
            const char* op) {
    const std::int32_t width = residual.ne[2];
    const std::int32_t batch = residual.ne[3];
    require(width >= 1 && batch >= 1 && batch <= kMaximumBatch, op,
            "residual must hold W >= 1 columns of 1..12 rows");
    if (publication == detail::PleStatePublication::EveryColumn) {
        require(width <= kMaximumSnapshotWidth, op, "a snapshot call admits W in [1,16]");
    }
    require(residual.dtype == DType::BF16 && residual.ne[0] == kPleHidden &&
                residual.ne[1] == kPleStreams && residual.is_contiguous() &&
                residual.data != nullptr &&
                (reinterpret_cast<std::uintptr_t>(residual.data) % 16) == 0,
            op, "residual must be aligned contiguous BF16 [2560,4,W,B]");
    require(shaped(key_projection, DType::FP32, kPleChannels, width, batch, 16), op,
            "key_projection must be aligned contiguous FP32 [10240,W,B]");
    require(shaped(value_projection, DType::FP32, kPleHidden, width, batch, 16), op,
            "value_projection must be aligned contiguous FP32 [2560,W,B]");
    for (const Tensor* norm : {&weights.norm_key, &weights.norm_query, &weights.norm_conv}) {
        require(shaped(*norm, DType::BF16, kPleChannels, 1, 1, 16), op,
                "norm weights must be aligned contiguous BF16 [10240]");
    }
    require(shaped(weights.conv, DType::BF16, kPleConvWidth, kPleChannels, 1, 16), op,
            "conv must be aligned contiguous BF16 [4,10240]");
    require(conv_states.ne[2] >= 1 && shaped(conv_states, DType::BF16, kPleConvHistory,
                                             kPleChannels, conv_states.ne[2], 16),
            op, "conv_states must be aligned contiguous BF16 [9,10240,S]");
    require_selector(initial_state_slots, batch, op, "state slots must be device I32 [B]");
    require_selector(publish_slots, batch, op, "state slots must be device I32 [B]");
    if (valid_columns.data != nullptr) {
        require_selector(valid_columns, batch, op, "valid_columns must be empty or I32 [B]");
    }
    const std::initializer_list<const Tensor*> inputs{
        &key_projection,      &value_projection,  &weights.norm_key,
        &weights.norm_query,  &weights.norm_conv, &weights.conv,
        &initial_state_slots, &publish_slots,     &valid_columns};
    for (const Tensor* other : inputs) {
        if (other->data == nullptr) { continue; }
        require(!overlaps(residual, *other) && !overlaps(conv_states, *other), op,
                "residual and conv_states must be disjoint from every input");
    }
    require(!overlaps(residual, conv_states), op, "residual and conv_states must be disjoint");

    // Columns per chunk: whole column ranges of every row, at most kChunkTokens tokens in total.
    const std::int32_t chunk_columns = std::min(width, std::max(1, kChunkTokens / batch));
    const std::int32_t chunk_tokens  = chunk_columns * batch;
    // A snapshot call (W <= 16, B <= 12) is always one chunk, so it reads one initial history.
    auto scope       = workspace.scope();
    auto* conv_input = static_cast<std::uint16_t*>(
        workspace.alloc_bytes(conv_input_bytes(chunk_tokens), kWorkspaceAlign).data);
    auto* gate =
        static_cast<float*>(workspace.alloc_bytes(gate_bytes(chunk_tokens), kWorkspaceAlign).data);
    detail::PleInjectChunk chunk{width,
                                 batch,
                                 0,
                                 0,
                                 static_cast<const std::int32_t*>(valid_columns.data),
                                 static_cast<const std::int32_t*>(initial_state_slots.data),
                                 static_cast<const std::int32_t*>(publish_slots.data)};
    for (std::int32_t begin = 0; begin < width; begin += chunk_columns) {
        chunk.chunk_begin   = begin;
        chunk.chunk_columns = std::min(chunk_columns, width - begin);
        detail::ple_inject_launch(static_cast<std::uint16_t*>(residual.data),
                                  static_cast<const float*>(key_projection.data),
                                  static_cast<const float*>(value_projection.data),
                                  bf16(weights.norm_key), bf16(weights.norm_query),
                                  bf16(weights.norm_conv), bf16(weights.conv),
                                  static_cast<std::uint16_t*>(conv_states.data), conv_input, gate,
                                  chunk, publication, stream);
    }
}

} // namespace

void ple_inject(Tensor& residual, const Tensor& key_projection, const Tensor& value_projection,
                const PleWeights& weights, Tensor& conv_states, const Tensor& valid_columns,
                const Tensor& state_slots, WorkspaceArena& workspace, cudaStream_t stream) {
    inject(residual, key_projection, value_projection, weights, conv_states, valid_columns,
           state_slots, state_slots, detail::PleStatePublication::Final, workspace, stream,
           "ple_inject");
}

void ple_inject_snapshot(Tensor& residual, const Tensor& key_projection,
                         const Tensor& value_projection, const PleWeights& weights,
                         Tensor& conv_states, const Tensor& valid_columns,
                         const Tensor& initial_state_slots, const Tensor& snapshot_base_slots,
                         WorkspaceArena& workspace, cudaStream_t stream) {
    inject(residual, key_projection, value_projection, weights, conv_states, valid_columns,
           initial_state_slots, snapshot_base_slots, detail::PleStatePublication::EveryColumn,
           workspace, stream, "ple_inject_snapshot");
}

} // namespace ninfer::ops
