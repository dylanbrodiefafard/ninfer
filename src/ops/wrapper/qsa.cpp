// ninfer::ops - QSA wrapper: implements the public api, validates parameters, carves the
// workspace, and dispatches to the launcher. Host-compiled; never includes the kernel code.
#include "ninfer/ops/qsa.h"

#include "ops/launcher/qsa.h"

#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <stdexcept>
#include <string>
#include <vector>

namespace ninfer::ops {
namespace {

// The launch grids carry W in a dimension limited to 65535.
constexpr std::int32_t kMaximumWidth  = 65535;
constexpr std::size_t kWorkspaceAlign = 256;

constexpr std::size_t align_up(std::size_t bytes) {
    return (bytes + kWorkspaceAlign - 1) / kWorkspaceAlign * kWorkspaceAlign;
}

void require(bool condition, const char* op, const char* what) {
    if (!condition) { throw std::invalid_argument(std::string(op) + ": " + what); }
}

bool aligned16(const void* pointer) {
    return (reinterpret_cast<std::uintptr_t>(pointer) % 16) == 0;
}

void require_tensor(const Tensor& t, DType dtype, std::int32_t ne0, std::int32_t ne1,
                    std::int32_t ne2, const char* op, const char* what) {
    require(t.dtype == dtype && t.ne[0] == ne0 && t.ne[1] == ne1 && t.ne[2] == ne2 &&
                t.ne[3] == 1 && t.is_contiguous() && t.data != nullptr && aligned16(t.data),
            op, what);
}

bool overlaps(const Tensor& a, const Tensor& b) {
    const auto a_begin = reinterpret_cast<std::uintptr_t>(a.data);
    const auto b_begin = reinterpret_cast<std::uintptr_t>(b.data);
    return a_begin < b_begin + b.bytes() && b_begin < a_begin + a.bytes();
}

void require_disjoint(std::initializer_list<const Tensor*> tensors, const char* op,
                      const char* what) {
    for (auto first = tensors.begin(); first != tensors.end(); ++first) {
        for (auto second = first + 1; second != tensors.end(); ++second) {
            require(!overlaps(**first, **second), op, what);
        }
    }
}

bool is_budget(QsaSelectBudget budget) {
    return budget == QsaSelectBudget::Blocks512 || budget == QsaSelectBudget::Blocks1024 ||
           budget == QsaSelectBudget::Blocks2048;
}

static_assert(kQsaPageTokens == kPagedKVPageSize);

bool plane_shaped(const Tensor& t, DType dtype, std::int32_t lead, std::int32_t heads,
                  std::int32_t pages) {
    return t.dtype == dtype && t.ne[0] == lead && t.ne[1] == kQsaPageTokens && t.ne[2] == heads &&
           t.ne[3] == pages && t.is_contiguous() && t.data != nullptr && aligned16(t.data);
}

// Validates every plane and the block tables of the paged state and returns its raw planes.
detail::QsaStatePlanes state_planes(const QsaPagedState& state, const char* op) {
    const std::int32_t pages = state.index_keys.ne[3];
    require(pages >= 1, op, "state must hold at least one page");
    require(plane_shaped(state.index_keys, DType::BF16, kQsaIndexHeadDim, 1, pages), op,
            "state index_keys must be aligned contiguous BF16 [128,64,1,P]");
    require(plane_shaped(state.positions, DType::I32, 1, 3, pages), op,
            "state positions must be aligned contiguous I32 [1,64,3,P]");
    require(plane_shaped(state.block_keys, DType::FP32,
                         kQsaPageBlocks * kQsaIndexHeadDim / kQsaPageTokens, 1, pages),
            op, "state block_keys must be aligned contiguous FP32 [32,64,1,P]");
    const Tensor& tables = state.block_tables;
    require(tables.dtype == DType::I32 && tables.ne[0] >= 1 &&
                tables.ne[0] <= kQsaMaximumCapacity / kQsaPageTokens && tables.ne[1] >= 1 &&
                tables.ne[2] == 1 && tables.ne[3] == 1 && tables.is_contiguous() &&
                tables.data != nullptr && aligned16(tables.data),
            op, "block_tables must be aligned contiguous I32 [L,rows] with 64 L <= 262144");
    detail::QsaStatePlanes planes{state.k.data,
                                  state.v.data,
                                  nullptr,
                                  nullptr,
                                  static_cast<std::uint16_t*>(state.index_keys.data),
                                  static_cast<std::int32_t*>(state.positions.data),
                                  static_cast<float*>(state.block_keys.data),
                                  static_cast<const std::int32_t*>(tables.data),
                                  tables.ne[0],
                                  tables.ne[1],
                                  pages};
    if (state.format == QsaKvFormat::BF16) {
        require(plane_shaped(state.k, DType::BF16, kQsaHeadDim, kQsaKvHeads, pages), op,
                "BF16 state k must be aligned contiguous BF16 [256,64,2,P]");
        require(plane_shaped(state.v, DType::BF16, kQsaHeadDim, kQsaKvHeads, pages), op,
                "BF16 state v must be aligned contiguous BF16 [256,64,2,P]");
        require(state.k_scales.data == nullptr && state.v_scales.data == nullptr, op,
                "BF16 state has no scale planes");
        require_disjoint({&state.k, &state.v, &state.index_keys, &state.positions,
                          &state.block_keys, &state.block_tables},
                         op, "state planes must be disjoint");
        return planes;
    }
    require(state.format == QsaKvFormat::NVFP4G16, op, "unknown state format");
    require(plane_shaped(state.k, DType::U8, kQsaHeadDim / 2, kQsaKvHeads, pages), op,
            "NVFP4 state k must be aligned contiguous U8 [128,64,2,P]");
    require(plane_shaped(state.v, DType::U8, kQsaHeadDim / 2, kQsaKvHeads, pages), op,
            "NVFP4 state v must be aligned contiguous U8 [128,64,2,P]");
    require(plane_shaped(state.k_scales, DType::FP8_E4M3FN, kQsaHeadDim / 16, kQsaKvHeads, pages),
            op, "NVFP4 state k_scales must be aligned contiguous FP8_E4M3FN [16,64,2,P]");
    require(plane_shaped(state.v_scales, DType::FP8_E4M3FN, kQsaHeadDim / 16, kQsaKvHeads, pages),
            op, "NVFP4 state v_scales must be aligned contiguous FP8_E4M3FN [16,64,2,P]");
    require_disjoint({&state.k, &state.v, &state.k_scales, &state.v_scales, &state.index_keys,
                      &state.positions, &state.block_keys, &state.block_tables},
                     op, "state planes must be disjoint");
    planes.k_scales = static_cast<std::uint8_t*>(state.k_scales.data);
    planes.v_scales = static_cast<std::uint8_t*>(state.v_scales.data);
    return planes;
}

// Every state plane present in the paged state; scale planes are null for BF16 state.
void require_disjoint_from_state(const Tensor& tensor, const QsaPagedState& state, const char* op,
                                 const char* what) {
    for (const Tensor* plane :
         {&state.k, &state.v, &state.k_scales, &state.v_scales, &state.index_keys, &state.positions,
          &state.block_keys, &state.block_tables}) {
        require(plane->data == nullptr || !overlaps(tensor, *plane), op, what);
    }
}

// Validates the device slots I32 [W,B] and lanes I32 [B] of a lane-batched call.
detail::QsaColumns columns_of(const Tensor& slots, const Tensor& lanes, const char* op) {
    const std::int32_t width = slots.ne[0];
    const std::int32_t rows  = slots.ne[1];
    require(width >= 1 && rows >= 1 && static_cast<std::int64_t>(width) * rows <= kMaximumWidth, op,
            "W*B must be in [1,65535]");
    require_tensor(slots, DType::I32, width, rows, 1, op,
                   "slots must be aligned contiguous I32 [W,B]");
    require_tensor(lanes, DType::I32, rows, 1, 1, op, "lanes must be aligned contiguous I32 [B]");
    return {static_cast<const std::int32_t*>(slots.data),
            static_cast<const std::int32_t*>(lanes.data), width, rows};
}

const std::uint16_t* bf16_data(const Tensor& t) {
    return static_cast<const std::uint16_t*>(t.data);
}

const std::int32_t* i32_data(const Tensor& t) { return static_cast<const std::int32_t*>(t.data); }

struct SelectLayout {
    std::int32_t blocks;
    std::int32_t chunk;
};

constexpr SelectLayout select_layout(std::int32_t width, std::int32_t max_visible_tokens) {
    const std::int32_t blocks = max_visible_tokens / kQsaBlockTokens;
    return {blocks, blocks > 0 ? detail::qsa_score_chunk_queries(width, blocks) : 0};
}

constexpr std::size_t query_bytes(std::int32_t width) {
    return static_cast<std::size_t>(width) * kQsaIndexQueryHeads * kQsaIndexHeadDim * sizeof(float);
}

constexpr std::size_t score_bytes(const SelectLayout& layout) {
    return static_cast<std::size_t>(layout.chunk) * static_cast<std::size_t>(layout.blocks) *
           sizeof(float);
}

std::size_t attention_accumulator_bytes(std::int32_t width, std::int32_t splits) {
    return static_cast<std::size_t>(width) * kQsaQueryHeads * static_cast<std::size_t>(splits) *
           kQsaHeadDim * sizeof(float);
}

std::size_t attention_stats_bytes(std::int32_t width, std::int32_t splits) {
    return static_cast<std::size_t>(width) * kQsaQueryHeads * static_cast<std::size_t>(splits) * 2 *
           sizeof(float);
}

} // namespace

std::vector<PagedKVPlaneSpec> qsa_paged_plane_specs(QsaKvFormat format) {
    constexpr std::size_t kAlignment = 256;
    const PagedKVPlaneSpec index_keys{DType::BF16, kQsaIndexHeadDim, 1, kAlignment};
    const PagedKVPlaneSpec positions{DType::I32, 1, 3, kAlignment};
    const PagedKVPlaneSpec block_keys{
        DType::FP32, kQsaPageBlocks * kQsaIndexHeadDim / kQsaPageTokens, 1, kAlignment};
    if (format == QsaKvFormat::BF16) {
        const PagedKVPlaneSpec kv{DType::BF16, kQsaHeadDim, kQsaKvHeads, kAlignment};
        return {kv, kv, index_keys, positions, block_keys};
    }
    if (format == QsaKvFormat::NVFP4G16) {
        const PagedKVPlaneSpec codes{DType::U8, kQsaHeadDim / 2, kQsaKvHeads, kAlignment};
        const PagedKVPlaneSpec scales{DType::FP8_E4M3FN, kQsaHeadDim / 16, kQsaKvHeads, kAlignment};
        return {codes, codes, scales, scales, index_keys, positions, block_keys};
    }
    throw std::invalid_argument("qsa_paged_plane_specs: unknown state format");
}

void qsa_state_append(const Tensor& k, const Tensor& v, const Tensor& index_keys,
                      const Tensor& positions, const Tensor& slots, const Tensor& lanes,
                      const Tensor& key_norm_weight, QsaPagedState& state, cudaStream_t stream) {
    constexpr const char* kOp           = "qsa_state_append";
    const detail::QsaColumns columns    = columns_of(slots, lanes, kOp);
    const detail::QsaStatePlanes planes = state_planes(state, kOp);
    const std::int32_t width            = columns.width;
    const std::int32_t rows             = columns.rows;
    require(k.dtype == DType::BF16 && k.ne[0] == kQsaHeadDim && k.ne[1] == kQsaKvHeads &&
                k.ne[2] == width && k.ne[3] == rows && k.is_contiguous() && k.data != nullptr &&
                aligned16(k.data),
            kOp, "k must be aligned contiguous BF16 [256,2,W,B]");
    require(v.dtype == DType::BF16 && v.ne[0] == kQsaHeadDim && v.ne[1] == kQsaKvHeads &&
                v.ne[2] == width && v.ne[3] == rows && v.is_contiguous() && v.data != nullptr &&
                aligned16(v.data),
            kOp, "v must be aligned contiguous BF16 [256,2,W,B]");
    require_tensor(index_keys, DType::BF16, kQsaIndexHeadDim, width, rows, kOp,
                   "index_keys must be aligned contiguous BF16 [128,W,B]");
    require_tensor(positions, DType::I32, width, 3, rows, kOp,
                   "positions must be aligned contiguous I32 [W,3,B]");
    require_tensor(key_norm_weight, DType::BF16, kQsaIndexHeadDim, 1, 1, kOp,
                   "key_norm_weight must be aligned contiguous BF16 [128]");
    for (const Tensor* input :
         {&k, &v, &index_keys, &positions, &slots, &lanes, &key_norm_weight}) {
        require_disjoint_from_state(*input, state, kOp, "inputs must be disjoint from the state");
    }
    detail::qsa_state_append_launch(state.format, bf16_data(k), bf16_data(v), bf16_data(index_keys),
                                    i32_data(positions), columns, bf16_data(key_norm_weight),
                                    planes, stream);
}

std::size_t qsa_index_select_workspace_bytes(std::int32_t queries, std::int32_t max_visible_tokens,
                                             QsaSelectBudget budget) {
    constexpr const char* kOp = "qsa_index_select_workspace_bytes";
    require(queries >= 1 && queries <= kMaximumWidth, kOp, "W*B must be in [1,65535]");
    require(max_visible_tokens >= 1 && max_visible_tokens <= kQsaMaximumCapacity, kOp,
            "max_visible_tokens must be in [1,262144]");
    require(is_budget(budget), kOp, "unknown selection budget");
    const SelectLayout layout = select_layout(queries, max_visible_tokens);
    if (layout.blocks <= static_cast<std::int32_t>(budget)) { return 0; }
    return align_up(query_bytes(queries)) + align_up(score_bytes(layout));
}

void qsa_index_select(const Tensor& index_query, const QsaPagedState& state, const Tensor& slots,
                      const Tensor& lanes, const Tensor& query_norm_weight, QsaSelectBudget budget,
                      std::int32_t max_visible_tokens, Tensor& selected_ids, Tensor& selected_count,
                      WorkspaceArena& workspace, cudaStream_t stream) {
    constexpr const char* kOp           = "qsa_index_select";
    const detail::QsaColumns columns    = columns_of(slots, lanes, kOp);
    const std::int32_t width            = columns.width;
    const std::int32_t rows             = columns.rows;
    const std::int32_t queries          = width * rows;
    const detail::QsaStatePlanes planes = state_planes(state, kOp);
    require(is_budget(budget), kOp, "unknown selection budget");
    require(max_visible_tokens >= 1 && max_visible_tokens <= kQsaPageTokens * planes.logical_pages,
            kOp, "max_visible_tokens must be in [1,C]");
    require(index_query.dtype == DType::BF16 && index_query.ne[0] == kQsaIndexHeadDim &&
                index_query.ne[1] == kQsaIndexQueryHeads && index_query.ne[2] == width &&
                index_query.ne[3] == rows && index_query.is_contiguous() &&
                index_query.data != nullptr && aligned16(index_query.data),
            kOp, "index_query must be aligned contiguous BF16 [128,4,W,B]");
    require_tensor(query_norm_weight, DType::BF16, kQsaIndexHeadDim, 1, 1, kOp,
                   "query_norm_weight must be aligned contiguous BF16 [128]");
    const std::int32_t selected_slots = qsa_selected_capacity(budget);
    require_tensor(selected_ids, DType::I32, selected_slots, width, rows, kOp,
                   "selected_ids must be aligned contiguous I32 [4*budget+3,W,B]");
    require_tensor(selected_count, DType::I32, width, rows, 1, kOp,
                   "selected_count must be aligned contiguous I32 [W,B]");
    for (const Tensor* input : {&index_query, &slots, &lanes, &query_norm_weight}) {
        require_disjoint_from_state(*input, state, kOp, "inputs must be disjoint from the state");
    }
    for (const Tensor* output : {&selected_ids, &selected_count}) {
        require_disjoint_from_state(*output, state, kOp, "outputs must be disjoint from the state");
        for (const Tensor* input : {&index_query, &slots, &lanes, &query_norm_weight}) {
            require(!overlaps(*output, *input), kOp, "outputs must be disjoint from the inputs");
        }
    }
    require(!overlaps(selected_ids, selected_count), kOp, "outputs must be disjoint");

    auto scope = workspace.scope();
    detail::QsaSelectWorkspace scratch{};
    const SelectLayout layout = select_layout(queries, max_visible_tokens);
    if (layout.blocks > static_cast<std::int32_t>(budget)) {
        scratch.query =
            static_cast<float*>(workspace.alloc_bytes(query_bytes(queries), kWorkspaceAlign).data);
        scratch.scores =
            static_cast<float*>(workspace.alloc_bytes(score_bytes(layout), kWorkspaceAlign).data);
    }
    detail::qsa_index_select_launch(
        bf16_data(index_query), planes, columns, bf16_data(query_norm_weight),
        static_cast<std::int32_t>(budget), max_visible_tokens,
        static_cast<std::int32_t*>(selected_ids.data),
        static_cast<std::int32_t*>(selected_count.data), scratch, stream);
}

std::size_t qsa_selected_attention_workspace_bytes(std::int32_t queries,
                                                   std::int32_t selected_slots) {
    constexpr const char* kOp = "qsa_selected_attention_workspace_bytes";
    require(queries >= 1 && queries <= kMaximumWidth, kOp, "W*B must be in [1,65535]");
    require(selected_slots >= 1 &&
                selected_slots <= qsa_selected_capacity(QsaSelectBudget::Blocks2048),
            kOp, "S must be in [1,8195]");
    const std::int32_t splits = detail::qsa_attention_splits(queries, selected_slots);
    if (splits == 1) { return 0; }
    return align_up(attention_accumulator_bytes(queries, splits)) +
           align_up(attention_stats_bytes(queries, splits));
}

void qsa_selected_attention(const Tensor& q, const Tensor& selected_ids,
                            const Tensor& selected_count, const Tensor& lanes,
                            const QsaPagedState& state, Tensor& out, WorkspaceArena& workspace,
                            cudaStream_t stream) {
    constexpr const char* kOp         = "qsa_selected_attention";
    const std::int32_t selected_slots = selected_ids.ne[0];
    // selected_count [W,B] carries the column layout; it stands in for the slots of columns_of.
    const detail::QsaColumns columns = columns_of(selected_count, lanes, kOp);
    const std::int32_t width         = columns.width;
    const std::int32_t rows          = columns.rows;
    const std::int32_t queries       = width * rows;
    require(selected_slots >= 1 &&
                selected_slots <= qsa_selected_capacity(QsaSelectBudget::Blocks2048),
            kOp, "S must be in [1,8195]");
    const detail::QsaStatePlanes planes = state_planes(state, kOp);
    const auto attention_shaped         = [&](const Tensor& t) {
        return t.dtype == DType::BF16 && t.ne[0] == kQsaHeadDim && t.ne[1] == kQsaQueryHeads &&
               t.ne[2] == width && t.ne[3] == rows && t.is_contiguous() && t.data != nullptr &&
               aligned16(t.data);
    };
    require(attention_shaped(q), kOp, "q must be aligned contiguous BF16 [256,24,W,B]");
    require(attention_shaped(out), kOp, "out must be aligned contiguous BF16 [256,24,W,B]");
    require_tensor(selected_ids, DType::I32, selected_slots, width, rows, kOp,
                   "selected_ids must be aligned contiguous I32 [S,W,B]");
    for (const Tensor* tensor :
         {&q, &selected_ids, &selected_count, &lanes, static_cast<const Tensor*>(&out)}) {
        require_disjoint_from_state(*tensor, state, kOp,
                                    "inputs and out must be disjoint from the state");
    }
    require_disjoint({&out, &q, &selected_ids, &selected_count, &lanes}, kOp,
                     "out must be disjoint from the inputs");

    auto scope = workspace.scope();
    detail::QsaAttentionWorkspace scratch{};
    const std::int32_t splits = detail::qsa_attention_splits(queries, selected_slots);
    if (splits > 1) {
        scratch.accumulators = static_cast<float*>(
            workspace.alloc_bytes(attention_accumulator_bytes(queries, splits), kWorkspaceAlign)
                .data);
        scratch.stats = static_cast<float*>(
            workspace.alloc_bytes(attention_stats_bytes(queries, splits), kWorkspaceAlign).data);
    }
    detail::qsa_selected_attention_launch(
        bf16_data(q), i32_data(selected_ids), i32_data(selected_count), columns, state.format,
        planes, selected_slots, static_cast<std::uint16_t*>(out.data), scratch, stream);
}

} // namespace ninfer::ops
