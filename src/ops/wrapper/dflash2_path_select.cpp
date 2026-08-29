// ninfer::ops - dflash2_path_select wrapper: validation, projection dispatch, launcher.
#include "ninfer/ops/dflash2_path_select.h"

#include "ninfer/ops/linear.h"
#include "ops/launcher/dflash2_path_select.h"

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>

namespace ninfer::ops {
namespace {

bool aligned_to(const void* pointer, std::uintptr_t alignment) {
    return pointer != nullptr && (reinterpret_cast<std::uintptr_t>(pointer) & (alignment - 1)) == 0;
}

bool is_quantized_projection(QType qtype) {
    return qtype == QType::Q4G64_F16S || qtype == QType::W8G32_F16S || qtype == QType::NVFP4;
}

void require_sequence_extent(std::int32_t tokens, std::int32_t batch, const char* op) {
    if (tokens <= 0 || batch <= 0 || batch > kDflash2PathSelectMaxBatch) {
        throw std::invalid_argument(std::string(op) + ": T and B must be positive with B<=8");
    }
    if (batch > 1 && tokens > kDflash2PathSelectMaxWidthWhenBatched) {
        throw std::invalid_argument(std::string(op) + ": B=2..8 admits T=1..16");
    }
}

void require_hidden(const Tensor& hidden, std::int32_t tokens, std::int32_t batch) {
    if (hidden.dtype != DType::BF16 || !hidden.is_contiguous() || hidden.data == nullptr) {
        throw std::invalid_argument("dflash2_path_select: hidden must be contiguous BF16");
    }
    if (hidden.ne[0] != kDflash2PathSelectHidden || hidden.ne[1] != tokens ||
        hidden.ne[2] != batch || hidden.ne[3] != 1) {
        throw std::invalid_argument("dflash2_path_select: hidden must be BF16 [5120,T] or [5120,T,B]");
    }
}

void require_logits(const Tensor& logits) {
    if (logits.dtype != DType::BF16 || !logits.is_contiguous() || logits.data == nullptr) {
        throw std::invalid_argument("dflash2_path_select: logits must be contiguous BF16");
    }
    if (logits.ne[0] < kDflash2PathSelectTopK || logits.ne[1] <= 0 || logits.ne[2] <= 0 ||
        logits.ne[3] != 1) {
        throw std::invalid_argument("dflash2_path_select: logits must be BF16 [V,T] or [V,T,B] with V>=16");
    }
    require_sequence_extent(logits.ne[1], logits.ne[2], "dflash2_path_select");
}

void require_codebook(const Tensor& codebook, std::int32_t min_rows, const char* label) {
    if (codebook.dtype != DType::BF16 || !codebook.is_contiguous() || codebook.data == nullptr) {
        throw std::invalid_argument(std::string("dflash2_path_select: ") + label +
                                    " must be contiguous BF16");
    }
    if (codebook.ne[0] != kDflash2PathSelectRank || codebook.ne[1] < min_rows ||
        codebook.ne[2] != 1 || codebook.ne[3] != 1) {
        throw std::invalid_argument(std::string("dflash2_path_select: ") + label +
                                    " must be BF16 [256, codebook_rows] with enough rows");
    }
}

void require_nvfp4_codebook(const Weight& codebook, std::int32_t min_rows, const char* label) {
    if (codebook.qtype != QType::NVFP4 || codebook.layout != QuantLayout::BlockScaleK16M128x4 ||
        codebook.k != kDflash2PathSelectRank || codebook.n < min_rows || (codebook.n % 128) != 0 ||
        codebook.qdata == nullptr || codebook.scales == nullptr ||
        !std::isfinite(codebook.weight_scale_divisor) || codebook.weight_scale_divisor <= 0.0F) {
        throw std::invalid_argument(std::string("dflash2_path_select: ") + label +
                                    " NVFP4 codebook must be [codebook_rows,256] with rows>=min");
    }
}

const Weight* resolved_nvfp4_codebook(const Weight* nvfp4, const Tensor& bf16,
                                      std::int32_t min_rows, const char* label) {
    if (nvfp4 != nullptr) {
        require_nvfp4_codebook(*nvfp4, min_rows, label);
        return nvfp4;
    }
    require_codebook(bf16, min_rows, label);
    return nullptr;
}

void require_logit_token_ids(const Tensor& token_ids, std::int32_t vocab) {
    if (token_ids.dtype != DType::I32 || !token_ids.is_contiguous() || token_ids.data == nullptr) {
        throw std::invalid_argument("dflash2_path_select: logit_token_ids must be contiguous I32");
    }
    if (token_ids.ne[0] != vocab || token_ids.ne[1] != 1 || token_ids.ne[2] != 1 ||
        token_ids.ne[3] != 1) {
        throw std::invalid_argument("dflash2_path_select: logit_token_ids must be I32 [V]");
    }
}

void require_anchors(const Tensor& anchors, std::int32_t batch) {
    if (anchors.dtype != DType::I32 || !anchors.is_contiguous() || anchors.data == nullptr) {
        throw std::invalid_argument("dflash2_path_select: anchors must be contiguous I32");
    }
    if (anchors.ne[0] != batch || anchors.ne[1] != 1 || anchors.ne[2] != 1 || anchors.ne[3] != 1) {
        throw std::invalid_argument("dflash2_path_select: anchors must be I32 [B]");
    }
}

void require_path(const Tensor& path, std::int32_t tokens, std::int32_t batch) {
    if (path.dtype != DType::I32 || !path.is_contiguous() || path.data == nullptr) {
        throw std::invalid_argument("dflash2_path_select: path must be contiguous I32");
    }
    if (path.ne[0] != tokens || path.ne[1] != batch || path.ne[2] != 1 || path.ne[3] != 1) {
        throw std::invalid_argument("dflash2_path_select: path must be I32 [T] or [T,B]");
    }
}

void require_selector(const Tensor& ids, const Tensor& q, std::int32_t tokens, std::int32_t batch) {
    if (ids.dtype != DType::I32 || !ids.is_contiguous() || ids.data == nullptr) {
        throw std::invalid_argument("dflash2_path_select: selector_ids must be contiguous I32");
    }
    if (q.dtype != DType::FP32 || !q.is_contiguous() || q.data == nullptr) {
        throw std::invalid_argument("dflash2_path_select: selector_q must be contiguous FP32");
    }
    if (ids.ne[0] != kDflash2PathSelectTopK || ids.ne[1] != tokens || ids.ne[2] != batch ||
        ids.ne[3] != 1) {
        throw std::invalid_argument(
            "dflash2_path_select: selector_ids must be I32 [16,T,1,1] or [16,T,B,1]");
    }
    if (q.ne[0] != kDflash2PathSelectTopK || q.ne[1] != tokens || q.ne[2] != batch || q.ne[3] != 1) {
        throw std::invalid_argument(
            "dflash2_path_select: selector_q must be FP32 [16,T,1,1] or [16,T,B,1]");
    }
}

void require_projection_weight(const Weight& weight) {
    if (weight.n != kDflash2PathSelectRank || weight.k != kDflash2PathSelectHidden) {
        throw std::invalid_argument(
            "dflash2_path_select: hidden_projection must be logical [256,5120]");
    }
    if (weight.qtype == QType::BF16_CTRL) {
        if (weight.layout != QuantLayout::Contiguous ||
            (weight.qdata == nullptr && weight.payload == nullptr)) {
            throw std::invalid_argument(
                "dflash2_path_select: BF16 hidden_projection must be contiguous and non-null");
        }
        return;
    }
    if (!is_quantized_projection(weight.qtype) ||
        (weight.qtype != QType::NVFP4 && weight.layout != QuantLayout::RowSplit) ||
        (weight.qtype == QType::NVFP4 && weight.layout != QuantLayout::BlockScaleK16M128x4)) {
        throw std::invalid_argument(
            "dflash2_path_select: hidden_projection must be BF16_CTRL, Q4/W8 row-split, or NVFP4");
    }
}

std::size_t hidden_proj_bytes(std::int32_t tokens, std::int32_t batch) {
    return static_cast<std::size_t>(kDflash2PathSelectRank) * static_cast<std::size_t>(tokens) *
           static_cast<std::size_t>(batch) * sizeof(std::uint16_t);
}

constexpr int kTopkSplits = 32;

std::size_t align_workspace(std::size_t bytes) {
    constexpr std::size_t kAlign = 256;
    return (bytes + kAlign - 1) & ~(kAlign - 1);
}

std::size_t topk_scratch_bytes(std::int32_t tokens, std::int32_t batch) {
    const std::size_t columns = static_cast<std::size_t>(tokens) * static_cast<std::size_t>(batch);
    const std::size_t split =
        columns * static_cast<std::size_t>(kTopkSplits) * kDflash2PathSelectTopK;
    const std::size_t merged = columns * kDflash2PathSelectTopK;
    return (split + merged) * (sizeof(float) + sizeof(std::int32_t));
}

struct TopkScratch {
    float* split_val;
    int* split_idx;
    float* cand_val;
    int* cand_idx;
};

TopkScratch alloc_topk_scratch(WorkspaceArena& workspace, std::int32_t tokens,
                               std::int32_t batch) {
    const std::size_t columns = static_cast<std::size_t>(tokens) * static_cast<std::size_t>(batch);
    const std::size_t split =
        columns * static_cast<std::size_t>(kTopkSplits) * kDflash2PathSelectTopK;
    const std::size_t merged = columns * kDflash2PathSelectTopK;
    auto* bytes = static_cast<std::byte*>(workspace.alloc_bytes(topk_scratch_bytes(tokens, batch)).data);
    TopkScratch out{};
    out.split_val = reinterpret_cast<float*>(bytes);
    bytes += split * sizeof(float);
    out.split_idx = reinterpret_cast<int*>(bytes);
    bytes += split * sizeof(std::int32_t);
    out.cand_val = reinterpret_cast<float*>(bytes);
    bytes += merged * sizeof(float);
    out.cand_idx = reinterpret_cast<int*>(bytes);
    return out;
}

} // namespace

std::size_t dflash2_path_select_workspace_capacity_bytes(QType qtype, std::int32_t min_tokens,
                                                         std::int32_t max_tokens,
                                                         std::int32_t batch) {
    if (min_tokens <= 0 || max_tokens < min_tokens || batch <= 0 ||
        batch > kDflash2PathSelectMaxBatch ||
        (batch > 1 && max_tokens > kDflash2PathSelectMaxWidthWhenBatched)) {
        throw std::invalid_argument("dflash2_path_select workspace: invalid token/batch interval");
    }
    std::size_t bytes = align_workspace(hidden_proj_bytes(max_tokens, batch)) +
                        align_workspace(topk_scratch_bytes(max_tokens, batch));
    if (qtype == QType::BF16_CTRL) { return bytes; }
    if (!is_quantized_projection(qtype)) {
        throw std::invalid_argument("dflash2_path_select workspace: unsupported projection qtype");
    }
    bytes += linear_workspace_capacity_bytes(qtype, kDflash2PathSelectRank,
                                             kDflash2PathSelectHidden, LinearPolicy::A16Only,
                                             min_tokens, max_tokens);
    return bytes;
}

void dflash2_path_select(const Tensor& logits, const Tensor& hidden,
                         const Weight& hidden_projection, const Tensor& pred_code,
                         const Tensor& succ_code, const Tensor& anchors,
                         const Tensor& logical_positions,
                         const SamplingConfig* configs, Tensor& path, WorkspaceArena& workspace,
                         cudaStream_t stream, const Tensor* logit_token_ids,
                         const Weight* pred_nvfp4, const Weight* succ_nvfp4, Tensor* selector_ids,
                         Tensor* selector_q, unsigned long long seed_xor,
                         std::int32_t position_offset, bool force_greedy) {
    require_logits(logits);
    const std::int32_t vocab  = logits.ne[0];
    const std::int32_t tokens = logits.ne[1];
    const std::int32_t batch  = logits.ne[2];
    require_hidden(hidden, tokens, batch);
    const std::int32_t min_codebook_rows =
        logit_token_ids != nullptr ? kDflash2PathSelectCodebookRows : vocab;
    const Weight* pred_q =
        resolved_nvfp4_codebook(pred_nvfp4, pred_code, min_codebook_rows, "pred_code");
    const Weight* succ_q =
        resolved_nvfp4_codebook(succ_nvfp4, succ_code, min_codebook_rows, "succ_code");
    if ((pred_q == nullptr) != (succ_q == nullptr)) {
        throw std::invalid_argument("dflash2_path_select: codebook formats must match");
    }
    if (pred_q == nullptr && pred_code.ne[1] != succ_code.ne[1]) {
        throw std::invalid_argument("dflash2_path_select: codebook row counts must match");
    }
    if (pred_q != nullptr && pred_q->n != succ_q->n) {
        throw std::invalid_argument("dflash2_path_select: codebook row counts must match");
    }
    if (logit_token_ids != nullptr) { require_logit_token_ids(*logit_token_ids, vocab); }
    if (configs == nullptr) {
        throw std::invalid_argument("dflash2_path_select: configs is required");
    }
    require_anchors(anchors, batch);
    require_anchors(logical_positions, batch);
    require_path(path, tokens, batch);
    require_projection_weight(hidden_projection);
    if ((selector_ids == nullptr) != (selector_q == nullptr)) {
        throw std::invalid_argument(
            "dflash2_path_select: selector_ids and selector_q must both be null or both set");
    }
    if (selector_ids != nullptr) { require_selector(*selector_ids, *selector_q, tokens, batch); }
    if (path.data == logits.data || path.data == hidden.data || path.data == pred_code.data ||
        path.data == succ_code.data || path.data == anchors.data ||
        path.data == logical_positions.data ||
        (logit_token_ids != nullptr && path.data == logit_token_ids->data) ||
        (selector_ids != nullptr &&
         (path.data == selector_ids->data || path.data == selector_q->data))) {
        throw std::invalid_argument("dflash2_path_select: path must not alias inputs");
    }

    auto scratch_scope           = workspace.scope();
    const DeviceSpan proj_span   = workspace.alloc_bytes(hidden_proj_bytes(tokens, batch));
    Tensor hidden_proj(proj_span.data, DType::BF16, {kDflash2PathSelectRank, tokens * batch});
    Tensor hidden_flat           = hidden.view({kDflash2PathSelectHidden, tokens * batch});
    if (hidden_projection.qtype == QType::BF16_CTRL) {
        if (!aligned_to(hidden_flat.data, 16) || !aligned_to(hidden_proj.data, 16)) {
            throw std::invalid_argument(
                "dflash2_path_select: hidden/projection scratch must be 16-byte aligned");
        }
        detail::dflash2_path_select_bf16_gemv_launch(hidden_flat, hidden_projection, hidden_proj,
                                                     stream);
    } else {
        ops::linear(hidden_flat, hidden_projection, hidden_proj, stream);
    }
    const TopkScratch topk = alloc_topk_scratch(workspace, tokens, batch);
    detail::dflash2_column_topk_launch(logits, topk.split_val, topk.split_idx, topk.cand_val,
                                       topk.cand_idx, logit_token_ids, stream);
    detail::dflash2_path_select_launch(topk.cand_val, topk.cand_idx, hidden_proj,
                                       pred_q == nullptr ? &pred_code : nullptr,
                                       succ_q == nullptr ? &succ_code : nullptr, pred_q, succ_q,
                                       anchors, logical_positions, path, tokens, batch, configs,
                                       stream, selector_ids, selector_q, seed_xor, position_offset,
                                       force_greedy);
}

void dflash2_tree_select(const Tensor& logits, const Tensor& hidden,
                         const Weight& hidden_projection, const Tensor& pred_code,
                         const Tensor& succ_code, const Tensor& anchors, const Tensor& frontiers,
                         Tensor& verify_ids, Tensor& parent_index, Tensor& cache_positions,
                         Tensor& rope_positions, Tensor& ancestor_mask, Tensor& valid_columns,
                         WorkspaceArena& workspace, cudaStream_t stream,
                         const Tensor* logit_token_ids, const Weight* pred_nvfp4,
                         const Weight* succ_nvfp4) {
    require_logits(logits);
    const std::int32_t vocab  = logits.ne[0];
    const std::int32_t tokens = logits.ne[1];
    const std::int32_t batch  = logits.ne[2];
    require_hidden(hidden, tokens, batch);
    const std::int32_t min_codebook_rows =
        logit_token_ids != nullptr ? kDflash2PathSelectCodebookRows : vocab;
    const Weight* pred_q =
        resolved_nvfp4_codebook(pred_nvfp4, pred_code, min_codebook_rows, "pred_code");
    const Weight* succ_q =
        resolved_nvfp4_codebook(succ_nvfp4, succ_code, min_codebook_rows, "succ_code");
    if ((pred_q == nullptr) != (succ_q == nullptr)) {
        throw std::invalid_argument("dflash2_tree_select: codebook formats must match");
    }
    if (pred_q == nullptr && pred_code.ne[1] != succ_code.ne[1]) {
        throw std::invalid_argument("dflash2_tree_select: codebook row counts must match");
    }
    if (pred_q != nullptr && pred_q->n != succ_q->n) {
        throw std::invalid_argument("dflash2_tree_select: codebook row counts must match");
    }
    if (logit_token_ids != nullptr) { require_logit_token_ids(*logit_token_ids, vocab); }
    require_anchors(anchors, batch);
    require_anchors(frontiers, batch);
    require_projection_weight(hidden_projection);
    const std::int32_t width = verify_ids.ne[0];
    if (width < 2 || width > kDflash2TreeExpandWidth) {
        throw std::invalid_argument("dflash2_tree_select: W must be in [2,16]");
    }
    auto require_wb = [&](const Tensor& tensor, const char* name, DType dtype) {
        if (tensor.dtype != dtype || !tensor.is_contiguous() || tensor.data == nullptr) {
            throw std::invalid_argument(std::string("dflash2_tree_select: ") + name +
                                        " must be contiguous");
        }
        if (tensor.ne[0] != width || tensor.ne[1] != batch || tensor.ne[2] != 1 ||
            tensor.ne[3] != 1) {
            throw std::invalid_argument(std::string("dflash2_tree_select: ") + name +
                                        " must be [W,B]");
        }
    };
    require_wb(verify_ids, "verify_ids", DType::I32);
    require_wb(parent_index, "parent_index", DType::I32);
    require_wb(cache_positions, "cache_positions", DType::I32);
    require_wb(rope_positions, "rope_positions", DType::I32);
    require_wb(ancestor_mask, "ancestor_mask", DType::I32);
    if (valid_columns.dtype != DType::I32 || !valid_columns.is_contiguous() ||
        valid_columns.data == nullptr || valid_columns.ne[0] != batch ||
        valid_columns.ne[1] != 1 || valid_columns.ne[2] != 1 || valid_columns.ne[3] != 1) {
        throw std::invalid_argument("dflash2_tree_select: valid_columns must be I32 [B]");
    }
    if (1 + kDflash2TreeFrontier * tokens > kDflash2TreeExpandWidth) {
        throw std::invalid_argument("dflash2_tree_select: T is too large for the packed tree");
    }

    auto scratch_scope         = workspace.scope();
    const DeviceSpan proj_span = workspace.alloc_bytes(hidden_proj_bytes(tokens, batch));
    Tensor hidden_proj(proj_span.data, DType::BF16, {kDflash2PathSelectRank, tokens * batch});
    Tensor hidden_flat = hidden.view({kDflash2PathSelectHidden, tokens * batch});
    if (hidden_projection.qtype == QType::BF16_CTRL) {
        detail::dflash2_path_select_bf16_gemv_launch(hidden_flat, hidden_projection, hidden_proj,
                                                     stream);
    } else {
        ops::linear(hidden_flat, hidden_projection, hidden_proj, stream);
    }
    const TopkScratch topk = alloc_topk_scratch(workspace, tokens, batch);
    detail::dflash2_column_topk_launch(logits, topk.split_val, topk.split_idx, topk.cand_val,
                                       topk.cand_idx, logit_token_ids, stream);
    detail::dflash2_tree_select_launch(topk.cand_val, topk.cand_idx, hidden_proj,
                                       pred_q == nullptr ? &pred_code : nullptr,
                                       succ_q == nullptr ? &succ_code : nullptr, pred_q, succ_q,
                                       anchors, frontiers, verify_ids, parent_index,
                                       cache_positions, rope_positions, ancestor_mask,
                                       valid_columns, tokens, batch, width, stream);
}

} // namespace ninfer::ops
