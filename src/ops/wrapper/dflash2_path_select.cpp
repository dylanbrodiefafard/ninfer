// ninfer::ops - dflash2_path_select wrapper: validation, projection dispatch, launcher.
#include "ninfer/ops/dflash2_path_select.h"

#include "ninfer/ops/linear.h"
#include "ops/launcher/dflash2_path_select.h"

#include <cstdint>
#include <stdexcept>
#include <string>

namespace ninfer::ops {
namespace {

bool aligned_to(const void* pointer, std::uintptr_t alignment) {
    return pointer != nullptr && (reinterpret_cast<std::uintptr_t>(pointer) & (alignment - 1)) == 0;
}

bool is_quantized_projection(QType qtype) {
    return qtype == QType::Q4G64_F16S || qtype == QType::W8G32_F16S;
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
    if (!is_quantized_projection(weight.qtype) || weight.layout != QuantLayout::RowSplit) {
        throw std::invalid_argument(
            "dflash2_path_select: hidden_projection must be BF16_CTRL or Q4/W8 row-split");
    }
}

std::size_t hidden_proj_bytes(std::int32_t tokens, std::int32_t batch) {
    return static_cast<std::size_t>(kDflash2PathSelectRank) * static_cast<std::size_t>(tokens) *
           static_cast<std::size_t>(batch) * sizeof(std::uint16_t);
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
    std::size_t bytes = hidden_proj_bytes(max_tokens, batch);
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
                         const Tensor& succ_code, const Tensor& anchors, float temperature,
                         unsigned long long seed, Tensor& path, WorkspaceArena& workspace,
                         cudaStream_t stream, const Tensor* logit_token_ids) {
    require_logits(logits);
    const std::int32_t vocab  = logits.ne[0];
    const std::int32_t tokens = logits.ne[1];
    const std::int32_t batch  = logits.ne[2];
    require_hidden(hidden, tokens, batch);
    const std::int32_t min_codebook_rows =
        logit_token_ids != nullptr ? kDflash2PathSelectCodebookRows : vocab;
    require_codebook(pred_code, min_codebook_rows, "pred_code");
    require_codebook(succ_code, min_codebook_rows, "succ_code");
    if (pred_code.ne[1] != succ_code.ne[1]) {
        throw std::invalid_argument("dflash2_path_select: codebook row counts must match");
    }
    if (logit_token_ids != nullptr) { require_logit_token_ids(*logit_token_ids, vocab); }
    require_anchors(anchors, batch);
    require_path(path, tokens, batch);
    require_projection_weight(hidden_projection);
    if (path.data == logits.data || path.data == hidden.data || path.data == pred_code.data ||
        path.data == succ_code.data || path.data == anchors.data ||
        (logit_token_ids != nullptr && path.data == logit_token_ids->data)) {
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
    detail::dflash2_path_select_launch(logits, hidden_proj, pred_code, succ_code, anchors, path,
                                       temperature, seed, stream, logit_token_ids);
}

} // namespace ninfer::ops
