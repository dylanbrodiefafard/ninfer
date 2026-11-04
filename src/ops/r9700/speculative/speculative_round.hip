// Implements: include/ninfer/ops/speculative_round.h
// Match: validated speculative state and BF16 verification logits.
// Algorithm assumptions: the per-row kernel always launches for the small route;
// the multiblock partial/group/mass pipeline owns every large mode. Launchers do
// not read device SamplingConfig.
#include "ninfer/ops/speculative_round.h"

#include "ops/common/math.h"
#include "ops/r9700/sampling/sampling_workspace.h"
#include "ops/r9700/speculative/speculative_kernels.h"
#include "core/device.h"

#include <algorithm>
#include <cstdint>
#include <stdexcept>

namespace ninfer::ops::detail {

void speculative_prepare_verify_inputs_launch(const Tensor& anchors, const Tensor& drafts,
                                              const Tensor& base_positions,
                                              const Tensor& current_extents, Tensor& verify_ids,
                                              Tensor& positions, hipStream_t stream) {
    constexpr int kBlock = 32;
    const int k          = drafts.ne[0];
    const int batch      = drafts.ne[1];
    const dim3 grid(static_cast<unsigned int>(div_up(k + 1, kBlock)),
                    static_cast<unsigned int>(batch));
    speculative_prepare_verify_inputs_kernel<<<grid, kBlock, 0, stream>>>(
        static_cast<const std::int32_t*>(anchors.data),
        static_cast<const std::int32_t*>(drafts.data),
        static_cast<const std::int32_t*>(base_positions.data),
        static_cast<const std::int32_t*>(current_extents.data),
        static_cast<std::int32_t*>(verify_ids.data), static_cast<std::int32_t*>(positions.data), k);
    HIP_CHECK(hipGetLastError());
}

void speculative_prepare_verify_ids_launch(const Tensor& anchors, const Tensor& drafts,
                                           const Tensor& current_extents, Tensor& verify_ids,
                                           hipStream_t stream) {
    constexpr int kBlock = 32;
    const int k          = drafts.ne[0];
    const int batch      = drafts.ne[1];
    const dim3 grid(static_cast<unsigned int>(div_up(k + 1, kBlock)),
                    static_cast<unsigned int>(batch));
    speculative_prepare_verify_inputs_kernel<<<grid, kBlock, 0, stream>>>(
        static_cast<const std::int32_t*>(anchors.data),
        static_cast<const std::int32_t*>(drafts.data), nullptr,
        static_cast<const std::int32_t*>(current_extents.data),
        static_cast<std::int32_t*>(verify_ids.data), nullptr, k);
    HIP_CHECK(hipGetLastError());
}

void speculative_accept_greedy_drafts_launch(
    const Tensor& target_tokens, const Tensor& logits, const Tensor& drafts,
    const Tensor& current_extents, Tensor& lengths, Tensor& anchors, Tensor& licensed_tokens,
    Tensor& licensed_counts, Tensor& accepted, std::int32_t token_domain,
    const SamplingConfig* configs, DeviceSpan workspace, hipStream_t stream,
    const Tensor* selector_ids, const Tensor* selector_q, Tensor* proposal_calibration) {
    const std::int32_t physical_rows = logits.ne[0];
    const std::int32_t cols          = drafts.ne[0] + 1;
    const std::int32_t batch         = drafts.ne[1];
    const std::int32_t selector_k    = selector_ids != nullptr ? selector_ids->ne[0] : 0;
    const std::int32_t* selector_id_ptr =
        selector_ids != nullptr ? static_cast<const std::int32_t*>(selector_ids->data) : nullptr;
    const float* selector_q_ptr =
        selector_q != nullptr ? static_cast<const float*>(selector_q->data) : nullptr;
    const SamplingWorkspaceLayout layout = make_sampling_workspace_layout(token_domain, cols);
    if (!layout.multiblock) {
        if (proposal_calibration != nullptr) {
            throw std::invalid_argument("speculative_accept_greedy_drafts: proposal calibration "
                                        "requires the multi-block sampler");
        }
        speculative_accept_greedy_drafts_kernel<<<batch, kSamplerBlock, 0, stream>>>(
            static_cast<const std::int32_t*>(target_tokens.data),
            static_cast<const hip_bfloat16*>(logits.data),
            static_cast<const std::int32_t*>(drafts.data),
            static_cast<const std::int32_t*>(current_extents.data),
            static_cast<std::int32_t*>(lengths.data), static_cast<std::int32_t*>(anchors.data),
            static_cast<std::int32_t*>(licensed_tokens.data),
            static_cast<std::int32_t*>(licensed_counts.data),
            static_cast<std::int32_t*>(accepted.data), configs, token_domain, physical_rows,
            drafts.ne[0], selector_id_ptr, selector_q_ptr, selector_k);
        HIP_CHECK(hipGetLastError());
        return;
    }
    const std::int32_t partial_blocks = div_up(token_domain, kSamplerPartialTileItems);
    const std::int32_t groups         = sampler_group_count(partial_blocks);
    const SamplingWorkspace scratch   = layout.bind(workspace);
    const dim3 partial_grid(static_cast<unsigned int>(partial_blocks),
                            static_cast<unsigned int>(cols), static_cast<unsigned int>(batch));
    speculative_sampling_partial_topk_kernel<<<partial_grid, kSamplerBlock, 0, stream>>>(
        static_cast<const hip_bfloat16*>(logits.data),
        static_cast<const std::int32_t*>(drafts.data),
        static_cast<const std::int32_t*>(current_extents.data), configs, token_domain,
        physical_rows, cols, drafts.ne[0], scratch, layout.bytes, nullptr, nullptr, nullptr);
    HIP_CHECK(hipGetLastError());
    const dim3 batched_group_grid(static_cast<unsigned int>(groups),
                                  static_cast<unsigned int>(cols),
                                  static_cast<unsigned int>(batch));
    speculative_sampling_group_finalize_kernel<<<batched_group_grid, kSamplerGroupBlock, 0,
                                                 stream>>>(
        static_cast<const std::int32_t*>(target_tokens.data),
        static_cast<const std::int32_t*>(drafts.data),
        static_cast<const std::int32_t*>(current_extents.data),
        static_cast<std::int32_t*>(lengths.data), static_cast<std::int32_t*>(anchors.data),
        static_cast<std::int32_t*>(licensed_tokens.data),
        static_cast<std::int32_t*>(licensed_counts.data), static_cast<std::int32_t*>(accepted.data),
        configs, token_domain, cols, partial_blocks, groups, scratch, layout.bytes, selector_id_ptr,
        selector_q_ptr, selector_k, nullptr, nullptr, nullptr, nullptr, nullptr);
    HIP_CHECK(hipGetLastError());
    const dim3 p_less_grid(
        static_cast<unsigned int>(sampler_p_less_workers_per_column(partial_blocks, cols)),
        static_cast<unsigned int>(cols), static_cast<unsigned int>(batch));
    speculative_sampling_p_less_mass_finalize_kernel<<<p_less_grid, kSamplerBlock, 0, stream>>>(
        static_cast<const hip_bfloat16*>(logits.data),
        static_cast<const std::int32_t*>(drafts.data),
        static_cast<const std::int32_t*>(current_extents.data),
        static_cast<std::int32_t*>(lengths.data), static_cast<std::int32_t*>(anchors.data),
        static_cast<std::int32_t*>(licensed_tokens.data),
        static_cast<std::int32_t*>(licensed_counts.data), static_cast<std::int32_t*>(accepted.data),
        configs, token_domain, physical_rows, cols, partial_blocks, scratch, layout.bytes,
        selector_id_ptr, selector_q_ptr, selector_k, nullptr, nullptr, nullptr, nullptr, nullptr);
    HIP_CHECK(hipGetLastError());
    if (proposal_calibration != nullptr) {
        PLessProposalCalibrationGrid grid{};
        for (int g = 0; g < kPLessProposalCalibrationTemperatureCount; ++g) {
            grid.temperature[g] =
                kPLessProposalCalibrationTemperatures[static_cast<std::size_t>(g)];
        }
        speculative_p_less_proposal_calibration_kernel<<<dim3(static_cast<unsigned int>(cols - 1),
                                                              static_cast<unsigned int>(batch)),
                                                         32, 0, stream>>>(
            static_cast<const hip_bfloat16*>(logits.data),
            static_cast<const std::int32_t*>(current_extents.data), configs, token_domain,
            physical_rows, cols, scratch, layout.bytes, selector_id_ptr, selector_q_ptr, selector_k,
            grid, static_cast<float*>(proposal_calibration->data));
        HIP_CHECK(hipGetLastError());
    }
}

void speculative_select_accepted_hidden_launch(const Tensor& hidden, const Tensor& selectors,
                                               Tensor& out, hipStream_t stream) {
    constexpr int kBlock = 256;
    const int rows       = hidden.ne[0];
    const int batch      = hidden.ne[2];
    const dim3 grid(static_cast<unsigned int>(std::max(1, div_up(rows, kBlock))),
                    static_cast<unsigned int>(batch));
    speculative_select_accepted_hidden_kernel<<<grid, kBlock, 0, stream>>>(
        static_cast<const hip_bfloat16*>(hidden.data),
        static_cast<const std::int32_t*>(selectors.data), static_cast<hip_bfloat16*>(out.data),
        rows, hidden.ne[1]);
    HIP_CHECK(hipGetLastError());
}

void proposal_remap_token_ids_launch(Tensor& proposal_tokens, const std::int32_t* id_map,
                                     std::int32_t n, hipStream_t stream) {
    constexpr int kBlock = 256;
    const int count      = proposal_tokens.ne[0];
    const int grid       = std::max(1, div_up(count, kBlock));
    proposal_remap_token_ids_kernel<<<grid, kBlock, 0, stream>>>(
        static_cast<std::int32_t*>(proposal_tokens.data), count, id_map, n);
    HIP_CHECK(hipGetLastError());
}

void speculative_accept_tree_drafts_launch(
    const Tensor& target_tokens, const Tensor& logits, const Tensor& verify_ids,
    const Tensor& parent_index, const Tensor& valid_columns, const Tensor& current_extents,
    Tensor& lengths, Tensor& anchors, Tensor& licensed_tokens, Tensor& licensed_counts,
    Tensor& accepted, Tensor& accepted_column, Tensor& fold_path, std::int32_t token_domain,
    const SamplingConfig* configs, DeviceSpan workspace, hipStream_t stream) {
    const std::int32_t batch             = verify_ids.ne[1];
    const std::int32_t width             = verify_ids.ne[0];
    const std::int32_t physical_rows     = logits.ne[0];
    const SamplingWorkspaceLayout layout = make_sampling_workspace_layout(token_domain, width);
    if (!layout.multiblock) {
        speculative_accept_tree_drafts_kernel<<<static_cast<unsigned int>(batch), kSamplerBlock, 0,
                                                stream>>>(
            static_cast<const std::int32_t*>(target_tokens.data),
            static_cast<const hip_bfloat16*>(logits.data),
            static_cast<const std::int32_t*>(verify_ids.data),
            static_cast<const std::int32_t*>(parent_index.data),
            static_cast<const std::int32_t*>(valid_columns.data),
            static_cast<const std::int32_t*>(current_extents.data),
            static_cast<std::int32_t*>(lengths.data), static_cast<std::int32_t*>(anchors.data),
            static_cast<std::int32_t*>(licensed_tokens.data),
            static_cast<std::int32_t*>(licensed_counts.data),
            static_cast<std::int32_t*>(accepted.data),
            static_cast<std::int32_t*>(accepted_column.data),
            static_cast<std::int32_t*>(fold_path.data), configs, token_domain, physical_rows,
            width);
        HIP_CHECK(hipGetLastError());
        return;
    }
    const std::int32_t partial_blocks = div_up(token_domain, kSamplerPartialTileItems);
    const std::int32_t groups         = sampler_group_count(partial_blocks);
    const SamplingWorkspace scratch   = layout.bind(workspace);
    const dim3 partial_grid(static_cast<unsigned int>(partial_blocks),
                            static_cast<unsigned int>(width), static_cast<unsigned int>(batch));
    speculative_sampling_partial_topk_kernel<<<partial_grid, kSamplerBlock, 0, stream>>>(
        static_cast<const hip_bfloat16*>(logits.data), nullptr,
        static_cast<const std::int32_t*>(current_extents.data), configs, token_domain,
        physical_rows, width, width - 1, scratch, layout.bytes,
        static_cast<const std::int32_t*>(verify_ids.data),
        static_cast<const std::int32_t*>(parent_index.data),
        static_cast<const std::int32_t*>(valid_columns.data));
    HIP_CHECK(hipGetLastError());
    const dim3 batched_group_grid(static_cast<unsigned int>(groups),
                                  static_cast<unsigned int>(width),
                                  static_cast<unsigned int>(batch));
    speculative_sampling_group_finalize_kernel<<<batched_group_grid, kSamplerGroupBlock, 0,
                                                 stream>>>(
        static_cast<const std::int32_t*>(target_tokens.data), nullptr,
        static_cast<const std::int32_t*>(current_extents.data),
        static_cast<std::int32_t*>(lengths.data), static_cast<std::int32_t*>(anchors.data),
        static_cast<std::int32_t*>(licensed_tokens.data),
        static_cast<std::int32_t*>(licensed_counts.data), static_cast<std::int32_t*>(accepted.data),
        configs, token_domain, width, partial_blocks, groups, scratch, layout.bytes, nullptr,
        nullptr, 0, static_cast<const std::int32_t*>(verify_ids.data),
        static_cast<const std::int32_t*>(parent_index.data),
        static_cast<const std::int32_t*>(valid_columns.data),
        static_cast<std::int32_t*>(accepted_column.data),
        static_cast<std::int32_t*>(fold_path.data));
    HIP_CHECK(hipGetLastError());
    const dim3 p_less_grid(
        static_cast<unsigned int>(sampler_p_less_workers_per_column(partial_blocks, width)),
        static_cast<unsigned int>(width), static_cast<unsigned int>(batch));
    speculative_sampling_p_less_mass_finalize_kernel<<<p_less_grid, kSamplerBlock, 0, stream>>>(
        static_cast<const hip_bfloat16*>(logits.data), nullptr,
        static_cast<const std::int32_t*>(current_extents.data),
        static_cast<std::int32_t*>(lengths.data), static_cast<std::int32_t*>(anchors.data),
        static_cast<std::int32_t*>(licensed_tokens.data),
        static_cast<std::int32_t*>(licensed_counts.data), static_cast<std::int32_t*>(accepted.data),
        configs, token_domain, physical_rows, width, partial_blocks, scratch, layout.bytes, nullptr,
        nullptr, 0, static_cast<const std::int32_t*>(verify_ids.data),
        static_cast<const std::int32_t*>(parent_index.data),
        static_cast<const std::int32_t*>(valid_columns.data),
        static_cast<std::int32_t*>(accepted_column.data),
        static_cast<std::int32_t*>(fold_path.data));
    HIP_CHECK(hipGetLastError());
}

} // namespace ninfer::ops::detail

#include "ninfer/types.h"
#include <limits>
#include <string>

namespace ninfer::ops {
namespace {
constexpr int kMaximumColumns = 16;
constexpr int kMaximumBatch   = static_cast<int>(kMaximumConcurrency);

struct ByteRange {
    std::uintptr_t begin = 0;
    std::uintptr_t end   = 0;
};

[[nodiscard]] ByteRange tensor_range(const Tensor& tensor) {
    const auto begin        = reinterpret_cast<std::uintptr_t>(tensor.data);
    const std::size_t bytes = tensor.bytes();
    if (begin > std::numeric_limits<std::uintptr_t>::max() - bytes) {
        throw std::invalid_argument("speculative_round: tensor storage range overflows");
    }
    return {begin, begin + bytes};
}

[[nodiscard]] bool overlaps(ByteRange lhs, ByteRange rhs) noexcept {
    return lhs.begin < rhs.end && rhs.begin < lhs.end;
}

[[nodiscard]] ByteRange pointer_range(const void* pointer, std::size_t bytes, const char* label) {
    const auto begin = reinterpret_cast<std::uintptr_t>(pointer);
    if (begin > std::numeric_limits<std::uintptr_t>::max() - bytes) {
        throw std::invalid_argument(std::string(label) + " storage range overflows");
    }
    return {begin, begin + bytes};
}

void require_disjoint(const Tensor* const* tensors, std::size_t count, const char* op) {
    for (std::size_t lhs = 0; lhs < count; ++lhs) {
        for (std::size_t rhs = lhs + 1; rhs < count; ++rhs) {
            if (overlaps(tensor_range(*tensors[lhs]), tensor_range(*tensors[rhs]))) {
                throw std::invalid_argument(std::string(op) + ": tensor operands overlap");
            }
        }
    }
}

void require_contiguous_nonnull(const Tensor& tensor, const char* op, const char* name) {
    if (!tensor.is_contiguous() || tensor.data == nullptr) {
        throw std::invalid_argument(std::string(op) + ": " + name +
                                    " must be contiguous and non-null");
    }
}

void require_matrix(const Tensor& tensor, DType dtype, int rows, int columns, const char* op,
                    const char* name) {
    if (tensor.dtype != dtype || rows <= 0 || columns <= 0 || tensor.ne[0] != rows ||
        tensor.ne[1] != columns || tensor.ne[2] != 1 || tensor.ne[3] != 1) {
        throw std::invalid_argument(std::string(op) + ": invalid shape or dtype for " + name);
    }
    require_contiguous_nonnull(tensor, op, name);
}

void require_vector(const Tensor& tensor, DType dtype, int count, const char* op,
                    const char* name) {
    require_matrix(tensor, dtype, count, 1, op, name);
}

void require_stream(hipStream_t stream, const char* op) {
    if (stream == nullptr) {
        throw std::invalid_argument(std::string(op) + ": stream must be non-null");
    }
}

std::size_t checked_product(std::size_t a, std::size_t b, const char* label) {
    if (a && b > std::numeric_limits<std::size_t>::max() / a) throw std::overflow_error(label);
    return a * b;
}

} // namespace

std::size_t speculative_accept_greedy_drafts_workspace_capacity_bytes(std::int32_t token_domain,
                                                                      std::int32_t min_drafts,
                                                                      std::int32_t max_drafts,
                                                                      std::int32_t min_batch,
                                                                      std::int32_t max_batch) {
    if (token_domain <= 0 || min_drafts < 1 || max_drafts < min_drafts ||
        max_drafts >= kMaximumColumns || min_batch < 1 || max_batch < min_batch ||
        max_batch > kMaximumBatch) {
        throw std::invalid_argument(
            "speculative accept workspace: require K in [1,15] and B in [1,16]");
    }
    return make_sampling_workspace_layout(token_domain, max_drafts + 1).bytes *
           static_cast<std::size_t>(max_batch);
}

std::size_t speculative_accept_tree_drafts_workspace_capacity_bytes(std::int32_t token_domain,
                                                                    std::int32_t min_width,
                                                                    std::int32_t max_width,
                                                                    std::int32_t min_batch,
                                                                    std::int32_t max_batch) {
    if (token_domain <= 0 || min_width < 2 || max_width < min_width ||
        max_width > kMaximumColumns || min_batch < 1 || max_batch < min_batch ||
        max_batch > kMaximumBatch) {
        throw std::invalid_argument(
            "speculative tree workspace: require W in [2,16] and B in [1,16]");
    }
    return make_sampling_workspace_layout(token_domain, max_width).bytes *
           static_cast<std::size_t>(max_batch);
}

void speculative_prepare_verify_inputs(const Tensor& anchors, const Tensor& drafts,
                                       const Tensor& base_positions, const Tensor& current_extents,
                                       Tensor& verify_ids, Tensor& positions, hipStream_t stream) {
    constexpr const char* op = "speculative_prepare_verify_inputs";
    require_stream(stream, op);
    const int drafts_per_row = drafts.ne[0];
    const int batch          = drafts.ne[1];
    if (drafts_per_row < 1 || drafts_per_row >= kMaximumColumns || batch < 1 ||
        batch > kMaximumBatch) {
        throw std::invalid_argument(
            "speculative_prepare_verify_inputs: require K in [1,15] and B in [1,16]");
    }
    require_vector(anchors, DType::I32, batch, op, "anchors");
    require_matrix(drafts, DType::I32, drafts_per_row, batch, op, "drafts");
    require_vector(base_positions, DType::I32, batch, op, "base_positions");
    require_vector(current_extents, DType::I32, batch, op, "current_extents");
    require_matrix(verify_ids, DType::I32, drafts_per_row + 1, batch, op, "verify_ids");
    require_matrix(positions, DType::I32, drafts_per_row + 1, batch, op, "positions");
    const Tensor* tensors[] = {&anchors,         &drafts,     &base_positions,
                               &current_extents, &verify_ids, &positions};
    require_disjoint(tensors, std::size(tensors), op);
    detail::speculative_prepare_verify_inputs_launch(
        anchors, drafts, base_positions, current_extents, verify_ids, positions, stream);
}

void speculative_prepare_verify_ids(const Tensor& anchors, const Tensor& drafts,
                                    const Tensor& current_extents, Tensor& verify_ids,
                                    hipStream_t stream) {
    constexpr const char* op = "speculative_prepare_verify_ids";
    require_stream(stream, op);
    const int drafts_per_row = drafts.ne[0];
    const int batch          = drafts.ne[1];
    if (drafts_per_row < 1 || drafts_per_row >= kMaximumColumns || batch < 1 ||
        batch > kMaximumBatch) {
        throw std::invalid_argument(
            "speculative_prepare_verify_ids: require K in [1,15] and B in [1,16]");
    }
    require_vector(anchors, DType::I32, batch, op, "anchors");
    require_matrix(drafts, DType::I32, drafts_per_row, batch, op, "drafts");
    require_vector(current_extents, DType::I32, batch, op, "current_extents");
    require_matrix(verify_ids, DType::I32, drafts_per_row + 1, batch, op, "verify_ids");
    const Tensor* tensors[] = {&anchors, &drafts, &current_extents, &verify_ids};
    require_disjoint(tensors, std::size(tensors), op);
    detail::speculative_prepare_verify_ids_launch(anchors, drafts, current_extents, verify_ids,
                                                  stream);
}

void speculative_select_accepted_hidden(const Tensor& hidden, const Tensor& selectors, Tensor& out,
                                        hipStream_t stream) {
    constexpr const char* op = "speculative_select_accepted_hidden";
    require_stream(stream, op);
    if (hidden.dtype != DType::BF16 || hidden.ne[0] <= 0 || hidden.ne[1] <= 0 ||
        hidden.ne[1] > kMaximumColumns || hidden.ne[2] <= 0 || hidden.ne[2] > kMaximumBatch ||
        hidden.ne[3] != 1) {
        throw std::invalid_argument(
            "speculative_select_accepted_hidden: hidden must be BF16 [D,T,B]");
    }
    require_contiguous_nonnull(hidden, op, "hidden");
    require_vector(selectors, DType::I32, hidden.ne[2], op, "selectors");
    require_matrix(out, DType::BF16, hidden.ne[0], hidden.ne[2], op, "out");
    const Tensor* tensors[] = {&hidden, &selectors, &out};
    require_disjoint(tensors, std::size(tensors), op);
    detail::speculative_select_accepted_hidden_launch(hidden, selectors, out, stream);
}

void proposal_remap_token_ids(Tensor& proposal_tokens, const std::int32_t* id_map,
                              std::int32_t count, hipStream_t stream) {
    constexpr const char* op = "proposal_remap_token_ids";
    require_stream(stream, op);
    if (proposal_tokens.dtype != DType::I32 || proposal_tokens.ne[0] <= 0 ||
        proposal_tokens.ne[1] != 1 || proposal_tokens.ne[2] != 1 || proposal_tokens.ne[3] != 1) {
        throw std::invalid_argument(
            "proposal_remap_token_ids: proposal_tokens must be non-empty I32 [T]");
    }
    require_contiguous_nonnull(proposal_tokens, op, "proposal_tokens");
    if (id_map == nullptr || count <= 0) {
        throw std::invalid_argument(
            "proposal_remap_token_ids: id_map must be non-null and count>0");
    }
    const ByteRange tokens = tensor_range(proposal_tokens);
    const std::size_t map_bytes =
        checked_product(static_cast<std::size_t>(count), sizeof(int), "proposal id-map bytes");
    if (overlaps(tokens, pointer_range(id_map, map_bytes, op))) {
        throw std::invalid_argument("proposal_remap_token_ids: token and id-map storage overlap");
    }
    detail::proposal_remap_token_ids_launch(proposal_tokens, id_map, count, stream);
}

void speculative_accept_greedy_drafts(const Tensor& target_tokens, const Tensor& logits,
                                      const Tensor& drafts, const Tensor& current_extents,
                                      Tensor& lengths, Tensor& anchors, Tensor& licensed_tokens,
                                      Tensor& licensed_counts, Tensor& accepted,
                                      std::int32_t token_domain, const SamplingConfig* configs,
                                      WorkspaceArena& workspace, hipStream_t stream,
                                      const Tensor* selector_ids, const Tensor* selector_q,
                                      Tensor* proposal_calibration) {
    constexpr const char* op = "speculative_accept_greedy_drafts";
    require_stream(stream, op);
    const int drafts_per_row = drafts.ne[0];
    const int columns        = drafts_per_row + 1;
    const int batch          = drafts.ne[1];
    if (drafts_per_row < 1 || columns > kMaximumColumns || batch < 1 || batch > kMaximumBatch) {
        throw std::invalid_argument(
            "speculative_accept_greedy_drafts: require K in [1,15] and B in [1,16]");
    }
    require_matrix(target_tokens, DType::I32, columns, batch, op, "target_tokens");
    require_matrix(drafts, DType::I32, drafts_per_row, batch, op, "drafts");
    require_vector(current_extents, DType::I32, batch, op, "current_extents");
    require_vector(lengths, DType::I32, batch, op, "lengths");
    require_vector(anchors, DType::I32, batch, op, "anchors");
    require_matrix(licensed_tokens, DType::I32, columns, batch, op, "licensed_tokens");
    require_vector(licensed_counts, DType::I32, batch, op, "licensed_counts");
    require_vector(accepted, DType::I32, batch, op, "accepted");
    if (logits.dtype != DType::BF16 || logits.ne[0] <= 0 || logits.ne[1] != columns ||
        logits.ne[2] != batch || logits.ne[3] != 1) {
        throw std::invalid_argument(
            "speculative_accept_greedy_drafts: logits must be BF16 [physical_rows,K+1,B]");
    }
    require_contiguous_nonnull(logits, op, "logits");
    if (token_domain <= 0 || token_domain > logits.ne[0]) {
        throw std::invalid_argument(
            "speculative_accept_greedy_drafts: token_domain is outside physical logits");
    }
    if (configs == nullptr) {
        throw std::invalid_argument("speculative_accept_greedy_drafts: configs must be non-null");
    }
    if ((selector_ids == nullptr) != (selector_q == nullptr)) {
        throw std::invalid_argument(
            "speculative_accept_greedy_drafts: selector ids/q must be paired");
    }
    int selector_count = 0;
    if (selector_ids != nullptr) {
        selector_count = selector_ids->ne[0];
        if (selector_ids->dtype != DType::I32 || selector_q->dtype != DType::FP32 ||
            selector_count < 1 || selector_q->ne[0] != selector_count ||
            selector_ids->ne[1] != drafts_per_row || selector_q->ne[1] != drafts_per_row ||
            selector_ids->ne[2] != batch || selector_q->ne[2] != batch ||
            selector_ids->ne[3] != 1 || selector_q->ne[3] != 1) {
            throw std::invalid_argument(
                "speculative_accept_greedy_drafts: selector ids/q must be matching [C,K,B]");
        }
        require_contiguous_nonnull(*selector_ids, op, "selector_ids");
        require_contiguous_nonnull(*selector_q, op, "selector_q");
    }
    const Tensor* base_tensors[] = {&target_tokens,   &logits,          &drafts,
                                    &current_extents, &lengths,         &anchors,
                                    &licensed_tokens, &licensed_counts, &accepted};
    require_disjoint(base_tensors, std::size(base_tensors), op);
    const ByteRange config_range =
        pointer_range(configs,
                      checked_product(static_cast<std::size_t>(batch), sizeof(SamplingConfig),
                                      "speculative chain config bytes"),
                      op);
    for (const Tensor* tensor : base_tensors) {
        if (overlaps(tensor_range(*tensor), config_range)) {
            throw std::invalid_argument(
                "speculative_accept_greedy_drafts: configs overlap tensor storage");
        }
    }
    if (selector_ids != nullptr) {
        for (const Tensor* tensor : base_tensors) {
            if (overlaps(tensor_range(*tensor), tensor_range(*selector_ids)) ||
                overlaps(tensor_range(*tensor), tensor_range(*selector_q)) ||
                overlaps(tensor_range(*selector_ids), tensor_range(*selector_q))) {
                throw std::invalid_argument(
                    "speculative_accept_greedy_drafts: selector operands overlap");
            }
        }
        if (overlaps(tensor_range(*selector_ids), config_range) ||
            overlaps(tensor_range(*selector_q), config_range)) {
            throw std::invalid_argument(
                "speculative_accept_greedy_drafts: configs overlap selector storage");
        }
    }

    if (proposal_calibration != nullptr) {
        if (selector_ids == nullptr || selector_count > 32) {
            throw std::invalid_argument("speculative_accept_greedy_drafts: proposal calibration "
                                        "requires selectors with C<=32");
        }
        if (proposal_calibration->dtype != DType::FP32 ||
            proposal_calibration->ne[0] != kPLessProposalCalibrationTemperatureCount ||
            proposal_calibration->ne[1] != drafts_per_row || proposal_calibration->ne[2] != batch ||
            proposal_calibration->ne[3] != 1) {
            throw std::invalid_argument(
                "speculative_accept_greedy_drafts: proposal_calibration must be FP32 [G,K,B]");
        }
        require_contiguous_nonnull(*proposal_calibration, op, "proposal_calibration");
        const ByteRange calibration_range = tensor_range(*proposal_calibration);
        for (const Tensor* tensor : base_tensors) {
            if (overlaps(tensor_range(*tensor), calibration_range)) {
                throw std::invalid_argument(
                    "speculative_accept_greedy_drafts: proposal_calibration overlaps an operand");
            }
        }
        if (overlaps(tensor_range(*selector_ids), calibration_range) ||
            overlaps(tensor_range(*selector_q), calibration_range) ||
            overlaps(config_range, calibration_range)) {
            throw std::invalid_argument(
                "speculative_accept_greedy_drafts: proposal_calibration overlaps an operand");
        }
    }

    auto scope               = workspace.scope();
    const auto bytes         = make_sampling_workspace_layout(token_domain, columns).bytes *
                               static_cast<std::size_t>(batch);
    const DeviceSpan scratch = bytes == 0 ? DeviceSpan{} : workspace.alloc_bytes(bytes);
    detail::speculative_accept_greedy_drafts_launch(
        target_tokens, logits, drafts, current_extents, lengths, anchors, licensed_tokens,
        licensed_counts, accepted, token_domain, configs, scratch, stream, selector_ids, selector_q,
        proposal_calibration);
}

void speculative_accept_tree_drafts(const Tensor& target_tokens, const Tensor& logits,
                                    const Tensor& verify_ids, const Tensor& parent_index,
                                    const Tensor& valid_columns, const Tensor& current_extents,
                                    Tensor& lengths, Tensor& anchors, Tensor& licensed_tokens,
                                    Tensor& licensed_counts, Tensor& accepted,
                                    Tensor& accepted_column, Tensor& fold_path,
                                    std::int32_t token_domain, const SamplingConfig* configs,
                                    WorkspaceArena& workspace, hipStream_t stream) {
    constexpr const char* op = "speculative_accept_tree_drafts";
    require_stream(stream, op);
    const int width = verify_ids.ne[0];
    const int batch = verify_ids.ne[1];
    if (width < 2 || width > kMaximumColumns || batch < 1 || batch > kMaximumBatch) {
        throw std::invalid_argument(
            "speculative_accept_tree_drafts: require W in [2,16] and B in [1,16]");
    }
    require_matrix(target_tokens, DType::I32, width, batch, op, "target_tokens");
    require_matrix(verify_ids, DType::I32, width, batch, op, "verify_ids");
    require_matrix(parent_index, DType::I32, width, batch, op, "parent_index");
    require_vector(valid_columns, DType::I32, batch, op, "valid_columns");
    require_vector(current_extents, DType::I32, batch, op, "current_extents");
    require_vector(lengths, DType::I32, batch, op, "lengths");
    require_vector(anchors, DType::I32, batch, op, "anchors");
    require_matrix(licensed_tokens, DType::I32, width, batch, op, "licensed_tokens");
    require_vector(licensed_counts, DType::I32, batch, op, "licensed_counts");
    require_vector(accepted, DType::I32, batch, op, "accepted");
    require_vector(accepted_column, DType::I32, batch, op, "accepted_column");
    require_matrix(fold_path, DType::I32, width, batch, op, "fold_path");
    if (logits.dtype != DType::BF16 || logits.ne[0] <= 0 || logits.ne[1] != width ||
        logits.ne[2] != batch || logits.ne[3] != 1) {
        throw std::invalid_argument(
            "speculative_accept_tree_drafts: logits must be BF16 [physical_rows,W,B]");
    }
    require_contiguous_nonnull(logits, op, "logits");
    if (token_domain <= 0 || token_domain > logits.ne[0]) {
        throw std::invalid_argument(
            "speculative_accept_tree_drafts: token_domain is outside physical logits");
    }
    if (configs == nullptr) {
        throw std::invalid_argument("speculative_accept_tree_drafts: configs must be non-null");
    }
    const Tensor* tensors[] = {&target_tokens,   &logits,          &verify_ids, &parent_index,
                               &valid_columns,   &current_extents, &lengths,    &anchors,
                               &licensed_tokens, &licensed_counts, &accepted,   &accepted_column,
                               &fold_path};
    require_disjoint(tensors, std::size(tensors), op);
    const ByteRange config_range =
        pointer_range(configs,
                      checked_product(static_cast<std::size_t>(batch), sizeof(SamplingConfig),
                                      "speculative tree config bytes"),
                      op);
    for (const Tensor* tensor : tensors) {
        if (overlaps(tensor_range(*tensor), config_range)) {
            throw std::invalid_argument(
                "speculative_accept_tree_drafts: configs overlap tensor storage");
        }
    }

    auto scope = workspace.scope();
    const auto bytes =
        make_sampling_workspace_layout(token_domain, width).bytes * static_cast<std::size_t>(batch);
    const DeviceSpan scratch = bytes == 0 ? DeviceSpan{} : workspace.alloc_bytes(bytes);
    detail::speculative_accept_tree_drafts_launch(
        target_tokens, logits, verify_ids, parent_index, valid_columns, current_extents, lengths,
        anchors, licensed_tokens, licensed_counts, accepted, accepted_column, fold_path,
        token_domain, configs, scratch, stream);
}
} // namespace ninfer::ops
