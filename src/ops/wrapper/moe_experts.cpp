// ninfer::ops - moe_experts wrapper: implements the public api, validates parameters, carves the
// workspace, and dispatches to the launcher. Host-compiled; never includes the kernel code.
#include "ninfer/ops/moe_experts.h"

#include "ninfer/ops/moe_helper_contract.h"

#include "ops/launcher/moe_experts.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <stdexcept>
#include <string>

namespace ninfer::ops {

// The local-expert bitset covers every routed expert of the moe_experts geometry.
static_assert(kMoeHelperExperts == kMoeExperts && kMoeExpertMaskWords * 32 == kMoeExperts);

namespace {

constexpr const char* kOp             = "moe_experts";
constexpr std::size_t kWorkspaceAlign = 256;

void require(bool condition, const char* what) {
    if (!condition) { throw std::invalid_argument(std::string(kOp) + ": " + what); }
}

void require_tensor(const Tensor& t, DType dtype, std::int32_t ne0, std::int32_t ne1,
                    const char* what) {
    require(t.dtype == dtype && t.ne[0] == ne0 && t.ne[1] == ne1 && t.ne[2] == 1 && t.ne[3] == 1 &&
                t.is_contiguous() && t.data != nullptr &&
                (reinterpret_cast<std::uintptr_t>(t.data) % 16) == 0,
            what);
}

bool overlaps(const Tensor& a, const Tensor& b) {
    const auto a_begin = reinterpret_cast<std::uintptr_t>(a.data);
    const auto b_begin = reinterpret_cast<std::uintptr_t>(b.data);
    return a_begin < b_begin + b.bytes() && b_begin < a_begin + a.bytes();
}

constexpr std::size_t align_up(std::size_t bytes) {
    return (bytes + kWorkspaceAlign - 1) / kWorkspaceAlign * kWorkspaceAlign;
}

// Region sizes of the workspace layout in detail::MoeExpertsWorkspace, in allocation order; the
// four A8 encodings are zero for A16.
struct WorkspaceBytes {
    std::size_t jobs;
    std::size_t schedule;
    std::size_t sorted;
    std::size_t gated;
    std::size_t x_codes;
    std::size_t x_scales;
    std::size_t gated_codes;
    std::size_t gated_scales;
    std::size_t contributions;
};

constexpr WorkspaceBytes workspace_bytes(std::int32_t tokens, MoeActivationProfile profile) {
    const auto rows        = static_cast<std::size_t>(tokens);
    const auto assignments = rows * detail::kMoeSlots;
    const bool a8          = profile == MoeActivationProfile::A8;
    const auto max_jobs    = static_cast<std::size_t>(
        detail::moe_experts_max_jobs(tokens, detail::moe_experts_token_tile(tokens, profile)));
    return {max_jobs * sizeof(detail::MoeJob),
            sizeof(detail::MoeJobSchedule),
            assignments * sizeof(std::int32_t),
            assignments * kMoeIntermediate * sizeof(std::uint16_t),
            a8 ? rows * kMoeHidden : 0,
            a8 ? rows * sizeof(float) : 0,
            a8 ? assignments * kMoeIntermediate : 0,
            a8 ? assignments * sizeof(float) : 0,
            assignments * kMoeHidden * sizeof(float)};
}

// High-water bytes from a 256-byte aligned cursor: internal alignment gaps, no trailing padding.
constexpr std::size_t workspace_total(std::int32_t tokens, MoeActivationProfile profile) {
    const WorkspaceBytes bytes = workspace_bytes(tokens, profile);
    return align_up(bytes.jobs) + align_up(bytes.schedule) + align_up(bytes.sorted) +
           align_up(bytes.gated) + align_up(bytes.x_codes) + align_up(bytes.x_scales) +
           align_up(bytes.gated_codes) + align_up(bytes.gated_scales) + bytes.contributions;
}

} // namespace

std::size_t moe_experts_workspace_capacity_bytes(std::int32_t max_tokens,
                                                 MoeActivationProfile profile) {
    if (max_tokens < 1 || max_tokens > kMoeMaxTokens) {
        throw std::invalid_argument(
            "moe_experts_workspace_capacity_bytes: max_tokens must be in [1,16384]");
    }
    // The A16 job bound is not monotone in T (the token tile widens with T), so take the maximum.
    std::size_t capacity = 0;
    for (std::int32_t tokens = 1; tokens <= max_tokens; ++tokens) {
        capacity = std::max(capacity, workspace_total(tokens, profile));
    }
    return capacity;
}

namespace {

// One call of either Op: an ordinary call writes BF16 y; a local-partial call (local_experts not
// null) writes FP32 y.
void run(const Tensor& x, const Tensor& expert_ids, const Tensor& route_weights,
         const Tensor& shared_scale, const MoeExpertBank& bank, MoeActivationProfile profile,
         const Tensor* local_experts, Tensor& y, WorkspaceArena& workspace, cudaStream_t stream,
         const MoeExpertsDeferral* deferral) {
    const std::int32_t tokens = x.ne[1];
    require(tokens >= 1 && tokens <= kMoeMaxTokens, "T must be in [1,16384]");
    require(profile == MoeActivationProfile::A16 || profile == MoeActivationProfile::A8,
            "profile must be A16 or A8");
    require_tensor(x, DType::BF16, kMoeHidden, tokens,
                   "x must be aligned contiguous BF16 [2560,T]");
    require_tensor(expert_ids, DType::I32, kMoeTopK, tokens,
                   "expert_ids must be aligned contiguous I32 [10,T]");
    require_tensor(route_weights, DType::FP32, kMoeTopK, tokens,
                   "route_weights must be aligned contiguous FP32 [10,T]");
    require_tensor(shared_scale, DType::FP32, tokens, 1,
                   "shared_scale must be aligned contiguous FP32 [T]");
    if (local_experts == nullptr) {
        require_tensor(y, DType::BF16, kMoeHidden, tokens,
                       "y must be aligned contiguous BF16 [2560,T]");
    } else {
        require_tensor(y, DType::FP32, kMoeHidden, tokens,
                       "partial must be aligned contiguous FP32 [2560,T]");
        require_tensor(*local_experts, DType::I32, kMoeExpertMaskWords, 1,
                       "local_experts must be aligned contiguous I32 [16]");
        require(!overlaps(y, *local_experts), "partial must be disjoint from local_experts");
    }
    require_tensor(bank.expert_table, DType::I64, kMoeExperts, 1,
                   "expert_table must be aligned contiguous I64 [512]");
    require_tensor(bank.matrix_scales, DType::FP32, 3, kMoeExperts,
                   "matrix_scales must be aligned contiguous FP32 [512,3]");
    require_tensor(bank.shared_record, DType::U8, static_cast<std::int32_t>(kMoeNvfp4RecordBytes),
                   1, "shared_record must be contiguous U8 [2764800]");
    require(reinterpret_cast<std::uintptr_t>(bank.shared_record.data) %
                    static_cast<std::uintptr_t>(kMoeRecordAlignment) ==
                0,
            "shared_record must be 256-byte aligned");
    require_tensor(bank.shared_matrix_scales, DType::FP32, 3, 1,
                   "shared_matrix_scales must be aligned contiguous FP32 [3]");
    const bool partial = local_experts != nullptr;
    for (const Tensor* input :
         {&x, &expert_ids, &route_weights, &shared_scale, &bank.expert_table, &bank.matrix_scales,
          &bank.shared_record, &bank.shared_matrix_scales}) {
        require(!overlaps(y, *input), partial ? "partial must be disjoint from every input"
                                              : "y must be disjoint from every input");
    }
    if (deferral != nullptr) {
        require_tensor(deferral->slots, DType::I64, kMoeExperts, 1,
                       "deferral slots must be aligned contiguous I64 [512]");
        require(deferral->ready != nullptr, "deferral needs a ready event");
        require(!overlaps(y, deferral->slots),
                partial ? "partial must be disjoint from the deferral slots"
                        : "y must be disjoint from the deferral slots");
    }

    auto scope                 = workspace.scope();
    const WorkspaceBytes bytes = workspace_bytes(tokens, profile);
    const auto carve           = [&](std::size_t size) {
        return workspace.alloc_bytes(size, kWorkspaceAlign).data;
    };
    detail::MoeExpertsWorkspace scratch{};
    scratch.jobs     = static_cast<detail::MoeJob*>(carve(bytes.jobs));
    scratch.schedule = static_cast<detail::MoeJobSchedule*>(carve(bytes.schedule));
    scratch.sorted   = static_cast<std::int32_t*>(carve(bytes.sorted));
    scratch.gated    = static_cast<std::uint16_t*>(carve(bytes.gated));
    if (profile == MoeActivationProfile::A8) {
        scratch.x_codes      = static_cast<std::uint8_t*>(carve(bytes.x_codes));
        scratch.x_scales     = static_cast<float*>(carve(bytes.x_scales));
        scratch.gated_codes  = static_cast<std::uint8_t*>(carve(bytes.gated_codes));
        scratch.gated_scales = static_cast<float*>(carve(bytes.gated_scales));
    }
    scratch.contributions = static_cast<float*>(carve(bytes.contributions));

    detail::MoeExpertsOperands operands;
    operands.profile       = profile;
    operands.x             = static_cast<const std::uint16_t*>(x.data);
    operands.expert_ids    = static_cast<const std::int32_t*>(expert_ids.data);
    operands.route_weights = static_cast<const float*>(route_weights.data);
    operands.shared_scale  = static_cast<const float*>(shared_scale.data);
    operands.expert_table  = static_cast<const std::uint64_t*>(bank.expert_table.data);
    if (deferral != nullptr) {
        operands.deferred_slots = static_cast<const std::uint64_t*>(deferral->slots.data);
        operands.deferred_ready = deferral->ready;
    }
    operands.matrix_scales        = static_cast<const float*>(bank.matrix_scales.data);
    operands.f3_books             = bank.f3_books;
    operands.shared_record        = static_cast<const std::uint8_t*>(bank.shared_record.data);
    operands.shared_matrix_scales = static_cast<const float*>(bank.shared_matrix_scales.data);
    if (local_experts == nullptr) {
        operands.y = static_cast<std::uint16_t*>(y.data);
    } else {
        operands.local_experts = static_cast<const std::uint32_t*>(local_experts->data);
        operands.partial       = static_cast<float*>(y.data);
    }
    operands.tokens = tokens;
    detail::moe_experts_launch(operands, scratch, stream);
}

} // namespace

void moe_experts(const Tensor& x, const Tensor& expert_ids, const Tensor& route_weights,
                 const Tensor& shared_scale, const MoeExpertBank& bank,
                 MoeActivationProfile profile, Tensor& y, WorkspaceArena& workspace,
                 cudaStream_t stream, const MoeExpertsDeferral* deferral) {
    run(x, expert_ids, route_weights, shared_scale, bank, profile, nullptr, y, workspace, stream,
        deferral);
}

void moe_experts_local_partial(const Tensor& x, const Tensor& expert_ids,
                               const Tensor& route_weights, const Tensor& shared_scale,
                               const MoeExpertBank& bank, MoeActivationProfile profile,
                               const Tensor& local_experts, Tensor& partial,
                               WorkspaceArena& workspace, cudaStream_t stream,
                               const MoeExpertsDeferral* deferral) {
    run(x, expert_ids, route_weights, shared_scale, bank, profile, &local_experts, partial,
        workspace, stream, deferral);
}

void moe_activation_quantize(const Tensor& x, Tensor& codes, Tensor& scales, cudaStream_t stream) {
    const std::int32_t tokens = x.ne[1];
    const auto fail           = [](const char* what) {
        throw std::invalid_argument(std::string("moe_activation_quantize: ") + what);
    };
    if (tokens < 1 || tokens > kMoeMaxTokens) { fail("T must be in [1,16384]"); }
    const auto valid = [&](const Tensor& t, DType dtype, std::int32_t ne0, std::int32_t ne1) {
        return t.dtype == dtype && t.ne[0] == ne0 && t.ne[1] == ne1 && t.ne[2] == 1 &&
               t.ne[3] == 1 && t.is_contiguous() && t.data != nullptr &&
               (reinterpret_cast<std::uintptr_t>(t.data) % 16) == 0;
    };
    if (!valid(x, DType::BF16, kMoeHidden, tokens)) {
        fail("x must be aligned contiguous BF16 [2560,T]");
    }
    if (!valid(codes, DType::FP8_E4M3FN, kMoeHidden, tokens)) {
        fail("codes must be aligned contiguous FP8_E4M3FN [2560,T]");
    }
    if (!valid(scales, DType::FP32, tokens, 1)) {
        fail("scales must be aligned contiguous FP32 [T]");
    }
    if (overlaps(codes, x) || overlaps(scales, x) || overlaps(codes, scales)) {
        fail("codes and scales must be disjoint from x and from each other");
    }
    detail::moe_activation_quantize_launch(static_cast<const std::uint16_t*>(x.data),
                                           static_cast<std::uint8_t*>(codes.data),
                                           static_cast<float*>(scales.data), tokens, stream);
}

} // namespace ninfer::ops
