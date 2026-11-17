#include "runtime/forward.h"
#include "runtime/dflash.h"

#include "core/device.h"
#include "core/layout.h"
#include "ninfer/ops/attn_input_proj.h"
#include "ninfer/ops/argmax.h"
#include "ninfer/ops/bf16_projection.h"
#include "ninfer/ops/causal_conv1d_silu.h"
#include "ninfer/ops/embedding.h"
#include "ninfer/ops/gated_delta_net.h"
#include "ninfer/ops/gated_rmsnorm.h"
#include "ninfer/ops/gdn_gating.h"
#include "ninfer/ops/gdn_input_proj.h"
#include "ninfer/ops/gdn_replay.h"
#include "ninfer/ops/hyper_connection.h"
#include "ninfer/ops/linear.h"
#include "ninfer/ops/moe_experts.h"
#include "ninfer/ops/moe_route.h"
#include "ninfer/ops/ple.h"
#include "ninfer/ops/qsa.h"
#include "ninfer/ops/rmsnorm.h"
#include "ninfer/ops/rope.h"
#include "ninfer/ops/sampling.h"
#include "ninfer/ops/scatter.h"
#include "ninfer/ops/speculative_round.h"
#include "ninfer/ops/stream_mean.h"
#include "ninfer/ops/sigmoid_mul.h"
#include "ninfer/ops/token_logprobs.h"

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <string>

namespace ninfer::targets::qwen3_8_flash_next {
namespace {

constexpr std::size_t kAlignment = 256;
// 1/sqrt(128), the GDN query scale.
constexpr float kGdnScale = 0.08838834764831845F;

// Head columns of one score() group with LogitColumns::Last: wide enough that the FP8 head stays
// at the DRAM roof (one weight read per group), small enough that the logits stay 32 MB.
constexpr std::int32_t kScoreColumns = 64;

// Columns of every sub-chunk tensor: the widest sub-chunk or the widest decode round.
std::int32_t batch_columns(const ForwardOptions& options) {
    return std::max(options.max_chunk_tokens, kMaxLanes * std::max(1, options.max_verify_width));
}

// Columns of every super-chunk tensor.
std::int32_t super_columns(const ForwardOptions& options) {
    return std::max(options.super_chunk_tokens, kMaxLanes * std::max(1, options.max_verify_width));
}

// Columns of the vocabulary head a chunk storage holds: every sub-chunk column (All), or a score
// group, which also covers every decode row (Last).
std::int32_t logit_columns(const ForwardOptions& options) {
    return options.logits == LogitColumns::All
               ? batch_columns(options)
               : std::max(kScoreColumns, kMaxLanes * options.max_verify_width);
}

static_assert(kScoreColumns >= kMaxLanes);

// A sub-chunk of T tokens views the first T columns of a [rows, max] storage tensor.
Tensor columns(const Tensor& storage, std::int32_t rows, std::int32_t tokens) {
    return Tensor(storage.data, storage.dtype, {rows, tokens});
}

// Input blocks of a super-chunk: one per sub-chunk of max_chunk_tokens.
std::int32_t input_block_count(const ForwardOptions& options) {
    return (options.super_chunk_tokens + options.max_chunk_tokens - 1) / options.max_chunk_tokens;
}

// The per-call scratch below is allocated through templates so that the dry run on a
// WorkspaceLayoutBuilder and the real WorkspaceArena allocations share every size, alignment and
// order.
struct GdnScratch {
    Tensor qkv;    // BF16 [10240,T], projected q|k|v before the convolution
    Tensor z;      // BF16 [6144,T] = [128,48,T]
    Tensor query;  // BF16 [2048,T] = [128,16,T], after convolution and SiLU
    Tensor key;    // BF16 [2048,T] = [128,16,T]
    Tensor value;  // BF16 [6144,T] = [128,48,T]
    Tensor gates;  // FP32 [96,T], a then b
    Tensor decay;  // FP32 [48,T], g
    Tensor beta;   // FP32 [48,T]
    Tensor core;   // BF16 [128,48,T], recurrence output
    Tensor normed; // BF16 [128,48,T], gated RMSNorm output
};

template <class Allocator>
GdnScratch allocate_gdn(Allocator& a, std::int32_t t) {
    return GdnScratch{
        .qkv    = a.alloc(DType::BF16, {kGdnConvChannels, t}),
        .z      = a.alloc(DType::BF16, {kGdnHeadDim, kGdnValueHeads, t}),
        .query  = a.alloc(DType::BF16, {kGdnKeyDim, t}),
        .key    = a.alloc(DType::BF16, {kGdnKeyDim, t}),
        .value  = a.alloc(DType::BF16, {kGdnValueDim, t}),
        .gates  = a.alloc(DType::FP32, {kGdnGateRows, t}),
        .decay  = a.alloc(DType::FP32, {kGdnValueHeads, t}),
        .beta   = a.alloc(DType::FP32, {kGdnValueHeads, t}),
        .core   = a.alloc(DType::BF16, {kGdnHeadDim, kGdnValueHeads, t}),
        .normed = a.alloc(DType::BF16, {kGdnHeadDim, kGdnValueHeads, t}),
    };
}

struct QsaScratch {
    Tensor query;          // BF16 [6144,T] = [256,24,T], projected
    Tensor gate;           // BF16 [6144,T], output gate logits
    Tensor key;            // BF16 [512,T] = [256,2,T], projected
    Tensor value;          // BF16 [512,T] = [256,2,T]
    Tensor query_rotated;  // BF16 [256,24,T], normalized and rotated
    Tensor key_rotated;    // BF16 [256,2,T], normalized and rotated
    Tensor index_query;    // BF16 [512,T] = [128,4,T]
    Tensor index_key;      // BF16 [128,T]
    Tensor selected_ids;   // I32 [S,T]
    Tensor selected_count; // I32 [T]
    Tensor attention;      // BF16 [256,24,T]
};

template <class Allocator>
QsaScratch allocate_qsa(Allocator& a, std::int32_t t, std::int32_t selected) {
    return QsaScratch{
        .query          = a.alloc(DType::BF16, {kQsaQueryDim, t}),
        .gate           = a.alloc(DType::BF16, {kQsaQueryDim, t}),
        .key            = a.alloc(DType::BF16, {kQsaKvDim, t}),
        .value          = a.alloc(DType::BF16, {kQsaKvDim, t}),
        .query_rotated  = a.alloc(DType::BF16, {ops::kQsaHeadDim, ops::kQsaQueryHeads, t}),
        .key_rotated    = a.alloc(DType::BF16, {ops::kQsaHeadDim, ops::kQsaKvHeads, t}),
        .index_query    = a.alloc(DType::BF16, {kQsaIndexQueryDim, t}),
        .index_key      = a.alloc(DType::BF16, {ops::kQsaIndexHeadDim, t}),
        .selected_ids   = a.alloc(DType::I32, {selected, t}),
        .selected_count = a.alloc(DType::I32, {t}),
        .attention      = a.alloc(DType::BF16, {ops::kQsaHeadDim, ops::kQsaQueryHeads, t}),
    };
}

struct PleScratch {
    Tensor embedding; // BF16 [2560,T], decoded n-gram rows
    Tensor key;       // FP32 [10240,T]
    Tensor value;     // FP32 [2560,T]
};

template <class Allocator>
PleScratch allocate_ple(Allocator& a, std::int32_t t) {
    return PleScratch{
        .embedding = a.alloc(DType::BF16, {ops::kPleEmbedding, t}),
        .key       = a.alloc(DType::FP32, {kPleKeyRows, t}),
        .value     = a.alloc(DType::FP32, {kPleValueRows, t}),
    };
}

struct MoeScratch {
    Tensor control;       // FP32 [544,T]
    Tensor expert_ids;    // I32 [10,T]
    Tensor route_weights; // FP32 [10,T]
    Tensor shared_scale;  // FP32 [T]
};

template <class Allocator>
MoeScratch allocate_moe(Allocator& a, std::int32_t t) {
    return MoeScratch{
        .control       = a.alloc(DType::FP32, {ops::kMoeControlRows, t}),
        .expert_ids    = a.alloc(DType::I32, {ops::kMoeTopK, t}),
        .route_weights = a.alloc(DType::FP32, {ops::kMoeTopK, t}),
        .shared_scale  = a.alloc(DType::FP32, {t}),
    };
}

// The staging budget (the Forward class comment): the bytes a staging fill copies while layer l's
// dense work runs, from layer l - 1's MoE merge to layer l's routing. Those windows, measured on
// the RTX 5090 at Gen5 x16 (fixed K3, 4096-token prompts, nsys, 2026-11-16), grow with the round's
// T tokens: GDN layers 82 / 102 / 116 us and QSA layers 100 / 121 / 140 us at T = 4 / 16 / 20. The
// lines through the T = 4 and 16 points bound them from below (6-9% under at T = 20). The fill
// copies at kStagingBytesPerNs (12 CTAs reach 44-52 GB/s) after kStagingSetupNs of stage kernel
// and stream wakeup.
constexpr std::int64_t kGdnWindowBaseNs     = 75'300;
constexpr std::int64_t kGdnWindowPerTokenNs = 1'670;
constexpr std::int64_t kQsaWindowBaseNs     = 93'000;
constexpr std::int64_t kQsaWindowPerTokenNs = 1'750;
constexpr std::int64_t kStagingSetupNs      = 5'000;
constexpr std::int64_t kStagingBytesPerNs   = 46;

std::size_t staging_budget_bytes(std::int32_t layer, std::int32_t tokens) {
    const std::int64_t window_ns = is_qsa_layer(layer)
                                       ? kQsaWindowBaseNs + kQsaWindowPerTokenNs * tokens
                                       : kGdnWindowBaseNs + kGdnWindowPerTokenNs * tokens;
    return static_cast<std::size_t>(std::max<std::int64_t>(window_ns - kStagingSetupNs, 0) *
                                    kStagingBytesPerNs);
}

// Columns of the widest decode or verification round.
std::int32_t round_columns(const ForwardOptions& options, std::int32_t lanes) {
    return lanes * std::max(1, options.max_verify_width);
}

// The staging span: the largest budget of any layer at the widest round (the budget grows with T).
std::size_t staging_span_bytes(std::int32_t tokens) {
    std::size_t bytes = 0;
    for (std::int32_t layer = kFirstStagedLayer; layer < kLayers; ++layer) {
        bytes = std::max(bytes, staging_budget_bytes(layer, tokens));
    }
    return bytes;
}

std::int32_t selected_slots(ops::QsaSelectBudget budget) {
    return ops::qsa_selected_capacity(budget);
}

void reserve(WorkspaceLayoutBuilder& builder, std::size_t bytes) {
    if (bytes != 0) { static_cast<void>(builder.alloc_bytes(bytes, kAlignment)); }
}

// Peak transient bytes over every prefill sub-chunk width in [1, max_chunk_tokens], every decode
// round of 1 .. lanes rows, and every visible prefix up to the lane capacity. Each mixer, PLE and
// MoE block allocates its scratch in a scope and then calls Ops that allocate their own workspace
// above it and release it on return; sampling runs at the top level.
std::size_t plan_work_bytes(const ModelView& model, const ForwardOptions& options,
                            const State& state) {
    const std::int32_t max_tokens = batch_columns(options);
    const std::int32_t lanes      = state.lanes();
    const std::int32_t capacity   = state.lane_capacity();
    const std::int32_t selected   = selected_slots(options.select_budget);
    // Selection workspace grows with the query count but not monotonically with the visible
    // prefix (its score rows are chunked by prefix length), and the attention split count, and so
    // its workspace, is not monotone in the width: take the maximum over each, for the widest
    // prefill sub-chunk and every decode batch.
    std::size_t qsa_ops = 0;
    for (std::int32_t visible = ops::kQsaBlockTokens; visible <= capacity;
         visible += ops::kQsaBlockTokens) {
        qsa_ops = std::max(qsa_ops, ops::qsa_index_select_workspace_bytes(max_tokens, visible,
                                                                          options.select_budget));
        for (std::int32_t rows = 1; rows <= lanes; ++rows) {
            qsa_ops = std::max(qsa_ops, ops::qsa_index_select_workspace_bytes(
                                            rows, visible, options.select_budget));
        }
    }
    for (std::int32_t width = 1; width <= max_tokens; ++width) {
        qsa_ops = std::max(qsa_ops, ops::qsa_selected_attention_workspace_bytes(width, selected));
    }
    // The decode GDN projection stages W*B columns; every layer shares the parent format.
    const Weight& parent  = model.gdn.front().query_key_value_z;
    std::size_t snapshots = 0;
    for (std::int32_t rows = 1; rows <= lanes; ++rows) {
        snapshots = std::max(snapshots, ops::gdn_input_proj_conv_snapshot_workspace_capacity_bytes(
                                            parent.qtype, kGdnProjectionRows, kHidden,
                                            ops::LinearPolicy::A16Only, rows, 1, 1));
    }
    if (options.max_verify_width != 0) {
        for (std::int32_t rows = 1; rows <= lanes; ++rows) {
            snapshots = std::max(
                snapshots, ops::gdn_input_proj_conv_record_workspace_capacity_bytes(
                               parent.qtype, kGdnProjectionRows, kHidden,
                               ops::LinearPolicy::A16Only, rows, 1, options.max_verify_width));
        }
    }
    WorkspaceLayoutBuilder builder;
    {
        auto scope = builder.scope();
        // The head reads groups of logit_columns columns (score) besides every batch width.
        reserve(builder, ops::hyper_connection_read_workspace_bytes(
                             std::max(max_tokens, logit_columns(options))));
    }
    {
        auto scope = builder.scope();
        static_cast<void>(allocate_gdn(builder, max_tokens));
        reserve(builder, std::max(ops::gated_delta_net_workspace_capacity_bytes(
                                      kGdnKeyHeads, kGdnValueHeads, true, 1, max_tokens),
                                  snapshots));
    }
    {
        auto scope = builder.scope();
        static_cast<void>(allocate_qsa(builder, max_tokens, selected));
        reserve(builder, qsa_ops);
    }
    {
        auto scope = builder.scope();
        static_cast<void>(allocate_ple(builder, max_tokens));
        reserve(builder, ops::ple_inject_workspace_bytes(max_tokens));
    }
    {
        auto scope = builder.scope();
        static_cast<void>(allocate_moe(builder, max_tokens));
        reserve(builder, ops::moe_experts_workspace_capacity_bytes(max_tokens));
    }
    {
        auto scope = builder.scope();
        reserve(builder, ops::sampling_workspace_capacity_bytes(kTokenDomain, 1, lanes));
    }
    if (options.dflash_features == DflashExecution::Enabled) {
        auto scope = builder.scope();
        reserve(builder, ops::speculative_accept_greedy_drafts_workspace_capacity_bytes(
                             kTokenDomain, 1, kDflashMaxK, 1, lanes));
    }
    return builder.peak_bytes(kAlignment);
}

// F3 slots each streaming window borrows: enough whole slots for expert_window_bytes, or 0 when
// no super-chunk can stream or the F3 pool cannot lend both windows.
std::int32_t window_slots(const ForwardOptions& options, const ops::MoeExpertCache& cache) {
    const ops::MoeExpertSlotPool& pool = cache.pools[static_cast<std::size_t>(kWindowCodec)];
    if (options.expert_window_bytes == 0 ||
        options.super_chunk_tokens < options.window_min_tokens || pool.record_bytes == 0) {
        return 0;
    }
    const std::size_t per_window =
        (options.expert_window_bytes + pool.record_bytes - 1) / pool.record_bytes;
    const std::size_t pool_slots = pool.slots.bytes / pool.record_bytes;
    return 2 * per_window <= pool_slots ? static_cast<std::int32_t>(per_window) : 0;
}

ForwardOptions validated(const ForwardOptions& options) {
    if (options.max_chunk_tokens < 1 || options.max_chunk_tokens > ops::kMoeMaxTokens) {
        throw std::invalid_argument("qwen3_8_flash_next Forward: max_chunk_tokens must be in [1, " +
                                    std::to_string(ops::kMoeMaxTokens) + "]");
    }
    if (options.super_chunk_tokens < options.max_chunk_tokens ||
        options.super_chunk_tokens > ops::kQsaMaximumCapacity) {
        throw std::invalid_argument("qwen3_8_flash_next Forward: super_chunk_tokens must lie in "
                                    "[max_chunk_tokens, " +
                                    std::to_string(ops::kQsaMaximumCapacity) + "]");
    }
    if (options.window_min_tokens < 2 ||
        options.expert_window_bytes % static_cast<std::size_t>(ops::kMoeRecordAlignment) != 0) {
        throw std::invalid_argument("qwen3_8_flash_next Forward: window_min_tokens must be at "
                                    "least 2 and expert_window_bytes a multiple of 256");
    }
    if (options.max_verify_width != 0 &&
        (options.max_verify_width < 2 || options.max_verify_width > 8)) {
        throw std::invalid_argument("Flash-Next verify capacity must be 0 or in [2,8]");
    }
    if (options.dflash_features == DflashExecution::Enabled &&
        options.max_verify_width != kDflashWidth) {
        throw std::invalid_argument("Flash-Next DFlash features require four-column capacity");
    }
    return options;
}

void check_tokens(std::span<const std::int32_t> tokens) {
    for (const std::int32_t token : tokens) {
        if (token < 0 || token >= kVocabRows) {
            throw std::invalid_argument("qwen3_8_flash_next Forward: token id " +
                                        std::to_string(token) + " is out of range");
        }
    }
}

// A device tensor at byte `offset` of `base`.
Tensor at(const Tensor& base, std::size_t offset, DType dtype,
          std::initializer_list<std::int32_t> shape) {
    return Tensor(static_cast<std::byte*>(base.data) + offset, dtype, shape);
}

} // namespace

template <class Allocator>
std::vector<Forward::ChunkInputs> Forward::allocate_inputs(Allocator& allocator,
                                                           const ForwardOptions& options) {
    const std::int32_t max_tokens = options.max_chunk_tokens;
    std::vector<ChunkInputs> blocks;
    blocks.reserve(static_cast<std::size_t>(input_block_count(options)));
    for (std::int32_t block = 0; block < input_block_count(options); ++block) {
        blocks.push_back(ChunkInputs{
            .token_ids = allocator.alloc(DType::I32, {max_tokens}),
            .slots     = allocator.alloc(DType::I32, {max_tokens}),
            .positions = allocator.alloc(DType::I32, {max_tokens, kMropeAxes}),
        });
    }
    return blocks;
}

template <class Allocator>
Forward::ChunkStorage Forward::allocate_chunk(Allocator& allocator, const ForwardOptions& options) {
    const std::int32_t max_tokens   = batch_columns(options);
    const std::int32_t super_tokens = super_columns(options);
    const std::int32_t logit_width  = logit_columns(options);
    ChunkStorage storage{.input_blocks = allocate_inputs(allocator, options)};
    storage.ple_rows =
        allocator.alloc(DType::U8, {ops::kPleInt8RowBytes, ops::kPleRowsPerToken, super_tokens});
    storage.residual =
        allocator.alloc(DType::BF16, {kHidden, ops::kHyperConnectionStreams, super_tokens});
    storage.block_input  = allocator.alloc(DType::BF16, {kHidden, max_tokens});
    storage.write_scale  = allocator.alloc(DType::FP32, {ops::kHyperConnectionStreams, max_tokens});
    storage.block_output = allocator.alloc(DType::BF16, {kHidden, max_tokens});
    storage.head_input   = allocator.alloc(DType::BF16, {kHidden, logit_width});
    storage.logits       = allocator.alloc(DType::BF16, {kVocabRows, logit_width});
    const std::int32_t gdn_columns = kMaxLanes * std::max(1, options.max_verify_width);
    storage.decode_gdn_query       = allocator.alloc(DType::BF16, {kGdnKeyDim, gdn_columns});
    storage.decode_gdn_key         = allocator.alloc(DType::BF16, {kGdnKeyDim, gdn_columns});
    storage.decode_gdn_value       = allocator.alloc(DType::BF16, {kGdnValueDim, gdn_columns});
    storage.decode_gdn_z           = allocator.alloc(DType::BF16, {kGdnValueDim, gdn_columns});
    storage.ingress =
        allocator.alloc(DType::U8, {static_cast<std::int32_t>(sizeof(DecodeIngress))});
    storage.egress =
        allocator.alloc(DType::U8, {static_cast<std::int32_t>(sizeof(SamplingEgress))});
    storage.row_enabled        = allocator.alloc(DType::I32, {1});
    storage.score_targets      = allocator.alloc(DType::I32, {super_tokens});
    storage.score_logprobs     = allocator.alloc(DType::FP32, {super_tokens});
    storage.score_top_ids      = allocator.alloc(DType::I32, {logit_width});
    storage.score_top_logprobs = allocator.alloc(DType::FP32, {logit_width});
    storage.expert_cache_counters =
        allocator.alloc(DType::I64, {ops::kMoeCacheCounterKinds, ops::kMoeCacheCodecTags});
    storage.window_columns = allocator.alloc(DType::I64, {ops::kMoeExperts, 2});
    if (options.dflash_features == DflashExecution::Enabled) {
        storage.dflash_features = allocator.alloc(DType::BF16, {kDflashFeatures, kDflashWindow});
    }
    return storage;
}

template <class Allocator>
Forward::StagingStorage
Forward::allocate_staging(Allocator& allocator, const ForwardOptions& options, std::int32_t lanes) {
    if (options.expert_staging == ExpertStaging::Disabled) { return {}; }
    const std::int32_t tokens = round_columns(options, lanes);
    return StagingStorage{
        .control  = allocator.alloc(DType::FP32, {ops::kMoeControlRows, tokens}),
        .ids      = allocator.alloc(DType::I32, {ops::kMoeTopK, tokens}),
        .weights  = allocator.alloc(DType::FP32, {ops::kMoeTopK, tokens}),
        .shared   = allocator.alloc(DType::FP32, {tokens}),
        .stage    = allocator.alloc(DType::I64, {ops::kMoeExperts, 2}),
        .source   = allocator.alloc(DType::I64, {ops::kMoeExperts}),
        .deferred = allocator.alloc(DType::I64, {ops::kMoeExperts}),
        .span     = allocator.alloc_bytes(staging_span_bytes(tokens),
                                          static_cast<std::size_t>(ops::kMoeRecordAlignment)),
    };
}

std::size_t Forward::plan_staging_bytes(const ForwardOptions& options, std::int32_t lanes) {
    WorkspaceLayoutBuilder builder;
    static_cast<void>(allocate_staging(builder, options, lanes));
    return builder.peak_bytes(kAlignment);
}

bool Forward::stages(std::int32_t layer, const SubChunk& chunk, bool streamed) const {
    return options_.expert_staging == ExpertStaging::Enabled && !streamed && chunk.lane < 0 &&
           layer >= kFirstStagedLayer && layer < kLayers;
}

std::size_t Forward::plan_input_bytes(const ForwardOptions& options) {
    WorkspaceLayoutBuilder builder;
    static_cast<void>(allocate_inputs(builder, options));
    return builder.peak_bytes(1);
}

std::size_t Forward::plan_activation_bytes(const ForwardOptions& options) {
    WorkspaceLayoutBuilder builder;
    static_cast<void>(allocate_chunk(builder, options));
    return builder.peak_bytes(kAlignment);
}

void Forward::validate(const ForwardOptions& options) { static_cast<void>(validated(options)); }

std::size_t Forward::device_bytes(const ModelView& model, const ForwardOptions& options,
                                  const State& state) {
    const ForwardOptions checked = validated(options);
    return plan_activation_bytes(checked) + plan_verify_bytes(checked, state.lanes()) +
           plan_staging_bytes(checked, state.lanes()) + plan_work_bytes(model, checked, state);
}

Forward::Forward(const ModelView& model, State& state, PleRowFeeder& feeder,
                 ExpertResidency& experts, const ForwardOptions& options, cudaStream_t stream,
                 Dflash* drafter)
    : model_(model), drafter_(drafter), state_(state), feeder_(feeder),
      expert_table_(experts.expert_table()), host_table_(experts.host_table()),
      fill_table_(experts.fill_table()), cache_(experts.cache()), options_(validated(options)),
      stream_(stream),
      activations_(plan_activation_bytes(options_) + plan_verify_bytes(options_, state.lanes()) +
                   plan_staging_bytes(options_, state.lanes())),
      work_(plan_work_bytes(model, options_, state)),
      chunk_(allocate_chunk(activations_, options_)),
      verify_(allocate_verify(activations_, options_, state.lanes())),
      staging_(allocate_staging(activations_, options_, state.lanes())),
      input_staging_(plan_input_bytes(options_)),
      inputs_uploaded_(cudaEventDisableTiming | cudaEventBlockingSync),
      decode_rows_(static_cast<std::size_t>(kMaxLanes) *
                   static_cast<std::size_t>(std::max(1, options_.max_verify_width)) *
                   ops::kPleRowsPerToken * ops::kPleInt8RowBytes),
      ingress_staging_(sizeof(DecodeIngress)), egress_(sizeof(SamplingEgress)),
      replay_start_(cudaEventBlockingSync), replay_stop_(cudaEventBlockingSync),
      admitted_(cudaEventDisableTiming), filled_(cudaEventDisableTiming),
      input_ready_(cudaEventDisableTiming), stage_planned_(cudaEventDisableTiming),
      experts_done_(cudaEventDisableTiming), window_slots_(window_slots(options_, cache_)),
      window_planned_{CudaEvent(cudaEventDisableTiming), CudaEvent(cudaEventDisableTiming)},
      window_ready_{CudaEvent(cudaEventDisableTiming), CudaEvent(cudaEventDisableTiming)},
      window_free_{CudaEvent(cudaEventDisableTiming), CudaEvent(cudaEventDisableTiming)},
      window_start_(cudaEventDisableTiming) {
    if (chunk_.input_blocks.front().token_ids.data != activations_.base()) {
        throw std::logic_error("qwen3_8_flash_next Forward: inputs must start the activations");
    }
    // The pinned frames hold trivially copyable values; start them in their default state.
    std::construct_at(static_cast<DecodeIngress*>(ingress_staging_.data()));
    std::construct_at(static_cast<SamplingEgress*>(egress_.data()));
    // D2H transfers the fixed frame, including inactive lanes and disabled logprob fields.
    CUDA_CHECK(cudaMemsetAsync(chunk_.egress.data, 0, chunk_.egress.bytes(), stream_));
    if (options_.max_verify_width != 0) {
        verify_ingress_staging_.emplace(sizeof(VerifyIngress));
        verify_commit_staging_.emplace(sizeof(VerifyCommitIngress));
        std::construct_at(static_cast<VerifyIngress*>(verify_ingress_staging_->data()));
        std::construct_at(static_cast<VerifyCommitIngress*>(verify_commit_staging_->data()));
    }
    if ((drafter_ != nullptr) != (options_.dflash_features == DflashExecution::Enabled)) {
        throw std::invalid_argument("Flash-Next drafter and feature-capture configuration differ");
    }
    if (options_.dflash_features == DflashExecution::Enabled) {
        if (drafter_ == nullptr || !state_.has_dflash()) {
            throw std::invalid_argument(
                "Flash-Next speculative features require the native drafter");
        }
        spec_ingress_staging_.emplace(sizeof(SpecIngress));
        spec_egress_staging_.emplace(sizeof(SpecEgress));
        std::construct_at(static_cast<SpecIngress*>(spec_ingress_staging_->data()));
        std::construct_at(static_cast<SpecEgress*>(spec_egress_staging_->data()));
        drafts_ready_.emplace(cudaEventDisableTiming | cudaEventBlockingSync);
        CUDA_CHECK(
            cudaMemsetAsync(verify_.spec_egress.data, 0, verify_.spec_egress.bytes(), stream_));
    }
    constexpr std::int32_t kOne = 1;
    CUDA_CHECK(cudaMemcpyAsync(chunk_.row_enabled.data, &kOne, sizeof(kOne), cudaMemcpyHostToDevice,
                               stream_));
    clear_expert_cache_counts();
    CUDA_CHECK(cudaStreamSynchronize(stream_));
    // Capture reads no staging contents and needs no idle feeder (thread-local capture mode), so
    // it can run before any request.
    if (options_.logits == LogitColumns::Last) { capture_decode_graphs(); }
    if (drafter_ != nullptr) { capture_spec_graphs(); }
    feature_columns_ = 0;
}

Forward::~Forward() {
    // No enqueued work may outlive the buffers below; errors cannot propagate from here.
    static_cast<void>(cudaStreamSynchronize(stream_));
    static_cast<void>(cudaStreamSynchronize(fill_stream_.get()));
}

void Forward::clear_expert_cache_counts() {
    CUDA_CHECK(cudaMemsetAsync(chunk_.expert_cache_counters.data, 0,
                               chunk_.expert_cache_counters.bytes(), stream_));
}

ExpertCacheCounts Forward::expert_cache_counts() {
    std::array<std::int64_t,
               static_cast<std::size_t>(ops::kMoeCacheCounterKinds) * ops::kMoeCacheCodecTags>
        values{};
    CUDA_CHECK(cudaMemcpyAsync(values.data(), chunk_.expert_cache_counters.data, sizeof(values),
                               cudaMemcpyDeviceToHost, stream_));
    CUDA_CHECK(cudaStreamSynchronize(stream_));
    ExpertCacheCounts counts;
    for (std::size_t codec = 0; codec < counts.lookups.size(); ++codec) {
        const std::size_t base = static_cast<std::size_t>(ops::kMoeCacheCounterKinds) * codec;
        counts.lookups[codec] =
            values[base + static_cast<std::size_t>(ops::MoeCacheCounter::Lookups)];
        counts.misses[codec] =
            values[base + static_cast<std::size_t>(ops::MoeCacheCounter::Misses)];
    }
    return counts;
}

Forward::DecodeIngress& Forward::ingress_host() const noexcept {
    return *static_cast<DecodeIngress*>(ingress_staging_.data());
}

Tensor Forward::logits(std::int32_t count) const {
    if (count < 1 || count > chunk_.logits.ne[1]) {
        throw std::out_of_range("qwen3_8_flash_next Forward: logits columns out of range");
    }
    return columns(chunk_.logits, kVocabRows, count);
}

std::int32_t Forward::prefill_step(std::int32_t lane, std::span<const std::int32_t> remaining,
                                   const LogitsSink& sink, const ForwardObserver* observer,
                                   std::int32_t token_limit, PrefillTail tail) {
    require_resolved();
    pending_verify_width_ = 0;
    if (token_limit < 0) {
        throw std::invalid_argument("Flash-Next prefill limit must be nonnegative");
    }
    if (remaining.empty()) { throw std::invalid_argument("qwen3_8_flash_next Forward: no tokens"); }
    const auto span     = static_cast<std::size_t>(options_.super_chunk_tokens);
    const std::size_t n = std::min(
        {span, remaining.size(), token_limit == 0 ? span : static_cast<std::size_t>(token_limit)});
    feature_offset_ = static_cast<std::int32_t>(n > kDflashWindow ? n - kDflashWindow : 0);
    feature_columns_ =
        static_cast<std::int32_t>(std::min(n, static_cast<std::size_t>(kDflashWindow)));
    const auto chunk        = remaining.first(n);
    const auto rest         = remaining.subspan(n);
    const auto next         = rest.first(std::min(span, rest.size()));
    const std::int64_t room = state_.reserved_tokens(lane) - state_.committed(lane);
    if (static_cast<std::int64_t>(n) > room) {
        throw std::out_of_range("qwen3_8_flash_next Forward: " + std::to_string(n) +
                                " tokens exceed lane " + std::to_string(lane) +
                                "'s remaining reservation of " + std::to_string(room));
    }
    check_tokens(chunk);
    check_tokens(next);
    // The feeder holds this super-chunk when the previous step of this lane prefetched it;
    // anything else pending (another lane's prefetch, or a prefetch this step does not continue)
    // is cancelled, which restores that lane's history.
    if (!feeder_.pending_matches(lane, chunk)) {
        feeder_.cancel();
        feeder_.begin(lane, chunk, options_.max_chunk_tokens);
    }
    run_super_chunk(lane, chunk, next, n == remaining.size() || tail == PrefillTail::Capture, sink,
                    observer, {});
    // The step's device work is complete on return, so its caller's wall clock measures it and a
    // lane closed afterwards has nothing in flight; the next super-chunk's n-gram rows keep
    // gathering on the feeder's thread.
    CUDA_CHECK(cudaStreamSynchronize(stream_));
    return static_cast<std::int32_t>(n);
}

void Forward::prefill(std::int32_t lane, std::span<const std::int32_t> tokens,
                      const LogitsSink& sink, const ForwardObserver* observer) {
    if (tokens.empty()) { throw std::invalid_argument("qwen3_8_flash_next Forward: no tokens"); }
    while (!tokens.empty()) {
        const std::int32_t consumed = prefill_step(lane, tokens, sink, observer);
        tokens                      = tokens.subspan(static_cast<std::size_t>(consumed));
    }
}

std::vector<float> Forward::score(std::int32_t lane, std::span<const std::int32_t> tokens) {
    require_resolved();
    pending_verify_width_ = 0;
    feature_columns_      = 0;
    if (tokens.size() < 2) {
        throw std::invalid_argument("qwen3_8_flash_next Forward: score needs two tokens");
    }
    const std::span<const std::int32_t> inputs  = tokens.first(tokens.size() - 1);
    const std::span<const std::int32_t> targets = tokens.subspan(1);
    if (static_cast<std::int64_t>(inputs.size()) >
        state_.reserved_tokens(lane) - state_.committed(lane)) {
        throw std::out_of_range("qwen3_8_flash_next Forward: the scored tokens exceed the lane's "
                                "reservation");
    }
    check_tokens(inputs);
    // A target is a log-probability row, so it must lie in the token domain.
    if (std::ranges::any_of(targets, [](std::int32_t t) { return t < 0 || t >= kTokenDomain; })) {
        throw std::invalid_argument("qwen3_8_flash_next Forward: a scored token is outside the "
                                    "token domain");
    }
    const auto span = static_cast<std::size_t>(options_.super_chunk_tokens);
    std::vector<float> out(inputs.size());
    feeder_.cancel();
    feeder_.begin(lane, inputs.first(std::min(span, inputs.size())), options_.max_chunk_tokens);
    for (std::size_t begin = 0; begin < inputs.size(); begin += span) {
        const std::size_t width      = std::min(span, inputs.size() - begin);
        const std::size_t next_begin = begin + width;
        const std::size_t next_width =
            next_begin < inputs.size() ? std::min(span, inputs.size() - next_begin) : 0;
        run_super_chunk(lane, inputs.subspan(begin, width), inputs.subspan(next_begin, next_width),
                        next_width == 0, LogitsSink{}, nullptr, targets.subspan(begin, width));
        // A pageable destination: the copy has completed when the call returns.
        CUDA_CHECK(cudaMemcpyAsync(out.data() + begin, chunk_.score_logprobs.data,
                                   width * sizeof(float), cudaMemcpyDeviceToHost, stream_));
        CUDA_CHECK(cudaStreamSynchronize(stream_));
    }
    return out;
}

std::vector<Forward::SubChunk> Forward::sub_chunks(std::int32_t lane, std::int32_t tokens,
                                                   std::int32_t first_position) const {
    std::vector<SubChunk> chunks;
    chunks.reserve(static_cast<std::size_t>((tokens + options_.max_chunk_tokens - 1) /
                                            options_.max_chunk_tokens));
    for (std::int32_t offset = 0; offset < tokens; offset += options_.max_chunk_tokens) {
        const auto index         = static_cast<std::int32_t>(chunks.size());
        const ChunkInputs& block = chunk_.input_blocks.at(static_cast<std::size_t>(index));
        const std::int32_t width = std::min(options_.max_chunk_tokens, tokens - offset);
        const std::int32_t first = first_position + offset;
        // One row: the [W,3] axis-major positions are also the QSA Ops' [W,3,1].
        chunks.push_back(SubChunk{
            .index          = index,
            .offset         = offset,
            .width          = width,
            .rows           = 1,
            .first_position = first,
            .max_visible    = first + width,
            .lane           = lane,
            .token_ids      = Tensor(block.token_ids.data, DType::I32, {width}),
            .rope_positions = Tensor(block.positions.data, DType::I32, {width, kMropeAxes}),
            .slots          = Tensor(block.slots.data, DType::I32, {width, 1}),
            .qsa_positions  = Tensor(block.positions.data, DType::I32, {width, kMropeAxes, 1}),
            .lanes          = state_.lane_id(lane),
        });
    }
    return chunks;
}

Forward::SubChunk Forward::decode_batch(std::int32_t rows) const {
    const Tensor& frame = chunk_.ingress;
    return SubChunk{
        .index          = 0,
        .offset         = 0,
        .width          = 1,
        .rows           = rows,
        .first_position = 0,
        .max_visible    = state_.lane_capacity(),
        .lane           = -1,
        .token_ids      = at(frame, offsetof(DecodeIngress, tokens), DType::I32, {rows}),
        .rope_positions =
            at(frame, offsetof(DecodeIngress, rope_positions), DType::I32, {rows, kMropeAxes}),
        .slots = at(frame, offsetof(DecodeIngress, slots), DType::I32, {1, rows}),
        .qsa_positions =
            at(frame, offsetof(DecodeIngress, qsa_positions), DType::I32, {1, kMropeAxes, rows}),
        .lanes = at(frame, offsetof(DecodeIngress, lanes), DType::I32, {rows}),
    };
}

void Forward::write_inputs(std::span<const SubChunk> chunks, std::span<const std::int32_t> tokens) {
    auto* host        = static_cast<std::byte*>(input_staging_.data());
    const auto* base  = static_cast<const std::byte*>(activations_.base());
    const auto offset = [base](const Tensor& tensor) {
        return static_cast<std::size_t>(static_cast<const std::byte*>(tensor.data) - base);
    };
    for (const SubChunk& chunk : chunks) {
        auto* ids       = reinterpret_cast<std::int32_t*>(host + offset(chunk.token_ids));
        auto* slots     = reinterpret_cast<std::int32_t*>(host + offset(chunk.slots));
        auto* positions = reinterpret_cast<std::int32_t*>(host + offset(chunk.rope_positions));
        const auto t    = static_cast<std::size_t>(chunk.width);
        std::memcpy(ids, tokens.data() + chunk.offset, t * sizeof(std::int32_t));
        // Text tokens carry their position on all three MRoPE axes; positions are axis-major [T,3].
        for (std::size_t i = 0; i < t; ++i) {
            const auto position = static_cast<std::int32_t>(chunk.first_position + i);
            slots[i]            = position;
            for (std::size_t axis = 0; axis < static_cast<std::size_t>(kMropeAxes); ++axis) {
                positions[axis * t + i] = position;
            }
        }
    }
}

void Forward::upload_inputs(std::size_t blocks) {
    const Tensor& last = chunk_.input_blocks.at(blocks - 1).positions;
    const auto bytes =
        static_cast<std::size_t>(static_cast<const std::byte*>(last.data) -
                                 static_cast<const std::byte*>(activations_.base())) +
        last.bytes();
    CUDA_CHECK(cudaMemcpyAsync(activations_.base(), input_staging_.data(), bytes,
                               cudaMemcpyHostToDevice, stream_));
}

void Forward::run_super_chunk(std::int32_t lane, std::span<const std::int32_t> tokens,
                              std::span<const std::int32_t> next_tokens, bool final_super_chunk,
                              const LogitsSink& sink, const ForwardObserver* observer,
                              std::span<const std::int32_t> score_targets) {
    const auto t                       = static_cast<std::int32_t>(tokens.size());
    const std::int32_t first_position  = state_.committed(lane);
    const std::vector<SubChunk> chunks = sub_chunks(lane, t, first_position);
    // The previous super-chunk's upload must have read the staging before it is rewritten.
    CUDA_CHECK(cudaEventSynchronize(inputs_uploaded_.get()));
    write_inputs(chunks, tokens);
    upload_inputs(chunks.size());
    CUDA_CHECK(cudaEventRecord(inputs_uploaded_.get(), stream_));

    enqueue_layers(chunks, windowed(t), RowSource::Feeder, next_tokens, observer);
    state_.advance(lane, t);

    if (!score_targets.empty()) {
        // A pageable source: the copy has read it when the call returns.
        CUDA_CHECK(cudaMemcpyAsync(chunk_.score_targets.data, score_targets.data(),
                                   score_targets.size_bytes(), cudaMemcpyHostToDevice, stream_));
        const std::int32_t group = logit_columns(options_);
        for (std::int32_t first = 0; first < t; first += group) {
            const std::int32_t width = std::min(group, t - first);
            head(first, width);
            const Tensor logits(chunk_.logits.data, DType::BF16, {kVocabRows, width, 1});
            const Tensor targets =
                at(chunk_.score_targets, static_cast<std::size_t>(first) * sizeof(std::int32_t),
                   DType::I32, {width, 1});
            Tensor logprobs =
                at(chunk_.score_logprobs, static_cast<std::size_t>(first) * sizeof(float),
                   DType::FP32, {width, 1});
            Tensor top_ids(chunk_.score_top_ids.data, DType::I32, {1, width, 1});
            Tensor top_logprobs(chunk_.score_top_logprobs.data, DType::FP32, {1, width, 1});
            const Tensor enabled(chunk_.row_enabled.data, DType::I32, {1});
            ops::token_logprobs(logits, targets, enabled, nullptr, nullptr, kTokenDomain, logprobs,
                                top_ids, top_logprobs, stream_);
        }
        return;
    }
    if (options_.logits == LogitColumns::All) {
        for (const SubChunk& chunk : chunks) {
            head(chunk.offset, chunk.width);
            if (sink) {
                sink(columns(chunk_.logits, kVocabRows, chunk.width), chunk.first_position);
            }
        }
    } else if (final_super_chunk) {
        head(t - 1, 1);
        const Tensor tail = state_.tail_logits(lane);
        CUDA_CHECK(cudaMemcpyAsync(tail.data, chunk_.logits.data, tail.bytes(),
                                   cudaMemcpyDeviceToDevice, stream_));
        if (sink) { sink(columns(chunk_.logits, kVocabRows, 1), first_position + t - 1); }
    }
}

SampledRows Forward::sample_retained(std::int32_t lane, const ops::SamplingConfig& config) {
    require_resolved();
    pending_verify_width_ = 0;
    const Tensor tail     = state_.tail_logits(lane);
    CUDA_CHECK(cudaMemcpyAsync(chunk_.logits.data, tail.data, tail.bytes(),
                               cudaMemcpyDeviceToDevice, stream_));
    return sample_last(config, state_.committed(lane) - 1);
}

Tensor Forward::residual_columns(const SubChunk& chunk) const {
    constexpr auto kColumnBytes =
        static_cast<std::size_t>(kHidden) * ops::kHyperConnectionStreams * sizeof(std::uint16_t);
    return Tensor(static_cast<std::byte*>(chunk_.residual.data) +
                      static_cast<std::size_t>(chunk.offset) * kColumnBytes,
                  DType::BF16, {kHidden, ops::kHyperConnectionStreams, chunk.tokens()});
}

void Forward::enqueue_layers(std::span<const SubChunk> chunks, bool streamed, RowSource rows,
                             std::span<const std::int32_t> next_tokens,
                             const ForwardObserver* observer) {
    for (const SubChunk& chunk : chunks) {
        Tensor embedded = columns(chunk_.block_input, kHidden, chunk.tokens());
        ops::embedding(chunk.token_ids, model_.embedding, embedded, stream_);
        Tensor residual = residual_columns(chunk);
        ops::hyper_connection_broadcast(embedded, residual, stream_);
    }
    if (streamed) {
        // The windows are borrowed cache slots: evicted here and returned empty after the last
        // layer. The first two layers' windows start copying behind all earlier work, including
        // the borrow and the last admit that set the residency this super-chunk streams against.
        borrow_windows();
        CUDA_CHECK(cudaEventRecord(window_start_.get(), stream_));
        CUDA_CHECK(cudaStreamWaitEvent(fill_stream_.get(), window_start_.get(), 0));
        enqueue_window(0, 0);
        enqueue_window(1, 1);
    }
    try {
        enqueue_layer_loop(chunks, streamed, rows, next_tokens, observer);
    } catch (...) {
        // A host-side failure (an n-gram read, say) leaves the lane closable and the Program
        // serving, so the borrowed slots go back to the cache behind every window copy already
        // enqueued. A CUDA failure here replaces the original error; the Forward state is then
        // unspecified either way.
        if (streamed) {
            for (const CudaEvent& ready : window_ready_) {
                CUDA_CHECK(cudaStreamWaitEvent(stream_, ready.get(), 0));
            }
            return_windows();
        }
        throw;
    }
    // Layer 47's moe_experts, the windows' last reader, precedes the return in stream order.
    if (streamed) { return_windows(); }
}

void Forward::enqueue_layer_loop(std::span<const SubChunk> chunks, bool streamed, RowSource rows,
                                 std::span<const std::int32_t> next_tokens,
                                 const ForwardObserver* observer, std::int32_t first_layer,
                                 std::int32_t past_layer) {
    // Staging keeps one span and one stage column per layer parity, so a staged round is one batch.
    if (options_.expert_staging == ExpertStaging::Enabled && chunks.size() > 1 &&
        chunks.front().lane < 0) {
        throw std::logic_error("qwen3_8_flash_next Forward: a round runs as one batch");
    }
    for (std::int32_t layer = first_layer; layer < past_layer; ++layer) {
        const LayerWeights& weights = model_.layers.at(static_cast<std::size_t>(layer));
        for (const SubChunk& chunk : chunks) {
            const std::int32_t t = chunk.tokens();
            if (layer == kPleLayer) {
                if (rows == RowSource::Feeder) {
                    // Blocks the host until this part is gathered; the device meanwhile runs the
                    // layers already enqueued (layer 0 and earlier sub-chunks of layer 1).
                    Tensor part(static_cast<std::byte*>(chunk_.ple_rows.data) +
                                    static_cast<std::size_t>(chunk.offset) * ops::kPleRowsPerToken *
                                        ops::kPleInt8RowBytes,
                                DType::U8, {ops::kPleInt8RowBytes, ops::kPleRowsPerToken, t});
                    feeder_.upload(part, chunk.index, stream_);
                    if (&chunk == &chunks.back() && !next_tokens.empty()) {
                        feeder_.begin(chunk.lane, next_tokens, options_.max_chunk_tokens);
                    }
                }
                ple(chunk);
            }
            Tensor residual    = residual_columns(chunk);
            Tensor block_input = columns(chunk_.block_input, kHidden, t);
            Tensor write_scale = columns(chunk_.write_scale, ops::kHyperConnectionStreams, t);
            ops::hyper_connection_read(residual, weights.mixer_connection, block_input, write_scale,
                                       work_, stream_);
            if (is_qsa_layer(layer)) {
                qsa_mixer(layer, chunk);
            } else {
                gdn_mixer(layer, chunk);
            }
            const Tensor block_output = columns(chunk_.block_output, kHidden, t);
            if (observer != nullptr && observer->after_mixer) {
                observer->after_mixer(layer, block_input, block_output, chunk.first_position);
            }
            ops::hyper_connection_inject(residual, block_output, write_scale, residual, stream_);
            moe(layer, chunk, streamed, &chunk == &chunks.back(), observer);
            capture_features(layer, chunk);
            if (observer != nullptr && observer->after_layer) {
                observer->after_layer(layer, residual, chunk.first_position);
            }
        }
    }
}

void Forward::enqueue_sampling(std::int32_t rows, std::int32_t purpose) {
    const Tensor logits = columns(chunk_.logits, kVocabRows, rows);
    const Tensor& out   = chunk_.egress;
    Tensor sampled      = at(out, offsetof(SamplingEgress, tokens), DType::I32, {rows});
    Tensor logprobs     = at(out, offsetof(SamplingEgress, logprobs), DType::FP32, {1, rows});
    Tensor top_ids =
        at(out, offsetof(SamplingEgress, top_ids), DType::I32, {kReportedAlternatives, 1, rows});
    Tensor top_logprobs = at(out, offsetof(SamplingEgress, top_logprobs), DType::FP32,
                             {kReportedAlternatives, 1, rows});
    const auto* configs = reinterpret_cast<const ops::SamplingConfig*>(
        static_cast<const std::byte*>(chunk_.ingress.data) + offsetof(DecodeIngress, sampling));
    const Tensor positions = at(chunk_.ingress, offsetof(DecodeIngress, slots), DType::I32, {rows});
    ops::sample(logits, sampled, kTokenDomain, configs, positions, purpose, work_, stream_);
    // The log probabilities of the unmodified distribution: no temperature, penalty or mask.
    const Tensor enabled =
        at(chunk_.ingress, offsetof(DecodeIngress, logprobs), DType::I32, {rows});
    ops::token_logprobs(logits.view({kVocabRows, 1, rows}), sampled.view({1, rows}), enabled,
                        nullptr, nullptr, kTokenDomain, logprobs, top_ids, top_logprobs, stream_);
    CUDA_CHECK(cudaMemcpyAsync(egress_.data(), out.data, sizeof(SamplingEgress),
                               cudaMemcpyDeviceToHost, stream_));
}

SampledRows Forward::sampled_rows(std::int32_t rows) const {
    const auto& egress = *static_cast<const SamplingEgress*>(egress_.data());
    const auto count   = static_cast<std::size_t>(rows);
    const auto ranked  = count * static_cast<std::size_t>(kReportedAlternatives);
    return SampledRows{
        .tokens       = std::span<const std::int32_t>(egress.tokens.data(), count),
        .logprobs     = std::span<const float>(egress.logprobs.data(), count),
        .top_ids      = std::span<const std::int32_t>(egress.top_ids.data(), ranked),
        .top_logprobs = std::span<const float>(egress.top_logprobs.data(), ranked),
    };
}

SampledRows Forward::sample_last(const ops::SamplingConfig& config, std::int32_t position) {
    require_resolved();
    if (options_.logits != LogitColumns::Last) {
        throw std::logic_error("qwen3_8_flash_next Forward: sample_last needs LogitColumns::Last");
    }
    // Every earlier reader of the pinned ingress (a decode round or sample_last) synchronized.
    DecodeIngress& ingress = ingress_host();
    ingress.sampling[0]    = config;
    ingress.slots[0]       = position;
    ingress.logprobs[0]    = 1;
    CUDA_CHECK(cudaMemcpyAsync(chunk_.ingress.data, ingress_staging_.data(), sizeof(DecodeIngress),
                               cudaMemcpyHostToDevice, stream_));
    enqueue_sampling(1, ops::kSamplePurposePrefill);
    CUDA_CHECK(cudaStreamSynchronize(stream_));
    return sampled_rows(1);
}

void Forward::enqueue_decode_a(std::int32_t rows, const ForwardObserver* observer) {
    feature_offset_  = 0;
    feature_columns_ = rows;
    CUDA_CHECK(cudaMemcpyAsync(chunk_.ingress.data, ingress_staging_.data(), sizeof(DecodeIngress),
                               cudaMemcpyHostToDevice, stream_));
    const std::array batch{decode_batch(rows)};
    const SubChunk& chunk = batch.front();
    Tensor embedded       = columns(chunk_.block_input, kHidden, rows);
    ops::embedding(chunk.token_ids, model_.embedding, embedded, stream_);
    Tensor residual = residual_columns(chunk);
    ops::hyper_connection_broadcast(embedded, residual, stream_);
    enqueue_layer_loop(batch, false, RowSource::Resident, {}, observer, 0, kPleLayer);
}

void Forward::enqueue_decode_b(std::int32_t rows, const ForwardObserver* observer) {
    CUDA_CHECK(cudaMemcpyAsync(chunk_.ple_rows.data, decode_rows_.data(),
                               static_cast<std::size_t>(rows) * ops::kPleRowsPerToken *
                                   ops::kPleInt8RowBytes,
                               cudaMemcpyHostToDevice, stream_));
    const std::array<SubChunk, 1> batch{decode_batch(rows)};
    enqueue_layer_loop(batch, false, RowSource::Resident, {}, observer, kPleLayer, kLayers);
    head(0, rows);
    enqueue_sampling(rows, ops::kSamplePurposeDecode);
}

void Forward::capture_decode_graphs() {
    // Every address (staging, frames, arenas, tables) is fixed for the Forward's lifetime, and the
    // workspace scopes allocate the same offsets on every replay; one graph pair per batch size.
    const auto batches = static_cast<std::size_t>(state_.lanes());
    decode_a_definitions_.resize(batches);
    decode_b_definitions_.resize(batches);
    decode_a_graphs_.resize(batches);
    decode_b_graphs_.resize(batches);
    for (std::size_t b = 0; b < batches; ++b) {
        const auto rows = static_cast<std::int32_t>(b + 1);
        decode_a_definitions_[b].capture(stream_,
                                         [this, rows] { enqueue_decode_a(rows, nullptr); });
        decode_b_definitions_[b].capture(stream_,
                                         [this, rows] { enqueue_decode_b(rows, nullptr); });
        decode_a_graphs_[b].instantiate(decode_a_definitions_[b]);
        decode_b_graphs_[b].instantiate(decode_b_definitions_[b]);
        decode_a_graphs_[b].upload(stream_);
        decode_b_graphs_[b].upload(stream_);
    }
    CUDA_CHECK(cudaStreamSynchronize(stream_));
}

SampledRows Forward::decode(std::span<const std::int32_t> lanes,
                            std::span<const std::int32_t> tokens,
                            std::span<const ops::SamplingConfig> configs, RoundLogprobs logprobs,
                            DecodeExecution execution, const ForwardObserver* observer) {
    require_resolved();
    pending_verify_width_  = 0;
    const std::size_t rows = lanes.size();
    if (rows < 1 || rows > static_cast<std::size_t>(state_.lanes()) || tokens.size() != rows ||
        configs.size() != rows) {
        throw std::invalid_argument("qwen3_8_flash_next Forward: a decode round needs 1 to " +
                                    std::to_string(state_.lanes()) +
                                    " rows with one token and configuration each");
    }
    if (execution == DecodeExecution::Graph && observer != nullptr) {
        throw std::invalid_argument("qwen3_8_flash_next Forward: a graph replay takes no "
                                    "observer");
    }
    if (execution == DecodeExecution::Graph && decode_a_graphs_.empty()) {
        throw std::logic_error("qwen3_8_flash_next Forward: decode graphs need "
                               "LogitColumns::Last");
    }
    check_tokens(tokens);
    DecodeIngress& ingress = ingress_host();
    for (std::size_t i = 0; i < rows; ++i) {
        const std::int32_t lane = lanes[i];
        const auto earlier      = lanes.first(i);
        if (std::find(earlier.begin(), earlier.end(), lane) != earlier.end()) {
            throw std::invalid_argument("Flash-Next decode: duplicate lane");
        }
        if (lane == feeder_.pending_lane()) {
            throw std::logic_error("Flash-Next decode: lane has a pending n-gram gather");
        }
        if (state_.committed(lane) >= state_.reserved_tokens(lane)) {
            throw std::out_of_range("qwen3_8_flash_next Forward: lane " + std::to_string(lane) +
                                    " has no reserved slot left");
        }
    }
    for (std::size_t i = 0; i < rows; ++i) {
        const std::int32_t position = state_.committed(lanes[i]);
        ingress.tokens[i]           = tokens[i];
        ingress.lanes[i]            = lanes[i];
        ingress.slots[i]            = position;
        ingress.sampling[i]         = configs[i];
        ingress.logprobs[i]         = logprobs == RoundLogprobs::Report ? 1 : 0;
        for (std::size_t axis = 0; axis < static_cast<std::size_t>(kMropeAxes); ++axis) {
            ingress.qsa_positions[static_cast<std::size_t>(kMropeAxes) * i + axis] = position;
            ingress.rope_positions[axis * rows + i]                                = position;
        }
    }

    const auto batch = static_cast<std::int32_t>(rows);
    feature_offset_  = 0;
    feature_columns_ = batch;
    try {
        CUDA_CHECK(cudaEventRecord(replay_start_.get(), stream_));
        if (execution == DecodeExecution::Graph) {
            decode_a_graphs_[rows - 1].launch(stream_);
        } else {
            enqueue_decode_a(batch, observer);
        }
        const auto gather_start = std::chrono::steady_clock::now();
        feeder_.gather_now(
            lanes, tokens,
            std::span<std::uint8_t>(static_cast<std::uint8_t*>(decode_rows_.data()),
                                    rows * static_cast<std::size_t>(kNgramTokenBytes)));
        last_gather_seconds_ =
            std::chrono::duration<double>(std::chrono::steady_clock::now() - gather_start).count();
        if (execution == DecodeExecution::Graph) {
            decode_b_graphs_[rows - 1].launch(stream_);
        } else {
            enqueue_decode_b(batch, observer);
        }
        for (std::size_t row = 0; row < rows; ++row) {
            const Tensor tail = state_.tail_logits(lanes[row]);
            const auto* source =
                static_cast<const std::byte*>(chunk_.logits.data) + row * tail.bytes();
            CUDA_CHECK(cudaMemcpyAsync(tail.data, source, tail.bytes(), cudaMemcpyDeviceToDevice,
                                       stream_));
        }
        CUDA_CHECK(cudaEventRecord(replay_stop_.get(), stream_));
        CUDA_CHECK(cudaEventSynchronize(replay_stop_.get()));
        CUDA_CHECK(cudaEventElapsedTime(&last_replay_ms_, replay_start_.get(), replay_stop_.get()));
    } catch (...) {
        // A failed row read can follow layer zero's launch; drain its ingress and workspace
        // readers before a caller closes or reuses the lane and pinned frames.
        CUDA_CHECK(cudaStreamSynchronize(stream_));
        CUDA_CHECK(cudaStreamSynchronize(fill_stream_.get()));
        throw;
    }
    for (const std::int32_t lane : lanes) { state_.advance(lane, 1); }
    return sampled_rows(batch);
}

void Forward::ple(const SubChunk& chunk) {
    const std::int32_t width = chunk.width;
    const std::int32_t rows  = chunk.rows;
    const std::int32_t t     = chunk.tokens();
    auto scope               = work_.scope();
    PleScratch scratch       = allocate_ple(work_, t);
    const Tensor ple_rows(static_cast<std::byte*>(chunk_.ple_rows.data) +
                              static_cast<std::size_t>(chunk.offset) * ops::kPleRowsPerToken *
                                  ops::kPleInt8RowBytes,
                          DType::U8, {ops::kPleInt8RowBytes, ops::kPleRowsPerToken, t});
    ops::ple_decode_rows(ple_rows, scratch.embedding, stream_);
    ops::bf16_projection(scratch.embedding, model_.ple.key_projection, scratch.key, stream_);
    ops::bf16_projection(scratch.embedding, model_.ple.value_projection, scratch.value, stream_);
    // B rows of W columns over the rows' lanes' history slots.
    Tensor residual =
        residual_columns(chunk).view({kHidden, ops::kHyperConnectionStreams, width, rows});
    const Tensor key   = scratch.key.view({kPleKeyRows, width, rows});
    const Tensor value = scratch.value.view({kPleValueRows, width, rows});
    if (chunk.state_mode == StateMode::Record) {
        ops::ple_inject_snapshot(residual, key, value, model_.ple.inject, verify_.ple_states,
                                 chunk.valid_columns, chunk.lanes, chunk.ple_snapshot_slots, work_,
                                 stream_);
    } else {
        const Tensor every_column;
        ops::ple_inject(residual, key, value, model_.ple.inject, state_.ple_history(), every_column,
                        chunk.lanes, work_, stream_);
    }
}

void Forward::gdn_mixer(std::int32_t layer, const SubChunk& chunk) {
    const std::int32_t tokens      = chunk.tokens();
    const std::int32_t index       = gdn_index(layer);
    const GdnLayerWeights& weights = model_.gdn.at(static_cast<std::size_t>(index));
    auto scope                     = work_.scope();
    GdnScratch s                   = allocate_gdn(work_, tokens);
    if (chunk.lane < 0) {
        // gdn_input_proj_conv_snapshot rejects outputs anywhere in the workspace arena, so a
        // decode round's q, k, v and z live in the activations.
        s.query = columns(chunk_.decode_gdn_query, kGdnKeyDim, tokens);
        s.key   = columns(chunk_.decode_gdn_key, kGdnKeyDim, tokens);
        s.value = columns(chunk_.decode_gdn_value, kGdnValueDim, tokens);
        s.z = Tensor(chunk_.decode_gdn_z.data, DType::BF16, {kGdnHeadDim, kGdnValueHeads, tokens});
    }
    const Tensor x = columns(chunk_.block_input, kHidden, tokens);
    Tensor z       = s.z.view({kGdnValueDim, tokens});

    if (chunk.lane >= 0) {
        // Prefill of one lane: the projection over every column, then the convolution and the
        // recurrence advancing the lane's slot in place.
        ops::gdn_input_proj(x, weights.query_key_value_z, s.qkv, z, stream_);
        Tensor convolution_state = state_.gdn_convolution(index, chunk.lane);
        ops::causal_conv1d_silu_split(s.qkv, weights.convolution, convolution_state, s.query, s.key,
                                      s.value, stream_);
    } else if (chunk.state_mode == StateMode::Record) {
        // Every verify width, including a K0 root (W=1), reads the canonical pool and leaves it
        // untouched; the records carry the round to the commit fold. At W=1 the record route runs
        // the width-one snapshot's projection and convolution, so a K0 root matches decode.
        const auto width   = chunk.width;
        const auto rows    = chunk.rows;
        const auto record  = verify_records(width).layer(index, rows);
        Tensor convolution = record.conv;
        Tensor query       = s.query.view({kGdnKeyDim, width, rows});
        Tensor key         = s.key.view({kGdnKeyDim, width, rows});
        Tensor value       = s.value.view({kGdnValueDim, width, rows});
        Tensor z_rows      = s.z.view({kGdnValueDim, width, rows});
        ops::gdn_input_proj_conv_record(x.view({kHidden, width, rows}), weights.query_key_value_z,
                                        weights.convolution, state_.gdn_convolution_pool(index),
                                        chunk.valid_columns, chunk.lanes, convolution, query, key,
                                        value, z_rows, work_, stream_);
    } else {
        // Decode: B rows of one column, each reading and rewriting its lane's convolution slot
        // (initial slot = snapshot slot = lane, W = 1).
        Tensor convolution_states = state_.gdn_convolution_pool(index);
        Tensor query              = s.query.view({kGdnKeyDim, 1, tokens});
        Tensor key                = s.key.view({kGdnKeyDim, 1, tokens});
        Tensor value              = s.value.view({kGdnValueDim, 1, tokens});
        Tensor z_rows             = s.z.view({kGdnValueDim, 1, tokens});
        const Tensor every_column;
        ops::gdn_input_proj_conv_snapshot(x.view({kHidden, 1, tokens}), weights.query_key_value_z,
                                          weights.convolution, convolution_states, every_column,
                                          chunk.lanes, chunk.lanes, query, key, value, z_rows,
                                          work_, stream_);
    }
    ops::bf16_projection(x, weights.gate_projection, s.gates, stream_);
    ops::gdn_gating_packed(s.gates, weights.a_log, weights.dt_bias, s.decay, s.beta, stream_);

    // q and k are L2-normalized inside the recurrence (epsilon 1e-6 inside the root); the
    // 1/sqrt(128) query scale is the Op's `scale`.
    if (chunk.lane >= 0) {
        const Tensor query = s.query.view({kGdnHeadDim, kGdnKeyHeads, tokens});
        const Tensor key   = s.key.view({kGdnHeadDim, kGdnKeyHeads, tokens});
        const Tensor value = s.value.view({kGdnHeadDim, kGdnValueHeads, tokens});
        Tensor recurrent   = state_.gdn_recurrent(index, chunk.lane);
        ops::gated_delta_net(query, key, value, s.decay, s.beta, kGdnScale,
                             /*normalize_qk=*/true, work_, recurrent, s.core, stream_);
    } else if (chunk.state_mode == StateMode::Record) {
        const auto width  = chunk.width;
        const auto rows   = chunk.rows;
        const auto record = verify_records(width).layer(index, rows);
        Tensor keys       = record.key;
        Tensor values     = record.value;
        Tensor gates      = record.gate;
        Tensor core       = s.core.view({kGdnHeadDim, kGdnValueHeads, width, rows});
        ops::gated_delta_net_replay_record(s.query.view({kGdnHeadDim, kGdnKeyHeads, width, rows}),
                                           s.key.view({kGdnHeadDim, kGdnKeyHeads, width, rows}),
                                           s.value.view({kGdnHeadDim, kGdnValueHeads, width, rows}),
                                           s.decay.view({kGdnValueHeads, width, rows}),
                                           s.beta.view({kGdnValueHeads, width, rows}), kGdnScale,
                                           state_.gdn_recurrent_pool(index), chunk.valid_columns,
                                           chunk.lanes, keys, values, gates, core, stream_);
    } else {
        const Tensor query = s.query.view({kGdnHeadDim, kGdnKeyHeads, 1, tokens});
        const Tensor key   = s.key.view({kGdnHeadDim, kGdnKeyHeads, 1, tokens});
        const Tensor value = s.value.view({kGdnHeadDim, kGdnValueHeads, 1, tokens});
        const Tensor decay = s.decay.view({kGdnValueHeads, 1, tokens});
        const Tensor beta  = s.beta.view({kGdnValueHeads, 1, tokens});
        Tensor recurrent   = state_.gdn_recurrent_pool(index);
        Tensor core        = s.core.view({kGdnHeadDim, kGdnValueHeads, 1, tokens});
        const Tensor every_column;
        ops::gated_delta_net_snapshot(query, key, value, decay, beta, kGdnScale,
                                      /*normalize_qk=*/true, recurrent, every_column, chunk.lanes,
                                      chunk.lanes, core, stream_);
    }
    ops::gated_rmsnorm(s.core, weights.norm, s.z, ops::RmsNormGate::Sigmoid, kRmsEpsilon, s.normed,
                       stream_);

    const Tensor normed = s.normed.view({kMixerOutputColumns, tokens});
    Tensor y            = columns(chunk_.block_output, kHidden, tokens);
    ops::linear(normed, weights.output, y, stream_);
}

void Forward::qsa_mixer(std::int32_t layer, const SubChunk& chunk) {
    const std::int32_t width       = chunk.width;
    const std::int32_t rows        = chunk.rows;
    const std::int32_t tokens      = chunk.tokens();
    const std::int32_t index       = qsa_index(layer);
    const QsaLayerWeights& weights = model_.qsa.at(static_cast<std::size_t>(index));
    ops::QsaPagedState& state      = state_.qsa(index);
    auto scope                     = work_.scope();
    QsaScratch s   = allocate_qsa(work_, tokens, selected_slots(options_.select_budget));
    const Tensor x = columns(chunk_.block_input, kHidden, tokens);

    ops::attn_input_proj(x, weights.query_key_gate_value, s.query, s.gate, s.key, s.value, stream_);
    const Tensor query = s.query.view({ops::kQsaHeadDim, ops::kQsaQueryHeads, tokens});
    const Tensor key   = s.key.view({ops::kQsaHeadDim, ops::kQsaKvHeads, tokens});
    ops::rmsnorm(query, weights.query_norm, kRmsEpsilon, /*unit_offset=*/true, s.query_rotated,
                 stream_);
    ops::rmsnorm(key, weights.key_norm, kRmsEpsilon, /*unit_offset=*/true, s.key_rotated, stream_);
    ops::rope(chunk.rope_positions, kRotaryDim, kRopeTheta, s.query_rotated, s.key_rotated,
              stream_);

    ops::bf16_projection(x, weights.index_query, s.index_query, stream_);
    ops::bf16_projection(x, weights.index_key, s.index_key, stream_);

    // The lane-batched Ops take every [.., T] tensor as [.., W, B].
    const Tensor rotated_key =
        s.key_rotated.view({ops::kQsaHeadDim, ops::kQsaKvHeads, width, rows});
    const Tensor value     = s.value.view({ops::kQsaHeadDim, ops::kQsaKvHeads, width, rows});
    const Tensor index_key = s.index_key.view({ops::kQsaIndexHeadDim, width, rows});
    ops::qsa_state_append(rotated_key, value, index_key, chunk.qsa_positions, chunk.slots,
                          chunk.lanes, weights.index_key_norm, state, stream_);
    const Tensor index_query =
        s.index_query.view({ops::kQsaIndexHeadDim, ops::kQsaIndexQueryHeads, width, rows});
    Tensor selected_ids =
        s.selected_ids.view({selected_slots(options_.select_budget), width, rows});
    Tensor selected_count = s.selected_count.view({width, rows});
    ops::qsa_index_select(index_query, state, chunk.slots, chunk.lanes, weights.index_query_norm,
                          options_.select_budget, chunk.max_visible, selected_ids, selected_count,
                          work_, stream_);
    const Tensor rotated_query =
        s.query_rotated.view({ops::kQsaHeadDim, ops::kQsaQueryHeads, width, rows});
    Tensor attention = s.attention.view({ops::kQsaHeadDim, ops::kQsaQueryHeads, width, rows});
    ops::qsa_selected_attention(rotated_query, selected_ids, selected_count, chunk.lanes, state,
                                attention, work_, stream_);

    Tensor attention_rows = s.attention.view({kQsaQueryDim, tokens});
    ops::sigmoid_mul(s.gate, attention_rows, stream_);
    Tensor y = columns(chunk_.block_output, kHidden, tokens);
    ops::linear(attention_rows, weights.output, y, stream_);
}

bool Forward::windowed(std::int32_t tokens) const noexcept {
    return window_slots_ > 0 && tokens >= options_.window_min_tokens;
}

std::size_t Forward::window_bytes() const noexcept {
    return 2 * static_cast<std::size_t>(window_slots_) *
           cache_.pools[static_cast<std::size_t>(kWindowCodec)].record_bytes;
}

void Forward::borrow_windows() {
    const ops::MoeExpertSlotPool& pool = cache_.pools[static_cast<std::size_t>(kWindowCodec)];
    const auto slots         = static_cast<std::int32_t>(pool.slots.bytes / pool.record_bytes);
    const std::int32_t count = 2 * window_slots_;
    // The hand is read once the stream is idle: the slots just ahead of it are CLOCK's next
    // victims, the coldest by its order. The windows take fixed slots only (slots = F counts
    // them), so a run that would pass slot F - 1, or a hand among the flex slots [F, N), starts at
    // slot 0 instead: the first fixed slots the hand reaches after it wraps.
    std::int32_t hand = 0;
    CUDA_CHECK(cudaMemcpyAsync(&hand,
                               static_cast<const std::int32_t*>(cache_.clock_hands.data) +
                                   static_cast<std::size_t>(kWindowCodec),
                               sizeof(hand), cudaMemcpyDeviceToHost, stream_));
    CUDA_CHECK(cudaStreamSynchronize(stream_));
    borrowed_first_ = hand <= slots - count ? hand : 0;
    ops::moe_expert_cache_borrow(expert_table_, host_table_, cache_, kWindowCodec,
                                 ops::MoeCacheSlotRegion::Fixed, borrowed_first_, count, stream_);
}

void Forward::return_windows() {
    ops::moe_expert_cache_return(cache_, kWindowCodec, ops::MoeCacheSlotRegion::Fixed,
                                 borrowed_first_, 2 * window_slots_, stream_);
    borrowed_first_ = -1;
}

void Forward::enqueue_window(std::int32_t layer, std::int32_t buffer) {
    const auto b                       = static_cast<std::size_t>(buffer);
    const Tensor column                = chunk_.window_columns.slice(1, buffer, 1); // I64 [512,1]
    Tensor placements                  = column.view({ops::kMoeExperts});
    const ops::MoeExpertSlotPool& pool = cache_.pools[static_cast<std::size_t>(kWindowCodec)];
    auto* window =
        static_cast<std::byte*>(pool.slots.data) +
        (static_cast<std::size_t>(borrowed_first_) + b * static_cast<std::size_t>(window_slots_)) *
            pool.record_bytes;
    cudaStream_t fs = fill_stream_.get();
    ops::moe_expert_cache_window(expert_table_, host_table_, layer,
                                 DeviceSpan{window, options_.expert_window_bytes}, placements, fs);
    CUDA_CHECK(cudaEventRecord(window_planned_[b].get(), fs));
    ops::moe_expert_cache_fill(host_table_.slice(1, layer, 1), column, 0, fs);
    CUDA_CHECK(cudaEventRecord(window_ready_[b].get(), fs));
}

void Forward::moe(std::int32_t layer, const SubChunk& chunk, bool streamed, bool last,
                  const ForwardObserver* observer) {
    const std::int32_t tokens   = chunk.tokens();
    const LayerWeights& weights = model_.layers.at(static_cast<std::size_t>(layer));
    Tensor residual             = residual_columns(chunk);
    Tensor x                    = columns(chunk_.block_input, kHidden, tokens);
    Tensor write_scale          = columns(chunk_.write_scale, ops::kHyperConnectionStreams, tokens);
    ops::hyper_connection_read(residual, weights.moe_connection, x, write_scale, work_, stream_);
    // Verify packs t = row * width + column, with at least the root valid in every row.
    // Route padding from the last valid logits so it adds no experts to admit or staging.
    const Tensor valid_columns =
        chunk.state_mode == StateMode::Record ? chunk.valid_columns : Tensor{};
    const std::int32_t route_width = chunk.state_mode == StateMode::Record ? chunk.width : 1;
    // Staging (the class comment): whether this layer was staged, and whether the round stages the
    // next one, whose prediction from x runs on the fill stream beside this layer's routing.
    const bool staged            = stages(layer, chunk, streamed);
    const bool next              = stages(layer + 1, chunk, streamed);
    const std::int32_t following = layer + 1;
    cudaStream_t fs              = fill_stream_.get();
    const auto parity            = [](std::int32_t l) { return l % 2; };
    if (next) {
        CUDA_CHECK(cudaEventRecord(input_ready_.get(), stream_));
        CUDA_CHECK(cudaStreamWaitEvent(fs, input_ready_.get(), 0));
        Tensor control       = columns(staging_.control, ops::kMoeControlRows, tokens);
        Tensor ids           = columns(staging_.ids, ops::kMoeTopK, tokens);
        Tensor route_weights = columns(staging_.weights, ops::kMoeTopK, tokens);
        Tensor shared        = Tensor(staging_.shared.data, DType::FP32, {tokens});
        ops::bf16_projection(x, model_.layers.at(static_cast<std::size_t>(following)).moe_control,
                             control, fs);
        ops::moe_route(control, ids, route_weights, shared, valid_columns, route_width, fs);
    }

    auto scope   = work_.scope();
    MoeScratch s = allocate_moe(work_, tokens);
    ops::bf16_projection(x, weights.moe_control, s.control, stream_);
    ops::moe_route(s.control, s.expert_ids, s.route_weights, s.shared_scale, valid_columns,
                   route_width, stream_);
    if (options_.trace_verify_routes && chunk.state_mode == StateMode::Record) {
        auto* destination = static_cast<std::byte*>(verify_.expert_routes.data) +
                            static_cast<std::size_t>(layer) * verify_.expert_routes.nb[2];
        CUDA_CHECK(cudaMemcpyAsync(destination, s.expert_ids.data, s.expert_ids.bytes(),
                                   cudaMemcpyDeviceToDevice, stream_));
    }
    if (observer != nullptr && observer->after_route) {
        observer->after_route(layer, s.expert_ids);
    }

    if (streamed) {
        // Every sub-chunk of layer l reads window l % 2 (its placement column once planned, its
        // records after the copy); the last one frees it for layer l + 2's copy.
        const auto b            = static_cast<std::size_t>(layer % 2);
        ops::MoeExpertBank bank = weights.experts;
        bank.expert_table       = expert_table_.slice(1, layer, 1).view({ops::kMoeExperts});
        CUDA_CHECK(cudaStreamWaitEvent(stream_, window_planned_[b].get(), 0));
        const ops::MoeExpertsDeferral deferral{
            .slots = chunk_.window_columns.slice(1, layer % 2, 1).view({ops::kMoeExperts}),
            .ready = window_ready_[b].get()};
        Tensor y = columns(chunk_.block_output, kHidden, tokens);
        ops::moe_experts(x, s.expert_ids, s.route_weights, s.shared_scale, bank, y, work_, stream_,
                         &deferral);
        if (last && layer + 2 < kLayers) {
            CUDA_CHECK(cudaEventRecord(window_free_[b].get(), stream_));
            CUDA_CHECK(cudaStreamWaitEvent(fill_stream_.get(), window_free_[b].get(), 0));
            enqueue_window(layer + 2, layer % 2);
        }
        ops::hyper_connection_inject(residual, y, write_scale, residual, stream_);
        return;
    }

    // The cache schedule of moe_expert_cache.h with the fill on fill_stream_: admit marks hits and
    // assigns victim slots to misses; the fill copies the admitted misses' records into their
    // slots while moe_experts computes the shared and resident experts, then reads the admitted
    // misses from their slots (the layer's fill column) after filled_; commit then publishes the
    // slots. A miss admit could not place, or did not admit, is read through mapped memory with
    // the resident ones. Decode rounds of two or more lanes filter admissions by frequency, which
    // keeps the lanes' competing working sets from thrashing the pools; one lane and narrow
    // prefill chunks admit every miss (expert_residency.h, kExpertReferenceLimit). A staged layer
    // (the class comment) admits once its stage has read the tables, fills its admitted misses from
    // the stage's source column, and reads its staged records through the deferral; the round
    // stages the next layer around this one's MoE.
    if (staged) { CUDA_CHECK(cudaStreamWaitEvent(stream_, stage_planned_.get(), 0)); }
    const ops::MoeCacheAdmission admission = chunk.rows >= 2
                                                 ? ops::MoeCacheAdmission::FrequencyFilter
                                                 : ops::MoeCacheAdmission::EveryMiss;
    ops::moe_expert_cache_admit(s.expert_ids, expert_table_, host_table_, fill_table_, cache_,
                                layer, admission, stream_, &chunk_.expert_cache_counters);
    CUDA_CHECK(cudaEventRecord(admitted_.get(), stream_));
    CUDA_CHECK(cudaStreamWaitEvent(fs, admitted_.get(), 0));
    if (staged) {
        ops::moe_expert_cache_fill(staging_.source.view({ops::kMoeExperts, 1}),
                                   fill_table_.slice(1, layer, 1), 0, fs);
    } else {
        ops::moe_expert_cache_fill(host_table_, fill_table_, layer, fs);
    }
    CUDA_CHECK(cudaEventRecord(filled_.get(), fs));
    if (next) {
        const DeviceSpan span = staging_.span;
        Tensor stage = staging_.stage.slice(1, parity(following), 1).view({ops::kMoeExperts});
        ops::moe_expert_cache_stage(columns(staging_.ids, ops::kMoeTopK, tokens),
                                    kStagingCandidates, expert_table_, host_table_, following, span,
                                    staging_budget_bytes(following, tokens), stage, staging_.source,
                                    fs);
        CUDA_CHECK(cudaEventRecord(stage_planned_.get(), fs));
    }
    ops::MoeExpertBank bank = weights.experts;
    bank.expert_table       = expert_table_.slice(1, layer, 1).view({ops::kMoeExperts});
    Tensor slots            = fill_table_.slice(1, layer, 1).view({ops::kMoeExperts});
    if (staged) {
        ops::moe_expert_cache_defer(
            fill_table_, layer, staging_.stage.slice(1, parity(layer), 1).view({ops::kMoeExperts}),
            staging_.deferred, stream_);
        slots = staging_.deferred;
    }
    // filled_ follows the layer's staging copy on the fill stream, so it covers the staged records.
    const ops::MoeExpertsDeferral deferral{.slots = slots, .ready = filled_.get()};
    Tensor y = columns(chunk_.block_output, kHidden, tokens);
    ops::moe_experts(x, s.expert_ids, s.route_weights, s.shared_scale, bank, y, work_, stream_,
                     &deferral);
    if (next) {
        // The span's last readers (this moe_experts and, earlier on the fill stream, the admitted
        // fill) precede the copy, which the next layer's admitted fill and deferral follow.
        CUDA_CHECK(cudaEventRecord(experts_done_.get(), stream_));
        CUDA_CHECK(cudaStreamWaitEvent(fs, experts_done_.get(), 0));
        ops::moe_expert_cache_fill(host_table_.slice(1, following, 1),
                                   staging_.stage.slice(1, parity(following), 1), 0, fs,
                                   ops::kMoeStagingFillBlocks);
    }
    ops::moe_expert_cache_commit(expert_table_, fill_table_, layer, stream_);
    ops::hyper_connection_inject(residual, y, write_scale, residual, stream_);
}

void Forward::require_resolved() const {
    if (speculative_pending_) {
        throw std::logic_error(
            "Flash-Next must resolve its speculative candidate before forwarding");
    }
}

Tensor Forward::dflash_features() const {
    if (options_.dflash_features != DflashExecution::Enabled || feature_columns_ < 1) {
        throw std::logic_error("Flash-Next target features are unavailable");
    }
    return columns(chunk_.dflash_features, kDflashFeatures, feature_columns_);
}

void Forward::capture_features(std::int32_t layer, const SubChunk& chunk) {
    if (options_.dflash_features != DflashExecution::Enabled || feature_columns_ == 0) { return; }
    const auto found = std::find(kDflashFeatureLayers.begin(), kDflashFeatureLayers.end(), layer);
    if (found == kDflashFeatureLayers.end()) { return; }
    const auto first = std::max(chunk.offset, feature_offset_);
    const auto end   = std::min(chunk.offset + chunk.tokens(), feature_offset_ + feature_columns_);
    if (first >= end) { return; }
    const Tensor residual = residual_columns(chunk).slice(2, first - chunk.offset, end - first);
    Tensor output =
        columns(chunk_.dflash_features, kDflashFeatures, feature_columns_)
            .slice(1, first - feature_offset_, end - first)
            .slice(0, static_cast<std::int32_t>(found - kDflashFeatureLayers.begin()) * kHidden,
                   kHidden);
    ops::stream_mean(residual, output, stream_);
}

void Forward::head(std::int32_t first, std::int32_t width) {
    pending_verify_width_ = 0;
    const Tensor residual(static_cast<std::byte*>(chunk_.residual.data) +
                              static_cast<std::size_t>(first) * kHidden *
                                  ops::kHyperConnectionStreams * sizeof(std::uint16_t),
                          DType::BF16, {kHidden, ops::kHyperConnectionStreams, width});
    Tensor x = columns(chunk_.head_input, kHidden, width);
    Tensor no_write_scale;
    ops::hyper_connection_read(residual, model_.final_connection, x, no_write_scale, work_,
                               stream_);
    Tensor logits = columns(chunk_.logits, kVocabRows, width);
    ops::linear(x, model_.head, logits, stream_);
}

#include "runtime/verify_impl.h"
#include "runtime/speculative_impl.h"

} // namespace ninfer::targets::qwen3_8_flash_next
