#include "runtime/dflash.h"

#include "core/device_copy.h"
#include "core/device.h"
#include "ninfer/ops/dflash2_path_select.h"
#include "ninfer/ops/embedding.h"
#include "ninfer/ops/grouped_dynamic_conv.h"
#include "ninfer/ops/linear.h"
#include "ninfer/ops/normalized_rope_kv_append.h"
#include "ninfer/ops/prepare_masked_block.h"
#include "ninfer/ops/residual_add.h"
#include "ninfer/ops/rmsnorm.h"
#include "ninfer/ops/rope.h"
#include "ninfer/ops/scatter.h"
#include "ninfer/ops/silu_mul.h"
#include "ninfer/ops/swa.h"

#include <algorithm>
#include <array>
#include <stdexcept>
#include <memory>

namespace ninfer::targets::qwen3_8_flash_next {
namespace {

constexpr std::int32_t kQuery        = 4096;
constexpr std::int32_t kKv           = 1024;
constexpr std::int32_t kIntermediate = 7680;
constexpr float kTheta               = 10000.0F;
constexpr float kScale               = 0.08838834764831845F;

struct AppendScratch {
    Tensor projected, normalized, key, value;
};

template <class Allocator>
AppendScratch append_scratch(Allocator& arena, std::int32_t columns) {
    return {.projected  = arena.alloc(DType::BF16, {kHidden, columns}),
            .normalized = arena.alloc(DType::BF16, {kHidden, columns}),
            .key        = arena.alloc(DType::BF16, {kKv, columns}),
            .value      = arena.alloc(DType::BF16, {kKv, columns})};
}

struct DraftScratch {
    Tensor ids, positions, residual, normalized, prepared, dynamic;
    Tensor qkv, query_raw, key_raw, value, query, key, attention, delta, gate_up, intermediate;
    Tensor draft_raw, draft_hidden, full_logits, logits;
};

template <class Allocator>
DraftScratch draft_scratch(Allocator& arena, std::int32_t width, std::int32_t batch) {
    const std::int32_t columns = width * batch;
    const std::int32_t drafts  = (width - 1) * batch;
    return {
        .ids        = arena.alloc(DType::I32, {width, batch}),
        .positions  = arena.alloc(DType::I32, {width, batch}),
        .normalized = arena.alloc(DType::BF16, {kHidden, columns}),
        .prepared   = arena.alloc(DType::BF16, {kHidden, columns}),
        .dynamic =
            arena.alloc(DType::BF16, {ops::grouped_dynamic_conv_groups(kHidden), 2, width, batch}),
        .qkv          = arena.alloc(DType::BF16, {6144, columns}),
        .query_raw    = arena.alloc(DType::BF16, {kQuery, columns}),
        .key_raw      = arena.alloc(DType::BF16, {kKv, columns}),
        .value        = arena.alloc(DType::BF16, {kKv, columns}),
        .query        = arena.alloc(DType::BF16, {128, 32, columns}),
        .key          = arena.alloc(DType::BF16, {128, 8, columns}),
        .attention    = arena.alloc(DType::BF16, {kQuery, columns}),
        .delta        = arena.alloc(DType::BF16, {kHidden, columns}),
        .gate_up      = arena.alloc(DType::BF16, {2 * kIntermediate, columns}),
        .intermediate = arena.alloc(DType::BF16, {kIntermediate, columns}),
        .draft_raw    = arena.alloc(DType::BF16, {kHidden, drafts}),
        .draft_hidden = arena.alloc(DType::BF16, {kHidden, drafts}),
        .logits       = arena.alloc(DType::BF16, {kTokenDomain, drafts}),
    };
}

Weight bf16_table(const Tensor& storage, std::int32_t rows, std::int32_t columns) {
    Weight table{};
    table.payload = table.qdata = storage.data;
    table.payload_bytes         = storage.bytes();
    table.qtype                 = QType::BF16_CTRL;
    table.layout                = QuantLayout::Contiguous;
    table.ndim                  = 2;
    table.n = table.shape[0] = table.padded_shape[0] = rows;
    table.k = table.shape[1] = table.padded_shape[1] = columns;
    return table;
}

void reserve(WorkspaceLayoutBuilder& builder, std::size_t bytes) {
    if (bytes != 0) { static_cast<void>(builder.alloc_bytes(bytes, 256)); }
}

std::size_t plan_work_bytes(const ModelView& model, const State& state) {
    if (!model.dflash || !state.has_dflash()) {
        throw std::invalid_argument(
            "Flash-Next drafter requires companion weights and context state");
    }
    const auto& weights = *model.dflash;
    WorkspaceLayoutBuilder builder;
    {
        auto scope = builder.scope();
        // The pending gather remains live through append(). BF16 embedding is an exact gather
        // over lane-owned fixed-width feature records, not an independently packed weight copy.
        static_cast<void>(
            builder.alloc(DType::BF16, {kDflashFeatures, kDflashWidth, state.lanes()}));
        static_cast<void>(
            append_scratch(builder, std::max(kDflashWindow, kDflashWidth * state.lanes())));
        reserve(builder, ops::linear_workspace_capacity_bytes(
                             QType::NVFP4, kHidden, kDflashFeatures, ops::LinearPolicy::A16Only, 1,
                             std::max(kDflashWindow, kDflashWidth * state.lanes())));
    }
    for (std::int32_t batch = 1; batch <= state.lanes(); ++batch) {
        for (std::int32_t width = 2; width <= kDflashWidth; ++width) {
            auto scope = builder.scope();
            static_cast<void>(draft_scratch(builder, width, batch));
            std::size_t child = ops::swa_workspace_capacity_bytes(
                {0, static_cast<std::uint32_t>(state.lane_capacity())}, width, width, batch);
            child = std::max(child, ops::grouped_dynamic_conv_prepare_workspace_capacity_bytes(
                                        QType::NVFP4, kHidden, width, width, batch));
            child = std::max(child, ops::dflash2_path_select_workspace_capacity_bytes(
                                        QType::NVFP4, kHidden, width - 1, width - 1, batch));
            const std::array<const Weight*, 4> matrices{
                &weights.layers.front().query_key_value, &weights.layers.front().attention_output,
                &weights.layers.front().gate_up, &weights.layers.front().down};
            for (const auto* matrix : matrices) {
                child =
                    std::max(child, ops::linear_workspace_capacity_bytes(
                                        matrix->qtype, matrix->n, matrix->k,
                                        ops::LinearPolicy::A16Only, width * batch, width * batch));
            }
            reserve(builder, child);
        }
    }
    return builder.peak_bytes(256);
}

} // namespace

Dflash::Dflash(const ModelView& model, State& state, cudaStream_t stream)
    : model_(model), weights_(model.dflash.value()), state_(state), stream_(stream),
      work_(plan_work_bytes(model, state)),
      // Both panel extents are multiples of 256 bytes at the registered dimensions.
      transfers_(std::size_t{2} * static_cast<std::size_t>(state.lanes()) *
                 (std::size_t{kHidden} * kDflashWidth + std::size_t{kVocabRows} * kDflashMaxK)),
      proposal_residual_(transfers_.alloc(DType::BF16, {kHidden, kDflashWidth * state.lanes()})),
      head_logits_(transfers_.alloc(DType::BF16, {kVocabRows, kDflashMaxK * state.lanes()})),
      inputs_(sizeof(HostInputs)),
      input_frame_(inputs_.alloc(DType::U8, {static_cast<std::int32_t>(sizeof(HostInputs))})),
      input_staging_(sizeof(HostInputs)),
      inputs_uploaded_(cudaEventDisableTiming | cudaEventBlockingSync) {
    std::construct_at(static_cast<HostInputs*>(input_staging_.data()));
    // Large graph memcpy sources are checked at launch, before their producer kernels run.
    // Stable initialized panels avoid depending on pre-replay contents of aliased scratch.
    CUDA_CHECK(cudaMemsetAsync(transfers_.base(), 0, transfers_.capacity(), stream_));
}

Dflash::~Dflash() {
    // Enqueued context appends may outlive their host call, but not their owned scratch/staging.
    static_cast<void>(cudaStreamSynchronize(stream_));
}

Dflash::HostInputs& Dflash::host_inputs() {
    CUDA_CHECK(cudaEventSynchronize(inputs_uploaded_.get()));
    return *static_cast<HostInputs*>(input_staging_.data());
}

void Dflash::upload_inputs() {
    CUDA_CHECK(cudaMemcpyAsync(input_frame_.data, input_staging_.data(), sizeof(HostInputs),
                               cudaMemcpyHostToDevice, stream_));
    CUDA_CHECK(cudaEventRecord(inputs_uploaded_.get(), stream_));
}

void Dflash::flush(std::span<const std::int32_t> lanes) {
    if (lanes.empty() || lanes.size() > static_cast<std::size_t>(state_.lanes())) {
        throw std::invalid_argument("Flash-Next context flush has invalid membership");
    }
    auto& host   = host_inputs();
    bool pending = false;
    for (std::size_t row = 0; row < lanes.size(); ++row) {
        if (std::find(lanes.begin(), lanes.begin() + static_cast<std::ptrdiff_t>(row),
                      lanes[row]) != lanes.begin() + static_cast<std::ptrdiff_t>(row)) {
            throw std::invalid_argument("Flash-Next context flush repeats a lane");
        }
        const auto frontier = state_.committed(lanes[row]);
        const auto count    = state_.dflash_pending_count(lanes[row]);
        host.lanes[row]     = lanes[row];
        host.counts[row]    = count;
        pending             = pending || count != 0;
        for (std::int32_t column = 0; column < kDflashWidth; ++column) {
            host.positions[row * kDflashWidth + static_cast<std::size_t>(column)] =
                frontier - count + column;
        }
    }
    if (!pending) { return; }
    upload_inputs();
    const auto batch = static_cast<std::int32_t>(lanes.size());
    const auto at    = [&](std::size_t offset, std::initializer_list<std::int32_t> shape) {
        return Tensor(static_cast<std::byte*>(input_frame_.data) + offset, DType::I32, shape);
    };
    append_pending(at(offsetof(HostInputs, positions), {kDflashWidth, batch}),
                   at(offsetof(HostInputs, counts), {batch}),
                   at(offsetof(HostInputs, lanes), {batch}));
    for (const auto lane : lanes) { state_.set_dflash_pending_count(lane, 0); }
}

void Dflash::append_prefill(std::int32_t lane, const Tensor& features) {
    const auto columns = features.ne[1];
    if (features.ne[0] != kDflashFeatures || columns < 1 || columns > kDflashWindow ||
        state_.dflash_pending_count(lane) != 0) {
        throw std::invalid_argument(
            "Flash-Next prefill context requires complete settled features");
    }
    auto& host       = host_inputs();
    host.lanes[0]    = lane;
    host.counts[0]   = columns;
    const auto first = state_.committed(lane) - columns;
    for (std::int32_t column = 0; column < columns; ++column) {
        host.positions[static_cast<std::size_t>(column)] = first + column;
    }
    upload_inputs();
    const auto at = [&](std::size_t offset, std::initializer_list<std::int32_t> shape) {
        return Tensor(static_cast<std::byte*>(input_frame_.data) + offset, DType::I32, shape);
    };
    append(features.view({kDflashFeatures, columns, 1}),
           at(offsetof(HostInputs, positions), {columns, 1}), at(offsetof(HostInputs, counts), {1}),
           at(offsetof(HostInputs, lanes), {1}));
}

void Dflash::save_decode_features(std::span<const std::int32_t> lanes, const Tensor& features) {
    auto& host       = host_inputs();
    const auto batch = static_cast<std::int32_t>(lanes.size());
    if (batch < 1 || batch > state_.lanes() || features.ne[0] != kDflashFeatures ||
        features.ne[1] != batch) {
        throw std::invalid_argument("Flash-Next decode features have invalid dimensions");
    }
    for (std::size_t row = 0; row < lanes.size(); ++row) {
        if (state_.dflash_pending_count(lanes[row]) != 0) {
            throw std::logic_error("Flash-Next decode would replace pending context features");
        }
        host.lanes[row]  = lanes[row];
        host.counts[row] = 1;
    }
    upload_inputs();
    const Tensor selected(static_cast<std::byte*>(input_frame_.data) + offsetof(HostInputs, lanes),
                          DType::I32, {batch});
    const Tensor counts(static_cast<std::byte*>(input_frame_.data) + offsetof(HostInputs, counts),
                        DType::I32, {batch});
    Tensor destination = state_.dflash_pending_features().slice(1, 0, 1);
    ops::scatter_bf16_batch(features.view({kDflashFeatures, 1, batch}), selected, counts,
                            destination, stream_);
    for (const auto lane : lanes) { state_.set_dflash_pending_count(lane, 1); }
}

void Dflash::append(const Tensor& features, const Tensor& positions, const Tensor& counts,
                    const Tensor& lanes) {
    const auto width = features.ne[1];
    const auto batch = features.ne[2];
    if (features.ne[0] != kDflashFeatures || width < 1 || width > kDflashWindow || batch < 1 ||
        batch > state_.lanes()) {
        throw std::invalid_argument("Flash-Next drafter context has invalid feature dimensions");
    }
    auto scope = work_.scope();
    auto s     = append_scratch(work_, width * batch);
    ops::linear_packed_sequences(features.view({kDflashFeatures, width * batch}),
                                 weights_.feature_projection, s.projected,
                                 ops::LinearPolicy::A16Only, work_, stream_, width);
    ops::rmsnorm(s.projected, weights_.context_norm, kRmsEpsilon, false, s.normalized, stream_);
    for (std::int32_t layer = 0; layer < kDflashLayers; ++layer) {
        const auto& weight = weights_.layers.at(static_cast<std::size_t>(layer));
        ops::linear_kv_projection(s.normalized, weight.query_key_value, s.key, s.value, stream_,
                                  width);
        ops::normalized_rope_kv_append(
            s.key.view({128, 8, width, batch}), s.value.view({128, 8, width, batch}),
            weight.key_norm, positions, counts, lanes, kRmsEpsilon, kTheta,
            {0, static_cast<std::uint32_t>(width)},
            state_.dflash_cache().layer_view(static_cast<std::uint32_t>(layer)), stream_);
    }
}

void Dflash::append_pending(const Tensor& positions, const Tensor& counts, const Tensor& lanes) {
    auto scope          = work_.scope();
    const auto batch    = lanes.ne[0];
    Tensor packed       = work_.alloc(DType::BF16, {kDflashFeatures * kDflashWidth, batch});
    const Tensor stored = state_.dflash_pending_features();
    ops::embedding(lanes, bf16_table(stored, state_.lanes(), kDflashFeatures * kDflashWidth),
                   packed, stream_);
    append(packed.view({kDflashFeatures, kDflashWidth, batch}), positions, counts, lanes);
}

void Dflash::backup_context(const Tensor& slots, Tensor& keys, Tensor& values, std::int32_t rows) {
    constexpr std::int32_t columns_per_row = 8 * kDflashWidth;
    // Cyclic storage is [128,capacity,8,lanes]; a gather row is one head of one slot.
    const Tensor packed_keys(keys.data, DType::BF16, {128, columns_per_row * rows, kDflashLayers});
    const Tensor packed_values(values.data, DType::BF16,
                               {128, columns_per_row * rows, kDflashLayers});
    for (std::int32_t layer = 0; layer < kDflashLayers; ++layer) {
        const auto context = state_.dflash_cache().layer_view(static_cast<std::uint32_t>(layer));
        const auto total_rows =
            static_cast<std::int32_t>(static_cast<std::int64_t>(context.padded_capacity) *
                                      context.num_kv_heads * context.lane_capacity);
        Tensor key   = packed_keys.slice(2, layer, 1).view({128, columns_per_row * rows});
        Tensor value = packed_values.slice(2, layer, 1).view({128, columns_per_row * rows});
        ops::embedding(slots, bf16_table(context.k, total_rows, 128), key, stream_);
        ops::embedding(slots, bf16_table(context.v, total_rows, 128), value, stream_);
    }
}

void Dflash::restore_context(const Tensor& slots, const Tensor& keys, const Tensor& values,
                             std::int32_t row, std::int32_t rows) {
    constexpr std::int32_t columns_per_row = 8 * kDflashWidth;
    const Tensor packed_keys(keys.data, DType::BF16, {128, columns_per_row * rows, kDflashLayers});
    const Tensor packed_values(values.data, DType::BF16,
                               {128, columns_per_row * rows, kDflashLayers});
    const Tensor indices = slots.slice(0, row * columns_per_row, columns_per_row);
    for (std::int32_t layer = 0; layer < kDflashLayers; ++layer) {
        const auto context = state_.dflash_cache().layer_view(static_cast<std::uint32_t>(layer));
        const auto total_rows =
            static_cast<std::int32_t>(static_cast<std::int64_t>(context.padded_capacity) *
                                      context.num_kv_heads * context.lane_capacity);
        const Tensor key   = packed_keys.slice(2, layer, 1)
                                 .view({128, columns_per_row * rows})
                                 .slice(1, row * columns_per_row, columns_per_row);
        const Tensor value = packed_values.slice(2, layer, 1)
                                 .view({128, columns_per_row * rows})
                                 .slice(1, row * columns_per_row, columns_per_row);
        Tensor destination_key(context.k.data, DType::BF16, {128, total_rows});
        Tensor destination_value(context.v.data, DType::BF16, {128, total_rows});
        ops::scatter(key, indices, destination_key, stream_);
        ops::scatter(value, indices, destination_value, stream_);
    }
}

void Dflash::propose(std::int32_t width, const Tensor& anchors, const Tensor& frontiers,
                     const Tensor& valid_columns, const Tensor& lanes,
                     const ops::SamplingConfig* configs, Tensor& drafts, Tensor& selector_ids,
                     Tensor& selector_q) {
    const auto batch = lanes.ne[0];
    if (width < 2 || width > kDflashWidth || batch < 1 || batch > state_.lanes()) {
        throw std::invalid_argument("Flash-Next drafter proposal dimensions are invalid");
    }
    auto scope         = work_.scope();
    auto s             = draft_scratch(work_, width, batch);
    s.residual         = proposal_residual_.slice(1, 0, width * batch);
    s.full_logits      = head_logits_.slice(1, 0, (width - 1) * batch);
    const auto columns = width * batch;
    ops::prepare_masked_block(anchors, frontiers, valid_columns, 248077, s.ids, s.positions,
                              stream_);
    ops::embedding(s.ids.view({columns}), model_.embedding, s.residual, stream_);
    ops::rmsnorm(s.residual, weights_.layers.front().input_norm, kRmsEpsilon, false, s.normalized,
                 stream_);
    for (std::int32_t layer = 0; layer < kDflashLayers; ++layer) {
        const auto& weight = weights_.layers.at(static_cast<std::size_t>(layer));
        Tensor prepared    = s.prepared.view({kHidden, width, batch});
        ops::grouped_dynamic_conv_prepare(
            s.normalized.view({kHidden, width, batch}), weight.attention_conv.base_kernel,
            weight.attention_conv.kernel_projection, prepared, s.dynamic, work_, stream_);
        ops::linear_packed_sequences(s.prepared, weight.query_key_value, s.qkv,
                                     ops::LinearPolicy::A16Only, work_, stream_, width);
        ops::extract_bf16_columns(s.qkv, 0, s.query_raw, stream_);
        ops::extract_bf16_columns(s.qkv, kQuery, s.key_raw, stream_);
        ops::extract_bf16_columns(s.qkv, kQuery + kKv, s.value, stream_);
        ops::rmsnorm(s.query_raw.view({128, 32, columns}), weight.query_norm, kRmsEpsilon, false,
                     s.query, stream_);
        ops::rmsnorm(s.key_raw.view({128, 8, columns}), weight.key_norm, kRmsEpsilon, false, s.key,
                     stream_);
        ops::rope(s.positions.view({columns}), 128, kTheta, s.query, s.key, stream_);
        Tensor attention = s.attention.view({128, 32, width, batch});
        ops::swa(s.query.view({128, 32, width, batch}), s.key.view({128, 8, width, batch}),
                 s.value.view({128, 8, width, batch}), s.positions, valid_columns, lanes, kScale,
                 state_.dflash_cache().layer_view(static_cast<std::uint32_t>(layer)),
                 {0, static_cast<std::uint32_t>(state_.lane_capacity())}, work_, attention,
                 stream_);
        ops::linear_packed_sequences(s.attention, weight.attention_output, s.delta,
                                     ops::LinearPolicy::A16Only, work_, stream_, width);
        Tensor residual   = s.residual.view({kHidden, width, batch});
        Tensor normalized = s.normalized.view({kHidden, width, batch});
        ops::grouped_dynamic_conv_finish_residual_rmsnorm(
            s.delta.view({kHidden, width, batch}), weight.attention_conv.base_kernel, s.dynamic,
            residual, weight.post_attention_norm, kRmsEpsilon, normalized, stream_);
        ops::grouped_dynamic_conv_prepare(normalized, weight.mlp_conv.base_kernel,
                                          weight.mlp_conv.kernel_projection, prepared, s.dynamic,
                                          work_, stream_);
        ops::linear_packed_sequences(s.prepared, weight.gate_up, s.gate_up,
                                     ops::LinearPolicy::A16Only, work_, stream_, width);
        ops::silu_mul(s.gate_up.slice(0, 0, kIntermediate),
                      s.gate_up.slice(0, kIntermediate, kIntermediate), s.intermediate, stream_);
        ops::linear_packed_sequences(s.intermediate, weight.down, s.delta,
                                     ops::LinearPolicy::A16Only, work_, stream_, width);
        if (layer + 1 < kDflashLayers) {
            ops::grouped_dynamic_conv_finish_residual_rmsnorm(
                s.delta.view({kHidden, width, batch}), weight.mlp_conv.base_kernel, s.dynamic,
                residual, weights_.layers.at(static_cast<std::size_t>(layer) + 1).input_norm,
                kRmsEpsilon, normalized, stream_);
        } else {
            ops::grouped_dynamic_conv_finish(s.delta.view({kHidden, width, batch}),
                                             weight.mlp_conv.base_kernel, s.dynamic, prepared,
                                             stream_);
            ops::residual_add(s.prepared, s.residual, stream_);
        }
    }
    const auto drafts_per_row = width - 1;
    const std::size_t pitch   = static_cast<std::size_t>(kHidden) * sizeof(std::uint16_t);
    copy_device_rows_async(s.draft_raw.data, pitch * static_cast<std::size_t>(drafts_per_row),
                           static_cast<const std::byte*>(s.residual.data) + pitch,
                           pitch * static_cast<std::size_t>(width),
                           pitch * static_cast<std::size_t>(drafts_per_row),
                           static_cast<std::size_t>(batch), stream_);
    ops::rmsnorm(s.draft_raw, weights_.final_norm, kRmsEpsilon, false, s.draft_hidden, stream_);
    ops::linear_packed_sequences(s.draft_hidden, model_.head, s.full_logits, stream_,
                                 drafts_per_row);
    ops::extract_bf16_columns(s.full_logits, 0, s.logits, stream_);
    ops::dflash2_path_select(s.logits.view({kTokenDomain, drafts_per_row, batch}),
                             s.draft_hidden.view({kHidden, drafts_per_row, batch}),
                             weights_.hidden_projection, weights_.predecessor_codebook,
                             weights_.successor_codebook, anchors, frontiers, configs, drafts,
                             work_, stream_, nullptr, nullptr, nullptr, &selector_ids, &selector_q);
}

} // namespace ninfer::targets::qwen3_8_flash_next
