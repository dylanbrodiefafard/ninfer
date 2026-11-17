#include "runtime/state.h"

#include "core/device.h"
#include "core/layout.h"

#include <algorithm>
#include <cstring>
#include <cmath>
#include <optional>
#include <limits>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

namespace ninfer::targets::qwen3_8_flash_next {
namespace {

constexpr std::size_t kAlignment        = 256;
constexpr std::size_t kDflashCountBytes = 256;
static_assert(ops::kPLessProposalCalibrationTemperatureCount == 8);
// I32 entries per lane id: each id starts its own 16-byte group, the alignment the QSA Ops require
// of a lanes selector.
constexpr std::int32_t kLaneIdStride = 4;
constexpr auto kPageTokens           = static_cast<std::int64_t>(ops::kQsaPageTokens);

struct StateLayout {
    LinearAttentionStatePoolLayout gdn;
    PagedKVPoolLayout qsa;
    TensorRegion ple_history;
    TensorRegion lane_ids;
    TensorRegion token_counts;
    TensorRegion allowed;
    TensorRegion tail_logits;
    std::optional<CyclicKVCacheLayout> dflash;
    TensorRegion dflash_pending_features;
    std::size_t bytes = 0;
};

// Pages of `tokens` tokens, rounded up; tokens >= 1.
std::int64_t pages_of(std::int64_t tokens) { return (tokens + kPageTokens - 1) / kPageTokens; }

StateLayout plan_state(std::int32_t lanes, std::int32_t lane_capacity, std::int64_t pool_pages,
                       ops::QsaKvFormat format, DflashExecution dflash) {
    LayoutBuilder builder;
    StateLayout layout;
    layout.gdn = plan_linear_attention_state_pool(
        builder, LinearAttentionStatePoolSpec{.layers         = kGdnLayers,
                                              .conv_channels  = kGdnConvChannels,
                                              .conv_width     = kGdnConvState,
                                              .value_heads    = kGdnValueHeads,
                                              .value_head_dim = kGdnHeadDim,
                                              .key_head_dim   = kGdnHeadDim,
                                              .slot_count     = lanes,
                                              .conv_dtype     = DType::BF16});
    const std::vector<PagedKVPlaneSpec> layer_planes = ops::qsa_paged_plane_specs(format);
    PagedKVPoolSpec spec{
        .page_group_count      = static_cast<std::uint32_t>(pool_pages),
        .logical_page_capacity = static_cast<std::uint32_t>(lane_capacity / ops::kQsaPageTokens),
        .table_rows            = lanes,
        .plane_order           = PagedKVPlaneOrder::PageMajor,
        .planes                = {},
    };
    spec.planes.reserve(layer_planes.size() * kQsaLayers);
    for (std::int32_t layer = 0; layer < kQsaLayers; ++layer) {
        spec.planes.insert(spec.planes.end(), layer_planes.begin(), layer_planes.end());
    }
    layout.qsa = plan_paged_kv_pool(builder, spec);
    layout.ple_history =
        builder.add_tensor(DType::BF16, {ops::kPleConvHistory, ops::kPleChannels, lanes},
                           kAlignment, "PLE convolution history");
    layout.lane_ids =
        builder.add_tensor(DType::I32, {kLaneIdStride, lanes}, kAlignment, "lane ids");
    layout.token_counts =
        builder.add_tensor(DType::I32, {kVocabRows, lanes}, kAlignment, "sampling token counts");
    // U32 words held as I32 storage (DType has no U32).
    layout.allowed = builder.add_tensor(DType::I32, {State::kAllowedTokenWordStride, lanes},
                                        kAlignment, "token eligibility bitsets");
    layout.tail_logits =
        builder.add_tensor(DType::BF16, {kVocabRows, lanes}, kAlignment, "retained tail logits");
    if (dflash == DflashExecution::Enabled) {
        layout.dflash = plan_cyclic_kv_cache(builder, kDflashLayers, kDflashWindow, 8, 128, lanes);
        layout.dflash_pending_features =
            builder.add_tensor(DType::BF16, {kDflashFeatures, kDflashWidth, lanes}, kAlignment,
                               "DFlash pending target features");
    }
    layout.bytes = builder.finish(kAlignment, "Qwen3.8-Flash-Next lane state");
    return layout;
}

// Binds QSA layer `layer`'s planes, in qsa_paged_plane_specs order, from the pool.
ops::QsaPagedState qsa_view(const PagedKVPool& pool, ops::QsaKvFormat format, std::int32_t layer) {
    const std::size_t per_layer = ops::qsa_paged_plane_specs(format).size();
    const std::size_t base      = static_cast<std::size_t>(layer) * per_layer;
    const auto plane            = [&](std::size_t index) { return pool.plane(base + index); };
    if (format == ops::QsaKvFormat::BF16) {
        return ops::QsaPagedState{.format       = format,
                                  .k            = plane(0),
                                  .v            = plane(1),
                                  .k_scales     = Tensor(),
                                  .v_scales     = Tensor(),
                                  .index_keys   = plane(2),
                                  .positions    = plane(3),
                                  .block_keys   = plane(4),
                                  .block_tables = pool.block_tables()};
    }
    return ops::QsaPagedState{.format       = format,
                              .k            = plane(0),
                              .v            = plane(1),
                              .k_scales     = plane(2),
                              .v_scales     = plane(3),
                              .index_keys   = plane(4),
                              .positions    = plane(5),
                              .block_keys   = plane(6),
                              .block_tables = pool.block_tables()};
}

} // namespace

State::State(std::int32_t lanes, std::int32_t context_tokens, std::int64_t pool_tokens,
             ops::QsaKvFormat format, DflashExecution dflash)
    : lanes_(lanes), format_(format) {
    if (lanes < 1 || lanes > kMaxLanes) {
        throw std::invalid_argument("qwen3_8_flash_next State: lanes must be in [1, " +
                                    std::to_string(kMaxLanes) + "]");
    }
    if (context_tokens < 1 || context_tokens > ops::kQsaMaximumCapacity) {
        throw std::invalid_argument("qwen3_8_flash_next State: the context must be in [1, " +
                                    std::to_string(ops::kQsaMaximumCapacity) + "]");
    }
    // A page id is an I32 table entry, and the pool's page count fits it.
    if (pool_tokens < 1 || pages_of(pool_tokens) > std::numeric_limits<std::int32_t>::max()) {
        throw std::invalid_argument("qwen3_8_flash_next State: the QSA pool must hold 1 to "
                                    "2^31 - 1 pages");
    }
    // Whole QSA pages; kQsaMaximumCapacity is itself a page multiple.
    lane_capacity_ = static_cast<std::int32_t>(pages_of(context_tokens) * kPageTokens);
    const StateLayout layout =
        plan_state(lanes, lane_capacity_, pages_of(pool_tokens), format, dflash);
    backing_ = DeviceBuffer(layout.bytes);
    const DeviceSpan span{backing_.p, backing_.bytes};
    gdn_  = LinearAttentionStatePool(span, layout.gdn);
    pool_ = std::make_unique<PagedKVPool>(span, layout.qsa);
    for (std::int32_t layer = 0; layer < kQsaLayers; ++layer) {
        qsa_.at(static_cast<std::size_t>(layer)) = qsa_view(*pool_, format, layer);
    }
    ple_history_  = layout.ple_history.bind(span);
    lane_ids_     = layout.lane_ids.bind(span);
    token_counts_ = layout.token_counts.bind(span);
    allowed_      = layout.allowed.bind(span);
    tail_logits_  = layout.tail_logits.bind(span);
    if (layout.dflash) {
        dflash_cache_            = std::make_unique<CyclicKVCache>(span, *layout.dflash);
        dflash_pending_features_ = layout.dflash_pending_features.bind(span);
    }
    lane_states_.resize(static_cast<std::size_t>(lanes));

    // Every lane starts closed: zero state and every block-table entry unmapped (-1).
    backing_.fill(0);
    const Tensor& tables = pool_->block_tables();
    CUDA_CHECK(cudaMemset(tables.data, 0xff, tables.bytes()));
    std::vector<std::int32_t> ids(static_cast<std::size_t>(kLaneIdStride) *
                                  static_cast<std::size_t>(lanes));
    for (std::int32_t lane = 0; lane < lanes; ++lane) {
        ids[static_cast<std::size_t>(kLaneIdStride) * static_cast<std::size_t>(lane)] = lane;
    }
    CUDA_CHECK(cudaMemcpy(lane_ids_.data, ids.data(), ids.size() * sizeof(std::int32_t),
                          cudaMemcpyHostToDevice));
}

std::int64_t State::pool_tokens() const noexcept {
    return static_cast<std::int64_t>(pool_->page_group_count()) * kPageTokens;
}

std::int64_t State::free_tokens() const noexcept {
    return static_cast<std::int64_t>(pool_->page_group_count() - pool_->entitled_pages()) *
           kPageTokens;
}

bool State::can_open(std::int64_t reserve_tokens) const noexcept {
    if (reserve_tokens < 1 || reserve_tokens > lane_capacity_) { return false; }
    bool free_lane = false;
    for (const LaneState& lane : lane_states_) { free_lane = free_lane || !lane.open; }
    return free_lane && pool_->can_reserve(static_cast<std::uint32_t>(pages_of(reserve_tokens)));
}

std::optional<std::int32_t> State::open(std::int64_t reserve_tokens, cudaStream_t stream) {
    if (!can_open(reserve_tokens)) { return std::nullopt; }
    std::int32_t lane = 0;
    while (lane_states_.at(static_cast<std::size_t>(lane)).open) { ++lane; }
    LaneState& state             = lane_states_[static_cast<std::size_t>(lane)];
    const auto pages             = static_cast<std::uint32_t>(pages_of(reserve_tokens));
    PagedKVAllocation allocation = pool_->reserve(pages);
    allocation.materialize_pages(pages, stream);
    allocation.bind_row(lane, stream);
    gdn_.zero_slot(lane, stream);
    const Tensor history = ple_history_.slice(2, lane, 1);
    CUDA_CHECK(cudaMemsetAsync(history.data, 0, history.bytes(), stream));
    const Tensor counts = token_counts_.slice(1, lane, 1);
    CUDA_CHECK(cudaMemsetAsync(counts.data, 0, counts.bytes(), stream));
    state = LaneState{.open            = true,
                      .committed       = 0,
                      .reserved_tokens = reserve_tokens,
                      .pages           = std::move(allocation)};
    return lane;
}

void State::close(std::int32_t lane, cudaStream_t stream) {
    static_cast<void>(open_lane(lane));
    LaneState& state = lane_states_[static_cast<std::size_t>(lane)];
    // The row is unmapped in stream order, after every enqueued reader of the lane.
    const Tensor row = pool_->block_table_row(lane);
    CUDA_CHECK(cudaMemsetAsync(row.data, 0xff, row.bytes(), stream));
    state = LaneState{};
}

const State::LaneState& State::open_lane(std::int32_t lane) const {
    if (lane < 0 || lane >= lanes_ || !lane_states_[static_cast<std::size_t>(lane)].open) {
        throw std::out_of_range("qwen3_8_flash_next State: lane " + std::to_string(lane) +
                                " is not open");
    }
    return lane_states_[static_cast<std::size_t>(lane)];
}

bool State::is_open(std::int32_t lane) const noexcept {
    return lane >= 0 && lane < lanes_ && lane_states_[static_cast<std::size_t>(lane)].open;
}

std::int32_t State::committed(std::int32_t lane) const { return open_lane(lane).committed; }

std::int64_t State::reserved_tokens(std::int32_t lane) const {
    return open_lane(lane).reserved_tokens;
}

void State::advance(std::int32_t lane, std::int32_t tokens) {
    const LaneState& checked = open_lane(lane);
    if (tokens < 0 || tokens > checked.reserved_tokens - checked.committed) {
        throw std::out_of_range("qwen3_8_flash_next State: " + std::to_string(tokens) +
                                " tokens exceed lane " + std::to_string(lane) +
                                "'s remaining reservation of " +
                                std::to_string(checked.reserved_tokens - checked.committed));
    }
    lane_states_[static_cast<std::size_t>(lane)].committed += tokens;
}

void State::resize(std::int32_t lane, std::int32_t frontier, std::int64_t reserve_tokens,
                   cudaStream_t stream) {
    const LaneState& checked = open_lane(lane);
    if (frontier < 0 || reserve_tokens < 1 || reserve_tokens > lane_capacity_ ||
        frontier > reserve_tokens || frontier > checked.committed) {
        throw std::out_of_range("Flash-Next resize exceeds the lane's valid extent");
    }
    auto& state          = lane_states_.at(static_cast<std::size_t>(lane));
    const auto pages     = static_cast<std::uint32_t>(pages_of(reserve_tokens));
    const auto old_pages = state.pages.mapped_page_count();
    const PagedKVResize change{.allocation       = &state.pages,
                               .mapped_pages     = std::min(state.pages.mapped_page_count(), pages),
                               .page_entitlement = pages};
    resize_paged_kv_bundle(std::span(&change, 1));
    state.pages.materialize_pages(pages, stream);
    if (pages < old_pages) {
        auto* table = static_cast<std::int32_t*>(pool_->block_table_row(lane).data);
        CUDA_CHECK(cudaMemsetAsync(
            table + pages, 0xff, static_cast<std::size_t>(old_pages - pages) * sizeof(std::int32_t),
            stream));
    }
    state.pages.publish_mapping(stream);
    state.committed       = frontier;
    state.reserved_tokens = reserve_tokens;
}

void State::reset_sampling(std::int32_t lane, cudaStream_t stream) {
    static_cast<void>(open_lane(lane));
    CUDA_CHECK(cudaMemsetAsync(token_counts(lane), 0,
                               static_cast<std::size_t>(kVocabRows) * sizeof(std::int32_t),
                               stream));
}

PagedKVAllocation& State::pages(std::int32_t lane) {
    (void)open_lane(lane);
    return lane_states_.at(static_cast<std::size_t>(lane)).pages;
}

std::size_t State::continuation_bytes() const noexcept {
    return gdn_.conv_host_image_bytes() + gdn_.recurrent_host_image_bytes() +
           static_cast<std::size_t>(ops::kPleConvHistory) * ops::kPleChannels *
               sizeof(std::uint16_t) +
           static_cast<std::size_t>(kVocabRows) * sizeof(std::uint16_t) +
           (dflash_cache_ == nullptr ? 0
                                     : dflash_cache_->lane_host_bytes() +
                                           static_cast<std::size_t>(kDflashFeatures) *
                                               kDflashWidth * sizeof(std::uint16_t) +
                                           kDflashCountBytes);
}

Tensor State::tail_logits(std::int32_t lane) const {
    static_cast<void>(open_lane(lane));
    return tail_logits_.slice(1, lane, 1);
}

void State::pack_continuation(std::int32_t lane, void* destination, cudaStream_t stream) const {
    static_cast<void>(open_lane(lane));
    auto* bytes            = static_cast<std::byte*>(destination);
    const std::size_t conv = gdn_.conv_host_image_bytes();
    gdn_.pack_slot_to_host(lane, bytes, bytes + conv, stream);
    bytes += conv + gdn_.recurrent_host_image_bytes();
    const Tensor ple = ple_history_.slice(2, lane, 1);
    CUDA_CHECK(cudaMemcpyAsync(bytes, ple.data, ple.bytes(), cudaMemcpyDeviceToHost, stream));
    bytes += ple.bytes();
    const Tensor logits = tail_logits(lane);
    CUDA_CHECK(cudaMemcpyAsync(bytes, logits.data, logits.bytes(), cudaMemcpyDeviceToHost, stream));
    if (dflash_cache_) {
        bytes += logits.bytes();
        dflash_cache_->copy_lane_to_host(lane, bytes, stream);
        bytes += dflash_cache_->lane_host_bytes();
        const Tensor pending = dflash_pending_features_.slice(2, lane, 1);
        CUDA_CHECK(
            cudaMemcpyAsync(bytes, pending.data, pending.bytes(), cudaMemcpyDeviceToHost, stream));
        bytes += pending.bytes();
        std::memset(bytes, 0, kDflashCountBytes);
        const auto& metadata = open_lane(lane).dflash;
        std::memcpy(bytes, &metadata.pending, sizeof(metadata.pending));
        std::memcpy(bytes + 4, &metadata.draft_temperature, sizeof(metadata.draft_temperature));
        std::memcpy(bytes + 8, metadata.calibration_sum.data(), sizeof(metadata.calibration_sum));
        std::memcpy(bytes + 72, metadata.calibration_count.data(),
                    sizeof(metadata.calibration_count));
        // The fixed 256-byte metadata block includes the request-local policy, independently
        // of host struct padding. Its windows survive RAM/SSD transport and lane relocation.
        std::memcpy(bytes + 136, metadata.policy.reached.data(), sizeof(metadata.policy.reached));
        std::memcpy(bytes + 148, metadata.policy.accepted.data(), sizeof(metadata.policy.accepted));
        std::memcpy(bytes + 160, metadata.policy.low_windows.data(),
                    sizeof(metadata.policy.low_windows));
        std::memcpy(bytes + 172, &metadata.policy.width, sizeof(metadata.policy.width));
        std::memcpy(bytes + 176, &metadata.policy.ceiling, sizeof(metadata.policy.ceiling));
    }
}

void State::unpack_continuation(std::int32_t lane, const void* source, cudaStream_t stream) {
    static_cast<void>(open_lane(lane));
    const auto* bytes = static_cast<const std::byte*>(source);
    DflashLaneState restored;
    if (dflash_cache_) {
        const auto* header = bytes + continuation_bytes() - kDflashCountBytes;
        std::memcpy(&restored.pending, header, sizeof(restored.pending));
        std::memcpy(&restored.draft_temperature, header + 4, sizeof(restored.draft_temperature));
        std::memcpy(restored.calibration_sum.data(), header + 8, sizeof(restored.calibration_sum));
        std::memcpy(restored.calibration_count.data(), header + 72,
                    sizeof(restored.calibration_count));
        std::memcpy(restored.policy.reached.data(), header + 136, sizeof(restored.policy.reached));
        std::memcpy(restored.policy.accepted.data(), header + 148,
                    sizeof(restored.policy.accepted));
        std::memcpy(restored.policy.low_windows.data(), header + 160,
                    sizeof(restored.policy.low_windows));
        std::memcpy(&restored.policy.width, header + 172, sizeof(restored.policy.width));
        std::memcpy(&restored.policy.ceiling, header + 176, sizeof(restored.policy.ceiling));
        if (restored.pending < 0 || restored.pending > kDflashWidth || restored.policy.width < 0 ||
            restored.policy.width > kDflashMaxK || restored.policy.ceiling < 0 ||
            restored.policy.ceiling > kDflashMaxK || !std::isfinite(restored.draft_temperature) ||
            !std::all_of(restored.calibration_sum.begin(), restored.calibration_sum.end(),
                         [](double value) { return std::isfinite(value) && value >= 0.0; })) {
            throw std::invalid_argument(
                "Flash-Next continuation has an invalid DFlash pending count");
        }
    }
    const std::size_t conv = gdn_.conv_host_image_bytes();
    gdn_.unpack_slot_from_host(lane, bytes, bytes + conv, stream);
    bytes += conv + gdn_.recurrent_host_image_bytes();
    const Tensor ple = ple_history_.slice(2, lane, 1);
    CUDA_CHECK(cudaMemcpyAsync(ple.data, bytes, ple.bytes(), cudaMemcpyHostToDevice, stream));
    bytes += ple.bytes();
    const Tensor logits = tail_logits(lane);
    CUDA_CHECK(cudaMemcpyAsync(logits.data, bytes, logits.bytes(), cudaMemcpyHostToDevice, stream));
    if (dflash_cache_) {
        bytes += logits.bytes();
        dflash_cache_->copy_lane_from_host(bytes, lane, stream);
        bytes += dflash_cache_->lane_host_bytes();
        const Tensor pending = dflash_pending_features_.slice(2, lane, 1);
        CUDA_CHECK(
            cudaMemcpyAsync(pending.data, bytes, pending.bytes(), cudaMemcpyHostToDevice, stream));
        lane_states_.at(static_cast<std::size_t>(lane)).dflash = restored;
    }
}

void State::copy_prefix(std::int32_t destination, std::int32_t source, std::int32_t frontier,
                        cudaStream_t stream) {
    const auto& from = open_lane(source);
    static_cast<void>(open_lane(destination));
    if (destination == source || frontier < 1 || frontier > from.committed) {
        throw std::out_of_range("Flash-Next prefix copy has an invalid source extent");
    }
    auto& to = lane_states_.at(static_cast<std::size_t>(destination));
    copy_paged_kv_allocation(from.pages, to.pages, *pool_,
                             static_cast<std::uint32_t>(pages_of(frontier)), stream);
    gdn_.copy_slot(source, destination, stream);
    const Tensor ple_from = ple_history_.slice(2, source, 1);
    const Tensor ple_to   = ple_history_.slice(2, destination, 1);
    CUDA_CHECK(cudaMemcpyAsync(ple_to.data, ple_from.data, ple_from.bytes(),
                               cudaMemcpyDeviceToDevice, stream));
    const Tensor logits_from = tail_logits(source);
    const Tensor logits_to   = tail_logits(destination);
    CUDA_CHECK(cudaMemcpyAsync(logits_to.data, logits_from.data, logits_from.bytes(),
                               cudaMemcpyDeviceToDevice, stream));
    if (dflash_cache_) {
        dflash_cache_->copy_lane_from(*dflash_cache_, source, destination, stream);
        const Tensor from_features = dflash_pending_features_.slice(2, source, 1);
        const Tensor to_features   = dflash_pending_features_.slice(2, destination, 1);
        CUDA_CHECK(cudaMemcpyAsync(to_features.data, from_features.data, from_features.bytes(),
                                   cudaMemcpyDeviceToDevice, stream));
        to.dflash = from.dflash;
    }
    to.committed = frontier;
}

DflashLaneState& State::dflash_lane(std::int32_t lane) {
    static_cast<void>(open_lane(lane));
    if (!dflash_cache_) { throw std::logic_error("Flash-Next DFlash state is disabled"); }
    return lane_states_.at(static_cast<std::size_t>(lane)).dflash;
}

void State::set_dflash_pending_count(std::int32_t lane, std::int32_t count) {
    if (!dflash_cache_ || count < 0 || count > kDflashWidth || count > committed(lane)) {
        throw std::invalid_argument("Flash-Next DFlash pending prefix is invalid");
    }
    lane_states_.at(static_cast<std::size_t>(lane)).dflash.pending = count;
}

Tensor State::gdn_convolution_pool(std::int32_t gdn_layer) const {
    return gdn_.conv.at(static_cast<std::size_t>(gdn_layer));
}

Tensor State::gdn_recurrent_pool(std::int32_t gdn_layer) const {
    return gdn_.recurrent.at(static_cast<std::size_t>(gdn_layer));
}

Tensor State::gdn_convolution(std::int32_t gdn_layer, std::int32_t lane) const {
    return gdn_.conv_slot(static_cast<std::uint32_t>(gdn_layer), lane);
}

Tensor State::gdn_recurrent(std::int32_t gdn_layer, std::int32_t lane) const {
    return gdn_.recurrent_slot(static_cast<std::uint32_t>(gdn_layer), lane);
}

ops::QsaPagedState& State::qsa(std::int32_t qsa_layer) {
    return qsa_.at(static_cast<std::size_t>(qsa_layer));
}

Tensor State::lane_id(std::int32_t lane) const {
    if (lane < 0 || lane >= lanes_) {
        throw std::out_of_range("qwen3_8_flash_next State: lane out of range");
    }
    return Tensor(lane_ids_.slice(1, lane, 1).data, DType::I32, {1});
}

std::int32_t* State::token_counts(std::int32_t lane) const {
    static_cast<void>(lane_id(lane)); // validates the lane
    return static_cast<std::int32_t*>(token_counts_.slice(1, lane, 1).data);
}

std::uint32_t* State::allowed_tokens(std::int32_t lane) const {
    static_cast<void>(lane_id(lane)); // validates the lane
    return static_cast<std::uint32_t*>(allowed_.slice(1, lane, 1).data);
}

} // namespace ninfer::targets::qwen3_8_flash_next
