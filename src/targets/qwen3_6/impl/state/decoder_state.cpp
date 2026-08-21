#include <ninfer/targets/qwen3_6/decoder_state.h>

#include <limits>
#include <stdexcept>

namespace ninfer::targets::qwen3_6 {
namespace {

std::uint32_t page_count(std::uint32_t capacity) {
    if (capacity == 0) { throw std::invalid_argument("Paged KV capacity must be positive"); }
    return 1U + (capacity - 1U) / static_cast<std::uint32_t>(kPagedKVPageSize);
}

PagedKVCacheLayout plan_cache(LayoutBuilder& builder, std::uint32_t layers, std::uint32_t capacity,
                               std::int32_t kv_heads, std::int32_t head_dim, DType dtype,
                               std::int32_t quant_group, std::int32_t table_rows,
                               std::uint32_t physical_page_groups, bool sage_pv) {
    if (layers == 0 ||
        layers > static_cast<std::uint32_t>(std::numeric_limits<std::int32_t>::max()) ||
        kv_heads <= 0 || head_dim <= 0 || table_rows <= 0) {
        throw std::invalid_argument("Paged KV cache geometry is invalid");
    }
    const bool int8  = dtype == DType::I8;
    const bool nvfp4 = dtype == DType::U8;
    if ((!int8 && !nvfp4 && (dtype != DType::BF16 || quant_group != 0)) ||
        (int8 && (quant_group != kKvQuantGroup || head_dim % quant_group != 0)) ||
        (nvfp4 && (quant_group != kKvNvfp4Group || head_dim % quant_group != 0 ||
                   (head_dim % 2) != 0))) {
        throw std::invalid_argument("Paged KV cache dtype or quantization is invalid");
    }

    const std::uint32_t logical_pages = page_count(capacity);
    if (physical_page_groups < logical_pages) {
        throw std::invalid_argument("Paged KV physical pages are below logical capacity");
    }

    PagedKVPoolSpec pool_spec;
    pool_spec.page_group_count      = physical_page_groups;
    pool_spec.logical_page_capacity = logical_pages;
    pool_spec.table_rows            = table_rows;
    const bool quantized = int8 || nvfp4;
    pool_spec.planes.reserve(static_cast<std::size_t>(layers) * (quantized ? 4ULL : 2ULL));
    for (std::uint32_t layer = 0; layer < layers; ++layer) {
        const std::int32_t code_extent = nvfp4 ? head_dim / 2 : head_dim;
        pool_spec.planes.push_back({dtype, code_extent, kv_heads, 256});
        pool_spec.planes.push_back({dtype, code_extent, kv_heads, 256});
        if (int8) {
            pool_spec.planes.push_back({DType::FP16, head_dim / quant_group, kv_heads, 256});
            pool_spec.planes.push_back({DType::FP16, head_dim / quant_group, kv_heads, 256});
        } else if (nvfp4) {
            pool_spec.planes.push_back({DType::FP8_E4M3FN, head_dim / quant_group, kv_heads, 256});
            pool_spec.planes.push_back({DType::FP8_E4M3FN, head_dim / quant_group, kv_heads, 256});
            if (sage_pv) {
                // Per-page (64-key tile) dequantized K sum for the meansim tile proxy:
                // d -> page_offset=d/4, leading=d%4 (4 floats). F32.
                pool_spec.planes.push_back({DType::FP32, 4, kv_heads, 256});
            }
        }
    }
    return PagedKVCacheLayout{
        .pool        = plan_paged_kv_pool(builder, pool_spec),
        .layers      = layers,
        .max_context = capacity,
        .kv_heads    = kv_heads,
        .head_dim    = head_dim,
        .dtype       = dtype,
        .quant_group = quant_group,
        .sage_pv     = sage_pv,
    };
}

} // namespace

DecoderStateLayout plan_decoder_state(LayoutBuilder& builder, const DecoderStateSpec& spec) {
    DecoderStateLayout layout;
    layout.text_kv = plan_cache(builder, spec.full_attention_layers, spec.capacity, spec.kv_heads,
                                spec.attention_head_dim, spec.kv_dtype, spec.kv_quant_group,
                                spec.kv_table_rows, spec.text_physical_page_groups,
                                spec.sage_attn);
    if (spec.enable_mtp) {
        layout.mtp_kv = plan_cache(builder, spec.mtp_layers, spec.capacity, spec.kv_heads,
                                   spec.attention_head_dim, spec.kv_dtype, spec.kv_quant_group,
                                   spec.kv_table_rows, spec.mtp_physical_page_groups,
                                   spec.sage_attn);
    }
    layout.linear_attention = plan_linear_attention_state_pool(builder, spec.linear_attention);
    return layout;
}

PagedKVCache::PagedKVCache(DeviceSpan backing, const PagedKVCacheLayout& layout)
    : pool_(backing, layout.pool), layers_(layout.layers), max_context_(layout.max_context),
      kv_heads_(layout.kv_heads), head_dim_(layout.head_dim), dtype_(layout.dtype),
      quant_group_(layout.quant_group), sage_pv_(layout.sage_pv) {}

PagedKVCacheView::PagedKVCacheView(const PagedKVCache& cache, Tensor block_table) noexcept
    : cache_(&cache), block_table_(block_table) {}

std::uint32_t PagedKVCacheView::max_context() const noexcept {
    return cache_ == nullptr ? 0 : cache_->max_context();
}

PagedKVLayerView PagedKVCacheView::layer_view(std::uint32_t layer) const {
    if (cache_ == nullptr) { throw std::logic_error("Paged KV execution view is empty"); }
    return cache_->layer_view(layer, block_table_);
}

PagedKVCacheView PagedKVCache::execution_view(const PagedKVAllocation& allocation) const {
    if (!allocation.belongs_to(pool_)) {
        throw std::invalid_argument("Paged KV allocation belongs to another cache pool");
    }
    return PagedKVCacheView(*this, allocation.block_table());
}

PagedKVLayerView PagedKVCache::layer_view(std::uint32_t layer, Tensor block_table) const {
    if (layer >= layers_) { throw std::out_of_range("Paged KV layer is out of range"); }
    const bool quantized     = dtype_ == DType::I8 || dtype_ == DType::U8;
    // A sage (U8 + sage_pv) layer stores 5 planes (k, v, k_scale, v_scale, k_mean); other
    // layouts store 4 (quantized) or 2 (BF16). The flat pool is laid out per layer, so the
    // base offset must stride by the per-layer plane count, not a fixed 4.
    const bool nvfp4_sage    = quantized && dtype_ == DType::U8 && sage_pv_;
    const std::size_t stride = nvfp4_sage ? 5ULL : (quantized ? 4ULL : 2ULL);
    const std::size_t base   = static_cast<std::size_t>(layer) * stride;
    return PagedKVLayerView{
        .k_pages       = pool_.plane(base),
        .v_pages       = pool_.plane(base + 1),
        .k_scale_pages = quantized ? pool_.plane(base + 2) : Tensor(),
        .v_scale_pages = quantized ? pool_.plane(base + 3) : Tensor(),
        .k_mean_pages  = nvfp4_sage ? pool_.plane(base + 4) : Tensor(),
        .block_table   = block_table,
        .head_dim      = head_dim_,
        .num_kv_heads  = kv_heads_,
        .dtype         = dtype_,
        .quant_group   = quant_group_,
        .sage_pv       = sage_pv_,
    };
}

PagedKVBatchLayerView PagedKVCache::batch_layer_view(std::uint32_t layer) const {
    if (layer >= layers_) { throw std::out_of_range("Paged KV layer is out of range"); }
    const bool quantized     = dtype_ == DType::I8 || dtype_ == DType::U8;
    // A sage (U8 + sage_pv) layer stores 5 planes (k, v, k_scale, v_scale, k_mean); other
    // layouts store 4 (quantized) or 2 (BF16). The flat pool is laid out per layer, so the
    // base offset must stride by the per-layer plane count, not a fixed 4.
    const bool nvfp4_sage    = quantized && dtype_ == DType::U8 && sage_pv_;
    const std::size_t stride = nvfp4_sage ? 5ULL : (quantized ? 4ULL : 2ULL);
    const std::size_t base   = static_cast<std::size_t>(layer) * stride;
    return PagedKVBatchLayerView{
        .k_pages       = pool_.plane(base),
        .v_pages       = pool_.plane(base + 1),
        .k_scale_pages = quantized ? pool_.plane(base + 2) : Tensor(),
        .v_scale_pages = quantized ? pool_.plane(base + 3) : Tensor(),
        .k_mean_pages  = nvfp4_sage ? pool_.plane(base + 4) : Tensor(),
        .block_tables  = pool_.block_tables(),
        .head_dim      = head_dim_,
        .num_kv_heads  = kv_heads_,
        .dtype         = dtype_,
        .quant_group   = quant_group_,
        .sage_pv       = sage_pv_,
    };
}

std::size_t DecoderStateLayout::kv_payload_bytes() const noexcept {
    return text_kv.payload_bytes() + (mtp_kv ? mtp_kv->payload_bytes() : 0);
}

DecoderState::DecoderState(DeviceSpan backing, const DecoderStateLayout& layout)
    : text_kv(backing, layout.text_kv), linear_attention(backing, layout.linear_attention) {
    if (layout.mtp_kv) { mtp_kv.emplace(backing, *layout.mtp_kv); }
}

PagedKVCache* DecoderState::mtp_cache() noexcept { return mtp_kv ? &*mtp_kv : nullptr; }

const PagedKVCache* DecoderState::mtp_cache() const noexcept { return mtp_kv ? &*mtp_kv : nullptr; }

} // namespace ninfer::targets::qwen3_6
