#include "core/gdn_history.h"

#include <array>
#include <limits>
#include <stdexcept>

namespace ninfer {
namespace {
constexpr std::size_t kAlignment = 256;

std::int32_t checked_outer(const GdnHistorySpec& spec) {
    if (spec.layers <= 0 || spec.slots <= 0 || spec.slots > 6 || spec.width <= 0 ||
        spec.width > 16 || spec.capacity <= 0 || spec.qk_heads <= 0 || spec.value_heads <= 0 ||
        spec.key_dim <= 0 || spec.value_dim <= 0 || spec.value_heads % spec.qk_heads != 0) {
        throw std::invalid_argument("GDN history physical dimensions are invalid");
    }
    const std::int64_t outer = static_cast<std::int64_t>(spec.layers) * spec.slots;
    if (outer > std::numeric_limits<std::int32_t>::max()) {
        throw std::overflow_error("GDN history outer extent exceeds int32");
    }
    return static_cast<std::int32_t>(outer);
}

void require_region(const TensorRegion& region, DType dtype,
                    const std::array<std::int32_t, 4>& shape) {
    const Tensor expected(nullptr, dtype, {shape[0], shape[1], shape[2], shape[3]});
    if (region.dtype != dtype || region.shape != shape || region.region.alignment != kAlignment ||
        region.region.bytes != expected.bytes()) {
        throw std::logic_error("GDN history plane disagrees with its layout");
    }
}

void validate_layout(const GdnHistoryLayout& layout) {
    const auto& spec = layout.spec;
    const auto outer = checked_outer(spec);
    require_region(layout.key, DType::FP32, {spec.key_dim, spec.qk_heads, spec.capacity, outer});
    require_region(layout.innovation, DType::FP32,
                   {spec.value_dim, spec.value_heads, spec.capacity, outer});
    require_region(layout.alpha, DType::FP32, {spec.value_heads, spec.capacity, outer, 1});
    require_region(layout.counts, DType::I32, {spec.slots, 1, 1, 1});
    require_region(layout.provisional_key, DType::FP32,
                   {spec.key_dim, spec.qk_heads, spec.width, outer});
    require_region(layout.provisional_innovation, DType::FP32,
                   {spec.value_dim, spec.value_heads, spec.width, outer});
    require_region(layout.provisional_alpha, DType::FP32, {spec.value_heads, spec.width, outer, 1});
    const TensorRegion* regions[]{&layout.key,
                                  &layout.innovation,
                                  &layout.alpha,
                                  &layout.counts,
                                  &layout.provisional_key,
                                  &layout.provisional_innovation,
                                  &layout.provisional_alpha};
    for (std::size_t i = 0; i < std::size(regions); ++i) {
        const auto& a = regions[i]->region;
        if (a.bytes > std::numeric_limits<std::size_t>::max() - a.offset) {
            throw std::overflow_error("GDN history plane extent overflows size_t");
        }
        for (std::size_t j = 0; j < i; ++j) {
            const auto& b = regions[j]->region;
            if (a.offset < b.offset + b.bytes && b.offset < a.offset + a.bytes) {
                throw std::logic_error("GDN history planes overlap");
            }
        }
    }
}
} // namespace

GdnHistoryLayout plan_gdn_history(LayoutBuilder& builder, const GdnHistorySpec& spec) {
    const auto outer = checked_outer(spec);
    return {
        .spec = spec,
        .key  = builder.add_tensor(DType::FP32, {spec.key_dim, spec.qk_heads, spec.capacity, outer},
                                   kAlignment, "GDN history keys"),
        .innovation = builder.add_tensor(DType::FP32,
                                         {spec.value_dim, spec.value_heads, spec.capacity, outer},
                                         kAlignment, "GDN history innovations"),
        .alpha      = builder.add_tensor(DType::FP32, {spec.value_heads, spec.capacity, outer},
                                         kAlignment, "GDN history decay"),
        .counts = builder.add_tensor(DType::I32, {spec.slots}, kAlignment, "GDN history lengths"),
        .provisional_key =
            builder.add_tensor(DType::FP32, {spec.key_dim, spec.qk_heads, spec.width, outer},
                               kAlignment, "GDN provisional keys"),
        .provisional_innovation =
            builder.add_tensor(DType::FP32, {spec.value_dim, spec.value_heads, spec.width, outer},
                               kAlignment, "GDN provisional innovations"),
        .provisional_alpha = builder.add_tensor(DType::FP32, {spec.value_heads, spec.width, outer},
                                                kAlignment, "GDN provisional decay"),
    };
}

GdnHistory::GdnHistory(DeviceSpan backing, const GdnHistoryLayout& layout)
    : key(layout.key.bind(backing)), innovation(layout.innovation.bind(backing)),
      alpha(layout.alpha.bind(backing)), counts(layout.counts.bind(backing)),
      provisional_key(layout.provisional_key.bind(backing)),
      provisional_innovation(layout.provisional_innovation.bind(backing)),
      provisional_alpha(layout.provisional_alpha.bind(backing)), spec(layout.spec) {
    validate_layout(layout);
}

GdnHistoryLayer GdnHistory::layer(std::int32_t index, std::int32_t row_begin,
                                  std::int32_t rows) const {
    (void)checked_outer(spec);
    if (index < 0 || index >= spec.layers || row_begin < 0 || rows <= 0 ||
        static_cast<std::int64_t>(row_begin) + rows > spec.slots) {
        throw std::out_of_range("GDN history layer/row window is invalid");
    }
    const auto first = index * spec.slots;
    return {
        .key                    = key.slice(3, first, spec.slots),
        .innovation             = innovation.slice(3, first, spec.slots),
        .alpha                  = alpha.slice(2, first, spec.slots),
        .counts                 = counts,
        .provisional_key        = provisional_key.slice(3, first + row_begin, rows),
        .provisional_innovation = provisional_innovation.slice(3, first + row_begin, rows),
        .provisional_alpha      = provisional_alpha.slice(2, first + row_begin, rows),
    };
}
} // namespace ninfer
