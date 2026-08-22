#include "core/arena.h"
#include "core/paged_kv_cache.h"
#include "ninfer/ops/gqa_attention.h"
#include "ops/op_tester.h"

#include <cuda_fp4.h>
#include <cuda_fp8.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <span>
#include <sstream>
#include <string>
#include <vector>

using namespace ninfer;
using namespace ninfer::test;

namespace {

constexpr std::int32_t kHeadDim         = 256;
constexpr std::int32_t kQuantGroup      = 64;
constexpr std::int32_t kQuantGroups     = kHeadDim / kQuantGroup;
constexpr std::int32_t kNvfp4Group      = 16;
constexpr std::int32_t kNvfp4Groups     = kHeadDim / kNvfp4Group;
constexpr std::int32_t kNvfp4CodeWidth  = kHeadDim / 2;
constexpr float kAttentionScale         = 0.0625f;
constexpr std::uint16_t kOutputCanary = 0x7fc1u;

// The Op has two registered compute profiles. A1 and A3 use the same criterion for a given
// profile; token count, geometry, execution envelope, and private launch route do not select it.
constexpr ReductionCriterion kAttentionBf16Criterion{
    /*relative_l2*/ 2.8e-3,
    /*gross_absolute*/ 1.0e-3,
    /*gross_relative_to_max_reference*/ 2.7e-3,
};

constexpr ReductionCriterion kAttentionInt8Criterion{
    /*relative_l2*/ 3.15e-3,
    /*gross_absolute*/ 1.1e-3,
    /*gross_relative_to_max_reference*/ 2.2e-3,
};

constexpr ReductionCriterion kAttentionNvfp4Criterion{
    /*relative_l2*/ 1.2e-2,
    /*gross_absolute*/ 4.0e-3,
    /*gross_relative_to_max_reference*/ 8.0e-3,
};

// The sage (SageAttention3-style) route adds FP4 P-quantization on top of the NVFP4 cache.
// The FP4 P-quant floor (rel_L2 between the P-quant-emulated host reference and the exact
// host reference) is 0.053-0.059 across the conformance cases, and a correct kernel sits at
// ~sqrt(2)*floor from the *independent* host emulation (independent e4m3/e2m1 rounding
// events), i.e. ~0.075 rel_L2. The max-point (gross) error is correspondingly inflated by the
// same independent-rounding distance (~0.19 worst case at max_ref~1.06). So this is NOT a tight
// conformance bound; it is a "correct, no-residual-bug" gate set above the FP4-P floor.
constexpr ReductionCriterion kAttentionNvfp4s3Criterion{
    /*relative_l2*/ 9.0e-2,
    /*gross_absolute*/ 8.0e-2,
    /*gross_relative_to_max_reference*/ 1.6e-1,
};

struct Geometry {
    const char* name;
    std::int32_t q_heads;
    std::int32_t kv_heads;

    [[nodiscard]] std::int32_t query_group() const { return q_heads / kv_heads; }
};

constexpr Geometry kGeometries[] = {
    {"qwen3_6_27b", 24, 4},
    {"qwen3_6_35b_a3b", 16, 2},
};

struct AttentionCase {
    std::int32_t tokens;
    std::int32_t base;
    std::uint32_t envelope_max;
    std::uint32_t seed;
};

enum class MappingPattern { Identity, Offset, Fragmented };

const char* mapping_name(MappingPattern pattern) {
    switch (pattern) {
    case MappingPattern::Identity:
        return "identity";
    case MappingPattern::Offset:
        return "offset";
    case MappingPattern::Fragmented:
        return "fragmented";
    }
    return "unknown";
}

std::int32_t align_up_page(std::int32_t value) {
    constexpr std::int32_t kFixtureAlignment = 2 * kPagedKVPageSize;
    return ((value + kFixtureAlignment - 1) / kFixtureAlignment) * kFixtureAlignment;
}

std::int32_t physical_page_count(std::int32_t logical_pages, MappingPattern pattern) {
    switch (pattern) {
    case MappingPattern::Identity:
        return logical_pages;
    case MappingPattern::Offset:
        return logical_pages + 2;
    case MappingPattern::Fragmented:
        return 2 * logical_pages + 1;
    }
    return 0;
}

std::vector<std::int32_t> make_block_table(std::int32_t logical_pages, MappingPattern pattern) {
    std::vector<std::int32_t> table(static_cast<std::size_t>(logical_pages));
    switch (pattern) {
    case MappingPattern::Identity:
        for (std::int32_t page = 0; page < logical_pages; ++page) { table[page] = page; }
        break;
    case MappingPattern::Offset:
        for (std::int32_t page = 0; page < logical_pages; ++page) { table[page] = page + 1; }
        break;
    case MappingPattern::Fragmented:
        for (std::int32_t page = 0; page < logical_pages; ++page) { table[page] = 2 * page + 1; }
        break;
    }
    return table;
}

std::size_t q_index(const Geometry& geometry, std::int32_t head, std::int32_t d,
                    std::int32_t token) {
    return static_cast<std::size_t>(d) +
           static_cast<std::size_t>(kHeadDim) *
               (static_cast<std::size_t>(head) +
                static_cast<std::size_t>(geometry.q_heads) * static_cast<std::size_t>(token));
}

std::size_t kv_input_index(const Geometry& geometry, std::int32_t head, std::int32_t d,
                           std::int32_t token) {
    return static_cast<std::size_t>(d) +
           static_cast<std::size_t>(kHeadDim) *
               (static_cast<std::size_t>(head) +
                static_cast<std::size_t>(geometry.kv_heads) * static_cast<std::size_t>(token));
}

std::size_t cache_index(const Geometry& geometry, std::int32_t padded_context, std::int32_t head,
                        std::int32_t position, std::int32_t d) {
    return static_cast<std::size_t>(d) +
           static_cast<std::size_t>(kHeadDim) *
               (static_cast<std::size_t>(position) +
                static_cast<std::size_t>(padded_context) * static_cast<std::size_t>(head));
}

std::size_t scale_index(const Geometry& geometry, std::int32_t padded_context, std::int32_t head,
                        std::int32_t position, std::int32_t group) {
    (void)geometry;
    return static_cast<std::size_t>(group) +
           static_cast<std::size_t>(kQuantGroups) *
               (static_cast<std::size_t>(position) +
                static_cast<std::size_t>(padded_context) * static_cast<std::size_t>(head));
}

std::size_t cache_elements(const Geometry& geometry, std::int32_t padded_context) {
    return static_cast<std::size_t>(kHeadDim) * static_cast<std::size_t>(padded_context) *
           static_cast<std::size_t>(geometry.kv_heads);
}

std::size_t nvfp4_code_index(const Geometry& geometry, std::int32_t padded_context,
                             std::int32_t head, std::int32_t position, std::int32_t byte) {
    (void)geometry;
    return static_cast<std::size_t>(byte) +
           static_cast<std::size_t>(kNvfp4CodeWidth) *
               (static_cast<std::size_t>(position) +
                static_cast<std::size_t>(padded_context) * static_cast<std::size_t>(head));
}

std::size_t nvfp4_scale_index(const Geometry& geometry, std::int32_t padded_context,
                              std::int32_t head, std::int32_t position, std::int32_t group) {
    (void)geometry;
    return static_cast<std::size_t>(group) +
           static_cast<std::size_t>(kNvfp4Groups) *
               (static_cast<std::size_t>(position) +
                static_cast<std::size_t>(padded_context) * static_cast<std::size_t>(head));
}

std::size_t nvfp4_code_elements(const Geometry& geometry, std::int32_t padded_context) {
    return static_cast<std::size_t>(kNvfp4CodeWidth) * static_cast<std::size_t>(padded_context) *
           static_cast<std::size_t>(geometry.kv_heads);
}

std::size_t nvfp4_scale_elements(const Geometry& geometry, std::int32_t padded_context) {
    return static_cast<std::size_t>(kNvfp4Groups) * static_cast<std::size_t>(padded_context) *
           static_cast<std::size_t>(geometry.kv_heads);
}

// Sage (SageAttention3-style) V scale plane: one e4m3 per (d, 16-key block) in d-major
// order within each page/head chunk of the NVFP4 scale plane (same 16-per-key extent):
// device plane element (page, head) offset = d * 4 + key_block. The host vector is the
// (position, leading) layout that scatter_paged maps to the device plane, so the (page, d,
// key_block) slot lives at host position 64*page + (d*4+key_block)/16.
std::size_t sage_v_scale_host_index(std::int32_t padded_context, std::int32_t head, std::int32_t page,
                                    std::int32_t d, std::int32_t key_block) {
    const std::int32_t e = d * 4 + key_block;
    return static_cast<std::size_t>(16 * (64 * page + e / 16) + (e % 16)) +
           static_cast<std::size_t>(kNvfp4Groups) * static_cast<std::int32_t>(padded_context) *
           static_cast<std::int32_t>(head);
}

std::uint8_t encode_e2m1_rne(float value) {
    return static_cast<std::uint8_t>(__nv_cvt_float_to_fp4(value, __NV_E2M1, cudaRoundNearest)) &
           0x0fu;
}

std::uint8_t encode_e2m1_rne(float value);
std::uint8_t encode_e4m3fn_rne(float value);
double decode_e4m3fn_word(std::uint8_t word);

// Quantize the 16-key block (page, key_block) for one d-pair in the sage layout: one e4m3
// scale per d (over the 16 keys) and one e2m1 code per key at the NVFP4 code position.
void encode_sage_v_dp(const Geometry& geometry, std::int32_t padded_context, std::int32_t head,
                      std::int32_t page, std::int32_t key_block, std::int32_t dp,
                      const std::vector<float>& logical_v, std::vector<std::uint8_t>& codes,
                      std::vector<std::uint8_t>& scales) {
    const std::int32_t d0 = dp * 2;
    const std::int32_t key0 = page * kPagedKVPageSize + key_block * 16;
    float max0 = 0.0f;
    float max1 = 0.0f;
    for (std::int32_t j = 0; j < 16; ++j) {
        const std::size_t base = cache_index(geometry, padded_context, head, key0 + j, d0);
        max0 = std::max(max0, std::abs(logical_v[base]));
        max1 = std::max(max1, std::abs(logical_v[base + 1]));
    }
    const std::uint8_t sc0 = encode_e4m3fn_rne(max0 / 6.0f);
    const std::uint8_t sc1 = encode_e4m3fn_rne(max1 / 6.0f);
    const float dec0       = static_cast<float>(decode_e4m3fn_word(sc0));
    const float dec1       = static_cast<float>(decode_e4m3fn_word(sc1));
    scales[sage_v_scale_host_index(padded_context, head, page, d0, key_block)]      = sc0;
    scales[sage_v_scale_host_index(padded_context, head, page, d0 + 1, key_block)]  = sc1;
    for (std::int32_t j = 0; j < 16; ++j) {
        const std::size_t base = cache_index(geometry, padded_context, head, key0 + j, d0);
        const std::size_t code = nvfp4_code_index(geometry, padded_context, head, key0 + j, dp);
        const float x          = dec0 == 0.0f ? 0.0f : logical_v[base] / dec0;
        const float y          = dec1 == 0.0f ? 0.0f : logical_v[base + 1] / dec1;
        codes[code] =
            static_cast<std::uint8_t>((encode_e2m1_rne(x) & 0x0fu) |
                                      (static_cast<std::uint32_t>(encode_e2m1_rne(y) & 0x0fu) << 4));
    }
}

std::size_t scale_elements(const Geometry& geometry, std::int32_t padded_context) {
    return static_cast<std::size_t>(kQuantGroups) * static_cast<std::size_t>(padded_context) *
           static_cast<std::size_t>(geometry.kv_heads);
}

std::size_t paged_index(std::int32_t leading_extent, const Geometry& geometry,
                        std::int32_t physical_page, std::int32_t head, std::int32_t position,
                        std::int32_t leading) {
    return static_cast<std::size_t>(leading) +
           static_cast<std::size_t>(leading_extent) *
               (static_cast<std::size_t>(position % kPagedKVPageSize) +
                static_cast<std::size_t>(kPagedKVPageSize) *
                    (static_cast<std::size_t>(head) + static_cast<std::size_t>(geometry.kv_heads) *
                                                          static_cast<std::size_t>(physical_page)));
}

template <typename T>
std::vector<T> scatter_paged(const std::vector<T>& logical, std::int32_t leading_extent,
                             const Geometry& geometry, std::int32_t logical_capacity,
                             std::span<const std::int32_t> block_table,
                             std::int32_t physical_pages) {
    std::vector<T> physical(static_cast<std::size_t>(leading_extent) * kPagedKVPageSize *
                            static_cast<std::size_t>(geometry.kv_heads) *
                            static_cast<std::size_t>(physical_pages));
    for (std::int32_t head = 0; head < geometry.kv_heads; ++head) {
        for (std::int32_t position = 0; position < logical_capacity; ++position) {
            const std::int32_t page =
                block_table[static_cast<std::size_t>(position) / kPagedKVPageSize];
            for (std::int32_t leading = 0; leading < leading_extent; ++leading) {
                const std::size_t source = static_cast<std::size_t>(leading) +
                                           static_cast<std::size_t>(leading_extent) *
                                               (static_cast<std::size_t>(position) +
                                                static_cast<std::size_t>(logical_capacity) * head);
                physical[paged_index(leading_extent, geometry, page, head, position, leading)] =
                    logical[source];
            }
        }
    }
    return physical;
}

template <typename T>
void scatter_paged_into(const std::vector<T>& logical, std::int32_t leading_extent,
                        const Geometry& geometry, std::int32_t logical_capacity,
                        std::span<const std::int32_t> block_table, std::vector<T>& physical) {
    for (std::int32_t head = 0; head < geometry.kv_heads; ++head) {
        for (std::int32_t position = 0; position < logical_capacity; ++position) {
            const std::int32_t page =
                block_table[static_cast<std::size_t>(position) / kPagedKVPageSize];
            for (std::int32_t leading = 0; leading < leading_extent; ++leading) {
                const std::size_t source = static_cast<std::size_t>(leading) +
                                           static_cast<std::size_t>(leading_extent) *
                                               (static_cast<std::size_t>(position) +
                                                static_cast<std::size_t>(logical_capacity) * head);
                physical[paged_index(leading_extent, geometry, page, head, position, leading)] =
                    logical[source];
            }
        }
    }
}

template <typename T>
std::vector<T> gather_paged(std::span<const T> physical, std::int32_t leading_extent,
                            const Geometry& geometry, std::int32_t logical_capacity,
                            std::span<const std::int32_t> block_table) {
    std::vector<T> logical(static_cast<std::size_t>(leading_extent) * logical_capacity *
                           static_cast<std::size_t>(geometry.kv_heads));
    for (std::int32_t head = 0; head < geometry.kv_heads; ++head) {
        for (std::int32_t position = 0; position < logical_capacity; ++position) {
            const std::int32_t page =
                block_table[static_cast<std::size_t>(position) / kPagedKVPageSize];
            for (std::int32_t leading = 0; leading < leading_extent; ++leading) {
                const std::size_t target = static_cast<std::size_t>(leading) +
                                           static_cast<std::size_t>(leading_extent) *
                                               (static_cast<std::size_t>(position) +
                                                static_cast<std::size_t>(logical_capacity) * head);
                logical[target] =
                    physical[paged_index(leading_extent, geometry, page, head, position, leading)];
            }
        }
    }
    return logical;
}

std::vector<float> make_bf16_values(std::size_t count, std::uint32_t seed, float lo, float hi) {
    std::vector<float> values(count);
    fill_uniform(values, seed, lo, hi);
    round_to_bf16(values);
    return values;
}

std::vector<std::uint16_t> to_bf16_bits(const std::vector<float>& values) {
    std::vector<std::uint16_t> bits(values.size());
    for (std::size_t i = 0; i < values.size(); ++i) { bits[i] = f32_to_bf16(values[i]); }
    return bits;
}

std::vector<double> bf16_bits_to_double(const std::vector<std::uint16_t>& bits) {
    std::vector<double> values(bits.size());
    for (std::size_t i = 0; i < bits.size(); ++i) {
        values[i] = static_cast<double>(bf16_to_f32(bits[i]));
    }
    return values;
}

std::uint16_t f32_to_f16_bits(float value) {
    std::uint32_t bits = 0;
    std::memcpy(&bits, &value, sizeof(bits));

    const std::uint32_t sign = (bits >> 16) & 0x8000u;
    const std::uint32_t exp  = (bits >> 23) & 0xffu;
    std::uint32_t mantissa   = bits & 0x007fffffu;
    if (exp == 0xffu) {
        return static_cast<std::uint16_t>(sign | (mantissa == 0 ? 0x7c00u : 0x7e00u));
    }

    const int half_exp = static_cast<int>(exp) - 127 + 15;
    if (half_exp >= 31) { return static_cast<std::uint16_t>(sign | 0x7c00u); }
    if (half_exp <= 0) {
        if (half_exp < -10) { return static_cast<std::uint16_t>(sign); }
        mantissa |= 0x00800000u;
        const int shift             = 14 - half_exp;
        std::uint32_t half_mantissa = mantissa >> shift;
        const std::uint32_t halfway = 1u << (shift - 1);
        const std::uint32_t tail    = mantissa & ((1u << shift) - 1u);
        if (tail > halfway || (tail == halfway && (half_mantissa & 1u) != 0u)) { ++half_mantissa; }
        return static_cast<std::uint16_t>(sign | half_mantissa);
    }

    std::uint32_t half_mantissa = mantissa >> 13;
    const std::uint32_t tail    = mantissa & 0x1fffu;
    std::uint32_t rounded_exp   = static_cast<std::uint32_t>(half_exp);
    if (tail > 0x1000u || (tail == 0x1000u && (half_mantissa & 1u) != 0u)) {
        ++half_mantissa;
        if (half_mantissa == 0x400u) {
            half_mantissa = 0;
            ++rounded_exp;
            if (rounded_exp >= 31) { return static_cast<std::uint16_t>(sign | 0x7c00u); }
        }
    }
    return static_cast<std::uint16_t>(sign | (rounded_exp << 10) | half_mantissa);
}

float f16_bits_to_f32(std::uint16_t bits) {
    const bool negative = (bits & 0x8000u) != 0;
    const int exp       = (bits >> 10) & 0x1f;
    const int mantissa  = bits & 0x03ff;
    float magnitude     = 0.0f;
    if (exp == 0) {
        magnitude = std::ldexp(static_cast<float>(mantissa), -24);
    } else if (exp == 31) {
        magnitude = mantissa == 0 ? std::numeric_limits<float>::infinity()
                                  : std::numeric_limits<float>::quiet_NaN();
    } else {
        magnitude = std::ldexp(1.0f + static_cast<float>(mantissa) / 1024.0f, exp - 15);
    }
    return negative ? -magnitude : magnitude;
}

std::int32_t round_even_to_i32(float value) {
    const float lower_f  = std::floor(value);
    const float fraction = value - lower_f;
    std::int32_t lower   = static_cast<std::int32_t>(lower_f);
    if (fraction < 0.5f) return lower;
    if (fraction > 0.5f) return lower + 1;
    return (lower & 1) == 0 ? lower : lower + 1;
}

struct HostCache {
    Geometry geometry;
    DType dtype;
    std::int32_t max_context;
    std::int32_t logical_capacity;
    bool sage = false;
    std::vector<std::uint16_t> k_bf16;
    std::vector<std::uint16_t> v_bf16;
    std::vector<std::int8_t> k_i8;
    std::vector<std::int8_t> v_i8;
    std::vector<std::uint16_t> k_scale;
    std::vector<std::uint16_t> v_scale;
    std::vector<std::uint8_t> k_u8;
    std::vector<std::uint8_t> v_u8;
    std::vector<std::uint8_t> k_fp8;
    std::vector<std::uint8_t> v_fp8;
};

void encode_group(const std::vector<float>& source, std::size_t source_base,
                  std::vector<std::int8_t>& codes, std::size_t code_base,
                  std::vector<std::uint16_t>& scales, std::size_t scale_offset) {
    float absmax = 0.0f;
    for (std::int32_t i = 0; i < kQuantGroup; ++i) {
        absmax = std::max(absmax, std::abs(source[source_base + static_cast<std::size_t>(i)]));
    }

    const float unrounded_scale    = absmax / 127.0f;
    const std::uint16_t scale_bits = f32_to_f16_bits(unrounded_scale);
    const float stored_scale       = f16_bits_to_f32(scale_bits);
    const float inverse_scale      = stored_scale == 0.0f ? 0.0f : 1.0f / stored_scale;
    scales[scale_offset]           = scale_bits;
    for (std::int32_t i = 0; i < kQuantGroup; ++i) {
        std::int32_t code = 0;
        if (stored_scale != 0.0f) {
            const float scaled = source[source_base + static_cast<std::size_t>(i)] * inverse_scale;
            code               = std::clamp(round_even_to_i32(scaled), -127, 127);
        }
        codes[code_base + static_cast<std::size_t>(i)] = static_cast<std::int8_t>(code);
    }
}

double decode_e2m1_word(std::uint8_t nibble) {
    constexpr double magnitudes[]{0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0};
    const double magnitude = magnitudes[nibble & 0x07u];
    return (nibble & 0x08u) == 0 ? magnitude : -magnitude;
}

double decode_e4m3fn_word(std::uint8_t word) {
    __nv_fp8_e4m3 value;
    value.__x = word;
    return static_cast<double>(static_cast<float>(value));
}

std::uint8_t encode_e4m3fn_rne(float value) {
    return static_cast<std::uint8_t>(__nv_cvt_float_to_fp8(value, __NV_SATFINITE, __NV_E4M3));
}

void encode_nvfp4_group(const std::vector<float>& source, std::size_t source_base,
                        std::vector<std::uint8_t>& codes, std::size_t code_base,
                        std::vector<std::uint8_t>& scales, std::size_t scale_offset) {
    float absmax = 0.0f;
    for (std::int32_t i = 0; i < kNvfp4Group; ++i) {
        absmax = std::max(absmax, std::abs(source[source_base + static_cast<std::size_t>(i)]));
    }
    const std::uint8_t scale_bits = encode_e4m3fn_rne(absmax / 6.0f);
    const float stored_scale      = static_cast<float>(decode_e4m3fn_word(scale_bits));
    scales[scale_offset]          = scale_bits;
    for (std::int32_t pair = 0; pair < kNvfp4Group / 2; ++pair) {
        float x = 0.0f;
        float y = 0.0f;
        if (stored_scale != 0.0f) {
            x = source[source_base + static_cast<std::size_t>(2 * pair)] / stored_scale;
            y = source[source_base + static_cast<std::size_t>(2 * pair + 1)] / stored_scale;
        }
        const float2 packed               = {x, y};
        codes[code_base + static_cast<std::size_t>(pair)] = __nv_cvt_float2_to_fp4x2(
            packed, __NV_E2M1, cudaRoundNearest);
    }
}

HostCache make_cache(const Geometry& geometry, DType dtype, std::int32_t max_context,
                     std::uint32_t seed, bool sage = false) {
    const std::int32_t logical_capacity = align_up_page(max_context);
    const std::size_t elements          = cache_elements(geometry, logical_capacity);
    std::vector<float> logical_k        = make_bf16_values(elements, seed, -0.25f, 0.25f);
    std::vector<float> logical_v        = make_bf16_values(elements, seed + 1u, -1.0f, 1.0f);

    HostCache cache{geometry, dtype, max_context, logical_capacity};
    cache.sage = sage;
    if (dtype == DType::BF16) {
        cache.k_bf16 = to_bf16_bits(logical_k);
        cache.v_bf16 = to_bf16_bits(logical_v);
        return cache;
    }
    if (dtype == DType::U8) {
        cache.k_u8.assign(nvfp4_code_elements(geometry, logical_capacity), 0);
        cache.v_u8.assign(nvfp4_code_elements(geometry, logical_capacity), 0);
        cache.k_fp8.assign(nvfp4_scale_elements(geometry, logical_capacity), 0);
        cache.v_fp8.assign(nvfp4_scale_elements(geometry, logical_capacity), 0);
        if (sage) {
            // K stays in the NVFP4 per-key-group layout; V moves to the sage d-major
            // per-(d, 16-key block) layout.
            for (std::int32_t head = 0; head < geometry.kv_heads; ++head) {
                for (std::int32_t position = 0; position < logical_capacity; ++position) {
                    for (std::int32_t group = 0; group < kNvfp4Groups; ++group) {
                        const std::int32_t d = group * kNvfp4Group;
                        const std::size_t source =
                            cache_index(geometry, logical_capacity, head, position, d);
                        const std::size_t code =
                            nvfp4_code_index(geometry, logical_capacity, head, position, group * 8);
                        const std::size_t scale =
                            nvfp4_scale_index(geometry, logical_capacity, head, position, group);
                        encode_nvfp4_group(logical_k, source, cache.k_u8, code, cache.k_fp8, scale);
                    }
                }
                for (std::int32_t page = 0; page < logical_capacity / kPagedKVPageSize; ++page) {
                    for (std::int32_t key_block = 0; key_block < kPagedKVPageSize / 16; ++key_block) {
                        for (std::int32_t dp = 0; dp < kNvfp4CodeWidth; ++dp) {
                            encode_sage_v_dp(geometry, logical_capacity, head, page, key_block, dp,
                                             logical_v, cache.v_u8, cache.v_fp8);
                        }
                    }
                }
            }
            return cache;
        }
        for (std::int32_t head = 0; head < geometry.kv_heads; ++head) {
            for (std::int32_t position = 0; position < logical_capacity; ++position) {
                for (std::int32_t group = 0; group < kNvfp4Groups; ++group) {
                    const std::int32_t d = group * kNvfp4Group;
                    const std::size_t source =
                        cache_index(geometry, logical_capacity, head, position, d);
                    const std::size_t code =
                        nvfp4_code_index(geometry, logical_capacity, head, position, group * 8);
                    const std::size_t scale =
                        nvfp4_scale_index(geometry, logical_capacity, head, position, group);
                    encode_nvfp4_group(logical_k, source, cache.k_u8, code, cache.k_fp8, scale);
                    encode_nvfp4_group(logical_v, source, cache.v_u8, code, cache.v_fp8, scale);
                }
            }
        }
        return cache;
    }

    cache.k_i8.assign(elements, 0);
    cache.v_i8.assign(elements, 0);
    const std::size_t scales = scale_elements(geometry, logical_capacity);
    cache.k_scale.assign(scales, 0);
    cache.v_scale.assign(scales, 0);
    for (std::int32_t head = 0; head < geometry.kv_heads; ++head) {
        for (std::int32_t position = 0; position < logical_capacity; ++position) {
            for (std::int32_t group = 0; group < kQuantGroups; ++group) {
                const std::int32_t d   = group * kQuantGroup;
                const std::size_t code = cache_index(geometry, logical_capacity, head, position, d);
                const std::size_t scale =
                    scale_index(geometry, logical_capacity, head, position, group);
                encode_group(logical_k, code, cache.k_i8, code, cache.k_scale, scale);
                encode_group(logical_v, code, cache.v_i8, code, cache.v_scale, scale);
            }
        }
    }
    return cache;
}

void append_cache(HostCache& cache, const std::vector<float>& k, const std::vector<float>& v,
                  const std::vector<std::int32_t>& positions) {
    const Geometry& geometry = cache.geometry;
    for (std::int32_t token = 0; token < static_cast<std::int32_t>(positions.size()); ++token) {
        const std::int32_t position = positions[static_cast<std::size_t>(token)];
        for (std::int32_t head = 0; head < geometry.kv_heads; ++head) {
            if (cache.dtype == DType::BF16) {
                for (std::int32_t d = 0; d < kHeadDim; ++d) {
                    const std::size_t source = kv_input_index(geometry, head, d, token);
                    const std::size_t target =
                        cache_index(geometry, cache.logical_capacity, head, position, d);
                    cache.k_bf16[target] = f32_to_bf16(k[source]);
                    cache.v_bf16[target] = f32_to_bf16(v[source]);
                }
                continue;
            }
            if (cache.dtype == DType::U8) {
                for (std::int32_t group = 0; group < kNvfp4Groups; ++group) {
                    const std::int32_t d     = group * kNvfp4Group;
                    const std::size_t source = kv_input_index(geometry, head, d, token);
                    const std::size_t target =
                        nvfp4_code_index(geometry, cache.logical_capacity, head, position, group * 8);
                    const std::size_t scale =
                        nvfp4_scale_index(geometry, cache.logical_capacity, head, position, group);
                    encode_nvfp4_group(k, source, cache.k_u8, target, cache.k_fp8, scale);
                    if (!cache.sage) {
                        encode_nvfp4_group(v, source, cache.v_u8, target, cache.v_fp8, scale);
                    }
                }
                if (cache.sage) {
                    // The sage V plane is filled single-shot per (d, 16-key block) after
                    // the token loop (mirrors the fill kernel contract); the per-token
                    // K encoding above already handled the sage K plane.
                    continue;
                }
                continue;
            }

            for (std::int32_t group = 0; group < kQuantGroups; ++group) {
                const std::int32_t d     = group * kQuantGroup;
                const std::size_t source = kv_input_index(geometry, head, d, token);
                const std::size_t target =
                    cache_index(geometry, cache.logical_capacity, head, position, d);
                const std::size_t scale =
                    scale_index(geometry, cache.logical_capacity, head, position, group);
                encode_group(k, source, cache.k_i8, target, cache.k_scale, scale);
                encode_group(v, source, cache.v_i8, target, cache.v_scale, scale);
            }
        }
    }
    if (cache.sage && cache.dtype == DType::U8 && !positions.empty()) {
        // Sage V fill, single-shot per (d, 16-key block) -- mirrors the fill kernel
    // contract exactly: the block scale is computed once as the max of (a) the
    // in-range new keys' per-d max and (b) the stored scale's implied max (s*6);
    // the new key codes are a single rounding at the final scale; the in-block
    // prefix codes (written by earlier appends under the stored scale) are
    // rescaled once with a single s_old/s_new ratio on a bump; out-of-range
    // in-block keys are never touched.
        const std::int32_t base = positions.front();
        for (std::size_t i = 0; i < positions.size(); ++i) {
            if (positions[static_cast<std::size_t>(i)] != base + static_cast<std::int32_t>(i)) {
                std::cerr << "SAGE single-shot oracle: append_cache expects contiguous "
                               "positions\n";
                std::abort();
            }
        }
        const std::int32_t count     = static_cast<std::int32_t>(positions.size());
        const std::int32_t first_blk = base / 16;
        const std::int32_t last_blk  = (base + count - 1) / 16;
        const std::int32_t blocks_per_page = kPagedKVPageSize / 16;
        for (std::int32_t blk = first_blk; blk <= last_blk; ++blk) {
            const std::int32_t blk_key0      = blk * 16;
            const std::int32_t inblock_begin = std::max(0, base - blk_key0);
            const std::int32_t inblock_end   = std::min(16, base + count - blk_key0);
            if (inblock_begin >= inblock_end) { continue; }
            const std::int32_t page      = blk / blocks_per_page;
            const std::int32_t key_block = blk % blocks_per_page;
            for (std::int32_t head = 0; head < geometry.kv_heads; ++head) {
                for (std::int32_t dp = 0; dp < kNvfp4CodeWidth; ++dp) {
                    const std::int32_t d0 = dp * 2;
                    const std::size_t sc0_off = sage_v_scale_host_index(
                        cache.logical_capacity, head, page, d0, key_block);
                    const std::size_t sc1_off =
                        sage_v_scale_host_index(cache.logical_capacity, head, page, d0 + 1,
                                               key_block);
                    const std::uint8_t scale_byte0 = cache.v_fp8[sc0_off];
                    const std::uint8_t scale_byte1 = cache.v_fp8[sc1_off];
                    const float s_cur0 = static_cast<float>(decode_e4m3fn_word(scale_byte0));
                    const float s_cur1 =
                        static_cast<float>(decode_e4m3fn_word(scale_byte1));
                    float v0s[16] = {0};
                    float v1s[16] = {0};
                    float m0 = 0.0f, m1 = 0.0f;
                    for (std::int32_t ki = inblock_begin; ki < inblock_end; ++ki) {
                        const std::int32_t token = blk_key0 + ki - base;
                        v0s[ki] = v[kv_input_index(geometry, head, d0, token)];
                        v1s[ki] = v[kv_input_index(geometry, head, d0 + 1, token)];
                        m0 = std::max(m0, std::abs(v0s[ki]));
                        m1 = std::max(m1, std::abs(v1s[ki]));
                    }
                    const float fin_max0 = std::max(m0, s_cur0 * 6.0f);
                    const float fin_max1 = std::max(m1, s_cur1 * 6.0f);
                    const std::uint8_t sc_new0 = encode_e4m3fn_rne(fin_max0 / 6.0f);
                    const std::uint8_t sc_new1 = encode_e4m3fn_rne(fin_max1 / 6.0f);
                    const float s_new0 = static_cast<float>(decode_e4m3fn_word(sc_new0));
                    const float s_new1 = static_cast<float>(decode_e4m3fn_word(sc_new1));
                    const bool bump0 = sc_new0 != scale_byte0;
                    const bool bump1 = sc_new1 != scale_byte1;
                    const float rescale0 =
                        (bump0 && s_cur0 > 0.0f && s_new0 > 0.0f) ? s_cur0 / s_new0 : 1.0f;
                    const float rescale1 =
                        (bump1 && s_cur1 > 0.0f && s_new1 > 0.0f) ? s_cur1 / s_new1 : 1.0f;
                    if (rescale0 != 1.0f || rescale1 != 1.0f) {
                        for (std::int32_t ki = 0; ki < inblock_begin; ++ki) {
                            const std::size_t code =
                                nvfp4_code_index(geometry, cache.logical_capacity, head,
                                                  blk_key0 + ki, dp);
                            const std::uint8_t byte = cache.v_u8[code];
                            const float lo =
                                (float)decode_e2m1_word(byte & 0x0fu) * rescale0;
                            const float hi =
                                (float)decode_e2m1_word((byte >> 4) & 0x0fu) * rescale1;
                            const float2 packed = {lo, hi};
                            cache.v_u8[code] =
                                __nv_cvt_float2_to_fp4x2(packed, __NV_E2M1, cudaRoundNearest);
                        }
                    }
                    for (std::int32_t ki = inblock_begin; ki < inblock_end; ++ki) {
                        const float c0 = (s_new0 > 0.0f) ? v0s[ki] / s_new0 : 0.0f;
                        const float c1 = (s_new1 > 0.0f) ? v1s[ki] / s_new1 : 0.0f;
                        const float2 packed = {c0, c1};
                        cache.v_u8[nvfp4_code_index(geometry, cache.logical_capacity, head,
                                                     blk_key0 + ki, dp)] =
                            __nv_cvt_float2_to_fp4x2(packed, __NV_E2M1, cudaRoundNearest);
                    }
                    if (bump0) { cache.v_fp8[sc0_off] = sc_new0; }
                    if (bump1) { cache.v_fp8[sc1_off] = sc_new1; }
                }
            }
        }
    }
}

double cache_value(const HostCache& cache, bool key, std::int32_t head, std::int32_t position,
                   std::int32_t d) {
    const std::size_t code = cache_index(cache.geometry, cache.logical_capacity, head, position, d);
    if (cache.dtype == DType::BF16) {
        return static_cast<double>(bf16_to_f32(key ? cache.k_bf16[code] : cache.v_bf16[code]));
    }
    if (cache.dtype == DType::U8) {
        const std::size_t packed =
            nvfp4_code_index(cache.geometry, cache.logical_capacity, head, position, d / 2);
        const std::size_t scale =
            !key && cache.sage
                ? sage_v_scale_host_index(cache.logical_capacity, head, position / kPagedKVPageSize,
                                          d, (position % kPagedKVPageSize) / 16)
                : nvfp4_scale_index(cache.geometry, cache.logical_capacity, head, position,
                                    d / kNvfp4Group);
        const auto& codes  = key ? cache.k_u8 : cache.v_u8;
        const auto& scales = key ? cache.k_fp8 : cache.v_fp8;
        const std::uint8_t byte   = codes[packed];
        const std::uint8_t nibble = (d & 1) == 0 ? (byte & 0x0fu) : (byte >> 4);
        return decode_e2m1_word(nibble) * decode_e4m3fn_word(scales[scale]);
    }

    const std::size_t scale =
        scale_index(cache.geometry, cache.logical_capacity, head, position, d / kQuantGroup);
    const auto& codes   = key ? cache.k_i8 : cache.v_i8;
    const auto& scales  = key ? cache.k_scale : cache.v_scale;
    const float decoded = static_cast<float>(codes[code]) * f16_bits_to_f32(scales[scale]);
    return static_cast<double>(decoded);
}

std::vector<double> q_nvfp4_log(const Geometry& geometry, const std::vector<float>& q,
                                 std::int32_t q_head, std::int32_t token) {
    // Emulate the kernel's NVFP4 Q-quantization: the QK mma runs on FP4 tensor
    // cores with per-16-d e4m3 scales + e2m1 codes (the same codec as the KV
    // cache), returning the dequantized q values the tensor cores see.
    std::vector<double> q_log(kHeadDim, 0.0);    for (std::int32_t grp = 0; grp < kHeadDim / kNvfp4Group; ++grp) {
        float absmax = 0.0f;
        for (std::int32_t i = 0; i < kNvfp4Group; ++i) {
            const std::int32_t d = grp * kNvfp4Group + i;
            absmax = std::max(absmax, std::abs(q[q_index(geometry, q_head, d, token)]));
        }
        const std::uint8_t scale_bits = encode_e4m3fn_rne(absmax / 6.0f);
        const float stored_scale = static_cast<float>(decode_e4m3fn_word(scale_bits));
        for (std::int32_t pair = 0; pair < kNvfp4Group / 2; ++pair) {
            const std::int32_t d0 = grp * kNvfp4Group + 2 * pair;
            const std::int32_t d1 = d0 + 1;
            float x = 0.0f;
            float y = 0.0f;
            if (stored_scale != 0.0f) {
                x = q[q_index(geometry, q_head, d0, token)] / stored_scale;
                y = q[q_index(geometry, q_head, d1, token)] / stored_scale;
            }
            const float2 packed = {x, y};
            const std::uint8_t code =
                __nv_cvt_float2_to_fp4x2(packed, __NV_E2M1, cudaRoundNearest);
            q_log[static_cast<std::size_t>(d0)] =
                static_cast<double>(decode_e2m1_word(code & 0x0Fu)) * stored_scale;
            q_log[static_cast<std::size_t>(d1)] =
                static_cast<double>(decode_e2m1_word((code >> 4) & 0x0Fu)) * stored_scale;
        }
    }
    return q_log;
}

// Step-exact emulation of the s3 kernel's online-softmax + FP4 P-quant loop in
// double precision: the ONLY discretization is the e4m3/e2m1 RNE codec rounding
// (matching the kernel's cvt). The running max updates per tile_keys keys (64
// prefill / 32 decode) and each 16-key P-block within a tile shares that running
// max — mirroring the kernel's S-argument exactly. A correct kernel must sit
// within ~1e-4 of this (its extra error is fp32 exp2/divide/accumulate noise);
// a residual of O(0.01+) localizes a real kernel bug.
void sage_orc_pdump(const std::string& label, const std::vector<float>& q, const HostCache& cache,
                    const std::vector<std::int32_t>& positions) {
    if (std::getenv("S3_ORC_DUMP") == nullptr) { return; }
    constexpr double kLog2E = 1.4426950408889634;
    constexpr double kAmp   = 2688.0;
    constexpr double kSMax  = 448.0;
    static const double kPTableEmu[8] = {0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0};
    const Geometry& geometry = cache.geometry;
    // CTA (q_head=0, q_block=0, kb=0) rows = the first 16 NEW tokens (local
    // index 0..15) of q_head 0; first tile = keys 0..63 (S-arg = tile max).
    for (std::int32_t t = 0; t < 16; ++t) {
        const std::int32_t q_head    = 0;
        const std::int32_t kv_head   = q_head / geometry.query_group();
        const std::int32_t q_token   = t;  // local new-token index (q tensor layout)
        const std::vector<double> q_log = q_nvfp4_log(geometry, q, q_head, q_token);
        std::vector<double> score(64, 0.0);
        double m_tile = -std::numeric_limits<double>::infinity();
        for (std::int32_t k = 0; k < 64; ++k) {
            double dot = 0.0;
            for (std::int32_t d = 0; d < kHeadDim; ++d) {
                dot += q_log[static_cast<std::size_t>(d)] * cache_value(cache, true, kv_head, k, d);
            }
            score[static_cast<std::size_t>(k)] = dot * static_cast<double>(kAttentionScale);
            m_tile                             = std::max(m_tile, score[static_cast<std::size_t>(k)]);
        }
        char line[1024] = "";
        std::snprintf(line, sizeof(line), "S3ORC %s h0 t%02d: m=%.4f s0=%.4f s15=%.4f s16=%.4f s63=%.4f q0=%.4f k0d0=%.4f |",
                      label.c_str(), t, m_tile, score[0], score[15], score[16], score[63], q_log[0],
                      cache_value(cache, true, kv_head, 0, 0));
        std::fprintf(stderr, "%s", line);
        for (std::int32_t nb = 0; nb < 4; ++nb) {
            double block_max = -std::numeric_limits<double>::infinity();
            for (std::int32_t k = 16 * nb; k < 16 * nb + 16; ++k) {
                block_max = std::max(block_max, score[static_cast<std::size_t>(k)]);
            }
            double s_arg = 0.0;
            if (block_max > -std::numeric_limits<double>::infinity()) {
                s_arg = kSMax * std::exp2((block_max - m_tile) * kLog2E);
            }
            const uint8_t s_bits = static_cast<uint8_t>(encode_e4m3fn_rne(static_cast<float>(s_arg)));
            const double dec_s   = decode_e4m3fn_word(s_bits);
            uint8_t bytes[8];
            for (std::int32_t j = 0; j < 8; ++j) {
                double ratio0 = 0.0, ratio1 = 0.0;
                const std::int32_t k0 = 16 * nb + 2 * j, k1 = k0 + 1;
                if (score[static_cast<std::size_t>(k0)] > -1e300) {
                    ratio0 = std::min(6.0, kAmp * std::exp2((score[static_cast<std::size_t>(k0)] - m_tile) * kLog2E) / dec_s);
                }
                if (score[static_cast<std::size_t>(k1)] > -1e300) {
                    ratio1 = std::min(6.0, kAmp * std::exp2((score[static_cast<std::size_t>(k1)] - m_tile) * kLog2E) / dec_s);
                }
                const uint8_t c0 = encode_e2m1_rne(static_cast<float>(ratio0)) & 0x0Fu;
                const uint8_t c1 = encode_e2m1_rne(static_cast<float>(ratio1)) & 0x0Fu;
                bytes[j]         = static_cast<uint8_t>((c1 << 4) | c0);  // even key = low nibble
            }
            char seg[96] = "";
            std::snprintf(seg, sizeof(seg), " b%d c=%02x%02x%02x%02x%02x%02x%02x%02x s=%02x", nb, bytes[0],
                          bytes[1], bytes[2], bytes[3], bytes[4], bytes[5], bytes[6], bytes[7],
                          s_bits);
            std::fprintf(stderr, "%s", seg);
        }
        std::fprintf(stderr, "\n");
    }
}

std::vector<double> sage_step_emulation(const std::vector<float>& q, const HostCache& cache,
                                        const std::vector<std::int32_t>& positions,
                                        std::int32_t tile_keys) {
    constexpr double kLog2E = 1.4426950408889634;
    constexpr double kAmp   = 2688.0;
    constexpr double kSMax  = 448.0;
    static const double kPTableEmu[8] = {0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0};
    const Geometry& geometry  = cache.geometry;
    const std::int32_t tokens = static_cast<std::int32_t>(positions.size());
    std::vector<double> output(static_cast<std::size_t>(kHeadDim) *
                               static_cast<std::size_t>(geometry.q_heads) *
                               static_cast<std::size_t>(tokens));

    for (std::int32_t token = 0; token < tokens; ++token) {
        const std::int32_t visible = positions[static_cast<std::size_t>(token)] + 1;
        for (std::int32_t q_head = 0; q_head < geometry.q_heads; ++q_head) {
            const std::int32_t kv_head = q_head / geometry.query_group();
            const std::vector<double> q_log = q_nvfp4_log(geometry, q, q_head, token);

            std::vector<double> score(visible, 0.0);
            double m_final              = -std::numeric_limits<double>::infinity();
            for (std::int32_t position = 0; position < visible; ++position) {
                double dot = 0.0;
                for (std::int32_t d = 0; d < kHeadDim; ++d) {
                    dot += q_log[static_cast<std::size_t>(d)] *
                           cache_value(cache, true, kv_head, position, d);
                }
                score[static_cast<std::size_t>(position)] =
                    dot * static_cast<double>(kAttentionScale);
                m_final = std::max(m_final, score[static_cast<std::size_t>(position)]);
            }

            // Per 16-key block: S-arg from the tile running max (kernel-faithful).
            struct PBlock {
                double dec_s;
                double p[16];  // quantized, amplified, relative to the tile max
            };
            const std::int32_t blocks = (visible + 15) / 16;
            std::vector<PBlock> pb(blocks);
            double running_m = -std::numeric_limits<double>::infinity();
            for (std::int32_t t0 = 0; t0 < visible; t0 += tile_keys) {
                const std::int32_t t1 = std::min(t0 + tile_keys, visible);
                for (std::int32_t k = t0; k < t1; ++k) {
                    running_m = std::max(running_m, score[static_cast<std::size_t>(k)]);
                }
                for (std::int32_t b = t0; b < t1; b += 16) {
                    const std::int32_t b1 = std::min(b + 16, visible);
                    double block_max = 0.0;
                    for (std::int32_t k = b; k < b1; ++k) {
                        block_max = std::max(block_max, score[static_cast<std::size_t>(k)]);
                    }
                    const std::size_t bi = static_cast<std::size_t>(b / 16);
                    if (block_max == -std::numeric_limits<double>::infinity()) {
                        pb[bi].dec_s = 0.0;
                        continue;
                    }
                    const double s_arg =
                        std::min(kSMax, kSMax * std::exp2((block_max - running_m) * kLog2E));
                    const std::uint8_t sc =
                        s_arg == 0.0 ? 0 : encode_e4m3fn_rne(static_cast<float>(s_arg));
                    pb[bi].dec_s = static_cast<double>(decode_e4m3fn_word(sc));
                    for (std::int32_t k = b; k < b1; ++k) {
                        const double p_amp = kAmp *
                                             std::exp2(
                                                 (score[static_cast<std::size_t>(k)] - running_m) *
                                                     kLog2E);
                        const double ratio =
                            pb[bi].dec_s == 0.0 ? 0.0 : std::min(6.0, p_amp / pb[bi].dec_s);
                        pb[bi].p[static_cast<std::size_t>(k - b)] =
                            static_cast<double>(kPTableEmu[encode_e2m1_rne(
                                static_cast<float>(ratio)) & 0x0Fu]);
                    }
                }
            }

            for (std::int32_t d = 0; d < kHeadDim; ++d) {
                double acc = 0.0;
                double l_sum = 0.0;
                for (std::int32_t b = 0; b < blocks; ++b) {
                    const std::int32_t b0 = 16 * b;
                    const std::int32_t b1 = std::min(b0 + 16, visible);
                    // Re-derive the tile running max for the rescale factor.
                    const std::int32_t t0 = (b0 / tile_keys) * tile_keys;
                    const std::int32_t t1 = std::min(t0 + tile_keys, visible);
                    double nm_t = -std::numeric_limits<double>::infinity();
                    for (std::int32_t k = t0; k < t1; ++k) {
                        nm_t = std::max(nm_t, score[static_cast<std::size_t>(k)]);
                    }
                    const double rescale =
                        std::exp2((nm_t - m_final) * kLog2E);  // 1 for t0 == m_final tile
                    double pv = 0.0;
                    for (std::int32_t k = b0; k < b1; ++k) {
                        pv += pb[static_cast<std::size_t>(b)].p[static_cast<std::size_t>(k - b0)] *
                              cache_value(cache, false, kv_head, k, d);
                    }
                    acc += rescale * pv * pb[static_cast<std::size_t>(b)].dec_s;
                    for (std::int32_t k = b0; k < b1; ++k) {
                        l_sum += std::exp2((score[static_cast<std::size_t>(k)] - m_final) * kLog2E);
                    }
                }
                output[q_index(geometry, q_head, d, token)] =
                    l_sum > 0.0 ? acc / (kAmp * l_sum) : 0.0;
            }
        }
    }
    return output;
}

std::vector<double> ideal_attention(const std::vector<float>& q, const HostCache& cache,
                                    const std::vector<std::int32_t>& positions) {
    const Geometry& geometry  = cache.geometry;
    const std::int32_t tokens = static_cast<std::int32_t>(positions.size());
    std::vector<double> output(static_cast<std::size_t>(kHeadDim) *
                               static_cast<std::size_t>(geometry.q_heads) *
                               static_cast<std::size_t>(tokens));

    std::vector<double> scores(static_cast<std::size_t>(positions.back()) + 1);
    std::vector<double> probabilities(scores.size());
    for (std::int32_t token = 0; token < tokens; ++token) {
        const std::int32_t visible = positions[static_cast<std::size_t>(token)] + 1;
        for (std::int32_t q_head = 0; q_head < geometry.q_heads; ++q_head) {
            const std::int32_t kv_head = q_head / geometry.query_group();
            double max_score           = -std::numeric_limits<double>::infinity();
            for (std::int32_t position = 0; position < visible; ++position) {
                double dot = 0.0;
                for (std::int32_t d = 0; d < kHeadDim; ++d) {
                    dot += static_cast<double>(q[q_index(geometry, q_head, d, token)]) *
                           cache_value(cache, true, kv_head, position, d);
                }
                const double score = dot * static_cast<double>(kAttentionScale);
                scores[static_cast<std::size_t>(position)] = score;
                max_score                                  = std::max(max_score, score);
            }

            double sum = 0.0;
            for (std::int32_t position = 0; position < visible; ++position) {
                const double probability =
                    std::exp(scores[static_cast<std::size_t>(position)] - max_score);
                probabilities[static_cast<std::size_t>(position)] = probability;
                sum += probability;
            }
            for (std::int32_t position = 0; position < visible; ++position) {
                probabilities[static_cast<std::size_t>(position)] /= sum;
            }

            for (std::int32_t d = 0; d < kHeadDim; ++d) {
                double value = 0.0;
                for (std::int32_t position = 0; position < visible; ++position) {
                    value += probabilities[static_cast<std::size_t>(position)] *
                             cache_value(cache, false, kv_head, position, d);
                }
                output[q_index(geometry, q_head, d, token)] = value;
            }
        }
    }
    return output;
}

template <typename T>
std::vector<T> copy_from_guarded(const GuardedDeviceBuffer& buffer, std::size_t count) {
    std::vector<T> values(count);
    buffer.copy_to_host(values.data(), values.size() * sizeof(T));
    return values;
}

// Sage (nvfp4s3) reference: exact softmax over the quantized K and Q — the QK
// mma runs on FP4 tensor cores, so Q is emulated with the same NVFP4 codec as
// the KV cache (per-16-d e4m3 scale + e2m1 codes) — then the kernel's FP4-PV
// path emulated: P is quantized to e2m1 with a per-16-key e4m3 block scale
// (S = e4m3(448*P_block_max), codes = 2688*P/S landing in [0,6]), V stays in
// FP4 (e2m1 codes + per-(d,16-key) e4m3 scale). The 448*6 amplification cancels
// in out = acc / L, mirroring the kernel's online-softmax invariant.
std::vector<double> sage_ideal_attention(const std::vector<float>& q, const HostCache& cache,
                                         const std::vector<std::int32_t>& positions) {
    constexpr double kLog2E = 1.4426950408889634;
    const Geometry& geometry  = cache.geometry;
    const std::int32_t tokens = static_cast<std::int32_t>(positions.size());
    std::vector<double> output(static_cast<std::size_t>(kHeadDim) *
                               static_cast<std::size_t>(geometry.q_heads) *
                               static_cast<std::size_t>(tokens));
    constexpr double kAmp    = 2688.0;  // 448 * 6
    constexpr double kSMax   = 448.0;   // e4m3 max finite
    constexpr double kPTable[8] = {0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0};

    for (std::int32_t token = 0; token < tokens; ++token) {
        const std::int32_t visible = positions[static_cast<std::size_t>(token)] + 1;
        for (std::int32_t q_head = 0; q_head < geometry.q_heads; ++q_head) {
            const std::int32_t kv_head = q_head / geometry.query_group();
            // Emulate the kernel's NVFP4 Q-quantization: the QK mma runs on FP4
            // tensor cores with per-16-d e4m3 scales + e2m1 codes (the same codec
            // as the KV cache), so the reference scores must use the quantized Q.
            std::vector<double> q_log(kHeadDim, 0.0);
            for (std::int32_t grp = 0; grp < kHeadDim / kNvfp4Group; ++grp) {
                float absmax = 0.0f;
                for (std::int32_t i = 0; i < kNvfp4Group; ++i) {
                    const std::int32_t d = grp * kNvfp4Group + i;
                    absmax = std::max(absmax, std::abs(q[q_index(geometry, q_head, d, token)]));
                }
                const std::uint8_t scale_bits = encode_e4m3fn_rne(absmax / 6.0f);
                const float stored_scale = static_cast<float>(decode_e4m3fn_word(scale_bits));
                for (std::int32_t pair = 0; pair < kNvfp4Group / 2; ++pair) {
                    const std::int32_t d0 = grp * kNvfp4Group + 2 * pair;
                    const std::int32_t d1 = d0 + 1;
                    float x = 0.0f;
                    float y = 0.0f;
                    if (stored_scale != 0.0f) {
                        x = q[q_index(geometry, q_head, d0, token)] / stored_scale;
                        y = q[q_index(geometry, q_head, d1, token)] / stored_scale;
                    }
                    const float2 packed = {x, y};
                    const std::uint8_t code =
                        __nv_cvt_float2_to_fp4x2(packed, __NV_E2M1, cudaRoundNearest);
                    q_log[static_cast<std::size_t>(d0)] =
                        static_cast<double>(decode_e2m1_word(code & 0x0Fu)) * stored_scale;
                    q_log[static_cast<std::size_t>(d1)] =
                        static_cast<double>(decode_e2m1_word((code >> 4) & 0x0Fu)) * stored_scale;
                }
            }
            std::vector<double> p(visible, 0.0);
            double max_score          = -std::numeric_limits<double>::infinity();
            for (std::int32_t position = 0; position < visible; ++position) {
                double dot = 0.0;
                for (std::int32_t d = 0; d < kHeadDim; ++d) {
                    dot += q_log[static_cast<std::size_t>(d)] *
                           cache_value(cache, true, kv_head, position, d);
                }
                const double score = dot * static_cast<double>(kAttentionScale);
                p[static_cast<std::size_t>(position)] = score;  // raw scores first
                max_score                              = std::max(max_score, score);
            }
            double l_sum = 0.0;
            for (std::int32_t position = 0; position < visible; ++position) {
                p[static_cast<std::size_t>(position)] =
                    (p[static_cast<std::size_t>(position)] == -std::numeric_limits<double>::infinity())
                        ? 0.0
                        : std::exp2(static_cast<double>(kLog2E) *
                                     (p[static_cast<std::size_t>(position)] - max_score));
                l_sum += p[static_cast<std::size_t>(position)];
            }

            std::vector<double> value(kHeadDim, 0.0);
            for (std::int32_t d = 0; d < kHeadDim; ++d) {
                double acc = 0.0;
                for (std::int32_t kb = 0; kb < visible; kb += 16) {
                    const std::int32_t k1        = std::min(kb + 16, visible);
                    double block_max             = 0.0;
                    for (std::int32_t k = kb; k < k1; ++k) block_max = std::max(block_max, p[static_cast<std::size_t>(k)]);
                    const std::uint8_t sc =
                        block_max == 0.0
                            ? 0
                            : encode_e4m3fn_rne(static_cast<float>(std::min(kSMax, kSMax * block_max)));
                    const double s_dec = static_cast<double>(decode_e4m3fn_word(sc));
                    const double v_dec = static_cast<double>(decode_e4m3fn_word(
                        cache.v_fp8[sage_v_scale_host_index(cache.logical_capacity, kv_head,
                                                             kb / kPagedKVPageSize, d,
                                                             (kb % kPagedKVPageSize) / 16)]));
                    double pv = 0.0;
                    for (std::int32_t k = kb; k < k1; ++k) {
                        const double p_p = s_dec == 0.0
                                               ? 0.0
                                               : std::min(6.0, kAmp * p[static_cast<std::size_t>(k)] / s_dec);
                        const std::uint8_t p_code =
                            static_cast<std::uint8_t>(encode_e2m1_rne(static_cast<float>(p_p)) & 0x0fu);
                        const std::size_t v_code_idx =
                            nvfp4_code_index(geometry, cache.logical_capacity, kv_head, k, d / 2);
                        const std::uint8_t v_byte  = cache.v_u8[v_code_idx];
                        const std::uint8_t v_nibble = (d & 1) == 0 ? (v_byte & 0x0fu) : (v_byte >> 4);
                        pv += kPTable[p_code] * decode_e2m1_word(v_nibble);
                    }
                    acc += pv * s_dec * v_dec;
                }
                value[static_cast<std::size_t>(d)] = acc;
            }
            for (std::int32_t d = 0; d < kHeadDim; ++d) {
                output[q_index(geometry, q_head, d, token)] =
                    l_sum > 0.0 ? value[static_cast<std::size_t>(d)] / (kAmp * l_sum) : 0.0;
            }
        }
    }
    return output;
}

class DeviceCache {
public:
    DeviceCache(const HostCache& cache, MappingPattern mapping)
        : geometry_(cache.geometry), dtype_(cache.dtype), sage_(cache.sage),
          max_context_(cache.max_context),
          logical_capacity_(cache.logical_capacity),
          logical_pages_(logical_capacity_ / kPagedKVPageSize),
          physical_pages_(physical_page_count(logical_pages_, mapping)),
          block_table_host_(make_block_table(logical_pages_, mapping)),
          code_leading_(dtype_ == DType::U8 ? kNvfp4CodeWidth : kHeadDim),
          scale_groups_(dtype_ == DType::U8 ? kNvfp4Groups
                                            : (dtype_ == DType::I8 ? kQuantGroups : 0)),
          code_elements_(static_cast<std::size_t>(code_leading_) * kPagedKVPageSize *
                         geometry_.kv_heads * physical_pages_),
          scale_elements_(static_cast<std::size_t>(std::max(scale_groups_, 1)) * kPagedKVPageSize *
                          geometry_.kv_heads * physical_pages_),
          k_(code_elements_ * (dtype_ == DType::BF16 ? sizeof(std::uint16_t) : sizeof(std::uint8_t))),
          v_(code_elements_ * (dtype_ == DType::BF16 ? sizeof(std::uint16_t) : sizeof(std::uint8_t))),
          k_scale_(dtype_ == DType::I8 ? scale_elements_ * sizeof(std::uint16_t)
                   : dtype_ == DType::U8 ? scale_elements_
                                         : 1),
          v_scale_(dtype_ == DType::I8 ? scale_elements_ * sizeof(std::uint16_t)
                   : dtype_ == DType::U8 ? scale_elements_
                                         : 1),
          block_table_(block_table_host_.size() * sizeof(std::int32_t)) {
        block_table_.copy_from_host(block_table_host_.data(),
                                    block_table_host_.size() * sizeof(std::int32_t));
        if (dtype_ == DType::BF16) {
            const auto k_physical =
                scatter_paged(cache.k_bf16, kHeadDim, geometry_, logical_capacity_,
                              block_table_host_, physical_pages_);
            const auto v_physical =
                scatter_paged(cache.v_bf16, kHeadDim, geometry_, logical_capacity_,
                              block_table_host_, physical_pages_);
            k_.copy_from_host(k_physical.data(), k_physical.size() * sizeof(std::uint16_t));
            v_.copy_from_host(v_physical.data(), v_physical.size() * sizeof(std::uint16_t));
        } else if (dtype_ == DType::U8) {
            const auto k_physical =
                scatter_paged(cache.k_u8, kNvfp4CodeWidth, geometry_, logical_capacity_,
                              block_table_host_, physical_pages_);
            const auto v_physical =
                scatter_paged(cache.v_u8, kNvfp4CodeWidth, geometry_, logical_capacity_,
                              block_table_host_, physical_pages_);
            const auto ks_physical =
                scatter_paged(cache.k_fp8, kNvfp4Groups, geometry_, logical_capacity_,
                              block_table_host_, physical_pages_);
            const auto vs_physical =
                scatter_paged(cache.v_fp8, kNvfp4Groups, geometry_, logical_capacity_,
                              block_table_host_, physical_pages_);
            k_.copy_from_host(k_physical.data(), k_physical.size());
            v_.copy_from_host(v_physical.data(), v_physical.size());
            k_scale_.copy_from_host(ks_physical.data(), ks_physical.size());
            v_scale_.copy_from_host(vs_physical.data(), vs_physical.size());
            if (sage_ && std::getenv("GQA_KEEP_FRAC") != nullptr) {
                // Sage meansim proxy plane: the fill kernel writes the per-(page, kv_head,
                // group) dequantized K mean when this plane is present, so a keep_frac < 1
                // run actually engages the tile-skip path. Without it the kernel
                // defensively falls back to the exact path and the A/B is void.
                k_mean_elements_ = 4 * kPagedKVPageSize * geometry_.kv_heads * physical_pages_;
                k_mean_ = std::make_unique<GuardedDeviceBuffer>(k_mean_elements_ * sizeof(float));
                k_mean_->fill(0);
            }
        } else {
            const auto k_physical =
                scatter_paged(cache.k_i8, kHeadDim, geometry_, logical_capacity_, block_table_host_,
                              physical_pages_);
            const auto v_physical =
                scatter_paged(cache.v_i8, kHeadDim, geometry_, logical_capacity_, block_table_host_,
                              physical_pages_);
            const auto ks_physical =
                scatter_paged(cache.k_scale, kQuantGroups, geometry_, logical_capacity_,
                              block_table_host_, physical_pages_);
            const auto vs_physical =
                scatter_paged(cache.v_scale, kQuantGroups, geometry_, logical_capacity_,
                              block_table_host_, physical_pages_);
            k_.copy_from_host(k_physical.data(), k_physical.size() * sizeof(std::int8_t));
            v_.copy_from_host(v_physical.data(), v_physical.size() * sizeof(std::int8_t));
            k_scale_.copy_from_host(ks_physical.data(), ks_physical.size() * sizeof(std::uint16_t));
            v_scale_.copy_from_host(vs_physical.data(), vs_physical.size() * sizeof(std::uint16_t));
        }
    }

    PagedKVLayerView view() {
        PagedKVLayerView result;
        result.k_pages      = Tensor(k_.data(), dtype_,
                                     {code_leading_, kPagedKVPageSize, geometry_.kv_heads, physical_pages_});
        result.v_pages      = Tensor(v_.data(), dtype_,
                                     {code_leading_, kPagedKVPageSize, geometry_.kv_heads, physical_pages_});
        result.block_table  = Tensor(block_table_.data(), DType::I32, {logical_pages_});
        result.num_kv_heads = geometry_.kv_heads;
        result.head_dim     = kHeadDim;
        result.dtype        = dtype_;
        if (dtype_ == DType::I8) {
            result.k_scale_pages =
                Tensor(k_scale_.data(), DType::FP16,
                       {kQuantGroups, kPagedKVPageSize, geometry_.kv_heads, physical_pages_});
            result.v_scale_pages =
                Tensor(v_scale_.data(), DType::FP16,
                       {kQuantGroups, kPagedKVPageSize, geometry_.kv_heads, physical_pages_});
            result.quant_group = kQuantGroup;
        } else if (dtype_ == DType::U8) {
            result.k_scale_pages =
                Tensor(k_scale_.data(), DType::FP8_E4M3FN,
                       {kNvfp4Groups, kPagedKVPageSize, geometry_.kv_heads, physical_pages_});
            result.v_scale_pages =
                Tensor(v_scale_.data(), DType::FP8_E4M3FN,
                       {kNvfp4Groups, kPagedKVPageSize, geometry_.kv_heads, physical_pages_});
            result.quant_group = kNvfp4Group;
            if (sage_ && k_mean_elements_ > 0) {
                result.k_mean_pages =
                    Tensor(k_mean_->data(), DType::FP32,
                           {4, kPagedKVPageSize, geometry_.kv_heads, physical_pages_});
            }
        }
        result.sage_pv = sage_;
        return result;
    }

    PagedKVBatchLayerView batch_view() {
        const PagedKVLayerView direct = view();
        return {
            .k_pages       = direct.k_pages,
            .v_pages       = direct.v_pages,
            .k_scale_pages = direct.k_scale_pages,
            .v_scale_pages = direct.v_scale_pages,
            .k_mean_pages  = direct.k_mean_pages,
            .block_tables  = direct.block_table.view({logical_pages_, 1}),
            .head_dim      = direct.head_dim,
            .num_kv_heads  = direct.num_kv_heads,
            .dtype         = direct.dtype,
            .quant_group   = direct.quant_group,
            .sage_pv       = direct.sage_pv,
        };
    }

    HostCache snapshot() const {
        HostCache cache{geometry_, dtype_, max_context_, logical_capacity_};
        if (dtype_ == DType::BF16) {
            const auto k_physical = copy_from_guarded<std::uint16_t>(k_, code_elements_);
            const auto v_physical = copy_from_guarded<std::uint16_t>(v_, code_elements_);
            cache.k_bf16          = gather_paged<std::uint16_t>(k_physical, kHeadDim, geometry_,
                                                                logical_capacity_, block_table_host_);
            cache.v_bf16          = gather_paged<std::uint16_t>(v_physical, kHeadDim, geometry_,
                                                                logical_capacity_, block_table_host_);
        } else if (dtype_ == DType::U8) {
            const auto k_physical  = copy_from_guarded<std::uint8_t>(k_, code_elements_);
            const auto v_physical  = copy_from_guarded<std::uint8_t>(v_, code_elements_);
            const auto ks_physical = copy_from_guarded<std::uint8_t>(k_scale_, scale_elements_);
            const auto vs_physical = copy_from_guarded<std::uint8_t>(v_scale_, scale_elements_);
            cache.k_u8             = gather_paged<std::uint8_t>(k_physical, kNvfp4CodeWidth, geometry_,
                                                               logical_capacity_, block_table_host_);
            cache.v_u8             = gather_paged<std::uint8_t>(v_physical, kNvfp4CodeWidth, geometry_,
                                                               logical_capacity_, block_table_host_);
            cache.k_fp8 = gather_paged<std::uint8_t>(ks_physical, kNvfp4Groups, geometry_,
                                                    logical_capacity_, block_table_host_);
            cache.v_fp8 = gather_paged<std::uint8_t>(vs_physical, kNvfp4Groups, geometry_,
                                                    logical_capacity_, block_table_host_);
        } else {
            const auto k_physical  = copy_from_guarded<std::int8_t>(k_, code_elements_);
            const auto v_physical  = copy_from_guarded<std::int8_t>(v_, code_elements_);
            const auto ks_physical = copy_from_guarded<std::uint16_t>(k_scale_, scale_elements_);
            const auto vs_physical = copy_from_guarded<std::uint16_t>(v_scale_, scale_elements_);
            cache.k_i8             = gather_paged<std::int8_t>(k_physical, kHeadDim, geometry_,
                                                               logical_capacity_, block_table_host_);
            cache.v_i8             = gather_paged<std::int8_t>(v_physical, kHeadDim, geometry_,
                                                               logical_capacity_, block_table_host_);
            cache.k_scale = gather_paged<std::uint16_t>(ks_physical, kQuantGroups, geometry_,
                                                        logical_capacity_, block_table_host_);
            cache.v_scale = gather_paged<std::uint16_t>(vs_physical, kQuantGroups, geometry_,
                                                        logical_capacity_, block_table_host_);
        }
        return cache;
    }

    int verify_guards(const std::string& label) const {
        int failures = 0;
        failures += k_.verify_guards((label + " cache-k").c_str());
        failures += v_.verify_guards((label + " cache-v").c_str());
        if (dtype_ == DType::I8 || dtype_ == DType::U8) {
            failures += k_scale_.verify_guards((label + " cache-k-scale").c_str());
            failures += v_scale_.verify_guards((label + " cache-v-scale").c_str());
        }
        failures += block_table_.verify_guards((label + " block-table").c_str());
        failures +=
            verify_exact((label + " block-table unchanged").c_str(),
                         copy_from_guarded<std::int32_t>(block_table_, block_table_host_.size()),
                         block_table_host_);
        return failures;
    }

private:
    Geometry geometry_;
    DType dtype_;
    bool sage_ = false;
    std::int32_t max_context_;
    std::int32_t logical_capacity_;
    std::int32_t logical_pages_;
    std::int32_t physical_pages_;
    std::vector<std::int32_t> block_table_host_;
    std::int32_t code_leading_ = kHeadDim;
    std::int32_t scale_groups_ = 0;
    std::size_t code_elements_;
    std::size_t scale_elements_;
    GuardedDeviceBuffer k_;
    GuardedDeviceBuffer v_;
    GuardedDeviceBuffer k_scale_;
    GuardedDeviceBuffer v_scale_;
    GuardedDeviceBuffer block_table_;
    std::size_t k_mean_elements_ = 0;
    std::unique_ptr<GuardedDeviceBuffer> k_mean_;
};

class BatchDeviceCache {
public:
    BatchDeviceCache(std::span<const HostCache> rows, MappingPattern mapping)
        : geometry_(rows.front().geometry), dtype_(rows.front().dtype), rows_(rows.size()),
          logical_capacity_(rows.front().logical_capacity),
          logical_pages_(logical_capacity_ / kPagedKVPageSize),
          physical_pages_(mapping == MappingPattern::Fragmented
                              ? 2 * static_cast<std::int32_t>(rows_) * logical_pages_ + 1
                              : static_cast<std::int32_t>(rows_) * logical_pages_),
          block_tables_host_(rows_ * static_cast<std::size_t>(logical_pages_)),
          code_elements_(static_cast<std::size_t>(kHeadDim) * kPagedKVPageSize *
                         geometry_.kv_heads * physical_pages_),
          scale_elements_(static_cast<std::size_t>(kQuantGroups) * kPagedKVPageSize *
                          geometry_.kv_heads * physical_pages_),
          k_(code_elements_ *
             (dtype_ == DType::BF16 ? sizeof(std::uint16_t) : sizeof(std::int8_t))),
          v_(code_elements_ *
             (dtype_ == DType::BF16 ? sizeof(std::uint16_t) : sizeof(std::int8_t))),
          k_scale_(dtype_ == DType::I8 ? scale_elements_ * sizeof(std::uint16_t) : 1),
          v_scale_(dtype_ == DType::I8 ? scale_elements_ * sizeof(std::uint16_t) : 1),
          block_tables_(block_tables_host_.size() * sizeof(std::int32_t)) {
        for (std::size_t row = 0; row < rows_; ++row) {
            const HostCache& cache = rows[row];
            if (cache.geometry.q_heads != geometry_.q_heads ||
                cache.geometry.kv_heads != geometry_.kv_heads || cache.dtype != dtype_ ||
                cache.logical_capacity != logical_capacity_) {
                throw std::invalid_argument("batch cache rows must share one physical geometry");
            }
            for (std::int32_t logical = 0; logical < logical_pages_; ++logical) {
                const std::int32_t linear =
                    static_cast<std::int32_t>(row) * logical_pages_ + logical;
                block_tables_host_[row * static_cast<std::size_t>(logical_pages_) + logical] =
                    mapping == MappingPattern::Fragmented ? 2 * linear + 1 : linear;
            }
        }
        block_tables_.copy_from_host(block_tables_host_.data(),
                                     block_tables_host_.size() * sizeof(std::int32_t));
        upload_rows(rows);
    }

    PagedKVBatchLayerView view() {
        PagedKVBatchLayerView result;
        result.k_pages      = Tensor(k_.data(), dtype_,
                                     {kHeadDim, kPagedKVPageSize, geometry_.kv_heads, physical_pages_});
        result.v_pages      = Tensor(v_.data(), dtype_,
                                     {kHeadDim, kPagedKVPageSize, geometry_.kv_heads, physical_pages_});
        result.block_tables = Tensor(block_tables_.data(), DType::I32,
                                     {logical_pages_, static_cast<std::int32_t>(rows_)});
        result.num_kv_heads = geometry_.kv_heads;
        result.head_dim     = kHeadDim;
        result.dtype        = dtype_;
        if (dtype_ == DType::I8) {
            result.k_scale_pages =
                Tensor(k_scale_.data(), DType::FP16,
                       {kQuantGroups, kPagedKVPageSize, geometry_.kv_heads, physical_pages_});
            result.v_scale_pages =
                Tensor(v_scale_.data(), DType::FP16,
                       {kQuantGroups, kPagedKVPageSize, geometry_.kv_heads, physical_pages_});
            result.quant_group = kQuantGroup;
        }
        return result;
    }

    int verify(const std::string& label, std::span<const HostCache> expected) const {
        if (expected.size() != rows_) {
            std::cerr << label << ": expected cache row count mismatch\n";
            return 1;
        }
        int failures = 0;
        if (dtype_ == DType::BF16) {
            std::vector<std::uint16_t> expected_k(code_elements_, 0);
            std::vector<std::uint16_t> expected_v(code_elements_, 0);
            scatter_bf16_rows(expected, expected_k, expected_v);
            failures +=
                verify_exact((label + " cache-k").c_str(),
                             copy_from_guarded<std::uint16_t>(k_, code_elements_), expected_k);
            failures +=
                verify_exact((label + " cache-v").c_str(),
                             copy_from_guarded<std::uint16_t>(v_, code_elements_), expected_v);
        } else {
            std::vector<std::int8_t> expected_k(code_elements_, 0);
            std::vector<std::int8_t> expected_v(code_elements_, 0);
            std::vector<std::uint16_t> expected_ks(scale_elements_, 0);
            std::vector<std::uint16_t> expected_vs(scale_elements_, 0);
            for (std::size_t row = 0; row < rows_; ++row) {
                const std::span<const std::int32_t> table = row_table(row);
                scatter_paged_into(expected[row].k_i8, kHeadDim, geometry_, logical_capacity_,
                                   table, expected_k);
                scatter_paged_into(expected[row].v_i8, kHeadDim, geometry_, logical_capacity_,
                                   table, expected_v);
                scatter_paged_into(expected[row].k_scale, kQuantGroups, geometry_,
                                   logical_capacity_, table, expected_ks);
                scatter_paged_into(expected[row].v_scale, kQuantGroups, geometry_,
                                   logical_capacity_, table, expected_vs);
            }
            failures +=
                verify_exact((label + " cache-k-code").c_str(),
                             copy_from_guarded<std::int8_t>(k_, code_elements_), expected_k);
            failures +=
                verify_exact((label + " cache-v-code").c_str(),
                             copy_from_guarded<std::int8_t>(v_, code_elements_), expected_v);
            failures += verify_exact((label + " cache-k-scale").c_str(),
                                     copy_from_guarded<std::uint16_t>(k_scale_, scale_elements_),
                                     expected_ks);
            failures += verify_exact((label + " cache-v-scale").c_str(),
                                     copy_from_guarded<std::uint16_t>(v_scale_, scale_elements_),
                                     expected_vs);
        }
        failures +=
            verify_exact((label + " block tables unchanged").c_str(),
                         copy_from_guarded<std::int32_t>(block_tables_, block_tables_host_.size()),
                         block_tables_host_);
        failures += k_.verify_guards((label + " cache-k guard").c_str());
        failures += v_.verify_guards((label + " cache-v guard").c_str());
        if (dtype_ == DType::I8) {
            failures += k_scale_.verify_guards((label + " cache-k-scale guard").c_str());
            failures += v_scale_.verify_guards((label + " cache-v-scale guard").c_str());
        }
        failures += block_tables_.verify_guards((label + " block tables guard").c_str());
        return failures;
    }

private:
    [[nodiscard]] std::span<const std::int32_t> row_table(std::size_t row) const {
        return std::span<const std::int32_t>(block_tables_host_.data() +
                                                 row * static_cast<std::size_t>(logical_pages_),
                                             static_cast<std::size_t>(logical_pages_));
    }

    void scatter_bf16_rows(std::span<const HostCache> rows, std::vector<std::uint16_t>& k,
                           std::vector<std::uint16_t>& v) const {
        for (std::size_t row = 0; row < rows_; ++row) {
            const std::span<const std::int32_t> table = row_table(row);
            scatter_paged_into(rows[row].k_bf16, kHeadDim, geometry_, logical_capacity_, table, k);
            scatter_paged_into(rows[row].v_bf16, kHeadDim, geometry_, logical_capacity_, table, v);
        }
    }

    void upload_rows(std::span<const HostCache> rows) {
        if (dtype_ == DType::BF16) {
            std::vector<std::uint16_t> physical_k(code_elements_, 0);
            std::vector<std::uint16_t> physical_v(code_elements_, 0);
            scatter_bf16_rows(rows, physical_k, physical_v);
            k_.copy_from_host(physical_k.data(), physical_k.size() * sizeof(std::uint16_t));
            v_.copy_from_host(physical_v.data(), physical_v.size() * sizeof(std::uint16_t));
            return;
        }
        std::vector<std::int8_t> physical_k(code_elements_, 0);
        std::vector<std::int8_t> physical_v(code_elements_, 0);
        std::vector<std::uint16_t> physical_ks(scale_elements_, 0);
        std::vector<std::uint16_t> physical_vs(scale_elements_, 0);
        for (std::size_t row = 0; row < rows_; ++row) {
            const std::span<const std::int32_t> table = row_table(row);
            scatter_paged_into(rows[row].k_i8, kHeadDim, geometry_, logical_capacity_, table,
                               physical_k);
            scatter_paged_into(rows[row].v_i8, kHeadDim, geometry_, logical_capacity_, table,
                               physical_v);
            scatter_paged_into(rows[row].k_scale, kQuantGroups, geometry_, logical_capacity_, table,
                               physical_ks);
            scatter_paged_into(rows[row].v_scale, kQuantGroups, geometry_, logical_capacity_, table,
                               physical_vs);
        }
        k_.copy_from_host(physical_k.data(), physical_k.size() * sizeof(std::int8_t));
        v_.copy_from_host(physical_v.data(), physical_v.size() * sizeof(std::int8_t));
        k_scale_.copy_from_host(physical_ks.data(), physical_ks.size() * sizeof(std::uint16_t));
        v_scale_.copy_from_host(physical_vs.data(), physical_vs.size() * sizeof(std::uint16_t));
    }

    Geometry geometry_;
    DType dtype_;
    std::size_t rows_;
    std::int32_t logical_capacity_;
    std::int32_t logical_pages_;
    std::int32_t physical_pages_;
    std::vector<std::int32_t> block_tables_host_;
    std::size_t code_elements_;
    std::size_t scale_elements_;
    GuardedDeviceBuffer k_;
    GuardedDeviceBuffer v_;
    GuardedDeviceBuffer k_scale_;
    GuardedDeviceBuffer v_scale_;
    GuardedDeviceBuffer block_tables_;
};

int verify_cache(const std::string& label, const HostCache& got, const HostCache& expected) {
    int failures = 0;
    if (expected.dtype == DType::BF16) {
        failures += verify_exact((label + " cache-k").c_str(), got.k_bf16, expected.k_bf16);
        failures += verify_exact((label + " cache-v").c_str(), got.v_bf16, expected.v_bf16);
    } else if (expected.dtype == DType::U8) {
        failures += verify_exact((label + " cache-k-code").c_str(), got.k_u8, expected.k_u8);
        failures += verify_exact((label + " cache-v-code").c_str(), got.v_u8, expected.v_u8);
        failures += verify_exact((label + " cache-k-scale").c_str(), got.k_fp8, expected.k_fp8);
        failures += verify_exact((label + " cache-v-scale").c_str(), got.v_fp8, expected.v_fp8);
        if (expected.sage && std::getenv("SAGE_DUMP") != nullptr) {
            auto first_diff = [](const std::vector<std::uint8_t>& a,
                                 const std::vector<std::uint8_t>& b) -> std::int64_t {
                const std::size_t n = std::min(a.size(), b.size());
                for (std::size_t i = 0; i < n; ++i) {
                    if (a[i] != b[i]) { return static_cast<std::int64_t>(i); }
                }
                return -1;
            };
            const std::int64_t dc = first_diff(got.v_u8, expected.v_u8);
            const std::int64_t ds = first_diff(got.v_fp8, expected.v_fp8);
            const int cap = static_cast<int>(expected.logical_capacity);
            const int heads = expected.geometry.kv_heads;
            std::cerr << "SAGE_DUMP v-code diff " << dc << " v-scale diff " << ds << '\n';
            if (dc >= 0) {
                const int h = static_cast<int>(dc / (128 * cap));
                const int rem = static_cast<int>(dc % (128 * cap));
                const int p = rem / 128;
                const int dp = rem % 128;
                std::cerr << "  v-code head=" << h << " pos=" << p << " dp=" << dp
                          << " device=" << static_cast<int>(got.v_u8[static_cast<std::size_t>(dc)])
                          << " host=" << static_cast<int>(expected.v_u8[static_cast<std::size_t>(dc)])
                          << '\n';
                // Reconstruct the quantization inputs for this (head, dp): the 16-key block's
                // scale (e4m3) and the host-side |v| lattice so the tie can be inspected.
                const int kb = p / 16;
                const std::size_t sd0 =
                    sage_v_scale_host_index(expected.logical_capacity, h, p / kPagedKVPageSize, dp * 2, kb);
                const std::size_t sd1 =
                    sage_v_scale_host_index(expected.logical_capacity, h, p / kPagedKVPageSize, dp * 2 + 1, kb);
                const float s0h = static_cast<float>(decode_e4m3fn_word(expected.v_fp8[sd0]));
                const float s1h = static_cast<float>(decode_e4m3fn_word(expected.v_fp8[sd1]));
                const float s0d = static_cast<float>(decode_e4m3fn_word(got.v_fp8[sd0]));
                const float s1d = static_cast<float>(decode_e4m3fn_word(got.v_fp8[sd1]));
                std::cerr << "  scale even-d host=" << s0h << " dev=" << s0d
                          << "  odd-d host=" << s1h << " dev=" << s1d << "\n";
            }
            if (ds >= 0) {
                const int h = static_cast<int>(ds / (16 * cap));
                const int rem = static_cast<int>(ds % (16 * cap));
                const int d   = rem / 4;
                const int kb  = rem % 4;
                std::cerr << "  v-scale head=" << h << " d=" << d << " kb=" << kb
                          << " device=" << static_cast<int>(got.v_fp8[static_cast<std::size_t>(ds)])
                          << " host=" << static_cast<int>(expected.v_fp8[static_cast<std::size_t>(ds)])
                          << '\n';
                // Per-key |v| reconstruction for this (head, d): the block's d-odd
                // codes live in the high nibble of byte dp = d/2; |value| =
                // code_val * scale. Print device vs host per key.
                const int key0 = kb * 16;
                const int dp   = d / 2;
                std::cerr << "  per-key |v| (head=" << h << " d=" << d << ", device vs host): ";
                for (int k = key0; k < key0 + 16; ++k) {
                    const std::size_t cidx =
                        static_cast<std::size_t>(h) * 128 * expected.logical_capacity +
                        static_cast<std::size_t>(k) * 128 + dp;
                    const unsigned db = got.v_u8[cidx] >> 4;
                    const unsigned hb = expected.v_u8[cidx] >> 4;
                    const float dscl = decode_e4m3fn_word(got.v_fp8[static_cast<std::size_t>(ds)]) / 6.0f * 6.0f; // scale*6 = max
                    (void)dscl;
                    std::cerr << k << ":[" << (db * 1) << "/" << (hb * 1) << "] ";
                }
                std::cerr << "  (e2m1 code vals; dev/host per key 48..63)\n";
            }
        }
    } else {
        failures += verify_exact((label + " cache-k-code").c_str(), got.k_i8, expected.k_i8);
        failures += verify_exact((label + " cache-v-code").c_str(), got.v_i8, expected.v_i8);
        failures += verify_exact((label + " cache-k-scale").c_str(), got.k_scale, expected.k_scale);
        failures += verify_exact((label + " cache-v-scale").c_str(), got.v_scale, expected.v_scale);
    }
    return failures;
}

int verify_input(const std::string& label, const GuardedDeviceBuffer& device,
                 const std::vector<std::uint16_t>& expected) {
    int failures = verify_exact(
        label.c_str(), copy_from_guarded<std::uint16_t>(device, expected.size()), expected);
    failures += device.verify_guards((label + " guard").c_str());
    return failures;
}

int verify_positions(const std::string& label, const GuardedDeviceBuffer& device,
                     const std::vector<std::int32_t>& expected) {
    int failures = verify_exact(label.c_str(),
                                copy_from_guarded<std::int32_t>(device, expected.size()), expected);
    failures += device.verify_guards((label + " guard").c_str());
    return failures;
}

const char* cache_name(DType dtype) {
    if (dtype == DType::BF16) { return "bf16"; }
    if (dtype == DType::U8) { return "nvfp4-g16"; }
    return "int8-g64";
}

ReductionCriterion attention_criterion(DType dtype, bool sage = false) {
    if (dtype == DType::BF16) { return kAttentionBf16Criterion; }
    if (dtype == DType::U8) {
        return sage ? kAttentionNvfp4s3Criterion : kAttentionNvfp4Criterion;
    }
    return kAttentionInt8Criterion;
}

int verify_attention(const std::string& label, const std::vector<double>& actual,
                     const std::vector<double>& reference, const ReductionCriterion& criterion) {
    return verify_reduction(label.c_str(), actual, reference, criterion);
}

std::vector<double> twin_cache_attention(const Geometry& geometry, DType twin_dtype,
                                         std::int32_t max_context, std::uint32_t cache_seed,
                                         const std::vector<float>& q,
                                         const std::vector<std::int32_t>& positions,
                                         const std::vector<float>* append_k = nullptr,
                                         const std::vector<float>* append_v = nullptr) {
    HostCache twin = make_cache(geometry, twin_dtype, max_context, cache_seed);
    if (append_k != nullptr) { append_cache(twin, *append_k, *append_v, positions); }
    return ideal_attention(q, twin, positions);
}

// FP4-P-quant floor diagnostic (gated by GQA_SAGE_FLOOR=1): for sage cases,
// print the rel_L2 between the P-quant-emulated reference (sage_ideal_attention)
// and the exact reference (ideal_attention) — that is the floor any FP4-PV
// kernel must sit above — plus the device output's rel_L2 against the exact
// reference. If device-vs-sage-ref ~= floor, the kernel is within the FP4-P
// quantization floor and the residual is not a kernel bug.
void sage_floor_report(const std::string& label, const std::vector<double>& actual,
                      const std::vector<double>& sage_reference, const std::vector<float>& q,
                      const HostCache& cache, const std::vector<std::int32_t>& positions) {
    if (std::getenv("GQA_SAGE_FLOOR") == nullptr) { return; }
    const std::vector<double> exact_reference = ideal_attention(q, cache, positions);
    const std::int64_t n = static_cast<std::int64_t>(actual.size());
    const double floor_l2 =
        compute_reduction_stats(sage_reference.data(), exact_reference.data(), n).relative_l2;
    const double dev_vs_exact =
        compute_reduction_stats(actual.data(), exact_reference.data(), n).relative_l2;
    const double dev_vs_sage =
        compute_reduction_stats(actual.data(), sage_reference.data(), n).relative_l2;
    // Step-exact audit: a correct kernel sits within ~1e-4 of the step emulation
    // (tile 64 = prefill route, tile 32 = decode route); the other granularity
    // differs only by the S-argument effect (small, bounded).
    const std::vector<double> step_prefill = sage_step_emulation(q, cache, positions, 64);
    const std::vector<double> step_decode = sage_step_emulation(q, cache, positions, 32);
    const double dev_vs_step_prefill =
        compute_reduction_stats(actual.data(), step_prefill.data(), n).relative_l2;
    const double dev_vs_step_decode =
        compute_reduction_stats(actual.data(), step_decode.data(), n).relative_l2;
    // Host-only cross-check: if the step and closed-form emulations agree with each
    // other (small), the device's distance from them is a real kernel bug; if the
    // two emulations disagree at floor scale, one of them has a structural bug.
    const double step64_vs_sage =
        compute_reduction_stats(step_prefill.data(), sage_reference.data(), n).relative_l2;
    const double step32_vs_sage =
        compute_reduction_stats(step_decode.data(), sage_reference.data(), n).relative_l2;
    const double step64_vs_exact =
        compute_reduction_stats(step_prefill.data(), exact_reference.data(), n).relative_l2;
    sage_orc_pdump(label, q, cache, positions);
    std::cerr << label << " SAGE_FLOOR: floor=" << floor_l2 << " device_vs_exact=" << dev_vs_exact
              << " device_vs_sage=" << dev_vs_sage << " dev_vs_step_pref64=" << dev_vs_step_prefill
              << " dev_vs_step_dec32=" << dev_vs_step_decode << " step64_vs_sage=" << step64_vs_sage
              << " step32_vs_sage=" << step32_vs_sage << " step64_vs_exact=" << step64_vs_exact
              << " n=" << n << "\n";
}

int verify_nvfp4_kernel_inside_codec_gap(const std::string& label,
                                         const std::vector<double>& kernel,
                                         const std::vector<double>& nvfp4_oracle,
                                         const std::vector<double>& int8_oracle,
                                         const std::vector<double>& bf16_oracle) {
    if (kernel.empty() || kernel.size() != nvfp4_oracle.size() ||
        kernel.size() != int8_oracle.size() || kernel.size() != bf16_oracle.size()) {
        std::cerr << label << ": NVFP4 codec-gap comparison size mismatch\n";
        return 1;
    }
    const auto count = static_cast<std::int64_t>(kernel.size());
    const ReductionStats kernel_vs_nvfp4 =
        compute_reduction_stats(kernel.data(), nvfp4_oracle.data(), count);
    const ReductionStats nvfp4_vs_int8 =
        compute_reduction_stats(nvfp4_oracle.data(), int8_oracle.data(), count);
    const ReductionStats int8_vs_bf16 =
        compute_reduction_stats(int8_oracle.data(), bf16_oracle.data(), count);
    const ReductionStats nvfp4_vs_bf16 =
        compute_reduction_stats(nvfp4_oracle.data(), bf16_oracle.data(), count);
    if (kernel_vs_nvfp4.first_non_finite >= 0 || nvfp4_vs_int8.first_non_finite >= 0 ||
        int8_vs_bf16.first_non_finite >= 0 || nvfp4_vs_bf16.first_non_finite >= 0) {
        std::cerr << label << ": non-finite NVFP4 codec-gap stats\n";
        return 1;
    }
    std::printf("KV_CODEC_GAP %s kernel_vs_nvfp4=%.6g nvfp4_vs_int8=%.6g int8_vs_bf16=%.6g "
                "nvfp4_vs_bf16=%.6g nvfp4/int8_vs_bf16=%.3g\n",
                label.c_str(), kernel_vs_nvfp4.relative_l2, nvfp4_vs_int8.relative_l2,
                int8_vs_bf16.relative_l2, nvfp4_vs_bf16.relative_l2,
                nvfp4_vs_bf16.relative_l2 /
                    std::max(int8_vs_bf16.relative_l2, 1.0e-30));
    constexpr double kKernelShareOfCodec = 0.5;
    if (nvfp4_vs_bf16.relative_l2 <= 0.0 ||
        kernel_vs_nvfp4.relative_l2 > kKernelShareOfCodec * nvfp4_vs_bf16.relative_l2) {
        std::cerr << label << ": kernel rel_l2=" << kernel_vs_nvfp4.relative_l2
                  << " is not inside NVFP4-vs-BF16 codec gap rel_l2=" << nvfp4_vs_bf16.relative_l2
                  << '\n';
        return 1;
    }
    if (nvfp4_vs_int8.relative_l2 <= 0.0 ||
        kernel_vs_nvfp4.relative_l2 > kKernelShareOfCodec * nvfp4_vs_int8.relative_l2) {
        std::cerr << label << ": kernel rel_l2=" << kernel_vs_nvfp4.relative_l2
                  << " is not inside NVFP4-vs-INT8 codec gap rel_l2=" << nvfp4_vs_int8.relative_l2
                  << '\n';
        return 1;
    }
    // Same logical K/V: E2M1/UE4M3-G16 is coarser than INT8-G64 versus BF16.
    if (nvfp4_vs_bf16.relative_l2 <= int8_vs_bf16.relative_l2) {
        std::cerr << label << ": NVFP4-vs-BF16 rel_l2=" << nvfp4_vs_bf16.relative_l2
                  << " is not larger than INT8-vs-BF16 rel_l2=" << int8_vs_bf16.relative_l2 << '\n';
        return 1;
    }
    return 0;
}

std::string case_label(const char* entry, const Geometry& geometry, DType dtype,
                       const AttentionCase& test_case, MappingPattern mapping) {
    return std::string(entry) + " " + geometry.name + " " + cache_name(dtype) +
           " mapping=" + mapping_name(mapping) + " T=" + std::to_string(test_case.tokens) +
           " keys=" + std::to_string(test_case.base + test_case.tokens) +
           " envelope_max=" + std::to_string(test_case.envelope_max);
}

void inject_codec_edges(const Geometry& geometry, std::int32_t tokens, std::vector<float>& k,
                        std::vector<float>& v) {
    if (tokens == 0) return;
    for (std::int32_t d = 0; d < kQuantGroup; ++d) {
        k[kv_input_index(geometry, 0, d, 0)]               = 0.0f;
        v[kv_input_index(geometry, 0, kQuantGroup + d, 0)] = 0.0f;
    }
    k[kv_input_index(geometry, geometry.kv_heads - 1, 0, tokens - 1)] = -1.0f;
    v[kv_input_index(geometry, geometry.kv_heads - 1, 0, tokens - 1)] = 1.0f;
}

int run_append_case(const Geometry& geometry, DType dtype, MappingPattern mapping,
                    std::uint32_t seed, std::int32_t tokens = 3, std::int32_t base = 63,
                    bool sage = false) {
    const std::int32_t max_context = base + tokens + 4;
    const std::size_t elements =
        static_cast<std::size_t>(kHeadDim) * static_cast<std::size_t>(geometry.kv_heads) * tokens;
    std::vector<float> k = make_bf16_values(elements, seed, -0.25f, 0.25f);
    std::vector<float> v = make_bf16_values(elements, seed + 1u, -1.0f, 1.0f);
    inject_codec_edges(geometry, tokens, k, v);
    const std::vector<std::uint16_t> k_bits = to_bf16_bits(k);
    const std::vector<std::uint16_t> v_bits = to_bf16_bits(v);
    std::vector<std::int32_t> positions(static_cast<std::size_t>(tokens));
    for (std::int32_t token = 0; token < tokens; ++token) {
        positions[static_cast<std::size_t>(token)] = base + token;
    }

    const HostCache initial = make_cache(geometry, dtype, max_context, seed + 10u, sage);
    HostCache expected      = initial;
    append_cache(expected, k, v, positions);
    DeviceCache cache(initial, mapping);

    GuardedDeviceBuffer dk(k_bits.size() * sizeof(std::uint16_t));
    GuardedDeviceBuffer dv(v_bits.size() * sizeof(std::uint16_t));
    GuardedDeviceBuffer dpositions(positions.size() * sizeof(std::int32_t));
    dk.copy_from_host(k_bits.data(), k_bits.size() * sizeof(std::uint16_t));
    dv.copy_from_host(v_bits.data(), v_bits.size() * sizeof(std::uint16_t));
    dpositions.copy_from_host(positions.data(), positions.size() * sizeof(std::int32_t));
    Tensor tk(dk.data(), DType::BF16, {kHeadDim, geometry.kv_heads, tokens});
    Tensor tv(dv.data(), DType::BF16, {kHeadDim, geometry.kv_heads, tokens});
    Tensor tp(dpositions.data(), DType::I32, {tokens});

    ops::gqa_kv_append(tk, tv, tp, cache.view(), nullptr);
    cuda_synchronize();

    const std::string label =
        std::string("gqa_kv_append ") + geometry.name + " " + cache_name(dtype) +
        (sage ? "(sage) " : " ") + "mapping=" + mapping_name(mapping);
    int failures = verify_cache(label, cache.snapshot(), expected);
    failures += verify_input(label + " k unchanged", dk, k_bits);
    failures += verify_input(label + " v unchanged", dv, v_bits);
    failures += verify_positions(label + " positions unchanged", dpositions, positions);
    failures += cache.verify_guards(label);
    return failures;
}

int run_a1_case(const Geometry& geometry, DType dtype, const AttentionCase& test_case,
                MappingPattern mapping, bool sage = false) {
    const std::int32_t total       = test_case.base + test_case.tokens;
    const std::int32_t max_context = static_cast<std::int32_t>(
        std::max<std::uint32_t>(static_cast<std::uint32_t>(total + 3), test_case.envelope_max));
    const std::size_t q_elements = static_cast<std::size_t>(kHeadDim) *
                                   static_cast<std::size_t>(geometry.q_heads) *
                                   static_cast<std::size_t>(test_case.tokens);
    const std::size_t kv_elements = static_cast<std::size_t>(kHeadDim) *
                                    static_cast<std::size_t>(geometry.kv_heads) *
                                    static_cast<std::size_t>(test_case.tokens);
    std::vector<float> q = make_bf16_values(q_elements, test_case.seed, -0.25f, 0.25f);
    std::vector<float> k = make_bf16_values(kv_elements, test_case.seed + 1u, -0.25f, 0.25f);
    std::vector<float> v = make_bf16_values(kv_elements, test_case.seed + 2u, -1.0f, 1.0f);
    if (sage && std::getenv("SAGE_DUMP") != nullptr) {
        const std::int32_t heads = geometry.kv_heads;
        for (std::int32_t t = 0; t < test_case.tokens; ++t) {
            std::cerr << "A1_INPUT t=" << t;
            for (std::int32_t h = 0; h < heads; ++h) {
                const std::size_t i16 = kv_input_index(geometry, h, 16, t);
                const std::size_t i17 = kv_input_index(geometry, h, 17, t);
                const std::size_t i32 = kv_input_index(geometry, h, 32, t);
                const std::size_t i33 = kv_input_index(geometry, h, 33, t);
                const std::size_t i29 = kv_input_index(geometry, h, 29, t);
                const std::size_t i30 = kv_input_index(geometry, h, 30, t);
                const std::size_t i250 = kv_input_index(geometry, h, 250, t);
                const std::size_t i251 = kv_input_index(geometry, h, 251, t);
                std::cerr << " h" << h << " d16=" << v[i16] << " d17=" << v[i17]
                          << " d29=" << v[i29] << " d30=" << v[i30]
                          << " d32=" << v[i32] << " d33=" << v[i33]
                          << " d250=" << v[i250] << " d251=" << v[i251];
            }
            std::cerr << '\n';
        }
    }
    inject_codec_edges(geometry, test_case.tokens, k, v);
    std::vector<std::int32_t> positions(static_cast<std::size_t>(test_case.tokens));
    for (std::int32_t token = 0; token < test_case.tokens; ++token) {
        positions[static_cast<std::size_t>(token)] = test_case.base + token;
    }
    // GQA_KEEP_FRAC: A/B the sage tile-skip lever (keep_frac in (0, 1] engages the
    // SpargeAttn meansim proxy on the A1 Prompt route; <= 0 or unset keeps the exact path).
    float keep_frac = 1.0f;
    if (const char* kf = std::getenv("GQA_KEEP_FRAC")) {
        const float parsed = std::strtof(kf, nullptr);
        if (parsed > 0.0f && parsed <= 1.0f) { keep_frac = parsed; }
    }

    const HostCache initial = make_cache(geometry, dtype, max_context, test_case.seed + 10u, sage);
    HostCache expected      = initial;
    append_cache(expected, k, v, positions);
    const std::vector<double> reference =
        sage ? sage_ideal_attention(q, expected, positions)
             : ideal_attention(q, expected, positions);
    const std::vector<double> int8_reference =
        dtype == DType::U8 && !sage
            ? twin_cache_attention(geometry, DType::I8, max_context, test_case.seed + 10u, q,
                                   positions, &k, &v)
            : std::vector<double>{};
    const std::vector<double> bf16_reference =
        dtype == DType::U8 && !sage
            ? twin_cache_attention(geometry, DType::BF16, max_context, test_case.seed + 10u, q,
                                   positions, &k, &v)
            : std::vector<double>{};
    DeviceCache cache(initial, mapping);
    if (sage && std::getenv("SAGE_DUMP") != nullptr) {
        // Decisive: pre-fill device scale plane vs host make_cache state.
        const auto pre = cache.snapshot();
        const std::size_t ns = std::min(pre.v_fp8.size(), initial.v_fp8.size());
        std::int64_t pre_diff = -1;
        for (std::size_t i = 0; i < ns; ++i) {
            if (pre.v_fp8[i] != initial.v_fp8[i]) { pre_diff = static_cast<std::int64_t>(i); break; }
        }
        std::cerr << "SAGE_PREFILL_DIFF scale " << pre_diff;
        if (pre_diff >= 0) {
            const int h = static_cast<int>(pre_diff / (16 * initial.logical_capacity));
            const int rem = static_cast<int>(pre_diff % (16 * initial.logical_capacity));
            std::cerr << " head=" << h << " d=" << rem / 4 << " kb=" << rem % 4
                      << " device=" << static_cast<int>(pre.v_fp8[static_cast<std::size_t>(pre_diff)])
                      << " host=" << static_cast<int>(initial.v_fp8[static_cast<std::size_t>(pre_diff)])
                      << '\n';
        } else {
            std::cerr << " (none — initial state matches)\n";
            const auto sc0 = [&](std::int32_t h, std::int32_t d, std::int32_t kb) {
                const std::size_t off =
                    static_cast<std::size_t>(h) * 16 * initial.logical_capacity + d * 4 + kb;
                return static_cast<int>(initial.v_fp8[off]);
            };
            std::cerr << "  init scale h1d29kb3=" << sc0(1, 29, 3)
                      << " h0d250kb3=" << sc0(0, 250, 3) << " h0d251kb3=" << sc0(0, 251, 3)
                      << '\n';
        }
    }

    const std::vector<std::uint16_t> q_bits = to_bf16_bits(q);
    const std::vector<std::uint16_t> k_bits = to_bf16_bits(k);
    const std::vector<std::uint16_t> v_bits = to_bf16_bits(v);
    GuardedDeviceBuffer dq(q_bits.size() * sizeof(std::uint16_t));
    GuardedDeviceBuffer dk(k_bits.size() * sizeof(std::uint16_t));
    GuardedDeviceBuffer dv(v_bits.size() * sizeof(std::uint16_t));
    GuardedDeviceBuffer dp(positions.size() * sizeof(std::int32_t));
    GuardedDeviceBuffer dtable_row(sizeof(std::int32_t));
    GuardedDeviceBuffer dout(q_bits.size() * sizeof(std::uint16_t));
    dq.copy_from_host(q_bits.data(), q_bits.size() * sizeof(std::uint16_t));
    dk.copy_from_host(k_bits.data(), k_bits.size() * sizeof(std::uint16_t));
    dv.copy_from_host(v_bits.data(), v_bits.size() * sizeof(std::uint16_t));
    dp.copy_from_host(positions.data(), positions.size() * sizeof(std::int32_t));
    const std::int32_t table_row = 0;
    dtable_row.copy_from_host(&table_row, sizeof(table_row));
    std::vector<std::uint16_t> output_canary(q_bits.size(), kOutputCanary);
    dout.copy_from_host(output_canary.data(), output_canary.size() * sizeof(std::uint16_t));

    Tensor tq(dq.data(), DType::BF16, {kHeadDim, geometry.q_heads, test_case.tokens});
    Tensor tk(dk.data(), DType::BF16, {kHeadDim, geometry.kv_heads, test_case.tokens});
    Tensor tv(dv.data(), DType::BF16, {kHeadDim, geometry.kv_heads, test_case.tokens});
    Tensor tp(dp.data(), DType::I32, {test_case.tokens});
    Tensor ttable_row(dtable_row.data(), DType::I32, {1});
    Tensor tout(dout.data(), DType::BF16, {kHeadDim, geometry.q_heads, test_case.tokens});
    const ops::GqaExecutionEnvelope envelope{static_cast<std::uint32_t>(total),
                                             test_case.envelope_max};
    const std::size_t workspace_bytes = ops::gqa_attention_workspace_capacity_bytes(
        geometry.q_heads, dtype, envelope, 1, test_case.tokens, test_case.tokens);
    GuardedDeviceBuffer workspace_buffer(std::max<std::size_t>(workspace_bytes, 256));
    WorkspaceArena workspace(DeviceSpan{workspace_buffer.data(), workspace_buffer.bytes()});

    if (sage && std::getenv("SAGE_DUMP") != nullptr) {
        // v-input buffer on device vs host at the t=0 (h1,d29) and (h0,d251) slots:
        const auto dv_bits = copy_from_guarded<std::uint16_t>(dv, v_bits.size());
        const auto idx = [&](std::int32_t h, std::int32_t d, std::int32_t t) {
            return static_cast<std::size_t>(d) +
                   static_cast<std::size_t>(256) *
                       (static_cast<std::size_t>(h) +
                        static_cast<std::size_t>(geometry.kv_heads) * static_cast<std::int32_t>(t));
        };
        const std::size_t iA = idx(1, 29, 0);
        const std::size_t iB = idx(0, 251, 0);
        std::cerr << "SAGE_DV dv[" << iA << "]=" << bf16_to_f32(dv_bits[iA])
                  << " host=" << v[iA] << " dv[" << iB << "]=" << bf16_to_f32(dv_bits[iB])
                  << " host=" << v[iB] << '\n';
    }
    ops::gqa_attention(tq, tk, tv, tp, Tensor{}, ttable_row, kAttentionScale, cache.batch_view(),
                       envelope, workspace, tout, nullptr, keep_frac);
    cuda_synchronize();

    const std::string label = case_label("gqa_attention", geometry, dtype, test_case, mapping);
    const std::vector<std::uint16_t> output_bits =
        copy_from_guarded<std::uint16_t>(dout, q_bits.size());
    const std::vector<double> actual = bf16_bits_to_double(output_bits);
    int failures = verify_attention(label, actual, reference, attention_criterion(dtype, sage));
    if (sage) { sage_floor_report(label, actual, reference, q, expected, positions); }
    if (dtype == DType::U8 && !sage) {
        failures += verify_nvfp4_kernel_inside_codec_gap(label, actual, reference, int8_reference,
                                                         bf16_reference);
    }
    failures += verify_cache(label, cache.snapshot(), expected);
    failures += verify_input(label + " q unchanged", dq, q_bits);
    failures += verify_input(label + " k unchanged", dk, k_bits);
    failures += verify_input(label + " v unchanged", dv, v_bits);
    failures += verify_positions(label + " positions unchanged", dp, positions);
    failures += verify_positions(label + " table row unchanged", dtable_row, {table_row});
    failures += dout.verify_guards((label + " output").c_str());
    failures += workspace_buffer.verify_guards((label + " workspace").c_str());
    if (workspace.used() != 0 || workspace.peak_used() != workspace_bytes) {
        std::cerr << label << ": workspace query/execution high-water mismatch\n";
        ++failures;
    }
    failures += cache.verify_guards(label);
    return failures;
}

int run_a3_case(const Geometry& geometry, DType dtype, const AttentionCase& test_case,
                MappingPattern mapping, bool sage = false) {
    const std::int32_t total       = test_case.base + test_case.tokens;
    const std::int32_t max_context = static_cast<std::int32_t>(
        std::max<std::uint32_t>(static_cast<std::uint32_t>(total + 3), test_case.envelope_max));
    const std::size_t q_elements = static_cast<std::size_t>(kHeadDim) *
                                   static_cast<std::size_t>(geometry.q_heads) *
                                   static_cast<std::size_t>(test_case.tokens);
    std::vector<float> q = make_bf16_values(q_elements, test_case.seed, -0.25f, 0.25f);
    std::vector<std::int32_t> positions(static_cast<std::size_t>(test_case.tokens));
    for (std::int32_t token = 0; token < test_case.tokens; ++token) {
        positions[static_cast<std::size_t>(token)] = test_case.base + token;
    }

    const HostCache cache_host =
        make_cache(geometry, dtype, max_context, test_case.seed + 10u, sage);
    const std::vector<double> reference =
        sage ? sage_ideal_attention(q, cache_host, positions)
             : ideal_attention(q, cache_host, positions);
    const std::vector<double> int8_reference =
        dtype == DType::U8 && !sage
            ? twin_cache_attention(geometry, DType::I8, max_context, test_case.seed + 10u, q,
                                   positions)
            : std::vector<double>{};
    const std::vector<double> bf16_reference =
        dtype == DType::U8 && !sage
            ? twin_cache_attention(geometry, DType::BF16, max_context, test_case.seed + 10u, q,
                                   positions)
            : std::vector<double>{};
    DeviceCache cache(cache_host, mapping);

    const std::vector<std::uint16_t> q_bits = to_bf16_bits(q);
    GuardedDeviceBuffer dq(q_bits.size() * sizeof(std::uint16_t));
    GuardedDeviceBuffer dp(positions.size() * sizeof(std::int32_t));
    GuardedDeviceBuffer dout(q_bits.size() * sizeof(std::uint16_t));
    dq.copy_from_host(q_bits.data(), q_bits.size() * sizeof(std::uint16_t));
    dp.copy_from_host(positions.data(), positions.size() * sizeof(std::int32_t));
    std::vector<std::uint16_t> output_canary(q_bits.size(), kOutputCanary);
    dout.copy_from_host(output_canary.data(), output_canary.size() * sizeof(std::uint16_t));

    Tensor tq(dq.data(), DType::BF16, {kHeadDim, geometry.q_heads, test_case.tokens});
    Tensor tp(dp.data(), DType::I32, {test_case.tokens});
    Tensor tout(dout.data(), DType::BF16, {kHeadDim, geometry.q_heads, test_case.tokens});
    const ops::GqaExecutionEnvelope envelope{static_cast<std::uint32_t>(total),
                                             test_case.envelope_max};
    const std::size_t workspace_bytes = ops::gqa_attention_workspace_capacity_bytes(
        geometry.q_heads, dtype, envelope, 1, test_case.tokens, test_case.tokens);
    GuardedDeviceBuffer workspace_buffer(std::max<std::size_t>(workspace_bytes, 256));
    WorkspaceArena workspace(DeviceSpan{workspace_buffer.data(), workspace_buffer.bytes()});

    ops::gqa_attention_cached(tq, tp, kAttentionScale, cache.view(), envelope, workspace, tout,
                              nullptr);
    cuda_synchronize();

    const std::string label = case_label("gqa_attention_cached", geometry, dtype, test_case, mapping);
    const std::vector<std::uint16_t> output_bits =
        copy_from_guarded<std::uint16_t>(dout, q_bits.size());
    const std::vector<double> actual = bf16_bits_to_double(output_bits);
    int failures = verify_attention(label, actual, reference, attention_criterion(dtype, sage));
    if (sage) { sage_floor_report(label, actual, reference, q, cache_host, positions); }
    if (dtype == DType::U8 && !sage) {
        failures += verify_nvfp4_kernel_inside_codec_gap(label, actual, reference, int8_reference,
                                                         bf16_reference);
    }
    failures += verify_cache(label + " cache unchanged", cache.snapshot(), cache_host);
    failures += verify_input(label + " q unchanged", dq, q_bits);
    failures += verify_positions(label + " positions unchanged", dp, positions);
    failures += dout.verify_guards((label + " output").c_str());
    failures += workspace_buffer.verify_guards((label + " workspace").c_str());
    if (workspace.used() != 0 || workspace.peak_used() != workspace_bytes) {
        std::cerr << label << ": workspace query/execution high-water mismatch\n";
        ++failures;
    }
    failures += cache.verify_guards(label);
    return failures;
}

struct BatchAttentionCase {
    std::int32_t width;
    std::vector<std::int32_t> contexts;
    std::vector<std::int32_t> valid_columns;
    std::vector<std::int32_t> table_rows;
    MappingPattern mapping;
    std::uint32_t seed;
};

std::vector<float> extract_request_columns(const std::vector<float>& source,
                                           std::size_t column_elements, std::int32_t width,
                                           std::int32_t request, std::int32_t valid) {
    const std::size_t begin = static_cast<std::size_t>(request) * width * column_elements;
    std::vector<float> result(static_cast<std::size_t>(valid) * column_elements);
    std::copy_n(source.begin() + static_cast<std::ptrdiff_t>(begin), result.size(), result.begin());
    return result;
}

void insert_request_columns(const std::vector<double>& source, std::size_t column_elements,
                            std::int32_t width, std::int32_t request,
                            std::vector<double>& destination) {
    const std::size_t begin = static_cast<std::size_t>(request) * width * column_elements;
    std::copy(source.begin(), source.end(),
              destination.begin() + static_cast<std::ptrdiff_t>(begin));
}

int verify_invalid_columns_zero(const std::string& label, std::span<const std::uint16_t> output,
                                const Geometry& geometry, std::int32_t width,
                                std::span<const std::int32_t> valid_columns) {
    int failures                      = 0;
    const std::size_t column_elements = static_cast<std::size_t>(kHeadDim) * geometry.q_heads;
    for (std::size_t batch = 0; batch < valid_columns.size(); ++batch) {
        for (std::int32_t token = valid_columns[batch]; token < width; ++token) {
            const std::size_t begin =
                (batch * static_cast<std::size_t>(width) + token) * column_elements;
            for (std::size_t element = 0; element < column_elements; ++element) {
                if (output[begin + element] != 0) {
                    if (failures == 0) {
                        std::cerr << label << ": invalid output column is not BF16 zero at row "
                                  << batch << " column " << token << '\n';
                    }
                    ++failures;
                }
            }
        }
    }
    return failures;
}

int run_batch_case(const Geometry& geometry, DType dtype, const BatchAttentionCase& test_case) {
    const std::int32_t batch = static_cast<std::int32_t>(test_case.contexts.size());
    if (batch <= 0 || test_case.valid_columns.size() != static_cast<std::size_t>(batch) ||
        test_case.table_rows.size() != static_cast<std::size_t>(batch)) {
        throw std::invalid_argument("invalid GQA batch test profile");
    }

    std::int32_t maximum_visible = 1;
    for (std::int32_t row = 0; row < batch; ++row) {
        maximum_visible =
            std::max(maximum_visible, test_case.contexts[static_cast<std::size_t>(row)] +
                                          test_case.valid_columns[static_cast<std::size_t>(row)]);
    }
    const std::int32_t max_context       = maximum_visible + 3;
    const std::size_t q_column_elements  = static_cast<std::size_t>(kHeadDim) * geometry.q_heads;
    const std::size_t kv_column_elements = static_cast<std::size_t>(kHeadDim) * geometry.kv_heads;
    const std::size_t columns            = static_cast<std::size_t>(test_case.width) * batch;
    std::vector<float> q =
        make_bf16_values(q_column_elements * columns, test_case.seed, -0.25f, 0.25f);
    std::vector<float> k =
        make_bf16_values(kv_column_elements * columns, test_case.seed + 1u, -0.25f, 0.25f);
    std::vector<float> v =
        make_bf16_values(kv_column_elements * columns, test_case.seed + 2u, -1.0f, 1.0f);
    inject_codec_edges(geometry, static_cast<std::int32_t>(columns), k, v);

    std::vector<std::int32_t> positions(columns, 0);
    for (std::int32_t row = 0; row < batch; ++row) {
        const std::int32_t valid = test_case.valid_columns[static_cast<std::size_t>(row)];
        for (std::int32_t token = 0; token < valid; ++token) {
            positions[static_cast<std::size_t>(row) * test_case.width + token] =
                test_case.contexts[static_cast<std::size_t>(row)] + token;
        }
        const std::int32_t padding_position =
            valid == 0 ? 0 : test_case.contexts[static_cast<std::size_t>(row)] + valid - 1;
        for (std::int32_t token = valid; token < test_case.width; ++token) {
            positions[static_cast<std::size_t>(row) * test_case.width + token] = padding_position;
        }
    }

    std::vector<HostCache> initial;
    initial.reserve(static_cast<std::size_t>(batch));
    for (std::int32_t row = 0; row < batch; ++row) {
        initial.push_back(
            make_cache(geometry, dtype, max_context, test_case.seed + 20u + 3u * row));
    }
    std::vector<HostCache> expected = initial;
    std::vector<double> reference(q_column_elements * columns, 0.0);
    for (std::int32_t request = 0; request < batch; ++request) {
        const std::int32_t valid = test_case.valid_columns[static_cast<std::size_t>(request)];
        if (valid == 0) { continue; }
        const std::int32_t table_row = test_case.table_rows[static_cast<std::size_t>(request)];
        std::vector<std::int32_t> row_positions(static_cast<std::size_t>(valid));
        std::copy_n(positions.begin() + static_cast<std::ptrdiff_t>(request * test_case.width),
                    valid, row_positions.begin());
        const std::vector<float> row_q =
            extract_request_columns(q, q_column_elements, test_case.width, request, valid);
        const std::vector<float> row_k =
            extract_request_columns(k, kv_column_elements, test_case.width, request, valid);
        const std::vector<float> row_v =
            extract_request_columns(v, kv_column_elements, test_case.width, request, valid);
        append_cache(expected[static_cast<std::size_t>(table_row)], row_k, row_v, row_positions);
        insert_request_columns(
            ideal_attention(row_q, expected[static_cast<std::size_t>(table_row)], row_positions),
            q_column_elements, test_case.width, request, reference);
    }

    BatchDeviceCache cache(initial, test_case.mapping);
    const std::vector<std::uint16_t> q_bits = to_bf16_bits(q);
    const std::vector<std::uint16_t> k_bits = to_bf16_bits(k);
    const std::vector<std::uint16_t> v_bits = to_bf16_bits(v);
    GuardedDeviceBuffer dq(q_bits.size() * sizeof(std::uint16_t));
    GuardedDeviceBuffer dk(k_bits.size() * sizeof(std::uint16_t));
    GuardedDeviceBuffer dv(v_bits.size() * sizeof(std::uint16_t));
    GuardedDeviceBuffer dp(positions.size() * sizeof(std::int32_t));
    GuardedDeviceBuffer dvalid(test_case.valid_columns.size() * sizeof(std::int32_t));
    GuardedDeviceBuffer dtable_rows(test_case.table_rows.size() * sizeof(std::int32_t));
    GuardedDeviceBuffer dout(q_bits.size() * sizeof(std::uint16_t));
    dq.copy_from_host(q_bits.data(), q_bits.size() * sizeof(std::uint16_t));
    dk.copy_from_host(k_bits.data(), k_bits.size() * sizeof(std::uint16_t));
    dv.copy_from_host(v_bits.data(), v_bits.size() * sizeof(std::uint16_t));
    dp.copy_from_host(positions.data(), positions.size() * sizeof(std::int32_t));
    dvalid.copy_from_host(test_case.valid_columns.data(),
                          test_case.valid_columns.size() * sizeof(std::int32_t));
    dtable_rows.copy_from_host(test_case.table_rows.data(),
                               test_case.table_rows.size() * sizeof(std::int32_t));
    std::vector<std::uint16_t> output_canary(q_bits.size(), kOutputCanary);
    dout.copy_from_host(output_canary.data(), output_canary.size() * sizeof(std::uint16_t));

    Tensor tq(dq.data(), DType::BF16, {kHeadDim, geometry.q_heads, test_case.width, batch});
    Tensor tk(dk.data(), DType::BF16, {kHeadDim, geometry.kv_heads, test_case.width, batch});
    Tensor tv(dv.data(), DType::BF16, {kHeadDim, geometry.kv_heads, test_case.width, batch});
    Tensor tp(dp.data(), DType::I32, {test_case.width, batch});
    Tensor tvalid(dvalid.data(), DType::I32, {batch});
    Tensor ttable_rows(dtable_rows.data(), DType::I32, {batch});
    Tensor tout(dout.data(), DType::BF16, {kHeadDim, geometry.q_heads, test_case.width, batch});
    const ops::GqaExecutionEnvelope envelope{static_cast<std::uint32_t>(maximum_visible),
                                             static_cast<std::uint32_t>(maximum_visible)};
    const std::size_t workspace_bytes = ops::gqa_attention_workspace_capacity_bytes(
        geometry.q_heads, dtype, envelope, batch, test_case.width, test_case.width);
    GuardedDeviceBuffer workspace_buffer(std::max<std::size_t>(workspace_bytes, 256));
    WorkspaceArena workspace(DeviceSpan{workspace_buffer.data(), workspace_buffer.bytes()});

    const bool masked = std::any_of(test_case.valid_columns.begin(), test_case.valid_columns.end(),
                                    [&](std::int32_t valid) { return valid != test_case.width; });
    ops::gqa_attention(tq, tk, tv, tp, masked ? tvalid : Tensor{}, ttable_rows, kAttentionScale,
                       cache.view(), envelope, workspace, tout, nullptr);
    cuda_synchronize();

    const std::string label = std::string("gqa_attention batch ") + geometry.name + " " +
                              cache_name(dtype) + " mapping=" + mapping_name(test_case.mapping) +
                              " B=" + std::to_string(batch) +
                              " W=" + std::to_string(test_case.width);
    const std::vector<std::uint16_t> output_bits =
        copy_from_guarded<std::uint16_t>(dout, q_bits.size());
    int failures = verify_attention(label, bf16_bits_to_double(output_bits), reference,
                                    attention_criterion(dtype));
    failures += verify_invalid_columns_zero(label, output_bits, geometry, test_case.width,
                                            test_case.valid_columns);
    failures += cache.verify(label, expected);
    failures += verify_input(label + " q unchanged", dq, q_bits);
    failures += verify_input(label + " k unchanged", dk, k_bits);
    failures += verify_input(label + " v unchanged", dv, v_bits);
    failures += verify_positions(label + " positions unchanged", dp, positions);
    if (masked) {
        failures +=
            verify_positions(label + " valid columns unchanged", dvalid, test_case.valid_columns);
    }
    failures +=
        verify_positions(label + " table rows unchanged", dtable_rows, test_case.table_rows);
    failures += dout.verify_guards((label + " output").c_str());
    failures += workspace_buffer.verify_guards((label + " workspace").c_str());
    if (workspace.used() != 0 || workspace.peak_used() != workspace_bytes) {
        std::cerr << label << ": workspace query/execution high-water mismatch\n";
        ++failures;
    }
    return failures;
}

int run_batch_cases() {
    int failures = 0;
    failures += run_batch_case(kGeometries[0], DType::I8,
                               {6, {127}, {3}, {0}, MappingPattern::Identity, 499u});
    failures += run_batch_case(kGeometries[0], DType::BF16,
                               {16, {49}, {7}, {0}, MappingPattern::Identity, 500u});
    failures += run_batch_case(kGeometries[0], DType::BF16,
                               {1, {63, 2048}, {1, 1}, {1, 0}, MappingPattern::Fragmented, 501u});
    failures += run_batch_case(kGeometries[1], DType::I8,
                               {1,
                                {0, 31, 63, 127, 511, 1023, 2047, 4095},
                                {1, 1, 1, 1, 1, 1, 1, 1},
                                {7, 0, 5, 2, 6, 1, 4, 3},
                                MappingPattern::Identity,
                                502u});
    failures +=
        run_batch_case(kGeometries[0], DType::I8,
                       {6, {61, 127, 511}, {6, 3, 0}, {2, 0, 1}, MappingPattern::Fragmented, 503u});
    failures += run_batch_case(kGeometries[1], DType::BF16,
                               {16, {49, 2041}, {16, 7}, {1, 0}, MappingPattern::Identity, 504u});
    return failures;
}

int run_geometry(const Geometry& geometry) {
    int failures = 0;
    const bool sage_only = std::getenv("GQA_SAGE_ONLY") != nullptr;
    for (const DType dtype : {DType::BF16, DType::I8, DType::U8}) {
        if (sage_only && dtype != DType::U8) { continue; }
        for (const MappingPattern mapping :
             {MappingPattern::Identity, MappingPattern::Offset, MappingPattern::Fragmented}) {
            if (!sage_only) {
                failures += run_append_case(geometry, dtype, mapping, 100u + geometry.q_heads);
                failures += run_a1_case(geometry, dtype, {6, 61, 67, 190u}, mapping);
                failures += run_a3_case(geometry, dtype, {1, 128, 129, 191u}, mapping);
            }
        }
        if ((dtype == DType::I8 || dtype == DType::U8) && !sage_only) {
            failures += run_append_case(geometry, dtype, MappingPattern::Fragmented,
                                        150u + geometry.q_heads, 129, 61);
        }

        if (!sage_only) {
            const AttentionCase a1_cases[] = {
            {1, 0, 1, 201u},    {6, 17, 23, 202u},   {7, 17, 512, 203u},
            {17, 31, 48, 204u}, {66, 63, 129, 205u},
        };
            for (const AttentionCase& test_case : a1_cases) {
                failures += run_a1_case(geometry, dtype, test_case, MappingPattern::Identity);
            }

            const AttentionCase a3_cases[] = {
                {1, 31, 32, 301u},
                {7, 17, 512, 302u},
                {17, 31, 48, 303u},
            };
            for (const AttentionCase& test_case : a3_cases) {
                failures += run_a3_case(geometry, dtype, test_case, MappingPattern::Identity);
            }
        }

        if (dtype == DType::U8 && !sage_only) {
            failures +=
                run_a1_case(geometry, dtype, {128, 64, 256, 210u}, MappingPattern::Identity);
            failures +=
                run_a1_case(geometry, dtype, {129, 64, 256, 211u}, MappingPattern::Identity);
            failures +=
                run_a1_case(geometry, dtype, {128, 64, 256, 212u}, MappingPattern::Fragmented);
            failures +=
                run_a3_case(geometry, dtype, {4, 512, 1024, 310u}, MappingPattern::Identity);
            failures +=
                run_a3_case(geometry, dtype, {4, 2048, 4096, 311u}, MappingPattern::Identity);
            failures +=
                run_a3_case(geometry, dtype, {4, 2048, 4096, 312u}, MappingPattern::Fragmented);
        }

        if (dtype == DType::U8) {
            // Sage (SageAttention3-style FP4-PV) pass: d-major V-scale cache + FP4 PV GEMM.
            // The fill case byte-checks the running-max rescale; A1 covers the prefill s3
            // kernel (fill + attention); A3 T=4 covers the decode s3 partial (Bc=32 and the
            // Bc=64 dynamic-arena tier). GQA_SAGE_FAST=1 drops the two T=128 multi-tile A1
            // cases (the expensive FP64 reference matrix); single-tile + decode coverage
            // remains in the fast tier.
            const bool sage_fast = std::getenv("GQA_SAGE_FAST") != nullptr;
            failures += run_append_case(geometry, dtype, MappingPattern::Identity, 500u +
                                                                                           geometry.q_heads,
                                        3, 57, /*sage=*/true);
            failures += run_append_case(geometry, dtype, MappingPattern::Identity, 501u +
                                                                                           geometry.q_heads,
                                        3, 121, /*sage=*/true);
            failures += run_a1_case(geometry, dtype, {6, 55, 64, 510u}, MappingPattern::Identity,
                                    /*sage=*/true);
            if (!sage_fast) {
                failures +=
                    run_a1_case(geometry, dtype, {128, 64, 256, 511u}, MappingPattern::Identity,
                                /*sage=*/true);
            }
            failures +=
                run_a3_case(geometry, dtype, {4, 512, 1024, 512u}, MappingPattern::Identity,
                            /*sage=*/true);
            failures +=
                run_a3_case(geometry, dtype, {4, 2048, 4096, 513u}, MappingPattern::Identity,
                            /*sage=*/true);
            failures +=
                run_a1_case(geometry, dtype, {12, 0, 64, 514u}, MappingPattern::Identity, true);
            if (!sage_fast) {
                failures +=
                    run_a1_case(geometry, dtype, {128, 0, 128, 515u}, MappingPattern::Identity,
                                true);
            }
        }

        if (geometry.q_heads == 16 && !sage_only) {
            // Loose execution envelopes straddle the two registered host-resource frontiers.
            // Device positions, not these bounds, continue to define the oracle result.
            failures += run_a1_case(geometry, dtype, {7, 17, 513, 401u}, MappingPattern::Identity);
            failures += run_a3_case(geometry, dtype, {7, 17, 513, 402u}, MappingPattern::Identity);
            failures +=
                run_a3_case(geometry, dtype, {16, 17, 1024, 403u}, MappingPattern::Identity);
            failures +=
                run_a3_case(geometry, dtype, {16, 17, 1025, 404u}, MappingPattern::Identity);
        }
    }
    return failures;
}

int verify_workspace_capacity_contract() {
    int failures = 0;
    for (const DType dtype : {DType::BF16, DType::I8, DType::U8}) {
        constexpr ops::GqaExecutionEnvelope envelope{1, 1025};
        const std::size_t interval =
            ops::gqa_attention_workspace_capacity_bytes(16, dtype, envelope, 1, 1, 17);
        std::size_t witness = 0;
        for (std::int32_t tokens = 1; tokens <= 17; ++tokens) {
            witness = std::max(witness, ops::gqa_attention_workspace_capacity_bytes(
                                            16, dtype, envelope, 1, tokens, tokens));
        }
        if (interval != witness) {
            std::cerr << "gqa_attention interval capacity has no exact route witness\n";
            ++failures;
        }
    }
    try {
        (void)ops::gqa_attention_workspace_capacity_bytes(
            16, DType::BF16, {1, ops::kGqaAttentionMaximumVisibleKeys}, 1, 1, 1);
    } catch (const std::invalid_argument&) {
        std::cerr << "gqa_attention rejected its maximum visible-key envelope\n";
        ++failures;
    }
    try {
        (void)ops::gqa_attention_workspace_capacity_bytes(
            16, DType::BF16, {1, ops::kGqaAttentionMaximumVisibleKeys + 1}, 1, 1, 1);
        std::cerr << "gqa_attention accepted an envelope outside the launcher domain\n";
        ++failures;
    } catch (const std::invalid_argument&) {}
    return failures;
}

} // namespace

// ---------------------------------------------------------------------------
// s3 prefill op-dump (tools/kdev --s3-dump): run one representative multi-tile
// sage A1 case through the gqa_attention_s3_dump side-band, compare the
// kernel's named intermediates against a tile-exact FP64 reference (the
// kernel's own online-softmax + e2m1/e4m3 P-quant pipeline re-implemented in
// double), and write a compact first-divergence JSON for tools/kdev/diff.py.
//
// Stage semantics (pipeline order):
//   score   raw QK dot per (row, in-tile key); -INF = causally masked
//   psf     e4m3 P-block scale byte (per 16-key block): S = 448*exp2((m_blk-nm)*sl2)
//   p_code  e2m1 P nibble (per key): RNE(P/S), P = 2688*exp2((s-nm)*sl2), in [0,6]
//   v_scale e4m3 V-block scale byte (per (d, 16-key block)) — a byte copy of the
//           cache plane, so ANY diff is a V-scale indexing bug (NoVf16 class)
//   m/l     running max / running L (amplified, tile frame) after the tile
//   acc     PV accumulator after the tile's mma, in the tile's running-max frame
// A correct kernel: float stages within ~1e-4 (FP32 mma noise), code stages
// only 1-step e2m1/e4m3 flips (the FP4-P quant floor), and acc within ~1e-3 of
// the FP64 recompute using the KERNEL'S OWN dumped codes (isolates the PV path
// from the P-quant floor).
// ---------------------------------------------------------------------------

namespace {

constexpr double kS3Sl2  = kAttentionScale * 1.4426950408889634;  // kernel scale_l2 = scale*Log2E (exp2 arg over the RAW score)
constexpr double kS3Amp  = 2688.0;  // 448 * 6: P amplification (exp2 arg constant)
constexpr double kS3Smax = 448.0;   // = 2^kGqaS3SfLog2 (P-block scale ceiling)
constexpr std::int32_t kS3Rows = 128;  // Br: q_block 0 row extent
constexpr std::int32_t kS3Keys = 64;   // Bc: keys per tile
constexpr std::int32_t kS3NB   = 4;    // 16-key P blocks per tile
constexpr double kS3NInf = -1e300;     // -INFINITY sentinel (JSON-safe)

struct S3First {
    bool found = false;
    std::int32_t h = -1, t = -1, r = -1, c = -1;
    double kernel = 0.0, ref = 0.0, rel = 0.0;
};

void s3_consider(S3First& f, std::int32_t h, std::int32_t t, std::int32_t r, int c, double kv,
                 double rv, double rel) {
    if (!f.found) { f = S3First{true, h, t, r, c, kv, rv, rel}; }
}

std::string s3_first_json(const S3First& f, const char* dim) {
    if (!f.found) { return "null"; }
    std::ostringstream os;
    os << "{\"h\":" << f.h << ",\"t\":" << f.t << ",\"r\":" << f.r << ",\"" << dim
       << "\":" << f.c << ",\"kernel\":" << f.kernel << ",\"ref\":" << f.ref
       << ",\"rel\":" << f.rel << "}";
    return os.str();
}

} // namespace

int s3_dump_case(const Geometry& geometry, const AttentionCase& test_case, const char* json_path) {
    const std::int32_t heads = geometry.q_heads;
    const std::int32_t group = geometry.query_group();
    const std::int32_t total = test_case.base + test_case.tokens;
    const std::int32_t max_context = static_cast<std::int32_t>(
        std::max<std::uint32_t>(static_cast<std::uint32_t>(total + 3), test_case.envelope_max));
    const std::int32_t rows    = std::min(kS3Rows, test_case.tokens);
    const std::int32_t max_tiles = (max_context + kS3Keys - 1) / kS3Keys;

    // --- inputs (mirror run_a1_case exactly: same seeds => same cache) --------
    const std::size_t q_elements  = static_cast<std::size_t>(kHeadDim) *
                                   static_cast<std::size_t>(heads) *
                                   static_cast<std::size_t>(test_case.tokens);
    const std::size_t kv_elements = static_cast<std::size_t>(kHeadDim) *
                                    static_cast<std::size_t>(geometry.kv_heads) *
                                    static_cast<std::size_t>(test_case.tokens);
    std::vector<float> q = make_bf16_values(q_elements, test_case.seed, -0.25f, 0.25f);
    std::vector<float> k = make_bf16_values(kv_elements, test_case.seed + 1u, -0.25f, 0.25f);
    std::vector<float> v = make_bf16_values(kv_elements, test_case.seed + 2u, -1.0f, 1.0f);
    inject_codec_edges(geometry, test_case.tokens, k, v);
    std::vector<std::int32_t> positions(static_cast<std::size_t>(test_case.tokens));
    for (std::int32_t token = 0; token < test_case.tokens; ++token) {
        positions[static_cast<std::size_t>(token)] = test_case.base + token;
    }
    const HostCache initial =
        make_cache(geometry, DType::U8, max_context, test_case.seed + 10u, /*sage=*/true);
    HostCache expected = initial;
    append_cache(expected, k, v, positions);

    // --- device tensors (mirror run_a1_case) ---------------------------------
    DeviceCache cache(initial, MappingPattern::Identity);
    const std::vector<std::uint16_t> q_bits = to_bf16_bits(q);
    const std::vector<std::uint16_t> k_bits = to_bf16_bits(k);
    const std::vector<std::uint16_t> v_bits = to_bf16_bits(v);
    GuardedDeviceBuffer dq(q_bits.size() * sizeof(std::uint16_t));
    GuardedDeviceBuffer dk(k_bits.size() * sizeof(std::uint16_t));
    GuardedDeviceBuffer dv(v_bits.size() * sizeof(std::uint16_t));
    GuardedDeviceBuffer dp(positions.size() * sizeof(std::int32_t));
    GuardedDeviceBuffer dtable_row(sizeof(std::int32_t));
    GuardedDeviceBuffer dout(q_bits.size() * sizeof(std::uint16_t));
    dq.copy_from_host(q_bits.data(), q_bits.size() * sizeof(std::uint16_t));
    dk.copy_from_host(k_bits.data(), k_bits.size() * sizeof(std::uint16_t));
    dv.copy_from_host(v_bits.data(), v_bits.size() * sizeof(std::uint16_t));
    dp.copy_from_host(positions.data(), positions.size() * sizeof(std::int32_t));
    const std::int32_t table_row = 0;
    dtable_row.copy_from_host(&table_row, sizeof(table_row));
    dout.copy_from_host(std::vector<std::uint16_t>(q_bits.size(), kOutputCanary).data(),
                        q_bits.size() * sizeof(std::uint16_t));
    Tensor tq(dq.data(), DType::BF16, {kHeadDim, heads, test_case.tokens});
    Tensor tk(dk.data(), DType::BF16, {kHeadDim, geometry.kv_heads, test_case.tokens});
    Tensor tv(dv.data(), DType::BF16, {kHeadDim, geometry.kv_heads, test_case.tokens});
    Tensor tp(dp.data(), DType::I32, {test_case.tokens});
    Tensor ttable_row(dtable_row.data(), DType::I32, {1});
    Tensor tout(dout.data(), DType::BF16, {kHeadDim, heads, test_case.tokens});
    const ops::GqaExecutionEnvelope envelope{static_cast<std::uint32_t>(total),
                                             test_case.envelope_max};
    const std::size_t workspace_bytes = ops::gqa_attention_workspace_capacity_bytes(
        heads, DType::U8, envelope, 1, test_case.tokens, test_case.tokens);
    GuardedDeviceBuffer workspace_buffer(std::max<std::size_t>(workspace_bytes, 256));
    WorkspaceArena workspace(DeviceSpan{workspace_buffer.data(), workspace_buffer.bytes()});

    // --- op-dump buffers -------------------------------------------------------
    const std::size_t n_score = static_cast<std::size_t>(heads) * max_tiles * rows * kS3Keys;
    const std::size_t n_psf   = static_cast<std::size_t>(heads) * max_tiles * rows * kS3NB;
    const std::size_t n_vsc   = static_cast<std::size_t>(heads) * max_tiles * kHeadDim * kS3NB;
    const std::size_t n_ml    = static_cast<std::size_t>(heads) * max_tiles * rows;
    const std::size_t n_acc   = n_ml * kHeadDim;
    const std::size_t n_vt    = static_cast<std::size_t>(heads) * max_tiles * kHeadDim * 32;
    GuardedDeviceBuffer b_score(n_score * sizeof(float));
    GuardedDeviceBuffer b_pcode(n_score * sizeof(std::uint8_t));
    GuardedDeviceBuffer b_psf(n_psf * sizeof(std::uint8_t));
    GuardedDeviceBuffer b_vsc(n_vsc * sizeof(std::uint8_t));
    GuardedDeviceBuffer b_vt(n_vt);
    GuardedDeviceBuffer b_m(n_ml * sizeof(float));
    GuardedDeviceBuffer b_l(n_ml * sizeof(float));
    GuardedDeviceBuffer b_acc(n_acc * sizeof(float));
    GuardedDeviceBuffer b_keep(static_cast<std::size_t>(heads) * max_tiles * sizeof(std::int32_t));
    GuardedDeviceBuffer b_tcnt(heads * sizeof(std::int32_t));
    ops::GqaS3PrefillDump dump{
        max_tiles,
        reinterpret_cast<float*>(b_score.data()),
        reinterpret_cast<std::uint8_t*>(b_pcode.data()),
        reinterpret_cast<std::uint8_t*>(b_psf.data()),
        reinterpret_cast<std::uint8_t*>(b_vsc.data()),
        reinterpret_cast<std::uint8_t*>(b_vt.data()),
        reinterpret_cast<float*>(b_m.data()),
        reinterpret_cast<float*>(b_l.data()),
        reinterpret_cast<float*>(b_acc.data()),
        reinterpret_cast<std::int32_t*>(b_keep.data()),
        reinterpret_cast<std::int32_t*>(b_tcnt.data()),
    };
    ops::gqa_attention_s3_dump(tq, tk, tv, tp, Tensor{}, ttable_row, kAttentionScale,
                               cache.batch_view(), envelope, workspace, tout, nullptr, 1.0f, dump);
    cuda_synchronize();
    auto h_score = from_device<float>(b_score.data(), n_score);
    auto h_pcode = from_device<std::uint8_t>(b_pcode.data(), n_score);
    auto h_psf   = from_device<std::uint8_t>(b_psf.data(), n_psf);
    auto h_vsc   = from_device<std::uint8_t>(b_vsc.data(), n_vsc);
    auto h_vt    = from_device<std::uint8_t>(b_vt.data(), n_vt);
    auto h_m     = from_device<float>(b_m.data(), n_ml);
    auto h_l     = from_device<float>(b_l.data(), n_ml);
    auto h_acc   = from_device<float>(b_acc.data(), n_acc);
    auto h_keep  = from_device<std::int32_t>(b_keep.data(), static_cast<std::size_t>(heads) * max_tiles);
    auto h_tcnt  = from_device<std::int32_t>(b_tcnt.data(), heads);
    int guard_failures = 0;
    guard_failures += b_score.verify_guards("s3 dump score");
    guard_failures += b_pcode.verify_guards("s3 dump p_code");
    guard_failures += b_psf.verify_guards("s3 dump psf");
    guard_failures += b_vsc.verify_guards("s3 dump v_scale");
    guard_failures += b_vt.verify_guards("s3 dump v_t");
    guard_failures += b_m.verify_guards("s3 dump m");
    guard_failures += b_l.verify_guards("s3 dump l");
    guard_failures += b_acc.verify_guards("s3 dump acc");

    // --- FP64 reference precompute: dequantized K and V per (kv head, key) ----
    const std::int32_t kv_heads = geometry.kv_heads;
    const std::size_t kdim      = static_cast<std::size_t>(kHeadDim);
    std::vector<double> klog(static_cast<std::size_t>(kv_heads) * total * kdim);
    std::vector<double> vc(static_cast<std::size_t>(kv_heads) * total * kdim);
    std::vector<double> vsc(static_cast<std::size_t>(kv_heads) * total * kdim);
    for (std::int32_t kv = 0; kv < kv_heads; ++kv) {
        for (std::int32_t key = 0; key < total; ++key) {
            const std::size_t base_idx =
                (static_cast<std::size_t>(kv) * total + static_cast<std::size_t>(key)) * kdim;
            for (std::int32_t d = 0; d < kHeadDim; ++d) {
                klog[base_idx + static_cast<std::size_t>(d)] =
                    cache_value(expected, true, kv, key, d);
                const std::size_t packed =
                    nvfp4_code_index(geometry, expected.logical_capacity, kv, key, d / 2);
                const std::uint8_t byte = expected.v_u8[packed];
                const std::uint8_t nib   = (d & 1) ? (byte >> 4) : (byte & 0x0fu);
                const std::size_t vsi = sage_v_scale_host_index(
                    expected.logical_capacity, kv, key / kPagedKVPageSize, d,
                    (key % kPagedKVPageSize) / 16);
                vc[base_idx + static_cast<std::size_t>(d)]     = decode_e2m1_word(nib);
                vsc[base_idx + static_cast<std::size_t>(d)]    = decode_e4m3fn_word(expected.v_fp8[vsi]);
            }
        }
    }

    // --- per-head tile-exact reference + stage-by-stage comparison ------------
    S3First first_score, first_psf, first_pcode, first_vsc, first_vt, first_m, first_l, first_acc;
    double max_rel_score = 0.0, max_rel_m = 0.0, max_rel_l = 0.0;
    double max_rel_acc_kv = 0.0, max_rel_acc_ref = 0.0;
    std::size_t flips_psf = 0, two_plus_psf = 0, flips_pcode = 0, two_plus_pcode = 0;
    std::size_t diffs_vsc = 0;
    std::size_t diffs_vt  = 0;
    int max_diff_psf = 0, max_diff_pcode = 0;
    // PV block-permutation probe result (head 0, tile 0): which (P-scale-block,
    // V-scale-block) order the kernel's mma actually uses, identity = block nb
    // covers keys 16nb..16nb+15 in natural order.
    std::string acc_probe_json = "null";

    for (std::int32_t h = 0; h < heads; ++h) {
        const std::int32_t kv = h / group;
        const std::int32_t tiles = h_tcnt[h];
        if (tiles > max_tiles) { return 2; }
        std::vector<double> qlog(static_cast<std::size_t>(rows) * kdim, 0.0);
        for (std::int32_t r = 0; r < rows; ++r) {
            const std::vector<double> ql = q_nvfp4_log(geometry, q, h, r);
            std::copy(ql.begin(), ql.end(), qlog.begin() + r * kHeadDim);
        }
        std::vector<double> sref(static_cast<std::size_t>(rows) * kS3Keys, kS3NInf);
        std::vector<double> run_m(static_cast<std::size_t>(rows), kS3NInf);
        std::vector<double> run_l(static_cast<std::size_t>(rows), 0.0);
        std::vector<double> acc_ref(static_cast<std::size_t>(rows) * kdim, 0.0);
        std::vector<double> acc_kv(static_cast<std::size_t>(rows) * kdim, 0.0);
        double g_rel = 0.0;
        int g_t = -1, g_r = -1, g_d = -1;
        double g_k = 0.0, g_m = 0.0;
        std::size_t n_gt_1e3 = 0, n_gt_1e2 = 0;

        for (std::int32_t t = 0; t < tiles; ++t) {
            const std::size_t dht_t = static_cast<std::size_t>(h) * max_tiles + static_cast<std::size_t>(t);
            const std::int32_t kb = h_keep[static_cast<std::size_t>(h) * max_tiles + static_cast<std::size_t>(t)];
            const std::int32_t k0 = kb * kS3Keys;
            // raw QK dot (FP64 dequantized Q . dequantized K), causally masked
            for (std::int32_t r = 0; r < rows; ++r) {
                const double* ql  = &qlog[static_cast<std::size_t>(r) * kdim];
                const double* klk = &klog[(static_cast<std::size_t>(kv) * total +
                                            static_cast<std::size_t>(k0)) * kdim];
                for (std::int32_t i = 0; i < kS3Keys; ++i) {
                    const std::int32_t key = k0 + i;
                    double* s = &sref[static_cast<std::size_t>(r) * kS3Keys + static_cast<std::size_t>(i)];
                    if (key > test_case.base + r) { *s = kS3NInf; continue; }
                    double dot = 0.0;
                    const double* kkey = klk + static_cast<std::size_t>(i) * kdim;
                    for (std::int32_t d = 0; d < kHeadDim; ++d) { dot += ql[d] * kkey[d]; }
                    *s = dot;
                }
            }
            // tile max, running max update, P-quant codes, L, and PV (both ref-code
            // and kernel-code variants so a PV-path bug is isolated from the floor)
            std::vector<double> nm(static_cast<std::size_t>(rows)), alpha(static_cast<std::size_t>(rows));
            std::vector<double> psf_ref_dec(static_cast<std::size_t>(rows) * kS3NB, 0.0);
            std::vector<double> psf_kv_dec(static_cast<std::size_t>(rows) * kS3NB, 0.0);
            std::vector<std::uint8_t> psf_ref_byte(static_cast<std::size_t>(rows) * kS3NB, 0);
            std::vector<double> p_ref(static_cast<std::size_t>(rows) * kS3Keys, 0.0);
            std::vector<std::uint8_t> p_ref_nib(static_cast<std::size_t>(rows) * kS3Keys, 0);
            for (std::int32_t r = 0; r < rows; ++r) {
                double tmax = kS3NInf;
                for (std::int32_t i = 0; i < kS3Keys; ++i) {
                    tmax = std::max(tmax, sref[static_cast<std::size_t>(r) * kS3Keys + static_cast<std::size_t>(i)]);
                }
                const double nm_new = std::max(run_m[r], tmax);
                nm[r]   = nm_new;
                alpha[r] = (run_m[r] < -1e299) ? 0.0 : std::exp2((run_m[r] - nm_new) * kS3Sl2);
            }
            for (std::int32_t r = 0; r < rows; ++r) {
                double l_add = 0.0;
                for (std::int32_t nb = 0; nb < kS3NB; ++nb) {
                    double bm = kS3NInf;
                    for (std::int32_t j = 0; j < 16; ++j) {
                        bm = std::max(bm, sref[static_cast<std::size_t>(r) * kS3Keys + static_cast<std::size_t>(nb * 16 + j)]);
                    }
                    const double S = (bm < -1e299) ? 0.0 : kS3Smax * std::exp2((bm - nm[r]) * kS3Sl2);
                    const std::uint8_t psf_byte = (S == 0.0) ? 0 : encode_e4m3fn_rne(static_cast<float>(S));
                    const double psf_dec = decode_e4m3fn_word(psf_byte);
                    const std::size_t psf_idx = static_cast<std::size_t>(r) * kS3NB + static_cast<std::size_t>(nb);
                    psf_ref_dec[psf_idx] = psf_dec;
                    psf_kv_dec[psf_idx]  = decode_e4m3fn_word(h_psf[dht_t * rows * kS3NB + psf_idx]);
                    psf_ref_byte[psf_idx] = psf_byte;
                    for (std::int32_t j = 0; j < 16; ++j) {
                        const std::size_t idx = static_cast<std::size_t>(r) * kS3Keys + static_cast<std::size_t>(nb * 16 + j);
                        const bool masked = sref[idx] < -1e299;
                        const double P = masked ? 0.0 : kS3Amp * std::exp2((sref[idx] - nm[r]) * kS3Sl2);
                        const double ratio = psf_dec > 0.0 ? std::min(6.0, P / psf_dec) : 0.0;
                        const std::uint8_t nib_ref = encode_e2m1_rne(static_cast<float>(ratio));
                        p_ref[idx] = decode_e2m1_word(nib_ref);
                        p_ref_nib[idx] = nib_ref;
                        l_add += P;
                    }
                }
                run_l[r] = alpha[r] * run_l[r] + l_add;
                run_m[r] = nm[r];
            }
            // PV accumulator update (tile frame): acc = alpha*acc_prev + sum_nb psf*v_scale*(p_code.v_code)
            std::vector<double> acc_ref_new(static_cast<std::size_t>(rows) * kdim, 0.0);
            std::vector<double> acc_kv_new(static_cast<std::size_t>(rows) * kdim, 0.0);
            for (std::int32_t r = 0; r < rows; ++r) {
                const std::size_t ridx = static_cast<std::size_t>(r);
                for (std::int32_t d = 0; d < kHeadDim; ++d) {
                    double pv_ref = 0.0, pv_kv = 0.0;
                    for (std::int32_t nb = 0; nb < kS3NB; ++nb) {
                        const std::size_t vbase =
                            (static_cast<std::size_t>(kv) * total +
                             static_cast<std::size_t>(k0 + nb * 16)) * kdim + static_cast<std::size_t>(d);
                        double ss_ref = 0.0, ss_kv = 0.0;
                        for (std::int32_t j = 0; j < 16; ++j) {
                            const std::size_t sidx = ridx * kS3Keys + static_cast<std::size_t>(nb * 16 + j);
                            const double vcv = vc[vbase + static_cast<std::size_t>(j) * kdim];
                            ss_ref += p_ref[sidx] * vcv;
                            ss_kv  += decode_e2m1_word(h_pcode[dht_t * rows * kS3Keys + sidx]) * vcv;
                        }
                        const std::size_t pidx = ridx * kS3NB + static_cast<std::size_t>(nb);
                        const double vsv = vsc[vbase];
                        pv_ref += psf_ref_dec[pidx] * vsv * ss_ref;
                        pv_kv  += psf_kv_dec[pidx] * vsv * ss_kv;
                    }
                    const std::size_t didx = ridx * kdim + static_cast<std::size_t>(d);
                    acc_ref_new[didx] = alpha[r] * acc_ref[didx] + pv_ref;
                    acc_kv_new[didx]  = alpha[r] * acc_kv[didx] + pv_kv;
                }
            }

            // Scale-decode convention probe: the block-scaled mma's scale operand is
            // ue4m3 (unsigned) while the cached scale bytes are e4m3fn-encoded
            // (signed). Which (psf, vsc) decode convention reproduces the kernel's
            // pv1 (r=0, d=0) tells us what the hardware actually applies.
            if (getenv("S3_DUMP_DEBUG") != nullptr && h == 0 && t == 1) {
                const std::size_t dht_probe = static_cast<std::size_t>(h) * max_tiles + static_cast<std::size_t>(t);
                const auto ue4m3d = [](std::uint8_t b) -> double {
                    if (b == 0xFFu) { return 1e300; }
                    return std::ldexp(1.0 + (b & 0x7u) / 4.0, static_cast<int>((b >> 3) & 0xF) - 8);
                };
                double pv1_fn_fn = 0.0, pv1_ue_ue = 0.0, pv1_ue_fn = 0.0, pv1_fn_ue = 0.0;
                std::uint8_t psf4[4] = {0, 0, 0, 0}, vsb4[4] = {0, 0, 0, 0};
                for (std::int32_t nb = 0; nb < kS3NB; ++nb) {
                    const std::uint8_t psfb =
                        h_psf[static_cast<std::size_t>(dht_probe) * rows * kS3NB + static_cast<std::size_t>(nb)];
                    const std::uint8_t vsb = expected.v_fp8[sage_v_scale_host_index(
                        expected.logical_capacity, kv, (k0 + nb * 16) / kPagedKVPageSize, 0,
                        ((k0 + nb * 16) % kPagedKVPageSize) / 16)];
                    psf4[nb] = psfb;
                    vsb4[nb] = vsb;
                    double ss = 0.0;
                    for (std::int32_t j = 0; j < 16; ++j) {
                        ss += decode_e2m1_word(
                                 h_pcode[static_cast<std::size_t>(dht_probe) * rows * kS3Keys +
                                         static_cast<std::size_t>(nb * 16 + j)]) *
                             vc[(static_cast<std::size_t>(kv) * total +
                                 static_cast<std::size_t>(k0 + nb * 16 + j)) * kdim + 0];
                    }
                    const double psf_fn = decode_e4m3fn_word(psfb);
                    const double psf_u  = ue4m3d(psfb);
                    const double vsc_fn = decode_e4m3fn_word(vsb);
                    const double vsc_u  = ue4m3d(vsb);
                    pv1_fn_fn += psf_fn * vsc_fn * ss;
                    pv1_ue_ue += psf_u * vsc_u * ss;
                    pv1_ue_fn += psf_u * vsc_fn * ss;
                    pv1_fn_ue += psf_fn * vsc_u * ss;
                }
                const double kp1 = h_acc[static_cast<std::size_t>(dht_probe) * rows * kHeadDim + 0];
                const double kp0 = h_acc[static_cast<std::size_t>(dht_probe - 1) * rows * kHeadDim + 0];
                std::fprintf(stderr, "SCALE_PROBE t1 r0 d0: psf=[%02x %02x %02x %02x] vsc=[%02x %02x %02x %02x]\n",
                             psf4[0], psf4[1], psf4[2], psf4[3], vsb4[0], vsb4[1], vsb4[2], vsb4[3]);
                std::fprintf(stderr,
                             "        kernel_pv1=%.4f  fn/fn=%.4f ue/ue=%.4f ue/fn=%.4f fn/ue=%.4f\n",
                             kp1 - kp0, pv1_fn_fn, pv1_ue_ue, pv1_ue_fn, pv1_fn_ue);
                // argmax of the true rel (kernel vs ref-code model) + pair-swap probe:
                // if the kernel swapped each e2m1 key pair (2j <-> 2j+1) in its P
                // codes, the swapped-P PV would match the kernel better.
                double worst_rel = 0.0;
                int wr = -1, wd = -1;
                double worst_k = 0.0, worst_m = 0.0;
                for (std::int32_t rr = 0; rr < rows; ++rr) {
                    for (std::int32_t dd = 0; dd < kHeadDim; ++dd) {
                        const std::size_t ax = static_cast<std::size_t>(rr) * kHeadDim + static_cast<std::size_t>(dd);
                        const double kvv = h_acc[dht_probe * rows * kHeadDim + ax];
                        const double mvv = acc_kv_new[ax]; // fixed: per-tile kernel codes
                        const double rel =
                            std::abs(kvv - mvv) / std::max({std::abs(kvv), std::abs(mvv), 1e-30});
                        if (rel > worst_rel) {
                            worst_rel = rel;
                            wr = rr;
                            wd = dd;
                            worst_k = kvv;
                            worst_m = mvv;
                        }
                    }
                }
                double pv1_swap = 0.0;
                {
                    const std::int32_t rr = wr;
                    for (std::int32_t nb = 0; nb < kS3NB; ++nb) {
                        const double psf = psf_kv_dec[static_cast<std::size_t>(rr) * kS3NB + static_cast<std::size_t>(nb)];
                        const std::size_t vbase =
                            (static_cast<std::size_t>(kv) * total + static_cast<std::size_t>(k0 + nb * 16)) * kdim +
                            static_cast<std::size_t>(wd);
                        double ss = 0.0;
                        for (std::int32_t j = 0; j < 16; ++j) {
                            // swapped pairing: P of key (2j+1) x V of key (2j)
                            const std::size_t sidx =
                                static_cast<std::size_t>(rr) * kS3Keys + static_cast<std::size_t>(nb * 16 + 2 * j + 1);
                            ss += decode_e2m1_word(h_pcode[dht_probe * rows * kS3Keys + sidx]) *
                                 vc[vbase + static_cast<std::size_t>(2 * j) * kdim];
                        }
                        pv1_swap += psf * vsc[vbase] * ss;
                    }
                    pv1_swap += alpha[rr] * acc_kv[static_cast<std::size_t>(rr) * kHeadDim + static_cast<std::size_t>(wd)];
                }
                std::fprintf(stderr,
                             "ARGMAX t1: (r=%d, d=%d) kernel=%.4f model=%.4f rel=%.5f diff=%.6f\n"
                             "        swapped-P model=%.4f (delta vs kernel %.6f)\n",
                             wr, wd, worst_k, worst_m, worst_rel, worst_k - worst_m, pv1_swap,
                             worst_k - pv1_swap);
            }
            // --- per-stage comparison (pipeline order, global (h,t,r,*) scan) --
            const std::size_t dht =
                static_cast<std::size_t>(h) * max_tiles + static_cast<std::size_t>(t);
            if (getenv("S3_DUMP_DEBUG") != nullptr && h == 0 && t == 1) {
                // t=1 decomposition: is the deviation in the alpha term or the pv term?
                const std::size_t a0 = 0;
                const double pv1_kv  = acc_kv_new[a0] - alpha[0] * acc_kv[a0];
                const double pv1_ref = acc_ref_new[a0] - alpha[0] * acc_ref[a0];
                const double k1       = h_acc[dht * rows * kHeadDim + a0];
                const double k1d1    = h_acc[dht * rows * kHeadDim + a0 + 1];
                const double kv1d1   = acc_kv_new[a0 + 1];
                const double m0row   = h_m[static_cast<std::size_t>(h) * max_tiles * rows + 0];
                const double m1row   = h_m[dht * rows + 0];
                std::fprintf(stderr,
                             "DBG t1 r0: alpha=%.6f m0=%.4f m1=%.4f\n"
                             "        acc_kv_prev=%.4f pv1_kv=%.4f acc_kv_new=%.4f\n"
                             "        pv1_ref=%.4f  kernel_acc1=%.4f kernel_d1=%.4f kv_d1=%.4f\n",
                             alpha[0], m0row, m1row, acc_kv[a0], pv1_kv, acc_kv_new[a0], pv1_ref,
                             k1, k1d1, kv1d1);
                // wrong-tile-V-codes hypothesis: kernel pv1 vs recompute using TILE-0 V codes
                {
                    for (std::int32_t d = 0; d < 8; ++d) {
                        double pv1_v0 = 0.0;
                        double pv1_v1 = 0.0;
                        for (std::int32_t nb = 0; nb < kS3NB; ++nb) {
                            const std::size_t pidx = 0 * kS3NB + static_cast<std::size_t>(nb);
                            const double vsv1 = vsc[(static_cast<std::size_t>(kv) * total +
                                                     static_cast<std::size_t>(k0 + nb * 16)) * kdim +
                                                    static_cast<std::size_t>(d)];
                            double ss0 = 0.0, ss1 = 0.0;
                            for (std::int32_t j = 0; j < 16; ++j) {
                                const std::size_t sidx =
                                    static_cast<std::size_t>(nb * 16 + j);
                                const double pdec = decode_e2m1_word(h_pcode[sidx]);
                                ss0 += pdec * vc[(static_cast<std::size_t>(kv) * total +
                                                  static_cast<std::size_t>(nb * 16 + j)) * kdim +
                                                 static_cast<std::size_t>(d)];
                                ss1 += pdec * vc[(static_cast<std::size_t>(kv) * total +
                                                  static_cast<std::size_t>(k0 + nb * 16 + j)) * kdim +
                                                 static_cast<std::size_t>(d)];
                            }
                            pv1_v0 += psf_kv_dec[pidx] * vsv1 * ss0;
                            pv1_v1 += psf_kv_dec[pidx] * vsv1 * ss1;
                        }
                        const double kv1 =
                            h_acc[dht * rows * kHeadDim + static_cast<std::size_t>(d)] -
                            h_acc[static_cast<std::size_t>(h) * max_tiles * rows * kHeadDim +
                                  static_cast<std::size_t>(d)];
                        std::fprintf(stderr, "DBG t1 r0 d=%d k_pv1=%.2f v0pv=%.2f v1pv=%.2f\n",
                                    d, kv1, pv1_v0, pv1_v1);
                    }
                }
            }
            if (getenv("S3_DUMP_DEBUG") != nullptr && h == 0 && t == 0) {
                std::fprintf(stderr, "DBG h0 t0 row0 sref[0..7] = ");
                for (int i = 0; i < 8; ++i) { std::fprintf(stderr, "%.4f ", sref[i]); }
                std::fprintf(stderr, "\nDBG h0 t0 row0 h_score[0..7] = ");
                for (int i = 0; i < 8; ++i) { std::fprintf(stderr, "%.4f ", h_score[i]); }
                std::fprintf(stderr, "\nDBG nm[0]=%.4f h_m[0]=%f run_l[0]=%.2f h_l[0]=%.2f\n",
                             nm[0], h_m[0], run_l[0], h_l[0]);
                std::fprintf(stderr, "DBG psf_ref[0..3] = %u %u %u %u | h_psf[0..3] = %u %u %u %u\n",
                             psf_ref_byte[0], psf_ref_byte[1], psf_ref_byte[2], psf_ref_byte[3],
                             h_psf[0], h_psf[1], h_psf[2], h_psf[3]);
            }
            for (std::int32_t r = 0; r < rows; ++r) {
                const std::size_t ridx = static_cast<std::size_t>(r);
                // score (r, i): FP32 tensor-core dot vs FP64 dequant dot
                for (std::int32_t i = 0; i < kS3Keys; ++i) {
                    const std::size_t sidx = ridx * kS3Keys + static_cast<std::size_t>(i);
                    const double kvv = h_score[dht * rows * kS3Keys + sidx];
                    const double rvv = sref[sidx];
                    if (rvv < -1e299) {
                        if (kvv > -1e20) { s3_consider(first_score, h, t, r, i, kvv, rvv, 1.0); }
                        continue;
                    }
                    const double rel =
                        std::abs(kvv - rvv) / std::max({std::abs(kvv), std::abs(rvv), 1e-30});
                    max_rel_score = std::max(max_rel_score, rel);
                    if (rel > 1e-4 && std::abs(kvv - rvv) > 1e-6) {
                        s3_consider(first_score, h, t, r, i, kvv, rvv, rel);
                    }
                }
                // psf (r, nb): 1-step e4m3 flip = P-quant floor, 2+ = bug
                for (std::int32_t nb = 0; nb < kS3NB; ++nb) {
                    const std::size_t pidx = ridx * kS3NB + static_cast<std::size_t>(nb);
                    const std::uint8_t kvb = h_psf[dht * rows * kS3NB + pidx];
                    const std::uint8_t rbb = psf_ref_byte[pidx];
                    const int d8 = std::abs(static_cast<int>(kvb) - static_cast<int>(rbb));
                    if (d8 == 1) { ++flips_psf; }
                    else if (d8 >= 2) {
                        ++two_plus_psf;
                        max_diff_psf = std::max(max_diff_psf, d8);
                        s3_consider(first_psf, h, t, r, nb, static_cast<double>(kvb),
                                    static_cast<double>(rbb), static_cast<double>(d8));
                    }
                }
                // p_code (r, j): 1-step e2m1 flip = P-quant floor, 2+ = bug
                for (std::int32_t j = 0; j < kS3Keys; ++j) {
                    const std::size_t sidx = ridx * kS3Keys + static_cast<std::size_t>(j);
                    const std::uint8_t kvb = h_pcode[dht * rows * kS3Keys + sidx];
                    const std::uint8_t rbb = p_ref_nib[sidx];
                    const int d8 = std::abs(static_cast<int>(kvb) - static_cast<int>(rbb));
                    if (d8 == 1) { ++flips_pcode; }
                    else if (d8 >= 2) {
                        ++two_plus_pcode;
                        max_diff_pcode = std::max(max_diff_pcode, d8);
                        s3_consider(first_pcode, h, t, r, j, static_cast<double>(kvb),
                                    static_cast<double>(rbb), static_cast<double>(d8));
                    }
                }
                // m / l (r): running max / running L after the tile
                const double kvm = h_m[dht * rows + ridx];
                const double kvl = h_l[dht * rows + ridx];
                const double relm = (nm[r] < -1e299)
                                        ? (kvm < -1e20 ? 0.0 : 1.0)
                                        : std::abs(kvm - nm[r]) /
                                              std::max({std::abs(kvm), std::abs(nm[r]), 1e-30});
                const double rell = std::abs(kvl - run_l[r]) /
                                    std::max({std::abs(kvl), std::abs(run_l[r]), 1e-30});
                max_rel_m = std::max(max_rel_m, relm);
                max_rel_l = std::max(max_rel_l, rell);
                if (relm > 1e-4 && std::abs(kvm - nm[r]) > 1e-6) {
                    s3_consider(first_m, h, t, r, 0, kvm, nm[r], relm);
                }
                if (rell > 1e-3 && std::abs(kvl - run_l[r]) > 1e-6) {
                    s3_consider(first_l, h, t, r, 0, kvl, run_l[r], rell);
                }
                // acc (r, d): PV-path signal = FP64 recompute with the KERNEL'S codes
                for (std::int32_t d = 0; d < kHeadDim; ++d) {
                    const std::size_t aidx = ridx * kHeadDim + static_cast<std::size_t>(d);
                    const double kval = h_acc[dht * rows * kHeadDim + aidx];
                    const double rel_kv =
                        std::abs(kval - acc_kv_new[aidx]) /
                        std::max({std::abs(kval), std::abs(acc_kv_new[aidx]), 1e-30});
                    const double rel_ref =
                        std::abs(kval - acc_ref_new[aidx]) /
                        std::max({std::abs(kval), std::abs(acc_ref_new[aidx]), 1e-30});
                    max_rel_acc_kv = std::max(max_rel_acc_kv, rel_kv);
                    max_rel_acc_ref = std::max(max_rel_acc_ref, rel_ref);
                    // The acc stage accumulates 64+ FP32 products, so its noise
                    // floor is ~2e-3 rel (measured across 1.5M positions on a
                    // verified-correct kernel); 1e-2 stays 5x below any real PV
                    // defect (a dropped/mis-scaled block is O(0.1+)).
                    if (rel_kv > 1e-2 && std::abs(kval - acc_kv_new[aidx]) > 1e-2) {
                        s3_consider(first_acc, h, t, r, d, kval, acc_kv_new[aidx], rel_kv);
                    }
                }
            }
            // v_scale (d, nb): byte copy of the cache plane — any diff = indexing bug
            for (std::int32_t d = 0; d < kHeadDim; ++d) {
                for (std::int32_t nb = 0; nb < kS3NB; ++nb) {
                    const std::size_t vidx = dht * kHeadDim * kS3NB +
                                             static_cast<std::size_t>(d) * kS3NB +
                                             static_cast<std::size_t>(nb);
                    const std::uint8_t kvb = h_vsc[vidx];
                    const std::int32_t key = k0 + nb * 16;
                    const std::uint8_t rbb = expected.v_fp8[sage_v_scale_host_index(
                        expected.logical_capacity, kv, key / kPagedKVPageSize, d,
                        (key % kPagedKVPageSize) / 16)];
                    if (kvb != rbb) {
                        ++diffs_vsc;
                        s3_consider(first_vsc, h, t, d, nb, static_cast<double>(kvb),
                                    static_cast<double>(rbb), 1.0);
                    }
                }
            }
            // v_t (d, kp): the kernel's transposed V-code B operand for this tile.
            // Byte kp holds the e2m1 codes for keys (2kp, 2kp+1) at d (high nibble
            // = the odd key), so any diff vs the raw cache plane = transpose/index bug.
            {
                std::vector<std::uint8_t> vcodes(static_cast<std::size_t>(kS3Keys) * kdim, 0);
                for (std::int32_t i = 0; i < kS3Keys; ++i) {
                    const std::int32_t key = k0 + i;
                    for (std::int32_t d = 0; d < kHeadDim; ++d) {
                        const std::uint8_t byte = expected.v_u8[nvfp4_code_index(
                            geometry, expected.logical_capacity, kv, key, d / 2)];
                        vcodes[(static_cast<std::size_t>(i) * kdim) + static_cast<std::size_t>(d)] =
                            static_cast<std::uint8_t>((byte >> ((d & 1) * 4)) & 0x0Fu);
                    }
                }
                for (std::int32_t d = 0; d < kHeadDim; ++d) {
                    for (std::int32_t kp = 0; kp < kS3Keys / 2; ++kp) {
                        const std::size_t vidx = dht * kdim * 32 +
                                                 static_cast<std::size_t>(d) * 32 +
                                                 static_cast<std::size_t>(kp);
                        const std::uint8_t kvb = h_vt[vidx];
                        const std::uint8_t rbb = static_cast<std::uint8_t>(
                            (static_cast<std::uint16_t>(vcodes[static_cast<std::size_t>(2 * kp + 1) *
                                                              kdim + static_cast<std::size_t>(d)])
                             << 4) |
                            vcodes[static_cast<std::size_t>(2 * kp) * kdim +
                                   static_cast<std::size_t>(d)]);
                        if (kvb != rbb) {
                            ++diffs_vt;
                            s3_consider(first_vt, h, t, d, kp, static_cast<double>(kvb),
                                        static_cast<double>(rbb), 1.0);
                        }
                    }
                }
            }
            if (h == 0 && (t == 0 || t == 1)) {
                // PV block-permutation probe: the mma applies one e4m3 P-scale and one
                // e4m3 V-scale per 16-key block. Try all 24x24 block orderings and find
                // which one reproduces the kernel's accumulator (identity = natural
                // order). A match at ~1e-6 pins the kernel's actual block mapping.
                const std::size_t k0idx = static_cast<std::size_t>(kv) * total +
                                           static_cast<std::size_t>(k0);
                std::vector<double> Bp(static_cast<std::size_t>(kS3NB) * kdim, 0.0);
                for (std::int32_t nb = 0; nb < kS3NB; ++nb) {
                    for (std::int32_t d = 0; d < kHeadDim; ++d) {
                        double s = 0.0;
                        for (std::int32_t j = 0; j < 16; ++j) {
                            const std::size_t sidx = static_cast<std::size_t>(nb * 16 + j);
                            s += decode_e2m1_word(h_pcode[sidx]) *
                                 vc[(k0idx + static_cast<std::size_t>(nb * 16 + j)) * kdim +
                                    static_cast<std::size_t>(d)];
                        }
                        Bp[static_cast<std::size_t>(nb) * kdim + static_cast<std::size_t>(d)] = s;
                    }
                }
                std::vector<std::vector<int>> perms;
                {
                    std::vector<int> p{0, 1, 2, 3};
                    do { perms.push_back(p); }
                    while (std::next_permutation(p.begin(), p.end()));
                }
                struct Probe { double rel; int pi; int vi; };
                std::vector<Probe> probes;
                double identity_rel = 1e30;
                for (int a = 0; a < static_cast<int>(perms.size()); ++a) {
                    for (int b = 0; b < static_cast<int>(perms.size()); ++b) {
                        const std::vector<int>& pp = perms[static_cast<std::size_t>(a)];
                        const std::vector<int>& vp = perms[static_cast<std::size_t>(b)];
                        double mrel = 0.0;
                        for (std::int32_t r = 0; r < rows; ++r) {
                            const std::size_t ridx = static_cast<std::size_t>(r);
                            for (std::int32_t d = 0; d < kHeadDim; ++d) {
                                double accv = alpha[ridx] * acc_kv[ridx * kdim + static_cast<std::size_t>(d)];
                                for (std::int32_t nb = 0; nb < kS3NB; ++nb) {
                                    accv += psf_kv_dec[ridx * kS3NB + static_cast<std::size_t>(pp[nb])] *
                                           vsc[(k0idx + static_cast<std::size_t>(vp[nb] * 16)) * kdim +
                                                static_cast<std::size_t>(d)] *
                                           Bp[static_cast<std::size_t>(nb) * kdim +
                                               static_cast<std::size_t>(d)];
                                }
                                const double kval = h_acc[ridx * kHeadDim + static_cast<std::size_t>(d)];
                                mrel = std::max(mrel, std::abs(kval - accv) / std::max(
                                                            {std::abs(kval), std::abs(accv), 1e-30}));
                            }
                        }
                        if (a == 0 && b == 0) { identity_rel = mrel; }
                        probes.push_back({mrel, a, b});
                    }
                }
                std::sort(probes.begin(), probes.end(),
                          [](const Probe& x, const Probe& y) { return x.rel < y.rel; });
                // subset probe: does the kernel's tile pv equal a subset of the 4 blocks?
                {
                    const std::size_t r0 = 0;
                    std::ostringstream so;
                    so << "\nS3_SUBSET_T" << t << " r0: [";
                    for (int d = 0; d < 8; ++d) {
                        const double kval =
                            h_acc[dht * rows * kHeadDim + static_cast<std::size_t>(d)] -
                            h_acc[static_cast<std::size_t>(h) * max_tiles * rows * kHeadDim +
                                  static_cast<std::size_t>(d)];
                        double best = 1e30;
                        int bestmask = -1;
                        for (int mask = 1; mask < 16; ++mask) {
                            double s = 0.0;
                            for (int nb = 0; nb < 4; ++nb) {
                                if (mask & (1 << nb)) {
                                    s += psf_kv_dec[r0 * kS3NB + static_cast<std::size_t>(nb)] *
                                         vsc[(k0idx + static_cast<std::size_t>(nb * 16)) * kdim +
                                            static_cast<std::size_t>(d)] *
                                         Bp[static_cast<std::size_t>(nb) * kdim +
                                            static_cast<std::size_t>(d)];
                                }
                            }
                            const double rel =
                                std::abs(kval - s) / std::max({std::abs(kval), std::abs(s), 1e-30});
                            if (rel < best) { best = rel; bestmask = mask; }
                        }
                        std::string bs = "";
                        for (int nb = 3; nb >= 0; --nb) { bs += (bestmask & (1 << nb)) ? '1' : '0'; }
                        so << (d ? " " : "") << "d" << d << ":k=" << std::setprecision(4) << kval
                           << ";best=0b" << bs << ":" << std::setprecision(4) << best;
                    }
                    so << "]";
                    std::fprintf(stderr, "%s\n", so.str().c_str());
                }
                // Per-d / per-r deviation profile (identity recompute) + worst elements,
                // so the deviation's structure (d-range, row pattern) is visible.
                std::vector<double> dev_d(kdim, 0.0), dev_r(static_cast<std::size_t>(rows), 0.0);
                std::vector<Probe> worst;
                for (std::int32_t r = 0; r < rows; ++r) {
                    const std::size_t ridx = static_cast<std::size_t>(r);
                    for (std::int32_t d = 0; d < kHeadDim; ++d) {
                        double accv = alpha[ridx] * acc_kv[ridx * kdim + static_cast<std::size_t>(d)];
                        for (std::int32_t nb = 0; nb < kS3NB; ++nb) {
                            accv += psf_kv_dec[ridx * kS3NB + static_cast<std::size_t>(nb)] *
                                   vsc[(k0idx + static_cast<std::size_t>(nb * 16)) * kdim +
                                       static_cast<std::size_t>(d)] *
                                   Bp[static_cast<std::size_t>(nb) * kdim +
                                       static_cast<std::size_t>(d)];
                        }
                        const double kval = h_acc[ridx * kHeadDim + static_cast<std::size_t>(d)];
                        const double rel = std::abs(kval - accv) / std::max(
                                                 {std::abs(kval), std::abs(accv), 1e-30});
                        dev_d[static_cast<std::size_t>(d)] =
                            std::max(dev_d[static_cast<std::size_t>(d)], rel);
                        dev_r[ridx] = std::max(dev_r[ridx], rel);
                        worst.push_back({rel, static_cast<int>(r), d});
                    }
                }
                std::sort(worst.rbegin(), worst.rend(),
                          [](const Probe& x, const Probe& y) { return x.rel < y.rel; });
                std::ostringstream po;
                po << "{\"tile\": " << t << ", \"identity_max_rel\": " << identity_rel << ", \"top\": [";
                for (int k = 0; k < 3 && k < static_cast<int>(probes.size()); ++k) {
                    const Probe& pb = probes[static_cast<std::size_t>(k)];
                    const std::vector<int>& pp = perms[static_cast<std::size_t>(pb.pi)];
                    const std::vector<int>& vp = perms[static_cast<std::size_t>(pb.vi)];
                    po << (k ? ", " : "")
                       << "{\"p\": [" << pp[0] << "," << pp[1] << "," << pp[2] << "," << pp[3]
                       << "], \"v\": [" << vp[0] << "," << vp[1] << "," << vp[2] << "," << vp[3]
                       << "], \"max_rel\": " << pb.rel << "}";
                }
                po << "], \"dev_d\": [";
                for (std::int32_t d = 0; d < kHeadDim; ++d) {
                    po << (d ? ", " : "") << dev_d[static_cast<std::size_t>(d)];
                }
                po << "], \"dev_r\": [";
                for (std::int32_t r = 0; r < rows; ++r) {
                    po << (r ? ", " : "") << dev_r[static_cast<std::size_t>(r)];
                }
                po << "], \"worst\": [";
                for (int k = 0; k < 5 && k < static_cast<int>(worst.size()); ++k) {
                    const Probe& w = worst[static_cast<std::size_t>(k)];
                    po << (k ? ", " : "") << "{\"r\": " << w.pi << ", \"d\": " << w.vi
                       << ", \"rel\": " << w.rel << "}";
                }
                po << "]}";
                if (acc_probe_json == "null") { acc_probe_json = po.str(); }
                else { acc_probe_json += ", " + po.str(); }
            }
            if (getenv("S3_ARGMAX") != nullptr) {
                const std::size_t dt = static_cast<std::size_t>(h) * max_tiles + static_cast<std::size_t>(t);
                for (std::int32_t rr = 0; rr < rows; ++rr) {
                    for (std::int32_t dd = 0; dd < kHeadDim; ++dd) {
                        const std::size_t ax = static_cast<std::size_t>(rr) * kHeadDim + static_cast<std::size_t>(dd);
                        const double kvv = h_acc[dt * rows * kHeadDim + ax];
                        const double mvv = acc_kv_new[ax];
                        const double rel = std::abs(kvv - mvv) /
                                           std::max({std::abs(kvv), std::abs(mvv), 1e-30});
                        if (rel > 1e-3) { ++n_gt_1e3; }
                        if (rel > 1e-2) { ++n_gt_1e2; }
                        if (rel > g_rel) {
                            g_rel = rel;
                            g_t = t;
                            g_r = rr;
                            g_d = dd;
                            g_k = kvv;
                            g_m = mvv;
                        }
                    }
                }
            }
            std::swap(acc_ref, acc_ref_new);
            std::swap(acc_kv, acc_kv_new);
        }
        if (getenv("S3_ARGMAX") != nullptr) {
            std::fprintf(stderr,
                         "ARGMAX h%d: t=%d (r=%d, d=%d) kernel=%.6f model=%.6f rel=%.5f | n>1e-3=%zu n>1e-2=%zu\n",
                         h, g_t, g_r, g_d, g_k, g_m, g_rel, n_gt_1e3, n_gt_1e2);
        }
    }

    // --- verdict: earliest non-clean stage in pipeline order = root-cause ----
    const bool score_clean = !first_score.found;
    const bool psf_clean   = two_plus_psf == 0;
    const bool pcode_clean = two_plus_pcode == 0;
    const bool vsc_clean   = diffs_vsc == 0;
    const bool vt_clean    = diffs_vt == 0;
    const bool m_clean     = !first_m.found;
    const bool l_clean     = !first_l.found;
    const bool acc_clean   = !first_acc.found;
    const char* verdict = "floor-only (P-quant 1-code flips only; PV path clean)";
    if (!score_clean) { verdict = "bug:score"; }
    else if (!psf_clean) { verdict = "bug:psf"; }
    else if (!pcode_clean) { verdict = "bug:p_code"; }
    else if (!vsc_clean) { verdict = "bug:v_scale"; }
    else if (!vt_clean) { verdict = "bug:v_t"; }
    else if (!m_clean) { verdict = "bug:m"; }
    else if (!l_clean) { verdict = "bug:l"; }
    else if (!acc_clean) { verdict = "bug:acc"; }

    const std::string label = case_label("gqa_attention_s3_dump", geometry, DType::U8, test_case,
                                         MappingPattern::Identity);
    std::ostringstream js;
    js << "{\n  \"op\": \"gqa_attention\", \"kind\": \"s3_prefill\",\n"
       << "  \"case\": \"" << label << "\",\n"
       << "  \"heads\": " << heads << ", \"rows\": " << rows << ", \"max_tiles\": " << max_tiles
       << ", \"tiles\": " << h_tcnt[0] << ", \"keep_frac\": 1.0,\n"
       << "  \"guards_clean\": " << (guard_failures == 0 ? "true" : "false") << ",\n"
       << "  \"stages\": [\n"
       << "    {\"name\": \"score\", \"clean\": " << (score_clean ? "true" : "false")
       << ", \"max_rel\": " << max_rel_score
       << ", \"first\": " << s3_first_json(first_score, "i") << "},\n"
       << "    {\"name\": \"psf\", \"clean\": " << (psf_clean ? "true" : "false")
       << ", \"one_step_flips\": " << flips_psf << ", \"two_plus\": " << two_plus_psf
       << ", \"max_diff\": " << max_diff_psf
       << ", \"first\": " << s3_first_json(first_psf, "nb") << "},\n"
       << "    {\"name\": \"p_code\", \"clean\": " << (pcode_clean ? "true" : "false")
       << ", \"one_step_flips\": " << flips_pcode << ", \"two_plus\": " << two_plus_pcode
       << ", \"max_diff\": " << max_diff_pcode
       << ", \"first\": " << s3_first_json(first_pcode, "j") << "},\n"
       << "    {\"name\": \"v_scale\", \"clean\": " << (vsc_clean ? "true" : "false")
       << ", \"diffs\": " << diffs_vsc
        << ", \"first\": " << s3_first_json(first_vsc, "d") << "},\n"
        << "    {\"name\": \"v_t\", \"clean\": " << (vt_clean ? "true" : "false")
        << ", \"diffs\": " << diffs_vt
        << ", \"first\": " << s3_first_json(first_vt, "d") << "},\n"
        << "    {\"name\": \"m\", \"clean\": " << (m_clean ? "true" : "false")
       << ", \"max_rel\": " << max_rel_m
       << ", \"first\": " << s3_first_json(first_m, "c") << "},\n"
       << "    {\"name\": \"l\", \"clean\": " << (l_clean ? "true" : "false")
       << ", \"max_rel\": " << max_rel_l
       << ", \"first\": " << s3_first_json(first_l, "c") << "},\n"
       << "    {\"name\": \"acc\", \"clean\": " << (acc_clean ? "true" : "false")
       << ", \"max_rel_kernel_codes\": " << max_rel_acc_kv
       << ", \"max_rel_ref_codes\": " << max_rel_acc_ref
        << ", \"first\": " << s3_first_json(first_acc, "d") << "}\n"
        << "  ],\n"
        << "  \"acc_probes\": [" << acc_probe_json << "],\n"
        << "  \"verdict\": \"" << verdict << "\"\n}\n";
    std::ofstream out(json_path);
    if (!out) {
        std::cerr << "S3_DUMP: cannot open " << json_path << " for write\n";
        return 1;
    }
    out << js.str() << std::flush;
    std::cout << "S3_DUMP " << label << "\n  verdict=" << verdict << "\n";
    if (!score_clean) { std::cout << "  score diverges: " << s3_first_json(first_score, "i") << "\n"; }
    if (!psf_clean) { std::cout << "  psf bug-diff: " << s3_first_json(first_psf, "nb") << "\n"; }
    if (!pcode_clean) { std::cout << "  p_code bug-diff: " << s3_first_json(first_pcode, "j") << "\n"; }
    if (!vsc_clean) { std::cout << "  v_scale diff: " << s3_first_json(first_vsc, "d") << "\n"; }
    if (!vt_clean) { std::cout << "  v_t diff: " << s3_first_json(first_vt, "d") << "\n"; }
    if (!m_clean) { std::cout << "  m diverges: " << s3_first_json(first_m, "c") << "\n"; }
    if (!l_clean) { std::cout << "  l diverges: " << s3_first_json(first_l, "c") << "\n"; }
    if (!acc_clean) { std::cout << "  acc (PV path) diverges: " << s3_first_json(first_acc, "d") << "\n"; }
    if (guard_failures != 0) { std::cout << "  WARNING: " << guard_failures << " guard canaries failed\n"; }
    return 0;
}

int main(int argc, char** argv) {
    if (argc > 2 && std::strcmp(argv[1], "--s3-dump") == 0) {
        // tools/kdev: run the representative multi-tile sage A1 case (T=128, base=64
        // -> 192 keys, 3 key tiles) through the s3 op-dump side-band.
        if (cuda_unavailable()) {
            std::cout << "SKIP: no usable CUDA device\n";
            return 77;
        }
        const AttentionCase s3_case{128, 64, 256, 511u};
        const int rc = s3_dump_case(kGeometries[0], s3_case, argv[2]);
        std::cout << (rc == 0 ? "PASS" : "FAIL") << " gqa_attention s3-dump\n";
        return rc;
    }
    if (cuda_unavailable()) {
        std::cout << "SKIP: no usable CUDA device\n";
        return 77;
    }

    int failures = 0;
    failures += verify_workspace_capacity_contract();
    for (const Geometry& geometry : kGeometries) { failures += run_geometry(geometry); }
    failures += run_batch_cases();
    std::cout << (failures == 0 ? "PASS" : "FAIL")
              << " gqa_attention public-contract correctness\n";
    return failures == 0 ? 0 : 1;
}
