#include "core/arena.h"
#include "core/decode_graph.h"
#include "core/device.h"
#include "ops/r9700/kv/fp8_int4_kv_attention.h"
#include "ops/r9700/kv/r9700_attention_profile.h"
#include "targets/qwen3/impl/runtime/r9700_cache_profile.h"
#include "targets/qwen3_8_27b/impl/r9700_full_attention.h"
#include "../../tests/ops/fp8_int4_kv_oracle.h"

#include <hip/hip_bfloat16.h>

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <limits>
#include <numeric>
#include <span>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

namespace q27    = ninfer::targets::qwen3_8_27b::detail;
namespace q3     = ninfer::targets::qwen3;
namespace oracle = ninfer::test::fp8_int4_kv_oracle;

constexpr std::size_t kHeadDim      = 256;
constexpr std::size_t kQueryHeads   = 24;
constexpr std::size_t kKvHeads      = 4;
constexpr std::size_t kQueriesPerKv = 6;
constexpr std::size_t kValueGroup =
    static_cast<std::size_t>(ninfer::targets::qwen3::detail::kR9700TextKVValueGroup);
constexpr std::size_t kGroups               = kHeadDim / kValueGroup;
constexpr std::size_t kPageSize             = 64;
constexpr std::size_t kContext              = 257;
constexpr std::size_t kMappedPages          = 5;
constexpr std::size_t kPhysicalPages        = 6;
constexpr std::size_t kPhysicalTokens       = kPhysicalPages * kPageSize;
constexpr std::size_t kMaximumQualifiedRows = 128;
constexpr float kAttentionScale             = 1.0F / 16.0F;

[[noreturn]] void fail(const std::string& message) { throw std::runtime_error(message); }

float represented_bf16(float value) {
    std::uint32_t bits = std::bit_cast<std::uint32_t>(value);
    bits += 0x7fffU + ((bits >> 16U) & 1U);
    return std::bit_cast<float>(bits & 0xffff0000U);
}

std::size_t physical_token(std::size_t logical, const std::vector<std::uint32_t>& page_table) {
    if (logical >= page_table.size() * kPageSize) { fail("oracle logical token is out of range"); }
    return static_cast<std::size_t>(page_table[logical / kPageSize]) * kPageSize +
           logical % kPageSize;
}

// These offsets are the independent physical-format oracle for the leaf's selected candidate:
// token-fastest K and feature-fastest V/V-scale. They intentionally do not call a production
// planner or device indexing helper.
std::size_t key_offset(std::size_t token, std::size_t head, std::size_t feature,
                       std::size_t physical_pages = kPhysicalPages) {
    return token % kPageSize +
           kPageSize * (feature + kHeadDim * (token / kPageSize + physical_pages * head));
}

std::size_t value_offset(std::size_t token, std::size_t head, std::size_t byte) {
    return byte + (kHeadDim / 2U) *
                      (token % kPageSize + kPageSize * (head + kKvHeads * (token / kPageSize)));
}

std::size_t scale_offset(std::size_t token, std::size_t head, std::size_t group) {
    return group +
           kGroups * (token % kPageSize + kPageSize * (head + kKvHeads * (token / kPageSize)));
}

std::vector<float> make_keys(std::size_t tokens = kContext) {
    std::vector<float> values(tokens * kKvHeads * kHeadDim);
    for (std::size_t token = 0; token < tokens; ++token) {
        for (std::size_t head = 0; head < kKvHeads; ++head) {
            for (std::size_t feature = 0; feature < kHeadDim; ++feature) {
                const int wave =
                    static_cast<int>((token * 13U + head * 29U + feature * 7U) % 97U) - 48;
                values[(token * kKvHeads + head) * kHeadDim + feature] =
                    represented_bf16(static_cast<float>(wave) / 32.0F);
            }
        }
    }
    return values;
}

std::vector<float> make_values(std::size_t tokens = kContext) {
    std::vector<float> values(tokens * kKvHeads * kHeadDim);
    for (std::size_t token = 0; token < tokens; ++token) {
        for (std::size_t head = 0; head < kKvHeads; ++head) {
            for (std::size_t feature = 0; feature < kHeadDim; ++feature) {
                const int wave =
                    static_cast<int>((token * 17U + head * 23U + feature * 11U) % 113U) - 56;
                values[(token * kKvHeads + head) * kHeadDim + feature] =
                    represented_bf16(static_cast<float>(wave) / 40.0F);
            }
        }
    }
    return values;
}

std::vector<float> make_queries(std::size_t rows) {
    std::vector<float> query(rows * kQueryHeads * kHeadDim);
    for (std::size_t row = 0; row < rows; ++row) {
        for (std::size_t head = 0; head < kQueryHeads; ++head) {
            for (std::size_t feature = 0; feature < kHeadDim; ++feature) {
                const int wave =
                    static_cast<int>((row * 11U + head * 31U + feature * 19U) % 89U) - 44;
                query[(row * kQueryHeads + head) * kHeadDim + feature] =
                    represented_bf16(static_cast<float>(wave) / 64.0F);
            }
        }
    }
    return query;
}

std::vector<hip_bfloat16> as_bf16(const std::vector<float>& represented) {
    std::vector<hip_bfloat16> result;
    result.reserve(represented.size());
    for (float value : represented) { result.emplace_back(value); }
    return result;
}

struct EncodedCache {
    oracle::Fp8Keys logical_keys;
    oracle::Int4Values logical_values;
    std::vector<std::uint8_t> physical_keys;
    std::vector<std::uint8_t> physical_values;
    std::vector<std::uint16_t> physical_scales;
};

EncodedCache encode_cache(const std::vector<float>& keys, const std::vector<float>& values,
                          const std::vector<std::uint32_t>& page_table,
                          std::size_t physical_pages = kPhysicalPages) {
    const std::size_t tokens = keys.size() / (kKvHeads * kHeadDim);
    if (tokens == 0 || keys.size() != tokens * kKvHeads * kHeadDim ||
        values.size() != keys.size()) {
        fail("encoded cache source geometry is invalid");
    }
    // Slots holding no represented token keep NaN keys and scales (as stale device memory may):
    // a route must not read them into a result.
    EncodedCache cache{
        oracle::encode_keys(keys, tokens * kKvHeads, kHeadDim),
        oracle::encode_values(values, tokens * kKvHeads, kHeadDim, kValueGroup),
        std::vector<std::uint8_t>(physical_pages * kPageSize * kKvHeads * kHeadDim, 0xFFU),
        std::vector<std::uint8_t>(physical_pages * kPageSize * kKvHeads * kHeadDim / 2U, 0U),
        std::vector<std::uint16_t>(physical_pages * kPageSize * kKvHeads * kGroups, 0xFFFFU)};
    for (std::size_t logical = 0; logical < tokens; ++logical) {
        const std::size_t physical = physical_token(logical, page_table);
        for (std::size_t head = 0; head < kKvHeads; ++head) {
            const std::size_t represented_token = logical * kKvHeads + head;
            for (std::size_t feature = 0; feature < kHeadDim; ++feature) {
                cache.physical_keys[key_offset(physical, head, feature, physical_pages)] =
                    cache.logical_keys.codes[represented_token * kHeadDim + feature];
            }
            for (std::size_t group = 0; group < kGroups; ++group) {
                cache.physical_scales[scale_offset(physical, head, group)] =
                    cache.logical_values.fp16_scales[represented_token * kGroups + group];
                for (std::size_t byte = 0; byte < kValueGroup / 2U; ++byte) {
                    cache.physical_values[value_offset(physical, head,
                                                       group * (kValueGroup / 2U) + byte)] =
                        cache.logical_values.packed_codes[represented_token * (kHeadDim / 2U) +
                                                          group * (kValueGroup / 2U) + byte];
                }
            }
        }
    }
    return cache;
}

std::vector<double> attention_oracle(const std::vector<float>& queries, std::size_t rows,
                                     const std::vector<std::size_t>& row_contexts,
                                     const EncodedCache& cache) {
    std::vector<double> output(rows * kQueryHeads * kHeadDim);
    for (std::size_t row = 0; row < rows; ++row) {
        const std::size_t visible = row_contexts[row];
        for (std::size_t query_head = 0; query_head < kQueryHeads; ++query_head) {
            const std::size_t kv_head = query_head / kQueriesPerKv;
            oracle::Fp8Keys keys{visible, kHeadDim, std::vector<std::uint8_t>(visible * kHeadDim)};
            oracle::Int4Values values{visible, kHeadDim, kValueGroup,
                                      std::vector<std::uint8_t>(visible * kHeadDim / 2U),
                                      std::vector<std::uint16_t>(visible * kGroups)};
            for (std::size_t token = 0; token < visible; ++token) {
                const std::size_t represented_token = token * kKvHeads + kv_head;
                std::copy_n(cache.logical_keys.codes.begin() +
                                static_cast<std::ptrdiff_t>(represented_token * kHeadDim),
                            kHeadDim,
                            keys.codes.begin() + static_cast<std::ptrdiff_t>(token * kHeadDim));
                std::copy_n(cache.logical_values.packed_codes.begin() +
                                static_cast<std::ptrdiff_t>(represented_token * kHeadDim / 2U),
                            kHeadDim / 2U,
                            values.packed_codes.begin() +
                                static_cast<std::ptrdiff_t>(token * kHeadDim / 2U));
                std::copy_n(cache.logical_values.fp16_scales.begin() +
                                static_cast<std::ptrdiff_t>(represented_token * kGroups),
                            kGroups,
                            values.fp16_scales.begin() +
                                static_cast<std::ptrdiff_t>(token * kGroups));
            }
            const std::size_t query_base   = (row * kQueryHeads + query_head) * kHeadDim;
            const std::vector<double> head = oracle::attention_fp64(
                std::span<const float>(queries.data() + query_base, kHeadDim), keys, values,
                static_cast<double>(kAttentionScale));
            std::copy(head.begin(), head.end(),
                      output.begin() + static_cast<std::ptrdiff_t>(query_base));
        }
    }
    return output;
}

std::vector<double> tree_attention_oracle(const std::vector<float>& queries,
                                          const std::vector<std::int32_t>& row_positions,
                                          const std::vector<std::int32_t>& prefix_lengths,
                                          const std::vector<std::int32_t>& ancestor_masks,
                                          const EncodedCache& cache) {
    const std::size_t rows = row_positions.size();
    if (queries.size() != rows * kQueryHeads * kHeadDim || prefix_lengths.size() != rows ||
        ancestor_masks.size() != rows) {
        fail("tree oracle inputs have inconsistent geometry");
    }
    std::vector<double> output(queries.size());
    for (std::size_t row = 0; row < rows; ++row) {
        const std::size_t row_context = static_cast<std::size_t>(row_positions[row]) + 1U;
        const std::size_t prefix      = static_cast<std::size_t>(prefix_lengths[row]);
        const std::uint32_t mask      = static_cast<std::uint32_t>(ancestor_masks[row]);
        std::vector<std::size_t> selected;
        selected.reserve(row_context);
        for (std::size_t logical = 0; logical < row_context; ++logical) {
            if (logical < prefix ||
                (mask & (1U << static_cast<std::uint32_t>(logical - prefix))) != 0U) {
                selected.push_back(logical);
            }
        }
        if (selected.empty()) { fail("tree oracle received an empty visible set"); }
        for (std::size_t query_head = 0; query_head < kQueryHeads; ++query_head) {
            const std::size_t kv_head = query_head / kQueriesPerKv;
            oracle::Fp8Keys keys{selected.size(), kHeadDim,
                                 std::vector<std::uint8_t>(selected.size() * kHeadDim)};
            oracle::Int4Values values{selected.size(), kHeadDim, kValueGroup,
                                      std::vector<std::uint8_t>(selected.size() * kHeadDim / 2U),
                                      std::vector<std::uint16_t>(selected.size() * kGroups)};
            for (std::size_t compact = 0; compact < selected.size(); ++compact) {
                const std::size_t represented = selected[compact] * kKvHeads + kv_head;
                std::copy_n(cache.logical_keys.codes.begin() +
                                static_cast<std::ptrdiff_t>(represented * kHeadDim),
                            kHeadDim,
                            keys.codes.begin() + static_cast<std::ptrdiff_t>(compact * kHeadDim));
                std::copy_n(cache.logical_values.packed_codes.begin() +
                                static_cast<std::ptrdiff_t>(represented * kHeadDim / 2U),
                            kHeadDim / 2U,
                            values.packed_codes.begin() +
                                static_cast<std::ptrdiff_t>(compact * kHeadDim / 2U));
                std::copy_n(cache.logical_values.fp16_scales.begin() +
                                static_cast<std::ptrdiff_t>(represented * kGroups),
                            kGroups,
                            values.fp16_scales.begin() +
                                static_cast<std::ptrdiff_t>(compact * kGroups));
            }
            const std::size_t query_base   = (row * kQueryHeads + query_head) * kHeadDim;
            const std::vector<double> head = oracle::attention_fp64(
                std::span<const float>(queries.data() + query_base, kHeadDim), keys, values,
                static_cast<double>(kAttentionScale));
            std::copy(head.begin(), head.end(),
                      output.begin() + static_cast<std::ptrdiff_t>(query_base));
        }
    }
    return output;
}

void require_invalid(hipError_t status, const char* label) {
    if (status != hipErrorInvalidValue) {
        fail(std::string(label) + " was not rejected before launch");
    }
}

// Exercise the actual target bridge with a pending publication and device-selected compact
// table row. Other rows remain unmapped: ignoring the selector cannot produce the oracle.
void qualify_long_dflash(ninfer::DeviceContext& device, std::uint32_t selected_context = 0U) {
    if constexpr (kValueGroup != 16U) return;
    constexpr std::size_t guard_bytes = 256U;
    for (const std::uint32_t context : selected_context != 0U
                                           ? std::vector<std::uint32_t>{selected_context}
                                           : std::vector<std::uint32_t>{15200U, 32768U}) {
        const std::uint32_t mapped_pages   = (context + kPageSize - 1U) / kPageSize;
        const std::uint32_t physical_pages = mapped_pages + 1U;
        ninfer::LayoutBuilder builder;
        const ninfer::Fp8KInt4VPagedKVSpec spec{
            .page_group_count      = physical_pages,
            .logical_page_capacity = mapped_pages,
            .table_rows            = 4U,
            .layer_count           = 1U,
            .head_dim              = 256,
            .num_kv_heads          = 4,
            .value_group           = 16,
            .plane_layouts = {.key   = ninfer::Fp8KInt4VPlaneLayout::TokenFastestHeadMajor,
                              .value = ninfer::Fp8KInt4VPlaneLayout::FeatureFastestPageMajor,
                              .value_scale =
                                  ninfer::Fp8KInt4VPlaneLayout::FeatureFastestPageMajor}};
        const q3::PagedKVCacheLayout layout{
            .storage     = ninfer::plan_fp8_k_int4_v_paged_kv_pool(builder, spec),
            .layers      = 1U,
            .max_context = mapped_pages * 64U};
        ninfer::DeviceBuffer backing(builder.finish(256, "long DFlash qualification cache"));
        q3::PagedKVCache typed_cache({backing.data(), backing.size()}, layout);
        auto first    = typed_cache.pool().reserve(1U);
        auto occupied = typed_cache.pool().reserve(1U);
        auto rest     = typed_cache.pool().reserve(mapped_pages - 1U);
        first.materialize_pages(1U, device.stream);
        occupied.materialize_pages(1U, device.stream);
        rest.materialize_pages(mapped_pages - 1U, device.stream);
        first.release();
        rest.release();
        auto allocation = typed_cache.pool().reserve(mapped_pages);
        allocation.materialize_pages(mapped_pages, device.stream);
        std::vector<std::uint32_t> pages(allocation.page_ids().begin(),
                                         allocation.page_ids().end());
        if (pages.size() != mapped_pages || pages[0] + 1U == pages[1])
            fail("long DFlash fixture requires fragmented physical pages");
        const auto source_keys = make_keys(context);
        auto source_values     = make_values(context);
        for (std::size_t t = 0; t < context; ++t)
            for (std::size_t h = 0; h < kKvHeads; ++h)
                for (std::size_t f = 0; f < kHeadDim; ++f) {
                    std::uint32_t hash = static_cast<std::uint32_t>(t) * 747796405U +
                                         static_cast<std::uint32_t>(h) * 2891336453U +
                                         static_cast<std::uint32_t>(f) * 277803737U;
                    hash               = ((hash >> ((hash >> 28U) + 4U)) ^ hash) * 277803737U;
                    hash               = (hash >> 22U) ^ hash;
                    const float mean   = (static_cast<int>((h * 7U + f) % 31U) - 15) / 8.0F;
                    source_values[(t * kKvHeads + h) * kHeadDim + f] =
                        represented_bf16(mean + (static_cast<int>(hash % 101U) - 50) / 64.0F);
                }
        const auto cache = encode_cache(source_keys, source_values, pages, physical_pages);
        for (const std::uint32_t rows : selected_context != 0U
                                            ? std::vector<std::uint32_t>{1U, 4U, 8U}
                                            : std::vector<std::uint32_t>{4U, 5U, 6U, 7U, 8U}) {
            // Both extents cover the five widths; together they select every compact C1..4 row.
            const std::int32_t selected_row =
                rows == 4U || rows == 8U
                    ? 0
                    : (rows == 6U ? 3 : ((context == 15200U) == (rows == 5U) ? 1 : 2));
            allocation.unbind_row();
            const auto& tables = typed_cache.pool().block_tables();
            HIP_CHECK(hipMemsetAsync(tables.data, 0xff, tables.bytes(), device.stream));
            allocation.bind_row(selected_row, device.stream);
            const auto view = typed_cache.layer_view(0, allocation);
            HIP_CHECK(hipMemcpyAsync(view.key_fp8_pages.data, cache.physical_keys.data(),
                                     cache.physical_keys.size(), hipMemcpyHostToDevice,
                                     device.stream));
            HIP_CHECK(hipMemcpyAsync(view.value_int4_pages.data, cache.physical_values.data(),
                                     cache.physical_values.size(), hipMemcpyHostToDevice,
                                     device.stream));
            HIP_CHECK(hipMemcpyAsync(view.value_scale_fp16_pages.data, cache.physical_scales.data(),
                                     cache.physical_scales.size() * sizeof(std::uint16_t),
                                     hipMemcpyHostToDevice, device.stream));
            const auto queries      = make_queries(rows);
            const auto bf16_queries = as_bf16(queries);
            std::vector<std::int32_t> positions(rows);
            std::vector<std::size_t> frontiers(rows);
            for (std::uint32_t r = 0; r < rows; ++r) {
                positions[r] = context - rows + r;
                frontiers[r] = static_cast<std::size_t>(positions[r]) + 1U;
            }
            ninfer::DeviceBuffer q(bf16_queries.size() * sizeof(hip_bfloat16));
            ninfer::DeviceBuffer pos(rows * sizeof(std::int32_t));
            ninfer::DeviceBuffer table_row(sizeof(std::int32_t)), base(sizeof(std::int32_t));
            ninfer::DeviceBuffer status(sizeof(std::uint32_t)), cursor(sizeof(std::uint32_t));
            q.copy_from_host(bf16_queries.data(), q.size());
            pos.copy_from_host(positions.data(), pos.size());
            table_row.copy_from_host(&selected_row, sizeof(selected_row));
            base.copy_from_host(positions.data(), sizeof(std::int32_t));
            constexpr std::size_t token_elements = kKvHeads * kHeadDim;
            const auto suffix_start =
                static_cast<std::ptrdiff_t>((context - rows) * token_elements);
            const auto append_keys =
                as_bf16(std::vector<float>(source_keys.begin() + suffix_start, source_keys.end()));
            const auto append_values = as_bf16(
                std::vector<float>(source_values.begin() + suffix_start, source_values.end()));
            ninfer::DeviceBuffer dk(append_keys.size() * sizeof(hip_bfloat16));
            ninfer::DeviceBuffer dv(append_values.size() * sizeof(hip_bfloat16));
            dk.copy_from_host(append_keys.data(), dk.size());
            dv.copy_from_host(append_values.data(), dv.size());
            q3::PagedKVPublication publication{
                .valid_frontier = context - rows, .healthy = true, .transaction_open = false};
            auto transaction = typed_cache.begin_device_segmented_append(
                allocation, publication, static_cast<const std::int32_t*>(base.data()),
                context - rows, context - rows, context,
                {.status = static_cast<std::uint32_t*>(status.data()),
                 .cursor = static_cast<std::uint32_t*>(cursor.data())},
                static_cast<const std::int32_t*>(table_row.data()));
            transaction.begin_device_segment(static_cast<const std::int32_t*>(pos.data()), rows,
                                             nullptr);
            transaction.launch_append_layer(0, static_cast<const hip_bfloat16*>(dk.data()),
                                            static_cast<const hip_bfloat16*>(dv.data()),
                                            device.stream);
            const auto read = transaction.pending_layer_read(0);
            transaction.end_device_segment(device.stream);
            const std::size_t scratch_bytes =
                q27::r9700_full_attention_workspace_capacity_bytes(rows, context, false);
            const std::size_t score_bytes =
                ninfer::ops::r9700::kv::fp8_int4_kv_attention_packed_decode_workspace_bytes(context,
                                                                                            rows);
            if (score_bytes == 0U || scratch_bytes < score_bytes)
                fail("long DFlash planner underallocates split partials");
            const std::size_t output_bytes = queries.size() * sizeof(float);
            ninfer::DeviceBuffer scratch(scratch_bytes + 2U * guard_bytes);
            ninfer::DeviceBuffer output(output_bytes + 2U * guard_bytes);
            auto* scratch_payload = static_cast<std::uint8_t*>(scratch.data()) + guard_bytes;
            auto* output_payload  = static_cast<std::uint8_t*>(output.data()) + guard_bytes;
            const q27::R9700FullAttentionArgs args{
                .query           = ninfer::Tensor(q.data(), ninfer::DType::BF16,
                                                  {256, 24, static_cast<std::int32_t>(rows)}),
                .cache_read      = read,
                .row_positions   = static_cast<const std::int32_t*>(pos.data()),
                .workspace       = scratch_payload,
                .workspace_bytes = scratch_bytes,
                .output          = ninfer::Tensor(output_payload, ninfer::DType::FP32,
                                                  {256, 24, static_cast<std::int32_t>(rows)})};
            auto short_workspace            = args;
            short_workspace.workspace_bytes = score_bytes - 1U;
            require_invalid(q27::r9700_qwen3_8_27b_full_attention(short_workspace, device.stream),
                            "long DFlash undersized score workspace");
            const auto poison = [&] {
                HIP_CHECK(hipMemsetAsync(scratch.data(), 0xa5, scratch.size(), device.stream));
                HIP_CHECK(hipMemsetAsync(output.data(), 0xa5, output.size(), device.stream));
                HIP_CHECK(hipMemsetAsync(output_payload, 0xff, output_bytes, device.stream));
            };
            const auto read_result = [&] {
                device.synchronize();
                const auto guards = [&](ninfer::DeviceBuffer& buffer) {
                    std::array<std::uint8_t, guard_bytes> front{}, back{};
                    HIP_CHECK(
                        hipMemcpy(front.data(), buffer.data(), guard_bytes, hipMemcpyDeviceToHost));
                    HIP_CHECK(hipMemcpy(back.data(),
                                        static_cast<std::uint8_t*>(buffer.data()) + buffer.size() -
                                            guard_bytes,
                                        guard_bytes, hipMemcpyDeviceToHost));
                    const auto intact = [](std::uint8_t x) { return x == 0xa5; };
                    if (!std::all_of(front.begin(), front.end(), intact) ||
                        !std::all_of(back.begin(), back.end(), intact))
                        fail("long DFlash output/workspace guard overwritten");
                };
                guards(scratch);
                guards(output);
                std::vector<float> result(queries.size());
                HIP_CHECK(
                    hipMemcpy(result.data(), output_payload, output_bytes, hipMemcpyDeviceToHost));
                return result;
            };
            poison();
            HIP_CHECK(q27::r9700_qwen3_8_27b_full_attention(args, device.stream));
            const auto actual = read_result();
            // Dense-attention criterion (represented BF16 Q, exact-BF16 K, FP16 probabilities and
            // V/8 with FP32 accumulation): |error| <= 2e-3 absolute or relative.
            const auto expected  = attention_oracle(queries, rows, frontiers, cache);
            double maximum_error = 0.0;
            for (std::size_t i = 0; i < actual.size(); ++i) {
                const double error = std::abs(actual[i] - expected[i]);
                if (!std::isfinite(actual[i]) ||
                    (error > 2e-3 && error / std::max(std::abs(expected[i]), 1e-6) > 2e-3))
                    fail("long DFlash public FP64 oracle mismatch");
                maximum_error = std::max(maximum_error, error);
            }
            ninfer::DecodeGraphDefinition definition;
            definition.capture(device.stream, [&] {
                HIP_CHECK(q27::r9700_qwen3_8_27b_full_attention(args, device.stream));
            });
            ninfer::DecodeGraphExecutable graph;
            graph.instantiate(definition);
            poison();
            graph.launch(device.stream);
            if (read_result() != actual)
                fail("long DFlash poisoned graph replay differs from eager");
            // MTP keeps fixed graph addresses while the device scalar changes the live prefix.
            // Poison inactive positions so reading tail metadata cannot silently pass.
            ninfer::DeviceBuffer active_count(sizeof(std::int32_t));
            auto active_args              = args;
            active_args.active_query_rows = static_cast<const std::int32_t*>(active_count.data());
            ninfer::DecodeGraphDefinition active_definition;
            active_definition.capture(device.stream, [&] {
                HIP_CHECK(q27::r9700_qwen3_8_27b_full_attention(active_args, device.stream));
            });
            ninfer::DecodeGraphExecutable active_graph;
            active_graph.instantiate(active_definition);
            for (const std::int32_t count :
                 {0, static_cast<std::int32_t>(rows / 2U), static_cast<std::int32_t>(rows), -1,
                  static_cast<std::int32_t>(rows + 1U), static_cast<std::int32_t>(rows)}) {
                const bool valid      = count >= 0 && count <= static_cast<std::int32_t>(rows);
                auto active_positions = positions;
                const auto live       = valid ? static_cast<std::size_t>(count) : 0U;
                std::fill(active_positions.begin() + static_cast<std::ptrdiff_t>(live),
                          active_positions.end(), -777);
                pos.copy_from_host(active_positions.data(), rows * sizeof(std::int32_t));
                active_count.copy_from_host(&count, sizeof(count));
                poison();
                active_graph.launch(device.stream);
                const auto result = read_result();
                for (std::size_t i = 0; i < result.size(); ++i) {
                    if (!valid) {
                        if (!std::isnan(result[i])) fail("invalid MTP count did not poison output");
                    } else if (i >= live * kQueryHeads * kHeadDim) {
                        if (std::bit_cast<std::uint32_t>(result[i]) != 0U)
                            fail("inactive MTP row is not exact positive zero");
                    } else {
                        const double error = std::abs(result[i] - expected[i]);
                        if (!std::isfinite(result[i]) ||
                            (error > 2e-3 && error / std::max(std::abs(expected[i]), 1e-6) > 2e-3))
                            fail("device-count MTP graph differs from FP64 oracle");
                    }
                }
            }
            std::uint32_t resolved_status = 0, resolved_cursor = 0;
            transaction.enqueue_segmented_resolution(&resolved_status, &resolved_cursor);
            device.synchronize();
            transaction.finish_segmented_resolution(resolved_status, resolved_cursor, context);
            if (resolved_status != 0U || resolved_cursor != context ||
                publication.valid_frontier != context || read.valid_for(device.stream))
                fail("long DFlash pending publication did not resolve correctly");
            std::cout << "long_dflash_leaf context=" << context << " rows=" << rows
                      << " compact_table_row=" << selected_row
                      << " public_max_abs=" << maximum_error
                      << " oracle/guards/poisoned_graph/device_counts=PASS\n";
        }
    }
}

} // namespace

int main(int argc, char** argv) {
    try {
        std::uint32_t selected_context = 0;
        if (argc == 3 && std::string(argv[1]) == "--long-context") {
            selected_context = static_cast<std::uint32_t>(std::stoul(argv[2]));
            if (selected_context < 8192U || selected_context > 262144U)
                fail("invalid long context");
        } else if (argc != 1)
            fail("usage: full_attention_qual [--long-context 8192..262144]");
        const auto selects_packed = [](std::uint32_t rows, std::size_t context, bool tree) {
            return ninfer::ops::r9700::kv::use_packed_decode_attention(rows, context, tree);
        };
        const auto selects_split512 = [](std::uint32_t rows, std::size_t context, bool tree) {
            return ninfer::ops::r9700::kv::use_split512_attention(rows, context, tree);
        };
        const auto selects_dense_prefill = [](std::uint32_t rows, std::size_t context) {
            return ninfer::ops::r9700::kv::use_dense_prefill_attention(rows, context);
        };
        if (selects_dense_prefill(127U, 127U) || !selects_dense_prefill(128U, 128U) ||
            !selects_dense_prefill(4096U, 4096U) || !selects_dense_prefill(8192U, 32768U) ||
            selects_dense_prefill(8193U, 32768U) || !selects_dense_prefill(4096U, 8192U)) {
            fail("dense-prefill row/context classifier is not the admitted boundary");
        }
        if (!selects_packed(1U, 64U, false) || !selects_packed(6U, 262144U, false) ||
            !selects_packed(8U, 262144U, false) || selects_packed(1U, 63U, false) ||
            selects_packed(9U, 4096U, false) || selects_packed(0U, 4096U, false) ||
            selects_packed(4U, 4096U, true) || selects_packed(1U, 262145U, false)) {
            fail("packed decode row/context/tree classifier is not the admitted boundary");
        }
        if (ninfer::ops::r9700::kv::kFp8QkWmmaDecode) {
            if (selects_split512(4U, 8191U, true) || !selects_split512(4U, 8192U, true) ||
                selects_split512(4U, 8192U, false) || selects_split512(1U, 8192U, true) ||
                selects_split512(5U, 32768U, true)) {
                fail("split-512 context/row classifier is not the admitted boundary");
            }
        } else if (selects_split512(4U, 8192U, true)) {
            fail("score-streaming control unexpectedly selects split-512 attention");
        }
        ninfer::DeviceContext device(0);
        if (!device.is_gfx1201() || device.props.warpSize != 32) {
            fail("selected physical device is not wave32 gfx1201");
        }
        qualify_long_dflash(device, selected_context);
        if (selected_context != 0U) return 0;

        ninfer::LayoutBuilder cache_builder;
        const ninfer::Fp8KInt4VPagedKVSpec cache_spec{
            .page_group_count      = static_cast<std::uint32_t>(kPhysicalPages),
            .logical_page_capacity = static_cast<std::uint32_t>(kMappedPages),
            .table_rows            = 1,
            .layer_count           = 1,
            .head_dim              = static_cast<std::int32_t>(kHeadDim),
            .num_kv_heads          = static_cast<std::int32_t>(kKvHeads),
            .value_group           = static_cast<std::int32_t>(kValueGroup),
            .plane_layouts = {.key   = ninfer::Fp8KInt4VPlaneLayout::TokenFastestHeadMajor,
                              .value = ninfer::Fp8KInt4VPlaneLayout::FeatureFastestPageMajor,
                              .value_scale =
                                  ninfer::Fp8KInt4VPlaneLayout::FeatureFastestPageMajor}};
        const q3::PagedKVCacheLayout cache_layout{
            .storage     = ninfer::plan_fp8_k_int4_v_paged_kv_pool(cache_builder, cache_spec),
            .layers      = 1,
            .max_context = static_cast<std::uint32_t>(kMappedPages * kPageSize)};
        ninfer::DeviceBuffer cache_backing(cache_builder.finish(256, "full-attention typed cache"));
        q3::PagedKVCache typed_cache({cache_backing.data(), cache_backing.size()}, cache_layout);

        // Leave physical page 1 occupied so the tested sequence receives fragmented pages 0,2.
        auto left   = typed_cache.pool().reserve(1);
        auto middle = typed_cache.pool().reserve(1);
        auto right  = typed_cache.pool().reserve(static_cast<std::uint32_t>(kPhysicalPages - 2U));
        left.materialize_pages(1, device.stream);
        middle.materialize_pages(1, device.stream);
        right.materialize_pages(static_cast<std::uint32_t>(kPhysicalPages - 2U), device.stream);
        left.release();
        right.release();
        auto allocation = typed_cache.pool().reserve(static_cast<std::uint32_t>(kMappedPages));
        allocation.materialize_pages(static_cast<std::uint32_t>(kMappedPages), device.stream);
        allocation.bind_row(0, device.stream);
        std::vector<std::uint32_t> page_table;
        page_table.reserve(kMappedPages);
        for (const std::uint32_t page : allocation.page_ids()) { page_table.push_back(page); }
        if (page_table.size() != kMappedPages) {
            fail("typed cache allocation has the wrong mapped-page count");
        }
        if (page_table[0] == page_table[1] || page_table[0] + 1U == page_table[1]) {
            fail("typed cache qualifier did not obtain a fragmented page map");
        }

        const std::vector<float> source_keys   = make_keys();
        const std::vector<float> source_values = make_values();
        const EncodedCache cache = encode_cache(source_keys, source_values, page_table);

        ninfer::DeviceBuffer d_queries(kMaximumQualifiedRows * kQueryHeads * kHeadDim *
                                       sizeof(hip_bfloat16));
        ninfer::DeviceBuffer d_output(kMaximumQualifiedRows * kQueryHeads * kHeadDim *
                                      sizeof(float));
        ninfer::DeviceBuffer d_row_positions(kMaximumQualifiedRows * sizeof(std::int32_t));
        ninfer::DeviceBuffer d_ancestor_masks(kMaximumQualifiedRows * sizeof(std::int32_t));
        ninfer::DeviceBuffer d_prefix_lengths(kMaximumQualifiedRows * sizeof(std::int32_t));
        ninfer::DeviceBuffer d_active_query_rows(sizeof(std::int32_t));
        // Dense prefill needs none; the packed decode partials (rows 1..6) and the single-row WMMA
        // score plane are the consumers.
        std::size_t score_workspace_capacity =
            q27::r9700_full_attention_score_workspace_capacity_bytes(kContext + 1U);
        for (std::uint32_t rows = 1U; rows <= kMaximumQualifiedRows; ++rows)
            score_workspace_capacity = std::max(
                score_workspace_capacity,
                q27::r9700_full_attention_workspace_capacity_bytes(rows, kContext + 1U, false));
        ninfer::DeviceBuffer d_score_workspace(score_workspace_capacity);
        const ninfer::Fp8KInt4VPagedKVLayerView view = typed_cache.layer_view(0, allocation);
        if (view.key_fp8_pages.bytes() != cache.physical_keys.size() ||
            view.value_int4_pages.bytes() != cache.physical_values.size() ||
            view.value_scale_fp16_pages.bytes() !=
                cache.physical_scales.size() * sizeof(std::uint16_t)) {
            fail("typed cache physical planes differ from the independent layout oracle");
        }
        HIP_CHECK(hipMemcpyAsync(view.key_fp8_pages.data, cache.physical_keys.data(),
                                 cache.physical_keys.size(), hipMemcpyHostToDevice, device.stream));
        HIP_CHECK(hipMemcpyAsync(view.value_int4_pages.data, cache.physical_values.data(),
                                 cache.physical_values.size(), hipMemcpyHostToDevice,
                                 device.stream));
        HIP_CHECK(hipMemcpyAsync(view.value_scale_fp16_pages.data, cache.physical_scales.data(),
                                 cache.physical_scales.size() * sizeof(std::uint16_t),
                                 hipMemcpyHostToDevice, device.stream));
        device.synchronize();

        q3::PagedKVPublication publication{.valid_frontier   = static_cast<std::uint32_t>(kContext),
                                           .healthy          = true,
                                           .transaction_open = false};
        const q3::PagedKVLayerRead committed_read =
            typed_cache.published_layer_read(allocation, publication, 0);
        double maximum_absolute        = 0.0;
        double maximum_relative        = 0.0;
        double packed_maximum_absolute = 0.0;
        double packed_maximum_relative = 0.0;
        std::array<float, 2> decode_latency_ms{};
        std::array<float, 3> prefill_latency_ms{};
        constexpr std::array<std::size_t, 11> qualified_widths{1U, 2U, 3U, 4U,  5U,  6U,
                                                               7U, 8U, 9U, 17U, 128U};
        for (const std::size_t rows : qualified_widths) {
            const std::vector<float> queries             = make_queries(rows);
            const std::vector<hip_bfloat16> bf16_queries = as_bf16(queries);
            std::vector<std::size_t> frontiers(rows);
            std::vector<std::int32_t> positions(rows);
            for (std::size_t row = 0; row < rows; ++row) {
                frontiers[row] = kContext - (rows - 1U - row);
                positions[row] = static_cast<std::int32_t>(frontiers[row] - 1U);
            }
            d_queries.copy_from_host(bf16_queries.data(),
                                     bf16_queries.size() * sizeof(hip_bfloat16));
            d_row_positions.copy_from_host(positions.data(),
                                           positions.size() * sizeof(std::int32_t));
            const std::size_t score_workspace_bytes =
                q27::r9700_full_attention_workspace_capacity_bytes(static_cast<std::uint32_t>(rows),
                                                                   kContext, false);
            const bool uses_packed = ninfer::ops::r9700::kv::use_packed_decode_attention(
                static_cast<std::uint32_t>(rows), kContext, false);
            const q27::R9700FullAttentionArgs args{
                .query           = ninfer::Tensor(d_queries.data(), ninfer::DType::BF16,
                                                  {static_cast<std::int32_t>(kHeadDim),
                                                   static_cast<std::int32_t>(kQueryHeads),
                                                   static_cast<std::int32_t>(rows)}),
                .cache_read      = committed_read,
                .row_positions   = static_cast<const std::int32_t*>(d_row_positions.data()),
                .workspace       = score_workspace_bytes == 0U ? nullptr : d_score_workspace.data(),
                .workspace_bytes = score_workspace_bytes,
                .output          = ninfer::Tensor(d_output.data(), ninfer::DType::FP32,
                                                  {static_cast<std::int32_t>(kHeadDim),
                                                   static_cast<std::int32_t>(kQueryHeads),
                                                   static_cast<std::int32_t>(rows)})};
            HIP_CHECK(q27::r9700_qwen3_8_27b_full_attention(args, device.stream));
            device.synchronize();
            std::vector<float> actual(queries.size());
            d_output.copy_to_host(actual.data(), actual.size() * sizeof(float));
            const std::vector<double> expected = attention_oracle(queries, rows, frontiers, cache);
            for (std::size_t index = 0; index < actual.size(); ++index) {
                const double absolute =
                    std::abs(static_cast<double>(actual[index]) - expected[index]);
                const double relative = absolute / std::max(std::abs(expected[index]), 1.0e-6);
                maximum_absolute      = std::max(maximum_absolute, absolute);
                maximum_relative      = std::max(maximum_relative, relative);
                if (uses_packed) {
                    packed_maximum_absolute = std::max(packed_maximum_absolute, absolute);
                    packed_maximum_relative = std::max(packed_maximum_relative, relative);
                }
                // Packed decode: the dense-attention criterion (BF16 Q, exact-BF16 K, FP16
                // probabilities and V/8 with FP32 accumulation).
                const double tolerance = uses_packed ? 2.0e-3 : 2.0e-4;
                if (!std::isfinite(actual[index]) || !std::isfinite(expected[index]) ||
                    !std::isfinite(absolute) || !std::isfinite(relative) ||
                    (absolute > tolerance && relative > tolerance)) {
                    std::ostringstream out;
                    out << "T=" << rows << " FP64 attention mismatch at " << index
                        << ": actual=" << actual[index] << " expected=" << expected[index]
                        << " abs=" << absolute << " rel=" << relative;
                    fail(out.str());
                }
            }
            if (rows <= 2U) {
                constexpr int kWarmupIterations = 20;
                for (int warmup = 0; warmup < kWarmupIterations; ++warmup) {
                    HIP_CHECK(q27::r9700_qwen3_8_27b_full_attention(args, device.stream));
                }
                constexpr int kTimingIterations = 50;
                ninfer::DeviceEventTimer timer(device);
                timer.start();
                for (int iteration = 0; iteration < kTimingIterations; ++iteration) {
                    HIP_CHECK(q27::r9700_qwen3_8_27b_full_attention(args, device.stream));
                }
                decode_latency_ms[rows - 1U] =
                    timer.stop_ms() / static_cast<float>(kTimingIterations);
            }
            if (rows == 9U || rows == 17U || rows == 128U) {
                constexpr int kWarmupIterations = 20;
                for (int warmup = 0; warmup < kWarmupIterations; ++warmup) {
                    HIP_CHECK(q27::r9700_qwen3_8_27b_full_attention(args, device.stream));
                }
                constexpr int kTimingIterations = 50;
                ninfer::DeviceEventTimer timer(device);
                timer.start();
                for (int iteration = 0; iteration < kTimingIterations; ++iteration) {
                    HIP_CHECK(q27::r9700_qwen3_8_27b_full_attention(args, device.stream));
                }
                const float elapsed = timer.stop_ms() / static_cast<float>(kTimingIterations);
                const std::size_t timing_index   = rows == 9U ? 0U : rows == 17U ? 1U : 2U;
                prefill_latency_ms[timing_index] = elapsed;
            }
        }

        // A captured MTP panel retains its fixed T allocation while a device scalar selects the
        // live prefix. The inactive rows deliberately carry invalid causal positions, proving
        // that zero-tail behavior is decided before any row metadata or cache lookup. Malformed
        // counts are a whole-invocation error and therefore poison every output element.
        constexpr std::size_t kActiveCountRows            = 4U;
        const std::vector<float> active_count_queries     = make_queries(kActiveCountRows);
        const std::vector<hip_bfloat16> active_count_bf16 = as_bf16(active_count_queries);
        const std::vector<float> output_sentinel(active_count_queries.size(), 17.0F);
        d_queries.copy_from_host(active_count_bf16.data(),
                                 active_count_bf16.size() * sizeof(hip_bfloat16));
        q27::R9700FullAttentionArgs active_count_args{
            .query             = ninfer::Tensor(d_queries.data(), ninfer::DType::BF16,
                                                {static_cast<std::int32_t>(kHeadDim),
                                                 static_cast<std::int32_t>(kQueryHeads),
                                                 static_cast<std::int32_t>(kActiveCountRows)}),
            .cache_read        = committed_read,
            .row_positions     = static_cast<const std::int32_t*>(d_row_positions.data()),
            .active_query_rows = static_cast<const std::int32_t*>(d_active_query_rows.data()),
            .workspace         = d_score_workspace.data(),
            .workspace_bytes = q27::r9700_full_attention_workspace_capacity_bytes(kActiveCountRows,
                                                                                  kContext, false),
            .output          = ninfer::Tensor(d_output.data(), ninfer::DType::FP32,
                                              {static_cast<std::int32_t>(kHeadDim),
                                               static_cast<std::int32_t>(kQueryHeads),
                                               static_cast<std::int32_t>(kActiveCountRows)})};
        ninfer::DecodeGraphDefinition active_count_graph;
        active_count_graph.capture(device.stream, [&] {
            HIP_CHECK(q27::r9700_qwen3_8_27b_full_attention(active_count_args, device.stream));
        });
        ninfer::DecodeGraphExecutable active_count_executable;
        active_count_executable.instantiate(active_count_graph);
        active_count_executable.upload(device.stream);
        const std::size_t active_row_width = kQueryHeads * kHeadDim;
        const auto qualify_active_count = [&](std::int32_t represented_count, const char* label) {
            std::vector<std::int32_t> positions(kActiveCountRows, -777);
            const std::size_t live_rows =
                represented_count >= 0 &&
                        represented_count <= static_cast<std::int32_t>(kActiveCountRows)
                    ? static_cast<std::size_t>(represented_count)
                    : 0U;
            std::vector<std::size_t> frontiers(live_rows);
            for (std::size_t row = 0; row < live_rows; ++row) {
                frontiers[row] = kContext - (kActiveCountRows - 1U - row);
                positions[row] = static_cast<std::int32_t>(frontiers[row] - 1U);
            }
            d_row_positions.copy_from_host(positions.data(),
                                           positions.size() * sizeof(std::int32_t));
            d_active_query_rows.copy_from_host(&represented_count, sizeof(represented_count));
            d_output.copy_from_host(output_sentinel.data(), output_sentinel.size() * sizeof(float));
            active_count_executable.launch(device.stream);
            device.synchronize();
            std::vector<float> actual(active_count_queries.size());
            d_output.copy_to_host(actual.data(), actual.size() * sizeof(float));
            if (represented_count < 0 ||
                represented_count > static_cast<std::int32_t>(kActiveCountRows)) {
                if (!std::all_of(actual.begin(), actual.end(),
                                 [](float value) { return std::isnan(value); })) {
                    fail(std::string(label) + " did not poison every output row");
                }
                return;
            }
            const std::vector<double> expected =
                attention_oracle(active_count_queries, live_rows, frontiers, cache);
            // Device-selected causal panels now use the same BF16-Q/FP16-probability packed
            // profile and its existing dense-attention criterion as host-fixed panels.
            for (std::size_t index = 0; index < live_rows * active_row_width; ++index) {
                const double absolute =
                    std::abs(static_cast<double>(actual[index]) - expected[index]);
                const double relative = absolute / std::max(std::abs(expected[index]), 1.0e-6);
                if (!std::isfinite(actual[index]) || (absolute > 2.0e-3 && relative > 2.0e-3)) {
                    fail(std::string(label) + " active prefix differs from the FP64 oracle");
                }
            }
            if (!std::all_of(
                    actual.begin() + static_cast<std::ptrdiff_t>(live_rows * active_row_width),
                    actual.end(),
                    [](float value) { return std::bit_cast<std::uint32_t>(value) == 0U; })) {
                fail(std::string(label) + " inactive tail is not exact positive zero");
            }
        };
        qualify_active_count(0, "zero active-row count");
        qualify_active_count(2, "partial active-row count");
        qualify_active_count(static_cast<std::int32_t>(kActiveCountRows), "full active-row count");
        qualify_active_count(-1, "negative active-row count");
        qualify_active_count(static_cast<std::int32_t>(kActiveCountRows + 1U),
                             "over-width active-row count");
        // A valid replay immediately after malformed device state must not inherit an error.
        qualify_active_count(2, "post-error partial active-row count");
        const std::int32_t full_active_count = static_cast<std::int32_t>(kActiveCountRows);
        const std::array<std::int32_t, kActiveCountRows> full_active_positions{253, 254, 255, 256};
        d_active_query_rows.copy_from_host(&full_active_count, sizeof(full_active_count));
        d_row_positions.copy_from_host(full_active_positions.data(),
                                       full_active_positions.size() * sizeof(std::int32_t));
        q27::R9700FullAttentionArgs fixed_count_args = active_count_args;
        fixed_count_args.active_query_rows           = nullptr;
        fixed_count_args.workspace_bytes = q27::r9700_full_attention_workspace_capacity_bytes(
            static_cast<std::uint32_t>(kActiveCountRows), kContext, false);
        fixed_count_args.workspace =
            fixed_count_args.workspace_bytes == 0U ? nullptr : d_score_workspace.data();
        constexpr int kActiveTimingWarmups = 20;
        for (int warmup = 0; warmup < kActiveTimingWarmups; ++warmup) {
            HIP_CHECK(q27::r9700_qwen3_8_27b_full_attention(fixed_count_args, device.stream));
            HIP_CHECK(q27::r9700_qwen3_8_27b_full_attention(active_count_args, device.stream));
        }
        constexpr int kActiveTimingIterations      = 50;
        constexpr std::size_t kActiveTimingSamples = 7;
        ninfer::DeviceEventTimer active_timer(device);
        const auto measure = [&](const q27::R9700FullAttentionArgs& timed_args) {
            active_timer.start();
            for (int iteration = 0; iteration < kActiveTimingIterations; ++iteration) {
                HIP_CHECK(q27::r9700_qwen3_8_27b_full_attention(timed_args, device.stream));
            }
            return active_timer.stop_ms() / static_cast<float>(kActiveTimingIterations);
        };
        std::array<float, kActiveTimingSamples> fixed_count_samples{};
        std::array<float, kActiveTimingSamples> device_count_samples{};
        for (std::size_t sample = 0; sample < kActiveTimingSamples; ++sample) {
            if ((sample & 1U) == 0U) {
                fixed_count_samples[sample]  = measure(fixed_count_args);
                device_count_samples[sample] = measure(active_count_args);
            } else {
                device_count_samples[sample] = measure(active_count_args);
                fixed_count_samples[sample]  = measure(fixed_count_args);
            }
        }
        std::sort(fixed_count_samples.begin(), fixed_count_samples.end());
        std::sort(device_count_samples.begin(), device_count_samples.end());
        const float fixed_count_latency_ms  = fixed_count_samples[kActiveTimingSamples / 2U];
        const float device_count_latency_ms = device_count_samples[kActiveTimingSamples / 2U];

        // Packed-tree visibility is checked against an independent FP64 oracle. Rows two and
        // three deliberately skip sibling token 254, so ordinary causal attention cannot pass.
        constexpr std::size_t kTreeRows           = 4U;
        const std::vector<float> tree_queries     = make_queries(kTreeRows);
        const std::vector<hip_bfloat16> tree_bf16 = as_bf16(tree_queries);
        const std::vector<std::int32_t> tree_positions{253, 254, 255, 256};
        const std::vector<std::int32_t> tree_prefixes{253, 253, 253, 253};
        const std::vector<std::int32_t> tree_masks{1, 3, 5, 13};
        d_queries.copy_from_host(tree_bf16.data(), tree_bf16.size() * sizeof(hip_bfloat16));
        d_row_positions.copy_from_host(tree_positions.data(),
                                       tree_positions.size() * sizeof(std::int32_t));
        d_prefix_lengths.copy_from_host(tree_prefixes.data(),
                                        tree_prefixes.size() * sizeof(std::int32_t));
        d_ancestor_masks.copy_from_host(tree_masks.data(),
                                        tree_masks.size() * sizeof(std::int32_t));
        const q27::R9700FullAttentionArgs tree_args{
            .query          = ninfer::Tensor(d_queries.data(), ninfer::DType::BF16,
                                             {static_cast<std::int32_t>(kHeadDim),
                                              static_cast<std::int32_t>(kQueryHeads),
                                              static_cast<std::int32_t>(kTreeRows)}),
            .cache_read     = committed_read,
            .row_positions  = static_cast<const std::int32_t*>(d_row_positions.data()),
            .ancestor_masks = static_cast<const std::int32_t*>(d_ancestor_masks.data()),
            .prefix_lengths = static_cast<const std::int32_t*>(d_prefix_lengths.data()),
            .output         = ninfer::Tensor(d_output.data(), ninfer::DType::FP32,
                                             {static_cast<std::int32_t>(kHeadDim),
                                              static_cast<std::int32_t>(kQueryHeads),
                                              static_cast<std::int32_t>(kTreeRows)})};
        HIP_CHECK(q27::r9700_qwen3_8_27b_full_attention(tree_args, device.stream));
        device.synchronize();
        std::vector<float> tree_actual(tree_queries.size());
        d_output.copy_to_host(tree_actual.data(), tree_actual.size() * sizeof(float));
        const std::vector<double> tree_expected =
            tree_attention_oracle(tree_queries, tree_positions, tree_prefixes, tree_masks, cache);
        double tree_maximum_absolute = 0.0;
        double tree_maximum_relative = 0.0;
        for (std::size_t index = 0; index < tree_actual.size(); ++index) {
            const double absolute =
                std::abs(static_cast<double>(tree_actual[index]) - tree_expected[index]);
            const double relative = absolute / std::max(std::abs(tree_expected[index]), 1.0e-6);
            tree_maximum_absolute = std::max(tree_maximum_absolute, absolute);
            tree_maximum_relative = std::max(tree_maximum_relative, relative);
            if (absolute > 2.0e-4 && relative > 2.0e-4) {
                std::ostringstream out;
                out << "packed-tree FP64 attention mismatch at " << index
                    << ": actual=" << tree_actual[index] << " expected=" << tree_expected[index]
                    << " abs=" << absolute << " rel=" << relative;
                fail(out.str());
            }
        }

        const std::array<std::int32_t, 1> broadcast_prefix{253};
        d_prefix_lengths.copy_from_host(broadcast_prefix.data(), sizeof(std::int32_t));
        q27::R9700FullAttentionArgs broadcast_tree_args = tree_args;
        broadcast_tree_args.prefix_length_stride        = 0U;
        HIP_CHECK(q27::r9700_qwen3_8_27b_full_attention(broadcast_tree_args, device.stream));
        device.synchronize();
        std::vector<float> broadcast_tree_actual(tree_queries.size());
        d_output.copy_to_host(broadcast_tree_actual.data(),
                              broadcast_tree_actual.size() * sizeof(float));
        for (std::size_t index = 0; index < broadcast_tree_actual.size(); ++index) {
            const double absolute =
                std::abs(static_cast<double>(broadcast_tree_actual[index]) - tree_expected[index]);
            const double relative = absolute / std::max(std::abs(tree_expected[index]), 1.0e-6);
            tree_maximum_absolute = std::max(tree_maximum_absolute, absolute);
            tree_maximum_relative = std::max(tree_maximum_relative, relative);
            if (absolute > 2.0e-4 && relative > 2.0e-4) {
                fail("broadcast-prefix packed-tree attention differs from the FP64 oracle");
            }
        }

        // Exercise the device count together with rowwise packed-tree metadata. Both inactive
        // rows contain malformed causal/tree data yet must remain exact zero without inspection.
        const std::vector<std::int32_t> partial_tree_positions{253, 254, -1, -1};
        const std::vector<std::int32_t> partial_tree_prefixes{253, 253, -1, -1};
        const std::vector<std::int32_t> partial_tree_masks{1, 3, 0, 0};
        const std::int32_t partial_tree_count = 2;
        d_row_positions.copy_from_host(partial_tree_positions.data(),
                                       partial_tree_positions.size() * sizeof(std::int32_t));
        d_prefix_lengths.copy_from_host(partial_tree_prefixes.data(),
                                        partial_tree_prefixes.size() * sizeof(std::int32_t));
        d_ancestor_masks.copy_from_host(partial_tree_masks.data(),
                                        partial_tree_masks.size() * sizeof(std::int32_t));
        d_active_query_rows.copy_from_host(&partial_tree_count, sizeof(partial_tree_count));
        d_output.copy_from_host(output_sentinel.data(), output_sentinel.size() * sizeof(float));
        q27::R9700FullAttentionArgs partial_tree_args = tree_args;
        partial_tree_args.active_query_rows =
            static_cast<const std::int32_t*>(d_active_query_rows.data());
        HIP_CHECK(q27::r9700_qwen3_8_27b_full_attention(partial_tree_args, device.stream));
        device.synchronize();
        std::vector<float> partial_tree_actual(tree_queries.size());
        d_output.copy_to_host(partial_tree_actual.data(),
                              partial_tree_actual.size() * sizeof(float));
        const std::vector<float> partial_tree_queries(
            tree_queries.begin(),
            tree_queries.begin() +
                static_cast<std::ptrdiff_t>(partial_tree_count * active_row_width));
        const std::vector<double> partial_tree_expected = tree_attention_oracle(
            partial_tree_queries, std::vector<std::int32_t>{253, 254},
            std::vector<std::int32_t>{253, 253}, std::vector<std::int32_t>{1, 3}, cache);
        for (std::size_t index = 0; index < partial_tree_expected.size(); ++index) {
            const double absolute = std::abs(static_cast<double>(partial_tree_actual[index]) -
                                             partial_tree_expected[index]);
            const double relative =
                absolute / std::max(std::abs(partial_tree_expected[index]), 1.0e-6);
            if (absolute > 2.0e-4 && relative > 2.0e-4) {
                fail("active-prefix packed-tree attention differs from the FP64 oracle");
            }
        }
        if (!std::all_of(partial_tree_actual.begin() +
                             static_cast<std::ptrdiff_t>(partial_tree_expected.size()),
                         partial_tree_actual.end(),
                         [](float value) { return std::bit_cast<std::uint32_t>(value) == 0U; })) {
            fail("inactive packed-tree tail is not exact positive zero");
        }

        // Restore the canonical tree corpus for the remaining host/device rejection cases.
        d_row_positions.copy_from_host(tree_positions.data(),
                                       tree_positions.size() * sizeof(std::int32_t));
        d_prefix_lengths.copy_from_host(broadcast_prefix.data(), sizeof(std::int32_t));
        d_ancestor_masks.copy_from_host(tree_masks.data(),
                                        tree_masks.size() * sizeof(std::int32_t));

        q27::R9700FullAttentionArgs incomplete_tree = tree_args;
        incomplete_tree.prefix_lengths              = nullptr;
        require_invalid(q27::r9700_qwen3_8_27b_full_attention(incomplete_tree, device.stream),
                        "ancestor mask without prefix length");
        incomplete_tree                = tree_args;
        incomplete_tree.ancestor_masks = nullptr;
        require_invalid(q27::r9700_qwen3_8_27b_full_attention(incomplete_tree, device.stream),
                        "prefix length without ancestor mask");
        q27::R9700FullAttentionArgs invalid_stride = tree_args;
        invalid_stride.prefix_length_stride        = 2U;
        require_invalid(q27::r9700_qwen3_8_27b_full_attention(invalid_stride, device.stream),
                        "unsupported packed-tree prefix stride");

        const auto& raw_cache = committed_read.layer();
        ninfer::ops::r9700::kv::Fp8Int4KvAttentionArgs raw_tree{
            .query       = static_cast<const hip_bfloat16*>(d_queries.data()),
            .key_codes   = static_cast<const std::uint8_t*>(raw_cache.key_fp8_pages.data),
            .value_codes = static_cast<const std::uint8_t*>(raw_cache.value_int4_pages.data),
            .value_scale_bits =
                static_cast<const std::uint16_t*>(raw_cache.value_scale_fp16_pages.data),
            .page_table           = static_cast<const std::uint32_t*>(raw_cache.block_table.data),
            .page_table_pages     = static_cast<std::uint32_t>(kMappedPages),
            .context              = kContext,
            .physical_tokens      = kPhysicalTokens,
            .query_heads          = static_cast<std::uint32_t>(kQueryHeads),
            .kv_heads             = static_cast<std::uint32_t>(kKvHeads),
            .q_to_kv              = static_cast<std::uint32_t>(kQueriesPerKv),
            .value_group          = static_cast<std::uint32_t>(kValueGroup),
            .query_rows           = static_cast<std::uint32_t>(kTreeRows),
            .row_positions        = static_cast<const std::int32_t*>(d_row_positions.data()),
            .ancestor_masks       = static_cast<const std::int32_t*>(d_ancestor_masks.data()),
            .prefix_lengths       = static_cast<const std::int32_t*>(d_prefix_lengths.data()),
            .prefix_length_stride = 0U,
            .attention_scale      = kAttentionScale,
            .key_layout           = ninfer::ops::r9700::kv::Fp8Int4KvPlaneLayout::TokenFastest,
            .value_layout         = ninfer::ops::r9700::kv::Fp8Int4KvPlaneLayout::FeatureFastest,
            .value_scale_layout   = ninfer::ops::r9700::kv::Fp8Int4KvPlaneLayout::FeatureFastest,
            .score_workspace      = static_cast<float*>(d_score_workspace.data()),
            .output               = static_cast<float*>(d_output.data())};
        require_invalid(ninfer::ops::r9700::kv::fp8_int4_kv_attention_wmma(raw_tree, device.stream),
                        "WMMA packed-tree route");
        raw_tree.prefix_length_stride = 2U;
        require_invalid(
            ninfer::ops::r9700::kv::fp8_int4_kv_attention_fused(raw_tree, device.stream),
            "raw unsupported packed-tree prefix stride");
        raw_tree.prefix_length_stride = 0U;
        raw_tree.prefix_lengths       = nullptr;
        require_invalid(
            ninfer::ops::r9700::kv::fp8_int4_kv_attention_fused(raw_tree, device.stream),
            "raw incomplete packed-tree pair");
        raw_tree.ancestor_masks    = nullptr;
        raw_tree.active_query_rows = static_cast<const std::int32_t*>(d_active_query_rows.data());
        require_invalid(ninfer::ops::r9700::kv::fp8_int4_kv_attention_wmma(raw_tree, device.stream),
                        "WMMA device-active-row route");

        // Device metadata errors are row-local. The first row remains valid; the following rows
        // independently exercise negative prefix, out-of-context position, a 32-token suffix,
        // and an empty selected set.
        constexpr std::size_t kInvalidTreeRows            = 5U;
        const std::vector<float> invalid_tree_queries     = make_queries(kInvalidTreeRows);
        const std::vector<hip_bfloat16> invalid_tree_bf16 = as_bf16(invalid_tree_queries);
        const std::vector<std::int32_t> invalid_tree_positions{253, 254, 257, 256, 0};
        const std::vector<std::int32_t> invalid_tree_prefixes{253, -1, 253, 225, 0};
        const std::vector<std::int32_t> invalid_tree_masks{1, 3, 7, -1, 0};
        d_queries.copy_from_host(invalid_tree_bf16.data(),
                                 invalid_tree_bf16.size() * sizeof(hip_bfloat16));
        d_row_positions.copy_from_host(invalid_tree_positions.data(),
                                       invalid_tree_positions.size() * sizeof(std::int32_t));
        d_prefix_lengths.copy_from_host(invalid_tree_prefixes.data(),
                                        invalid_tree_prefixes.size() * sizeof(std::int32_t));
        d_ancestor_masks.copy_from_host(invalid_tree_masks.data(),
                                        invalid_tree_masks.size() * sizeof(std::int32_t));
        q27::R9700FullAttentionArgs invalid_tree_args{
            .query          = ninfer::Tensor(d_queries.data(), ninfer::DType::BF16,
                                             {static_cast<std::int32_t>(kHeadDim),
                                              static_cast<std::int32_t>(kQueryHeads),
                                              static_cast<std::int32_t>(kInvalidTreeRows)}),
            .cache_read     = committed_read,
            .row_positions  = static_cast<const std::int32_t*>(d_row_positions.data()),
            .ancestor_masks = static_cast<const std::int32_t*>(d_ancestor_masks.data()),
            .prefix_lengths = static_cast<const std::int32_t*>(d_prefix_lengths.data()),
            .output         = ninfer::Tensor(d_output.data(), ninfer::DType::FP32,
                                             {static_cast<std::int32_t>(kHeadDim),
                                              static_cast<std::int32_t>(kQueryHeads),
                                              static_cast<std::int32_t>(kInvalidTreeRows)})};
        HIP_CHECK(q27::r9700_qwen3_8_27b_full_attention(invalid_tree_args, device.stream));
        device.synchronize();
        std::vector<float> invalid_tree_actual(invalid_tree_queries.size());
        d_output.copy_to_host(invalid_tree_actual.data(),
                              invalid_tree_actual.size() * sizeof(float));
        const std::vector<float> first_query(
            invalid_tree_queries.begin(),
            invalid_tree_queries.begin() + static_cast<std::ptrdiff_t>(kQueryHeads * kHeadDim));
        const std::vector<double> valid_tree_row = tree_attention_oracle(
            first_query, std::vector<std::int32_t>{253}, std::vector<std::int32_t>{253},
            std::vector<std::int32_t>{1}, cache);
        const std::size_t tree_row_width = kQueryHeads * kHeadDim;
        for (std::size_t index = 0; index < tree_row_width; ++index) {
            const double absolute =
                std::abs(static_cast<double>(invalid_tree_actual[index]) - valid_tree_row[index]);
            const double relative = absolute / std::max(std::abs(valid_tree_row[index]), 1.0e-6);
            if (absolute > 2.0e-4 && relative > 2.0e-4) {
                fail("valid row adjacent to invalid packed-tree metadata differs from oracle");
            }
        }
        if (!std::all_of(invalid_tree_actual.begin(),
                         invalid_tree_actual.begin() + static_cast<std::ptrdiff_t>(tree_row_width),
                         [](float value) { return std::isfinite(value); }) ||
            !std::all_of(invalid_tree_actual.begin() + static_cast<std::ptrdiff_t>(tree_row_width),
                         invalid_tree_actual.end(),
                         [](float value) { return std::isnan(value); })) {
            fail("invalid packed-tree metadata did not poison exactly its own rows");
        }

        const std::vector<float> rejection_queries     = make_queries(3U);
        const std::vector<hip_bfloat16> rejection_bf16 = as_bf16(rejection_queries);
        d_queries.copy_from_host(rejection_bf16.data(),
                                 rejection_bf16.size() * sizeof(hip_bfloat16));
        const std::vector<std::int32_t> device_positions{static_cast<std::int32_t>(kContext - 1U),
                                                         -1, static_cast<std::int32_t>(kContext)};
        d_row_positions.copy_from_host(device_positions.data(),
                                       device_positions.size() * sizeof(std::int32_t));
        q27::R9700FullAttentionArgs reject_args{
            .query = ninfer::Tensor(
                d_queries.data(), ninfer::DType::BF16,
                {static_cast<std::int32_t>(kHeadDim), static_cast<std::int32_t>(kQueryHeads), 3}),
            .cache_read    = committed_read,
            .row_positions = static_cast<const std::int32_t*>(d_row_positions.data()),
            .workspace     = d_score_workspace.data(),
            .workspace_bytes =
                q27::r9700_full_attention_workspace_capacity_bytes(3U, kContext, false),
            .output = ninfer::Tensor(
                d_output.data(), ninfer::DType::FP32,
                {static_cast<std::int32_t>(kHeadDim), static_cast<std::int32_t>(kQueryHeads), 3})};
        HIP_CHECK(q27::r9700_qwen3_8_27b_full_attention(reject_args, device.stream));
        device.synchronize();
        std::vector<float> rejected_output(rejection_queries.size());
        d_output.copy_to_host(rejected_output.data(), rejected_output.size() * sizeof(float));
        const std::size_t row_width = kQueryHeads * kHeadDim;
        const std::vector<double> valid_row_expected =
            attention_oracle(rejection_queries, 1U, std::vector<std::size_t>{kContext}, cache);
        for (std::size_t index = 0; index < row_width; ++index) {
            const double absolute =
                std::abs(static_cast<double>(rejected_output[index]) - valid_row_expected[index]);
            const double relative =
                absolute / std::max(std::abs(valid_row_expected[index]), 1.0e-6);
            // Packed decode criterion (BF16 Q, FP16 probabilities and V/8).
            if (absolute > 2.0e-3 && relative > 2.0e-3) {
                fail("valid row adjacent to rejected device frontiers differs from FP64 oracle");
            }
        }
        if (!std::all_of(rejected_output.begin(),
                         rejected_output.begin() + static_cast<std::ptrdiff_t>(row_width),
                         [](float value) { return std::isfinite(value); }) ||
            !std::all_of(rejected_output.begin() + static_cast<std::ptrdiff_t>(row_width),
                         rejected_output.end(), [](float value) { return std::isnan(value); })) {
            fail("device causal-frontier rejection did not poison exactly the invalid rows");
        }

        q27::R9700FullAttentionArgs host_reject = reject_args;
        host_reject.query                       = ninfer::Tensor(
            d_queries.data(), ninfer::DType::BF16,
            {static_cast<std::int32_t>(kHeadDim), static_cast<std::int32_t>(kQueryHeads), 1});
        host_reject.output = ninfer::Tensor(
            d_output.data(), ninfer::DType::FP32,
            {static_cast<std::int32_t>(kHeadDim), static_cast<std::int32_t>(kQueryHeads), 1});
        host_reject.row_positions = nullptr;
        host_reject.cache_read    = {};
        require_invalid(q27::r9700_qwen3_8_27b_full_attention(host_reject, device.stream),
                        "empty cache read");
        host_reject.cache_read       = committed_read;
        publication.transaction_open = true;
        require_invalid(q27::r9700_qwen3_8_27b_full_attention(host_reject, device.stream),
                        "open publication");
        publication.transaction_open = false;
        publication.healthy          = false;
        require_invalid(q27::r9700_qwen3_8_27b_full_attention(host_reject, device.stream),
                        "poisoned publication");
        publication.healthy        = true;
        publication.valid_frontier = 0U;
        require_invalid(q27::r9700_qwen3_8_27b_full_attention(host_reject, device.stream),
                        "empty publication");
        publication.valid_frontier = static_cast<std::uint32_t>(kMappedPages * kPageSize + 1U);
        require_invalid(q27::r9700_qwen3_8_27b_full_attention(host_reject, device.stream),
                        "publication beyond mapped pages");
        publication.valid_frontier = static_cast<std::uint32_t>(kContext);

        // Append token 65 and attend it on the same stream before the one-layer transaction
        // commits. The pending authorization must be unavailable before launch, invalid on every
        // other stream, and stale immediately after commit.
        const std::vector<float> all_keys   = make_keys(kContext + 1U);
        const std::vector<float> all_values = make_values(kContext + 1U);
        const std::size_t token_elements    = kKvHeads * kHeadDim;
        std::vector<hip_bfloat16> appended_keys(token_elements);
        std::vector<hip_bfloat16> appended_values(token_elements);
        for (std::size_t index = 0; index < token_elements; ++index) {
            appended_keys[index]   = hip_bfloat16(all_keys[kContext * token_elements + index]);
            appended_values[index] = hip_bfloat16(all_values[kContext * token_elements + index]);
        }
        ninfer::DeviceBuffer d_appended_keys(appended_keys.size() * sizeof(hip_bfloat16));
        ninfer::DeviceBuffer d_appended_values(appended_values.size() * sizeof(hip_bfloat16));
        ninfer::DeviceBuffer d_transaction_positions(sizeof(std::uint32_t));
        ninfer::DeviceBuffer d_transaction_status(sizeof(std::uint32_t));
        ninfer::DeviceBuffer d_transaction_cursor(sizeof(std::uint32_t));
        ninfer::DeviceBuffer d_transaction_base(sizeof(std::int32_t));
        ninfer::DeviceBuffer d_transaction_table_row(sizeof(std::int32_t));
        d_appended_keys.copy_from_host(appended_keys.data(), d_appended_keys.size());
        d_appended_values.copy_from_host(appended_values.data(), d_appended_values.size());
        const std::int32_t appended_position_i32 = static_cast<std::int32_t>(kContext);
        const std::int32_t appended_table_row    = allocation.bound_row();
        d_transaction_positions.copy_from_host(&appended_position_i32,
                                               sizeof(appended_position_i32));
        d_transaction_base.copy_from_host(&appended_position_i32, sizeof(appended_position_i32));
        d_transaction_table_row.copy_from_host(&appended_table_row, sizeof(appended_table_row));
        const q3::PagedKVTransactionWorkspace transaction_workspace{
            .status = static_cast<std::uint32_t*>(d_transaction_status.data()),
            .cursor = static_cast<std::uint32_t*>(d_transaction_cursor.data())};
        auto transaction = typed_cache.begin_device_segmented_append(
            allocation, publication, static_cast<const std::int32_t*>(d_transaction_base.data()),
            static_cast<std::uint32_t>(kContext), static_cast<std::uint32_t>(kContext),
            static_cast<std::uint32_t>(kContext + 1U), transaction_workspace,
            static_cast<const std::int32_t*>(d_transaction_table_row.data()));
        bool rejected_before_launch = false;
        try {
            (void)transaction.pending_layer_read(0);
        } catch (const std::logic_error&) { rejected_before_launch = true; }
        if (!rejected_before_launch) { fail("pending cache read was exposed before append"); }
        transaction.begin_device_segment(
            static_cast<const std::int32_t*>(d_transaction_positions.data()), 1U, nullptr);
        transaction.launch_append_layer(0, static_cast<const hip_bfloat16*>(d_appended_keys.data()),
                                        static_cast<const hip_bfloat16*>(d_appended_values.data()),
                                        device.stream);
        const q3::PagedKVLayerRead pending_read = transaction.pending_layer_read(0);
        if (!pending_read.valid_for(device.stream) || !pending_read.pending() ||
            pending_read.visible_frontier() != kContext + 1U ||
            committed_read.valid_for(device.stream)) {
            fail("pending cache authorization did not replace the committed read");
        }
        transaction.end_device_segment(device.stream);

        const std::vector<float> pending_query       = make_queries(1);
        const std::vector<hip_bfloat16> pending_bf16 = as_bf16(pending_query);
        const std::array<std::int32_t, 1> pending_position{static_cast<std::int32_t>(kContext)};
        d_queries.copy_from_host(pending_bf16.data(), pending_bf16.size() * sizeof(hip_bfloat16));
        d_row_positions.copy_from_host(pending_position.data(), sizeof(std::int32_t));
        q27::R9700FullAttentionArgs pending_args{
            .query = ninfer::Tensor(
                d_queries.data(), ninfer::DType::BF16,
                {static_cast<std::int32_t>(kHeadDim), static_cast<std::int32_t>(kQueryHeads), 1}),
            .cache_read    = pending_read,
            .row_positions = static_cast<const std::int32_t*>(d_row_positions.data()),
            .workspace     = d_score_workspace.data(),
            .workspace_bytes =
                q27::r9700_full_attention_workspace_capacity_bytes(1U, kContext + 1U, false),
            .output = ninfer::Tensor(
                d_output.data(), ninfer::DType::FP32,
                {static_cast<std::int32_t>(kHeadDim), static_cast<std::int32_t>(kQueryHeads), 1})};
        hipStream_t other_stream = nullptr;
        HIP_CHECK(hipStreamCreateWithFlags(&other_stream, hipStreamNonBlocking));
        require_invalid(q27::r9700_qwen3_8_27b_full_attention(pending_args, other_stream),
                        "cross-stream pending read");
        HIP_CHECK(hipStreamDestroy(other_stream));
        HIP_CHECK(q27::r9700_qwen3_8_27b_full_attention(pending_args, device.stream));
        std::uint32_t pending_status = 0;
        std::uint32_t pending_cursor = 0;
        transaction.enqueue_segmented_resolution(&pending_status, &pending_cursor);
        HIP_CHECK(hipStreamSynchronize(device.stream));
        transaction.finish_segmented_resolution(pending_status, pending_cursor,
                                                static_cast<std::uint32_t>(kContext + 1U));
        if (pending_cursor != kContext + 1U || pending_status != 0U ||
            pending_read.valid_for(device.stream) || publication.transaction_open ||
            publication.valid_frontier != kContext + 1U) {
            fail("pending attention did not commit atomically at the new frontier");
        }
        const q3::PagedKVLayerRead committed_after_append =
            typed_cache.published_layer_read(allocation, publication, 0);
        if (!committed_after_append.valid_for(device.stream) || committed_after_append.pending()) {
            fail("committed cache authorization was not restored after append");
        }
        std::vector<float> pending_actual(pending_query.size());
        d_output.copy_to_host(pending_actual.data(), pending_actual.size() * sizeof(float));
        const EncodedCache appended_cache          = encode_cache(all_keys, all_values, page_table);
        const std::vector<double> pending_expected = attention_oracle(
            pending_query, 1U, std::vector<std::size_t>{kContext + 1U}, appended_cache);
        for (std::size_t index = 0; index < pending_actual.size(); ++index) {
            const double absolute =
                std::abs(static_cast<double>(pending_actual[index]) - pending_expected[index]);
            const double relative = absolute / std::max(std::abs(pending_expected[index]), 1.0e-6);
            const double tolerance =
                ninfer::ops::r9700::kv::use_packed_decode_attention(1U, kContext + 1U, false)
                    ? 2.0e-3
                    : 2.0e-4;
            if (absolute > tolerance && relative > tolerance) {
                fail("same-stream pending attention differs from the FP64 oracle");
            }
        }

        std::cout << "r9700_full_attention_leaf: PASS gfx1201 wave32 D256 Hq24 Hkv4 G"
                  << kValueGroup << ' '
                  << "token-K/feature-V/feature-scale T=1..9,17,128 context=257 max_abs="
                  << maximum_absolute << " max_rel=" << maximum_relative
                  << " packed_max_abs=" << packed_maximum_absolute
                  << " packed_max_rel=" << packed_maximum_relative
                  << " whole_op_ms[T1/T2]=" << decode_latency_ms[0] << '/' << decode_latency_ms[1]
                  << " tree_max_abs=" << tree_maximum_absolute
                  << " tree_max_rel=" << tree_maximum_relative
                  << " whole_op_ms[T9/T17/T128]=" << prefill_latency_ms[0] << '/'
                  << prefill_latency_ms[1] << '/' << prefill_latency_ms[2]
                  << " median_whole_op_ms[T4-null/device-full]=" << fixed_count_latency_ms << '/'
                  << device_count_latency_ms
                  << " packed_tree_rowwise_and_broadcast_and_row_local_rejection=PASS"
                  << " device_active_rows_graph_replay[0/partial/full/negative/over]=PASS"
                  << " committed/pending-publication_and_device_frontiers=PASS\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "r9700_full_attention_leaf: FAIL: " << error.what() << '\n';
        return 1;
    }
}
