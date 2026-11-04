#include "qualification_device_guard.h"
#include "core/device.h"
#include "core/decode_graph.h"
#include "ops/r9700/kv/fp8_int4_kv_attention.h"
#include "ops/r9700/kv/r9700_attention_profile.h"
#include "../../tests/ops/fp8_int4_kv_oracle.h"

#include <hip/hip_bfloat16.h>
#include <hip/hip_runtime.h>

#include <nlohmann/json.hpp>

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <random>
#include <span>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

namespace kv     = ninfer::ops::r9700::kv;
namespace oracle = ninfer::test::fp8_int4_kv_oracle;
using Json       = nlohmann::json;

constexpr std::uint32_t kHeadDim      = 256U;
constexpr std::uint32_t kQueryHeads   = 24U;
constexpr std::uint32_t kKvHeads      = 4U;
constexpr std::uint32_t kQueriesPerKv = 6U;
constexpr std::uint32_t kValueGroup   = 16U;
constexpr std::uint32_t kGroups       = kHeadDim / kValueGroup;
constexpr std::uint32_t kPageSize     = 64U;
constexpr std::uint32_t kW1           = 1U;
constexpr std::uint32_t kW5           = 5U;
constexpr std::uint32_t kW6           = 6U;
constexpr std::uint32_t kTextRows     = 129U;
constexpr std::size_t kTextContext    = 129U;
std::size_t kW1Context                = 130U;
std::size_t kW5Context                = 134U;
std::size_t kW6Context                = 135U;
std::size_t kPhysicalPages            = 4U;
std::size_t kPhysicalTokens           = kPhysicalPages * kPageSize;
constexpr float kAttentionScale       = 1.0F / 16.0F;
constexpr std::uint32_t kGuard        = 0x7fc01234U;
constexpr int kTrials                 = 7;
constexpr int kTimingIterations       = 50;
static_assert(kQueryHeads * kTextContext * sizeof(float) == 12384U);

[[noreturn]] void fail(const std::string& message) { throw std::runtime_error(message); }

void hip_check(hipError_t status, const char* operation) {
    if (status != hipSuccess) { fail(std::string(operation) + ": " + hipGetErrorString(status)); }
}

float represented_bf16(float value) {
    std::uint32_t bits = std::bit_cast<std::uint32_t>(value);
    bits += 0x7fffU + ((bits >> 16U) & 1U);
    return std::bit_cast<float>(bits & 0xffff0000U);
}

template <typename T>
class DeviceArray {
public:
    explicit DeviceArray(std::size_t count) : count_(count) {
        hip_check(hipMalloc(&data_, count * sizeof(T)), "hipMalloc");
    }

    ~DeviceArray() {
        if (data_) (void)hipFree(data_);
    }

    DeviceArray(const DeviceArray&)            = delete;
    DeviceArray& operator=(const DeviceArray&) = delete;

    T* get() const { return static_cast<T*>(data_); }

    std::size_t size() const { return count_; }
private:
    void* data_        = nullptr;
    std::size_t count_ = 0;
};

template <typename T>
void upload(DeviceArray<T>& destination, const std::vector<T>& source) {
    if (destination.size() != source.size()) fail("upload extent mismatch");
    hip_check(hipMemcpy(destination.get(), source.data(), source.size() * sizeof(T),
                        hipMemcpyHostToDevice),
              "hipMemcpy H2D");
}

template <typename T>
std::vector<T> download(const DeviceArray<T>& source) {
    std::vector<T> result(source.size());
    hip_check(
        hipMemcpy(result.data(), source.get(), result.size() * sizeof(T), hipMemcpyDeviceToHost),
        "hipMemcpy D2H");
    return result;
}

std::size_t physical_token(std::size_t logical, const std::vector<std::uint32_t>& pages) {
    return static_cast<std::size_t>(pages.at(logical / kPageSize)) * kPageSize +
           logical % kPageSize;
}

std::size_t key_offset(std::size_t token, std::uint32_t head, std::uint32_t feature) {
    return token % kPageSize + kPageSize * (static_cast<std::size_t>(feature) +
                                            kHeadDim * (token / kPageSize + kPhysicalPages * head));
}

std::size_t value_offset(std::size_t token, std::uint32_t head, std::uint32_t feature) {
    return feature / 2U + (kHeadDim / 2U) * (token % kPageSize +
                                             kPageSize * (head + kKvHeads * (token / kPageSize)));
}

std::size_t scale_offset(std::size_t token, std::uint32_t head, std::uint32_t feature) {
    return feature / kValueGroup +
           kGroups * (token % kPageSize + kPageSize * (head + kKvHeads * (token / kPageSize)));
}

struct Fixture {
    std::vector<std::uint32_t> pages{2U, 0U, 3U};
    std::vector<float> queries;
    oracle::Fp8Keys logical_keys;
    oracle::Int4Values logical_values;
    std::vector<std::uint8_t> physical_keys;
    std::vector<std::uint8_t> physical_values;
    std::vector<std::uint16_t> physical_scales;
};

Fixture make_fixture() {
    std::vector<float> query(kW6 * kQueryHeads * kHeadDim);
    for (std::uint32_t row = 0; row < kW6; ++row) {
        for (std::uint32_t head = 0; head < kQueryHeads; ++head) {
            for (std::uint32_t feature = 0; feature < kHeadDim; ++feature) {
                const int value =
                    static_cast<int>((row * 37U + head * 19U + feature * 13U) % 113U) - 56;
                query[(static_cast<std::size_t>(row) * kQueryHeads + head) * kHeadDim + feature] =
                    represented_bf16(static_cast<float>(value) / 32.0F);
            }
        }
    }
    std::vector<float> keys(kW6Context * kKvHeads * kHeadDim);
    std::vector<float> values(keys.size());
    for (std::size_t token = 0; token < kW6Context; ++token) {
        for (std::uint32_t head = 0; head < kKvHeads; ++head) {
            for (std::uint32_t feature = 0; feature < kHeadDim; ++feature) {
                const std::size_t index = (token * kKvHeads + head) * kHeadDim + feature;
                const int key =
                    static_cast<int>((token * 29U + head * 17U + feature * 7U) % 101U) - 50;
                const int value =
                    static_cast<int>((token * 11U + head * 31U + feature * 23U) % 127U) - 63;
                keys[index]   = represented_bf16(static_cast<float>(key) / 16.0F);
                values[index] = represented_bf16(static_cast<float>(value) / 40.0F);
            }
        }
    }
    Fixture fixture{
        .queries      = std::move(query),
        .logical_keys = oracle::encode_keys(keys, kW6Context * kKvHeads, kHeadDim),
        .logical_values =
            oracle::encode_values(values, kW6Context * kKvHeads, kHeadDim, kValueGroup),
        .physical_keys   = std::vector<std::uint8_t>(kPhysicalTokens * kKvHeads * kHeadDim),
        .physical_values = std::vector<std::uint8_t>(kPhysicalTokens * kKvHeads * kHeadDim / 2U),
        .physical_scales = std::vector<std::uint16_t>(kPhysicalTokens * kKvHeads * kGroups)};
    if (kW6Context > 135U) {
        fixture.pages.resize((kW6Context + kPageSize - 1U) / kPageSize);
        // Physical page count is relatively prime to stride17: a nonmonotonic injective
        // logical-page mapping, with spare pages to catch accidental contiguous reads.
        for (std::size_t page = 0; page < fixture.pages.size(); ++page)
            fixture.pages[page] = static_cast<std::uint32_t>((page * 17U + 2U) % kPhysicalPages);
    }
    for (std::size_t logical = 0; logical < kW6Context; ++logical) {
        const std::size_t physical = physical_token(logical, fixture.pages);
        for (std::uint32_t head = 0; head < kKvHeads; ++head) {
            const std::size_t represented = logical * kKvHeads + head;
            for (std::uint32_t feature = 0; feature < kHeadDim; ++feature) {
                fixture.physical_keys[key_offset(physical, head, feature)] =
                    fixture.logical_keys.codes[represented * kHeadDim + feature];
                const std::uint8_t packed =
                    fixture.logical_values
                        .packed_codes[represented * (kHeadDim / 2U) + feature / 2U];
                std::uint8_t& destination =
                    fixture.physical_values[value_offset(physical, head, feature)];
                if ((feature & 1U) == 0U) destination = packed;
                fixture.physical_scales[scale_offset(physical, head, feature)] =
                    fixture.logical_values
                        .fp16_scales[represented * kGroups + feature / kValueGroup];
            }
        }
    }
    return fixture;
}

std::vector<float> make_text_queries(const std::vector<float>& selected_tail) {
    const std::size_t row_elements = kQueryHeads * kHeadDim;
    if (selected_tail.size() < row_elements) fail("selected Text tail query is incomplete");
    std::vector<float> queries(static_cast<std::size_t>(kTextRows) * row_elements);
    for (std::uint32_t row = 0; row + 1U < kTextRows; ++row) {
        for (std::uint32_t head = 0; head < kQueryHeads; ++head) {
            for (std::uint32_t feature = 0; feature < kHeadDim; ++feature) {
                const int value =
                    static_cast<int>((row * 43U + head * 23U + feature * 11U) % 109U) - 54;
                queries[(static_cast<std::size_t>(row) * kQueryHeads + head) * kHeadDim + feature] =
                    represented_bf16(static_cast<float>(value) / 32.0F);
            }
        }
    }
    std::copy_n(selected_tail.begin(), row_elements,
                queries.begin() + static_cast<std::ptrdiff_t>(128U * row_elements));
    return queries;
}

std::vector<double> attention_oracle(const Fixture& fixture,
                                     const std::vector<float>& represented_queries,
                                     std::uint32_t rows,
                                     const std::vector<std::int32_t>& positions) {
    if (represented_queries.size() < static_cast<std::size_t>(rows) * kQueryHeads * kHeadDim) {
        fail("attention oracle query extent is too small");
    }
    std::vector<double> result(static_cast<std::size_t>(rows) * kQueryHeads * kHeadDim);
    for (std::uint32_t row = 0; row < rows; ++row) {
        const std::size_t visible = static_cast<std::size_t>(positions.at(row)) + 1U;
        for (std::uint32_t qh = 0; qh < kQueryHeads; ++qh) {
            const std::uint32_t kvh = qh / kQueriesPerKv;
            oracle::Fp8Keys keys{visible, kHeadDim, std::vector<std::uint8_t>(visible * kHeadDim)};
            oracle::Int4Values values{visible, kHeadDim, kValueGroup,
                                      std::vector<std::uint8_t>(visible * kHeadDim / 2U),
                                      std::vector<std::uint16_t>(visible * kGroups)};
            for (std::size_t token = 0; token < visible; ++token) {
                const std::size_t represented = token * kKvHeads + kvh;
                std::copy_n(fixture.logical_keys.codes.begin() +
                                static_cast<std::ptrdiff_t>(represented * kHeadDim),
                            kHeadDim,
                            keys.codes.begin() + static_cast<std::ptrdiff_t>(token * kHeadDim));
                std::copy_n(fixture.logical_values.packed_codes.begin() +
                                static_cast<std::ptrdiff_t>(represented * kHeadDim / 2U),
                            kHeadDim / 2U,
                            values.packed_codes.begin() +
                                static_cast<std::ptrdiff_t>(token * kHeadDim / 2U));
                std::copy_n(fixture.logical_values.fp16_scales.begin() +
                                static_cast<std::ptrdiff_t>(represented * kGroups),
                            kGroups,
                            values.fp16_scales.begin() +
                                static_cast<std::ptrdiff_t>(token * kGroups));
            }
            const std::size_t base = (static_cast<std::size_t>(row) * kQueryHeads + qh) * kHeadDim;
            const std::vector<double> head = oracle::attention_fp64(
                std::span<const float>(represented_queries.data() + base, kHeadDim), keys, values,
                static_cast<double>(kAttentionScale));
            std::copy(head.begin(), head.end(), result.begin() + static_cast<std::ptrdiff_t>(base));
        }
    }
    return result;
}

struct Error {
    double maximum_absolute            = 0.0;
    double maximum_relative            = 0.0;
    std::size_t maximum_absolute_index = 0U;
    double maximum_absolute_actual     = 0.0;
    double maximum_absolute_expected   = 0.0;
};

Error require_oracle(const std::vector<float>& actual, const std::vector<double>& expected,
                     double limit, const char* label) {
    if (actual.size() != expected.size()) fail(std::string(label) + " extent mismatch");
    Error error;
    std::size_t violating_index = actual.size();
    for (std::size_t index = 0; index < actual.size(); ++index) {
        if (!std::isfinite(actual[index])) fail(std::string(label) + " produced nonfinite output");
        const double absolute = std::abs(static_cast<double>(actual[index]) - expected[index]);
        const double relative = absolute / std::max(std::abs(expected[index]), 1.0e-6);
        if (absolute > error.maximum_absolute) {
            error.maximum_absolute          = absolute;
            error.maximum_absolute_index    = index;
            error.maximum_absolute_actual   = actual[index];
            error.maximum_absolute_expected = expected[index];
        }
        error.maximum_relative = std::max(error.maximum_relative, relative);
        if (violating_index == actual.size() && absolute > limit && relative > limit) {
            violating_index = index;
        }
    }
    if (violating_index != actual.size()) {
        std::ostringstream message;
        message << std::setprecision(std::numeric_limits<double>::max_digits10) << label
                << " exceeds FP64 oracle criterion: max_abs=" << error.maximum_absolute
                << " max_rel=" << error.maximum_relative << " first_violation=" << violating_index
                << " worst_index=" << error.maximum_absolute_index
                << " actual=" << error.maximum_absolute_actual
                << " expected=" << error.maximum_absolute_expected;
        fail(message.str());
    }
    return error;
}

Error require_profile_bound(const std::vector<float>& actual,
                            const std::vector<double>& public_expected,
                            const std::vector<double>& profile_expected,
                            double implementation_limit, const char* label) {
    if (actual.size() != public_expected.size() || actual.size() != profile_expected.size()) {
        fail(std::string(label) + " extent mismatch");
    }
    Error error;
    for (std::size_t index = 0; index < actual.size(); ++index) {
        const double absolute =
            std::abs(static_cast<double>(actual[index]) - public_expected[index]);
        const double relative = absolute / std::max(std::abs(public_expected[index]), 1.0e-6);
        if (absolute > error.maximum_absolute) {
            error.maximum_absolute          = absolute;
            error.maximum_absolute_index    = index;
            error.maximum_absolute_actual   = actual[index];
            error.maximum_absolute_expected = public_expected[index];
        }
        error.maximum_relative = std::max(error.maximum_relative, relative);
        const double represented_profile_delta =
            std::abs(profile_expected[index] - public_expected[index]);
        const double implementation_allowance =
            implementation_limit * std::max(1.0, std::abs(profile_expected[index]));
        if (absolute > represented_profile_delta + implementation_allowance) {
            std::ostringstream message;
            message << std::setprecision(std::numeric_limits<double>::max_digits10) << label
                    << " exceeds BF16-public oracle plus explicit FP8-Q profile bound at index="
                    << index << " actual=" << actual[index]
                    << " public_expected=" << public_expected[index]
                    << " fp8_q_expected=" << profile_expected[index] << " public_abs=" << absolute
                    << " profile_delta=" << represented_profile_delta
                    << " implementation_allowance=" << implementation_allowance;
            fail(message.str());
        }
    }
    return error;
}

void fill_guarded(DeviceArray<float>& buffer, std::size_t payload) {
    std::vector<std::uint32_t> words(payload + 2U, kGuard);
    hip_check(hipMemcpy(buffer.get(), words.data(), words.size() * sizeof(std::uint32_t),
                        hipMemcpyHostToDevice),
              "initialize guarded output");
}

std::vector<float> read_guarded(const DeviceArray<float>& buffer, std::size_t payload,
                                const char* label) {
    const std::vector<float> all = download(buffer);
    if (std::bit_cast<std::uint32_t>(all.front()) != kGuard ||
        std::bit_cast<std::uint32_t>(all.back()) != kGuard) {
        fail(std::string(label) + " overwrote an output canary");
    }
    std::vector<float> result(all.begin() + 1,
                              all.begin() + 1 + static_cast<std::ptrdiff_t>(payload));
    if (std::any_of(result.begin(), result.end(),
                    [](float value) { return std::bit_cast<std::uint32_t>(value) == kGuard; })) {
        fail(std::string(label) + " did not write its complete output");
    }
    return result;
}

void require_guard_canaries(const DeviceArray<float>& buffer, const char* label) {
    const std::vector<float> all = download(buffer);
    if (std::bit_cast<std::uint32_t>(all.front()) != kGuard ||
        std::bit_cast<std::uint32_t>(all.back()) != kGuard) {
        fail(std::string(label) + " overwrote a workspace canary");
    }
}

double time_ms(hipStream_t stream, const auto& launch) {
    hipEvent_t begin = nullptr;
    hipEvent_t end   = nullptr;
    hip_check(hipEventCreate(&begin), "hipEventCreate begin");
    hip_check(hipEventCreate(&end), "hipEventCreate end");
    for (int index = 0; index < 3; ++index) hip_check(launch(), "timing warmup");
    hip_check(hipEventRecord(begin, stream), "hipEventRecord begin");
    for (int index = 0; index < kTimingIterations; ++index) hip_check(launch(), "timing launch");
    hip_check(hipEventRecord(end, stream), "hipEventRecord end");
    hip_check(hipEventSynchronize(end), "hipEventSynchronize");
    float elapsed = 0.0F;
    hip_check(hipEventElapsedTime(&elapsed, begin, end), "hipEventElapsedTime");
    (void)hipEventDestroy(begin);
    (void)hipEventDestroy(end);
    return static_cast<double>(elapsed) / kTimingIterations;
}

double median(std::array<double, kTrials> values) {
    std::sort(values.begin(), values.end());
    return values[kTrials / 2];
}

// Sequence-batched packed decode and append: every sequence of one batched launch must produce the
// exact bytes of its own one-sequence launch from identically poisoned outputs, workspaces and
// cache planes. The pool is shared as in production: disjoint shuffled physical pages per sequence,
// either selected through device table rows of one pool-wide table or passed as direct page tables.
Json sequence_batch_qualification(ninfer::DeviceContext& device) {
    constexpr std::array<std::size_t, 16> kContexts{37U,   600U,  4100U, 33000U, 129U, 2050U,
                                                    8200U, 1000U, 9U,    65U,    320U, 16385U,
                                                    257U,  3071U, 64U,   70000U};
    static_assert(kContexts.size() == kv::kPackedDecodeMaximumSequences &&
                  kContexts.size() == kv::kAppendMaximumSequences);
    constexpr std::size_t kSparePages  = 7U;
    constexpr std::size_t kRowElements = static_cast<std::size_t>(kQueryHeads) * kHeadDim;
    std::array<std::size_t, kContexts.size()> page_counts{};
    std::size_t pool_pages = kSparePages, table_stride = 0U;
    for (std::size_t i = 0; i < kContexts.size(); ++i) {
        page_counts[i] = (kContexts[i] + kPageSize - 1U) / kPageSize;
        pool_pages += page_counts[i];
        table_stride = std::max(table_stride, page_counts[i]);
    }
    const std::size_t pool_tokens = pool_pages * kPageSize;
    std::mt19937 random(0x5eb7a11U);
    std::vector<std::uint32_t> physical(pool_pages);
    for (std::size_t page = 0; page < pool_pages; ++page)
        physical[page] = static_cast<std::uint32_t>(page);
    std::shuffle(physical.begin(), physical.end(), random);
    // Pool-wide table [16][stride]; rows beyond a sequence's pages hold an out-of-pool page.
    std::vector<std::uint32_t> tables(kContexts.size() * table_stride,
                                      static_cast<std::uint32_t>(pool_pages));
    std::size_t next_page = 0U;
    for (std::size_t i = 0; i < kContexts.size(); ++i)
        for (std::size_t page = 0; page < page_counts[i]; ++page)
            tables[i * table_stride + page] = physical[next_page++];
    std::vector<std::uint8_t> keys(pool_tokens * kKvHeads * kHeadDim);
    std::vector<std::uint8_t> values(pool_tokens * kKvHeads * kHeadDim / 2U);
    std::vector<std::uint16_t> scales(pool_tokens * kKvHeads * kGroups);
    for (auto& code : keys) {
        code = static_cast<std::uint8_t>(random());
        if ((code & 0x7fU) == 0x7fU) code = static_cast<std::uint8_t>(code - 1U); // no E4M3FN NaN
    }
    for (auto& code : values) {
        code = static_cast<std::uint8_t>(random());
        if ((code & 0x0fU) == 0x08U) code = static_cast<std::uint8_t>(code & 0xf0U);
        if ((code & 0xf0U) == 0x80U) code = static_cast<std::uint8_t>(code & 0x0fU);
    }
    // Positive binary16 scales in [2^-10, 2^-3).
    for (auto& bits : scales)
        bits = static_cast<std::uint16_t>(((5U + random() % 7U) << 10U) | (random() & 0x3ffU));
    std::uniform_real_distribution<float> uniform(-2.0F, 2.0F);
    std::vector<hip_bfloat16> queries(8U * kRowElements * kContexts.size());
    for (auto& query : queries) query = hip_bfloat16(represented_bf16(uniform(random)));
    DeviceArray<std::uint8_t> d_keys(keys.size()), d_values(values.size());
    DeviceArray<std::uint16_t> d_scales(scales.size());
    DeviceArray<std::uint32_t> d_tables(tables.size());
    DeviceArray<hip_bfloat16> d_queries(queries.size());
    DeviceArray<std::int32_t> d_table_rows(kContexts.size());
    upload(d_keys, keys);
    upload(d_values, values);
    upload(d_scales, scales);
    upload(d_tables, tables);
    upload(d_queries, queries);
    upload(d_table_rows, std::vector<std::int32_t>{0, 1, 2, 3, 4, 5, 6, 7});

    struct Case {
        const char* name;
        std::vector<std::size_t> sequences;
        std::vector<std::uint32_t> rows;
        bool table_rows;
    };

    const std::vector<Case> cases{
        {"b1_w8_table_row", {3U}, {8U}, true},
        {"b2_w8_table_rows", {1U, 3U}, {8U, 8U}, true},
        {"b3_w8_table_rows", {1U, 2U, 3U}, {8U, 8U, 8U}, true},
        {"b4_w8_table_rows", {0U, 1U, 2U, 3U}, {8U, 8U, 8U, 8U}, true},
        {"b4_mixed_rows_direct_tables", {3U, 0U, 2U, 1U}, {8U, 5U, 1U, 7U}, false},
        {"b6_w8_table_rows", {0U, 1U, 2U, 3U, 4U, 5U}, {8U, 8U, 8U, 8U, 8U, 8U}, true},
        {"b8_w8_table_rows",
         {0U, 1U, 2U, 3U, 4U, 5U, 6U, 7U},
         std::vector<std::uint32_t>(8U, 8U),
         true},
        {"b8_mixed_rows_direct_tables",
         {6U, 3U, 0U, 7U, 2U, 5U, 1U, 4U},
         {8U, 5U, 1U, 7U, 4U, 8U, 2U, 6U},
         false},
        {"b12_w1_table_rows",
         {0U, 1U, 2U, 3U, 4U, 5U, 6U, 7U, 8U, 9U, 10U, 11U},
         std::vector<std::uint32_t>(12U, 1U),
         true},
        {"b16_w6_table_rows",
         {0U, 1U, 2U, 3U, 4U, 5U, 6U, 7U, 8U, 9U, 10U, 11U, 12U, 13U, 14U, 15U},
         std::vector<std::uint32_t>(16U, 6U),
         true},
        {"b16_mixed_rows_direct_tables",
         {14U, 6U, 3U, 9U, 0U, 15U, 7U, 12U, 2U, 5U, 11U, 1U, 8U, 13U, 4U, 10U},
         {8U, 5U, 1U, 7U, 4U, 8U, 2U, 6U, 1U, 3U, 6U, 8U, 2U, 1U, 5U, 4U},
         false},
    };
    Json attention_cells = Json::array();
    for (const Case& cell : cases) {
        const std::size_t count = cell.sequences.size();
        std::vector<std::unique_ptr<DeviceArray<float>>> outputs, workspaces;
        std::vector<std::unique_ptr<DeviceArray<std::int32_t>>> positions;
        std::vector<kv::Fp8Int4KvAttentionArgs> args(count);
        std::vector<std::size_t> elements(count), workspace_elements(count);
        for (std::size_t i = 0; i < count; ++i) {
            const std::size_t sequence = cell.sequences[i];
            const std::size_t context  = kContexts[sequence];
            const std::uint32_t rows   = cell.rows[i];
            elements[i]                = rows * kRowElements;
            workspace_elements[i] =
                kv::fp8_int4_kv_attention_packed_decode_workspace_bytes(context, rows) /
                sizeof(float);
            outputs.push_back(std::make_unique<DeviceArray<float>>(elements[i] + 2U));
            workspaces.push_back(std::make_unique<DeviceArray<float>>(workspace_elements[i] + 2U));
            positions.push_back(std::make_unique<DeviceArray<std::int32_t>>(rows));
            std::vector<std::int32_t> host_positions(rows);
            for (std::uint32_t row = 0; row < rows; ++row)
                host_positions[row] = static_cast<std::int32_t>(context - rows + row);
            upload(*positions.back(), host_positions);
            args[i] = kv::Fp8Int4KvAttentionArgs{
                .query            = d_queries.get() + sequence * 8U * kRowElements,
                .key_codes        = d_keys.get(),
                .value_codes      = d_values.get(),
                .value_scale_bits = d_scales.get(),
                .page_table =
                    cell.table_rows ? d_tables.get() : d_tables.get() + sequence * table_stride,
                .page_table_pages      = static_cast<std::uint32_t>(table_stride),
                .page_table_row        = cell.table_rows ? d_table_rows.get() + sequence : nullptr,
                .page_table_row_stride = static_cast<std::uint32_t>(table_stride),
                .page_table_row_count  = static_cast<std::uint32_t>(kContexts.size()),
                .context               = context,
                .physical_tokens       = pool_tokens,
                .query_heads           = kQueryHeads,
                .kv_heads              = kKvHeads,
                .q_to_kv               = kQueriesPerKv,
                .value_group           = kValueGroup,
                .query_rows            = rows,
                .row_positions         = positions.back()->get(),
                .attention_scale       = kAttentionScale,
                .key_layout            = kv::Fp8Int4KvPlaneLayout::TokenFastest,
                .value_layout          = kv::Fp8Int4KvPlaneLayout::FeatureFastest,
                .value_scale_layout    = kv::Fp8Int4KvPlaneLayout::FeatureFastest,
                .score_workspace       = workspaces.back()->get() + 1U,
                .output                = outputs.back()->get() + 1U};
        }
        const auto poison = [&] {
            for (std::size_t i = 0; i < count; ++i) {
                fill_guarded(*outputs[i], elements[i]);
                fill_guarded(*workspaces[i], workspace_elements[i]);
            }
        };
        const auto read_outputs = [&](const char* label) {
            std::vector<std::vector<std::uint32_t>> result;
            for (std::size_t i = 0; i < count; ++i) {
                const auto values = read_guarded(*outputs[i], elements[i], label);
                require_guard_canaries(*workspaces[i], label);
                std::vector<std::uint32_t> bits(values.size());
                std::transform(values.begin(), values.end(), bits.begin(),
                               [](float value) { return std::bit_cast<std::uint32_t>(value); });
                result.push_back(std::move(bits));
            }
            return result;
        };
        const auto serial = [&] {
            for (const auto& sequence : args) {
                if (const hipError_t error =
                        kv::fp8_int4_kv_attention_packed_decode(sequence, device.stream);
                    error != hipSuccess)
                    return error;
            }
            return hipSuccess;
        };
        const auto batched = [&] {
            return kv::fp8_int4_kv_attention_packed_decode(
                std::span<const kv::Fp8Int4KvAttentionArgs>(args), device.stream);
        };
        poison();
        hip_check(serial(), "sequence-batch per-sequence packed decode");
        device.synchronize();
        const auto reference  = read_outputs("sequence-batch per-sequence");
        std::size_t nonfinite = 0U;
        for (const auto& sequence : reference)
            nonfinite += static_cast<std::size_t>(
                std::count_if(sequence.begin(), sequence.end(), [](std::uint32_t bits) {
                    return !std::isfinite(std::bit_cast<float>(bits));
                }));
        if (nonfinite != 0U) fail(std::string(cell.name) + " reference has nonfinite outputs");
        poison();
        hip_check(batched(), "sequence-batch batched packed decode");
        device.synchronize();
        if (read_outputs("sequence-batch batched eager") != reference)
            fail(std::string(cell.name) + " batched eager bytes differ from per-sequence");
        ninfer::DecodeGraphDefinition definition;
        definition.capture(device.stream,
                           [&] { hip_check(batched(), "capture sequence-batch packed decode"); });
        ninfer::DecodeGraphExecutable executable;
        executable.instantiate(definition);
        executable.upload(device.stream);
        poison();
        executable.launch(device.stream);
        hip_check(hipGetLastError(), "sequence-batch graph replay");
        device.synchronize();
        if (read_outputs("sequence-batch batched graph") != reference)
            fail(std::string(cell.name) + " batched graph bytes differ from per-sequence");
        std::array<double, kTrials> serial_ms{}, batched_ms{};
        for (int trial = 0; trial < kTrials; ++trial) {
            if (trial % 2 == 0) {
                serial_ms[trial]  = time_ms(device.stream, serial);
                batched_ms[trial] = time_ms(device.stream, batched);
            } else {
                batched_ms[trial] = time_ms(device.stream, batched);
                serial_ms[trial]  = time_ms(device.stream, serial);
            }
        }
        Json contexts = Json::array();
        for (const std::size_t sequence : cell.sequences) contexts.push_back(kContexts[sequence]);
        attention_cells.push_back({{"name", cell.name},
                                   {"contexts", contexts},
                                   {"rows", cell.rows},
                                   {"device_table_rows", cell.table_rows},
                                   {"eager_bytes_equal", true},
                                   {"graph_bytes_equal", true},
                                   {"serial_median_ms", median(serial_ms)},
                                   {"batched_median_ms", median(batched_ms)}});
    }

    // Append: eight sequences write disjoint suffixes into one poisoned pool; one uses a device
    // token count, one a device table row, and one a gapped position panel (InvalidPosition).
    constexpr std::size_t kAppendRows = 8U;
    std::vector<float> source_values(kContexts.size() * kAppendRows * kKvHeads * kHeadDim);
    std::vector<hip_bfloat16> append_keys(source_values.size()),
        append_values(source_values.size());
    for (std::size_t i = 0; i < source_values.size(); ++i) {
        append_keys[i]   = hip_bfloat16(represented_bf16(uniform(random)));
        append_values[i] = hip_bfloat16(represented_bf16(uniform(random)));
    }
    DeviceArray<hip_bfloat16> d_append_keys(append_keys.size()),
        d_append_values(append_values.size());
    upload(d_append_keys, append_keys);
    upload(d_append_values, append_values);
    std::vector<std::uint32_t> append_positions(kContexts.size() * kAppendRows);
    for (std::size_t i = 0; i < kContexts.size(); ++i)
        for (std::size_t row = 0; row < kAppendRows; ++row)
            append_positions[i * kAppendRows + row] = static_cast<std::uint32_t>(
                kContexts[i] - kAppendRows + row + (i == 2U && row == 5U));
    DeviceArray<std::uint32_t> d_append_positions(append_positions.size());
    upload(d_append_positions, append_positions);
    DeviceArray<std::int32_t> d_token_count(1U);
    upload(d_token_count, std::vector<std::int32_t>{5});
    DeviceArray<std::uint32_t> d_status(kContexts.size());
    std::vector<kv::Fp8Int4KvAppendArgs> append(kContexts.size());
    const std::size_t source_stride = kAppendRows * kKvHeads * kHeadDim;
    for (std::size_t i = 0; i < kContexts.size(); ++i) {
        append[i] = kv::Fp8Int4KvAppendArgs{
            .keys           = d_append_keys.get() + i * source_stride,
            .values         = d_append_values.get() + i * source_stride,
            .positions      = d_append_positions.get() + i * kAppendRows,
            .tokens         = i == 0U ? 3U : kAppendRows,
            .token_count    = i == 1U ? d_token_count.get() : nullptr,
            .first_position = static_cast<std::uint32_t>(kContexts[i] - kAppendRows),
            .kv_heads       = kKvHeads,
            .page_table = (i == 3U || i == 6U) ? d_tables.get() : d_tables.get() + i * table_stride,
            .page_table_pages      = static_cast<std::uint32_t>(table_stride),
            .page_table_row        = (i == 3U || i == 6U) ? d_table_rows.get() + i : nullptr,
            .page_table_row_stride = static_cast<std::uint32_t>(table_stride),
            .page_table_row_count  = static_cast<std::uint32_t>(kContexts.size()),
            .physical_tokens       = pool_tokens,
            .key_codes             = d_keys.get(),
            .value_codes           = d_values.get(),
            .value_scale_bits      = d_scales.get(),
            .status                = d_status.get() + i,
            .value_group           = kValueGroup,
            .key_layout            = kv::Fp8Int4KvPlaneLayout::TokenFastest,
            .value_layout          = kv::Fp8Int4KvPlaneLayout::FeatureFastest,
            .value_scale_layout    = kv::Fp8Int4KvPlaneLayout::FeatureFastest};
    }
    const auto reset_pool = [&] {
        upload(d_keys, keys);
        upload(d_values, values);
        upload(d_scales, scales);
        upload(d_status, std::vector<std::uint32_t>(kContexts.size(), 0U));
    };
    reset_pool();
    for (const auto& sequence : append)
        hip_check(kv::fp8_int4_kv_append(sequence, device.stream), "per-sequence append");
    device.synchronize();
    const auto reference_keys   = download(d_keys);
    const auto reference_values = download(d_values);
    const auto reference_scales = download(d_scales);
    const auto reference_status = download(d_status);
    if (reference_status !=
        std::vector<std::uint32_t>{0U, 0U, kv::Fp8Int4KvAppendInvalidPosition, 0U, 0U, 0U, 0U, 0U})
        fail("per-sequence append status is not the fixture's expected isolation");
    if (reference_keys == keys || reference_values == values || reference_scales == scales)
        fail("per-sequence append wrote no cache bytes");
    reset_pool();
    hip_check(
        kv::fp8_int4_kv_append(std::span<const kv::Fp8Int4KvAppendArgs>(append), device.stream),
        "batched append");
    device.synchronize();
    if (download(d_keys) != reference_keys || download(d_values) != reference_values ||
        download(d_scales) != reference_scales || download(d_status) != reference_status)
        fail("batched append bytes or status differ from per-sequence launches");
    return {{"attention", attention_cells},
            {"attention_criterion", "per-sequence output bytes and output/workspace canaries "
                                    "equal to one-sequence launches, eager and graph replay"},
            {"append",
             {{"sequences", kContexts.size()},
              {"rows", {3, 8, 8, 8, 8, 8, 8, 8}},
              {"device_token_count_sequence", 1},
              {"device_table_row_sequences", {3, 6}},
              {"invalid_position_sequence", 2},
              {"status", reference_status},
              {"plane_bytes_equal", true}}},
            {"context37_note", "below the packed routing minimum; compared at Op level only"},
            {"timing",
             {{"evidence_eligible", false},
              {"diagnostic_only", true},
              {"iterations", kTimingIterations},
              {"trials", kTrials}}}};
}

} // namespace

int main(int argc, char** argv) {
    try {
        if (argc == 2 && std::string_view(argv[1]) == "--sequence-batch") {
            const int selected_device = ninfer::r9700_qualification::device_guard().select();
            ninfer::DeviceContext device(selected_device);
            std::cout << Json{{"sequence_batch", sequence_batch_qualification(device)}}.dump(2)
                      << '\n';
            return 0;
        }
        const bool long_context = argc == 2 && std::string_view(argv[1]) == "--long-context";
        const bool live_cell    = argc == 5 && std::string_view(argv[1]) == "--live-context-cell";
        const bool live_context =
            live_cell || (argc == 2 && std::string_view(argv[1]) == "--live-context");
        if (argc != 1 && !long_context && !live_context)
            fail("usage: dflash_attention_route_discriminator "
                 "[--long-context|--live-context|--live-context-cell ENVELOPE ROWS "
                 "LIVE|--sequence-batch]");
        std::vector<std::size_t> selected_envelopes{8192U, 65536U};
        std::vector<std::uint32_t> selected_widths{1U, 4U, 6U};
        std::vector<std::size_t> selected_live{4097U, 65U, 257U};
        if (live_cell) {
            const auto envelope = std::stoul(argv[2]);
            const auto rows     = std::stoul(argv[3]);
            const auto live     = std::stoul(argv[4]);
            if ((envelope != 8192U && envelope != 65536U) ||
                (rows != 1U && rows != 4U && rows != 6U) || live < rows || live > envelope)
                fail("live-context cell requires envelope8192/65536, rows1/4/6 and "
                     "rows<=live<=envelope");
            selected_envelopes = {envelope};
            selected_widths    = {static_cast<std::uint32_t>(rows)};
            selected_live      = {live};
        }
        if (live_context) {
            kW1Context      = 65531U;
            kW5Context      = 65535U;
            kW6Context      = 65536U;
            kPhysicalPages  = 1027U;
            kPhysicalTokens = kPhysicalPages * kPageSize;
        }
        if (long_context) {
            kW1Context      = 4097U;
            kW5Context      = 4101U;
            kW6Context      = 4102U;
            kPhysicalPages  = 67U;
            kPhysicalTokens = kPhysicalPages * kPageSize;
        }
        const bool text_selected = kv::kTextP129WmmaTailCandidate;
        if (!kv::use_packed_decode_attention(kW5, kW5Context, false) ||
            !kv::use_packed_decode_attention(kW6, kW6Context, false) ||
            kv::use_text_p129_wmma_tail(kTextRows, kTextContext) != text_selected ||
            !kv::use_packed_decode_attention(1U, kW1Context, false) ||
            !kv::use_packed_decode_attention(4U, kW5Context, false) ||
            kv::use_packed_decode_attention(0U, kW5Context, false) ||
            kv::use_packed_decode_attention(9U, kW6Context, false) ||
            kv::use_packed_decode_attention(kW5, kW5Context, true) ||
            kv::use_packed_decode_attention(kW6, kW6Context, true) ||
            kv::use_packed_decode_attention(kW5, 63U, false) ||
            kv::use_packed_decode_attention(kW6, 63U, false) ||
            kv::use_packed_decode_attention(kW5, 262145U, false) ||
            kv::use_packed_decode_attention(kW6, 262145U, false) ||
            kv::use_text_p129_wmma_tail(128U, kTextContext) ||
            kv::use_text_p129_wmma_tail(kTextRows, 130U)) {
            fail("attention parity selector escaped its exact Text/packed cells");
        }
        constexpr const char* split_geometry = "host-envelope-production";
        const int selected_device            = ninfer::r9700_qualification::device_guard().select();
        ninfer::DeviceContext device(selected_device);
        std::array<char, 32> pci{};
        hip_check(hipDeviceGetPCIBusId(pci.data(), static_cast<int>(pci.size()), selected_device),
                  "hipDeviceGetPCIBusId");
        const Fixture fixture                   = make_fixture();
        const std::vector<hip_bfloat16> queries = [&] {
            std::vector<hip_bfloat16> result;
            result.reserve(fixture.queries.size());
            for (float value : fixture.queries) result.emplace_back(value);
            return result;
        }();
        const std::vector<float> text_query_values   = make_text_queries(fixture.queries);
        const std::vector<hip_bfloat16> text_queries = [&] {
            std::vector<hip_bfloat16> result;
            result.reserve(text_query_values.size());
            for (float value : text_query_values) result.emplace_back(value);
            return result;
        }();
        const std::vector<std::uint32_t> pages(fixture.pages.begin(), fixture.pages.end());
        const auto first_position = static_cast<std::int32_t>(kW1Context - 1U);
        const std::vector<std::int32_t> w1_positions{first_position};
        const std::vector<std::int32_t> text_w1_positions{128};
        const std::vector<std::int32_t> w5_positions{first_position, first_position + 1,
                                                     first_position + 2, first_position + 3,
                                                     first_position + 4};
        const std::vector<std::int32_t> w6_positions{first_position,     first_position + 1,
                                                     first_position + 2, first_position + 3,
                                                     first_position + 4, first_position + 5};
        std::vector<std::int32_t> text_positions(kTextRows);
        for (std::uint32_t row = 0; row < kTextRows; ++row) {
            text_positions[row] = static_cast<std::int32_t>(row);
        }

        DeviceArray<hip_bfloat16> d_query(queries.size());
        upload(d_query, queries);
        DeviceArray<hip_bfloat16> d_text_query(text_queries.size());
        upload(d_text_query, text_queries);
        DeviceArray<std::uint8_t> d_keys(fixture.physical_keys.size());
        upload(d_keys, fixture.physical_keys);
        DeviceArray<std::uint8_t> d_values(fixture.physical_values.size());
        upload(d_values, fixture.physical_values);
        DeviceArray<std::uint16_t> d_scales(fixture.physical_scales.size());
        upload(d_scales, fixture.physical_scales);
        DeviceArray<std::uint32_t> d_pages(pages.size());
        upload(d_pages, pages);
        DeviceArray<std::int32_t> d_w1_positions(w1_positions.size());
        upload(d_w1_positions, w1_positions);
        DeviceArray<std::int32_t> d_text_w1_positions(text_w1_positions.size());
        upload(d_text_w1_positions, text_w1_positions);
        DeviceArray<std::int32_t> d_w5_positions(w5_positions.size());
        upload(d_w5_positions, w5_positions);
        DeviceArray<std::int32_t> d_w6_positions(w6_positions.size());
        upload(d_w6_positions, w6_positions);
        DeviceArray<std::int32_t> d_text_positions(text_positions.size());
        upload(d_text_positions, text_positions);
        const std::size_t row_elements = kQueryHeads * kHeadDim;
        DeviceArray<float> d_w1_output(row_elements + 2U);
        DeviceArray<float> d_w1_fused_output(row_elements + 2U);
        DeviceArray<float> d_w5_wmma_output(kW5 * row_elements + 2U);
        DeviceArray<float> d_w5_batched_output(kW5 * row_elements + 2U);
        DeviceArray<float> d_w5_fused_output(kW5 * row_elements + 2U);
        DeviceArray<float> d_w6_wmma_output(kW6 * row_elements + 2U);
        DeviceArray<float> d_w6_batched_output(kW6 * row_elements + 2U);
        DeviceArray<float> d_w6_fused_output(kW6 * row_elements + 2U);
        DeviceArray<float> d_w1_scores(kQueryHeads * kW1Context + 2U);
        DeviceArray<float> d_w5_scores(kQueryHeads * kW5Context + 2U);
        DeviceArray<float> d_w6_scores(kQueryHeads * kW6Context + 2U);
        // Split-context verify partials (bounded scratch; trailing chunk slots may stay unused).
        const std::size_t batched_workspace_elements =
            kv::fp8_int4_kv_attention_packed_decode_workspace_bytes(kW5Context, kW5) /
            sizeof(float);
        DeviceArray<float> d_w5_batched_scores(batched_workspace_elements + 2U);
        const std::size_t w6_batched_workspace_elements =
            kv::fp8_int4_kv_attention_packed_decode_workspace_bytes(kW6Context, kW6) /
            sizeof(float);
        DeviceArray<float> d_w6_batched_scores(w6_batched_workspace_elements + 2U);
        // Fused dense prefill needs no workspace; only the W1 tail candidate owns scores.
        const std::size_t text_candidate_workspace_elements =
            static_cast<std::size_t>(kQueryHeads) * kTextRows;
        DeviceArray<float> d_text_dense_output(static_cast<std::size_t>(kTextRows) * row_elements +
                                               2U);
        DeviceArray<float> d_text_candidate_output(
            static_cast<std::size_t>(kTextRows) * row_elements + 2U);
        DeviceArray<float> d_text_w1_output(row_elements + 2U);
        DeviceArray<float> d_text_candidate_workspace(text_candidate_workspace_elements + 2U);
        DeviceArray<float> d_text_w1_scores(kQueryHeads * kTextContext + 2U);

        const auto args = [&](std::uint32_t rows, std::size_t context,
                              const std::int32_t* positions, float* scores, float* output) {
            return kv::Fp8Int4KvAttentionArgs{
                .query              = d_query.get(),
                .key_codes          = d_keys.get(),
                .value_codes        = d_values.get(),
                .value_scale_bits   = d_scales.get(),
                .page_table         = d_pages.get(),
                .page_table_pages   = static_cast<std::uint32_t>(pages.size()),
                .context            = context,
                .physical_tokens    = kPhysicalTokens,
                .query_heads        = kQueryHeads,
                .kv_heads           = kKvHeads,
                .q_to_kv            = kQueriesPerKv,
                .value_group        = kValueGroup,
                .query_rows         = rows,
                .row_positions      = positions,
                .attention_scale    = kAttentionScale,
                .key_layout         = kv::Fp8Int4KvPlaneLayout::TokenFastest,
                .value_layout       = kv::Fp8Int4KvPlaneLayout::FeatureFastest,
                .value_scale_layout = kv::Fp8Int4KvPlaneLayout::FeatureFastest,
                .score_workspace    = scores,
                .output             = output};
        };
        if (live_context) {
            // The target passes a captured frontier envelope as context while row_positions
            // supply the live causal frontier. Keep one executable and allocation per envelope
            // and width, replaying shorter and then longer contexts to expose stale partials.
            Json cells = Json::array();
            for (const std::size_t envelope : selected_envelopes)
                for (const std::uint32_t rows : selected_widths) {
                    const std::size_t elements = rows * row_elements;
                    const std::size_t workspace_elements =
                        kv::fp8_int4_kv_attention_packed_decode_workspace_bytes(envelope, rows) /
                        sizeof(float);
                    DeviceArray<float> output(elements + 2U), scores(workspace_elements + 2U);
                    DeviceArray<std::int32_t> positions(rows);
                    auto packed =
                        args(rows, envelope, positions.get(), scores.get() + 1U, output.get() + 1U);
                    const auto launch = [&] {
                        return kv::fp8_int4_kv_attention_packed_decode(packed, device.stream);
                    };
                    // Capture performs no reads of the positions payload; initialize it anyway so
                    // this fixture remains valid if capture setup later adds device work.
                    upload(positions, std::vector<std::int32_t>(rows, 0));
                    ninfer::DecodeGraphDefinition definition;
                    definition.capture(device.stream, [&] {
                        hip_check(launch(), "capture live-context packed attention");
                    });
                    ninfer::DecodeGraphExecutable executable;
                    executable.instantiate(definition);
                    executable.upload(device.stream);
                    const auto replay = [&] {
                        executable.launch(device.stream);
                        return hipGetLastError();
                    };
                    // Zero live maximum must still overwrite every fixed-grid partial and
                    // propagate invalid rows. It must not divide by zero or reuse old payload.
                    upload(positions, std::vector<std::int32_t>(rows, -1));
                    fill_guarded(output, elements);
                    fill_guarded(scores, workspace_elements);
                    hip_check(replay(), "live-context all-invalid graph");
                    device.synchronize();
                    const auto invalid = read_guarded(output, elements, "live-context all-invalid");
                    if (!std::all_of(invalid.begin(), invalid.end(),
                                     [](float value) { return std::isnan(value); }))
                        fail("live-context all-invalid rows were not poisoned");
                    require_guard_canaries(scores, "live-context all-invalid workspace");
                    for (const std::size_t live : selected_live) {
                        std::vector<std::int32_t> host_positions(rows);
                        for (std::uint32_t row = 0; row < rows; ++row)
                            host_positions[row] = static_cast<std::int32_t>(live - rows + row);
                        upload(positions, host_positions);
                        const auto expected =
                            attention_oracle(fixture, fixture.queries, rows, host_positions);
                        fill_guarded(output, elements);
                        // Poison every partial before the eager launch. A missing empty-split
                        // write must not be hidden by zero-initialized allocation or prior replay.
                        fill_guarded(scores, workspace_elements);
                        hip_check(launch(), "live-context packed eager");
                        device.synchronize();
                        const auto eager = read_guarded(output, elements, "live-context eager");
                        const auto error = require_oracle(eager, expected, 2.0e-3,
                                                          "live-context public BF16 attention");
                        require_guard_canaries(scores, "live-context eager workspace");
                        fill_guarded(output, elements);
                        fill_guarded(scores, workspace_elements);
                        hip_check(replay(), "live-context graph replay");
                        device.synchronize();
                        const auto graph = read_guarded(output, elements, "live-context graph");
                        if (!std::equal(eager.begin(), eager.end(), graph.begin(),
                                        [](float a, float b) {
                                            return std::bit_cast<std::uint32_t>(a) ==
                                                   std::bit_cast<std::uint32_t>(b);
                                        }))
                            fail("live-context eager/graph mismatch");
                        require_guard_canaries(scores, "live-context graph workspace");
                        // Same represented queries/cache and causal positions, with a tight host
                        // frontier. Different split association need only satisfy the oracle.
                        auto tight              = packed;
                        tight.context           = live;
                        const auto launch_tight = [&] {
                            return kv::fp8_int4_kv_attention_packed_decode(tight, device.stream);
                        };
                        hip_check(launch_tight(), "live-context tight control");
                        device.synchronize();
                        const auto tight_error = require_oracle(
                            read_guarded(output, elements, "live-context tight output"), expected,
                            2.0e-3, "live-context tight oracle");
                        ninfer::r9700_qualification::device_guard().check(
                            "after-oracle-before-timing");
                        std::array<double, kTrials> eager_ms{}, graph_ms{}, tight_ms{};
                        for (int trial = 0; trial < kTrials; ++trial) {
                            if ((trial & 1) == 0) {
                                eager_ms[trial] = time_ms(device.stream, launch);
                                graph_ms[trial] = time_ms(device.stream, replay);
                                tight_ms[trial] = time_ms(device.stream, launch_tight);
                            } else {
                                tight_ms[trial] = time_ms(device.stream, launch_tight);
                                graph_ms[trial] = time_ms(device.stream, replay);
                                eager_ms[trial] = time_ms(device.stream, launch);
                            }
                        }
                        ninfer::r9700_qualification::device_guard().check("after-timing");
                        // Replay after the tight control without resetting the workspace: the
                        // captured operation must overwrite all partials its merge consumes.
                        hip_check(replay(), "live-context graph after tight control");
                        device.synchronize();
                        require_oracle(read_guarded(output, elements, "live-context final replay"),
                                       expected, 2.0e-3, "live-context final replay oracle");
                        require_guard_canaries(scores, "live-context timed workspace");
                        cells.push_back(
                            {{"rows", rows},
                             {"envelope", envelope},
                             {"live_context", live},
                             {"positions", host_positions},
                             {"workspace_bytes", workspace_elements * sizeof(float)},
                             {"oracle_max_absolute", error.maximum_absolute},
                             {"tight_oracle_max_absolute", tight_error.maximum_absolute},
                             {"eager_graph_bitwise_equal", true},
                             {"complete_verify_merge_eager_ms", median(eager_ms)},
                             {"complete_verify_merge_graph_ms", median(graph_ms)},
                             {"complete_verify_merge_tight_eager_ms", median(tight_ms)}});
                    }
                }
            std::cout << Json{{"schema_version", 1},
                              {"qualification", "packed_attention_live_context_envelope"},
                              {"split_geometry", split_geometry},
                              {"all_invalid_zero_maximum", true},
                              {"hardware",
                               {{"device", selected_device},
                                {"name", device.props.name},
                                {"architecture", "gfx1201"},
                                {"pci_bus_id", pci.data()}}},
                              {"oracle",
                               "FP64 from public BF16 Q, FP8 K, signed INT4 V and FP16 scales"},
                              {"criterion", "absolute <= 2e-3 or relative <= 2e-3"},
                              {"timing",
                               {{"scope", "complete verify plus merge"},
                                {"trials", kTrials},
                                {"iterations_per_trial", kTimingIterations},
                                {"warmup_per_trial", 3}}},
                              {"cells", cells}}
                             .dump(2)
                      << '\n';
            return 0;
        }
        // Widths 1..4 (ordinary decode, MTP, short DFlash chains) use the same dynamic-row device
        // instructions as W5/W6. Exercise their public inputs directly, including compact-C
        // page-table selection and distinct causal prefixes.
        Json small_width_cells = Json::array();
        for (const std::uint32_t rows : {1U, 2U, 3U, 4U})
            for (const std::size_t context : {std::size_t{64}, kW1Context + 3U}) {
                const auto elements = rows * row_elements;
                const auto score_elements =
                    kv::fp8_int4_kv_attention_packed_decode_workspace_bytes(context, rows) /
                    sizeof(float);
                DeviceArray<float> output(elements + 2U);
                DeviceArray<float> scores(score_elements + 2U);
                DeviceArray<std::int32_t> positions(rows), selected_row(1U);
                for (std::uint32_t concurrency = 1U; concurrency <= ninfer::kMaximumConcurrency;
                     ++concurrency) {
                    // Other table rows deliberately contain invalid physical pages:
                    // ignoring the device selector cannot pass as an equivalent map.
                    std::vector<std::uint32_t> tables(concurrency * pages.size(),
                                                      static_cast<std::uint32_t>(kPhysicalPages));
                    std::copy(pages.begin(), pages.end(),
                              tables.begin() +
                                  static_cast<std::ptrdiff_t>((concurrency - 1U) * pages.size()));
                    DeviceArray<std::uint32_t> device_tables(tables.size());
                    upload(device_tables, tables);
                    upload(selected_row,
                           std::vector<std::int32_t>{static_cast<std::int32_t>(concurrency - 1U)});
                    std::vector<std::int32_t> host_positions(rows);
                    for (std::uint32_t row = 0; row < rows; ++row)
                        host_positions[row] =
                            static_cast<std::int32_t>(context - rows + row - (concurrency - 1U));
                    upload(positions, host_positions);
                    auto batched =
                        args(rows, context, positions.get(), scores.get() + 1U, output.get() + 1U);
                    batched.page_table            = device_tables.get();
                    batched.page_table_row        = selected_row.get();
                    batched.page_table_row_stride = static_cast<std::uint32_t>(pages.size());
                    batched.page_table_row_count  = concurrency;
                    fill_guarded(output, elements);
                    fill_guarded(scores, score_elements);
                    hip_check(kv::fp8_int4_kv_attention_packed_decode(batched, device.stream),
                              "small-width split verify");
                    device.synchronize();
                    const auto actual =
                        read_guarded(output, elements, "small-width split verify output");
                    const auto public_oracle =
                        attention_oracle(fixture, fixture.queries, rows, host_positions);
                    // Dense-attention criterion (BF16 Q, exact-BF16 K, FP16 probabilities and V/8).
                    constexpr double implementation_limit = 2.0e-3;
                    const auto error = require_oracle(actual, public_oracle, implementation_limit,
                                                      "small-width public BF16 attention");
                    const auto exact = [](const auto& left, const auto& right) {
                        return std::equal(left.begin(), left.end(), right.begin(),
                                          [](float a, float b) {
                                              return std::bit_cast<std::uint32_t>(a) ==
                                                     std::bit_cast<std::uint32_t>(b);
                                          });
                    };
                    ninfer::DecodeGraphDefinition definition;
                    definition.capture(device.stream, [&] {
                        hip_check(kv::fp8_int4_kv_attention_packed_decode(batched, device.stream),
                                  "capture W4 batched WMMA");
                    });
                    ninfer::DecodeGraphExecutable executable;
                    executable.instantiate(definition);
                    executable.upload(device.stream);
                    fill_guarded(output, elements);
                    fill_guarded(scores, score_elements);
                    executable.launch(device.stream);
                    device.synchronize();
                    if (!exact(actual, read_guarded(output, elements, "small-width graph output")))
                        fail("small-width eager/graph mismatch");
                    require_guard_canaries(scores, "small-width split workspace");
                    auto missing            = batched;
                    missing.score_workspace = nullptr;
                    if (kv::fp8_int4_kv_attention_packed_decode(missing, device.stream) !=
                        hipErrorInvalidValue)
                        fail("small-width accepted missing workspace");
                    auto tree           = batched;
                    tree.ancestor_masks = positions.get();
                    if (kv::fp8_int4_kv_attention_packed_decode(tree, device.stream) !=
                        hipErrorInvalidValue)
                        fail("small-width accepted tree metadata");
                    // Device-resident invalid row positions must poison only that
                    // row; the remaining represented causal queries remain exact.
                    host_positions.back() = -1;
                    upload(positions, host_positions);
                    executable.launch(device.stream);
                    device.synchronize();
                    const auto masked = download(output);
                    for (std::size_t i = 0; i < elements; ++i) {
                        if (i >= (rows - 1U) * row_elements) {
                            if (!std::isnan(masked[i + 1U]))
                                fail("small-width invalid row was not poisoned");
                        } else if (std::bit_cast<std::uint32_t>(masked[i + 1U]) !=
                                   std::bit_cast<std::uint32_t>(actual[i]))
                            fail("small-width invalid row changed valid prefix");
                    }
                    require_guard_canaries(output, "small-width masked output");
                    require_guard_canaries(scores, "small-width masked workspace");
                    small_width_cells.push_back({{"rows", rows},
                                                 {"context", context},
                                                 {"concurrency", concurrency},
                                                 {"public_oracle_max_abs", error.maximum_absolute},
                                                 {"graph_exact", true},
                                                 {"device_row_metadata", true},
                                                 {"guards", true}});
                }
            }
        auto w1       = args(kW1, kW1Context, d_w1_positions.get(), d_w1_scores.get() + 1,
                             d_w1_output.get() + 1);
        auto w1_fused = w1;
        w1_fused.score_workspace = nullptr;
        w1_fused.output          = d_w1_fused_output.get() + 1;
        auto w5_wmma  = args(kW5, kW5Context, d_w5_positions.get(), d_w5_scores.get() + 1,
                             d_w5_wmma_output.get() + 1);
        auto w5_fused = w5_wmma;
        w5_fused.score_workspace   = nullptr;
        w5_fused.output            = d_w5_fused_output.get() + 1;
        auto w5_batched            = w5_wmma;
        w5_batched.score_workspace = d_w5_batched_scores.get() + 1;
        w5_batched.output          = d_w5_batched_output.get() + 1;
        auto w6_wmma  = args(kW6, kW6Context, d_w6_positions.get(), d_w6_scores.get() + 1,
                             d_w6_wmma_output.get() + 1);
        auto w6_fused = w6_wmma;
        w6_fused.score_workspace   = nullptr;
        w6_fused.output            = d_w6_fused_output.get() + 1;
        auto w6_batched            = w6_wmma;
        w6_batched.score_workspace = d_w6_batched_scores.get() + 1;
        w6_batched.output          = d_w6_batched_output.get() + 1;
        // Paired QK must poison every represented row (including the odd W5
        // tail) when device page metadata is invalid, without touching guards.
        for (const std::uint32_t rows : {1U, 2U, 3U, 4U, 5U, 6U}) {
            auto bad                = rows == 6U ? w6_batched : w5_batched;
            bad.query_rows          = rows;
            const auto output_count = rows * row_elements;
            const auto score_count =
                kv::fp8_int4_kv_attention_packed_decode_workspace_bytes(bad.context, rows) /
                sizeof(float);
            DeviceArray<float> invalid_output(output_count + 2U), invalid_scores(score_count + 2U);
            DeviceArray<std::int32_t> selected_row(1U);
            DeviceArray<std::uint32_t> invalid_pages(pages.size());
            bad.output              = invalid_output.get() + 1U;
            bad.score_workspace     = invalid_scores.get() + 1U;
            const auto check_poison = [&] {
                fill_guarded(invalid_output, output_count);
                fill_guarded(invalid_scores, score_count);
                hip_check(kv::fp8_int4_kv_attention_packed_decode(bad, device.stream),
                          "invalid paired-QK metadata");
                device.synchronize();
                const auto values =
                    read_guarded(invalid_output, output_count, "invalid metadata output");
                if (!std::all_of(values.begin(), values.end(),
                                 [](float v) { return std::isnan(v); }))
                    fail("paired-QK invalid metadata did not poison every row");
                require_guard_canaries(invalid_scores, "invalid metadata scores");
            };
            bad.page_table_row        = selected_row.get();
            bad.page_table_row_stride = static_cast<std::uint32_t>(pages.size());
            bad.page_table_row_count  = 1U;
            for (const auto invalid_row : {-1, 1}) {
                upload(selected_row, std::vector<std::int32_t>{invalid_row});
                check_poison();
            }
            bad.page_table_row      = nullptr;
            auto malformed_pages    = pages;
            malformed_pages.front() = static_cast<std::uint32_t>(kPhysicalPages);
            upload(invalid_pages, malformed_pages);
            bad.page_table = invalid_pages.get();
            check_poison();
        }
        auto text_dense       = args(kTextRows, kTextContext, d_text_positions.get(), nullptr,
                                     d_text_dense_output.get() + 1);
        text_dense.query      = d_text_query.get();
        auto text_candidate   = text_dense;
        text_candidate.output = d_text_candidate_output.get() + 1;
        auto text_tail        = args(kW1, kTextContext, d_text_w1_positions.get(),
                                     d_text_w1_scores.get() + 1, d_text_w1_output.get() + 1);
        text_tail.query       = d_text_query.get() + 128U * row_elements;

        auto missing_workspace            = w5_wmma;
        missing_workspace.score_workspace = nullptr;
        if (kv::fp8_int4_kv_attention_wmma(missing_workspace, device.stream) !=
            hipErrorInvalidValue)
            fail("WMMA accepted a missing exact score workspace");
        if (kv::fp8_int4_kv_attention_packed_decode(missing_workspace, device.stream) !=
            hipErrorInvalidValue)
            fail("batched W5 WMMA accepted a missing score workspace");
        auto w6_missing_workspace            = w6_wmma;
        w6_missing_workspace.score_workspace = nullptr;
        if (kv::fp8_int4_kv_attention_packed_decode(w6_missing_workspace, device.stream) !=
            hipErrorInvalidValue)
            fail("batched W6 WMMA accepted a missing score workspace");
        auto wrong_width       = w5_batched;
        wrong_width.query_rows = 0U;
        if (kv::fp8_int4_kv_attention_packed_decode(wrong_width, device.stream) !=
            hipErrorInvalidValue)
            fail("packed decode accepted zero rows");
        auto wrong_width_w9       = w6_batched;
        wrong_width_w9.query_rows = 9U;
        if (kv::fp8_int4_kv_attention_packed_decode(wrong_width_w9, device.stream) !=
            hipErrorInvalidValue)
            fail("packed decode accepted W9");
        auto missing_positions          = w5_batched;
        missing_positions.row_positions = nullptr;
        if (kv::fp8_int4_kv_attention_packed_decode(missing_positions, device.stream) !=
            hipErrorInvalidValue)
            fail("batched W5 WMMA accepted missing causal positions");
        auto w6_missing_positions          = w6_batched;
        w6_missing_positions.row_positions = nullptr;
        if (kv::fp8_int4_kv_attention_packed_decode(w6_missing_positions, device.stream) !=
            hipErrorInvalidValue)
            fail("batched W6 WMMA accepted missing causal positions");
        auto tree           = w5_batched;
        tree.ancestor_masks = d_w5_positions.get();
        tree.prefix_lengths = d_w5_positions.get();
        if (kv::fp8_int4_kv_attention_packed_decode(tree, device.stream) != hipErrorInvalidValue)
            fail("batched W5 WMMA accepted tree metadata");
        auto w6_tree           = w6_batched;
        w6_tree.ancestor_masks = d_w6_positions.get();
        w6_tree.prefix_lengths = d_w6_positions.get();
        if (kv::fp8_int4_kv_attention_packed_decode(w6_tree, device.stream) != hipErrorInvalidValue)
            fail("batched W6 WMMA accepted tree metadata");
        auto wrong_group        = w5_batched;
        wrong_group.value_group = 32U;
        if (kv::fp8_int4_kv_attention_packed_decode(wrong_group, device.stream) !=
            hipErrorInvalidValue)
            fail("batched W5 WMMA accepted unqualified G32 values");
        auto w6_wrong_group        = w6_batched;
        w6_wrong_group.value_group = 32U;
        if (kv::fp8_int4_kv_attention_packed_decode(w6_wrong_group, device.stream) !=
            hipErrorInvalidValue)
            fail("batched W6 WMMA accepted unqualified G32 values");
        auto wrong_layout         = w5_batched;
        wrong_layout.value_layout = kv::Fp8Int4KvPlaneLayout::TokenFastest;
        if (kv::fp8_int4_kv_attention_packed_decode(wrong_layout, device.stream) !=
            hipErrorInvalidValue)
            fail("batched W5 WMMA accepted an unqualified cache layout");
        auto w6_wrong_layout         = w6_batched;
        w6_wrong_layout.value_layout = kv::Fp8Int4KvPlaneLayout::TokenFastest;
        if (kv::fp8_int4_kv_attention_packed_decode(w6_wrong_layout, device.stream) !=
            hipErrorInvalidValue)
            fail("batched W6 WMMA accepted an unqualified cache layout");
        auto malformed       = w5_fused;
        malformed.query_rows = 0U;
        if (kv::fp8_int4_kv_attention_fused(malformed, device.stream) != hipErrorInvalidValue) {
            fail("fused route accepted zero rows");
        }

        const auto launch_text_dense = [&](kv::Fp8Int4KvAttentionArgs call) {
            return kv::fp8_int4_kv_attention_dense_prefill(call, device.stream);
        };
        const auto launch_text_candidate = [&] {
            hipError_t status = launch_text_dense(text_candidate);
            if (status != hipSuccess) return status;
            auto tail = text_candidate;
            tail.query += 128U * row_elements;
            tail.query_rows = 1U;
            tail.row_positions += 128U;
            tail.score_workspace = d_text_candidate_workspace.get() + 1;
            tail.output += 128U * row_elements;
            return kv::fp8_int4_kv_attention_wmma(tail, device.stream);
        };

        fill_guarded(d_w1_output, row_elements);
        fill_guarded(d_w1_fused_output, row_elements);
        fill_guarded(d_w5_wmma_output, kW5 * row_elements);
        fill_guarded(d_w5_batched_output, kW5 * row_elements);
        fill_guarded(d_w5_fused_output, kW5 * row_elements);
        fill_guarded(d_w6_wmma_output, kW6 * row_elements);
        fill_guarded(d_w6_batched_output, kW6 * row_elements);
        fill_guarded(d_w6_fused_output, kW6 * row_elements);
        fill_guarded(d_w1_scores, kQueryHeads * kW1Context);
        fill_guarded(d_w5_scores, kQueryHeads * kW5Context);
        fill_guarded(d_w6_scores, kQueryHeads * kW6Context);
        fill_guarded(d_w5_batched_scores, batched_workspace_elements);
        fill_guarded(d_w6_batched_scores, w6_batched_workspace_elements);
        fill_guarded(d_text_dense_output, static_cast<std::size_t>(kTextRows) * row_elements);
        fill_guarded(d_text_candidate_output, static_cast<std::size_t>(kTextRows) * row_elements);
        fill_guarded(d_text_w1_output, row_elements);
        fill_guarded(d_text_candidate_workspace, text_candidate_workspace_elements);
        fill_guarded(d_text_w1_scores, kQueryHeads * kTextContext);
        hip_check(kv::fp8_int4_kv_attention_wmma(w1, device.stream), "W1 WMMA");
        hip_check(kv::fp8_int4_kv_attention_fused(w1_fused, device.stream), "W1 fused");
        hip_check(kv::fp8_int4_kv_attention_wmma(w5_wmma, device.stream), "W5 serial WMMA");
        hip_check(kv::fp8_int4_kv_attention_packed_decode(w5_batched, device.stream),
                  "W5 batched WMMA");
        hip_check(kv::fp8_int4_kv_attention_fused(w5_fused, device.stream), "W5 fused");
        hip_check(kv::fp8_int4_kv_attention_wmma(w6_wmma, device.stream), "W6 serial WMMA");
        hip_check(kv::fp8_int4_kv_attention_packed_decode(w6_batched, device.stream),
                  "W6 batched WMMA");
        hip_check(kv::fp8_int4_kv_attention_fused(w6_fused, device.stream), "W6 fused");
        hip_check(launch_text_dense(text_dense), "Text P129 dense");
        hip_check(launch_text_candidate(), "Text P129 dense plus W1-tail overwrite");
        hip_check(kv::fp8_int4_kv_attention_wmma(text_tail, device.stream),
                  "Text standalone W1 at C129");
        device.synchronize();
        const std::vector<float> actual_w1 = read_guarded(d_w1_output, row_elements, "W1 WMMA");
        const std::vector<float> actual_w1_fused =
            read_guarded(d_w1_fused_output, row_elements, "W1 fused");
        const std::vector<float> actual_w5_wmma =
            read_guarded(d_w5_wmma_output, kW5 * row_elements, "W5 serial WMMA");
        const std::vector<float> actual_w5_batched =
            read_guarded(d_w5_batched_output, kW5 * row_elements, "W5 batched WMMA");
        const std::vector<float> actual_w5_fused =
            read_guarded(d_w5_fused_output, kW5 * row_elements, "W5 fused");
        const std::vector<float> actual_w6_wmma =
            read_guarded(d_w6_wmma_output, kW6 * row_elements, "W6 serial WMMA");
        const std::vector<float> actual_w6_batched =
            read_guarded(d_w6_batched_output, kW6 * row_elements, "W6 batched WMMA");
        const std::vector<float> actual_w6_fused =
            read_guarded(d_w6_fused_output, kW6 * row_elements, "W6 fused");
        const std::vector<float> actual_text_dense =
            read_guarded(d_text_dense_output, static_cast<std::size_t>(kTextRows) * row_elements,
                         "Text P129 dense");
        const std::vector<float> actual_text_candidate = read_guarded(
            d_text_candidate_output, static_cast<std::size_t>(kTextRows) * row_elements,
            "Text P129 dense plus W1-tail overwrite");
        const std::vector<float> actual_text_w1 =
            read_guarded(d_text_w1_output, row_elements, "Text standalone W1 at C129");
        (void)read_guarded(d_w1_scores, kQueryHeads * kW1Context, "W1 score workspace");
        (void)read_guarded(d_w5_scores, kQueryHeads * kW5Context, "W5 score workspace");
        (void)read_guarded(d_w6_scores, kQueryHeads * kW6Context, "W6 score workspace");
        require_guard_canaries(d_w5_batched_scores, "W5 split verify workspace");
        require_guard_canaries(d_w6_batched_scores, "W6 split verify workspace");
        // The dense QK plane intentionally does not materialize masked future-score cells.
        // Its contract is bounded scratch, not a fully written observable output.
        require_guard_canaries(d_text_candidate_workspace, "Text P129 candidate workspace");
        (void)read_guarded(d_text_w1_scores, kQueryHeads * kTextContext, "Text W1 score workspace");

        if (!std::equal(actual_w1.begin(), actual_w1.end(), actual_w5_wmma.begin(),
                        [](float left, float right) {
                            return std::bit_cast<std::uint32_t>(left) ==
                                   std::bit_cast<std::uint32_t>(right);
                        })) {
            fail("W1 and W5 serial-WMMA selected rows are not bit-exact");
        }
        if (!std::equal(actual_w1_fused.begin(), actual_w1_fused.end(), actual_w5_fused.begin(),
                        [](float left, float right) {
                            return std::bit_cast<std::uint32_t>(left) ==
                                   std::bit_cast<std::uint32_t>(right);
                        })) {
            fail("W1 and W5 fused selected rows are not bit-exact");
        }
        if (!std::equal(actual_w1.begin(), actual_w1.end(), actual_w6_wmma.begin(),
                        [](float left, float right) {
                            return std::bit_cast<std::uint32_t>(left) ==
                                   std::bit_cast<std::uint32_t>(right);
                        })) {
            fail("W1 and W6 serial-WMMA selected rows are not bit-exact");
        }
        if (!std::equal(actual_w1_fused.begin(), actual_w1_fused.end(), actual_w6_fused.begin(),
                        [](float left, float right) {
                            return std::bit_cast<std::uint32_t>(left) ==
                                   std::bit_cast<std::uint32_t>(right);
                        })) {
            fail("W1 and W6 fused selected rows are not bit-exact");
        }
        if (!std::equal(actual_text_dense.begin(), actual_text_dense.begin() + 128U * row_elements,
                        actual_text_candidate.begin(), [](float left, float right) {
                            return std::bit_cast<std::uint32_t>(left) ==
                                   std::bit_cast<std::uint32_t>(right);
                        })) {
            fail("Text P129 tail overwrite changed a dense prefix row");
        }
        if (!std::equal(actual_text_w1.begin(), actual_text_w1.end(),
                        actual_text_candidate.begin() + 128U * row_elements,
                        [](float left, float right) {
                            return std::bit_cast<std::uint32_t>(left) ==
                                   std::bit_cast<std::uint32_t>(right);
                        })) {
            fail("Text P129 overwritten tail is not bit-exact to standalone W1/C129");
        }
        std::size_t fused_mismatches       = 0U;
        double fused_pair_maximum_absolute = 0.0;
        for (std::size_t index = 0; index < row_elements; ++index) {
            fused_mismatches += std::bit_cast<std::uint32_t>(actual_w1[index]) !=
                                std::bit_cast<std::uint32_t>(actual_w5_fused[index]);
            fused_pair_maximum_absolute =
                std::max(fused_pair_maximum_absolute,
                         std::abs(static_cast<double>(actual_w1[index]) - actual_w5_fused[index]));
        }
        if (fused_mismatches == 0U) fail("fixture did not expose the fused/WMMA arithmetic split");

        std::vector<float> fp8_profile_queries(fixture.queries.size());
        std::transform(fixture.queries.begin(), fixture.queries.end(), fp8_profile_queries.begin(),
                       [](float value) {
                           return static_cast<float>(
                               oracle::decode_e4m3fn(oracle::encode_e4m3fn(value)));
                       });
        const std::vector<double> oracle_w1 =
            attention_oracle(fixture, fixture.queries, kW1, w1_positions);
        const std::vector<double> oracle_w5 =
            attention_oracle(fixture, fixture.queries, kW5, w5_positions);
        const std::vector<double> oracle_w6 =
            attention_oracle(fixture, fixture.queries, kW6, w6_positions);
        const std::vector<double> fp8_profile_oracle_w1 =
            attention_oracle(fixture, fp8_profile_queries, kW1, w1_positions);
        const std::vector<double> fp8_profile_oracle_w5 =
            attention_oracle(fixture, fp8_profile_queries, kW5, w5_positions);
        const std::vector<double> fp8_profile_oracle_w6 =
            attention_oracle(fixture, fp8_profile_queries, kW6, w6_positions);
        const std::vector<double> oracle_text_dense =
            attention_oracle(fixture, text_query_values, kTextRows, text_positions);
        const std::vector<float> text_tail_query(
            text_query_values.begin() + static_cast<std::ptrdiff_t>(128U * row_elements),
            text_query_values.end());
        std::vector<float> text_tail_fp8_profile(row_elements);
        std::transform(text_tail_query.begin(), text_tail_query.end(),
                       text_tail_fp8_profile.begin(), [](float value) {
                           return static_cast<float>(
                               oracle::decode_e4m3fn(oracle::encode_e4m3fn(value)));
                       });
        const std::vector<double> oracle_text_tail =
            attention_oracle(fixture, text_tail_query, kW1, text_w1_positions);
        const std::vector<double> fp8_profile_oracle_text_tail =
            attention_oracle(fixture, text_tail_fp8_profile, kW1, text_w1_positions);
        constexpr double kImplementationLimit      = 2.0e-4;
        constexpr double kDenseImplementationLimit = 2.0e-3;
        const Error w1_profile_error               = require_oracle(
            actual_w1, fp8_profile_oracle_w1, kImplementationLimit, "W1 WMMA FP8-Q profile");
        const Error w1_error = require_profile_bound(actual_w1, oracle_w1, fp8_profile_oracle_w1,
                                                     kImplementationLimit, "W1 WMMA");
        const Error w1_fused_error =
            require_oracle(actual_w1_fused, oracle_w1, kImplementationLimit, "W1 fused");
        const Error w5_wmma_profile_error =
            require_oracle(actual_w5_wmma, fp8_profile_oracle_w5, kImplementationLimit,
                           "W5 serial WMMA FP8-Q profile");
        const Error w5_wmma_error =
            require_profile_bound(actual_w5_wmma, oracle_w5, fp8_profile_oracle_w5,
                                  kImplementationLimit, "W5 serial WMMA");
        const Error w5_batched_error = require_oracle(
            actual_w5_batched, oracle_w5, kDenseImplementationLimit, "W5 split verify BF16-Q");
        const Error w5_fused_error =
            require_oracle(actual_w5_fused, oracle_w5, kImplementationLimit, "W5 fused");
        const Error w6_wmma_profile_error =
            require_oracle(actual_w6_wmma, fp8_profile_oracle_w6, kImplementationLimit,
                           "W6 serial WMMA FP8-Q profile");
        const Error w6_wmma_error =
            require_profile_bound(actual_w6_wmma, oracle_w6, fp8_profile_oracle_w6,
                                  kImplementationLimit, "W6 serial WMMA");
        const Error w6_batched_error = require_oracle(
            actual_w6_batched, oracle_w6, kDenseImplementationLimit, "W6 split verify BF16-Q");
        const Error w6_fused_error =
            require_oracle(actual_w6_fused, oracle_w6, kImplementationLimit, "W6 fused");
        const Error text_dense_error =
            require_oracle(actual_text_dense, oracle_text_dense, kDenseImplementationLimit,
                           "Text P129 dense BF16-Q");
        const Error text_tail_profile_error =
            require_oracle(actual_text_w1, fp8_profile_oracle_text_tail, kImplementationLimit,
                           "Text W1/C129 FP8-Q profile");
        const Error text_tail_error =
            require_profile_bound(actual_text_w1, oracle_text_tail, fp8_profile_oracle_text_tail,
                                  kImplementationLimit, "Text W1/C129");

        std::size_t text_original_tail_mismatches  = 0U;
        double text_original_tail_maximum_absolute = 0.0;
        for (std::size_t index = 0; index < row_elements; ++index) {
            const float dense     = actual_text_dense[128U * row_elements + index];
            const float canonical = actual_text_w1[index];
            text_original_tail_mismatches +=
                std::bit_cast<std::uint32_t>(dense) != std::bit_cast<std::uint32_t>(canonical);
            text_original_tail_maximum_absolute =
                std::max(text_original_tail_maximum_absolute,
                         std::abs(static_cast<double>(dense) - canonical));
        }
        if (text_original_tail_mismatches == 0U) {
            fail("fixture did not expose the Text dense-tail/W1 arithmetic split");
        }

        const auto launch_wmma = [&] {
            return kv::fp8_int4_kv_attention_wmma(w5_wmma, device.stream);
        };
        const auto launch_fused = [&] {
            return kv::fp8_int4_kv_attention_fused(w5_fused, device.stream);
        };
        const auto launch_batched = [&] {
            return kv::fp8_int4_kv_attention_packed_decode(w5_batched, device.stream);
        };
        const auto launch_w6_wmma = [&] {
            return kv::fp8_int4_kv_attention_wmma(w6_wmma, device.stream);
        };
        const auto launch_w6_fused = [&] {
            return kv::fp8_int4_kv_attention_fused(w6_fused, device.stream);
        };
        const auto launch_w6_batched = [&] {
            return kv::fp8_int4_kv_attention_packed_decode(w6_batched, device.stream);
        };

        fill_guarded(d_w5_batched_output, kW5 * row_elements);
        fill_guarded(d_w5_batched_scores, batched_workspace_elements);
        ninfer::DecodeGraphDefinition graph_definition;
        graph_definition.capture(device.stream,
                                 [&] { hip_check(launch_batched(), "capture W5 batched WMMA"); });
        ninfer::DecodeGraphExecutable graph_executable;
        graph_executable.instantiate(graph_definition);
        graph_executable.upload(device.stream);
        graph_executable.launch(device.stream);
        device.synchronize();
        const std::vector<float> graph_w5_batched =
            read_guarded(d_w5_batched_output, kW5 * row_elements, "graph W5 batched WMMA");
        require_guard_canaries(d_w5_batched_scores, "graph W5 split verify workspace");
        if (!std::equal(actual_w5_batched.begin(), actual_w5_batched.end(),
                        graph_w5_batched.begin(), [](float left, float right) {
                            return std::bit_cast<std::uint32_t>(left) ==
                                   std::bit_cast<std::uint32_t>(right);
                        })) {
            fail("captured W5 split verify differs from eager execution");
        }
        fill_guarded(d_w6_batched_output, kW6 * row_elements);
        fill_guarded(d_w6_batched_scores, w6_batched_workspace_elements);
        ninfer::DecodeGraphDefinition w6_graph_definition;
        w6_graph_definition.capture(
            device.stream, [&] { hip_check(launch_w6_batched(), "capture W6 batched WMMA"); });
        ninfer::DecodeGraphExecutable w6_graph_executable;
        w6_graph_executable.instantiate(w6_graph_definition);
        w6_graph_executable.upload(device.stream);
        w6_graph_executable.launch(device.stream);
        device.synchronize();
        const std::vector<float> graph_w6_batched =
            read_guarded(d_w6_batched_output, kW6 * row_elements, "graph W6 batched WMMA");
        require_guard_canaries(d_w6_batched_scores, "graph W6 split verify workspace");
        if (!std::equal(actual_w6_batched.begin(), actual_w6_batched.end(),
                        graph_w6_batched.begin(), [](float left, float right) {
                            return std::bit_cast<std::uint32_t>(left) ==
                                   std::bit_cast<std::uint32_t>(right);
                        })) {
            fail("captured W6 split verify differs from eager execution");
        }
        fill_guarded(d_text_candidate_output, static_cast<std::size_t>(kTextRows) * row_elements);
        fill_guarded(d_text_candidate_workspace, text_candidate_workspace_elements);
        ninfer::DecodeGraphDefinition text_graph_definition;
        text_graph_definition.capture(device.stream, [&] {
            hip_check(launch_text_candidate(), "capture Text P129 tail candidate");
        });
        ninfer::DecodeGraphExecutable text_graph_executable;
        text_graph_executable.instantiate(text_graph_definition);
        text_graph_executable.upload(device.stream);
        text_graph_executable.launch(device.stream);
        device.synchronize();
        const std::vector<float> graph_text_candidate = read_guarded(
            d_text_candidate_output, static_cast<std::size_t>(kTextRows) * row_elements,
            "graph Text P129 tail candidate");
        require_guard_canaries(d_text_candidate_workspace, "graph Text P129 candidate workspace");
        if (!std::equal(actual_text_candidate.begin(), actual_text_candidate.end(),
                        graph_text_candidate.begin(), [](float left, float right) {
                            return std::bit_cast<std::uint32_t>(left) ==
                                   std::bit_cast<std::uint32_t>(right);
                        })) {
            fail("captured Text P129 tail candidate differs from eager execution");
        }
        ninfer::r9700_qualification::device_guard().check("after-oracle-before-timing");
        std::array<double, kTrials> wmma_samples{};
        std::array<double, kTrials> fused_samples{};
        std::array<double, kTrials> batched_samples{};
        std::array<double, kTrials> w1_wmma_samples{};
        std::array<double, kTrials> w1_fused_samples{};
        std::array<double, kTrials> w6_wmma_samples{};
        std::array<double, kTrials> w6_fused_samples{};
        std::array<double, kTrials> w6_batched_samples{};
        std::array<double, kTrials> text_dense_samples{};
        std::array<double, kTrials> text_candidate_samples{};
        for (int trial = 0; trial < kTrials; ++trial) {
            // Repeat ABBA launch order so neither route owns one thermal/order position.
            if ((trial & 3) == 0 || (trial & 3) == 3) {
                w1_wmma_samples[trial]    = time_ms(device.stream, [&] {
                    return kv::fp8_int4_kv_attention_wmma(w1, device.stream);
                });
                w1_fused_samples[trial]   = time_ms(device.stream, [&] {
                    return kv::fp8_int4_kv_attention_fused(w1_fused, device.stream);
                });
                wmma_samples[trial]       = time_ms(device.stream, launch_wmma);
                batched_samples[trial]    = time_ms(device.stream, launch_batched);
                fused_samples[trial]      = time_ms(device.stream, launch_fused);
                w6_wmma_samples[trial]    = time_ms(device.stream, launch_w6_wmma);
                w6_batched_samples[trial] = time_ms(device.stream, launch_w6_batched);
                w6_fused_samples[trial]   = time_ms(device.stream, launch_w6_fused);
                text_dense_samples[trial] =
                    time_ms(device.stream, [&] { return launch_text_dense(text_dense); });
                text_candidate_samples[trial] = time_ms(device.stream, launch_text_candidate);
            } else {
                text_candidate_samples[trial] = time_ms(device.stream, launch_text_candidate);
                text_dense_samples[trial] =
                    time_ms(device.stream, [&] { return launch_text_dense(text_dense); });
                w6_fused_samples[trial]   = time_ms(device.stream, launch_w6_fused);
                w6_batched_samples[trial] = time_ms(device.stream, launch_w6_batched);
                w6_wmma_samples[trial]    = time_ms(device.stream, launch_w6_wmma);
                fused_samples[trial]      = time_ms(device.stream, launch_fused);
                batched_samples[trial]    = time_ms(device.stream, launch_batched);
                wmma_samples[trial]       = time_ms(device.stream, launch_wmma);
                w1_fused_samples[trial]   = time_ms(device.stream, [&] {
                    return kv::fp8_int4_kv_attention_fused(w1_fused, device.stream);
                });
                w1_wmma_samples[trial]    = time_ms(device.stream, [&] {
                    return kv::fp8_int4_kv_attention_wmma(w1, device.stream);
                });
            }
        }
        ninfer::r9700_qualification::device_guard().check("after-timing");
        const Json report{
            {"schema_version", 3},
            {"qualification", "text_p129_and_dflash_verify_attention_parity_discriminator"},
            {"split_geometry", split_geometry},
            {"small_width_qualification", small_width_cells},
            {"w4_w5_w6_invalid_page_metadata_poison", true},
            {"selector",
             {{"name", "NINFER_R9700_TEXT_P129_WMMA_TAIL_CANDIDATE"},
              {"compiled", kv::kTextP129WmmaTailCandidate},
              {"default", false}}},
            {"hardware",
             {{"device", selected_device},
              {"name", device.props.name},
              {"architecture", "gfx1201"},
              {"wave_size", 32},
              {"pci_bus_id", pci.data()}}},
            {"geometry",
             {{"query_heads", 24},
              {"kv_heads", 4},
              {"head_dim", 256},
              {"value_group", 16},
              {"w1_context", kW1Context},
              {"w5_context", kW5Context},
              {"w6_context", kW6Context},
              {"physical_pages", kPhysicalPages},
              {"logical_page_table", pages},
              {"first_dflash_position", first_position},
              {"w1_score_workspace_bytes", kQueryHeads * kW1Context * sizeof(float)},
              {"w5_serial_score_workspace_bytes", kQueryHeads * kW5Context * sizeof(float)},
              {"w6_serial_score_workspace_bytes", kQueryHeads * kW6Context * sizeof(float)},
              {"w5_batched_score_workspace_bytes", batched_workspace_elements * sizeof(float)},
              {"w6_batched_score_workspace_bytes", w6_batched_workspace_elements * sizeof(float)},
              {"text_rows", 129},
              {"text_context", 129},
              {"text_w1_score_workspace_bytes", 12384},
              {"text_candidate_workspace_bytes",
               text_candidate_workspace_elements * sizeof(float)}}},
            {"target_pair",
             {{"ordinary", long_context ? "W1/C4097/position4096 FP8-Q WMMA"
                                        : "W1/C130/position129 FP8-Q WMMA"},
              {"dflash_candidate", long_context ? "W5/C4101/positions4096..4100 split BF16-Q verify"
                                                : "W5/C134/positions129..133 split BF16-Q verify"},
              {"serial_wmma_same_route_row0_bit_exact", true},
              {"fused_same_route_row0_bit_exact", true},
              {"batched_graph_eager_all_rows_bit_exact", true},
              {"current_production_cross_route_row0_bit_exact", false},
              {"current_production_cross_route_row0_mismatches", fused_mismatches},
              {"current_production_cross_route_row0_max_abs", fused_pair_maximum_absolute}}},
            {"target_pair_w6",
             {{"ordinary", long_context ? "W1/C4097/position4096 FP8-Q WMMA"
                                        : "W1/C130/position129 FP8-Q WMMA"},
              {"dflash_candidate", long_context ? "W6/C4102/positions4096..4101 split BF16-Q verify"
                                                : "W6/C135/positions129..134 split BF16-Q verify"},
              {"serial_wmma_same_route_row0_bit_exact", true},
              {"fused_same_route_row0_bit_exact", true},
              {"batched_graph_eager_all_rows_bit_exact", true}}},
            {"text_pair",
             {{"fresh", "dense T129/C129 plus W1 tail overwrite"},
              {"append", "W1/C129/position128 FP8-Q WMMA"},
              {"dense_prefix_rows0_127_preserved_bit_exact", true},
              {"overwritten_tail_append_w1_bit_exact", true},
              {"candidate_graph_eager_bit_exact", true},
              {"original_dense_tail_append_w1_bit_exact", false},
              {"original_dense_tail_append_w1_mismatches", text_original_tail_mismatches},
              {"original_dense_tail_append_w1_max_abs", text_original_tail_maximum_absolute}}},
            {"oracle",
             {{"w1_wmma_max_abs", w1_error.maximum_absolute},
              {"w1_wmma_max_rel", w1_error.maximum_relative},
              {"w1_wmma_fp8_q_profile_max_abs", w1_profile_error.maximum_absolute},
              {"w1_wmma_fp8_q_profile_max_rel", w1_profile_error.maximum_relative},
              {"w1_fused_max_abs", w1_fused_error.maximum_absolute},
              {"w1_fused_max_rel", w1_fused_error.maximum_relative},
              {"w5_wmma_max_abs", w5_wmma_error.maximum_absolute},
              {"w5_wmma_max_rel", w5_wmma_error.maximum_relative},
              {"w5_wmma_fp8_q_profile_max_abs", w5_wmma_profile_error.maximum_absolute},
              {"w5_wmma_fp8_q_profile_max_rel", w5_wmma_profile_error.maximum_relative},
              {"w5_batched_max_abs", w5_batched_error.maximum_absolute},
              {"w5_batched_max_rel", w5_batched_error.maximum_relative},
              {"w5_fused_max_abs", w5_fused_error.maximum_absolute},
              {"w5_fused_max_rel", w5_fused_error.maximum_relative},
              {"w6_wmma_max_abs", w6_wmma_error.maximum_absolute},
              {"w6_wmma_max_rel", w6_wmma_error.maximum_relative},
              {"w6_wmma_fp8_q_profile_max_abs", w6_wmma_profile_error.maximum_absolute},
              {"w6_wmma_fp8_q_profile_max_rel", w6_wmma_profile_error.maximum_relative},
              {"w6_batched_max_abs", w6_batched_error.maximum_absolute},
              {"w6_batched_max_rel", w6_batched_error.maximum_relative},
              {"w6_fused_max_abs", w6_fused_error.maximum_absolute},
              {"w6_fused_max_rel", w6_fused_error.maximum_relative},
              {"text_dense_bf16_q_max_abs", text_dense_error.maximum_absolute},
              {"text_dense_bf16_q_max_rel", text_dense_error.maximum_relative},
              {"text_tail_w1_max_abs", text_tail_error.maximum_absolute},
              {"text_tail_w1_max_rel", text_tail_error.maximum_relative},
              {"text_tail_w1_fp8_q_profile_max_abs", text_tail_profile_error.maximum_absolute},
              {"text_tail_w1_fp8_q_profile_max_rel", text_tail_profile_error.maximum_relative},
              {"wmma_implementation_limit", kImplementationLimit},
              {"dense_bf16_q_implementation_limit", kDenseImplementationLimit},
              {"wmma_public_criterion", "pointwise_abs<=abs(fp8_q_oracle-bf16_oracle)+"
                                        "2e-4*max(1,abs(fp8_q_oracle))"}}},
            {"timing",
             {{"evidence_eligible", false},
              {"diagnostic_only", true},
              {"iterations", kTimingIterations},
              {"trials", kTrials},
              {"w1_wmma_median_ms", median(w1_wmma_samples)},
              {"w1_fused_median_ms", median(w1_fused_samples)},
              {"w5_serial_wmma_median_ms", median(wmma_samples)},
              {"w5_batched_wmma_median_ms", median(batched_samples)},
              {"w5_fused_median_ms", median(fused_samples)},
              {"w6_serial_wmma_median_ms", median(w6_wmma_samples)},
              {"w6_batched_wmma_median_ms", median(w6_batched_samples)},
              {"w6_fused_median_ms", median(w6_fused_samples)},
              {"text_dense_p129_median_ms", median(text_dense_samples)},
              {"text_dense_plus_tail_median_ms", median(text_candidate_samples)},
              {"text_tail_overwrite_median_overhead_ms",
               median(text_candidate_samples) - median(text_dense_samples)},
              {"w1_wmma_samples_ms", w1_wmma_samples},
              {"w1_fused_samples_ms", w1_fused_samples},
              {"w5_serial_wmma_samples_ms", wmma_samples},
              {"w5_batched_wmma_samples_ms", batched_samples},
              {"w5_fused_samples_ms", fused_samples},
              {"w6_serial_wmma_samples_ms", w6_wmma_samples},
              {"w6_batched_wmma_samples_ms", w6_batched_samples},
              {"w6_fused_samples_ms", w6_fused_samples},
              {"text_dense_p129_samples_ms", text_dense_samples},
              {"text_dense_plus_tail_samples_ms", text_candidate_samples},
              {"launch_order", "ABBA by trial"}}},
            {"default_production_routing_changed", false},
            {"compiled_text_p129_candidate_enabled", kv::kTextP129WmmaTailCandidate},
            {"dflash_verify_route_is_production", true}};
        const std::string serialized = report.dump(2);
        const Json parsed            = Json::parse(serialized);
        if (parsed != report || parsed.at("timing").at("evidence_eligible") != false ||
            parsed.at("dflash_verify_route_is_production") != true ||
            parsed.at("target_pair").at("current_production_cross_route_row0_mismatches") !=
                fused_mismatches ||
            parsed.at("text_pair").at("original_dense_tail_append_w1_mismatches") !=
                text_original_tail_mismatches) {
            fail("serialized report failed its exact JSON schema/round-trip contract");
        }
        std::cout << serialized << '\n';
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "dflash_attention_route_discriminator: " << error.what() << '\n';
        return 1;
    }
}
