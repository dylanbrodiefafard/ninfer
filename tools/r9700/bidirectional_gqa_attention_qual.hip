#include "core/arena.h"
#include "core/device.h"
#include "ninfer/ops/bidirectional_gqa_attention.h"
#include "ninfer/types.h"

#include <hip/hip_bfloat16.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <vector>

namespace {

constexpr int kD             = 128;
constexpr int kQh            = 32;
constexpr int kKh            = 8;
constexpr int kGroup         = 4;
constexpr int kPage          = 64;
constexpr int kLogicalPages  = 2;
constexpr int kPhysicalPages = 4;
constexpr int kTableRows     = 2;
constexpr float kScale       = 0.08838834764831844055F;

std::size_t q_index(int d, int head, int token, int batch, int tokens) {
    return static_cast<std::size_t>(d) +
           static_cast<std::size_t>(kD) *
               (static_cast<std::size_t>(head) +
                static_cast<std::size_t>(kQh) *
                    (static_cast<std::size_t>(token) + static_cast<std::size_t>(tokens) * batch));
}

std::size_t kv_index(int d, int head, int token, int batch, int tokens) {
    return static_cast<std::size_t>(d) +
           static_cast<std::size_t>(kD) *
               (static_cast<std::size_t>(head) +
                static_cast<std::size_t>(kKh) *
                    (static_cast<std::size_t>(token) + static_cast<std::size_t>(tokens) * batch));
}

std::size_t page_index(int d, int offset, int page, int head) {
    return static_cast<std::size_t>(d) +
           static_cast<std::size_t>(kD) * (static_cast<std::size_t>(offset) +
                                           static_cast<std::size_t>(kPage) *
                                               (static_cast<std::size_t>(page) +
                                                static_cast<std::size_t>(kPhysicalPages) * head));
}

hip_bfloat16 pattern(std::size_t index, float phase, float amplitude) {
    return hip_bfloat16(amplitude * std::sin(static_cast<float>(index) * 0.0137F + phase));
}

template <class T>
ninfer::DeviceBuffer upload(const std::vector<T>& host) {
    ninfer::DeviceBuffer buffer(host.size() * sizeof(T));
    buffer.copy_from_host(host.data(), buffer.size());
    return buffer;
}

struct Result {
    double max_abs       = 0.0;
    std::size_t compared = 0;
    double median_ms     = 0.0;
};

Result run_case(ninfer::DeviceContext& device, int tokens, const std::vector<std::int32_t>& lengths,
                const std::vector<std::int32_t>& valid, const std::vector<std::int32_t>& table_rows,
                ninfer::ops::GqaContextExecutionEnvelope envelope) {
    const int batch = static_cast<int>(lengths.size());
    if (batch != static_cast<int>(valid.size()) || batch != static_cast<int>(table_rows.size())) {
        throw std::logic_error("qualifier batch vectors disagree");
    }
    const std::size_t q_elements    = static_cast<std::size_t>(kD) * kQh * tokens * batch;
    const std::size_t kv_elements   = static_cast<std::size_t>(kD) * kKh * tokens * batch;
    const std::size_t page_elements = static_cast<std::size_t>(kD) * kPage * kPhysicalPages * kKh;
    std::vector<hip_bfloat16> q(q_elements);
    std::vector<hip_bfloat16> query_k(kv_elements);
    std::vector<hip_bfloat16> query_v(kv_elements);
    std::vector<hip_bfloat16> context_k(page_elements, hip_bfloat16(-0.75F));
    std::vector<hip_bfloat16> context_v(page_elements, hip_bfloat16(0.625F));
    for (std::size_t i = 0; i < q.size(); ++i) { q[i] = pattern(i, 0.17F, 0.31F); }
    for (std::size_t i = 0; i < query_k.size(); ++i) {
        query_k[i] = pattern(i, 0.73F, 0.29F);
        query_v[i] = pattern(i, 1.31F, 0.67F);
    }

    // Dimension zero is fastest. Each table row deliberately maps to disjoint, non-monotonic
    // physical pages so the oracle catches both table-row and page-address mistakes.
    const std::vector<std::int32_t> block_tables{2, 0, 3, 1};
    for (int batch_row = 0; batch_row < kTableRows; ++batch_row) {
        for (int position = 0; position < kLogicalPages * kPage; ++position) {
            const int logical_page = position / kPage;
            const int offset       = position % kPage;
            const int physical_page =
                block_tables[static_cast<std::size_t>(logical_page) +
                             static_cast<std::size_t>(kLogicalPages) * batch_row];
            for (int head = 0; head < kKh; ++head) {
                for (int d = 0; d < kD; ++d) {
                    const std::size_t semantic =
                        static_cast<std::size_t>(d) +
                        static_cast<std::size_t>(kD) *
                            (static_cast<std::size_t>(position) +
                             static_cast<std::size_t>(kLogicalPages * kPage) *
                                 (static_cast<std::size_t>(head) +
                                  static_cast<std::size_t>(kKh) * batch_row));
                    const std::size_t physical = page_index(d, offset, physical_page, head);
                    context_k[physical]        = pattern(semantic, 1.91F, 0.27F);
                    context_v[physical]        = pattern(semantic, 2.47F, 0.71F);
                }
            }
        }
    }

    // Materialized pages can contain unwritten tails. Keep only the union of the live
    // contexts finite; poison everything else, including inactive query columns.
    const hip_bfloat16 poison(std::numeric_limits<float>::quiet_NaN());
    for (int table_row = 0; table_row < kTableRows; ++table_row) {
        int live_length = 0;
        for (int b = 0; b < batch; ++b)
            if (table_rows[b] == table_row && valid[b] > 0)
                live_length = std::max(live_length, lengths[b]);
        for (int position = live_length; position < kLogicalPages * kPage; ++position) {
            const int physical_page =
                block_tables[static_cast<std::size_t>(position / kPage) +
                             static_cast<std::size_t>(kLogicalPages) * table_row];
            for (int head = 0; head < kKh; ++head)
                for (int d = 0; d < kD; ++d) {
                    const auto index = page_index(d, position % kPage, physical_page, head);
                    context_k[index] = context_v[index] = poison;
                }
        }
    }
    for (int b = 0; b < batch; ++b)
        for (int token = valid[b]; token < tokens; ++token) {
            for (int head = 0; head < kQh; ++head)
                for (int d = 0; d < kD; ++d) q[q_index(d, head, token, b, tokens)] = poison;
            for (int head = 0; head < kKh; ++head)
                for (int d = 0; d < kD; ++d) {
                    const auto index = kv_index(d, head, token, b, tokens);
                    query_k[index] = query_v[index] = poison;
                }
        }

    auto d_q          = upload(q);
    auto d_query_k    = upload(query_k);
    auto d_query_v    = upload(query_v);
    auto d_context_k  = upload(context_k);
    auto d_context_v  = upload(context_v);
    auto d_tables     = upload(block_tables);
    auto d_lengths    = upload(lengths);
    auto d_valid      = upload(valid);
    auto d_table_rows = upload(table_rows);
    ninfer::DeviceBuffer d_output(q_elements * sizeof(hip_bfloat16));
    d_output.fill(0x7f);
    ninfer::WorkspaceArena workspace(1);

    ninfer::Tensor q_tensor(d_q.data(), ninfer::DType::BF16, {kD, kQh, tokens, batch});
    ninfer::Tensor query_k_tensor(d_query_k.data(), ninfer::DType::BF16, {kD, kKh, tokens, batch});
    ninfer::Tensor query_v_tensor(d_query_v.data(), ninfer::DType::BF16, {kD, kKh, tokens, batch});
    ninfer::Tensor lengths_tensor(d_lengths.data(), ninfer::DType::I32, {batch});
    ninfer::Tensor valid_tensor(d_valid.data(), ninfer::DType::I32, {batch});
    ninfer::Tensor table_rows_tensor(d_table_rows.data(), ninfer::DType::I32, {batch});
    ninfer::Tensor output_tensor(d_output.data(), ninfer::DType::BF16, {kD, kQh, tokens, batch});
    const ninfer::ops::BidirectionalGqaBF16ContextView context{
        .key_pages   = ninfer::Tensor(d_context_k.data(), ninfer::DType::BF16,
                                      {kD, kPage, kPhysicalPages, kKh}),
        .value_pages = ninfer::Tensor(d_context_v.data(), ninfer::DType::BF16,
                                      {kD, kPage, kPhysicalPages, kKh}),
        .block_tables =
            ninfer::Tensor(d_tables.data(), ninfer::DType::I32, {kLogicalPages, kTableRows}),
        .head_dim     = kD,
        .num_kv_heads = kKh,
    };
    ninfer::ops::bidirectional_gqa_attention(
        q_tensor, query_k_tensor, query_v_tensor, lengths_tensor, valid_tensor, table_rows_tensor,
        kScale, context, envelope, workspace, output_tensor, device.stream);
    device.synchronize();

    // Unprofiled event timing at this case's extent. Correctness is checked against the oracle
    // above.
    Result result;
    const int iters = 200;
    device.synchronize();
    hipEvent_t ev0, ev1;
    HIP_CHECK(hipEventCreate(&ev0));
    HIP_CHECK(hipEventCreate(&ev1));
    for (int i = 0; i < 8; ++i) {
        ninfer::ops::bidirectional_gqa_attention(
            q_tensor, query_k_tensor, query_v_tensor, lengths_tensor, valid_tensor,
            table_rows_tensor, kScale, context, envelope, workspace, output_tensor, device.stream);
    }
    device.synchronize();
    HIP_CHECK(hipEventRecord(ev0, device.stream));
    for (int i = 0; i < iters; ++i) {
        ninfer::ops::bidirectional_gqa_attention(
            q_tensor, query_k_tensor, query_v_tensor, lengths_tensor, valid_tensor,
            table_rows_tensor, kScale, context, envelope, workspace, output_tensor, device.stream);
    }
    HIP_CHECK(hipEventRecord(ev1, device.stream));
    device.synchronize();
    float ms = 0.0F;
    HIP_CHECK(hipEventElapsedTime(&ms, ev0, ev1));
    HIP_CHECK(hipEventDestroy(ev0));
    HIP_CHECK(hipEventDestroy(ev1));
    result.median_ms = ms / static_cast<double>(iters);

    std::vector<hip_bfloat16> output(q_elements);
    d_output.copy_to_host(output.data(), d_output.size());
    for (int b = 0; b < batch; ++b) {
        for (int token = 0; token < tokens; ++token) {
            for (int q_head = 0; q_head < kQh; ++q_head) {
                const int kv_head = q_head / kGroup;
                if (token >= valid[b]) {
                    for (int d = 0; d < kD; ++d) {
                        if (static_cast<float>(output[q_index(d, q_head, token, b, tokens)]) !=
                            0.0F) {
                            throw std::runtime_error("inactive output tail was not exactly zero");
                        }
                    }
                    continue;
                }
                const int count = lengths[b] + valid[b];
                std::vector<double> scores(static_cast<std::size_t>(count));
                double maximum = -INFINITY;
                for (int key = 0; key < count; ++key) {
                    double dot            = 0.0;
                    const bool in_context = key < lengths[b];
                    const int position    = in_context ? key : key - lengths[b];
                    for (int d = 0; d < kD; ++d) {
                        const double qv =
                            static_cast<float>(q[q_index(d, q_head, token, b, tokens)]);
                        double kv;
                        if (in_context) {
                            const int logical_page = position / kPage;
                            const int physical_page =
                                block_tables[static_cast<std::size_t>(logical_page) +
                                             static_cast<std::size_t>(kLogicalPages) *
                                                 table_rows[b]];
                            kv = static_cast<float>(
                                context_k[page_index(d, position % kPage, physical_page, kv_head)]);
                        } else {
                            kv = static_cast<float>(
                                query_k[kv_index(d, kv_head, position, b, tokens)]);
                        }
                        dot += qv * kv;
                    }
                    scores[static_cast<std::size_t>(key)] = dot * static_cast<double>(kScale);
                    maximum = std::max(maximum, scores[static_cast<std::size_t>(key)]);
                }
                double denominator = 0.0;
                for (double score : scores) { denominator += std::exp(score - maximum); }
                for (int d = 0; d < kD; ++d) {
                    double numerator = 0.0;
                    for (int key = 0; key < count; ++key) {
                        const bool in_context = key < lengths[b];
                        const int position    = in_context ? key : key - lengths[b];
                        double vv;
                        if (in_context) {
                            const int logical_page = position / kPage;
                            const int physical_page =
                                block_tables[static_cast<std::size_t>(logical_page) +
                                             static_cast<std::size_t>(kLogicalPages) *
                                                 table_rows[b]];
                            vv = static_cast<float>(
                                context_v[page_index(d, position % kPage, physical_page, kv_head)]);
                        } else {
                            vv = static_cast<float>(
                                query_v[kv_index(d, kv_head, position, b, tokens)]);
                        }
                        numerator += std::exp(scores[static_cast<std::size_t>(key)] - maximum) * vv;
                    }
                    const double expected = numerator / denominator;
                    const double got =
                        static_cast<float>(output[q_index(d, q_head, token, b, tokens)]);
                    const double error = std::abs(got - expected);
                    result.max_abs     = std::max(result.max_abs, error);
                    ++result.compared;
                    if (!std::isfinite(got) || !std::isfinite(expected) ||
                        error > 0.0045 + 0.004 * std::abs(expected)) {
                        throw std::runtime_error("FP64 attention oracle mismatch");
                    }
                }
            }
        }
    }
    return result;
}

} // namespace

int main() {
    try {
        ninfer::DeviceContext device;
        if (ninfer::ops::bidirectional_gqa_attention_workspace_capacity_bytes(
                {0, 65}, 1, 16, static_cast<std::int32_t>(ninfer::kDFlashMaximumConcurrency)) !=
            0) {
            throw std::runtime_error("direct HIP route unexpectedly requested workspace");
        }
        const Result chain = run_case(device, 5, {0, 1}, {5, 3}, {0, 1}, {0, 1});
        const Result tree  = run_case(device, 12, {65, 64}, {12, 7}, {1, 0}, {0, 65});
        std::cout << "bidirectional_gqa_attention: PASS gfx1201 D128/Hq32/Hkv8 "
                     "T=5/12 B=2 fragmented/mixed poisoned-unused max_abs="
                  << std::max(chain.max_abs, tree.max_abs)
                  << " compared=" << chain.compared + tree.compared
                  << " median_ms(chain T5)=" << chain.median_ms
                  << " median_ms(tree T12)=" << tree.median_ms << '\n';
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "bidirectional_gqa_attention: FAIL: " << error.what() << '\n';
        return 1;
    }
}
