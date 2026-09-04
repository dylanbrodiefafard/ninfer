#include "ninfer/ops/qsa.h"
#include "ninfer_bench_common.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

using namespace ninfer;
using namespace ninfer::bench;

namespace {

struct State {
    explicit State(int capacity)
        : capacity(capacity), k_codes(static_cast<std::size_t>(128) * capacity * 2),
          v_codes(static_cast<std::size_t>(128) * capacity * 2),
          k_scales(static_cast<std::size_t>(16) * capacity * 2),
          v_scales(static_cast<std::size_t>(16) * capacity * 2),
          raw_keys(make_bf16(static_cast<std::size_t>(128) * capacity)),
          positions(static_cast<std::size_t>(3) * capacity * sizeof(std::int32_t)) {
        k_codes.fill(0x11);
        v_codes.fill(0x21);
        k_scales.fill(0x38);
        v_scales.fill(0x38);
        positions.fill(0);
    }

    ops::QsaStateView view() {
        return {
            Tensor(k_codes.p, DType::U8, {128, capacity, 2}),
            Tensor(v_codes.p, DType::U8, {128, capacity, 2}),
            Tensor(k_scales.p, DType::FP8_E4M3FN, {16, capacity, 2}),
            Tensor(v_scales.p, DType::FP8_E4M3FN, {16, capacity, 2}),
            Tensor(raw_keys.p, DType::BF16, {128, capacity}),
            Tensor(positions.p, DType::I32, {3, capacity}),
        };
    }

    int capacity;
    DeviceBuffer k_codes;
    DeviceBuffer v_codes;
    DeviceBuffer k_scales;
    DeviceBuffer v_scales;
    DeviceBuffer raw_keys;
    DeviceBuffer positions;
};

DeviceBuffer copy_i32(const std::vector<std::int32_t>& values) {
    DeviceBuffer out(values.size() * sizeof(std::int32_t));
    out.copy_from_host(values.data(), out.bytes);
    return out;
}

DeviceBuffer copy_f32(const std::vector<float>& values) {
    DeviceBuffer out(values.size() * sizeof(float));
    out.copy_from_host(values.data(), out.bytes);
    return out;
}

void run(int frontier) {
    constexpr int capacity = ops::kQsaMaximumTokens;
    State state(capacity);
    auto state_view = state.view();

    DeviceBuffer raw_query = make_bf16(128U * 4U);
    DeviceBuffer query = make_bf16(256U * 24U);
    std::vector<std::int32_t> visible(static_cast<std::size_t>(frontier));
    for (int i = 0; i < frontier; ++i) { visible[static_cast<std::size_t>(i)] = i; }
    DeviceBuffer query_ids = copy_i32({frontier - 1});
    DeviceBuffer visible_ids = copy_i32(visible);
    DeviceBuffer visible_offsets = copy_i32({0, frontier});
    DeviceBuffer query_norm = copy_f32(std::vector<float>(128, 1.0F));
    DeviceBuffer key_norm = copy_f32(std::vector<float>(128, 1.0F));
    DeviceBuffer selected(static_cast<std::size_t>(ops::kQsaSelectedCapacity) * sizeof(std::int32_t));
    DeviceBuffer selected_count(sizeof(std::int32_t));
    DeviceBuffer selector_workspace(ops::qsa_index_select_workspace_bytes(1));
    DeviceBuffer attention_workspace(ops::qsa_selected_attention_workspace_bytes());
    DeviceBuffer out(static_cast<std::size_t>(256) * 24U * sizeof(std::uint16_t));

    Tensor raw_query_t(raw_query.p, DType::BF16, {128, 4, 1});
    Tensor query_t(query.p, DType::BF16, {256, 24, 1});
    Tensor query_ids_t(query_ids.p, DType::I32, {1});
    Tensor visible_ids_t(visible_ids.p, DType::I32, {frontier});
    Tensor visible_offsets_t(visible_offsets.p, DType::I32, {2});
    Tensor query_norm_t(query_norm.p, DType::FP32, {128});
    Tensor key_norm_t(key_norm.p, DType::FP32, {128});
    Tensor selected_t(selected.p, DType::I32, {ops::kQsaSelectedCapacity, 1});
    Tensor attention_selected_t(
        selected.p, DType::I32,
        {frontier < ops::kQsaSelectedCapacity ? frontier : ops::kQsaSelectedCapacity, 1});
    Tensor selected_count_t(selected_count.p, DType::I32, {1});
    Tensor selector_workspace_t(selector_workspace.p, DType::U8,
                                {static_cast<int>(selector_workspace.bytes)});
    Tensor attention_workspace_t(attention_workspace.p, DType::U8,
                                 {static_cast<int>(attention_workspace.bytes)});
    Tensor out_t(out.p, DType::BF16, {256, 24, 1});

    cudaStream_t stream = nullptr;
    CUDA_CHECK(cudaStreamCreate(&stream));
    const auto select = [&](cudaStream_t s) {
        ops::qsa_index_select(raw_query_t, state_view, query_ids_t, visible_ids_t,
                              visible_offsets_t, query_norm_t, key_norm_t, selected_t,
                              selected_count_t, selector_workspace_t, s);
    };
    const ColdTiming select_timing = measure_launch(select, stream, 4, 24);
    select(stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));

    const auto attention = [&](cudaStream_t s) {
        ops::qsa_selected_attention(query_t, attention_selected_t, selected_count_t, state_view,
                                    out_t, attention_workspace_t, s);
    };
    const ColdTiming attention_timing = measure_launch(attention, stream, 4, 24);
    CUDA_CHECK(cudaStreamDestroy(stream));

    std::int32_t host_count = 0;
    CUDA_CHECK(cudaMemcpy(&host_count, selected_count.p, sizeof(host_count), cudaMemcpyDeviceToHost));
    std::printf("frontier=%d selected=%d selector_median_us=%.3f selector_p95_us=%.3f "
                "attention_median_us=%.3f attention_p95_us=%.3f\n",
                frontier, host_count, select_timing.median_us, select_timing.p95_us,
                attention_timing.median_us, attention_timing.p95_us);
}

} // namespace

int main(int argc, char** argv) {
    int device_count = 0;
    if (cudaGetDeviceCount(&device_count) != cudaSuccess || device_count == 0) {
        std::printf("SKIP: no usable CUDA device\n");
        return 0;
    }
    print_device_caps("qwen4-qsa");
    if (argc == 1) {
        for (int frontier : {1, 4, 64, 256, 2048, 4096}) { run(frontier); }
        return 0;
    }
    if (argc != 3 || std::strcmp(argv[1], "--frontier") != 0) {
        std::fprintf(stderr, "usage: %s [--frontier 1..4096]\n", argv[0]);
        return 2;
    }
    const int frontier = std::atoi(argv[2]);
    if (frontier <= 0 || frontier > ops::kQsaMaximumTokens) {
        std::fprintf(stderr, "frontier must be in [1,4096]\n");
        return 2;
    }
    run(frontier);
    return 0;
}
