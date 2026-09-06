#include "ops/kernel/gqa_xattn_sort.cuh"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {

constexpr int kThreads = 512;
constexpr int kMaxRank = 4096;

void cuda_check(cudaError_t status, const char* operation) {
    if (status != cudaSuccess) {
        throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(status));
    }
}

__global__ void xattn_sort_oracle_kernel(const float* input_keys, const int* input_ids,
                                         float* output_keys, int* output_ids, int nsort) {
    extern __shared__ unsigned char storage[];
    auto* keys = reinterpret_cast<float*>(storage);
    auto* ids  = reinterpret_cast<int*>(keys + kMaxRank);
    const int tid = static_cast<int>(threadIdx.x);
    for (int i = tid; i < nsort; i += kThreads) {
        keys[i] = input_keys[i];
        ids[i]  = input_ids[i];
    }
    __syncthreads();
    ninfer::ops::gqa_xattn_sort_desc<kThreads>(keys, ids, nsort, tid);
    for (int i = tid; i < nsort; i += kThreads) {
        output_keys[i] = keys[i];
        output_ids[i]  = ids[i];
    }
}

int next_rank_width(int live) {
    int width = 64;
    while (width < live && width < kMaxRank) { width *= 2; }
    return width;
}

struct RankedValue {
    float key;
    int id;
};

bool rank_before(const RankedValue& a, const RankedValue& b) {
    if (a.key != b.key) { return a.key > b.key; }
    if (a.id >= 0 && b.id < 0) { return true; }
    if (a.id < 0 && b.id >= 0) { return false; }
    return a.id >= 0 && b.id >= 0 && a.id < b.id;
}

int run_case(int live, float* d_input_keys, int* d_input_ids, float* d_output_keys,
             int* d_output_ids) {
    const int width = next_rank_width(live);
    std::vector<RankedValue> expected(static_cast<std::size_t>(width));
    for (int i = 0; i < width; ++i) {
        if (i < live) {
            // Many equal scores exercise the deterministic id tie break. Valid -inf
            // entries exercise ordering relative to the padded (-inf, -1) sentinel.
            const float key = (i % 113 == 0)
                                  ? -std::numeric_limits<float>::infinity()
                                  : static_cast<float>((i * 37 + 11) % 29 - 14) * 0.25f;
            expected[static_cast<std::size_t>(i)] = {key, i};
        } else {
            expected[static_cast<std::size_t>(i)] =
                {-std::numeric_limits<float>::infinity(), -1};
        }
    }
    std::vector<float> input_keys(static_cast<std::size_t>(width));
    std::vector<int> input_ids(static_cast<std::size_t>(width));
    for (int i = 0; i < width; ++i) {
        input_keys[static_cast<std::size_t>(i)] = expected[static_cast<std::size_t>(i)].key;
        input_ids[static_cast<std::size_t>(i)]  = expected[static_cast<std::size_t>(i)].id;
    }
    std::sort(expected.begin(), expected.end(), rank_before);

    cuda_check(cudaMemcpy(d_input_keys, input_keys.data(), input_keys.size() * sizeof(float),
                          cudaMemcpyHostToDevice),
               "copy input keys");
    cuda_check(cudaMemcpy(d_input_ids, input_ids.data(), input_ids.size() * sizeof(int),
                          cudaMemcpyHostToDevice),
               "copy input ids");
    constexpr std::size_t kSharedBytes =
        kMaxRank * (sizeof(float) + sizeof(int));
    xattn_sort_oracle_kernel<<<1, kThreads, kSharedBytes>>>(
        d_input_keys, d_input_ids, d_output_keys, d_output_ids, width);
    cuda_check(cudaGetLastError(), "launch XAttention sort oracle kernel");

    std::vector<float> actual_keys(static_cast<std::size_t>(width));
    std::vector<int> actual_ids(static_cast<std::size_t>(width));
    cuda_check(cudaMemcpy(actual_keys.data(), d_output_keys,
                          actual_keys.size() * sizeof(float), cudaMemcpyDeviceToHost),
               "copy output keys");
    cuda_check(cudaMemcpy(actual_ids.data(), d_output_ids, actual_ids.size() * sizeof(int),
                          cudaMemcpyDeviceToHost),
               "copy output ids");

    for (int i = 0; i < width; ++i) {
        const RankedValue want = expected[static_cast<std::size_t>(i)];
        const bool key_equal =
            std::memcmp(&actual_keys[static_cast<std::size_t>(i)], &want.key, sizeof(float)) == 0;
        if (!key_equal || actual_ids[static_cast<std::size_t>(i)] != want.id) {
            std::cerr << "XAttention rank live=" << live << " width=" << width << " item=" << i
                      << " got=(" << actual_keys[static_cast<std::size_t>(i)] << ","
                      << actual_ids[static_cast<std::size_t>(i)] << ") want=(" << want.key << ","
                      << want.id << ")\n";
            return 1;
        }
    }
    return 0;
}

} // namespace

int main() {
    try {
        int device_count = 0;
        if (cudaGetDeviceCount(&device_count) != cudaSuccess || device_count == 0) {
            std::cout << "SKIP gqa_xattn_sort: no usable CUDA device\n";
            return 0;
        }

        float* d_input_keys  = nullptr;
        int* d_input_ids     = nullptr;
        float* d_output_keys = nullptr;
        int* d_output_ids    = nullptr;
        cuda_check(cudaMalloc(&d_input_keys, kMaxRank * sizeof(float)), "allocate input keys");
        cuda_check(cudaMalloc(&d_input_ids, kMaxRank * sizeof(int)), "allocate input ids");
        cuda_check(cudaMalloc(&d_output_keys, kMaxRank * sizeof(float)), "allocate output keys");
        cuda_check(cudaMalloc(&d_output_ids, kMaxRank * sizeof(int)), "allocate output ids");

        int failures = 0;
        constexpr int kLiveCases[] = {1,    63,   64,   65,   127,  128,  129,
                                      255,  256,  257,  511,  512,  513,  1023,
                                      1024, 1025, 2047, 2048, 2049, 4095, 4096};
        for (int live : kLiveCases) {
            failures += run_case(live, d_input_keys, d_input_ids, d_output_keys, d_output_ids);
        }

        cuda_check(cudaFree(d_input_keys), "free input keys");
        cuda_check(cudaFree(d_input_ids), "free input ids");
        cuda_check(cudaFree(d_output_keys), "free output keys");
        cuda_check(cudaFree(d_output_ids), "free output ids");
        std::cout << (failures == 0 ? "PASS" : "FAIL")
                  << " gqa_xattn_sort exact dispatcher oracle\n";
        return failures == 0 ? 0 : 1;
    } catch (const std::exception& e) {
        std::cerr << "FAIL gqa_xattn_sort: " << e.what() << '\n';
        return 1;
    }
}
