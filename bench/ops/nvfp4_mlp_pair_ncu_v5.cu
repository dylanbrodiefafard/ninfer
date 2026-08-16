// Short pair + LM-head launches for NCU (no long warmup).
#include "core/device.h"
#include "ninfer/ops/linear.h"
#include "ninfer/ops/linear_swiglu.h"
#include "ninfer_bench_common.h"
#include "quantized_weight.cuh"

#include <algorithm>
#include <cstdio>

using namespace ninfer;

int main() {
    try {
        cudaDeviceReset();
        cudaStream_t stream = nullptr;
        CUDA_CHECK(cudaStreamCreate(&stream));
        constexpr int kT = 4;
        DeviceBuffer xin = bench::make_bf16(5120ull * kT);
        DeviceBuffer mid(17408ull * kT * 2);
        DeviceBuffer down_out(5120ull * kT * 2);
        DeviceBuffer logits(248320ull * kT * 2);
        auto gate = bench::make_nvfp4_weight(34816, 5120);
        auto down = bench::make_nvfp4_weight(5120, 17408);
        auto head = bench::make_row_split_weight(QType::W8G32_F16S, 248320, 5120, 5120);
        WorkspaceArena ws_sw(std::max<std::size_t>(
            ops::linear_swiglu_workspace_capacity_bytes(QType::NVFP4, 34816, 5120,
                                                       ops::LinearPolicy::AllowA4, kT, kT),
            256));
        WorkspaceArena ws_dn(std::max<std::size_t>(
            ops::linear_workspace_capacity_bytes(QType::NVFP4, 5120, 17408,
                                                 ops::LinearPolicy::AllowA4, kT, kT),
            256));
        Tensor x(xin.p, DType::BF16, {5120, kT});
        Tensor m(mid.p, DType::BF16, {17408, kT});
        Tensor o(down_out.p, DType::BF16, {5120, kT});
        Tensor h(logits.p, DType::BF16, {248320, kT});
        auto pair = [&] {
            ops::linear_swiglu(x, gate.weight, m, ops::LinearPolicy::AllowA4, ws_sw, stream);
            ops::linear(m, down.weight, o, ops::LinearPolicy::AllowA4, ws_dn, stream);
        };
        for (int i = 0; i < 2; ++i) pair();
        CUDA_CHECK(cudaStreamSynchronize(stream));
        for (int i = 0; i < 2; ++i) pair();
        for (int i = 0; i < 2; ++i) {
            ops::linear(x, head.weight, h, stream);
        }
        CUDA_CHECK(cudaStreamSynchronize(stream));
        std::printf("OK ncu pair+lmhead T=4\n");
        CUDA_CHECK(cudaStreamDestroy(stream));
        return 0;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "FAIL: %s\n", e.what());
        return 1;
    }
}
