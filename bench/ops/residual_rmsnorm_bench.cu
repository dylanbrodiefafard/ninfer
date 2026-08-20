// Bit-exact + performance bench for the fused residual-update + unit-offset RMSNorm op.
//
// The fused `residual_rmsnorm` op replaces the `residual_add` + `rmsnorm(unit_offset=true)`
// launch pair used by the MTP / draft attention path (one kernel instead of two). This bench is
// the bit-exact registration for that op: it runs the fused op and the unfused pair on identical
// inputs and requires the in-place residual AND the normalized output to be byte-identical (the
// kernel is documented bit-exact against the pair). A GB/s readout for both routes quantifies the
// launch-fusion + traffic win (the fused kernel saves one read of the residual).
//
// Examples:
//   ./ninfer_residual_rmsnorm_bench                     # D=5120, T=4, gate on
//   ./ninfer_residual_rmsnorm_bench --t-sweep 1,2,4,16
//   ./ninfer_residual_rmsnorm_bench --d 4096 --t 16 --no-bitexact
//   ncu ... ./ninfer_residual_rmsnorm_bench --d 5120 --t 4 --profile

#include "ninfer/ops/residual_add.h"
#include "ninfer/ops/residual_rmsnorm.h"
#include "ninfer/ops/rmsnorm.h"
#include "ninfer_bench_common.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

using namespace ninfer;
using namespace ninfer::bench;

namespace {

struct Options {
    int d              = 5120;
    float eps          = 1.0e-6f;
    std::vector<int> tokens{4};
    bool bitexact      = true;
    bool profile       = false;
    int warmup         = 20;
    int repeat         = 100;
};

// Seeded LCG -> varied BF16 in ~[-1,1] (exercises the reduction; exact values are irrelevant to
// the byte-exact gate, which compares two implementations of the same math).
std::vector<std::uint16_t> gen_bf16(std::size_t n, std::uint32_t seed) {
    std::vector<std::uint16_t> h(n);
    std::uint32_t state = seed ? seed : 1u;
    for (std::size_t i = 0; i < n; ++i) {
        state = state * 1664525u + 1013904223u;
        const float u   = static_cast<float>((state >> 8) & 0x00ffffffu) * (1.0f / 16777216.0f);
        h[i]            = f32_to_bf16(2.0f * u - 1.0f);
    }
    return h;
}

DeviceBuffer upload_bf16(const std::vector<std::uint16_t>& h) {
    DeviceBuffer d(h.size() * sizeof(std::uint16_t));
    d.copy_from_host(h.data(), d.bytes);
    return d;
}

std::vector<int> parse_tokens(std::string_view text) {
    std::vector<int> result;
    std::size_t begin = 0;
    while (begin <= text.size()) {
        const std::size_t end = text.find(',', begin);
        const std::string_view token =
            text.substr(begin, end == std::string_view::npos ? std::string_view::npos : end - begin);
        if (token.empty()) { throw std::invalid_argument("empty token in --t-sweep"); }
        const int v = std::atoi(std::string(token).c_str());
        if (v <= 0) { throw std::invalid_argument("tokens must be positive"); }
        result.push_back(v);
        if (end == std::string_view::npos) { break; }
        begin = end + 1;
    }
    return result;
}

Options parse_args(int argc, char** argv) {
    Options opt;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        auto next = [&](const char* what) -> std::string {
            if (i + 1 >= argc) { throw std::invalid_argument(std::string("missing value for ") + what); }
            return argv[++i];
        };
        if (a == "--d") {
            opt.d = std::atoi(next("--d").c_str());
        } else if (a == "--eps") {
            opt.eps = std::atof(next("--eps").c_str());
        } else if (a == "--t") {
            opt.tokens = {std::atoi(next("--t").c_str())};
        } else if (a == "--t-sweep") {
            opt.tokens = parse_tokens(next("--t-sweep"));
        } else if (a == "--no-bitexact") {
            opt.bitexact = false;
        } else if (a == "--profile") {
            opt.profile = true;
        } else if (a == "--warmup") {
            opt.warmup = std::atoi(next("--warmup").c_str());
        } else if (a == "--repeat") {
            opt.repeat = std::atoi(next("--repeat").c_str());
        } else if (a == "--help" || a == "-h") {
            std::printf("usage: %s [--d D] [--eps E] [--t T | --t-sweep T,...] [--no-bitexact] "
                        "[--profile] [--warmup N] [--repeat N]\n",
                        argv[0]);
            std::exit(0);
        } else {
            throw std::invalid_argument("unknown argument: " + a);
        }
    }
    return opt;
}

// Returns true when the fused op is byte-identical to the unfused pair (or the gate is disabled).
bool run(const Options& opt, int tokens) {
    const std::size_t E  = static_cast<std::size_t>(opt.d) * static_cast<std::size_t>(tokens);
    const auto yh        = gen_bf16(E, 0xA11CE5EDu);
    const auto xh        = gen_bf16(E, 0x5EEDBEEFu);
    const auto wh        = gen_bf16(opt.d, 0x0BADF00Du);

    DeviceBuffer y       = upload_bf16(yh);
    DeviceBuffer w       = upload_bf16(wh);
    // Gate buffers: two independent residual copies of the same x0.
    DeviceBuffer x_ref   = upload_bf16(xh);
    DeviceBuffer out_ref = make_zeros(E * sizeof(std::uint16_t));
    DeviceBuffer x_fused = upload_bf16(xh);
    DeviceBuffer out_fused = make_zeros(E * sizeof(std::uint16_t));

    Tensor ty(y.p, DType::BF16, {opt.d, tokens});
    Tensor tw(w.p, DType::BF16, {opt.d});
    Tensor txr(x_ref.p, DType::BF16, {opt.d, tokens});
    Tensor tor(out_ref.p, DType::BF16, {opt.d, tokens});
    Tensor txf(x_fused.p, DType::BF16, {opt.d, tokens});
    Tensor tof(out_fused.p, DType::BF16, {opt.d, tokens});

    cudaStream_t stream = nullptr;

    // Unfused reference pair (the MTP / draft path before the fusion).
    ops::residual_add(ty, txr, stream);
    ops::rmsnorm(txr, tw, opt.eps, true, tor, stream);
    // Fused single launch.
    ops::residual_rmsnorm(ty, txf, tw, opt.eps, tof, stream);
    cudaStreamSynchronize(stream);

    bool gate_ok = true;
    if (opt.bitexact) {
        std::vector<std::uint16_t> xr(E), xf(E), orr(E), ofr(E);
        x_ref.copy_to_host(xr.data(), x_ref.bytes);
        x_fused.copy_to_host(xf.data(), x_fused.bytes);
        out_ref.copy_to_host(orr.data(), out_ref.bytes);
        out_fused.copy_to_host(ofr.data(), out_fused.bytes);
        const bool x_ok  = std::memcmp(xr.data(), xf.data(), E * sizeof(std::uint16_t)) == 0;
        const bool out_ok = std::memcmp(orr.data(), ofr.data(), E * sizeof(std::uint16_t)) == 0;
        gate_ok          = x_ok && out_ok;
        std::printf("residual_rmsnorm bitexact [D=%d,T=%d] residual=%s normalized=%s\n", opt.d,
                    tokens, x_ok ? "OK" : "MISMATCH", out_ok ? "OK" : "MISMATCH");
    }

    if (opt.profile) {
        // Single-shot for the profiler; re-seed a fresh residual so ncu sees one clean launch.
        DeviceBuffer xp = upload_bf16(xh);
        DeviceBuffer op = make_zeros(E * sizeof(std::uint16_t));
        Tensor tpx(xp.p, DType::BF16, {opt.d, tokens});
        Tensor top(op.p, DType::BF16, {opt.d, tokens});
        ops::residual_rmsnorm(ty, tpx, tw, opt.eps, top, stream);
        cudaStreamSynchronize(stream);
        std::printf("profile residual_rmsnorm [D=%d,T=%d]\n", opt.d, tokens);
        return gate_ok;
    }

    // Performance readout: fused single launch vs the unfused pair. The fused kernel reads the
    // residual once (fused: 4*E + d elements moved); the pair reads it twice (5*E + d).
    DeviceBuffer y_p = upload_bf16(yh);
    DeviceBuffer w_p = upload_bf16(wh);
    DeviceBuffer xf_p = upload_bf16(xh);
    DeviceBuffer of_p = make_zeros(E * sizeof(std::uint16_t));
    DeviceBuffer xu_p = upload_bf16(xh);
    DeviceBuffer ou_p = make_zeros(E * sizeof(std::uint16_t));
    Tensor tyy(y_p.p, DType::BF16, {opt.d, tokens});
    Tensor twp(w_p.p, DType::BF16, {opt.d});
    Tensor txfp(xf_p.p, DType::BF16, {opt.d, tokens});
    Tensor tofp(of_p.p, DType::BF16, {opt.d, tokens});
    Tensor txup(xu_p.p, DType::BF16, {opt.d, tokens});
    Tensor toup(ou_p.p, DType::BF16, {opt.d, tokens});

    const double bytes_fused   = 2.0 * (4.0 * static_cast<double>(E) + static_cast<double>(opt.d));
    const double bytes_unfused = 2.0 * (5.0 * static_cast<double>(E) + static_cast<double>(opt.d));

    const auto fused_launch = [&](cudaStream_t s) {
        ops::residual_rmsnorm(tyy, txfp, twp, opt.eps, tofp, s);
    };
    const auto unfused_launch = [&](cudaStream_t s) {
        ops::residual_add(tyy, txup, s);
        ops::rmsnorm(txup, twp, opt.eps, true, toup, s);
    };

    const Result rf = bench_loop(fused_launch, bytes_fused, opt.warmup, opt.repeat);
    const Result ru = bench_loop(unfused_launch, bytes_unfused, opt.warmup, opt.repeat);

    char tag[48];
    std::snprintf(tag, sizeof(tag), "residual_rmsnorm T=%d", tokens);
    print_result(tag, rf);
    std::snprintf(tag, sizeof(tag), "residual_add+rmsnorm T=%d", tokens);
    print_result(tag, ru);

    return gate_ok;
}

} // namespace

int main(int argc, char** argv) {
    int device_count = 0;
    if (cudaGetDeviceCount(&device_count) != cudaSuccess || device_count == 0) {
        std::printf("SKIP: no usable CUDA device\n");
        return 0;
    }

    Options opt;
    try {
        opt = parse_args(argc, argv);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "error: %s\n", e.what());
        return 2;
    }

    // The launcher dispatches the CTA kernel for D in (3072, 8192] with D % 1024 == 0.
    if (opt.d % 1024 != 0 || opt.d <= 3072 || opt.d > 8192) {
        std::fprintf(stderr, "error: D=%d outside the fused-op range (3072<D<=8192, D%%1024==0)\n",
                     opt.d);
        return 2;
    }
    if (opt.repeat <= 0) {
        std::fprintf(stderr, "error: --repeat must be positive\n");
        return 2;
    }

    bool all_ok = true;
    for (const int tokens : opt.tokens) {
        all_ok = run(opt, tokens) && all_ok;
    }
    // Exit 3 distinguishes a bit-exact gate failure from a usage error (exit 2).
    return all_ok ? 0 : 3;
}