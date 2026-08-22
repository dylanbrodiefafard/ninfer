#include "ninfer/ops/l2norm.h"
#include "ops/norm_test_common.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

using namespace ninfer;
using namespace ninfer::test;

namespace {

constexpr float kEps = 1.0e-6F;

constexpr ReductionCriterion l2norm_bf16_criterion() {
    return {/*relative_l2*/ 1.9e-3, /*gross_absolute*/ 2.0e-7,
            /*gross_relative_to_max_reference*/ 3.2e-3};
}

struct Shape {
    std::int32_t d;
    std::int32_t heads;
    std::int32_t tokens;

    std::size_t elements() const {
        return static_cast<std::size_t>(d) * static_cast<std::size_t>(heads) *
               static_cast<std::size_t>(tokens);
    }
};

std::vector<double> l2norm_oracle(const std::vector<float>& x, const Shape& shape) {
    std::vector<double> output(x.size());
    const auto rows = static_cast<std::int64_t>(shape.heads) * shape.tokens;
    for (std::int64_t row = 0; row < rows; ++row) {
        const std::size_t base = static_cast<std::size_t>(row) * shape.d;
        double sum_squares     = 0.0;
        for (std::int32_t column = 0; column < shape.d; ++column) {
            const double value = x[base + column];
            sum_squares += value * value;
        }
        const double inverse = 1.0 / std::sqrt(sum_squares + kEps);
        for (std::int32_t column = 0; column < shape.d; ++column) {
            output[base + column] = static_cast<double>(x[base + column]) * inverse;
        }
    }
    return output;
}

int run_case(const char* label, const Shape& shape, std::uint32_t seed, float scale,
             bool bf16x2_unaligned = false) {
    const std::size_t n = shape.elements();
    std::vector<float> x(n);
    fill_uniform(x, seed, -scale, scale);
    round_to_bf16(x);
    const std::vector<double> reference = l2norm_oracle(x, shape);

    norm::DeviceInput device_x = norm::make_input(x, bf16x2_unaligned);
    const std::size_t leading  = bf16x2_unaligned ? sizeof(std::uint16_t) : 0;
    GuardedDeviceBuffer output(leading + n * sizeof(std::uint16_t));
    output.fill(0xff);
    void* output_data = static_cast<std::uint8_t*>(output.data()) + leading;

    Tensor tx(device_x.data, DType::BF16, {shape.d, shape.heads, shape.tokens});
    Tensor tout(output_data, DType::BF16, {shape.d, shape.heads, shape.tokens});
    ops::l2norm(tx, kEps, tout, nullptr);
    cuda_synchronize();

    int failures = verify_reduction(label, from_device_bf16(output_data, n), reference,
                                    l2norm_bf16_criterion());
    failures +=
        norm::verify_output_storage(std::string(label) + " output", output, bf16x2_unaligned);
    failures += norm::verify_preserved(std::string(label) + " preserves x", device_x);
    return failures;
}

template <class T>
std::string json_float_array(const std::vector<T>& v) {
    std::ostringstream os;
    os << "[";
    for (std::size_t i = 0; i < v.size(); ++i) {
        char buf[32];
        std::snprintf(buf, sizeof(buf), "%.9g", static_cast<double>(v[i]));
        if (i) { os << ", "; }
        os << buf;
    }
    os << "]";
    return os.str();
}

// Dev side-band: run one case through l2norm_dump and emit the kernel and FP64
// reference per-row intermediates (sumsq, inv_r) as JSON so tools/kdev/diff.py can
// localize the first divergent stage without editing the kernel.
int dump_case(const char* label, const Shape& shape, std::uint32_t seed, float scale,
              bool bf16x2_unaligned, const std::string& path) {
    const std::size_t n = shape.elements();
    std::vector<float> x(n);
    fill_uniform(x, seed, -scale, scale);
    round_to_bf16(x);

    const std::int64_t rows = static_cast<std::int64_t>(shape.heads) * shape.tokens;
    norm::DeviceInput device_x = norm::make_input(x, bf16x2_unaligned);
    const std::size_t leading  = bf16x2_unaligned ? sizeof(std::uint16_t) : 0;
    GuardedDeviceBuffer output(leading + n * sizeof(std::uint16_t));
    output.fill(0xff);
    void* output_data = static_cast<std::uint8_t*>(output.data()) + leading;

    Tensor tx(device_x.data, DType::BF16, {shape.d, shape.heads, shape.tokens});
    Tensor tout(output_data, DType::BF16, {shape.d, shape.heads, shape.tokens});

    DeviceBuffer d_sumsq(static_cast<std::size_t>(rows) * sizeof(float));
    DeviceBuffer d_invr(static_cast<std::size_t>(rows) * sizeof(float));
    ops::L2NormDump dump{static_cast<float*>(d_sumsq.p), static_cast<float*>(d_invr.p)};
    ops::l2norm_dump(tx, kEps, tout, nullptr, dump);
    cuda_synchronize();

    const std::vector<float> k_sumsq = from_device<float>(d_sumsq, rows);
    const std::vector<float> k_invr  = from_device<float>(d_invr, rows);

    const std::size_t rrows = static_cast<std::size_t>(rows);
    std::vector<double> r_sumsq(rrows), r_invr(rrows);
    for (std::int64_t row = 0; row < rows; ++row) {
        const std::size_t base = static_cast<std::size_t>(row) * shape.d;
        double s               = 0.0;
        for (std::int32_t c = 0; c < shape.d; ++c) {
            const double value = x[base + c];
            s += value * value;
        }
        r_sumsq[row] = s;
        r_invr[row]  = 1.0 / std::sqrt(s + static_cast<double>(kEps));
    }

    std::ostringstream doc;
    doc << "{ \"op\": \"l2norm\", \"case\": \"" << label << "\", \"rows\": " << rows
        << ", \"stages\": ["
        << "{\"name\":\"sumsq\",\"kernel\":" << json_float_array(k_sumsq)
        << ",\"ref\":" << json_float_array(r_sumsq) << "},"
        << "{\"name\":\"inv_r\",\"kernel\":" << json_float_array(k_invr)
        << ",\"ref\":" << json_float_array(r_invr) << "}"
        << "] }";
    std::ofstream out(path);
    out << doc.str() << "\n";
    return 0;
}

} // namespace

int main(int argc, char** argv) {
    if (cuda_unavailable()) {
        std::cerr << "FAIL: no usable CUDA device\n";
        return 1;
    }

    struct Case {
        const char* label;
        Shape shape;
        std::uint32_t seed;
        float scale;
        bool unaligned;
    };
    // Q/K normalization uses D=128 with 16 heads in both registered text targets.
    const std::vector<Case> cases = {
        {"l2norm [128,16,1]", {128, 16, 1}, 2101U, 4.0F, false},
        {"l2norm [128,16,7]", {128, 16, 7}, 2102U, 4.0F, false},
        {"l2norm [128,16,1024]", {128, 16, 1024}, 2103U, 4.0F, false},
        // Alignment is not part of the public contract.
        {"l2norm unaligned [128,16,7]", {128, 16, 7}, 2201U, 4.0F, true},
        // eps-dominated rows exercise the same public formula without an invalid case.
        {"l2norm near-zero [128,16,1]", {128, 16, 1}, 2202U, 1.0e-7F, false},
    };

    // Optional fast-signal filters for the dev control plane (tools/kdev). With no
    // argument the suite runs every case, exactly as ctest invokes it.
    std::string only;
    std::string dump_path;
    bool fast = false;
    for (int i = 1; i < argc; ++i) {
        if (!std::strcmp(argv[i], "--fast")) {
            fast = true; // run only the first (cheapest) case
        } else if (!std::strcmp(argv[i], "--only") && i + 1 < argc) {
            only = argv[++i]; // run only cases whose label contains this substring
        } else if (!std::strcmp(argv[i], "--dump") && i + 1 < argc) {
            dump_path = argv[++i]; // emit the first case's intermediates as JSON
        } else {
            std::fprintf(stderr, "usage: %s [--fast | --only <substr> | --dump <path>]\n", argv[0]);
            return 2;
        }
    }

    if (!dump_path.empty()) {
        const Case& c = cases.front(); // side-band dumps the cheap (fast) case
        dump_case(c.label, c.shape, c.seed, c.scale, c.unaligned, dump_path);
        std::cout << "DUMP " << c.label << " -> " << dump_path << "\n";
        return 0;
    }

    int failures = 0;
    int ran = 0;
    for (const auto& c : cases) {
        if (fast && ran != 0) break;
        if (!only.empty() && std::string(c.label).find(only) == std::string::npos) continue;
        failures += run_case(c.label, c.shape, c.seed, c.scale, c.unaligned);
        ++ran;
    }
    if (ran == 0) {
        std::fprintf(stderr, "l2norm: no case matched the filter\n");
        return 2;
    }

    std::cout << (failures ? "FAIL" : "OK") << " l2norm correctness (" << ran << " cases)\n";
    return failures ? 1 : 0;
}
