#pragma once

#include <algorithm>
#include <cmath>
#include <span>
#include <string>
#include <utility>
#include <vector>

namespace ninfer::test::qwen4_sequence {
// Only test-owned adapters to existing component fixtures and their single mathematical oracle.
struct Result {
    std::vector<float> actual, reference; int failures = 0;
    std::vector<int> discrete_ids; // Optional observable router/selector trace for storage experiments.
    Result() = default;
    Result(const std::vector<double>& values, std::vector<float> expected)
        : actual(values.begin(),values.end()), reference(std::move(expected)) {}
};
inline std::vector<double> wide(const std::vector<float>& values) { return {values.begin(),values.end()}; }
struct ReadResult { Result mixed, scale; };
Result final_read(const std::string& path,const Result& residual,bool partitioned=false);
Result residual_fp8_store(const Result& input, const std::string& path, int layer,
                          const std::string& kind);
inline float represented(double value) {
    if (value == 0 || !std::isfinite(value)) { return static_cast<float>(value); }
    const double magnitude = std::abs(value);
    const int exponent = std::max(-133, std::ilogb(magnitude) - 7);
    const double units = std::ldexp(magnitude, -exponent);
    double integral = std::floor(units);
    const double remainder = units - integral;
    if (remainder > .5 || (remainder == .5 && std::fmod(integral, 2.) != 0)) { ++integral; }
    return static_cast<float>(std::copysign(std::ldexp(integral, exponent), value));
}
inline std::vector<float> represented(std::span<const double> input) {
    std::vector<float> result(input.size());
    for (std::size_t i = 0; i < input.size(); ++i) { result[i] = represented(input[i]); }
    return result;
}
ReadResult read(const std::string& path, int layer, const Result& residual, const std::string& kind="attn",
                bool partitioned=false);
Result inject(const Result& residual, const Result& block, const Result& scale, bool partitioned=false);
Result gdn(const std::string& path, int layer, const Result& input, bool partitioned);
Result gdn_calibrated(const std::string& root, const std::string& path, int layer,
                      const Result& input, bool partitioned, int mask);
int gdn_a8_input_diagnostic(const std::string& root,const Result& input);
Result qsa(const std::string& path, const Result& input, bool partitioned, bool diagnostic_nvfp4=false);
Result qsa_calibrated(const std::string& root,const std::string& path,const Result& input,
                      bool partitioned);
Result ple(const std::string& root, const Result& input, bool partitioned, bool nvfp4_table=false,
           bool text_panel=false);
Result moe(const std::string& path, int layer, const Result& input, bool partitioned = false,
           bool allow_a4 = false);
void moe_a4_calibration_coverage(const std::string& path,int layer,
                                 std::span<const float> input,std::span<const int> experts);
Result moe_calibrated(const std::string& root, const std::string& path, int layer,
                      const Result& input, bool partitioned, int mask, bool allow_a4 = false);
}
