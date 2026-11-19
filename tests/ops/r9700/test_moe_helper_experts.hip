// moe_helper_experts Op test on the R9700 (gfx1201) against an independent FP64 oracle at the
// Qwen3.8-Flash-Next record geometry, for decode-round extents (max_tokens <= 48) and prefill
// extents up to the Op's capacity.
//
// Test experts are canonical codes and E4M3 scales written into the documented record layout by
// the test-side packer and read back exactly by an independent reader
// (ops/moe_expert_record_reference.h). The oracle decodes each code (through the matrix's F3 book
// or as E2M1) times its exact E4M3 scale, applies the FP32 matrix scales and evaluates the routed
// SwiGLU sum in FP64 from the represented BF16 x and FP32 route weights.
#include "ninfer/ops/moe_helper_experts.h"
#include "ops/moe_expert_record_reference.h"
#include "ops/op_check.h"

#include <hip/hip_runtime.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <exception>
#include <functional>
#include <iostream>
#include <memory>
#include <numeric>
#include <random>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

using namespace ninfer;
using namespace ninfer::test::moe_record;

namespace {

constexpr int kHidden  = ops::kMoeHidden;
constexpr int kInter   = ops::kMoeIntermediate;
constexpr int kExperts = ops::kMoeHelperExperts;
constexpr int kRoutes  = ops::kMoeHelperRoutesPerToken;
constexpr int kSkip    = 77;
// The prefill workspace of most prefill cases: one 4096-token chunk.
constexpr int kPrefillCapacity = 4096;

// Criterion. Against the FP64 oracle, one rounding dominates: the gate/up product enters the down
// projection rounded to BF16 (8 significant bits; relative RMS error about 2^-7 / sqrt(12) / 1.5
// ~= 1.5e-3 for values spread over a binade). Its independent element errors pass through the
// down matrix with the same gain as the product itself, so each route's contribution carries
// about 1.5e-3 relative RMS error, and so does their weighted sum. An FP32 output adds no output
// rounding; FP32 sums over K = 2560 and 640, the per-group E4M3 scaling and the FP32 SiLU stay
// below 1e-5. Expected relative L2 is about 1.5e-3; the bound allows 3e-3. Pointwise, the error
// is Gaussian-like with RMS near 1.5e-3 rms(y) ~= 4e-4 max|y| (rms/max ~ 1/4 for these sums), so
// six deviations stay under 2.5e-3 max|y|; the gross cap is 4e-3 of the largest reference
// magnitude of the case, an absolute bound that elements cancelling to near zero also meet.
constexpr test::ReductionCriterion kCriterion{/*relative_l2*/ 3.0e-3, /*gross_absolute*/ 0.0,
                                              /*gross_relative_to_max_reference*/ 4.0e-3};

// BF16 output criterion, derived from one BF16 rounding of the FP32 partial z that kCriterion
// bounds against the FP64 oracle r. Round to nearest even with 8 significant bits gives
// |bf16(z) - z| <= u |z| per element, u = 2^-8 (BF16 has FP32's exponent range, so only FP32
// subnormals, 2^-126 and below, could exceed it, by at most 2^-134 absolute). With
// ||z - r|| <= e ||r|| and so ||z|| <= (1 + e) ||r||, the triangle inequality gives
// ||bf16(z) - r|| <= (e + u (1 + e)) ||r||: in the L2 norm for relative_l2, and per element with
// |z_i| <= (1 + g) max|r| for the gross cap g. The BF16 route is checked against the same
// unrounded oracle with these bounds, and every element it writes must equal the RNE BF16 of the
// FP32 route's element bit for bit.
constexpr double kBf16UnitRoundoff = 0x1p-8;
constexpr test::ReductionCriterion kBf16Criterion{
    /*relative_l2*/ kCriterion.relative_l2 + kBf16UnitRoundoff * (1.0 + kCriterion.relative_l2),
    /*gross_absolute*/ 0.0,
    /*gross_relative_to_max_reference*/ kCriterion.gross_relative_to_max_reference +
        kBf16UnitRoundoff * (1.0 + kCriterion.gross_relative_to_max_reference)};

using OutputDtype = ops::MoeHelperOutputDtype;

void hip_check(hipError_t status, const char* what) {
    if (status != hipSuccess) {
        throw std::runtime_error(std::string(what) + ": " + hipGetErrorString(status));
    }
}

template <class T>
class DeviceArray {
public:
    explicit DeviceArray(std::size_t count) : count_(count) {
        void* pointer = nullptr;
        hip_check(hipMalloc(&pointer, count * sizeof(T)), "hipMalloc");
        data_ = static_cast<T*>(pointer);
    }

    ~DeviceArray() { (void)hipFree(data_); }

    DeviceArray(const DeviceArray&)            = delete;
    DeviceArray& operator=(const DeviceArray&) = delete;

    [[nodiscard]] T* data() const noexcept { return data_; }

    void upload(const T* host, std::size_t count, std::size_t offset = 0) {
        hip_check(hipMemcpy(data_ + offset, host, count * sizeof(T), hipMemcpyHostToDevice),
                  "upload");
    }

    void upload(const std::vector<T>& host) { upload(host.data(), host.size()); }

    [[nodiscard]] std::vector<T> download() const {
        std::vector<T> host(count_);
        hip_check(hipMemcpy(host.data(), data_, count_ * sizeof(T), hipMemcpyDeviceToHost),
                  "download");
        return host;
    }

    void fill_bytes(int value) {
        hip_check(hipMemset(data_, value, count_ * sizeof(T)), "hipMemset");
    }

private:
    T* data_ = nullptr;
    std::size_t count_;
};

float bf16_round(float value) {
    std::uint32_t bits = 0;
    std::memcpy(&bits, &value, 4);
    bits          = (bits + 0x7FFFU + ((bits >> 16) & 1U)) & 0xFFFF0000U;
    float rounded = 0.0F;
    std::memcpy(&rounded, &bits, 4);
    return rounded;
}

std::uint16_t bf16_bits(float bf16_value) {
    std::uint32_t bits = 0;
    std::memcpy(&bits, &bf16_value, 4);
    return static_cast<std::uint16_t>(bits >> 16);
}

// Distinct test records: kWideRecords wide-scale records (even: F3, odd: NVFP4), then one F3 and
// one NVFP4 record of subnormal scales. Helper-owned expert ids map onto them per bank.
constexpr int kWideRecords      = 12;
constexpr int kSubnormalRecords = 2;

struct Fixture {
    std::array<std::uint32_t, 3> books{};
    std::vector<Expert> experts;
    std::vector<std::size_t> offsets; // byte offset of each record in the device store
    std::size_t store_bytes = 0;
};

std::unique_ptr<Fixture> make_fixture(bool& round_trip) {
    auto fixture = std::make_unique<Fixture>();
    std::mt19937 rng(20261118U);
    // Gate uses the artifact's book; up and down are signed permutations including +-0 and +-1.5.
    fixture->books[0] = 0x76419CEFU;
    for (int m = 1; m < 3; ++m) {
        std::array<std::uint32_t, 8> entries = {0x0, 0xB, 0x2, 0xD, 0x5, 0xE, 0x7, 0x9};
        std::shuffle(entries.begin(), entries.end(), rng);
        std::uint32_t book = 0;
        for (std::size_t v = 0; v < entries.size(); ++v) { book |= entries[v] << (4 * v); }
        fixture->books[static_cast<std::size_t>(m)] = book;
    }
    const int records = kWideRecords + kSubnormalRecords;
    for (int r = 0; r < records; ++r) {
        const auto codec = r % 2 == 0 ? ops::MoeExpertCodec::F3G32 : ops::MoeExpertCodec::Nvfp4G16;
        const ScaleMode mode = r < kWideRecords ? ScaleMode::Wide : ScaleMode::Subnormal;
        fixture->experts.push_back(make_expert(codec, mode, 1000U + static_cast<std::uint32_t>(r),
                                               fixture->books, round_trip));
        fixture->offsets.push_back(fixture->store_bytes);
        fixture->store_bytes += fixture->experts.back().record.size();
    }
    return fixture;
}

// One layer's bank: which ids the helper owns, the record each owned id reads, its matrix scales.
struct BankState {
    std::vector<int> record; // -1: not owned
    std::vector<float> matrix_scales;
};

enum class RecordSet : std::uint8_t { Wide, Subnormal };

BankState make_bank(RecordSet set, std::uint32_t seed) {
    std::mt19937 rng(seed);
    BankState bank{std::vector<int>(kExperts, -1), std::vector<float>(kExperts * 3)};
    for (int id = 0; id < kExperts; ++id) {
        if (id % 3 == 0) { continue; } // the 5090 owns these
        bank.record[static_cast<std::size_t>(id)] =
            set == RecordSet::Wide ? (id * 7 + 3) % kWideRecords : kWideRecords + id % 2;
    }
    // Matrix scales keep weight magnitudes comparable across scale modes (RMS about 0.015).
    const double base = set == RecordSet::Wide ? 3.0e-5 : 0.6;
    std::uniform_real_distribution<double> spread(0.5, 1.5);
    for (auto& scale : bank.matrix_scales) { scale = static_cast<float>(base * spread(rng)); }
    return bank;
}

struct Inputs {
    int tokens; // N
    int max_tokens;
    std::vector<float> x; // [max_tokens][2560], BF16-representable
    std::vector<ops::MoeHelperRoute> routes;
};

// How a case routes its tokens. Skewed: moe_route-like top-10 of biased logits, so popular
// experts repeat across tokens; the entries the bank does not own are inactive. Concentrated:
// every token routes the same ten experts (in its own order), so each expert has T tokens.
// Mixed48: skewed plus the special tokens make_inputs lists.
enum class Routing : std::uint8_t { Skewed, Concentrated, Mixed48 };

void route_token(Inputs& inputs, const BankState& bank, int t, const std::vector<int>& ids,
                 std::mt19937& rng) {
    std::normal_distribution<double> normal(0.0, 1.0);
    std::vector<double> logits(ids.size());
    double total = 0.0;
    for (auto& logit : logits) {
        logit = 0.5 * normal(rng);
        total += std::exp(logit);
    }
    for (std::size_t j = 0; j < ids.size(); ++j) {
        auto& entry  = inputs.routes[static_cast<std::size_t>(t) * kRoutes + j];
        const int id = ids[j];
        entry.expert = id >= 0 && bank.record[static_cast<std::size_t>(id)] >= 0
                           ? id
                           : ops::kMoeHelperInactiveExpert;
        entry.weight = static_cast<float>(std::exp(logits[j]) / total);
    }
}

Inputs make_inputs(const BankState& bank, int tokens, int max_tokens, Routing routing,
                   std::uint32_t seed) {
    std::mt19937 rng(seed);
    std::normal_distribution<double> normal(0.0, 1.0);
    const auto rows = static_cast<std::size_t>(max_tokens);
    Inputs inputs{
        tokens, max_tokens, std::vector<float>(rows * kHidden),
        std::vector<ops::MoeHelperRoute>(rows * kRoutes, {ops::kMoeHelperInactiveExpert, 0.0F})};
    std::uniform_real_distribution<float> value(-2.0F, 2.0F);
    for (auto& v : inputs.x) { v = bf16_round(value(rng)); }
    std::vector<double> bias(kExperts);
    for (auto& b : bias) { b = 2.0 * normal(rng); }
    const auto top10 = [&] {
        std::vector<double> score(kExperts);
        for (std::size_t e = 0; e < score.size(); ++e) { score[e] = bias[e] + 1.5 * normal(rng); }
        std::vector<int> order(kExperts);
        std::iota(order.begin(), order.end(), 0);
        std::partial_sort(order.begin(), order.begin() + kRoutes, order.end(), [&](int a, int b) {
            return score[static_cast<std::size_t>(a)] > score[static_cast<std::size_t>(b)];
        });
        return std::vector<int>(order.begin(), order.begin() + kRoutes);
    };
    const std::vector<int> common = top10();
    for (int t = 0; t < tokens; ++t) {
        std::vector<int> ids = routing == Routing::Concentrated ? common : top10();
        if (routing == Routing::Concentrated) { std::shuffle(ids.begin(), ids.end(), rng); }
        route_token(inputs, bank, t, ids, rng);
    }
    if (routing == Routing::Mixed48) {
        // Tokens 0 and 47 have no helper route; token 1 routes only helper experts; tokens
        // 30..45 share the same ten helper experts (one job of 16 tokens per expert); token 2
        // names one expert twice.
        const std::vector<int> unowned = {0, 3, 6, 9, 12, 15, 18, 21, 24, 27};
        route_token(inputs, bank, 0, unowned, rng);
        route_token(inputs, bank, tokens - 1, unowned, rng);
        route_token(inputs, bank, 1, {1, 2, 4, 5, 7, 8, 10, 11, 13, 14}, rng);
        const std::vector<int> shared = {500, 41, 302, 77, 115, 260, 334, 23, 455, 181};
        for (int t = 30; t < 46; ++t) { route_token(inputs, bank, t, shared, rng); }
        route_token(inputs, bank, 2, {17, 17, -1, 50, 61, -1, 70, 88, 97, 101}, rng);
        // Ids at and above 512 are inactive like -1 (the contract's "outside [0,512)").
        inputs.routes[3 * kRoutes + 4].expert = kExperts;
        inputs.routes[3 * kRoutes + 7].expert = 0x7FFFFFFF;
    }
    return inputs;
}

bool active(const ops::MoeHelperRoute& route) {
    return route.expert >= 0 && route.expert < kExperts;
}

// FP64 oracle of one token: sum over its active entries of w * E(x).
std::vector<double> oracle_token(const Fixture& fixture, const BankState& bank,
                                 const Inputs& inputs, int t) {
    const float* x = inputs.x.data() + static_cast<std::size_t>(t) * kHidden;
    std::vector<double> y(kHidden, 0.0);
    std::vector<double> gated(kInter);
    for (int j = 0; j < kRoutes; ++j) {
        const auto& route = inputs.routes[static_cast<std::size_t>(t) * kRoutes + j];
        if (!active(route)) { continue; }
        const auto id         = static_cast<std::size_t>(route.expert);
        const Expert& expert  = fixture.experts[static_cast<std::size_t>(bank.record[id])];
        const double scale[3] = {bank.matrix_scales[id * 3], bank.matrix_scales[id * 3 + 1],
                                 bank.matrix_scales[id * 3 + 2]};
        for (std::size_t i = 0; i < gated.size(); ++i) {
            double g              = 0.0;
            double u              = 0.0;
            const float* gate_row = expert.values[0].data() + i * kHidden;
            const float* up_row   = expert.values[1].data() + i * kHidden;
            for (int k = 0; k < kHidden; ++k) {
                g += static_cast<double>(gate_row[k]) * x[k];
                u += static_cast<double>(up_row[k]) * x[k];
            }
            g *= scale[0];
            u *= scale[1];
            gated[i] = g / (1.0 + std::exp(-g)) * u;
        }
        for (std::size_t n = 0; n < y.size(); ++n) {
            double sum            = 0.0;
            const float* down_row = expert.values[2].data() + n * kInter;
            for (std::size_t i = 0; i < gated.size(); ++i) {
                sum += static_cast<double>(down_row[i]) * gated[i];
            }
            y[n] += static_cast<double>(route.weight) * scale[2] * sum;
        }
    }
    return y;
}

bool has_active(const Inputs& inputs, int t) {
    for (int j = 0; j < kRoutes; ++j) {
        if (active(inputs.routes[static_cast<std::size_t>(t) * kRoutes + j])) { return true; }
    }
    return false;
}

// Device bank, inputs and one Op invocation.
class Run {
public:
    Run(const Fixture& fixture, const DeviceArray<std::uint8_t>& store, const Inputs& inputs,
        OutputDtype dtype = OutputDtype::Fp32)
        : fixture_(fixture), store_(store), inputs_(inputs), dtype_(dtype),
          rows_(static_cast<std::size_t>(inputs.max_tokens)), tokens_(1), x_(rows_ * kHidden),
          routes_(rows_ * kRoutes), table_(kExperts), scales_(kExperts * 3),
          y_(rows_ * kHidden * (dtype == OutputDtype::Bf16 ? 1 : 2)) {
        std::vector<std::uint16_t> x_bits(inputs.x.size());
        for (std::size_t i = 0; i < x_bits.size(); ++i) { x_bits[i] = bf16_bits(inputs.x[i]); }
        x_.upload(x_bits);
        routes_.upload(inputs.routes);
        const std::int32_t n = inputs.tokens;
        tokens_.upload(&n, 1);
    }

    void set_bank(const BankState& bank) {
        std::vector<std::uint64_t> entries(kExperts, 0);
        for (std::size_t id = 0; id < entries.size(); ++id) {
            const int record = bank.record[id];
            if (record < 0) { continue; }
            const auto r = static_cast<std::size_t>(record);
            entries[id]  = reinterpret_cast<std::uintptr_t>(store_.data() + fixture_.offsets[r]) |
                           static_cast<std::uint64_t>(fixture_.experts[r].codec);
        }
        table_.upload(entries);
        scales_.upload(bank.matrix_scales);
    }

    [[nodiscard]] ops::MoeHelperExpertsArgs args() const {
        ops::MoeHelperExpertsArgs args;
        args.tokens             = tokens_.data();
        args.max_tokens         = inputs_.max_tokens;
        args.x                  = x_.data();
        args.routes             = routes_.data();
        args.bank.expert_table  = table_.data();
        args.bank.matrix_scales = scales_.data();
        args.bank.f3_books      = fixture_.books;
        args.y                  = y_.data();
        args.output_dtype       = dtype_;
        return args;
    }

    // Poisons y (all-ones bits, a NaN), runs the Op and returns y's bits.
    std::vector<std::uint32_t> invoke(ops::MoeHelperExpertsWorkspace& workspace) {
        poison();
        ops::moe_helper_experts(args(), workspace, nullptr);
        hip_check(hipDeviceSynchronize(), "moe_helper_experts");
        return output();
    }

    // Poisons y and waits for the fill, so no stream can run ahead of it.
    void poison() {
        y_.fill_bytes(0xFF);
        hip_check(hipDeviceSynchronize(), "poison y");
    }

    [[nodiscard]] std::vector<std::uint32_t> output() const {
        const std::vector<std::uint16_t> y = y_.download();
        std::vector<std::uint32_t> bits(rows_ * kHidden);
        if (dtype_ == OutputDtype::Bf16) {
            for (std::size_t i = 0; i < y.size(); ++i) {
                bits[i] = static_cast<std::uint32_t>(y[i]) << 16;
            }
        } else {
            std::memcpy(bits.data(), y.data(), bits.size() * sizeof(std::uint32_t));
        }
        return bits;
    }

private:
    const Fixture& fixture_;
    const DeviceArray<std::uint8_t>& store_;
    const Inputs& inputs_;
    OutputDtype dtype_;
    std::size_t rows_;
    DeviceArray<std::int32_t> tokens_;
    DeviceArray<std::uint16_t> x_;
    DeviceArray<ops::MoeHelperRoute> routes_;
    DeviceArray<std::uint64_t> table_;
    DeviceArray<float> scales_;
    DeviceArray<std::uint16_t> y_;
};

float bits_float(std::uint32_t bits) {
    float value = 0.0F;
    std::memcpy(&value, &bits, 4);
    return value;
}

// Tokens whose rows the oracle checks: every routed token up to kOracleTokens, else
// kOracleTokens of them evenly spaced (the FP64 oracle costs about 5M multiply-adds per entry).
constexpr std::size_t kOracleTokens = 512;

// Checks y (bits [max_tokens][2560]): rows of tokens with active routes against the oracle, rows
// of tokens without one exactly +0, rows at or past N untouched.
int verify(const std::string& label, const Fixture& fixture, const BankState& bank,
           const Inputs& inputs, const std::vector<std::uint32_t>& bits,
           const std::vector<std::uint32_t>& bf16 = {}) {
    int failures = 0;
    // Expanded BF16 bits preserve exact comparison, including +0 and the unwritten poison.
    for (std::size_t i = 0; i < bf16.size(); ++i) {
        const auto expected =
            i < static_cast<std::size_t>(inputs.tokens) * kHidden
                ? static_cast<std::uint32_t>(bf16_bits(bf16_round(bits_float(bits[i])))) << 16
                : 0xFFFF0000U;
        if (bf16[i] != expected) {
            std::cerr << label << ": BF16 differs from one RNE of FP32 at " << i << '\n';
            ++failures;
            break;
        }
    }
    std::vector<int> all_routed;
    for (int t = 0; t < inputs.tokens; ++t) {
        if (has_active(inputs, t)) {
            all_routed.push_back(t);
            continue;
        }
        for (int n = 0; n < kHidden; ++n) {
            if (bits[static_cast<std::size_t>(t) * kHidden + n] != 0U) {
                std::cerr << label << ": token " << t << " has no route but y != +0\n";
                ++failures;
                break;
            }
        }
    }
    for (std::size_t i = static_cast<std::size_t>(inputs.tokens) * kHidden; i < bits.size(); ++i) {
        if (bits[i] != 0xFFFFFFFFU) {
            std::cerr << label << ": y row " << i / kHidden << " at or past N was written\n";
            ++failures;
            break;
        }
    }
    if (all_routed.empty()) { return failures; }
    // Rows the oracle samples out are still checked as written and finite.
    for (const int t : all_routed) {
        for (int n = 0; n < kHidden; ++n) {
            if (!std::isfinite(bits_float(bits[static_cast<std::size_t>(t) * kHidden + n]))) {
                std::cerr << label << ": token " << t << " has a non-finite or unwritten y\n";
                ++failures;
                break;
            }
        }
    }
    std::vector<int> routed;
    for (std::size_t i = 0; i < kOracleTokens && i < all_routed.size(); ++i) {
        routed.push_back(
            all_routed[i * all_routed.size() / std::min(all_routed.size(), kOracleTokens)]);
    }

    std::vector<std::vector<double>> expected(routed.size());
    {
        constexpr std::size_t kOracleThreads = 8;
        std::vector<std::jthread> workers;
        workers.reserve(kOracleThreads);
        for (std::size_t w = 0; w < kOracleThreads; ++w) {
            workers.emplace_back([&, w] {
                for (std::size_t i = w; i < routed.size(); i += kOracleThreads) {
                    expected[i] = oracle_token(fixture, bank, inputs, routed[i]);
                }
            });
        }
    }
    std::vector<double> got;
    std::vector<double> reference;
    for (std::size_t i = 0; i < routed.size(); ++i) {
        const auto row = static_cast<std::size_t>(routed[i]) * kHidden;
        for (std::size_t n = 0; n < kHidden; ++n) {
            got.push_back(bits_float(bits[row + n]));
            reference.push_back(expected[i][n]);
        }
    }
    const auto stats = test::compute_reduction_stats(got.data(), reference.data(),
                                                     static_cast<std::int64_t>(got.size()));
    const bool pass =
        test::reduction_passes(stats, static_cast<std::int64_t>(got.size()), kCriterion);
    std::printf("%-38s tokens=%zu/%zu rel_l2=%.3e (limit %.1e) max_abs=%.3e (limit %.3e) %s\n",
                label.c_str(), routed.size(), all_routed.size(), stats.relative_l2,
                kCriterion.relative_l2, stats.maximum_absolute_error,
                test::gross_error_limit(stats, kCriterion), pass ? "ok" : "FAIL");
    if (!pass) {
        std::cerr << label << ": criterion failed at index " << stats.maximum_error_index
                  << " actual=" << stats.actual_at_maximum
                  << " reference=" << stats.reference_at_maximum << '\n';
        ++failures;
    }
    if (!bf16.empty()) {
        got.clear();
        for (const int t : routed) {
            const auto row = static_cast<std::size_t>(t) * kHidden;
            for (std::size_t n = 0; n < kHidden; ++n) { got.push_back(bits_float(bf16[row + n])); }
        }
        const auto bf16_stats = test::compute_reduction_stats(
            got.data(), reference.data(), static_cast<std::int64_t>(got.size()));
        const bool bf16_pass = test::reduction_passes(
            bf16_stats, static_cast<std::int64_t>(got.size()), kBf16Criterion);
        std::printf("%-38s BF16 rel_l2=%.3e (limit %.3e) max_abs=%.3e (limit %.3e) %s\n",
                    label.c_str(), bf16_stats.relative_l2, kBf16Criterion.relative_l2,
                    bf16_stats.maximum_absolute_error,
                    test::gross_error_limit(bf16_stats, kBf16Criterion), bf16_pass ? "ok" : "FAIL");
        if (!bf16_pass) { ++failures; }
    }
    return failures;
}

int expect_rejected(const std::string& label, const std::function<void()>& call) {
    try {
        call();
    } catch (const std::invalid_argument&) { return 0; }
    std::cerr << label << ": invalid call was accepted\n";
    return 1;
}

bool select_r9700() {
    int count = 0;
    if (hipGetDeviceCount(&count) != hipSuccess) { return false; }
    for (int device = 0; device < count; ++device) {
        hipDeviceProp_t properties{};
        hip_check(hipGetDeviceProperties(&properties, device), "hipGetDeviceProperties");
        if (std::strncmp(properties.gcnArchName, "gfx1201", 7) == 0) {
            hip_check(hipSetDevice(device), "hipSetDevice");
            return true;
        }
    }
    return false;
}

class Stream {
public:
    Stream() { hip_check(hipStreamCreateWithFlags(&stream_, hipStreamNonBlocking), "stream"); }

    ~Stream() { (void)hipStreamDestroy(stream_); }

    Stream(const Stream&)            = delete;
    Stream& operator=(const Stream&) = delete;

    [[nodiscard]] hipStream_t get() const noexcept { return stream_; }

private:
    hipStream_t stream_ = nullptr;
};

// Calls of different extents, decode-round and prefill, sharing one workspace with no host
// synchronization between them: queued back to back on one stream, then captured into one HIP
// graph replayed three times. Each call must reproduce its standalone output bit for bit, so every
// call found the counters its predecessor left.
int queued_calls(const Fixture& fixture, const DeviceArray<std::uint8_t>& store,
                 ops::MoeHelperExpertsWorkspace& workspace, const BankState& bank,
                 OutputDtype prefill_dtype) {
    const Inputs a = make_inputs(bank, 20, 20, Routing::Skewed, 31U);
    const Inputs b = make_inputs(bank, 48, 48, Routing::Mixed48, 32U);
    const Inputs c = make_inputs(bank, 1, 4, Routing::Skewed, 33U);
    const Inputs d = make_inputs(bank, 200, 300, Routing::Skewed, 34U);
    std::vector<std::unique_ptr<Run>> runs;
    std::vector<std::vector<std::uint32_t>> expected;
    for (const Inputs* inputs : {&a, &b, &c, &d}) {
        const auto dtype =
            inputs->max_tokens > ops::kMoeHelperMaxTokens ? prefill_dtype : OutputDtype::Fp32;
        runs.push_back(std::make_unique<Run>(fixture, store, *inputs, dtype));
        runs.back()->set_bank(bank);
        expected.push_back(runs.back()->invoke(workspace));
    }
    const Stream stream;
    // a, d, b, c, d, a: repeated calls overwrite their outputs with the same bits.
    const int order[]  = {0, 3, 1, 2, 3, 0};
    const auto enqueue = [&] {
        for (const int r : order) {
            ops::moe_helper_experts(runs[static_cast<std::size_t>(r)]->args(), workspace,
                                    stream.get());
        }
    };
    int failures     = 0;
    const auto check = [&](const std::string& label) {
        for (std::size_t r = 0; r < runs.size(); ++r) {
            if (runs[r]->output() != expected[r]) {
                std::cerr << label << ": call " << r << " differs from its standalone output\n";
                ++failures;
            }
        }
    };
    for (const auto& run : runs) { run->poison(); }
    enqueue();
    hip_check(hipStreamSynchronize(stream.get()), "queued calls");
    check("queued calls");

    hipGraph_t graph = nullptr;
    hip_check(hipStreamBeginCapture(stream.get(), hipStreamCaptureModeThreadLocal), "capture");
    enqueue();
    hip_check(hipStreamEndCapture(stream.get(), &graph), "end capture");
    hipGraphExec_t exec           = nullptr;
    const hipError_t instantiated = hipGraphInstantiate(&exec, graph, nullptr, nullptr, 0);
    (void)hipGraphDestroy(graph);
    hip_check(instantiated, "graph instantiate");
    for (int replay = 0; replay < 3; ++replay) {
        for (const auto& run : runs) { run->poison(); }
        hip_check(hipGraphLaunch(exec, stream.get()), "graph launch");
        hip_check(hipStreamSynchronize(stream.get()), "graph replay");
        check("graph replay " + std::to_string(replay));
    }
    (void)hipGraphExecDestroy(exec);
    return failures;
}

// At prefill extents a token's row depends only on its own inputs: reversing the tokens' order
// (x rows and route rows) must reverse y's rows bit for bit.
int order_independence(const Fixture& fixture, const DeviceArray<std::uint8_t>& store,
                       ops::MoeHelperExpertsWorkspace& workspace, const BankState& bank,
                       OutputDtype dtype) {
    const Inputs forward = make_inputs(bank, 300, 300, Routing::Skewed, 41U);
    Inputs reversed      = forward;
    const int tokens     = forward.tokens;
    for (int t = 0; t < tokens; ++t) {
        const auto from = static_cast<std::size_t>(tokens - 1 - t);
        const auto to   = static_cast<std::size_t>(t);
        std::copy_n(forward.x.begin() + static_cast<std::ptrdiff_t>(from * kHidden), kHidden,
                    reversed.x.begin() + static_cast<std::ptrdiff_t>(to * kHidden));
        std::copy_n(forward.routes.begin() + static_cast<std::ptrdiff_t>(from * kRoutes), kRoutes,
                    reversed.routes.begin() + static_cast<std::ptrdiff_t>(to * kRoutes));
    }
    Run a(fixture, store, forward, dtype);
    Run b(fixture, store, reversed, dtype);
    a.set_bank(bank);
    b.set_bank(bank);
    const std::vector<std::uint32_t> ya = a.invoke(workspace);
    const std::vector<std::uint32_t> yb = b.invoke(workspace);
    for (int t = 0; t < tokens; ++t) {
        const auto row_a = static_cast<std::size_t>(t) * kHidden;
        const auto row_b = static_cast<std::size_t>(tokens - 1 - t) * kHidden;
        if (!std::equal(ya.begin() + static_cast<std::ptrdiff_t>(row_a),
                        ya.begin() + static_cast<std::ptrdiff_t>(row_a + kHidden),
                        yb.begin() + static_cast<std::ptrdiff_t>(row_b))) {
            std::cerr << "order independence: token " << t << " differs after reversal\n";
            return 1;
        }
    }
    std::printf("%-38s tokens=%d rows bit-identical (%s)\n", "T300 reversed token order", tokens,
                dtype == OutputDtype::Bf16 ? "BF16" : "FP32");
    return 0;
}

int run() {
    if (!select_r9700()) {
        std::printf("SKIP: no gfx1201 device\n");
        return kSkip;
    }
    int failures       = 0;
    bool round_trip    = true;
    const auto fixture = make_fixture(round_trip);
    if (!round_trip) {
        std::cerr << "record layout round trip failed\n";
        return 1;
    }
    DeviceArray<std::uint8_t> store(fixture->store_bytes);
    for (std::size_t r = 0; r < fixture->experts.size(); ++r) {
        const auto& record = fixture->experts[r].record;
        store.upload(record.data(), record.size(), fixture->offsets[r]);
    }
    // A decode-round workspace, and a prefill workspace that also serves decode-round extents.
    ops::MoeHelperExpertsWorkspace workspace(ops::kMoeHelperMaxTokens);
    ops::MoeHelperExpertsWorkspace prefill(kPrefillCapacity);
    const BankState wide      = make_bank(RecordSet::Wide, 7U);
    const BankState subnormal = make_bank(RecordSet::Subnormal, 8U);

    struct Case {
        const char* label;
        const BankState* bank;
        int tokens;
        int max_tokens;
        Routing routing;
        std::uint32_t seed;
        ops::MoeHelperExpertsWorkspace* workspace;
    };

    const Case cases[] = {
        {"T1", &wide, 1, 1, Routing::Skewed, 11U, &workspace},
        {"T2 of max 4", &wide, 2, 4, Routing::Skewed, 12U, &workspace},
        {"T4", &wide, 4, 4, Routing::Skewed, 13U, &workspace},
        {"T20", &wide, 20, 20, Routing::Skewed, 14U, &workspace},
        {"T48 mixed ownership and sharing", &wide, 48, 48, Routing::Mixed48, 15U, &workspace},
        {"T4 subnormal scales", &subnormal, 4, 4, Routing::Skewed, 16U, &workspace},
        // Tokens per expert: 2 and 3 (SIMT), 6 (WMMA at its threshold, one partial block), 17
        // (WMMA, a full block and one token), 40 (WMMA, blocks of 16, 16 and 8).
        {"T2 concentrated", &wide, 2, 2, Routing::Concentrated, 18U, &workspace},
        {"T3 concentrated", &wide, 3, 3, Routing::Concentrated, 19U, &workspace},
        {"T6 concentrated", &wide, 6, 6, Routing::Concentrated, 23U, &workspace},
        {"T17 concentrated", &wide, 17, 17, Routing::Concentrated, 24U, &workspace},
        {"T40 concentrated", &wide, 40, 40, Routing::Concentrated, 20U, &workspace},
        {"T20 concentrated subnormal", &subnormal, 20, 20, Routing::Concentrated, 22U, &workspace},
        {"N0 of max 48", &wide, 0, 48, Routing::Skewed, 17U, &workspace},
        // Prefill extents (max_tokens > 48) on the prefill workspace, after a decode-round call
        // on it: the route boundary at max_tokens 48 / 49, specials (no helper route, only helper
        // routes, a repeated expert, ids >= 512), jobs of 128 + 128 + 44 tokens per expert,
        // subnormal scales, N below max_tokens, and N = 0.
        {"T48 on the prefill workspace", &wide, 48, 48, Routing::Mixed48, 15U, &prefill},
        {"T49 (prefill boundary)", &wide, 49, 49, Routing::Skewed, 25U, &prefill},
        {"T64 mixed ownership and sharing", &wide, 64, 64, Routing::Mixed48, 26U, &prefill},
        {"T300 concentrated", &wide, 300, 300, Routing::Concentrated, 27U, &prefill},
        {"T200 subnormal scales", &subnormal, 200, 200, Routing::Skewed, 28U, &prefill},
        {"T130 concentrated subnormal", &subnormal, 130, 130, Routing::Concentrated, 29U, &prefill},
        {"T2000 of max 4096", &wide, 2000, kPrefillCapacity, Routing::Skewed, 30U, &prefill},
        {"N0 of max 4096", &wide, 0, kPrefillCapacity, Routing::Skewed, 31U, &prefill},
    };
    const auto check_case = [&](const Case& c) {
        const Inputs inputs = make_inputs(*c.bank, c.tokens, c.max_tokens, c.routing, c.seed);
        std::vector<std::uint32_t> fp32;
        std::vector<std::uint32_t> bf16;
        for (const auto dtype : {OutputDtype::Fp32, OutputDtype::Bf16}) {
            if (dtype == OutputDtype::Bf16 && c.max_tokens <= ops::kMoeHelperMaxTokens) {
                continue;
            }
            Run run(*fixture, store, inputs, dtype);
            run.set_bank(*c.bank);
            auto& bits = dtype == OutputDtype::Bf16 ? bf16 : fp32;
            bits       = run.invoke(*c.workspace);
            // Every dtype must find reset counters and reproduce y, including at capacity.
            if (run.invoke(*c.workspace) != bits) {
                std::cerr << c.label << ": repeated call is not bit-identical\n";
                ++failures;
            }
        }
        failures += verify(c.label, *fixture, *c.bank, inputs, fp32, bf16);
    };
    for (const Case& c : cases) { check_case(c); }

    for (const auto dtype : {OutputDtype::Fp32, OutputDtype::Bf16}) {
        failures += queued_calls(*fixture, store, prefill, wide, dtype);
        failures += order_independence(*fixture, store, prefill, wide, dtype);
    }
    {
        // The Op's capacity: every token routed, on a workspace of exactly that capacity.
        ops::MoeHelperExpertsWorkspace full(ops::kMoeHelperExpertsMaxTokens);
        check_case({"T16384 (capacity)", &wide, ops::kMoeHelperExpertsMaxTokens,
                    ops::kMoeHelperExpertsMaxTokens, Routing::Skewed, 32U, &full});
    }

    // Contract rejection before any work is queued.
    const Inputs inputs = make_inputs(wide, 4, 4, Routing::Skewed, 21U);
    Run valid(*fixture, store, inputs);
    valid.set_bank(wide);
    ops::MoeHelperExpertsWorkspace small(2);
    failures += expect_rejected("max_tokens 0", [&] {
        auto args       = valid.args();
        args.max_tokens = 0;
        ops::moe_helper_experts(args, workspace, nullptr);
    });
    failures += expect_rejected("max_tokens 16385", [&] {
        auto args       = valid.args();
        args.max_tokens = ops::kMoeHelperExpertsMaxTokens + 1;
        ops::moe_helper_experts(args, prefill, nullptr);
    });
    failures += expect_rejected("BF16 output on decode extent", [&] {
        auto args         = valid.args();
        args.output_dtype = OutputDtype::Bf16;
        ops::moe_helper_experts(args, workspace, nullptr);
    });
    failures += expect_rejected("workspace too small",
                                [&] { ops::moe_helper_experts(valid.args(), small, nullptr); });
    failures += expect_rejected("prefill extent on a decode-round workspace", [&] {
        auto args       = valid.args();
        args.max_tokens = 49;
        ops::moe_helper_experts(args, workspace, nullptr);
    });
    failures += expect_rejected("workspace max_tokens 16385", [&] {
        ops::MoeHelperExpertsWorkspace too_large(ops::kMoeHelperExpertsMaxTokens + 1);
    });
    failures += expect_rejected("misaligned y", [&] {
        auto args = valid.args();
        args.y    = static_cast<std::uint16_t*>(args.y) + 1;
        ops::moe_helper_experts(args, workspace, nullptr);
    });
    failures += expect_rejected("y aliases x", [&] {
        auto args = valid.args();
        args.y    = const_cast<std::uint16_t*>(args.x);
        ops::moe_helper_experts(args, workspace, nullptr);
    });

    std::printf("%s\n", failures == 0 ? "PASS" : "FAIL");
    return failures == 0 ? 0 : 1;
}

} // namespace

int main() {
    try {
        return run();
    } catch (const std::exception& error) {
        std::cerr << "moe_helper_experts test: " << error.what() << '\n';
        return 1;
    }
}
