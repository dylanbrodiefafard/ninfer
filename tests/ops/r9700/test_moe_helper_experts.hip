// moe_helper_experts Op test on the R9700 (gfx1201) against an independent FP64 oracle at the
// Qwen3.8-Flash-Next record geometry, for decode-round extents (max_tokens <= 48) and prefill
// extents up to the Op's capacity, in both activation profiles (A16 and, at prefill extents, A8),
// plus an exact check of the A8 row quantizer the grouped route applies to the down input.
//
// Test experts are canonical codes and E4M3 scales written into the documented record layout by
// the test-side packer and read back exactly by an independent reader
// (ops/moe_expert_record_reference.h). The oracle decodes each code (through the matrix's F3 book
// or as E2M1) times its exact E4M3 scale, applies the FP32 matrix scales and evaluates the routed
// SwiGLU sum in FP64 from the represented BF16 x and FP32 route weights; it does not quantize.
// A8 calls take x as its E4M3 rows, made here by a host implementation of the shared A8 rule.
#include "ninfer/ops/moe_helper_experts.h"
#include "ops/moe_expert_record_reference.h"
#include "ops/op_check.h"
#include "ops/r9700/moe_helper_experts_grouped.h"

#include <hip/hip_runtime.h>

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <exception>
#include <functional>
#include <iostream>
#include <limits>
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

constexpr test::ReductionCriterion bf16_criterion(const test::ReductionCriterion& fp32) {
    return {fp32.relative_l2 + kBf16UnitRoundoff * (1.0 + fp32.relative_l2), 0.0,
            fp32.gross_relative_to_max_reference +
                kBf16UnitRoundoff * (1.0 + fp32.gross_relative_to_max_reference)};
}

// A8 criterion. Two E4M3 roundings join the terms above: x enters the gate/up products, and the
// BF16 gate/up product the down projection, as codes q with decode(q) s = v (1 + d), |d| <= u =
// 2^-4 in E4M3's normal range (the row maximum maps to 448, so only elements below 2^-6 s, under
// 2^-14.8 of the row maximum, fall below it, with absolute error at most 2^-10 s and negligible
// energy). For values spread over a binade the rounding error is uniform in half a quantum,
// 2^(e-4) for v in [2^e, 2^(e+1)), so its energy relative to the values' is
// (2^-6 / 12) / (7 / 3): relative RMS e_q = 2 u / sqrt(28) ~= 0.378 u ~= 0.024.
//   gate/up input: dx, of relative RMS e_q, reaches G x and U x as independent errors (G and U
//     rows are different projections) of relative RMS e_q each. In h = SiLU(g) u, the error
//     energy is e_q^2 (1 + E[g^2] E[SiLU'(g)^2] / E[SiLU(g)^2]) relative to h's; the ratio is 1
//     both for small g (SiLU ~ g / 2) and for large g (SiLU ~ ReLU), so h carries sqrt(2) e_q.
//   down input: the row quantization adds an independent e_q, so D h, each route's contribution,
//     and the token's weighted sum of independently erring routes carry sqrt(3) e_q ~= 0.041.
// The BF16 product rounding (1.5e-3) adds in quadrature (0.04 percent). As for A16 the bound
// allows twice the expected relative L2, and the gross cap 2.7 times it (six deviations of an
// error RMS of e rms(y) ~= e max|y| / 4); every sampled token's own relative L2 must also stay
// below 1.5 times the bound, so a token with a misapplied scale cannot hide in the aggregate.
constexpr double kE4m3UnitRoundoff = 0x1p-4;
constexpr double kA8Expected       = 1.7320508 * 2.0 * kE4m3UnitRoundoff / 5.2915026; // sqrt3 e_q
constexpr test::ReductionCriterion kA8Criterion{/*relative_l2*/ 2.0 * kA8Expected,
                                                /*gross_absolute*/ 0.0,
                                                /*gross_relative_to_max_reference*/ 2.7 *
                                                    kA8Expected};
constexpr double kRowFactor = 1.5;

using OutputDtype = ops::MoeHelperOutputDtype;
using Profile     = ops::MoeHelperActivationProfile;

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

float bf16_value(std::uint16_t bits) {
    const std::uint32_t word = static_cast<std::uint32_t>(bits) << 16;
    float value              = 0.0F;
    std::memcpy(&value, &word, 4);
    return value;
}

// --- Host A8 rule (independent of the device's conversion instruction) -------------------------

// Non-negative finite E4M3FN values in code order 0x00..0x7E: 2^-9 m below code 8, else
// (1 + m / 8) 2^(e - 7), e = code / 8, m = code % 8.
const std::array<double, 127>& e4m3_values() {
    static const std::array<double, 127> values = [] {
        std::array<double, 127> table{};
        for (int code = 0; code < 127; ++code) {
            const int e = code / 8;
            const int m = code % 8;
            table[static_cast<std::size_t>(code)] =
                e == 0 ? std::ldexp(m, -9) : std::ldexp(8 + m, e - 10);
        }
        return table;
    }();
    return values;
}

// E4M3FN code of a finite FP32 value: nearest value by exact comparison against the midpoint of
// its neighbors, ties to the even code, magnitudes at or above 448 saturated to 448 (0x7E); the
// sign bit is the value's (0x80 for -0 and negative underflow).
std::uint8_t e4m3_code(float value) {
    const auto& values       = e4m3_values();
    const std::uint8_t sign  = std::signbit(value) ? 0x80 : 0x00;
    const double magnitude   = std::fabs(static_cast<double>(value));
    if (magnitude >= values[126]) { return static_cast<std::uint8_t>(sign | 0x7E); }
    const auto above = std::upper_bound(values.begin(), values.end(), magnitude);
    const auto lower = static_cast<int>(above - values.begin()) - 1;
    const double low  = values[static_cast<std::size_t>(lower)];
    const double high = values[static_cast<std::size_t>(lower + 1)];
    const double mid  = 0.5 * (low + high); // exact: E4M3 values have at most four bits
    int code          = lower;
    if (magnitude > mid || (magnitude == mid && lower % 2 == 1)) { code = lower + 1; }
    return static_cast<std::uint8_t>(sign | code);
}

struct E4m3Rows {
    std::vector<std::uint8_t> codes;
    std::vector<float> scales;
};

// The shared A8 rule on BF16 rows [rows][width]: s = max|x| / 448 in FP32 (+0 for a zero row,
// whose codes are all +0), code = E4M3FN(x / s) with IEEE FP32 division.
E4m3Rows quantize_rows(const float* x, std::size_t rows, std::size_t width) {
    E4m3Rows out{std::vector<std::uint8_t>(rows * width), std::vector<float>(rows)};
    for (std::size_t r = 0; r < rows; ++r) {
        const float* row = x + r * width;
        float maximum    = 0.0F;
        for (std::size_t k = 0; k < width; ++k) { maximum = std::max(maximum, std::fabs(row[k])); }
        const float scale = maximum / 448.0F;
        out.scales[r]     = scale;
        for (std::size_t k = 0; k < width; ++k) {
            out.codes[r * width + k] = maximum == 0.0F ? 0 : e4m3_code(row[k] / scale);
        }
    }
    return out;
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
        OutputDtype dtype = OutputDtype::Fp32, Profile profile = Profile::A16)
        : fixture_(fixture), store_(store), inputs_(inputs), dtype_(dtype), profile_(profile),
          rows_(static_cast<std::size_t>(inputs.max_tokens)), tokens_(1),
          x_(profile == Profile::A16 ? rows_ * kHidden : 1),
          x_codes_(profile == Profile::A8 ? rows_ * kHidden : 1),
          x_scales_(profile == Profile::A8 ? rows_ : 1), routes_(rows_ * kRoutes),
          table_(kExperts), scales_(kExperts * 3),
          y_(rows_ * kHidden * (dtype == OutputDtype::Bf16 ? 1 : 2)) {
        if (profile == Profile::A8) {
            const E4m3Rows rows = quantize_rows(inputs.x.data(), rows_, kHidden);
            x_codes_.upload(rows.codes);
            x_scales_.upload(rows.scales);
        } else {
            std::vector<std::uint16_t> x_bits(inputs.x.size());
            for (std::size_t i = 0; i < x_bits.size(); ++i) { x_bits[i] = bf16_bits(inputs.x[i]); }
            x_.upload(x_bits);
        }
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
        args.tokens     = tokens_.data();
        args.max_tokens = inputs_.max_tokens;
        args.activation = profile_;
        if (profile_ == Profile::A8) {
            args.x_e4m3 = {x_codes_.data(), x_scales_.data()};
        } else {
            args.x = x_.data();
        }
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
    Profile profile_;
    std::size_t rows_;
    DeviceArray<std::int32_t> tokens_;
    DeviceArray<std::uint16_t> x_;
    DeviceArray<std::uint8_t> x_codes_;
    DeviceArray<float> x_scales_;
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

// One profile's outputs of a case: the FP32 y bits and, at prefill extents, the BF16 y bits (each
// element expanded to FP32 bits), with the profile's FP32 criterion.
struct Output {
    const char* profile;
    std::vector<std::uint32_t> fp32;
    std::vector<std::uint32_t> bf16; // empty at decode extents
    test::ReductionCriterion criterion;
};

// Structural checks of y (bits [max_tokens][2560]): rows of tokens without an active route exactly
// +0, rows at or past N still `poison` (the expanded bits of the unwritten fill), every routed row
// finite.
int check_structure(const std::string& label, const Inputs& inputs,
                    const std::vector<std::uint32_t>& bits, const std::vector<int>& all_routed,
                    std::uint32_t poison) {
    int failures = 0;
    for (int t = 0; t < inputs.tokens; ++t) {
        if (has_active(inputs, t)) { continue; }
        for (int n = 0; n < kHidden; ++n) {
            if (bits[static_cast<std::size_t>(t) * kHidden + n] != 0U) {
                std::cerr << label << ": token " << t << " has no route but y != +0\n";
                ++failures;
                break;
            }
        }
    }
    for (std::size_t i = static_cast<std::size_t>(inputs.tokens) * kHidden; i < bits.size(); ++i) {
        if (bits[i] != poison) {
            std::cerr << label << ": y row " << i / kHidden << " at or past N was written\n";
            ++failures;
            break;
        }
    }
    for (const int t : all_routed) {
        for (int n = 0; n < kHidden; ++n) {
            if (!std::isfinite(bits_float(bits[static_cast<std::size_t>(t) * kHidden + n]))) {
                std::cerr << label << ": token " << t << " has a non-finite or unwritten y\n";
                ++failures;
                break;
            }
        }
    }
    return failures;
}

// Checks each output against the FP64 oracle of the case's sampled tokens (computed once): the
// aggregate criterion, every sampled token's own relative L2 against kRowFactor times the bound,
// and for BF16 outputs one RNE of the same profile's FP32 output bit for bit.
int verify(const std::string& label, const Fixture& fixture, const BankState& bank,
           const Inputs& inputs, const std::vector<Output>& outputs) {
    int failures = 0;
    std::vector<int> all_routed;
    for (int t = 0; t < inputs.tokens; ++t) {
        if (has_active(inputs, t)) { all_routed.push_back(t); }
    }
    for (const Output& output : outputs) {
        const std::string name = label + " " + output.profile;
        failures += check_structure(name + " FP32", inputs, output.fp32, all_routed, 0xFFFFFFFFU);
        if (output.bf16.empty()) { continue; }
        failures += check_structure(name + " BF16", inputs, output.bf16, all_routed, 0xFFFF0000U);
        // Expanded BF16 bits preserve exact comparison, including +0.
        const std::size_t written = static_cast<std::size_t>(inputs.tokens) * kHidden;
        for (std::size_t i = 0; i < written; ++i) {
            const auto expected = static_cast<std::uint32_t>(
                                      bf16_bits(bf16_round(bits_float(output.fp32[i]))))
                                  << 16;
            if (output.bf16[i] != expected) {
                std::cerr << name << ": BF16 differs from one RNE of FP32 at " << i << '\n';
                ++failures;
                break;
            }
        }
    }
    if (all_routed.empty()) { return failures; }
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
    std::vector<double> reference;
    for (const auto& row : expected) { reference.insert(reference.end(), row.begin(), row.end()); }

    const auto check = [&](const std::string& name, const std::vector<std::uint32_t>& bits,
                           const test::ReductionCriterion& criterion) {
        std::vector<double> got;
        got.reserve(reference.size());
        for (const int t : routed) {
            const auto row = static_cast<std::size_t>(t) * kHidden;
            for (std::size_t n = 0; n < kHidden; ++n) { got.push_back(bits_float(bits[row + n])); }
        }
        const auto stats = test::compute_reduction_stats(got.data(), reference.data(),
                                                         static_cast<std::int64_t>(got.size()));
        bool pass = test::reduction_passes(stats, static_cast<std::int64_t>(got.size()), criterion);
        double worst_row = 0.0;
        for (std::size_t i = 0; i < routed.size(); ++i) {
            const auto row = test::compute_reduction_stats(got.data() + i * kHidden,
                                                           reference.data() + i * kHidden, kHidden);
            worst_row      = std::max(worst_row, row.relative_l2);
        }
        pass = pass && worst_row <= kRowFactor * criterion.relative_l2;
        std::printf("%-44s tokens=%zu/%zu rel_l2=%.3e (limit %.2e) row<=%.3e max_abs=%.3e "
                    "(limit %.3e) %s\n",
                    name.c_str(), routed.size(), all_routed.size(), stats.relative_l2,
                    criterion.relative_l2, worst_row, stats.maximum_absolute_error,
                    test::gross_error_limit(stats, criterion), pass ? "ok" : "FAIL");
        if (!pass) {
            std::cerr << name << ": criterion failed at index " << stats.maximum_error_index
                      << " actual=" << stats.actual_at_maximum
                      << " reference=" << stats.reference_at_maximum << '\n';
            ++failures;
        }
    };
    for (const Output& output : outputs) {
        const std::string name = label + " " + output.profile;
        check(name, output.fp32, output.criterion);
        if (!output.bf16.empty()) {
            check(name + " BF16", output.bf16, bf16_criterion(output.criterion));
        }
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

// Calls of different extents and profiles, decode-round and prefill, sharing one workspace with no
// host synchronization between them: queued back to back on one stream, then captured into one HIP
// graph replayed three times. Each call must reproduce its standalone output bit for bit, so every
// call found the counters its predecessor left.
int queued_calls(const Fixture& fixture, const DeviceArray<std::uint8_t>& store,
                 ops::MoeHelperExpertsWorkspace& workspace, const BankState& bank,
                 OutputDtype prefill_dtype) {
    const Inputs a = make_inputs(bank, 20, 20, Routing::Skewed, 31U);
    const Inputs b = make_inputs(bank, 48, 48, Routing::Mixed48, 32U);
    const Inputs c = make_inputs(bank, 1, 4, Routing::Skewed, 33U);
    const Inputs d = make_inputs(bank, 200, 300, Routing::Skewed, 34U);
    const Inputs e = make_inputs(bank, 250, 260, Routing::Skewed, 35U);
    std::vector<std::unique_ptr<Run>> runs;
    std::vector<std::vector<std::uint32_t>> expected;
    for (const Inputs* inputs : {&a, &b, &c, &d, &e}) {
        const bool prefill = inputs->max_tokens > ops::kMoeHelperMaxTokens;
        runs.push_back(std::make_unique<Run>(fixture, store, *inputs,
                                             prefill ? prefill_dtype : OutputDtype::Fp32,
                                             inputs == &e ? Profile::A8 : Profile::A16));
        runs.back()->set_bank(bank);
        expected.push_back(runs.back()->invoke(workspace));
    }
    const Stream stream;
    // a, d, e, b, c, e, d, a: repeated calls overwrite their outputs with the same bits.
    const int order[]  = {0, 3, 4, 1, 2, 4, 3, 0};
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
// (x rows and route rows; for A8 the E4M3 rows follow, being per token) must reverse y's rows bit
// for bit.
int order_independence(const Fixture& fixture, const DeviceArray<std::uint8_t>& store,
                       ops::MoeHelperExpertsWorkspace& workspace, const BankState& bank,
                       OutputDtype dtype, Profile profile) {
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
    Run a(fixture, store, forward, dtype, profile);
    Run b(fixture, store, reversed, dtype, profile);
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
    std::printf("%-44s tokens=%d rows bit-identical (%s %s)\n", "T300 reversed token order",
                tokens, profile == Profile::A8 ? "A8" : "A16",
                dtype == OutputDtype::Bf16 ? "BF16" : "FP32");
    return 0;
}

// The A8 row quantizer the grouped route applies to its down input (the production kernel, through
// its launcher), against the host rule bit for bit on rows of 640 BF16:
//   zero rows (+0, -0, mixed): scale +0 and every code +0;
//   every finite BF16 of magnitude <= 448 next to 448 (s = 1, so the quotient is x exactly): every
//     conversion case of a BF16 value, including ties, E4M3 subnormals and underflow to -0;
//   every finite BF16 in pattern order, 640 per row: general quotients;
//   one row per positive finite BF16 maximum m (m, -m and m times BF16 fractions): every scale,
//     including FP32-subnormal ones, and quotients above 448 that saturate;
//   SwiGLU-like products at three magnitudes.
// The last row lies past the device row count and must stay unwritten.
int quantizer_exact() {
    constexpr std::size_t kWidth = ops::kMoeIntermediate;
    std::vector<float> x;
    std::vector<float> row;
    const auto flush = [&] {
        row.resize(kWidth, 0.0F);
        x.insert(x.end(), row.begin(), row.end());
        row.clear();
    };
    const auto push = [&](float value) {
        row.push_back(value);
        if (row.size() == kWidth) { flush(); }
    };
    for (const float zero : {0.0F, -0.0F}) {
        row.assign(kWidth, zero);
        flush();
    }
    for (std::size_t k = 0; k < kWidth; ++k) { push(k % 3 == 0 ? -0.0F : 0.0F); }

    const auto finite = [](std::uint16_t bits) { return (bits & 0x7F80U) != 0x7F80U; };
    for (std::uint32_t bits = 0; bits <= 0xFFFFU; ++bits) {
        const float value = bf16_value(static_cast<std::uint16_t>(bits));
        if (!finite(static_cast<std::uint16_t>(bits)) || std::fabs(value) > 448.0F) { continue; }
        if (row.empty()) { row.push_back(448.0F); }
        push(value);
    }
    if (!row.empty()) { flush(); }
    for (std::uint32_t bits = 0; bits <= 0xFFFFU; ++bits) {
        if (finite(static_cast<std::uint16_t>(bits))) {
            push(bf16_value(static_cast<std::uint16_t>(bits)));
        }
    }
    if (!row.empty()) { flush(); }
    std::mt19937 rng(20261119U);
    std::uniform_real_distribution<float> fraction(-1.0F, 1.0F);
    std::uniform_int_distribution<int> shift(0, 24);
    for (std::uint32_t bits = 1; bits < 0x7F80U; ++bits) {
        const float m = bf16_value(static_cast<std::uint16_t>(bits));
        row           = {m, -m};
        while (row.size() < kWidth) {
            row.push_back(bf16_round(m * std::ldexp(fraction(rng), -shift(rng))));
        }
        flush();
    }
    std::normal_distribution<float> normal(0.0F, 1.0F);
    for (const float sigma : {1.0e-3F, 1.0F, 30.0F}) {
        for (int r = 0; r < 512; ++r) {
            for (std::size_t k = 0; k < kWidth; ++k) {
                const float g = sigma * normal(rng);
                push(bf16_round(g / (1.0F + std::exp(-g)) * sigma * normal(rng)));
            }
        }
    }
    const std::size_t rows = x.size() / kWidth;
    const std::size_t live = rows - 1;
    const E4m3Rows host    = quantize_rows(x.data(), live, kWidth);

    std::vector<std::uint16_t> x_bits(x.size());
    for (std::size_t i = 0; i < x.size(); ++i) { x_bits[i] = bf16_bits(x[i]); }
    DeviceArray<std::uint16_t> x_device(x.size());
    DeviceArray<std::uint32_t> rows_device(1);
    DeviceArray<std::uint8_t> codes(x.size());
    DeviceArray<float> scales(rows);
    x_device.upload(x_bits);
    const auto live_rows = static_cast<std::uint32_t>(live);
    rows_device.upload(&live_rows, 1);
    codes.fill_bytes(0xFF);
    scales.fill_bytes(0xFF);
    int device = 0;
    hip_check(hipGetDevice(&device), "hipGetDevice");
    ops::detail::moe_helper_quantize_rows(x_device.data(), rows_device.data(),
                                          static_cast<std::int32_t>(rows), codes.data(),
                                          scales.data(), device, nullptr);
    hip_check(hipDeviceSynchronize(), "quantize rows");
    const std::vector<std::uint8_t> got_codes = codes.download();
    const std::vector<float> got_scales       = scales.download();

    int failures = 0;
    for (std::size_t r = 0; r < live && failures < 8; ++r) {
        if (std::bit_cast<std::uint32_t>(got_scales[r]) !=
            std::bit_cast<std::uint32_t>(host.scales[r])) {
            std::cerr << "quantizer: row " << r << " scale " << got_scales[r] << " expected "
                      << host.scales[r] << '\n';
            ++failures;
        }
        for (std::size_t k = 0; k < kWidth; ++k) {
            const std::size_t i = r * kWidth + k;
            if (got_codes[i] != host.codes[i]) {
                std::fprintf(stderr, "quantizer: row %zu element %zu x=%a code 0x%02X expected 0x%02X\n",
                             r, k, static_cast<double>(x[i]), got_codes[i], host.codes[i]);
                ++failures;
                break;
            }
        }
    }
    if (std::bit_cast<std::uint32_t>(got_scales[live]) != 0xFFFFFFFFU ||
        std::any_of(got_codes.begin() + static_cast<std::ptrdiff_t>(live * kWidth),
                    got_codes.end(), [](std::uint8_t code) { return code != 0xFF; })) {
        std::cerr << "quantizer: a row past the row count was written\n";
        ++failures;
    }

    // The cases the rows must reach: saturating quotients, exact ties, E4M3 subnormal codes and
    // FP32-subnormal scales.
    const auto& values       = e4m3_values();
    std::size_t saturated    = 0;
    std::size_t ties         = 0;
    std::size_t subnormal    = 0;
    std::size_t tiny_scales  = 0;
    for (std::size_t r = 0; r < live; ++r) {
        const float scale = host.scales[r];
        if (scale != 0.0F && scale < std::numeric_limits<float>::min()) { ++tiny_scales; }
        if (scale == 0.0F) { continue; }
        for (std::size_t k = 0; k < kWidth; ++k) {
            const double q = std::fabs(static_cast<double>(x[r * kWidth + k] / scale));
            if (q > 448.0) { ++saturated; }
            const std::uint8_t code = host.codes[r * kWidth + k] & 0x7FU;
            if (code > 0 && code < 8) { ++subnormal; }
            const auto above = std::upper_bound(values.begin(), values.end(), q);
            if (above != values.begin() && above != values.end() &&
                q == 0.5 * (*(above - 1) + *above)) {
                ++ties;
            }
        }
    }
    std::printf("%-44s rows=%zu saturated=%zu ties=%zu e4m3_subnormal=%zu fp32_subnormal_scales=%zu "
                "%s\n",
                "A8 down-input quantizer (bit-exact)", live, saturated, ties, subnormal, tiny_scales,
                failures == 0 ? "ok" : "FAIL");
    if (saturated == 0 || ties == 0 || subnormal == 0 || tiny_scales == 0) {
        std::cerr << "quantizer: a required case was not reached\n";
        ++failures;
    }
    return failures;
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
    failures += quantizer_exact();
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
        const Inputs inputs  = make_inputs(*c.bank, c.tokens, c.max_tokens, c.routing, c.seed);
        const bool prefill   = c.max_tokens > ops::kMoeHelperMaxTokens;
        std::vector<Output> outputs;
        for (const auto profile : {Profile::A16, Profile::A8}) {
            if (profile == Profile::A8 && !prefill) { continue; }
            Output output{profile == Profile::A8 ? "A8" : "A16", {}, {},
                          profile == Profile::A8 ? kA8Criterion : kCriterion};
            for (const auto dtype : {OutputDtype::Fp32, OutputDtype::Bf16}) {
                if (dtype == OutputDtype::Bf16 && !prefill) { continue; }
                Run run(*fixture, store, inputs, dtype, profile);
                run.set_bank(*c.bank);
                auto& bits = dtype == OutputDtype::Bf16 ? output.bf16 : output.fp32;
                bits       = run.invoke(*c.workspace);
                // Every call must find reset counters and reproduce y, including at capacity.
                if (run.invoke(*c.workspace) != bits) {
                    std::cerr << c.label << " " << output.profile
                              << ": repeated call is not bit-identical\n";
                    ++failures;
                }
            }
            outputs.push_back(std::move(output));
        }
        failures += verify(c.label, *fixture, *c.bank, inputs, outputs);
    };
    for (const Case& c : cases) { check_case(c); }

    for (const auto dtype : {OutputDtype::Fp32, OutputDtype::Bf16}) {
        failures += queued_calls(*fixture, store, prefill, wide, dtype);
        for (const auto profile : {Profile::A16, Profile::A8}) {
            failures += order_independence(*fixture, store, prefill, wide, dtype, profile);
        }
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
    // A8 serves only prefill extents and takes exactly its own input form.
    const Inputs prefill_inputs = make_inputs(wide, 60, 64, Routing::Skewed, 22U);
    Run a8(*fixture, store, prefill_inputs, OutputDtype::Bf16, Profile::A8);
    a8.set_bank(wide);
    failures += expect_rejected("A8 on decode extent", [&] {
        auto args       = a8.args();
        args.max_tokens = ops::kMoeHelperMaxTokens;
        ops::moe_helper_experts(args, workspace, nullptr);
    });
    failures += expect_rejected("A8 with x", [&] {
        auto args = a8.args();
        args.x    = valid.args().x;
        ops::moe_helper_experts(args, prefill, nullptr);
    });
    failures += expect_rejected("A8 without scales", [&] {
        auto args          = a8.args();
        args.x_e4m3.scales = nullptr;
        ops::moe_helper_experts(args, prefill, nullptr);
    });
    failures += expect_rejected("A16 with x_e4m3", [&] {
        auto args   = valid.args();
        args.x_e4m3 = a8.args().x_e4m3;
        ops::moe_helper_experts(args, workspace, nullptr);
    });
    failures += expect_rejected("misaligned A8 codes", [&] {
        auto args = a8.args();
        args.x_e4m3.codes += 8;
        ops::moe_helper_experts(args, prefill, nullptr);
    });
    failures += expect_rejected("y aliases A8 scales", [&] {
        auto args   = a8.args();
        args.y      = const_cast<float*>(args.x_e4m3.scales);
        ops::moe_helper_experts(args, prefill, nullptr);
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
