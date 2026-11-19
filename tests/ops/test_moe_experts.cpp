// moe_experts Op test against an independent FP64 oracle at the Qwen3.8-Flash-Next geometry.
//
// Each test expert is generated as canonical codes and E4M3 scales, written into the documented
// record layout by a test-side packer, and read back by an independent reader that locates every
// (matrix, row, K) value from the header's byte formulas; the round trip must be exact. The oracle
// decodes the canonical codes (through the matrix's F3 codebook or as E2M1) times the exact E4M3
// scale, applies the FP32 matrix scales and evaluates the complete routed + shared SwiGLU sum in
// FP64 from the represented BF16 x. Both activation profiles are checked against it, each with
// its own criterion; the A8 profile is additionally checked against the same FP64 formula
// evaluated on its represented activations (the host A8 codec of x and of the BF16 gated rows), a
// supplementary control with the tight A16-like criterion.
#include "ninfer/ops/moe_expert_partition.h"
#include "ninfer/ops/moe_experts.h"
#include "ops/moe_a8_activation_reference.h"
#include "ops/moe_expert_record_reference.h"
#include "ops/op_tester.h"
#include "ops/sanitizer_scope.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <exception>
#include <iostream>
#include <limits>
#include <memory>
#include <numeric>
#include <random>
#include <set>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

using namespace ninfer;
using namespace ninfer::test;

namespace {

using ops::MoeActivationProfile;
using ops::MoeExpertCodec;

constexpr int kHidden        = ops::kMoeHidden;
constexpr int kInter         = ops::kMoeIntermediate;
constexpr int kExperts       = ops::kMoeExperts;
constexpr int kTopK          = ops::kMoeTopK;
constexpr int kOracleThreads = 8;

// Criterion. Two roundings dominate the error against the FP64 oracle. BF16 keeps 8 significant
// bits, so round-to-nearest has unit roundoff u = 2^-8 (relative error at most u) and, for values
// spread over a binade, a relative RMS error of about 2^-7 / sqrt(12) / 1.5 ~= 1.5e-3. (1) The
// gate/up product enters the down projection rounded to BF16; its independent element errors pass
// through the down matrix with the same gain as the product itself, about 1.5e-3 relative RMS of
// each expert output. (2) y is rounded once to BF16, about 1.5e-3 relative RMS. FP32 accumulation
// over K = 2560 and 640 and the FP32 SiLU stay below 1e-5. Expected relative L2 is about 2.1e-3;
// the bound allows 5e-3. Pointwise, (2) is at most u |y| <= 3.9e-3 max|y| and (1) is Gaussian-like
// with an RMS near 1.5e-3 rms(y) ~= 4e-4 max|y|, under 2.5e-3 max|y| beyond six deviations; the
// gross cap is 8e-3 of the largest reference magnitude.
constexpr ReductionCriterion output_criterion() {
    return {/*relative_l2*/ 5.0e-3, /*gross_absolute*/ 0.0,
            /*gross_relative_to_max_reference*/ 8.0e-3};
}

// moe_experts_local_partial keeps the FP32 sum, so only the gate/up product's BF16 rounding (1)
// of the output criterion remains: about 1.5e-3 relative RMS, bounded at 3e-3 relative L2, and
// pointwise under 2.5e-3 max|y| beyond six deviations, capped at 4e-3 of the largest reference
// magnitude.
constexpr ReductionCriterion partial_criterion() {
    return {/*relative_l2*/ 3.0e-3, /*gross_absolute*/ 0.0,
            /*gross_relative_to_max_reference*/ 4.0e-3};
}

// A8 criterion against the BF16-input oracle. E4M3 keeps 4 significant bits: round-to-nearest has
// unit roundoff u = 2^-4 per element in the normal range, and an absolute error of at most 2^-10
// scale below it (subnormal codes); with a row scale of max|v| / 448 the subnormal part adds at
// most 2^-10 sqrt(K) / 448 < 1.2e-4 of a row's norm. The rounding error of a value in [2^e,
// 2^(e+1)) is close to uniform on +-2^(e-4), so its relative RMS is 2^-3 / sqrt(12) = 3.6e-2 for a
// value at the bottom of its binade and about 2.6e-2 for values spread over binades (the test's
// uniform x gives 2.55e-2). Errors of distinct K are independent, so a weight row's dot product
// carries the relative RMS e of its input vector: (1) x's rounding gives the gate and up
// pre-activations e_x; (2) the gate/up product h = SiLU(g) u inherits sqrt(kappa^2 + 1) e_x, with
// kappa = d ln SiLU(g) / d ln g = 1 + g (1 - sigma(g)) at most 1.28 for g > 0 and below 1 in
// magnitude where |g| <~ 1 carries the energy; (3) h's own rounding adds e_h; (4) the down rows
// carry the vector error, and the eleven contributions' independent errors keep the relative
// size. Expected: sqrt(2 e^2 + e^2) with e = 2.6e-2 is 4.5e-2; the bound takes e = 3.6e-2 and
// kappa = 1.3: sqrt((1.64 * 3.6e-2)^2 + (3.6e-2)^2) = 6.9e-2; the independent A16 terms (2.1e-3)
// add in quadrature, so relative L2 <= 7e-2. Pointwise the error is Gaussian-like with RMS <= 7e-2
// rms(y); over the sampled outputs max|y| is about 4 rms(y), so six deviations stay below 1.1e-1
// max|y|.
constexpr ReductionCriterion a8_output_criterion() {
    return {/*relative_l2*/ 7.0e-2, /*gross_absolute*/ 0.0,
            /*gross_relative_to_max_reference*/ 1.1e-1};
}

// A8 local partial: the FP32 sum loses only the A16 output-rounding term, within the same bound.
constexpr ReductionCriterion a8_partial_criterion() { return a8_output_criterion(); }

// Supplementary A8 control against the oracle evaluated on the represented activations (the host
// codec of x and of the gated rows rounded to BF16): the activation roundings are shared, leaving
// the A16 terms (y's BF16 rounding, FP32 accumulation) plus rare one-ulp differences of a gated
// BF16 value between the FP32 kernel and the FP64 oracle, which can move that value's code or, for
// a row's largest value, the row's scale by about 2^-8 (shifting that row's represented values by
// about 0.4% plus a few re-rounded codes). The output criterion above bounds this.
constexpr ReductionCriterion represented_a8_criterion() { return output_criterion(); }

using moe_record::Expert;
using moe_record::ScaleMode;

// moe_record::make_expert, counting a failed record-layout round trip.
Expert make_expert(MoeExpertCodec codec, ScaleMode mode, std::uint32_t seed,
                   const std::array<std::uint32_t, 3>& books, int& failures) {
    bool round_trip = true;
    Expert expert   = moe_record::make_expert(codec, mode, seed, books, round_trip);
    if (!round_trip) {
        std::cerr << "moe_experts: record layout round trip failed (seed " << seed << ")\n";
        ++failures;
    }
    return expert;
}

class PinnedHost {
public:
    explicit PinnedHost(std::size_t bytes) {
        cuda_check(cudaHostAlloc(&host_, bytes, cudaHostAllocMapped), "cudaHostAlloc mapped");
        try {
            cuda_check(cudaHostGetDevicePointer(&device_, host_, 0), "cudaHostGetDevicePointer");
        } catch (...) {
            cudaFreeHost(host_);
            throw;
        }
    }

    ~PinnedHost() { cudaFreeHost(host_); }

    PinnedHost(const PinnedHost&)            = delete;
    PinnedHost& operator=(const PinnedHost&) = delete;

    [[nodiscard]] void* host() const noexcept { return host_; }

    [[nodiscard]] void* device() const noexcept { return device_; }

private:
    void* host_   = nullptr;
    void* device_ = nullptr;
};

// Distinct test records: kStandardRecords wide-scale records (even: F3, odd: NVFP4), then one F3
// and one NVFP4 record each of subnormal and narrow scales. Expert ids map onto them per case.
constexpr int kStandardRecords = 12;
constexpr int kSubnormalRecord = kStandardRecords;     // F3, then NVFP4
constexpr int kNarrowRecord    = kStandardRecords + 2; // F3, then NVFP4
constexpr int kRecords         = kStandardRecords + 4;

enum class RecordSet : std::uint8_t { Standard, Subnormal, Narrow };

// Every routed test record, each stored once in device memory and once in mapped pinned host
// memory, and the NVFP4G16 shared-expert record in device memory.
struct Fixture {
    std::array<std::uint32_t, 3> books{};
    std::vector<Expert> experts;
    std::vector<std::size_t> offsets; // byte offset of record r in both stores
    std::size_t store_bytes = 0;
    Expert shared;
    DeviceBuffer device_records;
    std::unique_ptr<PinnedHost> host_records;
    DeviceBuffer device_shared_record;
};

std::unique_ptr<Fixture> make_fixture(int& failures) {
    auto fixture = std::make_unique<Fixture>();
    std::mt19937 rng(20261114U);
    // Each F3 book is a permutation of the E2M1 nibbles of +-0.5, +-2, +-4, +-6.
    for (auto& book : fixture->books) {
        std::array<std::uint32_t, 8> entries = {0x1, 0x9, 0x4, 0xC, 0x6, 0xE, 0x7, 0xF};
        std::shuffle(entries.begin(), entries.end(), rng);
        book = 0;
        for (std::size_t v = 0; v < entries.size(); ++v) { book |= entries[v] << (4 * v); }
    }
    for (int r = 0; r < kRecords; ++r) {
        const MoeExpertCodec codec = r % 2 == 0 ? MoeExpertCodec::F3G32 : MoeExpertCodec::Nvfp4G16;
        const ScaleMode mode       = r < kSubnormalRecord ? ScaleMode::Wide
                                     : r < kNarrowRecord  ? ScaleMode::Subnormal
                                                          : ScaleMode::Narrow;
        fixture->experts.push_back(make_expert(codec, mode, 1000U + static_cast<std::uint32_t>(r),
                                               fixture->books, failures));
        fixture->offsets.push_back(fixture->store_bytes);
        fixture->store_bytes += fixture->experts.back().record.size();
    }
    fixture->device_records = DeviceBuffer(fixture->store_bytes);
    fixture->host_records   = std::make_unique<PinnedHost>(fixture->store_bytes);
    for (std::size_t r = 0; r < fixture->experts.size(); ++r) {
        const auto& record = fixture->experts[r].record;
        fixture->device_records.copy_from_host(record.data(), record.size(), fixture->offsets[r]);
        std::memcpy(static_cast<std::uint8_t*>(fixture->host_records->host()) + fixture->offsets[r],
                    record.data(), record.size());
    }
    fixture->shared =
        make_expert(MoeExpertCodec::Nvfp4G16, ScaleMode::Wide, 2000U, fixture->books, failures);
    fixture->device_shared_record = to_device(fixture->shared.record);
    return fixture;
}

// What the expert table and matrix scales say for every id: its record, whether the table points at
// the mapped host copy, and its three FP32 matrix scales; and the shared expert's matrix scales.
// shift remaps ids to records of the other codec.
struct BankState {
    std::vector<int> record;
    std::vector<bool> host_resident;
    std::vector<float> matrix_scales; // [512][3]
    std::array<float, 3> shared_matrix_scales{};
};

BankState make_bank_state(RecordSet set, int shift, bool mixed, std::uint32_t seed) {
    std::mt19937 rng(seed);
    BankState state{std::vector<int>(kExperts),
                    std::vector<bool>(kExperts, false),
                    std::vector<float>(static_cast<std::size_t>(kExperts) * 3),
                    {}};
    // Matrix scales keep weight magnitudes comparable across scale modes (RMS about 0.015).
    const double base = set == RecordSet::Standard    ? 3.0e-5
                        : set == RecordSet::Subnormal ? 0.6
                                                      : 1.7e-3;
    std::uniform_real_distribution<double> spread(0.5, 1.5);
    for (int id = 0; id < kExperts; ++id) {
        const auto index    = static_cast<std::size_t>(id);
        state.record[index] = set == RecordSet::Standard ? (id * 7 + 3 + shift) % kStandardRecords
                              : set == RecordSet::Subnormal ? kSubnormalRecord + (id + shift) % 2
                                                            : kNarrowRecord + (id + shift) % 2;
        state.host_resident[index] = mixed && (rng() & 1U) != 0;
    }
    for (auto& scale : state.matrix_scales) { scale = static_cast<float>(base * spread(rng)); }
    // The shared record has wide scales whatever the routed set (its contribution is off for the
    // subnormal and narrow sets).
    for (auto& scale : state.shared_matrix_scales) {
        scale = static_cast<float>(3.0e-5 * spread(rng));
    }
    return state;
}

enum class Routing : std::uint8_t { Skewed, Distinct, Concentrated };

struct Inputs {
    int tokens;
    std::vector<float> x; // [T][2560], BF16-representable
    std::vector<std::int32_t> ids;
    std::vector<float> weights;
    std::vector<float> shared_scale;
};

// Route weights are the renormalized softmax of the selected logits, as moe_route produces.
void set_token(Inputs& inputs, int t, const std::vector<int>& ids,
               const std::vector<double>& logits) {
    const double maximum = *std::max_element(logits.begin(), logits.end());
    double total         = 0.0;
    for (const double logit : logits) { total += std::exp(logit - maximum); }
    for (std::size_t j = 0; j < ids.size(); ++j) {
        const std::size_t slot = static_cast<std::size_t>(t) * kTopK + j;
        inputs.ids[slot]       = ids[j];
        inputs.weights[slot]   = static_cast<float>(std::exp(logits[j] - maximum) / total);
    }
}

// With outliers, every eighth token has one channel at 64 times the uniform range, so most of its
// A8 codes fall in the lower binades and some in the subnormal range.
Inputs make_inputs(const Fixture& fixture, const BankState& state, int tokens, Routing routing,
                   bool shared, std::uint32_t seed, bool outliers = false) {
    std::mt19937 rng(seed);
    std::normal_distribution<double> normal(0.0, 1.0);
    const auto extent = static_cast<std::size_t>(tokens);
    Inputs inputs{tokens, std::vector<float>(extent * kHidden),
                  std::vector<std::int32_t>(extent * kTopK), std::vector<float>(extent * kTopK),
                  std::vector<float>(extent, 0.0F)};
    fill_uniform(inputs.x, seed + 1, -2.0F, 2.0F);
    if (outliers) {
        std::uniform_int_distribution<int> channel(0, kHidden - 1);
        for (int t = 0; t < tokens; t += 8) {
            inputs
                .x[static_cast<std::size_t>(t) * kHidden + static_cast<std::size_t>(channel(rng))] =
                (t % 16 == 0 ? 128.0F : -128.0F);
        }
    }
    round_to_bf16(inputs.x);
    if (shared) {
        for (auto& scale : inputs.shared_scale) {
            scale = static_cast<float>(1.0 / (1.0 + std::exp(-normal(rng))));
        }
    }
    std::vector<double> bias(kExperts);
    for (auto& b : bias) { b = normal(rng); }
    std::vector<int> pool(kExperts);
    std::iota(pool.begin(), pool.end(), 0);
    std::shuffle(pool.begin(), pool.end(), rng);
    for (int t = 0; t < tokens; ++t) {
        std::vector<int> ids;
        std::vector<double> logits;
        if (routing == Routing::Skewed) {
            std::vector<double> all(kExperts);
            for (std::size_t e = 0; e < all.size(); ++e) {
                all[e] = 2.0 * bias[e] + 1.5 * normal(rng);
            }
            std::vector<int> order(kExperts);
            std::iota(order.begin(), order.end(), 0);
            std::stable_sort(order.begin(), order.end(), [&](int a, int b) {
                return all[static_cast<std::size_t>(a)] > all[static_cast<std::size_t>(b)];
            });
            ids.assign(order.begin(), order.begin() + kTopK);
            for (const int id : ids) { logits.push_back(all[static_cast<std::size_t>(id)]); }
        } else {
            // Distinct: ten unused experts per token; Concentrated: the same ten for every token.
            const int base = routing == Routing::Distinct ? kTopK * t : 0;
            ids.assign(pool.begin() + base, pool.begin() + base + kTopK);
            std::shuffle(ids.begin(), ids.end(), rng);
            for (int j = 0; j < kTopK; ++j) { logits.push_back(-0.3 * j + 0.2 * normal(rng)); }
        }
        set_token(inputs, t, ids, logits);
    }
    bool has[2] = {false, false};
    for (int j = 0; j < kTopK; ++j) {
        const int record =
            state.record[static_cast<std::size_t>(inputs.ids[static_cast<std::size_t>(j)])];
        has[static_cast<int>(fixture.experts[static_cast<std::size_t>(record)].codec)] = true;
    }
    if (!has[0] || !has[1]) { throw std::logic_error("moe_experts test: token 0 lacks a codec"); }
    return inputs;
}

// Which assignments of a token the oracle sums: all of them (moe_experts), the routed ones of
// local experts plus the shared expert (moe_experts_local_partial), or the routed ones of the
// other experts (the helper's share).
enum class Part : std::uint8_t { All, Local, Helper };

// The oracle's activations: the represented BF16 x (the Op's mathematical oracle), or the A8
// profile's represented activations (the supplementary control): x replaced by its A8 codec
// values and each expert's gate/up product rounded to BF16 and replaced by its A8 codec values.
enum class OracleInput : std::uint8_t { Bf16, RepresentedA8 };

// FP64 oracle of one token; `local` (one flag per expert id) selects Local and Helper.
std::vector<double> oracle_token(const Fixture& fixture, const BankState& state,
                                 const Inputs& inputs, int t, Part part = Part::All,
                                 const std::vector<bool>* local = nullptr,
                                 OracleInput input              = OracleInput::Bf16) {
    const float* row = inputs.x.data() + static_cast<std::size_t>(t) * kHidden;
    std::vector<double> x(row, row + kHidden);
    if (input == OracleInput::RepresentedA8) {
        x = moe_a8::decode(moe_a8::encode(std::span<const float>(row, kHidden), kHidden), kHidden);
    }
    std::vector<double> y(kHidden, 0.0);
    std::vector<double> gated(kInter);
    const auto expert_output = [&](const float* gate, const float* up, const float* down,
                                   const double (&scale)[3], double weight) {
        for (std::size_t i = 0; i < gated.size(); ++i) {
            double g              = 0.0;
            double u              = 0.0;
            const float* gate_row = gate + i * kHidden;
            const float* up_row   = up + i * kHidden;
            for (int k = 0; k < kHidden; ++k) {
                g += static_cast<double>(gate_row[k]) * x[k];
                u += static_cast<double>(up_row[k]) * x[k];
            }
            g *= scale[0];
            u *= scale[1];
            gated[i] = g / (1.0 + std::exp(-g)) * u;
        }
        if (input == OracleInput::RepresentedA8) {
            std::vector<float> rounded(gated.size());
            for (std::size_t i = 0; i < gated.size(); ++i) {
                rounded[i] = bf16_to_f32(f32_to_bf16(static_cast<float>(gated[i])));
            }
            gated = moe_a8::decode(moe_a8::encode(rounded, kInter), kInter);
        }
        for (std::size_t n = 0; n < y.size(); ++n) {
            double sum            = 0.0;
            const float* down_row = down + n * kInter;
            for (std::size_t i = 0; i < gated.size(); ++i) {
                sum += static_cast<double>(down_row[i]) * gated[i];
            }
            y[n] += weight * scale[2] * sum;
        }
    };
    for (int j = 0; j < kTopK; ++j) {
        const std::size_t slot = static_cast<std::size_t>(t) * kTopK + j;
        const auto id          = static_cast<std::size_t>(inputs.ids[slot]);
        if (part != Part::All && (*local)[id] != (part == Part::Local)) { continue; }
        const Expert& expert  = fixture.experts[static_cast<std::size_t>(state.record[id])];
        const double scale[3] = {state.matrix_scales[id * 3], state.matrix_scales[id * 3 + 1],
                                 state.matrix_scales[id * 3 + 2]};
        expert_output(expert.values[0].data(), expert.values[1].data(), expert.values[2].data(),
                      scale, inputs.weights[slot]);
    }
    if (part == Part::Helper) { return y; }
    const double shared[3] = {state.shared_matrix_scales[0], state.shared_matrix_scales[1],
                              state.shared_matrix_scales[2]};
    expert_output(fixture.shared.values[0].data(), fixture.shared.values[1].data(),
                  fixture.shared.values[2].data(), shared,
                  inputs.shared_scale[static_cast<std::size_t>(t)]);
    return y;
}

std::vector<int> sample_tokens(int tokens, int samples, std::uint32_t seed) {
    if (tokens <= samples) {
        std::vector<int> all(static_cast<std::size_t>(tokens));
        std::iota(all.begin(), all.end(), 0);
        return all;
    }
    std::set<int> chosen;
    for (int i = 0; i < 4; ++i) {
        chosen.insert(i);
        chosen.insert(tokens - 1 - i);
    }
    std::mt19937 rng(seed);
    std::uniform_int_distribution<int> pick(0, tokens - 1);
    while (static_cast<int>(chosen.size()) < samples) { chosen.insert(pick(rng)); }
    return {chosen.begin(), chosen.end()};
}

// The oracle of each token in `tokens`, computed on kOracleThreads threads.
std::vector<std::vector<double>> oracle_tokens(const Fixture& fixture, const BankState& state,
                                               const Inputs& inputs, const std::vector<int>& tokens,
                                               Part part                      = Part::All,
                                               const std::vector<bool>* local = nullptr,
                                               OracleInput input              = OracleInput::Bf16) {
    std::vector<std::vector<double>> expected(tokens.size());
    std::vector<std::jthread> workers;
    workers.reserve(kOracleThreads);
    for (std::size_t w = 0; w < kOracleThreads; ++w) {
        workers.emplace_back([&, w] {
            for (std::size_t i = w; i < tokens.size(); i += kOracleThreads) {
                expected[i] = oracle_token(fixture, state, inputs, tokens[i], part, local, input);
            }
        });
    }
    workers.clear();
    return expected;
}

// Compares the sampled tokens of y (BF16 bits [T][2560]) against the oracle with the profile's
// criterion and, for A8, against the represented-activation control.
int verify_output(const std::string& label, const Fixture& fixture, const BankState& state,
                  const Inputs& inputs, const std::vector<std::uint16_t>& bits,
                  const std::vector<int>& sampled, MoeActivationProfile profile) {
    std::vector<double> got;
    for (const int token : sampled) {
        const auto row = static_cast<std::size_t>(token) * kHidden;
        for (std::size_t n = 0; n < kHidden; ++n) { got.push_back(bf16_to_f32(bits[row + n])); }
    }
    const auto compare = [&](const std::string& name, OracleInput input,
                             const ReductionCriterion& criterion) {
        const std::vector<std::vector<double>> expected =
            oracle_tokens(fixture, state, inputs, sampled, Part::All, nullptr, input);
        std::vector<double> reference;
        for (const auto& token : expected) {
            reference.insert(reference.end(), token.begin(), token.end());
        }
        return verify_reduction(name, got, reference, criterion);
    };
    if (profile == MoeActivationProfile::A16) {
        return compare(label, OracleInput::Bf16, output_criterion());
    }
    return compare(label, OracleInput::Bf16, a8_output_criterion()) +
           compare(label + " represented-A8 control", OracleInput::RepresentedA8,
                   represented_a8_criterion());
}

class Stream {
public:
    Stream() {
        cuda_check(cudaStreamCreateWithFlags(&stream_, cudaStreamNonBlocking), "cudaStreamCreate");
    }

    ~Stream() { cudaStreamDestroy(stream_); }

    Stream(const Stream&)            = delete;
    Stream& operator=(const Stream&) = delete;

    [[nodiscard]] cudaStream_t get() const noexcept { return stream_; }

private:
    cudaStream_t stream_ = nullptr;
};

class Graph {
public:
    template <class Body>
    Graph(cudaStream_t stream, Body&& body) {
        cuda_check(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal),
                   "begin capture");
        try {
            body(stream);
        } catch (...) {
            cudaGraph_t discard = nullptr;
            cudaStreamEndCapture(stream, &discard);
            if (discard != nullptr) { cudaGraphDestroy(discard); }
            throw;
        }
        cuda_check(cudaStreamEndCapture(stream, &graph_), "end capture");
        try {
            cuda_check(cudaGraphInstantiate(&exec_, graph_, 0), "graph instantiate");
        } catch (...) {
            cudaGraphDestroy(graph_);
            throw;
        }
    }

    ~Graph() {
        if (exec_ != nullptr) { cudaGraphExecDestroy(exec_); }
        if (graph_ != nullptr) { cudaGraphDestroy(graph_); }
    }

    Graph(const Graph&)            = delete;
    Graph& operator=(const Graph&) = delete;

    void replay(cudaStream_t stream) const {
        cuda_check(cudaGraphLaunch(exec_, stream), "graph launch");
        cuda_synchronize(stream);
    }

private:
    cudaGraph_t graph_    = nullptr;
    cudaGraphExec_t exec_ = nullptr;
};

// Device copies of one case's inputs, expert table and matrix scales, and one Op invocation.
class Run {
public:
    Run(const Fixture& fixture, const Inputs& inputs, MoeActivationProfile profile)
        : fixture_(fixture), profile_(profile), tokens_(inputs.tokens),
          x_(to_device_bf16(inputs.x)), ids_(to_device(inputs.ids)),
          weights_(to_device(inputs.weights)), shared_scale_(to_device(inputs.shared_scale)),
          table_(static_cast<std::size_t>(kExperts) * sizeof(std::int64_t)),
          scales_(static_cast<std::size_t>(kExperts) * 3 * sizeof(float)),
          shared_scales_(3 * sizeof(float)),
          y_(static_cast<std::size_t>(inputs.tokens) * kHidden * sizeof(std::uint16_t)),
          partial_(static_cast<std::size_t>(inputs.tokens) * kHidden * sizeof(float)),
          workspace_(ops::moe_experts_workspace_capacity_bytes(inputs.tokens, profile)) {}

    void upload(const BankState& state) {
        std::vector<std::int64_t> entries(kExperts);
        for (std::size_t id = 0; id < entries.size(); ++id) {
            const auto record  = static_cast<std::size_t>(state.record[id]);
            auto* base         = state.host_resident[id]
                                     ? static_cast<std::uint8_t*>(fixture_.host_records->device())
                                     : static_cast<std::uint8_t*>(fixture_.device_records.p);
            const auto address = reinterpret_cast<std::uintptr_t>(base + fixture_.offsets[record]);
            entries[id]        = static_cast<std::int64_t>(
                address | static_cast<std::uintptr_t>(fixture_.experts[record].codec));
        }
        table_.copy_from_host(entries.data(), table_.bytes);
        scales_.copy_from_host(state.matrix_scales.data(), scales_.bytes);
        shared_scales_.copy_from_host(state.shared_matrix_scales.data(), shared_scales_.bytes);
    }

    // Overrides expert id's table entry (an address with its codec in bit 0).
    void set_entry(int id, std::uint64_t entry) {
        cuda_check(cudaMemcpy(static_cast<std::uint8_t*>(table_.p) +
                                  static_cast<std::size_t>(id) * sizeof(std::uint64_t),
                              &entry, sizeof(entry), cudaMemcpyHostToDevice),
                   "table entry upload");
    }

    [[nodiscard]] ops::MoeExpertBank bank() const {
        ops::MoeExpertBank bank;
        bank.expert_table         = Tensor(table_.p, DType::I64, {kExperts});
        bank.matrix_scales        = Tensor(scales_.p, DType::FP32, {3, kExperts});
        bank.f3_books             = fixture_.books;
        bank.shared_record        = Tensor(fixture_.device_shared_record.p, DType::U8,
                                           {static_cast<std::int32_t>(ops::kMoeNvfp4RecordBytes)});
        bank.shared_matrix_scales = Tensor(shared_scales_.p, DType::FP32, {3});
        return bank;
    }

    void invoke(cudaStream_t stream, const ops::MoeExpertsDeferral* deferral = nullptr) {
        Tensor x(x_.p, DType::BF16, {kHidden, tokens_});
        Tensor ids(ids_.p, DType::I32, {kTopK, tokens_});
        Tensor weights(weights_.p, DType::FP32, {kTopK, tokens_});
        Tensor scale(shared_scale_.p, DType::FP32, {tokens_});
        Tensor y(y_.data(), DType::BF16, {kHidden, tokens_});
        ops::moe_experts(x, ids, weights, scale, bank(), profile_, y, workspace_, stream, deferral);
    }

    // moe_experts_local_partial into the FP32 partial, local_experts I32 [16].
    void invoke_local(cudaStream_t stream, const DeviceBuffer& local_experts,
                      const ops::MoeExpertsDeferral* deferral = nullptr) {
        Tensor x(x_.p, DType::BF16, {kHidden, tokens_});
        Tensor ids(ids_.p, DType::I32, {kTopK, tokens_});
        Tensor weights(weights_.p, DType::FP32, {kTopK, tokens_});
        Tensor scale(shared_scale_.p, DType::FP32, {tokens_});
        Tensor mask(local_experts.p, DType::I32, {ops::kMoeExpertMaskWords});
        Tensor partial(partial_.data(), DType::FP32, {kHidden, tokens_});
        ops::moe_experts_local_partial(x, ids, weights, scale, bank(), profile_, mask, partial,
                                       workspace_, stream, deferral);
    }

    // Poisons y and the partial and orders the poison before work on any stream.
    void poison_output() {
        y_.fill(0xff);
        partial_.fill(0xff);
        cuda_synchronize();
    }

    [[nodiscard]] std::vector<float> partial_output() const {
        return from_device<float>(partial_.data(), static_cast<std::size_t>(tokens_) * kHidden);
    }

    [[nodiscard]] int verify_partial_guards(const std::string& label) const {
        return partial_.verify_guards(label);
    }

    [[nodiscard]] std::vector<std::uint16_t> output() const {
        return from_device<std::uint16_t>(y_.data(), static_cast<std::size_t>(tokens_) * kHidden);
    }

    [[nodiscard]] int verify_guards(const std::string& label) const {
        return y_.verify_guards(label);
    }

    // Raw device views for the rejection checks.
    [[nodiscard]] void* x_data() const { return x_.p; }

    [[nodiscard]] void* ids_data() const { return ids_.p; }

    [[nodiscard]] void* weights_data() const { return weights_.p; }

    [[nodiscard]] void* shared_scale_data() const { return shared_scale_.p; }

    [[nodiscard]] void* y_data() { return y_.data(); }

private:
    const Fixture& fixture_;
    MoeActivationProfile profile_;
    int tokens_;
    DeviceBuffer x_, ids_, weights_, shared_scale_, table_, scales_, shared_scales_;
    GuardedDeviceBuffer y_;
    GuardedDeviceBuffer partial_;
    WorkspaceArena workspace_;
};

class Event {
public:
    Event() {
        cuda_check(cudaEventCreateWithFlags(&event_, cudaEventDisableTiming), "cudaEventCreate");
    }

    ~Event() { cudaEventDestroy(event_); }

    Event(const Event&)            = delete;
    Event& operator=(const Event&) = delete;

    [[nodiscard]] cudaEvent_t get() const noexcept { return event_; }

private:
    cudaEvent_t event_ = nullptr;
};

// Deferred records: the first `count` routed experts in ascending id (host- and device-resident
// alike; none for count 0, a deferral with nothing deferred) whose expert-table entries name a
// poisoned record (every byte 0x7F, an E4M3 NaN in every scale byte; codec bit kept) and whose
// deferral slots start poisoned and receive the true record only from copies on a second stream
// that first sleeps on the host. y is therefore correct only if the Op reads the deferred records
// from the slots, and only after the ready event.
class DeferredRecords {
public:
    static constexpr std::uint8_t kPoison = 0x7F;

    DeferredRecords(const Fixture& fixture, const BankState& state, const Inputs& inputs,
                    std::size_t count)
        : poison_(static_cast<std::size_t>(ops::kMoeNvfp4RecordBytes)),
          column_(static_cast<std::size_t>(kExperts) * sizeof(std::uint64_t)) {
        poison_.fill(kPoison);
        const std::set<int> routed(inputs.ids.begin(), inputs.ids.end());
        std::vector<std::uint64_t> entries(kExperts, 0);
        for (const int id : routed) {
            if (ids_.size() == count) { break; }
            const auto& expert =
                fixture
                    .experts[static_cast<std::size_t>(state.record[static_cast<std::size_t>(id)])];
            ids_.push_back(id);
            codecs_.push_back(expert.codec);
            sources_.push_back(
                static_cast<const std::uint8_t*>(fixture.device_records.p) +
                fixture
                    .offsets[static_cast<std::size_t>(state.record[static_cast<std::size_t>(id)])]);
            slots_.push_back(std::make_unique<GuardedDeviceBuffer>(expert.record.size()));
            entries[static_cast<std::size_t>(id)] =
                reinterpret_cast<std::uintptr_t>(slots_.back()->data());
        }
        column_.copy_from_host(entries.data(), column_.bytes);
    }

    // Points every deferred expert's table entry at the poisoned record.
    void poison_table(Run& run) const {
        for (std::size_t i = 0; i < ids_.size(); ++i) {
            run.set_entry(ids_[i], reinterpret_cast<std::uintptr_t>(poison_.p) |
                                       static_cast<std::uint64_t>(codecs_[i]));
        }
    }

    void poison_slots() {
        for (const auto& slot : slots_) { slot->fill(kPoison); }
        cuda_synchronize();
    }

    // Enqueues on `copies`: a 30 ms host sleep, the true records into the slots, then ready.
    void enqueue_copies(cudaStream_t copies) {
        cuda_check(cudaLaunchHostFunc(
                       copies,
                       [](void*) { std::this_thread::sleep_for(std::chrono::milliseconds(30)); },
                       nullptr),
                   "delay");
        for (std::size_t i = 0; i < ids_.size(); ++i) {
            cuda_check(cudaMemcpyAsync(slots_[i]->data(), sources_[i], slots_[i]->bytes(),
                                       cudaMemcpyDeviceToDevice, copies),
                       "deferred record copy");
        }
        cuda_check(cudaEventRecord(ready_.get(), copies), "ready record");
    }

    [[nodiscard]] ops::MoeExpertsDeferral deferral() const {
        return {Tensor(column_.p, DType::I64, {kExperts}), ready_.get()};
    }

    int verify_guards(const std::string& label) const {
        int failures = 0;
        for (const auto& slot : slots_) { failures += slot->verify_guards(label + " slot"); }
        return failures;
    }

private:
    DeviceBuffer poison_;
    DeviceBuffer column_;
    Event ready_;
    std::vector<int> ids_;
    std::vector<MoeExpertCodec> codecs_;
    std::vector<const std::uint8_t*> sources_;
    std::vector<std::unique_ptr<GuardedDeviceBuffer>> slots_;
};

struct Case {
    int tokens;
    Routing routing;
    bool mixed;
    RecordSet records;
    int samples;
    bool graph;
    int deferred; // routed experts read through a deferral; 0: a deferral defers none; -1: none
    std::uint32_t seed;
    MoeActivationProfile profile = MoeActivationProfile::A16;
    bool outliers                = false; // make_inputs' outlier tokens
};

const char* profile_name(MoeActivationProfile profile) {
    return profile == MoeActivationProfile::A8 ? " a8" : "";
}

std::string describe(const Case& c) {
    const char* routing = c.routing == Routing::Skewed     ? "skewed"
                          : c.routing == Routing::Distinct ? "distinct"
                                                           : "concentrated";
    const char* records = c.records == RecordSet::Standard    ? ""
                          : c.records == RecordSet::Subnormal ? " subnormal-scales"
                                                              : " narrow-scales";
    return "moe_experts" + std::string(profile_name(c.profile)) + " T=" + std::to_string(c.tokens) +
           " " + routing + (c.mixed ? " mixed-host" : " device") + records +
           (c.outliers ? " outliers" : "") + (c.graph ? " graph" : "") +
           (c.deferred < 0 ? "" : " deferred=" + std::to_string(c.deferred));
}

int run_case(const Fixture& fixture, const Case& c) {
    const std::string label = describe(c);
    const BankState state   = make_bank_state(c.records, 0, c.mixed, c.seed + 5);
    const Inputs inputs     = make_inputs(fixture, state, c.tokens, c.routing,
                                          c.records == RecordSet::Standard, c.seed, c.outliers);
    const std::vector<int> sampled = sample_tokens(c.tokens, c.samples, c.seed + 9);
    Run run(fixture, inputs, c.profile);
    run.upload(state);
    run.poison_output();
    run.invoke(nullptr);
    cuda_synchronize();
    const std::vector<std::uint16_t> bits = run.output();
    int failures = verify_output(label, fixture, state, inputs, bits, sampled, c.profile);
    failures += run.verify_guards(label + " y");
    // Determinism: a second call on the same inputs is bit-identical.
    run.poison_output();
    run.invoke(nullptr);
    cuda_synchronize();
    failures += verify_exact((label + " repeat").c_str(), run.output(), bits);

    if (c.deferred >= 0) {
        // Eager: the copies run on a second stream after a host sleep; the Op's stream waits for
        // ready before the deferred experts. Then a captured fork/join replayed twice, with the
        // slots poisoned again before each replay.
        DeferredRecords deferred(fixture, state, inputs, static_cast<std::size_t>(c.deferred));
        deferred.poison_table(run);
        const Stream main;
        const Stream copies;
        const ops::MoeExpertsDeferral deferral = deferred.deferral();
        deferred.poison_slots();
        run.poison_output();
        deferred.enqueue_copies(copies.get());
        run.invoke(main.get(), &deferral);
        cuda_synchronize();
        failures += verify_exact((label + " eager").c_str(), run.output(), bits);
        failures += run.verify_guards(label + " deferred y");

        const Event fork;
        const Graph graph(main.get(), [&](cudaStream_t s) {
            cuda_check(cudaEventRecord(fork.get(), s), "fork record");
            cuda_check(cudaStreamWaitEvent(copies.get(), fork.get(), 0), "fork wait");
            deferred.enqueue_copies(copies.get());
            run.invoke(s, &deferral);
        });
        for (int replay = 0; replay < 2; ++replay) {
            deferred.poison_slots();
            run.poison_output();
            graph.replay(main.get());
            failures += verify_exact((label + " graph replay " + std::to_string(replay)).c_str(),
                                     run.output(), bits);
        }
        failures += deferred.verify_guards(label);
        run.upload(state);
    }

    if (c.graph) {
        // Capture, replay, then change every table entry (residency and codec) and the matrix
        // scales between replays.
        const Stream stream;
        const Graph graph(stream.get(), [&](cudaStream_t s) { run.invoke(s); });
        run.poison_output();
        graph.replay(stream.get());
        failures += verify_exact((label + " graph replay").c_str(), run.output(), bits);

        BankState changed = make_bank_state(c.records, 1, c.mixed, c.seed + 6);
        for (std::size_t id = 0; id < changed.host_resident.size(); ++id) {
            changed.host_resident[id] = !state.host_resident[id];
        }
        run.upload(changed);
        run.poison_output();
        graph.replay(stream.get());
        const std::vector<std::uint16_t> replayed = run.output();
        failures += verify_output(label + " replay after table change", fixture, changed, inputs,
                                  replayed, sampled, c.profile);
        run.poison_output();
        run.invoke(nullptr);
        cuda_synchronize();
        failures +=
            verify_exact((label + " eager after table change").c_str(), run.output(), replayed);
    }
    return failures;
}

// Expert-helper split: which experts are local.
enum class Locality : std::uint8_t { All, None, Random };

std::vector<bool> make_locality(Locality locality, std::uint32_t seed) {
    std::vector<bool> local(kExperts, locality == Locality::All);
    if (locality == Locality::Random) {
        std::mt19937 rng(seed);
        for (std::size_t id = 0; id < local.size(); ++id) { local[id] = (rng() & 1U) != 0; }
    }
    return local;
}

// The I32 [16] bitset of moe_experts_local_partial and moe_expert_partition.
std::vector<std::int32_t> locality_words(const std::vector<bool>& local) {
    std::vector<std::uint32_t> words(ops::kMoeExpertMaskWords, 0);
    for (std::size_t id = 0; id < local.size(); ++id) {
        if (local[id]) { words[id / 32] |= 1U << (id % 32); }
    }
    std::vector<std::int32_t> bits(words.size());
    std::memcpy(bits.data(), words.data(), words.size() * sizeof(std::uint32_t));
    return bits;
}

// An address no allocation ever occupies (below Linux's mmap_min_addr on the host side of the
// unified address space, and outside every device allocation), 256-byte aligned like a record.
constexpr std::uint64_t kUnmappedRecord = 0x100;

// Points every non-local expert's table entry at kUnmappedRecord, keeping its codec bit: any read
// of a non-local record, summed or not, faults and fails the test.
void unmap_remote(Run& run, const Fixture& fixture, const BankState& state,
                  const std::vector<bool>& local) {
    for (int id = 0; id < kExperts; ++id) {
        const auto index = static_cast<std::size_t>(id);
        if (local[index]) { continue; }
        const auto codec = fixture.experts[static_cast<std::size_t>(state.record[index])].codec;
        run.set_entry(id, kUnmappedRecord | static_cast<std::uint64_t>(codec));
    }
}

// Compares the FP32 partial [T][2560] of every token against the Local restriction of the oracle
// with the profile's criterion and, for A8, against the represented-activation control.
int verify_partial(const std::string& label, const Fixture& fixture, const BankState& state,
                   const Inputs& inputs, const std::vector<bool>& local,
                   const std::vector<float>& partial, MoeActivationProfile profile) {
    std::vector<int> tokens(static_cast<std::size_t>(inputs.tokens));
    std::iota(tokens.begin(), tokens.end(), 0);
    const std::vector<double> got(partial.begin(), partial.end());
    const auto compare = [&](const std::string& name, OracleInput input,
                             const ReductionCriterion& criterion) {
        const std::vector<std::vector<double>> expected =
            oracle_tokens(fixture, state, inputs, tokens, Part::Local, &local, input);
        std::vector<double> reference;
        reference.reserve(got.size());
        for (const auto& row : expected) {
            reference.insert(reference.end(), row.begin(), row.end());
        }
        return verify_reduction(name, got, reference, criterion);
    };
    if (profile == MoeActivationProfile::A16) {
        return compare(label, OracleInput::Bf16, partial_criterion());
    }
    return compare(label, OracleInput::Bf16, a8_partial_criterion()) +
           compare(label + " represented-A8 control", OracleInput::RepresentedA8,
                   partial_criterion());
}

struct SplitCase {
    int tokens;
    Routing routing;
    Locality locality;
    bool graph;
    int deferred; // routed experts (local or not) read through a deferral; -1: no deferral
    std::uint32_t seed;
    MoeActivationProfile profile = MoeActivationProfile::A16;
};

std::string describe(const SplitCase& c) {
    const char* routing  = c.routing == Routing::Skewed     ? "skewed"
                           : c.routing == Routing::Distinct ? "distinct"
                                                            : "concentrated";
    const char* locality = c.locality == Locality::All    ? "all-local"
                           : c.locality == Locality::None ? "none-local"
                                                          : "random-local";
    return "moe_experts" + std::string(profile_name(c.profile)) +
           " split T=" + std::to_string(c.tokens) + " " + routing + " " + locality +
           (c.graph ? " graph" : "") +
           (c.deferred < 0 ? "" : " deferred=" + std::to_string(c.deferred));
}

// moe_experts_local_partial against the Local oracle (with every non-local record unmapped), then
// moe_expert_partition and moe_expert_partial_join with the helper's share given as the FP32
// rounding of its oracle (for A8, of the represented-activation oracle): the joined y against the
// full oracle. All-local partials round to
// moe_experts' output bit for bit (the same slot-order sum). With graph, a captured local partial
// is replayed after the bitset and the table change.
int run_split_case(const Fixture& fixture, const SplitCase& c) {
    const std::string label       = describe(c);
    const BankState state         = make_bank_state(RecordSet::Standard, 0, true, c.seed + 5);
    const Inputs inputs           = make_inputs(fixture, state, c.tokens, c.routing, true, c.seed);
    const std::vector<bool> local = make_locality(c.locality, c.seed + 7);
    const auto tokens             = static_cast<std::size_t>(c.tokens);
    Run run(fixture, inputs, c.profile);
    run.upload(state);
    run.poison_output();
    run.invoke(nullptr);
    cuda_synchronize();
    const std::vector<std::uint16_t> ordinary = run.output();

    unmap_remote(run, fixture, state, local);
    DeviceBuffer mask = to_device(locality_words(local));
    run.poison_output();
    run.invoke_local(nullptr, mask);
    cuda_synchronize();
    const std::vector<float> partial = run.partial_output();
    int failures =
        verify_partial(label + " partial", fixture, state, inputs, local, partial, c.profile);
    failures += run.verify_partial_guards(label + " partial");
    if (c.locality == Locality::All) {
        std::vector<std::uint16_t> rounded(partial.size());
        std::transform(partial.begin(), partial.end(), rounded.begin(), f32_to_bf16);
        failures += verify_exact((label + " all-local rounding").c_str(), rounded, ordinary);
    }

    DeviceBuffer compact_map(tokens * sizeof(std::int32_t));
    DeviceBuffer compact_count(16);
    DeviceBuffer compact_x(tokens * kHidden * sizeof(std::uint16_t));
    DeviceBuffer compact_ids(tokens * kTopK * sizeof(std::int32_t));
    DeviceBuffer compact_weights(tokens * kTopK * sizeof(float));
    {
        Tensor x(run.x_data(), DType::BF16, {kHidden, c.tokens});
        Tensor ids(run.ids_data(), DType::I32, {kTopK, c.tokens});
        Tensor weights(run.weights_data(), DType::FP32, {kTopK, c.tokens});
        Tensor bits(mask.p, DType::I32, {ops::kMoeExpertMaskWords});
        Tensor map(compact_map.p, DType::I32, {c.tokens});
        Tensor count(compact_count.p, DType::I32, {1});
        Tensor cx(compact_x.p, DType::BF16, {kHidden, c.tokens});
        Tensor cids(compact_ids.p, DType::I32, {kTopK, c.tokens});
        Tensor cweights(compact_weights.p, DType::FP32, {kTopK, c.tokens});
        ops::moe_expert_partition(x, ids, weights, bits, map, count, cx, cids, cweights, nullptr);
    }
    cuda_synchronize();
    const std::vector<std::int32_t> map = from_device<std::int32_t>(compact_map.p, tokens);
    // Compact columns the join must not read hold NaN.
    std::vector<float> helper(tokens * kHidden, std::numeric_limits<float>::quiet_NaN());
    std::vector<int> participants;
    for (int t = 0; t < c.tokens; ++t) {
        if (map[static_cast<std::size_t>(t)] >= 0) { participants.push_back(t); }
    }
    // The helper's share follows the same activation profile (the R9700 helper's A8 prefill
    // applies the same codec), so for A8 it is the represented-activation oracle.
    const std::vector<std::vector<double>> remote = oracle_tokens(
        fixture, state, inputs, participants, Part::Helper, &local,
        c.profile == MoeActivationProfile::A8 ? OracleInput::RepresentedA8 : OracleInput::Bf16);
    for (std::size_t i = 0; i < participants.size(); ++i) {
        const auto column =
            static_cast<std::size_t>(map[static_cast<std::size_t>(participants[i])]) * kHidden;
        for (std::size_t n = 0; n < kHidden; ++n) {
            helper[column + n] = static_cast<float>(remote[i][n]);
        }
    }
    const DeviceBuffer helper_device = to_device(helper);
    GuardedDeviceBuffer y(tokens * kHidden * sizeof(std::uint16_t));
    y.fill(0xff);
    {
        const DeviceBuffer partial_device = to_device(partial);
        Tensor partial_view(partial_device.p, DType::FP32, {kHidden, c.tokens});
        Tensor map_view(compact_map.p, DType::I32, {c.tokens});
        Tensor helper_view(helper_device.p, DType::FP32, {kHidden, c.tokens});
        Tensor y_view(y.data(), DType::BF16, {kHidden, c.tokens});
        ops::moe_expert_partial_join(partial_view, map_view, helper_view, y_view, nullptr);
        cuda_synchronize();
    }
    std::vector<int> all(tokens);
    std::iota(all.begin(), all.end(), 0);
    failures +=
        verify_output(label + " joined", fixture, state, inputs,
                      from_device<std::uint16_t>(y.data(), tokens * kHidden), all, c.profile);
    failures += y.verify_guards(label + " joined y");

    if (c.deferred >= 0) {
        // The deferral of run_case on the local partial: deferred table entries name the poisoned
        // record and their slots receive the true records only after a host sleep on a second
        // stream; non-local deferred experts must stay unread.
        DeferredRecords deferred(fixture, state, inputs, static_cast<std::size_t>(c.deferred));
        deferred.poison_table(run);
        const Stream main;
        const Stream copies;
        const ops::MoeExpertsDeferral deferral = deferred.deferral();
        deferred.poison_slots();
        run.poison_output();
        deferred.enqueue_copies(copies.get());
        run.invoke_local(main.get(), mask, &deferral);
        cuda_synchronize();
        failures += verify_exact((label + " deferred").c_str(), run.partial_output(), partial);
        failures += deferred.verify_guards(label);
        run.upload(state);
        unmap_remote(run, fixture, state, local);
    }

    if (c.graph) {
        const Stream stream;
        const Graph graph(stream.get(), [&](cudaStream_t s) { run.invoke_local(s, mask); });
        run.poison_output();
        graph.replay(stream.get());
        failures += verify_exact((label + " graph replay").c_str(), run.partial_output(), partial);

        const std::vector<bool> changed = make_locality(Locality::Random, c.seed + 8);
        run.upload(state);
        unmap_remote(run, fixture, state, changed);
        const std::vector<std::int32_t> words = locality_words(changed);
        mask.copy_from_host(words.data(), mask.bytes);
        run.poison_output();
        graph.replay(stream.get());
        const std::vector<float> replayed = run.partial_output();
        failures += verify_partial(label + " replay after bitset change", fixture, state, inputs,
                                   changed, replayed, c.profile);
        run.poison_output();
        run.invoke_local(nullptr, mask);
        cuda_synchronize();
        failures += verify_exact((label + " eager after bitset change").c_str(),
                                 run.partial_output(), replayed);
    }
    return failures;
}

int invalid_argument_cases(const Fixture& fixture) {
    const BankState state = make_bank_state(RecordSet::Standard, 0, false, 3U);
    const Inputs inputs   = make_inputs(fixture, state, 2, Routing::Skewed, true, 3U);
    Run run(fixture, inputs, MoeActivationProfile::A16);
    run.upload(state);
    // Validation rejects these calls before the workspace is used.
    WorkspaceArena workspace(256);
    const auto call = [&](DType id_dtype, void* y_data) {
        Tensor x(run.x_data(), DType::BF16, {kHidden, 2});
        Tensor ids(run.ids_data(), id_dtype, {kTopK, 2});
        Tensor weights(run.weights_data(), DType::FP32, {kTopK, 2});
        Tensor scale(run.shared_scale_data(), DType::FP32, {2});
        Tensor y(y_data, DType::BF16, {kHidden, 2});
        ops::moe_experts(x, ids, weights, scale, run.bank(), MoeActivationProfile::A16, y,
                         workspace, nullptr);
    };
    int failures = 0;
    failures += expect_invalid_argument([&] { call(DType::FP32, run.y_data()); },
                                        "moe_experts accepted FP32 expert ids");
    failures += expect_invalid_argument([&] { call(DType::I32, run.x_data()); },
                                        "moe_experts accepted y aliasing x");
    failures += expect_invalid_argument(
        [] {
            (void)ops::moe_experts_workspace_capacity_bytes(ops::kMoeMaxTokens + 1,
                                                            MoeActivationProfile::A8);
        },
        "moe_experts_workspace_capacity_bytes accepted T=16385");
    failures += expect_invalid_argument(
        [] { (void)ops::moe_experts_workspace_capacity_bytes(0, MoeActivationProfile::A16); },
        "moe_experts_workspace_capacity_bytes accepted T=0");
    const DeviceBuffer mask = to_device(locality_words(make_locality(Locality::All, 0U)));
    const auto call_local   = [&](DType partial_dtype, std::int32_t mask_words) {
        Tensor x(run.x_data(), DType::BF16, {kHidden, 2});
        Tensor ids(run.ids_data(), DType::I32, {kTopK, 2});
        Tensor weights(run.weights_data(), DType::FP32, {kTopK, 2});
        Tensor scale(run.shared_scale_data(), DType::FP32, {2});
        Tensor bits(mask.p, DType::I32, {mask_words});
        Tensor partial(run.y_data(), partial_dtype, {kHidden, 2});
        ops::moe_experts_local_partial(x, ids, weights, scale, run.bank(),
                                       MoeActivationProfile::A16, bits, partial, workspace,
                                       nullptr);
    };
    failures += expect_invalid_argument([&] { call_local(DType::BF16, ops::kMoeExpertMaskWords); },
                                        "moe_experts_local_partial accepted a BF16 partial");
    failures += expect_invalid_argument([&] { call_local(DType::FP32, 8); },
                                        "moe_experts_local_partial accepted an I32 [8] bitset");
    return failures;
}

int run_full(const Fixture& fixture) {
    int failures = invalid_argument_cases(fixture);
    // Token-tile route boundaries are at T = 256/257 and 1024/1025; above T = 16 a concentrated
    // routing splits each expert into several jobs. T = 16, 20 and 48 are verification extents
    // (four, five and twelve lanes of four tokens); distinct routing there gives more work items
    // than one wave of CTAs, in the deferred range too when 40 experts are deferred.
    const std::vector<Case> cases = {
        {1, Routing::Skewed, false, RecordSet::Standard, 48, false, -1, 11U},
        {1, Routing::Skewed, true, RecordSet::Standard, 48, false, 6, 13U},
        {1, Routing::Skewed, true, RecordSet::Standard, 48, true, 0, 15U},
        {2, Routing::Skewed, true, RecordSet::Standard, 48, false, -1, 17U},
        {12, Routing::Distinct, true, RecordSet::Standard, 48, true, 6, 19U},
        {12, Routing::Concentrated, true, RecordSet::Standard, 48, false, -1, 23U},
        {16, Routing::Skewed, true, RecordSet::Standard, 48, false, 6, 73U},
        {16, Routing::Distinct, true, RecordSet::Standard, 48, true, 0, 75U},
        {17, Routing::Concentrated, true, RecordSet::Standard, 48, false, 6, 27U},
        {20, Routing::Skewed, true, RecordSet::Standard, 48, false, 0, 77U},
        {20, Routing::Distinct, true, RecordSet::Standard, 48, false, 40, 79U},
        {12, Routing::Skewed, true, RecordSet::Subnormal, 48, false, -1, 25U},
        {48, Routing::Skewed, true, RecordSet::Standard, 48, true, 40, 85U},
        {48, Routing::Distinct, false, RecordSet::Standard, 48, false, 0, 83U},
        {64, Routing::Skewed, true, RecordSet::Standard, 64, false, -1, 29U},
        {65, Routing::Skewed, true, RecordSet::Standard, 65, false, 6, 31U},
        {65, Routing::Skewed, true, RecordSet::Narrow, 65, false, -1, 33U},
        {256, Routing::Skewed, true, RecordSet::Standard, 48, false, -1, 37U},
        {257, Routing::Skewed, true, RecordSet::Standard, 48, false, -1, 41U},
        {300, Routing::Skewed, true, RecordSet::Standard, 48, false, 6, 43U},
        {1000, Routing::Skewed, true, RecordSet::Standard, 48, false, -1, 47U},
        {1024, Routing::Skewed, false, RecordSet::Standard, 48, false, -1, 53U},
        {1025, Routing::Skewed, true, RecordSet::Standard, 48, false, 6, 59U},
        {1025, Routing::Skewed, true, RecordSet::Subnormal, 48, false, -1, 61U},
        {4096, Routing::Skewed, true, RecordSet::Standard, 48, true, -1, 67U},
        {8192, Routing::Skewed, true, RecordSet::Standard, 48, false, -1, 71U},
        {16384, Routing::Skewed, false, RecordSet::Standard, 48, false, -1, 87U},
        // A8 (one 64-token tile at every T): a single token with a deferral, a concentrated
        // routing in one job per expert with a graph replay after a table change and in two jobs
        // per expert (64 + 1) with narrow scales (every block scale within a binade step, so a
        // misapplied block scale is not hidden by the scale spread), subnormal scales, outlier
        // activations (subnormal codes), distinct routing with more deferred work items than one
        // wave, and the prefill extents up to a 16,384-token super-chunk.
        {1, Routing::Skewed, true, RecordSet::Standard, 48, false, 6, 301U,
         MoeActivationProfile::A8},
        {17, Routing::Concentrated, true, RecordSet::Standard, 48, true, -1, 303U,
         MoeActivationProfile::A8},
        {48, Routing::Distinct, true, RecordSet::Standard, 48, false, 40, 305U,
         MoeActivationProfile::A8},
        {65, Routing::Concentrated, true, RecordSet::Narrow, 65, false, 6, 307U,
         MoeActivationProfile::A8},
        {300, Routing::Skewed, true, RecordSet::Subnormal, 48, false, -1, 309U,
         MoeActivationProfile::A8},
        {1025, Routing::Skewed, true, RecordSet::Standard, 48, false, 6, 311U,
         MoeActivationProfile::A8, true},
        {4096, Routing::Skewed, true, RecordSet::Standard, 48, true, 40, 313U,
         MoeActivationProfile::A8},
        {16384, Routing::Skewed, false, RecordSet::Standard, 48, false, -1, 315U,
         MoeActivationProfile::A8},
    };
    for (const Case& c : cases) { failures += run_case(fixture, c); }
    // The helper round's extents: one decode token, a 4-column verify lane, and twelve lanes
    // (moe_expert_partition's range), also under A8.
    const std::vector<SplitCase> split_cases = {
        {1, Routing::Skewed, Locality::Random, false, 6, 201U},
        {4, Routing::Skewed, Locality::Random, true, -1, 203U},
        {12, Routing::Concentrated, Locality::Random, false, 6, 205U},
        {20, Routing::Distinct, Locality::None, false, -1, 207U},
        {48, Routing::Skewed, Locality::Random, true, 40, 209U},
        {48, Routing::Distinct, Locality::All, false, -1, 211U},
        {48, Routing::Skewed, Locality::Random, true, 40, 213U, MoeActivationProfile::A8},
        {48, Routing::Distinct, Locality::All, false, -1, 215U, MoeActivationProfile::A8},
    };
    for (const SplitCase& c : split_cases) { failures += run_split_case(fixture, c); }
    return failures;
}

// Reduced cases running every token tile, codec, placement and activation profile once, the
// deferral at a single job, at split jobs, with nothing deferred, and with more deferred work
// items than one wave of CTAs, and the expert-helper split at a verify lane and at twelve lanes,
// for compute-sanitizer.
int run_sanitizer(const Fixture& fixture) {
    const std::vector<Case> cases = {
        {1, Routing::Skewed, true, RecordSet::Standard, 1, false, 6, 101U},
        {16, Routing::Skewed, true, RecordSet::Standard, 4, false, 0, 111U},
        {17, Routing::Concentrated, true, RecordSet::Standard, 4, false, 6, 103U},
        {48, Routing::Distinct, true, RecordSet::Standard, 4, false, 40, 113U},
        {257, Routing::Skewed, true, RecordSet::Standard, 4, false, -1, 107U},
        {1025, Routing::Skewed, true, RecordSet::Standard, 4, false, -1, 109U},
        {65, Routing::Concentrated, true, RecordSet::Standard, 4, false, 6, 115U,
         MoeActivationProfile::A8},
        {300, Routing::Skewed, true, RecordSet::Standard, 4, false, -1, 117U,
         MoeActivationProfile::A8},
    };
    int failures = 0;
    for (const Case& c : cases) { failures += run_case(fixture, c); }
    const std::vector<SplitCase> split_cases = {
        {4, Routing::Skewed, Locality::Random, false, 6, 221U},
        {48, Routing::Distinct, Locality::Random, false, -1, 223U},
        {48, Routing::Distinct, Locality::Random, false, -1, 225U, MoeActivationProfile::A8},
    };
    for (const SplitCase& c : split_cases) { failures += run_split_case(fixture, c); }
    return failures;
}

} // namespace

int main(int argc, char** argv) {
    if (require_cuda() != 0) { return 1; }
    try {
        int failures       = 0;
        const auto fixture = make_fixture(failures);
        failures += sanitizer_scope(argc, argv) ? run_sanitizer(*fixture) : run_full(*fixture);
        std::cout << (failures == 0 ? "OK" : "FAIL") << " moe_experts\n";
        return failures == 0 ? 0 : 1;
    } catch (const std::exception& error) {
        std::cerr << "moe_experts: " << error.what() << '\n';
        return 1;
    }
}
