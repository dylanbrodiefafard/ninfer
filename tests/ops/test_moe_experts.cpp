// moe_experts Op test against an independent FP64 oracle at the Qwen3.8-Flash-Next geometry.
//
// Each test expert is generated as canonical codes and E4M3 scales, written into the documented
// record layout by a test-side packer, and read back by an independent reader that locates every
// (matrix, row, K) value from the header's byte formulas; the round trip must be exact. The oracle
// decodes the canonical codes (through the matrix's F3 codebook or as E2M1) times the exact E4M3
// scale, applies the FP32 matrix scales and evaluates the complete routed + shared SwiGLU sum in
// FP64 from the represented BF16 x.
#include "ninfer/ops/moe_experts.h"
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

double e4m3_value(std::uint8_t bits) {
    const int exponent     = (bits >> 3) & 0xF;
    const int mantissa     = bits & 0x7;
    const double magnitude = exponent == 0 ? std::ldexp(mantissa / 8.0, -6)
                                           : std::ldexp(1.0 + mantissa / 8.0, exponent - 7);
    return (bits & 0x80) != 0 ? -magnitude : magnitude;
}

double e2m1_value(std::uint8_t nibble) {
    constexpr double kMagnitudes[8] = {0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0};
    const double magnitude          = kMagnitudes[nibble & 7];
    return (nibble & 8) != 0 ? -magnitude : magnitude;
}

struct MatrixDims {
    int rows;
    int k;
};

constexpr MatrixDims kDims[3] = {{kInter, kHidden}, {kInter, kHidden}, {kHidden, kInter}};

void write_u32(std::vector<std::uint8_t>& bytes, std::size_t offset, std::uint32_t value) {
    for (int b = 0; b < 4; ++b) { bytes[offset + b] = static_cast<std::uint8_t>(value >> (8 * b)); }
}

std::uint32_t read_u32(const std::vector<std::uint8_t>& bytes, std::size_t offset) {
    std::uint32_t value = 0;
    for (int b = 0; b < 4; ++b) {
        value |= static_cast<std::uint32_t>(bytes[offset + b]) << (8 * b);
    }
    return value;
}

// Canonical expert: per matrix m, codes [rows][K] (F3: 3-bit book index; NVFP4: E2M1 nibble) and
// scales [rows][K/group].
struct Canonical {
    MoeExpertCodec codec;
    std::array<std::vector<std::uint8_t>, 3> codes;
    std::array<std::vector<std::uint8_t>, 3> scales;

    int group() const { return codec == MoeExpertCodec::F3G32 ? 32 : 16; }
};

// Block-scale distributions. Wide: E4M3 0.5..448 with occasional subnormal and zero scales.
// Subnormal: only subnormal scales (2^-9 .. 7 * 2^-9), so a wrong subnormal decode changes every
// value. Narrow: 2..4, a single binade step.
enum class ScaleMode : std::uint8_t { Wide, Subnormal, Narrow };

Canonical make_canonical(MoeExpertCodec codec, ScaleMode mode, std::uint32_t seed) {
    std::mt19937 rng(seed);
    Canonical expert{codec, {}, {}};
    std::uniform_int_distribution<int> code(0, codec == MoeExpertCodec::F3G32 ? 7 : 15);
    std::uniform_int_distribution<int> normal_scale(0x30, 0x7E); // 0.5 .. 448
    std::uniform_int_distribution<int> subnormal_scale(0x01, 0x07);
    std::uniform_int_distribution<int> narrow_scale(0x40, 0x48); // 2 .. 4
    std::uniform_int_distribution<int> roll(0, 255);
    for (int m = 0; m < 3; ++m) {
        const auto elements = static_cast<std::size_t>(kDims[m].rows) * kDims[m].k;
        expert.codes[m].resize(elements);
        for (auto& value : expert.codes[m]) { value = static_cast<std::uint8_t>(code(rng)); }
        expert.scales[m].resize(elements / static_cast<std::size_t>(expert.group()));
        for (auto& value : expert.scales[m]) {
            int bits = 0;
            if (mode == ScaleMode::Subnormal) {
                bits = subnormal_scale(rng);
            } else if (mode == ScaleMode::Narrow) {
                bits = narrow_scale(rng);
            } else {
                const int r = roll(rng);
                bits        = r == 0 ? 0 : (r < 4 ? subnormal_scale(rng) : normal_scale(rng));
            }
            value = static_cast<std::uint8_t>(bits);
        }
    }
    return expert;
}

// Logical location of local row rho of plane tile r.
struct RowRef {
    int matrix;
    int row;
};

RowRef plane_row(bool down_plane, int tile, int rho) {
    if (down_plane) { return {2, 32 * tile + rho}; }
    return rho < 16 ? RowRef{0, 16 * tile + rho} : RowRef{1, 16 * tile + rho - 16};
}

// Packer: walks the physical layout (plane, tile, chunk, lane, unit) and writes every byte.
std::vector<std::uint8_t> pack_record(const Canonical& expert) {
    const bool f3          = expert.codec == MoeExpertCodec::F3G32;
    const std::size_t unit = static_cast<std::size_t>(ops::moe_expert_chunk_bytes(expert.codec));
    std::vector<std::uint8_t> record(
        static_cast<std::size_t>(ops::moe_expert_record_bytes(expert.codec)), 0);
    for (int plane = 0; plane < 2; ++plane) {
        const bool down  = plane == 1;
        const int tiles  = down ? 80 : 40;
        const int chunks = down ? 5 : 20;
        const int first  = down ? 800 : 0;
        for (int tile = 0; tile < tiles; ++tile) {
            for (int q = 0; q < chunks; ++q) {
                const std::size_t base = static_cast<std::size_t>(first + tile * chunks + q) * unit;
                for (int lane = 0; lane < 32; ++lane) {
                    const int g = lane / 4, c = lane % 4;
                    for (int k = 0; k < 4; ++k) {
                        const RowRef ref   = plane_row(down, tile, g + 8 * k);
                        const int width    = kDims[ref.matrix].k;
                        const int k_base   = 128 * q + 32 * c;
                        const auto& codes  = expert.codes[ref.matrix];
                        const auto& scales = expert.scales[ref.matrix];
                        const auto code_at = [&](int kk) {
                            return codes[static_cast<std::size_t>(ref.row) * width + kk];
                        };
                        if (f3) {
                            record[base + static_cast<std::size_t>(4 * lane + k)] =
                                scales[static_cast<std::size_t>(ref.row) * (width / 32) +
                                       k_base / 32];
                            std::uint32_t words[3] = {0, 0, 0};
                            for (int i = 0; i < 32; ++i) {
                                const std::uint32_t code = code_at(k_base + i);
                                if (i < 24) {
                                    words[i / 8] |= code << (4 * (i % 8));
                                } else {
                                    for (int j = 0; j < 3; ++j) {
                                        words[j] |= ((code >> j) & 1U) << (4 * (i - 24) + 3);
                                    }
                                }
                            }
                            for (int j = 0; j < 3; ++j) {
                                write_u32(record,
                                          base + static_cast<std::size_t>(
                                                     128 + 16 * (32 * j + lane) + 4 * k),
                                          words[j]);
                            }
                        } else {
                            for (int h = 0; h < 2; ++h) {
                                record[base + static_cast<std::size_t>(8 * lane + 2 * k + h)] =
                                    scales[static_cast<std::size_t>(ref.row) * (width / 16) +
                                           (k_base + 16 * h) / 16];
                            }
                            for (int j = 0; j < 4; ++j) {
                                for (int b = 0; b < 4; ++b) {
                                    const int kk = k_base + 8 * j + 2 * b;
                                    record[base + static_cast<std::size_t>(
                                                      256 + 16 * (32 * j + lane) + 4 * k + b)] =
                                        static_cast<std::uint8_t>(code_at(kk) |
                                                                  (code_at(kk + 1) << 4));
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    return record;
}

// Independent reader: locates the code and scale of (matrix, row, K) from the header formulas.
struct CodeAndScale {
    std::uint8_t code;
    std::uint8_t scale;
};

CodeAndScale read_record(const std::vector<std::uint8_t>& record, MoeExpertCodec codec, int matrix,
                         int row, int k) {
    const bool down  = matrix == 2;
    const int tile   = down ? row / 32 : row / 16;
    const int rho    = down ? row % 32 : row % 16 + (matrix == 1 ? 16 : 0);
    const int chunks = down ? 5 : 20;
    const int q = k / 128, kappa = k % 128, c = kappa / 32, i = kappa % 32;
    const int lane = 4 * (rho % 8) + c, unit = rho / 8;
    const std::size_t base = static_cast<std::size_t>((down ? 800 : 0) + tile * chunks + q) *
                             static_cast<std::size_t>(ops::moe_expert_chunk_bytes(codec));
    if (codec == MoeExpertCodec::F3G32) {
        const std::uint8_t scale = record[base + static_cast<std::size_t>(4 * lane + unit)];
        const auto word          = [&](int j) {
            return read_u32(record,
                            base + static_cast<std::size_t>(128 + 16 * (32 * j + lane) + 4 * unit));
        };
        if (i < 24) {
            return {static_cast<std::uint8_t>((word(i / 8) >> (4 * (i % 8))) & 7U), scale};
        }
        std::uint32_t code = 0;
        for (int j = 0; j < 3; ++j) { code |= ((word(j) >> (4 * (i - 24) + 3)) & 1U) << j; }
        return {static_cast<std::uint8_t>(code), scale};
    }
    const std::uint8_t scale =
        record[base + static_cast<std::size_t>(8 * lane + 2 * unit + i / 16)];
    const std::uint8_t byte =
        record[base +
               static_cast<std::size_t>(256 + 16 * (32 * (i / 8) + lane) + 4 * unit + (i % 8) / 2)];
    return {static_cast<std::uint8_t>((i % 2) != 0 ? byte >> 4 : byte & 0xF), scale};
}

// One distinct test expert: its record and its decoded values code * E4M3 scale (exact in FP32).
struct Expert {
    MoeExpertCodec codec;
    std::vector<std::uint8_t> record;
    std::array<std::vector<float>, 3> values;
};

Expert make_expert(MoeExpertCodec codec, ScaleMode mode, std::uint32_t seed,
                   const std::array<std::uint32_t, 3>& books, int& failures) {
    const Canonical canonical = make_canonical(codec, mode, seed);
    Expert expert{codec, pack_record(canonical), {}};
    bool round_trip = true;
    for (int m = 0; m < 3; ++m) {
        const int rows = kDims[m].rows, width = kDims[m].k;
        expert.values[m].resize(static_cast<std::size_t>(rows) * width);
        for (int row = 0; row < rows; ++row) {
            for (int k = 0; k < width; ++k) {
                const std::size_t index = static_cast<std::size_t>(row) * width + k;
                const std::uint8_t code = canonical.codes[m][index];
                const std::uint8_t scale =
                    canonical.scales[m][index / static_cast<std::size_t>(canonical.group())];
                const CodeAndScale read = read_record(expert.record, codec, m, row, k);
                round_trip &= read.code == code && read.scale == scale;
                const double element =
                    codec == MoeExpertCodec::F3G32
                        ? e2m1_value(static_cast<std::uint8_t>((books[m] >> (4 * code)) & 0xFU))
                        : e2m1_value(code);
                expert.values[m][index] = static_cast<float>(element * e4m3_value(scale));
            }
        }
    }
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

Inputs make_inputs(const Fixture& fixture, const BankState& state, int tokens, Routing routing,
                   bool shared, std::uint32_t seed) {
    std::mt19937 rng(seed);
    std::normal_distribution<double> normal(0.0, 1.0);
    const auto extent = static_cast<std::size_t>(tokens);
    Inputs inputs{tokens, std::vector<float>(extent * kHidden),
                  std::vector<std::int32_t>(extent * kTopK), std::vector<float>(extent * kTopK),
                  std::vector<float>(extent, 0.0F)};
    fill_uniform(inputs.x, seed + 1, -2.0F, 2.0F);
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

// FP64 oracle of one token.
std::vector<double> oracle_token(const Fixture& fixture, const BankState& state,
                                 const Inputs& inputs, int t) {
    const float* x = inputs.x.data() + static_cast<std::size_t>(t) * kHidden;
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
        const Expert& expert   = fixture.experts[static_cast<std::size_t>(state.record[id])];
        const double scale[3]  = {state.matrix_scales[id * 3], state.matrix_scales[id * 3 + 1],
                                  state.matrix_scales[id * 3 + 2]};
        expert_output(expert.values[0].data(), expert.values[1].data(), expert.values[2].data(),
                      scale, inputs.weights[slot]);
    }
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

// Compares the sampled tokens of y (BF16 bits [T][2560]) against the oracle.
int verify_output(const std::string& label, const Fixture& fixture, const BankState& state,
                  const Inputs& inputs, const std::vector<std::uint16_t>& bits,
                  const std::vector<int>& sampled) {
    std::vector<std::vector<double>> expected(sampled.size());
    {
        std::vector<std::jthread> workers;
        workers.reserve(kOracleThreads);
        for (std::size_t w = 0; w < kOracleThreads; ++w) {
            workers.emplace_back([&, w] {
                for (std::size_t i = w; i < sampled.size(); i += kOracleThreads) {
                    expected[i] = oracle_token(fixture, state, inputs, sampled[i]);
                }
            });
        }
    }
    std::vector<double> got;
    std::vector<double> reference;
    for (std::size_t i = 0; i < sampled.size(); ++i) {
        const auto row = static_cast<std::size_t>(sampled[i]) * kHidden;
        for (std::size_t n = 0; n < kHidden; ++n) {
            got.push_back(bf16_to_f32(bits[row + n]));
            reference.push_back(expected[i][n]);
        }
    }
    return verify_reduction(label, got, reference, output_criterion());
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
    Run(const Fixture& fixture, const Inputs& inputs)
        : fixture_(fixture), tokens_(inputs.tokens), x_(to_device_bf16(inputs.x)),
          ids_(to_device(inputs.ids)), weights_(to_device(inputs.weights)),
          shared_scale_(to_device(inputs.shared_scale)),
          table_(static_cast<std::size_t>(kExperts) * sizeof(std::int64_t)),
          scales_(static_cast<std::size_t>(kExperts) * 3 * sizeof(float)),
          shared_scales_(3 * sizeof(float)),
          y_(static_cast<std::size_t>(inputs.tokens) * kHidden * sizeof(std::uint16_t)),
          workspace_(ops::moe_experts_workspace_capacity_bytes(inputs.tokens)) {}

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
        ops::moe_experts(x, ids, weights, scale, bank(), y, workspace_, stream, deferral);
    }

    // Poisons y and orders the poison before work on any stream.
    void poison_output() {
        y_.fill(0xff);
        cuda_synchronize();
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
    int tokens_;
    DeviceBuffer x_, ids_, weights_, shared_scale_, table_, scales_, shared_scales_;
    GuardedDeviceBuffer y_;
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
};

std::string describe(const Case& c) {
    const char* routing = c.routing == Routing::Skewed     ? "skewed"
                          : c.routing == Routing::Distinct ? "distinct"
                                                           : "concentrated";
    const char* records = c.records == RecordSet::Standard    ? ""
                          : c.records == RecordSet::Subnormal ? " subnormal-scales"
                                                              : " narrow-scales";
    return "moe_experts T=" + std::to_string(c.tokens) + " " + routing +
           (c.mixed ? " mixed-host" : " device") + records + (c.graph ? " graph" : "") +
           (c.deferred < 0 ? "" : " deferred=" + std::to_string(c.deferred));
}

int run_case(const Fixture& fixture, const Case& c) {
    const std::string label = describe(c);
    const BankState state   = make_bank_state(c.records, 0, c.mixed, c.seed + 5);
    const Inputs inputs =
        make_inputs(fixture, state, c.tokens, c.routing, c.records == RecordSet::Standard, c.seed);
    const std::vector<int> sampled = sample_tokens(c.tokens, c.samples, c.seed + 9);
    Run run(fixture, inputs);
    run.upload(state);
    run.poison_output();
    run.invoke(nullptr);
    cuda_synchronize();
    const std::vector<std::uint16_t> bits = run.output();
    int failures = verify_output(label, fixture, state, inputs, bits, sampled);
    failures += run.verify_guards(label + " y");

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
                                  replayed, sampled);
        run.poison_output();
        run.invoke(nullptr);
        cuda_synchronize();
        failures +=
            verify_exact((label + " eager after table change").c_str(), run.output(), replayed);
    }
    return failures;
}

int invalid_argument_cases(const Fixture& fixture) {
    const BankState state = make_bank_state(RecordSet::Standard, 0, false, 3U);
    const Inputs inputs   = make_inputs(fixture, state, 2, Routing::Skewed, true, 3U);
    Run run(fixture, inputs);
    run.upload(state);
    // Validation rejects these calls before the workspace is used.
    WorkspaceArena workspace(256);
    const auto call = [&](DType id_dtype, void* y_data) {
        Tensor x(run.x_data(), DType::BF16, {kHidden, 2});
        Tensor ids(run.ids_data(), id_dtype, {kTopK, 2});
        Tensor weights(run.weights_data(), DType::FP32, {kTopK, 2});
        Tensor scale(run.shared_scale_data(), DType::FP32, {2});
        Tensor y(y_data, DType::BF16, {kHidden, 2});
        ops::moe_experts(x, ids, weights, scale, run.bank(), y, workspace, nullptr);
    };
    int failures = 0;
    failures += expect_invalid_argument([&] { call(DType::FP32, run.y_data()); },
                                        "moe_experts accepted FP32 expert ids");
    failures += expect_invalid_argument([&] { call(DType::I32, run.x_data()); },
                                        "moe_experts accepted y aliasing x");
    failures += expect_invalid_argument(
        [] { (void)ops::moe_experts_workspace_capacity_bytes(ops::kMoeMaxTokens + 1); },
        "moe_experts_workspace_capacity_bytes accepted T=8193");
    failures += expect_invalid_argument([] { (void)ops::moe_experts_workspace_capacity_bytes(0); },
                                        "moe_experts_workspace_capacity_bytes accepted T=0");
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
    };
    for (const Case& c : cases) { failures += run_case(fixture, c); }
    return failures;
}

// Reduced cases running every token tile, codec and placement once, and the deferral at a single
// job, at split jobs, with nothing deferred, and with more deferred work items than one wave of
// CTAs, for compute-sanitizer.
int run_sanitizer(const Fixture& fixture) {
    const std::vector<Case> cases = {
        {1, Routing::Skewed, true, RecordSet::Standard, 1, false, 6, 101U},
        {16, Routing::Skewed, true, RecordSet::Standard, 4, false, 0, 111U},
        {17, Routing::Concentrated, true, RecordSet::Standard, 4, false, 6, 103U},
        {48, Routing::Distinct, true, RecordSet::Standard, 4, false, 40, 113U},
        {257, Routing::Skewed, true, RecordSet::Standard, 4, false, -1, 107U},
        {1025, Routing::Skewed, true, RecordSet::Standard, 4, false, -1, 109U},
    };
    int failures = 0;
    for (const Case& c : cases) { failures += run_case(fixture, c); }
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
