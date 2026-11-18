#pragma once

// Test-side reference of the routed-expert record ("Routed-expert record" in
// ninfer/ops/moe_experts.h): canonical codes and E4M3 scales, a packer that walks the physical
// layout, an independent reader that locates every (matrix, row, K) value from the header's byte
// formulas, and the exact decoded values. Pure host code, shared by CUDA and HIP Op tests.

#include "ninfer/ops/moe_expert_record.h"

#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <random>
#include <vector>

namespace ninfer::test::moe_record {

using ops::MoeExpertCodec;

inline double e4m3_value(std::uint8_t bits) {
    const int exponent     = (bits >> 3) & 0xF;
    const int mantissa     = bits & 0x7;
    const double magnitude = exponent == 0 ? std::ldexp(mantissa / 8.0, -6)
                                           : std::ldexp(1.0 + mantissa / 8.0, exponent - 7);
    return (bits & 0x80) != 0 ? -magnitude : magnitude;
}

inline double e2m1_value(std::uint8_t nibble) {
    constexpr double kMagnitudes[8] = {0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0};
    const double magnitude          = kMagnitudes[nibble & 7];
    return (nibble & 8) != 0 ? -magnitude : magnitude;
}

struct MatrixDims {
    int rows;
    int k;
};

// Gate and up [640,2560], down [2560,640].
inline constexpr MatrixDims kDims[3] = {{ops::kMoeIntermediate, ops::kMoeHidden},
                                        {ops::kMoeIntermediate, ops::kMoeHidden},
                                        {ops::kMoeHidden, ops::kMoeIntermediate}};

// Canonical expert: per matrix m, codes [rows][K] (F3: 3-bit book index; NVFP4: E2M1 nibble) and
// E4M3 scales [rows][K / group].
struct Canonical {
    MoeExpertCodec codec;
    std::array<std::vector<std::uint8_t>, 3> codes;
    std::array<std::vector<std::uint8_t>, 3> scales;

    [[nodiscard]] int group() const { return codec == MoeExpertCodec::F3G32 ? 32 : 16; }
};

// Block-scale distributions. Wide: E4M3 0.5..448 with occasional zero, subnormal and negative
// scales. Subnormal: only subnormal scales (2^-9 .. 7 * 2^-9), so a wrong subnormal decode changes
// every value. Narrow: 2..4, a single binade step.
enum class ScaleMode : std::uint8_t { Wide, Subnormal, Narrow };

inline Canonical make_canonical(MoeExpertCodec codec, ScaleMode mode, std::uint32_t seed) {
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
                bits        = r == 0   ? 0
                              : r < 4  ? subnormal_scale(rng)
                              : r < 12 ? 0x80 | normal_scale(rng)
                                       : normal_scale(rng);
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

inline RowRef plane_row(bool down_plane, int tile, int rho) {
    if (down_plane) { return {2, 32 * tile + rho}; }
    return rho < 16 ? RowRef{0, 16 * tile + rho} : RowRef{1, 16 * tile + rho - 16};
}

inline void write_u32(std::vector<std::uint8_t>& bytes, std::size_t offset, std::uint32_t value) {
    for (int b = 0; b < 4; ++b) { bytes[offset + b] = static_cast<std::uint8_t>(value >> (8 * b)); }
}

inline std::uint32_t read_u32(const std::vector<std::uint8_t>& bytes, std::size_t offset) {
    std::uint32_t value = 0;
    for (int b = 0; b < 4; ++b) {
        value |= static_cast<std::uint32_t>(bytes[offset + b]) << (8 * b);
    }
    return value;
}

// Packer: walks the physical layout (plane, tile, chunk, lane, unit) and writes every byte.
inline std::vector<std::uint8_t> pack_record(const Canonical& expert) {
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

struct CodeAndScale {
    std::uint8_t code;
    std::uint8_t scale;
};

// Independent reader: locates the code and scale of (matrix, row, K) from the header formulas.
inline CodeAndScale read_record(const std::vector<std::uint8_t>& record, MoeExpertCodec codec,
                                int matrix, int row, int k) {
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

// One test expert: its record bytes and its decoded values code * E4M3 scale (exact in FP32),
// matrices [rows][K] in kDims order.
struct Expert {
    MoeExpertCodec codec;
    std::vector<std::uint8_t> record;
    std::array<std::vector<float>, 3> values;
};

// Builds an expert from canonical codes; `round_trip` is cleared when the reader does not recover
// every packed code and scale.
inline Expert make_expert(MoeExpertCodec codec, ScaleMode mode, std::uint32_t seed,
                          const std::array<std::uint32_t, 3>& books, bool& round_trip) {
    const Canonical canonical = make_canonical(codec, mode, seed);
    Expert expert{codec, pack_record(canonical), {}};
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
    return expert;
}

} // namespace ninfer::test::moe_record
