#include "ninfer/ops/ple.h"
#include "ops/op_tester.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <exception>
#include <iostream>
#include <string>
#include <vector>

#include <cuda_fp16.h>

using namespace ninfer;
using namespace ninfer::test;

namespace {

constexpr std::int32_t kHidden   = ops::kPleHidden;
constexpr std::int32_t kStreams  = ops::kPleStreams;
constexpr std::int32_t kChannels = ops::kPleChannels;
constexpr std::int32_t kHistory  = ops::kPleConvHistory;
constexpr std::int32_t kTaps     = ops::kPleConvWidth;

// R is an FP32 update rounded once to BF16 (relative 2^-8). A convolution input near a BF16
// rounding midpoint can round to a neighbour on the device (FP32) and in the oracle (FP64): one
// BF16 ulp of |C| <= 2.25 (2^-6) through a tap weight <= 0.6 and SiLU' <= 1.1 moves R by up to
// 1.04e-2, the absolute term.
constexpr PointwiseCriterion residual_criterion() {
    return {/*absolute*/ 1.04e-2, /*relative*/ 4.05e-3};
}

// The carried history holds BF16 convolution inputs, which may differ by that one BF16 ulp:
// relative 2^-7 at the bottom of a binade.
constexpr PointwiseCriterion state_criterion() {
    return {/*absolute*/ 1.0e-6, /*relative*/ 7.82e-3};
}

std::vector<std::uint16_t> encode_bf16(const std::vector<float>& values) {
    std::vector<std::uint16_t> bits(values.size());
    for (std::size_t i = 0; i < values.size(); ++i) { bits[i] = f32_to_bf16(values[i]); }
    return bits;
}

double sigmoid(double v) { return 1.0 / (1.0 + std::exp(-v)); }

double bf16_round(double v) { return bf16_to_f32(f32_to_bf16(static_cast<float>(v))); }

template <class T>
GuardedDeviceBuffer upload(const std::vector<T>& values) {
    GuardedDeviceBuffer buffer(values.size() * sizeof(T));
    buffer.copy_from_host(values.data(), buffer.bytes());
    return buffer;
}

int run_decode_rows(std::int32_t tokens, std::uint32_t seed) {
    std::vector<std::uint8_t> rows(static_cast<std::size_t>(ops::kPleInt8RowBytes) *
                                   ops::kPleRowsPerToken * tokens);
    std::vector<float> scales(static_cast<std::size_t>(ops::kPleRowsPerToken) * tokens);
    fill_uniform(scales, seed, 1.0e-4F, 2.0e-3F);
    std::vector<float> codes(static_cast<std::size_t>(ops::kPleRowWidth) * scales.size());
    fill_uniform(codes, seed + 1, -127.49F, 127.49F);
    std::vector<std::uint16_t> expected(static_cast<std::size_t>(ops::kPleEmbedding) * tokens);
    for (std::size_t r = 0; r < scales.size(); ++r) {
        const __half scale   = __float2half_rn(scales[r]);
        std::uint8_t* record = rows.data() + r * ops::kPleInt8RowBytes;
        std::memcpy(record, &scale, sizeof(scale));
        for (std::int32_t j = 0; j < ops::kPleRowWidth; ++j) {
            const auto code =
                static_cast<std::int8_t>(std::lround(codes[r * ops::kPleRowWidth + j]));
            record[2 + j] = static_cast<std::uint8_t>(code);
            expected[r * ops::kPleRowWidth + j] =
                f32_to_bf16(static_cast<float>(code) * __half2float(scale));
        }
    }
    auto device_rows = upload(rows);
    GuardedDeviceBuffer device_embedding(expected.size() * sizeof(std::uint16_t));
    Tensor rows_tensor(device_rows.data(), DType::U8,
                       {ops::kPleInt8RowBytes, ops::kPleRowsPerToken, tokens});
    Tensor embedding_tensor(device_embedding.data(), DType::BF16, {ops::kPleEmbedding, tokens});
    ops::ple_decode_rows(rows_tensor, embedding_tensor, nullptr);
    cuda_synchronize();
    int failures = verify_exact(
        "ple_decode_rows", from_device<std::uint16_t>(device_embedding.data(), expected.size()),
        expected);
    failures += device_embedding.verify_guards("ple_decode_rows embedding");
    return failures;
}

struct InjectInputs {
    std::vector<float> residual; // [10240][W*B] packed columns, BF16-representable
    std::vector<float> key;      // [10240][W*B]
    std::vector<float> value;    // [2560][W*B]
    std::vector<float> norm_key, norm_query, norm_conv, conv;
    std::vector<float> histories; // [S][10240][9]
};

InjectInputs make_inputs(std::int32_t columns, std::int32_t slots, std::uint32_t seed) {
    InjectInputs in;
    in.residual.resize(static_cast<std::size_t>(columns) * kChannels);
    in.key.resize(in.residual.size());
    in.value.resize(static_cast<std::size_t>(columns) * kHidden);
    in.norm_key.resize(kChannels);
    in.norm_query.resize(kChannels);
    in.norm_conv.resize(kChannels);
    in.conv.resize(static_cast<std::size_t>(kChannels) * kTaps);
    in.histories.resize(static_cast<std::size_t>(slots) * kChannels * kHistory);
    fill_uniform(in.residual, seed, -2.0F, 2.0F);
    fill_uniform(in.key, seed + 1, -1.5F, 1.5F);
    fill_uniform(in.value, seed + 2, -0.5F, 0.5F);
    fill_uniform(in.norm_key, seed + 3, -0.3F, 0.3F);
    fill_uniform(in.norm_query, seed + 4, -0.3F, 0.3F);
    fill_uniform(in.norm_conv, seed + 5, -0.3F, 0.3F);
    fill_uniform(in.conv, seed + 6, -0.6F, 0.6F);
    fill_uniform(in.histories, seed + 7, -1.5F, 1.5F);
    for (auto* v :
         {&in.residual, &in.norm_key, &in.norm_query, &in.norm_conv, &in.conv, &in.histories}) {
        round_to_bf16(*v);
    }
    return in;
}

// FP64 oracle of one row: columns `flat` (packed indices, in time order) starting from `history`
// (10240 x 9, positions fastest). Writes the new residual columns into expected_residual and
// returns the history after each column (every_column) or after the last one only.
std::vector<std::vector<double>> oracle_row(const InjectInputs& in,
                                            const std::vector<std::int32_t>& flat,
                                            const std::vector<double>& history,
                                            std::vector<double>& expected_residual,
                                            bool every_column) {
    const std::size_t count = flat.size();
    std::vector<double> c_rows(count * kChannels), gated(count * kChannels);
    for (std::size_t t = 0; t < count; ++t) {
        const std::size_t column = static_cast<std::size_t>(flat[t]);
        for (std::int32_t i = 0; i < kStreams; ++i) {
            const std::size_t base = column * kChannels + std::size_t(i) * kHidden;
            double kk = 0, rr = 0;
            for (std::int32_t d = 0; d < kHidden; ++d) {
                kk += double(in.key[base + d]) * in.key[base + d];
                rr += double(in.residual[base + d]) * in.residual[base + d];
            }
            const double ik = 1.0 / std::sqrt(kk / kHidden + 1e-6);
            const double ir = 1.0 / std::sqrt(rr / kHidden + 1e-6);
            double dot      = 0;
            for (std::int32_t d = 0; d < kHidden; ++d) {
                dot += double(in.key[base + d]) * ik * (1.0 + in.norm_key[i * kHidden + d]) *
                       double(in.residual[base + d]) * ir * (1.0 + in.norm_query[i * kHidden + d]);
            }
            const double s = dot / std::sqrt(double(kHidden));
            const double g =
                (s > 0 ? 1.0 : (s < 0 ? -1.0 : 0.0)) * std::sqrt(std::max(std::fabs(s), 1e-6));
            const double scale      = sigmoid(g);
            double gg               = 0;
            const std::size_t local = t * kChannels + std::size_t(i) * kHidden;
            for (std::int32_t d = 0; d < kHidden; ++d) {
                gated[local + d] = scale * in.value[column * kHidden + d];
                gg += gated[local + d] * gated[local + d];
            }
            const double ig = 1.0 / std::sqrt(gg / kHidden + 1e-6);
            for (std::int32_t d = 0; d < kHidden; ++d) {
                c_rows[local + d] =
                    bf16_round(gated[local + d] * ig * (1.0 + in.norm_conv[i * kHidden + d]));
            }
        }
    }
    const auto c_at = [&](std::int64_t t, std::int32_t c) {
        return t >= 0 ? c_rows[std::size_t(t) * kChannels + c]
                      : history[std::size_t(c) * kHistory + kHistory + t];
    };
    std::vector<std::vector<double>> snapshots(every_column ? count : (count > 0 ? 1 : 0),
                                               std::vector<double>(history.size()));
    for (std::size_t t = 0; t < count; ++t) {
        const std::size_t column = static_cast<std::size_t>(flat[t]);
        for (std::int32_t c = 0; c < kChannels; ++c) {
            double sum = 0;
            for (std::int32_t tap = 0; tap < kTaps; ++tap) {
                sum += double(in.conv[std::size_t(c) * kTaps + tap]) *
                       c_at(static_cast<std::int64_t>(t) - std::int64_t{3} * (kTaps - 1 - tap), c);
            }
            const std::size_t index = column * kChannels + c;
            expected_residual[index] =
                double(in.residual[index]) + gated[t * kChannels + c] + sum * sigmoid(sum);
            if (every_column || t + 1 == count) {
                std::vector<double>& snapshot = snapshots[every_column ? t : 0];
                for (std::int32_t j = 0; j < kHistory; ++j) {
                    snapshot[std::size_t(c) * kHistory + j] =
                        c_at(static_cast<std::int64_t>(t) + 1 - kHistory + j, c);
                }
            }
        }
    }
    return snapshots;
}

struct DeviceInputs {
    GuardedDeviceBuffer residual, key, value, norm_key, norm_query, norm_conv, conv, states;
    ops::PleWeights weights;
};

DeviceInputs upload_inputs(const InjectInputs& in) {
    DeviceInputs d{upload(encode_bf16(in.residual)),
                   upload(in.key),
                   upload(in.value),
                   upload(encode_bf16(in.norm_key)),
                   upload(encode_bf16(in.norm_query)),
                   upload(encode_bf16(in.norm_conv)),
                   upload(encode_bf16(in.conv)),
                   upload(encode_bf16(in.histories)),
                   {}};
    d.weights = {Tensor(d.norm_key.data(), DType::BF16, {kChannels}),
                 Tensor(d.norm_query.data(), DType::BF16, {kChannels}),
                 Tensor(d.norm_conv.data(), DType::BF16, {kChannels}),
                 Tensor(d.conv.data(), DType::BF16, {kTaps, kChannels})};
    return d;
}

enum class Form : std::uint8_t { Final, Snapshot };

// One packed call (or, with `split` > 0 in Final form, two calls over columns [0, split) and
// [split, W) of every row with carried history) of B rows of W columns. Checks the FP64 oracle,
// untouched invalid tails and slots, and bitwise equality with one B=1 call per row.
int run_inject(Form form, std::int32_t width, std::int32_t batch,
               std::vector<std::int32_t> valid_columns, std::int32_t split, std::uint32_t seed) {
    const std::int32_t columns = width * batch;
    const bool dense           = valid_columns.empty();
    if (dense) { valid_columns.assign(static_cast<std::size_t>(batch), width); }
    // Final: row b uses slot 2b + 1 (odd slots stay untouched). Snapshot: row b reads slot
    // b * (W + 1) + W / 2, inside its own interval [b * (W + 1), b * (W + 1) + W).
    std::vector<std::int32_t> initial(static_cast<std::size_t>(batch));
    std::vector<std::int32_t> base(static_cast<std::size_t>(batch));
    for (std::int32_t b = 0; b < batch; ++b) {
        base[std::size_t(b)] = form == Form::Final ? 2 * b + 1 : b * (width + 1);
        initial[std::size_t(b)] =
            form == Form::Final ? base[std::size_t(b)] : base[std::size_t(b)] + width / 2;
    }
    const std::int32_t slots     = form == Form::Final ? 2 * batch + 1 : batch * (width + 1);
    const InjectInputs in        = make_inputs(columns, slots, seed);
    const std::size_t slot_elems = static_cast<std::size_t>(kChannels) * kHistory;

    std::vector<double> expected_residual(in.residual.begin(), in.residual.end());
    std::vector<double> expected_states(in.histories.begin(), in.histories.end());
    for (std::int32_t b = 0; b < batch; ++b) {
        std::vector<std::int32_t> flat;
        flat.reserve(static_cast<std::size_t>(valid_columns[std::size_t(b)]));
        for (std::int32_t j = 0; j < valid_columns[std::size_t(b)]; ++j) {
            flat.push_back(b * width + j);
        }
        const std::size_t init = std::size_t(initial[std::size_t(b)]) * slot_elems;
        std::vector<double> history(in.histories.begin() + static_cast<std::ptrdiff_t>(init),
                                    in.histories.begin() +
                                        static_cast<std::ptrdiff_t>(init + slot_elems));
        const auto snapshots =
            oracle_row(in, flat, history, expected_residual, form == Form::Snapshot);
        if (form == Form::Final) {
            if (!snapshots.empty()) {
                std::copy(snapshots.back().begin(), snapshots.back().end(),
                          expected_states.begin() +
                              static_cast<std::ptrdiff_t>(std::size_t(base[std::size_t(b)]) *
                                                          slot_elems));
            }
        } else {
            for (std::size_t j = 0; j < snapshots.size(); ++j) {
                std::copy(snapshots[j].begin(), snapshots[j].end(),
                          expected_states.begin() +
                              static_cast<std::ptrdiff_t>((std::size_t(base[std::size_t(b)]) + j) *
                                                          slot_elems));
            }
        }
    }

    DeviceInputs batched = upload_inputs(in);
    DeviceInputs per_row = upload_inputs(in);
    auto device_valid    = upload(valid_columns);
    auto device_initial  = upload(initial);
    auto device_base     = upload(base);
    const auto launch    = [&](DeviceInputs& d, std::int32_t row_begin, std::int32_t rows,
                               std::int32_t column_begin, std::int32_t column_count,
                               std::int32_t row_stride, WorkspaceArena& workspace) {
        // A [*, column_count, rows] view of rows [row_begin, row_begin + rows) and columns
        // [column_begin, column_begin + column_count). It is contiguous whenever rows == 1 or
        // column_count == row_stride.
        const std::size_t first = std::size_t(row_begin) * row_stride + column_begin;
        auto* r                 = static_cast<std::uint8_t*>(d.residual.data()) +
                                  first * kChannels * sizeof(std::uint16_t);
        auto* k = static_cast<std::uint8_t*>(d.key.data()) + first * kChannels * sizeof(float);
        auto* v = static_cast<std::uint8_t*>(d.value.data()) + first * kHidden * sizeof(float);
        Tensor residual(r, DType::BF16, {kHidden, kStreams, column_count, rows});
        const Tensor key(k, DType::FP32, {kChannels, column_count, rows});
        const Tensor value(v, DType::FP32, {kHidden, column_count, rows});
        Tensor states(d.states.data(), DType::BF16, {kHistory, kChannels, slots});
        // A chunk's valid extents are relative to its first column, so a split call gets a
        // host-adjusted copy; whole calls use the original selector.
        std::vector<std::int32_t> chunk_valid;
        for (std::int32_t b = row_begin; b < row_begin + rows; ++b) {
            chunk_valid.push_back(
                std::clamp(valid_columns[std::size_t(b)] - column_begin, 0, column_count));
        }
        auto device_chunk_valid = upload(chunk_valid);
        const Tensor valid      = dense && column_begin == 0 && column_count == width
                                      ? Tensor{}
                                      : Tensor(device_chunk_valid.data(), DType::I32, {rows});
        const Tensor init(static_cast<std::int32_t*>(device_initial.data()) + row_begin, DType::I32,
                          {rows});
        const Tensor snap(static_cast<std::int32_t*>(device_base.data()) + row_begin, DType::I32,
                          {rows});
        if (form == Form::Final) {
            ops::ple_inject(residual, key, value, d.weights, states, valid, init, workspace,
                            nullptr);
        } else {
            ops::ple_inject_snapshot(residual, key, value, d.weights, states, valid, init, snap,
                                     workspace, nullptr);
        }
        cuda_synchronize();
    };
    WorkspaceArena workspace(ops::ple_inject_workspace_bytes(columns));
    if (split > 0) {
        // Column chunks of a packed [*, W, B] call are strided, so the split runs row by row.
        for (std::int32_t b = 0; b < batch; ++b) {
            launch(batched, b, 1, 0, split, width, workspace);
            launch(batched, b, 1, split, width - split, width, workspace);
        }
    } else {
        launch(batched, 0, batch, 0, width, width, workspace);
    }
    for (std::int32_t b = 0; b < batch; ++b) { launch(per_row, b, 1, 0, width, width, workspace); }

    const std::string label =
        std::string(form == Form::Final ? "ple_inject" : "ple_inject_snapshot") +
        " B=" + std::to_string(batch) + " W=" + std::to_string(width) +
        (split > 0 ? " split=" + std::to_string(split) : "");
    const auto residual_after = from_device_bf16(batched.residual.data(), in.residual.size());
    const auto states_after   = from_device_bf16(batched.states.data(), in.histories.size());
    int failures = verify_pointwise(label + " residual", residual_after, expected_residual,
                                    residual_criterion());
    failures +=
        verify_pointwise(label + " conv_states", states_after, expected_states, state_criterion());
    // Invalid tails and unwritten slots keep their exact input bits.
    const auto residual_bits =
        from_device<std::uint16_t>(batched.residual.data(), in.residual.size());
    const auto input_bits = encode_bf16(in.residual);
    for (std::int32_t b = 0; b < batch; ++b) {
        for (std::int32_t j = valid_columns[std::size_t(b)]; j < width; ++j) {
            const auto first = static_cast<std::ptrdiff_t>(std::size_t(b * width + j) * kChannels);
            if (!std::equal(input_bits.begin() + first, input_bits.begin() + first + kChannels,
                            residual_bits.begin() + first)) {
                std::cerr << label << ": invalid column " << j << " of row " << b
                          << " was modified\n";
                ++failures;
            }
        }
    }
    if (split > 0) {
        // Chunked calls carried through the history equal one call over the concatenation.
        failures +=
            verify_exact((label + " residual vs one call").c_str(), residual_bits,
                         from_device<std::uint16_t>(per_row.residual.data(), in.residual.size()));
    } else {
        failures +=
            verify_exact((label + " residual vs per-row calls").c_str(), residual_bits,
                         from_device<std::uint16_t>(per_row.residual.data(), in.residual.size()));
    }
    failures +=
        verify_exact((label + " conv_states vs per-row calls").c_str(),
                     from_device<std::uint16_t>(batched.states.data(), in.histories.size()),
                     from_device<std::uint16_t>(per_row.states.data(), in.histories.size()));
    failures += batched.residual.verify_guards(label + " residual");
    failures += batched.states.verify_guards(label + " conv_states");
    return failures;
}

} // namespace

int main() {
    if (require_cuda() != 0) { return 1; }
    try {
        int failures = run_decode_rows(5, 41U);
        // Decode (W=1) at B=1/4/12, a two-call split shorter and longer than the nine-token
        // history, a batched final call that chunks internally (B=12 W=100: 85 columns per
        // chunk) and a single row longer than one chunk, mixed valid prefixes including an empty
        // one, and snapshot calls with each row's initial slot inside its own interval.
        failures += run_inject(Form::Final, 1, 1, {}, 0, 101U);
        failures += run_inject(Form::Final, 1, 4, {}, 0, 103U);
        failures += run_inject(Form::Final, 1, 12, {1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1}, 0, 107U);
        failures += run_inject(Form::Final, 8, 1, {}, 5, 211U);
        failures += run_inject(Form::Final, 19, 1, {}, 12, 307U);
        failures += run_inject(Form::Final, 4, 4, {4, 2, 0, 3}, 0, 401U);
        failures += run_inject(Form::Final, 100, 12, {}, 0, 409U);
        failures += run_inject(Form::Final, 1100, 1, {}, 0, 419U);
        failures += run_inject(Form::Snapshot, 1, 12, {}, 0, 503U);
        failures += run_inject(Form::Snapshot, 4, 4, {4, 1, 3, 2}, 0, 509U);
        failures += run_inject(Form::Snapshot, 16, 3, {16, 9, 0}, 0, 521U);
        std::cout << (failures == 0 ? "OK" : "FAIL") << " ple\n";
        return failures == 0 ? 0 : 1;
    } catch (const std::exception& error) {
        std::cerr << "ple: " << error.what() << '\n';
        return 1;
    }
}
