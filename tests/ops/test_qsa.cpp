// QSA Op tests against independent FP64 and exact oracles at the Qwen3.8-Flash-Next geometry.
//
// The selection oracle follows qwen4:tools/reference/qwen4/qsa.py (`_select` with
// `source_grouped_rmsnorm` and `partial_rope`): FP32 mean of the four raw keys cast to BF16,
// RMSNorm with epsilon 1e-6 and gamma 1 + w, rotate-half rotary pairs (i, i+32) of the first 64
// features at angle position[axis(i)] * 1e7^(-2i/64) with interleaved sections [11,11,10] (axis i %
// 3), block start position for keys and the query's own position for queries, score sum_h ReLU(dot)
// / sqrt(128), and top-budget blocks with ties to the lower block. Two choices differ from qwen4's
// Op: visibility is the causal prefix [0, slot] of the query's own slot instead of a CSR list, and
// selected blocks are listed in ascending block order instead of score order (a set; attention
// does not depend on the order).
//
// The state is paged: every test lane maps its logical pages to a shuffled set of physical pages,
// spare pages belong to no lane, and the oracles read a lane's state gathered back into logical
// slot order. Lane-batched calls are checked against the oracle per lane and bit for bit against
// one call per row.
#include "ninfer/ops/qsa.h"
#include "ops/nvfp4_activation_ref.h"
#include "ops/op_tester.h"
#include "ops/quantized_weight.h"
#include "ops/sanitizer_scope.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <exception>
#include <iostream>
#include <iterator>
#include <numeric>
#include <random>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

using namespace ninfer;
using namespace ninfer::test;

namespace {

using ops::QsaKvFormat;
using ops::QsaSelectBudget;

constexpr int kHeadDim    = ops::kQsaHeadDim;
constexpr int kKvHeads    = ops::kQsaKvHeads;
constexpr int kQueryHeads = ops::kQsaQueryHeads;
constexpr int kIndexDim   = ops::kQsaIndexHeadDim;
constexpr int kIndexHeads = ops::kQsaIndexQueryHeads;
constexpr int kBlock      = ops::kQsaBlockTokens;
constexpr int kPairs      = 32;

// The attention output is an FP32-accumulated softmax-weighted sum whose weights carry at most
// 2^-16 relative error (each a sum of two BF16 values), rounded once to BF16 (relative <= 2^-8);
// the absolute term covers FP32 accumulation over at most 8195 slots of values of magnitude <= 3.
constexpr PointwiseCriterion attention_criterion() {
    return {/*absolute*/ 1.0e-4, /*relative*/ 4.0e-3};
}

// Kernel block scores are FP32 dots of FP32-normalized, FP32-rotated vectors (angle reduced in
// FP64): about 1e-6 of a score of order 10. A block may differ from the oracle's choice only when
// its oracle score lies this close to the oracle's budget-th score.
constexpr double kScoreTieWindow = 1.0e-4;

std::vector<std::uint16_t> encode_bf16(const std::vector<float>& values) {
    std::vector<std::uint16_t> bits(values.size());
    for (std::size_t i = 0; i < values.size(); ++i) { bits[i] = f32_to_bf16(values[i]); }
    return bits;
}

std::vector<float> random_bf16(std::size_t count, std::uint32_t seed, float lo, float hi) {
    std::vector<float> values(count);
    fill_uniform(values, seed, lo, hi);
    round_to_bf16(values);
    return values;
}

struct Upload {
    DeviceBuffer buffer;

    template <typename T>
    explicit Upload(const std::vector<T>& host) : buffer(to_device(host)) {}

    void* data() const { return buffer.p; }
};

constexpr int kPageTokens = ops::kQsaPageTokens;
constexpr int kPageBlocks = ops::kQsaPageBlocks;

int round_to_pages(int tokens) { return (tokens + kPageTokens - 1) / kPageTokens * kPageTokens; }

// Device planes of a paged QSA state of `lanes` block-table rows of capacity C (a multiple of 64)
// each, plus kSparePages pages no lane maps. Physical pages are a seeded permutation, so every
// lane's logical pages are scattered and interleaved with other lanes'. Every byte starts as a
// sentinel so tests see which bytes an Op wrote.
struct State {
    static constexpr std::uint8_t kSentinel = 0xcd;
    static constexpr int kSparePages        = 3;

    // One plane's page geometry: element bytes, lead elements, and heads per page slot.
    struct PlaneGeometry {
        std::size_t element_bytes;
        int lead;
        int heads;
    };

    State(QsaKvFormat format_in, int capacity_in, int lanes_in = 1, std::uint32_t seed = 7U)
        : format(format_in), capacity(capacity_in), lanes(lanes_in),
          logical_pages(capacity_in / kPageTokens), pages(lanes_in * logical_pages + kSparePages),
          k(plane_bytes(kv_geometry(format_in))), v(plane_bytes(kv_geometry(format_in))),
          k_scales(format_in == QsaKvFormat::BF16 ? 0 : plane_bytes(scale_geometry())),
          v_scales(format_in == QsaKvFormat::BF16 ? 0 : plane_bytes(scale_geometry())),
          index_keys(plane_bytes(index_geometry())), positions(plane_bytes(position_geometry())),
          block_keys(static_cast<std::size_t>(pages) * kPageBlocks * kIndexDim * sizeof(float)),
          table(static_cast<std::size_t>(logical_pages) * lanes_in) {
        if (capacity_in % kPageTokens != 0) {
            throw std::invalid_argument("test state capacity must be whole pages");
        }
        for (auto* plane : {&k, &v, &k_scales, &v_scales, &index_keys, &positions, &block_keys}) {
            plane->fill(kSentinel);
        }
        std::vector<std::int32_t> order(static_cast<std::size_t>(pages));
        std::iota(order.begin(), order.end(), 0);
        std::mt19937 generator(seed);
        std::shuffle(order.begin(), order.end(), generator);
        // Lane r's logical page p is order[r * L + p]; the last kSparePages entries stay unmapped.
        for (int lane = 0; lane < lanes; ++lane) {
            for (int page = 0; page < logical_pages; ++page) {
                table[static_cast<std::size_t>(lane) * logical_pages + page] =
                    order[static_cast<std::size_t>(lane) * logical_pages + page];
            }
        }
        spare.assign(order.end() - kSparePages, order.end());
        device_table = to_device(table);
    }

    PlaneGeometry kv_geometry(QsaKvFormat f) const {
        return f == QsaKvFormat::BF16 ? PlaneGeometry{2, kHeadDim, kKvHeads}
                                      : PlaneGeometry{1, kHeadDim / 2, kKvHeads};
    }

    static PlaneGeometry scale_geometry() { return {1, kHeadDim / 16, kKvHeads}; }

    static PlaneGeometry index_geometry() { return {2, kIndexDim, 1}; }

    static PlaneGeometry position_geometry() { return {4, 1, 3}; }

    std::size_t plane_bytes(const PlaneGeometry& g) const {
        return g.element_bytes * static_cast<std::size_t>(g.lead) * kPageTokens * g.heads * pages;
    }

    int page_of(int lane, int slot) const {
        return table[static_cast<std::size_t>(lane) * logical_pages + slot / kPageTokens];
    }

    // A plane of `lane` gathered into the single-request layout [lead, C, heads] (head-major rows:
    // row = head * C + slot), bytes.
    std::vector<std::uint8_t> logical(const GuardedDeviceBuffer& plane, const PlaneGeometry& g,
                                      int lane) const {
        const auto bytes            = from_device<std::uint8_t>(plane.data(), plane.bytes());
        const std::size_t row_bytes = g.element_bytes * static_cast<std::size_t>(g.lead);
        std::vector<std::uint8_t> out(row_bytes * capacity * g.heads);
        for (int head = 0; head < g.heads; ++head) {
            for (int slot = 0; slot < capacity; ++slot) {
                const std::size_t physical =
                    (static_cast<std::size_t>(page_of(lane, slot)) * g.heads + head) * kPageTokens +
                    slot % kPageTokens;
                std::copy_n(bytes.begin() + static_cast<std::ptrdiff_t>(physical * row_bytes),
                            row_bytes,
                            out.begin() + static_cast<std::ptrdiff_t>(
                                              (static_cast<std::size_t>(head) * capacity + slot) *
                                              row_bytes));
            }
        }
        return out;
    }

    // The lane's sealed block keys in logical order [C/4][128] FP32, bytes.
    std::vector<std::uint8_t> logical_block_keys(int lane) const {
        const auto bytes = from_device<std::uint8_t>(block_keys.data(), block_keys.bytes());
        constexpr std::size_t kKeyBytes = kIndexDim * sizeof(float);
        std::vector<std::uint8_t> out(kKeyBytes * (capacity / kBlock));
        for (int block = 0; block < capacity / kBlock; ++block) {
            const std::size_t physical =
                static_cast<std::size_t>(page_of(lane, block * kBlock)) * kPageBlocks +
                block % kPageBlocks;
            std::copy_n(bytes.begin() + static_cast<std::ptrdiff_t>(physical * kKeyBytes),
                        kKeyBytes, out.begin() + static_cast<std::ptrdiff_t>(block * kKeyBytes));
        }
        return out;
    }

    ops::QsaPagedState view() {
        ops::QsaPagedState state;
        state.format = format;
        if (format == QsaKvFormat::BF16) {
            state.k = Tensor(k.data(), DType::BF16, {kHeadDim, kPageTokens, kKvHeads, pages});
            state.v = Tensor(v.data(), DType::BF16, {kHeadDim, kPageTokens, kKvHeads, pages});
        } else {
            state.k = Tensor(k.data(), DType::U8, {kHeadDim / 2, kPageTokens, kKvHeads, pages});
            state.v = Tensor(v.data(), DType::U8, {kHeadDim / 2, kPageTokens, kKvHeads, pages});
            state.k_scales = Tensor(k_scales.data(), DType::FP8_E4M3FN,
                                    {kHeadDim / 16, kPageTokens, kKvHeads, pages});
            state.v_scales = Tensor(v_scales.data(), DType::FP8_E4M3FN,
                                    {kHeadDim / 16, kPageTokens, kKvHeads, pages});
        }
        state.index_keys =
            Tensor(index_keys.data(), DType::BF16, {kIndexDim, kPageTokens, 1, pages});
        state.positions    = Tensor(positions.data(), DType::I32, {1, kPageTokens, 3, pages});
        state.block_keys   = Tensor(block_keys.data(), DType::FP32,
                                    {kPageBlocks * kIndexDim / kPageTokens, kPageTokens, 1, pages});
        state.block_tables = Tensor(device_table.p, DType::I32, {logical_pages, lanes});
        return state;
    }

    // Guards, and every byte of the spare pages (which no lane maps) is still the sentinel.
    int verify_guards(const std::string& label) const {
        int failures = 0;
        for (const auto* plane :
             {&k, &v, &k_scales, &v_scales, &index_keys, &positions, &block_keys}) {
            failures += plane->verify_guards(label + " state");
            if (plane->bytes() == 0) { continue; }
            const auto bytes           = from_device<std::uint8_t>(plane->data(), plane->bytes());
            const std::size_t per_page = plane->bytes() / static_cast<std::size_t>(pages);
            for (const int page : spare) {
                const auto begin = bytes.begin() + static_cast<std::ptrdiff_t>(
                                                       per_page * static_cast<std::size_t>(page));
                if (!std::all_of(begin, begin + static_cast<std::ptrdiff_t>(per_page),
                                 [](std::uint8_t b) { return b == kSentinel; })) {
                    std::cerr << label << ": an unmapped page was written\n";
                    ++failures;
                    return failures;
                }
            }
        }
        return failures;
    }

    QsaKvFormat format;
    int capacity;
    int lanes;
    int logical_pages;
    int pages;
    GuardedDeviceBuffer k, v, k_scales, v_scales, index_keys, positions, block_keys;
    std::vector<std::int32_t> table; // [lanes][logical_pages]
    std::vector<std::int32_t> spare;
    DeviceBuffer device_table;
};

// One lane-batched qsa_state_append of B rows of W columns: k/v [B][W][2][256], keys [B][W][128],
// positions [B][3][W] (axis-major per row), slots [B][W], lanes [B].
void append_rows(State& state, const std::vector<std::uint16_t>& k,
                 const std::vector<std::uint16_t>& v, const std::vector<std::uint16_t>& keys,
                 const std::vector<std::int32_t>& positions, const std::vector<std::int32_t>& slots,
                 const std::vector<std::int32_t>& lanes, const Tensor& key_norm) {
    const int rows  = static_cast<int>(lanes.size());
    const int width = static_cast<int>(slots.size()) / rows;
    const Upload dk(k), dv(v), dkeys(keys), dpositions(positions), dslots(slots), dlanes(lanes);
    const Tensor k_t(dk.data(), DType::BF16, {kHeadDim, kKvHeads, width, rows});
    const Tensor v_t(dv.data(), DType::BF16, {kHeadDim, kKvHeads, width, rows});
    const Tensor keys_t(dkeys.data(), DType::BF16, {kIndexDim, width, rows});
    const Tensor positions_t(dpositions.data(), DType::I32, {width, 3, rows});
    const Tensor slots_t(dslots.data(), DType::I32, {width, rows});
    const Tensor lanes_t(dlanes.data(), DType::I32, {rows});
    auto view = state.view();
    ops::qsa_state_append(k_t, v_t, keys_t, positions_t, slots_t, lanes_t, key_norm, view, nullptr);
    cuda_synchronize();
}

// Positions axis-major [3][C]. Text tokens repeat the slot on every axis; the distinct-axis variant
// gives the three axes different values so the pair-to-axis interleave is observable.
std::vector<std::int32_t> make_positions(int capacity, bool distinct_axes) {
    std::vector<std::int32_t> positions(static_cast<std::size_t>(3) * capacity);
    for (int slot = 0; slot < capacity; ++slot) {
        positions[slot]                = slot;
        positions[capacity + slot]     = distinct_axes ? slot / 3 + 7 * (slot % 5) : slot;
        positions[2 * capacity + slot] = distinct_axes ? 2 * slot + 11 : slot;
    }
    return positions;
}

// --- Indexer key oracle -----------------------------------------------------------------------

const std::array<double, kPairs>& inverse_frequencies() {
    static const std::array<double, kPairs> table = [] {
        std::array<double, kPairs> values{};
        for (int i = 0; i < kPairs; ++i) { values[i] = std::pow(1.0e7, -2.0 * i / 64.0); }
        return values;
    }();
    return table;
}

// norm(x, w) = x / sqrt(mean(x^2) + 1e-6) * (1 + w), then rotate-half pairs (i, i+32) at the
// three-axis position of `slot` (positions axis-major [3][capacity]), pair i on axis i % 3.
std::array<double, kIndexDim> oracle_norm_rope(const double* x, const std::vector<float>& weight,
                                               const std::vector<std::int32_t>& positions,
                                               int capacity, int slot) {
    double sum = 0.0;
    for (int d = 0; d < kIndexDim; ++d) { sum += x[d] * x[d]; }
    const double inverse = 1.0 / std::sqrt(sum / kIndexDim + 1.0e-6);
    std::array<double, kIndexDim> y{};
    for (int d = 0; d < kIndexDim; ++d) { y[d] = x[d] * inverse * (1.0 + double(weight[d])); }
    for (int pair = 0; pair < kPairs; ++pair) {
        const double angle =
            double(positions[static_cast<std::size_t>(pair % 3) * capacity + slot]) *
            inverse_frequencies()[pair];
        const double c = std::cos(angle), s = std::sin(angle);
        const double first = y[pair], second = y[pair + kPairs];
        y[pair]          = first * c - second * s;
        y[pair + kPairs] = second * c + first * s;
    }
    return y;
}

// key[b] from represented BF16 raw keys [C][128] and positions [3][C].
std::array<double, kIndexDim> oracle_block_key(const std::vector<float>& keys,
                                               const std::vector<float>& key_norm,
                                               const std::vector<std::int32_t>& positions,
                                               int capacity, int block) {
    std::array<double, kIndexDim> pooled{};
    for (int d = 0; d < kIndexDim; ++d) {
        // The semantic pooling boundary: the FP32 slot-order sum times 0.25, rounded to BF16.
        float sum = 0.0F;
        for (int r = 0; r < kBlock; ++r) {
            sum += keys[(static_cast<std::size_t>(kBlock) * block + r) * kIndexDim + d];
        }
        pooled[d] = bf16_to_f32(f32_to_bf16(0.25F * sum));
    }
    return oracle_norm_rope(pooled.data(), key_norm, positions, capacity, kBlock * block);
}

// The sealed key is an FP32 norm and rotation of values of magnitude at most a few units: FP32
// sum-of-squares and rsqrt rounding (about 1e-7 relative) and sincosf of the FP64-reduced angle
// (about 1e-7 absolute) stay two orders of magnitude inside this bound.
constexpr PointwiseCriterion block_key_criterion() {
    return {/*absolute*/ 1.0e-5, /*relative*/ 1.0e-5};
}

// --- qsa_state_append --------------------------------------------------------------------------

int append_case(QsaKvFormat format) {
    const std::string label =
        std::string("qsa_state_append ") + (format == QsaKvFormat::BF16 ? "BF16" : "NVFP4");
    // Two pages of lane 1 of a two-lane state; lane 0 and the spare pages must stay untouched.
    constexpr int kCapacity = 128;
    constexpr int kLane     = 1;
    // Call 0 writes block 0 out of order inside one call (slot 3 by an earlier token than slots
    // 0..2), the first three slots of blocks 15 (the last block of page 0) and 16 (the first of
    // page 1) and scattered slots; call 1 completes blocks 15 and 16 across calls and the page
    // edge, and rewrites slots 2 and 3, so block 0 is sealed again from its new raw keys. Blocks
    // 1, 4, 10 and 31 stay partial and keep their sentinel keys. Each call's sealed keys are
    // checked right after it.
    const std::array<std::vector<std::int32_t>, 2> calls{
        std::vector<std::int32_t>{5, -1, 40, 62, 3, 1, 0, 17, 60, 2, 61, 64, 65, 66},
        std::vector<std::int32_t>{63, -1, 41, 2, 3, 67, 125}};
    const std::vector<float> key_norm = random_bf16(kIndexDim, 43U, -0.5F, 0.5F);
    const Upload dkey_norm(encode_bf16(key_norm));
    const Tensor key_norm_t(dkey_norm.data(), DType::BF16, {kIndexDim});

    State state(format, kCapacity, 2, 47U);
    auto view                     = state.view();
    const State::PlaneGeometry kv = state.kv_geometry(format);
    const auto logical_bytes      = [&](const State::PlaneGeometry& g) {
        return g.element_bytes * static_cast<std::size_t>(g.lead) * kCapacity * g.heads;
    };
    // Expected logical planes of lane 1: sentinel everywhere except the appended slots and sealed
    // blocks.
    std::vector<std::uint8_t> expected_k(logical_bytes(kv), State::kSentinel);
    std::vector<std::uint8_t> expected_v(logical_bytes(kv), State::kSentinel);
    const std::size_t scale_bytes =
        format == QsaKvFormat::BF16 ? 0 : logical_bytes(State::scale_geometry());
    std::vector<std::uint8_t> expected_k_scales(scale_bytes, State::kSentinel);
    std::vector<std::uint8_t> expected_v_scales(scale_bytes, State::kSentinel);
    std::vector<std::uint8_t> expected_keys(logical_bytes(State::index_geometry()),
                                            State::kSentinel);
    std::vector<std::uint8_t> expected_positions(logical_bytes(State::position_geometry()),
                                                 State::kSentinel);
    // Represented raw keys and positions of the appended slots, for the sealed-key oracle.
    std::vector<float> state_keys(static_cast<std::size_t>(kCapacity) * kIndexDim, 0.0F);
    std::vector<std::int32_t> state_positions(static_cast<std::size_t>(3) * kCapacity, 0);
    std::vector<bool> sealed(kCapacity / kBlock, false);

    int failures = 0;
    for (std::size_t call = 0; call < calls.size(); ++call) {
        const std::vector<std::int32_t>& slots = calls[call];
        const int width                        = static_cast<int>(slots.size());
        std::vector<int> sealed_now;
        const auto seed = static_cast<std::uint32_t>(31 + 100 * call);
        std::vector<float> k =
            random_bf16(static_cast<std::size_t>(kHeadDim) * kKvHeads * width, seed, -2.0F, 2.0F);
        std::vector<float> v = random_bf16(k.size(), seed + 6, -3.0F, 3.0F);
        if (call == 0) {
            // A zero group encodes scale 0 and zero codes; a group past 6 * 448 saturates scale and
            // codes; a tiny group exercises subnormal E4M3 scales.
            std::fill_n(k.begin() + 16, 16, 0.0F);
            for (int i = 0; i < 16; ++i) {
                v[kHeadDim + 32 + i] = (i % 2 == 0 ? 1.0F : -1.0F) * (3000.0F + 64.0F * i);
            }
            for (int i = 0; i < 16; ++i) {
                v[2 * kHeadDim * 2 + i] = 1.0e-2F * static_cast<float>(i - 7);
            }
            round_to_bf16(k);
            round_to_bf16(v);
        }
        const std::vector<float> keys =
            random_bf16(static_cast<std::size_t>(kIndexDim) * width, seed + 10, -1.0F, 1.0F);
        std::vector<std::int32_t> positions(static_cast<std::size_t>(3) * width);
        for (std::size_t i = 0; i < positions.size(); ++i) {
            positions[i] = static_cast<std::int32_t>(1000 + 500 * call + 17 * i);
        }

        append_rows(state, encode_bf16(k), encode_bf16(v), encode_bf16(keys), positions, slots,
                    {kLane}, key_norm_t);

        const auto key_bits = encode_bf16(keys);
        for (int token = 0; token < width; ++token) {
            const int slot = slots[static_cast<std::size_t>(token)];
            if (slot < 0) { continue; }
            for (int head = 0; head < kKvHeads; ++head) {
                const std::size_t source =
                    (static_cast<std::size_t>(token) * kKvHeads + head) * kHeadDim;
                const std::size_t row = static_cast<std::size_t>(head) * kCapacity + slot;
                for (int tensor = 0; tensor < 2; ++tensor) {
                    const std::vector<float>& values = tensor == 0 ? k : v;
                    auto& codes                      = tensor == 0 ? expected_k : expected_v;
                    if (format == QsaKvFormat::BF16) {
                        for (int d = 0; d < kHeadDim; ++d) {
                            const std::uint16_t bits = f32_to_bf16(values[source + d]);
                            codes[2 * (row * kHeadDim + d)] =
                                static_cast<std::uint8_t>(bits & 0xffU);
                            codes[2 * (row * kHeadDim + d) + 1] =
                                static_cast<std::uint8_t>(bits >> 8);
                        }
                        continue;
                    }
                    const auto reference = nvfp4_activation_reference(
                        std::span<const float>(values.data() + source, kHeadDim), 1.0F);
                    std::copy(reference.codes.begin(), reference.codes.end(),
                              codes.begin() + static_cast<std::ptrdiff_t>(row * (kHeadDim / 2)));
                    auto& scales = tensor == 0 ? expected_k_scales : expected_v_scales;
                    std::copy(reference.scales.begin(), reference.scales.end(),
                              scales.begin() + static_cast<std::ptrdiff_t>(row * (kHeadDim / 16)));
                }
            }
            for (int d = 0; d < kIndexDim; ++d) {
                const std::size_t source      = static_cast<std::size_t>(token) * kIndexDim + d;
                const std::size_t target      = static_cast<std::size_t>(slot) * kIndexDim + d;
                expected_keys[2 * target]     = static_cast<std::uint8_t>(key_bits[source] & 0xffU);
                expected_keys[2 * target + 1] = static_cast<std::uint8_t>(key_bits[source] >> 8);
                state_keys[target]            = keys[source];
            }
            for (int axis = 0; axis < 3; ++axis) {
                const std::int32_t position =
                    positions[static_cast<std::size_t>(axis) * width + token];
                std::memcpy(
                    &expected_positions[(static_cast<std::size_t>(axis) * kCapacity + slot) * 4],
                    &position, 4);
                state_positions[static_cast<std::size_t>(axis) * kCapacity + slot] = position;
            }
            if (slot % kBlock == kBlock - 1) { sealed_now.push_back(slot / kBlock); }
        }

        // Blocks sealed by this call hold the oracle key of the raw keys the state holds now.
        const auto block_bytes = state.logical_block_keys(kLane);
        std::vector<double> actual_keys;
        std::vector<double> expected_block_keys;
        for (const int block : sealed_now) {
            sealed[static_cast<std::size_t>(block)] = true;
            const auto key =
                oracle_block_key(state_keys, key_norm, state_positions, kCapacity, block);
            for (int d = 0; d < kIndexDim; ++d) {
                float value = 0.0F;
                std::memcpy(
                    &value,
                    &block_bytes[(static_cast<std::size_t>(block) * kIndexDim + d) * sizeof(float)],
                    sizeof(float));
                actual_keys.push_back(value);
                expected_block_keys.push_back(key[static_cast<std::size_t>(d)]);
            }
        }
        failures += verify_pointwise(label + " block_keys call " + std::to_string(call),
                                     actual_keys, expected_block_keys, block_key_criterion());
    }

    failures += verify_exact((label + " k").c_str(), state.logical(state.k, kv, kLane), expected_k);
    failures += verify_exact((label + " v").c_str(), state.logical(state.v, kv, kLane), expected_v);
    if (format == QsaKvFormat::NVFP4G16) {
        failures += verify_exact((label + " k_scales").c_str(),
                                 state.logical(state.k_scales, State::scale_geometry(), kLane),
                                 expected_k_scales);
        failures += verify_exact((label + " v_scales").c_str(),
                                 state.logical(state.v_scales, State::scale_geometry(), kLane),
                                 expected_v_scales);
    }
    failures += verify_exact((label + " index_keys").c_str(),
                             state.logical(state.index_keys, State::index_geometry(), kLane),
                             expected_keys);
    failures += verify_exact((label + " positions").c_str(),
                             state.logical(state.positions, State::position_geometry(), kLane),
                             expected_positions);
    // Lane 0's pages are untouched.
    for (const auto& [plane, geometry] :
         std::array<std::pair<const GuardedDeviceBuffer*, State::PlaneGeometry>, 3>{
             {{&state.k, kv},
              {&state.index_keys, State::index_geometry()},
              {&state.positions, State::position_geometry()}}}) {
        const auto other = state.logical(*plane, geometry, 0);
        if (!std::all_of(other.begin(), other.end(),
                         [](std::uint8_t b) { return b == State::kSentinel; })) {
            std::cerr << label << ": another lane's pages were written\n";
            ++failures;
        }
    }

    // Every block no call sealed keeps its sentinel key.
    const auto block_bytes = state.logical_block_keys(kLane);
    for (int block = 0; block < kCapacity / kBlock; ++block) {
        if (sealed[static_cast<std::size_t>(block)]) { continue; }
        const auto begin =
            block_bytes.begin() + static_cast<std::ptrdiff_t>(static_cast<std::size_t>(block) *
                                                              kIndexDim * sizeof(float));
        if (!std::all_of(begin, begin + static_cast<std::ptrdiff_t>(kIndexDim * sizeof(float)),
                         [](std::uint8_t byte) { return byte == State::kSentinel; })) {
            std::cerr << label << ": unsealed block " << block << " key was written\n";
            ++failures;
        }
    }
    failures += state.verify_guards(label);

    // A rejected call queues nothing: BF16 state must not carry scale planes.
    if (format == QsaKvFormat::BF16) {
        const std::vector<std::int32_t> slots{7};
        const std::vector<std::uint16_t> zero_kv(kHeadDim * kKvHeads, 0);
        const std::vector<std::uint16_t> zero_key(kIndexDim, 0);
        const std::vector<std::int32_t> zero_position(3, 0);
        const std::vector<std::int32_t> lanes{kLane};
        const Upload dk(zero_kv), dkeys(zero_key), dpositions(zero_position), dslots(slots),
            dlanes(lanes);
        Tensor k_t(dk.data(), DType::BF16, {kHeadDim, kKvHeads, 1, 1});
        Tensor keys_t(dkeys.data(), DType::BF16, {kIndexDim, 1, 1});
        Tensor positions_t(dpositions.data(), DType::I32, {1, 3, 1});
        Tensor slots_t(dslots.data(), DType::I32, {1, 1});
        Tensor lanes_t(dlanes.data(), DType::I32, {1});
        auto bad     = view;
        bad.k_scales = Tensor(state.k.data(), DType::FP8_E4M3FN,
                              {kHeadDim / 16, kPageTokens, kKvHeads, state.pages});
        failures += expect_invalid_argument(
            [&] {
                ops::qsa_state_append(k_t, k_t, keys_t, positions_t, slots_t, lanes_t, key_norm_t,
                                      bad, nullptr);
            },
            label + ": accepted BF16 state with a scale plane");
    }
    return failures;
}

// --- qsa_index_select oracle ------------------------------------------------------------------

struct IndexInputs {
    int capacity = 0;
    std::vector<float> keys;             // [C][128] represented BF16 raw keys
    std::vector<std::int32_t> positions; // [3][C]
    std::vector<float> query;            // [W][4][128] represented BF16 raw queries
    std::vector<float> query_norm;       // [128] zero-centered
    std::vector<float> key_norm;         // [128] zero-centered
    std::vector<std::int32_t> slots;     // [W]
};

std::vector<std::array<double, kIndexDim>> oracle_block_keys(const IndexInputs& in, int blocks) {
    std::vector<std::array<double, kIndexDim>> keys(static_cast<std::size_t>(blocks));
    for (int b = 0; b < blocks; ++b) {
        keys[static_cast<std::size_t>(b)] =
            oracle_block_key(in.keys, in.key_norm, in.positions, in.capacity, b);
    }
    return keys;
}

struct SelectResult {
    std::vector<std::int32_t> ids;
    std::vector<std::int32_t> count;
};

// The lane every single-lane case uses: row 1 of a two-row table, so row 0's pages interleave.
constexpr int kTestLane = 1;

// Appends every slot's raw key and position of `in` into `lane` through qsa_state_append (which
// seals every block) in calls of at most 32768 tokens. K and V are zero: the selection does not
// read them.
void fill_index_state(State& state, const IndexInputs& in, int lane = kTestLane) {
    constexpr int kChunk = 32768;
    const int chunk      = std::min(kChunk, in.capacity);
    const auto chunk_c   = static_cast<std::size_t>(chunk);
    const Upload dknorm(encode_bf16(in.key_norm));
    const Tensor knorm_t(dknorm.data(), DType::BF16, {kIndexDim});
    for (int begin = 0; begin < in.capacity; begin += chunk) {
        const int width = std::min(chunk, in.capacity - begin);
        const auto keys = encode_bf16(std::vector<float>(
            in.keys.begin() + static_cast<std::ptrdiff_t>(begin) * kIndexDim,
            in.keys.begin() + static_cast<std::ptrdiff_t>(begin + width) * kIndexDim));
        std::vector<std::int32_t> positions(static_cast<std::size_t>(3) * width);
        std::vector<std::int32_t> slots(static_cast<std::size_t>(width));
        for (int t = 0; t < width; ++t) {
            for (int axis = 0; axis < 3; ++axis) {
                positions[static_cast<std::size_t>(axis) * width + t] =
                    in.positions[static_cast<std::size_t>(axis) * in.capacity + begin + t];
            }
            slots[static_cast<std::size_t>(t)] = begin + t;
        }
        const std::vector<std::uint16_t> zero(static_cast<std::size_t>(width) * kKvHeads * kHeadDim,
                                              0);
        append_rows(state, zero, zero, keys, positions, slots, {lane}, knorm_t);
    }
    (void)chunk_c;
}

// One lane-batched selection: queries [B][W][4][128], slots [B][W], lanes [B].
SelectResult run_select_rows(State& state, const std::vector<float>& query,
                             const std::vector<std::int32_t>& slots,
                             const std::vector<std::int32_t>& lanes,
                             const std::vector<float>& query_norm, QsaSelectBudget budget,
                             int max_visible) {
    const int rows           = static_cast<int>(lanes.size());
    const int width          = static_cast<int>(slots.size()) / rows;
    const int queries        = width * rows;
    const int selected_slots = ops::qsa_selected_capacity(budget);
    const Upload dquery(encode_bf16(query)), dslots(slots), dlanes(lanes),
        dqnorm(encode_bf16(query_norm));
    GuardedDeviceBuffer selected(static_cast<std::size_t>(selected_slots) * queries *
                                 sizeof(std::int32_t));
    GuardedDeviceBuffer count(static_cast<std::size_t>(queries) * sizeof(std::int32_t));
    Tensor query_t(dquery.data(), DType::BF16, {kIndexDim, kIndexHeads, width, rows});
    Tensor slots_t(dslots.data(), DType::I32, {width, rows});
    Tensor lanes_t(dlanes.data(), DType::I32, {rows});
    Tensor qnorm_t(dqnorm.data(), DType::BF16, {kIndexDim});
    Tensor selected_t(selected.data(), DType::I32, {selected_slots, width, rows});
    Tensor count_t(count.data(), DType::I32, {width, rows});
    WorkspaceArena workspace(std::max<std::size_t>(
        ops::qsa_index_select_workspace_bytes(queries, max_visible, budget), 256));
    ops::qsa_index_select(query_t, state.view(), slots_t, lanes_t, qnorm_t, budget, max_visible,
                          selected_t, count_t, workspace, nullptr);
    cuda_synchronize();
    if (selected.verify_guards("qsa_index_select selected_ids") +
            count.verify_guards("qsa_index_select selected_count") !=
        0) {
        throw std::runtime_error("qsa_index_select wrote outside its outputs");
    }
    return {from_device<std::int32_t>(selected.data(),
                                      static_cast<std::size_t>(selected_slots) * queries),
            from_device<std::int32_t>(count.data(), static_cast<std::size_t>(queries))};
}

SelectResult run_select(State& state, const IndexInputs& in, QsaSelectBudget budget,
                        int max_visible) {
    return run_select_rows(state, in.query, in.slots, {kTestLane}, in.query_norm, budget,
                           max_visible);
}

// Compares the Op's selection against the FP64 oracle. With exact_ties every selected block must
// match; otherwise a block may differ only within kScoreTieWindow of the budget-th score.
int verify_select(const std::string& label, const IndexInputs& in, QsaSelectBudget budget,
                  int max_visible, const SelectResult& actual, bool exact_ties) {
    const int width        = static_cast<int>(in.slots.size());
    const int budget_count = static_cast<int>(budget);
    const int slots        = ops::qsa_selected_capacity(budget);
    const auto block_keys  = oracle_block_keys(in, max_visible / kBlock);
    int failures           = 0;
    int window_swaps       = 0;
    for (int t = 0; t < width; ++t) {
        const std::string column = label + " query " + std::to_string(t);
        const std::int32_t* ids  = actual.ids.data() + static_cast<std::size_t>(t) * slots;
        const int slot           = in.slots[static_cast<std::size_t>(t)];
        std::vector<std::int32_t> expected(static_cast<std::size_t>(slots), -1);
        if (slot < 0 || slot >= max_visible) {
            failures += verify_exact((column + " empty").c_str(),
                                     std::vector<std::int32_t>(ids, ids + slots), expected);
            if (actual.count[t] != 0) {
                std::cerr << column << ": count " << actual.count[t] << " != 0\n";
                ++failures;
            }
            continue;
        }
        const int visible  = slot + 1;
        const int complete = visible / kBlock;
        if (complete <= budget_count) {
            std::iota(expected.begin(), expected.begin() + visible, 0);
            failures += verify_exact((column + " prefix").c_str(),
                                     std::vector<std::int32_t>(ids, ids + slots), expected);
            if (actual.count[t] != visible) {
                std::cerr << column << ": count " << actual.count[t] << " != " << visible << '\n';
                ++failures;
            }
            continue;
        }

        std::array<std::array<double, kIndexDim>, kIndexHeads> query{};
        for (int h = 0; h < kIndexHeads; ++h) {
            std::array<double, kIndexDim> raw{};
            for (int d = 0; d < kIndexDim; ++d) {
                raw[d] = in.query[(static_cast<std::size_t>(t) * kIndexHeads + h) * kIndexDim + d];
            }
            query[h] = oracle_norm_rope(raw.data(), in.query_norm, in.positions, in.capacity, slot);
        }
        std::vector<double> score(static_cast<std::size_t>(complete));
        for (int b = 0; b < complete; ++b) {
            double total = 0.0;
            for (int h = 0; h < kIndexHeads; ++h) {
                double dot = 0.0;
                for (int d = 0; d < kIndexDim; ++d) {
                    dot += query[h][d] * block_keys[static_cast<std::size_t>(b)][d];
                }
                total += std::max(dot, 0.0);
            }
            score[static_cast<std::size_t>(b)] = total / std::sqrt(double(kIndexDim));
        }
        std::vector<int> order(static_cast<std::size_t>(complete));
        std::iota(order.begin(), order.end(), 0);
        std::stable_sort(order.begin(), order.end(),
                         [&](int a, int b) { return score[a] > score[b]; });
        std::vector<int> chosen(order.begin(), order.begin() + budget_count);
        std::sort(chosen.begin(), chosen.end());
        const double threshold =
            score[static_cast<std::size_t>(order[static_cast<std::size_t>(budget_count - 1)])];

        const int tail_begin     = kBlock * complete;
        const int expected_count = kBlock * budget_count + visible - tail_begin;
        if (actual.count[t] != expected_count) {
            std::cerr << column << ": count " << actual.count[t] << " != " << expected_count
                      << '\n';
            ++failures;
            continue;
        }
        // Structure: ascending complete blocks, each listing its four slots, then tail and padding.
        std::vector<int> listed;
        listed.reserve(static_cast<std::size_t>(budget_count));
        bool well_formed = true;
        for (int j = 0; j < budget_count; ++j) {
            const std::ptrdiff_t entry = static_cast<std::ptrdiff_t>(kBlock) * j;
            const int first            = ids[entry];
            well_formed = well_formed && first >= 0 && first % kBlock == 0 && first < tail_begin &&
                          (listed.empty() || first / kBlock > listed.back());
            for (int r = 1; r < kBlock; ++r) {
                well_formed = well_formed && ids[entry + r] == first + r;
            }
            listed.push_back(first / kBlock);
        }
        for (int i = kBlock * budget_count; i < slots; ++i) {
            const int id = tail_begin + i - kBlock * budget_count;
            well_formed  = well_formed && ids[i] == (id < visible ? id : -1);
        }
        if (!well_formed) {
            std::cerr << column << ": malformed selected list\n";
            ++failures;
            continue;
        }
        std::vector<int> difference;
        std::set_symmetric_difference(listed.begin(), listed.end(), chosen.begin(), chosen.end(),
                                      std::back_inserter(difference));
        for (const int block : difference) {
            const double gap = std::abs(score[static_cast<std::size_t>(block)] - threshold);
            if (exact_ties || gap > kScoreTieWindow * std::max(1.0, threshold)) {
                std::cerr << column << ": block " << block
                          << " differs from the oracle selection (score "
                          << score[static_cast<std::size_t>(block)] << ", threshold " << threshold
                          << ")\n";
                ++failures;
                break;
            }
            ++window_swaps;
        }
    }
    if (window_swaps != 0) {
        std::cout << label << ": " << window_swaps
                  << " blocks differ within the score tie window\n";
    }
    return failures;
}

// Random raw keys and queries, distinct-axis or text positions; queries at the given slots.
int select_random_case(QsaSelectBudget budget, int capacity, int max_visible,
                       const std::vector<std::int32_t>& slots, bool distinct_axes,
                       std::uint32_t seed) {
    const std::string label =
        "qsa_index_select budget=" + std::to_string(static_cast<int>(budget)) +
        " C=" + std::to_string(capacity) + (distinct_axes ? " mrope" : " text");
    IndexInputs in;
    in.capacity   = capacity;
    in.keys       = random_bf16(static_cast<std::size_t>(capacity) * kIndexDim, seed, -1.0F, 1.0F);
    in.positions  = make_positions(capacity, distinct_axes);
    in.query      = random_bf16(slots.size() * kIndexHeads * kIndexDim, seed + 1, -1.0F, 1.0F);
    in.query_norm = random_bf16(kIndexDim, seed + 2, -0.5F, 0.5F);
    in.key_norm   = random_bf16(kIndexDim, seed + 3, -0.5F, 0.5F);
    in.slots      = slots;

    State state(QsaKvFormat::NVFP4G16, capacity, 2, seed + 9);
    fill_index_state(state, in);
    const SelectResult actual = run_select(state, in, budget, max_visible);
    return verify_select(label, in, budget, max_visible, actual, /*exact_ties*/ false) +
           state.verify_guards(label);
}

// Every block's four raw keys repeat one of six patterns whose first 64 features are zero, so the
// rotary embedding leaves every pooled key unchanged and blocks sharing a pattern tie exactly in
// both the FP64 oracle and the kernel (pattern 0 is all zero and scores exactly zero). The
// budget boundary therefore falls inside a tie group: selection must take the lowest blocks.
int select_tie_case(QsaSelectBudget budget, const std::vector<std::int32_t>& slots,
                    std::uint32_t seed) {
    const std::string label =
        "qsa_index_select ties budget=" + std::to_string(static_cast<int>(budget));
    constexpr int kPatterns = 6;
    // The smallest capacity holding every query slot (a whole number of pages).
    const int capacity = round_to_pages(*std::max_element(slots.begin(), slots.end()) + 1);
    IndexInputs in;
    in.capacity  = capacity;
    in.positions = make_positions(capacity, true);
    std::vector<float> patterns =
        random_bf16(static_cast<std::size_t>(kPatterns) * kIndexDim, seed, -1.0F, 1.0F);
    for (int p = 0; p < kPatterns; ++p) {
        for (int d = 0; d < kIndexDim; ++d) {
            if (p == 0 || d < 2 * kPairs) {
                patterns[static_cast<std::size_t>(p) * kIndexDim + d] = 0.0F;
            }
        }
    }
    in.keys.resize(static_cast<std::size_t>(capacity) * kIndexDim);
    std::mt19937 generator(seed + 1);
    for (int b = 0; b * kBlock < capacity; ++b) {
        const auto pattern = static_cast<std::size_t>(generator() % kPatterns);
        for (int r = 0; r < kBlock && kBlock * b + r < capacity; ++r) {
            std::copy_n(
                patterns.begin() + static_cast<std::ptrdiff_t>(pattern * kIndexDim), kIndexDim,
                in.keys.begin() + static_cast<std::ptrdiff_t>(
                                      (static_cast<std::size_t>(kBlock) * b + r) * kIndexDim));
        }
    }
    in.query      = random_bf16(slots.size() * kIndexHeads * kIndexDim, seed + 2, -1.0F, 1.0F);
    in.query_norm = random_bf16(kIndexDim, seed + 3, -0.5F, 0.5F);
    in.key_norm   = random_bf16(kIndexDim, seed + 4, -0.5F, 0.5F);
    in.slots      = slots;

    // Distinct pattern scores must be far apart, or kernel rounding could reorder patterns.
    int failures = 0;
    for (std::size_t t = 0; t < slots.size(); ++t) {
        std::array<double, kPatterns> scores{};
        for (int p = 0; p < kPatterns; ++p) {
            std::array<double, kIndexDim> pooled{};
            for (int d = 0; d < kIndexDim; ++d) {
                pooled[d] = patterns[static_cast<std::size_t>(p) * kIndexDim + d];
            }
            const auto key =
                oracle_norm_rope(pooled.data(), in.key_norm, in.positions, in.capacity, 0);
            for (int h = 0; h < kIndexHeads; ++h) {
                std::array<double, kIndexDim> raw{};
                for (int d = 0; d < kIndexDim; ++d) {
                    raw[d] = in.query[(t * kIndexHeads + h) * kIndexDim + d];
                }
                const auto query = oracle_norm_rope(raw.data(), in.query_norm, in.positions,
                                                    in.capacity, slots[t]);
                double dot       = 0.0;
                for (int d = 0; d < kIndexDim; ++d) { dot += query[d] * key[d]; }
                scores[p] += std::max(dot, 0.0);
            }
        }
        for (int a = 0; a < kPatterns; ++a) {
            for (int b = a + 1; b < kPatterns; ++b) {
                if (scores[a] != scores[b] && std::abs(scores[a] - scores[b]) < 1.0e-3) {
                    std::cerr << label << ": test patterns " << a << " and " << b
                              << " score too close\n";
                    ++failures;
                }
            }
        }
    }

    State state(QsaKvFormat::BF16, capacity, 2, seed + 9);
    fill_index_state(state, in);
    const SelectResult actual = run_select(state, in, budget, capacity);
    return failures + verify_select(label, in, budget, capacity, actual, /*exact_ties*/ true);
}

// --- qsa_selected_attention --------------------------------------------------------------------

// One lane-batched attention call: q [B][W][24][256], selected [B][W][S], counts [B][W], lanes
// [B]. Returns the BF16 output bits [B][W][24][256].
std::vector<std::uint16_t> run_attention_rows(State& state, const std::vector<std::uint16_t>& q,
                                              const std::vector<std::int32_t>& selected,
                                              const std::vector<std::int32_t>& counts,
                                              const std::vector<std::int32_t>& lanes,
                                              int selected_slots) {
    const int rows    = static_cast<int>(lanes.size());
    const int width   = static_cast<int>(counts.size()) / rows;
    const int queries = width * rows;
    const Upload dq(q), dselected(selected), dcounts(counts), dlanes(lanes);
    GuardedDeviceBuffer out(static_cast<std::size_t>(kHeadDim) * kQueryHeads * queries *
                            sizeof(std::uint16_t));
    out.fill(0xff);
    Tensor q_t(dq.data(), DType::BF16, {kHeadDim, kQueryHeads, width, rows});
    Tensor selected_t(dselected.data(), DType::I32, {selected_slots, width, rows});
    Tensor counts_t(dcounts.data(), DType::I32, {width, rows});
    Tensor lanes_t(dlanes.data(), DType::I32, {rows});
    Tensor out_t(out.data(), DType::BF16, {kHeadDim, kQueryHeads, width, rows});
    WorkspaceArena workspace(std::max<std::size_t>(
        ops::qsa_selected_attention_workspace_bytes(queries, selected_slots), 256));
    ops::qsa_selected_attention(q_t, selected_t, counts_t, lanes_t, state.view(), out_t, workspace,
                                nullptr);
    cuda_synchronize();
    if (out.verify_guards("qsa_selected_attention out") != 0) {
        throw std::runtime_error("qsa_selected_attention wrote outside its output");
    }
    return from_device<std::uint16_t>(out.data(),
                                      static_cast<std::size_t>(kHeadDim) * kQueryHeads * queries);
}

struct AttentionCase {
    QsaKvFormat format;
    int capacity;
    int selected_slots;
    std::vector<int> counts; // per query; a count above selected_slots is malformed (zero output)
    std::uint32_t seed;
    // Queries are uniform in [-query_range, query_range): 1.5 gives logits of order one, 12 a
    // softmax concentrated on a few slots.
    float query_range = 1.5F;
};

int attention_case(const AttentionCase& c) {
    const int width = static_cast<int>(c.counts.size());
    const std::string label =
        std::string("qsa_selected_attention ") +
        (c.format == QsaKvFormat::BF16 ? "BF16" : "NVFP4") + " C=" + std::to_string(c.capacity) +
        " S=" + std::to_string(c.selected_slots) + " W=" + std::to_string(width);

    // Fill the state through qsa_state_append (verified exactly above), then decode the represented
    // state bytes on the host: the oracle consumes exactly what the Op reads.
    State state(c.format, c.capacity, 2, c.seed + 9);
    {
        const std::vector<float> k = random_bf16(
            static_cast<std::size_t>(kHeadDim) * kKvHeads * c.capacity, c.seed, -2.0F, 2.0F);
        const std::vector<float> v = random_bf16(k.size(), c.seed + 1, -3.0F, 3.0F);
        const std::vector<float> keys =
            random_bf16(static_cast<std::size_t>(kIndexDim) * c.capacity, c.seed + 2, -1.0F, 1.0F);
        const std::vector<std::int32_t> positions = make_positions(c.capacity, false);
        const std::vector<float> key_norm         = random_bf16(kIndexDim, c.seed + 5, -0.5F, 0.5F);
        std::vector<std::int32_t> slots(static_cast<std::size_t>(c.capacity));
        std::iota(slots.begin(), slots.end(), 0);
        const Upload dkey_norm(encode_bf16(key_norm));
        const Tensor key_norm_t(dkey_norm.data(), DType::BF16, {kIndexDim});
        append_rows(state, encode_bf16(k), encode_bf16(v), encode_bf16(keys), positions, slots,
                    {kTestLane}, key_norm_t);
    }
    const auto kv       = state.kv_geometry(c.format);
    const auto k_bytes  = state.logical(state.k, kv, kTestLane);
    const auto v_bytes  = state.logical(state.v, kv, kTestLane);
    const bool nvfp4    = c.format == QsaKvFormat::NVFP4G16;
    const auto k_scales = nvfp4 ? state.logical(state.k_scales, State::scale_geometry(), kTestLane)
                                : std::vector<std::uint8_t>{};
    const auto v_scales = nvfp4 ? state.logical(state.v_scales, State::scale_geometry(), kTestLane)
                                : std::vector<std::uint8_t>{};
    // decoded[(head * C + slot) * 256 + d]
    const auto decode = [&](const std::vector<std::uint8_t>& bytes,
                            const std::vector<std::uint8_t>& scales) {
        std::vector<double> values(static_cast<std::size_t>(kKvHeads) * c.capacity * kHeadDim);
        for (std::size_t i = 0; i < values.size(); ++i) {
            if (c.format == QsaKvFormat::BF16) {
                values[i] =
                    bf16_to_f32(static_cast<std::uint16_t>(bytes[2 * i] | (bytes[2 * i + 1] << 8)));
            } else {
                const std::uint8_t code = (i % 2 == 0) ? bytes[i / 2] & 0x0fU : bytes[i / 2] >> 4;
                values[i]               = quantized_weight::detail::decode_e2m1(code) *
                                          quantized_weight::detail::decode_e4m3fn(scales[i / 16]);
            }
        }
        return values;
    };
    const std::vector<double> keys_decoded   = decode(k_bytes, k_scales);
    const std::vector<double> values_decoded = decode(v_bytes, v_scales);

    const std::vector<float> q =
        random_bf16(static_cast<std::size_t>(kHeadDim) * kQueryHeads * width, c.seed + 3,
                    -c.query_range, c.query_range);
    std::vector<std::int32_t> selected(static_cast<std::size_t>(c.selected_slots) * width, -1);
    std::vector<std::int32_t> counts(c.counts.begin(), c.counts.end());
    std::mt19937 generator(c.seed + 4);
    std::vector<std::int32_t> all(static_cast<std::size_t>(c.capacity));
    std::iota(all.begin(), all.end(), 0);
    for (int t = 0; t < width; ++t) {
        std::shuffle(all.begin(), all.end(), generator);
        const int listed = std::min(c.counts[static_cast<std::size_t>(t)], c.selected_slots);
        std::copy_n(all.begin(), listed,
                    selected.begin() + static_cast<std::ptrdiff_t>(t) * c.selected_slots);
    }

    std::vector<double> expected(static_cast<std::size_t>(kHeadDim) * kQueryHeads * width, 0.0);
    for (int t = 0; t < width; ++t) {
        const int count = c.counts[static_cast<std::size_t>(t)];
        if (count <= 0 || count > c.selected_slots) { continue; }
        const std::int32_t* ids = selected.data() + static_cast<std::size_t>(t) * c.selected_slots;
        std::vector<double> logits(static_cast<std::size_t>(count));
        for (int h = 0; h < kQueryHeads; ++h) {
            const int g = h / (kQueryHeads / kKvHeads);
            const float* query =
                q.data() + (static_cast<std::size_t>(t) * kQueryHeads + h) * kHeadDim;
            double maximum = -INFINITY;
            for (int j = 0; j < count; ++j) {
                const double* key = keys_decoded.data() +
                                    (static_cast<std::size_t>(g) * c.capacity + ids[j]) * kHeadDim;
                double dot        = 0.0;
                for (int d = 0; d < kHeadDim; ++d) { dot += double(query[d]) * key[d]; }
                logits[static_cast<std::size_t>(j)] = dot / 16.0;
                maximum = std::max(maximum, logits[static_cast<std::size_t>(j)]);
            }
            double denominator = 0.0;
            for (auto& logit : logits) {
                logit = std::exp(logit - maximum);
                denominator += logit;
            }
            double* out =
                expected.data() + (static_cast<std::size_t>(t) * kQueryHeads + h) * kHeadDim;
            for (int j = 0; j < count; ++j) {
                const double* value =
                    values_decoded.data() +
                    (static_cast<std::size_t>(g) * c.capacity + ids[j]) * kHeadDim;
                const double weight = logits[static_cast<std::size_t>(j)] / denominator;
                for (int d = 0; d < kHeadDim; ++d) { out[d] += weight * value[d]; }
            }
        }
    }

    const auto actual_bits =
        run_attention_rows(state, encode_bf16(q), selected, counts, {kTestLane}, c.selected_slots);
    std::vector<double> actual(actual_bits.size());
    for (std::size_t i = 0; i < actual.size(); ++i) { actual[i] = bf16_to_f32(actual_bits[i]); }
    int failures = verify_pointwise(label, actual, expected, attention_criterion());
    // Empty and malformed columns are exact (positive) zero.
    for (int t = 0; t < width; ++t) {
        const int count = c.counts[static_cast<std::size_t>(t)];
        if (count > 0 && count <= c.selected_slots) { continue; }
        const auto begin =
            actual_bits.begin() + static_cast<std::ptrdiff_t>(t) * kQueryHeads * kHeadDim;
        if (!std::all_of(begin, begin + kQueryHeads * kHeadDim,
                         [](std::uint16_t bits) { return bits == 0; })) {
            std::cerr << label << ": empty query " << t << " is not exact zero\n";
            ++failures;
        }
    }
    failures += state.verify_guards(label);
    return failures;
}

// --- lane-batched calls --------------------------------------------------------------------------

// Three lanes of one paged state filled to different frontiers (one at a page edge plus two
// pages, one mid-page and mid-block) by lane-batched appends of rows padded with -1 slots, then
// lane-batched selection and attention over four rows (one per lane in permuted order plus a row
// whose lane is out of range) whose queries sit at page edges, block edges and their frontiers.
// Each lane's columns are checked against the FP64 oracles, and every row bit for bit against a
// one-row call (for attention, a one-row call of the same packed width whose other columns are
// padding).
int lanes_case(QsaKvFormat format, QsaSelectBudget budget, std::uint32_t seed) {
    const std::string label = std::string("qsa lanes ") +
                              (format == QsaKvFormat::BF16 ? "BF16" : "NVFP4") +
                              " budget=" + std::to_string(static_cast<int>(budget));
    constexpr int kLanes    = 3;
    constexpr int kCapacity = 9216;
    const int b             = static_cast<int>(budget);
    const std::array<int, kLanes> frontiers{9000, 4 * b + 2 * kPageTokens, 4 * b + 63};
    State state(format, kCapacity, kLanes, seed);
    const std::vector<float> key_norm = random_bf16(kIndexDim, seed + 1, -0.5F, 0.5F);
    const Upload dkey_norm(encode_bf16(key_norm));
    const Tensor key_norm_t(dkey_norm.data(), DType::BF16, {kIndexDim});

    // Per-lane logical inputs.
    std::array<IndexInputs, kLanes> lanes{};
    std::array<std::vector<float>, kLanes> lane_k, lane_v;
    for (int lane = 0; lane < kLanes; ++lane) {
        const auto lane_seed = seed + 100U * static_cast<std::uint32_t>(lane + 1);
        IndexInputs& in      = lanes[static_cast<std::size_t>(lane)];
        in.capacity          = kCapacity;
        in.keys =
            random_bf16(static_cast<std::size_t>(kCapacity) * kIndexDim, lane_seed, -1.0F, 1.0F);
        in.positions                           = make_positions(kCapacity, lane == 1);
        in.query_norm                          = random_bf16(kIndexDim, seed + 2, -0.5F, 0.5F);
        in.key_norm                            = key_norm;
        lane_k[static_cast<std::size_t>(lane)] = random_bf16(
            static_cast<std::size_t>(kCapacity) * kKvHeads * kHeadDim, lane_seed + 1, -2.0F, 2.0F);
        lane_v[static_cast<std::size_t>(lane)] = random_bf16(
            static_cast<std::size_t>(kCapacity) * kKvHeads * kHeadDim, lane_seed + 2, -3.0F, 3.0F);
    }
    // Lane-batched appends: row r writes lane r's next kChunk slots, padding past its frontier.
    constexpr int kChunk = 3000;
    for (int begin = 0; begin < frontiers[0]; begin += kChunk) {
        std::vector<std::uint16_t> k, v, keys;
        std::vector<std::int32_t> positions, slots, rows;
        for (int lane = 0; lane < kLanes; ++lane) {
            const IndexInputs& in = lanes[static_cast<std::size_t>(lane)];
            rows.push_back(lane);
            for (int axis = 0; axis < 3; ++axis) {
                for (int j = 0; j < kChunk; ++j) {
                    const int slot = std::min(begin + j, kCapacity - 1);
                    positions.push_back(
                        in.positions[static_cast<std::size_t>(axis) * kCapacity + slot]);
                }
            }
            for (int j = 0; j < kChunk; ++j) {
                const int slot  = begin + j;
                const bool real = slot < frontiers[static_cast<std::size_t>(lane)];
                slots.push_back(real ? slot : -1);
                const int source = std::min(slot, kCapacity - 1);
                const auto kv_at = [&](const std::vector<float>& plane) {
                    return plane.begin() +
                           static_cast<std::ptrdiff_t>(source) * kKvHeads * kHeadDim;
                };
                for (auto it = kv_at(lane_k[static_cast<std::size_t>(lane)]);
                     it != kv_at(lane_k[static_cast<std::size_t>(lane)]) + kKvHeads * kHeadDim;
                     ++it) {
                    k.push_back(f32_to_bf16(*it));
                }
                for (auto it = kv_at(lane_v[static_cast<std::size_t>(lane)]);
                     it != kv_at(lane_v[static_cast<std::size_t>(lane)]) + kKvHeads * kHeadDim;
                     ++it) {
                    v.push_back(f32_to_bf16(*it));
                }
                for (int d = 0; d < kIndexDim; ++d) {
                    keys.push_back(
                        f32_to_bf16(in.keys[static_cast<std::size_t>(source) * kIndexDim + d]));
                }
            }
        }
        append_rows(state, k, v, keys, positions, slots, rows, key_norm_t);
    }

    // Queries: five columns per row; row r reads lane row_lanes[r]; row 3's lane is out of range.
    constexpr int kWidth = 5;
    const std::array<std::int32_t, 4> row_lanes{2, 0, 1, 7};
    std::vector<std::int32_t> slots;
    for (int row = 0; row < 4; ++row) {
        const int lane     = row_lanes[static_cast<std::size_t>(row)];
        const int frontier = lane < kLanes ? frontiers[static_cast<std::size_t>(lane)] : 100;
        // The frontier, a column a block earlier, the two slots around the last page edge below
        // the frontier, and padding; every query slot lies below its lane's frontier.
        const int page_edge = (frontier - 1) / kPageTokens * kPageTokens;
        for (const int slot : {frontier - 1, frontier - 5, page_edge - 1, page_edge, -1}) {
            slots.push_back(slot);
        }
    }
    const std::vector<float> query =
        random_bf16(slots.size() * kIndexHeads * kIndexDim, seed + 3, -1.0F, 1.0F);
    const std::vector<std::int32_t> lane_rows(row_lanes.begin(), row_lanes.end());
    const int max_visible = frontiers[0];
    const SelectResult batched =
        run_select_rows(state, query, slots, lane_rows, lanes[0].query_norm, budget, max_visible);
    const int selected_slots = ops::qsa_selected_capacity(budget);

    int failures = 0;
    for (int row = 0; row < 4; ++row) {
        const auto column_begin = static_cast<std::ptrdiff_t>(row) * kWidth;
        const std::vector<std::int32_t> row_slots(slots.begin() + column_begin,
                                                  slots.begin() + column_begin + kWidth);
        const std::vector<float> row_query(query.begin() + column_begin * kIndexHeads * kIndexDim,
                                           query.begin() +
                                               (column_begin + kWidth) * kIndexHeads * kIndexDim);
        const SelectResult single =
            run_select_rows(state, row_query, row_slots, {row_lanes[static_cast<std::size_t>(row)]},
                            lanes[0].query_norm, budget, max_visible);
        const std::vector<std::int32_t> batched_ids(
            batched.ids.begin() + column_begin * selected_slots,
            batched.ids.begin() + (column_begin + kWidth) * selected_slots);
        const std::vector<std::int32_t> batched_count(
            batched.count.begin() + column_begin, batched.count.begin() + column_begin + kWidth);
        const std::string row_label = label + " row " + std::to_string(row);
        failures +=
            verify_exact((row_label + " ids vs one-row call").c_str(), batched_ids, single.ids);
        failures += verify_exact((row_label + " count vs one-row call").c_str(), batched_count,
                                 single.count);
        const int lane = row_lanes[static_cast<std::size_t>(row)];
        if (lane >= kLanes) {
            if (std::any_of(batched_count.begin(), batched_count.end(),
                            [](std::int32_t c) { return c != 0; }) ||
                std::any_of(batched_ids.begin(), batched_ids.end(),
                            [](std::int32_t id) { return id != -1; })) {
                std::cerr << row_label << ": out-of-range lane selected slots\n";
                ++failures;
            }
            continue;
        }
        IndexInputs in = lanes[static_cast<std::size_t>(lane)];
        in.query       = row_query;
        in.slots       = row_slots;
        failures += verify_select(row_label, in, budget, max_visible, {batched_ids, batched_count},
                                  /*exact_ties*/ false);
    }

    // Attention over the selection.
    const std::vector<float> q =
        random_bf16(slots.size() * kQueryHeads * kHeadDim, seed + 4, -1.5F, 1.5F);
    const auto q_bits = encode_bf16(q);
    const auto actual =
        run_attention_rows(state, q_bits, batched.ids, batched.count, lane_rows, selected_slots);
    const int queries = static_cast<int>(slots.size());
    for (int row = 0; row < 4; ++row) {
        const int lane              = row_lanes[static_cast<std::size_t>(row)];
        const std::string row_label = label + " attention row " + std::to_string(row);
        // One-row call of the same packed width: this row's columns, the rest padding.
        std::vector<std::int32_t> padded_counts(static_cast<std::size_t>(queries), 0);
        std::copy_n(batched.count.begin() + static_cast<std::ptrdiff_t>(row) * kWidth, kWidth,
                    padded_counts.begin());
        std::vector<std::int32_t> padded_ids(static_cast<std::size_t>(queries) * selected_slots,
                                             -1);
        std::copy_n(batched.ids.begin() +
                        static_cast<std::ptrdiff_t>(row) * kWidth * selected_slots,
                    kWidth * selected_slots, padded_ids.begin());
        std::vector<std::uint16_t> padded_q(q_bits.size(), 0);
        constexpr std::size_t kColumn = static_cast<std::size_t>(kQueryHeads) * kHeadDim;
        std::copy_n(q_bits.begin() + static_cast<std::ptrdiff_t>(static_cast<std::size_t>(row) *
                                                                 kWidth * kColumn),
                    kWidth * kColumn, padded_q.begin());
        const auto single =
            run_attention_rows(state, padded_q, padded_ids, padded_counts, {lane}, selected_slots);
        const std::vector<std::uint16_t> batched_row(
            actual.begin() +
                static_cast<std::ptrdiff_t>(static_cast<std::size_t>(row) * kWidth * kColumn),
            actual.begin() +
                static_cast<std::ptrdiff_t>(static_cast<std::size_t>(row + 1) * kWidth * kColumn));
        failures += verify_exact(
            (row_label + " vs one-row call").c_str(), batched_row,
            std::vector<std::uint16_t>(
                single.begin(), single.begin() + static_cast<std::ptrdiff_t>(kWidth * kColumn)));
        if (lane >= kLanes) {
            if (!std::all_of(batched_row.begin(), batched_row.end(),
                             [](std::uint16_t bits) { return bits == 0; })) {
                std::cerr << row_label << ": out-of-range lane is not exact zero\n";
                ++failures;
            }
            continue;
        }
        // FP64 oracle over the lane's represented (gathered and decoded) K/V.
        const auto kv       = state.kv_geometry(format);
        const auto k_bytes  = state.logical(state.k, kv, lane);
        const auto v_bytes  = state.logical(state.v, kv, lane);
        const bool nvfp4    = format == QsaKvFormat::NVFP4G16;
        const auto k_scales = nvfp4 ? state.logical(state.k_scales, State::scale_geometry(), lane)
                                    : std::vector<std::uint8_t>{};
        const auto v_scales = nvfp4 ? state.logical(state.v_scales, State::scale_geometry(), lane)
                                    : std::vector<std::uint8_t>{};
        const auto element  = [&](const std::vector<std::uint8_t>& bytes,
                                  const std::vector<std::uint8_t>& scales, std::size_t i) {
            if (!nvfp4) {
                return double(bf16_to_f32(
                    static_cast<std::uint16_t>(bytes[2 * i] | (bytes[2 * i + 1] << 8))));
            }
            const std::uint8_t code = (i % 2 == 0) ? bytes[i / 2] & 0x0fU : bytes[i / 2] >> 4;
            return quantized_weight::detail::decode_e2m1(code) *
                   quantized_weight::detail::decode_e4m3fn(scales[i / 16]);
        };
        std::vector<double> expected(static_cast<std::size_t>(kWidth) * kColumn, 0.0);
        for (int j = 0; j < kWidth; ++j) {
            const int t     = row * kWidth + j;
            const int count = batched.count[static_cast<std::size_t>(t)];
            const std::int32_t* ids =
                batched.ids.data() + static_cast<std::size_t>(t) * selected_slots;
            for (int h = 0; h < kQueryHeads && count > 0; ++h) {
                const int g = h / (kQueryHeads / kKvHeads);
                const float* qh =
                    q.data() + (static_cast<std::size_t>(t) * kQueryHeads + h) * kHeadDim;
                std::vector<double> logits(static_cast<std::size_t>(count));
                double maximum = -INFINITY;
                for (int r = 0; r < count; ++r) {
                    const std::size_t row_base =
                        (static_cast<std::size_t>(g) * kCapacity + ids[r]) * kHeadDim;
                    double dot = 0.0;
                    for (int d = 0; d < kHeadDim; ++d) {
                        dot += double(qh[d]) * element(k_bytes, k_scales, row_base + d);
                    }
                    logits[static_cast<std::size_t>(r)] = dot / 16.0;
                    maximum = std::max(maximum, logits[static_cast<std::size_t>(r)]);
                }
                double denominator = 0.0;
                for (auto& logit : logits) {
                    logit = std::exp(logit - maximum);
                    denominator += logit;
                }
                double* out =
                    expected.data() + (static_cast<std::size_t>(j) * kQueryHeads + h) * kHeadDim;
                for (int r = 0; r < count; ++r) {
                    const std::size_t row_base =
                        (static_cast<std::size_t>(g) * kCapacity + ids[r]) * kHeadDim;
                    const double weight = logits[static_cast<std::size_t>(r)] / denominator;
                    for (int d = 0; d < kHeadDim; ++d) {
                        out[d] += weight * element(v_bytes, v_scales, row_base + d);
                    }
                }
            }
        }
        std::vector<double> actual_values(batched_row.size());
        for (std::size_t i = 0; i < actual_values.size(); ++i) {
            actual_values[i] = bf16_to_f32(batched_row[i]);
        }
        failures += verify_pointwise(row_label, actual_values, expected, attention_criterion());
    }
    failures += state.verify_guards(label);
    return failures;
}

std::vector<int> random_counts(int width, int lo, int hi, std::uint32_t seed) {
    std::mt19937 generator(seed);
    std::uniform_int_distribution<int> distribution(lo, hi);
    std::vector<int> counts(static_cast<std::size_t>(width));
    for (auto& count : counts) { count = distribution(generator); }
    return counts;
}

int run_full() {
    int failures = append_case(QsaKvFormat::BF16);
    failures += append_case(QsaKvFormat::NVFP4G16);

    for (const QsaSelectBudget budget :
         {QsaSelectBudget::Blocks512, QsaSelectBudget::Blocks1024, QsaSelectBudget::Blocks2048}) {
        const int b = static_cast<int>(budget);
        // Padding and empty-prefix columns, prefixes at and around the budget boundary (B = budget
        // lists everything; B = budget + 1 ranks), 3000 and 9000 visible tokens with every tail
        // length, and a query past max_visible_tokens (malformed: empty).
        const std::vector<std::int32_t> slots{-1,        0,         2,    4 * b - 1, 4 * b + 3,
                                              4 * b + 4, 4 * b + 6, 2999, 8996,      8997,
                                              8998,      8999,      9000};
        failures += select_random_case(budget, 9216, 9000, slots, b == 1024,
                                       101U + static_cast<std::uint32_t>(b));
        failures += select_tie_case(budget, {4 * b + 3, 4 * b + 6, 12 * b + 1, 12 * b + 3},
                                    211U + static_cast<std::uint32_t>(b));
    }
    // Single-query decode: a short prefix that lists everything without ranking (no workspace),
    // and a long prefix that ranks.
    failures += select_random_case(QsaSelectBudget::Blocks512, 1024, 101, {100}, false, 263U);
    failures += select_random_case(QsaSelectBudget::Blocks512, 9216, 9000, {8999}, true, 269U);
    // Three queries with 8800-10000 complete blocks (rows longer than one listing tile): a select
    // CTA cluster per query, with the budget boundary inside a tie group spread over its CTAs.
    failures += select_tie_case(QsaSelectBudget::Blocks1024, {35203, 37400, 39702}, 277U);
    // Twenty queries (streaming scoring, one select CTA per query) whose 8800-10000 complete
    // blocks span two listing tiles, with the budget boundary inside a tie group.
    {
        std::vector<std::int32_t> slots(20);
        for (int i = 0; i < 20; ++i) { slots[static_cast<std::size_t>(i)] = 35200 + 237 * i; }
        failures += select_tie_case(QsaSelectBudget::Blocks1024, slots, 281U);
    }
    // Full capacity with 130 decode-like queries: two score chunks of at most 128 queries, the
    // first scored register-tiled and selected with one CTA per query, the second scored by the
    // streaming route and selected with a CTA cluster per query.
    {
        constexpr int kCapacity = ops::kQsaMaximumCapacity;
        std::vector<std::int32_t> slots(130);
        for (int i = 0; i < 130; ++i) {
            slots[static_cast<std::size_t>(i)] = kCapacity - 1 - 1999 * i;
        }
        failures += select_random_case(QsaSelectBudget::Blocks2048, kCapacity, kCapacity, slots,
                                       false, 307U);
    }

    failures += lanes_case(QsaKvFormat::NVFP4G16, QsaSelectBudget::Blocks512, 601U);
    failures += lanes_case(QsaKvFormat::BF16, QsaSelectBudget::Blocks1024, 607U);
    for (const QsaKvFormat format : {QsaKvFormat::BF16, QsaKvFormat::NVFP4G16}) {
        // Few queries split each query's slots across CTAs; 171 queries fill the GPU unsplit.
        failures += attention_case({format, 9024, 8195, {8195, 1, 0, 2051, 8196, 37}, 401U});
        failures += attention_case({format, 9024, 8195, {8195}, 431U}); // single-query decode
        // Counts around the 64-rank tile and the 128-rank split of S = 520 (five splits).
        failures += attention_case({format, 640, 520, {520, 64, 65, 127, 128, 129, 1}, 437U});
        // A softmax concentrated on a few slots, split and unsplit.
        failures += attention_case({format, 9024, 8195, {8195, 300, 2}, 443U, 12.0F});
        failures +=
            attention_case({format, 3008, 2051, random_counts(171, 1, 2051, 449U), 457U, 12.0F});
        std::vector<int> counts = random_counts(171, 1, 600, 409U);
        counts[3]               = 2051;
        counts[17]              = 0;
        failures += attention_case({format, 3008, 2051, counts, 419U});
    }
    return failures;
}

// Reduced cases covering every kernel once for compute-sanitizer.
int run_sanitizer() {
    int failures = append_case(QsaKvFormat::BF16);
    failures += append_case(QsaKvFormat::NVFP4G16);
    failures +=
        select_random_case(QsaSelectBudget::Blocks512, 3072, 3000, {-1, 7, 2999, 2997}, true, 501U);
    failures += select_tie_case(QsaSelectBudget::Blocks512, {2051, 2400}, 503U);
    // Two queries with rows longer than one listing tile: a select CTA cluster per query.
    failures += select_tie_case(QsaSelectBudget::Blocks512, {32995, 33300}, 505U);
    // Thirty-three queries: register-tiled scoring and one select CTA per query.
    {
        std::vector<std::int32_t> slots(33);
        for (int i = 0; i < 33; ++i) { slots[static_cast<std::size_t>(i)] = 2200 + 50 * i; }
        failures += select_tie_case(QsaSelectBudget::Blocks512, slots, 507U);
    }
    failures += lanes_case(QsaKvFormat::NVFP4G16, QsaSelectBudget::Blocks512, 513U);
    for (const QsaKvFormat format : {QsaKvFormat::BF16, QsaKvFormat::NVFP4G16}) {
        failures += attention_case({format, 640, 520, {520, 0, 33}, 509U});
        failures += attention_case({format, 320, 64, random_counts(171, 0, 64, 511U), 521U});
    }
    return failures;
}

} // namespace

int main(int argc, char** argv) {
    if (require_cuda() != 0) { return 1; }
    try {
        const int failures = sanitizer_scope(argc, argv) ? run_sanitizer() : run_full();
        std::cout << (failures == 0 ? "OK" : "FAIL") << " qsa\n";
        return failures == 0 ? 0 : 1;
    } catch (const std::exception& error) {
        std::cerr << "qsa: " << error.what() << '\n';
        return 1;
    }
}
