#include "ninfer/ops/qsa.h"
#include "ops/op_tester.h"

#include <cuda_fp8.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <vector>

using namespace ninfer;
using namespace ninfer::test;

namespace {

struct StateFixture {
    explicit StateFixture(int capacity)
        : capacity(capacity), k_codes(static_cast<std::size_t>(128) * capacity * 2),
          v_codes(static_cast<std::size_t>(128) * capacity * 2),
          k_scales(static_cast<std::size_t>(16) * capacity * 2),
          v_scales(static_cast<std::size_t>(16) * capacity * 2),
          raw_keys(static_cast<std::size_t>(128) * capacity * sizeof(std::uint16_t)),
          positions(static_cast<std::size_t>(3) * capacity * sizeof(std::int32_t)) {
        k_codes.fill(0xcd);
        v_codes.fill(0xcd);
        k_scales.fill(0xcd);
        v_scales.fill(0xcd);
        raw_keys.fill(0xcd);
        positions.fill(0xcd);
    }

    ops::QsaStateView view() {
        return {
            Tensor(k_codes.data(), DType::U8, {128, capacity, 2}),
            Tensor(v_codes.data(), DType::U8, {128, capacity, 2}),
            Tensor(k_scales.data(), DType::FP8_E4M3FN, {16, capacity, 2}),
            Tensor(v_scales.data(), DType::FP8_E4M3FN, {16, capacity, 2}),
            Tensor(raw_keys.data(), DType::BF16, {128, capacity}),
            Tensor(positions.data(), DType::I32, {3, capacity}),
        };
    }

    int capacity;
    GuardedDeviceBuffer k_codes;
    GuardedDeviceBuffer v_codes;
    GuardedDeviceBuffer k_scales;
    GuardedDeviceBuffer v_scales;
    GuardedDeviceBuffer raw_keys;
    GuardedDeviceBuffer positions;
};

struct Bf16MatrixFixture {
    Bf16MatrixFixture(int rows, int columns)
        : rows(rows), columns(columns), bits(static_cast<std::size_t>(rows) * columns, 0),
          device(bits.size() * sizeof(std::uint16_t)) {}

    void set(int row, int column, float value) {
        bits[static_cast<std::size_t>(row) * columns + column] = f32_to_bf16(value);
    }

    Weight finish() {
        device.copy_from_host(bits.data(), device.bytes);
        Weight weight{};
        weight.payload         = device.p;
        weight.payload_bytes   = device.bytes;
        weight.qdata           = device.p;
        weight.qtype           = QType::BF16_CTRL;
        weight.layout          = QuantLayout::Contiguous;
        weight.n               = rows;
        weight.k               = columns;
        weight.ndim            = 2;
        weight.shape[0]        = rows;
        weight.shape[1]        = columns;
        weight.padded_shape[0] = rows;
        weight.padded_shape[1] = columns;
        return weight;
    }

    int rows;
    int columns;
    std::vector<std::uint16_t> bits;
    DeviceBuffer device;
};

struct Q5MatrixFixture {
    static constexpr int kBlockValues = 256;
    static constexpr int kBlockBytes  = 176;

    Q5MatrixFixture(int rows, int columns)
        : rows(rows), columns(columns), row_bytes((columns / kBlockValues) * kBlockBytes),
          bytes(static_cast<std::size_t>(rows) * row_bytes, 0), device(bytes.size()) {
        if (columns % kBlockValues != 0) {
            throw std::invalid_argument("Q5 test matrix columns must be block aligned");
        }
    }

    // Direct Q5_K hand encoding for a represented value of +1: d=1, dmin=0, the selected
    // 32-value group has scale=1, low code=1, and all high code bits are zero.
    void set_unit(int row, int column) {
        auto* block = bytes.data() + static_cast<std::size_t>(row) * row_bytes +
                      static_cast<std::size_t>(column / kBlockValues) * kBlockBytes;
        block[0] = 0x00U;
        block[1] = 0x3cU; // IEEE binary16 1.0, little endian
        const int within = column % kBlockValues;
        const int group  = within / 32;
        const int lane   = within % 32;
        if (group < 4) {
            block[4 + group] = 1U;
        } else {
            block[8 + group] = 1U;
        }
        block[48 + 32 * (group / 2) + lane] |=
            static_cast<std::uint8_t>(1U << (4 * (group & 1)));
    }

    Weight finish() {
        device.copy_from_host(bytes.data(), device.bytes);
        Weight weight{};
        weight.payload         = device.p;
        weight.payload_bytes   = device.bytes;
        weight.qdata           = device.p;
        weight.qtype           = QType::GGML_Q5_K;
        weight.group_size      = kBlockValues;
        weight.group           = kBlockValues;
        weight.layout          = QuantLayout::GgmlBlockRow;
        weight.n               = rows;
        weight.k               = columns;
        weight.ndim            = 2;
        weight.shape[0]        = rows;
        weight.shape[1]        = columns;
        weight.padded_shape[0] = rows;
        weight.padded_shape[1] = columns;
        return weight;
    }

    int rows;
    int columns;
    int row_bytes;
    std::vector<std::uint8_t> bytes;
    DeviceBuffer device;
};

std::size_t code_index(int byte, int token, int head, int capacity) {
    return static_cast<std::size_t>(byte) + 128U *
               (static_cast<std::size_t>(token) + static_cast<std::size_t>(capacity) * head);
}

std::size_t scale_index(int group, int token, int head, int capacity) {
    return static_cast<std::size_t>(group) + 16U *
               (static_cast<std::size_t>(token) + static_cast<std::size_t>(capacity) * head);
}

double decode_e2m1(std::uint8_t nibble) {
    constexpr double magnitude[]{0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0};
    return (nibble & 8U) == 0 ? magnitude[nibble & 7U] : -magnitude[nibble & 7U];
}

double decode_e4m3(std::uint8_t bits) {
    __nv_fp8_e4m3 value;
    value.__x = bits;
    return static_cast<double>(static_cast<float>(value));
}

double decode_cache(const std::vector<std::uint8_t>& codes,
                    const std::vector<std::uint8_t>& scales, int d, int token, int head,
                    int capacity) {
    const std::uint8_t packed = codes[code_index(d / 2, token, head, capacity)];
    const std::uint8_t nibble = (d & 1) == 0 ? packed & 0x0fU : packed >> 4U;
    return decode_e2m1(nibble) *
           decode_e4m3(scales[scale_index(d / 16, token, head, capacity)]);
}

int append_codec_and_attention_case() {
    constexpr int capacity = 8;
    constexpr int width    = 2;
    StateFixture state(capacity);
    std::vector<float> k(static_cast<std::size_t>(256) * 2 * width, 0.0F);
    std::vector<float> v(k.size());
    for (int token = 0; token < width; ++token) {
        for (int head = 0; head < 2; ++head) {
            const float value = (head == 0 ? 1.0F : -1.0F) * (token == 0 ? 3.0F : 6.0F);
            for (int d = 0; d < 256; ++d) {
                v[d + 256 * (head + 2 * token)] = value;
                const float sign = token == 0 ? 1.0F : -1.0F;
                k[d + 256 * (head + 2 * token)] = sign * ((d & 1) == 0 ? 6.0F : -6.0F);
            }
        }
    }
    std::vector<float> raw(static_cast<std::size_t>(128) * width);
    for (std::size_t i = 0; i < raw.size(); ++i) { raw[i] = static_cast<float>(i) / 128.0F; }
    round_to_bf16(raw);
    const std::vector<std::int32_t> position{10, 20, 30, 40, 50, 60};
    const std::vector<std::int32_t> ids{0, 7};
    auto dk = to_device_bf16(k);
    auto dv = to_device_bf16(v);
    auto draw = to_device_bf16(raw);
    auto dposition = to_device(position);
    auto dids = to_device(ids);
    Tensor kt(dk.p, DType::BF16, {256, 2, width});
    Tensor vt(dv.p, DType::BF16, {256, 2, width});
    Tensor rt(draw.p, DType::BF16, {128, width});
    Tensor pt(dposition.p, DType::I32, {3, width});
    Tensor it(dids.p, DType::I32, {width});
    auto state_view = state.view();
    ops::qsa_state_append(kt, vt, rt, pt, it, state_view, nullptr);
    cuda_synchronize();

    const auto k_codes =
        from_device<std::uint8_t>(state.k_codes.data(), state.k_codes.bytes());
    const auto v_codes =
        from_device<std::uint8_t>(state.v_codes.data(), state.v_codes.bytes());
    const auto k_scales =
        from_device<std::uint8_t>(state.k_scales.data(), state.k_scales.bytes());
    const auto v_scales =
        from_device<std::uint8_t>(state.v_scales.data(), state.v_scales.bytes());
    int failures = 0;
    // Direct hand formula: absmax 6 -> scale 1 (E4M3 0x38); [+6,-6] -> E2M1 [7,15].
    for (int token : ids) {
        for (int head = 0; head < 2; ++head) {
            for (int group = 0; group < 16; ++group) {
                if (k_scales[scale_index(group, token, head, capacity)] != 0x38U) { ++failures; }
                for (int pair = 0; pair < 8; ++pair) {
                    const std::uint8_t expected_k = token == 0 ? 0xf7U : 0x7fU;
                    if (k_codes[code_index(group * 8 + pair, token, head, capacity)] != expected_k) {
                        ++failures;
                    }
                }
                const std::uint8_t expected_scale = token == 0 ? 0x30U : 0x38U;
                const std::uint8_t expected_code = head == 0 ? 0x77U : 0xffU;
                if (v_scales[scale_index(group, token, head, capacity)] != expected_scale) {
                    ++failures;
                }
                for (int pair = 0; pair < 8; ++pair) {
                    if (v_codes[code_index(group * 8 + pair, token, head, capacity)] !=
                        expected_code) {
                        ++failures;
                    }
                }
            }
        }
    }
    if (failures != 0) { std::cerr << "FAIL: qsa NVFP4 direct codec witness\n"; }

    const auto stored_raw = from_device<std::uint16_t>(state.raw_keys.data(),
                                                       static_cast<std::size_t>(128) * capacity);
    const auto stored_pos = from_device<std::int32_t>(state.positions.data(),
                                                      static_cast<std::size_t>(3) * capacity);
    for (int token = 0; token < width; ++token) {
        const int id = ids[token];
        for (int d = 0; d < 128; ++d) {
            if (stored_raw[d + 128 * id] != f32_to_bf16(raw[d + 128 * token])) { ++failures; }
        }
        for (int axis = 0; axis < 3; ++axis) {
            if (stored_pos[axis + 3 * id] != position[axis + 3 * token]) { ++failures; }
        }
    }

    // Independent FP64 attention oracle below starts from the exact stored code/scale bits. The
    // opposite K rows force a nonuniform softmax and witness the fixed scale and 12:1 head map.
    std::vector<float> q(static_cast<std::size_t>(256) * 24);
    for (int head = 0; head < 24; ++head) {
        for (int d = 0; d < 256; ++d) { q[d + 256 * head] = (d & 1) == 0 ? 1.0F / 256 : -1.0F / 256; }
    }
    round_to_bf16(q);
    std::vector<std::int32_t> selected(ops::kQsaSelectedCapacity, -1);
    selected[0] = 7;
    selected[1] = 0;
    const std::vector<std::int32_t> count{2};
    auto dq = to_device_bf16(q);
    auto dselected = to_device(selected);
    auto dcount = to_device(count);
    GuardedDeviceBuffer dout(static_cast<std::size_t>(256) * 24 * sizeof(std::uint16_t));
    Tensor qt(dq.p, DType::BF16, {256, 24, 1});
    Tensor st(dselected.p, DType::I32, {ops::kQsaSelectedCapacity, 1});
    Tensor ct(dcount.p, DType::I32, {1});
    Tensor ot(dout.data(), DType::BF16, {256, 24, 1});
    ops::qsa_selected_attention(qt, st, ct, state_view, ot, nullptr);
    cuda_synchronize();
    const auto actual = from_device_bf16(dout.data(), static_cast<std::size_t>(256) * 24);
    std::vector<double> expected(actual.size());
    constexpr int selected_host[]{7, 0};
    for (int head = 0; head < 24; ++head) {
        const int kv_head = head / 12;
        double logits[2]{};
        for (int j = 0; j < 2; ++j) {
            for (int d = 0; d < 256; ++d) {
                logits[j] += static_cast<double>(q[d + 256 * head]) *
                             decode_cache(k_codes, k_scales, d, selected_host[j], kv_head,
                                          capacity);
            }
            logits[j] /= 16.0;
        }
        const double maximum = std::max(logits[0], logits[1]);
        const double e0 = std::exp(logits[0] - maximum);
        const double e1 = std::exp(logits[1] - maximum);
        const double p0 = e0 / (e0 + e1);
        const double p1 = e1 / (e0 + e1);
        for (int d = 0; d < 256; ++d) {
            expected[d + 256 * head] =
                p0 * decode_cache(v_codes, v_scales, d, selected_host[0], kv_head, capacity) +
                p1 * decode_cache(v_codes, v_scales, d, selected_host[1], kv_head, capacity);
        }
    }
    failures += verify_pointwise("qsa selected attention decoded FP64 oracle", actual, expected,
                                 PointwiseCriterion{0.02, 0.005});
    failures += dout.verify_guards("qsa selected attention out");
    return failures;
}

int selector_ceiling_and_permutation_case() {
    constexpr int capacity = ops::kQsaMaximumTokens;
    constexpr int width    = 2;
    StateFixture state(capacity);
    std::vector<float> keys(static_cast<std::size_t>(128) * capacity, 0.0F);
    for (int id = 0; id < 4; ++id) { keys[128 * id] = -1.0F; }
    for (int id = 4; id < 8; ++id) { keys[128 * id] = 1.0F; }
    round_to_bf16(keys);
    std::vector<std::int32_t> positions(static_cast<std::size_t>(3) * capacity, 0);
    std::vector<std::uint16_t> key_bits(keys.size());
    for (std::size_t i = 0; i < keys.size(); ++i) { key_bits[i] = f32_to_bf16(keys[i]); }
    state.raw_keys.copy_from_host(key_bits.data(), key_bits.size() * sizeof(std::uint16_t));
    state.positions.copy_from_host(positions.data(), positions.size() * sizeof(std::int32_t));

    std::vector<float> query(static_cast<std::size_t>(128) * 4 * width, 0.0F);
    for (int head = 0; head < 4; ++head) { query[128 * (head + 4)] = 1.0F; }
    std::vector<float> norm(128, 0.0F);
    const std::vector<std::int32_t> query_ids{4095, 8};
    std::vector<std::int32_t> visible;
    visible.reserve(4105);
    for (int i = 0; i < capacity; ++i) { visible.push_back(i); }
    for (int i = 0; i < 9; ++i) { visible.push_back(i); }
    const std::vector<std::int32_t> offsets{0, capacity, capacity + 9};
    auto dquery = to_device_bf16(query);
    auto dnorm_q = to_device_f32(norm);
    auto dnorm_k = to_device_f32(norm);
    auto dquery_ids = to_device(query_ids);
    auto dvisible = to_device(visible);
    auto doffsets = to_device(offsets);
    GuardedDeviceBuffer dselected(static_cast<std::size_t>(ops::kQsaSelectedCapacity) * width *
                                  sizeof(std::int32_t));
    GuardedDeviceBuffer dcount(width * sizeof(std::int32_t));
    GuardedDeviceBuffer workspace(ops::qsa_index_select_workspace_bytes(width));
    Tensor query_t(dquery.p, DType::BF16, {128, 4, width});
    Tensor qid_t(dquery_ids.p, DType::I32, {width});
    Tensor visible_t(dvisible.p, DType::I32, {static_cast<int>(visible.size())});
    Tensor offsets_t(doffsets.p, DType::I32, {width + 1});
    Tensor qnorm_t(dnorm_q.p, DType::FP32, {128});
    Tensor knorm_t(dnorm_k.p, DType::FP32, {128});
    Tensor selected_t(dselected.data(), DType::I32, {ops::kQsaSelectedCapacity, width});
    Tensor count_t(dcount.data(), DType::I32, {width});
    Tensor workspace_t(workspace.data(), DType::U8,
                       {static_cast<int>(workspace.bytes())});
    auto state_view = state.view();
    ops::qsa_index_select(query_t, state_view, qid_t, visible_t, offsets_t, qnorm_t, knorm_t,
                          selected_t, count_t, workspace_t, nullptr);
    cuda_synchronize();
    const auto actual_count = from_device<std::int32_t>(dcount.data(), width);
    const auto actual = from_device<std::int32_t>(
        dselected.data(), static_cast<std::size_t>(ops::kQsaSelectedCapacity) * width);
    int failures = verify_exact("qsa selector counts", actual_count,
                                std::vector<std::int32_t>{2048, 9});
    for (int i = 0; i < 2048; ++i) {
        if (actual[i] != i) { ++failures; }
    }
    for (int i = 2048; i < ops::kQsaSelectedCapacity; ++i) {
        if (actual[i] != -1) { ++failures; }
    }
    // A represented gamma of zero must zero every normalized score. This distinguishes the GGUF
    // gamma contract from applying a second unit offset and makes both blocks an exact tie.
    const std::vector<std::int32_t> permutation{0, 1, 2, 3, 4, 5, 6, 7, 8};
    for (int i = 0; i < 9; ++i) {
        if (actual[ops::kQsaSelectedCapacity + i] != permutation[i]) { ++failures; }
    }
    for (int i = 9; i < ops::kQsaSelectedCapacity; ++i) {
        if (actual[ops::kQsaSelectedCapacity + i] != -1) { ++failures; }
    }
    if (failures != 0) { std::cerr << "FAIL: qsa selector ceiling/tie/permutation witness\n"; }
    failures += dselected.verify_guards("qsa selected ids");
    failures += dcount.verify_guards("qsa selected count");
    failures += workspace.verify_guards("qsa selector workspace");
    return failures;
}

std::array<double, 256> oracle_core_norm_rope(const std::array<double, 256>& raw,
                                              const std::array<float, 256>& gamma,
                                              const std::array<std::int32_t, 3>& position) {
    double sum2 = 0.0;
    for (double value : raw) { sum2 += value * value; }
    const double inv_rms = 1.0 / std::sqrt(sum2 / 256.0 + 1.0e-6);
    std::array<double, 256> normalized{};
    for (int d = 0; d < 256; ++d) { normalized[d] = raw[d] * inv_rms * gamma[d]; }
    auto output = normalized;
    for (int pair = 0; pair < 32; ++pair) {
        const double inverse_frequency = std::pow(1.0e7, -static_cast<double>(pair) / 32.0);
        const double phase = static_cast<double>(position[pair % 3]) * inverse_frequency;
        const double cosine = std::cos(phase);
        const double sine   = std::sin(phase);
        output[pair]      = normalized[pair] * cosine - normalized[pair + 32] * sine;
        output[pair + 32] = normalized[pair + 32] * cosine + normalized[pair] * sine;
    }
    return output;
}

int verifier_composite_real_shape_case() {
    constexpr int capacity = 16;
    StateFixture state(capacity);

    // Seed two complete visible blocks through the public state transition. Their opposite
    // index keys witness ranked-block permutation; their K/V rows make attention nonuniform.
    constexpr int old_tokens = 8;
    std::vector<float> old_k(static_cast<std::size_t>(256) * 2 * old_tokens);
    std::vector<float> old_v(old_k.size());
    std::vector<float> old_index(static_cast<std::size_t>(128) * old_tokens, 0.0F);
    std::vector<std::int32_t> old_positions(static_cast<std::size_t>(3) * old_tokens, 0);
    std::vector<std::int32_t> old_ids(old_tokens);
    for (int token = 0; token < old_tokens; ++token) {
        old_ids[token] = token;
        old_index[128 * token] = token < 4 ? -1.0F : 1.0F;
        const float key_sign = token < 4 ? -1.0F : 1.0F;
        for (int head = 0; head < 2; ++head) {
            const float value = (head == 0 ? 1.0F : -1.0F) * (token + 1) / 8.0F;
            for (int d = 0; d < 256; ++d) {
                old_k[d + 256 * (head + 2 * token)] = key_sign * ((d & 1) ? -1.0F : 1.0F);
                old_v[d + 256 * (head + 2 * token)] = value;
            }
        }
    }
    auto d_old_k = to_device_bf16(old_k);
    auto d_old_v = to_device_bf16(old_v);
    auto d_old_index = to_device_bf16(old_index);
    auto d_old_positions = to_device(old_positions);
    auto d_old_ids = to_device(old_ids);
    Tensor old_k_t(d_old_k.p, DType::BF16, {256, 2, old_tokens});
    Tensor old_v_t(d_old_v.p, DType::BF16, {256, 2, old_tokens});
    Tensor old_index_t(d_old_index.p, DType::BF16, {128, old_tokens});
    Tensor old_positions_t(d_old_positions.p, DType::I32, {3, old_tokens});
    Tensor old_ids_t(d_old_ids.p, DType::I32, {old_tokens});
    auto state_view = state.view();
    ops::qsa_state_append(old_k_t, old_v_t, old_index_t, old_positions_t, old_ids_t, state_view,
                          nullptr);

    // Real artifact extents with direct sparse witnesses. Every Q5_K nonzero is independently
    // encoded as d=1, scale=1, code=1 by Q5MatrixFixture::set_unit.
    Bf16MatrixFixture index_query(512, 2560);
    Bf16MatrixFixture index_key(128, 2560);
    for (int head = 0; head < 4; ++head) { index_query.set(128 * head, 0, 1.0F); }
    index_key.set(0, 0, 1.0F);

    Q5MatrixFixture core_query_gate(12288, 2560);
    Q5MatrixFixture core_key(512, 2560);
    Q5MatrixFixture core_value(512, 2560);
    Q5MatrixFixture output(2560, 6144);
    for (int head = 0; head < 24; ++head) {
        for (int d = 0; d < 256; ++d) {
            core_query_gate.set_unit(head * 512 + d, (d & 1) == 0 ? 0 : 2);
            core_query_gate.set_unit(head * 512 + 256 + d, (head & 1) == 0 ? 0 : 2);
        }
    }
    for (int head = 0; head < 2; ++head) {
        for (int d = 0; d < 256; ++d) {
            core_key.set_unit(head * 256 + d, (d & 1) == 0 ? 0 : 2);
            core_value.set_unit(head * 256 + d, head == 0 ? 1 : 3);
        }
    }
    for (int row = 0; row < 2560; ++row) { output.set_unit(row, row); }

    std::vector<float> x(2560, 0.0F);
    x[0] = 1.0F;
    x[1] = 0.5F;
    x[2] = -1.0F;
    x[3] = -0.5F;
    round_to_bf16(x);
    const std::vector<std::int32_t> token_id{8};
    const std::vector<std::int32_t> position{1, 2, 3};
    const std::vector<std::int32_t> visible_ids{0, 1, 2, 3, 4, 5, 6, 7, 8};
    const std::vector<std::int32_t> visible_offsets{0, 9};
    const std::vector<float> index_norm(128, 0.0F);
    const std::vector<float> core_norm(256, 0.5F);
    auto dx = to_device_bf16(x);
    auto dtoken_id = to_device(token_id);
    auto dposition = to_device(position);
    auto dvisible_ids = to_device(visible_ids);
    auto dvisible_offsets = to_device(visible_offsets);
    auto dindex_query_norm = to_device_f32(index_norm);
    auto dindex_key_norm = to_device_f32(index_norm);
    auto dcore_query_norm = to_device_f32(core_norm);
    auto dcore_key_norm = to_device_f32(core_norm);
    GuardedDeviceBuffer dselected(static_cast<std::size_t>(ops::kQsaSelectedCapacity) *
                                  sizeof(std::int32_t));
    GuardedDeviceBuffer dcount(sizeof(std::int32_t));
    GuardedDeviceBuffer dout(2560 * sizeof(std::uint16_t));
    GuardedDeviceBuffer workspace(ops::qsa_verifier_workspace_bytes());
    dout.fill(0xcd);
    workspace.fill(0xcd);

    ops::QsaVerifierWeights weights{
        index_query.finish(),
        index_key.finish(),
        core_query_gate.finish(),
        core_key.finish(),
        core_value.finish(),
        output.finish(),
        Tensor(dindex_query_norm.p, DType::FP32, {128}),
        Tensor(dindex_key_norm.p, DType::FP32, {128}),
        Tensor(dcore_query_norm.p, DType::FP32, {256}),
        Tensor(dcore_key_norm.p, DType::FP32, {256}),
    };
    Tensor x_t(dx.p, DType::BF16, {2560});
    Tensor token_id_t(dtoken_id.p, DType::I32, {1});
    Tensor position_t(dposition.p, DType::I32, {3});
    Tensor visible_ids_t(dvisible_ids.p, DType::I32, {9});
    Tensor visible_offsets_t(dvisible_offsets.p, DType::I32, {2});
    Tensor selected_t(dselected.data(), DType::I32, {ops::kQsaSelectedCapacity});
    Tensor count_t(dcount.data(), DType::I32, {1});
    Tensor out_t(dout.data(), DType::BF16, {2560});
    Tensor workspace_t(workspace.data(), DType::U8,
                       {static_cast<int>(workspace.bytes())});
    ops::qsa_verifier_token(x_t, token_id_t, position_t, visible_ids_t, visible_offsets_t,
                            weights, state_view, selected_t, count_t, out_t, workspace_t,
                            nullptr);
    cuda_synchronize();

    int failures = 0;
    const auto actual_count = from_device<std::int32_t>(dcount.data(), 1);
    const auto selected = from_device<std::int32_t>(dselected.data(), ops::kQsaSelectedCapacity);
    // Zero represented index gamma makes every complete-block score an exact tie. The stable
    // lower-rank order is also a strict witness that no implicit +1 is applied.
    const std::vector<std::int32_t> expected_ids{0, 1, 2, 3, 4, 5, 6, 7, 8};
    failures += verify_exact("qsa verifier selected count", actual_count,
                             std::vector<std::int32_t>{9});
    for (int i = 0; i < static_cast<int>(expected_ids.size()); ++i) {
        if (selected[i] != expected_ids[static_cast<std::size_t>(i)]) { ++failures; }
    }
    for (int i = static_cast<int>(expected_ids.size()); i < ops::kQsaSelectedCapacity; ++i) {
        if (selected[i] != -1) { ++failures; }
    }
    if (failures != 0) { std::cerr << "FAIL: qsa verifier exact selector ids\n"; }

    const auto k_codes = from_device<std::uint8_t>(state.k_codes.data(), state.k_codes.bytes());
    const auto v_codes = from_device<std::uint8_t>(state.v_codes.data(), state.v_codes.bytes());
    const auto k_scales =
        from_device<std::uint8_t>(state.k_scales.data(), state.k_scales.bytes());
    const auto v_scales =
        from_device<std::uint8_t>(state.v_scales.data(), state.v_scales.bytes());
    const auto raw_keys = from_device<std::uint16_t>(state.raw_keys.data(),
                                                     static_cast<std::size_t>(128) * capacity);
    const auto stored_positions = from_device<std::int32_t>(
        state.positions.data(), static_cast<std::size_t>(3) * capacity);
    int state_metadata_failures = 0;
    if (raw_keys[128 * 8] != f32_to_bf16(1.0F)) { ++state_metadata_failures; }
    for (int d = 1; d < 128; ++d) {
        if (raw_keys[d + 128 * 8] != f32_to_bf16(0.0F)) { ++state_metadata_failures; }
    }
    for (int axis = 0; axis < 3; ++axis) {
        if (stored_positions[axis + 3 * 8] != position[axis]) { ++state_metadata_failures; }
    }
    if (state_metadata_failures != 0) {
        std::cerr << "FAIL: qsa verifier current index-key/position transition: "
                  << state_metadata_failures << "\n";
    }
    failures += state_metadata_failures;

    std::array<double, 256> raw_q{};
    for (int d = 0; d < 256; ++d) { raw_q[d] = (d & 1) == 0 ? 1.0 : -1.0; }
    std::array<float, 256> core_gamma{};
    core_gamma.fill(0.5F);
    const std::array<std::int32_t, 3> position_array{1, 2, 3};
    const auto query = oracle_core_norm_rope(raw_q, core_gamma, position_array);

    // The newly projected K/V must cross the represented cache boundary. Compare K against the
    // independent ideal norm/MRoPE result with an NVFP4 criterion and V against its represented
    // projection; the complete oracle below then consumes only exact-decoded cache values.
    int cache_roundtrip_failures = 0;
    double maximum_k_bound_ratio = 0.0;
    double maximum_v_bound_ratio = 0.0;
    for (int head = 0; head < 2; ++head) {
        for (int group = 0; group < 16; ++group) {
            double group_max = 0.0;
            for (int lane = 0; lane < 16; ++lane) {
                group_max = std::max(group_max, std::abs(query[group * 16 + lane]));
            }
            // E4M3's normal-range round-to-nearest scale is at most 17/16 of M/6. The largest
            // E2M1 code gap is two, so nearest-code error is at most one scale (and saturation
            // after a downward scale round is smaller). BF16 round-to-nearest contributes at
            // most |x|/256; 1e-4 conservatively covers the FP32 norm/RoPE evaluation before its
            // BF16 staging boundary. This is fixed from the codecs, not from observed output.
            const double staged_group_max = group_max + group_max / 256.0 + 1.0e-4;
            const double codec_bound = (17.0 / 96.0) * staged_group_max;
            for (int lane = 0; lane < 16; ++lane) {
                const int d = group * 16 + lane;
                const double cached_k = decode_cache(k_codes, k_scales, d, 8, head, capacity);
                const double k_error = std::abs(cached_k - query[d]);
                const double k_bound = std::abs(query[d]) / 256.0 + 1.0e-4 + codec_bound;
                maximum_k_bound_ratio = std::max(maximum_k_bound_ratio, k_error / k_bound);
                if (k_error > k_bound) { ++cache_roundtrip_failures; }

                const double expected_v = head == 0 ? 0.5 : -0.5;
                const double cached_v = decode_cache(v_codes, v_scales, d, 8, head, capacity);
                const double v_error = std::abs(cached_v - expected_v);
                const double v_bound = (17.0 / 96.0) * std::abs(expected_v);
                maximum_v_bound_ratio = std::max(maximum_v_bound_ratio, v_error / v_bound);
                if (v_error > v_bound) { ++cache_roundtrip_failures; }
            }
        }
    }
    if (cache_roundtrip_failures != 0) {
        std::cerr << "FAIL: qsa verifier current-token cache round-trip: "
                  << cache_roundtrip_failures << " max_k_bound_ratio=" << maximum_k_bound_ratio
                  << " max_v_bound_ratio=" << maximum_v_bound_ratio << "\n";
    }
    if (error_stats_enabled()) {
        std::cout << "OP_ERROR_STATS kind=codec_bound max_k_bound_ratio="
                  << maximum_k_bound_ratio << " max_v_bound_ratio=" << maximum_v_bound_ratio
                  << " case=qsa_verifier_current_token_nvfp4\n";
    }
    failures += cache_roundtrip_failures;

    // Complete independent FP64 formula from represented x/weights and exact decoded cache.
    // The direct Q5_K output map makes rows [0,2560) select the first ten gated heads.
    std::vector<double> expected(2560);
    for (int head = 0; head < 10; ++head) {
        const int kv_head = head / 12;
        std::array<double, 9> logits{};
        double maximum = -INFINITY;
        for (int item = 0; item < 9; ++item) {
            const int id = expected_ids[static_cast<std::size_t>(item)];
            for (int d = 0; d < 256; ++d) {
                logits[item] += query[d] *
                                decode_cache(k_codes, k_scales, d, id, kv_head, capacity);
            }
            logits[item] /= 16.0;
            maximum = std::max(maximum, logits[item]);
        }
        std::array<double, 9> probability{};
        double denominator = 0.0;
        for (int item = 0; item < 9; ++item) {
            probability[item] = std::exp(logits[item] - maximum);
            denominator += probability[item];
        }
        const double raw_gate = (head & 1) == 0 ? 1.0 : -1.0;
        const double gate = 1.0 / (1.0 + std::exp(-raw_gate));
        for (int d = 0; d < 256; ++d) {
            double attention = 0.0;
            for (int item = 0; item < 9; ++item) {
                const int id = expected_ids[static_cast<std::size_t>(item)];
                attention += probability[item] / denominator *
                             decode_cache(v_codes, v_scales, d, id, kv_head, capacity);
            }
            expected[d + 256 * head] = gate * attention;
        }
    }
    const auto actual = from_device_bf16(dout.data(), 2560);
    failures += verify_pointwise("qsa verifier complete FP64 oracle", actual, expected,
                                 PointwiseCriterion{0.025, 0.02});
    failures += state.k_codes.verify_guards("qsa verifier K codes");
    failures += state.v_codes.verify_guards("qsa verifier V codes");
    failures += state.k_scales.verify_guards("qsa verifier K scales");
    failures += state.v_scales.verify_guards("qsa verifier V scales");
    failures += state.raw_keys.verify_guards("qsa verifier raw index keys");
    failures += state.positions.verify_guards("qsa verifier positions");
    failures += dselected.verify_guards("qsa verifier selected ids");
    failures += dcount.verify_guards("qsa verifier selected count");
    failures += dout.verify_guards("qsa verifier output");
    failures += workspace.verify_guards("qsa verifier workspace");
    return failures;
}

} // namespace

int main() {
    if (require_cuda() != 0) { return 1; }
    int failures = 0;
    failures += append_codec_and_attention_case();
    failures += selector_ceiling_and_permutation_case();
    failures += verifier_composite_real_shape_case();
    if (failures != 0) {
        std::cerr << "qsa tests failed: " << failures << "\n";
        return 1;
    }
    std::cout << "qsa tests passed\n";
    return 0;
}
