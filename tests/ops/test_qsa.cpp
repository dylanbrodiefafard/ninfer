#include "ninfer/ops/qsa.h"
#include "ops/op_tester.h"
#include "ops/native_projection_fixture.h"
#include "targets/qwen4/native_bf16_fixture.h"
#include "targets/qwen4/native_sequence_components.h"

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

template <typename Function>
int expect_invalid_argument(const char* label, Function&& function) {
    try {
        function();
    } catch (const std::invalid_argument&) {
        return 0;
    }
    std::cerr << "FAIL: " << label << " accepted invalid input\n";
    return 1;
}

int verify_filled(const char* label, const void* device, std::size_t bytes,
                  std::uint8_t expected) {
    const auto actual = from_device<std::uint8_t>(device, bytes);
    if (std::all_of(actual.begin(), actual.end(),
                    [expected](std::uint8_t value) { return value == expected; })) {
        return 0;
    }
    std::cerr << "FAIL: " << label << " changed after rejected call\n";
    return 1;
}

struct StateFixture {
    explicit StateFixture(int capacity, ops::QsaKvFormat format = ops::QsaKvFormat::NVFP4G16)
        : capacity(capacity), format(format), k_codes(static_cast<std::size_t>(format == ops::QsaKvFormat::BF16 ? 512 : 128) * capacity * 2),
          v_codes(k_codes.bytes()),
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
        const bool bf16 = format == ops::QsaKvFormat::BF16;
        return {
            format,
            Tensor(k_codes.data(), bf16 ? DType::BF16 : DType::U8, {bf16 ? 256 : 128, capacity, 2}),
            Tensor(v_codes.data(), bf16 ? DType::BF16 : DType::U8, {bf16 ? 256 : 128, capacity, 2}),
            bf16 ? Tensor{} : Tensor(k_scales.data(), DType::FP8_E4M3FN, {16, capacity, 2}),
            bf16 ? Tensor{} : Tensor(v_scales.data(), DType::FP8_E4M3FN, {16, capacity, 2}),
            Tensor(raw_keys.data(), DType::BF16, {128, capacity}),
            Tensor(positions.data(), DType::I32, {3, capacity}),
        };
    }

    int capacity;
    ops::QsaKvFormat format;
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
    QType type;
    quantized_weight::PackedWeight native;
    static constexpr int kBlockValues = 256;
    static constexpr int kBlockBytes  = 176;

    Q5MatrixFixture(int rows, int columns, QType format = QType::GGML_Q5_K)
        : type(format), rows(rows), columns(columns), row_bytes((columns / kBlockValues) * kBlockBytes),
          bytes(static_cast<std::size_t>(rows) * row_bytes, 0), device(bytes.size()) {
        if (columns % kBlockValues != 0) {
            throw std::invalid_argument("Q5 test matrix columns must be block aligned");
        }
        if (native_projection_format(type)) {
            native = native_sparse_fixture(type, rows, columns);
            device = DeviceBuffer(native.payload.size());
        }
    }

    // Direct Q5_K hand encoding for a represented value of +1: d=1, dmin=0, the selected
    // 32-value group has scale=1, low code=1, and all high code bits are zero.
    void set_unit(int row, int column) {
        if (native_projection_format(type)) { native_sparse_set(native, row, column, 1.0F); return; }
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
        if (native_projection_format(type)) {
            device.copy_from_host(native.payload.data(), native.payload.size());
            return native.device_weight(device.p);
        }
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
    const int exponent=(bits>>3)&15, mantissa=bits&7;
    if(exponent==15 && mantissa==7) { return NAN; }
    const double magnitude=exponent==0 ? std::ldexp(double(mantissa),-9)
        : std::ldexp(1.0+double(mantissa)/8.0,exponent-7);
    return std::copysign(magnitude,bits&128 ? -1.0 : 1.0);
}

double decode_cache(const std::vector<std::uint8_t>& codes,
                    const std::vector<std::uint8_t>& scales, int d, int token, int head,
                    int capacity) {
    if (scales.empty()) {
        const auto offset = 2ULL * (d + 256ULL * (token + static_cast<std::size_t>(capacity) * head));
        const auto bits = static_cast<std::uint16_t>(codes[offset] | (codes[offset + 1] << 8));
        return bf16_to_f32(bits);
    }
    const std::uint8_t packed = codes[code_index(d / 2, token, head, capacity)];
    const std::uint8_t nibble = (d & 1) == 0 ? packed & 0x0fU : packed >> 4U;
    return decode_e2m1(nibble) *
           decode_e4m3(scales[scale_index(d / 16, token, head, capacity)]);
}

void oracle_encode_cache(std::span<const double> values,int token,int head,int capacity,
    std::vector<std::uint8_t>& codes,std::vector<std::uint8_t>& scales);

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
    std::vector<std::uint8_t> expected_k_codes(k_codes.size()),expected_v_codes(v_codes.size()),
        expected_k_scales(k_scales.size()),expected_v_scales(v_scales.size());
    for(int token=0;token<width;++token) for(int head=0;head<2;++head) {
        const int base=256*(head+2*token);
        oracle_encode_cache(std::vector<double>(k.begin()+base,k.begin()+base+256),ids[token],head,
            capacity,expected_k_codes,expected_k_scales);
        oracle_encode_cache(std::vector<double>(v.begin()+base,v.begin()+base+256),ids[token],head,
            capacity,expected_v_codes,expected_v_scales);
        for(int byte=0;byte<128;++byte) {
            const auto i=code_index(byte,ids[token],head,capacity);
            if(k_codes[i]!=expected_k_codes[i] || v_codes[i]!=expected_v_codes[i]) { ++failures; }
        }
        for(int group=0;group<16;++group) {
            const auto i=scale_index(group,ids[token],head,capacity);
            if(k_scales[i]!=expected_k_scales[i] || v_scales[i]!=expected_v_scales[i]) { ++failures; }
        }
    }
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
    constexpr int attention_width = 2;
    std::vector<float> q(static_cast<std::size_t>(256) * 24 * attention_width);
    for (int token = 0; token < attention_width; ++token) {
        for (int head = 0; head < 24; ++head) {
            for (int d = 0; d < 256; ++d) {
                const float sign = token == 0 ? 1.0F : -1.0F;
                q[d + 256 * (head + 24 * token)] =
                    sign * ((d & 1) == 0 ? 1.0F / 256 : -1.0F / 256);
            }
        }
    }
    round_to_bf16(q);
    std::vector<std::int32_t> selected(4, -1);
    selected[0] = 7;
    selected[1] = 0;
    selected[2] = 0;
    const std::vector<std::int32_t> count{2, 1};
    auto dq = to_device_bf16(q);
    auto dselected = to_device(selected);
    auto dcount = to_device(count);
    GuardedDeviceBuffer dout(static_cast<std::size_t>(256) * 24 * attention_width *
                             sizeof(std::uint16_t));
    GuardedDeviceBuffer attention_workspace(ops::qsa_selected_attention_workspace_bytes());
    Tensor qt(dq.p, DType::BF16, {256, 24, attention_width});
    Tensor st(dselected.p, DType::I32, {2, attention_width});
    Tensor ct(dcount.p, DType::I32, {attention_width});
    Tensor ot(dout.data(), DType::BF16, {256, 24, attention_width});
    Tensor attention_workspace_t(attention_workspace.data(), DType::U8,
                                 {static_cast<int>(attention_workspace.bytes())});
    ops::qsa_selected_attention(qt, st, ct, state_view, ot, attention_workspace_t, nullptr);
    cuda_synchronize();
    const auto actual = from_device_bf16(
        dout.data(), static_cast<std::size_t>(256) * 24 * attention_width);
    std::vector<double> expected(actual.size());
    constexpr int selected_host[2][2]{{7, 0}, {0, -1}};
    constexpr int selected_counts[2]{2, 1};
    for (int token = 0; token < attention_width; ++token) {
        for (int head = 0; head < 24; ++head) {
            const int kv_head = head / 12;
            double logits[2]{};
            for (int j = 0; j < selected_counts[token]; ++j) {
                for (int d = 0; d < 256; ++d) {
                    logits[j] += static_cast<double>(q[d + 256 * (head + 24 * token)]) *
                                 decode_cache(k_codes, k_scales, d,
                                              selected_host[token][j], kv_head, capacity);
                }
                logits[j] /= 16.0;
            }
            const double maximum = selected_counts[token] == 1
                                       ? logits[0]
                                       : std::max(logits[0], logits[1]);
            double denominator = 0.0;
            double probabilities[2]{};
            for (int j = 0; j < selected_counts[token]; ++j) {
                probabilities[j] = std::exp(logits[j] - maximum);
                denominator += probabilities[j];
            }
            for (int d = 0; d < 256; ++d) {
                double value = 0.0;
                for (int j = 0; j < selected_counts[token]; ++j) {
                    value += probabilities[j] / denominator *
                             decode_cache(v_codes, v_scales, d, selected_host[token][j],
                                          kv_head, capacity);
                }
                expected[d + 256 * (head + 24 * token)] = value;
            }
        }
    }
    failures += verify_pointwise("qsa batched attention decoded FP64 oracle", actual, expected,
                                 PointwiseCriterion{0.02, 0.005});
    failures += dout.verify_guards("qsa selected attention out");
    failures += attention_workspace.verify_guards("qsa selected attention workspace");
    return failures;
}

int append_alignment_and_state_validation_case() {
    constexpr int capacity = 1;
    constexpr int width = 1;
    constexpr std::uint8_t sentinel = 0xcdU;
    StateFixture state(capacity);

    std::vector<float> k(static_cast<std::size_t>(256) * 2 * width, 0.0F);
    std::vector<float> v(k.size(), 0.0F);
    std::vector<float> raw(128, 0.0F);
    auto dk = to_device_bf16(k);
    auto dv = to_device_bf16(v);
    auto draw = to_device_bf16(raw);
    auto dposition = to_device(std::vector<std::int32_t>{0, 0, 0});
    auto did = to_device(std::vector<std::int32_t>{0});
    Tensor k_t(dk.p, DType::BF16, {256, 2, width});
    Tensor v_t(dv.p, DType::BF16, {256, 2, width});
    Tensor raw_t(draw.p, DType::BF16, {128, width});
    Tensor position_t(dposition.p, DType::I32, {3, width});
    Tensor id_t(did.p, DType::I32, {width});

    const std::size_t kv_bytes = k.size() * sizeof(std::uint16_t);
    DeviceBuffer misaligned_k(kv_bytes + 16U);
    DeviceBuffer misaligned_v(kv_bytes + 16U);
    DeviceBuffer misaligned_raw(raw.size() * sizeof(std::uint16_t) + 2U);
    DeviceBuffer misaligned_position(3U * sizeof(std::int32_t) + 4U);
    DeviceBuffer misaligned_id(sizeof(std::int32_t) + 4U);
    misaligned_k.fill(sentinel);
    misaligned_v.fill(sentinel);
    misaligned_raw.fill(sentinel);
    misaligned_position.fill(sentinel);
    misaligned_id.fill(sentinel);
    Tensor bad_k(static_cast<std::byte*>(misaligned_k.p) + 2, DType::BF16,
                 {256, 2, width});
    Tensor bad_v(static_cast<std::byte*>(misaligned_v.p) + 2, DType::BF16,
                 {256, 2, width});
    Tensor bad_raw(static_cast<std::byte*>(misaligned_raw.p) + 1, DType::BF16, {128, width});
    Tensor bad_position(static_cast<std::byte*>(misaligned_position.p) + 2, DType::I32,
                        {3, width});
    Tensor bad_id(static_cast<std::byte*>(misaligned_id.p) + 2, DType::I32, {width});

    int failures = 0;
    failures += expect_invalid_argument("qsa append misaligned K", [&] {
        ops::qsa_state_append(bad_k, v_t, raw_t, position_t, id_t, state.view(), nullptr);
    });
    failures += expect_invalid_argument("qsa append misaligned V", [&] {
        ops::qsa_state_append(k_t, bad_v, raw_t, position_t, id_t, state.view(), nullptr);
    });
    failures += expect_invalid_argument("qsa append misaligned raw index key", [&] {
        ops::qsa_state_append(k_t, v_t, bad_raw, position_t, id_t, state.view(), nullptr);
    });
    failures += expect_invalid_argument("qsa append misaligned position", [&] {
        ops::qsa_state_append(k_t, v_t, raw_t, bad_position, id_t, state.view(), nullptr);
    });
    failures += expect_invalid_argument("qsa append misaligned id", [&] {
        ops::qsa_state_append(k_t, v_t, raw_t, position_t, bad_id, state.view(), nullptr);
    });

    DeviceBuffer bad_k_codes(state.k_codes.bytes() + 4U);
    DeviceBuffer bad_v_codes(state.v_codes.bytes() + 4U);
    DeviceBuffer bad_raw_state(state.raw_keys.bytes() + 2U);
    DeviceBuffer bad_position_state(state.positions.bytes() + 4U);
    bad_k_codes.fill(sentinel);
    bad_v_codes.fill(sentinel);
    bad_raw_state.fill(sentinel);
    bad_position_state.fill(sentinel);

    auto bad_state = state.view();
    bad_state.k = Tensor(static_cast<std::byte*>(bad_k_codes.p) + 1, DType::U8,
                               {128, capacity, 2});
    failures += expect_invalid_argument("qsa append misaligned state K codes", [&] {
        ops::qsa_state_append(k_t, v_t, raw_t, position_t, id_t, bad_state, nullptr);
    });
    bad_state = state.view();
    bad_state.v = Tensor(static_cast<std::byte*>(bad_v_codes.p) + 1, DType::U8,
                               {128, capacity, 2});
    failures += expect_invalid_argument("qsa append misaligned state V codes", [&] {
        ops::qsa_state_append(k_t, v_t, raw_t, position_t, id_t, bad_state, nullptr);
    });
    bad_state = state.view();
    bad_state.raw_index_keys = Tensor(static_cast<std::byte*>(bad_raw_state.p) + 1, DType::BF16,
                                      {128, capacity});
    failures += expect_invalid_argument("qsa append misaligned state raw index keys", [&] {
        ops::qsa_state_append(k_t, v_t, raw_t, position_t, id_t, bad_state, nullptr);
    });
    bad_state = state.view();
    bad_state.positions = Tensor(static_cast<std::byte*>(bad_position_state.p) + 2, DType::I32,
                                 {3, capacity});
    failures += expect_invalid_argument("qsa append misaligned state positions", [&] {
        ops::qsa_state_append(k_t, v_t, raw_t, position_t, id_t, bad_state, nullptr);
    });
    bad_state = state.view();
    bad_state.v = bad_state.k;
    failures += expect_invalid_argument("qsa append overlapping state planes", [&] {
        ops::qsa_state_append(k_t, v_t, raw_t, position_t, id_t, bad_state, nullptr);
    });
    bad_state = state.view();
    bad_state.k = Tensor(dk.p, DType::U8, {128, capacity, 2});
    failures += expect_invalid_argument("qsa append overlapping input/state", [&] {
        ops::qsa_state_append(k_t, v_t, raw_t, position_t, id_t, bad_state, nullptr);
    });

    auto dvisible = to_device(std::vector<std::int32_t>{0});
    auto doffsets = to_device(std::vector<std::int32_t>{0, 1});
    auto dquery_norm = to_device_f32(std::vector<float>(128, 1.0F));
    auto dkey_norm = to_device_f32(std::vector<float>(128, 1.0F));
    GuardedDeviceBuffer selected(static_cast<std::size_t>(ops::kQsaSelectedCapacity) *
                                 sizeof(std::int32_t));
    GuardedDeviceBuffer count(sizeof(std::int32_t));
    GuardedDeviceBuffer selector_workspace(ops::qsa_index_select_workspace_bytes(1));
    selected.fill(sentinel);
    count.fill(sentinel);
    selector_workspace.fill(sentinel);
    Tensor alias_query(selector_workspace.data(), DType::BF16, {128, 4, 1});
    Tensor visible_t(dvisible.p, DType::I32, {1});
    Tensor offsets_t(doffsets.p, DType::I32, {2});
    Tensor query_norm_t(dquery_norm.p, DType::FP32, {128});
    Tensor key_norm_t(dkey_norm.p, DType::FP32, {128});
    Tensor selected_t(selected.data(), DType::I32, {ops::kQsaSelectedCapacity, 1});
    Tensor count_t(count.data(), DType::I32, {1});
    Tensor selector_workspace_t(selector_workspace.data(), DType::U8,
                                {static_cast<int>(selector_workspace.bytes())});
    failures += expect_invalid_argument("qsa selector overlapping query/workspace", [&] {
        ops::qsa_index_select(alias_query, state.view(), id_t, visible_t, offsets_t, query_norm_t,
                              key_norm_t, selected_t, count_t, selector_workspace_t, nullptr);
    });

    cuda_synchronize();
    failures += verify_filled("qsa rejected state K codes", state.k_codes.data(),
                              state.k_codes.bytes(), sentinel);
    failures += verify_filled("qsa rejected state V codes", state.v_codes.data(),
                              state.v_codes.bytes(), sentinel);
    failures += verify_filled("qsa rejected state K scales", state.k_scales.data(),
                              state.k_scales.bytes(), sentinel);
    failures += verify_filled("qsa rejected state V scales", state.v_scales.data(),
                              state.v_scales.bytes(), sentinel);
    failures += verify_filled("qsa rejected state raw keys", state.raw_keys.data(),
                              state.raw_keys.bytes(), sentinel);
    failures += verify_filled("qsa rejected state positions", state.positions.data(),
                              state.positions.bytes(), sentinel);
    failures += verify_filled("qsa rejected misaligned K-code storage", bad_k_codes.p,
                              bad_k_codes.bytes, sentinel);
    failures += verify_filled("qsa rejected misaligned V-code storage", bad_v_codes.p,
                              bad_v_codes.bytes, sentinel);
    failures += verify_filled("qsa rejected misaligned raw-key storage", bad_raw_state.p,
                              bad_raw_state.bytes, sentinel);
    failures += verify_filled("qsa rejected misaligned position storage", bad_position_state.p,
                              bad_position_state.bytes, sentinel);
    failures += verify_filled("qsa rejected selector ids", selected.data(), selected.bytes(),
                              sentinel);
    failures += verify_filled("qsa rejected selector count", count.data(), count.bytes(),
                              sentinel);
    failures += verify_filled("qsa rejected selector workspace", selector_workspace.data(),
                              selector_workspace.bytes(), sentinel);
    return failures;
}

int selector_ceiling_and_permutation_case(ops::QsaKvFormat format = ops::QsaKvFormat::NVFP4G16) {
    constexpr int capacity = ops::kQsaMaximumTokens;
    constexpr int width    = 2;
    StateFixture state(capacity, format);
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

int selector_boundary_ties_case(ops::QsaKvFormat format = ops::QsaKvFormat::NVFP4G16) {
    constexpr std::array<int, 7> lengths{2047, 2048, 2049, 2050, 2051, 2052, 2053};
    constexpr int width = static_cast<int>(lengths.size());
    StateFixture state(ops::kQsaMaximumTokens, format);
    state.raw_keys.fill(0);
    state.positions.fill(0);

    std::vector<float> query(static_cast<std::size_t>(128) * 4 * width, 0.0F);
    std::vector<float> norm(128, 0.0F);
    std::vector<std::int32_t> query_ids;
    std::vector<std::int32_t> visible;
    std::vector<std::int32_t> offsets{0};
    for (int length : lengths) {
        query_ids.push_back(length - 1);
        for (int id = 0; id < length; ++id) { visible.push_back(id); }
        offsets.push_back(static_cast<std::int32_t>(visible.size()));
    }
    auto dquery = to_device_bf16(query);
    auto dnorm_query = to_device_f32(norm);
    auto dnorm_key = to_device_f32(norm);
    auto dquery_ids = to_device(query_ids);
    auto dvisible = to_device(visible);
    auto doffsets = to_device(offsets);
    GuardedDeviceBuffer dselected(static_cast<std::size_t>(ops::kQsaSelectedCapacity) * width *
                                  sizeof(std::int32_t));
    GuardedDeviceBuffer dcount(width * sizeof(std::int32_t));
    GuardedDeviceBuffer workspace(ops::qsa_index_select_workspace_bytes(width));
    Tensor query_t(dquery.p, DType::BF16, {128, 4, width});
    Tensor query_ids_t(dquery_ids.p, DType::I32, {width});
    Tensor visible_t(dvisible.p, DType::I32, {static_cast<int>(visible.size())});
    Tensor offsets_t(doffsets.p, DType::I32, {width + 1});
    Tensor query_norm_t(dnorm_query.p, DType::FP32, {128});
    Tensor key_norm_t(dnorm_key.p, DType::FP32, {128});
    Tensor selected_t(dselected.data(), DType::I32, {ops::kQsaSelectedCapacity, width});
    Tensor count_t(dcount.data(), DType::I32, {width});
    Tensor workspace_t(workspace.data(), DType::U8, {static_cast<int>(workspace.bytes())});
    auto state_view = state.view();
    ops::qsa_index_select(query_t, state_view, query_ids_t, visible_t, offsets_t, query_norm_t,
                          key_norm_t, selected_t, count_t, workspace_t, nullptr);
    cuda_synchronize();

    const auto actual_count = from_device<std::int32_t>(dcount.data(), width);
    const auto actual = from_device<std::int32_t>(
        dselected.data(), static_cast<std::size_t>(ops::kQsaSelectedCapacity) * width);
    int failures = 0;
    for (int column = 0; column < width; ++column) {
        const int length = lengths[static_cast<std::size_t>(column)];
        const int expected_count = length <= 2051 ? length : 2048 + (length & 3);
        if (actual_count[static_cast<std::size_t>(column)] != expected_count) { ++failures; }
        const std::size_t base = static_cast<std::size_t>(ops::kQsaSelectedCapacity) * column;
        for (int i = 0; i < expected_count; ++i) {
            const int expected_id = i < 2048 ? i : length - (length & 3) + (i - 2048);
            if (actual[base + static_cast<std::size_t>(i)] != expected_id) { ++failures; }
        }
        for (int i = expected_count; i < ops::kQsaSelectedCapacity; ++i) {
            if (actual[base + static_cast<std::size_t>(i)] != -1) { ++failures; }
        }
    }
    if (failures != 0) { std::cerr << "FAIL: qsa selector 2047..2053 tie boundaries\n"; }
    failures += dselected.verify_guards("qsa boundary selected ids");
    failures += dcount.verify_guards("qsa boundary selected count");
    failures += workspace.verify_guards("qsa boundary selector workspace");
    return failures;
}

int selector_route_switch_and_malformed_case(ops::QsaKvFormat format = ops::QsaKvFormat::NVFP4G16) {
    constexpr int short_length = ops::kQsaSelectedCapacity;
    constexpr int long_length = short_length + 1;
    StateFixture state(ops::kQsaMaximumTokens, format);
    state.raw_keys.fill(0);
    state.positions.fill(0);

    std::vector<float> query(static_cast<std::size_t>(128) * 4, 0.0F);
    std::vector<float> norm(128, 0.0F);
    std::vector<std::int32_t> visible(long_length);
    for (int id = 0; id < long_length; ++id) {
        visible[static_cast<std::size_t>(id)] = id;
    }
    const std::vector<std::int32_t> malformed{0, 2, 1, 3};
    auto dquery = to_device_bf16(query);
    auto dnorm_query = to_device_f32(norm);
    auto dnorm_key = to_device_f32(norm);
    auto dshort_query_id = to_device(std::vector<std::int32_t>{short_length - 1});
    auto dlong_query_id = to_device(std::vector<std::int32_t>{long_length - 1});
    auto dmalformed_query_id = to_device(std::vector<std::int32_t>{3});
    auto dvisible = to_device(visible);
    auto dmalformed = to_device(malformed);
    auto dshort_offsets = to_device(std::vector<std::int32_t>{0, short_length});
    auto dlong_offsets = to_device(std::vector<std::int32_t>{0, long_length});
    auto dmalformed_offsets = to_device(std::vector<std::int32_t>{0, 4});
    GuardedDeviceBuffer short_selected(
        static_cast<std::size_t>(ops::kQsaSelectedCapacity) * sizeof(std::int32_t));
    GuardedDeviceBuffer long_selected(
        static_cast<std::size_t>(ops::kQsaSelectedCapacity) * sizeof(std::int32_t));
    GuardedDeviceBuffer malformed_selected(
        static_cast<std::size_t>(ops::kQsaSelectedCapacity) * sizeof(std::int32_t));
    GuardedDeviceBuffer short_count(sizeof(std::int32_t));
    GuardedDeviceBuffer long_count(sizeof(std::int32_t));
    GuardedDeviceBuffer malformed_count(sizeof(std::int32_t));
    GuardedDeviceBuffer workspace(ops::qsa_index_select_workspace_bytes(1));

    Tensor query_t(dquery.p, DType::BF16, {128, 4, 1});
    Tensor query_norm_t(dnorm_query.p, DType::FP32, {128});
    Tensor key_norm_t(dnorm_key.p, DType::FP32, {128});
    Tensor short_query_id_t(dshort_query_id.p, DType::I32, {1});
    Tensor long_query_id_t(dlong_query_id.p, DType::I32, {1});
    Tensor malformed_query_id_t(dmalformed_query_id.p, DType::I32, {1});
    Tensor short_visible_t(dvisible.p, DType::I32, {short_length});
    Tensor long_visible_t(dvisible.p, DType::I32, {long_length});
    Tensor malformed_visible_t(dmalformed.p, DType::I32, {4});
    Tensor short_offsets_t(dshort_offsets.p, DType::I32, {2});
    Tensor long_offsets_t(dlong_offsets.p, DType::I32, {2});
    Tensor malformed_offsets_t(dmalformed_offsets.p, DType::I32, {2});
    Tensor short_selected_t(short_selected.data(), DType::I32,
                            {ops::kQsaSelectedCapacity, 1});
    Tensor long_selected_t(long_selected.data(), DType::I32,
                           {ops::kQsaSelectedCapacity, 1});
    Tensor malformed_selected_t(malformed_selected.data(), DType::I32,
                                {ops::kQsaSelectedCapacity, 1});
    Tensor short_count_t(short_count.data(), DType::I32, {1});
    Tensor long_count_t(long_count.data(), DType::I32, {1});
    Tensor malformed_count_t(malformed_count.data(), DType::I32, {1});
    Tensor workspace_t(workspace.data(), DType::U8, {static_cast<int>(workspace.bytes())});
    auto state_view = state.view();
    cudaStream_t stream = nullptr;
    cuda_check(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking), "cudaStreamCreate");
    ops::qsa_index_select(query_t, state_view, short_query_id_t, short_visible_t,
                          short_offsets_t, query_norm_t, key_norm_t, short_selected_t,
                          short_count_t, workspace_t, stream);
    ops::qsa_index_select(query_t, state_view, long_query_id_t, long_visible_t,
                          long_offsets_t, query_norm_t, key_norm_t, long_selected_t, long_count_t,
                          workspace_t, stream);
    ops::qsa_index_select(query_t, state_view, malformed_query_id_t, malformed_visible_t,
                          malformed_offsets_t, query_norm_t, key_norm_t, malformed_selected_t,
                          malformed_count_t, workspace_t, stream);
    cuda_synchronize(stream);
    cuda_check(cudaStreamDestroy(stream), "cudaStreamDestroy");

    const auto short_actual =
        from_device<std::int32_t>(short_selected.data(), ops::kQsaSelectedCapacity);
    const auto long_actual =
        from_device<std::int32_t>(long_selected.data(), ops::kQsaSelectedCapacity);
    const auto malformed_actual =
        from_device<std::int32_t>(malformed_selected.data(), ops::kQsaSelectedCapacity);
    const auto short_actual_count = from_device<std::int32_t>(short_count.data(), 1);
    const auto long_actual_count = from_device<std::int32_t>(long_count.data(), 1);
    const auto malformed_actual_count = from_device<std::int32_t>(malformed_count.data(), 1);
    int failures = verify_exact("qsa selector route-switch short count", short_actual_count,
                                std::vector<std::int32_t>{short_length});
    failures += verify_exact("qsa selector route-switch long count", long_actual_count,
                             std::vector<std::int32_t>{2048});
    failures += verify_exact("qsa selector malformed count", malformed_actual_count,
                             std::vector<std::int32_t>{0});
    for (int i = 0; i < ops::kQsaSelectedCapacity; ++i) {
        if (short_actual[static_cast<std::size_t>(i)] != i) { ++failures; }
        const int expected_long = i < 2048 ? i : -1;
        if (long_actual[static_cast<std::size_t>(i)] != expected_long) { ++failures; }
        if (malformed_actual[static_cast<std::size_t>(i)] != -1) { ++failures; }
    }
    if (failures != 0) {
        std::cerr << "FAIL: qsa selector short/long switch or malformed back-to-back\n";
    }
    failures += short_selected.verify_guards("qsa route-switch short selected");
    failures += long_selected.verify_guards("qsa route-switch long selected");
    failures += malformed_selected.verify_guards("qsa malformed selected");
    failures += short_count.verify_guards("qsa route-switch short count");
    failures += long_count.verify_guards("qsa route-switch long count");
    failures += malformed_count.verify_guards("qsa malformed count");
    failures += workspace.verify_guards("qsa route-switch workspace");
    return failures;
}

int selector_score_order_case(ops::QsaKvFormat format = ops::QsaKvFormat::NVFP4G16) {
    constexpr int capacity = 8;
    StateFixture state(capacity, format);
    std::vector<float> keys(static_cast<std::size_t>(128) * capacity);
    for (int id = 0; id < capacity; ++id) {
        const float value = id < 4 ? -1.0F : 1.0F;
        std::fill_n(keys.begin() + static_cast<std::size_t>(128) * id, 128, value);
    }
    round_to_bf16(keys);
    std::vector<std::uint16_t> key_bits(keys.size());
    for (std::size_t i = 0; i < keys.size(); ++i) { key_bits[i] = f32_to_bf16(keys[i]); }
    state.raw_keys.copy_from_host(key_bits.data(), key_bits.size() * sizeof(std::uint16_t));
    state.positions.fill(0);

    std::vector<float> query(static_cast<std::size_t>(128) * 4, 1.0F);
    round_to_bf16(query);
    const std::vector<float> norm(128, 1.0F);
    const std::vector<std::int32_t> query_ids{7};
    const std::vector<std::int32_t> visible{0, 1, 2, 3, 4, 5, 6, 7};
    const std::vector<std::int32_t> offsets{0, 8};
    auto dquery = to_device_bf16(query);
    auto dnorm_query = to_device_f32(norm);
    auto dnorm_key = to_device_f32(norm);
    auto dquery_ids = to_device(query_ids);
    auto dvisible = to_device(visible);
    auto doffsets = to_device(offsets);
    GuardedDeviceBuffer dselected(static_cast<std::size_t>(ops::kQsaSelectedCapacity) *
                                  sizeof(std::int32_t));
    GuardedDeviceBuffer dcount(sizeof(std::int32_t));
    GuardedDeviceBuffer workspace(ops::qsa_index_select_workspace_bytes(1));
    Tensor query_t(dquery.p, DType::BF16, {128, 4, 1});
    Tensor query_ids_t(dquery_ids.p, DType::I32, {1});
    Tensor visible_t(dvisible.p, DType::I32, {8});
    Tensor offsets_t(doffsets.p, DType::I32, {2});
    Tensor query_norm_t(dnorm_query.p, DType::FP32, {128});
    Tensor key_norm_t(dnorm_key.p, DType::FP32, {128});
    Tensor selected_t(dselected.data(), DType::I32, {ops::kQsaSelectedCapacity, 1});
    Tensor count_t(dcount.data(), DType::I32, {1});
    Tensor workspace_t(workspace.data(), DType::U8, {static_cast<int>(workspace.bytes())});
    auto state_view = state.view();
    ops::qsa_index_select(query_t, state_view, query_ids_t, visible_t, offsets_t, query_norm_t,
                          key_norm_t, selected_t, count_t, workspace_t, nullptr);
    cuda_synchronize();

    const auto actual_count = from_device<std::int32_t>(dcount.data(), 1);
    const auto actual = from_device<std::int32_t>(dselected.data(), ops::kQsaSelectedCapacity);
    int failures = verify_exact("qsa selector distinct-score count", actual_count,
                                std::vector<std::int32_t>{8});
    constexpr std::array<std::int32_t, 8> expected{4, 5, 6, 7, 0, 1, 2, 3};
    for (std::size_t i = 0; i < expected.size(); ++i) {
        if (actual[i] != expected[i]) { ++failures; }
    }
    for (int i = static_cast<int>(expected.size()); i < ops::kQsaSelectedCapacity; ++i) {
        if (actual[static_cast<std::size_t>(i)] != -1) { ++failures; }
    }
    if (failures != 0) { std::cerr << "FAIL: qsa selector distinct-score block ordering\n"; }
    failures += dselected.verify_guards("qsa score-order selected ids");
    failures += dcount.verify_guards("qsa score-order selected count");
    failures += workspace.verify_guards("qsa score-order selector workspace");
    return failures;
}

int selected_attention_capacity_case() {
    constexpr int capacity = ops::kQsaSelectedCapacity;
    StateFixture state(capacity);
    std::vector<std::uint8_t> k_codes(state.k_codes.bytes(), 0);
    std::vector<std::uint8_t> v_codes(state.v_codes.bytes(), 0);
    std::vector<std::uint8_t> k_scales(state.k_scales.bytes(), 0x38U);
    std::vector<std::uint8_t> v_scales(state.v_scales.bytes(), 0x38U);
    for (int head = 0; head < 2; ++head) {
        for (int id = 0; id < capacity; ++id) {
            for (int pair = 0; pair < 128; ++pair) {
                const int d0 = 2 * pair;
                const auto k_nibble = [&](int d) {
                    if (d >= 16) { return std::uint8_t{0}; }
                    const std::uint8_t magnitude =
                        static_cast<std::uint8_t>(1 + (id + 3 * head + d) % 7);
                    return static_cast<std::uint8_t>(magnitude |
                                                     (((id + head + d) & 4) != 0 ? 8U : 0U));
                };
                const auto v_nibble = [&](int d) {
                    const std::uint8_t magnitude =
                        static_cast<std::uint8_t>(1 + (id + 5 * head + d / 3) % 7);
                    return static_cast<std::uint8_t>(magnitude |
                                                     (((id / 5 + head + d) & 2) != 0 ? 8U : 0U));
                };
                k_codes[code_index(pair, id, head, capacity)] = static_cast<std::uint8_t>(
                    k_nibble(d0) | static_cast<std::uint8_t>(k_nibble(d0 + 1) << 4U));
                v_codes[code_index(pair, id, head, capacity)] = static_cast<std::uint8_t>(
                    v_nibble(d0) | static_cast<std::uint8_t>(v_nibble(d0 + 1) << 4U));
            }
            for (int group = 0; group < 16; ++group) {
                // Exact E4M3 encodings 0x30=0.5, 0x38=1.0, and 0x40=2.0 create
                // nonuniform key scores and value scales across both KV heads.
                constexpr std::uint8_t scales[]{0x30U, 0x38U, 0x40U};
                k_scales[scale_index(group, id, head, capacity)] =
                    scales[(id + group + head) % 3];
                v_scales[scale_index(group, id, head, capacity)] =
                    scales[(2 * id + group + head) % 3];
            }
        }
    }
    state.k_codes.copy_from_host(k_codes.data(), k_codes.size());
    state.v_codes.copy_from_host(v_codes.data(), v_codes.size());
    state.k_scales.copy_from_host(k_scales.data(), k_scales.size());
    state.v_scales.copy_from_host(v_scales.data(), v_scales.size());

    std::vector<float> query(static_cast<std::size_t>(256) * 24, 0.0F);
    for (int head = 0; head < 24; ++head) {
        for (int d = 0; d < 16; ++d) {
            query[d + 256 * head] = ((head + d) & 1) == 0 ? 0.125F : -0.09375F;
        }
    }
    round_to_bf16(query);
    std::vector<std::int32_t> selected(capacity);
    for (int rank = 0; rank < capacity; ++rank) {
        selected[static_cast<std::size_t>(rank)] = (37 * rank + 11) % capacity;
    }
    auto dquery = to_device_bf16(query);
    auto dselected = to_device(selected);
    auto dcount = to_device(std::vector<std::int32_t>{capacity});
    GuardedDeviceBuffer dout(static_cast<std::size_t>(256) * 24 * sizeof(std::uint16_t));
    GuardedDeviceBuffer attention_workspace(ops::qsa_selected_attention_workspace_bytes());
    Tensor query_t(dquery.p, DType::BF16, {256, 24, 1});
    Tensor selected_t(dselected.p, DType::I32, {capacity, 1});
    Tensor count_t(dcount.p, DType::I32, {1});
    Tensor out_t(dout.data(), DType::BF16, {256, 24, 1});
    Tensor attention_workspace_t(attention_workspace.data(), DType::U8,
                                 {static_cast<int>(attention_workspace.bytes())});
    auto state_view = state.view();
    ops::qsa_selected_attention(query_t, selected_t, count_t, state_view, out_t,
                                attention_workspace_t, nullptr);
    cuda_synchronize();
    const auto actual = from_device_bf16(dout.data(), static_cast<std::size_t>(256) * 24);
    std::vector<double> expected(actual.size());
    std::vector<double> scores(capacity);
    for (int head = 0; head < 24; ++head) {
        const int kv_head = head / 12;
        double maximum = -INFINITY;
        for (int rank = 0; rank < capacity; ++rank) {
            const int id = selected[static_cast<std::size_t>(rank)];
            double score = 0.0;
            for (int d = 0; d < 256; ++d) {
                score += static_cast<double>(query[d + 256 * head]) *
                         decode_cache(k_codes, k_scales, d, id, kv_head, capacity);
            }
            scores[static_cast<std::size_t>(rank)] = score / 16.0;
            maximum = std::max(maximum, scores[static_cast<std::size_t>(rank)]);
        }
        double denominator = 0.0;
        for (double& score : scores) {
            score = std::exp(score - maximum);
            denominator += score;
        }
        for (int d = 0; d < 256; ++d) {
            double value = 0.0;
            for (int rank = 0; rank < capacity; ++rank) {
                const int id = selected[static_cast<std::size_t>(rank)];
                value += scores[static_cast<std::size_t>(rank)] / denominator *
                         decode_cache(v_codes, v_scales, d, id, kv_head, capacity);
            }
            expected[d + 256 * head] = value;
        }
    }
    int failures = verify_pointwise("qsa selected attention nonuniform 2051-entry FP64 oracle",
                                    actual, expected, PointwiseCriterion{0.02, 0.005});
    Tensor aliased_workspace(out_t.data, DType::U8,
                             {static_cast<int>(attention_workspace.bytes())});
    try {
        ops::qsa_selected_attention(query_t, selected_t, count_t, state_view, out_t,
                                    aliased_workspace, nullptr);
        std::cerr << "FAIL: qsa selected attention accepted aliased workspace/output\n";
        ++failures;
    } catch (const std::invalid_argument&) {
    }
    failures += dout.verify_guards("qsa selected attention capacity out");
    failures += attention_workspace.verify_guards("qsa selected attention capacity workspace");
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

std::array<double, 256> oracle_attention_head(
    const std::array<double, 256>& query, std::span<const int> ids, int head, int capacity,
    const std::vector<std::uint8_t>& k_codes, const std::vector<std::uint8_t>& k_scales,
    const std::vector<std::uint8_t>& v_codes, const std::vector<std::uint8_t>& v_scales) {
    std::vector<double> logits(ids.size());
    double maximum = -INFINITY;
    for (std::size_t item = 0; item < ids.size(); ++item) {
        for (int d = 0; d < 256; ++d) {
            logits[item] += query[d] * decode_cache(k_codes, k_scales, d, ids[item], head, capacity);
        }
        logits[item] /= 16.0;
        maximum = std::max(maximum, logits[item]);
    }
    double denominator = 0;
    for (auto& value : logits) { value = std::exp(value - maximum); denominator += value; }
    std::array<double, 256> result{};
    for (int d = 0; d < 256; ++d) {
        for (std::size_t item = 0; item < ids.size(); ++item) {
            result[d] += logits[item] / denominator * decode_cache(v_codes, v_scales, d, ids[item], head, capacity);
        }
    }
    return result;
}

int bf16_cache_case() {
    constexpr int capacity = 2053;
    constexpr int width = 2052; // One invalid suffix, two untouched physical rows.
    StateFixture state(capacity, ops::QsaKvFormat::BF16);
    auto state_view = state.view();
    std::vector<std::uint16_t> keys(width * 512), values(keys.size()), raw(width * 128);
    std::vector<int> ids(width), positions(width * 3);
    for (int t = 0; t < width; ++t) {
        ids[t] = t == width - 1 ? -1 : (37 * t + 11) % 2051;
        for (int d = 0; d < 512; ++d) {
            keys[t * 512 + d] = f32_to_bf16(float((t * 13 + d * 7) % 127 - 63) / 256);
            values[t * 512 + d] = f32_to_bf16(float((t * 19 + d * 3) % 113 - 56) / 64);
        }
        // Storage must preserve signed zero and subnormal bits exactly, not re-quantize them.
        keys[t * 512] = (t & 1) ? 0x8000 : 1;
        values[t * 512 + 511] = (t & 1) ? 0x8001 : 0;
        for (int d = 0; d < 128; ++d) { raw[t * 128 + d] = f32_to_bf16(float(t + d) / 512); }
        for (int a = 0; a < 3; ++a) { positions[t * 3 + a] = t + a; }
    }
    auto dk=to_device(keys),dv=to_device(values),dr=to_device(raw),di=to_device(ids),dp=to_device(positions);
    Tensor kt(dk.p,DType::BF16,{256,2,width}), vt(dv.p,DType::BF16,{256,2,width});
    Tensor rt(dr.p,DType::BF16,{128,width}),it(di.p,DType::I32,{width}),pt(dp.p,DType::I32,{3,width});
    std::vector<std::uint16_t> expected_k(capacity*512,0xcdcd),expected_v(expected_k),
        expected_raw(capacity*128,0xcdcd);
    std::vector<int> expected_positions(capacity*3,static_cast<int>(0xcdcdcdcd));
    for(int t=0;t<width-1;++t) {
        for(int h=0;h<2;++h) for(int d=0;d<256;++d) {
            const int dest=d+256*(ids[t]+capacity*h),source=d+256*(h+2*t);
            expected_k[dest]=keys[source]; expected_v[dest]=values[source];
        }
        std::copy_n(raw.begin()+128*t,128,expected_raw.begin()+128*ids[t]);
        std::copy_n(positions.begin()+3*t,3,expected_positions.begin()+3*ids[t]);
    }
    int failures=0;
    for(bool chunked:{false,true}) {
        state.k_codes.fill(0xcd); state.v_codes.fill(0xcd);
        state.raw_keys.fill(0xcd); state.positions.fill(0xcd);
        for(int begin=0;begin<width;) {
            const int count=chunked ? std::min(17,width-begin) : width;
            const auto kpart=kt.slice(2,begin,count),vpart=vt.slice(2,begin,count),
                rpart=rt.slice(1,begin,count),ipart=it.slice(0,begin,count),ppart=pt.slice(1,begin,count);
            ops::qsa_state_append(kpart,vpart,rpart,ppart,ipart,state_view,nullptr);
            begin+=count;
        }
        cuda_synchronize();
        failures+=verify_exact("BF16 QSA complete K plane",from_device<std::uint16_t>(state.k_codes.data(),expected_k.size()),expected_k);
        failures+=verify_exact("BF16 QSA complete V plane",from_device<std::uint16_t>(state.v_codes.data(),expected_v.size()),expected_v);
        failures+=verify_exact("BF16 QSA complete index plane",from_device<std::uint16_t>(state.raw_keys.data(),expected_raw.size()),expected_raw);
        failures+=verify_exact("BF16 QSA complete positions",from_device<int>(state.positions.data(),expected_positions.size()),expected_positions);
    }
    const auto kc=from_device<std::uint8_t>(state.k_codes.data(),state.k_codes.bytes()),
        vc=from_device<std::uint8_t>(state.v_codes.data(),state.v_codes.bytes());
    const std::vector<std::uint8_t> no_scales;
    std::vector<float> query(256*24*2);
    fill_uniform(query,90271,-0.25F,0.25F); round_to_bf16(query);
    auto dq=to_device_bf16(query);
    GuardedDeviceBuffer output(query.size()*2),workspace(ops::qsa_selected_attention_workspace_bytes());
    Tensor wt(workspace.data(),DType::U8,{static_cast<int>(workspace.bytes())});
    for(int count:{0,1,63,64,65,127,128,129,2047,2048,2051}) for(int tokens:{1,2}) {
        const int bound=std::max(1,count);
        std::vector<int> selected(bound*tokens,-1),counts(tokens,count);
        for(int t=0;t<tokens;++t) for(int r=0;r<count;++r) { selected[t*bound+r]=(r*37+11)%2051; }
        auto ds=to_device(selected),dc=to_device(counts);
        Tensor qt(dq.p,DType::BF16,{256,24,tokens}),st(ds.p,DType::I32,{bound,tokens}),
            ct(dc.p,DType::I32,{tokens}),ot(output.data(),DType::BF16,{256,24,tokens});
        ops::qsa_selected_attention(qt,st,ct,state_view,ot,wt,nullptr);
        cuda_synchronize();
        std::vector<double> expected(256*24*tokens);
        for(int t=0;t<tokens;++t) for(int h=0;h<24;++h) {
            std::array<double,256> q{};
            std::copy_n(query.begin()+256*(h+24*t),256,q.begin());
            const auto result=oracle_attention_head(q,std::span(selected).subspan(t*bound,count),h/12,
                                                   capacity,kc,no_scales,vc,no_scales);
            std::copy(result.begin(),result.end(),expected.begin()+256*(h+24*t));
        }
        failures+=verify_reduction("BF16 QSA attention FP64 oracle",from_device_bf16(output.data(),expected.size()),
                                    expected,ReductionCriterion{1.0/256,1e-6,1.0/128});
        if(count==0) {
            failures+=verify_exact("BF16 QSA empty attention is exact zero",
                from_device<std::uint16_t>(output.data(),expected.size()),
                std::vector<std::uint16_t>(expected.size(),0));
        }
    }
    auto malformed=state_view;
    malformed.k_scales=Tensor(state.k_scales.data(),DType::FP8_E4M3FN,{16,capacity,2});
    failures+=expect_invalid_argument("BF16 state with scales",[&]{ops::qsa_state_append(kt,vt,rt,pt,it,malformed,nullptr);});
    malformed=state_view; malformed.v=malformed.k;
    failures+=expect_invalid_argument("BF16 state alias",[&]{ops::qsa_state_append(kt,vt,rt,pt,it,malformed,nullptr);});
    failures+=verify_exact("BF16 QSA attention preserves K",from_device<std::uint16_t>(state.k_codes.data(),expected_k.size()),expected_k);
    failures+=verify_exact("BF16 QSA attention preserves V",from_device<std::uint16_t>(state.v_codes.data(),expected_v.size()),expected_v);
    failures+=verify_exact("BF16 QSA attention preserves index",from_device<std::uint16_t>(state.raw_keys.data(),expected_raw.size()),expected_raw);
    failures+=verify_exact("BF16 QSA attention preserves positions",from_device<int>(state.positions.data(),expected_positions.size()),expected_positions);
    failures+=state.k_codes.verify_guards("BF16 QSA K"); failures+=state.v_codes.verify_guards("BF16 QSA V");
    failures+=state.raw_keys.verify_guards("BF16 QSA index"); failures+=state.positions.verify_guards("BF16 QSA positions");
    failures+=output.verify_guards("BF16 QSA attention"); failures+=workspace.verify_guards("BF16 QSA workspace");
    return failures;
}

int verifier_composite_real_shape_case(QType type = QType::GGML_Q5_K) {
    constexpr int capacity = 16;
    constexpr int width = 2;
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

    Q5MatrixFixture core_query_gate(12288, 2560, type);
    Q5MatrixFixture core_key(512, 2560, type);
    Q5MatrixFixture core_value(512, 2560, type);
    Q5MatrixFixture output(2560, 6144, type);
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

    std::vector<float> x(static_cast<std::size_t>(2560) * width, 0.0F);
    for (int token = 0; token < width; ++token) {
        x[2560 * token] = 1.0F;
        x[2560 * token + 1] = 0.5F;
        x[2560 * token + 2] = -1.0F;
        x[2560 * token + 3] = -0.5F;
    }
    round_to_bf16(x);
    const std::vector<std::int32_t> token_id{8, 9};
    const std::vector<std::int32_t> position{1, 2, 3, 4, 5, 6};
    std::vector<std::int32_t> visible_ids;
    for (int end = 8; end <= 9; ++end) {
        for (int id = 0; id <= end; ++id) { visible_ids.push_back(id); }
    }
    const std::vector<std::int32_t> visible_offsets{0, 9, 19};
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
    GuardedDeviceBuffer dselected(static_cast<std::size_t>(ops::kQsaSelectedCapacity) * width *
                                  sizeof(std::int32_t));
    GuardedDeviceBuffer dcount(width * sizeof(std::int32_t));
    GuardedDeviceBuffer dout(static_cast<std::size_t>(2560) * width * sizeof(std::uint16_t));
    GuardedDeviceBuffer workspace(ops::qsa_verifier_workspace_bytes(width, type, type, type, type));
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
    Tensor x_t(dx.p, DType::BF16, {2560, width});
    Tensor token_id_t(dtoken_id.p, DType::I32, {width});
    Tensor position_t(dposition.p, DType::I32, {3, width});
    Tensor visible_ids_t(dvisible_ids.p, DType::I32,
                         {static_cast<int>(visible_ids.size())});
    Tensor visible_offsets_t(dvisible_offsets.p, DType::I32, {width + 1});
    Tensor selected_t(dselected.data(), DType::I32, {ops::kQsaSelectedCapacity, width});
    Tensor count_t(dcount.data(), DType::I32, {width});
    Tensor out_t(dout.data(), DType::BF16, {2560, width});
    Tensor workspace_t(workspace.data(), DType::U8,
                       {static_cast<int>(workspace.bytes())});
    ops::qsa_verifier(x_t, token_id_t, position_t, visible_ids_t, visible_offsets_t,
                            weights, state_view, selected_t, count_t, out_t, workspace_t,
                            nullptr);
    cuda_synchronize();

    int failures = 0;
    const auto actual_count = from_device<std::int32_t>(dcount.data(), width);
    const auto selected = from_device<std::int32_t>(
        dselected.data(), static_cast<std::size_t>(ops::kQsaSelectedCapacity) * width);
    // Zero represented index gamma makes every complete-block score an exact tie. The stable
    // lower-rank order is also a strict witness that no implicit +1 is applied.
    const std::array<std::vector<std::int32_t>, width> expected_ids{{
        {0, 1, 2, 3, 4, 5, 6, 7, 8},
        {0, 1, 2, 3, 4, 5, 6, 7, 8, 9},
    }};
    failures += verify_exact("qsa verifier selected count", actual_count,
                             std::vector<std::int32_t>{9, 10});
    for (int token = 0; token < width; ++token) {
        const std::size_t base = static_cast<std::size_t>(ops::kQsaSelectedCapacity) * token;
        for (int i = 0; i < static_cast<int>(expected_ids[token].size()); ++i) {
            if (selected[base + i] != expected_ids[token][static_cast<std::size_t>(i)]) {
                ++failures;
            }
        }
        for (int i = static_cast<int>(expected_ids[token].size());
             i < ops::kQsaSelectedCapacity; ++i) {
            if (selected[base + i] != -1) { ++failures; }
        }
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
    for (int token = 0; token < width; ++token) {
        const int id = token_id[token];
        if (raw_keys[128 * id] != f32_to_bf16(1.0F)) { ++state_metadata_failures; }
        for (int d = 1; d < 128; ++d) {
            if (raw_keys[d + 128 * id] != f32_to_bf16(0.0F)) {
                ++state_metadata_failures;
            }
        }
        for (int axis = 0; axis < 3; ++axis) {
            if (stored_positions[axis + 3 * id] != position[axis + 3 * token]) {
                ++state_metadata_failures;
            }
        }
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
    const std::array<std::array<std::int32_t, 3>, width> position_arrays{{
        {1, 2, 3},
        {4, 5, 6},
    }};
    std::array<std::array<double, 256>, width> queries{};
    for (int token = 0; token < width; ++token) {
        queries[token] = oracle_core_norm_rope(raw_q, core_gamma, position_arrays[token]);
    }

    // The newly projected K/V must cross the represented cache boundary. Compare K against the
    // independent ideal norm/MRoPE result with an NVFP4 criterion and V against its represented
    // projection; the complete oracle below then consumes only exact-decoded cache values.
    int cache_roundtrip_failures = 0;
    double maximum_k_bound_ratio = 0.0;
    double maximum_v_bound_ratio = 0.0;
    for (int token = 0; token < width; ++token) {
        const int id = token_id[token];
        const auto& query = queries[token];
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
                    const double cached_k =
                        decode_cache(k_codes, k_scales, d, id, head, capacity);
                    const double k_error = std::abs(cached_k - query[d]);
                    const double k_bound = std::abs(query[d]) / 256.0 + 1.0e-4 + codec_bound;
                    maximum_k_bound_ratio = std::max(maximum_k_bound_ratio, k_error / k_bound);
                    if (k_error > k_bound) { ++cache_roundtrip_failures; }

                    const double expected_v = head == 0 ? 0.5 : -0.5;
                    const double cached_v =
                        decode_cache(v_codes, v_scales, d, id, head, capacity);
                    const double v_error = std::abs(cached_v - expected_v);
                    const double v_bound = (17.0 / 96.0) * std::abs(expected_v);
                    maximum_v_bound_ratio = std::max(maximum_v_bound_ratio,
                                                     v_error / v_bound);
                    if (v_error > v_bound) { ++cache_roundtrip_failures; }
                }
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
    std::vector<double> expected(static_cast<std::size_t>(2560) * width);
    for (int token = 0; token < width; ++token) {
        const auto& query = queries[token];
        const auto& token_ids = expected_ids[token];
        for (int head = 0; head < 10; ++head) {
            const int kv_head = head / 12;
            const auto attention = oracle_attention_head(query, token_ids, kv_head, capacity,
                                                         k_codes, k_scales, v_codes, v_scales);
            const double raw_gate = (head & 1) == 0 ? 1.0 : -1.0;
            const double gate = 1.0 / (1.0 + std::exp(-raw_gate));
            for (int d = 0; d < 256; ++d) {
                expected[d + 256 * (head + 10 * token)] = gate * attention[d];
            }
        }
    }
    const auto actual = from_device_bf16(dout.data(), static_cast<std::size_t>(2560) * width);
    failures += verify_pointwise("qsa batched verifier complete FP64 oracle", actual, expected,
                                 PointwiseCriterion{0.025, 0.02});

    // A reused MTP domain is an immutable input, not a fresh causal selection. Newly appended
    // private rows must update cache/index state but must NOT enter attention implicitly.
    std::vector<std::int32_t> frozen_ids(ops::kQsaSelectedCapacity * width, -1);
    const std::vector<std::int32_t> frozen_domain{0, 2, 5, 7};
    for (int token = 0; token < width; ++token) {
        std::copy(frozen_domain.begin(), frozen_domain.end(),
                  frozen_ids.begin() + token * ops::kQsaSelectedCapacity);
    }
    auto dfrozen = to_device(frozen_ids);
    auto dfrozen_count = to_device(std::vector<std::int32_t>{4, 4});
    Tensor frozen_t(dfrozen.p, DType::I32, {ops::kQsaSelectedCapacity, width});
    Tensor frozen_count_t(dfrozen_count.p, DType::I32, {width});
    for (int round = 0; round < 3; ++round) {
        const std::vector<std::int32_t> append{10 + 2 * round, 11 + 2 * round};
        const std::vector<std::int32_t> rope{31 + round, 7, 91, 103 + round, 12, 2};
        auto da = to_device(append);
        auto dp = to_device(rope);
        Tensor at(da.p, DType::I32, {width});
        Tensor pt(dp.p, DType::I32, {3, width});
        ops::qsa_verifier_selected(x_t, at, pt, weights, state_view, frozen_t,
                                   frozen_count_t, out_t, workspace_t, nullptr);
        cuda_synchronize();
        failures += verify_exact("qsa frozen selection is immutable",
            from_device<std::int32_t>(dfrozen.p, frozen_ids.size()), frozen_ids);
        failures += verify_exact("qsa frozen counts are immutable",
            from_device<std::int32_t>(dfrozen_count.p, width), std::vector<std::int32_t>{4, 4});
        for (int token = 0; token < width; ++token) {
            const std::array<std::int32_t, 3> p{rope[3 * token], rope[3 * token + 1], rope[3 * token + 2]};
            const auto query = oracle_core_norm_rope(raw_q, core_gamma, p);
            // Only old frozen rows are consumed, so the previously captured exact cache
            // remains the complete represented input to this independent oracle.
            for (int head = 0; head < 10; ++head) {
                const auto a = oracle_attention_head(query, frozen_domain, head / 12,
                    capacity, k_codes, k_scales, v_codes, v_scales);
                const double gate = 1.0 / (1.0 + std::exp((head & 1) == 0 ? -1.0 : 1.0));
                for (int d = 0; d < 256; ++d) {
                    expected[d + 256 * (head + 10 * token)] = gate * a[d];
                }
            }
        }
        failures += verify_pointwise("qsa frozen-domain complete FP64 oracle",
            from_device_bf16(dout.data(), 2560 * width), expected, PointwiseCriterion{0.025, 0.02});
        const auto positions_after = from_device<std::int32_t>(state.positions.data(), 3 * capacity);
        const auto index_after = from_device<std::uint16_t>(state.raw_keys.data(), 128 * capacity);
        for (int token = 0; token < width; ++token) {
            for (int axis = 0; axis < 3; ++axis) {
                if (positions_after[3 * append[token] + axis] != rope[3 * token + axis]) { ++failures; }
            }
            for (int d = 0; d < 128; ++d) {
                if (index_after[128 * append[token] + d] != f32_to_bf16(d == 0 ? 1.0F : 0.0F)) { ++failures; }
            }
        }
    }

    // Composite validation is a synchronous transaction boundary: neither an alias nor a
    // malformed state may enqueue even the leading projections. Fill every mutable external
    // range, then make workspace itself the input so any missed preflight is observable.
    constexpr std::uint8_t rejection_sentinel = 0x5aU;
    const auto state_k_before =
        from_device<std::uint8_t>(state.k_codes.data(), state.k_codes.bytes());
    const auto state_v_before =
        from_device<std::uint8_t>(state.v_codes.data(), state.v_codes.bytes());
    const auto state_ks_before =
        from_device<std::uint8_t>(state.k_scales.data(), state.k_scales.bytes());
    const auto state_vs_before =
        from_device<std::uint8_t>(state.v_scales.data(), state.v_scales.bytes());
    const auto state_raw_before =
        from_device<std::uint8_t>(state.raw_keys.data(), state.raw_keys.bytes());
    const auto state_position_before =
        from_device<std::uint8_t>(state.positions.data(), state.positions.bytes());
    const auto fill_rejection_outputs = [&] {
        dselected.fill(rejection_sentinel);
        dcount.fill(rejection_sentinel);
        dout.fill(rejection_sentinel);
        workspace.fill(rejection_sentinel);
    };
    const auto verify_rejection_outputs = [&] {
        int rejection_failures = 0;
        rejection_failures += verify_filled(
            "qsa verifier rejected selected ids", dselected.data(), dselected.bytes(),
            rejection_sentinel);
        rejection_failures += verify_filled("qsa verifier rejected selected count", dcount.data(),
                                            dcount.bytes(), rejection_sentinel);
        rejection_failures += verify_filled("qsa verifier rejected output", dout.data(),
                                            dout.bytes(), rejection_sentinel);
        rejection_failures += verify_filled("qsa verifier rejected workspace", workspace.data(),
                                            workspace.bytes(), rejection_sentinel);
        return rejection_failures;
    };

    fill_rejection_outputs();
    Tensor aliased_x(workspace.data(), DType::BF16, {2560, width});
    failures += expect_invalid_argument("qsa verifier aliased input/workspace", [&] {
        ops::qsa_verifier(aliased_x, token_id_t, position_t, visible_ids_t,
                                visible_offsets_t, weights, state_view, selected_t, count_t, out_t,
                                workspace_t, nullptr);
    });
    cuda_synchronize();
    failures += verify_rejection_outputs();

    fill_rejection_outputs();
    auto malformed_state = state_view;
    malformed_state.positions.ne[1] = capacity - 1;
    failures += expect_invalid_argument("qsa verifier malformed state", [&] {
        ops::qsa_verifier(x_t, token_id_t, position_t, visible_ids_t, visible_offsets_t,
                                weights, malformed_state, selected_t, count_t, out_t, workspace_t,
                                nullptr);
    });
    cuda_synchronize();
    failures += verify_rejection_outputs();
    failures += verify_exact("qsa verifier rejected state K codes",
                             from_device<std::uint8_t>(state.k_codes.data(), state.k_codes.bytes()),
                             state_k_before);
    failures += verify_exact("qsa verifier rejected state V codes",
                             from_device<std::uint8_t>(state.v_codes.data(), state.v_codes.bytes()),
                             state_v_before);
    failures += verify_exact(
        "qsa verifier rejected state K scales",
        from_device<std::uint8_t>(state.k_scales.data(), state.k_scales.bytes()), state_ks_before);
    failures += verify_exact(
        "qsa verifier rejected state V scales",
        from_device<std::uint8_t>(state.v_scales.data(), state.v_scales.bytes()), state_vs_before);
    failures += verify_exact(
        "qsa verifier rejected state raw keys",
        from_device<std::uint8_t>(state.raw_keys.data(), state.raw_keys.bytes()), state_raw_before);
    failures += verify_exact(
        "qsa verifier rejected state positions",
        from_device<std::uint8_t>(state.positions.data(), state.positions.bytes()),
        state_position_before);

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

std::vector<double> native_qsa_output_oracle(const quantized_weight::PackedWeight& output,
    const std::vector<double>& raw_q, const std::array<float,256>& gamma,
    const std::array<int,3>& position, std::span<const int> causal, int capacity,
    const std::vector<std::uint8_t>& kc,const std::vector<std::uint8_t>& ks,
    const std::vector<std::uint8_t>& vc,const std::vector<std::uint8_t>& vs) {
    std::vector<double> gated(6144);
    for(int head=0;head<24;++head) {
        std::array<double,256> query{};
        std::copy_n(raw_q.begin()+head*512,256,query.begin());
        query=oracle_core_norm_rope(query,gamma,position);
        const auto attention=oracle_attention_head(query,causal,head/12,capacity,kc,ks,vc,vs);
        for(int d=0;d<256;++d) {
            gated[head*256+d]=attention[d]/(1+std::exp(-raw_q[head*512+256+d]));
        }
    }
    return native_projection_oracle(output,gated);
}

// Independent exact nearest-code enumeration for the registered NVFP4-G16 state codec.
// BF16 append input is public; FP32 division and E4M3/E2M1 RNE are codec boundaries.
void oracle_encode_cache(std::span<const double> values,int token,int head,int capacity,
    std::vector<std::uint8_t>& codes,std::vector<std::uint8_t>& scales) {
    if (scales.empty()) {
        for (int d = 0; d < 256; ++d) {
            const auto bits = f32_to_bf16(qwen4_sequence::represented(values[d]));
            const auto offset = 2ULL * (d + 256ULL * (token + static_cast<std::size_t>(capacity) * head));
            codes[offset] = bits & 255;
            codes[offset + 1] = bits >> 8;
        }
        return;
    }
    const auto nearest=[](double magnitude,int count,auto decode) {
        int best=0;
        double error=std::numeric_limits<double>::infinity();
        for(int code=0;code<count;++code) {
            const double candidate=std::abs(magnitude-decode(code));
            if(candidate<error || (candidate==error && (code&1)==0)) { best=code;error=candidate; }
        }
        return best;
    };
    for(int group=0;group<16;++group) {
        std::array<float,16> represented{};
        float maximum=0;
        for(int d=0;d<16;++d) {
            represented[d]=qwen4_sequence::represented(values[group*16+d]);
            maximum=std::max(maximum,std::abs(represented[d]));
        }
        const auto scale=nearest(static_cast<float>(maximum/6.F),127,decode_e4m3);
        scales[scale_index(group,token,head,capacity)]=scale;
        const float divisor=decode_e4m3(scale);
        for(int pair=0;pair<8;++pair) {
            unsigned packed=0;
            for(int item=0;item<2;++item) {
                const float value=represented[pair*2+item];
                const float normalized=divisor==0 ? 0 : value/divisor;
                const auto magnitude=nearest(std::abs(normalized),8,decode_e2m1);
                packed|=(magnitude|(std::signbit(normalized)?8:0))<<(item*4);
            }
            codes[code_index(group*8+pair,token,head,capacity)]=packed;
        }
    }
}

int native_qsa_case(const std::string& path, int width, bool partitioned = false,
    const qwen4_sequence::Result* sequence_input=nullptr,qwen4_sequence::Result* sequence_output=nullptr,
    ops::QsaKvFormat cache_format=ops::QsaKvFormat::BF16,const std::string& fp8_path={},
    std::vector<int>* selection_trace=nullptr) {
    qwen4_native::Bf16Source source(path, 3);
    const std::string p = "self_attn.";
    const auto index_bits = source.bits(p + "indexer.index_qk_proj.weight", {640,2560});
    const auto iq = qwen4_native::bf16_matrix(std::span(index_bits).first(512*2560), 512, 2560);
    const auto ik = qwen4_native::bf16_matrix(std::span(index_bits).subspan(512*2560), 128, 2560);
    auto q = qwen4_native::bf16_matrix(source.bits(p + "q_proj.weight", {12288,2560}), 12288,2560);
    auto k = qwen4_native::bf16_matrix(source.bits(p + "k_proj.weight", {512,2560}), 512,2560);
    auto v = qwen4_native::bf16_matrix(source.bits(p + "v_proj.weight", {512,2560}), 512,2560);
    auto o = qwen4_native::bf16_matrix(source.bits(p + "o_proj.weight", {2560,6144}), 2560,6144);
    if(!fp8_path.empty()) {
        artifact::Reader reader(fp8_path);
        if(reader.identity()!=artifact::ArtifactIdentity{
            "qwen4/native-fp8-projection-qualification","senfu-fp8-source"}) {
            throw std::invalid_argument("wrong QSA calibrated source");
        }
        const auto replace=[&](quantized_weight::PackedWeight& weight,const char* role) {
            const auto* object=reader.find(std::string("model.language_model.layers.3.self_attn.")+role+".weight");
            const auto* descriptor=object?std::get_if<artifact::TensorDescriptor>(object):nullptr;
            if(!descriptor || descriptor->format!=artifact::NumericFormat::FP8_E4M3FN_TENSOR_F32M ||
                descriptor->layout!=artifact::StorageLayout::TensorCalibratedV1 ||
                descriptor->shape!=std::vector<std::uint64_t>{static_cast<unsigned>(weight.weight.n),static_cast<unsigned>(weight.weight.k)}) {
                throw std::invalid_argument("invalid calibrated QSA source role");
            }
            const auto raw=reader.payload(*descriptor).data;
            weight.payload.resize(raw.size());std::memcpy(weight.payload.data(),raw.data(),raw.size());
            weight.code_plane_bytes=std::uint64_t(weight.weight.n)*weight.weight.k;
            weight.scale_plane_offset=weight.code_plane_bytes;weight.scale_plane_bytes=8;
            weight.weight.qtype=QType::FP8_E4M3FN_TENSOR_F32M;
            weight.weight.layout=QuantLayout::TensorCalibrated;weight.weight.scale_dtype=DType::FP32;
            weight.weight.payload_bytes=raw.size();weight.weight.scale_ne[0]=2;weight.weight.scale_nb[0]=4;
            for(int axis=1;axis<4;++axis) { weight.weight.scale_nb[axis]=8; }
        };
        replace(q,"q_proj");replace(k,"k_proj");replace(v,"v_proj");replace(o,"o_proj");
    }
    auto diq=to_device(iq.payload), dik=to_device(ik.payload), dq=to_device(q.payload),
         dk=to_device(k.payload), dv=to_device(v.payload), doutw=to_device(o.payload);
    auto iqn=source.values(p + "indexer.q_layernorm.weight", {128});
    auto ikn=source.values(p + "indexer.k_layernorm.weight", {128});
    auto qn=source.values(p + "q_norm.weight", {256});
    auto kn=source.values(p + "k_norm.weight", {256});
    for (auto* norm : {&iqn,&ikn,&qn,&kn}) { for (auto& value : *norm) { value += 1.0F; } }
    auto diqn=to_device_f32(iqn), dikn=to_device_f32(ikn), dqn=to_device_f32(qn), dkn=to_device_f32(kn);
    ops::QsaVerifierWeights weights{iq.device_weight(diq.p),ik.device_weight(dik.p),q.device_weight(dq.p),
        k.device_weight(dk.p),v.device_weight(dv.p),o.device_weight(doutw.p),
        Tensor(diqn.p,DType::FP32,{128}),Tensor(dikn.p,DType::FP32,{128}),
        Tensor(dqn.p,DType::FP32,{256}),Tensor(dkn.p,DType::FP32,{256})};
    const int capacity=std::max(16,((width+3)/4)*4);
    StateFixture state(capacity, cache_format);
    auto state_view=state.view();
    std::vector<float> x(width*2560);
    fill_uniform(x, 30303U, -0.2F, 0.2F); round_to_bf16(x);
    if(sequence_input) { x=sequence_input->actual; }
    std::vector<int> ids(width),positions(width*3),visible,offsets{0};
    for(int t=0;t<width;++t) {
        ids[t]=t;
        for(int axis=0;axis<3;++axis) { positions[t*3+axis]=t+axis; }
        for(int id=0;id<=t;++id) { visible.push_back(id); }
        offsets.push_back(visible.size());
    }
    auto dx=to_device_bf16(x), dids=to_device(ids), dpos=to_device(positions),
         dvisible=to_device(visible), doffsets=to_device(offsets);
    GuardedDeviceBuffer out(width*2560*2), selected(width*ops::kQsaSelectedCapacity*4), counts(width*4);
    GuardedDeviceBuffer workspace(ops::qsa_verifier_workspace_bytes(width,q.weight.qtype,
        k.weight.qtype,v.weight.qtype,o.weight.qtype));
    Tensor xt(dx.p,DType::BF16,{2560,width}), it(dids.p,DType::I32,{width}),pt(dpos.p,DType::I32,{3,width});
    Tensor vt(dvisible.p,DType::I32,{static_cast<int>(visible.size())}),ot(doffsets.p,DType::I32,{width+1});
    Tensor st(selected.data(),DType::I32,{ops::kQsaSelectedCapacity,width}),ct(counts.data(),DType::I32,{width});
    Tensor yt(out.data(),DType::BF16,{2560,width}),wt(workspace.data(),DType::U8,{static_cast<int>(workspace.bytes())});
    if (!partitioned) {
        ops::qsa_verifier(xt,it,pt,vt,ot,weights,state_view,st,ct,yt,wt,nullptr);
    } else {
        for (int start = 0; start < width;) {
            const int chunk = start == 0 ? (sequence_input || !fp8_path.empty() ? width-1 : 2) : width - start;
            const int begin = offsets[start], end = offsets[start + chunk];
            std::vector<int> local_offsets(offsets.begin()+start,offsets.begin()+start+chunk+1);
            for (auto& offset : local_offsets) { offset -= begin; }
            auto local_device = to_device(local_offsets);
            Tensor chunk_offsets(local_device.p,DType::I32,{chunk+1});
            auto chunk_x=xt.slice(1,start,chunk), chunk_ids=it.slice(0,start,chunk),
                 chunk_positions=pt.slice(1,start,chunk), chunk_visible=vt.slice(0,begin,end-begin),
                 chunk_selected=st.slice(1,start,chunk), chunk_counts=ct.slice(0,start,chunk),
                 chunk_out=yt.slice(1,start,chunk);
            ops::qsa_verifier(chunk_x,chunk_ids,chunk_positions,chunk_visible,chunk_offsets,
                weights,state_view,chunk_selected,chunk_counts,chunk_out,wt,nullptr);
            cuda_synchronize(); // local CSR storage remains owned through completion.
            start += chunk;
        }
    }
    cuda_synchronize();
    auto kc=from_device<std::uint8_t>(state.k_codes.data(),state.k_codes.bytes());
    auto ks=from_device<std::uint8_t>(state.k_scales.data(),state.k_scales.bytes());
    auto vc=from_device<std::uint8_t>(state.v_codes.data(),state.v_codes.bytes());
    auto vs=from_device<std::uint8_t>(state.v_scales.data(),state.v_scales.bytes());
    if (cache_format == ops::QsaKvFormat::BF16) { ks.clear(); vs.clear(); }
    auto raw_keys=from_device_bf16(state.raw_keys.data(),capacity*128);
    auto actual=from_device_bf16(out.data(),width*2560);
    auto actual_ids=from_device<int>(selected.data(),width*ops::kQsaSelectedCapacity);
    if(selection_trace) { *selection_trace=actual_ids; }
    auto actual_counts=from_device<int>(counts.data(),width);
    int failures=verify_exact("native QSA positions",from_device<int>(state.positions.data(),width*3),positions);
    std::array<float,256> qgamma{},kgamma{};
    std::copy(qn.begin(),qn.end(),qgamma.begin()); std::copy(kn.begin(),kn.end(),kgamma.begin());
    const bool diagnose=sequence_output && cache_format==ops::QsaKvFormat::NVFP4G16;
    std::vector<std::uint8_t> same_kc(kc.size()),same_ks(ks.size()),same_vc(vc.size()),same_vs(vs.size());
    std::vector<double> kernel_formula,same_formula;
    double key_error2=0,key_norm2=0,value_error2=0,value_norm2=0;
    for(int t=0;t<width;++t) {
        // All visible IDs fit below the budget; complete blocks may be ranked in another
        // order at T17. Check exact membership and the complete untouched padding suffix.
        if(actual_counts[t]!=t+1) { ++failures; }
        const auto begin=actual_ids.begin()+t*ops::kQsaSelectedCapacity;
        std::vector<int> membership(begin,begin+t+1);
        std::sort(membership.begin(),membership.end());
        for(int item=0;item<=t;++item) { if(membership[item]!=item) { ++failures; } }
        for(int item=t+1;item<ops::kQsaSelectedCapacity;++item) {
            if(actual_ids[t*ops::kQsaSelectedCapacity+item]!=-1) { ++failures; }
        }
        std::vector<double> input(x.begin()+t*2560,x.begin()+(t+1)*2560);
        const auto raw_q=native_projection_oracle(q,input),raw_k=native_projection_oracle(k,input),
                   raw_v=native_projection_oracle(v,input),raw_ik=native_projection_oracle(ik,input);
        failures+=verify_reduction("native QSA index projection",std::span(raw_keys).subspan(t*128,128),
                                    raw_ik,ReductionCriterion{1.0/256.0,1e-6,1.0/128.0});
        const std::array<int,3> position{positions[t*3],positions[t*3+1],positions[t*3+2]};
        for(int head=0;head<2;++head) {
            std::array<double,256> key{};
            std::copy_n(raw_k.begin()+head*256,256,key.begin());
            key=oracle_core_norm_rope(key,kgamma,position);
            if(diagnose) {
                oracle_encode_cache(key,t,head,capacity,same_kc,same_ks);
                oracle_encode_cache(std::span(raw_v).subspan(head*256,256),t,head,capacity,same_vc,same_vs);
            }
            for(int group=0;group<16;++group) {
                double km=0,vm=0;
                for(int d=group*16;d<(group+1)*16;++d) {
                    km=std::max(km,std::abs(key[d])); vm=std::max(vm,std::abs(raw_v[head*256+d]));
                }
                for(int d=group*16;d<(group+1)*16;++d) {
                    const double codec_error=cache_format==ops::QsaKvFormat::BF16 ? 0 : 17.0/96;
                    const double kb=km/256+1e-4+codec_error*(km+km/256+1e-4);
                    const double vb=vm/256+1e-4+codec_error*(vm+vm/256+1e-4);
                    if(std::abs(decode_cache(kc,ks,d,t,head,capacity)-key[d])>kb ||
                       std::abs(decode_cache(vc,vs,d,t,head,capacity)-raw_v[head*256+d])>vb) { ++failures; }
                    const double ke=decode_cache(kc,ks,d,t,head,capacity)-key[d],
                        ve=decode_cache(vc,vs,d,t,head,capacity)-raw_v[head*256+d];
                    key_error2+=ke*ke;key_norm2+=key[d]*key[d];
                    value_error2+=ve*ve;value_norm2+=raw_v[head*256+d]*raw_v[head*256+d];
                }
            }
        }
        std::vector<int> causal(ids.begin(),ids.begin()+t+1);
        const auto expected=native_qsa_output_oracle(o,raw_q,qgamma,position,causal,capacity,kc,ks,vc,vs);
        failures+=verify_reduction("native source QSA complete FP64 formula",std::span(actual).subspan(t*2560,2560),
                                    expected,ReductionCriterion{0.02,2.5e-4,0.02});
        if(diagnose || !fp8_path.empty()) {
            kernel_formula.insert(kernel_formula.end(),expected.begin(),expected.end());
        }
        if(diagnose) {
            const auto pure=native_qsa_output_oracle(o,raw_q,qgamma,position,causal,capacity,
                same_kc,same_ks,same_vc,same_vs);
            same_formula.insert(same_formula.end(),pure.begin(),pure.end());
        }
    }
    if(!fp8_path.empty()) {
        double error2=0,norm2=0,maximum=0;
        for(std::size_t i=0;i<actual.size();++i) {
            const double error=actual[i]-kernel_formula[i];error2+=error*error;
            norm2+=kernel_formula[i]*kernel_formula[i];maximum=std::max(maximum,std::abs(error));
        }
        std::cout<<"QSA_CALIBRATED_ERRORS local_output_relative_l2="<<std::sqrt(error2/norm2)
            <<" max_abs="<<maximum<<" K_state_relative_l2="<<std::sqrt(key_error2/key_norm2)
            <<" V_state_relative_l2="<<std::sqrt(value_error2/value_norm2)<<'\n';
    }
    if(sequence_output) {
        std::vector<std::uint8_t> ideal_kc(kc.size()),ideal_ks(ks.size()),ideal_vc(vc.size()),ideal_vs(vs.size());
        sequence_output->actual.assign(actual.begin(),actual.end());
        sequence_output->discrete_ids.assign(actual_ids.begin(),actual_ids.end());
        std::vector<float> computed_reference(actual.size());
        const auto& reference_input=sequence_input->reference;
        for(int t=0;t<width;++t) {
            std::vector<double> input(reference_input.begin()+t*2560,
                reference_input.begin()+(t+1)*2560);
            const auto raw_q=native_projection_oracle(q,input),raw_k=native_projection_oracle(k,input),
                raw_v=native_projection_oracle(v,input);
            const std::array<int,3> position{positions[t*3],positions[t*3+1],positions[t*3+2]};
            for(int head=0;head<2;++head) {
                std::array<double,256> key{};
                std::copy_n(raw_k.begin()+head*256,256,key.begin());
                key=oracle_core_norm_rope(key,kgamma,position);
                oracle_encode_cache(key,t,head,capacity,ideal_kc,ideal_ks);
                oracle_encode_cache(std::span(raw_v).subspan(head*256,256),t,head,capacity,ideal_vc,ideal_vs);
            }
            const std::vector<int> causal(ids.begin(),ids.begin()+t+1);
            const auto expected=native_qsa_output_oracle(o,raw_q,qgamma,position,causal,capacity,
                ideal_kc,ideal_ks,ideal_vc,ideal_vs);
            const auto represented=qwen4_sequence::represented(expected);
            std::copy(represented.begin(),represented.end(),computed_reference.begin()+t*2560);
        }
        sequence_output->reference=std::move(computed_reference);
        if(diagnose) {
            // Public BF16-state cross-check isolates codec input rounding. This is a
            // diagnostic implementation comparison, not the independent formula above.
            StateFixture public_bf16(capacity,ops::QsaKvFormat::BF16);
            auto public_view=public_bf16.view();
            for(int start=0;start<width;) {
                const int chunk=partitioned ? (start==0 ? width-1:1):width;
                const int begin=offsets[start],end=offsets[start+chunk];
                std::vector<int> local(offsets.begin()+start,offsets.begin()+start+chunk+1);
                for(auto& offset:local) { offset-=begin; }
                auto local_device=to_device(local);
                auto xx=xt.slice(1,start,chunk),ii=it.slice(0,start,chunk),pp=pt.slice(1,start,chunk),
                    vv=vt.slice(0,begin,end-begin),ss=st.slice(1,start,chunk),cc=ct.slice(0,start,chunk),
                    yy=yt.slice(1,start,chunk);
                Tensor oo(local_device.p,DType::I32,{chunk+1});
                ops::qsa_verifier(xx,ii,pp,vv,oo,weights,public_view,ss,cc,yy,wt,nullptr);
                cuda_synchronize();start+=chunk;
            }
            const auto bk=from_device<std::uint8_t>(public_bf16.k_codes.data(),public_bf16.k_codes.bytes()),
                bv=from_device<std::uint8_t>(public_bf16.v_codes.data(),public_bf16.v_codes.bytes());
            std::vector<std::uint8_t> encoded_k(kc.size()),encoded_ks(ks.size()),
                encoded_v(vc.size()),encoded_vs(vs.size()),empty;
            for(int t=0;t<width;++t) for(int h=0;h<2;++h) {
                std::array<double,256> key{},value{};
                for(int d=0;d<256;++d) {
                    key[d]=decode_cache(bk,empty,d,t,h,capacity);
                    value[d]=decode_cache(bv,empty,d,t,h,capacity);
                }
                oracle_encode_cache(key,t,h,capacity,encoded_k,encoded_ks);
                oracle_encode_cache(value,t,h,capacity,encoded_v,encoded_vs);
            }
            const auto compare_cache=[&](const char* label,const auto& ac,const auto& as,
                const auto& bc,const auto& bs) {
                int codes=0,scales=0;double error2=0,norm2=0,maximum=0;
                for(int h=0;h<2;++h) for(int t=0;t<width;++t) {
                    for(int g=0;g<16;++g) { scales+=as[scale_index(g,t,h,capacity)]!=bs[scale_index(g,t,h,capacity)]; }
                    for(int d=0;d<256;++d) {
                        const auto offset=code_index(d/2,t,h,capacity);
                        const int shift=4*(d&1);
                        codes+=((ac[offset]>>shift)&15)!=((bc[offset]>>shift)&15);
                        const double a=decode_cache(ac,as,d,t,h,capacity),b=decode_cache(bc,bs,d,t,h,capacity);
                        error2+=(a-b)*(a-b);norm2+=b*b;maximum=std::max(maximum,std::abs(a-b));
                    }
                }
                std::cout<<"NVFP4_STATE_DIAG T="<<width<<" chunk="<<partitioned<<" comparison="<<label
                    <<" code_differences="<<codes<<"/"<<width*512<<" scale_differences="<<scales<<"/"<<width*32
                    <<" decoded_relative_l2="<<std::sqrt(error2/norm2)<<" max_abs="<<maximum<<'\n';
                return codes+scales;
            };
            failures+=compare_cache("gpu_vs_publicBF16_codec_K",kc,ks,encoded_k,encoded_ks)!=0;
            failures+=compare_cache("gpu_vs_publicBF16_codec_V",vc,vs,encoded_v,encoded_vs)!=0;
            compare_cache("gpu_vs_sameInput_formula_K",kc,ks,same_kc,same_ks);
            compare_cache("gpu_vs_sameInput_formula_V",vc,vs,same_vc,same_vs);
            compare_cache("sameInput_vs_propagated_formula_K",same_kc,same_ks,ideal_kc,ideal_ks);
            compare_cache("sameInput_vs_propagated_formula_V",same_vc,same_vs,ideal_vc,ideal_vs);
            const auto report=[&](const char* label,const auto& a,const auto& b) {
                double e=0,n=0,m=0;
                for(std::size_t i=0;i<a.size();++i) { const double d=double(a[i])-double(b[i]);e+=d*d;n+=double(b[i])*b[i];m=std::max(m,std::abs(d)); }
                std::cout<<"NVFP4_OUTPUT_DIAG T="<<width<<" chunk="<<partitioned<<" comparison="<<label
                    <<" relative_l2="<<std::sqrt(e/n)<<" max_abs="<<m<<'\n';
            };
            report("GPU_vs_actualCache_formula",actual,kernel_formula);
            report("actualCache_vs_sameInput_formula",kernel_formula,same_formula);
            report("sameInput_vs_propagated_formula",same_formula,sequence_output->reference);
        }
    }
    failures+=out.verify_guards("native QSA output");
    failures+=selected.verify_guards("native QSA selected IDs");
    failures+=counts.verify_guards("native QSA selected counts");
    failures+=workspace.verify_guards("native QSA workspace");
    failures+=state.k_codes.verify_guards("native QSA K codes");
    failures+=state.v_codes.verify_guards("native QSA V codes");
    failures+=state.k_scales.verify_guards("native QSA K scales");
    failures+=state.v_scales.verify_guards("native QSA V scales");
    failures+=state.raw_keys.verify_guards("native QSA raw index keys");
    failures+=state.positions.verify_guards("native QSA positions");
    return failures;
}

// Assessment hypothesis, not a registered runtime codec: independent nearest-code
// enumeration, per-token/head FP32 scale, explicit FP32 reconstruction then BF16.
std::uint8_t assessment_e4m3(float value) {
    int best=0; double error=INFINITY;
    for(int code=0;code<=126;++code) {
        const double distance=std::abs(double(std::abs(value))-decode_e4m3(code));
        if(distance<error || (distance==error && !(code&1))) { best=code;error=distance; }
    }
    return best | (std::signbit(value)?128:0);
}

int native_fp8_kv_assessment(const std::string& path) {
    constexpr int width=129,capacity=129;
    int failures=0;
    for(const auto [value,code]:std::array<std::pair<float,int>,9>{{
        {0.F,0},{-0.F,128},{1.F,56},{1.0625F,56},{1.1875F,58},
        {448.F,126},{1000.F,126},{-1000.F,254},{0.0009765625F,0}}}) {
        if(assessment_e4m3(value)!=code) { ++failures; }
    }
    qwen4_native::Bf16Source source(path,3);
    const auto q=qwen4_native::bf16_matrix(source.bits("self_attn.q_proj.weight",{12288,2560}),12288,2560);
    const auto k=qwen4_native::bf16_matrix(source.bits("self_attn.k_proj.weight",{512,2560}),512,2560);
    const auto v=qwen4_native::bf16_matrix(source.bits("self_attn.v_proj.weight",{512,2560}),512,2560);
    const auto o=qwen4_native::bf16_matrix(source.bits("self_attn.o_proj.weight",{2560,6144}),2560,6144);
    const auto qn=source.values("self_attn.q_norm.weight",{256}),kn=source.values("self_attn.k_norm.weight",{256});
    std::array<float,256> qgamma{},kgamma{};
    for(int d=0;d<256;++d) { qgamma[d]=1.F+qn[d];kgamma[d]=1.F+kn[d]; }
    std::vector<float> x(width*2560),queries(width*6144);
    fill_uniform(x,30303U,-.2F,.2F);round_to_bf16(x);
    std::vector<std::vector<double>> raw_queries(width);
    std::vector<std::uint8_t> original_k(width*512*2),original_v(original_k.size()),empty;
    for(int t=0;t<width;++t) {
        const std::vector<double> input(x.begin()+t*2560,x.begin()+(t+1)*2560);
        raw_queries[t]=native_projection_oracle(q,input);
        const auto raw_k=native_projection_oracle(k,input),raw_v=native_projection_oracle(v,input);
        const std::array<int,3> position{t+37,t+38,t+39};
        for(int h=0;h<24;++h) {
            std::array<double,256> query{};
            std::copy_n(raw_queries[t].begin()+h*512,256,query.begin());
            query=oracle_core_norm_rope(query,qgamma,position);
            for(int d=0;d<256;++d) { queries[t*6144+h*256+d]=qwen4_sequence::represented(query[d]); }
        }
        for(int h=0;h<2;++h) {
            std::array<double,256> key{};
            std::copy_n(raw_k.begin()+h*256,256,key.begin());
            key=oracle_core_norm_rope(key,kgamma,position);
            oracle_encode_cache(key,t,h,capacity,original_k,empty);
            oracle_encode_cache(std::span(raw_v).subspan(h*256,256),t,h,capacity,original_v,empty);
        }
    }
    const auto compress=[&](const std::vector<std::uint8_t>& input) {
        auto result=input;
        for(int h=0;h<2;++h) for(int t=0;t<width;++t) {
            float maximum=0;
            for(int d=0;d<256;++d) { maximum=std::max(maximum,float(std::abs(decode_cache(input,empty,d,t,h,capacity)))); }
            const float scale=maximum==0 ? 1.F : maximum/448.F;
            for(int d=0;d<256;++d) {
                const float value=decode_cache(input,empty,d,t,h,capacity);
                const auto code=assessment_e4m3(value/scale);
                const float reconstructed=float(decode_e4m3(code))*scale;
                const auto bits=f32_to_bf16(reconstructed);
                const auto offset=2*(d+256*(t+capacity*h));
                result[offset]=bits&255;result[offset+1]=bits>>8;
            }
        }
        return result;
    };
    const auto compressed_k=compress(original_k),compressed_v=compress(original_v);
    const std::vector<std::uint8_t> zero_cache(original_k.size(),0);
    failures+=verify_exact("FP8 candidate zero-row reconstruction",compress(zero_cache),zero_cache);
    std::vector<int> selected(width*width,-1),counts(width);
    for(int t=0;t<width;++t) { counts[t]=t+1;for(int id=0;id<=t;++id) { selected[t*width+id]=id; } }
    auto dq=to_device_bf16(queries),ds=to_device(selected),dc=to_device(counts);
    GuardedDeviceBuffer out(width*6144*2),workspace(ops::qsa_selected_attention_workspace_bytes());
    Tensor qt(dq.p,DType::BF16,{256,24,width}),st(ds.p,DType::I32,{width,width}),
        ct(dc.p,DType::I32,{width}),ot(out.data(),DType::BF16,{256,24,width}),
        wt(workspace.data(),DType::U8,{static_cast<int>(workspace.bytes())});
    std::vector<double> baseline_attention,baseline_output;
    for(int mode=0;mode<4;++mode) {
        const auto& kc=(mode&1)?compressed_k:original_k;
        const auto& vc=(mode&2)?compressed_v:original_v;
        std::vector<double> expected(width*6144),projected;
        for(int t=0;t<width;++t) {
            std::vector<double> gated(6144);
            for(int h=0;h<24;++h) {
                std::array<double,256> query{};
                std::copy_n(queries.begin()+t*6144+h*256,256,query.begin());
                const auto attention=oracle_attention_head(query,std::span(selected).subspan(t*width,t+1),
                    h/12,capacity,kc,empty,vc,empty);
                for(int d=0;d<256;++d) {
                    expected[t*6144+h*256+d]=attention[d];
                    gated[h*256+d]=attention[d]/(1+std::exp(-raw_queries[t][h*512+256+d]));
                }
            }
            const auto output=native_projection_oracle(o,gated);
            projected.insert(projected.end(),output.begin(),output.end());
        }
        if(mode==0) { baseline_attention=expected;baseline_output=projected;continue; }
        const auto report=[&](const char* label,const auto& reference,const auto& candidate) {
            double error2=0,norm2=0,maximum=0;
            for(std::size_t i=0;i<reference.size();++i) {
                const double error=candidate[i]-reference[i];error2+=error*error;
                norm2+=reference[i]*reference[i];maximum=std::max(maximum,std::abs(error));
            }
            std::cout<<"FP8_KV_ASSESS mode="<<mode<<" boundary="<<label<<" relative_l2="
                <<std::sqrt(error2/norm2)<<" max_abs="<<maximum<<'\n';
            if(!std::isfinite(error2) || !std::isfinite(norm2)) { ++failures; }
        };
        report("attention",baseline_attention,expected);report("output_projection",baseline_output,projected);
        StateFixture state(capacity,ops::QsaKvFormat::BF16);auto view=state.view();
        cudaMemcpy(state.k_codes.data(),kc.data(),kc.size(),cudaMemcpyHostToDevice);
        cudaMemcpy(state.v_codes.data(),vc.data(),vc.size(),cudaMemcpyHostToDevice);
        for(bool partitioned:{false,true}) {
            for(int start=0;start<width;) {
                const int chunk=partitioned ? (start==0?128:1) : width;
                auto qpart=qt.slice(2,start,chunk),spart=st.slice(1,start,chunk),
                    cpart=ct.slice(0,start,chunk),opart=ot.slice(2,start,chunk);
                ops::qsa_selected_attention(qpart,spart,cpart,view,opart,wt,nullptr);
                start+=chunk;
            }
            cuda_synchronize();
            failures+=verify_reduction("candidate decoded BF16 attention same-input FP64 oracle",
                from_device_bf16(out.data(),expected.size()),expected,ReductionCriterion{1.0/256,1e-6,1.0/128});
        }
        failures+=verify_exact("FP8 candidate frozen selection",from_device<int>(ds.p,selected.size()),selected);
        failures+=verify_exact("FP8 candidate immutable K",from_device<std::uint8_t>(state.k_codes.data(),kc.size()),kc);
        failures+=verify_exact("FP8 candidate immutable V",from_device<std::uint8_t>(state.v_codes.data(),vc.size()),vc);
    }
    failures+=out.verify_guards("FP8 assessment output");
    failures+=workspace.verify_guards("FP8 assessment workspace");
    std::cout<<"FP8_KV_ASSESS payload_fraction=0.5078125 selected_ids=frozen hypothesis_only\n";
    return failures;
}

} // namespace

#ifdef NINFER_QWEN4_SEQUENCE_COMPONENTS
namespace ninfer::test::qwen4_sequence {
Result qsa(const std::string& path,const Result& input,bool partitioned,bool diagnostic_nvfp4) {
    Result output;
    output.failures=native_qsa_case(path,input.actual.size()/2560,partitioned,&input,&output,
        diagnostic_nvfp4 ? ops::QsaKvFormat::NVFP4G16 : ops::QsaKvFormat::BF16);
    return output;
}
Result qsa_calibrated(const std::string& root,const std::string& path,const Result& input,
                     bool partitioned) {
    Result output;
    output.failures=native_qsa_case(path,input.actual.size()/2560,partitioned,&input,&output,
        ops::QsaKvFormat::BF16,root+"/qwen4-fp8-projections.ninfer");
    return output;
}
}
#else
int main(int argc,char** argv) {
    if (require_cuda() != 0) { return 1; }
    if(argc==2 && std::string_view(argv[1])=="--native-fp8-real") {
        const char* root=std::getenv("NINFER_QWEN4_NATIVE_LAYERS");
        if(!root) { return 77; }
        const std::string source=std::string(root)+"/qwen4-layer-3.ninfer",
            calibrated=std::string(root)+"/qwen4-fp8-projections.ninfer";
        int failures=0;
        for(int width:{1,24,129}) {
            std::vector<int> baseline_selection;
            for(bool partitioned:{false,true}) {
                if(partitioned && width!=129) { continue; }
                std::vector<int> selection;
                const int result=native_qsa_case(source,width,partitioned,nullptr,nullptr,
                    ops::QsaKvFormat::BF16,calibrated,&selection);
                failures+=result;
                if(!partitioned) { baseline_selection=selection; }
                failures+=verify_exact("QSA calibrated protected selector",selection,baseline_selection);
                std::cout<<"QSA_CALIBRATED profile=A16 T="<<width
                    <<" chunk="<<partitioned<<" screen_failures="<<result<<std::endl;
            }
        }
        return failures?1:0;
    }
    if(argc==2 && std::string_view(argv[1])=="--native-fp8-kv-assessment") {
        const char* root=std::getenv("NINFER_QWEN4_NATIVE_LAYERS");
        if(!root) { return 77; }
        return native_fp8_kv_assessment(std::string(root)+"/qwen4-layer-3.ninfer")?1:0;
    }
    if(argc==2 && std::string_view(argv[1])=="--native-real") {
        const char* root=std::getenv("NINFER_QWEN4_NATIVE_LAYERS");
        if(!root) { return 77; }
        int failures=0;
        for(int width:{1,5,27,28,128,129}) {
            failures+=native_qsa_case(std::string(root)+"/qwen4-layer-3.ninfer",width);
        }
        failures+=native_qsa_case(std::string(root)+"/qwen4-layer-3.ninfer",5,true);
        failures+=native_qsa_case(std::string(root)+"/qwen4-layer-3.ninfer",5,false,nullptr,nullptr,
                                 ops::QsaKvFormat::NVFP4G16);
        std::cout<<(failures?"FAIL":"PASS")<<" native source QSA\n";
        return failures?1:0;
    }
    if(argc!=1) { std::cerr<<"Unknown QSA test argument\n"; return 1; }
    int failures = 0;
    failures += append_codec_and_attention_case();
    failures += append_alignment_and_state_validation_case();
    failures += selector_ceiling_and_permutation_case();
    failures += selector_boundary_ties_case();
    failures += selector_route_switch_and_malformed_case();
    failures += selector_score_order_case();
    failures += selected_attention_capacity_case();
    failures += bf16_cache_case();
    failures += selector_ceiling_and_permutation_case(ops::QsaKvFormat::BF16);
    failures += selector_boundary_ties_case(ops::QsaKvFormat::BF16);
    failures += selector_route_switch_and_malformed_case(ops::QsaKvFormat::BF16);
    failures += selector_score_order_case(ops::QsaKvFormat::BF16);
    failures += verifier_composite_real_shape_case();
    failures += verifier_composite_real_shape_case(QType::BF16_CTRL);
    failures += verifier_composite_real_shape_case(QType::NVFP4);
    failures += verifier_composite_real_shape_case(QType::FP8_E4M3FN_ROW_BF16S);
    if (failures != 0) {
        std::cerr << "qsa tests failed: " << failures << "\n";
        return 1;
    }
    std::cout << "qsa tests passed\n";
    return 0;
}
#endif
