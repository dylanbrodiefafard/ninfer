#include "ninfer/ops/gated_delta_net.h"

#include "ops/op_tester.h"

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <string>
#include <tuple>
#include <vector>

using namespace ninfer;
using namespace ninfer::test;

namespace {

constexpr std::int32_t kStateDim    = 128;
constexpr std::uint16_t kBf16Poison = 0xffffU;
constexpr std::uint32_t kFp32Poison = 0xffffffffU;

std::vector<std::uint16_t> make_bf16(std::size_t count, std::uint32_t seed) {
    std::vector<float> values(count);
    fill_uniform(values, seed, -0.08F, 0.08F);
    round_to_bf16(values);
    std::vector<std::uint16_t> bits(count);
    for (std::size_t index = 0; index < count; ++index) {
        bits[index] = f32_to_bf16(values[index]);
    }
    return bits;
}

int verify_equal(const std::string& label, const std::vector<std::uint16_t>& lhs,
                 const std::vector<std::uint16_t>& rhs) {
    if (lhs == rhs) { return 0; }
    std::cerr << label << ": BF16 bits differ\n";
    return 1;
}

int verify_equal(const std::string& label, const std::vector<std::uint32_t>& lhs,
                 const std::vector<std::uint32_t>& rhs) {
    if (lhs == rhs) { return 0; }
    std::cerr << label << ": FP32 bits differ\n";
    return 1;
}

struct DeviceTreeSchedule {
    DeviceBuffer parents;
    DeviceBuffer valid;
    DeviceBuffer schedule;
    Tensor tensor;
};

// parents is host I32 [width, rows]; valid_columns is empty (dense) or one extent per row.
DeviceTreeSchedule make_tree_schedule(const std::vector<std::int32_t>& parents, std::int32_t width,
                                      std::int32_t rows,
                                      const std::vector<std::int32_t>& valid_columns) {
    DeviceTreeSchedule out;
    out.parents  = to_device(parents);
    out.schedule = DeviceBuffer(static_cast<std::size_t>(ops::kGdnTreeScheduleWords) *
                                static_cast<std::size_t>(rows) * sizeof(std::int32_t));
    Tensor parent_tensor(out.parents.p, DType::I32, {width, rows});
    Tensor valid_tensor;
    if (!valid_columns.empty()) {
        out.valid    = to_device(valid_columns);
        valid_tensor = Tensor(out.valid.p, DType::I32, {rows});
    }
    out.tensor = Tensor(out.schedule.p, DType::I32, {ops::kGdnTreeScheduleWords, rows});
    ops::gated_delta_net_tree_schedule(parent_tensor, valid_tensor, out.tensor, nullptr);
    return out;
}

int run_case(std::int32_t value_heads, std::int32_t width, std::int32_t batch,
             std::vector<std::int32_t> valid_columns, std::uint32_t seed) {
    constexpr std::int32_t kQkHeads = 16;
    const bool dense                = valid_columns.empty();
    if (dense) { valid_columns.assign(static_cast<std::size_t>(batch), width); }
    const std::int32_t columns       = width * batch;
    const std::int32_t slots         = columns + batch;
    const std::size_t qk_elements    = static_cast<std::size_t>(kStateDim) * kQkHeads * columns;
    const std::size_t value_elements = static_cast<std::size_t>(kStateDim) * value_heads * columns;
    const std::size_t gate_elements  = static_cast<std::size_t>(value_heads) * columns;
    const std::size_t state_elements =
        static_cast<std::size_t>(kStateDim) * kStateDim * value_heads * slots;

    const std::vector<std::uint16_t> q_bits = make_bf16(qk_elements, seed);
    std::vector<std::uint16_t> k_bits       = make_bf16(qk_elements, seed + 1);
    std::vector<std::uint16_t> v_bits       = make_bf16(value_elements, seed + 2);
    std::vector<float> g(gate_elements);
    std::vector<float> beta(gate_elements);
    fill_uniform(g, seed + 3, -1.2F, -0.02F);
    fill_uniform(beta, seed + 4, 0.02F, 0.98F);
    k_bits[0] = 0x8000U;
    k_bits[1] = 0x0001U;
    v_bits[0] = 0x8000U;
    v_bits[1] = 0x0001U;
    g[0]      = std::bit_cast<float>(0x80000000U);
    beta[0]   = std::bit_cast<float>(0x00000001U);
    std::vector<float> state(state_elements);
    fill_uniform(state, seed + 5, -0.03F, 0.03F);

    std::vector<std::int32_t> initial_slots(static_cast<std::size_t>(batch));
    std::vector<std::int32_t> snapshot_bases(static_cast<std::size_t>(batch));
    for (std::int32_t row = 0; row < batch; ++row) {
        snapshot_bases[static_cast<std::size_t>(row)] = row * width;
        initial_slots[static_cast<std::size_t>(row)]  = columns + row;
    }

    DeviceBuffer device_q       = to_device(q_bits);
    DeviceBuffer device_k       = to_device(k_bits);
    DeviceBuffer device_v       = to_device(v_bits);
    DeviceBuffer device_g       = to_device(g);
    DeviceBuffer device_beta    = to_device(beta);
    DeviceBuffer snapshot_state = to_device(state);
    DeviceBuffer record_state   = to_device(state);
    DeviceBuffer device_initial = to_device(initial_slots);
    DeviceBuffer device_bases   = to_device(snapshot_bases);
    DeviceBuffer device_valid;
    if (!dense) { device_valid = to_device(valid_columns); }

    DeviceBuffer snapshot_out(value_elements * sizeof(std::uint16_t));
    DeviceBuffer record_out(value_elements * sizeof(std::uint16_t));
    DeviceBuffer key_record(qk_elements * sizeof(std::uint16_t));
    DeviceBuffer value_record(value_elements * sizeof(std::uint16_t));
    DeviceBuffer gate_record(gate_elements * 2 * sizeof(std::uint32_t));
    snapshot_out.fill(0xff);
    record_out.fill(0xff);
    key_record.fill(0xff);
    value_record.fill(0xff);
    gate_record.fill(0xff);

    Tensor q(device_q.p, DType::BF16, {kStateDim, kQkHeads, width, batch});
    Tensor k(device_k.p, DType::BF16, {kStateDim, kQkHeads, width, batch});
    Tensor v(device_v.p, DType::BF16, {kStateDim, value_heads, width, batch});
    Tensor g_tensor(device_g.p, DType::FP32, {value_heads, width, batch});
    Tensor beta_tensor(device_beta.p, DType::FP32, {value_heads, width, batch});
    Tensor snapshot_states(snapshot_state.p, DType::FP32,
                           {kStateDim, kStateDim, value_heads, slots});
    Tensor record_states(record_state.p, DType::FP32, {kStateDim, kStateDim, value_heads, slots});
    Tensor valid;
    if (!dense) { valid = Tensor(device_valid.p, DType::I32, {batch}); }
    Tensor initial(device_initial.p, DType::I32, {batch});
    Tensor bases(device_bases.p, DType::I32, {batch});
    Tensor snapshot_output(snapshot_out.p, DType::BF16, {kStateDim, value_heads, width, batch});
    Tensor record_output(record_out.p, DType::BF16, {kStateDim, value_heads, width, batch});
    Tensor key_record_tensor(key_record.p, DType::BF16, {kStateDim, kQkHeads, width, batch});
    Tensor value_record_tensor(value_record.p, DType::BF16, {kStateDim, value_heads, width, batch});
    Tensor gate_record_tensor(gate_record.p, DType::FP32, {2, value_heads, width, batch});

    const float kScale = 1.0F / std::sqrt(128.0F);
    ops::gated_delta_net_snapshot(q, k, v, g_tensor, beta_tensor, kScale, true, snapshot_states,
                                  valid, initial, bases, snapshot_output, nullptr);
    ops::gated_delta_net_replay_record(q, k, v, g_tensor, beta_tensor, kScale, record_states, valid,
                                       initial, key_record_tensor, value_record_tensor,
                                       gate_record_tensor, record_output, nullptr);
    cuda_synchronize();

    int failures             = 0;
    const std::string suffix = " Hv=" + std::to_string(value_heads) +
                               " T=" + std::to_string(width) + " B=" + std::to_string(batch);
    const std::vector<std::uint16_t> snapshot_output_bits =
        from_device<std::uint16_t>(snapshot_out, value_elements);
    const std::vector<std::uint16_t> record_output_bits =
        from_device<std::uint16_t>(record_out, value_elements);
    failures +=
        verify_equal("replay record output" + suffix, snapshot_output_bits, record_output_bits);

    const std::vector<std::uint16_t> key_bits_after =
        from_device<std::uint16_t>(key_record, qk_elements);
    const std::vector<std::uint16_t> value_bits_after =
        from_device<std::uint16_t>(value_record, value_elements);
    const std::vector<std::uint32_t> gate_bits_after =
        from_device<std::uint32_t>(gate_record, gate_elements * 2);
    for (std::int32_t row = 0; row < batch; ++row) {
        const std::int32_t valid_extent = valid_columns[static_cast<std::size_t>(row)];
        for (std::int32_t token = 0; token < width; ++token) {
            const std::int64_t column = static_cast<std::int64_t>(row) * width + token;
            const bool active         = token < valid_extent;
            for (std::int32_t head = 0; head < kQkHeads; ++head) {
                const std::size_t base =
                    static_cast<std::size_t>((column * kQkHeads + head) * kStateDim);
                for (std::int32_t dim = 0; dim < kStateDim; ++dim) {
                    const std::uint16_t expected = active ? k_bits[base + dim] : kBf16Poison;
                    if (key_bits_after[base + dim] != expected) {
                        std::cerr << "key record mismatch" << suffix << "\n";
                        return failures + 1;
                    }
                }
            }
            for (std::int32_t head = 0; head < value_heads; ++head) {
                const std::size_t vector_base =
                    static_cast<std::size_t>((column * value_heads + head) * kStateDim);
                for (std::int32_t dim = 0; dim < kStateDim; ++dim) {
                    const std::uint16_t expected = active ? v_bits[vector_base + dim] : kBf16Poison;
                    if (value_bits_after[vector_base + dim] != expected) {
                        std::cerr << "value record mismatch" << suffix << "\n";
                        return failures + 1;
                    }
                }
                const std::size_t gate_offset =
                    static_cast<std::size_t>((column * value_heads + head) * 2);
                const std::size_t source_offset =
                    static_cast<std::size_t>(column * value_heads + head);
                const std::uint32_t expected_g =
                    active ? std::bit_cast<std::uint32_t>(g[source_offset]) : kFp32Poison;
                const std::uint32_t expected_beta =
                    active ? std::bit_cast<std::uint32_t>(beta[source_offset]) : kFp32Poison;
                if (gate_bits_after[gate_offset] != expected_g ||
                    gate_bits_after[gate_offset + 1] != expected_beta) {
                    std::cerr << "gate record mismatch" << suffix << "\n";
                    return failures + 1;
                }
            }
            if (!active) {
                const std::size_t output_base =
                    static_cast<std::size_t>(column) * value_heads * kStateDim;
                for (std::int32_t index = 0; index < value_heads * kStateDim; ++index) {
                    if (record_output_bits[output_base + index] != 0) {
                        std::cerr << "record invalid output is not zero" << suffix << "\n";
                        return failures + 1;
                    }
                }
            }
        }
    }

    const std::vector<float> state_after = from_device<float>(record_state, state_elements);
    if (state_after != state) {
        std::cerr << "replay record modified source state" << suffix << "\n";
        ++failures;
    }

    DeviceBuffer repeat_state = to_device(state);
    DeviceBuffer repeat_out(value_elements * sizeof(std::uint16_t));
    DeviceBuffer repeat_key(qk_elements * sizeof(std::uint16_t));
    DeviceBuffer repeat_value(value_elements * sizeof(std::uint16_t));
    DeviceBuffer repeat_gate(gate_elements * 2 * sizeof(std::uint32_t));
    repeat_out.fill(0xff);
    repeat_key.fill(0xff);
    repeat_value.fill(0xff);
    repeat_gate.fill(0xff);
    Tensor repeat_states(repeat_state.p, DType::FP32, {kStateDim, kStateDim, value_heads, slots});
    Tensor repeat_output(repeat_out.p, DType::BF16, {kStateDim, value_heads, width, batch});
    Tensor repeat_key_t(repeat_key.p, DType::BF16, {kStateDim, kQkHeads, width, batch});
    Tensor repeat_value_t(repeat_value.p, DType::BF16, {kStateDim, value_heads, width, batch});
    Tensor repeat_gate_t(repeat_gate.p, DType::FP32, {2, value_heads, width, batch});
    ops::gated_delta_net_replay_record(q, k, v, g_tensor, beta_tensor, kScale, repeat_states, valid,
                                       initial, repeat_key_t, repeat_value_t, repeat_gate_t,
                                       repeat_output, nullptr);
    cuda_synchronize();
    const std::vector<float> repeat_state_after = from_device<float>(repeat_state, state_elements);
    if (repeat_state_after != state) {
        std::cerr << "repeat replay record modified source state" << suffix << "\n";
        ++failures;
    }
    failures += verify_equal("repeat key records" + suffix,
                             from_device<std::uint16_t>(repeat_key, qk_elements), key_bits_after);
    failures +=
        verify_equal("repeat value records" + suffix,
                     from_device<std::uint16_t>(repeat_value, value_elements), value_bits_after);
    failures +=
        verify_equal("repeat gate records" + suffix,
                     from_device<std::uint32_t>(repeat_gate, gate_elements * 2), gate_bits_after);

    const std::size_t slot_floats = static_cast<std::size_t>(kStateDim) * kStateDim * value_heads;
    DeviceBuffer t1_state(slot_floats * sizeof(float));
    DeviceBuffer t1_initial = to_device(std::vector<std::int32_t>{0});
    DeviceBuffer t1_bases   = to_device(std::vector<std::int32_t>{0});
    DeviceBuffer t1_out(value_elements * sizeof(std::uint16_t));
    t1_out.fill(0);
    Tensor t1_states(t1_state.p, DType::FP32, {kStateDim, kStateDim, value_heads, 1});
    Tensor t1_initial_t(t1_initial.p, DType::I32, {1});
    Tensor t1_bases_t(t1_bases.p, DType::I32, {1});
    Tensor t1_empty_valid;
    Tensor t1_full(t1_out.p, DType::BF16, {kStateDim, value_heads, width, batch});
    for (std::int32_t row = 0; row < batch; ++row) {
        const std::int32_t initial_slot = initial_slots[static_cast<std::size_t>(row)];
        t1_state.copy_from_host(state.data() + static_cast<std::size_t>(initial_slot) * slot_floats,
                                slot_floats * sizeof(float));
        Tensor q_row                    = q.slice(3, row, 1);
        Tensor k_row                    = k.slice(3, row, 1);
        Tensor v_row                    = v.slice(3, row, 1);
        Tensor g_row                    = g_tensor.slice(2, row, 1);
        Tensor beta_row                 = beta_tensor.slice(2, row, 1);
        Tensor out_row                  = t1_full.slice(3, row, 1);
        const std::int32_t valid_extent = valid_columns[static_cast<std::size_t>(row)];
        for (std::int32_t token = 0; token < valid_extent; ++token) {
            Tensor q_t    = q_row.slice(2, token, 1);
            Tensor k_t    = k_row.slice(2, token, 1);
            Tensor v_t    = v_row.slice(2, token, 1);
            Tensor g_t    = g_row.slice(1, token, 1);
            Tensor beta_t = beta_row.slice(1, token, 1);
            Tensor out_t  = out_row.slice(2, token, 1);
            ops::gated_delta_net_snapshot(q_t, k_t, v_t, g_t, beta_t, kScale, true, t1_states,
                                          t1_empty_valid, t1_initial_t, t1_bases_t, out_t, nullptr);
        }
    }
    cuda_synchronize();
    failures += verify_equal("replay record repeat vs T=1 snapshot" + suffix,
                             from_device<std::uint16_t>(repeat_out, value_elements),
                             from_device<std::uint16_t>(t1_out, value_elements));
    return failures;
}

int run_tree_case(std::int32_t value_heads, std::uint32_t seed) {
    constexpr std::int32_t kQkHeads = 16;
    constexpr std::int32_t kWidth   = 3;
    constexpr std::int32_t kBatch   = 1;
    constexpr std::int32_t kSlots   = 2;
    const std::size_t qk_tree       = static_cast<std::size_t>(kStateDim) * kQkHeads * kWidth;
    const std::size_t value_tree    = static_cast<std::size_t>(kStateDim) * value_heads * kWidth;
    const std::size_t gate_tree     = static_cast<std::size_t>(value_heads) * kWidth;
    const std::size_t qk_seq        = static_cast<std::size_t>(kStateDim) * kQkHeads * 2;
    const std::size_t value_seq     = static_cast<std::size_t>(kStateDim) * value_heads * 2;
    const std::size_t gate_seq      = static_cast<std::size_t>(value_heads) * 2;
    const std::size_t state_elements =
        static_cast<std::size_t>(kStateDim) * kStateDim * value_heads * kSlots;

    const std::vector<std::uint16_t> q_bits = make_bf16(qk_tree, seed);
    std::vector<std::uint16_t> k_bits       = make_bf16(qk_tree, seed + 1);
    std::vector<std::uint16_t> v_bits       = make_bf16(value_tree, seed + 2);
    std::vector<float> g(gate_tree);
    std::vector<float> beta(gate_tree);
    fill_uniform(g, seed + 3, -1.2F, -0.02F);
    fill_uniform(beta, seed + 4, 0.02F, 0.98F);
    std::vector<float> state(state_elements);
    fill_uniform(state, seed + 5, -0.03F, 0.03F);
    const std::int32_t initial_slot = 1;
    const std::vector<std::int32_t> parent_host{-1, 0, 0};

    const auto pack_pair = [&](std::int32_t second) {
        std::vector<std::uint16_t> q_pair(qk_seq);
        std::vector<std::uint16_t> k_pair(qk_seq);
        std::vector<std::uint16_t> v_pair(value_seq);
        std::vector<float> g_pair(gate_seq);
        std::vector<float> beta_pair(gate_seq);
        const std::array<std::int32_t, 2> tokens{0, second};
        for (std::int32_t dst = 0; dst < 2; ++dst) {
            const std::int32_t src = tokens[static_cast<std::size_t>(dst)];
            for (std::int32_t head = 0; head < kQkHeads; ++head) {
                const std::size_t src_base =
                    (static_cast<std::size_t>(src) * kQkHeads + head) * kStateDim;
                const std::size_t dst_base =
                    (static_cast<std::size_t>(dst) * kQkHeads + head) * kStateDim;
                std::copy_n(q_bits.begin() + static_cast<std::ptrdiff_t>(src_base), kStateDim,
                            q_pair.begin() + static_cast<std::ptrdiff_t>(dst_base));
                std::copy_n(k_bits.begin() + static_cast<std::ptrdiff_t>(src_base), kStateDim,
                            k_pair.begin() + static_cast<std::ptrdiff_t>(dst_base));
            }
            for (std::int32_t head = 0; head < value_heads; ++head) {
                const std::size_t src_base =
                    (static_cast<std::size_t>(src) * value_heads + head) * kStateDim;
                const std::size_t dst_base =
                    (static_cast<std::size_t>(dst) * value_heads + head) * kStateDim;
                std::copy_n(v_bits.begin() + static_cast<std::ptrdiff_t>(src_base), kStateDim,
                            v_pair.begin() + static_cast<std::ptrdiff_t>(dst_base));
                g_pair[static_cast<std::size_t>(dst) * value_heads + head] =
                    g[static_cast<std::size_t>(src) * value_heads + head];
                beta_pair[static_cast<std::size_t>(dst) * value_heads + head] =
                    beta[static_cast<std::size_t>(src) * value_heads + head];
            }
        }
        return std::tuple{q_pair, k_pair, v_pair, g_pair, beta_pair};
    };

    const auto run_record = [&](const std::vector<std::uint16_t>& q_host,
                                const std::vector<std::uint16_t>& k_host,
                                const std::vector<std::uint16_t>& v_host,
                                const std::vector<float>& g_host,
                                const std::vector<float>& beta_host, std::int32_t width,
                                bool tree) {
        DeviceBuffer device_q       = to_device(q_host);
        DeviceBuffer device_k       = to_device(k_host);
        DeviceBuffer device_v       = to_device(v_host);
        DeviceBuffer device_g       = to_device(g_host);
        DeviceBuffer device_beta    = to_device(beta_host);
        DeviceBuffer device_state   = to_device(state);
        DeviceBuffer device_initial = to_device(std::vector<std::int32_t>{initial_slot});
        DeviceBuffer out(static_cast<std::size_t>(kStateDim) * value_heads * width *
                         sizeof(std::uint16_t));
        DeviceBuffer key_record(static_cast<std::size_t>(kStateDim) * kQkHeads * width *
                                sizeof(std::uint16_t));
        DeviceBuffer value_record(static_cast<std::size_t>(kStateDim) * value_heads * width *
                                  sizeof(std::uint16_t));
        DeviceBuffer gate_record(static_cast<std::size_t>(value_heads) * width * 2 *
                                 sizeof(std::uint32_t));
        out.fill(0xff);
        key_record.fill(0xff);
        value_record.fill(0xff);
        gate_record.fill(0xff);

        Tensor q(device_q.p, DType::BF16, {kStateDim, kQkHeads, width, kBatch});
        Tensor k(device_k.p, DType::BF16, {kStateDim, kQkHeads, width, kBatch});
        Tensor v(device_v.p, DType::BF16, {kStateDim, value_heads, width, kBatch});
        Tensor g_tensor(device_g.p, DType::FP32, {value_heads, width, kBatch});
        Tensor beta_tensor(device_beta.p, DType::FP32, {value_heads, width, kBatch});
        Tensor states(device_state.p, DType::FP32, {kStateDim, kStateDim, value_heads, kSlots});
        Tensor valid;
        Tensor initial(device_initial.p, DType::I32, {kBatch});
        Tensor key_record_tensor(key_record.p, DType::BF16, {kStateDim, kQkHeads, width, kBatch});
        Tensor value_record_tensor(value_record.p, DType::BF16,
                                   {kStateDim, value_heads, width, kBatch});
        Tensor gate_record_tensor(gate_record.p, DType::FP32, {2, value_heads, width, kBatch});
        Tensor out_tensor(out.p, DType::BF16, {kStateDim, value_heads, width, kBatch});
        DeviceTreeSchedule schedule;
        if (tree) { schedule = make_tree_schedule(parent_host, width, kBatch, {}); }
        const float kScale = 1.0F / std::sqrt(128.0F);
        ops::gated_delta_net_replay_record(q, k, v, g_tensor, beta_tensor, kScale, states, valid,
                                           initial, key_record_tensor, value_record_tensor,
                                           gate_record_tensor, out_tensor, nullptr,
                                           tree ? &schedule.tensor : nullptr);
        cuda_synchronize();
        const std::vector<float> state_after = from_device<float>(device_state, state_elements);
        if (state_after != state) { std::cerr << "tree replay record modified source state\n"; }
        return std::tuple{
            from_device<std::uint16_t>(out,
                                       static_cast<std::size_t>(kStateDim) * value_heads * width),
            from_device<std::uint16_t>(key_record,
                                       static_cast<std::size_t>(kStateDim) * kQkHeads * width),
            from_device<std::uint16_t>(value_record,
                                       static_cast<std::size_t>(kStateDim) * value_heads * width),
            from_device<std::uint32_t>(gate_record,
                                       static_cast<std::size_t>(value_heads) * width * 2),
            state_after != state};
    };

    const auto [tree_out, tree_key, tree_value, tree_gate, tree_mutated] =
        run_record(q_bits, k_bits, v_bits, g, beta, kWidth, true);
    const auto [q0, k0, v0, g0, b0] = pack_pair(1);
    const auto [q1, k1, v1, g1, b1] = pack_pair(2);
    const auto [seq0_out, seq0_key, seq0_value, seq0_gate, seq0_mutated] =
        run_record(q0, k0, v0, g0, b0, 2, false);
    const auto [seq1_out, seq1_key, seq1_value, seq1_gate, seq1_mutated] =
        run_record(q1, k1, v1, g1, b1, 2, false);

    const std::string suffix = " tree Hv=" + std::to_string(value_heads);
    int failures             = 0;
    if (tree_mutated || seq0_mutated || seq1_mutated) { ++failures; }
    const auto compare_col = [&](const char* label, const std::vector<std::uint16_t>& tree_bits,
                                 const std::vector<std::uint16_t>& seq_bits, std::int32_t tree_col,
                                 std::int32_t seq_col, std::int32_t rows) {
        for (std::int32_t index = 0; index < rows; ++index) {
            const std::size_t tree_i =
                static_cast<std::size_t>(tree_col) * static_cast<std::size_t>(rows) + index;
            const std::size_t seq_i =
                static_cast<std::size_t>(seq_col) * static_cast<std::size_t>(rows) + index;
            if (tree_bits[tree_i] != seq_bits[seq_i]) {
                std::cerr << label << suffix << " col mismatch\n";
                ++failures;
                return;
            }
        }
    };

    compare_col("tree out vs sequential parent", tree_out, seq0_out, 0, 0, value_heads * kStateDim);
    compare_col("tree out vs sequential child 0", tree_out, seq0_out, 1, 1,
                value_heads * kStateDim);
    compare_col("tree out vs sequential child 1", tree_out, seq1_out, 2, 1,
                value_heads * kStateDim);
    compare_col("tree key record child 0", tree_key, seq0_key, 1, 1, kQkHeads * kStateDim);
    compare_col("tree key record child 1", tree_key, seq1_key, 2, 1, kQkHeads * kStateDim);
    compare_col("tree value record child 0", tree_value, seq0_value, 1, 1, value_heads * kStateDim);
    compare_col("tree value record child 1", tree_value, seq1_value, 2, 1, value_heads * kStateDim);
    for (std::int32_t head = 0; head < value_heads * 2; ++head) {
        if (tree_gate[1 * static_cast<std::size_t>(value_heads) * 2 + head] !=
                seq0_gate[1 * static_cast<std::size_t>(value_heads) * 2 + head] ||
            tree_gate[2 * static_cast<std::size_t>(value_heads) * 2 + head] !=
                seq1_gate[1 * static_cast<std::size_t>(value_heads) * 2 + head]) {
            std::cerr << "tree gate record mismatch" << suffix << "\n";
            return failures + 1;
        }
    }
    if (seq0_out != seq1_out &&
        std::equal(tree_out.begin() + static_cast<std::ptrdiff_t>(value_heads) * kStateDim,
                   tree_out.begin() + 2 * static_cast<std::ptrdiff_t>(value_heads) * kStateDim,
                   tree_out.begin() + 2 * static_cast<std::ptrdiff_t>(value_heads) * kStateDim)) {
        std::cerr << "tree sibling outputs are identical" << suffix << "\n";
        ++failures;
    }
    return failures;
}

int run_tree_chain_matches_sequential(std::int32_t value_heads, std::int32_t width,
                                      std::uint32_t seed) {
    constexpr std::int32_t kQkHeads  = 16;
    constexpr std::int32_t kBatch    = 1;
    constexpr std::int32_t kSlots    = 2;
    const std::size_t qk_elements    = static_cast<std::size_t>(kStateDim) * kQkHeads * width;
    const std::size_t value_elements = static_cast<std::size_t>(kStateDim) * value_heads * width;
    const std::size_t gate_elements  = static_cast<std::size_t>(value_heads) * width;
    const std::size_t state_elements =
        static_cast<std::size_t>(kStateDim) * kStateDim * value_heads * kSlots;

    const std::vector<std::uint16_t> q_bits = make_bf16(qk_elements, seed);
    std::vector<std::uint16_t> k_bits       = make_bf16(qk_elements, seed + 1);
    std::vector<std::uint16_t> v_bits       = make_bf16(value_elements, seed + 2);
    std::vector<float> g(gate_elements);
    std::vector<float> beta(gate_elements);
    fill_uniform(g, seed + 3, -1.2F, -0.02F);
    fill_uniform(beta, seed + 4, 0.02F, 0.98F);
    std::vector<float> state(state_elements);
    fill_uniform(state, seed + 5, -0.03F, 0.03F);
    std::vector<std::int32_t> parent_host(static_cast<std::size_t>(width));
    parent_host[0] = -1;
    for (std::int32_t col = 1; col < width; ++col) {
        parent_host[static_cast<std::size_t>(col)] = col - 1;
    }

    const auto run = [&](bool tree) {
        DeviceBuffer device_q       = to_device(q_bits);
        DeviceBuffer device_k       = to_device(k_bits);
        DeviceBuffer device_v       = to_device(v_bits);
        DeviceBuffer device_g       = to_device(g);
        DeviceBuffer device_beta    = to_device(beta);
        DeviceBuffer device_state   = to_device(state);
        DeviceBuffer device_initial = to_device(std::vector<std::int32_t>{1});
        DeviceBuffer out(value_elements * sizeof(std::uint16_t));
        DeviceBuffer key_record(qk_elements * sizeof(std::uint16_t));
        DeviceBuffer value_record(value_elements * sizeof(std::uint16_t));
        DeviceBuffer gate_record(gate_elements * 2 * sizeof(std::uint32_t));
        out.fill(0xff);
        key_record.fill(0xff);
        value_record.fill(0xff);
        gate_record.fill(0xff);
        Tensor q(device_q.p, DType::BF16, {kStateDim, kQkHeads, width, kBatch});
        Tensor k(device_k.p, DType::BF16, {kStateDim, kQkHeads, width, kBatch});
        Tensor v(device_v.p, DType::BF16, {kStateDim, value_heads, width, kBatch});
        Tensor g_tensor(device_g.p, DType::FP32, {value_heads, width, kBatch});
        Tensor beta_tensor(device_beta.p, DType::FP32, {value_heads, width, kBatch});
        Tensor states(device_state.p, DType::FP32, {kStateDim, kStateDim, value_heads, kSlots});
        Tensor valid;
        Tensor initial(device_initial.p, DType::I32, {kBatch});
        Tensor key_t(key_record.p, DType::BF16, {kStateDim, kQkHeads, width, kBatch});
        Tensor value_t(value_record.p, DType::BF16, {kStateDim, value_heads, width, kBatch});
        Tensor gate_t(gate_record.p, DType::FP32, {2, value_heads, width, kBatch});
        Tensor out_t(out.p, DType::BF16, {kStateDim, value_heads, width, kBatch});
        DeviceTreeSchedule schedule;
        if (tree) { schedule = make_tree_schedule(parent_host, width, kBatch, {}); }
        const float kScale = 1.0F / std::sqrt(128.0F);
        ops::gated_delta_net_replay_record(q, k, v, g_tensor, beta_tensor, kScale, states, valid,
                                           initial, key_t, value_t, gate_t, out_t, nullptr,
                                           tree ? &schedule.tensor : nullptr);
        cuda_synchronize();
        return from_device<std::uint16_t>(out, value_elements);
    };

    if (run(true) != run(false)) {
        std::cerr << "tree chain W=" << width << " Hv=" << value_heads
                  << " diverged from sequential record\n";
        return 1;
    }
    return 0;
}

// Every valid column of every row's tree must equal, bit for bit, sequential Record over that
// column's root path (checkpoint, ancestors, column), and publish raw records of its own inputs.
// Topologies cover depth-first trees, branch-slot exhaustion (replay steps), non-depth-first
// orders, and rows with different trees and valid extents in one launch.
int run_tree_paths_match_sequential(std::int32_t value_heads, std::int32_t width,
                                    const std::vector<std::vector<std::int32_t>>& row_parents,
                                    const std::vector<std::int32_t>& valid_columns,
                                    std::uint32_t seed) {
    constexpr std::int32_t kQkHeads = 16;
    const auto rows                 = static_cast<std::int32_t>(row_parents.size());
    const std::int32_t slots        = rows + 1;
    const std::size_t qk_column     = static_cast<std::size_t>(kStateDim) * kQkHeads;
    const std::size_t value_column  = static_cast<std::size_t>(kStateDim) * value_heads;
    const std::size_t gate_column   = static_cast<std::size_t>(value_heads);
    const std::size_t columns       = static_cast<std::size_t>(width) * rows;
    const std::size_t slot_floats   = static_cast<std::size_t>(kStateDim) * kStateDim * value_heads;

    const std::vector<std::uint16_t> q_bits = make_bf16(qk_column * columns, seed);
    const std::vector<std::uint16_t> k_bits = make_bf16(qk_column * columns, seed + 1);
    const std::vector<std::uint16_t> v_bits = make_bf16(value_column * columns, seed + 2);
    std::vector<float> g(gate_column * columns);
    std::vector<float> beta(gate_column * columns);
    fill_uniform(g, seed + 3, -1.2F, -0.02F);
    fill_uniform(beta, seed + 4, 0.02F, 0.98F);
    std::vector<float> state(slot_floats * static_cast<std::size_t>(slots));
    fill_uniform(state, seed + 5, -0.03F, 0.03F);
    std::vector<std::int32_t> parents;
    std::vector<std::int32_t> initial;
    for (std::int32_t row = 0; row < rows; ++row) {
        parents.insert(parents.end(), row_parents[static_cast<std::size_t>(row)].begin(),
                       row_parents[static_cast<std::size_t>(row)].end());
        initial.push_back(slots - 1 - row);
    }
    const float kScale = 1.0F / std::sqrt(128.0F);

    // Runs Record on host-packed [.., width, rows] inputs and returns out/key/value/gate bits.
    const auto record = [&](const std::vector<std::uint16_t>& q_host,
                            const std::vector<std::uint16_t>& k_host,
                            const std::vector<std::uint16_t>& v_host,
                            const std::vector<float>& g_host, const std::vector<float>& beta_host,
                            std::int32_t w, std::int32_t b, const std::vector<std::int32_t>& slot,
                            const Tensor* schedule, const std::vector<std::int32_t>& valid_host) {
        const std::size_t n = static_cast<std::size_t>(w) * b;
        DeviceBuffer dq = to_device(q_host), dk = to_device(k_host), dv = to_device(v_host);
        DeviceBuffer dg = to_device(g_host), db = to_device(beta_host);
        DeviceBuffer ds = to_device(state), di = to_device(slot);
        DeviceBuffer dvalid;
        DeviceBuffer out(value_column * n * sizeof(std::uint16_t));
        DeviceBuffer key(qk_column * n * sizeof(std::uint16_t));
        DeviceBuffer value(value_column * n * sizeof(std::uint16_t));
        DeviceBuffer gate(gate_column * n * 2 * sizeof(std::uint32_t));
        out.fill(0xff);
        key.fill(0xff);
        value.fill(0xff);
        gate.fill(0xff);
        Tensor valid_t;
        if (!valid_host.empty()) {
            dvalid  = to_device(valid_host);
            valid_t = Tensor(dvalid.p, DType::I32, {b});
        }
        Tensor q(dq.p, DType::BF16, {kStateDim, kQkHeads, w, b});
        Tensor k(dk.p, DType::BF16, {kStateDim, kQkHeads, w, b});
        Tensor v(dv.p, DType::BF16, {kStateDim, value_heads, w, b});
        Tensor g_tensor(dg.p, DType::FP32, {value_heads, w, b});
        Tensor beta_tensor(db.p, DType::FP32, {value_heads, w, b});
        Tensor state_tensor(ds.p, DType::FP32, {kStateDim, kStateDim, value_heads, slots});
        Tensor slot_tensor(di.p, DType::I32, {b});
        Tensor key_tensor(key.p, DType::BF16, {kStateDim, kQkHeads, w, b});
        Tensor value_tensor(value.p, DType::BF16, {kStateDim, value_heads, w, b});
        Tensor gate_tensor(gate.p, DType::FP32, {2, value_heads, w, b});
        Tensor out_tensor(out.p, DType::BF16, {kStateDim, value_heads, w, b});
        ops::gated_delta_net_replay_record(q, k, v, g_tensor, beta_tensor, kScale, state_tensor,
                                           valid_t, slot_tensor, key_tensor, value_tensor,
                                           gate_tensor, out_tensor, nullptr, schedule);
        cuda_synchronize();
        return std::tuple{from_device<std::uint16_t>(out, value_column * n),
                          from_device<std::uint16_t>(key, qk_column * n),
                          from_device<std::uint16_t>(value, value_column * n),
                          from_device<std::uint32_t>(gate, gate_column * n * 2)};
    };

    const DeviceTreeSchedule schedule = make_tree_schedule(parents, width, rows, valid_columns);
    const auto [tree_out, tree_key, tree_value, tree_gate] = record(
        q_bits, k_bits, v_bits, g, beta, width, rows, initial, &schedule.tensor, valid_columns);

    int failures = 0;
    for (std::int32_t row = 0; row < rows; ++row) {
        const std::vector<std::int32_t>& parent = row_parents[static_cast<std::size_t>(row)];
        const std::int32_t valid =
            valid_columns.empty() ? width : valid_columns[static_cast<std::size_t>(row)];
        for (std::int32_t col = 0; col < valid; ++col) {
            std::vector<std::int32_t> path;
            for (std::int32_t node = col; node >= 0;
                 node              = parent[static_cast<std::size_t>(node)]) {
                path.insert(path.begin(), node);
            }
            // Record requires T>=2; a lone root path is padded with a trailing copy.
            const auto path_len      = static_cast<std::int32_t>(path.size());
            const std::int32_t seq_w = std::max(path_len, 2);
            std::vector<std::uint16_t> q_seq, k_seq, v_seq;
            std::vector<float> g_seq, b_seq;
            for (std::int32_t i = 0; i < seq_w; ++i) {
                const std::size_t src =
                    static_cast<std::size_t>(row) * width +
                    static_cast<std::size_t>(
                        path[static_cast<std::size_t>(std::min(i, path_len - 1))]);
                const auto take = [&](auto& dst, const auto& from, std::size_t stride) {
                    dst.insert(dst.end(), from.begin() + static_cast<std::ptrdiff_t>(src * stride),
                               from.begin() + static_cast<std::ptrdiff_t>((src + 1) * stride));
                };
                take(q_seq, q_bits, qk_column);
                take(k_seq, k_bits, qk_column);
                take(v_seq, v_bits, value_column);
                take(g_seq, g, gate_column);
                take(b_seq, beta, gate_column);
            }
            const auto [seq_out, seq_key, seq_value, seq_gate] =
                record(q_seq, k_seq, v_seq, g_seq, b_seq, seq_w, 1,
                       {initial[static_cast<std::size_t>(row)]}, nullptr, {});
            const std::size_t tree_c = static_cast<std::size_t>(row) * width + col;
            const std::size_t seq_c  = static_cast<std::size_t>(path_len - 1);
            const bool out_ok        = std::equal(
                tree_out.begin() + static_cast<std::ptrdiff_t>(tree_c * value_column),
                tree_out.begin() + static_cast<std::ptrdiff_t>((tree_c + 1) * value_column),
                seq_out.begin() + static_cast<std::ptrdiff_t>(seq_c * value_column));
            const bool key_ok =
                std::equal(tree_key.begin() + static_cast<std::ptrdiff_t>(tree_c * qk_column),
                           tree_key.begin() + static_cast<std::ptrdiff_t>((tree_c + 1) * qk_column),
                           k_bits.begin() + static_cast<std::ptrdiff_t>(tree_c * qk_column));
            const bool value_ok = std::equal(
                tree_value.begin() + static_cast<std::ptrdiff_t>(tree_c * value_column),
                tree_value.begin() + static_cast<std::ptrdiff_t>((tree_c + 1) * value_column),
                v_bits.begin() + static_cast<std::ptrdiff_t>(tree_c * value_column));
            bool gate_ok = true;
            for (std::size_t head = 0; head < gate_column; ++head) {
                gate_ok = gate_ok &&
                          tree_gate[(tree_c * gate_column + head) * 2] ==
                              std::bit_cast<std::uint32_t>(g[tree_c * gate_column + head]) &&
                          tree_gate[(tree_c * gate_column + head) * 2 + 1] ==
                              std::bit_cast<std::uint32_t>(beta[tree_c * gate_column + head]);
            }
            if (!out_ok || !key_ok || !value_ok || !gate_ok) {
                std::cerr << "tree path W=" << width << " Hv=" << value_heads << " row " << row
                          << " column " << col << " mismatch (out " << out_ok << " key " << key_ok
                          << " value " << value_ok << " gate " << gate_ok << ")\n";
                ++failures;
            }
        }
        for (std::int32_t col = valid; col < width; ++col) {
            const std::size_t tree_c = static_cast<std::size_t>(row) * width + col;
            for (std::size_t i = 0; i < value_column; ++i) {
                if (tree_out[tree_c * value_column + i] != 0) {
                    std::cerr << "tree path invalid column output is not zero\n";
                    return failures + 1;
                }
            }
        }
    }
    return failures;
}

} // namespace

int main() {
    if (cuda_unavailable()) {
        std::cerr << "FAIL: no usable CUDA device\n";
        return 1;
    }

    int failures = 0;
    failures += run_case(32, 2, 1, {}, 1701U);
    failures += run_case(32, 16, 1, {7}, 1711U);
    failures += run_case(32, 6, 4, {6, 5, 4, 3}, 1721U);
    failures += run_case(48, 5, 3, {5, 5, 5}, 1726U);
    failures += run_case(48, 4, 1, {}, 1727U);
    failures += run_case(48, 5, 1, {}, 1728U);
    failures += run_case(48, 6, 1, {}, 1729U);
    failures += run_case(48, 2, 1, {1}, 1731U);
    failures += run_case(48, 6, 4, {6, 4, 3, 2}, 1741U);
    failures += run_tree_case(32, 1751U);
    failures += run_tree_case(48, 1761U);
    failures += run_tree_chain_matches_sequential(48, 12, 1766U);
    failures += run_tree_chain_matches_sequential(48, 16, 1771U);
    // Depth-first best-first tree (spine then siblings) and a deeper two-level branch tree.
    failures +=
        run_tree_paths_match_sequential(48, 12, {{-1, 0, 1, 2, 3, 4, 5, 6, 1, 8, 2, 0}}, {}, 1781U);
    failures += run_tree_paths_match_sequential(
        48, 16, {{-1, 0, 1, 2, 3, 4, 5, 6, 1, 8, 9, 2, 11, 0, 13, 3}}, {}, 1786U);
    // Six open branches exceed the three branch slots and force replay steps.
    failures += run_tree_paths_match_sequential(
        48, 16, {{-1, 0, 1, 2, 3, 4, 5, 5, 4, 3, 2, 1, 0, 12, 12, 0}}, {}, 1791U);
    // Star: one branch with fifteen children; breadth-first order (parents not adjacent).
    failures += run_tree_paths_match_sequential(
        32, 16, {{-1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}}, {}, 1796U);
    failures +=
        run_tree_paths_match_sequential(48, 12, {{-1, 0, 0, 1, 1, 2, 2, 3, 4, 5, 6, 7}}, {}, 1801U);
    // Two rows with different trees and valid extents in one launch.
    failures += run_tree_paths_match_sequential(
        48, 12, {{-1, 0, 1, 2, 3, 0, 5, 6, 1, 8, 0, 0}, {-1, 0, 0, 1, 2, 3, 3, 0, 7, 8, 9, 10}},
        {9, 12}, 1806U);
    std::cout << (failures == 0 ? "OK" : "FAIL") << " gated_delta_net_replay_record\n";
    return failures == 0 ? 0 : 1;
}
