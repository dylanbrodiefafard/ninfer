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

    constexpr float kScale = 1.0F / std::sqrt(128.0F);
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
    return failures;
}

int run_tree_case(std::int32_t value_heads, std::uint32_t seed) {
    constexpr std::int32_t kQkHeads = 16;
    constexpr std::int32_t kWidth   = 3;
    constexpr std::int32_t kBatch   = 1;
    constexpr std::int32_t kSlots   = 2;
    const std::size_t qk_tree    = static_cast<std::size_t>(kStateDim) * kQkHeads * kWidth;
    const std::size_t value_tree = static_cast<std::size_t>(kStateDim) * value_heads * kWidth;
    const std::size_t gate_tree  = static_cast<std::size_t>(value_heads) * kWidth;
    const std::size_t qk_seq     = static_cast<std::size_t>(kStateDim) * kQkHeads * 2;
    const std::size_t value_seq  = static_cast<std::size_t>(kStateDim) * value_heads * 2;
    const std::size_t gate_seq   = static_cast<std::size_t>(value_heads) * 2;
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
                    static_cast<std::size_t>((src * kQkHeads + head) * kStateDim);
                const std::size_t dst_base =
                    static_cast<std::size_t>((dst * kQkHeads + head) * kStateDim);
                std::copy_n(q_bits.begin() + static_cast<std::ptrdiff_t>(src_base), kStateDim,
                            q_pair.begin() + static_cast<std::ptrdiff_t>(dst_base));
                std::copy_n(k_bits.begin() + static_cast<std::ptrdiff_t>(src_base), kStateDim,
                            k_pair.begin() + static_cast<std::ptrdiff_t>(dst_base));
            }
            for (std::int32_t head = 0; head < value_heads; ++head) {
                const std::size_t src_base =
                    static_cast<std::size_t>((src * value_heads + head) * kStateDim);
                const std::size_t dst_base =
                    static_cast<std::size_t>((dst * value_heads + head) * kStateDim);
                std::copy_n(v_bits.begin() + static_cast<std::ptrdiff_t>(src_base), kStateDim,
                            v_pair.begin() + static_cast<std::ptrdiff_t>(dst_base));
                g_pair[static_cast<std::size_t>(dst * value_heads + head)] =
                    g[static_cast<std::size_t>(src * value_heads + head)];
                beta_pair[static_cast<std::size_t>(dst * value_heads + head)] =
                    beta[static_cast<std::size_t>(src * value_heads + head)];
            }
        }
        return std::tuple{q_pair, k_pair, v_pair, g_pair, beta_pair};
    };

    const auto run_record = [&](const std::vector<std::uint16_t>& q_host,
                                const std::vector<std::uint16_t>& k_host,
                                const std::vector<std::uint16_t>& v_host,
                                const std::vector<float>& g_host, const std::vector<float>& beta_host,
                                std::int32_t width, const Tensor* parent) {
        DeviceBuffer device_q    = to_device(q_host);
        DeviceBuffer device_k    = to_device(k_host);
        DeviceBuffer device_v    = to_device(v_host);
        DeviceBuffer device_g    = to_device(g_host);
        DeviceBuffer device_beta = to_device(beta_host);
        DeviceBuffer device_state = to_device(state);
        DeviceBuffer device_initial = to_device(std::vector<std::int32_t>{initial_slot});
        DeviceBuffer device_parent;
        if (parent != nullptr) { device_parent = to_device(parent_host); }
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
        Tensor parent_tensor;
        const Tensor* parent_arg = nullptr;
        if (parent != nullptr) {
            parent_tensor = Tensor(device_parent.p, DType::I32, {width});
            parent_arg    = &parent_tensor;
        }
        constexpr float kScale = 1.0F / std::sqrt(128.0F);
        WorkspaceArena tile_workspace(std::max<std::size_t>(
            256, ops::gated_delta_net_replay_record_workspace_capacity_bytes(value_heads, kBatch,
                                                                              width)));
        ops::gated_delta_net_replay_record(q, k, v, g_tensor, beta_tensor, kScale, states, valid,
                                           initial, key_record_tensor, value_record_tensor,
                                           gate_record_tensor, out_tensor, nullptr, parent_arg,
                                           parent_arg != nullptr ? &tile_workspace : nullptr);
        cuda_synchronize();
        const std::vector<float> state_after = from_device<float>(device_state, state_elements);
        if (state_after != state) {
            std::cerr << "tree replay record modified source state\n";
        }
        return std::tuple{from_device<std::uint16_t>(out, static_cast<std::size_t>(kStateDim) *
                                                              value_heads * width),
                          from_device<std::uint16_t>(key_record, static_cast<std::size_t>(kStateDim) *
                                                                    kQkHeads * width),
                          from_device<std::uint16_t>(
                              value_record, static_cast<std::size_t>(kStateDim) * value_heads * width),
                          from_device<std::uint32_t>(gate_record, static_cast<std::size_t>(value_heads) *
                                                                     width * 2),
                          state_after != state};
    };

    Tensor dummy_parent;
    const auto [tree_out, tree_key, tree_value, tree_gate, tree_mutated] =
        run_record(q_bits, k_bits, v_bits, g, beta, kWidth, &dummy_parent);
    const auto [q0, k0, v0, g0, b0] = pack_pair(1);
    const auto [q1, k1, v1, g1, b1] = pack_pair(2);
    const auto [seq0_out, seq0_key, seq0_value, seq0_gate, seq0_mutated] =
        run_record(q0, k0, v0, g0, b0, 2, nullptr);
    const auto [seq1_out, seq1_key, seq1_value, seq1_gate, seq1_mutated] =
        run_record(q1, k1, v1, g1, b1, 2, nullptr);

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

    compare_col("tree out vs sequential parent", tree_out, seq0_out, 0, 0,
                value_heads * kStateDim);
    compare_col("tree out vs sequential child 0", tree_out, seq0_out, 1, 1,
                value_heads * kStateDim);
    compare_col("tree out vs sequential child 1", tree_out, seq1_out, 2, 1,
                value_heads * kStateDim);
    compare_col("tree key record child 0", tree_key, seq0_key, 1, 1, kQkHeads * kStateDim);
    compare_col("tree key record child 1", tree_key, seq1_key, 2, 1, kQkHeads * kStateDim);
    compare_col("tree value record child 0", tree_value, seq0_value, 1, 1, value_heads * kStateDim);
    compare_col("tree value record child 1", tree_value, seq1_value, 2, 1, value_heads * kStateDim);
    for (std::int32_t head = 0; head < value_heads * 2; ++head) {
        if (tree_gate[static_cast<std::size_t>(1 * value_heads * 2 + head)] !=
                seq0_gate[static_cast<std::size_t>(1 * value_heads * 2 + head)] ||
            tree_gate[static_cast<std::size_t>(2 * value_heads * 2 + head)] !=
                seq1_gate[static_cast<std::size_t>(1 * value_heads * 2 + head)]) {
            std::cerr << "tree gate record mismatch" << suffix << "\n";
            return failures + 1;
        }
    }
    if (seq0_out != seq1_out &&
        std::equal(tree_out.begin() + static_cast<std::ptrdiff_t>(value_heads * kStateDim),
                   tree_out.begin() + static_cast<std::ptrdiff_t>(2 * value_heads * kStateDim),
                   tree_out.begin() + static_cast<std::ptrdiff_t>(2 * value_heads * kStateDim))) {
        std::cerr << "tree sibling outputs are identical" << suffix << "\n";
        ++failures;
    }
    return failures;
}

int run_tree_chain_matches_sequential(std::int32_t value_heads, std::int32_t width,
                                      std::uint32_t seed) {
    constexpr std::int32_t kQkHeads = 16;
    constexpr std::int32_t kBatch   = 1;
    constexpr std::int32_t kSlots   = 2;
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
        DeviceBuffer device_parent;
        if (tree) { device_parent = to_device(parent_host); }
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
        Tensor parent_tensor;
        const Tensor* parent_arg = nullptr;
        if (tree) {
            parent_tensor = Tensor(device_parent.p, DType::I32, {width});
            parent_arg    = &parent_tensor;
        }
        constexpr float kScale = 1.0F / std::sqrt(128.0F);
        WorkspaceArena tile_workspace(std::max<std::size_t>(
            256, ops::gated_delta_net_replay_record_workspace_capacity_bytes(value_heads, kBatch,
                                                                              width)));
        ops::gated_delta_net_replay_record(q, k, v, g_tensor, beta_tensor, kScale, states, valid,
                                           initial, key_t, value_t, gate_t, out_t, nullptr,
                                           parent_arg,
                                           parent_arg != nullptr ? &tile_workspace : nullptr);
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

} // namespace

int main() {
    if (cuda_unavailable()) {
        std::cerr << "FAIL: no usable CUDA device\n";
        return 1;
    }

    int failures = 0;
    failures += run_case(32, 2, 1, {}, 1701U);
    failures += run_case(32, 16, 1, {7}, 1711U);
    failures += run_case(32, 6, 8, {6, 5, 4, 3, 2, 1, 6, 2}, 1721U);
    failures += run_case(48, 5, 3, {5, 5, 5}, 1726U);
    failures += run_case(48, 2, 1, {1}, 1731U);
    failures += run_case(48, 6, 8, {6, 4, 3, 2, 1, 5, 6, 2}, 1741U);
    failures += run_tree_case(32, 1751U);
    failures += run_tree_case(48, 1761U);
    failures += run_tree_chain_matches_sequential(48, 12, 1766U);
    failures += run_tree_chain_matches_sequential(48, 16, 1771U);
    std::cout << (failures == 0 ? "OK" : "FAIL") << " gated_delta_net_replay_record\n";
    return failures == 0 ? 0 : 1;
}
