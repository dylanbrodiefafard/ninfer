// Qualifies the complete accepted-history Op against independent FP64 state/query mathematics.
#include "core/decode_graph.h"
#include "core/device.h"
#include <optional>
#include "ninfer/ops/gated_delta_net.h"
#include "ninfer/ops/gdn_replay.h"
#include "ops/op_tester.h"
#include "ops/sanitizer_scope.h"
#include "ninfer/ops/gdn_history.h"
#include "ops/gdn_history_ref.h"
#include <numeric>
#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>
using namespace ninfer;
namespace tr          = ninfer::test;
constexpr int kLayers = 48, kD = 128, kHv = 48, kHq = 16;
constexpr float kScale = 1.F / 11.313708498984761F;

struct TestGraph {
    DecodeGraphDefinition definition;
    DecodeGraphExecutable executable;

    template <class Body>
    void capture(cudaStream_t stream, Body body) {
        definition.capture(stream, [&] { body(stream); });
        executable.instantiate(definition);
    }

    void launch(cudaStream_t stream) { executable.launch(stream); }
};

struct AcceptedHistoryFixture {
    int batch, width;
    bool tree;
    const int round_count;
    const std::array<int, 3> observed_rounds;
    LinearAttentionStatePoolLayout state_layout;
    GdnReplayRecordLayout record_layout;
    DeviceBuffer initial_storage, baseline_storage, candidate_storage, record_storage;
    LinearAttentionStatePool baseline, candidate;
    GdnReplayRecords records;
    DeviceBuffer qd, kd, vd, gate_d, bd, slots_d, valid_d, parent_d, schedule_d, out_d, observed;
    DeviceBuffer history_storage;
    std::optional<GdnHistory> history;
    Tensor q, k, v, g, beta, slots, valid, schedule, out;
    std::vector<float> qh, kh, vh, gh, bh, initial_state;
    std::vector<std::int32_t> slot_h, parent_h;
    std::vector<double> expected_out, expected_final, zero_out, zero_final;
    std::vector<std::uint16_t> conv_fixture;
    std::size_t state_slot_elements = static_cast<std::size_t>(kHv) * kD * kD;
    std::size_t value_elements;
    DeviceContext device;

    static DeviceBuffer bf16_values(std::vector<float>& host, std::size_t n, unsigned seed,
                                    float lo, float hi) {
        host.resize(n);
        tr::fill_uniform(host, seed, lo, hi);
        tr::round_to_bf16(host);
        std::vector<std::uint16_t> words(n);
        for (std::size_t i = 0; i < n; ++i) { words[i] = tr::f32_to_bf16(host[i]); }
        return tr::to_device(words);
    }

    AcceptedHistoryFixture(int b, int w, bool t, int rounds = 64, bool build_cadence_oracle = true)
        : batch(b), width(w), tree(t), round_count(rounds),
          observed_rounds{rounds == 4 ? 0 : 1, rounds / 2 - 1, rounds - 1},
          value_elements(static_cast<std::size_t>(b) * w * kHv * kD) {
        LayoutBuilder sb;
        state_layout           = plan_linear_attention_state_pool(sb, {.layers         = kLayers,
                                                                       .conv_channels  = 10240,
                                                                       .conv_width     = 3,
                                                                       .value_heads    = kHv,
                                                                       .value_head_dim = kD,
                                                                       .key_head_dim   = kD,
                                                                       .slot_count     = batch + 1,
                                                                       .conv_dtype     = DType::BF16});
        const auto state_bytes = sb.finish(256);
        initial_storage        = DeviceBuffer(state_bytes);
        baseline_storage       = DeviceBuffer(state_bytes);
        candidate_storage      = DeviceBuffer(state_bytes);
        baseline  = LinearAttentionStatePool({baseline_storage.p, state_bytes}, state_layout);
        candidate = LinearAttentionStatePool({candidate_storage.p, state_bytes}, state_layout);
        initial_storage.fill(0);
        initial_state.resize(state_slot_elements * batch);
        tr::fill_uniform(initial_state, 19, -.02F, .02F);
        // This fixture rounding makes each integer layer scale exactly represented in FP32.
        tr::round_to_bf16(initial_state);
        LinearAttentionStatePool initial({initial_storage.p, state_bytes}, state_layout);
        for (int layer = 0; layer < kLayers; ++layer) {
            std::vector<float> scaled(initial_state.size());
            for (std::size_t i = 0; i < scaled.size(); ++i) {
                scaled[i] = initial_state[i] * static_cast<float>(layer + 1);
                if (static_cast<double>(scaled[i]) !=
                    static_cast<double>(initial_state[i]) * (layer + 1)) {
                    throw std::runtime_error(
                        "layer sentinel scaling is not exactly represented FP32");
                }
            }
            CUDA_CHECK(cudaMemcpy(initial.recurrent[layer].data, scaled.data(),
                                  scaled.size() * sizeof(float), cudaMemcpyHostToDevice));
        }
        LayoutBuilder rb;
        record_layout  = plan_gdn_replay_records(rb, {.layers          = kLayers,
                                                      .record_capacity = batch,
                                                      .width           = width,
                                                      .conv_channels   = 10240,
                                                      .qk_heads        = kHq,
                                                      .value_heads     = kHv,
                                                      .key_dim         = kD,
                                                      .value_dim       = kD});
        record_storage = DeviceBuffer(rb.finish(256));
        record_storage.fill(0);
        records = GdnReplayRecords({record_storage.p, record_storage.bytes}, record_layout);
        conv_fixture.resize(static_cast<std::size_t>(kLayers) * batch * width * 10240);
        for (std::size_t i = 0; i < conv_fixture.size(); ++i) {
            conv_fixture[i] =
                static_cast<std::uint16_t>(0x3e80 + ((i / 10240) * 13 + i % 10240) % 127);
        }
        CUDA_CHECK(cudaMemcpy(records.conv.data, conv_fixture.data(),
                              conv_fixture.size() * sizeof(std::uint16_t), cudaMemcpyHostToDevice));
        qd = bf16_values(qh, static_cast<std::size_t>(batch) * width * kHq * kD, 11, -.3F, .3F);
        kd = bf16_values(kh, qh.size(), 12, -.3F, .3F);
        vd = bf16_values(vh, value_elements, 13, -.3F, .3F);
        gh.resize(static_cast<std::size_t>(batch) * width * kHv);
        bh.resize(gh.size());
        tr::fill_uniform(gh, 14, -.1F, -.005F);
        tr::fill_uniform(bh, 15, .05F, .95F);
        // A represented gate drives a real zero-decay branch, exercising division-free storage.
        gh[0]  = -120.F;
        gate_d = tr::to_device(gh);
        bd     = tr::to_device(bh);
        slot_h.resize(batch);
        std::iota(slot_h.begin(), slot_h.end(), 0);
        std::reverse(slot_h.begin(), slot_h.end());
        slots_d  = tr::to_device(slot_h);
        valid_d  = tr::to_device(std::vector<std::int32_t>(batch, width));
        q        = Tensor(qd.p, DType::BF16, {kD, kHq, width, batch});
        k        = Tensor(kd.p, DType::BF16, {kD, kHq, width, batch});
        v        = Tensor(vd.p, DType::BF16, {kD, kHv, width, batch});
        g        = Tensor(gate_d.p, DType::FP32, {kHv, width, batch});
        beta     = Tensor(bd.p, DType::FP32, {kHv, width, batch});
        slots    = Tensor(slots_d.p, DType::I32, {batch});
        valid    = Tensor(valid_d.p, DType::I32, {batch});
        out_d    = DeviceBuffer(value_elements * 2);
        out      = Tensor(out_d.p, DType::BF16, {kD, kHv, width, batch});
        observed = DeviceBuffer(value_elements * 2 * (round_count + (kLayers - 1) * 3));
        LayoutBuilder history_builder;
        const auto history_layout = plan_gdn_history(history_builder, {.layers      = 48,
                                                                       .slots       = batch,
                                                                       .width       = width,
                                                                       .capacity    = 4,
                                                                       .qk_heads    = 16,
                                                                       .value_heads = 48,
                                                                       .key_dim     = 128,
                                                                       .value_dim   = 128});
        history_storage           = DeviceBuffer(history_builder.finish(256));
        history.emplace(DeviceSpan{history_storage.p, history_storage.bytes}, history_layout);
        ops::gdn_history_reset_all(history.value(), device.stream);
        parent_h.resize(static_cast<std::size_t>(batch) * width);
        for (int row = 0; row < batch; ++row)
            for (int col = 0; col < width; ++col) {
                parent_h[row * width + col] = col == 0 ? -1 : (col == 2 ? 0 : col - 1);
            }
        parent_d   = tr::to_device(parent_h);
        schedule_d = DeviceBuffer(static_cast<std::size_t>(batch) * ops::kGdnTreeScheduleWords *
                                  sizeof(std::int32_t));
        Tensor parents(parent_d.p, DType::I32, {width, batch});
        schedule = Tensor(schedule_d.p, DType::I32, {ops::kGdnTreeScheduleWords, batch});
        ops::gated_delta_net_tree_schedule(parents, valid, schedule, device.stream);
        CUDA_CHECK(cudaDeviceSynchronize());
        if (build_cadence_oracle) {
            make_oracle(false);
            auto first_out = std::move(expected_out), first_final = std::move(expected_final);
            make_oracle(true);
            zero_out       = std::move(expected_out);
            zero_final     = std::move(expected_final);
            expected_out   = std::move(first_out);
            expected_final = std::move(first_final);
        }
    }

    std::vector<int> path(int row, int node) const {
        std::vector<int> result;
        for (int c = node; c >= 0; c = tree ? parent_h[row * width + c] : c - 1) {
            result.push_back(c);
        }
        std::reverse(result.begin(), result.end());
        return result;
    }

    std::vector<ops::GdnReplayFoldRow> controls(int round, bool force = false) const {
        std::vector<ops::GdnReplayFoldRow> rows(batch);
        for (int row = 0; row < batch; ++row) {
            auto& r             = rows[row];
            r.linear_state_slot = slot_h[row];
            const int kind      = (round + row) % 4;
            const auto chosen   = path(row, kind == 1 ? width - 1 : 0);
            const int n =
                force || kind == 0 ? 0 : (kind == 1 ? static_cast<int>(chosen.size()) : 1);
            r.commit_columns = n;
            if (tree) {
                r.path_length = n;
                for (int j = 0; j < n; ++j) { r.path[j] = chosen[j]; }
            }
        }
        return rows;
    }

    tr::history_ref::Inputs oracle_input(int row, const std::vector<int>& columns,
                                         const std::vector<double>& state) const {
        tr::history_ref::Inputs in;
        in.head_dim    = kD;
        in.qk_heads    = kHq;
        in.value_heads = kHv;
        in.tokens      = columns.size();
        in.state       = state;
        for (int c : columns) {
            const auto qi = static_cast<std::size_t>(row * width + c) * kHq * kD;
            const auto vi = static_cast<std::size_t>(row * width + c) * kHv * kD;
            const auto gi = static_cast<std::size_t>(row * width + c) * kHv;
            in.q.insert(in.q.end(), qh.begin() + qi, qh.begin() + qi + kHq * kD);
            in.k.insert(in.k.end(), kh.begin() + qi, kh.begin() + qi + kHq * kD);
            in.v.insert(in.v.end(), vh.begin() + vi, vh.begin() + vi + kHv * kD);
            in.g.insert(in.g.end(), gh.begin() + gi, gh.begin() + gi + kHv);
            in.beta.insert(in.beta.end(), bh.begin() + gi, bh.begin() + gi + kHv);
        }
        return in;
    }

    void make_oracle(bool zero) {
        expected_out.resize(value_elements * round_count);
        expected_final.resize(state_slot_elements * batch);
        for (std::size_t i = 0; i < expected_final.size(); ++i) {
            expected_final[i] = zero ? 0.0 : initial_state[i];
        }
        for (int round = 0; round < round_count; ++round) {
            const auto rows = controls(round);
            for (int row = 0; row < batch; ++row) {
                const auto sb = static_cast<std::size_t>(slot_h[row]) * state_slot_elements;
                std::vector<double> state(expected_final.begin() + sb,
                                          expected_final.begin() + sb + state_slot_elements);
                tr::history_ref::Result chain;
                if (!tree) {
                    chain = tr::history_ref::evaluate(
                        oracle_input(row, path(row, width - 1), state), kScale, true, true);
                }
                for (int col = 0; col < width; ++col) {
                    const auto base = (static_cast<std::size_t>(round) * batch * width +
                                       static_cast<std::size_t>(row) * width + col) *
                                      kHv * kD;
                    if (tree) {
                        const auto branch = tr::history_ref::evaluate(
                            oracle_input(row, path(row, col), state), kScale, true);
                        std::copy(branch.out.end() - kHv * kD, branch.out.end(),
                                  expected_out.begin() + base);
                    } else {
                        std::copy(chain.out.begin() + static_cast<std::size_t>(col) * kHv * kD,
                                  chain.out.begin() + static_cast<std::size_t>(col + 1) * kHv * kD,
                                  expected_out.begin() + base);
                    }
                }
                const auto& control = rows[row];
                const int n         = control.commit_columns;
                if (n > 0) {
                    if (tree) {
                        std::vector<int> chosen(control.path.begin(), control.path.begin() + n);
                        const auto next = tr::history_ref::evaluate(
                            oracle_input(row, chosen, state), kScale, true);
                        std::copy(next.final_state.begin(), next.final_state.end(),
                                  expected_final.begin() + sb);
                    } else {
                        std::copy(chain.snapshots.begin() +
                                      static_cast<std::size_t>(n - 1) * state_slot_elements,
                                  chain.snapshots.begin() +
                                      static_cast<std::size_t>(n) * state_slot_elements,
                                  expected_final.begin() + sb);
                    }
                }
            }
        }
    }

    template <bool Parent>
    void candidate_record(int layer, cudaStream_t stream) {
        auto record         = records.layer(layer, batch);
        const auto retained = history.value().layer(layer, 0, batch);
        ops::gated_delta_net_history_record(q, k, v, g, beta, kScale, candidate.recurrent[layer],
                                            valid, slots, record.key, record.value, record.gate,
                                            out, retained, stream, Parent ? &schedule : nullptr);
    }

    void commit(const std::vector<ops::GdnReplayFoldRow>& rows, bool force, cudaStream_t stream) {
        if (force) {
            ops::gdn_history_materialize(records, history.value(), candidate.all_layers_view(),
                                         slot_h, stream);
        } else {
            ops::gdn_history_commit(records, history.value(), candidate.all_layers_view(), rows,
                                    stream);
        }
    }

    void cadence(bool trial, bool collect, cudaStream_t stream) {
        DeviceBuffer& target = trial ? candidate_storage : baseline_storage;
        CUDA_CHECK(cudaMemcpyAsync(target.p, initial_storage.p, target.bytes,
                                   cudaMemcpyDeviceToDevice, stream));
        if (trial) { ops::gdn_history_reset_all(history.value(), stream); }
        for (int round = 0; round < round_count; ++round) {
            for (int layer = 0; layer < kLayers; ++layer) {
                if (trial) {
                    if (tree) {
                        candidate_record<true>(layer, stream);
                    } else {
                        candidate_record<false>(layer, stream);
                    }
                } else {
                    auto r = records.layer(layer, batch);
                    ops::gated_delta_net_replay_record(
                        q, k, v, g, beta, kScale, baseline.recurrent[layer], valid, slots, r.key,
                        r.value, r.gate, out, stream, tree ? &schedule : nullptr);
                }
                if (collect) {
                    int frame = -1;
                    if (layer == 0) {
                        frame = round;
                    } else if (round == observed_rounds[0] || round == observed_rounds[1] ||
                               round == observed_rounds[2]) {
                        frame =
                            round_count + (layer - 1) * 3 +
                            (round == observed_rounds[0] ? 0
                                                         : (round == observed_rounds[1] ? 1 : 2));
                    }
                    if (frame >= 0) {
                        CUDA_CHECK(cudaMemcpyAsync(
                            static_cast<std::byte*>(observed.p) +
                                static_cast<std::size_t>(frame) * out_d.bytes,
                            out_d.p, out_d.bytes, cudaMemcpyDeviceToDevice, stream));
                    }
                }
            }
            const auto rows = controls(round);
            if (trial) {
                commit(rows, false, stream);
            } else {
                ops::gdn_replay_fold(records, baseline.all_layers_view(), rows, stream);
            }
        }
        if (trial) { commit(controls(0, true), true, stream); }
    }

    void qualify(bool trial, int executions = 3) {
        TestGraph graph;
        graph.capture(device.stream, [&](cudaStream_t stream) { cadence(trial, true, stream); });
        for (int replay = 0; replay < executions; ++replay) {
            if (replay == 0) {
                cadence(trial, true, device.stream);
            } else {
                graph.launch(device.stream);
            }
            device.synchronize();
            check_cadence(trial);
        }
    }

    void qualify_record_racecheck() {
        const std::vector<double> state(initial_state.begin(), initial_state.end());
        std::vector<double> reference(value_elements);
        for (int column = 0; column < width; ++column) {
            const auto result =
                tr::history_ref::evaluate(oracle_input(0, path(0, column), state), kScale, true);
            std::copy(result.out.end() - kHv * kD, result.out.end(),
                      reference.begin() + static_cast<std::size_t>(column) * kHv * kD);
        }
        const auto record_once = [&](cudaStream_t stream) {
            CUDA_CHECK(cudaMemcpyAsync(candidate_storage.p, initial_storage.p,
                                       candidate_storage.bytes, cudaMemcpyDeviceToDevice, stream));
            ops::gdn_history_reset_all(history.value(), stream);
            if (tree) {
                candidate_record<true>(0, stream);
            } else {
                candidate_record<false>(0, stream);
            }
        };
        TestGraph graph;
        graph.capture(device.stream, record_once);
        for (int execution = 0; execution < 2; ++execution) {
            if (execution == 0) {
                record_once(device.stream);
            } else {
                graph.launch(device.stream);
            }
            device.synchronize();
            std::vector<std::uint16_t> words(value_elements);
            CUDA_CHECK(cudaMemcpy(words.data(), out.data, out.bytes(), cudaMemcpyDeviceToHost));
            std::vector<double> actual(words.size());
            for (std::size_t i = 0; i < words.size(); ++i) {
                actual[i] = tr::bf16_to_f32(words[i]);
            }
            if (tr::verify_reduction("history record FP64 query", actual, reference,
                                     {.0041, 5e-6, .0055})) {
                throw std::runtime_error("bounded history record query oracle failed");
            }
            const auto record    = records.layer(0, batch);
            const auto check_raw = [&](const Tensor& tensor, const std::vector<float>& input) {
                std::vector<std::uint16_t> raw(static_cast<std::size_t>(tensor.numel()));
                CUDA_CHECK(
                    cudaMemcpy(raw.data(), tensor.data, tensor.bytes(), cudaMemcpyDeviceToHost));
                for (std::size_t i = 0; i < raw.size(); ++i) {
                    if (raw[i] != tr::f32_to_bf16(input[i])) {
                        throw std::runtime_error("bounded history raw BF16 record oracle failed");
                    }
                }
            };
            check_raw(record.key, kh);
            check_raw(record.value, vh);
            std::vector<float> gates(static_cast<std::size_t>(record.gate.numel()));
            CUDA_CHECK(cudaMemcpy(gates.data(), record.gate.data, record.gate.bytes(),
                                  cudaMemcpyDeviceToHost));
            for (std::size_t i = 0; i < gh.size(); ++i) {
                if (gates[2 * i] != gh[i] || gates[2 * i + 1] != bh[i]) {
                    throw std::runtime_error("bounded history raw gate oracle failed");
                }
            }
        }
    }

    void check_cadence(bool trial) {
        std::vector<std::uint16_t> words(value_elements * round_count);
        CUDA_CHECK(cudaMemcpy(words.data(), observed.p, words.size() * sizeof(std::uint16_t),
                              cudaMemcpyDeviceToHost));
        std::vector<double> actual(words.size());
        for (std::size_t i = 0; i < words.size(); ++i) { actual[i] = tr::bf16_to_f32(words[i]); }
        int failures = tr::verify_reduction(trial ? "history FP64 layer0 cadence queries"
                                                  : "baseline FP64 layer0 cadence queries",
                                            actual, expected_out, {.0041, 5e-6, .0055});
        for (int layer = 1; layer < kLayers; ++layer) {
            std::vector<std::uint16_t> samples(value_elements * 3);
            CUDA_CHECK(cudaMemcpy(samples.data(),
                                  static_cast<std::byte*>(observed.p) +
                                      static_cast<std::size_t>(round_count + (layer - 1) * 3) *
                                          out_d.bytes,
                                  samples.size() * sizeof(std::uint16_t), cudaMemcpyDeviceToHost));
            std::vector<double> sample_actual(samples.size()), sample_reference(samples.size());
            const auto& rounds = observed_rounds;
            for (int j = 0; j < 3; ++j)
                for (std::size_t i = 0; i < value_elements; ++i) {
                    const auto index  = static_cast<std::size_t>(rounds[j]) * value_elements + i;
                    const auto target = static_cast<std::size_t>(j) * value_elements + i;
                    sample_actual[target] = tr::bf16_to_f32(samples[target]);
                    sample_reference[target] =
                        zero_out[index] +
                        static_cast<double>(layer + 1) * (expected_out[index] - zero_out[index]);
                }
            failures += tr::verify_reduction("FP64 independently affine layer queries",
                                             sample_actual, sample_reference, {.0041, 5e-6, .0055});
        }
        const auto& state = trial ? candidate : baseline;
        for (int layer = 0; layer < kLayers; ++layer) {
            std::vector<float> values(initial_state.size());
            CUDA_CHECK(cudaMemcpy(values.data(), state.recurrent[layer].data,
                                  values.size() * sizeof(float), cudaMemcpyDeviceToHost));
            std::vector<double> state_actual(values.begin(), values.end()),
                state_reference(values.size());
            for (std::size_t i = 0; i < values.size(); ++i) {
                state_reference[i] = zero_final[i] + static_cast<double>(layer + 1) *
                                                         (expected_final[i] - zero_final[i]);
            }
            failures += tr::verify_reduction("FP64 independently affine final FP32 layer state",
                                             state_actual, state_reference, {.0027, 1e-5, .0039});
        }
        if (trial) {
            std::vector<std::int32_t> lengths(batch);
            CUDA_CHECK(cudaMemcpy(lengths.data(), history.value().counts.data,
                                  lengths.size() * sizeof(std::int32_t), cudaMemcpyDeviceToHost));
            for (int n : lengths) {
                if (n != 0) { throw std::runtime_error("final flush left a live ring"); }
            }
        }
        const auto check_raw = [&](const Tensor& tensor, const std::vector<float>& input) {
            std::vector<std::uint16_t> raw(static_cast<std::size_t>(tensor.numel()));
            CUDA_CHECK(cudaMemcpy(raw.data(), tensor.data, tensor.bytes(), cudaMemcpyDeviceToHost));
            for (std::size_t i = 0; i < raw.size(); ++i) {
                if (raw[i] != tr::f32_to_bf16(input[i % input.size()])) {
                    throw std::runtime_error("independent raw BF16 record oracle failed");
                }
            }
        };
        check_raw(records.key, kh);
        check_raw(records.value, vh);
        std::vector<float> raw_gates(static_cast<std::size_t>(records.gate.numel()));
        CUDA_CHECK(cudaMemcpy(raw_gates.data(), records.gate.data, records.gate.bytes(),
                              cudaMemcpyDeviceToHost));
        for (std::size_t i = 0; i < raw_gates.size() / 2; ++i) {
            if (raw_gates[2 * i] != gh[i % gh.size()] ||
                raw_gates[2 * i + 1] != bh[i % bh.size()]) {
                throw std::runtime_error("independent raw gate oracle failed");
            }
        }
        std::vector<std::array<int, 3>> tail(static_cast<std::size_t>(batch), {-1, -1, -1});
        for (int round = 0; round < round_count; ++round) {
            const auto rows = controls(round);
            for (int row = 0; row < batch; ++row) {
                for (int j = 0; j < rows[row].commit_columns; ++j) {
                    const int token = tree ? rows[row].path[j] : j;
                    auto& h         = tail[slot_h[row]];
                    h               = {h[1], h[2], row * width + token};
                }
            }
        }
        for (int layer = 0; layer < kLayers; ++layer) {
            std::vector<std::uint16_t> actual_conv(static_cast<std::size_t>(batch) * 3 * 10240);
            CUDA_CHECK(cudaMemcpy(actual_conv.data(), state.conv[layer].data,
                                  actual_conv.size() * sizeof(std::uint16_t),
                                  cudaMemcpyDeviceToHost));
            for (int slot = 0; slot < batch; ++slot)
                for (int history_index = 0; history_index < 3; ++history_index)
                    for (int channel = 0; channel < 10240; ++channel) {
                        const int node = tail[slot][history_index];
                        const std::uint16_t expected =
                            node < 0
                                ? 0
                                : conv_fixture[(static_cast<std::size_t>(layer) * batch * width +
                                                node) *
                                                   10240 +
                                               channel];
                        if (actual_conv[(static_cast<std::size_t>(slot) * 3 + history_index) *
                                            10240 +
                                        channel] != expected) {
                            throw std::runtime_error(
                                "independent exact conv3 history oracle failed");
                        }
                    }
        }

        if (failures) { throw std::runtime_error("independent history cadence oracle failed"); }
    }

    void lifecycle() {
        const int restored_slot = slot_h.front();
        const int staging_slot  = batch;
        DeviceBuffer count_observers(static_cast<std::size_t>(batch) * 3 * sizeof(std::int32_t));
        DeviceBuffer continuation_queries(static_cast<std::size_t>(kLayers) * out_d.bytes);
        std::vector<ops::GdnReplayFoldRow> one(batch), abort(batch);
        for (int row = 0; row < batch; ++row) {
            one[row].linear_state_slot   = slot_h[row];
            one[row].commit_columns      = 1;
            one[row].path_length         = tree ? 1 : -1;
            one[row].path[0]             = 0;
            abort[row].linear_state_slot = slot_h[row];
            abort[row].commit_columns    = 0;
            abort[row].path_length       = tree ? 0 : -1;
        }
        const auto record_all = [&](cudaStream_t stream, bool collect) {
            for (int layer = 0; layer < kLayers; ++layer) {
                if (tree) {
                    candidate_record<true>(layer, stream);
                } else {
                    candidate_record<false>(layer, stream);
                }
                if (collect) {
                    CUDA_CHECK(cudaMemcpyAsync(static_cast<std::byte*>(continuation_queries.p) +
                                                   static_cast<std::size_t>(layer) * out_d.bytes,
                                               out_d.p, out_d.bytes, cudaMemcpyDeviceToDevice,
                                               stream));
                }
            }
        };
        const auto observe_counts = [&](cudaStream_t stream, int frame) {
            CUDA_CHECK(cudaMemcpyAsync(
                static_cast<std::byte*>(count_observers.p) +
                    static_cast<std::size_t>(frame) * batch * sizeof(std::int32_t),
                history.value().counts.data, static_cast<std::size_t>(batch) * sizeof(std::int32_t),
                cudaMemcpyDeviceToDevice, stream));
        };
        const auto transition = [&](cudaStream_t stream) {
            CUDA_CHECK(cudaMemcpyAsync(candidate_storage.p, initial_storage.p,
                                       candidate_storage.bytes, cudaMemcpyDeviceToDevice, stream));
            ops::gdn_history_reset_all(history.value(), stream);
            record_all(stream, false);
            commit(one, false, stream);
            observe_counts(stream, 0);
            ops::gdn_history_materialize(records, history.value(), candidate.all_layers_view(),
                                         slot_h, stream);
            candidate.copy_slot_2d(restored_slot, staging_slot, stream);
            candidate.zero_slot(restored_slot, stream);
            ops::gdn_history_reset_slot(history.value(), restored_slot, stream);
            candidate.copy_slot_2d(staging_slot, restored_slot, stream);
            ops::gdn_history_reset_slot(history.value(), restored_slot, stream);
            record_all(stream, false);
            commit(one, false, stream);
            record_all(stream, false);
            commit(abort, false, stream);
            observe_counts(stream, 1);
            // Cancel/discard a genuinely deferred slot, then restore its earlier dense prefix.
            // Peers keep their retained second anchor; the restored slot resumes from the first.
            candidate.zero_slot(restored_slot, stream);
            ops::gdn_history_reset_slot(history.value(), restored_slot, stream);
            candidate.copy_slot_2d(staging_slot, restored_slot, stream);
            ops::gdn_history_reset_slot(history.value(), restored_slot, stream);
            record_all(stream, true);
            commit(one, false, stream);
            observe_counts(stream, 2);
            ops::gdn_history_materialize(records, history.value(), candidate.all_layers_view(),
                                         slot_h, stream);
        };
        transition(device.stream);
        device.synchronize();
        TestGraph graph;
        graph.capture(device.stream, transition);
        graph.launch(device.stream);
        device.synchronize();
        graph.launch(device.stream);
        device.synchronize();
        const auto observed_counts =
            tr::from_device<std::int32_t>(count_observers, static_cast<std::size_t>(batch) * 3);
        for (int slot = 0; slot < batch; ++slot) {
            if (observed_counts[slot] != 1 || observed_counts[batch + slot] != 1 ||
                observed_counts[2 * batch + slot] != (slot == restored_slot ? 1 : 2)) {
                throw std::runtime_error(
                    "snapshot/abort/restore fixture did not preserve live history ownership");
            }
        }
        int failures           = 0;
        const auto query_words = tr::from_device<std::uint16_t>(
            continuation_queries, static_cast<std::size_t>(kLayers) * value_elements);
        for (int record_row = 0; record_row < batch; ++record_row) {
            const int slot = slot_h[record_row];
            std::vector<double> initial(state_slot_elements);
            for (std::size_t index = 0; index < state_slot_elements; ++index) {
                initial[index] =
                    initial_state[static_cast<std::size_t>(slot) * state_slot_elements + index];
            }
            const std::vector<double> zero(state_slot_elements, 0.0);
            const auto base_first =
                tr::history_ref::evaluate(oracle_input(record_row, {0}, initial), kScale, true);
            const auto zero_first =
                tr::history_ref::evaluate(oracle_input(record_row, {0}, zero), kScale, true);
            const int steps = slot == restored_slot ? 2 : 3;
            const std::vector<int> columns(static_cast<std::size_t>(steps), 0);
            const auto base_final =
                tr::history_ref::evaluate(oracle_input(record_row, columns, initial), kScale, true);
            const auto zero_continuation =
                tr::history_ref::evaluate(oracle_input(record_row, columns, zero), kScale, true);
            for (int layer = 0; layer < kLayers; ++layer) {
                const double scale = layer + 1;
                std::vector<double> expected_state(state_slot_elements),
                    expected_first(state_slot_elements);
                for (std::size_t index = 0; index < state_slot_elements; ++index) {
                    expected_state[index] = scale * base_final.final_state[index] +
                                            (1.0 - scale) * zero_continuation.final_state[index];
                    expected_first[index] = scale * base_first.final_state[index] +
                                            (1.0 - scale) * zero_first.final_state[index];
                }
                std::vector<float> state(state_slot_elements);
                CUDA_CHECK(cudaMemcpy(state.data(), candidate.recurrent_slot(layer, slot).data,
                                      state.size() * sizeof(float), cudaMemcpyDeviceToHost));
                const std::vector<double> actual_state(state.begin(), state.end());
                failures +=
                    tr::verify_reduction("history restored/cancelled continuation FP64 state",
                                         actual_state, expected_state, {.0027, 1e-5, .0039});
                if (slot == restored_slot) {
                    CUDA_CHECK(cudaMemcpy(state.data(),
                                          candidate.recurrent_slot(layer, staging_slot).data,
                                          state.size() * sizeof(float), cudaMemcpyDeviceToHost));
                    const std::vector<double> snapshot_state(state.begin(), state.end());
                    failures +=
                        tr::verify_reduction("history materialized snapshot FP64 state",
                                             snapshot_state, expected_first, {.0027, 1e-5, .0039});
                }
                std::vector<double> query(static_cast<std::size_t>(kHv) * kD),
                    expected_query(query.size());
                const auto query_base     = static_cast<std::size_t>(layer) * value_elements +
                                            static_cast<std::size_t>(record_row) * width * kHv * kD;
                const auto reference_base = static_cast<std::size_t>(steps - 1) * kHv * kD;
                for (std::size_t index = 0; index < query.size(); ++index) {
                    query[index] = tr::bf16_to_f32(query_words[query_base + index]);
                    expected_query[index] =
                        scale * base_final.out[reference_base + index] +
                        (1.0 - scale) * zero_continuation.out[reference_base + index];
                }
                failures +=
                    tr::verify_reduction("history restored/cancelled continuation FP64 readout",
                                         query, expected_query, {.0041, 5e-6, .0055});
                std::vector<std::uint16_t> conv(3 * 10240);
                CUDA_CHECK(cudaMemcpy(conv.data(), candidate.conv_slot(layer, slot).data,
                                      conv.size() * sizeof(std::uint16_t), cudaMemcpyDeviceToHost));
                for (int tail = 0; tail < 3; ++tail)
                    for (int channel = 0; channel < 10240; ++channel) {
                        const auto source =
                            (static_cast<std::size_t>(layer) * batch + record_row) * width * 10240 +
                            channel;
                        const auto expected =
                            tail < 3 - steps ? std::uint16_t{0} : conv_fixture[source];
                        if (conv[static_cast<std::size_t>(tail) * 10240 + channel] != expected) {
                            throw std::runtime_error(
                                "history snapshot/abort/restore exact conv3 criterion failed");
                        }
                    }
            }
        }
        if (failures) {
            throw std::runtime_error("history lifecycle mathematical qualification failed");
        }
    }

    void stride_compaction() {
        constexpr int active_rows = 4;
        const std::array<std::int32_t, active_rows> selected_slots{5, 1, 4, 0};
        if (batch != 6 || width != 12 || tree) {
            throw std::logic_error("stride fixture requires C6 and ceiling12");
        }

        struct Step {
            int tokens;
            DeviceBuffer input_q, input_k, input_v, input_g, input_beta, valid_storage;
            DeviceBuffer raw_key, raw_value, raw_gate, output_storage, query_observers;
            Tensor query, key, value, gate, beta_control, valid_columns;
            Tensor key_record, value_record, gate_record, output;

            Step(AcceptedHistoryFixture& owner, int token_count) : tokens(token_count) {
                const auto bf16_plane = [&](const std::vector<float>& source, int heads) {
                    const auto row_values = static_cast<std::size_t>(tokens) * heads * kD;
                    std::vector<std::uint16_t> words(active_rows * row_values);
                    for (int row = 0; row < active_rows; ++row)
                        for (std::size_t index = 0; index < row_values; ++index) {
                            words[static_cast<std::size_t>(row) * row_values + index] =
                                tr::f32_to_bf16(source[static_cast<std::size_t>(row) * owner.width *
                                                           heads * kD +
                                                       index]);
                        }
                    return tr::to_device(words);
                };
                const auto control_plane = [&](const std::vector<float>& source) {
                    const auto row_values = static_cast<std::size_t>(tokens) * kHv;
                    std::vector<float> values(active_rows * row_values);
                    for (int row = 0; row < active_rows; ++row) {
                        std::copy_n(source.begin() +
                                        static_cast<std::size_t>(row) * owner.width * kHv,
                                    row_values,
                                    values.begin() + static_cast<std::size_t>(row) * row_values);
                    }
                    return tr::to_device(values);
                };
                input_q       = bf16_plane(owner.qh, kHq);
                input_k       = bf16_plane(owner.kh, kHq);
                input_v       = bf16_plane(owner.vh, kHv);
                input_g       = control_plane(owner.gh);
                input_beta    = control_plane(owner.bh);
                valid_storage = tr::to_device(std::vector<std::int32_t>(active_rows, tokens));
                raw_key =
                    DeviceBuffer(static_cast<std::size_t>(active_rows) * tokens * kHq * kD * 2);
                raw_value =
                    DeviceBuffer(static_cast<std::size_t>(active_rows) * tokens * kHv * kD * 2);
                raw_gate = DeviceBuffer(static_cast<std::size_t>(active_rows) * tokens * kHv * 8);
                output_storage  = DeviceBuffer(raw_value.bytes);
                query_observers = DeviceBuffer(kLayers * output_storage.bytes);
                query           = Tensor(input_q.p, DType::BF16, {kD, kHq, tokens, active_rows});
                key             = Tensor(input_k.p, DType::BF16, {kD, kHq, tokens, active_rows});
                value           = Tensor(input_v.p, DType::BF16, {kD, kHv, tokens, active_rows});
                gate            = Tensor(input_g.p, DType::FP32, {kHv, tokens, active_rows});
                beta_control    = Tensor(input_beta.p, DType::FP32, {kHv, tokens, active_rows});
                valid_columns   = Tensor(valid_storage.p, DType::I32, {active_rows});
                key_record      = Tensor(raw_key.p, DType::BF16, {kD, kHq, tokens, active_rows});
                value_record    = Tensor(raw_value.p, DType::BF16, {kD, kHv, tokens, active_rows});
                gate_record     = Tensor(raw_gate.p, DType::FP32, {2, kHv, tokens, active_rows});
                output = Tensor(output_storage.p, DType::BF16, {kD, kHv, tokens, active_rows});
            }
        };

        Step first(*this, 4), second(*this, 8);
        const std::array<Step*, 2> steps{&first, &second};
        DeviceBuffer selected_storage =
            tr::to_device(std::vector<std::int32_t>(selected_slots.begin(), selected_slots.end()));
        const Tensor selected_tensor(selected_storage.p, DType::I32, {active_rows});
        DeviceBuffer count_observers(3 * 6 * sizeof(std::int32_t));
        CUDA_CHECK(cudaDeviceSynchronize());
        using RowReferences = std::array<std::vector<double>, active_rows>;
        std::array<RowReferences, 2> base_queries, zero_queries;
        RowReferences base_state, zero_state;
        for (int row = 0; row < active_rows; ++row) {
            const auto offset = static_cast<std::size_t>(selected_slots[row]) * state_slot_elements;
            base_state[row].assign(initial_state.begin() + offset,
                                   initial_state.begin() + offset + state_slot_elements);
            zero_state[row].assign(state_slot_elements, 0.0);
            for (std::size_t step_index = 0; step_index < steps.size(); ++step_index) {
                std::vector<int> columns(static_cast<std::size_t>(steps[step_index]->tokens));
                std::iota(columns.begin(), columns.end(), 0);
                const auto base_query = tr::history_ref::evaluate(
                    oracle_input(row, columns, base_state[row]), kScale, true);
                const auto zero_query = tr::history_ref::evaluate(
                    oracle_input(row, columns, zero_state[row]), kScale, true);
                base_queries[step_index][row] = base_query.out;
                zero_queries[step_index][row] = zero_query.out;
                columns.resize(step_index + 1);
                base_state[row] = tr::history_ref::evaluate(
                                      oracle_input(row, columns, base_state[row]), kScale, true)
                                      .final_state;
                zero_state[row] = tr::history_ref::evaluate(
                                      oracle_input(row, columns, zero_state[row]), kScale, true)
                                      .final_state;
            }
        }
        const auto observe_counts = [&](cudaStream_t stream, std::size_t frame) {
            CUDA_CHECK(cudaMemcpyAsync(static_cast<std::byte*>(count_observers.p) +
                                           frame * 6 * sizeof(std::int32_t),
                                       history.value().counts.data, 6 * sizeof(std::int32_t),
                                       cudaMemcpyDeviceToDevice, stream));
        };
        const auto execute = [&](cudaStream_t stream) {
            CUDA_CHECK(cudaMemcpyAsync(candidate_storage.p, initial_storage.p,
                                       candidate_storage.bytes, cudaMemcpyDeviceToDevice, stream));
            ops::gdn_history_reset_all(history.value(), stream);
            for (std::size_t step_index = 0; step_index < steps.size(); ++step_index) {
                Step& step = *steps[step_index];
                for (int layer = 0; layer < kLayers; ++layer) {
                    for (const int row_begin : {0, 2}) {
                        Tensor raw_keys     = step.key_record.slice(3, row_begin, 2);
                        Tensor raw_values   = step.value_record.slice(3, row_begin, 2);
                        Tensor raw_controls = step.gate_record.slice(3, row_begin, 2);
                        Tensor outputs      = step.output.slice(3, row_begin, 2);
                        ops::gated_delta_net_history_record(
                            step.query.slice(3, row_begin, 2), step.key.slice(3, row_begin, 2),
                            step.value.slice(3, row_begin, 2), step.gate.slice(2, row_begin, 2),
                            step.beta_control.slice(2, row_begin, 2), kScale,
                            candidate.recurrent[layer], step.valid_columns.slice(0, row_begin, 2),
                            selected_tensor.slice(0, row_begin, 2), raw_keys, raw_values,
                            raw_controls, outputs, history.value().layer(layer, row_begin, 2),
                            stream);
                    }
                    const auto persistent = records.layer(layer, active_rows);
                    const auto pack       = [&](const Tensor& live, const Tensor& destination,
                                                std::size_t bytes_per_token) {
                        const auto source_pitch =
                            static_cast<std::size_t>(step.tokens) * bytes_per_token;
                        CUDA_CHECK(cudaMemcpy2DAsync(
                            destination.data, 12 * bytes_per_token, live.data, source_pitch,
                            source_pitch, active_rows, cudaMemcpyDeviceToDevice, stream));
                    };
                    pack(step.key_record, persistent.key, kHq * kD * 2);
                    pack(step.value_record, persistent.value, kHv * kD * 2);
                    pack(step.gate_record, persistent.gate, kHv * 8);
                    CUDA_CHECK(cudaMemcpyAsync(static_cast<std::byte*>(step.query_observers.p) +
                                                   static_cast<std::size_t>(layer) *
                                                       step.output_storage.bytes,
                                               step.output_storage.p, step.output_storage.bytes,
                                               cudaMemcpyDeviceToDevice, stream));
                }
                std::array<ops::GdnReplayFoldRow, active_rows> rows{};
                for (int row = 0; row < active_rows; ++row) {
                    rows[row].linear_state_slot = selected_slots[row];
                    rows[row].commit_columns    = static_cast<std::int32_t>(step_index + 1);
                    rows[row].path_length       = -1;
                }
                ops::gdn_history_commit(records, history.value(), candidate.all_layers_view(), rows,
                                        stream);
                observe_counts(stream, step_index);
            }
            ops::gdn_history_materialize(records, history.value(), candidate.all_layers_view(),
                                         selected_slots, stream);
            observe_counts(stream, 2);
        };
        const auto verify = [&] {
            const auto counts = tr::from_device<std::int32_t>(count_observers, 18);
            for (int slot = 0; slot < 6; ++slot) {
                const bool active = std::find(selected_slots.begin(), selected_slots.end(), slot) !=
                                    selected_slots.end();
                if (counts[slot] != (active ? 1 : 0) || counts[6 + slot] != (active ? 3 : 0) ||
                    counts[12 + slot] != 0) {
                    throw std::runtime_error("C6 compact history count/ownership criterion failed");
                }
            }
            int failures = 0;
            for (std::size_t step_index = 0; step_index < steps.size(); ++step_index) {
                const Step& step      = *steps[step_index];
                const auto row_values = static_cast<std::size_t>(step.tokens) * kHv * kD;
                const auto words      = tr::from_device<std::uint16_t>(
                    step.query_observers,
                    static_cast<std::size_t>(kLayers) * active_rows * row_values);
                for (int layer = 0; layer < kLayers; ++layer)
                    for (int row = 0; row < active_rows; ++row) {
                        std::vector<double> actual(row_values), reference(row_values);
                        for (std::size_t index = 0; index < row_values; ++index) {
                            actual[index] = tr::bf16_to_f32(
                                words[(static_cast<std::size_t>(layer) * active_rows + row) *
                                          row_values +
                                      index]);
                            reference[index] = zero_queries[step_index][row][index] +
                                               (layer + 1) * (base_queries[step_index][row][index] -
                                                              zero_queries[step_index][row][index]);
                        }
                        failures += tr::verify_reduction("C6 ceiling12 compact W4/W8 FP64 readout",
                                                         actual, reference, {.0041, 5e-6, .0055});
                    }
            }
            for (int layer = 0; layer < kLayers; ++layer) {
                std::vector<float> states(static_cast<std::size_t>(7) * state_slot_elements);
                CUDA_CHECK(cudaMemcpy(states.data(), candidate.recurrent[layer].data,
                                      states.size() * sizeof(float), cudaMemcpyDeviceToHost));
                for (int slot = 0; slot < 7; ++slot) {
                    const auto selected =
                        std::find(selected_slots.begin(), selected_slots.end(), slot);
                    const auto offset = static_cast<std::size_t>(slot) * state_slot_elements;
                    if (selected == selected_slots.end()) {
                        for (std::size_t index = 0; index < state_slot_elements; ++index) {
                            const float expected =
                                slot == 6 ? 0.F : initial_state[offset + index] * (layer + 1);
                            if (states[offset + index] != expected) {
                                throw std::runtime_error(
                                    "C6 compact history modified an inactive or staging slot");
                            }
                        }
                    } else {
                        const auto row =
                            static_cast<std::size_t>(selected - selected_slots.begin());
                        std::vector<double> actual(state_slot_elements),
                            reference(state_slot_elements);
                        for (std::size_t index = 0; index < state_slot_elements; ++index) {
                            actual[index] = states[offset + index];
                            reference[index] =
                                zero_state[row][index] +
                                (layer + 1) * (base_state[row][index] - zero_state[row][index]);
                        }
                        failures +=
                            tr::verify_reduction("C6 compact history FP64 materialized state",
                                                 actual, reference, {.0027, 1e-5, .0039});
                        std::vector<std::uint16_t> conv(3 * 10240);
                        CUDA_CHECK(cudaMemcpy(conv.data(), candidate.conv_slot(layer, slot).data,
                                              conv.size() * sizeof(std::uint16_t),
                                              cudaMemcpyDeviceToHost));
                        for (int tail = 0; tail < 3; ++tail)
                            for (int channel = 0; channel < 10240; ++channel) {
                                const auto source =
                                    ((static_cast<std::size_t>(layer) * 6 + row) * 12 +
                                     (tail == 2 ? 1 : 0)) *
                                        10240 +
                                    channel;
                                if (conv[static_cast<std::size_t>(tail) * 10240 + channel] !=
                                    conv_fixture[source]) {
                                    throw std::runtime_error(
                                        "C6 compact exact conv3 criterion failed");
                                }
                            }
                    }
                }
                const auto raw        = records.layer(layer, active_rows);
                const auto check_bf16 = [&](const Tensor& tensor, const std::vector<float>& input,
                                            int heads) {
                    std::vector<std::uint16_t> words(static_cast<std::size_t>(tensor.numel()));
                    CUDA_CHECK(cudaMemcpy(words.data(), tensor.data, tensor.bytes(),
                                          cudaMemcpyDeviceToHost));
                    const auto token_values = static_cast<std::size_t>(heads) * kD;
                    for (int row = 0; row < active_rows; ++row)
                        for (int token = 0; token < 8; ++token)
                            for (std::size_t index = 0; index < token_values; ++index) {
                                const auto position =
                                    (static_cast<std::size_t>(row) * 12 + token) * token_values +
                                    index;
                                if (words[position] != tr::f32_to_bf16(input[position])) {
                                    throw std::runtime_error(
                                        "C6 compact exact raw BF16 criterion failed");
                                }
                            }
                };
                check_bf16(raw.key, kh, kHq);
                check_bf16(raw.value, vh, kHv);
                std::vector<float> controls(static_cast<std::size_t>(raw.gate.numel()));
                CUDA_CHECK(cudaMemcpy(controls.data(), raw.gate.data, raw.gate.bytes(),
                                      cudaMemcpyDeviceToHost));
                for (int row = 0; row < active_rows; ++row)
                    for (int token = 0; token < 8; ++token)
                        for (int head = 0; head < kHv; ++head) {
                            const auto index =
                                (static_cast<std::size_t>(row) * 12 + token) * kHv + head;
                            if (controls[2 * index] != gh[index] ||
                                controls[2 * index + 1] != bh[index]) {
                                throw std::runtime_error(
                                    "C6 compact exact raw controls criterion failed");
                            }
                        }
            }
            if (failures) {
                throw std::runtime_error("compact stride mathematical qualification failed");
            }
        };
        execute(device.stream);
        device.synchronize();
        verify();
        TestGraph graph;
        graph.capture(device.stream, execute);
        graph.launch(device.stream);
        device.synchronize();
        verify();
        graph.launch(device.stream);
        device.synchronize();
        verify();
    }
};

int main(int argc, char** argv) {
    try {
        if (argc == 2 && std::string_view(argv[1]) == "--racecheck") {
            for (const bool tree : {false, true}) {
                AcceptedHistoryFixture fixture(1, tree ? 12 : 4, tree, 4, false);
                fixture.qualify_record_racecheck();
            }
            return 0;
        }
        if (tr::sanitizer_scope(argc, argv)) {
            {
                AcceptedHistoryFixture fixture(1, 4, false, 4);
                fixture.qualify(true, 2);
            }
            {
                AcceptedHistoryFixture fixture(1, 12, true, 4);
                fixture.qualify(true, 2);
            }
            return 0;
        }
        if (argc == 2 && std::string_view(argv[1]) == "--stride") {
            AcceptedHistoryFixture fixture(6, 12, false, 8, false);
            fixture.stride_compaction();
            return 0;
        }
        const int batch = argc > 1 ? std::stoi(argv[1]) : 1;
        const int width = argc > 2 ? std::stoi(argv[2]) : 4;
        const bool tree = argc > 3 && std::stoi(argv[3]) != 0;
        if ((batch != 1 && batch != 6) || (width != 4 && width != 8 && width != 12)) {
            throw std::invalid_argument("usage: gdn_history_test {1|6} {4|8|12} [tree0|1]");
        }
        {
            AcceptedHistoryFixture fixture(batch, width, tree);
            fixture.qualify(false);
            fixture.qualify(true);
            fixture.lifecycle();
        }
        if (argc == 1) {
            AcceptedHistoryFixture fixture(6, 12, false, 8, false);
            fixture.stride_compaction();
        }
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
