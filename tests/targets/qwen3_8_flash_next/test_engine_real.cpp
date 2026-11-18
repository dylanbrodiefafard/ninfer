// qwen3.8-flash-next through the public Engine on the real artifact and n-gram table.
//
//   --program-parity  raw greedy Engine output equals the target Program's own prefill and decode
//                     on the same prompt and capacity (a two-super-chunk prompt);
//   (default)         four concurrent requests preserve their first tokens and share decode
//                     rounds; capacity-limited requests wait and equal their solo runs; a short
//                     prompt interrupts a long prefill between super-chunks; cancellation releases
//                     lanes and leaves later requests unchanged.
//
// B1 requests with identical execution geometry are compared exactly. B1/B4 arithmetic profiles
// can differ in routing and reduction order. Their numerical comparison therefore uses identical
// per-position histories, not free-running answers: full next-token distributions on B1 teacher
// trajectories, and selected-token logprobs/top-1 on each public Engine request's published
// history. The latter also checks post-prefill request/state routing.

#include "ninfer/engine.h"

#include "artifact/reader.h"
#include "core/device.h"
#include "load/binder.h"
#include "runtime/program.h"
#include "ninfer/ops/hyper_connection.h"
#include "ninfer/ops/linear.h"
#include <ninfer/targets/qwen3_6/prepared_prompt.h>
#include <ninfer/targets/qwen3_8_flash_next/package.h>

#include <nlohmann/json.hpp>

#include <algorithm>
#include <array>
#include <bit>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <fstream>
#include <filesystem>
#include <iostream>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <thread>
#include <utility>
#include <vector>

namespace {

namespace fn = ninfer::targets::qwen3_8_flash_next;
using Tokens = std::vector<ninfer::TokenId>;
using Clock  = std::chrono::steady_clock;

struct Inputs {
    const char* artifact    = nullptr;
    const char* ngram_table = nullptr;
    ninfer::SpeculativeOptions speculative;
};

void require(bool condition, const std::string& message) {
    if (!condition) { throw std::runtime_error(message); }
}

// Deterministic in-vocabulary tokens; the content only needs to differ between prompts.
Tokens prompt_tokens(std::uint32_t count, std::uint32_t seed) {
    Tokens tokens(count);
    std::uint32_t state = seed;
    for (auto& token : tokens) {
        state = state * 1664525U + 1013904223U;
        token = static_cast<ninfer::TokenId>(1000U + (state >> 8) % 60000U);
    }
    return tokens;
}

ninfer::RequestOptions greedy(std::uint32_t outputs) {
    ninfer::RequestOptions options;
    options.execution.requested_output_tokens = outputs;
    options.execution.sampling.temperature    = 0.0F;
    options.execution.allow_prefix_reuse      = false;
    options.stop.include_model_defaults       = false;
    return options;
}

ninfer::EngineOptions engine_options(const Inputs& inputs, std::uint32_t lanes,
                                     std::uint32_t context, std::uint32_t kv_tokens) {
    ninfer::EngineOptions options;
    options.artifact_path    = inputs.artifact;
    options.ngram_table_path = inputs.ngram_table;
    options.speculative      = inputs.speculative;
    options.max_concurrency  = lanes;
    options.max_context      = context;
    options.kv_capacity      = ninfer::KvCapacityPolicy::explicit_capacity(kv_tokens);
    return options;
}

ninfer::GenerationResult run(ninfer::Engine& engine, const Tokens& prompt, std::uint32_t outputs) {
    return engine.generate(engine.prepare_tokens(prompt, false), greedy(outputs));
}

void require_equal(const Tokens& actual, const Tokens& expected, const std::string& label) {
    if (actual == expected) { return; }
    const auto mismatch =
        std::mismatch(actual.begin(), actual.end(), expected.begin(), expected.end());
    throw std::runtime_error(label + ": " + std::to_string(actual.size()) + " tokens against " +
                             std::to_string(expected.size()) + ", first difference at " +
                             std::to_string(mismatch.first - actual.begin()));
}

// --program-parity --------------------------------------------------------------------------

constexpr std::uint32_t kParityContext = 20480;
constexpr std::uint32_t kParityPrompt  = 17000; // two super-chunks
constexpr std::uint32_t kParityOutputs = 64;

Tokens program_greedy(const Inputs& inputs, const Tokens& prompt) {
    ninfer::DeviceContext device(0);
    const ninfer::artifact::Reader reader(inputs.artifact);
    const fn::LoadedModel model(reader, device);
    fn::ProgramOptions options;
    options.ngram_table        = inputs.ngram_table;
    options.context_tokens     = static_cast<std::int32_t>(kParityContext);
    options.lanes              = 1;
    options.kv_capacity_tokens = kParityContext;
    fn::Program program(model, options, device);

    const std::optional<fn::Lane> lane = program.open_lane(
        static_cast<std::int64_t>(prompt.size()) + kParityOutputs - 1, fn::LaneSampling{});
    require(lane.has_value(), "Program refused the parity lane");
    const fn::Lane parity_lane = lane.value();
    Tokens out;
    std::span<const ninfer::TokenId> remaining(prompt);
    while (!remaining.empty()) {
        const fn::PrefillStep step = program.prefill_step(parity_lane, remaining);
        remaining                  = remaining.subspan(static_cast<std::size_t>(step.consumed));
        if (step.complete) { out.push_back(step.next_token); }
    }
    require(out.size() == 1, "Program prefill did not sample exactly one token");
    while (out.size() < kParityOutputs) {
        const std::array<fn::Lane, 1> lanes{parity_lane};
        const std::array<ninfer::TokenId, 1> tokens{out.back()};
        std::array<ninfer::TokenId, 1> next{};
        program.decode(lanes, tokens, next);
        out.push_back(next[0]);
    }
    program.close_lane(parity_lane);
    return out;
}

void program_parity(const Inputs& inputs) {
    const Tokens prompt   = prompt_tokens(kParityPrompt, 7);
    const Tokens expected = program_greedy(inputs, prompt);
    ninfer::Engine engine(engine_options(inputs, 1, kParityContext, kParityContext));
    auto request = greedy(kParityOutputs);
    // Native parity uses the same unconstrained sampling domain as the Program.
    request.output.raw = true;
    const auto result  = engine.generate(engine.prepare_tokens(prompt, false), request);
    require(result.finish_reason == ninfer::FinishReason::OutputLimit,
            "parity request did not reach its output limit");
    require_equal(result.generated_token_ids, expected, "Engine against Program greedy decode");
    std::cout << "program parity: " << expected.size() << " tokens equal after a " << prompt.size()
              << "-token prompt\n";
}

// --reuse-parity -----------------------------------------------------------------------------

std::vector<std::uint16_t> logits(fn::Program& program, fn::Lane lane) {
    const auto tensor = program.state().tail_logits(lane);
    std::vector<std::uint16_t> values(static_cast<std::size_t>(fn::kTokenDomain));
    CUDA_CHECK(cudaMemcpy(values.data(), tensor.data, values.size() * sizeof(std::uint16_t),
                          cudaMemcpyDeviceToHost));
    return values;
}

float bf16(std::uint16_t value) {
    const std::uint32_t bits = static_cast<std::uint32_t>(value) << 16;
    float out                = 0.0F;
    std::memcpy(&out, &bits, sizeof(out));
    return out;
}

struct DistributionDifference {
    bool top1_equal = false;
    double kl       = 0.0;
    // Expected minus actual log-sum-exp, for selected-token logprob comparisons.
    double log_normalizer_delta = 0.0;
};

struct LogitSummary {
    std::size_t top1;
    double log_normalizer;
};

LogitSummary summarize_logits(std::span<const std::uint16_t> values) {
    const auto less = [](std::uint16_t a, std::uint16_t b) { return bf16(a) < bf16(b); };
    const auto top = static_cast<std::size_t>(std::max_element(values.begin(), values.end(), less) -
                                              values.begin());
    const double maximum = bf16(values[top]);
    double sum           = 0.0;
    for (const auto value : values) { sum += std::exp(bf16(value) - maximum); }
    return {.top1 = top, .log_normalizer = maximum + std::log(sum)};
}

DistributionDifference distribution_difference(std::span<const std::uint16_t> actual,
                                               std::span<const std::uint16_t> expected) {
    const auto actual_summary   = summarize_logits(actual);
    const auto expected_summary = summarize_logits(expected);
    const double log_a          = actual_summary.log_normalizer;
    const double log_e          = expected_summary.log_normalizer;
    double kl                   = 0.0;
    for (std::size_t i = 0; i < actual.size(); ++i) {
        const double e = bf16(expected[i]) - log_e;
        kl += std::exp(e) * (e - (bf16(actual[i]) - log_a));
    }
    return {.top1_equal           = actual_summary.top1 == expected_summary.top1,
            .kl                   = kl,
            .log_normalizer_delta = log_e - log_a};
}

struct VerifyRouteErrors {
    std::size_t mismatches = 0;
    std::vector<double> kl;

    void add(const DistributionDifference& difference) {
        require(std::isfinite(difference.kl), "nonfinite route KL");
        mismatches += difference.top1_equal ? 0U : 1U;
        kl.push_back(std::max(0.0, difference.kl));
    }

    [[nodiscard]] double quantile(double fraction) const {
        auto sorted = kl;
        std::sort(sorted.begin(), sorted.end());
        const auto index =
            static_cast<std::size_t>(fraction * static_cast<double>(sorted.size() - 1));
        return sorted[index];
    }
};

void verify_parity(const Inputs& inputs) {
    ninfer::DeviceContext device(0);
    const ninfer::artifact::Reader reader(inputs.artifact);
    const fn::LoadedModel model(reader, device);
    fn::ProgramOptions options;
    options.ngram_table              = inputs.ngram_table;
    options.context_tokens           = 8192;
    options.lanes                    = 11;
    options.kv_capacity_tokens       = 11 * 8192;
    options.forward.max_verify_width = 4;
    fn::Program program(model, options, device);
    constexpr std::int32_t steps = 256;
    std::array<VerifyRouteErrors, 3> verify_errors;
    std::array<VerifyRouteErrors, 3> floor_errors;
    std::array<VerifyRouteErrors, 3> prefill_errors;
    ninfer::DeviceBuffer prefill_hidden(std::size_t{4} * fn::kHidden * sizeof(std::uint16_t));
    ninfer::DeviceBuffer prefill_logits(std::size_t{4} * fn::kVocabRows * sizeof(std::uint16_t));
    ninfer::WorkspaceArena prefill_work(ninfer::ops::hyper_connection_read_workspace_bytes(4));
    constexpr std::array<std::size_t, 3> floor_starts{0, 2, 5};
    for (const std::uint32_t seed : {73U, 901U}) {
        const Tokens prompt = prompt_tokens(seed == 73U ? 1024U : 4097U, seed);
        const auto open     = [&] {
            const auto lane = program.open_lane(static_cast<std::int64_t>(prompt.size()) + steps,
                                                fn::LaneSampling{});
            require(lane.has_value(), "verify parity lane was refused");
            std::span<const ninfer::TokenId> remaining(prompt);
            std::int32_t first = -1;
            while (!remaining.empty()) {
                const auto step = program.prefill_step(*lane, remaining);
                remaining       = remaining.subspan(static_cast<std::size_t>(step.consumed));
                if (step.complete) { first = step.next_token; }
            }
            require(first >= 0, "verify parity prefill did not sample");
            return std::pair{*lane, first};
        };
        const auto [reference_lane, first] = open();
        Tokens chain{first};
        std::vector<std::vector<std::uint16_t>> reference;
        reference.reserve(steps);
        for (std::int32_t step = 0; step < steps; ++step) {
            const std::array lanes{reference_lane};
            const std::array tokens{chain.back()};
            std::array<std::int32_t, 1> next{};
            program.decode(lanes, tokens, next);
            chain.push_back(next[0]);
            reference.push_back(logits(program, reference_lane));
        }
        program.close_lane(reference_lane);
        const auto [lane, repeated_first] = open();
        require(repeated_first == first, "verify parity cold prefill was not reproducible");
        std::array<fn::Lane, 9> floor_lanes{};
        for (auto& floor_lane : floor_lanes) {
            floor_lane = program.open_lane(8192, fn::LaneSampling{}).value();
        }
        const auto prefill_lane = program.open_lane(8192, fn::LaneSampling{}).value();
        Tokens prefix           = prompt;
        ninfer::PinnedHostBuffer before(program.state().continuation_bytes());
        ninfer::PinnedHostBuffer after(program.state().continuation_bytes());
        ninfer::PinnedHostBuffer committed(program.state().continuation_bytes());
        for (std::int32_t offset = 0; offset < steps; offset += 4) {
            const auto frontier = program.state().committed(lane);
            if (offset == 0) {
                program.state().pack_continuation(lane, before.data(), device.stream);
                CUDA_CHECK(cudaStreamSynchronize(device.stream));
            }
            if (offset == 0) {
                const std::array lanes{lane};
                const std::array<std::int32_t, 1> valid{1};
                const auto input = std::span<const std::int32_t>(chain).first(1);
                static_cast<void>(program.forward().verify(lanes, input, 1, valid));
                program.forward().commit_verify(valid);
                program.state().pack_continuation(lane, after.data(), device.stream);
                CUDA_CHECK(cudaStreamSynchronize(device.stream));
                program.reuse_lane(lane, static_cast<std::int64_t>(prompt.size()) + steps, prefix,
                                   fn::LaneSampling{}, before.data());
                static_cast<void>(program.finish_prompt(lane));
                std::array<std::int32_t, 1> next{};
                program.decode(lanes, input, next);
                program.state().pack_continuation(lane, committed.data(), device.stream);
                CUDA_CHECK(cudaStreamSynchronize(device.stream));
                require(std::memcmp(after.data(), committed.data(), after.size()) == 0,
                        "deferred root state differs from ordinary width-one decode");
                program.reuse_lane(lane, static_cast<std::int64_t>(prompt.size()) + steps, prefix,
                                   fn::LaneSampling{}, before.data());
                static_cast<void>(program.finish_prompt(lane));
            }
            for (std::int32_t width = 2; width <= 4; ++width) {
                const std::array lanes{lane};
                const std::array valid{width};
                const auto input = std::span<const std::int32_t>(chain).subspan(
                    static_cast<std::size_t>(offset), static_cast<std::size_t>(width));
                const auto actual = program.forward().verify(lanes, input, width, valid);
                std::vector<std::uint16_t> values(static_cast<std::size_t>(fn::kVocabRows) *
                                                  static_cast<std::size_t>(width));
                CUDA_CHECK(cudaMemcpy(values.data(), actual.data, values.size() * sizeof(values[0]),
                                      cudaMemcpyDeviceToHost));
                require(program.state().committed(lane) == frontier,
                        "verification advanced the logical frontier");
                if (offset == 0) {
                    program.state().pack_continuation(lane, after.data(), device.stream);
                    CUDA_CHECK(cudaStreamSynchronize(device.stream));
                    require(std::memcmp(before.data(), after.data(), before.size()) == 0,
                            "verification changed canonical GDN, PLE or tail logits");
                }
                const std::array<std::int32_t, 1> zero_commit{0};
                program.forward().commit_verify(zero_commit);
                require(program.state().committed(lane) == frontier,
                        "rejecting verification advanced the frontier");
                if (offset == 0) {
                    program.state().pack_continuation(lane, after.data(), device.stream);
                    CUDA_CHECK(cudaStreamSynchronize(device.stream));
                    require(std::memcmp(before.data(), after.data(), before.size()) == 0,
                            "zero commit changed canonical continuation");
                }
                const auto arm = static_cast<std::size_t>(width - 2);
                for (std::int32_t column = 0; column < width; ++column) {
                    const auto distribution = std::span<const std::uint16_t>(values).subspan(
                        static_cast<std::size_t>(column) * fn::kVocabRows, fn::kTokenDomain);
                    const auto difference = distribution_difference(
                        distribution, reference[static_cast<std::size_t>(offset) +
                                                static_cast<std::size_t>(column)]);
                    verify_errors[arm].add(difference);
                }
                program.state().copy_prefix(prefill_lane, lane, frontier, device.stream);
                program.reuse_lane(prefill_lane, 8192, prefix, fn::LaneSampling{});
                fn::ForwardObserver observer;
                observer.after_layer = [&](std::int32_t layer, const ninfer::Tensor& residual,
                                           std::int32_t) {
                    if (layer != fn::kLayers - 1) { return; }
                    // The ordinary prefill path supplies its final residual. Evaluate every
                    // column's public head formula; no verification buffers or records are read.
                    ninfer::Tensor hidden(prefill_hidden.p, ninfer::DType::BF16,
                                          {fn::kHidden, width});
                    ninfer::Tensor output(prefill_logits.p, ninfer::DType::BF16,
                                          {fn::kVocabRows, width});
                    ninfer::Tensor no_write;
                    ninfer::ops::hyper_connection_read(residual, model.view().final_connection,
                                                       hidden, no_write, prefill_work,
                                                       device.stream);
                    ninfer::ops::linear(hidden, model.view().head, output, device.stream);
                };
                program.forward().prefill(prefill_lane, input, {}, &observer);
                CUDA_CHECK(cudaMemcpy(values.data(), prefill_logits.p,
                                      values.size() * sizeof(values[0]), cudaMemcpyDeviceToHost));
                for (std::int32_t column = 0; column < width; ++column) {
                    const auto distribution = std::span<const std::uint16_t>(values).subspan(
                        static_cast<std::size_t>(column) * fn::kVocabRows, fn::kTokenDomain);
                    prefill_errors[arm].add(distribution_difference(
                        distribution, reference[static_cast<std::size_t>(offset) +
                                                static_cast<std::size_t>(column)]));
                }
                if (offset == 0) {
                    // The reference tail is the final column of its same-width head, rather
                    // than the ordinary Last-only prefill head's separate width-one profile.
                    const auto tail          = program.state().tail_logits(prefill_lane);
                    const auto* final_column = static_cast<const std::byte*>(prefill_logits.p) +
                                               static_cast<std::size_t>(width - 1) * tail.bytes();
                    CUDA_CHECK(cudaMemcpyAsync(tail.data, final_column, tail.bytes(),
                                               cudaMemcpyDeviceToDevice, device.stream));
                    program.state().pack_continuation(prefill_lane, committed.data(),
                                                      device.stream);
                    CUDA_CHECK(cudaStreamSynchronize(device.stream));
                    const auto require_stale_rejected = [&] {
                        bool rejected = false;
                        try {
                            program.forward().commit_verify(valid);
                        } catch (const std::logic_error&) { rejected = true; }
                        require(rejected, "a shared-logit writer left a stale verify frame live");
                    };
                    static_cast<void>(program.forward().verify(lanes, input, width, valid));
                    static_cast<void>(program.finish_prompt(prefill_lane));
                    require_stale_rejected();
                    static_cast<void>(program.forward().verify(lanes, input, width, valid));
                    static_cast<void>(program.forward().score(prefill_lane, input.first(2)));
                    require_stale_rejected();
                    static_cast<void>(program.forward().verify(lanes, input, width, valid));
                    program.forward().commit_verify(valid);
                    require(program.state().committed(lane) == frontier + width,
                            "accepted verification has an incorrect frontier");
                    program.state().pack_continuation(lane, after.data(), device.stream);
                    CUDA_CHECK(cudaStreamSynchronize(device.stream));
                    require(std::memcmp(committed.data(), after.data(), committed.size()) == 0,
                            "accepted state differs from ordinary same-width prefill");
                    program.reuse_lane(lane, static_cast<std::int64_t>(prompt.size()) + steps,
                                       prefix, fn::LaneSampling{}, before.data());
                    const auto restored = program.finish_prompt(lane);
                    require(restored.next_token == chain[static_cast<std::size_t>(offset)],
                            "accepted-state test did not restore its source");
                    const auto restore_source = [&] {
                        program.reuse_lane(lane, static_cast<std::int64_t>(prompt.size()) + steps,
                                           prefix, fn::LaneSampling{}, before.data());
                        static_cast<void>(program.finish_prompt(lane));
                    };
                    for (std::int32_t count = 1; count < width; ++count) {
                        const std::array prefix_count{count};
                        static_cast<void>(program.forward().verify(lanes, input, width, valid));
                        program.forward().commit_verify(prefix_count);
                        program.state().pack_continuation(lane, after.data(), device.stream);
                        CUDA_CHECK(cudaStreamSynchronize(device.stream));
                        restore_source();
                        // A non-speculative teacher-forced same-width route has only this
                        // prefix available and commits all of it. Invalid future columns must
                        // not affect the accepted canonical image.
                        static_cast<void>(
                            program.forward().verify(lanes, input, width, prefix_count));
                        program.forward().commit_verify(prefix_count);
                        program.state().pack_continuation(lane, committed.data(), device.stream);
                        CUDA_CHECK(cudaStreamSynchronize(device.stream));
                        require(std::memcmp(after.data(), committed.data(), after.size()) == 0,
                                "partial commit depends on rejected future columns");
                        restore_source();
                    }
                }
            }
            for (std::int32_t column = 0; column < 4; ++column) {
                const auto index =
                    static_cast<std::size_t>(offset) + static_cast<std::size_t>(column);
                // Each floor row starts from an independently committed width-one checkpoint
                // at its own logical position. No verification records become floor inputs.
                for (std::size_t arm = 0; arm < floor_starts.size(); ++arm) {
                    if (static_cast<std::size_t>(column) >= arm + 2) { continue; }
                    const auto destination =
                        floor_lanes[floor_starts[arm] + static_cast<std::size_t>(column)];
                    program.state().copy_prefix(destination, lane, program.committed(lane),
                                                device.stream);
                    program.reuse_lane(destination, 8192, prefix, fn::LaneSampling{});
                    const auto restored = program.finish_prompt(destination);
                    require(restored.next_token == chain[index],
                            "copied floor checkpoint changed its tail sampling");
                }
                const std::array lanes{lane};
                const std::array tokens{chain[index]};
                std::array<std::int32_t, 1> next{};
                program.decode(lanes, tokens, next);
                require(next[0] == chain[index + 1],
                        "ordinary decode changed after provisional chain verification");
                prefix.push_back(chain[index]);
            }
            for (std::size_t arm = 0; arm < floor_starts.size(); ++arm) {
                const auto width = arm + 2;
                const auto lanes =
                    std::span<const fn::Lane>(floor_lanes).subspan(floor_starts[arm], width);
                const auto tokens = std::span<const std::int32_t>(chain).subspan(
                    static_cast<std::size_t>(offset), width);
                std::array<std::int32_t, 4> next{};
                program.decode(lanes, tokens, std::span<std::int32_t>(next).first(width));
                for (std::size_t column = 0; column < width; ++column) {
                    floor_errors[arm].add(distribution_difference(
                        logits(program, lanes[column]),
                        reference[static_cast<std::size_t>(offset) + column]));
                }
            }
        }
        for (const auto floor_lane : floor_lanes) { program.close_lane(floor_lane); }
        program.close_lane(prefill_lane);
        program.close_lane(lane);
    }
    bool qualified = true;
    for (std::size_t arm = 0; arm < verify_errors.size(); ++arm) {
        const auto& candidate = verify_errors[arm];
        const auto& floor     = floor_errors[arm];
        const auto& prefill   = prefill_errors[arm];
        require(candidate.kl.size() == floor.kl.size() && floor.kl.size() == prefill.kl.size(),
                "floor columns differ from verify columns");
        const auto count = candidate.kl.size();
        // This fixed-panel comparison is empirical, not a confidence claim about correlated
        // generated tokens. Central, high-tail and maximum KL must all fit the measured floor.
        const bool top1_pass =
            candidate.mismatches <= std::max(floor.mismatches, prefill.mismatches);
        bool kl_pass = true;
        for (const double quantile : {0.5, 0.95, 0.99, 1.0}) {
            kl_pass = kl_pass &&
                      candidate.quantile(quantile) <=
                          std::max(floor.quantile(quantile), prefill.quantile(quantile)) + 1e-12;
        }
        qualified = qualified && top1_pass && kl_pass;
        std::cout << "verify width=" << arm + 2 << " top1=" << count - candidate.mismatches << '/'
                  << count << " floor_top1=" << count - floor.mismatches << '/' << count
                  << " KL_p50/p95/p99/max=" << candidate.quantile(0.5) << '/'
                  << candidate.quantile(0.95) << '/' << candidate.quantile(0.99) << '/'
                  << candidate.quantile(1.0) << " floor_KL_p50/p95/p99/max=" << floor.quantile(0.5)
                  << '/' << floor.quantile(0.95) << '/' << floor.quantile(0.99) << '/'
                  << floor.quantile(1.0) << " prefill_top1=" << count - prefill.mismatches << '/'
                  << count << " prefill_KL_p50/p95/p99/max=" << prefill.quantile(0.5) << '/'
                  << prefill.quantile(0.95) << '/' << prefill.quantile(0.99) << '/'
                  << prefill.quantile(1.0) << " top1_pass=" << top1_pass << " KL_pass=" << kl_pass
                  << '\n';
    }
    require(qualified, "packed verification exceeds the measured same-column numerical floor");
}

#include "dflash_program_cases.h"

void reuse_parity(const Inputs& inputs) {
    ninfer::DeviceContext device(0);
    const ninfer::artifact::Reader reader(inputs.artifact);
    const fn::LoadedModel model(reader, device);
    fn::ProgramOptions options;
    options.ngram_table    = inputs.ngram_table;
    options.context_tokens = 18000;
    fn::Program program(model, options, device);
    ninfer::PinnedHostBuffer checkpoint(program.state().continuation_bytes());
    ninfer::PinnedHostBuffer verify(program.state().continuation_bytes());
    std::size_t matching_top1 = 0;
    std::vector<double> floor_kl;
    for (const std::uint32_t seed : {73U, 901U}) {
        const Tokens tokens = prompt_tokens(17500, seed);
        for (const std::uint32_t split : {63U, 64U, 65U, 4097U, 16385U}) {
            const auto end  = split + 71U;
            const auto lane = program.open_lane(end + 8U, fn::LaneSampling{}).value();
            auto remaining  = std::span(tokens).first(end);
            while (!remaining.empty()) {
                const auto step = program.prefill_step(lane, remaining);
                remaining       = remaining.subspan(static_cast<std::size_t>(step.consumed));
            }
            std::vector<std::vector<std::uint16_t>> cold{logits(program, lane)};
            const std::array<fn::Lane, 1> lanes{lane};
            for (std::uint32_t i = 0; i < 4; ++i) {
                const std::array<ninfer::TokenId, 1> input{tokens[end + i]};
                std::array<ninfer::TokenId, 1> next{};
                program.decode(lanes, input, next);
                cold.push_back(logits(program, lane));
            }
            program.close_lane(lane);

            // An independent cold route starts empty and shifts its first execution boundary, with
            // no completed request, reuse_lane call or continuation transport. This measures the
            // numerical floor of GDN/sub-chunk partitioning on exactly the reference inputs.
            const auto shifted_lane  = program.open_lane(end + 8U, fn::LaneSampling{}).value();
            remaining                = std::span(tokens).first(end);
            std::int32_t prefix_left = static_cast<std::int32_t>(split);
            while (prefix_left > 0) {
                const auto step = program.prefill_step(shifted_lane, remaining, prefix_left);
                require(step.consumed > 0 && step.consumed <= prefix_left && !step.complete,
                        "shifted cold route did not stop at its first boundary");
                prefix_left -= step.consumed;
                remaining = remaining.subspan(static_cast<std::size_t>(step.consumed));
            }
            while (!remaining.empty()) {
                const auto step = program.prefill_step(shifted_lane, remaining);
                remaining       = remaining.subspan(static_cast<std::size_t>(step.consumed));
            }
            std::vector<std::vector<std::uint16_t>> shifted{logits(program, shifted_lane)};
            const std::array<fn::Lane, 1> shifted_lanes{shifted_lane};
            for (std::uint32_t i = 0; i < 4; ++i) {
                const std::array<ninfer::TokenId, 1> input{tokens[end + i]};
                std::array<ninfer::TokenId, 1> next{};
                program.decode(shifted_lanes, input, next);
                shifted.push_back(logits(program, shifted_lane));
            }
            program.close_lane(shifted_lane);
            const auto reused_lane = program.open_lane(end + 8U, fn::LaneSampling{}).value();
            remaining              = std::span(tokens).first(split);
            while (!remaining.empty()) {
                const auto step = program.prefill_step(reused_lane, remaining);
                remaining       = remaining.subspan(static_cast<std::size_t>(step.consumed));
            }
            program.state().pack_continuation(reused_lane, checkpoint.data(), device.stream);
            CUDA_CHECK(cudaStreamSynchronize(device.stream));
            // The direct split is independent of checkpoint transport. After recording it, restore
            // the earlier prefix and reproduce its logits exactly before comparing with cold.
            program.reuse_lane(reused_lane, end + 8U, std::span(tokens).first(split),
                               fn::LaneSampling{});
            remaining = std::span(tokens).subspan(split, end - split);
            while (!remaining.empty()) {
                const auto step = program.prefill_step(reused_lane, remaining);
                remaining       = remaining.subspan(static_cast<std::size_t>(step.consumed));
            }
            std::vector<std::vector<std::uint16_t>> direct{logits(program, reused_lane)};
            const std::array<fn::Lane, 1> reused_lanes{reused_lane};
            for (std::uint32_t i = 0; i < 4; ++i) {
                const std::array<ninfer::TokenId, 1> input{tokens[end + i]};
                std::array<ninfer::TokenId, 1> next{};
                program.decode(reused_lanes, input, next);
                direct.push_back(logits(program, reused_lane));
            }
            program.reuse_lane(reused_lane, end + 8U, std::span(tokens).first(split),
                               fn::LaneSampling{}, checkpoint.data());
            program.state().pack_continuation(reused_lane, verify.data(), device.stream);
            CUDA_CHECK(cudaStreamSynchronize(device.stream));
            require(std::memcmp(checkpoint.data(), verify.data(), checkpoint.size()) == 0,
                    "restored continuation differs byte-for-byte");
            remaining = std::span(tokens).subspan(split, end - split);
            while (!remaining.empty()) {
                const auto step = program.prefill_step(reused_lane, remaining);
                remaining       = remaining.subspan(static_cast<std::size_t>(step.consumed));
            }
            require(logits(program, reused_lane) == direct[0],
                    "restored suffix differs from direct split logits");
            // Equality to the independently shifted route is stronger than matching its KL/top-1
            // distribution: retention must add no numerical difference above that route's floor.
            require(direct[0] == shifted[0],
                    "reuse exceeds the shifted cold-route floor at suffix");
            const auto initial = distribution_difference(shifted[0], cold[0]);
            floor_kl.push_back(initial.kl);
            double max_kl          = initial.kl;
            std::size_t top1_equal = initial.top1_equal ? 1U : 0U;
            for (std::uint32_t i = 0; i < 4; ++i) {
                const std::array<ninfer::TokenId, 1> input{tokens[end + i]};
                std::array<ninfer::TokenId, 1> next{};
                program.decode(reused_lanes, input, next);
                require(logits(program, reused_lane) == direct[i + 1],
                        "restored decode differs from direct split logits");
                require(direct[i + 1] == shifted[i + 1],
                        "reuse exceeds the shifted cold-route floor at decode");
                const auto difference = distribution_difference(shifted[i + 1], cold[i + 1]);
                floor_kl.push_back(difference.kl);
                std::cout << "route floor seed=" << seed << " split=" << split
                          << " decode=" << i + 1 << " top1=" << difference.top1_equal
                          << " KL=" << difference.kl << '\n';
                max_kl = std::max(max_kl, difference.kl);
                top1_equal += difference.top1_equal ? 1U : 0U;
            }
            program.close_lane(reused_lane);
            matching_top1 += top1_equal;
            std::cout << "reuse seed=" << seed << " split " << split << "+71: " << top1_equal
                      << "/5 cold top-1, max KL " << max_kl
                      << "; shifted cold, direct split and restore logits byte-identical\n";
        }
    }
    std::sort(floor_kl.begin(), floor_kl.end());
    std::cout << "cold-route floor: top-1 " << matching_top1 << '/' << floor_kl.size()
              << ", KL median=" << floor_kl.at(floor_kl.size() / 2)
              << " p95=" << floor_kl.at((floor_kl.size() - 1) * 95 / 100)
              << " max=" << floor_kl.back()
              << "; retention matches every floor distribution exactly\n";
}

void mixer_split_diagnostic(const Inputs& inputs) {
    ninfer::DeviceContext device(0);
    const ninfer::artifact::Reader reader(inputs.artifact);
    const fn::LoadedModel model(reader, device);
    fn::ProgramOptions options;
    options.ngram_table    = inputs.ngram_table;
    options.context_tokens = 18000;
    fn::Program program(model, options, device);
    const Tokens tokens = prompt_tokens(17500, 73);

    struct Captured {
        std::vector<std::uint16_t> inputs;
        std::vector<std::uint16_t> outputs;
        std::uint32_t columns = 0;
    };

    for (const std::uint32_t split : {63U, 64U, 65U}) {
        const auto count = split + 71U;
        auto capture     = [&](Captured& saved) {
            fn::ForwardObserver observer;
            observer.after_mixer = [&, count](std::int32_t layer, const ninfer::Tensor& input,
                                              const ninfer::Tensor& output, std::int32_t first) {
                if (layer != 0) { return; }
                require(saved.columns <= count && first >= 0 &&
                            first == static_cast<std::int32_t>(saved.columns) && input.ne[1] > 0 &&
                            static_cast<std::uint32_t>(input.ne[1]) <= count - saved.columns,
                        "mixer diagnostic columns are not consecutive");
                CUDA_CHECK(cudaStreamSynchronize(device.stream));
                auto download = [&](const ninfer::Tensor& tensor, std::vector<std::uint16_t>& dst) {
                    dst.resize(static_cast<std::size_t>(count) * fn::kHidden);
                    CUDA_CHECK(
                        cudaMemcpy(dst.data() + static_cast<std::size_t>(first) * fn::kHidden,
                                   tensor.data, tensor.bytes(), cudaMemcpyDeviceToHost));
                };
                download(input, saved.inputs);
                download(output, saved.outputs);
                saved.columns += static_cast<std::uint32_t>(input.ne[1]);
            };
            return observer;
        };
        Captured cold;
        auto cold_observer = capture(cold);
        auto lane          = program.open_lane(count, fn::LaneSampling{}).value();
        program.forward().prefill(lane, std::span(tokens).first(count), {}, &cold_observer);
        program.close_lane(lane);
        Captured segmented;
        auto segmented_observer = capture(segmented);
        lane                    = program.open_lane(count, fn::LaneSampling{}).value();
        program.forward().prefill(lane, std::span(tokens).first(split), {}, &segmented_observer);
        program.forward().prefill(lane, std::span(tokens).subspan(split, 71), {},
                                  &segmented_observer);
        program.close_lane(lane);
        auto difference = [](const auto& actual, const auto& expected) {
            double squared_error     = 0.0;
            double squared_reference = 0.0;
            for (std::size_t i = 0; i < actual.size(); ++i) {
                const double value = bf16(expected[i]);
                const double delta = bf16(actual[i]) - value;
                squared_error += delta * delta;
                squared_reference += value * value;
            }
            return std::sqrt(squared_error / std::max(squared_reference, 1e-30));
        };
        require(cold.columns == count && segmented.columns == count,
                "mixer diagnostic did not capture the complete input history");
        std::cout << "split mixer diagnostic " << split << "+71: layer=0, all " << count
                  << " input columns exact=" << (cold.inputs == segmented.inputs)
                  << " input_nrmse=" << difference(segmented.inputs, cold.inputs)
                  << " output_nrmse=" << difference(segmented.outputs, cold.outputs) << '\n';
    }
}

// --retained ---------------------------------------------------------------------------------

ninfer::RequestOptions reuse_options(std::uint32_t outputs, bool cache_write = true) {
    auto options                         = greedy(outputs);
    options.execution.allow_prefix_reuse = true;
    options.execution.cache_write        = cache_write;
    return options;
}

ninfer::GenerationResult reusable(ninfer::Engine& engine, const Tokens& prompt,
                                  std::uint32_t outputs = 1, bool cache_write = true) {
    return engine.generate(engine.prepare_tokens(prompt, true),
                           reuse_options(outputs, cache_write));
}

void retained_suite(const Inputs& inputs) {
    std::size_t cold_token_mismatches = 0;
    ninfer::Engine engine(engine_options(inputs, 4, 55000, 55000));
    ninfer::PromptInput chat;
    chat.options.enable_thinking = false;
    chat.messages.push_back(
        ninfer::ChatMessage{.role  = ninfer::ChatRole::User,
                            .parts = {ninfer::MessagePart{.text = "Name two prime numbers."}}});
    (void)engine.generate(engine.prepare(chat), reuse_options(4));
    const auto regenerated = engine.generate(engine.prepare(chat), reuse_options(4));
    require(regenerated.reused_prompt_tokens > 0 &&
                regenerated.prefix_reuse_path != ninfer::PrefixReusePath::FullReset,
            "chat regeneration failed to restore its exact rewrite checkpoint");
    const auto chat_cold = engine.generate(engine.prepare(chat), greedy(4));
    require_equal(regenerated.generated_token_ids, chat_cold.generated_token_ids,
                  "rewrite restore against cold chat");
    for (const std::uint32_t split : {63U, 64U, 65U, 4097U, 16385U}) {
        const Tokens whole = prompt_tokens(split + 71U, split);
        const Tokens prefix(whole.begin(), whole.begin() + split);
        const auto cold = run(engine, whole, 1);
        (void)reusable(engine, prefix);
        const auto before = engine.runtime_stats().computed_prefill_tokens;
        const auto reused = reusable(engine, whole);
        require(reused.reused_prompt_tokens == split, "wrong retained prefix length");
        require(reused.prefix_reuse_source == ninfer::PrefixReuseSource::VramResident,
                "retained prefix reported the wrong source");
        require(engine.runtime_stats().computed_prefill_tokens - before == 71,
                "reuse computed more than its suffix");
        if (reused.generated_token_ids != cold.generated_token_ids) {
            ++cold_token_mismatches;
            std::cout << "cold/reuse top-1 differs at split " << split << "+71\n";
        }
        const auto exact = reusable(engine, whole);
        require(exact.reused_prompt_tokens == whole.size(),
                "exact hit did not reuse the whole prompt");
        require_equal(exact.generated_token_ids, reused.generated_token_ids, "exact hit");
        std::cout << "Engine reuse " << split << "+71; cached=" << reused.reused_prompt_tokens
                  << ", TTFT " << cold.timings.first_token_seconds << " -> "
                  << reused.timings.first_token_seconds << " s\n";
    }
    Tokens original  = prompt_tokens(512, 888);
    const auto first = reusable(engine, original, 8);
    Tokens turn      = original;
    turn.insert(turn.end(), first.generated_token_ids.begin(), first.generated_token_ids.end());
    const auto second = reusable(engine, turn);
    require(second.reused_prompt_tokens == original.size() + 7,
            "output-tail reuse must exclude the sampled but unappended final token");

    // A disposable continuation must fork to independent allocations, not overwrite the source.
    const Tokens source = prompt_tokens(1024, 901);
    (void)reusable(engine, source);
    Tokens branch     = source;
    const auto suffix = prompt_tokens(65, 902);
    branch.insert(branch.end(), suffix.begin(), suffix.end());
    const auto fork = reusable(engine, branch, 4, false);
    require(fork.reused_prompt_tokens == source.size(),
            "disposable request failed to fork retention");
    const auto original_again = reusable(engine, source);
    require(original_again.reused_prompt_tokens == source.size(),
            "disposable request consumed its source");

    (void)reusable(engine, prompt_tokens(1000, 990));
    (void)reusable(engine, prompt_tokens(1000, 991));
    Tokens completed_branch = branch;
    completed_branch.insert(completed_branch.end(), fork.generated_token_ids.begin(),
                            fork.generated_token_ids.end() - 1);
    const auto branch_after_eviction = reusable(engine, completed_branch);
    require(branch_after_eviction.reused_prompt_tokens < completed_branch.size(),
            "disposable retention was not evicted before ordinary idle state");

    const Tokens decode_prefix = prompt_tokens(512, 992);
    const auto decoded         = engine.runtime_stats().committed_decode_tokens;
    const ninfer::CancellationView decode_cancel(
        [&] { return engine.runtime_stats().committed_decode_tokens >= decoded + 8; });
    const auto decode_cancelled = engine.generate(engine.prepare_tokens(decode_prefix, true),
                                                  reuse_options(64), nullptr, decode_cancel);
    require(decode_cancelled.finish_reason == ninfer::FinishReason::Cancelled &&
                !decode_cancelled.generated_token_ids.empty(),
            "retained decode cancellation failed");
    Tokens continuation = decode_prefix;
    continuation.insert(continuation.end(), decode_cancelled.generated_token_ids.begin(),
                        decode_cancelled.generated_token_ids.end());
    const auto resumed_decode = reusable(engine, continuation);
    require(resumed_decode.reused_prompt_tokens >= continuation.size() - 1 &&
                resumed_decode.reused_prompt_tokens <= continuation.size(),
            "decode cancellation did not retain its exact committed frontier");

    const Tokens cancel_prefix = prompt_tokens(512, 903);
    (void)reusable(engine, cancel_prefix);
    Tokens cancelled_prompt = cancel_prefix;
    const auto long_suffix  = prompt_tokens(22000, 904);
    cancelled_prompt.insert(cancelled_prompt.end(), long_suffix.begin(), long_suffix.end());
    const auto computed = engine.runtime_stats().computed_prefill_tokens;
    const ninfer::CancellationView cancel(
        [&] { return engine.runtime_stats().computed_prefill_tokens > computed; });
    const auto cancelled = engine.generate(engine.prepare_tokens(cancelled_prompt, true),
                                           reuse_options(4), nullptr, cancel);
    require(cancelled.finish_reason == ninfer::FinishReason::Cancelled &&
                cancelled.generated_token_ids.empty(),
            "suffix prefill cancellation failed");
    const auto rolled_back = reusable(engine, cancel_prefix);
    require(rolled_back.reused_prompt_tokens == cancel_prefix.size(),
            "cancelled suffix prefill lost its exact occupy frontier");

    // Interleaved turns retain four distinct continuations and then join a compact decode batch.
    std::vector<Tokens> agents;
    for (std::uint32_t i = 0; i < 4; ++i) {
        agents.push_back(prompt_tokens(1024 + i * 64, 10000 + i));
        (void)reusable(engine, agents.back());
    }
    std::vector<ninfer::GenerationHandle> handles;
    for (auto& agent : agents) {
        agent.push_back(7200);
        handles.push_back(engine.submit(engine.prepare_tokens(agent, true), reuse_options(8)));
    }
    for (std::size_t i = 0; i < handles.size(); ++i) {
        const auto result = handles[i].wait();
        require(result.reused_prompt_tokens == agents[i].size() - 1,
                "interleaved turn lost its retained lane");
    }

    // Page pressure reclaims idle retention, never active state; the evicted prefix becomes cold.
    const auto evicted = agents.front();
    (void)reusable(engine, prompt_tokens(54000, 12000));
    const auto after_pressure = reusable(engine, evicted);
    require(after_pressure.finish_reason == ninfer::FinishReason::OutputLimit &&
                after_pressure.reused_prompt_tokens == 0,
            "retained pressure did not evict its oldest idle prefix");
    std::cout << "retained Engine: exact hits, output tail, source-preserving fork, cancellation, "
                 "C=4 and page pressure pass; cold top-1 mismatches="
              << cold_token_mismatches << '\n';
    // The independent --reuse-parity panel checks these partition-dependent distributions
    // against the cold-route floor. This mode checks the Engine's ownership and reuse behavior.
}

void disk_close_before_drain(const Inputs& inputs, const ninfer::EngineOptions& options,
                             const Tokens& source, const Tokens& expected) {
    ninfer::DeviceContext device(0);
    const ninfer::artifact::Reader reader(inputs.artifact);
    const fn::LoadedModel model(reader, device);
    auto frontend = fn::Package::make_frontend(model);
    fn::EngineProgram program(model, options, device);

    struct DiskClaim {
        fn::EngineProgram& owner;
        const std::uint64_t id;

        DiskClaim(fn::EngineProgram& program, const ninfer::runtime::RequestPlanSummary& summary)
            : owner(program), id(summary.disk_entry_id) {
            require(owner.claim_disk_entry(id, summary.disk_execution_frontier,
                                           summary.disk_hash_f_lo, summary.disk_hash_f_hi,
                                           summary.reusable_prompt_tokens, summary.disk_reuse_path,
                                           summary.disk_committed_generation),
                    "disk-close fixture claim");
        }

        ~DiskClaim() { owner.release_disk_entry(id); }

        DiskClaim(const DiskClaim&)            = delete;
        DiskClaim& operator=(const DiskClaim&) = delete;
    };

    for (const bool close_pending : {true, false}) {
        auto prompt = frontend.prepare_tokens(source, true);
        ninfer::runtime::ResolvedExecutionOptions execution;
        execution.requested_output_tokens   = 1;
        execution.reservation_window_tokens = 1;
        execution.allow_prefix_reuse        = true;
        const auto base                     = program.plan_request_base(prompt, execution);
        auto plan                           = program.plan_disk_reuse(prompt, base);
        const auto summary                  = plan.summary();
        require(summary.reuse_source == ninfer::PrefixReuseSource::HostDisk &&
                    summary.reusable_prompt_tokens == source.size(),
                "disk-close fixture has no complete durable source");
        const DiskClaim claim(program, summary);
        program.restore_disk_entry(0, summary.disk_entry_id, plan);
        require(program.pending_disk_restore_ticket() != 0, "disk restore did not own a ticket");
        if (close_pending) {
            // No executor wait or cancellation precedes this close. The adapter owns draining
            // the destination's scatter and state transfers before its pages can be reused.
            program.evict_retained_lane(0);
            require(program.pending_disk_restore_ticket() == 0 && !program.has_retained_lane(0),
                    "closing an SSD destination left a live restore or lane");
        } else {
            program.wait_kv_disk_copies();
            program.wait_kv_ram_copies_on_compute();
            const auto first = program.start_prefill_lane(0, std::move(prompt), std::move(plan), {},
                                                          ninfer::runtime::PrefillPace::Exclusive);
            require(first.complete && !first.round.tokens.empty() && expected.size() == 1 &&
                        first.round.tokens.front() == expected.front(),
                    "SSD close/reopen changed the complete-state continuation");
            program.resolve_prefill_lane(0, true);
            program.evict_retained_lane(0);
        }
    }
    std::cout << "SSD destination close cancels its own pending restore; source restores exactly\n";
}

void tier_suite(const Inputs& inputs, bool disk) {
    const auto directory =
        std::filesystem::temp_directory_path() /
        ("ninfer-flash-tier-" + std::to_string(Clock::now().time_since_epoch().count()));
    if (disk) {
        require(std::filesystem::create_directory(directory), "cannot create disk test directory");
    }
    try {
        auto options                  = engine_options(inputs, 1, 4096, 4096);
        options.kv_ram_capacity_bytes = std::size_t{512} << 20;
        options.kv_ram_system_prompts = 2;
        const Tokens close_prompt     = prompt_tokens(600, 1950);
        Tokens close_expected;
        if (disk) {
            options.kv_disk_capacity_bytes = std::size_t{2048} << 20;
            options.kv_disk_location       = directory;
            options.kv_disk_compress       = ninfer::KvDiskCompress::Zstd;
        }
        {
            ninfer::Engine engine(options);
            const Tokens prefix = prompt_tokens(512, 1701);
            Tokens extended     = prefix;
            const auto suffix   = prompt_tokens(65, 1702);
            extended.insert(extended.end(), suffix.begin(), suffix.end());
            const auto first  = reusable(engine, prefix);
            const auto direct = reusable(engine, extended, 8);
            require(direct.reused_prompt_tokens == prefix.size(),
                    "tier reference did not use direct retained prefix");
            for (std::uint32_t i = 0; i < (disk ? 6U : 1U); ++i) {
                (void)reusable(engine, prompt_tokens(600, 1800 + i));
            }
            const auto restored = reusable(engine, prefix);
            require(restored.reused_prompt_tokens == prefix.size(),
                    "host tier did not restore rollback frontier");
            require(restored.prefix_reuse_source == (disk ? ninfer::PrefixReuseSource::HostDisk
                                                          : ninfer::PrefixReuseSource::HostRam),
                    "host tier reported the wrong source");
            require_equal(restored.generated_token_ids, first.generated_token_ids,
                          "host restored exact-prefix logits");
            const auto continued = reusable(engine, extended, 8);
            require_equal(continued.generated_token_ids, direct.generated_token_ids,
                          "host restored suffix and decode");
            // One lane forces a disposable turn to secure the full original in RAM before
            // releasing its GPU state. Its branch cannot consume or persist that source.
            const auto branch = reusable(engine, prefix, 1, false);
            require(branch.reused_prompt_tokens == prefix.size(),
                    "RAM-backed disposable fork missed its source");
            const auto source_again = reusable(engine, prefix);
            require(source_again.reused_prompt_tokens == prefix.size(),
                    "disposable host restore consumed its source");
            require_equal(source_again.generated_token_ids, first.generated_token_ids,
                          "disposable source preservation");
            ninfer::PromptInput chat;
            chat.options.enable_thinking = false;
            chat.messages                = {
                ninfer::ChatMessage{
                    .role  = ninfer::ChatRole::System,
                    .parts = {ninfer::MessagePart{.text = "Answer briefly using plain text."}}},
                ninfer::ChatMessage{
                    .role  = ninfer::ChatRole::User,
                    .parts = {ninfer::MessagePart{.text = "Name two prime numbers."}}}};
            auto prepared  = engine.prepare(chat);
            const auto ids = prepared.token_ids();
            std::string rendered;
            std::vector<std::size_t> byte_offsets;
            for (const auto token : ids) {
                byte_offsets.push_back(rendered.size());
                rendered += engine.token_bytes(token);
            }
            const auto boundary = rendered.find("<|im_start|>user\n");
            const auto found    = std::find(byte_offsets.begin(), byte_offsets.end(), boundary);
            require(boundary != std::string::npos && found != byte_offsets.end(),
                    "system frontier is not an exact token boundary");
            const auto system_frontier = static_cast<std::uint32_t>(found - byte_offsets.begin());
            const Tokens system_tokens(ids.begin(), ids.begin() + system_frontier);
            const auto system_cold = run(engine, system_tokens, 1);
            (void)engine.generate(std::move(prepared), reuse_options(1));
            (void)reusable(engine, prompt_tokens(600, 1900));
            const auto system_restored = reusable(engine, system_tokens);
            require(system_restored.reused_prompt_tokens == system_frontier &&
                        system_restored.prefix_reuse_source == ninfer::PrefixReuseSource::HostRam,
                    "exact system frontier did not restore from its RAM anchor");
            require_equal(system_restored.generated_token_ids, system_cold.generated_token_ids,
                          "system-prefix tail logits");
            const auto stats = engine.runtime_stats();
            std::cout << (disk ? "SSD" : "RAM")
                      << " Engine: rollback, exact hit, suffix/decode and disposable preservation "
                         "pass; captures="
                      << stats.kv_ram_captures << " RAM restores=" << stats.kv_ram_restores << '\n';
            if (disk) {
                auto request       = reuse_options(1);
                request.output.raw = true;
                close_expected = engine.generate(engine.prepare_tokens(close_prompt, true), request)
                                     .generated_token_ids;
            }
        }
        if (disk) { disk_close_before_drain(inputs, options, close_prompt, close_expected); }
    } catch (...) {
        if (disk) { std::filesystem::remove_all(directory); }
        throw;
    }
    if (disk) { std::filesystem::remove_all(directory); }
}

void live_tier_suite(const Inputs& inputs) {
    ninfer::DeviceContext device(0);
    const ninfer::artifact::Reader reader(inputs.artifact);
    const fn::LoadedModel model(reader, device);
    auto frontend                 = fn::Package::make_frontend(model);
    auto options                  = engine_options(inputs, 2, 1024, 2048);
    options.kv_ram_capacity_bytes = std::size_t{1024} << 20;
    fn::EngineProgram program(model, options, device);
    constexpr std::uint64_t owner = 71;
    auto execute                  = [&](bool suspend, bool relocate = false, bool p_less = false) {
        ninfer::runtime::ResolvedExecutionOptions execution;
        execution.requested_output_tokens    = 16;
        execution.reservation_window_tokens  = 4;
        execution.allow_prefix_reuse         = true;
        execution.sampling.temperature       = 0.8F;
        execution.sampling.p_less            = p_less;
        execution.sampling.top_k             = 20;
        execution.sampling.presence_penalty  = 0.4F;
        execution.sampling.frequency_penalty = 0.2F;
        execution.sampling.seed              = 73;
        auto prompt                          = frontend.prepare_tokens(prompt_tokens(65, 73), true);
        const auto base                      = program.plan_request_base(prompt, execution);
        require(base.summary().window_output_tokens == 4, "RAM-on initial window is not bounded");
        auto plan  = program.plan_request_for_lane(0, prompt, base);
        auto first = program.start_prefill_lane(0, std::move(prompt), std::move(plan), {},
                                                ninfer::runtime::PrefillPace::Exclusive);
        require(first.complete, "live tier fixture unexpectedly needs another prefill step");
        Tokens result{first.round.tokens.front()};
        program.resolve_prefill_lane(0, false);
        std::uint32_t lane = 0;
        for (std::uint32_t produced = 1; produced < 16; ++produced) {
            if (produced == 4) {
                require(program.grant_window(lane, program.plan_window(lane, 12)),
                        "window growth failed");
            }
            if (produced == 8 && !suspend) {
                // Both arms must use the same funded window and proposal geometry.
                require(program.grant_window(lane, program.plan_window(lane, 16)),
                        "matched live window growth failed");
            }
            if (produced == 8 && suspend) {
                const auto window = program.plan_window(lane, 16);
                require(program.reserve_suspension_credit(owner, window.suspension_bytes),
                        "live RAM credit failed");
                require(program.start_suspend(lane, owner), "live RAM capture failed");
                while (!program.suspension_ready(owner)) { std::this_thread::yield(); }
                program.finish_suspend(lane, owner);
                if (relocate) {
                    require(program.reserve_suspension_credit(owner + 1, std::size_t{1} << 20),
                            "cannot place a neighboring credit for the relocation case");
                }
                require(program.reserve_suspension_credit(owner, window.suspension_bytes +
                                                                     (std::size_t{1} << 20)),
                        "parked credit growth failed");
                auto noise           = frontend.prepare_tokens(prompt_tokens(8, 777), false);
                auto noise_execution = execution;
                noise_execution.allow_prefix_reuse = false;
                const auto noise_base = program.plan_request_base(noise, noise_execution);
                auto noise_plan       = program.plan_request_for_lane(0, noise, noise_base);
                (void)program.start_prefill_lane(0, std::move(noise), std::move(noise_plan), {},
                                                 ninfer::runtime::PrefillPace::Exclusive);
                program.resolve_prefill_lane(0, false);
                lane = 1;
                require(program.start_resume(lane, owner, program.plan_suspended_window(owner, 16)),
                        "live RAM restore failed");
                while (!program.resume_ready(owner)) { std::this_thread::yield(); }
                program.finish_resume(lane, owner);
                program.abort_lane(0);
            }
            const std::array<std::uint32_t, 1> lanes{lane};
            const std::array<ninfer::runtime::RoundBudget, 1> budget{{{16U - produced}}};
            const auto round = program.decode_batch(lanes, budget);
            result.push_back(round.tokens.front());
            const std::array<std::uint32_t, 1> accepted{1};
            const std::array<std::uint8_t, 1> terminal{static_cast<std::uint8_t>(produced == 15)};
            const std::array<std::uint8_t, 1> flags{0};
            program.resolve_pending_batch(lanes, accepted, terminal, flags);
        }
        if (inputs.speculative.backend == ninfer::SpeculativeBackend::DFlash) {
            const auto stats     = program.speculative_stats_lane(lane);
            std::uint64_t rounds = 0;
            for (const auto count : stats.rounds_per_draft) { rounds += count; }
            require(stats.enabled && stats.rounds + stats.fallback_steps == 15 && rounds == 15,
                    "live suspension lost cumulative request speculative counters");
        }
        program.abort_lane(lane);
        if (suspend) { program.release_suspension_credit(owner); }
        if (relocate) { program.release_suspension_credit(owner + 1); }
        return result;
    };
    for (const bool p_less : {false, true}) {
        if (p_less && inputs.speculative.backend == ninfer::SpeculativeBackend::None) { continue; }
        const auto uninterrupted = execute(false, false, p_less);
        require_equal(execute(true, false, p_less), uninterrupted,
                      "cross-lane live RAM resume after in-place credit growth");
        require_equal(execute(true, true, p_less), uninterrupted,
                      "cross-lane live RAM resume after relocated credit growth");
    }
    std::cout << "live RAM: 4 -> 12 -> 16 output windows; cross-lane resume preserves all 16 "
                 "sampled tokens with penalties\n";
}

void conversation_bench(const Inputs& inputs) {
    const char* path = std::getenv("NINFER_FLASH_NEXT_BENCH_TOKENS");
    require(path != nullptr && *path != '\0',
            "set NINFER_FLASH_NEXT_BENCH_TOKENS to an agent transcript I32 file");
    Tokens transcript(50000);
    std::ifstream file(path, std::ios::binary);
    require(static_cast<bool>(file.read(
                reinterpret_cast<char*>(transcript.data()),
                static_cast<std::streamsize>(transcript.size() * sizeof(ninfer::TokenId)))),
            "agent transcript must contain at least 50000 I32 tokens");
    ninfer::Engine engine(engine_options(inputs, 1, 55000, 55000));
    std::array<double, 4> cold{};
    std::size_t turn = 0;
    for (const auto count : {8192U, 16384U, 32768U, 50000U}) {
        const Tokens prompt(transcript.begin(), transcript.begin() + count);
        const auto result = run(engine, prompt, 1);
        cold.at(turn++)   = result.timings.first_token_seconds;
    }
    turn = 0;
    for (const auto count : {8192U, 16384U, 32768U, 50000U}) {
        const Tokens prompt(transcript.begin(), transcript.begin() + count);
        const auto result = reusable(engine, prompt);
        std::cout << "conversation turn=" << turn + 1 << " prompt=" << count
                  << " cached=" << result.reused_prompt_tokens << " cold_ttft_s=" << cold.at(turn)
                  << " reuse_ttft_s=" << result.timings.first_token_seconds << '\n';
        ++turn;
    }
}

void conversation_ram_bench(const Inputs& inputs) {
    const char* path = std::getenv("NINFER_FLASH_NEXT_BENCH_TOKENS");
    require(path != nullptr && *path != '\0',
            "set NINFER_FLASH_NEXT_BENCH_TOKENS to an agent transcript I32 file");
    Tokens transcript(50000);
    std::ifstream file(path, std::ios::binary);
    require(static_cast<bool>(file.read(
                reinterpret_cast<char*>(transcript.data()),
                static_cast<std::streamsize>(transcript.size() * sizeof(ninfer::TokenId)))),
            "agent transcript must contain at least 50000 I32 tokens");
    auto options                  = engine_options(inputs, 4, 55000, 220000);
    options.kv_ram_capacity_bytes = std::size_t{8192} << 20;
    ninfer::Engine engine(options);
    const std::array<std::uint32_t, 4> lengths{8192, 16384, 32768, 50000};
    std::array<double, 4> cold{};
    for (const bool reuse : {false, true}) {
        for (std::size_t turn = 0; turn < lengths.size(); ++turn) {
            std::vector<ninfer::GenerationHandle> handles;
            for (std::uint32_t agent = 0; agent < 6; ++agent) {
                Tokens prompt(transcript.begin(), transcript.begin() + lengths[turn]);
                // Distinct transcript identities compete for four lanes; each identity is
                // unchanged across its four teacher-forced growing turns.
                prompt.front() = static_cast<ninfer::TokenId>(1000U + agent);
                handles.push_back(engine.submit(engine.prepare_tokens(std::move(prompt), true),
                                                reuse ? reuse_options(1) : greedy(1)));
            }
            double ttft            = 0.0;
            std::uint64_t cached   = 0;
            std::uint32_t ram_hits = 0;
            for (auto& handle : handles) {
                const auto result = handle.wait();
                require(result.finish_reason == ninfer::FinishReason::OutputLimit,
                        "conversation benchmark did not complete");
                ttft += result.timings.first_token_seconds;
                cached += result.reused_prompt_tokens;
                ram_hits +=
                    result.prefix_reuse_source == ninfer::PrefixReuseSource::HostRam ? 1U : 0U;
            }
            ttft /= 6.0;
            if (!reuse) {
                cold[turn] = ttft;
            } else {
                const auto expected = turn == 0 ? 0ULL : 6ULL * lengths[turn - 1];
                require(cached == expected,
                        "RAM conversation benchmark lost a previous-turn prefix");
                std::cout << "conversation RAM C=4 agents=6 turn=" << turn + 1
                          << " prompt=" << lengths[turn] << " cached=" << cached
                          << " RAM_hits=" << ram_hits << " mean_cold_ttft_s=" << cold[turn]
                          << " mean_reuse_ttft_s=" << ttft << '\n';
            }
        }
    }
}

// --engine ----------------------------------------------------------------------------------

constexpr std::uint32_t kLanes         = 4;
constexpr std::uint32_t kEngineContext = 41000;
// 641 QSA pages: the long interruption owner and one short request fit together, two
// kWaitPrompt reservations (329 pages each) do not.
constexpr std::uint32_t kEngineKvTokens = 41000;
constexpr std::uint32_t kShortPrompt    = 300;
constexpr std::uint32_t kShortOutputs   = 48;
constexpr std::uint32_t kWaitPrompt     = 21000;
constexpr std::uint32_t kWaitOutputs    = 32;
constexpr std::uint32_t kLongPrompt     = 40000; // super-chunks 16384, 16384, 7232
// The interrupting request finishes within the mixed_forward_rounds (12) decode rounds that run
// before the resumed owner's next super-chunk: one token from prefill, seven from decode.
constexpr std::uint32_t kInterruptingOutputs = 8;

struct EngineBatchObservation {
    Tokens prompt;
    Tokens tokens;
    std::vector<float> selected_logprobs;
};

using EngineBatch = std::array<EngineBatchObservation, kLanes>;

#include "teacher_forced_cases.h"

std::vector<Tokens> concurrent(ninfer::Engine& engine, const std::vector<Tokens>& prompts,
                               std::uint32_t outputs, std::vector<double>* queued = nullptr) {
    std::vector<ninfer::GenerationHandle> handles;
    handles.reserve(prompts.size());
    for (const Tokens& prompt : prompts) {
        handles.push_back(engine.submit(engine.prepare_tokens(prompt, false), greedy(outputs)));
    }
    std::vector<Tokens> results;
    for (auto& handle : handles) {
        ninfer::GenerationResult result = handle.wait();
        require(result.finish_reason == ninfer::FinishReason::OutputLimit,
                "concurrent request did not reach its output limit");
        if (queued != nullptr) { queued->push_back(result.timings.queued_seconds); }
        results.push_back(std::move(result.generated_token_ids));
    }
    return results;
}

EngineBatch concurrent_observations(ninfer::Engine& engine) {
    std::vector<Tokens> prompts;
    std::vector<ninfer::TokenId> first;
    auto request                = greedy(kShortOutputs);
    request.output.raw          = true;
    request.output.top_logprobs = 1;
    for (std::uint32_t i = 0; i < kLanes; ++i) {
        prompts.push_back(prompt_tokens(kShortPrompt + 17 * i, 100 + i));
        const auto result = engine.generate(engine.prepare_tokens(prompts.back(), false), request);
        require(result.finish_reason == ninfer::FinishReason::OutputLimit &&
                    result.generated_token_ids.size() == kShortOutputs,
                "solo output limit");
        first.push_back(result.generated_token_ids.front());
    }
    const ninfer::RuntimeStats before = engine.runtime_stats();
    std::vector<ninfer::GenerationHandle> handles;
    handles.reserve(prompts.size());
    for (const Tokens& prompt : prompts) {
        handles.push_back(engine.submit(engine.prepare_tokens(prompt, false), request));
    }
    EngineBatch observations;
    for (std::size_t row = 0; row < handles.size(); ++row) {
        auto result = handles[row].wait();
        require(result.finish_reason == ninfer::FinishReason::OutputLimit &&
                    result.generated_token_ids.size() == kShortOutputs,
                "concurrent output limit");
        require(result.generated_token_ids.front() == first[row],
                "C=4 request " + std::to_string(row) + ": first tokens differ");
        auto& observation  = observations[row];
        observation.prompt = prompts[row];
        observation.tokens = std::move(result.generated_token_ids);
        require(result.content_logprobs.size() == observation.tokens.size(),
                "raw request lost a selected-token logprob");
        observation.selected_logprobs.reserve(result.content_logprobs.size());
        for (std::size_t position = 0; position < result.content_logprobs.size(); ++position) {
            const auto& record = result.content_logprobs[position];
            require(record.token == observation.tokens[position], "raw token/logprob alignment");
            observation.selected_logprobs.push_back(record.logprob);
        }
    }
    const ninfer::RuntimeStats after = engine.runtime_stats();
    const std::uint64_t rounds       = after.decode_rounds - before.decode_rounds;
    const std::uint64_t rows         = after.decode_row_rounds - before.decode_row_rounds;
    require(rows > rounds, "concurrent requests never shared a decode round");
    std::cout << "C=4: first tokens equal; " << rows << " rows in " << rounds << " decode rounds\n";
    return observations;
}

void capacity_waits(ninfer::Engine& engine) {
    std::vector<Tokens> prompts;
    std::vector<Tokens> solo;
    for (std::uint32_t i = 0; i < 3; ++i) {
        prompts.push_back(prompt_tokens(kWaitPrompt, 200 + i));
        solo.push_back(run(engine, prompts.back(), kWaitOutputs).generated_token_ids);
    }
    std::vector<double> queued;
    const ninfer::RuntimeStats before = engine.runtime_stats();
    const std::vector<Tokens> waited  = concurrent(engine, prompts, kWaitOutputs, &queued);
    const ninfer::RuntimeStats after  = engine.runtime_stats();
    for (std::uint32_t i = 0; i < 3; ++i) {
        require_equal(waited[i], solo[i], "capacity-waiting request " + std::to_string(i));
    }
    // Only one reservation fits the pool: the later two wait for at least one whole request.
    const auto waited_count =
        std::count_if(queued.begin(), queued.end(), [](double seconds) { return seconds > 0.5; });
    require(waited_count >= 2, "requests beyond the KV capacity did not wait for admission");
    // With room for two reservations, the second request would decode beside the first; under
    // the cap no decode round holds two rows.
    require(after.decode_row_rounds - before.decode_row_rounds ==
                after.decode_rounds - before.decode_rounds,
            "two requests whose reservations exceed the pool decoded together");
    std::cout << "KV-capacity waits: equal; queued " << queued[0] << " / " << queued[1] << " / "
              << queued[2] << " s\n";
}

void prefill_interruption(ninfer::Engine& engine) {
    const Tokens long_prompt  = prompt_tokens(kLongPrompt, 300);
    const Tokens short_prompt = prompt_tokens(kShortPrompt, 301);
    const Tokens long_solo    = run(engine, long_prompt, 16).generated_token_ids;
    const Tokens short_solo   = run(engine, short_prompt, kInterruptingOutputs).generated_token_ids;

    const std::uint64_t base = engine.runtime_stats().computed_prefill_tokens;
    ninfer::GenerationHandle owner =
        engine.submit(engine.prepare_tokens(long_prompt, false), greedy(16));
    while (engine.runtime_stats().computed_prefill_tokens < base + 16384) {
        std::this_thread::sleep_for(std::chrono::milliseconds(5));
    }
    const ninfer::GenerationResult interrupting =
        engine.generate(engine.prepare_tokens(short_prompt, false), greedy(kInterruptingOutputs));
    const std::uint64_t computed         = engine.runtime_stats().computed_prefill_tokens - base;
    const ninfer::GenerationResult owned = owner.wait();
    require_equal(interrupting.generated_token_ids, short_solo, "interrupting request");
    require_equal(owned.generated_token_ids, long_solo, "interrupted owner");
    require(computed < kLongPrompt + kShortPrompt,
            "the short request did not finish before the long prefill completed");
    std::cout << "prefill interruption: short request finished after " << computed
              << " prefill tokens\n";
}

void cancellation(ninfer::Engine& engine) {
    const Tokens reference_prompt = prompt_tokens(kShortPrompt, 400);
    const Tokens reference = run(engine, reference_prompt, kShortOutputs).generated_token_ids;

    const std::uint64_t decoded = engine.runtime_stats().committed_decode_tokens;
    const ninfer::CancellationView decode_cancel(
        [&] { return engine.runtime_stats().committed_decode_tokens >= decoded + 16; });
    const ninfer::GenerationResult cancelled_decode =
        engine.generate(engine.prepare_tokens(prompt_tokens(kShortPrompt, 401), false),
                        greedy(4000), nullptr, decode_cancel);
    require(cancelled_decode.finish_reason == ninfer::FinishReason::Cancelled,
            "decode cancellation did not finish as cancelled");
    require(cancelled_decode.generated_token_ids.size() < 4000,
            "decode cancellation ran to the output limit");

    const std::uint64_t prefilled = engine.runtime_stats().computed_prefill_tokens;
    const ninfer::CancellationView prefill_cancel(
        [&] { return engine.runtime_stats().computed_prefill_tokens > prefilled; });
    const ninfer::GenerationResult cancelled_prefill =
        engine.generate(engine.prepare_tokens(prompt_tokens(kLongPrompt, 402), false), greedy(16),
                        nullptr, prefill_cancel);
    const std::uint64_t computed = engine.runtime_stats().computed_prefill_tokens - prefilled;
    require(cancelled_prefill.finish_reason == ninfer::FinishReason::Cancelled &&
                cancelled_prefill.generated_token_ids.empty(),
            "prefill cancellation produced output");
    require(computed < kLongPrompt, "prefill cancellation did not stop the prompt early");

    require_equal(run(engine, reference_prompt, kShortOutputs).generated_token_ids, reference,
                  "request after cancellations");
    const ninfer::RuntimeStats stats = engine.runtime_stats();
    require(stats.running_requests == 0 && stats.waiting_requests == 0,
            "cancelled requests still hold lanes");
    std::cout << "cancellation: decode after " << cancelled_decode.generated_token_ids.size()
              << " tokens, prefill after " << computed << " of " << kLongPrompt
              << " prompt tokens; later request unchanged\n";
}

#include "dflash_engine_cases.h"
#include "dflash_mask_cases.h"

void engine_suite(const Inputs& inputs) {
    EngineBatch observations;
    {
        ninfer::Engine engine(engine_options(inputs, kLanes, kEngineContext, kEngineKvTokens));
        const ninfer::MemorySummary memory = engine.memory_summary();
        require(memory.kv_capacity >= kEngineKvTokens, "Engine resolved a smaller KV capacity");
        observations = concurrent_observations(engine);
        capacity_waits(engine);
        prefill_interruption(engine);
        cancellation(engine);
    }
    cross_batch_teacher_forcing(inputs, observations);
}

} // namespace

int main(int argc, char** argv) {
    const Inputs inputs{.artifact    = std::getenv("NINFER_QWEN3_8_FLASH_NEXT_WEIGHTS"),
                        .ngram_table = std::getenv("NINFER_QWEN3_8_FLASH_NEXT_NGRAM_TABLE")};
    if (inputs.artifact == nullptr || *inputs.artifact == '\0' || inputs.ngram_table == nullptr ||
        *inputs.ngram_table == '\0') {
        std::cout << "skip: set NINFER_QWEN3_8_FLASH_NEXT_WEIGHTS and "
                     "NINFER_QWEN3_8_FLASH_NEXT_NGRAM_TABLE\n";
        return 77;
    }
    try {
        const std::string_view mode = argc == 2 ? std::string_view(argv[1]) : std::string_view{};
        if (mode == "--program-parity") {
            program_parity(inputs);
        } else if (mode == "--reuse-parity") {
            reuse_parity(inputs);
        } else if (mode == "--verify-parity") {
            verify_parity(inputs);
        } else if (mode == "--dflash-program") {
            dflash_program_cases(inputs);
        } else if (mode == "--dflash-engine") {
            dflash_engine_cases(inputs);
        } else if (mode == "--dflash-masks" || mode == "--dflash-mask-program") {
            const ninfer::artifact::Reader reader(inputs.artifact);
            if (reader.find("dflash/feature_projection") == nullptr) {
                std::cout << "skip: masked speculation requires the Flash-Next DFlash2 companion\n";
                return 77;
            }
            if (mode == "--dflash-masks") {
                dflash_mask_cases(inputs);
            } else {
                dflash_mask_program_cases(inputs);
            }
        } else if (mode == "--dflash-ram-tiers" || mode == "--dflash-disk-tiers" ||
                   mode == "--dflash-live-tiers" || mode == "--dflash-retained") {
            Inputs speculative                   = inputs;
            speculative.speculative.backend      = ninfer::SpeculativeBackend::DFlash;
            speculative.speculative.draft_tokens = 3;
            if (mode == "--dflash-live-tiers") {
                live_tier_suite(speculative);
            } else if (mode == "--dflash-retained") {
                retained_suite(speculative);
            } else {
                tier_suite(speculative, mode == "--dflash-disk-tiers");
            }
        } else if (mode == "--retained") {
            retained_suite(inputs);
        } else if (mode == "--ram-tiers") {
            tier_suite(inputs, false);
        } else if (mode == "--disk-tiers") {
            tier_suite(inputs, true);
        } else if (mode == "--live-tiers") {
            live_tier_suite(inputs);
        } else if (mode == "--conversation-bench") {
            conversation_bench(inputs);
        } else if (mode == "--conversation-ram-bench") {
            conversation_ram_bench(inputs);
        } else if (mode == "--mixer-split-diagnostic") {
            mixer_split_diagnostic(inputs);
        } else if (mode.empty()) {
            engine_suite(inputs);
        } else {
            std::cerr
                << "usage: " << argv[0]
                << " [--program-parity|--reuse-parity|--verify-parity|--dflash-program|"
                   "--dflash-engine|--dflash-masks|--dflash-mask-program|--retained|--dflash-"
                   "retained|--ram-tiers|--disk-"
                   "tiers|"
                   "--live-tiers|--dflash-ram-tiers|--dflash-disk-tiers|--dflash-live-tiers|"
                   "--conversation-bench|--conversation-ram-bench|--mixer-split-diagnostic]\n";
            return 2;
        }
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "FAIL: " << error.what() << '\n';
        return 1;
    }
}
