// Real-artifact qualification of token logprobs through the public Engine: every decode route
// that can produce a token (prefill sample, ordinary decode, DFlash chain and packed-tree verify,
// adaptive and eager rounds, concurrent rows) must report that token's log-probability under the
// target model.
//
// Two independent references are used. Greedy decoding emits the argmax of the same logits the
// record ranks, so the emitted token must be rank 0 exactly; a record scored at the wrong
// verification column or row fails this. Teacher-forced Engine::score on a non-speculative
// Engine recomputes each token's NLL along the ordinary route.

#include "ninfer/engine.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace {

constexpr std::uint32_t kOutputs = 48;

// Same-route agreement: the scoring Engine replays the ordinary decode that produced the record,
// so only the two FP32 log-sum-exp reductions differ.
constexpr double kSameRouteNllTolerance = 1.0e-4;
// A speculative verify evaluates the target model on its own activation profile (packed widths,
// quantized verify activations), a deliberately different numeric route from ordinary decode, so
// agreement is behavioral. A record scored at the wrong column or row is off by whole nats on
// most tokens, which the mean bounds; the per-token cap bounds isolated corruption.
constexpr double kCrossRouteMeanNllTolerance = 0.15;
constexpr double kCrossRouteMaxNllTolerance  = 1.0;

ninfer::EngineOptions base_engine_options(const char* artifact) {
    ninfer::EngineOptions options;
    options.artifact_path = artifact;
    options.max_context   = 1024;
    options.kv_capacity   = ninfer::KvCapacityPolicy::explicit_capacity(1024);
    options.prefill_chunk = 256;
    options.kv_cache      = ninfer::KvCacheStorage::Nvfp4;
    options.enable_vision = false;
    return options;
}

// Chain verify with a pinned draft temperature, so a concurrent run draws the same drafts as its
// sequential reference.
ninfer::EngineOptions chain_engine_options(const char* artifact, std::uint32_t draft_tokens,
                                           std::uint32_t max_concurrency) {
    ninfer::EngineOptions options                       = base_engine_options(artifact);
    options.speculative.backend                         = ninfer::SpeculativeBackend::DFlash;
    options.speculative.draft_tokens                    = draft_tokens;
    options.speculative.proposal_head                   = ninfer::ProposalHead::Optimized;
    options.speculative.dflash_verify_width             = draft_tokens + 1;
    options.speculative.dflash_p_less_draft_temperature = 0.6F;
    options.max_concurrency                             = max_concurrency;
    return options;
}

ninfer::RequestOptions raw_options(std::optional<std::uint32_t> top_logprobs) {
    ninfer::RequestOptions options;
    options.execution.requested_output_tokens = kOutputs;
    options.execution.sampling.temperature    = 0.0F;
    options.execution.allow_prefix_reuse      = false;
    options.stop.include_model_defaults       = false;
    // Raw output publishes every generated token as content, so records align with token ids.
    options.output.raw          = true;
    options.output.top_logprobs = top_logprobs;
    return options;
}

ninfer::RequestOptions raw_p_less_options(std::uint32_t top_logprobs, std::uint64_t seed) {
    ninfer::RequestOptions options         = raw_options(top_logprobs);
    options.execution.sampling.temperature = 2.0F;
    options.execution.sampling.p_less      = true;
    options.execution.sampling.seed        = seed;
    return options;
}

ninfer::PromptInput chat_input(std::string text, bool enable_thinking) {
    ninfer::ChatMessage message;
    message.role = ninfer::ChatRole::User;
    message.parts.push_back(ninfer::MessagePart{
        .kind = ninfer::MessagePartKind::Text, .text = std::move(text), .media = {}});
    ninfer::PromptInput input;
    input.messages.push_back(std::move(message));
    input.options.enable_thinking = enable_thinking;
    return input;
}

std::vector<ninfer::TokenId> prompt_tokens(const ninfer::Engine& engine, std::string text) {
    const ninfer::PreparedPrompt prepared = engine.prepare(chat_input(std::move(text), false));
    const auto ids                        = prepared.token_ids();
    return {ids.begin(), ids.end()};
}

struct Run {
    std::string label;
    std::vector<ninfer::TokenId> prompt;
    ninfer::GenerationResult result;
    bool greedy = false;
};

int fail(const std::string& label, const std::string& message) {
    std::cerr << "FAIL " << label << ": " << message << '\n';
    return 1;
}

// Record invariants that hold for any route: one record per token, ranked alternatives, and a
// record consistent with its own alternatives.
int check_records(const Run& run, std::uint32_t top_logprobs) {
    const auto& tokens  = run.result.generated_token_ids;
    const auto& records = run.result.content_logprobs;
    if (tokens.size() != kOutputs) { return fail(run.label, "did not generate every token"); }
    if (!run.result.reasoning_logprobs.empty()) {
        return fail(run.label, "raw output attributed a token to reasoning");
    }
    if (records.size() != tokens.size()) {
        return fail(run.label, "has " + std::to_string(records.size()) + " records for " +
                                   std::to_string(tokens.size()) + " tokens");
    }
    for (std::size_t i = 0; i < records.size(); ++i) {
        const ninfer::TokenLogprob& record = records[i];
        const std::string at               = "token " + std::to_string(i);
        if (record.token != tokens[i]) { return fail(run.label, at + " record names another id"); }
        if (!std::isfinite(record.logprob) || record.logprob > 1.0e-4F) {
            return fail(run.label, at + " logprob is not a log-probability");
        }
        if (record.top_count != top_logprobs) {
            return fail(run.label, at + " has the wrong alternative count");
        }
        const auto alternatives = record.alternatives();
        double mass             = 0.0;
        bool listed             = false;
        for (std::size_t rank = 0; rank < alternatives.size(); ++rank) {
            const ninfer::TokenAlternative& alternative = alternatives[rank];
            if (!std::isfinite(alternative.logprob) || alternative.logprob > 1.0e-4F ||
                (rank > 0 && alternative.logprob > alternatives[rank - 1].logprob)) {
                return fail(run.label, at + " alternatives are not ranked log-probabilities");
            }
            mass += std::exp(static_cast<double>(alternative.logprob));
            if (alternative.token == record.token) {
                listed = true;
                // Both values are the same logit minus the same log-sum-exp.
                if (alternative.logprob != record.logprob) {
                    return fail(run.label, at + " disagrees with its own alternative");
                }
            }
        }
        if (mass > 1.0 + 1.0e-3) {
            return fail(run.label, at + " alternatives exceed unit probability");
        }
        if (!alternatives.empty() && !listed && record.logprob > alternatives.back().logprob) {
            return fail(run.label, at + " outranks the alternatives it is missing from");
        }
        if (run.greedy && !alternatives.empty() && alternatives.front().token != record.token) {
            return fail(run.label, at + " greedy token " + std::to_string(record.token) +
                                       " is not rank 0 (" +
                                       std::to_string(alternatives.front().token) + ")");
        }
    }
    return 0;
}

struct NllAgreement {
    double mean_abs = 0.0;
    double max_abs  = 0.0;
};

// Teacher-forces the run's tokens through `scorer` (a non-speculative Engine). The first
// generated token comes from the prefill logits and the rest from T=1 decode, as in generation.
std::optional<NllAgreement> score_agreement(ninfer::Engine& scorer, const Run& run) {
    const auto& generated = run.result.generated_token_ids;
    const auto& records   = run.result.content_logprobs;
    const auto prompt     = static_cast<std::uint32_t>(run.prompt.size());

    std::vector<ninfer::TokenId> first = run.prompt;
    first.push_back(generated.front());
    ninfer::ScoreOptions prefill;
    prefill.schedule               = ninfer::ScoreSchedule::Prefill;
    prefill.skip_tokens            = prompt - 1;
    const ninfer::ScoreResult head = scorer.score(scorer.prepare_tokens(first, false), prefill);

    std::vector<ninfer::TokenId> all = run.prompt;
    all.insert(all.end(), generated.begin(), generated.end());
    ninfer::ScoreOptions decode;
    decode.schedule                = ninfer::ScoreSchedule::Decode;
    decode.skip_tokens             = prompt;
    const ninfer::ScoreResult tail = scorer.score(scorer.prepare_tokens(all, false), decode);

    if (head.non_finite != 0 || tail.non_finite != 0 || head.token_nlls.size() != 1 ||
        tail.token_nlls.size() + 1 != generated.size()) {
        std::cerr << "FAIL " << run.label << ": teacher-forced score has the wrong extent\n";
        return std::nullopt;
    }
    NllAgreement agreement;
    for (std::size_t i = 0; i < generated.size(); ++i) {
        const double reference = i == 0 ? head.token_nlls[0] : tail.token_nlls[i - 1];
        const double delta     = std::abs(-static_cast<double>(records[i].logprob) - reference);
        agreement.mean_abs += delta;
        agreement.max_abs = std::max(agreement.max_abs, delta);
    }
    agreement.mean_abs /= static_cast<double>(generated.size());
    return agreement;
}

class CollectingSink final : public ninfer::OutputSink {
public:
    void publish(ninfer::OutputDelta delta) override {
        auto& text    = delta.channel == ninfer::OutputChannel::Reasoning ? reasoning : content;
        auto& records = delta.channel == ninfer::OutputChannel::Reasoning ? reasoning_logprobs
                                                                          : content_logprobs;
        text += delta.text;
        records.insert(records.end(), delta.logprobs.begin(), delta.logprobs.end());
    }

    std::string content;
    std::string reasoning;
    std::vector<ninfer::TokenLogprob> content_logprobs;
    std::vector<ninfer::TokenLogprob> reasoning_logprobs;
};

bool same_records(const std::vector<ninfer::TokenLogprob>& a,
                  const std::vector<ninfer::TokenLogprob>& b) {
    return a.size() == b.size() &&
           std::equal(a.begin(), a.end(), b.begin(),
                      [](const ninfer::TokenLogprob& x, const ninfer::TokenLogprob& y) {
                          return x.token == y.token && x.logprob == y.logprob &&
                                 x.top_count == y.top_count;
                      });
}

std::string token_text(const ninfer::Engine& engine,
                       const std::vector<ninfer::TokenLogprob>& records) {
    std::string text;
    for (const ninfer::TokenLogprob& record : records) { text += engine.token_bytes(record.token); }
    return text;
}

std::string_view trim_leading_space(std::string_view text) {
    const std::size_t first = text.find_first_not_of(" \t\r\n");
    return first == std::string_view::npos ? std::string_view{} : text.substr(first);
}

// A templated thinking chat turn, streamed: records split by channel, follow the streamed
// deltas, and spell the published text.
int check_chat_turn(ninfer::Engine& engine) {
    const std::string label = "chat thinking turn";
    ninfer::RequestOptions options;
    options.execution.requested_output_tokens = 400;
    options.execution.sampling.temperature    = 0.0F;
    options.execution.allow_prefix_reuse      = false;
    options.output.top_logprobs               = 3;

    const std::string question = "What is 17 + 25? Answer with just the number.";
    CollectingSink sink;
    ninfer::GenerationHandle handle = engine.submit(engine.prepare(chat_input(question, true)),
                                                    options, ninfer::OutputDelivery::Streaming);
    const ninfer::GenerationResult result = handle.wait(&sink);
    if (result.finish_reason != ninfer::FinishReason::StopToken) {
        return fail(label, "did not finish at the model stop");
    }
    if (result.content.empty() || result.reasoning.empty()) {
        return fail(label, "produced no reasoning or no answer");
    }
    if (sink.content != result.content || sink.reasoning != result.reasoning) {
        return fail(label, "streamed text differs from the result");
    }
    if (!same_records(sink.content_logprobs, result.content_logprobs) ||
        !same_records(sink.reasoning_logprobs, result.reasoning_logprobs)) {
        return fail(label, "streamed records differ from the result");
    }
    if (result.content_logprobs.empty() || result.reasoning_logprobs.empty()) {
        return fail(label, "a published channel has no records");
    }
    // Markers and the stop token publish no text, so they have no record.
    if (result.content_logprobs.size() + result.reasoning_logprobs.size() >=
        result.generated_token_ids.size()) {
        return fail(label, "control tokens received records");
    }
    if (token_text(engine, result.reasoning_logprobs) != result.reasoning) {
        return fail(label, "reasoning records do not spell the reasoning text");
    }
    if (trim_leading_space(token_text(engine, result.content_logprobs)) !=
        trim_leading_space(result.content)) {
        return fail(label, "content records do not spell the answer text");
    }
    for (const auto* records : {&result.content_logprobs, &result.reasoning_logprobs}) {
        for (const ninfer::TokenLogprob& record : *records) {
            if (record.top_count != 3 || !std::isfinite(record.logprob)) {
                return fail(label, "record is malformed");
            }
        }
    }

    CollectingSink plain_sink;
    options.output.top_logprobs.reset();
    ninfer::GenerationHandle plain_handle = engine.submit(
        engine.prepare(chat_input(question, true)), options, ninfer::OutputDelivery::Streaming);
    const ninfer::GenerationResult plain = plain_handle.wait(&plain_sink);
    if (!plain.content_logprobs.empty() || !plain.reasoning_logprobs.empty() ||
        !plain_sink.content_logprobs.empty() || !plain_sink.reasoning_logprobs.empty()) {
        return fail(label, "records were reported without a request for them");
    }
    if (plain.generated_token_ids != result.generated_token_ids) {
        return fail(label, "requesting logprobs changed the generated tokens");
    }
    std::cout << "ok " << label << " content_tokens=" << result.content_logprobs.size()
              << " reasoning_tokens=" << result.reasoning_logprobs.size() << '\n';
    return 0;
}

// Two concurrent rows of which only `logprob_row` asks: its records must equal the solo run's,
// and the other row must report nothing.
int check_concurrent_rows(ninfer::Engine& engine, const Run& solo,
                          const std::vector<ninfer::TokenId>& other_prompt,
                          std::size_t logprob_row) {
    const std::string label = "concurrent row " + std::to_string(logprob_row);
    std::array<ninfer::GenerationHandle, 2> handles;
    for (std::size_t row = 0; row < 2; ++row) {
        const bool asks = row == logprob_row;
        handles[row] =
            engine.submit(engine.prepare_tokens(asks ? solo.prompt : other_prompt),
                          raw_options(asks ? std::optional<std::uint32_t>(20) : std::nullopt));
    }
    std::array<ninfer::GenerationResult, 2> results;
    for (std::size_t row = 0; row < 2; ++row) { results[row] = handles[row].wait(); }
    const ninfer::GenerationResult& asked = results[logprob_row];
    const ninfer::GenerationResult& quiet = results[1 - logprob_row];
    if (!quiet.content_logprobs.empty() || quiet.generated_token_ids.size() != kOutputs) {
        return fail(label, "the row that did not ask reported records or stopped early");
    }
    Run concurrent{label, solo.prompt, asked, true};
    if (const int failed = check_records(concurrent, 20); failed != 0) { return failed; }
    if (asked.generated_token_ids != solo.result.generated_token_ids) {
        return fail(label, "tokens differ from the solo run");
    }
    // Batched and single-row verify may associate BF16 reductions differently.
    for (std::size_t i = 0; i < kOutputs; ++i) {
        const double delta = std::abs(static_cast<double>(asked.content_logprobs[i].logprob) -
                                      static_cast<double>(solo.result.content_logprobs[i].logprob));
        if (delta > kCrossRouteMaxNllTolerance) {
            return fail(label, "token " + std::to_string(i) + " logprob differs from the solo run");
        }
    }
    std::cout << "ok " << label << '\n';
    return 0;
}

} // namespace

int main() {
    const char* artifact = std::getenv("NINFER_QWEN3_8_27B_NVFP4_DFLASH_WEIGHTS");
    if (artifact == nullptr || *artifact == '\0') {
        std::cout << "skip: NINFER_QWEN3_8_27B_NVFP4_DFLASH_WEIGHTS is not set\n";
        return 77;
    }

    const std::string story =
        "Write a short story about a lighthouse keeper who finds a message in a bottle.";
    const std::string code = "Write a Python function that returns the n-th Fibonacci number.";
    std::vector<Run> speculative_runs;
    std::vector<ninfer::TokenId> story_prompt;
    std::vector<ninfer::TokenId> code_prompt;

    const auto generate = [](ninfer::Engine& engine, std::string label,
                             const std::vector<ninfer::TokenId>& prompt,
                             const ninfer::RequestOptions& options, bool greedy) {
        Run run{std::move(label), prompt, {}, greedy};
        run.result = engine.generate(engine.prepare_tokens(prompt), options);
        return run;
    };

    {
        ninfer::Engine engine(chain_engine_options(artifact, 4, 2));
        story_prompt = prompt_tokens(engine, story);
        code_prompt  = prompt_tokens(engine, code);

        Run greedy = generate(engine, "DFlash chain greedy", story_prompt, raw_options(20), true);
        if (check_records(greedy, 20) != 0) { return 1; }
        if (greedy.result.speculative.accepted_tokens == 0) {
            return fail(greedy.label, "no draft was accepted, so no column past 0 was scored");
        }
        Run sampled =
            generate(engine, "DFlash chain p-less", story_prompt, raw_p_less_options(5, 7), false);
        if (check_records(sampled, 5) != 0) { return 1; }
        if (sampled.result.generated_token_ids == greedy.result.generated_token_ids) {
            return fail(sampled.label, "sampling reproduced the greedy tokens");
        }
        Run bare =
            generate(engine, "DFlash chain no alternatives", code_prompt, raw_options(0), true);
        if (check_records(bare, 0) != 0) { return 1; }

        if (check_concurrent_rows(engine, greedy, code_prompt, 0) != 0) { return 1; }
        if (check_concurrent_rows(engine, greedy, code_prompt, 1) != 0) { return 1; }
        if (check_chat_turn(engine) != 0) { return 1; }
        speculative_runs.push_back(std::move(greedy));
        speculative_runs.push_back(std::move(sampled));
        speculative_runs.push_back(std::move(bare));
    }
    {
        // A verify width beyond k+1 pins the packed best-first tree: the accepted path visits
        // nodes out of column order, so each token is scored at its own node.
        ninfer::EngineOptions options           = chain_engine_options(artifact, 7, 1);
        options.speculative.dflash_verify_width = 12;
        ninfer::Engine engine(options);
        Run greedy = generate(engine, "DFlash tree greedy", story_prompt, raw_options(20), true);
        if (check_records(greedy, 20) != 0) { return 1; }
        if (greedy.result.speculative.accepted_tokens == 0) {
            return fail(greedy.label, "no draft was accepted, so no tree node was scored");
        }
        Run sampled =
            generate(engine, "DFlash tree p-less", story_prompt, raw_p_less_options(20, 11), false);
        if (check_records(sampled, 20) != 0) { return 1; }
        speculative_runs.push_back(std::move(greedy));
        speculative_runs.push_back(std::move(sampled));
    }
    {
        // Adaptive drafting runs rounds narrower than the frame, on the compact verify panels.
        ninfer::EngineOptions options      = chain_engine_options(artifact, 5, 1);
        options.speculative.adaptive_draft = true;
        ninfer::Engine engine(options);
        Run greedy = generate(engine, "DFlash adaptive greedy", code_prompt, raw_options(20), true);
        if (check_records(greedy, 20) != 0) { return 1; }
        speculative_runs.push_back(std::move(greedy));
    }
    {
        ninfer::EngineOptions options = chain_engine_options(artifact, 4, 1);
        options.use_cuda_graph        = false;
        ninfer::Engine engine(options);
        Run greedy = generate(engine, "DFlash eager greedy", story_prompt, raw_options(20), true);
        if (check_records(greedy, 20) != 0) { return 1; }
        speculative_runs.push_back(std::move(greedy));
    }

    ninfer::Engine scorer(base_engine_options(artifact));
    Run ordinary = generate(scorer, "ordinary greedy", story_prompt, raw_options(20), true);
    if (check_records(ordinary, 20) != 0) { return 1; }
    const std::optional<NllAgreement> same_route = score_agreement(scorer, ordinary);
    if (!same_route) { return 1; }
    std::cout << ordinary.label << " nll mean_abs=" << same_route->mean_abs
              << " max_abs=" << same_route->max_abs << '\n';
    if (same_route->max_abs > kSameRouteNllTolerance) {
        return fail(ordinary.label, "logprobs disagree with same-route teacher-forced NLL");
    }
    for (const Run& run : speculative_runs) {
        const std::optional<NllAgreement> agreement = score_agreement(scorer, run);
        if (!agreement) { return 1; }
        std::cout << run.label << " nll mean_abs=" << agreement->mean_abs
                  << " max_abs=" << agreement->max_abs << '\n';
        if (agreement->mean_abs > kCrossRouteMeanNllTolerance ||
            agreement->max_abs > kCrossRouteMaxNllTolerance) {
            return fail(run.label, "logprobs disagree with ordinary-route teacher-forced NLL");
        }
    }
    std::cout << "ok\n";
    return 0;
}
