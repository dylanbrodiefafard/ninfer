#include "ninfer/engine.h"

#include <nlohmann/json.hpp>

#include <array>
#include <cmath>
#include <chrono>
#include <cstdlib>
#include <iostream>
#include <string>
#include <utility>
#include <vector>

namespace {

class ContentSink final : public ninfer::OutputSink {
public:
    void publish(ninfer::OutputDelta delta) override {
        if (delta.channel == ninfer::OutputChannel::Content) {
            text += delta.text;
            logprobs.insert(logprobs.end(), delta.logprobs.begin(), delta.logprobs.end());
        }
    }

    std::string text;
    std::vector<ninfer::TokenLogprob> logprobs;
};

ninfer::PromptInput prompt(std::string text) {
    ninfer::PromptInput input;
    input.options.enable_thinking = false;
    ninfer::ChatMessage user;
    user.parts.push_back({.text = std::move(text)});
    input.messages.push_back(std::move(user));
    return input;
}

int exercise(const char* artifact, ninfer::SpeculativeBackend backend) {
    ninfer::EngineOptions options;
    options.artifact_path             = artifact;
    options.max_context               = 1024;
    options.kv_capacity               = ninfer::KvCapacityPolicy::explicit_capacity(3072);
    options.prefill_chunk             = 128;
    options.max_concurrency           = 3;
    options.kv_cache                  = ninfer::KvCacheStorage::Nvfp4;
    options.speculative.backend       = backend;
    options.speculative.draft_tokens  = backend == ninfer::SpeculativeBackend::None ? 0 : 3;
    options.speculative.proposal_head = backend == ninfer::SpeculativeBackend::None
                                            ? ninfer::ProposalHead::Full
                                            : ninfer::ProposalHead::Optimized;
    ninfer::Engine engine(options);
    ninfer::RequestOptions request;
    request.execution.requested_output_tokens = 128;
    request.execution.sampling.temperature    = 0.0F;
    request.execution.allow_prefix_reuse      = false;

    auto schema = prompt("Return an object with answer set to approved and count set to 3.");
    schema.options.output_constraint = ninfer::OutputConstraint{
        ninfer::OutputConstraintKind::JsonSchema,
        R"({"type":"object","properties":{"answer":{"const":"approved"},"count":{"type":"integer","minimum":3,"maximum":3}},"required":["answer","count"],"additionalProperties":false})"};
    const std::string expected =
        "approved approved approved approved approved approved approved approved";
    auto grammar                      = prompt("Reply exactly: " + expected);
    grammar.options.output_constraint = ninfer::OutputConstraint{
        ninfer::OutputConstraintKind::Grammar, "root ::= \"" + expected + "\""};
    auto ordinary = prompt("Write one short greeting.");

    // Both passes admit a mixed batch. The second also exercises cached compilation.
    for (int pass = 0; pass < 2; ++pass) {
        const auto prepare_start           = std::chrono::steady_clock::now();
        auto schema_prompt                 = engine.prepare(schema);
        auto grammar_prompt                = engine.prepare(grammar);
        auto ordinary_prompt               = engine.prepare(ordinary);
        const double prepare_ms            = std::chrono::duration<double, std::milli>(
                                                 std::chrono::steady_clock::now() - prepare_start)
                                                 .count();
        auto schema_request                = request;
        schema_request.output.top_logprobs = 3;
        auto schema_handle                 = engine.submit(std::move(schema_prompt), schema_request,
                                                           ninfer::OutputDelivery::Streaming);
        auto grammar_handle                = engine.submit(std::move(grammar_prompt), request);
        auto ordinary_handle               = engine.submit(std::move(ordinary_prompt), request);
        ContentSink sink;
        const auto json  = schema_handle.wait(&sink);
        const auto ebnf  = grammar_handle.wait();
        const auto text  = ordinary_handle.wait();
        const auto value = nlohmann::json::parse(json.content);
        if (!value.is_object() || value.size() != 2 || value.at("answer") != "approved" ||
            value.at("count") != 3 || sink.text != json.content ||
            json.finish_reason != ninfer::FinishReason::StopToken || ebnf.content != expected ||
            ebnf.finish_reason != ninfer::FinishReason::StopToken || !json.tool_calls.empty() ||
            text.content.empty()) {
            std::cerr << "mixed constrained output failed: JSON=" << json.content
                      << " EBNF=" << ebnf.content << '\n';
            return 1;
        }
        if (json.content_logprobs.empty() || json.content_logprobs.size() != sink.logprobs.size() ||
            json.content_logprobs.size() + 1 != json.generated_token_ids.size() ||
            !json.reasoning_logprobs.empty() || !ebnf.content_logprobs.empty() ||
            !text.content_logprobs.empty()) {
            std::cerr << "constrained output logprob attribution failed\n";
            return 1;
        }
        std::string recorded_text;
        for (std::size_t i = 0; i < json.content_logprobs.size(); ++i) {
            const auto& record   = json.content_logprobs[i];
            const auto& streamed = sink.logprobs[i];
            if (record.token != json.generated_token_ids[i] || record.token != streamed.token ||
                record.logprob != streamed.logprob || !std::isfinite(record.logprob) ||
                record.logprob > 0.0F || record.top_count != 3 || streamed.top_count != 3) {
                std::cerr << "constrained output logprob token identity failed\n";
                return 1;
            }
            for (std::size_t rank = 0; rank < record.top_count; ++rank) {
                if (record.top[rank].token != streamed.top[rank].token ||
                    record.top[rank].logprob != streamed.top[rank].logprob) {
                    std::cerr << "constrained output streamed alternatives differ\n";
                    return 1;
                }
            }
            recorded_text += engine.token_bytes(record.token);
        }
        if (recorded_text != json.content) {
            std::cerr << "constrained output logprob records do not reconstruct JSON\n";
            return 1;
        }
        if (backend != ninfer::SpeculativeBackend::None &&
            (json.speculative.rounds == 0 || ebnf.speculative.rounds == 0)) {
            std::cerr << "constrained requests did not execute speculative verification\n";
            return 1;
        }
        std::cout << "constraint backend=" << static_cast<int>(backend) << " pass=" << pass
                  << " prepare_ms=" << prepare_ms
                  << " json_tokens=" << json.generated_token_ids.size()
                  << " json_decode_s=" << json.timings.decode_seconds
                  << " speculative_rounds=" << json.speculative.rounds << '\n';
    }
    return 0;
}

} // namespace

int main() {
    const char* dflash = std::getenv("NINFER_QWEN3_8_27B_NVFP4_DFLASH_WEIGHTS");
    const char* mtp    = std::getenv("NINFER_QWEN3_8_27B_NVFP4_MTP_WEIGHTS");
    if (dflash == nullptr && mtp == nullptr) {
        std::cout << "SKIP: constrained-output Engine test needs a Qwen3.8 NVFP4 artifact\n";
        return 77;
    }
    if (exercise(dflash != nullptr ? dflash : mtp, ninfer::SpeculativeBackend::None) != 0) {
        return 1;
    }
    if (dflash != nullptr && exercise(dflash, ninfer::SpeculativeBackend::DFlash) != 0) {
        return 1;
    }
    if (mtp != nullptr && exercise(mtp, ninfer::SpeculativeBackend::Mtp) != 0) { return 1; }
    return 0;
}
