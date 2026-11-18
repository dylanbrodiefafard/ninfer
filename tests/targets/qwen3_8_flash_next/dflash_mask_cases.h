// Included inside the real-artifact test's namespace after its Engine helpers.
ninfer::PromptInput mask_chat(bool tools, bool thinking) {
    ninfer::PromptInput chat;
    chat.options.enable_thinking = thinking;
    chat.messages.push_back(ninfer::ChatMessage{
        .role  = ninfer::ChatRole::User,
        .parts = {ninfer::MessagePart{
            .text = thinking ? "Compute 93847 * 76239 step by step. Then call record_result with "
                               "value true if available."
                             : "Call record_result with value true. Do not answer in prose."}}});
    if (tools) {
        chat.options.tool_jsons.push_back(
            R"({"type":"function","function":{"name":"record_result","description":"Record the requested result.","parameters":{"type":"object","properties":{"value":{"type":"boolean","enum":[true]}},"required":["value"],"additionalProperties":false}}})");
    }
    return chat;
}

// Ordinary State root masks and Program::decode are independent of the speculative
// mask graph. Budget sweeps avoid qualifying only a close immediately after prefill.
void dflash_mask_cases(const Inputs& inputs) {
    struct Case {
        bool tools;
        std::uint32_t budget;
    };

    std::vector<Case> cases{{true, 0}};
    for (const bool tools : {false, true}) {
        for (std::uint32_t budget = 2; budget <= 6; ++budget) { cases.push_back({tools, budget}); }
    }
    std::vector<ninfer::GenerationResult> references(cases.size());
    for (const bool speculative : {false, true}) {
        auto options        = engine_options(inputs, 1, 4096, 4096);
        options.speculative = {};
        if (speculative) {
            options.speculative.backend      = ninfer::SpeculativeBackend::DFlash;
            options.speculative.draft_tokens = 3;
        }
        ninfer::Engine engine(options);
        std::uint64_t budget_accepted = 0;
        for (std::size_t test = 0; test < cases.size(); ++test) {
            const auto [tools, budget]          = cases[test];
            auto request                        = greedy(128);
            request.stop.include_model_defaults = true;
            if (budget != 0) { request.output.thinking_token_budget = budget; }
            const auto result =
                engine.generate(engine.prepare(mask_chat(tools, budget != 0)), request);
            if (!speculative) {
                references[test] = result;
                require(!result.speculative.enabled && result.speculative.drafted_tokens == 0,
                        "masked reference enabled a speculative path");
            } else {
                require(result.speculative.rounds > 0 && result.speculative.drafted_tokens > 0,
                        "session-backed output mask disabled DFlash drafting");
                require_equal(result.generated_token_ids, references[test].generated_token_ids,
                              "masked greedy K3 against ordinary decode case " +
                                  std::to_string(test));
                require(result.content == references[test].content &&
                            result.reasoning == references[test].reasoning &&
                            result.finish_reason == references[test].finish_reason,
                        "masked speculation changed output publication");
                require(result.reasoning_limit.used_tokens ==
                                references[test].reasoning_limit.used_tokens &&
                            result.reasoning_limit.forced_closes ==
                                references[test].reasoning_limit.forced_closes,
                        "masked speculation changed reasoning accounting");
                if (budget == 0) {
                    require(result.speculative.accepted_tokens > 0,
                            "tool mask regression exercised no accepted draft");
                } else {
                    budget_accepted += result.speculative.accepted_tokens;
                }
            }
            if (budget != 0) {
                require(result.reasoning_limit.used_tokens == budget &&
                            result.reasoning_limit.forced_closes == 1,
                        "masked chat did not exercise the reasoning-budget forced close");
            }
            if (tools) {
                require(result.tool_calls.size() == 1 &&
                            result.tool_calls.front().name == "record_result" &&
                            nlohmann::json::parse(result.tool_calls.front().arguments_json) ==
                                nlohmann::json{{"value", true}},
                        "masked chat failed to publish its schema-valid tool call");
            }
            std::cout << "DFlash masks enabled=" << speculative << " case=" << test
                      << " drafted=" << result.speculative.drafted_tokens
                      << " accepted=" << result.speculative.accepted_tokens << '\n';
        }
        if (speculative) { require(budget_accepted > 0, "budget sweep accepted no drafts"); }
    }
}

// Target-level evidence observes actual compact packets before frontend clipping.
// Physical membership [3,1,0,2] mixes session, one-shot and unmasked rows with
// extents [3,0,2,1], then drains them independently. Both greedy and p-less run.
void dflash_mask_program_cases(const Inputs& inputs) {
    namespace family = ninfer::targets::qwen3_6;
    ninfer::DeviceContext device(0);
    const ninfer::artifact::Reader reader(inputs.artifact);
    const fn::LoadedModel model(reader, device);
    const auto frontend = fn::Package::make_frontend(model);
    fn::ProgramOptions options;
    options.ngram_table              = inputs.ngram_table;
    options.lanes                    = 4;
    options.context_tokens           = 2048;
    options.kv_capacity_tokens       = 8192;
    options.speculative.backend      = ninfer::SpeculativeBackend::DFlash;
    options.speculative.draft_tokens = 3;
    fn::Program program(model, options, device);
    constexpr std::size_t words = (family::kTokenDomain + 31) / 32;
    const std::array<ninfer::TokenId, 1> root{0};
    const std::array<std::int32_t, 1> root_parent{-1};
    const std::array<std::int32_t, 4> parents{-1, 0, 1, 2};
    std::vector<std::uint32_t> root_mask(words), masks(4 * words), one_shot(words, 0);
    constexpr ninfer::TokenId fixed_token = 42;
    one_shot[fixed_token / 32]            = 1U << (fixed_token % 32);
    for (const bool p_less : {false, true}) {
        std::array<family::OutputSession, 4> sessions;
        std::array<fn::Lane, 4> owners{};
        std::array<ninfer::TokenId, 4> anchors{};
        std::array<std::uint32_t, 4> remaining{128, 16, 16, 128};
        std::array<std::vector<std::int32_t>, 4> expected_counts;
        for (std::size_t lane = 0; lane < owners.size(); ++lane) {
            fn::LaneSampling sampling;
            sampling.temperature = p_less ? 0.7F : 0.0F;
            sampling.p_less      = p_less;
            sampling.seed        = 100 + lane;
            owners[lane]         = program.open_lane(2048, sampling).value();
            require(owners[lane] == static_cast<fn::Lane>(lane),
                    "unexpected Program lane allocation");
            Tokens tokens;
            if (lane == 0 || lane == 3) {
                auto prepared = frontend.prepare(mask_chat(lane == 3, lane == 0));
                ninfer::OutputOptions output;
                if (lane == 0) { output.thinking_token_budget = 4; }
                sessions[lane] = frontend.make_output_session(prepared, {}, output);
                tokens         = family::PreparedPromptAccess::view(prepared).token_ids;
                sessions[lane].fill_token_masks(root, root_parent, root_mask);
                program.set_allowed_tokens(owners[lane], root_mask, &sessions[lane]);
            } else {
                tokens = prompt_tokens(32U + static_cast<std::uint32_t>(lane),
                                       600U + static_cast<std::uint32_t>(lane));
                if (lane == 1) { program.set_allowed_tokens(owners[lane], one_shot); }
            }
            const auto first = program.prefill_step(owners[lane], tokens);
            require(first.complete, "mixed mask prefill did not complete");
            anchors[lane] = first.next_token;
            if (lane == 0 || lane == 3) {
                const auto decision =
                    sessions[lane].preview(std::span(&anchors[lane], 1), remaining[lane],
                                           ninfer::FinishReason::OutputLimit);
                (void)sessions[lane].commit_preview();
                require(!decision.finished(), "mixed mask prompt terminated before decode");
            }
            --remaining[lane];
            expected_counts[lane].resize(fn::kVocabRows);
            CUDA_CHECK(cudaMemcpyAsync(expected_counts[lane].data(),
                                       program.state().token_counts(owners[lane]),
                                       expected_counts[lane].size() * sizeof(std::int32_t),
                                       cudaMemcpyDeviceToHost, device.stream));
        }
        CUDA_CHECK(cudaStreamSynchronize(device.stream));
        bool binding_nonroot = false, close_nonroot = false, ragged = false, compacted = false;
        for (;;) {
            std::vector<fn::Lane> lanes;
            Tokens inputs_now;
            std::vector<std::uint32_t> limits;
            for (const std::size_t lane : {3U, 1U, 0U, 2U}) {
                if (remaining[lane] == 0) { continue; }
                if (lane == 0 || lane == 3) {
                    sessions[lane].fill_token_masks(root, root_parent, root_mask);
                    program.set_allowed_tokens(owners[lane], root_mask, &sessions[lane]);
                } else if (lane == 1) {
                    program.set_allowed_tokens(owners[lane], one_shot);
                }
                lanes.push_back(owners[lane]);
                inputs_now.push_back(anchors[lane]);
                const std::uint32_t width =
                    lane == 3 ? 4U : (lane == 0 ? 3U : (lane == 1 ? 1U : 2U));
                limits.push_back(std::min(remaining[lane], width));
            }
            if (lanes.empty()) { break; }
            compacted = compacted || lanes.size() < 4;
            const auto packet =
                program.decode_round(lanes, inputs_now, limits, fn::RoundLogprobs::Skip);
            std::vector<std::int32_t> committed(packet.counts.begin(), packet.counts.end());
            ragged = ragged || (packet.width == 4 && lanes.size() == 4 && packet.counts[1] == 1);
            for (std::size_t row = 0; row < lanes.size(); ++row) {
                const auto lane = static_cast<std::size_t>(lanes[row]);
                require(packet.row_errors.empty() || !packet.row_errors[row],
                        "mixed mask row unexpectedly failed");
                auto tokens = packet.tokens.subspan(row * static_cast<std::size_t>(packet.width),
                                                    static_cast<std::size_t>(packet.counts[row]));
                if (lane == 0 || lane == 3) {
                    std::array<ninfer::TokenId, 4> nodes{anchors[lane]};
                    for (std::size_t node = 1; node < tokens.size(); ++node) {
                        nodes[node] = tokens[node - 1];
                    }
                    sessions[lane].fill_token_masks(std::span(nodes).first(tokens.size()),
                                                    std::span(parents).first(tokens.size()),
                                                    std::span(masks).first(tokens.size() * words));
                    for (std::size_t node = 0; node < tokens.size(); ++node) {
                        const auto token = static_cast<std::size_t>(tokens[node]);
                        require((masks[node * words + token / 32] & (1U << (token % 32))) != 0,
                                "licensed token violates its exact hypothetical-prefix mask");
                        if (lane == 0 && node > 0) {
                            std::size_t allowed = 0;
                            for (std::size_t word = 0; word < words; ++word) {
                                allowed += static_cast<std::size_t>(
                                    std::popcount(masks[node * words + word]));
                            }
                            close_nonroot = close_nonroot || allowed == 1;
                        }
                    }
                    const auto decision = sessions[lane].preview(tokens, remaining[lane],
                                                                 ninfer::FinishReason::OutputLimit);
                    committed[row]      = static_cast<std::int32_t>(decision.accepted_tokens);
                    if (lane == 3) {
                        for (std::size_t node = 1; node < decision.accepted_tokens; ++node) {
                            const auto begin =
                                masks.begin() + static_cast<std::ptrdiff_t>(node * words);
                            std::size_t allowed = 0;
                            for (auto it = begin; it != begin + static_cast<std::ptrdiff_t>(words);
                                 ++it) {
                                allowed += static_cast<std::size_t>(std::popcount(*it));
                            }
                            binding_nonroot =
                                binding_nonroot ||
                                (allowed < family::kTokenDomain / 2 &&
                                 !std::equal(begin, begin + static_cast<std::ptrdiff_t>(words),
                                             masks.begin()));
                        }
                    }
                    (void)sessions[lane].commit_preview();
                    if (decision.finished()) {
                        remaining[lane] = static_cast<std::uint32_t>(committed[row]);
                    }
                } else if (lane == 1) {
                    require(tokens.size() == 1 && tokens.front() == fixed_token,
                            "one-shot domain leaked across compact rows");
                }
                for (std::int32_t i = 0; i < committed[row]; ++i) {
                    if (p_less) {
                        ++expected_counts[lane][static_cast<std::size_t>(
                            tokens[static_cast<std::size_t>(i)])];
                    }
                }
                require(committed[row] > 0, "mixed row published no output");
                anchors[lane] = tokens[static_cast<std::size_t>(committed[row] - 1)];
                remaining[lane] -= static_cast<std::uint32_t>(committed[row]);
            }
            program.resolve_round(committed);
            for (const auto lane : lanes) {
                std::vector<std::int32_t> actual(fn::kVocabRows);
                CUDA_CHECK(cudaMemcpyAsync(actual.data(), program.state().token_counts(lane),
                                           actual.size() * sizeof(std::int32_t),
                                           cudaMemcpyDeviceToHost, device.stream));
                CUDA_CHECK(cudaStreamSynchronize(device.stream));
                require(actual == expected_counts[static_cast<std::size_t>(lane)],
                        "clipped mask round left uncommitted sampling counts");
            }
        }
        require(ragged && compacted && binding_nonroot && close_nonroot &&
                    program.speculative_stats(owners[3]).accepted_tokens > 0,
                "mixed mask test missed ragged/compact, binding accepted child or non-root "
                "forced-close evidence");
        std::cout << "DFlash mixed masks p_less=" << p_less
                  << " accepted=" << program.speculative_stats(owners[3]).accepted_tokens << '\n';
        for (const auto lane : owners) { program.close_lane(lane); }
    }
    // A root matcher failure still ran device acceptance. Resolve that row with
    // zero while another row progresses, restoring counts and pending draft context.
    auto prepared = frontend.prepare(mask_chat(false, true));
    auto output   = frontend.make_output_session(prepared, {}, {.thinking_token_budget = 1});
    fn::LaneSampling sampling;
    sampling.temperature    = 0.7F;
    sampling.p_less         = true;
    const auto failed_lane  = program.open_lane(2048, sampling).value();
    const auto healthy_lane = program.open_lane(2048, sampling).value();
    const auto first =
        program.prefill_step(failed_lane, family::PreparedPromptAccess::view(prepared).token_ids);
    (void)output.preview(std::span(&first.next_token, 1), 128, ninfer::FinishReason::OutputLimit);
    (void)output.commit_preview();
    const auto close = output.reasoning_limit_forced_close_token();
    require(close.has_value(), "root-error fixture did not reach its reasoning cap");
    sampling.suppressed_tokens = {close.value()};
    program.set_sampling(failed_lane, sampling);
    output.fill_token_masks(root, root_parent, root_mask);
    program.set_allowed_tokens(failed_lane, root_mask, &output);
    const auto healthy = program.prefill_step(healthy_lane, prompt_tokens(32, 904));
    ninfer::PinnedHostBuffer before(program.state().continuation_bytes()), after(before.size());
    program.state().pack_continuation(failed_lane, before.data(), device.stream);
    CUDA_CHECK(cudaStreamSynchronize(device.stream));
    const std::array lanes{healthy_lane, failed_lane};
    const std::array anchors{healthy.next_token, first.next_token};
    const std::array<std::uint32_t, 2> limits{2, 4};
    const auto packet = program.decode_round(lanes, anchors, limits, fn::RoundLogprobs::Skip);
    require(!packet.row_errors[0] && packet.row_errors[1] && packet.counts[1] == 0 &&
                packet.counts[0] > 0,
            "root matcher failure escaped its compact row");
    const std::array committed{packet.counts[0], 0};
    program.resolve_round(committed);
    program.state().pack_continuation(failed_lane, after.data(), device.stream);
    CUDA_CHECK(cudaStreamSynchronize(device.stream));
    require(std::memcmp(before.data(), after.data(), before.size()) == 0,
            "zero-publication matcher row changed counts or canonical continuation");
    program.close_lane(failed_lane);
    program.close_lane(healthy_lane);
}
