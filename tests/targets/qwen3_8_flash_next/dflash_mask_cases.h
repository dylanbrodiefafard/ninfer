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

std::vector<std::int32_t> mask_token_counts(fn::Program& program, fn::Lane lane,
                                            ninfer::DeviceContext& device) {
    std::vector<std::int32_t> counts(fn::kVocabRows);
    CUDA_CHECK(cudaMemcpyAsync(counts.data(), program.state().token_counts(lane),
                               counts.size() * sizeof(std::int32_t), cudaMemcpyDeviceToHost,
                               device.stream));
    CUDA_CHECK(cudaStreamSynchronize(device.stream));
    return counts;
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
            // Declared tools arm the grammar mask whatever the model then emits. A reasoning
            // budget of a few tokens can make the greedy model answer in prose instead of
            // calling the tool, so only the unbudgeted case requires the call itself; budgeted
            // speculative runs must publish exactly the reference's calls.
            if (tools && budget != 0 && speculative) {
                require(result.tool_calls.size() == references[test].tool_calls.size() &&
                            std::ranges::equal(result.tool_calls, references[test].tool_calls,
                                               [](const auto& a, const auto& b) {
                                                   return a.name == b.name &&
                                                          a.arguments_json == b.arguments_json;
                                               }),
                        "masked speculation changed the published tool calls");
            }
            if (tools && budget == 0) {
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

// Select an observed accepted reasoning draft with a rejected probe, then replay
// exactly that state with a cap one body token ahead. This fixes the cap at node 1
// independently of how many drafts preceding rounds accepted. Suppressing the close
// on the second replay turns the same boundary into a genuine hypothetical clamp.
void dflash_mask_clamp_case(const fn::LoadedModel& model,
                            const ninfer::targets::qwen3_6::Frontend& frontend,
                            fn::ProgramOptions options, ninfer::DeviceContext& device) {
    namespace family                   = ninfer::targets::qwen3_6;
    options.lanes                      = 1;
    options.kv_capacity_tokens         = options.context_tokens;
    options.speculative.adaptive_draft = true;
    fn::Program program(model, options, device);
    const auto prepared = frontend.prepare(mask_chat(false, true));
    auto history = frontend.make_output_session(prepared, {}, {.thinking_token_budget = 1024});
    fn::LaneSampling sampling;
    sampling.temperature        = 0.7F;
    sampling.p_less             = true;
    sampling.seed               = 317;
    sampling.suppressed_tokens  = frontend.default_stop_policy().token_ids;
    const auto lane             = program.open_lane(2048, sampling).value();
    constexpr std::size_t words = (family::kTokenDomain + 31) / 32;
    const std::array<ninfer::TokenId, 1> root{0};
    const std::array<std::int32_t, 1> root_parent{-1};
    std::vector<std::uint32_t> root_mask(words), child_masks(2 * words);
    history.fill_token_masks(root, root_parent, root_mask);
    program.set_allowed_tokens(lane, root_mask, &history);
    const auto first =
        program.prefill_step(lane, family::PreparedPromptAccess::view(prepared).token_ids);
    Tokens published{first.next_token};
    const auto publish_one = [&](family::OutputSession& output, ninfer::TokenId token) {
        const auto decision =
            output.preview(std::span(&token, 1), 512, ninfer::FinishReason::OutputLimit);
        require(decision.accepted_tokens == 1 && !decision.reject_generated_round &&
                    !decision.finished(),
                "clamp fixture left its nonterminal reasoning prefix");
        (void)output.commit_preview();
    };
    publish_one(history, first.next_token);
    auto close_probe = frontend.make_output_session(prepared, {}, {.thinking_token_budget = 1});
    publish_one(close_probe, first.next_token);
    const auto close = close_probe.reasoning_limit_forced_close_token();
    require(close.has_value(), "clamp fixture prefill did not charge a reasoning token");
    const std::array lanes{lane};
    const std::array<std::uint32_t, 1> limits{4};
    const auto same_policy = [](const auto& a, const auto& b) {
        return a.reached == b.reached && a.accepted == b.accepted &&
               a.low_windows == b.low_windows && a.width == b.width && a.ceiling == b.ceiling;
    };
    for (const bool suppress_close : {false, true}) {
        if (suppress_close) { sampling.suppressed_tokens.push_back(close.value()); }
        program.set_sampling(lane, sampling);
        family::OutputSession capped;
        bool selected = false;
        for (std::size_t attempt = 0; attempt < 16 && !selected; ++attempt) {
            capped = frontend.make_output_session(
                prepared, {},
                {.thinking_token_budget = history.reasoning_limit_stats().used_tokens + 1});
            for (const auto token : published) { publish_one(capped, token); }
            history.fill_token_masks(root, root_parent, root_mask);
            program.set_allowed_tokens(lane, root_mask, &history);
            const std::array anchors{published.back()};
            const auto probe =
                program.decode_round(lanes, anchors, limits, fn::RoundLogprobs::Skip);
            const auto token = probe.tokens[0];
            if (probe.counts[0] > 1) {
                const std::array nodes{published.back(), token};
                const std::array<std::int32_t, 2> parents{-1, 0};
                capped.fill_token_masks(nodes, parents, child_masks);
                std::size_t allowed = 0;
                for (std::size_t word = words; word < 2 * words; ++word) {
                    allowed += static_cast<std::size_t>(std::popcount(child_masks[word]));
                }
                selected = allowed == 1 &&
                           (child_masks[words + static_cast<std::size_t>(close.value()) / 32] &
                            (1U << (static_cast<unsigned>(close.value()) % 32))) != 0;
            }
            const std::array<std::int32_t, 1> commit{selected ? 0 : 1};
            program.resolve_round(commit);
            if (!selected) {
                publish_one(history, token);
                published.push_back(token);
            }
        }
        require(selected, "clamp fixture found no accepted reasoning draft in 16 probes");
        const auto before_counts = mask_token_counts(program, lane, device);
        const auto before_policy = program.state().dflash_lane(lane).policy;
        capped.fill_token_masks(root, root_parent, root_mask);
        program.set_allowed_tokens(lane, root_mask, &capped);
        const std::array anchors{published.back()};
        const auto packet = program.decode_round(lanes, anchors, limits, fn::RoundLogprobs::Skip);
        require(packet.width == 4 && !packet.row_errors[0], "hypothetical cap became a root error");
        if (!suppress_close) {
            require(packet.counts[0] > 1 && packet.tokens[1] == close.value(),
                    "controlled cap did not license its singleton close at column one");
            const std::array<std::int32_t, 1> discard{0};
            program.resolve_round(discard);
            require(mask_token_counts(program, lane, device) == before_counts,
                    "rejected non-root close changed token counts");
        } else {
            require(packet.counts[0] == 1 && packet.counts[0] < packet.raw_counts[0] &&
                        packet.counts[0] < packet.width - 1,
                    "suppressed hypothetical close did not clamp a nonempty raw walk");
            const auto token = packet.tokens[0];
            const std::array<std::int32_t, 1> commit{packet.counts[0]};
            program.resolve_round(commit);
            auto expected = before_counts;
            ++expected[static_cast<std::size_t>(token)];
            require(mask_token_counts(program, lane, device) == expected,
                    "hypothetical clamp retained unpublished token counts");
            publish_one(capped, token);
            require(capped.reasoning_limit_forced_close_token() == close,
                    "safe prefix did not reach the failing root policy");
            capped.fill_token_masks(root, root_parent, root_mask);
            program.set_allowed_tokens(lane, root_mask, &capped);
            const std::array next_anchor{token};
            const auto failure =
                program.decode_round(lanes, next_anchor, limits, fn::RoundLogprobs::Skip);
            require(failure.row_errors[0] && failure.counts[0] == 0,
                    "hypothetical error did not become request-local on the next arm");
            const std::array<std::int32_t, 1> discard{0};
            program.resolve_round(discard);
            require(mask_token_counts(program, lane, device) == expected,
                    "root retry after hypothetical clamp changed token counts");
        }
        require(same_policy(before_policy, program.state().dflash_lane(lane).policy),
                "rejected or matcher-clamped walk trained the adaptive policy");
    }
    program.close_lane(lane);
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
    {
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
            std::array<fn::LaneSampling, 4> sampling_by_lane;
            std::array<bool, 4> suppress_model_stops{};
            const auto arm_suppression = [&](std::size_t lane) {
                auto& sampling = sampling_by_lane[lane];
                sampling.suppressed_tokens.clear();
                if (suppress_model_stops[lane]) {
                    sampling.suppressed_tokens = frontend.default_stop_policy().token_ids;
                }
                if (const auto close = sessions[lane].suppressed_reasoning_end_token()) {
                    sampling.suppressed_tokens.push_back(close.value());
                }
                program.set_sampling(owners[lane], sampling);
            };
            for (std::size_t lane = 0; lane < owners.size(); ++lane) {
                fn::LaneSampling sampling;
                sampling.temperature   = p_less ? 0.7F : 0.0F;
                sampling.p_less        = p_less;
                sampling.seed          = 100 + lane;
                sampling_by_lane[lane] = sampling;
                owners[lane]           = program.open_lane(2048, sampling).value();
                require(owners[lane] == static_cast<fn::Lane>(lane),
                        "unexpected Program lane allocation");
                Tokens tokens;
                if (lane == 0 || lane == 3) {
                    auto prepared = frontend.prepare(mask_chat(lane == 3, lane == 0));
                    ninfer::OutputOptions output;
                    if (lane == 0) { output.thinking_token_budget = 4; }
                    sessions[lane]             = frontend.make_output_session(prepared, {}, output);
                    suppress_model_stops[lane] = !sessions[lane].model_stop_tokens_allowed();
                    arm_suppression(lane);
                    tokens = family::PreparedPromptAccess::view(prepared).token_ids;
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
                    require(decision.accepted_tokens == 1 && !decision.reject_generated_round &&
                                !decision.finished(),
                            "mixed mask prompt terminated before decode");
                    (void)sessions[lane].commit_preview();
                    suppress_model_stops[lane] = !sessions[lane].model_stop_tokens_allowed();
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
                        arm_suppression(lane);
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
                std::array<bool, 4> rejected{};
                ragged =
                    ragged || (packet.width == 4 && lanes.size() == 4 && packet.counts[1] == 1);
                for (std::size_t row = 0; row < lanes.size(); ++row) {
                    const auto lane = static_cast<std::size_t>(lanes[row]);
                    require(packet.row_errors.empty() || !packet.row_errors[row],
                            "mixed mask row unexpectedly failed");
                    auto tokens =
                        packet.tokens.subspan(row * static_cast<std::size_t>(packet.width),
                                              static_cast<std::size_t>(packet.counts[row]));
                    if (lane == 0 || lane == 3) {
                        std::array<ninfer::TokenId, 4> nodes{anchors[lane]};
                        for (std::size_t node = 1; node < tokens.size(); ++node) {
                            nodes[node] = tokens[node - 1];
                        }
                        sessions[lane].fill_token_masks(
                            std::span(nodes).first(tokens.size()),
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
                        const auto decision = sessions[lane].preview(
                            tokens, remaining[lane], ninfer::FinishReason::OutputLimit);
                        committed[row] = static_cast<std::int32_t>(decision.accepted_tokens);
                        if (decision.reject_generated_round) {
                            require(decision.accepted_tokens == 0 && !decision.finished() &&
                                        !suppress_model_stops[lane],
                                    "mixed sampler rejected a round with model stops already "
                                    "suppressed");
                            suppress_model_stops[lane] = true;
                            rejected[lane]             = true;
                            continue;
                        }
                        if (lane == 3) {
                            for (std::size_t node = 1; node < decision.accepted_tokens; ++node) {
                                const auto begin =
                                    masks.begin() + static_cast<std::ptrdiff_t>(node * words);
                                std::size_t allowed = 0;
                                for (auto it = begin;
                                     it != begin + static_cast<std::ptrdiff_t>(words); ++it) {
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
                        suppress_model_stops[lane] = !sessions[lane].model_stop_tokens_allowed();
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
                    if (rejected[static_cast<std::size_t>(lane)]) {
                        sessions[static_cast<std::size_t>(lane)].discard_preview();
                    }
                    std::vector<std::int32_t> actual(fn::kVocabRows);
                    CUDA_CHECK(cudaMemcpyAsync(actual.data(), program.state().token_counts(lane),
                                               actual.size() * sizeof(std::int32_t),
                                               cudaMemcpyDeviceToHost, device.stream));
                    CUDA_CHECK(cudaStreamSynchronize(device.stream));
                    require(actual == expected_counts[static_cast<std::size_t>(lane)],
                            "clipped mask round left uncommitted sampling counts");
                }
            }
            require(ragged && compacted && binding_nonroot &&
                        program.speculative_stats(owners[3]).accepted_tokens > 0,
                    "mixed mask test missed ragged/compact or binding accepted child evidence");
            std::cout << "DFlash mixed masks p_less=" << p_less
                      << " accepted=" << program.speculative_stats(owners[3]).accepted_tokens
                      << " incidental_nonroot_close=" << close_nonroot << '\n';
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
        const auto first        = program.prefill_step(
            failed_lane, family::PreparedPromptAccess::view(prepared).token_ids);
        (void)output.preview(std::span(&first.next_token, 1), 128,
                             ninfer::FinishReason::OutputLimit);
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
        const auto counts_before = mask_token_counts(program, failed_lane, device);
        const auto packet = program.decode_round(lanes, anchors, limits, fn::RoundLogprobs::Skip);
        require(!packet.row_errors[0] && packet.row_errors[1] && packet.counts[1] == 0 &&
                    packet.counts[0] > 0,
                "root matcher failure escaped its compact row");
        const std::array committed{packet.counts[0], 0};
        program.resolve_round(committed);
        program.state().pack_continuation(failed_lane, after.data(), device.stream);
        CUDA_CHECK(cudaStreamSynchronize(device.stream));
        require(std::memcmp(before.data(), after.data(), before.size()) == 0,
                "zero-publication matcher row changed canonical continuation");
        require(mask_token_counts(program, failed_lane, device) == counts_before,
                "zero-publication matcher row changed token counts");
        program.close_lane(failed_lane);
        program.close_lane(healthy_lane);
    }
    dflash_mask_clamp_case(model, frontend, options, device);
}
