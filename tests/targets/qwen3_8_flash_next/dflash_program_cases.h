// Included inside the test namespace by test_engine_real.cpp. The same target schedule is
// exercised through fixed and forced-adaptive selection; width-dependent arithmetic is allowed.

void dflash_program_cases(const Inputs& inputs) {
    ninfer::DeviceContext device(0);
    const ninfer::artifact::Reader reader(inputs.artifact);
    const fn::LoadedModel model(reader, device);
    require(model.view().dflash.has_value(), "DFlash cases require the companion artifact");

    {
        fn::ProgramOptions options;
        options.ngram_table              = inputs.ngram_table;
        options.lanes                    = 2;
        options.context_tokens           = 128;
        options.kv_capacity_tokens       = 256;
        options.speculative.backend      = ninfer::SpeculativeBackend::DFlash;
        options.speculative.draft_tokens = 0;
        fn::Program program(model, options, device);
        std::array prompts{prompt_tokens(1, 73), prompt_tokens(1, 901)};
        std::array<fn::Lane, 2> lanes{};
        std::array<std::int32_t, 2> anchors{};
        const auto bytes = program.state().continuation_bytes();
        std::array before{ninfer::PinnedHostBuffer(bytes), ninfer::PinnedHostBuffer(bytes)};
        std::array actual{ninfer::PinnedHostBuffer(bytes), ninfer::PinnedHostBuffer(bytes)};
        ninfer::PinnedHostBuffer expected(bytes);
        for (std::size_t row = 0; row < lanes.size(); ++row) {
            lanes[row]         = program.open_lane(32, {}).value();
            const auto prefill = program.prefill_step(lanes[row], prompts[row]);
            require(prefill.complete, "root-pair prefill did not complete");
            anchors[row] = prefill.next_token;
            program.state().pack_continuation(lanes[row], before[row].data(), device.stream);
        }
        CUDA_CHECK(cudaStreamSynchronize(device.stream));
        const std::array<std::uint32_t, 2> limits{4, 4};
        for (std::int32_t step = 0; step < 2; ++step) {
            static_cast<void>(
                program.decode_round(lanes, anchors, limits, fn::RoundLogprobs::Skip));
            const std::array<std::int32_t, 2> reject{0, 0};
            program.resolve_round(reject);
            for (std::size_t row = 0; row < lanes.size(); ++row) {
                program.state().pack_continuation(lanes[row], actual[row].data(), device.stream);
            }
            CUDA_CHECK(cudaStreamSynchronize(device.stream));
            for (std::size_t row = 0; row < lanes.size(); ++row) {
                require(std::memcmp(before[row].data(), actual[row].data(), bytes) == 0,
                        "two-row root rejection changed canonical state");
            }
            static_cast<void>(
                program.decode_round(lanes, anchors, limits, fn::RoundLogprobs::Skip));
            const std::array<std::int32_t, 2> accept{1, 1};
            program.resolve_round(accept);
            for (std::size_t row = 0; row < lanes.size(); ++row) {
                program.state().pack_continuation(lanes[row], actual[row].data(), device.stream);
            }
            CUDA_CHECK(cudaStreamSynchronize(device.stream));
            for (std::size_t row = 0; row < lanes.size(); ++row) {
                program.reuse_lane(lanes[row], 32, prompts[row], {}, before[row].data());
                require(program.finish_prompt(lanes[row]).next_token == anchors[row],
                        "root-pair anchor changed");
            }
            std::array<std::int32_t, 2> next{};
            program.decode(lanes, anchors, next);
            for (std::size_t row = 0; row < lanes.size(); ++row) {
                program.state().pack_continuation(lanes[row], expected.data(), device.stream);
                CUDA_CHECK(cudaStreamSynchronize(device.stream));
                require(std::memcmp(actual[row].data(), expected.data(), bytes) == 0,
                        "two-row deferred root differs from ordinary same-batch width-one decode");
                std::memcpy(before[row].data(), expected.data(), bytes);
                prompts[row].push_back(anchors[row]);
            }
            anchors = next;
        }
        for (const auto lane : lanes) { program.close_lane(lane); }
    }

    {
        fn::ProgramOptions options;
        options.ngram_table                = inputs.ngram_table;
        options.lanes                      = 3;
        options.context_tokens             = 128;
        options.kv_capacity_tokens         = 384;
        options.speculative.backend        = ninfer::SpeculativeBackend::DFlash;
        options.speculative.draft_tokens   = 3;
        options.speculative.adaptive_draft = true;
        fn::Program program(model, options, device);
        std::array<fn::Lane, 3> lanes{};
        std::array<std::int32_t, 3> anchors{};
        for (std::size_t row = 0; row < lanes.size(); ++row) {
            lanes[row]        = program.open_lane(32, {}).value();
            const auto prompt = prompt_tokens(1, 73U + static_cast<std::uint32_t>(row));
            anchors[row]      = program.prefill_step(lanes[row], prompt).next_token;
        }
        program.state().dflash_lane(lanes[0]).policy.ceiling = 0;
        program.state().dflash_lane(lanes[2]).policy.ceiling = 0;
        const std::array<std::uint32_t, 3> limits{4, 4, 4};
        ninfer::PinnedHostBuffer before(program.state().continuation_bytes());
        ninfer::PinnedHostBuffer after(before.size());
        program.state().pack_continuation(lanes[1], before.data(), device.stream);
        CUDA_CHECK(cudaStreamSynchronize(device.stream));
        static_cast<void>(program.decode_round(lanes, anchors, limits, fn::RoundLogprobs::Skip));
        const std::array<std::int32_t, 3> reject{0, 0, 0};
        program.resolve_round(reject);
        program.state().pack_continuation(lanes[1], after.data(), device.stream);
        CUDA_CHECK(cudaStreamSynchronize(device.stream));
        require(std::memcmp(before.data(), after.data(), before.size()) == 0,
                "rejected default round changed the request's table ceiling");
        for (const auto ceiling : {2, 1, 0}) {
            if (ceiling != 2) { program.state().dflash_lane(lanes[1]).policy.ceiling = ceiling; }
            const auto round =
                program.decode_round(lanes, anchors, limits, fn::RoundLogprobs::Skip);
            require(round.width == ceiling + 1 && round.counts[0] == 1,
                    "request step-down did not bound the shared width and root-only row");
            std::array<std::int32_t, 3> counts{round.counts[0], round.counts[1], round.counts[2]};
            for (std::size_t row = 0; row < lanes.size(); ++row) {
                anchors[row] = round.tokens[row * static_cast<std::size_t>(round.width) +
                                            static_cast<std::size_t>(counts[row] - 1)];
            }
            program.resolve_round(counts);
            if (ceiling == 2) {
                const auto drain = program.decode_round(
                    std::span(lanes).subspan(1, 1), std::span(anchors).subspan(1, 1),
                    std::span(limits).subspan(1, 1), fn::RoundLogprobs::Skip);
                require(drain.width == 3, "batch drain raised an existing request's ceiling");
                const std::array committed{drain.counts[0]};
                anchors[1] = drain.tokens[static_cast<std::size_t>(committed[0] - 1)];
                program.resolve_round(committed);
            }
        }
        const auto root     = program.speculative_stats(lanes[0]);
        const auto drafting = program.speculative_stats(lanes[1]);
        require(root.drafted_tokens == 0 && root.fallback_steps == 3 &&
                    drafting.rounds_per_draft[2] == 2 && drafting.rounds_per_draft[1] == 1 &&
                    drafting.rounds_per_draft[0] == 1,
                "request-local step-down changed the other row's arm history");
        for (const auto lane : lanes) { program.close_lane(lane); }
    }

    struct Case {
        std::int32_t lanes, k;
        float temperature;
        bool p_less;
    };

    for (const auto test :
         {Case{1, 1, 0.0F, false}, Case{1, 3, 0.0F, false}, Case{1, 3, 0.6F, true},
          Case{2, 2, 0.7F, false}, Case{1, 0, 0.6F, true}}) {
        const auto execute = [&](bool adaptive, bool force = true) {
            fn::ProgramOptions options;
            options.ngram_table                = inputs.ngram_table;
            options.lanes                      = test.lanes;
            options.context_tokens             = 4096;
            options.kv_capacity_tokens         = std::int64_t{4096} * test.lanes;
            options.speculative.backend        = ninfer::SpeculativeBackend::DFlash;
            options.speculative.draft_tokens   = adaptive ? 3U : static_cast<std::uint32_t>(test.k);
            options.speculative.adaptive_draft = adaptive;
            if (adaptive && force) {
                options.forced_adaptive_k = static_cast<std::uint32_t>(test.k);
            }
            fn::Program program(model, options, device);
            std::vector<fn::Lane> owners;
            std::vector<Tokens> output(static_cast<std::size_t>(test.lanes));
            for (std::int32_t row = 0; row < test.lanes; ++row) {
                fn::LaneSampling sampling;
                sampling.temperature       = test.temperature;
                sampling.p_less            = test.p_less;
                sampling.seed              = 100U + static_cast<std::uint64_t>(row);
                sampling.presence_penalty  = test.p_less ? 0.0F : 0.2F;
                sampling.frequency_penalty = test.p_less ? 0.0F : 0.1F;
                const auto lane            = program.open_lane(4096, sampling).value();
                owners.push_back(lane);
                const Tokens prompt =
                    prompt_tokens(test.k == 3 && test.temperature == 0.0F ? 2047U : 512U,
                                  73U + static_cast<std::uint32_t>(row));
                auto remaining = std::span<const ninfer::TokenId>(prompt);
                while (!remaining.empty()) {
                    const auto step = program.prefill_step(lane, remaining);
                    remaining       = remaining.subspan(static_cast<std::size_t>(step.consumed));
                    if (step.complete) {
                        output[static_cast<std::size_t>(row)].push_back(step.next_token);
                    }
                }
            }
            bool rejected_once     = false;
            bool partial_once      = false;
            bool published_once    = false;
            bool rejected_pending  = false;
            bool rejected_fallback = false;
            ninfer::PinnedHostBuffer before(program.state().continuation_bytes());
            ninfer::PinnedHostBuffer after(program.state().continuation_bytes());
            for (;;) {
                std::vector<fn::Lane> lanes;
                std::vector<std::int32_t> anchors;
                std::vector<std::uint32_t> limits;
                std::vector<std::size_t> row_of;
                for (std::size_t row = 0; row < owners.size(); ++row) {
                    if (output[row].size() >= 64) { continue; }
                    lanes.push_back(owners[row]);
                    anchors.push_back(output[row].back());
                    limits.push_back(static_cast<std::uint32_t>(64 - output[row].size()));
                    row_of.push_back(row);
                }
                if (lanes.empty()) { break; }
                const bool fallback = program.state().has_dflash() && test.k != 0 && partial_once &&
                                      published_once && rejected_pending && !rejected_fallback;
                if (fallback) { std::fill(limits.begin(), limits.end(), 1U); }
                const bool reject =
                    program.state().has_dflash() &&
                    (!rejected_once || (published_once && !rejected_pending) || fallback);
                if (reject) {
                    program.state().pack_continuation(lanes[0], before.data(), device.stream);
                    CUDA_CHECK(cudaStreamSynchronize(device.stream));
                }
                const auto round =
                    program.decode_round(lanes, anchors, limits, fn::RoundLogprobs::Report);
                std::vector<std::int32_t> committed(round.counts.begin(), round.counts.end());
                if (reject) {
                    committed[0] = 0;
                    if (fallback) {
                        rejected_fallback = true;
                    } else if (rejected_once) {
                        rejected_pending = true;
                    }
                    rejected_once = true;
                } else if (!partial_once && round.width > 1 && committed[0] > 1) {
                    committed[0] = 1;
                    partial_once = true;
                }
                for (std::size_t row = 0; row < lanes.size(); ++row) {
                    const auto start = row * static_cast<std::size_t>(round.width);
                    for (std::int32_t column = 0; column < committed[row]; ++column) {
                        const auto slot = start + static_cast<std::size_t>(column);
                        require(std::isfinite(round.logprobs[slot]),
                                "nonfinite licensed token logprob");
                        output[row_of[row]].push_back(round.tokens[slot]);
                    }
                }
                program.resolve_round(committed);
                published_once = published_once || committed[0] != 0;
                if (reject) {
                    program.state().pack_continuation(lanes[0], after.data(), device.stream);
                    CUDA_CHECK(cudaStreamSynchronize(device.stream));
                    require(std::memcmp(before.data(), after.data(), before.size()) == 0,
                            "rejected graph verification changed complete canonical state");
                }
            }
            if (test.k != 0) {
                require(partial_once && rejected_pending && rejected_fallback,
                        "DFlash case did not exercise partial, pending and root-only transactions");
            }
            program.flush_context(owners);
            for (std::size_t row = 0; row < owners.size(); ++row) {
                if (test.temperature > 0.0F) {
                    std::vector<std::int32_t> expected(fn::kVocabRows, 0), actual(fn::kVocabRows);
                    for (const auto token : output[row]) {
                        ++expected[static_cast<std::size_t>(token)];
                    }
                    CUDA_CHECK(cudaMemcpy(actual.data(), program.state().token_counts(owners[row]),
                                          actual.size() * sizeof(actual[0]),
                                          cudaMemcpyDeviceToHost));
                    require(actual == expected, "publication/rejection corrupted sampling counts");
                }
                const auto stats = program.speculative_stats(owners[row]);
                std::cout << "DFlash Program C=" << test.lanes << " K=" << test.k
                          << " adaptive_forced=" << adaptive << " T=" << test.temperature
                          << " p_less=" << test.p_less << " row=" << row
                          << " drafted=" << stats.drafted_tokens
                          << " accepted=" << stats.accepted_tokens
                          << " outputs=" << output[row].size() << '\n';
                program.close_lane(owners[row]);
            }
            return output;
        };
        const auto fixed  = execute(false);
        const auto forced = execute(true);
        require(fixed == forced, "picker-forced K differs from fixed K output");
        if (test.lanes == 1 && test.k == 3 && test.temperature == 0.0F) {
            static_cast<void>(execute(true, false));
        }
    }
}
