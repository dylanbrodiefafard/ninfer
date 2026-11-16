// Included inside the real-artifact test's namespace after its Engine helpers.
void dflash_engine_cases(const Inputs& inputs) {
    struct Sink final : ninfer::OutputSink {
        std::string text;
        std::vector<ninfer::TokenLogprob> records;

        void publish(ninfer::OutputDelta delta) override {
            require(delta.channel == ninfer::OutputChannel::Content,
                    "raw DFlash output used a non-content channel");
            text += delta.text;
            records.insert(records.end(), delta.logprobs.begin(), delta.logprobs.end());
        }
    };

    for (const bool adaptive : {false, true}) {
        auto options                       = engine_options(inputs, 2, 4096, 8192);
        options.speculative.backend        = ninfer::SpeculativeBackend::DFlash;
        options.speculative.draft_tokens   = 3;
        options.speculative.adaptive_draft = adaptive;
        ninfer::Engine engine(options);
        std::vector<ninfer::GenerationHandle> handles;
        for (std::uint32_t row = 0; row < 2; ++row) {
            auto request                                 = greedy(49 + row);
            request.execution.sampling.temperature       = 0.7F;
            request.execution.sampling.p_less            = row == 0;
            request.execution.sampling.seed              = 73 + row;
            request.execution.sampling.presence_penalty  = 0.4F;
            request.execution.sampling.frequency_penalty = 0.2F;
            request.output.raw                           = true;
            request.output.top_logprobs                  = 5;
            handles.push_back(engine.submit(
                engine.prepare_tokens(prompt_tokens(512 + 65 * row, 901 + row), false), request,
                ninfer::OutputDelivery::Streaming));
        }
        for (std::size_t row = 0; row < handles.size(); ++row) {
            Sink sink;
            const auto result = handles[row].wait(&sink);
            require(result.finish_reason == ninfer::FinishReason::OutputLimit &&
                        result.generated_token_ids.size() == 49 + row,
                    "DFlash Engine did not respect the row's output budget");
            require(result.speculative.enabled &&
                        result.speculative.rounds + result.speculative.fallback_steps > 0,
                    "DFlash Engine did not report its committed speculative rounds");
            require(sink.text == result.content &&
                        sink.records.size() == result.content_logprobs.size() &&
                        sink.records.size() == result.generated_token_ids.size(),
                    "DFlash streamed text/logprobs differ from terminal output");
            for (std::size_t i = 0; i < sink.records.size(); ++i) {
                const auto& record   = sink.records[i];
                const auto& terminal = result.content_logprobs[i];
                require(record.token == result.generated_token_ids[i] &&
                            record.token == terminal.token && record.logprob == terminal.logprob &&
                            std::isfinite(record.logprob) && record.top_count == 5 &&
                            terminal.top_count == 5,
                        "DFlash Engine lost token-aligned logprobs");
                for (std::uint32_t rank = 0; rank < record.top_count; ++rank) {
                    require(record.top[rank].token == terminal.top[rank].token &&
                                record.top[rank].logprob == terminal.top[rank].logprob &&
                                std::isfinite(record.top[rank].logprob),
                            "DFlash Engine changed a streamed ranked alternative");
                }
            }
            std::cout << "DFlash Engine adaptive=" << adaptive << " row=" << row
                      << " outputs=" << result.generated_token_ids.size()
                      << " drafted=" << result.speculative.drafted_tokens
                      << " accepted=" << result.speculative.accepted_tokens << '\n';
        }
        if (adaptive) { continue; }
        for (const std::uint32_t limit : {0U, 1U}) {
            auto unsupported                                 = greedy(limit);
            unsupported.execution.allow_prefix_reuse         = true;
            unsupported.execution.capture_context_checkpoint = true;
            bool rejected                                    = false;
            try {
                static_cast<void>(
                    engine.generate(engine.prepare_tokens(prompt_tokens(64, 7000)), unsupported));
            } catch (const std::invalid_argument&) { rejected = true; }
            require(rejected, "native DFlash enabled unsupported context-checkpoint capture");
        }
        const Tokens prompt  = prompt_tokens(512, 7001);
        auto request         = greedy(24);
        request.output.raw   = true;
        const auto reference = engine.generate(engine.prepare_tokens(prompt, false), request);
        const auto stop      = reference.generated_token_ids.at(4);
        auto stopped_request = request;
        stopped_request.stop.token_ids          = {stop};
        stopped_request.stop.publish_stop_token = true;
        const auto stopped = engine.generate(engine.prepare_tokens(prompt, false), stopped_request);
        const auto stop_at = std::find(reference.generated_token_ids.begin(),
                                       reference.generated_token_ids.end(), stop);
        const Tokens expected(reference.generated_token_ids.begin(), stop_at + 1);
        require(stopped.finish_reason == ninfer::FinishReason::StopToken,
                "DFlash Engine ignored a caller stop token");
        require_equal(stopped.generated_token_ids, expected, "DFlash stop-prefix publication");
        const auto before = engine.runtime_stats().committed_decode_tokens;
        const ninfer::CancellationView cancel(
            [&] { return engine.runtime_stats().committed_decode_tokens >= before + 8; });
        const auto cancelled = engine.generate(
            engine.prepare_tokens(prompt_tokens(512, 7002), false), greedy(1024), nullptr, cancel);
        require(cancelled.finish_reason == ninfer::FinishReason::Cancelled &&
                    cancelled.generated_token_ids.size() < 1024,
                "DFlash Engine failed to cancel a live decode");
        require_equal(
            engine.generate(engine.prepare_tokens(prompt, false), request).generated_token_ids,
            reference.generated_token_ids, "DFlash request after cancellation");
    }
}
