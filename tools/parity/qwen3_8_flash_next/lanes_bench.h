// Included inside parity.cpp's namespace. Product throughput uses only the public Engine.
int run_lanes_bench(const Options& options, std::span<const std::int32_t> stream_tokens) {
    if (!options.calib_counts.empty() || options.expert_cache_mib != -1 || options.window != 1.0 ||
        options.window_min_tokens != 512 || options.profile_decode ||
        options.expert_staging != fn::ExpertStaging::Enabled ||
        (options.super_chunk != 0 && options.super_chunk != 16384)) {
        throw std::invalid_argument(
            "lanes-bench uses production Engine planning, not private cache, "
            "streaming-window, staging, super-chunk or profiler controls");
    }
    const auto prompt_tokens = options.prompt_lengths.front();
    if (prompt_tokens < 1 || options.prompt_offset < 0) {
        throw std::invalid_argument(
            "lanes-bench needs a positive prompt length and nonnegative offset");
    }
    const std::int64_t reserve = std::int64_t{prompt_tokens} + options.decode_tokens + 2;
    std::filesystem::create_directories(options.out);
    std::ofstream report(options.out / "lanes_bench.json");
    if (!report) { throw std::runtime_error("cannot open lanes_bench.json"); }
    report << std::setprecision(8) << "[\n";
    for (std::size_t index = 0; index < options.lane_counts.size(); ++index) {
        const auto lanes = options.lane_counts[index];
        if (lanes < 1 || lanes > fn::kMaxLanes ||
            (static_cast<std::uint64_t>(lanes) +
             static_cast<std::uint64_t>(options.prompt_offset)) *
                    static_cast<std::uint64_t>(prompt_tokens) >
                stream_tokens.size()) {
            throw std::invalid_argument(
                "lanes-bench concurrency is outside [1,12] or tokens are insufficient");
        }
        const auto context = options.kv_capacity
                                 ? std::min<std::int64_t>(*options.kv_capacity / lanes,
                                                          ninfer::ops::kQsaMaximumCapacity)
                                 : std::max<std::int64_t>(reserve, options.context);
        if (context < reserve || context > ninfer::ops::kQsaMaximumCapacity) {
            throw std::invalid_argument("lanes-bench context cannot hold the requested lifetime");
        }
        ninfer::EngineOptions engine_options;
        engine_options.artifact_path     = options.artifact;
        engine_options.ngram_table_path  = options.ngram_table;
        engine_options.expert_helper_pci = options.expert_helper_pci;
        engine_options.max_concurrency   = static_cast<std::uint32_t>(lanes);
        engine_options.max_context       = static_cast<std::uint32_t>(context);
        engine_options.prefill_chunk     = static_cast<std::uint32_t>(options.chunk);
        engine_options.qsa_select_budget =
            ninfer::parse_qsa_select_budget_flag(std::to_string(options.budget));
        engine_options.kv_cache    = options.kv_format == ninfer::ops::QsaKvFormat::BF16
                                         ? ninfer::KvCacheStorage::BFloat16
                                         : ninfer::KvCacheStorage::Nvfp4;
        engine_options.speculative = options.speculative;
        engine_options.kv_capacity = ninfer::KvCapacityPolicy::automatic();
        if (options.kv_capacity) {
            engine_options.kv_capacity = ninfer::KvCapacityPolicy::explicit_capacity(
                static_cast<std::uint32_t>(*options.kv_capacity));
        }
        ninfer::Engine engine(engine_options);
        const auto total_start = std::chrono::steady_clock::now();
        std::vector<ninfer::GenerationHandle> handles;
        for (std::int32_t row = 0; row < lanes; ++row) {
            const auto offset =
                (static_cast<std::size_t>(row) + static_cast<std::size_t>(options.prompt_offset)) *
                static_cast<std::size_t>(prompt_tokens);
            const auto source =
                stream_tokens.subspan(offset, static_cast<std::size_t>(prompt_tokens));
            ninfer::RequestOptions request;
            request.execution.requested_output_tokens =
                static_cast<std::uint32_t>(options.decode_tokens);
            request.execution.sampling.temperature = 0.0F;
            request.execution.allow_prefix_reuse   = false;
            request.stop.include_model_defaults    = false;
            request.output.raw                     = true;
            handles.push_back(engine.submit(
                engine.prepare_tokens({source.begin(), source.end()}, false), request));
        }
        auto completed = std::async(std::launch::async, [handles = std::move(handles)]() mutable {
            std::vector<ninfer::GenerationResult> results;
            for (auto& handle : handles) { results.push_back(handle.wait()); }
            return results;
        });
        const auto expected_prefill =
            static_cast<std::uint64_t>(lanes) * static_cast<std::uint64_t>(prompt_tokens);
        while (engine.runtime_stats().computed_prefill_tokens < expected_prefill) {
            if (completed.wait_for(std::chrono::seconds(0)) == std::future_status::ready) {
                static_cast<void>(completed.get());
                throw std::runtime_error(
                    "lanes-bench completed before its post-prefill measurement interval");
            }
            std::this_thread::sleep_for(std::chrono::microseconds(100));
        }
        // A direct interval of monotonic Engine counters, not a rate inferred from kernel timing.
        // It includes natural request drain as rows finish, but excludes all initial prefill work.
        const auto begin            = engine.runtime_stats();
        const auto decode_start     = std::chrono::steady_clock::now();
        const auto results          = completed.get();
        const double decode_seconds = seconds_since(decode_start);
        const auto end              = engine.runtime_stats();
        const double total_seconds  = seconds_since(total_start);
        const auto measured_tokens  = end.committed_decode_tokens - begin.committed_decode_tokens;
        if (measured_tokens == 0) {
            throw std::runtime_error("lanes-bench has no post-prefill decode work");
        }
        std::uint64_t total_tokens = 0, rounds = 0, fallback = 0, drafted = 0, accepted = 0;
        std::array<std::uint64_t, fn::kDflashMaxK> positions{};
        std::array<std::uint64_t, fn::kDflashWidth> arms{};
        std::ofstream dump(options.out / ("tokens_c" + std::to_string(lanes) + ".i32"),
                           std::ios::binary);
        if (!dump) { throw std::runtime_error("cannot open generated token dump"); }
        for (const auto& result : results) {
            if (result.finish_reason != ninfer::FinishReason::OutputLimit ||
                result.generated_token_ids.size() !=
                    static_cast<std::size_t>(options.decode_tokens)) {
                throw std::runtime_error("lanes-bench request failed to reach its output limit");
            }
            total_tokens += result.generated_token_ids.size();
            dump.write(reinterpret_cast<const char*>(result.generated_token_ids.data()),
                       static_cast<std::streamsize>(result.generated_token_ids.size() *
                                                    sizeof(std::int32_t)));
            const auto& stats = result.speculative;
            rounds += stats.rounds;
            fallback += stats.fallback_steps;
            drafted += stats.drafted_tokens;
            accepted += stats.accepted_tokens;
            for (std::size_t i = 0; i < stats.accepted_per_position.size(); ++i) {
                positions.at(i) += stats.accepted_per_position[i];
            }
            for (std::size_t i = 0; i < stats.rounds_per_draft.size(); ++i) {
                arms.at(i) += stats.rounds_per_draft[i];
            }
        }
        if (!dump) { throw std::runtime_error("cannot write generated token dump"); }
        const double rate = static_cast<double>(measured_tokens) / decode_seconds;
        report << "{\"expert_helper_pci\":";
        if (options.expert_helper_pci) {
            // The parser limits this value to canonical hexadecimal PCI notation.
            report << '"' << *options.expert_helper_pci << '"';
        } else {
            report << "null";
        }
        report << ",\"lanes\":" << lanes << ",\"lane_context\":" << context
               << ",\"kv_capacity_tokens\":" << engine.memory_summary().kv_capacity
               << ",\"decode_tokens\":" << measured_tokens
               << ",\"decode_seconds\":" << decode_seconds << ",\"decode_tok_s\":" << rate
               << ",\"total_seconds\":" << total_seconds
               << ",\"end_to_end_tok_s\":" << static_cast<double>(total_tokens) / total_seconds
               << ",\"rounds\":" << rounds << ",\"fallback_steps\":" << fallback
               << ",\"drafted\":" << drafted << ",\"accepted\":" << accepted;
        const auto histogram = [&](const char* name, const auto& values) {
            report << ",\"" << name << "\":[";
            for (std::size_t i = 0; i < values.size(); ++i) {
                report << (i == 0 ? "" : ",") << values[i];
            }
            report << ']';
        };
        histogram("accepted_per_position", positions);
        histogram("rounds_per_draft", arms);
        report << '}' << (index + 1 < options.lane_counts.size() ? ",\n" : "\n");
        report.flush();
        if (!report) { throw std::runtime_error("cannot write lanes_bench.json"); }
        std::cerr << "Engine C=" << lanes << " post-prefill decode/drain=" << rate
                  << " tok/s, measured=" << measured_tokens << " tokens\n";
    }
    report << "]\n";
    if (!report) { throw std::runtime_error("cannot finish lanes_bench.json"); }
    return 0;
}
