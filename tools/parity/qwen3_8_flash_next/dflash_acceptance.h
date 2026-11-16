// Included inside parity.cpp's namespace. The study's fixed teacher-forced panel, with native
// realized accept walks rather than its accept-uniform-integrated probabilities.
int run_dflash_acceptance(const Options& options, const TokenFile& input) {
    if (input.sequences < 1 ||
        input.tokens.size() / static_cast<std::size_t>(input.sequences) != 2048) {
        throw std::invalid_argument("dflash-acceptance requires [S,2048] token sequences");
    }
    std::filesystem::create_directories(options.out);
    std::ofstream report(options.out / "dflash_acceptance.json");
    if (!report) { throw std::runtime_error("cannot open dflash_acceptance.json"); }
    ninfer::DeviceContext device(0);
    const ninfer::artifact::Reader reader(options.artifact);
    const fn::LoadedModel model(reader, device);
    const auto counts = options.calib_counts.empty() ? std::vector<std::uint64_t>{}
                                                     : read_counts(options.calib_counts);
    auto config       = program_options_for(options, 4096, fn::LogitColumns::Last, counts, 1);
    config.speculative.backend      = ninfer::SpeculativeBackend::DFlash;
    config.speculative.draft_tokens = 3;
    fn::Program program(model, config, device);
    std::array<std::uint64_t, 3> reached{}, accepted{};
    std::uint64_t rounds = 0, licensed = 0;
    report << std::setprecision(8)
           << "{\"K\":3,\"temperature\":1,\"top_k\":20,\"top_p\":0.95,"
              "\"p_less\":false,\"realized_accept_walks\":true,\"sequences\":[\n";
    for (std::int64_t sequence = 0; sequence < input.sequences; ++sequence) {
        const auto tokens =
            std::span(input.tokens).subspan(static_cast<std::size_t>(sequence) * 2048, 2048);
        fn::LaneSampling sampling;
        sampling.temperature = 1.0F;
        sampling.p_less      = false;
        sampling.top_k       = 20;
        sampling.top_p       = 0.95F;
        sampling.seed        = 73U + static_cast<std::uint64_t>(sequence);
        std::array<std::uint64_t, 3> sequence_reached{}, sequence_accepted{};
        for (std::int32_t frontier = 64; frontier < 2048; frontier += 64) {
            const auto lane = program.open_lane(frontier + 4, sampling);
            if (!lane) { throw std::runtime_error("acceptance panel could not open its lane"); }
            const auto anchor =
                prefill_prompt(program, *lane, tokens.first(static_cast<std::size_t>(frontier)));
            const std::array lanes{*lane};
            const std::array anchors{anchor};
            const std::array<std::uint32_t, 1> limit{4};
            const auto packet =
                program.decode_round(lanes, anchors, limit, fn::RoundLogprobs::Skip);
            const auto count = packet.accepted[0];
            if (packet.width != 4 || count < 0 || count > 3 || packet.counts[0] != count + 1) {
                throw std::runtime_error("acceptance panel returned an invalid full K3 chain");
            }
            ++rounds;
            licensed += static_cast<std::uint32_t>(count + 1);
            for (std::size_t hop = 0; hop < reached.size(); ++hop) {
                if (static_cast<std::size_t>(count) >= hop) {
                    ++reached[hop];
                    ++sequence_reached[hop];
                }
                if (static_cast<std::size_t>(count) > hop) {
                    ++accepted[hop];
                    ++sequence_accepted[hop];
                }
            }
            const std::array<std::int32_t, 1> reject{0};
            program.resolve_round(reject);
            program.close_lane(*lane);
        }
        report << "{\"sequence\":" << sequence << ",\"reached\":[" << sequence_reached[0] << ','
               << sequence_reached[1] << ',' << sequence_reached[2] << "],\"accepted\":["
               << sequence_accepted[0] << ',' << sequence_accepted[1] << ',' << sequence_accepted[2]
               << "]}" << (sequence + 1 < input.sequences ? ",\n" : "\n");
        report.flush();
        if (!report) { throw std::runtime_error("cannot write acceptance sequence results"); }
    }
    report << "],\"rounds\":" << rounds << ",\"mean_L\":" << static_cast<double>(licensed) / rounds
           << ",\"reached\":[" << reached[0] << ',' << reached[1] << ',' << reached[2]
           << "],\"accepted\":[" << accepted[0] << ',' << accepted[1] << ',' << accepted[2]
           << "]}\n";
    if (!report) { throw std::runtime_error("cannot finish dflash_acceptance.json"); }
    std::cerr << "Native sampled K3 rounds=" << rounds
              << " mean L=" << static_cast<double>(licensed) / rounds << " conditional hops=";
    for (std::size_t hop = 0; hop < reached.size(); ++hop) {
        std::cerr << ' ' << accepted[hop] << '/' << reached[hop];
    }
    std::cerr << '\n';
    return 0;
}
