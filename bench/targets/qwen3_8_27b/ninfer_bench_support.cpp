#include "ninfer_bench_support.h"
#include "ops/r9700/kv/r9700_attention_profile.h"
#include "ops/r9700/kv/r9700_xattention_profile.h"
#include "ops/r9700/linear/r9700_q4_activation_profile.h"
#include "ops/r9700/linear/r9700_w8_activation_profile.h"
#include "targets/qwen3/impl/runtime/r9700_cache_profile.h"

#include <algorithm>
#include <array>
#include <charconv>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <system_error>

namespace ninfer::bench {
namespace {

int parse_int(std::string_view text, const char* label) {
    if (text.empty()) { throw std::invalid_argument(std::string(label) + " is empty"); }
    int value         = 0;
    const auto result = std::from_chars(text.data(), text.data() + text.size(), value);
    if (result.ec != std::errc{} || result.ptr != text.data() + text.size()) {
        throw std::invalid_argument(std::string("invalid ") + label + ": " + std::string(text));
    }
    return value;
}

int parse_nonnegative(std::string_view text, const char* label) {
    const int value = parse_int(text, label);
    if (value < 0) { throw std::invalid_argument(std::string(label) + " must be nonnegative"); }
    return value;
}

int parse_positive(std::string_view text, const char* label) {
    const int value = parse_int(text, label);
    if (value <= 0) { throw std::invalid_argument(std::string(label) + " must be positive"); }
    return value;
}

std::uint32_t parse_u32(std::string_view text, const char* label, bool allow_zero = false) {
    if (text.empty()) { throw std::invalid_argument(std::string(label) + " is empty"); }
    std::uint64_t value = 0;
    const auto result   = std::from_chars(text.data(), text.data() + text.size(), value);
    if (result.ec != std::errc{} || result.ptr != text.data() + text.size() ||
        value > std::numeric_limits<std::uint32_t>::max() || (!allow_zero && value == 0)) {
        throw std::invalid_argument(std::string("invalid ") + label + ": " + std::string(text));
    }
    return static_cast<std::uint32_t>(value);
}

std::vector<int> parse_int_list(std::string_view value, const char* label) {
    std::vector<int> out;
    std::size_t start = 0;
    while (true) {
        const std::size_t comma      = value.find(',', start);
        const std::string_view piece = value.substr(
            start, comma == std::string_view::npos ? std::string_view::npos : comma - start);
        out.push_back(parse_positive(piece, label));
        if (comma == std::string_view::npos) { break; }
        start = comma + 1;
    }
    return out;
}

std::vector<std::pair<int, int>> parse_pair_list(std::string_view value, const char* label) {
    std::vector<std::pair<int, int>> out;
    std::size_t start = 0;
    while (true) {
        const std::size_t semi       = value.find(';', start);
        const std::string_view piece = value.substr(
            start, semi == std::string_view::npos ? std::string_view::npos : semi - start);
        const std::size_t comma = piece.find(',');
        if (comma == std::string_view::npos ||
            piece.find(',', comma + 1) != std::string_view::npos) {
            throw std::invalid_argument(std::string(label) + " expects P,G pairs");
        }
        out.emplace_back(parse_positive(piece.substr(0, comma), "prompt-gen P"),
                         parse_positive(piece.substr(comma + 1), "prompt-gen G"));
        if (semi == std::string_view::npos) { break; }
        start = semi + 1;
    }
    return out;
}

std::string existing_read_path(const std::string& path) {
    if (std::filesystem::exists(path)) { return path; }
#ifdef NINFER_SOURCE_DIR
    const auto source_relative =
        (std::filesystem::path(NINFER_SOURCE_DIR) / path).lexically_normal();
    if (std::filesystem::exists(source_relative)) { return source_relative.string(); }
#endif
    return path;
}

const char* kind_string(TestKind kind) {
    switch (kind) {
    case TestKind::Prefill:
        return "pp";
    case TestKind::Decode:
        return "tg";
    case TestKind::PrefillDecode:
        return "pp+tg";
    case TestKind::WholeInference:
        return "whole";
    }
    return "unknown";
}

std::uint32_t checked_context(std::uint64_t value, const char* label) {
    if (value == 0 || value > std::numeric_limits<std::uint32_t>::max()) {
        throw std::overflow_error(std::string(label) + " exceeds the supported context range");
    }
    return static_cast<std::uint32_t>(value);
}

std::string number(double value) {
    std::ostringstream out;
    out << std::setprecision(10) << value;
    return out.str();
}

std::string format_bytes(std::uint64_t bytes) {
    if (bytes == 0) { return "-"; }
    constexpr double gib = 1024.0 * 1024.0 * 1024.0;
    constexpr double mib = 1024.0 * 1024.0;
    std::ostringstream out;
    if (static_cast<double>(bytes) >= gib) {
        out << std::fixed << std::setprecision(2) << static_cast<double>(bytes) / gib << " GiB";
    } else {
        out << std::fixed << std::setprecision(1) << static_cast<double>(bytes) / mib << " MiB";
    }
    return out.str();
}

std::string rate_cell(const std::vector<double>& values) {
    if (values.empty()) { return "-"; }
    const Stats stats = compute_stats(values);
    std::ostringstream out;
    out << std::fixed << std::setprecision(2) << stats.mean;
    if (stats.count > 1) {
        out << " \u00b1 " << std::fixed << std::setprecision(2) << stats.stddev;
    }
    return out.str();
}

void append_stat(std::ostringstream& out, std::string_view name, const std::vector<double>& values,
                 std::string_view indent) {
    out << indent << '"' << name << "_mean\": ";
    if (values.empty()) {
        out << "null,\n" << indent << '"' << name << "_stddev\": null";
    } else {
        const Stats stats = compute_stats(values);
        out << number(stats.mean) << ",\n"
            << indent << '"' << name << "_stddev\": " << number(stats.stddev);
    }
}

SpeculativeStats aggregate_speculative(const TestResult& result) {
    SpeculativeStats out;
    for (const RepTiming& rep : result.reps) {
        const SpeculativeStats& in = rep.speculative;
        out.enabled                = out.enabled || in.enabled;
        out.draft_window           = std::max(out.draft_window, in.draft_window);
        out.rounds += in.rounds;
        out.drafted_tokens += in.drafted_tokens;
        out.accepted_tokens += in.accepted_tokens;
        out.fallback_steps += in.fallback_steps;
        if (out.accepted_per_position.size() < in.accepted_per_position.size()) {
            out.accepted_per_position.resize(in.accepted_per_position.size());
        }
        for (std::size_t i = 0; i < in.accepted_per_position.size(); ++i) {
            out.accepted_per_position[i] += in.accepted_per_position[i];
        }
        if (out.rounds_per_draft.size() < in.rounds_per_draft.size()) {
            out.rounds_per_draft.resize(in.rounds_per_draft.size());
        }
        for (std::size_t i = 0; i < in.rounds_per_draft.size(); ++i) {
            out.rounds_per_draft[i] += in.rounds_per_draft[i];
        }
    }
    return out;
}

std::uint64_t decode_engine_tokens(const TestResult& result, const RepTiming& rep) {
    if (!result.test.has_decode()) { return 0; }
    if (!rep.speculative.enabled) { return static_cast<std::uint64_t>(result.test.n_gen); }
    return rep.speculative.rounds + rep.speculative.accepted_tokens +
           rep.speculative.fallback_steps;
}

void append_arena_json(std::ostringstream& out, std::string_view name,
                       const ArenaMemorySummary& arena, std::string_view indent,
                       bool trailing_comma) {
    out << indent << '"' << name << "\": {\"capacity_bytes\": " << arena.capacity_bytes
        << ", \"used_bytes\": " << arena.used_bytes
        << ", \"peak_used_bytes\": " << arena.peak_used_bytes << '}';
    if (trailing_comma) { out << ','; }
    out << '\n';
}

void append_speculative_json(std::ostringstream& out, const SpeculativeStats& stats,
                             std::string_view indent) {
    out << indent << "\"speculative\": {\n"
        << indent << "  \"enabled\": " << (stats.enabled ? "true" : "false") << ",\n"
        << indent << "  \"draft_window\": " << stats.draft_window << ",\n"
        << indent << "  \"rounds\": " << stats.rounds << ",\n"
        << indent << "  \"drafted_tokens\": " << stats.drafted_tokens << ",\n"
        << indent << "  \"accepted_tokens\": " << stats.accepted_tokens << ",\n"
        << indent << "  \"fallback_steps\": " << stats.fallback_steps << ",\n"
        << indent << "  \"acceptance_rate\": ";
    if (stats.drafted_tokens == 0) {
        out << "null";
    } else {
        out << number(static_cast<double>(stats.accepted_tokens) /
                      static_cast<double>(stats.drafted_tokens));
    }
    out << ",\n" << indent << "  \"acceptance_length\": ";
    if (stats.rounds == 0) {
        out << "null";
    } else {
        out << number(1.0 + static_cast<double>(stats.accepted_tokens) /
                                static_cast<double>(stats.rounds));
    }
    out << ",\n" << indent << "  \"accepted_per_position\": [";
    for (std::size_t i = 0; i < stats.accepted_per_position.size(); ++i) {
        if (i != 0) { out << ", "; }
        out << stats.accepted_per_position[i];
    }
    out << "],\n" << indent << "  \"rounds_per_draft\": [";
    for (std::size_t i = 0; i < stats.rounds_per_draft.size(); ++i) {
        if (i != 0) { out << ", "; }
        out << stats.rounds_per_draft[i];
    }
    out << "]\n" << indent << '}';
}

void append_timings_json(std::ostringstream& out, const GenerationTimings& timings,
                         std::string_view indent) {
    out << indent << "\"timings\": {" << "\"prepare_seconds\": " << number(timings.prepare_seconds)
        << ", " << "\"vision_seconds\": " << number(timings.vision_seconds) << ", "
        << "\"prefill_seconds\": " << number(timings.prefill_seconds) << ", "
        << "\"decode_seconds\": " << number(timings.decode_seconds) << ", "
        << "\"total_seconds\": " << number(timings.total_seconds) << '}';
}

} // namespace

std::uint32_t BenchTest::requested_output_tokens() const {
    return checked_context(static_cast<std::uint64_t>(has_decode() ? n_gen : 0) + 1ULL,
                           "requested benchmark output");
}

std::uint32_t BenchTest::required_context(const SpeculativeOptions& spec) const {
    const std::uint64_t prompt =
        static_cast<std::uint64_t>(kind == TestKind::Decode ? kDecodeSeedTokens : n_prompt);
    const std::uint64_t decode = static_cast<std::uint64_t>(has_decode() ? n_gen : 0);
    std::uint64_t margin       = 0;
    if (spec.draft_tokens != 0 && spec.backend == SpeculativeBackend::Mtp) {
        margin = 2ULL * spec.draft_tokens;
    } else if (spec.draft_tokens != 0 && spec.backend == SpeculativeBackend::DFlash) {
        // 27B DFlash2 is chain W=k+1. An explicit dflash_verify_width wins. Two
        // verify-widths of headroom cover the transient uncommitted round state.
        const std::uint32_t width =
            spec.dflash_verify_width != 0 ? spec.dflash_verify_width : spec.draft_tokens + 1U;
        margin = 2ULL * width;
    }
    return checked_context(prompt + decode + margin, "benchmark context requirement");
}

std::string usage_text(std::string_view program) {
    if (program.empty()) { program = "ninfer_bench"; }
    std::ostringstream out;
    out << "Usage: " << program << " --weights <artifact.ninfer> [options]\n\n"
        << "Product-route throughput benchmark over ninfer::Engine. pp throughput is aggregate\n"
        << "prompt tokens over request-wave wall time; active pp throughput excludes queueing.\n"
        << "tg measures G generated tokens after an untimed one-token seed prefill.\n\n"
        << "Options:\n"
        << "  --weights <path>            required .ninfer artifact\n"
        << "  --corpus <path>             token-id corpus (default: " << kDefaultCorpusPath << ")\n"
        << "  -p, --n-prompt <list>       pp lengths, for example 512,2048\n"
        << "  -n, --n-gen <list>          tg lengths, for example 128\n"
        << "  -pg, --prompt-gen <P,G;..>  combined pp+tg tests\n"
        << "  --whole-pg <P,G;..>         end-to-end fresh-prompt pp+tg makespan tests\n"
        << "  -r, --repetitions <n>       measured repetitions (default: " << kDefaultRepetitions
        << ")\n"
        << "  --warmup <n>                discarded repetitions (default: " << kDefaultWarmup
        << ")\n"
        << "  --max-ctx <tokens>          override auto-sized context capacity\n"
        << "  --kv-capacity <workload|auto> exact workload capacity or maximum capacity with\n"
        << "                              the default 64 MiB headroom (default: workload)\n"
        << "  --prefill-chunk <tokens>    multiple of " << kPrefillChunkAlignment
        << " (default: " << kDefaultPrefillChunk << ")\n"
        << "  --mixed-forward auto|<N>    prefill-step forward width (multiple of 256) while\n"
        << "                              other requests decode; auto = the server default\n"
        << "                              (default: 0, prefill first)\n"
        << "  --mixed-forward-rounds <N>  decode rounds per prefill slice (default: 1)\n"
        << "  --contention <P,R>          C-1 lanes decode while one lane runs R fresh P-token\n"
        << "                              prefills; reports decode and prefill tok/s (C >= 2)\n"
        << "  --contention-context <L>    prompt tokens of each contention decode lane\n"
        << "                              (default: 512)\n"
        << "  --contention-lanes <N>      contention decode lanes, 1..C-1 (default: C-1)\n"
        << "  --concurrency <1..8>        concurrent Engine lanes (default: 1);\n"
        << "                              pp+tg at C>1 reports batched decode after a sequential\n"
        << "                              prefix-reuse seed so prefill/decode do not interleave\n"
        << "                              benchmark requests use a fixed non-expiring long-run\n"
        << "                              pending deadline; the product default is unchanged\n"
        << "  --spec <mtp|dflash>       speculative backend (default: mtp); dflash requires\n"
        << "                              the artifact to contain dflash/ objects\n"
        << "  --draft-tokens <0..7>     speculative draft window: mtp [0,5] (0 = none),\n"
        << "                              dflash [1,7] (default: 0)\n"
        << "  --adaptive-draft            pick live K in {3,4,5} by locking E[Y]/T(k,C,L); "
           "requires --spec mtp|dflash\n"
        << "  --dflash-verify-width <0|2..8> DFlash chain width; 0 = K+1; explicit width\n"
        << "                              must equal K+1 (dflash only)\n"
        << "  --lm-head-draft             use the optimized proposal head; requires "
           "--draft-tokens\n"
        << "                              greater than zero\n"
        << "  --device <id>               HIP device ordinal (default: 0)\n"
        << "  --no-device-graph           use eager decode\n"
        << "  --profile-measured          bracket one measured repetition as a profiler region\n"
        << "  --retain-token-ids          retain greedy token IDs by repetition and lane in JSON\n"
        << "  --isolate-prompt-decode     seed and discard one token before every -pg decode lane\n"
        << "  -o, --output <table|json|csv>  output format (default: table)\n"
        << "  --output-file <path>        write report to a file\n"
        << "  -h, --help                  show this help\n\n"
        << "With no -p/-n/-pg, the matrix is pp" << kDefaultNPrompt << " and tg" << kDefaultNGen
        << ".\n";
    return out.str();
}

BenchOptions parse_args(int argc, char** argv) {
    BenchOptions options;
    bool saw_artifact = false;
    for (int i = 1; i < argc; ++i) {
        const std::string_view arg(argv[i]);
        auto value = [&](const char* flag) -> std::string {
            if (i + 1 >= argc) {
                throw std::invalid_argument(std::string(flag) + " requires value");
            }
            return argv[++i];
        };
        if (arg == "-h" || arg == "--help") {
            options.help_requested = true;
            return options;
        }
        if (arg == "--weights") {
            options.artifact_path = value("--weights");
            saw_artifact          = true;
        } else if (arg == "--corpus") {
            options.corpus_path = value("--corpus");
        } else if (arg == "-p" || arg == "--n-prompt") {
            auto parsed = parse_int_list(value("--n-prompt"), "n-prompt");
            options.n_prompt.insert(options.n_prompt.end(), parsed.begin(), parsed.end());
        } else if (arg == "-n" || arg == "--n-gen") {
            auto parsed = parse_int_list(value("--n-gen"), "n-gen");
            options.n_gen.insert(options.n_gen.end(), parsed.begin(), parsed.end());
        } else if (arg == "-pg" || arg == "--prompt-gen") {
            auto parsed = parse_pair_list(value("--prompt-gen"), "prompt-gen");
            options.prompt_gen.insert(options.prompt_gen.end(), parsed.begin(), parsed.end());
        } else if (arg == "--whole-pg") {
            auto parsed = parse_pair_list(value("--whole-pg"), "whole-pg");
            options.whole_inference.insert(options.whole_inference.end(), parsed.begin(),
                                           parsed.end());
        } else if (arg == "-r" || arg == "--repetitions") {
            options.repetitions = parse_positive(value("--repetitions"), "repetitions");
        } else if (arg == "--warmup") {
            options.warmup = parse_nonnegative(value("--warmup"), "warmup");
        } else if (arg == "--max-ctx") {
            options.max_context = parse_u32(value("--max-ctx"), "max-ctx");
        } else if (arg == "--kv-capacity") {
            const std::string selected = value("--kv-capacity");
            if (selected == "auto") {
                options.automatic_kv_capacity = true;
            } else if (selected == "workload") {
                options.automatic_kv_capacity = false;
            } else {
                throw std::invalid_argument("--kv-capacity must be workload or auto");
            }
        } else if (arg == "--prefill-chunk") {
            options.prefill_chunk = parse_u32(value("--prefill-chunk"), "prefill-chunk");
        } else if (arg == "--mixed-forward") {
            const std::string forward = value("--mixed-forward");
            if (forward == "auto") {
                options.mixed_forward.reset();
            } else {
                options.mixed_forward = parse_u32(forward, "mixed-forward", true);
            }
        } else if (arg == "--mixed-forward-rounds") {
            options.mixed_forward_rounds =
                parse_u32(value("--mixed-forward-rounds"), "mixed-forward-rounds");
        } else if (arg == "--pair-check") {
            const auto parsed = parse_pair_list(value("--pair-check"), "pair-check");
            if (parsed.size() != 1 || parsed.front().first <= 0 || parsed.front().second <= 0) {
                throw std::invalid_argument("--pair-check takes exactly one positive P,G pair");
            }
            options.pair_check = parsed.front();
        } else if (arg == "--contention") {
            const auto parsed = parse_pair_list(value("--contention"), "contention");
            if (parsed.size() != 1 || parsed.front().first <= 0 || parsed.front().second <= 0) {
                throw std::invalid_argument("--contention takes exactly one positive P,R pair");
            }
            options.contention = parsed.front();
        } else if (arg == "--contention-lanes") {
            options.contention_lanes = static_cast<std::uint32_t>(
                parse_positive(value("--contention-lanes"), "contention-lanes"));
        } else if (arg == "--contention-context") {
            options.contention_context = static_cast<std::uint32_t>(
                parse_positive(value("--contention-context"), "contention-context"));
        } else if (arg == "--concurrency") {
            options.concurrency = parse_u32(value("--concurrency"), "concurrency");
            if (options.concurrency > kMaximumConcurrency) {
                throw std::invalid_argument("--concurrency must be in [1,8]");
            }
        } else if (arg == "--spec") {
            const std::string selected = value("--spec");
            if (selected == "mtp") {
                options.spec_backend = SpeculativeBackend::Mtp;
            } else if (selected == "dflash") {
                options.spec_backend = SpeculativeBackend::DFlash;
            } else {
                throw std::invalid_argument("--spec must be mtp or dflash");
            }
        } else if (arg == "--adaptive-draft") {
            options.adaptive_draft = true;
        } else if (arg == "--draft-tokens") {
            options.draft_tokens = parse_u32(value("--draft-tokens"), "draft-tokens", true);
            if (options.draft_tokens > kMaxDFlashDraftTokens) {
                throw std::invalid_argument("--draft-tokens must be in [0,7]");
            }
        } else if (arg == "--dflash-verify-width") {
            options.dflash_verify_width =
                parse_u32(value("--dflash-verify-width"), "dflash-verify-width", true);
            if (options.dflash_verify_width > kMaxDFlashVerifyWidth) {
                throw std::invalid_argument(
                    "--dflash-verify-width must be 0 or chain K+1 in [2,8]");
            }
        } else if (arg == "--lm-head-draft") {
            options.proposal_head = ProposalHead::Optimized;
        } else if (arg == "--device") {
            options.device = parse_nonnegative(value("--device"), "device");
        } else if (arg == "--no-device-graph") {
            options.use_device_graph = false;
        } else if (arg == "--profile-measured") {
            options.profile_measured = true;
        } else if (arg == "--retain-token-ids") {
            options.retain_token_ids = true;
        } else if (arg == "--isolate-prompt-decode") {
            options.isolate_prompt_decode = true;
        } else if (arg == "-o" || arg == "--output") {
            const std::string selected = value("--output");
            if (selected == "table") {
                options.output = OutputFormat::Table;
            } else if (selected == "json") {
                options.output = OutputFormat::Json;
            } else if (selected == "csv") {
                options.output = OutputFormat::Csv;
            } else {
                throw std::invalid_argument("--output must be table, json, or csv");
            }
        } else if (arg == "--output-file") {
            options.output_file = value("--output-file");
        } else {
            throw std::invalid_argument("unknown argument: " + std::string(arg));
        }
    }
    if (!saw_artifact) { throw std::invalid_argument("--weights is required"); }
    if (options.prefill_chunk % kPrefillChunkAlignment != 0) {
        throw std::invalid_argument("--prefill-chunk must be a multiple of 128");
    }
    if (options.mixed_forward && *options.mixed_forward % 256 != 0) {
        throw std::invalid_argument("--mixed-forward must be auto or a multiple of 256");
    }
    if (options.spec_backend == SpeculativeBackend::Mtp &&
        options.draft_tokens > kMaxMtpDraftTokens) {
        throw std::invalid_argument("--spec mtp requires --draft-tokens in [0,5]");
    }
    if (options.spec_backend == SpeculativeBackend::DFlash &&
        (options.draft_tokens == 0 || options.draft_tokens > kMaxDFlashDraftTokens)) {
        throw std::invalid_argument("--spec dflash requires --draft-tokens in [1,7]");
    }
    if (options.dflash_verify_width != 0 && options.spec_backend != SpeculativeBackend::DFlash) {
        throw std::invalid_argument("--dflash-verify-width requires --spec dflash");
    }
    if (options.dflash_verify_width != 0 && options.dflash_verify_width < 2) {
        throw std::invalid_argument("--dflash-verify-width must be 0 or chain K+1 in [2,8]");
    }
    if (options.spec_backend == SpeculativeBackend::DFlash) {
        (void)resolved_dflash_verify_width(options.draft_tokens, options.dflash_verify_width);
    }
    if (options.adaptive_draft && options.draft_tokens == 0) {
        throw std::invalid_argument("--adaptive-draft requires an active speculative backend");
    }
    if (options.proposal_head == ProposalHead::Optimized && options.draft_tokens == 0) {
        throw std::invalid_argument("--lm-head-draft requires --draft-tokens greater than zero");
    }
    if (options.retain_token_ids && options.output != OutputFormat::Json) {
        throw std::invalid_argument("--retain-token-ids requires --output json");
    }
    if (options.pair_check && options.concurrency != 2) {
        throw std::invalid_argument("--pair-check requires --concurrency 2");
    }
    if (options.contention && options.concurrency < 2) {
        throw std::invalid_argument("--contention requires --concurrency >= 2");
    }
    if (options.contention_lanes >= options.concurrency) {
        throw std::invalid_argument("--contention-lanes must be below --concurrency");
    }
    if (options.isolate_prompt_decode && options.prompt_gen.empty()) {
        throw std::invalid_argument("--isolate-prompt-decode requires -pg/--prompt-gen");
    }
    return options;
}

std::vector<BenchTest> expand_tests(const BenchOptions& options) {
    std::vector<int> prompts = options.n_prompt;
    std::vector<int> gens    = options.n_gen;
    auto combined            = options.prompt_gen;
    auto whole               = options.whole_inference;
    if (prompts.empty() && gens.empty() && combined.empty() && whole.empty()) {
        prompts.push_back(kDefaultNPrompt);
        gens.push_back(kDefaultNGen);
    }

    std::vector<BenchTest> tests;
    tests.reserve(prompts.size() + gens.size() + combined.size() + whole.size());
    for (const int p : prompts) {
        tests.push_back({TestKind::Prefill, p, 0, "pp" + std::to_string(p)});
    }
    for (const int g : gens) {
        tests.push_back({TestKind::Decode, 0, g, "tg" + std::to_string(g)});
    }
    for (const auto& [p, g] : combined) {
        tests.push_back(
            {TestKind::PrefillDecode, p, g, "pp" + std::to_string(p) + "+tg" + std::to_string(g)});
    }
    for (const auto& [p, g] : whole) {
        tests.push_back({TestKind::WholeInference, p, g,
                         "whole-pp" + std::to_string(p) + "+tg" + std::to_string(g)});
    }
    return tests;
}

std::uint32_t resolve_max_context(const std::vector<BenchTest>& tests,
                                  std::optional<std::uint32_t> override_max_context,
                                  const SpeculativeOptions& spec, bool use_device_graph,
                                  bool isolate_prompt_decode) {
    std::uint32_t required = 0;
    std::string driver;
    bool has_decode = false;
    for (const BenchTest& test : tests) {
        const std::uint32_t base      = test.required_context(spec);
        const std::uint32_t candidate = checked_context(
            static_cast<std::uint64_t>(base) +
                (isolate_prompt_decode && test.kind == TestKind::PrefillDecode ? 1ULL : 0ULL),
            "benchmark isolated context requirement");
        if (candidate > required) {
            required = candidate;
            driver   = test.label;
        }
        has_decode = has_decode || test.has_decode();
    }
    if (use_device_graph && has_decode) {
        const std::uint32_t candidate = decode_graph_prime_required_context(spec);
        if (candidate > required) {
            required = candidate;
            driver   = "decode graph prime";
        }
    }
    if (required == 0) { throw std::invalid_argument("no benchmark tests were selected"); }
    if (override_max_context && *override_max_context < required) {
        throw std::invalid_argument("--max-ctx " + std::to_string(*override_max_context) +
                                    " is too small; " + driver + " needs " +
                                    std::to_string(required));
    }
    return override_max_context.value_or(required);
}

std::uint32_t concurrent_kv_capacity_tokens(std::uint32_t max_context, std::uint32_t concurrency) {
    if (max_context == 0 || concurrency == 0) {
        throw std::invalid_argument("concurrent KV capacity requires max_context and concurrency");
    }
    const std::uint64_t pages =
        (static_cast<std::uint64_t>(max_context) + (kKvPageTokens - 1U)) / kKvPageTokens;
    const std::uint64_t tokens = pages * kKvPageTokens * concurrency;
    if (tokens > std::numeric_limits<std::uint32_t>::max()) {
        throw std::overflow_error("concurrent KV capacity exceeds uint32");
    }
    return static_cast<std::uint32_t>(tokens);
}

void validate_prompt_lengths(const std::vector<BenchTest>& tests, std::size_t corpus_tokens) {
    (void)tests;
    if (corpus_tokens < static_cast<std::size_t>(kDecodeSeedTokens)) {
        throw std::invalid_argument("corpus is too small to seed decode tests");
    }
}

std::vector<TokenId> load_corpus_ids(const std::string& path) {
    std::ifstream input(existing_read_path(path));
    if (!input) { throw std::runtime_error("failed to open corpus ids file: " + path); }
    std::vector<TokenId> ids;
    std::string word;
    while (input >> word) {
        const int parsed = parse_nonnegative(word, "corpus token id");
        ids.push_back(static_cast<TokenId>(parsed));
    }
    if (ids.empty()) { throw std::invalid_argument("corpus ids file is empty: " + path); }
    return ids;
}

std::vector<TokenId> prompt_slice(const std::vector<TokenId>& corpus, int n_prompt,
                                  std::size_t offset) {
    if (n_prompt <= 0 || corpus.empty()) {
        throw std::invalid_argument("invalid prompt slice length: " + std::to_string(n_prompt));
    }
    std::vector<TokenId> out(static_cast<std::size_t>(n_prompt));
    for (int i = 0; i < n_prompt; ++i) {
        out[static_cast<std::size_t>(i)] =
            corpus[(offset + static_cast<std::size_t>(i)) % corpus.size()];
    }
    return out;
}

std::string decode_path_name(bool use_device_graph, const SpeculativeOptions& spec) {
    std::string suffix = use_device_graph ? "device_graph" : "eager";
    if (spec.draft_tokens == 0) { return suffix; }
    const std::string kind = spec.backend == SpeculativeBackend::DFlash ? "dflash" : "mtp";
    return kind + "_" + suffix;
}

std::uint32_t resolved_dflash_verify_width(std::uint32_t draft_tokens,
                                           std::uint32_t requested_width) {
    if (draft_tokens == 0) { return 0; }
    if (draft_tokens > kMaxDFlashDraftTokens) {
        throw std::invalid_argument("DFlash2 requires K in [1,7]");
    }
    if (requested_width != 0 && requested_width != draft_tokens + 1) {
        throw std::invalid_argument("DFlash2 requires chain W=K+1");
    }
    return draft_tokens + 1;
}

std::uint32_t decode_graph_prime_output_tokens(const SpeculativeOptions& spec) {
    return spec.draft_tokens == 0 ? 3 : 2 * (spec.draft_tokens + 1) + 1;
}

std::uint32_t decode_graph_prime_required_context(const SpeculativeOptions& spec) {
    const std::uint64_t outputs = decode_graph_prime_output_tokens(spec);
    return checked_context(outputs + (spec.draft_tokens == 0 ? 0 : 2ULL * spec.draft_tokens),
                           "decode graph prime context requirement");
}

std::string speculative_backend_name(SpeculativeBackend backend) {
    switch (backend) {
    case SpeculativeBackend::None:
        return "none";
    case SpeculativeBackend::Mtp:
        return "mtp";
    case SpeculativeBackend::DFlash:
        return "dflash";
    }
    return "none";
}

Stats compute_stats(const std::vector<double>& values) {
    Stats out;
    out.count = static_cast<int>(values.size());
    if (values.empty()) { return out; }
    for (const double value : values) { out.mean += value; }
    out.mean /= static_cast<double>(values.size());
    if (values.size() > 1) {
        for (const double value : values) { out.stddev += (value - out.mean) * (value - out.mean); }
        out.stddev = std::sqrt(out.stddev / static_cast<double>(values.size() - 1));
    }
    return out;
}

std::vector<double> prefill_tok_s_series(const TestResult& result) {
    std::vector<double> out;
    if (!result.test.has_prefill()) { return out; }
    for (const RepTiming& rep : result.reps) {
        const double seconds =
            result.test.kind == TestKind::Prefill ? rep.wave_seconds : rep.timings.prefill_seconds;
        if (seconds > 0.0) {
            out.push_back(static_cast<double>(result.test.n_prompt) * result.concurrency / seconds);
        }
    }
    return out;
}

std::vector<double> prefill_active_tok_s_series(const TestResult& result) {
    std::vector<double> out;
    if (!result.test.has_prefill()) { return out; }
    for (const RepTiming& rep : result.reps) {
        if (rep.timings.prefill_seconds > 0.0) {
            out.push_back(static_cast<double>(result.test.n_prompt) * result.concurrency /
                          rep.timings.prefill_seconds);
        }
    }
    return out;
}

RepTiming fold_lane_results(const std::vector<GenerationResult>& generated,
                            std::uint32_t expected_per_lane) {
    RepTiming timing;
    timing.timings     = generated.front().timings;
    timing.speculative = generated.front().speculative;
    timing.generated_output_tokens =
        expected_per_lane * static_cast<std::uint32_t>(generated.size());
    timing.generated_token_ids_by_lane.reserve(generated.size());
    for (const GenerationResult& lane : generated) {
        timing.generated_token_ids_by_lane.push_back(lane.generated_token_ids);
        timing.prefill_tail_by_lane.push_back(
            {lane.timings.prefill_tail_tok_s, lane.timings.prefill_tail_window_s});
    }
    for (std::size_t i = 1; i < generated.size(); ++i) {
        const GenerationResult& result = generated[i];
        timing.timings.prepare_seconds += result.timings.prepare_seconds;
        timing.timings.vision_seconds += result.timings.vision_seconds;
        timing.timings.prefill_seconds += result.timings.prefill_seconds;
        timing.timings.decode_seconds =
            std::max(timing.timings.decode_seconds, result.timings.decode_seconds);
        timing.timings.total_seconds =
            std::max(timing.timings.total_seconds, result.timings.total_seconds);
        const SpeculativeStats& in = result.speculative;
        timing.speculative.enabled = timing.speculative.enabled || in.enabled;
        timing.speculative.draft_window =
            std::max(timing.speculative.draft_window, in.draft_window);
        timing.speculative.rounds += in.rounds;
        timing.speculative.drafted_tokens += in.drafted_tokens;
        timing.speculative.accepted_tokens += in.accepted_tokens;
        timing.speculative.fallback_steps += in.fallback_steps;
        if (timing.speculative.accepted_per_position.size() < in.accepted_per_position.size()) {
            timing.speculative.accepted_per_position.resize(in.accepted_per_position.size());
        }
        for (std::size_t j = 0; j < in.accepted_per_position.size(); ++j) {
            timing.speculative.accepted_per_position[j] += in.accepted_per_position[j];
        }
        if (timing.speculative.rounds_per_draft.size() < in.rounds_per_draft.size()) {
            timing.speculative.rounds_per_draft.resize(in.rounds_per_draft.size());
        }
        for (std::size_t j = 0; j < in.rounds_per_draft.size(); ++j) {
            timing.speculative.rounds_per_draft[j] += in.rounds_per_draft[j];
        }
    }
    return timing;
}

std::vector<double> decode_output_tok_s_series(const TestResult& result) {
    std::vector<double> out;
    if (!result.test.has_decode()) { return out; }
    for (const RepTiming& rep : result.reps) {
        if (rep.timings.decode_seconds > 0.0) {
            out.push_back(static_cast<double>(result.test.n_gen) * result.concurrency /
                          rep.timings.decode_seconds);
        }
    }
    return out;
}

std::vector<double> decode_engine_tok_s_series(const TestResult& result) {
    std::vector<double> out;
    if (!result.test.has_decode()) { return out; }
    for (const RepTiming& rep : result.reps) {
        if (rep.timings.decode_seconds > 0.0) {
            out.push_back(static_cast<double>(decode_engine_tokens(result, rep)) /
                          rep.timings.decode_seconds);
        }
    }
    return out;
}

std::vector<double> whole_output_tok_s_series(const TestResult& result) {
    std::vector<double> out;
    if (result.test.kind != TestKind::WholeInference) { return out; }
    for (const RepTiming& rep : result.reps) {
        if (rep.timings.total_seconds > 0.0) {
            out.push_back(static_cast<double>(rep.generated_output_tokens) /
                          rep.timings.total_seconds);
        }
    }
    return out;
}

template <double GenerationTimings::* Field>
std::vector<double> timing_series(const TestResult& result) {
    std::vector<double> out;
    out.reserve(result.reps.size());
    for (const RepTiming& rep : result.reps) { out.push_back(rep.timings.*Field); }
    return out;
}

std::vector<double> prepare_time_series(const TestResult& result) {
    return timing_series<&GenerationTimings::prepare_seconds>(result);
}

std::vector<double> prefill_time_series(const TestResult& result) {
    return result.test.has_prefill() ? timing_series<&GenerationTimings::prefill_seconds>(result)
                                     : std::vector<double>{};
}

std::vector<double> decode_time_series(const TestResult& result) {
    return result.test.has_decode() ? timing_series<&GenerationTimings::decode_seconds>(result)
                                    : std::vector<double>{};
}

std::vector<double> total_time_series(const TestResult& result) {
    return timing_series<&GenerationTimings::total_seconds>(result);
}

std::vector<double> wave_time_series(const TestResult& result) {
    std::vector<double> out;
    out.reserve(result.reps.size());
    for (const RepTiming& rep : result.reps) { out.push_back(rep.wave_seconds); }
    return out;
}

std::string format_table(const BenchEnvironment& env, const std::vector<TestResult>& results) {
    std::ostringstream out;
    out << "ninfer_bench product throughput report\n"
        << "  target:     " << env.load.target << '\n'
        << "  weights:    " << env.load.weights_id << '\n'
        << "  gpu:        " << env.gpu_name << " (" << env.architecture_name << ", device "
        << env.device_id << ")\n"
        << "  hip:        runtime " << env.hip_runtime_version << ", driver "
        << env.hip_driver_version << '\n'
        << "  artifact:   " << env.artifact_path << " (" << env.artifact_file_size_bytes
        << " bytes)\n"
        << "  load:       " << env.load.load_seconds << " s, upload " << env.load.upload_seconds
        << " s, read " << format_bytes(env.load.artifact_bytes_read) << ", H2D "
        << format_bytes(env.load.host_to_device_bytes) << ", staging peak "
        << format_bytes(env.load.peak_staging_bytes) << '\n'
        << "  memory:     weights " << format_bytes(env.memory.weights.capacity_bytes)
        << ", sequence " << format_bytes(env.memory.sequence.capacity_bytes) << ", workspace "
        << format_bytes(env.memory.workspace.capacity_bytes) << ", request transient "
        << format_bytes(env.memory.request_transient.capacity_bytes) << ", graph allowance "
        << format_bytes(env.memory.device_graph_allowance_bytes) << ", KV payload "
        << format_bytes(env.memory.kv_payload_bytes) << '\n'
        << "  corpus:     " << env.corpus_path << " (" << env.corpus_tokens << " tokens)\n"
        << "  config:     max_context=" << env.max_context << " prefill_chunk=" << env.prefill_chunk
        << " kv_format=fp8-k-int4-v concurrency=" << env.concurrency
        << " pending_timeout_ms=" << env.pending_timeout_ms << " pending_deadline=unbounded"
        << " q4_activation_bits=" << ninfer::ops::r9700::linear::kQ4ActivationBits
        << " q4_activation_profile=" << ninfer::ops::r9700::linear::kQ4ActivationProfile
        << " q4_prefill_gate_up_a4="
        << (ninfer::ops::r9700::linear::kQ4PrefillGateUpA4 ? "true" : "false")
        << " q4_prefill_cta_profile=" << ninfer::ops::r9700::linear::kQ4PrefillCtaProfile
        << " dflash_small_t_candidate="
        << (ninfer::ops::r9700::linear::kDFlashSmallTCandidateEnabled ? "true" : "false")
        << " dflash_mlp_down_t5_candidate="
        << (ninfer::ops::r9700::linear::kDFlashMlpDownT5CandidateEnabled ? "true" : "false")
        << " text_p129_wmma_tail_candidate="
        << (ninfer::ops::r9700::kv::kTextP129WmmaTailCandidate ? "true" : "false")
        << " w8_activation_bits=" << ninfer::ops::r9700::linear::kW8ActivationBits
        << " split512_enabled=" << (ninfer::ops::r9700::kv::kFp8QkWmmaDecode ? "true" : "false")
        << " decode_attention_profile=packed-t1to8-device-count-split512-tree-v1"
        << " packed_decode_min_context=" << ninfer::ops::r9700::kv::kPackedDecodeMinimumContext
        << " split512_min_context=" << ninfer::ops::r9700::kv::kSplit512MinimumContext
        << " spec=" << speculative_backend_name(env.speculative_backend)
        << " k=" << env.draft_tokens << " proposal_head=" << proposal_head_name(env.proposal_head)
        << " decode_path="
        << decode_path_name(env.use_device_graph,
                            SpeculativeOptions{env.speculative_backend, env.draft_tokens,
                                               env.proposal_head, env.dflash_verify_width})
        << " graph_prime="
        << (env.decode_graph_primed
                ? std::to_string(env.decode_graph_prime_output_tokens) + " outputs"
                : "n/a")
        << " repetitions=" << env.repetitions << " warmup=" << env.warmup << "\n\n";

    constexpr std::size_t cols                   = 10;
    const std::array<std::string, cols> headings = {
        "test",           "n_prompt",       "n_gen",    "prefill t/s",   "active pp t/s",
        "decode out t/s", "decode eng t/s", "spec acc", "spec round/fb", "work peak"};
    std::vector<std::array<std::string, cols>> rows;
    for (const TestResult& result : results) {
        const SpeculativeStats spec  = aggregate_speculative(result);
        const std::string acceptance = spec.drafted_tokens == 0
                                           ? "n/a"
                                           : number(static_cast<double>(spec.accepted_tokens) /
                                                    static_cast<double>(spec.drafted_tokens));
        const std::string rounds =
            spec.enabled ? std::to_string(spec.rounds) + "/" + std::to_string(spec.fallback_steps)
                         : "n/a";
        rows.push_back({result.test.label, std::to_string(result.test.n_prompt),
                        std::to_string(result.test.n_gen), rate_cell(prefill_tok_s_series(result)),
                        rate_cell(prefill_active_tok_s_series(result)),
                        rate_cell(decode_output_tok_s_series(result)),
                        rate_cell(decode_engine_tok_s_series(result)), acceptance, rounds,
                        format_bytes(result.workspace_peak_bytes)});
    }

    std::array<std::size_t, cols> widths{};
    for (std::size_t i = 0; i < cols; ++i) { widths[i] = headings[i].size(); }
    for (const auto& row : rows) {
        for (std::size_t i = 0; i < cols; ++i) { widths[i] = std::max(widths[i], row[i].size()); }
    }
    const auto print_row = [&](const auto& row) {
        for (std::size_t i = 0; i < cols; ++i) {
            if (i != 0) { out << "  "; }
            out << (i == 0 ? std::left : std::right) << std::setw(static_cast<int>(widths[i]))
                << row[i];
        }
        out << '\n';
    };
    print_row(headings);
    for (const auto& row : rows) { print_row(row); }
    return out.str();
}

std::string format_json(const BenchEnvironment& env, const std::string& command,
                        const std::vector<TestResult>& results) {
    std::ostringstream out;
    out << "{\n"
        << "  \"schema_version\": " << kSchemaVersion << ",\n"
        << "  \"phase_timing_semantics\": \"" << kPhaseTimingSemantics << "\",\n"
        << "  \"artifact_type\": \"" << kArtifactType << "\",\n"
        << "  \"tool\": \"ninfer_bench\",\n"
        << "  \"command\": \"" << json_escape(command) << "\",\n"
        << "  \"environment\": {\"gpu_name\": \"" << json_escape(env.gpu_name)
        << "\", \"architecture_name\": \"" << json_escape(env.architecture_name)
        << "\", \"hip_runtime_version\": \"" << json_escape(env.hip_runtime_version)
        << "\", \"hip_driver_version\": \"" << json_escape(env.hip_driver_version)
        << "\", \"device_id\": " << env.device_id << "},\n"
        << "  \"artifact\": {\"path\": \"" << json_escape(env.artifact_path)
        << "\", \"file_size_bytes\": " << env.artifact_file_size_bytes << "},\n"
        << "  \"load\": {\n"
        << "    \"target\": \"" << json_escape(env.load.target) << "\",\n"
        << "    \"weights_id\": \"" << json_escape(env.load.weights_id) << "\",\n"
        << "    \"load_seconds\": " << number(env.load.load_seconds) << ",\n"
        << "    \"upload_seconds\": " << number(env.load.upload_seconds) << ",\n"
        << "    \"artifact_bytes_read\": " << env.load.artifact_bytes_read << ",\n"
        << "    \"host_to_device_bytes\": " << env.load.host_to_device_bytes << ",\n"
        << "    \"peak_staging_bytes\": " << env.load.peak_staging_bytes << ",\n"
        << "    \"tensor_count\": " << env.load.tensor_count << ",\n"
        << "    \"resource_count\": " << env.load.resource_count << "\n"
        << "  },\n"
        << "  \"memory\": {\n"
        << "    \"device\": " << env.memory.device << ",\n"
        << "    \"max_context\": " << env.memory.max_context << ",\n"
        << "    \"kv_capacity_mode\": \""
        << (env.memory.kv_capacity_mode == KvCapacityMode::Automatic ? "auto" : "explicit")
        << "\",\n"
        << "    \"kv_capacity\": " << env.memory.kv_capacity << ",\n"
        << "    \"kv_capacity_page_groups\": " << env.memory.kv_capacity_page_groups << ",\n"
        << "    \"kv_capacity_max_page_groups\": " << env.memory.kv_capacity_max_page_groups
        << ",\n"
        << "    \"kv_cache_format\": \"fp8-k-int4-v\",\n";
    append_arena_json(out, "weights", env.memory.weights, "    ", true);
    append_arena_json(out, "sequence", env.memory.sequence, "    ", true);
    append_arena_json(out, "workspace", env.memory.workspace, "    ", true);
    append_arena_json(out, "request_transient", env.memory.request_transient, "    ", true);
    out << "    \"minimum_runtime_reservation_bytes\": "
        << env.memory.minimum_runtime_reservation_bytes << ",\n"
        << "    \"kv_capacity_increment_bytes\": " << env.memory.kv_capacity_increment_bytes
        << ",\n"
        << "    \"runtime_reservation_bytes\": " << env.memory.runtime_reservation_bytes << ",\n"
        << "    \"available_after_weights_bytes\": " << env.memory.available_after_weights_bytes
        << ",\n"
        << "    \"available_after_startup_bytes\": " << env.memory.available_after_startup_bytes
        << ",\n"
        << "    \"kv_capacity_headroom_bytes\": " << env.memory.kv_capacity_headroom_bytes << ",\n"
        << "    \"planned_slack_bytes\": " << env.memory.planned_slack_bytes << ",\n"
        << "    \"device_graph_allowance_bytes\": " << env.memory.device_graph_allowance_bytes
        << ",\n"
        << "    \"device_graph_observed_bytes\": " << env.memory.device_graph_observed_bytes
        << ",\n"
        << "    \"kv_payload_bytes\": " << env.memory.kv_payload_bytes << "\n"
        << "  },\n"
        << "  \"config\": {\n"
        << "    \"max_context\": " << env.max_context << ",\n"
        << "    \"prefill_chunk\": " << env.prefill_chunk << ",\n"
        << "    \"kv_cache_format\": \"fp8-k-int4-v\",\n"
        << "    \"kv_value_group\": " << ninfer::targets::qwen3::detail::kR9700TextKVValueGroup
        << ",\n"
        << "    \"kv_plane_layouts\": {\"key\": \""
        << ninfer::targets::qwen3::detail::kR9700TextKVKeyPlaneLayoutName << "\", \"value\": \""
        << ninfer::targets::qwen3::detail::kR9700TextKVValuePlaneLayoutName
        << "\", \"value_scale\": \""
        << ninfer::targets::qwen3::detail::kR9700TextKVValueScalePlaneLayoutName << "\"},\n"
        << "    \"q4_activation_bits\": " << ninfer::ops::r9700::linear::kQ4ActivationBits << ",\n"
        << "    \"q4_activation_profile\": \"" << ninfer::ops::r9700::linear::kQ4ActivationProfile
        << "\",\n"
        << "    \"q4_prefill_gate_up_a4\": "
        << (ninfer::ops::r9700::linear::kQ4PrefillGateUpA4 ? "true" : "false") << ",\n"
        << "    \"q4_prefill_cta_profile\": \"" << ninfer::ops::r9700::linear::kQ4PrefillCtaProfile
        << "\",\n"
        << "    \"dflash_small_t_candidate\": "
        << (ninfer::ops::r9700::linear::kDFlashSmallTCandidateEnabled ? "true" : "false") << ",\n"
        << "    \"dflash_mlp_down_t5_candidate\": "
        << (ninfer::ops::r9700::linear::kDFlashMlpDownT5CandidateEnabled ? "true" : "false")
        << ",\n"
        << "    \"text_p129_wmma_tail_candidate\": "
        << (ninfer::ops::r9700::kv::kTextP129WmmaTailCandidate ? "true" : "false") << ",\n"
        << "    \"w8_activation_bits\": " << ninfer::ops::r9700::linear::kW8ActivationBits << ",\n"
        << "    \"split512_enabled\": "
        << (ninfer::ops::r9700::kv::kFp8QkWmmaDecode ? "true" : "false") << ",\n"
        << "    \"decode_attention_profile\": \"packed-t1to8-device-count-split512-tree-v1\",\n"
        << "    \"packed_decode_min_context\": "
        << ninfer::ops::r9700::kv::kPackedDecodeMinimumContext << ",\n"
        << "    \"split512_min_context\": " << ninfer::ops::r9700::kv::kSplit512MinimumContext
        << ",\n"
#if defined(NINFER_R9700_XATTENTION_QUALIFICATION)
        << "    \"xattention_qualification\": true,\n"
        << "    \"xattention_profile\": \"b128-s" << ninfer::ops::r9700::kv::kXAttentionStride
        << "-tau" << ninfer::ops::r9700::kv::kXAttentionTauPermille << "\",\n"
        << "    \"xattention_find_block\": " << ninfer::ops::r9700::kv::kXAttentionFindBlockSize
        << ",\n"
        << "    \"xattention_stride\": " << ninfer::ops::r9700::kv::kXAttentionStride << ",\n"
        << "    \"xattention_tau_permille\": " << ninfer::ops::r9700::kv::kXAttentionTauPermille
        << ",\n"
#else
        << "    \"xattention_qualification\": false,\n"
#endif
        << "    \"concurrency\": " << env.concurrency << ",\n"
        << "    \"pending_timeout_ms\": " << env.pending_timeout_ms << ",\n"
        << "    \"pending_deadline\": \"unbounded\",\n"
        << "    \"spec\": \"" << speculative_backend_name(env.speculative_backend) << "\",\n"
        << "    \"draft_tokens\": " << env.draft_tokens << ",\n"
        << "    \"speculative_execution\": " << (env.draft_tokens == 0 ? "false" : "true") << ",\n"
        << "    \"dflash_verify_width_requested\": " << env.dflash_verify_width_requested << ",\n"
        << "    \"dflash_verify_width\": " << env.dflash_verify_width << ",\n"
        << "    \"proposal_head\": \"" << proposal_head_name(env.proposal_head) << "\",\n"
        << "    \"use_device_graph\": " << (env.use_device_graph ? "true" : "false") << ",\n"
        << "    \"retain_token_ids\": " << (env.retain_token_ids ? "true" : "false") << ",\n"
        << "    \"isolate_prompt_decode\": " << (env.isolate_prompt_decode ? "true" : "false")
        << ",\n"
        << "    \"decode_path\": \""
        << decode_path_name(env.use_device_graph,
                            SpeculativeOptions{env.speculative_backend, env.draft_tokens,
                                               env.proposal_head, env.dflash_verify_width})
        << "\",\n"
        << "    \"decode_graph_prime\": {\"primed\": "
        << (env.decode_graph_primed ? "true" : "false")
        << ", \"output_tokens\": " << env.decode_graph_prime_output_tokens << "},\n"
        << "    \"repetitions\": " << env.repetitions << ",\n"
        << "    \"warmup\": " << env.warmup << ",\n"
        << "    \"corpus_path\": \"" << json_escape(env.corpus_path) << "\",\n"
        << "    \"corpus_tokens\": " << env.corpus_tokens << "\n"
        << "  },\n"
        << "  \"tests\": [\n";

    for (std::size_t i = 0; i < results.size(); ++i) {
        const TestResult& result = results[i];
        out << "    {\n"
            << "      \"label\": \"" << json_escape(result.test.label) << "\",\n"
            << "      \"kind\": \"" << kind_string(result.test.kind) << "\",\n"
            << "      \"n_prompt\": " << result.test.n_prompt << ",\n"
            << "      \"n_gen\": " << result.test.n_gen << ",\n"
            << "      \"requested_output_tokens\": " << result.test.requested_output_tokens()
            << ",\n";
        append_stat(out, "prefill_tok_s", prefill_tok_s_series(result), "      ");
        out << ",\n";
        append_stat(out, "prefill_active_tok_s", prefill_active_tok_s_series(result), "      ");
        out << ",\n";
        append_stat(out, "decode_output_tok_s", decode_output_tok_s_series(result), "      ");
        out << ",\n";
        append_stat(out, "decode_engine_tok_s", decode_engine_tok_s_series(result), "      ");
        out << ",\n";
        append_stat(out, "whole_output_tok_s", whole_output_tok_s_series(result), "      ");
        out << ",\n";
        append_stat(out, "prepare_seconds", prepare_time_series(result), "      ");
        out << ",\n";
        append_stat(out, "prefill_seconds", prefill_time_series(result), "      ");
        out << ",\n";
        append_stat(out, "decode_seconds", decode_time_series(result), "      ");
        out << ",\n";
        append_stat(out, "total_seconds", total_time_series(result), "      ");
        out << ",\n";
        append_stat(out, "wave_seconds", wave_time_series(result), "      ");
        out << ",\n      \"workspace_peak_bytes\": " << result.workspace_peak_bytes
            << ",\n      \"workspace_allocator_peak_bytes\": "
            << result.workspace_allocator_peak_bytes << ",\n";
        append_speculative_json(out, aggregate_speculative(result), "      ");
        out << ",\n      \"reps\": [\n";
        for (std::size_t r = 0; r < result.reps.size(); ++r) {
            const RepTiming& rep = result.reps[r];
            out << "        {\n"
                << "          \"generated_output_tokens\": " << rep.generated_output_tokens << ",\n"
                << "          \"decode_output_tokens\": "
                << (result.test.has_decode() ? std::to_string(result.test.n_gen) : "null") << ",\n"
                << "          \"decode_engine_tokens\": "
                << (result.test.has_decode() ? std::to_string(decode_engine_tokens(result, rep))
                                             : "null")
                << ",\n";
            if (env.retain_token_ids) {
                out << "          \"generated_token_ids_by_lane\": [";
                for (std::size_t lane = 0; lane < rep.generated_token_ids_by_lane.size(); ++lane) {
                    if (lane != 0) { out << ", "; }
                    out << '[';
                    const auto& tokens = rep.generated_token_ids_by_lane[lane];
                    for (std::size_t token = 0; token < tokens.size(); ++token) {
                        if (token != 0) { out << ", "; }
                        out << tokens[token];
                    }
                    out << ']';
                }
                out << "],\n";
            }
            out << "          \"prefill_tail_by_lane\": [";
            for (std::size_t lane = 0; lane < rep.prefill_tail_by_lane.size(); ++lane) {
                if (lane != 0) { out << ", "; }
                const auto& tail = rep.prefill_tail_by_lane[lane];
                out << "{\"tok_s\": " << number(tail.tok_s)
                    << ", \"window_s\": " << number(tail.window_s) << '}';
            }
            out << "],\n          \"wave_seconds\": " << number(rep.wave_seconds) << ",\n";
            append_timings_json(out, rep.timings, "          ");
            out << ",\n";
            append_speculative_json(out, rep.speculative, "          ");
            out << "\n        }" << (r + 1 == result.reps.size() ? "" : ",") << '\n';
        }
        out << "      ]\n    }" << (i + 1 == results.size() ? "" : ",") << '\n';
    }
    out << "  ]\n}\n";
    return out.str();
}

std::string format_csv(const BenchEnvironment& env, const std::vector<TestResult>& results) {
    std::ostringstream out;
    out << "label,kind,n_prompt,n_gen,target,weights_id,max_context,prefill_chunk,concurrency,"
           "spec,draft_tokens,dflash_verify_width_requested,dflash_verify_width,proposal_head,"
           "decode_path,kv_cache_format,"
           "kv_value_group,kv_key_plane_layout,kv_value_plane_layout,"
           "kv_value_scale_plane_layout,q4_activation_bits,q4_activation_profile,"
           "q4_prefill_gate_up_a4,q4_prefill_cta_profile,"
           "dflash_small_t_candidate,dflash_mlp_down_t5_candidate,"
           "text_p129_wmma_tail_candidate,w8_activation_bits,"
           "split512_enabled,"
           "decode_attention_profile,packed_decode_min_context,split512_min_context,"
           "kv_payload_bytes,load_host_to_device_bytes,"
           "weights_capacity_bytes,sequence_capacity_bytes,workspace_capacity_bytes,"
           "request_transient_capacity_bytes,device_graph_allowance_bytes,"
           "workspace_peak_bytes,workspace_allocator_peak_bytes,"
           "spec_rounds,spec_fallback_steps,spec_acceptance_rate,"
           "repetitions,prefill_tok_s_mean,prefill_tok_s_stddev,prefill_active_tok_s_mean,"
           "prefill_active_tok_s_stddev,decode_output_tok_s_mean,"
           "decode_output_tok_s_stddev,decode_engine_tok_s_mean,decode_engine_tok_s_stddev,"
           "whole_output_tok_s_mean,whole_output_tok_s_stddev,"
           "prepare_seconds_mean,prefill_seconds_mean,decode_seconds_mean,total_seconds_mean,"
           "wave_seconds_mean\n";
    const auto mean = [](const std::vector<double>& values) {
        return values.empty() ? std::string() : number(compute_stats(values).mean);
    };
    const auto stddev = [](const std::vector<double>& values) {
        return values.empty() ? std::string() : number(compute_stats(values).stddev);
    };
    for (const TestResult& result : results) {
        const SpeculativeStats spec  = aggregate_speculative(result);
        const std::string acceptance = spec.drafted_tokens == 0
                                           ? std::string()
                                           : number(static_cast<double>(spec.accepted_tokens) /
                                                    static_cast<double>(spec.drafted_tokens));
        out << result.test.label << ',' << kind_string(result.test.kind) << ','
            << result.test.n_prompt << ',' << result.test.n_gen << ',' << env.load.target << ','
            << env.load.weights_id << ',' << env.max_context << ',' << env.prefill_chunk << ','
            << env.concurrency << ',' << speculative_backend_name(env.speculative_backend) << ','
            << env.draft_tokens << ',' << env.dflash_verify_width_requested << ','
            << env.dflash_verify_width << ',' << proposal_head_name(env.proposal_head) << ','
            << decode_path_name(env.use_device_graph,
                                SpeculativeOptions{env.speculative_backend, env.draft_tokens,
                                                   env.proposal_head, env.dflash_verify_width})
            << ',' << "fp8-k-int4-v," << ninfer::targets::qwen3::detail::kR9700TextKVValueGroup
            << ',' << ninfer::targets::qwen3::detail::kR9700TextKVKeyPlaneLayoutName << ','
            << ninfer::targets::qwen3::detail::kR9700TextKVValuePlaneLayoutName << ','
            << ninfer::targets::qwen3::detail::kR9700TextKVValueScalePlaneLayoutName << ','
            << ninfer::ops::r9700::linear::kQ4ActivationBits << ','
            << ninfer::ops::r9700::linear::kQ4ActivationProfile << ','
            << (ninfer::ops::r9700::linear::kQ4PrefillGateUpA4 ? "true" : "false") << ','
            << ninfer::ops::r9700::linear::kQ4PrefillCtaProfile << ','
            << (ninfer::ops::r9700::linear::kDFlashSmallTCandidateEnabled ? "true" : "false") << ','
            << (ninfer::ops::r9700::linear::kDFlashMlpDownT5CandidateEnabled ? "true" : "false")
            << ',' << (ninfer::ops::r9700::kv::kTextP129WmmaTailCandidate ? "true" : "false") << ','
            << ninfer::ops::r9700::linear::kW8ActivationBits << ','
            << (ninfer::ops::r9700::kv::kFp8QkWmmaDecode ? "true" : "false") << ','
            << "packed-t1to8-device-count-split512-tree-v1,"
            << ninfer::ops::r9700::kv::kPackedDecodeMinimumContext << ','
            << ninfer::ops::r9700::kv::kSplit512MinimumContext << ',' << env.memory.kv_payload_bytes
            << ',' << env.load.host_to_device_bytes << ',' << env.memory.weights.capacity_bytes
            << ',' << env.memory.sequence.capacity_bytes << ','
            << env.memory.workspace.capacity_bytes << ','
            << env.memory.request_transient.capacity_bytes << ','
            << env.memory.device_graph_allowance_bytes << ',' << result.workspace_peak_bytes << ','
            << result.workspace_allocator_peak_bytes << ',' << spec.rounds << ','
            << spec.fallback_steps << ',' << acceptance << ',' << result.reps.size() << ','
            << mean(prefill_tok_s_series(result)) << ',' << stddev(prefill_tok_s_series(result))
            << ',' << mean(prefill_active_tok_s_series(result)) << ','
            << stddev(prefill_active_tok_s_series(result)) << ','
            << mean(decode_output_tok_s_series(result)) << ','
            << stddev(decode_output_tok_s_series(result)) << ','
            << mean(decode_engine_tok_s_series(result)) << ','
            << stddev(decode_engine_tok_s_series(result)) << ','
            << mean(whole_output_tok_s_series(result)) << ','
            << stddev(whole_output_tok_s_series(result)) << ',' << mean(prepare_time_series(result))
            << ',' << mean(prefill_time_series(result)) << ',' << mean(decode_time_series(result))
            << ',' << mean(total_time_series(result)) << ',' << mean(wave_time_series(result))
            << '\n';
    }
    return out.str();
}

std::string json_escape(std::string_view value) {
    std::string out;
    for (const char c : value) {
        switch (c) {
        case '"':
            out += "\\\"";
            break;
        case '\\':
            out += "\\\\";
            break;
        case '\n':
            out += "\\n";
            break;
        case '\r':
            out += "\\r";
            break;
        case '\t':
            out += "\\t";
            break;
        default:
            if (static_cast<unsigned char>(c) < 0x20) {
                std::ostringstream escaped;
                escaped << "\\u" << std::hex << std::setw(4) << std::setfill('0')
                        << static_cast<int>(static_cast<unsigned char>(c));
                out += escaped.str();
            } else {
                out += c;
            }
        }
    }
    return out;
}

std::string proposal_head_name(ProposalHead head) {
    switch (head) {
    case ProposalHead::Full:
        return "full";
    case ProposalHead::Optimized:
        return "optimized";
    }
    return "unknown";
}

std::uint64_t file_size_or_zero(const std::string& path) {
    std::error_code error;
    const auto size = std::filesystem::file_size(path, error);
    return error ? 0 : static_cast<std::uint64_t>(size);
}

} // namespace ninfer::bench
