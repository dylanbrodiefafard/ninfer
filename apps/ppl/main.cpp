#include "artifact/reader.h"
#include "ninfer/engine.h"
#include "product/speculative_options.h"
#include "targets/qwen3_6/impl/frontend/tokenizer.h"

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace {

ninfer::KvCacheStorage parse_kv_dtype(std::string_view text) {
    if (text == "bf16") { return ninfer::KvCacheStorage::BFloat16; }
    if (text == "int8") { return ninfer::KvCacheStorage::Int8Group64; }
    if (text == "nvfp4") { return ninfer::KvCacheStorage::Nvfp4; }
    throw std::invalid_argument("--kv-dtype must be bf16, int8, or nvfp4");
}

ninfer::ScoreSchedule parse_schedule(std::string_view text) {
    if (text == "prefill") { return ninfer::ScoreSchedule::Prefill; }
    if (text == "decode") { return ninfer::ScoreSchedule::Decode; }
    throw std::invalid_argument("--schedule must be prefill or decode");
}

std::optional<std::uint32_t> parse_skip(std::string_view text) {
    if (text == "half") { return std::nullopt; }
    const unsigned long value = std::stoul(std::string(text));
    if (value > 2147483647ul) { throw std::invalid_argument("--skip is out of range"); }
    return static_cast<std::uint32_t>(value);
}

const char* kv_dtype_name(ninfer::KvCacheStorage storage) {
    switch (storage) {
    case ninfer::KvCacheStorage::BFloat16:
        return "bf16";
    case ninfer::KvCacheStorage::Int8Group64:
        return "int8";
    case ninfer::KvCacheStorage::Nvfp4:
        return "nvfp4";
    }
    return "unknown";
}

std::vector<ninfer::TokenId> load_ids(const std::string& path, std::uint32_t limit) {
    std::ifstream input(path);
    if (!input) { throw std::runtime_error("failed to open corpus ids: " + path); }
    std::vector<ninfer::TokenId> ids;
    std::string word;
    while (input >> word) {
        char* end                      = nullptr;
        const unsigned long value      = std::strtoul(word.c_str(), &end, 10);
        if (end == word.c_str() || *end != '\0' || value > 2147483647ul) {
            throw std::invalid_argument("invalid corpus token id: " + word);
        }
        ids.push_back(static_cast<ninfer::TokenId>(value));
        if (limit > 0 && ids.size() >= limit) { break; }
    }
    if (ids.size() < 2) { throw std::invalid_argument("corpus must contain at least two token ids"); }
    return ids;
}

std::string json_escape(std::string_view text) {
    std::ostringstream out;
    out << '"';
    for (const unsigned char c : text) {
        if (c == '"' || c == '\\') {
            out << '\\' << static_cast<char>(c);
        } else if (c < 0x20) {
            out << "\\u" << std::hex << std::setw(4) << std::setfill('0') << static_cast<int>(c)
                << std::dec;
        } else {
            out << static_cast<char>(c);
        }
    }
    out << '"';
    return out.str();
}

void write_token_nlls(const std::string& json_path, const std::vector<float>& values) {
    if (json_path.empty() || json_path == "-") { return; }
    const auto dot = json_path.rfind('.');
    const std::string nll_path =
        (dot == std::string::npos ? json_path : json_path.substr(0, dot)) + ".nllf32";
    std::ofstream output(nll_path, std::ios::binary);
    if (!output) { throw std::runtime_error("failed to write token nlls: " + nll_path); }
    output.write(reinterpret_cast<const char*>(values.data()),
                 static_cast<std::streamsize>(values.size() * sizeof(float)));
}

void write_cell_json(const std::string& path, const std::string& scheme, const std::string& weights,
                      const ninfer::LoadSummary& load, ninfer::KvCacheStorage kv_dtype,
                      std::uint32_t prefill_chunk, bool use_cuda_graph,
                      ninfer::SpeculativeBackend spec, std::uint32_t draft_tokens,
                      bool sage_attn, std::optional<float> keep_frac,
                      const ninfer::ScoreResult& score) {
    std::ostringstream body;
    body << std::setprecision(17);
    body << "{\n"
         << "  \"scheme\": " << json_escape(scheme) << ",\n"
         << "  \"weights\": " << json_escape(weights) << ",\n"
         << "  \"model_id\": " << json_escape(load.model_id) << ",\n"
         << "  \"weights_id\": " << json_escape(load.weights_id) << ",\n"
         << "  \"kv_dtype\": " << json_escape(kv_dtype_name(kv_dtype)) << ",\n"
         << "  \"schedule\": " << json_escape(ninfer::score_schedule_name(score.schedule)) << ",\n"
         << "  \"spec\": " << json_escape(ninfer::product::speculative_backend_name(spec)) << ",\n"
         << "  \"draft_tokens\": " << draft_tokens << ",\n"
         << "  \"cuda_graph\": " << (use_cuda_graph ? "true" : "false") << ",\n"
          << "  \"prefill_chunk\": " << prefill_chunk << ",\n"
          << "  \"sage_attn\": " << (sage_attn ? "true" : "false") << ",\n"
          << "  \"keep_frac\": " << keep_frac.value_or(1.0f) << ",\n"
         << "  \"skip_tokens\": " << score.skip_tokens << ",\n"
         << "  \"prompt_tokens\": " << score.prompt_tokens << ",\n"
         << "  \"tokens_scored\": " << score.tokens_scored << ",\n"
         << "  \"non_finite\": " << score.non_finite << ",\n"
         << "  \"terrible_tokens\": " << score.terrible_tokens << ",\n"
         << "  \"terrible_nll\": " << ninfer::kScoreTerribleNll << ",\n"
         << "  \"sum_nll\": " << score.sum_nll << ",\n"
         << "  \"mean_nll\": " << score.mean_nll << ",\n"
         << "  \"max_nll\": " << score.max_nll << ",\n"
         << "  \"ppl\": " << score.perplexity << ",\n"
         << "  \"score_seconds\": " << score.score_seconds << "\n"
         << "}\n";
    if (path.empty() || path == "-") {
        std::cout << body.str();
        return;
    }
    std::ofstream output(path);
    if (!output) { throw std::runtime_error("failed to write cell json: " + path); }
    output << body.str();
    write_token_nlls(path, score.token_nlls);
}

std::string load_resource(const ninfer::artifact::Reader& reader, std::string_view name) {
    const ninfer::artifact::PayloadSpan span = reader.payload(name);
    return std::string(reinterpret_cast<const char*>(span.data.data()), span.data.size());
}

std::string read_text_file(const std::string& path) {
    std::ifstream input(path, std::ios::binary);
    if (!input) { throw std::runtime_error("failed to open text file: " + path); }
    std::ostringstream body;
    body << input.rdbuf();
    return body.str();
}

void encode_text_file(const std::string& weights, const std::string& text_path,
                      const std::string& out_ids, std::uint32_t limit) {
    ninfer::artifact::Reader reader(weights);
    const std::string tokenizer_json = load_resource(reader, "frontend/tokenizer.json");
    const std::string tokenizer_config_json =
        load_resource(reader, "frontend/tokenizer_config.json");
    const std::string generation_config_json =
        load_resource(reader, "frontend/generation_config.json");
    const ninfer::targets::qwen3_6::frontend_internal::Tokenizer tokenizer(
        {.tokenizer_json         = tokenizer_json,
         .tokenizer_config_json  = tokenizer_config_json,
         .generation_config_json = generation_config_json});
    const std::vector<int> encoded = tokenizer.encode(read_text_file(text_path));
    if (encoded.size() < 2) {
        throw std::invalid_argument("encoded text produced fewer than two tokens");
    }
    const std::size_t count =
        limit == 0 ? encoded.size() : std::min(encoded.size(), static_cast<std::size_t>(limit));
    std::ofstream output(out_ids);
    if (!output) { throw std::runtime_error("failed to write ids: " + out_ids); }
    for (std::size_t i = 0; i < count; ++i) {
        if (i != 0) { output << ' '; }
        output << encoded[i];
    }
    output << '\n';
}

} // namespace

int main(int argc, char** argv) {
    try {
        std::string weights;
        std::string ids_path;
        std::string text_path;
        std::string scheme = "kv-bf16";
        std::string out_json;
        ninfer::KvCacheStorage kv_dtype = ninfer::KvCacheStorage::BFloat16;
        ninfer::ScoreOptions score_options;
        ninfer::SpeculativeOptions speculative;
        std::uint32_t tokens            = 0;
        std::uint32_t prefill_chunk     = 4096;
        int device                      = 0;
        bool sage_attn                  = false;
        std::optional<float> keep_frac;
        bool help                       = false;
        bool encode                     = false;
        bool use_cuda_graph             = true;
        for (int i = 1; i < argc; ++i) {
            const std::string_view arg(argv[i]);
            auto value = [&](const char* flag) -> const char* {
                if (i + 1 >= argc) { throw std::invalid_argument(std::string(flag) + " requires a value"); }
                return argv[++i];
            };
            if (arg == "-h" || arg == "--help") {
                help = true;
            } else if (arg == "--weights") {
                weights = value("--weights");
            } else if (arg == "--ids") {
                ids_path = value("--ids");
            } else if (arg == "--encode") {
                encode = true;
            } else if (arg == "--text") {
                text_path = value("--text");
            } else if (arg == "--scheme") {
                scheme = value("--scheme");
            } else if (arg == "--kv-dtype") {
                kv_dtype = parse_kv_dtype(value("--kv-dtype"));
            } else if (arg == "--sage") {
                sage_attn = true;
            } else if (arg == "--keep-frac") {
                const std::string keep_frac_raw = value("--keep-frac");
                char* keep_frac_end              = nullptr;
                const float keep_frac_parsed =
                    std::strtof(keep_frac_raw.c_str(), &keep_frac_end);
                if (keep_frac_end == keep_frac_raw.c_str() || *keep_frac_end != '\0' ||
                    !(keep_frac_parsed > 0.0f && keep_frac_parsed <= 1.0f)) {
                    throw std::invalid_argument("--keep-frac must be a float in (0, 1]");
                }
                keep_frac = keep_frac_parsed;
            } else if (arg == "--schedule") {
                score_options.schedule = parse_schedule(value("--schedule"));
            } else if (arg == "--skip") {
                score_options.skip_tokens = parse_skip(value("--skip"));
            } else if (arg == "--spec") {
                speculative.backend = ninfer::product::parse_speculative_backend(value("--spec"));
            } else if (arg == "--draft-tokens") {
                speculative.draft_tokens = static_cast<std::uint32_t>(std::stoul(value("--draft-tokens")));
            } else if (arg == "--cuda-graph") {
                use_cuda_graph = true;
            } else if (arg == "--no-cuda-graph") {
                use_cuda_graph = false;
            } else if (arg == "--tokens") {
                tokens = static_cast<std::uint32_t>(std::stoul(value("--tokens")));
            } else if (arg == "--prefill-chunk") {
                prefill_chunk = static_cast<std::uint32_t>(std::stoul(value("--prefill-chunk")));
            } else if (arg == "--device") {
                device = std::stoi(value("--device"));
            } else if (arg == "--out-json") {
                out_json = value("--out-json");
            } else {
                throw std::invalid_argument("unknown argument: " + std::string(arg));
            }
        }
        if (help) {
            std::cout
                << "Usage: ninfer-ppl --weights <artifact.ninfer> --ids <corpus.ids> [options]\n"
                << "       ninfer-ppl --encode --weights <artifact.ninfer> --text <file> --ids <out.ids>\n"
                << "  --scheme <name>             cell name (default: kv-bf16)\n"
                << "  --kv-dtype <bf16|int8|nvfp4>\n"
                << "  --sage                  sage_attn FP4-PV recipe (requires --kv-dtype nvfp4)\n"
                << "  --keep-frac <f>         key-tile keep fraction (0,1]; <1 requires --sage\n"
                << "  --schedule <prefill|decode> default prefill (prompt-route GQA)\n"
                << "  --skip <half|n>             warmup tokens not scored (default: half)\n"
                << "  --spec <mtp|dflash>         load a speculative backend (decode score: mtp)\n"
                << "  --draft-tokens <n>          required with --spec mtp|dflash\n"
                << "  --cuda-graph / --no-cuda-graph  default: graphs on (production decode)\n"
                << "  --tokens <n>                score/encode the first n ids (default: all)\n"
                << "  --prefill-chunk <n>         default 4096\n"
                << "  --device <id>\n"
                << "  --out-json <path|->\n";
            return 0;
        }
        if (weights.empty()) { throw std::invalid_argument("--weights is required"); }
        if (encode) {
            if (text_path.empty() || ids_path.empty()) {
                throw std::invalid_argument("--encode requires --text and --ids");
            }
            encode_text_file(weights, text_path, ids_path, tokens);
            return 0;
        }
        if (ids_path.empty()) {
            throw std::invalid_argument("--weights and --ids are required");
        }

        ninfer::product::validate_speculative_cli_options(speculative);
        if (speculative.backend == ninfer::SpeculativeBackend::DFlash) {
            throw std::invalid_argument(
                "ninfer-ppl does not teacher-force DFlash; use --spec mtp or ordinary decode");
        }
        const std::vector<ninfer::TokenId> ids = load_ids(ids_path, tokens);
        if (sage_attn && kv_dtype != ninfer::KvCacheStorage::Nvfp4) {
            throw std::invalid_argument("--sage requires --kv-dtype nvfp4 (sage_pv k_mean plane)");
        }
        if (keep_frac && *keep_frac < 1.0f && !sage_attn) {
            throw std::invalid_argument(
                "--keep-frac < 1.0 enables the sage_pv tile-skip proxy and requires --sage "
                "(nvfp4 KV); --keep-frac 1.0 is exact attention on any KV dtype");
        }
        ninfer::EngineOptions options;
        options.artifact_path    = weights;
        options.device           = device;
        options.max_context      = static_cast<std::uint32_t>(ids.size());
        options.kv_capacity      = ninfer::KvCapacityPolicy::explicit_capacity(options.max_context);
        options.max_concurrency  = 1;
        options.prefill_chunk    = prefill_chunk;
        options.kv_cache         = kv_dtype;
        options.sage_attn        = sage_attn;
        options.speculative      = speculative;
        options.enable_vision    = false;
        options.use_cuda_graph   = use_cuda_graph;

        if (keep_frac) {
            char keep_frac_env[32];
            std::snprintf(keep_frac_env, sizeof(keep_frac_env), "%.10g", *keep_frac);
            setenv("NINFER_KEEP_FRAC", keep_frac_env, 1);
        }

        ninfer::Engine engine(std::move(options));
        ninfer::PreparedPrompt prompt = engine.prepare_tokens(ids, false);
        const ninfer::ScoreResult score = engine.score(std::move(prompt), score_options);
        write_cell_json(out_json, scheme, weights, engine.load_summary(), kv_dtype, prefill_chunk,
                        use_cuda_graph, speculative.backend, speculative.draft_tokens, sage_attn,
                        keep_frac, score);
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "ninfer-ppl: " << error.what() << '\n';
        return 1;
    }
}
