#include "ninfer/engine.h"

#include <algorithm>
#include <array>
#include <cctype>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <exception>
#include <fstream>
#include <iostream>
#include <iterator>
#include <new>
#include <string>
#include <utility>
#include <vector>

namespace {

ninfer::EngineOptions base_engine_options(const char* artifact) {
    ninfer::EngineOptions options;
    options.artifact_path  = artifact;
    options.max_context    = 512;
    options.kv_capacity    = ninfer::KvCapacityPolicy::explicit_capacity(512);
    options.prefill_chunk  = 128;
    options.kv_cache       = ninfer::KvCacheStorage::Nvfp4;
    if (std::getenv("NINFER_DFLASH_TEST_BF16_KV") != nullptr) {
        options.kv_cache = ninfer::KvCacheStorage::BFloat16;
    }
    options.use_cuda_graph = std::getenv("NINFER_DFLASH_TEST_NO_GRAPH") == nullptr;
    options.enable_vision  = false;
    return options;
}

ninfer::EngineOptions speculative_engine_options(const char* artifact,
                                                 ninfer::SpeculativeBackend backend,
                                                 std::uint32_t draft_tokens,
                                                 std::uint32_t max_concurrency) {
    ninfer::EngineOptions options     = base_engine_options(artifact);
    options.speculative.backend       = backend;
    options.speculative.draft_tokens  = draft_tokens;
    options.speculative.proposal_head = ninfer::ProposalHead::Optimized;
    options.max_concurrency           = max_concurrency;
    return options;
}

ninfer::EngineOptions adaptive_engine_options(const char* artifact,
                                              ninfer::SpeculativeBackend backend,
                                              std::uint32_t draft_tokens,
                                              std::uint32_t max_concurrency) {
    ninfer::EngineOptions options =
        speculative_engine_options(artifact, backend, draft_tokens, max_concurrency);
    options.speculative.adaptive_draft = true;
    return options;
}

ninfer::RequestOptions greedy_options(std::uint32_t outputs) {
    ninfer::RequestOptions options;
    options.execution.requested_output_tokens = outputs;
    options.execution.sampling.temperature    = 0.0F;
    options.execution.allow_prefix_reuse      = false;
    options.stop.include_model_defaults       = false;
    return options;
}

ninfer::RequestOptions p_less_options(std::uint32_t outputs, std::uint64_t seed) {
    ninfer::RequestOptions options;
    options.execution.requested_output_tokens     = outputs;
    options.execution.sampling.temperature        = 2.0F;
    options.execution.sampling.top_k              = 1;
    options.execution.sampling.top_p              = 0.1F;
    options.execution.sampling.min_p              = 0.9F;
    options.execution.sampling.presence_penalty   = 2.0F;
    options.execution.sampling.frequency_penalty  = 2.0F;
    options.execution.sampling.seed               = seed;
    options.execution.sampling.p_less             = true;
    options.execution.allow_prefix_reuse          = false;
    options.stop.include_model_defaults           = false;
    return options;
}

ninfer::RequestOptions greedy_reuse(std::uint32_t outputs, bool reuse) {
    ninfer::RequestOptions options          = greedy_options(outputs);
    options.execution.allow_prefix_reuse    = reuse;
    return options;
}

ninfer::RequestOptions p_less_reuse(std::uint32_t outputs, std::uint64_t seed, bool reuse) {
    ninfer::RequestOptions options          = p_less_options(outputs, seed);
    options.execution.allow_prefix_reuse    = reuse;
    return options;
}

ninfer::RequestOptions p_less_chat(std::uint32_t outputs, std::uint64_t seed, bool reuse) {
    ninfer::RequestOptions options          = p_less_reuse(outputs, seed, reuse);
    options.stop.include_model_defaults     = true;
    return options;
}

ninfer::EngineOptions chat_dflash_options(const char* artifact) {
    ninfer::EngineOptions options =
        speculative_engine_options(artifact, ninfer::SpeculativeBackend::DFlash, 7, 1);
    options.max_context = 2048;
    options.kv_capacity = ninfer::KvCapacityPolicy::explicit_capacity(2048);
    options.prefill_chunk = 512;
    return options;
}

std::vector<ninfer::TokenId> resume_prefix(const std::vector<ninfer::TokenId>& keep,
                                           const std::vector<ninfer::TokenId>& generated) {
    std::vector<ninfer::TokenId> prefix = keep;
    if (!generated.empty()) {
        prefix.insert(prefix.end(), generated.begin(), generated.end() - 1);
    }
    return prefix;
}

void dump_tokens(const char* tag, const std::vector<ninfer::TokenId>& tokens) {
    std::cerr << tag << " [" << tokens.size() << "]:";
    for (ninfer::TokenId token : tokens) { std::cerr << ' ' << token; }
    std::cerr << '\n';
}

void dump_speculative(const char* tag, const ninfer::SpeculativeStats& stats) {
    std::cerr << tag << " rounds=" << stats.rounds << " drafted=" << stats.drafted_tokens
              << " accepted=" << stats.accepted_tokens << " fallback=" << stats.fallback_steps
              << " live_k=" << stats.live_draft_tokens << " per_pos=";
    for (std::uint64_t count : stats.accepted_per_position) { std::cerr << ' ' << count; }
    std::cerr << '\n';
}

void dump_c2_divergence(const char* label, const char* which,
                        const ninfer::GenerationResult& seq, const ninfer::GenerationResult& got) {
    const auto& want = seq.generated_token_ids;
    const auto& have = got.generated_token_ids;
    std::size_t index = 0;
    const std::size_t n = std::min(want.size(), have.size());
    for (; index < n && want[index] == have[index]; ++index) {}
    std::cerr << label << ' ' << which
              << " diverged from its sequential C=1 p-less oracle at generated index " << index
              << '\n';
    dump_tokens("  seq", want);
    dump_tokens("  c2", have);
    dump_speculative("  seq", seq.speculative);
    dump_speculative("  c2", got.speculative);
}

void dump_score(const char* tag, const ninfer::ScoreResult& score) {
    std::cerr << tag << " scored=" << score.tokens_scored << " non_finite=" << score.non_finite
              << " terrible=" << score.terrible_tokens << " mean_nll=" << score.mean_nll
              << " max_nll=" << score.max_nll << " ppl=" << score.perplexity << '\n';
    if (score.token_nlls.empty()) { return; }
    std::size_t worst = 0;
    for (std::size_t i = 1; i < score.token_nlls.size(); ++i) {
        if (score.token_nlls[i] > score.token_nlls[worst]) { worst = i; }
    }
    std::cerr << tag << " worst_at=" << worst << " nll=" << score.token_nlls[worst] << " nlls=";
    for (std::size_t i = 0; i < score.token_nlls.size(); ++i) {
        if (i > 0) { std::cerr << ','; }
        std::cerr << score.token_nlls[i];
    }
    std::cerr << '\n';
}

bool looks_like_counting_collapse(const std::string& text) {
    int run = 0;
    int best = 0;
    std::string tok;
    const auto flush = [&](bool integer) {
        if (integer && tok.size() >= 1 && tok.size() <= 4) {
            ++run;
            if (run > best) { best = run; }
        } else {
            run = 0;
        }
        tok.clear();
    };
    for (unsigned char c : text) {
        if (std::isdigit(c)) {
            tok.push_back(static_cast<char>(c));
            continue;
        }
        if (c == ' ' || c == '\n' || c == '\t') {
            flush(!tok.empty() &&
                  std::all_of(tok.begin(), tok.end(),
                              [](unsigned char ch) { return std::isdigit(ch) != 0; }));
            continue;
        }
        flush(false);
    }
    flush(!tok.empty());
    return best >= 20;
}

bool looks_like_word_salad(const std::string& text) {
    int run = 0;
    int best = 0;
    std::string tok;
    const auto flush = [&] {
        bool word = !tok.empty() && tok.size() <= 8;
        for (unsigned char ch : tok) {
            if (std::isalpha(ch) == 0) { word = false; }
        }
        if (word) {
            ++run;
            if (run > best) { best = run; }
        } else {
            run = 0;
        }
        tok.clear();
    };
    for (unsigned char c : text) {
        if (std::isalpha(c) != 0) {
            tok.push_back(static_cast<char>(c));
            continue;
        }
        flush();
    }
    flush();
    return best >= 24;
}

bool looks_like_role_token_loop(const std::vector<ninfer::TokenId>& ids) {
    int run = 0;
    int best = 0;
    for (ninfer::TokenId token : ids) {
        if (token == 248045 || token == 248046 || token == 198) {
            ++run;
            if (run > best) { best = run; }
        } else {
            run = 0;
        }
    }
    return best >= 24;
}

bool looks_like_punctuation_salad(const std::string& text) {
    if (text.size() < 80) { return false; }
    int punct = 0;
    for (unsigned char c : text) {
        if (std::isalnum(c) == 0 && std::isspace(c) == 0) { ++punct; }
    }
    return punct * 20 > static_cast<int>(text.size()) * 7;
}

bool looks_like_decoherence(const std::string& text) {
    return looks_like_counting_collapse(text) || looks_like_word_salad(text) ||
           looks_like_punctuation_salad(text);
}

const char* reuse_path_name(ninfer::PrefixReusePath path) {
    switch (path) {
    case ninfer::PrefixReusePath::FullReset:
        return "full_reset";
    case ninfer::PrefixReusePath::AppendAtFrontier:
        return "append_at_frontier";
    case ninfer::PrefixReusePath::RestoreTurnCheckpoint:
        return "restore_turn_checkpoint";
    case ninfer::PrefixReusePath::RestoreResponseCheckpoint:
        return "restore_response_checkpoint";
    case ninfer::PrefixReusePath::RestoreContextCheckpoint:
        return "restore_context_checkpoint";
    case ninfer::PrefixReusePath::RestoreTurnRollback:
        return "restore_turn_rollback";
    }
    return "unknown";
}

void dump_turn(const char* tag, const ninfer::GenerationResult& result) {
    std::cerr << tag << " finish=" << static_cast<int>(result.finish_reason)
              << " path=" << reuse_path_name(result.prefix_reuse_path)
              << " source=" << static_cast<int>(result.prefix_reuse_source)
              << " reused=" << result.reused_prompt_tokens
              << " prompt=" << result.prompt.prompt_tokens
              << " gen=" << result.generated_token_ids.size()
              << " reasoning_tokens=" << result.reasoning_tokens
              << " content_bytes=" << result.content.size()
              << " reasoning_bytes=" << result.reasoning.size() << '\n';
    dump_speculative(tag, result.speculative);
    if (!result.reasoning.empty()) { std::cerr << tag << "-reasoning: " << result.reasoning << '\n'; }
    std::cerr << tag << "-content: " << result.content << '\n';
}

ninfer::ChatMessage text_turn(ninfer::ChatRole role, std::string text) {
    ninfer::ChatMessage message;
    message.role = role;
    message.parts.push_back(ninfer::MessagePart{
        .kind = ninfer::MessagePartKind::Text, .text = std::move(text), .media = {}});
    return message;
}

ninfer::ChatMessage assistant_from_result(const ninfer::GenerationResult& result) {
    ninfer::ChatMessage message = text_turn(ninfer::ChatRole::Assistant, result.content);
    message.reasoning_content   = result.reasoning;
    return message;
}

int report_mismatch(const char* label, const ninfer::GenerationResult& result,
                    const std::vector<ninfer::TokenId>& want, const char* want_name) {
    const auto mismatch =
        std::mismatch(result.generated_token_ids.begin(), result.generated_token_ids.end(),
                      want.begin(), want.end());
    std::cerr << label << " diverged at "
              << static_cast<std::size_t>(mismatch.first - result.generated_token_ids.begin())
              << ": dflash="
              << (mismatch.first == result.generated_token_ids.end() ? -1 : *mismatch.first)
              << ' ' << want_name << '='
              << (mismatch.second == want.end() ? -1 : *mismatch.second) << '\n';
    dump_tokens("  got   ", result.generated_token_ids);
    std::cerr << "  " << want_name << " [" << want.size() << "]:";
    for (ninfer::TokenId token : want) { std::cerr << ' ' << token; }
    std::cerr << '\n';
    dump_speculative("  spec  ", result.speculative);
    return 1;
}

std::size_t match_prefix_length(const std::vector<ninfer::TokenId>& got,
                                const std::vector<ninfer::TokenId>& want) {
    const auto mismatch =
        std::mismatch(got.begin(), got.end(), want.begin(), want.end());
    return static_cast<std::size_t>(mismatch.first - got.begin());
}

bool relax_oracle() { return std::getenv("NINFER_DFLASH_TEST_RELAX_ORACLE") != nullptr; }

int check_speculative(const ninfer::GenerationResult& result, const char* label);
int fail_if_decohered(const char* label, const char* turn, const ninfer::GenerationResult& result);

enum class OracleKind { TargetOnly, StrictTargetOnly, C1DFlash };

// Target-only: packed/T=1 numerical identity. First token must agree. A later greedy
// flip fails the default run; NINFER_DFLASH_TEST_RELAX_ORACLE=1 prints it and continues.
// C=1 DFlash: overlapping C>1 must match sequential C=1 of the same k. Never relaxed —
// that comparison is packed-batch identity, not packed-versus-T=1 drift. Flattening NVFP4
// GDN conv-record to T=W*B compose (W4A4 GEMM + BF16 conv) flipped greedy col 0 vs
// C=1 fused SmallT+FP32; the Op guard is run_nvfp4_batched_matches_serial_fused.
int check_tokens(const char* label, const ninfer::GenerationResult& result,
                 const std::vector<ninfer::TokenId>& want, OracleKind kind) {
    const char* want_name = kind == OracleKind::C1DFlash ? "C=1 DFlash" : "target-only";
    if (check_speculative(result, label) != 0) { return 1; }
    if (result.generated_token_ids.size() != want.size()) {
        std::cerr << label << " generated " << result.generated_token_ids.size()
                  << " tokens, " << want_name << ' ' << want.size() << '\n';
        dump_tokens("  got   ", result.generated_token_ids);
        std::cerr << "  " << want_name << " [" << want.size() << "]:";
        for (ninfer::TokenId token : want) { std::cerr << ' ' << token; }
        std::cerr << '\n';
        return 1;
    }
    const std::size_t matched = match_prefix_length(result.generated_token_ids, want);
    if (matched < want.size()) { (void)report_mismatch(label, result, want, want_name); }
    if (matched == 0) {
        std::cerr << label << " first generated token disagrees with " << want_name << '\n';
        return 1;
    }
    if (result.speculative.drafted_tokens > 0 && result.speculative.accepted_tokens == 0) {
        std::cerr << label << " drafted tokens but accepted none\n";
        return 1;
    }
    const bool allow_relax = kind == OracleKind::TargetOnly && relax_oracle();
    if (matched != want.size() && !allow_relax) { return 1; }
    if (matched < want.size()) {
        std::cerr << label << " greedy match " << matched << '/' << want.size()
                  << " vs " << want_name
                  << " (NINFER_DFLASH_TEST_RELAX_ORACLE=1 continuing)\n";
    }
    return 0;
}

int check_dflash_load(const ninfer::Engine& engine) {
    const ninfer::LoadSummary load = engine.load_summary();
    if (load.target != "qwen3_8_27b" || load.weights_id != "nvfp4" ||
        load.host_to_device_bytes == 0) {
        std::cerr << "DFlash2 Engine materialized an invalid artifact payload: target="
                  << load.target << " weights=" << load.weights_id << '\n';
        return 1;
    }
    return 0;
}

int check_speculative(const ninfer::GenerationResult& result, const char* label) {
    if (result.speculative.backend != ninfer::SpeculativeBackend::DFlash ||
        result.speculative.rounds == 0) {
        std::cerr << label << " did not execute speculative decode\n";
        return 1;
    }
    return 0;
}

int exercise_dflash_vision(const char* artifact) {
    constexpr const char* label = "DFlash2 Vision p-less XAttention";
    const auto read_fixture = [](const char* relative) {
        const std::string path = std::string(NINFER_SOURCE_DIR) + relative;
        std::ifstream input(path, std::ios::binary);
        return std::vector<std::uint8_t>(std::istreambuf_iterator<char>(input), {});
    };
    const std::vector<std::uint8_t> image_bytes =
        read_fixture("/examples/cli/media/visual_chart.png");
    const std::vector<std::uint8_t> natural_bytes =
        read_fixture("/examples/cli/media/natural_scene.png");
    const std::vector<std::uint8_t> video_bytes =
        read_fixture("/examples/cli/media/temporal_events.mp4");
    if (image_bytes.empty() || natural_bytes.empty() || video_bytes.empty()) {
        std::cerr << label << " could not read the deterministic media fixtures\n";
        return 1;
    }
    const auto media_part = [](ninfer::MediaKind kind, const std::vector<std::uint8_t>& bytes,
                               std::string media_type, std::string name) {
        ninfer::MessagePart media;
        media.kind              = ninfer::MessagePartKind::Media;
        media.media.kind        = kind;
        media.media.bytes       = bytes;
        media.media.media_type  = std::move(media_type);
        media.media.source_name = std::move(name);
        return media;
    };

    const auto make_input = [&] {
        ninfer::ChatMessage message;
        message.role = ninfer::ChatRole::User;
        message.parts.push_back(
            media_part(ninfer::MediaKind::Image, image_bytes, "image/png", "visual_chart.png"));
        message.parts.push_back(ninfer::MessagePart{
            .kind = ninfer::MessagePartKind::Text,
            .text = "读取图中的标题，数出红色圆形，并判断蓝色方块在绿色三角形的哪一侧。"
                    "只输出：标题；数量；左侧或右侧。",
            .media = {}});
        ninfer::PromptInput input;
        input.messages.push_back(std::move(message));
        input.options.enable_thinking = false;
        return input;
    };
    const auto make_mixed_multiturn_input = [&] {
        ninfer::PromptInput input;
        ninfer::ChatMessage system;
        system.role = ninfer::ChatRole::System;
        system.parts.push_back(ninfer::MessagePart{
            .kind = ninfer::MessagePartKind::Text,
            .text = "准确保留不同轮次媒体中的可见事实。",
            .media = {}});
        input.messages.push_back(std::move(system));
        ninfer::ChatMessage image_turn;
        image_turn.role = ninfer::ChatRole::User;
        image_turn.parts.push_back(
            media_part(ninfer::MediaKind::Image, natural_bytes, "image/png", "natural_scene.png"));
        image_turn.parts.push_back(ninfer::MessagePart{
            .kind = ninfer::MessagePartKind::Text,
            .text = "请查看这幅场景，记住邮箱上的数字，暂时不要回答。",
            .media = {}});
        input.messages.push_back(std::move(image_turn));
        ninfer::ChatMessage assistant;
        assistant.role = ninfer::ChatRole::Assistant;
        assistant.parts.push_back(ninfer::MessagePart{
            .kind = ninfer::MessagePartKind::Text,
            .text = "好的，我会保留图像中的邮箱数字。",
            .media = {}});
        input.messages.push_back(std::move(assistant));
        ninfer::ChatMessage video_turn;
        video_turn.role = ninfer::ChatRole::User;
        video_turn.parts.push_back(media_part(ninfer::MediaKind::Video, video_bytes, "video/mp4",
                                              "temporal_events.mp4"));
        video_turn.parts.push_back(ninfer::MessagePart{
            .kind = ninfer::MessagePartKind::Text,
            .text = "把上一轮邮箱数字和视频末尾 END 数字用连字符连接。只输出连接结果。",
            .media = {}});
        input.messages.push_back(std::move(video_turn));
        input.options.enable_thinking = false;
        return input;
    };
    const auto make_text_input = [] {
        ninfer::ChatMessage message;
        message.role = ninfer::ChatRole::User;
        message.parts.push_back(ninfer::MessagePart{
            .kind = ninfer::MessagePartKind::Text,
            .text = "只输出：TEXT-42",
            .media = {}});
        ninfer::PromptInput input;
        input.messages.push_back(std::move(message));
        input.options.enable_thinking = false;
        return input;
    };

    ninfer::RequestOptions chart_greedy_options = greedy_options(24);
    chart_greedy_options.stop.include_model_defaults = true;
    ninfer::RequestOptions mixed_options = greedy_options(16);
    mixed_options.stop.include_model_defaults = true;
    std::vector<ninfer::TokenId> target_chart_tokens;
    std::vector<ninfer::TokenId> target_mixed_tokens;
    std::vector<ninfer::TokenId> target_text_tokens;
    {
        // The independent target-only Engine is deliberately destroyed before DFlash is loaded:
        // both resident models do not fit concurrently on the supported 32 GiB device.
        ninfer::EngineOptions target_options = base_engine_options(artifact);
        target_options.enable_vision          = true;
        target_options.max_context            = 2048;
        target_options.kv_capacity = ninfer::KvCapacityPolicy::explicit_capacity(2048);
        target_options.xattn_tau     = 0.9F;
        target_options.xattn_min_len = 0;
        if (std::getenv("NINFER_DFLASH_TEST_NO_GRAPH") != nullptr) {
            target_options.use_cuda_graph = false;
        }
        ninfer::Engine target(target_options);
        const ninfer::GenerationResult target_chart =
            target.generate(target.prepare(make_input()), chart_greedy_options);
        const ninfer::GenerationResult target_mixed =
            target.generate(target.prepare(make_mixed_multiturn_input()), mixed_options);
        const ninfer::GenerationResult target_text =
            target.generate(target.prepare(make_text_input()), greedy_options(5));
        if (!target_chart.prompt.has_media || !(target_chart.timings.vision_seconds > 0.0) ||
            target_chart.content.find("731；3；左侧") == std::string::npos ||
            !target_mixed.prompt.has_media || !(target_mixed.timings.vision_seconds > 0.0) ||
            target_mixed.content.find("24-9") == std::string::npos) {
            std::cerr << label << " target-only Vision oracle failed: chart='"
                      << target_chart.content << "' mixed='" << target_mixed.content << "'\n";
            return 1;
        }
        target_chart_tokens = target_chart.generated_token_ids;
        target_mixed_tokens = target_mixed.generated_token_ids;
        target_text_tokens  = target_text.generated_token_ids;
    }

    {
        // Use the native packed-tree route (k=7/W=12), not the simpler short-chain route.
        ninfer::EngineOptions options =
            speculative_engine_options(artifact, ninfer::SpeculativeBackend::DFlash, 7, 2);
        options.enable_vision = true;
        options.max_context   = 2048;
        options.kv_capacity   = ninfer::KvCapacityPolicy::explicit_capacity(2048);
        // Force the XAttention production route at this short fixture length. The public default
        // threshold remains 8192; this test is specifically its DFlash+Vision composition guard.
        options.xattn_tau     = 0.9F;
        options.xattn_min_len = 0;
        if (std::getenv("NINFER_DFLASH_TEST_NO_GRAPH") != nullptr) {
            options.use_cuda_graph = false;
        }
        ninfer::Engine engine(options);
        if (const int result = check_dflash_load(engine); result != 0) { return result; }

        const ninfer::GenerationResult greedy_chart =
            engine.generate(engine.prepare(make_input()), chart_greedy_options);
        if (check_tokens("DFlash2 Vision chart target parity", greedy_chart, target_chart_tokens,
                         OracleKind::StrictTargetOnly) != 0) {
            return 1;
        }

        ninfer::RequestOptions first_options = p_less_options(24, 12345);
        first_options.stop.include_model_defaults = true;
        const ninfer::GenerationResult result =
            engine.generate(engine.prepare(make_input()), first_options);
        if (!result.prompt.has_media || !(result.timings.vision_seconds > 0.0) ||
            check_speculative(result, label) != 0 || result.speculative.drafted_tokens == 0 ||
            result.speculative.accepted_tokens == 0 ||
            result.content.find("NIFER VISION 731；3；左侧") == std::string::npos) {
            std::cerr << label << " did not preserve the chart answer: content='" << result.content
                      << "' media=" << result.prompt.has_media
                      << " vision=" << result.timings.vision_seconds << '\n';
            dump_speculative("  spec", result.speculative);
            return 1;
        }

        ninfer::RequestOptions reuse_options = p_less_options(24, 12345);
        reuse_options.execution.allow_prefix_reuse = true;
        reuse_options.stop.include_model_defaults   = true;
        const ninfer::GenerationResult reused =
            engine.generate(engine.prepare(make_input()), reuse_options);
        if (reused.generated_token_ids != result.generated_token_ids ||
            reused.reused_prompt_tokens == 0 || reused.timings.vision_seconds != 0.0 ||
            check_speculative(reused, label) != 0 || reused.speculative.accepted_tokens == 0) {
            std::cerr << label << " did not restore the retained multimodal prefix: reused="
                      << reused.reused_prompt_tokens << " vision=" << reused.timings.vision_seconds
                      << '\n';
            dump_tokens("  first", result.generated_token_ids);
            dump_tokens("  reused", reused.generated_token_ids);
            return 1;
        }

        const ninfer::GenerationResult text_oracle =
            engine.generate(engine.prepare(make_text_input()), first_options);
        auto vision_handle = engine.submit(engine.prepare(make_input()), first_options);
        auto text_handle   = engine.submit(engine.prepare(make_text_input()), first_options);
        const ninfer::GenerationResult batched_vision = vision_handle.wait();
        const ninfer::GenerationResult batched_text   = text_handle.wait();
        if (batched_vision.generated_token_ids != result.generated_token_ids ||
            batched_text.generated_token_ids != text_oracle.generated_token_ids ||
            check_speculative(batched_vision, label) != 0 ||
            check_speculative(batched_text, label) != 0) {
            std::cerr
                << label
                << " mixed text/Vision batch changed a row relative to sequential execution\n";
            dump_tokens("  vision sequential", result.generated_token_ids);
            dump_tokens("  vision batched", batched_vision.generated_token_ids);
            dump_tokens("  text sequential", text_oracle.generated_token_ids);
            dump_tokens("  text batched", batched_text.generated_token_ids);
            return 1;
        }
        const ninfer::GenerationResult mixed =
            engine.generate(engine.prepare(make_mixed_multiturn_input()), mixed_options);
        if (!mixed.prompt.has_media || !(mixed.timings.vision_seconds > 0.0) ||
            mixed.content.find("24-9") == std::string::npos ||
            check_tokens("DFlash2 Vision mixed-media target parity", mixed, target_mixed_tokens,
                         OracleKind::StrictTargetOnly) != 0) {
            std::cerr << label << " failed the mixed multi-turn image/video oracle: content='"
                      << mixed.content << "'\n";
            return 1;
        }
        std::cout << "ok " << label << " content='" << result.content << "'\n" << std::flush;
    }

    {
        // Five requested tokens leave k=3 after prefill. Under adaptive N=7 the persistent
        // storage width is six, so W=4 exercises the compact strided-panel route at C=2.
        ninfer::EngineOptions options =
            adaptive_engine_options(artifact, ninfer::SpeculativeBackend::DFlash, 7, 2);
        options.enable_vision = true;
        options.max_context   = 2048;
        options.kv_capacity   = ninfer::KvCapacityPolicy::explicit_capacity(2048);
        options.xattn_tau     = 0.9F;
        options.xattn_min_len = 0;
        if (std::getenv("NINFER_DFLASH_TEST_NO_GRAPH") != nullptr) {
            options.use_cuda_graph = false;
        }
        ninfer::Engine engine(options);
        if (const int result = check_dflash_load(engine); result != 0) { return result; }
        const ninfer::RequestOptions short_options = greedy_options(5);
        const ninfer::RuntimeStats before = engine.runtime_stats();
        auto chart_handle = engine.submit(engine.prepare(make_input()), short_options);
        auto text_handle  = engine.submit(engine.prepare(make_text_input()), short_options);
        const ninfer::GenerationResult chart = chart_handle.wait();
        const ninfer::GenerationResult text  = text_handle.wait();
        (void)engine.memory_summary(); // Fence worker counter publication at this boundary.
        const ninfer::RuntimeStats after  = engine.runtime_stats();
        const std::uint64_t decode_rounds = after.decode_rounds - before.decode_rounds;
        const std::uint64_t row_rounds    = after.decode_row_rounds - before.decode_row_rounds;
        if (target_chart_tokens.size() < 5) {
            std::cerr << label << " target-only chart oracle is shorter than five tokens\n";
            return 1;
        }
        const std::vector<ninfer::TokenId> chart_oracle(target_chart_tokens.begin(),
                                                        target_chart_tokens.begin() + 5);
        if (target_text_tokens.size() != 5 ||
            check_tokens("adaptive compact Vision chart target parity", chart, chart_oracle,
                         OracleKind::StrictTargetOnly) != 0 ||
            check_tokens("adaptive compact text target parity", text, target_text_tokens,
                         OracleKind::StrictTargetOnly) != 0 ||
            chart.speculative.rounds_per_draft.size() <= 3 ||
            chart.speculative.rounds_per_draft[3] == 0 ||
            text.speculative.rounds_per_draft.size() <= 3 ||
            text.speculative.rounds_per_draft[3] == 0 || row_rounds <= decode_rounds) {
            std::cerr << label << " adaptive compact C=2 did not execute k=3/W=4\n";
            dump_speculative("  chart", chart.speculative);
            dump_speculative("  text", text.speculative);
            std::cerr << "  decode rounds=" << decode_rounds << " row rounds=" << row_rounds
                      << '\n';
            return 1;
        }
    }
    return 0;
}

// Regression: live serve uses k=5 W=6 (chain verify). Decode materializes frontier+W while
// entitlement used to omit W for non-tree mode, so short max_tokens on a page-aligned prompt
// threw "Paged KV materialize extent is outside entitlement" and poisoned the executor.
int exercise_chain_verify_short_output_entitlement(const char* artifact) {
    ninfer::EngineOptions options =
        speculative_engine_options(artifact, ninfer::SpeculativeBackend::DFlash, 5, 1);
    options.speculative.dflash_verify_width = 6;
    options.kv_cache                        = ninfer::KvCacheStorage::Nvfp4;
    options.max_context                     = 256;
    options.kv_capacity = ninfer::KvCapacityPolicy::explicit_capacity(256);
    ninfer::Engine engine(options);
    if (const int result = check_dflash_load(engine); result != 0) { return result; }

    // reserved_context = 60 + 5 - 1 = 64 (exact page). Without verify-width headroom,
    // first decode materialize(60+6) needs a second page and throws.
    std::vector<ninfer::TokenId> prompt(60, 198);
    prompt[0] = 248045;
    const ninfer::GenerationResult result =
        engine.generate(engine.prepare_tokens(prompt), greedy_options(5));
    if (result.finish_reason != ninfer::FinishReason::OutputLimit ||
        result.generated_token_ids.size() != 5) {
        std::cerr << "chain-verify short-output entitlement: finish="
                  << static_cast<int>(result.finish_reason)
                  << " gen=" << result.generated_token_ids.size() << '\n';
        return 1;
    }
    if (check_speculative(result, "chain-verify short-output entitlement") != 0) { return 1; }
    return 0;
}

int check_adaptive_dflash(const ninfer::GenerationResult& result, const char* label) {
    if (check_speculative(result, label) != 0) { return 1; }
    std::uint64_t hist = 0;
    for (std::uint64_t count : result.speculative.rounds_per_draft) { hist += count; }
    if (hist != result.speculative.rounds) {
        std::cerr << label << " rounds_per_draft sum " << hist << " != rounds "
                  << result.speculative.rounds << '\n';
        return 1;
    }
    const std::uint32_t live = result.speculative.live_draft_tokens;
    if (live != 3 && live != 4 && live != 5) {
        std::cerr << label << " live_draft_tokens " << live << " not in {3,4,5}\n";
        dump_speculative("  spec", result.speculative);
        return 1;
    }
    return 0;
}

int greedy_oracle(ninfer::Engine& engine, const std::vector<ninfer::TokenId>& prompt,
                  std::vector<ninfer::TokenId>& oracle, const char* label,
                  ninfer::GenerationResult* result_out = nullptr) {
    const ninfer::GenerationResult seq =
        engine.generate(engine.prepare_tokens(prompt), greedy_options(24));
    if (seq.generated_token_ids.size() != 24 || check_speculative(seq, label) != 0) {
        std::cerr << label << " C=1 oracle did not generate 24 tokens\n";
        return 1;
    }
    oracle = seq.generated_token_ids;
    if (result_out != nullptr) { *result_out = seq; }
    return 0;
}

int exercise_p_less_product_tree(ninfer::Engine& engine,
                                 const std::vector<ninfer::TokenId>& prompt) {
    constexpr const char* label = "DFlash2 p-less k=7 W=12";
    constexpr std::uint64_t seed = 0x123456789abcdef0ULL;
    const ninfer::GenerationResult first =
        engine.generate(engine.prepare_tokens(prompt), p_less_options(32, seed));
    const ninfer::GenerationResult replay =
        engine.generate(engine.prepare_tokens(prompt), p_less_options(32, seed));
    ninfer::RequestOptions top1_options = p_less_options(32, seed);
    top1_options.execution.sampling.p_less = false;
    const ninfer::GenerationResult top1 =
        engine.generate(engine.prepare_tokens(prompt), top1_options);
    if (first.finish_reason != ninfer::FinishReason::OutputLimit ||
        replay.finish_reason != ninfer::FinishReason::OutputLimit ||
        top1.finish_reason != ninfer::FinishReason::OutputLimit ||
        first.generated_token_ids.size() != 32 || replay.generated_token_ids.size() != 32 ||
        top1.generated_token_ids.size() != 32) {
        std::cerr << label << " did not complete seeded p-less/replay/top-1 requests\n";
        dump_tokens("  first", first.generated_token_ids);
        dump_tokens("  replay", replay.generated_token_ids);
        dump_tokens("  top-1", top1.generated_token_ids);
        return 1;
    }
    if (check_speculative(first, label) != 0 || check_speculative(replay, label) != 0 ||
        check_speculative(top1, label) != 0) {
        return 1;
    }
    if (first.speculative.live_draft_tokens != 7 || first.speculative.drafted_tokens == 0 ||
        first.speculative.accepted_tokens == 0) {
        std::cerr << label << " did not execute and accept from the native draft tree\n";
        dump_speculative("  spec", first.speculative);
        return 1;
    }
    if (first.generated_token_ids != replay.generated_token_ids ||
        first.speculative.rounds != replay.speculative.rounds ||
        first.speculative.drafted_tokens != replay.speculative.drafted_tokens ||
        first.speculative.accepted_tokens != replay.speculative.accepted_tokens) {
        std::cerr << label << " changed under identical seeded CUDA-graph replay\n";
        dump_tokens("  first", first.generated_token_ids);
        dump_tokens("  replay", replay.generated_token_ids);
        dump_speculative("  first", first.speculative);
        dump_speculative("  replay", replay.speculative);
        return 1;
    }
    if (first.generated_token_ids == top1.generated_token_ids) {
        std::cerr << label << " produced the top-1 control sequence; p-less was not observed\n";
        dump_tokens("  p-less", first.generated_token_ids);
        dump_tokens("  top-1", top1.generated_token_ids);
        return 1;
    }
    std::cout << "ok " << label << '\n' << std::flush;
    return 0;
}

int exercise_p_less_c2_matches_c1(const char* artifact,
                                  const std::array<std::vector<ninfer::TokenId>, 3>& prompts) {
    // Live OpenCode salad is one of two concurrent p-less tree decodes collapsing
    // into 248045/248046/198 while the other stays coherent. Greedy C=3 isolation
    // never enters the p-less mass-finalize walk, and B=2 GQA/path-select/accept
    // Op tests pass because they use peaked logits or numeric tolerances. C=2 vs
    // C=1 must match for the same seed and position RNG. Seed 0x9e3779b97f4a7c15
    // sits on that seam: C=1 continues a 248044 turn with 30097..., C=2 emits
    // role-token salad. Equal-length same prompt, graphs on or off, and submit
    // order do not remove the split.
    constexpr const char* label = "DFlash2 k=7 W=12 p-less C=2 matches C=1";
    constexpr std::uint64_t seed_a = 0x9e3779b97f4a7c15ULL;
    constexpr std::uint64_t seed_b = 0x123456789abcdef0ULL;
    constexpr std::uint32_t kTokens = 48;
    ninfer::EngineOptions options =
        speculative_engine_options(artifact, ninfer::SpeculativeBackend::DFlash, 7, 2);
    options.max_context = 2048;
    options.kv_capacity = ninfer::KvCapacityPolicy::explicit_capacity(2048);
    ninfer::Engine engine(options);

    auto extend = [](const std::vector<ninfer::TokenId>& base, std::size_t copies) {
        std::vector<ninfer::TokenId> out;
        out.reserve(base.size() * copies);
        for (std::size_t i = 0; i < copies; ++i) {
            out.insert(out.end(), base.begin(), base.end());
        }
        return out;
    };
    const std::vector<ninfer::TokenId> prompt_a = extend(prompts[0], 8);
    const std::vector<ninfer::TokenId> prompt_b = extend(prompts[0], 8);

    const ninfer::GenerationResult seq_a =
        engine.generate(engine.prepare_tokens(prompt_a), p_less_options(kTokens, seed_a));
    const ninfer::GenerationResult seq_b =
        engine.generate(engine.prepare_tokens(prompt_b), p_less_options(kTokens, seed_b));
    if (seq_a.generated_token_ids.size() != kTokens ||
        seq_b.generated_token_ids.size() != kTokens || check_speculative(seq_a, label) != 0 ||
        check_speculative(seq_b, label) != 0) {
        std::cerr << label << " sequential C=1 oracles did not complete\n";
        dump_speculative("  A", seq_a.speculative);
        dump_speculative("  B", seq_b.speculative);
        return 1;
    }

    auto handle_a =
        engine.submit(engine.prepare_tokens(prompt_a), p_less_options(kTokens, seed_a));
    auto handle_b =
        engine.submit(engine.prepare_tokens(prompt_b), p_less_options(kTokens, seed_b));
    ninfer::GenerationResult conc_a;
    ninfer::GenerationResult conc_b;
    try {
        conc_a = handle_a.wait();
        conc_b = handle_b.wait();
    } catch (const std::exception& error) {
        std::cerr << label << " concurrent p-less threw: " << error.what() << '\n';
        return 1;
    }
    if (conc_a.generated_token_ids.size() != kTokens ||
        conc_b.generated_token_ids.size() != kTokens || check_speculative(conc_a, label) != 0 ||
        check_speculative(conc_b, label) != 0) {
        std::cerr << label << " concurrent requests did not complete\n";
        dump_speculative("  A", conc_a.speculative);
        dump_speculative("  B", conc_b.speculative);
        return 1;
    }
    if (conc_a.generated_token_ids != seq_a.generated_token_ids) {
        dump_c2_divergence(label, "request A", seq_a, conc_a);
        return 1;
    }
    if (conc_b.generated_token_ids != seq_b.generated_token_ids) {
        dump_c2_divergence(label, "request B", seq_b, conc_b);
        return 1;
    }
    // Isolation contract is token identity with sequential C=1. This 48-token C=1
    // sample includes chat-template role tokens and Chinese punctuation, so the
    // decoherence heuristic is not applied after a match.
    std::cout << "ok " << label << '\n' << std::flush;
    return 0;
}

int exercise_p_less_c2_mixed_frontiers_matches_c1(
    const char* artifact, const std::array<std::vector<ninfer::TokenId>, 3>& prompts) {
    constexpr const char* label =
        "DFlash2 k=7 W=12 p-less C=2 mixed frontiers match C=1 without graphs";
    constexpr std::uint64_t seed_a = 0x9e3779b97f4a7c15ULL;
    constexpr std::uint64_t seed_b = 0x123456789abcdef0ULL;
    constexpr std::uint32_t kTokens = 24;
    ninfer::EngineOptions options =
        speculative_engine_options(artifact, ninfer::SpeculativeBackend::DFlash, 7, 2);
    options.max_context = 2048;
    options.kv_capacity = ninfer::KvCapacityPolicy::explicit_capacity(2048);
    if (options.use_cuda_graph) {
        std::cerr << label << " requires NINFER_DFLASH_TEST_NO_GRAPH\n";
        return 1;
    }
    ninfer::Engine engine(options);

    auto extend = [](const std::vector<ninfer::TokenId>& base, std::size_t copies) {
        std::vector<ninfer::TokenId> out;
        out.reserve(base.size() * copies);
        for (std::size_t i = 0; i < copies; ++i) {
            out.insert(out.end(), base.begin(), base.end());
        }
        return out;
    };
    // The short request remains below SWA's direct/split boundary for all 24 outputs,
    // while its peer starts above it. A shared maximum envelope would therefore run
    // the short C=2 proposal through split-KV instead of its sequential direct route.
    const std::vector<ninfer::TokenId> prompt_a = extend(prompts[0], 4);
    const std::vector<ninfer::TokenId> prompt_b = extend(prompts[1], 8);
    const ninfer::GenerationResult seq_a =
        engine.generate(engine.prepare_tokens(prompt_a), p_less_options(kTokens, seed_a));
    const ninfer::GenerationResult seq_b =
        engine.generate(engine.prepare_tokens(prompt_b), p_less_options(kTokens, seed_b));
    if (seq_a.generated_token_ids.size() != kTokens ||
        seq_b.generated_token_ids.size() != kTokens || check_speculative(seq_a, label) != 0 ||
        check_speculative(seq_b, label) != 0) {
        std::cerr << label << " sequential C=1 oracles did not complete\n";
        return 1;
    }

    auto handle_b =
        engine.submit(engine.prepare_tokens(prompt_b), p_less_options(kTokens, seed_b));
    auto handle_a =
        engine.submit(engine.prepare_tokens(prompt_a), p_less_options(kTokens, seed_a));
    ninfer::GenerationResult conc_a;
    ninfer::GenerationResult conc_b;
    try {
        conc_a = handle_a.wait();
        conc_b = handle_b.wait();
    } catch (const std::exception& error) {
        std::cerr << label << " concurrent p-less threw: " << error.what() << '\n';
        return 1;
    }
    if (conc_a.generated_token_ids.size() != kTokens ||
        conc_b.generated_token_ids.size() != kTokens || check_speculative(conc_a, label) != 0 ||
        check_speculative(conc_b, label) != 0) {
        std::cerr << label << " concurrent requests did not complete\n";
        return 1;
    }
    if (conc_a.generated_token_ids != seq_a.generated_token_ids) {
        dump_c2_divergence(label, "short request", seq_a, conc_a);
        return 1;
    }
    if (conc_b.generated_token_ids != seq_b.generated_token_ids) {
        dump_c2_divergence(label, "long request", seq_b, conc_b);
        return 1;
    }
    std::cout << "ok " << label << '\n' << std::flush;
    return 0;
}

int exercise_p_less_c2_long_context_matches_c1(
    const char* artifact, const std::array<std::vector<ninfer::TokenId>, 3>& prompts) {
    constexpr const char* label =
        "DFlash2 k=7 W=12 p-less C=2 long context matches C=1";
    constexpr std::uint64_t seed_a = 14784394741258868421ULL;
    constexpr std::uint64_t seed_b = 11406468648257731684ULL;
    constexpr std::size_t kPromptTokens = 39764;
    constexpr std::uint32_t kTokensA = 1024;
    constexpr std::uint32_t kTokensB = 128;
    ninfer::EngineOptions options =
        speculative_engine_options(artifact, ninfer::SpeculativeBackend::DFlash, 7, 2);
    options.max_context   = 65536;
    options.kv_capacity   = ninfer::KvCapacityPolicy::explicit_capacity(131072);
    options.prefill_chunk = 4096;
    ninfer::Engine engine(options);

    auto repeat_to = [](const std::vector<ninfer::TokenId>& base, std::size_t count) {
        std::vector<ninfer::TokenId> out;
        out.reserve(count);
        while (out.size() < count) {
            const std::size_t remaining = count - out.size();
            out.insert(out.end(), base.begin(),
                       base.begin() + static_cast<std::ptrdiff_t>(std::min(remaining, base.size())));
        }
        return out;
    };
    const std::vector<ninfer::TokenId> prompt_a = repeat_to(prompts[0], kPromptTokens);
    const std::vector<ninfer::TokenId> prompt_b = repeat_to(prompts[0], kPromptTokens);

    const ninfer::GenerationResult seq_a =
        engine.generate(engine.prepare_tokens(prompt_a), p_less_options(kTokensA, seed_a));
    const ninfer::GenerationResult seq_b =
        engine.generate(engine.prepare_tokens(prompt_b), p_less_options(kTokensB, seed_b));
    auto handle_a =
        engine.submit(engine.prepare_tokens(prompt_a), p_less_options(kTokensA, seed_a));
    auto handle_b =
        engine.submit(engine.prepare_tokens(prompt_b), p_less_options(kTokensB, seed_b));
    ninfer::GenerationResult conc_a;
    ninfer::GenerationResult conc_b;
    try {
        conc_a = handle_a.wait();
        conc_b = handle_b.wait();
    } catch (const std::exception& error) {
        std::cerr << label << " concurrent p-less threw: " << error.what() << '\n';
        return 1;
    }
    if (seq_a.generated_token_ids.size() != kTokensA ||
        seq_b.generated_token_ids.size() != kTokensB ||
        conc_a.generated_token_ids.size() != kTokensA ||
        conc_b.generated_token_ids.size() != kTokensB || check_speculative(seq_a, label) != 0 ||
        check_speculative(seq_b, label) != 0 || check_speculative(conc_a, label) != 0 ||
        check_speculative(conc_b, label) != 0) {
        std::cerr << label << " requests did not complete\n";
        return 1;
    }
    if (conc_a.generated_token_ids != seq_a.generated_token_ids) {
        dump_c2_divergence(label, "request A", seq_a, conc_a);
        return 1;
    }
    if (conc_b.generated_token_ids != seq_b.generated_token_ids) {
        dump_c2_divergence(label, "request B", seq_b, conc_b);
        return 1;
    }

    constexpr std::uint64_t reuse_seed = 2101703384980058981ULL;
    constexpr std::uint32_t kTurn1Tokens = 64;
    constexpr std::uint32_t kTurn2Tokens = 512;
    constexpr std::size_t kSuffixTokens = 5800;
    const auto make_follow = [&](const ninfer::GenerationResult& turn1) {
        std::vector<ninfer::TokenId> follow = prompt_b;
        follow.insert(follow.end(), turn1.generated_token_ids.begin(),
                      turn1.generated_token_ids.end());
        const std::vector<ninfer::TokenId> suffix = repeat_to(prompts[1], kSuffixTokens);
        follow.insert(follow.end(), suffix.begin(), suffix.end());
        return follow;
    };

    const ninfer::GenerationResult seq_turn1 = engine.generate(
        engine.prepare_tokens(prompt_b), p_less_reuse(kTurn1Tokens, reuse_seed, false));
    const std::vector<ninfer::TokenId> seq_follow = make_follow(seq_turn1);
    const ninfer::GenerationResult seq_turn2 =
        engine.generate(engine.prepare_tokens(seq_follow),
                        p_less_reuse(kTurn2Tokens, reuse_seed + 1, true));

    auto peer = engine.submit(engine.prepare_tokens(prompt_a), p_less_options(kTokensA, seed_a));
    const ninfer::GenerationResult conc_turn1 = engine.generate(
        engine.prepare_tokens(prompt_b), p_less_reuse(kTurn1Tokens, reuse_seed, false));
    const std::vector<ninfer::TokenId> conc_follow = make_follow(conc_turn1);
    auto follow =
        engine.submit(engine.prepare_tokens(conc_follow),
                      p_less_reuse(kTurn2Tokens, reuse_seed + 1, true));
    const ninfer::GenerationResult conc_peer  = peer.wait();
    const ninfer::GenerationResult conc_turn2 = follow.wait();
    if (seq_turn1.generated_token_ids.size() != kTurn1Tokens ||
        seq_turn2.generated_token_ids.size() != kTurn2Tokens ||
        conc_turn1.generated_token_ids.size() != kTurn1Tokens ||
        conc_turn2.generated_token_ids.size() != kTurn2Tokens ||
        conc_peer.generated_token_ids.size() != kTokensA ||
        seq_turn2.prefix_reuse_path == ninfer::PrefixReusePath::FullReset ||
        seq_turn2.reused_prompt_tokens == 0 ||
        conc_turn2.prefix_reuse_path == ninfer::PrefixReusePath::FullReset ||
        conc_turn2.reused_prompt_tokens == 0) {
        std::cerr << label << " long prefix-reuse seam did not complete\n";
        return 1;
    }
    if (conc_peer.generated_token_ids != seq_a.generated_token_ids) {
        dump_c2_divergence(label, "reuse peer", seq_a, conc_peer);
        return 1;
    }
    if (conc_turn1.generated_token_ids != seq_turn1.generated_token_ids) {
        dump_c2_divergence(label, "reuse turn 1", seq_turn1, conc_turn1);
        return 1;
    }
    if (conc_turn2.generated_token_ids != seq_turn2.generated_token_ids) {
        dump_c2_divergence(label, "reuse turn 2", seq_turn2, conc_turn2);
        return 1;
    }
    std::cout << "ok " << label << '\n' << std::flush;
    return 0;
}

int exercise_greedy_tree_matches_ordinary(const char* artifact,
                                          const std::vector<ninfer::TokenId>& prompt) {
    constexpr const char* label = "DFlash2 k=7 W=12 greedy matches ordinary";
    constexpr std::uint32_t kTokens = 64;
    std::vector<ninfer::TokenId> tree_tokens;
    std::vector<ninfer::TokenId> ordinary_tokens;
    {
        ninfer::EngineOptions options =
            speculative_engine_options(artifact, ninfer::SpeculativeBackend::DFlash, 7, 1);
        if (std::getenv("NINFER_DFLASH_TEST_CHAIN") != nullptr) {
            options.speculative.dflash_verify_width = 8;
        }
        ninfer::Engine engine(options);
        const ninfer::GenerationResult generated =
            engine.generate(engine.prepare_tokens(prompt), greedy_options(kTokens));
        if (generated.generated_token_ids.size() != kTokens ||
            generated.speculative.live_draft_tokens != 7 ||
            generated.speculative.drafted_tokens == 0) {
            std::cerr << label << " tree greedy did not complete on the native draft tree\n";
            dump_speculative("  spec", generated.speculative);
            return 1;
        }
        dump_speculative("  spec", generated.speculative);
        tree_tokens = generated.generated_token_ids;
        std::cerr << "  tree-text: " << generated.content << '\n';
    }
    {
        ninfer::Engine engine(base_engine_options(artifact));
        const ninfer::GenerationResult generated =
            engine.generate(engine.prepare_tokens(prompt), greedy_options(kTokens));
        if (generated.generated_token_ids.size() != kTokens) {
            std::cerr << label << " ordinary greedy did not complete\n";
            dump_tokens("  ordinary", generated.generated_token_ids);
            return 1;
        }
        ordinary_tokens = generated.generated_token_ids;
        std::cerr << "  ordinary-text: " << generated.content << '\n';
    }
    dump_tokens("  tree", tree_tokens);
    dump_tokens("  ordinary", ordinary_tokens);
    if (tree_tokens != ordinary_tokens) {
        const auto mismatch = std::mismatch(tree_tokens.begin(), tree_tokens.end(),
                                            ordinary_tokens.begin(), ordinary_tokens.end());
        std::cerr << label << " diverged at token "
                  << static_cast<std::size_t>(mismatch.first - tree_tokens.begin())
                  << " tree=" << (mismatch.first == tree_tokens.end() ? -1 : *mismatch.first)
                  << " ordinary="
                  << (mismatch.second == ordinary_tokens.end() ? -1 : *mismatch.second) << '\n';
        return 1;
    }
    std::cout << "ok " << label << '\n';
    return 0;
}

int exercise_p_less_tree_target_likelihood(const char* artifact,
                                           const std::vector<ninfer::TokenId>& prompt) {
    constexpr const char* label = "DFlash2 k=7 W=12 p-less target likelihood";
    constexpr std::uint64_t seed = 7632647173703958409ULL;
    constexpr std::uint32_t kTokens = 256;
    std::vector<ninfer::TokenId> tree_tokens;
    std::vector<ninfer::TokenId> ordinary_tokens;
    std::string tree_text;
    std::string ordinary_text;
    {
        ninfer::EngineOptions options =
            speculative_engine_options(artifact, ninfer::SpeculativeBackend::DFlash, 7, 1);
        if (std::getenv("NINFER_DFLASH_TEST_CHAIN") != nullptr) {
            options.speculative.dflash_verify_width = 8;
        }
        ninfer::Engine engine(options);
        const ninfer::GenerationResult generated =
            engine.generate(engine.prepare_tokens(prompt), p_less_options(kTokens, seed));
        if (generated.generated_token_ids.size() != kTokens ||
            generated.speculative.live_draft_tokens != 7 ||
            generated.speculative.drafted_tokens == 0) {
            std::cerr << label << " generation did not complete on the native draft tree\n";
            dump_speculative("  spec", generated.speculative);
            return 1;
        }
        dump_speculative("  spec", generated.speculative);
        tree_tokens = generated.generated_token_ids;
        tree_text   = generated.content;
    }
    {
        ninfer::Engine engine(base_engine_options(artifact));
        const ninfer::GenerationResult generated =
            engine.generate(engine.prepare_tokens(prompt), p_less_options(kTokens, seed));
        if (generated.generated_token_ids.size() != kTokens) {
            std::cerr << label << " ordinary p-less control did not complete\n";
            dump_tokens("  ordinary", generated.generated_token_ids);
            return 1;
        }
        ordinary_tokens = generated.generated_token_ids;
        ordinary_text   = generated.content;
    }
    if (tree_tokens.front() != ordinary_tokens.front()) {
        std::cerr << label << " hop 0 diverged from ordinary p-less sample: tree="
                  << tree_tokens.front() << " ordinary=" << ordinary_tokens.front() << '\n';
        dump_tokens("  tree", tree_tokens);
        dump_tokens("  ordinary", ordinary_tokens);
        return 1;
    }

    ninfer::Engine baseline(base_engine_options(artifact));
    ninfer::ScoreOptions score_options;
    score_options.schedule = ninfer::ScoreSchedule::Decode;
    score_options.skip_tokens = static_cast<std::uint32_t>(prompt.size() - 1);
    auto score_ids = [&](const std::vector<ninfer::TokenId>& generated) {
        std::vector<ninfer::TokenId> corpus = prompt;
        corpus.insert(corpus.end(), generated.begin(), generated.end());
        return baseline.score(baseline.prepare_tokens(std::move(corpus), false), score_options);
    };
    const ninfer::ScoreResult tree_score = score_ids(tree_tokens);
    const ninfer::ScoreResult ordinary_score = score_ids(ordinary_tokens);
    dump_score("  tree-nll", tree_score);
    dump_score("  ordinary-nll", ordinary_score);
    dump_tokens("  tree", tree_tokens);
    dump_tokens("  ordinary", ordinary_tokens);
    std::cerr << "  tree-text: " << tree_text << '\n';
    std::cerr << "  ordinary-text: " << ordinary_text << '\n';
    if (tree_score.tokens_scored != kTokens || tree_score.non_finite != 0) {
        std::cerr << label << " score did not cover the generated tree tokens\n";
        return 1;
    }
    if (looks_like_counting_collapse(tree_text)) {
        std::cerr << label << " collapsed into a counting run\n";
        return 1;
    }
    std::cout << "ok " << label << " mean_nll=" << tree_score.mean_nll
              << " max_nll=" << tree_score.max_nll
              << " ordinary_mean_nll=" << ordinary_score.mean_nll
              << " ordinary_max_nll=" << ordinary_score.max_nll << '\n';
    return 0;
}

int exercise_p_less_tree_commit_matches_reconstruction(
    const char* artifact, const std::vector<ninfer::TokenId>& prompt) {
    constexpr const char* label = "DFlash2 k=7 W=12 tree commit matches reconstruction";
    constexpr std::uint64_t seed = 7632647173703958409ULL;
    constexpr std::uint32_t kFirstTokens = 512;
    constexpr std::uint32_t kProbeTokens = 64;
    ninfer::EngineOptions options =
        speculative_engine_options(artifact, ninfer::SpeculativeBackend::DFlash, 7, 1);
    options.max_context = 2048;
    options.kv_capacity = ninfer::KvCapacityPolicy::explicit_capacity(2048);

    std::vector<ninfer::TokenId> history;
    ninfer::GenerationResult restored;
    {
        ninfer::Engine retained(options);
        const ninfer::GenerationResult first =
            retained.generate(retained.prepare_tokens(prompt), p_less_options(kFirstTokens, seed));
        if (first.generated_token_ids.size() != kFirstTokens ||
            first.speculative.accepted_tokens == 0) {
            std::cerr << label << " did not exercise an accepted tree path\n";
            dump_speculative("  first", first.speculative);
            return 1;
        }
        history = resume_prefix(prompt, first.generated_token_ids);
        restored =
            retained.generate(retained.prepare_tokens(history), greedy_reuse(kProbeTokens, true));
        if (restored.prefix_reuse_path == ninfer::PrefixReusePath::FullReset ||
            restored.reused_prompt_tokens == 0) {
            std::cerr << label << " did not restore the committed tree state\n";
            return 1;
        }
    }

    ninfer::Engine reconstructed(options);
    const ninfer::GenerationResult fresh = reconstructed.generate(
        reconstructed.prepare_tokens(history), greedy_reuse(kProbeTokens, false));
    if (restored.generated_token_ids != fresh.generated_token_ids) {
        dump_c2_divergence(label, "restored state", fresh, restored);
        return 1;
    }
    std::cout << "ok " << label << " reuse="
              << static_cast<int>(restored.prefix_reuse_path)
              << " reused=" << restored.reused_prompt_tokens << '\n';
    return 0;
}

int exercise_p_less_tree_chat_coherence(const char* artifact) {
    constexpr const char* label = "DFlash2 k=7 W=12 p-less chat coherence";
    constexpr std::uint64_t seed = 0x9e3779b97f4a7c15ULL;
    constexpr std::uint32_t kTokens = 192;
    ninfer::EngineOptions options =
        speculative_engine_options(artifact, ninfer::SpeculativeBackend::DFlash, 7, 1);
    options.max_context = 2048;
    options.kv_capacity = ninfer::KvCapacityPolicy::explicit_capacity(2048);
    ninfer::Engine engine(options);

    ninfer::PromptInput first;
    first.options.enable_thinking = false;
    first.messages.push_back(text_turn(
        ninfer::ChatRole::User,
        "Write one shell command that prints the first 25 git commits and the first 40 lines of "
        "README.md. Then explain in five sentences what that output is for."));
    const ninfer::GenerationResult turn1 =
        engine.generate(engine.prepare(first), p_less_options(kTokens, seed));
    std::cerr << label << " turn1 finish=" << static_cast<int>(turn1.finish_reason)
              << " gen=" << turn1.generated_token_ids.size()
              << " content_bytes=" << turn1.content.size() << '\n';
    dump_speculative("  turn1-spec", turn1.speculative);
    std::cerr << "  turn1-text: " << turn1.content << '\n';
    if (turn1.generated_token_ids.size() != kTokens) {
        std::cerr << label << " turn 1 did not complete\n";
        return 1;
    }
    if (looks_like_counting_collapse(turn1.content)) {
        std::cerr << label << " turn 1 collapsed into a counting run\n";
        return 1;
    }

    ninfer::PromptInput second = first;
    ninfer::ChatMessage assistant = text_turn(ninfer::ChatRole::Assistant, turn1.content);
    second.messages.push_back(std::move(assistant));
    second.messages.push_back(text_turn(
        ninfer::ChatRole::User,
        "Show first commits and README intro. Keep using complete sentences."));
    ninfer::RequestOptions turn2_opts = p_less_options(kTokens, seed + 1);
    turn2_opts.execution.allow_prefix_reuse = true;
    const ninfer::GenerationResult turn2 =
        engine.generate(engine.prepare(second), turn2_opts);
    std::cerr << label << " turn2 finish=" << static_cast<int>(turn2.finish_reason)
              << " gen=" << turn2.generated_token_ids.size()
              << " content_bytes=" << turn2.content.size() << '\n';
    dump_speculative("  turn2-spec", turn2.speculative);
    std::cerr << "  turn2-text: " << turn2.content << '\n';
    if (turn2.generated_token_ids.size() != kTokens) {
        std::cerr << label << " turn 2 did not complete\n";
        return 1;
    }
    if (looks_like_counting_collapse(turn2.content) ||
        looks_like_counting_collapse(turn1.content + " " + turn2.content)) {
        std::cerr << label << " turn 2 collapsed into a counting run\n";
        return 1;
    }
    std::cout << "ok " << label << '\n';
    return 0;
}

int fail_if_decohered(const char* label, const char* turn, const ninfer::GenerationResult& result) {
    const std::string visible = result.reasoning + "\n" + result.content;
    if (!looks_like_decoherence(visible) && !looks_like_role_token_loop(result.generated_token_ids)) {
        return 0;
    }
    std::cerr << label << " " << turn
              << " decohered (counting, word salad, or im_start/im_end loop)\n";
    dump_turn(turn, result);
    dump_tokens("  tokens", result.generated_token_ids);
    return 1;
}

int check_reuse_hit(const char* label, const ninfer::GenerationResult& result) {
    if (result.prefix_reuse_path == ninfer::PrefixReusePath::FullReset ||
        result.reused_prompt_tokens == 0) {
        std::cerr << label << " follow-up FullReset; did not hit a prefix-reuse seam\n";
        dump_turn("  follow-up", result);
        return 1;
    }
    return 0;
}

int check_reuse_hop0_vs_reset(ninfer::Engine& engine, const ninfer::PromptInput& followup,
                              const ninfer::GenerationResult& reused, std::uint64_t seed,
                              const char* label) {
    const ninfer::GenerationResult reset =
        engine.generate(engine.prepare(followup), p_less_chat(8, seed, false));
    dump_turn("  reset", reset);
    if (reused.generated_token_ids.empty() || reset.generated_token_ids.empty()) {
        std::cerr << label << " missing hop 0 tokens for reuse vs full-reset\n";
        return 1;
    }
    if (reused.generated_token_ids.front() != reset.generated_token_ids.front()) {
        std::cerr << label << " reuse hop 0 diverged from DFlash full-reset hop 0: reuse="
                  << reused.generated_token_ids.front() << " reset="
                  << reset.generated_token_ids.front() << " path="
                  << reuse_path_name(reused.prefix_reuse_path)
                  << " reused=" << reused.reused_prompt_tokens << '\n';
        dump_tokens("  reuse", reused.generated_token_ids);
        dump_tokens("  reset", reset.generated_token_ids);
        return 1;
    }
    return 0;
}

int exercise_p_less_thinking_followup(const char* artifact) {
    constexpr const char* label = "DFlash2 k=7 p-less thinking follow-up";
    constexpr std::uint64_t seed = 0x9e3779b97f4a7c15ULL;
    constexpr std::uint32_t kTurn1 = 96;
    constexpr std::uint32_t kTurn2 = 192;
    ninfer::Engine engine(chat_dflash_options(artifact));

    ninfer::PromptInput first;
    first.options.enable_thinking   = true;
    first.options.preserve_thinking = false;
    first.messages.push_back(text_turn(
        ninfer::ChatRole::User,
        "Write one shell command that prints the first 25 git commits and the first 40 lines of "
        "README.md. Then explain in five sentences what that output is for."));
    const ninfer::GenerationResult turn1 =
        engine.generate(engine.prepare(first), p_less_chat(kTurn1, seed, false));
    dump_turn("  turn1", turn1);
    if (turn1.generated_token_ids.size() < 16 || turn1.speculative.rounds == 0) {
        std::cerr << label << " turn 1 did not run DFlash decode\n";
        return 1;
    }
    if (fail_if_decohered(label, "turn1", turn1) != 0) { return 1; }

    ninfer::PromptInput second = first;
    second.messages.push_back(assistant_from_result(turn1));
    second.messages.push_back(text_turn(
        ninfer::ChatRole::User,
        "Show first commits and README intro. Keep using complete sentences."));
    const ninfer::GenerationResult turn2 =
        engine.generate(engine.prepare(second), p_less_chat(kTurn2, seed + 1, true));
    dump_turn("  turn2", turn2);
    if (turn2.generated_token_ids.empty()) {
        std::cerr << label << " turn 2 produced no tokens\n";
        return 1;
    }
    if (check_reuse_hit(label, turn2) != 0) { return 1; }
    int failed = fail_if_decohered(label, "turn2", turn2);
    if (check_reuse_hop0_vs_reset(engine, second, turn2, seed + 1, label) != 0) { failed = 1; }
    if (failed != 0) { return failed; }
    std::cout << "ok " << label << " path=" << reuse_path_name(turn2.prefix_reuse_path)
              << " reused=" << turn2.reused_prompt_tokens << '\n';
    return 0;
}

int exercise_p_less_finished_thinking_followup(const char* artifact) {
    constexpr const char* label = "DFlash2 k=7 p-less finished-thinking follow-up";
    constexpr std::uint64_t seed = 0xa5a5a5a5a5a5a5a5ULL;
    ninfer::Engine engine(chat_dflash_options(artifact));

    ninfer::PromptInput first;
    first.options.enable_thinking   = true;
    first.options.preserve_thinking = false;
    first.messages.push_back(text_turn(ninfer::ChatRole::User, "Reply with only the word hello."));
    const ninfer::GenerationResult turn1 =
        engine.generate(engine.prepare(first), p_less_chat(256, seed, false));
    dump_turn("  turn1", turn1);
    if (turn1.generated_token_ids.size() < 8 || turn1.speculative.rounds == 0) {
        std::cerr << label << " turn 1 did not run DFlash decode\n";
        return 1;
    }
    if (fail_if_decohered(label, "turn1", turn1) != 0) { return 1; }

    ninfer::PromptInput second = first;
    second.messages.push_back(assistant_from_result(turn1));
    second.messages.push_back(
        text_turn(ninfer::ChatRole::User, "Now reply with only the word bonjour."));
    const ninfer::GenerationResult turn2 =
        engine.generate(engine.prepare(second), p_less_chat(192, seed + 1, true));
    dump_turn("  turn2", turn2);
    if (turn2.generated_token_ids.empty()) {
        std::cerr << label << " turn 2 produced no tokens\n";
        return 1;
    }
    if (check_reuse_hit(label, turn2) != 0) { return 1; }
    if (fail_if_decohered(label, "turn2", turn2) != 0) { return 1; }
    std::cout << "ok " << label << " path=" << reuse_path_name(turn2.prefix_reuse_path)
              << " reused=" << turn2.reused_prompt_tokens
              << " turn1_content_bytes=" << turn1.content.size()
              << " turn2_content_bytes=" << turn2.content.size() << '\n';
    return 0;
}

ninfer::PromptInput tool_loop_prompt(int completed_responses, bool preserve_thinking,
                                     bool enable_thinking) {
    auto assistant_call = [](std::string reasoning, std::string id, std::string key) {
        ninfer::ChatMessage message = text_turn(ninfer::ChatRole::Assistant, "");
        message.reasoning_content   = std::move(reasoning);
        message.tool_calls.push_back(ninfer::ToolCall{
            .id = std::move(id), .name = "lookup", .arguments_json = "{\"key\":\"" + key + "\"}"});
        return message;
    };
    ninfer::PromptInput input;
    input.messages.push_back(text_turn(
        ninfer::ChatRole::User,
        "Use the lookup tool to fetch key alpha, then summarize the value in one sentence."));
    if (completed_responses >= 1) {
        input.messages.push_back(
            assistant_call("Need the alpha record before answering.", "call_alpha", "alpha"));
        ninfer::ChatMessage tool =
            text_turn(ninfer::ChatRole::Tool, "{\"value\":17,\"next\":\"beta\"}");
        tool.tool_call_id = "call_alpha";
        input.messages.push_back(std::move(tool));
    }
    if (completed_responses >= 2) {
        input.messages.push_back(
            assistant_call("Alpha requested beta. Looking that up next.", "call_beta", "beta"));
        ninfer::ChatMessage tool = text_turn(ninfer::ChatRole::Tool, "{\"value\":25}");
        tool.tool_call_id        = "call_beta";
        input.messages.push_back(std::move(tool));
    }
    input.options.enable_thinking   = enable_thinking;
    input.options.preserve_thinking = preserve_thinking;
    input.options.tool_jsons.push_back(
        R"({"type":"function","function":{"name":"lookup","parameters":{"type":"object","properties":{"key":{"type":"string"}},"required":["key"]}}})");
    return input;
}

// Live OpenCode: one slot keeps generating while the other prefills a tool-loop
// follow-up (prefix reuse) and then both decode at C=2. Greedy C=3 isolation
// never takes that seam, and the C=2 p-less match above uses two cold FullReset
// prefills with reuse disabled.
int exercise_p_less_c2_reuse_during_peer_decode(const char* artifact) {
    constexpr const char* label = "DFlash2 k=7 p-less C=2 reuse during peer decode";
    constexpr std::uint64_t seed_a = 0xa5a5a5a5a5a5a5a5ULL;
    constexpr std::uint64_t seed_b = 0x243f6a8885a308d3ULL;
    constexpr std::uint32_t kPeer  = 96;
    constexpr std::uint32_t kTurn1 = 16;
    constexpr std::uint32_t kTurn2 = 48;
    ninfer::EngineOptions options =
        speculative_engine_options(artifact, ninfer::SpeculativeBackend::DFlash, 7, 2);
    options.max_context = 2048;
    options.kv_capacity = ninfer::KvCapacityPolicy::explicit_capacity(2048);
    ninfer::Engine engine(options);

    ninfer::PromptInput story;
    story.options.enable_thinking = true;
    story.messages.push_back(text_turn(
        ninfer::ChatRole::User,
        "Write a long numbered list of repository subsystems. Keep going until the token budget."));

    const ninfer::PromptInput first  = tool_loop_prompt(0, false, true);
    const ninfer::PromptInput follow = tool_loop_prompt(1, false, true);

    const ninfer::GenerationResult seq_a =
        engine.generate(engine.prepare(story), p_less_options(kPeer, seed_a));
    const ninfer::GenerationResult seq_b1 =
        engine.generate(engine.prepare(first), p_less_chat(kTurn1, seed_b, false));
    const ninfer::GenerationResult seq_b2 =
        engine.generate(engine.prepare(follow), p_less_chat(kTurn2, seed_b + 1, true));
    if (seq_a.generated_token_ids.size() != kPeer || seq_b1.generated_token_ids.size() < 8 ||
        seq_b2.generated_token_ids.empty() || check_speculative(seq_a, label) != 0 ||
        check_reuse_hit(label, seq_b2) != 0) {
        std::cerr << label << " sequential oracles did not complete a reuse seam\n";
        dump_speculative("  A", seq_a.speculative);
        dump_turn("  B1", seq_b1);
        dump_turn("  B2", seq_b2);
        return 1;
    }

    auto handle_a = engine.submit(engine.prepare(story), p_less_options(kPeer, seed_a));
    ninfer::GenerationResult conc_b1;
    try {
        conc_b1 = engine.generate(engine.prepare(first), p_less_chat(kTurn1, seed_b, false));
    } catch (const std::exception& error) {
        std::cerr << label << " concurrent B turn1 threw: " << error.what() << '\n';
        return 1;
    }
    auto handle_b2 = engine.submit(engine.prepare(follow), p_less_chat(kTurn2, seed_b + 1, true));
    ninfer::GenerationResult conc_a;
    ninfer::GenerationResult conc_b2;
    try {
        conc_a  = handle_a.wait();
        conc_b2 = handle_b2.wait();
    } catch (const std::exception& error) {
        std::cerr << label << " concurrent wait threw: " << error.what() << '\n';
        return 1;
    }
    dump_turn("  concurrent-B2", conc_b2);
    if (conc_a.generated_token_ids.size() != kPeer || check_speculative(conc_a, label) != 0) {
        std::cerr << label << " concurrent peer A did not complete\n";
        dump_speculative("  A", conc_a.speculative);
        return 1;
    }
    if (conc_b2.generated_token_ids.empty() || check_reuse_hit(label, conc_b2) != 0) {
        std::cerr << label << " concurrent follow-up missed prefix reuse\n";
        return 1;
    }
    if (conc_a.generated_token_ids != seq_a.generated_token_ids) {
        std::cerr << label << " peer A diverged from its sequential p-less oracle\n";
        dump_tokens("  seq", seq_a.generated_token_ids);
        dump_tokens("  c2", conc_a.generated_token_ids);
        dump_speculative("  seq", seq_a.speculative);
        dump_speculative("  c2", conc_a.speculative);
        return 1;
    }
    if (conc_b1.generated_token_ids != seq_b1.generated_token_ids) {
        std::cerr << label << " B turn1 diverged from its sequential p-less oracle\n";
        dump_tokens("  seq", seq_b1.generated_token_ids);
        dump_tokens("  c2", conc_b1.generated_token_ids);
        return 1;
    }
    if (conc_b2.generated_token_ids != seq_b2.generated_token_ids) {
        std::cerr << label << " reused follow-up diverged from its sequential p-less oracle\n";
        dump_tokens("  seq", seq_b2.generated_token_ids);
        dump_tokens("  c2", conc_b2.generated_token_ids);
        dump_speculative("  seq", seq_b2.speculative);
        dump_speculative("  c2", conc_b2.speculative);
        return 1;
    }
    std::cout << "ok " << label << " reuse=" << reuse_path_name(conc_b2.prefix_reuse_path)
              << " reused=" << conc_b2.reused_prompt_tokens << '\n'
              << std::flush;
    return 0;
}

int check_tool_rewrite_hop0_greedy(ninfer::Engine& engine) {
    const bool variants[] = {false, true};
    for (bool preserve_thinking : variants) {
        const std::string label =
            std::string("DFlash2 k=7 greedy tool-loop hop0 preserve_thinking=") +
            (preserve_thinking ? "true" : "false");
        const ninfer::PromptInput first = tool_loop_prompt(0, preserve_thinking, true);
        const ninfer::PromptInput followup = tool_loop_prompt(1, preserve_thinking, true);
        const ninfer::GenerationResult turn1 =
            engine.generate(engine.prepare(first), greedy_reuse(16, false));
        dump_turn("  greedy-turn1", turn1);
        if (turn1.generated_token_ids.size() < 8 || turn1.speculative.rounds == 0) {
            std::cerr << label << " source request did not run DFlash decode\n";
            return 1;
        }
        const ninfer::GenerationResult reused =
            engine.generate(engine.prepare(followup), greedy_reuse(8, true));
        dump_turn("  greedy-reuse", reused);
        if (check_reuse_hit(label.c_str(), reused) != 0) { return 1; }
        if (reused.generated_token_ids.empty()) {
            std::cerr << label << " reuse produced no tokens\n";
            return 1;
        }
        const ninfer::GenerationResult reset =
            engine.generate(engine.prepare(followup), greedy_reuse(8, false));
        dump_turn("  greedy-reset", reset);
        if (reset.generated_token_ids.empty() ||
            reused.generated_token_ids.front() != reset.generated_token_ids.front()) {
            std::cerr << label << " reuse hop 0 diverged from DFlash full-reset hop 0: reuse="
                      << (reused.generated_token_ids.empty() ? -1
                                                             : reused.generated_token_ids.front())
                      << " reset="
                      << (reset.generated_token_ids.empty() ? -1 : reset.generated_token_ids.front())
                      << " path=" << reuse_path_name(reused.prefix_reuse_path)
                      << " reused=" << reused.reused_prompt_tokens << '\n';
            dump_tokens("  reuse", reused.generated_token_ids);
            dump_tokens("  reset", reset.generated_token_ids);
            return 1;
        }
        std::cout << "ok " << label << " path=" << reuse_path_name(reused.prefix_reuse_path)
                  << " hop0=" << reused.generated_token_ids.front() << '\n';
    }
    return 0;
}

int exercise_p_less_tool_history_reuse(const char* artifact) {
    constexpr std::uint64_t seed = 0x243f6a8885a308d3ULL;
    constexpr std::uint32_t kTurn2 = 192;
    ninfer::Engine engine(chat_dflash_options(artifact));
    if (const int result = check_tool_rewrite_hop0_greedy(engine); result != 0) { return result; }

    const bool variants[][2] = {{false, true}, {true, true}};
    for (const auto& variant : variants) {
        const bool preserve_thinking = variant[0];
        const bool enable_thinking   = variant[1];
        const std::string label =
            std::string("DFlash2 k=7 p-less tool-loop reuse preserve_thinking=") +
            (preserve_thinking ? "true" : "false");
        const std::uint32_t turn1_tokens = preserve_thinking ? 64 : 192;
        const ninfer::PromptInput first = tool_loop_prompt(0, preserve_thinking, enable_thinking);
        const ninfer::GenerationResult turn1 =
            engine.generate(engine.prepare(first), p_less_chat(turn1_tokens, seed, false));
        dump_turn("  turn1", turn1);
        if (turn1.generated_token_ids.size() < 8 || turn1.speculative.rounds == 0) {
            std::cerr << label << " source request did not run DFlash decode\n";
            return 1;
        }
        if (fail_if_decohered(label.c_str(), "turn1", turn1) != 0) { return 1; }

        const ninfer::PromptInput followup =
            tool_loop_prompt(1, preserve_thinking, enable_thinking);
        const ninfer::GenerationResult turn2 =
            engine.generate(engine.prepare(followup), p_less_chat(kTurn2, seed + 1, true));
        dump_turn("  turn2", turn2);
        if (turn2.generated_token_ids.empty()) {
            std::cerr << label << " tool follow-up produced no tokens\n";
            return 1;
        }
        if (check_reuse_hit(label.c_str(), turn2) != 0) { return 1; }
        int failed = fail_if_decohered(label.c_str(), "turn2", turn2);

        const ninfer::PromptInput second_loop =
            tool_loop_prompt(2, preserve_thinking, enable_thinking);
        const ninfer::GenerationResult turn3 =
            engine.generate(engine.prepare(second_loop), p_less_chat(kTurn2, seed + 2, true));
        dump_turn("  turn3", turn3);
        if (turn3.generated_token_ids.empty()) {
            std::cerr << label << " second tool follow-up produced no tokens\n";
            return 1;
        }
        if (check_reuse_hit(label.c_str(), turn3) != 0) { return 1; }
        // preserve_thinking=false captures TurnClosure at the first assistant start, so both
        // tool follow-ups restore the same base. ResponseReplay (preserve_thinking=true)
        // must walk forward with the rendered history.
        if (preserve_thinking &&
            (turn2.prefix_reuse_path != ninfer::PrefixReusePath::RestoreResponseCheckpoint ||
             turn3.prefix_reuse_path != ninfer::PrefixReusePath::RestoreResponseCheckpoint ||
             turn3.reused_prompt_tokens <= turn2.reused_prompt_tokens)) {
            std::cerr << label << " response checkpoint did not advance across the tool loop: first="
                      << turn2.reused_prompt_tokens << " path="
                      << reuse_path_name(turn2.prefix_reuse_path)
                      << " second=" << turn3.reused_prompt_tokens << " path="
                      << reuse_path_name(turn3.prefix_reuse_path) << '\n';
            failed = 1;
        }
        failed |= fail_if_decohered(label.c_str(), "turn3", turn3);
        // Greedy hop0 already checks rewrite-restore logits. p-less hop0 vs a later
        // FullReset is too sensitive: T=2 can flip when argmax agrees (see greedy hop0=760).
        if (failed != 0) { return failed; }
        std::cout << "ok " << label << " path=" << reuse_path_name(turn2.prefix_reuse_path)
                  << " reused=" << turn2.reused_prompt_tokens << "->" << turn3.reused_prompt_tokens
                  << '\n';
    }
    return 0;
}

int exercise_p_less_target_likelihood(const char* artifact,
                                      const std::vector<ninfer::TokenId>& prompt) {
    constexpr const char* label = "DFlash2 adaptive p-less target likelihood";
    constexpr std::uint64_t seed = 15446143373561885318ULL;
    constexpr std::uint32_t kTokens = 64;
    std::vector<ninfer::TokenId> dflash_tokens;
    std::vector<ninfer::TokenId> ordinary_tokens;
    {
        ninfer::Engine engine(adaptive_engine_options(
            artifact, ninfer::SpeculativeBackend::DFlash, 7, 1));
        const ninfer::GenerationResult generated =
            engine.generate(engine.prepare_tokens(prompt), p_less_options(kTokens, seed));
        if (generated.generated_token_ids.size() != kTokens ||
            check_adaptive_dflash(generated, label) != 0) {
            std::cerr << label << " generation did not complete\n";
            return 1;
        }
        dump_speculative("  spec", generated.speculative);
        dflash_tokens = generated.generated_token_ids;
    }
    {
        ninfer::Engine engine(base_engine_options(artifact));
        const ninfer::GenerationResult generated =
            engine.generate(engine.prepare_tokens(prompt), p_less_options(kTokens, seed));
        if (generated.generated_token_ids.size() != kTokens) {
            std::cerr << label << " ordinary p-less control did not complete\n";
            return 1;
        }
        ordinary_tokens = generated.generated_token_ids;
    }
    if (dflash_tokens.front() != ordinary_tokens.front()) {
        std::cerr << label << " hop 0 diverged from ordinary p-less sample: dflash="
                  << dflash_tokens.front() << " ordinary=" << ordinary_tokens.front() << '\n';
        dump_tokens("  dflash", dflash_tokens);
        dump_tokens("  ordinary", ordinary_tokens);
        return 1;
    }

    ninfer::Engine baseline(base_engine_options(artifact));
    ninfer::ScoreOptions score_options;
    score_options.schedule = ninfer::ScoreSchedule::Decode;
    score_options.skip_tokens = static_cast<std::uint32_t>(prompt.size() - 1);
    auto score_ids = [&](const std::vector<ninfer::TokenId>& generated) {
        std::vector<ninfer::TokenId> corpus = prompt;
        corpus.insert(corpus.end(), generated.begin(), generated.end());
        return baseline.score(baseline.prepare_tokens(std::move(corpus), false), score_options);
    };
    const ninfer::ScoreResult dflash_score = score_ids(dflash_tokens);
    const ninfer::ScoreResult ordinary_score = score_ids(ordinary_tokens);
    dump_score("  dflash-nll", dflash_score);
    dump_score("  ordinary-nll", ordinary_score);
    dump_tokens("  dflash", dflash_tokens);
    dump_tokens("  ordinary", ordinary_tokens);
    if (dflash_score.tokens_scored != kTokens || dflash_score.non_finite != 0 ||
        dflash_score.mean_nll > ordinary_score.mean_nll + 0.35 ||
        dflash_score.max_nll > ordinary_score.max_nll + 2.0 ||
        dflash_score.terrible_tokens > ordinary_score.terrible_tokens) {
        std::cerr << label << " drifted from ordinary p-less samples of the same seed\n";
        return 1;
    }
    std::cout << "ok " << label << " mean_nll=" << dflash_score.mean_nll
              << " max_nll=" << dflash_score.max_nll
              << " ordinary_mean_nll=" << ordinary_score.mean_nll
              << " ordinary_max_nll=" << ordinary_score.max_nll << '\n';
    return 0;
}

std::vector<ninfer::TokenId> token_prefix(const std::vector<ninfer::TokenId>& tokens,
                                          std::uint32_t length, const char* label) {
    if (tokens.size() < length) {
        std::cerr << label << " oracle has " << tokens.size() << " tokens, need " << length
                  << '\n';
        return {};
    }
    return std::vector<ninfer::TokenId>(tokens.begin(), tokens.begin() + length);
}

int run_overlapping_c3(ninfer::Engine& engine,
                       const std::array<std::vector<ninfer::TokenId>, 3>& prompts,
                       const std::array<std::vector<ninfer::TokenId>, 3>& dflash_oracles,
                       const std::array<std::vector<ninfer::TokenId>, 3>& target_oracles,
                       const char* label) {
    // Per-row isolation: each overlapping request must match sequential C=1 DFlash of the
    // same k. Target-only is a second oracle; a packed/T=1 greedy flip is not row mixing.
    constexpr std::array<std::uint32_t, 3> lengths{19, 13, 7};
    auto a = engine.submit(engine.prepare_tokens(prompts[0]), greedy_options(lengths[0]));
    auto b = engine.submit(engine.prepare_tokens(prompts[1]), greedy_options(lengths[1]));
    auto c = engine.submit(engine.prepare_tokens(prompts[2]), greedy_options(lengths[2]));
    std::array<ninfer::GenerationResult, 3> results;
    try {
        results = {a.wait(), b.wait(), c.wait()};
    } catch (const std::exception& error) {
        std::cerr << label << " concurrent speculative verify threw: " << error.what() << '\n';
        return 1;
    }
    const char* names[] = {"A", "B", "C"};
    int failed          = 0;
    for (std::size_t i = 0; i < 3; ++i) {
        const std::string request = std::string(label) + " request " + names[i];
        const std::string dflash_label = request + " C=1 DFlash";
        const std::string target_label = request + " target-only";
        const std::vector<ninfer::TokenId> dflash_want =
            token_prefix(dflash_oracles[i], lengths[i], dflash_label.c_str());
        const std::vector<ninfer::TokenId> target_want =
            token_prefix(target_oracles[i], lengths[i], target_label.c_str());
        if (dflash_want.empty() || target_want.empty()) { return 1; }
        const std::string vs_dflash = request + " vs C=1 DFlash";
        const std::string vs_target = request + " vs target-only";
        if (check_tokens(vs_dflash.c_str(), results[i], dflash_want, OracleKind::C1DFlash) != 0) {
            failed = 1;
        }
        if (check_tokens(vs_target.c_str(), results[i], target_want, OracleKind::TargetOnly) != 0) {
            failed = 1;
        }
    }
    return failed;
}

} // namespace

int main() {
    const char* artifact = std::getenv("NINFER_QWEN3_8_27B_NVFP4_DFLASH_WEIGHTS");
    if (artifact == nullptr || *artifact == '\0') {
        std::cout << "skip: NINFER_QWEN3_8_27B_NVFP4_DFLASH_WEIGHTS is not set\n";
        return 77;
    }

    // AIME is one generate (prompt → thinking → response). Live salad shows up after a
    // second user prompt or after a tool-call turn, i.e. prefix reuse, not a cold prefill.
    if (std::getenv("NINFER_DFLASH_TEST_TURNS_ONLY") != nullptr) {
        if (const int result = exercise_p_less_thinking_followup(artifact); result != 0) {
            return result;
        }
        if (const int result = exercise_p_less_finished_thinking_followup(artifact); result != 0) {
            return result;
        }
        if (const int result = exercise_p_less_tool_history_reuse(artifact); result != 0) {
            return result;
        }
        if (const int result = exercise_p_less_c2_reuse_during_peer_decode(artifact); result != 0) {
            return result;
        }
        return 0;
    }

    if (std::getenv("NINFER_DFLASH_TEST_SKIP_VISION") == nullptr) {
        std::cerr << "dflash_real: Vision-composed prefill and MRoPE decode\n";
        if (const int result = exercise_dflash_vision(artifact); result != 0) { return result; }
    }
    if (std::getenv("NINFER_DFLASH_TEST_ONLY_VISION") != nullptr) { return 0; }

    if (std::getenv("NINFER_DFLASH_TEST_SKIP_ENTITLEMENT") == nullptr &&
        std::getenv("NINFER_DFLASH_TEST_C2_PLESS") == nullptr) {
        std::cerr << "dflash_real: chain-verify short-output Main KV entitlement\n";
        if (const int result = exercise_chain_verify_short_output_entitlement(artifact);
            result != 0) {
            return result;
        }
    }

    // Packed verify is T=W (SmallT GQA at T<=6; Prompt GQA at T>6 on 24 heads). Ordinary
    // decode stays T=1 GEMV. C=1 vs target-only can flip a later greedy token
    // (k=4 prompt 0 token 21). C>1 must still match saved C=1 DFlash of the same k.
    const std::array<std::vector<ninfer::TokenId>, 3> prompts{
        std::vector<ninfer::TokenId>{
            248045, 846,    198, 109266, 3709,  96220, 117443, 97913,
            1710,   248046, 198, 248045, 74455, 198,   248068, 198,
        },
        std::vector<ninfer::TokenId>{
            248045, 846,    198, 109266, 4120,  96220, 117443, 97913,
            1710,   248046, 198, 248045, 74455, 198,   248068, 198,
        },
        std::vector<ninfer::TokenId>{
            248045, 846,    198, 109266, 5200,  96220, 117443, 97913,
            1710,   248046, 198, 248045, 74455, 198,   248068, 198,
        },
    };

    if (std::getenv("NINFER_DFLASH_TEST_TREE_STATE_ONLY") != nullptr) {
        return exercise_p_less_tree_commit_matches_reconstruction(artifact, prompts[0]);
    }
    if (std::getenv("NINFER_DFLASH_TEST_TREE_LIKELIHOOD_ONLY") != nullptr) {
        return exercise_p_less_tree_target_likelihood(artifact, prompts[0]);
    }
    if (std::getenv("NINFER_DFLASH_TEST_GREEDY_TREE_ONLY") != nullptr) {
        return exercise_greedy_tree_matches_ordinary(artifact, prompts[0]);
    }
    if (std::getenv("NINFER_DFLASH_TEST_CHAT_COHERENCE") != nullptr) {
        return exercise_p_less_tree_chat_coherence(artifact);
    }

    if (std::getenv("NINFER_DFLASH_TEST_C2_PLESS") != nullptr) {
        if (const int result = exercise_p_less_c2_matches_c1(artifact, prompts); result != 0) {
            return result;
        }
        if (std::getenv("NINFER_DFLASH_TEST_C2_LONG") != nullptr) {
            if (const int result = exercise_p_less_c2_long_context_matches_c1(artifact, prompts);
                result != 0) {
                return result;
            }
        }
        if (std::getenv("NINFER_DFLASH_TEST_NO_GRAPH") != nullptr) {
            if (const int result =
                    exercise_p_less_c2_mixed_frontiers_matches_c1(artifact, prompts);
                result != 0) {
                return result;
            }
        }
        if (const int result = exercise_p_less_c2_reuse_during_peer_decode(artifact); result != 0) {
            return result;
        }
        return 0;
    }

    if (std::getenv("NINFER_DFLASH_TEST_SKIP_LIKELIHOOD") == nullptr) {
        if (const int result = exercise_p_less_target_likelihood(artifact, prompts[0]);
            result != 0) {
            return result;
        }
        if (const int result = exercise_p_less_c2_matches_c1(artifact, prompts); result != 0) {
            return result;
        }
        if (std::getenv("NINFER_DFLASH_TEST_NO_GRAPH") != nullptr) {
            if (const int result =
                    exercise_p_less_c2_mixed_frontiers_matches_c1(artifact, prompts);
                result != 0) {
                return result;
            }
        }
        if (std::getenv("NINFER_DFLASH_TEST_LIKELIHOOD_ONLY") != nullptr) {
            return 0;
        }
    }

    auto run_k = [&](std::uint32_t draft_tokens, const char* label) -> int {
        std::array<std::vector<ninfer::TokenId>, 3> target_oracles;
        const char* only_prompt = std::getenv("NINFER_DFLASH_TEST_ONLY_PROMPT");
        {
            // A DFlash self-comparison can miss a verifier that consistently commits the
            // wrong token. Compare against the same artifact with speculative decoding disabled.
            ninfer::Engine baseline(base_engine_options(artifact));
            for (std::size_t i = 0; i < prompts.size(); ++i) {
                if (only_prompt != nullptr && std::string(only_prompt) != std::to_string(i)) {
                    continue;
                }
                const ninfer::GenerationResult result =
                    baseline.generate(baseline.prepare_tokens(prompts[i]), greedy_options(24));
                if (result.generated_token_ids.size() != 24) {
                    std::cerr << label << " baseline did not generate 24 tokens\n";
                    return 1;
                }
                target_oracles[i] = result.generated_token_ids;
            }
        }

        ninfer::EngineOptions dflash_options =
            speculative_engine_options(artifact, ninfer::SpeculativeBackend::DFlash, draft_tokens, 3);
        if (std::getenv("NINFER_DFLASH_TEST_MAX1") != nullptr) {
            dflash_options.max_concurrency = 1;
        }
        if (std::getenv("NINFER_DFLASH_TEST_FULL_HEAD") != nullptr) {
            dflash_options.speculative.proposal_head = ninfer::ProposalHead::Full;
        }
        if (std::getenv("NINFER_DFLASH_TEST_CHAIN") != nullptr) {
            dflash_options.speculative.dflash_verify_width = draft_tokens + 1;
        }
        if (const char* verify_width = std::getenv("NINFER_DFLASH_TEST_VERIFY_WIDTH")) {
            dflash_options.speculative.dflash_verify_width =
                static_cast<std::uint32_t>(std::atoi(verify_width));
        }
        try {
            ninfer::Engine engine(dflash_options);
            if (const int result = check_dflash_load(engine); result != 0) { return result; }
            std::array<std::vector<ninfer::TokenId>, 3> dflash_oracles;
            int failed = 0;
            for (std::size_t i = 0; i < 3; ++i) {
                if (only_prompt != nullptr &&
                    std::string(only_prompt) != std::to_string(i)) {
                    continue;
                }
                ninfer::GenerationResult dflash_result;
                if (const int result = greedy_oracle(engine, prompts[i], dflash_oracles[i], label,
                                                     &dflash_result);
                    result != 0) {
                    return result;
                }
                // Packed/T=1 identity. Do not return here: C>1 isolation still uses
                // the saved C=1 DFlash tokens even when a later greedy token flips.
                const std::string vs_target =
                    std::string(label) + " prompt " + std::to_string(i) + " vs target-only";
                if (check_tokens(vs_target.c_str(), dflash_result, target_oracles[i],
                                 OracleKind::TargetOnly) != 0) {
                    failed = 1;
                }
            }
            if (only_prompt == nullptr && (target_oracles[0] == target_oracles[1] ||
                                           target_oracles[0] == target_oracles[2] ||
                                           target_oracles[1] == target_oracles[2])) {
                std::cerr << label
                          << " target-only oracles are not distinct across the three prompts\n";
                return 1;
            }
            if (only_prompt == nullptr && (dflash_oracles[0] == dflash_oracles[1] ||
                                           dflash_oracles[0] == dflash_oracles[2] ||
                                           dflash_oracles[1] == dflash_oracles[2])) {
                std::cerr << label
                          << " C=1 DFlash oracles are not distinct across the three prompts\n";
                return 1;
            }
            if (only_prompt == nullptr && dflash_options.max_concurrency >= 3 &&
                run_overlapping_c3(engine, prompts, dflash_oracles, target_oracles, label) != 0) {
                failed = 1;
            }
            if (draft_tokens == 7 && dflash_options.speculative.dflash_verify_width == 0 &&
                (only_prompt == nullptr || std::string(only_prompt) == "0") &&
                exercise_p_less_product_tree(engine, prompts[0]) != 0) {
                failed = 1;
            }
            if (failed != 0) { return failed; }
        } catch (const std::bad_alloc&) {
            std::cerr << label << " engine std::bad_alloc (k=" << draft_tokens
                      << " max_conc=" << dflash_options.max_concurrency
                      << " graph=" << dflash_options.use_cuda_graph << ")\n";
            return 1;
        }
        std::cout << "ok " << label << '\n' << std::flush;
        return 0;
    };

    const char* only_k = std::getenv("NINFER_DFLASH_TEST_ONLY_K");
    if (only_k == nullptr || std::string(only_k) == "7") {
        std::string label = "DFlash2 k=7 W=12 tree C=3";
        if (std::getenv("NINFER_DFLASH_TEST_CHAIN") != nullptr) {
            label = "DFlash2 k=7 W=8 chain C=3";
        } else if (const char* width = std::getenv("NINFER_DFLASH_TEST_VERIFY_WIDTH")) {
            label = "DFlash2 k=7 W=" + std::string(width) + " override C=3";
        }
        if (const int result = run_k(7, label.c_str()); result != 0) { return result; }
    }
    if (only_k == nullptr || std::string(only_k) == "1") {
        if (const int result = run_k(1, "DFlash2 k=1 chain C=3"); result != 0) { return result; }
    }
    if (only_k == nullptr || std::string(only_k) == "4") {
        if (const int result = run_k(4, "DFlash2 k=4 chain C=3"); result != 0) { return result; }
    }
    if (only_k == nullptr || std::string(only_k) == "5") {
        if (const int result = run_k(5, "DFlash2 k=5 chain C=3"); result != 0) { return result; }
    }

    {
        const char* label = "DFlash2 adaptive N=7 {3,4,5}";
        ninfer::Engine engine(adaptive_engine_options(
            artifact, ninfer::SpeculativeBackend::DFlash, 7, 3));
        if (const int result = check_dflash_load(engine); result != 0) { return result; }
        // Prefill commits the first generated token. Seven tokens leave remaining=6 on
        // the first DFlash round: budget_extent=5, seed live_k=5, W(k)=6 under W_ceil=6.
        const ninfer::GenerationResult compact =
            engine.generate(engine.prepare_tokens(prompts[0]), greedy_options(7));
        if (compact.generated_token_ids.size() != 7 || check_speculative(compact, label) != 0) {
            std::cerr << label << " compact k=5 under N=7 did not complete\n";
            dump_tokens("  got", compact.generated_token_ids);
            dump_speculative("  spec", compact.speculative);
            return 1;
        }
        if (compact.speculative.rounds_per_draft.size() < 6 ||
            compact.speculative.rounds_per_draft[5] == 0) {
            std::cerr << label << " compact run did not record k=5 rounds under N=7\n";
            dump_speculative("  spec", compact.speculative);
            return 1;
        }
        const std::uint64_t decode_before = engine.runtime_stats().committed_decode_tokens;
        auto compact_a = engine.submit(engine.prepare_tokens(prompts[0]), greedy_options(7));
        auto compact_b = engine.submit(engine.prepare_tokens(prompts[1]), greedy_options(7));
        const ninfer::GenerationResult compact_ra = compact_a.wait();
        const ninfer::GenerationResult compact_rb = compact_b.wait();
        if (compact_ra.generated_token_ids.size() != 7 ||
            compact_rb.generated_token_ids.size() != 7 ||
            check_speculative(compact_ra, label) != 0 ||
            check_speculative(compact_rb, label) != 0 ||
            compact_ra.speculative.rounds_per_draft.size() < 6 ||
            compact_ra.speculative.rounds_per_draft[5] == 0) {
            std::cerr << label << " compact C=2 k=5 under N=7 failed\n";
            dump_speculative("  A", compact_ra.speculative);
            dump_speculative("  B", compact_rb.speculative);
            return 1;
        }
        // Each seven-token request commits its first token in prefill, then six tokens in decode.
        // A terminal wait must not become observable before those decode counters are published.
        if (engine.runtime_stats().committed_decode_tokens < decode_before + 12) {
            std::cerr << label << " terminal waits preceded committed decode-stat publication\n";
            return 1;
        }
        const ninfer::GenerationResult seq =
            engine.generate(engine.prepare_tokens(prompts[0]), greedy_options(24));
        if (seq.generated_token_ids.size() != 24 || check_speculative(seq, label) != 0) {
            std::cerr << label << " C=1 did not complete\n";
            dump_tokens("  got", seq.generated_token_ids);
            dump_speculative("  spec", seq.speculative);
            return 1;
        }
        std::uint64_t hist = 0;
        for (std::uint64_t count : seq.speculative.rounds_per_draft) { hist += count; }
        if (hist != seq.speculative.rounds) {
            std::cerr << label << " rounds_per_draft sum mismatch\n";
            return 1;
        }
        const std::uint32_t live = seq.speculative.live_draft_tokens;
        if (live != 3 && live != 4 && live != 5) {
            std::cerr << label << " live_draft_tokens " << live << " not in {3,4,5}\n";
            return 1;
        }
        auto a = engine.submit(engine.prepare_tokens(prompts[0]), greedy_options(19));
        auto b = engine.submit(engine.prepare_tokens(prompts[1]), greedy_options(13));
        auto c = engine.submit(engine.prepare_tokens(prompts[2]), greedy_options(7));
        const std::array<ninfer::GenerationResult, 3> results{a.wait(), b.wait(), c.wait()};
        const std::array<std::uint32_t, 3> lengths{19, 13, 7};
        for (std::size_t i = 0; i < 3; ++i) {
            if (results[i].generated_token_ids.size() != lengths[i] ||
                check_speculative(results[i], label) != 0) {
                std::cerr << label << " C=3 request " << i << " failed\n";
                dump_tokens("  got", results[i].generated_token_ids);
                dump_speculative("  spec", results[i].speculative);
                return 1;
            }
        }
        std::cout << "ok " << label << '\n' << std::flush;
    }

    {
        const char* label = "DFlash2 terminal delivery and queue telemetry";
        ninfer::Engine engine(adaptive_engine_options(
            artifact, ninfer::SpeculativeBackend::DFlash, 7, 1));
        if (const int result = check_dflash_load(engine); result != 0) { return result; }

        struct DiscardingSink final : ninfer::OutputSink {
            void publish(ninfer::OutputDelta) override {}
        } sink;
        auto terminal = engine.submit(engine.prepare_tokens(prompts[0]), greedy_options(7));
        bool rejected = false;
        try {
            (void)terminal.wait(&sink);
        } catch (const std::logic_error&) {
            rejected = true;
        }
        if (!rejected) {
            std::cerr << label << " accepted a sink for terminal-only delivery\n";
            return 1;
        }
        if (terminal.wait().generated_token_ids.size() != 7) {
            std::cerr << label << " terminal request did not complete after sink rejection\n";
            return 1;
        }

        auto active = engine.submit(engine.prepare_tokens(prompts[0]), greedy_options(16));
        auto queued = engine.submit(engine.prepare_tokens(prompts[1]), greedy_options(7));
        if (engine.runtime_stats().waiting_requests == 0) {
            std::cerr << label << " full engine did not publish queued depth at ingress\n";
            return 1;
        }
        if (active.wait().generated_token_ids.size() != 16 ||
            queued.wait().generated_token_ids.size() != 7) {
            std::cerr << label << " queued requests did not complete\n";
            return 1;
        }
        std::cout << "ok " << label << '\n' << std::flush;
    }

    {
        const char* label = "DFlash2 adaptive RAM reseed {3,4,5}";
        ninfer::EngineOptions options =
            adaptive_engine_options(artifact, ninfer::SpeculativeBackend::DFlash, 7, 1);
        options.kv_ram_capacity_bytes = 1024ULL * 1024ULL * 1024ULL;
        ninfer::Engine engine(options);
        if (const int result = check_dflash_load(engine); result != 0) { return result; }
        if (engine.memory_summary().kv_ram_capacity_bytes == 0) {
            std::cerr << label << " RAM tier is disabled\n";
            return 1;
        }
        const ninfer::GenerationResult first =
            engine.generate(engine.prepare_tokens(prompts[0]), greedy_reuse(8, false));
        if (first.generated_token_ids.size() != 8 ||
            check_adaptive_dflash(first, label) != 0 ||
            first.speculative.live_draft_tokens != 5) {
            std::cerr << label << " source did not stay at seeded k=5\n";
            dump_speculative("  spec", first.speculative);
            return 1;
        }
        const auto captures_before = engine.runtime_stats().kv_ram_captures;
        const ninfer::GenerationResult evictor =
            engine.generate(engine.prepare_tokens(prompts[1]), greedy_reuse(4, false));
        if (evictor.generated_token_ids.size() != 4 ||
            engine.runtime_stats().kv_ram_captures <= captures_before) {
            std::cerr << label << " evictor did not capture the first chat to RAM\n";
            return 1;
        }
        const std::vector<ninfer::TokenId> history =
            resume_prefix(prompts[0], first.generated_token_ids);
        const auto restores_before = engine.runtime_stats().kv_ram_restores;
        const ninfer::GenerationResult hit =
            engine.generate(engine.prepare_tokens(history), greedy_reuse(4, true));
        if (hit.prefix_reuse_source != ninfer::PrefixReuseSource::HostRam) {
            std::cerr << label << " restore source is "
                      << static_cast<int>(hit.prefix_reuse_source) << ", expected HostRam\n";
            return 1;
        }
        if (engine.runtime_stats().kv_ram_restores != restores_before + 1) {
            std::cerr << label << " kv_ram_restores did not increment\n";
            return 1;
        }
        if (hit.generated_token_ids.size() != 4 || check_adaptive_dflash(hit, label) != 0 ||
            hit.speculative.live_draft_tokens != 5) {
            std::cerr << label << " RAM restore leaked live_k off the seeded attractor\n";
            dump_speculative("  hit", hit.speculative);
            return 1;
        }
        std::cout << "ok " << label << '\n' << std::flush;
    }

    {
        const char* label = "DFlash2 adaptive RAM restore in flight {3,4,5}";
        ninfer::EngineOptions options =
            adaptive_engine_options(artifact, ninfer::SpeculativeBackend::DFlash, 7, 2);
        options.kv_ram_capacity_bytes = 1024ULL * 1024ULL * 1024ULL;
        ninfer::Engine engine(options);
        if (const int result = check_dflash_load(engine); result != 0) { return result; }
        const ninfer::GenerationResult first =
            engine.generate(engine.prepare_tokens(prompts[0]), greedy_reuse(8, false));
        const ninfer::GenerationResult second =
            engine.generate(engine.prepare_tokens(prompts[1]), greedy_reuse(4, false));
        if (first.generated_token_ids.size() != 8 || second.generated_token_ids.size() != 4) {
            std::cerr << label << " C=2 fill did not complete\n";
            return 1;
        }
        const auto captures_before = engine.runtime_stats().kv_ram_captures;
        const ninfer::GenerationResult evictor =
            engine.generate(engine.prepare_tokens(prompts[2]), greedy_reuse(4, false));
        if (evictor.generated_token_ids.size() != 4 ||
            engine.runtime_stats().kv_ram_captures <= captures_before) {
            std::cerr << label << " third request did not dump a chat to RAM\n";
            return 1;
        }
        const std::vector<ninfer::TokenId> history =
            resume_prefix(prompts[0], first.generated_token_ids);
        // Restored exact-prefix generate overlaps a compact k=5 decode (greedy 7 →
        // remaining=6 after prefill, budget_extent=5, W(k)=6 under W_ceil=6) on the
        // other lane while copy_stream H2D-restores cyclic DFlash KV / GDN / pages.
        auto restored_h =
            engine.submit(engine.prepare_tokens(history), greedy_reuse(4, true));
        auto inflight_h =
            engine.submit(engine.prepare_tokens(prompts[1]), greedy_reuse(7, false));
        const ninfer::GenerationResult hit      = restored_h.wait();
        const ninfer::GenerationResult inflight = inflight_h.wait();
        if (hit.generated_token_ids.size() != 4 || inflight.generated_token_ids.size() != 7 ||
            check_adaptive_dflash(hit, label) != 0 ||
            check_adaptive_dflash(inflight, label) != 0 ||
            hit.speculative.live_draft_tokens != 5) {
            std::cerr << label << " overlapping RAM restore dropped live_k or failed compact\n";
            dump_speculative("  hit", hit.speculative);
            dump_speculative("  inflight", inflight.speculative);
            return 1;
        }
        if (inflight.speculative.rounds_per_draft.size() < 6 ||
            inflight.speculative.rounds_per_draft[5] == 0) {
            std::cerr << label << " inflight compact did not record k=5 rounds under N=7\n";
            dump_speculative("  inflight", inflight.speculative);
            return 1;
        }
        if (hit.prefix_reuse_source != ninfer::PrefixReuseSource::HostRam &&
            hit.prefix_reuse_source != ninfer::PrefixReuseSource::VramResident) {
            std::cerr << label << " restore source is "
                      << static_cast<int>(hit.prefix_reuse_source) << '\n';
            return 1;
        }
        std::cout << "ok " << label << '\n' << std::flush;
    }
    return 0;
}
