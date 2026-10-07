#include "core/arena.h"
#include "core/decode_graph.h"
#include "core/device.h"
#include "targets/qwen3_6/impl/runtime/token_masks.h"
#include "targets/qwen3_6/impl/frontend/test_access.h"
#include "targets/qwen3_6/impl/frontend/tokenizer.h"
#include "text/unicode.h"
#include <nlohmann/json.hpp>
#include <algorithm>
#include <array>
#include <cstddef>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <vector>

namespace {
using namespace ninfer;
namespace family        = ninfer::targets::qwen3_6;
using FrontendResources = family::FrontendResources;

std::string template_source() {
    std::ifstream input(NINFER_SOURCE_DIR
                        "/tests/fixtures/frontend/thinking_toggle_chat_template.jinja");
    if (!input) throw std::runtime_error("missing component template fixture");
    std::string source{std::istreambuf_iterator<char>(input), {}};
    if (!source.empty() && source.back() == '\n') source.pop_back();
    return source;
}

nlohmann::json added(int id, std::string content, bool special = false) {
    return nlohmann::json{{"id", id},
                          {"content", std::move(content)},
                          {"single_word", false},
                          {"lstrip", false},
                          {"rstrip", false},
                          {"normalized", false},
                          {"special", special}};
}

nlohmann::json decoder_added(std::string content, bool special = false) {
    nlohmann::json value = added(0, std::move(content), special);
    value.erase("id");
    return value;
}

FrontendResources resources(const std::string& chat_template = template_source()) {
    FrontendResources result;
    result.chat_template_jinja  = chat_template;
    const nlohmann::json tokens = nlohmann::json::array({added(1, "helloST"),
                                                         added(2, "OPtail"),
                                                         added(3, "thought</thi"),
                                                         added(4, "nk>\n\nanswer"),
                                                         added(6, "<eos>", true),
                                                         added(7, "<0.0 seconds>"),
                                                         added(14, "   \n"),
                                                         added(15, "answer"),
                                                         added(16, "<tool_"),
                                                         added(17, "call>"),
                                                         added(18, "<function=f>"),
                                                         added(19, "</function>"),
                                                         added(20, "</tool_call>"),
                                                         added(21, "<tool_call>"),
                                                         added(22, "preface"),
                                                         added(23, "call"),
                                                         added(24, "a <"),
                                                         added(30, "user\n"),
                                                         added(31, "assistant\n"),
                                                         added(32, "\n"),
                                                         added(33, "system\n"),
                                                         added(248045, "<|im_start|>", true),
                                                         added(248046, "<|im_end|>", true),
                                                         added(248053, "<|vision_start|>", true),
                                                         added(248054, "<|vision_end|>", true),
                                                         added(248056, "<|image_pad|>", true),
                                                         added(248057, "<|video_pad|>", true),
                                                         added(248068, "<think>"),
                                                         added(248069, "</think>")});
    result.tokenizer_json       = nlohmann::json{
        {"model",
         {{"type", "BPE"},
          {"vocab", {{"x", 0}, {"ä", 10}, {"¸", 11}, {"Ń", 12}}},
          {"merges", nlohmann::json::array()}}},
        {"added_tokens",
         tokens}}.dump();

    nlohmann::json decoder = nlohmann::json::object();
    for (const nlohmann::json& token : tokens) {
        nlohmann::json value = token;
        const std::string id = std::to_string(value.at("id").get<int>());
        value.erase("id");
        decoder[id] = std::move(value);
    }
    decoder["248070"]            = decoder_added("<|audio_start|>", true);
    decoder["248071"]            = decoder_added("<|audio_end|>", true);
    decoder["248072"]            = decoder_added("<tts_pad>", true);
    decoder["248073"]            = decoder_added("<tts_text_bos>", true);
    decoder["248074"]            = decoder_added("<tts_text_eod>", true);
    decoder["248075"]            = decoder_added("<tts_text_bos_single>", true);
    decoder["248076"]            = decoder_added("<|audio_pad|>", true);
    result.tokenizer_config_json = nlohmann::json{
        {"add_bos_token", false},
        {"add_prefix_space", false},
        {"pad_token", "<|endoftext|>"},
        {"chat_template", result.chat_template_jinja},
        {"added_tokens_decoder",
         std::move(decoder)}}.dump();
    result.generation_config_json = R"({"eos_token_id":[6]})";
    result.preprocessor_config_json =
        R"({"patch_size":16,"temporal_patch_size":2,"merge_size":2,"image_mean":[0.5,0.5,0.5],"image_std":[0.5,0.5,0.5],"size":{"shortest_edge":4096,"longest_edge":16777216}})";
    result.video_preprocessor_config_json =
        R"({"patch_size":16,"temporal_patch_size":2,"merge_size":2,"image_mean":[0.5,0.5,0.5],"image_std":[0.5,0.5,0.5],"size":{"shortest_edge":4096,"longest_edge":25165824}})";
    return result;
}

std::string byte_level_symbol(std::uint8_t target) {
    std::uint32_t next = 256;
    for (int value = 0; value <= 255; ++value) {
        const bool visible = (value >= 33 && value <= 126) || (value >= 161 && value <= 172) ||
                             (value >= 174 && value <= 255);
        const std::uint32_t codepoint = visible ? static_cast<std::uint32_t>(value) : next++;
        if (value == target) {
            return ninfer::text::unicode_internal::codepoint_to_utf8(
                static_cast<std::int32_t>(codepoint));
        }
    }
    throw std::logic_error("byte-level test symbol is outside one byte");
}

void require(bool value, const char* message) {
    if (!value) throw std::runtime_error(message);
}

void run() {
    auto owned     = resources();
    auto tokenizer = nlohmann::json::parse(owned.tokenizer_json);
    // Rendered tool declarations are literal client text, so every printable ASCII
    // byte needs an ordinary byte-level vocabulary symbol.
    auto& vocab = tokenizer["model"]["vocab"];
    for (int c = 32; c < 127; ++c) {
        const std::string symbol = byte_level_symbol(static_cast<std::uint8_t>(c));
        if (!vocab.contains(symbol)) { vocab[symbol] = 1000 + c; }
    }
    owned.tokenizer_json    = tokenizer.dump();
    const auto frontend     = family::FrontendTestAccess::create_component(owned, false);
    constexpr int width_max = 5;
    constexpr int capacity  = 6;
    family::frontend_internal::Tokenizer encoder(
        {owned.tokenizer_json, owned.tokenizer_config_json, owned.generation_config_json});
    const auto prefix = encoder.encode("<tool_call>\n<function=f>\n<parameter=value>\n");
    std::array<family::OutputSession, capacity> sessions;
    for (int row = 0; row < capacity; ++row) {
        PromptInput input;
        ChatMessage user;
        user.role = ChatRole::User;
        user.parts.push_back(MessagePart{.kind = MessagePartKind::Text, .text = "x"});
        input.messages.push_back(std::move(user));
        input.options.enable_thinking = false;
        auto tool                     = nlohmann::json::parse(
            R"({"type":"function","function":{"name":"f","parameters":{"type":"object","properties":{"value":{"type":"integer","enum":[1]}},"required":["value"],"additionalProperties":false}}})");
        tool["function"]["parameters"]["properties"]["value"]["enum"] = {row + 1};
        input.options.tool_jsons.push_back(tool.dump());
        auto prompt   = frontend.prepare(std::move(input));
        sessions[row] = frontend.make_output_session(prompt, {});
        require(sessions[row].has_token_grammar(), "component tool grammar missing");
        (void)sessions[row].preview(prefix, 1024, FinishReason::OutputLimit);
        (void)sessions[row].commit_preview();
    }

    DeviceContext device;
    constexpr int words = (family::kTokenDomain + 31) / 32;
    DeviceArena arena(std::size_t{4} * 1024 * 1024);
    auto masks       = arena.alloc(DType::I32, {words, width_max, capacity});
    auto sampling    = arena.alloc(DType::I32, {int(sizeof(ops::SamplingConfig) / 4), capacity});
    auto nodes       = arena.alloc(DType::I32, {width_max, capacity, 2});
    auto ids_storage = arena.alloc(DType::I32, {width_max, capacity});
    auto parents_storage = arena.alloc(DType::I32, {width_max, capacity});
    auto counts_storage  = arena.alloc(DType::I32, {capacity});
    auto updated_ids     = arena.alloc(DType::I32, {width_max, capacity});
    auto updated_parents = arena.alloc(DType::I32, {width_max, capacity});
    auto updated_counts  = arena.alloc(DType::I32, {capacity});
    family::TokenMaskExchange exchange(masks, sampling, nodes, device.stream);
    PinnedHostBuffer result_masks(masks.bytes()), result_sampling(sampling.bytes());
    std::array<const family::OutputSession*, capacity> outputs{};
    std::array<ops::SamplingConfig, capacity> configs{};
    std::array<int, std::size_t{width_max} * capacity> ids{}, parents{};
    std::array<int, capacity> counts{};
    std::vector<std::uint32_t> expected(std::size_t{width_max} * words);

    for (int batch : {1, 2, 3, 4, 5, 6, 1}) {
        for (bool tree : {false, true}) {
            for (int width : {3, 5}) {
                // Production gives each (batch, draft width) its own topology.
                // CUDA cannot update a 2D memcpy node across changing extents.
                DecodeGraphExecutable graph;
                Tensor ids_view(ids_storage.data, DType::I32, {width, batch});
                Tensor parent_view(parents_storage.data, DType::I32, {width, batch});
                Tensor count_view(counts_storage.data, DType::I32, {batch});
                auto body = [&] {
                    const auto submission =
                        exchange.enqueue(ids_view, tree ? &parent_view : nullptr, count_view,
                                         device.stream, device.host_stream);
                    CUDA_CHECK(cudaStreamWaitEvent(device.stream, submission.ready, 0));
                    CUDA_CHECK(cudaMemcpyAsync(result_masks.data(), masks.data,
                                               std::size_t(batch) * width_max * words *
                                                   sizeof(std::uint32_t),
                                               cudaMemcpyDeviceToHost, device.stream));
                    CUDA_CHECK(cudaMemcpyAsync(result_sampling.data(), sampling.data,
                                               std::size_t(batch) * sizeof(ops::SamplingConfig),
                                               cudaMemcpyDeviceToHost, device.stream));
                };
                DecodeGraphDefinition definition;
                definition.capture(device.stream, body);
                graph.instantiate(definition);
                // Rebind between replays and change node counts/IDs. Captured callbacks
                // must use this round's sessions and dynamic valid-column counts.
                for (int replay = 0; replay < 4; ++replay) {
                    if (replay == 2) {
                        // A new frontier profile can retain shape while moving
                        // workspace input panels. The executable must update all
                        // source addresses, not merely replay identical pointers.
                        ids_view    = Tensor(updated_ids.data, DType::I32, {width, batch});
                        parent_view = Tensor(updated_parents.data, DType::I32, {width, batch});
                        count_view  = Tensor(updated_counts.data, DType::I32, {batch});
                        DecodeGraphDefinition updated;
                        updated.capture(device.stream, body);
                        graph.update(updated);
                    }
                    for (int row = 0; row < batch; ++row) {
                        outputs[row]      = ((row + replay) % 3 == 2) ? nullptr : &sessions[row];
                        configs[row].seed = 100 + row + replay * 10;
                        counts[row]       = width - (replay % 2);
                        const std::array<int, 5> tokens{0, 1000 + '1' + row, 32, 1000 + '<',
                                                        1000 + '/'};
                        for (int node = 0; node < width; ++node) {
                            ids[row * width + node]     = tokens[node];
                            parents[row * width + node] = tree && node == 2 ? 0 : node - 1;
                        }
                        if (replay % 2) ids[row * width + 1] = 22; // prose branch
                    }
                    exchange.bind({outputs.data(), std::size_t(batch)},
                                  {configs.data(), std::size_t(batch)});
                    // Inputs are ordered on the launch stream: the DeviceContext streams are
                    // nonblocking, so legacy-stream uploads could race the overlap branch.
                    CUDA_CHECK(cudaMemcpyAsync(ids_view.data, ids.data(),
                                               std::size_t(width) * batch * sizeof(int),
                                               cudaMemcpyHostToDevice, device.stream));
                    CUDA_CHECK(cudaMemcpyAsync(parent_view.data, parents.data(),
                                               std::size_t(width) * batch * sizeof(int),
                                               cudaMemcpyHostToDevice, device.stream));
                    CUDA_CHECK(cudaMemcpyAsync(count_view.data, counts.data(), batch * sizeof(int),
                                               cudaMemcpyHostToDevice, device.stream));
                    if (replay == 0)
                        body();
                    else
                        graph.launch(device.stream);
                    device.synchronize();
                    exchange.rethrow_error();
                    const auto* actual = static_cast<const std::uint32_t*>(result_masks.data());
                    const auto* actual_config =
                        static_cast<const ops::SamplingConfig*>(result_sampling.data());
                    for (int row = 0; row < batch; ++row) {
                        require(actual_config[row].seed == configs[row].seed,
                                "stale captured sampling binding");
                        if (!outputs[row]) {
                            require(actual_config[row].allowed_token_words == nullptr,
                                    "unconstrained row retained grammar");
                            continue;
                        }
                        std::vector<int> row_parents(counts[row]);
                        for (int node = 0; node < counts[row]; ++node)
                            row_parents[node] = tree ? parents[row * width + node] : node - 1;
                        sessions[row].fill_token_masks(
                            {ids.data() + std::ptrdiff_t(row) * width, std::size_t(counts[row])},
                            row_parents, {expected.data(), std::size_t(counts[row]) * words});
                        for (int other = 0; other < capacity; ++other) {
                            const int value_token = 1000 + '1' + other;
                            const bool allowed =
                                (expected[value_token / 32] >> (value_token % 32)) & 1U;
                            require(allowed == (other == row),
                                    "row fixture does not distinguish its own enum value");
                        }
                        require(std::equal(expected.begin(),
                                           expected.begin() + std::ptrdiff_t(counts[row]) * words,
                                           actual + std::ptrdiff_t(row) * width_max * words),
                                "GPU mask differs from committed CPU grammar snapshot");
                        require(actual_config[row].allowed_token_words ==
                                    static_cast<std::uint32_t*>(masks.data) +
                                        std::ptrdiff_t(row) * width_max * words,
                                "mask row pointer changed across graph replay");
                        require(actual_config[row].allowed_token_column_stride == words,
                                "invalid node mask stride");
                    }
                }
                outputs[0] = &sessions[0];
                exchange.bind({outputs.data(), std::size_t(batch)},
                              {configs.data(), std::size_t(batch)});
                counts[0] = 0; // callback failure must drain and surface on the owning thread
                CUDA_CHECK(cudaMemcpyAsync(count_view.data, counts.data(), batch * sizeof(int),
                                           cudaMemcpyHostToDevice, device.stream));
                graph.launch(device.stream);
                device.synchronize();
                bool failed = false;
                try {
                    exchange.rethrow_error();
                } catch (const std::logic_error&) { failed = true; }
                require(failed, "callback failure was lost");
                const auto* actual_config =
                    static_cast<const ops::SamplingConfig*>(result_sampling.data());
                for (int row = 0; row < batch; ++row)
                    require(actual_config[row].allowed_token_words == nullptr,
                            "callback failure left a constrained sampling domain");
            }
        }
    }
    // An exception between fork and join must drain the callback before its
    // borrowed session dies, even though compute never waits on masks_ready.
    {
        auto temporary = std::move(sessions[0]);
        outputs.fill(nullptr);
        outputs[0] = &temporary;
        exchange.bind({outputs.data(), 4}, {configs.data(), 4});
        counts.fill(1);
        CUDA_CHECK(cudaMemcpyAsync(counts_storage.data, counts.data(), 4 * sizeof(int),
                                   cudaMemcpyHostToDevice, device.stream));
        Tensor fork_ids(ids_storage.data, DType::I32, {1, 4});
        Tensor fork_counts(counts_storage.data, DType::I32, {4});
        CUDA_CHECK(cudaMemsetAsync(masks.data, 0, masks.bytes(), device.stream));
        bool forked = false;
        try {
            const auto submission =
                exchange.enqueue(fork_ids, nullptr, fork_counts, device.stream, device.host_stream);
            forked = submission.ready != nullptr;
            throw std::runtime_error("interrupted after mask fork");
        } catch (const std::runtime_error&) { device.synchronize_all(); }
        require(forked, "partial enqueue fixture did not fork");
        exchange.rethrow_error();
        CUDA_CHECK(cudaMemcpy(result_masks.data(), masks.data, words * sizeof(std::uint32_t),
                              cudaMemcpyDeviceToHost));
        const auto* actual = static_cast<const std::uint32_t*>(result_masks.data());
        require(((actual[(1000 + '1') / 32] >> ((1000 + '1') % 32)) & 1U) != 0,
                "partial enqueue did not finish its borrowed-session callback");
    }
    outputs[0] = nullptr;
    exchange.bind({outputs.data(), 1}, {configs.data(), 1});
    device.synchronize_all();
}
} // namespace

int main() {
    try {
        run();
        std::cout << "tool mask exchange eager/capture C1..6 and partial-fork drain passed\n";
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
