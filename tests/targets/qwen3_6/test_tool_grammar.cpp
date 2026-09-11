#include "targets/qwen3_6/impl/frontend/tool_grammar.h"
#include "ninfer/targets/qwen3_6/frontend.h"

#include <iostream>
#include <memory>
#include <fstream>
#include <chrono>
#include <algorithm>
#include <array>
#include <set>

std::string envelope(const std::string& name,
                     const std::vector<std::pair<std::string, std::string>>& arguments) {
    std::string text = "<tool_call>\n<function=" + name + ">\n";
    for (const auto& [key, value] : arguments) {
        text += "<parameter=" + key + ">\n" + value + "\n</parameter>\n";
    }
    if (arguments.empty()) { text += '\n'; }
    return text + "</function>\n</tool_call>";
}

namespace fi = ninfer::targets::qwen3_6::frontend_internal;
using Json = nlohmann::ordered_json;

int real_tokenizer_probe(const char* directory, const char* fixture_path) {
    auto read = [](const std::string& path) {
        std::ifstream stream(path, std::ios::binary);
        if (!stream) { throw std::runtime_error("cannot read " + path); }
        return std::string(std::istreambuf_iterator<char>(stream), {});
    };
    const std::string root(directory);
    const auto tokenizer_json = read(root + "/tokenizer.json");
    const auto config_json = read(root + "/tokenizer_config.json");
    const auto generation_json = read(root + "/generation_config.json");
    auto tokenizer = std::make_shared<const fi::Tokenizer>(fi::TokenizerResources{
        tokenizer_json, config_json, generation_json});
    fi::ToolGrammarCompiler compiler(tokenizer);
    const auto fixture = Json::parse(read(fixture_path));
    std::vector<std::string> tools;
    for (const auto& tool : fixture.at("tools")) { tools.push_back(tool.dump()); }
    const auto begin = std::chrono::steady_clock::now();
    auto grammar = compiler.compile(tools, true);
    const auto compiled = std::chrono::steady_clock::now();
    const std::string text = "I can use the existing result. </think>\n\n<tool_call>\n<function=read>\n"
        "<parameter=filePath>\n/tmp/example.cpp\n</parameter>\n<parameter=limit>\n64\n</parameter>\n"
        "</function>\n</tool_call>";
    auto ids = tokenizer->encode(text);
    xgrammar::GrammarMatcher matcher(grammar->compiled);
    std::vector<std::uint32_t> words((ninfer::targets::qwen3_6::kTokenDomain + 31) / 32);
    std::int64_t shape = words.size();
    DLTensor mask{};
    mask.data = words.data(); mask.device = {kDLCPU, 0}; mask.ndim = 1;
    mask.dtype = {kDLInt, 32, 1}; mask.shape = &shape;
    int failures = 0;
    for (int token : ids) {
        (void)matcher.FillNextTokenBitmask(&mask);
        if (!(words[token / 32] & (1u << (token % 32))) || !matcher.AcceptToken(token)) {
            std::cerr << "real tokenizer valid token rejected: " << token << '\n';
            return 1;
        }
    }
    const auto valid_done = std::chrono::steady_clock::now();
    auto malformed = text;
    malformed.insert(malformed.find("\n</function>"), "\n</parameter>");
    xgrammar::GrammarMatcher bad(grammar->compiled);
    bool rejected = false;
    for (int token : tokenizer->encode(malformed)) {
        if (!bad.AcceptToken(token)) { rejected = true; break; }
    }
    failures += !rejected;
    failures += !grammar->decode_call(text.substr(text.find("<tool_call>"))).has_value();
    const std::string unordered = "Think.</think>\n" + envelope("read",
        {{"limit", "64"}, {"filePath", "/tmp/example.cpp"}});
    xgrammar::GrammarMatcher reverse(grammar->compiled);
    if (!reverse.AcceptString(unordered) ||
        !grammar->decode_call(unordered.substr(unordered.find("<tool_call>")))) {
        std::cerr << "real read schema rejected limit before filePath\n";
        ++failures;
    }
    const std::string unordered_prefix = "Think.</think>\n<tool_call>\n<function=read>\n"
        "<parameter=limit>\n64\n</parameter>\n";
    {
        fi::ToolGrammarState state(grammar);
        state.preview(tokenizer->encode(unordered_prefix));
        state.commit_preview();
        const std::string valid_suffix = "<parameter=filePath>\n/tmp/example.cpp\n</parameter>\n"
            "</function>\n</tool_call>";
        std::vector<ninfer::TokenId> tree{0};
        std::vector<std::int32_t> parents{-1};
        std::vector<std::pair<std::size_t, std::size_t>> branches;
        for (const auto& suffix : {valid_suffix, std::string("<parameter=limit>\n64\n</parameter>"),
                                  std::string("</function>\n</tool_call>")}) {
            const auto start = tree.size();
            auto parent = 0;
            for (int token : tokenizer->encode(suffix)) {
                parents.push_back(parent);
                parent = tree.size();
                tree.push_back(token);
            }
            branches.emplace_back(start, tree.size());
        }
        std::vector<std::uint32_t> tree_mask(tree.size() * fi::ToolGrammarState::mask_words);
        state.fill_masks(tree, parents, tree_mask);
        for (std::size_t branch = 0; branch < branches.size(); ++branch) {
            bool rejected = false;
            for (auto node = branches[branch].first; node < branches[branch].second; ++node) {
                if (!(tree_mask[parents[node] * fi::ToolGrammarState::mask_words + tree[node] / 32] &
                      (1u << (tree[node] % 32)))) { rejected = true; break; }
            }
            if (rejected != (branch != 0)) {
                std::cerr << "real speculative unordered branch mask incorrect\n";
                ++failures;
            }
        }
        state.preview(tokenizer->encode(valid_suffix));
        state.discard_preview();
        auto repeated_mask = tree_mask;
        state.fill_masks(tree, parents, repeated_mask);
        if (repeated_mask != tree_mask) {
            std::cerr << "discard changed unordered property state\n";
            ++failures;
        }
        state.preview(tokenizer->encode(valid_suffix));
        state.commit_preview();
        state.fill_masks(std::vector<ninfer::TokenId>{0}, std::vector<std::int32_t>{-1},
                         std::span<std::uint32_t>(repeated_mask).first(fi::ToolGrammarState::mask_words));
        for (int stop : tokenizer->default_stop_token_ids()) {
            if (!(repeated_mask[stop / 32] & (1u << (stop % 32)))) {
                std::cerr << "committed unordered call cannot stop\n";
                ++failures;
            }
        }
    }
    {
        xgrammar::GrammarMatcher premature(grammar->compiled);
        bool masked = false;
        const auto call = text.substr(text.find("<tool_call>"));
        for (int token : tokenizer->encode("Let me make the call.\n" + call + "\n" + call)) {
            (void)premature.FillNextTokenBitmask(&mask);
            if (!(words[token / 32] & (1u << (token % 32)))) { masked = true; break; }
            if (!premature.AcceptToken(token)) {
                throw std::runtime_error("reasoning mask and matcher disagree");
            }
        }
        if (!masked) { std::cerr << "real tokenizer can rehearse calls inside reasoning\n"; ++failures; }
    }
    // Real Qwen BPE can put prose, framing and a closing tag in one token.
    // Check the production mask itself, not just string matching, on both
    // captured tails and on a full valid frame followed by those tails.
    std::vector<std::string> tails{"\n</function>\n</tool_call>",
        "\n</invoke>\n\n</parameter>\n</function>\n</tool_call>",
        "\n</invoke\n\n</parameter=\n</function\n", "</function = null>",
        "<parameter=limit>\n80\n", "<tool_call = null>"};
    std::vector<std::string> before_tails{text,
        "Think.</think>\nNow I understand the validator's contract.\n"};
    if (fixture.contains("malformed_tail")) {
        tails.push_back(fixture.at("malformed_tail").get<std::string>());
        before_tails.push_back("Think.</think>\n"+fixture.at("valid_prefix").get<std::string>());
    }
    for (const auto& tail : tails) {
        for (const auto& before : before_tails) {
            xgrammar::GrammarMatcher probe(grammar->compiled);
            bool masked = false;
            for (int token : tokenizer->encode(before + tail)) {
                (void)probe.FillNextTokenBitmask(&mask);
                if (!(words[token / 32] & (1u << (token % 32)))) { masked = true; break; }
                if (!probe.AcceptToken(token)) {
                    std::cerr << "mask licensed an orphan token rejected by the matcher\n";
                    return 1;
                }
            }
            if (!masked) { std::cerr << "real tokenizer orphan tail was not masked\n"; ++failures; }
        }
    }
    // The product supports this represented Qwen vocabulary, not arbitrary
    // synthetic token tables. At the sensitive framing boundaries, compare
    // every regular token's optimized mask with byte-wise grammar acceptance.
    // This also covers real tokens straddling a frame boundary in either direction.
    const auto info = grammar->compiled.GetTokenizerInfo();
    std::size_t compared = 0;
    for (const auto& prefix : {std::string("Thinking about the next action. "),
                               std::string("Think.</think>\n"),
                               text.substr(0, text.rfind("\n</parameter>")), text,
                               text+"</fun", unordered_prefix,
                               unordered_prefix+"<parameter="}) {
        xgrammar::GrammarMatcher boundary(grammar->compiled);
        if (!boundary.AcceptString(prefix)) { throw std::runtime_error("invalid probe prefix"); }
        (void)boundary.FillNextTokenBitmask(&mask);
        for (const auto& [id, bytes] : info.GetSortedDecodedVocab()) {
            if (bytes.empty()) { continue; }
            const bool allowed = (words[id / 32] & (1u << (id % 32))) != 0;
            auto direct = boundary.Fork();
            const bool accepted = direct.AcceptString(bytes);
            ++compared;
            if (allowed != accepted) {
                if (failures < 10) {
                    std::cerr << "real vocabulary mask mismatch token=" << id
                              << " allowed=" << allowed << " accepted=" << accepted << '\n';
                }
                ++failures;
            }
        }
    }
    std::cout << "real tool grammar tools=" << tools.size() << " tokens=" << ids.size()
              << " compile_ms=" << std::chrono::duration<double, std::milli>(compiled - begin).count()
              << " match_ms=" << std::chrono::duration<double, std::milli>(valid_done - compiled).count()
              << " boundary_tokens_compared=" << compared << " failures=" << failures << '\n';
    return failures ? 1 : 0;
}

int main(int argc, char** argv) {
    if (argc == 3) { return real_tokenizer_probe(argv[1], argv[2]); }
    if (argc != 1) { return 2; }
    const std::vector<std::string> pieces{
        "<tool_", "call>\n<function=read>\n<parameter=filePath>\n", "true",
        "\n</parameter>\n<parameter=limit>\n", "64", "\n</parameter>\n</function>\n</tool_call>",
        "<eos>", "abc", "</parameter>\n</parameter>", "Reasoning about a call.", "</think>\n\n",
        "</function>", "</invoke>", "</function", ">", "</parameter>", "</tool_call>",
        "</fun", "ction", "ction = null>", "</invoke\n\n</parameter=",
        "<parameter=limit>\n", "<function = null>", "call = null>"};
    auto added = Json::array();
    for (std::size_t i = 0; i < pieces.size(); ++i) {
        added.push_back(Json{{"id", i + 1}, {"content", pieces[i]}, {"single_word", false},
            {"lstrip", false}, {"rstrip", false}, {"normalized", false}, {"special", i == 6}});
    }
    const auto tokenizer_json = Json{{"model", Json{{"type", "BPE"}, {"vocab", Json{{"x", 0}}},
                                                   {"merges", Json::array()}}},
                                     {"added_tokens", added}}.dump();
    auto decoder = Json::object();
    for (auto token : added) {
        const auto id = std::to_string(token.at("id").get<int>());
        token.erase("id");
        decoder[id] = std::move(token);
    }
    const auto config_json = Json{{"added_tokens_decoder", decoder}}.dump();
    auto tokenizer = std::make_shared<const fi::Tokenizer>(fi::TokenizerResources{
        tokenizer_json, config_json, R"({"eos_token_id":[7]})"});
    fi::ToolGrammarCompiler compiler(tokenizer);
    const std::vector<std::string> tools{R"({"type":"function","function":{"name":"read","parameters":{"type":"object","properties":{"filePath":{"type":"string"},"limit":{"type":"integer","minimum":1,"maximum":64}},"required":["filePath","limit"],"additionalProperties":false}}})"};
    int failures = 0;
    auto check = [&](bool condition, const char* label) {
        if (!condition) { ++failures; std::cerr << label << '\n'; }
    };
    auto schema_grammar = [&](const Json& schema) {
        return compiler.compile(std::vector<std::string>{Json{{"type", "function"},
            {"function", Json{{"name", "check"}, {"parameters", schema}}}}.dump()}, false);
    };
    // Independent finite-domain oracle: enumerate every key sequence, including
    // omissions, repetitions, unknown keys, all permutations and property bounds.
    // Integer values keep framing unambiguous; raw XML strings are tested below.
    std::size_t order_cases = 0;
    for (const auto bounds : {std::pair{0, 3}, std::pair{2, 2}, std::pair{3, 3}}) {
        Json schema{{"type", "object"}, {"properties", Json{
            {"a", Json{{"type", "integer"}, {"minimum", 1}, {"maximum", 2}}},
            {"b", Json{{"const", 3}}}, {"c", Json{{"enum", {4, 5}}}}}},
            {"required", {"b"}}, {"additionalProperties", false},
            {"minProperties", bounds.first}, {"maxProperties", bounds.second}};
        auto grammar = schema_grammar(schema);
        for (int length = 0, combinations = 1; length <= 4; ++length, combinations *= 4) {
            for (int code = 0; code < combinations; ++code) {
                int remaining = code;
                std::set<int> seen;
                bool valid = length >= bounds.first && length <= bounds.second;
                std::vector<std::pair<std::string, std::string>> arguments;
                for (int i = 0; i < length; ++i, remaining /= 4) {
                    const int key = remaining % 4;
                    valid = seen.insert(key).second && key < 3 && valid;
                    arguments.emplace_back(std::string(1, 'a' + key), std::to_string(key + 2));
                }
                valid = valid && seen.contains(1);
                const auto text = envelope("check", arguments);
                xgrammar::GrammarMatcher matcher(grammar->compiled);
                const bool accepted = matcher.AcceptString(text) && matcher.AcceptToken(7);
                check(accepted == valid, "unordered generation differs from finite-domain oracle");
                check(grammar->decode_call(text).has_value() == valid,
                      "unordered typed decoding differs from finite-domain oracle");
                ++order_cases;
            }
        }
    }
    auto accepts_call = [&](const Json& schema,
                            const std::vector<std::pair<std::string, std::string>>& args,
                            bool expected) {
        auto grammar = schema_grammar(schema);
        const auto text = envelope("check", args);
        xgrammar::GrammarMatcher matcher(grammar->compiled);
        const bool accepted = matcher.AcceptString(text) && matcher.AcceptToken(7);
        if (accepted != expected) {
            std::cerr << "schema=" << schema.dump() << " expected=" << expected << " text=" << text << '\n';
        }
        check(accepted == expected, "unordered schema generation mismatch");
        check(grammar->decode_call(text).has_value() == expected, "unordered schema decoding mismatch");
    };
    const Json numeric{{"type", "integer"}, {"minimum", 1}, {"maximum", 2}};
    Json open_schema{{"type", "object"}, {"properties", Json{{"a", numeric}, {"ab", numeric}}},
        {"required", {"a"}}, {"additionalProperties", Json{{"const", 9}}}};
    accepts_call(open_schema, {{"extra", "9"}, {"a", "2"}, {"ab", "1"}}, true);
    accepts_call(open_schema, {{"a", "2"}, {"extra", "9"}, {"ab", "1"}}, true);
    accepts_call(open_schema, {{"a", "2"}, {"a", "9"}}, false);
    accepts_call(open_schema, {{"a", "9"}}, false);
    accepts_call(open_schema, {{"extra", "9"}}, false);
    accepts_call(open_schema, {{"a", "2"}, {"extra", "8"}}, false);
    open_schema["minProperties"] = 3;
    open_schema["maxProperties"] = 3;
    accepts_call(open_schema, {{"extra", "9"}, {"a", "2"}, {"more", "9"}}, true);
    accepts_call(open_schema, {{"a", "2"}, {"extra", "9"}}, false);
    accepts_call(open_schema, {{"a", "2"}, {"ab", "1"}, {"extra", "9"}, {"more", "9"}}, false);
    Json optional{{"type", "object"}, {"properties", Json{{"a", numeric}, {"b", numeric}}},
                  {"additionalProperties", false}};
    accepts_call(optional, {}, true);
    accepts_call(optional, {{"b", "1"}, {"a", "2"}}, true);
    optional["maxProperties"] = 0;
    accepts_call(optional, {}, true);
    accepts_call(optional, {{"a", "2"}}, false);
    accepts_call(Json{{"type", "object"}, {"properties", Json{{"quo\"te", numeric},
        {"back\\slash", numeric}, {"café", numeric}}}, {"required", {"quo\"te", "back\\slash", "café"}},
        {"additionalProperties", false}},
        {{"café", "2"}, {"back\\slash", "1"}, {"quo\"te", "2"}}, true);
    const Json extra_string{{"type", "object"}, {"properties", Json{{"a", numeric}}},
        {"required", {"a"}}, {"additionalProperties", Json{{"type", "string"}}}};
    auto extra_string_grammar = schema_grammar(extra_string);
    const auto extra_string_text = envelope("check", {{"extra", "  indented\n\n"}, {"a", "2"}});
    xgrammar::GrammarMatcher extra_string_matcher(extra_string_grammar->compiled);
    check(extra_string_matcher.AcceptString(extra_string_text), "additional string framing rejected");
    const auto extra_string_call = extra_string_grammar->decode_call(extra_string_text);
    check(extra_string_call && Json::parse(extra_string_call->arguments_json)["extra"] == "  indented\n\n",
          "additional string lost represented whitespace");
    Json unlisted{{"type", "object"}, {"required", {"a"}}, {"additionalProperties", numeric}};
    accepts_call(unlisted, {{"extra", "1"}, {"a", "2"}}, true);
    accepts_call(unlisted, {{"extra", "1"}}, false);
    accepts_call(Json{{"type", "object"}, {"required", {"first", "second"}}},
                 {{"second", "2"}, {"first", "1"}}, true);
    const Json nested{{"type", "object"},
        {"properties", Json{{"x", numeric}, {"y", numeric}}},
        {"required", {"x", "y"}}, {"additionalProperties", false}};
    Json containers{{"type", "object"}, {"$defs", Json{{"pair", nested}}},
        {"properties", Json{{"object", Json{{"$ref", "#/$defs/pair"}}},
             {"array", Json{{"type", "array"}, {"items", Json{{"$ref", "#/$defs/pair"}}},
                            {"minItems", 1}, {"maxItems", 2}}}}},
        {"required", {"object", "array"}}, {"additionalProperties", false}};
    accepts_call(containers, {{"array", R"([{"y":2,"x":1}])"},
                             {"object", R"({"y":1,"x":2})"}}, true);
    accepts_call(containers, {{"array", R"([{"y":2}])"},
                             {"object", R"({"y":1,"x":2})"}}, false);
    accepts_call(containers, {{"array", R"([{"y":2,"x":1}])"},
                             {"object", R"({"y":3,"x":2})"}}, false);
    accepts_call(containers, {{"array", R"([{"y":2,"x":1}])"},
                             {"object", R"({"y":1,"x":2,"y":2})"}}, false);
    // Upstream's additional JSON key rule admits escaped aliases. They must
    // never become executable calls after canonical decoding, even when the
    // overwritten value alone would satisfy the named property's schema.
    auto escaped = containers;
    escaped["$defs"]["pair"]["additionalProperties"] = numeric;
    auto escaped_grammar = schema_grammar(escaped);
    const auto escaped_duplicate = envelope("check", {{"array", R"([{"y":2,"x":1}])"},
        {"object", R"({"y":1,"x":1,"\u0078":2})"}});
    check(!escaped_grammar->decode_call(escaped_duplicate),
          "escaped duplicate nested key was published after lossy canonicalization");
    escaped["$defs"]["pair"]["additionalProperties"] = Json{{"const", 9}};
    escaped["$defs"]["pair"]["required"] = {"y"};
    escaped_grammar = schema_grammar(escaped);
    check(!escaped_grammar->decode_call(envelope("check", {{"array", R"([{"y":2,"x":1}])"},
        {"object", R"({"y":1,"\u0078":9})"}})),
          "escaped named key bypassed its value bound during final validation");
    const Json branch1{{"type", "object"}, {"properties", Json{
        {"a", Json{{"const", 1}}}, {"b", Json{{"const", 2}}}}},
        {"required", {"a", "b"}}, {"additionalProperties", false}};
    auto branch2 = branch1;
    branch2["properties"]["a"]["const"] = 3;
    branch2["properties"]["b"]["const"] = 4;
    const Json alternatives_schema{{"anyOf", {branch1, branch2}}};
    accepts_call(alternatives_schema, {{"b", "2"}, {"a", "1"}}, true);
    accepts_call(alternatives_schema, {{"b", "4"}, {"a", "3"}}, true);
    accepts_call(alternatives_schema, {{"b", "2"}, {"a", "3"}}, false);
    std::cout << "unordered finite-domain cases=" << order_cases << '\n';
    for (bool reasoning : {false, true}) {
        auto grammar = compiler.compile(tools, reasoning);
        std::string raw;
        for (int i = 0; i < 6; ++i) { raw += pieces[i]; }
        const std::string reasoning_prefix = reasoning ? "Reasoning about a call.</think>\n\n" : "";
        // These exact tails escaped in real SSE captures. They must be invalid
        // outside envelopes, regardless of whether a legitimate call precedes
        // them. Normal prose and subsequent complete calls remain licensed.
        for (const auto& tail : {std::string("</invoke>"), std::string("</parameter>"),
                                 std::string("</function>"), std::string("</tool_call>"),
                                 std::string("\n</function>\n</tool_call>"),
                                 std::string("\n</invoke>\n\n</parameter>\n</function>\n</tool_call>"),
                                 std::string("</invoke\n\n</parameter=\n</function\n\n<parameter=limit>\n80\n"),
                                 std::string("</function = null>"), std::string("</function"),
                                 std::string("<function = null>"), std::string("<parameter=limit>\n"),
                                 std::string("<invoke name=read>"), std::string("<tool_call = null>")}) {
            for (const auto& before : {reasoning_prefix + "Plain prose.\n",
                                      reasoning_prefix + "Plain prose.\n" + raw + "\n"}) {
                xgrammar::GrammarMatcher whole(grammar->compiled);
                check(!whole.AcceptString(before + tail), "orphan protocol close admitted in prose");
                xgrammar::GrammarMatcher split(grammar->compiled);
                check(split.AcceptString(before), "valid prefix before orphan tail rejected");
                bool rejected_tail = false;
                for (const char c : tail) {
                    if (!split.AcceptString(std::string(1, c))) { rejected_tail = true; break; }
                }
                check(rejected_tail, "byte-split orphan protocol close admitted");
            }
        }
        xgrammar::GrammarMatcher multiple(grammar->compiled);
        check(multiple.AcceptString(reasoning_prefix + "<div>ordinary XML</div>\n" + raw +
                                    "\nNext action.\n" + raw),
              "orphan exclusion blocked ordinary XML, inter-call prose, or a second valid call");
        auto literal_raw=raw;
        const std::string literal="literal </invoke </function = null> <parameter=limit> <tool_call = null>";
        literal_raw.replace(literal_raw.find("true"),4,literal);
        xgrammar::GrammarMatcher literal_argument(grammar->compiled);
        check(literal_argument.AcceptString(reasoning_prefix+literal_raw) &&
                  literal_argument.AcceptToken(7),
              "protocol prefix exclusion affected a schema-valid string argument");
        const auto literal_call=grammar->decode_call(literal_raw);
        check(literal_call && Json::parse(literal_call->arguments_json)["filePath"]==literal,
              "protocol prefix literal was not preserved in the decoded argument");
        xgrammar::GrammarMatcher prose_only(grammar->compiled);
        check(prose_only.AcceptString(reasoning_prefix + "The task is complete.") &&
                  prose_only.AcceptToken(7),
              "orphan exclusion forced a tool call instead of allowing an ordinary answer");
        if (reasoning) {
            xgrammar::GrammarMatcher reasoning_literal(grammar->compiled);
            check(reasoning_literal.AcceptString("Discuss </invoke> and </function> in reasoning.</think>" + raw),
                  "prose exclusion changed the reasoning contract");
            xgrammar::GrammarMatcher premature(grammar->compiled);
            check(!premature.AcceptString("Let me use the tool.\n" + raw),
                  "tool envelope admitted inside reasoning");
            xgrammar::GrammarMatcher reasoning_exit(grammar->compiled);
            check(reasoning_exit.AcceptString("Let me use the tool.</think>\n" + raw),
                  "reasoning cannot exit into a legitimate tool call");
            fi::ToolGrammarState thinking_state(grammar);
            std::vector<std::uint32_t> thinking_words(2 * fi::ToolGrammarState::mask_words);
            thinking_state.fill_masks(std::vector<ninfer::TokenId>{0, 1},
                                      std::vector<std::int32_t>{-1, 0}, thinking_words);
            check((thinking_words[fi::ToolGrammarState::mask_words] & (1u << 2)) == 0,
                  "speculative child can complete a tool opener inside reasoning");
            thinking_state.preview(std::vector<ninfer::TokenId>{1});
            thinking_state.commit_preview();
            thinking_state.fill_masks(std::vector<ninfer::TokenId>{0},
                                      std::vector<std::int32_t>{-1},
                                      std::span<std::uint32_t>(thinking_words).first(fi::ToolGrammarState::mask_words));
            check((thinking_words[0] & (1u << 2)) == 0,
                  "next committed round can complete a tool opener inside reasoning");
        }
        const auto call = grammar->decode_call(raw);
        check(call.has_value(), "valid complete tool failed typed decoding");
        if (call) {
            const auto args = Json::parse(call->arguments_json);
            check(args.at("filePath").is_string() && args.at("filePath") == "true",
                  "declared string was coerced into boolean");
            check(args.at("limit").is_number_integer() && args.at("limit") == 64,
                  "integer parameter was not decoded as integer");
        }
        auto bad = raw;
        bad.replace(bad.find("64"), 2, "65");
        check(!grammar->decode_call(bad), "completed-call validation missed numeric bound");
        auto whitespace = raw;
        whitespace.replace(whitespace.find("true"), 4, "  indented\n\n");
        const auto preserved = grammar->decode_call(whitespace);
        check(preserved && Json::parse(preserved->arguments_json)["filePath"] == "  indented\n\n",
              "tool string lost indentation or trailing newlines");
        xgrammar::GrammarMatcher matcher(grammar->compiled);
        std::vector<std::uint32_t> words((ninfer::targets::qwen3_6::kTokenDomain + 31) / 32);
        std::int64_t shape = words.size();
        DLTensor mask{};
        mask.data = words.data();
        mask.device = {kDLCPU, 0};
        mask.ndim = 1;
        mask.dtype = {kDLInt, 32, 1};
        mask.shape = &shape;
        auto allows = [&](int token) {
            (void)matcher.FillNextTokenBitmask(&mask);
            return (words[token / 32] & (1u << (token % 32))) != 0;
        };
        if (reasoning) {
            check(!allows(7), "grammar admitted EOS inside reasoning");
            check(matcher.AcceptToken(10), "reasoning text rejected");
            check(allows(11) && matcher.AcceptToken(11), "reasoning terminator rejected");
        }
        for (int token : {1, 2, 3, 4, 5, 6}) {
            if (token == 5) {
                check(!allows(8), "nonnumeric limit admitted");
                check(!allows(9), "extra parameter close admitted");
            }
            const bool permitted = allows(token);
            if (!permitted) {
                std::cerr << "masked valid token " << token << ": " << pieces[token - 1] << '\n';
            }
            check(permitted, "valid multi-boundary token masked");
            check(matcher.AcceptToken(token), "valid multi-boundary token rejected");
        }
        check(!allows(12) && !allows(13) && !allows(14) && !allows(21) &&
                  !allows(22) && !allows(23),
              "captured orphan tail token was not masked after a call");
        check(allows(7) && matcher.AcceptToken(7), "complete tool cannot terminate");
        matcher.Rollback(1);
        check(allows(7), "grammar rollback failed to restore terminal choice");

        fi::ToolGrammarState state(grammar);
        std::vector<ninfer::TokenId> prefix;
        if (reasoning) { prefix = {10, 11}; }
        prefix.insert(prefix.end(), {1, 2, 3, 4});
        state.preview(prefix);
        state.commit_preview();
        // The valid path is 0 -> 2 -> 4. Node number, not depth or
        // column adjacency, determines which grammar state licenses sampling.
        const std::vector<ninfer::TokenId> draft{0, 8, 5, 9, 6};
        const std::vector<std::int32_t> parents{-1, 0, 0, 2, 2};
        std::vector<std::uint32_t> tree_words(draft.size() * fi::ToolGrammarState::mask_words);
        auto tree_allows = [&](int node, int token) {
            return (tree_words[node * fi::ToolGrammarState::mask_words + token / 32] &
                    (1u << (token % 32))) != 0;
        };
        state.fill_masks(draft, parents, tree_words);
        check(tree_allows(0, 5) && !tree_allows(0, 8), "root grammar state was not applied");
        check(tree_allows(2, 6) && !tree_allows(2, 9), "tree child used another node's grammar");
        check(tree_allows(4, 7), "tree bonus did not use complete-call grammar");
        check(!tree_allows(4, 12) && !tree_allows(4, 13),
              "speculative bonus admitted an orphan tail after a complete call");
        check(!tree_allows(4, 16) && !tree_allows(4, 17),
              "speculative bonus admitted parameter or tool closing token as prose");
        check(!tree_allows(4,14) && !tree_allows(4,21) && !tree_allows(4,22),
              "speculative bonus admitted an incomplete protocol prefix");
        check(tree_allows(1, 8) && tree_allows(3, 9), "unreachable node has an empty domain");
        const std::vector<ninfer::TokenId> suffix{5, 6};
        state.preview(suffix);
        state.discard_preview();
        state.fill_masks(draft, parents, tree_words);
        check(tree_allows(0, 5) && !tree_allows(0, 7), "discard advanced committed grammar");
        bool bad_preview = false;
        try { state.preview(std::vector<ninfer::TokenId>{8}); }
        catch (const std::logic_error&) { bad_preview = true; }
        check(bad_preview, "invalid preview was accepted");
        state.preview(suffix);
        state.commit_preview();
        std::vector<std::uint32_t> root_words(fi::ToolGrammarState::mask_words);
        state.fill_masks(std::vector<ninfer::TokenId>{0},
                         std::vector<std::int32_t>{-1}, root_words);
        check((root_words[0] & (1u << 7)) != 0, "committed tool cannot terminate");
        check((root_words[0] & ((1u << 12) | (1u << 13))) == 0,
              "committed call root admitted orphan closing tokens");
        // A delimiter split over decode rounds must be constrained using its
        // committed byte prefix, and preview rollback must restore that state.
        state.preview(std::vector<ninfer::TokenId>{18});
        state.commit_preview();
        state.fill_masks(std::vector<ninfer::TokenId>{0},
                         std::vector<std::int32_t>{-1}, root_words);
        check((root_words[0] & ((1u << 19) | (1u << 20))) == 0,
              "split orphan prefix can be completed or escaped next round");
        // Use a fresh completed-call state for the following EOS controls.
        state = fi::ToolGrammarState(grammar);
        state.preview(prefix);
        state.commit_preview();
        state.preview(suffix);
        state.commit_preview();
        std::vector<std::uint32_t> stopped_words(3 * fi::ToolGrammarState::mask_words);
        state.fill_masks(std::vector<ninfer::TokenId>{0, 7, 8},
                         std::vector<std::int32_t>{-1, 0, 1}, stopped_words);
        check((stopped_words[fi::ToolGrammarState::mask_words] & (1u << 8)) != 0,
              "post-EOS speculative column tried to continue a terminated matcher");
        fi::ToolGrammarState ignored_stop(grammar, {7});
        ignored_stop.preview(prefix);
        ignored_stop.commit_preview();
        bool partial_stop = false;
        try { ignored_stop.preview(std::vector<ninfer::TokenId>{7}); }
        catch (const std::logic_error&) { partial_stop = true; }
        check(partial_stop, "disabled EOS was admitted inside an incomplete call");
        ignored_stop.preview(std::vector<ninfer::TokenId>{5, 6, 7, 7});
        ignored_stop.commit_preview();
        ignored_stop.fill_masks(std::vector<ninfer::TokenId>{0, 7, 1},
                                std::vector<std::int32_t>{-1, 0, 1}, stopped_words);
        check((stopped_words[fi::ToolGrammarState::mask_words] & (1u << 1)) != 0,
              "disabled EOS prevented a subsequent tool call");
        ignored_stop.preview(std::vector<ninfer::TokenId>{1, 2, 3, 4, 5, 6, 7});
        ignored_stop.commit_preview();
    }
    auto unsupported = Json::parse(tools[0]);
    unsupported["function"]["parameters"]["not"] = Json{{"required", {"filePath"}}};
    bool rejected = false;
    try { (void)compiler.compile(std::vector<std::string>{unsupported.dump()}, false); }
    catch (const ninfer::RequestError& error) {
        rejected = error.kind() == ninfer::RequestErrorKind::InvalidToolSchema;
    }
    check(rejected, "unsupported assertion was silently ignored");
    for (const std::string malformed : {"{", R"({"type":"function","function":{"name":7}})"}) {
        bool input_error = false;
        try { (void)compiler.compile(std::vector<std::string>{malformed}, false); }
        catch (const ninfer::RequestError& error) {
            input_error = error.kind() == ninfer::RequestErrorKind::InvalidToolSchema;
        }
        check(input_error, "malformed tool declaration lost its typed input error");
    }
    for (const Json& invalid : {
             Json{{"type", "object"}, {"properties", Json{{"x", Json{{"type", "string"},
                                                                     {"pattern", "["}}}}}},
             Json{{"$ref", "#/$defs/missing"}}}) {
        bool input_error = false;
        try { (void)schema_grammar(invalid); }
        catch (const ninfer::RequestError& error) {
            input_error = error.kind() == ninfer::RequestErrorKind::InvalidToolSchema &&
                          std::string(error.what()).find("invalid or unsupported tool schema:") !=
                              std::string::npos;
        }
        catch (const std::exception& error) {
            std::cerr << "schema compiler exposed a server error: " << error.what() << '\n';
        }
        check(input_error, "malformed schema compilation did not become a client input error");
    }
    for (const char* bound : {"minLength", "maxLength"}) {
        auto constrained = Json::parse(tools[0]);
        auto& property = constrained["function"]["parameters"]["properties"]["filePath"];
        property["pattern"] = "^[a-z]+$";
        property[bound] = 4;
        rejected = false;
        try { (void)compiler.compile(std::vector<std::string>{constrained.dump()}, false); }
        catch (const ninfer::RequestError& error) {
            rejected = error.kind() == ninfer::RequestErrorKind::InvalidToolSchema;
        }
        check(rejected, "pattern silently overrode a string length assertion");
    }
    auto alternatives = Json::parse(tools[0]);
    const auto string_branch = alternatives["function"]["parameters"];
    auto integer_branch = string_branch;
    integer_branch["properties"]["filePath"] = Json{{"type", "integer"}};
    alternatives["function"]["parameters"] = Json{{"anyOf", {integer_branch, string_branch}}};
    auto union_grammar = compiler.compile(std::vector<std::string>{alternatives.dump()}, false);
    std::string union_call;
    for (int i = 0; i < 6; ++i) { union_call += pieces[i]; }
    xgrammar::GrammarMatcher union_matcher(union_grammar->compiled);
    check(union_matcher.AcceptString(union_call), "root alternative rejected a valid Qwen envelope");
    const auto decoded_union = union_grammar->decode_call(union_call);
    check(decoded_union && Json::parse(decoded_union->arguments_json)["filePath"] == "true",
          "root schema alternatives lost the declared string type");
    auto scalar_union = Json::parse(tools[0]);
    scalar_union["function"]["parameters"]["properties"]["filePath"] = Json{
        {"anyOf", {Json{{"type", "integer"}, {"minimum", 2}}, Json{{"const", "1"}}}}};
    auto scalar_grammar = compiler.compile(std::vector<std::string>{scalar_union.dump()}, false);
    auto scalar_call = union_call;
    scalar_call.replace(scalar_call.find("true"), 4, "1");
    xgrammar::GrammarMatcher scalar_matcher(scalar_grammar->compiled);
    check(scalar_matcher.AcceptString(scalar_call), "valid scalar-alternative envelope rejected");
    const auto scalar = scalar_grammar->decode_call(scalar_call);
    check(scalar && Json::parse(scalar->arguments_json)["filePath"] == "1",
          "scalar union decoded a disallowed integer instead of its valid string alternative");
    scalar_call.replace(scalar_call.find("\n1\n"), 3, "\n3\n");
    const auto decoded_numeric = scalar_grammar->decode_call(scalar_call);
    check(decoded_numeric && Json::parse(decoded_numeric->arguments_json)["filePath"] == 3,
          "scalar union did not retain the valid numeric alternative");
    auto embedded = Json::parse(tools[0]);
    const std::string literal = "<tool_call>\n</invoke>\n</parameter>\n<parameter=limit>\n12\n</parameter>\n</function>\n</tool_call>";
    embedded["function"]["parameters"]["properties"]["filePath"] = Json{{"const", literal}};
    auto embedded_grammar = compiler.compile(std::vector<std::string>{embedded.dump()}, false);
    auto embedded_call = union_call;
    embedded_call.replace(embedded_call.find("true"), 4, literal);
    xgrammar::GrammarMatcher embedded_matcher(embedded_grammar->compiled);
    check(embedded_matcher.AcceptString(embedded_call), "declared XML-looking string was not generatable");
    const auto embedded_value = embedded_grammar->decode_call(embedded_call);
    check(embedded_value && Json::parse(embedded_value->arguments_json)["filePath"] == literal,
          "literal XML-looking argument was truncated or interpreted as another parameter");
    std::cout << "tool grammar failures=" << failures << '\n';
    return failures ? 1 : 0;
}
