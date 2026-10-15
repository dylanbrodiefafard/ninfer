#include <xgrammar/xgrammar.h>

#include <cstdint>
#include <iostream>
#include <optional>
#include <set>
#include <string>
#include <stdexcept>
#include <utility>
#include <vector>

namespace {

xgrammar::CompiledGrammar compile(xgrammar::GrammarCompiler& compiler, const std::string& schema,
                                  bool any_order) {
    return compiler.CompileJSONSchema(schema, false, std::nullopt,
                                      std::pair<std::string, std::string>{",", ":"}, true,
                                      std::nullopt, any_order);
}

int finite_language(bool any_order) {
    const std::string schema = R"({
        "type":"object",
        "properties":{
            "a":{"enum":[0,1]},
            "b":{"enum":["é",null]},
            "c":{"enum":[true]}
        },
        "required":["a"],"minProperties":2,"maxProperties":2,
        "additionalProperties":false
    })";
    std::set<std::string> words;
    for (const std::string a : {"0", "1"}) {
        for (const std::string tail : {"\"b\":\"é\"", "\"b\":null", "\"c\":true"}) {
            std::string ordered("{\"a\":");
            ordered.append(a).append(",").append(tail).append("}");
            words.insert(std::move(ordered));
            if (any_order) {
                std::string reversed("{");
                reversed.append(tail).append(",\"a\":").append(a).append("}");
                words.insert(std::move(reversed));
            }
        }
    }
    const std::vector<std::string> vocab{"{",
                                         "}",
                                         ",",
                                         ":",
                                         "\"",
                                         "a",
                                         "b",
                                         "c",
                                         "0",
                                         "1",
                                         "é",
                                         "null",
                                         "true",
                                         "false",
                                         "\"a\":",
                                         "0,\"b\":",
                                         "\"é\"}",
                                         "null}",
                                         "true,\"a\":1}",
                                         "{\"b\":null,\"a\":",
                                         "{\"a\":0,\"b\":null}",
                                         " ",
                                         "<eos>"};
    const int stop = static_cast<int>(vocab.size() - 1);
    xgrammar::GrammarCompiler compiler(xgrammar::TokenizerInfo(vocab, xgrammar::VocabType::RAW,
                                                               static_cast<int>(vocab.size()),
                                                               std::vector<std::int32_t>{stop}),
                                       1, false);
    const auto grammar      = compile(compiler, schema, any_order);
    const auto can_complete = [&](const std::string& prefix) {
        const auto next = words.lower_bound(prefix);
        return next != words.end() && next->starts_with(prefix);
    };
    std::set<std::string> prefixes{"",
                                   "{}",
                                   "{\"a\":0}",
                                   "{\"b\":null}",
                                   "{\"a\":0,\"a\":1}",
                                   "{\"a\":0,\"d\":null}",
                                   "{\"a\":0,\"b\":null,\"c\":true}",
                                   "{\"b\":null,\"a\":0}"};
    // A finite schema supplies an independent, exhaustive completion oracle.
    // Include every UTF-8 character boundary and token spellings crossing rules.
    for (const auto& word : words) {
        for (std::size_t i = 0; i <= word.size(); ++i) {
            if (i != word.size() && (static_cast<unsigned char>(word[i]) & 0xc0u) == 0x80u) {
                continue;
            }
            prefixes.insert(word.substr(0, i));
        }
    }
    int failures = 0;
    for (const auto& prefix : prefixes) {
        xgrammar::GrammarMatcher matcher(grammar);
        const bool accepted = matcher.AcceptString(prefix);
        const bool complete = words.contains(prefix);
        if (accepted != can_complete(prefix) || (accepted && matcher.IsCompleted()) != complete) {
            std::cerr << "schema language mismatch: order=" << any_order << " " << prefix << '\n';
            ++failures;
        }
        if (!accepted) { continue; }
        std::vector<std::uint32_t> bits((vocab.size() + 31) / 32);
        std::int64_t shape = static_cast<std::int64_t>(bits.size());
        DLTensor mask{.data        = bits.data(),
                      .device      = {kDLCPU, 0},
                      .ndim        = 1,
                      .dtype       = {kDLInt, 32, 1},
                      .shape       = &shape,
                      .strides     = nullptr,
                      .byte_offset = 0};
        (void)matcher.FillNextTokenBitmask(&mask);
        for (int token = 0; token <= stop; ++token) {
            const bool expected = token == stop ? complete : can_complete(prefix + vocab[token]);
            auto branch         = matcher.Fork();
            const bool masked   = (bits[token / 32] & (1u << (token % 32))) != 0;
            if (masked != expected || branch.AcceptToken(token) != expected) {
                std::cerr << "schema token mismatch: order=" << any_order << " " << prefix
                          << " token=" << token << '\n';
                ++failures;
            }
        }
    }
    return failures;
}

int fixtures(const std::string& schema, const std::vector<std::string>& valid,
             const std::vector<std::string>& invalid, bool any_order) {
    xgrammar::GrammarCompiler compiler(xgrammar::TokenizerInfo({}, xgrammar::VocabType::RAW, 0), 1,
                                       false);
    const auto grammar  = compile(compiler, schema, any_order);
    const auto rendered = compiler.CompileGrammar(grammar.GetGrammar().ToString());
    int failures        = 0;
    for (const auto& [examples, expected] : {std::pair{&valid, true}, std::pair{&invalid, false}}) {
        for (const auto& text : *examples) {
            xgrammar::GrammarMatcher matcher(grammar);
            xgrammar::GrammarMatcher diagnostic(rendered);
            const bool complete = matcher.AcceptString(text) && matcher.IsCompleted();
            const bool diagnostic_complete =
                diagnostic.AcceptString(text) && diagnostic.IsCompleted();
            if (complete != expected || diagnostic_complete != expected) {
                std::cerr << "schema fixture mismatch: order=" << any_order
                          << " expected=" << expected << " " << text << '\n';
                ++failures;
            }
        }
    }
    return failures;
}

int compiler_errors() {
    xgrammar::GrammarCompiler compiler(xgrammar::TokenizerInfo({}, xgrammar::VocabType::RAW, 0), 1,
                                       true);
    int failures = 0;
    for (const std::string schema :
         {R"({"$ref":"#/$defs/missing"})", R"({"type":"string","pattern":"["})", "{"}) {
        bool rejected = false;
        try {
            (void)compile(compiler, schema, true);
        } catch (const std::runtime_error&) { rejected = true; }
        const auto recovered = compile(compiler, R"({"const":7})", false);
        xgrammar::GrammarMatcher matcher(recovered);
        if (!rejected || !matcher.AcceptString("7") || !matcher.IsCompleted()) {
            std::cerr << "schema compilation error corrupted subsequent compilation\n";
            ++failures;
        }
    }
    for (const std::string source : {"root ::= \"a\"\nroot ::= \"b\"\n", "root ::= missing\n"}) {
        bool rejected = false;
        try {
            (void)compiler.CompileGrammar(source);
        } catch (const std::runtime_error&) { rejected = true; }
        const auto recovered = compiler.CompileGrammar("root ::= \"ok\"");
        xgrammar::GrammarMatcher matcher(recovered);
        if (!rejected || !matcher.AcceptString("ok") || !matcher.IsCompleted()) {
            std::cerr << "EBNF parser error corrupted subsequent compilation\n";
            ++failures;
        }
    }
    return failures;
}

} // namespace

int main() {
    int failures = finite_language(false) + finite_language(true) + compiler_errors();
    const std::string recursive = R"({
        "$defs":{
            "a1":{"enum":["left"]},
            "a2":{"enum":["right"]},
            "node":{
                "type":"object",
                "properties":{
                    "value":{"enum":[0,1]},
                    "next":{"anyOf":[{"type":"null"},{"$ref":"#/$defs/node"}]}
                },
                "required":["value","next"],"additionalProperties":false
            }
        },
        "type":"object",
        "properties":{
            "left":{"$ref":"#/$defs/a1"},
            "right":{"$ref":"#/$defs/a2"},
            "tree":{"$ref":"#/$defs/node"}
        },
        "required":["left","right","tree"],"additionalProperties":false
    })";
    const std::vector<std::string> recursive_valid{
        R"({"left":"left","right":"right","tree":{"value":0,"next":null}})",
        R"({"left":"left","right":"right","tree":{"value":1,"next":{"value":0,"next":null}}})"};
    const std::vector<std::string> recursive_invalid{
        R"({"left":"right","right":"right","tree":{"value":0,"next":null}})",
        R"({"left":"left","right":"left","tree":{"value":0,"next":null}})",
        R"({"left":"left","right":"right","tree":{"value":0}})",
        R"({"left":"left","right":"right","tree":{"value":0,"next":false}})",
        R"({"left":"left","right":"right","tree":{"value":0,"value":1,"next":null}})"};
    const std::string quantified = R"({
        "type":"object",
        "properties":{
            "text":{"anyOf":[{"type":"string","minLength":1,"maxLength":2},{"type":"null"}]},
            "items":{"type":"array","items":{"enum":[0,1]},"minItems":1,"maxItems":2}
        },
        "required":["text","items"],"additionalProperties":false
    })";
    const std::vector<std::string> quantified_valid{
        R"({"text":"é","items":[0]})", R"({"text":"汉字","items":[0,1]})",
        R"({"text":"aé","items":[1,1]})", R"({"text":null,"items":[1]})"};
    const std::vector<std::string> quantified_invalid{
        R"({"text":"","items":[0]})",      R"({"text":"abc","items":[0]})",
        R"({"text":"汉字é","items":[0]})", R"({"text":"é","items":[]})",
        R"({"text":"é","items":[0,1,0]})", R"({"text":"é","items":[2]})",
        R"({"text":false,"items":[0]})",   R"({"text":null,"items":[0],"extra":0})"};
    for (const bool any_order : {false, true}) {
        failures += fixtures(recursive, recursive_valid, recursive_invalid, any_order);
        failures += fixtures(quantified, quantified_valid, quantified_invalid, any_order);
    }
    failures += fixtures(
        recursive,
        {R"({"tree":{"next":{"next":null,"value":0},"value":1},"right":"right","left":"left"})"},
        {}, true);
    failures += fixtures(quantified, {R"({"items":[1,0],"text":"é"})"}, {}, true);
    return failures == 0 ? 0 : 1;
}
