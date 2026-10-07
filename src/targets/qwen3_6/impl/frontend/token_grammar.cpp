#include "targets/qwen3_6/impl/frontend/token_grammar.h"
#include "ninfer/targets/qwen3_6/frontend.h"

#include <algorithm>
#include <cmath>
#include <cctype>
#include <set>
#include <limits>
#include <stdexcept>
#include <string_view>
#include <utility>

#if defined(__GLIBC__)
#    include <malloc.h>
#endif

namespace ninfer::targets::qwen3_6::frontend_internal {
namespace {
using Json = nlohmann::ordered_json;

bool annotation(std::string_view key) {
    return key == "$schema" || key == "$id" || key == "title" || key == "description" ||
           key == "default" || key == "examples" || key == "deprecated" || key == "readOnly" ||
           key == "writeOnly" || key == "$comment";
}

// Rewrites a client schema into one the grammar compiler enforces, accepting a superset of the
// client's instances so a valid call is never masked out; the client still validates the
// arguments it receives. `format` and the content keywords are annotations in JSON Schema 2020-12.
// Assertions that only narrow the instance set are dropped, `oneOf` widens to `anyOf`, and a
// single-branch `allOf` with annotation-only siblings is inlined. Keywords that grant instances
// (`patternProperties`, a multi-branch `allOf`) stay and are rejected by check_schema.
void relax_schema(Json& schema) {
    if (!schema.is_object()) { return; }
    // Inline first so the inlined keywords are relaxed below.
    if (schema.contains("allOf") && schema["allOf"].is_array() && schema["allOf"].size() == 1 &&
        schema["allOf"][0].is_object()) {
        const bool annotation_siblings =
            std::all_of(schema.items().begin(), schema.items().end(), [](const auto& item) {
                return item.key() == "allOf" || annotation(item.key());
            });
        if (annotation_siblings) {
            Json inner = std::move(schema["allOf"][0]);
            schema.erase("allOf");
            for (auto it = inner.begin(); it != inner.end(); ++it) {
                schema[it.key()] = it.value();
            }
        }
    }
    for (const char* key : {"format", "contentEncoding", "contentMediaType", "contentSchema",
                            "uniqueItems", "multipleOf", "propertyNames", "not", "contains",
                            "minContains", "maxContains", "dependentRequired", "dependentSchemas",
                            "if", "then", "else", "unevaluatedProperties", "unevaluatedItems"}) {
        schema.erase(key);
    }
    if (schema.contains("oneOf") && !schema.contains("anyOf")) {
        schema["anyOf"] = std::move(schema["oneOf"]);
        schema.erase("oneOf");
    }
    for (const char* map : {"properties", "$defs", "definitions"}) {
        if (schema.contains(map) && schema[map].is_object()) {
            for (auto& child : schema[map]) { relax_schema(child); }
        }
    }
    for (const char* child : {"items", "additionalProperties"}) {
        if (schema.contains(child)) { relax_schema(schema[child]); }
    }
    for (const char* list : {"prefixItems", "anyOf", "allOf"}) {
        if (schema.contains(list) && schema[list].is_array()) {
            for (auto& child : schema[list]) { relax_schema(child); }
        }
    }
}

void check_schema(const Json& schema) {
    if (schema.is_boolean()) { return; }
    if (!schema.is_object()) {
        throw std::invalid_argument("tool schema must be an object or boolean");
    }
    // XGrammar warns and ignores several JSON Schema assertions. Reject them at
    // admission instead of claiming that a successful grammar compile enforces them.
    static const std::set<std::string_view> supported{"type",
                                                      "properties",
                                                      "required",
                                                      "additionalProperties",
                                                      "items",
                                                      "prefixItems",
                                                      "minItems",
                                                      "maxItems",
                                                      "minProperties",
                                                      "maxProperties",
                                                      "minimum",
                                                      "maximum",
                                                      "exclusiveMinimum",
                                                      "exclusiveMaximum",
                                                      "minLength",
                                                      "maxLength",
                                                      "pattern",
                                                      "enum",
                                                      "const",
                                                      "$ref",
                                                      "$defs",
                                                      "definitions",
                                                      "anyOf"};
    for (auto it = schema.begin(); it != schema.end(); ++it) {
        if (!annotation(it.key()) && !supported.contains(it.key())) {
            throw std::invalid_argument("unsupported tool schema assertion: " + it.key());
        }
    }
    if (schema.contains("pattern") &&
        (schema.contains("minLength") || schema.contains("maxLength"))) {
        throw std::invalid_argument(
            "unsupported tool schema combination: pattern with string length bounds");
    }
    // These converter branches take precedence over sibling assertions. Only
    // annotations/definitions may accompany them; silently dropping siblings
    // would accept data that the client's schema rejects.
    for (const char* branch : {"$ref", "enum", "const", "anyOf"}) {
        if (!schema.contains(branch)) { continue; }
        for (auto it = schema.begin(); it != schema.end(); ++it) {
            if (it.key() != branch && it.key() != "$defs" && it.key() != "definitions" &&
                !annotation(it.key())) {
                // Common typed enums are checked independently before compilation.
                if ((std::string_view(branch) == "enum" || std::string_view(branch) == "const") &&
                    it.key() == "type") {
                    continue;
                }
                throw std::invalid_argument("unsupported tool schema assertion sibling of " +
                                            std::string(branch) + ": " + it.key());
            }
        }
    }
    if (schema.contains("$ref") &&
        (!schema["$ref"].is_string() || !schema["$ref"].get<std::string>().starts_with("#"))) {
        throw std::invalid_argument("tool schema references must be local");
    }
    auto matches_type = [&](const Json& value) {
        const auto& type = schema.at("type");
        auto one         = [&](const std::string& name) {
            return (name == "string" && value.is_string()) ||
                   (name == "integer" &&
                    (value.is_number_integer() ||
                     (value.is_number_float() &&
                      std::floor(value.get<double>()) == value.get<double>()))) ||
                   (name == "number" && value.is_number()) ||
                   (name == "boolean" && value.is_boolean()) ||
                   (name == "null" && value.is_null()) || (name == "object" && value.is_object()) ||
                   (name == "array" && value.is_array());
        };
        if (type.is_string()) { return one(type.get<std::string>()); }
        if (!type.is_array()) { return false; }
        return std::any_of(type.begin(), type.end(), [&](const Json& t) {
            return t.is_string() && one(t.get<std::string>());
        });
    };
    if (schema.contains("type")) {
        if (schema.contains("const") && !matches_type(schema["const"])) {
            throw std::invalid_argument("tool schema const conflicts with type");
        }
        if (schema.contains("enum")) {
            if (!schema["enum"].is_array() || schema["enum"].empty()) {
                throw std::invalid_argument("tool schema enum must be a nonempty array");
            }
            for (const auto& value : schema["enum"]) {
                if (!matches_type(value)) {
                    throw std::invalid_argument("tool schema enum conflicts with type");
                }
            }
        }
    }
    for (const char* map : {"properties", "$defs", "definitions"}) {
        if (!schema.contains(map)) { continue; }
        if (!schema[map].is_object()) {
            throw std::invalid_argument("tool schema map must be an object");
        }
        for (const auto& child : schema[map]) { check_schema(child); }
    }
    for (const char* child : {"items", "additionalProperties"}) {
        if (schema.contains(child)) { check_schema(schema[child]); }
    }
    for (const char* list : {"prefixItems", "anyOf"}) {
        if (!schema.contains(list)) { continue; }
        if (!schema[list].is_array()) {
            throw std::invalid_argument("tool schema list must be an array");
        }
        for (const auto& child : schema[list]) { check_schema(child); }
    }
}

// The converter only evaluates type-specific assertions on an explicit/inferred type.
// Reject untyped assertions instead of silently compiling an unrestricted JSON value.
void check_response_schema(const Json& schema) {
    check_schema(schema);
    if (!schema.is_object()) { return; }
    if (schema.contains("pattern")) {
        throw std::invalid_argument(
            "output schema pattern is unsupported: JSON escaping is not enforced by the converter");
    }
    if (schema.contains("$ref")) {
        const auto reference = schema["$ref"].get<std::string>();
        if (reference != "#" && !reference.starts_with("#/")) {
            throw std::invalid_argument("output schema references must use local JSON pointers");
        }
    }
    for (const char* key :
         {"minLength", "maxLength", "minItems", "maxItems", "minProperties", "maxProperties"}) {
        if (schema.contains(key) && (!schema[key].is_number_integer() || schema[key] < 0 ||
                                     schema[key] > std::numeric_limits<int>::max())) {
            throw std::invalid_argument(
                std::string("output schema bound must be a nonnegative 32-bit integer: ") + key);
        }
    }
    if (schema.contains("minProperties") && schema["minProperties"] > 1 &&
        schema.value("additionalProperties", Json(true)) != Json(false)) {
        const auto required = schema.value("required", Json::array());
        if (!required.is_array() || required.size() < schema["minProperties"].get<std::size_t>()) {
            throw std::invalid_argument(
                "minProperties with additional keys requires enough distinct required properties");
        }
    }
    if (schema.contains("required") && schema["required"].is_array()) {
        std::set<std::string> required;
        for (const auto& key : schema["required"]) {
            if (!key.is_string() || !required.insert(key.get<std::string>()).second) {
                throw std::invalid_argument("required must contain distinct string property names");
            }
        }
    }
    if (!schema.contains("type")) {
        for (const char* key : {"minimum", "maximum", "exclusiveMinimum", "exclusiveMaximum",
                                "minLength", "maxLength", "pattern", "minItems", "maxItems",
                                "minProperties", "maxProperties", "required"}) {
            if (schema.contains(key)) {
                throw std::invalid_argument(std::string("output schema assertion requires type: ") +
                                            key);
            }
        }
    }
    for (const char* map : {"properties", "$defs", "definitions"}) {
        if (schema.contains(map)) {
            for (const auto& child : schema[map]) { check_response_schema(child); }
        }
    }
    for (const char* child : {"items", "additionalProperties"}) {
        if (schema.contains(child)) { check_response_schema(schema[child]); }
    }
    for (const char* list : {"prefixItems", "anyOf"}) {
        if (schema.contains(list)) {
            for (const auto& child : schema[list]) { check_response_schema(child); }
        }
    }
}

// A required key need not also occur in properties. Give the grammar an
// explicit named transition for it using its additional-property schema.
// Otherwise even an unordered converter can mistake another key for it.
void name_required_properties(Json& schema) {
    if (!schema.is_object()) { return; }
    for (const char* map : {"properties", "$defs", "definitions"}) {
        if (schema.contains(map)) {
            for (auto& child : schema[map]) { name_required_properties(child); }
        }
    }
    for (const char* child : {"items", "additionalProperties"}) {
        if (schema.contains(child)) { name_required_properties(schema[child]); }
    }
    for (const char* list : {"prefixItems", "anyOf"}) {
        if (schema.contains(list)) {
            for (auto& child : schema[list]) { name_required_properties(child); }
        }
    }
    if (!schema.contains("required")) { return; }
    if (!schema["required"].is_array()) {
        throw std::invalid_argument("tool schema required must be an array");
    }
    // Inserting properties can reallocate ordered_json's parent object. Keep
    // the iteration independent of that object's storage.
    const auto required = schema["required"];
    for (const auto& value : required) {
        if (!value.is_string()) {
            throw std::invalid_argument("tool schema required keys must be strings");
        }
        const auto key = value.get<std::string>();
        if (schema.contains("properties") && schema["properties"].contains(key)) { continue; }
        const auto additional = schema.value("additionalProperties", Json(true));
        if (additional == Json(false)) {
            throw std::invalid_argument("required tool property is forbidden: " + key);
        }
        schema["properties"][key] = additional;
    }
}
} // namespace

TokenGrammarState::TokenGrammarState(std::shared_ptr<const TokenGrammarData> grammar,
                                     std::vector<TokenId> ignored_model_stops)
    : grammar_(std::move(grammar)), ignored_model_stops_(std::move(ignored_model_stops)),
      committed_(grammar_->compiled) {}

bool TokenGrammarState::accept(xgrammar::GrammarMatcher& matcher, TokenId token) const {
    if (std::find(ignored_model_stops_.begin(), ignored_model_stops_.end(), token) !=
        ignored_model_stops_.end()) {
        // Preserve the caller's disabled model-stop semantics. Such a token is
        // still forbidden inside an incomplete envelope, but an eligible EOS
        // must not permanently terminate the grammar while generation continues.
        auto probe = matcher.Fork();
        return probe.AcceptToken(token);
    }
    return matcher.AcceptToken(token);
}

void TokenGrammarState::preview(std::span<const TokenId> tokens) {
    if (preview_) { throw std::logic_error("token grammar already has a preview"); }
    auto candidate = committed_.Fork();
    for (const auto token : tokens) {
        if (!accept(candidate, token)) {
            throw std::logic_error("generated token violates the declared token grammar");
        }
    }
    preview_.emplace(std::move(candidate));
}

void TokenGrammarState::commit_preview() noexcept {
    if (!preview_) { std::terminate(); }
    committed_ = std::move(*preview_);
    preview_.reset();
}

void TokenGrammarState::discard_preview() noexcept { preview_.reset(); }

void TokenGrammarState::fill_masks(std::span<const TokenId> tokens,
                                   std::span<const std::int32_t> parents,
                                   std::span<std::uint32_t> words) const {
    if (tokens.empty() || tokens.size() != parents.size() ||
        words.size() != tokens.size() * mask_words) {
        throw std::invalid_argument("invalid token grammar verification mask shape");
    }
    std::vector<std::optional<xgrammar::GrammarMatcher>> nodes(tokens.size());
    nodes[0].emplace(committed_.Fork());
    for (std::size_t node = 0; node < nodes.size(); ++node) {
        if (node != 0) {
            const auto parent = parents[node];
            if (parent < 0 || static_cast<std::size_t>(parent) >= node) {
                throw std::invalid_argument("token grammar tree parent must precede child");
            }
            if (nodes[parent]) {
                auto child = nodes[parent]->Fork();
                if (accept(child, tokens[node]) && !child.IsTerminated()) {
                    nodes[node].emplace(std::move(child));
                }
            }
        }
        auto output = words.subspan(node * mask_words, mask_words);
        if (!nodes[node]) {
            std::fill(output.begin(), output.end(), ~std::uint32_t{0});
            continue;
        }
        std::int64_t shape = mask_words;
        DLTensor mask{.data        = output.data(),
                      .device      = {kDLCPU, 0},
                      .ndim        = 1,
                      .dtype       = {kDLInt, 32, 1},
                      .shape       = &shape,
                      .strides     = nullptr,
                      .byte_offset = 0};
        (void)nodes[node]->FillNextTokenBitmask(&mask);
    }
}

TokenGrammarCompiler::TokenGrammarCompiler(std::shared_ptr<const Tokenizer> tokenizer)
    : tokenizer_(std::move(tokenizer)) {}

std::shared_ptr<const TokenGrammarData>
TokenGrammarCompiler::compile(std::span<const std::string> tools, bool starts_in_reasoning,
                              const std::optional<OutputConstraint>& constraint) {
    if (tools.empty() && !constraint) { return {}; }
    if (!tools.empty() && constraint) {
        throw RequestError(RequestErrorKind::InvalidOutputConstraint,
                           "output constraints cannot be combined with tools");
    }
    std::vector<TokenGrammarData::Definition> definitions;
    auto tags = Json::array();
    std::set<std::string> names;
    for (const auto& text : tools) {
        try {
            const auto tool      = Json::parse(text);
            const auto& function = tool.at("function");
            const auto name      = function.at("name").get<std::string>();
            if (name.empty() || name.find_first_of("<>\r\n") != std::string::npos ||
                !names.insert(name).second) {
                throw std::invalid_argument("invalid or duplicate tool name: " + name);
            }
            Json schema = function.value("parameters", Json{{"type", "object"}});
            relax_schema(schema);
            check_schema(schema);
            name_required_properties(schema);
            definitions.push_back({name, schema, std::nullopt});
            tags.push_back(Json{{"type", "tag"},
                                {"begin", "<tool_call>\n<function=" + name + ">\n"},
                                {"content", Json{{"type", "json_schema"},
                                                 {"style", "qwen_xml"},
                                                 {"any_order", true},
                                                 {"json_schema", schema}}},
                                {"end", "\n</function>\n</tool_call>"}});
        } catch (const Json::exception& error) {
            throw RequestError(RequestErrorKind::InvalidToolSchema,
                               std::string("invalid tool declaration: ") + error.what());
        } catch (const std::invalid_argument& error) {
            throw RequestError(RequestErrorKind::InvalidToolSchema, error.what());
        }
    }
    // After a complete call, triggered_tags returns to free text. Reserve
    // protocol prefixes, not only complete tags: otherwise malformed variants
    // such as </invoke\n or </function = null> escape as "prose". Exclusions apply only to
    // the dispatch/free-text region, not to schema-constrained argument values
    // or to a call's actual closing frame. The captured legacy </invoke> tail
    // is also protocol debris, not a supported Qwen invocation envelope.
    Json format{{"type", "triggered_tags"},
                {"triggers", {"<tool_call"}},
                {"tags", tags},
                {"excludes",
                 {"</invoke", "</parameter", "</function", "</tool_call", "<invoke", "<parameter",
                  "<function"}}};
    if (constraint) {
        try {
            if (constraint->kind == OutputConstraintKind::Grammar) {
                if (constraint->source.empty()) {
                    throw std::invalid_argument("grammar must not be empty");
                }
                format = Json{{"type", "grammar"}, {"grammar", constraint->source}};
            } else {
                Json schema = constraint->kind == OutputConstraintKind::JsonObject
                                  ? Json{{"type", "object"}, {"additionalProperties", true}}
                                  : Json::parse(constraint->source);
                // Response constraints enforce assertions; tool-schema relaxation does not apply.
                check_response_schema(schema);
                name_required_properties(schema);
                format =
                    Json{{"type", "json_schema"}, {"json_schema", schema}, {"any_order", false}};
            }
        } catch (const Json::exception& error) {
            throw RequestError(RequestErrorKind::InvalidOutputConstraint, error.what());
        } catch (const std::invalid_argument& error) {
            throw RequestError(RequestErrorKind::InvalidOutputConstraint, error.what());
        }
    }
    if (starts_in_reasoning) {
        format = Json{
            {"type", "sequence"},
            {"elements",
             Json::array(
                 {// A tool envelope belongs to the content/call phase. Without this
                  // exclusion, the model can rehearse complete calls indefinitely
                  // inside reasoning, where they must never be published as calls.
                  Json{{"type", "tag"},
                       {"begin", ""},
                       {"content", Json{{"type", "any_text"},
                                        {"excludes", constraint ? Json::array()
                                                                : Json::array({"<tool_call"})}}},
                       {"end", "</think>"}},
                  format})}};
    }
    std::scoped_lock lock(mutex_);

    // Compile temporaries are freed before this returns, including a rejected
    // schema. malloc_trim releases the free space at the top of each glibc
    // arena and leaves the compiler cache in place.
    struct ReleaseCompileHeap {
        ~ReleaseCompileHeap() noexcept {
#if defined(__GLIBC__)
            ::malloc_trim(0);
#endif
        }
    } release_compile_heap;

    if (!compiler_) {
        std::vector<std::string> vocab(kTokenDomain);
        for (std::size_t id = 0; id < kTokenDomain; ++id) {
            if (tokenizer_->is_valid_token(static_cast<int>(id))) {
                vocab[id] = tokenizer_->decode_token_bytes(static_cast<int>(id));
            }
        }
        xgrammar::TokenizerInfo info(vocab, xgrammar::VocabType::RAW, kTokenDomain,
                                     tokenizer_->default_stop_token_ids());
        compiler_.emplace(info, 4, true, 128 * 1024 * 1024);
    }
    try {
        if (constraint) {
            const auto& content = starts_in_reasoning ? format.at("elements").at(1) : format;
            auto compiled       = constraint->kind == OutputConstraintKind::Grammar
                                      ? compiler_->CompileGrammar(constraint->source)
                                      : compiler_->CompileJSONSchema(
                                            content.at("json_schema").dump(), false, std::nullopt,
                                            std::pair<std::string, std::string>{",", ":"}, false);
            if (starts_in_reasoning) {
                auto reasoning = compiler_->CompileStructuralTag(Json{
                    {"type", "structural_tag"},
                    {"format", format.at("elements").at(0)}}.dump());
                compiled       = compiler_->CompileGrammar(
                    xgrammar::Grammar::Concat({reasoning.GetGrammar(), compiled.GetGrammar()}));
            }
            return std::make_shared<const TokenGrammarData>(
                TokenGrammarData{{}, std::move(compiled)});
        }
        auto compiled = compiler_->CompileStructuralTag(
            Json{{"type", "structural_tag"}, {"format", format}}.dump());
        for (auto& definition : definitions) {
            definition.arguments_validator =
                compiler_->CompileJSONSchema(definition.parameters.dump(), true, std::nullopt,
                                             std::nullopt, false, std::nullopt, true);
        }
        return std::make_shared<const TokenGrammarData>(
            TokenGrammarData{std::move(definitions), std::move(compiled)});
    } catch (const std::runtime_error& error) {
        // This pinned compiler reports malformed patterns/references through
        // runtime_error, including its non-public LogFatalError. Translate only
        // the client-schema compilation boundary; allocation failures and
        // runtime matcher/initialization failures retain their original type.
        throw RequestError(constraint ? RequestErrorKind::InvalidOutputConstraint
                                      : RequestErrorKind::InvalidToolSchema,
                           std::string("invalid or unsupported ") +
                               (constraint ? "output constraint: " : "tool schema: ") +
                               error.what());
    }
}

} // namespace ninfer::targets::qwen3_6::frontend_internal

namespace ninfer::targets::qwen3_6 {
namespace {
using Json = nlohmann::ordered_json;

std::string_view trim_parameter(std::string_view text) {
    while (!text.empty() && std::isspace(static_cast<unsigned char>(text.front()))) {
        text.remove_prefix(1);
    }
    while (!text.empty() && std::isspace(static_cast<unsigned char>(text.back()))) {
        text.remove_suffix(1);
    }
    return text;
}

Json parse_parameter_json(std::string_view text) {
    // nlohmann otherwise overwrites duplicate object keys before the grammar
    // sees the serialized value. Preserve the raw-string alternative, but do
    // not let this lossy parse turn invalid nested JSON into a valid argument.
    std::vector<std::set<std::string>> objects;
    bool duplicate    = false;
    const auto result = Json::parse(
        text,
        [&](int, Json::parse_event_t event, Json& value) {
            if (event == Json::parse_event_t::object_start) { objects.emplace_back(); }
            if (event == Json::parse_event_t::key &&
                !objects.back().insert(value.get<std::string>()).second) {
                duplicate = true;
            }
            if (event == Json::parse_event_t::object_end) { objects.pop_back(); }
            return true;
        },
        false);
    return duplicate ? Json(Json::value_t::discarded) : result;
}

} // namespace

std::optional<ToolCall> TokenGrammarData::decode_call(std::string_view text) const {
    auto consume = [&](std::string_view literal) {
        if (!text.starts_with(literal)) { return false; }
        text.remove_prefix(literal.size());
        return true;
    };
    text = trim_parameter(text);
    if (!consume("<tool_call>")) { return std::nullopt; }
    text = trim_parameter(text);
    if (!consume("<function=")) { return std::nullopt; }
    auto end = text.find('>');
    if (end == std::string_view::npos) { return std::nullopt; }
    const std::string name(text.substr(0, end));
    auto definition = std::find_if(definitions.begin(), definitions.end(),
                                   [&](const Definition& item) { return item.name == name; });
    if (definition == definitions.end()) { return std::nullopt; }
    text.remove_prefix(end + 1);
    // Qwen strings are raw; other values use JSON. The compiled argument
    // grammar resolves types and ambiguous literal delimiters together. A raw
    // string may contain "</parameter>", so a lexical first-close parser is
    // insufficient even when generation obeyed its declared schema.
    xgrammar::GrammarMatcher validator(*definition->arguments_validator, std::nullopt, true);
    if (!validator.AcceptString("{")) { return std::nullopt; }
    std::set<std::string> keys;
    auto decode = [&](auto&& self, std::string_view remaining, xgrammar::GrammarMatcher matcher,
                      std::string serialized, std::size_t count) -> std::optional<ToolCall> {
        remaining                                 = trim_parameter(remaining);
        constexpr std::string_view function_close = "</function>";
        constexpr std::string_view tool_close     = "</tool_call>";
        if (remaining.starts_with(function_close)) {
            remaining = trim_parameter(remaining.substr(function_close.size()));
            if (!remaining.starts_with(tool_close) ||
                !trim_parameter(remaining.substr(tool_close.size())).empty() ||
                !matcher.AcceptString("}") || !matcher.IsCompleted()) {
                return std::nullopt;
            }
            serialized += '}';
            return ToolCall{.id = {}, .name = name, .arguments_json = std::move(serialized)};
        }
        constexpr std::string_view parameter_open  = "<parameter=";
        constexpr std::string_view parameter_close = "</parameter>";
        if (!remaining.starts_with(parameter_open)) { return std::nullopt; }
        remaining.remove_prefix(parameter_open.size());
        const auto key_end = remaining.find('>');
        if (key_end == 0 || key_end == std::string_view::npos) { return std::nullopt; }
        const std::string key(remaining.substr(0, key_end));
        if (keys.contains(key)) { return std::nullopt; }
        remaining.remove_prefix(key_end + 1);
        const auto prefix = (count ? "," : "") + Json(key).dump() + ":";
        if (!matcher.AcceptString(prefix)) { return std::nullopt; }
        serialized += prefix;
        keys.insert(key);
        for (auto close = remaining.find(parameter_close); close != std::string_view::npos;
             close      = remaining.find(parameter_close, close + parameter_close.size())) {
            auto raw = remaining.substr(0, close);
            // Only the two canonical framing LFs are removed; string whitespace
            // and embedded XML-looking text remain data.
            if (raw.size() < 2 || raw.front() != '\n' || raw.back() != '\n') { continue; }
            raw.remove_prefix(1);
            raw.remove_suffix(1);
            const auto parsed = parse_parameter_json(raw);
            for (int interpretation = 0; interpretation < 2; ++interpretation) {
                if (interpretation && (parsed.is_discarded() || parsed == Json(raw))) { continue; }
                const auto value = interpretation ? parsed.dump() : Json(raw).dump();
                auto candidate   = matcher.Fork();
                if (candidate.AcceptString(value)) {
                    if (auto call = self(self, remaining.substr(close + parameter_close.size()),
                                         std::move(candidate), serialized + value, count + 1)) {
                        return call;
                    }
                }
            }
        }
        keys.erase(key);
        return std::nullopt;
    };
    return decode(decode, text, std::move(validator), "{", 0);
}

} // namespace ninfer::targets::qwen3_6
