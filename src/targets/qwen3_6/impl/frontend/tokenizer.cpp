#include "targets/qwen3_6/impl/frontend/tokenizer.h"

#include "text/unicode.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <array>
#include <charconv>
#include <cctype>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <optional>
#include <span>
#include <sstream>
#include <stdexcept>
#include <tuple>
#include <string>
#include <string_view>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

namespace ninfer::targets::qwen3_6::frontend_internal {
namespace {

using Json    = nlohmann::json;
namespace uni = ninfer::text::unicode_internal;

constexpr std::int64_t kMaxTokenId = 1'000'000;

struct VocabMetadata {
    std::vector<std::string> id_to_token;
    std::unordered_map<std::string, int> token_to_id;
    std::unordered_set<int> occupied_ids;
};

Json read_json_asset(std::string_view contents, std::string_view label) {
    try {
        return Json::parse(contents);
    } catch (const nlohmann::json::exception& ex) {
        throw std::invalid_argument("malformed " + std::string(label) + ": " + ex.what());
    }
}

const Json& require_object_field(const Json& object, const char* field, std::string_view label) {
    if (!object.is_object() || !object.contains(field)) {
        throw std::invalid_argument("missing field " + std::string(field) + " in " +
                                    std::string(label));
    }
    const Json& value = object.at(field);
    if (!value.is_object()) {
        throw std::invalid_argument("field " + std::string(field) + " must be object in " +
                                    std::string(label));
    }
    return value;
}

const Json& require_array_field(const Json& object, const char* field, std::string_view label) {
    if (!object.is_object() || !object.contains(field)) {
        throw std::invalid_argument("missing field " + std::string(field) + " in " +
                                    std::string(label));
    }
    const Json& value = object.at(field);
    if (!value.is_array()) {
        throw std::invalid_argument("field " + std::string(field) + " must be array in " +
                                    std::string(label));
    }
    return value;
}

int parse_token_id(const Json& value, const char* field, std::string_view label) {
    if (!value.is_number_integer()) {
        throw std::invalid_argument("field " + std::string(field) + " must be integer in " +
                                    std::string(label));
    }
    if (value.is_number_unsigned()) {
        const std::uint64_t id = value.get<std::uint64_t>();
        if (id > static_cast<std::uint64_t>(kMaxTokenId)) {
            throw std::invalid_argument("field " + std::string(field) + " id is out of range in " +
                                        std::string(label));
        }
        return static_cast<int>(id);
    }

    const std::int64_t id = value.get<std::int64_t>();
    if (id < 0) {
        throw std::invalid_argument("field " + std::string(field) + " has negative id in " +
                                    std::string(label));
    }
    if (id > kMaxTokenId) {
        throw std::invalid_argument("field " + std::string(field) + " id is out of range in " +
                                    std::string(label));
    }
    return static_cast<int>(id);
}

std::string require_string_field(const Json& object, const char* field, std::string_view label) {
    if (!object.is_object() || !object.contains(field) || !object.at(field).is_string()) {
        throw std::invalid_argument("field " + std::string(field) + " must be string in " +
                                    std::string(label));
    }
    return object.at(field).get<std::string>();
}

bool require_bool_field(const Json& object, const char* field, std::string_view label) {
    if (!object.is_object() || !object.contains(field) || !object.at(field).is_boolean()) {
        throw std::invalid_argument("field " + std::string(field) + " must be boolean in " +
                                    std::string(label));
    }
    return object.at(field).get<bool>();
}

VocabMetadata load_vocab(const Json& model, std::string_view label) {
    if (!model.contains("type") || !model.at("type").is_string() ||
        model.at("type").get<std::string>() != "BPE") {
        throw std::invalid_argument("field model.type must be BPE in " + std::string(label));
    }
    const Json& vocab = require_object_field(model, "vocab", label);
    if (vocab.empty()) {
        throw std::invalid_argument("field model.vocab must not be empty in " + std::string(label));
    }

    int max_id = -1;
    VocabMetadata metadata;
    for (const auto& item : vocab.items()) {
        const int id = parse_token_id(item.value(), "model.vocab", label);
        if (!metadata.occupied_ids.insert(id).second) {
            throw std::invalid_argument("field model.vocab has duplicate id in " +
                                        std::string(label));
        }
        max_id = std::max(max_id, id);
    }

    metadata.id_to_token.resize(static_cast<std::size_t>(max_id + 1));
    for (const auto& item : vocab.items()) {
        const int id = parse_token_id(item.value(), "model.vocab", label);
        metadata.id_to_token.at(static_cast<std::size_t>(id)) = item.key();
        metadata.token_to_id.emplace(item.key(), id);
    }
    return metadata;
}

AddedToken parse_added_token(const Json& item, std::string_view label) {
    if (!item.is_object()) {
        throw std::invalid_argument("field added_tokens item must be object in " +
                                    std::string(label));
    }
    AddedToken token;
    if (!item.contains("id")) {
        throw std::invalid_argument("missing field added_tokens.id in " + std::string(label));
    }
    token.id          = parse_token_id(item.at("id"), "added_tokens.id", label);
    token.content     = require_string_field(item, "content", label);
    token.single_word = require_bool_field(item, "single_word", label);
    token.lstrip      = require_bool_field(item, "lstrip", label);
    token.rstrip      = require_bool_field(item, "rstrip", label);
    token.normalized  = require_bool_field(item, "normalized", label);
    token.special     = require_bool_field(item, "special", label);
    return token;
}

AddedToken parse_added_token_decoder_entry(int id, const Json& item, std::string_view label) {
    if (!item.is_object()) {
        throw std::invalid_argument("field added_tokens_decoder item must be object in " +
                                    std::string(label));
    }
    AddedToken token;
    token.id          = id;
    token.content     = require_string_field(item, "content", label);
    token.single_word = require_bool_field(item, "single_word", label);
    token.lstrip      = require_bool_field(item, "lstrip", label);
    token.rstrip      = require_bool_field(item, "rstrip", label);
    token.normalized  = require_bool_field(item, "normalized", label);
    token.special     = require_bool_field(item, "special", label);
    return token;
}

void validate_supported_added_token(const AddedToken& token, std::string_view label) {
    if (token.content.empty()) {
        throw std::invalid_argument("added token content must not be empty in " +
                                    std::string(label));
    }
    if (token.single_word || token.lstrip || token.rstrip || token.normalized) {
        throw std::invalid_argument("Tokenizer only supports added tokens with single_word=false, "
                                    "lstrip=false, rstrip=false, and normalized=false in " +
                                    std::string(label));
    }
}

bool same_added_token(const AddedToken& lhs, const AddedToken& rhs) noexcept {
    return lhs.id == rhs.id && lhs.content == rhs.content && lhs.single_word == rhs.single_word &&
           lhs.lstrip == rhs.lstrip && lhs.rstrip == rhs.rstrip &&
           lhs.normalized == rhs.normalized && lhs.special == rhs.special;
}

int parse_added_token_decoder_id(std::string_view key, std::string_view label) {
    std::int64_t parsed     = -1;
    const auto [end, error] = std::from_chars(key.data(), key.data() + key.size(), parsed);
    if (error != std::errc{} || end != key.data() + key.size() || parsed < 0 ||
        parsed > kMaxTokenId || std::to_string(parsed) != key) {
        throw std::invalid_argument("added_tokens_decoder key must be a nonnegative token id in " +
                                    std::string(label));
    }
    return static_cast<int>(parsed);
}

std::vector<AddedToken>
load_added_tokens(const Json& root, std::string_view label, std::vector<std::string>& id_to_token,
                  const std::unordered_set<int>& occupied_vocab_ids,
                  const std::unordered_map<std::string, int>& occupied_vocab_tokens) {
    const Json& added = require_array_field(root, "added_tokens", label);
    std::vector<AddedToken> tokens;
    tokens.reserve(added.size());
    std::unordered_set<int> seen_added_ids;
    std::unordered_map<std::string, int> seen_added_contents;
    for (const Json& item : added) {
        AddedToken token = parse_added_token(item, label);
        validate_supported_added_token(token, label);
        const auto index = static_cast<std::size_t>(token.id);
        if (occupied_vocab_ids.contains(token.id)) {
            throw std::invalid_argument("field added_tokens overlaps existing id in " +
                                        std::string(label));
        }
        if (!seen_added_ids.insert(token.id).second) {
            throw std::invalid_argument("field added_tokens has duplicate id in " +
                                        std::string(label));
        }
        if (occupied_vocab_tokens.contains(token.content) ||
            !seen_added_contents.emplace(token.content, token.id).second) {
            throw std::invalid_argument("field added_tokens has duplicate content mapping in " +
                                        std::string(label));
        }
        if (index >= id_to_token.size()) { id_to_token.resize(index + 1); }
        id_to_token.at(static_cast<std::size_t>(token.id)) = token.content;
        tokens.push_back(std::move(token));
    }
    return tokens;
}

void merge_added_tokens_decoder(const Json& root, std::string_view label,
                                std::vector<std::string>& id_to_token,
                                const std::unordered_set<int>& occupied_vocab_ids,
                                const std::unordered_map<std::string, int>& occupied_vocab_tokens,
                                std::vector<AddedToken>& tokens) {
    const Json& decoder = require_object_field(root, "added_tokens_decoder", label);
    std::unordered_map<int, std::size_t> token_by_id;
    std::unordered_map<std::string, int> token_by_content;
    std::unordered_set<int> decoder_ids;
    token_by_id.reserve(tokens.size() + decoder.size());
    token_by_content.reserve(tokens.size() + decoder.size());
    for (std::size_t index = 0; index < tokens.size(); ++index) {
        token_by_id.emplace(tokens[index].id, index);
        token_by_content.emplace(tokens[index].content, tokens[index].id);
    }

    for (const auto& item : decoder.items()) {
        const int id = parse_added_token_decoder_id(item.key(), label);
        if (!decoder_ids.insert(id).second) {
            throw std::invalid_argument("added_tokens_decoder has duplicate id mapping in " +
                                        std::string(label));
        }
        AddedToken token = parse_added_token_decoder_entry(id, item.value(), label);
        validate_supported_added_token(token, label);

        const auto existing_id = token_by_id.find(id);
        if (existing_id != token_by_id.end()) {
            if (!same_added_token(tokens.at(existing_id->second), token)) {
                throw std::invalid_argument("conflicting added-token definition for id " +
                                            std::to_string(id) +
                                            " between tokenizer.json and tokenizer_config.json");
            }
            continue;
        }
        if (occupied_vocab_ids.contains(id)) {
            throw std::invalid_argument("added_tokens_decoder overlaps vocabulary id " +
                                        std::to_string(id));
        }
        if (token_by_content.contains(token.content) ||
            occupied_vocab_tokens.contains(token.content)) {
            throw std::invalid_argument("conflicting added-token content mapping for " +
                                        token.content);
        }

        const auto index = static_cast<std::size_t>(id);
        if (index >= id_to_token.size()) { id_to_token.resize(index + 1); }
        if (!id_to_token[index].empty()) {
            throw std::invalid_argument("duplicate tokenizer mapping for id " + std::to_string(id));
        }
        id_to_token[index] = token.content;
        token_by_id.emplace(id, tokens.size());
        token_by_content.emplace(token.content, id);
        tokens.push_back(std::move(token));
    }
    std::sort(tokens.begin(), tokens.end(),
              [](const AddedToken& lhs, const AddedToken& rhs) { return lhs.id < rhs.id; });
}

std::vector<int> load_default_stop_token_ids(std::string_view contents) {
    constexpr std::string_view label = "generation_config.json";
    const Json root                  = read_json_asset(contents, label);
    if (!root.is_object() || !root.contains("eos_token_id")) {
        throw std::invalid_argument("missing field eos_token_id in generation_config.json");
    }

    const Json& eos = root.at("eos_token_id");
    if (eos.is_number_integer()) { return {parse_token_id(eos, "eos_token_id", label)}; }
    if (eos.is_array()) {
        if (eos.empty()) {
            throw std::invalid_argument(
                "field eos_token_id must not be empty in generation_config.json");
        }
        std::vector<int> ids;
        ids.reserve(eos.size());
        for (const Json& item : eos) { ids.push_back(parse_token_id(item, "eos_token_id", label)); }
        return ids;
    }
    throw std::invalid_argument(
        "field eos_token_id must be integer or array in generation_config.json");
}

constexpr std::uint64_t kEmptyBpeKey = ~std::uint64_t{0};

[[nodiscard]] constexpr std::uint64_t pack_bpe_pair(int left, int right) noexcept {
    return (static_cast<std::uint64_t>(static_cast<std::uint32_t>(left)) << 32U) |
           static_cast<std::uint32_t>(right);
}

[[nodiscard]] constexpr std::uint64_t hash_bpe_key(std::uint64_t key) noexcept {
    key ^= key >> 30U;
    key *= 0xbf58476d1ce4e5b9ULL;
    key ^= key >> 27U;
    key *= 0x94d049bb133111ebULL;
    key ^= key >> 31U;
    return key;
}

std::size_t bpe_table_size(std::size_t pair_count) {
    std::size_t size = 8;
    const std::size_t needed = pair_count * 2 + 1;
    while (size < needed) { size *= 2; }
    return size;
}

void insert_bpe_pair(BpePairTable& table, std::uint64_t key, int rank, int result) {
    std::size_t index = hash_bpe_key(key) & table.mask;
    for (std::size_t probe = 0; probe < table.slots.size(); ++probe) {
        BpePairSlot& slot = table.slots[index];
        if (slot.key == kEmptyBpeKey) {
            slot.key    = key;
            slot.rank   = rank;
            slot.result = result;
            return;
        }
        if (slot.key == key) { throw std::invalid_argument("duplicate merge pair in model.merges"); }
        index = (index + 1) & table.mask;
    }
    throw std::logic_error("BPE pair table is full");
}

const BpePairSlot* find_bpe_pair(const BpePairTable& table, int left, int right) noexcept {
    if (table.slots.empty()) { return nullptr; }
    const std::uint64_t key = pack_bpe_pair(left, right);
    std::size_t index       = hash_bpe_key(key) & table.mask;
    for (std::size_t probe = 0; probe < table.slots.size(); ++probe) {
        const BpePairSlot& slot = table.slots[index];
        if (slot.key == key) { return &slot; }
        if (slot.key == kEmptyBpeKey) { return nullptr; }
        index = (index + 1) & table.mask;
    }
    return nullptr;
}

struct BpeMergeSpec {
    std::string left;
    std::string right;
    int rank = 0;
};

std::vector<BpeMergeSpec> load_bpe_merges(const Json& model, std::string_view label) {
    const Json& merges = require_array_field(model, "merges", label);
    std::vector<BpeMergeSpec> specs;
    specs.reserve(merges.size());
    int rank = 0;
    for (const Json& item : merges) {
        BpeMergeSpec spec;
        spec.rank = rank++;
        if (item.is_array() && item.size() == 2 && item[0].is_string() && item[1].is_string()) {
            spec.left  = item[0].get<std::string>();
            spec.right = item[1].get<std::string>();
        } else if (item.is_string()) {
            const std::string pair  = item.get<std::string>();
            const std::size_t space = pair.find(' ');
            if (space == std::string::npos || space == 0 || space + 1 >= pair.size() ||
                pair.find(' ', space + 1) != std::string::npos) {
                throw std::invalid_argument("malformed model.merges entry in " +
                                            std::string(label));
            }
            spec.left  = pair.substr(0, space);
            spec.right = pair.substr(space + 1);
        } else {
            throw std::invalid_argument("field model.merges must contain symbol pairs in " +
                                        std::string(label));
        }
        specs.push_back(std::move(spec));
    }
    return specs;
}

std::unordered_map<std::uint32_t, char> build_byte_level_decoder() {
    std::unordered_map<std::uint32_t, char> decoder;
    std::uint32_t next = 256;
    for (int byte = 0; byte <= std::numeric_limits<unsigned char>::max(); ++byte) {
        const bool visible = (byte >= 33 && byte <= 126) || (byte >= 161 && byte <= 172) ||
                             (byte >= 174 && byte <= 255);
        const std::uint32_t codepoint = visible ? static_cast<std::uint32_t>(byte) : next++;
        decoder.emplace(codepoint, static_cast<char>(static_cast<unsigned char>(byte)));
    }
    return decoder;
}

std::unordered_map<unsigned char, std::string> build_byte_level_encoder() {
    std::unordered_map<unsigned char, std::string> encoder;
    std::uint32_t next = 256;
    for (int byte = 0; byte <= std::numeric_limits<unsigned char>::max(); ++byte) {
        const bool visible = (byte >= 33 && byte <= 126) || (byte >= 161 && byte <= 172) ||
                             (byte >= 174 && byte <= 255);
        const std::uint32_t codepoint = visible ? static_cast<std::uint32_t>(byte) : next++;
        encoder.emplace(static_cast<unsigned char>(byte),
                        uni::codepoint_to_utf8(static_cast<std::int32_t>(codepoint)));
    }
    return encoder;
}

bool is_newline(std::int32_t codepoint) noexcept { return codepoint == '\r' || codepoint == '\n'; }

bool is_letter_or_mark(std::int32_t codepoint) noexcept {
    return uni::is_letter(codepoint) || uni::is_mark(codepoint);
}

bool is_non_newline_non_letter_non_number(std::int32_t codepoint) noexcept {
    return !is_newline(codepoint) && !uni::is_letter(codepoint) && !uni::is_number(codepoint);
}

bool is_non_space_non_letter_mark_number(std::int32_t codepoint) noexcept {
    return !uni::is_whitespace(codepoint) && !uni::is_letter(codepoint) &&
           !uni::is_mark(codepoint) && !uni::is_number(codepoint);
}

bool ascii_ci_matches(std::string_view text, std::size_t offset, std::string_view suffix) {
    if (offset + suffix.size() > text.size()) { return false; }
    for (std::size_t i = 0; i < suffix.size(); ++i) {
        const unsigned char lhs = static_cast<unsigned char>(text[offset + i]);
        const unsigned char rhs = static_cast<unsigned char>(suffix[i]);
        if (std::tolower(lhs) != std::tolower(rhs)) { return false; }
    }
    return true;
}

std::size_t span_end_offset(std::string_view text, const std::vector<uni::CodepointSpan>& spans,
                            std::size_t end) {
    if (end == spans.size()) { return text.size(); }
    return spans.at(end).offset;
}

bool ascii_letter(std::int32_t cp) noexcept {
    return (cp >= 'A' && cp <= 'Z') || (cp >= 'a' && cp <= 'z');
}

bool ascii_number(std::int32_t cp) noexcept { return cp >= '0' && cp <= '9'; }

bool ascii_whitespace(std::int32_t cp) noexcept {
    return cp == ' ' || cp == '\t' || cp == '\n' || cp == '\r' || cp == '\v' || cp == '\f';
}

template <class Emit>
void for_each_ascii_qwen_word(std::string_view text, Emit&& emit) {
    const std::size_t n = text.size();
    for (std::size_t i = 0; i < n;) {
        const std::size_t begin        = i;
        const std::int32_t cp          = static_cast<unsigned char>(text[i]);

        if (cp == '\'') {
            constexpr std::string_view suffixes[] = {"s", "t", "re", "ve", "m", "ll", "d"};
            bool matched                          = false;
            for (std::string_view suffix : suffixes) {
                if (ascii_ci_matches(text, begin + 1, suffix)) {
                    i = begin + 1 + suffix.size();
                    emit(text.substr(begin, i - begin));
                    matched = true;
                    break;
                }
            }
            if (matched) { continue; }
        }

        if (ascii_letter(cp) ||
            (cp != '\r' && cp != '\n' && !ascii_letter(cp) && !ascii_number(cp) && i + 1 < n &&
             ascii_letter(static_cast<unsigned char>(text[i + 1])))) {
            if (!ascii_letter(cp)) { ++i; }
            while (i < n && ascii_letter(static_cast<unsigned char>(text[i]))) { ++i; }
            emit(text.substr(begin, i - begin));
            continue;
        }

        if (ascii_number(cp)) {
            emit(text.substr(begin, 1));
            ++i;
            continue;
        }

        const bool punct = !ascii_whitespace(cp) && !ascii_letter(cp) && !ascii_number(cp);
        if ((cp == ' ' && i + 1 < n &&
             !ascii_whitespace(static_cast<unsigned char>(text[i + 1])) &&
             !ascii_letter(static_cast<unsigned char>(text[i + 1])) &&
             !ascii_number(static_cast<unsigned char>(text[i + 1]))) ||
            punct) {
            if (cp == ' ') { ++i; }
            while (i < n) {
                const std::int32_t next = static_cast<unsigned char>(text[i]);
                if (ascii_whitespace(next) || ascii_letter(next) || ascii_number(next)) { break; }
                ++i;
            }
            while (i < n && (text[i] == '\r' || text[i] == '\n')) { ++i; }
            emit(text.substr(begin, i - begin));
            continue;
        }

        if (ascii_whitespace(cp)) {
            std::size_t run_end      = i;
            std::size_t last_newline = std::string_view::npos;
            while (run_end < n && ascii_whitespace(static_cast<unsigned char>(text[run_end]))) {
                if (text[run_end] == '\r' || text[run_end] == '\n') { last_newline = run_end; }
                ++run_end;
            }
            if (last_newline != std::string_view::npos) {
                i = last_newline + 1;
                emit(text.substr(begin, i - begin));
                continue;
            }
            if (run_end == n) {
                emit(text.substr(begin));
                break;
            }
            if (run_end - i >= 2) {
                i = run_end - 1;
                emit(text.substr(begin, i - begin));
                continue;
            }
            i = run_end;
            emit(text.substr(begin, i - begin));
            continue;
        }

        emit(text.substr(begin, 1));
        ++i;
    }
}

template <class Emit>
void for_each_unicode_qwen_word(std::string_view text, Emit&& emit) {
    const std::vector<uni::CodepointSpan> spans =
        uni::utf8_codepoints(text, "Tokenizer::encode input");
    for (std::size_t i = 0; i < spans.size();) {
        const std::size_t begin_offset = spans[i].offset;
        const std::int32_t cp          = spans[i].value;

        if (cp == '\'') {
            constexpr std::string_view suffixes[] = {"s", "t", "re", "ve", "m", "ll", "d"};
            bool matched                          = false;
            for (std::string_view suffix : suffixes) {
                if (ascii_ci_matches(text, begin_offset + 1, suffix)) {
                    std::size_t end = i + 1;
                    while (end < spans.size() &&
                           spans[end].offset < begin_offset + 1 + suffix.size()) {
                        ++end;
                    }
                    emit(text.substr(begin_offset, span_end_offset(text, spans, end) - begin_offset));
                    i       = end;
                    matched = true;
                    break;
                }
            }
            if (matched) { continue; }
        }

        if (is_letter_or_mark(cp) ||
            (is_non_newline_non_letter_non_number(cp) && i + 1 < spans.size() &&
             is_letter_or_mark(spans[i + 1].value))) {
            std::size_t end = i;
            if (!is_letter_or_mark(spans[end].value)) { ++end; }
            while (end < spans.size() && is_letter_or_mark(spans[end].value)) { ++end; }
            emit(text.substr(begin_offset, span_end_offset(text, spans, end) - begin_offset));
            i = end;
            continue;
        }

        if (uni::is_number(cp)) {
            emit(text.substr(begin_offset, spans[i].length));
            ++i;
            continue;
        }

        if ((cp == ' ' && i + 1 < spans.size() &&
             is_non_space_non_letter_mark_number(spans[i + 1].value)) ||
            is_non_space_non_letter_mark_number(cp)) {
            std::size_t end = i;
            if (spans[end].value == ' ') { ++end; }
            while (end < spans.size() && is_non_space_non_letter_mark_number(spans[end].value)) {
                ++end;
            }
            while (end < spans.size() && is_newline(spans[end].value)) { ++end; }
            emit(text.substr(begin_offset, span_end_offset(text, spans, end) - begin_offset));
            i = end;
            continue;
        }

        if (uni::is_whitespace(cp)) {
            std::size_t run_end      = i;
            std::size_t last_newline = std::string_view::npos;
            while (run_end < spans.size() && uni::is_whitespace(spans[run_end].value)) {
                if (is_newline(spans[run_end].value)) { last_newline = run_end; }
                ++run_end;
            }
            if (last_newline != std::string_view::npos) {
                const std::size_t end = last_newline + 1;
                emit(text.substr(begin_offset, span_end_offset(text, spans, end) - begin_offset));
                i = end;
                continue;
            }

            if (run_end == spans.size()) {
                emit(text.substr(begin_offset));
                break;
            }

            if (run_end - i >= 2) {
                const std::size_t end = run_end - 1;
                emit(text.substr(begin_offset, span_end_offset(text, spans, end) - begin_offset));
                i = end;
                continue;
            }

            emit(text.substr(begin_offset, span_end_offset(text, spans, run_end) - begin_offset));
            i = run_end;
            continue;
        }

        emit(text.substr(begin_offset, spans[i].length));
        ++i;
    }
}

bool is_ascii(std::string_view text) noexcept {
    for (const unsigned char byte : text) {
        if (byte >= 128U) { return false; }
    }
    return true;
}

template <class Emit>
void for_each_qwen_word(std::string_view text, Emit&& emit) {
    if (is_ascii(text)) {
        for_each_ascii_qwen_word(text, emit);
    } else {
        for_each_unicode_qwen_word(text, emit);
    }
}

bool is_added_token_id(const std::vector<AddedToken>& added_tokens, int id) {
    return std::any_of(added_tokens.begin(), added_tokens.end(),
                       [id](const AddedToken& token) { return token.id == id; });
}

bool is_stop_token_id(std::span<const int> stop_token_ids, int id) {
    return std::find(stop_token_ids.begin(), stop_token_ids.end(), id) != stop_token_ids.end();
}

void append_bpe_word(std::vector<int>& ids, std::string_view word, const BpePairTable& pair_table,
                     const std::array<int, 256>& byte_to_intern_id,
                     const std::vector<std::vector<int>>& intern_emit_ids) {
    if (word.empty()) { return; }
    struct Node {
        int intern_id = -1;
        int prev      = -1;
        int next      = -1;
    };
    std::vector<Node> nodes;
    nodes.reserve(word.size());
    for (const unsigned char byte : word) {
        const int intern_id = byte_to_intern_id[byte];
        if (intern_id < 0) {
            throw std::invalid_argument("Tokenizer::encode produced byte outside vocabulary");
        }
        const int index = static_cast<int>(nodes.size());
        nodes.push_back(Node{intern_id, index - 1, -1});
        if (index > 0) { nodes[static_cast<std::size_t>(index - 1)].next = index; }
    }
    int live = static_cast<int>(nodes.size());
    while (live > 1) {
        int best_rank   = std::numeric_limits<int>::max();
        int best_left   = -1;
        int best_result = -1;
        for (int index = 0; index >= 0; index = nodes[static_cast<std::size_t>(index)].next) {
            const int right = nodes[static_cast<std::size_t>(index)].next;
            if (right < 0) { break; }
            const BpePairSlot* found =
                find_bpe_pair(pair_table, nodes[static_cast<std::size_t>(index)].intern_id,
                              nodes[static_cast<std::size_t>(right)].intern_id);
            if (found != nullptr && found->rank < best_rank) {
                best_rank   = found->rank;
                best_left   = index;
                best_result = found->result;
            }
        }
        if (best_left < 0) { break; }
        Node& left                = nodes[static_cast<std::size_t>(best_left)];
        const int right           = left.next;
        left.intern_id            = best_result;
        left.next                 = nodes[static_cast<std::size_t>(right)].next;
        if (left.next >= 0) { nodes[static_cast<std::size_t>(left.next)].prev = best_left; }
        --live;
    }
    for (int index = 0; index >= 0; index = nodes[static_cast<std::size_t>(index)].next) {
        const int intern_id = nodes[static_cast<std::size_t>(index)].intern_id;
        if (intern_id < 0 || static_cast<std::size_t>(intern_id) >= intern_emit_ids.size() ||
            intern_emit_ids[static_cast<std::size_t>(intern_id)].empty()) {
            throw std::invalid_argument("Tokenizer::encode produced a symbol outside vocabulary");
        }
        const std::vector<int>& emitted = intern_emit_ids[static_cast<std::size_t>(intern_id)];
        ids.insert(ids.end(), emitted.begin(), emitted.end());
    }
}

void append_bpe_ids(std::vector<int>& ids, std::string_view text, bool has_bpe_merges,
                    const BpePairTable& pair_table, const std::array<int, 256>& byte_to_intern_id,
                    const std::vector<std::vector<int>>& intern_emit_ids) {
    if (text.empty()) { return; }
    if (!has_bpe_merges) {
        throw std::invalid_argument(
            "Tokenizer::encode ordinary BPE text requires embedded merges.txt");
    }

    std::string normalized_storage;
    std::string_view normalized = text;
    if (!is_ascii(text)) {
        normalized_storage = uni::normalize_nfc(text);
        normalized         = normalized_storage;
    }
    for_each_qwen_word(normalized, [&](std::string_view word) {
        append_bpe_word(ids, word, pair_table, byte_to_intern_id, intern_emit_ids);
    });
}

} // namespace

Tokenizer::Tokenizer(TokenizerResources resources) {
    if (resources.tokenizer_json.empty() || resources.tokenizer_config_json.empty() ||
        resources.generation_config_json.empty()) {
        throw std::invalid_argument("embedded tokenizer resources are empty");
    }
    constexpr std::string_view tokenizer_label        = "tokenizer.json";
    constexpr std::string_view tokenizer_config_label = "tokenizer_config.json";
    const Json root = read_json_asset(resources.tokenizer_json, tokenizer_label);
    const Json tokenizer_config =
        read_json_asset(resources.tokenizer_config_json, tokenizer_config_label);
    const Json& model = require_object_field(root, "model", tokenizer_label);

    VocabMetadata vocab_metadata = load_vocab(model, tokenizer_label);
    id_to_token_                 = std::move(vocab_metadata.id_to_token);
    vocab_token_to_id_           = std::move(vocab_metadata.token_to_id);
    valid_token_ids_.resize(id_to_token_.size());
    for (const int id : vocab_metadata.occupied_ids) {
        valid_token_ids_.at(static_cast<std::size_t>(id)) = true;
    }
    added_tokens_ = load_added_tokens(root, tokenizer_label, id_to_token_,
                                      vocab_metadata.occupied_ids, vocab_token_to_id_);
    merge_added_tokens_decoder(tokenizer_config, tokenizer_config_label, id_to_token_,
                               vocab_metadata.occupied_ids, vocab_token_to_id_, added_tokens_);
    if (valid_token_ids_.size() < id_to_token_.size()) {
        valid_token_ids_.resize(id_to_token_.size());
    }
    for (const AddedToken& token : added_tokens_) {
        valid_token_ids_.at(static_cast<std::size_t>(token.id)) = 1;
    }

    const std::vector<BpeMergeSpec> merges = load_bpe_merges(model, tokenizer_label);
    std::unordered_map<std::string, int> intern_ids;
    intern_ids.reserve(merges.size() + 256);
    std::vector<std::string> intern_str;
    intern_str.reserve(merges.size() + 256);
    std::vector<int> intern_to_vocab;
    intern_to_vocab.reserve(merges.size() + 256);
    auto intern_symbol = [&](const std::string& symbol) {
        const auto existing = intern_ids.find(symbol);
        if (existing != intern_ids.end()) { return existing->second; }
        const int intern_id = static_cast<int>(intern_str.size());
        intern_str.push_back(symbol);
        const auto vocab = vocab_token_to_id_.find(symbol);
        intern_to_vocab.push_back(vocab == vocab_token_to_id_.end() ? -1 : vocab->second);
        intern_ids.emplace(symbol, intern_id);
        return intern_id;
    };
    std::vector<std::tuple<int, int, int, int>> interned_pairs;
    interned_pairs.reserve(merges.size());
    for (const BpeMergeSpec& spec : merges) {
        interned_pairs.emplace_back(intern_symbol(spec.left), intern_symbol(spec.right),
                                    intern_symbol(spec.left + spec.right), spec.rank);
    }
    bpe_pair_table_.slots.assign(bpe_table_size(interned_pairs.size()), BpePairSlot{});
    bpe_pair_table_.mask = bpe_pair_table_.slots.empty() ? 0 : bpe_pair_table_.slots.size() - 1;
    for (const auto& [left, right, result, rank] : interned_pairs) {
        insert_bpe_pair(bpe_pair_table_, pack_bpe_pair(left, right), rank, result);
    }
    has_bpe_merges_ = true;

    const auto byte_encoder = build_byte_level_encoder();
    byte_to_intern_id_.fill(-1);
    for (int byte = 0; byte <= std::numeric_limits<unsigned char>::max(); ++byte) {
        byte_to_intern_id_[static_cast<std::size_t>(byte)] =
            intern_symbol(byte_encoder.at(static_cast<unsigned char>(byte)));
    }

    intern_emit_ids_.resize(intern_str.size());
    for (std::size_t intern_id = 0; intern_id < intern_str.size(); ++intern_id) {
        if (intern_to_vocab[intern_id] >= 0) {
            intern_emit_ids_[intern_id] = {intern_to_vocab[intern_id]};
            continue;
        }
        const std::vector<uni::CodepointSpan> pieces =
            uni::utf8_codepoints(intern_str[intern_id], "Tokenizer intern emit");
        if (pieces.size() <= 1) { continue; }
        intern_emit_ids_[intern_id].reserve(pieces.size());
        bool ok = true;
        for (const uni::CodepointSpan& piece : pieces) {
            const auto vocab =
                vocab_token_to_id_.find(intern_str[intern_id].substr(piece.offset, piece.length));
            if (vocab == vocab_token_to_id_.end()) {
                intern_emit_ids_[intern_id].clear();
                ok = false;
                break;
            }
            intern_emit_ids_[intern_id].push_back(vocab->second);
        }
        (void)ok;
    }

    special_token_ids_.assign(id_to_token_.size(), 0);
    added_token_ids_.assign(id_to_token_.size(), 0);
    for (const AddedToken& token : added_tokens_) {
        const auto index = static_cast<std::size_t>(token.id);
        added_token_ids_.at(index) = 1;
        if (token.special) { special_token_ids_.at(index) = 1; }
    }

    added_trie_.clear();
    added_trie_.emplace_back();
    added_start_bytes_.fill(0);
    for (std::size_t token_index = 0; token_index < added_tokens_.size(); ++token_index) {
        const std::string& content = added_tokens_[token_index].content;
        if (content.empty()) { continue; }
        added_start_bytes_[static_cast<unsigned char>(content[0])] = 1;
        int node = 0;
        for (const unsigned char byte : content) {
            int child = added_trie_[static_cast<std::size_t>(node)].next[byte];
            if (child < 0) {
                child = static_cast<int>(added_trie_.size());
                added_trie_[static_cast<std::size_t>(node)].next[byte] = child;
                added_trie_.emplace_back();
            }
            node = child;
        }
        int& first = added_trie_[static_cast<std::size_t>(node)].token_index;
        if (first < 0) { first = static_cast<int>(token_index); }
    }

    const auto byte_decoder = build_byte_level_decoder();
    id_to_decoded_bytes_.resize(id_to_token_.size());
    for (std::size_t id = 0; id < id_to_token_.size(); ++id) {
        if (valid_token_ids_[id] == 0) { continue; }
        if (added_token_ids_[id] != 0) {
            id_to_decoded_bytes_[id] = id_to_token_[id];
            continue;
        }
        std::string bytes;
        const std::vector<uni::CodepointSpan> codepoints =
            uni::utf8_codepoints(id_to_token_[id], "Tokenizer decode table");
        bytes.reserve(codepoints.size());
        for (const uni::CodepointSpan& codepoint : codepoints) {
            const auto byte = byte_decoder.find(static_cast<std::uint32_t>(codepoint.value));
            if (byte == byte_decoder.end()) {
                throw std::invalid_argument(
                    "Tokenizer vocabulary contains a character outside the byte-level alphabet");
            }
            bytes.push_back(byte->second);
        }
        id_to_decoded_bytes_[id] = std::move(bytes);
    }

    default_stop_token_ids_ = load_default_stop_token_ids(resources.generation_config_json);
}

std::optional<std::pair<std::size_t, int>>
Tokenizer::find_leftmost_added(std::string_view text, std::size_t pos) const {
    for (std::size_t i = pos; i < text.size(); ++i) {
        if (added_start_bytes_[static_cast<unsigned char>(text[i])] == 0) { continue; }
        int node             = 0;
        int best             = -1;
        std::size_t best_len = 0;
        for (std::size_t j = i; j < text.size(); ++j) {
            const int child =
                added_trie_[static_cast<std::size_t>(node)].next[static_cast<unsigned char>(
                    text[j])];
            if (child < 0) { break; }
            node                  = child;
            const int token_index = added_trie_[static_cast<std::size_t>(node)].token_index;
            if (token_index >= 0 && (best < 0 || token_index < best)) {
                best     = token_index;
                best_len = j - i + 1;
            }
        }
        if (best >= 0) {
            (void)best_len;
            return std::pair<std::size_t, int>{i, best};
        }
    }
    return std::nullopt;
}

std::vector<int> Tokenizer::encode(std::string_view text, EncodeOptions options) const {
    return encode(text, std::nullopt, options).ids;
}

bool Tokenizer::is_encode_loop_pos(std::string_view text, std::size_t n,
                                   EncodeOptions options) const {
    if (n == 0 || n == text.size()) { return true; }
    if (n > text.size()) { return false; }
    if (!options.parse_added_tokens) { return false; }
    std::size_t pos = 0;
    while (pos < text.size()) {
        const auto match = find_leftmost_added(text, pos);
        if (!match) { return false; }
        const std::size_t match_pos = match->first;
        const int match_index       = match->second;
        if (match_pos > pos) {
            if (n == match_pos) { return true; }
            if (n < match_pos) { return false; }
        }
        pos = match_pos + added_tokens_[static_cast<std::size_t>(match_index)].content.size();
        if (n == pos) { return true; }
        if (n < pos) { return false; }
    }
    return false;
}

EncodedText Tokenizer::encode(std::string_view text, std::optional<std::size_t> prefix_byte_end,
                              EncodeOptions options) const {
    EncodedText encoded;
    if (text.empty()) { return encoded; }
    auto mark_prefix = [&](std::size_t byte_pos) {
        if (prefix_byte_end && !encoded.prefix_tokens && byte_pos == *prefix_byte_end) {
            encoded.prefix_tokens = static_cast<std::uint32_t>(encoded.ids.size());
        }
    };
    mark_prefix(0);
    if (!options.parse_added_tokens) {
        append_bpe_ids(encoded.ids, text, has_bpe_merges_, bpe_pair_table_, byte_to_intern_id_,
                       intern_emit_ids_);
        mark_prefix(text.size());
        return encoded;
    }

    std::size_t pos = 0;
    while (pos < text.size()) {
        const auto match = find_leftmost_added(text, pos);
        if (!match) {
            append_bpe_ids(encoded.ids, text.substr(pos), has_bpe_merges_, bpe_pair_table_,
                           byte_to_intern_id_, intern_emit_ids_);
            mark_prefix(text.size());
            break;
        }
        const std::size_t match_pos   = match->first;
        const int match_index         = match->second;
        const AddedToken& match_token = added_tokens_[static_cast<std::size_t>(match_index)];
        if (match_pos > pos) {
            append_bpe_ids(encoded.ids, text.substr(pos, match_pos - pos), has_bpe_merges_,
                           bpe_pair_table_, byte_to_intern_id_, intern_emit_ids_);
            mark_prefix(match_pos);
        }

        encoded.ids.push_back(match_token.id);
        pos = match_pos + match_token.content.size();
        mark_prefix(pos);
    }
    return encoded;
}

std::string Tokenizer::decode(std::span<const int> ids, DecodeOptions options) const {
    std::string text;
    const std::size_t terminal_stop_index =
        (!ids.empty() && is_stop_token_id(options.stop_token_ids, ids.back())) ? ids.size() - 1
                                                                               : ids.size();

    for (std::size_t i = 0; i < ids.size(); ++i) {
        const int id = ids[i];
        if (i == terminal_stop_index) { continue; }
        text += decode_token_bytes(id, options.skip_special_tokens);
    }
    (void)uni::utf8_codepoints(text, "Tokenizer::decode reconstructed output");
    return text;
}

std::string_view Tokenizer::decode_token_bytes(int id, bool skip_special_tokens) const {
    if (skip_special_tokens && is_special_token(id)) { return {}; }
    if (id < 0) {
        throw std::invalid_argument("Tokenizer::decode received negative token id " +
                                    std::to_string(id));
    }
    const auto index = static_cast<std::size_t>(id);
    if (index >= id_to_decoded_bytes_.size() || index >= valid_token_ids_.size() ||
        valid_token_ids_[index] == 0) {
        throw std::out_of_range("Tokenizer::decode token id " + std::to_string(id) +
                                " is outside loaded vocabulary");
    }
    return id_to_decoded_bytes_[index];
}

bool Tokenizer::is_special_token(int id) const noexcept {
    return id >= 0 && static_cast<std::size_t>(id) < special_token_ids_.size() &&
           special_token_ids_[static_cast<std::size_t>(id)] != 0;
}

bool Tokenizer::is_valid_token(int id) const noexcept {
    return id >= 0 && static_cast<std::size_t>(id) < valid_token_ids_.size() &&
           valid_token_ids_[static_cast<std::size_t>(id)] != 0;
}

bool Tokenizer::has_exact_token_domain(std::size_t size) const noexcept {
    return valid_token_ids_.size() == size &&
           std::find(valid_token_ids_.begin(), valid_token_ids_.end(), 0) ==
               valid_token_ids_.end();
}

} // namespace ninfer::targets::qwen3_6::frontend_internal
