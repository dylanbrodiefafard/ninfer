#include "text/json_input.h"
#include <charconv>
#include <cstdint>
#include <limits>
#include <stdexcept>

#include <algorithm>
#include <utility>

namespace ninfer::text {
namespace {
using Json = nlohmann::ordered_json;

std::string escape(std::string_view key) {
    std::string result;
    for (char c : key) result += c == '~' ? "~0" : c == '/' ? "~1" : std::string(1, c);
    return result;
}

// Compare decimal values after the same serialization used to forward schemas. This
// accepts ordinary 0.1 and exponent/zero spelling differences, but rejects lost digits.
struct Decimal {
    std::string digits;
    std::int64_t exponent                 = 0;
    bool negative                         = false;
    bool operator==(const Decimal&) const = default;
};

std::optional<Decimal> decimal(std::string_view text) {
    Decimal out;
    out.negative = text.front() == '-';
    if (out.negative) text.remove_prefix(1);
    const auto e   = text.find_first_of("eE");
    auto mantissa  = text.substr(0, e);
    const auto dot = mantissa.find('.');
    for (char c : mantissa)
        if (c != '.') out.digits += c;
    const auto first = out.digits.find_first_not_of('0');
    if (first == std::string::npos) return Decimal{};
    out.digits.erase(0, first);
    if (e != std::string_view::npos) {
        auto power = text.substr(e + 1);
        if (power.front() == '+') power.remove_prefix(1);
        const auto parsed =
            std::from_chars(power.data(), power.data() + power.size(), out.exponent);
        if (parsed.ec != std::errc{} || parsed.ptr != power.data() + power.size()) return {};
    }
    const auto fractional = dot == std::string_view::npos ? 0 : mantissa.size() - dot - 1;
    if (out.exponent <
        std::numeric_limits<std::int64_t>::min() + static_cast<std::int64_t>(fractional))
        return {};
    out.exponent -= static_cast<std::int64_t>(fractional);
    const auto last     = out.digits.find_last_not_of('0');
    const auto trailing = out.digits.size() - last - 1;
    if (out.exponent >
        std::numeric_limits<std::int64_t>::max() - static_cast<std::int64_t>(trailing))
        return {};
    out.exponent += static_cast<std::int64_t>(trailing);
    out.digits.resize(last + 1);
    return out;
}

bool number_preserved(std::string_view source, double value) {
    const auto before = decimal(source);
    return before && before == decimal(Json(value).dump());
}

class NumberSax {
public:
    explicit NumberSax(ParsedJsonNumbers& result) : result_(result), dom_(result.value) {}

    bool null() {
        scalar();
        return dom_.null();
    }

    bool boolean(bool value) {
        scalar();
        return dom_.boolean(value);
    }

    bool number_integer(Json::number_integer_t value) {
        scalar();
        return dom_.number_integer(value);
    }

    bool number_unsigned(Json::number_unsigned_t value) {
        scalar();
        return dom_.number_unsigned(value);
    }

    bool number_float(Json::number_float_t value, const Json::string_t& source) {
        clear_value();
        if (!number_preserved(source, value)) result_.inexact_numbers.push_back(pointer());
        advance();
        return dom_.number_float(value, source);
    }

    bool string(Json::string_t& value) {
        scalar();
        return dom_.string(value);
    }

    bool binary(Json::binary_t& value) {
        scalar();
        return dom_.binary(value);
    }

    bool start_object(std::size_t size) {
        clear_value();
        stack_.push_back({});
        return dom_.start_object(size);
    }

    bool key(Json::string_t& value) {
        stack_.back().key = value;
        return dom_.key(value);
    }

    bool end_object() {
        stack_.pop_back();
        advance();
        return dom_.end_object();
    }

    bool start_array(std::size_t size) {
        clear_value();
        stack_.push_back({.array = true});
        return dom_.start_array(size);
    }

    bool end_array() {
        stack_.pop_back();
        advance();
        return dom_.end_array();
    }

    bool parse_error(std::size_t position, const std::string& token,
                     const nlohmann::detail::exception& error) {
        return dom_.parse_error(position, token, error);
    }

private:
    struct Frame {
        bool array        = false;
        std::size_t index = 0;
        std::string key;
    };

    std::string pointer() const {
        std::string path;
        for (const auto& frame : stack_)
            path += '/' + (frame.array ? std::to_string(frame.index) : escape(frame.key));
        return path;
    }

    void advance() {
        if (!stack_.empty() && stack_.back().array) ++stack_.back().index;
    }

    void clear_value() {
        if (result_.inexact_numbers.empty()) return;
        const auto path = pointer();
        std::erase_if(result_.inexact_numbers,
                      [&](const auto& old) { return old == path || old.starts_with(path + '/'); });
    }

    void scalar() {
        clear_value();
        advance();
    }

    ParsedJsonNumbers& result_;
    using InputAdapter =
        decltype(nlohmann::detail::input_adapter(std::declval<std::string_view>()));
    nlohmann::detail::json_sax_dom_parser<Json, InputAdapter> dom_;
    std::vector<Frame> stack_;
};

std::optional<std::string> find(const Json& node, const std::string& path,
                                const std::vector<std::string>& inexact, bool literal = false) {
    if (node.is_number()) {
        if (std::find(inexact.begin(), inexact.end(), path) != inexact.end()) return path;
        return {};
    }
    if (literal) {
        if (node.is_structured())
            for (const auto& [key, value] : node.items())
                if (auto found = find(value, path + '/' + escape(key), inexact, true)) return found;
        return {};
    }
    if (!node.is_object()) return {};
    for (const char* key :
         {"minimum", "maximum", "exclusiveMinimum", "exclusiveMaximum", "minLength", "maxLength",
          "minItems", "maxItems", "minProperties", "maxProperties", "const", "enum"})
        if (node.contains(key))
            if (auto found = find(node[key], path + '/' + key, inexact, true)) return found;
    for (const char* key :
         {"properties", "$defs", "definitions", "anyOf", "oneOf", "allOf", "prefixItems"}) {
        if (!node.contains(key) || !node[key].is_structured()) continue;
        for (const auto& [index, child] : node[key].items())
            if (auto found = find(child, path + '/' + key + '/' + escape(index), inexact))
                return found;
    }
    for (const char* key : {"items", "additionalItems", "additionalProperties"}) {
        if (!node.contains(key)) continue;
        if (node[key].is_array()) {
            for (std::size_t i = 0; i < node[key].size(); ++i)
                if (auto found =
                        find(node[key][i], path + '/' + key + '/' + std::to_string(i), inexact))
                    return found;
        } else if (auto found = find(node[key], path + '/' + key, inexact))
            return found;
    }
    return {};
}
} // namespace

ParsedJsonNumbers parse_json_numbers(std::string_view source) {
    ParsedJsonNumbers parsed;
    NumberSax sax(parsed);
    if (!Json::sax_parse(source, &sax)) { throw std::invalid_argument("failed to parse JSON"); }
    return parsed;
}

std::optional<std::string> inexact_schema_number(const ParsedJsonNumbers& parsed,
                                                 const std::string& schema_pointer) {
    if (parsed.inexact_numbers.empty()) return {};
    const Json::json_pointer pointer(schema_pointer);
    if (!parsed.value.contains(pointer)) return {};
    return find(parsed.value.at(pointer), schema_pointer, parsed.inexact_numbers);
}
} // namespace ninfer::text
