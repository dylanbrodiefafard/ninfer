#include <xgrammar/xgrammar.h>

#include <algorithm>
#include <cstdint>
#include <iostream>
#include <set>
#include <string>
#include <vector>

namespace {

int check_metadata(const xgrammar::TokenizerInfo& info, const std::vector<std::string>& decoded,
                   const std::vector<std::int32_t>& stops, int vocab_size) {
    int failures     = 0;
    const auto check = [&](bool condition, const char* message) {
        if (!condition) {
            std::cerr << message << '\n';
            ++failures;
        }
    };
    check(info.GetDecodedVocab() == decoded, "decoded vocabulary mismatch");
    check(info.GetStopTokenIds() == stops, "stop-token metadata mismatch");
    check(info.GetVocabSize() == vocab_size, "vocabulary domain mismatch");
    std::vector<std::int32_t> special;
    std::set<std::int32_t> expected_ids;
    for (int id = 0; id < vocab_size; ++id) {
        if (id >= static_cast<int>(decoded.size()) ||
            (decoded[id].empty() && std::find(stops.begin(), stops.end(), id) == stops.end())) {
            special.push_back(id);
        } else if (std::find(stops.begin(), stops.end(), id) == stops.end()) {
            expected_ids.insert(id);
        }
    }
    check(info.GetSpecialTokenIds() == special, "special-token metadata mismatch");
    const auto& sorted = info.GetSortedDecodedVocab();
    const auto& ranges = info.GetTrieSubtreeNodesRange();
    check(sorted.size() == expected_ids.size(), "sorted vocabulary size mismatch");
    check(ranges.size() == sorted.size(), "subtree range count mismatch");
    std::set<std::int32_t> actual_ids;
    for (std::size_t i = 0; i < sorted.size(); ++i) {
        const auto& [id, bytes] = sorted[i];
        check(id >= 0 && id < static_cast<int>(decoded.size()), "sorted token id outside input");
        if (id >= 0 && id < static_cast<int>(decoded.size())) {
            check(bytes == decoded[id], "sorted token bytes mismatch");
        }
        check(actual_ids.insert(id).second, "sorted vocabulary repeats token id");
        check(i == 0 || sorted[i - 1].second <= bytes, "vocabulary is not byte-sorted");
        // Independently scan the contiguous byte-prefix interval. A substring
        // appearing after byte zero does not belong to this token's subtree.
        std::size_t end = i + 1;
        while (end < sorted.size() && sorted[end].second.starts_with(bytes)) { ++end; }
        if (i < ranges.size()) {
            check(ranges[i] == static_cast<std::int32_t>(end), "prefix subtree interval mismatch");
        }
    }
    check(actual_ids == expected_ids, "sorted vocabulary token membership mismatch");
    return failures;
}

int check_masks(const std::vector<std::string>& vocab) {
    const auto stop   = static_cast<std::int32_t>(vocab.size() - 1);
    const auto domain = static_cast<int>(vocab.size() + 2);
    const std::vector<std::set<std::string>> languages{
        {"ba"}, {"bab"}, {"éa"}, {"xxéa"}, {"a", "ab", "ba", "bab", "baba"}};
    const std::vector<std::string> prefixes{"",   "a", "ab", "b",   "ba",   "bab",  "baba",  "é",
                                            "éa", "x", "xx", "xxé", "xxéa", "\xc3", "xx\xc3"};
    int failures = 0;
    for (const auto& words : languages) {
        std::string source = "root ::= ";
        for (const auto& word : words) {
            if (source != "root ::= ") { source += " | "; }
            source += "\"" + word + "\"";
        }
        // The compiled grammar retains its tokenizer after every input owner
        // and the compiler have gone out of scope.
        const auto grammar = [&]() {
            auto temporary = vocab;
            xgrammar::TokenizerInfo info(temporary, xgrammar::VocabType::RAW, domain,
                                         std::vector<std::int32_t>{stop});
            temporary.assign(temporary.size(), "overwritten");
            xgrammar::GrammarCompiler compiler(info, 1, false);
            return compiler.CompileGrammar(source);
        }();
        const auto can_complete = [&](const std::string& prefix) {
            return std::any_of(words.begin(), words.end(),
                               [&](const auto& word) { return word.starts_with(prefix); });
        };
        for (const auto& prefix : prefixes) {
            if (!can_complete(prefix)) { continue; }
            xgrammar::GrammarMatcher matcher(grammar);
            if (!matcher.AcceptString(prefix)) {
                std::cerr << "valid language prefix rejected\n";
                ++failures;
                continue;
            }
            std::uint32_t bits = 0;
            std::int64_t shape = 1;
            DLTensor mask{.data        = &bits,
                          .device      = {kDLCPU, 0},
                          .ndim        = 1,
                          .dtype       = {kDLInt, 32, 1},
                          .shape       = &shape,
                          .strides     = nullptr,
                          .byte_offset = 0};
            (void)matcher.FillNextTokenBitmask(&mask);
            for (int id = 0; id < domain; ++id) {
                const bool present = id < static_cast<int>(vocab.size());
                const bool expected =
                    id == stop ? words.contains(prefix)
                               : present && !vocab[id].empty() && can_complete(prefix + vocab[id]);
                if (((bits & (1u << id)) != 0) != expected) {
                    std::cerr << "token mask differs from finite-language oracle: token=" << id
                              << " prefix=" << prefix << '\n';
                    ++failures;
                }
                if (present && !vocab[id].empty()) {
                    auto branch = matcher.Fork();
                    if (branch.AcceptToken(id) != expected) {
                        std::cerr << "token acceptance differs from finite-language oracle\n";
                        ++failures;
                    }
                }
            }
        }
    }
    return failures;
}

} // namespace

int main() {
    const std::vector<std::string> vocab{"a", "ab", "ba", "bab", "baba", "a", "", "é", "éa", "xxéa",
                                         "x", "xx", std::string("a\0b", 3), std::string(512, 'a'),
                                         "<eos>",
                                         // RAW tokens can split a UTF-8 scalar between tokens.
                                         "\xc3", "\xa9", "\xa9\x61", "xx\xc3", "<stop>"};
    const auto domain = static_cast<int>(vocab.size() + 2);
    const auto stop   = static_cast<std::int32_t>(vocab.size() - 1);
    const auto owned  = [&]() {
        auto temporary = vocab;
        xgrammar::TokenizerInfo info(temporary, xgrammar::VocabType::RAW, domain,
                                     std::vector<std::int32_t>{stop});
        temporary.assign(temporary.size(), "overwritten");
        return info;
    }();
    int failures = check_metadata(owned, vocab, {stop}, domain);
    failures += check_metadata(xgrammar::TokenizerInfo(vocab, xgrammar::VocabType::RAW, domain),
                               vocab, {14}, domain);
    // An explicitly designated empty stop remains a stop rather than a special token.
    failures += check_metadata(xgrammar::TokenizerInfo(vocab, xgrammar::VocabType::RAW, domain,
                                                       std::vector<std::int32_t>{6, stop}),
                               vocab, {6, stop}, domain);
    const std::vector<std::string> fallback{"<0x41>", "▁a", "", "<eos>"};
    failures +=
        check_metadata(xgrammar::TokenizerInfo(fallback, xgrammar::VocabType::BYTE_FALLBACK),
                       {"A", " a", "", "<eos>"}, {3}, 4);
    failures += check_masks(vocab);
    return failures == 0 ? 0 : 1;
}
