#include "targets/qwen3_6/impl/runtime/score_index.h"

#include <cstdint>
#include <iostream>
#include <limits>
#include <optional>
#include <stdexcept>
#include <string_view>
#include <vector>

namespace {

int failures = 0;

void expect(bool condition, std::string_view message) {
    if (condition) { return; }
    ++failures;
    std::cerr << "FAIL: " << message << '\n';
}

void test_skip() {
    expect(ninfer::resolve_score_skip(2, std::nullopt) == 0, "n=2 default skip is 0");
    expect(ninfer::resolve_score_skip(3, std::nullopt) == 1, "n=3 default skip is 1");
    expect(ninfer::resolve_score_skip(8, std::nullopt) == 4, "n=8 default skip is n/2");
    expect(ninfer::resolve_score_skip(8192, std::nullopt) == 4096, "8k default skip is half");
    expect(ninfer::resolve_score_skip(8, 0) == 0, "explicit skip 0");
    expect(ninfer::resolve_score_skip(8, 6) == 6, "explicit skip n-2");
    try {
        (void)ninfer::resolve_score_skip(8, 7);
        expect(false, "skip n-1 must throw");
    } catch (const std::invalid_argument&) {}
    try {
        (void)ninfer::resolve_score_skip(1, std::nullopt);
        expect(false, "n<2 must throw");
    } catch (const std::invalid_argument&) {}
}

void test_decode_prefix() {
    expect(ninfer::resolve_decode_prefix(8, 4) == 4, "decode prefix follows skip");
    expect(ninfer::resolve_decode_prefix(8, 0) == 1, "decode prefix is at least 1");
    try {
        (void)ninfer::resolve_decode_prefix(2, 0);
        expect(false, "decode n<3 must throw");
    } catch (const std::invalid_argument&) {}
}

void test_chunk_targets() {
    const std::vector<ninfer::TokenId> ids{10, 11, 12, 13, 14, 15, 16, 17};
    const std::vector<std::int32_t> all = ninfer::prefill_chunk_targets(ids, 0, 8, 0);
    expect(all.size() == 7 && all.front() == 11 && all.back() == 17,
           "full sequence scores n-1 next-token ids");

    const std::vector<std::int32_t> half = ninfer::prefill_chunk_targets(ids, 0, 8, 4);
    expect(half.size() == 3 && half[0] == 15 && half[2] == 17,
           "skip n/2 scores positions 4..6");

    const std::vector<std::int32_t> first = ninfer::prefill_chunk_targets(ids, 0, 4, 4);
    expect(first.empty(), "warmup chunk contributes no targets");

    const std::vector<std::int32_t> second = ninfer::prefill_chunk_targets(ids, 4, 4, 4);
    expect(second.size() == 3 && second[0] == 15,
           "second chunk starts scoring at skip");

    const std::vector<std::int32_t> cross = ninfer::prefill_chunk_targets(ids, 0, 4, 0);
    expect(cross.size() == 4 && cross.back() == 14,
           "non-final chunk scores its last column against the next chunk");
}

void test_record_nll() {
    ninfer::ScoreResult result;
    ninfer::record_score_nll(result, 1.5);
    ninfer::record_score_nll(result, 10.0);
    ninfer::record_score_nll(result, std::numeric_limits<double>::infinity());
    expect(result.tokens_scored == 2, "non-finite NLL is not a scored token");
    expect(result.non_finite == 1, "non-finite counter");
    expect(result.terrible_tokens == 1, "nll >= 10 counts as terrible");
    expect(result.max_nll == 10.0, "max_nll tracks the worst finite token");
    expect(result.token_nlls.size() == 2 && result.token_nlls[0] == 1.5f,
           "token_nlls stores finite values only");
}

} // namespace

int main() {
    test_skip();
    test_decode_prefix();
    test_chunk_targets();
    test_record_nll();
    if (failures != 0) {
        std::cerr << "score_index failures=" << failures << '\n';
        return 1;
    }
    std::cout << "score_index: PASS\n";
    return 0;
}
