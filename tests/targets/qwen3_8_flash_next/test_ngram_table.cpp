// Qwen3.8-Flash-Next n-gram table: exact row addressing against vectors produced by the study's
// validated reference (study/fnstudy/ngram.py), on the checkpoint's constants and on multipliers
// whose products wrap and set the sign bit; direct-I/O gathering against a small synthetic table
// whose records encode their row, with page reads checked against an independent LRU model; and a
// failed read followed by reuse of the table.

#include "ngram_table.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <exception>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>

#include <unistd.h>

namespace fn = ninfer::targets::qwen3_8_flash_next;

namespace {

constexpr fn::NgramHashConstants kCheckpointConstants{
    {23703573157769ULL, 20109073645365ULL, 8052911324071ULL},
    {0, 20000003, 40000026, 60000059, 80000106, 100000165, 120000228, 140000297, 160000374,
     180000455, 200000548, 220000655, 240000802, 260000955, 280001114, 300001275},
    {20000003, 20000023, 20000033, 20000047, 20000059, 20000063, 20000069, 20000077, 20000081,
     20000093, 20000107, 20000147, 20000153, 20000159, 20000161, 20000171}};

// EOS first, an EOS mid-sequence, two consecutive EOS, the largest token id, and token 0.
constexpr std::array<std::int32_t, 14> kTokens{248044, 9707,   11, 1879,   0, 248044, 248044,
                                               151,    248319, 12, 248044, 7, 8,      9};
constexpr std::array<std::array<std::int64_t, 16>, 14> kExpectedRows{{
    {9663979, 26558231, 56120240, 74755659, 80459717, 109265651, 132697467, 151022725, 170054832,
     192967038, 200687722, 225763581, 259275737, 272983544, 297596484, 300986548},
    {16410909, 39682429, 55103279, 60931720, 87006904, 116506179, 131512017, 152932897, 169641436,
     182022480, 209277433, 237891023, 256841529, 277007269, 290665954, 300984276},
    {18158303, 36390029, 45652312, 70783524, 98191154, 114024957, 127804916, 159566246, 175132467,
     192583600, 217919476, 237199054, 259336807, 261799367, 289359313, 307699687},
    {6380558, 26411572, 56460672, 78566983, 94693008, 106742196, 124822692, 148942556, 164226950,
     190352573, 210933682, 238908951, 242182004, 265475238, 299910982, 312121804},
    {11899280, 26999030, 54577278, 65218531, 94369136, 117425429, 122015444, 148146117, 177818549,
     195090789, 208609478, 233152873, 241860531, 270575158, 286814936, 308024703},
    {16421076, 34964127, 58645314, 64737717, 85974391, 107327379, 120238726, 145766809, 164291590,
     188207263, 211464619, 238266694, 251347998, 265488680, 283771098, 316948345},
    {9663979, 26558231, 56120240, 74755659, 80459717, 109265651, 132697467, 151022725, 170054832,
     192967038, 200687722, 225763581, 259275737, 272983544, 297596484, 300986548},
    {12101806, 23705379, 43247137, 66794343, 93724372, 116832281, 122242040, 150851369, 170681863,
     196252359, 218950955, 232021704, 257683336, 264571131, 280473016, 302025818},
    {7476016, 23569574, 56028866, 74413808, 86189849, 111056583, 129239119, 141796441, 162247667,
     199002351, 200075892, 231982352, 240453329, 269885767, 293243645, 311634410},
    {12316222, 33730646, 48183168, 60611415, 89445062, 106521964, 132886381, 149437116, 170563307,
     197444899, 200016436, 222894412, 245771036, 269546310, 284337821, 319792804},
    {16375281, 32077204, 54337887, 62441661, 93973888, 118758743, 126817871, 159209732, 168428119,
     185612388, 207202419, 220673792, 249391129, 279072768, 282514216, 301328606},
    {2927653, 34980843, 54748278, 66612378, 97814964, 109013870, 126560393, 151352333, 167888935,
     182235580, 215170017, 237467519, 247510681, 278779700, 296141806, 304994522},
    {17329723, 28464492, 44031928, 69826401, 80507399, 110734489, 136075079, 156529177, 176740572,
     199715560, 201670584, 220899605, 256162416, 271784618, 290405356, 304107632},
    {49828, 31718286, 47552595, 65720705, 92722053, 115055895, 128556613, 153224292, 175994762,
     190636546, 214385437, 236525521, 243846641, 271167938, 280275043, 305810447},
}};

// Multipliers above 2^63, so t0 * m0 wraps for every token below and the signed hash takes both
// signs (sign bit of t0 * m0: 1,0,1,0,1,1); head sizes 1000003 + 2h, offsets cumulative.
constexpr fn::NgramHashConstants wrapping_constants() {
    fn::NgramHashConstants constants{
        {0x9E3779B97F4A7C15ULL, 0xC2B2AE3D27D4EB4FULL, 0x165667B19E3779F9ULL}, {}, {}};
    std::int64_t offset = 0;
    for (std::size_t head = 0; head < fn::kNgramHeads; ++head) {
        constants.head_offsets[head] = offset;
        constants.head_sizes[head]   = 1000003 + 2 * static_cast<std::int64_t>(head);
        offset += constants.head_sizes[head];
    }
    return constants;
}

constexpr std::array<std::int32_t, 6> kWrappingTokens{248319, 5, 248044, 123456, 77, 200000};
constexpr std::array<std::array<std::int64_t, 16>, 6> kWrappingRows{{
    {117289, 1676765, 2536300, 3696305, 4157177, 5919344, 6983190, 7349117, 8923638, 9942009,
     10062861, 11286442, 12612935, 13042515, 14575442, 15211860},
    {571785, 1247286, 2526060, 3407816, 4892277, 5979148, 6668141, 7958985, 8714537, 9869896,
     10839226, 11622425, 12219391, 13630051, 14854278, 15891970},
    {799806, 1553075, 2622209, 3006917, 4706933, 5721955, 6051695, 7695899, 8767908, 9275425,
     10220888, 11604201, 12425241, 13683935, 14380156, 15513835},
    {515703, 1533548, 2138661, 3330943, 4110281, 5476580, 6429723, 7969619, 8446001, 9382332,
     10147192, 11740324, 12161398, 13410184, 14486375, 15389689},
    {664045, 1727635, 2123326, 3851526, 4912615, 5306983, 6035035, 7097167, 8095002, 9284404,
     10542112, 11868342, 12263283, 13727203, 14260262, 15862736},
    {149307, 1808349, 2394229, 3906725, 4345587, 5710601, 6001509, 7218105, 8396669, 9453143,
     10727891, 11220864, 12932086, 13861481, 14009019, 15374736},
}};

template <std::size_t N>
int check_row_ids(const char* label, const std::array<std::int32_t, N>& tokens,
                  const std::array<std::array<std::int64_t, 16>, N>& expected,
                  const fn::NgramHashConstants& constants) {
    int failures = 0;
    fn::NgramHistory history;
    for (std::size_t t = 0; t < N; ++t) {
        if (fn::ngram_row_ids(tokens[t], history, constants) != expected[t]) {
            std::cerr << "ngram_row_ids " << label << " mismatch at token " << t << '\n';
            ++failures;
        }
    }
    return failures;
}

// Independent model of the documented cache contract: a gather reads each distinct page holding a
// row absent from the cache when the gather starts; afterwards each set holds its eight most
// recently used rows in logical access order (exact LRU).
class LruModel {
public:
    explicit LruModel(std::size_t cache_rows) : sets_(cache_rows / fn::kNgramCacheWays) {}

    // Pages a gather of `rows` must read: those holding a row absent from the cache.
    [[nodiscard]] std::set<std::int64_t> absent_pages(const std::vector<std::int64_t>& rows) {
        std::set<std::int64_t> pages;
        for (const std::int64_t row : rows) {
            const auto& set = set_of(row);
            if (std::find(set.begin(), set.end(), row) == set.end()) {
                pages.insert(row / fn::kNgramRowsPerPage);
            }
        }
        return pages;
    }

    std::uint64_t gather(const std::vector<std::int64_t>& rows) {
        const std::size_t pages = absent_pages(rows).size();
        for (const std::int64_t row : rows) {
            auto& set = set_of(row);
            set.erase(std::remove(set.begin(), set.end(), row), set.end());
            set.insert(set.begin(), row);
            if (set.size() > fn::kNgramCacheWays) { set.pop_back(); }
        }
        return pages;
    }

private:
    std::vector<std::int64_t>& set_of(std::int64_t row) {
        return sets_[static_cast<std::size_t>(row) % sets_.size()];
    }

    std::vector<std::vector<std::int64_t>> sets_;
};

// Small table: 16 heads of 61..76 rows (offsets cumulative). Record of row r: FP16 bits r & 0xffff
// as "scale", then code j = (r * 7 + j) mod 256.
std::vector<std::uint8_t> synthetic_record(std::int64_t row) {
    std::vector<std::uint8_t> record(fn::kNgramRecordBytes);
    const auto bits = static_cast<std::uint16_t>(row & 0xffff);
    std::memcpy(record.data(), &bits, sizeof(bits));
    for (std::int32_t j = 0; j < fn::kNgramRowWidth; ++j) {
        record[2 + static_cast<std::size_t>(j)] = static_cast<std::uint8_t>((row * 7 + j) & 0xff);
    }
    return record;
}

struct SyntheticTable {
    std::filesystem::path path;
    fn::NgramHashConstants constants;
    std::vector<std::uint8_t> bytes;
};

SyntheticTable make_table(const std::filesystem::path& directory) {
    SyntheticTable table{directory / ("ngram_table_test_" + std::to_string(::getpid()) + ".bin"),
                         kCheckpointConstants,
                         {}};
    std::int64_t rows = 0;
    for (std::size_t head = 0; head < fn::kNgramHeads; ++head) {
        table.constants.head_offsets[head] = rows;
        table.constants.head_sizes[head]   = 61 + static_cast<std::int64_t>(head);
        rows += table.constants.head_sizes[head];
    }
    const std::int64_t pages = (rows + fn::kNgramRowsPerPage - 1) / fn::kNgramRowsPerPage;
    const nlohmann::json header{{"format", "int8_row"},
                                {"rows", rows},
                                {"dim", fn::kNgramRowWidth},
                                {"row_bytes", fn::kNgramRecordBytes},
                                {"rows_per_page", fn::kNgramRowsPerPage},
                                {"page_bytes", fn::kNgramPageBytes},
                                {"body_offset", fn::kNgramPageBytes},
                                {"layer_multipliers", table.constants.multipliers},
                                {"ngram_heads_offsets", table.constants.head_offsets},
                                {"ngram_heads_vocab_sizes", table.constants.head_sizes}};
    table.bytes.assign(static_cast<std::size_t>(fn::kNgramPageBytes * (pages + 1)), 0);
    const std::string text = header.dump();
    std::memcpy(table.bytes.data(), text.data(), text.size());
    for (std::int64_t row = 0; row < rows; ++row) {
        const auto record = synthetic_record(row);
        std::memcpy(table.bytes.data() + fn::kNgramPageBytes +
                        (row / fn::kNgramRowsPerPage) * fn::kNgramPageBytes +
                        (row % fn::kNgramRowsPerPage) * fn::kNgramRecordBytes,
                    record.data(), record.size());
    }
    return table;
}

void write_file(const std::filesystem::path& path, const std::vector<std::uint8_t>& bytes) {
    std::ofstream stream(path, std::ios::binary | std::ios::trunc);
    stream.write(reinterpret_cast<const char*>(bytes.data()),
                 static_cast<std::streamsize>(bytes.size()));
    if (!stream) { throw std::runtime_error("cannot write " + path.string()); }
}

// Gathers kTokens from a fresh history (so every round accesses the same rows) and compares the
// bytes and the page-read count with the model; returns the failures.
int gather_round(fn::NgramTable& table, const fn::NgramHashConstants& constants, LruModel& model,
                 const std::string& label) {
    fn::NgramHistory history;
    fn::NgramHistory reference_history;
    std::vector<std::uint8_t> out(kTokens.size() * fn::kNgramTokenBytes);
    const std::uint64_t pages_before = table.pages_read();
    table.gather(kTokens, history, out);
    int failures = 0;
    std::vector<std::int64_t> accessed;
    for (std::size_t t = 0; t < kTokens.size(); ++t) {
        const auto ids = fn::ngram_row_ids(kTokens[t], reference_history, constants);
        for (std::size_t head = 0; head < ids.size(); ++head) {
            accessed.push_back(ids[head]);
            const auto expected = synthetic_record(ids[head]);
            if (std::memcmp(out.data() + (t * fn::kNgramHeads + head) * fn::kNgramRecordBytes,
                            expected.data(), expected.size()) != 0) {
                std::cerr << label << ": gather mismatch at token " << t << " head " << head
                          << '\n';
                ++failures;
            }
        }
    }
    if (history.tokens != reference_history.tokens) {
        std::cerr << label << ": history not advanced past the tokens\n";
        ++failures;
    }
    const std::uint64_t expected_pages = model.gather(accessed);
    if (table.pages_read() - pages_before != expected_pages) {
        std::cerr << label << ": read " << table.pages_read() - pages_before << " pages, expected "
                  << expected_pages << '\n';
        ++failures;
    }
    return failures;
}

int check_gather(const std::filesystem::path& directory) {
    const SyntheticTable synthetic = make_table(directory);
    write_file(synthetic.path, synthetic.bytes);
    int failures = 0;
    try {
        // Two sets force evictions within a round; 32 sets keep some rows across rounds; 4096 rows
        // hold the whole access set, so the second round is all hits and reads nothing.
        for (const std::size_t cache_rows :
             {std::size_t{16}, std::size_t{256}, std::size_t{4096}}) {
            fn::NgramTable table({synthetic.path, cache_rows});
            LruModel model(cache_rows);
            for (int round = 0; round < 3; ++round) {
                failures += gather_round(table, synthetic.constants, model,
                                         "cache " + std::to_string(cache_rows) + " round " +
                                             std::to_string(round));
            }
        }
        // A gather that fails part-way leaves the cache unchanged and the pool usable: warm the
        // cache, truncate the body so the next gather's misses hit both readable and missing pages,
        // then restore the file; the following rounds must match the model that never saw the
        // failure.
        fn::NgramTable table({synthetic.path, 256});
        LruModel model(256);
        failures += gather_round(table, synthetic.constants, model, "before failure");
        // Different tokens make the failing gather access a different, partly cached row set.
        std::array<std::int32_t, kTokens.size()> other_tokens{};
        std::vector<std::int64_t> shifted;
        fn::NgramHistory shifted_history;
        for (std::size_t t = 0; t < kTokens.size(); ++t) {
            other_tokens[t] = (kTokens[t] * 31 + 17) % 248320;
            const auto ids =
                fn::ngram_row_ids(other_tokens[t], shifted_history, synthetic.constants);
            shifted.insert(shifted.end(), ids.begin(), ids.end());
        }
        const std::int64_t body_pages =
            static_cast<std::int64_t>(synthetic.bytes.size()) / fn::kNgramPageBytes - 1;
        const std::int64_t kept_pages = body_pages / 2;
        const auto absent             = model.absent_pages(shifted);
        if (absent.empty() || *absent.begin() >= kept_pages || *absent.rbegin() < kept_pages) {
            std::cerr << "failure case does not miss pages on both sides of the truncation\n";
            ++failures;
        }
        std::filesystem::resize_file(
            synthetic.path, static_cast<std::uintmax_t>(fn::kNgramPageBytes * (1 + kept_pages)));
        bool threw = false;
        try {
            fn::NgramHistory history;
            std::vector<std::uint8_t> out(other_tokens.size() * fn::kNgramTokenBytes);
            table.gather(other_tokens, history, out);
        } catch (const std::runtime_error&) { threw = true; }
        if (!threw) {
            std::cerr << "gather from a truncated file did not throw\n";
            ++failures;
        }
        write_file(synthetic.path, synthetic.bytes);
        // Page counts against the unchanged model expose any membership or recency change.
        failures += gather_round(table, synthetic.constants, model, "after failure");
        failures += gather_round(table, synthetic.constants, model, "after failure, repeated");
    } catch (...) {
        std::filesystem::remove(synthetic.path);
        throw;
    }
    std::filesystem::remove(synthetic.path);
    return failures;
}

} // namespace

int main() {
    try {
        int failures = check_row_ids("checkpoint", kTokens, kExpectedRows, kCheckpointConstants);
        failures += check_row_ids("wrapping", kWrappingTokens, kWrappingRows, wrapping_constants());
        failures += check_gather(std::filesystem::temp_directory_path());
        std::cout << (failures == 0 ? "OK" : "FAIL") << " flash_next ngram_table\n";
        return failures == 0 ? 0 : 1;
    } catch (const std::exception& error) {
        std::cerr << "flash_next ngram_table: " << error.what() << '\n';
        return 1;
    }
}
