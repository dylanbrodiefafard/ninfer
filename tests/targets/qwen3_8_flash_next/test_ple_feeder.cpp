// Qwen3.8-Flash-Next PleRowFeeder: chunks begun ahead of their upload, uploaded whole or in parts,
// arrive on the device as the records of their tokens in sequence order, with the n-gram history
// carried across chunks and both staging buffers reused, and reset() restarts the history. Lanes
// keep separate histories: a decode-round gather of other lanes may run while a lane's prefetch is
// pending, and a cancelled prefetch leaves its lane's history as before it began. The table is a
// small synthetic int8_row file whose records encode their row.

#include "ngram_table.h"
#include "runtime/ple_feeder.h"

#include "core/arena.h"
#include "core/device.h"

#include <nlohmann/json.hpp>

#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <exception>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

#include <unistd.h>

namespace fn = ninfer::targets::qwen3_8_flash_next;

namespace {

constexpr std::int32_t kHeadRows = 97;

// Record of row r: FP16 scale bits r & 0xffff, then code j = (r * 5 + j) mod 256.
std::vector<std::uint8_t> record_of(std::int64_t row) {
    std::vector<std::uint8_t> record(fn::kNgramRecordBytes);
    const auto bits = static_cast<std::uint16_t>(row & 0xffff);
    std::memcpy(record.data(), &bits, sizeof(bits));
    for (std::int32_t j = 0; j < fn::kNgramRowWidth; ++j) {
        record[2 + static_cast<std::size_t>(j)] = static_cast<std::uint8_t>((row * 5 + j) & 0xff);
    }
    return record;
}

fn::NgramHashConstants write_table(const std::filesystem::path& path) {
    fn::NgramHashConstants constants;
    constants.multipliers = {1000003, 998244353, 7919};
    for (std::size_t head = 0; head < fn::kNgramHeads; ++head) {
        constants.head_offsets[head] = static_cast<std::int64_t>(head) * kHeadRows;
        constants.head_sizes[head]   = kHeadRows;
    }
    const std::int64_t rows  = static_cast<std::int64_t>(fn::kNgramHeads) * kHeadRows;
    const std::int64_t pages = (rows + fn::kNgramRowsPerPage - 1) / fn::kNgramRowsPerPage;
    const nlohmann::json header{{"format", "int8_row"},
                                {"rows", rows},
                                {"dim", fn::kNgramRowWidth},
                                {"row_bytes", fn::kNgramRecordBytes},
                                {"rows_per_page", fn::kNgramRowsPerPage},
                                {"page_bytes", fn::kNgramPageBytes},
                                {"body_offset", fn::kNgramPageBytes},
                                {"layer_multipliers", constants.multipliers},
                                {"ngram_heads_offsets", constants.head_offsets},
                                {"ngram_heads_vocab_sizes", constants.head_sizes}};
    std::vector<std::uint8_t> bytes(static_cast<std::size_t>(fn::kNgramPageBytes * (pages + 1)), 0);
    const std::string text = header.dump();
    std::memcpy(bytes.data(), text.data(), text.size());
    for (std::int64_t row = 0; row < rows; ++row) {
        const std::vector<std::uint8_t> record = record_of(row);
        const std::int64_t offset = fn::kNgramPageBytes * (1 + row / fn::kNgramRowsPerPage) +
                                    (row % fn::kNgramRowsPerPage) * fn::kNgramRecordBytes;
        std::memcpy(bytes.data() + offset, record.data(), record.size());
    }
    std::ofstream file(path, std::ios::binary | std::ios::trunc);
    file.write(reinterpret_cast<const char*>(bytes.data()),
               static_cast<std::streamsize>(bytes.size()));
    if (!file) { throw std::runtime_error("cannot write " + path.string()); }
    return constants;
}

// Compares `host` ([T][16][162] records) with the records of `tokens` after `reference`, which
// advances; returns the failures.
int check_records(std::span<const std::uint8_t> host, std::span<const std::int32_t> tokens,
                  fn::NgramHistory& reference, const fn::NgramHashConstants& constants,
                  const std::string& label) {
    int failures = 0;
    for (std::size_t t = 0; t < tokens.size(); ++t) {
        const auto ids = fn::ngram_row_ids(tokens[t], reference, constants);
        for (std::size_t head = 0; head < ids.size(); ++head) {
            const std::vector<std::uint8_t> expected = record_of(ids[head]);
            const std::size_t offset = (t * fn::kNgramHeads + head) * fn::kNgramRecordBytes;
            if (std::memcmp(host.data() + offset, expected.data(), expected.size()) != 0) {
                std::cerr << label << ": token " << t << " head " << head << " differs\n";
                ++failures;
            }
        }
    }
    return failures;
}

// Uploads every part of the pending gather of `tokens` into `rows` and reads them back.
std::vector<std::uint8_t> upload_all(fn::PleRowFeeder& feeder, ninfer::DeviceBuffer& rows,
                                     cudaStream_t stream, std::int32_t tokens,
                                     std::int32_t part_tokens) {
    for (std::int32_t first = 0, part = 0; first < tokens; first += part_tokens, ++part) {
        const std::int32_t width = std::min(part_tokens, tokens - first);
        ninfer::Tensor device(static_cast<std::uint8_t*>(rows.p) +
                                  static_cast<std::size_t>(first) * fn::kNgramTokenBytes,
                              ninfer::DType::U8, {fn::kNgramRecordBytes, fn::kNgramHeads, width});
        feeder.upload(device, part, stream);
    }
    std::vector<std::uint8_t> host(static_cast<std::size_t>(tokens) * fn::kNgramTokenBytes);
    CUDA_CHECK(cudaMemcpyAsync(host.data(), rows.p, host.size(), cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
    return host;
}

// Runs `chunks` through the feeder with each next chunk begun before the current one's rows are
// read back, and compares every uploaded record with the reference; returns the failures.
int run_request(fn::PleRowFeeder& feeder, ninfer::DeviceBuffer& rows, cudaStream_t stream,
                const fn::NgramHashConstants& constants,
                const std::vector<std::vector<std::int32_t>>& chunks, std::int32_t part_tokens,
                const std::string& label) {
    fn::NgramHistory reference;
    int failures = 0;
    feeder.reset(0);
    feeder.begin(0, chunks.front(), part_tokens);
    for (std::size_t c = 0; c < chunks.size(); ++c) {
        const auto tokens = static_cast<std::int32_t>(chunks[c].size());
        for (std::int32_t first = 0, part = 0; first < tokens; first += part_tokens, ++part) {
            const std::int32_t width = std::min(part_tokens, tokens - first);
            ninfer::Tensor device(static_cast<std::uint8_t*>(rows.p) +
                                      static_cast<std::size_t>(first) * fn::kNgramTokenBytes,
                                  ninfer::DType::U8,
                                  {fn::kNgramRecordBytes, fn::kNgramHeads, width});
            feeder.upload(device, part, stream);
        }
        if (c + 1 < chunks.size()) { feeder.begin(0, chunks[c + 1], part_tokens); }
        std::vector<std::uint8_t> host(static_cast<std::size_t>(tokens) * fn::kNgramTokenBytes);
        CUDA_CHECK(
            cudaMemcpyAsync(host.data(), rows.p, host.size(), cudaMemcpyDeviceToHost, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));
        for (std::int32_t t = 0; t < tokens; ++t) {
            const auto ids =
                fn::ngram_row_ids(chunks[c][static_cast<std::size_t>(t)], reference, constants);
            for (std::size_t head = 0; head < ids.size(); ++head) {
                const std::vector<std::uint8_t> expected = record_of(ids[head]);
                const std::size_t offset =
                    (static_cast<std::size_t>(t) * fn::kNgramHeads + head) * fn::kNgramRecordBytes;
                if (std::memcmp(host.data() + offset, expected.data(), expected.size()) != 0) {
                    std::cerr << label << ": chunk " << c << " token " << t << " head " << head
                              << " differs\n";
                    ++failures;
                }
            }
        }
    }
    return failures;
}

int run() {
    const std::filesystem::path path = std::filesystem::temp_directory_path() /
                                       ("ple_feeder_test_" + std::to_string(::getpid()) + ".bin");
    const fn::NgramHashConstants constants = write_table(path);
    int failures                           = 0;
    {
        ninfer::DeviceContext device(0);
        fn::NgramTable table({.path = path, .cache_rows = 64, .io_threads = 3});
        constexpr std::int32_t kMaxTokens = 7;
        fn::PleRowFeeder feeder(table, kMaxTokens, 2);
        ninfer::DeviceBuffer rows(static_cast<std::size_t>(kMaxTokens) * fn::kNgramTokenBytes);

        // Five chunks (both staging buffers reused twice), an EOS inside, widths 1..7.
        const std::vector<std::vector<std::int32_t>> chunks = {{11, 12, 13, 14, 15, 16, 17},
                                                               {18},
                                                               {19, fn::kNgramEosToken, 21},
                                                               {22, 23, 24, 25},
                                                               {26, 27, 28, 29, 30, 31, 32}};
        // Whole chunks as one part, then parts of 3 and of 1 token (the last part shorter).
        failures += run_request(feeder, rows, device.stream, constants, chunks, kMaxTokens,
                                "first request");
        failures += run_request(feeder, rows, device.stream, constants, chunks, 3,
                                "parts of 3 after reset");
        failures += run_request(feeder, rows, device.stream, constants, chunks, 1,
                                "parts of 1 after reset");

        // Proposal chains use independent raw histories, including across EOS, and a repeated
        // proposal or subsequent ordinary decode sees the same committed predecessors.
        {
            const std::array<std::int32_t, 2> lanes{1, 0};
            const std::array<std::int32_t, 4> prefix{17, 29, 31, 41};
            std::array<fn::NgramHistory, 2> committed{};
            for (std::size_t row = 0; row < lanes.size(); ++row) {
                const auto tokens = std::span<const std::int32_t>(prefix).subspan(row * 2, 2);
                feeder.restore_history(lanes[row], tokens);
                for (const auto token : tokens) {
                    static_cast<void>(fn::ngram_row_ids(token, committed[row], constants));
                }
            }
            const std::array<std::int32_t, 8> proposals{43, fn::kNgramEosToken, 47, 53, 59, 61, 67,
                                                        71};
            std::vector<std::uint8_t> proposed(proposals.size() * fn::kNgramTokenBytes);
            for (std::int32_t repeat = 0; repeat < 2; ++repeat) {
                feeder.gather_proposals(lanes, proposals, 4, proposed);
                auto expected = committed;
                for (std::size_t row = 0; row < lanes.size(); ++row) {
                    failures +=
                        check_records(std::span<const std::uint8_t>(proposed).subspan(
                                          row * 4 * fn::kNgramTokenBytes, 4 * fn::kNgramTokenBytes),
                                      std::span<const std::int32_t>(proposals).subspan(row * 4, 4),
                                      expected[row], constants, "proposal history fork");
                }
            }
            const std::array<std::int32_t, 2> next{73, 79};
            std::vector<std::uint8_t> ordinary(next.size() * fn::kNgramTokenBytes);
            feeder.gather_now(lanes, next, ordinary);
            for (std::size_t row = 0; row < lanes.size(); ++row) {
                failures += check_records(std::span<const std::uint8_t>(ordinary).subspan(
                                              row * fn::kNgramTokenBytes, fn::kNgramTokenBytes),
                                          std::span<const std::int32_t>(next).subspan(row, 1),
                                          committed[row], constants,
                                          "proposal preserved committed history");
            }
            const std::array<std::int32_t, 2> counts{2, 0};
            feeder.commit_proposals(lanes, proposals, 4, counts);
            for (std::int32_t column = 0; column < counts[0]; ++column) {
                static_cast<void>(fn::ngram_row_ids(proposals[static_cast<std::size_t>(column)],
                                                    committed[0], constants));
            }
            const std::array<std::int32_t, 2> after_commit{83, 89};
            feeder.gather_now(lanes, after_commit, ordinary);
            for (std::size_t row = 0; row < lanes.size(); ++row) {
                failures += check_records(
                    std::span<const std::uint8_t>(ordinary).subspan(row * fn::kNgramTokenBytes,
                                                                    fn::kNgramTokenBytes),
                    std::span<const std::int32_t>(after_commit).subspan(row, 1), committed[row],
                    constants, "accepted prefix and rejected row histories");
            }
        }

        // Two lanes: lane 0 prefetches its second chunk while lane 1 gathers a decode round; the
        // prefetch is then cancelled (another lane's prefill comes first), lane 1 gathers again,
        // and lane 0 restarts the same chunk from the restored history.
        {
            fn::NgramHistory lane0;
            fn::NgramHistory lane1;
            feeder.reset(0);
            feeder.reset(1);
            feeder.begin(0, chunks[0], kMaxTokens);
            failures +=
                check_records(upload_all(feeder, rows, device.stream,
                                         static_cast<std::int32_t>(chunks[0].size()), kMaxTokens),
                              chunks[0], lane0, constants, "lane 0 chunk 0");
            feeder.begin(0, chunks[3], 2);
            if (!feeder.pending_matches(0, chunks[3]) || feeder.pending_matches(1, chunks[3]) ||
                feeder.pending_matches(0, chunks[2])) {
                std::cerr << "pending_matches misreports the prefetch\n";
                ++failures;
            }
            std::vector<std::uint8_t> round(2 * static_cast<std::size_t>(fn::kNgramTokenBytes));
            const std::vector<std::int32_t> lane1_round{40, 41};
            const std::vector<std::int32_t> one_lane{1, 1};
            bool repeated_rejected = false;
            try {
                feeder.gather_now(one_lane, lane1_round, round);
            } catch (const std::invalid_argument&) { repeated_rejected = true; }
            bool pending_rejected = false;
            try {
                const std::vector<std::int32_t> lanes{1, 0};
                feeder.gather_now(lanes, lane1_round, round);
            } catch (const std::logic_error&) { pending_rejected = true; }
            if (!repeated_rejected || !pending_rejected) {
                std::cerr << "gather_now accepted a repeated lane or the pending lane\n";
                ++failures;
            }
            const std::vector<std::int32_t> lane1_only{1};
            for (const std::int32_t token : lane1_round) {
                std::vector<std::uint8_t> one(fn::kNgramTokenBytes);
                feeder.gather_now(lane1_only, std::span<const std::int32_t>(&token, 1), one);
                failures += check_records(one, std::span<const std::int32_t>(&token, 1), lane1,
                                          constants, "lane 1 decode");
            }
            feeder.cancel();
            if (feeder.pending_lane() != -1) {
                std::cerr << "cancel() left a pending gather\n";
                ++failures;
            }
            feeder.begin(0, chunks[3], 2);
            failures += check_records(upload_all(feeder, rows, device.stream,
                                                 static_cast<std::int32_t>(chunks[3].size()), 2),
                                      chunks[3], lane0, constants, "lane 0 after cancel");
        }

        bool rejected = false;
        try {
            feeder.begin(0, chunks.front(), kMaxTokens);
            feeder.begin(1, chunks.front(), kMaxTokens);
        } catch (const std::logic_error&) { rejected = true; }
        if (!rejected) {
            std::cerr << "a second begin() without upload was accepted\n";
            ++failures;
        }
        feeder.reset(0);
    }
    std::filesystem::remove(path);
    return failures;
}

} // namespace

int main() {
    try {
        const int failures = run();
        if (failures != 0) {
            std::cerr << failures << " failures\n";
            return 1;
        }
        std::cout << "PleRowFeeder: ok\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "PleRowFeeder test failed: " << error.what() << '\n';
        return 1;
    }
}
